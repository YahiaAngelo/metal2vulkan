//! Barriers and fences: `air.wg.barrier`, `air.simdgroup.barrier`, `air.atomic.fence`.
//!
//! A barrier is `void (i32 mem_flags, i32 scope)` in AIR 2.8 and `void (i32 mem_flags, i32 order,
//! i32 scope)` in AIR 2.9 (deployment target macOS 27); a fence is `void (i32 mem_flags, i32 order,
//! i32 scope)`; every operand is an immediate. `mem_flags` is Metal's `mem_flags` mask
//! (`mem_device` 1, `mem_threadgroup` 2, `mem_texture` 4, `mem_threadgroup_imageblock` 8,
//! `mem_object_data` 16). A barrier's order is the `memory_order` MSL 4.1's barriers take (MSL
//! §6.10.1); Metal's headers pass `memory_order_seq_cst` to the barrier builtin in every earlier
//! language version, and AIR 2.8 drops the operand, so the 2-operand form is a `seq_cst` barrier.
//! Which form a declaration has is read from its type.
//!
//! In both forms the name is the **execution** scope and the scope operand the **memory** scope
//! alone (MSL 4.1's `thread_scope` argument: the threads that observe the ordered accesses, MSL
//! §6.10.1): `threadgroup_barrier` (`air.wg.barrier`) synchronizes the threadgroup and
//! `simdgroup_barrier` (`air.simdgroup.barrier`) the simdgroup, whatever scope they are given.
//! The operand does not follow the name: Apple's compiler writes MSL 3.0 and 3.1's
//! `simdgroup_barrier` as `air.simdgroup.barrier` with scope 1 (threadgroup) in AIR 2.6 through 2.9,
//! and MSL 3.2's with scope 4 (simdgroup); `threadgroup_barrier(..., thread_scope_simdgroup)` is
//! `air.wg.barrier` with scope 4. Measured on an M1 Pro (macOS 27): an AIR 2.8
//! `air.simdgroup.barrier(mem_threadgroup, 1)` does not wait for the threadgroup's other
//! simdgroups, where `air.wg.barrier` does. A value outside the modeled enumerations leaves the
//! call unknown.

use super::atomic::{MemoryOrder, ThreadScope};
use super::sig::{ImmKind, Immediates, Operand, Signature};
use crate::error::IntrinsicError;

/// Metal's `mem_flags`: which memories a barrier, fence, or AIR 2.9 atomic orders.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemFlags {
    /// `mem_device` (1): device memory.
    pub device: bool,
    /// `mem_threadgroup` (2): threadgroup memory.
    pub threadgroup: bool,
    /// `mem_texture` (4): texture memory.
    pub texture: bool,
    /// `mem_threadgroup_imageblock` (8): threadgroup imageblock memory.
    pub threadgroup_imageblock: bool,
    /// `mem_object_data`: an object function's payload (MSL §6.10.1, §6.16.3). No stage this
    /// translation compiles has that memory; `mem_flags`' every-flag value (31), which MSL 4.1's
    /// non-explicit atomics pass, includes it.
    pub object_data: bool,
}

impl MemFlags {
    /// The flags of mask `v`; `None` when a bit outside the five is set.
    pub fn from_raw(v: u64) -> Option<MemFlags> {
        (v & !0x1f == 0).then_some(MemFlags {
            device: v & 1 != 0,
            threadgroup: v & 2 != 0,
            texture: v & 4 != 0,
            threadgroup_imageblock: v & 8 != 0,
            object_data: v & 16 != 0,
        })
    }
}

/// Which intrinsic a barrier was spelled with: its execution scope in either form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BarrierName {
    /// `air.wg.barrier`.
    Threadgroup,
    /// `air.simdgroup.barrier`.
    Simdgroup,
}

impl BarrierName {
    /// The execution scope the name spells: the threads the barrier waits for.
    fn scope(self) -> ThreadScope {
        match self {
            BarrierName::Threadgroup => ThreadScope::Threadgroup,
            BarrierName::Simdgroup => ThreadScope::Simdgroup,
        }
    }
}

/// A decoded barrier or fence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Sync {
    /// An execution barrier over `scope` (threadgroup or simdgroup) that orders `flags`' memories
    /// among the threads of `memory`.
    Barrier {
        /// The intrinsic the barrier was spelled with.
        name: BarrierName,
        /// The `mem_flags` immediate: the memories the barrier orders.
        flags: MemFlags,
        /// The AIR 2.9 form's `memory_order` immediate; `None` for the AIR 2.8 form, which has
        /// no such operand and orders as `seq_cst` ([`Sync::barrier_order`]).
        order: Option<MemoryOrder>,
        /// The execution scope: the name's, in either form.
        scope: ThreadScope,
        /// The memory scope: the operand, in either form.
        memory: ThreadScope,
    },
    /// A memory fence (`atomic_thread_fence`).
    Fence {
        /// The `mem_flags` immediate: the memories the fence orders.
        flags: MemFlags,
        /// The `memory_order` immediate.
        order: MemoryOrder,
        /// The `thread_scope` immediate: the threads that observe the ordering.
        scope: ThreadScope,
    },
}

impl Sync {
    pub(super) fn parse(rest: &str) -> Option<Sync> {
        let placeholder = (MemFlags::default(), ThreadScope::Threadgroup);
        let name = match rest {
            "wg.barrier" => BarrierName::Threadgroup,
            "simdgroup.barrier" => BarrierName::Simdgroup,
            "atomic.fence" => {
                return Some(Sync::Fence {
                    flags: placeholder.0,
                    order: MemoryOrder::Relaxed,
                    scope: placeholder.1,
                })
            }
            _ => return None,
        };
        Some(Sync::Barrier {
            name,
            flags: placeholder.0,
            order: None,
            scope: placeholder.1,
            memory: placeholder.1,
        })
    }

    /// The AIR 2.9 form of an AIR 2.8 barrier, the other declaration its name may have.
    pub(super) fn alternative(self) -> Option<Sync> {
        match self {
            Sync::Barrier {
                name,
                flags,
                order: None,
                scope,
                memory,
            } => Some(Sync::Barrier {
                name,
                flags,
                order: Some(MemoryOrder::SeqCst),
                scope,
                memory,
            }),
            _ => None,
        }
    }

    /// The order of a barrier: its `memory_order` operand, or `seq_cst` for the AIR 2.8 form
    /// (Metal's headers pass `memory_order_seq_cst` there). `None` for a fence.
    pub fn barrier_order(&self) -> Option<MemoryOrder> {
        match self {
            Sync::Barrier { order, .. } => Some(order.unwrap_or(MemoryOrder::SeqCst)),
            Sync::Fence { .. } => None,
        }
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        let i32s = Operand::Imm(ImmKind::I32);
        match self {
            Sync::Barrier { order: None, .. } => Signature::new(Operand::Void, &[i32s, i32s]),
            Sync::Barrier { order: Some(_), .. } | Sync::Fence { .. } => {
                Signature::new(Operand::Void, &[i32s, i32s, i32s])
            }
        }
    }

    /// Read the flags, order, and scopes. `Ok(None)`: a value outside the enumerations, or an AIR
    /// 2.8 barrier's scope other than threadgroup or simdgroup.
    pub(super) fn bind(self, imm: &Immediates<'_>) -> Result<Option<Sync>, IntrinsicError> {
        let Some(flags) = MemFlags::from_raw(imm.get(0)?) else {
            return Ok(None);
        };
        Ok(match self {
            // The AIR 2.8 form: the operand is the memory scope, the execution scope the name's.
            Sync::Barrier {
                name, order: None, ..
            } => match ThreadScope::from_raw(imm.get(1)?) {
                Some(memory @ (ThreadScope::Threadgroup | ThreadScope::Simdgroup)) => {
                    Some(Sync::Barrier {
                        name,
                        flags,
                        order: None,
                        scope: name.scope(),
                        memory,
                    })
                }
                _ => None,
            },
            // The AIR 2.9 form: the order sits between the flags and the memory scope, and the
            // execution scope is the name's.
            Sync::Barrier {
                name,
                order: Some(_),
                ..
            } => match (
                MemoryOrder::from_raw(imm.get(1)?),
                ThreadScope::from_raw(imm.get(2)?),
            ) {
                (Some(order), Some(memory)) => Some(Sync::Barrier {
                    name,
                    flags,
                    order: Some(order),
                    scope: name.scope(),
                    memory,
                }),
                _ => None,
            },
            Sync::Fence { .. } => {
                let (order, scope) = (imm.get(1)?, imm.get(2)?);
                match (MemoryOrder::from_raw(order), ThreadScope::from_raw(scope)) {
                    (Some(order), Some(scope)) => Some(Sync::Fence {
                        flags,
                        order,
                        scope,
                    }),
                    _ => None,
                }
            }
        })
    }
}
