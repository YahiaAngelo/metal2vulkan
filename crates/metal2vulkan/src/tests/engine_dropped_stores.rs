//! Stores through a buffer slot bound to memory whose GPU writes Metal drops.
//!
//! Metal takes a shader's stores to such memory nowhere and completes the command buffer with no
//! error: no-copy memory of its private option bit 24, or pages a consumer replaced read-only
//! (`metal2vulkan-runtime`'s `read_only_view.rs`, measured on an M3 Max; the user's own shape, a
//! reachable store gated off at runtime, completed in every one of 8,314 runs). Vulkan has no
//! read-only import of host memory, so the drop is the shader's: a function that stores through a
//! buffer slot reads the draw's or dispatch's `SpecKey::DroppedStoreSlots` word as push data and
//! every store through the slot reads the slot's bit of it (`lower::access::emit::writable`). One
//! pipeline runs with any word.
//!
//! Here the kernel of the owned fixture `engine/fca_kernel_absent_builtin`, whose one store writes
//! `out[tid]`, runs on the CPU interpreter with the word clear, with its slot's bit set, and with
//! another slot's bit set, on every committed profile. Its slot is a descriptor view, where the
//! bit still guards the store with a branch; `tests::engine_dropped_store_sink` runs the same word
//! over a slot reached by its device address, where the bit selects the store's address instead.
//! `llvm-as` assembles the fixture; without it the tests skip visibly. No GPU is used.

use std::collections::BTreeMap;

use crate::device::{DevicePolicy, DeviceProfile};
use crate::library::{FunctionConstantValue, FunctionConstantValues};
use crate::lower::select::Plan;
use crate::metal::DataType;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    engine_dir, llvm_assemble, profile, typed_module_specialized, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;

/// The fixture's entry, whose `has_width` constant is on: every thread stores
/// `threads_per_threadgroup`, [`THREADS`], to `out[tid]`.
const ENTRY: &str = "fca_kernel_absent_builtin";

/// The threads of the one threadgroup the kernel runs in, and so the value it stores.
const THREADS: u32 = 4;

/// The byte every word of the output buffer holds before the dispatch: a store that is dropped
/// leaves them all.
const SENTINEL: u8 = 0xab;

fn assemble() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{ENTRY}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {ENTRY}.ll: {why}"),
    }
}

/// Every word of `out` after one workgroup of [`THREADS`] threads ran `m` (planned as `plan`) with
/// the kernel stage's `SpecKey::DroppedStoreSlots` word `dropped`, the buffer filled with
/// [`SENTINEL`] first; identical in both scheduling orders. The interpreter writes the word into
/// the push bytes of its placed field (`interp::specialized_push`).
fn run_kernel(plan: &Plan, m: &TypedModule, dropped: u32) -> Vec<u32> {
    let mut spec = BTreeMap::new();
    for (axis, v) in [(Axis::X, THREADS), (Axis::Y, 1), (Axis::Z, 1)] {
        spec.insert(SpecKey::LocalSize(axis), u64::from(v));
    }
    spec.insert(SpecKey::FlatLocalSize, u64::from(THREADS));
    spec.insert(
        SpecKey::DroppedStoreSlots(Stage::Kernel),
        u64::from(dropped),
    );
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: [SENTINEL; 4 * THREADS as usize].to_vec(),
        }],
        push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            order,
            ..Config::default()
        };
        let out = run(m, &cfg, res.clone()).unwrap_or_else(|e| panic!("{order:?}: {e:?}"));
        outs.push(out.buffers);
    }
    assert_eq!(outs[0], outs[1], "scheduling orders disagree");
    outs[0][0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

/// The constants of the fixture with `has_width` on, so that the kernel stores
/// `threads_per_threadgroup`.
fn constants() -> FunctionConstantValues {
    FunctionConstantValues::defaults()
        .with_index(0, FunctionConstantValue::new(DataType::Bool, [1]))
}

fn prof(stem: &str) -> DeviceProfile {
    profile(stem, DevicePolicy::default()).expect("committed profile")
}

/// A kernel that stores through a buffer slot reads the dispatch's `DroppedStoreSlots` word from
/// its push window: the word is ordinary pipeline data, so one pipeline runs with any of its
/// values.
#[test]
fn a_storing_kernel_reads_the_dropped_store_word() {
    let Some(bc) = assemble() else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        let _ = prof(&stem);
        let (plan, _) = typed_module_specialized(&bc, ENTRY, &p, &constants())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let field = PushField::Specialized(SpecKey::DroppedStoreSlots(Stage::Kernel));
        let placed = plan
            .push
            .fields
            .iter()
            .find(|f| f.field == field)
            .unwrap_or_else(|| panic!("{stem}: the dropped-store word is not placed"));
        assert!(
            matches!(placed.loc, PushLoc::Push { .. }),
            "{stem}: {placed:?}"
        );
        assert_eq!(placed.size, 4, "{stem}");
    }
}

/// With the slot's bit of the word set, the kernel's store is dropped and the buffer's bytes are
/// the ones it held: Metal's drop, and the dispatch runs to its end. With the word clear, or with
/// another slot's bit set, the same pipeline stores.
#[test]
fn a_dropped_slots_store_leaves_the_bytes_and_the_dispatch_runs() {
    let Some(bc) = assemble() else {
        return;
    };
    let kept = u32::from_le_bytes([SENTINEL; 4]);
    for (stem, p) in committed_profiles().expect("profiles") {
        let (plan, m) = typed_module_specialized(&bc, ENTRY, &p, &constants())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        // No slot's writes dropped: the store lands.
        assert_eq!(
            run_kernel(&plan, &m, 0),
            [THREADS; THREADS as usize],
            "{stem}: the store did not land with the word clear"
        );
        // Slot 0's dropped: the store is taken nowhere and every word is as it was.
        assert_eq!(
            run_kernel(&plan, &m, 1),
            [kept; THREADS as usize],
            "{stem}: the store reached memory whose writes are dropped"
        );
        // Another slot's bit is another slot's: the store through slot 0 lands.
        assert_eq!(
            run_kernel(&plan, &m, 0b10),
            [THREADS; THREADS as usize],
            "{stem}: slot 1's bit dropped slot 0's store"
        );
        // Every bit of the word, slot 0's among them: still dropped, and nothing else changes.
        assert_eq!(
            run_kernel(&plan, &m, u32::MAX),
            [kept; THREADS as usize],
            "{stem}: a full word let the store through"
        );
    }
}
