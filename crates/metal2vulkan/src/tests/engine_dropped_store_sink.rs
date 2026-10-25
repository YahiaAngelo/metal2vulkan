//! A dropped store through a device-address slot lands in the sink, and nothing branches on it.
//!
//! A store through a buffer slot bound to memory whose GPU writes Metal drops is dropped on the
//! slot's bit of the draw's or dispatch's `SpecKey::DroppedStoreSlots` word
//! (`lower::access::emit::writable`). Where the slot is reached by device address
//! (`BufferForm::Psb`), the bit joins the bounds check in the one address selection
//! (`lower::access::psb::prepare`): the store runs unconditionally at
//! `select(in_bounds && keeps, own_address, sink)`, so the module holds no branch for it.
//!
//! That shape is the point. A branch around a store also moves every value only that store
//! consumes into the branch's block, and Apple's backend expands `air.fast_asinh.f32` and
//! `air.fast_acosh.f32` to different last bits depending on the basic block their call sits in, so
//! a branching guard changes what Metal's fast-math library answers.
//!
//! The kernel here stores one word through `[[buffer(0)]]`, lowered to a device address, and runs
//! on the CPU interpreter with the word clear, with the slot's bit set, and with another slot's
//! bit set. With the bit set the slot's bytes are the ones it held and the sink holds the word, so
//! the drop is a redirect and not a store that went missing; the module's instructions hold no
//! `OpSelectionMerge`, so no store branches. `llvm-as` assembles the kernel; without it the tests
//! skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::DevicePolicy;
use crate::lower::select::{LowerInputs, Plan};
use crate::pipeline::interface::{BufferForm, PushLoc, SlotInterface};
use crate::pipeline::sym::{
    Axis, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage,
};
use crate::pipeline::testing::{
    compile_with, llvm_assemble, profile, typed_module_with, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::ir::{Inst, Terminator, TypedModule};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// The kernel's name.
const ENTRY: &str = "drop_sink";

/// The threads of the one threadgroup, each storing its own word.
const THREADS: u32 = 4;

/// The word every thread stores.
const STORED: u32 = 0x5157_1a2b;

/// The byte every word of the slot's buffer and of the sink holds before the dispatch.
const SENTINEL: u8 = 0xab;

/// Device addresses of the run's allocations, 64 KiB apart: the slot's buffer, the root table and
/// the sink.
const SLOT_HOST: u64 = 0x1_0000;
const TABLE: u64 = 0x2_0000;
const SINK: u64 = 0x3_0000;
/// The slot's Metal address, which only the resident-set search reads.
const SLOT_GUEST: u64 = 0x10_0000;

/// Bytes of the sink, which holds a whole leaf whatever the leaf is
/// (`bind::NeutralResource::SINK_BYTES`).
const SINK_BYTES: usize = 64;

/// Profile A (device addresses) in both containment encodings: robust descriptors do not reach a
/// device address, so a PSB access is contained in the shader either way.
const PROFILES: [&str; 2] = ["moltenvk-m3", "lavapipe"];

fn prof(stem: &str) -> DeviceProfile {
    profile(stem, DevicePolicy::default()).expect("committed profile")
}

/// `[[buffer(index)]]` of the kernel stage.
fn slot(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    }
}

/// Lowering inputs that put `[[buffer(0)]]` in the device-address form.
fn addressed() -> LowerInputs {
    let mut inputs = LowerInputs::new();
    inputs.psb_slots.insert(slot(0));
    inputs
}

/// The kernel: thread `t` stores [`STORED`] to `b0[t]`.
fn source() -> String {
    format!(
        "{HEADER}define void @{ENTRY}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %b0, i32 %t) {{\n\
         entry:\n  \
           %o = mul i32 %t, 4\n  \
           %p = getelementptr i8, ptr addrspace(1) %b0, i32 %o\n  \
           store i32 {STORED}, ptr addrspace(1) %p, align 4\n  \
           ret void\n\
         }}\n\n\
         !air.kernel = !{{!0}}\n!air.version = !{{!90}}\n!air.language_version = !{{!91}}\n\
         !0 = !{{ptr @{ENTRY}, !1, !2}}\n!1 = !{{}}\n!2 = !{{!10, !11}}\n\
         !10 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", \
         !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, \
         !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"b0\"}}\n\
         !11 = !{{i32 1, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", \
         !\"air.arg_name\", !\"t\"}}\n\
         !90 = !{{i32 2, i32 8, i32 0}}\n!91 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

/// Assemble the kernel, or `None` where `llvm-as` is missing.
fn assemble() -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-dropped-store-sink-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, source()).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {ENTRY}: {why}"),
    }
}

/// The root table: the null root, the sink at root 1, and `[[buffer(0)]]` at root 2.
fn root_table(slot_bytes: usize) -> Vec<u8> {
    let mut t = vec![0u8; 32 * 3];
    let mut put = |root: usize, field: usize, v: u64| {
        let at = 32 * root + 8 * field;
        t[at..at + 8].copy_from_slice(&v.to_le_bytes());
    };
    put(1, 0, SINK);
    // The resident set is the slot alone, at root 2.
    put(1, 1, 2);
    put(1, 2, 1);
    put(2, 0, SLOT_HOST);
    put(2, 1, SLOT_GUEST);
    put(2, 2, slot_bytes as u64);
    t
}

/// The slot's words and the sink's words after one threadgroup of [`THREADS`] threads ran `m`
/// (planned as `plan`) with the kernel stage's `SpecKey::DroppedStoreSlots` word `dropped`, both
/// filled with [`SENTINEL`] first; identical in both scheduling orders.
fn run_kernel(plan: &Plan, m: &TypedModule, dropped: u32) -> (Vec<u32>, Vec<u32>) {
    let mut spec = BTreeMap::new();
    for (axis, v) in [(Axis::X, THREADS), (Axis::Y, 1), (Axis::Z, 1)] {
        spec.insert(SpecKey::LocalSize(axis), u64::from(v));
    }
    spec.insert(SpecKey::FlatLocalSize, u64::from(THREADS));
    spec.insert(
        SpecKey::DroppedStoreSlots(Stage::Kernel),
        u64::from(dropped),
    );
    let slot_bytes = vec![SENTINEL; 4 * THREADS as usize];
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        let PushLoc::Push { offset } = f.loc else {
            panic!("{:?} spilled to the ArgumentTable", f.field);
        };
        let at = offset as usize;
        match f.field {
            PushField::RootTableAddress => push[at..at + 8].copy_from_slice(&TABLE.to_le_bytes()),
            // The interpreter writes a specialized field's bytes itself.
            PushField::Specialized(_) => {}
            PushField::DispatchRegion => push[at..at + f.size as usize].fill(0),
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(ResourceSlot {
                    stage: Stage::Kernel,
                    kind: ResourceKind::Buffer,
                    index: 0,
                }),
                address: Some(SLOT_HOST),
                bytes: slot_bytes.clone(),
            },
            Buffer {
                slot: None,
                address: Some(TABLE),
                bytes: root_table(slot_bytes.len()),
            },
            Buffer {
                slot: None,
                address: Some(SINK),
                bytes: vec![SENTINEL; SINK_BYTES],
            },
        ],
        push,
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
    (words(&outs[0][0]), words(&outs[0][2]))
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The planned module of `stem`, with `[[buffer(0)]]` in the device-address form.
fn planned(bc: &[u8], stem: &str) -> (Plan, TypedModule) {
    let p = prof(stem);
    let (plan, m) =
        typed_module_with(bc, ENTRY, &p, &addressed()).unwrap_or_else(|e| panic!("{stem}: {e}"));
    (plan, m)
}

/// A store through a device-address slot reads the dispatch's `DroppedStoreSlots` word and selects
/// the sink's address with it: the uniform word and the bit test are there, and no store branches.
#[test]
fn a_dropped_device_address_store_selects_the_sink_without_a_branch() {
    let Some(bc) = assemble() else {
        return;
    };
    for stem in PROFILES {
        let p = prof(stem);
        let a =
            compile_with(&bc, ENTRY, &p, &addressed()).unwrap_or_else(|e| panic!("{stem}: {e}"));
        match a.interface.slots.get(&slot(0)) {
            Some(SlotInterface::Buffer(b)) => {
                assert_eq!(b.form, BufferForm::Psb, "{stem}: the slot is not addressed")
            }
            other => panic!("{stem}: {other:?}"),
        }
        let (plan, m) = planned(&bc, stem);
        let field = PushField::Specialized(SpecKey::DroppedStoreSlots(Stage::Kernel));
        assert!(
            plan.push.fields.iter().any(|f| f.field == field),
            "{stem}: the dropped-store word is not placed, so no store reads it"
        );
        let mut branches = 0;
        let mut selects = 0;
        for b in m.blocks() {
            if matches!(b.term, Terminator::CondBranch { .. }) {
                branches += 1;
            }
            selects += b
                .insts
                .iter()
                .filter(|n| matches!(n.inst, Inst::Select { .. }))
                .count();
        }
        assert_eq!(
            branches, 0,
            "{stem}: the module branches, so a guarded store moved a value into its block"
        );
        assert!(
            selects >= 2,
            "{stem}: {selects} selections, too few for the address and the offset"
        );
    }
}

/// With the slot's bit of the word set, the store lands in the sink and the slot's bytes are the
/// ones it held: Metal's drop, and the dispatch runs to its end. With the word clear, or with
/// another slot's bit set, the same pipeline stores through the slot and leaves the sink alone.
#[test]
fn a_dropped_slots_store_lands_in_the_sink() {
    let Some(bc) = assemble() else {
        return;
    };
    let kept = u32::from_le_bytes([SENTINEL; 4]);
    for stem in PROFILES {
        let (plan, m) = planned(&bc, stem);
        // No slot's writes dropped: the store lands and the sink is untouched.
        let (out, sink) = run_kernel(&plan, &m, 0);
        assert_eq!(
            out, [STORED; THREADS as usize],
            "{stem}: the store did not land with the word clear"
        );
        assert_eq!(
            sink,
            [kept; SINK_BYTES / 4],
            "{stem}: the sink was written with the word clear"
        );
        // Slot 0's writes dropped: every word of the slot is as it was, and the sink holds the
        // word the store carried, so the store ran and was redirected.
        let (out, sink) = run_kernel(&plan, &m, 1);
        assert_eq!(
            out, [kept; THREADS as usize],
            "{stem}: the store reached memory whose writes are dropped"
        );
        assert_eq!(sink[0], STORED, "{stem}: the dropped store reached no sink");
        // Another slot's bit is another slot's.
        let (out, sink) = run_kernel(&plan, &m, 0b10);
        assert_eq!(
            out, [STORED; THREADS as usize],
            "{stem}: slot 1's bit dropped slot 0's store"
        );
        assert_eq!(
            sink,
            [kept; SINK_BYTES / 4],
            "{stem}: slot 1's bit sent slot 0's store to the sink"
        );
        // Every bit of the word, slot 0's among them.
        let (out, sink) = run_kernel(&plan, &m, u32::MAX);
        assert_eq!(
            out, [kept; THREADS as usize],
            "{stem}: a full word let the store through"
        );
        assert_eq!(sink[0], STORED, "{stem}: the dropped store reached no sink");
    }
}
