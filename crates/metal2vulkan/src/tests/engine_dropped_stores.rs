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
//! bit selects the store's place between the slot's and the stage's write-only sink
//! (`SyntheticBinding::StoreSink`, `lower::access::sink`), so the module branches for no store:
//! with the bit set the slot keeps its bytes and the sink holds the word, which is the drop
//! proved as a redirect and not as a store that went missing.
//! `tests::engine_dropped_store_sink` runs the same word over a slot reached by its device
//! address, where the bit joins the bounds check in the one address selection instead.
//!
//! A profile whose device cannot select between two descriptors' pointers
//! (`variablePointersStorageBuffer`: SwiftShader's and the Vulkan 1.2 floor among the committed
//! ones) binds no sink and keeps the branch, which moves no stored value there; the branch's
//! shape is asserted per profile, so neither form can quietly become the other.
//!
//! `llvm-as` assembles the fixture; without it the tests skip visibly. No GPU is used.

use std::collections::BTreeMap;

use crate::device::{DevicePolicy, DeviceProfile};
use crate::library::{FunctionConstantValue, FunctionConstantValues};
use crate::lower::select::Plan;
use crate::metal::DataType;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile_specialized, engine_dir, llvm_assemble, profile, typed_module_specialized,
    AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::ir::{Inst, Terminator, TypedModule};
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

/// Bytes of the stage's sink, which holds a whole leaf whatever the leaf is
/// (`bind::NeutralResource::SINK_BYTES`).
const SINK_BYTES: usize = 64;

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

/// Every word of `out`, and of the stage's sink where it binds one, after one workgroup of
/// [`THREADS`] threads ran `m` (planned as `plan`) with the kernel stage's
/// `SpecKey::DroppedStoreSlots` word `dropped`, both filled with [`SENTINEL`] first; identical in
/// both scheduling orders. The interpreter writes the word into the push bytes of its placed field
/// (`interp::specialized_push`).
fn run_kernel(plan: &Plan, m: &TypedModule, dropped: u32) -> (Vec<u32>, Option<Vec<u32>>) {
    let mut spec = BTreeMap::new();
    for (axis, v) in [(Axis::X, THREADS), (Axis::Y, 1), (Axis::Z, 1)] {
        spec.insert(SpecKey::LocalSize(axis), u64::from(v));
    }
    spec.insert(SpecKey::FlatLocalSize, u64::from(THREADS));
    spec.insert(
        SpecKey::DroppedStoreSlots(Stage::Kernel),
        u64::from(dropped),
    );
    let sink = plan.families.access.store_sink.is_some();
    let mut buffers = vec![Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index: 0,
        }),
        address: None,
        bytes: [SENTINEL; 4 * THREADS as usize].to_vec(),
    }];
    if sink {
        buffers.push(Buffer {
            slot: Some(crate::lower::access::sink::symbol(Stage::Kernel)),
            address: None,
            bytes: vec![SENTINEL; SINK_BYTES],
        });
    }
    let res = Resources {
        buffers,
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
    let out = words(&outs[0][0]);
    let sink = sink.then(|| words(&outs[0][1]));
    (out, sink)
}

/// The module's conditional branches and selections.
fn shape(m: &TypedModule) -> (usize, usize) {
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
    (branches, selects)
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
/// the ones it held: Metal's drop, and the dispatch runs to its end. Where the stage binds its
/// sink, the sink then holds the word the store carried, so the store ran and was redirected
/// rather than going missing. With the word clear, or with another slot's bit set, the same
/// pipeline stores through the slot and leaves the sink alone.
#[test]
fn a_dropped_slots_store_leaves_the_bytes_and_the_dispatch_runs() {
    let Some(bc) = assemble() else {
        return;
    };
    let kept = u32::from_le_bytes([SENTINEL; 4]);
    for (stem, p) in committed_profiles().expect("profiles") {
        let (plan, m) = typed_module_specialized(&bc, ENTRY, &p, &constants())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let untouched = plan
            .families
            .access
            .store_sink
            .is_some()
            .then(|| vec![kept; SINK_BYTES / 4]);
        // No slot's writes dropped: the store lands and no sink is written.
        let (out, sink) = run_kernel(&plan, &m, 0);
        assert_eq!(
            out, [THREADS; THREADS as usize],
            "{stem}: the store did not land with the word clear"
        );
        assert_eq!(
            sink, untouched,
            "{stem}: the sink was written with the word clear"
        );
        // Slot 0's dropped: every word of the slot is as it was, and the sink holds the word the
        // store carried.
        let (out, sink) = run_kernel(&plan, &m, 1);
        assert_eq!(
            out, [kept; THREADS as usize],
            "{stem}: the store reached memory whose writes are dropped"
        );
        if let Some(sink) = &sink {
            assert_eq!(
                sink[0], THREADS,
                "{stem}: the dropped store reached no sink"
            );
        }
        // Another slot's bit is another slot's: the store through slot 0 lands.
        let (out, sink) = run_kernel(&plan, &m, 0b10);
        assert_eq!(
            out, [THREADS; THREADS as usize],
            "{stem}: slot 1's bit dropped slot 0's store"
        );
        assert_eq!(
            sink, untouched,
            "{stem}: slot 1's bit sent slot 0's store to the sink"
        );
        // Every bit of the word, slot 0's among them: still dropped, and nothing else changes.
        let (out, sink) = run_kernel(&plan, &m, u32::MAX);
        assert_eq!(
            out, [kept; THREADS as usize],
            "{stem}: a full word let the store through"
        );
        if let Some(sink) = &sink {
            assert_eq!(
                sink[0], THREADS,
                "{stem}: the dropped store reached no sink"
            );
        }
    }
}

/// The guard's shape, per profile: a device that can select between two descriptors' pointers
/// binds the stage's sink (`variablePointers`), and then the kernel's one store runs with no
/// conditional branch at all and at least the one selection of its place. A device that cannot
/// binds no sink and keeps the branch.
///
/// This is what separates the fix from a guard that never fires: the words above would also read
/// as Metal's if the condition were dropped, and would also read as Metal's if the store sat in a
/// branch. One profile of each kind is asserted to exist, so a profile set that lost either kind
/// fails here rather than passing vacuously.
#[test]
fn a_dropped_view_store_selects_its_place_where_the_device_selects_pointers() {
    let Some(bc) = assemble() else {
        return;
    };
    let mut selecting = 0;
    let mut branching = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let (plan, m) = typed_module_specialized(&bc, ENTRY, &p, &constants())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let (branches, selects) = shape(&m);
        let pointers = crate::lower::access::sink::selects_pointers(&p);
        assert_eq!(
            plan.families.access.store_sink.is_some(),
            pointers,
            "{stem}: the sink does not follow variablePointersStorageBuffer"
        );
        if pointers {
            selecting += 1;
            assert_eq!(
                branches, 0,
                "{stem}: the module branches, so a guarded store moved a value into its block"
            );
            assert!(
                selects >= 1,
                "{stem}: {selects} selections, too few for the store's place"
            );
            // The sink is one storage descriptor of the stage, whatever unit views it holds.
            let a = compile_specialized(&bc, ENTRY, &p, &constants())
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            let sinks = a
                .interface
                .descriptors
                .iter()
                .filter(|d| {
                    d.slot.kind
                        == ResourceKind::Synthetic(
                            crate::pipeline::sym::SyntheticBinding::StoreSink,
                        )
                        && d.ty == crate::vk::DescriptorType::STORAGE_BUFFER
                })
                .count();
            assert_eq!(sinks, 1, "{stem}: {sinks} sink descriptors");
        } else {
            branching += 1;
            assert!(
                branches >= 1,
                "{stem}: no branch, but the device cannot select between two descriptors\' \
                 pointers"
            );
        }
    }
    assert!(
        selecting > 0 && branching > 0,
        "{selecting} selecting and {branching} branching profiles"
    );
}

/// The selecting form reaches Metal. `engine/trn_math_fast`, the fixture the branching guard's
/// regression was found on, stores its answers through a buffer slot's descriptor view; on a
/// MoltenVK capture its module holds the selection of the store's place between two descriptors
/// (`OpSelect` of two `StorageBuffer` pointers, which declares `VariablePointers`), and MoltenVK's
/// own path -- SPIRV-Cross to MSL, then `xcrun metal` -- compiles it.
///
/// The selection is counted first, so the compile cannot pass by compiling a module without one.
/// `llvm-as` or the Metal toolchain missing skips visibly.
#[test]
fn the_selected_place_compiles_through_moltenvks_own_path() {
    use crate::spirv::ir::{StorageClass, Type};
    let path = engine_dir().join("trn_math_fast.ll");
    let bc = match llvm_assemble(&path) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected trn_math_fast.ll: {why}"),
    };
    let entry = "trn_math_fast";
    let mut compiled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::spirv::testing::is_moltenvk_capture(&stem) {
            continue;
        }
        let (plan, m) = crate::pipeline::testing::typed_module(&bc, entry, &p)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(
            plan.families.access.store_sink.is_some(),
            "{stem}: no store sink"
        );
        let pointer_selects = m
            .blocks()
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|n| match n.inst {
                crate::spirv::ir::Inst::Select { t, .. } => matches!(
                    m.types().get(m.values()[t.index()].ty),
                    Ok(Type::Pointer {
                        class: StorageClass::StorageBuffer,
                        ..
                    })
                ),
                _ => false,
            })
            .count();
        assert!(
            pointer_selects > 0,
            "{stem}: the module selects no store place, so the compile below proves nothing"
        );
        let a = crate::pipeline::testing::compile(&bc, entry, &p)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let (_, words) = crate::pipeline::testing::link_and_validate(&a, &p)
            .unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        match crate::pipeline::testing::moltenvk_msl_compile(&words) {
            Ok(()) => compiled += 1,
            Err(AssembleError::Missing(why)) => eprintln!("SKIPPED MSL on {stem}: {why}"),
            Err(AssembleError::Rejected(why)) => panic!("{stem}: MSL: {why}"),
        }
    }
    eprintln!("MoltenVK captures whose selecting module compiled to Metal: {compiled}");
}
