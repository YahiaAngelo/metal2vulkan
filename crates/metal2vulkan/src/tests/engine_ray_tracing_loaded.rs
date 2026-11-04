//! An acceleration structure loaded from an argument buffer.
//!
//! An argument buffer's `instance_acceleration_structure` or `primitive_acceleration_structure`
//! member holds the structure's 8-byte `gpuResourceID`. The load searches the stage's
//! acceleration-structure directory for it (`access.handle.structure_lookup`), and the intersect
//! call traverses the structure at the device address the directory's entry holds
//! (`OpConvertUToAccelerationStructureKHR`), never at the null entry's address 0. The module
//! declares no acceleration-structure descriptor, and the binder writes the directory from the
//! structures the consumer lists (`BindState::argument_acceleration_structures`).
//!
//! The shapes that stay refused: bytes of a plain buffer read as a structure (no member names
//! them), a member whose level is not the intersector's, a merge of loaded structures, any reader
//! but the intersect call, and a loaded structure's instance count.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use crate::bind::{
    AccelerationStructuresShape, ArgumentAccelerationStructure, BindState, BoundBuffer, StageTable,
    WriteSource,
};
use crate::budget::Budget;
use crate::device::requirements::VkFeature;
use crate::device::DeviceProfile;
use crate::error::{BindError, Capability, Error, Feature, Needs, Unsupported};
use crate::library::{FunctionArtifact, Library};
use crate::memory::handles::TABLE_ENTRY_BYTES;
use crate::pipeline::layout::BindingSource;
use crate::pipeline::sym::{ResourceKind, Stage, SyntheticBinding};
use crate::pipeline::testing::{compile, link_and_validate};
use crate::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, PrecisionNote, StageSource};
use crate::spirv::testing::{committed_profiles, spirv_dis};
use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};

use super::engine_argument_tables::{bytes_at, device, one_group};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// The features a ray query translation needs.
const RAY_FEATURES: [VkFeature; 3] = [
    VkFeature::RayQuery,
    VkFeature::AccelerationStructure,
    VkFeature::BufferDeviceAddress,
];

/// The single-level instanced intersector's result.
const INSTANCED: &str = "{ i32, float, i32, i32, ptr addrspace(1), i32, i32, <2 x float>, i1 }";
/// The primitive intersector's result.
const PRIMITIVE: &str = "{ i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 }";

/// Which intersector a kernel calls.
#[derive(Clone, Copy)]
enum Intersector {
    Instanced,
    Primitive,
}

impl Intersector {
    fn result(self) -> &'static str {
        match self {
            Intersector::Instanced => INSTANCED,
            Intersector::Primitive => PRIMITIVE,
        }
    }

    fn callee(self) -> &'static str {
        match self {
            Intersector::Instanced => "air.intersect.instancing.triangle_data",
            Intersector::Primitive => "air.intersect.triangle_data",
        }
    }

    /// The operands from the structure on, `%as` the structure.
    fn operands(self) -> &'static str {
        match self {
            Intersector::Instanced => {
                "ptr addrspace(1) %as, i32 255, ptr addrspace(1) %t, ptr null, i64 0, i32 0, \
                 i32 0, i32 0, i32 0, i32 0, i32 3, i32 -1, i32 -1, i32 0, i1 false, i1 false"
            }
            Intersector::Primitive => {
                "ptr addrspace(1) %as, ptr addrspace(1) %t, ptr null, i64 0, i32 0, i32 0, i32 0, \
                 i32 0, i32 0, i32 3, i32 -1, i32 -1, i32 0, i1 false"
            }
        }
    }

    fn parameters(self) -> &'static str {
        match self {
            Intersector::Instanced => {
                "ptr addrspace(1), i32, ptr addrspace(1), ptr, i64, i32, i32, i32, i32, i32, \
                 i32, i32, i32, i32, i1, i1"
            }
            Intersector::Primitive => {
                "ptr addrspace(1), ptr addrspace(1), ptr, i64, i32, i32, i32, i32, i32, i32, \
                 i32, i32, i32, i1"
            }
        }
    }
}

/// The scene parameter's metadata: an argument buffer in address space `space` of two
/// acceleration-structure members of `level` (`instance` or `primitive`) at offsets 0 and 8, ids
/// 0 and 1; or, for `level` `None`, a plain buffer of `ulong`.
fn scene(space: u32, level: Option<&str>) -> String {
    let Some(level) = level else {
        return format!(
            "!3 = !{{i32 0, !\"air.buffer\", !\"air.buffer_size\", i32 16, \
             !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 \
             {space}, !\"air.arg_type_name\", !\"ulong\", !\"air.arg_name\", !\"scene\"}}\n"
        );
    };
    let member = |k: u32, name: &str| {
        format!(
            "!{} = !{{i32 {k}, !\"air.{level}_acceleration_structure\", !\"air.location_index\", \
             i32 {k}, i32 1, !\"air.read\", !\"air.arg_type_name\", \
             !\"acceleration_structure<>\", !\"air.arg_name\", !\"{name}\"}}\n",
            10 + k
        )
    };
    format!(
        "!3 = !{{i32 0, !\"air.indirect_buffer\", !\"air.buffer_size\", i32 16, \
         !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 \
         {space}, !\"air.struct_type_info\", !9, !\"air.arg_type_size\", i32 16, \
         !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"Scene\", \
         !\"air.arg_name\", !\"scene\"}}\n!9 = !{{i32 0, i32 8, i32 0, \
         !\"acceleration_structure<>\", !\"a\", !\"air.indirect_argument\", !10, i32 8, i32 8, \
         i32 0, !\"acceleration_structure<>\", !\"b\", !\"air.indirect_argument\", !11}}\n{}{}",
        member(0, "a"),
        member(1, "b"),
    )
}

/// `k(Scene scene [[buffer(0)]], device uint *out [[buffer(1)]], uint tid)`: `pick` defines `%as`
/// from `%scene` (in address space `space`), one ray through it with the null table by
/// `intersector`, its type stored to `out[0]`, then `extra`.
fn kernel(
    space: u32,
    level: Option<&str>,
    intersector: Intersector,
    pick: &str,
    extra: &str,
) -> String {
    let (result, callee, operands, parameters) = (
        intersector.result(),
        intersector.callee(),
        intersector.operands(),
        intersector.parameters(),
    );
    let scene = scene(space, level);
    format!(
        "{HEADER}define void @k(ptr addrspace({space}) %scene, ptr addrspace(1) %out, i32 %tid) \
         {{\nentry:\n{pick}  %t = call ptr addrspace(1) \
         @air.get_null_intersection_function_table()\n  %r = call {result} @{callee}(<3 x \
         float> <float 0.0, float 0.0, float -1.0>, <3 x float> <float 0.0, float 0.0, float \
         1.0>, float 0.0, float 100.0, {operands})\n  %ty = extractvalue {result} %r, 0\n  store \
         i32 %ty, ptr addrspace(1) %out, align 4\n{extra}  ret void\n}}\n\ndeclare ptr \
         addrspace(1) @air.get_null_intersection_function_table()\ndeclare {result} \
         @{callee}(<3 x float>, <3 x float>, float, float, {parameters})\ndeclare i32 \
         @air.get_instance_count_instance_acceleration_structure(ptr addrspace(1))\n\n\
         !air.kernel = !{{!0}}\n!air.version = !{{!90}}\n!air.language_version = !{{!91}}\n!0 = \
         !{{ptr @k, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3, !4, !5}}\n{scene}!4 = !{{i32 1, \
         !\"air.buffer\", !\"air.buffer_size\", i32 4, !\"air.location_index\", i32 1, i32 1, \
         !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, \
         !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", \
         !\"air.arg_name\", !\"out\"}}\n!5 = !{{i32 2, !\"air.thread_position_in_grid\", \
         !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"tid\"}}\n!90 = !{{i32 2, i32 8, \
         i32 0}}\n!91 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

/// `%as` loaded from the scene's member at byte `offset`.
fn member_load(space: u32, offset: u32) -> String {
    format!(
        "  %at = getelementptr inbounds i8, ptr addrspace({space}) %scene, i64 {offset}\n  %as = \
         load ptr addrspace(1), ptr addrspace({space}) %at, align 8\n"
    )
}

/// Bitcode of `text` through `llvm-as`, or `None` (printing why) where it is missing.
fn assemble(text: &str) -> Option<Vec<u8>> {
    match llvm_assemble_text(text) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the kernel: {why}\n{text}"),
    }
}

/// Whether `p` enables every feature a ray query translation needs.
fn traces(p: &DeviceProfile) -> bool {
    let enabled = &p.fields().enabled.features;
    RAY_FEATURES.iter().all(|f| enabled.contains(f))
}

fn uses(a: &FunctionArtifact, name: &str) -> u32 {
    a.strategies.entries.get(name).map_or(0, |u| u.uses)
}

/// Whether `a` declares a descriptor of `kind`.
fn declares(a: &FunctionArtifact, kind: impl Fn(&ResourceKind) -> bool) -> bool {
    a.interface.descriptors.iter().any(|d| kind(&d.slot.kind))
}

/// `text` on every committed profile: translated and valid where the profile traces rays, its
/// structure reached through the directory's address and no structure descriptor; refused as
/// `RayTracing` naming the missing features elsewhere. Returns the profiles that translated it.
fn translates(text: &str) -> Vec<String> {
    let Some(bc) = assemble(text) else {
        return Vec::new();
    };
    let mut translated = Vec::new();
    for (stem, p) in committed_profiles().expect("committed profiles") {
        match compile(&bc, "k", &p) {
            Ok(a) => {
                assert!(traces(&p), "{stem} translated without ray queries");
                assert!(
                    a.precision.contains(&PrecisionNote::DeviceRayIntersection),
                    "{stem}: {:?}",
                    a.precision
                );
                assert_eq!(uses(&a, "access.handle.structure_lookup"), 1, "{stem}");
                assert_eq!(uses(&a, "raytracing.intersect"), 1, "{stem}");
                assert!(
                    declares(&a, |k| matches!(
                        k,
                        ResourceKind::Synthetic(SyntheticBinding::AccelerationStructureDirectory)
                    )),
                    "{stem}: no directory"
                );
                assert!(
                    !declares(&a, |k| matches!(k, ResourceKind::AccelerationStructure(_))),
                    "{stem}: a structure descriptor"
                );
                let (_, words) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
                let dis = spirv_dis(&words);
                for needle in [
                    "OpCapability RayQueryKHR",
                    "OpConvertUToAccelerationStructureKHR",
                    "OpRayQueryInitializeKHR",
                ] {
                    assert!(dis.contains(needle), "{stem}: {needle} missing");
                }
                translated.push(stem);
            }
            Err(Error::Unsupported(u)) => {
                assert!(!traces(&p), "{stem} refused: {u}");
                let missing: Vec<Capability> = RAY_FEATURES
                    .into_iter()
                    .filter(|f| !p.fields().enabled.features.contains(f))
                    .map(Capability::Feature)
                    .collect();
                assert_eq!(u.feature, Feature::RayTracing, "{stem}: {u}");
                assert_eq!(u.needs, Needs::AnyOf(vec![missing]), "{stem}: {u}");
            }
            Err(other) => panic!("{stem}: {other}"),
        }
    }
    translated
}

/// The refusal of `text` on every committed profile that traces rays: the same on each.
fn refusal(text: &str) -> Option<Unsupported> {
    let bc = assemble(text)?;
    let mut first: Option<Unsupported> = None;
    for (stem, p) in committed_profiles().expect("committed profiles") {
        if !traces(&p) {
            continue;
        }
        let u = match compile(&bc, "k", &p) {
            Err(Error::Unsupported(u)) => u,
            Err(other) => panic!("{stem}: {other}"),
            Ok(_) => panic!("{stem}: compiled"),
        };
        match &first {
            Some(f) => assert_eq!(&u, f, "{stem}"),
            None => first = Some(u),
        }
    }
    Some(first.expect("a committed profile that traces rays"))
}

/// An instance structure loaded from an argument buffer's member, in constant or device memory,
/// first or second member, traverses the structure the directory names: lavapipe among the
/// profiles that translate it.
#[test]
fn an_argument_buffer_structure_traverses_its_directory_address() {
    for space in [2, 1] {
        for offset in [0, 8] {
            let text = kernel(
                space,
                Some("instance"),
                Intersector::Instanced,
                &member_load(space, offset),
                "",
            );
            let translated = translates(&text);
            if translated.is_empty() {
                return;
            }
            assert!(
                translated.iter().any(|s| s == "lavapipe"),
                "{space} {offset}: {translated:?}"
            );
        }
    }
}

/// A volatile load of the member reads its id as a plain one does, ordered as a volatile load.
#[test]
fn a_volatile_member_load_traverses_its_directory_address() {
    let text = kernel(
        2,
        Some("instance"),
        Intersector::Instanced,
        "  %as = load volatile ptr addrspace(1), ptr addrspace(2) %scene, align 8\n",
        "",
    );
    let translated = translates(&text);
    assert!(
        translated.is_empty() || translated.iter().any(|s| s == "lavapipe"),
        "{translated:?}"
    );
}

/// A primitive structure member is traversed by a primitive intersector, as an argument is.
#[test]
fn a_primitive_member_is_traversed_by_a_primitive_intersector() {
    let text = kernel(
        2,
        Some("primitive"),
        Intersector::Primitive,
        &member_load(2, 0),
        "",
    );
    let translated = translates(&text);
    assert!(
        translated.is_empty() || translated.iter().any(|s| s == "lavapipe"),
        "{translated:?}"
    );
}

/// The refusals of loaded structures: each `NotImplemented(RayTracing)`, everywhere the device
/// traces rays.
#[test]
fn loaded_structures_outside_the_shape_stay_refused() {
    let load = member_load(2, 0);
    let second = "  %bt = getelementptr inbounds i8, ptr addrspace(2) %scene, i64 8\n  %b = load \
                  ptr addrspace(1), ptr addrspace(2) %bt, align 8\n";
    let cases: [(&str, String); 5] = [
        // A plain buffer's bytes: no member says they are a structure.
        (
            "plain buffer",
            kernel(
                1,
                None,
                Intersector::Instanced,
                "  %as = load ptr addrspace(1), ptr addrspace(1) %scene, align 8\n",
                "",
            ),
        ),
        // A primitive member traversed by an instanced intersector.
        (
            "level",
            kernel(2, Some("primitive"), Intersector::Instanced, &load, ""),
        ),
        // A merge of two loaded structures.
        (
            "select",
            kernel(
                2,
                Some("instance"),
                Intersector::Instanced,
                &format!(
                    "  %a = load ptr addrspace(1), ptr addrspace(2) %scene, align 8\n{second}  \
                     %c = icmp eq i32 %tid, 0\n  %as = select i1 %c, ptr addrspace(1) %a, ptr \
                     addrspace(1) %b\n"
                ),
                "",
            ),
        ),
        // A loaded structure compared, besides its traversal.
        (
            "compare",
            kernel(
                2,
                Some("instance"),
                Intersector::Instanced,
                &load,
                "  %q = icmp eq ptr addrspace(1) %as, null\n  %z = zext i1 %q to i32\n  %p = \
                 getelementptr i32, ptr addrspace(1) %out, i64 1\n  store i32 %z, ptr \
                 addrspace(1) %p, align 4\n",
            ),
        ),
        // A loaded structure's instance count: no slot the binder knows.
        (
            "instance count",
            kernel(
                2,
                Some("instance"),
                Intersector::Instanced,
                &load,
                "  %n = call i32 @air.get_instance_count_instance_acceleration_structure(ptr \
                 addrspace(1) %as)\n  %p = getelementptr i32, ptr addrspace(1) %out, i64 1\n  \
                 store i32 %n, ptr addrspace(1) %p, align 4\n",
            ),
        ),
    ];
    for (what, text) in cases {
        let Some(u) = refusal(&text) else {
            return;
        };
        assert_eq!(u.feature, Feature::RayTracing, "{what}: {u}");
        assert_eq!(u.needs, Needs::NotImplemented, "{what}: {u}");
    }
}

/// The entries `(id, address low, address high)` of a directory's bytes.
fn entries(bytes: &[u8]) -> Vec<(u64, u32, u32)> {
    bytes
        .chunks(TABLE_ENTRY_BYTES as usize)
        .map(|e| {
            let w = |i: usize| u32::from_le_bytes(e[4 * i..4 * i + 4].try_into().expect("word"));
            (u64::from(w(0)) | (u64::from(w(1)) << 32), w(2), w(3))
        })
        .collect()
}

/// The binder plans the directory as the class's 16 entries and writes, per dispatch, each listed
/// structure's id with its address's two words, sorted after leading null entries; a structure
/// without an id is named by nothing. A list leaving its class replans, a list of another class
/// than the plan's is the plan's misuse, and a listed structure without an address is refused.
#[test]
fn the_binder_writes_each_listed_structures_address() {
    let Some(bc) = assemble(&kernel(
        2,
        Some("instance"),
        Intersector::Instanced,
        &member_load(2, 0),
        "",
    )) else {
        return;
    };
    let dev = device("lavapipe");
    let lib = Library::from_air(&bc).expect("library");
    let f = lib.function("k").expect("entry");
    let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    let pipe = dev
        .compile_compute_pipeline(&d, &Budget::standard())
        .expect("compiled");
    let directory = pipe.layout.sets[0]
        .bindings
        .iter()
        .find(|b| {
            b.source
                == BindingSource::Synthetic {
                    stage: Stage::Kernel,
                    binding: SyntheticBinding::AccelerationStructureDirectory,
                }
        })
        .map(|b| (0, b.binding))
        .expect("a directory binding");
    let mut table = StageTable::default();
    for index in [0, 1] {
        table.buffers.insert(
            index,
            BoundBuffer::Buffer {
                length: 16,
                offset: 0,
                gpu_address: 0,
                device_address: None,
                attribute_stride: None,
            },
        );
    }
    let mut state = BindState::default();
    state.stages.insert(Stage::Kernel, table);
    state.argument_acceleration_structures = vec![
        ArgumentAccelerationStructure::new(7, 0x0000_0012_3456_7000),
        ArgumentAccelerationStructure::new(0, 0),
        ArgumentAccelerationStructure::new(3, 0x0000_0001_0000_0100),
    ];
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    let write = plan
        .descriptors
        .iter()
        .find(|w| (w.set, w.binding) == directory)
        .expect("the directory's write");
    assert_eq!(
        write.source,
        WriteSource::AccelerationStructureDirectory {
            stage: Stage::Kernel,
            bytes_len: 16 * TABLE_ENTRY_BYTES,
        }
    );
    // No structure is bound at a slot: the module reads none there.
    assert!(!plan
        .descriptors
        .iter()
        .any(|w| matches!(w.source, WriteSource::AccelerationStructure { .. })));
    let cmds = plan
        .instantiate_dispatch(&state, &one_group())
        .expect("dispatch");
    let mut want = vec![(0, 0, 0); 14];
    want.extend([(3, 0x100, 0x1), (7, 0x3456_7000, 0x12)]);
    assert_eq!(entries(bytes_at(&cmds.descriptor_bytes, directory)), want);

    // Up to 15 structures keep the class; a 16th leaves it.
    let mut fifteen = state.clone();
    fifteen
        .argument_acceleration_structures
        .extend((0..12).map(|i| ArgumentAccelerationStructure::new(0x100 + i, 0x8000 + 0x100 * i)));
    assert_eq!(fifteen.shape(), state.shape());
    assert!(plan.instantiate_dispatch(&fifteen, &one_group()).is_ok());
    let mut sixteen = fifteen.clone();
    sixteen
        .argument_acceleration_structures
        .push(ArgumentAccelerationStructure::new(0x200, 0x9000));
    assert_ne!(sixteen.shape(), state.shape());
    assert_eq!(
        sixteen.shape().argument_acceleration_structures,
        AccelerationStructuresShape::of(&sixteen.argument_acceleration_structures)
    );
    assert!(plan.instantiate_dispatch(&sixteen, &one_group()).is_err());
    let wider = pipe.plan(&dev, &sixteen.shape()).expect("plan");
    assert!(wider.descriptors.iter().any(|w| w.source
        == WriteSource::AccelerationStructureDirectory {
            stage: Stage::Kernel,
            bytes_len: 32 * TABLE_ENTRY_BYTES,
        }));
    let cmds = wider
        .instantiate_dispatch(&sixteen, &one_group())
        .expect("dispatch");
    assert_eq!(bytes_at(&cmds.descriptor_bytes, directory).len(), 32 * 16);

    // A listed structure the consumer has no address for.
    let mut unbuilt = state.clone();
    unbuilt.argument_acceleration_structures[0].address = 0;
    match plan.instantiate_dispatch(&unbuilt, &one_group()) {
        Err(BindError::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::RayTracing, "{u}");
            assert_eq!(u.needs, Needs::NotRepresentable, "{u}");
        }
        other => panic!("{other:?}"),
    }
}
