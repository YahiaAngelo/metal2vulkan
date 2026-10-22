//! The mesh arguments module: valid on every committed profile with `VK_EXT_mesh_shader`, and
//! Metal's rule on the CPU interpreter (copied, zeroed, and zeroed with `MESH_GRID_PAST_LIMITS`),
//! including the source residual and a short range.

use super::*;
use crate::bind::status::{ExecutionStatus, StatusReport};
use crate::device::VkExtension;
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// Run the module of `rule` over `source` (bound whole) with push `residual`: the scratch's
/// counts and the status report.
fn pass(rule: MeshGridRule, source: &[u32], residual: u32) -> ([u32; 3], StatusReport) {
    let m = typed_module(rule).expect("module");
    let mut out = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let res = Resources {
            buffers: vec![
                Buffer {
                    slot: Some(SOURCE),
                    address: None,
                    bytes: source.iter().flat_map(|w| w.to_le_bytes()).collect(),
                },
                Buffer {
                    slot: Some(SCRATCH),
                    address: None,
                    bytes: vec![0xEE; 16],
                },
                Buffer {
                    slot: Some(STATUS),
                    address: None,
                    bytes: ExecutionStatus::initial().to_vec(),
                },
            ],
            push: residual.to_le_bytes().to_vec(),
            spec: Default::default(),
        };
        let cfg = Config {
            order,
            ..Config::default()
        };
        out.push(run(&m, &cfg, res).expect("run").buffers);
    }
    assert_eq!(out[0], out[1], "order independent");
    let b = &out[0];
    let scratch = words(&b[1]);
    assert_eq!(scratch[3], 0xEEEE_EEEE, "the pass writes 12 bytes only");
    let status: [u8; 32] = b[2].as_slice().try_into().expect("status");
    (
        [scratch[0], scratch[1], scratch[2]],
        ExecutionStatus::decode(&status),
    )
}

/// RADV's mesh and task limits on the Radeon 780M.
fn radv() -> crate::device::MeshLimits {
    crate::device::MeshLimits {
        max_task_work_group_count: [65535; 3],
        max_task_work_group_total_count: 4_194_304,
        max_mesh_work_group_count: [65535; 3],
        max_mesh_work_group_total_count: 4_194_304,
        ..Default::default()
    }
}

#[test]
fn the_module_is_valid_on_every_committed_profile_with_mesh_shading() {
    for (name, p) in committed_profiles().expect("profiles") {
        if !p
            .fields()
            .enabled
            .extensions
            .contains(&VkExtension::ExtMeshShader)
        {
            continue;
        }
        for form in [MeshArgumentsForm::Mesh, MeshArgumentsForm::Object] {
            let rule = MeshGridRule::of(form, &p.fields().limits.mesh);
            // Every such profile launches each mesh grid Metal draws.
            if form == MeshArgumentsForm::Mesh {
                assert!(
                    rule.device.axes.iter().all(|&a| a >= METAL_MESH_GRID_AXIS)
                        && rule.device.total >= METAL_MESH_GRID_TOTAL,
                    "{name}: {rule:?}"
                );
            }
            assert_valid(typed_module(rule).expect("module"), &p);
        }
    }
}

/// Metal's mesh grids (the module docs): drawn up to 65,535 an axis and 1,048,575 in all, else
/// nothing, with no fault.
#[test]
fn a_mesh_grid_draws_as_metal_s() {
    let rule = MeshGridRule::of(MeshArgumentsForm::Mesh, &radv());
    for (counts, drawn) in [
        ([3, 1, 1], true),
        ([65535, 1, 1], true),
        ([1, 65535, 1], true),
        ([1, 1, 65535], true),
        ([1023, 1025, 1], true),
        ([16, 65535, 1], true),
        ([41, 41, 39], true),
        ([0, 5, 1], true),
        ([65536, 1, 1], false),
        ([1, 1, 65536], false),
        ([1024, 1024, 1], false),
        ([128, 128, 64], false),
        ([4096, 1024, 1], false),
        ([u32::MAX, u32::MAX, u32::MAX], false),
        ([65535, 65535, 65535], false),
    ] {
        let (out, report) = pass(rule, &counts, 0);
        let want = if drawn { counts } else { [0; 3] };
        assert_eq!(out, want, "{counts:?}");
        assert!(!report.mesh_grid_past_limits, "{counts:?}");
        assert_eq!(metal_mesh_grid_draws_nothing(counts), !drawn, "{counts:?}");
    }
}

/// An object grid Metal draws at every count: copied within the task limits, else zeroed and
/// reported.
#[test]
fn an_object_grid_past_the_task_limits_is_reported() {
    let rule = MeshGridRule::of(MeshArgumentsForm::Object, &radv());
    for (counts, within) in [
        ([3, 1, 1], true),
        ([65535, 64, 1], true),
        ([2048, 2048, 1], true),
        ([0, 70000, 1], false),
        ([65536, 1, 1], false),
        ([2048, 2049, 1], false),
        ([u32::MAX, 2, 1], false),
    ] {
        let (out, report) = pass(rule, &counts, 0);
        let want = if within { counts } else { [0; 3] };
        assert_eq!(out, want, "{counts:?}");
        assert_eq!(report.mesh_grid_past_limits, !within, "{counts:?}");
    }
}

#[test]
fn the_residual_selects_the_arguments_and_a_short_range_reads_zero() {
    let rule = MeshGridRule::of(MeshArgumentsForm::Mesh, &radv());
    let (out, _) = pass(rule, &[9, 9, 3, 4, 5, 9], 8);
    assert_eq!(out, [3, 4, 5]);
    let (out, report) = pass(rule, &[500, 2], 4);
    assert_eq!(out, [2, 0, 0]);
    assert!(!report.mesh_grid_past_limits);
}

#[test]
fn the_pass_binds_an_aligned_source_range() {
    let form = MeshArgumentsForm::Mesh;
    let a = MeshArguments::new(IndirectRef::new(3, 300), form, 256).expect("pass");
    assert_eq!((a.source_range.offset, a.source_range.range), (256, 56));
    assert_eq!(u32::from_le_bytes(a.push_constants), 44);
    let a = MeshArguments::new(IndirectRef::new(3, 12), form, 1).expect("pass");
    assert_eq!((a.source_range.offset, a.source_range.range), (12, 12));
    assert_eq!(MeshArguments::new(IndirectRef::new(3, 2), form, 4), None);
}
