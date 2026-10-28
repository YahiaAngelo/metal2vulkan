//! Texture block reads (`air.block_read_texture_2d.<T>`), end to end on the CPU interpreter.
//!
//! Canonicalization (`metal2vulkan_air::canon::block_read`) makes each block read of a constant
//! extent its texel reads, element `y · w + x` of the destination the texel at
//! `coord + offset + (x, y)`. Every texel of the images here names its own coordinate, so a wrong
//! order, axis, offset, or element stride lands a value that names another texel. Each kernel
//! runs on every committed profile, in both scheduling orders.
//!
//! A block read the expansion does not model (a status destination, an extent that is not a
//! constant) stays the unknown intrinsic it was, refused `UnknownIntrinsic` naming it.
//!
//! `llvm-as` assembles the synthetic kernels; without it the tests skip visibly.

use std::collections::BTreeMap;

use crate::error::{Error, Feature, Needs};
use crate::float16::f32_to_f16_bits;
use crate::lower::select::Plan;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{llvm_assemble, typed_module, AssembleError, Scratch};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::testing::committed_profiles;

const DATALAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    let text = format!(
        "target datalayout = \"{DATALAYOUT}\"\ntarget triple = \"air64-apple-ios18.0.0\"\n\n{body}"
    );
    let scratch = Scratch::new(&format!("m2v-block-read-{name}"));
    let ll = scratch.path().join("k.ll");
    std::fs::write(&ll, &text).expect("write scratch .ll");
    match llvm_assemble(&ll) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}\n{text}"),
    }
}

/// A kernel over `texture2d<{scalar}, sample>` slot 0 and `{scalar}4 *out` (buffer 0): one thread
/// block-reads the `3 x 2` block at `(1, 0)` with offset `(0, 1)`, passing `{sampler}` as the
/// sampler, `{status}` as the status destination and `{size}` as the extent, and stores its six
/// texels to `out[0..6]`.
const BLOCK_READ: &str = r#"
define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out, <2 x i32> %gid) {
entry:
  %block = alloca [6 x <4 x {ty}>], align 16
  %status = alloca i8, align 1
  %e0 = getelementptr inbounds [6 x <4 x {ty}>], ptr %block, i64 0, i64 0
  call void @air.block_read_texture_2d.v4{suffix}(ptr addrspace(1) %t, ptr addrspace(2) {sampler}, <2 x i32> <i32 1, i32 0>, <2 x i32> <i32 0, i32 1>, i32 0, i32 0, ptr %e0, ptr {status}, <2 x i32> {size})
  br label %copy

copy:
  %i = phi i64 [ 0, %entry ], [ %next, %copy ]
  %src = getelementptr inbounds [6 x <4 x {ty}>], ptr %block, i64 0, i64 %i
  %v = load <4 x {ty}>, ptr %src, align 8
  %dst = getelementptr inbounds <4 x {ty}>, ptr addrspace(1) %out, i64 %i
  store <4 x {ty}> %v, ptr addrspace(1) %dst, align 8
  %next = add i64 %i, 1
  %done = icmp eq i64 %next, 6
  br i1 %done, label %exit, label %copy

exit:
  ret void
}

declare void @air.block_read_texture_2d.v4{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32, ptr, ptr, <2 x i32>)

!air.kernel = !{!0}
!air.version = !{!6}
!air.language_version = !{!7}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<{scalar}, sample>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 {bytes}, !"air.arg_type_align_size", i32 {bytes}, !"air.arg_type_name", !"{scalar}4", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!6 = !{i32 2, i32 7, i32 0}
!7 = !{!"Metal", i32 3, i32 2, i32 0}
"#;

/// [`BLOCK_READ`] for `float` (`half` when `half`), with the operands the expansion models: no
/// sampler, no status destination, the constant extent `(3, 2)`.
fn kernel(half: bool, sampler: &str, status: &str, size: &str) -> String {
    let (ty, suffix, scalar, bytes) = if half {
        ("half", "f16", "half", "8")
    } else {
        ("float", "f32", "float", "16")
    };
    BLOCK_READ
        .replace("{ty}", ty)
        .replace("{suffix}", suffix)
        .replace("{scalar}", scalar)
        .replace("{bytes}", bytes)
        .replace("{sampler}", sampler)
        .replace("{status}", status)
        .replace("{size}", size)
}

fn texture(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Texture,
        index,
    }
}

/// Resources of one thread, `[[buffer(0)]]` holding `out`.
fn resources(plan: &Plan, out: Vec<u8>) -> Resources {
    let mut spec = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(axis), 1);
    }
    spec.insert(SpecKey::FlatLocalSize, 1);
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            (PushField::Specialized(_), PushLoc::Push { .. }) => {}
            (PushField::Specialized(SpecKey::DroppedStoreSlots(_)), _) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: out,
        }],
        push,
        spec,
    }
}

/// The value of channel `c` of texel `(x, y)`: it names the texel and the channel, and every one
/// is exact in a `half`.
fn channel(x: u32, y: u32, c: u32) -> f32 {
    (x + 8 * y + 64 * c) as f32
}

/// A `3 x 2` block read without a sampler, at `(1, 0)` with offset `(0, 1)`: element `y · 3 + x`
/// is texel `(1 + x, 1 + y)` of a 5x4 texture, for a `float4` and a `half4` element.
#[test]
fn a_block_read_is_its_texel_reads_row_by_row() {
    for half in [false, true] {
        let what = if half { "half" } else { "float" };
        let Some(bc) = assemble(what, &kernel(half, "poison", "null", "<i32 3, i32 2>")) else {
            return;
        };
        let mut ran = 0;
        for (stem, p) in committed_profiles().expect("profiles") {
            let (plan, m) =
                typed_module(&bc, "k", &p).unwrap_or_else(|e| panic!("{stem} {what}: {e}"));
            let (w, h) = (5u32, 4u32);
            let mut img = Image::new(texture(0), [w, h, 1], 1, 1, 1);
            for y in 0..h {
                for x in 0..w {
                    let i = img.index(0, 0, 0, [x, y, 0]).expect("texel");
                    img.texels[i] = [0, 1, 2, 3].map(|c| channel(x, y, c).to_bits());
                }
            }
            let lane_bytes = if half { 2 } else { 4 };
            let res = resources(&plan, vec![0; 6 * 4 * lane_bytes]);
            let mut outs = Vec::new();
            for order in [Order::Forward, Order::Reverse] {
                let cfg = Config {
                    order,
                    ..Config::default()
                };
                let (o, _) =
                    run_with_images(&m, &cfg, res.clone(), vec![img.clone()]).expect("run");
                outs.push(o.buffers);
            }
            assert_eq!(
                outs[0], outs[1],
                "{stem} {what}: scheduling orders disagree"
            );
            let out = &outs[0][0];
            for k in 0..6u32 {
                let (x, y) = (1 + k % 3, 1 + k / 3);
                for c in 0..4u32 {
                    let at = ((4 * k + c) as usize) * lane_bytes;
                    let got = &out[at..at + lane_bytes];
                    let want = channel(x, y, c);
                    if half {
                        let bits = u16::from_le_bytes([got[0], got[1]]);
                        assert_eq!(
                            bits,
                            f32_to_f16_bits(want),
                            "{stem} {what} element {k} lane {c}"
                        );
                    } else {
                        let bits = u32::from_le_bytes([got[0], got[1], got[2], got[3]]);
                        assert_eq!(bits, want.to_bits(), "{stem} {what} element {k} lane {c}");
                    }
                }
            }
            ran += 1;
        }
        assert!(ran > 0, "{what}");
    }
}

/// A block read with a status destination, or of an extent that is not a constant, is not
/// expanded: the call stays the unknown intrinsic, refused `UnknownIntrinsic` naming it.
#[test]
fn an_unmodeled_block_read_stays_unknown() {
    let cases = [
        (
            "status",
            kernel(false, "poison", "%status", "<i32 3, i32 2>"),
        ),
        ("extent", kernel(false, "poison", "null", "%gid")),
        (
            "too large",
            kernel(false, "poison", "null", "<i32 16, i32 16>"),
        ),
    ];
    for (what, text) in cases {
        let Some(bc) = assemble(&what.replace(' ', "-"), &text) else {
            return;
        };
        let (stem, p) = committed_profiles()
            .expect("profiles")
            .into_iter()
            .next()
            .expect("a profile");
        match typed_module(&bc, "k", &p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(
                    (u.feature, u.needs),
                    (Feature::UnknownIntrinsic, Needs::NotImplemented),
                    "{stem} {what}"
                );
                let symbol = u.symbol.as_ref().map(|s| s.to_string());
                assert_eq!(
                    symbol.as_deref(),
                    Some("air.block_read_texture_2d.v4f32"),
                    "{stem} {what}"
                );
            }
            other => panic!("{stem} {what}: {:?}", other.map(|_| ())),
        }
    }
}
