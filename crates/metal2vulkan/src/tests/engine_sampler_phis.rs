//! A phi of constexpr samplers is a phi of their handle indices.
//!
//! A sampler chosen on a branch, `s = c ? sampler_a : sampler_b` between two `constexpr sampler`s,
//! reaches the sampling as a `phi` of the two states' addresses where the frontend keeps the
//! branch (a `select` where it flattens it). The handle phi's copy on each edge stores the
//! incoming constant's handle index, as `select` takes it: a constexpr sampler's, or the null
//! handle for a null pointer. So the phi form binds and samples what the `select` form does. A phi
//! with any other constant (a pointer the module does not list as a sampler state) stays refused.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use crate::error::Error;
use crate::library::FunctionArtifact;
use crate::pipeline::sym::{ResourceKind, SamplerRef};
use crate::pipeline::testing::compile;
use crate::spirv::testing::committed_profiles;
use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};

const DATALAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

/// Clamp-to-edge on every axis, nearest filters, compare `never`, LOD clamps 0 and 65504.
const EDGE: u64 = 0x007b_ff00_0008_0049;
/// The same with `repeat` on every axis.
const REPEAT: u64 = 0x007b_ff00_0008_0092;

/// A kernel sampling texture 0 at (0.5, 0.5) through `@a` on even threads and through `other`
/// on odd ones, joined by `join` (`phi` or `select`), storing the `float` result to buffer 0.
fn kernel(join: &str, other: &str) -> String {
    let choice = match join {
        "phi" => format!(
            "  br i1 %even, label %pick, label %odd\nodd:\n  br label %pick\npick:\n  %s = phi ptr addrspace(2) [ @a, %entry ], [ {other}, %odd ]"
        ),
        _ => format!("  %s = select i1 %even, ptr addrspace(2) @a, ptr addrspace(2) {other}"),
    };
    format!(
        r#"target datalayout = "{DATALAYOUT}"
target triple = "air64_v28-apple-macosx26.0.0"

@a = internal addrspace(2) constant [2 x i64] [i64 {EDGE}, i64 0], align 8
@b = internal addrspace(2) constant [2 x i64] [i64 {REPEAT}, i64 0], align 8

declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out, i32 %gid) {{
entry:
  %bit = and i32 %gid, 1
  %even = icmp eq i32 %bit, 0
{choice}
  %r = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> <float 5.000000e-01, float 5.000000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %v4 = extractvalue {{ <4 x float>, i8 }} %r, 0
  %v = extractelement <4 x float> %v4, i32 0
  %i = zext i32 %gid to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %out, i64 %i
  store float %v, ptr addrspace(1) %p, align 4
  ret void
}}

!air.kernel = !{{!0}}
!air.sampler_states = !{{!6, !7}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!6 = !{{!"air.sampler_state", ptr addrspace(2) @a}}
!7 = !{{!"air.sampler_state", ptr addrspace(2) @b}}
"#
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
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the kernel: {why}"),
    }
}

/// The combined bindings a module declares, sorted: (texture index, sampler).
fn combined(a: &FunctionArtifact) -> Vec<(u32, SamplerRef)> {
    let mut out: Vec<_> = a
        .interface
        .descriptors
        .iter()
        .filter_map(|d| match d.slot.kind {
            ResourceKind::CombinedTextureSampler(c) => Some((d.slot.index, c.sampler)),
            _ => None,
        })
        .collect();
    out.sort_unstable_by_key(|x| format!("{x:?}"));
    out
}

#[test]
fn a_phi_of_constexpr_samplers_binds_what_their_select_binds() {
    let (Some(phi), Some(select)) = (
        assemble(&kernel("phi", "@b")),
        assemble(&kernel("select", "@b")),
    ) else {
        return;
    };
    let profiles = committed_profiles().expect("the committed profiles");
    let mut compiled = 0;
    for (name, p) in &profiles {
        let Ok(s) = compile(&select, "k", p) else {
            continue;
        };
        let f = compile(&phi, "k", p)
            .unwrap_or_else(|e| panic!("{name}: the phi compiles where the select does: {e}"));
        let bound = combined(&f);
        assert_eq!(bound, combined(&s), "{name}");
        assert_eq!(bound.len(), 2, "{name}: both samplers bound: {bound:?}");
        assert_ne!(bound[0].1, bound[1].1, "{name}: two samplers");
        compiled += 1;
    }
    assert!(compiled > 0, "no profile compiles the select form");
}

#[test]
fn a_phi_of_a_constexpr_sampler_and_null_binds_the_one_sampler() {
    let (Some(phi), Some(select)) = (
        assemble(&kernel("phi", "null")),
        assemble(&kernel("select", "null")),
    ) else {
        return;
    };
    let profiles = committed_profiles().expect("the committed profiles");
    let mut compiled = 0;
    for (name, p) in &profiles {
        let Ok(s) = compile(&select, "k", p) else {
            continue;
        };
        let f = compile(&phi, "k", p)
            .unwrap_or_else(|e| panic!("{name}: the phi compiles where the select does: {e}"));
        assert_eq!(combined(&f), combined(&s), "{name}");
        assert_eq!(combined(&f).len(), 1, "{name}");
        compiled += 1;
    }
    assert!(compiled > 0, "no profile compiles the select form");
}

#[test]
fn a_phi_with_another_constant_stays_refused() {
    // A pointer constant that is no listed sampler state.
    let Some(phi) = assemble(&kernel("phi", "inttoptr (i64 4096 to ptr addrspace(2))")) else {
        return;
    };
    for (name, p) in &committed_profiles().expect("the committed profiles") {
        assert!(
            matches!(compile(&phi, "k", p), Err(Error::Unsupported(_))),
            "{name}: refused"
        );
    }
}
