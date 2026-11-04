//! A constexpr sampler written inline as its state word is the sampler a listed global holding
//! the word is.
//!
//! Older frontends (macOS 10.15's SDK) pass a `constexpr sampler` to a sampling call as
//! `inttoptr (i64 W to ptr addrspace(2))`, the one-word state itself, with no global and no
//! `!air.sampler_states` entry (`air::canon::inline_sampler`). Such a kernel compiles, on every
//! committed profile, to the words of the same kernel with `W` in a listed constant global; one
//! word used twice is one sampler, two words two. A word that does not decode is
//! `BadSamplerState`, and the word anywhere but a sampler operand stays refused.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use crate::error::{AbiError, Error, InputErrorKind};
use crate::library::FunctionArtifact;
use crate::pipeline::sym::{ResourceKind, SamplerRef};
use crate::pipeline::testing::{compile, link_standalone};
use crate::spirv::testing::committed_profiles;
use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};

const DATALAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

/// The one-word form (bit 63 set) of clamp-to-edge on every axis, nearest filters, compare
/// `never`, LOD clamps 0 and 65504.
const EDGE: u64 = 0x807b_ff00_0008_0049;
/// The same with `repeat` on every axis.
const REPEAT: u64 = 0x807b_ff00_0008_0092;

/// `inttoptr (i64 w to ptr addrspace(2))`.
fn inline(w: u64) -> String {
    format!(
        "ptr addrspace(2) inttoptr (i64 {} to ptr addrspace(2))",
        w as i64
    )
}

/// A kernel sampling texture 0 at (0.5, 0.5) through `first` and at (0.25, 0.25) through
/// `second`, storing the sum of the two red lanes to buffer 0; `globals` and `listed` are its
/// constant globals and their `!air.sampler_states` nodes.
fn kernel(first: &str, second: &str, globals: &str, listed: &[&str]) -> String {
    let nodes: Vec<String> = (0..listed.len()).map(|i| format!("!{}", 6 + i)).collect();
    let states = if listed.is_empty() {
        String::new()
    } else {
        let mut s = format!("!air.sampler_states = !{{{}}}\n", nodes.join(", "));
        for (i, g) in listed.iter().enumerate() {
            s.push_str(&format!(
                "!{} = !{{!\"air.sampler_state\", ptr addrspace(2) {g}}}\n",
                6 + i
            ));
        }
        s
    };
    format!(
        r#"target datalayout = "{DATALAYOUT}"
target triple = "air64_v28-apple-macosx26.0.0"

{globals}

declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out, i32 %gid) {{
  %r0 = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, {first}, <2 x float> <float 5.000000e-01, float 5.000000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %r1 = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, {second}, <2 x float> <float 2.500000e-01, float 2.500000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %c0 = extractvalue {{ <4 x float>, i8 }} %r0, 0
  %c1 = extractvalue {{ <4 x float>, i8 }} %r1, 0
  %v0 = extractelement <4 x float> %c0, i32 0
  %v1 = extractelement <4 x float> %c1, i32 0
  %v = fadd float %v0, %v1
  %i = zext i32 %gid to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %out, i64 %i
  store float %v, ptr addrspace(1) %p, align 4
  ret void
}}

!air.kernel = !{{!0}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
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

/// The distinct samplers a module's combined bindings use.
fn samplers(a: &FunctionArtifact) -> Vec<SamplerRef> {
    let mut out: Vec<SamplerRef> = Vec::new();
    for d in &a.interface.descriptors {
        if let ResourceKind::CombinedTextureSampler(c) = d.slot.kind {
            if !out.contains(&c.sampler) {
                out.push(c.sampler);
            }
        }
    }
    out
}

/// What entry `k` of `bitcode` compiles to on each committed profile: its standalone words,
/// or its refusal.
fn outcomes(bitcode: &[u8]) -> Vec<(String, Result<Vec<u32>, String>)> {
    committed_profiles()
        .expect("the committed profiles")
        .iter()
        .map(|(name, p)| {
            let r = compile(bitcode, "k", p)
                .map_err(|e| e.to_string())
                .and_then(|a| {
                    link_standalone(&a, p)
                        .map(|(_, w)| w)
                        .map_err(|e| format!("link: {e}"))
                });
            (name.clone(), r)
        })
        .collect()
}

#[test]
fn an_inline_sampler_word_is_the_listed_sampler_it_encodes() {
    let listed = format!(
        "@a = internal addrspace(2) constant i64 {}, align 8\n@b = internal addrspace(2) constant i64 {}, align 8",
        EDGE as i64, REPEAT as i64
    );
    let (Some(words), Some(globals)) = (
        assemble(&kernel(&inline(EDGE), &inline(REPEAT), "", &[])),
        assemble(&kernel(
            "ptr addrspace(2) @a",
            "ptr addrspace(2) @b",
            &listed,
            &["@a", "@b"],
        )),
    ) else {
        return;
    };
    let (inline, global) = (outcomes(&words), outcomes(&globals));
    assert!(
        global.iter().any(|(_, r)| r.is_ok()),
        "the listed form compiles somewhere"
    );
    assert_eq!(inline, global);
}

#[test]
fn one_word_is_one_sampler_and_two_words_two() {
    let (Some(one), Some(two)) = (
        assemble(&kernel(&inline(EDGE), &inline(EDGE), "", &[])),
        assemble(&kernel(&inline(EDGE), &inline(REPEAT), "", &[])),
    ) else {
        return;
    };
    let mut compiled = 0;
    for (name, p) in &committed_profiles().expect("the committed profiles") {
        let (Ok(a), Ok(b)) = (compile(&one, "k", p), compile(&two, "k", p)) else {
            continue;
        };
        assert_eq!(samplers(&a).len(), 1, "{name}: one word");
        assert_eq!(samplers(&b).len(), 2, "{name}: two words");
        compiled += 1;
    }
    assert!(compiled > 0, "no profile compiles the kernels");
}

#[test]
fn an_inline_word_that_does_not_decode_is_a_bad_sampler_state() {
    // Bit 60 must be zero.
    let Some(bad) = assemble(&kernel(&inline(EDGE | 1 << 60), &inline(EDGE), "", &[])) else {
        return;
    };
    for (name, p) in &committed_profiles().expect("the committed profiles") {
        match compile(&bad, "k", p) {
            Err(Error::InvalidInput(e)) => assert_eq!(
                e.kind,
                InputErrorKind::Abi(AbiError::BadSamplerState),
                "{name}"
            ),
            other => panic!(
                "{name}: expected BadSamplerState, got {:?}",
                other.map(|_| ())
            ),
        }
    }
}

#[test]
fn an_inline_word_outside_a_sampler_operand_stays_refused() {
    // The word as the address of a constant-space load.
    let text = kernel(&inline(EDGE), &inline(EDGE), "", &[]).replace(
        "  %v = fadd float %v0, %v1\n",
        &format!(
            "  %w = load float, {}, align 4\n  %v = fadd float %v0, %w\n",
            inline(EDGE)
        ),
    );
    let Some(bitcode) = assemble(&text) else {
        return;
    };
    for (name, p) in &committed_profiles().expect("the committed profiles") {
        assert!(compile(&bitcode, "k", p).is_err(), "{name}: refused");
    }
}
