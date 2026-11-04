//! A barrier's name is its execution scope in AIR 2.8 as in AIR 2.9.
//!
//! Apple's compiler writes MSL 3.0 and 3.1's `simdgroup_barrier(mem_flags::mem_threadgroup)` as
//! `air.simdgroup.barrier(i32 2, i32 1)` in AIR 2.8 (scope 1, threadgroup) and as
//! `air.simdgroup.barrier(i32 2, i32 5, i32 1)` in AIR 2.9, one MSL source in both. On an M1 Pro
//! (macOS 27) the AIR 2.8 barrier does not wait for the threadgroup's other simdgroups, where
//! `air.wg.barrier(i32 2, i32 1)` does (`air::intrinsics::sync`). So the two spellings compile, on
//! every committed profile, to the same SPIR-V words, and the simdgroup barrier is not the
//! threadgroup one on some profile.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use crate::pipeline::testing::{compile, link_standalone};
use crate::spirv::testing::committed_profiles;
use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};

const DATALAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

/// A kernel that stores its thread index to threadgroup memory, runs `barrier` (`name` with
/// `args`), and stores its neighbour's word to buffer 0; in AIR 2.9 when `args` has three
/// operands, AIR 2.8 otherwise.
fn kernel(name: &str, args: &[u32]) -> String {
    let (triple, params) = match args.len() {
        3 => ("air64_v29-apple-macosx27.0.0", "i32, i32, i32"),
        _ => ("air64_v28-apple-macosx26.0.0", "i32, i32"),
    };
    let call: Vec<String> = args.iter().map(|a| format!("i32 {a}")).collect();
    let call = call.join(", ");
    format!(
        r#"target datalayout = "{DATALAYOUT}"
target triple = "{triple}"

@g = internal addrspace(3) global [64 x i32] undef, align 4

declare void @air.{name}({params})

define void @k(ptr addrspace(1) %out, i32 %i) {{
  %w = getelementptr inbounds [64 x i32], ptr addrspace(3) @g, i32 0, i32 %i
  store i32 %i, ptr addrspace(3) %w, align 4
  call void @air.{name}({call})
  %j = xor i32 %i, 1
  %r = getelementptr inbounds [64 x i32], ptr addrspace(3) @g, i32 0, i32 %j
  %v = load i32, ptr addrspace(3) %r, align 4
  %x = zext i32 %i to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %x
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}}

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}}
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

/// What entry `k` of `bitcode` compiles to on each committed profile: its standalone words, or
/// its refusal.
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
fn an_air_2_8_simdgroup_barrier_of_threadgroup_scope_is_its_air_2_9_spelling() {
    let (Some(old), Some(new)) = (
        assemble(&kernel("simdgroup.barrier", &[2, 1])),
        assemble(&kernel("simdgroup.barrier", &[2, 5, 1])),
    ) else {
        return;
    };
    let (old, new) = (outcomes(&old), outcomes(&new));
    assert!(
        new.iter().any(|(_, r)| r.is_ok()),
        "the AIR 2.9 form compiles somewhere"
    );
    assert_eq!(old, new);
}

#[test]
fn an_air_2_8_simdgroup_barrier_does_not_wait_for_the_threadgroup() {
    let (Some(sg), Some(wg)) = (
        assemble(&kernel("simdgroup.barrier", &[2, 1])),
        assemble(&kernel("wg.barrier", &[2, 1])),
    ) else {
        return;
    };
    let (sg, wg) = (outcomes(&sg), outcomes(&wg));
    assert!(
        sg.iter()
            .zip(&wg)
            .any(|((_, a), (_, b))| a.is_ok() && b.is_ok() && a != b),
        "the simdgroup barrier is the threadgroup one on every profile"
    );
}
