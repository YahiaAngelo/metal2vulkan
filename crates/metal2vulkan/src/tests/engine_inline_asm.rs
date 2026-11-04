//! An empty inline-assembly call is dropped; every other inline assembly is refused.
//!
//! `__asm__ volatile("")` (a compiler barrier some sources use) compiles to a call of the inline
//! assembly value `asm sideeffect "", ""` of type `void ()`. The empty text runs no instruction,
//! so the bitcode reader drops the call like a marker: a kernel with such calls compiles, on every
//! committed profile, to the same words as the kernel without them. Inline assembly with text,
//! constraints, operands or a result is outside the typed model and stays
//! `Unsupported(Constant { code: 30 })`.
//!
//! `llvm-as` assembles the text; without it the test skips visibly.

use crate::error::{Error, Feature};
use crate::pipeline::testing::{compile, link_standalone};
use crate::spirv::testing::committed_profiles;
use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};

/// `kernel_copy_word.ll`'s kernel, with `before` and `between` placed before its load and
/// between its load and store.
fn kernel(before: &str, between: &str) -> String {
    format!(
        r#"source_filename = "inline_asm.metal"

define void @copy_word(ptr addrspace(1) %input, ptr addrspace(1) %output) {{
{before}
  %value = load i32, ptr addrspace(1) %input, align 4
{between}
  store i32 %value, ptr addrspace(1) %output, align 4
  ret void
}}

!air.kernel = !{{!0}}
!0 = !{{ptr @copy_word, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4}}
!3 = !{{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"input"}}
!4 = !{{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"output"}}
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

#[test]
fn an_empty_inline_assembly_call_is_dropped() {
    let Some(plain) = assemble(&kernel("", "")) else {
        return;
    };
    // With and without `sideeffect`, twice, and once more between the load and the store: one
    // value used by three calls.
    let empty = "  call void asm sideeffect \"\", \"\"()\n  call void asm \"\", \"\"()";
    let Some(with) = assemble(&kernel(empty, "  call void asm sideeffect \"\", \"\"()")) else {
        return;
    };
    let profiles = committed_profiles().expect("the committed profiles");
    assert!(!profiles.is_empty());
    for (name, p) in &profiles {
        let words = |bitcode: &[u8]| {
            let a = compile(bitcode, "copy_word", p)
                .unwrap_or_else(|e| panic!("{name}: the kernel compiles: {e}"));
            link_standalone(&a, p)
                .unwrap_or_else(|e| panic!("{name}: the kernel links: {e}"))
                .1
        };
        assert_eq!(words(&with), words(&plain), "{name}");
    }
}

#[test]
fn other_inline_assembly_is_refused() {
    let profiles = committed_profiles().expect("the committed profiles");
    let (name, p) = profiles.first().expect("a committed profile");
    for (what, before, between) in [
        ("text", "  call void asm sideeffect \"nop\", \"\"()", ""),
        (
            "a clobber",
            "  call void asm sideeffect \"\", \"~{memory}\"()",
            "",
        ),
        (
            "an operand",
            "",
            "  call void asm sideeffect \"\", \"r\"(i32 %value)",
        ),
        ("a result", "  %r = call i32 asm \"\", \"=r\"()", ""),
    ] {
        let Some(bitcode) = assemble(&kernel(before, between)) else {
            return;
        };
        match compile(&bitcode, "copy_word", p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::Constant { code: 30 }, "{name}: {what}")
            }
            other => {
                panic!("{name}: inline assembly with {what}: expected the refusal, got {other:?}")
            }
        }
    }
}
