//! Program-scope bindings: a function that reads a texture or a buffer through an MSL 3.2
//! program-scope variable bound to a slot (`!air.global_bindings`, a load of the variable's
//! global) compiles to the words of the same function taking that texture or buffer as a
//! parameter at the same slot, on every committed profile, and reports the slot as the
//! parameter's. The global form is the shape `xcrun metal` 32023.921 writes for a fragment
//! function's `texture2d<half> lut [[texture(10)]];` at program scope.

use crate::pipeline::interface::SlotInterface;
use crate::pipeline::sym::{SlotClass, SlotRef, Stage};
use crate::pipeline::testing::{compile, link_and_validate, llvm_assemble, AssembleError, Scratch};
use crate::spirv::testing::committed_profiles;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64-apple-ios18.0.0\"\n\n";

const TRAILER: &str = "!air.version = !{!90}\n!air.language_version = !{!91}\n\
    !llvm.module.flags = !{!92}\n\
    !90 = !{i32 2, i32 7, i32 0}\n!91 = !{!\"Metal\", i32 3, i32 2, i32 0}\n\
    !92 = !{i32 7, !\"air.workaround.sampler_state_variable_length_encoding\", i1 true}\n";

/// The texture's metadata after its parameter index: `[[texture(10)]]`, sampled.
const TEXTURE_MD: &str = "!\"air.texture\", !\"air.location_index\", i32 10, i32 1, !\"air.sample\", !\"air.arg_type_name\", !\"texture2d<half, sample>\", !\"air.arg_name\", !\"lut\"";

/// A fragment function sampling a `texture2d<half>` at its varying `uv` with a constexpr sampler:
/// `global`, through a program-scope variable bound to `[[texture(10)]]`, read twice; otherwise
/// through a parameter at that slot, appended after the function's own (where canonicalization
/// appends the global's).
fn fragment(global: bool) -> String {
    let (decl, param, tex, md, list) = if global {
        (
            "%struct.tex = type { ptr addrspace(1) }\n\
             @lut = internal unnamed_addr addrspace(2) externally_initialized constant %struct.tex undef, align 8\n",
            "",
            "  %t = load ptr addrspace(1), ptr addrspace(2) @lut, align 8\n  \
               %t2 = load ptr addrspace(1), ptr addrspace(2) @lut, align 8\n",
            format!(
                "!air.global_bindings = !{{!30}}\n!30 = !{{!\"air.global_binding\", ptr addrspace(2) @lut, !31}}\n!31 = !{{i32 -1, {TEXTURE_MD}}}\n"
            ),
            "!12",
        )
    } else {
        (
            "",
            ", ptr addrspace(1) readonly %t",
            "  %t2 = bitcast ptr addrspace(1) %t to ptr addrspace(1)\n",
            format!("!13 = !{{i32 2, {TEXTURE_MD}}}\n"),
            "!12, !13",
        )
    };
    format!(
        "{HEADER}{decl}@__air_sampler_state = internal addrspace(2) constant i64 -9188470239253755319, align 8\n\n\
         define <4 x half> @frag(<4 x float> %pos, <2 x float> %uv{param}) {{\n\
         entry:\n{tex}\
         \x20 %s = tail call {{ <4 x half>, i8 }} @air.sample_texture_2d.v4f16(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)\n\
         \x20 %c = extractvalue {{ <4 x half>, i8 }} %s, 0\n\
         \x20 %s2 = tail call {{ <4 x half>, i8 }} @air.sample_texture_2d.v4f16(ptr addrspace(1) %t2, ptr addrspace(2) @__air_sampler_state, <2 x float> zeroinitializer, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)\n\
         \x20 %c2 = extractvalue {{ <4 x half>, i8 }} %s2, 0\n\
         \x20 %r = fadd <4 x half> %c, %c2\n\
         \x20 ret <4 x half> %r\n}}\n\n\
         declare {{ <4 x half>, i8 }} @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)\n\n\
         !air.fragment = !{{!9}}\n!air.sampler_states = !{{!20}}\n\
         !9 = !{{ptr @frag, !10, !11}}\n!10 = !{{!14}}\n!11 = !{{{list}}}\n\
         !12 = !{{i32 0, !\"air.position\", !\"air.center\", !\"air.no_perspective\", !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"pos\", !\"air.arg_unused\"}}\n\
         !15 = !{{i32 1, !\"air.fragment_input\", !\"generated(2uvDv2_f)\", !\"air.center\", !\"air.perspective\", !\"air.arg_type_name\", !\"float2\", !\"air.arg_name\", !\"uv\"}}\n\
         !14 = !{{!\"air.render_target\", i32 0, i32 0, !\"air.arg_type_name\", !\"half4\"}}\n\
         !20 = !{{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}}\n\
         {md}{TRAILER}"
    )
    .replace("!11 = !{!12", "!11 = !{!12, !15")
}

/// The buffer's metadata after its parameter index: `constant uint* [[buffer(3)]]`.
const BUFFER_MD: &str = "!\"air.buffer\", !\"air.location_index\", i32 3, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"table\"";

/// A kernel copying `table[0]` to `out[0]`: `global`, through `constant uint* constant table
/// [[buffer(3)]];` at program scope; otherwise through a parameter at that slot.
fn kernel(global: bool) -> String {
    let (decl, param, load, md, list) = if global {
        (
            "@table = internal unnamed_addr addrspace(2) externally_initialized constant ptr addrspace(2) undef, align 8\n\n",
            "",
            "  %tb = load ptr addrspace(2), ptr addrspace(2) @table, align 8\n",
            format!(
                "!air.global_bindings = !{{!30}}\n!30 = !{{!\"air.global_binding\", ptr addrspace(2) @table, !31}}\n!31 = !{{i32 -1, {BUFFER_MD}}}\n"
            ),
            "!12",
        )
    } else {
        (
            "",
            ", ptr addrspace(2) noundef %tb",
            "",
            format!("!13 = !{{i32 1, {BUFFER_MD}}}\n"),
            "!12, !13",
        )
    };
    format!(
        "{HEADER}{decl}define void @k(ptr addrspace(1) noundef %out{param}) {{\nentry:\n{load}\
         \x20 %v = load i32, ptr addrspace(2) %tb, align 4\n\
         \x20 store i32 %v, ptr addrspace(1) %out, align 4\n  ret void\n}}\n\n\
         !air.kernel = !{{!9}}\n!9 = !{{ptr @k, !10, !11}}\n!10 = !{{}}\n!11 = !{{{list}}}\n\
         !12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
         {md}{TRAILER}"
    )
}

/// `text` assembled from a scratch file, or `None` (a skip) without `llvm-as`.
fn assemble(tag: &str, text: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new(&format!("m2v-engine-global-bindings-{tag}"));
    let path = scratch.path().join("module.ll");
    std::fs::write(&path, text).expect("write");
    match llvm_assemble(&path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {tag}: {why}\n{text}"),
    }
}

/// The global form and the parameter form of `entry` compile on every committed profile to the
/// same words, or are refused alike; the global form's interface holds `slot`.
fn same_words(tag: &str, make: fn(bool) -> String, entry: &str, slot: SlotRef) {
    let (Some(global), Some(param)) = (
        assemble(&format!("{tag}-global"), &make(true)),
        assemble(&format!("{tag}-param"), &make(false)),
    ) else {
        return;
    };
    let mut compiled = Vec::new();
    for (stem, p) in committed_profiles().expect("profiles") {
        let words = |bc: &[u8]| {
            compile(bc, entry, &p).map(|a| {
                let (_, w) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
                (a, w)
            })
        };
        match (words(&global), words(&param)) {
            (Ok((a, g)), Ok((_, q))) => {
                assert!(g == q, "{stem}: the forms' words differ");
                assert!(
                    a.interface.slots.contains_key(&slot),
                    "{stem}: {:?}",
                    a.interface.slots.keys().collect::<Vec<_>>()
                );
                compiled.push(stem);
            }
            (Err(g), Err(q)) => assert_eq!(g.to_string(), q.to_string(), "{stem}"),
            (g, q) => panic!("{stem}: global {:?}, parameter {:?}", g.err(), q.err()),
        }
    }
    assert!(
        compiled.iter().any(|s| s == "moltenvk-m3"),
        "compiled on {compiled:?}"
    );
}

/// A fragment function's program-scope texture is its texture parameter at the bound slot.
#[test]
fn a_program_scope_texture_compiles_as_its_parameter() {
    let slot = SlotRef {
        stage: Stage::Fragment,
        class: SlotClass::Texture,
        index: 10,
    };
    same_words("fragment", fragment, "frag", slot);
}

/// A kernel's program-scope buffer pointer is its buffer parameter at the bound slot.
#[test]
fn a_program_scope_buffer_compiles_as_its_parameter() {
    let slot = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index: 3,
    };
    same_words("kernel", kernel, "k", slot);
}

/// The texture slot's interface is a texture's.
#[test]
fn a_program_scope_texture_slot_is_a_texture() {
    let Some(bc) = assemble("fragment-slot", &fragment(true)) else {
        return;
    };
    let (_, p) = committed_profiles()
        .expect("profiles")
        .into_iter()
        .find(|(s, _)| s == "moltenvk-m3")
        .expect("moltenvk-m3");
    let a = compile(&bc, "frag", &p).expect("compiles");
    let slot = SlotRef {
        stage: Stage::Fragment,
        class: SlotClass::Texture,
        index: 10,
    };
    assert!(
        matches!(
            a.interface.slots.get(&slot),
            Some(SlotInterface::Texture(_))
        ),
        "{:?}",
        a.interface.slots.get(&slot)
    );
}
