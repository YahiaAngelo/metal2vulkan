//! How many distinct vertex modules one vertex function turns into over a format sweep, and which
//! input accounts for each.
//!
//! A render pipeline creation translates its vertex function under the vertex-fetch projection the
//! descriptor and the device decide (`pipeline::link::vertex`), and the runtime makes one
//! `VkShaderModule` per distinct SPIR-V. So a sweep that creates one pipeline per
//! `MTLVertexFormat` pays for as many translations as it has distinct projections and as many
//! driver shader conversions as it has distinct modules — and the two numbers are not the same
//! number. This module measures both over the conformance suite's own sweeps, through the owned
//! fixture they use (`dsx_int_attributes`, whose `int_attribute_to_float4` and
//! `int_attribute_to_half4` read attribute 0 as a `float4` and as a `half4`), and attributes every
//! module to the input that makes it:
//!
//! - **fixed-function input, one module per `narrow_float`.** The whole format is
//!   `VertexAttribute` state and no instruction of the module names it, so every format the device
//!   fetches in fixed function shares one module — except that a narrow-float format (`Half*`,
//!   `FloatRG11B10`) read as a 32-bit `float` canonicalizes the device's widening to Metal's NaN,
//!   which is a second module. Necessary: the two differ instruction for instruction.
//! - **`narrow_float` of an attribute read as `half`: nothing.** The canonicalization is emitted
//!   only for a 32-bit `float` read, so the bit changes no instruction of a `half`-declared
//!   attribute's stage. `link::vertex::narrow_float_of_f32_reads` keeps it out of the projection
//!   there, and `a_half_read_compiles_one_module_for_the_whole_sweep` holds the projections and
//!   the modules at the same count.
//! - **one module per pulled format.** A format the fixed-function path cannot fetch, or one the
//!   profile asks the shader to convert (`VertexNormalizedPolicy::Exact`), is decoded in the
//!   shader, and each decode is its own code. Necessary, and the count of it is what the exact
//!   form costs: 2 modules against 21 over the 45-format sweep.
//!
//! Counts, not times: a count is the same on a loaded host as on a quiet one. No GPU is used, and
//! `llvm-as` assembles the fixture, so a missing tool skips visibly.

use std::collections::{BTreeMap, BTreeSet};

use crate::budget::SPIRV_WORDS;
use crate::device::{DevicePolicy, DeviceProfile, VertexNormalizedPolicy};
use crate::library::FunctionConstantValues;
use crate::metal::{VertexFormat, VertexStepFunction};
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::descriptor::{
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
use crate::pipeline::link::vertex::{narrow_float_of_f32_reads, vertex_fetch_projection};
use crate::pipeline::lower_inputs::{LoweringInputs, VertexFetch, VertexFetchProjection};
use crate::pipeline::stage_io::IoNumeric;
use crate::pipeline::testing::{engine_dir, llvm_assemble, meter, profile, AssembleError};

/// The venue the suite's recordings of these cases were taken on.
const PROFILE: &str = "moltenvk-m3";

/// Bytes the sweeps give each format in a record, which is also each attribute's offset step
/// (`metal2vulkan-conformance::attribute_coding::SLOT`).
const SLOT: u32 = 16;

/// The entry whose `[[attribute(0)]]` is a `float4`.
const TO_FLOAT4: &str = "int_attribute_to_float4";

/// The entry whose `[[attribute(0)]]` is a `half4`.
const TO_HALF4: &str = "int_attribute_to_half4";

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("dsx_int_attributes.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => {
            panic!("llvm-as rejected dsx_int_attributes.ll: {why}")
        }
    }
}

fn prof() -> DeviceProfile {
    profile(PROFILE, DevicePolicy::default()).expect("committed profile")
}

/// The same profile under Metal's own normalized vertex conversion
/// (`VertexNormalizedPolicy::Exact`, which `DevicePolicy::exact_answers` selects).
fn exact_prof() -> DeviceProfile {
    let policy = DevicePolicy {
        vertex_normalized: VertexNormalizedPolicy::Exact,
        ..DevicePolicy::default()
    };
    profile(PROFILE, policy).expect("committed profile")
}

/// Every format of the SDK table a float-class attribute can read, as `cases::vertex_formats`
/// selects them: all but `Invalid` and the eight 32-bit integer formats, which reach the shader as
/// integers and need an integer-declared attribute.
fn float_readable() -> Vec<VertexFormat> {
    VertexFormat::ALL
        .iter()
        .copied()
        .filter(|f| {
            !matches!(
                f,
                VertexFormat::Invalid
                    | VertexFormat::Int
                    | VertexFormat::Int2
                    | VertexFormat::Int3
                    | VertexFormat::Int4
                    | VertexFormat::UInt
                    | VertexFormat::UInt2
                    | VertexFormat::UInt3
                    | VertexFormat::UInt4
            )
        })
        .collect()
}

/// The sweeps' descriptor for `format` at slot `at`: attribute 0 of that format at byte
/// `SLOT · at` of buffer 1, whose one layout holds a record of `stride` bytes under `step`.
fn desc(
    format: VertexFormat,
    at: usize,
    stride: u32,
    step: VertexStepFunction,
) -> VertexDescriptor {
    let mut d = VertexDescriptor::new();
    d.attributes.insert(
        0,
        VertexAttributeDescriptor::new(format, SLOT * at as u32, 1),
    );
    let mut l = VertexBufferLayoutDescriptor::new(Some(stride));
    l.step_function = step;
    l.step_rate = 1;
    d.layouts.insert(1, l);
    d
}

/// The projection of attribute 0 for `format` at slot `at` under `step`, narrowed as a compile of
/// `entry` narrows it (`narrow_float_of_f32_reads`).
fn projection(
    p: &DeviceProfile,
    entry: &str,
    format: VertexFormat,
    at: usize,
    stride: u32,
    step: VertexStepFunction,
) -> VertexFetchProjection {
    let attributes = BTreeMap::from([(0, IoNumeric::Float)]);
    let mut proj = vertex_fetch_projection(Some(&desc(format, at, stride, step)), &attributes, p)
        .expect("the sweeps' descriptors are valid");
    narrow_float_of_f32_reads(&mut proj, |_| entry == TO_FLOAT4);
    proj
}

/// The final SPIR-V words of `entry` compiled under `proj`.
fn words(bc: &[u8], p: &DeviceProfile, entry: &str, proj: &VertexFetchProjection) -> Vec<u32> {
    let mut inputs = LoweringInputs::new();
    inputs.vertex_fetch = Some(proj.clone());
    compile_bitcode(
        bc,
        EntryRef::Name(entry),
        &FunctionConstantValues::defaults(),
        p,
        &inputs,
        &mut meter(),
        SPIRV_WORDS,
    )
    .expect("the fixture compiles under every projection of the sweeps")
    .module
    .words
    .to_vec()
}

/// One sweep's counts: its creations, the distinct projections they compile under (so the
/// translations), the distinct modules (so the driver's shader conversions), and the formats the
/// shader pulls.
struct Counted {
    creations: usize,
    projections: usize,
    modules: usize,
    pulled: Vec<&'static str>,
}

fn count(bc: &[u8], p: &DeviceProfile, entry: &str, step: VertexStepFunction) -> Counted {
    let formats = float_readable();
    let stride = SLOT * formats.len() as u32;
    let mut projections = BTreeSet::new();
    let mut modules = BTreeSet::new();
    let mut pulled = Vec::new();
    for (at, &format) in formats.iter().enumerate() {
        let proj = projection(p, entry, format, at, stride, step);
        if matches!(proj.attributes[&0], VertexFetch::Pulled { .. }) {
            pulled.push(format.name());
        }
        modules.insert(words(bc, p, entry, &proj));
        projections.insert(proj);
    }
    Counted {
        creations: formats.len(),
        projections: projections.len(),
        modules: modules.len(),
        pulled,
    }
}

/// The normalized formats, whose conversion `VertexNormalizedPolicy::Exact` moves into the
/// shader and the default leaves to fixed-function vertex input: the 8- and 16-bit `Normalized`
/// formats of either sign, the BGRA one, and the two packed words.
const NORMALIZED: [&str; 19] = [
    "MTLVertexFormatUCharNormalized",
    "MTLVertexFormatUChar2Normalized",
    "MTLVertexFormatUChar3Normalized",
    "MTLVertexFormatUChar4Normalized",
    "MTLVertexFormatUChar4Normalized_BGRA",
    "MTLVertexFormatCharNormalized",
    "MTLVertexFormatChar2Normalized",
    "MTLVertexFormatChar3Normalized",
    "MTLVertexFormatChar4Normalized",
    "MTLVertexFormatUShortNormalized",
    "MTLVertexFormatUShort2Normalized",
    "MTLVertexFormatUShort3Normalized",
    "MTLVertexFormatUShort4Normalized",
    "MTLVertexFormatShortNormalized",
    "MTLVertexFormatShort2Normalized",
    "MTLVertexFormatShort3Normalized",
    "MTLVertexFormatShort4Normalized",
    "MTLVertexFormatInt1010102Normalized",
    "MTLVertexFormatUInt1010102Normalized",
];

/// `vertex-format-float4`'s own sweep under the default: the device fetches every one of the 45
/// formats in fixed function, and the whole sweep is **two** modules -- one per `narrow_float`,
/// which is the only thing of the format the module's instructions name.
#[test]
fn a_float_read_compiles_two_modules_for_the_whole_sweep() {
    let Some(bc) = fixture() else { return };
    let got = count(&bc, &prof(), TO_FLOAT4, VertexStepFunction::PerVertex);
    assert_eq!(got.creations, 45, "the swept formats");
    assert_eq!(
        got.pulled,
        Vec::<&str>::new(),
        "no format of the sweep is pulled on {PROFILE} under the default"
    );
    assert_eq!(
        (got.projections, got.modules),
        (2, 2),
        "projections and modules of {} creations",
        got.creations
    );
}

/// The same sweep read as a `half4`: the device's widening is not canonicalized for a `half`
/// read, so the two fixed-function modules are **one**, and the projection does not key a second.
#[test]
fn a_half_read_compiles_one_module_for_the_whole_sweep() {
    let Some(bc) = fixture() else { return };
    let got = count(&bc, &prof(), TO_HALF4, VertexStepFunction::PerVertex);
    assert_eq!(got.creations, 45, "the swept formats");
    assert_eq!(
        (got.projections, got.modules),
        (1, 1),
        "projections and modules of {} creations",
        got.creations
    );
}

/// Under `VertexNormalizedPolicy::Exact` every normalized format is pulled and divided in the
/// shader, one module each: both forms are kept and the switch is what picks between them. The
/// count is the exact form's cost, and `NORMALIZED` is exactly what moves.
#[test]
fn metals_own_normalized_conversion_pulls_every_normalized_format() {
    let Some(bc) = fixture() else { return };
    let p = exact_prof();
    let float4 = count(&bc, &p, TO_FLOAT4, VertexStepFunction::PerVertex);
    let mut pulled = float4.pulled.clone();
    pulled.sort_unstable();
    let mut expected = NORMALIZED;
    expected.sort_unstable();
    assert_eq!(
        pulled, expected,
        "the formats the shader pulls under VertexNormalizedPolicy::Exact"
    );
    assert_eq!(
        (float4.projections, float4.modules),
        (2 + NORMALIZED.len(), 2 + NORMALIZED.len()),
        "projections and modules of {} creations",
        float4.creations
    );
    let half4 = count(&bc, &p, TO_HALF4, VertexStepFunction::PerVertex);
    assert_eq!(
        (half4.projections, half4.modules),
        (1 + NORMALIZED.len(), 1 + NORMALIZED.len()),
        "the `half` read keeps its one fixed-function module beside them"
    );
    assert_eq!(
        count(&bc, &prof(), TO_FLOAT4, VertexStepFunction::PerVertex).pulled,
        Vec::<&str>::new(),
        "the default fetches every format of the sweep in fixed function"
    );
}

/// `DevicePolicy::exact_answers` selects Metal's normalized vertex conversion, so the exact form
/// is reached by the one switch the owner's rule names and not field by field alone.
#[test]
fn exact_answers_selects_metals_normalized_vertex_conversion() {
    assert_eq!(
        DevicePolicy::default().exact_answers().vertex_normalized,
        VertexNormalizedPolicy::Exact
    );
    assert_eq!(
        DevicePolicy::default().vertex_normalized,
        VertexNormalizedPolicy::Device,
        "the fast form is the default"
    );
}

/// `narrow_float` is not key data of a `half`-declared attribute: the module compiled with the bit
/// set is the module compiled without it, word for word, for every narrow-float format the device
/// fetches in fixed function. The `float4` entry is the control: there the two differ.
#[test]
fn narrow_float_changes_no_instruction_of_a_half_read() {
    let Some(bc) = fixture() else { return };
    let p = prof();
    let formats = float_readable();
    let stride = SLOT * formats.len() as u32;
    let mut narrow = 0;
    for (at, &format) in formats.iter().enumerate() {
        let attributes = BTreeMap::from([(0, IoNumeric::Float)]);
        let d = desc(format, at, stride, VertexStepFunction::PerVertex);
        let raw = vertex_fetch_projection(Some(&d), &attributes, &p).expect("projection");
        if raw.attributes[&0] != (VertexFetch::FixedFunction { narrow_float: true }) {
            continue;
        }
        narrow += 1;
        let mut flat = raw.clone();
        narrow_float_of_f32_reads(&mut flat, |_| false);
        assert_eq!(
            flat.attributes[&0],
            VertexFetch::FixedFunction {
                narrow_float: false
            },
            "{}: the narrowing clears the bit",
            format.name()
        );
        assert_eq!(
            words(&bc, &p, TO_HALF4, &raw),
            words(&bc, &p, TO_HALF4, &flat),
            "{}: a `half` read compiles the same with and without `narrow_float`",
            format.name()
        );
        assert_ne!(
            words(&bc, &p, TO_FLOAT4, &raw),
            words(&bc, &p, TO_FLOAT4, &flat),
            "{}: a 32-bit `float` read does not",
            format.name()
        );
    }
    assert_eq!(
        narrow, 5,
        "the narrow-float formats fetched in fixed function"
    );
}

/// `integer-vertex-attributes`' own sweep: sixteen unnormalized 8- and 16-bit integer formats,
/// two entry points and two step functions. Metal's constant step is a fetch no fixed-function
/// vertex input expresses, so each of its 32 creations decodes its own format in the shader, while
/// all sixteen formats of a per-vertex creation share one module an entry. Two source functions,
/// 34 modules.
#[test]
fn the_integer_attribute_sweep_compiles_thirty_four_modules_of_two_functions() {
    let Some(bc) = fixture() else { return };
    let p = prof();
    const FORMATS: [VertexFormat; 16] = [
        VertexFormat::UChar,
        VertexFormat::UChar2,
        VertexFormat::UChar3,
        VertexFormat::UChar4,
        VertexFormat::Char,
        VertexFormat::Char2,
        VertexFormat::Char3,
        VertexFormat::Char4,
        VertexFormat::UShort,
        VertexFormat::UShort2,
        VertexFormat::UShort3,
        VertexFormat::UShort4,
        VertexFormat::Short,
        VertexFormat::Short2,
        VertexFormat::Short3,
        VertexFormat::Short4,
    ];
    let stride = SLOT * FORMATS.len() as u32;
    let mut projections = BTreeSet::new();
    let mut modules = BTreeSet::new();
    let mut creations = 0;
    let mut per_vertex_pulled = 0;
    for entry in [TO_FLOAT4, TO_HALF4] {
        for step in [VertexStepFunction::PerVertex, VertexStepFunction::Constant] {
            for (at, &format) in FORMATS.iter().enumerate() {
                creations += 1;
                let proj = projection(&p, entry, format, at, stride, step);
                if step == VertexStepFunction::PerVertex
                    && matches!(proj.attributes[&0], VertexFetch::Pulled { .. })
                {
                    per_vertex_pulled += 1;
                }
                modules.insert(words(&bc, &p, entry, &proj));
                projections.insert((entry, proj));
            }
        }
    }
    assert_eq!(creations, 64, "the sweep's pipelines");
    assert_eq!(
        per_vertex_pulled, 0,
        "every unnormalized integer format is fetched in fixed function per vertex"
    );
    // One module an entry for all sixteen per-vertex formats, and one per constant-step format.
    assert_eq!(
        (projections.len(), modules.len()),
        (2 + 2 * FORMATS.len(), 2 + 2 * FORMATS.len()),
        "projections and modules of {creations} creations"
    );
}
