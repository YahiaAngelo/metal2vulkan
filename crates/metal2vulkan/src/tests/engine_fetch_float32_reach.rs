//! A blended 32-bit float attachment a fragment function does **not** read, moved into the
//! interlocked storage form for Metal's own blend alone (`DevicePolicy::half_blend`,
//! `HalfBlendPolicy::Exact`; `pipeline::compile::fetch_inputs` names it in
//! `FramebufferFetchInputs::half_blends`).
//!
//! The device's own blend stays the default: under `HalfBlendPolicy::Device` nothing here is
//! named and no attachment moves. Under the opt-in, a four-lane output of a module **with** fast
//! math into `R32Float`, `RG32Float` or `RGBA32Float`, blended at a state whose forms are Metal's
//! (`extended_range::Blend::float32`, `Blend::fitted`) and whose factors are not one and zero,
//! takes the form; the state's words were measured against plain Metal on the M3 Max
//! (`extended_range::blend::MEASURED_THIRTY_TWO`, and the forms that need no measurement).
//!
//! What is **not** named, so that no site gets worse than the device's blend left it: a
//! one-channel attachment's `SourceColor` over `DestinationColor` (Metal rounds the destination's
//! product first there and the generic blend's mirror forms are Metal's at every word), a state
//! whose factors read the blend color (unmeasured at this width, and two such states do differ),
//! a module without fast math (measured in no state), and an output of fewer than four lanes.
//!
//! `llvm-as` assembles the fixture; a missing tool skips visibly. No GPU is used.

use crate::cache::VariantKey;
use crate::device::{DevicePolicy, DeviceProfile, HalfBlendPolicy};
use crate::error::{Error, Needs};
use crate::metal::{BlendFactor as F, BlendOperation as O, PixelFormat};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::compile::{fetch_inputs, FragmentPass};
use crate::pipeline::descriptor::ColorAttachmentDescriptor;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::link::RenderLinkInputs;
use crate::pipeline::lower_inputs::{BlendEquation, LoweringInputs};
use crate::pipeline::testing::{
    compile_with, engine_dir, fragment_inputs, link_and_validate, profile, specialized_entry,
    AssembleError,
};

/// The fixture's fragment entry: a `half4` output to color 0, which reads no `[[color(n)]]`
/// input.
const ENTRY: &str = "stage_io_fragment_point";

/// The fixture's own math mode, which every edit below replaces.
const MODE: &str = "!70 = !{!\"air.compile.fast_math_enable\"}";

/// The fixture's own return, whose alpha lane is a constant: Metal's compiler may fold a lane of
/// it into the blend (`fetch::folds_into_blend`), so the storage form is not exact for it.
const RETURN: &str = "  ret <4 x half> %7\n}";

/// A four-lane return every lane of which is opaque to Metal's fold: a call's value broadcast, no
/// constant and no fast-math operation of the function's own.
const OPAQUE: &str = "  %p = shufflevector <4 x half> %7, <4 x half> poison, \
                      <4 x i32> <i32 2, i32 2, i32 2, i32 2>\n  ret <4 x half> %p\n}";

/// `stage_io.ll` with math mode `mode` and, where `opaque`, a return no lane of which folds into
/// the blend; `None` (skipped) without `llvm-as`.
fn fixture(mode: &str, opaque: bool) -> Option<Vec<u8>> {
    let text = std::fs::read_to_string(engine_dir().join("stage_io.ll")).expect("the fixture");
    assert_eq!(text.matches(MODE).count(), 1, "the fixture's math mode");
    assert_eq!(text.matches(RETURN).count(), 1, "the fixture's return");
    let text = text.replacen(MODE, &format!("!70 = !{{!\"{mode}\"}}"), 1);
    let text = match opaque {
        true => text.replacen(RETURN, OPAQUE, 1),
        false => text,
    };
    match metal2vulkan_devtools::llvm_assemble_text(&text) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the edit: {why}"),
    }
}

/// The equation of a blend whose color and alpha take the one factor pair, added.
fn added(fs: F, fd: F) -> BlendEquation {
    BlendEquation {
        source_rgb: fs,
        destination_rgb: fd,
        rgb_operation: O::Add,
        source_alpha: fs,
        destination_alpha: fd,
        alpha_operation: O::Add,
    }
}

/// A pipeline of one attachment of `pf` blended by `eq` at color 0, and nothing else.
fn pass(pf: PixelFormat, eq: BlendEquation) -> RenderLinkInputs {
    let mut c = ColorAttachmentDescriptor::new(pf);
    c.blending_enabled = true;
    c.source_rgb_blend_factor = eq.source_rgb;
    c.destination_rgb_blend_factor = eq.destination_rgb;
    c.rgb_blend_operation = eq.rgb_operation;
    c.source_alpha_blend_factor = eq.source_alpha;
    c.destination_alpha_blend_factor = eq.destination_alpha;
    c.alpha_blend_operation = eq.alpha_operation;
    assert_eq!(BlendEquation::of(&c), Some(eq));
    let mut r = RenderLinkInputs::new();
    r.color_attachments[0] = Some(c);
    r
}

/// The colors `fetch_inputs` names for Metal's blend alone, for entry `ENTRY` of `bc` in pipeline
/// `r` on `p`.
fn named(bc: &[u8], r: &RenderLinkInputs, p: &DeviceProfile) -> Vec<u32> {
    let m = &mut crate::budget::Meter::new(&crate::budget::Budget::standard());
    let spec = specialized_entry(bc, ENTRY, p, &Default::default(), &[], m).expect("the entry");
    let d = FragmentPass {
        color_attachments: &r.color_attachments,
        depth_or_stencil: false,
        depth16: false,
        raster_sample_count: 1,
        rasterization_enabled: true,
        alpha_to_coverage: false,
        max_vertex_amplification_count: 1,
        max_call_stack_depth: 1,
        texture_write_rounding_mode: crate::metal::TextureWriteRoundingMode::TowardZero,
        sample_mask: None,
        alpha_test: None,
        depth_compare_clamp_mask: 0,
        color_group: None,
    };
    let inputs = fetch_inputs(&spec, d, &VariantKey::default(), p);
    inputs.map_or_else(Vec::new, |i| i.half_blends.into_iter().collect())
}

/// The lowering inputs of entry `ENTRY` of `bc` in pipeline `r` on `p`, the framebuffer-fetch
/// state as `fetch_inputs` names it: nothing is hand-built, so what the lowering sees is what the
/// compile gives it.
fn state(bc: &[u8], r: &RenderLinkInputs, p: &DeviceProfile) -> LoweringInputs {
    let m = &mut crate::budget::Meter::new(&crate::budget::Budget::standard());
    let spec = specialized_entry(bc, ENTRY, p, &Default::default(), &[], m).expect("the entry");
    let d = FragmentPass {
        color_attachments: &r.color_attachments,
        depth_or_stencil: false,
        depth16: false,
        raster_sample_count: 1,
        rasterization_enabled: true,
        alpha_to_coverage: false,
        max_vertex_amplification_count: 1,
        max_call_stack_depth: 1,
        texture_write_rounding_mode: crate::metal::TextureWriteRoundingMode::TowardZero,
        sample_mask: None,
        alpha_test: None,
        depth_compare_clamp_mask: 0,
        color_group: None,
    };
    let mut inputs = fragment_inputs(r, p);
    inputs.framebuffer_fetch = fetch_inputs(&spec, d, &VariantKey::default(), p);
    inputs
}

/// Whether color 0 of `a` took the interlocked storage form, and the blend it computes there.
fn fetched(a: &crate::library::FunctionArtifact) -> Option<Option<BlendEquation>> {
    a.facts.iter().find_map(|f| match *f {
        RuntimeFact::InterlockedFramebufferFetch {
            color: 0, blend, ..
        } => Some(blend),
        _ => None,
    })
}

/// Every committed profile that orders fetch through the pixel interlock alone, under the exact
/// half-blend policy.
fn profiles() -> Vec<(String, DeviceProfile)> {
    let exact = DevicePolicy {
        half_blend: HalfBlendPolicy::Exact,
        ..DevicePolicy::default()
    };
    crate::spirv::testing::committed_profiles_with(exact)
        .expect("profiles")
        .into_iter()
        .filter(|(_, p)| crate::lower::fragment::blends_half_in_shader(p))
        .collect()
}

/// Under `HalfBlendPolicy::Exact`, a 32-bit float attachment of two or four channels the function
/// does not read, blended at a measured state by a module with fast math, is named for Metal's
/// blend alone and written in the storage form with that blend, recording no blend note of any
/// kind and passing L3. Under the default policy nothing is named.
///
/// The naming is `pipeline::compile::fetch_inputs`', called here as the compile calls it, so the
/// state the lowering sees is the compile's own.
#[test]
fn an_unfetched_32_bit_float_attachment_takes_metals_measured_blend() {
    let Some(bc) = fixture("air.compile.fast_math_enable", true) else {
        return;
    };
    let default = DevicePolicy::default();
    let mut checked = 0;
    for (stem, p) in profiles() {
        let device = profile(&stem, default).expect("the default profile");
        for pf in [PixelFormat::RG32Float, PixelFormat::RGBA32Float] {
            // The two states the measurement holds word for word, and the fused multiply-add and
            // the maximum, whose forms `Blend::fitted` holds without one.
            for (fs, fd) in [
                (F::SourceColor, F::DestinationColor),
                (F::DestinationAlpha, F::OneMinusDestinationAlpha),
                (F::One, F::OneMinusSourceAlpha),
            ] {
                let eq = added(fs, fd);
                let what = format!("{stem} {pf:?} {fs:?} {fd:?}");
                let r = pass(pf, eq);
                assert_eq!(named(&bc, &r, &p), [0], "{what}");
                // The device's blend is the default: nothing is named there.
                assert!(named(&bc, &r, &device).is_empty(), "{what}: the default");
                let a = match compile_with(&bc, ENTRY, &p, &state(&bc, &r, &p)) {
                    Ok(a) => a,
                    // The device cannot view the format as a storage image.
                    Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => {
                        eprintln!("{what}: {u}");
                        continue;
                    }
                    Err(e) => panic!("{what}: {e}"),
                };
                assert_eq!(fetched(&a), Some(Some(eq)), "{what}: {:?}", a.facts);
                for note in [
                    PrecisionNote::UnmeasuredFetchBlend,
                    PrecisionNote::HalfOutputFastMathBlend,
                    PrecisionNote::HalfOutputPrecision,
                    PrecisionNote::DeviceColorConversion,
                ] {
                    assert!(!a.precision.contains(&note), "{what}: {:?}", a.precision);
                }
                assert!(a.interface.io.half_blend_colors.is_empty(), "{what}");
                assert!(a.interface.io.half_fast_colors.is_empty(), "{what}");
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "no committed profile took the form");
}

/// What the widening leaves to the device, each for a reason the measurement states: a
/// one-channel attachment's `SourceColor` over `DestinationColor` (Metal rounds the destination's
/// product first there, where the generic blend's forms are Metal's at every word), a state whose
/// factors read the blend color (unmeasured at this width, and `R32Float`'s `BlendAlpha` over
/// `OneMinusBlendAlpha` differs at 77 of 256 words under the fitted forms), a blend of one and
/// zero (which the device stores as the source anyway), and a module without fast math (Metal's
/// arithmetic there is measured in no state).
///
/// None of these is named, so none of them moves into the storage form and each keeps the blend
/// it had.
#[test]
fn an_unmeasured_32_bit_float_state_keeps_the_devices_blend() {
    let (Some(fast), Some(safe)) = (
        fixture("air.compile.fast_math_enable", true),
        fixture("air.compile.fast_math_disable", true),
    ) else {
        return;
    };
    let blend_alpha = added(F::BlendAlpha, F::OneMinusBlendAlpha);
    let wide = [PixelFormat::RG32Float, PixelFormat::RGBA32Float];
    let mut checked = 0;
    for (stem, p) in profiles() {
        // One channel: the state the measurement excludes at this width alone.
        let one = pass(
            PixelFormat::R32Float,
            added(F::SourceColor, F::DestinationColor),
        );
        assert!(named(&fast, &one, &p).is_empty(), "{stem} R32Float sc-dc");
        for pf in wide {
            // A state whose factors read the blend color, at two and four channels.
            let color = pass(pf, blend_alpha);
            assert!(named(&fast, &color, &p).is_empty(), "{stem} {pf:?} ba-1mba");
            // One and zero, added: the device stores the source, as blending disabled does.
            let replaces = pass(pf, added(F::One, F::Zero));
            assert!(
                named(&fast, &replaces, &p).is_empty(),
                "{stem} {pf:?} one-zero"
            );
            // Without fast math, at a state the fast forms fit.
            let measured = pass(pf, added(F::SourceColor, F::DestinationColor));
            assert!(
                named(&safe, &measured, &p).is_empty(),
                "{stem} {pf:?} sc-dc without fast math"
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "no committed profile was checked");
}
