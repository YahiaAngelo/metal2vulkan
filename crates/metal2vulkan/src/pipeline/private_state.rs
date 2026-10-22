//! Metal's private fixed-function pipeline state of OpenGL on Metal (Apple SPI of
//! `MTLRenderPipelineDescriptor`): which properties the translation builds, and the refusal of the
//! rest.
//!
//! Measured on an Apple M4 (macOS 27.0.1; owned Objective-C and MSL probes setting each property
//! by key-value coding):
//!
//! - `sampleMask` ANDs the coverage of a pipeline with a fragment function, the bits past the
//!   pipeline's sample count ignored, after the function runs: the function runs for every
//!   rasterized sample, masked or not, its `[[sample_mask]]` reads the rasterized coverage, and the
//!   color, depth and stencil writes and the visibility count then keep only the mask's samples
//!   (ANDed with the function's own `[[sample_mask]]` output and alpha to coverage); under
//!   `[[early_fragment_tests]]` only the colors. A pipeline without a fragment function ignores the
//!   mask: its depth and stencil writes and its visibility count keep every sample. Vulkan's
//!   `pSampleMask` ([`GraphicsState::sample_mask`]) masks before the function runs, so it is
//!   Metal's only where the function cannot tell; elsewhere the function applies the mask itself in
//!   its `SampleMask` output (`LoweringInputs::sample_mask`, [`shader_sample_mask`]), which Vulkan
//!   applies after it runs, and after early tests to the colors alone ([`sample_mask`]).
//! - `clipDistanceEnableMask`, `pointCoordLowerLeft`, `pointSizeOutputVS`, `twoSideEnabled`,
//!   `vertexDepthCompareClampMask`, `depthStencilWriteDisabled`, `sampleCoverageInvert` and
//!   `needsCustomBorderColorSamplers` (render and compute) change nothing Metal draws, at any
//!   value tried: they are kept and read by nothing. A sampler's private custom border color
//!   (`borderColorSPI` 3 with its four values) samples transparent black, the values ignored,
//!   with the flag or without: the raw border color 3 the runtime already
//!   samples so (`sampler_border_color`).
//! - `alphaTestEnabled` tests each fragment after its function runs (probes on an Apple M3 Max,
//!   macOS 27.0.1): color 0's alpha and the encoder's reference (`setAlphaTestReferenceValue:`, 1
//!   until set in each encoder) are each clamped to `[0, 1]`, a NaN taken as 0, and converted to an
//!   8-bit unorm code as an 8-bit unorm attachment stores a float, whatever color 0's format; the
//!   fragment passes where `alphaTestFunction` holds between the alpha's code and the
//!   reference's, and one that fails writes nothing and counts no visibility. The fragment
//!   function runs the test itself ([`alpha_test`], `LoweringInputs::alpha_test`); `Always`
//!   changes nothing. Metal's compiler fails a function that writes no color 0 under the test,
//!   and a function past `Always`, a pipeline without a fragment function and a color 0 other
//!   than four `float` or `half` lanes are refused, their answers unmeasured.
//! - `fragmentDepthCompareClampMask` clamps the reference of a fragment function's depth
//!   comparison (`sample_compare` and `gather_compare`) to `[0, 1]`, a NaN taken as 0: bit `s`
//!   through `[[sampler(s)]]`, whichever texture slot it compares, and bit 31 through any
//!   constexpr sampler (probes on an Apple M3 Max, macOS 27.0.1). The fragment function clamps
//!   them itself (`LoweringInputs::depth_compare_clamp_mask`, [`depth_compares_clamped`]); a
//!   comparison through an argument buffer's sampler, which no bit names, is refused under a mask
//!   other than 0.
//! - `sampleCoverage`, `logicOperationEnabled` and `pointSmoothEnabled` change what Metal draws;
//!   until the translation builds one, a value other than Metal's default refuses the pipeline
//!   `NotImplemented` ([`Feature::PrivateRenderState`]) naming it.
//! - `textureWriteRoundingMode` 2 rounds a fragment function's texture writes to half-float and
//!   `RG11B10Float` textures to nearest even, which the translation has not built: such a write is
//!   refused naming it (`PrivateRenderField::TextureWriteRounding`), where the function's slot
//!   declares such a format, and at the draw where one is bound (`lower::texture`'s
//!   `mode_rounded_write`). Its color writes are built.
//!
//! [`GraphicsState::sample_mask`]: crate::pipeline::GraphicsState::sample_mask

use crate::error::{Error, Feature, PrivateRenderField, RuntimeInput, Unsupported};
use crate::library::FunctionArtifact;
use crate::metal::CompareFunction;
use crate::pipeline::descriptor::{PrivateRenderState, RenderPipelineDescriptor};
use crate::pipeline::stage_io::{DepthCompares, StageBuiltin};
use crate::pipeline::GraphicsState;

/// The refusal of `field` at a value the translation has not built.
fn refused(field: PrivateRenderField) -> Error {
    Error::Unsupported(Unsupported::not_implemented(Feature::PrivateRenderState(
        field,
    )))
}

/// `Ok` where every property of `s` is Metal's default or built (the module docs), else the
/// refusal naming the first that is not.
pub(crate) fn check_render(s: &PrivateRenderState) -> Result<(), Error> {
    use PrivateRenderField as F;
    alpha_test(s)?;
    let unbuilt = [
        (s.sample_coverage != 1.0, F::SampleCoverage),
        (s.logic_operation_enabled, F::LogicOperation),
        (s.point_smooth_enabled, F::PointSmooth),
    ];
    match unbuilt.into_iter().find(|(set, _)| *set) {
        Some((_, field)) => Err(refused(field)),
        None => Ok(()),
    }
}

/// The alpha test of the private state `s` its fragment function runs
/// (`LoweringInputs::alpha_test`): the comparison where the test is on, `None` where it is off or
/// `Always`, which keeps every fragment; a function past `Always` is refused
/// (`PrivateRenderField::AlphaTest`), its answer unmeasured.
pub(crate) fn alpha_test(s: &PrivateRenderState) -> Result<Option<CompareFunction>, Error> {
    if !s.alpha_test_enabled {
        return Ok(None);
    }
    let function = u32::try_from(s.alpha_test_function)
        .ok()
        .and_then(|raw| CompareFunction::try_from(raw).ok())
        .ok_or_else(|| refused(PrivateRenderField::AlphaTest))?;
    Ok((function != CompareFunction::Always).then_some(function))
}

/// `Ok` where the fragment function `fragment` of the pipeline `d` runs the pipeline's alpha
/// test ([`alpha_test`]) as compiled: a pipeline with the test on and no fragment function is
/// refused (`PrivateRenderField::AlphaTest`, Metal's answer unmeasured), and a function compiled
/// without the test, or for another comparison, is compiled for the pipeline again
/// (`NeedsInput(AlphaTest)`).
pub(crate) fn alpha_tested(
    d: &RenderPipelineDescriptor,
    fragment: Option<&FunctionArtifact>,
) -> Result<(), Error> {
    let wanted = alpha_test(&d.private)?;
    match fragment {
        None if wanted.is_some() => Err(refused(PrivateRenderField::AlphaTest)),
        None => Ok(()),
        Some(f) if f.interface.io.alpha_test != wanted => {
            Err(Error::NeedsInput(RuntimeInput::AlphaTest))
        }
        Some(_) => Ok(()),
    }
}

/// `Ok` where the fragment function `fragment` of the pipeline `d` clamps the references of its
/// depth comparisons as the pipeline's `fragmentDepthCompareClampMask` says
/// (`LoweringInputs::depth_compare_clamp_mask`): one through an argument buffer's sampler under a
/// mask other than 0 is refused (`PrivateRenderField::FragmentDepthCompareClamp`, Metal's answer
/// unmeasured), and one compiled for another mask over its samplers is compiled for the pipeline
/// again (`NeedsInput(DepthCompareClamp)`).
pub(crate) fn depth_compares_clamped(
    d: &RenderPipelineDescriptor,
    fragment: Option<&FunctionArtifact>,
) -> Result<(), Error> {
    let mask = d.private.fragment_depth_compare_clamp_mask;
    let Some(c) = fragment.map(|f| f.interface.io.depth_compares) else {
        return Ok(());
    };
    if mask != 0 && c.samplers & DepthCompares::ARGUMENT != 0 {
        return Err(refused(PrivateRenderField::FragmentDepthCompareClamp));
    }
    match c.clamped == mask & c.samplers & !DepthCompares::ARGUMENT {
        true => Ok(()),
        false => Err(Error::NeedsInput(RuntimeInput::DepthCompareClamp)),
    }
}

/// The sample mask Metal applies to the pipeline `d`: the private mask's bits below the sample
/// count, where `d` has a fragment function under enabled rasterization; `None` where that keeps
/// every sample, or `d` has no fragment stage, which ignores the mask.
pub(crate) fn applied_sample_mask(d: &RenderPipelineDescriptor) -> Option<u32> {
    if d.fragment.is_none() || !d.rasterization_enabled {
        return None;
    }
    let all = match d.raster_sample_count {
        n if n >= 32 => u32::MAX,
        n => (1u32 << n) - 1,
    };
    // Metal ignores the bits past the sample count, so the high word is dropped.
    let mask = (d.private.sample_mask as u32) & all;
    (mask != all).then_some(mask)
}

/// Whether the fragment function `a` could tell Vulkan's sample mask, applied before it runs,
/// from Metal's, applied after: it writes memory (Metal runs it for the masked samples too), reads
/// `[[sample_mask]]` (Metal's is the rasterized coverage), or declares `[[early_fragment_tests]]`
/// (Metal's mask then keeps every sample's depth, stencil and visibility count, and only the
/// colors' masked samples unwritten).
fn observes_sample_mask(a: &FunctionArtifact) -> bool {
    let io = &a.interface.io;
    a.interface.writes_memory
        || io.builtins.contains(&StageBuiltin::SampleMaskIn)
        || io.early_fragment_tests
}

/// The private sample mask the fragment function of `d`, compiled without it as `a`, applies
/// itself (`LoweringInputs::sample_mask`): the mask Metal applies, where `a` could tell Vulkan's
/// from it ([`observes_sample_mask`]); `None` elsewhere.
pub(crate) fn shader_sample_mask(
    d: &RenderPipelineDescriptor,
    a: &FunctionArtifact,
) -> Option<u32> {
    applied_sample_mask(d).filter(|_| observes_sample_mask(a))
}

/// [`GraphicsState::sample_mask`] for the pipeline `d`, its fragment function `fragment` and
/// its graphics state `g`, or the refusal where neither Vulkan's mask nor the function's own
/// (`StageIo::sample_mask`) is Metal's. A function that could tell Vulkan's mask from Metal's
/// applies the mask itself, and the pipeline has no Vulkan mask; one compiled without it, or for
/// another mask, is compiled for the pipeline again (`NeedsInput(SampleMask)`). A pipeline whose
/// fragment function writes an attachment itself (a storage attachment, a shadow stencil or an
/// imageblock, none measured under the mask) is refused.
pub(crate) fn sample_mask(
    d: &RenderPipelineDescriptor,
    fragment: Option<&FunctionArtifact>,
    g: &GraphicsState,
) -> Result<Option<u32>, Error> {
    let applied = applied_sample_mask(d);
    let own = fragment.and_then(|f| f.interface.io.sample_mask);
    if own.is_some() && own != applied {
        return Err(Error::NeedsInput(RuntimeInput::SampleMask));
    }
    let Some(mask) = applied else {
        return Ok(None);
    };
    let writes_itself = !g.storage_attachments.is_empty()
        || g.shadow_stencil.is_some()
        || g.fragment_imageblock.is_some();
    if writes_itself {
        return Err(refused(PrivateRenderField::SampleMask));
    }
    match (own, fragment) {
        (Some(_), _) => Ok(None),
        (None, Some(f)) if observes_sample_mask(f) => {
            Err(Error::NeedsInput(RuntimeInput::SampleMask))
        }
        (None, _) => Ok(Some(mask)),
    }
}

/// The text the key of the render pipeline `d` appends for its private state: empty at Metal's
/// defaults and wherever only properties that change nothing differ, so every other pipeline
/// keeps its key.
pub(crate) fn render_key_fields(d: &RenderPipelineDescriptor) -> String {
    let mask = match applied_sample_mask(d) {
        Some(mask) => format!(" sample_mask {mask:#x}"),
        None => String::new(),
    };
    match alpha_test(&d.private) {
        Ok(Some(function)) => format!("{mask} alpha_test {}", function.as_raw()),
        _ => mask,
    }
}
