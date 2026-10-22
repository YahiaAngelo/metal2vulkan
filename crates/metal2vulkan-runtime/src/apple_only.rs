//! The Apple-only calls of the consumer's trait: each is accepted where it is exact on Vulkan, and
//! otherwise refused with [`AppleOnly`] naming the missing feature, and [`AppleOnly::gap`] saying
//! why: Metal's answer is not measured, or the runtime has not built it. None is a device's gap:
//! each is the runtime's own.
//!
//! Exact means one of:
//! - the call states Metal's default, which every runtime object already has: a line width of
//!   1, the viewport transform on, no protection options;
//! - the call sets encoder state the runtime draws as Metal does: a provoking vertex mode of 0
//!   or 1, `render/provoking.rs`;
//! - the call acts on state no runtime object has: Vulkan shows no texture compression, so a
//!   reinterpretation flush has nothing to flush, and after an invalidation Metal leaves the
//!   contents undefined, so keeping them is one of the outcomes Metal allows; and no pipeline the
//!   runtime builds has a fixed-function alpha test, so its reference value has no effect, as on
//!   Metal with the test off.
//!
//! - `setPointSize:`, which Metal measured makes a no-op: the point is the vertex function's
//!   `[[point_size]]`.
//!
//! A provoking vertex mode past 1, which Metal draws a third way that differs between pipeline
//! states, is accepted and drawn as mode 1, announced at the call (`provoking-vertex-unmatched`).
//!
//! Every other call is refused, and records nothing:
//! - nonzero protection options, whose effect on Metal is not measured (Vulkan's protected
//!   memory is the nearest form);
//! - a line width other than 1 (Apple's wide-line rule is unmeasured, where Vulkan has
//!   `wideLines`), and vertex amplification mode, and depth or stencil "cleared", whose meaning
//!   or raw values are unmeasured;
//! - turning the viewport transform off, which Vulkan has no switch for and whose meaning on
//!   Metal is not measured;
//! - a texture over an IOSurface made with `kIOSurfaceClientDirection`, a key in no header of the
//!   public macOS SDK, refused by the IOSurface plane's stand-in
//!   (`Buffer::new_texture_with_iosurface_plane`), which is no call of the table.
//!
//! Texture fills (`blit_fill.rs`) refuse nothing here: a multisample texture's samples, a texture
//! buffer's bytes and a color packed in a format Metal packs no color in (the command buffer's
//! last packed texel, announced as `unpacked-fill-color`) are built. A multisample fill a device
//! cannot copy into the texture is refused there, `Unsupported`, as the device's gap.
//!
//! [`AppleOnlyCall`] is the table, and every call of it is here: the command buffer's, the
//! compute, render and blit encoders', and the texture's protection options. A texture's
//! rotation is the runtime's (`TextureDescriptor::rotation`), and command predication, loops
//! included, is `predication.rs`'s.

use super::blit::BlitCommandEncoder;
use super::blit_texture::check_slice_level;
use super::command_buffer::CommandBuffer;
use super::compute::ComputeCommandEncoder;
use super::error::Error;
use super::render::RenderCall;
use super::render::RenderCommandEncoder;
use super::render::RenderLowering;
use super::replay::BlitCall;
use super::texture::Texture;

/// An Apple-only feature the runtime refuses, each for the reason [`AppleOnly::gap`] gives (No
/// device lacks any of them; each is a gap of the runtime's own).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AppleOnly {
    /// Protected content: nonzero `setProtectionOptions:`.
    ProtectedContent,
    /// A line width other than 1 (`setLineWidth:`).
    LineWidth,
    /// The viewport transform turned off (`setViewportTransformEnabled:NO`).
    ViewportTransformDisabled,
    /// `setVertexAmplificationMode:value:`.
    VertexAmplificationMode,
    /// `setDepthCleared`.
    DepthCleared,
    /// `setStencilCleared`.
    StencilCleared,
    /// A texture over an IOSurface made with `kIOSurfaceClientDirection`
    /// (`newTextureWithDescriptor:iosurface:plane:` of such a surface): IOSurface SPI, a key
    /// IOSurface.framework exports and no header of the public macOS SDK declares, so the
    /// surface is the consumer's to build. The consumer reports that Metal drops a GPU write into
    /// a surface made with value 2 (read-only), without a fault; the runtime does not model the
    /// key's values.
    SurfaceClientDirection,
}

/// Why the runtime refuses an [`AppleOnly`] feature. None is a device's gap: no measured Vulkan
/// device lacks what an exact form would need. Each is the runtime's own, and a measurement can
/// lift it. A color fill whose bytes Metal leaves undefined, once refused as having no one
/// answer, is built: the runtime gives one of Metal's answers.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AppleOnlyGap {
    /// What Metal does on Apple's GPU is not measured, so the runtime does not guess it: the
    /// runtime's gap, which a measurement would let it build.
    Unmeasured,
}

impl AppleOnly {
    /// Why the runtime refuses the feature ([`AppleOnlyGap`]).
    pub const fn gap(self) -> AppleOnlyGap {
        match self {
            Self::ProtectedContent
            | Self::LineWidth
            | Self::ViewportTransformDisabled
            | Self::VertexAmplificationMode
            | Self::DepthCleared
            | Self::StencilCleared
            | Self::SurfaceClientDirection => AppleOnlyGap::Unmeasured,
        }
    }
}

/// Metal's `protectionOptions` of a resource made without any, and of every runtime texture.
pub const DEFAULT_PROTECTION_OPTIONS: u64 = 0;

/// One Apple-only call (class A) with the arguments its answer depends on: the table each
/// encoder's, descriptor's and command buffer's method of that call answers by.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AppleOnlyCall {
    /// `invalidateCompressedTexture:` (blit encoder).
    InvalidateCompressedTexture,
    /// `invalidateCompressedTexture:slice:level:` (blit encoder).
    InvalidateCompressedTextureSlice,
    /// `insertCompressedTextureReinterpretationFlush` (compute encoder).
    CompressedTextureReinterpretationFlush,
    /// `setProtectionOptions:` (command buffer).
    SetProtectionOptions(u64),
    /// `setAlphaTestReferenceValue:` (render encoder).
    SetAlphaTestReferenceValue(f32),
    /// `setLineWidth:` (render encoder).
    SetLineWidth(f32),
    /// `setPointSize:` (render encoder).
    SetPointSize(f32),
    /// `setProvokingVertexMode:` (render encoder).
    SetProvokingVertexMode(u64),
    /// `setViewportTransformEnabled:` (render encoder).
    SetViewportTransformEnabled(bool),
    /// `setVertexAmplificationMode:value:` (render encoder).
    SetVertexAmplificationMode {
        /// The mode given.
        mode: u64,
        /// The value given.
        value: u64,
    },
    /// `setDepthCleared` (render encoder).
    SetDepthCleared,
    /// `setStencilCleared` (render encoder).
    SetStencilCleared,
}

impl AppleOnlyCall {
    /// Metal's selector.
    pub fn selector(self) -> &'static str {
        match self {
            AppleOnlyCall::InvalidateCompressedTexture => "invalidateCompressedTexture:",
            AppleOnlyCall::InvalidateCompressedTextureSlice => {
                "invalidateCompressedTexture:slice:level:"
            }
            AppleOnlyCall::CompressedTextureReinterpretationFlush => {
                "insertCompressedTextureReinterpretationFlush"
            }
            AppleOnlyCall::SetProtectionOptions(_) => "setProtectionOptions:",
            AppleOnlyCall::SetAlphaTestReferenceValue(_) => "setAlphaTestReferenceValue:",
            AppleOnlyCall::SetLineWidth(_) => "setLineWidth:",
            AppleOnlyCall::SetPointSize(_) => "setPointSize:",
            AppleOnlyCall::SetProvokingVertexMode(_) => "setProvokingVertexMode:",
            AppleOnlyCall::SetViewportTransformEnabled(_) => "setViewportTransformEnabled:",
            AppleOnlyCall::SetVertexAmplificationMode { .. } => "setVertexAmplificationMode:value:",
            AppleOnlyCall::SetDepthCleared => "setDepthCleared",
            AppleOnlyCall::SetStencilCleared => "setStencilCleared",
        }
    }

    /// The runtime's answer: `Ok` where it takes the call (it states Metal's default, acts on
    /// state no runtime object has, so it records nothing, or sets the encoder's provoking
    /// vertex, [`RenderCommandEncoder::set_provoking_vertex_mode`]), else the refusal naming the
    /// missing feature.
    pub fn check(self) -> Result<(), AppleOnly> {
        match self {
            AppleOnlyCall::InvalidateCompressedTexture
            | AppleOnlyCall::InvalidateCompressedTextureSlice
            | AppleOnlyCall::CompressedTextureReinterpretationFlush
            | AppleOnlyCall::SetAlphaTestReferenceValue(_)
            | AppleOnlyCall::SetPointSize(_)
            | AppleOnlyCall::SetProvokingVertexMode(_) => Ok(()),
            AppleOnlyCall::SetProtectionOptions(options) => default_or(
                options == DEFAULT_PROTECTION_OPTIONS,
                AppleOnly::ProtectedContent,
            ),
            AppleOnlyCall::SetLineWidth(width) => default_or(width == 1.0, AppleOnly::LineWidth),
            AppleOnlyCall::SetViewportTransformEnabled(enabled) => {
                default_or(enabled, AppleOnly::ViewportTransformDisabled)
            }
            AppleOnlyCall::SetVertexAmplificationMode { .. } => {
                Err(AppleOnly::VertexAmplificationMode)
            }
            AppleOnlyCall::SetDepthCleared => Err(AppleOnly::DepthCleared),
            AppleOnlyCall::SetStencilCleared => Err(AppleOnly::StencilCleared),
        }
    }
}

/// `Ok` where the call states Metal's default, else the refusal of `feature`.
fn default_or(is_default: bool, feature: AppleOnly) -> Result<(), AppleOnly> {
    if is_default {
        Ok(())
    } else {
        Err(feature)
    }
}

/// The runtime error of `call`'s answer.
fn answer(call: AppleOnlyCall) -> Result<(), Error> {
    call.check().map_err(Error::AppleOnly)
}

impl CommandBuffer {
    /// `setProtectionOptions:` (SPI): accepted at Metal's default, no protection options
    /// ([`DEFAULT_PROTECTION_OPTIONS`]); any other value is refused
    /// ([`AppleOnly::ProtectedContent`]): an SPI's feature is the consumer's to build.
    pub fn set_protection_options(&self, options: u64) -> Result<(), Error> {
        answer(AppleOnlyCall::SetProtectionOptions(options))
    }
}

impl Texture {
    /// `protectionOptions` (SPI): [`DEFAULT_PROTECTION_OPTIONS`], the only protection options a
    /// runtime texture has.
    pub fn protection_options(&self) -> u64 {
        DEFAULT_PROTECTION_OPTIONS
    }
}

impl BlitCommandEncoder {
    /// `invalidateCompressedTexture:` (SPI): a no-op. Vulkan shows no texture compression, and
    /// after an invalidation Metal leaves the contents undefined, so keeping them is an outcome
    /// Metal allows. Refused (`Invalid`, `ForeignObject`) for another device's texture, as every
    /// blit is.
    pub fn invalidate_compressed_texture(&mut self, texture: &Texture) -> Result<(), Error> {
        self.log(|_| BlitCall::InvalidateCompressedTexture(texture.clone()));
        self.core()
            .check_device("invalidate_compressed_texture", texture.device())?;
        answer(AppleOnlyCall::InvalidateCompressedTexture)
    }

    /// `invalidateCompressedTexture:slice:level:` (SPI): a no-op, as
    /// [`invalidate_compressed_texture`](Self::invalidate_compressed_texture) is, for one slice
    /// and level. Refused (`Invalid`) also for a slice or level the texture does not have, as the
    /// blit encoder's hints of a slice and level are.
    pub fn invalidate_compressed_texture_slice(
        &mut self,
        texture: &Texture,
        slice: u64,
        level: u64,
    ) -> Result<(), Error> {
        self.log(|_| BlitCall::InvalidateCompressedTextureSlice(texture.clone(), slice, level));
        const CALL: &str = "invalidate_compressed_texture_slice";
        self.core().check_device(CALL, texture.device())?;
        check_slice_level(CALL, texture, slice, level)?;
        answer(AppleOnlyCall::InvalidateCompressedTextureSlice)
    }
}

impl ComputeCommandEncoder {
    /// `insertCompressedTextureReinterpretationFlush` (SPI): a no-op. Vulkan shows no texture
    /// compression, so there is nothing to flush.
    pub fn insert_compressed_texture_reinterpretation_flush(&mut self) {}
}

impl RenderLowering {
    /// The lowering of [`RenderCommandEncoder::set_alpha_test_reference_value`].
    pub(crate) fn set_alpha_test_reference_value(&mut self, value: f32) -> Result<(), Error> {
        self.log(|_| RenderCall::AlphaTestReferenceValue(value));
        answer(AppleOnlyCall::SetAlphaTestReferenceValue(value))
    }

    /// The lowering of [`RenderCommandEncoder::set_line_width`].
    pub(crate) fn set_line_width(&mut self, width: f32) -> Result<(), Error> {
        self.log(|_| RenderCall::LineWidth(width));
        answer(AppleOnlyCall::SetLineWidth(width))
    }

    /// The lowering of [`RenderCommandEncoder::set_point_size`].
    pub(crate) fn set_point_size(&mut self, size: f32) -> Result<(), Error> {
        self.log(|_| RenderCall::PointSize(size));
        answer(AppleOnlyCall::SetPointSize(size))
    }

    /// The lowering of [`RenderCommandEncoder::set_provoking_vertex_mode`].
    pub(crate) fn set_provoking_vertex_mode(&mut self, mode: u64) -> Result<(), Error> {
        self.log(|_| RenderCall::ProvokingVertexMode(mode));
        self.set_provoking_vertex(mode);
        Ok(())
    }

    /// The lowering of [`RenderCommandEncoder::set_viewport_transform_enabled`].
    pub(crate) fn set_viewport_transform_enabled(&mut self, enabled: bool) -> Result<(), Error> {
        self.log(|_| RenderCall::ViewportTransformEnabled(enabled));
        answer(AppleOnlyCall::SetViewportTransformEnabled(enabled))
    }

    /// The lowering of [`RenderCommandEncoder::set_vertex_amplification_mode`].
    pub(crate) fn set_vertex_amplification_mode(
        &mut self,
        mode: u64,
        value: u64,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::VertexAmplificationMode(mode, value));
        answer(AppleOnlyCall::SetVertexAmplificationMode { mode, value })
    }

    /// The lowering of [`RenderCommandEncoder::set_depth_cleared`].
    pub(crate) fn set_depth_cleared(&mut self) -> Result<(), Error> {
        self.log(|_| RenderCall::DepthCleared);
        answer(AppleOnlyCall::SetDepthCleared)
    }

    /// The lowering of [`RenderCommandEncoder::set_stencil_cleared`].
    pub(crate) fn set_stencil_cleared(&mut self) -> Result<(), Error> {
        self.log(|_| RenderCall::StencilCleared);
        answer(AppleOnlyCall::SetStencilCleared)
    }
}

impl RenderCommandEncoder {
    /// `setAlphaTestReferenceValue:` (SPI): a no-op. No pipeline the runtime builds has a
    /// fixed-function alpha test, so the reference has no effect, as on Metal with the test off.
    pub fn set_alpha_test_reference_value(&mut self, value: f32) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_alpha_test_reference_value(value)),
            false => self.defer(|_| RenderCall::AlphaTestReferenceValue(value)),
        }
    }

    /// `setLineWidth:` (SPI): accepted at Metal's default width of 1, which every line the
    /// runtime draws has; any other width is refused ([`AppleOnly::LineWidth`]): Apple's
    /// wide-line rule is unmeasured.
    pub fn set_line_width(&mut self, width: f32) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_line_width(width)),
            false => self.defer(|_| RenderCall::LineWidth(width)),
        }
    }

    /// `setPointSize:` (SPI): a no-op. On an M4 it changes nothing Metal draws at any size (0,
    /// 0.5, 1, 3, 5, 100, -2, NaN, infinity), whether the vertex function writes `[[point_size]]`
    /// or not and whether the pipeline's private `pointSizeOutputVS` is set.
    pub fn set_point_size(&mut self, size: f32) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_point_size(size)),
            false => self.defer(|_| RenderCall::PointSize(size)),
        }
    }

    /// `setProvokingVertexMode:` (SPI): the vertex the encoder's draws after the call take each
    /// primitive's flat inputs from, until the next call or the encoder's end, as Metal draws it
    /// (the M3 Max, macOS 27.0.1): mode 0, Metal's default, the first vertex; mode 1 the last, a
    /// triangle strip's odd triangle then in Vulkan's last-vertex order for its barycentric
    /// weights too. An execution of indirect render commands takes the first vertex whatever the
    /// mode, as on Metal. Mode 1 runs a pipeline of the last vertex where the device enables
    /// `VK_EXT_provoking_vertex`'s `provokingVertexLast` with `provokingVertexModePerPipeline`,
    /// and elsewhere draws each primitive whose vertex order a pipeline state reads from a list
    /// led by its last vertex, which a GPU pass writes (the CPU for a direct draw without
    /// indices); what that list draws otherwise than Metal is announced
    /// (`provoking-vertex-lines`, `provoking-vertex-unmatched`). Metal draws every other mode,
    /// its top bit set or not, a third way, which differs between pipeline states (every line's
    /// flat inputs and `[[primitive_id]]` 0), and the runtime as mode 1, announced at the call
    /// (`provoking-vertex-unmatched`). No mode is refused.
    pub fn set_provoking_vertex_mode(&mut self, mode: u64) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_provoking_vertex_mode(mode)),
            false => self.defer(|_| RenderCall::ProvokingVertexMode(mode)),
        }
    }

    /// `setViewportTransformEnabled:` (SPI): accepted on, Metal's default and every runtime
    /// draw's; off is refused ([`AppleOnly::ViewportTransformDisabled`]): Vulkan has no form for
    /// it.
    pub fn set_viewport_transform_enabled(&mut self, enabled: bool) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_viewport_transform_enabled(enabled)),
            false => self.defer(|_| RenderCall::ViewportTransformEnabled(enabled)),
        }
    }

    /// `setVertexAmplificationMode:value:` (SPI): refused
    /// ([`AppleOnly::VertexAmplificationMode`]), its meaning unmeasured.
    pub fn set_vertex_amplification_mode(&mut self, mode: u64, value: u64) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_vertex_amplification_mode(mode, value)),
            false => self.defer(|_| RenderCall::VertexAmplificationMode(mode, value)),
        }
    }

    /// `setDepthCleared` (SPI): refused ([`AppleOnly::DepthCleared`]), its meaning unmeasured.
    pub fn set_depth_cleared(&mut self) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_depth_cleared()),
            false => self.defer(|_| RenderCall::DepthCleared),
        }
    }

    /// `setStencilCleared` (SPI): refused ([`AppleOnly::StencilCleared`]), its meaning
    /// unmeasured.
    pub fn set_stencil_cleared(&mut self) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_stencil_cleared()),
            false => self.defer(|_| RenderCall::StencilCleared),
        }
    }
}
