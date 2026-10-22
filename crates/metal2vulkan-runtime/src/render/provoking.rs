//! The provoking vertex: Apple's `setProvokingVertexMode:`.
//!
//! Measured on the M3 Max with macOS 27.0.1 (each primitive's flat inputs, `[[primitive_id]]` and
//! barycentric weights, solved for its vertices):
//!
//! - Mode 0, and every value whose top bit is set, takes a primitive's flat inputs from its first
//!   vertex; mode 1 from its last. The values from 2 to `2^63 - 1` draw a third way (a triangle
//!   from its last vertex, an odd strip triangle `(i + 2, i + 1, i)` from `i`, every line's flat
//!   inputs and `[[primitive_id]]` 0), which the runtime draws as mode 1, announcing it at the
//!   call (`Degradation::ProvokingVertexUnmatched`). No value is a validation error.
//! - At mode 1 the vertex is a line list's `2j + 1`, a line strip's `j + 1`, a triangle list's
//!   `3j + 2`, a triangle strip's `j + 2`, and Metal's private types' list primitives' last
//!   (`DrawPrimitiveType`). The barycentric order is mode 0's but for a triangle strip's odd
//!   triangle, `(j + 1, j, j + 2)` where mode 0 gives `(j, j + 2, j + 1)`: the Vulkan
//!   specification's last-vertex strip order. Indexed draws (16- and 32-bit, a base vertex,
//!   restarts, each piece's parity its own), indirect draws, instances, tessellated patches and
//!   mesh draws follow the mode; an execution of indirect render commands does not, its
//!   primitives taking the first vertex.
//! - The mode is the encoder's: the last call wins, before or after its pipeline state is set,
//!   and a new encoder starts at the first vertex.
//!
//! A draw at the first vertex, and one of a pipeline whose primitives' vertex order nothing reads
//! (its fragment function reads no flat input and no `[[barycentric_coord]]`, and its vertex
//! function selects no slice or viewport), draws as it is ([`Provoking::AsIs`]). Otherwise:
//!
//! - **Native**: on a device that enables `VK_EXT_provoking_vertex`'s `provokingVertexLast` and
//!   whose pipelines of one render pass instance may differ in mode
//!   (`provokingVertexModePerPipeline`), the draw's `VkPipeline` takes the last vertex
//!   (`FixedState::provoking_last`), whose primitive topologies' provoking vertices and
//!   barycentric orders are Metal's mode 1 (the Vulkan specification's primitive topologies and
//!   barycentric interpolation). A list the runtime draws for a draw (a strip restarting at
//!   another index than Vulkan's, a private type's) is written in Metal's mode-1 order
//!   ([`RewriteOrder::Last`]). Announced (`Degradation::ProvokingVertexUnmatched`): a strip whose
//!   vertex function computes `[[primitive_id]]` into its varying (a device without
//!   `geometryShader` and `tessellationShader`), whose IDs then read its primitives' last
//!   vertices; and a slice or viewport the vertex function selects on a device that selects it in
//!   a geometry stage (without `shaderOutputLayer` or `shaderOutputViewportIndex`), from each
//!   primitive's first vertex. Tessellated patches take the vertex Vulkan leaves to the device
//!   ("the provoking vertex may be any of the vertices in each primitive", the Vulkan
//!   specification's Flat Shading), which `PrecisionNote::TessellationTriangleOrder` covers.
//! - **Listed**, on any other device: the draw is drawn as the list the index rewrite pass writes
//!   (the CPU for a direct draw without indices, `index_rewrite.rs`), each primitive in Metal's
//!   mode-1 order with its last vertex moved first ([`RewriteOrder::LastFirst`]), so the
//!   pipeline's first vertex is Metal's: exact for a triangle's flat inputs, slice and viewport
//!   (a rotated triangle covers the same pixels and interpolates its smooth inputs alike, bit for
//!   bit, over 200 perspective triangles on the M3 Max). A line drawn from its last vertex to its
//!   first covers other pixels and interpolates otherwise (206 pixels and 2922 interpolated
//!   values of 200 lines on the M3 Max), announced (`Degradation::ProvokingVertexLines`). A strip
//!   whose fragment function reads barycentrics and no flat input is listed in Metal's mode-1
//!   order instead ([`RewriteOrder::Last`]), exact; a list, a line or a private type of such a
//!   pipeline draws as it is, its order mode 0's. What the list cannot draw as Metal does is
//!   announced (`Degradation::ProvokingVertexUnmatched`): a triangle whose fragment function reads
//!   flat inputs and barycentrics (its weights in the rotated order), and patches and mesh draws,
//!   whose primitives no list holds (their flat inputs from the first vertex).
//!
//! An execution's commands take the first vertex: its snapshot of the encoder's state does
//! (`render/executions.rs`).

use super::{Assembly, RenderLowering};
use crate::degradation::Degradation;
use crate::pipeline::GraphicsState as DrawingState;
use crate::render_state::EncoderState;
use crate::Device;
use metal2vulkan::bind::RewriteOrder;
use metal2vulkan::device::{VkFeature, VkProperty};
use metal2vulkan::metal::PrimitiveType;

/// How a draw takes the encoder's provoking vertex (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Provoking {
    /// As it is: at the first vertex, or nothing reads its primitives' vertex order.
    AsIs,
    /// Through a `VkPipeline` whose provoking vertex is the last.
    Native,
    /// As the list of its primitives in `RewriteOrder`, through a first-vertex pipeline.
    Listed(RewriteOrder),
}

impl Provoking {
    /// The order a list the runtime draws for the draw is written in: Metal's mode-1 order under
    /// a last-vertex pipeline, the listed order, else mode 0's.
    pub(super) fn order(self) -> RewriteOrder {
        match self {
            Provoking::AsIs => RewriteOrder::First,
            Provoking::Native => RewriteOrder::Last,
            Provoking::Listed(order) => order,
        }
    }
}

/// How Metal draws `setProvokingVertexMode:`'s `mode` (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// The first vertex: 0, and every value whose top bit is set.
    First,
    /// The last vertex: 1.
    Last,
    /// The third way: the values from 2 to `2^63 - 1`.
    Third,
}

impl Mode {
    fn of(mode: u64) -> Mode {
        match mode {
            0 => Mode::First,
            1 => Mode::Last,
            m if m >> 63 == 1 => Mode::First,
            _ => Mode::Third,
        }
    }
}

/// What of a draw of `pipeline` reads its primitives' provoking vertex and vertex order: a flat
/// input, or a slice or viewport its vertex function selects; and the barycentric weights.
fn reads(pipeline: &DrawingState) -> (bool, bool) {
    let graphics = &pipeline.base().graphics;
    let flat = graphics.flat_fragment_inputs
        || pipeline.vertex_chooses_slice()
        || graphics.writes_viewport_index;
    (flat, graphics.reads_barycentric_coord)
}

/// Whether `device` draws the last provoking vertex natively: it enables
/// `VK_EXT_provoking_vertex`'s `provokingVertexLast`, and its pipelines of one render pass
/// instance may differ in mode (`provokingVertexModePerPipeline`; where they may not, "all
/// pipelines bound within a render pass instance must have the same provokingVertexMode", the
/// Vulkan specification's Flat Shading, and an execution's commands, which take the first vertex,
/// share the encoder's instances).
fn native(device: &Device) -> bool {
    let enabled = device.compile_device().profile().enabled();
    enabled.features.contains(&VkFeature::ProvokingVertexLast)
        && enabled
            .properties
            .contains(&VkProperty::ProvokingVertexModePerPipeline)
}

/// Whether a draw of `pipeline` under `state` runs a `VkPipeline` of the last provoking vertex
/// ([`Provoking::Native`]).
pub(super) fn natively_last(
    device: &Device,
    state: &EncoderState,
    pipeline: &DrawingState,
) -> bool {
    let (flat, weights) = reads(pipeline);
    state.provoking_last() && (flat || weights) && native(device)
}

/// The classes of primitives a draw assembles, as the provoking vertex treats them.
enum Class {
    Points,
    Lines,
    Triangles {
        strip: bool,
    },
    /// Patches and meshes: primitives no list the runtime writes holds.
    Unlisted,
}

impl Class {
    fn of(assembly: Assembly) -> Class {
        use metal2vulkan::bind::RewriteProgram as R;
        match assembly {
            Assembly::Primitives(PrimitiveType::Point) => Class::Points,
            Assembly::Primitives(PrimitiveType::Line | PrimitiveType::LineStrip)
            | Assembly::Private(R::LineLoop) => Class::Lines,
            Assembly::Primitives(PrimitiveType::TriangleStrip) => Class::Triangles { strip: true },
            Assembly::Primitives(PrimitiveType::Triangle) | Assembly::Private(_) => {
                Class::Triangles { strip: false }
            }
            Assembly::Patches | Assembly::Meshes => Class::Unlisted,
        }
    }
}

impl RenderLowering {
    /// The lowering of `setProvokingVertexMode:`'s `mode`: the encoder's draws from here take
    /// the first vertex or the last, the third way drawn as the last and announced (the module
    /// docs).
    pub(crate) fn set_provoking_vertex(&mut self, mode: u64) {
        let mode = Mode::of(mode);
        if mode == Mode::Third {
            self.core
                .device()
                .degrade(Degradation::ProvokingVertexUnmatched, || {
                    (
                        "setProvokingVertexMode: of a mode from 2 to 2^63 - 1".to_owned(),
                        "Metal draws an odd triangle strip triangle (i + 2, i + 1, i) from vertex \
                         i, and every line's flat inputs and [[primitive_id]] as 0; the runtime \
                         draws the mode as mode 1"
                            .to_owned(),
                    )
                });
        }
        let last = mode != Mode::First;
        // Only a change takes the state to a new version (`versioned.rs`), so the draws' memo
        // holds across a call that repeats the mode.
        if self.state.provoking_last() != last {
            self.state.set_provoking_last(last);
        }
    }

    /// How `call`'s draw of `pipeline` as `assembly` takes the encoder's provoking vertex (the
    /// module docs), announcing what a listed draw cannot draw as Metal does.
    pub(super) fn provoking(
        &self,
        call: &'static str,
        pipeline: &DrawingState,
        assembly: Assembly,
    ) -> Provoking {
        let (flat, weights) = reads(pipeline);
        if !self.state.provoking_last() || !(flat || weights) {
            return Provoking::AsIs;
        }
        let device = self.core.device();
        if native(device) {
            // A strip's primitive ID computed into a varying reads its last vertex's stream
            // position there, past Metal's count by the strip's vertices less one
            // (`LoweringInputs::primitive_id_varying`).
            let strip = matches!(
                assembly,
                Assembly::Primitives(PrimitiveType::LineStrip | PrimitiveType::TriangleStrip)
            );
            if strip && super::writes_primitive_ids(pipeline) {
                device.degrade(Degradation::ProvokingVertexUnmatched, || {
                    (
                        format!(
                            "{call}: a draw of a strip under setProvokingVertexMode: 1 whose \
                             vertex function computes [[primitive_id]] into its varying"
                        ),
                        "the varying reads each primitive's last vertex, so the IDs run one \
                         or two past Metal's"
                            .to_owned(),
                    )
                });
            }
            // A slice or viewport the synthesized geometry stage selects is its primitive's first
            // vertex's (`link::geometry`) on a device that cannot store it from a vertex function.
            let features = &device.compile_device().profile().enabled().features;
            let geometry_slice = pipeline.vertex_chooses_slice()
                && !features.contains(&VkFeature::ShaderOutputLayer);
            let geometry_viewport = pipeline.base().graphics.writes_viewport_index
                && !features.contains(&VkFeature::ShaderOutputViewportIndex);
            if geometry_slice || geometry_viewport {
                device.degrade(Degradation::ProvokingVertexUnmatched, || {
                    (
                        format!(
                            "{call}: a draw under setProvokingVertexMode: 1 whose vertex function \
                             selects its primitives' slice or viewport"
                        ),
                        "the device selects it in a geometry stage, from each primitive's first \
                         vertex"
                            .to_owned(),
                    )
                });
            }
            return Provoking::Native;
        }
        match (Class::of(assembly), flat) {
            (Class::Points, _) => Provoking::AsIs,
            (Class::Lines, true) => {
                device.degrade(Degradation::ProvokingVertexLines, || {
                    (
                        format!(
                            "{call}: a draw of lines under setProvokingVertexMode: 1 whose \
                             pipeline state reads their provoking vertex"
                        ),
                        "the device has no last provoking vertex \
                         (VK_EXT_provoking_vertex's provokingVertexLast, with \
                         provokingVertexModePerPipeline), so each line is drawn from its last \
                         vertex to its first"
                            .to_owned(),
                    )
                });
                Provoking::Listed(RewriteOrder::LastFirst)
            }
            // A line's weights, and a list's, are in mode 0's order at mode 1 too.
            (Class::Lines | Class::Triangles { strip: false }, false) => Provoking::AsIs,
            (Class::Triangles { strip: true }, false) => Provoking::Listed(RewriteOrder::Last),
            (Class::Triangles { .. }, true) => {
                if weights {
                    device.degrade(Degradation::ProvokingVertexUnmatched, || {
                        (
                            format!(
                                "{call}: a draw of triangles under setProvokingVertexMode: 1 \
                                 whose fragment function reads flat inputs and \
                                 [[barycentric_coord]]"
                            ),
                            "the device has no last provoking vertex, so each triangle is \
                             drawn from its last vertex, and its weights are in that order"
                                .to_owned(),
                        )
                    });
                }
                Provoking::Listed(RewriteOrder::LastFirst)
            }
            (Class::Unlisted, _) => {
                device.degrade(Degradation::ProvokingVertexUnmatched, || {
                    (
                        format!(
                            "{call}: a draw of patches or meshes under setProvokingVertexMode: 1 \
                             whose pipeline state reads their provoking vertex"
                        ),
                        "the device has no last provoking vertex, and no list holds the \
                         primitives, so they take the first vertex"
                            .to_owned(),
                    )
                });
                Provoking::AsIs
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Mode;

    /// The values Metal takes: 0 and every value whose top bit is set the first vertex, 1 the
    /// last, the values between the third way (Metal probes on the M3 Max, macOS 27.0.1).
    #[test]
    fn the_modes_are_metal_s_measured_ones() {
        for mode in [0, 1 << 63, u64::MAX, u64::MAX - 1] {
            assert_eq!(Mode::of(mode), Mode::First, "{mode:#x}");
        }
        assert_eq!(Mode::of(1), Mode::Last);
        for mode in [
            2,
            3,
            5,
            0xff,
            0x8000_0000,
            0xffff_ffff,
            1 << 32,
            i64::MAX as u64,
        ] {
            assert_eq!(Mode::of(mode), Mode::Third, "{mode:#x}");
        }
    }
}
