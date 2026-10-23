//! What Metal writes into a pass's visibility-result buffer: the words a `Boolean` mode stores,
//! the samples a `Counting` mode counts, and what an offset nobody set keeps.
//!
//! Ported from the runtime lane `runtime_lane/visibility.rs`. Every pass here is one the lane
//! drives, drawn with the owned fixture `rt6_render_state`: the fullscreen triangle covers every
//! pixel of a 4 by 4 target, and the same draw **from vertex 3** covers none. Each visibility
//! buffer starts as `0xaa` bytes, so a word Metal never wrote is visible as the bytes that were
//! there rather than as a zero nobody can tell from a result.
//!
//! The lane's own table was measured on an M4, and its comment says so. Metal's answer here is
//! **run**, not copied: record mode takes it on this M3 Max, and this module's tests cross-check
//! the recordings against the lane's words, which is the evidence the port owes the lane. A fresh
//! answer that disagrees with the lane is a finding about one of the two chips, not a nuisance.
//!
//! The shapes, each its own case so a device that lacks precise occlusion queries loses only the
//! counted rows:
//!
//! - `Boolean` at a spread of offsets: drawn, a draw covering nothing, an offset set with no draw
//!   after it, `Disabled` at another offset, and two draws at one offset; then a second pass over
//!   the same buffer, which rewrites only its own offset.
//! - An offset taken again after another offset, in both orders.
//! - A draw behind the cleared depth, which passes no sample: the one shape here with a **depth
//!   attachment** and a depth/stencil state.
//! - An indirect draw of no instances.
//! - The same questions counted, and the fullscreen triangle's count at **4x**, which is the one
//!   shape here with a multisampled attachment.
//!
//! Nothing here can fault a GPU: one 4 by 4 target a pass, three vertices a draw, no loop.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, PassDepth, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    TargetSpec,
};
use crate::replay::NOTICES_SITE;
use crate::transcript::{ByteLayout, Digest, Element, Node, Observed, Sites, Value};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType};
use metal2vulkan_runtime::{Degradation, StoreAction, VisibilityResultMode};

/// The owned fixture the lane draws with: `rt6_full` takes a depth and a slope as one `float2` at
/// vertex buffer 0, and `rt6_white` returns white.
const FIXTURE: &str = "rt6_render_state";
const VERTEX: &str = "rt6_full";
const FRAGMENT: &str = "rt6_white";

/// The lane's target: a 4 by 4 `RGBA8Unorm` render target, whose 16 pixels are the 16 samples a
/// covering draw passes at 1x.
const SIDE: u32 = 4;
const COLOR: PixelFormat = PixelFormat::RGBA8Unorm;

/// The byte every visibility buffer starts as, so a word Metal never wrote is readable as itself.
const UNWRITTEN: u8 = 0xaa;

/// The depth the lane's draws take unless the shape says otherwise.
const DEPTH: f32 = 0.5;

/// One command a case encodes into a pass, in the lane's own vocabulary.
#[derive(Clone, Copy, Debug)]
enum Act {
    /// `setVisibilityResultMode:offset:`.
    Mode(VisibilityResultMode, u64),
    /// The fullscreen triangle at `z`, or the same draw from vertex 3, which covers no pixel, of
    /// `instances` instances.
    Draw {
        z: f32,
        covering: bool,
        instances: u32,
    },
    /// `drawPrimitives:indirectBuffer:indirectBufferOffset:` of three vertices and **no**
    /// instances, which the lane's arguments buffer holds.
    DrawIndirect,
}

impl Act {
    /// The fullscreen triangle at the lane's depth, covering every pixel.
    const fn draw() -> Self {
        Act::Draw {
            z: DEPTH,
            covering: true,
            instances: 1,
        }
    }

    /// The same draw from vertex 3, which covers no pixel.
    const fn draw_nothing() -> Self {
        Act::Draw {
            z: DEPTH,
            covering: false,
            instances: 1,
        }
    }
}

/// The depth state one shape takes: a compare function, with no depth write, over a depth
/// attachment of `format` cleared to `clear`.
#[derive(Clone, Copy, Debug)]
struct Depth {
    format: PixelFormat,
    clear: f64,
    compare: CompareFunction,
}

/// What one case asks: its name, its buffer's words, the passes it runs, its depth attachment and
/// its sample count.
struct Shape {
    name: &'static str,
    /// 64-bit words the case's visibility buffer holds.
    words: usize,
    /// The passes, in order, each over its own fresh color target and all over **one** visibility
    /// buffer, as the lane runs them.
    passes: &'static [&'static [Act]],
    depth: Option<Depth>,
    /// `rasterSampleCount` and the attachment's samples: 1, or 4 for the lane's 4x pass.
    samples: u32,
}

/// `Boolean`, `Counting` and `Disabled`, as short as the lane writes them.
const BOOL: VisibilityResultMode = VisibilityResultMode::Boolean;
const COUNT: VisibilityResultMode = VisibilityResultMode::Counting;
const OFF: VisibilityResultMode = VisibilityResultMode::Disabled;

/// Every case of the family, in the order the report lists them.
const SHAPES: &[Shape] = &[
    // Probe 1's visibility_modes: the Boolean offsets, then a second pass that rewrites only its
    // own offset.
    Shape {
        name: "visibility-boolean-offsets",
        words: 8,
        passes: &[
            &[
                Act::Mode(BOOL, 0),
                Act::draw(),
                Act::Mode(BOOL, 24),
                Act::draw_nothing(),
                Act::Mode(BOOL, 40),
                Act::Mode(OFF, 48),
                Act::draw(),
                Act::Mode(BOOL, 56),
                Act::draw(),
                Act::draw(),
            ],
            &[Act::Mode(BOOL, 8), Act::draw()],
        ],
        depth: None,
        samples: 1,
    },
    // Probe 5's vis_revisit_boolean_then_none: offset 0 taken again after offset 8, whose first
    // draw covered and whose second did not.
    Shape {
        name: "visibility-boolean-revisit-after-covering",
        words: 2,
        passes: &[&[
            Act::Mode(BOOL, 0),
            Act::draw(),
            Act::Mode(BOOL, 8),
            Act::draw_nothing(),
            Act::Mode(BOOL, 0),
            Act::draw_nothing(),
        ]],
        depth: None,
        samples: 1,
    },
    // Probe 5's vis_revisit_none_then_boolean: the same three draws the other way round.
    Shape {
        name: "visibility-boolean-revisit-after-nothing",
        words: 2,
        passes: &[&[
            Act::Mode(BOOL, 0),
            Act::draw_nothing(),
            Act::Mode(BOOL, 8),
            Act::draw(),
            Act::Mode(BOOL, 0),
            Act::draw(),
        ]],
        depth: None,
        samples: 1,
    },
    // Probe 1's visibility_occluded: a draw at 0.5 behind the cleared depth 0.25 under `Less`
    // passes no sample, and one at 0.125 in front of it passes every one.
    Shape {
        name: "visibility-boolean-occluded",
        words: 2,
        passes: &[&[
            Act::Mode(BOOL, 8),
            Act::draw(),
            Act::Mode(BOOL, 0),
            Act::Draw {
                z: 0.125,
                covering: true,
                instances: 1,
            },
        ]],
        depth: Some(Depth {
            format: PixelFormat::Depth32Float,
            clear: 0.25,
            compare: CompareFunction::Less,
        }),
        samples: 1,
    },
    // Probe 5's vis_indirect_no_instances, as Boolean: an indirect draw of three vertices and no
    // instances.
    Shape {
        name: "visibility-boolean-indirect-no-instances",
        words: 1,
        passes: &[&[Act::Mode(BOOL, 0), Act::DrawIndirect]],
        depth: None,
        samples: 1,
    },
    // Probe 1's counts at their offsets: drawn, a draw covering nothing, two draws, three
    // instances, set with no draw; beside a `Boolean` offset, and a draw after `Disabled`.
    Shape {
        name: "visibility-counting-offsets",
        words: 8,
        passes: &[&[
            Act::Mode(COUNT, 0),
            Act::draw(),
            Act::Mode(COUNT, 8),
            Act::draw_nothing(),
            Act::Mode(COUNT, 16),
            Act::draw(),
            Act::draw(),
            Act::Mode(COUNT, 24),
            Act::Draw {
                z: DEPTH,
                covering: true,
                instances: 3,
            },
            Act::Mode(COUNT, 32),
            Act::Mode(BOOL, 40),
            Act::draw(),
            Act::Mode(OFF, 0),
            Act::draw(),
        ]],
        depth: None,
        samples: 1,
    },
    // Probe 1's visibility_occluded, counted.
    Shape {
        name: "visibility-counting-occluded",
        words: 2,
        passes: &[&[
            Act::Mode(COUNT, 8),
            Act::draw(),
            Act::Mode(COUNT, 0),
            Act::Draw {
                z: 0.125,
                covering: true,
                instances: 1,
            },
        ]],
        depth: Some(Depth {
            format: PixelFormat::Depth32Float,
            clear: 0.25,
            compare: CompareFunction::Less,
        }),
        samples: 1,
    },
    // Probe 5's vis_revisit_counting: offset 0 taken again after offset 8 sums its draws.
    Shape {
        name: "visibility-counting-revisit",
        words: 2,
        passes: &[&[
            Act::Mode(COUNT, 0),
            Act::draw(),
            Act::Mode(COUNT, 8),
            Act::draw(),
            Act::Mode(COUNT, 0),
            Act::draw(),
        ]],
        depth: None,
        samples: 1,
    },
    // Probe 5's vis_revisit_after_disabled: the same, with `Disabled` between.
    Shape {
        name: "visibility-counting-revisit-after-disabled",
        words: 2,
        passes: &[&[
            Act::Mode(COUNT, 0),
            Act::draw(),
            Act::Mode(OFF, 0),
            Act::draw(),
            Act::Mode(COUNT, 0),
            Act::draw(),
        ]],
        depth: None,
        samples: 1,
    },
    // Probe 1's 4x pass: the fullscreen triangle passes every sample of the 4 by 4 target, which
    // is 64 at 4x. The attachment is multisampled and private, and the pass does not store it.
    Shape {
        name: "visibility-counting-four-samples",
        words: 1,
        passes: &[&[Act::Mode(COUNT, 0), Act::draw()]],
        depth: None,
        samples: 4,
    },
];

/// The arguments an indirect draw of three vertices and no instances reads:
/// `MTLDrawPrimitivesIndirectArguments`, four 32-bit words.
const NO_INSTANCES: [u32; 4] = [3, 0, 0, 0];

/// One case of the family.
pub struct Visibility {
    shape: &'static Shape,
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Visibility> {
    SHAPES.iter().map(|shape| Visibility { shape }).collect()
}

/// The degrade notices a visibility pass may announce. As with every family, it is a list rather
/// than a device query, because a case never branches on the device: it asks by running and reads
/// what came back. A venue that announces a kind not here fails the case until its kind is added.
const PASS_NOTICES: &[Degradation] = &[
    // A `Shared` buffer the device keeps in system memory, which the visibility buffer is.
    Degradation::SharedInSystemMemory,
    // The device's own compiler rewrote the stage's fast-math arithmetic as it made the pipeline,
    // which MoltenVK 1.4.2 on this M3 Max announces for this fixture's vertex stage.
    Degradation::FastMathRewrites,
];

/// The bytes a buffer of `words` 64-bit words starts as.
fn start(words: usize) -> Vec<u8> {
    vec![UNWRITTEN; 8 * words]
}

/// The eight bytes `rt6_full` reads at vertex buffer 0: the depth, then the slope, which every
/// shape here leaves at 0.
fn depth_bytes(z: f32) -> Vec<u8> {
    let mut bytes = z.to_le_bytes().to_vec();
    bytes.extend(0.0f32.to_le_bytes());
    bytes
}

/// The indirect arguments' bytes.
fn argument_bytes() -> Vec<u8> {
    NO_INSTANCES
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
}

/// Every depth a shape's draws take, once each, in the order they first appear: one vertex buffer
/// a depth is enough, since a buffer a draw reads is read at the draw.
fn depths(shape: &Shape) -> Vec<f32> {
    let mut found: Vec<f32> = Vec::new();
    let hold = |z: f32, found: &mut Vec<f32>| {
        if !found.iter().any(|held| held.to_bits() == z.to_bits()) {
            found.push(z);
        }
    };
    for pass in shape.passes {
        for act in *pass {
            match act {
                Act::Draw { z, .. } => hold(*z, &mut found),
                // An indirect draw reads the same vertex buffer a direct one does, at [`DEPTH`];
                // only its vertex and instance counts come from the arguments buffer.
                Act::DrawIndirect => hold(DEPTH, &mut found),
                Act::Mode(..) => {}
            }
        }
    }
    found
}

/// Whether any act of `shape` draws indirectly.
fn draws_indirect(shape: &Shape) -> bool {
    shape
        .passes
        .iter()
        .flat_map(|pass| pass.iter())
        .any(|act| matches!(act, Act::DrawIndirect))
}

impl Visibility {
    /// The fixture the case runs.
    fn fixture(&self) -> Fixture {
        Fixture::new(FIXTURE)
    }

    /// The one color state the pipeline holds.
    fn color_state(&self) -> [Option<ColorState>; 1] {
        [Some(ColorState::new(COLOR))]
    }

    /// The pipeline the case makes: the fullscreen triangle and white, over the color attachment,
    /// at the shape's sample count, naming the depth format where the shape has one.
    fn pipeline_spec<'a>(&self, colors: &'a [Option<ColorState>]) -> RenderPipelineSpec<'a> {
        RenderPipelineSpec::new(VERTEX, FRAGMENT, colors)
            .multisampled(self.shape.samples)
            .depth_stencil(self.shape.depth.map(|depth| depth.format), None)
    }

    /// The depth and stencil state the shape takes: its compare function, no depth write, and
    /// Metal's default stencil on both faces.
    fn depth_state(&self) -> Option<DepthStencilDescriptor> {
        self.shape.depth.map(|depth| {
            let mut state = DepthStencilDescriptor::default();
            state.depth_compare_function = depth.compare;
            state.depth_write_enabled = false;
            state
        })
    }

    /// The target the shape's passes render into: a 4 by 4 `RGBA8Unorm` render target, shared and
    /// stored at 1x, private and discarded at 4x, because Metal stores no multisampled texture a
    /// host reads.
    fn target_spec(&self) -> TargetSpec {
        let spec = TargetSpec::plain(COLOR, [SIDE, SIDE]);
        match self.shape.samples {
            1 => spec.shared(),
            samples => spec.multisampled(samples),
        }
    }
}

impl Case for Visibility {
    fn name(&self) -> &'static str {
        self.shape.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The notices site alone: a visibility result is a number Metal wrote, and a number that
        // differs is a bug, not a near-equivalent. Nothing else here is loosened.
        PASS_NOTICES
            .iter()
            .map(|kind| {
                Allowance::new(
                    format!("case/{NOTICES_SITE}/*"),
                    Loosening::Unchecked,
                    Justification::notice(*kind),
                )
            })
            .collect()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.fixture();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{}{extension}", fixture.name),
                    entry: format!("{VERTEX}+{FRAGMENT}"),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        let colors = self.color_state();
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("render-pipeline", &self.pipeline_spec(&colors).words()),
            Input::bytes("target", &self.target_spec().words()),
            Input::bytes("visibility", &start(self.shape.words)),
            Input::bytes("acts", format!("{:?}", self.shape.passes).as_bytes()),
        ];
        if let Some(state) = self.depth_state() {
            inputs.push(Input::bytes(
                "depth-stencil-state",
                format!("{state:?}").as_bytes(),
            ));
        }
        if let Some(depth) = self.shape.depth {
            inputs.push(Input::bytes(
                "depth-attachment",
                format!("{:?} cleared to {}", depth.format, depth.clear).as_bytes(),
            ));
        }
        for z in depths(self.shape) {
            inputs.push(Input::bytes(
                format!("buffer:0@{}", z.to_bits()),
                &depth_bytes(z),
            ));
        }
        if draws_indirect(self.shape) {
            inputs.push(Input::bytes("arguments", &argument_bytes()));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = self.fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = crate::case::made(sites, "library", side.library(&loaded.source()))?;
        let colors = self.color_state();
        let state = crate::case::made(
            sites,
            "render-pipeline",
            side.render_pipeline(library, &self.pipeline_spec(&colors)),
        )?;
        let depth_state = match self.depth_state() {
            Some(spec) => Some(crate::case::made(
                sites,
                "depth-stencil-state",
                side.depth_stencil_state(&spec),
            )?),
            None => None,
        };
        // One vertex buffer a depth, one indirect-arguments buffer where a shape draws indirectly,
        // and one visibility buffer every pass of the case writes into.
        let mut vertex: Vec<(u32, Handle)> = Vec::new();
        for z in depths(self.shape) {
            let handle = crate::case::made(
                sites,
                &format!("buffer:0@{}", z.to_bits()),
                side.buffer(&depth_bytes(z)),
            )?;
            vertex.push((z.to_bits(), handle));
        }
        let arguments = match draws_indirect(self.shape) {
            true => Some(crate::case::made(
                sites,
                "arguments",
                side.buffer(&argument_bytes()),
            )?),
            false => None,
        };
        let visibility =
            crate::case::made(sites, "visibility", side.buffer(&start(self.shape.words)))?;
        let length = 8 * self.shape.words;
        for (index, acts) in self.shape.passes.iter().enumerate() {
            let at = format!("pass-{index}");
            let stop = |at: String, failure: Failure| Abort { at, failure };
            // A fresh target a pass, as the lane makes one a pass, so no pass reads what another
            // left in the color attachment.
            let target = crate::case::made(
                sites,
                &format!("{at}/target"),
                side.target(&self.target_spec()),
            )?;
            let mut steps = vec![RenderStep::Pipeline(state)];
            if let Some(held) = depth_state {
                steps.push(RenderStep::DepthStencilState(held));
            }
            for act in *acts {
                match act {
                    Act::Mode(mode, offset) => steps.push(RenderStep::VisibilityResultMode {
                        mode: *mode,
                        offset: *offset,
                    }),
                    Act::Draw {
                        z,
                        covering,
                        instances,
                    } => {
                        let held = vertex
                            .iter()
                            .find(|(bits, _)| *bits == z.to_bits())
                            .map(|(_, handle)| *handle)
                            .ok_or_else(|| {
                                stop(at.clone(), Failure::named("NoDepthBuffer", format!("{z}")))
                            })?;
                        steps.push(RenderStep::vertex_buffer(held, 0));
                        steps.push(RenderStep::Draw {
                            primitive: PrimitiveType::Triangle,
                            first: if *covering { 0 } else { 3 },
                            count: 3,
                            instances: *instances,
                        });
                    }
                    Act::DrawIndirect => {
                        let held = arguments.ok_or_else(|| {
                            stop(
                                at.clone(),
                                Failure::named("NoArguments", "an indirect draw"),
                            )
                        })?;
                        let depth = vertex
                            .iter()
                            .find(|(bits, _)| *bits == DEPTH.to_bits())
                            .map(|(_, handle)| *handle)
                            .ok_or_else(|| {
                                stop(
                                    at.clone(),
                                    Failure::named("NoDepthBuffer", "an indirect draw"),
                                )
                            })?;
                        steps.push(RenderStep::vertex_buffer(depth, 0));
                        steps.push(RenderStep::DrawIndirect {
                            primitive: PrimitiveType::Triangle,
                            arguments: held,
                            offset: 0,
                        });
                    }
                }
            }
            // At 4x the attachment is private and the pass does not store it: the question is the
            // samples the draws passed, not the texels.
            let color = match self.shape.samples {
                1 => PassColor::cleared(target, [0.0; 4]),
                _ => PassColor::cleared(target, [0.0; 4]).stored(StoreAction::DontCare),
            };
            let mut pass = RenderPass::clearing(vec![color])
                .drawing(steps.clone())
                .counting_into(visibility);
            if let Some(depth) = self.shape.depth {
                let attachment = crate::case::made(
                    sites,
                    &format!("{at}/depth"),
                    side.target(&TargetSpec::plain(depth.format, [SIDE, SIDE])),
                )?;
                pass = pass.with_depth(
                    PassDepth::cleared(attachment, depth.clear).stored(StoreAction::DontCare),
                );
            }
            let outcome = side
                .render(&[RenderWork::Pass(pass)])
                .map_err(|failure| stop(format!("{at}/render"), failure))?;
            let mut node = Node::branch(at.clone(), Vec::new());
            let mut calls = Node::branch("calls", Vec::new());
            for (piece, refused) in outcome.refused.iter().enumerate() {
                calls = calls.with(Node::leaf(
                    format!("{piece}-render"),
                    Observed::value(match refused {
                        Some(failure) => failure.value.clone(),
                        None => Value::ok(Value::Absent),
                    }),
                ));
            }
            node = node.with(calls);
            if let Some(failure) = outcome.refused.iter().flatten().next() {
                sites.push(node);
                return Err(Abort {
                    at: format!("{at}/calls/0-render"),
                    failure: failure.clone(),
                });
            }
            node = node.with(Node::leaf(
                "status",
                Observed::value(Value::ok(outcome.status.clone())),
            ));
            node = node.with(Node::leaf(
                "command-buffer-error",
                Observed::value(match &outcome.error {
                    Some(error) => Value::Variant {
                        name: "Some".to_owned(),
                        payload: Box::new(error.clone()),
                    },
                    None => Value::unit_variant("None"),
                }),
            ));
            node = node.with(Node::leaf(
                "words",
                Observed::bytes(
                    side.read_buffer(visibility, 0, length)
                        .map_err(|failure| stop(format!("{at}/words"), failure))?,
                    ByteLayout::buffer(0, length as u64, Element::U64),
                ),
            ));
            sites.push(node);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! The cross-check the port owes the lane: the words Metal wrote, as the recordings hold them,
    //! against the lane's table.
    //!
    //! The table is the lane's (`runtime_lane/visibility.rs`), restated here because a conformance
    //! case may not depend on the validation crate. Its comment says it was measured on an **M4**;
    //! this recording is of an M3 Max. A disagreement is therefore a fact about one of the two
    //! chips, and the test names the case and the word.

    use super::*;
    use crate::recording::Recording;
    use crate::transcript::Transcript;

    /// A word Metal never wrote, as a 64-bit value.
    const U: u64 = u64::from_le_bytes([UNWRITTEN; 8]);

    /// The lane's words, by case: what Metal wrote after each pass, in pass order.
    const TABLE: &[(&str, &[&[u64]])] = &[
        (
            "visibility-boolean-offsets",
            &[&[1, U, U, 0, U, 0, U, 1], &[1, 1, U, 0, U, 0, U, 1]],
        ),
        ("visibility-boolean-revisit-after-covering", &[&[1, 0]]),
        ("visibility-boolean-revisit-after-nothing", &[&[1, 1]]),
        ("visibility-boolean-occluded", &[&[1, 0]]),
        ("visibility-boolean-indirect-no-instances", &[&[0]]),
        (
            "visibility-counting-offsets",
            &[&[16, 0, 32, 48, 0, 1, U, U]],
        ),
        ("visibility-counting-occluded", &[&[16, 0]]),
        ("visibility-counting-revisit", &[&[32, 16]]),
        ("visibility-counting-revisit-after-disabled", &[&[32, U]]),
        ("visibility-counting-four-samples", &[&[64]]),
    ];

    /// The one outcome of `name`'s recorded Metal answer.
    fn only_outcome(name: &str) -> Transcript {
        let recording: Recording = crate::recording::read(name)
            .unwrap_or_else(|why| panic!("{name}: no recording ({why})"));
        let answer = recording
            .answers
            .first()
            .unwrap_or_else(|| panic!("{name}: no answer"));
        assert_eq!(answer.outcomes.len(), 1, "{name}: more than one outcome");
        answer.outcomes[0].transcript.clone()
    }

    /// The 64-bit words the site at `path` of `case`'s `transcript` holds.
    fn words(case: &str, transcript: &Transcript, path: &str) -> Vec<u64> {
        let mut node = &transcript.root;
        for name in path.trim_start_matches("case/").split('/') {
            node = node
                .child(name)
                .unwrap_or_else(|| panic!("{case}: no site {path} (stopped at {name})"));
        }
        let Value::Bytes(bytes) = &node.observed.value else {
            panic!("{case}: {path} holds no bytes");
        };
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|word| u64::from_le_bytes(*word))
            .collect()
    }

    /// Metal's own visibility results on this M3 Max are the lane's words, pass by pass.
    ///
    /// Every case the table names must have a recording, so a case that was never recorded fails
    /// here rather than being skipped: the table is the claim, and a claim with no run behind it
    /// is the thing this suite exists to stop.
    #[test]
    fn the_recorded_results_are_the_lanes_words() {
        for (name, passes) in TABLE {
            assert!(
                SHAPES.iter().any(|shape| shape.name == *name),
                "{name}: the table names a case the family does not hold"
            );
            let recorded = only_outcome(name);
            for (index, want) in passes.iter().enumerate() {
                assert_eq!(
                    words(name, &recorded, &format!("case/pass-{index}/words")),
                    **want,
                    "{name}: pass {index}"
                );
            }
        }
    }

    /// Every case of the family is in the table, states its inputs, and loosens **only** its
    /// notices site.
    ///
    /// A visibility result is a number Metal wrote; a number that differs is a bug, never a
    /// near-equivalent. A case that ever loosened its words would fail here.
    #[test]
    fn every_case_is_tabled_and_loosens_its_notices_alone() {
        for case in all() {
            assert!(
                TABLE.iter().any(|(name, _)| *name == case.name()),
                "{}: no row in the lane's table",
                case.name()
            );
            let inputs = case.inputs().unwrap();
            assert!(
                inputs.iter().any(|input| input.name == "visibility"),
                "{}: no visibility buffer in its inputs",
                case.name()
            );
            assert_eq!(case.metal_source(), MetalSource::Run);
            case.shaders().unwrap();
            for allowance in case.allowances() {
                assert_eq!(
                    allowance.path,
                    format!("case/{NOTICES_SITE}/*"),
                    "{}: an allowance outside the notices site",
                    case.name()
                );
                match &allowance.justified_by {
                    Justification::Degradation(code) => assert!(
                        PASS_NOTICES.iter().any(|kind| kind.code() == code),
                        "{}: {code} is not a visibility-pass notice",
                        case.name()
                    ),
                    other => panic!("{}: an unexpected tolerance: {other:?}", case.name()),
                }
            }
        }
    }
}
