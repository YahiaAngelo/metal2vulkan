//! A render encoder's **barriers and residency calls**: the memory barrier a later draw needs to
//! see an earlier draw's texture writes, and the `useResource` and `useHeap` calls that declare
//! what its draws may reach.
//!
//! | case | what it asks |
//! |---|---|
//! | `render-state-render-barriers` | whether a draw sees the texels a draw before it in the same pass wrote, across `memoryBarrierWithScope:afterStages:beforeStages:` and `memoryBarrierWithResources:count:afterStages:beforeStages:`, with the four `useResource` forms around them |
//! | `render-state-heap-residency` | whether the four `useHeap` forms are answered on an empty heap, and that the pass's own draw is untouched by them |
//!
//! **The barrier case is the one with teeth, and the round-trip is inside one command buffer.**
//! The first draw's fragment function writes a texel of a scratch texture under every pixel it
//! covers and returns opaque black; the second draw's reads that texel and returns it. Both are in
//! one pass of one command buffer, with the barrier between them, and the texels the host reads
//! back are the second draw's. What the first draw wrote is arithmetic this module states — the
//! fixture writes `(x / 8, y / 8, 0.75, 1)` at the pixel's own texel — so the answer is a value the
//! case names, not whatever the scratch texture happened to hold. A barrier that did not order the
//! two draws leaves the second reading a texture the first had not finished writing, and the
//! difference is every texel.
//!
//! **What the residency calls claim here is narrow, and the module says so rather than implying
//! more.** Every texture and buffer these passes reach is bound at a slot, which is the one way of
//! reaching a resource Metal needs no residency declaration for. So the claim is that **Metal
//! answers each of the eight calls** — `useResource:usage:` and `useResources:count:usage:` with
//! and without stages, `useHeap:` and `useHeaps:count:` with and without stages — and that the
//! pass's own texels are what they are without them. It is **not** a claim that residency makes a
//! resource reachable: no function here reads one through an address or a resource ID held in
//! memory, so nothing here could show that. A case that could needs a fragment function that reads
//! an argument buffer, and the fixtures hold none whose stage input a committed vertex function
//! produces; that is a packet of its own.
//!
//! The heap is a sparse heap holding nothing, because that is the one heap the suite's vocabulary
//! can make. A venue that cannot make one refuses the call typed, which is the venue's own gap and
//! not ours, and it costs this one case rather than the barrier case beside it — which is why the
//! two are not one case.
//!
//! Nothing here can fault a GPU: one pass a case, one or two draws of six vertices over a 16 by 16
//! or 4 by 4 target, no loop in the fixture, and a heap that holds no memory until a mapping takes
//! a tile, which nothing here asks for.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side, TextureRole, TextureSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, RenderPass, RenderPipelineSpec, RenderStep, RenderWork, StepResource,
    TargetSpec,
};
use crate::sparse_query::SparseHeapSpec;
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{BarrierScope, PixelFormat, PrimitiveType, RenderStages, ResourceUsage};

/// The format both targets and the scratch texture carry: a `float` stored as it is, so the whole
/// expectation is arithmetic.
const FORMAT: PixelFormat = PixelFormat::RGBA32Float;

/// Bytes one texel of [`FORMAT`] takes.
const TEXEL: u32 = 16;

/// The side of the barrier case's target and scratch texture, in texels.
const SIDE: u32 = 16;

/// The side of the heap case's target.
const SMALL: u32 = 4;

/// The clear a pass loads its target with, which no draw of either case can store.
const CLEAR: f32 = -7.5;

/// The owned fixture both cases draw with.
const FIXTURE: &str = "tob_renders";

/// `tob_rect`: a rectangle from a buffer's two clip-space corners, six vertices.
const RECT: &str = "tob_rect";

/// `tob_solid`: the `float4` a fragment buffer holds, unchanged.
const SOLID: &str = "tob_solid";

/// `tob_write`: writes `(x / 8, y / 8, 0.75, 1)` into texture 0's texel under the pixel and
/// returns opaque black.
const WRITE: &str = "tob_write";

/// `tob_read`: the texel of texture 0 under the pixel.
const READ: &str = "tob_read";

/// The entry points a case's shader reference names.
const ENTRIES: &str = "tob_rect+tob_solid+tob_write+tob_read";

/// The draw a rectangle takes: two triangles of six vertices, one instance.
const RECT_DRAW: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 6,
    instances: 1,
};

/// The rectangle covering the whole target.
const WHOLE: [f32; 4] = [-1.0, -1.0, 1.0, 1.0];

/// The rectangle as the vertex function's `constant float4 &` binding takes it.
fn rect_bytes() -> Vec<u8> {
    WHOLE.iter().flat_map(|edge| edge.to_le_bytes()).collect()
}

/// The colour the heap case's draw stores, which its clear is not.
const COLOR: [f32; 4] = [0.25, 0.5, 0.75, 1.0];

/// The colour as the fragment function's binding takes it.
fn color_bytes() -> Vec<u8> {
    COLOR
        .iter()
        .flat_map(|channel| channel.to_le_bytes())
        .collect()
}

/// The scratch texture the barrier case's first draw writes and its second reads: of [`FORMAT`] at
/// the target's extent, in both shader roles, and nothing written into it by the host, so what the
/// second draw reads can only have come from the first draw.
fn scratch_spec() -> TextureSpec {
    TextureSpec::new(FORMAT, [SIDE; 2], TEXEL * SIDE, TextureRole::ReadWrite)
}

/// The words the scratch texture's own descriptor fingerprints as.
///
/// A case usually fingerprints a texture by the texels it hands in, and this one hands in none —
/// that is the point of it — so its descriptor is what the fingerprint has to cover: the format,
/// the extent, the row the case states and the usage its role names.
fn scratch_words() -> Vec<u8> {
    let spec = scratch_spec();
    let mut bytes = spec.format.as_raw().to_le_bytes().to_vec();
    for axis in spec.size {
        bytes.extend(axis.to_le_bytes());
    }
    bytes.extend(spec.row_pitch.to_le_bytes());
    bytes.push(match spec.role {
        TextureRole::Read => 0,
        TextureRole::Write => 1,
        TextureRole::ReadWrite => 2,
        TextureRole::RenderTarget => 3,
    });
    bytes
}

/// The target a case draws into: private, of `MTLTextureUsageRenderTarget` alone.
fn target_spec(side: u32) -> TargetSpec {
    TargetSpec::plain(FORMAT, [side; 2])
}

/// Bytes the whole of a `side` by `side` level 0 takes.
fn plane(side: u32) -> u32 {
    TEXEL * side * side
}

/// The layout a target's whole level 0 is read under.
fn layout(side: u32) -> ByteLayout {
    let row_pitch = TEXEL * side;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [side, side, 1],
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: 4 * side,
        row_pitch,
        rows: side,
        image_pitch: row_pitch * side,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The piece that copies a target's level 0 into a buffer the host reads.
fn read_target(texture: Handle, into: Handle, side: u32) -> RenderWork {
    RenderWork::ReadTexture {
        texture,
        into,
        offset: 0,
        row_pitch: TEXEL * side,
        size: [side; 2],
    }
}

/// The size the sparse heap takes, in bytes: one 16-KiB page, which is the descriptor's own page
/// size. A sparse heap holds no memory of its own until a mapping takes a tile, and nothing here
/// asks for one.
const HEAP_BYTES: u64 = 16 * 1024;

/// How many runs record mode takes a case of this module over, and why eight rather than three:
/// the barrier case is the one place in this packet where a wrong answer could be *intermittent*
/// rather than constant, because an unordered pair of draws may happen to come out ordered. Eight
/// runs that fold into one outcome are the evidence the recording then carries; three are too few
/// to call it.
const RUNS: u32 = 8;

/// The shader reference both texts of the one fixture give.
fn shader_refs() -> Result<Vec<ShaderRef>, String> {
    let fixture = Fixture::new(FIXTURE);
    [RUNTIME_TEXT, METAL_TEXT]
        .into_iter()
        .map(|extension| {
            Ok(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: ENTRIES.to_owned(),
                source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
            })
        })
        .collect()
}

/// The two inputs the fixture's own texts are.
fn fixture_inputs() -> Result<Vec<Input>, String> {
    let fixture = Fixture::new(FIXTURE);
    Ok(vec![
        Input {
            name: "air".to_owned(),
            digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
        },
        Input {
            name: "air-typed".to_owned(),
            digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
        },
    ])
}

/// The library of the module's one fixture.
fn library(side: &mut dyn Side, sites: &mut Sites) -> Result<Handle, Abort> {
    let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
        at: "library".to_owned(),
        failure: Failure::named("FixtureNotRead", why),
    })?;
    made(sites, "library", side.library(&loaded.source()))
}

/// Run `work` as one command buffer, emit what each piece answered, and read the target's texels.
fn drawn(
    side: &mut dyn Side,
    sites: &mut Sites,
    work: &[RenderWork],
    read_back: Handle,
    extent: u32,
) -> Result<(), Abort> {
    let outcome = call(sites, "render", side.render(work), |outcome| {
        outcome.status.clone()
    })?;
    let names = ["pass", "read"];
    let mut pieces = Node::branch("pieces", Vec::new());
    for (at, what) in names.iter().enumerate() {
        pieces = pieces.with(Node::leaf(
            *what,
            Observed::value(match outcome.refused.get(at).and_then(Option::as_ref) {
                Some(failure) => failure.recorded(),
                None => Value::ok(Value::Absent),
            }),
        ));
    }
    sites.push(pieces);
    if let Some((at, failure)) = names
        .iter()
        .enumerate()
        .find_map(|(at, what)| outcome.refused.get(at)?.as_ref().map(|f| (*what, f)))
    {
        return Err(Abort {
            at: format!("pieces/{at}"),
            failure: failure.clone(),
        });
    }
    let bytes = call(
        sites,
        "read-back",
        side.read_buffer(read_back, 0, plane(extent) as usize),
        |_| Value::Absent,
    )?;
    sites.push(
        Node::branch("drawn", Vec::new())
            .with(Node::leaf("texels", Observed::bytes(bytes, layout(extent)))),
    );
    Ok(())
}

/// The colour state both pipelines of either case take.
const PLAIN: [Option<ColorState>; 1] = [Some(ColorState::new(FORMAT))];

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![Box::new(RenderBarriers), Box::new(HeapResidency)]
}

// ------------------------------------------------------------------------------------------------
// render-state-render-barriers

/// The stages the barriers name on both sides: a fragment function's writes read by a fragment
/// function.
const FRAGMENT: RenderStages = RenderStages::FRAGMENT;

/// Whether a draw sees the texels a draw before it in the same pass wrote, across the two render
/// memory barriers, with the four `useResource` forms around them.
pub struct RenderBarriers;

impl RenderBarriers {
    /// The two pipelines: the one whose fragment function writes the scratch texture, and the one
    /// whose reads it.
    fn pipelines(colors: &'static [Option<ColorState>]) -> [RenderPipelineSpec<'static>; 2] {
        [
            RenderPipelineSpec::new(RECT, WRITE, colors),
            RenderPipelineSpec::new(RECT, READ, colors),
        ]
    }

    /// The pass's steps.
    fn steps(writing: Handle, reading: Handle, rects: Handle, scratch: Handle) -> Vec<RenderStep> {
        let texture = StepResource::Texture(scratch);
        let buffer = StepResource::Buffer(rects);
        vec![
            RenderStep::Pipeline(writing),
            RenderStep::VertexBuffer {
                buffer: rects,
                offset: 0,
                index: 0,
            },
            RenderStep::FragmentTexture {
                texture: scratch,
                index: 0,
            },
            // The four residency forms, over the scratch texture the fragment function writes and
            // the buffer the vertex function reads. Each is a call Metal answers; none of them is
            // what makes either reachable, because both are bound at a slot.
            RenderStep::UseResourceInStages {
                resource: texture,
                usage: ResourceUsage::WRITE,
                stages: FRAGMENT,
            },
            RenderStep::UseResource {
                resource: texture,
                usage: ResourceUsage::WRITE,
            },
            RenderStep::UseResourcesInStages {
                resources: vec![texture, buffer],
                usage: ResourceUsage::READ.union(ResourceUsage::WRITE),
                stages: RenderStages::VERTEX.union(RenderStages::FRAGMENT),
            },
            RenderStep::UseResources {
                resources: vec![texture, buffer],
                usage: ResourceUsage::READ,
            },
            RECT_DRAW,
            // The barriers: the scope form over every texture, and the resource form over the one
            // texture the next draw reads. Either alone would order the pair; the case states both
            // because each is an item of Metal no render case of the suite reached.
            RenderStep::BarrierScope {
                scope: BarrierScope::TEXTURES,
                after: FRAGMENT,
                before: FRAGMENT,
            },
            RenderStep::BarrierResources {
                resources: vec![texture],
                after: FRAGMENT,
                before: FRAGMENT,
            },
            RenderStep::Pipeline(reading),
            RenderStep::FragmentTexture {
                texture: scratch,
                index: 0,
            },
            RECT_DRAW,
        ]
    }
}

impl Case for RenderBarriers {
    fn name(&self) -> &'static str {
        "render-state-render-barriers"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing is loosened and no notice is declared. A `RGBA32Float` texel is the `float4` the
        // fragment function returned, the fixture's arithmetic is a multiplication by an eighth of
        // an integer, and a texel read at integer coordinates takes no device's float bounds. A
        // notice our side announced here would be undeclared and the case divergent, which is the
        // honest reading: a declaration stands only on a venue measured announcing it.
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs()?;
        for (at, pipeline) in Self::pipelines(&PLAIN).iter().enumerate() {
            inputs.push(Input::bytes(
                format!("render-pipeline:{at}"),
                &pipeline.words(),
            ));
        }
        inputs.push(Input::bytes("target", &target_spec(SIDE).words()));
        inputs.push(Input::bytes("texture:scratch", &scratch_words()));
        inputs.push(Input::bytes("buffer:rects", &rect_bytes()));
        inputs.push(Input::bytes("clear", &CLEAR.to_le_bytes()));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let library = library(side, sites)?;
        let mut states = Vec::new();
        for (at, pipeline) in Self::pipelines(&PLAIN).iter().enumerate() {
            states.push(made(
                sites,
                &format!("render-pipeline:{at}"),
                side.render_pipeline(library, pipeline),
            )?);
        }
        let target = made(sites, "target", side.target(&target_spec(SIDE)))?;
        let scratch = made(
            sites,
            "texture:scratch",
            side.texture_2d_in_role(&scratch_spec(), &[]),
        )?;
        let rects = made(sites, "buffer:rects", side.buffer(&rect_bytes()))?;
        let read_back = made(
            sites,
            "buffer:read-back",
            side.zeroed_buffer(plane(SIDE) as usize),
        )?;
        let work = vec![
            RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(target, [f64::from(CLEAR); 4])])
                    .drawing(Self::steps(states[0], states[1], rects, scratch)),
            ),
            read_target(target, read_back, SIDE),
        ];
        drawn(side, sites, &work, read_back, SIDE)
    }
}

// ------------------------------------------------------------------------------------------------
// render-state-heap-residency

/// Whether the four `useHeap` forms are answered on a heap that holds nothing, and that the pass's
/// own draw is untouched by them.
pub struct HeapResidency;

impl HeapResidency {
    /// The one pipeline: the rectangle and the solid colour.
    fn pipeline(colors: &'static [Option<ColorState>]) -> RenderPipelineSpec<'static> {
        RenderPipelineSpec::new(RECT, SOLID, colors)
    }

    /// The pass's steps.
    fn steps(state: Handle, rects: Handle, color: Handle, heap: Handle) -> Vec<RenderStep> {
        vec![
            RenderStep::Pipeline(state),
            RenderStep::VertexBuffer {
                buffer: rects,
                offset: 0,
                index: 0,
            },
            RenderStep::FragmentBuffer {
                buffer: color,
                offset: 0,
                index: 0,
            },
            RenderStep::UseHeap(heap),
            RenderStep::UseHeapInStages {
                heap,
                stages: FRAGMENT,
            },
            RenderStep::UseHeaps(vec![heap]),
            RenderStep::UseHeapsInStages {
                heaps: vec![heap],
                stages: RenderStages::VERTEX.union(RenderStages::FRAGMENT),
            },
            RECT_DRAW,
        ]
    }
}

impl Case for HeapResidency {
    fn name(&self) -> &'static str {
        "render-state-heap-residency"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing loosened and no notice declared, for the reason the barrier case states.
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs()?;
        inputs.push(Input::bytes(
            "render-pipeline",
            &Self::pipeline(&PLAIN).words(),
        ));
        inputs.push(Input::bytes("target", &target_spec(SMALL).words()));
        inputs.push(Input::bytes("buffer:rects", &rect_bytes()));
        inputs.push(Input::bytes("buffer:color", &color_bytes()));
        inputs.push(Input::bytes(
            "heap",
            &SparseHeapSpec::new(HEAP_BYTES).words(),
        ));
        inputs.push(Input::bytes("clear", &CLEAR.to_le_bytes()));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let library = library(side, sites)?;
        let state = made(
            sites,
            "render-pipeline",
            side.render_pipeline(library, &Self::pipeline(&PLAIN)),
        )?;
        let target = made(sites, "target", side.target(&target_spec(SMALL)))?;
        let rects = made(sites, "buffer:rects", side.buffer(&rect_bytes()))?;
        let color = made(sites, "buffer:color", side.buffer(&color_bytes()))?;
        let heap = made(
            sites,
            "heap",
            side.sparse_heap(&SparseHeapSpec::new(HEAP_BYTES)),
        )?;
        let read_back = made(
            sites,
            "buffer:read-back",
            side.zeroed_buffer(plane(SMALL) as usize),
        )?;
        let work = vec![
            RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(target, [f64::from(CLEAR); 4])])
                    .drawing(Self::steps(state, rects, color, heap)),
            ),
            read_target(target, read_back, SMALL),
        ];
        drawn(side, sites, &work, read_back, SMALL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Transcript;
    use std::collections::BTreeSet;

    /// The one outcome of `case`'s recorded Metal answer, or `None` where nothing recorded it.
    ///
    /// The assertion matters more here than anywhere else in the packet: a pair of draws a barrier
    /// failed to order may come out ordered by luck, so a case whose eight runs did **not** fold
    /// into one outcome is reporting exactly the non-determinism the barrier is there to remove.
    fn only_outcome(case: &str) -> Option<Transcript> {
        let recording = crate::recording::read(case).ok()?;
        let answer = recording.answers.first()?;
        assert_eq!(
            answer.outcomes.len(),
            1,
            "{case}: {} outcomes over {} runs, so the pass's answer moved between runs",
            answer.outcomes.len(),
            answer.runs
        );
        Some(answer.outcomes[0].transcript.clone())
    }

    /// The target's recorded texels of `case`.
    fn recorded(case: &str, side: u32) -> Option<Vec<u8>> {
        let transcript = only_outcome(case)?;
        let node = transcript
            .root
            .child("drawn")
            .and_then(|node| node.child("texels"))
            .expect("no drawn/texels site");
        let Value::Bytes(bytes) = &node.observed.value else {
            panic!("drawn/texels holds no bytes");
        };
        assert_eq!(bytes.len(), plane(side) as usize);
        Some(bytes.clone())
    }

    /// The `float4` at texel `(x, y)`.
    fn value(bytes: &[u8], x: u32, y: u32, side: u32) -> [f32; 4] {
        let at = ((y * side + x) * TEXEL) as usize;
        let mut channels = [0f32; 4];
        for (channel, held) in channels.iter_mut().enumerate() {
            let word = at + 4 * channel;
            *held = f32::from_le_bytes([
                bytes[word],
                bytes[word + 1],
                bytes[word + 2],
                bytes[word + 3],
            ]);
        }
        channels
    }

    /// What `tob_write` leaves in the scratch texture's texel under pixel `(x, y)`: the fixture's own
    /// arithmetic, every value an exact binary fraction.
    fn written_texel(x: u32, y: u32) -> [f32; 4] {
        [0.125 * x as f32, 0.125 * y as f32, 0.75, 1.0]
    }

    /// **Metal's recorded target is what the first draw wrote into the scratch texture**, in every
    /// texel: the second draw read the first draw's writes across the barrier.
    #[test]
    fn metal_let_the_second_draw_see_the_first_draws_texels() {
        let Some(bytes) = recorded(RenderBarriers.name(), SIDE) else {
            println!("render-state-render-barriers: no recording in the tree yet");
            return;
        };
        let mut differ = Vec::new();
        for y in 0..SIDE {
            for x in 0..SIDE {
                let (got, want) = (value(&bytes, x, y, SIDE), written_texel(x, y));
                if got != want {
                    differ.push(format!(
                        "({x}, {y}): Metal {got:?}, the fixture writes {want:?}"
                    ));
                }
            }
        }
        for line in differ.iter().take(20) {
            println!("BARRIER-DIFFERS: {line}");
        }
        assert!(
            differ.is_empty(),
            "{} of {} texels are not what the first draw wrote, so the second draw did not see it",
            differ.len(),
            SIDE * SIDE
        );
    }

    /// **The texels the first draw writes are distinct from the clear, from the colour the other
    /// case stores, and from each other**, so the barrier case cannot pass on a target nothing
    /// wrote.
    ///
    /// This is the check the round's own vacuity incident earns: a draw that wrote nothing and a
    /// round trip whose two halves never met both read as a pass when the values on either side
    /// are indistinguishable. Here they are not.
    #[test]
    fn the_written_texels_are_a_value_nothing_else_leaves() {
        let written: BTreeSet<[u32; 4]> = (0..SIDE)
            .flat_map(|y| (0..SIDE).map(move |x| written_texel(x, y).map(f32::to_bits)))
            .collect();
        assert_eq!(
            written.len(),
            (SIDE * SIDE) as usize,
            "two texels answer alike"
        );
        assert!(!written.contains(&[CLEAR; 4].map(f32::to_bits)));
        assert!(!written.contains(&COLOR.map(f32::to_bits)));
        // Opaque black is what the writing draw's fragment function returns into the target, so
        // the target holding it anywhere would mean the reading draw never ran.
        assert!(!written.contains(&[0.0, 0.0, 0.0, 1.0].map(f32::to_bits)));
        // Zero is what an untouched texture of shared storage may read, so a second draw that saw
        // nothing of the first is told apart from one that saw all of it.
        assert!(!written.contains(&[0.0; 4].map(f32::to_bits)));
        // The scratch texture is in both shader roles and the host writes nothing into it, which
        // is what makes the first draw the only source of what the second reads.
        assert_eq!(scratch_spec().role, TextureRole::ReadWrite);
        assert_eq!(layout(SIDE).extent(), plane(SIDE) as usize);
        // The two pipelines are two pipelines.
        let [writing, reading] = RenderBarriers::pipelines(&PLAIN);
        assert_ne!(writing.words(), reading.words());
    }

    /// **The heap case's target is the colour its binding holds**, in every texel: the four
    /// `useHeap` forms changed nothing the pass observes.
    #[test]
    fn metal_drew_the_heap_case_as_its_binding_says() {
        let Some(bytes) = recorded(HeapResidency.name(), SMALL) else {
            println!("render-state-heap-residency: no recording in the tree yet");
            return;
        };
        for y in 0..SMALL {
            for x in 0..SMALL {
                assert_eq!(
                    value(&bytes, x, y, SMALL),
                    COLOR,
                    "({x}, {y}) is not the colour the fragment binding holds"
                );
            }
        }
        assert_ne!(COLOR, [CLEAR; 4]);
    }

    /// **Every case of the family names itself once**, under the settings the module states.
    #[test]
    fn the_family_names_two_cases_apart() {
        let names: Vec<&str> = all().iter().map(|case| case.name()).collect();
        let distinct: BTreeSet<&str> = names.iter().copied().collect();
        assert_eq!(distinct.len(), names.len(), "two cases share a name");
        assert_eq!(names.len(), 2);
        for case in all() {
            assert_eq!(case.record_runs(), RUNS);
            assert!(matches!(case.metal_source(), MetalSource::Run));
            assert!(case.allowances().is_empty());
        }
        // The barriers name the stage whose writes the next draw reads, on both sides of each.
        assert_eq!(FRAGMENT, RenderStages::FRAGMENT);
        // The heap is one page of the descriptor's own page size, which holds no memory until a
        // mapping takes a tile.
        assert_eq!(HEAP_BYTES, 16 * 1024);
    }
}
