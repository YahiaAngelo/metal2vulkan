//! Device tests of Metal's private primitive types 5 to 8 ([`DrawPrimitiveType`]: a line loop, a
//! triangle fan, quads and a quad strip) on the device the cfg selects, under the validation
//! layer. The owned fixture `rt6_private_types` places vertex `v` at the clip-space position
//! buffer 0 holds at `min(v, 65535)`, moved by buffer 1's offset of its instance, and its fragment
//! functions write, into two `RGBA32Float` and two `RGBA32Uint` attachments, the barycentric
//! weights, a smooth value of the vertex ID, the flat vertex ID, base vertex, instance ID and
//! base instance, and the primitive ID and front-facing bit (`pt_full`), or some of them.
//!
//! Metal's answers (probes on the Apple M3 Max, macOS 27.0.1: counts 0 to 12, a vertex start,
//! instances and a base instance, a base vertex, 16- and 32-bit indices, the all-ones indices and
//! `setPrimitiveRestartEnabled:index:`, indirect and indexed indirect draws, 160 draws and more,
//! every fragment's weights solved for its triangle's vertices): plain Metal draws each private
//! type as the list [`metal_list`] states, in its order, each list primitive's barycentric order
//! the list's and its flat inputs its first vertex's under provoking vertex mode 0;
//! `[[primitive_id]]` counts the loop's segments and the fan's triangles, and the quads, one for
//! both triangles of each. The line loop, fan and quads never restart; the quad strip restarts as
//! the strips do. `[[vertex_id]]`, `[[base_vertex]]` (a draw without indices: its vertex start),
//! `[[instance_id]]` (from the base instance) and `[[base_instance]]` are the public draw's.
//!
//! So each check draws a private type and the public list of [`metal_list`]'s indices over the
//! same vertices, base vertex and instances on the same device, and compares every texel of the
//! four attachments: coverage, the order (the last primitive over a pixel wins), the weights, the
//! interpolated and flat values and the primitive ID. A negative control draws the same indices as
//! the public strip of their class and asks it to differ, so the comparison tells the forms apart.
//! A direct draw without indices draws the list the runtime writes on the CPU; every other form
//! the list the index rewrite pass writes on the GPU, from indices or arguments a blit may write
//! earlier in the command buffer. Quads and a quad strip under a fragment function that reads
//! `[[primitive_id]]` are refused (`NotImplemented(PrivatePrimitives)`): the list's triangles
//! count two IDs where Metal counts one.

use super::{buffer_with, validated_device, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{IndexType, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandBufferError, CommandBufferStatus, CommandQueue,
    Device, DrawPrimitiveType, Error, Library, LoadAction, PipelineOptions, Region,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions, RuntimeGap,
    StoreAction, TextureDescriptor,
};

/// The target's side, in pixels.
const SIDE: u64 = 32;

/// The vertices the walks of indices name.
const SCATTERED: u32 = 64;

/// The vertices the positions scatter over the target, besides vertex 65535: those the walks name
/// and those of the long draws without indices.
const PLACED: usize = 4096;

/// The attachments' formats: the weights, the smooth value, the flat values, the primitive ID.
const FORMATS: [PixelFormat; 4] = [
    PixelFormat::RGBA32Float,
    PixelFormat::RGBA32Float,
    PixelFormat::RGBA32Uint,
    PixelFormat::RGBA32Uint,
];

/// The four private types.
const PRIVATE: [DrawPrimitiveType; 4] = [
    DrawPrimitiveType::LineLoop,
    DrawPrimitiveType::TriangleFan,
    DrawPrimitiveType::Quads,
    DrawPrimitiveType::QuadStrip,
];

/// Whether `primitive` draws lines.
fn lines(primitive: DrawPrimitiveType) -> bool {
    primitive == DrawPrimitiveType::LineLoop
}

/// Whether `primitive` counts one `[[primitive_id]]` for two triangles.
fn quads(primitive: DrawPrimitiveType) -> bool {
    matches!(
        primitive,
        DrawPrimitiveType::Quads | DrawPrimitiveType::QuadStrip
    )
}

/// The list Metal draws of `values` (each position's index, as Metal reads it) under the private
/// type `primitive`, restarting a quad strip at `restart` (the probes' model, written apart from
/// the runtime's).
fn metal_list(primitive: DrawPrimitiveType, values: &[u32], restart: u32) -> Vec<u32> {
    let mut out = Vec::new();
    match primitive {
        DrawPrimitiveType::LineLoop => {
            if values.len() >= 2 {
                for i in 0..values.len() - 1 {
                    out.extend([values[i], values[i + 1]]);
                }
                out.extend([values[values.len() - 1], values[0]]);
            }
        }
        DrawPrimitiveType::TriangleFan => {
            for j in 0..values.len().saturating_sub(2) {
                out.extend([values[0], values[j + 1], values[j + 2]]);
            }
        }
        DrawPrimitiveType::Quads => {
            for q in values.chunks_exact(4) {
                out.extend([q[0], q[1], q[3], q[1], q[2], q[3]]);
            }
        }
        DrawPrimitiveType::QuadStrip => {
            for piece in values.split(|&v| v == restart) {
                for k in 0..piece.len().saturating_sub(2) / 2 {
                    let (a, b, c, d) = (
                        piece[2 * k],
                        piece[2 * k + 1],
                        piece[2 * k + 2],
                        piece[2 * k + 3],
                    );
                    out.extend([a, b, d, c, a, d]);
                }
            }
        }
        _ => unreachable!("a public type"),
    }
    out
}

/// The public list type of `primitive`'s list.
fn list_type(primitive: DrawPrimitiveType) -> PrimitiveType {
    match lines(primitive) {
        true => PrimitiveType::Line,
        false => PrimitiveType::Triangle,
    }
}

/// The public strip of `primitive`'s class: the negative control.
fn strip_type(primitive: DrawPrimitiveType) -> PrimitiveType {
    match lines(primitive) {
        true => PrimitiveType::LineStrip,
        false => PrimitiveType::TriangleStrip,
    }
}

/// The fixture's 65536 positions, `(x, y, z, w)` each: vertices 0 to [`PLACED`] less 1 and 65535 at
/// points of a fixed pseudo-random walk over the target and a little past it, at depths and `w`
/// that make the weights perspective-correct; every other vertex off the target.
fn positions() -> Vec<u8> {
    let mut all = vec![[5.0f32, 5.0, 0.0, 1.0]; 65536];
    let mut state = 0x2545_f491_u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for vertex in (0..PLACED).chain([65535]) {
        let unit = |n: u32| (n % 10_000) as f32 / 10_000.0;
        let (x, y) = (unit(next()) * 2.4 - 1.2, unit(next()) * 2.4 - 1.2);
        let w = 1.0 + 2.0 * unit(next());
        all[vertex] = [x * w, y * w, 0.5 * w * unit(next()), w];
    }
    all.iter().flatten().flat_map(|v| v.to_le_bytes()).collect()
}

/// Each instance's offset, of instance IDs 0 to 3.
fn offsets() -> Vec<u8> {
    [[0.0f32, 0.0], [0.06, -0.04], [-0.05, 0.07], [0.03, 0.03]]
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

/// `values` as indices of `index_type`.
fn bytes_of(index_type: IndexType, values: &[u32]) -> Vec<u8> {
    match index_type {
        IndexType::UInt16 => values
            .iter()
            .map(|&i| u16::try_from(i).unwrap())
            .flat_map(u16::to_le_bytes)
            .collect(),
        _ => values.iter().flat_map(|i| i.to_le_bytes()).collect(),
    }
}

/// How a case draws.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Form {
    /// `draw_primitives`.
    Direct,
    /// `draw_primitives_indirect`.
    Indirect,
    /// `draw_indexed_primitives`, of indices of this type.
    Indexed(IndexType),
    /// `draw_indexed_primitives`, of indices of this type a blit earlier in the command buffer
    /// writes over others.
    Written(IndexType),
    /// `draw_indexed_primitives_indirect`, of indices of this type.
    IndexedIndirect(IndexType),
}

impl Form {
    /// The form's index type, `None` for a draw without indices.
    fn index_type(self) -> Option<IndexType> {
        match self {
            Form::Direct | Form::Indirect => None,
            Form::Indexed(t) | Form::Written(t) | Form::IndexedIndirect(t) => Some(t),
        }
    }
}

/// A case: one draw of a private type.
#[derive(Clone, Debug)]
struct Case {
    primitive: DrawPrimitiveType,
    form: Form,
    /// The positions it draws.
    count: u32,
    /// A draw without indices: its vertex start; with: its first index.
    start: u32,
    instances: u32,
    base_instance: u32,
    base_vertex: i32,
    /// The index buffer's indices (empty for a draw without indices): those past the buffer read
    /// 0.
    indices: Vec<u32>,
    /// The encoder's `set_primitive_restart_enabled` index before the draw, if any.
    k: Option<u64>,
}

impl Case {
    /// A direct draw without indices.
    fn direct(primitive: DrawPrimitiveType, count: u32) -> Case {
        Case {
            primitive,
            form: Form::Direct,
            count,
            start: 3,
            instances: 2,
            base_instance: 1,
            base_vertex: 0,
            indices: Vec::new(),
            k: None,
        }
    }

    /// A draw of `indices` in `form`.
    fn indexed(primitive: DrawPrimitiveType, form: Form, indices: Vec<u32>) -> Case {
        Case {
            primitive,
            form,
            count: indices.len() as u32,
            start: 0,
            instances: 1,
            base_instance: 0,
            base_vertex: 2,
            indices,
            k: None,
        }
    }

    /// Each position's index, as Metal reads it: a draw without indices its vertex less the
    /// start (the reference's base vertex adds it), an indexed one the index at its position from
    /// the first, 0 past the buffer.
    fn values(&self) -> Vec<u32> {
        match self.form.index_type() {
            None => (0..self.count).collect(),
            Some(_) => (0..self.count)
                .map(|p| {
                    let at = (self.start + p) as usize;
                    self.indices.get(at).copied().unwrap_or(0)
                })
                .collect(),
        }
    }

    /// The index a quad strip restarts at: `k'` after the encoder's call, else the index type's
    /// all-ones index; none for a draw without indices.
    fn restart(&self) -> Option<u32> {
        let index_type = self.form.index_type()?;
        Some(match (self.k, index_type) {
            (Some(k), IndexType::UInt16) => (k as u32).min(0xffff),
            (Some(k), _) => k as u32,
            (None, IndexType::UInt16) => 0xffff,
            (None, _) => u32::MAX,
        })
    }

    /// Metal's list of the case (a draw without indices, whose positions are below `u32::MAX`,
    /// restarts nowhere).
    fn list(&self) -> Vec<u32> {
        let restart = self.restart().unwrap_or(u32::MAX);
        metal_list(self.primitive, &self.values(), restart)
    }

    /// The reference draw's base vertex: the vertex start of a draw without indices.
    fn reference_base_vertex(&self) -> i32 {
        match self.form.index_type() {
            None => self.start as i32,
            Some(_) => self.base_vertex,
        }
    }
}

/// The lane's device, its queue, the fixture's library and the positions.
struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
    positions: Buffer,
    offsets: Buffer,
}

/// Each attachment's texels, four words each.
type Texels = Vec<Vec<[u32; 4]>>;

impl Lane {
    fn new(device: Device) -> Lane {
        let bitcode = llvm_assemble(&engine_dir().join("rt6_private_types.ll")).unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            library: device.new_library_with_air(&bitcode).unwrap(),
            positions: buffer_with(&device, &positions()).unwrap(),
            offsets: buffer_with(&device, &offsets()).unwrap(),
            device,
        }
    }

    /// The pipeline of `pt_vertex` and `fragment` into [`FORMATS`].
    fn pipeline(&self, fragment: &str) -> RenderPipelineState {
        let function = |name| {
            StageSource::Function(
                self.library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("pt_vertex"));
        descriptor.fragment = Some(function(fragment));
        for (i, format) in FORMATS.iter().enumerate() {
            descriptor.color_attachments[i] = Some(ColorAttachmentDescriptor::new(*format));
        }
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    /// Run `encode` (given the command buffer, before its render encoder, then the encoder with
    /// `pipeline` and the vertex buffers set) over cleared targets, commit, and return the
    /// command buffer and every attachment's texels.
    fn run(
        &self,
        pipeline: &RenderPipelineState,
        before: impl FnOnce(&CommandBuffer) -> Result<(), Error>,
        encode: impl FnOnce(&mut RenderCommandEncoder) -> Result<(), Error>,
    ) -> Result<(CommandBuffer, Texels), Error> {
        let targets: Vec<_> = FORMATS
            .iter()
            .map(|format| {
                let mut descriptor = TextureDescriptor::texture_2d(*format, SIDE, SIDE, false);
                descriptor.usage = TextureUsage::RENDER_TARGET;
                descriptor.resource_options = ResourceOptions::shared();
                self.device.new_texture(&descriptor).unwrap()
            })
            .collect();
        let mut pass = RenderPassDescriptor::new();
        for (i, (target, format)) in targets.iter().zip(FORMATS).enumerate() {
            let color = &mut pass.color_attachments[i];
            color.texture = Some(target.clone());
            color.load_action = LoadAction::Clear;
            color.clear_color = match format {
                PixelFormat::RGBA32Uint => ClearColor::new(0.0, 0.0, 0.0, 0.0),
                _ => ClearColor::new(-1.0, -1.0, -1.0, 0.0),
            };
            color.store_action = StoreAction::Store;
        }
        let command_buffer = self.queue.command_buffer()?;
        before(&command_buffer)?;
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        encoder.set_render_pipeline_state(pipeline)?;
        encoder.set_vertex_buffer(Some(&self.positions), 0, 0)?;
        encoder.set_vertex_buffer(Some(&self.offsets), 0, 1)?;
        let drawn = encode(&mut encoder);
        let ended = encoder.end_encoding();
        command_buffer.commit()?;
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        drawn?;
        ended?;
        let texels = targets
            .iter()
            .map(|target| {
                let mut out = vec![0u8; (SIDE * SIDE * 16) as usize];
                target
                    .get_bytes(
                        &mut out,
                        SIDE * 16,
                        SIDE * SIDE * 16,
                        Region::new_2d(0, 0, SIDE, SIDE),
                        0,
                        0,
                    )
                    .unwrap();
                out.as_chunks::<16>()
                    .0
                    .iter()
                    .map(|t| {
                        let (words, _) = t.as_chunks::<4>();
                        std::array::from_fn(|i| u32::from_le_bytes(words[i]))
                    })
                    .collect()
            })
            .collect();
        Ok((command_buffer, texels))
    }

    /// Metal's answer to `case`: [`Case::list`] drawn as the public list of 32-bit indices, with
    /// the case's instances, base instance and base vertex (a draw without indices: its vertex
    /// start).
    fn reference(&self, pipeline: &RenderPipelineState, case: &Case) -> Texels {
        self.public(pipeline, case, list_type(case.primitive), &case.list())
    }

    /// `indices` drawn as the public `primitive` of 32-bit indices, with `case`'s instances, base
    /// instance and base vertex.
    fn public(
        &self,
        pipeline: &RenderPipelineState,
        case: &Case,
        primitive: PrimitiveType,
        indices: &[u32],
    ) -> Texels {
        let bytes = bytes_of(IndexType::UInt32, indices);
        let buffer = buffer_with(
            &self.device,
            &bytes.iter().copied().chain([0; 4]).collect::<Vec<_>>(),
        )
        .unwrap();
        let (command_buffer, texels) = self
            .run(
                pipeline,
                |_| Ok(()),
                |encoder| {
                    encoder.draw_indexed_primitives(
                        primitive,
                        indices.len() as u32,
                        IndexType::UInt32,
                        &buffer,
                        0,
                        case.instances,
                        case.reference_base_vertex(),
                        case.base_instance,
                    )
                },
            )
            .unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        texels
    }

    /// `case` drawn as its private type, in its form.
    fn private(&self, pipeline: &RenderPipelineState, case: &Case) -> Result<Texels, Error> {
        let index_type = case.form.index_type().unwrap_or(IndexType::UInt32);
        let index_bytes = bytes_of(index_type, &case.indices);
        // A buffer of 4 bytes at least, for a draw without indices that names none.
        let padded: Vec<u8> = index_bytes.iter().copied().chain([0; 4]).collect();
        let size = match index_type {
            IndexType::UInt16 => 2,
            _ => 4,
        };
        let staging = buffer_with(&self.device, &padded).unwrap();
        // Indices a blit writes: the buffer holds a walk the other way first, and exactly the
        // case's whole indices, so positions past them read 0.
        let first: Vec<u32> = match case.form {
            Form::Written(_) => (0..case.indices.len() as u32)
                .map(|i| (i * 3 + 2) % SCATTERED)
                .collect(),
            _ => case.indices.clone(),
        };
        let index_buffer = match case.indices.is_empty() {
            true => staging.clone(),
            false => buffer_with(&self.device, &bytes_of(index_type, &first)).unwrap(),
        };
        let words: Vec<u32> = match case.form {
            Form::Indirect => vec![case.count, case.instances, case.start, case.base_instance],
            _ => vec![
                case.count,
                case.instances,
                case.start,
                case.base_vertex as u32,
                case.base_instance,
            ],
        };
        let arguments: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let indirect = buffer_with(&self.device, &arguments).unwrap();
        let (command_buffer, texels) = self.run(
            pipeline,
            |command_buffer| {
                if let Form::Written(_) = case.form {
                    let mut blit = command_buffer.blit_command_encoder()?;
                    let length = index_bytes.len() as u64;
                    blit.copy_from_buffer_to_buffer(&staging, 0, &index_buffer, 0, length)?;
                    blit.end_encoding()?;
                }
                Ok(())
            },
            |encoder| {
                if let Some(k) = case.k {
                    encoder.set_primitive_restart_enabled(true, k);
                }
                match case.form {
                    Form::Direct => encoder.draw_primitives(
                        case.primitive,
                        case.start,
                        case.count,
                        case.instances,
                        case.base_instance,
                    ),
                    Form::Indirect => {
                        encoder.draw_primitives_indirect(case.primitive, &indirect, 0)
                    }
                    Form::Indexed(t) | Form::Written(t) => encoder.draw_indexed_primitives(
                        case.primitive,
                        case.count,
                        t,
                        &index_buffer,
                        u64::from(case.start) * size,
                        case.instances,
                        case.base_vertex,
                        case.base_instance,
                    ),
                    Form::IndexedIndirect(t) => encoder.draw_indexed_primitives_indirect(
                        case.primitive,
                        t,
                        &index_buffer,
                        0,
                        &indirect,
                        0,
                    ),
                }
            },
        )?;
        // A call lowered after it returned fails its command buffer with the call's error.
        match command_buffer.error() {
            Some(CommandBufferError::CallRefused(_)) => {
                return Err(command_buffer.refusal().expect("the refused call's error"));
            }
            Some(other) => panic!("{case:?}: the command buffer failed: {other:?}"),
            None => {}
        }
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        Ok(texels)
    }
}

/// Whether the case's texels cover a pixel.
fn covers(texels: &[Vec<[u32; 4]>]) -> bool {
    texels[3].iter().any(|t| t[3] == 1)
}

/// The fragment function a case of `primitive` reads everything Metal gives it with: the primitive
/// ID too, but under quads, where it is refused.
fn full(primitive: DrawPrimitiveType) -> &'static str {
    match quads(primitive) {
        true => "pt_weights",
        false => "pt_full",
    }
}

/// The direct draws without indices: each type at every count from 0 to 11, from vertex start 3,
/// two instances from base instance 1.
fn direct_cases() -> Vec<Case> {
    PRIVATE
        .iter()
        .flat_map(|&p| (0..12).map(move |count| Case::direct(p, count)))
        .collect()
}

/// The indexed and indirect draws.
fn indexed_cases() -> Vec<Case> {
    use IndexType::{UInt16, UInt32};
    let walk =
        |n: u32, step: u32| -> Vec<u32> { (0..n).map(|i| (i * step + 1) % SCATTERED).collect() };
    let mut cases = Vec::new();
    for &p in &PRIVATE {
        // 14 indices with each type's all-ones index and 7 among them.
        let with = |ones: u32| {
            let mut indices = walk(14, 5);
            indices[5] = ones;
            indices[9] = 7;
            indices
        };
        cases.push(Case::indexed(p, Form::Indexed(UInt16), with(0xffff)));
        cases.push(Case::indexed(p, Form::Indexed(UInt32), with(u32::MAX)));
        // A 32-bit 0xffff, a vertex of every type.
        cases.push(Case::indexed(p, Form::Indexed(UInt32), with(0xffff)));
        // After the encoder's call: 7 restarts a quad strip, and the all-ones index is a vertex.
        let mut k7 = Case::indexed(p, Form::Indexed(UInt16), with(0xffff));
        k7.k = Some(7);
        cases.push(k7);
        // From a first index, two instances, past the buffer's indices.
        let mut past = Case::indexed(p, Form::Indexed(UInt32), walk(11, 7));
        past.start = 2;
        past.count = 13;
        past.instances = 2;
        past.base_instance = 1;
        cases.push(past);
        // 28 bytes: a blit copies multiples of 4.
        cases.push(Case::indexed(p, Form::Written(UInt16), walk(14, 3)));
        let mut indirect = Case::indexed(p, Form::IndexedIndirect(UInt16), with(0xffff));
        indirect.start = 1;
        indirect.count = 12;
        indirect.instances = 2;
        indirect.base_instance = 2;
        cases.push(indirect);
        let mut vertices = Case::direct(p, 10);
        vertices.form = Form::Indirect;
        cases.push(vertices);
        // Many positions across the pass's runs and groups: restarts at 7 every 37 for the quad
        // strip, and the all-ones index every 41 (a vertex of the others).
        let long: Vec<u32> = (0..3001u32)
            .map(|i| match (i % 37, i % 41) {
                (36, _) => 7,
                (_, 40) => 0xffff,
                _ => ((i * 11 + 1) % SCATTERED) | 8,
            })
            .collect();
        let mut long16 = Case::indexed(p, Form::Indexed(UInt16), long.clone());
        long16.k = Some(7);
        cases.push(long16);
        cases.push(Case::indexed(p, Form::IndexedIndirect(UInt32), long));
        let mut many = Case::direct(p, 2999);
        many.form = Form::Indirect;
        cases.push(many);
    }
    cases
}

/// Each case of `cases` draws, at every texel of every attachment, what Metal's list drawn as the
/// public list draws, under the fragment function [`full`] names, and most cover a pixel.
fn draws_metal_s_lists(cases: &[Case]) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let pipelines: Vec<_> = PRIVATE
            .iter()
            .map(|&p| (p, lane.pipeline(full(p))))
            .collect();
        let (mut drawn, mut covered) = (0, 0);
        for case in cases {
            let pipeline = &pipelines
                .iter()
                .find(|(p, _)| *p == case.primitive)
                .unwrap()
                .1;
            let expected = lane.reference(pipeline, case);
            let texels = lane
                .private(pipeline, case)
                .unwrap_or_else(|e| panic!("{case:?} on {name}: {e}"));
            drawn += usize::from(covers(&expected));
            covered += expected[3].iter().filter(|t| t[3] == 1).count();
            for (a, (got, want)) in texels.iter().zip(&expected).enumerate() {
                let differ = got.iter().zip(want).filter(|(g, w)| g != w).count();
                assert_eq!(
                    differ, 0,
                    "{case:?} on {name}: attachment {a} differs from Metal's list at {differ} texels"
                );
            }
        }
        assert!(
            drawn * 2 >= cases.len(),
            "on {name}: only {drawn} of {} cases cover a pixel",
            cases.len()
        );
        eprintln!(
            "{name}: {} private draws equal Metal's lists at every texel; {drawn} cover pixels, \
             {covered} covered pixels in all",
            cases.len()
        );
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// Every direct draw without indices draws Metal's list, from the CPU's.
#[test]
fn direct_private_draws_are_metal_s_lists() {
    draws_metal_s_lists(&direct_cases());
}

/// Every indexed or indirect draw draws Metal's list, from the index rewrite pass's.
#[test]
fn indexed_and_indirect_private_draws_are_metal_s_lists() {
    draws_metal_s_lists(&indexed_cases());
}

/// Under fragment functions that read fewer of the inputs (the primitive ID without the weights,
/// or neither), a draw of each form draws Metal's list too; and the same indices drawn as the
/// public strip of their class differ from Metal's list, so the comparison tells the forms apart.
#[test]
fn other_fragments_draw_metal_s_lists_and_strips_differ() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let walk: Vec<u32> = (0..14).map(|i| (i * 5 + 1) % SCATTERED).collect();
        for fragment in ["pt_primitive", "pt_plain"] {
            let pipeline = lane.pipeline(fragment);
            for &p in &PRIVATE {
                if fragment == "pt_primitive" && quads(p) {
                    continue;
                }
                let cases = [
                    Case::direct(p, 10),
                    Case::indexed(p, Form::Indexed(IndexType::UInt16), walk.clone()),
                ];
                for case in &cases {
                    let expected = lane.reference(&pipeline, case);
                    let texels = lane
                        .private(&pipeline, case)
                        .unwrap_or_else(|e| panic!("{fragment} {case:?} on {name}: {e}"));
                    assert!(
                        covers(&expected),
                        "{fragment} {case:?}: Metal's list covers nothing"
                    );
                    assert!(
                        texels == expected,
                        "{fragment} {case:?} on {name}: differs from Metal's list"
                    );
                }
                let strip = lane.public(&pipeline, &cases[1], strip_type(p), &cases[1].values());
                assert!(
                    strip != lane.reference(&pipeline, &cases[1]),
                    "{fragment} {p:?} on {name}: the public strip draws Metal's list"
                );
            }
        }
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// Quads and a quad strip under a fragment function that reads `[[primitive_id]]` are refused
/// `NotImplemented(PrivatePrimitives)`, direct and indexed, and draw nothing; the line loop and
/// the fan under it draw.
#[test]
fn quads_under_a_primitive_id_are_refused() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let walk: Vec<u32> = (0..14).map(|i| (i * 5 + 1) % SCATTERED).collect();
        for fragment in ["pt_full", "pt_primitive"] {
            let pipeline = lane.pipeline(fragment);
            for &p in &PRIVATE {
                for case in [
                    Case::direct(p, 10),
                    Case::indexed(p, Form::Indexed(IndexType::UInt32), walk.clone()),
                ] {
                    let result = lane.private(&pipeline, &case);
                    match quads(p) {
                        true => assert!(
                            matches!(
                                result,
                                Err(Error::NotImplemented(RuntimeGap::PrivatePrimitives))
                            ),
                            "{fragment} {case:?} on {name}: {:?}",
                            result.err()
                        ),
                        false => assert!(result.is_ok(), "{fragment} {case:?} on {name}"),
                    }
                }
            }
        }
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}
