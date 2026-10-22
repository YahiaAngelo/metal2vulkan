//! The index rewrite pass of an indexed strip that restarts where Vulkan does not, of a draw of
//! one of Metal's private primitive types over indices or from indirect arguments, and of a draw
//! under Apple's last provoking vertex that a device without it draws as a list led by each
//! primitive's last vertex (`render/provoking.rs`) (the modules and their contract are
//! `bind::index_rewrite`); the list the CPU writes for such a direct draw without indices
//! ([`WrittenList`]); and the lists of the index gather pass ([`GatherRuns`]). Each list is
//! written in the draw's `RewriteOrder`.
//!
//! **The gather pass** (`bind::index_gather_module`) writes the list of a direct indexed draw of
//! a program that never restarts (`RewriteProgram::gathers`: line and triangle lists, Apple's
//! triangle fan and quads) that reads its indices in place and that no earlier draw's write
//! orders before its render pass instance: the list the rewrite pass would write, index for
//! index. The draw joins an open run of its index buffer, program and order (up to
//! [`RUN_DRAWS`] draws, into an output buffer of the run's own of at least [`RUN_OUTPUT_WORDS`]
//! words), and draws its part of the run's output with `vkCmdDrawIndexed`. Its reads are
//! tracked as it joins; the open runs (up to [`OPEN_RUNS`]) record before the encoder's next
//! command and before its `VkCommandBuffer` ends (`encoder.rs`): each run's table written, then
//! one dispatch per run between one barrier from everything before and one to everything after.
//! Every other draw's list is the rewrite pass's, of three dispatches between barriers each: on
//! MoltenVK on the M3 Max, 78 us of frame time per draw of two triangles, where a gathered draw
//! takes 0.26 us more than at provoking vertex mode 0 (host_ab's `provoking`).
//!
//! After `RenderCommandEncoder::set_primitive_restart_enabled` (Apple's
//! `setPrimitiveRestartEnabled:index:`), Metal restarts an indexed line or triangle strip at the
//! call's index `k'` alone, and draws the index type's all-ones index as a vertex; Vulkan restarts
//! at the all-ones index alone. Where `k'` is not the all-ones index, the render encoder draws the
//! strip as the list this pass writes:
//!
//! - **Before the pass** (hoisted, as the index bounds pass is, `index_bounds.rs`): three
//!   dispatches of the program's modules, one per `RewritePhase` (count, place, write), read
//!   Metal's indices (and an indirect draw's arguments) and write the list and the draw's
//!   `VkDrawIndexedIndirectCommand` into the draw's rewrite words (an upload of
//!   [`IndexRewritePass::BYTES`]), between a barrier from everything before and one to
//!   everything after. The count and write dispatches spread the draw's positions over up to
//!   `IndexRewritePass::MAX_GROUPS` workgroups, as many as its index buffer's whole indices need.
//! - **The list** of a direct draw lies in a buffer of its own, of the program's capacity of the
//!   draw's count, made for the draw; an indirect draw's, whose count only the GPU knows, in the
//!   command buffer's index pool, which the index bounds pass shares (its counter in the arena,
//!   its indices in its queue's block, which the command buffer leases, `gpu_pools.rs`): the count
//!   and place dispatches bind the counter as the output, the write dispatch the block. A list the
//!   pool cannot hold draws nothing and fails the command buffer with
//!   `CommandBufferError::IndexPoolOverflow`.
//! - **In the pass**, one indexed indirect draw of 32-bit indices over the list, of the class's
//!   list topology with restart off (`render.rs`): every primitive keeps its winding, its
//!   provoking vertex and its place in `primitive_id`'s count.
//! - **The hazard**, as the index bounds pass's: where an earlier draw or tile dispatch of the
//!   encoder writes the indices or arguments behind a memory barrier, the pass records before the
//!   draw's own render pass instance, after the writer's; a write since the last barrier that the
//!   pass would read in place is refused (`NotImplemented(RestartedStrip)`), as is any earlier
//!   write where an execution's command draws. A blit's or another encoder's write is ordered
//!   before the pass by the command buffer's tracking, as any compute read is.

use super::buffer::Buffer;
use super::degradation::Degradation;
use super::draw_source::{DrawSource, IndexBind, SizedIndexBind};
use super::encoder::EncoderCore;
use super::error::{internal, Error, RuntimeGap};
use super::gpu_pools::IndexPool;
use super::index_bounds;
use super::owned_kernel::{OwnedKernel, OwnedModule, OwnedSets};
use super::resource::ResourceKey;
use super::tracker::{Access, Barrier, Use, UseRange};
use super::upload::Upload;
use super::upload::Uploads;
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::{
    ExecutionStatus, GatherDraw, IndexBufferRef, IndexGather, IndexGatherModule,
    IndexRewriteModule, IndexRewritePass, IndirectRef, RewriteOrder, RewriteOutput, RewritePhase,
    RewriteProgram, RewriteSource,
};
use metal2vulkan::device::VkFeature;

/// A rewritten draw's Metal arguments: a direct draw's values, or an indirect draw's buffer.
#[derive(Clone, Copy)]
pub(super) enum Arguments<'a> {
    /// A direct draw's index count, instances, base vertex and base instance.
    Direct {
        count: u32,
        instances: u32,
        base_vertex: i32,
        base_instance: u32,
    },
    /// An indirect draw's `MTLDrawIndexedPrimitivesIndirectArguments`.
    Indirect(&'a DrawSource<'a>),
    /// An indirect draw's `MTLDrawPrimitivesIndirectArguments`, of a draw without indices.
    IndirectVertices(&'a DrawSource<'a>),
}

/// Where a pass writes its list.
enum Output {
    /// A direct draw's own buffer.
    Owned(Buffer),
    /// The command buffer's index pool.
    Pooled(IndexPool),
}

/// One draw's pass, planned: what records before the render pass instance, and the draw in it.
pub(super) struct RewritePass {
    words: Upload,
    output: Output,
    /// The kernels of its three dispatches, in order (`RewritePhase::ALL`).
    kernels: [OwnedKernel; 3],
    /// The pass the kernels run: its workgroups.
    pass: IndexRewritePass,
    /// The set of the count and place dispatches, and the write dispatch's where it differs (a
    /// pooled list's: the block where the others bind the counter).
    sets: (OwnedSets, Option<OwnedSets>),
    /// The pass's reads of the Metal arguments and indices.
    uses: Vec<Use<ResourceKey>>,
    /// An indirect draw's Metal index offset and the buffer's length, where the offset lies at or
    /// past the length: every position draws index 0 there, where Metal reads memory.
    past_length: Option<(u64, u64)>,
}

impl RewritePass {
    /// The pass of `program` restarting at `restart` (`k'`) in `order` over the `index_size`-byte
    /// indices of `index` (`None` for a draw without indices), with `arguments`, in `core`'s
    /// command buffer, whose reads of the Metal buffers are `uses` ([`reads`]). Refused
    /// (`NotImplemented(gap)`: `RestartedStrip`, `PrivatePrimitives` for a private type's list,
    /// `ProvokingVertex` for a list of the last provoking vertex) where a direct draw's list is
    /// past what one storage range binds or a `u32` counts, or carries a base instance the
    /// device's indirect draws cannot (`drawIndirectFirstInstance`), and where the binding layer
    /// cannot bind the pass's ranges; and as buffers, uploads, kernels and descriptor sets are
    /// refused.
    pub(super) fn new(
        core: &mut EncoderCore,
        ((program, restart, order), refused): ((RewriteProgram, u32, RewriteOrder), RuntimeGap),
        index: Option<&DrawSource<'_>>,
        arguments: &Arguments<'_>,
        index_size: u64,
        uses: Vec<Use<ResourceKey>>,
    ) -> Result<RewritePass, Error> {
        let device = core.device().clone();
        let limits = &device.properties().description.limits;
        let gap = || Error::NotImplemented(refused.clone());
        let (source, owned) = match *arguments {
            Arguments::Direct {
                count,
                instances,
                base_vertex,
                base_instance,
            } => {
                if base_instance != 0
                    && !device
                        .enabled()
                        .features
                        .contains(&VkFeature::DrawIndirectFirstInstance)
                {
                    return Err(gap());
                }
                let capacity = program.capacity(u64::from(count));
                let bytes = capacity * 4;
                if bytes > u64::from(limits.max_storage_buffer_range) {
                    return Err(gap());
                }
                let capacity = u32::try_from(capacity).map_err(|_| gap())?;
                let source = RewriteSource::Direct {
                    count,
                    instances,
                    base_vertex,
                    base_instance,
                };
                (source, Some((capacity, bytes.max(16))))
            }
            Arguments::Indirect(indirect) => {
                let source = RewriteSource::Indirect {
                    arguments: IndirectRef::new(0, indirect.at),
                };
                (source, None)
            }
            Arguments::IndirectVertices(indirect) => {
                let source = RewriteSource::IndirectVertices {
                    arguments: IndirectRef::new(0, indirect.at),
                };
                (source, None)
            }
        };
        let output = match owned {
            Some((capacity, _)) => RewriteOutput::Owned { capacity },
            None => RewriteOutput::Pooled {
                pool_bytes: u32::try_from(device.shared().gpu_pools.index_bytes)
                    .map_err(|_| internal("an index pool past a u32's bytes"))?,
            },
        };
        // The indices the draw reads before where Metal's fetch stops (the source's bytes); a draw
        // without indices reads none.
        let terms = match index {
            Some(index) => (index_size, index.at, index.at + index.bound(index.at)),
            None => (4, 0, 0),
        };
        let pass = IndexRewritePass::new(
            source,
            restart,
            terms,
            output,
            limits.min_storage_buffer_offset_alignment,
            limits.max_storage_buffer_range,
        )
        .ok_or_else(gap)?
        .with_order(order);
        let kernel = |phase| device.owned_kernel(OwnedModule::IndexRewrite(program, phase));
        let kernels = [
            kernel(RewritePhase::Count)?,
            kernel(RewritePhase::Place)?,
            kernel(RewritePhase::Write)?,
        ];
        // A direct draw's list holds no value before the pass writes it: a buffer of the
        // device's, made without a submission.
        let output = match owned {
            Some((_, bytes)) => Output::Owned(device.scratch_buffer(bytes)?),
            None => {
                let queue = core.command_buffer().command_queue().id();
                Output::Pooled(core.command_buffer().state().index_pool(&device, queue)?)
            }
        };
        let (words, status) = {
            let mut state = core.command_buffer().state();
            let words =
                state
                    .uploads
                    .upload(&device, &pass.parameter_bytes(), IndexRewritePass::BYTES)?;
            (words, state.status_words(&device)?)
        };
        let region = |upload: &Upload, len: u64| (upload.raw(), upload.offset(), len);
        let words_range = region(&words, IndexRewritePass::BYTES);
        let mut set = [(vk::Buffer::null(), 0, 0); 5];
        set[IndexRewriteModule::WORDS_BINDING as usize] = words_range;
        // A direct draw reads no arguments, and a draw with no index inside the buffer no index:
        // nothing reads the binding, which takes the words.
        set[IndexRewriteModule::ARGUMENTS_BINDING as usize] =
            match (pass.arguments_range, arguments) {
                (
                    Some(range),
                    Arguments::Indirect(indirect) | Arguments::IndirectVertices(indirect),
                ) => (indirect.raw, range.offset, range.range),
                _ => words_range,
            };
        set[IndexRewriteModule::INDICES_BINDING as usize] = match (pass.indices_range, index) {
            (Some(range), Some(index)) => (index.raw, range.offset, range.range),
            _ => words_range,
        };
        set[IndexRewriteModule::STATUS_BINDING as usize] =
            region(&status, ExecutionStatus::BYTES as u64);
        // A pooled list's count and place dispatches bind the pool's counter, its write dispatch
        // the pool's block, whose indices start where the place dispatch's first word says.
        let (output_range, write_range) = match &output {
            Output::Owned(buffer) => {
                let (raw, at) = buffer.at(0);
                ((raw, at, buffer.length()), None)
            }
            Output::Pooled(pool) => (
                region(&pool.counter, IndexRewritePass::POOL_HEADER_BYTES),
                Some((pool.block, 0, pool.bytes)),
            ),
        };
        set[IndexRewriteModule::OUTPUT_BINDING as usize] = output_range;
        if set
            .iter()
            .chain(&write_range)
            .any(|&(buffer, _, _)| buffer == vk::Buffer::null())
        {
            return Err(internal("an index rewrite pass binding without a buffer"));
        }
        // The three's set layouts are defined alike: one set for all, or one for the write.
        let sets = (
            OwnedSets::write(&device, &kernels[0], &[set.to_vec()])?,
            match write_range {
                Some(range) => {
                    set[IndexRewriteModule::OUTPUT_BINDING as usize] = range;
                    Some(OwnedSets::write(&device, &kernels[2], &[set.to_vec()])?)
                }
                None => None,
            },
        );
        let past_length = match (arguments, index) {
            (Arguments::Indirect(_), Some(index)) => {
                let (buffer, offset) = index.metal();
                (offset >= buffer.length()).then(|| (offset, buffer.length()))
            }
            // A direct draw's positions past the buffer read a padded copy, which the draw
            // announces (`render.rs`); a draw without indices reads none.
            _ => None,
        };
        Ok(RewritePass {
            words,
            output,
            kernels,
            pass,
            sets,
            uses,
            past_length,
        })
    }

    /// The pass's reads of the Metal arguments and indices ([`reads`]).
    pub(super) fn uses(&self) -> &[Use<ResourceKey>] {
        &self.uses
    }

    /// The draw of the pass: its index binding, the `VkBuffer` of its arguments, and its command,
    /// with `sized` the device's index range binding.
    pub(super) fn draw(
        &self,
        sized: Option<SizedIndexBind>,
    ) -> (
        Option<IndexBind>,
        Option<vk::Buffer>,
        metal2vulkan::vk::DrawCommand,
    ) {
        let (buffer, offset, bytes) = match &self.output {
            Output::Owned(buffer) => {
                let (raw, at) = buffer.at(0);
                (raw, at, buffer.length())
            }
            Output::Pooled(pool) => (
                pool.block,
                IndexRewritePass::POOL_HEADER_BYTES,
                pool.index_bytes(),
            ),
        };
        let index_type = metal2vulkan::vk::IndexType::UINT32;
        let draw = metal2vulkan::vk::DrawCommand::DrawIndexedIndirect {
            index_type,
            index_buffer_offset: offset,
            offset: self.words.offset() + IndexRewritePass::DRAW_OFFSET,
            draw_count: 1,
            stride: 20,
        };
        let bind = IndexBind {
            buffer,
            offset,
            index_type: index_type.into(),
            sized: sized.map(|bind| (bind, bytes)),
        };
        (Some(bind), Some(self.words.raw()), draw)
    }

    /// Hold the pass's set and list in `core`'s command buffer until it completes, and announce
    /// an indirect draw from past its index buffer once; what is left records the dispatch, before
    /// the render pass or before the render pass instance the draw records into (`render.rs`).
    /// Call it once the draw records.
    pub(super) fn into_steps(self, core: &mut EncoderCore) -> RewriteSteps {
        let RewritePass {
            output,
            kernels,
            pass,
            sets,
            uses,
            past_length,
            ..
        } = self;
        let (sets, write) = sets;
        let steps = RewriteSteps {
            kernels,
            pass,
            sets: [
                sets.sets.clone(),
                sets.sets.clone(),
                write.as_ref().unwrap_or(&sets).sets.clone(),
            ],
            uses,
        };
        core.retain(sets);
        if let Some(write) = write {
            core.retain(write);
        }
        if let Output::Owned(buffer) = output {
            core.retain(buffer);
        }
        if let Some((offset, length)) = past_length {
            core.device().degrade(Degradation::ZeroedIndexFetch, || {
                (
                    format!(
                        "an indexed indirect draw from offset {offset} of an index buffer of \
                         {length} bytes"
                    ),
                    "its offset lies at or past the buffer's length, so each position draws \
                     index 0 plus the base vertex"
                        .to_owned(),
                )
            });
        }
        steps
    }
}

/// The list of a direct draw without indices of one of Metal's private primitive types, or under
/// the last provoking vertex on a device without it, which the CPU writes into the command
/// buffer's upload arena while the draw is encoded: its positions are its vertices, so the list
/// follows from its count alone ([`RewriteProgram::sequential_list`] in the draw's order, relative
/// to its first vertex), and the draw draws it from its vertex start as base vertex. The arena
/// holds it until the command buffer completes, and the host writes it before the draw's batch is
/// submitted (`upload.rs`).
pub(super) struct WrittenList {
    upload: Upload,
    indices: u32,
}

impl WrittenList {
    /// The list of `program` in `order` for a draw of `count` vertices from `vertex_start`, in
    /// `core`'s command buffer. Refused (`NotImplemented(gap)`: `PrivatePrimitives`, or
    /// `ProvokingVertex`) from a vertex start past `i32::MAX`, which a list's base vertex cannot
    /// carry, and for a list of more indices than a `u32` counts; and as uploads are refused.
    pub(super) fn new(
        core: &mut EncoderCore,
        ((program, order), refused): ((RewriteProgram, RewriteOrder), RuntimeGap),
        (vertex_start, count): (u32, u32),
    ) -> Result<WrittenList, Error> {
        let gap = || Error::NotImplemented(refused.clone());
        i32::try_from(vertex_start).map_err(|_| gap())?;
        let indices = u32::try_from(program.capacity(u64::from(count))).map_err(|_| gap())?;
        let device = core.device().clone();
        let bytes: Vec<u8> = program
            .sequential_list(count, order)
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let len = (bytes.len() as u64).max(4);
        let upload = core
            .command_buffer()
            .state()
            .uploads
            .upload(&device, &bytes, len)?;
        Ok(WrittenList { upload, indices })
    }

    /// The draw of the list: its index binding, with `sized` the device's index range binding,
    /// and the binder's draw of the vertices `draw` (`vkCmdDraw`'s), drawn as the indexed draw of
    /// the list from its first vertex as base vertex, with its instances. `None` for another
    /// command.
    pub(super) fn draw(
        &self,
        draw: metal2vulkan::vk::DrawCommand,
        sized: Option<SizedIndexBind>,
    ) -> Option<(IndexBind, metal2vulkan::vk::DrawCommand)> {
        let metal2vulkan::vk::DrawCommand::Draw {
            instance_count,
            first_vertex,
            first_instance,
            ..
        } = draw
        else {
            return None;
        };
        let index_type = metal2vulkan::vk::IndexType::UINT32;
        let offset = self.upload.offset();
        let draw = metal2vulkan::vk::DrawCommand::DrawIndexed {
            index_type,
            index_buffer_offset: offset,
            index_count: self.indices,
            instance_count,
            first_index: 0,
            vertex_offset: i32::try_from(first_vertex).ok()?,
            first_instance,
        };
        let bind = IndexBind {
            buffer: self.upload.raw(),
            offset,
            index_type: index_type.into(),
            sized: sized.map(|bind| (bind, (u64::from(self.indices) * 4).max(4))),
        };
        Some((bind, draw))
    }
}

/// The pass's reads of the Metal buffers: an indirect draw's, as the index bounds pass's
/// (`index_bounds::reads`); a direct draw's indices from its offset through its count, or to
/// where Metal's fetch stops; an indirect draw's arguments alone where it has no indices; in the
/// compute stage.
pub(super) fn reads(
    index: Option<&DrawSource<'_>>,
    arguments: &Arguments<'_>,
    index_size: u64,
) -> Vec<Use<ResourceKey>> {
    let read = Access::new(
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_STORAGE_READ,
    );
    let (index, count) = match (*arguments, index) {
        (Arguments::Indirect(indirect), Some(index)) => {
            return index_bounds::reads(index, indirect, index_size)
        }
        (Arguments::IndirectVertices(indirect), _) => {
            let (buffer, offset) = indirect.metal();
            let (identity, range) = buffer.tracked(offset..offset + 16);
            return vec![Use::reads(
                identity.key,
                identity.tracked,
                UseRange::Bytes(range),
                read,
            )];
        }
        (Arguments::Direct { count, .. }, Some(index)) => (index, u64::from(count)),
        _ => return Vec::new(),
    };
    let (indices, offset) = index.metal();
    let end = IndexBufferRef::new(0, indices.length()).fetch_end(offset);
    let whole = end.saturating_sub(offset) / index_size * index_size;
    let bytes = whole.min(count * index_size);
    if bytes == 0 {
        return Vec::new();
    }
    let (identity, range) = indices.tracked(offset..offset + bytes);
    vec![Use::reads(
        identity.key,
        identity.tracked,
        UseRange::Bytes(range),
        read,
    )]
}

/// Of [`reads`], those of the bytes the pass reads in place, which a write since the encoder's
/// last memory barrier races: an indirect draw's as the index bounds pass's
/// (`index_bounds::in_place`), its arguments alone where it has no indices unless they were
/// copied for a race; a direct draw's indices all (it copies none for a race).
pub(super) fn in_place(
    index: Option<&DrawSource<'_>>,
    arguments: &Arguments<'_>,
    index_size: u64,
) -> Vec<Use<ResourceKey>> {
    match (*arguments, index) {
        (Arguments::Indirect(indirect), Some(index)) => {
            index_bounds::in_place(index, indirect, index_size)
        }
        (Arguments::IndirectVertices(indirect), _) if indirect.racing() => Vec::new(),
        _ => reads(index, arguments, index_size),
    }
}

/// A planned pass's dispatch, its set held by the command buffer ([`RewritePass::into_steps`]).
pub(super) struct RewriteSteps {
    /// The kernels of its three dispatches, in order.
    kernels: [OwnedKernel; 3],
    /// The pass the kernels run: its workgroups.
    pass: IndexRewritePass,
    /// Each dispatch's sets, in order.
    sets: [Vec<vk::DescriptorSet>; 3],
    /// The pass's reads of the Metal arguments and indices.
    uses: Vec<Use<ResourceKey>>,
}

impl RewriteSteps {
    /// The pass's reads of the Metal arguments and indices.
    pub(super) fn uses(&self) -> &[Use<ResourceKey>] {
        &self.uses
    }

    /// Record the dispatches into `core`'s command buffer before the pass, between a barrier from
    /// everything before and one to everything after.
    pub(super) fn record(&self, core: &mut EncoderCore) {
        core.barrier(Barrier::everything());
        core.command(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            &self.uses,
            |raw, command_buffer| {
                // SAFETY: the encoder's command buffer is recording outside a render pass.
                unsafe { self.dispatch(raw, command_buffer) }
            },
        );
        core.barrier(Barrier::everything());
    }

    /// Record the dispatches into `command_buffer` between a barrier from everything before and one
    /// to everything after: before a render pass instance of the encoder's pass, after the
    /// instances before it (`render.rs`).
    ///
    /// # Safety
    ///
    /// `command_buffer` is the encoder's, recording outside a render pass, and only the encoder
    /// records it.
    pub(super) unsafe fn record_in(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: as the caller states; barriers need nothing more.
        unsafe {
            Barrier::everything().record(raw, command_buffer);
            self.dispatch(raw, command_buffer);
            Barrier::everything().record(raw, command_buffer);
        }
    }

    /// The pass's three dispatches, in order, each after the one before's writes
    /// (`Barrier::between_dispatches`): the count and write dispatches over the pass's groups,
    /// the place dispatch over one workgroup.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, and only one encoder records it.
    unsafe fn dispatch(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        let phases = self.kernels.iter().zip(RewritePhase::ALL).zip(&self.sets);
        for (at, ((kernel, phase), sets)) in phases.enumerate() {
            // SAFETY: as the caller states; each kernel is the device's, its sets (whose layout
            // each kernel's is defined as) are live until the command buffer completes (retained
            // by `into_steps`), and so are the uploads, the list and the Metal buffers they name.
            unsafe {
                if at > 0 {
                    Barrier::between_dispatches().record(raw, command_buffer);
                }
                raw.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    kernel.pipeline,
                );
                raw.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    kernel.layout,
                    0,
                    sets,
                    &[],
                );
                raw.cmd_dispatch(command_buffer, phase.workgroups(&self.pass), 1, 1);
            }
        }
    }
}

/// A list a draw draws in place of its Metal primitives, which the CPU or a run of the index gather
/// pass wrote (the index rewrite pass's draws apart, `RewritePass::draw`).
pub(super) enum Listed {
    /// A direct draw's without indices, the CPU's ([`WrittenList`]).
    Written(WrittenList),
    /// A direct indexed draw's, a gather run's ([`GatheredList`]).
    Gathered(GatheredList),
}

impl Listed {
    /// The draw of the list: its index binding, with `sized` the device's index range binding,
    /// and the indexed draw of it the binder's draw `draw` becomes. `None` for a command of
    /// another kind than the list's draw takes.
    pub(super) fn draw(
        &self,
        draw: metal2vulkan::vk::DrawCommand,
        sized: Option<SizedIndexBind>,
    ) -> Option<(IndexBind, metal2vulkan::vk::DrawCommand)> {
        match self {
            Listed::Written(list) => list.draw(draw, sized),
            Listed::Gathered(list) => list.draw(draw, sized),
        }
    }
}

/// The output words of a run's own buffer, where its first draw's list is shorter.
const RUN_OUTPUT_WORDS: u32 = 1 << 14;

/// The positions a run's list holds, where its first draw's list is shorter.
const RUN_POSITIONS: u32 = 1536;

/// The most draws of a run: its table's records, which fit an ordinary chunk of the upload arena
/// (`upload.rs`).
const RUN_DRAWS: u32 = 480;

/// The most runs an encoder keeps open before it records them.
const OPEN_RUNS: usize = 8;

/// A direct indexed draw whose list the index gather pass writes (`bind::index_gather_module`):
/// its program and order, its Metal index source, Metal's count, and its index size.
pub(super) struct GatherRequest<'a> {
    pub(super) program: RewriteProgram,
    pub(super) order: RewriteOrder,
    pub(super) index: &'a DrawSource<'a>,
    pub(super) count: u32,
    pub(super) wide: bool,
}

/// The runs of an encoder's gathered draws (the module docs, `index_rewrite.rs`'s gather): each
/// one dispatch over one index buffer, recorded before the encoder's next command, between a
/// barrier from everything before and one to everything after for all of them.
#[derive(Default)]
pub(super) struct GatherRuns {
    runs: Vec<GatherRun>,
}

/// One run: its draws, the index `VkBuffer` and the bytes it binds of it, its table's region,
/// and its kernel and set.
pub(super) struct GatherRun {
    gather: IndexGather,
    key: (vk::Buffer, u64, u64, RewriteProgram, RewriteOrder),
    table: Upload,
    kernel: OwnedKernel,
    sets: Vec<vk::DescriptorSet>,
    output: (vk::Buffer, u64),
}

/// Where a gathered draw's list lies: its `VkBuffer`, its offset there, and its indices.
pub(super) struct GatheredList {
    buffer: vk::Buffer,
    offset: u64,
    indices: u32,
}

/// What a new run's command buffer holds until it completes: its output and its set.
pub(super) struct GatherHeld {
    pub(super) output: Buffer,
    pub(super) sets: OwnedSets,
}

impl GatherRuns {
    /// Whether no run is open.
    pub(super) fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// Whether the runs are as many as an encoder keeps open: they record before another opens.
    pub(super) fn full(&self) -> bool {
        self.runs.len() >= OPEN_RUNS
    }

    /// The open runs, which record now ([`record_runs`]).
    pub(super) fn take(&mut self) -> Vec<GatherRun> {
        self.runs.drain(..).collect()
    }

    /// `request`'s draw joins an open run of its index buffer, program and order that takes it,
    /// or a new one, with its table's region in `uploads`: the list's place, and what a new run's
    /// command buffer holds. `None` where no run takes the draw (its list is past what a run's
    /// dispatch or bindings take), which the index rewrite pass then writes. Refused as uploads,
    /// buffers, kernels and descriptor sets are refused.
    pub(super) fn join(
        &mut self,
        device: &Device,
        uploads: &mut Uploads,
        request: GatherRequest<'_>,
    ) -> Result<Option<(GatheredList, Option<GatherHeld>)>, Error> {
        let GatherRequest {
            program,
            order,
            index,
            count,
            wide,
        } = request;
        // The run binds the Metal buffer's bytes, from its start, through where Metal's fetch
        // stops (its length rounded up to 4, `IndexBufferRef::fetch_end`).
        let start = index.at - index.metal().1;
        let draw = GatherDraw {
            count,
            index_offset: index.at,
            index_end: index.at + index.bound(index.at),
            wide,
        };
        let key = (index.raw, start, index.length, program, order);
        let joined = self
            .runs
            .iter_mut()
            .filter(|run| run.key == key && run.gather.len() < RUN_DRAWS as usize)
            .find_map(|run| {
                let first = run.gather.push(draw)?;
                Some(run.list(first, &draw))
            });
        if let Some(list) = joined {
            return Ok(Some((list, None)));
        }
        // A run's output and positions hold its first draw's list, at least the defaults.
        let Ok(capacity) = u32::try_from(program.capacity(u64::from(count))) else {
            return Ok(None);
        };
        let words = capacity.max(RUN_OUTPUT_WORDS);
        let limits = &device.properties().description.limits;
        let Some(mut gather) = IndexGather::new(
            (program, order),
            (start, index.length),
            (
                limits.min_storage_buffer_offset_alignment,
                limits.max_storage_buffer_range,
            ),
            (words, capacity.max(RUN_POSITIONS)),
        ) else {
            return Ok(None);
        };
        let Some(first) = gather.push(draw) else {
            return Ok(None);
        };
        let output_bytes = u64::from(words) * 4;
        if output_bytes > u64::from(limits.max_storage_buffer_range) {
            return Ok(None);
        }
        let positions: Vec<u8> = gather
            .positions_list()
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let table_bytes = u64::from(
            IndexGatherModule::HEADER_WORDS + IndexGatherModule::RECORD_WORDS * RUN_DRAWS,
        ) * 4;
        let table = uploads.upload(device, &[], table_bytes)?;
        let positions = uploads.upload(device, &positions, (positions.len() as u64).max(4))?;
        let output = device.scratch_buffer(output_bytes)?;
        let kernel = device.owned_kernel(OwnedModule::IndexGather)?;
        let region = |upload: &Upload| (upload.raw(), upload.offset(), upload.len());
        let (output_raw, output_at) = output.at(0);
        let mut set = [(vk::Buffer::null(), 0, 0); 4];
        set[IndexGatherModule::TABLE_BINDING as usize] = region(&table);
        set[IndexGatherModule::POSITIONS_BINDING as usize] = region(&positions);
        // A run whose draws read no index binds any 4 bytes there.
        set[IndexGatherModule::INDICES_BINDING as usize] = match gather.indices_range() {
            Some(range) => (index.raw, range.offset, range.range),
            None => region(&table),
        };
        set[IndexGatherModule::OUTPUT_BINDING as usize] = (output_raw, output_at, output_bytes);
        let sets = OwnedSets::write(device, &kernel, &[set.to_vec()])?;
        let run = GatherRun {
            gather,
            key,
            table,
            kernel,
            sets: sets.sets.clone(),
            output: (output_raw, output_at),
        };
        let list = run.list(first, &draw);
        self.runs.push(run);
        Ok(Some((list, Some(GatherHeld { output, sets }))))
    }
}

impl GatherRun {
    /// The place of the run's list whose first word is `first`, of `draw`.
    fn list(&self, first: u32, draw: &GatherDraw) -> GatheredList {
        let (buffer, at) = self.output;
        GatheredList {
            buffer,
            offset: at + u64::from(first) * 4,
            indices: self.gather.capacity(*draw).unwrap_or(0),
        }
    }
}

impl GatheredList {
    /// The draw of the list: its index binding, with `sized` the device's index range binding,
    /// and the binder's draw of the indices `draw` (`vkCmdDrawIndexed`'s), drawn as the indexed
    /// draw of the list from its first index, with its instances and bases. `None` for another
    /// command.
    pub(super) fn draw(
        &self,
        draw: metal2vulkan::vk::DrawCommand,
        sized: Option<SizedIndexBind>,
    ) -> Option<(IndexBind, metal2vulkan::vk::DrawCommand)> {
        let metal2vulkan::vk::DrawCommand::DrawIndexed {
            instance_count,
            vertex_offset,
            first_instance,
            ..
        } = draw
        else {
            return None;
        };
        let index_type = metal2vulkan::vk::IndexType::UINT32;
        let draw = metal2vulkan::vk::DrawCommand::DrawIndexed {
            index_type,
            index_buffer_offset: self.offset,
            index_count: self.indices,
            instance_count,
            first_index: 0,
            vertex_offset,
            first_instance,
        };
        let bind = IndexBind {
            buffer: self.buffer,
            offset: self.offset,
            index_type: index_type.into(),
            sized: sized.map(|bind| (bind, (u64::from(self.indices) * 4).max(4))),
        };
        Some((bind, draw))
    }
}

/// Record `runs` into `command_buffer`: each run's table written into its region, then between a
/// barrier from everything before and one to everything after, each run's dispatch.
///
/// # Safety
///
/// `command_buffer` is recording outside a render pass, and only one encoder records it; each
/// run's kernel is the device's and its set and output are held until the command buffer
/// completes ([`GatherHeld`]), and its table's region is the command buffer's.
pub(super) unsafe fn record_runs(raw: &Vk, command_buffer: vk::CommandBuffer, runs: &[GatherRun]) {
    for run in runs {
        run.table.write(&run.gather.table_bytes());
    }
    // SAFETY: as the caller states.
    unsafe {
        Barrier::everything().record(raw, command_buffer);
        for run in runs {
            raw.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                run.kernel.pipeline,
            );
            raw.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                run.kernel.layout,
                0,
                &run.sets,
                &[],
            );
            raw.cmd_dispatch(command_buffer, run.gather.groups(), 1, 1);
        }
        Barrier::everything().record(raw, command_buffer);
    }
}
