//! The runtime side of the case vocabulary: `metal2vulkan_runtime` on whatever Vulkan device the
//! host has.
//!
//! Every call answers the runtime's typed error where it refuses, so a case records what it got
//! rather than panicking, and the comparison can tell our refusal from the venue's honest gap.
//!
//! No wait here can hang: a command buffer is waited for with a bound, and a wait that runs out
//! answers `NeverReturned` as a recorded fact rather than blocking the suite.

use crate::case::{
    Answer, ArgumentSlot, Bindings, CommandDraw, Failure, Fill, FragmentFill, FragmentFilled,
    Handle, IndirectReadBack, MeshCommand, MeshFunctions, MeshStep, PassOutcome, PlaneTexture,
    RateMapFacts, RateMapRequest, SamplerSpec, ShaderSource, Side, StageInRegion, StageInput,
    TextureRole, TextureSpec, TileFacts, TilePipelineSpec, TileStep, Venue,
};
use crate::queue_slots::QueueSlots;
use crate::render::{
    ColorState, RenderPass, RenderPipelineSpec, RenderStep, RenderWork, StageFunction, TargetSpec,
};
use crate::sparse_query::{
    HeapFacts, QueryPage, SparseHeapSpec, SparseTextureFacts, SparseTextureSpec, SparseTileQuery,
    TileSize,
};
use crate::texture_query::{HeapTextureQuery, SizeAndAlign, Storage};
use crate::transcript::{Negative, Value};
use metal2vulkan::bind::{RateMapDescriptor, RateMapLayerDescriptor, SamplerDescriptor};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    BarrierScope, BindingAccess, IndexType, IndirectCommandType, PixelFormat, PrimitiveType,
    RenderStages, ResourceUsage, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, MeshPipelineDescriptor, RenderPipelineDescriptor,
    StageInputAttributeDescriptor, StageInputDescriptor, StageInputLayoutDescriptor, StageSource,
    TileRenderPipelineDescriptor, VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
    VertexDescriptor,
};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus, CommandQueue,
    ComputePipelineState, DepthStencilState, Device, DrawPrimitiveType, Heap, HeapDescriptor,
    HeapType, IndirectCommand, IndirectCommandBuffer, IndirectCommandBufferDescriptor,
    IndirectDraw, IoSurfacePlane, IoSurfaceYCbCr, Library, LoadAction, Origin, PipelineOption,
    PipelineOptions, RasterizationRateMap, Region, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, ResourceRef, SamplerState, SamplerStateOptions, Size, StoreAction, Texture,
    TextureDescriptor, Viewport,
};
use std::time::Duration;

/// The longest the side waits for a command buffer. A wait that runs out is recorded, not retried.
const WAIT: Duration = Duration::from_secs(60);

/// The bytes the handler-wait probe's command buffer fills, so it has real work to be scheduled and
/// completed for rather than being an empty commit.
const HANDLER_FILL: u64 = 64;

/// The runtime side: the device, the queue the case's command buffers come from, and the objects
/// the case has made, by handle.
pub struct RuntimeSide<'a> {
    device: &'a Device,
    queue: Option<CommandQueue>,
    libraries: Vec<Library>,
    pipelines: Vec<ComputePipelineState>,
    buffers: Vec<Buffer>,
    textures: Vec<Texture>,
    /// The sampler states the case has made. Its own pool: a sampler state is its own runtime
    /// object, not another name for one of the pools above.
    samplers: Vec<SamplerState>,
    /// The rasterization-rate maps the case has made. Its own pool: a rate map is its own
    /// runtime object, not another name for one of the pools above.
    rate_maps: Vec<RasterizationRateMap>,
    /// For each argument buffer the case has made, the textures it names: a resource a kernel
    /// reaches through an argument buffer rather than through a slot is resident only where the
    /// encoder was told so, so binding the buffer makes them resident. Not a pool — the buffer
    /// itself lives in `buffers`.
    argument_residents: Vec<(Handle, Vec<Handle>)>,
    command_buffers: Vec<CommandBuffer>,
    /// The render pipeline states the case has made, each with whether its creation was asked
    /// for a reflection. One pool, because a mesh state, a tile state and a graphics state are
    /// one runtime object (`RenderPipelineState`) made from three descriptors.
    ///
    /// The runtime answers `texture_access` off the state itself, so nothing is kept beside it
    /// here; the Metal side has to hold the reflection its creation answered, because Metal
    /// answers one only then.
    render_pipelines: Vec<RenderPipelineState>,
    /// The indirect command buffers the case has made.
    command_buffers_indirect: Vec<IndirectCommandBuffer>,
    /// The depth and stencil states the case has made. Its own pool: a `DepthStencilState` is its
    /// own runtime object, not another name for one of the pools above.
    depth_stencil_states: Vec<DepthStencilState>,
    /// The heaps the case has made. Its own pool: a `Heap` is its own runtime object, not another
    /// name for one of the pools above, and not another name for the depth and stencil states
    /// either.
    heaps: Vec<Heap>,
    /// Which buffer stands for the surface behind a texture handle that lies over an IOSurface
    /// plane.
    ///
    /// **A lookup, not a pool, and not another name for `heaps` or `buffers`.** The buffer it
    /// names is a case's own and already lives in `buffers`; what this adds is the way back from
    /// the texture handle to it, which a rotated target's plane read and write need and nothing
    /// else does. There is no surface on this side at all — the stand-in buffer is the whole point
    /// of the family, and is why these cases run where there is no IOSurface.
    planes: Vec<(Handle, Buffer)>,
    /// The degrade-notice codes the device has delivered so far, which `announced` takes away.
    notices: Box<dyn FnMut() -> Vec<String> + 'a>,
}

impl<'a> RuntimeSide<'a> {
    /// The side on `device`, taking the notices it announced from `notices` (the lane's message
    /// sink, which hands them over and clears itself).
    pub fn new(device: &'a Device, notices: impl FnMut() -> Vec<String> + 'a) -> Self {
        Self {
            device,
            queue: None,
            libraries: Vec::new(),
            pipelines: Vec::new(),
            buffers: Vec::new(),
            textures: Vec::new(),
            samplers: Vec::new(),
            rate_maps: Vec::new(),
            argument_residents: Vec::new(),
            command_buffers: Vec::new(),
            render_pipelines: Vec::new(),
            command_buffers_indirect: Vec::new(),
            depth_stencil_states: Vec::new(),
            heaps: Vec::new(),
            planes: Vec::new(),
            notices: Box::new(notices),
        }
    }

    /// The device the side runs on, for a case that reads an answer no portable call covers yet.
    pub fn device(&self) -> &Device {
        self.device
    }

    /// The queue the case's command buffers come from, made on first use.
    fn queue(&mut self) -> Answer<&CommandQueue> {
        if self.queue.is_none() {
            self.queue = Some(
                self.device
                    .new_command_queue()
                    .map_err(|error| Failure::of(&error))?,
            );
        }
        Ok(self.queue.as_ref().expect("the queue was just made"))
    }

    /// The buffer `handle` names.
    fn held(&self, handle: Handle) -> Answer<&Buffer> {
        self.buffers
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("buffer {}", handle.0)))
    }

    /// Level 0 of `texture` read back through a blit into a shared buffer of its own, at
    /// `row_pitch` bytes a row: what a private texture's bytes are, since the host reaches none of
    /// them directly.
    fn blit_out(&mut self, texture: &Texture, size: [u32; 2], row_pitch: u32) -> Answer<Vec<u8>> {
        let length = u64::from(row_pitch) * u64::from(size[1]);
        let back = self
            .device
            .new_buffer(length.max(1), ResourceOptions::shared())
            .map_err(Failure::from_error)?;
        let command_buffer = self
            .queue()?
            .command_buffer()
            .map_err(Failure::from_error)?;
        let mut blit = command_buffer
            .blit_command_encoder()
            .map_err(Failure::from_error)?;
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(size[0].into(), size[1].into(), 1),
            &back,
            0,
            row_pitch.into(),
            length,
            BlitOption::None,
        )
        .map_err(Failure::from_error)?;
        blit.end_encoding().map_err(Failure::from_error)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        if command_buffer.wait_until_completed_for(WAIT).is_err() {
            return Err(Failure::named(
                "ReadBackNeverReturned",
                "the blit of the target into a buffer did not complete",
            ));
        }
        let handle = keep(&mut self.buffers, back);
        keep(&mut self.command_buffers, command_buffer);
        self.read_buffer(handle, 0, length as usize)
    }

    /// The descriptor of a mesh render pipeline of `library`'s `functions`.
    fn mesh_descriptor(
        &self,
        library: Handle,
        functions: &MeshFunctions<'_>,
    ) -> Answer<MeshPipelineDescriptor> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let stage = |name: &str| -> Answer<StageSource> {
            Ok(StageSource::Function(
                library
                    .new_function_with_name(name)
                    .map_err(Failure::from_error)?
                    .function_descriptor(),
            ))
        };
        let mut descriptor = MeshPipelineDescriptor::new(stage(functions.mesh)?);
        descriptor.object = functions.object.map(stage).transpose()?;
        descriptor.fragment = Some(stage(functions.fragment)?);
        for attachment in &mut descriptor.color_attachments[..functions.attachments as usize] {
            *attachment = Some(ColorAttachmentDescriptor::new(TARGET_PIXEL_FORMAT));
        }
        Ok(descriptor)
    }

    /// The texture `handle` names.
    fn held_texture(&self, handle: Handle) -> Answer<&Texture> {
        self.textures
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("texture {}", handle.0)))
    }

    /// The heap `handle` names.
    fn held_heap(&self, handle: Handle) -> Answer<&Heap> {
        self.heaps
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("heap {}", handle.0)))
    }

    /// The sampler state `handle` names.
    fn held_sampler(&self, handle: Handle) -> Answer<&SamplerState> {
        self.samplers
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("sampler {}", handle.0)))
    }

    /// The rasterization-rate map `handle` names.
    fn held_rate_map(&self, handle: Handle) -> Answer<&RasterizationRateMap> {
        self.rate_maps
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("rate map {}", handle.0)))
    }

    /// One command buffer with one compute encoder: the pipeline set, `bindings` bound at 0 up,
    /// each of `stage_in` set in order, `dispatch` encoded, and `commit`. The two dispatch calls
    /// differ in that one line alone, so they share this body and a case cannot pick up a
    /// different binding order with one of them.
    fn encode(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        stage_in: &[StageInRegion],
        dispatch: impl FnOnce(
            &mut metal2vulkan_runtime::ComputeCommandEncoder,
        ) -> Result<(), metal2vulkan_runtime::Error>,
    ) -> Answer<Handle> {
        let bound: Vec<Buffer> = bindings
            .buffers
            .iter()
            .map(|handle| self.held(*handle).cloned())
            .collect::<Answer<Vec<Buffer>>>()?;
        let sampled: Vec<Texture> = bindings
            .textures
            .iter()
            .map(|handle| self.held_texture(*handle).cloned())
            .collect::<Answer<Vec<Texture>>>()?;
        let states: Vec<SamplerState> = bindings
            .samplers
            .iter()
            .map(|handle| self.held_sampler(*handle).cloned())
            .collect::<Answer<Vec<SamplerState>>>()?;
        // The textures the bound argument buffers name: Metal keeps a resource reached through an
        // argument buffer resident only where the encoder was told so.
        let resident: Vec<Texture> = self
            .argument_residents
            .iter()
            .filter(|(buffer, _)| bindings.buffers.contains(buffer))
            .flat_map(|(_, textures)| textures.clone())
            .map(|handle| self.held_texture(handle).cloned())
            .collect::<Answer<Vec<Texture>>>()?;
        let state = self
            .pipelines
            .get(pipeline.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", pipeline.0)))?
            .clone();
        let command_buffer = self
            .queue()?
            .command_buffer()
            .map_err(Failure::from_error)?;
        let mut encoder = command_buffer
            .compute_command_encoder()
            .map_err(Failure::from_error)?;
        encoder
            .set_compute_pipeline_state(&state)
            .map_err(Failure::from_error)?;
        for (at, buffer) in bound.iter().enumerate() {
            encoder
                .set_buffer(Some(buffer), 0, at as u32)
                .map_err(Failure::from_error)?;
        }
        for (at, texture) in sampled.iter().enumerate() {
            encoder
                .set_texture(Some(texture), at as u32)
                .map_err(Failure::from_error)?;
        }
        for (at, sampler) in states.iter().enumerate() {
            encoder
                .set_sampler_state(Some(sampler), at as u32)
                .map_err(Failure::from_error)?;
        }
        for texture in &resident {
            encoder
                .use_resource(ResourceRef::Texture(texture), ResourceUsage::READ)
                .map_err(Failure::from_error)?;
        }
        for region in stage_in {
            match region {
                StageInRegion::Direct { origin, size } => encoder
                    .set_stage_in_region(Region::new_3d(
                        origin[0].into(),
                        origin[1].into(),
                        origin[2].into(),
                        size[0].into(),
                        size[1].into(),
                        size[2].into(),
                    ))
                    .map_err(Failure::from_error)?,
                StageInRegion::Indirect { buffer, offset } => encoder
                    .set_stage_in_region_indirect(self.held(*buffer)?, *offset)
                    .map_err(Failure::from_error)?,
            }
        }
        dispatch(&mut encoder).map_err(Failure::from_error)?;
        encoder.end_encoding().map_err(Failure::from_error)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        Ok(keep(&mut self.command_buffers, command_buffer))
    }
}

/// The usage a texture's role is.
fn usage_of(role: TextureRole) -> TextureUsage {
    match role {
        TextureRole::Read => TextureUsage::SHADER_READ,
        TextureRole::Write => TextureUsage::SHADER_WRITE,
        TextureRole::ReadWrite => TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        TextureRole::RenderTarget => TextureUsage::RENDER_TARGET,
    }
}

/// Three axes as the runtime's size.
fn size(axes: [u32; 3]) -> Size {
    Size::new(axes[0].into(), axes[1].into(), axes[2].into())
}

/// Every field of a case's [`RenderPipelineSpec`] **except its two stages**, as the runtime side
/// states them on the pipeline descriptor.
///
/// It exists to be testable without a device. A whole `RenderPipelineDescriptor` cannot be built
/// without one -- its `new` takes a `StageSource`, and neither a `Function` nor a
/// `CompiledFunction` can be made without a device -- so the stage-independent fields are mapped
/// into this value first and copied onto the descriptor after. That is not a mock: it is the value
/// the real path builds and uses.
///
/// The guard over it is `tests::the_runtime_side_forwards_every_field_of_a_pipeline_spec`, and the
/// reason it is wanted is that this side has now twice silently failed to state a field the Metal
/// side states: `write_mask`, and then `depth_format` and `stencil_format`.
pub(crate) struct PipelineFields {
    pub color_attachments: [Option<ColorAttachmentDescriptor>; 8],
    pub depth_format: Option<metal2vulkan::metal::PixelFormat>,
    pub stencil_format: Option<metal2vulkan::metal::PixelFormat>,
    pub raster_sample_count: u32,
    pub rasterization_enabled: bool,
    pub vertex_descriptor: Option<VertexDescriptor>,
}

/// [`PipelineFields`] of `spec`: `Err` where the spec names a colour attachment past the eight a
/// Metal pipeline holds.
pub(crate) fn pipeline_fields(spec: &RenderPipelineSpec<'_>) -> Answer<PipelineFields> {
    let mut color_attachments: [Option<ColorAttachmentDescriptor>; 8] = Default::default();
    for (at, color) in spec.colors.iter().enumerate() {
        let Some(color) = color else { continue };
        let attachment = color_attachments
            .get_mut(at)
            .ok_or_else(|| Failure::named("NoSuchAttachment", format!("color {at}")))?;
        *attachment = Some(color_descriptor(color));
    }
    Ok(PipelineFields {
        color_attachments,
        depth_format: spec.depth_format,
        stencil_format: spec.stencil_format,
        raster_sample_count: spec.samples,
        rasterization_enabled: spec.rasterization,
        vertex_descriptor: spec.vertex_layout.as_ref().map(vertex_descriptor),
    })
}

/// The runtime's vertex descriptor for `layout`: the same attribute indices, buffer indices and
/// step functions the case states, so both sides fetch the same way.
fn vertex_descriptor(layout: &crate::render::VertexLayout<'_>) -> VertexDescriptor {
    let mut descriptor = VertexDescriptor::new();
    for attribute in layout.attributes {
        descriptor.attributes.insert(
            attribute.index,
            VertexAttributeDescriptor::new(attribute.format, attribute.offset, attribute.buffer),
        );
    }
    for buffer in layout.layouts {
        let mut held = VertexBufferLayoutDescriptor::new(Some(buffer.stride));
        held.step_function = buffer.step;
        held.step_rate = buffer.rate;
        descriptor.layouts.insert(buffer.index, held);
    }
    descriptor
}

/// The runtime's stage-input descriptor for `stage_input`: the same attribute indices, buffer
/// indices, formats, strides, steps and rates, since the case's vocabulary is Metal's own.
fn stage_input_descriptor(stage_input: &StageInput) -> StageInputDescriptor {
    let mut descriptor = StageInputDescriptor::new();
    for attribute in &stage_input.attributes {
        descriptor.attributes.insert(
            attribute.index,
            StageInputAttributeDescriptor::new(
                attribute.format,
                attribute.offset,
                attribute.buffer_index,
            ),
        );
    }
    for layout in &stage_input.layouts {
        let mut written = StageInputLayoutDescriptor::new(layout.stride);
        written.step_function = layout.step_function;
        written.step_rate = layout.step_rate;
        descriptor.layouts.insert(layout.buffer_index, written);
    }
    descriptor.index_type = stage_input.index_type;
    descriptor.index_buffer_index = stage_input.index_buffer_index;
    descriptor
}

/// `device`'s library of AIR bitcode: the translator parses it, and the device takes the parsed
/// library. The runtime's own `new_library_with_data` takes a metallib alone, as Metal's
/// `newLibraryWithData:` does, and the suite's shaders are AIR.
fn new_library_with_air(
    device: &Device,
    bytes: &[u8],
) -> Result<Library, metal2vulkan_runtime::Error> {
    let library = metal2vulkan::library::Library::parse(bytes).map_err(|error| match error {
        metal2vulkan::error::Error::Unsupported(unsupported) => {
            metal2vulkan_runtime::Error::Unsupported(unsupported)
        }
        other => metal2vulkan_runtime::Error::Compile(other),
    })?;
    Ok(device.new_library_with_compile_library(library))
}

/// `slots`'s handle at `at` pushed onto `kept`, answering its index.
fn keep<T>(kept: &mut Vec<T>, value: T) -> Handle {
    kept.push(value);
    Handle(u32::try_from(kept.len() - 1).expect("fewer than 2^32 objects"))
}

/// The variant a command buffer's status records as.
fn status_value(status: CommandBufferStatus) -> Value {
    Value::unit_variant(format!("{status:?}"))
}

/// `length` bytes from `offset` of a shared buffer the probe made itself, for an event probe whose
/// buffers never enter the case's handle pool.
fn read_shared(buffer: &Buffer, offset: u64, length: usize) -> Result<Vec<u8>, Failure> {
    let contents = buffer
        .contents()
        .ok_or_else(|| Failure::named("NoContents", "a buffer the host does not reach"))?;
    // SAFETY: the probe made the buffer at least `offset + length` bytes long, the host reaches
    // them, and the caller waited for the command buffers that wrote them.
    let bytes =
        unsafe { std::slice::from_raw_parts(contents.as_ptr().add(offset as usize), length) };
    Ok(bytes.to_vec())
}

/// The names a command buffer's handlers logged, in the order they were called, with each name
/// also sent so the probe can wait for one rather than guess when the log is whole.
///
/// The channel is what makes the log readable: the probe waits for the name it cares about to
/// arrive instead of resting on when a wait of its own returns relative to a handler, which is one
/// of the very orders a case over this family is there to measure.
struct HandlerLog {
    called: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    sender: std::sync::mpsc::Sender<String>,
    arrived: std::sync::mpsc::Receiver<String>,
}

impl HandlerLog {
    /// An empty log.
    fn new() -> Self {
        let (sender, arrived) = std::sync::mpsc::channel();
        Self {
            called: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            sender,
            arrived,
        }
    }

    /// A handler that logs itself as `name`.
    fn handler(&self, name: &'static str) -> metal2vulkan_runtime::Handler {
        let (called, sender) = (std::sync::Arc::clone(&self.called), self.sender.clone());
        Box::new(move |_: &CommandBuffer| {
            if let Ok(mut called) = called.lock() {
                called.push(name.to_owned());
            }
            let _ = sender.send(name.to_owned());
        })
    }

    /// Wait until the handler named `name` has been called, or `bound` runs out.
    fn wait_for(&self, name: &str, bound: Duration) {
        let deadline = std::time::Instant::now() + bound;
        loop {
            if self
                .called
                .lock()
                .is_ok_and(|called| called.iter().any(|called| called == name))
            {
                return;
            }
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() || self.arrived.recv_timeout(left).is_err() {
                return;
            }
        }
    }

    /// The names logged so far, in order.
    fn names(&self) -> Vec<String> {
        self.called
            .lock()
            .map(|called| called.clone())
            .unwrap_or_default()
    }
}

impl Side for RuntimeSide<'_> {
    fn venue(&self) -> Venue {
        let properties = self.device.properties();
        Venue {
            device: properties.name.clone(),
            driver: format!(
                "vendor {:#06x} device {:#06x}",
                properties.vendor_id, properties.device_id
            ),
            api: format!(
                "{}.{}",
                properties.api_version.major, properties.api_version.minor
            ),
            metal_level: properties.metal_level.macos().to_owned(),
        }
    }

    fn library(&mut self, source: &ShaderSource<'_>) -> Answer<Handle> {
        let library = new_library_with_air(self.device, source.bitcode)
            .map_err(|error| Failure::of(&error))?;
        Ok(keep(&mut self.libraries, library))
    }

    fn compute_pipeline(&mut self, library: Handle, entry: &str) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .new_function_with_name(entry)
            .map_err(|error| Failure::of(&error))?;
        let descriptor = metal2vulkan::pipeline::ComputePipelineDescriptor::new(
            metal2vulkan::pipeline::StageSource::Function(function.function_descriptor()),
        );
        let pipeline = self
            .device
            .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
            .map_err(|error| Failure::of(&error))?;
        Ok(keep(&mut self.pipelines, pipeline))
    }

    fn static_threadgroup_memory_length(&mut self, pipeline: Handle) -> Answer<u64> {
        let state = self
            .pipelines
            .get(pipeline.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", pipeline.0)))?;
        Ok(u64::from(state.static_threadgroup_memory_length()))
    }

    fn buffer(&mut self, bytes: &[u8]) -> Answer<Handle> {
        let handle = self.zeroed_buffer(bytes.len())?;
        let buffer = self.held(handle)?;
        let contents = buffer
            .contents()
            .ok_or_else(|| Failure::named("NoContents", "a buffer the host does not reach"))?;
        // SAFETY: the buffer was just made with at least `bytes.len()` bytes, the host reaches
        // them, and no command uses it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len());
        }
        Ok(handle)
    }

    fn zeroed_buffer(&mut self, length: usize) -> Answer<Handle> {
        let buffer = self
            .device
            .new_buffer(length.max(1) as u64, ResourceOptions::shared())
            .map_err(|error| Failure::of(&error))?;
        Ok(keep(&mut self.buffers, buffer))
    }

    fn texture_2d(
        &mut self,
        format: PixelFormat,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<Handle> {
        let mut descriptor =
            TextureDescriptor::texture_2d(format, size[0].into(), size[1].into(), false);
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        descriptor.resource_options = ResourceOptions::shared();
        let texture = self
            .device
            .new_texture(&descriptor)
            .map_err(|error| Failure::of(&error))?;
        texture
            .replace_region(
                Region::new_2d(0, 0, size[0].into(), size[1].into()),
                0,
                0,
                bytes,
                row_pitch.into(),
                0,
            )
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.textures, texture))
    }

    fn read_texture(&mut self, texture: Handle, size: [u32; 2], row_pitch: u32) -> Answer<Vec<u8>> {
        let texture = self.held_texture(texture)?;
        let mut bytes = vec![0u8; row_pitch as usize * size[1] as usize];
        texture
            .get_bytes(
                &mut bytes,
                row_pitch.into(),
                0,
                Region::new_2d(0, 0, size[0].into(), size[1].into()),
                0,
                0,
            )
            .map_err(Failure::from_error)?;
        Ok(bytes)
    }

    fn texture_2d_in_role(&mut self, spec: &TextureSpec, bytes: &[u8]) -> Answer<Handle> {
        let mut descriptor = TextureDescriptor::texture_2d(
            spec.format,
            spec.size[0].into(),
            spec.size[1].into(),
            false,
        );
        descriptor.usage = usage_of(spec.role);
        descriptor.resource_options = ResourceOptions::shared();
        let texture = self
            .device
            .new_texture(&descriptor)
            .map_err(|error| Failure::of(&error))?;
        if !bytes.is_empty() {
            texture
                .replace_region(
                    Region::new_2d(0, 0, spec.size[0].into(), spec.size[1].into()),
                    0,
                    0,
                    bytes,
                    spec.row_pitch.into(),
                    0,
                )
                .map_err(Failure::from_error)?;
        }
        Ok(keep(&mut self.textures, texture))
    }

    fn replace_region(
        &mut self,
        texture: Handle,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<()> {
        let texture = self.held_texture(texture)?;
        texture
            .replace_region(
                Region::new_2d(0, 0, size[0].into(), size[1].into()),
                0,
                0,
                bytes,
                row_pitch.into(),
                0,
            )
            .map_err(Failure::from_error)
    }

    fn texture(&mut self, request: &crate::texture::TextureRequest<'_>) -> Answer<Handle> {
        let format = PixelFormat::try_from(request.format).map_err(|problem| {
            Failure::named(
                "NoSuchPixelFormat",
                format!("{}: {problem:?}", request.format),
            )
        })?;
        let descriptor = texture_descriptor(request, format);
        // One of the three Metal calls that answers a texture of this descriptor, as the request
        // says: the device's own, a buffer's, or a parent texture's view.
        let texture = match request.over {
            None => self.device.new_texture(&descriptor),
            Some(crate::texture::Over::Buffer {
                buffer,
                offset,
                bytes_per_row,
            }) => self
                .held(buffer)?
                .new_texture(&descriptor, offset, bytes_per_row),
            Some(crate::texture::Over::View { texture }) => {
                self.held_texture(texture)?.new_texture_view_with_swizzle(
                    format,
                    request.texture_type,
                    0..u64::from(request.levels.max(1)),
                    0..u64::from(request.slices.max(1)),
                    request.swizzle,
                )
            }
        }
        .map_err(|error| Failure::of(&error))?;
        for upload in request.uploads {
            texture
                .replace_region(
                    Region {
                        origin: Origin::new(
                            upload.origin[0].into(),
                            upload.origin[1].into(),
                            upload.origin[2].into(),
                        ),
                        size: Size::new(
                            upload.size[0].into(),
                            upload.size[1].into(),
                            upload.size[2].max(1).into(),
                        ),
                    },
                    upload.level.into(),
                    upload.slice.into(),
                    upload.bytes,
                    upload.row_pitch.into(),
                    upload.image_pitch.into(),
                )
                .map_err(Failure::from_error)?;
        }
        Ok(keep(&mut self.textures, texture))
    }

    fn sampler(&mut self, spec: &SamplerSpec) -> Answer<Handle> {
        let mut descriptor = SamplerDescriptor::new();
        descriptor.min_filter = spec.min_filter;
        descriptor.mag_filter = spec.mag_filter;
        descriptor.s_address_mode = spec.s_address_mode;
        descriptor.t_address_mode = spec.t_address_mode;
        descriptor.border_color = spec.border_color;
        descriptor.normalized_coordinates = spec.normalized_coordinates;
        let mut options = SamplerStateOptions::default();
        options.support_argument_buffers = spec.support_argument_buffers;
        let sampler = self
            .device
            .new_sampler_state_with_options(&descriptor, options)
            .map_err(|error| Failure::of(&error))?;
        Ok(keep(&mut self.samplers, sampler))
    }

    fn argument_buffer(&mut self, slots: &[ArgumentSlot]) -> Answer<Handle> {
        let mut words = Vec::with_capacity(slots.len() * 8);
        let mut named = Vec::new();
        for slot in slots {
            let id = match *slot {
                ArgumentSlot::Texture(handle) => {
                    named.push(handle);
                    self.held_texture(handle)?.gpu_resource_id().0
                }
                ArgumentSlot::Sampler(handle) => self.held_sampler(handle)?.gpu_resource_id().0,
            };
            words.extend(id.to_le_bytes());
        }
        let buffer = self.buffer(&words)?;
        self.argument_residents.push((buffer, named));
        Ok(buffer)
    }

    fn dispatch(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        threadgroups: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle> {
        self.encode(pipeline, bindings, &[], |encoder| {
            encoder.dispatch_threadgroups(size(threadgroups), size(threads))
        })
    }

    fn compute_pipeline_with_stage_input(
        &mut self,
        library: Handle,
        entry: &str,
        stage_input: &StageInput,
    ) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .new_function_with_name(entry)
            .map_err(|error| Failure::of(&error))?;
        let mut descriptor = metal2vulkan::pipeline::ComputePipelineDescriptor::new(
            metal2vulkan::pipeline::StageSource::Function(function.function_descriptor()),
        );
        descriptor.stage_input = Some(stage_input_descriptor(stage_input));
        let pipeline = self
            .device
            .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
            .map_err(|error| Failure::of(&error))?;
        Ok(keep(&mut self.pipelines, pipeline))
    }

    fn dispatch_threads(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        stage_in: &[StageInRegion],
        grid: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle> {
        self.encode(pipeline, bindings, stage_in, |encoder| {
            encoder.dispatch_threads(size(grid), size(threads))
        })
    }

    fn wait(&mut self, command_buffer: Handle) -> Answer<Value> {
        let command_buffer = self
            .command_buffers
            .get(command_buffer.0 as usize)
            .ok_or_else(|| {
                Failure::named(
                    "NoSuchHandle",
                    format!("command buffer {}", command_buffer.0),
                )
            })?;
        if command_buffer.wait_until_completed_for(WAIT).is_err() {
            return Ok(Value::Nothing(Negative::NeverReturned));
        }
        Ok(status_value(command_buffer.status()))
    }

    fn read_buffer(&mut self, buffer: Handle, offset: u64, length: usize) -> Answer<Vec<u8>> {
        let buffer = self.held(buffer)?;
        let past = offset
            .checked_add(length as u64)
            .is_none_or(|end| end > buffer.length());
        if past {
            return Err(Failure::named(
                "PastTheBuffer",
                format!(
                    "{length} bytes at {offset} of a buffer of {}",
                    buffer.length()
                ),
            ));
        }
        let contents = buffer
            .contents()
            .ok_or_else(|| Failure::named("NoContents", "a buffer the host does not reach"))?;
        // SAFETY: `[offset, offset + length)` is inside the buffer's bytes, the host reaches
        // them, and the caller waited for the command buffer that wrote them.
        let bytes =
            unsafe { std::slice::from_raw_parts(contents.as_ptr().add(offset as usize), length) };
        Ok(bytes.to_vec())
    }

    fn announced(&mut self) -> Vec<String> {
        (self.notices)()
    }

    fn indirect_draw_read_back(
        &mut self,
        primitive_type: u64,
        indexed: bool,
    ) -> Answer<IndirectReadBack> {
        let mut descriptor = metal2vulkan_runtime::IndirectCommandBufferDescriptor::new();
        descriptor.command_types =
            IndirectCommandType::DRAW.union(IndirectCommandType::DRAW_INDEXED);
        descriptor.max_vertex_buffer_bind_count = 1;
        descriptor.max_fragment_buffer_bind_count = 1;
        let made = self
            .device
            .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
            .map_err(Failure::from_error)?;
        let command = made
            .indirect_render_command_at(0)
            .map_err(Failure::from_error)?;
        let primitive = DrawPrimitiveType::from_raw(primitive_type).map_err(|problem| {
            Failure::named(
                "NoSuchPrimitiveType",
                format!("{primitive_type}: {problem:?}"),
            )
        })?;
        if indexed {
            let indices = self
                .device
                .new_buffer(4096, ResourceOptions::shared())
                .map_err(Failure::from_error)?;
            command
                .draw_indexed_primitives(primitive, 6, IndexType::UInt16, &indices, 0, 1, 0, 0)
                .map_err(Failure::from_error)?;
        } else {
            command
                .draw_primitives(primitive, 0, 6, 1, 0)
                .map_err(Failure::from_error)?;
        }
        let answered = command
            .draw()
            .map_err(Failure::from_error)?
            .map(drawn_primitive_type);
        let recorded = match made
            .commands(0..1)
            .map_err(Failure::from_error)?
            .into_iter()
            .next()
        {
            Some(IndirectCommand::Render(render)) => render.draw.map(drawn_primitive_type),
            _ => None,
        };
        Ok(IndirectReadBack {
            command_type: command
                .command_type()
                .map_err(Failure::from_error)?
                .as_raw()
                .into(),
            answered_primitive_type: answered,
            recorded_primitive_type: recorded,
        })
    }

    fn work(&mut self, work: &crate::work::Work) -> Answer<crate::work::Done> {
        let queue = self.queue()?.clone();
        crate::runtime_work::run(self.device, &queue, work)
    }

    fn fragment_fill(&mut self, fill: &FragmentFill<'_>) -> Answer<FragmentFilled> {
        let state = {
            let library = self.libraries.get(fill.library.0 as usize).ok_or_else(|| {
                Failure::named("NoSuchHandle", format!("library {}", fill.library.0))
            })?;
            let function = |name: &str| -> Answer<StageSource> {
                Ok(StageSource::Function(
                    library
                        .new_function_with_name(name)
                        .map_err(|error| Failure::of(&error))?
                        .function_descriptor(),
                ))
            };
            let mut descriptor = RenderPipelineDescriptor::new(function(fill.vertex)?);
            descriptor.fragment = Some(function(fill.fragment)?);
            descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(fill.format));
            self.device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .map_err(Failure::from_error)?
        };
        let mut target = TextureDescriptor::texture_2d(
            fill.format,
            u64::from(fill.size[0]),
            u64::from(fill.size[1]),
            false,
        );
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::private();
        let target = self
            .device
            .new_texture(&target)
            .map_err(Failure::from_error)?;
        let bound: Vec<Buffer> = fill
            .buffers
            .iter()
            .map(|handle| self.held(*handle).cloned())
            .collect::<Answer<Vec<Buffer>>>()?;
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(target.clone());
        attachment.load_action = LoadAction::DontCare;
        attachment.store_action = StoreAction::Store;
        let command_buffer = self
            .queue()?
            .command_buffer()
            .map_err(Failure::from_error)?;
        let mut encoder = command_buffer
            .render_command_encoder(&pass)
            .map_err(Failure::from_error)?;
        encoder
            .set_render_pipeline_state(&state)
            .map_err(Failure::from_error)?;
        for (index, buffer) in bound.iter().enumerate() {
            encoder
                .set_fragment_buffer(Some(buffer), 0, index as u32)
                .map_err(Failure::from_error)?;
        }
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, fill.vertices, 1, 0)
            .map_err(Failure::from_error)?;
        encoder.end_encoding().map_err(Failure::from_error)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        let status = if command_buffer.wait_until_completed_for(WAIT).is_err() {
            Value::Nothing(Negative::NeverReturned)
        } else {
            status_value(command_buffer.status())
        };
        keep(&mut self.command_buffers, command_buffer);
        let bytes = self.blit_out(&target, fill.size, fill.row_pitch)?;
        keep(&mut self.textures, target);
        Ok(FragmentFilled { status, bytes })
    }

    fn mesh_pipeline(&mut self, library: Handle, functions: &MeshFunctions<'_>) -> Answer<Handle> {
        let descriptor = self.mesh_descriptor(library, functions)?;
        let state = self
            .device
            .new_mesh_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.render_pipelines, state))
    }

    fn render_target(&mut self, width: u32, height: u32) -> Answer<Handle> {
        let mut descriptor = TextureDescriptor::texture_2d(
            TARGET_PIXEL_FORMAT,
            u64::from(width),
            u64::from(height),
            false,
        );
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::shared();
        let texture = self
            .device
            .new_texture(&descriptor)
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.textures, texture))
    }

    fn mesh_pass(&mut self, targets: &[Handle], steps: &[MeshStep]) -> Answer<PassOutcome> {
        let attachments: Vec<Texture> = targets
            .iter()
            .map(|handle| self.held_texture(*handle).cloned())
            .collect::<Answer<Vec<Texture>>>()?;
        let states: Vec<RenderPipelineState> = self.render_pipelines.clone();
        let buffers: Vec<Buffer> = self.buffers.clone();
        let held = |handle: Handle| -> Answer<&Buffer> {
            buffers
                .get(handle.0 as usize)
                .ok_or_else(|| Failure::named("NoSuchHandle", format!("buffer {}", handle.0)))
        };
        let mut pass = RenderPassDescriptor::new();
        for (attachment, target) in pass.color_attachments.iter_mut().zip(&attachments) {
            attachment.texture = Some(target.clone());
            attachment.load_action = LoadAction::Clear;
            attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            attachment.store_action = StoreAction::Store;
        }
        let command_buffer = self
            .queue()?
            .command_buffer()
            .map_err(Failure::from_error)?;
        let mut encoder = command_buffer
            .render_command_encoder(&pass)
            .map_err(Failure::from_error)?;
        let mut refused = Vec::with_capacity(steps.len());
        for step in steps {
            let answered = match step {
                MeshStep::Pipeline(state) => states
                    .get(state.0 as usize)
                    .ok_or_else(|| {
                        Failure::named("NoSuchHandle", format!("mesh pipeline {}", state.0))
                    })
                    .and_then(|state| {
                        encoder
                            .set_render_pipeline_state(state)
                            .map_err(Failure::from_error)
                    }),
                MeshStep::ObjectBuffer {
                    buffer,
                    offset,
                    index,
                } => held(*buffer).and_then(|buffer| {
                    encoder
                        .set_object_buffer(Some(buffer), *offset, *index)
                        .map_err(Failure::from_error)
                }),
                MeshStep::ObjectBytes { bytes, index } => encoder
                    .set_object_bytes(bytes, *index)
                    .map_err(Failure::from_error),
                MeshStep::MeshBuffer {
                    buffer,
                    offset,
                    index,
                } => held(*buffer).and_then(|buffer| {
                    encoder
                        .set_mesh_buffer(Some(buffer), *offset, *index)
                        .map_err(Failure::from_error)
                }),
                MeshStep::FragmentBuffer {
                    buffer,
                    offset,
                    index,
                } => held(*buffer).and_then(|buffer| {
                    encoder
                        .set_fragment_buffer(Some(buffer), *offset, *index)
                        .map_err(Failure::from_error)
                }),
                MeshStep::DrawThreadgroups {
                    groups,
                    object,
                    mesh,
                } => encoder
                    .draw_mesh_threadgroups(size(*groups), size(*object), size(*mesh))
                    .map_err(Failure::from_error),
                MeshStep::DrawThreads {
                    threads,
                    object,
                    mesh,
                } => encoder
                    .draw_mesh_threads(size(*threads), size(*object), size(*mesh))
                    .map_err(Failure::from_error),
                MeshStep::DrawIndirect {
                    buffer,
                    offset,
                    object,
                    mesh,
                } => held(*buffer).and_then(|buffer| {
                    encoder
                        .draw_mesh_threadgroups_indirect(
                            buffer,
                            *offset,
                            size(*object),
                            size(*mesh),
                        )
                        .map_err(Failure::from_error)
                }),
                MeshStep::BarrierScope {
                    scope,
                    after,
                    before,
                } => scope_bits(*scope).and_then(|scope| {
                    let (after, before) = (stages(*after)?, stages(*before)?);
                    encoder.memory_barrier_with_scope(scope, after, before);
                    Ok(())
                }),
                MeshStep::BarrierResources {
                    buffers: over,
                    after,
                    before,
                } => over
                    .iter()
                    .map(|handle| held(*handle))
                    .collect::<Answer<Vec<&Buffer>>>()
                    .and_then(|over| {
                        let resources: Vec<ResourceRef<'_>> =
                            over.into_iter().map(ResourceRef::Buffer).collect();
                        encoder
                            .memory_barrier_with_resources(
                                &resources,
                                stages(*after)?,
                                stages(*before)?,
                            )
                            .map_err(Failure::from_error)
                    }),
            };
            refused.push(answered.err());
        }
        encoder.end_encoding().map_err(Failure::from_error)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        if command_buffer.wait_until_completed_for(WAIT).is_err() {
            return Ok(PassOutcome {
                refused,
                status: Value::Nothing(Negative::NeverReturned),
                error: None,
            });
        }
        Ok(PassOutcome {
            refused,
            status: status_value(command_buffer.status()),
            error: command_buffer
                .error()
                .map(|error| Value::unit_variant(format!("{error:?}"))),
        })
    }

    fn mesh_pipeline_for_commands(
        &mut self,
        library: Handle,
        functions: &MeshFunctions<'_>,
    ) -> Answer<Handle> {
        let mut descriptor = self.mesh_descriptor(library, functions)?;
        descriptor.support_indirect_command_buffers = true;
        let state = self
            .device
            .new_mesh_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.render_pipelines, state))
    }

    fn mesh_command_buffer(&mut self, commands: &[MeshCommand]) -> Answer<Handle> {
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW_MESH_THREADGROUPS
            .union(IndirectCommandType::DRAW_MESH_THREADS);
        descriptor.inherit_pipeline_state = false;
        descriptor.inherit_buffers = false;
        descriptor.max_object_buffer_bind_count = 1;
        descriptor.max_mesh_buffer_bind_count = 1;
        let made = self
            .device
            .new_indirect_command_buffer(
                &descriptor,
                commands.len() as u64,
                ResourceOptions::shared(),
            )
            .map_err(Failure::from_error)?;
        for (at, command) in commands.iter().enumerate() {
            let encoded = made
                .indirect_render_command_at(at as u64)
                .map_err(Failure::from_error)?;
            let state = self
                .render_pipelines
                .get(command.pipeline.0 as usize)
                .ok_or_else(|| {
                    Failure::named(
                        "NoSuchHandle",
                        format!("mesh pipeline {}", command.pipeline.0),
                    )
                })?
                .clone();
            encoded
                .set_render_pipeline_state(&state)
                .map_err(Failure::from_error)?;
            for (buffer, index) in &command.object_buffers {
                let buffer = self.held(*buffer)?.clone();
                encoded
                    .set_object_buffer(Some(&buffer), 0, *index)
                    .map_err(Failure::from_error)?;
            }
            for (buffer, index) in &command.mesh_buffers {
                let buffer = self.held(*buffer)?.clone();
                encoded
                    .set_mesh_buffer(Some(&buffer), 0, *index)
                    .map_err(Failure::from_error)?;
            }
            match command.draw {
                CommandDraw::Threadgroups {
                    groups,
                    object,
                    mesh,
                } => encoded.draw_mesh_threadgroups(size(groups), size(object), size(mesh)),
                CommandDraw::Threads {
                    threads,
                    object,
                    mesh,
                } => encoded.draw_mesh_threads(size(threads), size(object), size(mesh)),
            }
            .map_err(Failure::from_error)?;
        }
        Ok(keep(&mut self.command_buffers_indirect, made))
    }

    fn execute_mesh_commands(
        &mut self,
        targets: &[Handle],
        commands: Handle,
        used: &[Handle],
    ) -> Answer<PassOutcome> {
        let attachments: Vec<Texture> = targets
            .iter()
            .map(|handle| self.held_texture(*handle).cloned())
            .collect::<Answer<Vec<Texture>>>()?;
        let resident: Vec<Buffer> = used
            .iter()
            .map(|handle| self.held(*handle).cloned())
            .collect::<Answer<Vec<Buffer>>>()?;
        let icb = self
            .command_buffers_indirect
            .get(commands.0 as usize)
            .ok_or_else(|| {
                Failure::named("NoSuchHandle", format!("command buffer {}", commands.0))
            })?
            .clone();
        let mut pass = RenderPassDescriptor::new();
        for (attachment, target) in pass.color_attachments.iter_mut().zip(&attachments) {
            attachment.texture = Some(target.clone());
            attachment.load_action = LoadAction::Clear;
            attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            attachment.store_action = StoreAction::Store;
        }
        let command_buffer = self
            .queue()?
            .command_buffer()
            .map_err(Failure::from_error)?;
        let mut encoder = command_buffer
            .render_command_encoder(&pass)
            .map_err(Failure::from_error)?;
        let stages = RenderStages::OBJECT
            .union(RenderStages::MESH)
            .union(RenderStages::FRAGMENT);
        let mut refused = Vec::new();
        for buffer in &resident {
            refused.push(
                encoder
                    .use_resource(
                        ResourceRef::Buffer(buffer),
                        ResourceUsage::READ.union(ResourceUsage::WRITE),
                        stages,
                    )
                    .err()
                    .map(|error| Failure::of(&error)),
            );
        }
        refused.push(
            encoder
                .execute_commands_in_buffer(&icb, 0..icb.size())
                .err()
                .map(|error| Failure::of(&error)),
        );
        encoder.end_encoding().map_err(Failure::from_error)?;
        let committed = command_buffer
            .commit()
            .err()
            .map(|error| Failure::of(&error));
        if committed.is_some() {
            refused.push(committed);
            return Ok(PassOutcome {
                refused,
                status: status_value(command_buffer.status()),
                error: None,
            });
        }
        refused.push(None);
        if command_buffer.wait_until_completed_for(WAIT).is_err() {
            return Ok(PassOutcome {
                refused,
                status: Value::Nothing(Negative::NeverReturned),
                error: None,
            });
        }
        Ok(PassOutcome {
            refused,
            status: status_value(command_buffer.status()),
            error: command_buffer
                .error()
                .map(|error| Value::unit_variant(format!("{error:?}"))),
        })
    }

    fn plane_texture(&mut self, buffer: Handle, over: &PlaneTexture) -> Answer<Handle> {
        let format = PixelFormat::try_from(over.format).map_err(|problem| {
            Failure::named("NoSuchPixelFormat", format!("{}: {problem}", over.format))
        })?;
        let [width, height] = over.size;
        let mut descriptor =
            TextureDescriptor::texture_2d(format, u64::from(width), u64::from(height), false);
        descriptor.resource_options = ResourceOptions::shared();
        let buffer = self.held(buffer)?.clone();
        let texture = match over.ostype {
            Some(ostype) => {
                descriptor.usage = TextureUsage::SHADER_READ;
                let (offset, row) = over.planes[over.plane as usize];
                let (chroma_offset, chroma_row) = over.planes[1];
                let facts = IoSurfaceYCbCr::new(
                    u32::from_be_bytes(ostype),
                    u64::from(over.plane),
                    [u64::from(width), u64::from(height)],
                    chroma_offset,
                    chroma_row,
                );
                let plane = IoSurfacePlane::new(offset, row, true).with_ycbcr(facts);
                buffer.new_texture_with_iosurface_plane(&descriptor, plane)
            }
            None => {
                descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
                buffer.new_texture(&descriptor, over.planes[0].0, over.planes[0].1)
            }
        }
        .map_err(Failure::from_error)?;
        Ok(keep(&mut self.textures, texture))
    }

    fn fill_texture(&mut self, texture: Handle, fill: &Fill) -> Answer<Value> {
        let texture = self.held_texture(texture)?.clone();
        let command_buffer = self
            .queue()?
            .command_buffer()
            .map_err(Failure::from_error)?;
        let mut blit = command_buffer
            .blit_command_encoder()
            .map_err(Failure::from_error)?;
        let region = |[x, y, width, height]: [u32; 4]| {
            Region::new_2d(
                u64::from(x),
                u64::from(y),
                u64::from(width),
                u64::from(height),
            )
        };
        let filled = match fill {
            Fill::Pattern {
                level,
                slice,
                region: at,
                pattern,
            } => blit.fill_texture(
                &texture,
                u64::from(*level),
                u64::from(*slice),
                region(*at),
                pattern,
            ),
            Fill::Color {
                level,
                slice,
                region: at,
                color,
                packed_into,
            } => PixelFormat::try_from(*packed_into)
                .map_err(metal2vulkan_runtime::Error::Compile)
                .and_then(|packed_into| {
                    blit.fill_texture_color(
                        &texture,
                        u64::from(*level),
                        u64::from(*slice),
                        region(*at),
                        ClearColor::new(color[0], color[1], color[2], color[3]),
                        Some(packed_into),
                    )
                }),
        };
        blit.end_encoding().map_err(Failure::from_error)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        // The fill's own answer is the case's subject, so a refusal is reported as the call's
        // typed error rather than aborting the encoding: the command buffer still ends and is
        // waited for, as the probe's did, so the bytes it did or did not write are readable.
        let refused = filled.err().map(|error| Failure::of(&error));
        if command_buffer.wait_until_completed_for(WAIT).is_err() {
            return Ok(Value::Nothing(Negative::NeverReturned));
        }
        match refused {
            Some(failure) => Err(failure),
            None => Ok(status_value(command_buffer.status())),
        }
    }

    fn rate_map(&mut self, request: &RateMapRequest<'_>) -> Answer<Handle> {
        let mut descriptor = RateMapDescriptor::new(request.screen);
        descriptor.label = request.label.map(str::to_owned);
        descriptor.layers = request
            .layers
            .iter()
            .map(|layer| {
                RateMapLayerDescriptor::new(layer.horizontal.clone(), layer.vertical.clone())
            })
            .collect();
        let map = self
            .device
            .new_rasterization_rate_map(&descriptor)
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.rate_maps, map))
    }

    fn rate_map_facts(&mut self, map: Handle, layers: &[u64]) -> Answer<RateMapFacts> {
        let map = self.held_rate_map(map)?;
        let size = |size: Size| [size.width, size.height, size.depth];
        let sized = map.parameter_buffer_size_and_align();
        Ok(RateMapFacts {
            label: map.label().map(str::to_owned),
            screen_size: size(map.screen_size()),
            physical_granularity: size(map.physical_granularity()),
            layer_count: map.layer_count(),
            physical_sizes: layers
                .iter()
                .map(|layer| size(map.physical_size_for_layer(*layer)))
                .collect(),
            parameter_size: sized.size,
            parameter_align: sized.align,
        })
    }

    fn copy_rate_map_parameters(&mut self, map: Handle, into: Handle, offset: u64) -> Answer<()> {
        let buffer = self.held(into)?.clone();
        let map = self.held_rate_map(map)?;
        map.copy_parameter_data_to_buffer(&buffer, offset)
            .map_err(Failure::from_error)?;
        Ok(())
    }

    fn read_texture_over(
        &mut self,
        texture: Handle,
        size: [u32; 2],
        row_pitch: u32,
        before: &[u8],
    ) -> Answer<Vec<u8>> {
        let wanted = row_pitch as usize * size[1] as usize;
        if before.len() != wanted {
            return Err(Failure::named(
                "TooFewBytes",
                format!(
                    "{} bytes before the read, and it reads {wanted}",
                    before.len()
                ),
            ));
        }
        let texture = self.held_texture(texture)?;
        let mut bytes = before.to_vec();
        texture
            .get_bytes(
                &mut bytes,
                row_pitch.into(),
                0,
                Region::new_2d(0, 0, size[0].into(), size[1].into()),
                0,
                0,
            )
            .map_err(Failure::from_error)?;
        Ok(bytes)
    }

    fn render_pipeline_with_reflection(
        &mut self,
        library: Handle,
        vertex: &str,
        fragment: &str,
        attachment: PixelFormat,
    ) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let stage = |name: &str| -> Answer<StageSource> {
            Ok(StageSource::Function(
                library
                    .new_function_with_name(name)
                    .map_err(Failure::from_error)?
                    .function_descriptor(),
            ))
        };
        let mut descriptor = RenderPipelineDescriptor::new(stage(vertex)?);
        descriptor.fragment = Some(stage(fragment)?);
        descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(attachment));
        let options = PipelineOptions::new(PipelineOption::BINDING_INFO);
        // The reflection out-parameter is asked for, as the case's Metal side asks for it, and
        // dropped here: what the case reads is `texture_access`, the state's own answer, and the
        // runtime's rule is that asking for a reflection may change what the creation computes.
        let (state, _) = self
            .device
            .new_render_pipeline_state_with_reflection(&descriptor, &options)
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.render_pipelines, state))
    }

    fn texture_accesses(
        &mut self,
        pipeline: Handle,
        stage: RenderStages,
        indices: &[u64],
    ) -> Answer<Vec<Option<BindingAccess>>> {
        let state = self
            .render_pipelines
            .get(pipeline.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", pipeline.0)))?
            .clone();
        indices
            .iter()
            .map(|index| {
                state
                    .texture_access(stage, *index)
                    .map_err(Failure::from_error)
            })
            .collect()
    }

    fn tile_pipeline(&mut self, library: Handle, spec: &TilePipelineSpec<'_>) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .new_function_with_name(spec.entry)
            .map_err(Failure::from_error)?;
        let mut descriptor = TileRenderPipelineDescriptor::new(StageSource::Function(
            function.function_descriptor(),
        ));
        for (slot, format) in descriptor
            .color_attachment_formats
            .iter_mut()
            .zip(spec.attachments)
        {
            *slot = Some(*format);
        }
        descriptor.max_total_threads_per_threadgroup = spec.max_total_threads_per_threadgroup;
        descriptor.threadgroup_size_matches_tile_size = spec.threadgroup_size_matches_tile_size;
        let state = self
            .device
            .new_tile_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.render_pipelines, state))
    }

    fn target(&mut self, spec: &TargetSpec) -> Answer<Handle> {
        if let Some(plane) = &spec.surface {
            return self.surface_target(spec, plane);
        }
        if let Some(over) = &spec.over_buffer {
            return self.buffer_backed_target(spec, over);
        }
        let mut descriptor = TextureDescriptor::texture_2d(
            spec.format,
            u64::from(spec.size[0]),
            u64::from(spec.size[1]),
            false,
        );
        let mut usage = TextureUsage::RENDER_TARGET;
        if spec.shader_read {
            usage |= TextureUsage::SHADER_READ;
        }
        if spec.shader_write {
            usage |= TextureUsage::SHADER_WRITE;
        }
        descriptor.usage = usage;
        descriptor.resource_options = match spec.shared {
            true => ResourceOptions::shared(),
            false => ResourceOptions::private(),
        };
        if spec.samples > 1 {
            descriptor.texture_type = TextureType::Type2DMultisample;
            descriptor.sample_count = u64::from(spec.samples);
        }
        let texture = self
            .device
            .new_texture(&descriptor)
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.textures, texture))
    }

    fn render_pipeline(
        &mut self,
        library: Handle,
        spec: &RenderPipelineSpec<'_>,
    ) -> Answer<Handle> {
        let stage = |stage: &StageFunction<'_>| -> Answer<StageSource> {
            let named = stage.library.unwrap_or(library);
            let held = self
                .libraries
                .get(named.0 as usize)
                .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", named.0)))?;
            Ok(StageSource::Function(
                held.new_function_with_name(stage.entry)
                    .map_err(Failure::from_error)?
                    .function_descriptor(),
            ))
        };
        let mut descriptor = RenderPipelineDescriptor::new(stage(&spec.vertex)?);
        descriptor.fragment = spec.fragment.as_ref().map(&stage).transpose()?;
        let apart = pipeline_fields(spec)?;
        descriptor.color_attachments = apart.color_attachments;
        descriptor.depth_format = apart.depth_format;
        descriptor.stencil_format = apart.stencil_format;
        descriptor.raster_sample_count = apart.raster_sample_count;
        descriptor.rasterization_enabled = apart.rasterization_enabled;
        descriptor.vertex_descriptor = apart.vertex_descriptor;
        let state = self
            .device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.render_pipelines, state))
    }

    fn tile_facts(&mut self, state: Handle) -> Answer<TileFacts> {
        let state = self
            .render_pipelines
            .get(state.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("tile pipeline {}", state.0)))?;
        Ok(TileFacts {
            max_total_threads_per_threadgroup: u64::from(state.max_total_threads_per_threadgroup()),
            threadgroup_size_matches_tile_size: state.threadgroup_size_matches_tile_size(),
            imageblock_sample_length: state.imageblock_sample_length().ok(),
        })
    }

    fn tile_pass(
        &mut self,
        targets: &[Handle],
        tile: [u32; 2],
        steps: &[TileStep],
    ) -> Answer<PassOutcome> {
        let attachments: Vec<Texture> = targets
            .iter()
            .map(|handle| self.held_texture(*handle).cloned())
            .collect::<Answer<Vec<Texture>>>()?;
        let states: Vec<RenderPipelineState> = self.render_pipelines.clone();
        let buffers: Vec<Buffer> = self.buffers.clone();
        let textures: Vec<Texture> = self.textures.clone();
        let mut pass = RenderPassDescriptor::new();
        for (attachment, target) in pass.color_attachments.iter_mut().zip(&attachments) {
            attachment.texture = Some(target.clone());
            attachment.load_action = LoadAction::Load;
            attachment.store_action = StoreAction::Store;
        }
        pass.tile_width = u64::from(tile[0]);
        pass.tile_height = u64::from(tile[1]);
        let command_buffer = self
            .queue()?
            .command_buffer()
            .map_err(Failure::from_error)?;
        let mut encoder = command_buffer
            .render_command_encoder(&pass)
            .map_err(Failure::from_error)?;
        let mut refused = Vec::with_capacity(steps.len());
        for step in steps {
            let answered = match step {
                TileStep::Pipeline(state) => states
                    .get(state.0 as usize)
                    .ok_or_else(|| {
                        Failure::named("NoSuchHandle", format!("tile pipeline {}", state.0))
                    })
                    .and_then(|state| {
                        encoder
                            .set_render_pipeline_state(state)
                            .map_err(Failure::from_error)
                    }),
                TileStep::TileBuffer {
                    buffer,
                    offset,
                    index,
                } => buffers
                    .get(buffer.0 as usize)
                    .ok_or_else(|| Failure::named("NoSuchHandle", format!("buffer {}", buffer.0)))
                    .and_then(|buffer| {
                        encoder
                            .set_tile_buffer(Some(buffer), *offset, *index)
                            .map_err(Failure::from_error)
                    }),
                TileStep::TileBytes { bytes, index } => encoder
                    .set_tile_bytes(bytes, *index)
                    .map_err(Failure::from_error),
                TileStep::TileTexture { texture, index } => textures
                    .get(texture.0 as usize)
                    .ok_or_else(|| Failure::named("NoSuchHandle", format!("texture {}", texture.0)))
                    .and_then(|texture| {
                        encoder
                            .set_tile_texture(Some(texture), *index)
                            .map_err(Failure::from_error)
                    }),
                TileStep::DispatchThreadsPerTile(threads) => encoder
                    .dispatch_threads_per_tile(size(*threads))
                    .map_err(Failure::from_error),
            };
            refused.push(answered.err());
        }
        encoder.end_encoding().map_err(Failure::from_error)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        if command_buffer.wait_until_completed_for(WAIT).is_err() {
            return Ok(PassOutcome {
                refused,
                status: Value::Nothing(Negative::NeverReturned),
                error: None,
            });
        }
        Ok(PassOutcome {
            refused,
            status: status_value(command_buffer.status()),
            error: command_buffer
                .error()
                .map(|error| Value::unit_variant(format!("{error:?}"))),
        })
    }

    fn depth_stencil_state(&mut self, spec: &DepthStencilDescriptor) -> Answer<Handle> {
        let state = self
            .device
            .new_depth_stencil_state(spec)
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.depth_stencil_states, state))
    }

    fn write_plane(&mut self, target: Handle, bytes: &[u8]) -> Answer<()> {
        let buffer = self.held_plane(target)?.clone();
        if bytes.len() != buffer.length() as usize {
            return Err(Failure::named(
                "PlaneLength",
                format!("{} bytes for a plane of {}", bytes.len(), buffer.length()),
            ));
        }
        let contents = buffer.contents().ok_or_else(|| {
            Failure::named("NoContents", "the plane's buffer is not host-visible")
        })?;
        // SAFETY: the buffer is of shared storage and at least `bytes.len()` bytes long, checked
        // just above, and no command uses it now: a case writes the plane before it opens its pass.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        Ok(())
    }

    fn read_plane(&mut self, target: Handle) -> Answer<Vec<u8>> {
        let buffer = self.held_plane(target)?.clone();
        let length = buffer.length() as usize;
        let contents = buffer.contents().ok_or_else(|| {
            Failure::named("NoContents", "the plane's buffer is not host-visible")
        })?;
        // SAFETY: the buffer's own `length` bytes of shared storage, and the caller waited for the
        // command buffer that wrote them.
        let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), length) };
        Ok(bytes.to_vec())
    }

    fn render(&mut self, work: &[RenderWork]) -> Answer<PassOutcome> {
        // The pools are cloned once, so each piece of the program can reach the case's objects
        // while the command buffer is borrowed.
        let textures: Vec<Texture> = self.textures.clone();
        let buffers: Vec<Buffer> = self.buffers.clone();
        let states: Vec<RenderPipelineState> = self.render_pipelines.clone();
        let depth_stencil: Vec<DepthStencilState> = self.depth_stencil_states.clone();
        let samplers: Vec<SamplerState> = self.samplers.clone();
        let computes: Vec<ComputePipelineState> = self.pipelines.clone();
        let texture = |handle: Handle| -> Answer<&Texture> {
            textures
                .get(handle.0 as usize)
                .ok_or_else(|| Failure::named("NoSuchHandle", format!("texture {}", handle.0)))
        };
        let buffer = |handle: Handle| -> Answer<&Buffer> {
            buffers
                .get(handle.0 as usize)
                .ok_or_else(|| Failure::named("NoSuchHandle", format!("buffer {}", handle.0)))
        };
        let command_buffer = self
            .queue()?
            .command_buffer()
            .map_err(Failure::from_error)?;
        let mut refused = Vec::with_capacity(work.len());
        for piece in work {
            let answered = match piece {
                RenderWork::Pass(pass) => encode_pass(
                    &command_buffer,
                    pass,
                    &textures,
                    &buffers,
                    &states,
                    &depth_stencil,
                    &samplers,
                ),
                RenderWork::ReadTexture {
                    texture: from,
                    into,
                    offset,
                    row_pitch,
                    size: extent,
                } => texture(*from).and_then(|from| {
                    let into = buffer(*into)?;
                    let mut blit = command_buffer
                        .blit_command_encoder()
                        .map_err(Failure::from_error)?;
                    blit.copy_from_texture_to_buffer(
                        from,
                        0,
                        0,
                        Origin::new(0, 0, 0),
                        Size::new(u64::from(extent[0]), u64::from(extent[1]), 1),
                        into,
                        *offset,
                        u64::from(*row_pitch),
                        u64::from(*row_pitch) * u64::from(extent[1]),
                        BlitOption::None,
                    )
                    .map_err(Failure::from_error)?;
                    blit.end_encoding().map_err(Failure::from_error)
                }),
                RenderWork::Dispatch(dispatch) => computes
                    .get(dispatch.pipeline.0 as usize)
                    .ok_or_else(|| {
                        Failure::named(
                            "NoSuchHandle",
                            format!("compute pipeline {}", dispatch.pipeline.0),
                        )
                    })
                    .and_then(|state| {
                        let mut encoder = command_buffer
                            .compute_command_encoder()
                            .map_err(Failure::from_error)?;
                        encoder
                            .set_compute_pipeline_state(state)
                            .map_err(Failure::from_error)?;
                        for (at, (held, offset)) in dispatch.buffers.iter().enumerate() {
                            encoder
                                .set_buffer(Some(buffer(*held)?), *offset, at as u32)
                                .map_err(Failure::from_error)?;
                        }
                        for (at, held) in dispatch.textures.iter().enumerate() {
                            encoder
                                .set_texture(Some(texture(*held)?), at as u32)
                                .map_err(Failure::from_error)?;
                        }
                        encoder
                            .dispatch_threads(size(dispatch.grid), size(dispatch.threads))
                            .map_err(Failure::from_error)?;
                        encoder.end_encoding().map_err(Failure::from_error)
                    }),
            };
            refused.push(answered.err());
        }
        command_buffer.commit().map_err(Failure::from_error)?;
        if command_buffer.wait_until_completed_for(WAIT).is_err() {
            return Ok(PassOutcome {
                refused,
                status: Value::Nothing(Negative::NeverReturned),
                error: None,
            });
        }
        Ok(PassOutcome {
            refused,
            status: status_value(command_buffer.status()),
            error: command_buffer
                .error()
                .map(|error| Value::unit_variant(format!("{error:?}"))),
        })
    }

    fn heap_texture_size_and_align(&mut self, query: &HeapTextureQuery) -> Answer<SizeAndAlign> {
        let answered = self
            .device
            .heap_texture_size_and_align(&queried_descriptor(query))
            .map_err(Failure::from_error)?;
        Ok(SizeAndAlign {
            size: answered.size,
            align: answered.align,
        })
    }

    fn queue_slots(&mut self, max: Option<u32>, cap: u32) -> Answer<QueueSlots> {
        let queue = match max {
            None => self.device.new_command_queue(),
            Some(max) => self
                .device
                .new_command_queue_with_max_command_buffer_count(max as usize),
        }
        .map_err(Failure::from_error)?;
        let other = self
            .device
            .new_command_queue()
            .map_err(Failure::from_error)?;
        let making = queue.clone();
        crate::queue_slots::probe(
            cap,
            move || making.command_buffer().map_err(Failure::from_error),
            || other.command_buffer().is_ok(),
        )
    }

    fn handler_wait(&mut self, wait: crate::handler_waits::OwnWait) -> Answer<Value> {
        use crate::handler_waits::{HandlerKind, WaitKind};
        use std::sync::mpsc;
        let queue = self
            .device
            .new_command_queue()
            .map_err(Failure::from_error)?;
        let buffer = self
            .device
            .new_buffer(HANDLER_FILL, ResourceOptions::shared())
            .map_err(Failure::from_error)?;
        let command_buffer = queue.command_buffer().map_err(Failure::from_error)?;
        let mut blit = command_buffer
            .blit_command_encoder()
            .map_err(Failure::from_error)?;
        blit.fill_buffer(&buffer, 0..HANDLER_FILL, 0x11)
            .map_err(Failure::from_error)?;
        blit.end_encoding().map_err(Failure::from_error)?;
        let (sender, answer) = mpsc::channel();
        let kind = wait.wait;
        let handler: metal2vulkan_runtime::Handler = Box::new(move |own: &CommandBuffer| {
            let answered = match kind {
                WaitKind::UntilCompleted => own.wait_until_completed(),
                WaitKind::UntilScheduled => own.wait_until_scheduled(),
            };
            let _ = sender.send(answered.err().map(|error| Failure::of(&error)));
        });
        match wait.from {
            HandlerKind::Scheduled => command_buffer.add_scheduled_handler(handler),
            HandlerKind::Completed => command_buffer.add_completed_handler(handler),
        }
        .map_err(Failure::from_error)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        // Bounded, so a runtime that did not refuse the handler's wait is recorded rather than
        // hanging the suite: the handler would then still be inside its wait and the command
        // buffer would never complete.
        if command_buffer.wait_until_completed_for(WAIT).is_err() {
            return Ok(Value::Nothing(Negative::NeverReturned));
        }
        match answer.recv_timeout(WAIT) {
            Ok(None) => Ok(Value::ok(Value::Absent)),
            Ok(Some(failure)) => Err(failure),
            Err(_) => Ok(Value::Nothing(Negative::NeverReturned)),
        }
    }

    fn shared_event_rises(&mut self, label: &str) -> Answer<crate::shared_events::EventRise> {
        use crate::shared_events::{EventRise, WAIT_MS};
        let bound = Duration::from_millis(WAIT_MS);
        let queue = self
            .device
            .new_command_queue()
            .map_err(Failure::from_error)?;
        let event = self
            .device
            .new_shared_event()
            .map_err(Failure::from_error)?;
        event.set_label(label);
        let read_label = event.label();
        let initial = event.signaled_value();
        // The host's sets: a lower value leaves the event where it was.
        event.set_signaled_value(5);
        event.set_signaled_value(3);
        event.set_signaled_value(5);
        let after_host_sets = event.signaled_value();
        let signal = |value: u64| -> Answer<u64> {
            let command_buffer = queue.command_buffer().map_err(Failure::from_error)?;
            command_buffer
                .encode_signal_event(metal2vulkan_runtime::EventRef::from(&event), value)
                .map_err(Failure::from_error)?;
            command_buffer.commit().map_err(Failure::from_error)?;
            if command_buffer.wait_until_completed_for(WAIT).is_err() {
                return Err(Failure::named(
                    "NeverCompleted",
                    format!("the command buffer signalling {value}"),
                ));
            }
            Ok(event.signaled_value())
        };
        let after_gpu_signals = [signal(3)?, signal(5)?];
        let waits = [
            event.wait_until_signaled_value(3, bound),
            event.wait_until_signaled_value(5, bound),
            event.wait_until_signaled_value(6, bound),
        ];
        let after_max_signal = signal(u64::MAX)?;
        let waited_for_max = event.wait_until_signaled_value(u64::MAX, bound);
        Ok(EventRise {
            label: read_label,
            initial,
            after_host_sets,
            after_gpu_signals,
            waits,
            after_max_signal,
            waited_for_max,
        })
    }

    fn event_cross_queue_wait(&mut self) -> Answer<crate::event_orders::CrossQueueWait> {
        use crate::event_orders::{
            CrossQueueWait, Status, HALF, SCRATCH, SIGNALLING_BYTE, WAITING_BYTE,
        };
        let queue = || self.device.new_command_queue().map_err(Failure::from_error);
        let (waiting_queue, signalling_queue) = (queue()?, queue()?);
        let event = self.device.new_event().map_err(Failure::from_error)?;
        let scratch = self
            .device
            .new_buffer(SCRATCH, ResourceOptions::shared())
            .map_err(Failure::from_error)?;
        let fill = |command_buffer: &CommandBuffer,
                    range: std::ops::Range<u64>,
                    byte: u8|
         -> Result<(), Failure> {
            let mut blit = command_buffer
                .blit_command_encoder()
                .map_err(Failure::from_error)?;
            blit.fill_buffer(&scratch, range, byte)
                .map_err(Failure::from_error)?;
            blit.end_encoding().map_err(Failure::from_error)
        };
        let waiting = waiting_queue
            .command_buffer()
            .map_err(Failure::from_error)?;
        waiting
            .encode_wait_for_event(metal2vulkan_runtime::EventRef::from(&event), 1)
            .map_err(Failure::from_error)?;
        fill(&waiting, 0..HALF, WAITING_BYTE)?;
        let signalling = signalling_queue
            .command_buffer()
            .map_err(Failure::from_error)?;
        fill(&signalling, HALF..SCRATCH, SIGNALLING_BYTE)?;
        signalling
            .encode_signal_event(metal2vulkan_runtime::EventRef::from(&event), 1)
            .map_err(Failure::from_error)?;
        // The waiting one first, which is the whole point: its signal is not committed yet, so a
        // runtime that submitted in commit order onto one shared queue would deadlock here.
        waiting.commit().map_err(Failure::from_error)?;
        let waiting_after_its_commit = Status::read(format!("{:?}", waiting.status()));
        signalling.commit().map_err(Failure::from_error)?;
        let settled = |command_buffer: &CommandBuffer| {
            if command_buffer
                .wait_until_completed_for(crate::event_orders::COMPLETION)
                .is_err()
            {
                Status::NeverSettled
            } else {
                Status::read(format!("{:?}", command_buffer.status()))
            }
        };
        let signalling_at_the_end = settled(&signalling);
        let waiting_at_the_end = settled(&waiting);
        let bytes = read_shared(&scratch, 0, SCRATCH as usize)?;
        Ok(CrossQueueWait {
            waiting_after_its_commit,
            waiting_at_the_end,
            signalling_at_the_end,
            bytes,
        })
    }

    fn event_highest_signal(&mut self, values: [u64; 2]) -> Answer<u64> {
        use crate::event_orders::{COMPLETION, SCRATCH};
        let queue = || self.device.new_command_queue().map_err(Failure::from_error);
        let queues = [queue()?, queue()?];
        let event = self
            .device
            .new_shared_event()
            .map_err(Failure::from_error)?;
        let scratch = self
            .device
            .new_buffer(SCRATCH, ResourceOptions::private())
            .map_err(Failure::from_error)?;
        let mut committed = Vec::new();
        for (queue, &value) in queues.iter().zip(&values) {
            let command_buffer = queue.command_buffer().map_err(Failure::from_error)?;
            let mut blit = command_buffer
                .blit_command_encoder()
                .map_err(Failure::from_error)?;
            blit.fill_buffer(&scratch, 0..SCRATCH, value as u8)
                .map_err(Failure::from_error)?;
            blit.end_encoding().map_err(Failure::from_error)?;
            command_buffer
                .encode_signal_event(metal2vulkan_runtime::EventRef::from(&event), value)
                .map_err(Failure::from_error)?;
            command_buffer.commit().map_err(Failure::from_error)?;
            committed.push(command_buffer);
        }
        for command_buffer in &committed {
            if command_buffer.wait_until_completed_for(COMPLETION).is_err() {
                return Err(Failure::named(
                    "NeverCompleted",
                    "a signalling command buffer",
                ));
            }
        }
        Ok(event.signaled_value())
    }

    fn held_command_buffer(&mut self) -> Answer<crate::event_orders::HeldCommandBuffer> {
        use crate::event_orders::{
            HeldCommandBuffer, Status, COMPLETED, COMPLETION, HELD_FOR, SCHEDULED, SCRATCH,
        };
        let queue = || self.device.new_command_queue().map_err(Failure::from_error);
        let (held_queue, other_queue) = (queue()?, queue()?);
        let event = self
            .device
            .new_shared_event()
            .map_err(Failure::from_error)?;
        let scratch = self
            .device
            .new_buffer(SCRATCH, ResourceOptions::private())
            .map_err(Failure::from_error)?;
        let fill = |command_buffer: &CommandBuffer, byte: u8| -> Result<(), Failure> {
            let mut blit = command_buffer
                .blit_command_encoder()
                .map_err(Failure::from_error)?;
            blit.fill_buffer(&scratch, 0..SCRATCH, byte)
                .map_err(Failure::from_error)?;
            blit.end_encoding().map_err(Failure::from_error)
        };
        let log = HandlerLog::new();
        let held = held_queue.command_buffer().map_err(Failure::from_error)?;
        held.encode_wait_for_event(metal2vulkan_runtime::EventRef::from(&event), 1)
            .map_err(Failure::from_error)?;
        fill(&held, 1)?;
        held.add_scheduled_handler(log.handler(SCHEDULED))
            .map_err(Failure::from_error)?;
        held.add_completed_handler(log.handler(COMPLETED))
            .map_err(Failure::from_error)?;
        held.commit().map_err(Failure::from_error)?;
        let behind = held_queue.command_buffer().map_err(Failure::from_error)?;
        fill(&behind, 2)?;
        behind.commit().map_err(Failure::from_error)?;
        let other = other_queue.command_buffer().map_err(Failure::from_error)?;
        fill(&other, 3)?;
        other.commit().map_err(Failure::from_error)?;
        let other_queue_status = if other.wait_until_completed_for(COMPLETION).is_err() {
            Status::NeverSettled
        } else {
            Status::read(format!("{:?}", other.status()))
        };
        // The window the negative facts are false in: a submission that was going to happen has
        // had its chance by now.
        std::thread::sleep(HELD_FOR);
        let held_while_waiting = Status::read(format!("{:?}", held.status()));
        let behind_while_waiting = Status::read(format!("{:?}", behind.status()));
        let handlers_while_waiting = log.names();
        event.set_signaled_value(1);
        // The held command buffer's own completed handler is the bound: it is the last of its
        // handlers to run, so once it has, the log is whole and the status is settled.
        log.wait_for(COMPLETED, COMPLETION);
        let handlers_at_the_end = log.names();
        let held_at_the_end = if handlers_at_the_end.iter().any(|name| name == COMPLETED) {
            Status::read(format!("{:?}", held.status()))
        } else {
            Status::NeverSettled
        };
        let behind_at_the_end = if behind.wait_until_completed_for(COMPLETION).is_err() {
            Status::NeverSettled
        } else {
            Status::read(format!("{:?}", behind.status()))
        };
        Ok(HeldCommandBuffer {
            other_queue: other_queue_status,
            held_while_waiting,
            behind_while_waiting,
            handlers_while_waiting,
            handlers_at_the_end,
            held_at_the_end,
            behind_at_the_end,
        })
    }

    fn signal_before_a_wait(&mut self) -> Answer<crate::event_orders::SignalBeforeWait> {
        use crate::event_orders::{Status, COMPLETION, HELD_FOR, SCRATCH, SIGNAL_WAIT};
        let queue = self
            .device
            .new_command_queue()
            .map_err(Failure::from_error)?;
        let event = || self.device.new_shared_event().map_err(Failure::from_error);
        let (before, gate, after) = (event()?, event()?, event()?);
        let scratch = self
            .device
            .new_buffer(SCRATCH, ResourceOptions::private())
            .map_err(Failure::from_error)?;
        let command_buffer = queue.command_buffer().map_err(Failure::from_error)?;
        let fill = |byte: u8| -> Result<(), Failure> {
            let mut blit = command_buffer
                .blit_command_encoder()
                .map_err(Failure::from_error)?;
            blit.fill_buffer(&scratch, 0..SCRATCH, byte)
                .map_err(Failure::from_error)?;
            blit.end_encoding().map_err(Failure::from_error)
        };
        let signal = |event: &metal2vulkan_runtime::SharedEvent| {
            command_buffer
                .encode_signal_event(metal2vulkan_runtime::EventRef::from(event), 1)
                .map_err(Failure::from_error)
        };
        fill(1)?;
        signal(&before)?;
        command_buffer
            .encode_wait_for_event(metal2vulkan_runtime::EventRef::from(&gate), 1)
            .map_err(Failure::from_error)?;
        fill(2)?;
        signal(&after)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        let before_landed = before.wait_until_signaled_value(1, SIGNAL_WAIT);
        // The window the two negative facts are false in: the signal behind the wait has not
        // landed, and the command buffer is no further along than committed.
        std::thread::sleep(HELD_FOR);
        let after_while_pending = after.signaled_value();
        let status_while_pending = Status::read(format!("{:?}", command_buffer.status()));
        gate.set_signaled_value(1);
        let status_at_the_end = if command_buffer.wait_until_completed_for(COMPLETION).is_err() {
            Status::NeverSettled
        } else {
            Status::read(format!("{:?}", command_buffer.status()))
        };
        Ok(crate::event_orders::SignalBeforeWait {
            before_landed,
            after_while_pending,
            status_while_pending,
            after_at_the_end: after.signaled_value(),
            status_at_the_end,
        })
    }

    fn scheduled_handler_order(
        &mut self,
    ) -> Answer<crate::scheduled_handlers::ScheduledHandlerOrder> {
        use crate::event_orders::{Status, COMPLETED, SCHEDULED, SCRATCH};
        use crate::scheduled_handlers::{
            Returned, ScheduledHandlerOrder, ARRIVAL, HOLD_LIMIT, WINDOW,
        };
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{mpsc, Arc, Mutex};
        let queue = self
            .device
            .new_command_queue()
            .map_err(Failure::from_error)?;
        let scratch = self
            .device
            .new_buffer(SCRATCH, ResourceOptions::private())
            .map_err(Failure::from_error)?;
        let command_buffer = queue.command_buffer().map_err(Failure::from_error)?;
        let mut blit = command_buffer
            .blit_command_encoder()
            .map_err(Failure::from_error)?;
        blit.fill_buffer(&scratch, 0..SCRATCH, 1)
            .map_err(Failure::from_error)?;
        blit.end_encoding().map_err(Failure::from_error)?;
        let order: Arc<Mutex<Vec<String>>> = Arc::default();
        let (entered, entering) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        {
            let order = Arc::clone(&order);
            command_buffer
                .add_scheduled_handler(Box::new(move |own: &CommandBuffer| {
                    if let Ok(mut order) = order.lock() {
                        order.push(SCHEDULED.to_owned());
                    }
                    let _ = entered.send(format!("{:?}", own.status()));
                    // Held, so the probe can read the status and the wait while this handler is
                    // inside itself. Bounded, so the handler leaves whatever the probe does.
                    let _ = released.recv_timeout(HOLD_LIMIT);
                }))
                .map_err(Failure::from_error)?;
        }
        let (done, completing) = mpsc::channel();
        {
            let order = Arc::clone(&order);
            command_buffer
                .add_completed_handler(Box::new(move |own: &CommandBuffer| {
                    if let Ok(mut order) = order.lock() {
                        order.push(COMPLETED.to_owned());
                    }
                    let _ = done.send(format!("{:?}", own.status()));
                }))
                .map_err(Failure::from_error)?;
        }
        command_buffer.commit().map_err(Failure::from_error)?;
        let inside_the_handler = match entering.recv_timeout(ARRIVAL) {
            Ok(word) => Status::read(word),
            Err(_) => Status::NeverSettled,
        };
        let while_the_handler_ran = Status::read(format!("{:?}", command_buffer.status()));
        let returned = Arc::new(AtomicBool::new(false));
        let (answered, answer) = mpsc::channel();
        {
            let (waited, returned) = (command_buffer.clone(), Arc::clone(&returned));
            // The wait runs on a thread of its own because the fact under test is that it does
            // **not** return; the thread is left to finish rather than joined, so a wait that
            // never came back leaves that thread and not the suite.
            std::thread::spawn(move || {
                let answer = waited.wait_until_scheduled();
                returned.store(true, Ordering::SeqCst);
                let _ = answered.send(answer.err().map(|error| Failure::of(&error)));
            });
        }
        // The window the negative fact is false in.
        std::thread::sleep(WINDOW);
        let wait_returned_while_the_handler_ran = returned.load(Ordering::SeqCst);
        let _ = release.send(());
        let wait_answered = match answer.recv_timeout(ARRIVAL) {
            Ok(None) => Returned::Ok,
            Ok(Some(failure)) => Returned::Err(failure.recorded()),
            Err(_) => Returned::NotWithinTheWindow,
        };
        let inside_the_completed_handler = match completing.recv_timeout(ARRIVAL) {
            Ok(word) => Status::read(word),
            Err(_) => Status::NeverSettled,
        };
        let handler_order = order.lock().map(|order| order.clone()).unwrap_or_default();
        Ok(ScheduledHandlerOrder {
            inside_the_handler,
            while_the_handler_ran,
            wait_returned_while_the_handler_ran,
            wait_answered,
            inside_the_completed_handler,
            handler_order,
        })
    }

    fn buffer_texture(
        &mut self,
        request: &crate::buffer_texture::BufferTextureRequest,
    ) -> Answer<crate::buffer_texture::BufferTextureFacts> {
        let buffer = self
            .device
            .new_buffer(
                request.buffer_length,
                resource_options(request.buffer_storage),
            )
            .map_err(Failure::from_error)?;
        let texture = buffer
            .new_texture(
                &queried_descriptor(&request.descriptor),
                request.offset,
                request.bytes_per_row,
            )
            .map_err(Failure::from_error)?;
        let facts = crate::buffer_texture::BufferTextureFacts {
            texture_type: u64::from(texture.texture_type().as_raw()),
            pixel_format: u64::from(texture.pixel_format().as_raw()),
            width: texture.width(),
            height: texture.height(),
            depth: texture.depth(),
            mipmap_level_count: texture.mipmap_level_count(),
            array_length: texture.array_length(),
            sample_count: texture.sample_count(),
            usage: u64::from(texture.usage().as_raw()),
            storage_mode: texture.storage_mode().raw(),
            cpu_cache_mode: texture.cpu_cache_mode().raw(),
            hazard_tracking_mode: texture.hazard_tracking_mode().raw(),
            gpu_optimized: texture.allow_gpu_optimized_contents(),
        };
        // Both are kept, as the Metal side keeps them: a texture over a buffer the case dropped
        // would be a texture over freed memory, and a later row could read it.
        self.buffers.push(buffer);
        self.textures.push(texture);
        Ok(facts)
    }

    fn makes_texture(&mut self, query: &HeapTextureQuery) -> Answer<bool> {
        match self.device.new_texture(&queried_descriptor(query)) {
            Ok(_) => Ok(true),
            // Metal answers nil for a descriptor it describes no texture of, and the runtime
            // answers the typed `Nil` for the same descriptor: the same fact, so both sides answer
            // `false`. Anything else is our own refusal and is recorded as one.
            Err(metal2vulkan_runtime::Error::Nil(_)) => Ok(false),
            Err(error) => Err(Failure::of(&error)),
        }
    }

    fn sparse_tile_size(&mut self, query: &SparseTileQuery) -> Answer<TileSize> {
        let samples = u64::from(query.samples);
        let size = match query.page {
            None => self
                .device
                .sparse_tile_size(query.texture_type, query.format, samples),
            Some(page) => self.device.sparse_tile_size_for_page(
                query.texture_type,
                query.format,
                samples,
                page,
            ),
        };
        Ok(TileSize {
            width: size.width,
            height: size.height,
            depth: size.depth,
        })
    }

    fn sparse_tile_size_in_bytes(&mut self, page: QueryPage) -> Answer<u64> {
        Ok(match page {
            None => self.device.sparse_tile_size_in_bytes(),
            Some(page) => self.device.sparse_tile_size_in_bytes_for_page(page),
        })
    }

    fn sparse_heap(&mut self, spec: &SparseHeapSpec) -> Answer<Handle> {
        let mut descriptor =
            HeapDescriptor::new(HeapType::Sparse, spec.size, ResourceOptions::private());
        descriptor.sparse_page_size = spec.page;
        let heap = self
            .device
            .new_heap(&descriptor)
            .map_err(Failure::from_error)?;
        self.heaps.push(heap);
        Ok(Handle(self.heaps.len() as u32 - 1))
    }

    fn heap_facts(&mut self, heap: Handle) -> Answer<HeapFacts> {
        let heap = self.held_heap(heap)?;
        Ok(HeapFacts {
            heap_type: heap.heap_type().raw(),
            size: heap.size(),
            used_size: heap.used_size(),
        })
    }

    fn heap_texture(&mut self, heap: Handle, spec: &SparseTextureSpec) -> Answer<Handle> {
        let descriptor = sparse_texture_descriptor(spec);
        let texture = self
            .held_heap(heap)?
            .new_texture(&descriptor)
            .map_err(Failure::from_error)?;
        self.textures.push(texture);
        Ok(Handle(self.textures.len() as u32 - 1))
    }

    fn sparse_texture_facts(&mut self, texture: Handle) -> Answer<SparseTextureFacts> {
        let texture = self.held_texture(texture)?;
        Ok(SparseTextureFacts {
            is_sparse: texture.is_sparse(),
            // Metal answers both getters of every texture; the runtime answers `None` for one
            // that is not sparse, which is the same fact in the two APIs' spellings, and the
            // lane's own reading of it. A case asks them only of a sparse texture.
            first_mipmap_in_tail: texture.first_mipmap_in_tail().unwrap_or(0),
            tail_size_in_bytes: texture.tail_size_in_bytes().unwrap_or(0),
        })
    }
}

/// The runtime's descriptor for a case's sparse texture: the case's vocabulary is Metal's own, so
/// every field reaches the descriptor unchanged.
/// The runtime's texture descriptor for the one a case states: every field of
/// [`TextureRequest`](crate::texture::TextureRequest) reaches it.
///
/// A function of its own, and not inlined in the call, so that
/// `both_sides_state_every_field_of_a_texture_request` can hold it to that field by field with no
/// device — the swizzle most of all. A swizzle this side dropped while the Metal side set it would
/// leave every swizzle case passing for the wrong reason: both legs would be reading the texture's
/// unswizzled texels, and the comparison would be of two answers to a question neither leg was
/// asked.
///
/// [`crate::texture::Over`] is **not** here: it is which object the call is sent to, not a field of
/// the descriptor, and both sides send it in their own API's spelling.
pub(crate) fn texture_descriptor(
    request: &crate::texture::TextureRequest<'_>,
    format: PixelFormat,
) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(
        format,
        request.size[0].into(),
        request.size[1].into(),
        false,
    );
    descriptor.texture_type = request.texture_type;
    descriptor.depth = request.size[2].max(1).into();
    descriptor.mipmap_level_count = request.levels.max(1).into();
    descriptor.array_length = request.slices.max(1).into();
    descriptor.usage = request.usage;
    descriptor.resource_options = if request.private {
        ResourceOptions::private()
    } else {
        ResourceOptions::shared()
    };
    descriptor.allow_gpu_optimized_contents = request.gpu_optimized;
    descriptor.swizzle = request.swizzle;
    descriptor
}

fn sparse_texture_descriptor(spec: &SparseTextureSpec) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::default();
    descriptor.texture_type = spec.texture_type;
    descriptor.pixel_format = spec.format;
    descriptor.width = u64::from(spec.size[0]);
    descriptor.height = u64::from(spec.size[1]);
    descriptor.depth = u64::from(spec.size[2]);
    descriptor.mipmap_level_count = u64::from(spec.levels);
    descriptor.sample_count = u64::from(spec.samples);
    descriptor.array_length = u64::from(spec.array_length);
    descriptor.usage = if spec.render_target {
        TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET
    } else {
        TextureUsage::SHADER_READ
    };
    descriptor.resource_options = ResourceOptions::private();
    descriptor
}

/// The runtime's color attachment descriptor for one the case states: every field of
/// [`ColorState`] reaches it, because the case's vocabulary is Metal's own and the two sides must
/// give the two implementations the same descriptor.
///
/// It is a function of its own, and not inlined in the call, so that
/// `the_runtime_side_forwards_every_field_of_a_color_state` can hold it to that field by field
/// with no device. A field the Metal side sets and this one drops is a harness bug that reads as
/// our implementation's: the write mask was dropped here, which made the tile-store family's
/// masked draw write every channel on our side while Metal's wrote none.
pub(crate) fn color_descriptor(color: &ColorState) -> ColorAttachmentDescriptor {
    let mut made = ColorAttachmentDescriptor::new(color.format);
    made.write_mask = color.write_mask;
    if let Some(blend) = &color.blend {
        made.blending_enabled = true;
        made.source_rgb_blend_factor = blend.source_rgb;
        made.destination_rgb_blend_factor = blend.destination_rgb;
        made.rgb_blend_operation = blend.rgb;
        made.source_alpha_blend_factor = blend.source_alpha;
        made.destination_alpha_blend_factor = blend.destination_alpha;
        made.alpha_blend_operation = blend.alpha;
    }
    made
}

/// The runtime's descriptor for a queried one: the case's vocabulary is Metal's own, so every
/// field reaches the descriptor unchanged.
fn queried_descriptor(query: &HeapTextureQuery) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::default();
    descriptor.texture_type = query.texture_type;
    descriptor.pixel_format = query.format;
    descriptor.width = u64::from(query.size[0]);
    descriptor.height = u64::from(query.size[1]);
    descriptor.depth = u64::from(query.size[2]);
    descriptor.mipmap_level_count = u64::from(query.levels);
    descriptor.sample_count = u64::from(query.samples);
    descriptor.array_length = u64::from(query.array_length);
    descriptor.usage = query.usage;
    descriptor.resource_options = resource_options(query.storage);
    descriptor
}

/// The resource options a [`Storage`] names: the runtime side's one spelling of the three modes,
/// which [`crate::runtime_work`] reads too, so a case's storage mode cannot mean one thing to a
/// queried descriptor and another to a program's buffer.
pub fn resource_options(storage: Storage) -> ResourceOptions {
    match storage {
        Storage::Shared => ResourceOptions::shared(),
        Storage::Private => ResourceOptions::private(),
        Storage::Managed => ResourceOptions::new(
            metal2vulkan_runtime::StorageMode::Managed,
            metal2vulkan_runtime::CpuCacheMode::DefaultCache,
            metal2vulkan_runtime::HazardTrackingMode::Default,
        ),
    }
}

/// Encode one render pass of a program into `command_buffer`, answering the first refusal it met.
impl RuntimeSide<'_> {
    /// The buffer standing for the surface behind the texture `target`.
    fn held_plane(&self, target: Handle) -> Answer<&Buffer> {
        self.planes
            .iter()
            .find(|(held, _)| *held == target)
            .map(|(_, buffer)| buffer)
            .ok_or_else(|| {
                Failure::named(
                    "NotOverASurface",
                    format!("texture {} lies over no surface plane", target.0),
                )
            })
    }

    /// A target as a texture over a buffer the case made, through the runtime's own
    /// `newTextureWithDescriptor:offset:bytesPerRow:`.
    ///
    /// The stride is refused before the call for the same reason the Metal side refuses it: a
    /// `bytesPerRow` below the target's own row ends Metal's process, so neither side may make
    /// such a texture and both must refuse the same shape, or the comparison would be of two
    /// different questions.
    fn buffer_backed_target(
        &mut self,
        spec: &TargetSpec,
        over: &crate::render::OverBuffer,
    ) -> Answer<Handle> {
        if let Some(why) = over.unsafe_stride() {
            return Err(Failure::named("WouldEndMetalsProcess", why));
        }
        let mut descriptor = TextureDescriptor::texture_2d(
            spec.format,
            u64::from(spec.size[0]),
            u64::from(spec.size[1]),
            false,
        );
        let mut usage = TextureUsage::RENDER_TARGET;
        if spec.shader_read {
            usage |= TextureUsage::SHADER_READ;
        }
        if spec.shader_write {
            usage |= TextureUsage::SHADER_WRITE;
        }
        descriptor.usage = usage;
        // A texture over a buffer holds the buffer's own bytes, so its storage is the buffer's.
        descriptor.resource_options = ResourceOptions::shared();
        let buffer = self.held(over.buffer)?.clone();
        let texture = buffer
            .new_texture(&descriptor, over.offset, over.bytes_per_row)
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.textures, texture))
    }

    /// A framebuffer-only rotated texture over a buffer standing for an IOSurface plane
    /// (`Buffer::new_texture_with_iosurface_plane`), whose bytes start as the plane's pad byte.
    ///
    /// The runtime's stand-in is a buffer rather than a surface, which is the whole reason a case
    /// of this family runs where there is no IOSurface at all. The **stride rule Metal asserts on**
    /// is checked here too ([`crate::surface::PlaneSpec::unsafe_stride`]): not because the runtime
    /// would end the process, but so the two sides refuse the same planes and a case cannot pass on
    /// one side by asking a question the other was never allowed to be asked.
    fn surface_target(
        &mut self,
        spec: &TargetSpec,
        plane: &crate::surface::PlaneSpec,
    ) -> Answer<Handle> {
        if plane.unsafe_stride(spec.size) {
            return Err(Failure::named(
                "WouldEndMetalsProcess",
                format!(
                    "a plane of {} bytes a row behind a {} texel row",
                    plane.row_pitch,
                    spec.size[0] * plane.texel
                ),
            ));
        }
        let length = plane.length() as u64;
        let buffer = self
            .device
            .new_buffer(length, ResourceOptions::shared())
            .map_err(Failure::from_error)?;
        let contents = buffer.contents().ok_or_else(|| {
            Failure::named("NoContents", "the plane's buffer is not host-visible")
        })?;
        // SAFETY: the buffer was just made with `length` bytes of shared storage and no command
        // uses it yet.
        unsafe { std::ptr::write_bytes(contents.as_ptr(), plane.pad, length as usize) };
        let mut descriptor = TextureDescriptor::texture_2d(
            spec.format,
            u64::from(spec.size[0]),
            u64::from(spec.size[1]),
            false,
        );
        let mut usage = TextureUsage::RENDER_TARGET;
        if spec.shader_read {
            usage |= TextureUsage::SHADER_READ;
        }
        if spec.shader_write {
            usage |= TextureUsage::SHADER_WRITE;
        }
        descriptor.usage = usage;
        descriptor.resource_options = ResourceOptions::shared();
        descriptor.framebuffer_only = true;
        descriptor.rotation = u64::from(plane.rotation);
        let texture = buffer
            .new_texture_with_iosurface_plane(
                &descriptor,
                metal2vulkan_runtime::IoSurfacePlane::new(0, u64::from(plane.row_pitch), false),
            )
            .map_err(Failure::from_error)?;
        let handle = keep(&mut self.textures, texture);
        self.planes.push((handle, buffer));
        Ok(handle)
    }
}

fn encode_pass(
    command_buffer: &CommandBuffer,
    pass: &RenderPass,
    textures: &[Texture],
    buffers: &[Buffer],
    states: &[RenderPipelineState],
    depth_stencil: &[DepthStencilState],
    samplers: &[SamplerState],
) -> Answer<()> {
    let sampler = |handle: Handle| -> Answer<&SamplerState> {
        samplers
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("sampler {}", handle.0)))
    };
    let texture = |handle: Handle| -> Answer<&Texture> {
        textures
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("texture {}", handle.0)))
    };
    let buffer = |handle: Handle| -> Answer<&Buffer> {
        buffers
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("buffer {}", handle.0)))
    };
    let mut descriptor = RenderPassDescriptor::new();
    if let Some(size) = pass.size {
        descriptor.render_target_width = u64::from(size[0]);
        descriptor.render_target_height = u64::from(size[1]);
        descriptor.default_raster_sample_count = 1;
    }
    for (at, color) in pass.colors.iter().enumerate() {
        let Some(held) = color.texture else { continue };
        let attachment = descriptor
            .color_attachments
            .get_mut(at)
            .ok_or_else(|| Failure::named("NoSuchAttachment", format!("color {at}")))?;
        attachment.texture = Some(texture(held)?.clone());
        attachment.load_action = color.load;
        attachment.store_action = color.store;
        attachment.clear_color = ClearColor::new(
            color.clear[0],
            color.clear[1],
            color.clear[2],
            color.clear[3],
        );
        attachment.store_action_options = color.options;
        if let Some(resolve) = color.resolve {
            attachment.resolve_texture = Some(texture(resolve)?.clone());
        }
    }
    if let Some(depth) = &pass.depth {
        let attachment = &mut descriptor.depth_attachment;
        attachment.texture = Some(texture(depth.texture)?.clone());
        attachment.load_action = depth.load;
        attachment.store_action = depth.store;
        attachment.clear_depth = depth.clear;
        attachment.depth_resolve_filter = depth.filter;
        attachment.store_action_options = depth.options;
        if let Some(resolve) = depth.resolve {
            attachment.resolve_texture = Some(texture(resolve)?.clone());
        }
    }
    if let Some(stencil) = &pass.stencil {
        let attachment = &mut descriptor.stencil_attachment;
        attachment.texture = Some(texture(stencil.texture)?.clone());
        attachment.load_action = stencil.load;
        attachment.store_action = stencil.store;
        attachment.clear_stencil = stencil.clear;
        attachment.stencil_resolve_filter = stencil.filter;
        attachment.store_action_options = stencil.options;
        if let Some(resolve) = stencil.resolve {
            attachment.resolve_texture = Some(texture(resolve)?.clone());
        }
    }
    if let Some(held) = pass.visibility {
        descriptor.visibility_result_buffer = Some(buffer(held)?.clone());
    }
    // The runtime takes Metal's own positions where the case states none, and refuses what Metal
    // refuses, typed: no guard of the suite's stands between a case and that refusal, because the
    // refusal is what the case compares.
    descriptor.sample_positions = pass.positions.clone();
    let mut encoder = command_buffer
        .render_command_encoder(&descriptor)
        .map_err(Failure::from_error)?;
    for step in &pass.steps {
        match step {
            RenderStep::Pipeline(state) => {
                let state = states.get(state.0 as usize).ok_or_else(|| {
                    Failure::named("NoSuchHandle", format!("render pipeline {}", state.0))
                })?;
                encoder
                    .set_render_pipeline_state(state)
                    .map_err(Failure::from_error)?;
            }
            RenderStep::DepthStencilState(state) => {
                let state = depth_stencil.get(state.0 as usize).ok_or_else(|| {
                    Failure::named("NoSuchHandle", format!("depth/stencil state {}", state.0))
                })?;
                encoder
                    .set_depth_stencil_state(state)
                    .map_err(Failure::from_error)?;
            }
            RenderStep::StencilReference(value) => encoder.set_stencil_reference_value(*value),
            RenderStep::VertexBuffer {
                buffer: held,
                offset,
                index,
            } => encoder
                .set_vertex_buffer(Some(buffer(*held)?), *offset, *index)
                .map_err(Failure::from_error)?,
            RenderStep::FragmentBuffer {
                buffer: held,
                offset,
                index,
            } => encoder
                .set_fragment_buffer(Some(buffer(*held)?), *offset, *index)
                .map_err(Failure::from_error)?,
            RenderStep::FragmentTexture {
                texture: held,
                index,
            } => encoder
                .set_fragment_texture(Some(texture(*held)?), *index)
                .map_err(Failure::from_error)?,
            RenderStep::FragmentSampler {
                sampler: held,
                index,
            } => encoder
                .set_fragment_sampler_state(Some(sampler(*held)?), *index)
                .map_err(Failure::from_error)?,
            RenderStep::VertexBytes { bytes, index } => encoder
                .set_vertex_bytes(bytes, *index)
                .map_err(Failure::from_error)?,
            RenderStep::Viewport(rect) => encoder
                .set_viewport(Viewport::new(
                    rect.origin[0],
                    rect.origin[1],
                    rect.size[0],
                    rect.size[1],
                    rect.depth[0],
                    rect.depth[1],
                ))
                .map_err(Failure::from_error)?,
            RenderStep::VisibilityResultMode { mode, offset } => encoder
                .set_visibility_result_mode(*mode, *offset)
                .map_err(Failure::from_error)?,
            RenderStep::BlendColor(channels) => {
                let [red, green, blue, alpha] = channels.0;
                encoder.set_blend_color(red, green, blue, alpha);
            }
            RenderStep::Draw {
                primitive,
                first,
                count,
                instances,
            } => encoder
                .draw_primitives(*primitive, *first, *count, *instances, 0)
                .map_err(Failure::from_error)?,
            RenderStep::DrawIndexed {
                primitive,
                count,
                index_type,
                indices,
                offset,
                instances,
                base_vertex,
                base_instance,
            } => encoder
                .draw_indexed_primitives(
                    *primitive,
                    *count,
                    *index_type,
                    buffer(*indices)?,
                    *offset,
                    *instances,
                    *base_vertex,
                    *base_instance,
                )
                .map_err(Failure::from_error)?,
            RenderStep::DrawIndirect {
                primitive,
                arguments,
                offset,
            } => encoder
                .draw_primitives_indirect(*primitive, buffer(*arguments)?, *offset)
                .map_err(Failure::from_error)?,
        }
    }
    encoder.end_encoding().map_err(Failure::from_error)
}

/// The pixel format [`crate::case::TARGET_FORMAT`] names, as the runtime spells it.
const TARGET_PIXEL_FORMAT: PixelFormat = PixelFormat::RGBA32Uint;

/// The raw `MTLRenderStages` bits as the runtime takes them.
fn stages(bits: u64) -> Answer<RenderStages> {
    RenderStages::try_from(bits as u32)
        .map_err(|problem| Failure::named("NoSuchRenderStages", format!("{bits:#x}: {problem}")))
}

/// The raw `MTLBarrierScope` bits as the runtime takes them.
fn scope_bits(bits: u64) -> Answer<BarrierScope> {
    BarrierScope::try_from(bits as u32)
        .map_err(|problem| Failure::named("NoSuchBarrierScope", format!("{bits:#x}: {problem}")))
}

impl Failure {
    /// The failure a typed runtime error is, for a call that returns one by value.
    fn from_error(error: metal2vulkan_runtime::Error) -> Self {
        Failure::of(&error)
    }
}

/// The raw primitive type a draw holds; a draw with no primitive type (patches, mesh) reads as
/// points, as Metal's own getters answer such a record.
fn drawn_primitive_type(draw: IndirectDraw) -> u64 {
    match draw {
        IndirectDraw::Primitives { primitive_type, .. }
        | IndirectDraw::IndexedPrimitives { primitive_type, .. } => primitive_type.raw(),
        _ => DrawPrimitiveType::from(PrimitiveType::Point).raw(),
    }
}
