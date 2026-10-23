//! The runtime side of the case vocabulary: `metal2vulkan_runtime` on whatever Vulkan device the
//! host has.
//!
//! Every call answers the runtime's typed error where it refuses, so a case records what it got
//! rather than panicking, and the comparison can tell our refusal from the venue's honest gap.
//!
//! No wait here can hang: a command buffer is waited for with a bound, and a wait that runs out
//! answers `NeverReturned` as a recorded fact rather than blocking the suite.

use crate::case::{Answer, Failure, Handle, IndirectReadBack, ShaderSource, Side, Venue};
use crate::transcript::{Negative, Value};
use metal2vulkan::metal::{IndexType, IndirectCommandType, PrimitiveType};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Device,
    DrawPrimitiveType, IndirectCommand, IndirectDraw, Library, PipelineOptions, ResourceOptions,
    Size,
};
use std::time::Duration;

/// The longest the side waits for a command buffer. A wait that runs out is recorded, not retried.
const WAIT: Duration = Duration::from_secs(60);

/// The runtime side: the device, the queue the case's command buffers come from, and the objects
/// the case has made, by handle.
pub struct RuntimeSide<'a> {
    device: &'a Device,
    queue: Option<CommandQueue>,
    libraries: Vec<Library>,
    pipelines: Vec<ComputePipelineState>,
    buffers: Vec<Buffer>,
    command_buffers: Vec<CommandBuffer>,
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
            command_buffers: Vec::new(),
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

    fn dispatch(
        &mut self,
        pipeline: Handle,
        buffers: &[Handle],
        threadgroups: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle> {
        let bound: Vec<Buffer> = buffers
            .iter()
            .map(|handle| self.held(*handle).cloned())
            .collect::<Answer<Vec<Buffer>>>()?;
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
        let size = |axes: [u32; 3]| Size::new(axes[0].into(), axes[1].into(), axes[2].into());
        encoder
            .dispatch_threadgroups(size(threadgroups), size(threads))
            .map_err(Failure::from_error)?;
        encoder.end_encoding().map_err(Failure::from_error)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        Ok(keep(&mut self.command_buffers, command_buffer))
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
