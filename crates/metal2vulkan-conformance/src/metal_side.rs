//! The Metal side of the case vocabulary: native Metal through `objc2-metal`, on macOS only.
//!
//! This is what record mode runs. It exists so a case's Metal answer is taken by running the case
//! rather than copied off a terminal: the same case body, the same sites, the same transcript
//! shape.
//!
//! A shader reaches Metal as its fixture's typed-pointer AIR text: `xcrun metal-as` reassembles it
//! and `xcrun metallib` packages the library, which `newLibraryWithURL:` loads. Apple's assembler
//! reads that dialect and not the one a current `llvm-as` writes, which is why a fixture commits
//! both forms.
//!
//! Metal announces no degrade notice, so this side announces none: a notice in a transcript is
//! ours, and a case that holds one must name what justifies it.

use crate::case::{
    Answer, ArgumentSlot, Bindings, CommandDraw, Failure, FragmentFill, FragmentFilled, Handle,
    MeshCommand, MeshFunctions, MeshStep, PassOutcome, RateMapFacts, RateMapRequest, SamplerSpec,
    ShaderSource, Side, StageInRegion, StageInput, TextureRole, TextureSpec, TileFacts,
    TilePipelineSpec, TileStep, Venue,
};
use crate::queue_slots::QueueSlots;
use crate::render::{
    Dispatch, RenderPass, RenderPipelineSpec, RenderStep, RenderWork, StageFunction, TargetSpec,
};
use crate::sparse_query::{
    HeapFacts, QueryPage, SparseHeapSpec, SparseTextureFacts, SparseTextureSpec, SparseTileQuery,
    TileSize,
};
use crate::texture_query::{HeapTextureQuery, SizeAndAlign, Storage};
use crate::transcript::Value;
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{BindingAccess, PixelFormat, RenderStages};
use metal2vulkan_runtime::SparsePageSize;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{
    MTLAttributeFormat, MTLBarrierScope, MTLBinding, MTLBindingAccess, MTLBindingType,
    MTLBlendFactor, MTLBlendOperation, MTLBlitCommandEncoder, MTLBuffer, MTLClearColor,
    MTLColorWriteMask, MTLCommandBuffer, MTLCommandBufferHandler, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLCompareFunction, MTLComputeCommandEncoder,
    MTLComputePipelineDescriptor, MTLComputePipelineState, MTLCreateSystemDefaultDevice,
    MTLDepthStencilDescriptor, MTLDepthStencilState, MTLDevice, MTLEvent, MTLFunction, MTLHeap,
    MTLHeapDescriptor, MTLHeapType, MTLIndexType, MTLIndirectCommandBuffer,
    MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType, MTLIndirectRenderCommand,
    MTLLibrary, MTLLoadAction, MTLMeshRenderPipelineDescriptor, MTLMultisampleDepthResolveFilter,
    MTLMultisampleStencilResolveFilter, MTLOrigin, MTLPipelineOption, MTLPixelFormat,
    MTLPrimitiveType, MTLRasterizationRateLayerDescriptor, MTLRasterizationRateMap,
    MTLRasterizationRateMapDescriptor, MTLRegion, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineReflection, MTLRenderPipelineState,
    MTLRenderStages, MTLResource, MTLResourceOptions, MTLResourceUsage, MTLSamplePosition,
    MTLSamplerAddressMode, MTLSamplerBorderColor, MTLSamplerDescriptor, MTLSamplerMinMagFilter,
    MTLSamplerState, MTLSharedEvent, MTLSize, MTLSparsePageSize, MTLStageInputOutputDescriptor,
    MTLStencilDescriptor, MTLStencilOperation, MTLStepFunction, MTLStorageMode, MTLStoreAction,
    MTLStoreActionOptions, MTLTexture, MTLTextureDescriptor, MTLTextureSwizzle,
    MTLTextureSwizzleChannels, MTLTextureType, MTLTextureUsage, MTLTileRenderPipelineDescriptor,
    MTLVertexDescriptor, MTLVertexFormat, MTLVertexStepFunction, MTLViewport,
    MTLVisibilityResultMode,
};

/// The Metal level this side answers at: the host's own, since Metal answers as the release it runs
/// on.
const METAL_LEVEL: &str = crate::recording::ORACLE_OS_VERSION;

/// Native Metal, with the objects the case has made.
pub struct MetalSide {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    libraries: Vec<Retained<ProtocolObject<dyn MTLLibrary>>>,
    pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    textures: Vec<Retained<ProtocolObject<dyn MTLTexture>>>,
    /// The sampler states the case has made. Its own pool: a sampler state is its own Metal
    /// object, not another name for one of the pools above.
    samplers: Vec<Retained<ProtocolObject<dyn MTLSamplerState>>>,
    command_buffers: Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    /// The render pipeline states the case has made, each with the reflection its creation
    /// answered where the case asked for one. One pool, because a mesh state, a tile state and a
    /// graphics state are one Metal object (`MTLRenderPipelineState`) made from three descriptors.
    render_pipelines: Vec<RenderPipeline>,
    /// The indirect command buffers the case has made.
    command_buffers_indirect: Vec<Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>>,
    /// The rasterization-rate maps the case has made. Its own pool: a rate map is its own
    /// Metal object, not another name for one of the pools above.
    rate_maps: Vec<Retained<ProtocolObject<dyn MTLRasterizationRateMap>>>,
    /// For each argument buffer the case has made, the textures it names: Metal keeps a resource a
    /// kernel reaches through an argument buffer resident only where the encoder was told so, so
    /// binding the buffer makes them resident. Not a pool — the buffer itself lives in `buffers`.
    argument_residents: Vec<(Handle, Vec<Handle>)>,
    /// The depth and stencil states the case has made. Its own pool: an `MTLDepthStencilState` is
    /// its own Metal object, not another name for one of the pools above.
    depth_stencil_states: Vec<Retained<ProtocolObject<dyn MTLDepthStencilState>>>,
    /// The heaps the case has made. Its own pool: an `MTLHeap` is its own Metal object, not
    /// another name for one of the pools above, and not another name for the depth and stencil
    /// states either.
    heaps: Vec<Retained<ProtocolObject<dyn MTLHeap>>>,
    /// The IOSurfaces the case's plane textures lie over. Not a pool a handle names: a surface is
    /// not a case's object at all — the case states a buffer, and this side turns it into the
    /// surface Metal needs. They are kept so a surface outlives the texture that reads it.
    surfaces: Vec<crate::metal_surface::Surface>,
    /// The scratch directory the metallibs are built in, removed when the side is dropped.
    scratch: Scratch,
}

/// One render pipeline state of the case's, with the reflection its creation answered.
///
/// The reflection lives here, beside the state, rather than in a pool of its own, because **Metal
/// answers a reflection only at the state's creation**: `MTLRenderPipelineState`'s own
/// `reflection` property is nil for a state an `MTLDevice` made. So a handle has to stand for the
/// pair, and the state is kept so it outlives its own reflection. It is `None` for a state whose
/// creation was not asked for one, which is every state but
/// [`Side::render_pipeline_with_reflection`]'s.
struct RenderPipeline {
    state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    reflection: Option<Retained<MTLRenderPipelineReflection>>,
}

impl RenderPipeline {
    /// A state whose creation was not asked for a reflection.
    fn unreflected(state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>) -> Self {
        Self {
            state,
            reflection: None,
        }
    }
}

impl MetalSide {
    /// The system default device and a queue on it.
    pub fn new() -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| "MTLCreateSystemDefaultDevice returned nil".to_owned())?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| "MTLDevice::newCommandQueue returned nil".to_owned())?;
        Ok(Self {
            device,
            queue,
            libraries: Vec::new(),
            pipelines: Vec::new(),
            buffers: Vec::new(),
            textures: Vec::new(),
            samplers: Vec::new(),
            command_buffers: Vec::new(),
            render_pipelines: Vec::new(),
            command_buffers_indirect: Vec::new(),
            rate_maps: Vec::new(),
            argument_residents: Vec::new(),
            depth_stencil_states: Vec::new(),
            heaps: Vec::new(),
            surfaces: Vec::new(),
            scratch: Scratch::new()?,
        })
    }

    /// The buffer `handle` names.
    fn held(&self, handle: Handle) -> Answer<&ProtocolObject<dyn MTLBuffer>> {
        self.buffers
            .get(handle.0 as usize)
            .map(|buffer| &**buffer)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("buffer {}", handle.0)))
    }

    /// The texture `handle` names.
    fn held_texture(&self, handle: Handle) -> Answer<&ProtocolObject<dyn MTLTexture>> {
        self.textures
            .get(handle.0 as usize)
            .map(|texture| &**texture)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("texture {}", handle.0)))
    }

    /// The sampler state `handle` names.
    fn held_sampler(&self, handle: Handle) -> Answer<&ProtocolObject<dyn MTLSamplerState>> {
        self.samplers
            .get(handle.0 as usize)
            .map(|sampler| &**sampler)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("sampler {}", handle.0)))
    }

    /// The heap `handle` names.
    fn held_heap(&self, handle: Handle) -> Answer<&ProtocolObject<dyn MTLHeap>> {
        self.heaps
            .get(handle.0 as usize)
            .map(|heap| &**heap)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("heap {}", handle.0)))
    }

    /// The rasterization-rate map `handle` names.
    fn held_rate_map(
        &self,
        handle: Handle,
    ) -> Answer<&ProtocolObject<dyn MTLRasterizationRateMap>> {
        self.rate_maps
            .get(handle.0 as usize)
            .map(|map| &**map)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("rate map {}", handle.0)))
    }

    /// Level 0 of `texture` read back through a blit into a shared buffer of its own, at
    /// `row_pitch` bytes a row: what a private texture's bytes are, since the host reaches none of
    /// them directly.
    fn blit_out(
        &mut self,
        texture: &ProtocolObject<dyn MTLTexture>,
        size: [u32; 2],
        row_pitch: u32,
    ) -> Answer<Vec<u8>> {
        let length = row_pitch as usize * size[1] as usize;
        let back = self
            .device
            .newBufferWithLength_options(length.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                Failure::named(
                    "NewBufferWithLength",
                    format!("{length} bytes returned nil"),
                )
            })?;
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        let blit = command_buffer
            .blitCommandEncoder()
            .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil".to_owned()))?;
        // SAFETY: the buffer holds `size[1]` rows of `row_pitch` bytes, which is what the copy
        // writes, and the texture holds level 0 of that extent.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                texture,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize {
                    width: size[0] as usize,
                    height: size[1] as usize,
                    depth: 1,
                },
                &back,
                0,
                row_pitch as usize,
                length,
            );
        }
        blit.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let mut bytes = vec![0u8; length];
        // SAFETY: the buffer is shared storage of at least `length` bytes and the copy into it has
        // completed.
        unsafe {
            std::ptr::copy_nonoverlapping(
                back.contents().as_ptr().cast::<u8>(),
                bytes.as_mut_ptr(),
                length,
            );
        }
        keep(&mut self.buffers, back);
        keep(&mut self.command_buffers, command_buffer);
        Ok(bytes)
    }

    /// One command buffer with one compute encoder: the pipeline set, `bindings` bound at 0 up,
    /// each of `stage_in` set in order, `dispatch` encoded, and `commit`. The two dispatch calls
    /// share this body, so neither side can pick up a different binding order with one of them.
    fn encode(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        stage_in: &[StageInRegion],
        dispatch: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) -> Answer<Handle> {
        let bound: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = bindings
            .buffers
            .iter()
            .map(|handle| self.held(*handle).map(Retained::from))
            .collect::<Answer<Vec<_>>>()?;
        let sampled: Vec<Retained<ProtocolObject<dyn MTLTexture>>> = bindings
            .textures
            .iter()
            .map(|handle| self.held_texture(*handle).map(Retained::from))
            .collect::<Answer<Vec<_>>>()?;
        let states: Vec<Retained<ProtocolObject<dyn MTLSamplerState>>> = bindings
            .samplers
            .iter()
            .map(|handle| self.held_sampler(*handle).map(Retained::from))
            .collect::<Answer<Vec<_>>>()?;
        // The textures the bound argument buffers name: Metal keeps a resource reached through an
        // argument buffer resident only where the encoder was told so.
        let resident: Vec<Retained<ProtocolObject<dyn MTLResource>>> = self
            .argument_residents
            .iter()
            .filter(|(buffer, _)| bindings.buffers.contains(buffer))
            .flat_map(|(_, textures)| textures.clone())
            .map(|handle| {
                self.held_texture(handle).map(|texture| {
                    let resource: &ProtocolObject<dyn MTLResource> =
                        ProtocolObject::from_ref(texture);
                    Retained::from(resource)
                })
            })
            .collect::<Answer<Vec<_>>>()?;
        let state = self
            .pipelines
            .get(pipeline.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", pipeline.0)))?
            .clone();
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| Failure::named("ComputeCommandEncoder", "returned nil".to_owned()))?;
        encoder.setComputePipelineState(&state);
        for (at, buffer) in bound.iter().enumerate() {
            // SAFETY: the buffer is at least as long as the kernel's binding, and the encoder
            // outlives the call.
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, at) };
        }
        for (at, texture) in sampled.iter().enumerate() {
            // SAFETY: the texture is of the kind and access the kernel declares at that slot, and
            // the encoder outlives the call.
            unsafe { encoder.setTexture_atIndex(Some(texture), at) };
        }
        for (at, sampler) in states.iter().enumerate() {
            // SAFETY: the kernel declares a sampler at that slot, and the encoder outlives the
            // call.
            unsafe { encoder.setSamplerState_atIndex(Some(sampler), at) };
        }
        for resource in &resident {
            encoder.useResource_usage(resource, MTLResourceUsage::Read);
        }
        for region in stage_in {
            match region {
                StageInRegion::Direct { origin, size } => {
                    encoder.setStageInRegion(MTLRegion {
                        origin: MTLOrigin {
                            x: origin[0] as usize,
                            y: origin[1] as usize,
                            z: origin[2] as usize,
                        },
                        size: MTLSize {
                            width: size[0] as usize,
                            height: size[1] as usize,
                            depth: size[2] as usize,
                        },
                    });
                }
                StageInRegion::Indirect { buffer, offset } => {
                    let held = self.held(*buffer)?;
                    // SAFETY: the buffer holds the region's six words from `offset`, which the
                    // case states as one of its inputs, and the encoder outlives the call.
                    unsafe {
                        encoder.setStageInRegionWithIndirectBuffer_indirectBufferOffset(
                            held,
                            *offset as usize,
                        );
                    }
                }
            }
        }
        dispatch(&encoder);
        encoder.endEncoding();
        command_buffer.commit();
        Ok(keep(&mut self.command_buffers, command_buffer))
    }
}

/// Three axes as Metal's size.
fn size(axes: [u32; 3]) -> MTLSize {
    MTLSize {
        width: axes[0] as usize,
        height: axes[1] as usize,
        depth: axes[2] as usize,
    }
}

/// `MTLBufferLayoutStrideDynamic`, the stride a layout takes from the encoder's binding.
const STRIDE_DYNAMIC: usize = usize::MAX;

/// Metal's vertex descriptor for `layout`. The case's enumerations are Metal's own, so each one
/// crosses as its raw value.
fn vertex_descriptor(layout: &crate::render::VertexLayout<'_>) -> Retained<MTLVertexDescriptor> {
    let descriptor = MTLVertexDescriptor::vertexDescriptor();
    for attribute in layout.attributes {
        // SAFETY: the index is below the attribute array every descriptor holds, which the case's
        // own guard keeps it to.
        let held = unsafe {
            descriptor
                .attributes()
                .objectAtIndexedSubscript(attribute.index as usize)
        };
        held.setFormat(MTLVertexFormat(attribute.format.as_raw() as usize));
        // SAFETY: the offset and the buffer index are the case's own numbers, both inside the
        // record the case binds and the buffer array the descriptor holds.
        unsafe {
            held.setOffset(attribute.offset as usize);
            held.setBufferIndex(attribute.buffer as usize);
        }
    }
    for buffer in layout.layouts {
        // SAFETY: as the attributes', for the layout array.
        let held = unsafe {
            descriptor
                .layouts()
                .objectAtIndexedSubscript(buffer.index as usize)
        };
        held.setStepFunction(MTLVertexStepFunction(buffer.step.as_raw() as usize));
        // SAFETY: the stride is the case's own record size and the rate is the step function's
        // own, 0 for the constant step and 1 otherwise.
        unsafe {
            held.setStride(buffer.stride as usize);
            held.setStepRate(buffer.rate as usize);
        }
    }
    descriptor
}

/// Metal's stage-input descriptor for `stage_input`. The case's enumerations are Metal's own, so
/// each reaches `MTLAttributeFormat`, `MTLStepFunction` and `MTLIndexType` by its raw ordinal.
fn stage_input_descriptor(stage_input: &StageInput) -> Retained<MTLStageInputOutputDescriptor> {
    let descriptor = MTLStageInputOutputDescriptor::stageInputOutputDescriptor();
    let attributes = descriptor.attributes();
    for attribute in &stage_input.attributes {
        // SAFETY: `objectAtIndexedSubscript:` is Metal's own accessor for an attribute index the
        // device's attribute limit covers, which the case's indices are inside.
        let written = unsafe { attributes.objectAtIndexedSubscript(attribute.index as usize) };
        written.setFormat(MTLAttributeFormat(attribute.format.as_raw() as usize));
        written.setOffset(attribute.offset as usize);
        // SAFETY: the buffer index is inside the device's buffer argument table, which the
        // case's indices are.
        unsafe { written.setBufferIndex(attribute.buffer_index as usize) };
    }
    let layouts = descriptor.layouts();
    for layout in &stage_input.layouts {
        // SAFETY: as above, for a buffer index inside the device's buffer limit.
        let written = unsafe { layouts.objectAtIndexedSubscript(layout.buffer_index as usize) };
        written.setStride(
            layout
                .stride
                .map_or(STRIDE_DYNAMIC, |stride| stride as usize),
        );
        written.setStepFunction(MTLStepFunction(layout.step_function.as_raw() as usize));
        written.setStepRate(layout.step_rate as usize);
    }
    descriptor.setIndexType(MTLIndexType(stage_input.index_type.as_raw() as usize));
    // SAFETY: as above, for the index buffer's own slot.
    unsafe { descriptor.setIndexBufferIndex(stage_input.index_buffer_index as usize) };
    descriptor
}

/// `kept` with `value` pushed on, answering its index.
/// The `MTLTextureSwizzle` a channel's source is, by the raw ordinal both enumerations share.
fn swizzle_of(source: metal2vulkan::metal::TextureSwizzle) -> MTLTextureSwizzle {
    MTLTextureSwizzle(source.as_raw() as u8)
}

fn keep<T>(kept: &mut Vec<T>, value: T) -> Handle {
    kept.push(value);
    Handle(u32::try_from(kept.len() - 1).expect("fewer than 2^32 objects"))
}

/// A directory the metallibs are built in.
struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new() -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!(
            "metal2vulkan-conformance-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|since| since.as_nanos())
                .unwrap_or_default()
        ));
        std::fs::create_dir_all(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        Ok(Self(path))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Run `xcrun tool args`, answering why it could not.
fn xcrun(tool: &str, args: &[&std::ffi::OsStr]) -> Result<(), String> {
    let output = std::process::Command::new("/usr/bin/xcrun")
        .arg(tool)
        .args(args)
        .output()
        .map_err(|error| format!("xcrun {tool}: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "xcrun {tool} failed: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

impl Side for MetalSide {
    fn venue(&self) -> Venue {
        Venue {
            device: self.device.name().to_string(),
            driver: "Metal".to_owned(),
            api: "Metal".to_owned(),
            metal_level: METAL_LEVEL.to_owned(),
        }
    }

    fn library(&mut self, source: &ShaderSource<'_>) -> Answer<Handle> {
        let serial = self.libraries.len();
        let text = self.scratch.0.join(format!("{serial}.air.ll"));
        let air = self.scratch.0.join(format!("{serial}.air"));
        let metallib = self.scratch.0.join(format!("{serial}.metallib"));
        let build = || -> Result<(), String> {
            std::fs::write(&text, source.air_text)
                .map_err(|error| format!("{}: {error}", text.display()))?;
            xcrun(
                "metal-as",
                &[text.as_os_str(), "-o".as_ref(), air.as_os_str()],
            )?;
            xcrun(
                "metallib",
                &[air.as_os_str(), "-o".as_ref(), metallib.as_os_str()],
            )
        };
        build().map_err(|why| Failure::named("MetalToolFailed", why))?;
        let path = metallib
            .to_str()
            .ok_or_else(|| Failure::named("NonUtf8Path", metallib.display().to_string()))?;
        let library = self
            .device
            .newLibraryWithURL_error(&NSURL::fileURLWithPath(&NSString::from_str(path)))
            .map_err(|error| Failure::named("NewLibraryWithURL", error.to_string()))?;
        Ok(keep(&mut self.libraries, library))
    }

    fn compute_pipeline(&mut self, library: Handle, entry: &str) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or_else(|| Failure::named("NoSuchFunction", entry.to_owned()))?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| Failure::named("NewComputePipelineState", error.to_string()))?;
        Ok(keep(&mut self.pipelines, pipeline))
    }

    fn static_threadgroup_memory_length(&mut self, pipeline: Handle) -> Answer<u64> {
        let state = self
            .pipelines
            .get(pipeline.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", pipeline.0)))?;
        Ok(state.staticThreadgroupMemoryLength() as u64)
    }

    fn buffer(&mut self, bytes: &[u8]) -> Answer<Handle> {
        let handle = self.zeroed_buffer(bytes.len())?;
        let buffer = self.held(handle)?;
        // SAFETY: the buffer was just made with at least `bytes.len()` bytes of shared storage,
        // and no command uses it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.contents().as_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
        Ok(handle)
    }

    fn zeroed_buffer(&mut self, length: usize) -> Answer<Handle> {
        let buffer = self
            .device
            .newBufferWithLength_options(length.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                Failure::named(
                    "NewBufferWithLength",
                    format!("{length} bytes returned nil"),
                )
            })?;
        Ok(keep(&mut self.buffers, buffer))
    }

    fn texture_2d(
        &mut self,
        format: PixelFormat,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<Handle> {
        // SAFETY: the descriptor constructor reads only the format and the size it is given.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat(format.as_raw() as usize),
                size[0] as usize,
                size[1] as usize,
                false,
            )
        };
        descriptor.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        descriptor.setStorageMode(MTLStorageMode::Shared);
        let texture = self
            .device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptor",
                    format!("{format:?} {}x{} returned nil", size[0], size[1]),
                )
            })?;
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: size[0] as usize,
                height: size[1] as usize,
                depth: 1,
            },
        };
        if bytes.len() < row_pitch as usize * size[1] as usize {
            return Err(Failure::named(
                "TooFewBytes",
                format!("{} bytes for {} rows of {row_pitch}", bytes.len(), size[1]),
            ));
        }
        // SAFETY: `bytes` holds `size[1]` rows of `row_pitch` bytes, checked just above, which is
        // the region the call writes.
        unsafe {
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region,
                0,
                std::ptr::NonNull::from(bytes).cast(),
                row_pitch as usize,
            );
        }
        Ok(keep(&mut self.textures, texture))
    }

    fn read_texture(&mut self, texture: Handle, size: [u32; 2], row_pitch: u32) -> Answer<Vec<u8>> {
        let texture = self.held_texture(texture)?;
        let mut bytes = vec![0u8; row_pitch as usize * size[1] as usize];
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: size[0] as usize,
                height: size[1] as usize,
                depth: 1,
            },
        };
        // SAFETY: `bytes` is `size[1]` rows of `row_pitch` bytes, which is the region the call
        // reads, and the caller waited for the command buffer that wrote the texture.
        unsafe {
            texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                std::ptr::NonNull::from(bytes.as_mut_slice()).cast(),
                row_pitch as usize,
                region,
                0,
            );
        }
        Ok(bytes)
    }

    fn texture_2d_in_role(&mut self, spec: &TextureSpec, bytes: &[u8]) -> Answer<Handle> {
        // SAFETY: the descriptor constructor reads only the format and the size it is given.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat(spec.format.as_raw() as usize),
                spec.size[0] as usize,
                spec.size[1] as usize,
                false,
            )
        };
        descriptor.setUsage(match spec.role {
            TextureRole::Read => MTLTextureUsage::ShaderRead,
            TextureRole::Write => MTLTextureUsage::ShaderWrite,
            TextureRole::ReadWrite => MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
            TextureRole::RenderTarget => MTLTextureUsage::RenderTarget,
        });
        descriptor.setStorageMode(MTLStorageMode::Shared);
        let texture = self
            .device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptor",
                    format!(
                        "{:?} {}x{} returned nil",
                        spec.format, spec.size[0], spec.size[1]
                    ),
                )
            })?;
        if !bytes.is_empty() {
            if bytes.len() < spec.row_pitch as usize * spec.size[1] as usize {
                return Err(Failure::named(
                    "TooFewBytes",
                    format!(
                        "{} bytes for {} rows of {}",
                        bytes.len(),
                        spec.size[1],
                        spec.row_pitch
                    ),
                ));
            }
            let region = MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize {
                    width: spec.size[0] as usize,
                    height: spec.size[1] as usize,
                    depth: 1,
                },
            };
            // SAFETY: `bytes` holds the rows of the pitch the region writes, checked just above.
            unsafe {
                texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                    region,
                    0,
                    std::ptr::NonNull::from(bytes).cast(),
                    spec.row_pitch as usize,
                );
            }
        }
        Ok(keep(&mut self.textures, texture))
    }

    fn texture(&mut self, request: &crate::texture::TextureRequest<'_>) -> Answer<Handle> {
        let descriptor = MTLTextureDescriptor::new();
        descriptor.setPixelFormat(MTLPixelFormat(request.format as usize));
        descriptor.setTextureType(MTLTextureType(request.texture_type.as_raw() as usize));
        // SAFETY: the setters take the descriptor's own words; nothing is borrowed.
        unsafe {
            descriptor.setWidth(request.size[0] as usize);
            descriptor.setHeight(request.size[1] as usize);
            descriptor.setDepth(request.size[2].max(1) as usize);
            descriptor.setMipmapLevelCount(request.levels.max(1) as usize);
            descriptor.setArrayLength(request.slices.max(1) as usize);
        }
        descriptor.setUsage(MTLTextureUsage(request.usage.as_raw() as usize));
        descriptor.setStorageMode(if request.private {
            MTLStorageMode::Private
        } else {
            MTLStorageMode::Shared
        });
        descriptor.setAllowGPUOptimizedContents(request.gpu_optimized);
        descriptor.setSwizzle(MTLTextureSwizzleChannels {
            red: swizzle_of(request.swizzle.red),
            green: swizzle_of(request.swizzle.green),
            blue: swizzle_of(request.swizzle.blue),
            alpha: swizzle_of(request.swizzle.alpha),
        });
        let texture = self
            .device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptor",
                    format!(
                        "format {} {:?} {}x{}x{} returned nil",
                        request.format,
                        request.texture_type,
                        request.size[0],
                        request.size[1],
                        request.size[2]
                    ),
                )
            })?;
        for upload in request.uploads {
            if upload.bytes.len() < upload.needed() {
                return Err(Failure::named(
                    "TooFewBytes",
                    format!(
                        "{} bytes for {} rows of {}",
                        upload.bytes.len(),
                        upload.rows,
                        upload.row_pitch
                    ),
                ));
            }
            let region = MTLRegion {
                origin: MTLOrigin {
                    x: upload.origin[0] as usize,
                    y: upload.origin[1] as usize,
                    z: upload.origin[2] as usize,
                },
                size: MTLSize {
                    width: upload.size[0] as usize,
                    height: upload.size[1] as usize,
                    depth: upload.size[2].max(1) as usize,
                },
            };
            // SAFETY: `upload.bytes` holds the rows of the pitch the region writes, checked just
            // above, and the region lies in the level and slice the texture was made with.
            unsafe {
                texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    region,
                    upload.level as usize,
                    upload.slice as usize,
                    std::ptr::NonNull::from(upload.bytes).cast(),
                    upload.row_pitch as usize,
                    upload.image_pitch as usize,
                );
            }
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
        if bytes.len() < row_pitch as usize * size[1] as usize {
            return Err(Failure::named(
                "TooFewBytes",
                format!("{} bytes for {} rows of {row_pitch}", bytes.len(), size[1]),
            ));
        }
        let texture: Retained<ProtocolObject<dyn MTLTexture>> =
            Retained::from(self.held_texture(texture)?);
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: size[0] as usize,
                height: size[1] as usize,
                depth: 1,
            },
        };
        // SAFETY: `bytes` holds `size[1]` rows of `row_pitch` bytes, checked just above, which is
        // the region the call writes.
        unsafe {
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region,
                0,
                std::ptr::NonNull::from(bytes).cast(),
                row_pitch as usize,
            );
        }
        Ok(())
    }

    fn sampler(&mut self, spec: &SamplerSpec) -> Answer<Handle> {
        let filter = |filter: metal2vulkan::metal::SamplerMinMagFilter| {
            MTLSamplerMinMagFilter(filter.as_raw() as usize)
        };
        let address = |mode: metal2vulkan::metal::SamplerAddressMode| {
            MTLSamplerAddressMode(mode.as_raw() as usize)
        };
        let descriptor = MTLSamplerDescriptor::new();
        descriptor.setMinFilter(filter(spec.min_filter));
        descriptor.setMagFilter(filter(spec.mag_filter));
        descriptor.setSAddressMode(address(spec.s_address_mode));
        descriptor.setTAddressMode(address(spec.t_address_mode));
        descriptor.setBorderColor(MTLSamplerBorderColor(spec.border_color.as_raw() as usize));
        descriptor.setNormalizedCoordinates(spec.normalized_coordinates);
        descriptor.setSupportArgumentBuffers(spec.support_argument_buffers);
        let sampler = self
            .device
            .newSamplerStateWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewSamplerStateWithDescriptor",
                    format!("{spec:?} returned nil"),
                )
            })?;
        Ok(keep(&mut self.samplers, sampler))
    }

    fn argument_buffer(&mut self, slots: &[ArgumentSlot]) -> Answer<Handle> {
        let mut words = Vec::with_capacity(slots.len() * 8);
        let mut named = Vec::new();
        for slot in slots {
            let id = match *slot {
                ArgumentSlot::Texture(handle) => {
                    named.push(handle);
                    self.held_texture(handle)?.gpuResourceID().to_raw()
                }
                ArgumentSlot::Sampler(handle) => {
                    self.held_sampler(handle)?.gpuResourceID().to_raw()
                }
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
            encoder.dispatchThreadgroups_threadsPerThreadgroup(size(threadgroups), size(threads));
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
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or_else(|| Failure::named("NoSuchFunction", entry.to_owned()))?;
        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(&function));
        descriptor.setStageInputDescriptor(Some(&stage_input_descriptor(stage_input)));
        let pipeline = self
            .device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|error| Failure::named("NewComputePipelineState", error.to_string()))?;
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
            encoder.dispatchThreads_threadsPerThreadgroup(size(grid), size(threads));
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
        command_buffer.waitUntilCompleted();
        Ok(Value::unit_variant(status_word(command_buffer.status())))
    }

    fn read_buffer(&mut self, buffer: Handle, offset: u64, length: usize) -> Answer<Vec<u8>> {
        let buffer = self.held(buffer)?;
        let past = offset
            .checked_add(length as u64)
            .is_none_or(|end| end > buffer.length() as u64);
        if past {
            return Err(Failure::named(
                "PastTheBuffer",
                format!(
                    "{length} bytes at {offset} of a buffer of {}",
                    buffer.length()
                ),
            ));
        }
        // SAFETY: `[offset, offset + length)` is inside the buffer's shared storage, and the
        // caller waited for the command buffer that wrote it.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                buffer.contents().as_ptr().cast::<u8>().add(offset as usize),
                length,
            )
        };
        Ok(bytes.to_vec())
    }

    /// A copy-and-fill program, through public Metal alone ([`crate::metal_work`]).
    ///
    /// The fills (`fillTexture:` and `fillTextureColor:`) are Apple SPI and the interpreter refuses
    /// those two steps, so a case built on a fill still carries its Metal answer. Everything else a
    /// program states -- heaps, placed textures, textures over a buffer, views, every blit,
    /// `replaceRegion:` and `getBytes:` -- is public Metal whatever pixel format the case names,
    /// including Apple's two-plane extended-range 550 and 551.
    fn work(&mut self, work: &crate::work::Work) -> Answer<crate::work::Done> {
        crate::metal_work::run(&self.device, &self.queue, work)
    }

    fn announced(&mut self) -> Vec<String> {
        Vec::new()
    }

    fn mesh_pipeline_for_commands(
        &mut self,
        library: Handle,
        functions: &MeshFunctions<'_>,
    ) -> Answer<Handle> {
        let descriptor = self.mesh_descriptor(library, functions)?;
        descriptor.setSupportIndirectCommandBuffers(true);
        self.made_mesh_pipeline(&descriptor)
    }

    fn mesh_command_buffer(&mut self, commands: &[MeshCommand]) -> Answer<Handle> {
        let descriptor = MTLIndirectCommandBufferDescriptor::new();
        descriptor.setCommandTypes(
            MTLIndirectCommandType::DrawMeshThreadgroups | MTLIndirectCommandType::DrawMeshThreads,
        );
        descriptor.setInheritPipelineState(false);
        descriptor.setInheritBuffers(false);
        // SAFETY: one object and one mesh binding, which is what every command of a case binds.
        unsafe {
            descriptor.setMaxObjectBufferBindCount(1);
            descriptor.setMaxMeshBufferBindCount(1);
        }
        // SAFETY: the count is the commands the case encodes, and every command below is inside
        // it.
        let made = unsafe {
            self.device
                .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                    &descriptor,
                    commands.len().max(1),
                    MTLResourceOptions::StorageModeShared,
                )
        }
        .ok_or_else(|| {
            Failure::named(
                "NewIndirectCommandBuffer",
                format!("{} commands returned nil", commands.len()),
            )
        })?;
        for (at, command) in commands.iter().enumerate() {
            // SAFETY: `at` is below the buffer's command count, just set to `commands.len()`.
            let encoded = unsafe { made.indirectRenderCommandAtIndex(at) };
            let state = &self
                .render_pipelines
                .get(command.pipeline.0 as usize)
                .ok_or_else(|| {
                    Failure::named(
                        "NoSuchHandle",
                        format!("mesh pipeline {}", command.pipeline.0),
                    )
                })?
                .state;
            encoded.setRenderPipelineState(state);
            for (buffer, index) in &command.object_buffers {
                let buffer = self.held(*buffer)?;
                // SAFETY: the buffer is at least as long as the function's binding, and the
                // command outlives the call.
                unsafe { encoded.setObjectBuffer_offset_atIndex(buffer, 0, *index as usize) };
            }
            for (buffer, index) in &command.mesh_buffers {
                let buffer = self.held(*buffer)?;
                // SAFETY: as the object buffer's.
                unsafe { encoded.setMeshBuffer_offset_atIndex(buffer, 0, *index as usize) };
            }
            let size = |axes: [u32; 3]| MTLSize {
                width: axes[0] as usize,
                height: axes[1] as usize,
                depth: axes[2] as usize,
            };
            match command.draw {
                CommandDraw::Threadgroups {
                    groups,
                    object,
                    mesh,
                } => encoded
                    .drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                        size(groups),
                        size(object),
                        size(mesh),
                    ),
                CommandDraw::Threads {
                    threads,
                    object,
                    mesh,
                } => encoded.drawMeshThreads_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                    size(threads),
                    size(object),
                    size(mesh),
                ),
            }
        }
        Ok(keep(&mut self.command_buffers_indirect, made))
    }

    fn execute_mesh_commands(
        &mut self,
        targets: &[Handle],
        commands: Handle,
        used: &[Handle],
    ) -> Answer<PassOutcome> {
        let icb = self
            .command_buffers_indirect
            .get(commands.0 as usize)
            .ok_or_else(|| {
                Failure::named("NoSuchHandle", format!("command buffer {}", commands.0))
            })?
            .clone();
        let resident: Vec<&ProtocolObject<dyn MTLResource>> = used
            .iter()
            .map(|handle| self.held(*handle).map(ProtocolObject::from_ref))
            .collect::<Answer<Vec<_>>>()?;
        let (command_buffer, encoder) = self.pass(targets)?;
        let stages = MTLRenderStages::Object | MTLRenderStages::Mesh | MTLRenderStages::Fragment;
        for resource in &resident {
            encoder.useResource_usage_stages(
                resource,
                MTLResourceUsage::Read | MTLResourceUsage::Write,
                stages,
            );
        }
        // SAFETY: the range is the buffer's own command count.
        unsafe {
            encoder.executeCommandsInBuffer_withRange(
                &icb,
                objc2_foundation::NSRange::new(0, icb.size()),
            );
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        // Metal's calls answer nothing, so no step of this side can refuse: one `None` per
        // resident resource, one for the execution and one for the commit, as the runtime side
        // answers them.
        let refused = vec![None; resident.len() + 2];
        let error = command_buffer
            .error()
            .map(|error| Value::unit_variant(format!("code {}", error.code())));
        let status = Value::unit_variant(status_word(command_buffer.status()));
        keep(&mut self.command_buffers, command_buffer);
        Ok(PassOutcome {
            refused,
            status,
            error,
        })
    }

    fn fragment_fill(&mut self, fill: &FragmentFill<'_>) -> Answer<FragmentFilled> {
        let format = MTLPixelFormat(fill.format.as_raw() as usize);
        let (vertex, fragment) = {
            let library = self.libraries.get(fill.library.0 as usize).ok_or_else(|| {
                Failure::named("NoSuchHandle", format!("library {}", fill.library.0))
            })?;
            let function = |name: &str| {
                library
                    .newFunctionWithName(&NSString::from_str(name))
                    .ok_or_else(|| Failure::named("NoSuchFunction", name.to_owned()))
            };
            (function(fill.vertex)?, function(fill.fragment)?)
        };
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(&vertex));
        descriptor.setFragmentFunction(Some(&fragment));
        // SAFETY: attachment 0 is inside the eight every Metal pipeline descriptor holds.
        let attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
        attachment.setPixelFormat(format);
        let state = self
            .device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .map_err(|error| Failure::named("NewRenderPipelineState", error.to_string()))?;
        // SAFETY: the descriptor constructor reads only the format and the size it is given.
        let target_descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                format,
                fill.size[0] as usize,
                fill.size[1] as usize,
                false,
            )
        };
        target_descriptor.setUsage(MTLTextureUsage::RenderTarget);
        target_descriptor.setStorageMode(MTLStorageMode::Private);
        let target = self
            .device
            .newTextureWithDescriptor(&target_descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptor",
                    format!(
                        "{:?} {}x{} render target returned nil",
                        fill.format, fill.size[0], fill.size[1]
                    ),
                )
            })?;
        let bound: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = fill
            .buffers
            .iter()
            .map(|handle| self.held(*handle).map(Retained::from))
            .collect::<Answer<Vec<_>>>()?;
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        // SAFETY: as the pipeline's attachment 0.
        let color = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        color.setTexture(Some(&target));
        color.setLoadAction(MTLLoadAction::DontCare);
        color.setStoreAction(MTLStoreAction::Store);
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        let encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or_else(|| Failure::named("RenderCommandEncoder", "returned nil".to_owned()))?;
        encoder.setRenderPipelineState(&state);
        for (index, buffer) in bound.iter().enumerate() {
            // SAFETY: the buffer is at least as long as the function's binding, and the encoder
            // outlives the call.
            unsafe { encoder.setFragmentBuffer_offset_atIndex(Some(buffer), 0, index) };
        }
        // SAFETY: the pipeline is set, the vertex function reads no buffer, and the count is the
        // case's own small one.
        unsafe {
            encoder.drawPrimitives_vertexStart_vertexCount(
                MTLPrimitiveType::Triangle,
                0,
                fill.vertices as usize,
            );
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let status = Value::unit_variant(status_word(command_buffer.status()));
        keep(&mut self.command_buffers, command_buffer);
        let bytes = self.blit_out(&target, fill.size, fill.row_pitch)?;
        keep(&mut self.textures, target);
        Ok(FragmentFilled { status, bytes })
    }

    fn mesh_pipeline(&mut self, library: Handle, functions: &MeshFunctions<'_>) -> Answer<Handle> {
        let descriptor = self.mesh_descriptor(library, functions)?;
        self.made_mesh_pipeline(&descriptor)
    }

    fn render_target(&mut self, width: u32, height: u32) -> Answer<Handle> {
        // SAFETY: the extent is a case's own small one, and the descriptor is this call's own.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                TARGET_PIXEL_FORMAT,
                width as usize,
                height as usize,
                false,
            )
        };
        descriptor.setUsage(MTLTextureUsage::RenderTarget);
        descriptor.setStorageMode(MTLStorageMode::Shared);
        let texture = self
            .device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptor",
                    format!("{width}x{height} returned nil"),
                )
            })?;
        Ok(keep(&mut self.textures, texture))
    }

    fn mesh_pass(&mut self, targets: &[Handle], steps: &[MeshStep]) -> Answer<PassOutcome> {
        let (command_buffer, encoder) = self.pass(targets)?;
        // Metal's encoder methods answer nothing, so no step of this side can refuse. A refusal
        // of ours then compares against Metal's answer and reads as a refusal.
        let refused = vec![None; steps.len()];
        for step in steps {
            self.encode_step(&encoder, step)?;
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let error = command_buffer
            .error()
            .map(|error| Value::unit_variant(format!("code {}", error.code())));
        let status = Value::unit_variant(status_word(command_buffer.status()));
        keep(&mut self.command_buffers, command_buffer);
        Ok(PassOutcome {
            refused,
            status,
            error,
        })
    }

    fn rate_map(&mut self, request: &RateMapRequest<'_>) -> Answer<Handle> {
        let descriptor = unsafe {
            MTLRasterizationRateMapDescriptor::rasterizationRateMapDescriptorWithScreenSize(
                MTLSize {
                    width: request.screen[0] as usize,
                    height: request.screen[1] as usize,
                    depth: 0,
                },
            )
        };
        if let Some(label) = request.label {
            descriptor.setLabel(Some(&NSString::from_str(label)));
        }
        for (at, layer) in request.layers.iter().enumerate() {
            // `initWithSampleCount:horizontal:vertical:` reads `width` samples from `horizontal`
            // and `height` from `vertical`, and answers nil for an axis of none, which is why
            // `RateMapLayer` holds at least one sample on each.
            let count = MTLSize {
                width: layer.horizontal.len(),
                height: layer.vertical.len(),
                depth: 0,
            };
            // SAFETY: each pointer is to at least as many floats as its axis's count, and Metal
            // copies the samples during the call.
            let made = unsafe {
                MTLRasterizationRateLayerDescriptor::initWithSampleCount_horizontal_vertical(
                    MTLRasterizationRateLayerDescriptor::alloc(),
                    count,
                    std::ptr::NonNull::new(layer.horizontal.as_ptr().cast_mut())
                        .ok_or_else(|| Failure::named("NoSamples", "a layer axis of none"))?,
                    std::ptr::NonNull::new(layer.vertical.as_ptr().cast_mut())
                        .ok_or_else(|| Failure::named("NoSamples", "a layer axis of none"))?,
                )
            };
            // SAFETY: the index is inside the array the descriptor grows on demand.
            unsafe { descriptor.setLayer_atIndex(Some(&made), at) };
        }
        let map = self
            .device
            .newRasterizationRateMapWithDescriptor(&descriptor)
            .ok_or_else(|| Failure::named("NewRasterizationRateMap", "returned nil".to_owned()))?;
        Ok(keep(&mut self.rate_maps, map))
    }

    fn rate_map_facts(&mut self, map: Handle, layers: &[u64]) -> Answer<RateMapFacts> {
        let map = self.held_rate_map(map)?;
        let size = |size: MTLSize| [size.width as u64, size.height as u64, size.depth as u64];
        let sized = map.parameterBufferSizeAndAlign();
        Ok(RateMapFacts {
            label: map.label().map(|label| label.to_string()),
            screen_size: size(map.screenSize()),
            physical_granularity: size(map.physicalGranularity()),
            layer_count: map.layerCount() as u64,
            // SAFETY: `physicalSizeForLayer:` is not bounds-checked, and the case asks only about
            // layer indices Metal's own answers are measured at.
            physical_sizes: layers
                .iter()
                .map(|layer| size(unsafe { map.physicalSizeForLayer(*layer as usize) }))
                .collect(),
            parameter_size: sized.size as u64,
            parameter_align: sized.align as u64,
        })
    }

    fn copy_rate_map_parameters(&mut self, map: Handle, into: Handle, offset: u64) -> Answer<()> {
        let buffer: Retained<ProtocolObject<dyn MTLBuffer>> = Retained::from(self.held(into)?);
        let map = self.held_rate_map(map)?;
        if u64::try_from(buffer.length()).unwrap_or(u64::MAX)
            < offset + map.parameterBufferSizeAndAlign().size as u64
        {
            // Metal writes past the buffer's length here, so the case never asks it to: a short
            // buffer is refused before the copy rather than measured.
            return Err(Failure::named(
                "PastTheBuffer",
                format!(
                    "{} parameter bytes at {offset} of a buffer of {}",
                    map.parameterBufferSizeAndAlign().size,
                    buffer.length()
                ),
            ));
        }
        // SAFETY: the buffer is of shared storage, is long enough for the parameter data at the
        // offset, is kept alive by `self.buffers`, and no command uses it.
        unsafe { map.copyParameterDataToBuffer_offset(&buffer, offset as usize) };
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
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: size[0] as usize,
                height: size[1] as usize,
                depth: 1,
            },
        };
        // SAFETY: `bytes` is `size[1]` rows of `row_pitch` bytes, checked just above, which is the
        // region the call reads, and no command buffer of this case is in flight.
        unsafe {
            texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                std::ptr::NonNull::from(bytes.as_mut_slice()).cast(),
                row_pitch as usize,
                region,
                0,
            );
        }
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
        let function = |name: &str| -> Answer<Retained<ProtocolObject<dyn MTLFunction>>> {
            library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| Failure::named("NewFunctionWithName", name.to_owned()))
        };
        let descriptor = MTLRenderPipelineDescriptor::new();
        let (vertex, fragment) = (function(vertex)?, function(fragment)?);
        descriptor.setVertexFunction(Some(&vertex));
        descriptor.setFragmentFunction(Some(&fragment));
        let attachments = descriptor.colorAttachments();
        // SAFETY: attachment 0 is inside the fixed-size array every descriptor holds.
        let first = unsafe { attachments.objectAtIndexedSubscript(0) };
        first.setPixelFormat(MTLPixelFormat(attachment.as_raw() as usize));
        let mut reflection = None;
        let state = self
            .device
            .newRenderPipelineStateWithDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::BindingInfo,
                Some(&mut reflection),
            )
            .map_err(|error| {
                Failure::named(
                    "NewRenderPipelineState",
                    error.localizedDescription().to_string(),
                )
            })?;
        let reflection = reflection.ok_or_else(|| {
            Failure::named(
                "NoReflection",
                "the creation answered no reflection".to_owned(),
            )
        })?;
        Ok(keep(
            &mut self.render_pipelines,
            RenderPipeline {
                state,
                reflection: Some(reflection),
            },
        ))
    }

    fn texture_accesses(
        &mut self,
        pipeline: Handle,
        stage: RenderStages,
        indices: &[u64],
    ) -> Answer<Vec<Option<BindingAccess>>> {
        let reflection = self
            .render_pipelines
            .get(pipeline.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", pipeline.0)))?
            .reflection
            .as_ref()
            .ok_or_else(|| {
                // Not a Metal gap and not a venue's: the case asked a state whose creation never
                // asked Metal for a reflection, and Metal cannot be asked afterwards.
                Failure::named(
                    "NoReflection",
                    format!("pipeline {} was not created with one", pipeline.0),
                )
            })?;
        // One stage, because that is how Metal states it: a binding list a stage. A union has no
        // Metal answer, only a composition rule of ours.
        let bindings = match stage.as_raw() {
            raw if raw == RenderStages::VERTEX.as_raw() => reflection.vertexBindings(),
            raw if raw == RenderStages::FRAGMENT.as_raw() => reflection.fragmentBindings(),
            raw if raw == RenderStages::TILE.as_raw() => reflection.tileBindings(),
            raw if raw == RenderStages::OBJECT.as_raw() => reflection.objectBindings(),
            raw if raw == RenderStages::MESH.as_raw() => reflection.meshBindings(),
            raw => {
                return Err(Failure::unavailable(&format!(
                    "texture_accesses of MTLRenderStages({raw:#x}), which Metal's reflection \
                     states no binding list for"
                )))
            }
        };
        // Each texture binding covers `arrayLength` slots from its own index, and an argument the
        // function never references is listed with `isUsed` false.
        let mut covered: Vec<(u64, u64, BindingAccess, bool)> = Vec::new();
        for binding in &bindings {
            if binding.r#type() != MTLBindingType::Texture {
                continue;
            }
            let access = match binding.access() {
                MTLBindingAccess::ReadOnly => BindingAccess::ReadOnly,
                MTLBindingAccess::ReadWrite => BindingAccess::ReadWrite,
                MTLBindingAccess::WriteOnly => BindingAccess::WriteOnly,
                other => {
                    return Err(Failure::named(
                        "UnknownBindingAccess",
                        format!("MTLBindingAccess({})", other.0),
                    ))
                }
            };
            // SAFETY: `MTLTextureBinding` extends `MTLBinding` with `arrayLength`, and the
            // binding answered `MTLBindingTypeTexture` just above, so it conforms to it. objc2
            // offers no safe cast between two protocol objects, so the selector is sent directly.
            let length: usize = unsafe { objc2::msg_send![&*binding, arrayLength] };
            covered.push((
                binding.index() as u64,
                length as u64,
                access,
                binding.isUsed(),
            ));
        }
        Ok(indices
            .iter()
            .map(|index| {
                covered
                    .iter()
                    .find(|(first, length, _, _)| *index >= *first && *index < *first + *length)
                    .and_then(|(_, _, access, used)| used.then_some(*access))
            })
            .collect())
    }

    fn tile_pipeline(&mut self, library: Handle, spec: &TilePipelineSpec<'_>) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(spec.entry))
            .ok_or_else(|| Failure::named("NoSuchFunction", spec.entry.to_owned()))?;
        let descriptor = MTLTileRenderPipelineDescriptor::new();
        // SAFETY: the function is of this library, and the descriptor is this call's own.
        unsafe { descriptor.setTileFunction(&function) };
        for (at, format) in spec.attachments.iter().enumerate() {
            // SAFETY: `at` is below the eight attachments the descriptor's array holds.
            unsafe {
                descriptor
                    .colorAttachments()
                    .objectAtIndexedSubscript(at)
                    .setPixelFormat(MTLPixelFormat(format.as_raw() as usize));
            }
        }
        if let Some(limit) = spec.max_total_threads_per_threadgroup {
            descriptor.setMaxTotalThreadsPerThreadgroup(limit as usize);
        }
        descriptor.setThreadgroupSizeMatchesTileSize(spec.threadgroup_size_matches_tile_size);
        let state = self
            .device
            .newRenderPipelineStateWithTileDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|error| Failure::named("NewTileRenderPipelineState", error.to_string()))?;
        Ok(keep(
            &mut self.render_pipelines,
            RenderPipeline::unreflected(state),
        ))
    }

    fn tile_facts(&mut self, state: Handle) -> Answer<TileFacts> {
        let state = &self
            .render_pipelines
            .get(state.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("tile pipeline {}", state.0)))?
            .state;
        Ok(TileFacts {
            max_total_threads_per_threadgroup: state.maxTotalThreadsPerThreadgroup() as u64,
            threadgroup_size_matches_tile_size: state.threadgroupSizeMatchesTileSize(),
            imageblock_sample_length: Some(state.imageblockSampleLength() as u64),
        })
    }

    fn tile_pass(
        &mut self,
        targets: &[Handle],
        tile: [u32; 2],
        steps: &[TileStep],
    ) -> Answer<PassOutcome> {
        let (command_buffer, encoder) = self.loading_pass(targets, tile)?;
        // Metal's encoder methods answer nothing, so no step of this side can refuse.
        let refused = vec![None; steps.len()];
        for step in steps {
            self.encode_tile_step(&encoder, step)?;
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let error = command_buffer
            .error()
            .map(|error| Value::unit_variant(format!("code {}", error.code())));
        let status = Value::unit_variant(status_word(command_buffer.status()));
        keep(&mut self.command_buffers, command_buffer);
        Ok(PassOutcome {
            refused,
            status,
            error,
        })
    }

    fn target(&mut self, spec: &TargetSpec) -> Answer<Handle> {
        // SAFETY: the descriptor constructor reads only the format and the size it is given.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat(spec.format.as_raw() as usize),
                spec.size[0] as usize,
                spec.size[1] as usize,
                false,
            )
        };
        let mut usage = MTLTextureUsage::RenderTarget;
        if spec.shader_read {
            usage |= MTLTextureUsage::ShaderRead;
        }
        if spec.shader_write {
            usage |= MTLTextureUsage::ShaderWrite;
        }
        descriptor.setUsage(usage);
        descriptor.setStorageMode(match spec.shared {
            true => MTLStorageMode::Shared,
            false => MTLStorageMode::Private,
        });
        if spec.samples > 1 {
            descriptor.setTextureType(MTLTextureType::Type2DMultisample);
            // SAFETY: the count is 2 or 4, which the case's own guard keeps it to, and Metal takes
            // both on every device; the descriptor is this call's own.
            unsafe { descriptor.setSampleCount(spec.samples as usize) };
        }
        let texture = self
            .device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptor",
                    format!(
                        "a {:?} {}x{} target at {} samples returned nil",
                        spec.format, spec.size[0], spec.size[1], spec.samples
                    ),
                )
            })?;
        Ok(keep(&mut self.textures, texture))
    }

    fn render_pipeline(
        &mut self,
        library: Handle,
        spec: &RenderPipelineSpec<'_>,
    ) -> Answer<Handle> {
        let function = |stage: &StageFunction<'_>| {
            let named = stage.library.unwrap_or(library);
            let held = self
                .libraries
                .get(named.0 as usize)
                .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", named.0)))?;
            held.newFunctionWithName(&NSString::from_str(stage.entry))
                .ok_or_else(|| Failure::named("NoSuchFunction", stage.entry.to_owned()))
        };
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(&*function(&spec.vertex)?));
        if let Some(fragment) = &spec.fragment {
            descriptor.setFragmentFunction(Some(&*function(fragment)?));
        }
        descriptor.setRasterSampleCount(spec.samples as usize);
        descriptor.setRasterizationEnabled(spec.rasterization);
        if let Some(layout) = &spec.vertex_layout {
            descriptor.setVertexDescriptor(Some(&vertex_descriptor(layout)));
        }
        if let Some(format) = spec.depth_format {
            descriptor.setDepthAttachmentPixelFormat(MTLPixelFormat(format.as_raw() as usize));
        }
        if let Some(format) = spec.stencil_format {
            descriptor.setStencilAttachmentPixelFormat(MTLPixelFormat(format.as_raw() as usize));
        }
        for (at, color) in spec.colors.iter().enumerate() {
            let Some(color) = color else { continue };
            // SAFETY: `at` is below the eight color attachments every Metal pipeline holds, which
            // the case's own guard keeps it to.
            let attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(at) };
            attachment.setPixelFormat(MTLPixelFormat(color.format.as_raw() as usize));
            attachment.setWriteMask(MTLColorWriteMask(color.write_mask.as_raw() as usize));
            if let Some(blend) = &color.blend {
                let factor = |factor: metal2vulkan::metal::BlendFactor| {
                    MTLBlendFactor(factor.as_raw() as usize)
                };
                let operation = |operation: metal2vulkan::metal::BlendOperation| {
                    MTLBlendOperation(operation.as_raw() as usize)
                };
                attachment.setBlendingEnabled(true);
                attachment.setSourceRGBBlendFactor(factor(blend.source_rgb));
                attachment.setDestinationRGBBlendFactor(factor(blend.destination_rgb));
                attachment.setRgbBlendOperation(operation(blend.rgb));
                attachment.setSourceAlphaBlendFactor(factor(blend.source_alpha));
                attachment.setDestinationAlphaBlendFactor(factor(blend.destination_alpha));
                attachment.setAlphaBlendOperation(operation(blend.alpha));
            }
        }
        let state = self
            .device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .map_err(|error| Failure::named("NewRenderPipelineState", error.to_string()))?;
        Ok(keep(
            &mut self.render_pipelines,
            RenderPipeline::unreflected(state),
        ))
    }

    fn depth_stencil_state(&mut self, spec: &DepthStencilDescriptor) -> Answer<Handle> {
        let descriptor = MTLDepthStencilDescriptor::new();
        descriptor.setDepthCompareFunction(compare_function(spec.depth_compare_function));
        descriptor.setDepthWriteEnabled(spec.depth_write_enabled);
        descriptor.setFrontFaceStencil(Some(&stencil_descriptor(&spec.front_face_stencil)));
        descriptor.setBackFaceStencil(Some(&stencil_descriptor(&spec.back_face_stencil)));
        let state = self
            .device
            .newDepthStencilStateWithDescriptor(&descriptor)
            .ok_or_else(|| Failure::named("NewDepthStencilState", "returned nil".to_owned()))?;
        Ok(keep(&mut self.depth_stencil_states, state))
    }

    fn render(&mut self, work: &[RenderWork]) -> Answer<PassOutcome> {
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        // Metal's encoder methods answer nothing, so no piece of this side's program can refuse. A
        // refusal of ours then compares against Metal's answer and reads as a refusal.
        let refused = vec![None; work.len()];
        for piece in work {
            match piece {
                RenderWork::Pass(pass) => self.encode_pass(&command_buffer, pass)?,
                RenderWork::ReadTexture {
                    texture,
                    into,
                    offset,
                    row_pitch,
                    size,
                } => self.encode_read_texture(
                    &command_buffer,
                    *texture,
                    *into,
                    *offset,
                    *row_pitch,
                    *size,
                )?,
                RenderWork::Dispatch(dispatch) => {
                    self.encode_dispatch(&command_buffer, dispatch)?;
                }
            }
        }
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let error = command_buffer
            .error()
            .map(|error| Value::unit_variant(format!("code {}", error.code())));
        let status = Value::unit_variant(status_word(command_buffer.status()));
        keep(&mut self.command_buffers, command_buffer);
        Ok(PassOutcome {
            refused,
            status,
            error,
        })
    }

    fn heap_texture_size_and_align(&mut self, query: &HeapTextureQuery) -> Answer<SizeAndAlign> {
        let descriptor = queried_descriptor(query);
        let answered = self
            .device
            .heapTextureSizeAndAlignWithDescriptor(&descriptor);
        Ok(SizeAndAlign {
            size: answered.size as u64,
            align: answered.align as u64,
        })
    }

    fn queue_slots(&mut self, max: Option<u32>, cap: u32) -> Answer<QueueSlots> {
        let queue = match max {
            None => self.device.newCommandQueue(),
            Some(max) => self
                .device
                .newCommandQueueWithMaxCommandBufferCount(max as usize),
        }
        .ok_or_else(|| Failure::named("NewCommandQueue", format!("{max:?} returned nil")))?;
        let other = self
            .device
            .newCommandQueue()
            .ok_or_else(|| Failure::named("NewCommandQueue", "the second queue returned nil"))?;
        crate::queue_slots::probe(
            cap,
            move || {
                // Each make gets an autorelease pool of its own, and that is load-bearing rather
                // than tidy. `commandBuffer` answers an autoreleased object; the `Retained` is a
                // second reference, so without a pool that drains, the probe's own release frees
                // nothing, the slot never comes back, and the blocked make it is meant to release
                // stays blocked (measured: the whole probe timed out). A worker thread has no pool
                // of its own, so one is made here.
                objc2::rc::autoreleasepool(|_| {
                    queue
                        .commandBuffer()
                        .map(Made)
                        .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))
                })
            },
            || objc2::rc::autoreleasepool(|_| other.commandBuffer().is_some()),
        )
    }

    fn shared_event_rises(&mut self, label: &str) -> Answer<crate::shared_events::EventRise> {
        use crate::shared_events::{EventRise, WAIT_MS};
        let event = self
            .device
            .newSharedEvent()
            .ok_or_else(|| Failure::named("NewSharedEvent", "returned nil".to_owned()))?;
        event.setLabel(Some(&NSString::from_str(label)));
        let read_label = event.label().map(|label| label.to_string());
        let initial = event.signaledValue();
        // The host's sets: a lower value leaves the event where it was.
        event.setSignaledValue(5);
        event.setSignaledValue(3);
        event.setSignaledValue(5);
        let after_host_sets = event.signaledValue();
        let as_event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event);
        let signal = |value: u64| -> Answer<u64> {
            // One autorelease pool per signal: the command buffer this answers is autoreleased,
            // and a loop that makes several without a pool that drains keeps every one of them
            // alive.
            objc2::rc::autoreleasepool(|_| {
                let command_buffer = self
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
                command_buffer.encodeSignalEvent_value(as_event, value);
                command_buffer.commit();
                command_buffer.waitUntilCompleted();
                Ok(event.signaledValue())
            })
        };
        let after_gpu_signals = [signal(3)?, signal(5)?];
        let waits = [
            event.waitUntilSignaledValue_timeoutMS(3, WAIT_MS),
            event.waitUntilSignaledValue_timeoutMS(5, WAIT_MS),
            event.waitUntilSignaledValue_timeoutMS(6, WAIT_MS),
        ];
        let after_max_signal = signal(u64::MAX)?;
        let waited_for_max = event.waitUntilSignaledValue_timeoutMS(u64::MAX, WAIT_MS);
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
        // One pool around the whole probe: `commandBuffer`, `newEvent` and `newCommandQueue` each
        // answer an autoreleased object, and a probe's thread has no pool of its own. Without this
        // the command buffers would live in the never-drained top-level pool and hold their
        // queues' slots for good.
        objc2::rc::autoreleasepool(|_| {
            let queue = || {
                self.device
                    .newCommandQueue()
                    .ok_or_else(|| Failure::named("NewCommandQueue", "returned nil"))
            };
            let (waiting_queue, signalling_queue) = (queue()?, queue()?);
            let event = self
                .device
                .newEvent()
                .ok_or_else(|| Failure::named("NewEvent", "returned nil"))?;
            let scratch = self
                .device
                .newBufferWithLength_options(
                    SCRATCH as usize,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| Failure::named("NewBuffer", "returned nil"))?;
            let fill = |command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
                        offset: u64,
                        length: u64,
                        byte: u8|
             -> Result<(), Failure> {
                let encoder = command_buffer
                    .blitCommandEncoder()
                    .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil"))?;
                encoder.fillBuffer_range_value(
                    &scratch,
                    objc2_foundation::NSRange::new(offset as usize, length as usize),
                    byte,
                );
                encoder.endEncoding();
                Ok(())
            };
            let waiting = waiting_queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            waiting.encodeWaitForEvent_value(&event, 1);
            fill(&waiting, 0, HALF, WAITING_BYTE)?;
            let signalling = signalling_queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            fill(&signalling, HALF, HALF, SIGNALLING_BYTE)?;
            signalling.encodeSignalEvent_value(&event, 1);
            // Bounded completion, rather than `waitUntilCompleted`, which Metal has no bounded
            // form of: the handler is added before the commit and the probe waits on its channel.
            let waiting_done = completion_channel(&waiting);
            let signalling_done = completion_channel(&signalling);
            // The waiting one first, which is the whole point: its signal is not committed yet.
            waiting.commit();
            let waiting_after_its_commit = Status::read(status_word(waiting.status()));
            signalling.commit();
            let signalling_at_the_end = settled(&signalling_done, &signalling);
            let waiting_at_the_end = settled(&waiting_done, &waiting);
            // SAFETY: `SCRATCH` bytes of the shared buffer, whose writers have both completed.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    scratch.contents().as_ptr().cast::<u8>(),
                    SCRATCH as usize,
                )
            }
            .to_vec();
            Ok(CrossQueueWait {
                waiting_after_its_commit,
                waiting_at_the_end,
                signalling_at_the_end,
                bytes,
            })
        })
    }

    fn event_highest_signal(&mut self, values: [u64; 2]) -> Answer<u64> {
        use crate::event_orders::SCRATCH;
        objc2::rc::autoreleasepool(|_| {
            let queue = || {
                self.device
                    .newCommandQueue()
                    .ok_or_else(|| Failure::named("NewCommandQueue", "returned nil"))
            };
            let queues = [queue()?, queue()?];
            let event = self
                .device
                .newSharedEvent()
                .ok_or_else(|| Failure::named("NewSharedEvent", "returned nil"))?;
            let as_event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event);
            let scratch = self
                .device
                .newBufferWithLength_options(
                    SCRATCH as usize,
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| Failure::named("NewBuffer", "returned nil"))?;
            let mut done = Vec::new();
            for (queue, &value) in queues.iter().zip(&values) {
                let command_buffer = queue
                    .commandBuffer()
                    .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
                let encoder = command_buffer
                    .blitCommandEncoder()
                    .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil"))?;
                encoder.fillBuffer_range_value(
                    &scratch,
                    objc2_foundation::NSRange::new(0, SCRATCH as usize),
                    value as u8,
                );
                encoder.endEncoding();
                command_buffer.encodeSignalEvent_value(as_event, value);
                let channel = completion_channel(&command_buffer);
                command_buffer.commit();
                done.push(channel);
            }
            for channel in &done {
                if channel
                    .done
                    .recv_timeout(crate::event_orders::COMPLETION)
                    .is_err()
                {
                    return Err(Failure::named(
                        "NeverCompleted",
                        "a signalling command buffer",
                    ));
                }
            }
            Ok(event.signaledValue())
        })
    }

    fn held_command_buffer(&mut self) -> Answer<crate::event_orders::HeldCommandBuffer> {
        use crate::event_orders::{
            HeldCommandBuffer, Status, COMPLETED, COMPLETION, HELD_FOR, SCHEDULED, SCRATCH,
        };
        objc2::rc::autoreleasepool(|_| {
            let queue = || {
                self.device
                    .newCommandQueue()
                    .ok_or_else(|| Failure::named("NewCommandQueue", "returned nil"))
            };
            let (held_queue, other_queue) = (queue()?, queue()?);
            let event = self
                .device
                .newSharedEvent()
                .ok_or_else(|| Failure::named("NewSharedEvent", "returned nil"))?;
            let as_event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event);
            let scratch = self
                .device
                .newBufferWithLength_options(
                    SCRATCH as usize,
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| Failure::named("NewBuffer", "returned nil"))?;
            let fill = |command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
                        byte: u8|
             -> Result<(), Failure> {
                let encoder = command_buffer
                    .blitCommandEncoder()
                    .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil"))?;
                encoder.fillBuffer_range_value(
                    &scratch,
                    objc2_foundation::NSRange::new(0, SCRATCH as usize),
                    byte,
                );
                encoder.endEncoding();
                Ok(())
            };
            let mut log = HandlerLog::new();
            let held = held_queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            held.encodeWaitForEvent_value(as_event, 1);
            fill(&held, 1)?;
            // SAFETY: each block is a valid heap block the log keeps alive for as long as this
            // probe runs, which outlives every call Metal makes of it.
            log.add(SCHEDULED, |block| unsafe {
                held.addScheduledHandler(block)
            });
            log.add(COMPLETED, |block| unsafe {
                held.addCompletedHandler(block)
            });
            held.commit();
            let behind = held_queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            fill(&behind, 2)?;
            let behind_done = completion_channel(&behind);
            behind.commit();
            let other = other_queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            fill(&other, 3)?;
            let other_done = completion_channel(&other);
            other.commit();
            let other_queue_status = settled(&other_done, &other);
            // The window the negative facts are false in: a submission that was going to happen
            // has had its chance by now.
            std::thread::sleep(HELD_FOR);
            let held_while_waiting = Status::read(status_word(held.status()));
            let behind_while_waiting = Status::read(status_word(behind.status()));
            let handlers_while_waiting = log.names();
            event.setSignaledValue(1);
            // The held command buffer's own completed handler is the bound: it is the last of its
            // handlers to run, so once it has, the log is whole and the status is settled.
            log.wait_for(COMPLETED, COMPLETION);
            let handlers_at_the_end = log.names();
            let held_at_the_end = if handlers_at_the_end.iter().any(|name| name == COMPLETED) {
                Status::read(status_word(held.status()))
            } else {
                Status::NeverSettled
            };
            let behind_at_the_end = settled(&behind_done, &behind);
            Ok(HeldCommandBuffer {
                other_queue: other_queue_status,
                held_while_waiting,
                behind_while_waiting,
                handlers_while_waiting,
                handlers_at_the_end,
                held_at_the_end,
                behind_at_the_end,
            })
        })
    }

    fn signal_before_a_wait(&mut self) -> Answer<crate::event_orders::SignalBeforeWait> {
        use crate::event_orders::{Status, HELD_FOR, SCRATCH, SIGNAL_WAIT};
        objc2::rc::autoreleasepool(|_| {
            let queue = self
                .device
                .newCommandQueue()
                .ok_or_else(|| Failure::named("NewCommandQueue", "returned nil"))?;
            let event = || {
                self.device
                    .newSharedEvent()
                    .ok_or_else(|| Failure::named("NewSharedEvent", "returned nil"))
            };
            let (before, gate, after) = (event()?, event()?, event()?);
            let scratch = self
                .device
                .newBufferWithLength_options(
                    SCRATCH as usize,
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| Failure::named("NewBuffer", "returned nil"))?;
            let command_buffer = queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            let fill = |byte: u8| -> Result<(), Failure> {
                let encoder = command_buffer
                    .blitCommandEncoder()
                    .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil"))?;
                encoder.fillBuffer_range_value(
                    &scratch,
                    objc2_foundation::NSRange::new(0, SCRATCH as usize),
                    byte,
                );
                encoder.endEncoding();
                Ok(())
            };
            let before_event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*before);
            let gate_event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*gate);
            let after_event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*after);
            fill(1)?;
            command_buffer.encodeSignalEvent_value(before_event, 1);
            command_buffer.encodeWaitForEvent_value(gate_event, 1);
            fill(2)?;
            command_buffer.encodeSignalEvent_value(after_event, 1);
            let done = completion_channel(&command_buffer);
            command_buffer.commit();
            let before_landed =
                before.waitUntilSignaledValue_timeoutMS(1, SIGNAL_WAIT.as_millis() as u64);
            // The window the two negative facts are false in: the signal behind the wait has not
            // landed, and the command buffer is no further along than committed.
            std::thread::sleep(HELD_FOR);
            let after_while_pending = after.signaledValue();
            let status_while_pending = Status::read(status_word(command_buffer.status()));
            gate.setSignaledValue(1);
            let status_at_the_end = settled(&done, &command_buffer);
            Ok(crate::event_orders::SignalBeforeWait {
                before_landed,
                after_while_pending,
                status_while_pending,
                after_at_the_end: after.signaledValue(),
                status_at_the_end,
            })
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
        objc2::rc::autoreleasepool(|_| {
            let queue = self
                .device
                .newCommandQueue()
                .ok_or_else(|| Failure::named("NewCommandQueue", "returned nil"))?;
            let scratch = self
                .device
                .newBufferWithLength_options(
                    SCRATCH as usize,
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| Failure::named("NewBuffer", "returned nil"))?;
            let command_buffer = queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            let encoder = command_buffer
                .blitCommandEncoder()
                .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil"))?;
            encoder.fillBuffer_range_value(
                &scratch,
                objc2_foundation::NSRange::new(0, SCRATCH as usize),
                1,
            );
            encoder.endEncoding();
            let order: Arc<Mutex<Vec<String>>> = Arc::default();
            let (entered, entering) = mpsc::channel();
            let (release, released) = mpsc::channel::<()>();
            let held = {
                let order = Arc::clone(&order);
                block2::RcBlock::new(
                    move |own: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                        if let Ok(mut order) = order.lock() {
                            order.push(SCHEDULED.to_owned());
                        }
                        // SAFETY: Metal hands the handler its own command buffer, which is alive
                        // for the call.
                        let status = unsafe { own.as_ref() }.status();
                        let _ = entered.send(status_word(status).to_owned());
                        // Held, so the probe can read the status and the wait while this handler
                        // is inside itself. Bounded, so the handler leaves Metal's handler thread
                        // whatever the probe does.
                        let _ = released.recv_timeout(HOLD_LIMIT);
                    },
                )
            };
            let (done, completing) = mpsc::channel();
            let finished = {
                let order = Arc::clone(&order);
                block2::RcBlock::new(
                    move |own: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                        if let Ok(mut order) = order.lock() {
                            order.push(COMPLETED.to_owned());
                        }
                        // SAFETY: as above.
                        let status = unsafe { own.as_ref() }.status();
                        let _ = done.send(status_word(status).to_owned());
                    },
                )
            };
            // SAFETY: both blocks are valid heap blocks held in `held` and `finished` for the rest
            // of this probe, which outlives every call Metal makes of them: the probe does not
            // return before the completed handler ran or its bound ran out.
            unsafe {
                command_buffer.addScheduledHandler(block2::RcBlock::as_ptr(&held));
                command_buffer.addCompletedHandler(block2::RcBlock::as_ptr(&finished));
            }
            command_buffer.commit();
            let inside_the_handler = match entering.recv_timeout(ARRIVAL) {
                Ok(word) => Status::read(word),
                Err(_) => Status::NeverSettled,
            };
            let while_the_handler_ran = Status::read(status_word(command_buffer.status()));
            let returned = Arc::new(AtomicBool::new(false));
            let (answered, answer) = mpsc::channel();
            {
                let (waited, returned) = (Waited(command_buffer.clone()), Arc::clone(&returned));
                // The wait runs on a thread of its own because the fact under test is that it does
                // **not** return; the thread is left to finish rather than joined, so a wait that
                // never came back leaves that thread and not the suite.
                std::thread::spawn(move || {
                    waited.wait_until_scheduled();
                    returned.store(true, Ordering::SeqCst);
                    let _ = answered.send(());
                });
            }
            // The window the negative fact is false in.
            std::thread::sleep(WINDOW);
            let wait_returned_while_the_handler_ran = returned.load(Ordering::SeqCst);
            let _ = release.send(());
            // Metal's `waitUntilScheduled` answers nothing, so a return is the whole answer.
            let wait_answered = match answer.recv_timeout(ARRIVAL) {
                Ok(()) => Returned::Ok,
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
        })
    }

    /// A texture over one plane of a real IOSurface laid out as the case says its stand-in buffer
    /// is (`newTextureWithDescriptor:iosurface:plane:`), or over the buffer itself.
    ///
    /// The surface is the only way Metal makes a texture of a two-plane YCbCr format at all: the
    /// descriptor alone answers nil for every one of the 69, which is what
    /// `ycbcr-two-plane-format-descriptors` records. So this is where the buffer the case states
    /// becomes the thing Metal takes — same plane offsets, same row pitches, same bytes from byte
    /// 0 — and the surface is kept beside the texture so it outlives it.
    fn plane_texture(
        &mut self,
        buffer: Handle,
        over: &crate::case::PlaneTexture,
    ) -> Answer<Handle> {
        let bytes = {
            let held = self.held(buffer)?;
            let length = held.length();
            // SAFETY: the buffer's own shared storage, whole, and no command uses it yet.
            unsafe {
                std::slice::from_raw_parts(held.contents().as_ptr().cast::<u8>(), length).to_vec()
            }
        };
        let [width, height] = over.size;
        // The descriptor and the texture are autoreleased, and a case may make several over a 24
        // MiB surface, so they are made inside a pool rather than left to a thread that drains
        // none.
        let made = objc2::rc::autoreleasepool(|_| {
            let descriptor = MTLTextureDescriptor::new();
            descriptor.setTextureType(MTLTextureType::Type2D);
            descriptor.setPixelFormat(MTLPixelFormat(over.format as usize));
            // SAFETY: the case's own extent; the setters read nothing but the value given.
            unsafe {
                descriptor.setWidth(width as usize);
                descriptor.setHeight(height as usize);
            }
            descriptor.setStorageMode(MTLStorageMode::Shared);
            match over.ostype {
                Some(ostype) => {
                    // The usage the runtime side's plane texture takes: a surface's plane is read.
                    descriptor.setUsage(MTLTextureUsage::ShaderRead);
                    let format = PixelFormat::try_from(over.format).map_err(|problem| {
                        Failure::named("NoSuchPixelFormat", format!("{}: {problem}", over.format))
                    })?;
                    let planes = crate::metal_surface::planes_of(format, over.size, over.planes)
                        .map_err(|why| Failure::named("NoSuchPlaneLayout", why))?;
                    crate::metal_surface::planes_hold_their_texels(&planes, bytes.len())
                        .map_err(|why| Failure::named("PlanesPastTheBuffer", why))?;
                    let surface = crate::metal_surface::Surface::two_plane(
                        ostype,
                        over.size,
                        &planes,
                        bytes.len(),
                    )
                    .map_err(|why| Failure::named("NoSurface", why))?;
                    surface
                        .planes_are_as_asked(&planes)
                        .map_err(|why| Failure::named("SurfaceLaidOtherwise", why))?;
                    surface
                        .fill(&bytes)
                        .map_err(|why| Failure::named("SurfaceNotFilled", why))?;
                    let device: &ProtocolObject<dyn MTLDevice> = &self.device;
                    let descriptor: &MTLTextureDescriptor = &descriptor;
                    let plane = over.plane as usize;
                    // SAFETY: `newTextureWithDescriptor:iosurface:plane:` of a live surface whose
                    // planes it states, at one of them; the surface is kept below, so it outlives
                    // the texture that reads it.
                    let texture: Option<Retained<ProtocolObject<dyn MTLTexture>>> = unsafe {
                        objc2::msg_send![device, newTextureWithDescriptor: descriptor, iosurface: surface.as_ptr(), plane: plane]
                    };
                    let texture = texture.ok_or_else(|| {
                        Failure::named(
                            "NewTextureWithIOSurfacePlane",
                            format!(
                                "format {} over plane {} of a {width}x{height} '{}' surface \
                                 returned nil",
                                over.format,
                                over.plane,
                                crate::metal_surface::four_cc(ostype)
                            ),
                        )
                    })?;
                    Ok((texture, Some(surface)))
                }
                None => {
                    // The usage the runtime side's texture over a buffer takes.
                    descriptor.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
                    let (offset, row) = over.planes[0];
                    let held = self.held(buffer)?;
                    let texture = held
                        .newTextureWithDescriptor_offset_bytesPerRow(
                            &descriptor,
                            offset as usize,
                            row as usize,
                        )
                        .ok_or_else(|| {
                            Failure::named(
                                "NewTextureWithDescriptorOffsetBytesPerRow",
                                format!(
                                    "format {} {width}x{height} at {offset} of {row} a row \
                                     returned nil",
                                    over.format
                                ),
                            )
                        })?;
                    Ok((texture, None))
                }
            }
        })?;
        let (texture, surface) = made;
        if let Some(surface) = surface {
            self.surfaces.push(surface);
        }
        Ok(keep(&mut self.textures, texture))
    }

    fn makes_texture(&mut self, query: &HeapTextureQuery) -> Answer<bool> {
        let descriptor = queried_descriptor(query);
        // The texture, where Metal makes one, is released as this statement ends: the case asks
        // this only of descriptors Metal makes none of, and nothing here keeps a handle, so no
        // sweep of the family holds a texture it does not read.
        Ok(self.device.newTextureWithDescriptor(&descriptor).is_some())
    }

    fn sparse_tile_size(&mut self, query: &SparseTileQuery) -> Answer<TileSize> {
        let texture_type = MTLTextureType(query.texture_type.as_raw() as usize);
        let format = MTLPixelFormat(query.format.as_raw() as usize);
        let samples = query.samples as usize;
        // SAFETY: both getters read the three scalars they are given and answer an `MTLSize`.
        // Neither allocates, compiles or reaches the GPU, and neither takes an object.
        let size = unsafe {
            match query.page {
                None => self
                    .device
                    .sparseTileSizeWithTextureType_pixelFormat_sampleCount(
                        texture_type,
                        format,
                        samples,
                    ),
                Some(page) => self
                    .device
                    .sparseTileSizeWithTextureType_pixelFormat_sampleCount_sparsePageSize(
                        texture_type,
                        format,
                        samples,
                        metal_page(page),
                    ),
            }
        };
        Ok(TileSize {
            width: size.width as u64,
            height: size.height as u64,
            depth: size.depth as u64,
        })
    }

    fn sparse_tile_size_in_bytes(&mut self, page: QueryPage) -> Answer<u64> {
        Ok(match page {
            None => self.device.sparseTileSizeInBytes() as u64,
            Some(page) => {
                self.device
                    .sparseTileSizeInBytesForSparsePageSize(metal_page(page)) as u64
            }
        })
    }

    fn sparse_heap(&mut self, spec: &SparseHeapSpec) -> Answer<Handle> {
        let descriptor = MTLHeapDescriptor::new();
        descriptor.setType(MTLHeapType::Sparse);
        descriptor.setStorageMode(MTLStorageMode::Private);
        descriptor.setSparsePageSize(metal_page(spec.page));
        descriptor.setSize(spec.size as usize);
        let heap = self
            .device
            .newHeapWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewHeapWithDescriptor",
                    format!("a sparse heap of {} bytes returned nil", spec.size),
                )
            })?;
        self.heaps.push(heap);
        Ok(Handle(self.heaps.len() as u32 - 1))
    }

    fn heap_facts(&mut self, heap: Handle) -> Answer<HeapFacts> {
        let heap = self.held_heap(heap)?;
        Ok(HeapFacts {
            heap_type: heap.r#type().0 as u64,
            size: heap.size() as u64,
            used_size: heap.usedSize() as u64,
        })
    }

    fn heap_texture(&mut self, heap: Handle, spec: &SparseTextureSpec) -> Answer<Handle> {
        let descriptor = sparse_texture_descriptor(spec);
        let texture = self
            .held_heap(heap)?
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptor",
                    format!("{} on a heap returned nil", spec.site()),
                )
            })?;
        self.textures.push(texture);
        Ok(Handle(self.textures.len() as u32 - 1))
    }

    fn sparse_texture_facts(&mut self, texture: Handle) -> Answer<SparseTextureFacts> {
        let texture = self.held_texture(texture)?;
        Ok(SparseTextureFacts {
            is_sparse: texture.isSparse(),
            first_mipmap_in_tail: texture.firstMipmapInTail() as u64,
            tail_size_in_bytes: texture.tailSizeInBytes() as u64,
        })
    }
}

/// Metal's `MTLSparsePageSize` for a case's page, by its raw value, so the two spellings cannot
/// drift apart.
fn metal_page(page: SparsePageSize) -> MTLSparsePageSize {
    MTLSparsePageSize(page.raw() as isize)
}

/// Metal's descriptor for a case's sparse texture: every field the case names, by its raw ordinal.
///
/// The storage is the heap's (`Private`) and the usage is shader-read, with the render-target bit
/// where the case asks for it. Metal stops any other storage for a heap's texture even without its
/// validation layer, so neither is a field a case may set.
fn sparse_texture_descriptor(spec: &SparseTextureSpec) -> Retained<MTLTextureDescriptor> {
    let descriptor = MTLTextureDescriptor::new();
    descriptor.setTextureType(MTLTextureType(spec.texture_type.as_raw() as usize));
    descriptor.setPixelFormat(MTLPixelFormat(spec.format.as_raw() as usize));
    // SAFETY: each extent and count is a case's own, and the setters read nothing but the value
    // they are given.
    unsafe {
        descriptor.setWidth(spec.size[0] as usize);
        descriptor.setHeight(spec.size[1] as usize);
        descriptor.setDepth(spec.size[2] as usize);
        descriptor.setMipmapLevelCount(spec.levels as usize);
        descriptor.setSampleCount(spec.samples as usize);
        descriptor.setArrayLength(spec.array_length as usize);
    }
    let usage = if spec.render_target {
        MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget
    } else {
        MTLTextureUsage::ShaderRead
    };
    descriptor.setUsage(usage);
    descriptor.setStorageMode(MTLStorageMode::Private);
    descriptor
}

/// A command buffer the slot probe holds, so the thread that frees a slot need not be the thread
/// that made it.
///
/// `objc2` gives `MTLCommandBuffer` no `Send`, because Metal documents a command buffer as unsafe
/// to *encode into* from more than one thread. The probe encodes into none: it makes each command
/// buffer on one thread and only ever releases it from another, and releasing an Objective-C object
/// off the thread that made it is sound, since the retain count is atomic.
pub struct Made(pub Retained<ProtocolObject<dyn MTLCommandBuffer>>);

// SAFETY: as the type's documentation states: the probe never encodes into a held command buffer,
// and the only operation that crosses threads is the release.
unsafe impl Send for Made {}

/// Metal's descriptor for a queried one: every field the case names, by its raw ordinal, so a
/// private format and a private usage bit reach Metal as readily as a public one.
fn queried_descriptor(query: &HeapTextureQuery) -> Retained<MTLTextureDescriptor> {
    let descriptor = MTLTextureDescriptor::new();
    descriptor.setTextureType(MTLTextureType(query.texture_type.as_raw() as usize));
    descriptor.setPixelFormat(MTLPixelFormat(query.format.as_raw() as usize));
    // SAFETY: each extent, count and index is a case's own, and the setters read nothing but the
    // value they are given.
    unsafe {
        descriptor.setWidth(query.size[0] as usize);
        descriptor.setHeight(query.size[1] as usize);
        descriptor.setDepth(query.size[2] as usize);
        descriptor.setMipmapLevelCount(query.levels as usize);
        descriptor.setSampleCount(query.samples as usize);
        descriptor.setArrayLength(query.array_length as usize);
    }
    descriptor.setUsage(MTLTextureUsage(query.usage.as_raw() as usize));
    descriptor.setStorageMode(storage_mode(query.storage).0);
    descriptor
}

/// The case's enumerations reach Metal's by their raw ordinals, which are Metal's own.
fn compare_function(function: metal2vulkan::metal::CompareFunction) -> MTLCompareFunction {
    MTLCompareFunction(function.as_raw() as usize)
}

fn stencil_operation(operation: metal2vulkan::metal::StencilOperation) -> MTLStencilOperation {
    MTLStencilOperation(operation.as_raw() as usize)
}

/// One face's stencil state as Metal's descriptor.
fn stencil_descriptor(face: &StencilDescriptor) -> Retained<MTLStencilDescriptor> {
    let descriptor = MTLStencilDescriptor::new();
    descriptor.setStencilCompareFunction(compare_function(face.stencil_compare_function));
    descriptor.setStencilFailureOperation(stencil_operation(face.stencil_failure_operation));
    descriptor.setDepthFailureOperation(stencil_operation(face.depth_failure_operation));
    descriptor.setDepthStencilPassOperation(stencil_operation(face.depth_stencil_pass_operation));
    descriptor.setReadMask(face.read_mask);
    descriptor.setWriteMask(face.write_mask);
    descriptor
}

/// The storage mode and resource options a [`Storage`] names: the Metal side's one spelling of the
/// three modes, which [`crate::metal_work`] reads too, so a case's storage mode cannot mean one
/// thing to a queried descriptor and another to a program's buffer.
pub fn storage_mode(storage: Storage) -> (MTLStorageMode, MTLResourceOptions) {
    match storage {
        Storage::Shared => (
            MTLStorageMode::Shared,
            MTLResourceOptions::StorageModeShared,
        ),
        Storage::Managed => (
            MTLStorageMode::Managed,
            MTLResourceOptions::StorageModeManaged,
        ),
        Storage::Private => (
            MTLStorageMode::Private,
            MTLResourceOptions::StorageModePrivate,
        ),
    }
}

/// The pixel format [`crate::case::TARGET_FORMAT`] names, as Metal spells it.
const TARGET_PIXEL_FORMAT: MTLPixelFormat = MTLPixelFormat::RGBA32Uint;

impl MetalSide {
    /// The descriptor of a mesh render pipeline of `library`'s `functions`.
    fn mesh_descriptor(
        &self,
        library: Handle,
        functions: &MeshFunctions<'_>,
    ) -> Answer<Retained<MTLMeshRenderPipelineDescriptor>> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = |name: &str| {
            library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| Failure::named("NoSuchFunction", name.to_owned()))
        };
        let descriptor = MTLMeshRenderPipelineDescriptor::new();
        // SAFETY: each function is of this library, and the descriptor is this call's own.
        unsafe {
            descriptor.setMeshFunction(Some(&*function(functions.mesh)?));
            if let Some(object) = functions.object {
                descriptor.setObjectFunction(Some(&*function(object)?));
            }
            descriptor.setFragmentFunction(Some(&*function(functions.fragment)?));
            let attachments = descriptor.colorAttachments();
            for at in 0..functions.attachments as usize {
                attachments
                    .objectAtIndexedSubscript(at)
                    .setPixelFormat(TARGET_PIXEL_FORMAT);
            }
        }
        Ok(descriptor)
    }

    /// The state `descriptor` makes, kept by handle.
    fn made_mesh_pipeline(
        &mut self,
        descriptor: &MTLMeshRenderPipelineDescriptor,
    ) -> Answer<Handle> {
        let state = self
            .device
            .newRenderPipelineStateWithMeshDescriptor_options_reflection_error(
                descriptor,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|error| Failure::named("NewMeshRenderPipelineState", error.to_string()))?;
        Ok(keep(
            &mut self.render_pipelines,
            RenderPipeline::unreflected(state),
        ))
    }

    /// One command buffer and one render encoder over `targets`, each attachment cleared to zero
    /// and stored.
    #[allow(clippy::type_complexity)]
    fn pass(
        &self,
        targets: &[Handle],
    ) -> Answer<(
        Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>,
    )> {
        let attachments: Vec<Retained<ProtocolObject<dyn MTLTexture>>> = targets
            .iter()
            .map(|handle| self.held_texture(*handle).map(Retained::from))
            .collect::<Answer<Vec<_>>>()?;
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        for (at, target) in attachments.iter().enumerate() {
            // SAFETY: `at` is below the eight attachments every Metal pass holds.
            let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(at) };
            attachment.setTexture(Some(target));
            attachment.setLoadAction(MTLLoadAction::Clear);
            attachment.setClearColor(MTLClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            });
            attachment.setStoreAction(MTLStoreAction::Store);
        }
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        let encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or_else(|| Failure::named("RenderCommandEncoder", "returned nil".to_owned()))?;
        Ok((command_buffer, encoder))
    }

    /// One command buffer and one render encoder over `targets` in `tile`-sized tiles, each
    /// attachment loaded and stored.
    #[allow(clippy::type_complexity)]
    fn loading_pass(
        &self,
        targets: &[Handle],
        tile: [u32; 2],
    ) -> Answer<(
        Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>,
    )> {
        let attachments: Vec<Retained<ProtocolObject<dyn MTLTexture>>> = targets
            .iter()
            .map(|handle| self.held_texture(*handle).map(Retained::from))
            .collect::<Answer<Vec<_>>>()?;
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        for (at, target) in attachments.iter().enumerate() {
            // SAFETY: `at` is below the eight attachments every Metal pass holds.
            let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(at) };
            attachment.setTexture(Some(target));
            attachment.setLoadAction(MTLLoadAction::Load);
            attachment.setStoreAction(MTLStoreAction::Store);
        }
        pass.setTileWidth(tile[0] as usize);
        pass.setTileHeight(tile[1] as usize);
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        let encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or_else(|| Failure::named("RenderCommandEncoder", "returned nil".to_owned()))?;
        Ok((command_buffer, encoder))
    }

    /// Encode one of a case's tile-pass steps into `encoder`.
    fn encode_tile_step(
        &self,
        encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        step: &TileStep,
    ) -> Answer<()> {
        match step {
            TileStep::Pipeline(state) => {
                let state = &self
                    .render_pipelines
                    .get(state.0 as usize)
                    .ok_or_else(|| {
                        Failure::named("NoSuchHandle", format!("tile pipeline {}", state.0))
                    })?
                    .state;
                encoder.setRenderPipelineState(state);
            }
            TileStep::TileBuffer {
                buffer,
                offset,
                index,
            } => {
                let buffer = self.held(*buffer)?;
                // SAFETY: the buffer is at least as long as the function's binding, and the
                // encoder outlives the call.
                unsafe {
                    encoder.setTileBuffer_offset_atIndex(
                        Some(buffer),
                        *offset as usize,
                        *index as usize,
                    )
                };
            }
            TileStep::TileBytes { bytes, index } => {
                let pointer =
                    std::ptr::NonNull::new(bytes.as_ptr().cast::<std::ffi::c_void>().cast_mut())
                        .ok_or_else(|| Failure::named("NoBytes", "an empty inline binding"))?;
                // SAFETY: `bytes` lives for the call, which copies them.
                unsafe {
                    encoder.setTileBytes_length_atIndex(pointer, bytes.len(), *index as usize)
                };
            }
            TileStep::TileTexture { texture, index } => {
                let texture = self.held_texture(*texture)?;
                // SAFETY: the texture is kept alive by `self.textures`, and the encoder outlives
                // the call.
                unsafe { encoder.setTileTexture_atIndex(Some(texture), *index as usize) };
            }
            TileStep::DispatchThreadsPerTile(threads) => {
                encoder.dispatchThreadsPerTile(MTLSize {
                    width: threads[0] as usize,
                    height: threads[1] as usize,
                    depth: threads[2] as usize,
                });
            }
        }
        Ok(())
    }

    /// Encode one of a case's render-pass steps into `encoder`.
    fn encode_step(
        &self,
        encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        step: &MeshStep,
    ) -> Answer<()> {
        let size = |axes: [u32; 3]| MTLSize {
            width: axes[0] as usize,
            height: axes[1] as usize,
            depth: axes[2] as usize,
        };
        match step {
            MeshStep::Pipeline(state) => {
                let state = &self
                    .render_pipelines
                    .get(state.0 as usize)
                    .ok_or_else(|| {
                        Failure::named("NoSuchHandle", format!("mesh pipeline {}", state.0))
                    })?
                    .state;
                encoder.setRenderPipelineState(state);
            }
            MeshStep::ObjectBuffer {
                buffer,
                offset,
                index,
            } => {
                let buffer = self.held(*buffer)?;
                // SAFETY: the buffer is at least as long as the function's binding, and the
                // encoder outlives the call.
                unsafe {
                    encoder.setObjectBuffer_offset_atIndex(
                        Some(buffer),
                        *offset as usize,
                        *index as usize,
                    )
                };
            }
            MeshStep::ObjectBytes { bytes, index } => {
                let pointer =
                    std::ptr::NonNull::new(bytes.as_ptr().cast::<std::ffi::c_void>().cast_mut())
                        .ok_or_else(|| Failure::named("NoBytes", "an empty inline binding"))?;
                // SAFETY: `bytes` lives for the call, which copies them.
                unsafe {
                    encoder.setObjectBytes_length_atIndex(pointer, bytes.len(), *index as usize)
                };
            }
            MeshStep::MeshBuffer {
                buffer,
                offset,
                index,
            } => {
                let buffer = self.held(*buffer)?;
                // SAFETY: as the object buffer's.
                unsafe {
                    encoder.setMeshBuffer_offset_atIndex(
                        Some(buffer),
                        *offset as usize,
                        *index as usize,
                    )
                };
            }
            MeshStep::FragmentBuffer {
                buffer,
                offset,
                index,
            } => {
                let buffer = self.held(*buffer)?;
                // SAFETY: as the object buffer's.
                unsafe {
                    encoder.setFragmentBuffer_offset_atIndex(
                        Some(buffer),
                        *offset as usize,
                        *index as usize,
                    )
                };
            }
            MeshStep::DrawThreadgroups {
                groups,
                object,
                mesh,
            } => encoder
                .drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                    size(*groups),
                    size(*object),
                    size(*mesh),
                ),
            MeshStep::DrawThreads {
                threads,
                object,
                mesh,
            } => encoder.drawMeshThreads_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                size(*threads),
                size(*object),
                size(*mesh),
            ),
            MeshStep::DrawIndirect {
                buffer,
                offset,
                object,
                mesh,
            } => {
                let buffer = self.held(*buffer)?;
                // SAFETY: the buffer holds three 32-bit counts at the offset, which the case
                // wrote before the pass.
                unsafe {
                    encoder.drawMeshThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                        buffer,
                        *offset as usize,
                        size(*object),
                        size(*mesh),
                    );
                }
            }
            MeshStep::BarrierScope {
                scope,
                after,
                before,
            } => encoder.memoryBarrierWithScope_afterStages_beforeStages(
                MTLBarrierScope(*scope as usize),
                MTLRenderStages(*after as usize),
                MTLRenderStages(*before as usize),
            ),
            MeshStep::BarrierResources {
                buffers,
                after,
                before,
            } => {
                let over: Vec<&ProtocolObject<dyn MTLResource>> = buffers
                    .iter()
                    .map(|handle| self.held(*handle).map(ProtocolObject::from_ref))
                    .collect::<Answer<Vec<_>>>()?;
                let mut pointers: Vec<std::ptr::NonNull<ProtocolObject<dyn MTLResource>>> =
                    over.iter().map(|resource| (*resource).into()).collect();
                let first = std::ptr::NonNull::new(pointers.as_mut_ptr())
                    .ok_or_else(|| Failure::named("NoResources", "an empty barrier"))?;
                // SAFETY: `pointers` holds exactly `pointers.len()` resource pointers, each
                // retained by `over` for the whole call.
                unsafe {
                    encoder.memoryBarrierWithResources_count_afterStages_beforeStages(
                        first,
                        pointers.len(),
                        MTLRenderStages(*after as usize),
                        MTLRenderStages(*before as usize),
                    );
                }
            }
        }
        Ok(())
    }

    /// The samples of the pass's own attachments, which is the count Metal holds its custom
    /// sample positions to: the first attachment the pass states, and 1 for a pass with none.
    fn pass_samples(&self, pass: &RenderPass) -> Answer<u32> {
        let attached = pass
            .colors
            .iter()
            .filter_map(|color| color.texture)
            .chain(pass.depth.as_ref().map(|depth| depth.texture))
            .chain(pass.stencil.as_ref().map(|stencil| stencil.texture))
            .next();
        match attached {
            Some(handle) => Ok(self.held_texture(handle)?.sampleCount() as u32),
            None => Ok(1),
        }
    }

    /// Encode one render pass of a program into `command_buffer`.
    fn encode_pass(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        pass: &RenderPass,
    ) -> Answer<()> {
        let descriptor = MTLRenderPassDescriptor::renderPassDescriptor();
        for (at, color) in pass.colors.iter().enumerate() {
            let Some(texture) = color.texture else {
                continue;
            };
            let texture: Retained<ProtocolObject<dyn MTLTexture>> =
                Retained::from(self.held_texture(texture)?);
            // SAFETY: `at` is below the eight color attachments every Metal pass holds, which the
            // case's own guard keeps it to.
            let attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(at) };
            attachment.setTexture(Some(&texture));
            attachment.setLoadAction(MTLLoadAction(color.load.raw() as usize));
            attachment.setStoreAction(MTLStoreAction(color.store.raw() as usize));
            attachment.setClearColor(MTLClearColor {
                red: color.clear[0],
                green: color.clear[1],
                blue: color.clear[2],
                alpha: color.clear[3],
            });
            attachment.setStoreActionOptions(MTLStoreActionOptions::from_bits_retain(
                color.options.raw() as usize,
            ));
            if let Some(resolve) = color.resolve {
                let resolve: Retained<ProtocolObject<dyn MTLTexture>> =
                    Retained::from(self.held_texture(resolve)?);
                attachment.setResolveTexture(Some(&resolve));
            }
        }
        if let Some(depth) = &pass.depth {
            let texture: Retained<ProtocolObject<dyn MTLTexture>> =
                Retained::from(self.held_texture(depth.texture)?);
            let attachment = descriptor.depthAttachment();
            attachment.setTexture(Some(&texture));
            attachment.setLoadAction(MTLLoadAction(depth.load.raw() as usize));
            attachment.setStoreAction(MTLStoreAction(depth.store.raw() as usize));
            attachment.setClearDepth(depth.clear);
            attachment.setStoreActionOptions(MTLStoreActionOptions::from_bits_retain(
                depth.options.raw() as usize,
            ));
            attachment.setDepthResolveFilter(MTLMultisampleDepthResolveFilter(
                depth.filter.raw() as usize
            ));
            if let Some(resolve) = depth.resolve {
                let resolve: Retained<ProtocolObject<dyn MTLTexture>> =
                    Retained::from(self.held_texture(resolve)?);
                attachment.setResolveTexture(Some(&resolve));
            }
        }
        if let Some(stencil) = &pass.stencil {
            let texture: Retained<ProtocolObject<dyn MTLTexture>> =
                Retained::from(self.held_texture(stencil.texture)?);
            let attachment = descriptor.stencilAttachment();
            attachment.setTexture(Some(&texture));
            attachment.setLoadAction(MTLLoadAction(stencil.load.raw() as usize));
            attachment.setStoreAction(MTLStoreAction(stencil.store.raw() as usize));
            attachment.setClearStencil(stencil.clear);
            attachment.setStoreActionOptions(MTLStoreActionOptions::from_bits_retain(
                stencil.options.raw() as usize,
            ));
            attachment.setStencilResolveFilter(MTLMultisampleStencilResolveFilter(
                stencil.filter.raw() as usize,
            ));
            if let Some(resolve) = stencil.resolve {
                let resolve: Retained<ProtocolObject<dyn MTLTexture>> =
                    Retained::from(self.held_texture(resolve)?);
                attachment.setResolveTexture(Some(&resolve));
            }
        }
        if let Some(buffer) = pass.visibility {
            descriptor.setVisibilityResultBuffer(Some(self.held(buffer)?));
        }
        if let Some(size) = pass.size {
            descriptor.setRenderTargetWidth(size[0] as usize);
            descriptor.setRenderTargetHeight(size[1] as usize);
            descriptor.setDefaultRasterSampleCount(1);
        }
        if !pass.positions.is_empty() {
            // The one place the Metal leg encodes `setSamplePositions:count:`, so the guard that
            // keeps the suite off Metal's fatal shapes is reached by every case that states
            // positions, whatever else it does. The samples are the pass's own attachments', which
            // is what Metal counts the positions against.
            let samples = self.pass_samples(pass)?;
            if let Some(refusal) = crate::render::forbidden_positions(&pass.positions, samples) {
                return Err(refusal);
            }
            let positions: Vec<MTLSamplePosition> = pass
                .positions
                .iter()
                .map(|position| MTLSamplePosition {
                    x: position.x,
                    y: position.y,
                })
                .collect();
            // SAFETY: the pointer is to `positions.len()` positions that live for the call, and
            // the count is at or below the pass's samples, which the guard above has just held.
            unsafe {
                descriptor.setSamplePositions_count(positions.as_ptr(), positions.len());
            }
        }
        let encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&descriptor)
            .ok_or_else(|| Failure::named("RenderCommandEncoder", "returned nil".to_owned()))?;
        for step in &pass.steps {
            self.encode_step_of(&encoder, step)?;
        }
        encoder.endEncoding();
        Ok(())
    }

    /// Encode one of a case's render-pass steps into `encoder`.
    fn encode_step_of(
        &self,
        encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        step: &RenderStep,
    ) -> Answer<()> {
        let primitive =
            |kind: metal2vulkan::metal::PrimitiveType| MTLPrimitiveType(kind.as_raw() as usize);
        match step {
            RenderStep::Pipeline(state) => {
                let state = self.held_render_state(*state)?;
                encoder.setRenderPipelineState(&state);
            }
            RenderStep::DepthStencilState(state) => {
                let state = self
                    .depth_stencil_states
                    .get(state.0 as usize)
                    .ok_or_else(|| {
                        Failure::named("NoSuchHandle", format!("depth/stencil state {}", state.0))
                    })?;
                encoder.setDepthStencilState(Some(state));
            }
            RenderStep::StencilReference(value) => encoder.setStencilReferenceValue(*value),
            RenderStep::VertexBuffer {
                buffer,
                offset,
                index,
            } => {
                let held = self.held(*buffer)?;
                // SAFETY: the buffer is at least as long as the vertex function's binding from
                // that offset, which the case states as one of its inputs, and the encoder
                // outlives the call.
                unsafe {
                    encoder.setVertexBuffer_offset_atIndex(
                        Some(held),
                        *offset as usize,
                        *index as usize,
                    );
                }
            }
            RenderStep::FragmentBuffer {
                buffer,
                offset,
                index,
            } => {
                let held = self.held(*buffer)?;
                // SAFETY: as the vertex buffer's, for the fragment function's binding.
                unsafe {
                    encoder.setFragmentBuffer_offset_atIndex(
                        Some(held),
                        *offset as usize,
                        *index as usize,
                    );
                }
            }
            RenderStep::FragmentTexture { texture, index } => {
                let held = self.held_texture(*texture)?;
                // SAFETY: the texture is of the kind and access the fragment function declares at
                // that slot, and the encoder outlives the call.
                unsafe { encoder.setFragmentTexture_atIndex(Some(held), *index as usize) };
            }
            RenderStep::FragmentSampler { sampler, index } => {
                let held = self.held_sampler(*sampler)?;
                // SAFETY: the fragment function declares a sampler at that slot, and the encoder
                // outlives the call.
                unsafe { encoder.setFragmentSamplerState_atIndex(Some(held), *index as usize) };
            }
            RenderStep::VertexBytes { bytes, index } => {
                let pointer =
                    std::ptr::NonNull::new(bytes.as_ptr().cast::<std::ffi::c_void>().cast_mut())
                        .ok_or_else(|| Failure::named("NoBytes", "an empty inline binding"))?;
                // SAFETY: `bytes` lives for the call, which copies them, and is at least as long
                // as the vertex function's binding at that slot.
                unsafe {
                    encoder.setVertexBytes_length_atIndex(pointer, bytes.len(), *index as usize)
                };
            }
            RenderStep::Viewport(rect) => encoder.setViewport(MTLViewport {
                originX: rect.origin[0],
                originY: rect.origin[1],
                width: rect.size[0],
                height: rect.size[1],
                znear: rect.depth[0],
                zfar: rect.depth[1],
            }),
            RenderStep::VisibilityResultMode { mode, offset } => {
                encoder.setVisibilityResultMode_offset(
                    MTLVisibilityResultMode(mode.raw() as usize),
                    *offset as usize,
                );
            }
            RenderStep::Draw {
                primitive: kind,
                first,
                count,
                instances,
            } => {
                // SAFETY: the draw names a vertex range of the case's own triangle, and every
                // buffer the vertex function reads was bound by a step before it.
                unsafe {
                    encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                        primitive(*kind),
                        *first as usize,
                        *count as usize,
                        *instances as usize,
                        0,
                    );
                }
            }
            RenderStep::DrawIndexed {
                primitive: kind,
                count,
                index_type,
                indices,
                offset,
                instances,
                base_vertex,
                base_instance,
            } => {
                let buffer = self.held(*indices)?;
                // SAFETY: the index buffer holds the case's own indices from that offset, which
                // the case states as one of its inputs, and the counts are its own.
                unsafe {
                    encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                        primitive(*kind),
                        *count as usize,
                        MTLIndexType(index_type.as_raw() as usize),
                        buffer,
                        *offset as usize,
                        *instances as usize,
                        *base_vertex as isize,
                        *base_instance as usize,
                    );
                }
            }
            RenderStep::DrawIndirect {
                primitive: kind,
                arguments,
                offset,
            } => {
                let buffer = self.held(*arguments)?;
                // SAFETY: the buffer holds the draw's four 32-bit arguments at that offset, which
                // the case wrote before the pass.
                unsafe {
                    encoder.drawPrimitives_indirectBuffer_indirectBufferOffset(
                        primitive(*kind),
                        buffer,
                        *offset as usize,
                    );
                }
            }
        }
        Ok(())
    }
}

/// The render vocabulary ([`crate::render`]) on native Metal.
impl MetalSide {
    /// The render pipeline state `handle` names.
    fn held_render_state(
        &self,
        handle: Handle,
    ) -> Answer<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        self.render_pipelines
            .get(handle.0 as usize)
            .map(|held| held.state.clone())
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("render pipeline {}", handle.0)))
    }

    /// Encode one compute dispatch of a program into `command_buffer`.
    fn encode_dispatch(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        dispatch: &Dispatch,
    ) -> Answer<()> {
        let state = self
            .pipelines
            .get(dispatch.pipeline.0 as usize)
            .ok_or_else(|| {
                Failure::named(
                    "NoSuchHandle",
                    format!("compute pipeline {}", dispatch.pipeline.0),
                )
            })?
            .clone();
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| Failure::named("ComputeCommandEncoder", "returned nil".to_owned()))?;
        encoder.setComputePipelineState(&state);
        for (at, (buffer, offset)) in dispatch.buffers.iter().enumerate() {
            let held = self.held(*buffer)?;
            // SAFETY: the buffer is at least as long as the kernel's binding from that offset,
            // which the case states as one of its inputs, and the encoder outlives the call.
            unsafe { encoder.setBuffer_offset_atIndex(Some(held), *offset as usize, at) };
        }
        for (at, texture) in dispatch.textures.iter().enumerate() {
            let held = self.held_texture(*texture)?;
            // SAFETY: the texture is of the kind and access the kernel declares at that slot, and
            // the encoder outlives the call.
            unsafe { encoder.setTexture_atIndex(Some(held), at) };
        }
        encoder.dispatchThreads_threadsPerThreadgroup(size(dispatch.grid), size(dispatch.threads));
        encoder.endEncoding();
        Ok(())
    }

    /// Encode one texture-to-buffer blit of a program into `command_buffer`.
    fn encode_read_texture(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        texture: Handle,
        into: Handle,
        offset: u64,
        row_pitch: u32,
        extent: [u32; 2],
    ) -> Answer<()> {
        let texture: Retained<ProtocolObject<dyn MTLTexture>> =
            Retained::from(self.held_texture(texture)?);
        let buffer: Retained<ProtocolObject<dyn MTLBuffer>> = Retained::from(self.held(into)?);
        let bytes = row_pitch as u64 * u64::from(extent[1]);
        if u64::try_from(buffer.length()).unwrap_or(0) < offset + bytes {
            return Err(Failure::named(
                "PastTheBuffer",
                format!(
                    "{bytes} bytes at {offset} of a buffer of {}",
                    buffer.length()
                ),
            ));
        }
        let encoder = command_buffer
            .blitCommandEncoder()
            .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil".to_owned()))?;
        // SAFETY: the buffer holds `bytes` from `offset`, checked just above, and the region is
        // level 0's whole extent of a texture the case made at that extent.
        unsafe {
            encoder
                .copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                    &texture,
                    0,
                    0,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                    MTLSize {
                        width: extent[0] as usize,
                        height: extent[1] as usize,
                        depth: 1,
                    },
                    &buffer,
                    offset as usize,
                    row_pitch as usize,
                    bytes as usize,
                );
        }
        encoder.endEncoding();
        Ok(())
    }
}

/// A command-buffer handler block the probes keep alive: the owning half of
/// [`MTLCommandBufferHandler`], which is the raw pointer Metal's setters take.
type HandlerBlock =
    block2::RcBlock<dyn Fn(std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>)>;

/// A command buffer a probe waits for the scheduling of on a thread of its own.
///
/// `objc2` gives `MTLCommandBuffer` no `Send`, because Metal documents a command buffer as unsafe
/// to **encode into** from more than one thread. The probe encodes into none on that thread: it
/// calls `waitUntilScheduled`, which Metal documents as a wait any thread may make, and then
/// releases the reference. Both are sound off the thread that made it, since the retain count is
/// atomic.
struct Waited(Retained<ProtocolObject<dyn MTLCommandBuffer>>);

// SAFETY: as the type's documentation states: the probe never encodes into it on the waiting
// thread, and the only calls that cross threads are `waitUntilScheduled` and the release.
unsafe impl Send for Waited {}

impl Waited {
    /// `waitUntilScheduled`.
    ///
    /// A method rather than a field read at the call site, and that is load-bearing: a closure
    /// reaching `waited.0` captures the `Retained` on its own, which is not `Send`, so the whole
    /// wrapper — the thing this type's `Send` is about — would never be what crossed the thread.
    fn wait_until_scheduled(&self) {
        self.0.waitUntilScheduled();
    }
}

/// A command buffer's completion, as a bound the probe's own thread can wait on.
///
/// Metal has **no bounded wait**: `waitUntilCompleted` either returns or does not. So an event
/// probe, whose whole subject is a command buffer held back by a value that has not arrived, never
/// calls it — it adds a completed handler before the commit and waits on this channel with a bound.
/// A bound that runs out is a recorded fact ([`crate::event_orders::Status::NeverSettled`]), not a
/// hang.
struct Completion {
    done: std::sync::mpsc::Receiver<()>,
    /// The block Metal calls, kept alive for as long as the probe runs. `addCompletedHandler:`
    /// takes a pointer, and keeping our own reference means the block's lifetime does not rest on
    /// what Metal does with it.
    _block: HandlerBlock,
}

/// A completed handler on `command_buffer` that signals its channel, added before the commit.
fn completion_channel(command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> Completion {
    let (sender, done) = std::sync::mpsc::channel();
    let block = block2::RcBlock::new(
        move |_: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            let _ = sender.send(());
        },
    );
    // SAFETY: the block is a valid heap block, kept alive in the `Completion` this answers for as
    // long as the probe runs, which outlives every call Metal makes of it.
    unsafe { command_buffer.addCompletedHandler(block2::RcBlock::as_ptr(&block)) };
    Completion {
        done,
        _block: block,
    }
}

/// The status `command_buffer` reads once its completion arrived, or that it never did.
fn settled(
    completion: &Completion,
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
) -> crate::event_orders::Status {
    if completion
        .done
        .recv_timeout(crate::event_orders::COMPLETION)
        .is_err()
    {
        crate::event_orders::Status::NeverSettled
    } else {
        crate::event_orders::Status::read(status_word(command_buffer.status()))
    }
}

/// The names a command buffer's handlers logged, in the order Metal called them, with each name
/// also sent so the probe can wait for one rather than guess when the log is whole.
///
/// The channel is what makes the log readable: Metal fixes no order between two handlers of one
/// command buffer that this probe could rely on, so the probe waits for the name it cares about to
/// arrive instead of assuming the handler it added first is the one that ran first.
struct HandlerLog {
    called: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    sender: std::sync::mpsc::Sender<String>,
    arrived: std::sync::mpsc::Receiver<String>,
    /// The blocks Metal calls, kept alive for as long as the probe runs.
    blocks: Vec<HandlerBlock>,
}

impl HandlerLog {
    /// An empty log.
    fn new() -> Self {
        let (sender, arrived) = std::sync::mpsc::channel();
        Self {
            called: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            sender,
            arrived,
            blocks: Vec::new(),
        }
    }

    /// Add a handler that logs itself as `name` to `command_buffer`, as the kind `add` names.
    ///
    /// `add` is the raw setter so one body serves both `addScheduledHandler:` and
    /// `addCompletedHandler:`.
    fn add(&mut self, name: &'static str, add: impl FnOnce(MTLCommandBufferHandler)) {
        let (called, sender) = (std::sync::Arc::clone(&self.called), self.sender.clone());
        let block = block2::RcBlock::new(
            move |_: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                if let Ok(mut called) = called.lock() {
                    called.push(name.to_owned());
                }
                let _ = sender.send(name.to_owned());
            },
        );
        add(block2::RcBlock::as_ptr(&block));
        self.blocks.push(block);
    }

    /// Wait until the handler named `name` has been called, or `bound` runs out.
    fn wait_for(&self, name: &str, bound: std::time::Duration) {
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

/// The word a command buffer's status records as, in the runtime's spelling so the two sides'
/// transcripts hold one vocabulary.
pub(crate) fn status_word(status: MTLCommandBufferStatus) -> &'static str {
    match status {
        MTLCommandBufferStatus::NotEnqueued => "NotEnqueued",
        MTLCommandBufferStatus::Enqueued => "Enqueued",
        MTLCommandBufferStatus::Committed => "Committed",
        MTLCommandBufferStatus::Scheduled => "Scheduled",
        MTLCommandBufferStatus::Completed => "Completed",
        MTLCommandBufferStatus::Error => "Error",
        _ => "Unknown",
    }
}
