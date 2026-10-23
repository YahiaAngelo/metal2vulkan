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
    Answer, Bindings, CommandDraw, Failure, Handle, MeshCommand, MeshFunctions, MeshStep,
    PassOutcome, RateMapFacts, RateMapRequest, SamplerSpec, ShaderSource, Side, StageInRegion,
    StageInput, TextureRole, TextureSpec, Venue,
};
use crate::transcript::Value;
use metal2vulkan::metal::PixelFormat;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{
    MTLAttributeFormat, MTLBarrierScope, MTLBuffer, MTLClearColor, MTLCommandBuffer,
    MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineDescriptor, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLIndexType, MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor,
    MTLIndirectCommandType, MTLIndirectRenderCommand, MTLLibrary, MTLLoadAction,
    MTLMeshRenderPipelineDescriptor, MTLOrigin, MTLPipelineOption, MTLPixelFormat,
    MTLRasterizationRateLayerDescriptor, MTLRasterizationRateMap,
    MTLRasterizationRateMapDescriptor, MTLRegion, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineState, MTLRenderStages, MTLResource, MTLResourceOptions, MTLResourceUsage,
    MTLSamplerAddressMode, MTLSamplerBorderColor, MTLSamplerDescriptor, MTLSamplerMinMagFilter,
    MTLSamplerState, MTLSize, MTLStageInputOutputDescriptor, MTLStepFunction, MTLStorageMode,
    MTLStoreAction, MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
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
    /// The mesh render pipeline states the case has made.
    mesh_pipelines: Vec<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    /// The indirect command buffers the case has made.
    command_buffers_indirect: Vec<Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>>,
    /// The rasterization-rate maps the case has made. Its own pool: a rate map is its own
    /// Metal object, not another name for one of the pools above.
    rate_maps: Vec<Retained<ProtocolObject<dyn MTLRasterizationRateMap>>>,
    /// The scratch directory the metallibs are built in, removed when the side is dropped.
    scratch: Scratch,
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
            mesh_pipelines: Vec::new(),
            command_buffers_indirect: Vec::new(),
            rate_maps: Vec::new(),
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
            let state = self
                .mesh_pipelines
                .get(command.pipeline.0 as usize)
                .ok_or_else(|| {
                    Failure::named(
                        "NoSuchHandle",
                        format!("mesh pipeline {}", command.pipeline.0),
                    )
                })?;
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
        Ok(keep(&mut self.mesh_pipelines, state))
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
                let state = self.mesh_pipelines.get(state.0 as usize).ok_or_else(|| {
                    Failure::named("NoSuchHandle", format!("mesh pipeline {}", state.0))
                })?;
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
