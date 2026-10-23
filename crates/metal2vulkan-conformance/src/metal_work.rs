//! Running a [`Work`] program on native Metal, through public `objc2-metal` calls alone.
//!
//! This exists because a carried-only row's stated reason was wrong. `two_plane_copies` and
//! `two_plane_heap_blits` said the two-plane extended-range formats are Apple's private ones and so
//! Metal has no public call for them. **The format is not private.** `MTLTextureDescriptor`'s pixel
//! format is a raw value and Metal makes, writes, blits and reads back 550 and 551 through entirely
//! public calls; what is private is `fillTexture:`, the *fill*. Two lanes lost their Metal leg to
//! that confusion, and a case can only run on Metal if something runs the program.
//!
//! So the interpreter mirrors [`crate::runtime_work`] call for call, in the same order, with one
//! blit encoder at a time, and refuses exactly the steps Metal really has no public call for:
//! `fillTexture:` and `fillTextureColor:`, which are Apple SPI. A case built on those still carries
//! its Metal answer; a case built on copies does not.
//!
//! Nothing here is unsafe for the GPU that the probes these cases came from were not: the same
//! shapes, through the same public calls, are what produced the tables in the first place.

use crate::case::{Answer, Failure};
use crate::work::{Done, Plane, Rect, Resource, Step, TextureKind, TextureSpec, Work};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBlitOption, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
    MTLCommandQueue, MTLDevice, MTLHeap, MTLHeapDescriptor, MTLHeapType, MTLOrigin, MTLPixelFormat,
    MTLRegion, MTLResourceOptions, MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
    MTLTextureType, MTLTextureUsage,
};

/// A resource the program made on Metal.
enum Made {
    Buffer(Retained<ProtocolObject<dyn MTLBuffer>>),
    Heap(Retained<ProtocolObject<dyn MTLHeap>>),
    Texture(Retained<ProtocolObject<dyn MTLTexture>>),
}

/// Run `work` on `device` through `queue`, answering what it read.
pub fn run(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    work: &Work,
) -> Answer<Done> {
    let mut made: Vec<Made> = Vec::with_capacity(work.resources.len());
    for (at, resource) in work.resources.iter().enumerate() {
        let one = make(device, &made, resource).map_err(|failure| at_resource(at, failure))?;
        made.push(one);
    }
    let mut state = Run {
        queue,
        made,
        command_buffer: None,
        encoder: None,
        reads: vec![None; work.reads()],
        next_read: 0,
        statuses: Vec::new(),
    };
    for (at, step) in work.steps.iter().enumerate() {
        state.step(step).map_err(|failure| at_step(at, failure))?;
    }
    state.close()?;
    let reads = state
        .reads
        .into_iter()
        .enumerate()
        .map(|(slot, bytes)| {
            bytes.ok_or_else(|| Failure::named("UnansweredRead", format!("read {slot}")))
        })
        .collect::<Answer<Vec<Vec<u8>>>>()?;
    Ok(Done {
        reads,
        statuses: state.statuses,
    })
}

/// The failure a resource's refusal is, naming which resource. The shape is
/// [`crate::runtime_work`]'s, so a refusal reads the same whichever side gave it.
fn at_resource(at: usize, failure: Failure) -> Failure {
    Failure {
        value: crate::transcript::Value::fields([
            ("resource", crate::transcript::Value::Uint(at as u64)),
            ("refused", failure.value),
        ]),
        lacks: failure.lacks,
    }
}

/// The failure a step's refusal is, naming which step.
fn at_step(at: usize, failure: Failure) -> Failure {
    Failure {
        value: crate::transcript::Value::fields([
            ("step", crate::transcript::Value::Uint(at as u64)),
            ("refused", failure.value),
        ]),
        lacks: failure.lacks,
    }
}

/// The storage mode and resource options a private or shared resource takes.
fn options(private: bool) -> (MTLStorageMode, MTLResourceOptions) {
    if private {
        (
            MTLStorageMode::Private,
            MTLResourceOptions::StorageModePrivate,
        )
    } else {
        (
            MTLStorageMode::Shared,
            MTLResourceOptions::StorageModeShared,
        )
    }
}

/// The descriptor `spec` names, field for field as [`crate::runtime_work`] builds it: the same
/// usage, the same array length, levels, samples and storage, so neither side can read a shape the
/// other does not.
fn descriptor(spec: &TextureSpec) -> Answer<Retained<MTLTextureDescriptor>> {
    let format = MTLPixelFormat(spec.format as usize);
    let mut usage = MTLTextureUsage::ShaderRead;
    if spec.render_target {
        usage |= MTLTextureUsage::RenderTarget | MTLTextureUsage::PixelFormatView;
    }
    if let Some(stated) = spec.usage {
        usage = MTLTextureUsage(stated as usize);
    }
    let (storage, resource_options) = options(spec.private);
    if spec.kind == TextureKind::TextureBuffer {
        // SAFETY: the constructor reads only the format, the width, the options and the usage it
        // is given.
        let descriptor = unsafe {
            MTLTextureDescriptor::textureBufferDescriptorWithPixelFormat_width_resourceOptions_usage(
                format,
                spec.width as usize,
                resource_options,
                usage,
            )
        };
        return Ok(descriptor);
    }
    // SAFETY: the constructor reads only the format and the size it is given.
    let descriptor = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            format,
            spec.width as usize,
            spec.height as usize,
            false,
        )
    };
    descriptor.setTextureType(match spec.kind {
        TextureKind::Type2D => MTLTextureType::Type2D,
        TextureKind::Type2DArray => MTLTextureType::Type2DArray,
        TextureKind::Type2DMultisample => MTLTextureType::Type2DMultisample,
        TextureKind::Type2DMultisampleArray => MTLTextureType::Type2DMultisampleArray,
        TextureKind::TextureBuffer => MTLTextureType::TypeTextureBuffer,
    });
    // SAFETY: each setter writes one field of a descriptor this call owns; nothing reads it until
    // the texture is made.
    unsafe {
        descriptor.setArrayLength(spec.slices.max(1) as usize);
        descriptor.setMipmapLevelCount(spec.levels.max(1) as usize);
        descriptor.setSampleCount(spec.samples.max(1) as usize);
    }
    descriptor.setUsage(usage);
    descriptor.setStorageMode(storage);
    descriptor.setAllowGPUOptimizedContents(spec.gpu_optimized);
    Ok(descriptor)
}

/// Make one resource.
fn make(
    device: &ProtocolObject<dyn MTLDevice>,
    made: &[Made],
    resource: &Resource,
) -> Answer<Made> {
    let heap_at = |at: usize| match made.get(at) {
        Some(Made::Heap(heap)) => Ok(heap),
        _ => Err(Failure::named("NoSuchResource", format!("heap {at}"))),
    };
    let buffer_at = |at: usize| match made.get(at) {
        Some(Made::Buffer(buffer)) => Ok(buffer),
        _ => Err(Failure::named("NoSuchResource", format!("buffer {at}"))),
    };
    let texture_at = |at: usize| match made.get(at) {
        Some(Made::Texture(texture)) => Ok(texture),
        _ => Err(Failure::named("NoSuchResource", format!("texture {at}"))),
    };
    let heap_of = |size: u64, private: bool| -> Answer<Made> {
        let (_, resource_options) = options(private);
        let descriptor = MTLHeapDescriptor::new();
        descriptor.setType(MTLHeapType::Placement);
        descriptor.setSize(size as usize);
        descriptor.setResourceOptions(resource_options);
        let heap = device.newHeapWithDescriptor(&descriptor).ok_or_else(|| {
            Failure::named("NewHeapWithDescriptor", format!("{size} returned nil"))
        })?;
        Ok(Made::Heap(heap))
    };
    match resource {
        Resource::Buffer { bytes } => {
            let buffer = device
                .newBufferWithLength_options(
                    bytes.len().max(1),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| {
                    Failure::named(
                        "NewBufferWithLength",
                        format!("{} bytes returned nil", bytes.len()),
                    )
                })?;
            let contents = buffer.contents();
            // SAFETY: the buffer was just made with at least `bytes.len()` bytes of shared
            // storage, which the host reaches, and no command uses it yet.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    contents.as_ptr().cast(),
                    bytes.len(),
                );
            }
            Ok(Made::Buffer(buffer))
        }
        Resource::PrivateBuffer { length } => {
            let buffer = device
                .newBufferWithLength_options(
                    (*length).max(1) as usize,
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| {
                    Failure::named(
                        "NewBufferWithLength",
                        format!("{length} private bytes returned nil"),
                    )
                })?;
            Ok(Made::Buffer(buffer))
        }
        Resource::Heap { size, private } => heap_of(*size, *private),
        Resource::HeapForTexture { spec, private } => {
            let descriptor = descriptor(spec)?;
            let sized = device.heapTextureSizeAndAlignWithDescriptor(&descriptor);
            heap_of(sized.size as u64, *private)
        }
        Resource::HeapBuffer {
            heap,
            length,
            offset,
        } => {
            let held = heap_at(*heap)?;
            let resource_options = held.resourceOptions();
            // SAFETY: the heap is a placement heap whose bytes `[offset, offset + length)` are
            // inside it, which the case states, and no resource of this program overlaps the
            // range before its own steps run.
            let buffer = unsafe {
                held.newBufferWithLength_options_offset(
                    *length as usize,
                    resource_options,
                    *offset as usize,
                )
            }
            .ok_or_else(|| {
                Failure::named(
                    "NewBufferWithLengthOffset",
                    format!("{length} bytes at {offset} returned nil"),
                )
            })?;
            Ok(Made::Buffer(buffer))
        }
        Resource::Texture { spec } => {
            let texture = device
                .newTextureWithDescriptor(&descriptor(spec)?.clone())
                .ok_or_else(|| {
                    Failure::named("NewTextureWithDescriptor", format!("{spec:?} returned nil"))
                })?;
            Ok(Made::Texture(texture))
        }
        Resource::HeapTexture { heap, spec, offset } => {
            let held = heap_at(*heap)?;
            let descriptor = descriptor(spec)?;
            // SAFETY: the heap is a placement heap large enough for the texture at `offset`,
            // which the case states as the probe's own placement.
            let texture =
                unsafe { held.newTextureWithDescriptor_offset(&descriptor, *offset as usize) }
                    .ok_or_else(|| {
                        Failure::named(
                            "NewTextureWithDescriptorOffset",
                            format!("{spec:?} at {offset} returned nil"),
                        )
                    })?;
            Ok(Made::Texture(texture))
        }
        Resource::TextureOverBuffer {
            buffer,
            spec,
            offset,
            bytes_per_row,
        } => {
            let held = buffer_at(*buffer)?;
            let texture = held
                .newTextureWithDescriptor_offset_bytesPerRow(
                    &descriptor(spec)?.clone(),
                    *offset as usize,
                    *bytes_per_row as usize,
                )
                .ok_or_else(|| {
                    Failure::named(
                        "NewTextureOverBuffer",
                        format!("{spec:?} at {offset} by {bytes_per_row} returned nil"),
                    )
                })?;
            Ok(Made::Texture(texture))
        }
        Resource::View {
            texture,
            format,
            slice,
            flatten,
        } => {
            let held = texture_at(*texture)?;
            let kind = if *flatten {
                MTLTextureType::Type2DMultisample
            } else {
                held.textureType()
            };
            let levels = objc2_foundation::NSRange::new(0, 1);
            let slices = objc2_foundation::NSRange::new(*slice as usize, 1);
            // SAFETY: the view's format is of the texture's own size class, which the case states,
            // and the level and slice ranges are inside the texture.
            let view = unsafe {
                held.newTextureViewWithPixelFormat_textureType_levels_slices(
                    MTLPixelFormat(*format as usize),
                    kind,
                    levels,
                    slices,
                )
            }
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureView",
                    format!("format {format} slice {slice} returned nil"),
                )
            })?;
            Ok(Made::Texture(view))
        }
    }
}

/// The interpreter's state as it runs a program.
struct Run<'a> {
    queue: &'a ProtocolObject<dyn MTLCommandQueue>,
    made: Vec<Made>,
    command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    encoder: Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>>,
    reads: Vec<Option<Vec<u8>>>,
    next_read: usize,
    statuses: Vec<String>,
}

impl Run<'_> {
    /// The buffer resource `at`.
    fn buffer(&self, at: usize) -> Answer<Retained<ProtocolObject<dyn MTLBuffer>>> {
        match self.made.get(at) {
            Some(Made::Buffer(buffer)) => Ok(buffer.clone()),
            _ => Err(Failure::named("NoSuchResource", format!("buffer {at}"))),
        }
    }

    /// The texture resource `at`.
    fn texture(&self, at: usize) -> Answer<Retained<ProtocolObject<dyn MTLTexture>>> {
        match self.made.get(at) {
            Some(Made::Texture(texture)) => Ok(texture.clone()),
            _ => Err(Failure::named("NoSuchResource", format!("texture {at}"))),
        }
    }

    /// The command buffer the steps encode into, made on first use.
    fn command_buffer(&mut self) -> Answer<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if self.command_buffer.is_none() {
            self.command_buffer = Some(
                self.queue
                    .commandBuffer()
                    .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?,
            );
        }
        Ok(self
            .command_buffer
            .clone()
            .expect("the command buffer was just made"))
    }

    /// The blit encoder the copies go into, opened on first use.
    fn encoder(&mut self) -> Answer<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>> {
        if self.encoder.is_none() {
            let command_buffer = self.command_buffer()?;
            self.encoder = Some(
                command_buffer
                    .blitCommandEncoder()
                    .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil"))?,
            );
        }
        Ok(self.encoder.clone().expect("the encoder was just opened"))
    }

    /// The slot the next read answers.
    fn slot(&mut self) -> usize {
        let slot = self.next_read;
        self.next_read += 1;
        slot
    }

    /// Run one step.
    fn step(&mut self, step: &Step) -> Answer<()> {
        match step {
            // The two fills are Apple SPI: `fillTexture:` and `fillTextureColor:` are the calls
            // that have no public form, and they are the only reason a copy-and-fill case has to
            // carry its Metal answer. The format a case names is not the reason.
            Step::FillTexture { .. } => Err(Failure::unavailable(
                "fillTexture:level:slice:region:pattern: (Apple SPI)",
            )),
            Step::FillTextureColor { .. } => Err(Failure::unavailable(
                "fillTextureColor:level:slice:region:color:packedInFormat: (Apple SPI)",
            )),
            Step::CopyBufferToTexture {
                buffer,
                offset,
                bytes_per_row,
                bytes_per_image,
                size,
                texture,
                slice,
                level,
                origin,
                plane,
            } => {
                let buffer = self.buffer(*buffer)?;
                let texture = self.texture(*texture)?;
                let encoder = self.encoder()?;
                // SAFETY: the buffer holds `bytes_per_image` bytes from `offset` for the depth the
                // size names, and the region is inside the texture's level and slice; both are the
                // case's own statement of what the probe encoded.
                unsafe {
                    encoder.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                        &buffer,
                        *offset as usize,
                        *bytes_per_row as usize,
                        *bytes_per_image as usize,
                        size_of(size),
                        &texture,
                        *slice as usize,
                        *level as usize,
                        origin_of(origin),
                        option_of(*plane),
                    );
                }
                Ok(())
            }
            Step::CopyTextureToBuffer {
                texture,
                slice,
                level,
                origin,
                size,
                buffer,
                offset,
                bytes_per_row,
                bytes_per_image,
                plane,
            } => {
                let buffer = self.buffer(*buffer)?;
                let texture = self.texture(*texture)?;
                let encoder = self.encoder()?;
                // SAFETY: the region is inside the texture's level and slice and the buffer holds
                // the bytes the pitches name from `offset`, as the case states.
                unsafe {
                    encoder.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                        &texture,
                        *slice as usize,
                        *level as usize,
                        origin_of(origin),
                        size_of(size),
                        &buffer,
                        *offset as usize,
                        *bytes_per_row as usize,
                        *bytes_per_image as usize,
                        option_of(*plane),
                    );
                }
                Ok(())
            }
            Step::CopyTextureToTexture {
                from,
                from_slice,
                from_level,
                from_origin,
                size,
                to,
                to_slice,
                to_level,
                to_origin,
            } => {
                let source = self.texture(*from)?;
                let destination = self.texture(*to)?;
                let encoder = self.encoder()?;
                // SAFETY: both regions are inside their texture's level and slice, as the case
                // states.
                unsafe {
                    encoder.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &source,
                        *from_slice as usize,
                        *from_level as usize,
                        origin_of(from_origin),
                        size_of(size),
                        &destination,
                        *to_slice as usize,
                        *to_level as usize,
                        origin_of(to_origin),
                    );
                }
                Ok(())
            }
            Step::CopyBufferToBuffer {
                from,
                from_offset,
                to,
                to_offset,
                length,
            } => {
                let source = self.buffer(*from)?;
                let destination = self.buffer(*to)?;
                let encoder = self.encoder()?;
                // SAFETY: both ranges are inside their buffer, as the case states.
                unsafe {
                    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &source,
                        *from_offset as usize,
                        &destination,
                        *to_offset as usize,
                        *length as usize,
                    );
                }
                Ok(())
            }
            Step::FillBuffer {
                buffer,
                offset,
                length,
                value,
            } => {
                let buffer = self.buffer(*buffer)?;
                let encoder = self.encoder()?;
                encoder.fillBuffer_range_value(
                    &buffer,
                    objc2_foundation::NSRange::new(*offset as usize, *length as usize),
                    *value,
                );
                Ok(())
            }
            // A render pass that clears one slice, and a per-sample read-back by a dispatch, are
            // both public Metal. Neither is built here, because every case that needs one also
            // needs a private fill and so carries its answer anyway; a case that needs one without
            // a fill is where to build it.
            Step::ClearPass { .. } => Err(Failure::unavailable(
                "a clearing render pass on the Metal side of a Work program (not built)",
            )),
            Step::ReadSamples { .. } => Err(Failure::unavailable(
                "a per-sample read-back on the Metal side of a Work program (not built)",
            )),
            Step::ReplaceRegion {
                texture,
                level,
                slice,
                region,
                bytes,
                bytes_per_row,
                bytes_per_image,
            } => {
                let texture = self.texture(*texture)?;
                let wanted = (*bytes_per_image as usize)
                    .max(*bytes_per_row as usize * region.size[1].max(1) as usize);
                if bytes.len() < wanted {
                    return Err(Failure::named(
                        "TooFewBytes",
                        format!("{} bytes for a region needing {wanted}", bytes.len()),
                    ));
                }
                // SAFETY: `bytes` holds at least the rows and images the pitches name for the
                // region, checked just above, and the region is inside the texture's level and
                // slice as the case states.
                unsafe {
                    texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                        region_of(region),
                        *level as usize,
                        *slice as usize,
                        std::ptr::NonNull::from(bytes.as_slice()).cast(),
                        *bytes_per_row as usize,
                        *bytes_per_image as usize,
                    );
                }
                Ok(())
            }
            Step::GetBytes {
                texture,
                level,
                slice,
                region,
                bytes_per_row,
                bytes_per_image,
                length,
            } => {
                let texture = self.texture(*texture)?;
                let mut bytes = vec![0u8; *length];
                // SAFETY: `bytes` is `length` bytes, which the case sized for the region's rows
                // and images at these pitches, and the caller waited for the command buffer that
                // wrote the texture.
                unsafe {
                    texture.getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice(
                        std::ptr::NonNull::from(bytes.as_mut_slice()).cast(),
                        *bytes_per_row as usize,
                        *bytes_per_image as usize,
                        region_of(region),
                        *level as usize,
                        *slice as usize,
                    );
                }
                let slot = self.slot();
                self.reads[slot] = Some(bytes);
                Ok(())
            }
            Step::ReadBuffer {
                buffer,
                offset,
                length,
            } => {
                let buffer = self.buffer(*buffer)?;
                let bytes = read_buffer(&buffer, *offset, *length)?;
                let slot = self.slot();
                self.reads[slot] = Some(bytes);
                Ok(())
            }
            Step::Commit => self.close(),
        }
    }

    /// Close the blit encoder, if one is open.
    fn end_encoding(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
    }

    /// Close the encoder, commit the command buffer, wait for it and record its status.
    fn close(&mut self) -> Answer<()> {
        self.end_encoding();
        let Some(command_buffer) = self.command_buffer.take() else {
            return Ok(());
        };
        command_buffer.commit();
        // Metal's own wait, as the probes these cases came from used. A case here encodes copies
        // alone, so there is nothing for the GPU to spin on.
        command_buffer.waitUntilCompleted();
        self.statuses
            .push(crate::metal_side::status_word(command_buffer.status()).to_owned());
        Ok(())
    }
}

/// `length` bytes of `buffer` from `offset`, as the host reads a shared buffer.
fn read_buffer(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    offset: u64,
    length: usize,
) -> Answer<Vec<u8>> {
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
    let contents = buffer.contents();
    // SAFETY: `[offset, offset + length)` is inside the buffer's bytes, checked just above; the
    // buffer is of shared storage, so the host reaches them; and the caller waited for the command
    // buffer that wrote them.
    let bytes = unsafe {
        std::slice::from_raw_parts(contents.as_ptr().cast::<u8>().add(offset as usize), length)
    };
    Ok(bytes.to_vec())
}

/// The region a rectangle names.
fn region_of(rect: &Rect) -> MTLRegion {
    MTLRegion {
        origin: origin_of(&rect.origin),
        size: size_of(&rect.size),
    }
}

/// The origin an axis triple names.
fn origin_of(origin: &[u64; 3]) -> MTLOrigin {
    MTLOrigin {
        x: origin[0] as usize,
        y: origin[1] as usize,
        z: origin[2] as usize,
    }
}

/// The size an axis triple names.
fn size_of(size: &[u64; 3]) -> MTLSize {
    MTLSize {
        width: size[0] as usize,
        height: size[1] as usize,
        depth: size[2] as usize,
    }
}

/// The blit option a plane names.
fn option_of(plane: Plane) -> MTLBlitOption {
    match plane {
        Plane::Whole => MTLBlitOption::empty(),
        Plane::Depth => MTLBlitOption::DepthFromDepthStencil,
        Plane::Stencil => MTLBlitOption::StencilFromDepthStencil,
    }
}
