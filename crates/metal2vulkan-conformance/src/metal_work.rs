//! Running a [`Work`] program on native Metal.
//!
//! This exists because a carried-only row's stated reason was wrong. `two_plane_copies` and
//! `two_plane_heap_blits` said the two-plane extended-range formats are Apple's private ones and so
//! Metal has no public call for them. **The format is not private.** `MTLTextureDescriptor`'s pixel
//! format is a raw value and Metal makes, writes, blits and reads back 550 and 551 through entirely
//! public calls; what is private is `fillTexture:`, the *fill*. Two lanes lost their Metal leg to
//! that confusion, and a case can only run on Metal if something runs the program.
//!
//! So the interpreter mirrors [`crate::runtime_work`] call for call, in the same order, with one
//! blit encoder at a time.
//!
//! # Apple SPI is not the same thing as absent
//!
//! The pattern fill was the second stated reason a copy-and-fill case had to carry its answer, and
//! it does not hold either. `fillTexture:level:slice:region:bytes:length:` has no public header, so
//! `objc2-metal` binds nothing for it, but the selector is there: the committed probe
//! `probes/surface_buffer_fills.m` calls it, and so does this interpreter, by selector, after
//! **`respondsToSelector:`** — a release that drops it answers [`Failure::unavailable`] and the
//! case reads as unrecorded rather than the process dying on an unrecognised selector. So a case
//! built on [`Step::FillTexture`] can be recorded on Metal, which is what the heap-run cases of
//! `texture_fills` are.
//!
//! The **colour** fills stay refused. `fillTexture:level:slice:region:color:` and
//! `fillTexture:level:slice:region:color:pixelFormat:` are reachable the same way, and the probe
//! checks both selectors, but no case here needs them yet and an arm nothing calls is dead code.
//! A `Fill::Color` case still carries its Metal answer; whoever needs one adds the arm beside the
//! pattern fill's.
//!
//! # Two shapes this side must never hand Metal
//!
//! Reaching the fill means reaching its two known hazards, so both are refused before anything is
//! encoded rather than left to the device
//! (`metal2vulkan_runtime::Error::EmptyFillPattern` and `LongTextureBufferFillPattern` record what
//! each does on an Apple M3 Max):
//!
//! - a **pattern of no bytes**, where plain Metal reads a unit's bytes from the pattern's address
//!   all the same — bytes the call never gave it;
//! - a **texture buffer's pattern over 32,752 bytes**, where **Metal's process ends** in the call.
//!   That is one of the shapes the round's safety rule names, and it is reachable from here the
//!   moment the arm exists.
//!
//! Nothing else here is unsafe for the GPU that the probes these cases came from were not: the same
//! shapes, through the same calls, are what produced the tables in the first place.

use crate::case::{Answer, Failure};
use crate::work::{Done, Plane, Rect, Resource, Step, Storage, TextureKind, TextureSpec, Work};
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObjectProtocol, ProtocolObject, Sel};
use objc2::sel;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBlitOption, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
    MTLCommandQueue, MTLDevice, MTLHeap, MTLHeapDescriptor, MTLHeapType, MTLOrigin, MTLPixelFormat,
    MTLRegion, MTLResourceOptions, MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
    MTLTextureType, MTLTextureUsage,
};

/// `fillTexture:level:slice:region:bytes:length:` (Apple SPI), the pattern fill. Named as a
/// function because `sel!` interns on first use.
fn fill_texture_selector() -> Sel {
    sel!(fillTexture:level:slice:region:bytes:length:)
}

/// The longest pattern a **texture buffer's** fill may take: past this Metal's process ends in the
/// call (`metal2vulkan_runtime::Error::LongTextureBufferFillPattern`, measured on an Apple M3 Max).
/// A fill of any other texture takes a pattern of any length.
const LONGEST_TEXTURE_BUFFER_PATTERN: usize = 32_752;

/// The pattern fill's two known hazards, refused before anything is encoded.
///
/// It decides from the pattern's length and the target's kind alone, so a test reaches both
/// refusals without a device — and, for the second, without ever building the shape that ends
/// Metal's process to watch the guard hold.
fn refused_pattern(pattern: usize, texture_buffer: bool) -> Option<Failure> {
    if pattern == 0 {
        return Some(Failure::named(
            "EmptyFillPattern",
            "a fill of no pattern bytes: plain Metal reads a unit's bytes from the pattern's \
             address all the same",
        ));
    }
    if texture_buffer && pattern > LONGEST_TEXTURE_BUFFER_PATTERN {
        return Some(Failure::named(
            "LongTextureBufferFillPattern",
            format!(
                "a texture buffer's fill of {pattern} pattern bytes, past the \
                 {LONGEST_TEXTURE_BUFFER_PATTERN} where Metal's process ends in the call"
            ),
        ));
    }
    None
}

/// A resource the program made on Metal, or a slot a [`Step::Release`] emptied.
enum Made {
    Buffer(Retained<ProtocolObject<dyn MTLBuffer>>),
    Heap(Retained<ProtocolObject<dyn MTLHeap>>),
    Texture(Retained<ProtocolObject<dyn MTLTexture>>),
    /// The resource a [`Step::Release`] dropped, as [`crate::runtime_work`] keeps one.
    Released,
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
        device,
        queue,
        made,
        resources: &work.resources,
        command_buffer: None,
        encoder: None,
        reads: vec![None; work.reads()],
        next_read: 0,
        statuses: Vec::new(),
    };
    // One autorelease pool per command buffer's worth of steps, and it is load-bearing rather than
    // tidy. `commandBuffer` and `blitCommandEncoder` answer **autoreleased** objects, so the
    // `Retained` this interpreter drops is a second reference and the first one outlives it. Two
    // things follow where no pool drains: the objects accumulate for the whole program, and — the
    // reason the pool is per command buffer rather than one around the program — a command buffer
    // Metal still holds keeps every resource it referred to alive, so a `Step::Release` after it
    // would free nothing and a case about what reused memory reads would quietly prove nothing.
    // The pool closes where the case's own `Step::Commit` does, which is where the program says
    // the work is done with.
    let mut at = 0;
    while at < work.steps.len() {
        let end = work.steps[at..]
            .iter()
            .position(|step| matches!(step, Step::Commit))
            .map_or(work.steps.len(), |found| at + found + 1);
        objc2::rc::autoreleasepool(|_| -> Answer<()> {
            for (index, step) in work.steps[at..end].iter().enumerate() {
                state
                    .step(step)
                    .map_err(|failure| at_step(at + index, failure))?;
            }
            // A group the case left open is committed here, as `run` has always committed what the
            // last step left open, so the pool drains with no command buffer live.
            state.close()
        })?;
        at = end;
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

/// The storage mode and resource options a [`Storage`] takes, which is
/// [`crate::metal_side::storage_mode`]: one spelling of the three modes on this side.
fn options(storage: Storage) -> (MTLStorageMode, MTLResourceOptions) {
    crate::metal_side::storage_mode(storage)
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
    let (storage, resource_options) = options(spec.storage);
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
        TextureKind::Cube => MTLTextureType::TypeCube,
        TextureKind::Type3D => MTLTextureType::Type3D,
        TextureKind::Type2DMultisample => MTLTextureType::Type2DMultisample,
        TextureKind::Type2DMultisampleArray => MTLTextureType::Type2DMultisampleArray,
        TextureKind::TextureBuffer => MTLTextureType::TypeTextureBuffer,
    });
    // SAFETY: each setter writes one field of a descriptor this call owns; nothing reads it until
    // the texture is made.
    unsafe {
        descriptor.setDepth(spec.depth.max(1) as usize);
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
    let heap_of = |size: u64, storage: Storage| -> Answer<Made> {
        let (_, resource_options) = options(storage);
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
        Resource::UnwrittenBuffer { length, storage } => {
            let (_, resource_options) = options(*storage);
            let buffer = device
                .newBufferWithLength_options((*length).max(1) as usize, resource_options)
                .ok_or_else(|| {
                    Failure::named(
                        "NewBufferWithLength",
                        format!("{length} unwritten {} bytes returned nil", storage.word()),
                    )
                })?;
            Ok(Made::Buffer(buffer))
        }
        Resource::Heap { size, storage } => heap_of(*size, *storage),
        Resource::HeapForTexture { spec, storage } => {
            let descriptor = descriptor(spec)?;
            let sized = device.heapTextureSizeAndAlignWithDescriptor(&descriptor);
            heap_of(sized.size as u64, *storage)
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
            level,
            level_count,
            slice,
            slice_count,
            flatten,
        } => {
            let held = texture_at(*texture)?;
            let kind = if *flatten {
                MTLTextureType::Type2DMultisample
            } else {
                held.textureType()
            };
            let levels = objc2_foundation::NSRange::new(*level as usize, *level_count as usize);
            let slices = objc2_foundation::NSRange::new(*slice as usize, *slice_count as usize);
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
    /// The device, which a [`Step::Make`] makes a released resource on again.
    device: &'a ProtocolObject<dyn MTLDevice>,
    queue: &'a ProtocolObject<dyn MTLCommandQueue>,
    made: Vec<Made>,
    /// The program's declarations, which a [`Step::Make`] reads to make a released resource again.
    resources: &'a [Resource],
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
            Step::FillTexture {
                texture,
                level,
                slice,
                region,
                pattern,
            } => self.fill_texture(*texture, *level, *slice, region, pattern),
            // The colour fills are reachable by selector exactly as the pattern fill is, and the
            // probe checks both, but no case here needs one and an arm nothing calls is dead code.
            // Such a case carries its Metal answer; whoever needs one adds the arm above.
            Step::FillTextureColor { .. } => Err(Failure::unavailable(
                "fillTexture:level:slice:region:color:pixelFormat: (Apple SPI, no arm built)",
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
            Step::Release { resource } => self.release(*resource),
            Step::Make { resource } => self.make_again(*resource),
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
                rows,
                bytes_per_row,
                bytes_per_image,
            } => {
                let texture = self.texture(*texture)?;
                // The most Metal reads: every image's pitch but the last, then the last image's
                // rows at their pitch. The rows are the case's, because a block-compressed
                // region's rows are its block rows and nothing here can derive them.
                let images = region.size[2].max(1) as usize;
                let wanted = *bytes_per_image as usize * (images - 1)
                    + *bytes_per_row as usize * (*rows).max(1) as usize;
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

    /// `fillTexture:level:slice:region:bytes:length:` (Apple SPI): every unit of `region` of the
    /// texture's `level` and `slice` takes `pattern`'s bytes.
    ///
    /// The selector is looked up and checked before it is sent, and both of the call's known
    /// hazards are refused first (the module docs).
    fn fill_texture(
        &mut self,
        texture: usize,
        level: u64,
        slice: u64,
        region: &Rect,
        pattern: &[u8],
    ) -> Answer<()> {
        let held = self.texture(texture)?;
        if let Some(refusal) = refused_pattern(
            pattern.len(),
            held.textureType() == MTLTextureType::TypeTextureBuffer,
        ) {
            return Err(refusal);
        }
        let encoder = self.encoder()?;
        let selector = fill_texture_selector();
        if !encoder.respondsToSelector(selector) {
            return Err(Failure::unavailable(
                "fillTexture:level:slice:region:bytes:length:",
            ));
        }
        let target: &AnyObject = encoder.as_ref();
        let bytes = pattern.as_ptr().cast::<std::ffi::c_void>();
        // SAFETY: the selector is there, checked just above, and takes the arguments the committed
        // probe `probes/surface_buffer_fills.m` declares and Metal's own
        // `methodSignatureForSelector:` confirmed. `pattern` outlives the call, which copies its
        // bytes as `fillBuffer:` does, and it is neither empty nor past the length that ends
        // Metal's process, both refused above.
        unsafe {
            let _: () = objc2::msg_send![
                target,
                fillTexture: &*held,
                level: level as usize,
                slice: slice as usize,
                region: region_of(region),
                bytes: bytes,
                length: pattern.len(),
            ];
        }
        Ok(())
    }

    /// Drop this side's reference to the resource at `at`.
    ///
    /// Refused while a command buffer is open, as [`crate::runtime_work`] refuses it, and for the
    /// same reason twice over here: Metal keeps its own reference to every resource an uncommitted
    /// command buffer refers to, so the release would free nothing.
    fn release(&mut self, at: usize) -> Answer<()> {
        if self.command_buffer.is_some() || self.encoder.is_some() {
            return Err(Failure::named(
                "ReleaseWhileEncoding",
                format!("resource {at}, with a command buffer still open"),
            ));
        }
        match self.made.get(at) {
            Some(Made::Released) | None => Err(Failure::named(
                "NoSuchResource",
                format!("resource {at} is not made"),
            )),
            // The assignment drops what the slot held, which releases this side's reference.
            Some(_) => {
                self.made[at] = Made::Released;
                Ok(())
            }
        }
    }

    /// Make the resource at `at` again, from its own declaration.
    fn make_again(&mut self, at: usize) -> Answer<()> {
        match self.made.get(at) {
            None => {
                return Err(Failure::named(
                    "NoSuchResource",
                    format!("resource {at} is not declared"),
                ))
            }
            Some(Made::Released) => {}
            Some(_) => {
                return Err(Failure::named(
                    "AlreadyMade",
                    format!("resource {at} is still made; release it first"),
                ))
            }
        }
        let declared = self
            .resources
            .get(at)
            .ok_or_else(|| Failure::named("NoSuchResource", format!("resource {at}")))?;
        let one = make(self.device, &self.made, declared)?;
        self.made[at] = one;
        Ok(())
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
        Plane::RowLinearPvrtc => MTLBlitOption::RowLinearPVRTC,
    }
}

#[cfg(test)]
mod tests {
    use super::{refused_pattern, LONGEST_TEXTURE_BUFFER_PATTERN};
    use crate::case::Failure;
    use crate::transcript::Value;

    /// The variant a refusal records as, so a test names the hazard rather than its prose.
    fn variant(refusal: &Failure) -> String {
        let Value::Variant { payload, .. } = &refusal.value else {
            panic!("a refusal records as an error variant: {:?}", refusal.value);
        };
        match payload.as_ref() {
            Value::Variant { name, .. } => name.clone(),
            other => panic!("an error carries its variant: {other:?}"),
        }
    }

    #[test]
    fn an_empty_fill_pattern_is_refused_for_every_target() {
        for texture_buffer in [false, true] {
            let refusal = refused_pattern(0, texture_buffer)
                .expect("a fill of no pattern bytes is refused before it is encoded");
            assert_eq!(variant(&refusal), "EmptyFillPattern");
        }
    }

    /// The length that ends Metal's process is a texture buffer's alone, and the refusal starts one
    /// byte past the longest pattern a texture buffer's fill is known to survive. This is the only
    /// proof of that guard there can be: the round forbids a case that ends Metal's process, so the
    /// shape itself is never built.
    #[test]
    fn a_texture_buffer_s_pattern_is_refused_one_byte_past_what_metal_survives() {
        assert!(
            refused_pattern(LONGEST_TEXTURE_BUFFER_PATTERN, true).is_none(),
            "the longest measured pattern is the longest allowed, not the first refused"
        );
        let refusal = refused_pattern(LONGEST_TEXTURE_BUFFER_PATTERN + 1, true)
            .expect("one byte more ends Metal's process, so it is refused");
        assert_eq!(variant(&refusal), "LongTextureBufferFillPattern");
        assert!(
            refused_pattern(LONGEST_TEXTURE_BUFFER_PATTERN + 1, false).is_none(),
            "the hazard is the texture buffer's; another texture's fill takes any length"
        );
    }
}
