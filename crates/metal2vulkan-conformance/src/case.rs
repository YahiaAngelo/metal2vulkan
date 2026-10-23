//! The case API: a case is written once and run by either side.
//!
//! A case names itself, sets its resources up, runs, and emits the sites it observes
//! ([`Case::run`]). It is written against [`Side`], the vocabulary both sides answer in, so porting
//! a case is a matter of saying which calls it makes and which sites it emits, and nothing about
//! which implementation is running it.
//!
//! A side with no public call for something a case reads answers [`Failure::unavailable`] rather
//! than pretending: Metal exposes an indirect command's read-back only privately, so the Metal side
//! answers `Unavailable` there and the case's Metal answer is carried from a measurement this tree
//! already holds ([`MetalSource::Carried`]). The case body stays one body.
//!
//! The vocabulary grows one family at a time as cases are ported. Nothing here reads an
//! environment variable, a device name or a vendor: a case that needs a feature finds out by making
//! the call and reading the typed refusal.

use crate::allowance::Allowance;
use crate::recording::ShaderRef;
use crate::transcript::{Node, NodeName, Observed, Sites, Transcript, Value};
use crate::verdict::device_gap;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    AttributeFormat, BindingAccess, IndexType, PixelFormat, RenderStages, SamplerAddressMode,
    SamplerBorderColor, SamplerMinMagFilter, StepFunction,
};

/// A shader as both sides can take it.
///
/// One shader, two committed texts, because the two sides read different dialects of it: the
/// runtime parses the bitcode `llvm-as` assembles from the fixture's `.ll`, and Metal's own
/// assembler reads the fixture's `.air.ll` (typed pointers), which `xcrun metal-as` reassembles and
/// `xcrun metallib` packages. A case's input fingerprint covers both, so a change to either is a
/// change to the case's inputs.
#[derive(Clone, Copy, Debug)]
pub struct ShaderSource<'a> {
    /// The AIR bitcode the runtime parses.
    pub bitcode: &'a [u8],
    /// The fixture's typed-pointer AIR text, which Metal's assembler reads.
    pub air_text: &'a str,
    /// The entry point, for a side that names it as it makes the library.
    pub entry: &'a str,
}

/// A resource a case made on a side. The number is the side's own; nothing compares it, and no
/// recording holds one, because a Metal object's identity is not observable across hosts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Handle(pub u32);

/// What a dispatch binds: the buffers at 0 up, the textures at 0 up, and the sampler states at 0
/// up.
///
/// One value rather than one argument per kind, so a case that binds a kind the vocabulary gains
/// later does not change every call site, and so a reader of a case sees the whole argument table
/// in one place.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Bindings<'a> {
    /// The buffers, bound at Metal buffer index 0 up in this order.
    pub buffers: &'a [Handle],
    /// The textures, bound at Metal texture index 0 up in this order.
    pub textures: &'a [Handle],
    /// The sampler states, bound at Metal sampler index 0 up in this order
    /// (`setSamplerState:atIndex:`).
    pub samplers: &'a [Handle],
}

impl<'a> Bindings<'a> {
    /// `buffers` at 0 up, and no texture.
    pub const fn buffers(buffers: &'a [Handle]) -> Self {
        Self {
            buffers,
            textures: &[],
            samplers: &[],
        }
    }

    /// `textures` at 0 up, and no buffer.
    pub const fn textures(textures: &'a [Handle]) -> Self {
        Self {
            buffers: &[],
            textures,
            samplers: &[],
        }
    }

    /// `textures` at 0 up, `buffers` at 0 up, and no sampler.
    pub const fn new(textures: &'a [Handle], buffers: &'a [Handle]) -> Self {
        Self {
            buffers,
            textures,
            samplers: &[],
        }
    }

    /// The bindings with `samplers` bound at sampler index 0 up.
    #[must_use]
    pub const fn sampled(mut self, samplers: &'a [Handle]) -> Self {
        self.samplers = samplers;
        self
    }
}

/// What a case's texture is for, which is the usage and storage the side makes it with.
///
/// A case states this rather than letting each side choose, because Metal refuses a depth texture
/// of shader-write usage and a sampled read of a write-only one: the role is part of what the case
/// means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureRole {
    /// `MTLTextureUsageShaderRead`: a texture a kernel reads or samples.
    Read,
    /// `MTLTextureUsageShaderWrite`: a texture a kernel writes.
    Write,
    /// Both, as [`Side::texture_2d`] makes one.
    ReadWrite,
    /// `MTLTextureUsageRenderTarget`: a texture a pass renders into, and nothing else. A tile
    /// function's implicit imageblock is its pass's attachments, so a tile case's targets carry
    /// this usage alone, as its probe made them.
    RenderTarget,
}

/// A 2D texture a case makes: `MTLTextureDescriptor`'s facts as the cases need them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureSpec {
    /// The pixel format, by its raw `MTLPixelFormat` value, so a case may name one of Apple's
    /// private formats (the two-plane extended-range family) as readily as a public one.
    pub format: PixelFormat,
    pub size: [u32; 2],
    /// Bytes a row of the bytes the case hands in and reads back. Every case states its own, since
    /// the pitch is part of what the case means.
    pub row_pitch: u32,
    pub role: TextureRole,
}

impl TextureSpec {
    /// A texture of `format`, `size` texels, at `row_pitch` bytes a row, in the role `role`.
    pub const fn new(
        format: PixelFormat,
        size: [u32; 2],
        row_pitch: u32,
        role: TextureRole,
    ) -> Self {
        Self {
            format,
            size,
            row_pitch,
            role,
        }
    }
}

/// One slot of an argument buffer a case builds ([`Side::argument_buffer`]): the object whose GPU
/// resource ID goes in that word.
///
/// The slots are what a kernel's argument-buffer struct declares, in `[[id(n)]]` order, and an
/// eight-byte resource ID each, which is how Metal 3 and the runtime both write one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgumentSlot {
    /// A texture's resource ID. A side that binds the buffer makes this texture resident.
    Texture(Handle),
    /// A sampler state's resource ID. The state must support argument buffers.
    Sampler(Handle),
}

/// A sampler state a case makes: `MTLSamplerDescriptor`'s facts a sampled case states.
///
/// The vocabulary is Metal's own enumerations as the translator's public API names them, so a value
/// a case writes reaches `MTLSamplerMinMagFilter`, `MTLSamplerAddressMode` and
/// `MTLSamplerBorderColor` by their raw ordinals and reaches the runtime's descriptor unchanged.
/// There is no second spelling to keep in step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplerSpec {
    pub min_filter: SamplerMinMagFilter,
    pub mag_filter: SamplerMinMagFilter,
    pub s_address_mode: SamplerAddressMode,
    pub t_address_mode: SamplerAddressMode,
    pub border_color: SamplerBorderColor,
    /// `normalizedCoordinates`: `false` is a pixel-coordinate sampler.
    pub normalized_coordinates: bool,
    /// `supportArgumentBuffers`: whether the state has a GPU resource ID, which is what an
    /// argument buffer needs of it ([`ArgumentSlot::Sampler`]). Metal gives a state made without
    /// it no resource ID at all.
    pub support_argument_buffers: bool,
}

impl SamplerSpec {
    /// `MTLSamplerDescriptor`'s defaults: nearest filtering without mipmaps, clamp to edge,
    /// transparent black borders, normalized coordinates.
    pub const fn new() -> Self {
        Self {
            min_filter: SamplerMinMagFilter::Nearest,
            mag_filter: SamplerMinMagFilter::Nearest,
            s_address_mode: SamplerAddressMode::ClampToEdge,
            t_address_mode: SamplerAddressMode::ClampToEdge,
            border_color: SamplerBorderColor::TransparentBlack,
            normalized_coordinates: true,
            support_argument_buffers: false,
        }
    }

    /// This state with `supportArgumentBuffers` set, so an argument buffer can name it.
    #[must_use]
    pub const fn for_argument_buffer(mut self) -> Self {
        self.support_argument_buffers = true;
        self
    }

    /// The defaults with `filter` as both the minifying and the magnifying filter.
    pub const fn filtered(filter: SamplerMinMagFilter) -> Self {
        let mut spec = Self::new();
        spec.min_filter = filter;
        spec.mag_filter = filter;
        spec
    }

    /// The sampler with `mode` on both axes.
    #[must_use]
    pub const fn addressed(mut self, mode: SamplerAddressMode) -> Self {
        self.s_address_mode = mode;
        self.t_address_mode = mode;
        self
    }

    /// The words the case's input fingerprint covers: every field, so a sampler state that moves
    /// is a change to the case's inputs.
    ///
    /// `supportArgumentBuffers` is a word only where it is set. The two forms still fingerprint
    /// apart — one carries the word and the other does not — and a case that never puts a sampler
    /// in an argument buffer fingerprints its state as it did before the field existed, so its
    /// recording stays the answer to the inputs it was taken from.
    pub fn words(&self) -> Vec<u8> {
        let mut words = vec![
            self.min_filter.as_raw(),
            self.mag_filter.as_raw(),
            self.s_address_mode.as_raw(),
            self.t_address_mode.as_raw(),
            self.border_color.as_raw(),
            u32::from(self.normalized_coordinates),
        ];
        if self.support_argument_buffers {
            words.push(1);
        }
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }
}

impl Default for SamplerSpec {
    fn default() -> Self {
        Self::new()
    }
}

/// A kernel's `[[stage_in]]` descriptor, as both sides take it: Metal's
/// `MTLStageInputOutputDescriptor`.
///
/// The vocabulary is Metal's own enumerations, as the translator's public API names them, so a
/// value a case writes reaches `MTLAttributeFormat`, `MTLStepFunction` and `MTLIndexType` by their
/// raw ordinals and reaches the runtime's descriptor unchanged. There is no second spelling to
/// keep in step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageInput {
    /// The attributes a kernel reads, by their Metal attribute index.
    pub attributes: Vec<StageInAttribute>,
    /// The buffer layouts the attributes read, by their Metal buffer index.
    pub layouts: Vec<StageInLayout>,
    /// The index type an indexed step reads.
    pub index_type: IndexType,
    /// The Metal buffer index an indexed step reads its indices from.
    pub index_buffer_index: u32,
}

impl StageInput {
    /// A descriptor of `attributes` and `layouts`, whose indices are 32-bit words at
    /// `index_buffer_index`.
    pub fn new(
        attributes: Vec<StageInAttribute>,
        layouts: Vec<StageInLayout>,
        index_buffer_index: u32,
    ) -> Self {
        Self {
            attributes,
            layouts,
            index_type: IndexType::UInt32,
            index_buffer_index,
        }
    }
}

/// One `[[stage_in]]` attribute: Metal's `MTLAttributeDescriptor` at an attribute index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageInAttribute {
    /// The Metal attribute index, as the shader declares it.
    pub index: u32,
    pub format: AttributeFormat,
    /// The attribute's byte offset inside an element.
    pub offset: u32,
    /// The Metal buffer index of the layout it reads.
    pub buffer_index: u32,
}

/// One `[[stage_in]]` buffer layout: Metal's `MTLBufferLayoutDescriptor` at a buffer index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageInLayout {
    /// The Metal buffer index the layout is of.
    pub buffer_index: u32,
    /// Bytes between elements; `None` is `MTLBufferLayoutStrideDynamic`, the stride the encoder
    /// binds the buffer with.
    pub stride: Option<u32>,
    pub step_function: StepFunction,
    /// The step rate; Metal requires 0 of a constant step.
    pub step_rate: u32,
}

/// The stage-in region a dispatch fetches its `[[stage_in]]` attributes from: the two calls Metal
/// sets it with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageInRegion {
    /// `setStageInRegion:`.
    Direct { origin: [u32; 3], size: [u32; 3] },
    /// `setStageInRegionWithIndirectBuffer:indirectBufferOffset:`, whose buffer holds the origin's
    /// three words and then the size's.
    Indirect { buffer: Handle, offset: u64 },
}

/// What a side answered a call with.
pub type Answer<T> = Result<T, Failure>;

/// A call that did not answer: the typed error as the transcript records it, and whether it is the
/// venue's honest gap rather than ours.
///
/// # Why the `Err` is not stored
///
/// The comparison reads a **refusal** out of one shape and one only: Metal's site answered and
/// ours is an `Err` variant ([`crate::compare`]). So a failure whose recorded value is anything
/// else is a refusal the suite cannot see — it reads as two sides observing different kinds of
/// value, which is a divergence, which the known-gaps ledger cannot carry as a refusal and the
/// refused count never sees.
///
/// That had happened twice, in two files, for the same reason: a helper that **named where** a
/// failure came from wrapped a fields value around the whole `Err`. So the `Err` layer is not a
/// field here at all. A failure carries the typed error alone, [`Failure::recorded`] adds the
/// `Err` as the site is written, and the annotation ([`Failure::at_step`]) can only reach the
/// inside of it. Nothing a failure carries, now or later, can displace the layer the comparison
/// reads.
#[derive(Clone, Debug, PartialEq)]
pub struct Failure {
    /// The typed error **inside** the call's `Err`: a `Value::Variant` of the error's variant
    /// path and its payload. Private, so the `Err` layer cannot be lost: see the type's own
    /// documentation.
    error: Value,
    /// What the venue lacks, where the typed refusal names capabilities a device could enable or
    /// a measured driver defect ([`device_gap`]). `None` where the refusal is ours to answer for.
    pub lacks: Option<String>,
}

impl Failure {
    /// The failure a typed runtime error is.
    pub fn of(error: &metal2vulkan_runtime::Error) -> Self {
        Self {
            error: Value::Variant {
                name: error_variant(error),
                payload: Box::new(Value::Lines(vec![error.to_string()])),
            },
            lacks: device_gap(error),
        }
    }

    /// The failure a side with no public call for `call` answers. It is neither a divergence nor a
    /// refusal: it says the side cannot be asked, and a case whose Metal side answers it carries
    /// its Metal answer instead.
    pub fn unavailable(call: &str) -> Self {
        Self::of_variant(
            "Unavailable",
            Value::fields([("call", Value::Lines(vec![call.to_owned()]))]),
        )
    }

    /// A failure of `variant` with `detail`, for a side whose call failed outside the runtime's
    /// typed errors.
    pub fn named(variant: &str, detail: impl Into<String>) -> Self {
        Self::of_variant(variant, Value::Lines(vec![detail.into()]))
    }

    /// A failure of the typed variant `variant` carrying `payload`.
    fn of_variant(variant: &str, payload: Value) -> Self {
        Self {
            error: Value::Variant {
                name: variant.to_owned(),
                payload: Box::new(payload),
            },
            lacks: None,
        }
    }

    /// What the transcript records at a site the call did not answer: `Err` of the typed error.
    ///
    /// The only way out of a failure into a value, so every site a failure reaches holds the shape
    /// the comparison reads a refusal from.
    pub fn recorded(&self) -> Value {
        Value::err(self.error.clone())
    }

    /// The same failure, said to have happened at step `index` of a [`crate::work::Work`]
    /// program.
    #[must_use]
    pub fn at_step(self, index: usize) -> Self {
        self.inside("step", index)
    }

    /// The same failure, said to have happened while making resource `index` of one.
    #[must_use]
    pub fn at_resource(self, index: usize) -> Self {
        self.inside("resource", index)
    }

    /// [`Failure::at_step`] and [`Failure::at_resource`], which differ only in the field they
    /// name.
    ///
    /// One implementation for both sides of the harness. Each interpreter used to carry its own
    /// copy, with a comment saying the two must agree — and the shape they agreed on was the one
    /// that hid the refusal, so the agreement proved nothing. A program's part is named **inside**
    /// the typed error, never around it.
    fn inside(self, part: &'static str, index: usize) -> Self {
        // Both fields are named, with no `..`, so the next field anyone adds to a failure is a
        // compile error here rather than a field silently dropped as a failure is annotated.
        Self {
            error: Value::fields([(part, Value::Uint(index as u64)), ("refused", self.error)]),
            lacks: self.lacks,
        }
    }
}

/// The variant path a typed runtime error records as: the error's outermost variant, which is what
/// a consumer matches on.
fn error_variant(error: &metal2vulkan_runtime::Error) -> String {
    let debug = format!("{error:?}");
    let head = debug
        .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        .find(|word| !word.is_empty())
        .unwrap_or("Error");
    format!("Error::{head}")
}

/// Where a case's run stopped: a call it needed did not answer. An abort is not a verdict; the
/// comparison reads the sites emitted up to it, and the refused call is one of them.
#[derive(Clone, Debug, PartialEq)]
pub struct Abort {
    /// The site the call was emitted at.
    pub at: String,
    pub failure: Failure,
}

/// The venue a side runs on: what a result carries so a report can say where it ran. A case never
/// branches on any of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Venue {
    /// The physical device's name.
    pub device: String,
    /// The driver, as the device reports it.
    pub driver: String,
    /// The API version the device runs.
    pub api: String,
    /// The Metal level the side answers as ([`metal2vulkan_runtime::MetalLevel::macos`]).
    pub metal_level: String,
}

impl Venue {
    /// The short label the report's matrix prints as a column.
    pub fn label(&self) -> String {
        self.device.clone()
    }
}

impl std::fmt::Display for Venue {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            out,
            "{} ({}, API {}, Metal level {})",
            self.device, self.driver, self.api, self.metal_level
        )
    }
}

/// The vocabulary a case is written against.
///
/// Every call answers [`Answer`], so a case records what it got and carries on or stops. A side
/// that has no public call for one answers [`Failure::unavailable`].
pub trait Side {
    /// Where the side runs.
    fn venue(&self) -> Venue;

    /// A library of the shader's AIR, from whichever of its forms the side reads
    /// ([`ShaderSource`]).
    fn library(&mut self, source: &ShaderSource<'_>) -> Answer<Handle>;

    /// `newComputePipelineStateWithFunction:error:` of the library's `entry`.
    fn compute_pipeline(&mut self, library: Handle, entry: &str) -> Answer<Handle>;

    /// `staticThreadgroupMemoryLength` of a compute pipeline state: the bytes the kernel's own
    /// `threadgroup` arrays take.
    ///
    /// This getter and not its two neighbours. `maxTotalThreadsPerThreadgroup` and
    /// `threadExecutionWidth` answer the **device's** limits for the kernel — Metal's 1024 and 32
    /// on Apple's GPUs, lower where a device's shared memory or its emulated simdgroups leave room
    /// for fewer — so a venue that answers less is not diverging from Metal, it is a venue with a
    /// smaller device, and the suite has no way to say that of a number (only a refused call
    /// carries what the venue lacks). A case pins the device's limit by **dispatching at Metal's
    /// own threadgroup size** instead, which such a venue refuses typed. The static threadgroup
    /// length is a fact about the shader, so it is the same wherever the shader is compiled, and
    /// comparing it is exact.
    fn static_threadgroup_memory_length(&mut self, pipeline: Handle) -> Answer<u64> {
        let _ = pipeline;
        Err(Failure::unavailable("static_threadgroup_memory_length"))
    }

    /// `newBufferWithBytes:length:options:` of shared storage.
    fn buffer(&mut self, bytes: &[u8]) -> Answer<Handle>;

    /// `newBufferWithLength:options:` of shared storage, zeroed.
    fn zeroed_buffer(&mut self, length: usize) -> Answer<Handle>;

    /// `newBufferWithBytesNoCopy:length:options:deallocator:` over pages **the side maps**, every
    /// byte the spec's pattern ([`crate::host_pages::HostBufferSpec`]).
    ///
    /// This is the one buffer in the suite that is not the device's own memory, and it is here for
    /// one reason: Metal's private `MTLResourceOptions` bit 24 is the only way a Metal program can
    /// say "the GPU's writes to this memory go nowhere", and whether we drop such a write exactly
    /// as Metal drops it is the owner's standing rule. A spec with `read_only` off is the control
    /// that makes the read-only run a measurement rather than a guess about whether the write path
    /// ran at all.
    ///
    /// The pages stay mapped for as long as the buffer, and the side never writes through them
    /// after the buffer is made: a CPU write through a read-only mapping ends the process, which is
    /// Metal's own behaviour and not something this suite asks about.
    /// [`Side::read_buffer`] of the handle is then the **CPU's** view of the pages; a copy the GPU
    /// makes out of the buffer into an ordinary one ([`crate::render::RenderWork::CopyBuffer`]) is
    /// the **GPU's**, and the two differing is the whole answer.
    fn host_buffer(&mut self, spec: &crate::host_pages::HostBufferSpec) -> Answer<Handle> {
        let _ = spec;
        Err(Failure::unavailable("newBufferWithBytesNoCopy"))
    }

    /// A 2D texture of `format`, `size` texels, one level, shader-read and shader-write, of shared
    /// storage, whose level 0 is then `bytes` at `row_pitch` bytes a row
    /// (`replaceRegion:mipmapLevel:withBytes:bytesPerRow:`).
    ///
    /// Every case states its own row pitch, because that pitch is part of what the case means and
    /// a side that computed its own could disagree with the other silently.
    fn texture_2d(
        &mut self,
        format: PixelFormat,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<Handle>;

    /// `getBytes:bytesPerRow:fromRegion:mipmapLevel:` of a texture's whole level 0, at `row_pitch`
    /// bytes a row.
    fn read_texture(&mut self, texture: Handle, size: [u32; 2], row_pitch: u32) -> Answer<Vec<u8>>;

    /// A 2D texture the case describes whole ([`TextureSpec`]): one level, of shared storage, in
    /// the usage its role names, whose level 0 is then `bytes` at the spec's row pitch.
    ///
    /// [`Side::texture_2d`] is the read-and-write case of this; a case that samples a texture, or
    /// writes one of a format Metal holds no readable-and-writable texture of, states its role
    /// here instead. An empty `bytes` makes the texture and writes nothing into it, which is what a
    /// case that asks what a **fresh** texture reads needs.
    fn texture_2d_in_role(&mut self, spec: &TextureSpec, bytes: &[u8]) -> Answer<Handle> {
        let _ = (spec, bytes);
        Err(Failure::unavailable("texture_2d_in_role"))
    }

    /// [`Side::read_texture`] into a buffer that already holds `before`, so the bytes the copy does
    /// **not** write stay visible in the answer.
    ///
    /// A case reading a texture at a row pitch wider than its rows asks two questions at once:
    /// what the rows hold, and whether the copy left the bytes between them alone. A read into
    /// zeroed memory can only answer the first — zero is indistinguishable from a copy that wrote
    /// zero — so the case states the bytes that were there and the answer carries them back. The
    /// bytes outside the rows then compare under [`Gaps::Compared`](crate::transcript::Gaps).
    ///
    /// `before` is the whole destination, `row_pitch * size[1]` bytes.
    fn read_texture_over(
        &mut self,
        texture: Handle,
        size: [u32; 2],
        row_pitch: u32,
        before: &[u8],
    ) -> Answer<Vec<u8>> {
        let _ = (texture, size, row_pitch, before);
        Err(Failure::unavailable("read_texture_over"))
    }

    /// `newRenderPipelineStateWithDescriptor:options:reflection:error:` of `library`'s `vertex` and
    /// `fragment` functions, over one color attachment of `attachment`, asking for binding info.
    ///
    /// The reflection is asked for at the creation and kept with the state, because that is the
    /// only time Metal answers it: `MTLRenderPipelineState`'s own `reflection` property is nil for
    /// a state an `MTLDevice` made.
    fn render_pipeline_with_reflection(
        &mut self,
        library: Handle,
        vertex: &str,
        fragment: &str,
        attachment: PixelFormat,
    ) -> Answer<Handle> {
        let _ = (library, vertex, fragment, attachment);
        Err(Failure::unavailable("render_pipeline_with_reflection"))
    }

    /// How the state's function of `stage` accesses the texture at each of `indices`: the access
    /// where a used texture argument covers the slot, and `None` where none does or where the
    /// argument that covers it is never referenced.
    ///
    /// `stage` is one stage, not a union. Metal states this per stage — the reflection holds a
    /// binding list a stage — so one bit has a Metal answer and a union has only a composition
    /// rule of ours, which belongs in a lane rather than in a transcript. A side answers
    /// [`Failure::unavailable`] for anything but a single stage.
    fn texture_accesses(
        &mut self,
        pipeline: Handle,
        stage: RenderStages,
        indices: &[u64],
    ) -> Answer<Vec<Option<BindingAccess>>> {
        let _ = (pipeline, stage, indices);
        Err(Failure::unavailable("texture_accesses"))
    }

    /// `replaceRegion:mipmapLevel:withBytes:bytesPerRow:` of a texture's whole level 0, at
    /// `row_pitch` bytes a row.
    ///
    /// A case that writes a texture twice in a stated order needs this: making it with its bytes
    /// cannot say "the kernel wrote the alpha plane first, then the host wrote the colour words",
    /// and that order is part of what such a case means.
    fn replace_region(
        &mut self,
        texture: Handle,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<()> {
        let _ = (texture, size, row_pitch, bytes);
        Err(Failure::unavailable("replace_region"))
    }

    /// `newTextureWithDescriptor:` of the texture the case describes whole
    /// ([`crate::texture::TextureRequest`]), loaded with the request's `replaceRegion:` calls in
    /// order.
    ///
    /// [`Side::texture_2d_in_role`] is the one-level shared 2D case of this. A case that needs a
    /// level count, an array's slices, a cube's faces, private storage,
    /// `allowGPUOptimizedContents`, a swizzle, or a block-compressed format whose rows are blocks,
    /// states the whole descriptor here instead.
    fn texture(&mut self, request: &crate::texture::TextureRequest<'_>) -> Answer<Handle> {
        let _ = request;
        Err(Failure::unavailable("texture"))
    }

    /// `newSamplerStateWithDescriptor:` of the state the case describes ([`SamplerSpec`]).
    fn sampler(&mut self, spec: &SamplerSpec) -> Answer<Handle> {
        let _ = spec;
        Err(Failure::unavailable("sampler"))
    }

    /// A buffer of the GPU resource IDs of `slots`, in order: the argument buffer a kernel reads
    /// its texture and its sampler through, as Metal 3 writes one.
    ///
    /// The case states the slots and nothing about residency: a side that binds this buffer makes
    /// the resources it names resident for the dispatch (`useResource:usage:`), because Metal
    /// requires that of a resource a kernel reaches through an argument buffer rather than through
    /// a slot. Binding it is binding a buffer, so it goes in [`Bindings::buffers`] at the index the
    /// kernel declares.
    ///
    /// A sampler a case puts here must be one [`SamplerSpec::support_argument_buffers`] is set on:
    /// Metal has no resource ID for a sampler state made without it.
    fn argument_buffer(&mut self, slots: &[ArgumentSlot]) -> Answer<Handle> {
        let _ = slots;
        Err(Failure::unavailable("argument_buffer"))
    }

    /// `newFunctionWithName:` of `entry`, then `newArgumentEncoderWithBufferIndex:` of
    /// `buffer_index`: the encoder of the argument buffer that function takes at that buffer
    /// index, as its own AIR lays it out.
    ///
    /// The encoder has its own handle space: an `MTLArgumentEncoder` is its own Metal object, not
    /// another name for a library, a pipeline or a buffer. No argument buffer is set on it yet, so
    /// a case states that step itself ([`Side::set_argument_buffer`]).
    fn argument_encoder(
        &mut self,
        library: Handle,
        entry: &str,
        buffer_index: u32,
    ) -> Answer<Handle> {
        let _ = (library, entry, buffer_index);
        Err(Failure::unavailable("argument_encoder"))
    }

    /// `newArgumentEncoderWithArguments:` of the descriptors `arguments` states: the encoder of a
    /// layout a program describes itself rather than taking from a function.
    ///
    /// It joins the same handle space as [`Side::argument_encoder`]: one encoder kind, made from
    /// two descriptions.
    fn argument_encoder_with_arguments(
        &mut self,
        arguments: &[crate::argument_encoders::ArgumentDescriptorSpec],
    ) -> Answer<Handle> {
        let _ = arguments;
        Err(Failure::unavailable("argument_encoder_with_arguments"))
    }

    /// `setLabel:` of `label`, then what the encoder answers of itself
    /// ([`EncoderFacts`](crate::argument_encoders::EncoderFacts)): `label`, `encodedLength`,
    /// `alignment` and whether `device` is the case's own.
    ///
    /// The label is set in the same call as the getters are read, because what a case means by it
    /// is a round trip: a label Metal keeps is a label Metal answers.
    fn argument_encoder_facts(
        &mut self,
        encoder: Handle,
        label: &str,
    ) -> Answer<crate::argument_encoders::EncoderFacts> {
        let _ = (encoder, label);
        Err(Failure::unavailable("argument_encoder_facts"))
    }

    /// `setArgumentBuffer:offset:` of `buffer` at `offset`, or
    /// `setArgumentBuffer:startOffset:arrayElement:` where `element` names one, which puts the
    /// arguments at `offset + element * encodedLength`.
    ///
    /// One call for the two selectors, because they are the same step of a case: where in a buffer
    /// this encoder's arguments go.
    fn set_argument_buffer(
        &mut self,
        encoder: Handle,
        buffer: Handle,
        offset: u64,
        element: Option<u64>,
    ) -> Answer<()> {
        let _ = (encoder, buffer, offset, element);
        Err(Failure::unavailable("set_argument_buffer"))
    }

    /// One write through an argument encoder
    /// ([`ArgumentWrite`](crate::argument_encoders::ArgumentWrite)).
    ///
    /// A side that binds the encoded buffer for a dispatch makes the textures the write names
    /// resident (`useResource:usage:`), because Metal requires that of a resource a function
    /// reaches through an argument buffer rather than through a slot.
    fn encode_argument(
        &mut self,
        encoder: Handle,
        write: &crate::argument_encoders::ArgumentWrite,
    ) -> Answer<()> {
        let _ = (encoder, write);
        Err(Failure::unavailable("encode_argument"))
    }

    /// `newComputePipelineStateWithFunction:options:reflection:error:` of the library's `entry`,
    /// asking for binding information, and every argument the reflection answers
    /// ([`crate::function_bindings`]).
    ///
    /// One call, because what it observes is a **tree**: each binding, the subprotocol its own
    /// `type` says it conforms to, and for an argument buffer the struct type whose members are the
    /// layout contract. A case cannot state that as independent calls, because which getter to ask
    /// of a binding is decided by what an earlier getter answered.
    fn compute_pipeline_bindings(
        &mut self,
        library: Handle,
        entry: &str,
    ) -> Answer<Vec<crate::function_bindings::BindingFacts>> {
        let _ = (library, entry);
        Err(Failure::unavailable("compute_pipeline_bindings"))
    }

    /// `newComputePipelineStateWithDescriptor:options:reflection:error:` of the descriptor
    /// `spec` describes whole ([`ComputeStateSpec`](crate::pipeline_facts::ComputeStateSpec)).
    ///
    /// The state joins the same handle space as [`Side::compute_pipeline`]: one Metal object kind
    /// made from a function or from a descriptor.
    fn compute_pipeline_with_descriptor(
        &mut self,
        library: Handle,
        spec: &crate::pipeline_facts::ComputeStateSpec<'_>,
    ) -> Answer<Handle> {
        let _ = (library, spec);
        Err(Failure::unavailable("compute_pipeline_with_descriptor"))
    }

    /// What a compute pipeline state answers of itself
    /// ([`ComputeStateFacts`](crate::pipeline_facts::ComputeStateFacts)), with
    /// `imageblockMemoryLengthForDimensions:` asked for `imageblock`.
    ///
    /// The dimensions are the case's own rather than fixed, because the answer is a function of
    /// them: a case states the extent whose length it means.
    fn compute_pipeline_facts(
        &mut self,
        state: Handle,
        imageblock: [u32; 3],
    ) -> Answer<crate::pipeline_facts::ComputeStateFacts> {
        let _ = (state, imageblock);
        Err(Failure::unavailable("compute_pipeline_facts"))
    }

    /// What a render pipeline state answers of itself
    /// ([`RenderStateFacts`](crate::pipeline_facts::RenderStateFacts)), with
    /// `imageblockMemoryLengthForDimensions:` asked for `imageblock`.
    ///
    /// [`Side::tile_facts`] is three of these getters read off a **tile** state, which is the
    /// other descriptor the same Metal object is made from. This is the drawing state's whole
    /// portable surface, and the two are kept apart because what each answers is decided by which
    /// descriptor made it.
    fn render_pipeline_facts(
        &mut self,
        state: Handle,
        imageblock: [u32; 3],
    ) -> Answer<crate::pipeline_facts::RenderStateFacts> {
        let _ = (state, imageblock);
        Err(Failure::unavailable("render_pipeline_facts"))
    }

    /// What a render pipeline **descriptor** answers of itself
    /// ([`RenderDescriptorFacts`](crate::render_descriptors::RenderDescriptorFacts)): the whole of
    /// `MTLRenderPipelineDescriptor`'s own surface, in the shape
    /// [`spec`](crate::render_descriptors::DescriptorSpec) states.
    ///
    /// One call, for the reason [`Side::function_facts`] gives: the descriptor is mutable and the
    /// observation is what it answers **after** a stated sequence of setters, a `reset` among them,
    /// so a case cannot state it as independent calls over an object neither side's handle space
    /// holds. No pipeline is made from the descriptor, which is what keeps the family off the GPU.
    ///
    /// [`Side::render_pipeline_facts`] is the **state**'s side of the same descriptor: what the
    /// made object echoes back. This is the descriptor before any state exists, which is where
    /// Metal's defaults are.
    fn render_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &crate::render_descriptors::DescriptorSpec<'_>,
    ) -> Answer<crate::render_descriptors::RenderDescriptorFacts> {
        let _ = (library, spec);
        Err(Failure::unavailable("render_descriptor_facts"))
    }

    /// `newFunctionWithName:` of `entry`, or `newFunctionWithName:constantValues:error:` where
    /// `constants` holds a value, and what the function then answers about **itself**
    /// ([`crate::function_facts::FunctionFacts`]).
    ///
    /// One call, because what it observes is a **tree** whose shape one getter decides: which of
    /// `vertexAttributes` and `stageInputAttributes` answers an array at all is a fact about the
    /// function type, and each element is an object with six getters of its own. A case cannot
    /// state that as independent calls.
    ///
    /// The function is made and read inside the call rather than kept in a pool of its own: every
    /// fact here is the function's, no case holds one past reading it, and the two sides then need
    /// no handle space for an object whose identity nothing compares.
    ///
    /// `constants` is the whole of `MTLFunctionConstantValues`' own surface, as three setters
    /// ([`ConstantAt`](crate::function_facts::ConstantAt)). A side whose constant values have no
    /// range form expands the range into one value an index, which is what the range setter means.
    fn function_facts(
        &mut self,
        library: Handle,
        entry: &str,
        constants: &[crate::function_facts::ConstantValue],
    ) -> Answer<crate::function_facts::FunctionFacts> {
        let _ = (library, entry, constants);
        Err(Failure::unavailable("function_facts"))
    }

    /// Where `constantDataAtIndex:` of `index` points, as a byte offset inside the encoder's
    /// arguments.
    ///
    /// The call answers a CPU address, which is the host's own number and belongs in no
    /// recording. Its offset from the arguments' start is the layout fact the address stands for,
    /// and it is exactly comparable.
    fn constant_data_offset(&mut self, encoder: Handle, index: u32) -> Answer<u64> {
        let _ = (encoder, index);
        Err(Failure::unavailable("constant_data_offset"))
    }

    /// One command buffer with one compute encoder: the pipeline set, `bindings` bound at 0 up per
    /// kind, one `dispatchThreadgroups:threadsPerThreadgroup:`, and `commit`.
    fn dispatch(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        threadgroups: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle>;

    /// `waitUntilCompleted` and then `status`, as a variant of the status's name.
    fn wait(&mut self, command_buffer: Handle) -> Answer<Value>;

    /// `contents` of a shared buffer: `length` bytes from `offset`.
    fn read_buffer(&mut self, buffer: Handle, offset: u64, length: usize) -> Answer<Vec<u8>>;

    /// The degrade notices the side has announced so far, by code, and taken away. Metal announces
    /// none, so its side answers an empty list.
    fn announced(&mut self) -> Vec<String>;

    /// `newComputePipelineStateWithDescriptor:options:reflection:error:` of the library's `entry`,
    /// under the stage-input descriptor `stage_input` ([`StageInput`]).
    fn compute_pipeline_with_stage_input(
        &mut self,
        library: Handle,
        entry: &str,
        stage_input: &StageInput,
    ) -> Answer<Handle>;

    /// One command buffer with one compute encoder: the pipeline set, `bindings` bound at 0 up per
    /// kind, each of `stage_in` set in the order the case lists them, one
    /// `dispatchThreads:threadsPerThreadgroup:` of `grid` threads, and `commit`.
    ///
    /// `stage_in` is a list rather than one region because setting it twice is a case of its own:
    /// Metal's later call replaces the earlier, and a case states that by listing both.
    fn dispatch_threads(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        stage_in: &[StageInRegion],
        grid: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle>;

    /// **One** command buffer with **one** compute encoder holding every command of `pass` in
    /// order, then `endEncoding`, `commit` and a bounded wait
    /// ([`ComputePass`](crate::compute_pass::ComputePass)).
    ///
    /// [`dispatch`](Side::dispatch) and [`dispatch_threads`](Side::dispatch_threads) are a fixed
    /// pass — pipeline, bindings at 0 up, one dispatch, commit — and four of
    /// `MTLComputeCommandEncoder`'s items are reachable that way. A case that states its commands
    /// reaches the rest: the inline bytes, the plural setters, the LOD-clamped sampler bindings,
    /// the dynamic threadgroup allocation, the residency declarations, the barriers, the
    /// threadgroup-grid dispatch and the indirect one.
    ///
    /// Every command is encoded whatever an earlier one answered, as
    /// [`mesh_pass`](Side::mesh_pass) and [`tile_pass`](Side::tile_pass) do, and the answer names
    /// which one refused. **The side makes nothing resident of its own**: a case that reaches a
    /// resource other than through a bound slot says so with a residency command, which is both
    /// what Metal requires and how those items are exercised.
    fn compute_pass(
        &mut self,
        pass: &crate::compute_pass::ComputePass<'_>,
    ) -> Answer<crate::compute_pass::ComputePassOutcome> {
        let _ = pass;
        Err(Failure::unavailable("compute_pass"))
    }

    /// The read-back of an indirect render command the CPU encoded, as the side's getters answer
    /// it: `getCommandType` and the draw. Metal exposes these privately only, so the Metal side
    /// answers [`Failure::unavailable`] and such a case's Metal answer is carried.
    ///
    /// `primitive_type` is the raw `MTLPrimitiveType` value, so a case can set one of Metal's
    /// private types; `indexed` chooses the indexed setter.
    fn indirect_draw_read_back(
        &mut self,
        primitive_type: u64,
        indexed: bool,
    ) -> Answer<IndirectReadBack> {
        let _ = (primitive_type, indexed);
        Err(Failure::unavailable("indirect_draw_read_back"))
    }

    /// Run a copy-and-fill program ([`crate::work`]) and answer what it read.
    ///
    /// The Metal side runs a program through public `objc2-metal` calls and refuses the two fills
    /// alone, because `fillTexture:` and `fillTextureColor:` are Apple SPI. The pixel format a
    /// case names is never the reason: it is a raw value, and Metal handles Apple's two-plane
    /// extended-range formats publicly. So a program that fills carries its Metal answer; a
    /// program of copies records on Metal. A side that runs no program at all answers
    /// [`Failure::unavailable`], which is this default.
    fn work(&mut self, work: &crate::work::Work) -> Answer<crate::work::Done> {
        let _ = work;
        Err(Failure::unavailable("work"))
    }

    /// Run one full-screen fragment fill ([`FragmentFill`]) and answer the pass's status and the
    /// target's bytes.
    ///
    /// Metal's whole path here is public, so both sides run it: `newRenderPipelineStateWithDescriptor:
    /// error:` of the two functions over one color attachment of the case's format, a private
    /// render target of the case's extent, one render pass that loads nothing and stores, the
    /// fragment buffers set at 0 and up, one `drawPrimitives:`, then a blit of level 0 into a
    /// shared buffer the host reads.
    fn fragment_fill(&mut self, fill: &FragmentFill<'_>) -> Answer<FragmentFilled> {
        let _ = fill;
        Err(Failure::unavailable("fragment_fill"))
    }

    /// `newRenderPipelineStateWithMeshDescriptor:options:reflection:error:` of the library's
    /// functions, over `functions.attachments` color attachments of [`TARGET_FORMAT`].
    fn mesh_pipeline(&mut self, library: Handle, functions: &MeshFunctions<'_>) -> Answer<Handle> {
        let _ = (library, functions);
        Err(Failure::unavailable("mesh_pipeline"))
    }

    /// `newTextureWithDescriptor:` of a 2D [`TARGET_FORMAT`] render target of `width` by
    /// `height`, whose bytes the host reads.
    ///
    /// The target joins the same handle space as [`texture_2d`](Side::texture_2d), so
    /// [`read_texture`](Side::read_texture) reads it back and a texture has one pool however it
    /// was made. Only the descriptor differs: a target is a render target of
    /// [`TARGET_FORMAT`], not a shader-read-and-write texture of the case's format.
    fn render_target(&mut self, width: u32, height: u32) -> Answer<Handle> {
        let _ = (width, height);
        Err(Failure::unavailable("render_target"))
    }

    /// One command buffer with one render pass over `targets`, each cleared to zero and stored,
    /// whose commands are `steps` in order, then `endEncoding`, `commit` and a bounded wait.
    ///
    /// Every step is encoded, whatever an earlier one answered, because that is what the probe
    /// does: a refused draw leaves the pass and its other commands as they were, and the answer
    /// names which step refused ([`PassOutcome::refused`]).
    fn mesh_pass(&mut self, targets: &[Handle], steps: &[MeshStep]) -> Answer<PassOutcome> {
        let _ = (targets, steps);
        Err(Failure::unavailable("mesh_pass"))
    }

    /// `newRenderPipelineStateWithMeshDescriptor:` as [`mesh_pipeline`] does, with
    /// `supportIndirectCommandBuffers` set, so an indirect command may set the state.
    ///
    /// [`mesh_pipeline`]: Side::mesh_pipeline
    fn mesh_pipeline_for_commands(
        &mut self,
        library: Handle,
        functions: &MeshFunctions<'_>,
    ) -> Answer<Handle> {
        let _ = (library, functions);
        Err(Failure::unavailable("mesh_pipeline_for_commands"))
    }

    /// `newIndirectCommandBufferWithDescriptor:maxCommandCount:options:` of `commands` commands
    /// of both mesh draw kinds, binding its own pipeline state, one object buffer and one mesh
    /// buffer, then every command of `commands` encoded by the host in order.
    fn mesh_command_buffer(&mut self, commands: &[MeshCommand]) -> Answer<Handle> {
        let _ = commands;
        Err(Failure::unavailable("mesh_command_buffer"))
    }

    /// `newComputePipelineStateWithDescriptor:options:reflection:error:` of the library's `entry`
    /// with `supportIndirectCommandBuffers` set, so an indirect compute command may set the
    /// state, under the stage-input descriptor `stage_input` where the case names one.
    ///
    /// It is a call of its own rather than a flag on
    /// [`compute_pipeline`](Side::compute_pipeline), because the flag is a property of the state
    /// Metal checks at the setter rather than at the dispatch: a state made without it refuses to
    /// be set on a command, and a case that executes commands means the state it names to be one
    /// a command may hold.
    fn compute_pipeline_for_commands(
        &mut self,
        library: Handle,
        entry: &str,
        stage_input: Option<&StageInput>,
    ) -> Answer<Handle> {
        let _ = (library, entry, stage_input);
        Err(Failure::unavailable("compute_pipeline_for_commands"))
    }

    /// `newIndirectCommandBufferWithDescriptor:maxCommandCount:options:` of the descriptor the
    /// case states whole ([`IcbDescriptor`]), of shared storage and of as many commands as
    /// `commands` holds, and then every command of `commands` encoded by the host in order
    /// ([`EncodedCommand`]).
    ///
    /// Every step of every command is encoded, whatever an earlier one answered, as the pass
    /// vocabularies do: a refused setter leaves the rest of the encoding as it was, and the answer
    /// names which step refused ([`CommandsMade::refused`]).
    ///
    /// [`IcbDescriptor`]: crate::indirect_commands::IcbDescriptor
    /// [`EncodedCommand`]: crate::indirect_commands::EncodedCommand
    /// [`CommandsMade::refused`]: crate::indirect_commands::CommandsMade::refused
    fn indirect_command_buffer(
        &mut self,
        descriptor: &crate::indirect_commands::IcbDescriptor,
        commands: &[crate::indirect_commands::EncodedCommand],
    ) -> Answer<crate::indirect_commands::CommandsMade> {
        let _ = (descriptor, commands);
        Err(Failure::unavailable("indirect_command_buffer"))
    }

    /// `size` of an indirect command buffer: the commands it holds.
    fn indirect_commands_size(&mut self, commands: Handle) -> Answer<u64> {
        let _ = commands;
        Err(Failure::unavailable("indirect_commands_size"))
    }

    /// `resetWithRange:` of `count` commands from `first`, which puts them back as a fresh
    /// buffer's are.
    fn reset_indirect_commands(&mut self, commands: Handle, first: u64, count: u64) -> Answer<()> {
        let _ = (commands, first, count);
        Err(Failure::unavailable("reset_indirect_commands"))
    }

    /// One command buffer with one render pass over `targets`, each cleared to zero and stored,
    /// which makes `used` resident for the object, mesh and fragment stages, executes every
    /// command of `commands`, then `endEncoding`, `commit` and a bounded wait.
    fn execute_mesh_commands(
        &mut self,
        targets: &[Handle],
        commands: Handle,
        used: &[Handle],
    ) -> Answer<PassOutcome> {
        let _ = (targets, commands, used);
        Err(Failure::unavailable("execute_mesh_commands"))
    }

    /// A texture over `buffer`: over one plane of the surface the buffer stands for, or over the
    /// buffer itself ([`PlaneTexture`]).
    ///
    /// The buffer stands for the surface with its planes where the surface's lie, which is how the
    /// probe wrote these cases: a host without CoreVideo then runs them unchanged, and the bytes a
    /// fill wrote are the buffer's own.
    fn plane_texture(&mut self, buffer: Handle, over: &PlaneTexture) -> Answer<Handle> {
        let _ = (buffer, over);
        Err(Failure::unavailable("plane_texture"))
    }

    /// One command buffer with one blit encoder holding one fill of `texture` ([`Fill`]),
    /// committed and waited for.
    ///
    /// Metal's texture fills are private selectors, so Metal's own side answers
    /// [`Failure::unavailable`] here and a case that uses this carries its Metal answer.
    fn fill_texture(&mut self, texture: Handle, fill: &Fill) -> Answer<Value> {
        let _ = (texture, fill);
        Err(Failure::unavailable("fill_texture"))
    }

    /// `newRasterizationRateMapWithDescriptor:` of `request`.
    ///
    /// Nothing about a rate map reaches the GPU until a render pass takes one, and no case here
    /// opens such a pass, so the whole family is a CPU-side call: the descriptor is compiled, the
    /// map answers its queries, and its parameter data is copied by the CPU.
    fn rate_map(&mut self, request: &RateMapRequest<'_>) -> Answer<Handle> {
        let _ = request;
        Err(Failure::unavailable("rate_map"))
    }

    /// What the map answers: its label, screen size, granularity, layer count, parameter size and
    /// alignment, and `physicalSizeForLayer:` of each index in `layers`.
    fn rate_map_facts(&mut self, map: Handle, layers: &[u64]) -> Answer<RateMapFacts> {
        let _ = (map, layers);
        Err(Failure::unavailable("rate_map_facts"))
    }

    /// `copyParameterDataToBuffer:offset:` into the buffer `into` at `offset`. The bytes are read
    /// back with [`Side::read_buffer`], so the copy itself observes nothing of its own.
    fn copy_rate_map_parameters(&mut self, map: Handle, into: Handle, offset: u64) -> Answer<()> {
        let _ = (map, into, offset);
        Err(Failure::unavailable("copy_rate_map_parameters"))
    }

    /// `newRenderPipelineStateWithTileDescriptor:options:reflection:error:` of the library's tile
    /// function, over the color attachments `spec` names ([`TilePipelineSpec`]).
    ///
    /// The state joins the same handle space as [`mesh_pipeline`](Side::mesh_pipeline): a tile
    /// state and a mesh state are one Metal object kind (`MTLRenderPipelineState`), made from two
    /// descriptors.
    fn tile_pipeline(&mut self, library: Handle, spec: &TilePipelineSpec<'_>) -> Answer<Handle> {
        let _ = (library, spec);
        Err(Failure::unavailable("tile_pipeline"))
    }

    /// What a tile render pipeline state answers ([`TileFacts`]).
    fn tile_facts(&mut self, state: Handle) -> Answer<TileFacts> {
        let _ = state;
        Err(Failure::unavailable("tile_facts"))
    }

    /// One command buffer with one render pass over `targets` in `tile`-sized tiles, each
    /// attachment **loaded and stored**, whose commands are `steps` in order, then `endEncoding`,
    /// `commit` and a bounded wait.
    ///
    /// Loaded, not cleared, because a tile function reads the cells it writes: a case states what
    /// the attachment held by writing it before the pass, and the pass's answer is what the tile
    /// function left there.
    ///
    /// Every step is encoded, whatever an earlier one answered, as [`Side::mesh_pass`] does, and
    /// the answer names which step refused ([`PassOutcome::refused`]).
    fn tile_pass(
        &mut self,
        targets: &[Handle],
        tile: [u32; 2],
        steps: &[TileStep],
    ) -> Answer<PassOutcome> {
        let _ = (targets, tile, steps);
        Err(Failure::unavailable("tile_pass"))
    }

    /// `heapTextureSizeAndAlignWithDescriptor:` of the descriptor `query` names
    /// ([`crate::texture_query`]).
    ///
    /// A pure getter: it compiles nothing, allocates nothing and reaches no GPU, so a case over it
    /// is safe on every device and its answer can be re-measured on the oracle as often as anybody
    /// asks.
    fn heap_texture_size_and_align(
        &mut self,
        query: &crate::texture_query::HeapTextureQuery,
    ) -> Answer<crate::texture_query::SizeAndAlign> {
        let _ = query;
        Err(Failure::unavailable("heap_texture_size_and_align"))
    }

    /// `heapBufferSizeAndAlignWithLength:options:` of the length and the raw `MTLResourceOptions`
    /// bits `query` names ([`crate::device_queries::HeapBufferQuery`]).
    ///
    /// A pure getter, as [`Side::heap_texture_size_and_align`] is: arithmetic over a length, with
    /// nothing allocated and no heap made.
    fn heap_buffer_size_and_align(
        &mut self,
        query: &crate::device_queries::HeapBufferQuery,
    ) -> Answer<crate::texture_query::SizeAndAlign> {
        let _ = query;
        Err(Failure::unavailable("heap_buffer_size_and_align"))
    }

    /// `supportsVertexAmplificationCount:` of `count`.
    ///
    /// A device query whose answer is a **model**: the runtime answers yes up to a constant its own
    /// documentation attributes to a Metal probe on this chip and on an M4, so it is the same on
    /// every device that enables multiview and is a contract with Metal rather than a reading of
    /// the device ([`crate::device_queries`] says which queries pass that test and which do not).
    fn supports_vertex_amplification_count(&mut self, count: u64) -> Answer<bool> {
        let _ = count;
        Err(Failure::unavailable("supports_vertex_amplification_count"))
    }

    /// `supportsRasterizationRateMapWithLayerCount:` of `layers`.
    ///
    /// A model as [`Side::supports_vertex_amplification_count`] is: the runtime answers yes for the
    /// layer counts Apple's map compile was measured with and reproduces on every device.
    fn supports_rasterization_rate_map_layers(&mut self, layers: u64) -> Answer<bool> {
        let _ = layers;
        Err(Failure::unavailable(
            "supports_rasterization_rate_map_layers",
        ))
    }

    /// `minimumLinearTextureAlignmentForPixelFormat:` of `format`.
    ///
    /// A model as the two above are: the runtime answers one alignment for every format a buffer
    /// can back, on every device, and refuses the formats Metal's own query asserts on. A case asks
    /// only formats the runtime answers a number for, which is what keeps Metal's assertion out of
    /// reach ([`crate::device_queries`]).
    fn minimum_linear_texture_alignment(
        &mut self,
        format: metal2vulkan::metal::PixelFormat,
    ) -> Answer<u64> {
        let _ = format;
        Err(Failure::unavailable("minimum_linear_texture_alignment"))
    }

    /// `convertSparsePixelRegions:toTileRegions:withTileSize:alignmentMode:numRegions:` of the
    /// regions, tile size and alignment mode `query` names, as the tile boxes it wrote.
    ///
    /// Arithmetic over the case's own numbers: nothing is allocated, no texture is sparse and no
    /// device memory is bound. Metal's one `numRegions` is the slices' single length, so the case
    /// hands one list and reads one back.
    fn convert_sparse_pixel_regions(
        &mut self,
        query: &crate::device_queries::RegionQuery,
    ) -> Answer<Vec<crate::device_queries::Box6>> {
        let _ = query;
        Err(Failure::unavailable("convert_sparse_pixel_regions"))
    }

    /// `convertSparseTileRegions:toPixelRegions:withTileSize:numRegions:` of the regions and tile
    /// size `query` names, as the texel boxes it wrote. The mode is not read: this direction takes
    /// none.
    fn convert_sparse_tile_regions(
        &mut self,
        query: &crate::device_queries::RegionQuery,
    ) -> Answer<Vec<crate::device_queries::Box6>> {
        let _ = query;
        Err(Failure::unavailable("convert_sparse_tile_regions"))
    }

    /// Whether `newTextureWithDescriptor:` of the descriptor `query` names answers a texture.
    ///
    /// The answer is a **fact about the descriptor**, not about the object: Metal answers nil for a
    /// descriptor it describes no texture of, and the runtime answers a typed `Nil` error for the
    /// same descriptor, which is the same fact in the two APIs' spellings. So both sides answer
    /// `false` there, and a side that refuses the call for any other reason answers its typed
    /// error, which compares as a refusal.
    ///
    /// A case asks this only of descriptors Metal makes no texture of, so nothing is allocated.
    fn makes_texture(&mut self, query: &crate::texture_query::HeapTextureQuery) -> Answer<bool> {
        let _ = query;
        Err(Failure::unavailable("makes_texture"))
    }

    /// `newTextureWithDescriptor:offset:bytesPerRow:` on a buffer of the request's own length and
    /// storage mode, and what the made texture answers of itself
    /// ([`crate::buffer_texture`]).
    ///
    /// The whole shape is one call because what it observes is a **rule**: whether Metal lays a
    /// texture of this descriptor over a buffer at that offset and that row pitch at all. A case
    /// cannot state that as a resource plus a step, because a resource that cannot be made stops a
    /// program, and this family's subject is exactly the shapes that cannot.
    ///
    /// It allocates no texels: the texture's memory **is** the buffer's, and the buffer is the
    /// request's own length. Nothing is encoded and nothing reaches the GPU.
    ///
    /// A request whose row pitch is below the texture's own row ends Metal's process, so the Metal
    /// side refuses it before making the call
    /// ([`crate::buffer_texture::ends_metals_process`]) and a case of that shape says
    /// [`MetalSource::did_not_return`].
    fn buffer_texture(
        &mut self,
        request: &crate::buffer_texture::BufferTextureRequest,
    ) -> Answer<crate::buffer_texture::BufferTextureFacts> {
        let _ = request;
        Err(Failure::unavailable("buffer_texture"))
    }

    /// `sparseTileSizeWithTextureType:pixelFormat:sampleCount:`, or its `sparsePageSize:` form
    /// where the query names a page ([`crate::sparse_query`]).
    ///
    /// A pure getter, like [`heap_texture_size_and_align`](Side::heap_texture_size_and_align): it
    /// compiles nothing, allocates nothing, reaches no GPU and asks nothing of the device's sparse
    /// capabilities, so a case over it is safe on every device and answers on a venue that binds no
    /// sparse memory at all.
    fn sparse_tile_size(
        &mut self,
        query: &crate::sparse_query::SparseTileQuery,
    ) -> Answer<crate::sparse_query::TileSize> {
        let _ = query;
        Err(Failure::unavailable("sparse_tile_size"))
    }

    /// `sparseTileSizeInBytes`, or `sparseTileSizeInBytesForSparsePageSize:` where `page` names
    /// one: the bytes one sparse tile of that page takes.
    ///
    /// A pure getter, as [`sparse_tile_size`](Side::sparse_tile_size) is.
    fn sparse_tile_size_in_bytes(&mut self, page: crate::sparse_query::QueryPage) -> Answer<u64> {
        let _ = page;
        Err(Failure::unavailable("sparse_tile_size_in_bytes"))
    }

    /// `newHeapWithDescriptor:` of a `MTLHeapTypeSparse` heap the case describes
    /// ([`crate::sparse_query::SparseHeapSpec`]).
    ///
    /// The heap holds no memory of its own until a mapping takes a tile, so making one allocates
    /// nothing a GPU can fault on and encodes nothing. A device that cannot make one refuses typed,
    /// which is how a case finds out rather than by reading a device name.
    fn sparse_heap(&mut self, spec: &crate::sparse_query::SparseHeapSpec) -> Answer<Handle> {
        let _ = spec;
        Err(Failure::unavailable("sparse_heap"))
    }

    /// What a heap answers of itself ([`crate::sparse_query::HeapFacts`]).
    fn heap_facts(&mut self, heap: Handle) -> Answer<crate::sparse_query::HeapFacts> {
        let _ = heap;
        Err(Failure::unavailable("heap_facts"))
    }

    /// `newTextureWithDescriptor:` on a heap, of the texture the case describes
    /// ([`crate::sparse_query::SparseTextureSpec`]).
    ///
    /// The texture joins the same handle space as [`texture_2d`](Side::texture_2d), so a texture
    /// has one pool however it was made. Nothing is written into it: a case that asks what a
    /// **fresh** sparse texture answers needs it untouched, and a case that writes one says so with
    /// a command of its own.
    fn heap_texture(
        &mut self,
        heap: Handle,
        spec: &crate::sparse_query::SparseTextureSpec,
    ) -> Answer<Handle> {
        let _ = (heap, spec);
        Err(Failure::unavailable("heap_texture"))
    }

    /// What a sparse texture answers of its tiles
    /// ([`crate::sparse_query::SparseTextureFacts`]).
    fn sparse_texture_facts(
        &mut self,
        texture: Handle,
    ) -> Answer<crate::sparse_query::SparseTextureFacts> {
        let _ = texture;
        Err(Failure::unavailable("sparse_texture_facts"))
    }

    /// A command queue's limit on uncompleted command buffers, by the probe both sides run
    /// ([`crate::queue_slots`]): the queue is made with `max` as its
    /// `maxCommandBufferCount` (`newCommandQueue` where `max` is `None`), and at most `cap` command
    /// buffers are made on a thread of its own.
    ///
    /// The whole experiment is one call because what it observes is an **order of events** — a make
    /// that did not return, another queue's make that did, and the first one answering once a slot
    /// came back — and a case cannot state that as a sequence of independent calls.
    ///
    /// Nothing is committed, so nothing reaches the GPU, and no make can be left unanswered: the
    /// probe releases a blocked make by dropping a command buffer held uncommitted.
    fn queue_slots(
        &mut self,
        max: Option<u32>,
        cap: u32,
    ) -> Answer<crate::queue_slots::QueueSlots> {
        let _ = (max, cap);
        Err(Failure::unavailable("queue_slots"))
    }

    /// `newTextureWithDescriptor:` of a private render target the case describes whole
    /// ([`TargetSpec`]).
    ///
    /// It joins the same handle space as [`texture_2d`](Side::texture_2d), so a kernel may bind it
    /// and [`RenderWork::ReadTexture`] may read it back: a texture has one pool however it was
    /// made.
    ///
    /// [`TargetSpec`]: crate::render::TargetSpec
    /// [`RenderWork::ReadTexture`]: crate::render::RenderWork::ReadTexture
    fn target(&mut self, spec: &crate::render::TargetSpec) -> Answer<Handle> {
        let _ = spec;
        Err(Failure::unavailable("target"))
    }

    /// `newRenderPipelineStateWithDescriptor:error:` of a vertex-and-fragment pipeline the case
    /// describes whole ([`RenderPipelineSpec`]).
    ///
    /// [`RenderPipelineSpec`]: crate::render::RenderPipelineSpec
    fn render_pipeline(
        &mut self,
        library: Handle,
        spec: &crate::render::RenderPipelineSpec<'_>,
    ) -> Answer<Handle> {
        let _ = (library, spec);
        Err(Failure::unavailable("render_pipeline"))
    }

    /// `newDepthStencilStateWithDescriptor:` of the state `spec` names.
    ///
    /// The descriptor is `MTLDepthStencilDescriptor` as the translator's public API already names
    /// it, so a compare function and a stencil operation have one spelling across the two sides.
    /// Its own handle space: `MTLDepthStencilState` is its own Metal object, not another name for
    /// one of the pools above.
    fn depth_stencil_state(&mut self, spec: &DepthStencilDescriptor) -> Answer<Handle> {
        let _ = spec;
        Err(Failure::unavailable("depth_stencil_state"))
    }

    /// **One** command buffer holding every piece of `work` in order, `commit`, and a bounded
    /// wait ([`RenderWork`]).
    ///
    /// Every piece is encoded whatever an earlier one answered, as the probes did: a refused draw
    /// leaves the pass and its other commands as they were, and the answer names which piece
    /// refused ([`PassOutcome::refused`], one entry per piece of `work`).
    ///
    /// [`RenderWork`]: crate::render::RenderWork
    /// The host's write of every byte of the plane a target lies over
    /// ([`crate::surface::PlaneSpec`]): `IOSurfaceLock`, a copy of `bytes`, `IOSurfaceUnlock` on
    /// the Metal side, and the stand-in buffer's own bytes on the runtime's.
    ///
    /// It is its own call rather than [`replace_region`](Side::replace_region) because the plane is
    /// addressed as **the surface's bytes**, stride and padding included, not as the texture's
    /// texels: a case asserts that the bytes between the rows stayed as they were, which a region
    /// write cannot say.
    fn write_plane(&mut self, target: Handle, bytes: &[u8]) -> Answer<()> {
        let _ = (target, bytes);
        Err(Failure::unavailable("write_plane"))
    }

    /// Every byte of the plane a target lies over, read back by the host after the GPU is done
    /// with it: the whole stride of every row, so the bytes between the rows are observed too.
    fn read_plane(&mut self, target: Handle) -> Answer<Vec<u8>> {
        let _ = target;
        Err(Failure::unavailable("read_plane"))
    }

    fn render(&mut self, work: &[crate::render::RenderWork]) -> Answer<PassOutcome> {
        let _ = work;
        Err(Failure::unavailable("render"))
    }

    /// What a wait made from a command buffer's own handler, for that same command buffer,
    /// answered ([`crate::handler_waits`]).
    ///
    /// The whole experiment is one call because the fact it observes is only visible from inside a
    /// handler: a command buffer is made with one small fill, the handler `wait.from` names makes
    /// the wait `wait.wait` names on it, and the answer comes back on a channel. The answer is a
    /// [`Value`]: `Ok` where the wait returned, the typed error where it did not.
    ///
    /// **Metal has no side here, deliberately.** Metal does not come back from this call — the
    /// probe that measured it ended with it — so [`crate::metal_side::MetalSide`] leaves this
    /// default in place and must go on leaving it. A case over it says
    /// [`MetalSource::did_not_return`] and record mode refuses the row before anything is encoded.
    ///
    /// Our own leg cannot hang: the runtime refuses the wait at once, so the handler returns and
    /// the command buffer completes, and the side waits for it with a bound.
    fn handler_wait(&mut self, wait: crate::handler_waits::OwnWait) -> Answer<Value> {
        let _ = wait;
        Err(Failure::unavailable("handler_wait"))
    }

    /// `detachBacking` (Metal's private `MTLBuffer` method) of the no-copy buffer `buffer`, called
    /// from a completed handler of a command buffer whose blit fill named it, `after` into that
    /// handler: the whole program in one call — the queue, the command buffer, the fill, the
    /// handler, the commit and this thread's own bounded wait — and what the detach answered
    /// inside the handler ([`crate::host_detach`]).
    ///
    /// One call rather than several, because the fact is a **relation** between a handler and the
    /// work a command buffer still holds, and a case cannot state "from inside the handler" as
    /// calls whose order nothing fixes.
    ///
    /// `after` is what makes the answer comparable: Metal's release of what the command buffer
    /// used races the start of its handlers, so the first microseconds of a handler have no
    /// answer, and this call asks in the region past the race.
    ///
    /// Bounded on both legs, and it holds no GPU: the handler sleeps `after` and returns, and a
    /// command buffer that does not arrive inside the wait is recorded as the negative fact.
    fn detach_in_a_completed_handler(
        &mut self,
        buffer: Handle,
        after: std::time::Duration,
    ) -> Answer<crate::host_detach::DetachedInHandler> {
        let _ = (buffer, after);
        Err(Failure::unavailable("detachBacking"))
    }

    /// What a shared event's value did, by the rising-value probe ([`crate::shared_events`]):
    /// `newSharedEvent`, its label, the host's sets, two GPU signals, three host waits, and a
    /// signal of `u64::MAX`.
    ///
    /// One call rather than a dozen, because the facts are a **sequence** — a value that did not
    /// fall when a lower one was set — and a case cannot state that as independent calls whose
    /// order nothing fixes.
    /// `newSharedEventHandle` of a shared event, then `newSharedEventWithHandle:` of what it
    /// answered, and what the two events then say about themselves
    /// ([`EventHandleRoundTrip`](crate::shared_events::EventHandleRoundTrip)).
    ///
    /// One call, because the fact under test is a **relation** between two objects: that the event
    /// the device opens from the handle is the one the handle came from, which only a value set on
    /// one and read on the other can say. Two independent calls over a handle space would be two
    /// objects whose identity nothing compares.
    ///
    /// Nothing here reaches the GPU: every value is set and read by the host, and both waits are
    /// bounded.
    fn shared_event_handle_round_trip(
        &mut self,
        label: &str,
    ) -> Answer<crate::shared_events::EventHandleRoundTrip> {
        let _ = label;
        Err(Failure::unavailable("shared_event_handle_round_trip"))
    }

    fn shared_event_rises(&mut self, label: &str) -> Answer<crate::shared_events::EventRise> {
        let _ = label;
        Err(Failure::unavailable("shared_event_rises"))
    }

    /// What the cross-queue wait probe observed ([`crate::event_orders::CrossQueueWait`]): one
    /// queue's command buffer waits for an event the other's signals, and the waiting one is
    /// committed first.
    ///
    /// One call rather than six, because what it observes is an **order**: a status read between
    /// two commits, and then the bytes both command buffers wrote. A case cannot state "this status
    /// was read while the other command buffer had not been committed yet" as independent calls.
    fn event_cross_queue_wait(&mut self) -> Answer<crate::event_orders::CrossQueueWait> {
        Err(Failure::unavailable("event_cross_queue_wait"))
    }

    /// `signaledValue` of one shared event two queues' command buffers signalled with `values`, in
    /// that submission order, once both completed.
    ///
    /// One call, because the fact is about the pair: whichever signal executed last, the value a
    /// lower signal cannot lower is what the event holds.
    fn event_highest_signal(&mut self, values: [u64; 2]) -> Answer<u64> {
        let _ = values;
        Err(Failure::unavailable("event_highest_signal"))
    }

    /// What the held-command-buffer probe observed
    /// ([`crate::event_orders::HeldCommandBuffer`]): a command buffer held by a shared event the
    /// host raises later, one committed behind it on its own queue, and one on another queue.
    ///
    /// One call, because every fact is an order — which command buffer went on while another was
    /// held, and which handler ran before which.
    fn held_command_buffer(&mut self) -> Answer<crate::event_orders::HeldCommandBuffer> {
        Err(Failure::unavailable("held_command_buffer"))
    }

    /// What the signal-before-a-wait probe observed
    /// ([`crate::event_orders::SignalBeforeWait`]): one command buffer that fills, signals, waits
    /// for a gate, fills again and signals again.
    ///
    /// One call, because the fact is that the first signal landed **while** the wait was still
    /// pending, which is a relation between two moments of one command buffer's life.
    fn signal_before_a_wait(&mut self) -> Answer<crate::event_orders::SignalBeforeWait> {
        Err(Failure::unavailable("signal_before_a_wait"))
    }

    /// What the scheduled-handler order probe observed
    /// ([`crate::scheduled_handlers::ScheduledHandlerOrder`]): a command buffer whose scheduled
    /// handler is held inside itself while the probe reads the status and a wait for the scheduling
    /// on another thread.
    ///
    /// One call, because the fact is a relation between a handler, a status and a wait on a third
    /// thread, and holding the handler is the only way any of it is observable at all.
    fn scheduled_handler_order(
        &mut self,
    ) -> Answer<crate::scheduled_handlers::ScheduledHandlerOrder> {
        Err(Failure::unavailable("scheduled_handler_order"))
    }
}

/// A tile render pipeline a case makes: `MTLTileRenderPipelineDescriptor`'s facts the cases state.
///
/// The attachment formats are the case's own rather than one fixed format, because a tile
/// function's implicit imageblock **is** the pass's attachments: their formats decide the
/// imageblock's layout, and so what the function reads and writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TilePipelineSpec<'a> {
    /// The tile function, by its name in the library.
    pub entry: &'a str,
    /// The color attachment formats, at attachment index 0 up in this order.
    pub attachments: &'a [PixelFormat],
    /// `maxTotalThreadsPerThreadgroup`: `None` is Metal's 0, the pipeline's own maximum.
    pub max_total_threads_per_threadgroup: Option<u32>,
    /// `threadgroupSizeMatchesTileSize`.
    pub threadgroup_size_matches_tile_size: bool,
}

impl<'a> TilePipelineSpec<'a> {
    /// The tile function `entry` over `attachments`, with Metal's defaults for the rest.
    pub const fn new(entry: &'a str, attachments: &'a [PixelFormat]) -> Self {
        Self {
            entry,
            attachments,
            max_total_threads_per_threadgroup: None,
            threadgroup_size_matches_tile_size: false,
        }
    }

    /// The descriptor with `maxTotalThreadsPerThreadgroup` set to `limit`.
    #[must_use]
    pub const fn limited(mut self, limit: u32) -> Self {
        self.max_total_threads_per_threadgroup = Some(limit);
        self
    }

    /// The descriptor with `threadgroupSizeMatchesTileSize` set.
    #[must_use]
    pub const fn tile_sized(mut self) -> Self {
        self.threadgroup_size_matches_tile_size = true;
        self
    }

    /// The words the case's input fingerprint covers: every field of the descriptor, so a tile
    /// pipeline that moves is a change to the case's inputs.
    pub fn words(&self) -> Vec<u8> {
        let mut out = self.entry.as_bytes().to_vec();
        for format in self.attachments {
            out.extend(format.as_raw().to_le_bytes());
        }
        out.extend(
            self.max_total_threads_per_threadgroup
                .unwrap_or(0)
                .to_le_bytes(),
        );
        out.push(u8::from(self.threadgroup_size_matches_tile_size));
        out
    }
}

/// What a tile render pipeline state answers. Each field is a Metal getter's answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileFacts {
    /// `maxTotalThreadsPerThreadgroup`: Metal's 1024 where the device runs a threadgroup of 1024
    /// threads, lowered by the descriptor's own limit to the largest threadgroup of even sides
    /// within it.
    pub max_total_threads_per_threadgroup: u64,
    /// `threadgroupSizeMatchesTileSize`, which the state echoes from its descriptor.
    pub threadgroup_size_matches_tile_size: bool,
    /// `imageblockSampleLength`: the attachments' texels packed. `None` where the side answered a
    /// typed error for it rather than a length.
    pub imageblock_sample_length: Option<u64>,
}

impl TileFacts {
    /// The facts as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        Node::branch(name, Vec::new())
            .with(Node::leaf(
                "max_total_threads_per_threadgroup",
                Observed::value(Value::Uint(self.max_total_threads_per_threadgroup)),
            ))
            .with(Node::leaf(
                "threadgroup_size_matches_tile_size",
                Observed::value(Value::Bool(self.threadgroup_size_matches_tile_size)),
            ))
            .with(Node::leaf(
                "imageblock_sample_length",
                Observed::value(match self.imageblock_sample_length {
                    Some(length) => Value::Variant {
                        name: "Some".to_owned(),
                        payload: Box::new(Value::Uint(length)),
                    },
                    None => Value::unit_variant("None"),
                }),
            ))
    }
}

/// One command a case encodes into a tile render pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileStep {
    /// `setRenderPipelineState:` of a tile state.
    Pipeline(Handle),
    /// `setTileBuffer:offset:atIndex:`.
    TileBuffer {
        buffer: Handle,
        offset: u64,
        index: u32,
    },
    /// `setTileBytes:length:atIndex:`.
    TileBytes { bytes: Vec<u8>, index: u32 },
    /// `setTileTexture:atIndex:`.
    TileTexture { texture: Handle, index: u32 },
    /// `dispatchThreadsPerTile:`.
    DispatchThreadsPerTile([u32; 3]),
}

impl TileStep {
    /// The site name the step's answer is emitted at: the call it makes, so a refusal reads as the
    /// Metal call we refused.
    pub fn call(&self) -> &'static str {
        match self {
            TileStep::Pipeline(_) => "setRenderPipelineState",
            TileStep::TileBuffer { .. } => "setTileBuffer",
            TileStep::TileBytes { .. } => "setTileBytes",
            TileStep::TileTexture { .. } => "setTileTexture",
            TileStep::DispatchThreadsPerTile(_) => "dispatchThreadsPerTile",
        }
    }
}

/// One layer of a rasterization-rate map's descriptor
/// (`MTLRasterizationRateLayerDescriptor`), as its quality samples along each axis.
///
/// Both axes hold at least one sample, because Metal's
/// `initWithSampleCount:horizontal:vertical:` answers nil for an axis of none (verified while
/// porting this family). A descriptor of **no layers** is the shape whose `layerCount` is 0, and a
/// case says that by holding no layer at all rather than by a layer with an empty axis.
#[derive(Clone, Debug, PartialEq)]
pub struct RateMapLayer {
    pub horizontal: Vec<f32>,
    pub vertical: Vec<f32>,
}

impl RateMapLayer {
    /// A layer of one quality sample on each axis.
    pub fn uniform(horizontal: f32, vertical: f32) -> Self {
        Self {
            horizontal: vec![horizontal],
            vertical: vec![vertical],
        }
    }
}

/// A rasterization-rate map's descriptor, as both sides take it.
#[derive(Clone, Copy, Debug)]
pub struct RateMapRequest<'a> {
    /// `screenSize`'s width and height in screen pixels.
    pub screen: [u32; 2],
    /// `layers`, from layer 0.
    pub layers: &'a [RateMapLayer],
    /// `label`, which the map keeps.
    pub label: Option<&'a str>,
}

/// What a rasterization-rate map answers. Each field is a Metal getter's answer, and a size's
/// depth is kept because Metal's is 1 where its own documentation says 0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RateMapFacts {
    pub label: Option<String>,
    pub screen_size: [u64; 3],
    pub physical_granularity: [u64; 3],
    pub layer_count: u64,
    /// `physicalSizeForLayer:` of each index the case asked about, in that order.
    pub physical_sizes: Vec<[u64; 3]>,
    pub parameter_size: u64,
    pub parameter_align: u64,
}

impl RateMapFacts {
    /// The facts as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let size = |axes: &[u64; 3]| {
            Value::fields([
                ("width", Value::Uint(axes[0])),
                ("height", Value::Uint(axes[1])),
                ("depth", Value::Uint(axes[2])),
            ])
        };
        let label = match &self.label {
            Some(label) => Value::Variant {
                name: "Some".to_owned(),
                payload: Box::new(Value::Lines(vec![label.clone()])),
            },
            None => Value::unit_variant("None"),
        };
        let mut node = Node::branch(name, Vec::new())
            .with(Node::leaf("label", Observed::value(label)))
            .with(Node::leaf(
                "screen_size",
                Observed::value(size(&self.screen_size)),
            ))
            .with(Node::leaf(
                "physical_granularity",
                Observed::value(size(&self.physical_granularity)),
            ))
            .with(Node::leaf(
                "layer_count",
                Observed::value(Value::Uint(self.layer_count)),
            ))
            .with(Node::leaf(
                "parameter_size",
                Observed::value(Value::Uint(self.parameter_size)),
            ))
            .with(Node::leaf(
                "parameter_align",
                Observed::value(Value::Uint(self.parameter_align)),
            ));
        for (at, axes) in self.physical_sizes.iter().enumerate() {
            node = node.with(Node::leaf(
                format!("physical_size_for_layer/{at}"),
                Observed::value(size(axes)),
            ));
        }
        node
    }
}

/// One render command the host encodes into an indirect command buffer: the command's own pipeline
/// state, the buffers it binds, and its one mesh draw.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshCommand {
    /// The state the command sets, from [`Side::mesh_pipeline_for_commands`].
    pub pipeline: Handle,
    /// The object buffers the command binds, each at its index.
    pub object_buffers: Vec<(Handle, u32)>,
    /// The mesh buffers the command binds, each at its index.
    pub mesh_buffers: Vec<(Handle, u32)>,
    /// The command's draw: `drawMeshThreadgroups` of a grid, or `drawMeshThreads` of threads.
    pub draw: CommandDraw,
}

/// An indirect render command's mesh draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandDraw {
    /// `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
    Threadgroups {
        groups: [u32; 3],
        object: [u32; 3],
        mesh: [u32; 3],
    },
    /// `drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
    Threads {
        threads: [u32; 3],
        object: [u32; 3],
        mesh: [u32; 3],
    },
}

/// A texture a case makes over a buffer: over one plane of the surface the buffer stands for, or
/// over the buffer itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaneTexture {
    /// The surface's four-character OSType, `None` for a texture over the buffer itself.
    pub ostype: Option<[u8; 4]>,
    /// The texture's raw `MTLPixelFormat`.
    pub format: u32,
    /// The texture's extent in texels.
    pub size: [u32; 2],
    /// The plane of the surface the texture covers.
    pub plane: u32,
    /// Each of the surface's two planes: its first byte in the buffer and its `bytesPerRow`. A
    /// texture over the buffer itself uses the first alone.
    pub planes: [(u64, u64); 2],
}

/// One fill of a texture, as the probe encoded it.
#[derive(Clone, Debug, PartialEq)]
pub enum Fill {
    /// `fillTexture:level:slice:region:pattern:` (Apple SPI) of these pattern bytes.
    Pattern {
        level: u32,
        slice: u32,
        /// The region as `x`, `y`, `width`, `height`.
        region: [u32; 4],
        pattern: Vec<u8>,
    },
    /// `fillTexture:level:slice:region:color:pixelFormat:` (Apple SPI) of this color packed in
    /// this raw `MTLPixelFormat`.
    Color {
        level: u32,
        slice: u32,
        region: [u32; 4],
        /// The color's four channels.
        color: [f64; 4],
        /// The raw `MTLPixelFormat` the color packs in.
        packed_into: u32,
    },
}

impl Fill {
    /// The site name the fill's answer is emitted at: the call it makes.
    pub fn call(&self) -> &'static str {
        match self {
            Fill::Pattern { .. } => "fillTexture",
            Fill::Color { .. } => "fillTextureColor",
        }
    }

    /// The words the fill's parameters fingerprint as.
    pub fn words(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Fill::Pattern {
                level,
                slice,
                region,
                pattern,
            } => {
                out.extend(b"pattern");
                out.extend(level.to_le_bytes());
                out.extend(slice.to_le_bytes());
                for axis in region {
                    out.extend(axis.to_le_bytes());
                }
                out.extend(pattern);
            }
            Fill::Color {
                level,
                slice,
                region,
                color,
                packed_into,
            } => {
                out.extend(b"color");
                out.extend(level.to_le_bytes());
                out.extend(slice.to_le_bytes());
                for axis in region {
                    out.extend(axis.to_le_bytes());
                }
                for channel in color {
                    out.extend(channel.to_le_bytes());
                }
                out.extend(packed_into.to_le_bytes());
            }
        }
        out
    }
}

/// One full-screen fragment fill: a classic vertex-and-fragment pipeline, a private render target
/// of its own, one draw of three vertices, and the target's bytes read back through a blit.
///
/// This is the shape every store, clear and load question of a render pass is written in: the
/// vertex function computes the full-screen triangle from its `[[vertex_id]]`, so the pass needs no
/// vertex buffer, and what is observed is the bytes the fragment function's return value stored.
/// The target is **private** and read back by a blit into a shared buffer, because that is how the
/// probes measured it and storage mode is part of what a padded format's answer means.
#[derive(Clone, Copy, Debug)]
pub struct FragmentFill<'a> {
    /// The library both functions come from.
    pub library: Handle,
    /// The vertex function, which takes no buffer.
    pub vertex: &'a str,
    /// The fragment function, whose return value the attachment stores.
    pub fragment: &'a str,
    /// The attachment's and the target's pixel format.
    pub format: PixelFormat,
    /// The target's extent in texels.
    pub size: [u32; 2],
    /// Bytes a row of the read-back, which the case states because the pitch is part of what it
    /// means.
    pub row_pitch: u32,
    /// The buffers the pass sets as fragment buffers 0 and up, in order.
    pub buffers: &'a [Handle],
    /// The vertices the one `drawPrimitives:` of `MTLPrimitiveTypeTriangle` draws.
    pub vertices: u32,
}

/// What a [`FragmentFill`] answered: the pass's command-buffer status and the target's bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FragmentFilled {
    /// The render pass's command-buffer status, as a variant of its name.
    pub status: Value,
    /// The target's level 0, row after row at the fill's row pitch.
    pub bytes: Vec<u8>,
}

/// The color format every mesh case's attachments take: `MTLPixelFormatRGBA32Uint`, whose texel is
/// four 32-bit words, so what a shader stores is read back as the words it stored and a diff names
/// the word rather than a byte.
pub const TARGET_FORMAT: &str = "RGBA32Uint";

/// The bytes one [`TARGET_FORMAT`] texel takes.
pub const TARGET_TEXEL: u32 = 16;

/// The functions of a mesh render pipeline, as a case names them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshFunctions<'a> {
    /// The object function, where the pipeline has one.
    pub object: Option<&'a str>,
    pub mesh: &'a str,
    pub fragment: &'a str,
    /// The color attachments of [`TARGET_FORMAT`] the pipeline writes.
    pub attachments: u32,
}

/// One command a case encodes into a render pass.
///
/// The stage and scope fields are the raw `MTLRenderStages` and `MTLBarrierScope` bits, so the
/// vocabulary names neither API's types and a case reads as the Metal call it makes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeshStep {
    /// `setRenderPipelineState:`.
    Pipeline(Handle),
    /// `setObjectBuffer:offset:atIndex:`.
    ObjectBuffer {
        buffer: Handle,
        offset: u64,
        index: u32,
    },
    /// `setObjectBytes:length:atIndex:`.
    ObjectBytes { bytes: Vec<u8>, index: u32 },
    /// `setMeshBuffer:offset:atIndex:`.
    MeshBuffer {
        buffer: Handle,
        offset: u64,
        index: u32,
    },
    /// `setFragmentBuffer:offset:atIndex:`.
    FragmentBuffer {
        buffer: Handle,
        offset: u64,
        index: u32,
    },
    /// `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
    DrawThreadgroups {
        groups: [u32; 3],
        object: [u32; 3],
        mesh: [u32; 3],
    },
    /// `drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
    DrawThreads {
        threads: [u32; 3],
        object: [u32; 3],
        mesh: [u32; 3],
    },
    /// `drawMeshThreadgroupsWithIndirectBuffer:indirectBufferOffset:
    /// threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
    DrawIndirect {
        buffer: Handle,
        offset: u64,
        object: [u32; 3],
        mesh: [u32; 3],
    },
    /// `memoryBarrierWithScope:afterStages:beforeStages:`.
    BarrierScope { scope: u64, after: u64, before: u64 },
    /// `memoryBarrierWithResources:count:afterStages:beforeStages:` over buffers.
    BarrierResources {
        buffers: Vec<Handle>,
        after: u64,
        before: u64,
    },
}

impl MeshStep {
    /// The site name the step's answer is emitted at: the call it makes, so a refusal reads as the
    /// Metal call we refused.
    pub fn call(&self) -> &'static str {
        match self {
            MeshStep::Pipeline(_) => "setRenderPipelineState",
            MeshStep::ObjectBuffer { .. } => "setObjectBuffer",
            MeshStep::ObjectBytes { .. } => "setObjectBytes",
            MeshStep::MeshBuffer { .. } => "setMeshBuffer",
            MeshStep::FragmentBuffer { .. } => "setFragmentBuffer",
            MeshStep::DrawThreadgroups { .. } => "drawMeshThreadgroups",
            MeshStep::DrawThreads { .. } => "drawMeshThreads",
            MeshStep::DrawIndirect { .. } => "drawMeshThreadgroupsWithIndirectBuffer",
            MeshStep::BarrierScope { .. } => "memoryBarrierWithScope",
            MeshStep::BarrierResources { .. } => "memoryBarrierWithResources",
        }
    }
}

/// `MTLBarrierScopeBuffers`.
pub const SCOPE_BUFFERS: u64 = 0x1;

/// `MTLRenderStageVertex`.
pub const STAGE_VERTEX: u64 = 0x1;

/// `MTLRenderStageFragment`.
pub const STAGE_FRAGMENT: u64 = 0x2;

/// `MTLRenderStageObject`.
pub const STAGE_OBJECT: u64 = 0x8;

/// `MTLRenderStageMesh`.
pub const STAGE_MESH: u64 = 0x10;

/// What one render pass came to.
#[derive(Clone, Debug, PartialEq)]
pub struct PassOutcome {
    /// What each step of the pass answered, in the order the case gave them: `None` where the call
    /// answered, the typed error where it did not. A side whose calls cannot refuse (Metal's
    /// encoder methods return nothing) answers `None` throughout, so a refusal of ours compares
    /// against Metal's answer and reads as a refusal rather than a divergence.
    pub refused: Vec<Option<Failure>>,
    /// The command buffer's status after the bounded wait, as the variant of its name.
    pub status: Value,
    /// The command buffer's error after it completed, where it reported one.
    pub error: Option<Value>,
}

/// What an indirect render command's getters answered, beside what the whole record holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndirectReadBack {
    /// `getCommandType`'s answer, as its bits.
    pub command_type: u64,
    /// The primitive type the draw getter answered, or `None` where it answered no draw.
    pub answered_primitive_type: Option<u64>,
    /// The primitive type the whole record's read-back keeps.
    pub recorded_primitive_type: Option<u64>,
}

/// Where a case's Metal answer comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetalSource {
    /// Record mode runs the case on Metal.
    Run,
    /// Metal has no public call for what the case reads, or running it on Metal is not safe, so
    /// its answer is carried from a measurement this tree already holds: the module whose table it
    /// came from, and whether record mode may replace it.
    Carried {
        from: &'static str,
        re_record: ReRecord,
        /// The chip the carried measurement was taken on, which is the oracle's unless the case
        /// says otherwise ([`MetalSource::measured_on`]).
        measured_on: &'static str,
    },
    /// **Metal did not return from the call.** There is nothing to carry and nothing to run.
    ///
    /// A row of this shape has no value and no behaviour on Metal's side: the probe that found it
    /// out ended Metal's process, so the lane it came from holds no line for it either — it is not
    /// recorded-only material, it simply has no answer. What the case pins is the one fact there
    /// is, that Metal never came back from the call, against our typed refusal at that site
    /// ([`Verdict::MetalNeverReturned`](crate::verdict::Verdict::MetalNeverReturned)).
    ///
    /// **It is never produced by running the shape.** The transcript is stated by the case
    /// ([`Case::carried_transcript`]) from a probe that already died, and record mode refuses the
    /// row outright rather than taking an answer from it: this is the one source whose Metal leg
    /// would end the process, and the suite's standing rule is that no case may do that.
    DidNotReturn {
        /// Where the probe that died is written down.
        from: &'static str,
        /// The call Metal did not return from, as the case names the site.
        call: &'static str,
        /// What happened, in the probe's own words: how Metal went away.
        says: &'static str,
    },
}

impl MetalSource {
    /// A carried answer record mode may replace by running the case on Metal.
    pub const fn carried(from: &'static str) -> Self {
        Self::Carried {
            from,
            re_record: ReRecord::Allowed,
            measured_on: crate::recording::ORACLE_CHIP,
        }
    }

    /// A carried answer record mode must refuse to replace, because running the case on Metal is
    /// not safe.
    pub const fn carried_only(from: &'static str, reason: &'static str) -> Self {
        Self::Carried {
            from,
            re_record: ReRecord::Forbidden { reason },
            measured_on: crate::recording::ORACLE_CHIP,
        }
    }

    /// The same carried answer, measured on `chip` rather than on the oracle this round records
    /// on.
    ///
    /// A case says this where its table predates the oracle — the two families carrying an Apple
    /// M4's values — so the recording holds the chip in a field a reader and the report can see,
    /// instead of in the `from` prose where nothing could.
    pub const fn measured_on(self, chip: &'static str) -> Self {
        match self {
            Self::Carried {
                from, re_record, ..
            } => Self::Carried {
                from,
                re_record,
                measured_on: chip,
            },
            Self::Run => Self::Run,
            source @ Self::DidNotReturn { .. } => source,
        }
    }

    /// A row whose Metal leg does not come back: `call` is the call, `says` is how Metal went
    /// away, and `from` is where the probe that found that out is written down.
    pub const fn did_not_return(
        from: &'static str,
        call: &'static str,
        says: &'static str,
    ) -> Self {
        Self::DidNotReturn { from, call, says }
    }

    /// Why record mode must not run the case on Metal, where it must not.
    pub const fn forbidden(&self) -> Option<&'static str> {
        match self {
            MetalSource::Carried {
                re_record: ReRecord::Forbidden { reason },
                ..
            } => Some(reason),
            // Record mode must never run this one: the call it would make is the call Metal does
            // not come back from. The case's own words are the reason a refusal prints.
            MetalSource::DidNotReturn { says, .. } => Some(says),
            MetalSource::Run | MetalSource::Carried { .. } => None,
        }
    }
}

/// Whether record mode may run a case on Metal.
///
/// A case whose Metal leg faults the GPU, trips its watchdog or ends the process carries its
/// answer and forbids re-recording. Record mode refuses such a case by this flag, so a later
/// re-record of a whole selection cannot crash the GPU or the process by walking into one. Our own
/// leg of such a case is safe and compares normally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReRecord {
    Allowed,
    Forbidden { reason: &'static str },
}

/// A case of the suite.
pub trait Case {
    /// The case's name: its recording's file name, and its row in the report.
    fn name(&self) -> &'static str;

    /// Where the case's Metal answer comes from.
    fn metal_source(&self) -> MetalSource;

    /// The shaders the case runs, as committed fixtures; empty where it runs none.
    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        Ok(Vec::new())
    }

    /// The fingerprint of everything the case feeds its run: the bytes it writes into each buffer
    /// and texture before it runs, its specialization constants, and the parameters it sets.
    ///
    /// The suite emits this itself, at the reserved site [`INPUTS_SITE`], and the comparison checks
    /// it before any output. A case whose inputs are not the recording's is
    /// [`Unrecorded`](crate::verdict::Verdict::Unrecorded), never a pass: its Metal answer was
    /// taken from other bytes. Without it a recording decays in silence the first time a fixture
    /// or a generator changes.
    fn inputs(&self) -> Result<Vec<Input>, String>;

    /// Set the case up on `side`, run it, and emit the sites it observes. A call that did not
    /// answer stops the run with an [`Abort`]; the sites emitted before it still compare.
    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort>;

    /// What the case loosens, and what justifies each. Empty is the default, and the default is
    /// what a case should keep: comparison is exact.
    fn allowances(&self) -> Vec<Allowance> {
        Vec::new()
    }

    /// How many runs record mode takes the case's Metal answer over.
    ///
    /// The default ([`crate::run::RECORD_RUNS`]) is enough for a case whose behaviour is settled:
    /// the runs then all fold into one outcome and the recording says so. A case whose work Metal
    /// orders by nothing must ask for more, because its recording's value is the **set** of
    /// outcomes with their frequencies, and a set sampled three times states a frequency nobody
    /// should read. Asking for more runs never invents one: the recording holds the outcomes this
    /// many runs actually produced.
    fn record_runs(&self) -> u32 {
        crate::run::RECORD_RUNS
    }

    /// The transcript the case's carried measurement states, and the runs it stands on, for a case
    /// whose [`MetalSource`] is [`MetalSource::Carried`]. `None` for a case record mode runs on
    /// Metal.
    ///
    /// A carried answer is the case stating what our own measurement of this chip found, in the
    /// shape the case emits, so it compares through the same path a recorded run does.
    fn carried_transcript(&self) -> Result<Option<(u32, Transcript)>, String> {
        Ok(None)
    }
}

/// The reserved site the suite emits a case's input fingerprint at.
pub const INPUTS_SITE: &str = "inputs";

/// One input a case feeds its run, and its digest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Input {
    /// What the input is, as the case names it: `buffer:0`, `constant:width`, `texture:albedo`.
    pub name: String,
    pub digest: crate::transcript::Digest,
}

impl Input {
    /// The input `name`, fingerprinted by the SHA-256 of `bytes`.
    pub fn bytes(name: impl Into<String>, bytes: &[u8]) -> Self {
        Self {
            name: name.into(),
            digest: crate::transcript::Digest::sha256(bytes),
        }
    }
}

/// The node the suite emits for a case's inputs.
pub fn inputs_node(inputs: &[Input]) -> Node {
    Node::branch(
        INPUTS_SITE,
        inputs
            .iter()
            .map(|input| {
                Node::leaf(
                    input.name.clone(),
                    Observed::absent().digested(input.digest.clone()),
                )
            })
            .collect(),
    )
}

/// The site a call that answered an object emits, as a carried measurement states it: the call
/// answered, and nothing of what it answered. A carried transcript holds the same sites a run
/// emits, so it states this one too.
pub fn answered() -> Observed {
    Observed::value(Value::ok(Value::Absent))
}

/// Emit a site for `answer` and go on with what it answered, or stop the run.
///
/// `observed` says what the site records of a successful answer: `Value::Absent` for a call that
/// answers an object, whose identity is not observable.
pub fn call<T>(
    sites: &mut Sites,
    at: &str,
    answer: Answer<T>,
    observed: impl FnOnce(&T) -> Value,
) -> Result<T, Abort> {
    match answer {
        Ok(value) => {
            sites.value(at, Value::ok(observed(&value)));
            Ok(value)
        }
        Err(failure) => {
            sites.push(Node::leaf(at, Observed::value(failure.recorded())));
            // A device's honest gap is recorded at the site, not only where it stops the case.
            // The whole-case `LacksFeature` verdict comes from an abort, so without this a case
            // that carries on past an unsupported call would have that site read as a refusal of
            // ours — the one thing the verdict exists to prevent.
            if let Some(needs) = &failure.lacks {
                sites.device_gap(at, needs.clone());
            }
            Err(Abort {
                at: at.to_owned(),
                failure,
            })
        }
    }
}

/// [`call`] for a **sweep that carries on** past a call that did not answer: the site records the
/// answer either way, the venue's own gap is recorded with it, and nothing aborts.
///
/// A sweep over many queries wants one refused query to compare as a refusal at its own site while
/// the rest stay readable, which an abort would lose. Written by hand that reads
/// `Err(failure) => failure.recorded()`, and then [`Failure::lacks`] — whether the typed refusal is the
/// **venue's** honest gap rather than ours — is silently dropped, so a device that simply cannot
/// run one query of the sweep has that site scored as a refusal of ours. That is the one thing the
/// `lacks-feature` verdict exists to prevent, and it was happening on lavapipe for every
/// multisampled sparse query. Use this rather than matching, so the fact cannot be forgotten.
pub fn probe<T>(
    sites: &mut Sites,
    at: &str,
    answer: Answer<T>,
    observed: impl FnOnce(&T) -> Value,
) -> Option<T> {
    match answer {
        Ok(value) => {
            sites.value(at, Value::ok(observed(&value)));
            Some(value)
        }
        Err(failure) => {
            sites.push(Node::leaf(at, Observed::value(failure.recorded())));
            if let Some(needs) = &failure.lacks {
                sites.device_gap(at, needs.clone());
            }
            None
        }
    }
}

/// [`call`] for a call that answers an object: the site records that it answered, and nothing of
/// what it answered.
pub fn made<T>(sites: &mut Sites, at: &str, answer: Answer<T>) -> Result<T, Abort> {
    call(sites, at, answer, |_| Value::Absent)
}
