//! The render vocabulary: a render target, a vertex-and-fragment pipeline, and the work one
//! command buffer holds.
//!
//! The mesh families reach the device through [`crate::case::MeshStep`], which says nothing about a
//! load action, a clear color, a blend or a resolve, because a mesh case needs none of them. The
//! families where Metal's float rules live need all four: what a target stores turns on the pass's
//! load action and clear value, on the pipeline's blend, and on whether the samples are resolved.
//! So this is a second vocabulary beside that one rather than a widening of it, and it keeps the
//! mesh cases' shape untouched.
//!
//! Four things a case states, and nothing else:
//!
//! - **A target** ([`TargetSpec`]) is private by default, because that is what a render target of
//!   one of Apple's extended-range or bfloat formats is made with, and because Metal's clear route
//!   reads the texture's usage and extent rather than where it lives. A private target is read back
//!   with [`RenderWork::ReadTexture`], the blit every host can run.
//! - **A pipeline** ([`RenderPipelineSpec`]) names its two functions, the format, **write mask**
//!   and blend of each color attachment, the depth and stencil formats, and its sample count.
//! - **A depth and stencil state** ([`crate::case::Side::depth_stencil_state`]) is Metal's
//!   `MTLDepthStencilDescriptor` as the translator already names it, so there is no second
//!   spelling of a compare function or a stencil operation either.
//! - **The work** ([`RenderWork`]) is a list, and the whole list goes into **one** command buffer,
//!   committed and waited for once. That is deliberate: the clear sweeps ask Metal hundreds of
//!   one-texel questions, and one command buffer a question would cost hundreds of round trips on
//!   both sides. A case that wants several command buffers calls [`crate::case::Side::render`] more
//!   than once, so the batching is the case's to state rather than the vocabulary's to assume.
//!
//! A pass ([`RenderPass`]) holds its color attachments, its **depth** and **stencil** attachments
//! and its **visibility-result buffer**, and its steps ([`RenderStep`]) are the pipeline and
//! depth-stencil state, the stencil reference, the vertex and fragment buffer bindings, the
//! visibility-result mode, and the draws: direct, indexed and indirect.
//!
//! **A multisampled attachment is read one sample at a time through a kernel**, not through a
//! call of its own: neither side can `getBytes` or blit a multisample texture, so a case makes the
//! target `shader_read` as well ([`TargetSpec::multisampled`] beside
//! [`TargetSpec::read`]), binds it to a [`Dispatch`] in the same command buffer, and the kernel
//! writes each sample's texels into a buffer. A resolve is the other half of that question and is
//! the descriptor's ([`PassColor::resolving`]).
//!
//! Every enumeration here is Metal's own as the translator and the runtime name it, so a value a
//! case writes reaches `MTLBlendFactor`, `MTLBlendOperation`, `MTLColorWriteMask`, `MTLLoadAction`,
//! `MTLStoreAction`, `MTLPrimitiveType`, `MTLIndexType`, `MTLVisibilityResultMode` and the
//! multisample resolve filters by its raw ordinal on the Metal side, and reaches the runtime's
//! descriptor unchanged. There is no second spelling to keep in step.

use crate::case::Handle;
use metal2vulkan::metal::{
    BlendFactor, BlendOperation, ColorWriteMask, IndexType, PixelFormat, PrimitiveType,
};
use metal2vulkan_runtime::{
    LoadAction, MultisampleDepthResolveFilter, MultisampleStencilResolveFilter, StoreAction,
    VisibilityResultMode,
};

/// A render target a case makes: `MTLTextureDescriptor`'s facts a render case states.
///
/// The usage is part of what the case means, not a detail: Metal clears a `BGR10_XR` target
/// straight from the `float` for a plain private target and through `half` for one of
/// `ShaderWrite` usage, so a case that let the side choose its usage would be asking a different
/// question on each side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetSpec {
    pub format: PixelFormat,
    pub size: [u32; 2],
    /// `sampleCount`: 1, or the samples of a multisampled target.
    pub samples: u32,
    /// `MTLTextureUsageShaderRead` beside `MTLTextureUsageRenderTarget`.
    pub shader_read: bool,
    /// `MTLTextureUsageShaderWrite` beside `MTLTextureUsageRenderTarget`.
    pub shader_write: bool,
    /// Whether the target is of shared storage rather than private.
    ///
    /// Private is the default and what an extended-range target is normally made with. A case says
    /// shared when the host has to put **arbitrary raw codes** into the target before the pass:
    /// a blend's destination texels are random codes, and neither a clear nor a kernel write can
    /// set an arbitrary code, so they go in through `replaceRegion:`, which no private texture
    /// takes.
    pub shared: bool,
}

impl TargetSpec {
    /// A single-sampled target of `format` and `size` a shader also reads.
    pub const fn read(format: PixelFormat, size: [u32; 2]) -> Self {
        Self {
            format,
            size,
            samples: 1,
            shader_read: true,
            shader_write: false,
            shared: false,
        }
    }

    /// A single-sampled target of `format` and `size` of `MTLTextureUsageRenderTarget` **alone**,
    /// which is how the lanes of the tile-store and visibility families make theirs.
    ///
    /// The usage is part of the question, not a detail: Metal refuses a depth texture of
    /// shader-write usage, and a target a shader may read is a different texture to the driver, so
    /// a case that asks what a pass leaves in a plain render target says so here.
    pub const fn plain(format: PixelFormat, size: [u32; 2]) -> Self {
        Self {
            format,
            size,
            samples: 1,
            shader_read: false,
            shader_write: false,
            shared: false,
        }
    }

    /// The target of shared storage, so the host may write its texels with `replaceRegion:`.
    #[must_use]
    pub const fn shared(mut self) -> Self {
        self.shared = true;
        self
    }

    /// The target at `samples` samples.
    #[must_use]
    pub const fn multisampled(mut self, samples: u32) -> Self {
        self.samples = samples;
        self
    }

    /// The target with `MTLTextureUsageShaderWrite` as well.
    #[must_use]
    pub const fn written(mut self) -> Self {
        self.shader_write = true;
        self
    }

    /// The words a case's input fingerprint covers: every field, so a target that moves is a
    /// change to the case's inputs rather than a quiet pass against another target's answer.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = self.format.as_raw().to_le_bytes().to_vec();
        for axis in self.size {
            bytes.extend(axis.to_le_bytes());
        }
        bytes.extend(self.samples.to_le_bytes());
        bytes.push(u8::from(self.shader_read));
        bytes.push(u8::from(self.shader_write));
        bytes.push(u8::from(self.shared));
        bytes
    }
}

/// A color attachment's blend: `MTLRenderPipelineColorAttachmentDescriptor`'s six fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blend {
    pub source_rgb: BlendFactor,
    pub destination_rgb: BlendFactor,
    pub rgb: BlendOperation,
    pub source_alpha: BlendFactor,
    pub destination_alpha: BlendFactor,
    pub alpha: BlendOperation,
}

impl Blend {
    /// The blend that adds the source times `source` to the destination times `destination`, on
    /// the color channels and the alpha channel alike.
    pub const fn added(source: BlendFactor, destination: BlendFactor) -> Self {
        Self {
            source_rgb: source,
            destination_rgb: destination,
            rgb: BlendOperation::Add,
            source_alpha: source,
            destination_alpha: destination,
            alpha: BlendOperation::Add,
        }
    }

    /// The color channels' blend and the alpha channel's, stated apart.
    pub const fn separate(
        rgb: (BlendFactor, BlendFactor, BlendOperation),
        alpha: (BlendFactor, BlendFactor, BlendOperation),
    ) -> Self {
        Self {
            source_rgb: rgb.0,
            destination_rgb: rgb.1,
            rgb: rgb.2,
            source_alpha: alpha.0,
            destination_alpha: alpha.1,
            alpha: alpha.2,
        }
    }

    /// The words a case's input fingerprint covers.
    pub fn words(&self) -> Vec<u8> {
        [
            self.source_rgb.as_raw(),
            self.destination_rgb.as_raw(),
            self.rgb.as_raw(),
            self.source_alpha.as_raw(),
            self.destination_alpha.as_raw(),
            self.alpha.as_raw(),
        ]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
    }
}

/// One color attachment of a render pipeline: its format, and its blend where blending is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorState {
    pub format: PixelFormat,
    /// `writeMask`: which of the attachment's channels a fragment writes.
    ///
    /// Metal's default is every channel, and a mask of **none** is a question of its own: a draw
    /// that writes no texel still makes the pass a drawing pass, which is how the tile-store
    /// family asks what Metal's store leaves behind in the texels nothing wrote.
    pub write_mask: ColorWriteMask,
    /// `blendingEnabled` and the six factors, `None` for blending off.
    pub blend: Option<Blend>,
}

impl ColorState {
    /// An attachment of `format` with Metal's defaults: every channel written, blending off.
    pub const fn new(format: PixelFormat) -> Self {
        Self {
            format,
            write_mask: ColorWriteMask::ALL,
            blend: None,
        }
    }

    /// The attachment with `writeMask` set to `mask`.
    #[must_use]
    pub const fn masked(mut self, mask: ColorWriteMask) -> Self {
        self.write_mask = mask;
        self
    }

    /// The attachment with `blend` enabled.
    #[must_use]
    pub const fn blended(mut self, blend: Blend) -> Self {
        self.blend = Some(blend);
        self
    }

    /// The words a case's input fingerprint covers.
    ///
    /// The write mask is written **only where it is not Metal's default**, so an attachment that
    /// writes every channel fingerprints to the bytes it always has. Every widening here keeps
    /// that property: a recording is invalidated by a case's own inputs moving, never by a field
    /// another family needed being added beside them.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = self.format.as_raw().to_le_bytes().to_vec();
        if self.write_mask != ColorWriteMask::ALL {
            bytes.extend(self.write_mask.as_raw().to_le_bytes());
        }
        match &self.blend {
            Some(blend) => {
                bytes.push(1);
                bytes.extend(blend.words());
            }
            None => bytes.push(0),
        }
        bytes
    }
}

/// One stage of a render pipeline: the function's entry name, and the library it comes from when
/// the pipeline's two stages do not share one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageFunction<'a> {
    /// The library the function comes from, when that is not the one the call to
    /// [`crate::case::Side::render_pipeline`] names.
    pub library: Option<Handle>,
    pub entry: &'a str,
}

impl<'a> StageFunction<'a> {
    /// A function of the library the call names.
    #[must_use]
    pub const fn named(entry: &'a str) -> Self {
        Self {
            library: None,
            entry,
        }
    }

    /// A function of `library`, whichever library the call names.
    #[must_use]
    pub const fn of(library: Handle, entry: &'a str) -> Self {
        Self {
            library: Some(library),
            entry,
        }
    }

    /// The words a case's input fingerprint covers: the entry, and the library only where the
    /// stage names one of its own, so a pipeline of one library fingerprints as it always has.
    fn words(&self) -> Vec<u8> {
        let mut bytes = self.entry.as_bytes().to_vec();
        bytes.push(0);
        if let Some(library) = self.library {
            bytes.extend(library.0.to_le_bytes());
        }
        bytes
    }
}

/// A vertex-and-fragment render pipeline a case makes:
/// `newRenderPipelineStateWithDescriptor:error:`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderPipelineSpec<'a> {
    pub vertex: StageFunction<'a>,
    /// The fragment stage, or `None` for a pipeline that rasterizes without one.
    pub fragment: Option<StageFunction<'a>>,
    /// The color attachments by index; `None` at an index of `MTLPixelFormatInvalid`, which is how
    /// a pipeline that writes attachment 1 alone is stated.
    pub colors: &'a [Option<ColorState>],
    /// `depthAttachmentPixelFormat`; `None` is `MTLPixelFormatInvalid`, a pipeline without one.
    pub depth_format: Option<PixelFormat>,
    /// `stencilAttachmentPixelFormat`; `None` is `MTLPixelFormatInvalid`.
    pub stencil_format: Option<PixelFormat>,
    /// `rasterSampleCount`.
    pub samples: u32,
}

impl<'a> RenderPipelineSpec<'a> {
    /// A single-sampled pipeline of `vertex` and `fragment`, both of the library the call names,
    /// over `colors`.
    pub const fn new(vertex: &'a str, fragment: &'a str, colors: &'a [Option<ColorState>]) -> Self {
        Self::of(
            StageFunction::named(vertex),
            Some(StageFunction::named(fragment)),
            colors,
        )
    }

    /// A single-sampled pipeline of stages that may each name their own library, and whose fragment
    /// stage may be absent.
    pub const fn of(
        vertex: StageFunction<'a>,
        fragment: Option<StageFunction<'a>>,
        colors: &'a [Option<ColorState>],
    ) -> Self {
        Self {
            vertex,
            fragment,
            colors,
            depth_format: None,
            stencil_format: None,
            samples: 1,
        }
    }

    /// The pipeline at `samples` samples.
    #[must_use]
    pub const fn multisampled(mut self, samples: u32) -> Self {
        self.samples = samples;
        self
    }

    /// The pipeline whose depth and stencil attachment formats are `depth` and `stencil`. A
    /// pipeline that tests or writes either must name its format, as Metal's descriptor must.
    #[must_use]
    pub const fn depth_stencil(
        mut self,
        depth: Option<PixelFormat>,
        stencil: Option<PixelFormat>,
    ) -> Self {
        self.depth_format = depth;
        self.stencil_format = stencil;
        self
    }

    /// The words a case's input fingerprint covers: the functions, every attachment, the depth and
    /// stencil formats where the pipeline names them, and the sample count.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = self.vertex.words();
        match &self.fragment {
            Some(fragment) => bytes.extend(fragment.words()),
            None => bytes.push(0xff),
        }
        for color in self.colors {
            match color {
                Some(state) => bytes.extend(state.words()),
                None => bytes.push(0xff),
            }
        }
        for format in [self.depth_format, self.stencil_format]
            .into_iter()
            .flatten()
        {
            bytes.extend(format.as_raw().to_le_bytes());
        }
        bytes.extend(self.samples.to_le_bytes());
        bytes
    }
}

/// The depth attachment of a render pass: `MTLRenderPassDepthAttachmentDescriptor`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PassDepth {
    pub texture: Handle,
    pub load: LoadAction,
    pub store: StoreAction,
    /// `clearDepth`, as `f64`, since Metal's is a double.
    pub clear: f64,
    /// `resolveTexture`.
    pub resolve: Option<Handle>,
    /// `depthResolveFilter`.
    pub filter: MultisampleDepthResolveFilter,
}

impl PassDepth {
    /// `texture` cleared to `clear` and stored.
    pub const fn cleared(texture: Handle, clear: f64) -> Self {
        Self {
            texture,
            load: LoadAction::Clear,
            store: StoreAction::Store,
            clear,
            resolve: None,
            filter: MultisampleDepthResolveFilter::Sample0,
        }
    }

    /// The attachment with `storeAction` set to `store`.
    #[must_use]
    pub const fn stored(mut self, store: StoreAction) -> Self {
        self.store = store;
        self
    }

    /// The attachment resolved into `resolve` through `filter`, stored and resolved.
    #[must_use]
    pub const fn resolving(
        mut self,
        resolve: Handle,
        filter: MultisampleDepthResolveFilter,
    ) -> Self {
        self.resolve = Some(resolve);
        self.filter = filter;
        self.store = StoreAction::StoreAndMultisampleResolve;
        self
    }

    /// The words a case's input fingerprint covers. Nothing of a handle is fingerprinted: a
    /// side's handle numbers are its own.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = vec![u8::from(self.resolve.is_some())];
        bytes.extend(self.load.raw().to_le_bytes());
        bytes.extend(self.store.raw().to_le_bytes());
        bytes.extend(self.clear.to_bits().to_le_bytes());
        bytes.extend(self.filter.raw().to_le_bytes());
        bytes
    }
}

/// The stencil attachment of a render pass: `MTLRenderPassStencilAttachmentDescriptor`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassStencil {
    pub texture: Handle,
    pub load: LoadAction,
    pub store: StoreAction,
    /// `clearStencil`; Metal clears its low 8 bits.
    pub clear: u32,
    /// `resolveTexture`.
    pub resolve: Option<Handle>,
    /// `stencilResolveFilter`.
    pub filter: MultisampleStencilResolveFilter,
}

impl PassStencil {
    /// `texture` cleared to `clear` and stored.
    pub const fn cleared(texture: Handle, clear: u32) -> Self {
        Self {
            texture,
            load: LoadAction::Clear,
            store: StoreAction::Store,
            clear,
            resolve: None,
            filter: MultisampleStencilResolveFilter::Sample0,
        }
    }

    /// The attachment with `storeAction` set to `store`.
    #[must_use]
    pub const fn stored(mut self, store: StoreAction) -> Self {
        self.store = store;
        self
    }

    /// The attachment resolved into `resolve` through `filter`, stored and resolved.
    #[must_use]
    pub const fn resolving(
        mut self,
        resolve: Handle,
        filter: MultisampleStencilResolveFilter,
    ) -> Self {
        self.resolve = Some(resolve);
        self.filter = filter;
        self.store = StoreAction::StoreAndMultisampleResolve;
        self
    }

    /// The words a case's input fingerprint covers.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = vec![u8::from(self.resolve.is_some())];
        bytes.extend(self.load.raw().to_le_bytes());
        bytes.extend(self.store.raw().to_le_bytes());
        bytes.extend(self.clear.to_le_bytes());
        bytes.extend(self.filter.raw().to_le_bytes());
        bytes
    }
}

/// One color attachment of a render pass: `MTLRenderPassColorAttachmentDescriptor`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PassColor {
    /// The texture the attachment renders into; `None` leaves the index unattached, which is how a
    /// pass that attaches color 1 alone is stated.
    pub texture: Option<Handle>,
    pub load: LoadAction,
    pub store: StoreAction,
    /// `clearColor`'s four channels, as `f64`, since Metal's clear color is a double and the
    /// double a case names is part of what the clear route rounds.
    pub clear: [f64; 4],
    /// `resolveTexture`: the single-sampled texture a resolving store action writes.
    pub resolve: Option<Handle>,
}

impl PassColor {
    /// An attachment that is not there.
    pub const fn none() -> Self {
        Self {
            texture: None,
            load: LoadAction::DontCare,
            store: StoreAction::Store,
            clear: [0.0, 0.0, 0.0, 1.0],
            resolve: None,
        }
    }

    /// `texture` cleared to `clear` and stored.
    pub const fn cleared(texture: Handle, clear: [f64; 4]) -> Self {
        Self {
            texture: Some(texture),
            load: LoadAction::Clear,
            store: StoreAction::Store,
            clear,
            resolve: None,
        }
    }

    /// `texture` loaded and stored, the shape a blend draws over.
    pub const fn loaded(texture: Handle) -> Self {
        Self {
            texture: Some(texture),
            load: LoadAction::Load,
            store: StoreAction::Store,
            clear: [0.0, 0.0, 0.0, 1.0],
            resolve: None,
        }
    }

    /// `texture` whose texels the pass does not read and does not keep: the attachment a routes
    /// case draws into beside the one it is asking about.
    pub const fn discarded(texture: Handle) -> Self {
        Self {
            texture: Some(texture),
            load: LoadAction::DontCare,
            store: StoreAction::Store,
            clear: [0.0, 0.0, 0.0, 1.0],
            resolve: None,
        }
    }

    /// The attachment with `storeAction` set to `store`: a multisampled attachment a pass does
    /// not keep says `DontCare` here, since Metal stores no multisample texture a host reads.
    #[must_use]
    pub const fn stored(mut self, store: StoreAction) -> Self {
        self.store = store;
        self
    }

    /// The attachment with `resolve` as its resolve texture, stored and resolved.
    #[must_use]
    pub const fn resolving(mut self, resolve: Handle) -> Self {
        self.resolve = Some(resolve);
        self.store = StoreAction::StoreAndMultisampleResolve;
        self
    }

    /// The words a case's input fingerprint covers: the actions and the clear color's bits, so a
    /// clear value that moves by one `f64` step is a change to the case's inputs. Nothing of a
    /// handle is fingerprinted: a side's handle numbers are its own.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = vec![
            u8::from(self.texture.is_some()),
            u8::from(self.resolve.is_some()),
        ];
        bytes.extend(self.load.raw().to_le_bytes());
        bytes.extend(self.store.raw().to_le_bytes());
        for channel in self.clear {
            bytes.extend(channel.to_bits().to_le_bytes());
        }
        bytes
    }
}

/// One viewport: `MTLViewport`'s six numbers.
///
/// The depths are kept beside the rectangle because `znear` and `zfar` are part of the same Metal
/// call, and a case that sets a viewport without saying what it does to depth would be two
/// questions in one.
///
/// Two viewports are the same viewport when their six numbers hold the same bits, which is how the
/// input fingerprint reads them too ([`ViewportRect::words`]). Comparing the bits rather than the
/// numbers keeps the equality reflexive, so the type is honestly [`Eq`] despite holding `f64`s, and
/// it is what a case wants: a viewport edge one `f64` step away is a different viewport.
#[derive(Clone, Copy, Debug)]
pub struct ViewportRect {
    /// `originX` and `originY`, in pixels.
    pub origin: [f64; 2],
    /// `width` and `height`, in pixels; a negative extent mirrors the draw, as Metal's does.
    pub size: [f64; 2],
    /// `znear` and `zfar`.
    pub depth: [f64; 2],
}

impl PartialEq for ViewportRect {
    fn eq(&self, other: &Self) -> bool {
        self.words() == other.words()
    }
}

impl Eq for ViewportRect {}

impl ViewportRect {
    /// A viewport of `origin` and `size` over the whole depth range.
    pub const fn new(origin: [f64; 2], size: [f64; 2]) -> Self {
        Self {
            origin,
            size,
            depth: [0.0, 1.0],
        }
    }

    /// The words a case's input fingerprint covers: every number by its bits, so a viewport edge
    /// that moves by one `f64` step is a change to the case's inputs.
    pub fn words(&self) -> Vec<u8> {
        self.origin
            .iter()
            .chain(&self.size)
            .chain(&self.depth)
            .flat_map(|number| number.to_bits().to_le_bytes())
            .collect()
    }
}

/// One command a case encodes into a render pass, in the order it lists them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenderStep {
    /// `setRenderPipelineState:`.
    Pipeline(Handle),
    /// `setDepthStencilState:` of a state from
    /// [`Side::depth_stencil_state`](crate::case::Side::depth_stencil_state).
    DepthStencilState(Handle),
    /// `setStencilReferenceValue:`.
    StencilReference(u32),
    /// `setVertexBuffer:offset:atIndex:`.
    VertexBuffer {
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
    /// `setFragmentTexture:atIndex:`.
    FragmentTexture { texture: Handle, index: u32 },
    /// `setFragmentSamplerState:atIndex:`.
    FragmentSampler { sampler: Handle, index: u32 },
    /// `setVertexBytes:length:atIndex:`: a vertex binding whose bytes the call copies.
    ///
    /// A case uses this where the binding is a handful of bytes that change every draw — a point's
    /// two coordinates — because a buffer a draw would be hundreds of allocations saying the same
    /// thing, and because the inline binding is its own Metal path.
    VertexBytes { bytes: Vec<u8>, index: u32 },
    /// `setViewport:`.
    Viewport(ViewportRect),
    /// `setVisibilityResultMode:offset:`. A pass that sets one needs a visibility-result buffer
    /// ([`RenderPass::counting_into`]); setting it without one is a refusal a case may be asking
    /// about.
    VisibilityResultMode {
        mode: VisibilityResultMode,
        offset: u64,
    },
    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:`, whose base instance
    /// is 0.
    Draw {
        primitive: PrimitiveType,
        first: u32,
        count: u32,
        instances: u32,
    },
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:
    /// instanceCount:baseVertex:baseInstance:`.
    DrawIndexed {
        primitive: PrimitiveType,
        count: u32,
        index_type: IndexType,
        indices: Handle,
        offset: u64,
        instances: u32,
        base_vertex: i32,
        base_instance: u32,
    },
    /// `drawPrimitives:indirectBuffer:indirectBufferOffset:`, whose buffer holds the draw's four
    /// 32-bit arguments.
    DrawIndirect {
        primitive: PrimitiveType,
        arguments: Handle,
        offset: u64,
    },
}

impl RenderStep {
    /// The fullscreen triangle every fill fixture here draws: three vertices of one instance.
    pub const fn triangle() -> Self {
        Self::Draw {
            primitive: PrimitiveType::Triangle,
            first: 0,
            count: 3,
            instances: 1,
        }
    }

    /// The same triangle from vertex 3, which covers no pixel: how a visibility case asks what a
    /// draw that passes no sample writes.
    pub const fn no_triangle() -> Self {
        Self::Draw {
            primitive: PrimitiveType::Triangle,
            first: 3,
            count: 3,
            instances: 1,
        }
    }

    /// A vertex buffer bound at `index`, from its start.
    pub const fn vertex_buffer(buffer: Handle, index: u32) -> Self {
        Self::VertexBuffer {
            buffer,
            offset: 0,
            index,
        }
    }

    /// A fragment buffer bound at `index`, from its start.
    pub const fn fragment_buffer(buffer: Handle, index: u32) -> Self {
        Self::FragmentBuffer {
            buffer,
            offset: 0,
            index,
        }
    }

    /// The site name the step's answer is emitted at: the call it makes, so a refusal reads as the
    /// Metal call we refused.
    pub fn call(&self) -> &'static str {
        match self {
            RenderStep::Pipeline(_) => "setRenderPipelineState",
            RenderStep::DepthStencilState(_) => "setDepthStencilState",
            RenderStep::StencilReference(_) => "setStencilReferenceValue",
            RenderStep::VertexBuffer { .. } => "setVertexBuffer",
            RenderStep::FragmentBuffer { .. } => "setFragmentBuffer",
            RenderStep::FragmentTexture { .. } => "setFragmentTexture",
            RenderStep::FragmentSampler { .. } => "setFragmentSamplerState",
            RenderStep::VertexBytes { .. } => "setVertexBytes",
            RenderStep::Viewport(_) => "setViewport",
            RenderStep::VisibilityResultMode { .. } => "setVisibilityResultMode",
            RenderStep::Draw { .. } => "drawPrimitives",
            RenderStep::DrawIndexed { .. } => "drawIndexedPrimitives",
            RenderStep::DrawIndirect { .. } => "drawPrimitivesWithIndirectBuffer",
        }
    }

    /// The parameters the step's own call carries, for a case's input fingerprint: everything but
    /// the handles, whose numbers are a side's own.
    pub fn parameters(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        match self {
            RenderStep::Pipeline(_) | RenderStep::DepthStencilState(_) => {}
            RenderStep::StencilReference(value) => bytes.extend(value.to_le_bytes()),
            RenderStep::VertexBuffer { offset, index, .. }
            | RenderStep::FragmentBuffer { offset, index, .. } => {
                bytes.extend(offset.to_le_bytes());
                bytes.extend(index.to_le_bytes());
            }
            // A texture or a sampler a step names is fingerprinted by the case through its own
            // texels or its own descriptor words, as a buffer is, so only the binding index is the
            // call's own parameter here.
            RenderStep::FragmentTexture { index, .. }
            | RenderStep::FragmentSampler { index, .. } => bytes.extend(index.to_le_bytes()),
            // The inline bytes and the viewport are the case's own values rather than a handle, so
            // they are part of what the case feeds its run and the fingerprint covers them whole.
            RenderStep::VertexBytes {
                bytes: inline,
                index,
            } => {
                bytes.extend(inline);
                bytes.extend(index.to_le_bytes());
            }
            RenderStep::Viewport(rect) => bytes.extend(rect.words()),
            RenderStep::VisibilityResultMode { mode, offset } => {
                bytes.extend(mode.raw().to_le_bytes());
                bytes.extend(offset.to_le_bytes());
            }
            RenderStep::Draw {
                primitive,
                first,
                count,
                instances,
            } => {
                bytes.extend(primitive.as_raw().to_le_bytes());
                bytes.extend(first.to_le_bytes());
                bytes.extend(count.to_le_bytes());
                bytes.extend(instances.to_le_bytes());
            }
            RenderStep::DrawIndexed {
                primitive,
                count,
                index_type,
                offset,
                instances,
                base_vertex,
                base_instance,
                ..
            } => {
                bytes.extend(primitive.as_raw().to_le_bytes());
                bytes.extend(count.to_le_bytes());
                bytes.extend(index_type.as_raw().to_le_bytes());
                bytes.extend(offset.to_le_bytes());
                bytes.extend(instances.to_le_bytes());
                bytes.extend(base_vertex.to_le_bytes());
                bytes.extend(base_instance.to_le_bytes());
            }
            RenderStep::DrawIndirect {
                primitive, offset, ..
            } => {
                bytes.extend(primitive.as_raw().to_le_bytes());
                bytes.extend(offset.to_le_bytes());
            }
        }
        bytes
    }
}

/// One render pass: its color attachments by index, and the commands encoded into it.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderPass {
    pub colors: Vec<PassColor>,
    /// `depthAttachment`, where the pass has one.
    pub depth: Option<PassDepth>,
    /// `stencilAttachment`, where the pass has one: the same texture as the depth attachment's,
    /// or another one.
    pub stencil: Option<PassStencil>,
    /// `visibilityResultBuffer`: the buffer the pass's visibility results are written into.
    pub visibility: Option<Handle>,
    /// The commands, in order. **A pass of none is a pass with no draw call**, which is one of the
    /// facts Metal's clear route reads, so the empty list is a case rather than an oversight.
    pub steps: Vec<RenderStep>,
}

impl RenderPass {
    /// A pass over `colors` that encodes nothing: a clear, stored, with no draw call.
    pub fn clearing(colors: Vec<PassColor>) -> Self {
        Self {
            colors,
            depth: None,
            stencil: None,
            visibility: None,
            steps: Vec::new(),
        }
    }

    /// The pass with `steps` encoded into it.
    #[must_use]
    pub fn drawing(mut self, steps: Vec<RenderStep>) -> Self {
        self.steps = steps;
        self
    }

    /// The pass with `depth` as its depth attachment.
    #[must_use]
    pub fn with_depth(mut self, depth: PassDepth) -> Self {
        self.depth = Some(depth);
        self
    }

    /// The pass with `stencil` as its stencil attachment.
    #[must_use]
    pub fn with_stencil(mut self, stencil: PassStencil) -> Self {
        self.stencil = Some(stencil);
        self
    }

    /// The pass whose visibility results are written into `buffer`.
    #[must_use]
    pub fn counting_into(mut self, buffer: Handle) -> Self {
        self.visibility = Some(buffer);
        self
    }

    /// The words a case's input fingerprint covers: every attachment's actions and clear color,
    /// the depth, stencil and visibility attachments **where the pass has them**, and the shape of
    /// every step. A pipeline or a buffer a step names is fingerprinted by the case through the
    /// pipeline's own words and the buffer's own bytes.
    ///
    /// A pass that attaches no depth, no stencil and no visibility buffer writes none of those
    /// three, so a color-only pass fingerprints to the bytes it always has.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for color in &self.colors {
            bytes.extend(color.words());
        }
        if let Some(depth) = &self.depth {
            bytes.push(1);
            bytes.extend(depth.words());
        }
        if let Some(stencil) = &self.stencil {
            bytes.push(2);
            bytes.extend(stencil.words());
        }
        if self.visibility.is_some() {
            bytes.push(3);
        }
        for step in &self.steps {
            bytes.extend(step.call().as_bytes());
            bytes.extend(step.parameters());
        }
        bytes
    }
}

/// One compute dispatch inside a render program's command buffer, so the kernel that reads a
/// target reads it in the command buffer that drew it and one wait covers both.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dispatch {
    pub pipeline: Handle,
    /// The buffers bound at Metal buffer index 0 up: each buffer and the offset its binding starts
    /// at, because a sweep writes each answer at its own place in one buffer.
    pub buffers: Vec<(Handle, u64)>,
    /// The textures bound at Metal texture index 0 up.
    pub textures: Vec<Handle>,
    /// `dispatchThreads:threadsPerThreadgroup:`.
    pub grid: [u32; 3],
    pub threads: [u32; 3],
}

/// One piece of the work a command buffer holds.
#[derive(Clone, Debug, PartialEq)]
pub enum RenderWork {
    /// One render pass, opened with its descriptor and ended.
    Pass(RenderPass),
    /// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:
    /// destinationOffset:destinationBytesPerRow:destinationBytesPerImage:` of level 0's whole
    /// extent, which is how a private target's texels reach the host.
    ReadTexture {
        texture: Handle,
        into: Handle,
        offset: u64,
        row_pitch: u32,
        size: [u32; 2],
    },
    /// One compute dispatch.
    Dispatch(Dispatch),
}

impl RenderWork {
    /// The site name the work's answer is emitted at: the call it makes.
    pub fn call(&self) -> &'static str {
        match self {
            RenderWork::Pass(_) => "renderCommandEncoder",
            RenderWork::ReadTexture { .. } => "copyFromTextureToBuffer",
            RenderWork::Dispatch(_) => "dispatchThreads",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// A handle to name in a step. Which handle a step holds never reaches the fingerprint, so any
    /// one will do here.
    fn handle() -> Handle {
        Handle(0)
    }

    /// Every step this module's four additions brought names a Metal call no other step names, so
    /// a refusal reads as the call we refused and the fingerprint keys the step by its own name.
    #[test]
    fn the_four_added_steps_name_calls_no_other_step_names() {
        let added = [
            RenderStep::FragmentTexture {
                texture: handle(),
                index: 0,
            },
            RenderStep::FragmentSampler {
                sampler: handle(),
                index: 0,
            },
            RenderStep::VertexBytes {
                bytes: Vec::new(),
                index: 0,
            },
            RenderStep::Viewport(ViewportRect::new([0.0; 2], [1.0; 2])),
        ];
        let others = [
            RenderStep::Pipeline(handle()),
            RenderStep::DepthStencilState(handle()),
            RenderStep::StencilReference(0),
            RenderStep::vertex_buffer(handle(), 0),
            RenderStep::fragment_buffer(handle(), 0),
            RenderStep::triangle(),
        ];
        let named: BTreeSet<&str> = added
            .iter()
            .chain(&others)
            .map(|step| step.call())
            .collect();
        assert_eq!(named.len(), added.len() + others.len());
    }

    /// A viewport's six numbers all reach its words, and two viewports are equal exactly when
    /// those words are: `-0.0` is not `0.0` here, which is the bit-exact equality the type claims.
    #[test]
    fn a_viewport_compares_and_fingerprints_by_its_bits() {
        let rect = ViewportRect::new([4.5, 0.0], [1.0, 8.0]);
        assert_eq!(rect.words().len(), 6 * 8);
        assert_eq!(rect, rect);
        for moved in [
            ViewportRect::new([4.5f64.next_up(), 0.0], [1.0, 8.0]),
            ViewportRect::new([4.5, -0.0], [1.0, 8.0]),
            ViewportRect::new([4.5, 0.0], [-1.0, 8.0]),
            ViewportRect {
                depth: [0.0, 0.5],
                ..rect
            },
        ] {
            assert_ne!(rect, moved, "{moved:?}");
            assert_ne!(rect.words(), moved.words(), "{moved:?}");
        }
    }

    /// A viewport step's every number reaches its parameters, so a case whose viewport moved reads
    /// as `Unrecorded` rather than comparing against the old viewport's answer.
    #[test]
    fn a_viewport_step_s_numbers_are_its_parameters() {
        let rect = ViewportRect::new([4.25, 0.0], [1.5, 8.0]);
        assert_eq!(RenderStep::Viewport(rect).parameters(), rect.words());
        let moved = ViewportRect {
            depth: [0.25, 1.0],
            ..rect
        };
        assert_ne!(
            RenderStep::Viewport(rect).parameters(),
            RenderStep::Viewport(moved).parameters()
        );
    }

    /// An inline vertex binding's bytes and its index are both its parameters: the bytes are the
    /// case's own values, not a handle, so a draw whose position moved is a different input.
    #[test]
    fn inline_vertex_bytes_and_their_index_are_its_parameters() {
        let step =
            |bytes: Vec<u8>, index: u32| RenderStep::VertexBytes { bytes, index }.parameters();
        assert_eq!(step(vec![1, 2, 3, 4], 0), vec![1, 2, 3, 4, 0, 0, 0, 0]);
        assert_ne!(step(vec![1, 2, 3, 4], 0), step(vec![1, 2, 3, 5], 0));
        assert_ne!(step(vec![1, 2, 3, 4], 0), step(vec![1, 2, 3, 4], 1));
    }

    /// A fragment texture's and a fragment sampler's binding index is its one parameter, and the
    /// handle is not: the case fingerprints the texels and the sampler's descriptor itself. The two
    /// are told apart by their call names, as every step is.
    #[test]
    fn a_fragment_texture_and_sampler_fingerprint_by_their_index() {
        let texture = |index| {
            RenderStep::FragmentTexture {
                texture: handle(),
                index,
            }
            .parameters()
        };
        let sampler = |index| {
            RenderStep::FragmentSampler {
                sampler: handle(),
                index,
            }
            .parameters()
        };
        assert_eq!(texture(0), 0u32.to_le_bytes());
        assert_ne!(texture(0), texture(1));
        assert_ne!(sampler(0), sampler(1));
        assert_eq!(texture(3), sampler(3));
    }

    /// A pass keys each step by its call name and then its parameters, so adding a variant
    /// renumbers nothing: a pass of the steps the other cases use fingerprints to the same bytes
    /// whether or not this module's four variants exist.
    #[test]
    fn a_step_contributes_its_name_and_its_parameters_and_no_discriminant() {
        let steps = vec![
            RenderStep::fragment_buffer(handle(), 1),
            RenderStep::triangle(),
        ];
        let mut expected = Vec::new();
        for step in &steps {
            expected.extend(step.call().as_bytes());
            expected.extend(step.parameters());
        }
        let pass = RenderPass::clearing(Vec::new()).drawing(steps);
        assert!(pass.words().ends_with(&expected));
    }
}
