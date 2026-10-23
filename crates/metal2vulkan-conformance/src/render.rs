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
//! Three things a case states, and nothing else:
//!
//! - **A target** ([`TargetSpec`]) is private, because that is what a render target of one of
//!   Apple's extended-range or bfloat formats is made with, and because Metal's clear route reads
//!   the texture's usage and extent rather than where it lives. A private target is read back with
//!   [`RenderWork::ReadTexture`], the blit every host can run.
//! - **A pipeline** ([`RenderPipelineSpec`]) names its two functions, the format and blend of each
//!   color attachment, and its sample count.
//! - **The work** ([`RenderWork`]) is a list, and the whole list goes into **one** command buffer,
//!   committed and waited for once. That is deliberate: the clear sweeps ask Metal hundreds of
//!   one-texel questions, and one command buffer a question would cost hundreds of round trips on
//!   both sides. A case that wants several command buffers calls [`crate::case::Side::render`] more
//!   than once, so the batching is the case's to state rather than the vocabulary's to assume.
//!
//! Every enumeration here is Metal's own as the translator and the runtime name it, so a value a
//! case writes reaches `MTLBlendFactor`, `MTLBlendOperation`, `MTLLoadAction`, `MTLStoreAction` and
//! `MTLPrimitiveType` by its raw ordinal on the Metal side and reaches the runtime's descriptor
//! unchanged. There is no second spelling to keep in step.

use crate::case::Handle;
use metal2vulkan::metal::{BlendFactor, BlendOperation, PixelFormat, PrimitiveType};
use metal2vulkan_runtime::{LoadAction, StoreAction};

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
        }
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
    /// `blendingEnabled` and the six factors, `None` for blending off.
    pub blend: Option<Blend>,
}

impl ColorState {
    /// An attachment of `format` with blending off, Metal's default.
    pub const fn new(format: PixelFormat) -> Self {
        Self {
            format,
            blend: None,
        }
    }

    /// The attachment with `blend` enabled.
    #[must_use]
    pub const fn blended(mut self, blend: Blend) -> Self {
        self.blend = Some(blend);
        self
    }

    /// The words a case's input fingerprint covers.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = self.format.as_raw().to_le_bytes().to_vec();
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

/// A vertex-and-fragment render pipeline a case makes:
/// `newRenderPipelineStateWithDescriptor:error:`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderPipelineSpec<'a> {
    pub vertex: &'a str,
    pub fragment: &'a str,
    /// The color attachments by index; `None` at an index of `MTLPixelFormatInvalid`, which is how
    /// a pipeline that writes attachment 1 alone is stated.
    pub colors: &'a [Option<ColorState>],
    /// `rasterSampleCount`.
    pub samples: u32,
}

impl<'a> RenderPipelineSpec<'a> {
    /// A single-sampled pipeline of `vertex` and `fragment` over `colors`.
    pub const fn new(vertex: &'a str, fragment: &'a str, colors: &'a [Option<ColorState>]) -> Self {
        Self {
            vertex,
            fragment,
            colors,
            samples: 1,
        }
    }

    /// The pipeline at `samples` samples.
    #[must_use]
    pub const fn multisampled(mut self, samples: u32) -> Self {
        self.samples = samples;
        self
    }

    /// The words a case's input fingerprint covers: the functions, every attachment and the sample
    /// count.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = self.vertex.as_bytes().to_vec();
        bytes.push(0);
        bytes.extend(self.fragment.as_bytes());
        bytes.push(0);
        for color in self.colors {
            match color {
                Some(state) => bytes.extend(state.words()),
                None => bytes.push(0xff),
            }
        }
        bytes.extend(self.samples.to_le_bytes());
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

/// One command a case encodes into a render pass, in the order it lists them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenderStep {
    /// `setRenderPipelineState:`.
    Pipeline(Handle),
    /// `setFragmentBuffer:offset:atIndex:`.
    FragmentBuffer {
        buffer: Handle,
        offset: u64,
        index: u32,
    },
    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:`, whose base instance
    /// is 0.
    Draw {
        primitive: PrimitiveType,
        first: u32,
        count: u32,
        instances: u32,
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

    /// The site name the step's answer is emitted at: the call it makes, so a refusal reads as the
    /// Metal call we refused.
    pub fn call(&self) -> &'static str {
        match self {
            RenderStep::Pipeline(_) => "setRenderPipelineState",
            RenderStep::FragmentBuffer { .. } => "setFragmentBuffer",
            RenderStep::Draw { .. } => "drawPrimitives",
        }
    }
}

/// One render pass: its color attachments by index, and the commands encoded into it.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderPass {
    pub colors: Vec<PassColor>,
    /// The commands, in order. **A pass of none is a pass with no draw call**, which is one of the
    /// facts Metal's clear route reads, so the empty list is a case rather than an oversight.
    pub steps: Vec<RenderStep>,
}

impl RenderPass {
    /// A pass over `colors` that encodes nothing: a clear, stored, with no draw call.
    pub fn clearing(colors: Vec<PassColor>) -> Self {
        Self {
            colors,
            steps: Vec::new(),
        }
    }

    /// The pass with `steps` encoded into it.
    #[must_use]
    pub fn drawing(mut self, steps: Vec<RenderStep>) -> Self {
        self.steps = steps;
        self
    }

    /// The words a case's input fingerprint covers: every attachment's actions and clear color,
    /// and the shape of every step. A pipeline or a buffer a step names is fingerprinted by the
    /// case through the pipeline's own words and the buffer's own bytes.
    pub fn words(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for color in &self.colors {
            bytes.extend(color.words());
        }
        for step in &self.steps {
            bytes.extend(step.call().as_bytes());
            if let RenderStep::Draw {
                primitive,
                first,
                count,
                instances,
            } = step
            {
                bytes.extend(primitive.as_raw().to_le_bytes());
                bytes.extend(first.to_le_bytes());
                bytes.extend(count.to_le_bytes());
                bytes.extend(instances.to_le_bytes());
            }
            if let RenderStep::FragmentBuffer { offset, index, .. } = step {
                bytes.extend(offset.to_le_bytes());
                bytes.extend(index.to_le_bytes());
            }
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
