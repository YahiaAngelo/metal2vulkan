//! The copy-and-fill vocabulary: a case's resources and the steps it encodes, as data.
//!
//! The five lanes this batch ports all say the same kind of thing — make a texture, write known
//! bytes into it, run one blit, read every byte back — so the vocabulary they need is one
//! declarative program rather than sixteen trait methods. A case states its [`Work`]; a side runs
//! it and answers what it read ([`Done`]). The case body stays a description of the probe, which
//! is what makes a port readable beside the lane it came from.
//!
//! A side with no public call for a step answers [`crate::case::Failure::unavailable`] for that
//! step. On the Metal side that is the **fill** and the fill alone: `fillTexture:` and
//! `fillTextureColor:` are Apple SPI ([`crate::metal_work`]). The *formats* a case names are not
//! private — a pixel format is a raw value in `MTLTextureDescriptor`, and Metal makes, writes,
//! blits and reads back Apple's two-plane extended-range 550 and 551 through public calls like any
//! other. So a program that fills carries its Metal answer
//! ([`crate::case::MetalSource::Carried`]); a program of copies records on Metal.
//!
//! Nothing here names a device, a vendor or an environment variable. A resource the venue cannot
//! make comes back as the runtime's typed refusal, and the suite reads the venue's honest gap out
//! of that refusal rather than out of a device name.

/// A rectangle of a texture: an origin and a size, in texels, with depth so a case can state a
/// region of no images.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub origin: [u64; 3],
    pub size: [u64; 3],
}

impl Rect {
    /// A 2D rectangle of one image.
    pub const fn new(x: u64, y: u64, width: u64, height: u64) -> Self {
        Self {
            origin: [x, y, 0],
            size: [width, height, 1],
        }
    }

    /// A rectangle with every axis stated.
    pub const fn new_3d(origin: [u64; 3], size: [u64; 3]) -> Self {
        Self { origin, size }
    }
}

/// How a texture is shaped. The format is its raw `MTLPixelFormat`, because a case states the
/// probe's number and some of those numbers have no name in a public header. A number with no
/// public name is still a number Metal takes: `MTLTextureDescriptor`'s pixel format is a raw
/// value, so both sides make, write, blit and read such a texture through public calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureSpec {
    /// The raw `MTLPixelFormat`.
    pub format: u32,
    pub kind: TextureKind,
    pub width: u64,
    pub height: u64,
    pub levels: u64,
    pub slices: u64,
    pub samples: u64,
    /// Whether the texture's storage is private; shared otherwise.
    pub private: bool,
    /// Whether the texture is a render target as well as read and viewed.
    pub render_target: bool,
}

impl TextureSpec {
    /// A one-level 2D texture read by a shader.
    pub const fn texture_2d(format: u32, width: u64, height: u64) -> Self {
        Self {
            format,
            kind: TextureKind::Type2D,
            width,
            height,
            levels: 1,
            slices: 1,
            samples: 1,
            private: true,
            render_target: false,
        }
    }

    /// The same texture with `levels` mip levels.
    #[must_use]
    pub const fn levels(mut self, levels: u64) -> Self {
        self.levels = levels;
        self
    }

    /// The same texture as a 2D array of `slices`.
    #[must_use]
    pub const fn array(mut self, slices: u64) -> Self {
        self.kind = TextureKind::Type2DArray;
        self.slices = slices;
        self
    }

    /// The same texture in shared storage.
    #[must_use]
    pub const fn shared(mut self) -> Self {
        self.private = false;
        self
    }

    /// The same texture as a multisample one of `samples`, a render target that may be viewed in
    /// another format (what a per-sample read-back needs).
    #[must_use]
    pub const fn multisample(mut self, samples: u64, slices: u64) -> Self {
        self.kind = if slices > 1 {
            TextureKind::Type2DMultisampleArray
        } else {
            TextureKind::Type2DMultisample
        };
        self.samples = samples;
        self.slices = if slices > 1 { slices } else { 1 };
        self.render_target = true;
        self
    }
}

/// Which kind of texture a spec names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureKind {
    Type2D,
    Type2DArray,
    Type2DMultisample,
    Type2DMultisampleArray,
    /// A texture over a buffer's bytes; its width is its texels and its height is 1.
    TextureBuffer,
}

/// Which plane of a combined format a copy moves, as Metal's `MTLBlitOption` names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plane {
    /// Every byte of the texture's own format.
    Whole,
    Depth,
    Stencil,
}

/// A resource a case makes. Each is named by its index in [`Work::resources`], so a step refers to
/// one by position and nothing in a transcript holds an object's identity.
#[derive(Clone, Debug, PartialEq)]
pub enum Resource {
    /// A buffer holding `bytes`, in shared storage so the host reads it back.
    Buffer { bytes: Vec<u8> },
    /// A buffer of `length` bytes of **private** storage, which no host call reaches.
    ///
    /// Its own variant rather than a flag on [`Resource::Buffer`], because a private buffer has no
    /// bytes to be made with: a case that wants known bytes in one writes them with a
    /// [`Step::FillBuffer`] or a copy, and that is part of what such a case means. Nothing here
    /// assumes what a fresh buffer holds — Metal does not promise it — so a case over private
    /// memory states its starting bytes rather than reading whatever the allocator left.
    PrivateBuffer { length: u64 },
    /// A placement heap of `size` bytes.
    Heap { size: u64, private: bool },
    /// A buffer over `length` bytes of `heap` from `offset`.
    HeapBuffer {
        heap: usize,
        length: u64,
        offset: u64,
    },
    /// A texture of its own memory.
    Texture { spec: TextureSpec },
    /// A placement heap exactly as large as one texture of `spec` needs, as Metal sizes it: what
    /// a probe that placed one texture in a heap of Metal's size made.
    HeapForTexture { spec: TextureSpec, private: bool },
    /// A texture placed in `heap` at `offset`.
    HeapTexture {
        heap: usize,
        spec: TextureSpec,
        offset: u64,
    },
    /// A texture over `buffer`'s bytes from `offset`, its rows `bytes_per_row` apart.
    TextureOverBuffer {
        buffer: usize,
        spec: TextureSpec,
        offset: u64,
        bytes_per_row: u64,
    },
    /// A view of `texture`'s `slice` in `format`, as a 2D multisample texture where `flatten`.
    View {
        texture: usize,
        format: u32,
        slice: u64,
        flatten: bool,
    },
}

/// One step of a case's program.
///
/// The copies and fills are encoded into one blit encoder; [`Step::Commit`] closes it, commits the
/// command buffer and waits for it. A read is answered in the order the steps asked for it,
/// whether it is a host call or a read-back the device had to run first.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// `fillTexture:`'s private fill of `pattern`.
    FillTexture {
        texture: usize,
        level: u64,
        slice: u64,
        region: Rect,
        pattern: Vec<u8>,
    },
    /// `fillTexture:`'s private fill of a color packed in `packed_in`, or in the texture's own
    /// format where it is `None`.
    FillTextureColor {
        texture: usize,
        level: u64,
        slice: u64,
        region: Rect,
        color: [f64; 4],
        packed_in: Option<u32>,
    },
    CopyBufferToTexture {
        buffer: usize,
        offset: u64,
        bytes_per_row: u64,
        bytes_per_image: u64,
        size: [u64; 3],
        texture: usize,
        slice: u64,
        level: u64,
        origin: [u64; 3],
        plane: Plane,
    },
    CopyTextureToBuffer {
        texture: usize,
        slice: u64,
        level: u64,
        origin: [u64; 3],
        size: [u64; 3],
        buffer: usize,
        offset: u64,
        bytes_per_row: u64,
        bytes_per_image: u64,
        plane: Plane,
    },
    CopyTextureToTexture {
        from: usize,
        from_slice: u64,
        from_level: u64,
        from_origin: [u64; 3],
        size: [u64; 3],
        to: usize,
        to_slice: u64,
        to_level: u64,
        to_origin: [u64; 3],
    },
    CopyBufferToBuffer {
        from: usize,
        from_offset: u64,
        to: usize,
        to_offset: u64,
        length: u64,
    },
    /// `fillBuffer:range:value:`: every byte of `[offset, offset + length)` takes `value`.
    ///
    /// A fill of no bytes is a step of its own and not nothing: plain Metal skips it wherever its
    /// range lies, and whether we do is a case.
    FillBuffer {
        buffer: usize,
        offset: u64,
        length: u64,
        value: u8,
    },
    /// A render pass over one slice that clears it and draws nothing: how Metal writes a known
    /// value into a multisample texture, which no CPU copy reaches.
    ClearPass {
        texture: usize,
        slice: u64,
        clear: Clear,
    },
    /// `replaceRegion:`: the host writes `bytes` into the texture.
    ReplaceRegion {
        texture: usize,
        level: u64,
        slice: u64,
        region: Rect,
        bytes: Vec<u8>,
        bytes_per_row: u64,
        bytes_per_image: u64,
    },
    /// `getBytes:`: the host reads the region into `length` bytes, which answers a read.
    GetBytes {
        texture: usize,
        level: u64,
        slice: u64,
        region: Rect,
        bytes_per_row: u64,
        bytes_per_image: u64,
        length: usize,
    },
    /// A read of a buffer's bytes, which answers a read.
    ReadBuffer {
        buffer: usize,
        offset: u64,
        length: usize,
    },
    /// Every sample of a multisample texture, read by a dispatch of the committed fixture's
    /// kernel, one thread a sample: the texels in rows, each texel's samples in order, `texel_bytes`
    /// each. It answers a read.
    ReadSamples {
        texture: usize,
        kernel: &'static str,
        width: u32,
        height: u32,
        samples: u32,
        /// The bytes of one sample the read keeps: the kernel writes whole words, and a format of
        /// fewer bytes keeps its own.
        texel_bytes: usize,
        /// The words the kernel writes per sample.
        words: usize,
    },
    /// Close the blit encoder, commit the command buffer and wait for it: its status is recorded
    /// and every read-back the steps before it asked for is answered.
    Commit,
}

/// What a [`Step::ClearPass`] clears.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Clear {
    Color([f64; 4]),
    Depth(f64),
    Stencil(u32),
    /// A combined depth and stencil attachment, both cleared in one pass.
    DepthStencil(f64, u32),
}

/// A case's program: the resources it makes, in order, and the steps it runs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Work {
    pub resources: Vec<Resource>,
    pub steps: Vec<Step>,
    /// The fixture whose kernels a [`Step::ReadSamples`] runs, where the program holds one.
    pub kernels: Option<&'static str>,
}

impl Work {
    /// An empty program.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `resource`, answering its index.
    pub fn add(&mut self, resource: Resource) -> usize {
        self.resources.push(resource);
        self.resources.len() - 1
    }

    /// Add `step`.
    pub fn step(&mut self, step: Step) {
        self.steps.push(step);
    }

    /// Add a buffer holding `bytes`, answering its index.
    pub fn buffer(&mut self, bytes: Vec<u8>) -> usize {
        self.add(Resource::Buffer { bytes })
    }

    /// How many reads the program asks for: a step that answers one.
    pub fn reads(&self) -> usize {
        self.steps.iter().filter(|step| step.reads()).count()
    }
}

impl Step {
    /// Whether the step answers a read.
    pub fn reads(&self) -> bool {
        matches!(
            self,
            Step::GetBytes { .. } | Step::ReadBuffer { .. } | Step::ReadSamples { .. }
        )
    }
}

/// What running a [`Work`] answered: the reads, in the order the steps asked for them, and each
/// command buffer's status, in commit order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Done {
    pub reads: Vec<Vec<u8>>,
    pub statuses: Vec<String>,
}

impl Done {
    /// Read `index`, or an error naming what the program asked for.
    pub fn read(&self, index: usize) -> Result<&[u8], String> {
        self.reads.get(index).map(Vec::as_slice).ok_or_else(|| {
            format!(
                "the run answered {} reads, not {}",
                self.reads.len(),
                index + 1
            )
        })
    }
}
