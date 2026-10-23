//! A texture descriptor as both sides take it, for the calls that only *ask* about one.
//!
//! `MTLDevice`'s `heapTextureSizeAndAlignWithDescriptor:` is a pure getter: it compiles nothing,
//! allocates nothing and reaches no GPU, and its answer is Metal's own layout arithmetic. That is
//! exactly the shape a conformance case wants — the whole descriptor space is reachable from the
//! CPU, every answer is two numbers, and the whole family can be re-measured on the oracle as often
//! as anybody asks.
//!
//! It is also the family whose numbers were, until now, held by a comment: `runtime_lane`'s
//! `heap_metal_sizes` carries tables of what an Apple M4 answered, with nothing mechanical tying
//! them to a Metal run. A case over this type takes the answer from Metal.
//!
//! The vocabulary is Metal's own enumerations as the translator's public API names them, so a value
//! a case writes reaches `MTLTextureType`, `MTLPixelFormat` and `MTLTextureUsage` by its raw
//! ordinal and reaches the runtime's descriptor unchanged. A case may therefore name one of Apple's
//! private formats (`Unspecialized`, the 4:2:2 pair, the two-plane extended-range pair) and one of
//! Apple's private usage bits as readily as a public one.

use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};

/// The storage mode a case asks a resource for: three of Metal's four modes, and the suite's one
/// spelling of them.
///
/// It lives here because the queried descriptor was the first thing to need it, and it is used
/// wherever a case states where a resource lives — a queried descriptor
/// ([`HeapTextureQuery::storage`]) and a copy-and-fill program's buffers, heaps and textures
/// ([`crate::work`]). One spelling rather than a second enumeration or a `private` flag per type,
/// so a case that sweeps the modes sweeps one list.
///
/// `MTLStorageModeMemoryless` is the one mode no case states: it describes a texture whose contents
/// exist only inside a render pass, which no case of this suite reads back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    /// `MTLStorageModeShared`.
    Shared,
    /// `MTLStorageModeManaged`: the device's own copy, which the host reaches through an explicit
    /// synchronization. A case that reads such a resource reads it through a copy into a `Shared`
    /// one, which needs none.
    Managed,
    /// `MTLStorageModePrivate`.
    Private,
}

impl Storage {
    /// The word the site's name carries.
    pub fn word(self) -> &'static str {
        match self {
            Storage::Shared => "shared",
            Storage::Managed => "managed",
            Storage::Private => "private",
        }
    }
}

/// A descriptor a case asks Metal about: every `MTLTextureDescriptor` field the heap size query
/// reads.
///
/// `allowGPUOptimizedContents` is deliberately **not** a field. It is left at Metal's default, on,
/// in every case of the family: with it off the runtime announces `optimized-depth-stencil-texture`
/// on MoltenVK for a depth or stencil texture, and a case whose subject is Metal's arithmetic
/// should not be carrying a device's announcement as well. A case that wants the property off is a
/// case of its own, and will say so by growing this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapTextureQuery {
    pub texture_type: TextureType,
    pub format: PixelFormat,
    /// `width`, `height` and `depth`, in texels.
    pub size: [u32; 3],
    /// `mipmapLevelCount`.
    pub levels: u32,
    /// `sampleCount`.
    pub samples: u32,
    /// `arrayLength`.
    pub array_length: u32,
    pub usage: TextureUsage,
    pub storage: Storage,
}

impl HeapTextureQuery {
    /// A 2D descriptor of `format`, `width` by `height`, one level, one sample, one slice, read by
    /// shaders, of `storage`.
    pub const fn texture_2d(format: PixelFormat, size: [u32; 2], storage: Storage) -> Self {
        Self {
            texture_type: TextureType::Type2D,
            format,
            size: [size[0], size[1], 1],
            levels: 1,
            samples: 1,
            array_length: 1,
            usage: TextureUsage::SHADER_READ,
            storage,
        }
    }

    /// The descriptor with `texture_type`, and the `height`, `depth` and `arrayLength` that type
    /// takes: a 1D or buffer texture one row tall, an array type `slices` slices, a 3D texture
    /// `depth` texels deep.
    #[must_use]
    pub const fn of_type(mut self, texture_type: TextureType, depth: u32, slices: u32) -> Self {
        self.texture_type = texture_type;
        match texture_type {
            TextureType::Type1D | TextureType::Type1DArray | TextureType::TextureBuffer => {
                self.size[1] = 1;
            }
            TextureType::Type3D => self.size[2] = depth,
            _ => {}
        }
        self.array_length = slices;
        self
    }

    /// The descriptor with `levels` mipmap levels.
    #[must_use]
    pub const fn with_levels(mut self, levels: u32) -> Self {
        self.levels = levels;
        self
    }

    /// The descriptor with `usage`.
    #[must_use]
    pub const fn with_usage(mut self, usage: TextureUsage) -> Self {
        self.usage = usage;
        self
    }

    /// The descriptor with `samples` samples.
    #[must_use]
    pub const fn with_samples(mut self, samples: u32) -> Self {
        self.samples = samples;
        self
    }

    /// The site name the query's answer sits at: every field that can change the answer, so two
    /// descriptors of one family never share a site and a reader of a diff knows which descriptor
    /// moved without counting rows.
    pub fn site(&self) -> String {
        format!(
            "{}/{:?}/{}x{}x{}/levels-{}/samples-{}/slices-{}/usage-{:#x}/{}",
            self.format.name(),
            self.texture_type,
            self.size[0],
            self.size[1],
            self.size[2],
            self.levels,
            self.samples,
            self.array_length,
            self.usage.as_raw(),
            self.storage.word(),
        )
    }

    /// The words the descriptor fingerprints as, for a case's input fingerprint: every field, so a
    /// descriptor that moves is a change to the case's inputs rather than a silent re-point of the
    /// recording.
    pub fn words(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(self.texture_type.as_raw().to_le_bytes());
        out.extend(self.format.as_raw().to_le_bytes());
        for axis in self.size {
            out.extend(axis.to_le_bytes());
        }
        out.extend(self.levels.to_le_bytes());
        out.extend(self.samples.to_le_bytes());
        out.extend(self.array_length.to_le_bytes());
        out.extend(self.usage.as_raw().to_le_bytes());
        // The two modes that were here first keep their bytes, so a recording taken before the
        // third existed is still the answer to the inputs it was taken from.
        out.extend(match self.storage {
            Storage::Shared => [0u8],
            Storage::Private => [1u8],
            Storage::Managed => [2u8],
        });
        out
    }
}

/// `MTLSizeAndAlign`: what the heap size query answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizeAndAlign {
    pub size: u64,
    pub align: u64,
}

impl SizeAndAlign {
    /// The answer as the fields a site records: both numbers, so a diff names which of the two
    /// moved.
    pub fn value(&self) -> crate::transcript::Value {
        crate::transcript::Value::fields([
            ("size", crate::transcript::Value::Uint(self.size)),
            ("align", crate::transcript::Value::Uint(self.align)),
        ])
    }
}
