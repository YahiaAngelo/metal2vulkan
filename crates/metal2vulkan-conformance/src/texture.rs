//! A texture a case describes whole: `MTLTextureDescriptor` as the format families need one.
//!
//! [`crate::case::TextureSpec`] says "a one-level 2D texture of shared storage, in this role, at
//! this row pitch", which is what the first cases needed. The format families need the rest of the
//! descriptor: a level count, an array's slices, a cube's faces, private storage,
//! `allowGPUOptimizedContents`, a swizzle, and a block-compressed format whose level 0 is loaded
//! from block bytes rather than texel bytes. They also load more than one level, so the bytes a
//! texture starts from are a list of `replaceRegion:` calls instead of one.
//!
//! The vocabulary is Metal's own, as the translator's public API names it
//! (`MTLTextureType`, `MTLTextureUsage`, `MTLTextureSwizzleChannels`), so a value a case writes
//! reaches both sides' descriptors unchanged and there is no second spelling to keep in step. The
//! format is a raw `MTLPixelFormat`, as elsewhere in the suite, so a case may name one of Apple's
//! private formats as readily as a public one.
//!
//! Nothing here reads a device name, a vendor or an environment variable: a case that asks for a
//! texture the venue cannot make finds out from the typed refusal the call answers.

use metal2vulkan::metal::{TextureSwizzleChannels, TextureType, TextureUsage};

/// One `replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:` a texture is loaded
/// with as it is made.
///
/// A compressed format states its pitches in **blocks**, as Metal takes them, while its origin and
/// size stay in texels: that difference is the whole subject of the compressed families, so the
/// case states both rather than either side computing one from the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Upload<'a> {
    pub level: u32,
    pub slice: u32,
    /// The region's origin in texels.
    pub origin: [u32; 3],
    /// The region's size in texels.
    pub size: [u32; 3],
    pub bytes: &'a [u8],
    /// `bytesPerRow`: bytes from one row of blocks to the next.
    pub row_pitch: u32,
    /// The rows of bytes the upload hands, which is the region's **block** rows: a side checks
    /// `bytes` holds them before the call reads, and a compressed format's block rows are not its
    /// texel rows, which is exactly the difference these families are about.
    pub rows: u32,
    /// `bytesPerImage`: bytes from one image to the next; 0 for a texture of one image.
    pub image_pitch: u32,
}

impl Upload<'_> {
    /// Level 0, slice 0, the whole of a 2D extent, `rows` rows of `row_pitch` bytes.
    pub const fn level_0(size: [u32; 2], row_pitch: u32, rows: u32, bytes: &[u8]) -> Upload<'_> {
        Upload {
            level: 0,
            slice: 0,
            origin: [0, 0, 0],
            size: [size[0], size[1], 1],
            bytes,
            row_pitch,
            rows,
            image_pitch: 0,
        }
    }

    /// The bytes the upload must hold for the call to read inside them.
    pub const fn needed(&self) -> usize {
        self.row_pitch as usize * self.rows as usize
    }

    /// The words the upload's parameters fingerprint as, beside its bytes.
    pub fn words(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for word in [self.level, self.slice] {
            out.extend(word.to_le_bytes());
        }
        for axis in self.origin.iter().chain(&self.size) {
            out.extend(axis.to_le_bytes());
        }
        for word in [self.row_pitch, self.rows, self.image_pitch] {
            out.extend(word.to_le_bytes());
        }
        out.extend(self.bytes);
        out
    }
}

/// A texture a case makes, as `MTLTextureDescriptor` describes one, with the bytes it starts from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureRequest<'a> {
    /// The raw `MTLPixelFormat`, so a case may name one of Apple's private formats.
    pub format: u32,
    pub texture_type: TextureType,
    /// `width`, `height` and `depth`, in texels.
    pub size: [u32; 3],
    /// `mipmapLevelCount`.
    pub levels: u32,
    /// `arrayLength`: an array type's slices, 1 otherwise. A cube's six faces are its slices and
    /// not its array length, as Metal counts them.
    pub slices: u32,
    pub usage: TextureUsage,
    /// Whether the storage is private; shared otherwise, so the host reads it back.
    pub private: bool,
    /// `allowGPUOptimizedContents`.
    pub gpu_optimized: bool,
    pub swizzle: TextureSwizzleChannels,
    /// The `replaceRegion:` calls the texture is loaded with, in order.
    pub uploads: &'a [Upload<'a>],
}

impl<'a> TextureRequest<'a> {
    /// A one-level 2D texture of `format`, `width` by `height`, read by a shader, in shared
    /// storage, GPU-optimized as Metal's descriptor is by default, with the default swizzle and
    /// nothing loaded.
    pub const fn read_2d(format: u32, width: u32, height: u32) -> Self {
        Self {
            format,
            texture_type: TextureType::Type2D,
            size: [width, height, 1],
            levels: 1,
            slices: 1,
            usage: TextureUsage::SHADER_READ,
            private: false,
            gpu_optimized: true,
            swizzle: TextureSwizzleChannels::DEFAULT,
            uploads: &[],
        }
    }

    /// The same texture with `levels` mip levels.
    #[must_use]
    pub const fn levels(mut self, levels: u32) -> Self {
        self.levels = levels;
        self
    }

    /// The same texture in private storage, which no host call reaches.
    #[must_use]
    pub const fn private(mut self) -> Self {
        self.private = true;
        self
    }

    /// The same texture in `usage`.
    #[must_use]
    pub const fn used(mut self, usage: TextureUsage) -> Self {
        self.usage = usage;
        self
    }

    /// The same texture as a cube of `side` texels a face.
    #[must_use]
    pub const fn cube(mut self, side: u32) -> Self {
        self.texture_type = TextureType::Cube;
        self.size = [side, side, 1];
        self
    }

    /// The same texture read through `swizzle`.
    #[must_use]
    pub const fn swizzled(mut self, swizzle: TextureSwizzleChannels) -> Self {
        self.swizzle = swizzle;
        self
    }

    /// The same texture loaded with `uploads`.
    #[must_use]
    pub const fn loaded(mut self, uploads: &'a [Upload<'a>]) -> Self {
        self.uploads = uploads;
        self
    }

    /// The words the request fingerprints as: every field of the descriptor and every upload, so a
    /// texture whose shape or starting bytes move is a change to the case's inputs rather than a
    /// quiet pass against a recording of other bytes.
    pub fn words(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(self.format.to_le_bytes());
        out.extend(self.texture_type.as_raw().to_le_bytes());
        for axis in &self.size {
            out.extend(axis.to_le_bytes());
        }
        for word in [self.levels, self.slices, self.usage.as_raw()] {
            out.extend(word.to_le_bytes());
        }
        out.push(u8::from(self.private));
        out.push(u8::from(self.gpu_optimized));
        for channel in [
            self.swizzle.red,
            self.swizzle.green,
            self.swizzle.blue,
            self.swizzle.alpha,
        ] {
            out.extend(channel.as_raw().to_le_bytes());
        }
        for upload in self.uploads {
            out.extend(upload.words());
        }
        out
    }
}
