//! A sparse tile query as both sides take it, for `MTLDevice`'s two sparse tile getters.
//!
//! `sparseTileSizeWithTextureType:pixelFormat:sampleCount:` and its `sparsePageSize:` form answer
//! the tile a sparse texture maps in, in texels; `sparseTileSizeInBytes` and its
//! `ForSparsePageSize:` form answer that tile's bytes. All four are **pure getters**: they compile
//! nothing, allocate nothing, reach no GPU and need no sparse-capable device, so a case over them
//! is safe on every venue and comparable on every venue — including one with no sparse binding at
//! all, which is what makes this the first sparse family the suite can carry end to end.
//!
//! The runtime answers them from a model of Metal's rule (`sparse::tile_texels`), the same on every
//! device at one Metal level, whose comment says it was measured on an Apple M4 and whose only
//! possible oracle is Metal. `runtime_lane::sparse_textures` asserts exactly three of its answers —
//! `(Type2D, R32Float, 1)`, `(Cube, R32Float, 1)` and `(Type2DMultisample, RGBA8Unorm, 2 and 4)` —
//! so the rule's two interesting claims, that the answer does not depend on the texture type and
//! that it halves the tile's sides as the block's bytes double, have never been checked against
//! Metal over a sweep. A case over this type checks them.
//!
//! The vocabulary is Metal's own enumerations as the translator's public API names them, so a value
//! a case writes reaches `MTLTextureType`, `MTLPixelFormat` and `MTLSparsePageSize` by its raw
//! ordinal and reaches the runtime's call unchanged.

use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan_runtime::SparsePageSize;

/// The page a query asks in: one of Metal's three, or none, which is the getter without a page
/// argument (whose answer Metal documents as the 16-KiB page's).
///
/// `None` is a query of its own rather than a spelling of `Some(Size16)`, because the two are two
/// Metal selectors and the suite's business is what each selector answers. A release where they
/// part fails the case rather than passing under a rule nobody re-measured.
pub type QueryPage = Option<SparsePageSize>;

/// The word a page carries in a site's name.
pub fn page_word(page: QueryPage) -> &'static str {
    match page {
        None => "default-page",
        Some(SparsePageSize::Size16) => "page-16",
        Some(SparsePageSize::Size64) => "page-64",
        Some(SparsePageSize::Size256) => "page-256",
        // `SparsePageSize` is `#[non_exhaustive]`: a page Metal gains is named by its raw value
        // rather than silently sharing a site with one of these.
        Some(page) => match page.raw() {
            101 => "page-16",
            102 => "page-64",
            103 => "page-256",
            _ => "page-unknown",
        },
    }
}

/// The bytes a page fingerprints as.
pub fn page_words(page: QueryPage) -> Vec<u8> {
    match page {
        None => vec![0u8; 9],
        Some(page) => {
            let mut out = vec![1u8];
            out.extend(page.raw().to_le_bytes());
            out
        }
    }
}

/// One sparse tile query: the three arguments of
/// `sparseTileSizeWithTextureType:pixelFormat:sampleCount:`, and the page it asks in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseTileQuery {
    pub texture_type: TextureType,
    pub format: PixelFormat,
    /// `sampleCount`. A case states one the texture type takes: 1 for a single-sampled type, and
    /// for a multisample type a count the oracle's device holds.
    pub samples: u32,
    pub page: QueryPage,
}

impl SparseTileQuery {
    /// The query of `texture_type`, `format` and `samples` in the page `page`.
    pub const fn new(
        texture_type: TextureType,
        format: PixelFormat,
        samples: u32,
        page: QueryPage,
    ) -> Self {
        Self {
            texture_type,
            format,
            samples,
            page,
        }
    }

    /// The site name the query's answer sits at: every argument, so two queries of one family
    /// never share a site and a reader of a diff knows which query moved without counting rows.
    pub fn site(&self) -> String {
        format!(
            "{}/{:?}/samples-{}/{}",
            self.format.name(),
            self.texture_type,
            self.samples,
            page_word(self.page),
        )
    }

    /// The words the query fingerprints as, for a case's input fingerprint: every argument, so a
    /// query that moves is a change to the case's inputs rather than a silent re-point of the
    /// recording.
    pub fn words(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(self.texture_type.as_raw().to_le_bytes());
        out.extend(self.format.as_raw().to_le_bytes());
        out.extend(self.samples.to_le_bytes());
        out.extend(page_words(self.page));
        out
    }
}

/// `MTLSize`: what a sparse tile getter answers, in texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileSize {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

impl TileSize {
    /// The answer as the fields a site records: all three axes, so a diff names which one moved.
    pub fn value(&self) -> crate::transcript::Value {
        crate::transcript::Value::fields([
            ("width", crate::transcript::Value::Uint(self.width)),
            ("height", crate::transcript::Value::Uint(self.height)),
            ("depth", crate::transcript::Value::Uint(self.depth)),
        ])
    }
}
