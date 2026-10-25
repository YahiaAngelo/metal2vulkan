//! What `heapTextureSizeAndAlignWithDescriptor:` counts, for the formats this crate holds no
//! image of.
//!
//! The heap size query is **arithmetic over a descriptor**: how many bytes Metal's own layout
//! takes for a texture of that shape, and at what alignment. It makes nothing and reads nothing.
//! So a format can have a footprint this crate answers exactly while having no image this crate
//! can make a texture, a view or a copy in: the two are different questions, and
//! `repr`'s `NotImplemented` answers only the second.
//!
//! [`footprint`] is the first question for the formats whose answer the format's own structure
//! decides — its block's bytes and texel extent, and how many planes Metal lays its bytes in —
//! and `None` for every format whose footprint the image this crate maps it to decides, which
//! `metal2vulkan-runtime`'s `texture/heap_tiled.rs` reads from that image instead. The two never
//! answer for one format (`tests::footprint_answers_where_no_image_does`), and every byte count
//! here is held to the bytes of a format the crate **does** hold whose element is the same size
//! (`tests::footprint_bytes_match_a_held_twin`), so no number here is one nobody checked.
//!
//! Measured on an Apple M3 Max under macOS 27.0.1 (26A434), plain Metal with no API validation
//! layer: the `format-footprints-*` conformance recordings, 1581 (format, descriptor) answers
//! over all 33 formats, and the probe sweep these rules were fitted against, which the runtime's
//! `tests::heap_footprint_formats` replays.

use super::block_extent;
use crate::metal::PixelFormat;

/// The bytes of one ASTC block, whatever its texel extent.
const ASTC_BLOCK: u32 = 16;

/// How Metal lays out the bytes `heapTextureSizeAndAlignWithDescriptor:` counts for a texture of
/// a format whose footprint the format's own structure decides ([`footprint`]).
///
/// Every variant is Metal's layout, not a device's: the same answer on every device at the same
/// Metal level, because nothing about it depends on an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Footprint {
    /// Elements of `bytes` bytes, each covering `block` texels, in Metal's tiles: the layout of
    /// every plain and block-compressed format, which the runtime's tiled model counts
    /// (`texture/heap_tiled.rs`). `block` is `[1, 1]` where an element is one texel.
    Tiled {
        /// The bytes of one element: a texel, or a compressed format's block.
        bytes: u32,
        /// The texels one element covers ([`block_extent`]).
        block: [u32; 2],
    },
    /// One linear plane of `bytes`-byte texels, packed: no row padding, no tiles and no
    /// lossless-compression metadata. Apple's one-plane video formats lie in a surface's rows,
    /// which is why their alignment is 16 and not 128.
    OnePlane {
        /// The bytes of one texel.
        bytes: u32,
        /// Whether two texels across share one chroma pair (4:2:2 rather than 4:4:4). Metal makes
        /// a writable or renderable texture of such a format only with both sides even, and the
        /// runtime holds that rule (`texture_descriptor.rs`).
        pairs: bool,
    },
    /// Two planes, a 4-byte colour word a texel and an alpha byte a texel, the colour plane
    /// rounded up to 16 KiB before the alpha plane: `RGB10A8_2P`'s shape, which the runtime
    /// already answers for the extended-range twins of it
    /// ([`ExtendedRange::Planes`](super::ExtendedRange::Planes)).
    TwoPlanes,
    /// Metal describes the format, takes a descriptor naming it, and makes no texture of it in
    /// any type: the query answers 0 bytes.
    NoTexture,
}

/// Metal's footprint layout of `pf`, where the format's own structure decides it and no image of
/// this crate does: the 32 formats Metal answers `heapTextureSizeAndAlignWithDescriptor:` for
/// that `repr` holds no Vulkan candidate of (`astc_without_a_vulkan_block!` and
/// `formats_named_not_held!`). `None` for every other format, whose footprint its image decides.
///
/// This is **not** support for the format. A texture, a view, a copy, a read and a write of one
/// stay refused `NotImplemented(Format)`, because this crate holds no image of it; a footprint is
/// counted bytes, which needs none.
pub fn footprint(pf: PixelFormat) -> Option<Footprint> {
    use PixelFormat as P;
    // ASTC's 4x2 and 8x4 footprints, LDR, sRGB and HDR: an ASTC block is 16 bytes whatever its
    // texel extent, and `block_extent` already answers theirs, so their footprint is the block
    // layout every other ASTC format's is. That no Vulkan ASTC block has either extent is why
    // the crate holds no image of them, not a reason Metal cannot count their bytes.
    if matches!(
        pf,
        P::ASTC4x2LDR
            | P::ASTC4x2SRGB
            | P::ASTC4x2HDR
            | P::ASTC8x4LDR
            | P::ASTC8x4SRGB
            | P::ASTC8x4HDR
    ) {
        return Some(Footprint::Tiled {
            bytes: ASTC_BLOCK,
            block: block_extent(pf)?,
        });
    }
    let tiled = |bytes| {
        Some(Footprint::Tiled {
            bytes,
            block: [1, 1],
        })
    };
    match pf {
        // `ARGB8Unorm_sRGB`, an ordinary texel in an ordinary layout: a byte a channel
        // (`ARGB8Unorm`'s texel). Its sibling straggler `BGRA4Unorm` has an image of its own
        // now, so its footprint is that image's and not this table's.
        P::ARGB8UnormSRGB => tiled(4),
        // The padded 10- and 12-bit display formats: one 16-bit word a channel, the code in the
        // word's top bits and the low bits unread (`R10Unorm_X6`'s texel, `RG10Unorm_X12`'s), and
        // the transfer-function twins of `RGB10A2Unorm` and `BGR10A2Unorm`, one 32-bit packed
        // word. A transfer function changes how a code decodes, never how many bytes the texel
        // takes, so each twin's footprint is its base format's.
        P::R12UnormX4 | P::R12UnormX4PQ | P::R12UnormX4HLG => tiled(2),
        P::R10UnormX6PQ | P::R10UnormX6HLG => tiled(2),
        P::RG12UnormX8 => tiled(4),
        P::RGB10A2UnormPQ | P::RGB10A2UnormHLG => tiled(4),
        P::BGR10A2UnormSRGB | P::BGR10A2UnormPQ => tiled(4),
        // Apple's **one-plane** YCbCr formats, 4:2:2 and 4:4:4: the codes of one surface plane in
        // rows, two bytes a texel at 4:2:2 (`GBGR422`'s texel, a chroma pair in two texels) and
        // four at 4:4:4 (a 10-bit code triple in a 32-bit word; `YCBCRA8_444_1P` a byte a
        // channel). Their bytes are a plane's rows and not tiles, which is why they answer
        // `OnePlane` and their alignment is 16.
        P::YCBCR84221P | P::YCBCR84221PSRGB | P::YCBCR84221PREV => Some(Footprint::OnePlane {
            bytes: 2,
            pairs: true,
        }),
        P::YCBCR104441P
        | P::YCBCR104441PSRGB
        | P::YCBCR104441PPQ
        | P::YCBCR104441PXR
        | P::YCBCR104441PHLG
        | P::YCBCRA84441P => Some(Footprint::OnePlane {
            bytes: 4,
            pairs: false,
        }),
        // Apple's two-plane 10-bit-plus-alpha format and its three transfer-function twins: the
        // same two planes as `RGB10A8_2P_XR10`, a 4-byte colour word a texel and an alpha byte a
        // texel.
        P::RGB10A82P | P::RGB10A82PSRGB | P::RGB10A82PPQ | P::RGB10A82PHLG => {
            Some(Footprint::TwoPlanes)
        }
        // `R10Unorm_PACKED` and `RG10Unorm_PACKED`: Metal takes a descriptor naming either and
        // makes no texture of it in any type in a mode that holds memory, so the query answers 0
        // bytes. A packed 10-bit code crosses byte boundaries, so there is no element size to count
        // and Metal counts none. `Memoryless` storage is the one mode where
        // `newTextureWithDescriptor:` answers a texture of either, which is why this crate's
        // refusal there names the format and is its own gap rather than Metal's nil.
        P::R10UnormPACKED | P::RG10UnormPACKED => Some(Footprint::NoTexture),
        _ => None,
    }
}
