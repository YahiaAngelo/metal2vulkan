//! Texture buffers' texel buffer views (`buffer_texture`): the usage a view takes beside its
//! buffer's, by the kinds of texel buffer its format serves on the device; a staged shadow's
//! regions back into its buffer; and Metal's `bytesPerRow` rule for a texture over an IOSurface
//! plane, which no test may build the texture for.

use super::super::buffer_texture::{staged_rows_back, turned_plane, view_usage, ViewUsage};
use super::super::error::{Error, Invalid, Problem, RuntimeGap};
use super::super::linear_plan::{
    BufferFacts, BufferLayout, BufferTexture, BufferTextureCall, BufferTextureProblem,
};
use super::super::texture_descriptor::TextureLayout;
use super::super::values::StorageMode;
use ash::vk;
use metal2vulkan::device::FormatUsage;
use metal2vulkan::metal::{PixelFormat, TextureType};

/// Whether `result` is the typed gap of a view the runtime does not make.
fn not_made(result: Result<ViewUsage, Error>) -> bool {
    matches!(result, Err(Error::NotImplemented(RuntimeGap::Textures)))
}

/// A view takes its buffer's usage where its format serves both kinds of texel buffer. Where the
/// format serves only the view's own kind (NVIDIA takes `R64_UINT` for storage texel buffers
/// only), a device with `maintenance5` makes the view naming that kind alone; one without it does
/// not make the view, nor does any device make one in a format that does not serve its kind.
#[test]
fn a_view_names_the_one_kind_its_format_serves_where_maintenance5_allows() {
    let both = |_: FormatUsage| true;
    let storage_only = |usage: FormatUsage| usage == FormatUsage::StorageTexelBuffer;
    let uniform_only = |usage: FormatUsage| usage == FormatUsage::UniformTexelBuffer;
    for storage in [false, true] {
        for maintenance5 in [false, true] {
            assert_eq!(
                view_usage(both, storage, maintenance5).unwrap(),
                ViewUsage::Buffer
            );
        }
    }
    assert_eq!(
        view_usage(storage_only, true, true).unwrap(),
        ViewUsage::Only { storage: true }
    );
    assert_eq!(
        view_usage(uniform_only, false, true).unwrap(),
        ViewUsage::Only { storage: false }
    );
    assert!(not_made(view_usage(storage_only, true, false)));
    assert!(not_made(view_usage(uniform_only, false, false)));
    assert!(not_made(view_usage(storage_only, false, true)));
    assert!(not_made(view_usage(uniform_only, true, true)));
    assert!(not_made(view_usage(|_| false, true, true)));
}

/// A staged shadow's texels go back into the buffer as one region where its rows lie one after
/// another (a 1080p surface's packed rows: one region, not 1080), and a region a row where bytes
/// lie between them; each slice of packed rows joins the next where the slices lie one after
/// another too.
#[test]
fn packed_rows_go_back_as_one_region() {
    let layout = |row_pitch: u64, rows: u64, slices: u64| BufferLayout {
        offset: 0,
        row_pitch,
        slice_pitch: row_pitch * rows,
        row_bytes: 7680,
        rows,
        slices,
    };
    let regions = |layout: BufferLayout, start: u64, at: u64| -> Vec<(u64, u64, u64)> {
        (staged_rows_back(&layout, start, at).iter())
            .map(|copy: &vk::BufferCopy| (copy.src_offset, copy.dst_offset, copy.size))
            .collect()
    };
    assert_eq!(
        regions(layout(7680, 1080, 1), 64, 3),
        [(64, 3, 7680 * 1080)]
    );
    assert_eq!(regions(layout(7680, 4, 2), 0, 5), [(0, 5, 7680 * 8)]);
    assert_eq!(
        regions(layout(7744, 3, 1), 0, 5),
        [
            (0, 5, 7680),
            (7744, 7749, 7680),
            (2 * 7744, 2 * 7744 + 5, 7680)
        ]
    );
}

/// A `width x height` texture of 4-byte texels over an IOSurface plane whose rows are
/// `bytes_per_row` bytes apart.
fn over_a_plane(width: u32, height: u32, bytes_per_row: u64) -> BufferTexture {
    BufferTexture {
        layout: TextureLayout {
            texture_type: TextureType::Type2D,
            pixel_format: PixelFormat::BGRA8Unorm,
            extent: [width, height, 1],
            levels: 1,
            layers: 1,
            samples: 1,
        },
        texel_bytes: 4,
        storage: StorageMode::Shared,
        offset: 0,
        bytes_per_row,
        call: BufferTextureCall::IoSurfacePlane { planar: false },
    }
}

/// `turned_plane`'s answer for a `width x height` texture of `rotation` over a plane whose rows
/// are `bytes_per_row` apart, in a buffer long enough for either way round.
fn turned(
    rotation: u8,
    width: u32,
    height: u32,
    bytes_per_row: u64,
) -> Result<(), BufferTextureProblem> {
    let texture = over_a_plane(width, height, bytes_per_row);
    let length = bytes_per_row * u64::from(width.max(height)) + 1024;
    match turned_plane("t", &texture, rotation, &BufferFacts { length }) {
        Ok(_) => Ok(()),
        Err(Error::Invalid(Invalid {
            problem: Problem::BufferTexture(problem),
            ..
        })) => Err(problem),
        Err(other) => panic!("rotation {rotation}, {width}x{height} at {bytes_per_row}: {other}"),
    }
}

/// Metal holds an IOSurface texture's `bytesPerRow` to the **texture's** own row, whatever the
/// plane holds and whatever the rotation: measured on the M3 Max under plain Metal with no
/// validation layer, a 4x8 plane at its tight stride of 16 behind an 8x4 texture ends the
/// process, `_mtlValidateStrideTextureParameters:1938: failed assertion 'Texture Descriptor
/// Validation IOSurface texture: bytesPerRow (16) must be greater or equal to (32) bytes'`.
///
/// A quarter turn (1, 3 and 4) stores the frame turned, so its plane's own tight row is the
/// shorter of the two and the stored extent's plan takes a pitch Metal asserts on; a half turn
/// (2) and no turn store it as it is, and were always held to the texture's row. The refusal is
/// reached through the plan from the input's own shape with no device, because the texture it
/// stands for ends Metal's process and no test may build it.
#[test]
fn an_iosurface_pitch_below_the_textures_own_row_is_refused_at_every_rotation() {
    let below = |bytes_per_row, row_bytes| {
        Err(BufferTextureProblem::RowPitch {
            bytes_per_row,
            row_bytes,
        })
    };
    for rotation in 1..=4u8 {
        // The measured shape: an 8x4 texture whose plane is 4x8 at the plane's tight 16.
        assert_eq!(
            turned(rotation, 8, 4, 16),
            below(16, 32),
            "rotation {rotation}"
        );
        // The texture's own row, and a padded one, are taken.
        assert_eq!(turned(rotation, 8, 4, 32), Ok(()), "rotation {rotation}");
        assert_eq!(turned(rotation, 8, 4, 48), Ok(()), "rotation {rotation}");
        // The rotated lane's own shape: a 64x16 texture whose quarter-turned plane is 16 blocks
        // wide, so the plane's tight row is 64 while the texture's own is 256.
        assert_eq!(
            turned(rotation, 64, 16, 64),
            below(64, 256),
            "rotation {rotation}"
        );
        assert_eq!(turned(rotation, 64, 16, 256), Ok(()), "rotation {rotation}");
        // A texture taller than wide, at a pitch covering both rows.
        assert_eq!(turned(rotation, 16, 64, 256), Ok(()), "rotation {rotation}");
    }
    // A half turn stores the frame as it is, so the plane's row and the texture's are the same
    // one and this rule adds nothing: the stored extent's plan already held the pitch to it.
    // That is why the bug was a quarter turn's alone.
    assert_eq!(turned(2, 16, 64, 64), Ok(()));
    assert_eq!(turned(2, 16, 64, 48), below(48, 64));
    // A quarter turn of the same texture stores it 64 texels wide, so its plane needs 256 bytes
    // a row for the stored frame -- which the stored extent's plan has always required. The rule
    // added here is the other direction, the one no plan held: the texture's own row where the
    // plane's is shorter, which is the 64x16 pair above.
    assert_eq!(turned(1, 16, 64, 64), below(64, 256));
}
