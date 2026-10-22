//! The buffer's side of a blit between a buffer and a multisampled texture (`sample_interleave.rs`):
//! each row's samples lie in the buffer, or the copy is refused.

use super::super::blit_plan::BufferLayout;
use super::super::sample_interleave::rows_within;

fn layout(offset: u64, bytes_per_row: u64) -> BufferLayout {
    BufferLayout {
        offset,
        bytes_per_row,
        bytes_per_image: 0,
    }
}

/// Rows of 4 texels at 4 samples of 4 bytes (64 bytes) reach the buffer's last byte and no
/// further; a guest's offset or `bytesPerRow` whose rows pass `u64::MAX` lies past every buffer,
/// never a wrapped offset inside it.
#[test]
fn rows_lie_in_the_buffer_up_to_its_last_byte() {
    let size = [4, 4, 1];
    let last = 4096 - (3 * 64 + 64);
    assert!(rows_within(&layout(last, 64), size, 4, 4, 4096));
    assert!(!rows_within(&layout(last + 1, 64), size, 4, 4, 4096));
    // Rows that overlap, into the texture: the last row ends past every other.
    assert!(rows_within(
        &layout(4096 - (3 * 8 + 64), 8),
        size,
        4,
        4,
        4096
    ));
    assert!(!rows_within(
        &layout(4096 - (3 * 8 + 63), 8),
        size,
        4,
        4,
        4096
    ));
    for l in [
        layout(u64::MAX, 64),
        layout(u64::MAX - 63, 64),
        layout(0, 1 << 63),
        layout(0, 0x5555_5555_5555_5560),
        layout(0, u64::MAX),
    ] {
        assert!(!rows_within(&l, size, 4, 4, 4096), "{l:?}");
    }
    // An empty region reads nothing.
    assert!(rows_within(&layout(u64::MAX, u64::MAX), [0, 4, 1], 4, 4, 0));
}
