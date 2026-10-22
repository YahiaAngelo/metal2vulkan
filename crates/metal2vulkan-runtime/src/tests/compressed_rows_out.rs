//! A blit out of a texture Metal keeps losslessly compressed into a buffer at an offset off the
//! texel, rows apart by a pitch off it too (the staged path, `staged-texture-copy`), rewrites
//! the buffer's rows the copy wrote as Metal's blit writes them, and only them
//! (`blit_plan::copied_rows`, `compressed_blit.rs`): each row equals the same texels copied out
//! at the texel's own layout, the bytes between and around the rows keep their value, and the
//! device counts the rows it rewrote (`WorkCounts::compressed_rows_rewritten`), so a pass shows
//! the rewrite ran.
//!
//! Each format prints a `COMPRESSED-ROWS` line. Device tests: compiled only under the validation
//! crate's GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    BlitOption, Buffer, CommandBufferStatus, CommandQueue, Device, Origin, ResourceOptions, Size,
    Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The texture's side.
const SIDE: u64 = 64;

/// The region copied out at the offset off the texel: its origin and side.
const AT: u64 = 8;
const REGION: u64 = 8;

/// The offset off the texel, and the bytes each row's pitch adds past the row.
const OFFSET: u64 = 4;
const PAST: u64 = 12;

/// The byte the destination holds before the copy.
const KEPT: u8 = 0xee;

/// `n` bytes, every bit of each varied.
fn bytes(n: u64) -> Vec<u8> {
    let mut x = 0x9e37_79b9_u32;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x >> 24) as u8
        })
        .collect()
}

/// A shared buffer of `bytes`.
fn shared(device: &Device, bytes: &[u8]) -> Buffer {
    (device.new_buffer_with_bytes(bytes, ResourceOptions::shared())).unwrap()
}

/// A buffer's bytes, read back with no command buffer using it.
fn read(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().expect("a shared buffer");
    // SAFETY: the buffer holds `length` bytes at `contents`, alive with it; no command buffer
    // uses it now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// Copy `side` texels square at `at` of `texture` out into `buffer` at `offset`, rows `pitch`
/// apart, in a command buffer of `queue`'s run to completion.
fn out(
    queue: &CommandQueue,
    texture: &Texture,
    (at, side): (u64, u64),
    (buffer, offset, pitch): (&Buffer, u64, u64),
) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(at, at, 0),
        Size::new(side, side, 1),
        buffer,
        offset,
        pitch,
        0,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
}

/// `BGRA10_XR` (8-byte texels, so the offset of 4 and the pitch are off the texel: staged) and
/// `RG11B10Float` (4-byte texels, on it: one region a row), each a `Private` `ShaderRead`
/// texture with GPU-optimized contents that Metal keeps compressed.
#[test]
fn a_copy_out_off_the_texel_rewrites_the_rows_it_wrote() {
    let scratch = Scratch::new("compressed-rows-out");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    for (format, per) in [
        (PixelFormat::BGRA10XR, 8u64),
        (PixelFormat::RG11B10Float, 4),
    ] {
        let mut descriptor = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor.allow_gpu_optimized_contents = true;
        descriptor.resource_options = ResourceOptions::private();
        let Ok(texture) = device.new_texture(&descriptor) else {
            eprintln!("COMPRESSED-ROWS: {format:?}: skipped: no such texture here");
            continue;
        };
        let row = SIDE * per;
        let raw = bytes(row * SIDE);
        let source = shared(&device, &raw);
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            row,
            0,
            Size::new(SIDE, SIDE, 1),
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        // The texels as Metal's blit writes them, at the texel's own layout.
        let whole = shared(&device, &vec![0; (row * SIDE) as usize]);
        out(&queue, &texture, (0, SIDE), (&whole, 0, row));
        let whole = read(&whole);
        // The rule changes the region's random texels, so a row the rewrite missed shows.
        let changed = (0..REGION).any(|y| {
            let from = ((AT + y) * row + AT * per) as usize;
            let texels = (REGION * per) as usize;
            whole[from..from + texels] != raw[from..from + texels]
        });
        assert!(
            changed,
            "{format:?}: Metal's rule changed none of the region's texels"
        );
        // The region out at the offset off the texel.
        let pitch = REGION * per + PAST;
        let length = OFFSET + pitch * REGION;
        let destination = shared(&device, &vec![KEPT; length as usize]);
        let before = device.work_counts().compressed_rows_rewritten;
        out(
            &queue,
            &texture,
            (AT, REGION),
            (&destination, OFFSET, pitch),
        );
        let rewritten = device.work_counts().compressed_rows_rewritten - before;
        assert_eq!(rewritten, REGION, "{format:?}: the rows rewritten");
        let mut want = vec![KEPT; length as usize];
        for y in 0..REGION {
            let from = ((AT + y) * row + AT * per) as usize;
            let to = (OFFSET + y * pitch) as usize;
            let texels = (REGION * per) as usize;
            want[to..to + texels].copy_from_slice(&whole[from..from + texels]);
        }
        assert_eq!(read(&destination), want, "{format:?}");
        eprintln!("COMPRESSED-ROWS: {format:?}: {rewritten} rows rewritten");
    }
}
