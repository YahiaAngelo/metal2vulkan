//! Metal's private texture fills of texture buffers (`BlitCommandEncoder::fill_texture` and
//! `fill_texture_color`), on the device the cfg selects, under the validation layer, against
//! Metal's bytes.
//!
//! Each case replays our probe `tb.m`'s (an Apple M3 Max under macOS 27.0.1): a texture buffer
//! 12 texels wide, of a format of 1 to 16 bytes a texel, at an offset into a `Shared` buffer of
//! 512 bytes whose byte `i` is first `i · 37 mod 251` (256 bytes a row), its fills in one
//! encoder, and every byte of the buffer read back. A fill is the probe's argument
//! `x,w[,y,h,d,level,slice,z]:` (`y` 0, `h` and `d` 1, the rest 0 where left out) and its pattern:
//! `p<n>` the `n` bytes `a1 a2 ..`, `p<n>@<b>` from the byte `b`, `c<pf>` the color
//! (0.25, -0.5, 1.25, 0.75) packed in `pf`. Metal's answer is the first 16 hex digits of the
//! SHA-256 of the buffer's bytes; a group's, of its cases' buffers in order.
//!
//! - **Streams**: patterns longer than the texel, of lengths not a power of two, and up to 256
//!   bytes; regions past the texture's width, past the buffer's end, of no rows, of no images, at
//!   a level, slice, row and depth past 0, and of no texels; a fill after a fill (the consumer's 9
//!   cases among them). Metal writes the buffer's bytes, the pattern across them, and the bytes of
//!   its block past the pattern's copies as zeros, which the runtime writes and announces once
//!   (`texture-buffer-fill-leftover`).
//! - **Groups** (the consumer's set): each format at offsets 0 and 64, and 4 and 68 for a texel of
//!   4 bytes or fewer, a pattern of each power of two of bytes up to the texel whole, inside,
//!   ending at the width and at a level, slice, row, height and depth past 0 and 1, a color packed
//!   in each format of no more bytes than the texel, and fills of no texels between others.
//!
//! `tests::blit_fill` checks the bytes of the fills without a device. A differing case prints a
//! `TEXTURE-BUFFER-FILLS-DIFFERS` line with the runs of bytes its fills changed.

use super::degrade_notices::{
    assert_once, forced, on_device, shared_in_system_memory, TEXTURE_BUFFER_FILL,
};
use super::{buffer_with, read_buffer, WAIT};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    ClearColor, CommandQueue, Degradation, Device, Origin, Region, ResourceOptions, Size,
    TextureDescriptor,
};
use sha2::{Digest, Sha256};

/// The buffer's bytes.
const BYTES: usize = 512;

/// A stream: its name, the format's raw value, the texture's offset, its fills, and Metal's
/// answer.
type Stream = (
    &'static str,
    u32,
    u64,
    &'static [&'static str],
    &'static str,
);

/// A group: its name, the format's raw value, the texture's offset, its cases' fills, and
/// Metal's answer.
type Group = (
    &'static str,
    u32,
    u64,
    &'static [&'static [&'static str]],
    &'static str,
);

/// The streams, as the probe ran them.
const STREAMS: [Stream; 26] = [
    ("r8-8-whole", 10, 0, &["0,12:p8"], "0ee2aff21c7fc968"),
    ("r8-8-odd", 10, 4, &["3,5:p8"], "4e2a928549345a34"),
    ("rgba8-8-whole", 70, 0, &["0,12:p8"], "84f5d37ca71f4e7f"),
    ("rgba8-8-x4", 70, 64, &["4,8:p8"], "5a2c0a26da0f6e23"),
    ("rgba8-c115", 70, 0, &["4,6:c115"], "4db1e92e18d57d38"),
    ("r8-past-width", 10, 0, &["12,3:p1"], "0260417c8c16e763"),
    ("rgba8-ending-past", 70, 0, &["10,5:p4"], "e78f2f4f540bb93d"),
    ("r8-no-rows", 10, 0, &["1,2,0,0,1:p1"], "4bd9db7e5278e209"),
    (
        "rgba8-no-images",
        70,
        64,
        &["1,2,0,1,0:p4"],
        "85a8c2ef070cc49a",
    ),
    ("rgba8-3-whole", 70, 0, &["0,12:p3"], "e3999e838bee0e1c"),
    (
        "rgba8-9-after-8",
        70,
        0,
        &["0,12:p8", "0,12:p9@b1"],
        "456210529ae74916",
    ),
    ("rgba8-12-x1", 70, 4, &["1,10:p12"], "19ce1fc8bdec16f0"),
    ("rgba32-17", 125, 0, &["0,12:p17"], "d31c4c6fb6e0e200"),
    ("rgba32-24-x1", 125, 64, &["1,10:p24"], "2178521d4be0aa92"),
    ("rgba8-33-past", 70, 0, &["0,40:p33"], "5f7b1f135056b512"),
    ("r8-65-long", 10, 0, &["3,300:p65"], "3366332bf93b2371"),
    ("rgba8-100", 70, 4, &["1,100:p100"], "8d688b0eca7b3728"),
    ("rgba8-256", 70, 0, &["0,100:p256"], "9096b69c455dc4e0"),
    ("rg8-5-x6", 30, 0, &["6,40:p5"], "f2b4ec3d60883b4e"),
    ("rgba16-24", 115, 64, &["3,30:p24"], "53fb1bd384886492"),
    (
        "rgba32-past-end",
        125,
        64,
        &["20,20:p16"],
        "b11323ede905084e",
    ),
    ("r8-past-buffer", 10, 0, &["600,3:p1"], "d71bdb461ee51a7e"),
    ("rgba8-width-0", 70, 0, &["2,0:p3"], "d71bdb461ee51a7e"),
    (
        "rgba8-level-slice-z",
        70,
        4,
        &["2,4,5,3,2,3,7,4:p3"],
        "c8766e71a2d1ea36",
    ),
    (
        "rgba8-c115-then-12",
        70,
        0,
        &["0,12:c115", "0,12:p12"],
        "f5e35d82b4b3bcd8",
    ),
    ("r8-3-odd", 10, 4, &["1,30:p3"], "8faf563f27272b2b"),
];

/// The fills of a group of `r8` texels, one case each.
const R8: &[&[&str]] = &[
    &["0,12:p1"],
    &["3,5:p1"],
    &["7,5:p1"],
    &["2,4,5,3,2,3,7,4:p1"],
    &["1,6,4,2,1,1,2,0:c10"],
    &["0,12:p1", "5,0:p1", "12,0:p1", "9,2:c10"],
];

/// The fills of a group of `rg8` texels, one case each.
const RG8: &[&[&str]] = &[
    &["0,12:p1"],
    &["3,5:p1"],
    &["7,5:p1"],
    &["2,4,5,3,2,3,7,4:p1"],
    &["0,12:p2"],
    &["3,5:p2"],
    &["7,5:p2"],
    &["2,4,5,3,2,3,7,4:p2"],
    &["1,6,4,2,1,1,2,0:c10"],
    &["1,6,4,2,1,1,2,0:c30"],
    &["0,12:p1", "5,0:p1", "12,0:p1", "9,2:c10"],
];

/// The fills of a group of `rgba8` texels, one case each.
const RGBA8: &[&[&str]] = &[
    &["0,12:p1"],
    &["3,5:p1"],
    &["7,5:p1"],
    &["2,4,5,3,2,3,7,4:p1"],
    &["0,12:p2"],
    &["3,5:p2"],
    &["7,5:p2"],
    &["2,4,5,3,2,3,7,4:p2"],
    &["0,12:p4"],
    &["3,5:p4"],
    &["7,5:p4"],
    &["2,4,5,3,2,3,7,4:p4"],
    &["1,6,4,2,1,1,2,0:c10"],
    &["1,6,4,2,1,1,2,0:c30"],
    &["1,6,4,2,1,1,2,0:c70"],
    &["0,12:p1", "5,0:p1", "12,0:p1", "9,2:c10"],
];

/// The fills of a group of `rgba16` texels, one case each.
const RGBA16: &[&[&str]] = &[
    &["0,12:p1"],
    &["3,5:p1"],
    &["7,5:p1"],
    &["2,4,5,3,2,3,7,4:p1"],
    &["0,12:p2"],
    &["3,5:p2"],
    &["7,5:p2"],
    &["2,4,5,3,2,3,7,4:p2"],
    &["0,12:p4"],
    &["3,5:p4"],
    &["7,5:p4"],
    &["2,4,5,3,2,3,7,4:p4"],
    &["0,12:p8"],
    &["3,5:p8"],
    &["7,5:p8"],
    &["2,4,5,3,2,3,7,4:p8"],
    &["1,6,4,2,1,1,2,0:c10"],
    &["1,6,4,2,1,1,2,0:c30"],
    &["1,6,4,2,1,1,2,0:c70"],
    &["1,6,4,2,1,1,2,0:c115"],
    &["0,12:p1", "5,0:p1", "12,0:p1", "9,2:c10"],
];

/// The fills of a group of `rgba32` texels, one case each.
const RGBA32: &[&[&str]] = &[
    &["0,12:p1"],
    &["3,5:p1"],
    &["7,5:p1"],
    &["2,4,5,3,2,3,7,4:p1"],
    &["0,12:p2"],
    &["3,5:p2"],
    &["7,5:p2"],
    &["2,4,5,3,2,3,7,4:p2"],
    &["0,12:p4"],
    &["3,5:p4"],
    &["7,5:p4"],
    &["2,4,5,3,2,3,7,4:p4"],
    &["0,12:p8"],
    &["3,5:p8"],
    &["7,5:p8"],
    &["2,4,5,3,2,3,7,4:p8"],
    &["0,12:p16"],
    &["3,5:p16"],
    &["7,5:p16"],
    &["2,4,5,3,2,3,7,4:p16"],
    &["1,6,4,2,1,1,2,0:c10"],
    &["1,6,4,2,1,1,2,0:c30"],
    &["1,6,4,2,1,1,2,0:c70"],
    &["1,6,4,2,1,1,2,0:c115"],
    &["1,6,4,2,1,1,2,0:c125"],
    &["0,12:p1", "5,0:p1", "12,0:p1", "9,2:c10"],
];

/// The groups, as the probe ran them.
const GROUPS: [Group; 16] = [
    ("r8-at-0", 10, 0, R8, "715e234bffb7dcf7"),
    ("r8-at-4", 10, 4, R8, "e1f8ea12e12b9db9"),
    ("r8-at-64", 10, 64, R8, "de7442c0bb4f290f"),
    ("r8-at-68", 10, 68, R8, "1a85c929b8d80219"),
    ("rg8-at-0", 30, 0, RG8, "b8fcc13a29cea9c1"),
    ("rg8-at-4", 30, 4, RG8, "62103fb7d6791aea"),
    ("rg8-at-64", 30, 64, RG8, "fa35a4670beb6d3a"),
    ("rg8-at-68", 30, 68, RG8, "9ddd3dab429ace88"),
    ("rgba8-at-0", 70, 0, RGBA8, "94b6756df9a34a32"),
    ("rgba8-at-4", 70, 4, RGBA8, "dde8bea1606de5aa"),
    ("rgba8-at-64", 70, 64, RGBA8, "14795fa2287dd1c0"),
    ("rgba8-at-68", 70, 68, RGBA8, "bf7d8a1f8a105af8"),
    ("rgba16-at-0", 115, 0, RGBA16, "f9d0a1f338d87e0b"),
    ("rgba16-at-64", 115, 64, RGBA16, "c2a934eb4a41e124"),
    ("rgba32-at-0", 125, 0, RGBA32, "c9332419d8b8bfad"),
    ("rgba32-at-64", 125, 64, RGBA32, "ebaf73cdcde2c014"),
];

/// The buffer's first bytes: byte `i` is `i · 37 mod 251`.
fn first_bytes() -> Vec<u8> {
    (0..BYTES).map(|i| (i * 37 % 251) as u8).collect()
}

/// The buffer's bytes after `fills` of a texture buffer of `format` at `offset`.
fn case(
    device: &Device,
    queue: &CommandQueue,
    format: u32,
    offset: u64,
    fills: &[&str],
) -> Vec<u8> {
    let buffer = buffer_with(device, &first_bytes()).unwrap();
    let d = TextureDescriptor::texture_buffer(
        PixelFormat::try_from(format).unwrap(),
        12,
        ResourceOptions::shared(),
        TextureUsage::SHADER_READ,
    );
    let texture = buffer.new_texture(&d, offset, 256).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for fill in fills {
        let (spec, pattern) = fill.split_once(':').unwrap();
        let mut v = [0, 0, 0, 1, 1, 0, 0, 0];
        for (k, number) in spec.split(',').enumerate() {
            v[k] = number.parse().unwrap();
        }
        let [x, w, y, h, depth, level, slice, z] = v;
        let region = Region {
            origin: Origin::new(x, y, z),
            size: Size::new(w, h, depth),
        };
        if let Some(packed) = pattern.strip_prefix('c') {
            let pf = PixelFormat::try_from(packed.parse::<u32>().unwrap()).unwrap();
            let color = ClearColor::new(0.25, -0.5, 1.25, 0.75);
            blit.fill_texture_color(&texture, level, slice, region, color, Some(pf))
                .unwrap();
        } else {
            let counted = pattern.strip_prefix('p').unwrap();
            let (n, from) = counted.split_once('@').unwrap_or((counted, "a1"));
            let n: usize = n.parse().unwrap();
            let from = u8::from_str_radix(from, 16).unwrap();
            let bytes: Vec<u8> = (0..n).map(|i| from.wrapping_add(i as u8)).collect();
            blit.fill_texture(&texture, level, slice, region, &bytes)
                .unwrap();
        }
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    read_buffer(&buffer, 0, BYTES as u64).unwrap()
}

/// The first 16 hex digits of the SHA-256 of `bytes`.
fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The runs of `bytes` that differ from the buffer's first bytes, as the probe prints them.
fn runs(bytes: &[u8]) -> String {
    let first = first_bytes();
    let mut line = String::new();
    let mut previous = false;
    for (at, (&byte, &was)) in bytes.iter().zip(&first).enumerate() {
        let changed = byte != was;
        if changed && !previous {
            line += &format!(" | {at:#x}:");
        }
        if changed {
            line += &format!(" {byte:02x}");
        }
        previous = changed;
    }
    line
}

/// Every stream and group writes the buffer's bytes as Metal's fills do, and the fills that read
/// bytes past their pattern's copies announce it once.
#[test]
fn texture_buffer_fills_write_metal_s_bytes() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let mut wrong = Vec::new();
        for (name, format, offset, fills, metal) in STREAMS {
            let ours = case(device, &queue, format, offset, fills);
            if digest(&ours) != metal {
                println!("TEXTURE-BUFFER-FILLS-DIFFERS {name}:{}", runs(&ours));
                wrong.push(name.to_owned());
            }
        }
        for (name, format, offset, cases, metal) in GROUPS {
            let ours: Vec<Vec<u8>> = cases
                .iter()
                .map(|fills| case(device, &queue, format, offset, fills))
                .collect();
            if digest(&ours.concat()) != metal {
                for (k, bytes) in ours.iter().enumerate() {
                    println!("TEXTURE-BUFFER-FILLS-DIFFERS {name} {k}:{}", runs(bytes));
                }
                wrong.push(name.to_owned());
            }
        }
        let mut kinds = forced(&[(
            Degradation::SharedInSystemMemory,
            shared_in_system_memory(device),
        )]);
        kinds.extend(TEXTURE_BUFFER_FILL);
        assert_once(&messages.take_notices(), &Vec::from_iter(kinds));
        assert!(wrong.is_empty(), "fills unlike Metal's: {wrong:?}");
    });
}
