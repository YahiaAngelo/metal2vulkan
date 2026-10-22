//! The bytes Metal's private texture fills write (`blit_fill.rs`): each multisampled sample's,
//! from our probe `ms.m`'s read-backs, and a texture buffer's, from our probe `tb.m`'s (the M3
//! Max under macOS 27.0.1), and the formats a fill has a unit for.

use crate::blit_fill::{fill_unit, sample_units, words, StreamFill};
use metal2vulkan::metal::PixelFormat;
use std::ops::Range;

/// Bytes from hex.
fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&text[at..at + 2], 16).unwrap())
        .collect()
}

/// Each sample's bytes as hex.
fn samples(pattern: &str, bytes: u32, count: u32) -> Vec<String> {
    sample_units(&hex(pattern), bytes, count)
        .iter()
        .map(|unit| unit.iter().map(|b| format!("{b:02x}")).collect())
        .collect()
}

/// The probe's 16-byte pattern.
const P16: &str = "00112233445566778899aabbccddeeff";

/// The probe's 32-byte pattern.
const P32: &str = "00112233445566778899aabbccddeeff102132435465768798a9bacbdcedfe0f";

/// Where a texel's samples hold 16 bytes or fewer, sample `s` takes the pattern's bytes from
/// `s · b`, the pattern repeated and cut at the samples' bytes; past 16, each sample the pattern
/// cut at its texel.
#[test]
fn multisampled_texels_take_metal_s_bytes_of_the_pattern() {
    // `R8Unorm`, `Stencil8` at 4 and 2 samples.
    assert_eq!(samples(P16, 1, 4), ["00", "11", "22", "33"]);
    assert_eq!(samples(P32, 1, 2), ["00", "11"]);
    // `RG8Unorm`, `Depth16Unorm` and the 16-bit packed formats.
    assert_eq!(samples(P16, 2, 4), ["0011", "2233", "4455", "6677"]);
    assert_eq!(samples(P32, 2, 2), ["0011", "2233"]);
    // `RGBA8Unorm`, `Depth32Float`, `BGR10_XR`, `RGB10A2Unorm_sRGB`; a 32-byte pattern cut at 16.
    let four = ["00112233", "44556677", "8899aabb", "ccddeeff"];
    assert_eq!(samples(P16, 4, 4), four);
    assert_eq!(samples(P32, 4, 4), four);
    assert_eq!(samples(P32, 4, 2), ["00112233", "44556677"]);
    // `RGBA16Float`, `BGRA10_XR`: 16 bytes at 2 samples, the texel's 8 each at 4.
    assert_eq!(samples(P32, 8, 2), ["0011223344556677", "8899aabbccddeeff"]);
    assert_eq!(samples(P16, 8, 4), ["0011223344556677"; 4]);
    // `RGBA32Float` at 2 and 4.
    assert_eq!(samples(P32, 16, 2), [P16; 2]);
    assert_eq!(samples(P16, 16, 4), [P16; 4]);
    // Patterns that do not divide the samples' bytes.
    assert_eq!(samples("a1b2c3", 1, 4), ["a1", "b2", "c3", "a1"]);
    assert_eq!(samples("a1b2c3", 2, 4), ["a1b2", "c3a1", "b2c3", "a1b2"]);
    assert_eq!(
        samples("a1b2c3d4e5", 4, 4),
        ["a1b2c3d4", "e5a1b2c3", "d4e5a1b2", "c3d4e5a1"]
    );
    assert_eq!(samples("a1b2c3d4e5", 4, 2), ["a1b2c3d4", "e5a1b2c3"]);
    assert_eq!(
        samples("a1b2c3d4e5f6", 8, 2),
        ["a1b2c3d4e5f6a1b2", "c3d4e5f6a1b2c3d4"]
    );
    assert_eq!(
        samples("a1b2c3", 16, 4),
        ["a1b2c3a1b2c3a1b2c3a1b2c3a1b2c3a1"; 4]
    );
    // A 20-byte pattern into 16.
    assert_eq!(
        samples("0102030405060708090a0b0c0d0e0f1011121314", 4, 4),
        ["01020304", "05060708", "090a0b0c", "0d0e0f10"]
    );
}

/// A sample's bytes as the words of a clear value of the unsigned format of its size.
#[test]
fn sample_bytes_become_clear_words() {
    assert_eq!(words(&hex("a1")), [0xa1, 0, 0, 0]);
    assert_eq!(words(&hex("a1b2")), [0xb2a1, 0, 0, 0]);
    assert_eq!(words(&hex("00112233")), [0x3322_1100, 0, 0, 0]);
    assert_eq!(
        words(&hex("0011223344556677")),
        [0x3322_1100, 0x7766_5544, 0, 0]
    );
    assert_eq!(
        words(&hex(P16)),
        [0x3322_1100, 0x7766_5544, 0xbbaa_9988, 0xffee_ddcc]
    );
}

/// Every pixel format the crate holds a texture of (one with aspects) has a fill unit, but the
/// 24-bit depth formats, which Apple GPUs make no texture of (refused at the texture's making).
#[test]
fn every_texture_format_has_a_fill_unit() {
    use metal2vulkan::metal::format::aspects;
    use PixelFormat as P;
    let none: Vec<PixelFormat> = PixelFormat::ALL
        .iter()
        .copied()
        .filter(|&pf| aspects(pf).is_some() && fill_unit(pf).is_none())
        .collect();
    assert_eq!(
        none,
        [P::Depth24UnormStencil8, P::X24Stencil8],
        "formats without a fill unit"
    );
}

/// The pattern of `n` bytes `a1 a2 ..`, the probe's.
fn counting(n: usize, from: u8) -> Vec<u8> {
    (0..n).map(|i| from.wrapping_add(i as u8)).collect()
}

/// The bytes `bytes` of a texture buffer's buffer a fill of `pattern` writes, as hex.
fn streamed(pattern: &[u8], bytes: Range<u64>) -> String {
    let stream = StreamFill::new(pattern);
    bytes
        .clone()
        .map(|at| format!("{:02x}", stream.byte(&bytes, at)))
        .collect()
}

/// A texture buffer's fill writes the bytes Metal's writes: the pattern by the masked index of the
/// region's byte, but in each 16-byte chunk of the buffer the region covers whole, 16 bytes of the
/// block of the pattern's whole copies, zeros past them (`tb.m`'s cases, the region's bytes from
/// the texture's offset plus `x` texels).
#[test]
fn texture_buffer_fills_write_metal_s_bytes() {
    // 8 bytes into 5 `R8Unorm` texels from byte 7, and 8 `RGBA8Unorm` texels from byte 80.
    assert_eq!(streamed(&counting(8, 0xa1), 7..12), "a1a2a3a4a5");
    assert_eq!(
        streamed(&counting(8, 0xa1), 80..112),
        "a1a2a3a4a5a6a7a8a1a2a3a4a5a6a7a8a1a2a3a4a5a6a7a8a1a2a3a4a5a6a7a8"
    );
    // 3 bytes, 12 `RGBA8Unorm` texels from 0: 5 copies a chunk, a zero past them.
    assert_eq!(
        streamed(&counting(3, 0xa1), 0..48),
        "a1a2a3a1a2a3a1a2a3a1a2a3a1a2a300a1a2a3a1a2a3a1a2a3a1a2a3a1a2a300a1a2a3a1a2a3a1a2a3a1a2a3a1a2a300"
    );
    // 3 bytes, 30 `R8Unorm` texels from 5: `p0 p0 p2 p2` outside the chunk.
    assert_eq!(
        streamed(&counting(3, 0xa1), 5..35),
        "a1a1a3a3a1a1a3a3a1a1a3a1a2a3a1a2a3a1a2a3a1a2a3a1a2a300a3a1a1"
    );
    // 12 bytes from byte 8: the chunks read from the block's byte 8, past the one copy.
    assert_eq!(
        streamed(&counting(12, 0xa1), 8..48),
        "a1a2a3a4a1a2a3a4a9aaabac000000000000000000000000a9aaabac000000000000000000000000"
    );
    // 5 bytes into 40 `RG8Unorm` texels from byte 12.
    assert_eq!(
        streamed(&counting(5, 0xa1), 12..92),
        "a1a1a1a1a5a1a2a3a4a5a1a2a3a4a50000000000a5a1a2a3a4a5a1a2a3a4a50000000000\
         a5a1a2a3a4a5a1a2a3a4a50000000000a5a1a2a3a4a5a1a2a3a4a50000000000a5a5a5a5a1a1a1a1a5a5a5a5"
    );
    // 17 bytes into 12 `RGBA32Float` texels: a 32-byte block of one copy.
    let seventeen = "a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1000000000000000000000000000000";
    assert_eq!(streamed(&counting(17, 0xa1), 0..192), seventeen.repeat(6));
    // 24 bytes into 30 `RGBA16Float` texels from byte 88.
    let copy = "a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b80000000000000000";
    assert_eq!(
        streamed(&counting(24, 0xa1), 88..328),
        format!("a1a2a3a4a5a6a7a8{}a1a2a3a4a5a6a7a8", copy.repeat(7))
    );
    // 100 bytes into 40 `RGBA8Unorm` texels from byte 4: a 128-byte block.
    assert_eq!(
        streamed(&counting(100, 0x10), 4..164),
        "101112131011121310111213101112131415161718191a1b1c1d1e1f101112131415161718191a1b1c1d1e1f\
         303132333435363738393a3b3c3d3e3f303132333435363738393a3b3c3d3e3f505152535455565758595a5b\
         5c5d5e5f505152535455565758595a5b5c5d5e5f707172730000000000000000000000007071727300000000\
         0000000000000000101112131415161718191a1b1c1d1e1f10111213"
    );
}

/// A texture buffer's fill as its spans (each its unit repeated from the span's first byte)
/// writes every byte the fill does, and announces the bytes past the pattern's copies where a
/// chunk reads one: patterns of 1 to 70 bytes over regions of 0 to 200 bytes from 0 to 47.
#[test]
fn texture_buffer_fill_spans_are_its_bytes() {
    for n in 1..=70 {
        let pattern = counting(n, 0x01);
        let stream = StreamFill::new(&pattern);
        let copies = (16.max(n.next_power_of_two()) / n * n) as u64;
        for start in 0..48u64 {
            for length in 0..200u64 {
                let bytes = start..start + length;
                let mut written = vec![None; (start + length) as usize];
                for (span, unit) in stream.spans(bytes.clone()) {
                    for (k, at) in span.enumerate() {
                        assert!(written[at as usize].is_none(), "{n} {bytes:?}: twice");
                        written[at as usize] = Some(unit[k % unit.len()]);
                    }
                }
                let want: Vec<Option<u8>> = (0..start + length)
                    .map(|at| bytes.contains(&at).then(|| stream.byte(&bytes, at)))
                    .collect();
                assert_eq!(written, want, "{n} {bytes:?}");
                // Brute force: a whole chunk's byte read from the block's copies or past them.
                let leftover = (start.next_multiple_of(16)..(start + length) / 16 * 16).any(|at| {
                    let chunk = at / 16 * 16;
                    ((chunk - start) & (n as u64 - 1) & !3) + at % 16 >= copies
                });
                assert_eq!(
                    stream.reads_leftover(bytes.clone()),
                    leftover,
                    "{n} {bytes:?}"
                );
            }
        }
    }
}
