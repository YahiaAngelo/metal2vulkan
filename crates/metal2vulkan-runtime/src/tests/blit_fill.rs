//! The bytes Metal's private texture fills write (`blit_fill.rs`): each multisampled sample's,
//! from our probe `ms.m`'s read-backs (the M3 Max under macOS 27.0.1), and the formats a fill
//! has a unit for.

use crate::blit_fill::{fill_unit, sample_units, words};
use metal2vulkan::metal::PixelFormat;

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

/// Every pixel format but those the runtime makes no texture of has a fill unit.
#[test]
fn every_texture_format_has_a_fill_unit() {
    use PixelFormat as P;
    let none: Vec<PixelFormat> = PixelFormat::ALL
        .iter()
        .copied()
        .filter(|&pf| fill_unit(pf).is_none())
        .collect();
    assert_eq!(
        none,
        [P::Invalid, P::Depth24UnormStencil8, P::X24Stencil8],
        "formats without a fill unit"
    );
}
