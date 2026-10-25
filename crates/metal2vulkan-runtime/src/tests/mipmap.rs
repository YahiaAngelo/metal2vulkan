//! `generate_mipmaps`' per-format check (`mipmap.rs`): Metal's measured answers, the units in
//! the last place of each field, and the spread between an Apple M3 Max's and an M4's answers.

use super::super::mipmap::{halves, measured, Field, MeasuredFormat};
use super::super::mipmap_plan::{format_problem, not_implemented, picks};
use metal2vulkan::metal::PixelFormat;

/// Every format Metal filters into mipmaps (of the formats the crate maps) has its measured
/// answers, but the extended-range formats, whose mipmaps no blit gives and the runtime computes
/// with Metal's filter or refuses. An integer format's mipmaps are picked, exactly, with no answers
/// to check (`mipmap_plan::plan_picks`).
#[test]
fn metals_answers_cover_every_format_it_mipmaps() {
    let mut unmeasured = Vec::new();
    let mut count = 0;
    for &pf in PixelFormat::ALL {
        let answers = measured(pf);
        if not_implemented(pf) {
            assert!(
                answers.is_none(),
                "{pf:?}: answers for a format no blit gives"
            );
            continue;
        }
        if format_problem(pf).is_some() || picks(pf) {
            assert!(
                answers.is_none(),
                "{pf:?}: answers for a format Metal refuses or picks"
            );
            continue;
        }
        match answers {
            Some(answers) => {
                count += 1;
                assert_eq!(answers.cases.len(), 3, "{pf:?}");
                let bits: u32 = answers.fields.iter().map(|f| field_bits(*f)).sum();
                assert_eq!(bits as usize, 8 * answers.texel_bytes, "{pf:?}");
            }
            None => unmeasured.push(pf),
        }
    }
    assert_eq!(count, 34);
    assert!(unmeasured.is_empty(), "{unmeasured:?}");
}

fn field_bits(field: Field) -> u32 {
    match field {
        Field::Unorm(bits) | Field::Snorm(bits) => bits,
        Field::Float {
            exponent,
            mantissa,
            signed,
        } => exponent + mantissa + u32::from(signed),
        Field::Rgb9e5 => 32,
    }
}

/// Both Macs' answers lie within the spread; probe 4's analysis found them at most 1 unit in the
/// last place apart except at the boxes of extremes.
#[test]
fn both_macs_answers_lie_within_the_spread() {
    for &pf in PixelFormat::ALL {
        let Some(answers) = measured(pf) else {
            continue;
        };
        for case in &answers.cases {
            for [m3, m4] in &case.levels {
                assert!(answers.within_spread(m3, m3, m4), "{pf:?}");
                assert!(answers.within_spread(m4, m3, m4), "{pf:?}");
            }
        }
    }
}

/// The first component where the two Macs' answers lie further apart than `apart` raw units,
/// as (case, level, byte offset of the component).
fn far_apart(answers: &MeasuredFormat, width: usize, apart: u64) -> (usize, usize, usize) {
    let read = |bytes: &[u8], at: usize| {
        bytes[at..at + width]
            .iter()
            .rev()
            .fold(0u64, |v, &b| (v << 8) | u64::from(b))
    };
    for (c, case) in answers.cases.iter().enumerate() {
        for (l, [m3, m4]) in case.levels.iter().enumerate() {
            for at in (0..m3.len()).step_by(width) {
                if read(m3, at).abs_diff(read(m4, at)) > apart {
                    return (c, l, at);
                }
            }
        }
    }
    panic!("no components far apart")
}

fn with(bytes: &[u8], at: usize, value: u64, width: usize) -> Vec<u8> {
    let mut out = bytes.to_vec();
    out[at..at + width].copy_from_slice(&value.to_le_bytes()[..width]);
    out
}

/// Below a box of a format's largest and smallest values the two Macs answer far apart (0 and
/// 0.25 in 16-bit float; about 7.45e-9 and 0 in 32-bit float, a level further down); a device's
/// value strictly between the two, past the tolerance of both, is refused.
#[test]
fn a_value_between_the_two_macs_answers_is_refused() {
    for (pf, width, m3_value, m4_value, tolerance) in [
        (PixelFormat::RGBA16Float, 2, 0x0000u64, 0x3400u64, 1u64),
        (PixelFormat::R16Float, 2, 0x0000, 0x3400, 1),
        (PixelFormat::RGBA32Float, 4, 0x3200_0000, 0x0000_0000, 2),
        (PixelFormat::RG32Float, 4, 0x3200_0000, 0x0000_0000, 2),
    ] {
        let answers = measured(pf).unwrap();
        let (c, l, at) = far_apart(&answers, width, 2);
        let [m3, m4] = &answers.cases[c].levels[l];
        assert_eq!(with(m3, at, m3_value, width), *m3, "{pf:?}");
        assert_eq!(with(m4, at, m4_value, width), *m4, "{pf:?}");
        let device = |value| with(m3, at, value, width);
        let accepted = |value| answers.within_spread(&device(value), m3, m4);
        // Both answers are positive, so their raw encodings are in order of value.
        let (low, high) = (m3_value.min(m4_value), m3_value.max(m4_value));
        assert!(accepted(m3_value), "{pf:?}");
        assert!(accepted(m4_value), "{pf:?}");
        assert!(accepted(low + tolerance), "{pf:?}");
        assert!(accepted(high - tolerance), "{pf:?}");
        assert!(!accepted(low + tolerance + 1), "{pf:?}");
        assert!(!accepted(high - tolerance - 1), "{pf:?}");
        assert!(!accepted((low + high) / 2), "{pf:?}");
    }
}

#[test]
fn units_in_the_last_place_follow_each_fields_encoding() {
    let snorm = Field::Snorm(8);
    // -128 and -127 are one code apart (both read -1).
    assert!(snorm.near(0x80, 0x81));
    assert!(!snorm.near(0x80, 0x82));
    assert!(snorm.near(0x00, 0xff));
    assert!(!snorm.near(0x7f, 0x80));
    let unorm = Field::Unorm(10);
    assert!(unorm.near(0x3ff, 0x3fe));
    assert!(!unorm.near(0x3ff, 0x3fd));
    let half = Field::Float {
        exponent: 5,
        mantissa: 10,
        signed: true,
    };
    // -0 and +0 are one value; the smallest denormals of each sign are two units apart.
    assert!(half.near(0x8000, 0x0000));
    assert!(half.near(0x8001, 0x0000));
    assert!(!half.near(0x8001, 0x0001));
    assert!(half.near(0x3c00, 0x3bff));
    let single = Field::Float {
        exponent: 8,
        mantissa: 23,
        signed: true,
    };
    assert!(single.near(0x3f80_0002, 0x3f80_0000));
    assert!(!single.near(0x3f80_0003, 0x3f80_0000));
    let small = Field::Float {
        exponent: 5,
        mantissa: 6,
        signed: false,
    };
    assert!(small.near(0x7bf, 0x7be));
    assert!(!small.near(0x040, 0x03e));
    // RGB9E5: units of the larger exponent; 256 at exponent 16 is 128 at exponent 17.
    let shared = |m: [u64; 3], e: u64| m[0] | (m[1] << 9) | (m[2] << 18) | (e << 27);
    assert!(Field::Rgb9e5.near(shared([256, 0, 2], 16), shared([128, 0, 1], 17)));
    assert!(Field::Rgb9e5.near(shared([256, 0, 4], 16), shared([128, 0, 1], 17)));
    assert!(!Field::Rgb9e5.near(shared([256, 0, 6], 16), shared([128, 0, 1], 17)));
}

/// A chain halves when each level is exactly half the one above on every axis (1 stays 1); the
/// check then uses the measured halving textures only.
#[test]
fn chains_that_halve_every_axis_are_told_apart() {
    assert!(halves([8, 8, 1], 4));
    assert!(halves([8, 4, 1], 4));
    assert!(halves([4, 4, 4], 3));
    assert!(halves([6, 6, 1], 2));
    assert!(!halves([6, 6, 1], 3));
    assert!(!halves([7, 5, 1], 3));
    assert!(halves([7, 5, 1], 1));
    assert!(!halves([8, 8, 3], 2));
}

/// `bind::MipmapFilterChain` computes Metal's measured chains of the packed 4-bit formats.
///
/// `mipmap_answers.txt` holds two columns, an Apple M3 Max on macOS 26.7 and an Apple M4 on
/// 27.0.1, and 27.0.1's rule ([`MipmapRule::Filter`]) is its filter at every level. So the chain
/// under that rule must give the second column for the formats measured on both Macs, which is a
/// two-sided control: the first column differs from it at four of `ABGR4Unorm`'s seven levels, so a
/// chain that reproduced the wrong release, or a comparison that could not tell the two apart,
/// fails here rather than passing quietly.
///
/// `BGRA4Unorm`'s own block was measured on an M3 Max under macOS 27.0.1
/// (`metal2vulkan-conformance/probes/bgra4_write_and_mipmap.m`), so the same rule must give it.
/// Its four 4-bit fields from bit 0 are `ABGR4Unorm`'s, in a channel order neither Metal's filter
/// nor `bind::mipmap_filter`'s rule reads, and the run bears that out: every level came out
/// `ABGR4Unorm`'s own answer on that machine, byte for byte.
#[test]
fn the_computed_chain_gives_the_packed_four_bit_formats_measured_levels() {
    use metal2vulkan::bind::{MipmapFilterChain, MipmapRule};

    let mut checked = 0;
    for pf in [
        PixelFormat::ABGR4Unorm,
        PixelFormat::BGRA4Unorm,
        PixelFormat::A1BGR5Unorm,
    ] {
        let answers = measured(pf).unwrap_or_else(|| panic!("{pf:?} has measured answers"));
        for case in &answers.cases {
            let volume = case.extent[2] > 1;
            let levels = case.levels.len() as u32 + 1;
            let chain = MipmapFilterChain::plan(
                pf,
                case.extent,
                1,
                levels,
                volume,
                false,
                MipmapRule::Filter,
                16,
                u32::MAX,
                [u16::MAX as u32, u16::MAX as u32],
                None,
            )
            .unwrap_or_else(|| panic!("{pf:?} {:?} plans a chain", case.extent));
            let got = chain
                .expected_levels(&case.input)
                .unwrap_or_else(|| panic!("{pf:?} {:?} computes levels", case.extent));
            assert_eq!(got.len(), case.levels.len() + 1, "{pf:?} {:?}", case.extent);
            assert_eq!(got[0], case.input, "{pf:?} {:?} level 0", case.extent);
            for (n, (computed, [_m3, m4])) in got[1..].iter().zip(&case.levels).enumerate() {
                assert_eq!(computed, m4, "{pf:?} {:?} level {}", case.extent, n + 1);
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 21, "every level of three formats' three cases");

    // The control is two-sided: the 26.7 column really does differ, so matching the 27.0.1 one is
    // a reading and not a coincidence of the two being equal everywhere.
    let abgr4 = measured(PixelFormat::ABGR4Unorm).expect("answers");
    let apart = abgr4
        .cases
        .iter()
        .flat_map(|case| &case.levels)
        .filter(|[m3, m4]| m3 != m4)
        .count();
    assert_eq!(apart, 4, "ABGR4Unorm's releases differ at four levels");

    // And `BGRA4Unorm`'s block is its twin's answer, as the run found and the file's comment says.
    let bgra4 = measured(PixelFormat::BGRA4Unorm).expect("answers");
    for (a, b) in bgra4.cases.iter().zip(&abgr4.cases) {
        assert_eq!(a.input, b.input, "the same inputs");
        let twin: Vec<&Vec<u8>> = b.levels.iter().map(|[_, m4]| m4).collect();
        let mine: Vec<&Vec<u8>> = a.levels.iter().map(|[m3, _]| m3).collect();
        assert_eq!(mine, twin);
    }
}
