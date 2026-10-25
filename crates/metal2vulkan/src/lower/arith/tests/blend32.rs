//! A fetched 32-bit float attachment's fast-math blend on the interpreter
//! (`extended_range::blend::blend_float32`), over the very sources and destinations
//! `half-output-blends-r32f`, `-rg32f` and `-rgba32f` compare against plain Metal.
//!
//! **The rule, off those cases' recordings:** Metal rounds the **source** product to the
//! attachment's precision and carries the **destination** product into the sum unrounded,
//! `f(f(Sf · S) + Df · D)` -- one fused multiply-add on the destination term -- where a device's
//! own blend is the mirror image, `f(Sf · S + f(Df · D))`. It is the 16-bit family's rule
//! (`tests::blend16`) a width up, computed in `float` rather than in `half`, and
//! `SourceColor` over `DestinationColor` is the state that shows it: the source product `S · S` is
//! exact in the attachment, so only the destination product's rounding is left and the two orders
//! part by a code.
//!
//! `DestinationAlpha` over `OneMinusDestinationAlpha` adds the alpha class, which is neither
//! order: `f(da · f(f(1 + sa) - da))`, the common factor's form (`extended_range::blend::plan`).
//!
//! **One channel is out, and this module holds that too.** Into `R32Float`, `SourceColor` over
//! `DestinationColor` is Metal's in the *other* order (Metal rounds the destination's product
//! first at one channel, 18 of that site's 256 words apart under this one), so the state is not
//! measured at one channel and `Blend::fitted` refuses it: the device's own blend is Metal's
//! there and keeps it. The 16-bit family found the same width-dependence at
//! `SourceAlphaSaturated` over `OneMinusSourceAlpha`, so neither width's order is the other's.

use super::*;
use crate::lower::arith::extended_range::blend::{blend_float32, Blend};
use crate::metal::{BlendFactor as F, BlendOperation as O, ColorWriteMask, PixelFormat};
use crate::pipeline::lower_inputs::BlendEquation;

/// The pixels one of those cases draws, one source value of the sweep each.
const PIXELS: usize = 256;

/// The source values `half-output-blends-*` sweeps: `k / 16 - 2` for `k` in 0 to 63, every
/// sixteenth between -2 and 2, each exact in `half` and in `float`.
const SOURCES: usize = 64;

/// How far apart two channels of one pixel are in the sweep, so each source value lands at every
/// channel position across the pixels.
const STRIDE: usize = 7;

fn source(index: usize) -> f64 {
    index as f64 / 16.0 - 2.0
}

/// `x` rounded to `float`, nearest even.
fn fl(x: f64) -> f64 {
    f64::from(x as f32)
}

/// The destination codes `half-output-blends-*` draws for a 32-bit float format: xorshift64\*
/// seeded from the format's Metal value, a multiple of 2^-20 in [0, 1) per channel per pixel, each
/// exact in `float`. Restated here so this test owns its inputs; the words are the words of that
/// family's recordings.
struct WideRng(u64);

impl WideRng {
    fn new(seed: u64) -> Self {
        let mut state = seed
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(0x632b_e59b_d9b4_e019);
        if state == 0 {
            state = 1;
        }
        let mut generator = Self(state);
        for _ in 0..8 {
            generator.next();
        }
        generator
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(2_685_821_657_736_338_717)
    }
}

/// The equation of a blend whose color and alpha take the one factor pair, added.
fn added(fs: F, fd: F) -> BlendEquation {
    BlendEquation {
        source_rgb: fs,
        destination_rgb: fd,
        rgb_operation: O::Add,
        source_alpha: fs,
        destination_alpha: fd,
        alpha_operation: O::Add,
    }
}

/// Metal's rounding in a fast-math blend into a 32-bit float attachment at the states
/// `Blend::fitted` holds by measurement (`blend::MEASURED_THIRTY_TWO`), on the interpreter, over
/// the sources and destinations `half-output-blends-rg32f` and `-rgba32f` compare against plain
/// Metal.
///
/// 2,560 words: 512 of `RG32Float` and 1,024 of `RGBA32Float` at
/// `SourceColor`/`DestinationColor`, and all 1,024 of `RGBA32Float` at
/// `DestinationAlpha`/`OneMinusDestinationAlpha`, its 256 alpha words among them.
///
/// Every value here is exact in `f64`, so the reference rounds exactly where Metal's rule rounds
/// and nowhere else: a source is a multiple of `2^-4` below 2 in magnitude, a destination a
/// multiple of `2^-20` below one, their products multiples of `2^-24` and `2^-40` below 4, and
/// every sum of two of those a multiple of `2^-40` below 8 -- 43 bits, inside `f64`'s 53.
#[test]
fn a_measured_fast_blend_rounds_the_source_product_first_in_float() {
    let f4 = vec(float(32), 4);
    let states = [
        (F::SourceColor, F::DestinationColor),
        (F::DestinationAlpha, F::OneMinusDestinationAlpha),
    ];
    // The two formats whose states the measurement fits, with the Metal value their case seeds
    // the destination generator from.
    let formats = [
        (PixelFormat::RG32Float, 2u32, 105u64),
        (PixelFormat::RGBA32Float, 4, 125),
    ];
    let mut words = 0usize;
    for (pf, channels, raw) in formats {
        let mut rng = WideRng::new(raw);
        // The case's destination codes, one row a pixel, and a missing channel's one, as Metal's
        // blend and Vulkan's both read it.
        let destinations: Vec<[f64; 4]> = (0..PIXELS)
            .map(|_| {
                std::array::from_fn(|c| match c < channels as usize {
                    true => f64::from((rng.next() % (1 << 20)) as f32 / (1u32 << 20) as f32),
                    false => 1.0,
                })
            })
            .collect();
        let cases: Vec<Case> = (0..PIXELS)
            .map(|pixel| {
                let s: Vec<u64> = (0..4)
                    .map(|c| {
                        let v = source((pixel + STRIDE * c as usize) % SOURCES);
                        u64::from((v as f32).to_bits())
                    })
                    .collect();
                let d: Vec<u64> = destinations[pixel]
                    .iter()
                    .map(|&v| u64::from((v as f32).to_bits()))
                    .collect();
                vec![s, d]
            })
            .collect();
        for (fs, fd) in states {
            let eq = added(fs, fd);
            let what = format!("{pf:?} {fs:?} {fd:?}");
            let b = Blend::float32(&eq, pf, ColorWriteMask::ALL)
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            assert!(b.fitted(), "{what}: {b:?}");
            // A format without alpha drops the destination's term of
            // `DestinationAlpha`/`OneMinusDestinationAlpha` (its destination alpha is one, so the
            // destination's factor is zero), a lone term `Blend::fitted` holds without a
            // measurement: the contracted pair below is the four-channel format's.
            let dropped = channels < 4 && fs == F::DestinationAlpha;
            // Metal's value in lane `l`: the color class in the three color lanes, and in alpha
            // the attachment's own class where it holds one.
            let metal = |s: [f64; 4], d: [f64; 4], l: usize| -> u64 {
                let alpha = l == 3;
                let (sl, dl) = (s[l], d[l]);
                let (sa, da) = (s[3], d[3]);
                if alpha && channels < 4 {
                    // The lane the attachment does not hold keeps the texel's.
                    return u64::from((dl as f32).to_bits());
                }
                let value = match (fs, alpha) {
                    // The alpha of the destination alpha over its one minus: the common factor's
                    // form, `f(da · f(f(1 + sa) - da))`.
                    (F::DestinationAlpha, true) => fl(da * fl(fl(1.0 + sa) - da)),
                    _ => {
                        let (ks, kd) = match fs {
                            F::SourceColor => (sl, dl),
                            F::DestinationAlpha => (da, fl(1.0 - da)),
                            other => panic!("no factor {other:?} here"),
                        };
                        // The source's product rounded to the attachment, the destination's fused
                        // into the sum.
                        fl(fl(sl * ks) + dl * kd)
                    }
                };
                u64::from((value as f32).to_bits())
            };
            for q in HALF_FORMS {
                check(
                    &format!("the measured blend's {what}, by {q:?}"),
                    &[f4, f4],
                    f4,
                    &|fb, x| {
                        fb.module().set_quantize_form(q);
                        blend_float32(fb, &b, (x[0], x[1]), None)
                    },
                    &cases,
                    &|c| {
                        let s: [f64; 4] = std::array::from_fn(|l| f64::from(f32v(c[0][l])));
                        let d: [f64; 4] = std::array::from_fn(|l| f64::from(f32v(c[1][l])));
                        Some((0..4).map(|l| metal(s, d, l)).collect())
                    },
                    false,
                );
            }
            if !dropped {
                // The words of the site: one pixel a row, the channels the format holds.
                words += PIXELS * channels as usize;
            }
        }
    }
    // The three sites the measurement needs a state for: 512 and 1,024 words of
    // `SourceColor`/`DestinationColor` and 1,024 of
    // `DestinationAlpha`/`OneMinusDestinationAlpha`.
    assert_eq!(words, 2_560, "every word of the measured states' sites");
}

/// A one-channel 32-bit float attachment's `SourceColor` over `DestinationColor` is not a measured
/// state (`blend::MEASURED_THIRTY_TWO`): Metal rounds the destination's product first there, so no
/// form `Blend::fitted` holds is Metal's and the device's own blend keeps the state.
///
/// The same state at two and four channels is fitted (the test above), which is what says the
/// exclusion is the one channel's and not the state's.
#[test]
fn one_channel_is_no_measured_state_of_a_contracted_pair() {
    let eq = added(F::SourceColor, F::DestinationColor);
    let one = Blend::float32(&eq, PixelFormat::R32Float, ColorWriteMask::ALL).expect("planned");
    assert!(
        !one.fitted(),
        "R32Float's SourceColor over DestinationColor: {one:?}"
    );
    for pf in [PixelFormat::RG32Float, PixelFormat::RGBA32Float] {
        let b = Blend::float32(&eq, pf, ColorWriteMask::ALL).expect("planned");
        assert!(b.fitted(), "{pf:?}: {b:?}");
    }
}

/// A 32-bit float attachment's blend color difference is no fitted form, where a 16-bit float
/// one's is (`blend::constant_difference`): the state is measured in no word of
/// `half-output-blends-*32f`, and two of the states whose factors read the blend color do differ
/// from Metal under these forms.
#[test]
fn a_blend_colors_difference_is_no_fitted_form_in_float() {
    for op in [O::Subtract, O::ReverseSubtract] {
        let eq = BlendEquation {
            source_rgb: F::BlendColor,
            destination_rgb: F::OneMinusBlendColor,
            rgb_operation: op,
            source_alpha: F::BlendAlpha,
            destination_alpha: F::OneMinusBlendAlpha,
            alpha_operation: op,
        };
        let wide =
            Blend::float32(&eq, PixelFormat::RGBA32Float, ColorWriteMask::ALL).expect("planned");
        assert!(!wide.fitted(), "RGBA32Float {op:?}: {wide:?}");
        let narrow = Blend::half_float(
            &eq,
            PixelFormat::RGBA16Float,
            ColorWriteMask::ALL,
            crate::lower::arith::extended_range::blend::BlendMath::Fast,
        )
        .expect("planned");
        assert!(narrow.fitted(), "RGBA16Float {op:?}: {narrow:?}");
    }
}
