//! Interpreter tests of a snapshot's codes (`lower::fragment`'s `snapshot`, `SnapshotForm`): the
//! copy's encode of a fetched texel and a read's decode give back, bit for bit, every value a
//! fetch of a texel of the form's formats returns, and Metal's value where a device's decode is
//! an ulp off.

use super::*;
use crate::lower::fragment::{snapshot_decode, snapshot_encode};
use crate::pipeline::sym::SnapshotForm;

fn v4f() -> Form {
    vec(float(32), 4)
}

fn v4u() -> Form {
    vec(int(32), 4)
}

/// The copy's encode and a read's decode of `form`, back to the input's type.
fn round_trip(form: SnapshotForm) -> impl Fn(&mut FunctionBuilder<'_>, &[Value]) -> BResult<Value> {
    move |fb, x| {
        let texel = fb.value_ty(x[0])?;
        let codes = snapshot_encode(fb, form, x[0])?;
        snapshot_decode(fb, form, codes, texel)
    }
}

/// `c / d` correctly rounded to `f32`: IEEE division of two integers `f32` holds exactly.
fn exact(c: u32, d: u32) -> u32 {
    (c as f32 / d as f32).to_bits()
}

/// Every lane value in `values`, four to a case, the last case padded with the first values.
fn cases(values: &[u64]) -> Vec<Case> {
    values
        .chunks(4)
        .map(|c| {
            let mut lanes = c.to_vec();
            lanes.extend(values.iter().take(4 - c.len()));
            vec![lanes]
        })
        .collect()
}

/// Bit-for-bit agreement, NaNs included.
fn same(g: u64, w: u64) -> bool {
    g == w
}

/// An unsigned normalized code's value comes back exactly, and a device's value an ulp off its
/// code's (lavapipe's reciprocal) comes back as Metal's, `c / (2^b - 1)` correctly rounded.
#[test]
fn unsigned_normalized_codes_come_back() {
    for (form, d) in [
        (SnapshotForm::Unorm8, 255u32),
        (SnapshotForm::Unorm16, 65535),
    ] {
        let step = if d > 255 { 7 } else { 1 };
        let codes: Vec<u32> = (0..=d).step_by(step).chain([d - 1, d]).collect();
        let values: Vec<u64> = codes.iter().map(|&c| u64::from(exact(c, d))).collect();
        let body = round_trip(form);
        check_accept(
            &format!("{form:?}"),
            &[v4f()],
            v4f(),
            &body,
            &cases(&values),
            &|c| Some(c[0].clone()),
            &same,
        );
        // The reciprocal's product, an ulp off on some codes.
        let recip = (1.0f32 / d as f32).to_bits();
        let off: Vec<u64> = codes
            .iter()
            .map(|&c| u64::from((c as f32 * f32::from_bits(recip)).to_bits()))
            .collect();
        let wanted: Vec<u64> = codes.iter().map(|&c| u64::from(exact(c, d))).collect();
        let off_cases = cases(&off);
        let wanted_cases = cases(&wanted);
        check_accept(
            &format!("{form:?} off an ulp"),
            &[v4f()],
            v4f(),
            &body,
            &off_cases,
            &|c| {
                let i = off_cases.iter().position(|o| o == c)?;
                Some(wanted_cases[i][0].clone())
            },
            &same,
        );
    }
}

/// A signed normalized code's value comes back exactly: `max(c / (2^(b-1) - 1), -1)` for every
/// code, the lowest and the one above it both `-1`.
#[test]
fn signed_normalized_codes_come_back() {
    for (form, d) in [
        (SnapshotForm::Snorm8, 127u32),
        (SnapshotForm::Snorm16, 32767),
    ] {
        let step = if d > 127 { 7 } else { 1 };
        let lowest = -(d as i64) - 1;
        let codes: Vec<i64> = (lowest..=i64::from(d))
            .step_by(step)
            .chain([lowest, lowest + 1, -1, 0, 1, i64::from(d)])
            .collect();
        let values: Vec<u64> = codes
            .iter()
            .map(|&c| {
                let m = exact(c.unsigned_abs().min(u64::from(d)) as u32, d);
                let v = if c < 0 { m | 0x8000_0000 } else { m };
                u64::from(v)
            })
            .collect();
        let body = round_trip(form);
        check_accept(
            &format!("{form:?}"),
            &[v4f()],
            v4f(),
            &body,
            &cases(&values),
            &|c| Some(c[0].clone()),
            &same,
        );
    }
}

/// Every half value widened comes back bit for bit: zeros, denormals, infinities, and each NaN
/// payload as the widening places it.
#[test]
fn half_codes_come_back() {
    let values: Vec<u64> = (0..=u16::MAX)
        .map(|h| {
            let b = crate::lower::arith::narrow::f32_of_half_bits(u32::from(h));
            u64::from(b)
        })
        .collect();
    let body = round_trip(SnapshotForm::Half);
    check_accept(
        "Half",
        &[v4f()],
        v4f(),
        &body,
        &cases(&values),
        &|c| Some(c[0].clone()),
        &same,
    );
}

/// Every integer code comes back: an unsigned one as it is, a signed one sign-extended.
#[test]
fn integer_codes_come_back() {
    for (form, bits, signed) in [
        (SnapshotForm::Uint8, 8u32, false),
        (SnapshotForm::Sint8, 8, true),
        (SnapshotForm::Uint16, 16, false),
        (SnapshotForm::Sint16, 16, true),
    ] {
        let step = if bits > 8 { 3 } else { 1 };
        let values: Vec<u64> = (0..1u64 << bits)
            .step_by(step)
            .map(|c| match signed {
                // Sign-extended into the 32-bit lane.
                true => u64::from((((c as u32) << (32 - bits)) as i32 >> (32 - bits)) as u32),
                false => c,
            })
            .collect();
        let body = round_trip(form);
        check_accept(
            &format!("{form:?}"),
            &[v4u()],
            v4u(),
            &body,
            &cases(&values),
            &|c| Some(c[0].clone()),
            &same,
        );
    }
}

/// The bits form keeps any lanes, NaNs included.
#[test]
fn bits_come_back() {
    let values: Vec<u64> = [
        0u32,
        1,
        0x7FC0_0000,
        0xFFFF_FFFF,
        0x8000_0000,
        0x3F80_0000,
        0x7F80_0001,
    ]
    .iter()
    .map(|&b| u64::from(b))
    .collect();
    let body = round_trip(SnapshotForm::Bits);
    check_accept(
        "Bits",
        &[v4f()],
        v4f(),
        &body,
        &cases(&values),
        &|c| Some(c[0].clone()),
        &same,
    );
}

/// A snapshot of a 16-bit float texture holds the texel's own 16 bits, from the slot's unsigned
/// integer alias, and the read widens them: each of the 65,536 half codes comes back as itself in
/// a `half` result, the 2,046 NaN codes included, so a NaN's sign, payload and signalling bit
/// survive the copy (`lower::fragment`'s `snapshot::fetches_half_bits`).
///
/// Without that path the copy fetches the slot's image and the device decodes the texel first: a
/// device whose `f32` decode canonicalizes a NaN hands one `0x7fc00000` for every NaN code, and
/// `encode` then stores the one code `0x7e00` ([`a_canonicalized_nan_loses_its_code`], the gap
/// this closes).
#[test]
fn half_bits_come_back_through_a_snapshot_for_every_code() {
    use crate::lower::fragment::snapshot_mask_codes;
    use crate::lower::texture::{alias_halves, TexelConv, TexelForm};
    // An `RGBA16Float` texture's class: the half-float family, every channel present.
    let class = crate::metal::texel_format::texel_format(crate::metal::PixelFormat::RGBA16Float).0;
    let form = TexelForm {
        conv: TexelConv::F16,
        lanes: 4,
    };
    // The copy's store of the alias's fetched lanes, and the read's widening of the snapshot's
    // codes: the two halves of the path, composed as the module runs them.
    let body = move |fb: &mut FunctionBuilder<'_>, x: &[Value]| {
        let word = fb.u32_const(class)?;
        let word = fb.module().const_value(word)?;
        let codes = snapshot_mask_codes(fb, SnapshotForm::Half, x[0])?;
        alias_halves(fb, word, form, codes)
    };
    let values: Vec<u64> = (0..=u16::MAX).map(u64::from).collect();
    check_accept(
        "half bits through a snapshot",
        &[v4u()],
        vec(float(16), 4),
        &body,
        &cases(&values),
        &|c| Some(c[0].clone()),
        &same,
    );
}

/// What the copy's fetch of the slot's own image loses on a device whose `f32` decode
/// canonicalizes a NaN: every NaN code's texel arrives as `0x7fc00000`, which `encode` stores as
/// the one code `0x7e00`, so the read hands back that one quiet positive NaN for all 2,046 of
/// them. The bound for `docs/FAST-PATHS.md`'s `half_texel_bits` row, and what
/// [`half_bits_come_back_through_a_snapshot_for_every_code`] closes.
#[test]
fn a_canonicalized_nan_loses_its_code() {
    let canonical = u64::from(f32::NAN.to_bits());
    assert_eq!(canonical, 0x7FC0_0000);
    let body = round_trip(SnapshotForm::Half);
    let widened = crate::lower::arith::narrow::f32_of_half_bits(0x7E00);
    check_accept(
        "a canonicalized NaN",
        &[v4f()],
        v4f(),
        &body,
        &[vec![vec![canonical; 4]]],
        &|_| Some(vec![u64::from(widened); 4]),
        &same,
    );
    // No NaN code's own widening is that value, so the code is gone, not merely requantized.
    let kept = (0..=u16::MAX)
        .filter(|h| h & 0x7C00 == 0x7C00 && h & 0x3FF != 0)
        .filter(|h| crate::lower::arith::narrow::f32_of_half_bits(u32::from(*h)) == widened)
        .count();
    assert_eq!(kept, 1, "only `0x7e00` widens to the canonical NaN");
}
