//! Every uncompressed colour format's texture, ported from
//! `runtime_lane::textures`'s `every_format_makes_a_texture_or_is_refused_typed`.
//!
//! One case a format. Each one makes an 8 by 8 `Shared` 2D texture of its format, read by shaders,
//! at a row pitch one texel wider than its rows, then asks three things of it:
//!
//! - what a **fresh** texture reads (`get_bytes`), which Metal answers as zero;
//! - what it reads back after `replace_region` wrote the case's pattern over its whole level 0;
//! - whether the **bytes between the rows** stayed as they were.
//!
//! The third is why both reads go through [`Side::read_texture_over`] rather than
//! [`Side::read_texture`]: the destination holds [`BEFORE`] before the copy, so a byte the copy did
//! not write is that byte rather than a zero nobody can tell from a written one, and the site's
//! layout compares the gaps ([`Gaps::Compared`]).
//!
//! **What the port adds to the lane.** The lane asserts *rules*: a fresh texture reads zero, the
//! bytes come back, the padding is untouched, a format the device's Metal level describes none of
//! is stopped. It holds no Metal value at all, and it walks all 301 formats in one test, so a
//! format the venue has no texture of leaves the test green with nothing compared. Here Metal's own
//! bytes are recorded per format, and a format the venue honestly lacks stops that format's case
//! alone and counts as [`LacksFeature`](crate::verdict::Verdict::LacksFeature) — neither a pass nor
//! a failure, and visible in the matrix.
//!
//! **What is left in the lane.** Whether a texture of each of the 301 formats is made at all, which
//! the lane checks against two rules (the Metal level's format list, and the formats Metal answers
//! nil for). One case per format would record Metal's answer for each, but the suite's
//! `LacksFeature` outcome arrives only when a case stops, so a case covering many formats would
//! have to swallow a venue's gap to keep going. That is a verdict-model question, not something to
//! decide inside a case, so this family covers the formats whose texels are Metal's own bytes and
//! says so.
//!
//! Nothing here reaches the GPU: a texture is created and the host copies bytes into and out of it.

use crate::case::{Abort, Case, Input, MetalSource, Side, TextureRole, TextureSpec};
use crate::transcript::{Aspect, ByteLayout, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::PixelFormat;

/// The texture's side in texels, which is the lane's extent.
const SIDE: u32 = 8;

/// The byte every read-back destination holds before the copy, so a byte the copy did not write is
/// a recorded value rather than a zero nobody can read.
pub const BEFORE: u8 = 0xee;

/// The formats this family covers: every format of [`PixelFormat::ALL`] whose texels are Metal's
/// own bytes — an uncompressed colour format of one texel a block — with the bytes one texel takes
/// and the name its case holds.
///
/// The table is the case names' home, because a case names itself with a static string. Nothing
/// here may drift: [`tests::the_table_is_exactly_the_formats_whose_texels_are_metals_bytes`] holds
/// it against the predicate over every format Metal describes.
const FORMATS: [(u32, u32, &str); 75] = [
    (1, 1, "texture-round-trip-1"),      // MTLPixelFormatA8Unorm
    (10, 1, "texture-round-trip-10"),    // MTLPixelFormatR8Unorm
    (11, 1, "texture-round-trip-11"),    // MTLPixelFormatR8Unorm_sRGB
    (12, 1, "texture-round-trip-12"),    // MTLPixelFormatR8Snorm
    (13, 1, "texture-round-trip-13"),    // MTLPixelFormatR8Uint
    (14, 1, "texture-round-trip-14"),    // MTLPixelFormatR8Sint
    (20, 2, "texture-round-trip-20"),    // MTLPixelFormatR16Unorm
    (22, 2, "texture-round-trip-22"),    // MTLPixelFormatR16Snorm
    (23, 2, "texture-round-trip-23"),    // MTLPixelFormatR16Uint
    (24, 2, "texture-round-trip-24"),    // MTLPixelFormatR16Sint
    (25, 2, "texture-round-trip-25"),    // MTLPixelFormatR16Float
    (30, 2, "texture-round-trip-30"),    // MTLPixelFormatRG8Unorm
    (31, 2, "texture-round-trip-31"),    // MTLPixelFormatRG8Unorm_sRGB
    (32, 2, "texture-round-trip-32"),    // MTLPixelFormatRG8Snorm
    (33, 2, "texture-round-trip-33"),    // MTLPixelFormatRG8Uint
    (34, 2, "texture-round-trip-34"),    // MTLPixelFormatRG8Sint
    (40, 2, "texture-round-trip-40"),    // MTLPixelFormatB5G6R5Unorm
    (41, 2, "texture-round-trip-41"),    // MTLPixelFormatA1BGR5Unorm
    (42, 2, "texture-round-trip-42"),    // MTLPixelFormatABGR4Unorm
    (43, 2, "texture-round-trip-43"),    // MTLPixelFormatBGR5A1Unorm
    (53, 4, "texture-round-trip-53"),    // MTLPixelFormatR32Uint
    (54, 4, "texture-round-trip-54"),    // MTLPixelFormatR32Sint
    (55, 4, "texture-round-trip-55"),    // MTLPixelFormatR32Float
    (60, 4, "texture-round-trip-60"),    // MTLPixelFormatRG16Unorm
    (62, 4, "texture-round-trip-62"),    // MTLPixelFormatRG16Snorm
    (63, 4, "texture-round-trip-63"),    // MTLPixelFormatRG16Uint
    (64, 4, "texture-round-trip-64"),    // MTLPixelFormatRG16Sint
    (65, 4, "texture-round-trip-65"),    // MTLPixelFormatRG16Float
    (70, 4, "texture-round-trip-70"),    // MTLPixelFormatRGBA8Unorm
    (71, 4, "texture-round-trip-71"),    // MTLPixelFormatRGBA8Unorm_sRGB
    (72, 4, "texture-round-trip-72"),    // MTLPixelFormatRGBA8Snorm
    (73, 4, "texture-round-trip-73"),    // MTLPixelFormatRGBA8Uint
    (74, 4, "texture-round-trip-74"),    // MTLPixelFormatRGBA8Sint
    (80, 4, "texture-round-trip-80"),    // MTLPixelFormatBGRA8Unorm
    (81, 4, "texture-round-trip-81"),    // MTLPixelFormatBGRA8Unorm_sRGB
    (90, 4, "texture-round-trip-90"),    // MTLPixelFormatRGB10A2Unorm
    (91, 4, "texture-round-trip-91"),    // MTLPixelFormatRGB10A2Uint
    (92, 4, "texture-round-trip-92"),    // MTLPixelFormatRG11B10Float
    (93, 4, "texture-round-trip-93"),    // MTLPixelFormatRGB9E5Float
    (94, 4, "texture-round-trip-94"),    // MTLPixelFormatBGR10A2Unorm
    (554, 4, "texture-round-trip-554"),  // MTLPixelFormatBGR10_XR
    (555, 4, "texture-round-trip-555"),  // MTLPixelFormatBGR10_XR_sRGB
    (103, 8, "texture-round-trip-103"),  // MTLPixelFormatRG32Uint
    (104, 8, "texture-round-trip-104"),  // MTLPixelFormatRG32Sint
    (105, 8, "texture-round-trip-105"),  // MTLPixelFormatRG32Float
    (110, 8, "texture-round-trip-110"),  // MTLPixelFormatRGBA16Unorm
    (112, 8, "texture-round-trip-112"),  // MTLPixelFormatRGBA16Snorm
    (113, 8, "texture-round-trip-113"),  // MTLPixelFormatRGBA16Uint
    (114, 8, "texture-round-trip-114"),  // MTLPixelFormatRGBA16Sint
    (115, 8, "texture-round-trip-115"),  // MTLPixelFormatRGBA16Float
    (552, 8, "texture-round-trip-552"),  // MTLPixelFormatBGRA10_XR
    (553, 8, "texture-round-trip-553"),  // MTLPixelFormatBGRA10_XR_sRGB
    (123, 16, "texture-round-trip-123"), // MTLPixelFormatRGBA32Uint
    (124, 16, "texture-round-trip-124"), // MTLPixelFormatRGBA32Sint
    (125, 16, "texture-round-trip-125"), // MTLPixelFormatRGBA32Float
    (84, 4, "texture-round-trip-84"),    // MTLPixelFormatARGB8Unorm
    (26, 2, "texture-round-trip-26"),    // MTLPixelFormatR16Bfloat
    (66, 4, "texture-round-trip-66"),    // MTLPixelFormatRG16Bfloat
    (116, 8, "texture-round-trip-116"),  // MTLPixelFormatRGBA16Bfloat
    (300, 4, "texture-round-trip-300"),  // MTLPixelFormatRGBX8Unorm
    (301, 4, "texture-round-trip-301"),  // MTLPixelFormatRGBX8Unorm_sRGB
    (302, 4, "texture-round-trip-302"),  // MTLPixelFormatRGBX8Snorm
    (303, 4, "texture-round-trip-303"),  // MTLPixelFormatRGBX8Uint
    (304, 4, "texture-round-trip-304"),  // MTLPixelFormatRGBX8Sint
    (313, 8, "texture-round-trip-313"),  // MTLPixelFormatRGBX16Uint
    (314, 8, "texture-round-trip-314"),  // MTLPixelFormatRGBX16Sint
    (315, 8, "texture-round-trip-315"),  // MTLPixelFormatRGBX16Float
    (323, 16, "texture-round-trip-323"), // MTLPixelFormatRGBX32Uint
    (324, 16, "texture-round-trip-324"), // MTLPixelFormatRGBX32Sint
    (325, 16, "texture-round-trip-325"), // MTLPixelFormatRGBX32Float
    (576, 2, "texture-round-trip-576"),  // MTLPixelFormatR10Unorm_X6
    (577, 2, "texture-round-trip-577"),  // MTLPixelFormatR10Unorm_X6_sRGB
    (578, 4, "texture-round-trip-578"),  // MTLPixelFormatRG10Unorm_X12
    (579, 4, "texture-round-trip-579"),  // MTLPixelFormatRG10Unorm_X12_sRGB
    (586, 4, "texture-round-trip-586"),  // MTLPixelFormatRGB10A2Unorm_sRGB
];

/// The bytes of one texel of `pf`, for the uncompressed colour formats this family round-trips:
/// none for a format whose image does not hold Metal's bytes, whose host copies the runtime
/// refuses. The lane's own predicate, over the translator's public format table.
///
/// It is the table's guard and nothing else reads it: [`FORMATS`] holds the names a case must
/// answer with as static strings, so the list cannot be built from the predicate at run time, and
/// this is what keeps the two from parting.
#[cfg(test)]
fn texel_bytes(pf: PixelFormat) -> Option<u32> {
    let block = metal2vulkan::metal::format::byte_format(pf)?.block()?;
    let compressed =
        block.texels > 1 || (block.packed_bits.is_none() && block.component_bits.is_none());
    (block.color.is_some() && !compressed).then_some(block.size)
}

/// `n` bytes of the lane's pattern, seeded by `seed`.
fn pattern(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|at| (at as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

/// A case: one format's texture.
pub struct TextureRoundTrip {
    name: &'static str,
    format: PixelFormat,
    /// The bytes one texel takes.
    texel: u32,
}

/// Every case of the family: one a format.
pub fn all() -> Vec<TextureRoundTrip> {
    FORMATS
        .iter()
        .filter_map(|&(raw, texel, name)| {
            // A raw value the translator's `MTLPixelFormat` no longer names makes no case, and
            // the table's own guard test then fails rather than the family quietly shrinking.
            Some(TextureRoundTrip {
                name,
                format: PixelFormat::try_from(raw).ok()?,
                texel,
            })
        })
        .collect()
}

impl TextureRoundTrip {
    /// The bytes one row of texels takes.
    fn row(&self) -> u32 {
        SIDE * self.texel
    }

    /// The row pitch the case reads and writes at: its rows and one texel of padding after each,
    /// which is what makes the gaps a question.
    fn pitch(&self) -> u32 {
        self.row() + self.texel
    }

    /// The whole destination of a read-back, before the copy.
    fn before(&self) -> Vec<u8> {
        vec![BEFORE; (self.pitch() * SIDE) as usize]
    }

    /// The bytes `replace_region` writes: the lane's pattern over the whole destination, padding
    /// included, so what lands in the gaps is the copy's business and not the case's.
    fn written(&self) -> Vec<u8> {
        pattern((self.pitch() * SIDE) as usize, self.format.as_raw() as u8)
    }

    /// The texture the case makes: one level, `Shader_Read` usage and `Shared` storage, which is
    /// what `MTLTextureDescriptor`'s own defaults give and what the lane's descriptor holds.
    fn spec(&self) -> TextureSpec {
        TextureSpec::new(self.format, [SIDE, SIDE], self.pitch(), TextureRole::Read)
    }

    /// The layout a read-back's bytes are compared under: `SIDE` rows of the row's bytes at the
    /// case's pitch, byte by byte, with the bytes between the rows compared too.
    fn layout(&self) -> ByteLayout {
        ByteLayout {
            region: Region::Texture {
                level: 0,
                slice: 0,
                aspect: Aspect::Color,
                sample: 0,
                origin: [0, 0, 0],
                size: [SIDE, SIDE, 1],
            },
            element: Element::U8,
            row_offset: 0,
            row_elements: self.row(),
            row_pitch: self.pitch(),
            rows: SIDE,
            image_pitch: self.pitch() * SIDE,
            images: 1,
            // The question the padding asks: a copy that wrote a byte between the rows differs
            // from Metal, and this is what says so.
            gaps: Gaps::Compared,
        }
    }
}

/// The site a creator's answer records at, with the side's own error kept out of it.
///
/// Metal has one way to refuse a texture — `newTextureWithDescriptor:` answers nil — and no typed
/// error at all, while the runtime refuses with one of several. So the comparable fact is whether a
/// texture of the format was made, and the record holds that and no more: a side that made none
/// says `Err(NotMade)` whatever it would have called the reason. Our *reason* has no Metal
/// counterpart, so it stays out, as the design says an answer with no Metal counterpart must.
///
/// This keeps the verdict that matters. A format Metal makes a texture of and we refuse is Metal
/// answering `Ok` against our `Err`, which the comparison reads as
/// [`Refused`](crate::verdict::Verdict::Refused) — never a pass, and never buried in a count.
///
/// What it costs: our typed error is not in the record, so a `known-gaps.tsv` row for such a
/// format reads `Err(NotMade)` and cannot tell one refusal of ours from another at the same site.
/// The reason is one call away in the runtime and the lane keeps it, so the cost is a reader's
/// convenience rather than a fact nobody holds — and it buys the alternative's absence: an error
/// payload in the record would read as a divergence on the day Metal refuses a format too, when
/// the two sides in fact agree.
fn made_value(made: bool) -> Value {
    if made {
        Value::ok(Value::Absent)
    } else {
        Value::error("NotMade", Value::Absent)
    }
}

impl Case for TextureRoundTrip {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Ok(vec![
            Input::bytes("format", &self.format.as_raw().to_le_bytes()),
            Input::bytes(
                "extent-and-pitch",
                &[SIDE, SIDE, self.pitch()]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
            Input::bytes("before", &self.before()),
            Input::bytes("replace_region:bytes", &self.written()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        // Made with no bytes, because what a *fresh* texture reads is the first question.
        let texture = match side.texture_2d_in_role(&self.spec(), &[]) {
            Ok(texture) => {
                sites.value("texture", made_value(true));
                texture
            }
            Err(failure) => {
                sites.value("texture", made_value(false));
                // The run stops: a format with no texture has no bytes to read, and stopping is
                // what turns a venue's own gap into `LacksFeature` rather than a difference.
                return Err(Abort {
                    at: "texture".to_owned(),
                    failure,
                });
            }
        };
        let read = |side: &mut dyn Side, sites: &mut Sites, at: &str| -> Result<Vec<u8>, Abort> {
            match side.read_texture_over(texture, [SIDE, SIDE], self.pitch(), &self.before()) {
                Ok(bytes) => Ok(bytes),
                Err(failure) => {
                    sites.push(Node::leaf(at, Observed::value(failure.value.clone())));
                    Err(Abort {
                        at: at.to_owned(),
                        failure,
                    })
                }
            }
        };
        let fresh = read(side, sites, "fresh")?;
        sites.leaf("fresh", Observed::bytes(fresh, self.layout()));
        match side.replace_region(texture, [SIDE, SIDE], self.pitch(), &self.written()) {
            Ok(()) => sites.value("replace_region", Value::ok(Value::Absent)),
            Err(failure) => {
                sites.push(Node::leaf(
                    "replace_region",
                    Observed::value(failure.value.clone()),
                ));
                return Err(Abort {
                    at: "replace_region".to_owned(),
                    failure,
                });
            }
        }
        let back = read(side, sites, "round-trip")?;
        sites.leaf("round-trip", Observed::bytes(back, self.layout()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The table is exactly the formats whose texels are Metal's own bytes, with the bytes a texel
    /// each one takes. Without this the table would be a hand-kept list that a new format or a
    /// changed block silently falls out of, and the family's coverage would shrink unannounced.
    #[test]
    fn the_table_is_exactly_the_formats_whose_texels_are_metals_bytes() {
        let tabled: BTreeSet<u32> = FORMATS.iter().map(|&(raw, _, _)| raw).collect();
        assert_eq!(tabled.len(), FORMATS.len(), "the table repeats a format");
        let mut wanted = BTreeSet::new();
        for &pf in PixelFormat::ALL {
            let Some(bytes) = texel_bytes(pf) else {
                continue;
            };
            wanted.insert(pf.as_raw());
            let row = FORMATS
                .iter()
                .find(|&&(raw, _, _)| raw == pf.as_raw())
                .unwrap_or_else(|| panic!("{pf:?} is not in the table"));
            assert_eq!(row.1, bytes, "{pf:?} takes other bytes a texel");
        }
        let extra: Vec<&u32> = tabled.difference(&wanted).collect();
        assert!(
            extra.is_empty(),
            "the table holds {extra:?}, which no longer round-trip"
        );
    }

    /// Every case's name is its own, and names its format, so a recording cannot be read as
    /// another format's.
    #[test]
    fn every_case_names_its_own_format() {
        let cases = all();
        assert_eq!(cases.len(), FORMATS.len(), "a format made no case");
        let names: BTreeSet<&str> = cases.iter().map(|case| case.name).collect();
        assert_eq!(names.len(), cases.len());
        for case in &cases {
            assert!(
                case.name.ends_with(&format!("-{}", case.format.as_raw())),
                "{} does not name {:?}",
                case.name,
                case.format
            );
        }
    }

    /// The layout a read-back compares under covers every byte the read fills, names the byte as
    /// its element, and compares the padding. A layout that stopped at the rows would let a copy
    /// scribble between them unnoticed, which is the assertion the lane makes and this port must
    /// keep.
    #[test]
    fn the_layout_covers_the_padding() {
        for case in all() {
            let layout = case.layout();
            assert_eq!(layout.gaps, Gaps::Compared, "{}", case.name);
            assert_eq!(
                layout.extent(),
                case.before().len() - case.texel as usize,
                "{}",
                case.name
            );
            assert_eq!(layout.row_elements + case.texel, layout.row_pitch);
        }
    }

    /// The pattern the case writes differs from the byte the destination holds before a read, so a
    /// round trip that copied nothing reads as a difference rather than as a match.
    #[test]
    fn the_written_pattern_is_not_the_byte_before_it() {
        for case in all() {
            let written = case.written();
            assert!(written.iter().any(|&byte| byte != BEFORE), "{}", case.name);
        }
    }
}
