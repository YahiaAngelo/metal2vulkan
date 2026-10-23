//! How block-compressed textures move their blocks, ported from
//! `runtime_lane::compressed_moves`'s first lane.
//!
//! The consumer's x86 guests hold BC textures, and corrupt texels on NVIDIA were what made this a
//! lane. Its subject is every BC format's blocks moving through shapes whose levels end in partial
//! blocks: a 2D texture 13 x 7 of four levels, a 2D array 10 x 6 of three slices and two levels, a
//! cube 12 x 12 of three levels, a 3D texture 9 x 5 x 3 of two levels, and a 2D texture 8 x 8
//! whose last two levels are smaller than a block. Per shape, five steps ([`Step`]):
//!
//! - `put`: `replaceRegion:` of every level and slice whole, rows padded two blocks past their
//!   blocks and a 3D image one block past its rows, then one block replaced alone at block (1, 0)
//!   of level 0; read back by `getBytes:` of every level and slice whole, rows three blocks past
//!   and a 3D image two blocks past, into bytes of `0xcd`;
//! - `getsub`: `getBytes:` of each level wider than a block, from texel x 4 to its edges;
//! - `out`: a blit of every level and slice into one buffer of `0xcd`, each five blocks past the
//!   last, rows one block past their blocks;
//! - `in`: a blit of a buffer into a second, private texture, each level and slice three blocks
//!   past the last, rows two blocks past, then that texture blitted out tight;
//! - `t2t`: the first texture copied whole into a third, then a block-aligned region of the second
//!   copied over part of it, then the third blitted out tight.
//!
//! **What the port changes, and why it is the point.** The lane compares our bytes with a byte
//! model written in Rust ([`Model`]), and ties that model to Metal by a **single SHA-256 of a line
//! transcript** — 350 lines of `"<format> <shape> <step>: <n> bytes, sha <six bytes>"`, measured by
//! our own probe on an Apple **M4**. So the lane could say that our bytes and the model disagree,
//! never which of the two had moved, and its one Metal fact was a six-byte digest per step from a
//! machine we no longer have. Here record mode runs the same steps on **this M3 Max** and each
//! recording holds Metal's own bytes, padding included, per level and per slice. The model stays as
//! a cross-check (the test `every_recorded_move_is_the_lanes_byte_model`), which is the one direction in
//! which a difference says something: either Apple's copies moved between the chips or the releases,
//! or the model was never Metal's.
//!
//! **One case per format**, so a venue that holds no texture of one answers its typed refusal there
//! and the suite counts that format as [`LacksFeature`](crate::verdict::Verdict::LacksFeature). The
//! lane walks fourteen formats in one test, where a device missing one would take the whole lane
//! down. The port also covers BC6H and BC7, which no other case in the suite reaches.
//!
//! Every byte of every read-back is compared, the padding between the rows included
//! ([`Gaps::Compared`]): the lane asserts the bytes past each row's blocks stay `0xcd`, and here
//! that is the recorded value rather than an assertion beside it.
//!
//! **Not ported, and why.** The lane's second test adds a 2D texture of 1030 x 520 and all eleven
//! levels, whose level 0 alone is over 500 KiB of blocks — past the recording's whole-value ceiling,
//! so it wants the digest-plus-window form no case has exercised yet. Its other three shapes'
//! worth is covered by [`crate::cases::compressed_views`]. Stated here rather than left implicit.
//!
//! Nothing here can fault a GPU: host copies and blit copies between textures the program makes, no
//! dispatch, no draw, no loop, and every region block-aligned and inside its level.

use crate::allowance::Allowance;
use crate::case::{made, Abort, Case, Failure, Input, MetalSource, Side};
use crate::transcript::{Aspect, ByteLayout, Element, Gaps, Node, Observed, Region, Sites, Value};
use crate::work::{Plane, Rect, Resource, Step as WorkStep, TextureKind, TextureSpec, Work};
use metal2vulkan_runtime::Degradation;

/// The byte a read-back's buffer holds before the copy, so a byte no block reached is a recorded
/// value and not whatever was there.
pub const BEFORE: u8 = 0xcd;

/// One BC format: the case's name, the raw `MTLPixelFormat` and its block's bytes.
struct Format {
    name: &'static str,
    /// The raw `MTLPixelFormat`, as `metal2vulkan::metal::PixelFormat` numbers it.
    format: u32,
    block: u64,
}

/// Every BC format, in the lane's order, because a shape's seed is its place in that order and the
/// bytes must stay the bytes the lane measured.
const FORMATS: [Format; 14] = [
    Format {
        name: "compressed-moves-bc1-rgba",
        format: 130,
        block: 8,
    },
    Format {
        name: "compressed-moves-bc1-rgba-srgb",
        format: 131,
        block: 8,
    },
    Format {
        name: "compressed-moves-bc2-rgba",
        format: 132,
        block: 16,
    },
    Format {
        name: "compressed-moves-bc2-rgba-srgb",
        format: 133,
        block: 16,
    },
    Format {
        name: "compressed-moves-bc3-rgba",
        format: 134,
        block: 16,
    },
    Format {
        name: "compressed-moves-bc3-rgba-srgb",
        format: 135,
        block: 16,
    },
    Format {
        name: "compressed-moves-bc4-r-unorm",
        format: 140,
        block: 8,
    },
    Format {
        name: "compressed-moves-bc4-r-snorm",
        format: 141,
        block: 8,
    },
    Format {
        name: "compressed-moves-bc5-rg-unorm",
        format: 142,
        block: 16,
    },
    Format {
        name: "compressed-moves-bc5-rg-snorm",
        format: 143,
        block: 16,
    },
    Format {
        name: "compressed-moves-bc6h-rgb-float",
        format: 150,
        block: 16,
    },
    Format {
        name: "compressed-moves-bc6h-rgb-ufloat",
        format: 151,
        block: 16,
    },
    Format {
        name: "compressed-moves-bc7-rgba-unorm",
        format: 152,
        block: 16,
    },
    Format {
        name: "compressed-moves-bc7-rgba-unorm-srgb",
        format: 153,
        block: 16,
    },
];

/// A texture's shape: its kind, its size in texels (the third axis a 3D texture's depth), its array
/// length and its levels. The lane's five, unchanged, because a shape's bytes come from a seeded
/// stream keyed to its place.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub name: &'static str,
    pub kind: TextureKind,
    pub size: [u64; 3],
    /// `arrayLength`; a cube's is 1 and its faces are its slices.
    pub array: u64,
    pub levels: u64,
}

/// The lane's five shapes, in its order.
pub const SHAPES: [Shape; 5] = [
    Shape {
        name: "2d-13x7",
        kind: TextureKind::Type2D,
        size: [13, 7, 1],
        array: 1,
        levels: 4,
    },
    Shape {
        name: "array-10x6x3",
        kind: TextureKind::Type2DArray,
        size: [10, 6, 1],
        array: 3,
        levels: 2,
    },
    Shape {
        name: "cube-12",
        kind: TextureKind::Cube,
        size: [12, 12, 1],
        array: 1,
        levels: 3,
    },
    Shape {
        name: "3d-9x5x3",
        kind: TextureKind::Type3D,
        size: [9, 5, 3],
        array: 1,
        levels: 2,
    },
    Shape {
        name: "2d-8x8",
        kind: TextureKind::Type2D,
        size: [8, 8, 1],
        array: 1,
        levels: 4,
    },
];

impl Shape {
    /// The slices a copy may name: a cube's six faces, otherwise the array length.
    pub fn slices(&self) -> u64 {
        match self.kind {
            TextureKind::Cube => 6,
            _ => self.array,
        }
    }

    /// Whether the shape is a 3D texture, whose copies carry a `bytesPerImage`.
    pub fn three_d(&self) -> bool {
        self.kind == TextureKind::Type3D
    }

    /// Level `l`'s size in texels.
    pub fn level(&self, l: u64) -> [u64; 3] {
        let [w, h, d] = self.size;
        [(w >> l).max(1), (h >> l).max(1), (d >> l).max(1)]
    }

    /// Level `l`'s size in blocks: its texels rounded up to 4 x 4, its depth as it is.
    pub fn blocks(&self, l: u64) -> [u64; 3] {
        let [w, h, d] = self.level(l);
        [w.div_ceil(4), h.div_ceil(4), d]
    }

    /// Every (level, slice), levels outermost, as the lane walks them.
    pub fn each(&self) -> Vec<(u64, u64)> {
        (0..self.levels)
            .flat_map(|l| (0..self.slices()).map(move |s| (l, s)))
            .collect()
    }

    /// The `bytesPerImage` of a level `l` copy whose rows are `row` apart, `extra` past its rows:
    /// a 3D texture's own images, 0 for a 2D texture, and a slice's rows for an array or a cube
    /// (which Metal takes but which names one image).
    pub fn image(&self, l: u64, row: u64, extra: u64) -> u64 {
        match self.kind {
            TextureKind::Type3D => row * self.blocks(l)[1] + extra,
            TextureKind::Type2D => 0,
            _ => row * self.blocks(l)[1],
        }
    }

    /// The texture the shape names, in `format`, private where `private`.
    pub fn spec(&self, format: u32, private: bool) -> TextureSpec {
        let [w, h, d] = self.size;
        let mut spec = TextureSpec::texture_2d(format, w, h).levels(self.levels);
        spec = match self.kind {
            TextureKind::Cube => spec.cube(w),
            TextureKind::Type3D => spec.volume(d),
            TextureKind::Type2DArray => spec.array(self.array),
            _ => spec,
        };
        if private {
            spec
        } else {
            spec.shared()
        }
    }
}

/// A splitmix64 stream of bytes: the lane's own, so a shape's blocks are the lane's blocks.
pub struct Rng(pub u64);

impl Rng {
    /// The next `n` bytes.
    pub fn bytes(&mut self, n: u64) -> Vec<u8> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                (z ^ (z >> 31)) as u8
            })
            .collect()
    }
}

/// A texture's blocks as Metal's copies leave them: `(level, slice, z, y, x)` to the block's bytes.
///
/// It is the lane's model, kept whole, because the whole point of the port is to hold Metal's own
/// recorded bytes against it.
#[derive(Clone, Default)]
pub struct Model(std::collections::BTreeMap<(u64, u64, u64, u64, u64), Vec<u8>>);

impl Model {
    /// The blocks of `count` (x, y, z) at block `at` of `(l, s)` read out of `bytes` at `offset`,
    /// rows `row` and images `image` apart.
    pub fn take(
        &mut self,
        (l, s): (u64, u64),
        at: [u64; 3],
        count: [u64; 3],
        (bytes, offset, row, image): (&[u8], u64, u64, u64),
        block: u64,
    ) {
        for z in 0..count[2] {
            for y in 0..count[1] {
                for x in 0..count[0] {
                    let from = (offset + z * image + y * row + x * block) as usize;
                    let key = (l, s, at[2] + z, at[1] + y, at[0] + x);
                    self.0
                        .insert(key, bytes[from..from + block as usize].to_vec());
                }
            }
        }
    }

    /// The blocks of `count` at block `at` of `(l, s)` written over `bytes` at `offset`, rows `row`
    /// and images `image` apart.
    pub fn give(
        &self,
        (l, s): (u64, u64),
        at: [u64; 3],
        count: [u64; 3],
        (bytes, offset, row, image): (&mut [u8], u64, u64, u64),
        block: u64,
    ) {
        for z in 0..count[2] {
            for y in 0..count[1] {
                for x in 0..count[0] {
                    let to = (offset + z * image + y * row + x * block) as usize;
                    let key = (l, s, at[2] + z, at[1] + y, at[0] + x);
                    bytes[to..to + block as usize].copy_from_slice(&self.0[&key]);
                }
            }
        }
    }

    /// `other`'s blocks of `count` at `from` of `(l, s)` copied to `to` of `(l2, s2)`.
    pub fn copy(
        &mut self,
        other: &Model,
        ((l, s), from): ((u64, u64), [u64; 3]),
        ((l2, s2), to): ((u64, u64), [u64; 3]),
        count: [u64; 3],
    ) {
        for z in 0..count[2] {
            for y in 0..count[1] {
                for x in 0..count[0] {
                    let block = other.0[&(l, s, from[2] + z, from[1] + y, from[0] + x)].clone();
                    self.0
                        .insert((l2, s2, to[2] + z, to[1] + y, to[0] + x), block);
                }
            }
        }
    }
}

/// The steps, in the lane's order, each naming the site its read-backs sit under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    Put,
    GetSub,
    Out,
    In,
    TextureToTexture,
}

impl Step {
    /// The site name the step's reads sit under.
    const fn site(self) -> &'static str {
        match self {
            Step::Put => "put",
            Step::GetSub => "getsub",
            Step::Out => "out",
            Step::In => "in",
            Step::TextureToTexture => "t2t",
        }
    }
}

/// One read-back the program asks for: where it sits in the transcript, what the model says it
/// should hold, and the layout the comparison reads it under.
pub struct Expected {
    /// The step the read belongs to, which is the branch it sits under.
    pub step: &'static str,
    /// The leaf's name inside the step: `<level>-<slice>` for a per-level read, `buffer` for a
    /// whole buffer read out tight.
    pub leaf: String,
    pub bytes: Vec<u8>,
    pub layout: ByteLayout,
}

/// A shape's program and what the model expects each of its reads to hold, in the order the program
/// asks for them.
pub struct Plan {
    pub work: Work,
    pub expected: Vec<Expected>,
}

/// The layout a read-back of `(l, s)` of `shape` is compared under: `across` blocks of `block`
/// bytes in a row, rows `row` apart, `down` rows an image, images `image` apart, every byte
/// compared including the padding past each row.
fn texture_layout(
    shape: &Shape,
    (l, s): (u64, u64),
    at: [u64; 3],
    [across, down, deep]: [u64; 3],
    (block, row, image): (u64, u64, u64),
) -> ByteLayout {
    ByteLayout {
        region: Region::Texture {
            level: l as u32,
            slice: s as u32,
            aspect: Aspect::Color,
            sample: 0,
            origin: [at[0] as u32 * 4, at[1] as u32 * 4, at[2] as u32],
            size: [across as u32 * 4, down as u32 * 4, shape.level(l)[2] as u32],
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: (across * block) as u32,
        row_pitch: row as u32,
        rows: down as u32,
        image_pitch: if image == 0 {
            (row * down) as u32
        } else {
            image as u32
        },
        images: deep as u32,
        gaps: Gaps::Compared,
    }
}

/// The plan for `format` in `shape`, seeded as the lane seeds it: the whole five-step program and
/// the model's bytes for each read.
///
/// The one function builds both, so the program's offsets and the model's offsets cannot drift
/// apart — the failure mode the formats batch hit twice was two sides of one shape written twice.
fn plan(format: u32, block: u64, shape: &Shape, seed: u64) -> Plan {
    let mut rng = Rng(seed);
    let mut work = Work::new();
    let mut expected = Vec::new();
    let three_d = shape.three_d();
    let first = work.add(Resource::Texture {
        spec: shape.spec(format, false),
    });
    let second = work.add(Resource::Texture {
        spec: shape.spec(format, true),
    });
    let third = work.add(Resource::Texture {
        spec: shape.spec(format, true),
    });
    let mut model = Model::default();

    // put: every level and slice whole, rows two blocks past, then one block alone at block (1, 0)
    // of level 0, slice 0.
    for (l, s) in shape.each() {
        let [w, h, d] = shape.level(l);
        let [bx, by, bz] = shape.blocks(l);
        let row = bx * block + 2 * block;
        let image = if three_d { row * by + block } else { 0 };
        let total = if three_d { image * bz } else { row * by };
        let bytes = rng.bytes(total);
        model.take((l, s), [0; 3], [bx, by, bz], (&bytes, 0, row, image), block);
        work.step(WorkStep::ReplaceRegion {
            texture: first,
            level: l,
            slice: s,
            region: Rect::new_3d([0, 0, 0], [w, h, d]),
            bytes,
            rows: by,
            bytes_per_row: row,
            bytes_per_image: image,
        });
    }
    if shape.blocks(0)[0] >= 2 {
        let h = shape.level(0)[1].min(4);
        let bytes = rng.bytes(block);
        let image = if three_d { block } else { 0 };
        model.take(
            (0, 0),
            [1, 0, 0],
            [1, 1, 1],
            (&bytes, 0, block, image),
            block,
        );
        work.step(WorkStep::ReplaceRegion {
            texture: first,
            level: 0,
            slice: 0,
            region: Rect::new_3d([4, 0, 0], [4, h, 1]),
            bytes,
            rows: 1,
            bytes_per_row: block,
            bytes_per_image: image,
        });
    }
    // The read-back: every level and slice whole, rows three blocks past, a 3D image two past.
    for (l, s) in shape.each() {
        let [w, h, d] = shape.level(l);
        let [bx, by, bz] = shape.blocks(l);
        let row = bx * block + 3 * block;
        let image = if three_d { row * by + 2 * block } else { 0 };
        let total = if three_d { image * bz } else { row * by };
        let mut want = vec![BEFORE; total as usize];
        model.give(
            (l, s),
            [0; 3],
            [bx, by, bz],
            (&mut want, 0, row, image),
            block,
        );
        expected.push(Expected {
            step: Step::Put.site(),
            leaf: format!("{l}-{s}"),
            bytes: want,
            layout: texture_layout(shape, (l, s), [0; 3], [bx, by, bz], (block, row, image)),
        });
        work.step(WorkStep::GetBytes {
            texture: first,
            level: l,
            slice: s,
            region: Rect::new_3d([0, 0, 0], [w, h, d]),
            bytes_per_row: row,
            bytes_per_image: image,
            length: total as usize,
        });
    }

    // getsub: from texel x 4 to each level's edges, where the level is wider than a block.
    for (l, s) in shape.each() {
        let [w, h, d] = shape.level(l);
        let [bx, by, bz] = shape.blocks(l);
        if w <= 4 {
            continue;
        }
        let across = bx - 1;
        let row = across * block + block;
        let image = if three_d { row * by + block } else { 0 };
        let total = if three_d { image * bz } else { row * by };
        let mut want = vec![BEFORE; total as usize];
        model.give(
            (l, s),
            [1, 0, 0],
            [across, by, bz],
            (&mut want, 0, row, image),
            block,
        );
        expected.push(Expected {
            step: Step::GetSub.site(),
            leaf: format!("{l}-{s}"),
            bytes: want,
            layout: texture_layout(
                shape,
                (l, s),
                [1, 0, 0],
                [across, by, bz],
                (block, row, image),
            ),
        });
        work.step(WorkStep::GetBytes {
            texture: first,
            level: l,
            slice: s,
            region: Rect::new_3d([4, 0, 0], [w - 4, h, d]),
            bytes_per_row: row,
            bytes_per_image: image,
            length: total as usize,
        });
    }

    // out: every level and slice into one buffer of BEFORE, each five blocks past the last, rows
    // one block past their blocks.
    let mut places = Vec::new();
    let mut at = 0;
    for (l, s) in shape.each() {
        let [bx, by, bz] = shape.blocks(l);
        let row = bx * block + block;
        let image = shape.image(l, row, block);
        at += 5 * block;
        places.push((l, s, at, row, image));
        at += if three_d { image * bz } else { row * by };
    }
    let out = work.buffer(vec![BEFORE; at as usize]);
    for &(l, s, offset, row, image) in &places {
        let [w, h, d] = shape.level(l);
        work.step(WorkStep::CopyTextureToBuffer {
            texture: first,
            slice: s,
            level: l,
            origin: [0, 0, 0],
            size: [w, h, d],
            buffer: out,
            offset,
            bytes_per_row: row,
            bytes_per_image: image,
            plane: Plane::Whole,
        });
    }
    work.step(WorkStep::Commit);
    let mut want = vec![BEFORE; at as usize];
    for &(l, s, offset, row, image) in &places {
        let image = if three_d { image } else { 0 };
        model.give(
            (l, s),
            [0; 3],
            shape.blocks(l),
            (&mut want, offset, row, image),
            block,
        );
    }
    expected.push(Expected {
        step: Step::Out.site(),
        leaf: "buffer".to_owned(),
        bytes: want,
        layout: ByteLayout::buffer(0, at, Element::U8),
    });
    work.step(WorkStep::ReadBuffer {
        buffer: out,
        offset: 0,
        length: at as usize,
    });

    // in: a buffer into the second texture, each level and slice three blocks past the last, rows
    // two blocks past; then that texture out tight.
    let mut places = Vec::new();
    let mut at = 0;
    for (l, s) in shape.each() {
        let [bx, by, bz] = shape.blocks(l);
        let row = bx * block + 2 * block;
        let image = shape.image(l, row, 2 * block);
        at += 3 * block;
        places.push((l, s, at, row, image));
        at += if three_d { image * bz } else { row * by };
    }
    let source = rng.bytes(at);
    let source_buffer = work.buffer(source.clone());
    for &(l, s, offset, row, image) in &places {
        let [w, h, d] = shape.level(l);
        work.step(WorkStep::CopyBufferToTexture {
            buffer: source_buffer,
            offset,
            bytes_per_row: row,
            bytes_per_image: image,
            size: [w, h, d],
            texture: second,
            slice: s,
            level: l,
            origin: [0, 0, 0],
            plane: Plane::Whole,
        });
    }
    work.step(WorkStep::Commit);
    let mut second_model = Model::default();
    for &(l, s, offset, row, image) in &places {
        let image = if three_d { image } else { 0 };
        second_model.take(
            (l, s),
            [0; 3],
            shape.blocks(l),
            (&source, offset, row, image),
            block,
        );
    }
    out_tight(
        &mut work,
        &mut expected,
        shape,
        block,
        second,
        &second_model,
        Step::In.site(),
    );

    // t2t: the first texture whole into a third, then a region of the second over part of it.
    let Moved {
        from,
        to,
        size,
        from_at,
        to_at,
    } = region(shape);
    for (l, s) in shape.each() {
        let [w, h, d] = shape.level(l);
        work.step(WorkStep::CopyTextureToTexture {
            from: first,
            from_slice: s,
            from_level: l,
            from_origin: [0, 0, 0],
            size: [w, h, d],
            to: third,
            to_slice: s,
            to_level: l,
            to_origin: [0, 0, 0],
        });
    }
    work.step(WorkStep::CopyTextureToTexture {
        from: second,
        from_slice: from_at.1,
        from_level: from_at.0,
        from_origin: from,
        size,
        to: third,
        to_slice: to_at.1,
        to_level: to_at.0,
        to_origin: to,
    });
    work.step(WorkStep::Commit);
    let mut third_model = Model::default();
    for (l, s) in shape.each() {
        third_model.copy(&model, ((l, s), [0; 3]), ((l, s), [0; 3]), shape.blocks(l));
    }
    let in_blocks = |v: [u64; 3]| [v[0] / 4, v[1] / 4, v[2]];
    let count = [size[0].div_ceil(4), size[1].div_ceil(4), size[2]];
    third_model.copy(
        &second_model,
        (from_at, in_blocks(from)),
        (to_at, in_blocks(to)),
        count,
    );
    out_tight(
        &mut work,
        &mut expected,
        shape,
        block,
        third,
        &third_model,
        Step::TextureToTexture.site(),
    );
    Plan { work, expected }
}

/// The texture-to-texture copy a shape's `t2t` step makes: where the blocks come from, where they
/// go, and how many.
struct Moved {
    /// The source origin in texels.
    from: [u64; 3],
    /// The destination origin in texels.
    to: [u64; 3],
    /// The size in texels.
    size: [u64; 3],
    /// The (level, slice) the source names.
    from_at: (u64, u64),
    /// The (level, slice) the destination names.
    to_at: (u64, u64),
}

/// The lane's region for `shape`, each chosen so the copy is block-aligned and reaches a level's
/// edge in at least one axis.
fn region(shape: &Shape) -> Moved {
    let (from, to, size, from_at, to_at) = match shape.name {
        // Blocks (1..3, 1) of level 0 to (0..2, 1): x interior, y to the edge.
        "2d-13x7" => ([4, 4, 0], [0, 4, 0], [8, 3, 1], (0, 0), (0, 0)),
        // Block (1, 1) of level 0 to (0, 0).
        "2d-8x8" => ([4, 4, 0], [0, 0, 0], [4, 4, 1], (0, 0), (0, 0)),
        // Level 1 of slice 2 whole to slice 0.
        "array-10x6x3" => ([0, 0, 0], [0, 0, 0], [5, 3, 1], (1, 2), (1, 0)),
        // Blocks (1..3, 0..3) of face 2 to face 5: to the edges.
        "cube-12" => ([4, 0, 0], [4, 0, 0], [8, 12, 1], (0, 2), (0, 5)),
        // Depth 1..3 of level 0 to depth 0..2.
        _ => ([0, 0, 1], [0, 0, 0], [9, 5, 2], (0, 0), (0, 0)),
    };
    Moved {
        from,
        to,
        size,
        from_at,
        to_at,
    }
}

/// Every level and slice of `texture` blitted out tight, one after another from byte 0, into a
/// buffer of [`BEFORE`], with the model's bytes recorded for it under `step`.
///
/// It is the lane's `out_tight`, which both compressed families use as their one way of reading a
/// private texture's blocks back.
pub fn out_tight(
    work: &mut Work,
    expected: &mut Vec<Expected>,
    shape: &Shape,
    block: u64,
    texture: usize,
    model: &Model,
    step: &'static str,
) {
    let three_d = shape.three_d();
    let mut places = Vec::new();
    let mut at = 0;
    for (l, s) in shape.each() {
        let [bx, by, bz] = shape.blocks(l);
        let row = bx * block;
        let image = shape.image(l, row, 0);
        places.push((l, s, at, row, image));
        at += row * by * bz;
    }
    let buffer = work.buffer(vec![BEFORE; at as usize]);
    for &(l, s, offset, row, image) in &places {
        let [w, h, d] = shape.level(l);
        work.step(WorkStep::CopyTextureToBuffer {
            texture,
            slice: s,
            level: l,
            origin: [0, 0, 0],
            size: [w, h, d],
            buffer,
            offset,
            bytes_per_row: row,
            bytes_per_image: image,
            plane: Plane::Whole,
        });
    }
    work.step(WorkStep::Commit);
    let mut want = vec![BEFORE; at as usize];
    for &(l, s, offset, row, image) in &places {
        let image = if three_d { image } else { 0 };
        model.give(
            (l, s),
            [0; 3],
            shape.blocks(l),
            (&mut want, offset, row, image),
            block,
        );
    }
    expected.push(Expected {
        step,
        leaf: "buffer".to_owned(),
        bytes: want,
        layout: ByteLayout::buffer(0, at, Element::U8),
    });
    work.step(WorkStep::ReadBuffer {
        buffer,
        offset: 0,
        length: at as usize,
    });
}

/// The case for one format.
pub struct CompressedMoves {
    at: usize,
}

/// Every format's case, in the lane's order.
pub fn all() -> Vec<CompressedMoves> {
    (0..FORMATS.len())
        .map(|at| CompressedMoves { at })
        .collect()
}

impl CompressedMoves {
    /// The format the case moves.
    fn format(&self) -> &'static Format {
        &FORMATS[self.at]
    }

    /// The plans of every shape, in the lane's order, each seeded as the lane seeds it: the block
    /// size and the shape's place, so the bytes are the bytes the lane measured.
    fn plans(&self) -> Vec<(&'static Shape, Plan)> {
        let format = self.format();
        SHAPES
            .iter()
            .enumerate()
            .map(|(k, shape)| {
                let seed = 0x5eed_0000 + format.block * 0x100 + k as u64;
                (shape, plan(format.format, format.block, shape, seed))
            })
            .collect()
    }
}

impl Case for CompressedMoves {
    fn name(&self) -> &'static str {
        FORMATS[self.at].name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Our `replaceRegion:` into a compressed texture goes through a staging buffer and a
        // buffer-to-image copy, which the runtime announces once per device. Metal's own
        // `replaceRegion:` writes the texels where it stands and announces nothing, so the only
        // site loosened is the set of notices; every block and every padding byte still compares
        // exactly. The allowance fails the case if the notice did not fire.
        vec![crate::replay::notice_allowance(
            Degradation::StagedTextureCopy.code(),
        )]
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // The whole program of every shape, as the interpreter will run it: every descriptor, every
        // region, every pitch and every byte written. A case whose program moves is `Unrecorded`
        // rather than compared against a recording of other bytes.
        let mut inputs = vec![Input::bytes("format", &self.format().format.to_le_bytes())];
        for (shape, plan) in self.plans() {
            inputs.push(Input::bytes(
                format!("program:{}", shape.name),
                format!("{:?}", plan.work).as_bytes(),
            ));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for (shape, plan) in self.plans() {
            let done = made(
                sites,
                &format!("{}/work", shape.name),
                side.work(&plan.work),
            )?;
            let mut nodes: Vec<Node> = Vec::new();
            for (at, expected) in plan.expected.iter().enumerate() {
                let bytes = done.read(at).map_err(|why| Abort {
                    at: format!("{}/{}/{}", shape.name, expected.step, expected.leaf),
                    failure: Failure::named("UnansweredRead", why),
                })?;
                let node = Node::leaf(
                    expected.leaf.clone(),
                    Observed::bytes(bytes.to_vec(), expected.layout.clone()),
                );
                match nodes.last_mut() {
                    Some(last) if last.name == expected.step => last.children.push(node),
                    _ => nodes.push(Node::branch(expected.step.to_owned(), vec![node])),
                }
            }
            sites.push(Node::branch(shape.name.to_owned(), nodes));
            sites.value(
                format!("{}/statuses", shape.name),
                Value::Lines(done.statuses.clone()),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The cases name distinct recordings, since a case's name is its recording's file name.
    #[test]
    fn the_formats_name_distinct_cases() {
        let named: BTreeSet<&str> = FORMATS.iter().map(|format| format.name).collect();
        assert_eq!(named.len(), FORMATS.len());
    }

    /// Every plan asks for exactly as many reads as it holds expectations, and every expectation's
    /// layout covers its bytes: the two halves of one plan agree by construction.
    #[test]
    fn every_plans_reads_match_its_expectations() {
        for case in all() {
            for (shape, plan) in case.plans() {
                assert_eq!(
                    plan.work.reads(),
                    plan.expected.len(),
                    "{} {}",
                    case.name(),
                    shape.name
                );
                for expected in &plan.expected {
                    // A padded read-back's last row keeps its padding, which the layout's extent
                    // does not count: the extent reaches the last element, and the value holds the
                    // bytes past it too, which `Gaps::Compared` is what compares.
                    assert!(
                        expected.layout.extent() <= expected.bytes.len(),
                        "{} {}/{}: a layout reaching {} bytes over a value of {}",
                        case.name(),
                        expected.step,
                        expected.leaf,
                        expected.layout.extent(),
                        expected.bytes.len(),
                    );
                }
            }
        }
    }

    /// Every one of Metal's recorded read-backs is the lane's byte model's, byte for byte, padding
    /// included.
    ///
    /// This is the cross-check the port exists for. The recordings hold what Metal answered on this
    /// M3 Max; the model was tied to Metal only by a six-byte digest per step, measured by our own
    /// probe on an Apple M4. A difference means one of two things, and either is a finding rather
    /// than something to edit away: Apple's copies moved between the chips or the releases, or the
    /// model was never Metal's.
    ///
    /// A case with no recording yet is named, so a missing recording is record mode's work and not
    /// a silent pass.
    #[test]
    fn every_recorded_move_is_the_lanes_byte_model() {
        let mut unrecorded = Vec::new();
        let mut differ = Vec::new();
        let mut checked = 0usize;
        for case in all() {
            let recording = match crate::recording::read(case.name()) {
                Ok(recording) => recording,
                Err(_) => {
                    unrecorded.push(case.name());
                    continue;
                }
            };
            let plans = case.plans();
            for answer in &recording.answers {
                for outcome in &answer.outcomes {
                    for (shape, plan) in &plans {
                        for expected in &plan.expected {
                            let path =
                                format!("{}/{}/{}", shape.name, expected.step, expected.leaf);
                            let node = outcome
                                .transcript
                                .root
                                .child(shape.name)
                                .and_then(|node| node.child(expected.step))
                                .and_then(|node| node.child(&expected.leaf));
                            let Some(node) = node else {
                                differ.push(format!("{}: no site {path}", case.name()));
                                continue;
                            };
                            let crate::transcript::Value::Bytes(bytes) = &node.observed.value
                            else {
                                differ.push(format!("{}/{path}: no bytes", case.name()));
                                continue;
                            };
                            checked += bytes.len();
                            if bytes != &expected.bytes && differ.len() < 24 {
                                let at =
                                    bytes.iter().zip(&expected.bytes).position(|(a, b)| a != b);
                                differ.push(format!(
                                    "{}/{path}: {} bytes against the model's {}, first difference \
                                     at {at:?}",
                                    case.name(),
                                    bytes.len(),
                                    expected.bytes.len(),
                                ));
                            }
                        }
                    }
                }
            }
        }
        assert!(
            differ.is_empty(),
            "{checked} bytes checked; Metal's recorded copies differ from the lane's byte \
             model:\n{}",
            differ.join("\n")
        );
        assert!(
            unrecorded.is_empty(),
            "no recording yet for {unrecorded:?}: run `conformance record` on a Mac"
        );
    }
}
