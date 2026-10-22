//! `getBytes` and `replaceRegion` of a `Shared` texture with the caller's bytes laid out packed and
//! pitched, of a 2D texture's whole level and part of it and of a 3D texture's images: each row
//! is read or written where its layout puts it; `getBytes` leaves the bytes between rows and
//! images and past the last row as they were, and `replaceRegion` writes no texel outside the
//! region. On whichever path the device copies through (on the host, `texture/host_copy.rs`, which
//! reads and writes a packed layout in place; on its staging queue, `texture/staging_copy.rs`; or
//! through a submission), which the tests print.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch};
use crate::{Device, Region, ResourceOptions, Texture, TextureDescriptor};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};

/// The byte a caller's bytes hold where no row lands.
const UNTOUCHED: u8 = 0xab;

/// Each case: the texture's size, the region, the caller's bytes per row and per image, and the
/// length of the caller's bytes.
const CASES: [([u64; 3], Region, u64, u64, usize); 7] = [
    // Packed, whole: exactly its bytes, and with bytes past them.
    ([7, 5, 1], region([0, 0, 0], [7, 5, 1]), 28, 0, 140),
    ([7, 5, 1], region([0, 0, 0], [7, 5, 1]), 28, 0, 160),
    // Pitched rows.
    ([7, 5, 1], region([0, 0, 0], [7, 5, 1]), 36, 0, 36 * 4 + 28),
    // Part of the level, packed and pitched.
    ([7, 5, 1], region([2, 1, 0], [3, 2, 1]), 12, 0, 24),
    ([7, 5, 1], region([2, 1, 0], [3, 2, 1]), 20, 0, 32),
    // A 3D texture's images, packed and apart.
    ([4, 3, 2], region([0, 0, 0], [4, 3, 2]), 16, 48, 96),
    ([4, 3, 2], region([0, 0, 0], [4, 3, 2]), 16, 64, 112),
];

/// The region of `size` at `origin`.
const fn region(origin: [u64; 3], size: [u64; 3]) -> Region {
    Region {
        origin: crate::Origin {
            x: origin[0],
            y: origin[1],
            z: origin[2],
        },
        size: crate::Size {
            width: size[0],
            height: size[1],
            depth: size[2],
        },
    }
}

/// A `Shared` `RGBA8Unorm` texture of `width` x `height` x `depth` (3D where `depth` is above 1),
/// each byte of its texels its own (`pattern(_, 0)`), written whole.
fn texture(device: &Device, [width, height, depth]: [u64; 3]) -> Texture {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, width, height, false);
    if depth > 1 {
        descriptor.texture_type = TextureType::Type3D;
        descriptor.depth = depth;
    }
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes = pattern([width, height, depth], 0);
    let region = Region::new_3d(0, 0, 0, width, height, depth);
    let row = 4 * width;
    (texture.replace_region(region, 0, 0, &bytes, row, row * height)).unwrap();
    texture
}

/// Texels of a texture of `size`, packed: byte `i` is `i * 7 + 1 + seed` modulo 251, so no two
/// bytes of a row or of neighbouring rows are alike, and two seeds' bytes differ everywhere.
fn pattern([width, height, depth]: [u64; 3], seed: u64) -> Vec<u8> {
    (0..4 * width * height * depth)
        .map(|i| ((i * 7 + 1 + seed) % 251) as u8)
        .collect()
}

/// The packed offset in a texture of `size` of the first texel of row `y` of image `z` of
/// `region`, and the offset in the caller's bytes laid out `bytes_per_row` and `bytes_per_image`
/// apart; each row is `4 * region.size.width` bytes.
fn rows(
    size: [u64; 3],
    region: Region,
    (bytes_per_row, bytes_per_image): (u64, u64),
) -> impl Iterator<Item = (usize, usize)> {
    (0..region.size.depth).flat_map(move |z| {
        (0..region.size.height).map(move |y| {
            let texel =
                ((region.origin.z + z) * size[1] + region.origin.y + y) * size[0] + region.origin.x;
            (
                4 * texel as usize,
                (z * bytes_per_image + y * bytes_per_row) as usize,
            )
        })
    })
}

/// The reads and writes the device made on the host and on its staging queue while `run` ran.
fn paths(device: &Device, run: impl FnOnce()) -> [u64; 2] {
    let before = device.work_counts();
    run();
    let after = device.work_counts();
    [
        after.host_texel_copies - before.host_texel_copies,
        after.staging_submissions - before.staging_submissions,
    ]
}

/// Each layout of each region reads its rows where the layout puts them, and nothing else.
#[test]
fn get_bytes_reads_each_layout_its_rows_alone() {
    let scratch = Scratch::new("get-bytes-layouts");
    let device = device(&scratch.0);
    let mut counted = [0; 2];
    for (size, region, bytes_per_row, bytes_per_image, len) in CASES {
        let texture = texture(&device, size);
        let mut bytes = vec![UNTOUCHED; len];
        let [host, staged] = paths(&device, || {
            texture
                .get_bytes(&mut bytes, bytes_per_row, bytes_per_image, region, 0, 0)
                .unwrap()
        });
        counted = [counted[0] + host, counted[1] + staged];
        let texels = pattern(size, 0);
        let mut wanted = vec![UNTOUCHED; len];
        let row = 4 * region.size.width as usize;
        for (texel, at) in rows(size, region, (bytes_per_row, bytes_per_image)) {
            wanted[at..at + row].copy_from_slice(&texels[texel..texel + row]);
        }
        assert_eq!(
            bytes, wanted,
            "{size:?} {region:?}, {bytes_per_row} and {bytes_per_image} apart"
        );
    }
    let [host, staged] = counted;
    eprintln!(
        "{}: {host} of {} reads on the host, {staged} on the staging queue",
        device.name(),
        CASES.len()
    );
}

/// Each layout of each region writes its rows where the layout puts them into the region, and no
/// other texel: the texture read back whole holds the caller's rows in the region and its own
/// texels elsewhere.
#[test]
fn replace_region_writes_each_layout_its_rows_alone() {
    let scratch = Scratch::new("replace-region-layouts");
    let device = device(&scratch.0);
    let mut counted = [0; 2];
    for (size, region, bytes_per_row, bytes_per_image, len) in CASES {
        let texture = texture(&device, size);
        let written = pattern(size, 100);
        let layout = (bytes_per_row, bytes_per_image);
        let mut bytes = vec![UNTOUCHED; len];
        let row = 4 * region.size.width as usize;
        let mut wanted = pattern(size, 0);
        for (texel, at) in rows(size, region, layout) {
            bytes[at..at + row].copy_from_slice(&written[texel..texel + row]);
            wanted[texel..texel + row].copy_from_slice(&written[texel..texel + row]);
        }
        let [host, staged] = paths(&device, || {
            texture
                .replace_region(region, 0, 0, &bytes, bytes_per_row, bytes_per_image)
                .unwrap()
        });
        counted = [counted[0] + host, counted[1] + staged];
        let mut back = vec![0; wanted.len()];
        let whole = Region::new_3d(0, 0, 0, size[0], size[1], size[2]);
        let row = 4 * size[0];
        (texture.get_bytes(&mut back, row, row * size[1], whole, 0, 0)).unwrap();
        assert_eq!(
            back, wanted,
            "{size:?} {region:?}, {bytes_per_row} and {bytes_per_image} apart"
        );
    }
    let [host, staged] = counted;
    eprintln!(
        "{}: {host} of {} writes on the host, {staged} on the staging queue",
        device.name(),
        CASES.len()
    );
}
