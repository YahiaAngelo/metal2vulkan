//! `getBytes` of a `Shared` texture into the caller's bytes laid out packed and pitched, of a 2D
//! texture's whole level and part of it and of a 3D texture's images: each row lands where its
//! layout puts it, the bytes between rows and images and past the last row are left as they were,
//! on whichever path the device copies through (on the host, `texture/host_copy.rs`, which reads
//! a packed layout in place; on its staging queue, `texture/staging_copy.rs`; or through a
//! submission), which the test prints.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch};
use crate::{Region, ResourceOptions, Texture, TextureDescriptor};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};

/// The byte a caller's bytes hold where no row lands.
const UNTOUCHED: u8 = 0xab;

/// A `Shared` `RGBA8Unorm` texture of `width` x `height` x `depth` (3D where `depth` is above 1),
/// each byte of its texels its own (`pattern`), written whole.
fn texture(device: &crate::Device, [width, height, depth]: [u64; 3]) -> Texture {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, width, height, false);
    if depth > 1 {
        descriptor.texture_type = TextureType::Type3D;
        descriptor.depth = depth;
    }
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes = pattern([width, height, depth]);
    let region = Region::new_3d(0, 0, 0, width, height, depth);
    let row = 4 * width;
    (texture.replace_region(region, 0, 0, &bytes, row, row * height)).unwrap();
    texture
}

/// The texels of [`texture`], packed: byte `i` is `i * 7 + 1` modulo 251, so no two bytes of a
/// row or of neighbouring rows are alike.
fn pattern([width, height, depth]: [u64; 3]) -> Vec<u8> {
    (0..4 * width * height * depth)
        .map(|i| ((i * 7 + 1) % 251) as u8)
        .collect()
}

/// What `getBytes` of `region` of a texture of `size` must leave in a caller's `len` bytes laid
/// out `bytes_per_row` and `bytes_per_image` apart, starting [`UNTOUCHED`].
fn expected(
    size: [u64; 3],
    region: Region,
    (bytes_per_row, bytes_per_image): (u64, u64),
    len: usize,
) -> Vec<u8> {
    let texels = pattern(size);
    let mut bytes = vec![UNTOUCHED; len];
    let row = 4 * region.size.width as usize;
    for z in 0..region.size.depth {
        for y in 0..region.size.height {
            let source = (((region.origin.z + z) * size[1] + region.origin.y + y) * size[0]
                + region.origin.x) as usize
                * 4;
            let at = (z * bytes_per_image + y * bytes_per_row) as usize;
            bytes[at..at + row].copy_from_slice(&texels[source..source + row]);
        }
    }
    bytes
}

/// Each layout of each region reads its rows where the layout puts them, and nothing else.
#[test]
fn get_bytes_reads_each_layout_its_rows_alone() {
    let scratch = Scratch::new("get-bytes-layouts");
    let device = device(&scratch.0);
    let (mut host_reads, mut staged_reads) = (0, 0);
    // (texture size, region, bytes per row, bytes per image, the caller's bytes)
    let cases: [([u64; 3], Region, u64, u64, usize); 7] = [
        // Packed, whole: exactly its bytes, and with bytes past them.
        ([7, 5, 1], Region::new_2d(0, 0, 7, 5), 28, 0, 140),
        ([7, 5, 1], Region::new_2d(0, 0, 7, 5), 28, 0, 160),
        // Pitched rows.
        ([7, 5, 1], Region::new_2d(0, 0, 7, 5), 36, 0, 36 * 4 + 28),
        // Part of the level, packed and pitched.
        ([7, 5, 1], Region::new_2d(2, 1, 3, 2), 12, 0, 24),
        ([7, 5, 1], Region::new_2d(2, 1, 3, 2), 20, 0, 32),
        // A 3D texture's images, packed and apart.
        ([4, 3, 2], Region::new_3d(0, 0, 0, 4, 3, 2), 16, 48, 96),
        ([4, 3, 2], Region::new_3d(0, 0, 0, 4, 3, 2), 16, 64, 112),
    ];
    for (size, region, bytes_per_row, bytes_per_image, len) in cases {
        let texture = texture(&device, size);
        let mut bytes = vec![UNTOUCHED; len];
        let before = device.work_counts();
        texture
            .get_bytes(&mut bytes, bytes_per_row, bytes_per_image, region, 0, 0)
            .unwrap();
        let after = device.work_counts();
        host_reads += after.host_texel_copies - before.host_texel_copies;
        staged_reads += after.staging_submissions - before.staging_submissions;
        let wanted = expected(size, region, (bytes_per_row, bytes_per_image), len);
        assert_eq!(
            bytes, wanted,
            "{size:?} {region:?}, {bytes_per_row} and {bytes_per_image} apart"
        );
    }
    let cases = cases.len();
    eprintln!(
        "{}: {host_reads} of {cases} reads on the host, {staged_reads} on the staging queue",
        device.name()
    );
}
