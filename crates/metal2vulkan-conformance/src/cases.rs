//! The suite's case set.
//!
//! Two seed families proved the loop end to end, chosen to be as unlike each other as the first
//! increment allowed: one compares the bytes a compute pass wrote, the other a typed answer a
//! getter gives.
//!
//! Most of the rest are ports of the runtime lanes whose expectation is already a transcript of a
//! Metal probe's printed lines, which is what made them the first batch. Where Metal's whole path
//! is public, record mode **runs** the probe rather than copying it: the mesh draws, the mesh draws
//! behind a barrier, and the mesh draws of an indirect command buffer's own-state commands. Where
//! it is not, the case carries our own measurement and forbids re-recording: the texture fills,
//! whose `fillTexture:` is Apple SPI, and an indirect command's read-back through private getters.
//! The reason is always the *call*, never the pixel format a case names.
//!
//! [`rate_maps`] is the family where Metal's own API validation layer and plain Metal disagree,
//! and it is what exercises the recording's second oracle ([`crate::validated`]).
//!
//! Two families are ports of lanes that held **no Metal value at all** — a rule, or a list in a
//! module's prose — and whose values this round measured on the oracle for the first time:
//! [`texture_accesses`], where Metal's render-pipeline reflection states each texture slot's
//! access, and [`textures`], where every uncompressed colour format's texture answers what a fresh
//! level 0 reads, what a host round trip gives back, and whether the bytes between the rows stayed
//! as they were.
//!
//! No case here can fault a GPU, hang it, trip its watchdog or end Metal's process.

pub mod command_queues;
pub mod depth_texel_reads;
pub mod extended_range_planes;
pub mod extended_range_two_planes;
pub mod heap_metal_sizes;
pub mod indirect_private_types;
pub mod math_routines;
pub mod mesh_barriers;
pub mod mesh_commands;
pub mod mesh_draws;
pub mod pass_bytes;
pub mod pixel_downscales;
pub mod pixel_samplers;
pub mod rate_maps;
pub mod rgb10a2_srgb;
pub mod sample_fills;
pub mod stage_in_regions;
pub mod surface_fills;
pub mod texture_accesses;
pub mod texture_buffer_fills;
pub mod texture_fills;
pub mod textures;
pub mod tile_threadgroups;
pub mod two_plane_copies;
pub mod two_plane_heap_blits;
pub mod two_plane_samples;

use crate::case::Case;

/// Every case of the suite, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    let mut cases: Vec<Box<dyn Case>> = Vec::new();
    for case in pass_bytes::all() {
        cases.push(Box::new(case));
    }
    cases.push(Box::new(indirect_private_types::private_type_getters()));
    for case in math_routines::all() {
        cases.push(Box::new(case));
    }
    cases.push(Box::new(stage_in_regions::stage_in_region_clamps()));
    for case in extended_range_planes::all() {
        cases.push(Box::new(case));
    }
    for case in depth_texel_reads::all() {
        cases.push(Box::new(case));
    }
    for case in extended_range_two_planes::all() {
        cases.push(Box::new(case));
    }
    for case in two_plane_samples::all() {
        cases.push(Box::new(case));
    }
    for case in rgb10a2_srgb::all() {
        cases.push(Box::new(case));
    }
    for case in pixel_samplers::all() {
        cases.push(Box::new(case));
    }
    for case in pixel_downscales::all() {
        cases.push(Box::new(case));
    }
    cases.push(Box::new(texture_buffer_fills::streams()));
    cases.push(Box::new(texture_buffer_fills::groups()));
    cases.push(Box::new(texture_buffer_fills::long_fills()));
    cases.push(Box::new(two_plane_heap_blits::heap_blits()));
    cases.push(Box::new(two_plane_copies::row_blits()));
    cases.push(Box::new(two_plane_copies::announced_row_blits()));
    cases.push(Box::new(two_plane_copies::pad_fills()));
    for case in texture_fills::all() {
        cases.push(Box::new(case));
    }
    for case in sample_fills::all() {
        cases.push(Box::new(case));
    }
    for case in mesh_draws::all() {
        cases.push(Box::new(case));
    }
    for case in mesh_barriers::all() {
        cases.push(Box::new(case));
    }
    cases.push(Box::new(mesh_commands::own_state_commands()));
    cases.push(Box::new(mesh_commands::unordered_draws()));
    cases.push(Box::new(surface_fills::over_planes_and_buffers()));
    for case in rate_maps::all() {
        cases.push(Box::new(case));
    }
    for case in tile_threadgroups::all() {
        cases.push(Box::new(case));
    }
    for case in heap_metal_sizes::all() {
        cases.push(Box::new(case));
    }
    cases.push(Box::new(command_queues::queue_slots()));
    for case in texture_accesses::all() {
        cases.push(Box::new(case));
    }
    for case in textures::all() {
        cases.push(Box::new(case));
    }
    cases
}

/// The case named `name`, where the suite holds one.
pub fn named(name: &str) -> Option<Box<dyn Case>> {
    all().into_iter().find(|case| case.name() == name)
}
