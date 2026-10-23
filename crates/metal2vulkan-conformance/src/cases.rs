//! The suite's cases.
//!
//! Two seed families prove the loop end to end, chosen to be as unlike each other as the first
//! increment allows: one compares the bytes a pass wrote, the other a typed answer a getter gives.
//! Neither can fault a GPU.

pub mod extended_range_planes;
pub mod indirect_private_types;
pub mod math_routines;
pub mod pass_bytes;
pub mod stage_in_regions;
pub mod texture_buffer_fills;
pub mod two_plane_heap_blits;

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
    cases.push(Box::new(texture_buffer_fills::streams()));
    cases.push(Box::new(texture_buffer_fills::groups()));
    cases.push(Box::new(texture_buffer_fills::long_fills()));
    cases.push(Box::new(two_plane_heap_blits::heap_blits()));
    cases
}

/// The case named `name`, where the suite holds one.
pub fn named(name: &str) -> Option<Box<dyn Case>> {
    all().into_iter().find(|case| case.name() == name)
}
