//! The suite's cases.
//!
//! Two seed cases prove the loop end to end, chosen to be as unlike each other as the first
//! increment allows: one compares the bytes a pass wrote, the other a typed answer a getter gives.
//! Neither can fault a GPU.

pub mod indirect_private_types;
pub mod pass_bytes;

use crate::case::Case;

/// Every case of the suite, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![
        Box::new(pass_bytes::srgb_decode_all_bytes()),
        Box::new(indirect_private_types::private_type_getters()),
    ]
}

/// The case named `name`, where the suite holds one.
pub fn named(name: &str) -> Option<Box<dyn Case>> {
    all().into_iter().find(|case| case.name() == name)
}
