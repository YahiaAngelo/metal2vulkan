//! A run of render pipeline creations over one vertex-and-fragment pair, differing only in blend
//! state, misses the driver's cache **every single time** on a device that reports what it found
//! (`ShaderCacheStatistics::driver_pipeline_hits`, `driver_pipeline_misses`).
//!
//! # What this records, and why it is worth a test
//!
//! Every making of the runtime is given a `VkPipelineCache` of its own, made from a record named
//! by the whole graphics key and destroyed once the call returned (`shader_cache/driver.rs`).
//! A sweep of states over one function pair therefore gets a record each: 32 states over two
//! modules are 32 records and 32 caches, no two of which the other can read. The sibling test
//! `render_sibling_work` holds the translator's side of such a sweep -- the modules, the layout
//! and the specialized entries are shared, so nothing is translated twice. This holds the
//! **driver's** side, which is the opposite: nothing is shared, and the count says so.
//!
//! The assertion is that **no creation of the run hits**, which is a fact about the arrangement
//! and not about speed, so it holds under any load. A hit would mean the device found the
//! sibling's pipeline in a cache seeded from a record the sibling's own key names, which within
//! one process and one empty cache directory cannot happen: each making's record is written after
//! its own making and read by no other key.
//!
//! The run asserts the hits and misses **together against the creations**, so a device that
//! reports nothing (no `VK_EXT_pipeline_creation_feedback`, where neither counter moves) reads as
//! a skip rather than as a pass: a zero-hit assertion alone would pass vacuously there.
//!
//! # What it does not say
//!
//! It does not say the driver converted each sibling's stages again. Whether a miss costs a
//! conversion is the driver's own business and `VK_EXT_pipeline_creation_feedback` reports
//! nothing finer than a hit per stage; on MoltenVK 1.4.2 a miss does cost one, measured with the
//! driver's private per-activity tracking in
//! `metal2vulkan-conformance/probes/pipeline_cache_across_blend_states.c`, and that is a probe and
//! not a test for exactly that reason. What the counter proves here is the arrangement that
//! leaves the driver no cache to find anything in.
//!
//! The device takes a shader cache directory of its own, empty, so no run reads a record an
//! earlier run stored.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. The functions are the owned fixture
//! `validation/fixtures/public/engine/ubo_bind_offsets.ll`'s (`llvm-as`).

use super::blit_holds::{device, Scratch};
use super::descriptor_key_once::descriptor;
use crate::PipelineOptions;
use metal2vulkan::metal::{BlendFactor, BlendOperation};

/// The siblings the run makes after the warm-up's one.
const RUN: u64 = 16;

/// Blend factors no two of which make one pipeline: each sibling is its own descriptor, and so
/// its own record.
const FACTORS: [BlendFactor; 8] = [
    BlendFactor::Zero,
    BlendFactor::One,
    BlendFactor::SourceColor,
    BlendFactor::OneMinusSourceColor,
    BlendFactor::SourceAlpha,
    BlendFactor::OneMinusSourceAlpha,
    BlendFactor::DestinationColor,
    BlendFactor::DestinationAlpha,
];

/// Every sibling render creation of one function pair misses the driver's cache: a making's cache
/// holds its own key's record alone, so a sibling finds nothing of the first's in it.
#[test]
fn every_sibling_render_creation_misses_the_driver_s_cache() {
    let scratch = Scratch::new("driver-pipeline-compile");
    let device = device(&scratch.0);
    let base = descriptor(&device);
    let options = PipelineOptions::default();
    let sibling = |i: u64| {
        let mut d = base.clone();
        let color = d.color_attachments[0]
            .as_mut()
            .expect("the fixture's target");
        color.blending_enabled = true;
        color.source_rgb_blend_factor = FACTORS[(i % 8) as usize];
        color.destination_rgb_blend_factor = FACTORS[(i / 8 % 8) as usize];
        color.rgb_blend_operation = BlendOperation::Add;
        d
    };
    // The warm-up's creation compiles the stages and makes their modules, so the run's siblings
    // ask the driver for nothing but a pipeline over stages it has already seen.
    let _first = device
        .new_render_pipeline_state(&sibling(0), &options)
        .unwrap();
    let before = device.shader_cache_statistics();
    let mut kept = Vec::with_capacity(RUN as usize);
    for i in 0..RUN {
        kept.push(
            device
                .new_render_pipeline_state(&sibling(1 + i), &options)
                .unwrap(),
        );
    }
    let after = device.shader_cache_statistics();
    let hits = after.driver_pipeline_hits - before.driver_pipeline_hits;
    let misses = after.driver_pipeline_misses - before.driver_pipeline_misses;
    if hits + misses == 0 {
        // The device reports nothing it found, so there is nothing to read here. Saying so is the
        // point: without it a zero-hit assertion would pass on such a device while measuring
        // nothing at all.
        return;
    }
    assert_eq!(
        (hits, misses),
        (0, RUN),
        "a device that reports what it found must report every one of {RUN} sibling creations as a \
         miss, each making's cache holding its own key's record alone: {hits} hits and {misses} \
         misses"
    );
}
