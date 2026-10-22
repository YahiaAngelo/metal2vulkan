//! A command's argument samplers, listed again as its command buffer commits and as the batch
//! holding the command is submitted.
//!
//! Metal's function reaches a sampler state by its resource ID as the command runs, as it does a
//! texture (`resource_ids.rs`). A command reaches the states its sampler table lists
//! (`BindState::argument_samplers`): the device's live states as it was encoded. So a state made
//! after that, or one that took a listed state's ID, would be missed. Each command that writes a
//! sampler table keeps it here, with its encoder's place, and the command buffer lists the states
//! live as it commits, and again as the batch holding the encoder is submitted where a state was
//! made or went since (the count of the sampler IDs' slots' changes moved,
//! `Device::sampler_slots_changed`):
//!
//! - **Where they fit the plan** (as many as the planned entries hold beside the null entry, twice
//!   the class's, each of a shape the plan was made for or one its checks admit,
//!   `BindingPlan::relisted_sampler_tables`), their table is written over the command's region of
//!   the arena, and their elements into the heap set the command binds
//!   (`binding::argument_sampler_writes`, `heap_sets::bind`): the set's one version, whose new
//!   elements no pending command buffer reads, or the command buffer's own, still written until it
//!   commits (`heap_sets.rs`). Before the batch's `vkQueueSubmit`, so the host's writes are visible
//!   to it (the host write ordering guarantee).
//! - **Where they do not** (a state past the command's set-1 sampler arrays,
//!   `heap_sets::fit_samplers`; more states than its tables hold; or a shape its checks refuse,
//!   such as a cubic state beside a fragment stage), the command cannot reach every state as
//!   Metal's function would: it runs with the states that fit the arrays where its tables hold
//!   them, else with those it listed before, and a function reading a state left out reads the
//!   neutral sampler. Announced as `unlisted-argument-sampler` and counted once per command buffer
//!   (`WorkCounts::unlisted_argument_sampler_buffers`), whether or not a function reaches the
//!   state left out.
//!
//! **Why the submission.** A batch is submitted only once every event it waits for is reached
//! (`scheduler.rs`), and an event wait starts a batch of its own (`batches.rs`). So a state made
//! after the commit and before a wait it is ordered by (the program writes its ID into an argument
//! buffer, then signals the event the command buffer waits for, or commits the command buffer that
//! signals it) is made before that batch is submitted, and its command reads it, as Metal's does
//! (the M3 Max under macOS 27.0.1). A state made after the batch's submission can reach the
//! batch's function only through memory the program wrote after the submission with nothing
//! ordering that write before the function's read, a data race on Metal too, whose answer hangs on
//! when the GPU read the memory. On a device without `descriptorBindingUpdateUnusedWhilePending`,
//! whose command buffers bind their own versions of the heap set, written until they commit
//! (`Degradation::HeapSetCopied`), the commit's listing stands: a state made after the commit is
//! the neutral sampler there, as a texture made after it is.
//!
//! A command a host step records after the commit is listed again by no commit: it lists the states
//! live as it is recorded, and where more live than its set-1 sampler arrays hold, so that it
//! leaves one out (`heap_sets::fit_samplers`), it runs with the others, announced and counted as
//! above (`State::unlisted_argument_sampler`).

use super::binding;
use super::error::{internal, Error};
use super::heap_sets::{self, HeapUse};
use super::pipeline::VariantObjects;
use super::sampler::{OwnedSampler, SamplerState};
use super::upload::Upload;
use super::Device;
use metal2vulkan::bind::{ArgumentSampler, BindingPlan, HeapPlacement};
use metal2vulkan::pipeline::DescriptorSetLayout;
use std::sync::Arc;

/// One command's sampler tables and what lists them again as its command buffer commits.
pub(super) struct SamplerListing {
    plan: Arc<BindingPlan>,
    objects: Arc<VariantObjects>,
    /// Set 1 of the command's layout, whose sampler arrays' capacity bounds the elements.
    heap: DescriptorSetLayout,
    placement: HeapPlacement,
    /// The argument samplers the command listed.
    listed: Vec<ArgumentSampler>,
    /// The count of the sampler IDs' slots' changes `listed` was taken at, once the commit listed
    /// them (`Device::sampler_slots_changed`).
    listed_at: Option<u64>,
    /// The place of the command's encoder among its command buffer's ended encoders.
    encoder: usize,
    /// Each sampler table's `(set, binding)` and region.
    tables: Vec<((u32, u32), Upload)>,
    /// The commands, named, for the notices.
    commands: String,
}

impl SamplerListing {
    /// The listing of a command of `plan` and `objects` whose layout's set 1 is `heap`, which
    /// listed `listed` under `placement` in `tables`: `None` where it has no sampler table.
    pub(super) fn of(
        (plan, objects): (&Arc<BindingPlan>, &Arc<VariantObjects>),
        heap: Option<&DescriptorSetLayout>,
        (listed, placement): (&[ArgumentSampler], HeapPlacement),
        tables: Vec<((u32, u32), Upload)>,
        commands: impl FnOnce() -> String,
    ) -> Option<SamplerListing> {
        if tables.is_empty() {
            return None;
        }
        Some(SamplerListing {
            plan: Arc::clone(plan),
            objects: Arc::clone(objects),
            heap: heap?.clone(),
            placement,
            listed: listed.to_vec(),
            listed_at: None,
            encoder: 0,
            tables,
            commands: commands(),
        })
    }

    /// The command's encoder takes place `encoder` among its command buffer's ended encoders.
    pub(super) fn at_encoder(&mut self, encoder: usize) {
        self.encoder = encoder;
    }

    /// The place of the command's encoder among its command buffer's ended encoders.
    pub(super) fn encoder(&self) -> usize {
        self.encoder
    }

    /// Whether the states listed may no longer be the live ones: the commit has not listed them,
    /// or a state was made or went since (`changed`, the count of the sampler IDs' slots' changes
    /// now).
    pub(super) fn stale(&self, changed: u64) -> bool {
        self.listed_at != Some(changed)
    }

    /// List `live` again, the device's live states at the count `changed` of the sampler IDs'
    /// slots' changes, as the command buffer whose heap use is `heaps` commits or as the batch
    /// holding the command is submitted (the module docs): the Vulkan samplers the new elements
    /// name, which the command buffer holds until it completes, and, where the command's plan
    /// cannot list every one, the object and the reason of the `unlisted-argument-sampler` notice,
    /// for the caller to announce. The command then lists those that fit its sampler arrays, where
    /// its tables hold them, else keeps those it listed.
    pub(super) fn relist(
        &mut self,
        device: &Device,
        (changed, live): (u64, &[SamplerState]),
        heaps: &mut HeapUse,
    ) -> Result<Relisted, Error> {
        self.listed_at = Some(changed);
        let mut live = live.to_vec();
        let mut past = None;
        // A state past the command's sampler arrays, which no element of them holds, is left out.
        heap_sets::fit_samplers(Some(&self.heap), &mut live, &mut past);
        let unlisted = past.map(|past| heap_sets::unlisted_notice(past, &self.commands));
        let samplers = heap_sets::listed(&live);
        if samplers == self.listed {
            return Ok(Relisted {
                samplers: Vec::new(),
                unlisted,
            });
        }
        let Some(bytes) = self
            .plan
            .relisted_sampler_tables(device.compile_device(), &samplers, self.placement)
            .map_err(Error::Bind)?
        else {
            let unlisted = unlisted.unwrap_or_else(|| {
                (
                    format!("{} listing the argument samplers", self.commands),
                    "the sampler states live as their command buffer commits, or as their batch \
                     is submitted, are more than their plan's sampler tables hold, or of a shape \
                     its checks refuse, so the commands keep the states they listed before, and \
                     a function reading another reads the neutral sampler"
                        .to_owned(),
                )
            });
            return Ok(Relisted {
                samplers: Vec::new(),
                unlisted: Some(unlisted),
            });
        };
        for (key, region) in &self.tables {
            let table = bytes
                .iter()
                .find(|table| (table.set, table.binding) == *key)
                .ok_or_else(|| internal("a sampler table its plan does not list"))?;
            region.write(&table.bytes);
        }
        let (writes, owned) = binding::argument_sampler_writes(device, &self.plan, &live)?;
        if !writes.is_empty() {
            heap_sets::bind(device, &self.objects, heaps, writes, None)?;
        }
        self.listed = samplers;
        Ok(Relisted {
            samplers: owned,
            unlisted,
        })
    }
}

/// What [`SamplerListing::relist`] made of the live states.
pub(super) struct Relisted {
    /// The Vulkan samplers the new elements name, none where nothing changed.
    pub(super) samplers: Vec<Arc<OwnedSampler>>,
    /// Where a live state is left out, the object and the reason of the
    /// `unlisted-argument-sampler` notice.
    pub(super) unlisted: Option<(String, String)>,
}
