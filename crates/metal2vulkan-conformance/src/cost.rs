//! The cost half: what **our** side did to produce a case's transcript.
//!
//! The suite's correctness half records what a case *observed*, field by field, against Metal.
//! This is the other half: for the same case set, the operations our own implementation performed
//! to answer it. The two never mix.
//!
//! # Counters are ours, not Metal's
//!
//! Metal has no counterpart for any of these counts, and a host with no Metal can have no answer
//! for them, so **a count never enters the comparison with Metal and never produces a correctness
//! verdict**. The cost half is a gate of its own, with its own three verdicts
//! ([`CostVerdict`]): `Within`, `Over`, `Unmeasured`.
//!
//! # Why counts and not times
//!
//! A count is deterministic, so it is assertable on any host under any load; a time is not. A
//! regression then reads "this case now does two allocations where it did none", which is a stable
//! gate and a better bug report than a percentage nobody can reproduce. Where a count was measured
//! here *not* to be deterministic, the record says so and names the range it was seen in
//! ([`CostBound::range`]), and that count is reported rather than gated — a bound that fails at
//! random is worse than no bound.
//!
//! # The mechanism is the runtime's own
//!
//! [`metal2vulkan_runtime::WorkCounts`] already counts what the runtime did; nothing here invents a
//! parallel mechanism. It is read twice around each case, and the difference is that case's cost.
//! The read is cheap and always on, because product behaviour may not branch on the environment:
//! the suite never asks the runtime to count *more* for a cost run than for an ordinary one.
//!
//! # Attribution
//!
//! Three regions are kept apart ([`Region`]), because a change that moves work from one to another
//! is not a win:
//!
//! - **translation and pipeline creation**, the one-time cost that already has its own ceilings
//!   (20 seconds and 500 MiB per translation);
//! - **the device API**, what a call does on the calling thread — the gap the plan names;
//! - **submission**, what the commit path, our own other threads and the GPU do afterwards. A call
//!   that returns at once while queueing work for a later commit must not score as a win, so the
//!   work it queued is counted here rather than vanishing.
//!
//! # Where a bound lives
//!
//! Beside the case's transcript, in the case's own recording file ([`crate::recording::Recording`]),
//! so a host with no Metal replays it unchanged. A record names the venue it was measured on and
//! the measurement that justified it; a venue the record does not hold is `Unmeasured`, never a
//! pass.

use metal2vulkan_runtime::WorkCounts;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A field of [`WorkCounts`] the cost half does not read, and why.
///
/// A `*_sites` field is a table of the runtime's own source positions — a file, a line and what
/// was recorded there — rather than a count, so the difference of two readings of it is not a
/// case's cost. What those tables name is already counted elsewhere: `copy_sites` says *where* the
/// copies were recorded, and the copies themselves are `commands_replayed.cmd_copy_*`. The rule is
/// a suffix rather than a list of names because the runtime keeps growing such tables.
const NOT_A_COUNT: [(&str, &str); 1] = [(
    "_sites",
    "a table of the runtime's source positions and what each recorded, not a count; what those \
     places did is counted in the counts beside it",
)];

/// A suffix a field name carries when it holds a high-water mark or an accumulated time rather
/// than a count, with why the cost half does not read it.
///
/// A maximum does not subtract: the difference of two high-water marks is not the work one case
/// did. A time is not a fact a gate can hold, which is the whole reason this half gates on counts.
const NOT_SUBTRACTABLE: [(&str, &str); 3] = [
    ("_most", "a high-water mark, which does not subtract"),
    ("_peak", "a high-water mark, which does not subtract"),
    ("_nanos", "an accumulated time, which no count gate holds"),
];

/// Whether `name` is a field the cost half reads, or why it is not.
pub fn excluded(name: &str) -> Option<&'static str> {
    let head = name.split('.').next().unwrap_or(name);
    for (suffix, why) in NOT_A_COUNT {
        if head.ends_with(suffix) {
            return Some(why);
        }
    }
    let leaf = name.rsplit('.').next().unwrap_or(name);
    for (suffix, why) in NOT_SUBTRACTABLE {
        if leaf.ends_with(suffix) {
            return Some(why);
        }
    }
    None
}

/// Why a count of what is alive, parked or waiting **at the instant the counts were read** is
/// reported and never gated.
///
/// It is not work a case did; it is a reading of process-wide state that happens to be taken at
/// the case's boundary. Measured on this M3 Max: two replays of the whole set on the same binary
/// over the same 350 cases put `empty_memory_blocks_live` over its bound on
/// `xr-two-planes-rgb10a8-2p-xr10-srgb` in one run and on `depth16-texel-reads`,
/// `texture-fills-53` and `sample-fills-d32-2` in the other, and `memory_blocks_live` over it in
/// the second run only. Run alone, the same three texture-buffer cases read
/// `empty_memory_blocks_live` 2 and `memory_blocks_live` 1 where inside the set they read 0 or 1.
/// A count that lands on a different case every run cannot be bounded on any of them.
const INSTANTANEOUS: &str = "a reading of what was alive, parked or waiting at the instant the \
     counts were read, which is process-wide state and not work the case did: measured landing on \
     a different case in each of two replays of the same binary over the same cases";

/// Why a count of what an object's **release** got round to is reported and never gated: the same
/// reason as the `object_churn.` tree, for the counts that live outside it.
///
/// Measured: `sampler_sets_destroyed` reached 1 where the measurement's most was 0 on
/// `xr-two-planes-rgb10a8-2p-xr10-srgb` in two replays of the whole set, and reads 0 when that
/// case runs alone.
const RELEASED: &str = "what a release got round to destroying, which runs on whichever thread \
     drops the object and may land in a later case's window";

/// Why a count of an object **found already made** rather than made is reported and never gated.
///
/// Whether this case makes a pipeline layout or finds one depends on what every case before it in
/// the same process made, so the split between the two is a fact about the selection and not about
/// the case. Its partner stays gated and loses nothing: finding one only ever **lowers** the
/// `_made` count, and the gate is an upper bound.
///
/// Measured, in the one direction that matters. `depth16-texel-reads` run alone makes 2 pipeline
/// layouts and shares none; inside the 350-case set it makes 1 and shares 1. Two and two.
/// `xr-two-planes-rgb10a8-2p-xr10-srgb` run alone makes 8 pipeline variants and takes none back;
/// inside the set it makes 4 and takes 4 back — and `variants_taken_back` is the one count of this
/// family measured to go **up** with more cases around it, from 0 to 4, which is the direction an
/// upper bound cannot survive.
///
/// Every other count that moved between the two readings moved the safe way, higher when the case
/// ran alone: `addressable_buffers_made`, `pooled_memory_blocks_made`, `arena_pools_made`,
/// `device_memory_allocated`, `creation_pipelines_made`, `variant_pipelines_made`,
/// `shader_modules_made`, `pipeline_layouts_made`, `command_buffers_begun` and
/// `command_buffers_submitted` are all first-case-pays counts, so the whole-set bound sits at or
/// under what a lone run would do and more cases can only lower them. They stay gated.
const FOUND: &str = "an object found already made rather than made, which depends on what every \
     case before it in the same process made. Its `_made` partner stays gated, and loses nothing: \
     finding one only lowers that count, and the gate is an upper bound";

/// Why `completed_waits` is reported and never gated.
///
/// It looked like the one deterministic breach — exactly one over its bound, in the same three
/// cases, in two replays of the whole set. The contrast run says otherwise: run **alone**, those
/// cases wait 25, 225 and 5 times; inside the 350-case set they wait 30, 230 and 10, and inside
/// the 126-case set the bounds were taken from, 29, 229 and 9. So the case's own waits are the
/// smaller number and the rest are waits of whatever ran before it, landing in this case's window
/// because the completion waiter is a thread of ours. How many it borrows moves with the
/// selection, which is why a bound on it is a bound on the selection.
const BORROWED: &str = "waits a case is charged for that belong to whatever ran before it: the \
     completion waiter is a thread of ours, so a wait lands in whichever case's window it reaches. \
     Measured: 25, 225 and 5 waits when three cases run alone against 30, 230 and 10 inside the \
     whole set";

/// Why a count of an object made because no spare was free is reported and never gated.
const POOLED: &str = "an object made because no spare was free, where a spare comes back on \
     another thread: which case has to make one depends on when that thread returned the one \
     before it, not on the case's own work. Measured: `fences_made` reached 1 where the \
     measurement's most was 0 on three whole-set replays in a row, in a different case each time";

/// A count whose value says **which thread arrived first**, with why it is reported and never
/// gated, whatever a measurement happened to see.
///
/// The cost half gates on counts because a count is deterministic. These are not, structurally,
/// and no number of measurements makes them so: holding them to a bound gives a gate that fires at
/// random on honest work. Every entry's reason is the runtime's own words about itself, and
/// every one of them was then measured here.
///
/// Measured on this M3 Max over seven measurements of the whole case set (3 to 10 rounds of 126
/// cases, about 6,300 counts a round): 24, 27, 35, 38, 47, 49 and 68 readings moved, and every one
/// of them was one of these families. Replays of the whole set then went over a bound six times,
/// again every time one of them, which is what sent the rule here instead of into a longer
/// measurement.
///
/// `fences_made` is the one that had to be read twice. Each measurement saw it move by a single
/// reading in a single case, which looked like too little to be a family — until three replays in
/// a row went over its bound in a **different case each time**. A measurement's range cannot cover
/// that, because the case the race lands in is not fixed, and the mechanism is the same as the
/// others': a fence is made where no spare one is free, and a spare comes back on the completion
/// thread.
///
/// `completed_waits` is the one that had to be read a third time, and it is the sharpest warning
/// against trusting a measurement over an argument: it held still over every round of every
/// measurement here, then came in at **exactly one over** its bound in the same three cases in two
/// replays of a larger case set — which reads like a regression and is not one. See [`BORROWED`].
/// `slot_returns` beside it stays gated: it counts work items, one a command buffer, and has held
/// still alone and in both case sets.
const RACY: [(&str, &str); 25] = [
    ("fences_made", POOLED),
    // What is alive, parked or waiting at the instant the counts were read.
    ("memory_blocks_live", INSTANTANEOUS),
    ("empty_memory_blocks_live", INSTANTANEOUS),
    ("dedicated_memory_blocks_live", INSTANTANEOUS),
    ("host_imports_live", INSTANTANEOUS),
    ("dma_buf_memories_live", INSTANTANEOUS),
    ("dma_buf_image_memories_live", INSTANTANEOUS),
    ("shadow_merge_kept_images_live", INSTANTANEOUS),
    ("freed_memory_parked", INSTANTANEOUS),
    ("freed_memory_waiting", INSTANTANEOUS),
    ("host_import_releases_pending", INSTANTANEOUS),
    ("host_imports_resident", INSTANTANEOUS),
    // A release the owning thread got round to, outside the `object_churn.` tree.
    ("sampler_sets_destroyed", RELEASED),
    // Found already made rather than made. Named one by one rather than by the `_shared` suffix,
    // which `set_zero_writes_if_shapes_shared` also carries and which is a per-draw count of the
    // case's own work.
    ("pipeline_layouts_shared", FOUND),
    ("shader_modules_shared", FOUND),
    ("sampler_sets_shared", FOUND),
    ("variants_taken_back", FOUND),
    ("completed_waits", BORROWED),
    ("command_pools_made", POOLED),
    ("spare_command_pools_", POOLED),
    ("live_tables_uploaded", POOLED),
    (
        "scheduler_",
        "how many times one of our own background threads woke, slept or made a pass, which is \
         when that thread ran and not what the case asked for",
    ),
    (
        "wakes_",
        "whether a transition had to wake a thread or found one already running, which is a \
         question about timing rather than about the case's work",
    ),
    (
        "command_buffers_replayed",
        "which thread replayed a deferred command buffer: the recorder's, or one that reached it \
         first. Their total is command_buffers_deferred, which is gated, so nothing is lost",
    ),
    (
        "object_churn.",
        "the release path, which runs on whichever thread drops the object. A release may land in \
         a later case's window, and whether a making takes a parked object or makes a new one \
         depends on whether the release before it has landed",
    ),
];

/// Why `name` is reported and never gated, where it is.
pub fn racy(name: &str) -> Option<&'static str> {
    RACY.iter()
        .find(|(pattern, _)| name.starts_with(pattern))
        .map(|(_, why)| *why)
}

/// Which region of the work a count belongs to.
///
/// The regions are kept apart in every report, because moving work between them is not a win.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Region {
    /// Making a shader module, a pipeline layout or a pipeline: the one-time cost, which already
    /// has its own ceilings.
    Translation,
    /// What a call does on the calling thread: the region the plan names as the gap.
    DeviceApi,
    /// What the commit path, our own other threads and the GPU do: the commands replayed into
    /// `VkCommandBuffer`s, the submissions, the scheduler's passes and the waits. Work a call
    /// queues for later is counted here, so a call that returns at once cannot score as a win.
    Submission,
}

impl Region {
    /// Every region, in the order a report prints them.
    pub const ALL: [Region; 3] = [Region::Translation, Region::DeviceApi, Region::Submission];

    /// The words a report prints.
    pub fn word(self) -> &'static str {
        match self {
            Region::Translation => "translation",
            Region::DeviceApi => "device-api",
            Region::Submission => "submission",
        }
    }
}

/// The name prefixes that put a count in [`Region::Translation`]: making a module, a layout or a
/// pipeline.
const TRANSLATION: [&str; 15] = [
    "shader_modules_",
    "pipeline_layouts_",
    "variant_pipelines_",
    "pipelines_pooled",
    "pipeline_pool_busy",
    "creation_pipelines_",
    "graphics_pipelines_looked_up",
    "pipeline_libraries_",
    "fragment_shader_libraries_",
    "pipelines_linked",
    "dispatch_pipelines_pooled",
    "tile_pipelines_pooled",
    "variants_taken_back",
    "flights_followed",
    "residual_form_makings",
];

/// The name prefixes that put a count in [`Region::Submission`]: the commit path, our own other
/// threads and the GPU.
const SUBMISSION: [&str; 41] = [
    "commands_replayed.",
    "command_buffers_",
    "command_pools_made",
    "spare_command_pools_",
    "encoder_command_buffers_begun",
    "encoder_primaries_begun_ahead",
    "primaries_begun_ahead",
    "queue_submissions",
    "submit_infos",
    "internal_submissions",
    "staging_submissions",
    "grouped_submissions",
    "batches_",
    "group_ends_",
    "groups_",
    "registrations_waited",
    "scheduler_",
    "wakes_",
    "slot_returns",
    "slot_waits",
    "completed_waits",
    "event_waits_met",
    "timeline_waits",
    "missed_completion_wakes",
    "completed_where_seen",
    "scheduled_where_submitted",
    "timestamp_slots_",
    "calibration",
    "queue_drains",
    "commits_handed_off",
    "render_log_",
    "replay_logs_kept",
    "encoders_reencoded",
    "segments_recorded_again",
    "replay_tables_pointed",
    "reencodes_refused",
    "finales_",
    "barrier_buffers_kept",
    "stamped_buffers_",
    "bare_starts",
    "prefixed_starts",
];

/// Which region `name` belongs to.
///
/// The longest matching prefix wins, and a name no rule names is the device API's: that is the
/// residual on purpose, because the device API is where a count lands unless it is demonstrably
/// one of the other two, and a count quietly attributed to translation or to submission would read
/// as a call costing nothing.
pub fn region_of(name: &str) -> Region {
    let best = |rules: &[&str]| -> usize {
        rules
            .iter()
            .filter(|rule| name.starts_with(**rule))
            .map(|rule| rule.len())
            .max()
            .unwrap_or(0)
    };
    let translation = best(&TRANSLATION);
    let submission = best(&SUBMISSION);
    if translation == 0 && submission == 0 {
        return Region::DeviceApi;
    }
    if translation >= submission {
        Region::Translation
    } else {
        Region::Submission
    }
}

/// The counts one run moved, by name: the cost half's reading of [`WorkCounts`].
///
/// The names are the runtime's own field names, nested ones joined with a dot
/// (`commands_replayed.cmd_draw`), so a failure names the counter a reader can grep for.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Counts(BTreeMap<String, u64>);

impl Counts {
    /// The counts `counts` holds, flattened.
    ///
    /// Read out of the derived `Debug`, which is the one complete view of the struct a sibling
    /// crate has: [`WorkCounts`] has 386 fields of its own plus three nested count structs, and a
    /// hand-written list of them would be stale the day a runtime change added one. A field whose
    /// value is not a count fails here by name rather than being dropped, so a new field of
    /// another shape is a loud failure in this crate's own tests instead of a silent hole in the
    /// gate.
    pub fn of(counts: &WorkCounts) -> Result<Self, String> {
        Self::read(&format!("{counts:#?}"))
    }

    /// [`Counts::of`]'s parse, over the `Debug` text itself, so a test can feed it a shape.
    pub fn read(debug: &str) -> Result<Self, String> {
        let mut read: BTreeMap<String, u64> = BTreeMap::new();
        let mut path: Vec<String> = Vec::new();
        let mut skipping: Option<usize> = None;
        for (index, line) in debug.lines().enumerate() {
            let at = index + 1;
            let body = line.trim();
            if body.is_empty() {
                continue;
            }
            let indent = line.len() - line.trim_start().len();
            if indent % 4 != 0 {
                return Err(format!("line {at}: {body}: an indent of {indent} spaces"));
            }
            let depth = indent / 4;
            if let Some(from) = skipping {
                if depth == from && (body == "]," || body == "}," || body == "]" || body == "}") {
                    skipping = None;
                }
                continue;
            }
            if body == "}," || body == "]," || body == "}" || body == "]" {
                path.pop();
                continue;
            }
            if body.ends_with('{') || body.ends_with('[') {
                let name = match body.split_once(':') {
                    Some((name, _)) => name.trim().to_owned(),
                    // The root (`WorkCounts {`), or an element of an array.
                    None => String::new(),
                };
                if excluded(&name).is_some() {
                    skipping = Some(depth);
                } else {
                    path.push(name);
                }
                continue;
            }
            let (name, value) = body
                .split_once(':')
                .ok_or_else(|| format!("line {at}: {body}: neither a field nor a brace"))?;
            let value = value.trim().trim_end_matches(',');
            let count: u64 = value.parse().map_err(|_| {
                format!(
                    "line {at}: {}: {value} is not a count. A field of WorkCounts that is not a \
                     count must be named in cost::NOT_A_COUNT with the reason, so the cost gate \
                     does not quietly drop it.",
                    name.trim()
                )
            })?;
            let mut full = String::new();
            for part in path.iter().skip(1) {
                full.push_str(part);
                full.push('.');
            }
            full.push_str(name.trim());
            if excluded(&full).is_none() {
                read.insert(full, count);
            }
        }
        if !path.is_empty() {
            return Err(format!("{} unclosed brace(s)", path.len()));
        }
        Ok(Self(read))
    }

    /// The work done between `before` and this reading: what one case cost.
    ///
    /// A count that went down (a count of live objects, where the case released more than it made)
    /// reads as zero: the cost half holds what a case *did*, and a bound on a negative difference
    /// would hold nothing.
    pub fn since(&self, before: &Counts) -> Counts {
        let mut moved = BTreeMap::new();
        for (name, after) in &self.0 {
            let was = before.0.get(name).copied().unwrap_or(0);
            let did = after.saturating_sub(was);
            if did > 0 {
                moved.insert(name.clone(), did);
            }
        }
        Counts(moved)
    }

    /// The counts and their values, by name.
    pub fn iter(&self) -> impl Iterator<Item = (&str, u64)> {
        self.0.iter().map(|(name, count)| (name.as_str(), *count))
    }

    /// What `name` reached, zero where the run never moved it.
    pub fn get(&self, name: &str) -> u64 {
        self.0.get(name).copied().unwrap_or(0)
    }

    /// How many counts the run moved.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the run moved none.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The counts of `region`, with their values.
    pub fn in_region(&self, region: Region) -> Vec<(&str, u64)> {
        self.iter()
            .filter(|(name, _)| region_of(name) == region)
            .collect()
    }

    /// The largest of two readings, count by count: what a measurement of several runs holds.
    pub fn largest(&self, other: &Counts) -> Counts {
        let mut most = self.0.clone();
        for (name, count) in &other.0 {
            let held = most.entry(name.clone()).or_insert(0);
            *held = (*held).max(*count);
        }
        Counts(most)
    }

    /// The smallest of two readings, count by count: the other end of a measurement's range.
    pub fn smallest(&self, other: &Counts) -> Counts {
        let mut least: BTreeMap<String, u64> = BTreeMap::new();
        for name in self.0.keys().chain(other.0.keys()) {
            let mine = self.get(name);
            let theirs = other.get(name);
            least.insert(name.clone(), mine.min(theirs));
        }
        Counts(least)
    }
}

/// One count a case is held to, and the measurement's own reading of it.
///
/// A view of one entry of a [`CostRecord`], built from it rather than stored: the record keeps the
/// counts as two maps of name to number, one line a count, because the case set's bounds are about
/// 6,300 counts and what a reader diffs after a re-measurement should be one line per count that
/// moved rather than five.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CostBound {
    /// The runtime's name for the count.
    pub count: String,
    /// Which region of the work it belongs to, as [`region_of`] says. Read from the name rather
    /// than stored, so a file can never disagree with the code's own classification.
    pub region: Region,
    /// The most the measurement saw: the bound. A run over it is [`CostVerdict::Over`].
    pub most: u64,
    /// The least the measurement saw, where the measurement's runs disagreed.
    ///
    /// A count that moved between runs of the same case on the same venue is **not** deterministic,
    /// so it is reported rather than gated: a bound that fails at random is worse than no bound.
    /// This is the evidence for not gating it, and it names the range it was seen in.
    pub range: Option<u64>,
    /// Why this count is reported and never gated whatever the measurement saw ([`racy`]).
    pub racy: Option<&'static str>,
}

impl CostBound {
    /// Whether this count is gated: the measurement found it deterministic, and it is not one of
    /// the counts that say which thread arrived first.
    pub fn gated(&self) -> bool {
        self.range.is_none() && self.racy.is_none()
    }
}

/// What one venue's runs of a case cost, and the measurement behind it.
///
/// It lives in the case's recording, beside the Metal answers, so a host with no Metal reads it
/// with the rest of the file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostRecord {
    /// The venue the counts were measured on ([`crate::case::Venue::label`]). A run on any other
    /// venue is [`CostVerdict::Unmeasured`]: our own cost is the implementation's *and* the
    /// driver's, and one venue's counts are no bound for another's.
    pub venue: String,
    /// The runs behind the bounds.
    pub runs: u32,
    /// The measurement that justified the bounds, in words: what was run, where, and how the
    /// bounds were taken from it. The suite's rule is that a bound names its measurement, and this
    /// is where a reader finds it.
    pub measured: String,
    /// The degrade-notice codes the measurement's runs announced, sorted.
    ///
    /// **This is what keeps a device's gap from reading as a cost win.** A count that is lower
    /// because the device did less of the case — it lacks the exact form, so the runtime announced
    /// a notice and did something cheaper — is not work we saved. A run whose announced set is not
    /// this one did a different amount of work, so it is [`CostVerdict::Unmeasured`] against these
    /// bounds rather than compared with them: the same rule the correctness half applies when it
    /// keeps a device's honest gap out of the pass and the failure counts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub degraded: Vec<String>,
    /// The most each count the runs moved reached. A count absent from here is bounded at zero, so
    /// a case that starts doing something it did not do before is `Over` rather than unnoticed.
    pub bounds: BTreeMap<String, u64>,
    /// The least each count reached whose readings **disagreed** between the runs: the other end
    /// of its range, and the record of the fact that the count is not deterministic here. A count
    /// named here is reported rather than gated.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub moving: BTreeMap<String, u64>,
}

impl CostRecord {
    /// The bound for `count`, where the record holds one.
    pub fn bound(&self, count: &str) -> Option<CostBound> {
        let most = *self.bounds.get(count)?;
        Some(CostBound {
            count: count.to_owned(),
            region: region_of(count),
            most,
            range: self.moving.get(count).copied(),
            racy: racy(count),
        })
    }

    /// Every bound the record holds, in name order.
    pub fn held(&self) -> Vec<CostBound> {
        self.bounds
            .keys()
            .filter_map(|count| self.bound(count))
            .collect()
    }

    /// How many of its bounds are gated rather than only reported.
    pub fn gated(&self) -> usize {
        self.held().iter().filter(|bound| bound.gated()).count()
    }
}

/// A count a run moved further than its bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Overrun {
    pub count: String,
    pub region: Region,
    pub bound: u64,
    pub found: u64,
}

impl std::fmt::Display for Overrun {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            out,
            "{} ({}) reached {} where the measurement's most was {}",
            self.count,
            self.region.word(),
            self.found,
            self.bound
        )
    }
}

/// What the cost half says of one case on one venue. Never a correctness verdict, and never mixed
/// with one: Metal has no counterpart for any of these counts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CostVerdict {
    /// Every gated count stayed at or under its bound.
    Within {
        /// The gated counts the run was held to.
        held: usize,
    },
    /// A gated count went over its bound: this case now does more work than the measurement saw.
    Over { over: Vec<Overrun> },
    /// No bound stands for this run, and why. Counted, never a pass: "untimed" and "uncounted" are
    /// tracked the way "unrecorded" is.
    Unmeasured { why: String },
}

impl CostVerdict {
    /// The word a report prints.
    pub fn word(&self) -> &'static str {
        match self {
            CostVerdict::Within { .. } => "within",
            CostVerdict::Over { .. } => "over",
            CostVerdict::Unmeasured { .. } => "unmeasured",
        }
    }

    /// Whether the run is within its bounds. An unmeasured run is not over them, so it does not
    /// fail the gate; it is counted as the gap it is.
    pub fn is_within(&self) -> bool {
        !matches!(self, CostVerdict::Over { .. })
    }
}

impl std::fmt::Display for CostVerdict {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CostVerdict::Within { held } => write!(out, "within {held} bound(s)"),
            CostVerdict::Over { over } => {
                write!(out, "over {} bound(s)", over.len())?;
                for one in over {
                    write!(out, "\n  {one}")?;
                }
                Ok(())
            }
            CostVerdict::Unmeasured { why } => write!(out, "unmeasured: {why}"),
        }
    }
}

/// Hold `observed` to the bounds `records` holds for `venue`.
///
/// `whole_set` says whether the run was of the whole case set, which is the measurement's own
/// shape: a case's counts depend on what the process did before it, because the runtime pools
/// command pools, memory blocks and descriptor sets across cases. A selection of a few cases does
/// *more* fresh work than the same cases inside a whole-set run, so its counts are not the ones
/// the bounds were measured over, and holding them to those bounds would fail honestly-identical
/// work. Such a run is `Unmeasured`.
pub fn judge(
    records: &[CostRecord],
    venue: &str,
    whole_set: bool,
    observed: &Counts,
    stopped: Option<&str>,
    degraded: &[String],
) -> CostVerdict {
    if let Some(why) = stopped {
        return CostVerdict::Unmeasured {
            why: format!("the run stopped before the case finished ({why}), so its counts are of a part of the case"),
        };
    }
    if !whole_set {
        return CostVerdict::Unmeasured {
            why: "the selection is not the whole case set, and a case's counts depend on what ran \
                  before it in the same process"
                .to_owned(),
        };
    }
    let Some(record) = records.iter().find(|record| record.venue == venue) else {
        let held: Vec<&str> = records.iter().map(|record| record.venue.as_str()).collect();
        return CostVerdict::Unmeasured {
            why: if held.is_empty() {
                format!("no cost record was taken for {venue}, nor for any venue")
            } else {
                format!("no cost record was taken for {venue}; the case holds {held:?}")
            },
        };
    };
    // A device that could not do the case exactly did less of it, and its lower counts are its
    // gap and not our saving. Comparing such a run with a measurement of a different gap would
    // read the gap as a win, so the two are not comparable at all.
    if record.degraded != degraded {
        return CostVerdict::Unmeasured {
            why: format!(
                "this run announced {degraded:?} and the measurement announced {:?}, so the two \
                 did not do the same amount of the case: a count lower because the device lacks \
                 the exact form is the device's gap, not a cost we saved",
                record.degraded
            ),
        };
    }
    let mut over = Vec::new();
    for (count, found) in observed.iter() {
        let (bound, gated) = match record.bound(count) {
            Some(bound) => (bound.most, bound.gated()),
            // A count the measurement never saw move is bounded at zero, and it is gated unless
            // it is one of the thread-race counts: a case that starts doing something it did not
            // do before is exactly what this half is for.
            None => (0, racy(count).is_none()),
        };
        if gated && found > bound {
            over.push(Overrun {
                count: count.to_owned(),
                region: region_of(count),
                bound,
                found,
            });
        }
    }
    if over.is_empty() {
        CostVerdict::Within {
            held: record.gated(),
        }
    } else {
        CostVerdict::Over { over }
    }
}

/// What one case's cost run came to: the counts it moved, and how long it took.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaseCost {
    pub case: String,
    /// The counts of the run with the largest readings, where several runs were taken.
    pub counts: Counts,
    /// The smallest reading of each count, where several runs were taken: the other end of the
    /// range. Equal to `counts` where every run agreed.
    pub least: Counts,
    /// The runs taken.
    pub runs: u32,
    /// Why the run stopped before the case finished, where it did. Its counts are of a part of the
    /// case, so nothing is recorded or gated from them.
    pub stopped: Option<String>,
    /// The degrade-notice codes the runs announced, sorted: what the device could not do exactly,
    /// and so how much of the case the counts are of.
    pub degraded: Vec<String>,
    /// How long the longest run took, in nanoseconds. **Advisory**: a time is not a fact this
    /// gate holds, and per-call timing against Metal is its own step. It is printed so a reader
    /// can see which cases the counts are large for.
    pub nanos: u64,
}

impl CaseCost {
    /// The record this measurement justifies, for `venue`, with `measured` the words that say what
    /// was run.
    ///
    /// A count whose runs disagreed carries its range and is reported rather than gated.
    pub fn record(&self, venue: &str, measured: String) -> CostRecord {
        let mut bounds = BTreeMap::new();
        let mut moving = BTreeMap::new();
        for (count, most) in self.counts.iter() {
            bounds.insert(count.to_owned(), most);
            let least = self.least.get(count);
            // A count the rule already names as racy is not written here: the rule ungates it
            // whatever a measurement saw, so an entry would be redundant. What that leaves is the
            // useful list — a count written here moved for a reason the rule does not explain.
            if least < most && racy(count).is_none() {
                moving.insert(count.to_owned(), least);
            }
        }
        CostRecord {
            venue: venue.to_owned(),
            runs: self.runs,
            measured,
            degraded: self.degraded.clone(),
            bounds,
            moving,
        }
    }

    /// The counts whose readings disagreed between runs: the ones no bound gates.
    pub fn moved(&self) -> Vec<(&str, u64, u64)> {
        self.counts
            .iter()
            .filter_map(|(count, most)| {
                let least = self.least.get(count);
                (least < most && racy(count).is_none()).then_some((count, least, most))
            })
            .collect()
    }

    /// The counts whose readings disagreed for a reason the [`racy`] rule already names: which
    /// thread arrived first. Reported apart, because they explain themselves.
    pub fn moved_by_a_race(&self) -> Vec<(&str, u64, u64)> {
        self.counts
            .iter()
            .filter_map(|(count, most)| {
                let least = self.least.get(count);
                (least < most && racy(count).is_some()).then_some((count, least, most))
            })
            .collect()
    }
}

/// How many counts a reading moved in each region, for the cost report's own summary.
///
/// How *many*, not their sum. The counts are of different things — command buffers, images,
/// allocations, texels copied, bytes of device memory — so adding them gives a number with no
/// unit: one case's reading here summed to 1,073,750,616 in the device-API region, almost all of
/// it one texel count. The individual counts are what the gate holds; this says where a case's
/// work falls.
pub fn by_region(counts: &Counts) -> BTreeMap<Region, usize> {
    let mut many = BTreeMap::new();
    for (name, _) in counts.iter() {
        *many.entry(region_of(name)).or_default() += 1;
    }
    many
}
