//! Running the suite: record a selection's Metal answers, and replay a selection on a venue.
//!
//! One function per leg, so the command, a test and a device lane all drive the same code.

use crate::case::{Case, Side};
use crate::cost::{self, CaseCost, Counts};
use crate::device::{self, Notices, Policy, ScratchCache, Venues};
use crate::replay::replay_known;
use crate::report::{CostRow, Report};
use crate::runtime_side::RuntimeSide;

/// The runs record mode takes a case's answer over by default, so a case whose behaviour varies
/// records its outcomes with their frequencies rather than whichever run came last.
///
/// A case whose work nothing orders asks for more through
/// [`crate::case::Case::record_runs`].
pub const RECORD_RUNS: u32 = 3;

/// The cases `names` selects, or every case where `names` is empty.
pub fn selected(names: &[String]) -> Result<Vec<Box<dyn Case>>, String> {
    if names.is_empty() {
        return Ok(crate::cases::all());
    }
    names
        .iter()
        .map(|name| {
            crate::cases::named(name).ok_or_else(|| format!("the suite holds no case {name}"))
        })
        .collect()
}

/// Replay `cases` on the venue `venues` names, with exact answers, as a report.
pub fn replay_on(venues: Venues, cases: &[Box<dyn Case>]) -> Result<Report, String> {
    replay_on_under(venues, Policy::Exact, cases)
}

/// [`replay_on`] under `policy`: either form of every speed-for-accuracy choice.
pub fn replay_on_under(
    venues: Venues,
    policy: Policy,
    cases: &[Box<dyn Case>],
) -> Result<Report, String> {
    let notices = Notices::default();
    let cache = ScratchCache::new().map_err(|error| error.to_string())?;
    let device = device::open_under(venues, policy, &notices, &cache)?;
    // The notices a device announces as it is made are the venue's, not any case's: they say what
    // this host's driver and release cost, once, before a case runs. They go in the report's venue
    // line, and the sink is cleared so a case's own notices site holds only what its work
    // announced.
    let mut report = Report::new();
    report.venue_announced(device.name(), notices.take());
    // A notice is announced once per device per kind, so the codes a case provokes are the ones
    // nothing earlier in the run provoked. The set carries forward: a case's own notices site
    // holds its own work's notices alone, and the device's notices justify a loosened field in
    // any case of the run.
    let mut announced_on_device: Vec<String> = Vec::new();
    // The cost half's gate: the counts the runtime moved over each case, held to the bounds the
    // case's own recording carries for this venue. It rides along with the correctness replay
    // because the two readings must be of the same run, and it is cheap: two reads of
    // `WorkCounts` a case. Nothing here enters a correctness verdict.
    let whole_set = cases.len() == crate::cases::all().len();
    let venue_label = device.name().to_owned();
    for case in cases {
        let taker = notices.clone();
        let before = read_counts(&device);
        let started = std::time::Instant::now();
        let mut side = RuntimeSide::new(&device, move || taker.take());
        let replayed = replay_known(case.as_ref(), &mut side, &announced_on_device);
        drop(side);
        let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let moved = match (before, read_counts(&device)) {
            (Ok(before), Ok(after)) => Ok(after.since(&before)),
            (Err(why), _) | (_, Err(why)) => Err(why),
        };
        let mut degraded: Vec<String> = match &replayed {
            Ok(replayed) => replayed.ours.announced_notices().into_keys().collect(),
            Err(_) => Vec::new(),
        };
        degraded.sort();
        degraded.dedup();
        if let Ok(replayed) = &replayed {
            for code in replayed.ours.announced_notices().into_keys() {
                if !announced_on_device.contains(&code) {
                    announced_on_device.push(code);
                }
            }
        }
        report.add_cost(cost_row(
            case.as_ref(),
            &venue_label,
            whole_set,
            moved,
            nanos,
            &degraded,
            &replayed,
        ));
        report.add_result(replayed);
    }
    Ok(report)
}

/// The device's counts, or nothing where they could not be read as counts.
///
/// A reading that does not parse is the cost half's own failure, never the case's: it is reported
/// as an unmeasured row naming the reason, so a runtime field of a new shape shows up as a hole in
/// the gate instead of as a correctness result.
fn read_counts(device: &metal2vulkan_runtime::Device) -> Result<Counts, String> {
    Counts::of(&device.work_counts())
}

/// How one case's counts stand against its recording's bounds.
fn cost_row(
    case: &dyn Case,
    venue: &str,
    whole_set: bool,
    moved: Result<Counts, String>,
    nanos: u64,
    degraded: &[String],
    replayed: &Result<crate::replay::Replayed, crate::replay::ReplayError>,
) -> CostRow {
    let counts = match moved {
        Ok(counts) => counts,
        Err(why) => {
            return CostRow {
                case: case.name().to_owned(),
                verdict: cost::CostVerdict::Unmeasured {
                    why: format!("the runtime's counts could not be read: {why}"),
                },
                counts: Counts::default(),
                nanos,
                reported: [0; 3],
            }
        }
    };
    // A run the venue stopped — it lacks the feature, or the case could not be replayed at all —
    // did a part of the case, so its counts are not the ones any bound was measured over.
    let stopped = match replayed {
        Ok(row) => match &row.verdict {
            crate::verdict::Verdict::LacksFeature { needs } => {
                Some(format!("the venue lacks {needs}"))
            }
            _ => None,
        },
        Err(error) => Some(error.to_string()),
    };
    let records = crate::recording::read(case.name())
        .map(|recording| recording.costs)
        .unwrap_or_default();
    let verdict = cost::judge(
        &records,
        venue,
        whole_set,
        &counts,
        stopped.as_deref(),
        degraded,
    );
    let reported = records
        .iter()
        .find(|record| record.venue == venue)
        .map(|record| {
            let held = record.held();
            [
                held.iter().filter(|one| one.racy.is_some()).count(),
                held.iter()
                    .filter(|one| one.racy.is_none() && one.range.is_some())
                    .count(),
                held.iter()
                    .filter(|one| {
                        one.racy.is_none() && one.range.is_none() && one.composed.is_some()
                    })
                    .count(),
            ]
        })
        .unwrap_or([0; 3]);
    CostRow {
        case: case.name().to_owned(),
        verdict,
        counts,
        nanos,
        reported,
    }
}

/// Measure what `cases` cost on the venue `venues` names, over `runs` rounds.
///
/// This is the cost half's own leg: it runs our side alone, with no recording read and no
/// comparison, because a count has no Metal counterpart to compare with. Each round runs the whole
/// selection once and a count seen at two values over the rounds is reported rather than gated.
///
/// Two properties of the measurement's shape are load-bearing, and both were learned by measuring:
///
/// - **The whole selection in order**, not one case repeatedly. The runtime pools command pools,
///   memory blocks and descriptor sets across cases, so a case's counts depend on what ran before
///   it. A measurement must have the shape the gate will have.
/// - **A fresh device each round**, because the gate's own run is a fresh one. Reusing a device
///   would measure a *warm* recycler and a warm memory pool: in a first measurement here, 41 cases
///   made a `VkImage` on the first round and took a parked one on the second, which read as a
///   non-deterministic count when it was simply a second round. A bound taken from a warm round
///   would be below what a replay on a cold device does, and the gate would fire on honest work.
pub fn cost_on_under(
    venues: Venues,
    policy: Policy,
    cases: &[Box<dyn Case>],
    runs: u32,
) -> Result<(String, Vec<CaseCost>), String> {
    let mut held: Vec<CaseCost> = Vec::new();
    let mut venue = String::new();
    for round in 0..runs.max(1) {
        let notices = Notices::default();
        let cache = ScratchCache::new().map_err(|error| error.to_string())?;
        let device = device::open_under(venues, policy, &notices, &cache)?;
        notices.take();
        venue = device.name().to_owned();
        for (index, case) in cases.iter().enumerate() {
            let taker = notices.clone();
            let before = read_counts(&device)?;
            let started = std::time::Instant::now();
            let mut side = RuntimeSide::new(&device, move || taker.take());
            let mut sites = crate::transcript::Sites::new();
            let inputs = case
                .inputs()
                .map_err(|why| format!("{}: {why}", case.name()))?;
            sites.push(crate::case::inputs_node(&inputs));
            let abort = case.run(&mut side, &mut sites).err();
            let mut announced = side.announced();
            drop(side);
            let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let moved = read_counts(&device)?.since(&before);
            let stopped = abort.map(|abort| {
                let what = abort
                    .failure
                    .lacks
                    .as_deref()
                    .unwrap_or("a call did not answer");
                format!("{what}, at {}", abort.at)
            });
            announced.extend(sites.finish().announced_notices().into_keys());
            announced.sort();
            announced.dedup();
            if round == 0 {
                held.push(CaseCost {
                    case: case.name().to_owned(),
                    counts: moved.clone(),
                    least: moved,
                    runs: 1,
                    stopped,
                    degraded: announced,
                    nanos,
                });
            } else {
                let held = &mut held[index];
                held.counts = held.counts.largest(&moved);
                held.least = held.least.smallest(&moved);
                held.runs += 1;
                held.nanos = held.nanos.max(nanos);
                // A run that stopped where an earlier one did not is itself a fact about the
                // measurement: the case is not recorded from a measurement that ran only
                // sometimes.
                if held.stopped.is_none() {
                    held.stopped = stopped;
                }
                // The notices a run announced say how much of the case the device could do
                // exactly. A measurement whose runs announced different sets is not one
                // measurement, and the union is what makes that visible.
                for code in announced {
                    if !held.degraded.contains(&code) {
                        held.degraded.push(code);
                    }
                }
                held.degraded.sort();
            }
        }
    }
    Ok((venue, held))
}

/// Measure each of `cases` on its own: a selection of one, on a device no other case has touched,
/// over `runs` rounds.
///
/// This is the other shape a cost measurement can have, and it answers a different question.
/// [`cost_on_under`] measures a case where the gate will run it, after everything before it in the
/// set; this measures the case with nothing before it at all. The difference matters for every
/// count the runtime pools, because a case that runs alone makes what a case in company finds
/// already made: alone is where a `*_made` count is at its largest and a `*_shared` count at its
/// smallest.
///
/// A bound taken alone is therefore an upper bound that no composition can exceed — and so it does
/// not go stale when a case lands beside it, which a bound taken over one particular set does. It
/// buys that with looseness: for a count that pooling lowers a lot, a bound measured alone sits
/// well above what the case does in company, and a gate holding it will not notice a smaller
/// regression.
pub fn cost_alone_on_under(
    venues: Venues,
    policy: Policy,
    cases: &[Box<dyn Case>],
    runs: u32,
) -> Result<(String, Vec<CaseCost>), String> {
    let mut venue = String::new();
    let mut held = Vec::with_capacity(cases.len());
    for case in cases {
        let (name, measured) = cost_on_under(venues, policy, std::slice::from_ref(case), runs)?;
        venue = name;
        held.extend(measured);
    }
    Ok((venue, held))
}

/// Write what `measured` says each case cost on `venue` into the cases' own recordings, answering
/// where each went or why it could not be written.
///
/// A case whose run stopped is **not** recorded: its counts are of a part of the case.
pub fn record_costs(
    venue: &str,
    shape: crate::cost::Shape,
    measured: &[CaseCost],
    words: &str,
) -> Vec<(String, Result<std::path::PathBuf, String>)> {
    measured
        .iter()
        .map(|cost| {
            let written = if let Some(why) = &cost.stopped {
                Err(format!(
                    "the run stopped before the case finished ({why}), so its counts are of a \
                     part of the case"
                ))
            } else {
                record_one_cost(venue, shape, cost, words)
            };
            (cost.case.clone(), written)
        })
        .collect()
}

/// Put one case's cost record in its recording, replacing the venue's own record and keeping every
/// other venue's.
///
/// The bounds are the new measurement's, so a case that now does less work shows it. The set of
/// counts a measurement saw **move** is the union of the new measurement's and the record's,
/// because a count once seen to take two values on this venue is not deterministic there, and a
/// later measurement that happened not to catch it moving has not made it so. Three measurements
/// of this case set found 24, 38, 47 and 68 moving readings of the same ~6,300, all of them the
/// rows that say which thread arrived first, so a single measurement's set is a floor rather than
/// the answer. The union only ever stops a count being gated; it never raises a bound.
fn record_one_cost(
    venue: &str,
    shape: crate::cost::Shape,
    cost: &CaseCost,
    words: &str,
) -> Result<std::path::PathBuf, String> {
    let mut recording = crate::recording::read(&cost.case)?;
    // A measurement of the cases alone does not replace the gate's own shape: it folds into it,
    // and what the two read differently becomes the counts the selection decides.
    if shape == crate::cost::Shape::Alone {
        let held = recording
            .costs
            .iter()
            .find(|held| held.venue == venue)
            .ok_or_else(|| {
                format!(
                    "no cost record for {venue} to contrast with: measure the whole case set \
                     first, then the cases alone"
                )
            })?;
        let folded = cost.fold_alone_into(held, words.to_owned());
        recording.costs.retain(|held| held.venue != venue);
        recording.costs.push(folded);
        recording
            .costs
            .sort_by(|one, two| one.venue.cmp(&two.venue));
        return crate::recording::write(&recording);
    }
    let mut record = cost.record(venue, words.to_owned());
    if let Some(held) = recording.costs.iter().find(|held| held.venue == venue) {
        // The contrast an earlier command took is kept: re-measuring the gate's own shape says
        // nothing new about the other one, and dropping it would silently re-gate every count the
        // selection decides.
        record.contrasted = held.contrasted.clone();
        for (count, least) in &held.moving {
            if crate::cost::racy(count).is_some() {
                continue;
            }
            // A count the new measurement never saw move at all is the sharpest case: without
            // this it would be bounded at zero and gated, and the next run that happened to see
            // the race would read as a regression. It keeps the larger of the two bounds.
            let most = held.bounds.get(count).copied().unwrap_or(0);
            let bound = record.bounds.entry(count.clone()).or_insert(most);
            *bound = (*bound).max(most);
            let seen = record.moving.entry(count.clone()).or_insert(*least);
            *seen = (*seen).min(*least);
        }
        // The same for a count an earlier measurement found the selection decides. A measurement
        // that took only one shape has nothing to say about the contrast, so what an earlier one
        // established is kept rather than silently dropped and the count re-gated.
        for (count, apart) in &held.composed {
            if crate::cost::racy(count).is_some() {
                continue;
            }
            let most = held.bounds.get(count).copied().unwrap_or(0);
            let bound = record.bounds.entry(count.clone()).or_insert(most);
            *bound = (*bound).max(most).max(*apart);
            record.composed.entry(count.clone()).or_insert(*apart);
        }
    }
    recording.costs.retain(|held| held.venue != venue);
    recording.costs.push(record);
    recording
        .costs
        .sort_by(|one, two| one.venue.cmp(&two.venue));
    crate::recording::write(&recording)
}

/// Take `cases`' Metal answers on this Mac and write their recordings, answering where each
/// recording went or why it could not be written.
///
/// A case record mode must not run on Metal is refused there before anything is encoded; a case
/// whose answer is carried writes what its own measurement states. So one command fills a whole
/// selection's recordings without walking into a case whose Metal leg is not safe.
#[cfg(target_os = "macos")]
pub fn record_all(cases: &[Box<dyn Case>]) -> Vec<(String, Result<std::path::PathBuf, String>)> {
    cases
        .iter()
        .map(|case| (case.name().to_owned(), record_one(case.as_ref())))
        .collect()
}

/// Take one case's answer and write its recording.
///
/// A case record mode runs on Metal is run **twice**: plain, which is the behavioural oracle and
/// the transcript comparison uses, and once more under Metal's own API validation layer, in a
/// child process of its own ([`crate::validated`]). The second run is a fact of its own and is
/// never compared with our behaviour.
///
/// A case whose answer is carried was never run here, so its validated answer is
/// [`Validated::NotRun`]: the absence is recorded, not guessed at.
#[cfg(target_os = "macos")]
fn record_one(case: &dyn Case) -> Result<std::path::PathBuf, String> {
    use crate::recording::Validated;
    let level = metal2vulkan_runtime::MetalLevel::NEWEST;
    let answer = match case.metal_source() {
        crate::case::MetalSource::Run => {
            let validated = crate::validated::take(case).map_err(|error| error.to_string())?;
            let mut side = crate::metal_side::MetalSide::new()?;
            crate::record::record(case, &mut side, case.record_runs(), level, validated)
                .map_err(|error| error.to_string())?
        }
        // Never run, and never written from here either: the call this row names is the call
        // Metal does not come back from, so there is nothing for record mode to take. Its
        // recording is written by hand from the probe that died
        // ([`crate::record::never_returned`]).
        crate::case::MetalSource::DidNotReturn { call, says, .. } => {
            return Err(format!(
                "{}: record mode must not run {call}: {says}. Its recording is written by hand \
                 from the probe that died, never by running the shape",
                case.name()
            ))
        }
        crate::case::MetalSource::Carried {
            from, measured_on, ..
        } => {
            let (runs, transcript) = case
                .carried_transcript()?
                .ok_or_else(|| format!("{}: it carries no answer", case.name()))?;
            crate::record::carried(
                level,
                from,
                measured_on,
                crate::recording::ORACLE_CHIP,
                runs,
                transcript,
                Validated::NotRun,
            )
        }
    };
    crate::record::write(case, vec![answer])
}

/// Run one case on native Metal once, announcing each call, and say nothing of what it observed.
///
/// This is the body of the child process [`crate::validated::take`] spawns: the parent reads what
/// this run's Metal said on its standard error. A call that did not answer is not a failure here —
/// the point is what the validation layer printed, whatever the call went on to do.
#[cfg(target_os = "macos")]
pub fn validated_leg(case: &dyn Case) -> Result<(), String> {
    use crate::transcript::Sites;
    let inputs = case.inputs()?;
    let mut side = crate::validated::Announcing::new(crate::metal_side::MetalSide::new()?);
    let mut sites = Sites::new();
    sites.push(crate::case::inputs_node(&inputs));
    let _ = case.run(&mut side, &mut sites);
    Ok(())
}
