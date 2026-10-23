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
    CostRow {
        case: case.name().to_owned(),
        verdict,
        counts,
        nanos,
    }
}

/// Measure what `cases` cost on the venue `venues` names, over `runs` runs of each.
///
/// This is the cost half's own leg: it runs our side alone, with no recording read and no
/// comparison, because a count has no Metal counterpart to compare with. The whole selection runs
/// once, then again, so a count that moves between two runs of the same case on the same venue is
/// seen to move rather than assumed deterministic.
///
/// The runs are of the **whole selection in order**, not of one case repeatedly: the runtime pools
/// command pools, memory blocks and descriptor sets across cases, so a case's counts depend on
/// what ran before it, and a measurement must have the shape the gate will have.
pub fn cost_on_under(
    venues: Venues,
    policy: Policy,
    cases: &[Box<dyn Case>],
    runs: u32,
) -> Result<(String, Vec<CaseCost>), String> {
    let notices = Notices::default();
    let cache = ScratchCache::new().map_err(|error| error.to_string())?;
    let device = device::open_under(venues, policy, &notices, &cache)?;
    notices.take();
    let mut held: Vec<CaseCost> = Vec::new();
    for round in 0..runs.max(1) {
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
    Ok((device.name().to_owned(), held))
}

/// Write what `measured` says each case cost on `venue` into the cases' own recordings, answering
/// where each went or why it could not be written.
///
/// A case whose run stopped is **not** recorded: its counts are of a part of the case.
pub fn record_costs(
    venue: &str,
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
                record_one_cost(venue, cost, words)
            };
            (cost.case.clone(), written)
        })
        .collect()
}

/// Put one case's cost record in its recording, replacing the venue's own record and keeping every
/// other venue's.
fn record_one_cost(
    venue: &str,
    cost: &CaseCost,
    words: &str,
) -> Result<std::path::PathBuf, String> {
    let mut recording = crate::recording::read(&cost.case)?;
    let record = cost.record(venue, words.to_owned());
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
