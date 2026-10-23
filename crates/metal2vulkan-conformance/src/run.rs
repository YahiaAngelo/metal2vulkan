//! Running the suite: record a selection's Metal answers, and replay a selection on a venue.
//!
//! One function per leg, so the command, a test and a device lane all drive the same code.

use crate::case::Case;
use crate::device::{self, Notices, Policy, ScratchCache, Venues};
use crate::replay::replay_known;
use crate::report::Report;
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
    for case in cases {
        let taker = notices.clone();
        let mut side = RuntimeSide::new(&device, move || taker.take());
        let replayed = replay_known(case.as_ref(), &mut side, &announced_on_device);
        if let Ok(replayed) = &replayed {
            for code in replayed.ours.announced_notices().into_keys() {
                if !announced_on_device.contains(&code) {
                    announced_on_device.push(code);
                }
            }
        }
        report.add_result(replayed);
    }
    Ok(report)
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
