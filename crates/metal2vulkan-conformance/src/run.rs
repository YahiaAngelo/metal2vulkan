//! Running the suite: record a selection's Metal answers, and replay a selection on a venue.
//!
//! One function per leg, so the command, a test and a device lane all drive the same code.

use crate::case::Case;
use crate::device::{self, Notices, ScratchCache, Venues};
use crate::replay::replay;
use crate::report::Report;
use crate::runtime_side::RuntimeSide;

/// The runs record mode takes a case's answer over, so a case whose behaviour varies records its
/// outcomes with their frequencies rather than whichever run came last.
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

/// Replay `cases` on the venue `venues` names, as a report.
pub fn replay_on(venues: Venues, cases: &[Box<dyn Case>]) -> Result<Report, String> {
    let notices = Notices::default();
    let cache = ScratchCache::new().map_err(|error| error.to_string())?;
    let device = device::open(venues, &notices, &cache)?;
    // The notices a device announces as it is made are the venue's, not any case's: they say what
    // this host's driver and release cost, once, before a case runs. They go in the report's venue
    // line, and the sink is cleared so a case's own notices site holds only what its work
    // announced.
    let mut report = Report::new();
    report.venue_announced(&device.name().to_owned(), notices.take());
    for case in cases {
        let taker = notices.clone();
        let mut side = RuntimeSide::new(&device, move || taker.take());
        report.add_result(replay(case.as_ref(), &mut side));
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
#[cfg(target_os = "macos")]
fn record_one(case: &dyn Case) -> Result<std::path::PathBuf, String> {
    use crate::recording::Validated;
    let level = metal2vulkan_runtime::MetalLevel::NEWEST;
    let answer = match case.metal_source() {
        crate::case::MetalSource::Run => {
            let mut side = crate::metal_side::MetalSide::new()?;
            crate::record::record(case, &mut side, RECORD_RUNS, level, Validated::NotRun)
                .map_err(|error| error.to_string())?
        }
        crate::case::MetalSource::Carried { from, .. } => {
            let (runs, transcript) = case
                .carried_transcript()?
                .ok_or_else(|| format!("{}: it carries no answer", case.name()))?;
            crate::record::carried(
                level,
                from,
                crate::recording::ORACLE_CHIP,
                runs,
                transcript,
                Validated::NotRun,
            )
        }
    };
    crate::record::write(case, vec![answer])
}
