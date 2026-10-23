//! Replay: read a case's recording, run our side, compare, and say what it was.
//!
//! Nothing here needs Metal. The recording is a file, the shaders are committed fixtures, and the
//! device is whatever Vulkan the host has, so the whole leg runs on a Linux host with lavapipe as
//! it runs on this Mac with MoltenVK.

use crate::allowance::{Justification, Offered};
use crate::case::{inputs_node, Case, MetalSource, Side, Venue, INPUTS_SITE};
use crate::compare::{compare_run, Unfired};
use crate::recording::{self, Answer, Lookup, Recording};
use crate::transcript::{Node, Sites, Transcript};
use crate::verdict::Verdict;
use metal2vulkan_runtime::MetalLevel;

/// The reserved site replay emits the degrade notices our run announced at. Metal announces none,
/// so a recording holds an empty set here and any notice of ours must be justified by naming its
/// code: the rule the suite exists for, checked by execution.
pub const NOTICES_SITE: &str = "notices";

/// What replaying one case on one venue came to.
#[derive(Clone, Debug, PartialEq)]
pub struct Replayed {
    pub case: String,
    pub venue: Venue,
    pub verdict: Verdict,
    /// How many allowances the case named, and which of them nothing on this run needed
    /// ([`crate::compare::Unfired`]). Empty where no comparison happened at all — a venue that
    /// lacks the feature fires nothing, and saying its allowances went unused would be noise.
    pub allowances: Allowances,
    /// The chip a carried answer was measured on, where that is not the oracle's
    /// ([`crate::recording::Provenance::measured_elsewhere`]). The report counts these: an answer
    /// carried from a machine that is gone is weaker evidence than one taken here, and the suite's
    /// own rule is that a recording of another oracle is stale, so the one place that rule is
    /// relaxed has to be visible rather than implied.
    pub measured_on: Option<String>,
    /// What our side observed, for a reader who wants the whole transcript after a failure.
    pub ours: Transcript,
}

/// What a case's allowances did on one row.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Allowances {
    /// How many the case named.
    pub named: usize,
    /// The ones no site of this run needed.
    pub unfired: Vec<Unfired>,
}

impl Allowances {
    /// How many fired.
    pub fn fired(&self) -> usize {
        self.named.saturating_sub(self.unfired.len())
    }
}

/// Why a case could not be replayed at all. These are never verdicts: a stale recording must fail
/// loudly rather than resolve to a pass or to a difference nobody can read.
#[derive(Clone, Debug, PartialEq)]
pub enum ReplayError {
    /// The recording file could not be read.
    NoRecording { case: String, why: String },
    /// The recording is not current evidence.
    Stale { case: String, why: String },
    /// The case could not state its own inputs or shaders.
    Case { case: String, why: String },
    /// The venue answers at a Metal level no release of this runtime names.
    UnknownMetalLevel { case: String, level: String },
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplayError::NoRecording { case, why } => {
                write!(out, "{case}: no recording ({why})")
            }
            ReplayError::Stale { case, why } => write!(out, "{case}: a stale recording: {why}"),
            ReplayError::Case { case, why } => write!(out, "{case}: {why}"),
            ReplayError::UnknownMetalLevel { case, level } => {
                write!(
                    out,
                    "{case}: the venue answers at an unknown Metal level {level}"
                )
            }
        }
    }
}

/// Run `case` on `side` and compare it with its recording.
pub fn replay(case: &dyn Case, side: &mut dyn Side) -> Result<Replayed, ReplayError> {
    replay_known(case, side, &[])
}

/// [`replay`] where `device_notices` are the codes this device has already announced earlier in
/// the run.
pub fn replay_known(
    case: &dyn Case,
    side: &mut dyn Side,
    device_notices: &[String],
) -> Result<Replayed, ReplayError> {
    let name = case.name();
    let recording = recording::read(name).map_err(|why| ReplayError::NoRecording {
        case: name.to_owned(),
        why,
    })?;
    replay_against_known(case, side, &recording, device_notices)
}

/// [`replay`] against a recording already in hand.
pub fn replay_against(
    case: &dyn Case,
    side: &mut dyn Side,
    recording: &Recording,
) -> Result<Replayed, ReplayError> {
    replay_against_known(case, side, recording, &[])
}

/// [`replay_against`] where `device_notices` are the codes this device has already announced
/// earlier in the run.
///
/// A degrade notice is announced **once per device per kind**, so a case that runs after the one
/// that first provoked a notice sees none of it. Its own notices site still holds only what its
/// own work announced — that is what makes the site readable — but the device's notice is a fact
/// about the device for the whole run, so it justifies a loosened field in any case of that run.
/// Without this a case could only name a notice it happened to provoke first, which the order of
/// the case set decides.
pub fn replay_against_known(
    case: &dyn Case,
    side: &mut dyn Side,
    recording: &Recording,
    device_notices: &[String],
) -> Result<Replayed, ReplayError> {
    let name = case.name();
    let venue = side.venue();
    let level = MetalLevel::ALL
        .into_iter()
        .find(|level| level.macos() == venue.metal_level)
        .ok_or_else(|| ReplayError::UnknownMetalLevel {
            case: name.to_owned(),
            level: venue.metal_level.clone(),
        })?;
    let inputs = case.inputs().map_err(|why| ReplayError::Case {
        case: name.to_owned(),
        why,
    })?;

    let mut sites = Sites::new();
    sites.push(inputs_node(&inputs));
    let abort = case.run(side, &mut sites).err();
    let mut announced = side.announced();
    sites.notices(NOTICES_SITE, &announced);
    let ours = sites.finish();
    announced.extend(ours.announced_notices().into_keys());
    announced.extend(device_notices.iter().cloned());

    let answer = match recording.answer(level) {
        Ok(answer) => answer,
        Err(Lookup::Unrecorded { level }) => {
            return Ok(Replayed {
                case: name.to_owned(),
                venue,
                verdict: Verdict::Unrecorded {
                    why: format!("no Metal answer at Metal level {level}"),
                },
                allowances: Allowances::default(),
                measured_on: None,
                ours,
            })
        }
        Err(stale @ Lookup::Stale(_)) => {
            return Err(ReplayError::Stale {
                case: name.to_owned(),
                why: stale.to_string(),
            })
        }
    };

    // A run that stopped because the venue honestly lacks what the case needs is its own outcome:
    // never a pass, never a failure, and never counted as coverage.
    if let Some(lacks) = abort.as_ref().and_then(|abort| abort.failure.lacks.clone()) {
        return Ok(Replayed {
            case: name.to_owned(),
            venue,
            verdict: Verdict::LacksFeature { needs: lacks },
            allowances: Allowances::default(),
            measured_on: answer.provenance.measured_elsewhere().map(str::to_owned),
            ours,
        });
    }

    let (verdict, allowances) = verdict_over(case, answer, &ours, &announced, abort.is_some());
    Ok(Replayed {
        case: name.to_owned(),
        venue,
        verdict,
        allowances,
        measured_on: answer.provenance.measured_elsewhere().map(str::to_owned),
        ours,
    })
}

/// The verdict of `ours` against `answer`'s outcomes: the most favourable of them, because Metal
/// produced any of the set and matching one of them is matching Metal. With it, what the case's
/// allowances did against that outcome.
///
/// The allowances follow the verdict rather than being gathered over every outcome, because they
/// are a fact about the comparison the verdict came from: a racy case whose run matched the
/// outcome that needed no tolerance did not need the tolerance, however the other outcomes read.
fn verdict_over(
    case: &dyn Case,
    answer: &Answer,
    ours: &Transcript,
    announced: &[String],
    stopped: bool,
) -> (Verdict, Allowances) {
    let allowances = case.allowances();
    let carried = match case.metal_source() {
        MetalSource::Carried { from, .. } => Some(from),
        MetalSource::Run | MetalSource::DidNotReturn { .. } => None,
    };
    let mut best: Option<(Verdict, Allowances)> = None;
    for outcome in &answer.outcomes {
        // The inputs come first: an answer taken from other bytes is no answer for this run.
        if let Some(why) = inputs_differ(&outcome.transcript, ours) {
            best = Some(better(
                best,
                (
                    Verdict::Unrecorded {
                        why: match carried {
                            Some(from) => format!("{why}; the answer is carried from {from}"),
                            None => why,
                        },
                    },
                    Allowances::default(),
                ),
            ));
            continue;
        }
        let offered = Offered::new(announced, &answer.validated);
        let found = compare_run(&outcome.transcript, ours, &allowances, offered, stopped);
        let used = Allowances {
            named: found.named,
            unfired: found.unfired.clone(),
        };
        best = Some(better(best, (Verdict::of(&found, &answer.validated), used)));
    }
    best.unwrap_or_else(|| {
        (
            Verdict::Unrecorded {
                why: "the recording's answer holds no outcome".to_owned(),
            },
            Allowances::default(),
        )
    })
}

/// Why the inputs our run fed are not the ones `metal`'s answer was taken from, if they are not.
fn inputs_differ(metal: &Transcript, ours: &Transcript) -> Option<String> {
    let recorded = metal.root.child(INPUTS_SITE);
    let fed = ours.root.child(INPUTS_SITE);
    match (recorded, fed) {
        (Some(recorded), Some(fed)) if recorded == fed => None,
        (None, _) => Some("the recording states no input fingerprint".to_owned()),
        (Some(recorded), Some(fed)) => Some(format!(
            "the inputs this run fed are not the recording's: {}",
            input_diff(recorded, fed)
        )),
        (Some(_), None) => Some("this run stated no input fingerprint".to_owned()),
    }
}

/// Which inputs moved between two input fingerprints, in words.
fn input_diff(recorded: &Node, fed: &Node) -> String {
    let mut moved = Vec::new();
    for input in &recorded.children {
        match fed.child(&input.name) {
            Some(ours) if ours.observed == input.observed => {}
            Some(_) => moved.push(format!("{} differs", input.name)),
            None => moved.push(format!("{} is missing", input.name)),
        }
    }
    for input in &fed.children {
        if recorded.child(&input.name).is_none() {
            moved.push(format!("{} is extra", input.name));
        }
    }
    moved.join(", ")
}

/// The more favourable of two verdicts, with the allowances of whichever is kept.
fn better(
    held: Option<(Verdict, Allowances)>,
    found: (Verdict, Allowances),
) -> (Verdict, Allowances) {
    match held {
        Some(held) if rank(&held.0) <= rank(&found.0) => held,
        _ => found,
    }
}

/// How favourable a verdict is, lowest first.
fn rank(verdict: &Verdict) -> u8 {
    match verdict {
        Verdict::Exact => 0,
        Verdict::Announced { .. } => 1,
        Verdict::MetalNeverReturned { .. } => 2,
        Verdict::LacksFeature { .. } => 3,
        Verdict::Refused { .. } => 4,
        Verdict::Divergent { .. } => 5,
        Verdict::Unrecorded { .. } => 6,
    }
}

/// The allowance a case needs to announce `code`, for a case whose own degrade notice is the
/// difference: the notice site, loosened by the code it names.
pub fn notice_allowance(code: &str) -> crate::allowance::Allowance {
    crate::allowance::Allowance::new(
        format!("case/{NOTICES_SITE}/*"),
        crate::allowance::Loosening::Unchecked,
        Justification::Degradation(code.to_owned()),
    )
}
