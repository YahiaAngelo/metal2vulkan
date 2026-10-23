//! Replay: read a case's recording, run our side, compare, and say what it was.
//!
//! Nothing here needs Metal. The recording is a file, the shaders are committed fixtures, and the
//! device is whatever Vulkan the host has, so the whole leg runs on a Linux host with lavapipe as
//! it runs on this Mac with MoltenVK.

use crate::allowance::Justification;
use crate::case::{inputs_node, Case, MetalSource, Side, Venue, INPUTS_SITE};
use crate::compare::compare_run;
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
    /// What our side observed, for a reader who wants the whole transcript after a failure.
    pub ours: Transcript,
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
    let name = case.name();
    let recording = recording::read(name).map_err(|why| ReplayError::NoRecording {
        case: name.to_owned(),
        why,
    })?;
    replay_against(case, side, &recording)
}

/// [`replay`] against a recording already in hand.
pub fn replay_against(
    case: &dyn Case,
    side: &mut dyn Side,
    recording: &Recording,
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

    let answer = match recording.answer(level) {
        Ok(answer) => answer,
        Err(Lookup::Unrecorded { level }) => {
            return Ok(Replayed {
                case: name.to_owned(),
                venue,
                verdict: Verdict::Unrecorded {
                    why: format!("no Metal answer at Metal level {level}"),
                },
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
            ours,
        });
    }

    let verdict = verdict_over(case, answer, &ours, &announced, abort.is_some());
    Ok(Replayed {
        case: name.to_owned(),
        venue,
        verdict,
        ours,
    })
}

/// The verdict of `ours` against `answer`'s outcomes: the most favourable of them, because Metal
/// produced any of the set and matching one of them is matching Metal.
fn verdict_over(
    case: &dyn Case,
    answer: &Answer,
    ours: &Transcript,
    announced: &[String],
    stopped: bool,
) -> Verdict {
    let allowances = case.allowances();
    let validation = answer.validated.stopped();
    let carried = match case.metal_source() {
        MetalSource::Carried { from, .. } => Some(from),
        MetalSource::Run => None,
    };
    let mut best: Option<Verdict> = None;
    for outcome in &answer.outcomes {
        // The inputs come first: an answer taken from other bytes is no answer for this run.
        if let Some(why) = inputs_differ(&outcome.transcript, ours) {
            best = Some(better(
                best,
                Verdict::Unrecorded {
                    why: match carried {
                        Some(from) => format!("{why}; the answer is carried from {from}"),
                        None => why,
                    },
                },
            ));
            continue;
        }
        let found = compare_run(&outcome.transcript, ours, &allowances, announced, stopped);
        best = Some(better(best, Verdict::of(&found, validation.clone())));
    }
    best.unwrap_or(Verdict::Unrecorded {
        why: "the recording's answer holds no outcome".to_owned(),
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

/// The more favourable of two verdicts.
fn better(held: Option<Verdict>, found: Verdict) -> Verdict {
    match held {
        Some(held) if rank(&held) <= rank(&found) => held,
        _ => found,
    }
}

/// How favourable a verdict is, lowest first.
fn rank(verdict: &Verdict) -> u8 {
    match verdict {
        Verdict::Exact => 0,
        Verdict::Announced { .. } => 1,
        Verdict::LacksFeature { .. } => 2,
        Verdict::Refused { .. } => 3,
        Verdict::Divergent { .. } => 4,
        Verdict::Unrecorded { .. } => 5,
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
