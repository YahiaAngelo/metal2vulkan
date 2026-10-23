//! What comparing one case on one venue says.
//!
//! The verdicts are kept apart on purpose. A refusal is never buried in a failure count, because
//! the owner's rule is that a refusal of something Metal does is a bug of its own kind; a case with
//! no Metal answer is counted rather than skipped; and a device that honestly lacks a feature the
//! case needs is neither a pass nor a failure, because conflating the two makes coverage fiction.

use crate::compare::Comparison;
use crate::diff::Divergence;
use crate::recording::Validated;
use metal2vulkan::error::{Needs, Unsupported};
use metal2vulkan_runtime::Error;

/// What one case's comparison on one venue says.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// Every field matched, and the case used no allowance.
    Exact,
    /// Fields differed only inside allowances that hold, and the notices they name were
    /// announced.
    Announced { fields: Vec<String> },
    /// A field differs with nothing to justify it: the bug class.
    Divergent { found: Vec<Divergence> },
    /// We returned a typed error where Metal answered the call.
    ///
    /// **Never compliant, and no allowance can change that.** Plain Metal is the behavioural
    /// oracle, because that is how a consumer runs the runtime, and the owner's standing rule is
    /// that what Metal allows we must allow; so a refusal of something plain Metal does is a gap
    /// of ours. A case may name what documents the refusal — Metal's own API validation refusing
    /// the same call is the one thing that can — and that is printed beside the row, but the row
    /// still counts here. Metal's validation stopping a call is not Metal refusing it.
    Refused {
        /// The sites we refused, as their paths and our errors.
        at: Vec<String>,
        /// What the recording's second oracle says of the same case
        /// ([`Validated::describe`]): a refusal Metal's own validation also refuses is a
        /// different fact from one its validation let through, and both differ again from a case
        /// whose validated leg was never taken. The refusal is a failure in all three; what
        /// changes is what a reader should do about it.
        metal_validation: String,
    },
    /// **Metal did not return from the call, and we gave a typed error there.**
    ///
    /// Its own verdict, because neither of the others fits and squeezing it into one would say
    /// something false. `Exact` would claim a value matched where Metal produced none. `Divergent`
    /// would claim a behaviour of Metal's to differ from where the call ended Metal's process.
    /// `Refused` is for a call plain Metal **answered**, which is the gap the owner's rule is
    /// about, and counting this there would bury a real gap among rows that are not one.
    ///
    /// It is **compliant**, and it loosens nothing: no allowance can reach it
    /// ([`crate::compare::NeverReturned`]), the recording is written by hand from a probe that died
    /// rather than by running the shape, and record mode refuses the row. Answering where Metal
    /// did not return is a `Divergent` row, not this one, so the verdict can still fail.
    MetalNeverReturned {
        /// The sites, as their paths and our errors.
        at: Vec<String>,
        /// What the row still loosened, where it loosened anything: a difference inside a
        /// justified allowance at another site is reported here rather than lost, because a row
        /// does not stop being a row about tolerances just because one of its calls killed Metal.
        justified: Vec<String>,
    },
    /// No Metal answer stands for this run: none was recorded for the venue's Metal level, or the
    /// inputs the run fed are not the ones the recording's answer was taken from. Counted, never
    /// skipped and never a pass.
    Unrecorded { why: String },
    /// The venue honestly lacks what the case needs, by the typed refusal it gave: neither a pass
    /// nor a failure.
    LacksFeature { needs: String },
}

impl Verdict {
    /// The verdict `found` says, with `validated` what the recording holds of Metal's validated
    /// run.
    pub fn of(found: &Comparison, validated: &Validated) -> Self {
        if !found.divergences.is_empty() || !found.bad_allowances.is_empty() {
            let mut diverged = found.divergences.clone();
            for bad in &found.bad_allowances {
                diverged.push(Divergence::new(
                    bad.path.clone(),
                    crate::diff::DivergenceKind::Unjustified {
                        why: bad.why.to_string(),
                    },
                ));
            }
            return Verdict::Divergent { found: diverged };
        }
        if !found.refusals.is_empty() {
            return Verdict::Refused {
                at: found
                    .refusals
                    .iter()
                    .map(|refusal| match &refusal.documented_by {
                        Some(documented) => format!(
                            "{}: {} (documented, not excused, by {documented})",
                            refusal.path, refusal.ours
                        ),
                        None => format!("{}: {}", refusal.path, refusal.ours),
                    })
                    .collect(),
                metal_validation: validated.describe(),
            };
        }
        // After the refusals, because a refusal of something plain Metal answered is a gap of
        // ours and must not be hidden behind a call Metal died on; before the announced and exact
        // verdicts, because a row where Metal went away is not a row where everything matched.
        if !found.never_returned.is_empty() {
            return Verdict::MetalNeverReturned {
                at: found
                    .never_returned
                    .iter()
                    .map(|site| format!("{}: {}", site.path, site.ours))
                    .collect(),
                justified: announced_fields(found),
            };
        }
        if found.announced.is_empty() {
            return Verdict::Exact;
        }
        Verdict::Announced {
            fields: announced_fields(found),
        }
    }

    /// The word the report's matrix prints.
    pub fn word(&self) -> &'static str {
        match self {
            Verdict::Exact => "exact",
            Verdict::Announced { .. } => "announced",
            Verdict::Divergent { .. } => "divergent",
            Verdict::Refused { .. } => "refused",
            Verdict::MetalNeverReturned { .. } => "never-returned",
            Verdict::Unrecorded { .. } => "unrecorded",
            Verdict::LacksFeature { .. } => "lacks-feature",
        }
    }

    /// Whether the verdict is one a compliant run may hold: exact, announced inside a justified
    /// allowance, or a venue that lacks the feature. A divergence, a refusal and an unrecorded
    /// case are not.
    pub fn is_compliant(&self) -> bool {
        matches!(
            self,
            Verdict::Exact
                | Verdict::Announced { .. }
                | Verdict::MetalNeverReturned { .. }
                | Verdict::LacksFeature { .. }
        )
    }
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::Exact => out.write_str("exact"),
            Verdict::Announced { fields } => {
                write!(out, "announced: {}", fields.join("; "))
            }
            Verdict::Divergent { found } => {
                out.write_str("divergent")?;
                for divergence in found {
                    write!(out, "\n  {divergence}")?;
                }
                Ok(())
            }
            Verdict::Refused {
                at,
                metal_validation,
            } => {
                out.write_str("refused")?;
                for site in at {
                    write!(out, "\n  {site}")?;
                }
                write!(out, "\n  plain Metal answered the call")?;
                write!(out, "\n  Metal's API validation: {metal_validation}")
            }
            Verdict::MetalNeverReturned { at, justified } => {
                out.write_str("Metal did not return from the call, and we refused it")?;
                for site in at {
                    write!(out, "\n  {site}")?;
                }
                for field in justified {
                    write!(out, "\n  and differed, justified: {field}")?;
                }
                Ok(())
            }
            Verdict::Unrecorded { why } => write!(out, "unrecorded: {why}"),
            Verdict::LacksFeature { needs } => write!(out, "lacks the device feature: {needs}"),
        }
    }
}

/// How the report prints each difference a run made inside a justified allowance.
fn announced_fields(found: &Comparison) -> Vec<String> {
    found
        .announced
        .iter()
        .map(|announced| {
            let largest = match announced.largest {
                Some(largest) => format!(", the largest difference {largest}"),
                None => String::new(),
            };
            format!(
                "{} x{} ({}{largest})",
                announced.path, announced.count, announced.justified_by
            )
        })
        .collect()
}

/// Whether `error` is a device's honest gap rather than ours: the typed refusal names capabilities
/// a device could enable, or a measured driver defect. A refusal the translator or the runtime has
/// not built, and one Metal itself gives, is ours to answer for.
///
/// Two variants carry such a refusal, not one. `Error::Unsupported` is the plain form; a dispatch
/// or draw whose shape the device cannot take is refused by the **bind stage**, which wraps the
/// same typed `Unsupported` in `Error::Bind`, and that is where a device's threadgroup and
/// threadgroup-memory limits are met. Reading only the plain form made a device limit read as a
/// divergence or a refusal of ours, which is the one thing the `LacksFeature` verdict exists to
/// prevent (verified on lavapipe, whose emulated 32-lane simdgroups refuse a tile threadgroup of
/// 1024 threads that Metal runs). What the bind stage has **not built** — `Needs::NotImplemented`
/// and the rest — is still ours to answer for, by the same test as the plain form.
pub fn device_gap(error: &Error) -> Option<String> {
    match error {
        Error::Unsupported(unsupported) => device_gap_of(unsupported),
        Error::Bind(metal2vulkan::error::BindError::Unsupported(unsupported)) => {
            device_gap_of(unsupported)
        }
        _ => None,
    }
}

/// [`device_gap`] of a typed `Unsupported`.
pub fn device_gap_of(unsupported: &Unsupported) -> Option<String> {
    match &unsupported.needs {
        Needs::AnyOf(alternatives) => Some(format!(
            "{:?} needs any of {alternatives:?}",
            unsupported.feature
        )),
        Needs::DriverDefect(workaround) => Some(format!(
            "{:?} on a driver measured to run it wrong ({workaround:?})",
            unsupported.feature
        )),
        Needs::RuntimeDriverDefect(defect) => Some(format!(
            "{:?} on a driver measured to run it wrong ({defect:?})",
            unsupported.feature
        )),
        // Everything else is ours to answer for: a form the translator or the runtime has not
        // built, and one Metal itself refuses, are not the device's gap.
        _ => None,
    }
}
