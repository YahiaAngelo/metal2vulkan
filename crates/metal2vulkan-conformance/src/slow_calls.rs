//! The per-call cost gate, and the ledger of calls we already know are over it.
//!
//! # Why the gate is per call and not per case
//!
//! The per-case margin is the quantity nothing can dodge by moving work between the three
//! attributions, and it is **insensitive**. Measured over the whole set: no case exceeds 3.9x, and
//! the largest absolute gap of a comparable case is 2.6x — while one *call*,
//! `replace_region`, runs at 265x on the 16 cases that make it, because a case's total dilutes one
//! expensive call. A case-level margin could never have fired on that, which is the argument for
//! this module and not for a tighter margin.
//!
//! So the gate holds each **paired call row** — summed over the cases both legs did exactly
//! ([`crate::times::paired`]) — to [`crate::times::MARGIN`], an order of magnitude. That margin is
//! kept rather than tightened on measured grounds: over two whole-set runs of one binary, Metal's
//! own per-call totals moved a median 1.28x and as much as 2.43x, so a 2x margin would fire on the
//! machine being busy while a 10x margin did not move a verdict.
//!
//! # Why this is a second file and not a row in the known-gaps ledger
//!
//! [`crate::known_gaps`] holds refusals and divergences in **one** file on an explicit argument:
//! `AGENTS.md` forbids parallel string lists, and two ledgers for two kinds of correctness verdict
//! would be exactly that. That argument has to be answered here rather than ignored, because this
//! is a third file of the same shape.
//!
//! It is answered by the line the cost half already draws. A gap row is a **correctness** verdict
//! on a case at a site, held by `replay`, which needs no Metal. A row here is a **cost** result on
//! a call, held by `times`, which needs both legs on one machine. The counters half made the same
//! split deliberately and for the same reason — its `CostVerdict` is `Within`/`Over`/`Unmeasured`
//! rather than a new `Verdict` variant, because a count is ours and not Metal's and must never
//! touch a correctness verdict. Folding a ratio and a band into a file whose rows name a verdict
//! and a venue would put cost inside the correctness vocabulary, which is the thing both halves
//! have been careful not to do.
//!
//! What it does share is the mechanism, and that is on purpose: a known gap that makes every run
//! red teaches everyone to stop reading the report, which costs us the next real one.
//!
//! **If the round would rather have one file, this folds into it**: the row gains a verdict kind
//! and a ratio, and `held` moves beside `known_gaps::held`. Nothing here depends on being separate
//! except the argument above.
//!
//! # The ledger is an expectation, not an allowlist, and it is held in both directions
//!
//! - **A call over the margin that the ledger does not list fails the run** ([`Expectation::Unlisted`]).
//! - **A listed call that has come back inside the band also fails the run**
//!   ([`Expectation::NoLongerSlow`]), saying the row looks fixed and must go.
//!
//! That second direction is the point: closing a gap forces the ledger edit, so a row cannot
//! outlive the bug it stands for.
//!
//! # The band, and why a ratio needs one where a refusal does not
//!
//! A refusal is a yes or a no, so `known-refusals.tsv` needs no tolerance. A ratio is a
//! measurement, and this one moves: the 2.43x worst movement above means a row measured at 11x can
//! read 5x on a quiet run through nothing changing. Holding it to "still over 10x" would make the
//! `NoLongerSlow` direction fire on noise.
//!
//! So a listed row is reported, not failed, until it falls under [`band`] — `MARGIN / 2.5`, where
//! 2.5 is above the measured 2.43x. Between the band and the margin a listed row is neither a pass
//! nor a failure, and the report says so in those words rather than picking a side a single run
//! cannot support.
//!
//! # What the gate deliberately cannot see
//!
//! It is an **upper bound on our own cost**, so a row where we are *faster* than Metal never fires
//! it. That is the right asymmetry here: being cheaper than Metal for a reason the suite has not
//! explained is not something a gate should bless — `library` reads 0.005x because of shader reuse
//! and a cache below us, not because of a compiler win ([`crate::run::Shaders`]) — and a gate that
//! credited it would turn an unexplained setup difference into a green light.

/// The committed ledger, compiled in so the check needs no path and no filesystem.
const LEDGER: &str = include_str!("../known-slow-calls.tsv");

/// Where a listed row stops being reported and starts failing as fixed: [`crate::times::MARGIN`]
/// divided by this.
///
/// 2.5, which is above the 2.43x worst run-to-run movement measured over two whole-set runs of one
/// binary on this Mac. A smaller number would make a row's `NoLongerSlow` direction fire on a quiet
/// machine.
pub const BAND_DIVISOR: f64 = 2.5;

/// The ratio a listed row must fall under before the run reads it as fixed.
pub fn band() -> f64 {
    crate::times::MARGIN as f64 / BAND_DIVISOR
}

/// **Why a cold measurement is reported and never judged.** `--cold-shaders` gives each case a
/// device and a shader cache of its own, so it changes two things at once and only on our leg: no
/// case is handed another's compile, and every per-device first-use cost lands inside whichever
/// call touches a resource first. The whole-set numbers say how much of the second there is --
/// warm against cold on this M3 Max, `target` goes 0.9x to 14.3x, `dispatch` 1.0x to 17.7x and
/// `copy_rate_map_parameters` 0.4x to 26.4x, none of which compiles anything. A cold reading over
/// the margin therefore does not say which of the two it is, and a gate wants a reading that
/// localises. Nine calls were over the margin in the cold whole-set run against four warm; seven
/// of the nine would have been one row each for one mechanism, which is the ledger its own header
/// warns about.
pub const WHY_COLD_IS_NOT_JUDGED: &str =
    "the margin is reported and not judged for a cold measurement: --cold-shaders also gives \
     every case\na fresh device, so a reading over the margin does not separate our compile \
     cost from the cost of\nfirst use on a new device -- measured, `target` is 0.9x warm and \
     14.3x cold without compiling anything.\nRun it without --cold-shaders for the gate.";

/// **The ceiling on host load, measured.**
///
/// This was 4.0, set conservatively and labelled at the time as stated rather than localised.
/// It has since been measured, and the measurement says absolute load is very nearly the wrong
/// variable: over twenty passes of one twelve-case selection on this M3 Max, spanning a load
/// average of 8.59 to 26.45 (3.08x), `render_pipeline` stayed between 51.6x and 56.6x -- a spread
/// of **1.10x** -- and `compute_pipeline` between 29.3x and 33.1x, a spread of 1.13x. There is no
/// knee and no drift worth a gate: against a 10x margin a 1.1x wobble is nothing.
///
/// One confound is declared rather than hidden: load fell monotonically across those passes, so
/// load and pass order are collinear and the small residual correlation (+0.64 with load, -0.59
/// with order) cannot be attributed to either. It does not matter for the bound, because the
/// *effect size* is about 10% whichever variable owns it.
///
/// So the bound is the top of the range that was actually measured, and nothing above it is
/// claimed. What does corrupt a ratio is [`LOAD_MAY_MOVE`]: the legs run one after the other, so
/// load moving *between* them compares two machines.
pub const QUIET_ENOUGH: f64 = 26.0;

/// **How far the load may move across one measurement before its two halves are not one
/// condition.** A factor, applied to the larger reading over the smaller.
///
/// **This is the guard that the evidence actually supports**, and the mechanism is the leg order:
/// `times` measures Metal's leg to completion and then ours, deliberately, because interleaving
/// would put each leg's threads in the other's off-thread CPU reading. So the two legs occupy two
/// different windows -- about six seconds apart in a twelve-case pass, but minutes apart in a
/// whole-set pass -- and load moving between those windows divides one machine's Metal figure by
/// another machine's figure of ours. That, rather than the absolute level, is what produced the
/// 4.5x-against-1.6x contrast between two whole-set runs, and it is why a twelve-case pass is
/// flat across a 3x load range while an eleven-minute one is not.
///
/// The factor itself is still stated rather than localised. A better instrument is to bracket
/// *each leg* and compare the two legs' own loads, which measures the drift directly instead of
/// through the whole run's endpoints; that is the next change, not this one.
pub const LOAD_MAY_MOVE: f64 = 2.0;

/// What the host was doing while one leg ran: its one-minute load average on each side of that
/// leg, and nothing from the other leg's window.
#[derive(Clone, Copy, Debug, Default)]
pub struct LegLoad {
    /// The reading taken immediately before this leg started.
    pub before: Option<f64>,
    /// The reading taken immediately after it finished.
    pub after: Option<f64>,
}

impl LegLoad {
    /// The load to compare this leg by: the mean of its two readings, or whichever one the host
    /// answered.
    pub fn typical(self) -> Option<f64> {
        match (self.before, self.after) {
            (Some(before), Some(after)) => Some((before + after) / 2.0),
            (Some(one), None) | (None, Some(one)) => Some(one),
            (None, None) => None,
        }
    }

    /// How far the host moved while this one leg ran, as a factor.
    fn moved(self) -> Option<f64> {
        let (before, after) = (self.before?, self.after?);
        let (low, high) = if before < after {
            (before, after)
        } else {
            (after, before)
        };
        if low <= 0.0 {
            return None;
        }
        Some(high / low)
    }
}

/// Whether the gate judges a measurement taken under `shaders` with `metal` and `ours` describing
/// the host while each leg ran.
///
/// **The per-leg brackets are the point.** `times` measures Metal's leg to completion and then
/// ours, so the quantity that corrupts a ratio is the host moving *between* those two windows:
/// that divides one machine's Metal figure by another machine's figure of ours. Bracketing the
/// whole run could never see it, because the run's endpoints average over exactly the interval
/// where the damage happens. These brackets measure it directly.
///
/// Refused four ways, each with its reason for the caller to print: a cold reading, which does
/// not separate our compile cost from first use on a new device; any reading over
/// [`QUIET_ENOUGH`]; a single leg whose own load moved by more than [`LOAD_MAY_MOVE`], so that
/// leg is not one measurement; and the two legs' loads differing by more than [`LOAD_MAY_MOVE`],
/// so the ratio is of two machines.
///
/// A host that answers no load average is judged on what it does answer, and judged outright if
/// it answers nothing: a missing reading is not evidence of a busy machine.
pub fn judges(shaders: crate::run::Shaders, metal: LegLoad, ours: LegLoad) -> Result<(), String> {
    if matches!(shaders, crate::run::Shaders::Cold) {
        return Err(WHY_COLD_IS_NOT_JUDGED.to_owned());
    }
    let why_load = |whose: &str, seen: f64| {
        format!(
            "the margin is reported and not judged: this host's one-minute load average was \
             {seen:.2} while\n{whose} ran, and the gate wants it at {QUIET_ENOUGH:.1} or under. \
             Measured on this host, the ratios are\nflat to within 1.1x from a load of 8.6 to \
             26.5, and nothing above that was measured. Re-run it on a\nquieter host for a \
             verdict."
        )
    };
    for (whose, leg) in [("Metal's leg", metal), ("our leg", ours)] {
        if let Some(seen) = leg
            .before
            .into_iter()
            .chain(leg.after)
            .find(|seen| *seen > QUIET_ENOUGH)
        {
            return Err(why_load(whose, seen));
        }
        if let Some(moved) = leg.moved() {
            if moved > LOAD_MAY_MOVE {
                return Err(format!(
                    "the margin is reported and not judged: while {whose} ran, this host's \
                     one-minute load\naverage moved from {:.2} to {:.2}, a factor of {moved:.2} \
                     and more than the {LOAD_MAY_MOVE:.1}x the gate\nallows. That leg is not one \
                     measurement, so neither is the ratio. Re-run it on a host that\nstays put \
                     for a verdict.",
                    leg.before.unwrap_or_default(),
                    leg.after.unwrap_or_default()
                ));
            }
        }
    }
    // The interval that actually corrupts a ratio: between the two legs' windows.
    if let (Some(metal_load), Some(our_load)) = (metal.typical(), ours.typical()) {
        let (low, high) = if metal_load < our_load {
            (metal_load, our_load)
        } else {
            (our_load, metal_load)
        };
        if low > 0.0 && high / low > LOAD_MAY_MOVE {
            return Err(format!(
                "the margin is reported and not judged: Metal's leg ran at a load average of \
                 {metal_load:.2} and\nours at {our_load:.2}, a factor of {:.2} apart and more \
                 than the {LOAD_MAY_MOVE:.1}x the gate allows. The legs\nrun one after the \
                 other, so a ratio across that gap divides one machine's figure by another's.\n\
                 Re-run it on a host that stays put for a verdict.",
                high / low
            ));
        }
    }
    Ok(())
}
/// One call we have written down as costing an order of magnitude more than Metal's.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct KnownSlowCall {
    /// The [`crate::case::Side`] call.
    pub call: String,
    /// The ratio the measurement that justified the row read, as its text: part of the row so a
    /// reader sees what it was when it was written, and so a row cannot be added without one.
    pub measured: String,
    /// Where the cost is, as the row's author found it: the family, and what the figure splits
    /// into.
    pub where_it_is: String,
    /// The gap it waits on: what Metal does, what we do instead, and where the fix belongs.
    pub waiting_on: String,
}

/// The committed rows.
pub fn ledger() -> Result<Vec<KnownSlowCall>, String> {
    parse(LEDGER)
}

/// `text`'s rows: tab-separated, with blank lines and lines from `#` ignored.
pub fn parse(text: &str) -> Result<Vec<KnownSlowCall>, String> {
    let mut rows = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let at = number + 1;
        let fields: Vec<&str> = line.split('\t').collect();
        let [call, measured, where_it_is, waiting_on] = fields.as_slice() else {
            return Err(format!(
                "known-slow-calls.tsv line {at}: {} tab-separated fields, and a row has four: the \
                 call, the ratio its measurement read, where the cost is, and the gap it waits on",
                fields.len()
            ));
        };
        if measured.trim().is_empty() {
            return Err(format!(
                "known-slow-calls.tsv line {at}: {call} names no measurement. Every bound in this \
                 suite names the measurement that justified it, and a ratio with no measurement is \
                 an opinion"
            ));
        }
        if waiting_on.trim().is_empty() {
            return Err(format!(
                "known-slow-calls.tsv line {at}: {call} names no gap. A row is an open bug, so it \
                 says what Metal does, what we do instead, and where the fix belongs"
            ));
        }
        rows.push(KnownSlowCall {
            call: call.trim().to_owned(),
            measured: measured.trim().to_owned(),
            where_it_is: where_it_is.trim().to_owned(),
            waiting_on: waiting_on.trim().to_owned(),
        });
    }
    Ok(rows)
}

/// How one paired call row stands against the margin and the ledger.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expectation {
    /// Over the margin and listed: an open bug, reported and not failing the run.
    Known { call: String, waiting_on: String },
    /// Over the margin and **not** listed: a per-call cost nobody has written down.
    Unlisted { call: String, ratio: String },
    /// Listed, and now under [`band`]: the gap looks closed and the row must go.
    NoLongerSlow { call: String, ratio: String },
    /// Listed, between the band and the margin, where one run cannot say which side it is on.
    InTheBand { call: String, ratio: String },
}

impl Expectation {
    /// Whether it fails the run.
    pub fn fails(&self) -> bool {
        matches!(
            self,
            Expectation::Unlisted { .. } | Expectation::NoLongerSlow { .. }
        )
    }
}

impl std::fmt::Display for Expectation {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expectation::Known { call, waiting_on } => write!(
                out,
                "{call} is over the margin and known-slow-calls.tsv lists it, waiting on \
                 {waiting_on}"
            ),
            Expectation::Unlisted { call, ratio } => write!(
                out,
                "{call} costs {ratio} what Metal's does and known-slow-calls.tsv does not list \
                 it: a per-call cost nobody has written down. Localise it with `times --call \
                 {call}` before adding a row"
            ),
            Expectation::NoLongerSlow { call, ratio } => write!(
                out,
                "{call} now costs {ratio} what Metal's does, inside the band, and the \
                 known-slow-calls ledger still lists it: the gap looks closed, so drop the row"
            ),
            Expectation::InTheBand { call, ratio } => write!(
                out,
                "{call} costs {ratio} what Metal's does, between the band and the margin, where \
                 one run cannot say which side it is on"
            ),
        }
    }
}

/// Hold `rows` to the margin and the ledger.
///
/// A row the measurement could not compare at all is left out: its ratio is not a number, and a
/// call with no Metal figure is a coverage hole rather than a cost result.
///
/// A cold measurement never reaches here: [`judges`] turns the gate off for one, because its
/// readings do not separate our compile cost from the cost of first use on a new device.
pub fn held(rows: &[crate::times::Paired], ledger: &[KnownSlowCall]) -> Vec<Expectation> {
    let margin = crate::times::MARGIN as f64;
    let band = band();
    let mut held = Vec::new();
    for row in rows {
        let Some(ratio) = row.ratio() else {
            continue;
        };
        let listed = ledger.iter().find(|known| known.call == row.call);
        let said = format!("{ratio:.1}x");
        held.push(match (listed, ratio) {
            (Some(known), ratio) if ratio > margin => Expectation::Known {
                call: row.call.clone(),
                waiting_on: known.waiting_on.clone(),
            },
            (Some(_), ratio) if ratio < band => Expectation::NoLongerSlow {
                call: row.call.clone(),
                ratio: said,
            },
            (Some(_), _) => Expectation::InTheBand {
                call: row.call.clone(),
                ratio: said,
            },
            (None, ratio) if ratio > margin => Expectation::Unlisted {
                call: row.call.clone(),
                ratio: said,
            },
            (None, _) => continue,
        });
    }
    held
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::times::{CallTotal, Paired};

    /// A paired row at `ratio` times Metal's cost.
    fn row(call: &str, ratio: u64) -> Paired {
        Paired {
            call: call.to_owned(),
            metal: CallTotal {
                calls: 1,
                wall: 100,
                on_cpu: 100,
                blocked: 0,
                typical: 100,
            },
            ours: CallTotal {
                calls: 1,
                wall: 100 * ratio,
                on_cpu: 100 * ratio,
                blocked: 0,
                typical: 100 * ratio,
            },
        }
    }

    /// The committed ledger parses, and every row names its measurement and its gap.
    #[test]
    fn the_ledger_parses_and_every_row_names_its_evidence() {
        let rows = ledger().unwrap();
        assert!(!rows.is_empty(), "the ledger holds no rows");
        for known in &rows {
            assert!(!known.call.is_empty());
            assert!(
                known.measured.contains('x'),
                "{}: {}",
                known.call,
                known.measured
            );
            assert!(
                known.waiting_on.len() > 20,
                "{}: too thin a gap",
                known.call
            );
        }
    }

    /// A row with a missing measurement, a missing gap or an unknown mode is refused, so a row
    /// cannot be added without its evidence.
    #[test]
    fn a_row_without_its_evidence_is_refused() {
        assert!(
            parse("a\t\twhere\tgap").is_err(),
            "a row with no measurement parsed"
        );
        assert!(parse("a\t3x\twhere\t").is_err(), "a row with no gap parsed");
        assert!(parse("a\t3x\twhere").is_err(), "a three-field row parsed");
        assert!(parse("# just a comment\n\n").unwrap().is_empty());
    }

    /// **A cold measurement is reported and never judged**, because `--cold-shaders` also hands
    /// every case a fresh device and a reading over the margin then does not say which of the two
    /// it is. Measured on this M3 Max, `target` is 0.9x warm and 14.3x cold without compiling
    /// anything.
    /// One leg that held still at `load`.
    fn steady(load: f64) -> LegLoad {
        LegLoad {
            before: Some(load),
            after: Some(load),
        }
    }

    #[test]
    fn the_gate_does_not_judge_a_cold_measurement() {
        assert!(
            judges(crate::run::Shaders::Warm, steady(1.0), steady(1.0)).is_ok(),
            "the warm gate is off on a quiet host"
        );
        let cold = judges(crate::run::Shaders::Cold, steady(1.0), steady(1.0)).expect_err(
            "a cold measurement was judged, and its readings do not separate the two causes",
        );
        assert!(
            cold.contains("--cold-shaders"),
            "the reason does not name the switch that causes it: {cold}"
        );
    }

    /// **A loaded host is reported and never judged**, whichever leg was on it. The bound is the
    /// top of the range the ratios were measured flat across, and nothing above it is claimed.
    #[test]
    fn the_gate_does_not_judge_a_loaded_host() {
        // Either leg over the bound refuses, and the reason names which one.
        let why = judges(
            crate::run::Shaders::Warm,
            steady(QUIET_ENOUGH + 0.01),
            steady(QUIET_ENOUGH),
        )
        .expect_err("a loaded Metal leg was judged");
        assert!(
            why.contains("Metal's leg"),
            "the reason does not name the loaded leg: {why}"
        );
        assert!(
            judges(
                crate::run::Shaders::Warm,
                steady(QUIET_ENOUGH),
                steady(QUIET_ENOUGH + 0.01)
            )
            .is_err(),
            "a loaded leg of ours was judged"
        );
        assert!(
            judges(
                crate::run::Shaders::Warm,
                steady(QUIET_ENOUGH),
                steady(QUIET_ENOUGH)
            )
            .is_ok(),
            "the bound itself is refused, so no host can ever meet it"
        );
        assert!(
            judges(
                crate::run::Shaders::Warm,
                LegLoad::default(),
                LegLoad::default()
            )
            .is_ok(),
            "a host with no load average was refused a verdict for not having one"
        );
    }

    /// **A leg that did not hold still is not one measurement.** Its own two readings disagree, so
    /// the figures it produced are of a machine that changed under it.
    #[test]
    fn the_gate_does_not_judge_a_leg_that_moved_under_itself() {
        let moved = LegLoad {
            before: Some(1.0),
            after: Some(1.0 * LOAD_MAY_MOVE + 1.0),
        };
        // **This is the case the run-level bracket could not see.** Its two readings are the
        // run's first and last -- `moved.before` is 1.0 and `steady(1.0).after` is 1.0 -- so the
        // endpoints are identical and a bracket of the whole run reports a host that never
        // changed, while the load in fact tripled inside Metal's leg.
        assert_eq!(
            moved.before,
            steady(1.0).after,
            "this test no longer pins the case the run-level bracket missed"
        );
        let why = judges(crate::run::Shaders::Warm, moved, steady(1.0))
            .expect_err("a leg whose own load moved was judged");
        assert!(
            why.contains("Metal's leg") && why.contains("moved from"),
            "the reason does not name the leg that moved: {why}"
        );
        assert!(
            judges(crate::run::Shaders::Warm, steady(1.0), moved).is_err(),
            "our leg moving under itself was judged"
        );
    }

    /// **The interval that actually corrupts a ratio is between the legs**, because they run one
    /// after the other: a ratio across that gap divides one machine's Metal figure by another
    /// machine's figure of ours. This is the case bracketing the whole run could never catch,
    /// since the run's endpoints average over exactly that interval. The numbers are the measured
    /// ones: a whole-set pass ran Metal's leg near 16.8 and ours near 55.9.
    #[test]
    fn the_gate_does_not_judge_a_ratio_taken_across_a_load_change() {
        let why = judges(
            crate::run::Shaders::Warm,
            steady(1.0),
            steady(LOAD_MAY_MOVE + 1.0),
        )
        .expect_err("a ratio taken across a load change between the legs was judged");
        assert!(
            why.contains("Metal's leg ran at") && why.contains("ours at"),
            "the reason does not give both legs' loads: {why}"
        );
        // Exactly at the allowance is still one machine.
        assert!(
            judges(
                crate::run::Shaders::Warm,
                steady(1.0),
                steady(LOAD_MAY_MOVE)
            )
            .is_ok(),
            "the allowance itself is refused"
        );
        // An idle host reads near zero on both legs and must not trip the ratio.
        assert!(
            judges(crate::run::Shaders::Warm, steady(0.0), steady(0.0)).is_ok(),
            "an idle host was refused for having no load to compare"
        );
    }

    /// **Both directions.** An unlisted call over the margin fails; a listed call back inside the
    /// band fails too, so a row cannot outlive its bug.
    #[test]
    fn the_gate_is_held_in_both_directions() {
        let known = vec![KnownSlowCall {
            call: "listed".to_owned(),
            measured: "100.0x".to_owned(),
            where_it_is: "somewhere".to_owned(),
            waiting_on: "a gap written down at length somewhere".to_owned(),
        }];
        let over = held(&[row("listed", 100), row("unlisted", 50)], &known);
        assert!(matches!(&over[0], Expectation::Known { .. }), "{over:?}");
        assert!(matches!(&over[1], Expectation::Unlisted { .. }), "{over:?}");
        assert!(!over[0].fails());
        assert!(
            over[1].fails(),
            "an unlisted call over the margin did not fail the run"
        );

        // The listed call, fixed: now inside the band.
        let fixed = held(&[row("listed", 2)], &known);
        assert!(
            matches!(&fixed[0], Expectation::NoLongerSlow { .. }),
            "{fixed:?}"
        );
        assert!(
            fixed[0].fails(),
            "a ledger row whose gap closed did not fail the run, so the row could outlive its bug"
        );
    }

    /// A call that is *faster* than Metal's never fires the gate: it is an upper bound on our own
    /// cost, and an unexplained win is not something a gate should bless.
    #[test]
    fn being_faster_than_metal_never_fires_the_gate() {
        let quick = held(&[row("quick", 0)], &[]);
        assert!(quick.is_empty(), "{quick:?}");
    }

    /// Between the band and the margin, a listed row is neither a pass nor a failure, because one
    /// run cannot say which side it is on: Metal's own totals moved 2.43x between two whole-set
    /// runs of one binary.
    #[test]
    fn a_listed_row_in_the_band_is_neither_a_pass_nor_a_failure() {
        let known = vec![KnownSlowCall {
            call: "listed".to_owned(),
            measured: "11.0x".to_owned(),
            where_it_is: "somewhere".to_owned(),
            waiting_on: "a gap written down at length somewhere".to_owned(),
        }];
        let inside = held(&[row("listed", 6)], &known);
        assert!(
            matches!(&inside[0], Expectation::InTheBand { .. }),
            "{inside:?}"
        );
        assert!(!inside[0].fails());
        assert!(
            band() > 2.43,
            "the band is inside the measured run-to-run movement"
        );
        assert_eq!(band(), 4.0);
    }
}
