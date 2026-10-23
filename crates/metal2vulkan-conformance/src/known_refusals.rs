//! The known-refusals ledger: our open gaps, each one an open bug with a todo behind it.
//!
//! A refusal is never compliant ([`Verdict::Refused`]) and this
//! module does not change that. What it changes is the **exit status** of a replay run, and only
//! that: a case that holds a known gap visible would otherwise make every run red, and a report
//! that is always red is a report nobody reads — which would cost us the next real refusal.
//!
//! # It is an expectation, not an allowlist
//!
//! `AGENTS.md` is explicit that an allowlist resolves no gap, and this must not become one. Two
//! things keep it from becoming one, and both are mechanical rather than a matter of discipline:
//!
//! - **A refusal the ledger does not list fails the run** ([`Expectation::Unlisted`]): a new gap
//!   appeared and somebody has to look at it.
//! - **A listed row that is no longer refused also fails the run**
//!   ([`Expectation::NoLongerRefused`]), saying the gap looks fixed and the row must go.
//!
//! That second direction is the point. Closing a gap forces the ledger edit, so a row cannot
//! outlive the bug it stands for. A one-directional check would quietly keep passing after the fix
//! and the file would drift into permission for whatever happened to be in it.
//!
//! The report keeps printing every one of these rows as `Refused`, with our error, with Metal's own
//! words where its API validation refuses the call too, and counted in the suite's own `refused`
//! number. Nothing is laundered.
//!
//! A row is held only against the cases a run selected, so replaying one case does not read every
//! other row as fixed.

use crate::replay::Replayed;
use crate::verdict::Verdict;
use std::collections::BTreeSet;

/// The committed ledger, compiled in so the check needs no path and no filesystem: a replay run
/// inside a container reads the same rows as one on this Mac.
const LEDGER: &str = include_str!("../known-refusals.tsv");

/// One refusal we have written down as an open gap.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct KnownRefusal {
    /// The case that refuses.
    pub case: String,
    /// The site it refuses at, as node names from the root.
    pub site: String,
    /// Our typed error there, as the report prints it. Part of the row's identity, so a refusal
    /// that changed into another refusal at the same site is a new gap rather than this one.
    pub error: String,
    /// The gap the row is waiting on, in words: what Metal does, what we do instead, and where the
    /// fix belongs. Not compared — it is what a reader needs — but a row must carry one.
    pub waiting_on: String,
}

impl KnownRefusal {
    /// What identifies the row: the case, the site and our error there.
    fn key(&self) -> (&str, &str, &str) {
        (&self.case, &self.site, &self.error)
    }
}

/// Why a run is not what the ledger expects. Each of these fails the run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expectation {
    /// A refusal the ledger does not list: a gap nobody has written down.
    Unlisted {
        case: String,
        venue: String,
        site: String,
        error: String,
    },
    /// A listed row the run did not refuse: the gap looks fixed, and the row must go.
    NoLongerRefused(KnownRefusal),
    /// A failure the ledger has no business covering. Only a refusal is ever an expected gap; a
    /// divergence, an unrecorded case and a stale recording are never written down.
    NotLedgerable { case: String, why: String },
}

impl std::fmt::Display for Expectation {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expectation::Unlisted {
                case,
                venue,
                site,
                error,
            } => write!(
                out,
                "{case} on {venue} refused {site} and the known-refusals ledger does not list it: \
                 {error}\n  a new gap. Fix it, or add a row to known-refusals.tsv naming the gap \
                 it waits on"
            ),
            Expectation::NoLongerRefused(known) => write!(
                out,
                "{} no longer refuses {}, and the known-refusals ledger still lists it\n  the gap \
                 looks fixed: drop the row from known-refusals.tsv. The ledger must not outlive \
                 the bug it stands for.\n  the row was waiting on: {}",
                known.case, known.site, known.waiting_on
            ),
            Expectation::NotLedgerable { case, why } => write!(
                out,
                "{case}: {why}\n  the ledger covers refusals alone; this is not one"
            ),
        }
    }
}

/// The committed ledger's rows.
pub fn ledger() -> Result<Vec<KnownRefusal>, String> {
    parse(LEDGER)
}

/// `text`'s rows: tab-separated, with blank lines and lines from `#` ignored.
pub fn parse(text: &str) -> Result<Vec<KnownRefusal>, String> {
    let mut rows = Vec::new();
    for (number, line) in text.lines().enumerate() {
        // Only the carriage return is stripped: a trailing tab is a field a row left empty, and
        // trimming the line would hide it as a missing field instead.
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let at = number + 1;
        let fields: Vec<&str> = line.split('\t').collect();
        let [case, site, error, waiting_on] = fields.as_slice() else {
            return Err(format!(
                "known-refusals.tsv line {at}: {} tab-separated fields, and a row has four: the \
                 case, the site, our error and the gap it waits on",
                fields.len()
            ));
        };
        if waiting_on.trim().is_empty() {
            return Err(format!(
                "known-refusals.tsv line {at}: {case} names no gap. A row is an open bug, so it \
                 says what Metal does, what we do instead, and where the fix belongs"
            ));
        }
        rows.push(KnownRefusal {
            case: case.trim().to_owned(),
            site: site.trim().to_owned(),
            error: error.trim().to_owned(),
            waiting_on: waiting_on.trim().to_owned(),
        });
    }
    Ok(rows)
}

/// How a replay run's refusals stand against `ledger`: nothing, where the refused set is exactly
/// the ledger's rows over the cases the run selected, and otherwise every way they differ.
///
/// `rows` is what replay observed. A row the ledger covers still holds its `Refused` verdict; this
/// only says whether the run held any surprise.
pub fn against(rows: &[Replayed], ledger: &[KnownRefusal]) -> Vec<Expectation> {
    let selected: BTreeSet<&str> = rows.iter().map(|row| row.case.as_str()).collect();
    let listed: BTreeSet<(&str, &str, &str)> = ledger.iter().map(KnownRefusal::key).collect();
    let mut found = Vec::new();
    let mut refused: BTreeSet<(&str, &str, &str)> = BTreeSet::new();

    for row in rows {
        match &row.verdict {
            Verdict::Refused { at, .. } => {
                for site in at {
                    let (site, error) = split_site(site);
                    let key = (row.case.as_str(), site, error);
                    refused.insert(key);
                    if !listed.contains(&key) {
                        found.push(Expectation::Unlisted {
                            case: row.case.clone(),
                            venue: row.venue.label(),
                            site: site.to_owned(),
                            error: error.to_owned(),
                        });
                    }
                }
            }
            verdict if !verdict.is_compliant() => found.push(Expectation::NotLedgerable {
                case: row.case.clone(),
                why: verdict.to_string(),
            }),
            _ => {}
        }
    }

    // The other direction, which is what keeps the file from rotting into a silencer: a row the
    // run did not refuse is a gap that looks fixed, and the ledger has to lose it. Held only
    // against the cases the run selected, so replaying one case does not read the rest as fixed.
    for known in ledger {
        if selected.contains(known.case.as_str()) && !refused.contains(&known.key()) {
            found.push(Expectation::NoLongerRefused(known.clone()));
        }
    }
    found
}

/// A refused site as the verdict prints it, split back into the site and our error there.
///
/// [`Verdict::Refused`] holds `"<site>: <error>"`, and the error may hold the separator itself, so
/// the split takes the first one alone.
fn split_site(printed: &str) -> (&str, &str) {
    let (site, error) = printed.split_once(": ").unwrap_or((printed, ""));
    // The verdict appends what documents the refusal, which is prose for a reader and not part of
    // the row's identity: a gap does not change because Apple reworded its validation message.
    let error = error
        .split_once(" (documented, not excused, by ")
        .map_or(error, |(error, _)| error);
    (site.trim(), error.trim())
}
