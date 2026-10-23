//! The report over a selection: the verdict counts, and the matrix of case against venue.
//!
//! Errors are counted apart from verdicts. A stale recording is not a verdict, and printing it as
//! one would let it read as coverage.

use crate::replay::{ReplayError, Replayed};
use crate::verdict::Verdict;
use std::collections::{BTreeMap, BTreeSet};

/// A report over a selection of cases and venues.
#[derive(Clone, Debug, Default)]
pub struct Report {
    rows: Vec<Replayed>,
    errors: Vec<ReplayError>,
}

impl Report {
    /// An empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add what replaying one case on one venue came to.
    pub fn add(&mut self, row: Replayed) {
        self.rows.push(row);
    }

    /// Add a case that could not be replayed.
    pub fn add_error(&mut self, error: ReplayError) {
        self.errors.push(error);
    }

    /// Add either.
    pub fn add_result(&mut self, result: Result<Replayed, ReplayError>) {
        match result {
            Ok(row) => self.add(row),
            Err(error) => self.add_error(error),
        }
    }

    /// The rows.
    pub fn rows(&self) -> &[Replayed] {
        &self.rows
    }

    /// The cases that could not be replayed.
    pub fn errors(&self) -> &[ReplayError] {
        &self.errors
    }

    /// How many rows hold each verdict, by the verdict's word.
    pub fn counts(&self) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for row in &self.rows {
            *counts.entry(row.verdict.word()).or_default() += 1;
        }
        counts
    }

    /// Whether the selection is compliant: no divergence, no refusal, no unrecorded case and no
    /// error. A venue that lacks a feature does not spoil it; a case nobody recorded does.
    pub fn is_compliant(&self) -> bool {
        self.errors.is_empty() && self.rows.iter().all(|row| row.verdict.is_compliant())
    }

    /// Every row whose verdict is not one a compliant run may hold, for a test's failure message.
    pub fn failures(&self) -> Vec<&Replayed> {
        self.rows
            .iter()
            .filter(|row| !row.verdict.is_compliant())
            .collect()
    }

    /// The report: the counts, the matrix of case against venue, and then each failure in full.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("verdicts\n");
        if self.rows.is_empty() {
            out.push_str("  none\n");
        }
        for (word, count) in self.counts() {
            out.push_str(&format!("  {word:<14} {count}\n"));
        }
        if !self.errors.is_empty() {
            out.push_str(&format!("  {:<14} {}\n", "errors", self.errors.len()));
        }
        out.push_str(&self.matrix());
        for error in &self.errors {
            out.push_str(&format!("\nerror: {error}\n"));
        }
        for row in self.failures() {
            out.push_str(&format!(
                "\n{} on {}: {}\n",
                row.case, row.venue, row.verdict
            ));
        }
        out
    }

    /// The matrix of case against venue, one row per case and one column per venue.
    pub fn matrix(&self) -> String {
        let venues: BTreeSet<String> = self.rows.iter().map(|row| row.venue.label()).collect();
        let cases: BTreeSet<&str> = self.rows.iter().map(|row| row.case.as_str()).collect();
        if venues.is_empty() {
            return String::new();
        }
        let mut cells: BTreeMap<(&str, String), &'static str> = BTreeMap::new();
        for row in &self.rows {
            cells.insert((row.case.as_str(), row.venue.label()), row.verdict.word());
        }
        let case_width = cases
            .iter()
            .map(|case| case.len())
            .max()
            .unwrap_or(4)
            .max(4);
        let widths: Vec<usize> = venues
            .iter()
            .map(|venue| venue.len().max("lacks-feature".len()))
            .collect();
        let mut out = String::from("\nmatrix\n");
        out.push_str(&format!("  {:<case_width$}", "case"));
        for (venue, width) in venues.iter().zip(&widths) {
            out.push_str(&format!("  {venue:<width$}"));
        }
        out.push('\n');
        for case in &cases {
            out.push_str(&format!("  {case:<case_width$}"));
            for (venue, width) in venues.iter().zip(&widths) {
                let cell = cells.get(&(case, venue.clone())).copied().unwrap_or("-");
                out.push_str(&format!("  {cell:<width$}"));
            }
            out.push('\n');
        }
        out
    }
}

/// Whether `verdict` is one the suite's own tests may let stand, with its words for a failure.
pub fn must_hold(verdict: &Verdict) -> Result<(), String> {
    if verdict.is_compliant() {
        return Ok(());
    }
    Err(verdict.to_string())
}
