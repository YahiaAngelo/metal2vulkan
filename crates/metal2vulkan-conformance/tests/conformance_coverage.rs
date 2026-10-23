//! `docs/CONFORMANCE-COVERAGE.md` says what it is derived from.
//!
//! The document is written by `scripts/conformance-coverage/coverage.py` from the suite's own
//! sources, the Metal API coverage map and the committed recordings. Nothing in it is maintained
//! by hand, so the only way it can be wrong is by being stale, and these tests are what makes
//! staleness a failure:
//!
//! - [`the_conformance_coverage_document_is_what_the_sources_derive`] runs the generator's
//!   `--check`, which is the generator itself: one derivation, one comparison, and no second
//!   computation of the same rule to drift from the first.
//! - [`the_case_set_is_the_recordings_the_document_counted`] holds the one link the generator
//!   cannot see. It derives per-case facts from the committed recordings, which is only the case
//!   set if the case set and the recordings are the same names. A case with no recording, or a
//!   recording no case claims, would leave the document counting something that is not the suite.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The repository root.
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The document is what the sources and the recordings derive, as the generator writes it.
#[test]
fn the_conformance_coverage_document_is_what_the_sources_derive() {
    let script = root().join("scripts/conformance-coverage/coverage.py");
    // `-B` so the run leaves no `__pycache__` beside the script: a test writes nothing into the
    // tree it is checking.
    let run = Command::new("python3")
        .arg("-I")
        .arg("-B")
        .arg(&script)
        .arg("--check")
        .current_dir(root())
        .output()
        .unwrap_or_else(|error| panic!("python3 {}: {error}", script.display()));
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(run.status.success(), "{out}");
}

/// Every case of the suite has a committed recording, and every recording is a case's.
///
/// The generator reads the recordings to say how many cases a family holds and what each one's
/// Metal answer rests on. That is a statement about the **suite** only while the two sets are the
/// same, which nothing else checks: the inputs guard holds a case to *its own* recording and says
/// nothing about a case that has none.
#[test]
fn the_case_set_is_the_recordings_the_document_counted() {
    let cases: BTreeSet<String> = metal2vulkan_conformance::cases::all()
        .iter()
        .map(|case| case.name().to_owned())
        .collect();
    let mut recorded = BTreeSet::new();
    for entry in fs::read_dir(root().join("crates/metal2vulkan-conformance/recordings")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "json") {
            recorded.insert(path.file_stem().unwrap().to_string_lossy().into_owned());
        }
    }
    let unrecorded: Vec<&String> = cases.difference(&recorded).collect();
    let orphaned: Vec<&String> = recorded.difference(&cases).collect();
    assert!(
        unrecorded.is_empty() && orphaned.is_empty(),
        "docs/CONFORMANCE-COVERAGE.md counts cases through their recordings, and the two sets \
         differ: {} case(s) with no recording {unrecorded:?}, {} recording(s) no case claims \
         {orphaned:?}",
        unrecorded.len(),
        orphaned.len()
    );
}
