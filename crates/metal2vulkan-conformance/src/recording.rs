//! What a case's Metal answer is stored as.
//!
//! A recording is one file per case, keyed by the oracle it was taken from: the chip, the macOS
//! build, and the Metal level the runtime answers as ([`OracleKey`]). A case whose answer depends on
//! the Metal level holds one answer per level.
//!
//! Each answer holds **two** Metal facts, kept apart. Plain Metal is the behavioural oracle, because
//! that is how a consumer runs the runtime, and its transcript is what comparison uses. The same
//! case under Metal's API validation layer is a second fact: where the validated run stops what
//! plain Metal does, it is what may justify one of our refusals. Collapsing the two would lose the
//! distinction the owner's rules turn on.
//!
//! A case whose behaviour varies holds the **set** of transcripts seen, each with the runs behind
//! it, so a racy answer records as a frequency rather than as whichever run was recorded last.
//!
//! Nothing in a recording is host-specific: no Metal object addresses, no absolute paths. A shader
//! is named by its committed fixture and its bitcode's digest, because a host without Metal cannot
//! compile one.

use crate::transcript::Transcript;
use metal2vulkan_runtime::MetalLevel;
use serde::{Deserialize, Serialize};

/// The recording file's format. Bump it when a stored field changes meaning; a recording of another
/// format is not read.
pub const RECORDING_FORMAT_VERSION: u32 = 1;

/// The suite's own version: what a case runs, emits and compares. A recording taken under another
/// value is not current evidence for its case, because the transcript it holds may not be the one
/// the case emits now.
pub const SUITE_VERSION: u32 = 1;

/// The chip this round records on.
pub const ORACLE_CHIP: &str = "apple-m3-max";

/// The macOS release this round records on, as macOS names it.
pub const ORACLE_OS_VERSION: &str = "27.0.1";

/// The macOS build this round records on. A recording of another build is stale and fails: Apple
/// changed Metal between builds before, and a stale row that passes quietly is worse than no row.
pub const ORACLE_OS_BUILD: &str = "26A434";

/// Which oracle an answer came from. Keyed, not commented, because a case may hold one answer per
/// Metal level.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OracleKey {
    /// The chip, as the suite names it (`apple-m3-max`).
    pub chip: String,
    /// The macOS release, as macOS names it (`27.0.1`).
    pub os_version: String,
    /// The macOS build (`26A434`).
    pub os_build: String,
    /// The Metal level the answer is of ([`MetalLevel::macos`]).
    pub metal_level: String,
}

impl OracleKey {
    /// The oracle this round records on, at `level`.
    pub fn current(level: MetalLevel) -> Self {
        Self {
            chip: ORACLE_CHIP.to_owned(),
            os_version: ORACLE_OS_VERSION.to_owned(),
            os_build: ORACLE_OS_BUILD.to_owned(),
            metal_level: level.macos().to_owned(),
        }
    }

    /// The Metal level the key names, where a release names one.
    pub fn level(&self) -> Option<MetalLevel> {
        MetalLevel::ALL
            .into_iter()
            .find(|level| level.macos() == self.metal_level)
    }
}

impl std::fmt::Display for OracleKey {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            out,
            "{} on macOS {} ({}), Metal level {}",
            self.chip, self.os_version, self.os_build, self.metal_level
        )
    }
}

/// How a recorded answer was obtained.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    /// Record mode ran the case on Metal on the chip the key names.
    MetalRun,
    /// The answer is a Metal measurement this tree already holds, carried over rather than
    /// re-probed: the module whose table it came from. Its measurement is as old as that table,
    /// and re-recording it on Metal replaces it.
    Carried { from: String },
}

/// What Metal's API validation layer did with the case: a fact of its own, never compared with the
/// runtime's behaviour.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Validated {
    /// Metal's validation let the case run, and the plain transcript held.
    Ran,
    /// Metal's validation stopped the case: the call it stopped at, and its message.
    Stopped { call: String, message: String },
    /// The case was not run under Metal's validation.
    NotRun,
}

impl Validated {
    /// The words the report prints, whichever of the three the recording holds.
    ///
    /// All three say something, including [`Validated::NotRun`]. An unexercised leg read as "plain
    /// Metal ran it" once, which claims a fact nobody took: a report must tell "Metal's validation
    /// let this through" from "nobody asked Metal's validation".
    pub fn describe(&self) -> String {
        match self {
            Validated::Stopped { call, message } => {
                format!("it stopped at {call}: {message}")
            }
            Validated::Ran => "it let the case run".to_owned(),
            Validated::NotRun => {
                "the case was never run under it, so there is no such fact".to_owned()
            }
        }
    }
}

/// One transcript plain Metal produced, and how many of the runs produced it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Outcome {
    /// The runs that produced this transcript.
    pub runs: u32,
    pub transcript: Transcript,
}

/// One oracle's answer for a case.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Answer {
    pub key: OracleKey,
    pub provenance: Provenance,
    /// The device Metal named, for the reader. Nothing compares against it.
    pub device: String,
    /// The runs behind the answer: the sum of the outcomes' runs.
    pub runs: u32,
    /// The transcripts plain Metal produced, as a set with their frequencies. One outcome means
    /// the case was deterministic over `runs` runs.
    pub outcomes: Vec<Outcome>,
    /// What Metal's API validation did with the same case.
    pub validated: Validated,
}

impl Answer {
    /// The one transcript the answer holds, where the case was deterministic.
    pub fn deterministic(&self) -> Option<&Transcript> {
        match self.outcomes.as_slice() {
            [only] => Some(&only.transcript),
            _ => None,
        }
    }
}

/// A shader a case runs, as a host without Metal can get it: the committed fixture under the
/// public fixtures, and the digest of its committed text.
///
/// The digest is of the fixture's own text, not of the bitcode `llvm-as` assembles from it: the
/// text is what the tree commits and what every host reads alike, while the assembler's output can
/// differ between `llvm-as` releases.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShaderRef {
    /// The fixture, named as the case names it.
    pub fixture: String,
    /// The entry point the case runs.
    pub entry: String,
    /// The SHA-256 of the fixture's committed text.
    pub source_sha256: String,
}

/// A case's recording: every oracle's answer for it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recording {
    pub format: u32,
    /// The case the recording is of.
    pub case: String,
    /// The suite version the answers were taken under.
    pub suite_version: u32,
    /// The shaders the case runs, committed beside it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shaders: Vec<ShaderRef>,
    pub answers: Vec<Answer>,
}

impl Recording {
    /// A recording of `case` holding `answers`, under this suite version.
    pub fn new(case: impl Into<String>, shaders: Vec<ShaderRef>, answers: Vec<Answer>) -> Self {
        Self {
            format: RECORDING_FORMAT_VERSION,
            case: case.into(),
            suite_version: SUITE_VERSION,
            shaders,
            answers,
        }
    }

    /// The answer for the venue's declared Metal `level`, or why there is none.
    ///
    /// The two outcomes are kept apart because they mean different things. A recording of another
    /// chip, macOS release, macOS build, file format or suite version is **stale**: it fails,
    /// because a recording taken under one Metal and compared under another proves nothing.
    /// A recording that holds answers for this oracle but none for `level` is simply
    /// **unrecorded** at that level: nine lanes today pick their expected row from the live host's
    /// Metal level, which a host without Metal cannot read, so the level is the recording's key
    /// and a missing level is a gap to fill rather than a failure to blame on the venue.
    pub fn answer(&self, level: MetalLevel) -> Result<&Answer, Lookup> {
        if self.format != RECORDING_FORMAT_VERSION {
            return Err(Lookup::Stale(Stale::Format {
                recorded: self.format,
                current: RECORDING_FORMAT_VERSION,
            }));
        }
        if self.suite_version != SUITE_VERSION {
            return Err(Lookup::Stale(Stale::SuiteVersion {
                recorded: self.suite_version,
                current: SUITE_VERSION,
            }));
        }
        let wanted = OracleKey::current(level);
        if let Some(answer) = self.answers.iter().find(|answer| answer.key == wanted) {
            return Ok(answer);
        }
        let recorded: Vec<OracleKey> = self
            .answers
            .iter()
            .map(|answer| answer.key.clone())
            .collect();
        // An answer of this oracle at another Metal level is a level the case has not been
        // recorded at; anything else is a recording of another Metal.
        let other_level = recorded.iter().any(|key| {
            key.chip == wanted.chip
                && key.os_version == wanted.os_version
                && key.os_build == wanted.os_build
        });
        if other_level || recorded.is_empty() {
            return Err(Lookup::Unrecorded {
                level: wanted.metal_level,
            });
        }
        Err(Lookup::Stale(Stale::Oracle { wanted, recorded }))
    }
}

/// Why a recording holds no answer for a run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Lookup {
    /// The recording is not current evidence: it fails.
    Stale(Stale),
    /// The case has no answer at the venue's Metal level: it is unrecorded there.
    Unrecorded { level: String },
}

impl std::fmt::Display for Lookup {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Lookup::Stale(stale) => write!(out, "{stale}"),
            Lookup::Unrecorded { level } => {
                write!(out, "the case has no Metal answer at Metal level {level}")
            }
        }
    }
}

/// Why a recording is not current evidence. Each of these fails the case; none of them passes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stale {
    /// The file's format is not this one.
    Format { recorded: u32, current: u32 },
    /// The answers were taken under another suite version, so the transcript they hold may not be
    /// the one the case emits now.
    SuiteVersion { recorded: u32, current: u32 },
    /// The answers are of another chip, macOS release or macOS build.
    Oracle {
        wanted: OracleKey,
        recorded: Vec<OracleKey>,
    },
}

impl std::fmt::Display for Stale {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Stale::Format { recorded, current } => write!(
                out,
                "the recording's format is {recorded}, and this suite reads {current}"
            ),
            Stale::SuiteVersion { recorded, current } => write!(
                out,
                "the answers were recorded under suite version {recorded}, and this is \
                 {current}: re-record the case"
            ),
            Stale::Oracle { wanted, recorded } => {
                write!(out, "no answer is of {wanted}; the recording holds")?;
                for key in recorded {
                    write!(out, "\n  {key}")?;
                }
                out.write_str(
                    "\nre-record the case on this host, or correct ORACLE_CHIP, \
                     ORACLE_OS_VERSION and ORACLE_OS_BUILD",
                )
            }
        }
    }
}

/// The directory the recordings live in: this crate's own, beside its case set. They are the
/// conformance suite's files in the conformance suite's format, and nothing else reads them.
pub fn recordings_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("recordings")
}

/// `case`'s recording file.
pub fn recording_path(case: &str) -> std::path::PathBuf {
    recordings_dir().join(format!("{case}.json"))
}

/// `case`'s recording, read from the file [`recording_path`] names.
pub fn read(case: &str) -> Result<Recording, String> {
    let path = recording_path(case);
    let text =
        std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))
}

/// Write `recording` to the file [`recording_path`] names, pretty-printed so a diff of a
/// re-recording reads.
pub fn write(recording: &Recording) -> Result<std::path::PathBuf, String> {
    let dir = recordings_dir();
    std::fs::create_dir_all(&dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    let path = recording_path(&recording.case);
    let mut text = serde_json::to_string_pretty(recording)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    text.push('\n');
    std::fs::write(&path, text).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(path)
}
