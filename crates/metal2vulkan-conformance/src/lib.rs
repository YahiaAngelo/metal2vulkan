//! The Metal conformance suite: one case, written once, run on Metal and on the runtime, and
//! compared field by field.
//!
//! The claim under test is that any difference from Metal is a bug unless we announced it. The
//! suite makes that sentence executable:
//!
//! - A case emits a **transcript** ([`transcript`]): a tree of named sites holding what a Metal
//!   program can observe — bytes the GPU wrote with their layout, the answer or typed error of
//!   every call, command buffer statuses, the notices our side announced, and explicit facts about
//!   what did not happen.
//! - Comparison is **exact by default** ([`compare`]). A loosened field names either a degrade
//!   notice's code or an entry of `docs/API.md`'s accepted differences, and the comparison fails
//!   where that entry is gone or the notice did not fire ([`allowance`]).
//! - The **verdicts** stay apart ([`verdict`]): exact, announced, divergent, refused, unrecorded,
//!   and per venue "lacks the device feature".
//! - **Two Metal oracles** ([`validated`]). Plain Metal is the behavioural oracle, because that is
//!   how a consumer runs the runtime. The same case under Metal's own API validation layer is a
//!   second recorded fact, taken in a child process so no assertion of Apple's can end a record
//!   run, and it is the only thing that may justify our refusing a call plain Metal answered.
//! - **Record mode** ([`record`]) takes Metal's answer on this Mac and writes one recording per
//!   case, keyed by chip, macOS build and Metal level, with the runs behind it and the set of
//!   outcomes where behaviour varies. **Replay** ([`mod@replay`]) reads the recording, runs our side
//!   and compares, and needs no Metal at all.
//! - A **report** ([`report`]) over a selection gives the verdict counts and the matrix of case
//!   against venue.
//! - A **known-refusals ledger** ([`known_refusals`]) holds the gaps we have written down, so one
//!   case that keeps an open bug visible does not make every run red. It changes the exit status
//!   and nothing else: the row still reads `Refused` and still counts. It is an expectation, not
//!   an allowlist — a refusal it does not list fails the run, and a row it lists that is no longer
//!   refused fails the run too, so a row cannot outlive the bug it stands for.
//!
//! What stays out, because Metal has no counterpart and a host without Metal can have no answer:
//! internal counters, and which variant or pipeline the implementation chose. Those stay device
//! lanes. The closed-vocabulary notice census stays a lane too, and reads the suite's runs through
//! [`transcript::Transcript::announced_notices`].
//!
//! A case that blocks takes the shape the queue and handler lanes take: every make that can block
//! runs on its own thread and is answered within the wait bound, so no wait in a case can hang.

pub mod allowance;
pub mod case;
pub mod cases;
pub mod compare;
pub mod device;
pub mod diff;
pub mod fixture;
pub mod known_refusals;
pub mod measured;
pub mod queue_slots;
pub mod record;
pub mod recording;
pub mod render;
pub mod replay;
pub mod report;
pub mod run;
pub mod runtime_side;
pub mod runtime_work;
pub mod texture_query;
pub mod transcript;
pub mod validated;
pub mod verdict;
pub mod work;

#[cfg(target_os = "macos")]
pub mod metal_side;

#[cfg(target_os = "macos")]
pub mod metal_work;

#[cfg(test)]
mod tests;

pub use allowance::{Allowance, Justification, Loosening};
pub use case::{Case, Input, MetalSource, ReRecord, Side, Venue};
pub use recording::{Recording, SUITE_VERSION};
pub use replay::{replay, replay_against, Replayed};
pub use report::Report;
pub use transcript::{Node, Observed, Sites, Transcript, Value};
pub use verdict::Verdict;

/// The SHA-256 of `bytes`, lowercase hexadecimal: the one digest the suite takes.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    let digest = sha2::Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

/// Bytes in a recording, as Base64: a JSON file a reader can diff, with the bytes not spread over
/// one line per element.
pub mod base64_bytes {
    use base64::Engine as _;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], out: S) -> Result<S::Ok, S::Error> {
        out.serialize_str(&base64::engine::general_purpose::STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(input: D) -> Result<Vec<u8>, D::Error> {
        let text = String::deserialize(input)?;
        base64::engine::general_purpose::STANDARD
            .decode(text.as_bytes())
            .map_err(serde::de::Error::custom)
    }
}
