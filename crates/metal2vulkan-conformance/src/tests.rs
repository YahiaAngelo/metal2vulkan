//! Tests of the suite's own machinery, with no device: a stub side stands in for a GPU, so the
//! transcript, the comparison, the justification rule, the verdicts, the recording's staleness and
//! the report are all checked where there is neither Metal nor Vulkan.

use crate::allowance::{
    accepted_differences, accepted_slug, Allowance, Justification, Loosening, Offered, Unjustified,
};
use crate::case::{
    call, made, Abort, Answer, Case, Failure, Handle, Input, MetalSource, ReRecord, ShaderSource,
    Side, Venue,
};
use crate::compare::compare;
use crate::diff::DivergenceKind;
use crate::record::{record, RecordError};
use crate::recording::{
    Answer as RecordedAnswer, Lookup, Outcome, Provenance, Recording, Validated, ORACLE_OS_VERSION,
};
use crate::replay::{replay_against, NOTICES_SITE};
use crate::report::Report;
use crate::transcript::{
    ByteLayout, Digest, Element, Gaps, Negative, Node, Observed, Region, Sites, Transcript, Value,
};
use crate::verdict::Verdict;
use metal2vulkan_runtime::{Degradation, MetalLevel};

/// A side that answers from a script rather than from a device, so the suite's own machinery is
/// testable with no GPU.
struct StubSide {
    /// The bytes `read_buffer` answers.
    output: Vec<u8>,
    /// The status `wait` answers.
    status: &'static str,
    /// The notice codes `announced` hands over, once.
    notices: Vec<String>,
    /// A call that fails, by its name, with the failure it gives.
    refuse: Option<(&'static str, Failure)>,
    made: u32,
}

impl StubSide {
    fn new(output: Vec<u8>) -> Self {
        Self {
            output,
            status: "Completed",
            notices: Vec::new(),
            refuse: None,
            made: 0,
        }
    }

    fn announcing(mut self, codes: &[Degradation]) -> Self {
        self.notices = codes.iter().map(|kind| kind.code().to_owned()).collect();
        self
    }

    fn refusing(mut self, call: &'static str, failure: Failure) -> Self {
        self.refuse = Some((call, failure));
        self
    }

    /// The failure the side was told to give at `call`, if any.
    fn refusal(&self, call: &str) -> Option<Failure> {
        self.refuse
            .as_ref()
            .filter(|(named, _)| *named == call)
            .map(|(_, failure)| failure.clone())
    }

    fn handle(&mut self) -> Handle {
        self.made += 1;
        Handle(self.made - 1)
    }
}

impl Side for StubSide {
    fn venue(&self) -> Venue {
        Venue {
            device: "stub".to_owned(),
            driver: "stub".to_owned(),
            api: "none".to_owned(),
            metal_level: ORACLE_OS_VERSION.to_owned(),
        }
    }

    fn library(&mut self, _source: &ShaderSource<'_>) -> Answer<Handle> {
        match self.refusal("library") {
            Some(failure) => Err(failure),
            None => Ok(self.handle()),
        }
    }

    fn compute_pipeline(&mut self, _library: Handle, _entry: &str) -> Answer<Handle> {
        match self.refusal("pipeline") {
            Some(failure) => Err(failure),
            None => Ok(self.handle()),
        }
    }

    fn buffer(&mut self, _bytes: &[u8]) -> Answer<Handle> {
        Ok(self.handle())
    }

    fn zeroed_buffer(&mut self, _length: usize) -> Answer<Handle> {
        Ok(self.handle())
    }

    fn texture_2d(
        &mut self,
        _format: metal2vulkan::metal::PixelFormat,
        _size: [u32; 2],
        _row_pitch: u32,
        _bytes: &[u8],
    ) -> Answer<Handle> {
        Ok(self.handle())
    }

    fn read_texture(
        &mut self,
        _texture: Handle,
        size: [u32; 2],
        row_pitch: u32,
    ) -> Answer<Vec<u8>> {
        let mut bytes = self.output.clone();
        bytes.resize(row_pitch as usize * size[1] as usize, 0);
        Ok(bytes)
    }

    fn dispatch(
        &mut self,
        _pipeline: Handle,
        _bindings: &crate::case::Bindings<'_>,
        _threadgroups: [u32; 3],
        _threads: [u32; 3],
    ) -> Answer<Handle> {
        match self.refusal("dispatch") {
            Some(failure) => Err(failure),
            None => Ok(self.handle()),
        }
    }

    fn compute_pipeline_with_stage_input(
        &mut self,
        _library: Handle,
        _entry: &str,
        _stage_input: &crate::case::StageInput,
    ) -> Answer<Handle> {
        match self.refusal("pipeline") {
            Some(failure) => Err(failure),
            None => Ok(self.handle()),
        }
    }

    fn dispatch_threads(
        &mut self,
        _pipeline: Handle,
        _bindings: &crate::case::Bindings<'_>,
        _stage_in: &[crate::case::StageInRegion],
        _grid: [u32; 3],
        _threads: [u32; 3],
    ) -> Answer<Handle> {
        match self.refusal("dispatch") {
            Some(failure) => Err(failure),
            None => Ok(self.handle()),
        }
    }

    fn wait(&mut self, _command_buffer: Handle) -> Answer<Value> {
        Ok(Value::unit_variant(self.status))
    }

    fn read_buffer(&mut self, _buffer: Handle, _offset: u64, length: usize) -> Answer<Vec<u8>> {
        let mut bytes = self.output.clone();
        bytes.resize(length, 0);
        Ok(bytes)
    }

    fn announced(&mut self) -> Vec<String> {
        std::mem::take(&mut self.notices)
    }
}

/// A case over one buffer of 32-bit words, the smallest thing that exercises the whole loop.
struct WordsCase {
    name: &'static str,
    words: u32,
    source: MetalSource,
    allowances: Vec<Allowance>,
    /// The input fingerprint the case states, so a test can make the inputs move.
    input: Vec<u8>,
    /// Whether the output site holds its digest alone, as a case ported from a probe that kept
    /// one does.
    digest_only: bool,
}

impl WordsCase {
    fn new(name: &'static str, words: u32) -> Self {
        Self {
            name,
            words,
            source: MetalSource::Run,
            allowances: Vec::new(),
            input: vec![0u8; 4],
            digest_only: false,
        }
    }

    fn layout(&self) -> ByteLayout {
        ByteLayout {
            region: Region::Buffer {
                offset: 0,
                length: u64::from(self.words) * 4,
            },
            element: Element::U32,
            row_offset: 0,
            row_elements: 2,
            row_pitch: 8,
            rows: self.words / 2,
            image_pitch: u32::from(u16::try_from(self.words).unwrap()) * 4,
            images: 1,
            gaps: Gaps::Ignored,
        }
    }

    /// What the output site observed: its bytes, or its digest alone where the case is
    /// digest-only.
    fn observed(&self, output: Vec<u8>) -> Observed {
        if self.digest_only {
            return Observed::absent()
                .digested(crate::transcript::Digest::sha256_prefix16(&output));
        }
        Observed::bytes(output, self.layout())
    }

    /// The transcript the case emits when its run answered `output` and its status was `status`.
    fn transcript(&self, output: Vec<u8>, status: &str, notices: &[&str]) -> Transcript {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs().unwrap()));
        sites.value("library", Value::ok(Value::Absent));
        sites.value("pipeline", Value::ok(Value::Absent));
        sites.value("buffer/0", Value::ok(Value::Absent));
        sites.value("dispatch", Value::ok(Value::Absent));
        sites.value("status", Value::ok(Value::unit_variant(status)));
        sites.value("read-back", Value::ok(Value::Absent));
        sites.push(Node::leaf("output", self.observed(output)));
        sites.notices(NOTICES_SITE, notices);
        sites.finish()
    }
}

impl Case for WordsCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        self.source.clone()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Ok(vec![Input::bytes("buffer:0", &self.input)])
    }

    fn allowances(&self) -> Vec<Allowance> {
        self.allowances.clone()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let source = ShaderSource {
            bitcode: &[],
            air_text: "",
            entry: "words",
        };
        let library = made(sites, "library", side.library(&source))?;
        let pipeline = made(sites, "pipeline", side.compute_pipeline(library, "words"))?;
        let buffer = made(
            sites,
            "buffer/0",
            side.zeroed_buffer(self.words as usize * 4),
        )?;
        let command_buffer = made(
            sites,
            "dispatch",
            side.dispatch(
                pipeline,
                &crate::case::Bindings::buffers(&[buffer]),
                [1, 1, 1],
                [self.words, 1, 1],
            ),
        )?;
        call(sites, "status", side.wait(command_buffer), Clone::clone)?;
        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(buffer, 0, self.words as usize * 4),
            |_| Value::Absent,
        )?;
        sites.push(Node::leaf("output", self.observed(bytes)));
        Ok(())
    }
}

/// The words `values` are, as little-endian bytes.
fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// A recording of one deterministic answer.
fn recording_of(case: &dyn Case, transcript: Transcript, runs: u32) -> Recording {
    Recording::new(
        case.name(),
        Vec::new(),
        vec![RecordedAnswer {
            key: crate::recording::OracleKey::current(MetalLevel::NEWEST),
            provenance: Provenance::MetalRun,
            device: "the oracle".to_owned(),
            runs,
            outcomes: vec![Outcome { runs, transcript }],
            validated: Validated::Ran,
        }],
    )
}

#[test]
fn an_identical_run_is_exact() {
    let case = WordsCase::new("words", 4);
    let bytes = words(&[1, 2, 3, 4]);
    let recording = recording_of(&case, case.transcript(bytes.clone(), "Completed", &[]), 3);
    let mut side = StubSide::new(bytes);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert_eq!(replayed.verdict, Verdict::Exact, "{}", replayed.verdict);
}

#[test]
fn one_word_moving_names_its_row_and_index() {
    let case = WordsCase::new("words", 4);
    let recording = recording_of(
        &case,
        case.transcript(words(&[1, 2, 3, 4]), "Completed", &[]),
        3,
    );
    let mut side = StubSide::new(words(&[1, 2, 9, 4]));
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    let Verdict::Divergent { found } = &replayed.verdict else {
        panic!("not divergent: {}", replayed.verdict);
    };
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].path, "case/output");
    let DivergenceKind::Element { row, index, at, .. } = &found[0].kind else {
        panic!("not an element: {:?}", found[0].kind);
    };
    assert_eq!((*row, *index, *at), (1, 0, 8));
    // The message names the leaf, not two whole regions.
    let printed = replayed.verdict.to_string();
    assert!(printed.contains("row 1"), "{printed}");
    assert!(printed.contains("Metal 3"), "{printed}");
}

#[test]
fn a_notice_we_announce_and_metal_does_not_is_divergent_until_it_is_named() {
    let kind = Degradation::ALL[0];
    let bytes = words(&[1, 2]);
    let plain = WordsCase::new("words", 2);
    let recording = recording_of(&plain, plain.transcript(bytes.clone(), "Completed", &[]), 3);

    // Unnamed, the notice is a difference nobody is accounting for.
    let mut side = StubSide::new(bytes.clone()).announcing(&[kind]);
    let replayed = replay_against(&plain, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::Divergent { .. }),
        "{}",
        replayed.verdict
    );

    // Named, with the notice announced, it is announced rather than divergent.
    let named = WordsCase {
        allowances: vec![crate::replay::notice_allowance(kind.code())],
        ..WordsCase::new("words", 2)
    };
    let mut side = StubSide::new(bytes).announcing(&[kind]);
    let replayed = replay_against(&named, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::Announced { .. }),
        "{}",
        replayed.verdict
    );
}

#[test]
fn an_allowance_whose_notice_did_not_fire_fails_the_case() {
    let kind = Degradation::ALL[0];
    let case = WordsCase {
        allowances: vec![Allowance::new(
            "case/output",
            Loosening::Unchecked,
            Justification::notice(kind),
        )],
        ..WordsCase::new("words", 2)
    };
    let recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    // The bytes differ and the allowance would cover them, but no notice was announced.
    let mut side = StubSide::new(words(&[1, 7]));
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    let Verdict::Divergent { found } = &replayed.verdict else {
        panic!("not divergent: {}", replayed.verdict);
    };
    let printed = format!("{found:?}");
    assert!(printed.contains("announced no"), "{printed}");
}

/// A notice is announced once per device per kind, so which of a case's notices its own run sees
/// depends on what ran before it. A case that names several therefore has to be justified by any
/// one of them that fired, not by whichever it listed first: the first ported batch failed two
/// cases this way, over digests that matched Metal exactly.
#[test]
fn any_one_of_a_sites_allowances_firing_justifies_the_difference() {
    let fired = Degradation::ALL[1];
    let quiet = Degradation::ALL[0];
    let bytes = words(&[1, 2]);
    let plain = WordsCase::new("words", 2);
    let recording = recording_of(&plain, plain.transcript(bytes.clone(), "Completed", &[]), 3);
    // The quiet kind is listed first, and only the other one was announced.
    let case = WordsCase {
        allowances: vec![
            crate::replay::notice_allowance(quiet.code()),
            crate::replay::notice_allowance(fired.code()),
        ],
        ..WordsCase::new("words", 2)
    };
    let mut side = StubSide::new(bytes).announcing(&[fired]);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::Announced { .. }),
        "{}",
        replayed.verdict
    );
}

/// A row whose Metal answer is a digest, on a device that announced it cannot match it, has to be
/// loosenable: the comparison checks the digest before anything else, so without consulting the
/// case's allowance there a digest-only site could never be loosened at all.
#[test]
fn a_digest_only_site_compares_under_its_allowance() {
    let kind = Degradation::ALL[0];
    let case = WordsCase {
        allowances: vec![
            Allowance::new(
                "case/output",
                Loosening::Unchecked,
                Justification::notice(kind),
            ),
            crate::replay::notice_allowance(kind.code()),
        ],
        digest_only: true,
        ..WordsCase::new("words", 2)
    };
    let recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    // Unnamed notice: the digest moving is a divergence.
    let mut side = StubSide::new(words(&[1, 7]));
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::Divergent { .. }),
        "{}",
        replayed.verdict
    );
    // With the notice announced, the digest goes unchecked, named and counted.
    let mut side = StubSide::new(words(&[1, 7])).announcing(&[kind]);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::Announced { .. }),
        "{}",
        replayed.verdict
    );
}

#[test]
fn an_allowance_naming_no_catalogue_entry_fails_the_case() {
    for justification in [
        Justification::Degradation("metal2vulkan-degrade-no-such-kind".to_owned()),
        Justification::accepted("no-such-accepted-difference"),
    ] {
        let case = WordsCase {
            allowances: vec![Allowance::new(
                "case/output",
                Loosening::Unchecked,
                justification.clone(),
            )],
            ..WordsCase::new("words", 2)
        };
        let bytes = words(&[1, 2]);
        let recording = recording_of(&case, case.transcript(bytes.clone(), "Completed", &[]), 3);
        // Even an exactly matching run fails: the allowance's justification is gone.
        let mut side = StubSide::new(bytes);
        let replayed = replay_against(&case, &mut side, &recording).unwrap();
        assert!(
            matches!(replayed.verdict, Verdict::Divergent { .. }),
            "{justification:?}: {}",
            replayed.verdict
        );
    }
}

#[test]
fn an_accepted_difference_in_the_document_justifies_a_loosened_field() {
    let slug = accepted_differences()
        .first()
        .expect("docs/API.md section 10 holds entries")
        .clone();
    let case = WordsCase {
        allowances: vec![Allowance::new(
            "case/output",
            Loosening::AnyOf(vec![Value::Uint(7)]),
            Justification::accepted(slug),
        )],
        ..WordsCase::new("words", 2)
    };
    let recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    let mut side = StubSide::new(words(&[1, 7]));
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::Announced { .. }),
        "{}",
        replayed.verdict
    );
}

#[test]
fn the_accepted_differences_are_read_from_the_document() {
    let entries = accepted_differences();
    assert!(entries.len() > 20, "{} entries", entries.len());
    assert!(entries.contains(&accepted_slug(
        "A command's read-back answers Metal's private getters"
    )));
    assert!(!entries.contains(&"nothing-of-the-sort".to_owned()));
}

/// An entry whose bold lead-in wraps onto the next line is an entry. Reading the first line alone
/// made those invisible, so an allowance naming one could never be justified. The first of these
/// wraps in the document today; the second, on one line, is the control.
#[test]
fn an_entry_whose_lead_in_wraps_is_read_whole() {
    let entries = accepted_differences();
    for lead_in in [
        "An object grid of more than 2^29 threadgroups fails its command buffer by its count, at \
         once",
        "A mesh grid of 1,048,577 to 2,097,151 threadgroups draws nothing",
    ] {
        let slug = accepted_slug(lead_in);
        assert!(entries.contains(&slug), "{slug} is not an entry");
    }
}

#[test]
fn a_typed_error_where_metal_answered_is_refused_not_divergent() {
    let case = WordsCase::new("words", 2);
    let recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    let mut side = StubSide::new(Vec::new()).refusing(
        "dispatch",
        Failure::named("Error::Invalid", "the encoder refused the dispatch"),
    );
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    let Verdict::Refused {
        at,
        metal_validation,
    } = &replayed.verdict
    else {
        panic!("not refused: {}", replayed.verdict);
    };
    assert_eq!(at.len(), 1, "{at:?}");
    assert!(at[0].starts_with("case/dispatch"), "{at:?}");
    // The recording's validated leg let the case run, so nothing excuses the refusal — and the
    // verdict says which of the three facts that is rather than leaving it to be guessed at.
    assert_eq!(*metal_validation, Validated::Ran.describe());
    let printed = replayed.verdict.to_string();
    assert!(
        printed.contains("plain Metal answered the call"),
        "{printed}"
    );
    assert!(printed.contains("let the case run"), "{printed}");
}

/// A refusal whose recording never took a validated leg says so, rather than reading as a case
/// Metal's validation let through. The two are different facts and the report must tell them
/// apart: one is a bug of ours, the other is a missing measurement.
#[test]
fn a_refusal_with_no_validated_leg_says_the_fact_is_missing() {
    let case = WordsCase::new("words", 2);
    let mut recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    recording.answers[0].validated = Validated::NotRun;
    let mut side = StubSide::new(Vec::new()).refusing(
        "dispatch",
        Failure::named("Error::Invalid", "the encoder refused the dispatch"),
    );
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    let Verdict::Refused {
        metal_validation, ..
    } = &replayed.verdict
    else {
        panic!("not refused: {}", replayed.verdict);
    };
    assert_eq!(*metal_validation, Validated::NotRun.describe());
    assert!(
        metal_validation.contains("never run under it"),
        "{metal_validation}"
    );
    // Never a pass, whatever the validated leg says.
    assert!(!replayed.verdict.is_compliant());
}

#[test]
fn a_venue_that_lacks_the_feature_is_its_own_outcome() {
    let case = WordsCase::new("words", 2);
    let recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    let lacking = Failure {
        value: Value::error("Error::Unsupported", Value::Absent),
        lacks: Some("the device enables no such capability".to_owned()),
    };
    let mut side = StubSide::new(Vec::new()).refusing("pipeline", lacking);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::LacksFeature { .. }),
        "{}",
        replayed.verdict
    );
    // It is neither a pass nor a failure, and it never counts as a divergence.
    assert!(replayed.verdict.is_compliant());
    assert_eq!(replayed.verdict.word(), "lacks-feature");
}

#[test]
fn inputs_that_are_not_the_recordings_are_unrecorded_never_a_pass() {
    let recorded = WordsCase::new("words", 2);
    let bytes = words(&[1, 2]);
    let recording = recording_of(
        &recorded,
        recorded.transcript(bytes.clone(), "Completed", &[]),
        3,
    );
    let moved = WordsCase {
        input: vec![9u8; 4],
        ..WordsCase::new("words", 2)
    };
    let mut side = StubSide::new(bytes);
    let replayed = replay_against(&moved, &mut side, &recording).unwrap();
    let Verdict::Unrecorded { why } = &replayed.verdict else {
        panic!("not unrecorded: {}", replayed.verdict);
    };
    assert!(why.contains("buffer:0 differs"), "{why}");
    assert!(!replayed.verdict.is_compliant());
}

#[test]
fn a_recording_of_another_oracle_is_stale_and_fails() {
    let case = WordsCase::new("words", 2);
    let mut recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    recording.answers[0].key.os_build = "00A000".to_owned();
    let Err(Lookup::Stale(stale)) = recording.answer(MetalLevel::NEWEST) else {
        panic!("a recording of another build is not stale");
    };
    let printed = stale.to_string();
    assert!(printed.contains("ORACLE_OS_BUILD"), "{printed}");

    // And replay refuses it rather than resolving it to a verdict.
    let mut side = StubSide::new(words(&[1, 2]));
    let error = replay_against(&case, &mut side, &recording).unwrap_err();
    assert!(format!("{error}").contains("stale"), "{error}");
}

#[test]
fn a_level_with_no_answer_is_unrecorded_rather_than_stale() {
    let case = WordsCase::new("words", 2);
    let mut recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    recording.answers[0].key.metal_level = "26.7".to_owned();
    assert_eq!(
        recording.answer(MetalLevel::NEWEST),
        Err(Lookup::Unrecorded {
            level: MetalLevel::NEWEST.macos().to_owned()
        })
    );
}

#[test]
fn a_racy_recording_passes_when_the_run_matches_one_of_its_outcomes() {
    let case = WordsCase::new("words", 2);
    let mut recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 24);
    recording.answers[0].outcomes = vec![
        Outcome {
            runs: 23,
            transcript: case.transcript(words(&[1, 2]), "Completed", &[]),
        },
        Outcome {
            runs: 1,
            transcript: case.transcript(words(&[1, 3]), "Completed", &[]),
        },
    ];
    for seen in [words(&[1, 2]), words(&[1, 3])] {
        let mut side = StubSide::new(seen.clone());
        let replayed = replay_against(&case, &mut side, &recording).unwrap();
        assert_eq!(replayed.verdict, Verdict::Exact, "{seen:?}");
    }
    // A third answer is still a divergence.
    let mut side = StubSide::new(words(&[1, 4]));
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::Divergent { .. }),
        "{}",
        replayed.verdict
    );
}

#[test]
fn record_mode_refuses_a_case_that_must_not_be_re_recorded() {
    let case = WordsCase {
        source: MetalSource::Carried {
            from: "a measured table",
            re_record: ReRecord::Forbidden {
                reason: "its Metal leg faults the GPU",
            },
            measured_on: crate::recording::ORACLE_CHIP,
        },
        ..WordsCase::new("words", 2)
    };
    let mut side = StubSide::new(words(&[1, 2]));
    let error = record(&case, &mut side, 1, MetalLevel::NEWEST, Validated::NotRun).unwrap_err();
    assert_eq!(
        error,
        RecordError::Forbidden {
            case: "words".to_owned(),
            reason: "its Metal leg faults the GPU".to_owned()
        }
    );
    assert!(format!("{error}").contains("must not be re-recorded"));
}

#[test]
fn record_mode_counts_the_outcomes_it_saw_with_their_runs() {
    // Off macOS there is no Metal to record from, and record mode says so rather than writing a
    // recording keyed to an oracle that is not here.
    #[cfg(not(target_os = "macos"))]
    {
        let case = WordsCase::new("words", 2);
        let mut side = StubSide::new(words(&[1, 2]));
        assert_eq!(
            record(&case, &mut side, 2, MetalLevel::NEWEST, Validated::Ran).unwrap_err(),
            RecordError::NoMetal
        );
    }
    #[cfg(target_os = "macos")]
    {
        let case = WordsCase::new("words", 2);
        let mut side = StubSide::new(words(&[1, 2]));
        let answer = record(&case, &mut side, 3, MetalLevel::NEWEST, Validated::Ran).unwrap();
        assert_eq!(answer.runs, 3);
        assert_eq!(
            answer.outcomes.len(),
            1,
            "a deterministic case: one outcome"
        );
        assert_eq!(answer.outcomes[0].runs, 3);
        assert_eq!(answer.provenance, Provenance::MetalRun);
    }
}

#[test]
fn a_set_diff_prints_what_is_missing_and_what_is_extra() {
    let metal = Transcript::new(vec![Node::leaf(
        "kinds",
        Observed::value(Value::set(["a", "b", "c"])),
    )]);
    let ours = Transcript::new(vec![Node::leaf(
        "kinds",
        Observed::value(Value::set(["b", "c", "d"])),
    )]);
    let found = compare(&metal, &ours, &[], Offered::notices(&[]));
    assert_eq!(found.divergences.len(), 1);
    let printed = found.divergences[0].to_string();
    assert!(printed.contains("2 in both"), "{printed}");
    assert!(printed.contains("missing {\"a\"}"), "{printed}");
    assert!(printed.contains("extra {\"d\"}"), "{printed}");
}

#[test]
fn an_ordered_list_of_lines_names_the_line_that_moved() {
    let lines = |words: &[&str]| {
        Transcript::new(vec![Node::leaf(
            "printed",
            Observed::value(Value::Lines(words.iter().map(|w| w.to_string()).collect())),
        )])
    };
    let found = compare(
        &lines(&["one", "two", "three"]),
        &lines(&["one", "TWO", "three"]),
        &[],
        Offered::notices(&[]),
    );
    assert_eq!(found.divergences.len(), 1);
    let printed = found.divergences[0].to_string();
    assert!(printed.contains("line 1"), "{printed}");
    assert!(printed.contains("\"TWO\""), "{printed}");
}

#[test]
fn a_negative_fact_differs_from_a_value_rather_than_from_an_absent_field() {
    let metal = Transcript::new(vec![Node::leaf(
        "blocked",
        Observed::value(Value::Nothing(Negative::DidNotBlock)),
    )]);
    let ours = Transcript::new(vec![Node::leaf("blocked", Observed::value(Value::Uint(3)))]);
    let found = compare(&metal, &ours, &[], Offered::notices(&[]));
    assert_eq!(found.divergences.len(), 1);
    assert!(matches!(
        found.divergences[0].kind,
        DivergenceKind::Negative {
            metal: Some(Negative::DidNotBlock),
            ours: None
        }
    ));
}

#[test]
fn the_tree_descends_to_the_leaf_that_moved() {
    let tree = |leaf: Value| {
        Transcript::new(vec![Node::branch(
            "level-0",
            vec![Node::branch(
                "level-1",
                vec![Node::leaf("level-2", Observed::value(leaf))],
            )],
        )])
    };
    let found = compare(
        &tree(Value::Uint(1)),
        &tree(Value::Uint(2)),
        &[],
        Offered::notices(&[]),
    );
    assert_eq!(found.divergences.len(), 1);
    assert_eq!(found.divergences[0].path, "case/level-0/level-1/level-2");
}

#[test]
fn the_gaps_between_rows_compare_where_the_case_asks() {
    let layout = |gaps: Gaps| ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: 8,
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: 2,
        row_pitch: 4,
        rows: 2,
        image_pitch: 8,
        images: 1,
        gaps,
    };
    let bytes = |pad: u8| vec![1, 2, pad, pad, 3, 4, pad, pad];
    for (gaps, divergences) in [(Gaps::Ignored, 0), (Gaps::Compared, 1)] {
        let metal = Transcript::new(vec![Node::leaf(
            "bytes",
            Observed::bytes(bytes(0xaa), layout(gaps)),
        )]);
        let ours = Transcript::new(vec![Node::leaf(
            "bytes",
            Observed::bytes(bytes(0x00), layout(gaps)),
        )]);
        let found = compare(&metal, &ours, &[], Offered::notices(&[]));
        assert_eq!(found.divergences.len(), divergences, "{gaps:?}");
    }
}

/// An allowance in units in the last place reaches a float **inside a byte column**.
///
/// This is the shape every exact-numbers case needs and the one easiest to assume is missing: a
/// case's answer is a buffer of `float`s, not a `Value::Float`, and `Loosening::Ulps` is written
/// against `Value::Float`. The comparison reaches it because it consults the allowance twice --
/// once over the whole `Value::Bytes`, where `Ulps` cannot hold and so falls through, and again
/// per element, where `Element::read` has turned the four bytes into a `Value::Float` of the
/// column's width. So the loosening works, and it works at the right granularity: the element
/// that moved further than the bound is still named.
///
/// The test pins all three facts, because each could regress on its own: one ulp inside a bound of
/// one is announced, two ulps is not, and the bound counts in the column's own width (a `F16`
/// column's ulp is not a `F32` column's).
#[test]
fn an_ulps_allowance_reaches_a_float_inside_a_byte_column() {
    let layout = |element: Element, elements: u32| ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(elements * element.size()),
        },
        element,
        row_offset: 0,
        row_elements: elements,
        row_pitch: elements * element.size(),
        rows: 1,
        image_pitch: elements * element.size(),
        images: 1,
        gaps: Gaps::Ignored,
    };
    let site = |bytes: Vec<u8>, element: Element, elements: u32| {
        Transcript::new(vec![Node::leaf(
            "output",
            Observed::bytes(bytes, layout(element, elements)),
        )])
    };
    let words =
        |words: &[u32]| -> Vec<u8> { words.iter().flat_map(|word| word.to_le_bytes()).collect() };
    let allowance = |ulps: u32| {
        vec![Allowance::new(
            "case/output",
            Loosening::Ulps(ulps),
            Justification::Degradation(Degradation::DeviceTranscendentals.code().to_owned()),
        )]
    };
    let announced = [Degradation::DeviceTranscendentals.code().to_owned()];

    // One ulp of a `F32` column, inside a bound of one: announced, nothing divergent.
    let metal = site(words(&[0x3f80_0000, 0x4000_0000]), Element::F32, 2);
    let ours = site(words(&[0x3f80_0001, 0x4000_0000]), Element::F32, 2);
    let found = compare(&metal, &ours, &allowance(1), Offered::notices(&announced));
    assert!(
        found.divergences.is_empty(),
        "one ulp inside a bound of one: {:?}",
        found.divergences
    );
    assert_eq!(found.announced.len(), 1);
    assert_eq!(found.announced[0].path, "case/output");
    assert_eq!(found.announced[0].count, 1);

    // Two ulps against the same bound: the element that moved too far is still named.
    let far = site(words(&[0x3f80_0002, 0x4000_0000]), Element::F32, 2);
    let found = compare(&metal, &far, &allowance(1), Offered::notices(&announced));
    assert_eq!(found.divergences.len(), 1);
    assert_eq!(found.divergences[0].path, "case/output");
    assert!(
        matches!(
            found.divergences[0].kind,
            DivergenceKind::Element {
                element: Element::F32,
                index: 0,
                ..
            }
        ),
        "{:?}",
        found.divergences[0].kind
    );

    // The bound counts in the column's own width: the same two bytes are one ulp apart as a
    // `F16` column and so are announced, where as a `F32` column they are not comparable at all.
    let halves =
        |pair: [u16; 2]| -> Vec<u8> { pair.iter().flat_map(|half| half.to_le_bytes()).collect() };
    let metal_half = site(halves([0x3c00, 0x4000]), Element::F16, 2);
    let ours_half = site(halves([0x3c01, 0x4000]), Element::F16, 2);
    let found = compare(
        &metal_half,
        &ours_half,
        &allowance(1),
        Offered::notices(&announced),
    );
    assert!(
        found.divergences.is_empty(),
        "one ulp of a half column: {:?}",
        found.divergences
    );
}

#[test]
fn a_digest_only_recording_compares_by_its_digest() {
    let digested = |bytes: &[u8]| {
        Transcript::new(vec![Node::leaf(
            "output",
            Observed::absent().digested(Digest::sha256(bytes)),
        )])
    };
    let ours = |bytes: &[u8]| {
        Transcript::new(vec![Node::leaf(
            "output",
            Observed::value(Value::Bytes(bytes.to_vec())).digested(Digest::sha256(bytes)),
        )])
    };
    assert!(compare(
        &digested(b"same"),
        &ours(b"same"),
        &[],
        Offered::notices(&[])
    )
    .is_exact());
    let found = compare(
        &digested(b"same"),
        &ours(b"other"),
        &[],
        Offered::notices(&[]),
    );
    assert!(matches!(
        found.divergences.first().map(|d| &d.kind),
        Some(DivergenceKind::Digest { .. })
    ));
}

#[test]
fn the_report_counts_the_verdicts_and_lays_out_the_matrix() {
    let case = WordsCase::new("words", 2);
    let bytes = words(&[1, 2]);
    let recording = recording_of(&case, case.transcript(bytes.clone(), "Completed", &[]), 3);
    let mut report = Report::new();
    let mut exact = StubSide::new(bytes);
    report.add_result(replay_against(&case, &mut exact, &recording));
    let mut diverged = StubSide::new(words(&[1, 9]));
    report.add_result(replay_against(&case, &mut diverged, &recording));
    let counts = report.counts();
    assert_eq!(counts.get("exact"), Some(&1));
    assert_eq!(counts.get("divergent"), Some(&1));
    let rendered = report.render();
    assert!(rendered.contains("matrix"), "{rendered}");
    assert!(rendered.contains("words"), "{rendered}");
    assert!(!report.is_compliant());
    assert_eq!(report.failures().len(), 1);
}

#[test]
fn every_seed_case_states_its_inputs_and_its_metal_source() {
    for case in crate::cases::all() {
        let inputs = case
            .inputs()
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
        assert!(
            !inputs.is_empty(),
            "{}: a case must prove it fed Metal's inputs",
            case.name()
        );
        // A carried answer says where it came from; a forbidden one says why it must stay carried.
        if let MetalSource::Carried { from, .. } = case.metal_source() {
            assert!(!from.is_empty(), "{}", case.name());
        }
        case.shaders()
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
    }
}

/// A carried answer taken on another chip says so in its recording, in a field, and one taken on
/// the oracle says nothing extra.
///
/// Two families here hold an Apple M4's values under a key that names this M3 Max — the machine is
/// gone, so they cannot be re-taken. Before the field existed the fact lived only in the
/// provenance's prose, where no reader and no check could reach it. This test is what keeps the
/// field honest in both directions: a case that starts carrying another chip's values without
/// saying so fails here, and a case that claims another chip while the oracle took its values
/// fails too.
#[test]
fn a_carried_answer_taken_on_another_chip_names_it() {
    use crate::recording::{Provenance, APPLE_M4, ORACLE_CHIP};
    let mut elsewhere = Vec::new();
    for case in crate::cases::all() {
        let claimed = match case.metal_source() {
            MetalSource::Carried { measured_on, .. } => measured_on,
            MetalSource::Run | MetalSource::DidNotReturn { .. } => continue,
        };
        let recorded = crate::recording::read(case.name())
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()))
            .answer(MetalLevel::NEWEST)
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()))
            .provenance
            .clone();
        let Provenance::Carried { measured_on, .. } = &recorded else {
            panic!("{}: a carried case recorded as {recorded:?}", case.name());
        };
        assert_eq!(measured_on, claimed, "{}", case.name());
        if claimed != ORACLE_CHIP {
            elsewhere.push(case.name());
        }
        assert_eq!(
            recorded.measured_elsewhere(),
            (claimed != ORACLE_CHIP).then_some(claimed),
            "{}",
            case.name()
        );
    }
    // These families, and no others. One the list does not name would mean a table nobody noticed
    // came from another machine. `two_plane_heap_blits` used to be here and now records on Metal:
    // nothing it does is Apple SPI, so its answer is this chip's.
    //
    // - `texture-fills-*`: the M4's fill digests, which `fillTexture:` being Apple SPI keeps
    //   carried.
    // - `ycbcr-minify-band-*`: the M4's per-band digests of a minified two-plane YCbCr texture's
    //   samples, which stay carried because a texture of such a format exists only over an
    //   IOSurface and the Metal side has no call that makes one.
    // - `compute-threadgroup-past-metals-limit`: the M4's answer for a threadgroup past Metal's
    //   1024, which stays carried because this round may not hand a GPU an over-limit
    //   threadgroup to ask again.
    const KNOWN: [&str; 3] = [
        "texture-fills-",
        "ycbcr-minify-band-",
        crate::cases::compute_threadgroups::PAST_METALS_LIMIT,
    ];
    for family in KNOWN {
        assert!(
            elsewhere.iter().any(|name| name.starts_with(family)),
            "{family} carries the M4's values: {elsewhere:?}"
        );
    }
    for name in &elsewhere {
        assert!(
            KNOWN.iter().any(|family| name.starts_with(family)),
            "{name} carries another chip's values and is not one of the known families"
        );
    }
    // The file says the chip only where it differs: an oracle-taken recording holds no such field.
    let on_oracle = crate::recording::read(crate::cases::surface_fills::SURFACE_FILLS).unwrap();
    assert_eq!(
        on_oracle.answers[0].provenance.measured_elsewhere(),
        None,
        "a recording taken on the oracle names no other chip"
    );
    assert!(!serde_json::to_string(&on_oracle.answers[0].provenance)
        .unwrap()
        .contains(APPLE_M4));
}

#[test]
fn the_carried_table_of_the_private_type_getters_is_metals_measured_one() {
    let case = crate::cases::indirect_private_types::private_type_getters();
    let transcript = case.measured_transcript().unwrap();
    let answers = |raw: u64, form: &str| {
        let node = transcript
            .root
            .child(&format!("type/{raw}"))
            .unwrap_or_else(|| panic!("no type/{raw}"))
            .child(form)
            .unwrap_or_else(|| panic!("no type/{raw}/{form}"));
        let read = |name: &str| node.child(name).unwrap().observed.value.clone();
        (
            read("command_type"),
            read("answered_primitive_type"),
            read("recorded_primitive_type"),
        )
    };
    let some = |value: u64| Value::Variant {
        name: "Some".to_owned(),
        payload: Box::new(Value::Uint(value)),
    };
    // Types 0 to 6 read back as set; 7, 8 and 9 read back as points.
    assert_eq!(answers(6, "direct"), (Value::Uint(1), some(6), some(6)));
    assert_eq!(answers(7, "direct"), (Value::Uint(1), some(0), some(7)));
    assert_eq!(answers(8, "indexed"), (Value::Uint(2), some(0), some(8)));
    // The command type of an indexed draw of type 9 is none, and its record keeps type 9.
    assert_eq!(answers(9, "indexed"), (Value::Uint(0), some(0), some(9)));
    assert_eq!(answers(9, "direct"), (Value::Uint(1), some(0), some(9)));
}

/// The validated oracle's reader attributes a complaint to the call the marker before it named,
/// strips `NSLog`'s host-specific prefix, and drops the layer's own banner.
#[test]
fn the_validated_leg_reads_metals_own_words() {
    let said = format!(
        "2026-10-23 07:33:40.512 conformance[51249:39389017] {}\n\
         {}rate_map\n\
         2026-10-23 07:33:40.513 conformance[51249:39389017] RasterizationRateMap Layers \
         Validation\n\
         layerCount (0) must be >0 and <=2.\n\
         layers must not be nil.\n",
        "Metal API Validation Enabled",
        crate::validated::MARKER,
    );
    let read = crate::validated::read(&said).expect("a complaint");
    let Validated::Stopped { call, message } = &read else {
        panic!("{read:?}");
    };
    assert_eq!(call, "rate_map");
    // The complaint keeps Metal's words and none of this host's: no process id, no timestamp.
    assert_eq!(
        message,
        "RasterizationRateMap Layers Validation\nlayerCount (0) must be >0 and <=2.\n\
         layers must not be nil."
    );
    assert!(!message.contains("51249"), "{message}");
    // A leg the layer complained about nothing in is not a stop.
    assert_eq!(
        crate::validated::read(&format!(
            "2026-10-23 07:33:40.512 conformance[1:2] Metal API Validation Enabled\n{}rate_map\n",
            crate::validated::MARKER
        )),
        None
    );
}

/// A refusal the recording's validated oracle justifies holds, and the same allowance holds for
/// nothing else: not where the validated leg let the call run, not where it was never taken, and
/// not where Apple's check has moved.
#[test]
fn metals_validation_justifies_a_refusal_and_only_a_recorded_one() {
    let says = "screenSize.width (0) must be >0 and <=32768.";
    let justification = Justification::metal_validation(says);
    let stopped = Validated::Stopped {
        call: "rate_map".to_owned(),
        message: format!("RasterizationRateMap Layers Validation\n{says}"),
    };
    assert_eq!(
        justification.unjustified(&Offered::new(&[], &stopped)),
        None
    );
    assert_eq!(
        justification.unjustified(&Offered::new(&[], &Validated::Ran)),
        Some(Unjustified::MetalValidationRan)
    );
    // The one the whole axis turns on: an unexercised validated leg justifies nothing.
    assert_eq!(
        justification.unjustified(&Offered::notices(&[])),
        Some(Unjustified::MetalValidationNotRun)
    );
    let moved = Validated::Stopped {
        call: "rate_map".to_owned(),
        message: "screenSize.width (0) must be >0 and <=16384.".to_owned(),
    };
    assert!(matches!(
        justification.unjustified(&Offered::new(&[], &moved)),
        Some(Unjustified::MetalValidationSaysOtherwise { .. })
    ));
}

/// A refusal stays a refusal however well documented it is.
///
/// Metal's API validation stopping a call is not Metal refusing it: plain Metal answered, so the
/// row is `Refused` and is not compliant, and the justification buys the words in the report and
/// nothing else. Where the recording holds no such fact, the claim itself fails too.
#[test]
fn a_refusal_reads_by_what_metals_validation_did() {
    let says = "screenSize.width (0) must be >0 and <=32768.";
    let metal = Transcript::new(vec![Node::leaf(
        "map",
        Observed::value(Value::ok(Value::Absent)),
    )]);
    let ours = Transcript::new(vec![Node::leaf(
        "map",
        Observed::value(Value::error("Error::Invalid", Value::Absent)),
    )]);
    let allowances = vec![Allowance::new(
        "case/map",
        Loosening::Unchecked,
        Justification::metal_validation(says),
    )];
    let stopped = Validated::Stopped {
        call: "rate_map".to_owned(),
        message: says.to_owned(),
    };
    let found = crate::compare::compare_run(
        &metal,
        &ours,
        &allowances,
        Offered::new(&[], &stopped),
        true,
    );
    // The refusal is not laundered into an announced difference: it stays its own row.
    assert_eq!(found.refusals.len(), 1, "{found:?}");
    assert!(found.announced.is_empty(), "{found:?}");
    assert!(found.bad_allowances.is_empty(), "{found:?}");
    // What the justification bought: Metal's own words beside the row, and nothing else.
    assert_eq!(
        found.refusals[0].documented_by,
        Some(Justification::metal_validation(says).describe())
    );
    let verdict = Verdict::of(&found, &stopped);
    assert_eq!(verdict.word(), "refused");
    assert!(!verdict.is_compliant(), "a refusal is never compliant");
    let printed = verdict.to_string();
    assert!(printed.contains("documented, not excused"), "{printed}");
    assert!(printed.contains(says), "{printed}");

    // Without the validated leg the claim fails too, so the case says it has no explanation
    // rather than offering one nobody took.
    let found =
        crate::compare::compare_run(&metal, &ours, &allowances, Offered::notices(&[]), true);
    assert_eq!(found.bad_allowances.len(), 1, "{found:?}");
    assert_eq!(
        found.bad_allowances[0].why,
        Unjustified::MetalValidationNotRun
    );
    assert_eq!(found.refusals.len(), 1, "{found:?}");
    assert_eq!(found.refusals[0].documented_by, None);
    // A claim that does not hold fails the case on its own, ahead of the refusal it was made for.
    assert!(!Verdict::of(&found, &Validated::NotRun).is_compliant());
}

/// A refusal nothing documents is still exactly one refusal, and the words say plain Metal
/// answered. This is the shape the ported lanes will mostly meet.
#[test]
fn an_undocumented_refusal_is_one_row_and_names_no_excuse() {
    let metal = Transcript::new(vec![Node::leaf(
        "map",
        Observed::value(Value::ok(Value::Absent)),
    )]);
    let ours = Transcript::new(vec![Node::leaf(
        "map",
        Observed::value(Value::error("Error::Invalid", Value::Absent)),
    )]);
    let found =
        crate::compare::compare_run(&metal, &ours, &[], Offered::new(&[], &Validated::Ran), true);
    assert_eq!(found.refusals.len(), 1, "{found:?}");
    assert_eq!(found.refusals[0].documented_by, None);
    let verdict = Verdict::of(&found, &Validated::Ran);
    assert_eq!(verdict.word(), "refused");
    assert!(!verdict.is_compliant());
    let printed = verdict.to_string();
    assert!(
        printed.contains("plain Metal answered the call"),
        "{printed}"
    );
    assert!(!printed.contains("documented"), "{printed}");
}

/// The rate-map cases state the descriptor they feed, name the words Metal's validation used, and
/// keep the refusal's allowance to the call's own site.
#[test]
fn the_rate_map_cases_name_what_justifies_them() {
    let cases = crate::cases::rate_maps::all();
    let names: Vec<&str> = cases.iter().map(|case| case.name()).collect();
    assert_eq!(
        names,
        vec![
            crate::cases::rate_maps::NO_LAYERS,
            crate::cases::rate_maps::ZERO_SCREEN_WIDTH
        ]
    );
    for case in &cases {
        // Record mode runs both on Metal: neither can fault the GPU, because neither reaches it.
        assert_eq!(case.metal_source(), MetalSource::Run);
        assert!(case.metal_source().forbidden().is_none());
        // Every case states its own inputs, and the descriptor is what it feeds.
        let inputs = case.inputs().unwrap();
        assert!(
            inputs.iter().any(|input| input.name == "descriptor"),
            "{inputs:?}"
        );
        // Every allowance names something that exists: the rate-map degradation, over the
        // notices site alone, so every answer of the map itself still compares exactly.
        for allowance in case.allowances() {
            match &allowance.justified_by {
                Justification::Degradation(code) => {
                    assert_eq!(code, Degradation::RasterizationRateMap.code());
                    assert_eq!(allowance.path, "case/notices/*");
                }
                other => panic!("an unexpected tolerance: {other:?}"),
            }
        }
    }
    // The map of the identity announces nothing and carries no allowance; the one that is not the
    // identity announces its degradation when it is made. Neither rests on the validated oracle
    // any more: the zero-screen-width refusal that did is closed.
    assert!(cases[0].allowances().is_empty());
    assert_eq!(cases[1].allowances().len(), 1);
    for case in &cases {
        assert!(
            !case.allowances().iter().any(|allowance| matches!(
                allowance.justified_by,
                Justification::MetalValidation { .. }
            )),
            "{}",
            case.name()
        );
    }
}

/// The announcing side forwards **every** `Side` call, and this is what keeps it from rotting.
///
/// `Side` gives most of its calls a default that answers `Failure::unavailable`, so a method the
/// announcing wrapper forgets to forward makes `Announcing<MetalSide>` answer "no such call" where
/// `MetalSide` answers perfectly well — and the compiler says nothing, because the default is
/// there to be inherited. The trait grows with every ported batch, so the guard must be mechanical
/// rather than a list somebody remembers to extend.
///
/// Comparing transcripts through a stub does not do it: a stub that does not implement the method
/// either inherits the same default, so both sides agree and the check is vacuous. (Verified by
/// removing a forward and watching such a check pass.) So the guard reads the two sources and
/// compares the method names directly: a call `Side` declares and `Announcing` does not forward
/// fails here, naming it.
#[test]
fn the_announcing_side_forwards_every_call() {
    /// The names of the `fn` items `block` declares at one level of indentation.
    fn methods(source: &str, opening: &str) -> std::collections::BTreeSet<String> {
        let from = source
            .find(opening)
            .unwrap_or_else(|| panic!("no {opening} in the source"))
            + opening.len();
        let rest = &source[from..];
        // The block ends at the first `}` in the first column, which closes it.
        let to = rest.find("\n}\n").unwrap_or(rest.len());
        rest[..to]
            .lines()
            .filter_map(|line| {
                line.strip_prefix("    fn ")
                    .or(line.strip_prefix("    pub fn "))
            })
            .filter_map(|rest| rest.split('(').next())
            .map(|name| name.trim().to_owned())
            .collect()
    }
    let declared = methods(include_str!("case.rs"), "\npub trait Side {");
    let forwarded = methods(
        include_str!("validated.rs"),
        "\nimpl<S: Side> Side for Announcing<S> {",
    );
    assert!(declared.len() > 10, "the trait was not found: {declared:?}");
    let missing: Vec<&String> = declared.difference(&forwarded).collect();
    assert!(
        missing.is_empty(),
        "validated::Announcing does not forward {missing:?}, so it answers Unavailable where the \
         side it wraps answers. Forward each of them."
    );
    let extra: Vec<&String> = forwarded.difference(&declared).collect();
    assert!(
        extra.is_empty(),
        "Announcing forwards {extra:?}, which Side no longer declares"
    );
}

/// Wrapping a side changes nothing a case observes: the announcing wrapper is a marker on standard
/// error and nothing else. This catches a forward wired to the wrong inner call, which the
/// name-set check above cannot see.
#[test]
fn announcing_a_side_changes_no_transcript() {
    let transcript_of = |case: &dyn Case, announcing: bool| {
        let inputs = case.inputs().unwrap_or_default();
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&inputs));
        if announcing {
            let mut side = crate::validated::Announcing::new(StubSide::new(vec![0; 4096]));
            let _ = case.run(&mut side, &mut sites);
        } else {
            let mut side = StubSide::new(vec![0; 4096]);
            let _ = case.run(&mut side, &mut sites);
        }
        sites.finish()
    };
    for case in crate::cases::all() {
        assert_eq!(
            transcript_of(case.as_ref(), false),
            transcript_of(case.as_ref(), true),
            "{}: the announcing side answered otherwise than the side it wraps",
            case.name()
        );
    }
}

/// The committed recordings hold the second Metal oracle, and hold all three of its answers.
///
/// This ties the recordings to the cases: a re-record that lost the validated leg, or an Apple
/// release whose wording moved, fails here with no device and no Metal, instead of surfacing as a
/// puzzling verdict later. It is also the standing proof that the axis has been exercised — the
/// thing the suite could not say before these cases existed.
#[test]
fn the_committed_recordings_carry_all_three_validated_answers() {
    let validated = |case: &str| {
        crate::recording::read(case)
            .unwrap_or_else(|why| panic!("{case}: {why}"))
            .answer(MetalLevel::NEWEST)
            .unwrap_or_else(|why| panic!("{case}: {why}"))
            .validated
            .clone()
    };
    let says = |case: &str, words: &str| match validated(case) {
        Validated::Stopped { call, message } => {
            assert_eq!(call, "rate_map", "{case}");
            assert!(message.contains(words), "{case}: {message}");
        }
        other => panic!("{case}: {other:?}"),
    };
    // Stopped: Metal's own API validation refuses what plain Metal answered, in its own words.
    says(
        crate::cases::rate_maps::NO_LAYERS,
        crate::cases::rate_maps::NO_LAYERS_SAYS,
    );
    says(
        crate::cases::rate_maps::ZERO_SCREEN_WIDTH,
        crate::cases::rate_maps::ZERO_WIDTH_SAYS,
    );
    // Ran: the layer had nothing to say about a compute case, and the plain answer stood.
    for case in crate::cases::pass_bytes::all() {
        assert_eq!(validated(case.name()), Validated::Ran, "{}", case.name());
    }
    // NotRun: a carried answer was never run here at all, and the file says so rather than
    // implying Metal's validation was content with it.
    assert_eq!(
        validated(crate::cases::indirect_private_types::PRIVATE_TYPE_GETTERS),
        Validated::NotRun
    );
}

/// The committed ledger parses, every row names a gap, and each row's case is a real case.
///
/// A row whose case has been renamed away would otherwise read as "no longer refused" forever, or
/// never be held at all, and either way the row would stop meaning anything.
#[test]
fn the_known_refusals_ledger_parses_and_names_real_cases() {
    let ledger = crate::known_refusals::ledger().unwrap();
    assert!(
        !ledger.is_empty(),
        "the ledger is empty; drop the machinery"
    );
    for known in &ledger {
        assert!(
            crate::cases::named(&known.case).is_some(),
            "{}: no such case",
            known.case
        );
        assert!(known.site.starts_with("case/"), "{known:?}");
        assert!(!known.error.is_empty(), "{known:?}");
        // A row is an open bug, so it says where the fix belongs.
        assert!(
            known.waiting_on.len() > 40,
            "{}: the gap is not described: {}",
            known.case,
            known.waiting_on
        );
    }
    // The rows this round left open, and what they are waiting on: `drawMeshThreads` whose
    // threads leave a partial object threadgroup, which Metal runs with the threads left over.
    let partial: Vec<&crate::known_refusals::KnownRefusal> = ledger
        .iter()
        .filter(|known| {
            known
                .case
                .starts_with("mesh-object-draws-partial-threadgroup-")
        })
        .collect();
    assert_eq!(partial.len(), 2, "{ledger:?}");
    for known in partial {
        assert!(known.site.ends_with("drawMeshThreads"), "{known:?}");
        assert!(
            known.waiting_on.contains("partial object threadgroup"),
            "{}",
            known.waiting_on
        );
    }
    // The zero-screen-width refusal is closed, so no row may name that case: the ledger holds
    // rows in both directions and a stale row would fail the run it is selected in.
    assert!(
        !ledger
            .iter()
            .any(|known| known.case == crate::cases::rate_maps::ZERO_SCREEN_WIDTH),
        "the zero-screen-width gap is closed; its row must be gone"
    );
}

/// A malformed ledger is refused rather than read loosely: a row must carry all four fields, and
/// must name the gap it waits on.
#[test]
fn a_ledger_row_without_its_gap_is_refused() {
    let parse = crate::known_refusals::parse;
    assert!(parse("# just a comment\n\n").unwrap().is_empty());
    let row = "a-case\tcase/map\tErr(Error::Invalid)\tthe runtime refuses what Metal does";
    assert_eq!(parse(row).unwrap().len(), 1);
    let why = parse("a-case\tcase/map\tErr(Error::Invalid)\t").unwrap_err();
    assert!(why.contains("names no gap"), "{why}");
    let why = parse("a-case\tcase/map").unwrap_err();
    assert!(why.contains("four"), "{why}");
}

/// The ledger is an expectation held in both directions, and it changes no verdict.
#[test]
fn the_ledger_fails_on_a_new_refusal_and_on_one_that_is_gone() {
    use crate::known_refusals::{against, Expectation, KnownRefusal};
    let case = WordsCase::new("words", 2);
    let recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    let mut side = StubSide::new(Vec::new()).refusing(
        "dispatch",
        Failure::named("Error::Invalid", "the encoder refused the dispatch"),
    );
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    // The verdict is untouched by any ledger: a refusal stays a refusal and stays non-compliant.
    assert_eq!(replayed.verdict.word(), "refused");
    assert!(!replayed.verdict.is_compliant());
    let rows = vec![replayed];

    // Nothing written down: the refusal is a new gap and fails the run.
    let found = against(&rows, &[]);
    assert_eq!(found.len(), 1, "{found:?}");
    let Expectation::Unlisted {
        case: named,
        site,
        error,
        ..
    } = &found[0]
    else {
        panic!("{found:?}");
    };
    assert_eq!(named, "words");
    assert_eq!(site, "case/dispatch");

    // Written down exactly: nothing unexpected, so the run may exit 0.
    let listed = vec![KnownRefusal {
        case: "words".to_owned(),
        site: site.clone(),
        error: error.clone(),
        waiting_on: "the stub refuses the dispatch, which is the point of the stub".to_owned(),
    }];
    assert_eq!(against(&rows, &listed), Vec::new());

    // The direction that keeps the file from rotting: the same ledger against a run that did NOT
    // refuse fails, and says the row must go.
    let mut clean = StubSide::new(words(&[1, 2]));
    let passed = replay_against(&case, &mut clean, &recording).unwrap();
    assert_eq!(passed.verdict.word(), "exact");
    let found = against(std::slice::from_ref(&passed), &listed);
    assert_eq!(found.len(), 1, "{found:?}");
    let Expectation::NoLongerRefused(known) = &found[0] else {
        panic!("{found:?}");
    };
    assert_eq!(known.case, "words");
    let printed = found[0].to_string();
    assert!(printed.contains("looks fixed"), "{printed}");
    assert!(printed.contains("drop the row"), "{printed}");

    // A row is held only against the cases a run selected, so replaying one case does not read
    // every other row as fixed.
    let elsewhere = vec![KnownRefusal {
        case: "another-case".to_owned(),
        site: "case/map".to_owned(),
        error: "Err(Error::Invalid)".to_owned(),
        waiting_on: "a gap of some other case, which this run did not select".to_owned(),
    }];
    assert_eq!(
        against(&rows, &elsewhere).len(),
        1,
        "the unlisted one alone"
    );

    // A divergence is never a ledger row: only a refusal is ever an expected gap.
    let mut wrong = StubSide::new(words(&[1, 9]));
    let diverged = replay_against(&case, &mut wrong, &recording).unwrap();
    assert_eq!(diverged.verdict.word(), "divergent");
    let found = against(std::slice::from_ref(&diverged), &[]);
    assert!(
        matches!(found.as_slice(), [Expectation::NotLedgerable { .. }]),
        "{found:?}"
    );
}

#[test]
fn an_allowance_no_site_needed_is_counted_and_named_rather_than_silently_kept() {
    // The hole this closes: the up-front check over a case's allowances has to exempt a degrade
    // notice the run did not announce, because a notice is announced once per device per kind. So
    // a case could name a tolerance that fires on no venue, naming a notice no venue announces,
    // and every run would read green with nobody told. Two allowances of exactly this shape were
    // written and deleted by hand in this round.
    let mut case = WordsCase::new("words", 2);
    case.allowances = vec![
        // A tolerance nothing needs, for a notice nothing announces: speculation.
        Allowance::new(
            "case/output",
            Loosening::Absolute(8.0),
            Justification::notice(Degradation::DeviceArithmetic),
        ),
        // A tolerance nothing needs, for a notice the run did announce: not speculation, just
        // unused on this row.
        Allowance::new(
            "case/output",
            Loosening::Absolute(8.0),
            Justification::notice(Degradation::ShadowedTexture),
        ),
    ];
    let bytes = words(&[1, 2]);
    // The notice the second allowance names is announced, so it is justified; the run is
    // byte-identical, so neither allowance fires.
    let announced = [Degradation::ShadowedTexture.code()];
    let recording = recording_of(
        &case,
        case.transcript(bytes.clone(), "Completed", &announced),
        3,
    );
    let mut side = StubSide::new(bytes).announcing(&[Degradation::ShadowedTexture]);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();

    // Report-only: the run is still exact, and still compliant.
    assert_eq!(replayed.verdict, Verdict::Exact, "{}", replayed.verdict);
    assert_eq!(replayed.allowances.named, 2);
    assert_eq!(replayed.allowances.fired(), 0);
    assert_eq!(
        replayed.allowances.unfired.len(),
        2,
        "{:?}",
        replayed.allowances.unfired
    );

    let mut report = Report::new();
    report.add(replayed);
    assert!(report.is_compliant());
    assert_eq!(report.allowance_counts(), (2, 2));
    assert_eq!(report.unfired_everywhere().len(), 2);
    // Only the one nothing justifies is called out as a tolerance no run has checked.
    let unchecked = report.unchecked_everywhere();
    assert_eq!(unchecked.len(), 1, "{unchecked:?}");
    assert_eq!(
        unchecked[0].1.justified_by,
        Justification::notice(Degradation::DeviceArithmetic).describe()
    );
    let rendered = report.render();
    assert!(
        rendered.contains("2 named, 0 fired, 2 never fired"),
        "{rendered}"
    );
    assert!(
        rendered.contains(Degradation::DeviceArithmetic.code()),
        "{rendered}"
    );
    assert!(rendered.contains("words                "), "{rendered}");
}

#[test]
fn an_allowance_that_fires_is_not_counted_as_unfired() {
    // The other half of the guard: a tolerance the run needs must not show up in the unfired
    // count, or the count would say "nobody checked this" of the one allowance a run did check.
    let mut case = WordsCase::new("words", 2);
    case.allowances = vec![
        Allowance::new(
            "case/output",
            Loosening::Absolute(8.0),
            Justification::notice(Degradation::DeviceArithmetic),
        ),
        // The notice site differs too, because the device announced what the tolerance names and
        // Metal announces nothing. That allowance fires as well, which is the point: both are
        // claims the run used.
        crate::replay::notice_allowance(Degradation::DeviceArithmetic.code()),
    ];
    let recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    // One word differs by 7, inside the allowance, and the notice it names is announced.
    let mut side = StubSide::new(words(&[1, 9])).announcing(&[Degradation::DeviceArithmetic]);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::Announced { .. }),
        "{}",
        replayed.verdict
    );
    assert_eq!(replayed.allowances.fired(), 2);
    assert!(
        replayed.allowances.unfired.is_empty(),
        "{:?}",
        replayed.allowances.unfired
    );
    let mut report = Report::new();
    report.add(replayed);
    assert!(report.unfired_everywhere().is_empty());
    assert!(report.render().contains("2 named, 2 fired, 0 never fired"));
}

#[test]
fn metal_on_this_chip_answers_the_two_plane_blits_exactly_as_the_probes_printed_them() {
    // Two rows lost their Metal leg to a reason that named the wrong thing: `NO_PUBLIC_CALL` said
    // the two-plane extended-range formats are Apple's private ones, so Metal had no public call
    // for them. The fill is private; the format is not. Both rows now record on Metal, and this
    // holds the committed recordings to the tables the probes printed — so the conversion is
    // checked rather than asserted, and a Metal that starts answering differently shows up here
    // instead of in a quiet re-record.
    let held = |case: &dyn Case, carried: &Transcript| {
        let recorded = crate::recording::read(case.name())
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
        let answer = recorded
            .answer(MetalLevel::NEWEST)
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
        assert_eq!(
            answer.provenance,
            Provenance::MetalRun,
            "{}: the case records on Metal now",
            case.name()
        );
        let taken = answer
            .deterministic()
            .unwrap_or_else(|| panic!("{}: one outcome", case.name()));
        // Field by field, through the suite's own comparison: a difference names the site.
        let found = compare(carried, taken, &[], Offered::notices(&[]));
        // The first few sites, not the whole comparison: a difference has to read on one screen.
        assert!(
            found.is_exact(),
            "{}: Metal's recorded answer is not the probe's table:\n  {}",
            case.name(),
            found
                .divergences
                .iter()
                .take(4)
                .map(|divergence| divergence.to_string())
                .collect::<Vec<String>>()
                .join("\n  ")
        );
    };

    let heap_blits = crate::cases::two_plane_heap_blits::heap_blits();
    let (runs, carried) = heap_blits.carried_from_the_m4().unwrap();
    assert_eq!(runs, 6, "the probe's runs behind the carried digests");
    held(&heap_blits, &carried);

    let announced = crate::cases::two_plane_copies::announced_row_blits();
    let printed = announced.printed_by_the_probe().unwrap();
    held(&announced, &printed);
    // And the case that still carries, still carries: its reason is the fill, not the format.
    let fills = crate::cases::two_plane_copies::row_blits();
    assert!(
        matches!(
            fills.metal_source(),
            MetalSource::Carried {
                re_record: ReRecord::Forbidden { reason },
                ..
            } if reason.contains("fillTexture:") && !reason.contains("formats are Apple's private")
        ),
        "{:?}",
        fills.metal_source()
    );
}

/// A case whose Metal leg does not come back: the shape there was no verdict for.
///
/// `pass`'s call answers on our side, which is what the row pins; `answers` makes it answer
/// instead, which is the break the verdict has to survive.
struct DiedOnMetal {
    /// Whether our side answers the call rather than refusing it.
    answers: bool,
}

/// The call the probe died in, as the case names its site.
const DIED_AT: &str = "the-call";

/// What the probe saw, in its own words.
const DIED_SAYS: &str = "Metal's process went away inside the call and never came back";

impl DiedOnMetal {
    /// The transcript the case states by hand: the inputs, and the call Metal never returned
    /// from.
    fn stated(&self) -> Transcript {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs().unwrap()));
        sites.value(DIED_AT, Value::Nothing(Negative::NeverReturned));
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        sites.finish()
    }
}

impl Case for DiedOnMetal {
    fn name(&self) -> &'static str {
        "died-on-metal"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::did_not_return("the suite's own tests", DIED_AT, DIED_SAYS)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Ok(vec![Input::bytes("buffer:0", &[0u8; 4])])
    }

    fn carried_transcript(&self) -> Result<Option<(u32, Transcript)>, String> {
        Ok(Some((1, self.stated())))
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        if self.answers {
            // We came back from the call Metal did not: the divergence the row exists to catch.
            let _ = made(sites, DIED_AT, side.zeroed_buffer(4))?;
            return Ok(());
        }
        sites.push(Node::leaf(
            DIED_AT,
            Observed::value(Failure::named("Unsupported", "this call is not built").value),
        ));
        Ok(())
    }
}

#[test]
fn a_call_metal_did_not_return_from_is_its_own_verdict() {
    // The hole this closes: a row that ends Metal's process has nothing to carry -- the lane holds
    // no line for it -- so it was not recorded-only material, it had no verdict at all. `Exact`
    // would claim a value matched where Metal produced none; `Divergent` would claim a behaviour
    // of Metal's to differ from; `Refused` is for a call plain Metal answered, and counting it
    // there would bury a real gap of ours among rows that are not one.
    let case = DiedOnMetal { answers: false };
    let (runs, stated) = case.carried_transcript().unwrap().unwrap();
    let recording = Recording::new(
        case.name(),
        Vec::new(),
        vec![crate::record::never_returned(
            MetalLevel::NEWEST,
            "the suite's own tests",
            DIED_AT,
            "the probe's device",
            runs,
            stated,
        )],
    );
    let mut side = StubSide::new(Vec::new());
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    let Verdict::MetalNeverReturned { at, justified } = &replayed.verdict else {
        panic!("{}", replayed.verdict);
    };
    assert_eq!(at.len(), 1, "{at:?}");
    assert!(at[0].starts_with(&format!("case/{DIED_AT}: ")), "{at:?}");
    assert!(justified.is_empty(), "{justified:?}");

    // Counted like the others, and it does not loosen compliance: it is a distinct word in the
    // report, not a quiet pass hidden inside "exact".
    let mut report = Report::new();
    report.add(replayed);
    assert_eq!(report.counts().get("never-returned"), Some(&1));
    assert_eq!(report.counts().get("exact"), None);
    assert!(report.is_compliant());
    let rendered = report.render();
    assert!(rendered.contains("never-returned"), "{rendered}");
    assert!(
        rendered.contains("Metal did not return from the call, and we refused it"),
        "{rendered}"
    );

    // No allowance can reach it: a case that loosens the site whole still gets this verdict and
    // not `announced`, so the row cannot be dressed up as a justified difference.
    let unchecked = vec![Allowance::new(
        format!("case/{DIED_AT}"),
        Loosening::Unchecked,
        Justification::accepted(accepted_differences()[0].clone()),
    )];
    let found = compare(
        &case.stated(),
        &{
            let mut side = StubSide::new(Vec::new());
            replay_against(&case, &mut side, &recording).unwrap().ours
        },
        &unchecked,
        Offered::notices(&[]),
    );
    assert_eq!(found.never_returned.len(), 1, "{found:?}");
    assert!(found.announced.is_empty(), "{found:?}");
}

#[test]
fn answering_where_metal_did_not_return_is_divergent() {
    // The other half, and the reason the verdict is a mechanism and not a label: the one fact the
    // row holds is that *we* refuse where Metal went away. Answering instead has to fail.
    let case = DiedOnMetal { answers: true };
    let (runs, stated) = DiedOnMetal { answers: false }
        .carried_transcript()
        .unwrap()
        .unwrap();
    let recording = Recording::new(
        case.name(),
        Vec::new(),
        vec![crate::record::never_returned(
            MetalLevel::NEWEST,
            "the suite's own tests",
            DIED_AT,
            "the probe's device",
            runs,
            stated,
        )],
    );
    let mut side = StubSide::new(Vec::new());
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    let Verdict::Divergent { found } = &replayed.verdict else {
        panic!("{}", replayed.verdict);
    };
    assert!(
        found.iter().any(|divergence| matches!(
            divergence.kind,
            DivergenceKind::AnsweredWhereMetalDidNotReturn { .. }
        )),
        "{found:?}"
    );
    assert!(!replayed.verdict.is_compliant());
}

#[test]
fn record_mode_refuses_a_row_metal_does_not_return_from() {
    // It may never be produced by running the shape: making the call is what ends Metal's
    // process. `forbidden` carries the probe's own words, so `record` refuses before anything is
    // encoded, on a host with no Metal as on this one.
    let case = DiedOnMetal { answers: false };
    assert_eq!(case.metal_source().forbidden(), Some(DIED_SAYS));
    let mut side = StubSide::new(Vec::new());
    let refused = record(&case, &mut side, 1, MetalLevel::NEWEST, Validated::NotRun);
    assert_eq!(
        refused,
        Err(RecordError::Forbidden {
            case: case.name().to_owned(),
            reason: DIED_SAYS.to_owned(),
        }),
        "{refused:?}"
    );
    // And the answer that is written instead says so in the file: the provenance names the call,
    // and Metal's API validation was never asked either.
    let answer = crate::record::never_returned(
        MetalLevel::NEWEST,
        "the suite's own tests",
        DIED_AT,
        "the probe's device",
        1,
        case.stated(),
    );
    assert_eq!(answer.provenance.never_returned_from(), Some(DIED_AT));
    assert_eq!(answer.provenance.measured_elsewhere(), None);
    assert_eq!(answer.validated, Validated::NotRun);
}

#[test]
fn the_report_says_how_many_answers_came_from_another_chip() {
    // 53 recordings carried an Apple M4's values and `conformance cases` showed it only through a
    // `Debug` print, so a run could read entirely green while a fifth of its answers came from a
    // machine that is gone. The suite's own rule is that a recording of another oracle is stale;
    // a carried table is the one place that rule is relaxed, so the relaxation has to be counted
    // where a reader of the report sees it.
    use crate::recording::APPLE_M4;
    let carried = |case: &dyn Case, chip: &str| {
        let bytes = words(&[1, 2]);
        let transcript = WordsCase::new(case.name(), 2).transcript(bytes, "Completed", &[]);
        Recording::new(
            case.name(),
            Vec::new(),
            vec![RecordedAnswer {
                key: crate::recording::OracleKey::current(MetalLevel::NEWEST),
                provenance: Provenance::Carried {
                    from: "a probe this tree holds".to_owned(),
                    measured_on: chip.to_owned(),
                },
                device: "the probe's device".to_owned(),
                runs: 1,
                outcomes: vec![Outcome {
                    runs: 1,
                    transcript,
                }],
                validated: Validated::NotRun,
            }],
        )
    };
    let mut elsewhere = WordsCase::new("from-the-m4", 2);
    elsewhere.source = MetalSource::carried("a probe this tree holds").measured_on(APPLE_M4);
    let mut here = WordsCase::new("from-the-oracle", 2);
    here.source = MetalSource::carried("a probe this tree holds");

    let mut report = Report::new();
    let mut side = StubSide::new(words(&[1, 2]));
    let far = replay_against(&elsewhere, &mut side, &carried(&elsewhere, APPLE_M4)).unwrap();
    assert_eq!(far.verdict, Verdict::Exact, "{}", far.verdict);
    assert_eq!(far.measured_on.as_deref(), Some(APPLE_M4));
    report.add(far);
    let mut side = StubSide::new(words(&[1, 2]));
    let near = replay_against(
        &here,
        &mut side,
        &carried(&here, crate::recording::ORACLE_CHIP),
    )
    .unwrap();
    // An answer taken on the oracle names no other chip: the count is of the relaxation, not of
    // every carried row.
    assert_eq!(near.measured_on, None);
    report.add(near);

    assert_eq!(
        report.carried_from_another_chip(),
        [(APPLE_M4, 1)].into_iter().collect()
    );
    let rendered = report.render();
    assert!(rendered.contains("carried from another chip"), "{rendered}");
    assert!(
        rendered.contains(&format!("1 row(s) answered from {APPLE_M4}")),
        "{rendered}"
    );
    // The rows are still compliant: a carried answer from another chip is weaker evidence, not a
    // failure. Counting it is the whole mechanism.
    assert!(report.is_compliant());
}

// The cost half: what our side did to produce a case's transcript. Counts, never Metal's, and
// never a correctness verdict.

/// The cost half reads **every** count the runtime keeps, and fails loudly on a field it cannot.
///
/// `WorkCounts` has 386 fields of its own plus three nested count structs, and a hand-written list
/// of them would be stale the day a runtime change added one, so the reading comes out of the
/// derived `Debug`. The risk that buys is a field of another shape being dropped silently, which
/// would be a hole in the gate nobody could see — so a leaf that is not a count is an error here,
/// and the only fields left out are the ones named with their reason.
#[test]
fn the_cost_half_reads_every_count_the_runtime_keeps() {
    let counts = crate::cost::Counts::of(&metal2vulkan_runtime::WorkCounts::default())
        .expect("the runtime's counts read as counts");
    assert!(
        counts.len() > 350,
        "only {} counts were read, so the reading is not of the whole struct",
        counts.len()
    );
    // One count from each nesting level, so a change that flattened or renamed a level fails here.
    for name in [
        "queue_submissions",
        "variant_pipelines_made",
        "commands_replayed.cmd_draw",
        "object_churn.images_parked",
        "shadow_keeps.render_cut",
    ] {
        assert_eq!(counts.get(name), 0, "{name} was not read");
        assert!(
            counts.iter().any(|(held, _)| held == name),
            "{name} is not among the counts the cost half reads"
        );
    }
    // The site tables are left out, and nothing of them leaks in under another name.
    assert!(
        !counts.iter().any(|(name, _)| name.contains("copy_sites")),
        "a site table was read as a count"
    );
    assert!(crate::cost::excluded("copy_sites").is_some());
    assert!(crate::cost::excluded("row_copy_regions_most").is_some());
    assert!(crate::cost::excluded("batch_submit_nanos").is_some());
    assert!(crate::cost::excluded("queue_submissions").is_none());
}

/// A field of the runtime's counts that is not a count fails the reading by name.
///
/// This is the guard that makes the test above load-bearing: without it, a runtime field of a new
/// shape would simply not be counted, and the gate would quietly stop holding it.
#[test]
fn a_field_that_is_not_a_count_fails_the_reading() {
    let shape = "WorkCounts {\n    queue_submissions: 3,\n    a_new_flag: true,\n}";
    let why = crate::cost::Counts::read(shape).expect_err("a bool is not a count");
    assert!(why.contains("a_new_flag"), "{why}");
    assert!(why.contains("NOT_A_COUNT"), "{why}");
    // And the same shape with the field left out reads fine, so the failure is the field's.
    let read = crate::cost::Counts::read("WorkCounts {\n    queue_submissions: 3,\n}")
        .expect("a count reads");
    assert_eq!(read.get("queue_submissions"), 3);
}

/// Nesting, arrays and the site tables all come out of the reading as they went in.
#[test]
fn the_reading_follows_the_nesting_and_skips_the_site_tables() {
    let shape = "\
WorkCounts {
    queue_submissions: 2,
    commands_replayed: CommandsReplayed {
        cmd_draw: 7,
    },
    copy_sites: [
        CopySite {
            command: \"vkCmdCopyBuffer\",
            file: \"buffer.rs\",
            line: 10,
            calls: 1,
        },
        CopySite {
            command: \"\",
            file: \"\",
            line: 0,
            calls: 0,
        },
    ],
    render_passes_made: 1,
}";
    let read = crate::cost::Counts::read(shape).expect("the shape reads");
    assert_eq!(read.get("queue_submissions"), 2);
    assert_eq!(read.get("commands_replayed.cmd_draw"), 7);
    assert_eq!(read.get("render_passes_made"), 1);
    assert_eq!(read.len(), 3, "{:?}", read);
}

/// The three regions are the ones the cost regime names, and the device API is the residual.
#[test]
fn the_work_is_attributed_to_the_three_regions() {
    use crate::cost::{region_of, Region};
    for name in [
        "variant_pipelines_made",
        "shader_modules_made",
        "pipeline_layouts_shared",
        "graphics_pipelines_looked_up",
    ] {
        assert_eq!(region_of(name), Region::Translation, "{name}");
    }
    for name in [
        "commands_replayed.cmd_draw",
        "queue_submissions",
        "command_buffers_submitted",
        "scheduler_passes",
        "wakes_handler_threads",
        "completed_waits",
    ] {
        assert_eq!(region_of(name), Region::Submission, "{name}");
    }
    // What a call does on the calling thread, including the counts whose names mention a pipeline
    // or a submission but which are draws and binds.
    for name in [
        "draw_shapes_made",
        "object_churn.images_parked",
        "stage_tables_read",
        "fills_recorded",
        "provoking_last_pipeline_draws",
        "instance_splits_pending_pipeline",
    ] {
        assert_eq!(region_of(name), Region::DeviceApi, "{name}");
    }
}

/// A count the measurement never saw move is bounded at zero.
///
/// This is the whole point of the gate: a case that starts doing something it did not do before
/// reads as `Over` rather than as nothing at all.
#[test]
fn a_count_the_measurement_never_saw_is_bounded_at_zero() {
    let record = crate::cost::CostRecord {
        venue: "a venue".to_owned(),
        runs: 2,
        measured: "two runs of the whole selection".to_owned(),
        degraded: Vec::new(),
        bounds: [("queue_submissions".to_owned(), 1)].into_iter().collect(),
        moving: Default::default(),
    };
    let judge = |text: &str| {
        crate::cost::judge(
            std::slice::from_ref(&record),
            "a venue",
            true,
            &crate::cost::Counts::read(text).expect("the shape reads"),
            None,
            &[],
        )
    };
    // At the bound: within.
    assert_eq!(
        judge("WorkCounts {\n    queue_submissions: 1,\n}").word(),
        "within"
    );
    // Over the bound: over, naming the count, the region, the bound and what was found.
    let over = judge("WorkCounts {\n    queue_submissions: 2,\n}");
    assert_eq!(over.word(), "over");
    assert!(
        over.to_string().contains(
            "queue_submissions (submission) reached 2 where the \
             measurement's most was 1"
        ),
        "{over}"
    );
    // A count nothing bounded, which the measurement never saw: over, bounded at zero.
    let fresh = judge("WorkCounts {\n    queue_submissions: 1,\n    render_passes_made: 1,\n}");
    assert_eq!(fresh.word(), "over");
    assert!(fresh.to_string().contains("render_passes_made"), "{fresh}");
}

/// A count whose readings moved between runs of the same case on the same venue is reported, not
/// gated: a bound that fails at random is worse than no bound.
#[test]
fn a_count_whose_readings_moved_is_reported_and_not_gated() {
    let bound = |range: Option<u64>| crate::cost::CostRecord {
        venue: "a venue".to_owned(),
        runs: 3,
        measured: "three runs".to_owned(),
        degraded: Vec::new(),
        bounds: [("render_passes_made".to_owned(), 2)].into_iter().collect(),
        moving: range
            .map(|least| {
                [("render_passes_made".to_owned(), least)]
                    .into_iter()
                    .collect()
            })
            .unwrap_or_default(),
    };
    let found = crate::cost::Counts::read("WorkCounts {\n    render_passes_made: 9,\n}").unwrap();
    // Deterministic in the measurement: held, and 9 is over 2.
    let gated = bound(None);
    assert_eq!(
        crate::cost::judge(
            std::slice::from_ref(&gated),
            "a venue",
            true,
            &found,
            None,
            &[]
        )
        .word(),
        "over"
    );
    assert_eq!(gated.gated(), 1);
    // Seen to move between 0 and 2: reported, and 9 does not fail the run.
    let moving = bound(Some(0));
    assert_eq!(
        crate::cost::judge(
            std::slice::from_ref(&moving),
            "a venue",
            true,
            &found,
            None,
            &[]
        )
        .word(),
        "within"
    );
    assert_eq!(moving.gated(), 0);
}

/// Every way a run can fail to have a bound is its own `Unmeasured`, with the reason in words.
///
/// Four of them, and each matters: a venue the measurement was not taken on; a selection that is
/// not the whole case set, because a case's counts depend on what ran before it; a run that
/// stopped part way; and — the one the correctness half's own fix is about — a run the **device**
/// degraded otherwise than the measurement did, whose lower counts are the device's gap and not a
/// cost we saved.
#[test]
fn a_run_with_no_bound_that_stands_is_unmeasured_and_says_why() {
    let record = crate::cost::CostRecord {
        venue: "the measured venue".to_owned(),
        runs: 2,
        measured: "two runs".to_owned(),
        degraded: vec!["half-nan-quieted".to_owned()],
        bounds: [("queue_submissions".to_owned(), 1)].into_iter().collect(),
        moving: Default::default(),
    };
    let found = crate::cost::Counts::read("WorkCounts {\n    queue_submissions: 1,\n}").unwrap();
    let records = std::slice::from_ref(&record);
    let degraded = [String::from("half-nan-quieted")];
    let why = |verdict: crate::cost::CostVerdict| match verdict {
        crate::cost::CostVerdict::Unmeasured { why } => why,
        other => panic!("{other:?}"),
    };
    // The measurement's own shape: within.
    assert_eq!(
        crate::cost::judge(records, "the measured venue", true, &found, None, &degraded).word(),
        "within"
    );
    // Another venue.
    let other = why(crate::cost::judge(
        records,
        "another venue",
        true,
        &found,
        None,
        &degraded,
    ));
    assert!(other.contains("another venue"), "{other}");
    assert!(other.contains("the measured venue"), "{other}");
    // A selection rather than the whole case set.
    let part = why(crate::cost::judge(
        records,
        "the measured venue",
        false,
        &found,
        None,
        &degraded,
    ));
    assert!(part.contains("whole case set"), "{part}");
    // A run that stopped part way.
    let stopped = why(crate::cost::judge(
        records,
        "the measured venue",
        true,
        &found,
        Some("the venue lacks 2x multisampling"),
        &degraded,
    ));
    assert!(stopped.contains("2x multisampling"), "{stopped}");
    // A run the device degraded otherwise: the gap must not read as a win.
    let elsewhere = why(crate::cost::judge(
        records,
        "the measured venue",
        true,
        &found,
        None,
        &[],
    ));
    assert!(
        elsewhere.contains("half-nan-quieted") && elsewhere.contains("device's gap"),
        "{elsewhere}"
    );
}

/// The cost record rides in the recording, and a recording written before the cost half existed
/// still reads.
#[test]
fn the_cost_record_rides_in_the_recording() {
    let answer = RecordedAnswer {
        key: crate::recording::OracleKey::current(MetalLevel::NEWEST),
        provenance: Provenance::MetalRun,
        device: "a device".to_owned(),
        runs: 1,
        outcomes: vec![Outcome {
            runs: 1,
            transcript: Sites::new().finish(),
        }],
        validated: Validated::NotRun,
    };
    let costs = vec![crate::cost::CostRecord {
        venue: "a venue".to_owned(),
        runs: 2,
        measured: "two runs of the whole selection".to_owned(),
        degraded: Vec::new(),
        bounds: [("queue_submissions".to_owned(), 1)].into_iter().collect(),
        moving: Default::default(),
    }];
    let recording = Recording::new("a case", Vec::new(), vec![answer]).with_costs(costs.clone());
    let text = serde_json::to_string_pretty(&recording).expect("it writes");
    let read: Recording = serde_json::from_str(&text).expect("it reads back");
    assert_eq!(read.costs, costs);
    // A recording of the format as it was before the cost half holds no `costs` member at all: it
    // reads, with no cost record, which is `Unmeasured` and never a pass. So the member is left
    // out where it is empty, and the 127 recordings already committed stay byte for byte as they
    // are until a cost measurement is written into them.
    let mut without = recording.clone();
    without.costs = Vec::new();
    let text = serde_json::to_string(&without).expect("it writes");
    assert!(!text.contains("costs"), "an empty cost half is not written");
    let read: Recording = serde_json::from_str(&text).expect("it reads back");
    assert!(read.costs.is_empty());
}

/// A measurement turns into bounds that name it, and a count whose runs disagreed carries its
/// range.
#[test]
fn a_measurement_becomes_bounds_that_name_it() {
    let cost = crate::cost::CaseCost {
        case: "a case".to_owned(),
        counts: crate::cost::Counts::read(
            "WorkCounts {\n    queue_submissions: 2,\n    render_passes_made: 5,\n}",
        )
        .unwrap(),
        least: crate::cost::Counts::read(
            "WorkCounts {\n    queue_submissions: 2,\n    render_passes_made: 1,\n}",
        )
        .unwrap(),
        runs: 3,
        stopped: None,
        degraded: vec!["a-code".to_owned()],
        nanos: 1_000,
    };
    let record = cost.record("a venue", "three runs of the whole selection".to_owned());
    assert_eq!(record.venue, "a venue");
    assert_eq!(record.runs, 3);
    assert_eq!(record.measured, "three runs of the whole selection");
    assert_eq!(record.degraded, vec!["a-code".to_owned()]);
    let submissions = record.bound("queue_submissions").expect("it is bounded");
    assert_eq!((submissions.most, submissions.range), (2, None));
    assert!(submissions.gated());
    let passes = record.bound("render_passes_made").expect("it is bounded");
    assert_eq!((passes.most, passes.range), (5, Some(1)));
    assert!(!passes.gated());
    assert_eq!(record.gated(), 1);
    assert_eq!(cost.moved(), vec![("render_passes_made", 1, 5)]);
}

/// Re-taking a case's Metal answer keeps its cost record.
///
/// The two halves are measured by different legs — Metal's answer on this Mac, our cost on a venue
/// — so a re-record must not empty the cost gate. Without this, every re-recording would silently
/// turn the cost half off for the case it touched.
#[test]
fn re_recording_a_metal_answer_keeps_the_cost_record() {
    let held = crate::recording::read(crate::cases::pass_bytes::all()[0].name())
        .expect("a committed recording");
    let costs = vec![crate::cost::CostRecord {
        venue: "a venue".to_owned(),
        runs: 1,
        measured: "one run".to_owned(),
        degraded: Vec::new(),
        bounds: Default::default(),
        moving: Default::default(),
    }];
    let kept = Recording::new(
        held.case.clone(),
        held.shaders.clone(),
        held.answers.clone(),
    )
    .with_costs(costs.clone());
    assert_eq!(kept.costs, costs);
    // `record::write` reads what is committed and keeps its cost half, so this is the shape it
    // writes. The call itself is not made here: it would write over a committed recording.
    assert!(
        include_str!("record.rs").contains("let costs = recording::read(case.name())"),
        "record::write no longer keeps the cost half of the recording it replaces"
    );
}

/// A count that says which thread arrived first is reported and never gated, whatever a
/// measurement saw — and the count that holds its total still is.
///
/// Four measurements of the whole case set and three replays of it put every moving reading and
/// every overrun in these two families, so the rule is theirs and not a measurement's. The test
/// pins both halves of it: the race is ungated, and the deterministic total beside it is not.
#[test]
fn a_count_that_says_which_thread_arrived_first_is_never_gated() {
    use crate::cost::racy;
    for count in [
        "command_buffers_replayed",
        "command_buffers_replayed_inline",
        "object_churn.images_evicted",
        "object_churn.vk_images_destroyed",
        "object_churn.vk_images_made",
        "scheduler_passes",
        "scheduler_wakes_host",
        "wakes_spared",
        "wakes_handler_threads",
    ] {
        assert!(racy(count).is_some(), "{count}");
    }
    // Counts of work items rather than of thread arrivals, which held still over every round of
    // every measurement: one slot return and one completed wait a command buffer.
    for count in [
        "command_buffers_deferred",
        "command_buffers_submitted",
        "queue_submissions",
        "commands_replayed.cmd_draw",
        "render_passes_made",
        "slot_returns",
        "completed_waits",
    ] {
        assert!(racy(count).is_none(), "{count}");
    }
    // A record that bounds both at 1, with neither seen to move.
    let record = crate::cost::CostRecord {
        venue: "a venue".to_owned(),
        runs: 8,
        measured: "eight rounds".to_owned(),
        degraded: Vec::new(),
        bounds: [
            ("command_buffers_replayed_inline".to_owned(), 1),
            ("command_buffers_deferred".to_owned(), 1),
        ]
        .into_iter()
        .collect(),
        moving: Default::default(),
    };
    assert_eq!(record.gated(), 1, "only the total is gated");
    let judge = |text: &str| {
        crate::cost::judge(
            std::slice::from_ref(&record),
            "a venue",
            true,
            &crate::cost::Counts::read(text).expect("the shape reads"),
            None,
            &[],
        )
    };
    // The race going over its bound is not a failure.
    assert_eq!(
        judge(
            "WorkCounts {\n    command_buffers_deferred: 1,\n    \
             command_buffers_replayed_inline: 9,\n}"
        )
        .word(),
        "within"
    );
    // The total going over it is.
    let over = judge(
        "WorkCounts {\n    command_buffers_deferred: 2,\n    \
         command_buffers_replayed_inline: 1,\n}",
    );
    assert_eq!(over.word(), "over");
    assert!(
        over.to_string().contains("command_buffers_deferred"),
        "{over}"
    );
    // And a racy count the record never bounded at all is still not gated, which is the case that
    // sent this rule out of the measurement and into the code.
    assert_eq!(
        judge(
            "WorkCounts {\n    object_churn: ObjectChurn {\n        images_evicted: 6,\n    },\n}"
        )
        .word(),
        "within"
    );
}

/// Every field of a [`crate::render::ColorState`] reaches the color attachment descriptor the
/// runtime side gives the runtime (`runtime_side::color_descriptor`), checked field by field with
/// no device.
///
/// It is a guard against the bug that wrote it: the write mask was never assigned there, so a case
/// that asked for a mask of no channel got a pipeline that wrote every channel, and the family
/// asking what Metal's tile store leaves behind read as our implementation ignoring a legal Metal
/// state. The Metal side set the mask, so the two sides were asking two different questions and
/// nothing in the suite noticed.
#[test]
fn the_runtime_side_forwards_every_field_of_a_color_state() {
    use crate::render::{Blend, ColorState};
    use metal2vulkan::metal::{BlendFactor, BlendOperation, ColorWriteMask, PixelFormat};

    // The mask, at the three shapes that tell a dropped field from a kept one: none, some, all.
    for mask in [
        ColorWriteMask::NONE,
        ColorWriteMask::RED | ColorWriteMask::ALPHA,
        ColorWriteMask::ALL,
    ] {
        let made = crate::runtime_side::color_descriptor(
            &ColorState::new(PixelFormat::RGBA8Unorm).masked(mask),
        );
        assert_eq!(made.write_mask, mask, "the write mask {mask:?}");
        assert_eq!(made.pixel_format, PixelFormat::RGBA8Unorm);
        assert!(!made.blending_enabled, "no blend was asked for");
    }

    // The blend's six factors and operations, each a different value so that a field taking
    // another's value fails rather than passing by coincidence.
    let blend = Blend {
        source_rgb: BlendFactor::SourceColor,
        destination_rgb: BlendFactor::DestinationAlpha,
        rgb: BlendOperation::ReverseSubtract,
        source_alpha: BlendFactor::OneMinusSourceAlpha,
        destination_alpha: BlendFactor::BlendColor,
        alpha: BlendOperation::Min,
    };
    let made = crate::runtime_side::color_descriptor(
        &ColorState::new(PixelFormat::R16Snorm)
            .masked(ColorWriteMask::GREEN)
            .blended(blend),
    );
    assert_eq!(made.pixel_format, PixelFormat::R16Snorm);
    assert_eq!(made.write_mask, ColorWriteMask::GREEN);
    assert!(made.blending_enabled);
    assert_eq!(made.source_rgb_blend_factor, blend.source_rgb);
    assert_eq!(made.destination_rgb_blend_factor, blend.destination_rgb);
    assert_eq!(made.rgb_blend_operation, blend.rgb);
    assert_eq!(made.source_alpha_blend_factor, blend.source_alpha);
    assert_eq!(made.destination_alpha_blend_factor, blend.destination_alpha);
    assert_eq!(made.alpha_blend_operation, blend.alpha);
}
