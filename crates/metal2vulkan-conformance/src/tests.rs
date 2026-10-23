//! Tests of the suite's own machinery, with no device: a stub side stands in for a GPU, so the
//! transcript, the comparison, the justification rule, the verdicts, the recording's staleness and
//! the report are all checked where there is neither Metal nor Vulkan.

use crate::allowance::{accepted_differences, accepted_slug, Allowance, Justification, Loosening};
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
    // Metal's validation let the case run, so nothing excuses the refusal.
    assert_eq!(*metal_validation, None);
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
    let found = compare(&metal, &ours, &[], &[]);
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
        &[],
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
    let found = compare(&metal, &ours, &[], &[]);
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
    let found = compare(&tree(Value::Uint(1)), &tree(Value::Uint(2)), &[], &[]);
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
        let found = compare(&metal, &ours, &[], &[]);
        assert_eq!(found.divergences.len(), divergences, "{gaps:?}");
    }
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
    assert!(compare(&digested(b"same"), &ours(b"same"), &[], &[]).is_exact());
    let found = compare(&digested(b"same"), &ours(b"other"), &[], &[]);
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
