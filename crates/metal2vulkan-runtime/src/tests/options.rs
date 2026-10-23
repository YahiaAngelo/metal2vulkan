//! A pipeline state creation's options (`pipeline::options`): Metal's bit values, every raw bit
//! kept, and a budget whose deadline only ever comes earlier.

use crate::{PipelineOption, PipelineOptions};
use metal2vulkan::CancelToken;
use std::time::{Duration, Instant};

/// `MTLPipelineOption`'s values in `MTLDevice.h` (the macOS 26.5 and 27.0 SDKs): `None` 0,
/// `BindingInfo` (and the deprecated `ArgumentInfo`) `1 << 0`, `BufferTypeInfo` `1 << 1`,
/// `FailOnBinaryArchiveMiss` `1 << 2`.
#[test]
fn the_defined_bits_are_metals() {
    assert_eq!(PipelineOption::NONE.raw(), 0);
    assert_eq!(PipelineOption::BINDING_INFO.raw(), 1 << 0);
    assert_eq!(PipelineOption::BUFFER_TYPE_INFO.raw(), 1 << 1);
    assert_eq!(PipelineOption::FAIL_ON_BINARY_ARCHIVE_MISS.raw(), 1 << 2);
    assert_eq!(PipelineOption::default(), PipelineOption::NONE);
    assert_eq!(PipelineOptions::default().options, PipelineOption::NONE);
}

/// Metal accepts every raw value (measured on an M3 Max and an M4), so every bit is kept.
#[test]
fn undefined_bits_are_kept() {
    for raw in [8, 16, 1 << 10, 1 << 31, 1 << 40, u64::MAX] {
        assert_eq!(PipelineOption::from_raw(raw).raw(), raw);
    }
    let both = PipelineOption::FAIL_ON_BINARY_ARCHIVE_MISS | PipelineOption::from_raw(1 << 40);
    assert_eq!(both.raw(), (1 << 40) | 4);
    assert_eq!(PipelineOptions::new(both).options, both);
}

#[test]
fn a_deadline_only_shortens_the_standard_budget() {
    let standard = Duration::from_secs(20);
    let before = Instant::now();
    let budget = PipelineOptions::default().budget();
    let after = Instant::now();
    let deadline = budget.deadline.expect("the standard deadline");
    assert!(before + standard <= deadline && deadline <= after + standard);
    assert!(budget.cancel.is_none());

    let mut options = PipelineOptions::default();
    let soon = Instant::now() + Duration::from_secs(1);
    options.deadline = Some(soon);
    assert_eq!(
        options.budget().deadline,
        Some(soon),
        "an earlier deadline holds"
    );
    options.deadline = Some(Instant::now() + Duration::from_secs(600));
    let after = Instant::now();
    let deadline = options.budget().deadline.expect("a deadline");
    assert!(deadline <= Instant::now() + standard && deadline >= after + standard);

    let token = CancelToken::new();
    options.cancel = Some(token.clone());
    let budget = options.budget();
    token.cancel();
    assert!(
        budget.cancel.expect("the token").is_cancelled(),
        "the budget shares it"
    );
}

/// **Every non-zero `MTLPipelineOption` value asks for reflection, `FailOnBinaryArchiveMiss`
/// alone included.**
///
/// Measured on an M3 Max under macOS 27.0.1, over all eight combinations of the three bits Apple's
/// header names: `newComputePipelineStateWithFunction:options:reflection:error:` filled its
/// `reflection` out-parameter for all seven non-zero values and left it nil only for
/// `MTLPipelineOptionNone`. The conformance suite's `pipeline_options` family is the recording.
///
/// The row that matters is `FAIL_ON_BINARY_ARCHIVE_MISS` on its own: Apple's documentation says
/// nothing about reflection for that bit, and reading only the two information bits answered no
/// reflection where Metal answers one, which is a reflection a consumer asked for and did not get.
#[test]
fn every_named_bit_asks_for_reflection_and_none_alone_does_not() {
    assert!(!PipelineOption::NONE.reflects(), "none reflects nothing");
    assert!(PipelineOption::BINDING_INFO.reflects());
    assert!(PipelineOption::BUFFER_TYPE_INFO.reflects());
    assert!(
        PipelineOption::FAIL_ON_BINARY_ARCHIVE_MISS.reflects(),
        "the archive-miss bit alone: Metal fills the reflection here (measured)"
    );
    for raw in 1..8u64 {
        assert!(
            PipelineOption::from_raw(raw).reflects(),
            "{raw:#b} is a non-zero value of the three named bits"
        );
    }
    // An undefined bit on its own is not measured, so it asks for nothing.
    assert!(!PipelineOption::from_raw(1 << 40).reflects());
}
