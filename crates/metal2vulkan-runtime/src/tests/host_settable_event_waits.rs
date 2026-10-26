//! A wait for a value the host may set is never handed to Vulkan (`event_timeline.rs`,
//! `event.rs`).
//!
//! Metal's `MTLSharedEvent` carries one value the host and the device both see, so
//! `setSignaledValue:` releases a command buffer's `encodeWaitForEvent:` as soon as the value is
//! reached, whatever the device is running. A Vulkan wait a batch was submitted with cannot be
//! released by the host: the value it waits for is the signal index of a GPU signal submitted
//! before it, and signalling that index from the host would make that signal's own
//! `vkQueueSubmit` illegal ("must have a value greater than the current value of the semaphore
//! when the semaphore signal operation is executed",
//! `VUID-vkQueueSubmit-pSignalSemaphores-03242`). So such a wait is held until the value is
//! reached and then goes with no Vulkan wait at all, and only a plain event's (`MTLEvent`, whose
//! value no host call sets) chains on the `VkQueue`.

use crate::event_timeline::{EventTimeline, Signal, WaitFor};

/// Submit a GPU signal of `value` on `queue`, kept when it is.
fn submit(timeline: &mut EventTimeline<usize>, value: u64, queue: usize) {
    if let Signal::Kept { index, .. } = timeline.signal(value, queue) {
        timeline.keep(index, value, queue);
    }
}

/// The shape a consumer measured: a long command buffer's signal is pending, a later command
/// buffer waits for a value it reaches, and the host provides that value. Metal lets the waiter
/// go at the set; a Vulkan wait for the signal's index would hold it for the whole of the long
/// command buffer's work.
#[test]
fn a_wait_the_host_may_satisfy_is_held_instead_of_handed_to_vulkan() {
    let mut timeline: EventTimeline<usize> = EventTimeline::new(true);
    submit(&mut timeline, 0x9607, 0);
    // Submitted and pending, so the value is known, and yet no batch may be handed a wait for it.
    assert_eq!(timeline.known(), 0x9607);
    for value in [1, 0x482, 0x9606, 0x9607] {
        assert_eq!(timeline.wait(value), None, "{value:#x} handed to Vulkan");
    }
    // The host sets it: every wait it reaches goes at once, with no Vulkan wait.
    assert!(timeline.set(0x9607));
    for value in [1, 0x482, 0x9606, 0x9607] {
        assert_eq!(timeline.wait(value), Some(WaitFor::Reached), "{value:#x}");
    }
    // One above the set is held again; the GPU signal still reaches only its own value.
    assert_eq!(timeline.wait(0x9608), None);
    assert!(!timeline.observe(1));
    assert_eq!(timeline.value(), 0x9607);
}

/// The same wait, reached by the GPU signal instead: the batch is held until it is observed, and
/// then needs no Vulkan wait either.
#[test]
fn a_held_wait_goes_once_the_gpu_signal_is_observed() {
    let mut timeline: EventTimeline<usize> = EventTimeline::new(true);
    submit(&mut timeline, 100, 0);
    assert_eq!(timeline.wait(50), None);
    assert!(timeline.observe(1));
    assert_eq!(timeline.value(), 100);
    assert_eq!(timeline.wait(50), Some(WaitFor::Reached));
    assert_eq!(timeline.wait(100), Some(WaitFor::Reached));
    assert_eq!(timeline.wait(101), None);
}

/// A plain event's value no host call sets, so its wait is exact on the `VkQueue` and keeps
/// chaining there: the hold costs a scheduler pass and buys nothing.
#[test]
fn a_plain_event_s_wait_still_chains_on_the_vk_queue() {
    let mut timeline: EventTimeline<usize> = EventTimeline::new(false);
    submit(&mut timeline, 3, 0);
    submit(&mut timeline, 8, 0);
    assert_eq!(timeline.wait(0), Some(WaitFor::Reached));
    assert_eq!(timeline.wait(3), Some(WaitFor::Index(1)));
    assert_eq!(timeline.wait(4), Some(WaitFor::Index(2)));
    assert_eq!(timeline.wait(8), Some(WaitFor::Index(2)));
    // Beyond what is submitted, either kind is held.
    assert_eq!(timeline.wait(9), None);
}

/// Holding changes nothing about the signals themselves: Metal's maximum drops a signal not above
/// what is known whichever kind the event is, and the kept ones take the same indices.
#[test]
fn the_signals_of_either_kind_take_the_same_indices() {
    for settable in [false, true] {
        let mut timeline: EventTimeline<usize> = EventTimeline::new(settable);
        assert_eq!(
            timeline.signal(4, 0),
            Signal::Kept {
                index: 1,
                after: None
            }
        );
        timeline.keep(1, 4, 0);
        assert_eq!(timeline.signal(4, 1), Signal::Dropped);
        assert_eq!(
            timeline.signal(5, 1),
            Signal::Kept {
                index: 2,
                after: Some(1)
            }
        );
    }
}
