//! A shared event's value only rises, from the host and from the GPU alike.
//!
//! `MTLSharedEvent` holds one 64-bit value. The host reads and writes it with `signaledValue` and
//! `setSignaledValue:`, the GPU raises it with `encodeSignalEvent:value:`, and the host waits for it
//! with `waitUntilSignaledValue:timeoutMS:`. A **lower** value, set or signalled, leaves it where it
//! was: the value is a high-water mark, not a register.
//!
//! The case walks that in one shape:
//!
//! - a new event's value;
//! - the host sets 5, then 3, then 5 — it reads 5, the 3 having changed nothing;
//! - a command buffer signals 3, then another signals 5 — it reads 5;
//! - a host wait for 3 and one for 5 answer, and one for 6 runs out;
//! - a command buffer signals `u64::MAX`, and a host wait for it answers.
//!
//! `runtime_lane::events` holds this as `a_shared_events_value_only_rises`, which asserts the
//! runtime agrees with itself. Here both sides answer and the answers are compared field by field,
//! so the numbers are Metal's rather than ours.
//!
//! # Why it is safe and quick
//!
//! The GPU's part is three command buffers that encode one signal each and nothing else: no shader,
//! no pass, no dispatch. Every wait of the case's own thread is bounded
//! ([`crate::shared_events::WAIT_MS`]), including the one that is meant to run out, and the two
//! bounded waits that are meant to answer are for values the event has already reached.
//!
//! The ordering facts of the event family — a wait that holds work until a signal arrives, and the
//! order two queues see — are not here. They need completed and scheduled handlers in the
//! vocabulary, which nothing has yet.

use crate::case::{Abort, Case, Input, MetalSource, Side};
use crate::shared_events::WAIT_MS;
use crate::transcript::{Node, Observed, Sites};

/// The case's name.
pub const RISING_VALUE: &str = "shared-event-rising-value";

/// The label the case sets, so `label` has something of ours to answer with.
const LABEL: &str = "conformance-rising-value";

/// The site the row sits at.
const AT: &str = "rising-value";

/// A case over what a shared event's value does.
pub struct RisingValue {
    name: &'static str,
}

/// The case.
pub fn rising_value() -> RisingValue {
    RisingValue { name: RISING_VALUE }
}

impl Case for RisingValue {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // What the case feeds: the label it sets, the values the host sets and the GPU signals in
        // order, the values it waits for, and the bound each wait is given. The bound is an input
        // because it decides what a wait's answer means.
        let mut words = Vec::new();
        for value in [5u64, 3, 5] {
            words.extend(value.to_le_bytes());
        }
        let mut signals = Vec::new();
        for value in [3u64, 5, u64::MAX] {
            signals.extend(value.to_le_bytes());
        }
        let mut waits = Vec::new();
        for value in [3u64, 5, 6, u64::MAX] {
            waits.extend(value.to_le_bytes());
        }
        Ok(vec![
            Input::bytes("label", LABEL.as_bytes()),
            Input::bytes("host-sets", &words),
            Input::bytes("gpu-signals", &signals),
            Input::bytes("host-waits-for", &waits),
            Input::bytes("wait-bound-millis", &WAIT_MS.to_le_bytes()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        match side.shared_event_rises(LABEL) {
            Ok(rise) => sites.push(rise.node(AT)),
            // The row records the failure and the run ends: every fact the case has is in the one
            // call, so there is nothing after it to read.
            Err(failure) => sites.push(Node::leaf(AT, Observed::value(failure.value))),
        }
        Ok(())
    }
}
