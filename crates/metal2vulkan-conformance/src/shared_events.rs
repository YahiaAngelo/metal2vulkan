//! What a shared event's value does: the vocabulary for the event family's first case.
//!
//! `MTLSharedEvent`'s value **only rises**, from the host and from the GPU alike: a lower value set
//! or signalled leaves it where it was. A host wait for a value the event has reached answers at
//! once, and one for a value above it runs out.
//!
//! Every fact here is a value or a boolean, so the record is a small set of fields rather than
//! bytes, and the GPU's part is three command buffers that encode one signal each and nothing else.

use crate::transcript::{Node, NodeName, Observed, Value};

/// How long a host wait is given before it counts as having run out.
///
/// Short on purpose: the waits a case makes are for values the event has **already** reached, which
/// answer at once, and for one it has not, which never will. A long bound would only make the case
/// slow.
pub const WAIT_MS: u64 = 10;

/// What one run of the rising-value probe observed. Each field is one call's answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventRise {
    /// `label`, after `setLabel:`.
    pub label: Option<String>,
    /// `signaledValue` of a new event.
    pub initial: u64,
    /// `signaledValue` after the host set 5, then 3, then 5.
    pub after_host_sets: u64,
    /// `signaledValue` after a command buffer signalled 3, and then after one signalled 5.
    pub after_gpu_signals: [u64; 2],
    /// `waitUntilSignaledValue:timeoutMS:` for 3, for 5 and for 6, each with [`WAIT_MS`].
    pub waits: [bool; 3],
    /// `signaledValue` after a command buffer signalled `u64::MAX`.
    pub after_max_signal: u64,
    /// Whether the host wait for `u64::MAX` answered.
    pub waited_for_max: bool,
}

impl EventRise {
    /// The observations as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let label = match &self.label {
            Some(label) => Value::Variant {
                name: "Some".to_owned(),
                payload: Box::new(Value::Lines(vec![label.clone()])),
            },
            None => Value::unit_variant("None"),
        };
        let mut node = Node::branch(name, Vec::new())
            .with(Node::leaf("label", Observed::value(label)))
            .with(Node::leaf(
                "signaled-value/new",
                Observed::value(Value::Uint(self.initial)),
            ))
            .with(Node::leaf(
                "signaled-value/after-host-sets",
                Observed::value(Value::Uint(self.after_host_sets)),
            ));
        for (at, (value, signalled)) in self.after_gpu_signals.iter().zip([3u64, 5]).enumerate() {
            node = node.with(Node::leaf(
                format!("signaled-value/after-gpu-signal-{at}-of-{signalled}"),
                Observed::value(Value::Uint(*value)),
            ));
        }
        for (answered, wanted) in self.waits.iter().zip([3u64, 5, 6]) {
            node = node.with(Node::leaf(
                format!("wait-until-signaled-value/{wanted}"),
                Observed::value(Value::Bool(*answered)),
            ));
        }
        node.with(Node::leaf(
            "signaled-value/after-gpu-signal-of-u64-max",
            Observed::value(Value::Uint(self.after_max_signal)),
        ))
        .with(Node::leaf(
            "wait-until-signaled-value/u64-max",
            Observed::value(Value::Bool(self.waited_for_max)),
        ))
    }
}
