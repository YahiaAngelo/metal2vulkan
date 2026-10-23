//! Field-by-field comparison of two transcripts, exact by default.
//!
//! The walk pairs the sides' nodes by name and descends. At a leaf it compares the value by its own
//! kind: bytes element by element under their layout, lines in order, sets without order, a variant
//! by its name and then its payload. A site a case loosened compares under its allowance, and only
//! once the allowance's justification holds ([`crate::allowance`]).
//!
//! Where Metal answered a call and we returned a typed error, the site is a refusal rather than a
//! divergence: the owner's rule keeps that outcome visible instead of burying it in a failure
//! count.

use crate::allowance::{Allowance, Loosening, Unjustified};
use crate::diff::{Divergence, DivergenceKind, Side};
use crate::transcript::{ByteLayout, Gaps, Node, Observed, Shape, Transcript, Value};

/// A site that differed inside an allowance that holds.
#[derive(Clone, Debug, PartialEq)]
pub struct Announced {
    /// The site, as node names from the root.
    pub path: String,
    /// What the case loosened there.
    pub how: Loosening,
    /// What justifies it, in words.
    pub justified_by: String,
}

/// A site where Metal answered a call and we returned a typed error.
#[derive(Clone, Debug, PartialEq)]
pub struct Refusal {
    /// The site, as node names from the root.
    pub path: String,
    /// Our typed error, as the report prints it.
    pub ours: String,
}

/// An allowance that does not hold: the case names it, and why it cannot be used.
#[derive(Clone, Debug, PartialEq)]
pub struct BadAllowance {
    /// The allowance's path, as the case wrote it.
    pub path: String,
    pub why: Unjustified,
}

/// What comparing one run against its recording found.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Comparison {
    pub announced: Vec<Announced>,
    pub divergences: Vec<Divergence>,
    pub refusals: Vec<Refusal>,
    /// Allowances whose justification is gone. Each of these fails the case on its own: a
    /// difference nobody is accounting for.
    pub bad_allowances: Vec<BadAllowance>,
}

impl Comparison {
    /// Whether every field matched exactly and every allowance held unused.
    pub fn is_exact(&self) -> bool {
        self.announced.is_empty()
            && self.divergences.is_empty()
            && self.refusals.is_empty()
            && self.bad_allowances.is_empty()
    }
}

/// Compare `ours` against the recorded `metal`, under a case's `allowances`, with `announced` the
/// degrade-notice codes our run delivered.
pub fn compare(
    metal: &Transcript,
    ours: &Transcript,
    allowances: &[Allowance],
    announced: &[String],
) -> Comparison {
    compare_run(metal, ours, allowances, announced, false)
}

/// [`compare`] where `stopped` says the run stopped at a call that did not answer.
///
/// A run that stopped reached fewer sites than Metal did, and the sites it never reached are not
/// differences of their own: the refused call is the fact, and burying it under one divergence per
/// site it never got to would hide it.
pub fn compare_run(
    metal: &Transcript,
    ours: &Transcript,
    allowances: &[Allowance],
    announced: &[String],
    stopped: bool,
) -> Comparison {
    let mut found = Comparison::default();
    // Every allowance names a justification that must exist, whether or not the allowance fires.
    // A case that keeps an allowance for a notice kind or a documented difference that has since
    // gone fails here rather than comparing under it.
    for allowance in allowances {
        let missing = match &allowance.justified_by {
            justification @ crate::allowance::Justification::Degradation(code) => {
                match justification.unjustified(std::slice::from_ref(code)) {
                    // The run's own notices are checked where the allowance fires, not here.
                    Some(Unjustified::NoticeNotAnnounced(_)) | None => None,
                    other => other,
                }
            }
            justification => justification.unjustified(announced),
        };
        if let Some(why) = missing {
            found.bad_allowances.push(BadAllowance {
                path: allowance.path.clone(),
                why,
            });
        }
    }
    let mut walk = Walk {
        allowances,
        announced,
        stopped,
        found: &mut found,
    };
    walk.node("", &metal.root, &ours.root);
    found
}

/// The comparison's state as it descends.
struct Walk<'a> {
    allowances: &'a [Allowance],
    announced: &'a [String],
    /// Whether the run stopped before the end: a site only Metal reached is then no divergence.
    stopped: bool,
    found: &'a mut Comparison,
}

impl Walk<'_> {
    /// Compare one pair of sites, under `parent`'s path, and descend.
    fn node(&mut self, parent: &str, metal: &Node, ours: &Node) {
        let path = if parent.is_empty() {
            metal.name.clone()
        } else {
            format!("{parent}/{}", metal.name)
        };
        self.observed(&path, &metal.observed, &ours.observed);
        for child in &metal.children {
            match ours.child(&child.name) {
                Some(theirs) => self.node(&path, child, theirs),
                None if self.stopped => {}
                None => self.found.divergences.push(Divergence::new(
                    format!("{path}/{}", child.name),
                    DivergenceKind::Site {
                        held_by: Side::Metal,
                    },
                )),
            }
        }
        for child in &ours.children {
            if metal.child(&child.name).is_none() {
                self.found.divergences.push(Divergence::new(
                    format!("{path}/{}", child.name),
                    DivergenceKind::Site {
                        held_by: Side::Ours,
                    },
                ));
            }
        }
    }

    /// Compare what one site observed.
    fn observed(&mut self, path: &str, metal: &Observed, ours: &Observed) {
        if let (Some(recorded), Some(taken)) = (&metal.digest, &ours.digest) {
            if recorded != taken {
                self.found.divergences.push(Divergence::new(
                    path,
                    DivergenceKind::Digest {
                        metal: recorded.hex.clone(),
                        ours: taken.hex.clone(),
                    },
                ));
                return;
            }
        }
        if metal.is_absent() {
            // A recording taken digest-first holds the digest alone: the digests matched above,
            // and there is no recorded value to compare against.
            return;
        }
        if ours.is_absent() && metal.digest.is_some() {
            return;
        }
        match (&metal.shape, &ours.shape) {
            (Shape::Bytes(layout), _) | (_, Shape::Bytes(layout)) => {
                self.bytes(path, layout, &metal.value, &ours.value);
            }
            _ => self.value(path, &metal.value, &ours.value),
        }
    }

    /// Compare a byte region element by element, and its gaps where the layout asks.
    fn bytes(&mut self, path: &str, layout: &ByteLayout, metal: &Value, ours: &Value) {
        if self.allowed(path, metal, ours) {
            return;
        }
        let (Value::Bytes(recorded), Value::Bytes(read)) = (metal, ours) else {
            self.mismatched_kind(path, metal, ours);
            return;
        };
        if recorded.len() != read.len() {
            self.found.divergences.push(Divergence::new(
                path,
                DivergenceKind::Length {
                    metal: recorded.len(),
                    ours: read.len(),
                },
            ));
            return;
        }
        let size = layout.element.size() as usize;
        for image in 0..layout.images {
            for row in 0..layout.rows {
                for index in 0..layout.row_elements {
                    let at = layout.at(image, row, index);
                    let Some((expected, actual)) = recorded
                        .get(at..at + size)
                        .zip(read.get(at..at + size))
                        .filter(|(expected, actual)| expected != actual)
                    else {
                        continue;
                    };
                    let Some((expected, actual)) = layout
                        .element
                        .read(expected)
                        .zip(layout.element.read(actual))
                    else {
                        continue;
                    };
                    if self.allowed(path, &expected, &actual) {
                        continue;
                    }
                    self.found.divergences.push(Divergence::new(
                        path,
                        DivergenceKind::Element {
                            element: layout.element,
                            image,
                            row,
                            index,
                            at,
                            metal: expected,
                            ours: actual,
                        },
                    ));
                    return;
                }
            }
        }
        if layout.gaps == Gaps::Compared {
            self.gaps(path, layout, recorded, read);
        }
    }

    /// Compare the bytes a layout's elements do not cover: the padding after each row's elements
    /// and after each image's rows.
    fn gaps(&mut self, path: &str, layout: &ByteLayout, metal: &[u8], ours: &[u8]) {
        let covered = |at: usize| -> bool {
            let image = at % layout.image_pitch.max(1) as usize;
            let row = image % layout.row_pitch.max(1) as usize;
            image < layout.rows as usize * layout.row_pitch as usize
                && row < layout.row_elements as usize * layout.element.size() as usize
        };
        for (at, (expected, actual)) in metal.iter().zip(ours).enumerate() {
            if expected != actual && !covered(at) {
                self.found.divergences.push(Divergence::new(
                    path,
                    DivergenceKind::Gap {
                        at,
                        metal: *expected,
                        ours: *actual,
                    },
                ));
                return;
            }
        }
    }

    /// Compare two values by their kind.
    fn value(&mut self, path: &str, metal: &Value, ours: &Value) {
        if metal == ours {
            return;
        }
        // An allowance over the whole site holds whatever kind of value moved there, so a case can
        // justify a notice it announced or a set that gained a member, not only a number that
        // drifted.
        if self.allowed(path, metal, ours) {
            return;
        }
        match (metal, ours) {
            (Value::Nothing(_), _) | (_, Value::Nothing(_)) => {
                let fact = |value: &Value| match value {
                    Value::Nothing(fact) => Some(*fact),
                    _ => None,
                };
                self.found.divergences.push(Divergence::new(
                    path,
                    DivergenceKind::Negative {
                        metal: fact(metal),
                        ours: fact(ours),
                    },
                ));
            }
            (Value::Lines(recorded), Value::Lines(read)) => {
                let at = recorded
                    .iter()
                    .zip(read)
                    .position(|(expected, actual)| expected != actual)
                    .unwrap_or(recorded.len().min(read.len()));
                self.found.divergences.push(Divergence::new(
                    path,
                    DivergenceKind::Line {
                        at,
                        metal: recorded.get(at).cloned(),
                        ours: read.get(at).cloned(),
                    },
                ));
            }
            (Value::Set(recorded), Value::Set(read)) => {
                self.found.divergences.push(Divergence::new(
                    path,
                    DivergenceKind::Set {
                        missing: recorded.difference(read).cloned().collect(),
                        extra: read.difference(recorded).cloned().collect(),
                        common: recorded.intersection(read).count(),
                    },
                ));
            }
            (Value::List(recorded), Value::List(read)) => {
                if recorded.len() != read.len() {
                    self.found.divergences.push(Divergence::new(
                        path,
                        DivergenceKind::ListLength {
                            metal: recorded.len(),
                            ours: read.len(),
                        },
                    ));
                    return;
                }
                for (at, (expected, actual)) in recorded.iter().zip(read).enumerate() {
                    self.value(&format!("{path}/{at}"), expected, actual);
                }
            }
            (
                Value::Variant {
                    name: recorded,
                    payload: expected,
                },
                Value::Variant {
                    name: read,
                    payload: actual,
                },
            ) => {
                if recorded == read {
                    self.value(&format!("{path}/{read}"), expected, actual);
                } else if recorded == "Ok" && read == "Err" {
                    self.found.refusals.push(Refusal {
                        path: path.to_owned(),
                        ours: crate::diff::print(actual),
                    });
                } else {
                    self.found.divergences.push(Divergence::new(
                        path,
                        DivergenceKind::Variant {
                            metal: recorded.clone(),
                            ours: read.clone(),
                        },
                    ));
                }
            }
            (Value::Fields(recorded), Value::Fields(read)) => {
                for (name, expected) in recorded {
                    match read.get(name) {
                        Some(actual) => self.value(&format!("{path}/{name}"), expected, actual),
                        None => self.found.divergences.push(Divergence::new(
                            path,
                            DivergenceKind::Field {
                                name: name.clone(),
                                held_by: Side::Metal,
                            },
                        )),
                    }
                }
                for name in read.keys() {
                    if !recorded.contains_key(name) {
                        self.found.divergences.push(Divergence::new(
                            path,
                            DivergenceKind::Field {
                                name: name.clone(),
                                held_by: Side::Ours,
                            },
                        ));
                    }
                }
            }
            _ if metal.kind() != ours.kind() => self.mismatched_kind(path, metal, ours),
            _ => {
                if !self.allowed(path, metal, ours) {
                    self.found.divergences.push(Divergence::new(
                        path,
                        DivergenceKind::Value {
                            metal: metal.clone(),
                            ours: ours.clone(),
                        },
                    ));
                }
            }
        }
    }

    /// Note that the sides observed different kinds of value at one site.
    fn mismatched_kind(&mut self, path: &str, metal: &Value, ours: &Value) {
        self.found.divergences.push(Divergence::new(
            path,
            DivergenceKind::Kind {
                metal: metal.kind(),
                ours: ours.kind(),
            },
        ));
    }

    /// Whether an allowance of the case covers `path` and holds for this difference. An allowance
    /// whose notice the run did not announce does not hold, and the case fails by it.
    fn allowed(&mut self, path: &str, metal: &Value, ours: &Value) -> bool {
        let Some(allowance) = self
            .allowances
            .iter()
            .find(|allowance| allowance.covers(path) && allowance.within(metal, ours))
        else {
            return false;
        };
        if let Some(why) = allowance.justified_by.unjustified(self.announced) {
            self.found.bad_allowances.push(BadAllowance {
                path: allowance.path.clone(),
                why,
            });
            return false;
        }
        self.found.announced.push(Announced {
            path: path.to_owned(),
            how: allowance.loosening.clone(),
            justified_by: allowance.justified_by.describe(),
        });
        true
    }
}
