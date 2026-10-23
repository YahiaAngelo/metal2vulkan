//! Field-by-field comparison of two transcripts, exact by default.
//!
//! The walk pairs the sides' nodes by name and descends. At a leaf it compares the value by its own
//! kind: bytes element by element under their layout, lines in order, sets without order, a variant
//! by its name and then its payload. A site a case loosened compares under its allowance, and only
//! once the allowance's justification holds ([`crate::allowance`]).
//!
//! Where Metal answered a call and we returned a typed error, the site is a **refusal**, and no
//! allowance can make it anything else. A refusal is its own verdict so that it stays visible
//! rather than being buried in a failure count, and it is never compliant, because plain Metal is
//! the behavioural oracle and the owner's standing rule is that what Metal allows, we must allow.
//! A case may name what *documents* its refusal — Metal's own API validation refusing the same
//! call is the one thing that can — and the justification is still checked, but it buys an
//! explanation in the report and not a pass. See [`crate::validated`].

use crate::allowance::{Allowance, Loosening, Offered, Unjustified};
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
    /// How many sites, or elements of one site, the allowance covered.
    pub count: usize,
    /// The largest difference the allowance let through, where the values are numbers: so a bound
    /// is read off a run rather than guessed, and a bound far above what the device needs shows as
    /// one.
    pub largest: Option<f64>,
}

/// A site where Metal answered a call and we returned a typed error.
///
/// Never compliant, whatever the case names at the site. A refusal of something plain Metal does
/// is a gap of ours, and the suite's job is to say so rather than to launder it.
#[derive(Clone, Debug, PartialEq)]
pub struct Refusal {
    /// The site, as node names from the root.
    pub path: String,
    /// Our typed error, as the report prints it.
    pub ours: String,
    /// What the case named to **document** the refusal, where it named something and that thing
    /// holds ([`Allowance::justified_by`]): the words of Metal's own API validation refusing the
    /// same call, typically. It explains the row; it does not excuse it. `None` where the case
    /// named nothing, which is a refusal nobody has accounted for at all.
    pub documented_by: Option<String>,
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
    /// Note that the allowance at `path` does not hold, unless the same fault is already held.
    ///
    /// The same allowance is examined twice — once before the walk, so an allowance naming a
    /// justification that is gone fails whether or not it fires, and again where it does fire —
    /// and one fault reported twice reads as two.
    fn note_bad(&mut self, path: &str, why: Unjustified) {
        if self
            .bad_allowances
            .iter()
            .any(|held| held.path == path && held.why == why)
        {
            return;
        }
        self.bad_allowances.push(BadAllowance {
            path: path.to_owned(),
            why,
        });
    }

    /// Whether every field matched exactly and every allowance held unused.
    pub fn is_exact(&self) -> bool {
        self.announced.is_empty()
            && self.divergences.is_empty()
            && self.refusals.is_empty()
            && self.bad_allowances.is_empty()
    }
}

/// Compare `ours` against the recorded `metal`, under a case's `allowances`, with `offered` what
/// the run and its recording offer to justify a difference.
pub fn compare(
    metal: &Transcript,
    ours: &Transcript,
    allowances: &[Allowance],
    offered: Offered<'_>,
) -> Comparison {
    compare_run(metal, ours, allowances, offered, false)
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
    offered: Offered<'_>,
    stopped: bool,
) -> Comparison {
    let mut found = Comparison::default();
    // Every allowance names a justification that must exist, whether or not the allowance fires.
    // A case that keeps an allowance for a notice kind, a documented difference or a Metal
    // validation check that has since gone fails here rather than comparing under it.
    for allowance in allowances {
        let missing = match &allowance.justified_by {
            justification @ crate::allowance::Justification::Degradation(code) => {
                let announced = std::slice::from_ref(code);
                match justification.unjustified(&Offered::new(announced, offered.metal_validation))
                {
                    // The run's own notices are checked where the allowance fires, not here.
                    Some(Unjustified::NoticeNotAnnounced(_)) | None => None,
                    other => other,
                }
            }
            justification => justification.unjustified(&offered),
        };
        if let Some(why) = missing {
            found.note_bad(&allowance.path, why);
        }
    }
    let mut walk = Walk {
        allowances,
        offered,
        stopped,
        found: &mut found,
    };
    walk.node("", &metal.root, &ours.root);
    found
}

/// Whether the sides' answers at one site are Metal answering a call and us refusing it.
fn refused(metal: &Value, ours: &Value) -> bool {
    matches!(
        (metal, ours),
        (
            Value::Variant { name: recorded, .. },
            Value::Variant { name: read, .. },
        ) if recorded == "Ok" && read == "Err"
    )
}

/// The comparison's state as it descends.
struct Walk<'a> {
    allowances: &'a [Allowance],
    /// What the run and its recording offer to justify a difference.
    offered: Offered<'a>,
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
                // A site a case loosened compares under its allowance here too. Without this a
                // digest-only site could never be loosened, and a row whose Metal answer is a
                // digest the device announced it cannot match -- a fill of depths outside
                // [0, 1] on a device whose depth copies go by a draw -- would have no honest
                // shape in the suite at all.
                let hex =
                    |digest: &crate::transcript::Digest| Value::Lines(vec![digest.hex.clone()]);
                if !self.allowed(path, &hex(recorded), &hex(taken)) {
                    self.found.divergences.push(Divergence::new(
                        path,
                        DivergenceKind::Digest {
                            metal: recorded.hex.clone(),
                            ours: taken.hex.clone(),
                        },
                    ));
                }
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
        // Only a difference consults an allowance. A site that matched must not fire one: an
        // allowance that fires where nothing moved would fail every exact run whose device
        // announced no notice.
        if metal != ours && self.allowed(path, metal, ours) {
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
            let first = layout.row_offset as usize;
            image < layout.rows as usize * layout.row_pitch as usize
                && row >= first
                && row < first + layout.row_elements as usize * layout.element.size() as usize
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
        // A refusal is settled before any allowance is consulted, because no allowance may turn
        // one into a pass: Metal answered the call and we did not.
        if refused(metal, ours) {
            self.refusal(path, ours);
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

    /// Note that we refused a call Metal answered, with whatever the case named to document it.
    ///
    /// The case's own allowances at the site are still held to their justifications: a site may
    /// name several, so the documentation is the first whose justification **holds**, as
    /// [`Walk::allowed`] picks one. Where none holds, the first that does not is noted as a fault
    /// of its own, because a case must not claim an explanation it cannot show. What a
    /// justification cannot do is remove the row from [`Comparison::refusals`].
    fn refusal(&mut self, path: &str, ours: &Value) {
        let mut documented_by = None;
        let mut first_unjustified = None;
        for allowance in self
            .allowances
            .iter()
            .filter(|allowance| allowance.covers(path))
        {
            match allowance.justified_by.unjustified(&self.offered) {
                None => {
                    documented_by = Some(allowance.justified_by.describe());
                    break;
                }
                Some(why) => {
                    first_unjustified = first_unjustified.or(Some((allowance.path.clone(), why)));
                }
            }
        }
        if documented_by.is_none() {
            if let Some((path, why)) = first_unjustified {
                self.found.note_bad(&path, why);
            }
        }
        self.found.refusals.push(Refusal {
            path: path.to_owned(),
            ours: crate::diff::print(ours),
            documented_by,
        });
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

    /// Whether an allowance of the case covers `path` and holds for this difference.
    ///
    /// A site may carry several allowances, and must: a notice is announced once per device per
    /// kind, so which of a case's notices its own run sees depends on what ran before it. The
    /// difference is justified where **any** of the covering allowances holds; the case fails only
    /// where none does, and then by the first of them, so the report names a justification the
    /// case actually claimed.
    fn allowed(&mut self, path: &str, metal: &Value, ours: &Value) -> bool {
        let mut covering = self
            .allowances
            .iter()
            .filter(|allowance| allowance.covers(path) && allowance.within(metal, ours))
            .peekable();
        if covering.peek().is_none() {
            return false;
        }
        let mut first_unjustified = None;
        let mut held = None;
        for allowance in covering {
            match allowance.justified_by.unjustified(&self.offered) {
                None => {
                    held = Some(allowance);
                    break;
                }
                Some(why) => {
                    first_unjustified = first_unjustified.or(Some((allowance, why)));
                }
            }
        }
        let Some(allowance) = held else {
            let (allowance, why) = first_unjustified.expect("a covering allowance");
            let path = allowance.path.clone();
            self.found.note_bad(&path, why);
            return false;
        };
        let difference = crate::allowance::difference(metal, ours);
        match self.found.announced.iter_mut().find(|held| {
            held.path == path && held.justified_by == allowance.justified_by.describe()
        }) {
            Some(held) => {
                held.count += 1;
                held.largest = match (held.largest, difference) {
                    (Some(held), Some(found)) => Some(held.max(found)),
                    (held, found) => held.or(found),
                };
            }
            None => self.found.announced.push(Announced {
                path: path.to_owned(),
                how: allowance.loosening.clone(),
                justified_by: allowance.justified_by.describe(),
                count: 1,
                largest: difference,
            }),
        }
        true
    }
}
