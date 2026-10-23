//! What a divergence is and how it prints.
//!
//! A failure names the site that moved by its path from the root, and prints the one leaf fact that
//! differs: the element of a byte region, the line of an ordered list, or what a set is missing and
//! holds extra. It never prints two whole values and leaves the reader to bisect them.

use crate::transcript::{Element, Negative, Value};
use std::collections::BTreeSet;

/// A site whose observation differs, with what moved there.
#[derive(Clone, Debug, PartialEq)]
pub struct Divergence {
    /// The site, as node names from the root, `/`-separated.
    pub path: String,
    pub kind: DivergenceKind,
}

impl Divergence {
    /// A divergence at `path`.
    pub fn new(path: impl Into<String>, kind: DivergenceKind) -> Self {
        Self {
            path: path.into(),
            kind,
        }
    }
}

impl std::fmt::Display for Divergence {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(out, "{}: {}", self.path, self.kind)
    }
}

/// What moved at a site.
#[derive(Clone, Debug, PartialEq)]
pub enum DivergenceKind {
    /// One element of a byte region differs.
    Element {
        element: Element,
        image: u32,
        row: u32,
        index: u32,
        /// The byte offset of the element inside the region, for a reader who wants the raw bytes.
        at: usize,
        metal: Value,
        ours: Value,
    },
    /// A byte between the rows or the images of a region differs, where the case compares the
    /// gaps.
    Gap { at: usize, metal: u8, ours: u8 },
    /// A region's bytes are a different length, so there is no element to name.
    Length { metal: usize, ours: usize },
    /// One line of an ordered list differs, or one side has a line the other does not.
    Line {
        at: usize,
        metal: Option<String>,
        ours: Option<String>,
    },
    /// A set differs: what Metal held and we did not, what we held and Metal did not, and how many
    /// both held.
    Set {
        missing: BTreeSet<String>,
        extra: BTreeSet<String>,
        common: usize,
    },
    /// A list is a different length.
    ListLength { metal: usize, ours: usize },
    /// A variant differs: the name, not the payload.
    Variant { metal: String, ours: String },
    /// Two values of one kind differ, and the kind has no finer diff.
    Value { metal: Value, ours: Value },
    /// The two sides observed different kinds of value at one site.
    Kind {
        metal: &'static str,
        ours: &'static str,
    },
    /// Metal recorded a fact about what did not happen, and we observed something; or the other
    /// way about.
    Negative {
        metal: Option<Negative>,
        ours: Option<Negative>,
    },
    /// The digest both sides took differs, where the recording carries no value to diff.
    Digest { metal: String, ours: String },
    /// A site one side holds and the other does not.
    Site { held_by: Side },
    /// The case loosened this site, and the allowance does not hold: the difference it would have
    /// covered is one nobody is accounting for.
    Unjustified { why: String },
    /// The site's fields differ in which fields they hold.
    Field { name: String, held_by: Side },
}

/// Which side holds something the other does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Metal,
    Ours,
}

impl std::fmt::Display for Side {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Side::Metal => out.write_str("Metal"),
            Side::Ours => out.write_str("ours"),
        }
    }
}

impl std::fmt::Display for DivergenceKind {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DivergenceKind::Element {
                element,
                image,
                row,
                index,
                at,
                metal,
                ours,
            } => write!(
                out,
                "element [image {image}, row {row}, {index}] ({element:?}, byte {at}): \
                 Metal {}, ours {}",
                print(metal),
                print(ours)
            ),
            DivergenceKind::Gap { at, metal, ours } => write!(
                out,
                "the byte at {at}, between the rows, is {metal:#04x} on Metal and {ours:#04x} here"
            ),
            DivergenceKind::Length { metal, ours } => {
                write!(out, "Metal wrote {metal} bytes, we read {ours}")
            }
            DivergenceKind::Line { at, metal, ours } => match (metal, ours) {
                (Some(metal), Some(ours)) => {
                    write!(out, "line {at}: Metal {metal:?}, ours {ours:?}")
                }
                (Some(metal), None) => write!(out, "line {at} is missing; Metal has {metal:?}"),
                (None, Some(ours)) => write!(out, "line {at} is extra: {ours:?}"),
                (None, None) => write!(out, "line {at}"),
            },
            DivergenceKind::Set {
                missing,
                extra,
                common,
            } => {
                write!(out, "{common} in both")?;
                if !missing.is_empty() {
                    write!(out, ", missing {missing:?}")?;
                }
                if !extra.is_empty() {
                    write!(out, ", extra {extra:?}")?;
                }
                Ok(())
            }
            DivergenceKind::ListLength { metal, ours } => {
                write!(out, "Metal listed {metal} values, we listed {ours}")
            }
            DivergenceKind::Variant { metal, ours } => {
                write!(out, "Metal answered {metal}, we answered {ours}")
            }
            DivergenceKind::Value { metal, ours } => {
                write!(out, "Metal {}, ours {}", print(metal), print(ours))
            }
            DivergenceKind::Kind { metal, ours } => {
                write!(out, "Metal observed a {metal}, we observed a {ours}")
            }
            DivergenceKind::Negative { metal, ours } => match (metal, ours) {
                (Some(fact), None) => write!(out, "Metal recorded {fact:?}, we observed a value"),
                (None, Some(fact)) => write!(out, "we recorded {fact:?}, Metal observed a value"),
                (Some(metal), Some(ours)) => write!(out, "Metal {metal:?}, ours {ours:?}"),
                (None, None) => write!(out, "a negative fact"),
            },
            DivergenceKind::Digest { metal, ours } => {
                write!(out, "the digest is {metal} on Metal and {ours} here")
            }
            DivergenceKind::Site { held_by } => write!(out, "only {held_by} observed this site"),
            DivergenceKind::Unjustified { why } => {
                write!(out, "the case loosens this site, and {why}")
            }
            DivergenceKind::Field { name, held_by } => {
                write!(out, "only {held_by} holds the field {name}")
            }
        }
    }
}

/// `value` in one line, short enough for a failure message.
pub fn print(value: &Value) -> String {
    match value {
        Value::Absent => "absent".to_owned(),
        Value::Nothing(fact) => format!("{fact:?}"),
        Value::Bool(word) => word.to_string(),
        Value::Uint(word) => format!("{word} ({word:#x})"),
        Value::Int(word) => word.to_string(),
        Value::Float { bits, width } => format!("{width:?} {bits:#x}"),
        Value::Bytes(bytes) => format!("{} bytes", bytes.len()),
        // One line is printed, because a typed error's payload is one line and a refusal's row is
        // a bug report: "1 lines" told a reader nothing they could act on. More than one stays a
        // count, so a long probe transcript does not fill the report.
        Value::Lines(lines) => match lines.as_slice() {
            [only] => format!("{only:?}"),
            many => format!("{} lines", many.len()),
        },
        Value::Set(words) => format!("{words:?}"),
        Value::List(values) => format!("{} values", values.len()),
        Value::Variant { name, payload } => match **payload {
            Value::Absent => name.clone(),
            _ => format!("{name}({})", print(payload)),
        },
        Value::Fields(fields) => format!("{} fields", fields.len()),
    }
}
