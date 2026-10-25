//! docs/API.md's degrade-notice catalogue against the declarations in `degradation.rs`. The
//! catalogue is prose written for a consumer, so it recasts a kind's rustdoc rather than copying
//! it, and most of a declaration's words are no contract: its trigger names the file that lowers
//! the degradation, where the entry names none, and a sentence of either may say in its own terms
//! what the other says in its. Two things the declarations state exactly, which these tests keep:
//!
//! - **The item.** An entry opens with the Rust item it documents, so an entry cannot drift onto
//!   another kind: a runtime kind's [`Degradation`] variant, a note's kind the `PrecisionNote`
//!   variant it announces.
//! - **The cost.** A runtime kind's entry gives, after `Cost: `, the words
//!   [`Degradation::cost`] announces, before whatever more it says. The announced text is a
//!   contract (the stderr line and `DebugMessage::text` carry it), so the catalogue states it
//!   rather than paraphrasing it; an entry may go on past it, which 11 do today. Every note's
//!   kind announces one generic cost, which its entry states in the note's own terms, so a note's
//!   entry is held to its item alone.

use crate::Degradation;
use std::collections::BTreeMap;

/// docs/API.md's catalogue, each code's entry as one line of text: the words after the code and
/// its category (`- \`<code>\` (<category>). `), with every continuation line, retired codes among
/// them.
fn catalogue() -> BTreeMap<String, String> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/API.md");
    let guide = std::fs::read_to_string(path).unwrap();
    let mut entries = BTreeMap::new();
    let mut open: Option<(String, String)> = None;
    for line in guide.lines() {
        if let Some(rest) = line.strip_prefix("- `metal2vulkan-degrade-") {
            if let Some((code, text)) = open.take() {
                entries.insert(code, text);
            }
            let (kind, after) = rest
                .split_once('`')
                .expect("an entry's code closes its span");
            let (_, text) = after
                .split_once(')')
                .expect("an entry's category closes its bracket");
            let text = text.trim_start_matches('.').trim_start();
            open = Some((format!("metal2vulkan-degrade-{kind}"), text.to_owned()));
        } else if open.is_some() {
            match line
                .strip_prefix("  ")
                .filter(|more| !more.trim().is_empty())
            {
                Some(more) => {
                    let (_, text) = open.as_mut().expect("an entry is open");
                    text.push(' ');
                    text.push_str(more);
                }
                None => {
                    let (code, text) = open.take().expect("an entry is open");
                    entries.insert(code, text);
                }
            }
        }
    }
    if let Some((code, text)) = open {
        entries.insert(code, text);
    }
    entries
}

/// `text`'s words, so that the catalogue's markdown and an announced string compare: every run of
/// whitespace one space (the catalogue wraps its lines, a cost string its literal) and backticks
/// dropped, so a code span and the plain identifier a notice prints read alike.
fn words(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace('`', "")
}

/// The words `entry` states as its cost, after `Cost: `, or `None` where it states none.
fn stated_cost(entry: &str) -> Option<String> {
    let entry = words(entry);
    let (_, cost) = entry.split_once("Cost: ")?;
    Some(cost.to_owned())
}

/// Whether `entry` gives `cost` as the cost it states: those words, and then whatever more the
/// entry says.
fn states_cost(entry: &str, cost: &str) -> bool {
    stated_cost(entry).is_some_and(|stated| stated.starts_with(&words(cost)))
}

/// Every entry opens with the Rust item whose kind it documents.
#[test]
fn an_entry_opens_with_its_kind_s_item() {
    let catalogue = catalogue();
    for &kind in Degradation::ALL {
        let entry = catalogue
            .get(kind.code())
            .unwrap_or_else(|| panic!("{kind:?} ({}) has no entry", kind.code()));
        let item = match kind.precision_note() {
            Some(note) => format!("`PrecisionNote::{note:?}`:"),
            None => format!("`{kind:?}`:"),
        };
        assert!(
            entry.trim_start().starts_with(&item),
            "{}'s entry opens with {:?}, not {item}",
            kind.code(),
            entry
                .trim_start()
                .chars()
                .take(item.len())
                .collect::<String>(),
        );
    }
}

/// A runtime kind's entry states the cost the kind announces, word for word, before whatever more
/// it says.
#[test]
fn a_runtime_kind_s_entry_states_the_cost_it_announces() {
    let catalogue = catalogue();
    for &kind in Degradation::ALL {
        if kind.precision_note().is_some() {
            continue;
        }
        let entry = &catalogue[kind.code()];
        let Some(stated) = stated_cost(entry) else {
            panic!("{}'s entry states no cost", kind.code());
        };
        let cost = words(kind.cost());
        assert!(
            stated.starts_with(&cost),
            "{}'s entry states a cost the kind does not announce.\n\
             announced: {cost}\n\
             stated:    {stated}\n\
             they part at: {}",
            kind.code(),
            parting(&cost, &stated),
        );
    }
}

/// Where `announced` and `stated` first differ, with what follows in each.
fn parting(announced: &str, stated: &str) -> String {
    let at = announced
        .char_indices()
        .zip(stated.char_indices())
        .find(|((_, a), (_, b))| a != b)
        .map_or(announced.len().min(stated.len()), |((i, _), _)| i);
    format!(
        "{:?} then announced {:?}, stated {:?}",
        &announced[..at],
        &announced[at..],
        stated.get(at..).unwrap_or_default(),
    )
}

/// The cost check reads a difference rather than only a resemblance: an entry that paraphrases a
/// cost, shortens it, drops a clause from its middle or states none at all fails it. These stand
/// in for an entry whose words drift from the kind's, which is how the check is meant to fail.
#[test]
fn an_entry_that_only_resembles_the_cost_does_not_state_it() {
    let cost = "a scratch image per pass, kept while the pass runs";
    // The words, alone or before more of the entry's own.
    assert!(states_cost(
        "`K`: A trigger. Cost: a scratch image per pass, kept while the pass runs.",
        cost
    ));
    assert!(states_cost(
        "`K`: A trigger. Cost: a scratch image per pass, kept while the pass runs; the copy in is \
         cut to the render area.",
        cost
    ));
    // Wrapped over the catalogue's lines, and with a code span where the cost prints plainly.
    assert!(states_cost(
        "Cost: a scratch   image per\npass, kept while the pass runs.",
        cost
    ));
    assert!(states_cost(
        "Cost: a scratch image per `pass`, kept while the pass runs.",
        "a scratch image per pass, kept while the pass runs"
    ));
    // A paraphrase, a shortening, a clause dropped from the middle, another kind's cost, and no
    // cost at all.
    assert!(!states_cost(
        "Cost: a scratch image for each pass, kept while the pass runs.",
        cost
    ));
    assert!(!states_cost("Cost: a scratch image per pass.", cost));
    assert!(!states_cost(
        "Cost: a scratch image per pass, kept while it runs.",
        cost
    ));
    assert!(!states_cost(
        "Cost: its texels are copied into an image.",
        cost
    ));
    assert!(!states_cost(
        "`K`: A trigger: a scratch image per pass, kept while the pass runs.",
        cost
    ));
}
