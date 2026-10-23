#!/usr/bin/env python3
"""Write docs/CONFORMANCE-COVERAGE.md: which Metal items the conformance suite verifies.

`docs/METAL-COVERAGE.md` records what we **implement**. This records what the suite **compares
against Metal**, which is a different and narrower claim: an item is only verified where a case's
native Metal leg calls it, the case has a committed recording of Metal's answer, and our side's
answer is compared with it.

Two axes, because one of them alone reads as more than it is. The **items** axis is reachability:
one call of a selector is enough, so a selector that takes an enum counts as exercised whatever
value reached it. The **option space** axis (`option_space.py`) is depth: of each Metal enum's and
option set's values, which ones the suite actually compared.

Nothing here is a maintained list. Every number is derived, at the moment this runs, from four
things already in the tree:

1. `scripts/metal-coverage/coverage.tsv` — the SDK's items and whether we implement each. This
   does not re-derive that; it joins onto it.
2. `scripts/conformance-coverage/objc2_metal.tsv` — the selector each `objc2-metal` binding sends,
   which type declares it and what it answers. `--api` checks it against the crate's own sources
   on a Mac; it is committed so this runs anywhere.
3. `crates/metal2vulkan-conformance/src/` — the suite's own sources. `metal_leg.py` types each
   receiver in the Metal leg and resolves its calls to SDK items, and the reachability walk below
   says which of the suite's own vocabulary reaches each call.
4. `crates/metal2vulkan-conformance/recordings/` — the committed recordings, which are what makes
   a case's Metal answer a recorded fact rather than an intention.
5. `crates/metal2vulkan/src/metal/generated.rs` and the front end's — the SDK's enumerator names
   and ordinals, which `scripts/regen-metal-enums/` extracts from the Metal headers. They are the
   option-space axis's denominator; no value list is written here.

Usage:
  coverage.py            write docs/CONFORMANCE-COVERAGE.md
  coverage.py --check    fail if docs/CONFORMANCE-COVERAGE.md is not what the sources derive
  coverage.py --api      (Mac only) fail if objc2_metal.tsv is not what objc2-metal's sources say
  coverage.py --selftest run the option-space axis over inputs whose answers are known
"""

import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import metal_leg  # noqa: E402  (the resolver, beside this script)
import option_space  # noqa: E402  (the option-space axis, beside this script)

SUITE = ROOT / "crates/metal2vulkan-conformance"
SRC = SUITE / "src"
DOC = ROOT / "docs/CONFORMANCE-COVERAGE.md"
API = HERE / "objc2_metal.tsv"
TABLE = ROOT / "scripts/metal-coverage/coverage.tsv"

#: The oracle every committed recording is of, as the suite states it.
ORACLE = "apple-m3-max"

STATUS = {"I": "implemented", "M": "missing", "O": "out of scope"}


# --------------------------------------------------------------------------------------------
# The SDK's items


def sdk_rows() -> list[dict[str, str]]:
    """Every row of the Metal API coverage map."""
    with TABLE.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def item_index(rows) -> dict[tuple[str, str], dict[str, str]]:
    """The rows by `(container, name)`."""
    return {(row["container"], row["name"]): row for row in rows}


def sdk_item(index, api: metal_leg.Api, ty: str, selector: str):
    """The coverage map's key for the item a call of `selector` on `ty` exercises, or `None`.

    The map keys an item by the container that **declares** it, so a call on `MTLTexture` of
    `label` is `MTLResource`'s item; and a setter is the property it writes, which the map holds
    once under the property's own name.
    """
    # `objc2` writes an out-parameter error as a trailing `_`, which the header does not.
    selector = selector.removesuffix("_")
    names = [selector]
    if selector.endswith(":") and selector.count(":") == 1 and selector.startswith("set"):
        bare = selector[3:-1]
        names.append(bare[0].lower() + bare[1:])
    # A `BOOL` property's getter is spelled `isFoo`; the header declares the property `foo`.
    if ":" not in selector and re.fullmatch(r"is[A-Z][A-Za-z0-9]*", selector):
        names.append(selector[2].lower() + selector[3:])
    for owner in api.chain(ty):
        for name in names:
            if (owner, name) in index:
                return (owner, name)
    return None


# --------------------------------------------------------------------------------------------
# The suite's vocabulary and what reaches each call


def crate_enums() -> set[str]:
    """Every enum the suite declares."""
    out = set()
    for path in sorted(SRC.rglob("*.rs")):
        text = metal_leg.blanked(path.read_text())
        out.update(re.findall(r"\benum\s+([A-Za-z_][A-Za-z0-9_]*)", text))
    return out


def gate_enums(enums: set[str]) -> set[str]:
    """The enums a `match` arm in the Metal leg may be gated on.

    Only an enum whose variants a case **names** can gate: the attribution reads a case's
    vocabulary off its own text, so an arm on an enum no case names carries no information about
    which case reaches it, and treating it as a gate would drop the call from every family.
    """
    named: set[str] = set()
    for path in sorted((SRC / "cases").glob("*.rs")):
        text = metal_leg.blanked(path.read_text())
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)::[A-Za-z_][A-Za-z0-9_]*", text):
            if match.group(1) in enums:
                named.add(match.group(1))
    return named


def leg_files() -> list[str]:
    """The suite's files that reach Metal, by what they import rather than by name."""
    out = []
    for path in sorted(SRC.rglob("*.rs")):
        if "objc2_metal" in metal_leg.blanked(path.read_text()):
            out.append(str(path.relative_to(SRC)))
    return out


def side_methods() -> list[str]:
    """Every method of the `Side` trait: the suite's whole vocabulary for reaching a device."""
    text = metal_leg.blanked((SRC / "case.rs").read_text())
    at = text.index("trait Side")
    start, end = metal_leg.block(text, at)
    return sorted(set(re.findall(r"\bfn\s+([a-z_][a-z0-9_]*)", text[start:end])))


def variant_constructors(enums: set[str]) -> dict[str, set[str]]:
    """Each `Enum::helper` that builds variants, to the variants it builds.

    A case says `RenderStep::triangle(..)` where the vocabulary's own constructor makes a
    `RenderStep::Draw`, so following these is what keeps a case's step set from reading empty.
    """
    out: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.rs")):
        text = metal_leg.blanked(path.read_text())
        for match in re.finditer(r"\bimpl\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", text):
            owner = match.group(1)
            if owner not in enums:
                continue
            start, end = metal_leg.block(text, match.start())
            for fn in metal_leg.functions(text[start:end], str(path)):
                built = {
                    f"{owner}::{name}"
                    for name in re.findall(
                        rf"(?:Self|{owner})::([A-Za-z_][A-Za-z0-9_]*)",
                        text[start:end][fn.start : fn.end],
                    )
                }
                if built:
                    out.setdefault(f"{owner}::{fn.name}", set()).update(built)
    # A constructor that only calls another constructor resolves through it.
    for _ in range(4):
        for name, built in out.items():
            out[name] = {v for b in built for v in out.get(b, {b})}
    return out


def sdk_keyed(leg: metal_leg.Leg, index, api: metal_leg.Api):
    """Each call's SDK item, and the calls whose item the coverage map does not list.

    `objc2-metal` and the coverage map key an item by different containers: `objc2` by the trait
    nearest the receiver, the map by the header that declares it. And the map holds a property
    once, under its getter's name, while a call may be its setter. So the receiver's own supertype
    chain is walked until the map has the item.
    """
    keyed: dict[metal_leg.Call, tuple[str, str]] = {}
    unlisted: list[tuple[str, str]] = []
    for call in leg.calls:
        if not call.item:
            continue
        found = sdk_item(index, api, call.receiver or call.item[0], call.item[1])
        if found is None:
            unlisted.append(call.item)
        else:
            keyed[id(call)] = found
    return keyed, sorted(set(unlisted))


def gated_items(leg: metal_leg.Leg, entry_points: set[str], keyed):
    """For each entry point, the items it reaches and the variants each reaching needs.

    One walk of the leg's own call graph. A call inside `Step::GetBytes =>` is reached only by a
    case that encodes that step, so the variants of the `match` arms around a call travel with it;
    a call under no arm is reached by every case that calls the entry point at all.
    """
    reached: dict[str, dict[tuple[str, str], set[frozenset]]] = {}

    def walk(fn: str, stack: tuple[str, ...]) -> dict[tuple[str, str], set[frozenset]]:
        if fn in reached:
            return reached[fn]
        if fn in stack:
            return {}
        here: dict[tuple[str, str], set[frozenset]] = defaultdict(set)
        for call in leg.calls:
            if call.function == fn and id(call) in keyed:
                here[keyed[id(call)]].add(frozenset(call.arms))
        for callee in leg.calls_function.get(fn, ()):
            below = walk(callee, stack + (fn,))
            if not below:
                continue
            for arms in leg.function_arms.get((fn, callee), {()}):
                for item, gates in below.items():
                    for gate in gates:
                        here[item].add(frozenset(gate | set(arms)))
        if fn not in stack:
            reached[fn] = dict(here)
        return dict(here)

    return {fn: walk(fn, ()) for fn in sorted(entry_points)}


# --------------------------------------------------------------------------------------------
# The case set


def case_families(side: list[str], enums: set[str], constructors) -> dict[str, dict]:
    """Each case module, with the vocabulary its cases use and what it loosens.

    A module's cases share one `run`, so the module is the unit a call site can be attributed to
    without running anything: which `Side` methods its text calls, and which variants of the
    vocabulary its text builds.
    """
    out: dict[str, dict] = {}
    for path in sorted((SRC / "cases").glob("*.rs")):
        source = path.read_text()
        text = metal_leg.blanked(source)
        # An allowance's site is a string, so it is read from the source as written, with the
        # comments dropped so prose about an allowance is not counted as one.
        written = "\n".join(line.split("//")[0] for line in source.splitlines())
        # `rustfmt` breaks `side.mesh_pass(..)` across lines, so the dot is not always beside the
        # receiver.
        calls = {m for m in re.findall(r"\bside\s*\.\s*([a-z_][a-z0-9_]*)\s*\(", text) if m in side}
        variants: set[str] = set()
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)::([A-Za-z_][A-Za-z0-9_]*)", text):
            name = f"{match.group(1)}::{match.group(2)}"
            if match.group(1) not in enums:
                continue
            variants.update(constructors.get(name, {name}))
        out[path.stem] = {
            "calls": calls,
            "variants": variants,
            # What the module loosens, and what it names as the justification. Comparison is
            # exact by default, so a module with no tolerance compares every measured field.
            #
            # A **notice declaration** and a **tolerance** are both `Allowance`s and are not the
            # same thing: a declaration loosens only the notices our side announces, which Metal
            # has none of, so the case can be read at all on a venue that announces one. It
            # loosens no measured value. A tolerance does, and is the number that matters.
            "declarations": _declarations(written),
            "allowances": _tolerances(written),
            "loosenings": sorted(set(re.findall(r"\bLoosening::([A-Za-z]+)", text))),
            "justifications": sorted(
                {
                    {"notice": "Degradation", "accepted": "AcceptedDifference"}.get(name, name)
                    for name in re.findall(r"\bJustification::([A-Za-z]+)", text)
                }
            ),
            # A carried answer record mode must refuse to replace, because its Metal leg is unsafe.
            "carried_only": len(re.findall(r"\bcarried_only\s*\(", text)),
            "cases": [],
            # The cases whose Metal answer record mode took by **running** the native leg. A
            # carried answer, and a call Metal never returned from, run no leg at all, so they
            # exercise nothing however much Metal the module's code names.
            "ran": [],
        }
    return out


def _allowance_paths(text: str) -> list[str]:
    """The site each `Allowance::new` in `text` loosens, as the source writes it."""
    out = []
    for match in re.finditer(r"\bAllowance::new\s*\(", text):
        depth = 1
        i = match.end()
        while i < len(text) and depth:
            if text[i] in "([{":
                depth += 1
            elif text[i] in ")]}":
                depth -= 1
            elif text[i] == "," and depth == 1:
                break
            i += 1
        out.append(text[match.end() : i])
    return out


def _is_declaration(path: str) -> bool:
    """Whether an allowance of this site is a notice declaration rather than a tolerance."""
    return "NOTICES_SITE" in path or "notices" in path


def _declarations(text: str) -> int:
    """How many notice declarations `text` makes."""
    return len(re.findall(r"\bnotice_allowance\s*\(", text)) + sum(
        1 for path in _allowance_paths(text) if _is_declaration(path)
    )


def _tolerances(text: str) -> int:
    """How many measured values `text` loosens."""
    return sum(1 for path in _allowance_paths(text) if not _is_declaration(path))


def _provenance(answer: dict) -> str:
    """Where an answer came from: `metal_run`, or the kind of answer it carries instead."""
    found = answer.get("provenance", "")
    return found if isinstance(found, str) else next(iter(found), "")


def recordings(families: dict[str, dict]) -> list[dict]:
    """Every committed recording, with the case module that declares its case's name.

    A case's name is a literal of its own module, so the recording and the code it came from are
    joined by the name itself rather than by a table saying which is which.
    """
    # The literals of each module, read from the source as written: a case's name is a string, and
    # the blanking the rest of this derivation needs would take exactly the thing being looked up.
    literals = {
        module: set(re.findall(r'"([^"\\\n]*)"', (SRC / "cases" / f"{module}.rs").read_text()))
        for module in families
    }
    out = []
    for path in sorted((SUITE / "recordings").glob("*.json")):
        data = json.loads(path.read_text())
        name = data["case"]
        owners = [module for module, names in literals.items() if name in names]
        if len(owners) != 1:
            raise SystemExit(
                f"{path.name}: the case name {name!r} is a literal of {len(owners)} case modules "
                f"({', '.join(owners) or 'none'}); this derivation needs exactly one"
            )
        answers = data.get("answers", [])
        out.append(
            {
                "case": name,
                "module": owners[0],
                "runs": sum(answer.get("runs", 0) for answer in answers),
                "provenance": sorted({_provenance(answer) for answer in answers}),
                "chips": sorted({answer.get("key", {}).get("chip", "") for answer in answers}),
                "outcomes": sum(len(answer.get("outcomes", [])) for answer in answers),
                "shaders": len(data.get("shaders", [])),
            }
        )
        families[owners[0]]["cases"].append(name)
        if "metal_run" in out[-1]["provenance"]:
            families[owners[0]]["ran"].append(name)
    return out


def known_gaps() -> list[tuple[str, str]]:
    """The ledger's rows, as `(case, what it is waiting on)`."""
    out = []
    for name in ("known-gaps.tsv", "known-refusals.tsv"):
        path = SUITE / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split("\t")
            out.append((fields[0], fields[-1]))
    return out


# --------------------------------------------------------------------------------------------
# The join


def inventory():
    """Everything the document states, derived."""
    api = metal_leg.Api.read(API)
    rows = sdk_rows()
    index = item_index(rows)
    enums = crate_enums()
    gates = gate_enums(enums)
    files = leg_files()
    leg = metal_leg.read_leg(api, SRC, files, gates)
    if leg.mistyped:
        raise SystemExit(
            "a receiver typed to something that does not bind the selector sent to it, which is a "
            "bug in metal_leg.py rather than a Metal call:\n"
            + "\n".join(f"  {f}::{fn} sent {rust} to {ty}" for f, fn, rust, ty in sorted(set(leg.mistyped)))
        )
    keyed, unlisted = sdk_keyed(leg, index, api)
    side = side_methods()
    constructors = variant_constructors(enums)
    families = case_families(side, enums, constructors)
    records = recordings(families)

    # The call graph, from each `Side` method the Metal leg implements.
    entries = {fn for fn in leg.calls_function} | {call.function for call in leg.calls}
    gated = gated_items(leg, entries & set(side), keyed)

    # An item to the families that reach it, and how.
    by_item: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for module, family in families.items():
        if not family["ran"]:
            # Nothing of this module's Metal is ever executed.
            continue
        for method in family["calls"]:
            for item, gates in gated.get(method, {}).items():
                for gate in gates:
                    if gate <= family["variants"]:
                        by_item[item][module].add(method)
                        break

    # Calls the leg makes that no family was found to reach: a call site this cannot attribute.
    reached = {item for method in gated.values() for item in method}
    called = set(keyed.values())
    return {
        "rows": rows,
        "index": index,
        "api": api,
        "leg": leg,
        "side": side,
        "families": families,
        "records": records,
        "gated": gated,
        "by_item": by_item,
        "unattributed": sorted(called - set(by_item)),
        "unreachable": sorted(called - reached),
        "keyed": keyed,
        "unlisted": unlisted,
        "gaps": known_gaps(),
        # The second axis. A module with no case whose answer Metal ran puts no value to Metal,
        # which is the same gate the items axis applies to a call.
        "option_space": option_space.inventory(
            ROOT, {module for module, family in families.items() if family["ran"]}
        ),
    }


# --------------------------------------------------------------------------------------------
# The document


def _container(name: str) -> str:
    """A container's name as the document prints it; the map leaves a free function's empty."""
    return f"`{name}`" if name else "the header's free functions"


def families_line(modules: dict[str, set[str]]) -> str:
    """The families that reach an item, as the document prints them."""
    return ", ".join(f"`{module}`" for module in sorted(modules))


def names_line(names: list[str]) -> str:
    """Enumerator names as the document prints them."""
    return ", ".join(f"`{name}`" for name in names)


def write_option_space(w, space) -> None:
    """The second axis: of each Metal enum's and option set's values, which the suite compared."""
    rows = space["rows"]
    values = sum(len(row["values"]) for row in rows)
    compared = sum(len(row["compared"]) for row in rows)
    shared = sum(len(row["shared"]) for row in rows)
    missing = sum(len(row["missing"]) for row in rows)

    w("## By option space, which is depth rather than reach")
    w("")
    w(
        "The axis above counts a Metal **item** as exercised where one call of it ran. A selector "
        "that takes an enum therefore counts whatever value reached it, so that axis says nothing "
        "about how much of an enum the suite ever put to Metal. This one does, over the values "
        "themselves."
    )
    w("")
    w(
        "The denominator is generated, never written here: "
        "`crates/metal2vulkan/src/metal/generated.rs` and the front end's hold every enumerator "
        "name and ordinal `scripts/regen-metal-enums/` extracts from the SDK's Metal headers. A "
        "Metal type this tree models by hand instead has **no denominator here** and is listed "
        "apart, below."
    )
    w("")
    w("A value is **compared** where either of two things holds, which are different claims:")
    w("")
    w(
        "- **answered** — the enumerator's name is in a committed recording's transcript of an "
        "answer Metal **ran**. Metal named the value back at a site the comparison reads, so the "
        "value is in the compared text itself."
    )
    w(
        "- **in effect** — a case module's own text names the value, and that module has at least "
        "one case whose recording is a Metal run. Metal's recorded answer was taken with the "
        "value in effect, so what is compared is Metal's behaviour **under** it, even where no "
        "site reads it back. A blend factor is this and not the first: what the recording holds is "
        "the texels Metal blended with it."
    )
    w("")
    w("What this axis cannot see, measured rather than asserted:")
    w("")
    w(
        "- **A value Metal answers as a bare ordinal.** A transcript holds a `uint`, and nothing "
        "in it says which enum the number belongs to, so only a site that records the enumerator's "
        "**name** is read. The *answered* column is a floor by every site that records an ordinal "
        "instead."
    )
    w(
        "- **A value a case states as a raw ordinal.** A case may hand the Metal leg an enum's "
        "ordinal as a number — a sweep over `0..n` rather than over named variants — and no "
        "reading of the text can say which value a computed number is. So *in effect*, and the "
        "*stated* column of the hand-modelled table below, are floors by every sweep written that "
        "way: a type they report short of its values may in fact be swept exhaustively, and the "
        "case is what says so. Read the case before taking a row of the work list as work."
    )
    w(
        "- **Which case put a value in effect.** The unit is the case module, as it is for the "
        "items axis: a value named anywhere in a module whose cases include a Metal run counts for "
        "that module, even where one of its cases names it and the rest do not."
    )
    w(
        f"- **A value only the suite's shared sources name** — {shared} of them. Some case puts it "
        "to Metal through the render or texture vocabulary, but which case is not a property of "
        "the text, so it is neither compared nor missing and has its own column."
    )
    w(
        "- **Combinations.** Every column here is per value. A type whose values are all compared "
        "says nothing about the pairs and triples of them, which is where most of the behaviour "
        "an implementation can get wrong lives."
    )
    w(
        "- **Metal's API validation is excluded.** Its complaints name enumerators, and the suite "
        "records that leg and never compares it, so reading it would count Apple's prose as a "
        "comparison. So are the cost and timing tables: a site name there is what was measured, "
        "not an answer that was compared."
    )
    w(
        "- **A `#[cfg(test)]` item is excluded.** One of the suite's unit tests names four blend "
        "factors to assert that no case uses them; counting those would have this report the "
        "opposite of what the code says."
    )
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| Metal enums and option sets with a generated value list | {len(rows)} |")
    w(f"| their values | {values} |")
    w(f"| values compared | {compared} |")
    w(f"| share compared | {100.0 * compared / values:.0f}% |")
    w(f"| values only the suite's shared sources name | {shared} |")
    w(f"| values no source names and no recording holds | {missing} |")
    whole = sum(1 for row in rows if not row["missing"] and not row["shared"])
    w(f"| types every value of which is compared | {whole} |")
    w(f"| types no value of which is compared | {sum(1 for r in rows if not r['compared'])} |")
    w("")
    largest = sorted(rows, key=lambda r: -len(r["values"]))[:4]
    w(
        "The share is weakly informative on its own: "
        + ", ".join(f"`{row['metal']}`" for row in largest)
        + f" hold {sum(len(row['values']) for row in largest)} of the {values} values between "
        "them, so an aggregate over types this uneven is mostly theirs. The per-type table is the "
        "one to read."
    )
    w("")
    w(
        "### Depth by type"
    )
    w("")
    w(
        "Deepest gap first, which is the order the work list reads in. *answered* and *in effect* "
        "overlap; *compared* is their union."
    )
    w("")
    w("| Metal type | values | compared | answered | in effect | shared only | missing |")
    w("|---|---|---|---|---|---|---|")
    for row in sorted(rows, key=lambda r: (-len(r["missing"]), -len(r["shared"]), r["metal"])):
        w(
            f"| `{row['metal']}` | {len(row['values'])} | {len(row['compared'])} | "
            f"{len(row['answered'])} | {len(row['in_effect'])} | {len(row['shared'])} | "
            f"{len(row['missing'])} |"
        )
    w("")
    w("### The values not compared")
    w("")
    w(
        "The work list. A name here is a value of Metal's that no committed recording holds and no "
        "case module states **by name**: it is not a claim that Metal accepts the value on every "
        "device, and not a claim that no case reaches it, because a case that sweeps a type by "
        "ordinal states none of its values here. Each row is a question for the case, not a "
        "verdict on it."
    )
    w("")
    for row in sorted(rows, key=lambda r: (-len(r["missing"]), r["metal"])):
        if not row["missing"]:
            continue
        w(f"- **`{row['metal']}`**, {len(row['missing'])} of {len(row['values'])}: "
          + names_line(row["missing"]))
    w("")
    only = [row for row in rows if row["shared"]]
    if only:
        w(
            "And the values only the suite's shared sources name, which some case reaches and this "
            "cannot attribute:"
        )
        w("")
        for row in sorted(only, key=lambda r: r["metal"]):
            w(f"- **`{row['metal']}`**: " + names_line(row["shared"]))
        w("")
    w("### Metal types with no generated value list")
    w("")
    w(
        "These the tree models by hand: a Rust item whose own doc comment opens by naming a Metal "
        "type, and whose shape is a closed enum or a newtype over one integer. "
        "`regen-metal-enums` holds no value list for them, so **this axis has no denominator "
        "here**, and *declared* is our own sources' variant count rather than the SDK's — a "
        "newtype option set declares none. There is no *answered* column either: a recording names "
        "a value by its Metal enumerator, and for these types this has no list of those names to "
        "match against. Giving one a denominator means appending its Metal name to "
        "`scripts/regen-metal-enums/`'s `ENUMS` or `OPTIONS` and regenerating."
    )
    w("")
    w(
        f"These {len(space['by_hand'])} are what that shape finds, and the list is a floor: a type "
        "the tree models without saying so on the item lands in neither table, and so is nowhere "
        "in this axis at all."
    )
    w("")
    w("| Metal type | the tree's type | declared | stated by a recorded case | shared only |")
    w("|---|---|---|---|---|")
    for row in space["by_hand"]:
        w(
            f"| `{row['metal']}` | `{row['rust']}` | {len(row['declared']) or '— a bit set'} | "
            f"{len(row['stated'])} | {len(row['shared'])} |"
        )
    w("")


def write(data) -> str:
    """The document's text."""
    rows, index, api, leg = data["rows"], data["index"], data["api"], data["leg"]
    families, records, by_item = data["families"], data["records"], data["by_item"]
    implemented = [row for row in rows if row["status"] == "I"]
    keys = {(row["container"], row["name"]) for row in implemented}
    exercised = {key for key in by_item if key in keys}
    cases_of = defaultdict(list)
    for record in records:
        cases_of[record["module"]].append(record)

    out: list[str] = []
    w = out.append
    w("# Metal conformance coverage")
    w("")
    w(
        "Generated by `scripts/conformance-coverage/coverage.py` from the suite's own sources and "
        "its committed recordings. Do not edit this file; change the sources or the suite and run "
        "the script."
    )
    w("")
    w(
        "This is not [METAL-COVERAGE.md](METAL-COVERAGE.md), which says what we **implement**. "
        "This says what the conformance suite **compares against Metal**. An item counts as "
        "*exercised* here only where a case's native Metal leg calls it; a case's Metal answer is "
        "a recorded fact only where the recording is committed. The two claims are different "
        "sizes, and only the second is conformance."
    )
    w("")
    w(
        "It says it along **two axes**, because either one alone reads as more than it is. The "
        "**items** axis is reach: one call of a selector is enough, so a selector that takes an "
        "enum counts as exercised whatever value reached it. The **option space** axis is depth: "
        "of each Metal enum's and option set's values, which ones the suite compared. A high "
        "number on the first and a low one on the second is what a suite that touches everything "
        "shallowly looks like."
    )
    w("")

    w("## What this measures, and what it cannot")
    w("")
    w(
        "The suite reaches Metal through `objc2-metal`, whose Rust name for a binding is the "
        "selector with each `:` written `_`. So a call site names its selector outright, and the "
        "derivation resolves the **receiver's type** by following what made it — a class's "
        "constructor, a getter whose answer `objc2_metal.tsv` records, a field of the side's own "
        "struct, a parameter's declared type — and then walks that type's supertypes to the "
        "container the SDK declares the item on. A message to a class, written "
        "`MTLType::selector(..)`, names its own receiver and needs no typing."
    )
    w("")
    w("Its limits, measured rather than asserted:")
    w("")
    sites = len(leg.calls)
    typed = len(data["keyed"])
    w(
        f"- **{typed} of {sites} call sites** in the Metal leg resolve to an item of the SDK. "
        f"{len(leg.by_uniqueness)} of those resolve because only one type in `objc2-metal` binds "
        "the name at all, so no typing was needed."
    )
    only = sorted({r for _, _, r, _ in leg.unresolved} - {c.rust for c in leg.calls if c.item})
    w(
        f"- **{len(leg.unresolved)} call sites resolve to no item**: an instance send whose "
        "receiver this could not type, or a message to a Metal class whose name `objc2-metal` "
        f"declares on no Metal type. They cost the count {len(only)} selectors, which the suite "
        "calls and which this attributes to no item:"
    )
    for rust in only:
        owners = sorted(api.receivers(rust))
        if owners:
            spelled = api.selector[next((t, n) for t, n in api.selector if n == rust)]
            w(f"  - `{spelled}`, bound by " + ", ".join(f"`{ty}`" for ty in owners))
        else:
            w(f"  - `{rust}`, which `objc2-metal` declares on no Metal type, so it is an item of "
              "no Metal container")
    w(
        "  So every per-item count below is a **floor**, by at least these. That list is what "
        "this derivation recognises as a call to Metal and cannot attribute; it is not a proof "
        "that nothing else is unmeasured."
    )
    w(
        "- **0 call sites typed to a receiver that does not bind the selector sent to it.** The "
        "generator fails rather than writing this document if any does, which is what keeps the "
        "typing from quietly counting the wrong item."
    )
    w(
        "- The unit a call is attributed to is the **case module**, not the case: a module's cases "
        "share one `run`, so which of them encodes which step is not a property of the text. "
        "Where a module holds one case the two are the same."
    )
    w(
        "- **No verdict is recorded here.** A verdict is what a replay on one venue measured, and "
        "a number like that in a generated document is a journal entry that rots. What is durable, "
        "and is here, is whether a case compares **exactly**: the allowances a module declares, "
        "and the rows of the suite's ledger."
    )
    w("")

    w("## The headline")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| implemented Metal items | {len(implemented)} |")
    w(f"| of those, exercised by the suite's Metal leg | {len(exercised)} |")
    w(f"| of those, not exercised | {len(implemented) - len(exercised)} |")
    share = 100.0 * len(exercised) / len(implemented)
    w(f"| share exercised | {share:.0f}% |")
    loose = {
        key
        for key in exercised
        if any(families[module]["allowances"] for module in by_item[key])
    }
    w(f"| of those exercised, no case reaching it loosens anything | {len(exercised) - len(loose)} |")
    w(f"| of those exercised, a case reaching it loosens some site | {len(loose)} |")
    w(f"| case modules | {len(families)} |")
    w(f"| committed recordings | {len(records)} |")
    w(f"| distinct SDK items the Metal leg calls | {len(set(data['keyed'].values()))} |")
    w(f"| ledger rows (a case whose gap is recorded) | {len(data['gaps'])} |")
    w("")
    w(
        "A tolerance names a **site of a transcript**, not an item, so the last two rows are read "
        "as written: a case that loosens one site of its own transcript may compare every other "
        "field of it exactly, and this derivation cannot tell which of a case's fields came from "
        "which Metal call. Where the distinction matters, read the case."
    )
    w("")
    w(
        "*Exercised* and *recorded* are one number here, not two, and deliberately: a case is "
        "credited with exercising an item only through its cases whose answer record mode took by "
        "**running** Metal, and such a case has a committed recording by construction. A case that "
        "carries an answer, or whose call Metal never returned from, runs no native leg and "
        "exercises nothing, however much Metal its code names."
    )
    w("")

    write_option_space(w, data["option_space"])

    w("## By container")
    w("")
    w("Every container with an implemented item, by how much of it the suite reaches.")
    w("")
    w("| container | implemented | exercised | unexercised | families |")
    w("|---|---|---|---|---|")
    by_container: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in implemented:
        by_container[row["container"]].append(row)
    order = sorted(
        by_container,
        key=lambda c: (
            -(len(by_container[c]) - len({k for k in exercised if k[0] == c})),
            c,
        ),
    )
    for container in order:
        rows_here = by_container[container]
        hit = {key for key in exercised if key[0] == container}
        modules = sorted({m for key in hit for m in by_item[key]})
        w(
            f"| {_container(container)} | {len(rows_here)} | {len(hit)} | "
            f"{len(rows_here) - len(hit)} | "
            + (", ".join(f"`{m}`" for m in modules[:4]) + ("…" if len(modules) > 4 else ""))
            + " |"
        )
    w("")

    w("## By header, which is the family of behaviour")
    w("")
    w(
        "A container is one Objective-C type; the SDK header it sits in is the family of Metal "
        "behaviour. This is the table to pick a batch from: a header with nothing exercised is a "
        "family the suite says nothing about, and the rest are ordered by how much of each is left."
    )
    w("")
    w("| header | implemented | exercised | unexercised |")
    w("|---|---|---|---|")
    by_header: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in implemented:
        by_header[row["header"]].append(row)
    def left(header: str) -> int:
        return sum(1 for row in by_header[header] if (row["container"], row["name"]) not in exercised)
    for header in sorted(by_header, key=lambda h: (-left(h), h)):
        hit = len(by_header[header]) - left(header)
        w(f"| `{header}` | {len(by_header[header])} | {hit} | {left(header)} |")
    w("")
    empty = [h for h in sorted(by_header) if left(h) == len(by_header[h])]
    w(
        f"{len(empty)} of {len(by_header)} headers have no exercised item at all, "
        f"{sum(len(by_header[h]) for h in empty)} implemented items between them: "
        + ", ".join(f"`{h}`" for h in empty)
        + "."
    )
    w("")

    w("## Containers with no case at all")
    w("")
    w(
        "An implemented container no case's Metal leg touches. These are the families of Metal "
        "behaviour the suite says nothing about."
    )
    w("")
    w("| container | implemented items | header |")
    w("|---|---|---|")
    untouched = [c for c in sorted(by_container) if not any(k[0] == c for k in exercised)]
    for container in untouched:
        header = by_container[container][0]["header"]
        w(f"| {_container(container)} | {len(by_container[container])} | `{header}` |")
    w("")
    w(f"{len(untouched)} containers, {sum(len(by_container[c]) for c in untouched)} implemented items.")
    w("")

    w("## The suite's shape")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| case modules | {len(families)} |")
    w(f"| committed recordings | {len(records)} |")
    runs = sorted(record["runs"] for record in records)
    w(f"| recorded runs behind a case, median | {runs[len(runs) // 2]} |")
    w(f"| recorded runs, least and most | {runs[0]} and {runs[-1]} |")
    w(f"| cases whose recording holds more than one outcome | {sum(1 for r in records if r['outcomes'] > 1)} |")
    w(f"| cases running a committed shader fixture | {sum(1 for r in records if r['shaders'])} |")
    provenance = Counter(kind for record in records for kind in record["provenance"])
    for kind, count in provenance.most_common():
        w(f"| cases whose Metal answer is `{kind}` | {count} |")
    w(f"| modules that loosen a measured value | {sum(1 for f in families.values() if f['allowances'])} |")
    w(f"| tolerances in all | {sum(f['allowances'] for f in families.values())} |")
    w(f"| modules declaring a notice (loosening no value) | {sum(1 for f in families.values() if f['declarations'])} |")
    loosenings = Counter(k for f in families.values() for k in f["loosenings"])
    w(f"| loosenings used | {', '.join(f'`{k}`' for k, _ in loosenings.most_common()) or 'none'} |")
    justs = Counter(k for f in families.values() for k in f["justifications"])
    w(f"| justifications named | {', '.join(f'`{k}`' for k, _ in justs.most_common()) or 'none'} |")
    w(f"| modules forbidding a re-record | {sum(1 for f in families.values() if f['carried_only'])} |")
    w(f"| `Side` methods (the suite's whole device vocabulary) | {len(data['side'])} |")
    w(f"| `Side` methods the Metal leg reaches an SDK item through | {sum(1 for m in data['gated'] if data['gated'][m])} |")
    per_case = sorted(
        len({key for key in by_item if module in by_item[key]})
        for record in records
        for module in [record["module"]]
    )
    w(f"| SDK items a case's Metal leg reaches, median | {per_case[len(per_case) // 2]} |")
    w(f"| the same, least and most | {per_case[0]} and {per_case[-1]} |")
    w(f"| Metal call sites in the leg | {len(leg.calls)} |")
    w(f"| distinct selectors the leg sends | {len({c.rust for c in leg.calls})} |")
    w(f"| of those, resolved to an item of the coverage map | {len({c.rust for c in leg.calls if id(c) in data['keyed']})} |")
    chips = Counter(chip for record in records for chip in record["chips"])
    w(f"| oracles the recordings name | {', '.join(f'`{c}` ×{n}' for c, n in chips.most_common())} |")
    w("")

    w("## By family")
    w("")
    w(
        "Each case module: the cases it holds, the `Side` methods it calls, and how many "
        "implemented items its Metal leg reaches through them."
    )
    w("")
    w("| family | cases | Metal ran | compared | items reached |")
    w("|---|---|---|---|---|")
    for module in sorted(families):
        family = families[module]
        here = {key for key in exercised if module in by_item[key]}
        exact = (
            "every field"
            if not family["allowances"]
            else f"{family['allowances']} tolerance(s): "
            + ", ".join(f"`{k}`" for k in family["justifications"])
        )
        w(
            f"| `{module}` | {len(family['cases'])} | {len(family['ran'])} | {exact} | "
            f"{len(here)} |"
        )
    w("")

    w("## Every implemented item")
    w("")
    w(
        "One row per implemented item of the SDK, in the coverage map's order. *exercised* is "
        "whether a case's native Metal leg calls it; *families* are the case modules that reach "
        "it; the last column is how many of their cases record an answer Metal actually ran."
    )
    w("")
    for container in sorted(by_container):
        w(f"### {_container(container)}")
        w("")
        w("| item | exercised | families | cases Metal ran |")
        w("|---|---|---|---|")
        for row in by_container[container]:
            key = (container, row["name"])
            modules = by_item.get(key, {})
            count = sum(len(families[module]["ran"]) for module in modules)
            w(
                f"| `{row['name']}` | {'yes' if key in exercised else 'no'} | "
                f"{families_line(modules) or '—'} | {count or '—'} |"
            )
        w("")

    w("## Exercised, but not ours")
    w("")
    w(
        "An item the suite's Metal leg calls whose coverage-map status is not implemented. Read "
        "each one before concluding anything: a call the leg makes to **set its own Metal up** is "
        "not a behaviour under test, and our side need not have it. A call the case's answer "
        "depends on is the other thing, and would be a case that can only be exact by accident."
    )
    w("")
    w("| item | status | note | families |")
    w("|---|---|---|---|")
    for key in sorted(by_item):
        row = index.get(key)
        if row is None or row["status"] == "I":
            continue
        w(
            f"| `{key[0]}` `{key[1]}` | {STATUS[row['status']]} | {row['note'] or '—'} | "
            f"{families_line(by_item[key])} |"
        )
    w("")

    w("## Called, attributed to no family")
    w("")
    w(
        "An item the Metal leg calls that no case module was found to reach: either a `Side` "
        "method no case calls, or a step of the vocabulary no case encodes. Each is suite code "
        "that no case runs."
    )
    w("")
    if data["unattributed"]:
        w("| item | status |")
        w("|---|---|")
        for key in data["unattributed"]:
            row = index.get(key)
            w(f"| `{key[0]}` `{key[1]}` | {STATUS[row['status']] if row else 'not in the map'} |")
    else:
        w("None: every item the leg calls is reached by a case module.")
    w("")

    if data["unlisted"]:
        w("## Called, not in the coverage map")
        w("")
        w(
            "The Metal leg sends these selectors to a receiver this typed, and "
            "`scripts/metal-coverage/coverage.tsv` holds no row for the item. Either the selector "
            "is `NSObject`'s, which `objc2-metal` generates on every class it binds and no Metal "
            "header declares, or the map is short of the SDK, or the name is spelled differently "
            "there; the last two are worth a look."
        )
        w("")
        w("| receiver | selector |")
        w("|---|---|")
        for ty, selector in data["unlisted"]:
            w(f"| `{ty}` | `{selector}` |")
        w("")

    w("## The ledger")
    w("")
    w(
        "A case whose difference from Metal is recorded rather than compared away. The ledger "
        "fails a run in both directions, so a row that no longer holds is a report of its own."
    )
    w("")
    if data["gaps"]:
        w("| case | waiting on |")
        w("|---|---|")
        for case, waiting in data["gaps"]:
            w(f"| `{case}` | {waiting[:160]}{'…' if len(waiting) > 160 else ''} |")
    else:
        w("No rows.")
    w("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------------
# Entry points


def check_api() -> int:
    """Hold `objc2_metal.tsv` to what `objc2-metal`'s own sources say."""
    found = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--manifest-path", str(SUITE / "Cargo.toml")],
        capture_output=True,
        text=True,
        check=True,
    )
    meta = json.loads(found.stdout)
    roots = [
        Path(package["manifest_path"]).parent / "src"
        for package in meta["packages"]
        if package["name"] == "objc2-metal"
    ]
    if not roots:
        print("objc2-metal is not in this workspace's dependency graph")
        return 1
    import objc2_items

    types, calls = objc2_items.rows(roots[0])
    want = "# type\tkind\tsupertypes\n" + "\n".join(types) + "\n# type\tselector\trust\tanswers\n"
    want += "\n".join(calls) + "\n"
    if API.read_text() != want:
        print(
            f"{API.relative_to(ROOT)} is not what objc2-metal's sources say: run\n"
            f"  python3 -I scripts/conformance-coverage/objc2_items.py {roots[0]} > "
            f"{API.relative_to(ROOT)}"
        )
        return 1
    print(f"{API.relative_to(ROOT)}: {len(calls)} bindings, as objc2-metal declares them")
    return 0


def selftest() -> int:
    """Run the option-space axis over inputs whose answers are known."""
    import unittest

    import test_option_space

    loaded = unittest.TestLoader().loadTestsFromModule(test_option_space)
    found = unittest.TextTestRunner(verbosity=2).run(loaded)
    return 0 if found.wasSuccessful() else 1


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    if "--api" in argv:
        return check_api()
    text = write(inventory())
    if "--check" in argv:
        if not DOC.exists() or DOC.read_text() != text:
            print(
                f"{DOC.relative_to(ROOT)} is not what the suite's sources and recordings derive: "
                "run\n  python3 -I scripts/conformance-coverage/coverage.py"
            )
            return 1
        print(f"{DOC.relative_to(ROOT)}: as derived")
        return 0
    DOC.write_text(text)
    print(f"wrote {DOC.relative_to(ROOT)} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
