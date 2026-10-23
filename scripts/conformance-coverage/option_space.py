"""The suite's depth in each Metal enum and option set: which of a type's values it compared.

`coverage.py`'s other axis is **reachability**: an item counts as exercised where a case's Metal
leg calls it, so one call is enough and a selector that takes an enum says nothing about how many
of that enum's values ever reached Metal. This is the second axis, over the values themselves.

The denominator is generated, never written here: `crates/metal2vulkan/src/metal/generated.rs` and
`crates/metal2vulkan-air/src/metal/generated.rs` hold every enumerator name and ordinal of the
Metal types `scripts/regen-metal-enums/` extracts from the SDK headers. A Metal type the tree
models by hand instead has **no denominator here** and is reported as such rather than guessed at;
the fix is appending its name to that script's lists and regenerating.

The numerator is two derived sets, which are different claims and are kept apart:

- **answered** — the enumerator's name appears in a committed recording's transcript of an answer
  Metal **ran**. Metal named the value back to us at a site the comparison reads, so the value is
  in the compared text itself.
- **in effect** — a case module's own text names the value, and that module has at least one case
  whose recording is a Metal run. Metal's recorded answer was taken with the value in effect, so
  the comparison is of Metal's behaviour **under** that value even where no site reads it back.

`compared` is their union. A value that only the suite's **shared** sources name (a default of the
render vocabulary, say) is counted in neither: some case puts it to Metal, but which case is not a
property of the text, so it is reported in its own column and is neither compared nor missing.

# Why a value stated as a raw ordinal is not resolved, and must not be

Both numerators read **names**. A case may instead hand the Metal leg an enum's ordinal as a
number — a sweep over `0..n`, or a literal field — and such a value appears nowhere by name, so
every column reads short of what the case does. That is stated as a limit in the document rather
than worked around, because the obvious way to close it is unsound:

the leg types some of these fields itself (`MTLPixelFormat(request.format as usize)` says that
`format` holds a pixel format's ordinal), which invites reading every `format: <number>` in the
suite as a pixel format. It is not one. `recording.rs` declares `format: u32` for the **recording
format's version**, so `format: 1` there is version 1 and not `MTLPixelFormatA8Unorm`, and a rule
keyed on the field's name would quietly credit the suite with a value nothing ever set. One
vocabulary's word for a thing is not another's.

So the floor stays a floor and says so. Closing it properly means a case stating its values by
name, or a transcript site recording the enumerator beside the ordinal it already stores — not a
cleverer reading of the text here.
"""

import json
import re
from pathlib import Path

import metal_leg

#: The generated files holding the SDK's enumerator names and ordinals.
GENERATED = (
    "crates/metal2vulkan/src/metal/generated.rs",
    "crates/metal2vulkan-air/src/metal/generated.rs",
)

#: The published crates' sources, where a Metal type the tree models by hand is declared.
BY_HAND = (
    "crates/metal2vulkan-runtime/src",
    "crates/metal2vulkan/src",
    "crates/metal2vulkan-air/src",
)

#: A recording answer whose transcript is Metal's own, rather than one the case carries.
RAN = "metal_run"


class Type:
    """One Metal enum or option set, with the SDK's values as the generated table holds them."""

    def __init__(self, metal: str, rust: str, kind: str, values: list[tuple[str, str]]):
        #: The Metal type name, `MTLBlendFactor`.
        self.metal = metal
        #: The Rust name the tree gives it, `BlendFactor`.
        self.rust = rust
        #: `enum` for an `NS_ENUM`, `flags` for an `NS_OPTIONS`.
        self.kind = kind
        #: Each value as `(Rust name, Metal enumerator name)`, in declaration order.
        self.values = values

    @property
    def names(self) -> list[str]:
        """The Metal enumerator names, in declaration order."""
        return [metal for _, metal in self.values]


def read_types(root: Path) -> dict[str, Type]:
    """Every Metal type the generated tables hold, by Metal name.

    `MTLDataType` is in both files (the front end's, which `metal` re-exports); the first reading
    wins, and the two are the same table.
    """
    out: dict[str, Type] = {}
    for name in GENERATED:
        path = root / name
        text = path.read_text()
        found = 0
        for block in re.finditer(r"metal_(enum|flags)!\s*\{(.*?)\n\}\n", text, re.S):
            kind, body = block.group(1), block.group(2)
            head = re.match(r".*?\b(\w+)\s*=\s*\"(MTL\w+)\"\s*\{", body, re.S)
            if head is None:
                raise SystemExit(f"{name}: a {kind} block with no `Rust = \"MTLName\" {{` head")
            declared, _, _ = body[head.end() :].partition("aliases {")
            values = re.findall(
                r"(\w+)\s*=\s*(?:0x[0-9a-fA-F]+|\d+),\s*\"(MTL\w+)\"\s*;", declared
            )
            if not values:
                raise SystemExit(f"{name}: {head.group(2)} parsed to no values")
            found += 1
            out.setdefault(head.group(2), Type(head.group(2), head.group(1), kind, values))
        if not found:
            # The parse silently matching nothing is the one way this axis can report a denominator
            # of zero and look like a closed option space, so it is a failure, not a count.
            raise SystemExit(f"{name}: no metal_enum!/metal_flags! block parsed")
    return out


def without_tests(text: str) -> str:
    """`blanked(text)` with every `#[cfg(test)]` item blanked too.

    A unit test naming a value is not the suite putting it to Metal, and one of them names four
    blend factors to **assert that no case uses them**. Counting those would have the axis report
    the opposite of what the code says.
    """
    out = list(metal_leg.blanked(text))
    for attr in re.finditer(r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]", "".join(out)):
        i = attr.end()
        n = len(out)
        # Skip the attributes between this one and the item it sits on.
        while True:
            while i < n and out[i].isspace():
                i += 1
            if i < n and out[i] == "#":
                depth = 0
                while i < n:
                    if out[i] in "([{":
                        depth += 1
                    elif out[i] in ")]}":
                        depth -= 1
                        if depth == 0:
                            i += 1
                            break
                    i += 1
                continue
            break
        # The item ends at its body's closing brace, or at the `;` of one with no body. A `;` or a
        # brace inside brackets (`[u8; 4]`) is the type's, not the item's.
        depth = 0
        end = None
        while i < n:
            char = out[i]
            if char in "([":
                depth += 1
            elif char in ")]":
                depth -= 1
            elif depth == 0 and char == "{":
                _, end = metal_leg.block("".join(out), i)
                break
            elif depth == 0 and char == ";":
                end = i + 1
                break
            i += 1
        if end is None:
            raise SystemExit("a #[cfg(test)] item with neither a body nor a `;`")
        for at in range(attr.start(), end):
            if not out[at].isspace():
                out[at] = " "
    return "".join(out)


def _stated_in(text: str, by_rust: dict[str, Type], by_metal: dict[str, Type]):
    """The `(Metal type, value)` pairs `text` names, in either of the two spellings it uses.

    The suite names a Metal value two ways, and both have to be read:

    - through the tree's own type, `BlendFactor::SourceAlpha` — and `rotated_targets.rs` writes
      `use BlendFactor as F;` and then every factor as `F::SourceAlpha`, so a derivation reading
      only `BlendFactor::` spellings reports a third of that module's blend factors as never
      exercised;
    - through `objc2-metal`'s own enum, `MTLStorageMode::Private`, which the Metal leg writes
      directly. `objc2` names a variant with its type's prefix dropped, so the enumerator is the
      two spelled back together, and it counts only where that concatenation is a name the
      generated table holds — never by assuming the rule held.
    """
    alias: dict[str, str] = {}
    for found in re.finditer(r"\b([A-Za-z_]\w*)\s+as\s+([A-Za-z_]\w*)\b", text):
        if found.group(1) in by_rust:
            alias[found.group(2)] = found.group(1)
    out: set[tuple[str, str]] = set()
    for found in re.finditer(r"\b([A-Za-z_]\w*)::([A-Za-z_]\w*)\b", text):
        head, tail = found.group(1), found.group(2)
        held = by_rust.get(alias.get(head, head))
        if held is not None:
            for rust, value in held.values:
                if rust == tail:
                    out.add((held.metal, value))
        held = by_metal.get(head)
        if held is None:
            continue
        if head + tail in held.names:
            out.add((held.metal, head + tail))
        elif held.kind == "by hand" and tail in held.names:
            # A hand-modelled type has no SDK enumerator list, so our own variant name is the only
            # name there is; `objc2` spells the same value the same way after the prefix.
            out.add((held.metal, tail))
    return out


def stated(root: Path, types: dict[str, Type], ran: set[str]):
    """The values each case module names, and the ones only the suite's shared sources name.

    `ran` is the case modules with at least one committed recording of an answer Metal ran; a
    module not in it puts nothing to Metal however much its text names.
    """
    by_rust = {held.rust: held for held in types.values()}
    by_metal = {held.metal: held for held in types.values()}
    src = root / "crates/metal2vulkan-conformance/src"
    in_effect: dict[tuple[str, str], set[str]] = {}
    for path in sorted((src / "cases").glob("*.rs")):
        if path.stem not in ran:
            continue
        for pair in _stated_in(without_tests(path.read_text()), by_rust, by_metal):
            in_effect.setdefault(pair, set()).add(path.stem)
    shared: set[tuple[str, str]] = set()
    for path in sorted(src.rglob("*.rs")):
        held = path.relative_to(src)
        if held.parts[0] in ("cases", "tests", "bin") or held.name == "tests.rs":
            continue
        shared |= _stated_in(without_tests(path.read_text()), by_rust, by_metal)
    return in_effect, shared


def _tokens(value, key: str | None, out: list[str]) -> None:
    """Every identifier-shaped token of a transcript value, less the base64 of its bytes."""
    if isinstance(value, str):
        if key != "bytes":
            out.extend(re.findall(r"[A-Za-z0-9_]+", value))
    elif isinstance(value, dict):
        for name, held in value.items():
            _tokens(held, name, out)
    elif isinstance(value, list):
        for held in value:
            _tokens(held, key, out)


def answered(root: Path, types: dict[str, Type]) -> dict[tuple[str, str], set[str]]:
    """The values a committed recording's compared transcript holds, by the cases holding them.

    Only `answers[].outcomes[].transcript` is read. An answer's `validated` leg is Metal's API
    validation, which the suite records and **never compares**, and its complaints name
    enumerators; reading it would count Apple's prose as a comparison. The cost and timing tables
    are not read either: a site name there is what was measured, not an answer that was compared.
    """
    of_name = {metal: held.metal for held in types.values() for metal in held.names}
    out: dict[tuple[str, str], set[str]] = {}
    for path in sorted((root / "crates/metal2vulkan-conformance/recordings").glob("*.json")):
        data = json.loads(path.read_text())
        found: list[str] = []
        for answer in data.get("answers", []):
            if answer.get("provenance") != RAN:
                continue
            for outcome in answer.get("outcomes", []):
                _tokens(outcome.get("transcript"), None, found)
        for token in set(found):
            if token in of_name:
                out.setdefault((of_name[token], token), set()).add(data["case"])
    return out


#: A Rust item the tree declares to be one Metal enum or option set: a doc comment naming the
#: Metal type, then the item it documents, which is a closed enum (an `NS_ENUM`) or a newtype over
#: one integer (an `NS_OPTIONS`). A multi-field struct is a Metal *structure*, and a class or
#: descriptor is neither, so the shape of the item is what tells the three apart.
MODELS = re.compile(
    r"///\s*`(MTL\w+)`[^\n]*\n(?:[ \t]*(?:///[^\n]*|#\[[^\n]*\])\n)*"
    r"[ \t]*pub\s+(?:(enum)\s+([A-Za-z_]\w*)\s*\{"
    r"|(struct)\s+([A-Za-z_]\w*)\s*\(\s*u(?:8|16|32|64|size)\s*\)\s*;)"
)


def _opens_the_doc(source: str, at: int) -> bool:
    """Whether the doc line at `at` is the first of its item's doc block.

    The generated enums write the Metal name on the first line, and so do the hand-written ones. A
    name mentioned part-way through a paragraph is prose about a Metal type, not a declaration of
    being one: `CommandDescriptor`'s doc says it models `MTLIndirectCommandBufferDescriptor` "as
    far as the record format needs it", which is a description and not that enum.
    """
    before = source.rfind("\n", 0, at)
    if before < 0:
        return True
    line = source[source.rfind("\n", 0, before) + 1 : before]
    return not line.lstrip().startswith("///")


def by_hand(root: Path, types: dict[str, Type]) -> list[tuple[str, str, list[str]]]:
    """The Metal enums and option sets the tree models by hand, with our own variant names.

    Found by shape: a doc comment naming a Metal type on an item that is a closed Rust enum or a
    newtype over one integer. These have **no generated value list**, so this axis has no
    denominator for them, and the names returned are our own sources', not the SDK's. An option-set
    newtype declares no variants and returns an empty list.

    The list is itself a floor: a type the tree models without saying so in a doc comment on the
    item is in neither table and nowhere in this axis.
    """
    out: dict[str, tuple[str, list[str]]] = {}
    for where in BY_HAND:
        for path in sorted((root / where).rglob("*.rs")):
            if "tests" in path.parts or path.name in ("generated.rs", "tests.rs"):
                continue
            source = path.read_text()
            # `blanked` keeps every offset, so the brace walk below reads against the source. It
            # is a character loop over the whole file, so it runs only where a match needs it:
            # almost no file in these crates declares one of these types.
            text: str | None = None
            for found in MODELS.finditer(source):
                metal = found.group(1)
                if metal in types or not _opens_the_doc(source, found.start()):
                    continue
                if found.group(2):
                    if text is None:
                        text = metal_leg.blanked(source)
                    start, end = metal_leg.block(text, found.end() - 1)
                    names = re.findall(r"^\s*([A-Z]\w*)\s*(?:,|\{|\(|$)", text[start:end], re.M)
                    out[metal] = (found.group(3), names)
                else:
                    out[metal] = (found.group(5), [])
    return [(metal, rust, names) for metal, (rust, names) in sorted(out.items())]


def inventory(root: Path, ran: set[str]) -> dict:
    """The axis, derived: every type with its three sets of values."""
    types = read_types(root)
    hand = by_hand(root, types)
    # A hand-modelled type is carried through the same resolution, keyed by its own Rust variant
    # names: this axis has no SDK enumerator list for it, so a variant is all there is to name.
    held = dict(types)
    for metal, rust, names in hand:
        held[metal] = Type(metal, rust, "by hand", [(name, name) for name in names])
    effect, shared = stated(root, held, ran)
    read_back = answered(root, types)
    rows = []
    for metal, held in sorted(types.items()):
        compared, only_shared, missing = [], [], []
        for name in held.names:
            key = (metal, name)
            if key in read_back or key in effect:
                compared.append(name)
            elif key in shared:
                only_shared.append(name)
            else:
                missing.append(name)
        rows.append(
            {
                "metal": metal,
                "rust": held.rust,
                "kind": held.kind,
                "values": held.names,
                "compared": compared,
                "answered": [n for n in held.names if (metal, n) in read_back],
                "in_effect": [n for n in held.names if (metal, n) in effect],
                "shared": only_shared,
                "missing": missing,
                "modules": sorted({m for n in held.names for m in effect.get((metal, n), ())}),
            }
        )
    hand_rows = [
        {
            "metal": metal,
            "rust": rust,
            "declared": names,
            "stated": [name for name in names if (metal, name) in effect],
            "shared": [
                name
                for name in names
                if (metal, name) not in effect and (metal, name) in shared
            ],
        }
        for metal, rust, names in hand
    ]
    return {
        "rows": rows,
        "by_hand": hand_rows,
        "answered": read_back,
        "in_effect": effect,
    }
