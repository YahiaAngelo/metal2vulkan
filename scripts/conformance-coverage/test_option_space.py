"""The option-space axis over inputs whose answers are known.

Run by `coverage.py --selftest`, which is what the suite's
`the_option_space_axis_still_measures_what_it_claims` runs.

The axis is a derivation over text, and the way a derivation like it fails is by quietly matching
nothing: a renamed JSON key, a spelling the sources use that the regex does not, and every count
reads zero while the document still looks like a measurement. So each test here builds its own
little tree, where the answer is known by construction, and asserts both directions — that a value
it should find is found, and that a value it must not count is not.

Every test owns its inputs. Nothing here reads the repository.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import option_space

#: Every tree `tree` made, removed by `tearDownModule`. Each test wants a fresh one and none of
#: them wants it afterwards, so one list and one sweep is the whole lifetime.
MADE: list[Path] = []


def tearDownModule() -> None:
    """Remove the trees the tests built."""
    while MADE:
        shutil.rmtree(MADE.pop(), ignore_errors=True)

#: A generated table of one enum and one option set, as `regen-metal-enums` writes them.
GENERATED = """\
// Generated from the macOS 27.0 SDK Metal.framework headers (names and values).

metal_enum! {
    /// `MTLThing`.
    #[non_exhaustive]
    Thing = "MTLThing" {
        Alpha = 0, "MTLThingAlpha";
        Beta = 1, "MTLThingBeta";
        Gamma = 2, "MTLThingGamma";
        Delta = 3, "MTLThingDelta";
        Epsilon = 4, "MTLThingEpsilon";
        Zeta = 5, "MTLThingZeta";
    }
    aliases {
        "MTLThingOldBeta" => Beta;
    }
}

metal_flags! {
    /// `MTLSpread`.
    Spread = "MTLSpread" {
        NONE = 0, "MTLSpreadNone";
        LEFT = 0x1, "MTLSpreadLeft";
        RIGHT = 0x2, "MTLSpreadRight";
    }
    aliases {
    }
    dropped {
    }
}
"""


def tree(
    cases: dict[str, str],
    shared: str = "",
    recordings: tuple[dict, ...] = (),
    generated: str = GENERATED,
) -> Path:
    """A repository-shaped tree of exactly these sources and recordings."""
    root = Path(tempfile.mkdtemp(prefix="option-space-selftest-"))
    MADE.append(root)
    for name in option_space.GENERATED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # The front end's file holds a different type in the tree; here one copy is enough, and
        # `read_types` keeping the first reading is what this relies on.
        path.write_text(generated)
    for name in option_space.BY_HAND:
        (root / name).mkdir(parents=True, exist_ok=True)
    suite = root / "crates/metal2vulkan-conformance"
    (suite / "src/cases").mkdir(parents=True, exist_ok=True)
    (suite / "recordings").mkdir(parents=True, exist_ok=True)
    for module, text in cases.items():
        (suite / "src/cases" / f"{module}.rs").write_text(text)
    (suite / "src/side.rs").write_text(shared)
    for data in recordings:
        (suite / "recordings" / f"{data['case']}.json").write_text(json.dumps(data))
    return root


def recording(case: str, value: str, provenance: str = option_space.RAN, where: str = "ran") -> dict:
    """A recording of `case` whose transcript, or whose validated leg, names `value`."""
    node = {
        "name": "case",
        "observed": {"value": {"fields": {"factor": {"fields": {"name": {"lines": [value]}}}}}},
    }
    answer = {"provenance": provenance, "runs": 1, "outcomes": []}
    if where == "ran":
        answer["outcomes"] = [{"runs": 1, "transcript": {"root": node}}]
    else:
        answer["outcomes"] = [{"runs": 1, "transcript": {"root": {"name": "case"}}}]
        answer["validated"] = {"stopped": {"message": f"Thing Validation\n{value} is not allowed"}}
    return {"format": 1, "case": case, "suite_version": 1, "answers": [answer]}


def axis(root: Path, ran: set[str]):
    """The axis over `root`, as the one row of `MTLThing`."""
    held = option_space.inventory(root, ran)
    return next(row for row in held["rows"] if row["metal"] == "MTLThing")


class Denominator(unittest.TestCase):
    """The value list comes from the generated tables, and a parse of nothing is a failure."""

    def test_the_values_are_the_generated_table_s(self):
        row = axis(tree({}), set())
        self.assertEqual(len(row["values"]), 6)
        self.assertEqual(row["values"][0], "MTLThingAlpha")
        self.assertEqual(row["compared"], [])
        self.assertEqual(row["missing"], row["values"])

    def test_an_option_set_s_named_combinations_are_values(self):
        held = option_space.inventory(tree({}), set())
        row = next(r for r in held["rows"] if r["metal"] == "MTLSpread")
        self.assertEqual(row["values"], ["MTLSpreadNone", "MTLSpreadLeft", "MTLSpreadRight"])

    def test_a_generated_file_that_parses_to_nothing_fails(self):
        with self.assertRaises(SystemExit):
            option_space.read_types(tree({}, generated="// nothing at all\n"))

    def test_a_type_that_parses_to_no_values_fails(self):
        broken = GENERATED.replace('Alpha = 0, "MTLThingAlpha";', "").replace(
            'Beta = 1, "MTLThingBeta";', ""
        )
        broken = broken.replace('Gamma = 2, "MTLThingGamma";', "")
        broken = broken.replace('Delta = 3, "MTLThingDelta";', "")
        broken = broken.replace('Epsilon = 4, "MTLThingEpsilon";', "")
        broken = broken.replace('Zeta = 5, "MTLThingZeta";', "")
        with self.assertRaises(SystemExit):
            option_space.read_types(tree({}, generated=broken))


class InEffect(unittest.TestCase):
    """A value a recorded case module states is in effect; one it does not is not."""

    def test_a_value_a_case_module_names_is_in_effect(self):
        row = axis(tree({"blends": "fn go() { let x = Thing::Beta; }"}), {"blends"})
        self.assertEqual(row["in_effect"], ["MTLThingBeta"])
        self.assertIn("MTLThingBeta", row["compared"])
        self.assertNotIn("MTLThingBeta", row["missing"])

    def test_an_aliased_spelling_is_in_effect(self):
        # `use Thing as T;` and then `T::Gamma`, which is how the blend sweeps are written. A
        # derivation reading only `Thing::` spellings reports this value as never exercised.
        row = axis(tree({"blends": "use Thing as T;\nfn go() { let x = T::Gamma; }"}), {"blends"})
        self.assertEqual(row["in_effect"], ["MTLThingGamma"])

    def test_the_objc2_spelling_is_in_effect(self):
        # `MTLThing::Delta`, which is how the Metal leg names a fixed value: `objc2` drops the
        # type's prefix, so the enumerator is the two spelled back together.
        row = axis(tree({"blends": "fn go() { d.set(MTLThing::Delta); }"}), {"blends"})
        self.assertEqual(row["in_effect"], ["MTLThingDelta"])

    def test_an_objc2_spelling_of_no_generated_name_is_not_counted(self):
        row = axis(tree({"blends": "fn go() { d.set(MTLThing::Omega); }"}), {"blends"})
        self.assertEqual(row["in_effect"], [])

    def test_a_module_with_no_metal_run_recording_puts_nothing_in_effect(self):
        text = "fn go() { let x = Thing::Beta; }"
        self.assertEqual(axis(tree({"blends": text}), set())["in_effect"], [])
        self.assertEqual(axis(tree({"blends": text}), {"blends"})["in_effect"], ["MTLThingBeta"])

    def test_a_cfg_test_item_is_not_in_effect(self):
        text = (
            "fn go() { let x = Thing::Beta; }\n"
            "#[cfg(test)]\nmod tests {\n"
            "    #[test]\n    fn refused() { assert!(!matches!(f, Thing::Zeta)); }\n"
            "}\n"
        )
        row = axis(tree({"blends": text}), {"blends"})
        self.assertEqual(row["in_effect"], ["MTLThingBeta"])
        self.assertIn("MTLThingZeta", row["missing"])

    def test_a_cfg_test_item_with_no_body_is_blanked_to_its_semicolon(self):
        text = (
            "#[cfg(test)]\nconst WIDTHS: [u8; 4] = [Thing::Zeta as u8; 4];\n"
            "fn go() { let x = Thing::Beta; }\n"
        )
        row = axis(tree({"blends": text}), {"blends"})
        self.assertEqual(row["in_effect"], ["MTLThingBeta"])

    def test_a_sweep_by_ordinal_states_nothing(self):
        # The boundary of this column, kept visible: a case that hands the leg `0..n` reaches every
        # value and names none, so the column reads zero and the document says it is a floor by
        # exactly this. There is no sound textual reading of a computed ordinal, so the test holds
        # the limit rather than a fix for it.
        text = "fn go() { for raw in 0..6 { side.set_thing(raw); } }"
        self.assertEqual(axis(tree({"sweep": text}), {"sweep"})["in_effect"], [])

    def test_a_comment_naming_a_value_is_not_in_effect(self):
        row = axis(tree({"blends": "// Thing::Beta is what Metal defaults to.\nfn go() {}"}), {"blends"})
        self.assertEqual(row["in_effect"], [])


class Answered(unittest.TestCase):
    """A value a recording's compared transcript holds is answered; the other legs are not read."""

    def test_a_value_in_a_run_transcript_is_answered(self):
        root = tree({}, recordings=(recording("one", "MTLThingGamma"),))
        row = axis(root, set())
        self.assertEqual(row["answered"], ["MTLThingGamma"])
        self.assertIn("MTLThingGamma", row["compared"])

    def test_a_carried_answer_is_not_answered(self):
        root = tree({}, recordings=(recording("one", "MTLThingGamma", provenance="carried"),))
        row = axis(root, set())
        self.assertEqual(row["answered"], [])
        self.assertIn("MTLThingGamma", row["missing"])

    def test_the_validated_leg_is_not_answered(self):
        root = tree({}, recordings=(recording("one", "MTLThingGamma", where="validated"),))
        row = axis(root, set())
        self.assertEqual(row["answered"], [])

    def test_base64_bytes_are_not_read_as_names(self):
        # A recorded buffer is base64, and the alphabet is alphanumeric, so a payload **can** spell
        # an enumerator name exactly: `MTLThingZeta` is twelve characters of the base64 alphabet,
        # which is to say nine bytes of some texture. Reading it would credit the suite with a
        # value from the contents of a buffer.
        data = recording("one", "MTLThingGamma")
        data["answers"][0]["outcomes"][0]["transcript"]["root"]["observed"] = {
            "value": {"bytes": "MTLThingZeta"}
        }
        row = axis(tree({}, recordings=(data,)), set())
        self.assertEqual(row["answered"], [])

    def test_a_compound_site_name_is_split_into_its_tokens(self):
        data = recording("one", "MTLThingGamma")
        data["answers"][0]["outcomes"][0]["transcript"]["root"] = {
            "name": "MTLThingZeta/TextureBuffer/64x1x1",
            "observed": {"value": "absent"},
        }
        row = axis(tree({}, recordings=(data,)), set())
        self.assertEqual(row["answered"], ["MTLThingZeta"])


class SharedOnly(unittest.TestCase):
    """A value only the suite's shared sources name is its own column, not a gap."""

    def test_a_value_only_shared_sources_name_is_neither_compared_nor_missing(self):
        row = axis(tree({}, shared="fn go() { d.set(Thing::Delta); }"), set())
        self.assertEqual(row["shared"], ["MTLThingDelta"])
        self.assertNotIn("MTLThingDelta", row["compared"])
        self.assertNotIn("MTLThingDelta", row["missing"])

    def test_a_case_module_naming_it_takes_it_out_of_shared(self):
        root = tree(
            {"blends": "fn go() { let x = Thing::Delta; }"},
            shared="fn go() { d.set(Thing::Delta); }",
        )
        row = axis(root, {"blends"})
        self.assertEqual(row["shared"], [])
        self.assertEqual(row["in_effect"], ["MTLThingDelta"])

    def test_every_value_is_in_exactly_one_column(self):
        root = tree(
            {"blends": "use Thing as T;\nfn go() { let x = T::Beta; }"},
            shared="fn go() { d.set(Thing::Delta); }",
            recordings=(recording("one", "MTLThingGamma"),),
        )
        row = axis(root, {"blends"})
        self.assertEqual(
            len(row["compared"]) + len(row["shared"]) + len(row["missing"]), len(row["values"])
        )
        self.assertEqual(sorted(row["compared"]), ["MTLThingBeta", "MTLThingGamma"])
        self.assertEqual(row["shared"], ["MTLThingDelta"])
        self.assertEqual(row["missing"], ["MTLThingAlpha", "MTLThingEpsilon", "MTLThingZeta"])


class ByHand(unittest.TestCase):
    """A Metal type the tree models by hand is reported apart, with no denominator."""

    SOURCE = """\
/// `MTLThingy`.
pub enum Thingy {
    /// `MTLThingyFirst`.
    First,
    /// `MTLThingySecond`.
    Second,
}

impl Thingy {
    pub fn from_raw(raw: u64) -> Result<Thingy, Error> {
        Ok(match raw {
            0 => Thingy::First,
            1 => Thingy::Second,
            raw => return Err(undefined("MTLThingy", raw)),
        })
    }
}
"""

    def held(self, cases: dict[str, str], ran: set[str]):
        root = tree(cases)
        (root / option_space.BY_HAND[0] / "values.rs").write_text(self.SOURCE)
        found = option_space.inventory(root, ran)
        return next(row for row in found["by_hand"] if row["metal"] == "MTLThingy")

    def test_a_hand_modelled_type_is_found_with_its_own_variants(self):
        row = self.held({}, set())
        self.assertEqual(row["rust"], "Thingy")
        self.assertEqual(row["declared"], ["First", "Second"])
        self.assertEqual(row["stated"], [])

    def test_a_recorded_case_naming_one_states_it(self):
        row = self.held({"blends": "fn go() { let x = Thingy::Second; }"}, {"blends"})
        self.assertEqual(row["stated"], ["Second"])

    def test_the_objc2_spelling_of_a_hand_modelled_type_states_it(self):
        row = self.held({"blends": "fn go() { c.setThing(MTLThingy::First); }"}, {"blends"})
        self.assertEqual(row["stated"], ["First"])

    def test_an_option_set_newtype_is_listed_with_no_variants(self):
        root = tree({})
        (root / option_space.BY_HAND[0] / "values.rs").write_text(
            "/// `MTLThingOptions`.\n#[repr(transparent)]\npub struct ThingOptions(u64);\n"
        )
        found = option_space.inventory(root, set())
        row = next(r for r in found["by_hand"] if r["metal"] == "MTLThingOptions")
        self.assertEqual((row["rust"], row["declared"]), ("ThingOptions", []))

    def test_a_metal_structure_is_not_an_option_space(self):
        # `MTLOrigin` is a structure of three numbers, not an enum; it has no values to count.
        root = tree({})
        (root / option_space.BY_HAND[0] / "values.rs").write_text(
            "/// `MTLWhere`.\npub struct Where {\n    pub x: u64,\n    pub y: u64,\n}\n"
        )
        self.assertEqual(option_space.inventory(root, set())["by_hand"], [])

    def test_a_metal_name_part_way_through_a_doc_is_prose(self):
        # A type that says it models part of a Metal type is describing itself, not declaring that
        # it is that type's value list.
        root = tree({})
        (root / option_space.BY_HAND[0] / "values.rs").write_text(
            "/// What a buffer allows (Metal's\n"
            "/// `MTLThingy`, as far as the format needs it): the kinds it takes.\n"
            "pub enum Shape {\n    First,\n    Second,\n}\n"
        )
        self.assertEqual(option_space.inventory(root, set())["by_hand"], [])

    def test_a_generated_type_is_not_listed_as_hand_modelled(self):
        root = tree({})
        (root / option_space.BY_HAND[0] / "values.rs").write_text(
            self.SOURCE.replace("MTLThingy", "MTLThing")
        )
        found = option_space.inventory(root, set())
        self.assertEqual([row["metal"] for row in found["by_hand"]], [])


if __name__ == "__main__":
    unittest.main()
