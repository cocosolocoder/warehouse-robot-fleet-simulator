"""Regression coverage for the two map definitions inside checkpoint files.

A version 2 checkpoint carries two maps: ``grid`` is the map in use when the
checkpoint was saved, while ``base_grid`` is the map as built before any
runtime obstacle edit. Each keeps its own width, height and obstacle set, and
the recorded ``map_changes`` history must turn the latter into the former.
Every test here goes through the public ``FleetSimulator.load_checkpoint``
entry (the same way real callers restore a save), pinning both the accepted
restoration results and the existing ``ValueError`` wording: format errors say
whether they come from the current or the initial map and keep the offending
obstacle's index, while disagreement between two individually legal maps is
reported as the separate consistency problem it is.
"""

import copy
import json
import os
import tempfile
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


def _load(document: object) -> FleetSimulator:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return FleetSimulator.load_checkpoint(path)


def _saved_document(simulator: FleetSimulator) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        simulator.save_checkpoint(path)
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)


def _plain_document() -> dict[str, object]:
    """A valid zero-tick v2 checkpoint on a 5x2 map with one robot."""
    simulator = FleetSimulator(GridMap(5, 2), [Robot("R", (0, 0))], [])
    return _saved_document(simulator)


def _as_version_one(document: dict[str, object]) -> dict[str, object]:
    document["version"] = 1
    for key in ("base_grid", "map_changes", "paused_tasks", "traffic_waits"):
        document.pop(key, None)
    document["replay"] = [
        {key: value for key, value in frame.items() if key != "type"}
        for frame in document["replay"]
    ]
    return document


def _expect_rejected(testcase: unittest.TestCase, document: object, fragment: str) -> str:
    with testcase.assertRaises(ValueError) as context:
        _load(document)
    message = str(context.exception)
    testcase.assertIn(fragment, message)
    return message


class MapDefinitionRecoveryTests(unittest.TestCase):
    """Legal saves restore both maps independently."""

    def test_reopened_cell_is_open_now_but_still_closed_on_initial_map(self) -> None:
        # A genuine edit archive: cell (2, 0) starts closed, is reopened before
        # the first tick, and the task drives through it. The two restored
        # maps must disagree on exactly that cell while sharing the history.
        simulator = FleetSimulator(
            GridMap(5, 1, frozenset({(2, 0)})),
            [Robot("R", (0, 0))],
            [Task("T", (0, 0), (4, 0))],
        )
        simulator.modify_obstacles(removed=[(2, 0)])
        for _ in range(4):
            simulator.step()
        self.assertTrue(simulator.tasks["T"].completed)
        document = _saved_document(simulator)

        loaded = _load(document)

        self.assertEqual((loaded.grid.width, loaded.grid.height), (5, 1))
        self.assertEqual((loaded.base_grid.width, loaded.base_grid.height), (5, 1))
        self.assertIsNot(loaded.grid, loaded.base_grid)
        # The saved (current) map is the post-edit map: the reopened cell is
        # traversable and carries no obstacle.
        self.assertEqual(loaded.grid.obstacles, frozenset())
        self.assertTrue(loaded.grid.traversable((2, 0)))
        # The initial map still records the cell as it was built: closed.
        self.assertEqual(loaded.base_grid.obstacles, frozenset({(2, 0)}))
        self.assertFalse(loaded.base_grid.traversable((2, 0)))
        # The current obstacles must never be copied back into the initial map,
        # nor the other way round.
        self.assertNotEqual(loaded.grid.obstacles, loaded.base_grid.obstacles)
        # The edit that explains the difference is accepted as history.
        history = loaded.map_change_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["removed"], [[2, 0]])
        self.assertEqual(history[0]["tick"], 0)
        self.assertTrue(loaded.tasks["T"].completed)

    def test_empty_obstacle_lists_are_legal_for_both_maps(self) -> None:
        loaded = _load(_plain_document())
        self.assertEqual(loaded.grid.obstacles, frozenset())
        self.assertEqual(loaded.base_grid.obstacles, frozenset())
        self.assertTrue(loaded.grid.traversable((4, 1)))
        self.assertTrue(loaded.base_grid.traversable((4, 1)))

    def test_duplicate_coordinate_loads_as_one_obstacle(self) -> None:
        # (1, 0) is a pre-existing obstacle present in both definitions; (2, 0)
        # is added by a tick-0 edit. The current map lists (2, 0) twice and the
        # initial map lists (1, 0) twice: each duplicate collapses to a single
        # obstacle, the file still loads, and the other obstacle is untouched.
        document = _plain_document()
        document["base_grid"]["obstacles"] = [[1, 0], [1, 0]]
        document["grid"]["obstacles"] = [[1, 0], [2, 0], [2, 0]]
        change = {
            "type": "map_change",
            "tick": 0,
            "sequence": 1,
            "added": [[2, 0]],
            "removed": [],
        }
        document["map_changes"] = [
            {key: value for key, value in change.items() if key != "type"}
        ]
        document["replay"] = [copy.deepcopy(change)]

        loaded = _load(document)

        self.assertEqual(loaded.base_grid.obstacles, frozenset({(1, 0)}))
        self.assertEqual(loaded.grid.obstacles, frozenset({(1, 0), (2, 0)}))
        # The shared obstacle survives and the duplicated one still blocks.
        self.assertFalse(loaded.grid.traversable((1, 0)))
        self.assertFalse(loaded.grid.traversable((2, 0)))
        self.assertTrue(loaded.grid.traversable((3, 0)))
        # And the duplicate in the current map was never treated as an edit:
        # the initial map stays free of (2, 0).
        self.assertTrue(loaded.base_grid.traversable((2, 0)))
        self.assertEqual(len(loaded.map_change_history()), 1)

    def test_maps_keep_their_own_dimensions(self) -> None:
        # Two legal, equally sized but differently shaped obstacle sets: the
        # restore result exposes exactly what each definition recorded.
        document = _plain_document()
        document["base_grid"]["obstacles"] = [[1, 0]]
        document["grid"]["obstacles"] = [[1, 0], [3, 1]]
        change = {
            "type": "map_change",
            "tick": 0,
            "sequence": 1,
            "added": [[3, 1]],
            "removed": [],
        }
        document["map_changes"] = [
            {key: value for key, value in change.items() if key != "type"}
        ]
        document["replay"] = [copy.deepcopy(change)]

        loaded = _load(document)

        self.assertEqual(loaded.grid.width, loaded.base_grid.width)
        self.assertEqual(loaded.grid.height, loaded.base_grid.height)
        self.assertEqual(loaded.base_grid.obstacles, frozenset({(1, 0)}))
        self.assertEqual(loaded.grid.obstacles, frozenset({(1, 0), (3, 1)}))


class VersionOneBaselineTests(unittest.TestCase):
    """Version 1 files have no initial-map field and keep the old convention."""

    def test_saved_grid_is_taken_as_the_initial_map(self) -> None:
        simulator = FleetSimulator(
            GridMap(5, 1, frozenset({(1, 0)})), [Robot("R", (0, 0))], []
        )
        document = _as_version_one(_saved_document(simulator))
        self.assertNotIn("base_grid", document)

        loaded = _load(document)

        # No version 2 fields are demanded; the one saved map is the baseline.
        self.assertEqual(loaded.base_grid.width, 5)
        self.assertEqual(loaded.base_grid.height, 1)
        self.assertEqual(loaded.base_grid.obstacles, frozenset({(1, 0)}))
        self.assertEqual(loaded.grid.obstacles, loaded.base_grid.obstacles)
        self.assertEqual(loaded.map_change_history(), [])

    def test_version_one_replay_still_loads_without_v2_fields(self) -> None:
        simulator = FleetSimulator(
            GridMap(4, 1), [Robot("R", (0, 0))], [Task("T", (0, 0), (3, 0))]
        )
        simulator.step()
        document = _as_version_one(_saved_document(simulator))

        loaded = _load(document)

        self.assertEqual(loaded.tick, 1)
        self.assertEqual(loaded.base_grid.obstacles, loaded.grid.obstacles)
        loaded.step()
        loaded.step()
        self.assertTrue(loaded.tasks["T"].completed)


# (label, mutation, grid-fragment, base-grid-fragment)
# Each mutation breaks exactly one of the two map definitions while leaving
# the other legal. The paired fragments pin the existing wording: they prove
# the report identifies the current map (``grid``) versus the initial map
# (``base grid`` / ``base obstacle``) and keeps an offending obstacle's index.
FORMAT_MUTATIONS = [
    (
        "definition missing",
        lambda doc, section: doc.pop(section),
        "missing required field 'grid'",
        "missing required field 'base_grid'",
    ),
    (
        "definition not an object",
        lambda doc, section: doc.__setitem__(section, []),
        "field 'grid' must be an object",
        "field 'base_grid' must be an object",
    ),
    (
        "definition null",
        lambda doc, section: doc.__setitem__(section, None),
        "field 'grid' must be an object",
        "field 'base_grid' must be an object",
    ),
    (
        "width missing",
        lambda doc, section: doc[section].pop("width"),
        "grid width must be a positive integer",
        "base grid width must be a positive integer",
    ),
    (
        "width boolean",
        lambda doc, section: doc[section].__setitem__("width", True),
        "grid width must be a positive integer",
        "base grid width must be a positive integer",
    ),
    (
        "width zero",
        lambda doc, section: doc[section].__setitem__("width", 0),
        "grid width must be a positive integer",
        "base grid width must be a positive integer",
    ),
    (
        "width negative",
        lambda doc, section: doc[section].__setitem__("width", -2),
        "grid width must be a positive integer",
        "base grid width must be a positive integer",
    ),
    (
        "width float",
        lambda doc, section: doc[section].__setitem__("width", 2.0),
        "grid width must be a positive integer",
        "base grid width must be a positive integer",
    ),
    (
        "width string",
        lambda doc, section: doc[section].__setitem__("width", "5"),
        "grid width must be a positive integer",
        "base grid width must be a positive integer",
    ),
    (
        "height boolean",
        lambda doc, section: doc[section].__setitem__("height", False),
        "grid height must be a positive integer",
        "base grid height must be a positive integer",
    ),
    (
        "height zero",
        lambda doc, section: doc[section].__setitem__("height", 0),
        "grid height must be a positive integer",
        "base grid height must be a positive integer",
    ),
    (
        "height negative",
        lambda doc, section: doc[section].__setitem__("height", -1),
        "grid height must be a positive integer",
        "base grid height must be a positive integer",
    ),
    (
        "height float",
        lambda doc, section: doc[section].__setitem__("height", 2.5),
        "grid height must be a positive integer",
        "base grid height must be a positive integer",
    ),
    (
        "obstacles missing",
        lambda doc, section: doc[section].pop("obstacles"),
        "grid 'obstacles' must be a list",
        "base grid 'obstacles' must be a list",
    ),
    (
        "obstacles null",
        lambda doc, section: doc[section].__setitem__("obstacles", None),
        "grid 'obstacles' must be a list",
        "base grid 'obstacles' must be a list",
    ),
    (
        "obstacles object",
        lambda doc, section: doc[section].__setitem__("obstacles", {}),
        "grid 'obstacles' must be a list",
        "base grid 'obstacles' must be a list",
    ),
    (
        "obstacles string",
        lambda doc, section: doc[section].__setitem__("obstacles", "[]"),
        "grid 'obstacles' must be a list",
        "base grid 'obstacles' must be a list",
    ),
    (
        "entry is an integer",
        lambda doc, section: doc[section]["obstacles"].append(3),
        "obstacle entry 0 must be a [x, y] integer pair",
        "base obstacle entry 0 must be a [x, y] integer pair",
    ),
    (
        "entry is an object",
        lambda doc, section: doc[section]["obstacles"].append({"x": 1, "y": 0}),
        "obstacle entry 0 must be a [x, y] integer pair",
        "base obstacle entry 0 must be a [x, y] integer pair",
    ),
    (
        "entry has one element",
        lambda doc, section: doc[section]["obstacles"].append([1]),
        "obstacle entry 0 must be a [x, y] integer pair",
        "base obstacle entry 0 must be a [x, y] integer pair",
    ),
    (
        "entry has three elements",
        lambda doc, section: doc[section]["obstacles"].append([1, 0, 0]),
        "obstacle entry 0 must be a [x, y] integer pair",
        "base obstacle entry 0 must be a [x, y] integer pair",
    ),
    (
        "entry coordinate float",
        lambda doc, section: doc[section]["obstacles"].append([1.0, 0]),
        "obstacle entry 0 must be a [x, y] integer pair",
        "base obstacle entry 0 must be a [x, y] integer pair",
    ),
    (
        "entry coordinate string",
        lambda doc, section: doc[section]["obstacles"].append(["1", 0]),
        "obstacle entry 0 must be a [x, y] integer pair",
        "base obstacle entry 0 must be a [x, y] integer pair",
    ),
    (
        "entry coordinate null",
        lambda doc, section: doc[section]["obstacles"].append([None, 0]),
        "obstacle entry 0 must be a [x, y] integer pair",
        "base obstacle entry 0 must be a [x, y] integer pair",
    ),
    (
        "entry coordinate boolean x",
        lambda doc, section: doc[section]["obstacles"].append([True, 0]),
        "obstacle entry 0 must be a [x, y] integer pair",
        "base obstacle entry 0 must be a [x, y] integer pair",
    ),
    (
        "entry coordinate boolean y",
        lambda doc, section: doc[section]["obstacles"].append([0, False]),
        "obstacle entry 0 must be a [x, y] integer pair",
        "base obstacle entry 0 must be a [x, y] integer pair",
    ),
    (
        "entry out of bounds on x (high)",
        lambda doc, section: doc[section]["obstacles"].append([5, 0]),
        "obstacle [5, 0] lies outside the map",
        "base obstacle [5, 0] lies outside the map",
    ),
    (
        "entry out of bounds on x (negative)",
        lambda doc, section: doc[section]["obstacles"].append([-1, 0]),
        "obstacle [-1, 0] lies outside the map",
        "base obstacle [-1, 0] lies outside the map",
    ),
    (
        "entry out of bounds on y",
        lambda doc, section: doc[section]["obstacles"].append([0, 2]),
        "obstacle [0, 2] lies outside the map",
        "base obstacle [0, 2] lies outside the map",
    ),
]


class MapDefinitionFormatRejectionTests(unittest.TestCase):
    """Both definitions obey the same shape and range rules.

    Messages keep their existing wording (``grid`` for the current map,
    ``base grid``/``base obstacle`` for the initial map), so a user can tell
    which definition is wrong, and obstacle items keep their positional index.
    """

    def test_every_rule_is_enforced_for_both_definitions(self) -> None:
        for section, base_case in (("grid", False), ("base_grid", True)):
            for label, mutate, grid_fragment, base_fragment in FORMAT_MUTATIONS:
                fragment = base_fragment if base_case else grid_fragment
                with self.subTest(section=section, case=label):
                    document = _plain_document()
                    mutate(document, section)
                    _expect_rejected(self, document, fragment)

    def test_obstacle_entry_index_is_reported_for_both_definitions(self) -> None:
        # Two well-formed entries precede the bad one; its index must survive.
        document = _plain_document()
        document["grid"]["obstacles"] = [[0, 0], [1, 0], [2]]
        _expect_rejected(
            self, document, "obstacle entry 2 must be a [x, y] integer pair"
        )

        document = _plain_document()
        document["base_grid"]["obstacles"] = [[0, 0], [1, 0], [2]]
        _expect_rejected(
            self, document, "base obstacle entry 2 must be a [x, y] integer pair"
        )

    def test_bad_maps_are_rejected_through_the_public_file_entry(self) -> None:
        # Sanity pin on the exception contract: the public file-based entry
        # raises ValueError (never KeyError/TypeError from parsing the shape).
        document = _plain_document()
        document["grid"]["obstacles"] = [[0, 0], [1]]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(document, handle)
            with self.assertRaises(ValueError):
                FleetSimulator.load_checkpoint(path)

    def test_no_silent_coercion_or_truncation_of_bad_dimensions(self) -> None:
        # A float-looking width must not be rounded into a legal map: the
        # rejection is the format error, not a downstream consistency error.
        for section, fragment in (
            ("grid", "grid width"),
            ("base_grid", "base grid width"),
        ):
            with self.subTest(section=section):
                document = _plain_document()
                document[section]["width"] = 4.9
                message = _expect_rejected(self, document, fragment)
                self.assertNotIn("dimensions must match", message)
                self.assertNotIn("does not match the saved grid", message)


class MapDefinitionConsistencyRejectionTests(unittest.TestCase):
    """Two individually legal maps can still disagree with each other."""

    def test_legal_maps_with_mismatched_dimensions_are_rejected(self) -> None:
        document = _plain_document()
        document["base_grid"]["width"] = 6
        document["base_grid"]["obstacles"] = []
        # Both definitions parse cleanly on their own; only their combination
        # is wrong, and the report says so rather than blaming a shape.
        message = _expect_rejected(self, document, "dimensions must match")
        self.assertNotIn("positive integer", message)
        self.assertNotIn("obstacle entry", message)

    def test_matching_dimensions_do_not_make_the_archive_valid(self) -> None:
        # Same width/height on both maps, but the current grid carries an
        # obstacle that no recorded edit ever produced from the initial map.
        document = _plain_document()
        document["grid"]["obstacles"] = [[2, 0]]
        _expect_rejected(self, document, "does not match the saved grid")

    def test_history_end_state_not_reproducing_grid_is_rejected(self) -> None:
        # The history opens a cell the current map still shows closed: both
        # maps are legal and equally sized, but replaying the edits ends at a
        # different obstacle set than the saved current grid.
        document = _plain_document()
        document["base_grid"]["obstacles"] = [[1, 0]]
        document["grid"]["obstacles"] = [[1, 0]]
        change = {
            "type": "map_change",
            "tick": 0,
            "sequence": 1,
            "added": [],
            "removed": [[1, 0]],
        }
        document["map_changes"] = [
            {key: value for key, value in change.items() if key != "type"}
        ]
        document["replay"] = [copy.deepcopy(change)]
        _expect_rejected(self, document, "does not match the saved grid")

    def test_history_removing_an_absent_obstacle_is_rejected(self) -> None:
        document = _plain_document()
        change = {
            "type": "map_change",
            "tick": 0,
            "sequence": 1,
            "added": [],
            "removed": [[1, 0]],
        }
        document["map_changes"] = [
            {key: value for key, value in change.items() if key != "type"}
        ]
        document["replay"] = [copy.deepcopy(change)]
        _expect_rejected(
            self, document, "removes obstacle [1, 0] that is absent according to the history"
        )

    def test_history_adding_a_present_obstacle_is_rejected(self) -> None:
        document = _plain_document()
        for section in ("grid", "base_grid"):
            document[section]["obstacles"] = [[1, 0]]
        change = {
            "type": "map_change",
            "tick": 0,
            "sequence": 1,
            "added": [[1, 0]],
            "removed": [],
        }
        document["map_changes"] = [
            {key: value for key, value in change.items() if key != "type"}
        ]
        document["replay"] = [copy.deepcopy(change)]
        _expect_rejected(
            self, document, "adds obstacle [1, 0] that is already present according to the history"
        )

    def test_resizing_cannot_force_a_broken_history_to_hold(self) -> None:
        # The initial map is given room for the out-of-grid edit cell, but its
        # dimensions then disagree with the current map: the dimension rule is
        # checked before history is replayed and cannot be bypassed by editing
        # sizes to make the recorded changes in-bounds.
        document = _plain_document()
        document["base_grid"]["width"] = 6
        change = {
            "type": "map_change",
            "tick": 0,
            "sequence": 1,
            "added": [[5, 0]],
            "removed": [],
        }
        document["map_changes"] = [
            {key: value for key, value in change.items() if key != "type"}
        ]
        document["replay"] = [copy.deepcopy(change)]
        _expect_rejected(self, document, "dimensions must match")


if __name__ == "__main__":
    unittest.main()
