"""Map-definition checks for restored checkpoints.

A version 2 checkpoint carries two map definitions: ``grid`` (the map in use
when the file was saved) and ``base_grid`` (the map before any runtime
obstacle edit). ``load_checkpoint`` must restore each definition with its own
dimensions and obstacles, apply the same value rules to both, and keep the
existing cross-definition constraints: matching dimensions and an obstacle
history that replays from ``base_grid`` onto ``grid``. Malformed definitions
raise :class:`ValueError` whose message says which map is at fault; two
individually legal maps that disagree raise the dedicated consistency errors.
"""

import copy
import json
import os
import tempfile
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot


def load(document):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return FleetSimulator.load_checkpoint(path)


def expect_rejected(testcase, document, fragment):
    with testcase.assertRaises(ValueError) as context:
        load(document)
    testcase.assertIn(fragment, str(context.exception))
    return str(context.exception)


def round_trip(simulator):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        simulator.save_checkpoint(path)
        return FleetSimulator.load_checkpoint(path)


class MapDocumentBuilder:
    """A real zero-tick v2 checkpoint whose two map definitions can be edited.

    The starting document has no obstacles, no map changes and an empty
    replay, so any well-formed grid/base_grid pair with matching dimensions
    and a reproducible history loads; everything not under test stays valid.
    The single robot stands at (0, 0), clear of every cell the tests block.
    """

    def __init__(self, width=4, height=3):
        simulator = FleetSimulator(GridMap(width, height), [Robot("A", (0, 0))], [])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            simulator.save_checkpoint(path)
            with open(path, encoding="utf-8") as handle:
                self.document = json.load(handle)

    def version_one(self):
        document = self.document
        document["version"] = 1
        for key in ("base_grid", "map_changes", "paused_tasks", "traffic_waits"):
            document.pop(key, None)
        document["replay"] = [
            {key: value for key, value in frame.items() if key != "type"}
            for frame in document["replay"]
        ]
        return self


def dimension_fragment(map_key, dimension):
    prefix = "checkpoint grid" if map_key == "grid" else "checkpoint base grid"
    return f"{prefix} {dimension} must be a positive integer"


def obstacle_entry_fragment(map_key, index):
    prefix = "obstacle entry" if map_key == "grid" else "base obstacle entry"
    return f"{prefix} {index} must be a [x, y] integer pair"


def obstacle_bounds_fragment(map_key, cell):
    prefix = "obstacle" if map_key == "grid" else "base obstacle"
    return f"{prefix} {list(cell)} lies outside the map"


class RestoredMapTests(unittest.TestCase):
    """Legal documents restore both maps with their own contents."""

    def test_reopened_cell_is_traversable_now_but_blocked_in_base(self):
        # A cell that was closed at start and reopened later: the current map
        # must be drivable there while the base map keeps the old obstacle,
        # and the recorded edit history is accepted as is.
        simulator = FleetSimulator(
            GridMap(3, 2, frozenset({(1, 0)})), [Robot("R", (0, 0))], []
        )
        simulator.step()
        simulator.modify_obstacles(removed=[(1, 0)])
        loaded = round_trip(simulator)
        self.assertTrue(loaded.grid.traversable((1, 0)))
        self.assertEqual(loaded.grid.obstacles, frozenset())
        self.assertEqual(loaded.base_grid.obstacles, frozenset({(1, 0)}))
        self.assertFalse(loaded.base_grid.traversable((1, 0)))
        self.assertEqual(loaded.tick, 1)
        self.assertEqual(
            loaded.map_change_history(),
            [
                {
                    "type": "map_change",
                    "tick": 1,
                    "sequence": 1,
                    "added": [],
                    "removed": [[1, 0]],
                }
            ],
        )

    def test_newly_closed_cell_is_blocked_now_but_open_in_base(self):
        simulator = FleetSimulator(GridMap(3, 2), [Robot("R", (0, 0))], [])
        simulator.modify_obstacles(added=[(2, 1)])
        loaded = round_trip(simulator)
        self.assertEqual(loaded.grid.obstacles, frozenset({(2, 1)}))
        self.assertEqual(loaded.base_grid.obstacles, frozenset())
        self.assertTrue(loaded.base_grid.traversable((2, 1)))

    def test_each_map_keeps_its_own_dimensions(self):
        # Both definitions are read from the file, not copied from one
        # another: rewriting both consistently changes the restored map.
        builder = MapDocumentBuilder()
        for key in ("grid", "base_grid"):
            builder.document[key]["width"] = 6
            builder.document[key]["height"] = 2
        loaded = load(builder.document)
        self.assertEqual((loaded.grid.width, loaded.grid.height), (6, 2))
        self.assertEqual((loaded.base_grid.width, loaded.base_grid.height), (6, 2))

    def test_empty_obstacle_lists_are_legal(self):
        loaded = load(MapDocumentBuilder().document)
        self.assertEqual(loaded.grid.obstacles, frozenset())
        self.assertEqual(loaded.base_grid.obstacles, frozenset())

    def test_duplicate_obstacle_coordinate_counts_once(self):
        # A coordinate repeated within one obstacle list denotes a single
        # obstacle; the file is not rejected and other obstacles are kept.
        for key in ("grid", "base_grid"):
            with self.subTest(map=key):
                builder = MapDocumentBuilder()
                builder.document[key]["obstacles"] = [[1, 1], [1, 1], [2, 1]]
                other = "base_grid" if key == "grid" else "grid"
                builder.document[other]["obstacles"] = [[1, 1], [2, 1]]
                loaded = load(builder.document)
                self.assertEqual(
                    loaded.grid.obstacles, frozenset({(1, 1), (2, 1)})
                )
                self.assertEqual(
                    loaded.base_grid.obstacles, frozenset({(1, 1), (2, 1)})
                )

    def test_version_one_grid_becomes_the_base_grid(self):
        # Version 1 files have no base_grid field: the saved grid is taken as
        # the baseline without demanding v2 fields from the old document.
        builder = MapDocumentBuilder()
        builder.document["grid"]["obstacles"] = [[1, 1]]
        loaded = load(builder.version_one().document)
        self.assertEqual(loaded.grid.obstacles, frozenset({(1, 1)}))
        self.assertEqual(loaded.base_grid.obstacles, frozenset({(1, 1)}))
        self.assertEqual(
            (loaded.base_grid.width, loaded.base_grid.height),
            (loaded.grid.width, loaded.grid.height),
        )
        self.assertEqual(loaded.map_change_history(), [])


class MapValueRuleTests(unittest.TestCase):
    """Both map definitions obey the same shape and value rules."""

    def setUp(self):
        self.prototype = MapDocumentBuilder().document

    def document(self):
        # Every subTest starts from an untouched valid checkpoint, so a
        # mutation under test never leaks into the next case.
        return copy.deepcopy(self.prototype)

    def test_missing_map_definition_rejected(self):
        for key in ("grid", "base_grid"):
            with self.subTest(map=key):
                document = self.document()
                del document[key]
                expect_rejected(
                    self, document, f"checkpoint is missing required field {key!r}"
                )

    def test_map_definition_must_be_an_object(self):
        for key in ("grid", "base_grid"):
            for bad in ([4, 3], "map", 7):
                with self.subTest(map=key, value=bad):
                    document = self.document()
                    document[key] = bad
                    expect_rejected(
                        self, document, f"checkpoint field {key!r} must be an object"
                    )

    def test_width_must_be_a_positive_integer(self):
        for key in ("grid", "base_grid"):
            for bad in (0, -2, 1.5, "4", True, False, None):
                with self.subTest(map=key, value=bad):
                    document = self.document()
                    document[key]["width"] = bad
                    expect_rejected(self, document, dimension_fragment(key, "width"))

    def test_height_must_be_a_positive_integer(self):
        for key in ("grid", "base_grid"):
            for bad in (0, -1, 2.5, "3", True, False, None):
                with self.subTest(map=key, value=bad):
                    document = self.document()
                    document[key]["height"] = bad
                    expect_rejected(self, document, dimension_fragment(key, "height"))

    def test_missing_dimension_rejected(self):
        for key in ("grid", "base_grid"):
            for dimension in ("width", "height"):
                with self.subTest(map=key, dimension=dimension):
                    document = self.document()
                    del document[key][dimension]
                    expect_rejected(
                        self, document, dimension_fragment(key, dimension)
                    )

    def test_obstacles_must_be_a_list(self):
        fragment = {
            "grid": "checkpoint grid 'obstacles' must be a list",
            "base_grid": "checkpoint base grid 'obstacles' must be a list",
        }
        for key in ("grid", "base_grid"):
            for bad in ({"x": 1}, "1,1", 3, None, True):
                with self.subTest(map=key, value=bad):
                    document = self.document()
                    document[key]["obstacles"] = bad
                    expect_rejected(self, document, fragment[key])
            with self.subTest(map=key, value="missing"):
                document = self.document()
                del document[key]["obstacles"]
                expect_rejected(self, document, fragment[key])

    def test_obstacle_entry_must_be_an_integer_pair(self):
        bad_entries = (
            [1],
            [1, 2, 3],
            [],
            "1,2",
            5,
            None,
            {"x": 1, "y": 2},
            [1, "2"],
            ["1", 2],
            [1.5, 2],
            [1, 2.0],
            [1, True],
            [False, 1],
            [[1], 2],
        )
        for key in ("grid", "base_grid"):
            for bad in bad_entries:
                with self.subTest(map=key, value=bad):
                    document = self.document()
                    document[key]["obstacles"] = [bad]
                    expect_rejected(
                        self, document, obstacle_entry_fragment(key, 0)
                    )

    def test_obstacle_entry_index_is_reported(self):
        for key in ("grid", "base_grid"):
            with self.subTest(map=key):
                document = self.document()
                document[key]["obstacles"] = [[1, 1], [2, "x"]]
                expect_rejected(self, document, obstacle_entry_fragment(key, 1))

    def test_obstacle_must_lie_inside_its_map(self):
        for key in ("grid", "base_grid"):
            for cell in ((-1, 0), (0, -1), (4, 0), (0, 3), (9, 9)):
                with self.subTest(map=key, cell=cell):
                    document = self.document()
                    document[key]["obstacles"] = [list(cell)]
                    expect_rejected(
                        self, document, obstacle_bounds_fragment(key, cell)
                    )


class CrossMapConsistencyTests(unittest.TestCase):
    """Two individually legal maps must still agree with each other."""

    def setUp(self):
        self.prototype = MapDocumentBuilder().document

    def document(self):
        return copy.deepcopy(self.prototype)

    def test_mismatched_widths_rejected(self):
        document = self.document()
        document["base_grid"]["width"] = 5
        expect_rejected(
            self, document, "base grid dimensions must match the current grid"
        )

    def test_mismatched_heights_rejected(self):
        document = self.document()
        document["base_grid"]["height"] = 4
        expect_rejected(
            self, document, "base grid dimensions must match the current grid"
        )

    def test_matching_dimensions_do_not_excuse_a_broken_history(self):
        # Same dimensions, both maps well formed, but no recorded change
        # explains how the base obstacle disappeared from the current map.
        document = self.document()
        document["base_grid"]["obstacles"] = [[1, 1]]
        expect_rejected(
            self,
            document,
            "the map reproduced from 'map_changes' does not match the saved grid",
        )

    def test_current_obstacle_without_a_recorded_edit_rejected(self):
        # The mirror image: the current map gained an obstacle the history
        # never added.
        document = self.document()
        document["grid"]["obstacles"] = [[1, 1]]
        expect_rejected(
            self,
            document,
            "the map reproduced from 'map_changes' does not match the saved grid",
        )

    def test_format_error_is_not_reported_as_a_consistency_error(self):
        # A malformed base grid must surface as a base-grid format error,
        # not as a dimension mismatch or a history mismatch.
        document = self.document()
        document["base_grid"]["width"] = True
        message = expect_rejected(
            self, document, "checkpoint base grid width must be a positive integer"
        )
        self.assertNotIn("dimensions must match", message)
        self.assertNotIn("does not match the saved grid", message)

    def test_consistency_error_is_not_reported_as_a_format_error(self):
        # Both maps are well formed here; only their agreement fails, so the
        # message must not blame either definition's shape.
        document = self.document()
        document["base_grid"]["width"] = 5
        message = expect_rejected(
            self, document, "base grid dimensions must match the current grid"
        )
        self.assertNotIn("must be a positive integer", message)
        self.assertNotIn("must be a [x, y] integer pair", message)


if __name__ == "__main__":
    unittest.main()
