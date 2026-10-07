"""Strict input validation for maps created directly through ``GridMap``.

A map built through the public Python interface must already satisfy every map
rule checkpoint loading enforces: width and height are positive plain integers
(booleans are not integers and floats are never coerced), and every obstacle is
a tuple of exactly two plain integers lying inside the map. Bad inputs fail at
construction with a :class:`ValueError` that names the dimension or the
offending obstacle -- never with a raw unpacking/hashing :class:`TypeError` --
so a map that could not be reloaded after saving can never be created in the
first place. Rejection is all-or-nothing and never mutates the caller's
collection, while legal set/frozenset/omitted inputs and the creation-time
obstacle snapshot keep their existing semantics.
"""

import os
import tempfile
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot
from warehouse_fleet.pathfinding import shortest_path


class DimensionValidationTests(unittest.TestCase):
    def assert_value_error_naming(self, kwargs, fragment):
        with self.assertRaises(ValueError) as context:
            GridMap(**kwargs)
        self.assertIn(fragment, str(context.exception))
        self.assertNotIn("TypeError", str(context.exception))

    def test_width_must_be_a_positive_plain_integer(self):
        for bad in (0, -1, -3, 3.0, 1.5, True, False, "3", None):
            with self.subTest(bad=bad):
                self.assert_value_error_naming(
                    {"width": bad, "height": 2}, "width"
                )

    def test_height_must_be_a_positive_plain_integer(self):
        for bad in (0, -2, 2.0, 4.7, True, False, "2", None):
            with self.subTest(bad=bad):
                self.assert_value_error_naming(
                    {"width": 2, "height": bad}, "height"
                )

    def test_float_equal_to_an_integer_is_not_accepted(self):
        with self.assertRaises(ValueError):
            GridMap(3.0, 2)
        with self.assertRaises(ValueError):
            GridMap(3, 2.0)

    def test_boolean_is_not_an_integer_even_though_it_compares_like_one(self):
        with self.assertRaises(ValueError):
            GridMap(True, 2)
        with self.assertRaises(ValueError):
            GridMap(3, False)

    def test_positive_integer_dimensions_still_load(self):
        grid = GridMap(1, 1)
        self.assertEqual((grid.width, grid.height), (1, 1))
        grid = GridMap(7, 5, frozenset({(6, 4)}))
        self.assertEqual((grid.width, grid.height), (7, 5))


class ObstacleShapeValidationTests(unittest.TestCase):
    def _reject(self, obstacles):
        with self.assertRaises(ValueError) as context:
            GridMap(4, 3, obstacles)
        return str(context.exception)

    def test_boolean_and_float_components_are_rejected(self):
        for bad in (True, False, 1.0, 0.0):
            with self.subTest(bad=bad):
                message = self._reject({(bad, 0)})
                self.assertIn("obstacle", message)
                message = self._reject({(0, bad)})
                self.assertIn("obstacle", message)

    def test_non_integer_components_are_rejected(self):
        for obstacles in (
            {("1", 0)},
            {(0, "0")},
            {(None, 0)},
            {(0, None)},
            {(1.5, 0)},
        ):
            with self.subTest(obstacles=obstacles):
                self.assertIn("obstacle", self._reject(obstacles))

    def test_wrong_arity_pairs_are_rejected(self):
        for obstacles in ({(1,)}, {(1, 2, 3)}, {()}, frozenset({(1,)})):
            with self.subTest(obstacles=obstacles):
                self.assertIn("obstacle", self._reject(obstacles))

    def test_scalar_string_null_and_number_entries_are_rejected_as_obstacles(self):
        for obstacles in ({"x"}, {None}, {1}, frozenset({1}), ["x"], [None], [2]):
            with self.subTest(obstacles=obstacles):
                message = self._reject(obstacles)
                # The message must identify the offending input as an obstacle.
                self.assertIn("obstacle", message)

    def test_list_cell_is_rejected_with_value_error_not_type_error(self):
        # A list used as one obstacle used to die inside frozenset()/unpacking
        # with a TypeError; it must now be reported as a bad obstacle.
        message = self._reject([[1, 0]])
        self.assertIn("obstacle", message)

    def test_non_iterable_obstacle_argument_is_rejected(self):
        for obstacles in ("x", None, 42, True):
            with self.subTest(obstacles=obstacles):
                self.assertIn("obstacles", self._reject(obstacles))

    def test_no_type_error_escapes_for_any_bad_shape(self):
        bad_inputs = [
            {(1.0, 0)},
            {(True, 0)},
            {("1", 0)},
            {(1, None)},
            {(1,)},
            {(1, 2, 3)},
            {"x"},
            {1},
            {None},
            [[1, 0]],
            [(1, 0), "x"],
            [(1, 0), (1, 2, 3)],
            "x",
            None,
            42,
        ]
        for obstacles in bad_inputs:
            with self.subTest(obstacles=obstacles):
                try:
                    GridMap(4, 3, obstacles)
                except ValueError:
                    pass
                else:
                    self.fail(f"{obstacles!r} was accepted")

    def test_index_is_reported_for_an_ordered_input(self):
        message = self._reject([(0, 0), (1, 0), "bad"])
        self.assertIn("obstacle 2", message)


class ObstacleBoundsValidationTests(unittest.TestCase):
    def test_boundary_cells_are_legal(self):
        grid = GridMap(3, 2, frozenset({(0, 0), (2, 0), (0, 1), (2, 1)}))
        self.assertEqual(
            grid.obstacles, frozenset({(0, 0), (2, 0), (0, 1), (2, 1)})
        )

    def test_out_of_bounds_cells_are_rejected_not_truncated_or_ignored(self):
        for cell in ((3, 0), (0, 2), (-1, 0), (0, -1), (99, 1)):
            with self.subTest(cell=cell):
                with self.assertRaises(ValueError) as context:
                    GridMap(3, 2, frozenset({cell}))
                message = str(context.exception)
                self.assertIn("outside the", message)
                self.assertIn(str(cell[0]), message)

    def test_one_bad_cell_rejects_the_whole_map_with_legal_cells(self):
        # No partial map keeping only the cell checked first may survive.
        with self.assertRaises(ValueError):
            GridMap(3, 2, {(0, 0), (9, 9)})
        with self.assertRaises(ValueError):
            GridMap(3, 2, [(0, 0), (1, 0), (3, 0)])


class LegalInputTests(unittest.TestCase):
    def test_set_frozenset_omitted_and_empty_inputs(self):
        cells = {(1, 0), (0, 1)}
        self.assertEqual(GridMap(3, 2, set(cells)).obstacles, frozenset(cells))
        self.assertEqual(
            GridMap(3, 2, frozenset(cells)).obstacles, frozenset(cells)
        )
        self.assertEqual(GridMap(3, 2, set()).obstacles, frozenset())
        self.assertEqual(GridMap(3, 2, frozenset()).obstacles, frozenset())
        self.assertEqual(GridMap(3, 2).obstacles, frozenset())

    def test_obstacles_is_always_a_frozenset(self):
        for obstacles in (set(), frozenset(), {(1, 0)}, [(1, 0)], ()):
            with self.subTest(obstacles=obstacles):
                self.assertIsInstance(GridMap(3, 2, obstacles).obstacles, frozenset)

    def test_duplicate_legal_cells_count_once(self):
        self.assertEqual(
            GridMap(3, 2, [(1, 0), (1, 0), (0, 1)]).obstacles,
            frozenset({(1, 0), (0, 1)}),
        )
        self.assertEqual(
            GridMap(3, 2, ((2, 1), (2, 1))).obstacles,
            frozenset({(2, 1)}),
        )


class SnapshotAndCallerSafetyTests(unittest.TestCase):
    def test_later_caller_mutations_never_reach_the_map(self):
        cells = {(1, 0)}
        grid = GridMap(5, 2, cells)
        cells.add((2, 0))
        cells.discard((1, 0))
        cells.clear()
        cells.add((3, 1))
        self.assertEqual(grid.obstacles, frozenset({(1, 0)}))
        self.assertTrue(grid.traversable((2, 0)))
        self.assertTrue(grid.traversable((3, 1)))

    def test_successful_construction_does_not_mutate_the_callers_collection(self):
        cells = [(1, 0), (1, 0), (0, 1)]
        GridMap(5, 2, cells)
        self.assertEqual(cells, [(1, 0), (1, 0), (0, 1)])

        cells_set = {(1, 0)}
        GridMap(5, 2, cells_set)
        self.assertEqual(cells_set, {(1, 0)})

    def test_failed_construction_leaves_the_callers_collection_untouched(self):
        cells = {(0, 0), (9, 9)}
        before = set(cells)
        with self.assertRaises(ValueError):
            GridMap(2, 2, cells)
        self.assertEqual(cells, before)

        ordered = [(0, 0), (1.0, 0)]
        ordered_before = list(ordered)
        with self.assertRaises(ValueError):
            GridMap(2, 2, ordered)
        self.assertEqual(ordered, ordered_before)

        mixed = {(0, 0), "bad"}
        mixed_before = set(mixed)
        with self.assertRaises(ValueError):
            GridMap(2, 2, mixed)
        self.assertEqual(mixed, mixed_before)

    def test_failed_construction_never_returns_a_partial_map(self):
        with self.assertRaises(ValueError):
            GridMap(0, 2, {(0, 0)})
        with self.assertRaises(ValueError):
            GridMap(2, 2, {(0, 0), (1.0, 0)})


class ValidatedMapBehaviorTests(unittest.TestCase):
    """Legal integer maps behave exactly as before validation was added."""

    def test_traversability_and_pathfinding_keep_their_semantics(self):
        grid = GridMap(5, 1, frozenset({(2, 0)}))
        self.assertTrue(grid.traversable((0, 0)))
        self.assertFalse(grid.traversable((2, 0)))
        self.assertFalse(grid.contains((5, 0)))
        with self.assertRaises(ValueError):
            shortest_path(grid, (0, 0), (4, 0))
        open_grid = GridMap(5, 1)
        self.assertEqual(
            shortest_path(open_grid, (0, 0), (4, 0)),
            [(1, 0), (2, 0), (3, 0), (4, 0)],
        )

    def test_formal_obstacle_edits_keep_working(self):
        simulator = FleetSimulator(
            GridMap(4, 2, frozenset({(3, 0)})),
            [Robot("R", (0, 0))],
            [],
        )
        simulator.modify_obstacles(added=[(1, 0)], removed=[(3, 0)])
        self.assertEqual(simulator.grid.obstacles, frozenset({(1, 0)}))
        self.assertEqual(simulator.base_grid.obstacles, frozenset({(3, 0)}))

    def test_save_and_load_preserve_dimensions_and_obstacles(self):
        grid = GridMap(4, 2, frozenset({(3, 0), (0, 1)}))
        simulator = FleetSimulator(grid, [Robot("R", (1, 1))], [])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.json")
            simulator.save_checkpoint(path)
            loaded = FleetSimulator.load_checkpoint(path)
        self.assertEqual((loaded.grid.width, loaded.grid.height), (4, 2))
        self.assertEqual(loaded.grid.obstacles, frozenset({(3, 0), (0, 1)}))
        self.assertEqual(
            (loaded.base_grid.width, loaded.base_grid.height), (4, 2)
        )
        self.assertEqual(
            loaded.base_grid.obstacles, frozenset({(3, 0), (0, 1)})
        )


if __name__ == "__main__":
    unittest.main()
