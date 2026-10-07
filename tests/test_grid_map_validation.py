"""Input validation for maps created directly through the public API.

A grid built through ``GridMap`` must obey the same map rules a checkpoint is
checked against: width and height are plain positive integers (booleans are
not integers and whole-valued floats are never coerced), and every obstacle
is an in-bounds pair of two plain integers. Bad input is rejected with a
``ValueError`` at construction -- naming the dimension or obstacle at fault,
never leaking a bare ``TypeError`` -- so a map that could never be reloaded
cannot be created, saved and only then discovered unrecoverable.
"""

import unittest

from warehouse_fleet import GridMap
from warehouse_fleet.pathfinding import shortest_path


class DimensionValidationTests(unittest.TestCase):
    def reject(self, **kwargs):
        with self.assertRaises(ValueError) as ctx:
            GridMap(**kwargs)
        return str(ctx.exception)

    def test_floats_are_rejected_even_when_whole_valued(self) -> None:
        message = self.reject(width=3.0, height=3)
        self.assertIn("width", message)
        message = self.reject(width=3, height=2.0)
        self.assertIn("height", message)
        self.reject(width=4.0, height=4.0)

    def test_booleans_are_not_integers(self) -> None:
        message = self.reject(width=True, height=3)
        self.assertIn("width", message)
        message = self.reject(width=3, height=False)
        self.assertIn("height", message)

    def test_zero_negative_and_non_numbers_are_rejected(self) -> None:
        for bad_width in (0, -1, -10, "3", None, [3], 2.5):
            message = self.reject(width=bad_width, height=3)
            self.assertIn("width", message, msg=f"width={bad_width!r}")
        for bad_height in (0, -1, -10, "2", None, [2], 0.5):
            message = self.reject(width=3, height=bad_height)
            self.assertIn("height", message, msg=f"height={bad_height!r}")

    def test_message_distinguishes_width_from_height(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            GridMap(-1, 5)
        self.assertIn("width", str(ctx.exception))
        self.assertNotIn("height", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            GridMap(5, -1)
        self.assertIn("height", str(ctx.exception))
        self.assertNotIn("width", str(ctx.exception))


class ObstacleValidationTests(unittest.TestCase):
    def reject(self, obstacles):
        with self.assertRaises(ValueError) as ctx:
            GridMap(4, 4, obstacles)
        return str(ctx.exception)

    def test_float_components_are_not_coerced(self) -> None:
        for cell in ((1.0, 0), (0, 2.0), (1.0, 1.0)):
            message = self.reject({cell})
            self.assertIn("obstacle", message)

    def test_boolean_components_are_rejected(self) -> None:
        for cell in ((True, 0), (0, False), (True, True)):
            message = self.reject({cell})
            self.assertIn("obstacle", message)

    def test_string_and_none_components_are_rejected(self) -> None:
        for cell in (("1", 0), (0, "0"), (None, 0), (0, None)):
            self.reject(frozenset({cell}))

    def test_wrong_arity_entries_are_rejected(self) -> None:
        for entry in ((1,), (1, 0, 1), ()):
            message = self.reject({entry})
            self.assertIn("obstacle", message)

    def test_non_pair_entries_are_rejected_with_value_error(self) -> None:
        # A bare string, None or single number names an obstacle entry; the
        # caller must never see a raw TypeError from unpacking or hashing.
        for entry in ("x", None, 1, 1.0, True):
            message = self.reject({entry})
            self.assertIn("obstacle", message)

    def test_non_collection_obstacle_argument_is_value_error(self) -> None:
        for argument in (None, 1, "(1, 0)", 1.5):
            with self.assertRaises(ValueError):
                GridMap(4, 4, argument)

    def test_message_identifies_the_offending_obstacle(self) -> None:
        message = self.reject({(0, 0), (9, 9)})
        self.assertIn("(9, 9)", message)
        message = self.reject(frozenset({(1, 0), ("bad", 0)}))
        self.assertIn("bad", message)

    def test_boundary_coordinates_are_exact(self) -> None:
        # x in [0, width-1], y in [0, height-1]: corners are legal, one step
        # beyond is rejected, never truncated or silently ignored.
        self.assertEqual(GridMap(3, 2, {(2, 1), (0, 0)}).obstacles,
                         frozenset({(2, 1), (0, 0)}))
        for cell in ((3, 0), (0, 2), (-1, 0), (0, -1)):
            with self.assertRaises(ValueError):
                GridMap(3, 2, {cell})

    def test_invalid_set_creates_no_partial_map_and_leaves_input_intact(self) -> None:
        cells = {(0, 0), (9, 9)}
        with self.assertRaises(ValueError):
            GridMap(2, 2, cells)
        # The caller's collection is untouched by the failed attempt.
        self.assertEqual(cells, {(0, 0), (9, 9)})


class LegalObstacleInputTests(unittest.TestCase):
    def test_set_frozenset_omitted_and_empty(self) -> None:
        self.assertEqual(GridMap(3, 2, {(1, 0)}).obstacles, frozenset({(1, 0)}))
        self.assertEqual(
            GridMap(3, 2, frozenset({(1, 0)})).obstacles, frozenset({(1, 0)})
        )
        self.assertEqual(GridMap(3, 2).obstacles, frozenset())
        self.assertEqual(GridMap(3, 2, set()).obstacles, frozenset())
        self.assertEqual(GridMap(3, 2, frozenset()).obstacles, frozenset())

    def test_duplicate_legal_cells_count_once(self) -> None:
        grid = GridMap(3, 2, frozenset({(1, 0), (1, 0), (2, 1)}))
        self.assertEqual(grid.obstacles, frozenset({(1, 0), (2, 1)}))

    def test_obstacles_attribute_is_a_frozenset(self) -> None:
        grid = GridMap(3, 2, {(1, 0)})
        self.assertIsInstance(grid.obstacles, frozenset)

    def test_caller_collection_is_snapshotted(self) -> None:
        cells = {(1, 0)}
        grid = GridMap(5, 2, cells)
        cells.add((2, 0))
        cells.discard((1, 0))
        self.assertEqual(grid.obstacles, frozenset({(1, 0)}))
        self.assertTrue(grid.traversable((2, 0)))

    def test_construction_never_mutates_the_caller_collection(self) -> None:
        cells = {(1, 0)}
        GridMap(5, 2, cells)
        self.assertEqual(cells, {(1, 0)})
        with self.assertRaises(ValueError):
            GridMap(2, 2, cells | {(5, 5)})
        self.assertEqual(cells, {(1, 0)})

    def test_validated_map_keeps_simulation_and_pathfinding_consistent(self) -> None:
        grid = GridMap(5, 1, frozenset({(2, 0)}))
        self.assertFalse(grid.traversable((2, 0)))
        self.assertTrue(grid.traversable((1, 0)))
        with self.assertRaises(ValueError):
            shortest_path(grid, (0, 0), (4, 0))
        open_grid = GridMap(5, 1)
        self.assertEqual(
            tuple(shortest_path(open_grid, (0, 0), (4, 0))),
            ((1, 0), (2, 0), (3, 0), (4, 0)),
        )


if __name__ == "__main__":
    unittest.main()
