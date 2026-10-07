"""Regression guarantees for the isolation of ``GridMap`` obstacle content.

A map keeps exactly the obstacle content it was created with: the caller's
collection is snapshotted into a frozenset at construction, so adding,
removing or clearing entries in the original set afterwards never rewrites
an existing map -- not before a fleet exists, and not after a fleet has
applied formal map edits on top of it. ``GridMap.obstacles`` is a genuine
frozenset (direct mutation attempts raise ``AttributeError``), construction
never mutates the caller's collection, and the change history only ever
records edits made through ``FleetSimulator.modify_obstacles``.
"""

import os
import tempfile
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot
from warehouse_fleet.pathfinding import shortest_path


class GridMapSnapshotTests(unittest.TestCase):
    """Isolation holds from the moment ``GridMap`` construction finishes."""

    def test_caller_add_remove_clear_never_reach_created_map(self) -> None:
        cells = {(1, 0)}
        grid = GridMap(5, 2, cells)

        cells.add((2, 0))
        self.assertTrue(grid.traversable((2, 0)))
        cells.discard((1, 0))
        self.assertFalse(grid.traversable((1, 0)))
        cells.clear()
        cells.add((3, 1))
        self.assertEqual(grid.obstacles, frozenset({(1, 0)}))
        self.assertTrue(grid.traversable((3, 1)))

    def test_pathfinding_uses_creation_time_content(self) -> None:
        cells = {(2, 0)}
        grid = GridMap(5, 1, cells)
        cells.clear()
        # The corridor is still sealed at (2, 0) as far as the map knows.
        with self.assertRaises(ValueError):
            shortest_path(grid, (0, 0), (4, 0))
        cells.update({(0, 0), (1, 0), (3, 0), (4, 0)})
        self.assertTrue(grid.traversable((0, 0)))
        self.assertTrue(grid.traversable((4, 0)))

    def test_obstacles_is_a_genuine_frozenset(self) -> None:
        grid = GridMap(5, 2, {(1, 0)})
        self.assertIsInstance(grid.obstacles, frozenset)
        for mutation in (
            lambda: grid.obstacles.add((2, 0)),
            lambda: grid.obstacles.remove((1, 0)),
            lambda: grid.obstacles.clear(),
        ):
            with self.assertRaises(AttributeError):
                mutation()
        self.assertEqual(grid.obstacles, frozenset({(1, 0)}))

    def test_construction_does_not_mutate_the_caller_collection(self) -> None:
        cells = {(1, 0)}
        GridMap(5, 2, cells)
        self.assertEqual(cells, {(1, 0)})

    def test_empty_and_omitted_obstacles_stay_supported(self) -> None:
        self.assertEqual(GridMap(3, 3, set()).obstacles, frozenset())
        self.assertEqual(GridMap(3, 3, frozenset()).obstacles, frozenset())
        self.assertEqual(GridMap(3, 3).obstacles, frozenset())

    def test_same_source_set_gives_each_map_its_own_creation_content(self) -> None:
        shared = {(0, 0)}
        first = GridMap(3, 3, shared)
        shared.add((1, 1))
        second = GridMap(3, 3, shared)
        shared.clear()
        self.assertEqual(first.obstacles, frozenset({(0, 0)}))
        self.assertEqual(second.obstacles, frozenset({(0, 0), (1, 1)}))

    def test_out_of_bounds_rejection_and_dimensions_unchanged(self) -> None:
        with self.assertRaises(ValueError):
            GridMap(2, 2, {(2, 0)})
        with self.assertRaises(ValueError):
            GridMap(0, 2)
        with self.assertRaises(ValueError):
            GridMap(2, -1)


class FleetIsolationTests(unittest.TestCase):
    """Caller-side set mutations are never simulation map edits."""

    def test_fleet_uses_creation_time_map_for_positions_and_routes(self) -> None:
        cells = {(1, 0)}
        grid = GridMap(5, 1, cells)
        cells.clear()

        # (1, 0) is still closed on the map the fleet was handed.
        with self.assertRaises(ValueError):
            FleetSimulator(grid, [Robot("R", (1, 0))], [])
        with self.assertRaises(ValueError):
            FleetSimulator(grid, [Robot("R", (0, 0), route=[(1, 0), (2, 0)])], [])

        # A legal start on the creation-time map still works.
        simulator = FleetSimulator(grid, [Robot("R", (0, 0))], [])
        self.assertEqual(simulator.grid.obstacles, frozenset({(1, 0)}))

    def test_caller_mutation_after_formal_edit_rewrites_nothing(self) -> None:
        cells = {(1, 0)}
        grid = GridMap(5, 2, cells)
        simulator = FleetSimulator(grid, [Robot("R", (0, 0))], [])
        simulator.modify_obstacles(added=[(3, 0)])

        cells.clear()
        cells.add((2, 0))

        # The current map keeps exactly the initial content plus the formal
        # edit; the initial map keeps exactly the creation-time content.
        self.assertEqual(simulator.grid.obstacles, frozenset({(1, 0), (3, 0)}))
        self.assertEqual(simulator.base_grid.obstacles, frozenset({(1, 0)}))
        self.assertTrue(simulator.grid.traversable((2, 0)))
        # The only recorded change is the formal closure of (3, 0).
        self.assertEqual(
            simulator.map_change_history(),
            [
                {
                    "type": "map_change",
                    "tick": 0,
                    "sequence": 1,
                    "added": [[3, 0]],
                    "removed": [],
                }
            ],
        )

    def test_saved_state_reloads_consistently_after_caller_mutation(self) -> None:
        cells = {(1, 0)}
        grid = GridMap(5, 2, cells)
        simulator = FleetSimulator(grid, [Robot("R", (0, 0))], [])
        simulator.modify_obstacles(added=[(3, 0)])
        cells.clear()
        cells.add((2, 0))

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.json")
            simulator.save_checkpoint(path)
            loaded = FleetSimulator.load_checkpoint(path)

        self.assertEqual(loaded.grid.obstacles, frozenset({(1, 0), (3, 0)}))
        self.assertEqual(loaded.base_grid.obstacles, frozenset({(1, 0)}))
        self.assertEqual(loaded.map_change_history(), simulator.map_change_history())
        self.assertEqual(len(loaded.map_change_history()), 1)

    def test_editing_one_fleet_never_touches_another_fleets_maps(self) -> None:
        cells = {(1, 0)}
        grid = GridMap(4, 1, cells)
        first = FleetSimulator(grid, [Robot("A", (0, 0))], [])
        second = FleetSimulator(grid, [Robot("B", (3, 0))], [])

        first.modify_obstacles(added=[(2, 0)])
        cells.clear()

        self.assertEqual(first.grid.obstacles, frozenset({(1, 0), (2, 0)}))
        self.assertEqual(second.grid.obstacles, frozenset({(1, 0)}))
        self.assertEqual(second.base_grid.obstacles, frozenset({(1, 0)}))
        self.assertEqual(second.map_change_history(), [])


if __name__ == "__main__":
    unittest.main()
