"""Regression guarantees for the independence of a map's initial obstacles.

A ``GridMap`` must keep the obstacle layout it was created with even when the
caller built it from a mutable ``set`` and keeps editing that set afterwards:
adding, removing or clearing cells on the caller's container is ordinary
caller-side bookkeeping, never a simulation map edit. The checks here verify
all of the following together:

* ``GridMap.obstacles`` is an actual ``frozenset`` snapshot of the construction
  moment, not an alias of the caller's container, so ``add``/``remove``/``clear``
  on the attribute raise ``AttributeError`` and leave the map unchanged;
* later mutation of the source set -- adding cells, deleting them or clearing
  the whole container -- changes neither traversability nor ``shortest_path``
  results of an existing map, and the map is isolated as soon as it is built,
  before any fleet exists;
* building a map never mutates the passed-in container, the same source may be
  reused for several maps that each freeze their own construction-time view,
  and ``set``, ``frozenset``, an empty container and an omitted argument all
  keep working;
* a formal ``modify_obstacles`` edit is the only thing that changes a fleet's
  current map and map-change history: after such an edit, source-set mutation
  can neither rewrite the frozen initial (``base_grid``) map nor fabricate or
  erase change records, a later fleet construction still validates starting
  positions and routes against the frozen map, editing one fleet never reaches
  another fleet, and the saved checkpoint still passes the strict
  history-reproduces-grid consistency check after a round trip -- isolation is
  never achieved by loosening loading validation.
"""

import json
import os
import tempfile
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task, shortest_path


class GridMapObstacleSnapshotTests(unittest.TestCase):
    def test_obstacles_is_a_real_frozenset_not_an_alias(self) -> None:
        source = {(1, 0)}
        grid = GridMap(5, 2, source)
        self.assertIsInstance(grid.obstacles, frozenset)
        self.assertIsNot(grid.obstacles, source)
        self.assertEqual(grid.obstacles, frozenset({(1, 0)}))

    def test_frozen_obstacles_reject_mutating_methods(self) -> None:
        grid = GridMap(5, 2, {(1, 0)})
        with self.assertRaises(AttributeError):
            grid.obstacles.add((2, 0))
        with self.assertRaises(AttributeError):
            grid.obstacles.remove((1, 0))
        with self.assertRaises(AttributeError):
            grid.obstacles.clear()
        # A rejected mutation attempt leaves the map intact.
        self.assertEqual(grid.obstacles, frozenset({(1, 0)}))
        self.assertTrue(grid.traversable((2, 0)))
        self.assertFalse(grid.traversable((1, 0)))

    def test_source_set_mutation_never_edits_an_existing_map(self) -> None:
        source = {(1, 0)}
        grid = GridMap(5, 2, source)

        # A caller-side addition is not a map closure.
        source.add((2, 0))
        self.assertEqual(grid.obstacles, frozenset({(1, 0)}))
        self.assertTrue(grid.traversable((2, 0)))
        self.assertFalse(grid.traversable((1, 0)))
        # The route comes back to the y=0 row at the open cell (2, 0).
        self.assertEqual(
            shortest_path(grid, (0, 0), (4, 0)),
            [(0, 1), (1, 1), (2, 1), (2, 0), (3, 0), (4, 0)],
        )

        # A caller-side deletion does not reopen a map obstacle.
        source.discard((1, 0))
        self.assertFalse(grid.traversable((1, 0)))
        with self.assertRaises(ValueError):
            shortest_path(grid, (1, 0), (0, 0))

        # Clearing the caller's container changes nothing on the built map.
        source.clear()
        self.assertEqual(grid.obstacles, frozenset({(1, 0)}))
        self.assertFalse(grid.traversable((1, 0)))
        self.assertTrue(grid.traversable((3, 0)))

    def test_isolation_holds_before_any_fleet_exists(self) -> None:
        source = {(0, 0)}
        grid = GridMap(3, 2, source)
        source.update({(1, 0), (2, 0)})
        source.clear()
        source.add((2, 1))
        self.assertEqual(grid.obstacles, frozenset({(0, 0)}))
        self.assertFalse(grid.traversable((0, 0)))
        self.assertTrue(grid.traversable((1, 0)))
        self.assertTrue(grid.traversable((2, 1)))
        self.assertEqual(shortest_path(grid, (0, 1), (2, 1)), [(1, 1), (2, 1)])

    def test_construction_never_mutates_the_source_container(self) -> None:
        source = {(1, 0), (2, 0)}
        GridMap(5, 2, source)
        self.assertEqual(source, {(1, 0), (2, 0)})

        # A rejected construction leaves the caller's container untouched too.
        bad = {(9, 9)}
        with self.assertRaises(ValueError):
            GridMap(2, 2, bad)
        self.assertEqual(bad, {(9, 9)})

    def test_supported_construction_forms_keep_working(self) -> None:
        self.assertEqual(GridMap(3, 2, frozenset({(0, 0)})).obstacles, frozenset({(0, 0)}))
        self.assertEqual(GridMap(3, 2, {(0, 0)}).obstacles, frozenset({(0, 0)}))
        self.assertEqual(GridMap(3, 2, set()).obstacles, frozenset())
        self.assertEqual(GridMap(3, 2, []).obstacles, frozenset())
        self.assertEqual(GridMap(3, 2).obstacles, frozenset())

    def test_same_source_reused_each_map_freezes_its_own_view(self) -> None:
        source = {(0, 0)}
        first = GridMap(4, 1, source)
        source.add((1, 0))
        second = GridMap(4, 1, source)
        source.discard((0, 0))
        third = GridMap(4, 1, source)
        source.clear()

        self.assertEqual(first.obstacles, frozenset({(0, 0)}))
        self.assertEqual(second.obstacles, frozenset({(0, 0), (1, 0)}))
        self.assertEqual(third.obstacles, frozenset({(1, 0)}))


def edited_fleet(source: set[tuple[int, int]]) -> tuple[FleetSimulator, GridMap]:
    """Map 5x2 built from *source*; a fleet then formally closes (3, 0)."""
    grid = GridMap(5, 2, source)
    simulator = FleetSimulator(
        grid,
        [Robot("A", (0, 0))],
        [Task("T1", (2, 0), (4, 0))],
    )
    simulator.modify_obstacles(added=[(3, 0)])
    return simulator, grid


class FleetInitialObstacleIsolationTests(unittest.TestCase):
    def test_source_mutation_after_formal_edit_changes_nothing_recorded(self) -> None:
        source = {(1, 0)}
        simulator, _grid = edited_fleet(source)

        # The caller repurposes its own set after the formal edit.
        source.clear()
        source.add((2, 0))

        # The current map keeps both closures, the initial map only the first,
        # and the history records only the one formal edit.
        self.assertEqual(simulator.grid.obstacles, frozenset({(1, 0), (3, 0)}))
        self.assertEqual(simulator.base_grid.obstacles, frozenset({(1, 0)}))
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
        self.assertEqual(
            simulator.status()["obstacles"], [[1, 0], [3, 0]]
        )
        # The caller's new cell is open in the simulation.
        self.assertTrue(simulator.grid.traversable((2, 0)))

    def test_later_fleet_validates_against_the_frozen_map(self) -> None:
        source = {(1, 0)}
        _simulator, grid = edited_fleet(source)
        source.clear()
        source.add((2, 0))

        # (2, 0) is open on the frozen map even though the caller's set closes
        # it, so the start and route validate; (1, 0) stays an obstacle.
        fleet = FleetSimulator(
            grid,
            [Robot("B", (2, 0), route=[(2, 1)])],
            [Task("T2", (2, 0), (0, 0))],
        )
        self.assertEqual(fleet.robots["B"].position, (2, 0))
        self.assertEqual(fleet.robots["B"].route, [(2, 1)])
        with self.assertRaises(ValueError):
            FleetSimulator(grid, [Robot("C", (1, 0))], [])

    def test_editing_one_fleet_never_reaches_another_fleet(self) -> None:
        source = {(1, 0)}
        first, grid = edited_fleet(source)
        second = FleetSimulator(grid, [Robot("B", (0, 1))], [])

        first.modify_obstacles(added=[(4, 1)])
        first.modify_obstacles(removed=[(1, 0)])

        self.assertEqual(second.grid.obstacles, frozenset({(1, 0)}))
        self.assertEqual(second.base_grid.obstacles, frozenset({(1, 0)}))
        self.assertEqual(second.map_change_history(), [])
        self.assertFalse(second.grid.traversable((1, 0)))
        self.assertTrue(second.grid.traversable((4, 1)))

    def test_checkpoint_round_trip_survives_later_source_mutation(self) -> None:
        source = {(1, 0)}
        simulator, _grid = edited_fleet(source)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.json")
            simulator.save_checkpoint(path)
            source.clear()
            source.add((2, 0))

            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
            self.assertEqual(document["grid"]["obstacles"], [[1, 0], [3, 0]])
            self.assertEqual(document["base_grid"]["obstacles"], [[1, 0]])
            self.assertEqual(
                [frame["added"] for frame in document["map_changes"]], [[[3, 0]]]
            )

            loaded = FleetSimulator.load_checkpoint(path)
            self.assertEqual(loaded.grid.obstacles, frozenset({(1, 0), (3, 0)}))
            self.assertEqual(loaded.base_grid.obstacles, frozenset({(1, 0)}))
            self.assertEqual(
                [frame["added"] for frame in loaded.map_change_history()], [[[3, 0]]]
            )
            # Further source mutation still produces no record on the loaded fleet.
            source.add((4, 1))
            self.assertEqual(
                [frame["added"] for frame in loaded.map_change_history()], [[[3, 0]]]
            )
            self.assertTrue(loaded.grid.traversable((4, 1)))

    def test_checkpoint_consistency_check_is_not_weakened(self) -> None:
        source = {(1, 0)}
        simulator, _grid = edited_fleet(source)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.json")
            simulator.save_checkpoint(path)
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
            # Pretend the initial map had a different obstacle: the recorded
            # history can no longer reproduce the saved current grid.
            document["base_grid"]["obstacles"] = [[2, 0]]
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(document, handle)
            with self.assertRaises(ValueError):
                FleetSimulator.load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
