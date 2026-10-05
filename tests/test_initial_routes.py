"""Walkability validation of remaining routes on directly constructed fleets.

A robot handed to ``FleetSimulator`` may already carry a remaining ``route``.
The route omits the robot's current cell: a non-empty route must reach its
first waypoint with one orthogonal step and every later waypoint with one
orthogonal step from the previous one. Construction rejects anything that
cannot be walked cell by cell -- jumps, diagonals, repeated consecutive
waypoints, a first waypoint on the current cell, out-of-map or obstructed
waypoints, and malformed coordinates -- instead of discovering the problem
while stepping (where a jump used to count as one move and one unit of
mileage). The check applies to every robot with a route, including idle
robots that currently name no task; traffic conflicts between robots are not
creation errors and are deliberately left to execution.
"""

import copy
import json
import os
import tempfile
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


def make_sim(robots, tasks=(), grid=None):
    return FleetSimulator(
        GridMap(6, 3) if grid is None else grid, list(robots), list(tasks)
    )


def expect_rejected(testcase: unittest.TestCase, robots, *fragments, tasks=(), grid=None):
    with testcase.assertRaises(ValueError) as context:
        make_sim(robots, tasks, grid)
    message = str(context.exception)
    for fragment in fragments:
        testcase.assertIn(fragment, message)
    return message


class IllegalRouteRejectionTests(unittest.TestCase):
    def test_jump_from_start_to_third_cell_rejected(self) -> None:
        # The original bug: (0, 0) -> (3, 0) used to count as one move.
        expect_rejected(
            self,
            [Robot("R-1", (0, 0), route=[(3, 0)])],
            "R-1",
            "non-adjacently",
            "[0, 0]",
            "[3, 0]",
        )

    def test_jump_between_later_waypoints_rejected(self) -> None:
        # The defect may sit at the very end of the route; the segment and both
        # waypoints are named.
        expect_rejected(
            self,
            [Robot("R-1", (0, 0), route=[(1, 0), (2, 0), (4, 0)])],
            "R-1",
            "non-adjacently",
            "[2, 0]",
            "[4, 0]",
            "waypoint 1",
            "waypoint 2",
        )

    def test_diagonal_step_rejected(self) -> None:
        expect_rejected(
            self,
            [Robot("R-1", (0, 0), route=[(1, 0), (2, 1)])],
            "R-1",
            "non-adjacently",
            "[1, 0]",
            "[2, 1]",
        )

    def test_two_identical_consecutive_waypoints_rejected(self) -> None:
        expect_rejected(
            self,
            [Robot("R-1", (0, 0), route=[(1, 0), (1, 0), (2, 0)])],
            "R-1",
            "repeats",
            "[1, 0]",
        )

    def test_first_waypoint_equal_to_current_cell_rejected(self) -> None:
        expect_rejected(
            self,
            [Robot("R-1", (2, 0), route=[(2, 0), (3, 0)])],
            "R-1",
            "current cell",
            "[2, 0]",
        )

    def test_waypoint_outside_map_rejected(self) -> None:
        expect_rejected(
            self,
            [Robot("R-1", (4, 0), route=[(5, 0), (6, 0)])],
            "R-1",
            "outside the map",
            "[6, 0]",
            grid=GridMap(6, 2),
        )

    def test_waypoint_inside_obstacle_rejected(self) -> None:
        expect_rejected(
            self,
            [Robot("R-1", (1, 0), route=[(2, 0), (3, 0)])],
            "R-1",
            "obstacle",
            "[2, 0]",
            grid=GridMap(5, 1, frozenset({(2, 0)})),
        )

    def test_illegal_route_rejected_for_robot_without_task(self) -> None:
        # The walkability check must not be skipped just because the robot
        # currently names no task.
        expect_rejected(
            self,
            [Robot("R-idle", (0, 0), route=[(0, 2)])],
            "R-idle",
            "non-adjacently",
        )

    def test_boolean_coordinates_rejected(self) -> None:
        expect_rejected(
            self,
            [Robot("R-1", (0, 0), route=[(True, 0)])],
            "R-1",
            "integer",
        )
        expect_rejected(
            self,
            [Robot("R-1", (0, 0), route=[[1, False]])],
            "R-1",
            "integer",
        )

    def test_malformed_waypoint_shapes_raise_value_error(self) -> None:
        # Nothing here may leak a tuple-unpacking TypeError or a set-hash
        # TypeError: every malformed coordinate is a ValueError naming the
        # robot and the offending waypoint.
        bad_routes = (
            [(1, 0), (2, 0, 1)],          # entry too long
            [(1, 0), (2,)],               # entry too short
            [(1, 0), [2, "0"]],           # non-integer element
            [(1, 0), None],               # not a pair at all
            [(1, 0), 5],                  # bare integer
            [(1, 0), "(2, 0)"],           # string, despite being length five
            (1, 0),                        # the route itself is one int pair
        )
        for bad_route in bad_routes:
            with self.subTest(bad_route=bad_route):
                with self.assertRaises(ValueError) as context:
                    make_sim([Robot("R-1", (0, 0), route=bad_route)])
                self.assertIn("R-1", str(context.exception))

    def test_route_container_of_wrong_type_raises_value_error(self) -> None:
        def generator_route():
            yield (1, 0)
            yield (2, 0)

        with self.assertRaises(ValueError):
            make_sim([Robot("R-1", (0, 0), route=generator_route())])
        with self.assertRaises(ValueError):
            make_sim([Robot("R-1", (0, 0), route={(1, 0), (2, 0)})])

    def test_rejection_never_modifies_caller_objects(self) -> None:
        robots = [
            Robot("R-1", (0, 0), route=[(1, 0), (2, 0), (4, 0)], task_id="T-1"),
            Robot("R-2", (5, 1)),
        ]
        tasks = [Task("T-1", (1, 0), (5, 0), assigned_robot="R-1", picked_up=True)]
        robots_before = copy.deepcopy(robots)
        tasks_before = copy.deepcopy(tasks)
        with self.assertRaises(ValueError):
            make_sim(robots, tasks)
        # No truncation at the bad end, no padding, no reordering, nothing.
        self.assertEqual(robots, robots_before)
        self.assertEqual(tasks, tasks_before)
        self.assertEqual(robots[0].route, [(1, 0), (2, 0), (4, 0)])


class LegalRouteAcceptanceTests(unittest.TestCase):
    def test_empty_route_stays_legal_for_idle_and_bound_robots(self) -> None:
        simulator = make_sim(
            [
                Robot("R-idle", (0, 0)),
                Robot(
                    "R-busy",
                    (5, 0),
                    route=[],
                    task_id="T-1",
                    distance_travelled=3,
                ),
            ],
            [Task("T-1", (1, 0), (5, 0), assigned_robot="R-busy", picked_up=True)],
            grid=GridMap(6, 1),
        )
        self.assertEqual(simulator.robots["R-idle"].route, [])
        self.assertEqual(simulator.robots["R-busy"].route, [])

    def test_non_shortest_detour_with_revisits_is_accepted(self) -> None:
        # Walks in a little loop and comes back through cells already visited,
        # including the start cell; none of that is shortest, and all of it is
        # legal.
        route = [(0, 1), (1, 1), (1, 0), (0, 0), (1, 0), (2, 0)]
        simulator = make_sim(
            [Robot("R-1", (0, 0), route=list(route), task_id="T-1")],
            [Task("T-1", (2, 0), (2, 1), assigned_robot="R-1")],
        )
        self.assertEqual(simulator.robots["R-1"].route, route)

    def test_route_through_another_robot_is_not_a_creation_error(self) -> None:
        # R-1 plans straight through R-2's current cell; two robots also share
        # a planned cell. Traffic conflicts are settled during execution.
        simulator = make_sim(
            [
                Robot("R-1", (0, 0), route=[(1, 0), (2, 0), (3, 0)]),
                Robot("R-2", (1, 0), route=[(2, 0), (3, 0)]),
            ],
            grid=GridMap(5, 1),
        )
        self.assertEqual(simulator.robots["R-1"].route, [(1, 0), (2, 0), (3, 0)])
        self.assertEqual(simulator.robots["R-2"].route, [(2, 0), (3, 0)])

    def test_successful_creation_preserves_all_initial_state(self) -> None:
        route = [(1, 0), (2, 0), (2, 1)]
        simulator = make_sim(
            [
                Robot(
                    "R-1",
                    (0, 0),
                    route=list(route),
                    task_id="T-1",
                    distance_travelled=7,
                )
            ],
            [Task("T-1", (2, 0), (2, 1), assigned_robot="R-1")],
        )
        robot = simulator.robots["R-1"]
        task = simulator.tasks["T-1"]
        # Route, position, binding, pickup state and mileage are exactly what
        # the caller supplied ...
        self.assertEqual(robot.route, route)
        self.assertEqual(robot.position, (0, 0))
        self.assertEqual(robot.task_id, "T-1")
        self.assertEqual(robot.distance_travelled, 7)
        self.assertEqual(task.assigned_robot, "R-1")
        self.assertFalse(task.picked_up)
        self.assertFalse(task.completed)
        # ... the clock is still at zero, no history exists, and construction
        # confirmed neither a pickup nor a delivery.
        self.assertEqual(simulator.tick, 0)
        self.assertEqual(simulator.replay, [])
        self.assertEqual(simulator.map_changes, [])

    def test_legal_route_still_moves_one_cell_per_tick(self) -> None:
        # The motivating failure mode: a valid adjacent route advances exactly
        # one cell and one mileage unit per tick.
        simulator = make_sim(
            [Robot("R-1", (0, 0), route=[(1, 0), (2, 0), (3, 0)])],
            grid=GridMap(4, 1),
        )
        event = simulator.step()
        self.assertEqual(event["moved"], ["R-1"])
        self.assertEqual(simulator.robots["R-1"].position, (1, 0))
        self.assertEqual(simulator.robots["R-1"].distance_travelled, 1)
        self.assertEqual(simulator.robots["R-1"].route, [(2, 0), (3, 0)])

    def test_map_paused_robot_with_empty_route_round_trips(self) -> None:
        # A robot paused by map unreachability keeps an empty route; it must
        # still construct directly, survive a checkpoint round trip and resume
        # once the way reopens -- with the checkpoint format untouched.
        grid = GridMap(4, 1, frozenset({(3, 0)}))
        simulator = make_sim(
            [Robot("A", (1, 0), route=[], task_id="T-1")],
            [Task("T-1", (1, 0), (3, 0), assigned_robot="A", picked_up=True)],
            grid=grid,
        )
        simulator.paused_tasks.add("T-1")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            simulator.save_checkpoint(path)
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
            self.assertEqual(document["version"], 2)
            loaded = FleetSimulator.load_checkpoint(path)
        self.assertEqual(loaded.paused_tasks, {"T-1"})
        self.assertEqual(loaded.robots["A"].route, [])
        loaded.modify_obstacles(removed=[(3, 0)])
        self.assertEqual(loaded.paused_tasks, set())
        self.assertEqual(loaded.robots["A"].route, [(2, 0), (3, 0)])


if __name__ == "__main__":
    unittest.main()
