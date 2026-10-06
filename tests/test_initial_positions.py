"""Validation and normalization of robot starting positions on construction.

A robot handed to ``FleetSimulator`` may name its starting position with a
list or a tuple of exactly two plain integers; the spellings may be mixed in
the same fleet. Construction rejects everything that cannot denote such a
coordinate pair -- booleans, floats, strings, too-short or too-long pairs,
bare numbers and other non-pair values -- with a :class:`ValueError` naming
the robot and its starting position, instead of letting a boolean/float act
as a coordinate or leaking a set-hashing ``TypeError`` when the position came
in as a list.

Legal starts still have to lie in bounds and off every obstacle, no two
robots may occupy the same start cell regardless of container spelling, and
a successful construction stores every position as an independent integer
tuple: editing the caller's original coordinate list afterwards cannot move
the created robot. Nothing is committed until every robot and task passes,
so a rejection anywhere -- even on a later robot's route or task binding --
leaves all caller objects exactly as supplied.
"""

import copy
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


class MalformedStartingPositionTests(unittest.TestCase):
    BAD_POSITIONS = (
        True,                 # boolean masquerading as an int
        (True, False),        # two booleans
        [False, 1],
        (1, True),
        (1, 2.0),             # float, never rounded
        [1.5, 2],
        ("0", 0),             # string, never converted
        [0, "1"],
        (0,),                 # missing component
        [0],
        (0, 0, 0),            # extra component
        [0, 0, 1],
        5,                    # bare number, not a pair
        None,
        "(0, 0)",             # string despite being the right text
        {0, 1},               # unordered container, not a pair
        b"(0, 0)",            # bytes, never treated as a character sequence
    )

    def test_every_bad_shape_is_a_value_error_naming_robot_and_position(self) -> None:
        for bad in self.BAD_POSITIONS:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as context:
                    make_sim([Robot("R-1", bad)])
                message = str(context.exception)
                self.assertIn("R-1", message)
                self.assertIn("starting position", message)
                # The rejection is always a ValueError, never a leaked
                # unpacking/hashing TypeError.
                self.assertNotIn("unhashable", message)

    def test_non_sequence_position_never_leaks_a_type_error(self) -> None:
        def broken_pair():
            yield 0
            yield 0

        for bad in (broken_pair(), iter([0, 0])):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    make_sim([Robot("R-1", bad)])

    def test_bad_position_is_rejected_even_when_another_robot_overlaps(self) -> None:
        # Shape validation runs before the overlap check, so the malformed
        # start is reported as such rather than slipping past or crashing.
        expect_rejected(
            self,
            [Robot("R-1", (0, 0)), Robot("R-2", [True, 0])],
            "R-2",
            "starting position",
        )

    def test_bad_start_reported_for_task_bound_and_preset_route_robots(self) -> None:
        # No robot class is exempt from the coordinate rule.
        expect_rejected(
            self,
            [
                Robot(
                    "R-busy",
                    (True, 0),
                    route=[(1, 0)],
                    task_id="T-1",
                )
            ],
            "R-busy",
            "starting position",
            tasks=[Task("T-1", (0, 0), (2, 0), assigned_robot="R-busy", picked_up=True)],
        )
        expect_rejected(
            self,
            [Robot("R-preset", 1.5, route=[(1, 0)])],
            "R-preset",
            "starting position",
        )


class LegalStartingPositionTests(unittest.TestCase):
    def test_list_position_is_normalized_to_an_integer_tuple(self) -> None:
        simulator = make_sim([Robot("R-1", [2, 1])])
        position = simulator.robots["R-1"].position
        self.assertEqual(position, (2, 1))
        self.assertIsInstance(position, tuple)
        self.assertTrue(all(isinstance(coord, int) for coord in position))
        self.assertFalse(any(isinstance(coord, bool) for coord in position))

    def test_list_and_tuple_spellings_may_be_mixed_in_one_fleet(self) -> None:
        simulator = make_sim(
            [
                Robot("R-list", [0, 0]),
                Robot("R-tuple", (1, 0)),
                Robot("R-mixed", [2, 0]),
            ],
            grid=GridMap(4, 1),
        )
        self.assertEqual(
            {robot_id: robot.position for robot_id, robot in simulator.robots.items()},
            {"R-list": (0, 0), "R-tuple": (1, 0), "R-mixed": (2, 0)},
        )

    def test_list_and_tuple_starts_produce_identical_fleets_and_runs(self) -> None:
        # The two spellings must agree on position, movement, remaining routes
        # and task assignment at every tick.
        def build(start):
            return FleetSimulator(
                GridMap(4, 2),
                [
                    Robot(
                        "A",
                        start,
                        route=[[1, 0], (2, 0), (2, 1)],
                        task_id="T-1",
                    )
                ],
                [Task("T-1", (2, 0), (2, 1), assigned_robot="A")],
            )

        list_sim = build([0, 0])
        tuple_sim = build((0, 0))
        for tick in range(3):
            with self.subTest(tick=tick):
                list_event = list_sim.step()
                tuple_event = tuple_sim.step()
                self.assertEqual(
                    list_sim.robots["A"].position, tuple_sim.robots["A"].position
                )
                self.assertEqual(
                    list_sim.robots["A"].route, tuple_sim.robots["A"].route
                )
                self.assertEqual(
                    list_sim.robots["A"].distance_travelled,
                    tuple_sim.robots["A"].distance_travelled,
                )
                self.assertEqual(list_event, tuple_event)
        self.assertTrue(list_sim.tasks["T-1"].completed)
        self.assertTrue(tuple_sim.tasks["T-1"].completed)

    def test_first_route_waypoint_equal_to_list_start_is_still_rejected(self) -> None:
        # A list start compares equal by value to a list/tuple first waypoint:
        # the current-cell rejection must not depend on container spelling.
        expect_rejected(
            self,
            [Robot("R-1", [0, 0], route=[[0, 0], [1, 0]])],
            "R-1",
            "current cell",
            "[0, 0]",
            grid=GridMap(3, 1),
        )
        expect_rejected(
            self,
            [Robot("R-1", [0, 0], route=[(0, 0), (1, 0)])],
            "R-1",
            "current cell",
            grid=GridMap(3, 1),
        )


class CallerObjectOwnershipTests(unittest.TestCase):
    def test_caller_position_list_is_neither_mutated_nor_aliased(self) -> None:
        caller_start = [3, 1]
        snapshot = list(caller_start)
        simulator = make_sim([Robot("R-1", caller_start)])
        self.assertEqual(caller_start, snapshot)
        self.assertIsNot(simulator.robots["R-1"].position, caller_start)

    def test_editing_the_callers_position_list_after_creation_changes_nothing(self) -> None:
        caller_start = [0, 0]
        simulator = make_sim(
            [Robot("A", caller_start)],
            [Task("T-1", (4, 0), (5, 0))],
            grid=GridMap(6, 1),
        )
        caller_start[0] = 5
        caller_start.append(9)
        simulator.step()
        self.assertEqual(simulator.robots["A"].position, (1, 0))
        # The task was assigned from the stored (0, 0), never from the mutated
        # list, and mileage/history are ordinary.
        self.assertEqual(simulator.robots["A"].distance_travelled, 1)
        self.assertEqual(simulator.tasks["T-1"].assigned_robot, "A")

    def test_failed_creation_leaves_every_robot_and_task_untouched(self) -> None:
        robots = [
            Robot("R-1", [0, 0], route=[(1, 0), (2, 0)], task_id="T-1"),
            Robot("R-2", [5, 1]),
            # R-3 passes every position check; its later route check fails on
            # the non-adjacent (3, 1) -> (5, 1) jump.
            Robot("R-3", [3, 0], route=[(3, 1), (5, 1)]),
        ]
        tasks = [Task("T-1", (2, 0), (5, 0), assigned_robot="R-1")]
        robots_before = copy.deepcopy(robots)
        tasks_before = copy.deepcopy(tasks)
        with self.assertRaises(ValueError):
            make_sim(robots, tasks, grid=GridMap(6, 2))
        self.assertEqual(robots, robots_before)
        self.assertEqual(tasks, tasks_before)
        # Earlier robots kept their exact list positions and routes: no
        # normalized tuple or trimmed waypoint list was committed anywhere.
        self.assertIsInstance(robots[0].position, list)
        self.assertEqual(robots[0].position, [0, 0])
        self.assertEqual(robots[0].route, [(1, 0), (2, 0)])
        self.assertEqual(robots[2].route, [(3, 1), (5, 1)])

    def test_ownership_failure_after_valid_positions_leaves_objects_as_passed(self) -> None:
        robots = [
            Robot("R-1", [0, 0], task_id="T-missing"),
            Robot("R-2", (1, 0)),
        ]
        robots_before = copy.deepcopy(robots)
        with self.assertRaises(ValueError):
            make_sim(robots, [])
        self.assertEqual(robots, robots_before)
        self.assertIsInstance(robots[0].position, list)
        self.assertEqual(robots[0].position, [0, 0])


class OverlappingStartTests(unittest.TestCase):
    def test_overlap_detected_across_container_spellings(self) -> None:
        for first, second in (
            ([1, 1], (1, 1)),
            ((1, 1), [1, 1]),
            ([1, 1], [1, 1]),
            ((1, 1), (1, 1)),
        ):
            with self.subTest(first=first, second=second):
                message = expect_rejected(
                    self,
                    [Robot("A", first), Robot("B", second)],
                    "share",
                    "A",
                    "B",
                    "[1, 1]",
                )
                self.assertNotIn("unhashable", message)

    def test_overlap_names_conflicting_pair_among_three_robots(self) -> None:
        message = expect_rejected(
            self,
            [
                Robot("A", [0, 0]),
                Robot("B", (2, 0)),
                Robot("C", [2, 0]),
            ],
            "B",
            "C",
            "[2, 0]",
        )
        self.assertNotIn("'A'", message.split("share")[0])

    def test_overlap_rule_covers_every_kind_of_robot(self) -> None:
        # Taskless idle robot vs task-bound robot carrying a route.
        expect_rejected(
            self,
            [
                Robot("A", (1, 0)),
                Robot(
                    "B",
                    [1, 0],
                    route=[(2, 0), (3, 0)],
                    task_id="T-1",
                ),
            ],
            "A",
            "B",
            tasks=[Task("T-1", (3, 0), (4, 0), assigned_robot="B", picked_up=True)],
            grid=GridMap(6, 1),
        )
        # No robot is dropped to make the batch succeed: taskless preset-route
        # robot overlapping an idle one is rejected too.
        expect_rejected(
            self,
            [
                Robot("A", [1, 0]),
                Robot("P", (1, 0), route=[(2, 0)]),
            ],
            "A",
            "P",
            grid=GridMap(4, 1),
        )

    def test_start_on_out_of_bounds_or_obstacle_cell_is_rejected(self) -> None:
        expect_rejected(
            self,
            [Robot("R-1", [6, 0])],
            "R-1",
            "[6, 0]",
            "outside the map or inside an obstacle",
            grid=GridMap(6, 2),
        )
        expect_rejected(
            self,
            [Robot("R-1", (2, 2))],
            "R-1",
            "[2, 2]",
            "outside the map or inside an obstacle",
            grid=GridMap(4, 4, frozenset({(2, 2)})),
        )


class CreationSideEffectsTests(unittest.TestCase):
    def test_creation_does_not_move_advance_or_record(self) -> None:
        route = [(1, 0), (2, 0)]
        simulator = make_sim(
            [Robot("A", [0, 0], route=list(route))],
            grid=GridMap(4, 1),
        )
        robot = simulator.robots["A"]
        self.assertEqual(robot.position, (0, 0))
        # The remaining route still starts with the cell after the start.
        self.assertEqual(robot.route, [(1, 0), (2, 0)])
        self.assertEqual(robot.distance_travelled, 0)
        self.assertEqual(simulator.tick, 0)
        self.assertEqual(simulator.replay, [])
        self.assertEqual(simulator.map_changes, [])

    def test_integer_start_round_trips_through_a_checkpoint(self) -> None:
        grid = GridMap(5, 2)
        simulator = FleetSimulator(
            grid,
            [Robot("A", [0, 0], route=[(1, 0), (2, 0)])],
            [],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            simulator.save_checkpoint(path)
            loaded = FleetSimulator.load_checkpoint(path)
        self.assertEqual(loaded.robots["A"].position, (0, 0))
        self.assertIsInstance(loaded.robots["A"].position, tuple)
        self.assertEqual(loaded.robots["A"].route, [(1, 0), (2, 0)])
        event = loaded.step()
        self.assertEqual(loaded.robots["A"].position, (1, 0))
        self.assertEqual(event["robots"]["A"], [1, 0])


if __name__ == "__main__":
    unittest.main()
