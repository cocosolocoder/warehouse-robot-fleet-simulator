"""Validation of robot starting positions on directly constructed fleets.

A robot handed to ``FleetSimulator`` carries its starting cell in
``position``. Construction must apply the same strict coordinate rule the
route waypoints use: only a list or tuple of two plain integers is accepted,
every accepted position becomes a fresh integer tuple, illegal values are
rejected (never rounded or coerced), positions must lie in traversable map
space, and no two robots may share one start cell -- regardless of list/tuple
spelling. Nothing is committed until the whole batch passes, so a later
robot's route or task failure leaves every caller object untouched.
"""

import os
import tempfile
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


GRID = GridMap(6, 3, frozenset({(3, 2)}))


def make_sim(robots, tasks=(), grid=None):
    return FleetSimulator(
        GridMap(6, 3) if grid is None else grid, list(robots), list(tasks)
    )


def expect_rejected(testcase, robots, *fragments, tasks=(), grid=None):
    with testcase.assertRaises(ValueError) as context:
        make_sim(robots, tasks, grid)
    message = str(context.exception)
    for fragment in fragments:
        testcase.assertIn(fragment, message)
    return message


class MalformedStartingPositionTests(unittest.TestCase):
    def test_booleans_floats_strings_and_scalars_are_rejected(self) -> None:
        for bad in (True, False, 1, 1.0, 0.0, "00", None, {1: 2}):
            expect_rejected(
                self, [Robot("R-1", bad)], "R-1", "starting position"
            )

    def test_wrong_lengths_and_components_are_rejected(self) -> None:
        for bad in ([1], (1,), [1, 2, 3], (1, 2, 3), [], ()):
            expect_rejected(
                self, [Robot("R-1", bad)], "R-1", "starting position"
            )

    def test_non_integer_components_are_rejected_without_coercion(self) -> None:
        for bad in (
            [True, 0],
            [0, False],
            [1.0, 2],
            [1, 2.0],
            ["1", 2],
            [1, "2"],
            [None, 0],
        ):
            expect_rejected(
                self, [Robot("R-1", bad)], "R-1", "starting position"
            )

    def test_rejection_names_the_offending_robot(self) -> None:
        message = expect_rejected(
            self,
            [Robot("R-1", (0, 0)), Robot("R-bad", [1, True])],
            "R-bad",
            "starting position",
        )
        self.assertNotIn("TypeError", message)

    def test_malformed_pair_raises_value_error_not_type_error(self) -> None:
        # A list position used to reach a set/hashed comparison that raised
        # TypeError before the overlap check; it must now be a ValueError
        # explaining the robot and the position instead.
        for bad in (True, 1.5, [0, 0, 0]):
            with self.assertRaises(ValueError):
                make_sim([Robot("R-1", [0, 0]), Robot("R-2", bad)])


class PositionNormalizationTests(unittest.TestCase):
    def test_list_and_tuple_create_identical_tuple_positions(self) -> None:
        list_sim = make_sim([Robot("R-1", [0, 0])])
        tuple_sim = make_sim([Robot("R-1", (0, 0))])
        self.assertEqual(list_sim.robots["R-1"].position, (0, 0))
        self.assertEqual(tuple_sim.robots["R-1"].position, (0, 0))
        self.assertIsInstance(list_sim.robots["R-1"].position, tuple)

    def test_spellings_may_be_mixed_in_one_batch(self) -> None:
        sim = make_sim([Robot("R-1", [0, 0]), Robot("R-2", (1, 0))])
        self.assertEqual(
            [robot.position for robot in sim.robots.values()],
            [(0, 0), (1, 0)],
        )

    def test_list_and_tuple_starts_move_and_assign_identically(self) -> None:
        task_list = Task("T-1", (0, 2), (2, 2))
        task_tuple = Task("T-1", (0, 2), (2, 2))
        list_sim = make_sim(
            [Robot("R-1", [0, 0], [(0, 1)])], [task_list],
            grid=GridMap(6, 4),
        )
        tuple_sim = make_sim(
            [Robot("R-1", (0, 0), [(0, 1)])], [task_tuple],
            grid=GridMap(6, 4),
        )
        for _ in range(4):
            list_sim.step()
            tuple_sim.step()
        self.assertEqual(
            list_sim.robots["R-1"].position, tuple_sim.robots["R-1"].position
        )
        self.assertEqual(
            list_sim.robots["R-1"].distance_travelled,
            tuple_sim.robots["R-1"].distance_travelled,
        )
        self.assertEqual(
            [frame["robots"] for frame in list_sim.replay],
            [frame["robots"] for frame in tuple_sim.replay],
        )

    def test_caller_list_is_not_mutated_by_construction(self) -> None:
        origin = [0, 0]
        robot = Robot("R-1", origin)
        make_sim([robot])
        self.assertEqual(origin, [0, 0])
        self.assertIsNot(robot.position, origin)
        self.assertEqual(robot.position, (0, 0))

    def test_mutating_the_caller_list_after_creation_changes_nothing(self) -> None:
        origin = [0, 0]
        robot = Robot("R-1", origin)
        sim = make_sim([robot])
        origin[0] = 5
        origin.append(9)
        self.assertEqual(sim.robots["R-1"].position, (0, 0))
        self.assertEqual(robot.position, (0, 0))


class OverlappingStartTests(unittest.TestCase):
    def test_same_spelling_collision_is_rejected(self) -> None:
        expect_rejected(
            self,
            [Robot("R-1", (1, 1)), Robot("R-2", (1, 1))],
            "R-1",
            "R-2",
            grid=GRID,
        )

    def test_list_and_tuple_collision_is_value_comparison(self) -> None:
        message = expect_rejected(
            self,
            [Robot("R-1", [1, 1]), Robot("R-2", (1, 1))],
            "R-1",
            "R-2",
            grid=GRID,
        )
        self.assertNotIn("TypeError", message)

    def test_reverse_order_still_names_both_robots(self) -> None:
        expect_rejected(
            self,
            [Robot("R-first", (2, 2)), Robot("R-second", [2, 2])],
            "R-first",
            "R-second",
            grid=GRID,
        )

    def test_out_of_bounds_and_obstacle_starts_are_rejected(self) -> None:
        expect_rejected(
            self, [Robot("R-1", [6, 0])], "R-1", "starting position", grid=GRID
        )
        expect_rejected(
            self, [Robot("R-1", [-1, 0])], "R-1", "starting position", grid=GRID
        )
        expect_rejected(
            self, [Robot("R-1", [3, 2])], "R-1", "starting position", grid=GRID
        )


class RuleAppliesToEveryRobotTests(unittest.TestCase):
    def test_taskless_robot_with_preset_route_checked(self) -> None:
        expect_rejected(
            self,
            [Robot("R-1", [0, 0], [(0, 1)]), Robot("R-2", True)],
            "R-2",
        )

    def test_task_bound_robot_checked(self) -> None:
        task = Task("T-1", (0, 1), (0, 2), assigned_robot="R-1")
        robot = Robot("R-1", [0, True], [(0, 1)], task_id="T-1")
        expect_rejected(
            self, [robot], "R-1", "starting position", tasks=[task]
        )

    def test_overlap_caught_even_when_one_robot_is_preset_routed(self) -> None:
        expect_rejected(
            self,
            [
                Robot("R-1", (1, 0), [(1, 1)]),
                Robot("R-2", [1, 0]),
            ],
            "R-1",
            "R-2",
        )


class AtomicRejectionTests(unittest.TestCase):
    def test_later_bad_position_leaves_earlier_robot_and_task_untouched(self) -> None:
        task = Task("T-1", (0, 1), (0, 2), assigned_robot="R-1")
        first = Robot("R-1", [0, 0], [(0, 1)], task_id="T-1")
        second = Robot("R-2", (1.5, 0))
        with self.assertRaises(ValueError):
            make_sim([first, second], [task])
        self.assertEqual(first.position, [0, 0])
        self.assertIsInstance(first.position, list)
        self.assertEqual(first.route, [(0, 1)])
        self.assertEqual(first.task_id, "T-1")
        self.assertEqual(task.assigned_robot, "R-1")
        self.assertEqual(second.position, (1.5, 0))

    def test_later_bad_route_leaves_earlier_position_and_route_untouched(self) -> None:
        first = Robot("R-1", [0, 0], [(0, 1)])
        second = Robot("R-2", [2, 2], [(5, 5)])
        with self.assertRaises(ValueError):
            make_sim([first, second], grid=GridMap(8, 8))
        self.assertEqual(first.position, [0, 0])
        self.assertEqual(first.route, [(0, 1)])
        self.assertEqual(second.position, [2, 2])
        self.assertEqual(second.route, [(5, 5)])

    def test_later_ownership_failure_leaves_positions_uncommitted(self) -> None:
        first = Robot("R-1", [0, 0], task_id="T-missing")
        second = Robot("R-2", (1, 0))
        with self.assertRaises(ValueError):
            make_sim([first, second])
        self.assertEqual(first.position, [0, 0])
        self.assertEqual(second.position, (1, 0))

    def test_failed_construction_creates_no_partial_state_on_objects(self) -> None:
        first = Robot("R-1", [0, 0])
        with self.assertRaises(ValueError):
            make_sim([first, Robot("R-2", (0, 0))])
        # The collision must be reported without normalizing the first robot
        # in place.
        self.assertEqual(first.position, [0, 0])


class CreationSideEffectTests(unittest.TestCase):
    def test_creation_does_not_advance_or_record_anything(self) -> None:
        robot = Robot("R-1", [0, 0], [(0, 1)])
        sim = make_sim([robot], grid=GridMap(6, 4))
        self.assertEqual(sim.tick, 0)
        self.assertEqual(sim.replay, [])
        self.assertEqual(sim.map_change_history(), [])
        self.assertEqual(robot.distance_travelled, 0)
        self.assertEqual(robot.position, (0, 0))
        self.assertEqual(robot.route, [(0, 1)])

    def test_integer_start_round_trips_through_checkpoint(self) -> None:
        sim = make_sim(
            [Robot("R-1", [0, 0], [(0, 1)]), Robot("R-2", (5, 0))],
            grid=GridMap(6, 4),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "fleet.json")
            sim.save_checkpoint(path)
            restored = FleetSimulator.load_checkpoint(path)
        self.assertEqual(
            restored.robots["R-1"].position, sim.robots["R-1"].position
        )
        self.assertEqual(
            restored.robots["R-2"].position, (5, 0)
        )
        self.assertEqual(restored.tick, 0)
        self.assertEqual(
            restored.robots["R-1"].route, sim.robots["R-1"].route
        )


if __name__ == "__main__":
    unittest.main()
