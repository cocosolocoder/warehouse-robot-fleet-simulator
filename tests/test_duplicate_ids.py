"""Duplicate id rejection when a fleet is constructed directly.

Every robot id may appear exactly once in the robot list and every task id
exactly once in the task list. A repeated id rejects the *whole* request with a
:class:`ValueError` naming the kind of record and the id -- nothing is kept,
merged or renamed -- and the caller's objects are left exactly as passed in.
The two lists have independent id spaces: a robot and a task may share one
string, and a task's ``assigned_robot`` is only a reference, never a second
robot record.

These tests observe construction only: a duplicate must fail immediately
rather than hiding behind later simulation.
"""

import copy
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


def expect_rejected(testcase: unittest.TestCase, robots, tasks, *fragments) -> str:
    with testcase.assertRaises(ValueError) as context:
        FleetSimulator(GridMap(8, 2), list(robots), list(tasks))
    message = str(context.exception)
    for fragment in fragments:
        testcase.assertIn(fragment, message)
    return message


class DuplicateIdRejectionTests(unittest.TestCase):
    def test_same_robot_id_different_positions_rejected(self) -> None:
        expect_rejected(
            self,
            [Robot("R-1", (0, 0)), Robot("R-1", (5, 1))],
            [],
            "robot",
            "R-1",
        )

    def test_same_task_id_different_points_and_state_rejected(self) -> None:
        # One waiting task and one finished task under the same id still clash.
        expect_rejected(
            self,
            [Robot("R-hist", (0, 0))],
            [
                Task("T-1", (1, 0), (2, 0)),
                Task(
                    "T-1",
                    (3, 0),
                    (4, 0),
                    assigned_robot="R-hist",
                    picked_up=True,
                    completed=True,
                ),
            ],
            "task",
            "T-1",
        )

    def test_byte_identical_repeated_records_rejected(self) -> None:
        expect_rejected(
            self,
            [Robot("R-1", (0, 0)), Robot("R-1", (0, 0))],
            [],
            "duplicate",
            "R-1",
        )
        expect_rejected(
            self,
            [Robot("R-2", (0, 0))],
            [
                Task("T-9", (1, 0), (2, 0)),
                Task("T-9", (1, 0), (2, 0)),
            ],
            "duplicate",
            "T-9",
        )

    def test_duplicate_rejected_regardless_of_robot_state(self) -> None:
        # Both idle.
        expect_rejected(
            self,
            [Robot("R-1", (0, 0)), Robot("R-1", (1, 1))],
            [],
            "R-1",
        )
        # One carrying a preset route.
        expect_rejected(
            self,
            [
                Robot("R-2", (0, 0), route=[(1, 0), (2, 0)]),
                Robot("R-2", (5, 0)),
            ],
            [],
            "R-2",
        )
        # One mid-delivery.
        expect_rejected(
            self,
            [
                Robot("R-3", (0, 0), route=[(1, 0)], task_id="T-1"),
                Robot("R-3", (6, 0)),
            ],
            [Task("T-1", (1, 0), (2, 0), assigned_robot="R-3", picked_up=True)],
            "R-3",
        )

    def test_duplicate_at_end_leaves_earlier_objects_untouched(self) -> None:
        robots = [
            Robot("R-1", [0, 0], route=[[1, 0], [2, 0]]),
            Robot("R-2", [5, 0]),
            Robot("R-1", [6, 0]),  # duplicate of an already checked robot
        ]
        tasks = [Task("T-1", [1, 0], [2, 0])]
        robots_before = copy.deepcopy(robots)
        tasks_before = copy.deepcopy(tasks)
        with self.assertRaises(ValueError):
            FleetSimulator(GridMap(8, 1), robots, tasks)
        # Positions and routes keep their original containers and contents --
        # normalization commits nothing when the batch is rejected.
        self.assertEqual(robots, robots_before)
        self.assertEqual(tasks, tasks_before)
        self.assertIsInstance(robots[0].position, list)
        self.assertIsInstance(robots[0].route, list)
        self.assertEqual(robots[0].route, [[1, 0], [2, 0]])
        self.assertIsInstance(tasks[0].pickup, list)

    def test_duplicate_task_leaves_all_objects_untouched(self) -> None:
        robots = [Robot("R-1", [0, 0], route=[[1, 0]])]
        tasks = [
            Task("T-1", [1, 0], [2, 0], assigned_robot="R-1", picked_up=True),
            Task("T-2", [3, 0], [4, 0]),
            Task("T-1", [5, 0], [6, 0]),
        ]
        robots_before = copy.deepcopy(robots)
        tasks_before = copy.deepcopy(tasks)
        with self.assertRaises(ValueError):
            FleetSimulator(GridMap(8, 1), robots, tasks)
        self.assertEqual(robots, robots_before)
        self.assertEqual(tasks, tasks_before)

    def test_neither_record_survives_rejection(self) -> None:
        robots = [Robot("R-1", (0, 0)), Robot("R-1", (1, 0))]
        try:
            FleetSimulator(GridMap(8, 1), robots, [])
        except ValueError:
            pass
        else:
            self.fail("duplicate robot ids were accepted")
        # The caller's own list is never trimmed either.
        self.assertEqual(len(robots), 2)


class IndependentIdSpaceTests(unittest.TestCase):
    def test_robot_and_task_may_share_one_id(self) -> None:
        simulator = FleetSimulator(
            GridMap(4, 1),
            [Robot("S", (0, 0))],
            [Task("S", (1, 0), (3, 0))],
        )
        self.assertIn("S", simulator.robots)
        self.assertIn("S", simulator.tasks)

    def test_task_owner_is_not_a_second_robot_record(self) -> None:
        # One active task mutually bound to R-1 and one finished task R-1
        # historically completed: both name R-1, but only one robot record
        # exists and neither reference counts as a duplicate robot.
        simulator = FleetSimulator(
            GridMap(6, 1),
            [Robot("R-1", (0, 0), route=[(1, 0), (2, 0)], task_id="T-1")],
            [
                Task("T-1", (1, 0), (2, 0), assigned_robot="R-1", picked_up=True),
                Task(
                    "T-2", (3, 0), (4, 0),
                    assigned_robot="R-1", picked_up=True, completed=True,
                ),
            ],
        )
        self.assertEqual(len(simulator.robots), 1)
        self.assertEqual(len(simulator.tasks), 2)

    def test_several_completed_tasks_may_name_same_missing_robot(self) -> None:
        simulator = FleetSimulator(
            GridMap(6, 1),
            [Robot("R-here", (0, 0))],
            [
                Task(
                    "T-a", (1, 0), (2, 0),
                    assigned_robot="R-gone", picked_up=True, completed=True,
                ),
                Task(
                    "T-b", (3, 0), (4, 0),
                    assigned_robot="R-gone", picked_up=True, completed=True,
                ),
            ],
        )
        self.assertEqual(simulator.metrics()["tasks_completed"], 2)
        self.assertNotIn("R-gone", simulator.robots)

    def test_distinct_tasks_with_identical_points_are_both_kept(self) -> None:
        simulator = FleetSimulator(
            GridMap(8, 1),
            [Robot("R-1", (0, 0)), Robot("R-2", (7, 0))],
            [Task("T-1", (3, 0), (4, 0)), Task("T-2", (3, 0), (4, 0))],
        )
        self.assertEqual(set(simulator.tasks), {"T-1", "T-2"})
        simulator.assign_tasks()
        owners = {simulator.tasks["T-1"].assigned_robot,
                  simulator.tasks["T-2"].assigned_robot}
        self.assertEqual(owners, {"R-1", "R-2"})
        for _ in range(8):
            simulator.step()
        self.assertEqual(simulator.metrics()["tasks_completed"], 2)

    def test_valid_unique_batch_keeps_existing_takeover_behavior(self) -> None:
        carrier = Robot(
            "R-1", (0, 0), route=[(1, 0), (2, 0), (3, 0)],
            task_id="T-1", distance_travelled=4,
        )
        history = Task(
            "T-0", (5, 0), (6, 0),
            assigned_robot="R-old", picked_up=True, completed=True,
        )
        in_flight = Task(
            "T-1", (1, 0), (3, 0), assigned_robot="R-1", picked_up=True
        )
        simulator = FleetSimulator(GridMap(8, 1), [carrier], [in_flight, history])
        self.assertEqual(simulator.robots["R-1"].task_id, "T-1")
        self.assertEqual(simulator.robots["R-1"].distance_travelled, 4)
        self.assertTrue(simulator.tasks["T-0"].completed)
        for _ in range(3):
            simulator.step()
        self.assertTrue(simulator.tasks["T-1"].completed)
        self.assertEqual(simulator.metrics()["tasks_completed"], 2)


if __name__ == "__main__":
    unittest.main()
