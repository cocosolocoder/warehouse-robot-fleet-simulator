"""Duplicate-identifier rejection for directly constructed fleets.

Each robot id may name at most one robot in the robot list handed to
``FleetSimulator`` and each task id at most one task in the task list; the two
lists keep independent namespaces. The identifier alone decides, so identical
records passed twice, robots differing only in position/route/idle state, and
tasks differing in pickup/dropoff, owner or completion (one finished beside
one still waiting) are all rejected as a whole with :class:`ValueError` --
nothing is kept, merged or renamed. These tests also pin the read-only
guarantee: even a duplicate at the end of a list leaves every earlier object
holding its original list-shaped coordinates/routes, bindings, mileage and
flags. Distinct-id tasks with equal points stay distinct and both take part
in assignment and completion statistics.
"""

import copy
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


def make_sim(robots=(), tasks=()):
    return FleetSimulator(GridMap(8, 1), list(robots), list(tasks))


def expect_rejected(testcase, robots, tasks, *fragments) -> str:
    with testcase.assertRaises(ValueError) as context:
        make_sim(robots, tasks)
    message = str(context.exception)
    for fragment in fragments:
        testcase.assertIn(fragment, message)
    return message


class DuplicateRobotTests(unittest.TestCase):
    def test_same_id_different_positions_rejected(self) -> None:
        # The exact bug: the second record used to silently replace the first,
        # losing the robot at the earlier position.
        expect_rejected(
            self,
            [Robot("R-1", (0, 0)), Robot("R-1", (5, 0))],
            [],
            "duplicate robot",
            "R-1",
        )

    def test_same_id_on_same_cell_still_rejected(self) -> None:
        # Even sharing the position does not make two records one entity.
        expect_rejected(
            self,
            [Robot("R-1", (0, 0)), Robot("R-1", (0, 0))],
            [],
            "duplicate robot",
        )

    def test_identical_object_passed_twice_rejected(self) -> None:
        robot = Robot("R-1", (0, 0))
        expect_rejected(self, [robot, robot], [], "duplicate robot", "R-1")

    def test_duplicate_regardless_of_state(self) -> None:
        # Idle, preset-route and in-flight robots all clash on the id alone.
        variants = [
            Robot("R-1", (5, 0)),
            Robot("R-1", (5, 0), route=[(6, 0)]),
            Robot("R-1", (1, 0), route=[(2, 0)], task_id="T-9"),
        ]
        tasks = [Task("T-9", (2, 0), (3, 0), assigned_robot="R-1")]
        for other in variants:
            expect_rejected(
                self,
                [Robot("R-1", (0, 0)), other],
                tasks if other.task_id is not None else [],
                "duplicate robot",
            )

    def test_three_records_same_id_rejected(self) -> None:
        expect_rejected(
            self,
            [Robot("R-1", (0, 0)), Robot("R-1", (2, 0)), Robot("R-1", (4, 0))],
            [],
            "duplicate robot",
            "R-1",
        )

    def test_message_names_first_conflicting_positions(self) -> None:
        message = expect_rejected(
            self,
            [Robot("R-1", (0, 0)), Robot("R-2", (1, 0)), Robot("R-1", (2, 0))],
            [],
            "duplicate robot",
            "R-1",
        )
        self.assertIn("0", message)
        self.assertIn("2", message)


class DuplicateTaskTests(unittest.TestCase):
    def test_same_id_different_points_rejected(self) -> None:
        expect_rejected(
            self,
            [],
            [Task("T-1", (1, 0), (2, 0)), Task("T-1", (4, 0), (5, 0))],
            "duplicate task",
            "T-1",
        )

    def test_identical_task_passed_twice_rejected(self) -> None:
        task = Task("T-1", (1, 0), (2, 0))
        expect_rejected(self, [], [task, copy.deepcopy(task)], "duplicate task")

    def test_one_completed_beside_one_waiting_rejected(self) -> None:
        # Different completion state does not turn one id into two tasks.
        expect_rejected(
            self,
            [Robot("R-1", (0, 0))],
            [
                Task("T-1", (1, 0), (2, 0)),
                Task(
                    "T-1",
                    (3, 0),
                    (4, 0),
                    assigned_robot="R-1",
                    picked_up=True,
                    completed=True,
                ),
            ],
            "duplicate task",
            "T-1",
        )

    def test_same_points_but_duplicate_id_still_rejected(self) -> None:
        expect_rejected(
            self,
            [],
            [Task("T-1", (1, 0), (2, 0)), Task("T-1", (1, 0), (2, 0))],
            "duplicate task",
        )


class CallerObjectUntouchedTests(unittest.TestCase):
    def test_duplicate_robot_at_end_leaves_earlier_objects_as_supplied(self) -> None:
        robots = [
            Robot("R-1", [0, 0], route=[[1, 0]], task_id="T-1"),
            Robot("R-2", [7, 0]),
            Robot("R-1", [6, 0]),
        ]
        tasks = [Task("T-1", [1, 0], [3, 0], assigned_robot="R-1")]
        robots_before = copy.deepcopy(robots)
        tasks_before = copy.deepcopy(tasks)
        with self.assertRaises(ValueError):
            make_sim(robots, tasks)
        self.assertEqual(robots, robots_before)
        self.assertEqual(tasks, tasks_before)
        # The normalization copies never replaced the caller's list containers.
        self.assertIsInstance(robots[0].position, list)
        self.assertIsInstance(robots[0].route, list)
        self.assertIsInstance(robots[0].route[0], list)
        self.assertIsInstance(tasks[0].pickup, list)
        self.assertIsInstance(tasks[0].dropoff, list)
        self.assertEqual(robots[0].distance_travelled, 0)

    def test_duplicate_task_at_end_leaves_earlier_objects_as_supplied(self) -> None:
        robots = [Robot("R-1", [0, 0])]
        tasks = [
            Task("T-1", [1, 0], [2, 0]),
            Task("T-2", [3, 0], [4, 0]),
            Task("T-1", [5, 0], [6, 0]),
        ]
        robots_before = copy.deepcopy(robots)
        tasks_before = copy.deepcopy(tasks)
        with self.assertRaises(ValueError):
            make_sim(robots, tasks)
        self.assertEqual(robots, robots_before)
        self.assertEqual(tasks, tasks_before)
        for task in tasks:
            self.assertIsInstance(task.pickup, list)
            self.assertIsInstance(task.dropoff, list)


class IndependentNamespacesTests(unittest.TestCase):
    def test_robot_and_task_may_share_one_id(self) -> None:
        simulator = make_sim(
            [Robot("shared", (0, 0))],
            [Task("shared", (1, 0), (2, 0))],
        )
        self.assertIn("shared", simulator.robots)
        self.assertIn("shared", simulator.tasks)

    def test_distinct_id_tasks_with_equal_points_both_kept(self) -> None:
        simulator = make_sim(
            [Robot("R-1", (0, 0))],
            [Task("A", (1, 0), (2, 0)), Task("B", (1, 0), (2, 0))],
        )
        self.assertEqual(set(simulator.tasks), {"A", "B"})
        self.assertEqual(simulator.metrics()["tasks_total"], 2)
        # Both tasks are independently assigned and completed.
        for _ in range(12):
            simulator.step()
        self.assertTrue(simulator.tasks["A"].completed)
        self.assertTrue(simulator.tasks["B"].completed)
        self.assertEqual(simulator.metrics()["tasks_completed"], 2)

    def test_completed_tasks_share_one_historical_robot(self) -> None:
        # Several finished records may name the same robot, including one that
        # is not in the fleet; neither reference is a robot-list entry.
        simulator = make_sim(
            [Robot("R-1", (0, 0))],
            [
                Task(
                    "C1", (1, 0), (2, 0),
                    assigned_robot="gone", picked_up=True, completed=True,
                ),
                Task(
                    "C2", (3, 0), (4, 0),
                    assigned_robot="gone", picked_up=True, completed=True,
                ),
            ],
        )
        self.assertEqual(simulator.metrics()["tasks_completed"], 2)
        self.assertEqual(set(simulator.robots), {"R-1"})

    def test_valid_batch_keeps_existing_takeover_and_history_behavior(self) -> None:
        # No duplicate ids: an in-flight takeover and finished history keep
        # their bindings and completion counts exactly as before.
        simulator = make_sim(
            [
                Robot("R-1", (1, 0), route=[(2, 0), (3, 0)], task_id="T-1"),
                Robot("R-2", (7, 0)),
            ],
            [
                Task(
                    "T0", (5, 0), (6, 0),
                    assigned_robot="R-retired", picked_up=True, completed=True,
                ),
                Task("T-1", (1, 0), (3, 0), assigned_robot="R-1", picked_up=True),
            ],
        )
        self.assertEqual(simulator.robots["R-1"].task_id, "T-1")
        self.assertEqual(simulator.robots["R-1"].route, [(2, 0), (3, 0)])
        self.assertTrue(simulator.tasks["T0"].completed)
        self.assertEqual(simulator.metrics()["tasks_completed"], 1)


if __name__ == "__main__":
    unittest.main()
