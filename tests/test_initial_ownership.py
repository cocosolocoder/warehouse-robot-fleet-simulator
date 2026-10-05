"""Initial task-ownership validation for directly constructed simulators.

Robots and tasks can be handed to ``FleetSimulator`` mid-work: a robot carries a
remaining ``route`` and a ``task_id``, and a task can already be assigned or
even loaded. Such bindings must describe one coherent situation *at creation
time* -- impossible ownership must raise ``ValueError`` immediately instead of
failing later while stepping. Work still in progress requires a mutual binding;
finished tasks only keep historical ownership; unassigned tasks simply wait.

These tests observe only construction (and, for reachability, a single
``assign_tasks`` call): no tick is ever executed, so a rejection can never hide
behind later simulation.
"""

import copy
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


def sealed_grid() -> GridMap:
    # A ring of obstacles seals the 3x3 interior (centre (2, 2)) off from
    # every outside cell.
    return GridMap(
        5,
        5,
        frozenset(
            {
                (1, 1), (2, 1), (3, 1),
                (1, 2), (3, 2),
                (1, 3), (2, 3), (3, 3),
            }
        ),
    )


def expect_rejected(testcase: unittest.TestCase, robots, tasks, *fragments) -> str:
    with testcase.assertRaises(ValueError) as context:
        FleetSimulator(GridMap(8, 2), list(robots), list(tasks))
    message = str(context.exception)
    for fragment in fragments:
        testcase.assertIn(fragment, message)
    return message


class BindingRejectionTests(unittest.TestCase):
    def test_robot_executing_unknown_task_rejected(self) -> None:
        expect_rejected(
            self,
            [Robot("R-1", (0, 0), task_id="T-missing")],
            [Task("T-other", (1, 0), (2, 0))],
            "R-1",
            "T-missing",
            "unknown task",
        )

    def test_robot_bound_to_completed_task_rejected(self) -> None:
        expect_rejected(
            self,
            [Robot("R-1", (0, 0), task_id="T-done")],
            [
                Task(
                    "T-done",
                    (0, 0),
                    (2, 0),
                    assigned_robot="R-1",
                    picked_up=True,
                    completed=True,
                )
            ],
            "R-1",
            "T-done",
            "already completed",
        )

    def test_robot_executing_task_assigned_to_another_robot_rejected(self) -> None:
        expect_rejected(
            self,
            [
                Robot("R-1", (0, 0), task_id="T-1"),
                Robot("R-2", (4, 0), task_id="T-2"),
            ],
            [
                Task("T-1", (1, 0), (2, 0), assigned_robot="R-2"),
                Task("T-2", (3, 0), (4, 0), assigned_robot="R-1"),
            ],
            "R-1",
            "T-1",
            "R-2",
        )

    def test_task_assigned_to_unknown_robot_rejected(self) -> None:
        expect_rejected(
            self,
            [Robot("R-1", (0, 0))],
            [Task("T-1", (1, 0), (2, 0), assigned_robot="R-ghost")],
            "T-1",
            "R-ghost",
            "unknown robot",
        )

    def test_task_claiming_idle_robot_rejected(self) -> None:
        expect_rejected(
            self,
            [Robot("R-1", (0, 0))],
            [Task("T-1", (1, 0), (2, 0), assigned_robot="R-1")],
            "T-1",
            "R-1",
        )

    def test_two_unfinished_tasks_claiming_one_robot_rejected(self) -> None:
        # R-1 executes T-1; T-2 also claims R-1 while no robot executes it.
        expect_rejected(
            self,
            [
                Robot("R-1", (0, 0), route=[(1, 0)], task_id="T-1"),
                Robot("R-2", (4, 0)),
            ],
            [
                Task("T-1", (1, 0), (2, 0), assigned_robot="R-1", picked_up=True),
                Task("T-2", (3, 0), (4, 0), assigned_robot="R-1"),
            ],
            "T-2",
            "R-1",
            "T-1",
        )

    def test_crossed_bindings_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FleetSimulator(
                GridMap(8, 1),
                [
                    Robot("R-1", (0, 0), task_id="T-1"),
                    Robot("R-2", (4, 0), task_id="T-2"),
                ],
                [
                    Task("T-1", (1, 0), (2, 0), assigned_robot="R-2"),
                    Task("T-2", (3, 0), (4, 0), assigned_robot="R-1"),
                ],
            )

    def test_picked_up_task_without_executing_robot_rejected(self) -> None:
        expect_rejected(
            self,
            [Robot("R-1", (0, 0)), Robot("R-2", (4, 0))],
            [Task("T-1", (1, 0), (2, 0), picked_up=True, assigned_robot=None)],
            "T-1",
            "no assigned robot",
        )

    def test_picked_up_task_named_by_unbound_robot_rejected(self) -> None:
        # The robot claims the task, but the task does not claim the robot
        # back; the goods' ownership is contradictory and must not be repaired.
        expect_rejected(
            self,
            [Robot("R-1", (0, 0), task_id="T-1")],
            [Task("T-1", (1, 0), (2, 0), picked_up=True, assigned_robot=None)],
            "R-1",
            "T-1",
        )

    def test_conflict_is_rejected_even_when_earlier_records_are_valid(self) -> None:
        robots = [
            Robot("R-1", (0, 0), route=[(1, 0), (2, 0)], task_id="T-1"),
            Robot("R-2", (6, 0)),
        ]
        tasks = [
            Task("T-1", (1, 0), (2, 0), assigned_robot="R-1", picked_up=True),
            Task("T-2", (4, 0), (5, 0), assigned_robot="R-1"),
        ]
        robots_before = copy.deepcopy(robots)
        tasks_before = copy.deepcopy(tasks)
        with self.assertRaises(ValueError):
            FleetSimulator(GridMap(8, 1), robots, tasks)
        # Rejection leaves the caller's objects exactly as supplied; nothing is
        # rewritten, unassigned or dropped to make the input fit.
        self.assertEqual(robots, robots_before)
        self.assertEqual(tasks, tasks_before)


class LegalStateAcceptanceTests(unittest.TestCase):
    def test_bound_in_progress_task_keeps_assignment_pickup_and_route(self) -> None:
        route = [(1, 0), (2, 0), (3, 0)]
        simulator = FleetSimulator(
            GridMap(6, 1),
            [Robot("R-1", (0, 0), route=list(route), task_id="T-1")],
            [Task("T-1", (1, 0), (3, 0), assigned_robot="R-1", picked_up=False)],
        )
        robot = simulator.robots["R-1"]
        task = simulator.tasks["T-1"]
        self.assertEqual(robot.task_id, "T-1")
        self.assertEqual(task.assigned_robot, "R-1")
        self.assertFalse(task.picked_up)
        self.assertFalse(task.completed)
        self.assertEqual(robot.route, route)
        self.assertEqual(robot.position, (0, 0))
        # Creation advances nothing and records no history.
        self.assertEqual(simulator.tick, 0)
        self.assertEqual(simulator.replay, [])
        self.assertEqual(robot.distance_travelled, 0)

    def test_completed_task_history_with_robot_on_new_task_accepted(self) -> None:
        # T0 was finished by R; R now executes T1. T0 keeps its historical
        # owner and must not look like a second occupation of R.
        simulator = FleetSimulator(
            GridMap(8, 1),
            [Robot("R", (0, 0), route=[(1, 0), (2, 0)], task_id="T1")],
            [
                Task(
                    "T0",
                    (4, 0),
                    (5, 0),
                    assigned_robot="R",
                    picked_up=True,
                    completed=True,
                ),
                Task("T1", (1, 0), (2, 0), assigned_robot="R", picked_up=True),
            ],
        )
        self.assertEqual(simulator.robots["R"].task_id, "T1")
        self.assertTrue(simulator.tasks["T0"].completed)
        self.assertEqual(simulator.tasks["T0"].assigned_robot, "R")
        self.assertFalse(simulator.tasks["T1"].completed)
        self.assertEqual(simulator.tasks["T1"].assigned_robot, "R")

    def test_completed_task_history_with_idle_robot_accepted(self) -> None:
        simulator = FleetSimulator(
            GridMap(6, 1),
            [Robot("R", (5, 0))],
            [
                Task(
                    "T0",
                    (1, 0),
                    (5, 0),
                    assigned_robot="R",
                    picked_up=True,
                    completed=True,
                )
            ],
        )
        self.assertTrue(simulator.robots["R"].idle)
        self.assertEqual(simulator.robots["R"].route, [])
        self.assertTrue(simulator.tasks["T0"].completed)

    def test_completed_task_keeps_history_of_robot_no_longer_in_fleet(self) -> None:
        # Historical ownership names a robot absent from the current fleet;
        # finished tasks are never required to re-occupy anyone.
        simulator = FleetSimulator(
            GridMap(6, 1),
            [Robot("R-2", (0, 0))],
            [
                Task(
                    "T0",
                    (1, 0),
                    (5, 0),
                    assigned_robot="R-retired",
                    picked_up=True,
                    completed=True,
                )
            ],
        )
        self.assertTrue(simulator.tasks["T0"].completed)
        self.assertEqual(simulator.tasks["T0"].assigned_robot, "R-retired")
        self.assertTrue(simulator.robots["R-2"].idle)

    def test_unassigned_uncollected_task_and_idle_robots_accepted(self) -> None:
        simulator = FleetSimulator(
            GridMap(6, 1),
            [Robot("R-1", (0, 0)), Robot("R-2", (5, 0))],
            [Task("T-1", (1, 0), (2, 0))],
        )
        self.assertIsNone(simulator.tasks["T-1"].assigned_robot)
        self.assertFalse(simulator.tasks["T-1"].picked_up)
        self.assertTrue(all(robot.idle for robot in simulator.robots.values()))

    def test_task_nobody_can_reach_waits_without_ownership_error(self) -> None:
        # The pickup lies in a sealed component; unassigned and uncollected,
        # the task is unreachable but not wrongly owned, so it simply waits.
        simulator = FleetSimulator(
            sealed_grid(),
            [Robot("R-1", (0, 0)), Robot("R-2", (4, 4))],
            [Task("T-sealed", (2, 2), (4, 0))],
        )
        simulator.assign_tasks()
        self.assertIsNone(simulator.tasks["T-sealed"].assigned_robot)
        self.assertFalse(simulator.tasks["T-sealed"].picked_up)
        self.assertNotIn("T-sealed", simulator.paused_tasks)
        self.assertTrue(all(robot.idle for robot in simulator.robots.values()))

    def test_mixture_of_history_active_and_waiting_tasks_accepted(self) -> None:
        simulator = FleetSimulator(
            GridMap(8, 1),
            [
                Robot("R-1", (0, 0), route=[(1, 0)], task_id="T-1"),
                Robot("R-2", (7, 0)),
            ],
            [
                Task(
                    "T0",
                    (5, 0),
                    (6, 0),
                    assigned_robot="R-1",
                    picked_up=True,
                    completed=True,
                ),
                Task("T-1", (1, 0), (3, 0), assigned_robot="R-1", picked_up=True),
                Task("T-wait", (6, 0), (7, 0)),
            ],
        )
        self.assertEqual(simulator.robots["R-1"].task_id, "T-1")
        self.assertTrue(simulator.robots["R-2"].idle)
        self.assertIsNone(simulator.tasks["T-wait"].assigned_robot)
        self.assertEqual(simulator.tick, 0)


if __name__ == "__main__":
    unittest.main()
