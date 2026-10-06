"""Regression tests for tasks whose pickup point equals their dropoff point.

Such a task is still a real pickup-and-delivery task: the robot has to reach
the shared cell, and only then are pickup and delivery confirmed together. The
checkpoint tests already cover the empty route through saved-state loading;
these tests pin the behaviour down through the normal public flow -- direct
construction, ``assign_tasks`` and step-by-step ``step()`` execution -- and
cross-check task state, robot position and mileage, the completion statistics
and the replay frames against each other.
"""

import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


def tick_frames(simulator: FleetSimulator) -> list[dict[str, object]]:
    return [frame for frame in simulator.replay if frame.get("type") == "tick"]


def assert_no_collisions(testcase: unittest.TestCase, simulator: FleetSimulator) -> None:
    for frame in tick_frames(simulator):
        positions = [tuple(cell) for cell in frame["robots"].values()]
        testcase.assertEqual(len(positions), len(set(positions)))
        for position in positions:
            testcase.assertTrue(simulator.grid.traversable(position))


class AssignedWhileStandingOnTheCellTests(unittest.TestCase):
    def build(self) -> FleetSimulator:
        return FleetSimulator(
            GridMap(5, 1),
            [Robot("R", (2, 0))],
            [Task("T", (2, 0), (2, 0))],
        )

    def test_assignment_binds_both_sides_but_completes_nothing(self) -> None:
        simulator = self.build()
        simulator.assign_tasks()

        task = simulator.tasks["T"]
        robot = simulator.robots["R"]
        # The assignment establishes the mutual ownership binding ...
        self.assertEqual(task.assigned_robot, "R")
        self.assertEqual(robot.task_id, "T")
        # ... leaves a feasible zero-step remaining route ...
        self.assertEqual(robot.route, [])
        # ... but never collects, delivers, drives or records time by itself.
        self.assertFalse(task.picked_up)
        self.assertFalse(task.completed)
        self.assertEqual(robot.position, (2, 0))
        self.assertEqual(robot.distance_travelled, 0)
        self.assertEqual(simulator.tick, 0)
        self.assertEqual(simulator.replay, [])
        self.assertEqual(simulator.map_change_history(), [])
        self.assertEqual(simulator.paused_tasks, set())
        self.assertEqual(simulator.status()["traffic_waits"], [])
        self.assertEqual(simulator.metrics()["tasks_completed"], 0)
        self.assertEqual(simulator.metrics()["distance_total"], 0)

    def test_one_step_completes_pickup_and_delivery_in_place(self) -> None:
        simulator = self.build()
        simulator.assign_tasks()
        event = simulator.step()

        task = simulator.tasks["T"]
        robot = simulator.robots["R"]
        # Both confirmations happen on this one tick ...
        self.assertTrue(task.picked_up)
        self.assertTrue(task.completed)
        # ... the robot is released, while the task keeps its historical owner.
        self.assertIsNone(robot.task_id)
        self.assertEqual(task.assigned_robot, "R")
        # ... nothing moved: same cell, no mileage, not on the moved list.
        self.assertEqual(robot.position, (2, 0))
        self.assertEqual(robot.distance_travelled, 0)
        self.assertEqual(event["moved"], [])
        self.assertEqual(event["robots"], {"R": [2, 0]})
        self.assertEqual(event["completed"], ["T"])
        # Statistics and replay agree with the in-place completion.
        self.assertEqual(simulator.tick, 1)
        self.assertEqual(len(tick_frames(simulator)), 1)
        metrics = simulator.metrics()
        self.assertEqual(metrics["tasks_completed"], 1)
        self.assertEqual(metrics["tasks_total"], 1)
        self.assertEqual(metrics["completion_ratio"], 1.0)
        self.assertEqual(metrics["distance_total"], 0)
        self.assertEqual(metrics["tasks_paused"], [])
        self.assertEqual(metrics["traffic_waits"], [])
        self.assertEqual(tick_frames(simulator)[0]["completed"], ["T"])
        assert_no_collisions(self, simulator)

    def test_further_steps_record_only_empty_wait_frames(self) -> None:
        simulator = self.build()
        simulator.assign_tasks()
        simulator.step()
        event = simulator.step()

        # The completion is not redone or undone: the robot stands, no mileage
        # is added and the completed task stays in the cumulative replay list.
        self.assertEqual(event["moved"], [])
        self.assertEqual(event["completed"], ["T"])
        self.assertEqual(simulator.robots["R"].position, (2, 0))
        self.assertEqual(simulator.robots["R"].distance_travelled, 0)
        self.assertEqual(simulator.metrics()["tasks_completed"], 1)
        self.assertEqual(simulator.metrics()["distance_total"], 0)
        self.assertTrue(simulator.tasks["T"].completed)

    def test_plain_step_without_explicit_assignment_behaves_the_same(self) -> None:
        # Normal usage never has to call assign_tasks by hand: step() assigns
        # first and still completes only after advancing a tick.
        simulator = self.build()
        event = simulator.step()

        self.assertTrue(simulator.tasks["T"].completed)
        self.assertEqual(simulator.tasks["T"].assigned_robot, "R")
        self.assertIsNone(simulator.robots["R"].task_id)
        self.assertEqual(simulator.robots["R"].position, (2, 0))
        self.assertEqual(simulator.robots["R"].distance_travelled, 0)
        self.assertEqual(event["moved"], [])
        self.assertEqual(event["completed"], ["T"])


class ApproachingTheSharedCellTests(unittest.TestCase):
    def test_arrival_step_confirms_pickup_and_delivery_together(self) -> None:
        simulator = FleetSimulator(
            GridMap(6, 1),
            [Robot("R", (0, 0))],
            [Task("T", (3, 0), (3, 0))],
        )
        simulator.assign_tasks()
        robot = simulator.robots["R"]
        self.assertEqual(robot.task_id, "T")
        self.assertEqual(robot.route, [(1, 0), (2, 0), (3, 0)])

        # Tick 1: one cell closer, goods not collected just because the
        # pickup and dropoff share coordinates.
        first = simulator.step()
        self.assertEqual(robot.position, (1, 0))
        self.assertEqual(robot.route, [(2, 0), (3, 0)])
        self.assertFalse(simulator.tasks["T"].picked_up)
        self.assertFalse(simulator.tasks["T"].completed)
        self.assertEqual(first["moved"], ["R"])
        self.assertEqual(first["completed"], [])
        self.assertEqual(robot.distance_travelled, 1)

        # Tick 2: passing another ordinary cell changes nothing business-wise.
        simulator.step()
        self.assertEqual(robot.position, (2, 0))
        self.assertFalse(simulator.tasks["T"].picked_up)
        self.assertFalse(simulator.tasks["T"].completed)
        self.assertEqual(tick_frames(simulator)[-1]["completed"], [])

        # Tick 3: the step that actually enters the target cell collects and
        # delivers at once, exhausts the route and frees the robot.
        arrival = simulator.step()
        task = simulator.tasks["T"]
        self.assertTrue(task.picked_up)
        self.assertTrue(task.completed)
        self.assertEqual(robot.position, (3, 0))
        self.assertEqual(robot.route, [])
        self.assertIsNone(robot.task_id)
        self.assertEqual(task.assigned_robot, "R")
        self.assertEqual(arrival["moved"], ["R"])
        self.assertEqual(arrival["completed"], ["T"])
        # Mileage counts only the cells actually driven -- no round trip and
        # no extra waiting tick after arrival.
        self.assertEqual(robot.distance_travelled, 3)
        self.assertEqual(simulator.metrics()["distance_total"], 3)
        self.assertEqual(simulator.metrics()["tasks_completed"], 1)
        self.assertEqual(simulator.tick, 3)
        self.assertEqual(len(tick_frames(simulator)), 3)
        assert_no_collisions(self, simulator)

    def test_no_extra_step_is_needed_once_the_cell_is_reached(self) -> None:
        simulator = FleetSimulator(
            GridMap(6, 1),
            [Robot("R", (0, 0))],
            [Task("T", (3, 0), (3, 0))],
        )
        for _ in range(3):
            simulator.step()
        task = simulator.tasks["T"]
        robot = simulator.robots["R"]
        self.assertTrue(task.completed)
        self.assertEqual(robot.position, (3, 0))
        self.assertEqual(robot.route, [])
        self.assertIsNone(robot.task_id)
        # The robot must not leave the cell, drive back and forth, or wait one
        # more tick to deliver: one further step is an empty frame only.
        idle_event = simulator.step()
        self.assertEqual(idle_event["moved"], [])
        self.assertEqual(robot.position, (3, 0))
        self.assertEqual(robot.distance_travelled, 3)
        self.assertEqual(simulator.metrics()["tasks_completed"], 1)

    def test_detour_completes_on_actual_arrival_with_real_mileage(self) -> None:
        # The direct corridor is blocked; the shortest route loops through the
        # bottom row. Early pickup/completion must not happen en route.
        grid = GridMap(4, 2, frozenset({(2, 0)}))
        simulator = FleetSimulator(
            grid,
            [Robot("R", (0, 0))],
            [Task("T", (3, 0), (3, 0))],
        )
        simulator.assign_tasks()
        self.assertEqual(
            simulator.robots["R"].route,
            [(1, 0), (1, 1), (2, 1), (3, 1), (3, 0)],
        )

        for _ in range(4):
            simulator.step()
            self.assertFalse(simulator.tasks["T"].picked_up)
            self.assertFalse(simulator.tasks["T"].completed)
            self.assertIsNotNone(simulator.robots["R"].task_id)

        arrival = simulator.step()
        self.assertTrue(simulator.tasks["T"].completed)
        self.assertEqual(simulator.robots["R"].position, (3, 0))
        self.assertEqual(arrival["moved"], ["R"])
        self.assertEqual(arrival["completed"], ["T"])
        # Every one of the five actually driven detour cells counts, and only
        # those.
        self.assertEqual(simulator.robots["R"].distance_travelled, 5)
        self.assertEqual(simulator.metrics()["distance_total"], 5)
        self.assertEqual(simulator.metrics()["tasks_completed"], 1)


class BlockedTargetCellTests(unittest.TestCase):
    def build(self) -> FleetSimulator:
        # A straight one-wide corridor: B drives its own taskless preset route
        # onto the shared pickup/dropoff cell and then parks there with no
        # neighbouring cell it could yield onto.
        return FleetSimulator(
            GridMap(5, 1),
            [
                Robot("A", (0, 0)),
                Robot("B", (3, 0), route=[(4, 0)]),
            ],
            [Task("T", (4, 0), (4, 0))],
        )

    def test_robot_waits_behind_parked_blocker_until_the_cell_frees(self) -> None:
        simulator = self.build()
        simulator.assign_tasks()
        task = simulator.tasks["T"]
        carrier = simulator.robots["A"]
        blocker = simulator.robots["B"]

        # The taskless-but-moving robot keeps its preset route and is never
        # hired; the task goes to A with the full shortest route.
        self.assertEqual(task.assigned_robot, "A")
        self.assertEqual(carrier.task_id, "T")
        self.assertEqual(carrier.route, [(1, 0), (2, 0), (3, 0), (4, 0)])
        self.assertIsNone(blocker.task_id)
        self.assertEqual(blocker.route, [(4, 0)])

        # Tick 1: both drive one cell; B ends parked exactly on the target.
        simulator.step()
        self.assertEqual(carrier.position, (1, 0))
        self.assertEqual(blocker.position, (4, 0))
        self.assertEqual(blocker.route, [])
        self.assertFalse(task.completed)
        self.assertEqual(simulator.status()["traffic_waits"], [])

        # Ticks 2 and 3: A closes the remaining distance but does not collect
        # anything while short of the shared cell.
        simulator.step()
        simulator.step()
        self.assertEqual(carrier.position, (3, 0))
        self.assertEqual(carrier.route, [(4, 0)])
        self.assertFalse(task.picked_up)
        self.assertFalse(task.completed)

        # From tick 4 on the target cell is occupied by a parked robot with no
        # safe side cell: A waits, keeps the binding and the unconsumed route,
        # adds no mileage and completes nothing.
        waited_event = simulator.step()
        self.assertEqual(waited_event["moved"], [])
        self.assertEqual(carrier.position, (3, 0))
        self.assertEqual(carrier.route, [(4, 0)])
        self.assertEqual(carrier.task_id, "T")
        self.assertFalse(task.picked_up)
        self.assertFalse(task.completed)
        expected_wait = [{"robot_id": "A", "blocked_by": ["B"], "ticks": 1}]
        self.assertEqual(simulator.status()["traffic_waits"], expected_wait)
        self.assertEqual(simulator.metrics()["traffic_waits"], expected_wait)

        simulator.step()
        self.assertEqual(
            simulator.status()["traffic_waits"],
            [{"robot_id": "A", "blocked_by": ["B"], "ticks": 2}],
        )
        self.assertEqual(carrier.position, (3, 0))
        self.assertEqual(carrier.route, [(4, 0)])
        self.assertFalse(task.completed)

        # This is vehicle blocking, never a map pause: the task stays active,
        # the completion statistics stay at zero and waiting adds no mileage
        # (A drove three cells, B one, and nothing since).
        self.assertEqual(simulator.paused_tasks, set())
        self.assertEqual(simulator.metrics()["tasks_paused"], [])
        self.assertEqual(simulator.metrics()["tasks_completed"], 0)
        self.assertEqual(carrier.distance_travelled, 3)
        self.assertEqual(blocker.distance_travelled, 1)
        self.assertEqual(simulator.metrics()["distance_total"], 4)
        # The yielding robot is not hired by standing in the way.
        self.assertIsNone(blocker.task_id)
        # Every replay frame keeps the completion list empty and collision-free.
        for frame in tick_frames(simulator):
            self.assertEqual(frame["completed"], [])
        assert_no_collisions(self, simulator)


class UnreachableUnassignedCellTests(unittest.TestCase):
    def build(self) -> FleetSimulator:
        # The shared pickup/dropoff cell (3, 0) is sealed in the initial map;
        # an ordinary task on the open cells is available to the only robot.
        return FleetSimulator(
            GridMap(4, 1, frozenset({(3, 0)})),
            [Robot("R", (0, 0))],
            [
                Task("T-sealed", (3, 0), (3, 0)),
                Task("T-other", (1, 0), (2, 0)),
            ],
        )

    def test_unplannable_task_keeps_waiting_and_does_not_occupy_robot(self) -> None:
        simulator = self.build()
        simulator.assign_tasks()

        sealed = simulator.tasks["T-sealed"]
        other = simulator.tasks["T-other"]
        robot = simulator.robots["R"]
        # No feasible route exists, so the task is neither assigned nor treated
        # as an in-place zero-route job; the reachable task gets the robot.
        self.assertIsNone(sealed.assigned_robot)
        self.assertFalse(sealed.picked_up)
        self.assertFalse(sealed.completed)
        self.assertEqual(other.assigned_robot, "R")
        self.assertEqual(robot.task_id, "T-other")
        self.assertEqual(robot.route, [(1, 0), (2, 0)])
        # An unassigned task that merely waits is never reported as paused.
        self.assertEqual(simulator.paused_tasks, set())

        simulator.step()
        simulator.step()
        self.assertTrue(other.completed)
        self.assertEqual(robot.position, (2, 0))
        self.assertIsNone(robot.task_id)
        # Even with the robot now idle right next to the sealed cell, the
        # unplannable task neither binds nor completes "in place".
        idle_frame = simulator.step()
        self.assertIsNone(sealed.assigned_robot)
        self.assertFalse(sealed.picked_up)
        self.assertFalse(sealed.completed)
        self.assertEqual(robot.position, (2, 0))
        self.assertEqual(robot.distance_travelled, 2)
        self.assertEqual(idle_frame["moved"], [])
        self.assertEqual(idle_frame["completed"], ["T-other"])
        self.assertEqual(simulator.paused_tasks, set())
        self.assertEqual(simulator.metrics()["tasks_completed"], 1)

    def test_reopening_the_cell_completes_on_actual_arrival(self) -> None:
        simulator = self.build()
        for _ in range(2):
            simulator.step()
        self.assertTrue(simulator.tasks["T-other"].completed)
        self.assertIsNone(simulator.tasks["T-sealed"].assigned_robot)
        self.assertEqual(simulator.robots["R"].position, (2, 0))

        # Restore access with the existing map-edit call; the edit itself moves
        # nothing and completes nothing.
        record = simulator.modify_obstacles(removed=[(3, 0)])
        self.assertEqual(record["removed"], [[3, 0]])
        self.assertFalse(simulator.tasks["T-sealed"].completed)
        self.assertEqual(simulator.robots["R"].position, (2, 0))
        self.assertEqual(simulator.robots["R"].distance_travelled, 2)

        # The next step assigns the waiting task and finishes it on the step
        # that really enters the reopened pickup/dropoff cell.
        arrival = simulator.step()
        sealed = simulator.tasks["T-sealed"]
        robot = simulator.robots["R"]
        self.assertTrue(sealed.picked_up)
        self.assertTrue(sealed.completed)
        self.assertEqual(sealed.assigned_robot, "R")
        self.assertIsNone(robot.task_id)
        self.assertEqual(robot.position, (3, 0))
        self.assertEqual(robot.route, [])
        self.assertEqual(arrival["moved"], ["R"])
        self.assertEqual(arrival["completed"], ["T-other", "T-sealed"])
        self.assertEqual(robot.distance_travelled, 3)
        metrics = simulator.metrics()
        self.assertEqual(metrics["tasks_completed"], 2)
        self.assertEqual(metrics["distance_total"], 3)
        self.assertEqual(metrics["tasks_paused"], [])
        self.assertEqual(tick_frames(simulator)[-1]["completed"],
                         ["T-other", "T-sealed"])
        assert_no_collisions(self, simulator)


class OrdinaryDistinctPointsTasksTests(unittest.TestCase):
    """Guard: tasks with different pickup and dropoff keep their old behaviour."""

    def test_normal_task_drives_pickup_then_dropoff(self) -> None:
        simulator = FleetSimulator(
            GridMap(4, 1),
            [Robot("R", (0, 0))],
            [Task("T", (1, 0), (3, 0))],
        )
        simulator.assign_tasks()
        robot = simulator.robots["R"]
        self.assertEqual(robot.route, [(1, 0), (2, 0), (3, 0)])
        self.assertFalse(simulator.tasks["T"].completed)

        first = simulator.step()
        self.assertEqual(robot.position, (1, 0))
        self.assertTrue(simulator.tasks["T"].picked_up)
        self.assertFalse(simulator.tasks["T"].completed)
        self.assertEqual(first["completed"], [])

        simulator.step()
        self.assertFalse(simulator.tasks["T"].completed)
        arrival = simulator.step()
        self.assertTrue(simulator.tasks["T"].completed)
        self.assertEqual(robot.position, (3, 0))
        self.assertEqual(arrival["completed"], ["T"])
        self.assertEqual(robot.distance_travelled, 3)
        self.assertEqual(simulator.metrics()["tasks_completed"], 1)

    def test_starting_on_pickup_with_remote_dropoff_still_delivers_only_on_arrival(self) -> None:
        # Standing on the pickup cell alone must not look like the coincident
        # case: the delivery still requires reaching the different dropoff.
        simulator = FleetSimulator(
            GridMap(4, 1),
            [Robot("R", (1, 0))],
            [Task("T", (1, 0), (3, 0))],
        )
        simulator.assign_tasks()
        robot = simulator.robots["R"]
        self.assertEqual(robot.route, [(2, 0), (3, 0)])
        self.assertFalse(simulator.tasks["T"].completed)

        first = simulator.step()
        self.assertTrue(simulator.tasks["T"].picked_up)
        self.assertFalse(simulator.tasks["T"].completed)
        self.assertEqual(robot.position, (2, 0))
        self.assertEqual(first["moved"], ["R"])
        self.assertEqual(first["completed"], [])

        arrival = simulator.step()
        self.assertTrue(simulator.tasks["T"].completed)
        self.assertEqual(robot.position, (3, 0))
        self.assertEqual(arrival["completed"], ["T"])
        self.assertEqual(robot.distance_travelled, 2)


if __name__ == "__main__":
    unittest.main()
