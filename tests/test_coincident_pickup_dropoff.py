"""Regression tests for tasks whose pickup point and dropoff share one cell.

The saved-state tests in ``test_checkpoint_task_routes.py`` already pin the
empty-route checkpoint form of this situation. These tests pin the same rule
through ordinary programmatic use instead: build the simulator, call
``assign_tasks`` and drive it tick by tick with ``step``. A coincident
pickup/dropoff is a normal task whose *route* is empty only when the robot is
already standing on that cell; it is never a free completion. The goods are
collected and delivered in one tick, but only once a robot has actually
arrived on the cell, and every observable channel has to agree about when that
happens: task flags, robot binding and remaining route, position and mileage,
completion metrics, the replay frame for that tick and the traffic reports.
"""

import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task, shortest_path


# ---------------------------------------------------------------------------
# Shared correspondence checks
# ---------------------------------------------------------------------------


def assert_tick_frames_track_state(
    testcase: unittest.TestCase,
    replay: list[dict[str, object]],
    initial_positions: dict[str, tuple[int, int]],
) -> dict[str, tuple[int, int]]:
    """Replay tick frames must mirror the moves and positions they describe.

    Tick frames are numbered consecutively from one; a frame's ``moved`` list
    names exactly the robots whose cell differs from the previous tick frame
    (the initial layout serves as the predecessor of frame one), two robots
    never share a cell at a tick boundary, and interspersed map-change frames
    carry the tick they took effect at without resetting the numbering. The
    final positions are returned so a caller can compare them with the live
    robots.
    """
    previous = dict(initial_positions)
    tick_number = 0
    for frame in replay:
        if frame["type"] != "tick":
            testcase.assertEqual(frame["tick"], tick_number)
            continue
        tick_number += 1
        testcase.assertEqual(frame["tick"], tick_number)
        positions = {
            robot_id: tuple(cell) for robot_id, cell in frame["robots"].items()
        }
        testcase.assertEqual(len(positions), len(set(positions.values())))
        changed = sorted(
            robot_id
            for robot_id, position in positions.items()
            if position != previous[robot_id]
        )
        testcase.assertEqual(sorted(frame["moved"]), changed)
        previous = positions
    return previous


# ---------------------------------------------------------------------------
# One robot, already standing on the pickup/dropoff cell
# ---------------------------------------------------------------------------


class StandingOnCoincidentPointsTests(unittest.TestCase):
    CELL = (2, 1)

    def build(self) -> FleetSimulator:
        return FleetSimulator(
            GridMap(5, 3),
            [Robot("R-1", self.CELL)],
            [Task("T-1", self.CELL, self.CELL)],
        )

    def test_assignment_binds_task_but_does_not_collect_or_finish(self) -> None:
        simulator = self.build()
        simulator.assign_tasks()

        robot = simulator.robots["R-1"]
        task = simulator.tasks["T-1"]
        # Assignment establishes the mutual ownership binding...
        self.assertEqual(robot.task_id, "T-1")
        self.assertEqual(task.assigned_robot, "R-1")
        # ...and leaves the feasible zero-step remaining route empty, but the
        # task is neither picked up nor completed merely by being assigned.
        self.assertEqual(robot.route, [])
        self.assertFalse(task.picked_up)
        self.assertFalse(task.completed)
        # Assignment itself advances nothing: no tick, no movement, no mileage,
        # no replay or map history.
        self.assertEqual(simulator.tick, 0)
        self.assertEqual(robot.position, self.CELL)
        self.assertEqual(robot.distance_travelled, 0)
        self.assertEqual(simulator.replay, [])
        self.assertEqual(simulator.map_change_history(), [])
        self.assertEqual(simulator.paused_tasks, set())
        self.assertEqual(simulator.metrics()["tasks_completed"], 0)
        self.assertEqual(simulator.metrics()["distance_total"], 0)

    def test_one_step_collects_and_delivers_in_place(self) -> None:
        simulator = self.build()
        simulator.assign_tasks()
        event = simulator.step()

        robot = simulator.robots["R-1"]
        task = simulator.tasks["T-1"]
        self.assertTrue(task.picked_up)
        self.assertTrue(task.completed)
        # The robot is released for new work; the task keeps the robot that
        # executed it as its historical owner.
        self.assertIsNone(robot.task_id)
        self.assertEqual(task.assigned_robot, "R-1")
        self.assertEqual(robot.route, [])
        # Nothing was driven: position and mileage are unchanged and the replay
        # frame for the completion tick records no movement.
        self.assertEqual(robot.position, self.CELL)
        self.assertEqual(robot.distance_travelled, 0)
        self.assertEqual(event["moved"], [])
        self.assertEqual(event["robots"], {"R-1": list(self.CELL)})
        self.assertEqual(event["completed"], ["T-1"])
        self.assertEqual(simulator.tick, 1)
        metrics = simulator.metrics()
        self.assertEqual(metrics["tasks_completed"], 1)
        self.assertEqual(metrics["distance_total"], 0)
        # An in-place completion is traffic, not a pause.
        self.assertEqual(simulator.status()["traffic_waits"], [])
        self.assertEqual(simulator.paused_tasks, set())

    def test_later_steps_stay_idle_without_double_counting(self) -> None:
        simulator = self.build()
        simulator.assign_tasks()
        simulator.step()
        event = simulator.step()

        # The robot simply stands there; no second completion is counted and no
        # mileage or movement ever appears.
        self.assertEqual(event["moved"], [])
        self.assertEqual(event["completed"], ["T-1"])
        self.assertEqual(simulator.robots["R-1"].position, self.CELL)
        self.assertEqual(simulator.robots["R-1"].distance_travelled, 0)
        self.assertIsNone(simulator.robots["R-1"].task_id)
        self.assertEqual(simulator.metrics()["tasks_completed"], 1)
        self.assertEqual(simulator.tick, 2)
        final_positions = assert_tick_frames_track_state(
            self, simulator.replay, {"R-1": self.CELL}
        )
        self.assertEqual(final_positions, {"R-1": self.CELL})


# ---------------------------------------------------------------------------
# Robot elsewhere: drive the shortest route and finish exactly on arrival
# ---------------------------------------------------------------------------


class ApproachCoincidentPointsTests(unittest.TestCase):
    def test_straight_corridor_completes_on_arrival_step_only(self) -> None:
        target = (3, 0)
        simulator = FleetSimulator(
            GridMap(5, 1),
            [Robot("R-1", (0, 0))],
            [Task("T-1", target, target)],
        )
        simulator.assign_tasks()
        robot = simulator.robots["R-1"]
        task = simulator.tasks["T-1"]
        self.assertEqual(robot.route, [(1, 0), (2, 0), (3, 0)])
        self.assertEqual(robot.task_id, "T-1")
        self.assertFalse(task.picked_up)

        first = simulator.step()
        self.assertEqual(robot.position, (1, 0))
        self.assertEqual(robot.route, [(2, 0), (3, 0)])
        self.assertFalse(task.picked_up)
        self.assertFalse(task.completed)
        self.assertEqual(first["moved"], ["R-1"])
        self.assertEqual(first["completed"], [])

        second = simulator.step()
        # Passing through a cell that is neither the shared point nor the end
        # of the route collects nothing.
        self.assertEqual(robot.position, (2, 0))
        self.assertEqual(robot.route, [(3, 0)])
        self.assertFalse(task.picked_up)
        self.assertFalse(task.completed)
        self.assertEqual(second["completed"], [])

        arrival = simulator.step()
        # Entering the target cell both collects and delivers on that same
        # tick: the route is exhausted, no extra wait or return trip follows.
        self.assertEqual(robot.position, target)
        self.assertEqual(robot.route, [])
        self.assertTrue(task.picked_up)
        self.assertTrue(task.completed)
        self.assertIsNone(robot.task_id)
        self.assertEqual(task.assigned_robot, "R-1")
        self.assertEqual(robot.distance_travelled, 3)
        self.assertEqual(arrival["moved"], ["R-1"])
        self.assertEqual(arrival["completed"], ["T-1"])

        after = simulator.step()
        self.assertEqual(after["moved"], [])
        self.assertEqual(after["completed"], ["T-1"])
        self.assertEqual(robot.position, target)
        self.assertEqual(robot.distance_travelled, 3)
        self.assertEqual(simulator.metrics()["tasks_completed"], 1)
        self.assertEqual(simulator.metrics()["distance_total"], 3)
        final_positions = assert_tick_frames_track_state(
            self, simulator.replay, {"R-1": (0, 0)}
        )
        self.assertEqual(final_positions, {"R-1": target})

    def test_detour_uses_shortest_route_and_finishes_without_extra_wait(self) -> None:
        # The straight approach is sealed at (2, 0)/(2, 1); the shortest route
        # loops one row further up and still ends exactly on the shared point.
        grid = GridMap(5, 3, frozenset({(2, 0), (2, 1)}))
        start, target = (0, 0), (2, 2)
        expected_route = shortest_path(grid, start, target)
        self.assertEqual(expected_route, [(1, 0), (1, 1), (1, 2), (2, 2)])

        simulator = FleetSimulator(
            grid, [Robot("R-1", start)], [Task("T-1", target, target)]
        )
        simulator.assign_tasks()
        robot = simulator.robots["R-1"]
        task = simulator.tasks["T-1"]
        self.assertEqual(robot.route, expected_route)

        for _ in range(len(expected_route) - 1):
            simulator.step()
            self.assertFalse(task.picked_up, "goods must not be collected en route")
            self.assertFalse(task.completed)
            self.assertIsNotNone(robot.task_id)

        arrival = simulator.step()
        self.assertEqual(robot.position, target)
        self.assertTrue(task.picked_up)
        self.assertTrue(task.completed)
        self.assertEqual(robot.route, [])
        self.assertIsNone(robot.task_id)
        self.assertEqual(robot.distance_travelled, len(expected_route))
        self.assertEqual(arrival["completed"], ["T-1"])
        self.assertEqual(arrival["moved"], ["R-1"])
        self.assertEqual(simulator.metrics()["distance_total"], len(expected_route))


# ---------------------------------------------------------------------------
# The target cell is occupied by another robot with no safe cell to yield to
# ---------------------------------------------------------------------------


class VehicleBlockageTests(unittest.TestCase):
    """A dead-end target held by a parked robot is ordinary traffic.

    The map is a two-row strip whose side cell at the target (2, 1) is closed,
    so the cell (2, 0) is the end of a one-wide corridor: a robot parked there
    has nowhere safe to yield onto. B delivers its own coincident-points task
    straight onto that cell and parks; A, carrying a second such task to the
    very same cell, must wait like behind any other vehicle -- it must not be
    treated as "arrived" while it stands one cell short.
    """

    def build(self) -> FleetSimulator:
        grid = GridMap(3, 2, frozenset({(2, 1)}))
        robot_a = Robot("A", (0, 0))
        # B is already delivering its own task into the dead-end cell.
        robot_b = Robot("B", (1, 0), route=[(2, 0)], task_id="TB")
        tasks = [
            Task("TA", (2, 0), (2, 0)),
            Task("TB", (2, 0), (2, 0), assigned_robot="B"),
        ]
        return FleetSimulator(grid, [robot_a, robot_b], tasks)

    def test_waits_behind_parked_blocker_without_completing(self) -> None:
        simulator = self.build()
        simulator.assign_tasks()
        a, b = simulator.robots["A"], simulator.robots["B"]
        task_a = simulator.tasks["TA"]
        self.assertEqual(a.task_id, "TA")
        self.assertEqual(task_a.assigned_robot, "A")
        self.assertEqual(a.route, [(1, 0), (2, 0)])

        # Tick 1: B drives into the dead end and finishes its own task; A lets
        # the blocker drive on this tick and keeps its whole route unconsumed.
        first = simulator.step()
        self.assertEqual(b.position, (2, 0))
        self.assertTrue(simulator.tasks["TB"].completed)
        self.assertIsNone(b.task_id)
        self.assertEqual(a.position, (0, 0))
        self.assertEqual(a.route, [(1, 0), (2, 0)])
        self.assertEqual(first["moved"], ["B"])
        self.assertNotIn("TA", first["completed"])
        self.assertEqual(simulator.status()["traffic_waits"], [])

        # Tick 2: A advances to the cell before the parked blocker.
        second = simulator.step()
        self.assertEqual(a.position, (1, 0))
        self.assertEqual(a.route, [(2, 0)])
        self.assertFalse(task_a.picked_up)
        self.assertFalse(task_a.completed)
        self.assertEqual(a.distance_travelled, 1)
        self.assertEqual(second["moved"], ["A"])
        self.assertEqual(simulator.status()["traffic_waits"], [])

        # Ticks 3 and 4: B has no safe side cell, so A is held by traffic.
        for tick, expected_count in ((3, 1), (4, 2)):
            event = simulator.step()
            self.assertEqual(a.position, (1, 0))
            self.assertEqual(a.route, [(2, 0)], "unconsumed route kept while waiting")
            self.assertEqual(a.task_id, "TA", "task binding kept while waiting")
            self.assertFalse(task_a.picked_up)
            self.assertFalse(task_a.completed, "standing one cell short is not arrival")
            self.assertEqual(a.distance_travelled, 1, "waiting adds no mileage")
            self.assertEqual(event["moved"], [])
            self.assertNotIn("TA", event["completed"])
            # The robots never share a cell.
            self.assertEqual(len({a.position, b.position}), 2)
            self.assertEqual(
                simulator.status()["traffic_waits"],
                [{"robot_id": "A", "blocked_by": ["B"], "ticks": expected_count}],
            )
            self.assertEqual(
                simulator.metrics()["traffic_waits"],
                [{"robot_id": "A", "blocked_by": ["B"], "ticks": expected_count}],
            )
        # This is a vehicle wait, never a map-unreachability pause.
        self.assertEqual(simulator.paused_tasks, set())
        self.assertEqual(simulator.metrics()["tasks_completed"], 1)
        self.assertEqual(simulator.metrics()["distance_total"], 2)
        final_positions = assert_tick_frames_track_state(
            self, simulator.replay, {"A": (0, 0), "B": (1, 0)}
        )
        self.assertEqual(final_positions, {"A": (1, 0), "B": (2, 0)})

    def test_blockage_clears_and_task_completes_on_the_real_arrival_tick(self) -> None:
        simulator = self.build()
        simulator.assign_tasks()
        for _ in range(4):
            simulator.step()
        self.assertEqual(
            simulator.status()["traffic_waits"],
            [{"robot_id": "A", "blocked_by": ["B"], "ticks": 2}],
        )

        # Reopen the side cell; the edit itself moves nothing.
        record = simulator.modify_obstacles(removed=[(2, 1)])
        self.assertEqual(record["tick"], 4)
        self.assertEqual(record["added"], [])
        self.assertEqual(record["removed"], [[2, 1]])
        a, b = simulator.robots["A"], simulator.robots["B"]
        self.assertEqual(a.position, (1, 0))
        self.assertEqual(a.distance_travelled, 1)

        # B yields onto the reopened side cell and A enters the target cell in
        # the same tick; A's task is collected and delivered exactly then.
        event = simulator.step()
        self.assertEqual(b.position, (2, 1))
        self.assertEqual(b.distance_travelled, 2)
        self.assertEqual(a.position, (2, 0))
        self.assertEqual(a.route, [])
        self.assertIsNone(a.task_id)
        self.assertTrue(simulator.tasks["TA"].completed)
        self.assertEqual(simulator.tasks["TA"].assigned_robot, "A")
        self.assertEqual(a.distance_travelled, 2)
        self.assertEqual(event["moved"], ["B", "A"])
        self.assertEqual(event["completed"], ["TA", "TB"])
        self.assertEqual(simulator.status()["traffic_waits"], [])
        self.assertEqual(simulator.metrics()["tasks_completed"], 2)
        self.assertEqual(simulator.metrics()["distance_total"], 4)


# ---------------------------------------------------------------------------
# The shared cell is unreachable until the map is edited
# ---------------------------------------------------------------------------


class UnreachableCoincidentPointsTests(unittest.TestCase):
    """A sealed shared point is an unassigned waiting task, not an instant job.

    A ring of obstacles seals the interior cell (2, 2) off from the rest of
    the map. The coincident-pickup/dropoff task there must keep waiting for a
    robot that can actually reach the cell: an empty planner result must never
    be mistaken for the empty route of a robot standing on the point, and the
    only robot must stay free to take ordinary work. Reopening a gap lets the
    task be assigned and finished on the real arrival step.
    """

    RING = frozenset(
        {
            (1, 1), (2, 1), (3, 1),
            (1, 2), (3, 2),
            (1, 3), (2, 3), (3, 3),
        }
    )

    def build(self) -> FleetSimulator:
        return FleetSimulator(
            GridMap(5, 5, self.RING),
            [Robot("R-1", (0, 0))],
            [
                Task("T-sealed", (2, 2), (2, 2)),
                Task("T-other", (0, 1), (0, 2)),
            ],
        )

    def test_unreachable_task_waits_and_leaves_robot_for_other_work(self) -> None:
        simulator = self.build()
        simulator.assign_tasks()
        sealed = simulator.tasks["T-sealed"]
        other = simulator.tasks["T-other"]
        robot = simulator.robots["R-1"]

        # The robot is hired by the reachable ordinary task; the sealed
        # coincident-points task stays unassigned and registers no half-state.
        self.assertEqual(other.assigned_robot, "R-1")
        self.assertEqual(robot.task_id, "T-other")
        self.assertIsNone(sealed.assigned_robot)
        self.assertFalse(sealed.picked_up)
        self.assertFalse(sealed.completed)
        # Being unassigned is waiting for a feasible robot, not a map pause.
        self.assertEqual(simulator.paused_tasks, set())
        self.assertEqual(simulator.tick, 0)
        self.assertEqual(simulator.replay, [])

        simulator.step()  # collect T-other at (0, 1)
        simulator.step()  # deliver T-other at (0, 2)
        self.assertTrue(other.completed)
        self.assertIsNone(robot.task_id)
        self.assertEqual(robot.position, (0, 2))
        self.assertEqual(robot.distance_travelled, 2)
        self.assertFalse(sealed.completed)

        # Further stepping changes nothing: the still-unreachable task is not
        # force-assigned, never completed in place, and the idle robot waits.
        idle_event = simulator.step()
        self.assertIsNone(sealed.assigned_robot)
        self.assertFalse(sealed.completed)
        self.assertIsNone(robot.task_id)
        self.assertEqual(robot.position, (0, 2))
        self.assertEqual(robot.distance_travelled, 2)
        self.assertEqual(idle_event["moved"], [])
        self.assertEqual(idle_event["completed"], ["T-other"])
        self.assertEqual(simulator.metrics()["tasks_completed"], 1)

    def test_reopening_cell_completes_on_actual_arrival(self) -> None:
        simulator = self.build()
        for _ in range(3):
            simulator.step()
        sealed = simulator.tasks["T-sealed"]
        robot = simulator.robots["R-1"]
        self.assertIsNone(sealed.assigned_robot)
        self.assertEqual(robot.position, (0, 2))

        record = simulator.modify_obstacles(removed=[(1, 2)])
        self.assertEqual(record["tick"], 3)
        self.assertEqual(record["sequence"], 1)
        self.assertEqual(record["removed"], [[1, 2]])
        self.assertEqual(simulator.map_change_history()[-1]["type"], "map_change")
        # The edit moves no robot and adds no mileage.
        self.assertEqual(robot.position, (0, 2))
        self.assertEqual(robot.distance_travelled, 2)

        # Tick 4: the now-reachable task is assigned with a real two-cell route
        # (not an empty in-place route) and the robot takes the first step.
        entering = simulator.step()
        self.assertEqual(sealed.assigned_robot, "R-1")
        self.assertEqual(robot.task_id, "T-sealed")
        self.assertEqual(robot.position, (1, 2))
        self.assertEqual(robot.route, [(2, 2)])
        self.assertFalse(sealed.picked_up)
        self.assertFalse(sealed.completed)
        self.assertEqual(entering["moved"], ["R-1"])
        self.assertNotIn("T-sealed", entering["completed"])

        # Tick 5: actual arrival on (2, 2) collects and delivers at once.
        arrival = simulator.step()
        self.assertEqual(robot.position, (2, 2))
        self.assertEqual(robot.route, [])
        self.assertTrue(sealed.picked_up)
        self.assertTrue(sealed.completed)
        self.assertIsNone(robot.task_id)
        self.assertEqual(sealed.assigned_robot, "R-1")
        self.assertEqual(robot.distance_travelled, 4)
        self.assertEqual(arrival["moved"], ["R-1"])
        self.assertEqual(arrival["completed"], ["T-other", "T-sealed"])

        metrics = simulator.metrics()
        self.assertEqual(metrics["tasks_completed"], 2)
        self.assertEqual(metrics["distance_total"], 4)
        self.assertEqual(metrics["tasks_paused"], [])
        self.assertEqual(metrics["traffic_waits"], [])

        # The replay keeps the edit frame interspersed at tick 3 and the tick
        # frames either side of it unchanged.
        self.assertEqual(simulator.replay[3]["type"], "map_change")
        self.assertEqual(simulator.replay[3]["tick"], 3)
        tick_frames = [frame for frame in simulator.replay if frame["type"] == "tick"]
        self.assertEqual([frame["tick"] for frame in tick_frames], [1, 2, 3, 4, 5])
        self.assertEqual(tick_frames[3]["completed"], ["T-other"])
        self.assertEqual(tick_frames[4]["completed"], ["T-other", "T-sealed"])
        final_positions = assert_tick_frames_track_state(
            self, simulator.replay, {"R-1": (0, 0)}
        )
        self.assertEqual(final_positions, {"R-1": (2, 2)})


# ---------------------------------------------------------------------------
# Ordinary tasks with distinct pickup/dropoff keep their existing timing
# ---------------------------------------------------------------------------


class DistinctPointsUnchangedTests(unittest.TestCase):
    def test_pickup_and_delivery_still_happen_at_their_own_cells(self) -> None:
        simulator = FleetSimulator(
            GridMap(6, 1),
            [Robot("R-1", (0, 0))],
            [Task("T-1", (2, 0), (4, 0))],
        )
        simulator.assign_tasks()
        robot = simulator.robots["R-1"]
        task = simulator.tasks["T-1"]
        self.assertEqual(robot.route, [(1, 0), (2, 0), (3, 0), (4, 0)])

        simulator.step()  # (1, 0): nothing yet
        self.assertFalse(task.picked_up)
        simulator.step()  # (2, 0): collect only
        self.assertEqual(robot.position, (2, 0))
        self.assertTrue(task.picked_up)
        self.assertFalse(task.completed)
        simulator.step()  # (3, 0): carrying goods, not delivered
        self.assertFalse(task.completed)
        event = simulator.step()  # (4, 0): delivery
        self.assertEqual(robot.position, (4, 0))
        self.assertTrue(task.completed)
        self.assertIsNone(robot.task_id)
        self.assertEqual(task.assigned_robot, "R-1")
        self.assertEqual(robot.distance_travelled, 4)
        self.assertEqual(event["moved"], ["R-1"])
        self.assertEqual(event["completed"], ["T-1"])


if __name__ == "__main__":
    unittest.main()
