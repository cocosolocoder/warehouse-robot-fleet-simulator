"""Checkpoint save/restore of a taskless preset route held up by a closed cell.

A robot with no bound task that still drives a preset route keeps every
unconsumed waypoint when a runtime edit closes the cell ahead: it drives the
still-open prefix and then simply waits. That state is produced by normal
running, so a version 2 checkpoint must save and restore it instead of
rejecting the remaining route as an illegal walk. Recovery preserves the
position, waypoint order, mileage and map-change history, advances neither
time nor robots, and the restored robot continues the very same route -- no
shortcut, no skipped waypoint -- once the cell reopens.

The allowance is deliberately limited to version 2 robots without a task and
to blocked waypoints the recorded change history can explain: direct fleet
construction still rejects a route through an initial obstacle, version 1
files keep the old rule, task-bound robots keep their reroute/pause rules,
and malformed, out-of-map and non-adjacent waypoints stay rejected.
"""

import copy
import json
import os
import tempfile
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


def save_load(simulator: FleetSimulator) -> FleetSimulator:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        simulator.save_checkpoint(path)
        return FleetSimulator.load_checkpoint(path)


def document_of(simulator: FleetSimulator) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        simulator.save_checkpoint(path)
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)


def load_document(document: dict) -> FleetSimulator:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return FleetSimulator.load_checkpoint(path)


def waiting_simulator(*, stepped: bool = False) -> FleetSimulator:
    # One row of four cells, taskless robot with a three-waypoint preset
    # route; (2, 0) is shut by a runtime edit.
    simulator = FleetSimulator(
        GridMap(4, 1),
        [Robot("R", (0, 0), route=[(1, 0), (2, 0), (3, 0)])],
        [],
    )
    simulator.modify_obstacles(added=[(2, 0)])
    if stepped:
        simulator.step()  # robot reaches (1, 0) and waits there
    return simulator


class BlockedPresetRouteRoundTripTests(unittest.TestCase):
    def test_saving_immediately_after_closing_loads(self) -> None:
        loaded = save_load(waiting_simulator())
        robot = loaded.robots["R"]
        self.assertEqual(robot.position, (0, 0))
        self.assertEqual(robot.route, [(1, 0), (2, 0), (3, 0)])
        self.assertIsNone(robot.task_id)
        self.assertEqual(robot.distance_travelled, 0)
        self.assertEqual(loaded.tick, 0)

    def test_saving_after_the_robot_walked_up_and_waited_loads(self) -> None:
        loaded = save_load(waiting_simulator(stepped=True))
        robot = loaded.robots["R"]
        self.assertEqual(robot.position, (1, 0))
        self.assertEqual(robot.route, [(2, 0), (3, 0)])
        self.assertIsNone(robot.task_id)
        self.assertEqual(robot.distance_travelled, 1)
        self.assertEqual(loaded.tick, 1)

    def test_save_and_load_change_nothing(self) -> None:
        simulator = waiting_simulator(stepped=True)
        simulator.step()  # another wait tick must survive the round trip too
        before = copy.deepcopy(simulator.snapshot())
        history_before = simulator.map_change_history()
        metrics_before = simulator.metrics()
        loaded = save_load(simulator)
        self.assertEqual(loaded.snapshot(), before)
        self.assertEqual(loaded.map_change_history(), history_before)
        self.assertEqual(loaded.metrics(), metrics_before)
        self.assertEqual(loaded.grid.obstacles, simulator.grid.obstacles)
        self.assertEqual(loaded.base_grid.obstacles, simulator.base_grid.obstacles)
        # The save left the original instance at the very same tick.
        self.assertEqual(simulator.tick, loaded.tick)

    def test_map_change_history_round_trips(self) -> None:
        simulator = waiting_simulator(stepped=True)
        simulator.modify_obstacles(added=[(0, 0)])  # a second, unrelated edit
        loaded = save_load(simulator)
        self.assertEqual(loaded.map_change_history(), simulator.map_change_history())
        self.assertEqual(loaded.grid.obstacles, frozenset({(0, 0), (2, 0)}))
        self.assertEqual(loaded.base_grid.obstacles, frozenset())

    def test_two_loads_of_one_file_are_independent(self) -> None:
        document = document_of(waiting_simulator(stepped=True))
        first = load_document(document)
        second = load_document(document)
        first.modify_obstacles(removed=[(2, 0)])
        first.step()
        self.assertIn((2, 0), second.grid.obstacles)
        self.assertEqual(second.robots["R"].route, [(2, 0), (3, 0)])
        self.assertIsNot(first.robots["R"].route, second.robots["R"].route)


class RestoredWaitingBehaviorTests(unittest.TestCase):
    def test_restored_robot_finishes_open_waypoints_then_waits(self) -> None:
        # Saved still at (0, 0): it may drive onto (1, 0) but never into the
        # closed (2, 0), and waiting preserves route and mileage.
        loaded = save_load(waiting_simulator())
        event = loaded.step()
        self.assertEqual(event["moved"], ["R"])
        robot = loaded.robots["R"]
        self.assertEqual(robot.position, (1, 0))
        self.assertEqual(robot.route, [(2, 0), (3, 0)])
        self.assertEqual(robot.distance_travelled, 1)
        for _ in range(4):
            event = loaded.step()
            self.assertEqual(event["moved"], [])
            self.assertEqual(robot.position, (1, 0))
            self.assertEqual(robot.route, [(2, 0), (3, 0)])
            self.assertEqual(robot.distance_travelled, 1)

    def test_waiting_is_not_a_pause_and_creates_no_traffic_record(self) -> None:
        loaded = save_load(waiting_simulator(stepped=True))
        for _ in range(3):
            loaded.step()
        self.assertEqual(loaded.paused_tasks, set())
        self.assertEqual(loaded.status()["paused_tasks"], [])
        self.assertEqual(loaded.status()["traffic_waits"], [])
        self.assertEqual(loaded.metrics()["traffic_waits"], [])
        self.assertFalse(any(t.picked_up for t in loaded.tasks.values()))

    def test_reopen_continues_the_original_route_without_shortcut(self) -> None:
        loaded = save_load(waiting_simulator(stepped=True))
        for _ in range(2):
            loaded.step()  # wait ticks
        loaded.modify_obstacles(removed=[(2, 0)])
        # The edit alone neither moves the robot nor advances time.
        self.assertEqual(loaded.robots["R"].position, (1, 0))
        self.assertEqual(loaded.tick, 3)
        first = loaded.step()
        self.assertEqual(first["moved"], ["R"])
        self.assertEqual(loaded.robots["R"].position, (2, 0))
        self.assertEqual(loaded.robots["R"].route, [(3, 0)])
        second = loaded.step()
        self.assertEqual(second["moved"], ["R"])
        self.assertEqual(loaded.robots["R"].position, (3, 0))
        self.assertEqual(loaded.robots["R"].route, [])
        # Three original waypoints total: one before the save, two after.
        self.assertEqual(loaded.robots["R"].distance_travelled, 3)
        # The blocked waypoint was visited, not skipped.
        visited = [
            tuple(frame["robots"]["R"])
            for frame in loaded.replay
            if frame.get("type") == "tick"
        ]
        self.assertEqual(visited[-3:], [(1, 0), (2, 0), (3, 0)])

    def test_traffic_behind_the_waiting_robot_round_trips_and_resolves(self) -> None:
        # A queues directly behind the blocked preset robot B in a one-row
        # corridor, so A is genuinely held by traffic while B faces the
        # closed cell. Only A's wait is a traffic record; B has none.
        simulator = FleetSimulator(
            GridMap(4, 1),
            [
                Robot("A", (0, 0), route=[(1, 0), (2, 0)]),
                Robot("B", (1, 0), route=[(2, 0), (3, 0)]),
            ],
            [],
        )
        simulator.modify_obstacles(added=[(2, 0)])
        simulator.step()
        simulator.step()
        self.assertEqual(
            simulator.status()["traffic_waits"],
            [{"robot_id": "A", "blocked_by": ["B"], "ticks": 2}],
        )
        loaded = save_load(simulator)
        self.assertEqual(
            loaded.status()["traffic_waits"],
            [{"robot_id": "A", "blocked_by": ["B"], "ticks": 2}],
        )
        loaded.modify_obstacles(removed=[(2, 0)])
        loaded.step()  # B advances into (2, 0); A's queued wait clears
        self.assertEqual(loaded.robots["B"].position, (2, 0))
        self.assertEqual(loaded.robots["A"].position, (0, 0))
        self.assertEqual(loaded.status()["traffic_waits"], [])
        loaded.step()  # A follows into (1, 0), B reaches (3, 0)
        loaded.step()  # A drives through the formerly blocked (2, 0)
        self.assertEqual(loaded.robots["B"].position, (3, 0))
        self.assertEqual(loaded.robots["B"].route, [])
        self.assertEqual(loaded.robots["A"].position, (2, 0))
        self.assertEqual(loaded.robots["A"].route, [])

    def test_restored_waiter_is_not_pushed_aside_nor_hired(self) -> None:
        # Two-row map gives a genuine side cell; a task-bound robot behind the
        # waiter must never push it off its preset route, and a pending task
        # must never bind the waiter -- the separate idle robot takes it.
        simulator = FleetSimulator(
            GridMap(4, 2),
            [
                Robot("B", (1, 0), route=[(2, 0), (3, 0)]),
                Robot("C", (0, 1)),
            ],
            [Task("T-1", (2, 1), (3, 1))],
        )
        simulator.modify_obstacles(added=[(2, 0)])
        loaded = save_load(simulator)
        loaded.step()
        waiter = loaded.robots["B"]
        self.assertEqual(waiter.position, (1, 0))
        self.assertEqual(waiter.route, [(2, 0), (3, 0)])
        self.assertIsNone(waiter.task_id)
        self.assertEqual(loaded.tasks["T-1"].assigned_robot, "C")
        loaded.modify_obstacles(removed=[(2, 0)])
        for _ in range(4):
            loaded.step()
        self.assertEqual(loaded.robots["B"].position, (3, 0))
        self.assertEqual(loaded.robots["B"].route, [])
        self.assertEqual(loaded.robots["B"].position[1], 0)


class ResumeDeterminismTests(unittest.TestCase):
    def test_interrupted_resume_matches_uninterrupted_run(self) -> None:
        def script() -> FleetSimulator:
            simulator = FleetSimulator(
                GridMap(4, 1),
                [Robot("R", (0, 0), route=[(1, 0), (2, 0), (3, 0)])],
                [],
            )
            simulator.modify_obstacles(added=[(2, 0)])
            simulator.step()  # reaches (1, 0)
            return simulator

        full = script()
        resumed = save_load(script())
        for _ in range(3):  # wait through the closure
            full.step()
            resumed.step()
        full.modify_obstacles(removed=[(2, 0)])
        resumed.modify_obstacles(removed=[(2, 0)])
        for _ in range(3):
            full.step()
            resumed.step()
        self.assertEqual(resumed.snapshot(), full.snapshot())
        self.assertEqual(resumed.map_change_history(), full.map_change_history())
        self.assertEqual(resumed.metrics(), full.metrics())


class RejectedBlockedRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = document_of(waiting_simulator())

    def reject(self, document: dict, fragment: str) -> None:
        with self.assertRaises(ValueError) as context:
            load_document(document)
        self.assertIn(fragment, str(context.exception))

    def mutated(self, mutate) -> dict:
        document = copy.deepcopy(self.document)
        mutate(document)
        return document

    def test_direct_construction_rejects_route_through_initial_obstacle(self) -> None:
        with self.assertRaises(ValueError) as context:
            FleetSimulator(
                GridMap(4, 1, frozenset({(2, 0)})),
                [Robot("R", (0, 0), route=[(1, 0), (2, 0), (3, 0)])],
                [],
            )
        self.assertIn("inside an obstacle", str(context.exception))

    def test_version_one_file_keeps_the_strict_route_rule(self) -> None:
        document = copy.deepcopy(self.document)
        document["version"] = 1
        for key in ("base_grid", "map_changes", "paused_tasks", "traffic_waits"):
            document.pop(key, None)
        document["replay"] = [
            {key: value for key, value in frame.items() if key != "type"}
            for frame in document["replay"]
            if frame.get("type") == "tick"
        ]
        self.reject(document, "inside an obstacle")

    def test_base_map_obstacle_not_explained_by_history_rejected(self) -> None:
        # Putting the closed cell into the base map makes the recorded add
        # illegal per the existing history rules; either way it must reject.
        document = self.mutated(lambda d: d["base_grid"]["obstacles"].append([2, 0]))
        with self.assertRaises(ValueError):
            load_document(document)

    def test_blocked_waypoint_without_matching_history_rejected(self) -> None:
        # The saved grid blocks (2, 0) and the route waits there, but the
        # change history is empty and the base map already lists the cell as
        # an obstacle: nothing explains a preset robot driving into it, so the
        # route stays illegal even though grid and history are mutually
        # consistent.
        document = self.mutated(
            lambda d: (
                d["base_grid"]["obstacles"].append([2, 0]),
                d.__setitem__("map_changes", []),
                d.__setitem__(
                    "replay",
                    [
                        frame
                        for frame in d["replay"]
                        if frame.get("type") == "tick"
                    ],
                ),
            )
        )
        self.reject(document, "inside an obstacle")

    def test_task_bound_robot_never_gets_the_exemption(self) -> None:
        document = copy.deepcopy(self.document)
        document["robots"][0]["task_id"] = "T-1"
        document["tasks"] = [
            {
                "task_id": "T-1",
                "pickup": [0, 0],
                "dropoff": [3, 0],
                "assigned_robot": "R",
                "picked_up": True,
                "completed": False,
            }
        ]
        for frame in document["replay"]:
            if frame.get("type") == "tick":
                frame["completed"] = []
        self.reject(document, "inside an obstacle")

    def test_current_position_on_the_obstacle_rejected(self) -> None:
        document = self.mutated(
            lambda d: d["robots"][0].__setitem__("position", [2, 0])
        )
        with self.assertRaises(ValueError):
            load_document(document)

    def test_out_of_bounds_blocked_waypoint_rejected(self) -> None:
        document = self.mutated(
            lambda d: d["robots"][0]["route"].__setitem__(2, [4, 0])
        )
        self.reject(document, "outside the map or inside an obstacle")

    def test_non_integer_waypoint_rejected(self) -> None:
        document = self.mutated(
            lambda d: d["robots"][0]["route"].__setitem__(2, [3.0, 0])
        )
        self.reject(document, "must be a [x, y] integer pair")

    def test_jump_between_waypoints_rejected_even_when_target_open(self) -> None:
        document = self.mutated(
            lambda d: d["robots"][0]["route"].__setitem__(1, [3, 0])
        )
        self.reject(document, "non-adjacently")

    def test_reopened_cell_is_not_a_blocked_waypoint_and_still_loads(self) -> None:
        # Closed and reopened before the save nets to open: ordinary walkable
        # route, and it must load through the normal (non-exempt) path.
        simulator = FleetSimulator(
            GridMap(4, 1),
            [Robot("R", (0, 0), route=[(1, 0), (2, 0), (3, 0)])],
            [],
        )
        simulator.modify_obstacles(added=[(2, 0)])
        simulator.modify_obstacles(removed=[(2, 0)])
        loaded = save_load(simulator)
        self.assertEqual(
            loaded.robots["R"].route, [(1, 0), (2, 0), (3, 0)]
        )
        self.assertNotIn((2, 0), loaded.grid.obstacles)


if __name__ == "__main__":
    unittest.main()
