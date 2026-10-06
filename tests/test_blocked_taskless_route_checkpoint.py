"""Checkpoint save/restore of a taskless preset route held up by a map edit.

A robot with no bound task that still drives a preset route keeps every
unvisited waypoint when one of them is closed at runtime: it walks up to the
closure and waits, consuming no waypoint and no mileage, until the cell
reopens. Such a waiting state is produced by ordinary, legal runs, so a
version 2 checkpoint captured while the robot waits -- including one captured
immediately after the closing edit, before any tick -- must load again with
the current position, remaining waypoint order, mileage, map and edit history
identical to the saved state, and the restored robot must keep driving the
very same preset route cell by cell: it never enters the closed cell, never
takes a shortcut around the closure, and resumes from the blocked waypoint
the moment the cell reopens.

The allowance is deliberately narrow. Only a *taskless* robot in a *version 2*
checkpoint may name waypoints that were open on the initial map and were
closed by a valid, recorded map change that is still in effect when saved.
Direct fleet construction still rejects a route through an initial obstacle,
a checkpoint obstacle the history cannot explain (one already present on the
base map) is still rejected, and old version 1 files keep their strict rule.
The exemption never excuses the robot standing on an obstacle, an
out-of-bounds or non-integer waypoint, or a jump between waypoints. A
task-bound robot is rerouted or paused on an edit, so its route may never
cross a current obstacle either.
"""

import json
import os
import tempfile
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


ROW_WIDTH = 4


def waiting_simulator(walk_first: bool = False) -> FleetSimulator:
    """One-row map with (2, 0) dynamically closed in front of robot ``R``.

    The taskless robot starts at (0, 0) with the preset route
    (1, 0) -> (2, 0) -> (3, 0). When *walk_first* is false the checkpoint
    scenario is captured immediately after the closing edit (still at
    tick 0); otherwise the robot has already stepped onto (1, 0), the last
    traversable cell before the closure, and waited there.
    """
    simulator = FleetSimulator(
        GridMap(ROW_WIDTH, 1),
        [Robot("R", (0, 0), route=[(1, 0), (2, 0), (3, 0)])],
        [],
    )
    simulator.modify_obstacles(added=[(2, 0)])
    if walk_first:
        event = simulator.step()
        assert event["moved"] == ["R"]
        assert simulator.robots["R"].position == (1, 0)
    return simulator


def round_trip(simulator: FleetSimulator) -> FleetSimulator:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        simulator.save_checkpoint(path)
        return FleetSimulator.load_checkpoint(path)


def document_for(simulator: FleetSimulator) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        simulator.save_checkpoint(path)
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)


def load_document(document: dict) -> FleetSimulator:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return FleetSimulator.load_checkpoint(path)


def expect_rejected(testcase: unittest.TestCase, document: dict, *fragments: str) -> str:
    with testcase.assertRaises(ValueError) as context:
        load_document(document)
    message = str(context.exception)
    for fragment in fragments:
        testcase.assertIn(fragment, message)
    return message


class WaitingStateRoundTripTests(unittest.TestCase):
    def assert_restored_matches_save(
        self, original: FleetSimulator, restored: FleetSimulator
    ) -> None:
        before = original.robots["R"]
        after = restored.robots["R"]
        self.assertEqual(after.position, before.position)
        self.assertEqual(after.route, before.route)
        self.assertEqual(after.distance_travelled, before.distance_travelled)
        self.assertIsNone(after.task_id)
        self.assertEqual(restored.tick, original.tick)
        self.assertEqual(restored.grid.obstacles, original.grid.obstacles)
        self.assertEqual(restored.base_grid.obstacles, original.base_grid.obstacles)
        self.assertEqual(restored.map_change_history(), original.map_change_history())
        self.assertEqual(restored.paused_tasks, set())
        self.assertEqual(restored.status()["traffic_waits"], [])
        self.assertEqual(restored.metrics(), original.metrics())

    def test_save_immediately_after_closing_loads(self) -> None:
        original = waiting_simulator(walk_first=False)
        self.assertEqual(original.tick, 0)
        restored = round_trip(original)
        self.assertEqual(restored.tick, 0)
        self.assertEqual(restored.robots["R"].position, (0, 0))
        self.assertEqual(restored.robots["R"].route, [(1, 0), (2, 0), (3, 0)])
        self.assertEqual(restored.robots["R"].distance_travelled, 0)
        self.assert_restored_matches_save(original, restored)
        self.assertEqual(
            restored.map_change_history(),
            [
                {
                    "type": "map_change",
                    "tick": 0,
                    "sequence": 1,
                    "added": [[2, 0]],
                    "removed": [],
                }
            ],
        )

    def test_save_after_walking_to_the_closure_loads(self) -> None:
        original = waiting_simulator(walk_first=True)
        self.assertEqual(original.tick, 1)
        restored = round_trip(original)
        self.assertEqual(restored.robots["R"].position, (1, 0))
        self.assertEqual(restored.robots["R"].route, [(2, 0), (3, 0)])
        self.assertEqual(restored.robots["R"].distance_travelled, 1)
        self.assert_restored_matches_save(original, restored)

    def test_saving_an_loaded_instance_reproduces_the_file(self) -> None:
        original = waiting_simulator(walk_first=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            first_path = os.path.join(tmpdir, "first.json")
            second_path = os.path.join(tmpdir, "second.json")
            original.save_checkpoint(first_path)
            restored = FleetSimulator.load_checkpoint(first_path)
            restored.save_checkpoint(second_path)
            with open(first_path, encoding="utf-8") as handle:
                first_document = json.load(handle)
            with open(second_path, encoding="utf-8") as handle:
                second_document = json.load(handle)
        self.assertEqual(second_document, first_document)

    def test_two_loads_run_independently(self) -> None:
        document = document_for(waiting_simulator(walk_first=True))
        first = load_document(document)
        second = load_document(document)
        first.modify_obstacles(removed=[(2, 0)])
        first.step()
        self.assertEqual(second.robots["R"].position, (1, 0))
        self.assertEqual(second.robots["R"].route, [(2, 0), (3, 0)])
        self.assertEqual(second.grid.obstacles, frozenset({(2, 0)}))


class RestoredWaitingBehaviorTests(unittest.TestCase):
    def test_restored_robot_reaches_closure_then_waits_without_cost(self) -> None:
        restored = round_trip(waiting_simulator(walk_first=False))
        event = restored.step()
        self.assertEqual(event["moved"], ["R"])
        self.assertEqual(restored.robots["R"].position, (1, 0))
        self.assertEqual(restored.robots["R"].route, [(2, 0), (3, 0)])
        self.assertEqual(restored.robots["R"].distance_travelled, 1)
        # Held up purely by the closed cell: the wait never consumes a
        # waypoint, books mileage, pauses a task or fakes a traffic record.
        for _ in range(4):
            event = restored.step()
            self.assertEqual(event["moved"], [])
            robot = restored.robots["R"]
            self.assertEqual(robot.position, (1, 0))
            self.assertEqual(robot.route, [(2, 0), (3, 0)])
            self.assertEqual(robot.distance_travelled, 1)
        self.assertEqual(restored.paused_tasks, set())
        self.assertEqual(restored.status()["traffic_waits"], [])

    def test_restored_waiting_robot_keeps_waiting_without_cost(self) -> None:
        restored = round_trip(waiting_simulator(walk_first=True))
        tick = restored.tick
        for _ in range(4):
            event = restored.step()
            self.assertEqual(event["moved"], [])
        robot = restored.robots["R"]
        self.assertEqual(robot.position, (1, 0))
        self.assertEqual(robot.route, [(2, 0), (3, 0)])
        self.assertEqual(robot.distance_travelled, 1)
        self.assertEqual(restored.tick, tick + 4)
        self.assertEqual(restored.paused_tasks, set())
        self.assertEqual(restored.status()["traffic_waits"], [])

    def test_reopening_resumes_the_original_route_with_no_shortcut(self) -> None:
        # Two-row map: the bottom row offers a way around (2, 0). The
        # restored robot must ignore it, keep waiting, and drive the preset
        # route straight through the reopened cell in order.
        simulator = FleetSimulator(
            GridMap(4, 2),
            [Robot("R", (0, 0), route=[(1, 0), (2, 0), (3, 0)])],
            [],
        )
        simulator.modify_obstacles(added=[(2, 0)])
        simulator.step()
        restored = round_trip(simulator)
        for _ in range(3):
            self.assertEqual(restored.step()["moved"], [])
        self.assertEqual(restored.robots["R"].position, (1, 0))

        restored.modify_obstacles(removed=[(2, 0)])
        self.assertEqual(restored.robots["R"].route, [(2, 0), (3, 0)])
        event = restored.step()
        self.assertEqual(event["moved"], ["R"])
        self.assertEqual(restored.robots["R"].position, (2, 0))
        self.assertEqual(restored.robots["R"].route, [(3, 0)])
        self.assertEqual(restored.robots["R"].distance_travelled, 2)
        event = restored.step()
        self.assertEqual(event["moved"], ["R"])
        self.assertEqual(restored.robots["R"].position, (3, 0))
        self.assertEqual(restored.robots["R"].route, [])
        self.assertEqual(restored.robots["R"].distance_travelled, 3)
        for frame in restored.replay:
            if frame.get("type") == "tick":
                self.assertEqual(frame["robots"]["R"][1], 0)

    def test_restored_robot_is_not_assigned_a_waiting_task(self) -> None:
        simulator = FleetSimulator(
            GridMap(6, 1),
            [
                Robot("blocked", (0, 0), route=[(1, 0), (2, 0), (3, 0)]),
                Robot("free", (5, 0)),
            ],
            [Task("T", (4, 0), (5, 0))],
        )
        simulator.modify_obstacles(added=[(2, 0)])
        restored = round_trip(simulator)
        # One tick: the blocked robot only drives the open prefix of its own
        # route, the available robot receives the pending task. The blocked
        # robot's remaining route is never overwritten by the assignment.
        restored.step()
        blocked = restored.robots["blocked"]
        self.assertIsNone(blocked.task_id)
        self.assertEqual(blocked.position, (1, 0))
        self.assertEqual(blocked.route, [(2, 0), (3, 0)])
        self.assertEqual(blocked.distance_travelled, 1)
        self.assertEqual(restored.tasks["T"].assigned_robot, "free")
        self.assertEqual(restored.robots["free"].task_id, "T")
        # Waiting at the closure never makes it eligible later either.
        restored.step()
        self.assertIsNone(restored.robots["blocked"].task_id)
        self.assertEqual(restored.robots["blocked"].route, [(2, 0), (3, 0)])


class BlockedRouteExemptionScopeTests(unittest.TestCase):
    def test_direct_construction_still_rejects_an_initial_obstacle(self) -> None:
        with self.assertRaises(ValueError) as context:
            FleetSimulator(
                GridMap(4, 1, frozenset({(2, 0)})),
                [Robot("R", (0, 0), route=[(1, 0), (2, 0), (3, 0)])],
                [],
            )
        self.assertIn("inside an obstacle", str(context.exception))

    def test_checkpoint_obstacle_on_the_base_map_is_still_rejected(self) -> None:
        # An obstacle present on the *initial* map cannot be explained away by
        # an edit history. Both maps carry the cell and no edit touches it, so
        # the history still reproduces the saved grid, but the taskless route
        # through it must be rejected just like direct construction.
        document = document_for(waiting_simulator(walk_first=False))
        document["base_grid"]["obstacles"] = [[2, 0]]
        document["map_changes"] = []
        document["replay"] = [
            frame for frame in document["replay"] if frame.get("type") != "map_change"
        ]
        expect_rejected(self, document, "R", "[2, 0]", "inside an obstacle")

    def test_version_one_file_keeps_the_strict_obstacle_rule(self) -> None:
        document = document_for(waiting_simulator(walk_first=False))
        document["version"] = 1
        for key in ("base_grid", "map_changes", "paused_tasks", "traffic_waits"):
            document.pop(key, None)
        document["replay"] = [
            {key: value for key, value in frame.items() if key != "type"}
            for frame in document["replay"]
            if frame.get("type") == "tick"
        ]
        expect_rejected(self, document, "inside an obstacle")

    def test_task_bound_route_through_a_current_obstacle_is_rejected(self) -> None:
        # Runtime pauses the task and clears the route; hand the saved file a
        # bound robot that still drives through the closed cell instead -- the
        # waiting exemption is for taskless robots only.
        simulator = FleetSimulator(
            GridMap(4, 1),
            [
                Robot(
                    "R",
                    (0, 0),
                    route=[(1, 0), (2, 0), (3, 0)],
                    task_id="T",
                )
            ],
            [Task("T", (0, 0), (3, 0), assigned_robot="R", picked_up=True)],
        )
        simulator.modify_obstacles(added=[(2, 0)])
        self.assertEqual(simulator.paused_tasks, {"T"})
        document = document_for(simulator)
        document["robots"][0]["route"] = [[1, 0], [2, 0], [3, 0]]
        document["paused_tasks"] = []
        expect_rejected(self, document, "inside an obstacle")


class ExemptionNeverWeakensShapeRulesTests(unittest.TestCase):
    def document(self) -> dict:
        return document_for(waiting_simulator(walk_first=False))

    def test_robot_position_on_the_closed_cell_rejected(self) -> None:
        document = self.document()
        document["robots"][0]["position"] = [2, 0]
        document["robots"][0]["route"] = [[3, 0]]
        expect_rejected(
            self, document, "position", "outside the map or inside an obstacle"
        )

    def test_out_of_bounds_blocked_waypoint_rejected(self) -> None:
        document = self.document()
        document["robots"][0]["route"][2] = [4, 0]
        expect_rejected(self, document, "outside the map")

    def test_non_integer_blocked_waypoint_rejected(self) -> None:
        document = self.document()
        document["robots"][0]["route"][2] = [3.0, 0]
        expect_rejected(self, document, "must be a [x, y] integer pair")
        document = self.document()
        document["robots"][0]["route"][2] = [True, 0]
        expect_rejected(self, document, "must be a [x, y] integer pair")

    def test_jump_between_waypoints_rejected(self) -> None:
        document = self.document()
        document["robots"][0]["route"] = [[1, 0], [3, 0]]
        expect_rejected(self, document, "non-adjacently")

    def test_first_waypoint_jump_from_position_rejected(self) -> None:
        document = self.document()
        document["robots"][0]["route"] = [[2, 0], [3, 0]]
        expect_rejected(self, document, "non-adjacently")


if __name__ == "__main__":
    unittest.main()
