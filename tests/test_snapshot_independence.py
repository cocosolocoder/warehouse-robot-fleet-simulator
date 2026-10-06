"""Regression guarantees for the independence of ``FleetSimulator.snapshot()``.

A snapshot is a view of one moment only: it must keep representing the tick at
which it was taken while the fleet keeps running, and trimming or rewriting the
returned structure must never reach the live simulator. These tests pin down
that contract for robots, tasks, metrics and replay, including the empty,
never-stepped case.
"""

import copy
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


def carrying_fleet() -> FleetSimulator:
    """Two robots carrying two in-progress tasks on a two-row grid.

    After two ticks both robots have collected their goods and still carry a
    remaining route: R-01 drives the top row towards (7, 0), R-02 the bottom
    row towards (3, 1). The second row leaves spare cells for obstacle edits
    that never touch either route.
    """
    return FleetSimulator(
        GridMap(8, 2),
        [Robot("R-01", (0, 0)), Robot("R-02", (5, 1))],
        [Task("T-01", (2, 0), (7, 0)), Task("T-02", (6, 1), (3, 1))],
    )


def fleet_at_carrying_moment() -> FleetSimulator:
    """The shared scenario advanced to the tick at which snapshots are taken."""
    simulator = carrying_fleet()
    simulator.step()
    simulator.step()
    # An effective map edit away from either route, so the snapshot already
    # contains a map_change event alongside its tick frames.
    simulator.modify_obstacles(added=[(0, 1)])
    return simulator


def _event_timeline(replay):
    return [(event["type"], event.get("tick"), event.get("sequence")) for event in replay]


class SnapshotFreezesMomentTests(unittest.TestCase):
    def test_old_snapshot_keeps_tick_robots_tasks_mileage_and_replay(self) -> None:
        simulator = fleet_at_carrying_moment()
        snapshot = simulator.snapshot()
        expected = copy.deepcopy(snapshot)

        # Close a cell R-01 already passed through, then drive both tasks home.
        simulator.modify_obstacles(added=[(1, 0)])
        for _ in range(5):
            simulator.step()

        # The live fleet moved on: R-01 delivered, five more ticks elapsed.
        self.assertEqual(simulator.tick, 7)
        self.assertEqual(simulator.robots["R-01"].position, (7, 0))
        self.assertTrue(simulator.tasks["T-01"].completed)

        # The handed-out result still describes the moment it was taken.
        self.assertEqual(snapshot, expected)
        self.assertEqual(snapshot["tick"], 2)
        self.assertEqual([robot["robot_id"] for robot in snapshot["robots"]], ["R-01", "R-02"])
        r01 = snapshot["robots"][0]
        self.assertEqual(r01["position"], (2, 0))
        self.assertEqual(
            r01["route"], [(3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]
        )
        self.assertEqual(r01["task_id"], "T-01")
        self.assertEqual(r01["distance_travelled"], 2)
        r02 = snapshot["robots"][1]
        self.assertEqual(r02["position"], (5, 1))
        self.assertEqual(r02["route"], [(4, 1), (3, 1)])
        self.assertEqual(r02["distance_travelled"], 2)
        for task in snapshot["tasks"]:
            self.assertTrue(task["picked_up"])
            self.assertFalse(task["completed"])
            self.assertIsNotNone(task["assigned_robot"])
        self.assertEqual(snapshot["metrics"]["ticks"], 2)
        self.assertEqual(snapshot["metrics"]["distance_total"], 4)
        self.assertEqual(snapshot["metrics"]["tasks_completed"], 0)

    def test_historical_replay_frames_are_not_rewritten_by_later_maps(self) -> None:
        simulator = fleet_at_carrying_moment()
        snapshot = simulator.snapshot()

        # R-01's tick-1 frame records it on (1, 0); closing that cell later must
        # not rewrite the past position into something the current map allows.
        simulator.modify_obstacles(added=[(1, 0)])
        simulator.step()

        self.assertEqual(
            _event_timeline(snapshot["replay"]),
            [("tick", 1, None), ("tick", 2, None), ("map_change", 2, 1)],
        )
        self.assertEqual(snapshot["replay"][0]["robots"]["R-01"], [1, 0])
        self.assertEqual(snapshot["replay"][1]["robots"]["R-01"], [2, 0])
        map_event = snapshot["replay"][2]
        self.assertEqual(map_event["added"], [[0, 1]])
        self.assertEqual(map_event["removed"], [])
        self.assertEqual(len(snapshot["replay"]), 3)

    def test_later_movement_completion_and_edits_only_appear_in_new_snapshots(self) -> None:
        simulator = fleet_at_carrying_moment()
        old = simulator.snapshot()

        simulator.modify_obstacles(added=[(1, 0)])
        for _ in range(5):
            simulator.step()
        new = simulator.snapshot()

        self.assertEqual(old["tick"], 2)
        self.assertEqual(len(old["replay"]), 3)
        self.assertNotIn(("map_change", 2, 2), _event_timeline(old["replay"]))
        self.assertNotIn(("tick", 3, None), _event_timeline(old["replay"]))

        self.assertEqual(new["tick"], 7)
        self.assertEqual(
            _event_timeline(new["replay"]),
            [
                ("tick", 1, None),
                ("tick", 2, None),
                ("map_change", 2, 1),
                ("map_change", 2, 2),
                ("tick", 3, None),
                ("tick", 4, None),
                ("tick", 5, None),
                ("tick", 6, None),
                ("tick", 7, None),
            ],
        )
        second_edit = next(
            event
            for event in new["replay"]
            if event["type"] == "map_change" and event["sequence"] == 2
        )
        self.assertEqual(second_edit["added"], [[1, 0]])
        self.assertEqual(new["replay"][-1]["robots"]["R-01"], [7, 0])
        self.assertEqual(new["metrics"]["tasks_completed"], 2)
        self.assertEqual(new["metrics"]["distance_total"], 11)

    def test_taking_a_snapshot_has_no_side_effects(self) -> None:
        # A pending task must stay unassigned and nothing may move or accrue
        # history merely because the state was read.
        simulator = FleetSimulator(
            GridMap(4, 1), [Robot("R", (0, 0))], [Task("T", (1, 0), (3, 0))]
        )
        before = copy.deepcopy(simulator.snapshot())
        simulator.snapshot()
        simulator.snapshot()
        self.assertEqual(simulator.tick, 0)
        self.assertIsNone(simulator.robots["R"].task_id)
        self.assertIsNone(simulator.tasks["T"].assigned_robot)
        self.assertEqual(simulator.robots["R"].position, (0, 0))
        self.assertEqual(simulator.robots["R"].distance_travelled, 0)
        self.assertEqual(simulator.replay, [])
        self.assertEqual(simulator.map_change_history(), [])
        self.assertEqual(simulator.snapshot(), before)

        # The same holds mid-run, with robots, routes and history present.
        running = fleet_at_carrying_moment()
        running_before = copy.deepcopy(running.snapshot())
        replay_length = len(running.replay)
        running.snapshot()
        self.assertEqual(running.tick, 2)
        self.assertEqual(len(running.replay), replay_length)
        self.assertEqual(running.snapshot(), running_before)


class SnapshotMutationIsolationTests(unittest.TestCase):
    def test_mutating_returned_robots_tasks_and_routes_never_reaches_simulator(self) -> None:
        simulator = fleet_at_carrying_moment()
        snapshot = simulator.snapshot()

        snapshot["robots"].pop(0)
        remaining = snapshot["robots"][0]
        remaining["position"] = (0, 0)
        remaining["route"].append((9, 9))
        remaining["route"][0] = (0, 1)
        remaining["task_id"] = None
        remaining["distance_travelled"] = 999
        snapshot["tasks"].pop()
        snapshot["tasks"][0]["assigned_robot"] = None
        snapshot["tasks"][0]["picked_up"] = False
        snapshot["tasks"][0]["completed"] = True
        snapshot["metrics"]["distance_total"] = -1

        live_r01 = simulator.robots["R-01"]
        self.assertEqual(live_r01.position, (2, 0))
        self.assertEqual(
            live_r01.route, [(3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]
        )
        self.assertEqual(live_r01.task_id, "T-01")
        self.assertEqual(live_r01.distance_travelled, 2)
        self.assertEqual(simulator.robots["R-02"].route, [(4, 1), (3, 1)])
        self.assertEqual(len(simulator.tasks), 2)
        task = simulator.tasks["T-01"]
        self.assertEqual(task.assigned_robot, "R-01")
        self.assertTrue(task.picked_up)
        self.assertFalse(task.completed)
        self.assertEqual(simulator.metrics()["distance_total"], 4)

    def test_rewriting_returned_replay_events_never_touches_recorded_history(self) -> None:
        simulator = fleet_at_carrying_moment()
        snapshot = simulator.snapshot()

        snapshot["replay"].pop(0)
        tick_frame = snapshot["replay"][0]
        self.assertEqual(tick_frame["type"], "tick")
        tick_frame["robots"]["R-01"] = [6, 6]
        tick_frame["moved"].append("GHOST")
        tick_frame["completed"].append("T-99")
        map_event = snapshot["replay"][1]
        self.assertEqual(map_event["type"], "map_change")
        map_event["added"].append([5, 5])
        map_event["removed"] = [[0, 1]]

        live_tick1 = simulator.replay[0]
        self.assertEqual(live_tick1["robots"]["R-01"], [1, 0])
        self.assertEqual(live_tick1["moved"], ["R-01", "R-02"])
        self.assertEqual(live_tick1["completed"], [])
        live_edit = simulator.replay[2]
        self.assertEqual(live_edit["added"], [[0, 1]])
        self.assertEqual(live_edit["removed"], [])
        self.assertEqual(len(simulator.replay), 3)

    def test_fleet_runs_on_identically_after_returned_result_was_rewritten(self) -> None:
        mutated = fleet_at_carrying_moment()
        control = fleet_at_carrying_moment()
        snapshot = mutated.snapshot()

        # Damage every mutable part of the returned result before continuing.
        snapshot["robots"].pop(0)
        snapshot["robots"][0]["route"].append((9, 9))
        snapshot["robots"][0]["distance_travelled"] = 999
        snapshot["tasks"].pop()
        snapshot["tasks"][0]["picked_up"] = False
        snapshot["replay"].pop(0)
        snapshot["replay"][0]["robots"]["R-01"] = [6, 6]
        for event in snapshot["replay"]:
            if event["type"] == "map_change":
                event["added"].append([5, 5])

        for _ in range(5):
            mutated.step()
            control.step()

        # Pickup, delivery, mileage and history match a fleet whose snapshot
        # nobody ever touched.
        self.assertEqual(mutated.snapshot(), control.snapshot())
        self.assertEqual(mutated.replay, control.replay)
        self.assertTrue(mutated.tasks["T-01"].completed)
        self.assertTrue(mutated.tasks["T-02"].completed)
        self.assertEqual(mutated.robots["R-01"].distance_travelled, 7)
        self.assertEqual(mutated.robots["R-02"].distance_travelled, 4)


class SnapshotPeerIndependenceTests(unittest.TestCase):
    def test_two_snapshots_of_one_moment_are_independent_of_each_other(self) -> None:
        simulator = fleet_at_carrying_moment()
        first = simulator.snapshot()
        second = simulator.snapshot()
        first_kept = copy.deepcopy(first)

        first["tick"] = 99
        first["robots"].pop(0)
        first["robots"][0]["route"].append((9, 9))
        first["tasks"][0]["completed"] = True
        first["replay"].pop(0)
        first["replay"][0]["robots"]["R-01"] = [6, 6]

        # The sibling still describes the shared moment in full.
        self.assertEqual(second, first_kept)
        self.assertEqual(second["tick"], 2)
        self.assertEqual(len(second["robots"]), 2)
        self.assertEqual(
            second["robots"][0]["route"], [(3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]
        )
        self.assertFalse(second["tasks"][0]["completed"])
        self.assertEqual(len(second["replay"]), 3)
        self.assertEqual(second["replay"][0]["robots"]["R-01"], [1, 0])

        # A later snapshot reports the real simulator, not either tampered copy.
        simulator.modify_obstacles(added=[(1, 0)])
        for _ in range(5):
            simulator.step()
        later = simulator.snapshot()
        self.assertEqual(later["tick"], 7)
        self.assertEqual(later["robots"][0]["position"], (7, 0))
        self.assertEqual(later["metrics"]["tasks_completed"], 2)
        self.assertEqual(second, first_kept)


class EmptyAndInitialSnapshotTests(unittest.TestCase):
    def test_empty_fleet_empty_tasks_empty_replay(self) -> None:
        simulator = FleetSimulator(GridMap(3, 3), [], [])
        snapshot = simulator.snapshot()

        self.assertEqual(
            sorted(snapshot), ["metrics", "replay", "robots", "tasks", "tick"]
        )
        self.assertEqual(snapshot["tick"], 0)
        self.assertEqual(snapshot["robots"], [])
        self.assertEqual(snapshot["tasks"], [])
        self.assertEqual(snapshot["replay"], [])
        self.assertEqual(snapshot["metrics"]["tasks_total"], 0)
        self.assertEqual(snapshot["metrics"]["completion_ratio"], 1.0)

        # Reading and rewriting the empty result changes nothing about later
        # simulation, including the all-waiting tick frame an empty fleet emits.
        snapshot["tick"] = 5
        snapshot["robots"].append({"robot_id": "FAKE"})
        snapshot["tasks"].append({"task_id": "FAKE"})
        snapshot["replay"].append({"type": "tick", "tick": 1})
        simulator.step()
        self.assertEqual(simulator.tick, 1)
        self.assertEqual(
            simulator.replay[-1],
            {"type": "tick", "tick": 1, "moved": [], "robots": {}, "completed": []},
        )
        fresh = simulator.snapshot()
        self.assertEqual(fresh["robots"], [])
        self.assertEqual(fresh["tasks"], [])
        self.assertEqual(len(fresh["replay"]), 1)

    def test_snapshot_before_any_step_can_be_rewritten_without_breaking_run(self) -> None:
        def fresh_corridor() -> FleetSimulator:
            return FleetSimulator(
                GridMap(4, 1),
                [Robot("R", (0, 0))],
                [Task("T", (1, 0), (3, 0))],
            )

        mutated = fresh_corridor()
        control = fresh_corridor()
        initial = mutated.snapshot()
        self.assertEqual(initial["tick"], 0)
        self.assertEqual(initial["replay"], [])
        self.assertIsNone(initial["tasks"][0]["assigned_robot"])

        initial["robots"][0]["position"] = (3, 0)
        initial["robots"][0]["route"] = [(9, 9)]
        initial["robots"][0]["distance_travelled"] = 42
        initial["tasks"][0]["picked_up"] = True
        initial["tasks"][0]["completed"] = True
        initial["replay"].append({"type": "tick", "tick": 1})

        for _ in range(3):
            mutated.step()
            control.step()

        # The real run assigns, picks up, delivers and records history exactly
        # as if the returned result had never been touched.
        self.assertEqual(mutated.snapshot(), control.snapshot())
        self.assertTrue(mutated.tasks["T"].completed)
        self.assertEqual(mutated.robots["R"].position, (3, 0))
        self.assertEqual(mutated.robots["R"].distance_travelled, 3)
        self.assertEqual([frame["tick"] for frame in mutated.replay], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
