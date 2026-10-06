"""Regression guarantees for the independence of ``snapshot()`` results.

A snapshot handed to a caller must keep representing the moment it was taken:
later ticks, task completions and map edits (including closing a cell a robot
once passed through) never reach an older snapshot, and nothing the caller
changes inside the returned robots, tasks, metrics or replay ever reaches the
running fleet. Two snapshots taken at the same moment are independent of each
other, and the guarantees already hold before any step has run -- including an
empty fleet with empty tasks and an empty replay.
"""

import copy
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


def mid_run_simulator() -> FleetSimulator:
    """Two robots mid-delivery, each carrying goods with a route still ahead.

    After construction the caller runs three ticks and one effective map
    edit (closing ``(0, 1)``); at that point both tasks are picked up but
    unfinished, both robots carry a non-empty remaining route, and the replay
    already holds three tick frames plus one map-change record.
    """
    return FleetSimulator(
        GridMap(8, 3),
        [Robot("A", (0, 0)), Robot("B", (7, 2))],
        [Task("T-1", (1, 0), (6, 0)), Task("T-2", (6, 2), (1, 2))],
    )


def advance_to_midpoint(simulator: FleetSimulator) -> None:
    for _ in range(3):
        simulator.step()
    # An effective edit already present in the capture-time replay/history.
    simulator.modify_obstacles(added=[(0, 1)])


def continue_after_midpoint(simulator: FleetSimulator) -> None:
    # Close a cell robot A occupied back on tick 1, then drive both tasks to
    # completion. The closure must not rewrite any past replay frame.
    simulator.modify_obstacles(added=[(1, 0)])
    for _ in range(4):
        simulator.step()


def vandalize(snapshot: dict[str, object]) -> None:
    """Mutate every mutable part of a snapshot the way a display caller might.

    Robot and task records are dropped, the remaining robot's position, route
    and mileage and the remaining task's status/coordinates are overwritten,
    replay frames are deleted and rewritten (including a tick frame's position
    and a map-change frame's obstacle coordinates), and the nested metrics
    lists are extended. None of it may ever reach the simulator.
    """
    robots = snapshot["robots"]
    tasks = snapshot["tasks"]
    # Drop a second record when the fleet has one; the remaining record is then
    # rewritten. (A one-record fleet still exercises every mutation path.)
    if len(robots) > 1:
        robots.pop(1)
    if len(tasks) > 1:
        tasks.pop(1)
    robot = robots[0]
    robot["position"] = (9, 9)
    if robot["route"]:
        robot["route"][0] = (7, 7)
    robot["route"].append((8, 8))
    robot["task_id"] = "BORROWED"
    robot["distance_travelled"] = 999
    task = tasks[0]
    task["pickup"] = (9, 9)
    task["dropoff"] = (8, 8)
    task["assigned_robot"] = None
    task["picked_up"] = False
    task["completed"] = True
    metrics = snapshot["metrics"]
    metrics["tasks_paused"].append("PHANTOM")
    metrics["traffic_waits"].append({"robot_id": "GHOST", "blocked_by": ["A"], "ticks": 5})
    metrics["distance_total"] = -7
    replay = snapshot["replay"]
    if replay:
        replay.pop(0)
    # Rewrite a position recorded in an existing tick frame.
    tick_frame = next(
        (frame for frame in replay if frame["type"] == "tick"), None
    )
    if tick_frame is not None:
        if "A" in tick_frame["robots"]:
            tick_frame["robots"]["A"] = [9, 9]
        tick_frame["moved"].append("GHOST")
        tick_frame["completed"].append("PHANTOM")
    # Rewrite obstacle coordinates in an existing map-change frame.
    change_frame = next(
        (frame for frame in replay if frame["type"] == "map_change"), None
    )
    if change_frame is not None:
        change_frame["added"][0] = [8, 8]
        change_frame["added"].append([9, 9])
        change_frame["removed"].append([7, 7])
    replay.append({"type": "tick", "tick": 999, "moved": [], "robots": {}, "completed": []})


class SnapshotFormatTests(unittest.TestCase):
    def test_snapshot_keeps_existing_field_set_and_ordering(self) -> None:
        simulator = mid_run_simulator()
        advance_to_midpoint(simulator)
        # No new snapshot format is introduced.
        snap = simulator.snapshot()
        self.assertEqual(
            sorted(snap),
            ["metrics", "replay", "robots", "tasks", "tick"],
        )
        self.assertEqual([robot["robot_id"] for robot in snap["robots"]], ["A", "B"])
        self.assertEqual([task["task_id"] for task in snap["tasks"]], ["T-1", "T-2"])


class SnapshotFreezeTests(unittest.TestCase):
    """An older snapshot always represents the moment it was captured."""

    def test_capture_time_values_survive_later_steps_and_map_changes(self) -> None:
        simulator = mid_run_simulator()
        advance_to_midpoint(simulator)
        snapshot = simulator.snapshot()
        frozen = copy.deepcopy(snapshot)

        self.assertEqual(snapshot["tick"], 3)
        self.assertEqual(
            snapshot["robots"],
            [
                {
                    "robot_id": "A",
                    "position": (3, 0),
                    "route": [(4, 0), (5, 0), (6, 0)],
                    "task_id": "T-1",
                    "distance_travelled": 3,
                },
                {
                    "robot_id": "B",
                    "position": (4, 2),
                    "route": [(3, 2), (2, 2), (1, 2)],
                    "task_id": "T-2",
                    "distance_travelled": 3,
                },
            ],
        )
        self.assertEqual(
            [(t["task_id"], t["picked_up"], t["completed"]) for t in snapshot["tasks"]],
            [("T-1", True, False), ("T-2", True, False)],
        )
        self.assertEqual(
            snapshot["metrics"]["distance_total"], 6
        )
        self.assertEqual(snapshot["metrics"]["tasks_completed"], 0)

        continue_after_midpoint(simulator)
        later = simulator.snapshot()

        # The old snapshot did not follow the simulation at all.
        self.assertEqual(snapshot, frozen)
        self.assertEqual(snapshot["tick"], 3)
        self.assertEqual(snapshot["robots"][0]["position"], (3, 0))
        self.assertEqual(snapshot["robots"][0]["route"], [(4, 0), (5, 0), (6, 0)])
        self.assertEqual(snapshot["robots"][0]["distance_travelled"], 3)
        self.assertFalse(snapshot["tasks"][0]["completed"])
        self.assertTrue(snapshot["tasks"][0]["picked_up"])
        self.assertEqual(snapshot["metrics"]["distance_total"], 6)

        # Movement, completions and the new edit appear only in a later capture.
        self.assertEqual(later["tick"], 7)
        self.assertEqual(later["robots"][0]["position"], (6, 0))
        self.assertEqual(later["robots"][0]["route"], [])
        self.assertIsNone(later["robots"][0]["task_id"])
        self.assertEqual(later["robots"][0]["distance_travelled"], 6)
        self.assertTrue(all(task["completed"] for task in later["tasks"]))
        self.assertEqual(later["metrics"]["distance_total"], 12)
        self.assertEqual(later["metrics"]["tasks_completed"], 2)

    def test_replay_keeps_past_events_content_and_order_including_closed_cells(self) -> None:
        simulator = mid_run_simulator()
        advance_to_midpoint(simulator)
        snapshot = simulator.snapshot()
        frozen_replay = copy.deepcopy(snapshot["replay"])

        self.assertEqual(
            [
                (frame["type"], frame.get("tick"), frame.get("sequence"))
                for frame in snapshot["replay"]
            ],
            [("tick", 1, None), ("tick", 2, None), ("tick", 3, None),
             ("map_change", 3, 1)],
        )
        # On tick 1 robot A ended on (1, 0), a cell that is closed afterwards.
        self.assertEqual(snapshot["replay"][0]["robots"]["A"], [1, 0])

        continue_after_midpoint(simulator)
        later = simulator.snapshot()

        # The old replay is byte-for-byte the capture-time history.
        self.assertEqual(snapshot["replay"], frozen_replay)
        self.assertEqual(len(snapshot["replay"]), 4)
        # The historical position is not re-judged under the current map.
        self.assertEqual(snapshot["replay"][0]["robots"]["A"], [1, 0])
        self.assertEqual(
            [frame["type"] for frame in snapshot["replay"]],
            ["tick", "tick", "tick", "map_change"],
        )

        # The newer snapshot keeps the shared prefix and only then adds the
        # later closure and tick frames.
        self.assertEqual(later["replay"][:4], frozen_replay)
        self.assertEqual(
            [
                (frame["type"], frame.get("tick"), frame.get("sequence"))
                for frame in later["replay"]
            ],
            [
                ("tick", 1, None), ("tick", 2, None), ("tick", 3, None),
                ("map_change", 3, 1), ("map_change", 3, 2),
                ("tick", 4, None), ("tick", 5, None),
                ("tick", 6, None), ("tick", 7, None),
            ],
        )
        self.assertEqual(later["replay"][4]["added"], [[1, 0]])
        self.assertNotIn(
            [1, 0],
            [cell for frame in snapshot["replay"] if frame["type"] == "map_change"
             for cell in frame["added"]],
        )

    def test_taking_a_snapshot_changes_nothing(self) -> None:
        simulator = mid_run_simulator()
        # Before any step: taking a snapshot must not assign the waiting tasks.
        self.assertTrue(all(task.assigned_robot is None for task in simulator.tasks.values()))
        zero = simulator.snapshot()
        self.assertEqual(simulator.tick, 0)
        self.assertEqual(simulator.replay, [])
        self.assertTrue(all(task.assigned_robot is None for task in simulator.tasks.values()))
        self.assertEqual(zero["tick"], 0)
        self.assertEqual(zero["replay"], [])

        advance_to_midpoint(simulator)
        before = copy.deepcopy(simulator.snapshot())
        history_before = simulator.map_change_history()
        replay_len_before = len(simulator.replay)
        # Querying repeatedly only reads.
        for _ in range(5):
            simulator.snapshot()
        self.assertEqual(simulator.snapshot(), before)
        self.assertEqual(simulator.tick, before["tick"])
        self.assertEqual(len(simulator.replay), replay_len_before)
        self.assertEqual(simulator.map_change_history(), history_before)


class SnapshotMutationIsolationTests(unittest.TestCase):
    """Edits to a returned snapshot never flow back into the simulator."""

    def test_mutating_returned_robots_tasks_routes_and_replay_stays_local(self) -> None:
        victim = mid_run_simulator()
        control = mid_run_simulator()
        for simulator in (victim, control):
            advance_to_midpoint(simulator)

        vandalize(victim.snapshot())

        # Internal simulator state is untouched in every shape the mutation hit.
        self.assertEqual(victim.snapshot(), control.snapshot())
        self.assertEqual(victim.robots["B"].position, (4, 2))
        self.assertEqual(victim.robots["B"].route, [(3, 2), (2, 2), (1, 2)])
        self.assertEqual(victim.robots["B"].task_id, "T-2")
        self.assertEqual(victim.robots["B"].distance_travelled, 3)
        self.assertIn("A", victim.robots)
        self.assertIn("T-1", victim.tasks)
        self.assertFalse(victim.tasks["T-2"].completed)
        self.assertTrue(victim.tasks["T-2"].picked_up)
        self.assertEqual(victim.tasks["T-2"].assigned_robot, "B")
        self.assertEqual(victim.tasks["T-2"].pickup, (6, 2))
        self.assertEqual(victim.map_change_history(), control.map_change_history())
        first_frame = victim.replay[0]
        self.assertEqual(first_frame["robots"], {"A": [1, 0], "B": [6, 2]})
        self.assertEqual(first_frame["moved"], ["A", "B"])
        self.assertEqual(first_frame["completed"], [])
        change = victim.replay[-1]
        self.assertEqual(change["added"], [[0, 1]])
        self.assertEqual(change["removed"], [])
        self.assertEqual(len(victim.replay), 4)
        self.assertEqual(victim.metrics()["tasks_paused"], [])
        self.assertEqual(victim.metrics()["traffic_waits"], [])
        self.assertEqual(victim.metrics()["distance_total"], 6)

    def test_fleet_runs_on_unchanged_after_snapshot_mutation(self) -> None:
        victim = mid_run_simulator()
        control = mid_run_simulator()
        for simulator in (victim, control):
            advance_to_midpoint(simulator)

        # Mutate both a capture taken mid-run...
        vandalize(victim.snapshot())
        continue_after_midpoint(victim)
        continue_after_midpoint(control)

        # ...and another one taken after more movement, then keep stepping.
        def rest(simulator: FleetSimulator) -> None:
            simulator.modify_obstacles(removed=[(1, 0)])
            for _ in range(2):
                simulator.step()

        later_victim = victim.snapshot()
        vandalize(later_victim)
        rest(victim)
        rest(control)

        # Pickup, delivery, mileage, replay and map history are exactly what an
        # undisturbed run produces; corruption never waits for the next query.
        self.assertEqual(victim.snapshot(), control.snapshot())
        self.assertEqual(victim.metrics(), control.metrics())
        self.assertEqual(
            victim.map_change_history(), control.map_change_history()
        )
        self.assertEqual(victim.replay, control.replay)
        self.assertTrue(all(task.completed for task in victim.tasks.values()))
        self.assertEqual(
            {robot.robot_id: robot.distance_travelled for robot in victim.robots.values()},
            {"A": 6, "B": 6},
        )
        # The second vandalized copy still carries its local garbage.
        self.assertNotEqual(later_victim, victim.snapshot())


class SameMomentSnapshotIndependenceTests(unittest.TestCase):
    def test_same_moment_captures_are_independent_and_later_ones_track_truth(self) -> None:
        simulator = mid_run_simulator()
        advance_to_midpoint(simulator)
        first = simulator.snapshot()
        second = simulator.snapshot()
        self.assertEqual(first, second)

        frozen_second = copy.deepcopy(second)
        vandalize(first)

        # The untouched twin still fully represents that moment.
        self.assertEqual(second, frozen_second)
        self.assertEqual(second, simulator.snapshot())
        self.assertEqual(len(second["robots"]), 2)
        self.assertEqual(len(second["tasks"]), 2)
        self.assertEqual(len(second["replay"]), 4)

        continue_after_midpoint(simulator)
        newer = simulator.snapshot()
        # The later capture reflects the simulator's real new state...
        self.assertEqual(newer["tick"], 7)
        self.assertTrue(all(task["completed"] for task in newer["tasks"]))
        # ...while neither older capture moved.
        self.assertEqual(second, frozen_second)
        self.assertEqual(second["tick"], 3)
        self.assertNotEqual(first, second)


class ZeroStepSnapshotTests(unittest.TestCase):
    def test_empty_fleet_empty_tasks_empty_replay(self) -> None:
        simulator = FleetSimulator(GridMap(4, 4), [], [])
        snapshot = simulator.snapshot()
        self.assertEqual(snapshot["tick"], 0)
        self.assertEqual(snapshot["robots"], [])
        self.assertEqual(snapshot["tasks"], [])
        self.assertEqual(snapshot["replay"], [])
        self.assertEqual(
            snapshot["metrics"],
            {
                "ticks": 0,
                "tasks_total": 0,
                "tasks_completed": 0,
                "tasks_paused": [],
                "completion_ratio": 1.0,
                "distance_total": 0,
                "traffic_waits": [],
            },
        )

        snapshot["robots"].append({"robot_id": "FAKE"})
        snapshot["tasks"].append({"task_id": "FAKE"})
        snapshot["replay"].append({"type": "tick", "tick": 1})
        snapshot["metrics"]["tasks_paused"].append("FAKE")
        snapshot["metrics"]["traffic_waits"].append({"robot_id": "FAKE"})

        for _ in range(3):
            simulator.step()
        self.assertEqual(simulator.tick, 3)
        self.assertEqual(list(simulator.robots), [])
        self.assertEqual(list(simulator.tasks), [])
        # Only the three real ticks were ever recorded.
        self.assertEqual(len(simulator.replay), 3)
        self.assertEqual([frame["tick"] for frame in simulator.replay], [1, 2, 3])
        self.assertTrue(all(frame["robots"] == {} for frame in simulator.replay))
        self.assertEqual(simulator.metrics()["distance_total"], 0)
        self.assertEqual(simulator.metrics()["tasks_paused"], [])
        self.assertEqual(simulator.metrics()["traffic_waits"], [])

    def test_mutating_zero_step_snapshot_does_not_disturb_later_run(self) -> None:
        def fresh() -> FleetSimulator:
            return FleetSimulator(
                GridMap(6, 1),
                [Robot("A", (0, 0))],
                [Task("T-1", (2, 0), (4, 0))],
            )

        victim = fresh()
        control = fresh()
        snapshot = victim.snapshot()
        vandalize(snapshot)

        # The waiting task still assigns and runs normally from tick 1 on.
        self.assertIsNone(victim.tasks["T-1"].assigned_robot)
        for simulator in (victim, control):
            for _ in range(4):
                simulator.step()

        self.assertEqual(victim.snapshot(), control.snapshot())
        self.assertEqual(victim.metrics(), control.metrics())
        self.assertEqual(victim.replay, control.replay)
        self.assertTrue(victim.tasks["T-1"].completed)
        self.assertTrue(victim.tasks["T-1"].picked_up)
        self.assertEqual(victim.robots["A"].position, (4, 0))
        self.assertEqual(victim.robots["A"].distance_travelled, 4)


if __name__ == "__main__":
    unittest.main()
