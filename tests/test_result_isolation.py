"""Regression tests for the independence of query results from simulator state.

``step()`` frames and ``snapshot()`` results belong to the caller: deleting
replay entries, rewriting coordinates, or editing the moved/completed and
added/removed lists of a returned result must never rewrite the simulator's
real positions, task states, clock, recorded replay or map-change history.
Every fetched result is also independent of every other result, and querying
never advances the simulation.
"""

import os
import tempfile
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


def one_robot_simulator(width: int = 6) -> FleetSimulator:
    return FleetSimulator(
        GridMap(width, 1),
        [Robot("A", (0, 0))],
        [Task("T-1", (1, 0), (width - 2, 0))],
    )


class StepFrameIsolationTests(unittest.TestCase):
    def test_returned_frame_edit_does_not_rewrite_recorded_history(self) -> None:
        simulator = one_robot_simulator()
        frame = simulator.step()
        frame["robots"]["A"][0] = 99
        frame["moved"].append("GHOST")
        frame["completed"].append("T-1")

        recorded = simulator.replay[0]
        self.assertIsNot(recorded, frame)
        self.assertEqual(recorded["robots"], {"A": [1, 0]})
        self.assertEqual(recorded["moved"], ["A"])
        self.assertEqual(recorded["completed"], [])

    def test_nested_frame_lists_are_independent_copies(self) -> None:
        simulator = one_robot_simulator()
        frame = simulator.step()
        frame["robots"]["A"].reverse()
        frame["moved"].reverse()
        frame["completed"].reverse()
        self.assertEqual(simulator.replay[0]["robots"]["A"], [1, 0])

    def test_returned_frame_matches_recorded_frame_by_value(self) -> None:
        simulator = one_robot_simulator()
        frame = simulator.step()
        self.assertEqual(frame, simulator.replay[-1])
        self.assertIsNot(frame, simulator.replay[-1])


class SnapshotGrowthTests(unittest.TestCase):
    def test_old_snapshot_replay_does_not_grow_with_later_steps(self) -> None:
        simulator = one_robot_simulator()
        simulator.step()
        old = simulator.snapshot()
        self.assertEqual(old["tick"], 1)
        simulator.step()
        simulator.step()
        self.assertEqual(len(old["replay"]), 1)
        self.assertEqual(old["tick"], 1)
        self.assertEqual(old["robots"][0]["position"], (1, 0))
        self.assertEqual(len(simulator.replay), 3)

    def test_old_snapshot_replay_does_not_gain_map_change_events(self) -> None:
        simulator = one_robot_simulator()
        simulator.step()
        old = simulator.snapshot()
        # The robot is at (1, 0); closing (5, 0) cannot touch its copy.
        simulator.modify_obstacles(added=[(5, 0)])
        simulator.step()
        self.assertEqual(len(old["replay"]), 1)
        self.assertEqual(old["tick"], 1)
        kinds = [frame["type"] for frame in simulator.replay]
        self.assertEqual(kinds, ["tick", "map_change", "tick"])

    def test_empty_replay_snapshot_at_tick_zero_stays_empty(self) -> None:
        simulator = FleetSimulator(GridMap(3, 3), [Robot("X", (0, 0))], [])
        empty = simulator.snapshot()
        self.assertEqual(empty["replay"], [])
        simulator.modify_obstacles(added=[(2, 2)])
        simulator.step()
        self.assertEqual(empty["replay"], [])
        self.assertEqual(empty["tick"], 0)
        self.assertEqual(len(simulator.replay), 2)


class SnapshotEditIsolationTests(unittest.TestCase):
    def test_editing_snapshot_replay_leaves_real_history_intact(self) -> None:
        simulator = one_robot_simulator()
        simulator.step()
        snapshot = simulator.snapshot()
        snapshot["replay"].clear()
        self.assertEqual(len(simulator.replay), 1)

        snapshot = simulator.snapshot()
        snapshot["replay"][0]["robots"]["A"] = [9, 9]
        snapshot["replay"][0]["moved"].append("GHOST")
        snapshot["replay"][0]["completed"].append("T-1")
        recorded = simulator.replay[0]
        self.assertEqual(recorded["robots"], {"A": [1, 0]})
        self.assertEqual(recorded["moved"], ["A"])
        self.assertEqual(recorded["completed"], [])

    def test_editing_snapshot_robots_and_tasks_leaves_state_intact(self) -> None:
        simulator = one_robot_simulator()
        simulator.step()
        snapshot = simulator.snapshot()
        snapshot["robots"][0]["position"] = (9, 9)
        snapshot["robots"][0]["route"].append((8, 8))
        snapshot["robots"][0]["distance_travelled"] = 50
        snapshot["tasks"][0]["assigned_robot"] = None
        snapshot["tasks"][0]["picked_up"] = False
        snapshot["tasks"][0]["completed"] = True
        self.assertEqual(simulator.robots["A"].position, (1, 0))
        self.assertEqual(simulator.robots["A"].distance_travelled, 1)
        self.assertFalse(simulator.tasks["T-1"].completed)
        self.assertTrue(simulator.tasks["T-1"].picked_up)
        self.assertEqual(simulator.tasks["T-1"].assigned_robot, "A")

    def test_map_change_coordinates_in_snapshot_are_independent(self) -> None:
        simulator = one_robot_simulator()
        simulator.step()
        simulator.modify_obstacles(added=[(5, 0)])
        snapshot = simulator.snapshot()
        change = next(frame for frame in snapshot["replay"] if frame["type"] == "map_change")
        change["added"].append([0, 0])
        change["removed"].append([1, 0])
        real_change = next(
            frame for frame in simulator.replay if frame["type"] == "map_change"
        )
        self.assertEqual(real_change["added"], [[5, 0]])
        self.assertEqual(real_change["removed"], [])
        self.assertNotIn((0, 0), simulator.grid.obstacles)
        self.assertIn((5, 0), simulator.grid.obstacles)
        self.assertEqual(
            [frame["added"] for frame in simulator.map_change_history()], [[[5, 0]]]
        )

    def test_mixed_replay_frame_kinds_have_independent_internals(self) -> None:
        simulator = one_robot_simulator()
        simulator.step()
        simulator.modify_obstacles(added=[(5, 0)])
        simulator.step()
        snapshot = simulator.snapshot()
        tick_frames = [f for f in snapshot["replay"] if f["type"] == "tick"]
        change_frames = [f for f in snapshot["replay"] if f["type"] == "map_change"]
        tick_frames[0]["robots"]["A"][0] = 40
        tick_frames[1]["moved"].reverse()
        change_frames[0]["added"][0][0] = 4
        change_frames[0]["removed"].append([2, 2])

        real_ticks = [f for f in simulator.replay if f["type"] == "tick"]
        real_change = next(f for f in simulator.replay if f["type"] == "map_change")
        self.assertEqual(real_ticks[0]["robots"]["A"], [1, 0])
        self.assertEqual(real_ticks[1]["moved"], ["A"])
        self.assertEqual(real_change["added"], [[5, 0]])
        self.assertEqual(real_change["removed"], [])


class ResultIndependenceTests(unittest.TestCase):
    def test_step_frame_and_later_snapshot_are_independent(self) -> None:
        simulator = one_robot_simulator()
        frame = simulator.step()
        snapshot = simulator.snapshot()
        # Edit the step frame: the snapshot copy of that step stays as taken.
        frame["robots"]["A"][0] = 50
        frame["moved"].append("GHOST")
        self.assertEqual(snapshot["replay"][0]["robots"]["A"], [1, 0])
        self.assertEqual(snapshot["replay"][0]["moved"], ["A"])
        # Edit the snapshot: the previously returned frame stays as returned.
        snapshot["replay"][0]["robots"]["A"][0] = 60
        self.assertEqual(frame["robots"]["A"], [50, 0])
        # And the real history matches neither mutation.
        self.assertEqual(simulator.replay[0]["robots"]["A"], [1, 0])

    def test_two_snapshots_of_same_moment_are_independent(self) -> None:
        simulator = one_robot_simulator()
        simulator.step()
        first = simulator.snapshot()
        second = simulator.snapshot()
        first["replay"].pop()
        first["replay"].append({"type": "forged"})
        first["robots"][0]["position"] = (9, 9)
        self.assertEqual(len(second["replay"]), 1)
        self.assertEqual(second["replay"][0]["type"], "tick")
        self.assertEqual(second["robots"][0]["position"], (1, 0))
        self.assertEqual(simulator.snapshot()["replay"], second["replay"])

    def test_old_snapshot_state_matches_its_moment_while_run_continues(self) -> None:
        simulator = one_robot_simulator(width=8)
        simulator.step()
        old = simulator.snapshot()
        for _ in range(4):
            simulator.step()
        new = simulator.snapshot()
        self.assertEqual(old["tick"], 1)
        self.assertEqual(len(old["replay"]), 1)
        self.assertEqual(old["robots"][0]["position"], (1, 0))
        self.assertEqual(new["tick"], 5)
        self.assertEqual(len(new["replay"]), 5)
        self.assertEqual(len(simulator.replay), 5)


class QueryEffectsTests(unittest.TestCase):
    def test_snapshot_advances_nothing(self) -> None:
        simulator = one_robot_simulator()
        simulator.step()
        clock = simulator.tick
        replay_size = len(simulator.replay)
        position = simulator.robots["A"].position
        for _ in range(3):
            simulator.snapshot()
        self.assertEqual(simulator.tick, clock)
        self.assertEqual(len(simulator.replay), replay_size)
        self.assertEqual(simulator.robots["A"].position, position)

    def test_step_still_advances_exactly_one_tick(self) -> None:
        simulator = one_robot_simulator()
        event = simulator.step()
        self.assertEqual(simulator.tick, 1)
        self.assertEqual(len(simulator.replay), 1)
        self.assertEqual(event["tick"], 1)
        self.assertEqual(event["robots"], {"A": [1, 0]})
        self.assertEqual(event["moved"], ["A"])


class TamperingDoesNotBreakCheckpointsTests(unittest.TestCase):
    def test_checkpoint_after_tampering_records_real_history(self) -> None:
        simulator = one_robot_simulator()
        frame = simulator.step()
        frame["robots"]["A"][0] = 99
        frame["moved"].append("GHOST")
        snapshot = simulator.snapshot()
        snapshot["replay"][0]["robots"]["A"] = [9, 9]
        simulator.modify_obstacles(added=[(5, 0)])
        snapshot["replay"].clear()
        simulator.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            simulator.save_checkpoint(path)
            loaded = FleetSimulator.load_checkpoint(path)
        self.assertEqual(loaded.tick, simulator.tick)
        self.assertEqual(loaded.snapshot()["replay"], simulator.snapshot()["replay"])
        real_ticks = [f for f in loaded.replay if f["type"] == "tick"]
        self.assertEqual(real_ticks[0]["robots"]["A"], [1, 0])
        self.assertEqual(real_ticks[0]["moved"], ["A"])

    def test_tampered_snapshot_cannot_be_saved_as_state(self) -> None:
        # Saving always reads the simulator's own state, never the caller's
        # edited snapshot: run the real simulation and save a clean checkpoint
        # after handing out (and mutating) results.
        simulator = one_robot_simulator()
        for _ in range(2):
            result = simulator.step()
            result["completed"].append("T-1")
        simulator.snapshot()["replay"].pop()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            simulator.save_checkpoint(path)
            loaded = FleetSimulator.load_checkpoint(path)
            self.assertFalse(loaded.tasks["T-1"].completed)
            self.assertEqual(len(loaded.replay), 2)


if __name__ == "__main__":
    unittest.main()
