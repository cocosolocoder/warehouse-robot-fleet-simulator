"""History-plausibility checks for restored checkpoints.

A checkpoint whose last frame matches the current state can still carry an
impossible past: robots out of bounds or sharing cells, teleports and swaps
between tick frames, ``moved`` lists that do not describe the frame, or robots
standing in cells that were obstacles at the time. ``load_checkpoint`` must
reject every such document while accepting everything the simulator itself
could produce, including traffic follow-moves and cells that close after a
robot has left them.
"""

import copy
import json
import os
import tempfile
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


def tick_frame(number: int, moved, positions, completed=()):
    return {
        "type": "tick",
        "tick": number,
        "moved": list(moved),
        "robots": {robot_id: list(cell) for robot_id, cell in positions.items()},
        "completed": list(completed),
    }


def map_change(tick, sequence, added=(), removed=()):
    return {
        "type": "map_change",
        "tick": tick,
        "sequence": sequence,
        "added": [list(cell) for cell in added],
        "removed": [list(cell) for cell in removed],
    }


class HistoryDocumentBuilder:
    """Builds v1/v2 checkpoint docs around arbitrary replay histories.

    The starting document is a real zero-tick checkpoint, so every field not
    under test keeps a valid shape; routes and tasks stay empty.
    """

    def __init__(self, width=5, height=2, robots=(("A", (0, 0)), ("B", (4, 1)))):
        simulator = FleetSimulator(
            GridMap(width, height),
            [Robot(robot_id, position) for robot_id, position in robots],
            [],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            simulator.save_checkpoint(path)
            with open(path, encoding="utf-8") as handle:
                self.document = json.load(handle)
            self._tmp = tmpdir

    def frames(self, frames, *, base_obstacles=(), grid_obstacles=None):
        document = self.document
        document["tick"] = len(frames)
        document["replay"] = frames
        document["base_grid"]["obstacles"] = [list(cell) for cell in base_obstacles]
        document["grid"]["obstacles"] = (
            [list(cell) for cell in grid_obstacles]
            if grid_obstacles is not None
            else [list(cell) for cell in base_obstacles]
        )
        return self

    def dynamic(self, ordered_events, *, final_positions, base_obstacles=()):
        """ordered_events: replay frames/change records in storage order."""
        document = self.document
        tick_frames = [e for e in ordered_events if e["type"] == "tick"]
        changes = [e for e in ordered_events if e["type"] == "map_change"]
        document["tick"] = len(tick_frames)
        document["replay"] = copy.deepcopy(ordered_events)
        document["map_changes"] = [
            {key: copy.deepcopy(value) for key, value in event.items()}
            for event in changes
        ]
        document["base_grid"]["obstacles"] = [list(cell) for cell in base_obstacles]
        obstacles = set(base_obstacles)
        for event in changes:
            obstacles.update(tuple(cell) for cell in event["added"])
            obstacles.difference_update(tuple(cell) for cell in event["removed"])
        document["grid"]["obstacles"] = [list(cell) for cell in sorted(obstacles)]
        records = {record["robot_id"]: record for record in document["robots"]}
        for robot_id, cell in final_positions.items():
            records[robot_id]["position"] = list(cell)
            records[robot_id]["route"] = []
        return self

    def final(self, positions):
        records = {record["robot_id"]: record for record in self.document["robots"]}
        for robot_id, cell in positions.items():
            records[robot_id]["position"] = list(cell)
            records[robot_id]["route"] = []
        return self

    def version_one(self):
        document = self.document
        document["version"] = 1
        for key in ("base_grid", "map_changes", "paused_tasks", "traffic_waits"):
            document.pop(key, None)
        document["replay"] = [
            {key: value for key, value in frame.items() if key != "type"}
            for frame in document["replay"]
        ]
        return self

    def load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.document, handle)
            return FleetSimulator.load_checkpoint(path)


def expect_rejected(testcase, document, fragment=None):
    with testcase.assertRaises(ValueError) as context:
        _load(document)
    if fragment is not None:
        testcase.assertIn(fragment, str(context.exception))
    return str(context.exception)


def _load(document):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return FleetSimulator.load_checkpoint(path)


class FramePlausibilityTests(unittest.TestCase):
    def setUp(self):
        self.builder = HistoryDocumentBuilder()

    def reject(self, document, fragment=None):
        return expect_rejected(self, document, fragment)

    def test_legit_history_loads(self):
        loaded = (
            self.builder.frames(
                [
                    tick_frame(1, ["A", "B"], {"A": (1, 0), "B": (3, 1)}),
                    tick_frame(2, ["A"], {"A": (2, 0), "B": (3, 1)}),
                    tick_frame(3, ["B"], {"A": (2, 0), "B": (3, 0)}),
                ]
            )
            .final({"A": (2, 0), "B": (3, 0)})
            .load()
        )
        self.assertEqual(loaded.tick, 3)

    def test_multi_cell_jump_rejected(self):
        document = (
            self.builder.frames(
                [
                    tick_frame(1, ["A"], {"A": (1, 0), "B": (4, 1)}),
                    tick_frame(2, ["A"], {"A": (3, 0), "B": (4, 1)}),
                    tick_frame(3, [], {"A": (3, 0), "B": (4, 1)}),
                ]
            )
            .final({"A": (3, 0), "B": (4, 1)})
            .document
        )
        self.reject(document, "moves more than one cell")

    def test_diagonal_step_rejected(self):
        document = (
            self.builder.frames(
                [
                    tick_frame(1, ["A"], {"A": (1, 0), "B": (3, 1)}),
                    tick_frame(2, ["A"], {"A": (2, 1), "B": (3, 1)}),
                    tick_frame(3, [], {"A": (2, 1), "B": (3, 1)}),
                ]
            )
            .final({"A": (2, 1), "B": (3, 1)})
            .document
        )
        self.reject(document, "moves more than one cell")

    def test_historical_out_of_bounds_rejected(self):
        document = (
            self.builder.frames(
                [
                    tick_frame(1, ["A"], {"A": (1, 0), "B": (4, 1)}),
                    tick_frame(2, ["A"], {"A": (9, 0), "B": (4, 1)}),
                    tick_frame(3, ["A"], {"A": (3, 0), "B": (4, 1)}),
                ]
            )
            .final({"A": (3, 0), "B": (4, 1)})
            .document
        )
        self.reject(document, "outside the map")

    def test_historical_obstacle_cell_rejected(self):
        document = (
            self.builder.frames(
                [
                    tick_frame(1, ["A"], {"A": (1, 0), "B": (4, 1)}),
                    tick_frame(2, ["A"], {"A": (2, 0), "B": (4, 1)}),
                    tick_frame(3, ["A"], {"A": (3, 0), "B": (4, 1)}),
                ],
                base_obstacles=((2, 0),),
            )
            .final({"A": (3, 0), "B": (4, 1)})
            .document
        )
        self.reject(document, "an obstacle")

    def test_middle_frame_overlap_rejected_even_if_later_separated(self):
        document = (
            self.builder.frames(
                [
                    tick_frame(1, ["A", "B"], {"A": (1, 0), "B": (3, 1)}),
                    tick_frame(2, ["A", "B"], {"A": (2, 0), "B": (2, 0)}),
                    tick_frame(3, ["B"], {"A": (2, 0), "B": (3, 0)}),
                ]
            )
            .final({"A": (2, 0), "B": (3, 0)})
            .document
        )
        self.reject(document, "both occupy")

    def test_swap_within_one_tick_rejected(self):
        document = (
            self.builder.frames(
                [
                    tick_frame(1, ["A", "B"], {"A": (1, 0), "B": (2, 0)}),
                    tick_frame(2, ["A", "B"], {"A": (2, 0), "B": (1, 0)}),
                ]
            )
            .final({"A": (2, 0), "B": (1, 0)})
            .document
        )
        self.reject(document, "swap")

    def test_following_into_vacated_cell_is_legal(self):
        # Both robots advance one cell; A takes the cell B just left, and the
        # recorded order is deliberately not sorted by robot id.
        loaded = (
            self.builder.frames(
                [
                    tick_frame(1, ["A", "B"], {"A": (1, 0), "B": (2, 0)}),
                    tick_frame(2, ["B", "A"], {"A": (2, 0), "B": (3, 0)}),
                ]
            )
            .final({"A": (2, 0), "B": (3, 0)})
            .load()
        )
        self.assertEqual(loaded.replay[1]["moved"], ["B", "A"])

    def test_legit_wait_frame_loads(self):
        loaded = (
            self.builder.frames(
                [
                    tick_frame(1, ["A"], {"A": (1, 0), "B": (4, 1)}),
                    tick_frame(2, [], {"A": (1, 0), "B": (4, 1)}),
                    tick_frame(3, ["A"], {"A": (2, 0), "B": (4, 1)}),
                ]
            )
            .final({"A": (2, 0), "B": (4, 1)})
            .load()
        )
        self.assertEqual(loaded.tick, 3)

    def test_moved_unknown_robot_rejected(self):
        document = (
            self.builder.frames(
                [tick_frame(1, ["NOPE"], {"A": (1, 0), "B": (4, 1)})]
            )
            .final({"A": (1, 0), "B": (4, 1)})
            .document
        )
        self.reject(document, "unknown robot")

    def test_moved_duplicate_robot_rejected(self):
        document = (
            self.builder.frames(
                [tick_frame(1, ["A", "A"], {"A": (1, 0), "B": (4, 1)})]
            )
            .final({"A": (1, 0), "B": (4, 1)})
            .document
        )
        self.reject(document, "more than once")

    def test_moved_missing_a_mover_rejected(self):
        document = (
            self.builder.frames(
                [
                    tick_frame(1, ["A", "B"], {"A": (1, 0), "B": (3, 1)}),
                    tick_frame(2, ["A"], {"A": (2, 0), "B": (3, 0)}),
                ]
            )
            .final({"A": (2, 0), "B": (3, 0)})
            .document
        )
        self.reject(document, "movers missing")

    def test_moved_lists_a_waiting_robot_rejected(self):
        document = (
            self.builder.frames(
                [
                    tick_frame(1, ["A"], {"A": (1, 0), "B": (4, 1)}),
                    tick_frame(2, ["A", "B"], {"A": (2, 0), "B": (4, 1)}),
                ]
            )
            .final({"A": (2, 0), "B": (4, 1)})
            .document
        )
        self.reject(document, "listed without moving")

    def test_first_frame_moved_content_is_not_guessed(self):
        # No predecessor frame exists: the first moved list cannot be checked
        # against position changes (only its ids), regardless of its contents.
        for moved in ([], ["A"], ["B", "A"]):
            with self.subTest(moved=moved):
                loaded = (
                    HistoryDocumentBuilder()
                    .frames(
                        [
                            tick_frame(1, moved, {"A": (1, 0), "B": (3, 1)}),
                            tick_frame(2, [], {"A": (1, 0), "B": (3, 1)}),
                        ]
                    )
                    .final({"A": (1, 0), "B": (3, 1)})
                    .load()
                )
                self.assertEqual(loaded.tick, 2)


class VersionOneHistoryTests(unittest.TestCase):
    def test_version_one_teleport_rejected(self):
        document = (
            HistoryDocumentBuilder()
            .frames(
                [
                    tick_frame(1, ["A"], {"A": (1, 0), "B": (4, 1)}),
                    tick_frame(2, ["A"], {"A": (3, 0), "B": (4, 1)}),
                ]
            )
            .final({"A": (3, 0), "B": (4, 1)})
            .version_one()
            .document
        )
        expect_rejected(self, document, "moves more than one cell")

    def test_version_one_wait_frames_load(self):
        loaded = (
            HistoryDocumentBuilder()
            .frames(
                [
                    tick_frame(1, [], {"A": (1, 0), "B": (3, 1)}),
                    tick_frame(2, [], {"A": (1, 0), "B": (3, 1)}),
                ]
            )
            .final({"A": (1, 0), "B": (3, 1)})
            .version_one()
            .load()
        )
        self.assertEqual(loaded.map_change_history(), [])


class DynamicMapHistoryTests(unittest.TestCase):
    def test_closing_a_cell_after_the_robot_moved_on_is_legal(self):
        builder = HistoryDocumentBuilder()
        loaded = (
            builder.dynamic(
                [
                    tick_frame(1, ["A"], {"A": (3, 0), "B": (4, 1)}),
                    tick_frame(2, ["A"], {"A": (4, 0), "B": (4, 1)}),
                    map_change(2, 1, added=((3, 0),)),
                ],
                final_positions={"A": (4, 0), "B": (4, 1)},
            )
            .load()
        )
        self.assertEqual(loaded.grid.obstacles, frozenset({(3, 0)}))

    def test_add_obstacle_on_cell_occupied_at_tick_end_rejected(self):
        builder = HistoryDocumentBuilder()
        document = (
            builder.dynamic(
                [
                    tick_frame(1, ["A"], {"A": (4, 0), "B": (3, 1)}),
                    map_change(1, 1, added=((4, 0),)),
                    tick_frame(2, ["A"], {"A": (3, 0), "B": (3, 1)}),
                ],
                final_positions={"A": (3, 0), "B": (3, 1)},
            )
            .document
        )
        expect_rejected(self, document, "occupied by robot")

    def test_add_then_remove_on_occupied_cell_same_tick_rejected(self):
        builder = HistoryDocumentBuilder()
        document = (
            builder.dynamic(
                [
                    tick_frame(1, ["A"], {"A": (4, 0), "B": (3, 1)}),
                    map_change(1, 1, added=((4, 0),)),
                    map_change(1, 2, removed=((4, 0),)),
                ],
                final_positions={"A": (4, 0), "B": (3, 1)},
            )
            .document
        )
        expect_rejected(self, document, "occupied by robot")

    def test_appearing_in_a_cell_before_it_opens_rejected(self):
        builder = HistoryDocumentBuilder()
        document = (
            builder.dynamic(
                [
                    tick_frame(1, ["A"], {"A": (2, 0), "B": (4, 1)}),
                    map_change(1, 1, removed=((2, 0),)),
                ],
                final_positions={"A": (2, 0), "B": (4, 1)},
                base_obstacles=((2, 0),),
            )
            .document
        )
        expect_rejected(self, document, "an obstacle")

    def test_entering_a_cell_after_it_opens_is_legal(self):
        builder = HistoryDocumentBuilder()
        loaded = (
            builder.dynamic(
                [
                    tick_frame(1, ["A"], {"A": (1, 0), "B": (4, 1)}),
                    map_change(1, 1, removed=((2, 0),)),
                    tick_frame(2, ["A"], {"A": (2, 0), "B": (4, 1)}),
                ],
                final_positions={"A": (2, 0), "B": (4, 1)},
                base_obstacles=((2, 0),),
            )
            .load()
        )
        self.assertEqual(loaded.grid.obstacles, frozenset())

    def test_adjacency_between_frames_not_broken_by_sandwiched_changes(self):
        # Two change records sit between the frames; the robot still only moves
        # one cell and must not be penalised for the intervening edit records.
        builder = HistoryDocumentBuilder()
        loaded = (
            builder.dynamic(
                [
                    tick_frame(1, ["A"], {"A": (3, 0), "B": (4, 0)}),
                    map_change(1, 1, added=((2, 1),)),
                    map_change(1, 2, removed=((2, 1),)),
                    tick_frame(2, ["A"], {"A": (3, 1), "B": (4, 0)}),
                ],
                final_positions={"A": (3, 1), "B": (4, 0)},
            )
            .load()
        )
        self.assertEqual(loaded.tick, 2)

    def test_real_close_behind_run_round_trips(self):
        simulator = FleetSimulator(
            GridMap(6, 1), [Robot("R", (0, 0))], [Task("T", (1, 0), (5, 0))]
        )
        simulator.step()  # R at (1,0)
        simulator.modify_obstacles(added=[(0, 0)])  # closes a cell already left
        simulator.step()
        simulator.step()  # R at (3,0)
        simulator.modify_obstacles(added=[(2, 0)])  # closes frame-2 cell after
        simulator.step()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            simulator.save_checkpoint(path)
            loaded = FleetSimulator.load_checkpoint(path)
        self.assertEqual(loaded.tick, 4)
        self.assertEqual(
            loaded.grid.obstacles, frozenset({(0, 0), (2, 0)})
        )


if __name__ == "__main__":
    unittest.main()
