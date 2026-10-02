import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


def load_document(simulator: FleetSimulator) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        simulator.save_checkpoint(path)
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)


def demo_document(ticks: int = 4) -> dict:
    grid = GridMap(8, 6, frozenset({(3, 1), (3, 2), (3, 3), (5, 4)}))
    sim = FleetSimulator(
        grid,
        [Robot("R-01", (0, 0)), Robot("R-02", (7, 5))],
        [Task("T-100", (1, 4), (6, 0)), Task("T-200", (6, 5), (0, 2))],
    )
    for _ in range(ticks):
        sim.step()
    return load_document(sim)


def pair_document(ticks: int = 3) -> dict:
    sim = FleetSimulator(
        GridMap(5, 2),
        [Robot("R-01", (0, 0)), Robot("R-02", (0, 1))],
        [Task("T-1", (1, 0), (4, 0)), Task("T-2", (1, 1), (4, 1))],
    )
    for _ in range(ticks):
        sim.step()
    return load_document(sim)


def corridor_change_document() -> dict:
    # 6x1 corridor, one robot; a tick-1 edit seals (4,0) and pauses the task.
    sim = FleetSimulator(
        GridMap(6, 1),
        [Robot("R", (0, 0))],
        [Task("T", (1, 0), (5, 0))],
    )
    sim.step()  # tick 1: R at (1,0), picks up
    sim.modify_obstacles(added=[(4, 0)])  # change stamped tick 1
    sim.step()  # tick 2: R waits at (1,0)
    sim.step()  # tick 3: R waits at (1,0)
    return load_document(sim)


def tick_frames(document: dict) -> list[dict]:
    return [frame for frame in document["replay"] if frame.get("type", "tick")]


def as_v1(document: dict) -> dict:
    doc = copy.deepcopy(document)
    doc["version"] = 1
    for key in ("base_grid", "map_changes", "paused_tasks", "traffic_waits"):
        doc.pop(key, None)
    for frame in doc["replay"]:
        frame.pop("type", None)
    return doc


def set_map_changes(document: dict, changes: list[tuple[int, int, list, list]]) -> dict:
    """Replace map_changes and the replay map-change frames, keeping tick frames."""
    doc = copy.deepcopy(document)
    doc["map_changes"] = [
        {"type": "map_change", "tick": tick, "sequence": seq, "added": added, "removed": removed}
        for tick, seq, added, removed in changes
    ]
    replay: list[dict] = []
    for frame in doc["replay"]:
        if frame.get("type", "tick") == "tick":
            replay.append(frame)
            for tick, seq, added, removed in changes:
                if tick == frame["tick"]:
                    replay.append(
                        {
                            "type": "map_change",
                            "tick": tick,
                            "sequence": seq,
                            "added": added,
                            "removed": removed,
                        }
                    )
        # drop existing map_change frames; they are rebuilt above
    doc["replay"] = replay
    return doc


class ReplayRejectionTests(unittest.TestCase):
    def reload_expect_error(self, document: object, fragment: str) -> None:
        with self.assertRaises(ValueError) as context:
            FleetSimulator._from_checkpoint(document)
        self.assertIn(fragment, str(context.exception))

    def test_historical_position_on_obstacle_rejected(self) -> None:
        doc = demo_document()
        frames = tick_frames(doc)
        frames[1]["robots"]["R-01"] = [3, 1]  # a wall cell
        self.reload_expect_error(doc, "not traversable")

    def test_historical_position_out_of_bounds_rejected(self) -> None:
        doc = demo_document()
        frames = tick_frames(doc)
        frames[1]["robots"]["R-02"] = [99, 99]
        self.reload_expect_error(doc, "not traversable")

    def test_frame_end_overlap_rejected(self) -> None:
        doc = pair_document()
        frames = tick_frames(doc)
        frames[1]["robots"]["R-02"] = copy.deepcopy(frames[1]["robots"]["R-01"])
        self.reload_expect_error(doc, "share position")

    def test_multi_cell_jump_rejected(self) -> None:
        doc = pair_document()
        frames = tick_frames(doc)
        # tick 1 R-01 at (1,0); move it two cells to (3,0) in tick 2.
        frames[1]["robots"]["R-01"] = [3, 0]
        self.reload_expect_error(doc, "jumps")

    def test_position_skip_with_map_between_frames_rejected(self) -> None:
        # A map change between ticks must not hide a two-cell skip.
        doc = corridor_change_document()
        frames = tick_frames(doc)
        # tick 1 R at (1,0); tick 2 R at (1,0) (waiting). Make it jump to (3,0).
        frames[1]["robots"]["R"] = [3, 0]
        self.reload_expect_error(doc, "jumps")

    def test_swap_within_a_tick_rejected(self) -> None:
        doc = pair_document()
        frames = tick_frames(doc)
        # tick 1: R-01 (1,0), R-02 (1,1). Swap them in tick 2.
        frames[1]["robots"]["R-01"] = [1, 1]
        frames[1]["robots"]["R-02"] = [1, 0]
        self.reload_expect_error(doc, "swap")

    def test_moved_unknown_robot_rejected(self) -> None:
        doc = pair_document()
        frames = tick_frames(doc)
        frames[1]["moved"] = ["R-01", "NOPE"]
        self.reload_expect_error(doc, "unknown robot")

    def test_moved_duplicate_robot_rejected(self) -> None:
        doc = pair_document()
        frames = tick_frames(doc)
        frames[1]["moved"] = ["R-01", "R-01"]
        self.reload_expect_error(doc, "more than once")

    def test_moved_missing_a_mover_rejected(self) -> None:
        doc = pair_document()
        frames = tick_frames(doc)
        frames[1]["moved"] = ["R-01"]  # R-02 also moved in tick 2
        self.reload_expect_error(doc, "does not match")

    def test_moved_listing_a_robot_that_stayed_put_rejected(self) -> None:
        doc = pair_document()
        frames = tick_frames(doc)
        # Make R-02 wait in tick 2; the real moved list still lists it.
        frames[1]["robots"]["R-02"] = copy.deepcopy(frames[0]["robots"]["R-02"])
        self.reload_expect_error(doc, "does not match")

    def test_moved_accuracy_not_checked_for_first_frame(self) -> None:
        # The first frame has no predecessor: moved ids are still validated, but
        # the list is not compared against a change set.
        doc = pair_document(ticks=1)
        frames = tick_frames(doc)
        frames[0]["moved"] = []  # both robots really moved; no frame to compare
        loaded = FleetSimulator._from_checkpoint(doc)
        self.assertEqual(loaded.tick, 1)

    def test_map_change_closes_occupied_cell_rejected(self) -> None:
        doc = corridor_change_document()
        # The tick-1 change adds (4,0) while R is at (1,0). Change it to close
        # (1,0), which R occupies in tick frame 1.
        doc["map_changes"][0]["added"] = [[1, 0]]
        doc["map_changes"][0]["removed"] = []
        doc["replay"][1]["added"] = [[1, 0]]
        doc["replay"][1]["removed"] = []
        doc["grid"]["obstacles"] = [[1, 0]]
        self.reload_expect_error(doc, "closes cell")

    def test_map_change_add_then_remove_same_tick_still_rejected(self) -> None:
        doc = corridor_change_document()
        # Two changes at tick 1: first closes (1,0) under R, second reopens it.
        doc["grid"]["obstacles"] = []
        doc = set_map_changes(
            doc,
            [
                (1, 1, [[1, 0]], []),
                (1, 2, [], [[1, 0]]),
            ],
        )
        self.reload_expect_error(doc, "closes cell")

    def test_cell_not_yet_open_rejected(self) -> None:
        # R stands on (1,0) in tick frame 1, but a tick-1 map change closes
        # (1,0): at frame time the cell is already an obstacle.
        doc = corridor_change_document()
        doc["map_changes"][0]["added"] = [[1, 0]]
        doc["map_changes"][0]["removed"] = []
        doc["replay"][1]["added"] = [[1, 0]]
        doc["replay"][1]["removed"] = []
        doc["grid"]["obstacles"] = [[1, 0]]
        self.reload_expect_error(doc, "not traversable")

    def test_v1_corrupted_jump_rejected(self) -> None:
        doc = as_v1(corridor_change_document())
        frames = tick_frames(doc)
        frames[1]["robots"]["R"] = [3, 0]
        self.reload_expect_error(doc, "jumps")


class ReplayAcceptanceTests(unittest.TestCase):
    def test_zero_tick_empty_fleet_reads(self) -> None:
        sim = FleetSimulator(GridMap(3, 3), [], [])
        loaded = FleetSimulator._from_checkpoint(load_document(sim))
        self.assertEqual(loaded.tick, 0)
        self.assertEqual(loaded.robots, {})

    def test_zero_tick_with_robots_reads(self) -> None:
        sim = FleetSimulator(GridMap(4, 4), [Robot("A", (0, 0))], [])
        loaded = FleetSimulator._from_checkpoint(load_document(sim))
        self.assertEqual(loaded.tick, 0)
        self.assertEqual(loaded.robots["A"].position, (0, 0))

    def test_wait_frames_read(self) -> None:
        # corridor_change_document has R waiting in ticks 2 and 3.
        doc = corridor_change_document()
        loaded = FleetSimulator._from_checkpoint(doc)
        self.assertEqual(loaded.tick, 3)
        self.assertEqual(loaded.robots["R"].position, (1, 0))

    def test_vacated_cell_entry_is_legal(self) -> None:
        # A robot may drive into a cell another robot just vacated; this is the
        # yielding case and must not be rejected.
        sim = FleetSimulator(
            GridMap(5, 1),
            [Robot("R-1", (0, 0)), Robot("R-2", (4, 0))],
            [Task("T-1", (1, 0), (4, 0)), Task("T-2", (3, 0), (0, 0))],
        )
        for _ in range(4):
            sim.step()
        loaded = FleetSimulator._from_checkpoint(load_document(sim))
        self.assertEqual(loaded.tick, 4)

    def test_unsorted_moved_list_is_legal(self) -> None:
        # moved is a set of ids, not an ordered list; reordering must not matter.
        doc = pair_document()
        frames = tick_frames(doc)
        for frame in frames:
            frame["moved"] = list(reversed(frame["moved"]))
        loaded = FleetSimulator._from_checkpoint(doc)
        self.assertEqual(loaded.tick, 3)

    def test_v1_valid_file_reads(self) -> None:
        doc = as_v1(corridor_change_document())
        loaded = FleetSimulator._from_checkpoint(doc)
        self.assertEqual(loaded.tick, 3)
        self.assertEqual(loaded.map_change_history(), [])


class ResumeCliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        import sys

        repo_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(repo_root))
        return subprocess.run(
            [sys.executable, "-m", "warehouse_fleet", *arguments],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

    def test_resume_replay_rejection_is_nonzero_without_snapshot_or_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "good.json")
            bad = os.path.join(tmp, "bad.json")
            sim = FleetSimulator(
                GridMap(5, 1),
                [Robot("R", (0, 0))],
                [Task("T", (1, 0), (4, 0))],
            )
            for _ in range(3):
                sim.step()
            sim.save_checkpoint(good)
            doc = load_document(sim)
            frames = tick_frames(doc)
            frames[1]["robots"]["R"] = [3, 0]  # two-cell jump
            with open(bad, "w", encoding="utf-8") as handle:
                json.dump(doc, handle)
            before = Path(bad).read_bytes()
            result = self.run_cli("resume", bad, "--steps", "3")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("resume:", result.stderr)
            self.assertIn("jumps", result.stderr)
            # The input file is left untouched.
            self.assertEqual(Path(bad).read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
