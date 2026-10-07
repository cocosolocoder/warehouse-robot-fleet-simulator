"""Replay completion history must agree with the saved task states.

A tick frame's ``completed`` list is the cumulative set of tasks finished by
the end of that tick. Loading must reject frames that name unknown tasks,
repeat a task within one frame, drop a previously recorded completion, or end
on a set that differs from the tasks saved as completed -- for both checkpoint
versions -- while accepting first frames that already list completions,
zero-tick checkpoints with initially completed tasks and trailing map edits.
"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task
from warehouse_fleet.__main__ import main as cli_main


def tick_frame(number, moved, positions, completed=()):
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


POSITIONS = {"A": (0, 0), "B": (4, 1)}


def build_document():
    """A real zero-tick checkpoint with two robots and two pending tasks."""
    simulator = FleetSimulator(
        GridMap(5, 2),
        [Robot("A", (0, 0)), Robot("B", (4, 1))],
        [Task("T1", (1, 0), (2, 0)), Task("T2", (3, 1), (4, 0))],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        simulator.save_checkpoint(path)
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)


def with_history(document, frames, *, completed=(), owners=None):
    """Attach a replay history and mark *completed* tasks as done.

    *owners* optionally records the historical delivery robot per task; every
    completed task without an explicit owner keeps ``"A"`` (the robot is idle
    again by the time the checkpoint is saved).
    """
    owners = owners or {}
    tick_frames = [frame for frame in frames if frame["type"] == "tick"]
    document["tick"] = len(tick_frames)
    document["replay"] = frames
    if tick_frames:
        for robot_id, cell in tick_frames[-1]["robots"].items():
            record = next(
                record
                for record in document["robots"]
                if record["robot_id"] == robot_id
            )
            record["position"] = list(cell)
            record["route"] = []
    for task_id in completed:
        record = next(
            record for record in document["tasks"] if record["task_id"] == task_id
        )
        record["picked_up"] = True
        record["completed"] = True
        # Completed tasks keep their historical owner; the robot is idle now.
        record["assigned_robot"] = owners.get(task_id, "A")
    return document


def as_version_one(document):
    document = json.loads(json.dumps(document))
    document["version"] = 1
    for key in ("base_grid", "map_changes", "paused_tasks", "traffic_waits"):
        document.pop(key, None)
    document["replay"] = [
        {key: value for key, value in frame.items() if key != "type"}
        for frame in document["replay"]
    ]
    return document


def load(document):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return FleetSimulator.load_checkpoint(path)


def expect_rejected(testcase, document, *fragments):
    with testcase.assertRaises(ValueError) as context:
        load(document)
    message = str(context.exception)
    for fragment in fragments:
        testcase.assertIn(fragment, message)
    return message


class CompletionHistoryAcceptanceTests(unittest.TestCase):
    def test_growing_completion_history_loads(self):
        document = with_history(
            build_document(),
            [
                tick_frame(1, ["A"], {"A": (1, 0), "B": (4, 1)}),
                tick_frame(2, ["A"], {"A": (2, 0), "B": (4, 1)}, ["T1"]),
                tick_frame(3, ["B"], {"A": (2, 0), "B": (4, 0)}, ["T1", "T2"]),
            ],
            completed=("T1", "T2"),
            owners={"T1": "A", "T2": "B"},
        )
        loaded = load(document)
        self.assertEqual(loaded.tick, 3)
        self.assertTrue(loaded.tasks["T1"].completed)
        self.assertTrue(loaded.tasks["T2"].completed)
        self.assertEqual(loaded.metrics()["tasks_completed"], 2)

    def test_first_frame_may_already_list_completions(self):
        # No earlier frame exists, so completions may appear without a
        # preceding empty list.
        document = with_history(
            build_document(),
            [tick_frame(1, [], POSITIONS, ["T1"])],
            completed=("T1",),
        )
        loaded = load(document)
        self.assertTrue(loaded.tasks["T1"].completed)

    def test_historical_owner_may_have_left_the_fleet_in_first_frame(self):
        # A completion already present in the first frame may predate the
        # recording, delivered by a robot no longer in the fleet.
        document = with_history(
            build_document(),
            [tick_frame(1, [], POSITIONS, ["T1"])],
            completed=("T1",),
            owners={"T1": "GONE"},
        )
        loaded = load(document)
        self.assertEqual(loaded.tasks["T1"].assigned_robot, "GONE")

    def test_order_within_frame_and_repeats_across_frames_are_legal(self):
        document = with_history(
            build_document(),
            [
                tick_frame(1, ["A"], {"A": (2, 0), "B": (4, 1)}, ["T1"]),
                tick_frame(2, [], {"A": (2, 0), "B": (4, 1)}, ["T1"]),
                tick_frame(
                    3,
                    ["B"],
                    {"A": (2, 0), "B": (4, 0)},
                    ["T2", "T1"],
                ),
            ],
            completed=("T1", "T2"),
            owners={"T1": "A", "T2": "B"},
        )
        loaded = load(document)
        self.assertEqual(loaded.replay[2]["completed"], ["T2", "T1"])

    def test_tick_zero_with_initially_completed_tasks_loads(self):
        document = with_history(build_document(), [], completed=("T1", "T2"))
        loaded = load(document)
        self.assertEqual(loaded.tick, 0)
        self.assertEqual(loaded.metrics()["tasks_completed"], 2)

    def test_trailing_map_change_does_not_count_as_a_time_step(self):
        # The edit is stamped after the final tick; the completion judgement
        # still comes from the last tick frame alone.
        document = with_history(
            build_document(),
            [
                tick_frame(1, [], POSITIONS, ["T1"]),
                map_change(1, 1, added=((2, 1),)),
            ],
            completed=("T1",),
        )
        document["map_changes"] = [map_change(1, 1, added=((2, 1),))]
        document["grid"]["obstacles"] = [[2, 1]]
        loaded = load(document)
        self.assertEqual(loaded.tick, 1)
        self.assertEqual(loaded.grid.obstacles, frozenset({(2, 1)}))
        self.assertTrue(loaded.tasks["T1"].completed)

    def test_legit_history_preserves_tasks_replay_and_metrics(self):
        frames = [
            tick_frame(1, ["A"], {"A": (1, 0), "B": (4, 1)}),
            tick_frame(2, ["A"], {"A": (2, 0), "B": (4, 1)}, ["T1"]),
        ]
        document = with_history(build_document(), frames, completed=("T1",))
        loaded = load(document)
        self.assertEqual(loaded.replay, frames)
        self.assertEqual(loaded.metrics()["tasks_completed"], 1)
        self.assertFalse(loaded.tasks["T2"].completed)
        # Resuming appends new frames; the recorded history is not rewritten.
        loaded.step()
        self.assertEqual(loaded.replay[:2], frames)
        self.assertEqual(loaded.tick, 3)


class CompletionHistoryRejectionTests(unittest.TestCase):
    def test_unknown_completed_task_rejected(self):
        document = with_history(
            build_document(),
            [tick_frame(1, [], POSITIONS, ["NOPE"])],
        )
        expect_rejected(self, document, "1", "NOPE", "unknown completed task")

    def test_duplicate_completion_within_one_frame_rejected(self):
        document = with_history(
            build_document(),
            [tick_frame(1, [], POSITIONS, ["T1", "T1"])],
            completed=("T1",),
        )
        expect_rejected(self, document, "1", "T1", "more than once")

    def test_completion_dropped_by_later_frame_rejected(self):
        document = with_history(
            build_document(),
            [
                tick_frame(1, [], POSITIONS, ["T1"]),
                tick_frame(2, [], POSITIONS),
            ],
        )
        expect_rejected(self, document, "2", "T1", "drops")

    def test_completion_dropped_while_robot_moves_on_rejected(self):
        # The robot starting another task is no excuse: the historical
        # completion record must survive every later tick.
        document = with_history(
            build_document(),
            [
                tick_frame(1, [], POSITIONS, ["T1", "T2"]),
                tick_frame(2, [], POSITIONS, ["T2"]),
            ],
            completed=("T2",),
        )
        expect_rejected(self, document, "2", "T1", "drops")

    def test_last_frame_missing_a_saved_completion_rejected(self):
        document = with_history(
            build_document(),
            [tick_frame(1, [], POSITIONS)],
            completed=("T1",),
        )
        expect_rejected(self, document, "1", "T1", "do not match")

    def test_last_frame_listing_an_unfinished_task_rejected(self):
        document = with_history(
            build_document(),
            [tick_frame(1, [], POSITIONS, ["T1"])],
        )
        expect_rejected(self, document, "1", "T1", "do not match")

    def test_version_one_unknown_completed_task_rejected(self):
        document = as_version_one(
            with_history(
                build_document(),
                [tick_frame(1, [], POSITIONS, ["NOPE"])],
            )
        )
        expect_rejected(self, document, "1", "NOPE", "unknown completed task")

    def test_version_one_dropped_completion_rejected(self):
        document = as_version_one(
            with_history(
                build_document(),
                [
                    tick_frame(1, [], POSITIONS, ["T1"]),
                    tick_frame(2, [], POSITIONS),
                ],
            )
        )
        expect_rejected(self, document, "2", "T1", "drops")

    def test_version_one_mismatched_final_frame_rejected(self):
        document = as_version_one(
            with_history(
                build_document(),
                [tick_frame(1, [], POSITIONS)],
                completed=("T1",),
            )
        )
        expect_rejected(self, document, "1", "T1", "do not match")

    def test_version_one_legit_completion_history_loads(self):
        document = as_version_one(
            with_history(
                build_document(),
                [
                    tick_frame(1, ["A"], {"A": (2, 0), "B": (4, 1)}, ["T1"]),
                    tick_frame(2, ["B"], {"A": (2, 0), "B": (4, 0)}, ["T1", "T2"]),
                ],
                completed=("T1", "T2"),
                owners={"T1": "A", "T2": "B"},
            )
        )
        loaded = load(document)
        self.assertEqual(loaded.metrics()["tasks_completed"], 2)

    def test_rejection_leaves_the_document_untouched(self):
        document = with_history(
            build_document(),
            [tick_frame(1, [], POSITIONS, ["T1"])],
        )
        with self.assertRaises(ValueError):
            load(document)
        self.assertFalse(document["tasks"][0]["completed"])
        self.assertEqual(document["replay"][0]["completed"], ["T1"])


class CompletionLocationAcceptanceTests(unittest.TestCase):
    def test_robot_at_dropoff_when_task_first_completed_loads(self):
        document = with_history(
            build_document(),
            [
                tick_frame(1, ["A"], {"A": (1, 0), "B": (4, 1)}),
                tick_frame(2, ["A"], {"A": (2, 0), "B": (4, 1)}, ["T1"]),
            ],
            completed=("T1",),
        )
        loaded = load(document)
        self.assertTrue(loaded.tasks["T1"].completed)
        self.assertEqual(loaded.tasks["T1"].assigned_robot, "A")

    def test_task_may_complete_without_the_robot_moving_that_tick(self):
        # A is already standing on the dropoff when the tick begins and waits
        # there; finishing needs no movement, so 'moved' need not list it.
        document = with_history(
            build_document(),
            [
                tick_frame(1, ["B"], {"A": (2, 0), "B": (3, 1)}),
                tick_frame(2, [], {"A": (2, 0), "B": (3, 1)}, ["T1"]),
            ],
            completed=("T1",),
        )
        loaded = load(document)
        self.assertTrue(loaded.tasks["T1"].completed)
        self.assertEqual(loaded.replay[1]["moved"], [])

    def test_robot_may_leave_the_dropoff_in_a_later_frame(self):
        document = with_history(
            build_document(),
            [
                tick_frame(1, ["A"], {"A": (1, 0), "B": (4, 1)}),
                tick_frame(2, ["A"], {"A": (2, 0), "B": (4, 1)}, ["T1"]),
                tick_frame(3, ["A"], {"A": (1, 0), "B": (4, 1)}, ["T1"]),
            ],
            completed=("T1",),
        )
        loaded = load(document)
        self.assertEqual(loaded.robots["A"].position, (1, 0))
        self.assertTrue(loaded.tasks["T1"].completed)

    def test_repeating_a_completion_never_reopens_the_location_check(self):
        # Only the frame in which T1 first appears is judged; later frames may
        # keep listing it while A works far away from the dropoff.
        document = with_history(
            build_document(),
            [
                tick_frame(1, ["A"], {"A": (2, 0), "B": (4, 1)}, ["T1"]),
                tick_frame(2, ["A"], {"A": (1, 0), "B": (4, 1)}, ["T1"]),
                tick_frame(3, ["A"], {"A": (0, 0), "B": (4, 1)}, ["T1"]),
            ],
            completed=("T1",),
        )
        loaded = load(document)
        self.assertEqual(loaded.robots["A"].position, (0, 0))

    def test_map_change_between_frames_keeps_the_location_requirement(self):
        frames = [
            tick_frame(1, ["A"], {"A": (2, 0), "B": (4, 1)}),
            map_change(1, 1, added=((0, 1),)),
            tick_frame(2, [], {"A": (2, 0), "B": (4, 1)}, ["T1"]),
        ]
        document = with_history(build_document(), frames, completed=("T1",))
        document["map_changes"] = [map_change(1, 1, added=((0, 1),))]
        document["grid"]["obstacles"] = [[0, 1]]
        loaded = load(document)
        self.assertEqual(loaded.tick, 2)
        self.assertEqual(loaded.grid.obstacles, frozenset({(0, 1)}))
        self.assertTrue(loaded.tasks["T1"].completed)

    def test_real_delivery_round_trip_preserves_tasks_replay_and_metrics(self):
        simulator = FleetSimulator(
            GridMap(5, 2),
            [Robot("A", (0, 0)), Robot("B", (4, 1))],
            [Task("T1", (1, 0), (2, 0)), Task("T2", (3, 1), (4, 0))],
        )
        while simulator.metrics()["tasks_completed"] < 2:
            simulator.step()
        snapshot = simulator.snapshot()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            simulator.save_checkpoint(path)
            loaded = FleetSimulator.load_checkpoint(path)
        self.assertEqual(loaded.snapshot()["tasks"], snapshot["tasks"])
        self.assertEqual(loaded.replay, snapshot["replay"])
        self.assertEqual(loaded.metrics(), snapshot["metrics"])


class CompletionLocationRejectionTests(unittest.TestCase):
    def test_robot_away_from_dropoff_when_task_completed_rejected(self):
        document = with_history(
            build_document(),
            [
                tick_frame(1, [], POSITIONS),
                tick_frame(2, [], POSITIONS, ["T1"]),
            ],
            completed=("T1",),
        )
        expect_rejected(self, document, "2", "T1", "A", "dropoff", "[2, 0]", "[0, 0]")

    def test_delivery_robot_missing_from_frame_rejected(self):
        frame_two = {
            "type": "tick",
            "tick": 2,
            "moved": [],
            "robots": {"B": [4, 1]},
            "completed": ["T1"],
        }
        document = with_history(
            build_document(),
            [tick_frame(1, [], POSITIONS), frame_two],
            completed=("T1",),
        )
        expect_rejected(self, document, "2", "T1", "A", "absent")

    def test_retired_owner_allowed_for_history_but_not_a_new_completion(self):
        # A retired robot may own a completion already present in frame 1, but
        # a task first completed in a later frame still needs that robot
        # present and on the dropoff in that very frame.
        frame_two = {
            "type": "tick",
            "tick": 2,
            "moved": [],
            "robots": {"A": [0, 0], "B": [4, 1]},
            "completed": ["T1"],
        }
        document = with_history(
            build_document(),
            [tick_frame(1, [], POSITIONS), frame_two],
            completed=("T1",),
            owners={"T1": "GONE"},
        )
        expect_rejected(self, document, "2", "T1", "GONE", "absent")

    def test_another_robot_on_the_dropoff_does_not_count(self):
        # B happens to stand on T1's dropoff, but T1's recorded delivery robot
        # is A, and A is nowhere near it.
        positions = {"A": (0, 0), "B": (2, 0)}
        document = with_history(
            build_document(),
            [
                tick_frame(1, [], positions),
                tick_frame(2, [], positions, ["T1"]),
            ],
            completed=("T1",),
        )
        expect_rejected(self, document, "T1", "A", "dropoff")

    def test_map_change_cannot_stand_in_for_completion_frame_evidence(self):
        # The edit between the frames is not a time step: T1 still has to be
        # delivered in tick frame 2 itself.
        frames = [
            tick_frame(1, [], POSITIONS),
            map_change(1, 1, added=((1, 1),)),
            tick_frame(2, [], POSITIONS, ["T1"]),
        ]
        document = with_history(build_document(), frames, completed=("T1",))
        document["map_changes"] = [map_change(1, 1, added=((1, 1),))]
        document["grid"]["obstacles"] = [[1, 1]]
        expect_rejected(self, document, "2", "T1", "A", "dropoff")

    def test_version_one_robot_away_from_dropoff_rejected(self):
        document = as_version_one(
            with_history(
                build_document(),
                [
                    tick_frame(1, [], POSITIONS),
                    tick_frame(2, [], POSITIONS, ["T1"]),
                ],
                completed=("T1",),
            )
        )
        expect_rejected(self, document, "2", "T1", "A", "dropoff")

    def test_version_one_delivery_robot_missing_from_frame_rejected(self):
        frame_two = {
            "type": "tick",
            "tick": 2,
            "moved": [],
            "robots": {"B": [4, 1]},
            "completed": ["T1"],
        }
        document = as_version_one(
            with_history(
                build_document(),
                [tick_frame(1, [], POSITIONS), frame_two],
                completed=("T1",),
            )
        )
        expect_rejected(self, document, "2", "T1", "A", "absent")


class ResumeCliCompletionTests(unittest.TestCase):
    def resume(self, steps, document=None, fragment="T1"):
        if document is None:
            document = with_history(
                build_document(),
                [tick_frame(1, [], POSITIONS, ["T1"])],
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            output = os.path.join(tmpdir, "next.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(document, handle)
            with open(path, "rb") as handle:
                original = handle.read()
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as context:
                    cli_main(
                        ["resume", path, "--steps", str(steps), "--checkpoint", output]
                    )
            self.assertEqual(context.exception.code, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("resume:", stderr.getvalue())
            self.assertIn(fragment, stderr.getvalue())
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), original)
            self.assertFalse(os.path.exists(output))

    def test_resume_rejects_contradictory_completion_history(self):
        self.resume(5)

    def test_resume_with_zero_steps_still_validates(self):
        self.resume(0)

    def forged_delivery_document(self):
        # T1 is first recorded as completed in frame 2 while its delivery
        # robot A still stands at (0, 0), nowhere near the (2, 0) dropoff.
        return with_history(
            build_document(),
            [
                tick_frame(1, [], POSITIONS),
                tick_frame(2, [], POSITIONS, ["T1"]),
            ],
            completed=("T1",),
        )

    def test_resume_rejects_forged_delivery_location(self):
        self.resume(5, self.forged_delivery_document())

    def test_resume_rejects_forged_delivery_location_with_zero_steps(self):
        # Even with no ticks to run, loading must fail: no simulation output on
        # stdout and no output checkpoint written.
        self.resume(0, self.forged_delivery_document())


if __name__ == "__main__":
    unittest.main()
