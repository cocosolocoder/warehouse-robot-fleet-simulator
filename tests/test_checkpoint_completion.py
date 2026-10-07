"""Replay completion history must agree with the saved task states.

A tick frame's ``completed`` list is the cumulative set of tasks finished by
the end of that tick. Loading must reject frames that name unknown tasks,
repeat a task within one frame, drop a previously recorded completion, claim
a task completed in a frame whose recorded delivery robot is not standing on
the task dropoff, or end on a set that differs from the tasks saved as
completed -- for both checkpoint versions -- while accepting first frames
that already list completions, zero-tick checkpoints with initially completed
tasks and trailing map edits.
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


def build_custom_document(*, robots, tasks, width=5, height=2):
    """A real zero-tick checkpoint for an arbitrary fleet/task set.

    *robots* is a sequence of ``(id, (x, y))`` and *tasks* of
    ``(id, (pickup_x, pickup_y), (dropoff_x, dropoff_y))``; everything not
    under test keeps the shape of a genuinely saved document.
    """
    simulator = FleetSimulator(
        GridMap(width, height),
        [Robot(robot_id, position) for robot_id, position in robots],
        [Task(task_id, pickup, dropoff) for task_id, pickup, dropoff in tasks],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        simulator.save_checkpoint(path)
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)


def with_history(document, frames, *, completed=(), finishers=None):
    """Attach a replay history and mark *completed* tasks as done.

    *finishers* maps a completed task to the robot recorded as its delivery
    robot; tasks without an entry keep the helper default ``"A"``.
    """
    finishers = finishers or {}
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
        record["assigned_robot"] = finishers.get(task_id, "A")
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
        # A delivers T1 while standing on its dropoff (2, 0); B delivers T2
        # from the neighbouring cell onto its own dropoff (4, 0). Both
        # completions are first recorded in frame 2, and frame 3 merely
        # repeats the cumulative set.
        document = with_history(
            build_document(),
            [
                tick_frame(
                    1, ["A", "B"], {"A": (1, 0), "B": (4, 0)}
                ),
                tick_frame(
                    2, ["A"], {"A": (2, 0), "B": (4, 0)}, ["T1", "T2"]
                ),
                tick_frame(
                    3, [], {"A": (2, 0), "B": (4, 0)}, ["T1", "T2"]
                ),
            ],
            completed=("T1", "T2"),
            finishers={"T2": "B"},
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

    def test_order_within_frame_and_repeats_across_frames_are_legal(self):
        # T1 is grandfathered by the first frame even with A nowhere near its
        # dropoff; T2 first appears in frame 3, where B -- its recorded
        # delivery robot -- waits on T2's dropoff (4, 0). B moved there a tick
        # earlier, so finishing needs no entry in frame 3's moved list.
        document = with_history(
            build_document(),
            [
                tick_frame(
                    1, ["B"], {"A": (0, 0), "B": (4, 0)}, ["T1"]
                ),
                tick_frame(
                    2, [], {"A": (0, 0), "B": (4, 0)}, ["T1"]
                ),
                tick_frame(
                    3, [], {"A": (0, 0), "B": (4, 0)}, ["T2", "T1"]
                ),
            ],
            completed=("T1", "T2"),
            finishers={"T2": "B"},
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
        # A reaches T1's dropoff (2, 0) in frame 2, where T1 is first marked
        # completed; the saved positions match the final frame.
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
        # T1 rides the first-frame exemption; T2 is finished in frame 2 by B,
        # which steps onto T2's dropoff (4, 0) during that frame.
        document = as_version_one(
            with_history(
                build_document(),
                [
                    tick_frame(
                        1, ["B"], {"A": (0, 0), "B": (4, 0)}, ["T1"]
                    ),
                    tick_frame(
                        2, [], {"A": (0, 0), "B": (4, 0)}, ["T1", "T2"]
                    ),
                ],
                completed=("T1", "T2"),
                finishers={"T2": "B"},
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


class CompletionDropoffPresenceTests(unittest.TestCase):
    """A completion first recorded after the first frame needs its robot.

    From the second tick frame on, every task newly appearing in a frame's
    completed list must find its retained delivery robot in that frame's
    positions, standing on the task dropoff. First-frame and zero-tick
    completions keep their historical exemption, and later frames prove
    nothing once the completion frame itself was legal.
    """

    def test_completion_recorded_while_robot_is_off_dropoff_rejected(self):
        # The scenario from the rule itself: T's dropoff is (2, 0) and R is
        # its recorded delivery robot, but the frame that first lists T as
        # completed leaves R at (0, 0).
        document = build_custom_document(
            robots=(("R", (0, 0)),),
            tasks=(("T", (0, 0), (2, 0)),),
        )
        document = with_history(
            document,
            [
                tick_frame(1, [], {"R": (0, 0)}),
                tick_frame(2, [], {"R": (0, 0)}, ["T"]),
            ],
            completed=("T",),
            finishers={"T": "R"},
        )
        message = expect_rejected(
            self, document, "2", "T", "R", "dropoff", "[0, 0]", "[2, 0]"
        )
        self.assertIn("newly lists", message)

    def test_saved_current_position_cannot_substitute_for_completion_frame(self):
        # The robot only reaches the dropoff in a later frame; the saved
        # current position equals the dropoff, but the completion was claimed
        # two frames early while the robot was still at (0, 0).
        document = with_history(
            build_document(),
            [
                tick_frame(1, [], {"A": (0, 0), "B": (4, 1)}),
                tick_frame(2, [], {"A": (0, 0), "B": (4, 1)}, ["T1"]),
                tick_frame(3, ["A"], {"A": (1, 0), "B": (4, 1)}, ["T1"]),
                tick_frame(4, ["A"], {"A": (2, 0), "B": (4, 1)}, ["T1"]),
            ],
            completed=("T1",),
        )
        expect_rejected(self, document, "2", "T1", "A", "[0, 0]", "[2, 0]")

    def test_recorded_delivery_robot_missing_from_fleet_rejected(self):
        # A completed task may keep a historical robot that has left the
        # fleet, but a completion that first appears after the first frame
        # still needs that robot's position in the frame -- it cannot be
        # reconstructed from anything else.
        document = with_history(
            build_document(),
            [
                tick_frame(1, [], POSITIONS),
                tick_frame(2, [], POSITIONS, ["T1"]),
            ],
            completed=("T1",),
        )
        next(record for record in document["tasks"] if record["task_id"] == "T1")[
            "assigned_robot"
        ] = "GHOST"
        expect_rejected(self, document, "2", "T1", "GHOST", "missing")

    def test_robot_may_leave_dropoff_after_the_completion_frame(self):
        # Frame 2 is the frame of proof: A stands on T1's dropoff while T1
        # first appears completed. Afterwards A drives away (frame 3) and the
        # later frames keep listing T1 without A ever returning -- that stays
        # valid.
        document = with_history(
            build_document(),
            [
                tick_frame(1, ["A"], {"A": (1, 0), "B": (4, 1)}),
                tick_frame(2, ["A"], {"A": (2, 0), "B": (4, 1)}, ["T1"]),
                tick_frame(3, ["A"], {"A": (3, 0), "B": (4, 1)}, ["T1"]),
            ],
            completed=("T1",),
        )
        loaded = load(document)
        self.assertTrue(loaded.tasks["T1"].completed)
        self.assertEqual(loaded.robots["A"].position, (3, 0))

    def test_completion_in_a_tick_without_a_move_is_legal(self):
        # A is already parked on the dropoff in frame 1; frame 2 is a pure
        # wait (empty moved list) and T1 first appears completed there.
        document = with_history(
            build_document(),
            [
                tick_frame(1, [], {"A": (2, 0), "B": (4, 1)}),
                tick_frame(2, [], {"A": (2, 0), "B": (4, 1)}, ["T1"]),
            ],
            completed=("T1",),
        )
        loaded = load(document)
        self.assertTrue(loaded.tasks["T1"].completed)
        self.assertEqual(loaded.replay[1]["moved"], [])

    def test_interspersed_map_change_cannot_stand_in_for_the_frame(self):
        # The edit between the frames is not a time step and carries no robot
        # positions; frame 2 still has to show the delivery robot on the
        # dropoff itself.
        document = with_history(
            build_document(),
            [
                tick_frame(1, [], POSITIONS),
                map_change(1, 1, added=((3, 0),)),
                tick_frame(2, [], POSITIONS, ["T1"]),
            ],
            completed=("T1",),
        )
        document["map_changes"] = [map_change(1, 1, added=((3, 0),))]
        document["grid"]["obstacles"] = [[3, 0]]
        expect_rejected(self, document, "2", "T1", "A", "dropoff")

    def test_version_one_off_dropoff_completion_rejected(self):
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

    def test_first_frame_completion_with_absent_historical_robot_loads(self):
        # Already completed at recording start: neither the dropoff presence
        # nor the robot's continued fleet membership is investigated.
        document = with_history(
            build_document(),
            [tick_frame(1, [], POSITIONS, ["T1"])],
            completed=("T1",),
        )
        next(record for record in document["tasks"] if record["task_id"] == "T1")[
            "assigned_robot"
        ] = "RETIRED"
        loaded = load(document)
        self.assertTrue(loaded.tasks["T1"].completed)

    def test_tick_zero_completion_with_absent_historical_robot_loads(self):
        document = with_history(build_document(), [], completed=("T1",))
        next(record for record in document["tasks"] if record["task_id"] == "T1")[
            "assigned_robot"
        ] = "RETIRED"
        loaded = load(document)
        self.assertEqual(loaded.tick, 0)
        self.assertTrue(loaded.tasks["T1"].completed)

    def test_real_delivery_round_trip_keeps_tasks_replay_and_metrics(self):
        simulator = FleetSimulator(
            GridMap(5, 1),
            [Robot("A", (0, 0))],
            [Task("T1", pickup=(0, 0), dropoff=(2, 0))],
        )
        simulator.step()
        simulator.step()  # A arrives at (2, 0) and T1 completes
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            simulator.save_checkpoint(path)
            loaded = FleetSimulator.load_checkpoint(path)
        self.assertEqual(loaded.tick, 2)
        self.assertTrue(loaded.tasks["T1"].completed)
        self.assertEqual(loaded.tasks["T1"].assigned_robot, "A")
        self.assertEqual(loaded.robots["A"].position, (2, 0))
        self.assertEqual(loaded.replay, simulator.replay)
        self.assertEqual(loaded.metrics(), simulator.metrics())


class ResumeCliCompletionTests(unittest.TestCase):
    def resume(self, steps):
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
            self.assertIn("T1", stderr.getvalue())
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), original)
            self.assertFalse(os.path.exists(output))

    def test_resume_rejects_contradictory_completion_history(self):
        self.resume(5)

    def test_resume_with_zero_steps_still_validates(self):
        self.resume(0)

    def resume_off_dropoff(self, steps):
        # A saved-as-completed delivery whose completion first appears in a
        # frame that does not show the delivery robot on the dropoff. The
        # document must fail to load even before a single resume tick runs.
        document = build_custom_document(
            robots=(("R", (0, 0)),),
            tasks=(("T", (0, 0), (2, 0)),),
        )
        document = with_history(
            document,
            [
                tick_frame(1, [], {"R": (0, 0)}),
                tick_frame(2, [], {"R": (0, 0)}, ["T"]),
            ],
            completed=("T",),
            finishers={"T": "R"},
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
            error = stderr.getvalue()
            self.assertIn("resume:", error)
            for fragment in ("2", "T", "R", "dropoff"):
                self.assertIn(fragment, error)
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), original)
            self.assertFalse(os.path.exists(output))

    def test_resume_rejects_off_dropoff_delivery_record(self):
        self.resume_off_dropoff(5)

    def test_resume_off_dropoff_record_rejected_with_zero_steps(self):
        self.resume_off_dropoff(0)


if __name__ == "__main__":
    unittest.main()
