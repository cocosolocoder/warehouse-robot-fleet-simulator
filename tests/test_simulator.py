import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task, shortest_path


def demo_scenario() -> FleetSimulator:
    return FleetSimulator(
        GridMap(8, 6, frozenset({(3, 1), (3, 2), (3, 3), (5, 4)})),
        [Robot("R-01", (0, 0)), Robot("R-02", (7, 5))],
        [Task("T-100", (1, 4), (6, 0)), Task("T-200", (6, 5), (0, 2))],
    )


def run_until_done(simulator: FleetSimulator, limit: int = 30) -> FleetSimulator:
    for _ in range(limit):
        metrics = simulator.metrics()
        if metrics["tasks_completed"] == metrics["tasks_total"]:
            break
        simulator.step()
        metrics = simulator.metrics()
        if metrics["tasks_completed"] == metrics["tasks_total"]:
            break
    return simulator


def mid_delivery_scenario(steps: int = 2) -> FleetSimulator:
    """A run where both robots carry picked-up but undelivered cargo.

    The two robots drive on separate rows so no traffic interaction can
    interfere; after *steps* ticks both tasks are claimed and loaded, neither
    is completed, and the replay holds one frame per tick.
    """
    simulator = FleetSimulator(
        GridMap(6, 2),
        [Robot("R-1", (0, 0)), Robot("R-2", (5, 1))],
        [Task("T-1", (1, 0), (5, 0)), Task("T-2", (4, 1), (0, 1))],
    )
    for _ in range(steps):
        simulator.step()
    return simulator


class PathfindingTests(unittest.TestCase):
    def test_shortest_path_avoids_obstacle(self) -> None:
        grid = GridMap(3, 3, frozenset({(1, 0)}))
        path = shortest_path(grid, (0, 0), (2, 0))
        self.assertEqual(path[-1], (2, 0))
        self.assertNotIn((1, 0), path)
        self.assertEqual(len(path), 4)


class SimulatorTests(unittest.TestCase):
    def test_task_completes_and_metrics_are_recorded(self) -> None:
        simulator = FleetSimulator(
            GridMap(4, 2),
            [Robot("R-01", (0, 0))],
            [Task("T-01", (1, 0), (3, 0))],
        )
        for _ in range(3):
            simulator.step()
        self.assertTrue(simulator.tasks["T-01"].completed)
        self.assertEqual(simulator.metrics()["tasks_completed"], 1)
        self.assertEqual(simulator.metrics()["distance_total"], 3)


class CheckpointRoundTripTests(unittest.TestCase):
    def _write_and_load(self, simulator: FleetSimulator) -> FleetSimulator:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            simulator.save_checkpoint(path)
            return FleetSimulator.load_checkpoint(path)

    def test_save_does_not_advance_or_mutate_instance(self) -> None:
        simulator = run_until_done(demo_scenario(), 6)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            before = simulator.snapshot()
            simulator.save_checkpoint(path)
            after = simulator.snapshot()
            self.assertEqual(before, after)
            self.assertEqual(simulator.tick, before["tick"])

    def test_zero_tick_empty_fleet_empty_tasks(self) -> None:
        simulator = FleetSimulator(GridMap(3, 3), [], [])
        loaded = self._write_and_load(simulator)
        self.assertEqual(loaded.tick, 0)
        self.assertEqual(loaded.replay, [])
        self.assertEqual(loaded.robots, {})
        self.assertEqual(loaded.tasks, {})
        self.assertEqual(loaded.metrics()["completion_ratio"], 1.0)

    def test_zero_tick_with_robots_round_trips(self) -> None:
        simulator = FleetSimulator(
            GridMap(4, 4), [Robot("A", (0, 0))], [Task("T", (1, 0), (3, 0))]
        )
        loaded = self._write_and_load(simulator)
        self.assertEqual(loaded.tick, 0)
        self.assertEqual(loaded.replay, [])
        run_until_done(loaded)
        self.assertTrue(loaded.tasks["T"].completed)

    def test_picked_up_but_not_delivered_resumes(self) -> None:
        simulator = FleetSimulator(
            GridMap(4, 1), [Robot("A", (0, 0))], [Task("T", (1, 0), (3, 0))]
        )
        simulator.step()
        simulator.step()
        self.assertTrue(simulator.tasks["T"].picked_up)
        self.assertFalse(simulator.tasks["T"].completed)
        loaded = self._write_and_load(simulator)
        self.assertTrue(loaded.tasks["T"].picked_up)
        self.assertFalse(loaded.tasks["T"].completed)
        self.assertEqual(loaded.robots["A"].task_id, "T")
        self.assertEqual(loaded.robots["A"].route, [(3, 0)])
        loaded.step()
        self.assertTrue(loaded.tasks["T"].completed)

    def test_completed_task_keeps_history_robot_takes_new_work(self) -> None:
        simulator = FleetSimulator(
            GridMap(6, 1),
            [Robot("R", (0, 0))],
            [Task("T1", (1, 0), (2, 0)), Task("T2", (4, 0), (0, 0))],
        )
        run_until_done(simulator, 2)
        self.assertTrue(simulator.tasks["T1"].completed)
        loaded = self._write_and_load(simulator)
        self.assertEqual(loaded.tasks["T1"].assigned_robot, "R")
        self.assertIsNone(loaded.robots["R"].task_id)
        run_until_done(loaded)
        self.assertTrue(all(task.completed for task in loaded.tasks.values()))

    def test_unassigned_pending_task_waits_then_assigns_after_resume(self) -> None:
        # One robot: while it works T-1, T-2 has no idle robot and must remain
        # unassigned (rather than being reassigned) across the checkpoint.
        simulator = FleetSimulator(
            GridMap(6, 1),
            [Robot("R", (0, 0))],
            [Task("T-1", (1, 0), (2, 0)), Task("T-2", (4, 0), (0, 0))],
        )
        simulator.step()
        self.assertIsNone(simulator.tasks["T-2"].assigned_robot)
        loaded = self._write_and_load(simulator)
        self.assertIsNone(loaded.tasks["T-2"].assigned_robot)
        self.assertEqual(loaded.robots["R"].task_id, "T-1")
        run_until_done(loaded)
        self.assertTrue(loaded.tasks["T-1"].completed)
        self.assertTrue(loaded.tasks["T-2"].completed)
        self.assertEqual(loaded.tasks["T-2"].assigned_robot, "R")


class InterruptedParityTests(unittest.TestCase):
    def assert_resume_matches_uninterrupted(self, split: int) -> None:
        full = run_until_done(demo_scenario())
        interrupted = demo_scenario()
        run_until_done(interrupted, split)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            interrupted.save_checkpoint(path)
            resumed = run_until_done(FleetSimulator.load_checkpoint(path))
        self.assertEqual(resumed.snapshot(), full.snapshot())

    def test_parity_at_every_split_point(self) -> None:
        for split in range(15):
            with self.subTest(split=split):
                self.assert_resume_matches_uninterrupted(split)

    def test_per_tick_outputs_match_after_resume(self) -> None:
        full = run_until_done(demo_scenario())
        interrupted = run_until_done(demo_scenario(), 5)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            interrupted.save_checkpoint(path)
            resumed = FleetSimulator.load_checkpoint(path)
            for expected_event in full.replay[5:]:
                event = resumed.step()
                self.assertEqual(event, expected_event)

    def test_same_file_loaded_twice_runs_independently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            run_until_done(demo_scenario(), 3).save_checkpoint(path)
            first = FleetSimulator.load_checkpoint(path)
            second = FleetSimulator.load_checkpoint(path)
            first.step()
            first.tasks["T-100"].completed = True
            self.assertEqual(second.tick, 3)
            self.assertFalse(second.tasks["T-100"].completed)
            third = FleetSimulator.load_checkpoint(path)
            self.assertEqual(second.robots["R-01"].position, third.robots["R-01"].position)
            second.step()
            self.assertEqual(first.tick, second.tick)
            self.assertFalse(second.tasks["T-100"].completed)
            self.assertIsNot(first.replay, second.replay)


class CheckpointFormatTests(unittest.TestCase):
    def setUp(self) -> None:
        simulator = run_until_done(demo_scenario(), 4)
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "state.json")
        simulator.save_checkpoint(self.path)
        with open(self.path, encoding="utf-8") as handle:
            self.document = json.load(handle)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_versioned_utf8_json_document(self) -> None:
        raw = Path(self.path).read_bytes()
        raw.decode("utf-8")
        self.assertEqual(self.document["version"], 2)
        for key in (
            "version",
            "grid",
            "base_grid",
            "tick",
            "robots",
            "tasks",
            "replay",
            "map_changes",
            "paused_tasks",
        ):
            self.assertIn(key, self.document)

    def reload_expect_error(self, document: object, fragment: str) -> None:
        with self.assertRaises(ValueError) as context:
            FleetSimulator._from_checkpoint(document)
        self.assertIn(fragment, str(context.exception))

    def mutated(self, mutate) -> object:
        import copy

        document = copy.deepcopy(self.document)
        mutate(document)
        return document

    def test_malformed_json_is_value_error(self) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("{not valid json")
        with self.assertRaises(ValueError):
            FleetSimulator.load_checkpoint(self.path)

    def test_missing_file_is_os_error(self) -> None:
        with self.assertRaises(OSError):
            FleetSimulator.load_checkpoint(os.path.join(self.tmp.name, "missing.json"))

    def test_missing_required_fields(self) -> None:
        for field in ("version", "grid", "tick", "robots", "tasks", "replay"):
            document = self.mutated(lambda d, f=field: d.pop(f))
            self.reload_expect_error(document, field)

    def test_unsupported_or_typed_wrong_version(self) -> None:
        self.reload_expect_error(self.mutated(lambda d: d.update(version=3)), "version")
        self.reload_expect_error(self.mutated(lambda d: d.update(version="2")), "version")
        self.reload_expect_error(self.mutated(lambda d: d.update(version=True)), "version")

    def test_field_type_errors(self) -> None:
        self.reload_expect_error(self.mutated(lambda d: d.update(tick="4")), "tick")
        self.reload_expect_error(self.mutated(lambda d: d.update(tick=-1)), "tick")
        self.reload_expect_error(self.mutated(lambda d: d.update(robots={})), "robots")
        self.reload_expect_error(self.mutated(lambda d: d.update(tasks={})), "tasks")
        self.reload_expect_error(self.mutated(lambda d: d.update(replay={})), "replay")
        self.reload_expect_error(
            self.mutated(lambda d: d["grid"].update(width=0)), "width"
        )

    def test_duplicate_ids_and_overlapping_positions(self) -> None:
        import copy

        document = self.mutated(
            lambda d: d["robots"].append(copy.deepcopy(d["robots"][0]))
        )
        self.reload_expect_error(document, "duplicate robot")
        document = self.mutated(
            lambda d: d["tasks"].append(copy.deepcopy(d["tasks"][0]))
        )
        self.reload_expect_error(document, "duplicate task")
        document = self.mutated(
            lambda d: d["robots"][1].__setitem__(
                "position", d["robots"][0]["position"]
            )
        )
        self.reload_expect_error(document, "share a position")

    def test_robot_position_out_of_bounds_or_blocked(self) -> None:
        self.reload_expect_error(
            self.mutated(lambda d: d["robots"][0].__setitem__("position", [99, 0])),
            "outside the map",
        )
        self.reload_expect_error(
            self.mutated(lambda d: d["robots"][0].__setitem__("position", [3, 1])),
            "outside the map",
        )

    def test_route_out_of_bounds_through_obstacle_nonadjacent(self) -> None:
        self.reload_expect_error(
            self.mutated(lambda d: d["robots"][0]["route"].__setitem__(0, [99, 0])),
            "outside the map",
        )
        self.reload_expect_error(
            self.mutated(lambda d: d["robots"][0]["route"].__setitem__(0, [3, 1])),
            "outside the map",
        )
        self.reload_expect_error(
            self.mutated(lambda d: d["robots"][0]["route"].__setitem__(0, [7, 5])),
            "non-adjacently",
        )

    def test_task_ownership_inconsistency(self) -> None:
        self.reload_expect_error(
            self.mutated(lambda d: d["robots"][0].__setitem__("task_id", "NOPE")),
            "unknown task",
        )
        self.reload_expect_error(
            self.mutated(lambda d: d["robots"][0].__setitem__("task_id", "T-200")),
            "assigned to",
        )
        self.reload_expect_error(
            self.mutated(
                lambda d: d["tasks"].append(
                    {
                        "task_id": "TX",
                        "pickup": [0, 0],
                        "dropoff": [1, 0],
                        "assigned_robot": "NOPE",
                        "picked_up": False,
                        "completed": False,
                    }
                )
            ),
            "unknown robot",
        )

    def test_replay_tick_continuity_and_last_frame(self) -> None:
        self.reload_expect_error(
            self.mutated(lambda d: d["replay"][0].__setitem__("tick", 2)),
            "consecutively",
        )
        shorter = self.mutated(lambda d: d.update(replay=d["replay"][:-1]))
        self.reload_expect_error(shorter, "frames")
        self.reload_expect_error(
            self.mutated(
                lambda d: d["replay"][-1]["robots"]["R-01"].__setitem__(0, 0)
            ),
            "last replay frame",
        )

    def test_zero_tick_needs_no_last_frame(self) -> None:
        simulator = FleetSimulator(GridMap(3, 3), [Robot("A", (0, 0))], [])
        path = os.path.join(self.tmp.name, "zero.json")
        simulator.save_checkpoint(path)
        loaded = FleetSimulator.load_checkpoint(path)
        self.assertEqual(loaded.tick, 0)


class AtomicSaveTests(unittest.TestCase):
    def test_existing_file_kept_when_write_fails(self) -> None:
        simulator = run_until_done(demo_scenario(), 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            simulator.save_checkpoint(path)
            original = Path(path).read_bytes()
            with self.assertRaises(OSError):
                simulator.save_checkpoint(os.path.join(tmp, "missing-dir", "x.json"))
            self.assertEqual(Path(path).read_bytes(), original)
            leftovers = [
                name for name in os.listdir(tmp) if name.startswith(".checkpoint-")
            ]
            self.assertEqual(leftovers, [])

    def test_failed_save_to_new_target_leaves_nothing(self) -> None:
        simulator = run_until_done(demo_scenario(), 2)
        with tempfile.TemporaryDirectory() as tmp:
            target_dir = os.path.join(tmp, "sub")
            with self.assertRaises(OSError):
                simulator.save_checkpoint(os.path.join(target_dir, "x.json"))
            self.assertFalse(os.path.exists(target_dir))
            self.assertEqual(os.listdir(tmp), [])

    def test_saving_twice_keeps_latest(self) -> None:
        simulator = demo_scenario()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            simulator.save_checkpoint(path)
            simulator.step()
            simulator.save_checkpoint(path)
            loaded = FleetSimulator.load_checkpoint(path)
            self.assertEqual(loaded.tick, 1)
            self.assertEqual(loaded.snapshot(), simulator.snapshot())


class _PartialWrite:
    """File proxy that stores only part of the payload, then fails the write.

    Stands in for a filesystem error that strikes after the new checkpoint
    content has started flowing to disk but before it is complete.
    """

    def __init__(self, handle) -> None:
        self._handle = handle

    def write(self, data: bytes) -> int:
        self._handle.write(data[: max(1, len(data) // 3)])
        raise OSError("simulated write failure")

    def __getattr__(self, name: str):
        return getattr(self._handle, name)


class AtomicSaveFailureTests(unittest.TestCase):
    """Failures after a save has started must not corrupt the destination."""

    @staticmethod
    def _failure_patchers() -> dict[str, object]:
        """One factory per failure point between temp creation and replace."""
        real_fdopen = os.fdopen

        def partial_fdopen(fd, mode, *args, **kwargs):
            return _PartialWrite(real_fdopen(fd, mode, *args, **kwargs))

        return {
            # The write itself dies partway through the payload.
            "write": lambda: mock.patch("os.fdopen", new=partial_fdopen),
            # Content is written but syncing it to disk fails pre-replace.
            "sync": lambda: mock.patch(
                "os.fsync", side_effect=OSError("simulated sync failure")
            ),
            # New content is fully on disk but the atomic replace fails.
            "replace": lambda: mock.patch(
                "os.replace", side_effect=OSError("simulated replace failure")
            ),
        }

    def _assert_mid_delivery(self, simulator: FleetSimulator, tick: int) -> None:
        # The saved state carries business meaning: both tasks are claimed
        # and their cargo loaded, yet neither delivery has completed.
        self.assertEqual(simulator.tick, tick)
        for task_id in ("T-1", "T-2"):
            task = simulator.tasks[task_id]
            self.assertIsNotNone(task.assigned_robot)
            self.assertTrue(task.picked_up)
            self.assertFalse(task.completed)
        self.assertEqual(len(simulator.replay), tick)

    def _assert_state_unchanged(self, simulator: FleetSimulator, before: dict) -> None:
        # One snapshot covers everything a failed save must not touch: the
        # clock, robot positions, remaining routes, task claim/load/completion
        # flags, the replay and the recorded metrics.
        self.assertEqual(simulator.snapshot(), before)

    def test_failed_save_preserves_existing_checkpoint(self) -> None:
        for name, make_patcher in self._failure_patchers().items():
            with self.subTest(failure=name):
                self._check_existing_target_preserved(make_patcher)

    def test_failed_save_to_new_target_leaves_nothing(self) -> None:
        for name, make_patcher in self._failure_patchers().items():
            with self.subTest(failure=name):
                self._check_new_target_left_absent(make_patcher)

    def _check_existing_target_preserved(self, make_patcher) -> None:
        simulator = mid_delivery_scenario(steps=2)
        self._assert_mid_delivery(simulator, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            simulator.save_checkpoint(path)
            old_bytes = Path(path).read_bytes()
            # snapshot() aliases the live replay list, so freeze a deep copy
            # before the clock advances.
            old_snapshot = copy.deepcopy(simulator.snapshot())

            # Advance so the pending save clearly differs from the old file.
            simulator.step()
            simulator.step()
            self._assert_mid_delivery(simulator, 4)
            self.assertNotEqual(simulator.snapshot(), old_snapshot)

            unrelated = os.path.join(tmp, "unrelated.txt")
            Path(unrelated).write_text("keep me", encoding="utf-8")

            before = copy.deepcopy(simulator.snapshot())
            with make_patcher():
                with self.assertRaises(OSError):
                    simulator.save_checkpoint(path)
            self._assert_state_unchanged(simulator, before)

            # The destination keeps the old checkpoint byte for byte, and
            # reading it back yields the old progress with nothing of the
            # pending state (positions, tasks, replay) mixed in.
            self.assertEqual(Path(path).read_bytes(), old_bytes)
            loaded = FleetSimulator.load_checkpoint(path)
            self.assertEqual(loaded.snapshot(), old_snapshot)
            self.assertEqual(loaded.tick, 2)
            self.assertNotEqual(loaded.robots["R-1"].position, simulator.robots["R-1"].position)
            self.assertEqual(len(loaded.replay), 2)

            # No half-written temp file survives; unrelated files are intact.
            self.assertEqual(sorted(os.listdir(tmp)), ["state.json", "unrelated.txt"])
            self.assertEqual(Path(unrelated).read_text(encoding="utf-8"), "keep me")

            # Once the error clears the same simulator saves successfully:
            # the old file is fully superseded by a readable new checkpoint.
            simulator.save_checkpoint(path)
            self.assertNotEqual(Path(path).read_bytes(), old_bytes)
            recovered = FleetSimulator.load_checkpoint(path)
            self.assertEqual(recovered.snapshot(), simulator.snapshot())
            self.assertEqual(recovered.tick, 4)

    def _check_new_target_left_absent(self, make_patcher) -> None:
        simulator = mid_delivery_scenario(steps=2)
        self._assert_mid_delivery(simulator, 2)
        with tempfile.TemporaryDirectory() as tmp:
            unrelated = os.path.join(tmp, "unrelated.txt")
            Path(unrelated).write_text("keep me", encoding="utf-8")
            path = os.path.join(tmp, "state.json")

            before = copy.deepcopy(simulator.snapshot())
            with make_patcher():
                with self.assertRaises(OSError):
                    simulator.save_checkpoint(path)
            self._assert_state_unchanged(simulator, before)

            # The target stays absent, the temp file is cleaned up and the
            # unrelated neighbour is untouched.
            self.assertFalse(os.path.exists(path))
            self.assertEqual(os.listdir(tmp), ["unrelated.txt"])
            self.assertEqual(Path(unrelated).read_text(encoding="utf-8"), "keep me")

            # After the error clears the same simulator saves successfully
            # and the fresh checkpoint loads back the same progress.
            simulator.save_checkpoint(path)
            recovered = FleetSimulator.load_checkpoint(path)
            self.assertEqual(recovered.snapshot(), simulator.snapshot())
            self.assertEqual(recovered.tick, 2)


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        repo_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(repo_root))
        return subprocess.run(
            [sys.executable, "-m", "warehouse_fleet", *arguments],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

    def test_demo_output_shape_is_unchanged(self) -> None:
        result = self.run_cli("demo", "--steps", "12")
        self.assertEqual(result.returncode, 0, result.stderr)
        snapshot = json.loads(result.stdout)
        self.assertEqual(
            sorted(snapshot), ["metrics", "replay", "robots", "tasks", "tick"]
        )

    def test_resume_reproduces_full_demo(self) -> None:
        full = self.run_cli("demo", "--steps", "20")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = os.path.join(tmp, "state.json")
            self.run_cli("demo", "--steps", "6", "--checkpoint", checkpoint)
            resumed = self.run_cli("resume", checkpoint, "--steps", "20")
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(json.loads(resumed.stdout), json.loads(full.stdout))

    def test_resume_zero_and_already_complete_print_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = os.path.join(tmp, "state.json")
            done = self.run_cli("demo", "--steps", "20", "--checkpoint", checkpoint)
            for args in (
                ("resume", checkpoint, "--steps", "0"),
                ("resume", checkpoint, "--steps", "5"),
            ):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), json.loads(done.stdout))

    def test_resume_without_checkpoint_leaves_source_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = os.path.join(tmp, "state.json")
            self.run_cli("demo", "--steps", "4", "--checkpoint", checkpoint)
            before = Path(checkpoint).read_bytes()
            self.run_cli("resume", checkpoint, "--steps", "3")
            self.assertEqual(Path(checkpoint).read_bytes(), before)

    def test_resume_checkpoint_same_path_updates_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = os.path.join(tmp, "state.json")
            self.run_cli("demo", "--steps", "4", "--checkpoint", checkpoint)
            self.run_cli(
                "resume", checkpoint, "--steps", "4", "--checkpoint", checkpoint
            )
            loaded = FleetSimulator.load_checkpoint(checkpoint)
            self.assertEqual(loaded.tick, 8)

    def test_corrupt_and_missing_files_fail_nonzero_without_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corrupt = os.path.join(tmp, "corrupt.json")
            with open(corrupt, "w", encoding="utf-8") as handle:
                handle.write("{broken")
            result = self.run_cli("resume", corrupt, "--steps", "3")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("resume:", result.stderr)

            result = self.run_cli(
                "resume", os.path.join(tmp, "missing.json"), "--steps", "3"
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
