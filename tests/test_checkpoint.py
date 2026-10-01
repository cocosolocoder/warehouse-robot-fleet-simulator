import json
import os
import subprocess
import sys
import tempfile
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def demo_simulator() -> FleetSimulator:
    grid = GridMap(8, 6, frozenset({(3, 1), (3, 2), (3, 3), (5, 4)}))
    return FleetSimulator(
        grid,
        [Robot("R-01", (0, 0)), Robot("R-02", (7, 5))],
        [Task("T-100", (1, 4), (6, 0)), Task("T-200", (6, 5), (0, 2))],
    )


def run_to_completion(simulator: FleetSimulator, limit: int = 30) -> None:
    for _ in range(limit):
        metrics = simulator.metrics()
        if metrics["tasks_completed"] == metrics["tasks_total"]:
            break
        simulator.step()


class CheckpointRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "checkpoint.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_tick_zero_round_trip(self) -> None:
        simulator = FleetSimulator(GridMap(4, 2), [Robot("R-01", (0, 0))], [Task("T-01", (1, 0), (3, 0))])
        simulator.save_checkpoint(self.path)
        loaded = FleetSimulator.load_checkpoint(self.path)
        self.assertEqual(loaded.snapshot(), simulator.snapshot())
        self.assertEqual(loaded.tick, 0)

    def test_empty_fleet_and_tasks(self) -> None:
        simulator = FleetSimulator(GridMap(3, 3), [], [])
        for _ in range(3):
            simulator.step()
        simulator.save_checkpoint(self.path)
        loaded = FleetSimulator.load_checkpoint(self.path)
        self.assertEqual(loaded.snapshot(), simulator.snapshot())
        self.assertEqual(loaded.metrics()["completion_ratio"], 1.0)

    def test_picked_up_but_not_delivered(self) -> None:
        simulator = FleetSimulator(
            GridMap(4, 1), [Robot("R-01", (0, 0))], [Task("T-01", (1, 0), (3, 0))]
        )
        simulator.step()
        simulator.step()
        task = simulator.tasks["T-01"]
        self.assertTrue(task.picked_up)
        self.assertFalse(task.completed)
        simulator.save_checkpoint(self.path)
        loaded = FleetSimulator.load_checkpoint(self.path)
        self.assertEqual(loaded.snapshot(), simulator.snapshot())
        run_to_completion(loaded)
        self.assertTrue(loaded.tasks["T-01"].completed)

    def test_completed_task_keeps_historical_owner(self) -> None:
        simulator = FleetSimulator(
            GridMap(4, 1),
            [Robot("R-01", (0, 0))],
            [Task("T-01", (1, 0), (3, 0)), Task("T-02", (3, 0), (0, 0))],
        )
        run_to_completion(simulator)
        self.assertTrue(simulator.tasks["T-01"].completed)
        self.assertEqual(simulator.tasks["T-01"].assigned_robot, "R-01")
        simulator.save_checkpoint(self.path)
        loaded = FleetSimulator.load_checkpoint(self.path)
        self.assertEqual(loaded.snapshot(), simulator.snapshot())

    def test_unreachable_task_waits_for_assignment(self) -> None:
        simulator = FleetSimulator(
            GridMap(3, 1, frozenset({(2, 0)})),
            [Robot("R-01", (0, 0))],
            [Task("T-01", (2, 0), (2, 0))],
        )
        for _ in range(3):
            simulator.step()
        self.assertIsNone(simulator.tasks["T-01"].assigned_robot)
        simulator.save_checkpoint(self.path)
        loaded = FleetSimulator.load_checkpoint(self.path)
        self.assertEqual(loaded.snapshot(), simulator.snapshot())
        self.assertIsNone(loaded.tasks["T-01"].assigned_robot)

    def test_resume_matches_uninterrupted_run(self) -> None:
        full = demo_simulator()
        run_to_completion(full)

        partial = demo_simulator()
        for _ in range(6):
            partial.step()
        partial.save_checkpoint(self.path)
        resumed = FleetSimulator.load_checkpoint(self.path)
        for _ in range(14):
            resumed.step()
            if resumed.metrics()["tasks_completed"] == resumed.metrics()["tasks_total"]:
                break
        self.assertEqual(resumed.snapshot(), full.snapshot())

    def test_resume_at_every_tick_matches_uninterrupted_run(self) -> None:
        full = demo_simulator()
        run_to_completion(full)
        for tick in range(full.tick + 1):
            with self.subTest(tick=tick):
                partial = demo_simulator()
                for _ in range(tick):
                    partial.step()
                partial.save_checkpoint(self.path)
                resumed = FleetSimulator.load_checkpoint(self.path)
                for _ in range(full.tick - tick):
                    resumed.step()
                    if resumed.metrics()["tasks_completed"] == resumed.metrics()["tasks_total"]:
                        break
                self.assertEqual(resumed.snapshot(), full.snapshot())

    def test_save_does_not_mutate_simulator(self) -> None:
        simulator = demo_simulator()
        for _ in range(5):
            simulator.step()
        before = simulator.snapshot()
        simulator.save_checkpoint(self.path)
        self.assertEqual(simulator.snapshot(), before)
        self.assertEqual(simulator.tick, 5)

    def test_loaded_instances_are_independent(self) -> None:
        simulator = demo_simulator()
        for _ in range(6):
            simulator.step()
        simulator.save_checkpoint(self.path)
        first = FleetSimulator.load_checkpoint(self.path)
        second = FleetSimulator.load_checkpoint(self.path)
        self.assertEqual(first.snapshot(), second.snapshot())
        first.step()
        # Stepping one instance must not affect the other: first advanced,
        # second stays at the saved tick.
        self.assertNotEqual(first.snapshot(), second.snapshot())
        self.assertEqual(second.tick, 6)
        # The untouched instance keeps its own state and steps independently.
        second.step()
        self.assertEqual(second.tick, 7)
        # Both started from the same saved state and stepped once, so
        # determinism makes them equal again.
        self.assertEqual(first.snapshot(), second.snapshot())

    def test_checkpoint_file_is_versioned_utf8_json(self) -> None:
        simulator = demo_simulator()
        simulator.save_checkpoint(self.path)
        with open(self.path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["version"], 1)
        self.assertEqual(
            set(document),
            {"version", "grid", "robots", "tasks", "tick", "replay"},
        )


class CheckpointValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "checkpoint.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def base_document(self) -> dict[str, object]:
        return {
            "version": 1,
            "grid": {"width": 4, "height": 2, "obstacles": []},
            "robots": [
                {
                    "robot_id": "R-01",
                    "position": [0, 0],
                    "route": [],
                    "task_id": None,
                    "distance_travelled": 0,
                }
            ],
            "tasks": [
                {
                    "task_id": "T-01",
                    "pickup": [1, 0],
                    "dropoff": [3, 0],
                    "assigned_robot": None,
                    "picked_up": False,
                    "completed": False,
                }
            ],
            "tick": 0,
            "replay": [],
        }

    def reject(self, document: object) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        with self.assertRaises(ValueError):
            FleetSimulator.load_checkpoint(self.path)

    def test_corrupt_json(self) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("{not valid json")
        with self.assertRaises(ValueError):
            FleetSimulator.load_checkpoint(self.path)

    def test_root_must_be_object(self) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("[]")
        with self.assertRaises(ValueError):
            FleetSimulator.load_checkpoint(self.path)

    def test_missing_required_fields(self) -> None:
        for field in ("version", "grid", "robots", "tasks", "tick", "replay"):
            document = self.base_document()
            del document[field]
            self.reject(document)
        document = self.base_document()
        del document["grid"]["width"]
        self.reject(document)
        document = self.base_document()
        del document["robots"][0]["route"]
        self.reject(document)
        document = self.base_document()
        del document["tasks"][0]["picked_up"]
        self.reject(document)

    def test_wrong_field_types(self) -> None:
        document = self.base_document()
        document["version"] = "1"
        self.reject(document)
        document = self.base_document()
        document["version"] = True
        self.reject(document)
        document = self.base_document()
        document["tick"] = 1.5
        self.reject(document)
        document = self.base_document()
        document["tick"] = -1
        self.reject(document)
        document = self.base_document()
        document["grid"]["width"] = "4"
        self.reject(document)
        document = self.base_document()
        document["robots"][0]["position"] = [0]
        self.reject(document)
        document = self.base_document()
        document["robots"][0]["position"] = ["0", 0]
        self.reject(document)
        document = self.base_document()
        document["robots"][0]["position"] = [True, 0]
        self.reject(document)
        document = self.base_document()
        document["robots"][0]["route"] = [[0, 0, 0]]
        self.reject(document)
        document = self.base_document()
        document["robots"][0]["task_id"] = 5
        self.reject(document)
        document = self.base_document()
        document["robots"][0]["distance_travelled"] = -1
        self.reject(document)
        document = self.base_document()
        document["tasks"][0]["picked_up"] = "yes"
        self.reject(document)
        document = self.base_document()
        document["tasks"][0]["completed"] = 0
        self.reject(document)
        document = self.base_document()
        document["replay"] = {}
        self.reject(document)

    def test_unsupported_version(self) -> None:
        document = self.base_document()
        document["version"] = 2
        self.reject(document)

    def test_duplicate_robot_and_task_ids(self) -> None:
        document = self.base_document()
        document["robots"].append(dict(document["robots"][0]))
        self.reject(document)
        document = self.base_document()
        document["tasks"].append(dict(document["tasks"][0]))
        self.reject(document)

    def test_robot_overlap(self) -> None:
        document = self.base_document()
        document["robots"].append(
            {
                "robot_id": "R-02",
                "position": [0, 0],
                "route": [],
                "task_id": None,
                "distance_travelled": 0,
            }
        )
        self.reject(document)

    def test_robot_out_of_bounds_or_on_obstacle(self) -> None:
        document = self.base_document()
        document["robots"][0]["position"] = [4, 0]
        self.reject(document)
        document = self.base_document()
        document["robots"][0]["position"] = [-1, 0]
        self.reject(document)
        document = self.base_document()
        document["grid"]["obstacles"] = [[0, 0]]
        self.reject(document)

    def test_route_out_of_bounds_or_through_obstacle(self) -> None:
        document = self.base_document()
        document["robots"][0]["route"] = [[4, 0]]
        self.reject(document)
        document = self.base_document()
        document["grid"]["obstacles"] = [[2, 0]]
        document["robots"][0]["route"] = [[1, 0], [2, 0]]
        self.reject(document)

    def test_route_non_adjacent_move(self) -> None:
        document = self.base_document()
        document["robots"][0]["route"] = [[2, 0]]
        self.reject(document)
        document = self.base_document()
        document["robots"][0]["route"] = [[0, 1], [2, 1]]
        self.reject(document)

    def test_ownership_inconsistency(self) -> None:
        document = self.base_document()
        document["robots"][0]["task_id"] = "T-01"
        self.reject(document)
        document = self.base_document()
        document["tasks"][0]["assigned_robot"] = "R-01"
        self.reject(document)
        document = self.base_document()
        document["tasks"][0]["assigned_robot"] = "R-99"
        self.reject(document)
        document = self.base_document()
        document["robots"][0]["task_id"] = "T-99"
        self.reject(document)
        document = self.base_document()
        document["robots"].append(
            {
                "robot_id": "R-02",
                "position": [3, 1],
                "route": [],
                "task_id": None,
                "distance_travelled": 0,
            }
        )
        document["robots"][0]["task_id"] = "T-01"
        document["tasks"][0]["assigned_robot"] = "R-02"
        self.reject(document)

    def test_replay_must_run_from_one_to_tick(self) -> None:
        document = self.base_document()
        document["tick"] = 2
        document["replay"] = [
            {"tick": 1, "moved": [], "robots": {"R-01": [0, 0]}, "completed": []}
        ]
        self.reject(document)
        document = self.base_document()
        document["tick"] = 1
        document["replay"] = [
            {"tick": 2, "moved": [], "robots": {"R-01": [0, 0]}, "completed": []}
        ]
        self.reject(document)

    def test_replay_references_unknown_entities(self) -> None:
        document = self.base_document()
        document["tick"] = 1
        document["replay"] = [
            {"tick": 1, "moved": ["R-99"], "robots": {"R-01": [0, 0]}, "completed": []}
        ]
        self.reject(document)
        document = self.base_document()
        document["tick"] = 1
        document["replay"] = [
            {"tick": 1, "moved": [], "robots": {"R-01": [0, 0]}, "completed": ["T-99"]}
        ]
        self.reject(document)

    def test_last_frame_must_match_current_positions(self) -> None:
        document = self.base_document()
        document["tick"] = 1
        document["replay"] = [
            {"tick": 1, "moved": ["R-01"], "robots": {"R-01": [1, 0]}, "completed": []}
        ]
        self.reject(document)
        document = self.base_document()
        document["tick"] = 1
        document["replay"] = [
            {"tick": 1, "moved": ["R-01"], "robots": {}, "completed": []}
        ]
        self.reject(document)


class CheckpointIOFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "checkpoint.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_load_missing_file_raises_oserror(self) -> None:
        with self.assertRaises(OSError):
            FleetSimulator.load_checkpoint(os.path.join(self.tmp.name, "missing.json"))

    def test_load_unreadable_file_raises_oserror(self) -> None:
        simulator = FleetSimulator(GridMap(3, 3), [], [])
        simulator.save_checkpoint(self.path)
        os.chmod(self.path, 0o000)
        try:
            with self.assertRaises(OSError):
                FleetSimulator.load_checkpoint(self.path)
        finally:
            os.chmod(self.path, 0o600)

    def test_save_failure_preserves_existing_file(self) -> None:
        simulator = FleetSimulator(GridMap(4, 2), [Robot("R-01", (0, 0))], [Task("T-01", (1, 0), (3, 0))])
        simulator.save_checkpoint(self.path)
        with open(self.path, encoding="utf-8") as handle:
            before = handle.read()
        os.chmod(self.tmp.name, 0o500)
        try:
            with self.assertRaises(OSError):
                simulator.save_checkpoint(self.path)
        finally:
            os.chmod(self.tmp.name, 0o700)
        with open(self.path, encoding="utf-8") as handle:
            after = handle.read()
        self.assertEqual(before, after)
        self.assertEqual(
            [name for name in os.listdir(self.tmp.name) if name.startswith(".fleet-checkpoint-")],
            [],
        )

    def test_save_failure_leaves_no_target_when_new(self) -> None:
        simulator = FleetSimulator(GridMap(3, 3), [], [])
        target = os.path.join(self.tmp.name, "new.json")
        os.chmod(self.tmp.name, 0o500)
        try:
            with self.assertRaises(OSError):
                simulator.save_checkpoint(target)
        finally:
            os.chmod(self.tmp.name, 0o700)
        self.assertFalse(os.path.exists(target))

    def test_save_crash_preserves_existing_file(self) -> None:
        import warehouse_fleet.simulator as simulator_module

        simulator = FleetSimulator(GridMap(4, 2), [Robot("R-01", (0, 0))], [Task("T-01", (1, 0), (3, 0))])
        simulator.save_checkpoint(self.path)
        with open(self.path, encoding="utf-8") as handle:
            before = handle.read()
        original_replace = simulator_module.os.replace

        def crash(source: str, destination: str) -> None:
            raise RuntimeError("simulated interruption")

        simulator_module.os.replace = crash
        try:
            with self.assertRaises(RuntimeError):
                simulator.save_checkpoint(self.path)
        finally:
            simulator_module.os.replace = original_replace
        with open(self.path, encoding="utf-8") as handle:
            after = handle.read()
        self.assertEqual(before, after)
        self.assertEqual(
            [name for name in os.listdir(self.tmp.name) if name.startswith(".fleet-checkpoint-")],
            [],
        )


class CommandLineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "warehouse_fleet", *arguments],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

    def test_demo_output_unchanged_structure(self) -> None:
        result = self.run_cli("demo", "--steps", "12")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, "")
        snapshot = json.loads(result.stdout)
        self.assertEqual(
            set(snapshot), {"tick", "robots", "tasks", "metrics", "replay"}
        )
        self.assertEqual(snapshot["tick"], 12)

    def test_demo_with_checkpoint_saves_progress(self) -> None:
        path = os.path.join(self.tmp.name, "demo.json")
        result = self.run_cli("demo", "--steps", "6", "--checkpoint", path)
        self.assertEqual(result.returncode, 0)
        snapshot = json.loads(result.stdout)
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["tick"], 6)
        self.assertEqual(document["version"], 1)
        self.assertEqual(snapshot["tick"], 6)

    def test_resume_matches_uninterrupted_demo(self) -> None:
        checkpoint = os.path.join(self.tmp.name, "checkpoint.json")
        demo = self.run_cli("demo", "--steps", "12")
        self.assertEqual(demo.returncode, 0)
        saved = self.run_cli("demo", "--steps", "6", "--checkpoint", checkpoint)
        self.assertEqual(saved.returncode, 0)
        resumed = self.run_cli("resume", checkpoint, "--steps", "6")
        self.assertEqual(resumed.returncode, 0)
        self.assertEqual(resumed.stderr, "")
        self.assertEqual(json.loads(resumed.stdout), json.loads(demo.stdout))

    def test_resume_zero_steps_outputs_current_state(self) -> None:
        checkpoint = os.path.join(self.tmp.name, "checkpoint.json")
        self.run_cli("demo", "--steps", "6", "--checkpoint", checkpoint)
        result = self.run_cli("resume", checkpoint, "--steps", "0")
        self.assertEqual(result.returncode, 0)
        snapshot = json.loads(result.stdout)
        self.assertEqual(snapshot["tick"], 6)

    def test_resume_stops_early_when_all_complete(self) -> None:
        checkpoint = os.path.join(self.tmp.name, "checkpoint.json")
        self.run_cli("demo", "--steps", "6", "--checkpoint", checkpoint)
        result = self.run_cli("resume", checkpoint, "--steps", "100")
        self.assertEqual(result.returncode, 0)
        snapshot = json.loads(result.stdout)
        self.assertEqual(
            snapshot["metrics"]["tasks_completed"], snapshot["metrics"]["tasks_total"]
        )

    def test_resume_without_checkpoint_does_not_modify_source(self) -> None:
        checkpoint = os.path.join(self.tmp.name, "checkpoint.json")
        self.run_cli("demo", "--steps", "6", "--checkpoint", checkpoint)
        with open(checkpoint, encoding="utf-8") as handle:
            before = handle.read()
        result = self.run_cli("resume", checkpoint, "--steps", "3")
        self.assertEqual(result.returncode, 0)
        with open(checkpoint, encoding="utf-8") as handle:
            after = handle.read()
        self.assertEqual(before, after)

    def test_resume_with_same_path_checkpoint_updates_progress(self) -> None:
        checkpoint = os.path.join(self.tmp.name, "checkpoint.json")
        self.run_cli("demo", "--steps", "6", "--checkpoint", checkpoint)
        result = self.run_cli(
            "resume", checkpoint, "--steps", "6", "--checkpoint", checkpoint
        )
        self.assertEqual(result.returncode, 0)
        with open(checkpoint, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["tick"], 12)

    def test_resume_missing_file_fails_without_snapshot(self) -> None:
        result = self.run_cli("resume", os.path.join(self.tmp.name, "missing.json"), "--steps", "3")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("error", result.stderr)

    def test_resume_corrupt_file_fails_without_snapshot(self) -> None:
        checkpoint = os.path.join(self.tmp.name, "corrupt.json")
        with open(checkpoint, "w", encoding="utf-8") as handle:
            handle.write("garbage")
        result = self.run_cli("resume", checkpoint, "--steps", "3")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("error", result.stderr)

    def test_resume_invalid_checkpoint_fails_without_snapshot(self) -> None:
        checkpoint = os.path.join(self.tmp.name, "invalid.json")
        with open(checkpoint, "w", encoding="utf-8") as handle:
            json.dump({"version": 1}, handle)
        result = self.run_cli("resume", checkpoint, "--steps", "3")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("error", result.stderr)

    def test_demo_unwritable_checkpoint_fails_without_snapshot(self) -> None:
        result = self.run_cli(
            "demo", "--steps", "3", "--checkpoint",
            os.path.join(self.tmp.name, "no-dir", "checkpoint.json"),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("error", result.stderr)

    def test_negative_steps_rejected(self) -> None:
        result = self.run_cli("demo", "--steps", "-1")
        self.assertNotEqual(result.returncode, 0)
        result = self.run_cli("resume", os.path.join(self.tmp.name, "x.json"), "--steps", "-1")
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
