"""Checkpoint validation that a remaining route can finish its bound task.

Traversability and adjacency checks do not prove a route completes the pickup
and delivery the robot was assigned: a route straight to the dropoff bypasses
the pickup, and an empty route can strand a robot mid-task. Loading must
reject those documents for both checkpoint versions while accepting detours,
early dropoff visits, the standing-on-pickup state and map-paused tasks.
"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task
from warehouse_fleet.__main__ import main as cli_main


def checkpoint_document(
    *,
    position=(0, 0),
    route=(),
    pickup=(2, 0),
    dropoff=(5, 0),
    picked_up=False,
    completed=False,
    robot_bound=True,
    assigned="R",
    obstacles=(),
    paused=(),
    width=8,
    height=2,
):
    """Build the JSON dict for a single-robot, single-task checkpoint."""
    grid = GridMap(width, height, frozenset(obstacles))
    robots = [Robot("R", position, list(route), "T" if robot_bound else None)]
    tasks = [Task("T", pickup, dropoff, assigned, picked_up, completed)]
    simulator = FleetSimulator(grid, robots, tasks)
    simulator.paused_tasks = set(paused)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        simulator.save_checkpoint(path)
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)


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


class BypassRouteRejectionTests(unittest.TestCase):
    def test_route_to_dropoff_bypassing_pickup_rejected(self):
        # Second-row detour reaches the dropoff without ever touching the
        # pickup cell (2, 0); every step is adjacent and traversable.
        route = [(0, 1), (1, 1), (2, 1), (3, 1), (3, 0), (4, 0), (5, 0)]
        document = checkpoint_document(position=(0, 0), route=route)
        expect_rejected(self, document, "T", "R", "pickup point")

    def test_route_ending_short_of_dropoff_rejected(self):
        # Visits the pickup but the last remaining waypoint is not the
        # dropoff, so following the route cannot finish the task.
        route = [(1, 0), (2, 0), (3, 0), (4, 0)]
        document = checkpoint_document(position=(0, 0), route=route)
        expect_rejected(self, document, "T", "R", "dropoff")

    def test_version_one_bypass_route_rejected(self):
        route = [(0, 1), (1, 1), (2, 1), (3, 1), (3, 0), (4, 0), (5, 0)]
        document = as_version_one(
            checkpoint_document(position=(0, 0), route=route)
        )
        expect_rejected(self, document, "T", "R", "pickup point")

    def test_loading_does_not_repair_or_complete_anything(self):
        route = [(1, 0), (2, 0), (3, 0), (4, 0)]
        document = checkpoint_document(position=(0, 0), route=route)
        with self.assertRaises(ValueError):
            load(document)
        # The rejected document itself is untouched: no route was appended,
        # no pickup flag flipped, no completion recorded.
        self.assertEqual(document["robots"][0]["route"][-1], [4, 0])
        self.assertFalse(document["tasks"][0]["picked_up"])
        self.assertFalse(document["tasks"][0]["completed"])


class LegalRouteAcceptanceTests(unittest.TestCase):
    def test_detour_through_pickup_to_dropoff_loads(self):
        # Longer than the planner's straight-line shortest route, but it still
        # visits the pickup and ends at the dropoff: avoidance detours are
        # legal and need not match any replanned shortest path.
        route = [(0, 1), (1, 1), (2, 1), (2, 0), (3, 0), (4, 0), (5, 0)]
        loaded = load(checkpoint_document(position=(0, 0), route=route))
        self.assertEqual(loaded.robots["R"].route[-1], (5, 0))
        for _ in range(8):
            loaded.step()
            if loaded.tasks["T"].picked_up:
                break
        self.assertTrue(loaded.tasks["T"].picked_up)

    def test_early_dropoff_visit_then_pickup_and_return_loads(self):
        # The robot is already standing on the dropoff; its route leaves,
        # collects at the pickup and comes back. Passing the dropoff early
        # must not be mistaken for completion.
        route = [(4, 0), (3, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
        loaded = load(
            checkpoint_document(
                position=(5, 0), route=route, pickup=(2, 0), dropoff=(5, 0)
            )
        )
        self.assertFalse(loaded.tasks["T"].completed)
        for _ in range(8):
            loaded.step()
            if loaded.tasks["T"].completed:
                break
        self.assertTrue(loaded.tasks["T"].completed)

    def test_standing_on_pickup_without_confirmation_is_legal(self):
        # Already at the pickup, goods not yet confirmed: resume collects on
        # the next step, so the route only has to lead to the dropoff.
        route = [(3, 0), (4, 0), (5, 0)]
        loaded = load(
            checkpoint_document(
                position=(2, 0), route=route, pickup=(2, 0), dropoff=(5, 0)
            )
        )
        self.assertFalse(loaded.tasks["T"].picked_up)
        loaded.step()
        self.assertTrue(loaded.tasks["T"].picked_up)
        self.assertFalse(loaded.tasks["T"].completed)

    def test_loaded_robot_route_ignores_a_reclosed_pickup(self):
        # Goods already collected; the pickup cell has since been closed and
        # the detour never enters it. Only the dropoff ending matters.
        route = [(0, 1), (1, 1), (2, 1), (3, 1), (3, 0), (4, 0)]
        document = checkpoint_document(
            position=(0, 0),
            route=route,
            pickup=(2, 0),
            dropoff=(4, 0),
            picked_up=True,
            obstacles={(2, 0)},
            width=5,
        )
        loaded = load(document)
        for _ in range(8):
            loaded.step()
            if loaded.tasks["T"].completed:
                break
        self.assertTrue(loaded.tasks["T"].completed)


class EmptyRouteRuleTests(unittest.TestCase):
    def test_empty_route_loaded_robot_at_dropoff_loads_but_does_not_complete(self):
        document = checkpoint_document(
            position=(5, 0),
            route=(),
            pickup=(2, 0),
            dropoff=(5, 0),
            picked_up=True,
        )
        loaded = load(document)
        # Loading confirms neither delivery nor anything else.
        self.assertFalse(loaded.tasks["T"].completed)
        loaded.step()
        self.assertTrue(loaded.tasks["T"].completed)

    def test_empty_route_coincident_points_load(self):
        document = checkpoint_document(
            position=(2, 0),
            route=(),
            pickup=(2, 0),
            dropoff=(2, 0),
        )
        loaded = load(document)
        self.assertFalse(loaded.tasks["T"].picked_up)
        loaded.step()
        self.assertTrue(loaded.tasks["T"].completed)

    def test_empty_route_waiting_at_pickup_with_different_dropoff_rejected(self):
        document = checkpoint_document(
            position=(2, 0), route=(), pickup=(2, 0), dropoff=(5, 0)
        )
        expect_rejected(self, document, "T", "R", "empty remaining route")

    def test_empty_route_loaded_robot_away_from_dropoff_rejected(self):
        document = checkpoint_document(
            position=(3, 0),
            route=(),
            pickup=(2, 0),
            dropoff=(5, 0),
            picked_up=True,
        )
        expect_rejected(self, document, "T", "R", "dropoff")

    def test_empty_route_before_pickup_elsewhere_rejected(self):
        document = checkpoint_document(
            position=(1, 0), route=(), pickup=(2, 0), dropoff=(5, 0)
        )
        expect_rejected(self, document, "T", "R", "empty remaining route")

    def test_paused_task_with_empty_route_keeps_its_binding(self):
        # Obstacle (2, 0) severs the 1-wide corridor to the dropoff; the task
        # is genuinely unreachable, paused, and the empty route is allowed.
        document = checkpoint_document(
            position=(0, 0),
            route=(),
            pickup=(1, 0),
            dropoff=(3, 0),
            obstacles={(2, 0)},
            paused={"T"},
            width=4,
            height=1,
        )
        loaded = load(document)
        self.assertEqual(loaded.paused_tasks, {"T"})
        self.assertEqual(loaded.robots["R"].task_id, "T")
        self.assertFalse(loaded.tasks["T"].completed)


class InactiveTaskExemptionTests(unittest.TestCase):
    def test_unassigned_task_is_not_route_checked(self):
        document = checkpoint_document(
            position=(0, 0), route=(), robot_bound=False, assigned=None
        )
        loaded = load(document)
        self.assertIsNone(loaded.tasks["T"].assigned_robot)
        self.assertTrue(loaded.robots["R"].idle)

    def test_completed_task_is_not_route_checked(self):
        # The completed task keeps its historical owner; the robot is now idle
        # with no route, which must not look like a stranded delivery.
        document = checkpoint_document(
            position=(5, 0),
            route=(),
            pickup=(2, 0),
            dropoff=(5, 0),
            picked_up=True,
            completed=True,
            robot_bound=False,
            assigned="R",
        )
        loaded = load(document)
        self.assertTrue(loaded.tasks["T"].completed)
        self.assertTrue(loaded.robots["R"].idle)


class ResumeCliRejectionTests(unittest.TestCase):
    def test_resume_reports_error_and_leaves_source_untouched(self):
        route = [(0, 1), (1, 1), (2, 1), (3, 1), (3, 0), (4, 0), (5, 0)]
        document = checkpoint_document(position=(0, 0), route=route)
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
                    cli_main(["resume", path, "--steps", "5", "--checkpoint", output])
            self.assertEqual(context.exception.code, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("pickup point", stderr.getvalue())
            self.assertIn("resume:", stderr.getvalue())
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), original)
            self.assertFalse(os.path.exists(output))


if __name__ == "__main__":
    unittest.main()
