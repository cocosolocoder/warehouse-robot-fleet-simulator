"""Route/task consistency checks for restored checkpoints.

Traversability and adjacency checks alone accept a remaining route that drives
straight to the dropoff while bypassing the pickup point, leaving a task that
can never complete after resume. Loading must additionally confirm the saved
route can finish the robot's bound task given its current pickup state. These
tests pin that rule for both checkpoint versions, including empty routes and
the map-unreachability pause exception, and check that the command line
reports such a file without touching it.
"""

import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task
from warehouse_fleet.__main__ import main


def write_and_load(document):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return FleetSimulator.load_checkpoint(path)


def expect_rejected(testcase, document, fragment=None):
    with testcase.assertRaises(ValueError) as context:
        write_and_load(document)
    message = str(context.exception)
    if fragment is not None:
        testcase.assertIn(fragment, message)
    return message


class RouteDocumentBuilder:
    """Zero-tick checkpoint on an open 8x2 grid with one robot R and task T."""

    PICKUP = (2, 0)
    DROPOFF = (6, 0)

    def __init__(self):
        simulator = FleetSimulator(
            GridMap(8, 2),
            [Robot("R", (0, 0))],
            [Task("T", self.PICKUP, self.DROPOFF)],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            simulator.save_checkpoint(path)
            with open(path, encoding="utf-8") as handle:
                self.document = json.load(handle)

    def state(
        self,
        position,
        route,
        *,
        picked_up,
        pickup=None,
        dropoff=None,
        paused=False,
        completed=False,
        assigned="R",
    ):
        document = self.document
        document["robots"][0]["position"] = list(position)
        document["robots"][0]["route"] = [list(cell) for cell in route]
        document["robots"][0]["task_id"] = "T" if assigned is not None else None
        document["tasks"][0] = {
            "task_id": "T",
            "pickup": list(self.PICKUP if pickup is None else pickup),
            "dropoff": list(self.DROPOFF if dropoff is None else dropoff),
            "assigned_robot": assigned,
            "picked_up": picked_up,
            "completed": completed,
        }
        document["paused_tasks"] = ["T"] if paused else []
        return document

    @staticmethod
    def as_version_one(document):
        legacy = copy.deepcopy(document)
        legacy["version"] = 1
        for key in ("base_grid", "map_changes", "paused_tasks", "traffic_waits"):
            legacy.pop(key, None)
        legacy["replay"] = [
            {key: value for key, value in frame.items() if key != "type"}
            for frame in legacy["replay"]
        ]
        return legacy


class NonEmptyRouteConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.builder = RouteDocumentBuilder()

    def test_route_to_dropoff_bypassing_pickup_rejected(self):
        # Travels the bottom row, straight to the dropoff, never touching the
        # pickup cell on the top row.
        document = self.builder.state(
            (5, 1),
            [(6, 1), (7, 1), (7, 0), (6, 0)],
            picked_up=False,
            pickup=(1, 0),
        )
        message = expect_rejected(self, document, "never reaches the pickup")
        self.assertIn("'T'", message)
        self.assertIn("'R'", message)

    def test_route_ending_short_of_dropoff_rejected(self):
        # Passes the pickup but the remaining route stops somewhere else.
        document = self.builder.state(
            (0, 0), [(1, 0), (2, 0), (3, 0), (4, 0)], picked_up=False
        )
        expect_rejected(self, document, "instead of the dropoff")

    def test_passing_dropoff_early_then_pickup_then_dropoff_loads(self):
        route = [
            (6, 0), (7, 0), (7, 1), (6, 1), (5, 1), (4, 1), (3, 1), (2, 1),
            (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
        ]
        document = self.builder.state((5, 0), route, picked_up=False)
        loaded = write_and_load(document)
        self.assertEqual(loaded.robots["R"].route[-1], (6, 0))

    def test_loaded_robot_need_not_revisit_pickup(self):
        # Goods already aboard: a straight route to the dropoff that does not
        # pass the pickup is fine.
        document = self.builder.state(
            (3, 0), [(4, 0), (5, 0), (6, 0)], picked_up=True, pickup=(2, 0)
        )
        loaded = write_and_load(document)
        self.assertNotIn((2, 0), loaded.robots["R"].route)

    def test_loaded_robot_accepted_when_pickup_cell_since_closed(self):
        simulator = FleetSimulator(
            GridMap(8, 1), [Robot("R", (0, 0))], [Task("T", (2, 0), (6, 0))]
        )
        simulator.step()
        simulator.step()
        simulator.step()  # robot at (3,0), goods collected at (2,0)
        simulator.modify_obstacles(added=[(2, 0)])  # pickup closes behind it
        self.assertTrue(simulator.tasks["T"].picked_up)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            simulator.save_checkpoint(path)
            loaded = FleetSimulator.load_checkpoint(path)
        self.assertNotIn((2, 0), loaded.robots["R"].route)
        self.assertEqual(loaded.robots["R"].route[-1], (6, 0))

    def test_standing_on_pickup_with_route_to_dropoff_loads(self):
        # Already at the pickup cell but pickup not yet confirmed: the cell is
        # collected at the next step, so the route need not name the pickup.
        document = self.builder.state(
            (2, 0), [(3, 0), (4, 0), (5, 0), (6, 0)], picked_up=False
        )
        loaded = write_and_load(document)
        for _ in range(4):
            loaded.step()
        self.assertTrue(loaded.tasks["T"].completed)

    def test_detour_route_via_pickup_loads(self):
        # A legal avoidance detour that is not a shortest path: dip to the
        # bottom row around the pickup column and back.
        document = self.builder.state(
            (0, 0),
            [(1, 0), (1, 1), (2, 1), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)],
            picked_up=False,
        )
        loaded = write_and_load(document)
        self.assertEqual(loaded.robots["R"].route[-1], (6, 0))

    def test_unpicked_route_missing_pickup_rejected_in_version_one(self):
        document = self.builder.state(
            (5, 1),
            [(6, 1), (7, 1), (7, 0), (6, 0)],
            picked_up=False,
            pickup=(1, 0),
        )
        legacy = RouteDocumentBuilder.as_version_one(document)
        expect_rejected(self, legacy, "never reaches the pickup")

    def test_valid_route_loads_in_version_one(self):
        document = self.builder.state(
            (3, 0), [(4, 0), (5, 0), (6, 0)], picked_up=True, pickup=(2, 0)
        )
        legacy = RouteDocumentBuilder.as_version_one(document)
        loaded = write_and_load(legacy)
        self.assertEqual(loaded.robots["R"].route[-1], (6, 0))


class EmptyRouteConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.builder = RouteDocumentBuilder()

    def test_picked_up_at_dropoff_with_empty_route_loads_and_finishes(self):
        document = self.builder.state((6, 0), [], picked_up=True)
        loaded = write_and_load(document)
        self.assertFalse(loaded.tasks["T"].completed)
        loaded.step()
        self.assertTrue(loaded.tasks["T"].completed)

    def test_unpicked_at_shared_pickup_dropoff_with_empty_route_loads(self):
        document = self.builder.state(
            (4, 0), [], picked_up=False, pickup=(4, 0), dropoff=(4, 0)
        )
        loaded = write_and_load(document)
        loaded.step()
        self.assertTrue(loaded.tasks["T"].completed)

    def test_unpicked_at_pickup_distinct_from_dropoff_empty_route_rejected(self):
        document = self.builder.state((2, 0), [], picked_up=False)
        expect_rejected(self, document, "empty route")

    def test_unpicked_elsewhere_empty_route_rejected(self):
        document = self.builder.state((3, 0), [], picked_up=False)
        expect_rejected(self, document, "empty route")

    def test_picked_up_but_not_at_dropoff_empty_route_rejected(self):
        document = self.builder.state((4, 0), [], picked_up=True)
        expect_rejected(self, document, "instead of the dropoff")

    def test_paused_task_keeps_empty_route_and_binding(self):
        simulator = FleetSimulator(
            GridMap(8, 1), [Robot("R", (0, 0))], [Task("T", (2, 0), (6, 0))]
        )
        simulator.step()  # at (1,0), goods collected
        simulator.modify_obstacles(added=[(4, 0)])  # dropoff leg unreachable
        self.assertEqual(simulator.paused_tasks, {"T"})
        self.assertEqual(simulator.robots["R"].route, [])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            simulator.save_checkpoint(path)
            loaded = FleetSimulator.load_checkpoint(path)
        self.assertEqual(loaded.paused_tasks, {"T"})
        self.assertEqual(loaded.robots["R"].task_id, "T")
        self.assertEqual(loaded.robots["R"].route, [])
        self.assertFalse(loaded.tasks["T"].completed)

    def test_unpaused_empty_route_is_not_mistaken_for_a_pause(self):
        # Same shape as a pause record but the task is absent from
        # paused_tasks: it must be rejected, not silently treated as paused.
        simulator = FleetSimulator(
            GridMap(8, 1), [Robot("R", (0, 0))], [Task("T", (2, 0), (6, 0))]
        )
        simulator.step()
        simulator.modify_obstacles(added=[(4, 0)])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            simulator.save_checkpoint(path)
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
        document["paused_tasks"] = []
        with self.assertRaises(ValueError):
            write_and_load(document)


class ExemptStatesAndNoSideEffectsTests(unittest.TestCase):
    def setUp(self):
        self.builder = RouteDocumentBuilder()

    def test_unassigned_task_is_exempt(self):
        document = self.builder.state(
            (0, 0), [], picked_up=False, assigned=None
        )
        document["tasks"][0]["assigned_robot"] = None
        loaded = write_and_load(document)
        self.assertIsNone(loaded.robots["R"].task_id)
        self.assertIsNone(loaded.tasks["T"].assigned_robot)

    def test_completed_task_is_exempt(self):
        document = self.builder.state(
            (6, 0), [], picked_up=True, completed=True, assigned=None
        )
        # Completed tasks keep their historical robot binding in the task.
        document["tasks"][0]["assigned_robot"] = "R"
        loaded = write_and_load(document)
        self.assertTrue(loaded.tasks["T"].completed)

    def test_loading_does_not_confirm_pickup_or_delivery(self):
        document = self.builder.state((6, 0), [], picked_up=True)
        loaded = write_and_load(document)
        self.assertFalse(loaded.tasks["T"].completed)
        self.assertTrue(loaded.tasks["T"].picked_up)
        self.assertEqual(loaded.tick, 0)
        self.assertEqual(loaded.robots["R"].distance_travelled, 0)

    def test_loading_keeps_pickup_flag_for_uncollected_goods(self):
        route = [(3, 0), (4, 0), (5, 0), (6, 0)]
        document = self.builder.state((2, 0), route, picked_up=False)
        loaded = write_and_load(document)
        self.assertFalse(loaded.tasks["T"].picked_up)
        self.assertEqual(loaded.robots["R"].route, route)


class ResumeCliRejectionTests(unittest.TestCase):
    def _invalid_path(self):
        builder = RouteDocumentBuilder()
        # Direct-to-dropoff route that skips the pickup point.
        document = builder.state(
            (5, 1),
            [(6, 1), (7, 1), (7, 0), (6, 0)],
            picked_up=False,
            pickup=(1, 0),
        )
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "bad.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return path

    def test_resume_reports_error_with_nonzero_exit_and_no_snapshot(self):
        path = self._invalid_path()
        with open(path, encoding="utf-8") as handle:
            original = handle.read()
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            main(["resume", path, "--steps", "5"])
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("never reaches the pickup", err.getvalue())
        self.assertEqual(out.getvalue(), "")
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), original)

    def test_resume_does_not_write_checkpoint_output_on_failure(self):
        path = self._invalid_path()
        target = os.path.join(os.path.dirname(path), "out.json")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                main(["resume", path, "--steps", "5", "--checkpoint", target])
        self.assertEqual(cm.exception.code, 1)
        self.assertFalse(os.path.exists(target))
        self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
