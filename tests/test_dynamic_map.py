import copy
import json
import os
import tempfile
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


def corridor_sim(length: int = 6) -> FleetSimulator:
    # One row, one robot at x=0, pickup x=1, dropoff x=length-1.
    return FleetSimulator(
        GridMap(length, 1),
        [Robot("R", (0, 0))],
        [Task("T", (1, 0), (length - 1, 0))],
    )


def save_load(simulator: FleetSimulator) -> FleetSimulator:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        simulator.save_checkpoint(path)
        return FleetSimulator.load_checkpoint(path)


class ModifyValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = FleetSimulator(
            GridMap(4, 2, frozenset({(3, 0)})),
            [Robot("R", (0, 0))],
            [Task("T", (1, 0), (2, 0))],
        )

    def assert_state_unchanged(self) -> None:
        self.assertEqual(self.sim.tick, 0)
        self.assertEqual(self.sim.robots["R"].position, (0, 0))
        self.assertEqual(self.sim.robots["R"].distance_travelled, 0)
        self.assertEqual(self.sim.grid.obstacles, frozenset({(3, 0)}))
        self.assertEqual(self.sim.map_change_history(), [])
        self.assertEqual(
            [frame for frame in self.sim.replay if frame.get("type") == "map_change"],
            [],
        )
        self.assertEqual(self.sim.paused_tasks, set())

    def test_rejects_out_of_bounds_coordinates(self) -> None:
        for added, removed in (([(4, 0)], []), ([(-1, 0)], []), ([(0, 2)], []),
                               ([], [(9, 9)])):
            with self.subTest(added=added, removed=removed):
                with self.assertRaises(ValueError):
                    self.sim.modify_obstacles(added=added, removed=removed)
                self.assert_state_unchanged()

    def test_rejects_non_integer_and_boolean_coordinates(self) -> None:
        for bad in ([(True, 0)], [("1", 0)], [(1, None)], [[1]], [[1, 2, 3]],
                    "not-a-list", None, 42, [(1.0, 0)]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.sim.modify_obstacles(added=bad)
                with self.assertRaises(ValueError):
                    self.sim.modify_obstacles(removed=bad)
                self.assert_state_unchanged()

    def test_rejects_same_cell_added_and_removed(self) -> None:
        with self.assertRaises(ValueError):
            self.sim.modify_obstacles(added=[(1, 0)], removed=[(1, 0)])
        self.assert_state_unchanged()

    def test_rejects_obstacle_on_robot_cell_even_when_duplicated(self) -> None:
        with self.assertRaises(ValueError):
            self.sim.modify_obstacles(added=[(0, 0), (0, 0)])
        self.assert_state_unchanged()

    def test_batch_validation_is_all_or_nothing(self) -> None:
        # Valid cell followed by an invalid one: nothing changes.
        with self.assertRaises(ValueError):
            self.sim.modify_obstacles(added=[(1, 0), (99, 0)])
        self.assert_state_unchanged()
        # Two valid adds plus one add/remove conflict: nothing changes.
        with self.assertRaises(ValueError):
            self.sim.modify_obstacles(added=[(1, 0), (2, 0)], removed=[(2, 0)])
        self.assert_state_unchanged()

    def test_accepts_tuples_and_lists(self) -> None:
        record = self.sim.modify_obstacles(added=[(1, 0)], removed=((3, 0),))
        self.assertEqual(record["added"], [[1, 0]])
        self.assertEqual(record["removed"], [[3, 0]])


class ModifySemanticsTests(unittest.TestCase):
    def test_edit_does_not_advance_clock_or_move_anything(self) -> None:
        sim = corridor_sim()
        sim.step()
        before = copy.deepcopy(sim.snapshot())
        sim.modify_obstacles(added=[])
        after = sim.snapshot()
        self.assertEqual(before["tick"], after["tick"])
        self.assertEqual(before["robots"], after["robots"])
        # A real edit likewise.
        sim.modify_obstacles(added=[(3, 0)])
        self.assertEqual(sim.tick, before["tick"])
        self.assertEqual(sim.robots["R"].position, (1, 0))
        self.assertEqual(sim.robots["R"].distance_travelled, 1)

    def test_duplicate_coordinates_processed_once(self) -> None:
        sim = corridor_sim()
        record = sim.modify_obstacles(
            added=[(2, 0), (2, 0)], removed=[(4, 0), (4, 0)]
        )
        self.assertEqual(record["added"], [[2, 0]])
        self.assertEqual(record["removed"], [])
        self.assertEqual(len(sim.map_change_history()), 1)

    def test_adding_existing_and_removing_free_is_noop_without_record(self) -> None:
        sim = FleetSimulator(
            GridMap(3, 1, frozenset({(2, 0)})), [Robot("R", (0, 0))], []
        )
        before_frames = len(sim.replay)
        record = sim.modify_obstacles(added=[(2, 0)], removed=[(0, 0)])
        self.assertEqual(record["added"], [])
        self.assertEqual(record["removed"], [])
        self.assertEqual(sim.map_change_history(), [])
        self.assertEqual(len(sim.replay), before_frames)
        self.assertEqual(sim.grid.obstacles, frozenset({(2, 0)}))

    def test_partial_effect_batch_records_only_actual_cells(self) -> None:
        sim = FleetSimulator(
            GridMap(4, 1, frozenset({(3, 0)})), [Robot("R", (0, 0))], []
        )
        record = sim.modify_obstacles(added=[(3, 0), (1, 0)], removed=[(2, 0)])
        self.assertEqual(record["added"], [[1, 0]])
        self.assertEqual(record["removed"], [])

    def test_map_dimensions_never_change(self) -> None:
        sim = corridor_sim()
        sim.modify_obstacles(added=[(2, 0), (3, 0)])
        self.assertEqual((sim.grid.width, sim.grid.height), (6, 1))
        loaded = save_load(sim)
        self.assertEqual((loaded.grid.width, loaded.grid.height), (6, 1))


class RerouteAndPauseTests(unittest.TestCase):
    def test_unreachable_after_close_clears_route_and_pauses(self) -> None:
        sim = corridor_sim(6)
        sim.step()
        sim.step()  # robot at (2,0), picked up; dropoff (5,0)
        sim.modify_obstacles(added=[(3, 0)])
        robot = sim.robots["R"]
        task = sim.tasks["T"]
        self.assertTrue(task.picked_up)
        self.assertEqual(robot.route, [])
        self.assertEqual(robot.task_id, "T")
        self.assertEqual(sim.status()["paused_tasks"], ["T"])
        self.assertFalse(task.completed)
        # While paused, ticks accumulate but the robot never moves.
        position = robot.position
        for _ in range(3):
            sim.step()
        self.assertEqual(robot.position, position)
        self.assertEqual(robot.distance_travelled, 2)
        self.assertEqual(sim.paused_tasks, {"T"})

    def test_reopen_resumes_on_next_step_and_completes(self) -> None:
        sim = corridor_sim(6)
        sim.step()
        sim.step()
        sim.modify_obstacles(added=[(3, 0)])
        sim.step()
        sim.modify_obstacles(removed=[(3, 0)])
        # The edit does not itself move the robot; the next step does.
        self.assertEqual(sim.robots["R"].position, (2, 0))
        self.assertEqual(sim.robots["R"].route, [(3, 0), (4, 0), (5, 0)])
        for _ in range(3):
            sim.step()
        self.assertTrue(sim.tasks["T"].completed)
        self.assertEqual(sim.paused_tasks, set())

    def test_picked_up_robot_routes_straight_to_dropoff(self) -> None:
        sim = corridor_sim(6)
        sim.step()
        sim.step()  # at (2,0), picked up at (1,0)
        sim.modify_obstacles(added=[(1, 0)])  # close the old pickup cell
        route = sim.robots["R"].route
        self.assertNotIn((1, 0), route)
        self.assertEqual(route[0], (3, 0))
        for _ in range(3):
            sim.step()
        self.assertTrue(sim.tasks["T"].completed)
        self.assertEqual(sim.robots["R"].distance_travelled, 5)

    def test_unpicked_robot_replans_via_pickup_and_confirms_on_step(self) -> None:
        # 3x2 grid: robot en route to pickup (2,1); closing (2,0) forces a
        # one-cell detour through (1,1) while the goods are not yet collected.
        sim = FleetSimulator(
            GridMap(3, 2),
            [Robot("R", (0, 0))],
            [Task("T", (2, 1), (0, 1))],
        )
        sim.step()  # robot moves to (1,0), still unpicked
        self.assertEqual(sim.robots["R"].position, (1, 0))
        self.assertFalse(sim.tasks["T"].picked_up)
        sim.modify_obstacles(added=[(2, 0)])
        route = sim.robots["R"].route
        # Replanned path visits the pickup before the dropoff and avoids (2,0).
        self.assertNotIn((2, 0), route)
        self.assertEqual(route.index((2, 1)) < route.index((0, 1)), True)
        for _ in range(5):
            sim.step()
            if sim.tasks["T"].completed:
                break
        self.assertTrue(sim.tasks["T"].completed)
        self.assertTrue(sim.tasks["T"].picked_up)

    def test_robot_already_at_pickup_confirms_before_routing_to_dropoff(self) -> None:
        # At step start the robot is on the pickup cell; close pickup after the
        # step, goods must already be on board and no return needed.
        sim = corridor_sim(5)
        sim.step()  # at (1,0): step start logic confirms pickup
        self.assertTrue(sim.tasks["T"].picked_up)
        sim.step()  # move to (2,0)
        sim.modify_obstacles(added=[(1, 0)])
        self.assertTrue(sim.tasks["T"].picked_up)
        self.assertNotIn((1, 0), sim.robots["R"].route)

    def test_task_keeps_robot_while_paused(self) -> None:
        sim = corridor_sim(6)
        sim.step()
        sim.step()
        sim.modify_obstacles(added=[(3, 0)])
        self.assertEqual(sim.tasks["T"].assigned_robot, "R")
        self.assertEqual(sim.robots["R"].task_id, "T")
        loaded = save_load(sim)
        self.assertEqual(loaded.tasks["T"].assigned_robot, "R")
        self.assertEqual(loaded.robots["R"].task_id, "T")
        self.assertEqual(loaded.status()["paused_tasks"], ["T"])

    def test_unassigned_unreachable_task_keeps_waiting(self) -> None:
        # Single robot working T1; seal T2's pickup cell away from everyone.
        sim = FleetSimulator(
            GridMap(4, 2),
            [Robot("R", (0, 0))],
            [Task("T1", (1, 0), (3, 0)), Task("T2", (0, 1), (3, 1))],
        )
        sim.step()  # R at (1,0) on T1
        sim.modify_obstacles(added=[(0, 1)])
        self.assertIsNone(sim.tasks["T2"].assigned_robot)
        self.assertEqual(sim.paused_tasks, set())
        for _ in range(6):
            sim.step()
        self.assertTrue(sim.tasks["T1"].completed)
        self.assertIsNone(sim.tasks["T2"].assigned_robot)
        # Reopen: T2 assigns automatically on the next step and later finishes.
        sim.modify_obstacles(removed=[(0, 1)])
        for _ in range(12):
            sim.step()
            if sim.tasks["T2"].completed:
                break
        self.assertTrue(sim.tasks["T2"].completed)

    def test_pause_is_distinct_from_waiting_for_right_of_way(self) -> None:
        # A robot blocked only by traffic keeps its route and is never paused,
        # whereas closing the cell ahead does produce a paused task.
        sim = FleetSimulator(
            GridMap(6, 1),
            [Robot("A", (0, 0)), Robot("B", (1, 0))],
            [Task("TA", (2, 0), (5, 0)), Task("TB", (3, 0), (4, 0))],
        )
        sim.step()  # B moves on; A may be stuck behind B for a tick
        self.assertEqual(sim.status()["paused_tasks"], [])
        active = [robot for robot in sim.robots.values() if robot.task_id is not None]
        self.assertTrue(active)
        for robot in active:
            self.assertTrue(robot.route)  # congestion never clears the route
        # Once a real obstacle blocks TA's only path, it pauses.
        sim.modify_obstacles(added=[(4, 0)])
        self.assertIn("TA", sim.status()["paused_tasks"])

    def test_other_robots_keep_moving_while_one_task_paused(self) -> None:
        sim = FleetSimulator(
            GridMap(5, 2),
            [Robot("A", (0, 0)), Robot("B", (0, 1))],
            [Task("TA", (1, 0), (0, 0)), Task("TB", (1, 1), (4, 1))],
        )
        sim.step()  # both at their pickups at tick 1
        # Seal the right side off; A's task stays entirely on the left.
        sim.modify_obstacles(added=[(2, 0), (2, 1)])
        self.assertEqual(sim.paused_tasks, {"TB"})
        self.assertEqual(sim.robots["B"].route, [])
        for _ in range(3):
            sim.step()
        self.assertTrue(sim.tasks["TA"].completed)
        self.assertFalse(sim.tasks["TB"].completed)
        # B never moved after the pause.
        self.assertEqual(sim.robots["B"].position, (1, 1))
        self.assertEqual(sim.robots["B"].distance_travelled, 1)
        # Reopening lets B finish.
        sim.modify_obstacles(removed=[(2, 0), (2, 1)])
        for _ in range(8):
            sim.step()
            if sim.tasks["TB"].completed:
                break
        self.assertTrue(sim.tasks["TB"].completed)


class ReplayMapChangeTests(unittest.TestCase):
    def test_map_change_frames_recorded_with_effective_tick(self) -> None:
        sim = corridor_sim(6)
        sim.step()
        sim.modify_obstacles(added=[(4, 0)])
        sim.step()
        sim.step()
        sim.modify_obstacles(removed=[(4, 0)])
        events = [frame for frame in sim.replay if frame.get("type") == "map_change"]
        self.assertEqual(
            [(e["tick"], e["sequence"], e["added"], e["removed"]) for e in events],
            [
                (1, 1, [[4, 0]], []),
                (3, 2, [], [[4, 0]]),
            ],
        )

    def test_consecutive_edits_same_tick_keep_order(self) -> None:
        sim = FleetSimulator(GridMap(5, 1), [Robot("R", (0, 0))], [])
        sim.modify_obstacles(added=[(1, 0)])
        sim.modify_obstacles(added=[(2, 0)], removed=[(1, 0)])
        history = sim.map_change_history()
        self.assertEqual([h["sequence"] for h in history], [1, 2])
        self.assertEqual([h["tick"] for h in history], [0, 0])
        self.assertEqual(history[1]["removed"], [[1, 0]])
        self.assertEqual(sim.grid.obstacles, frozenset({(2, 0)}))

    def test_tick_frames_are_not_overwritten_by_later_maps(self) -> None:
        sim = corridor_sim(6)
        sim.step()
        snapshot_frame = copy.deepcopy(sim.replay[-1])
        sim.modify_obstacles(added=[(4, 0)])
        sim.modify_obstacles(removed=[(4, 0)])
        for _ in range(3):
            sim.step()
        tick_frames = [f for f in sim.replay if f.get("type", "tick") == "tick"]
        self.assertEqual(tick_frames[0], snapshot_frame)
        # Every tick frame still carries the original payload keys.
        for frame in tick_frames:
            self.assertEqual(
                sorted(frame), ["completed", "moved", "robots", "tick", "type"]
            )

    def test_status_reports_current_obstacles_and_paused_ids(self) -> None:
        sim = FleetSimulator(
            GridMap(3, 1, frozenset({(0, 0)})), [Robot("R", (1, 0))], []
        )
        sim.modify_obstacles(added=[(2, 0)], removed=[(0, 0)])
        status = sim.status()
        self.assertEqual(status["obstacles"], [[2, 0]])
        self.assertEqual(status["paused_tasks"], [])
        self.assertEqual((status["width"], status["height"]), (3, 1))

    def test_step_event_shape_includes_type(self) -> None:
        sim = corridor_sim(3)
        event = sim.step()
        self.assertEqual(event["type"], "tick")
        self.assertIn("moved", event)
        self.assertIn("robots", event)
        self.assertIn("completed", event)


class DeterminismTests(unittest.TestCase):
    def _scripted_run(self) -> FleetSimulator:
        sim = corridor_sim(8)
        sim.step()
        sim.step()
        sim.modify_obstacles(added=[(4, 0)])
        sim.step()
        sim.modify_obstacles(added=[(6, 0)])
        sim.step()
        sim.modify_obstacles(removed=[(4, 0)])
        for _ in range(8):
            sim.step()
            if sim.tasks["T"].completed:
                break
        return sim

    def test_identical_operation_sequences_are_identical(self) -> None:
        first = self._scripted_run()
        second = self._scripted_run()
        self.assertEqual(first.snapshot(), second.snapshot())
        self.assertEqual(first.map_change_history(), second.map_change_history())
        self.assertEqual(first.metrics(), second.metrics())

    def test_resume_after_every_edit_matches_uninterrupted(self) -> None:
        full = self._scripted_run()
        interrupted = corridor_sim(8)
        interrupted.step()
        interrupted.step()
        interrupted.modify_obstacles(added=[(4, 0)])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            interrupted.save_checkpoint(path)
            resumed = FleetSimulator.load_checkpoint(path)
        resumed.step()
        resumed.modify_obstacles(added=[(6, 0)])
        resumed.step()
        resumed.modify_obstacles(removed=[(4, 0)])
        for _ in range(8):
            resumed.step()
            if resumed.tasks["T"].completed:
                break
        self.assertEqual(resumed.snapshot(), full.snapshot())
        self.assertEqual(resumed.map_change_history(), full.map_change_history())
        self.assertEqual(resumed.metrics(), full.metrics())

    def test_two_instances_from_one_file_are_independent(self) -> None:
        sim = corridor_sim(6)
        sim.step()
        sim.step()
        sim.modify_obstacles(added=[(3, 0)])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            sim.save_checkpoint(path)
            first = FleetSimulator.load_checkpoint(path)
            second = FleetSimulator.load_checkpoint(path)
        first.modify_obstacles(removed=[(3, 0)])
        first.step()
        self.assertIn((3, 0), second.grid.obstacles)
        self.assertEqual(second.paused_tasks, {"T"})
        self.assertEqual(second.tick, 2)
        self.assertIsNot(first.replay, second.replay)


class DynamicCheckpointFormatTests(unittest.TestCase):
    def setUp(self) -> None:
        sim = corridor_sim(6)
        sim.step()
        sim.modify_obstacles(added=[(4, 0)])
        sim.step()
        sim.step()
        sim.modify_obstacles(added=[(3, 0)])  # pauses T
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "state.json")
        sim.save_checkpoint(self.path)
        with open(self.path, encoding="utf-8") as handle:
            self.document = json.load(handle)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def reload_expect_error(self, document: object, fragment: str) -> None:
        with self.assertRaises(ValueError) as context:
            FleetSimulator._from_checkpoint(document)
        self.assertIn(fragment, str(context.exception))

    def mutated(self, mutate) -> object:
        document = copy.deepcopy(self.document)
        mutate(document)
        return document

    def test_version_two_document_shape(self) -> None:
        self.assertEqual(self.document["version"], 2)
        self.assertIn("base_grid", self.document)
        self.assertIn("map_changes", self.document)
        self.assertIn("paused_tasks", self.document)
        self.assertEqual(self.document["paused_tasks"], ["T"])
        self.assertEqual(self.document["map_changes"][0]["tick"], 1)

    def test_round_trip_restores_map_pause_and_history(self) -> None:
        loaded = FleetSimulator.load_checkpoint(self.path)
        self.assertEqual(loaded.grid.obstacles, frozenset({(3, 0), (4, 0)}))
        self.assertEqual(loaded.base_grid.obstacles, frozenset())
        self.assertEqual(loaded.paused_tasks, {"T"})
        self.assertEqual(len(loaded.map_change_history()), 2)
        self.assertEqual(loaded.tick, 3)
        self.assertEqual(loaded.robots["R"].route, [])
        loaded.modify_obstacles(removed=[(3, 0), (4, 0)])
        for _ in range(6):
            loaded.step()
            if loaded.tasks["T"].completed:
                break
        self.assertTrue(loaded.tasks["T"].completed)

    def test_version_one_file_loads_without_history(self) -> None:
        # Build a plain v1 document with no edit-related keys.
        basic = FleetSimulator(
            GridMap(4, 1),
            [Robot("R", (0, 0))],
            [Task("T", (1, 0), (3, 0))],
        )
        basic.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v1.json")
            basic.save_checkpoint(path)
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
        document["version"] = 1
        for frame in document["replay"]:
            frame.pop("type", None)
        document.pop("base_grid", None)
        document.pop("map_changes", None)
        document.pop("paused_tasks", None)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v1.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(document, handle)
            loaded = FleetSimulator.load_checkpoint(path)
        self.assertEqual(loaded.map_change_history(), [])
        self.assertEqual(loaded.paused_tasks, set())
        self.assertEqual(loaded.grid.obstacles, frozenset())
        loaded.step()
        loaded.step()
        self.assertTrue(loaded.tasks["T"].completed)

    def test_out_of_bounds_history_coordinate_rejected(self) -> None:
        document = self.mutated(
            lambda d: d["map_changes"][0]["added"].__setitem__(0, [99, 0])
        )
        self.reload_expect_error(document, "outside the map")

    def test_add_remove_conflict_in_history_rejected(self) -> None:
        document = self.mutated(
            lambda d: d["map_changes"][0].__setitem__("removed", [[4, 0]])
        )
        self.reload_expect_error(document, "adds and removes")

    def test_history_not_reproducing_saved_grid_rejected(self) -> None:
        document = self.mutated(lambda d: d["grid"]["obstacles"].append([2, 0]))
        self.reload_expect_error(document, "does not match the saved grid")

    def test_base_grid_mismatch_rejected(self) -> None:
        document = self.mutated(lambda d: d["base_grid"].__setitem__("width", 9))
        self.reload_expect_error(document, "dimensions must match")

    def test_replay_map_frames_must_match_history(self) -> None:
        document = self.mutated(lambda d: d["replay"][1]["added"].append([2, 0]))
        self.reload_expect_error(document, "do not match")

    def test_history_sequence_gap_rejected(self) -> None:
        document = self.mutated(
            lambda d: d["map_changes"][1].__setitem__("sequence", 5)
        )
        self.reload_expect_error(document, "consecutively")

    def test_paused_unknown_task_rejected(self) -> None:
        document = self.mutated(lambda d: d["paused_tasks"].append("NOPE"))
        self.reload_expect_error(document, "not among the checkpoint tasks")

    def test_paused_completed_task_rejected(self) -> None:
        document = self.mutated(
            lambda d: (
                d["paused_tasks"].__setitem__(0, "T"),
                d["tasks"][0].__setitem__("completed", True),
            )
        )
        self.reload_expect_error(document, "already completed")

    def test_paused_but_reachable_rejected(self) -> None:
        # A dedicated document: the history is erased entirely while the task
        # stays paused on an otherwise open map, which is inconsistent.
        basic = corridor_sim(6)
        basic.step()  # robot at (1,0), picked up
        basic.modify_obstacles(added=[(4, 0)])  # pauses T
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            basic.save_checkpoint(path)
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
        document["map_changes"] = []
        document["replay"] = [
            frame for frame in document["replay"] if frame.get("type") == "tick"
        ]
        document["grid"]["obstacles"] = []
        document["base_grid"]["obstacles"] = []
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(document, handle)
            with self.assertRaises(ValueError) as context:
                FleetSimulator.load_checkpoint(path)
            self.assertIn("reachable", str(context.exception))

    def test_duplicate_paused_entries_rejected(self) -> None:
        document = self.mutated(lambda d: d["paused_tasks"].append("T"))
        self.reload_expect_error(document, "duplicates")

    def test_v2_requires_new_fields(self) -> None:
        for key in ("base_grid", "map_changes", "paused_tasks"):
            document = self.mutated(lambda d, k=key: d.pop(k))
            self.reload_expect_error(document, key)

    def test_map_change_out_of_order_rejected(self) -> None:
        # Move the second history change before the first while replay stays
        # otherwise consistent: history tick ordering must be rejected.
        document = self.mutated(
            lambda d: (
                d["map_changes"][1].__setitem__("tick", 0),
                d["replay"][4].__setitem__("tick", 0),
            )
        )
        self.reload_expect_error(document, "stamped tick 0")


if __name__ == "__main__":
    unittest.main()
