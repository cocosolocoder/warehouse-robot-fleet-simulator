import copy
import json
import os
import tempfile
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


# 5 wide, 2 tall: the whole top row is drivable, the bottom row only at (2, 1).
POCKET_OBSTACLES = frozenset((x, 1) for x in (0, 1, 3, 4))


def pocket_grid() -> GridMap:
    return GridMap(5, 2, POCKET_OBSTACLES)


def swap_scenario(
    left_id: str = "R-1", right_id: str = "R-2", reverse_input: bool = False
) -> FleetSimulator:
    """Two loaded robots on the top row delivering to each other's start."""
    robots = [
        Robot(left_id, (0, 0), route=[(1, 0), (2, 0), (3, 0), (4, 0)], task_id="T-1"),
        Robot(right_id, (4, 0), route=[(3, 0), (2, 0), (1, 0), (0, 0)], task_id="T-2"),
    ]
    tasks = [
        Task("T-1", (0, 0), (4, 0), assigned_robot=left_id, picked_up=True),
        Task("T-2", (4, 0), (0, 0), assigned_robot=right_id, picked_up=True),
    ]
    if reverse_input:
        robots.reverse()
        tasks.reverse()
    return FleetSimulator(pocket_grid(), robots, tasks)


def run_until_done(simulator: FleetSimulator, limit: int) -> int:
    """Step until every task completes; return the number of ticks used."""
    for used in range(1, limit + 1):
        simulator.step()
        if all(task.completed for task in simulator.tasks.values()):
            return used
    return -1


def assert_no_collisions(testcase: unittest.TestCase, simulator: FleetSimulator) -> None:
    for frame in simulator.replay:
        if frame.get("type") != "tick":
            continue
        positions = [tuple(cell) for cell in frame["robots"].values()]
        testcase.assertEqual(len(positions), len(set(positions)))
        for position in positions:
            testcase.assertTrue(simulator.grid.traversable(position))


class HeadOnSwapTests(unittest.TestCase):
    def test_swap_completes_within_twenty_ticks(self) -> None:
        sim = swap_scenario()
        used = run_until_done(sim, 20)
        self.assertNotEqual(used, -1)
        self.assertLessEqual(used, 20)
        self.assertEqual(sim.robots["R-1"].position, (4, 0))
        self.assertEqual(sim.robots["R-2"].position, (0, 0))
        assert_no_collisions(self, sim)

    def test_swap_is_robust_to_ids_and_input_order(self) -> None:
        for left_id, right_id, reverse in (
            ("R-2", "R-1", False),
            ("R-1", "R-2", True),
            ("R-2", "R-1", True),
        ):
            with self.subTest(left=left_id, right=right_id, reverse=reverse):
                sim = swap_scenario(left_id, right_id, reverse)
                used = run_until_done(sim, 20)
                self.assertNotEqual(used, -1)
                self.assertLessEqual(used, 20)
                assert_no_collisions(self, sim)

    def test_side_pocket_is_actually_used(self) -> None:
        sim = swap_scenario()
        run_until_done(sim, 20)
        visited = set()
        for frame in sim.replay:
            if frame.get("type") == "tick":
                visited.update(tuple(cell) for cell in frame["robots"].values())
        self.assertIn((2, 1), visited)

    def test_yield_moves_count_as_mileage_waits_do_not(self) -> None:
        sim = swap_scenario()
        used = run_until_done(sim, 20)
        self.assertNotEqual(used, -1)
        # The robot taking the side pocket drives two extra cells (in and out
        # of the pocket) compared to the straight four-cell corridor; the
        # other drives exactly four. Waiting ticks add nothing.
        distances = sorted(
            robot.distance_travelled for robot in sim.robots.values()
        )
        self.assertEqual(distances, [4, 6])
        moved_ticks = sum(
            1
            for frame in sim.replay
            if frame.get("type") == "tick" and frame["moved"]
        )
        self.assertLessEqual(moved_ticks, used)


class IdleYieldTests(unittest.TestCase):
    def build(self) -> FleetSimulator:
        return FleetSimulator(
            pocket_grid(),
            [Robot("R-1", (0, 0)), Robot("R-2", (2, 0))],
            [Task("T-1", (1, 0), (4, 0))],
        )

    def test_idle_robot_yields_and_delivery_meets_deadline(self) -> None:
        sim = self.build()
        used = run_until_done(sim, 20)
        self.assertNotEqual(used, -1)
        self.assertLessEqual(used, 20)
        self.assertTrue(sim.tasks["T-1"].completed)
        assert_no_collisions(self, sim)

    def test_idle_yield_creates_no_task_and_robot_stays_put(self) -> None:
        sim = self.build()
        run_until_done(sim, 20)
        blocker = sim.robots["R-2"]
        # The idle robot stepped aside into the pocket and was never hired.
        self.assertEqual(blocker.position, (2, 1))
        self.assertIsNone(blocker.task_id)
        self.assertEqual(blocker.distance_travelled, 1)
        self.assertEqual(sim.tasks["T-1"].assigned_robot, "R-1")
        # Once out of the way it does not wander back.
        position = blocker.position
        for _ in range(3):
            sim.step()
        self.assertEqual(blocker.position, position)
        self.assertEqual(blocker.distance_travelled, 1)

    def test_idle_robot_can_take_tasks_after_yielding(self) -> None:
        sim = self.build()
        run_until_done(sim, 20)
        sim.tasks["T-2"] = Task("T-2", (2, 1), (0, 0))
        sim.step()
        self.assertEqual(sim.tasks["T-2"].assigned_robot, "R-2")
        used = run_until_done(sim, 20)
        self.assertNotEqual(used, -1)
        self.assertTrue(sim.tasks["T-2"].completed)


class NoSafePassingTests(unittest.TestCase):
    def build(self) -> FleetSimulator:
        # Pure 5x1 corridor: no side cell exists at all.
        return FleetSimulator(
            GridMap(5, 1),
            [
                Robot("R-1", (0, 0), route=[(1, 0), (2, 0), (3, 0), (4, 0)], task_id="T-1"),
                Robot("R-2", (4, 0), route=[(3, 0), (2, 0), (1, 0), (0, 0)], task_id="T-2"),
            ],
            [
                Task("T-1", (0, 0), (4, 0), assigned_robot="R-1", picked_up=True),
                Task("T-2", (4, 0), (0, 0), assigned_robot="R-2", picked_up=True),
            ],
        )

    def test_step_advances_without_collision_or_false_completion(self) -> None:
        sim = self.build()
        for _ in range(10):
            event = sim.step()
            self.assertEqual(event["type"], "tick")
        self.assertEqual(sim.tick, 10)
        # Both tasks are kept, cargo stays on board, nothing completes.
        for task_id, robot_id in (("T-1", "R-1"), ("T-2", "R-2")):
            task = sim.tasks[task_id]
            self.assertFalse(task.completed)
            self.assertTrue(task.picked_up)
            self.assertEqual(task.assigned_robot, robot_id)
            self.assertEqual(sim.robots[robot_id].task_id, task_id)
        assert_no_collisions(self, sim)
        self.assertEqual(sim.status()["paused_tasks"], [])

    def test_traffic_waits_accumulate_consecutively(self) -> None:
        sim = self.build()
        sim.step()  # both move
        sim.step()  # R-1 moves to (2,0); R-2 is blocked and waits
        waits = {entry["robot_id"]: entry for entry in sim.status()["traffic_waits"]}
        self.assertEqual(set(waits), {"R-2"})
        self.assertEqual(waits["R-2"]["blocked_by"], ["R-1"])
        self.assertEqual(waits["R-2"]["ticks"], 1)
        sim.step()  # now both are stuck head-on
        waits = {entry["robot_id"]: entry for entry in sim.status()["traffic_waits"]}
        self.assertEqual(set(waits), {"R-1", "R-2"})
        self.assertEqual(waits["R-1"]["ticks"], 1)
        self.assertEqual(waits["R-2"]["ticks"], 2)
        self.assertEqual(waits["R-1"]["blocked_by"], ["R-2"])
        sim.step()
        waits = {entry["robot_id"]: entry for entry in sim.status()["traffic_waits"]}
        self.assertEqual(waits["R-1"]["ticks"], 2)
        self.assertEqual(waits["R-2"]["ticks"], 3)


class FutileRetreatTests(unittest.TestCase):
    """A blocked robot must not retreat and return to the same blockage."""

    def build(self, width: int = 3) -> FleetSimulator:
        # Pure one-row corridor: A loaded one cell before its dropoff, an idle
        # robot parked exactly on the dropoff, with only empty cells behind A.
        return FleetSimulator(
            GridMap(width, 1),
            [
                Robot("A", (width - 2, 0), route=[(width - 1, 0)], task_id="T-1"),
                Robot("B", (width - 1, 0)),
            ],
            [
                Task(
                    "T-1",
                    (width - 2, 0),
                    (width - 1, 0),
                    assigned_robot="A",
                    picked_up=True,
                )
            ],
        )

    def test_both_robots_wait_in_place_without_retreat(self) -> None:
        sim = self.build()
        for _ in range(5):
            event = sim.step()
            self.assertEqual(event["moved"], [])
        self.assertEqual(sim.tick, 5)
        self.assertEqual(sim.robots["A"].position, (1, 0))
        self.assertEqual(sim.robots["B"].position, (2, 0))

    def test_empty_cells_behind_do_not_justify_ping_pong(self) -> None:
        # More room behind A changes nothing: there is still no way around B.
        sim = self.build(width=6)
        for _ in range(5):
            event = sim.step()
            self.assertEqual(event["moved"], [])
        self.assertEqual(sim.robots["A"].position, (4, 0))
        self.assertEqual(sim.robots["B"].position, (5, 0))

    def test_waiting_accrues_no_mileage_and_keeps_replay_positions(self) -> None:
        sim = self.build()
        sim.step()
        sim.step()
        self.assertEqual(sim.robots["A"].distance_travelled, 0)
        self.assertEqual(sim.robots["B"].distance_travelled, 0)
        self.assertEqual(sim.metrics()["distance_total"], 0)
        for frame in sim.replay:
            if frame.get("type") == "tick":
                self.assertEqual(frame["moved"], [])
                self.assertEqual(frame["robots"], {"A": [1, 0], "B": [2, 0]})

    def test_task_stays_picked_up_uncompleted_and_owned_by_a(self) -> None:
        sim = self.build()
        for _ in range(4):
            sim.step()
        task = sim.tasks["T-1"]
        self.assertTrue(task.picked_up)
        self.assertFalse(task.completed)
        self.assertEqual(task.assigned_robot, "A")
        self.assertEqual(sim.robots["A"].task_id, "T-1")
        self.assertIsNone(sim.robots["B"].task_id)
        # A traffic jam is not a map-unreachability pause.
        self.assertEqual(sim.status()["paused_tasks"], [])
        self.assertEqual(sim.metrics()["tasks_paused"], [])

    def test_traffic_wait_reports_grow_and_match_in_status_and_metrics(self) -> None:
        sim = self.build()
        sim.step()
        self.assertEqual(
            sim.status()["traffic_waits"],
            [{"robot_id": "A", "blocked_by": ["B"], "ticks": 1}],
        )
        for expected in (2, 3, 4):
            sim.step()
            self.assertEqual(
                sim.status()["traffic_waits"],
                [{"robot_id": "A", "blocked_by": ["B"], "ticks": expected}],
            )
        self.assertEqual(sim.metrics()["traffic_waits"], sim.status()["traffic_waits"])

    def test_resuming_once_blocker_clears_resets_the_wait_counter(self) -> None:
        # The side pocket at (2, 1) is closed at first: B cannot yield, A waits.
        grid = GridMap(3, 2, frozenset({(0, 1), (1, 1), (2, 1)}))
        sim = FleetSimulator(
            grid,
            [
                Robot("A", (0, 0), route=[(1, 0), (2, 0)], task_id="T-1"),
                Robot("B", (2, 0)),
            ],
            [Task("T-1", (0, 0), (2, 0), assigned_robot="A", picked_up=True)],
        )
        sim.step()  # A -> (1,0)
        sim.step()  # blocked, pocket closed
        sim.step()  # still blocked
        self.assertEqual(
            sim.status()["traffic_waits"],
            [{"robot_id": "A", "blocked_by": ["B"], "ticks": 2}],
        )
        sim.modify_obstacles(removed=[(2, 1)])
        sim.step()  # B yields into the pocket; A takes (2,0) and completes
        self.assertEqual(sim.status()["traffic_waits"], [])
        self.assertTrue(sim.tasks["T-1"].completed)
        self.assertEqual(sim.robots["A"].position, (2, 0))
        self.assertEqual(sim.robots["B"].position, (2, 1))
        # The waiting ticks themselves never counted as mileage.
        self.assertEqual(sim.robots["A"].distance_travelled, 2)
        self.assertEqual(sim.robots["B"].distance_travelled, 1)


class TrafficWaitResetTests(unittest.TestCase):
    def test_wait_counter_resets_once_the_robot_moves(self) -> None:
        sim = swap_scenario()
        sim.step()
        sim.step()  # R-2 waits behind R-1 for one tick
        waits = {entry["robot_id"]: entry for entry in sim.status()["traffic_waits"]}
        self.assertEqual(set(waits), {"R-2"})
        self.assertEqual(waits["R-2"]["ticks"], 1)
        sim.step()  # R-1 sidesteps into the pocket, R-2 drives on
        waits = {entry["robot_id"]: entry for entry in sim.status()["traffic_waits"]}
        self.assertNotIn("R-2", waits)

    def test_metrics_and_status_both_report_waits(self) -> None:
        sim = FleetSimulator(
            GridMap(5, 1),
            [
                Robot("R-1", (0, 0), route=[(1, 0), (2, 0), (3, 0), (4, 0)], task_id="T-1"),
                Robot("R-2", (4, 0), route=[(3, 0), (2, 0), (1, 0), (0, 0)], task_id="T-2"),
            ],
            [
                Task("T-1", (0, 0), (4, 0), assigned_robot="R-1", picked_up=True),
                Task("T-2", (4, 0), (0, 0), assigned_robot="R-2", picked_up=True),
            ],
        )
        sim.step()
        sim.step()
        self.assertEqual(
            sim.metrics()["traffic_waits"], sim.status()["traffic_waits"]
        )
        self.assertEqual(sim.paused_tasks, set())


class PausedRobotsDoNotYieldTests(unittest.TestCase):
    def test_paused_robot_stays_put_and_never_yields(self) -> None:
        # 5x1 corridor; R-2's dropoff gets sealed off so it pauses, and R-1
        # ends up queued behind it.
        sim = FleetSimulator(
            GridMap(5, 1),
            [Robot("R-1", (0, 0)), Robot("R-2", (2, 0))],
            [Task("T-1", (1, 0), (4, 0)), Task("T-2", (3, 0), (4, 0))],
        )
        sim.step()  # R-1 -> (1,0) picks up; R-2 -> (3,0) picks up
        sim.modify_obstacles(added=[(4, 0)])
        self.assertEqual(sim.paused_tasks, {"T-1", "T-2"})
        position = sim.robots["R-2"].position
        for _ in range(4):
            sim.step()
        self.assertEqual(sim.robots["R-2"].position, position)
        self.assertEqual(sim.robots["R-2"].distance_travelled, 1)
        # The paused robot is not reported as traffic-waiting either.
        waits = {entry["robot_id"] for entry in sim.status()["traffic_waits"]}
        self.assertNotIn("R-2", waits)


class SideCellCloseReopenTests(unittest.TestCase):
    def test_closing_side_cell_blocks_yield_reopen_resumes(self) -> None:
        sim = swap_scenario()
        sim.step()
        sim.step()  # R-1 at (2,0), R-2 waiting at (3,0)
        sim.modify_obstacles(added=[(2, 1)])
        self.assertEqual(sim.paused_tasks, set())  # corridor itself still open
        positions = {rid: sim.robots[rid].position for rid in ("R-1", "R-2")}
        for _ in range(3):
            sim.step()
        # No side cell: everyone waits in place, tasks and cargo intact.
        self.assertEqual(
            {rid: sim.robots[rid].position for rid in ("R-1", "R-2")}, positions
        )
        self.assertFalse(any(task.completed for task in sim.tasks.values()))
        waits = sim.status()["traffic_waits"]
        self.assertTrue(waits)
        assert_no_collisions(self, sim)
        # Reopening lets the next ticks retry and finish well within budget.
        sim.modify_obstacles(removed=[(2, 1)])
        used = run_until_done(sim, 20)
        self.assertNotEqual(used, -1)
        self.assertTrue(all(task.completed for task in sim.tasks.values()))

    def test_stale_yield_route_is_not_reused_after_reopen(self) -> None:
        sim = swap_scenario()
        sim.step()
        sim.step()  # R-1 at (2,0), R-2 waiting behind at (3,0)
        sim.modify_obstacles(added=[(2, 1)])  # close the only side cell
        for _ in range(2):
            sim.step()
        # Nobody is holding a route through the closed cell.
        for robot in sim.robots.values():
            self.assertNotIn((2, 1), robot.route)
        sim.modify_obstacles(removed=[(2, 1)])
        # From the next tick on the side cell is tried again, fresh.
        used = run_until_done(sim, 20)
        self.assertNotEqual(used, -1)
        visited = set()
        for frame in sim.replay:
            if frame.get("type") == "tick":
                visited.update(tuple(cell) for cell in frame["robots"].values())
        self.assertIn((2, 1), visited)


class TrafficWaitCheckpointTests(unittest.TestCase):
    def build_waiting_sim(self) -> FleetSimulator:
        sim = NoSafePassingTests().build()
        for _ in range(3):
            sim.step()
        return sim

    def test_round_trip_preserves_waits_and_continuation(self) -> None:
        sim = self.build_waiting_sim()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            sim.save_checkpoint(path)
            loaded = FleetSimulator.load_checkpoint(path)
        self.assertEqual(loaded.status(), sim.status())
        for _ in range(4):
            sim.step()
            loaded.step()
        self.assertEqual(loaded.snapshot(), sim.snapshot())
        self.assertEqual(loaded.status(), sim.status())

    def test_resume_mid_yield_matches_uninterrupted(self) -> None:
        full = swap_scenario()
        run_until_done(full, 20)
        interrupted = swap_scenario()
        for _ in range(3):
            interrupted.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            interrupted.save_checkpoint(path)
            resumed = FleetSimulator.load_checkpoint(path)
        run_until_done(resumed, 20)
        self.assertEqual(resumed.snapshot(), full.snapshot())
        self.assertEqual(resumed.metrics(), full.metrics())

    def test_saved_document_carries_traffic_waits(self) -> None:
        sim = self.build_waiting_sim()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            sim.save_checkpoint(path)
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
        self.assertEqual(
            document["traffic_waits"],
            [
                {"robot_id": "R-1", "blocked_by": ["R-2"], "ticks": 1},
                {"robot_id": "R-2", "blocked_by": ["R-1"], "ticks": 2},
            ],
        )

    def test_version_one_and_two_files_default_to_zero_waits(self) -> None:
        sim = self.build_waiting_sim()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            sim.save_checkpoint(path)
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
        # A version 2 file without the new field.
        v2 = copy.deepcopy(document)
        v2.pop("traffic_waits")
        # A version 1 file: no edit-related keys, untyped tick frames.
        v1 = copy.deepcopy(document)
        v1["version"] = 1
        for key in ("base_grid", "map_changes", "paused_tasks", "traffic_waits"):
            v1.pop(key, None)
        v1["replay"] = [
            {k: v for k, v in frame.items() if k != "type"}
            for frame in v1["replay"]
            if frame.get("type", "tick") == "tick"
        ]
        for doc in (v2, v1):
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "state.json")
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(doc, handle)
                loaded = FleetSimulator.load_checkpoint(path)
            self.assertEqual(loaded.status()["traffic_waits"], [])
            self.assertEqual(loaded.metrics()["traffic_waits"], [])

    def test_invalid_traffic_waits_are_rejected(self) -> None:
        sim = self.build_waiting_sim()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            sim.save_checkpoint(path)
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)

        def expect_error(mutate, fragment: str) -> None:
            doc = copy.deepcopy(document)
            mutate(doc)
            with self.assertRaises(ValueError) as context:
                FleetSimulator._from_checkpoint(doc)
            self.assertIn(fragment, str(context.exception))

        expect_error(lambda d: d.update(traffic_waits={}), "must be a list")
        expect_error(
            lambda d: d["traffic_waits"].append(
                {"robot_id": "NOPE", "blocked_by": ["R-1"], "ticks": 1}
            ),
            "unknown robot",
        )
        expect_error(
            lambda d: d["traffic_waits"].append(
                {"robot_id": "R-1", "blocked_by": ["R-2"], "ticks": 1}
            ),
            "duplicate",
        )
        expect_error(
            lambda d: d["traffic_waits"][0].update(ticks=0), "positive integer"
        )
        expect_error(
            lambda d: d["traffic_waits"][0].update(blocked_by=[]), "non-empty"
        )
        expect_error(
            lambda d: d["traffic_waits"][0].update(blocked_by=["R-1"]), "itself"
        )
        expect_error(
            lambda d: d["traffic_waits"][0].update(blocked_by=["NOPE"]),
            "unknown blocking robot",
        )


class YieldDeterminismTests(unittest.TestCase):
    def test_input_list_order_does_not_change_results(self) -> None:
        first = swap_scenario()
        second = swap_scenario(reverse_input=True)
        run_until_done(first, 20)
        run_until_done(second, 20)
        self.assertEqual(first.snapshot(), second.snapshot())
        self.assertEqual(first.metrics(), second.metrics())

    def test_identical_runs_are_identical(self) -> None:
        first = swap_scenario()
        second = swap_scenario()
        run_until_done(first, 20)
        run_until_done(second, 20)
        self.assertEqual(first.snapshot(), second.snapshot())


if __name__ == "__main__":
    unittest.main()
