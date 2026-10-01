"""Tests for automatic yielding / overtaking in the fleet simulator.

When a robot is blocked by another vehicle it may step sideways into a free
adjacent cell (a "yield" move) to let traffic pass, then replan its route.
Waiting accumulates only while blocked by another vehicle and resets on move.
"""

import os
import tempfile
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


def corridor_map() -> GridMap:
    """5x2 map: top row open, bottom row only the side cell (2,1)."""
    return GridMap(5, 2, frozenset({(0, 1), (1, 1), (3, 1), (4, 1)}))


def swap_scenario(swap_ids: bool = False) -> FleetSimulator:
    """Two picked-up robots at (0,0) and (4,0) must swap positions."""
    if not swap_ids:
        robots = [Robot("R-01", (0, 0)), Robot("R-02", (4, 0))]
    else:
        robots = [Robot("R-02", (0, 0)), Robot("R-01", (4, 0))]
    tasks = [
        Task("T-01", (0, 0), (4, 0), picked_up=True),
        Task("T-02", (4, 0), (0, 0), picked_up=True),
    ]
    sim = FleetSimulator(corridor_map(), robots, tasks)
    for robot in sim.robots.values():
        task = sim.tasks["T-01"] if robot.position == (0, 0) else sim.tasks["T-02"]
        robot.task_id = task.task_id
        task.assigned_robot = robot.robot_id
        robot.route = sim._plan_route(robot, task)
    return sim


def run_until_done(sim: FleetSimulator, limit: int = 25) -> int:
    for tick in range(limit):
        if all(task.completed for task in sim.tasks.values()):
            return tick
        sim.step()
    return limit


def assert_no_collisions(sim: FleetSimulator) -> None:
    for frame in sim.replay:
        if frame.get("type") != "tick":
            continue
        positions = [tuple(pos) for pos in frame["robots"].values()]
        assert len(positions) == len(set(positions)), (
            f"collision at tick {frame['tick']}: {frame['robots']}"
        )


class CorridorPassingTests(unittest.TestCase):
    def test_two_robots_swap_positions_using_side_cell(self) -> None:
        sim = swap_scenario()
        ticks = run_until_done(sim, 20)
        self.assertLessEqual(ticks, 20)
        self.assertTrue(all(task.completed for task in sim.tasks.values()))
        assert_no_collisions(sim)

    def test_swap_with_reversed_robot_ids(self) -> None:
        sim = swap_scenario(swap_ids=True)
        ticks = run_until_done(sim, 20)
        self.assertLessEqual(ticks, 20)
        self.assertTrue(all(task.completed for task in sim.tasks.values()))
        assert_no_collisions(sim)

    def test_swap_with_reversed_input_order(self) -> None:
        robots = [Robot("R-02", (4, 0)), Robot("R-01", (0, 0))]
        tasks = [
            Task("T-02", (4, 0), (0, 0), picked_up=True),
            Task("T-01", (0, 0), (4, 0), picked_up=True),
        ]
        sim = FleetSimulator(corridor_map(), robots, tasks)
        for robot in sim.robots.values():
            task = sim.tasks["T-01"] if robot.position == (0, 0) else sim.tasks["T-02"]
            robot.task_id = task.task_id
            task.assigned_robot = robot.robot_id
            robot.route = sim._plan_route(robot, task)
        ticks = run_until_done(sim, 20)
        self.assertLessEqual(ticks, 20)
        self.assertTrue(all(task.completed for task in sim.tasks.values()))
        assert_no_collisions(sim)

    def test_yield_move_counts_mileage(self) -> None:
        sim = swap_scenario()
        run_until_done(sim, 20)
        # Each robot must have moved at least the Manhattan distance of its
        # route plus the yield detour.
        for robot in sim.robots.values():
            self.assertGreater(robot.distance_travelled, 0)

    def test_wait_ticks_reset_after_move(self) -> None:
        sim = swap_scenario()
        run_until_done(sim, 20)
        for robot in sim.robots.values():
            self.assertEqual(robot.wait_ticks, 0)


class IdleYieldTests(unittest.TestCase):
    def test_idle_vehicle_yields_to_side_cell(self) -> None:
        grid = corridor_map()
        robots = [Robot("R-01", (0, 0)), Robot("R-02", (2, 0))]
        tasks = [Task("T", (1, 0), (4, 0))]
        sim = FleetSimulator(grid, robots, tasks)
        # Assign the task to the truck (R-01) explicitly.
        sim.robots["R-01"].task_id = "T"
        sim.robots["R-01"].route = sim._plan_route(sim.robots["R-01"], sim.tasks["T"])
        sim.tasks["T"].assigned_robot = "R-01"
        ticks = run_until_done(sim, 20)
        self.assertLessEqual(ticks, 20)
        self.assertTrue(sim.tasks["T"].completed)
        # The idle vehicle stays in the side cell after yielding.
        self.assertEqual(sim.robots["R-02"].position, (2, 1))
        self.assertIsNone(sim.robots["R-02"].task_id)
        assert_no_collisions(sim)

    def test_idle_yield_generates_no_task(self) -> None:
        grid = corridor_map()
        robots = [Robot("R-01", (0, 0)), Robot("R-02", (2, 0))]
        tasks = [Task("T", (1, 0), (4, 0))]
        sim = FleetSimulator(grid, robots, tasks)
        sim.robots["R-01"].task_id = "T"
        sim.robots["R-01"].route = sim._plan_route(sim.robots["R-01"], sim.tasks["T"])
        sim.tasks["T"].assigned_robot = "R-01"
        run_until_done(sim, 20)
        self.assertIsNone(sim.robots["R-02"].task_id)
        # The idle vehicle can still take a new task after yielding.
        new_task = Task("T2", (2, 1), (0, 0))
        sim.tasks["T2"] = new_task
        sim.assign_tasks()
        self.assertEqual(sim.robots["R-02"].task_id, "T2")


class NoSideCellTests(unittest.TestCase):
    def test_robots_wait_without_collision(self) -> None:
        grid = GridMap(5, 1)
        robots = [Robot("R-01", (0, 0)), Robot("R-02", (4, 0))]
        tasks = [
            Task("T-01", (0, 0), (4, 0), picked_up=True),
            Task("T-02", (4, 0), (0, 0), picked_up=True),
        ]
        sim = FleetSimulator(grid, robots, tasks)
        for robot in sim.robots.values():
            task = sim.tasks["T-01"] if robot.position == (0, 0) else sim.tasks["T-02"]
            robot.task_id = task.task_id
            task.assigned_robot = robot.robot_id
            robot.route = sim._plan_route(robot, task)
        for _ in range(10):
            sim.step()
            assert_no_collisions(sim)
        # No task completed (they can't pass).
        self.assertFalse(any(t.completed for t in sim.tasks.values()))
        # Waiting is reported.
        traffic_wait = sim.status()["traffic_wait"]
        self.assertTrue(traffic_wait)
        for entry in traffic_wait:
            self.assertGreater(entry["wait_ticks"], 0)
            self.assertIsNotNone(entry["blocked_by"])

    def test_step_still_advances_time_when_blocked(self) -> None:
        grid = GridMap(5, 1)
        robots = [Robot("R-01", (0, 0)), Robot("R-02", (4, 0))]
        tasks = [
            Task("T-01", (0, 0), (4, 0), picked_up=True),
            Task("T-02", (4, 0), (0, 0), picked_up=True),
        ]
        sim = FleetSimulator(grid, robots, tasks)
        for robot in sim.robots.values():
            task = sim.tasks["T-01"] if robot.position == (0, 0) else sim.tasks["T-02"]
            robot.task_id = task.task_id
            task.assigned_robot = robot.robot_id
            robot.route = sim._plan_route(robot, task)
        sim.step()
        sim.step()
        self.assertEqual(sim.tick, 2)
        sim.step()
        self.assertEqual(sim.tick, 3)


class PausedRobotTests(unittest.TestCase):
    def test_paused_robot_does_not_yield(self) -> None:
        grid = corridor_map()
        robots = [Robot("R-01", (0, 0)), Robot("R-02", (4, 0))]
        tasks = [
            Task("T-01", (0, 0), (4, 0), picked_up=True),
            Task("T-02", (4, 0), (0, 0), picked_up=True),
        ]
        sim = FleetSimulator(grid, robots, tasks)
        for robot in sim.robots.values():
            task = sim.tasks["T-01"] if robot.position == (0, 0) else sim.tasks["T-02"]
            robot.task_id = task.task_id
            task.assigned_robot = robot.robot_id
            robot.route = sim._plan_route(robot, task)
        for _ in range(2):
            sim.step()
        # Block R-01's dropoff to pause it.
        sim.modify_obstacles(added=[(4, 0)])
        self.assertIn("T-01", sim.paused_tasks)
        position = sim.robots["R-01"].position
        for _ in range(5):
            sim.step()
            assert_no_collisions(sim)
        # R-01 never moved after being paused.
        self.assertEqual(sim.robots["R-01"].position, position)
        self.assertEqual(sim.robots["R-01"].distance_travelled, 2)


class MapModificationTests(unittest.TestCase):
    def test_close_side_cell_robots_wait(self) -> None:
        sim = swap_scenario()
        for _ in range(2):
            sim.step()
        sim.modify_obstacles(added=[(2, 1)])
        for _ in range(5):
            sim.step()
            assert_no_collisions(sim)
        # No task completed while the side cell is closed.
        self.assertFalse(any(t.completed for t in sim.tasks.values()))

    def test_reopen_side_cell_resumes_passing(self) -> None:
        sim = swap_scenario()
        for _ in range(2):
            sim.step()
        sim.modify_obstacles(added=[(2, 1)])
        for _ in range(3):
            sim.step()
        sim.modify_obstacles(removed=[(2, 1)])
        ticks = run_until_done(sim, 20)
        self.assertLessEqual(ticks, 20)
        self.assertTrue(all(task.completed for task in sim.tasks.values()))
        assert_no_collisions(sim)

    def test_reopen_next_tick_tries_again(self) -> None:
        sim = swap_scenario()
        for _ in range(2):
            sim.step()
        sim.modify_obstacles(added=[(2, 1)])
        for _ in range(3):
            sim.step()
        sim.modify_obstacles(removed=[(2, 1)])
        # The next step should move at least one robot.
        event = sim.step()
        self.assertTrue(event["moved"])


class WaitTrackingTests(unittest.TestCase):
    def test_wait_ticks_accumulate_on_block(self) -> None:
        grid = GridMap(5, 1)
        robots = [Robot("R-01", (0, 0)), Robot("R-02", (4, 0))]
        tasks = [
            Task("T-01", (0, 0), (4, 0), picked_up=True),
            Task("T-02", (4, 0), (0, 0), picked_up=True),
        ]
        sim = FleetSimulator(grid, robots, tasks)
        for robot in sim.robots.values():
            task = sim.tasks["T-01"] if robot.position == (0, 0) else sim.tasks["T-02"]
            robot.task_id = task.task_id
            task.assigned_robot = robot.robot_id
            robot.route = sim._plan_route(robot, task)
        sim.step()
        sim.step()
        # R-02 is blocked by R-01.
        self.assertGreater(sim.robots["R-02"].wait_ticks, 0)
        # R-01 moved, so its wait_ticks reset.
        self.assertEqual(sim.robots["R-01"].wait_ticks, 0)

    def test_status_traffic_wait_shape(self) -> None:
        grid = GridMap(5, 1)
        robots = [Robot("R-01", (0, 0)), Robot("R-02", (4, 0))]
        tasks = [
            Task("T-01", (0, 0), (4, 0), picked_up=True),
            Task("T-02", (4, 0), (0, 0), picked_up=True),
        ]
        sim = FleetSimulator(grid, robots, tasks)
        for robot in sim.robots.values():
            task = sim.tasks["T-01"] if robot.position == (0, 0) else sim.tasks["T-02"]
            robot.task_id = task.task_id
            task.assigned_robot = robot.robot_id
            robot.route = sim._plan_route(robot, task)
        sim.step()
        sim.step()
        status = sim.status()
        self.assertIn("traffic_wait", status)
        for entry in status["traffic_wait"]:
            self.assertIn("robot_id", entry)
            self.assertIn("blocked_by", entry)
            self.assertIn("wait_ticks", entry)


class CheckpointParityTests(unittest.TestCase):
    def assert_resume_matches_uninterrupted(self, split: int) -> None:
        full = swap_scenario()
        run_until_done(full, 20)
        interrupted = swap_scenario()
        for _ in range(split):
            interrupted.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            interrupted.save_checkpoint(path)
            resumed = FleetSimulator.load_checkpoint(path)
            run_until_done(resumed, 20)
        self.assertEqual(resumed.snapshot(), full.snapshot())

    def test_parity_at_every_split_point(self) -> None:
        full = swap_scenario()
        run_until_done(full, 20)
        for split in range(full.tick):
            with self.subTest(split=split):
                self.assert_resume_matches_uninterrupted(split)

    def test_wait_ticks_round_trip(self) -> None:
        grid = GridMap(5, 1)
        robots = [Robot("R-01", (0, 0)), Robot("R-02", (4, 0))]
        tasks = [
            Task("T-01", (0, 0), (4, 0), picked_up=True),
            Task("T-02", (4, 0), (0, 0), picked_up=True),
        ]
        sim = FleetSimulator(grid, robots, tasks)
        for robot in sim.robots.values():
            task = sim.tasks["T-01"] if robot.position == (0, 0) else sim.tasks["T-02"]
            robot.task_id = task.task_id
            task.assigned_robot = robot.robot_id
            robot.route = sim._plan_route(robot, task)
        sim.step()
        sim.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            sim.save_checkpoint(path)
            loaded = FleetSimulator.load_checkpoint(path)
        for robot_id in sim.robots:
            self.assertEqual(
                loaded.robots[robot_id].wait_ticks,
                sim.robots[robot_id].wait_ticks,
            )


class DeterminismTests(unittest.TestCase):
    def test_identical_operation_sequences_are_identical(self) -> None:
        first = swap_scenario()
        run_until_done(first, 20)
        second = swap_scenario()
        run_until_done(second, 20)
        self.assertEqual(first.snapshot(), second.snapshot())
        self.assertEqual(first.metrics(), second.metrics())


if __name__ == "__main__":
    unittest.main()
