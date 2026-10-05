"""Taskless robots that still carry a remaining route.

Creating a fleet through the Python interface may leave a robot without a
bound task while it still has a legitimate remaining route -- a route that is
executed cell by cell in later ``step()`` calls just like any other. Such a
robot is *driving*, not parked: traffic must never push it onto a side cell
off its route (the old code equated "no task" with "parked", which moved the
robot but kept its route, so the next tick stepped diagonally back toward the
next waypoint), it may never be asked to clear the route, and a legal initial
state must not be rejected to dodge the issue. Only once the route is
exhausted does the robot become an ordinary parked robot that yields onto a
safe side cell.
"""

import random
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task, shortest_path


def positions(simulator: FleetSimulator) -> dict[str, tuple[int, int]]:
    return {robot_id: robot.position for robot_id, robot in simulator.robots.items()}


class TasklessRouteDrivesOnTests(unittest.TestCase):
    def build(self) -> FleetSimulator:
        # 4 wide, 2 tall, no obstacles, no tasks: A queued behind B, both
        # carrying routes east along the top row.
        return FleetSimulator(
            GridMap(4, 2),
            [
                Robot("A", (0, 0), route=[(1, 0), (2, 0), (3, 0)]),
                Robot("B", (1, 0), route=[(2, 0), (3, 0)]),
            ],
            [],
        )

    def test_first_step_blocker_drives_its_route_and_waits_arent_recorded(self) -> None:
        sim = self.build()
        event = sim.step()
        # B follows its own route; A waits this tick but is never jammed at
        # tick end because B vacated (1, 0).
        self.assertEqual(event["moved"], ["B"])
        self.assertEqual(sim.robots["A"].position, (0, 0))
        self.assertEqual(sim.robots["A"].route, [(1, 0), (2, 0), (3, 0)])
        self.assertEqual(sim.robots["A"].distance_travelled, 0)
        self.assertEqual(sim.robots["B"].position, (2, 0))
        self.assertEqual(sim.robots["B"].route, [(3, 0)])
        self.assertEqual(sim.robots["B"].distance_travelled, 1)
        self.assertIsNone(sim.robots["B"].task_id)
        self.assertEqual(sim.status()["traffic_waits"], [])

    def test_follower_moves_in_behind_and_route_ends_on_axis(self) -> None:
        sim = self.build()
        sim.step()
        event = sim.step()
        self.assertEqual(event["moved"], ["A", "B"])
        self.assertEqual(sim.robots["A"].position, (1, 0))
        self.assertEqual(sim.robots["A"].route, [(2, 0), (3, 0)])
        self.assertEqual(sim.robots["B"].position, (3, 0))
        self.assertEqual(sim.robots["B"].route, [])
        event = sim.step()
        self.assertEqual(event["moved"], ["A"])
        self.assertEqual(sim.robots["A"].position, (2, 0))
        self.assertEqual(sim.robots["B"].position, (3, 0))
        # B never appears on the side row while it still had a route.
        for frame in sim.replay:
            if frame.get("type") == "tick":
                self.assertEqual(frame["robots"]["B"][1], 0)

    def test_once_route_is_exhausted_the_robot_yields_like_any_parked_robot(self) -> None:
        sim = self.build()
        for _ in range(3):
            sim.step()
        # B is now at (3, 0) with an empty route and no task; A needs (3, 0)
        # and the side cell (3, 1) is free.
        event = sim.step()
        self.assertEqual(event["moved"], ["B", "A"])
        self.assertEqual(sim.robots["B"].position, (3, 1))
        self.assertIsNone(sim.robots["B"].task_id)
        self.assertEqual(sim.robots["A"].position, (3, 0))
        # B drove both of its own route cells and then one yield cell.
        self.assertEqual(sim.robots["B"].distance_travelled, 3)
        # Parked after yielding: it does not wander back.
        for _ in range(3):
            sim.step()
        self.assertEqual(sim.robots["B"].position, (3, 1))
        self.assertEqual(sim.robots["B"].distance_travelled, 3)

    def test_full_run_never_moves_diagonally_or_loses_waypoints(self) -> None:
        sim = self.build()
        previous = positions(sim)
        for _ in range(6):
            event = sim.step()
            for robot_id, cell in event["robots"].items():
                old = previous[robot_id]
                distance = abs(cell[0] - old[0]) + abs(cell[1] - old[1])
                self.assertLessEqual(distance, 1)
            previous = {rid: tuple(cell) for rid, cell in event["robots"].items()}
        # Every route waypoint B started with was actually visited in order.
        visited_b = [
            tuple(frame["robots"]["B"])
            for frame in sim.replay
            if frame.get("type") == "tick"
        ]
        self.assertEqual(visited_b[:3], [(2, 0), (3, 0), (3, 0)])


class TasklessRouteIsNotPushedTests(unittest.TestCase):
    def test_open_side_cell_does_not_side_move_a_blocked_taskless_robot(self) -> None:
        # (1, 1) is a free side cell right next to B, but B's route points at
        # (2, 0) where parked P blocks it. B must stay on (1, 0) with its
        # route intact, never sidestep into (1, 1).
        sim = FleetSimulator(
            GridMap(3, 2, frozenset({(2, 1)})),
            [
                Robot("B", (1, 0), route=[(2, 0)]),
                Robot("P", (2, 0)),
            ],
            [],
        )
        for _ in range(4):
            event = sim.step()
            self.assertEqual(event["moved"], [])
            self.assertEqual(sim.robots["B"].position, (1, 0))
            self.assertEqual(sim.robots["B"].route, [(2, 0)])
            self.assertEqual(sim.robots["B"].distance_travelled, 0)
            self.assertEqual(sim.robots["P"].position, (2, 0))

    def test_requester_behind_may_not_push_a_taskless_moving_blocker_aside(self) -> None:
        # A is task-bound behind B, who is taskless but still driving east;
        # parked P seals the corridor at (2, 0). (1, 1) beside B is open --
        # the old behaviour parked B there, stranding it off its route.
        grid = GridMap(3, 2, frozenset({(0, 1), (2, 1)}))
        sim = FleetSimulator(
            grid,
            [
                Robot("A", (0, 0), route=[(1, 0), (2, 0)], task_id="T-1"),
                Robot("B", (1, 0), route=[(2, 0)]),
                Robot("P", (2, 0)),
            ],
            [Task("T-1", (0, 0), (2, 0), assigned_robot="A", picked_up=True)],
        )
        event = sim.step()
        self.assertEqual(event["moved"], [])
        self.assertEqual(sim.robots["A"].position, (0, 0))
        self.assertEqual(sim.robots["B"].position, (1, 0))
        self.assertEqual(sim.robots["B"].route, [(2, 0)])
        self.assertEqual(sim.robots["P"].position, (2, 0))
        # Waits reflect the real end-of-tick chain: A behind B behind P.
        self.assertEqual(
            sim.status()["traffic_waits"],
            [
                {"robot_id": "A", "blocked_by": ["B"], "ticks": 1},
                {"robot_id": "B", "blocked_by": ["P"], "ticks": 1},
            ],
        )
        # Give P somewhere to yield to: the whole chain moves in one tick and
        # B is still on its own route, not coming from the side pocket.
        sim.modify_obstacles(removed=[(2, 1)])
        event = sim.step()
        self.assertEqual(event["moved"], ["P", "B"])
        self.assertEqual(sim.robots["P"].position, (2, 1))
        self.assertEqual(sim.robots["B"].position, (2, 0))
        self.assertEqual(sim.robots["B"].route, [])
        self.assertEqual(sim.robots["A"].position, (0, 0))
        self.assertEqual(sim.status()["traffic_waits"], [])

    def test_taskless_blocker_that_drives_on_clears_the_requesters_wait(self) -> None:
        # Same-direction chain: B cannot clear (1, 0) before it is processed,
        # so A records a provisional wait, but B drives on in the same tick.
        sim = FleetSimulator(
            GridMap(4, 1),
            [
                Robot("A", (0, 0), route=[(1, 0), (2, 0)], task_id="T-1"),
                Robot("B", (1, 0), route=[(2, 0), (3, 0)]),
            ],
            [Task("T-1", (0, 0), (2, 0), assigned_robot="A", picked_up=True)],
        )
        event = sim.step()
        self.assertEqual(event["moved"], ["B"])
        self.assertEqual(sim.robots["B"].position, (2, 0))
        self.assertEqual(sim.robots["B"].route, [(3, 0)])
        self.assertEqual(sim.status()["traffic_waits"], [])
        event = sim.step()
        self.assertEqual(event["moved"], ["A", "B"])
        self.assertEqual(sim.robots["A"].position, (1, 0))
        self.assertEqual(sim.robots["B"].position, (3, 0))
        self.assertEqual(sim.robots["B"].route, [])
        # A follows through the vacated cells and delivers with no waits ever.
        event = sim.step()
        self.assertEqual(event["moved"], ["A"])
        self.assertTrue(sim.tasks["T-1"].completed)
        self.assertEqual(sim.robots["A"].position, (2, 0))
        self.assertEqual(sim.status()["traffic_waits"], [])


class TasklessRouteInvariantFuzzTests(unittest.TestCase):
    """Random fleets: taskless routes are always followed, never side-stepped."""

    def _random_fleet(self, rng: random.Random):
        width, height = 5, 3
        cells = [(x, y) for x in range(width) for y in range(height)]
        obstacles = frozenset(
            cell for cell in cells[1:] if rng.random() < 0.12 and cell != (0, 0)
        )
        grid = GridMap(width, height, obstacles)
        free = [cell for cell in cells if grid.traversable(cell)]
        rng.shuffle(free)
        robots = []
        tasks = []
        for index in range(rng.randint(2, 5)):
            if len(free) <= index:
                break
            start = free[index]
            kind = rng.choice(("parked", "taskless", "bound"))
            if kind == "parked":
                robots.append(Robot(f"R{index}", start))
                continue
            # Build a short legal orthogonal walk through free cells.
            route = []
            current = start
            for _ in range(rng.randint(1, 5)):
                options = [
                    cell
                    for cell in grid.neighbors(current)
                    if not route or cell != route[-1]
                ]
                if not options:
                    break
                current = rng.choice(options)
                route.append(current)
            if kind == "taskless" or not route:
                robots.append(Robot(f"R{index}", start, route=route))
                continue
            # Bind a feasible loaded delivery: the remaining route is the
            # shortest walk to the route's end cell, which is the dropoff.
            dropoff = route[-1]
            planned = shortest_path(grid, start, dropoff)
            if planned is None:
                robots.append(Robot(f"R{index}", start))
                continue
            task_id = f"T{index}"
            tasks.append(
                Task(
                    task_id,
                    start,
                    dropoff,
                    assigned_robot=f"R{index}",
                    picked_up=True,
                )
            )
            robots.append(Robot(f"R{index}", start, route=planned, task_id=task_id))
        return grid, robots, tasks

    def _assert_frame_physics(
        self, grid: GridMap, previous: dict, event: dict, sim: FleetSimulator
    ) -> None:
        positions_now = {rid: tuple(cell) for rid, cell in event["robots"].items()}
        # No overlaps, no obstacle entries.
        self.assertEqual(len(positions_now), len(set(positions_now.values())))
        for cell in positions_now.values():
            self.assertTrue(grid.traversable(cell))
        movers = set()
        if previous:
            for robot_id, cell in positions_now.items():
                old = previous[robot_id]
                distance = abs(cell[0] - old[0]) + abs(cell[1] - old[1])
                self.assertLessEqual(distance, 1, (robot_id, old, cell))
                if distance == 1:
                    movers.add(robot_id)
            # No two-robot swaps.
            for robot_id, cell in positions_now.items():
                for other_id, other_cell in positions_now.items():
                    if robot_id < other_id:
                        self.assertFalse(
                            cell == previous[other_id]
                            and other_cell == previous[robot_id]
                        )
        self.assertEqual(set(event["moved"]), movers)
        return positions_now

    def test_invariants_hold_over_many_random_fleets(self) -> None:
        rng = random.Random(20240517)
        for seed in range(120):
            rng.seed(20240517 + seed)
            grid, robots, tasks = self._random_fleet(rng)
            try:
                sim = FleetSimulator(grid, robots, tasks)
            except ValueError:
                continue
            taskless = {
                robot.robot_id
                for robot in robots
                if robot.task_id is None and robot.route
            }
            if not taskless:
                continue
            previous = positions(sim)
            for _ in range(25):
                before = {
                    robot_id: (
                        sim.robots[robot_id].position,
                        list(sim.robots[robot_id].route),
                    )
                    for robot_id in taskless
                }
                event = sim.step()
                current = self._assert_frame_physics(grid, previous, event, sim)
                previous = current
                for robot_id in taskless:
                    robot = sim.robots[robot_id]
                    old_position, old_route = before[robot_id]
                    if not old_route:
                        # Route finished earlier: this is an ordinary parked
                        # robot now and a yield side-step is allowed; the frame
                        # physics above already bound the move to one cell.
                        continue
                    if robot.position != old_position:
                        # While a route remains, the robot may only ever
                        # advance onto the head of its own remaining route,
                        # consuming exactly that waypoint -- never a side
                        # cell, never a jump.
                        self.assertEqual(robot.position, old_route[0])
                        self.assertEqual(robot.route, old_route[1:])
                    else:
                        self.assertEqual(
                            robot.route,
                            old_route,
                            f"taskless robot {robot_id} lost waypoints while waiting",
                        )


if __name__ == "__main__":
    unittest.main()
