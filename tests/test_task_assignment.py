"""Regression tests for assigning waiting tasks to idle robots.

The tests call ``FleetSimulator.assign_tasks`` directly and observe the result
through the public ``robots`` / ``tasks`` entities: which robot got the task,
the mutual ownership binding, and the remaining route left on the robot. No
tick is ever executed, so a wrong choice can never hide behind the task still
being completed by some robot later on.
"""

import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task, shortest_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assignment_view(simulator: FleetSimulator) -> tuple[dict[str, str], dict[str, list]]:
    """Task -> robot and robot -> remaining route after an assignment pass."""
    owners = {
        task.task_id: task.assigned_robot
        for task in simulator.tasks.values()
    }
    routes = {
        robot.robot_id: list(robot.route)
        for robot in simulator.robots.values()
    }
    return owners, routes


def assert_route_is_feasible(
    testcase: unittest.TestCase,
    grid: GridMap,
    start: tuple[int, int],
    route: list[tuple[int, int]],
    pickup: tuple[int, int],
    dropoff: tuple[int, int],
) -> None:
    """The remaining route starts next to the robot and covers pickup->dropoff.

    Every waypoint must be traversable and orthogonally adjacent to the
    previous cell (the route never jumps or crosses an obstacle), the route
    must end at the dropoff and must visit the pickup unless the robot is
    already standing on it, and its length must be the shortest feasible
    robot -> pickup -> dropoff distance on the current map.
    """
    testcase.assertTrue(route, "an assigned task must leave a non-empty route")
    previous = start
    for cell in route:
        testcase.assertTrue(
            grid.traversable(cell), f"route crosses blocked/outside cell {cell}"
        )
        testcase.assertEqual(
            abs(cell[0] - previous[0]) + abs(cell[1] - previous[1]),
            1,
            f"route jumps non-adjacently from {previous} to {cell}",
        )
        previous = cell
    testcase.assertEqual(route[-1], dropoff, "route must finish at the dropoff")
    if start != pickup:
        testcase.assertIn(pickup, route, "route must pass through the pickup point")
    to_pickup = shortest_path(grid, start, pickup) if start != pickup else []
    to_dropoff = shortest_path(grid, pickup, dropoff)
    testcase.assertEqual(
        len(route),
        len(to_pickup) + len(to_dropoff),
        "route is not a shortest feasible route via the pickup point",
    )


def assert_no_side_effects(
    testcase: unittest.TestCase,
    simulator: FleetSimulator,
    positions: dict[str, tuple[int, int]],
) -> None:
    """Assignment alone must not drive, count distance, collect or finish."""
    testcase.assertEqual(simulator.tick, 0)
    testcase.assertEqual(simulator.replay, [])
    testcase.assertEqual(simulator.map_change_history(), [])
    testcase.assertEqual(simulator.paused_tasks, set())
    for robot_id, position in positions.items():
        robot = simulator.robots[robot_id]
        testcase.assertEqual(robot.position, position)
        testcase.assertEqual(robot.distance_travelled, 0)
    for task in simulator.tasks.values():
        testcase.assertFalse(task.picked_up)
        testcase.assertFalse(task.completed)


def initial_position(
    specs: list[tuple[str, tuple[int, int]]], robot_id: str
) -> tuple[int, int]:
    return dict(specs)[robot_id]


def assert_bindings_consistent(testcase: unittest.TestCase, simulator: FleetSimulator) -> None:
    """Every assigned task and its robot point at each other, and only each other."""
    for task in simulator.tasks.values():
        if task.assigned_robot is None:
            continue
        robot = simulator.robots[task.assigned_robot]
        testcase.assertEqual(
            robot.task_id,
            task.task_id,
            f"task {task.task_id} claims {task.assigned_robot} but the robot "
            f"executes {robot.task_id!r}",
        )
    for robot in simulator.robots.values():
        if robot.task_id is None:
            continue
        testcase.assertEqual(
            simulator.tasks[robot.task_id].assigned_robot,
            robot.robot_id,
            f"robot {robot.robot_id} executes {robot.task_id} but the task does "
            "not claim it back",
        )
    unassigned = [
        task.task_id
        for task in simulator.tasks.values()
        if task.assigned_robot is None
    ]
    claimed = {robot.task_id for robot in simulator.robots.values()}
    testcase.assertTrue(claimed.isdisjoint(unassigned))


# ---------------------------------------------------------------------------
# Robot selection
# ---------------------------------------------------------------------------


class NearestFeasibleRobotTests(unittest.TestCase):
    def test_obstacle_detour_picks_robot_with_shorter_actual_route(self) -> None:
        # Wall at x=4 for rows 0..3; the only gap is through y=4.
        # R-near sits close to the pickup on the far side but must loop all the
        # way around (7 steps to pickup); R-far approaches through the gap and
        # reaches the pickup in 6 despite its larger straight-line distance.
        grid = GridMap(7, 5, frozenset({(4, 0), (4, 1), (4, 2), (4, 3)}))
        pickup, dropoff = (5, 2), (6, 2)
        near_route = shortest_path(grid, (3, 2), pickup) + shortest_path(
            grid, pickup, dropoff
        )
        far_route = shortest_path(grid, (2, 4), pickup) + shortest_path(
            grid, pickup, dropoff
        )
        self.assertGreater(len(near_route), len(far_route))

        simulator = FleetSimulator(
            grid,
            [Robot("R-near", (3, 2)), Robot("R-far", (2, 4))],
            [Task("T-1", pickup, dropoff)],
        )
        simulator.assign_tasks()

        self.assertEqual(simulator.tasks["T-1"].assigned_robot, "R-far")
        winner = simulator.robots["R-far"]
        self.assertEqual(winner.task_id, "T-1")
        self.assertIsNone(simulator.robots["R-near"].task_id)
        self.assertEqual(
            winner.route,
            [(3, 4), (4, 4), (5, 4), (5, 3), (5, 2), (6, 2)],
        )
        assert_route_is_feasible(self, grid, (2, 4), winner.route, pickup, dropoff)
        assert_bindings_consistent(self, simulator)
        assert_no_side_effects(self, simulator, {"R-near": (3, 2), "R-far": (2, 4)})

    def test_closer_busy_robot_is_not_reassigned(self) -> None:
        # R-1 already works T-1 and stands one cell from T-2's pickup; R-2 is
        # much farther. T-2 must go to idle R-2, never stealing R-1, and R-1's
        # existing binding/route must be left exactly as they were.
        grid = GridMap(8, 2)
        busy_route = [(2, 0), (3, 0)]
        robots = [
            Robot("R-1", (1, 0), route=list(busy_route), task_id="T-1"),
            Robot("R-2", (7, 0)),
        ]
        tasks = [
            Task("T-1", (2, 0), (3, 0), assigned_robot="R-1"),
            Task("T-2", (0, 0), (0, 1)),
        ]
        simulator = FleetSimulator(grid, robots, tasks)
        simulator.assign_tasks()

        self.assertEqual(simulator.tasks["T-1"].assigned_robot, "R-1")
        self.assertEqual(simulator.robots["R-1"].task_id, "T-1")
        self.assertEqual(simulator.robots["R-1"].route, busy_route)
        self.assertEqual(simulator.tasks["T-2"].assigned_robot, "R-2")
        self.assertEqual(
            simulator.robots["R-2"].route,
            [(6, 0), (5, 0), (4, 0), (3, 0), (2, 0), (1, 0), (0, 0), (0, 1)],
        )
        assert_bindings_consistent(self, simulator)

    def test_equal_route_length_breaks_tie_by_robot_id_string(self) -> None:
        # Both idle robots need exactly 4 steps; lexicographically "R-10" <
        # "R-9" even though 9 < 10 numerically.
        grid = GridMap(5, 3)
        pickup, dropoff = (2, 0), (2, 2)
        simulator = FleetSimulator(
            grid,
            [Robot("R-9", (4, 0)), Robot("R-10", (0, 0))],
            [Task("T-1", pickup, dropoff)],
        )
        simulator.assign_tasks()

        self.assertEqual(simulator.tasks["T-1"].assigned_robot, "R-10")
        winner = simulator.robots["R-10"]
        loser = simulator.robots["R-9"]
        self.assertEqual(winner.task_id, "T-1")
        self.assertIsNone(loser.task_id)
        self.assertEqual(winner.route, [(1, 0), (2, 0), (2, 1), (2, 2)])
        tied_loser_route = shortest_path(grid, (4, 0), pickup) + shortest_path(
            grid, pickup, dropoff
        )
        self.assertEqual(len(tied_loser_route), len(winner.route))
        assert_route_is_feasible(self, grid, (0, 0), winner.route, pickup, dropoff)
        assert_bindings_consistent(self, simulator)

    def test_route_starts_at_next_cell_and_covers_full_pickup_dropoff_legs(self) -> None:
        grid = GridMap(6, 3, frozenset({(2, 1)}))
        pickup, dropoff = (4, 2), (1, 0)
        simulator = FleetSimulator(
            grid, [Robot("R-1", (0, 0))], [Task("T-1", pickup, dropoff)]
        )
        simulator.assign_tasks()
        robot = simulator.robots["R-1"]
        # The remaining route begins at the cell after the current position:
        # it never repeats the robot's own cell and its first waypoint is one
        # orthogonal step away.
        self.assertNotEqual(robot.route[0], (0, 0))
        self.assertEqual(
            abs(robot.route[0][0]) + abs(robot.route[0][1]), 1
        )
        assert_route_is_feasible(self, grid, (0, 0), robot.route, pickup, dropoff)

    def test_passing_dropoff_on_the_way_still_includes_loaded_return_leg(self) -> None:
        # Pickup lies beyond the dropoff along a one-row corridor: the empty
        # robot passes the dropoff cell on its way out and must still route back
        # to it after collecting.
        grid = GridMap(6, 1)
        pickup, dropoff = (5, 0), (2, 0)
        simulator = FleetSimulator(
            grid, [Robot("R-1", (0, 0))], [Task("T-1", pickup, dropoff)]
        )
        simulator.assign_tasks()
        route = simulator.robots["R-1"].route
        self.assertEqual(
            route,
            [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (4, 0), (3, 0), (2, 0)],
        )
        # Dropoff is visited in transit (index 1) and again after pickup.
        self.assertEqual(route.index((2, 0)), 1)
        self.assertEqual(route.count((2, 0)), 2)
        self.assertEqual(route[-1], (2, 0))
        assert_route_is_feasible(self, grid, (0, 0), route, pickup, dropoff)


# ---------------------------------------------------------------------------
# Multiple waiting tasks
# ---------------------------------------------------------------------------


class MultipleTasksTests(unittest.TestCase):
    def test_tasks_processed_in_task_id_string_order(self) -> None:
        grid = GridMap(7, 3)
        robots = [Robot("R-2", (0, 0)), Robot("R-1", (4, 0))]
        tasks = [Task("T-b", (6, 0), (6, 2)), Task("T-a", (3, 0), (3, 2))]
        simulator = FleetSimulator(grid, robots, tasks)
        simulator.assign_tasks()

        # "T-a" is processed first and takes nearer idle robot R-1.
        self.assertEqual(simulator.tasks["T-a"].assigned_robot, "R-1")
        self.assertEqual(simulator.robots["R-1"].route, [(3, 0), (3, 1), (3, 2)])
        self.assertEqual(simulator.tasks["T-b"].assigned_robot, "R-2")
        self.assertEqual(
            simulator.robots["R-2"].route,
            [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (6, 1), (6, 2)],
        )
        assert_bindings_consistent(self, simulator)

    def test_robot_taken_by_earlier_task_is_not_candidate_for_later_task(self) -> None:
        grid = GridMap(7, 3)
        # Considered on its own, R-1 would win T-b as well (4 steps vs R-2's 8);
        # it must be unavailable because T-a already claimed it.
        alone = FleetSimulator(
            grid,
            [Robot("R-1", (4, 0)), Robot("R-2", (0, 0))],
            [Task("T-b", (6, 0), (6, 2))],
        )
        alone.assign_tasks()
        self.assertEqual(alone.tasks["T-b"].assigned_robot, "R-1")

        simulator = FleetSimulator(
            grid,
            [Robot("R-1", (4, 0)), Robot("R-2", (0, 0))],
            [Task("T-a", (3, 0), (3, 2)), Task("T-b", (6, 0), (6, 2))],
        )
        simulator.assign_tasks()
        self.assertEqual(simulator.tasks["T-a"].assigned_robot, "R-1")
        self.assertEqual(simulator.tasks["T-b"].assigned_robot, "R-2")
        self.assertEqual(simulator.robots["R-1"].task_id, "T-a")
        assert_bindings_consistent(self, simulator)

    def test_same_state_is_independent_of_input_entity_order(self) -> None:
        grid = GridMap(8, 4, frozenset({(4, 1), (4, 2)}))
        robot_specs = [
            ("R-1", (0, 0)),
            ("R-2", (7, 3)),
            ("R-10", (1, 3)),
            ("R-3", (6, 0)),
        ]
        task_specs = [
            ("T-2", (5, 3), (0, 2)),
            ("T-1", (2, 0), (7, 1)),
            ("T-20", (3, 3), (6, 2)),
        ]

        def build(reversed_entities: bool) -> FleetSimulator:
            robots = [Robot(rid, pos) for rid, pos in robot_specs]
            tasks = [Task(tid, pickup, dropoff) for tid, pickup, dropoff in task_specs]
            if reversed_entities:
                robots.reverse()
                tasks.reverse()
            simulator = FleetSimulator(grid, robots, tasks)
            simulator.assign_tasks()
            return simulator

        forward = build(False)
        reversed_order = build(True)
        self.assertEqual(assignment_view(forward), assignment_view(reversed_order))
        assert_bindings_consistent(self, forward)
        assert_bindings_consistent(self, reversed_order)
        for task in forward.tasks.values():
            if task.assigned_robot is not None:
                robot = forward.robots[task.assigned_robot]
                assert_route_is_feasible(
                    self,
                    grid,
                    initial_position(robot_specs, robot.robot_id),
                    robot.route,
                    task.pickup,
                    task.dropoff,
                )

    def test_assignment_is_idempotent(self) -> None:
        grid = GridMap(7, 2)
        simulator = FleetSimulator(
            grid,
            [Robot("R-1", (0, 0)), Robot("R-2", (6, 0))],
            [Task("T-a", (2, 0), (3, 0)), Task("T-b", (5, 0), (1, 0))],
        )
        simulator.assign_tasks()
        first = assignment_view(simulator)
        simulator.assign_tasks()
        simulator.assign_tasks()
        self.assertEqual(assignment_view(simulator), first)
        self.assertEqual(simulator.tick, 0)
        self.assertEqual(simulator.replay, [])


# ---------------------------------------------------------------------------
# Tasks nobody can take
# ---------------------------------------------------------------------------


class UnassignableTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        # A ring of obstacles seals the 3x3 interior component (its centre is
        # (2, 2)) off from every outside cell.
        self.ring = frozenset(
            {
                (1, 1), (2, 1), (3, 1),
                (1, 2), (3, 2),
                (1, 3), (2, 3), (3, 3),
            }
        )
        self.grid = GridMap(5, 5, self.ring)

    def assert_task_left_waiting(
        self, simulator: FleetSimulator, task_id: str, robot_ids: list[str]
    ) -> None:
        task = simulator.tasks[task_id]
        self.assertIsNone(task.assigned_robot)
        self.assertFalse(task.picked_up)
        self.assertFalse(task.completed)
        for robot_id in robot_ids:
            robot = simulator.robots[robot_id]
            self.assertIsNone(robot.task_id)
            self.assertEqual(robot.route, [])
        self.assertNotIn(task_id, simulator.paused_tasks)

    def test_no_idle_robot_leaves_task_waiting(self) -> None:
        simulator = FleetSimulator(
            GridMap(6, 1),
            [Robot("R-1", (0, 0), route=[(1, 0), (2, 0)], task_id="T-busy")],
            [
                Task("T-busy", (1, 0), (2, 0), assigned_robot="R-1"),
                Task("T-lonely", (4, 0), (5, 0)),
            ],
        )
        simulator.assign_tasks()
        self.assertIsNone(simulator.tasks["T-lonely"].assigned_robot)
        self.assertEqual(simulator.robots["R-1"].task_id, "T-busy")
        self.assertEqual(simulator.robots["R-1"].route, [(1, 0), (2, 0)])
        self.assertEqual(simulator.tick, 0)
        self.assertEqual(simulator.replay, [])

    def test_pickup_unreachable_cannot_be_accepted(self) -> None:
        simulator = FleetSimulator(
            self.grid,
            [Robot("R-1", (0, 0)), Robot("R-2", (4, 4))],
            [Task("T-sealed-pick", (2, 2), (4, 0))],
        )
        simulator.assign_tasks()
        self.assert_task_left_waiting(simulator, "T-sealed-pick", ["R-1", "R-2"])

    def test_reachable_pickup_but_unreachable_dropoff_cannot_be_accepted(self) -> None:
        simulator = FleetSimulator(
            self.grid,
            [Robot("R-1", (0, 0))],
            [Task("T-sealed-drop", (0, 2), (2, 2))],
        )
        simulator.assign_tasks()
        # The pickup is plainly reachable from the robot; only the delivery is
        # sealed off -- accepting is still forbidden.
        self.assertEqual(shortest_path(self.grid, (0, 0), (0, 2)), [(0, 1), (0, 2)])
        self.assert_task_left_waiting(simulator, "T-sealed-drop", ["R-1"])

    def test_earlier_unreachable_task_does_not_block_later_reachable_one(self) -> None:
        simulator = FleetSimulator(
            self.grid,
            [Robot("R-1", (0, 0)), Robot("R-2", (4, 0))],
            [
                Task("T-a-pick-sealed", (2, 2), (4, 4)),
                Task("T-b-drop-sealed", (0, 2), (2, 2)),
                Task("T-c-fine", (0, 2), (4, 2)),
            ],
        )
        simulator.assign_tasks()

        self.assertIsNone(simulator.tasks["T-a-pick-sealed"].assigned_robot)
        self.assertIsNone(simulator.tasks["T-b-drop-sealed"].assigned_robot)
        self.assertEqual(simulator.tasks["T-c-fine"].assigned_robot, "R-1")
        winner = simulator.robots["R-1"]
        assert_route_is_feasible(
            self, self.grid, (0, 0), winner.route, (0, 2), (4, 2)
        )
        self.assertIsNone(simulator.robots["R-2"].task_id)
        # Rejected tasks leave no half-registered ownership anywhere.
        for robot in simulator.robots.values():
            self.assertNotIn(robot.task_id, ("T-a-pick-sealed", "T-b-drop-sealed"))
        assert_bindings_consistent(self, simulator)

    def test_failed_assignment_advances_neither_clock_nor_replay(self) -> None:
        simulator = FleetSimulator(
            self.grid,
            [Robot("R-1", (0, 0))],
            [
                Task("T-a", (2, 2), (4, 0)),
                Task("T-b", (2, 2), (0, 4)),
            ],
        )
        positions = {"R-1": (0, 0)}
        simulator.assign_tasks()
        assert_no_side_effects(self, simulator, positions)
        # A second pass after every task is still waiting changes nothing.
        simulator.assign_tasks()
        assert_no_side_effects(self, simulator, positions)


if __name__ == "__main__":
    unittest.main()
