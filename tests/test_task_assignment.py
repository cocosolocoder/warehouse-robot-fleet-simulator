import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task, shortest_path


def assign_result(simulator: FleetSimulator) -> dict[str, tuple[str, tuple]]:
    """Task id -> (assigned robot id, remaining route) for assigned tasks."""
    return {
        task.task_id: (task.assigned_robot, tuple(simulator.robots[task.assigned_robot].route))
        for task in simulator.tasks.values()
        if task.assigned_robot is not None
    }


class RouteValidityMixin:
    def assert_route_valid(
        self,
        simulator: FleetSimulator,
        robot: Robot,
        task: Task,
    ) -> None:
        """The remaining route is a continuous walk from the robot's next
        cell over traversable adjacent cells, covering the full pickup and
        dropoff legs without crossing obstacles."""
        self.assertTrue(robot.route, "assigned robot must hold a remaining route")
        cells = [robot.position, *robot.route]
        for previous, current in zip(cells, cells[1:]):
            self.assertEqual(
                abs(previous[0] - current[0]) + abs(previous[1] - current[1]),
                1,
                f"route step {previous} -> {current} is not adjacent",
            )
        for cell in robot.route:
            self.assertTrue(
                simulator.grid.traversable(cell),
                f"route crosses untraversable cell {cell}",
            )
        if robot.position != task.pickup:
            self.assertIn(
                task.pickup, robot.route, "route must reach the pickup cell"
            )
        self.assertEqual(
            robot.route[-1],
            task.dropoff,
            "route must end on the dropoff cell",
        )
        if task.pickup in robot.route and task.pickup != task.dropoff:
            # The pickup leg comes first; the delivery leg follows it.
            pickup_index = robot.route.index(task.pickup)
            self.assertLess(pickup_index, len(robot.route) - 1)


class ShortestRouteSelectionTests(RouteValidityMixin, unittest.TestCase):
    def setUp(self) -> None:
        # Wall column x=2 with a single gap at (2, 3): R-near looks closer to
        # the pickup by straight-line distance but must detour around the
        # wall, so R-far has the shorter actual route.
        self.grid = GridMap(5, 4, frozenset({(2, 0), (2, 1), (2, 2)}))
        self.sim = FleetSimulator(
            self.grid,
            [Robot("R-near", (1, 1)), Robot("R-far", (4, 3))],
            [Task("T", (3, 1), (4, 1))],
        )

    def test_obstacle_detour_loses_to_shorter_actual_route(self) -> None:
        self.sim.assign_tasks()
        task = self.sim.tasks["T"]
        self.assertEqual(task.assigned_robot, "R-far")
        # R-near could have completed the task too, but its route around the
        # wall is longer, so it must stay unassigned and idle.
        self.assertIsNone(self.sim.robots["R-near"].task_id)
        self.assertEqual(self.sim.robots["R-near"].route, [])
        near_leg = shortest_path(self.grid, (1, 1), (3, 1))
        far_leg = shortest_path(self.grid, (4, 3), (3, 1))
        self.assertGreater(len(near_leg), len(far_leg))

    def test_assigned_route_is_the_shortest_full_route(self) -> None:
        self.sim.assign_tasks()
        robot = self.sim.robots["R-far"]
        task = self.sim.tasks["T"]
        expected = shortest_path(self.grid, (4, 3), (3, 1)) + shortest_path(
            self.grid, (3, 1), (4, 1)
        )
        self.assertEqual(robot.route, expected)
        self.assert_route_valid(self.sim, robot, task)

    def test_busy_robot_is_not_reassigned_even_when_closer(self) -> None:
        sim = FleetSimulator(
            GridMap(6, 1),
            [Robot("R-busy", (1, 0)), Robot("R-idle", (4, 0))],
            [
                Task("T-old", (0, 0), (5, 0)),
                Task("T-new", (2, 0), (3, 0)),
            ],
        )
        sim.robots["R-busy"].task_id = "T-old"
        sim.tasks["T-old"].assigned_robot = "R-busy"
        sim.robots["R-busy"].route = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
        sim.assign_tasks()
        # R-busy sits closer to the pickup but already executes a task.
        self.assertEqual(sim.tasks["T-new"].assigned_robot, "R-idle")
        self.assertEqual(sim.robots["R-busy"].task_id, "T-old")
        self.assertEqual(
            sim.robots["R-busy"].route,
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)],
        )
        self.assertEqual(
            sim.robots["R-idle"].route, [(3, 0), (2, 0), (3, 0)]
        )


class AssignmentStateTests(RouteValidityMixin, unittest.TestCase):
    def test_task_and_robot_records_match(self) -> None:
        sim = FleetSimulator(
            GridMap(4, 2),
            [Robot("R", (0, 0))],
            [Task("T", (1, 0), (3, 0))],
        )
        sim.assign_tasks()
        task = sim.tasks["T"]
        robot = sim.robots["R"]
        self.assertEqual(task.assigned_robot, "R")
        self.assertEqual(robot.task_id, "T")
        self.assertFalse(robot.idle)
        self.assert_route_valid(sim, robot, task)

    def test_route_may_pass_dropoff_before_pickup(self) -> None:
        # The only way to the pickup leads straight through the dropoff cell;
        # the route must still include the post-pickup leg back to it.
        sim = FleetSimulator(
            GridMap(3, 1),
            [Robot("R", (0, 0))],
            [Task("T", (2, 0), (1, 0))],
        )
        sim.assign_tasks()
        robot = sim.robots["R"]
        self.assertEqual(sim.tasks["T"].assigned_robot, "R")
        self.assertEqual(robot.route, [(1, 0), (2, 0), (1, 0)])
        self.assertEqual(robot.route[-1], (1, 0))
        self.assertIn((2, 0), robot.route)

    def test_assignment_does_not_move_or_advance_anything(self) -> None:
        sim = FleetSimulator(
            GridMap(4, 2, frozenset({(1, 1)})),
            [Robot("R-1", (0, 0)), Robot("R-2", (3, 1))],
            [Task("T-1", (1, 0), (3, 0)), Task("T-2", (2, 1), (0, 1))],
        )
        sim.assign_tasks()
        self.assertEqual(sim.tick, 0)
        self.assertEqual(sim.replay, [])
        self.assertEqual(sim.map_change_history(), [])
        for robot in sim.robots.values():
            self.assertEqual(robot.distance_travelled, 0)
        self.assertEqual(sim.robots["R-1"].position, (0, 0))
        self.assertEqual(sim.robots["R-2"].position, (3, 1))
        for task in sim.tasks.values():
            self.assertFalse(task.picked_up)
            self.assertFalse(task.completed)

    def test_assignment_on_pickup_cell_does_not_register_pickup(self) -> None:
        sim = FleetSimulator(
            GridMap(3, 1),
            [Robot("R", (0, 0))],
            [Task("T", (0, 0), (2, 0))],
        )
        sim.assign_tasks()
        task = sim.tasks["T"]
        robot = sim.robots["R"]
        self.assertEqual(task.assigned_robot, "R")
        # The pickup leg is empty; only the delivery leg remains.
        self.assertEqual(robot.route, [(1, 0), (2, 0)])
        # Assigning alone never collects the goods or moves the robot.
        self.assertFalse(task.picked_up)
        self.assertFalse(task.completed)
        self.assertEqual(robot.position, (0, 0))
        self.assertEqual(robot.distance_travelled, 0)
        self.assertEqual(sim.tick, 0)
        self.assertEqual(sim.replay, [])


class OrderingAndTieBreakTests(unittest.TestCase):
    def test_tasks_are_processed_in_task_id_string_order(self) -> None:
        # "T-10" sorts before "T-9" as strings; both prefer R-a, so the
        # string-smaller task must claim R-a and the other must fall back
        # to the farther R-b.
        sim = FleetSimulator(
            GridMap(6, 1),
            [Robot("R-a", (0, 0)), Robot("R-b", (5, 0))],
            [
                Task("T-9", (2, 0), (3, 0)),
                Task("T-10", (1, 0), (2, 0)),
            ],
        )
        sim.assign_tasks()
        self.assertEqual(sim.tasks["T-10"].assigned_robot, "R-a")
        self.assertEqual(sim.tasks["T-9"].assigned_robot, "R-b")

    def test_robot_used_by_earlier_task_is_unavailable_for_later_ones(self) -> None:
        sim = FleetSimulator(
            GridMap(6, 1),
            [Robot("R-a", (0, 0)), Robot("R-b", (5, 0))],
            [
                Task("T-1", (1, 0), (2, 0)),
                Task("T-2", (2, 0), (3, 0)),
            ],
        )
        sim.assign_tasks()
        # R-a is closest to both pickups but can only take the first task.
        self.assertEqual(sim.tasks["T-1"].assigned_robot, "R-a")
        self.assertEqual(sim.tasks["T-2"].assigned_robot, "R-b")
        self.assertEqual(sim.robots["R-b"].route, [(4, 0), (3, 0), (2, 0), (3, 0)])

    def test_equal_route_length_tie_breaks_by_robot_id_string(self) -> None:
        # Both robots are equally far; "R-10" < "R-2" in string order even
        # though 2 < 10 numerically.
        sim = FleetSimulator(
            GridMap(5, 2),
            [Robot("R-2", (0, 0)), Robot("R-10", (4, 0))],
            [Task("T", (2, 0), (2, 1))],
        )
        sim.assign_tasks()
        self.assertEqual(sim.tasks["T"].assigned_robot, "R-10")
        self.assertIsNone(sim.robots["R-2"].task_id)

    def test_entity_order_does_not_change_the_outcome(self) -> None:
        obstacles = frozenset({(3, 0), (3, 1)})
        robots = [
            Robot("R-1", (0, 0)),
            Robot("R-2", (6, 2)),
            Robot("R-3", (0, 2)),
        ]
        tasks = [
            Task("T-a", (1, 1), (2, 2)),
            Task("T-b", (5, 1), (6, 0)),
            Task("T-c", (4, 2), (5, 2)),
        ]
        forward = FleetSimulator(GridMap(7, 3, obstacles), robots, tasks)
        shuffled = FleetSimulator(
            GridMap(7, 3, obstacles),
            [robots[2], robots[0], robots[1]],
            [tasks[1], tasks[2], tasks[0]],
        )
        forward.assign_tasks()
        shuffled.assign_tasks()
        self.assertEqual(assign_result(forward), assign_result(shuffled))
        self.assertEqual(len(assign_result(forward)), 3)


class UnassignableTaskTests(unittest.TestCase):
    def enclosed_corner_grid(self) -> GridMap:
        # (3, 2) is cut off: its only neighbors (2, 2) and (3, 1) are walls.
        return GridMap(4, 3, frozenset({(2, 2), (3, 1)}))

    def test_no_idle_robot_leaves_task_pending(self) -> None:
        sim = FleetSimulator(
            GridMap(4, 1),
            [Robot("R", (0, 0))],
            [Task("T-old", (1, 0), (3, 0)), Task("T-new", (1, 0), (2, 0))],
        )
        sim.robots["R"].task_id = "T-old"
        sim.tasks["T-old"].assigned_robot = "R"
        sim.robots["R"].route = [(1, 0), (2, 0), (3, 0)]
        sim.assign_tasks()
        self.assertIsNone(sim.tasks["T-new"].assigned_robot)
        self.assertEqual(sim.robots["R"].task_id, "T-old")
        self.assertEqual(sim.robots["R"].route, [(1, 0), (2, 0), (3, 0)])

    def test_unreachable_pickup_stays_unassigned(self) -> None:
        sim = FleetSimulator(
            self.enclosed_corner_grid(),
            [Robot("R", (0, 0))],
            [Task("T", (3, 2), (0, 2))],
        )
        sim.assign_tasks()
        task = sim.tasks["T"]
        robot = sim.robots["R"]
        self.assertIsNone(task.assigned_robot)
        self.assertIsNone(robot.task_id)
        self.assertTrue(robot.idle)
        self.assertEqual(robot.route, [])

    def test_unreachable_dropoff_stays_unassigned(self) -> None:
        # The pickup is reachable; only the delivery leg is impossible.
        sim = FleetSimulator(
            self.enclosed_corner_grid(),
            [Robot("R", (0, 0))],
            [Task("T", (1, 0), (3, 2))],
        )
        sim.assign_tasks()
        self.assertIsNone(sim.tasks["T"].assigned_robot)
        self.assertIsNone(sim.robots["R"].task_id)
        self.assertEqual(sim.robots["R"].route, [])
        # Same pickup with a reachable dropoff assigns fine, proving the
        # dropoff leg is what fails above.
        reachable = FleetSimulator(
            self.enclosed_corner_grid(),
            [Robot("R", (0, 0))],
            [Task("T", (1, 0), (0, 2))],
        )
        reachable.assign_tasks()
        self.assertEqual(reachable.tasks["T"].assigned_robot, "R")

    def test_unreachable_task_does_not_block_later_tasks(self) -> None:
        sim = FleetSimulator(
            self.enclosed_corner_grid(),
            [Robot("R", (0, 0))],
            [
                Task("T-1-blocked", (3, 2), (0, 2)),
                Task("T-2-ok", (1, 0), (0, 2)),
            ],
        )
        sim.assign_tasks()
        self.assertIsNone(sim.tasks["T-1-blocked"].assigned_robot)
        self.assertEqual(sim.tasks["T-2-ok"].assigned_robot, "R")
        self.assertEqual(sim.robots["R"].task_id, "T-2-ok")

    def test_failed_assignment_leaves_no_partial_state(self) -> None:
        sim = FleetSimulator(
            self.enclosed_corner_grid(),
            [Robot("R-1", (0, 0)), Robot("R-2", (1, 1))],
            [Task("T", (3, 2), (0, 2))],
        )
        sim.assign_tasks()
        self.assertIsNone(sim.tasks["T"].assigned_robot)
        for robot in sim.robots.values():
            self.assertIsNone(robot.task_id)
            self.assertEqual(robot.route, [])
            self.assertEqual(robot.distance_travelled, 0)
        self.assertEqual(sim.tick, 0)
        self.assertEqual(sim.replay, [])

    def test_repeat_assignment_preserves_existing_results(self) -> None:
        sim = FleetSimulator(
            self.enclosed_corner_grid(),
            [Robot("R", (0, 0))],
            [
                Task("T-old", (1, 0), (0, 2)),
                Task("T-blocked", (3, 2), (0, 2)),
            ],
        )
        sim.assign_tasks()
        before = assign_result(sim)
        self.assertEqual(set(before), {"T-old"})
        sim.assign_tasks()
        # A second pass changes nothing: the assigned task keeps its robot
        # and route, the unreachable one stays pending, and no time passes.
        self.assertEqual(assign_result(sim), before)
        self.assertIsNone(sim.tasks["T-blocked"].assigned_robot)
        self.assertEqual(sim.tick, 0)
        self.assertEqual(sim.replay, [])


if __name__ == "__main__":
    unittest.main()
