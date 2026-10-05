"""Pending pickup/delivery tasks vs. taskless robots still driving preset routes.

The Python interface allows a robot with no bound task to carry a remaining
route. Such a robot is *driving*, not available: a waiting task may never be
assigned to it merely because ``task_id is None`` -- only a robot with no bound
task **and** an empty remaining route can receive one. These tests pin that
rule for both entry points into assignment -- the public
:meth:`FleetSimulator.assign_tasks` call and the automatic assignment pass
inside :meth:`FleetSimulator.step` -- and pin the timing at which a robot that
has just exhausted its preset route becomes eligible again.

The checks deliberately inspect ownership and remaining routes *per tick*:
a wrong early binding one tick ahead can always be hidden by judging only the
eventually completed task, so "still unbound right after the last preset
waypoint was driven" and "bound only on the next assignment pass, planned from
the then-current position" are asserted separately from final delivery.
"""

import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task, shortest_path


# ---------------------------------------------------------------------------
# Direct assign_tasks() checks: no tick is ever run, so a wrong binding cannot
# hide behind later movement.
# ---------------------------------------------------------------------------


class DirectAssignmentExcludesTasklessRoutedRobotsTests(unittest.TestCase):
    def assert_direct_assignment_pristine(
        self,
        simulator: FleetSimulator,
        before: dict[str, tuple[tuple[int, int], list[tuple[int, int]], int]],
        route_unchanged: tuple[str, ...] = (),
    ) -> None:
        """A direct assignment pass moves nothing and books nothing.

        *before* maps robot id to (position, remaining route, distance) as it
        was before the call. The clock, replay, map history, positions and
        mileage must all be untouched, and no task may be collected or
        completed by being assigned. Only robots named in *route_unchanged*
        have their remaining routes compared; a hired robot legitimately gets
        a freshly planned task route.
        """
        self.assertEqual(simulator.tick, 0)
        self.assertEqual(simulator.replay, [])
        self.assertEqual(simulator.map_change_history(), [])
        self.assertEqual(simulator.paused_tasks, set())
        for robot_id, (position, route, distance) in before.items():
            robot = simulator.robots[robot_id]
            self.assertEqual(robot.position, position)
            self.assertEqual(robot.distance_travelled, distance)
            if robot_id in route_unchanged:
                self.assertEqual(robot.route, route)
        for task in simulator.tasks.values():
            self.assertFalse(task.picked_up)
            self.assertFalse(task.completed)

    def snapshot(
        self, simulator: FleetSimulator
    ) -> dict[str, tuple[tuple[int, int], list[tuple[int, int]], int]]:
        return {
            robot.robot_id: (robot.position, list(robot.route), robot.distance_travelled)
            for robot in simulator.robots.values()
        }

    def assert_mutual_binding(
        self, simulator: FleetSimulator, task_id: str, robot_id: str
    ) -> None:
        self.assertEqual(simulator.tasks[task_id].assigned_robot, robot_id)
        self.assertEqual(simulator.robots[robot_id].task_id, task_id)

    def test_closer_taskless_robot_with_route_is_skipped(self) -> None:
        # One-row corridor: R-routed sits one cell from the pickup while still
        # driving a preset route (away from the task); R-idle is three cells
        # farther and is the only truly available robot.
        grid = GridMap(7, 1)
        pickup, dropoff = (3, 0), (4, 0)
        routed_route = [(1, 0), (0, 0)]
        simulator = FleetSimulator(
            grid,
            [
                Robot("R-routed", (2, 0), route=list(routed_route)),
                Robot("R-idle", (6, 0)),
            ],
            [Task("T-1", pickup, dropoff)],
        )
        # The excluded robot really would offer the shorter feasible route.
        self.assertLess(
            len(
                shortest_path(grid, (2, 0), pickup)
                + shortest_path(grid, pickup, dropoff)
            ),
            len(
                shortest_path(grid, (6, 0), pickup)
                + shortest_path(grid, pickup, dropoff)
            ),
        )
        before = self.snapshot(simulator)

        simulator.assign_tasks()

        self.assert_mutual_binding(simulator, "T-1", "R-idle")
        self.assertEqual(
            simulator.robots["R-idle"].route,
            [(5, 0), (4, 0), (3, 0), (4, 0)],
        )
        excluded = simulator.robots["R-routed"]
        self.assertIsNone(excluded.task_id)
        self.assertEqual(excluded.route, routed_route)
        self.assert_direct_assignment_pristine(
            simulator, before, route_unchanged=("R-routed",)
        )

    def test_equal_distance_earlier_id_taskless_robot_is_still_skipped(self) -> None:
        # Both candidates need the same shortest feasible length, and the
        # taskless routed robot's id ("R-a") would win the id tie break -- it
        # must be removed from the candidate pool first, so R-b gets the task.
        grid = GridMap(5, 3)
        pickup, dropoff = (2, 0), (2, 1)
        simulator = FleetSimulator(
            grid,
            [
                Robot("R-a", (0, 0), route=[(0, 1)]),
                Robot("R-b", (4, 0)),
            ],
            [Task("T-1", pickup, dropoff)],
        )
        route_a = (
            shortest_path(grid, (0, 0), pickup) + shortest_path(grid, pickup, dropoff)
        )
        route_b = (
            shortest_path(grid, (4, 0), pickup) + shortest_path(grid, pickup, dropoff)
        )
        self.assertEqual(len(route_a), len(route_b))
        self.assertLess("R-a", "R-b")

        simulator.assign_tasks()

        self.assert_mutual_binding(simulator, "T-1", "R-b")
        self.assertEqual(
            simulator.robots["R-b"].route, [(3, 0), (2, 0), (2, 1)]
        )
        excluded = simulator.robots["R-a"]
        self.assertIsNone(excluded.task_id)
        self.assertEqual(excluded.route, [(0, 1)])

    def test_taskless_robot_standing_on_pickup_does_not_take_or_collect(self) -> None:
        # The routed robot already occupies the pickup cell; "standing on the
        # goods" with a remaining route must neither bind the task nor confirm
        # the pickup. The idle robot is hired and routed through that cell.
        grid = GridMap(4, 1)
        pickup, dropoff = (2, 0), (3, 0)
        simulator = FleetSimulator(
            grid,
            [
                Robot("R-routed", (2, 0), route=[(3, 0)]),
                Robot("R-idle", (0, 0)),
            ],
            [Task("T-1", pickup, dropoff)],
        )

        simulator.assign_tasks()

        task = simulator.tasks["T-1"]
        self.assert_mutual_binding(simulator, "T-1", "R-idle")
        self.assertFalse(task.picked_up)
        self.assertEqual(
            simulator.robots["R-idle"].route, [(1, 0), (2, 0), (3, 0)]
        )
        excluded = simulator.robots["R-routed"]
        self.assertIsNone(excluded.task_id)
        self.assertEqual(excluded.route, [(3, 0)])

    def test_earlier_task_takes_idle_robot_later_task_never_uses_routed_one(self) -> None:
        # R-routed is closer to both waiting tasks but still drives a preset
        # route. T-a is processed first and hires R-idle; T-b then has no
        # available robot and must keep waiting rather than fall back to
        # R-routed, which keeps its own route and unbound state.
        grid = GridMap(6, 1)
        simulator = FleetSimulator(
            grid,
            [
                Robot("R-routed", (1, 0), route=[(0, 0)]),
                Robot("R-idle", (5, 0)),
            ],
            [
                Task("T-a", (2, 0), (3, 0)),
                Task("T-b", (4, 0), (5, 0)),
            ],
        )

        simulator.assign_tasks()

        self.assert_mutual_binding(simulator, "T-a", "R-idle")
        self.assertEqual(
            simulator.robots["R-idle"].route,
            [(4, 0), (3, 0), (2, 0), (3, 0)],
        )
        waiting = simulator.tasks["T-b"]
        self.assertIsNone(waiting.assigned_robot)
        self.assertFalse(waiting.picked_up)
        self.assertFalse(waiting.completed)
        excluded = simulator.robots["R-routed"]
        self.assertIsNone(excluded.task_id)
        self.assertEqual(excluded.route, [(0, 0)])

    def test_when_routed_robot_is_the_sole_feasible_robot_task_keeps_waiting(self) -> None:
        # A solid wall separates the map: the pickup and dropoff are on the
        # routed robot's side and the idle robot cannot reach them. The task
        # must stay unassigned -- the one robot that could finish it is
        # unavailable while its preset route remains.
        wall = frozenset((2, y) for y in range(5))
        grid = GridMap(5, 5, wall)
        pickup, dropoff = (1, 0), (1, 1)
        simulator = FleetSimulator(
            grid,
            [
                Robot("R-routed", (0, 1), route=[(0, 0)]),
                Robot("R-idle", (4, 0)),
            ],
            [Task("T-1", pickup, dropoff)],
        )

        simulator.assign_tasks()

        task = simulator.tasks["T-1"]
        self.assertIsNone(task.assigned_robot)
        self.assertFalse(task.picked_up)
        self.assertFalse(task.completed)
        self.assertNotIn("T-1", simulator.paused_tasks)
        excluded = simulator.robots["R-routed"]
        self.assertIsNone(excluded.task_id)
        self.assertEqual(excluded.route, [(0, 0)])
        # The infeasible idle candidate is not left holding a phantom route.
        self.assertEqual(simulator.robots["R-idle"].route, [])
        self.assertEqual(simulator.tick, 0)
        self.assertEqual(simulator.replay, [])


# ---------------------------------------------------------------------------
# Automatic assignment inside step()
# ---------------------------------------------------------------------------


class StepAutoAssignmentExcludesTasklessRoutedRobotsTests(unittest.TestCase):
    def test_step_hires_idle_robot_and_routed_robot_drives_its_own_route(self) -> None:
        # Grid 5x3: R-routed is one cell from the pickup but drives a preset
        # route down column 2; R-idle starts three cells away. The step's
        # assignment pass must hire R-idle; R-routed moves along *its own*
        # waypoints, never along the task route, and goods are collected at the
        # pickup before the delivery completes.
        grid = GridMap(5, 3)
        pickup, dropoff = (1, 0), (0, 0)
        simulator = FleetSimulator(
            grid,
            [
                Robot("R-routed", (2, 0), route=[(2, 1), (2, 2)]),
                Robot("R-idle", (4, 0)),
            ],
            [Task("T-1", pickup, dropoff)],
        )

        # Step 1: task binds to R-idle; both robots then move one cell, each on
        # its own business.
        event = simulator.step()
        self.assertEqual(event["moved"], ["R-idle", "R-routed"])
        task = simulator.tasks["T-1"]
        self.assertEqual(task.assigned_robot, "R-idle")
        self.assertEqual(simulator.robots["R-idle"].task_id, "T-1")
        routed = simulator.robots["R-routed"]
        self.assertIsNone(routed.task_id)
        self.assertEqual(routed.position, (2, 1))
        self.assertEqual(routed.route, [(2, 2)])
        self.assertEqual(routed.distance_travelled, 1)
        self.assertEqual(simulator.robots["R-idle"].position, (3, 0))
        self.assertFalse(task.picked_up)

        # Step 2: R-routed finishes its preset route; the task is already bound
        # and stays with R-idle, whose route passes through the now-vacated
        # (2, 0). R-routed is not reassigned anything.
        simulator.step()
        self.assertEqual(routed.position, (2, 2))
        self.assertEqual(routed.route, [])
        self.assertIsNone(routed.task_id)
        self.assertEqual(simulator.robots["R-idle"].position, (2, 0))
        self.assertFalse(task.picked_up)
        self.assertFalse(task.completed)

        # Step 3: R-idle reaches the pickup and only then collects.
        simulator.step()
        self.assertEqual(simulator.robots["R-idle"].position, (1, 0))
        self.assertTrue(task.picked_up)
        self.assertFalse(task.completed)

        # Step 4: delivery one tick after pickup, never before.
        simulator.step()
        self.assertEqual(simulator.robots["R-idle"].position, (0, 0))
        self.assertTrue(task.completed)
        # Every recorded position of R-routed lies on the preset route only.
        routed_frames = [
            tuple(frame["robots"]["R-routed"])
            for frame in simulator.replay
            if frame.get("type") == "tick"
        ]
        self.assertEqual(routed_frames, [(2, 1), (2, 2), (2, 2), (2, 2)])
        self.assertNotIn("T-1", simulator.paused_tasks)

    def test_all_robots_routed_and_blocked_task_keeps_waiting(self) -> None:
        # Two taskless robots, both carrying remaining routes, deadlocked
        # head-on: A needs (2, 0) held by B, B needs (1, 0) held by A. The open
        # side cell (1, 1) must change nothing -- a stopped routing robot has
        # not finished its route -- and with no parked/available robot the
        # pending task simply waits, tick after tick.
        grid = GridMap(3, 2, frozenset({(2, 1)}))
        self.assertTrue(grid.traversable((1, 1)))  # the open side cell
        simulator = FleetSimulator(
            grid,
            [
                Robot("A", (1, 0), route=[(2, 0)]),
                Robot("B", (2, 0), route=[(1, 0)]),
            ],
            [Task("T-1", (0, 0), (0, 1))],
        )

        for tick in range(1, 4):
            event = simulator.step()
            self.assertEqual(event["moved"], [])
            self.assertEqual(simulator.tick, tick)
            task = simulator.tasks["T-1"]
            self.assertIsNone(task.assigned_robot)
            self.assertFalse(task.picked_up)
            self.assertFalse(task.completed)
            self.assertNotIn("T-1", event["completed"])
            # Neither robot may skip, lose or pop a preset waypoint while
            # waiting, and neither may be sidestepped onto the open cell.
            self.assertEqual(simulator.robots["A"].position, (1, 0))
            self.assertEqual(simulator.robots["A"].route, [(2, 0)])
            self.assertEqual(simulator.robots["B"].position, (2, 0))
            self.assertEqual(simulator.robots["B"].route, [(1, 0)])
            self.assertEqual(simulator.robots["A"].distance_travelled, 0)
            self.assertEqual(simulator.robots["B"].distance_travelled, 0)
            # This is traffic waiting, not a map-unreachability pause.
            self.assertEqual(simulator.paused_tasks, set())
        waits = {entry["robot_id"]: entry for entry in simulator.status()["traffic_waits"]}
        self.assertEqual(waits["A"]["blocked_by"], ["B"])
        self.assertEqual(waits["A"]["ticks"], 3)
        self.assertEqual(waits["B"]["blocked_by"], ["A"])
        self.assertEqual(waits["B"]["ticks"], 3)

    def test_chain_of_routed_robots_then_become_eligible_at_real_positions(self) -> None:
        # R-1 starts boxed in behind R-2; both are taskless with preset routes
        # and a task is waiting. They must simply drive (and wait) off their
        # own routes without the task hopping onto either one, and only the
        # next assignment pass after BOTH routes ran out may bind it -- to the
        # robot closest from where it actually stands then.
        grid = GridMap(6, 1)
        pickup, dropoff = (5, 0), (4, 0)
        simulator = FleetSimulator(
            grid,
            [
                Robot("R-1", (0, 0), route=[(1, 0), (2, 0)]),
                Robot("R-2", (1, 0), route=[(2, 0), (3, 0), (4, 0)]),
            ],
            [Task("T-1", pickup, dropoff)],
        )

        # Step 1: R-2 drives on; R-1 waits one tick and consumes nothing.
        event = simulator.step()
        self.assertEqual(event["moved"], ["R-2"])
        self.assertIsNone(simulator.tasks["T-1"].assigned_robot)
        self.assertEqual(simulator.robots["R-1"].position, (0, 0))
        self.assertEqual(simulator.robots["R-1"].route, [(1, 0), (2, 0)])
        self.assertEqual(simulator.robots["R-2"].position, (2, 0))
        self.assertEqual(simulator.robots["R-2"].route, [(3, 0), (4, 0)])

        # Step 2: both advance one waypoint.
        simulator.step()
        self.assertIsNone(simulator.tasks["T-1"].assigned_robot)
        self.assertEqual(simulator.robots["R-1"].position, (1, 0))
        self.assertEqual(simulator.robots["R-1"].route, [(2, 0)])
        self.assertEqual(simulator.robots["R-2"].position, (3, 0))
        self.assertEqual(simulator.robots["R-2"].route, [(4, 0)])

        # Step 3: both reach their LAST preset waypoint -- this movement still
        # belongs to the preset routes, so the task must not be grabbed in the
        # same tick (assignment ran before the moves).
        simulator.step()
        task = simulator.tasks["T-1"]
        self.assertIsNone(task.assigned_robot)
        self.assertIsNone(simulator.robots["R-1"].task_id)
        self.assertIsNone(simulator.robots["R-2"].task_id)
        self.assertEqual(simulator.robots["R-1"].position, (2, 0))
        self.assertEqual(simulator.robots["R-1"].route, [])
        self.assertEqual(simulator.robots["R-2"].position, (4, 0))
        self.assertEqual(simulator.robots["R-2"].route, [])

        # Step 4: eligible at last. R-2 stands one cell from the pickup vs
        # R-1's three, so R-2 is hired and the new route is planned from its
        # real position (4, 0), never from its initial (1, 0).
        event = simulator.step()
        self.assertEqual(task.assigned_robot, "R-2")
        self.assertEqual(simulator.robots["R-2"].task_id, "T-1")
        self.assertIsNone(simulator.robots["R-1"].task_id)
        self.assertEqual(event["moved"], ["R-2"])
        self.assertEqual(simulator.robots["R-2"].position, (5, 0))
        # Assigned route was [(5, 0), (4, 0)]: one cell just driven plus the
        # remaining waypoint. Planning from the initial (1, 0) would instead
        # have produced five cells.
        self.assertEqual(simulator.robots["R-2"].route, [(4, 0)])
        planned_from_current = (
            shortest_path(grid, (4, 0), pickup) + shortest_path(grid, pickup, dropoff)
        )
        self.assertEqual(planned_from_current, [(5, 0), (4, 0)])
        self.assertNotEqual(
            shortest_path(grid, (1, 0), pickup) + shortest_path(grid, pickup, dropoff),
            [(5, 0), (4, 0)],
        )
        self.assertTrue(task.picked_up)
        self.assertFalse(task.completed)

        # Step 5: delivery strictly after pickup.
        simulator.step()
        self.assertTrue(task.completed)
        self.assertEqual(simulator.robots["R-2"].position, (4, 0))
        self.assertEqual(simulator.robots["R-2"].route, [])


# ---------------------------------------------------------------------------
# Eligibility timing around the very last preset waypoint
# ---------------------------------------------------------------------------


class RequalificationAfterLastWaypointTests(unittest.TestCase):
    def test_last_waypoint_step_binds_nothing_next_pass_plans_from_current_cell(
        self,
    ) -> None:
        # A single taskless robot drives a three-waypoint preset route while a
        # task waits. The final waypoint is reached during a step; that move is
        # still part of the preset route. The task binds only on the next
        # assignment pass, planned from the robot's then-current cell, and the
        # robot still has to collect at the pickup before it may deliver.
        grid = GridMap(6, 1)
        pickup, dropoff = (5, 0), (4, 0)
        simulator = FleetSimulator(
            grid,
            [Robot("R", (0, 0), route=[(1, 0), (2, 0), (3, 0)])],
            [Task("T-1", pickup, dropoff)],
        )

        def assert_unbound(position, route, distance, tick) -> None:
            self.assertIsNone(simulator.tasks["T-1"].assigned_robot)
            self.assertIsNone(simulator.robots["R"].task_id)
            robot = simulator.robots["R"]
            self.assertEqual(robot.position, position)
            self.assertEqual(robot.route, route)
            self.assertEqual(robot.distance_travelled, distance)
            self.assertEqual(simulator.tick, tick)
            self.assertFalse(simulator.tasks["T-1"].picked_up)

        simulator.step()  # -> (1, 0)
        assert_unbound((1, 0), [(2, 0), (3, 0)], 1, 1)
        simulator.step()  # -> (2, 0)
        assert_unbound((2, 0), [(3, 0)], 2, 2)

        # The last preset waypoint is reached on this tick: still no binding.
        simulator.step()  # -> (3, 0), route empty
        assert_unbound((3, 0), [], 3, 3)

        # Next assignment pass (inside this step) hires the robot and plans
        # from (3, 0), not from the original (0, 0); the tick then drives the
        # first NEW-route cell only.
        event = simulator.step()
        task = simulator.tasks["T-1"]
        self.assertEqual(task.assigned_robot, "R")
        self.assertEqual(simulator.robots["R"].task_id, "T-1")
        self.assertEqual(event["moved"], ["R"])
        self.assertEqual(simulator.robots["R"].position, (4, 0))
        self.assertEqual(simulator.robots["R"].route, [(5, 0), (4, 0)])
        self.assertEqual(simulator.robots["R"].distance_travelled, 4)
        self.assertFalse(task.picked_up)
        self.assertFalse(task.completed)

        # Pickup tick...
        simulator.step()
        self.assertEqual(simulator.robots["R"].position, (5, 0))
        self.assertTrue(task.picked_up)
        self.assertFalse(task.completed)
        self.assertEqual(simulator.robots["R"].route, [(4, 0)])

        # ...delivery only one tick later.
        simulator.step()
        self.assertEqual(simulator.robots["R"].position, (4, 0))
        self.assertTrue(task.completed)
        self.assertEqual(simulator.robots["R"].route, [])
        self.assertEqual(simulator.robots["R"].distance_travelled, 6)
        # The task never appears as completed before the final tick frame.
        completed_by_frame = [
            list(frame["completed"])
            for frame in simulator.replay
            if frame.get("type") == "tick"
        ]
        self.assertEqual(
            completed_by_frame,
            [[], [], [], [], [], ["T-1"]],
        )


if __name__ == "__main__":
    unittest.main()
