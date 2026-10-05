"""Regression: a pending task must never be assigned to a taskless robot that
still carries a remaining preset route.

The Python interface permits a robot with no bound task to keep walking a
remaining route. "No task" must not be read as "free to take a task": only a
robot with no bound task *and* no remaining route is selectable. These tests
pin that rule for both entry points -- the public ``assign_tasks()`` call and
the automatic assignment pass inside ``step()`` -- including:

* a routed robot that is closer to the pickup, or whose id would win an
  equal-distance tie, being skipped in favour of a truly idle robot, with its
  unbound state and unconsumed route left untouched;
* a direct assignment moving nothing: no waypoint consumed, no pickup
  confirmed early, no clock, mileage or replay change;
* a routed robot stopped by a vehicle ahead (even beside an open side cell)
  keeping its route while a task waits, and a fleet in which every robot
  still carries a route leaving the task unassigned, not dropped or finished;
* re-qualification timing: a robot that reaches its last waypoint during a
  ``step()`` is still driving the old route on that tick; the waiting task is
  bound only on the next assignment pass and routed from the robot's actual
  position then.

The checks assert the binding on every tick, so an assignment taken one tick
too early fails even when the task would eventually be completed anyway.
"""

import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


def mutual_bindings(
    testcase: unittest.TestCase, simulator: FleetSimulator
) -> None:
    """Every unfinished bound task and its robot reference only each other.

    Completed tasks keep their historical owner while the robot is already
    released, mirroring ``test_task_assignment`` and the ownership contract.
    """
    for task in simulator.tasks.values():
        if task.assigned_robot is None or task.completed:
            continue
        robot = simulator.robots[task.assigned_robot]
        testcase.assertEqual(robot.task_id, task.task_id)
    for robot in simulator.robots.values():
        if robot.task_id is not None:
            testcase.assertEqual(
                simulator.tasks[robot.task_id].assigned_robot, robot.robot_id
            )


# ---------------------------------------------------------------------------
# Direct assign_tasks() calls
# ---------------------------------------------------------------------------


class DirectAssignmentSkipsRoutedTasklessRobotTests(unittest.TestCase):
    def test_closer_routed_robot_is_skipped_for_farther_idle_robot(self) -> None:
        # R-B drives a preset route east on the top row and ends exactly on the
        # pickup cell: only 3 steps from pickup plus 2 to the dropoff. R-F,
        # truly idle on the bottom row, must loop around (6 steps just to reach
        # the pickup). R-B must still be skipped while its route lasts.
        grid = GridMap(5, 3)
        pickup, dropoff = (4, 0), (4, 2)
        sim = FleetSimulator(
            grid,
            [
                Robot("R-B", (1, 0), route=[(2, 0), (3, 0), (4, 0)]),
                Robot("R-F", (0, 2)),
            ],
            [Task("T-1", pickup, dropoff)],
        )
        sim.assign_tasks()

        task = sim.tasks["T-1"]
        self.assertEqual(task.assigned_robot, "R-F")
        self.assertEqual(sim.robots["R-F"].task_id, "T-1")
        routed = sim.robots["R-B"]
        self.assertIsNone(routed.task_id)
        # The excluded robot keeps its state, position and every waypoint.
        self.assertEqual(routed.position, (1, 0))
        self.assertEqual(routed.route, [(2, 0), (3, 0), (4, 0)])
        self.assertEqual(routed.distance_travelled, 0)
        # The winner is planned from its own real position via pickup then
        # dropoff -- the BFS tie break runs up the left edge and back.
        self.assertEqual(
            sim.robots["R-F"].route,
            [
                (0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
                (4, 1), (4, 2),
            ],
        )
        self.assertFalse(task.picked_up)
        self.assertFalse(task.completed)
        mutual_bindings(self, sim)

    def test_direct_assignment_has_no_clock_movement_pickup_or_replay_effect(self) -> None:
        grid = GridMap(5, 3)
        sim = FleetSimulator(
            grid,
            [
                Robot("R-B", (1, 0), route=[(2, 0), (3, 0), (4, 0)]),
                Robot("R-F", (0, 2)),
            ],
            [Task("T-1", (4, 0), (4, 2))],
        )
        sim.assign_tasks()
        # Pure assignment: it drives nothing and records nothing.
        self.assertEqual(sim.tick, 0)
        self.assertEqual(sim.replay, [])
        self.assertEqual(sim.map_change_history(), [])
        self.assertEqual(sim.robots["R-B"].position, (1, 0))
        self.assertEqual(sim.robots["R-F"].position, (0, 2))
        self.assertEqual(sim.robots["R-B"].distance_travelled, 0)
        self.assertEqual(sim.robots["R-F"].distance_travelled, 0)
        # No pickup is confirmed just because the winning route later visits it.
        self.assertFalse(sim.tasks["T-1"].picked_up)
        # Repeating the pass leaves the routed robot and the binding untouched.
        sim.assign_tasks()
        self.assertEqual(sim.robots["R-B"].route, [(2, 0), (3, 0), (4, 0)])
        self.assertIsNone(sim.robots["R-B"].task_id)
        self.assertEqual(sim.tasks["T-1"].assigned_robot, "R-F")
        self.assertEqual(sim.tick, 0)
        self.assertEqual(sim.replay, [])

    def test_equal_distance_earlier_id_routed_robot_loses_to_idle_robot(self) -> None:
        # Both robots are exactly 2 steps from the pickup; "R-A" would win the
        # id tie break if it were a candidate. Its route disqualifies it, so
        # the task goes to "R-F" instead.
        grid = GridMap(5, 1)
        sim = FleetSimulator(
            grid,
            [
                Robot("R-A", (0, 0), route=[(1, 0)]),
                Robot("R-F", (4, 0)),
            ],
            [Task("T-1", (2, 0), (4, 0))],
        )
        sim.assign_tasks()
        self.assertEqual(sim.tasks["T-1"].assigned_robot, "R-F")
        self.assertEqual(sim.robots["R-F"].task_id, "T-1")
        routed = sim.robots["R-A"]
        self.assertIsNone(routed.task_id)
        self.assertEqual(routed.route, [(1, 0)])
        self.assertEqual(routed.position, (0, 0))
        mutual_bindings(self, sim)

    def test_two_pending_tasks_skip_the_routed_robot_in_task_id_order(self) -> None:
        # T-a would suit the routed R-B best and is processed first; it takes
        # the only idle robot R-F, leaving T-b waiting once no idle robot
        # remains. R-B is never borrowed by either task.
        grid = GridMap(6, 2)
        sim = FleetSimulator(
            grid,
            [
                Robot("R-B", (1, 0), route=[(2, 0), (3, 0)]),
                Robot("R-F", (5, 1)),
            ],
            [
                Task("T-a", (3, 0), (0, 1)),
                Task("T-b", (4, 0), (0, 0)),
            ],
        )
        sim.assign_tasks()
        self.assertEqual(sim.tasks["T-a"].assigned_robot, "R-F")
        self.assertIsNone(sim.tasks["T-b"].assigned_robot)
        self.assertFalse(sim.tasks["T-b"].completed)
        routed = sim.robots["R-B"]
        self.assertIsNone(routed.task_id)
        self.assertEqual(routed.route, [(2, 0), (3, 0)])
        self.assertEqual(routed.position, (1, 0))
        self.assertNotIn("T-b", sim.paused_tasks)
        mutual_bindings(self, sim)

    def test_blocked_routed_robot_keeps_route_when_parked_blocker_takes_task(self) -> None:
        # R-B's one remaining waypoint (2, 0) is held by parked R-P; (1, 1)
        # beside R-B is open. The task pickup is closer to R-B, but R-B must be
        # skipped; R-P (genuinely idle) gets the task even though its planned
        # route leads through R-B's cell -- planning ignores traffic, and
        # nothing moves during a direct assignment.
        grid = GridMap(3, 2, frozenset({(2, 1)}))
        sim = FleetSimulator(
            grid,
            [
                Robot("R-B", (1, 0), route=[(2, 0)]),
                Robot("R-P", (2, 0)),
            ],
            [Task("T-1", (0, 0), (0, 1))],
        )
        sim.assign_tasks()
        self.assertEqual(sim.tasks["T-1"].assigned_robot, "R-P")
        self.assertEqual(sim.robots["R-P"].task_id, "T-1")
        self.assertEqual(sim.robots["R-P"].route, [(1, 0), (0, 0), (0, 1)])
        routed = sim.robots["R-B"]
        self.assertIsNone(routed.task_id)
        self.assertEqual(routed.position, (1, 0))
        self.assertEqual(routed.route, [(2, 0)])
        self.assertEqual(routed.distance_travelled, 0)
        self.assertEqual(sim.tick, 0)
        self.assertEqual(sim.replay, [])
        mutual_bindings(self, sim)


# ---------------------------------------------------------------------------
# Automatic assignment inside step()
# ---------------------------------------------------------------------------


class StepAssignmentSkipsRoutedTasklessRobotTests(unittest.TestCase):
    def test_idle_robot_gets_task_while_routed_robot_walks_its_own_route(self) -> None:
        # R-B starts on the top row with a 2-cell preset route; the pickup is
        # on that same row, so R-B would be the cheaper candidate (6 planned
        # cells vs R-F's 8 around the edge). Its remaining route disqualifies
        # it nonetheless: on the first tick the task binds to the truly idle
        # R-F, while R-B consumes its own preset waypoints and never becomes
        # the carrier.
        grid = GridMap(7, 3)
        sim = FleetSimulator(
            grid,
            [
                Robot("R-B", (0, 0), route=[(1, 0), (2, 0)]),
                Robot("R-F", (0, 2)),
            ],
            [Task("T-1", (4, 0), (6, 0))],
        )

        event = sim.step()
        self.assertEqual(sim.tasks["T-1"].assigned_robot, "R-F")
        self.assertIsNone(sim.robots["R-B"].task_id)
        # R-B advanced exactly one cell of its OWN preset route, not a task
        # route, and no pickup was confirmed merely because a task was bound.
        self.assertEqual(sim.robots["R-B"].position, (1, 0))
        self.assertEqual(sim.robots["R-B"].route, [(2, 0)])
        self.assertEqual(event["moved"], ["R-B", "R-F"])
        self.assertFalse(sim.tasks["T-1"].picked_up)

        event = sim.step()
        # R-B reaches its last preset waypoint this tick, still unbound.
        self.assertEqual(sim.robots["R-B"].position, (2, 0))
        self.assertEqual(sim.robots["R-B"].route, [])
        self.assertIsNone(sim.robots["R-B"].task_id)

        # Run to completion: the owner never changes and R-F collects before
        # it delivers. (Once parked, R-B may later yield a cell like any idle
        # robot -- that is unrelated to carrying the task.)
        for _ in range(6):
            sim.step()
        task = sim.tasks["T-1"]
        self.assertEqual(task.assigned_robot, "R-F")
        self.assertIsNone(sim.robots["R-B"].task_id)
        self.assertTrue(task.picked_up)
        self.assertTrue(task.completed)
        self.assertEqual(sim.robots["R-F"].position, (6, 0))
        # R-B drove its two preset cells first and only ever moved one
        # orthogonal cell per tick.
        self.assertEqual(sim.robots["R-B"].distance_travelled, 3)
        previous = (0, 0)
        for frame in sim.replay:
            if frame.get("type") != "tick":
                continue
            cell = tuple(frame["robots"]["R-B"])
            self.assertLessEqual(
                abs(cell[0] - previous[0]) + abs(cell[1] - previous[1]), 1
            )
            previous = cell

    def test_all_robots_routed_head_on_keeps_the_task_waiting(self) -> None:
        # One-wide corridor: R-B heading east and R-C heading west meet head on
        # after the first tick and can neither pass nor sidestep. Every robot
        # still carries a route, so the pending task must remain unassigned
        # tick after tick -- never dropped, never completed -- and neither
        # robot may skip or lose a waypoint while it waits.
        grid = GridMap(4, 1)
        sim = FleetSimulator(
            grid,
            [
                Robot("R-B", (1, 0), route=[(2, 0), (3, 0)]),
                Robot("R-C", (3, 0), route=[(2, 0), (1, 0)]),
            ],
            [Task("T-1", (0, 0), (3, 0))],
        )

        event = sim.step()
        # R-B advances along its original route; R-C waits.
        self.assertEqual(event["moved"], ["R-B"])
        self.assertEqual(sim.robots["R-B"].position, (2, 0))
        self.assertEqual(sim.robots["R-B"].route, [(3, 0)])
        self.assertEqual(sim.robots["R-C"].position, (3, 0))
        self.assertEqual(sim.robots["R-C"].route, [(2, 0), (1, 0)])
        self.assertIsNone(sim.tasks["T-1"].assigned_robot)

        for tick in range(2, 6):
            event = sim.step()
            self.assertEqual(event["moved"], [], tick)
            self.assertEqual(sim.robots["R-B"].position, (2, 0))
            self.assertEqual(sim.robots["R-B"].route, [(3, 0)])
            self.assertEqual(sim.robots["R-C"].position, (3, 0))
            # R-C kept both unconsumed waypoints: waiting skipped nothing.
            self.assertEqual(sim.robots["R-C"].route, [(2, 0), (1, 0)])
            task = sim.tasks["T-1"]
            self.assertIsNone(task.assigned_robot, tick)
            self.assertFalse(task.picked_up, tick)
            self.assertFalse(task.completed, tick)
            self.assertIn("T-1", sim.tasks)
        # Waiting still advances the clock and records frames, but adds no
        # mileage once the block forms.
        self.assertEqual(sim.tick, 5)
        self.assertEqual(len(sim.replay), 5)
        self.assertEqual(sim.robots["R-B"].distance_travelled, 1)
        self.assertEqual(sim.robots["R-C"].distance_travelled, 0)

    def test_open_side_cell_never_reassigns_or_displaces_a_blocked_routed_robot(self) -> None:
        # R-P parked at (2, 0) blocks taskless-routed R-B; (1, 1) right beside
        # R-B is open and (2, 1) is walled off. On the first tick R-P receives
        # the task but R-B blocks its planned route. R-B must stay on (1, 0)
        # with its single waypoint forever: it may neither sidestep into the
        # open cell nor have the waiting task reassigned onto it.
        grid = GridMap(3, 2, frozenset({(2, 1)}))
        sim = FleetSimulator(
            grid,
            [
                Robot("R-B", (1, 0), route=[(2, 0)]),
                Robot("R-P", (2, 0)),
            ],
            [Task("T-1", (0, 0), (0, 1))],
        )
        for tick in range(1, 4):
            event = sim.step()
            self.assertEqual(event["moved"], [], tick)
            routed = sim.robots["R-B"]
            self.assertEqual(routed.position, (1, 0), tick)
            self.assertEqual(routed.route, [(2, 0)], tick)
            self.assertEqual(routed.distance_travelled, 0, tick)
            self.assertIsNone(routed.task_id, tick)
            for frame in sim.replay:
                if frame.get("type") == "tick":
                    self.assertEqual(frame["robots"]["R-B"], [1, 0])
            # The task stays with the parked robot it was assigned to; nothing
            # is collected or finished in the jam.
            self.assertEqual(sim.tasks["T-1"].assigned_robot, "R-P", tick)
            self.assertFalse(sim.tasks["T-1"].picked_up, tick)
            self.assertFalse(sim.tasks["T-1"].completed, tick)

    def test_task_binds_only_after_route_finishes_and_from_real_position(self) -> None:
        # R-B starts at (0, 0) with a preset route ending at (3, 0); the task
        # pickup is (5, 0). While the route lasts R-B cannot be chosen, and
        # the tick on which it rolls onto its final waypoint still belongs to
        # the old route. Binding happens on the next assignment pass, routed
        # from (3, 0) -- never from the original (0, 0) -- and the goods are
        # collected before delivery.
        grid = GridMap(6, 2)
        sim = FleetSimulator(
            grid,
            [Robot("R-B", (0, 0), route=[(1, 0), (2, 0), (3, 0)])],
            [Task("T-1", (5, 0), (5, 1))],
        )

        sim.step()
        self.assertIsNone(sim.tasks["T-1"].assigned_robot)
        self.assertEqual(sim.robots["R-B"].route, [(2, 0), (3, 0)])

        sim.step()
        # One tick before the last waypoint is reached: still not hired.
        self.assertIsNone(sim.tasks["T-1"].assigned_robot)
        self.assertEqual(sim.robots["R-B"].position, (2, 0))
        self.assertEqual(sim.robots["R-B"].route, [(3, 0)])

        sim.step()
        # The final waypoint is reached THIS tick; that move still belongs to
        # the preset route, so no binding may have happened yet.
        self.assertIsNone(sim.tasks["T-1"].assigned_robot)
        self.assertFalse(sim.tasks["T-1"].picked_up)
        self.assertFalse(sim.tasks["T-1"].completed)
        robot = sim.robots["R-B"]
        self.assertEqual(robot.position, (3, 0))
        self.assertEqual(robot.route, [])
        self.assertIsNone(robot.task_id)

        sim.step()
        # Next assignment pass: eligible at last. The tick moved one cell and
        # two waypoints remain, so the plan created at (3, 0) had length 3 --
        # a plan mistakenly rooted at the original (0, 0) would have length 6.
        task = sim.tasks["T-1"]
        self.assertEqual(task.assigned_robot, "R-B")
        self.assertEqual(robot.task_id, "T-1")
        self.assertEqual(robot.position, (4, 0))
        self.assertEqual(robot.route, [(5, 0), (5, 1)])
        self.assertEqual(1 + len(robot.route), 3)
        # The robot is not yet at the pickup when bound, so nothing is
        # collected during the assignment itself.
        self.assertFalse(task.picked_up)

        sim.step()
        # Reaching the pickup collects the goods, but delivery is a later tick.
        self.assertEqual(robot.position, (5, 0))
        self.assertTrue(sim.tasks["T-1"].picked_up)
        self.assertFalse(sim.tasks["T-1"].completed)
        sim.step()
        # Pickup strictly precedes delivery.
        self.assertEqual(robot.position, (5, 1))
        self.assertTrue(sim.tasks["T-1"].completed)
        self.assertEqual(robot.distance_travelled, 6)
        mutual_bindings(self, sim)

    def test_early_binding_is_caught_per_tick_not_hidden_by_completion(self) -> None:
        # Same layout as the timing test, but the per-tick assertions are
        # stated from a fresh simulator here to document intent: judging only
        # the final state (task eventually completed) cannot distinguish
        # on-time hiring from hiring one pass early.
        grid = GridMap(6, 2)
        sim = FleetSimulator(
            grid,
            [Robot("R-B", (0, 0), route=[(1, 0), (2, 0), (3, 0)])],
            [Task("T-1", (5, 0), (5, 1))],
        )
        unassigned_for_ticks = 0
        for _ in range(3):
            sim.step()
            if sim.tasks["T-1"].assigned_robot is None:
                unassigned_for_ticks += 1
        # Arrival tick included: the task must wait through all three ticks
        # in which the preset route (partly) existed.
        self.assertEqual(unassigned_for_ticks, 3)
        self.assertEqual(sim.robots["R-B"].route, [])
        self.assertIsNone(sim.tasks["T-1"].assigned_robot)
        sim.step()
        self.assertEqual(sim.tasks["T-1"].assigned_robot, "R-B")


if __name__ == "__main__":
    unittest.main()
