"""Regression coverage for automatic yielding against pickup timing.

The older yielding regressions start from robots that already carry their
goods. These cases pin the actual pickup leg instead: the robot is still on
its way to the pickup point when another robot blocks it, sidesteps around the
blocker on a small map that offers a safe side road, and only then collects and
delivers. The rules under test:

* a detour may change the driving route, but it can never skip the pickup;
* passing the dropoff cell before the goods are collected is a pass-through,
  not a delivery;
* a robot starting a tick on the pickup point collects first and only then
  sidesteps -- it must not leave as an empty car or replan a fresh pickup trip;
* the task keeps its original robot for the whole detour;
* every move is exactly one orthogonal cell, with the replay and mileage
  matching the moves that really happened;
* when no safe side road exists the robot waits in place: time advances, no
  mileage is added, cargo/ownership stay unchanged and the wait is reported as
  traffic rather than map unreachability.
"""

import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task, shortest_path


# 5 wide, 3 tall. The blocked cells leave row 2 as the only corridor that
# bends from the left side of row 1 to its right end:
#   y=0:  # . # . .
#   y=1:  S B . . P
#   y=2:  . . D . .
# S is the carrier's start, B the head-on blocker, P=(4,1) the pickup and
# D=(2,2) the dropoff -- which sits right on the detour corridor, so the robot
# physically stands on the dropoff on its way to the pickup.
DETOUR_OBSTACLES = frozenset({(0, 0), (2, 0), (3, 1)})
DETOUR_PICKUP = (4, 1)
DETOUR_DROPOFF = (2, 2)
DETOUR_START = (0, 1)


def detour_grid() -> GridMap:
    return GridMap(5, 3, DETOUR_OBSTACLES)


def detour_route() -> list[tuple[int, int]]:
    leg1 = shortest_path(detour_grid(), DETOUR_START, DETOUR_PICKUP)
    leg2 = shortest_path(detour_grid(), DETOUR_PICKUP, DETOUR_DROPOFF)
    return leg1 + leg2


def detour_simulator(carrier_idle: bool = False) -> FleetSimulator:
    route = detour_route()
    carrier = (
        Robot("R-1", DETOUR_START)
        if carrier_idle
        else Robot("R-1", DETOUR_START, route=route, task_id="T-1")
    )
    # R-2 drives head-on into R-1's start cell and finishes at (0,1), so it
    # cannot clear the way by simply driving on during the first tick.
    robots = [
        carrier,
        Robot("R-2", (1, 1), route=[(0, 1)], task_id="T-2"),
    ]
    tasks = [
        Task(
            "T-1",
            DETOUR_PICKUP,
            DETOUR_DROPOFF,
            assigned_robot=None if carrier_idle else "R-1",
            picked_up=False,
        ),
        Task("T-2", (1, 1), (0, 1), assigned_robot="R-2", picked_up=True),
    ]
    return FleetSimulator(detour_grid(), robots, tasks)


def tick_frames(simulator: FleetSimulator) -> list[dict[str, object]]:
    return [
        frame for frame in simulator.replay if frame.get("type") == "tick"
    ]


def assert_physics_and_bookkeeping(
    testcase: unittest.TestCase, simulator: FleetSimulator
) -> None:
    """One cell per move, no overlaps/swaps, replay and mileage match reality."""
    frames = tick_frames(simulator)
    moves_per_robot = {robot_id: 0 for robot_id in simulator.robots}
    previous: dict[str, tuple[int, int]] | None = None
    for frame in frames:
        positions = {
            robot_id: tuple(cell) for robot_id, cell in frame["robots"].items()
        }
        # Every robot stands on a traversable cell, none share one.
        for position in positions.values():
            testcase.assertTrue(simulator.grid.traversable(position))
        testcase.assertEqual(len(positions), len(set(positions.values())))
        moved = frame["moved"]
        testcase.assertEqual(len(moved), len(set(moved)))
        if previous is not None:
            actually_moved: set[str] = set()
            for robot_id, position in positions.items():
                distance = (
                    abs(position[0] - previous[robot_id][0])
                    + abs(position[1] - previous[robot_id][1])
                )
                # Exactly one orthogonal step at most -- never two in one tick.
                testcase.assertLessEqual(distance, 1)
                if distance == 1:
                    actually_moved.add(robot_id)
            # No two robots may swap cells within one tick; following a robot
            # into the cell it just vacated is legal.
            mover_ids = sorted(actually_moved)
            for index, robot_id in enumerate(mover_ids):
                for other_id in mover_ids[index + 1 :]:
                    testcase.assertFalse(
                        positions[robot_id] == previous[other_id]
                        and positions[other_id] == previous[robot_id]
                    )
            testcase.assertEqual(set(moved), actually_moved)
        for robot_id in moved:
            moves_per_robot[robot_id] += 1
        previous = positions
    for robot_id, robot in simulator.robots.items():
        testcase.assertEqual(
            robot.distance_travelled,
            moves_per_robot[robot_id],
            f"mileage of {robot_id} disagrees with its replay moves",
        )


class DetourBeforePickupTests(unittest.TestCase):
    def run_detour(self, carrier_idle: bool = False) -> FleetSimulator:
        sim = detour_simulator(carrier_idle=carrier_idle)
        for _ in range(9):
            sim.step()
            if sim.tasks["T-1"].completed:
                break
        return sim

    def test_blocked_route_head_and_detour_geometry(self) -> None:
        sim = detour_simulator()
        route = list(sim.robots["R-1"].route)
        # The first planned waypoint is the cell R-2 occupies at tick start.
        self.assertEqual(route[0], (1, 1))
        self.assertEqual(sim.robots["R-2"].position, (1, 1))
        # Both task points are reachable on the map ignoring the traffic.
        self.assertEqual(shortest_path(sim.grid, DETOUR_START, DETOUR_PICKUP)[-1:], [(4, 1)])
        self.assertEqual(
            shortest_path(sim.grid, DETOUR_PICKUP, DETOUR_DROPOFF)[-1:],
            [DETOUR_DROPOFF],
        )

    def test_robot_actually_sidesteps_on_the_first_tick(self) -> None:
        sim = detour_simulator()
        event = sim.step()
        carrier = sim.robots["R-1"]
        # A genuine side move: one adjacent cell, but not the planned waypoint.
        self.assertIn("R-1", event["moved"])
        self.assertEqual(event["moved"].count("R-1"), 1)
        self.assertEqual(carrier.position, (0, 2))
        self.assertNotEqual(carrier.position, (1, 1))
        self.assertEqual(
            abs(carrier.position[0] - DETOUR_START[0])
            + abs(carrier.position[1] - DETOUR_START[1]),
            1,
        )
        self.assertEqual(carrier.distance_travelled, 1)
        # The detour route still visits the pickup before the dropoff instead
        # of driving straight to the delivery.
        self.assertIn(DETOUR_PICKUP, carrier.route)
        self.assertEqual(carrier.route[-1], DETOUR_DROPOFF)

    def test_not_picked_up_or_completed_before_the_pickup_point(self) -> None:
        sim = detour_simulator()
        for _ in range(5):
            event = sim.step()
            task = sim.tasks["T-1"]
            self.assertFalse(task.picked_up)
            self.assertFalse(task.completed)
            self.assertNotIn("T-1", event["completed"])
            # The task keeps its original robot throughout the whole detour.
            self.assertEqual(task.assigned_robot, "R-1")
            self.assertEqual(sim.robots["R-1"].task_id, "T-1")

    def test_passing_dropoff_before_pickup_is_not_a_delivery(self) -> None:
        sim = detour_simulator()
        # Tick 3 leaves the robot standing on the dropoff cell, goods not yet
        # collected: this is an ordinary pass-through, not a completion.
        for _ in range(3):
            event = sim.step()
        carrier = sim.robots["R-1"]
        task = sim.tasks["T-1"]
        self.assertEqual(carrier.position, DETOUR_DROPOFF)
        self.assertFalse(task.picked_up)
        self.assertFalse(task.completed)
        self.assertEqual(event["completed"], ["T-2"])
        # The route still demands the pickup before the eventual return.
        self.assertIn(DETOUR_PICKUP, carrier.route)

    def test_pickup_is_confirmed_at_the_pickup_step_and_kept(self) -> None:
        sim = detour_simulator()
        for tick in range(1, 10):
            sim.step()
            carrier = sim.robots["R-1"]
            task = sim.tasks["T-1"]
            if tick < 6:
                self.assertNotEqual(carrier.position, DETOUR_PICKUP)
                self.assertFalse(task.picked_up)
            elif tick == 6:
                # The step that reaches the pickup confirms the goods at once.
                self.assertEqual(carrier.position, DETOUR_PICKUP)
                self.assertTrue(task.picked_up)
            else:
                # Cargo stays on board for the rest of the delivery.
                self.assertTrue(task.picked_up)
                self.assertEqual(task.assigned_robot, "R-1")

    def test_completion_recorded_only_after_loaded_arrival(self) -> None:
        sim = detour_simulator()
        first_completed: int | None = None
        for tick in range(1, 10):
            event = sim.step()
            if "T-1" in event["completed"]:
                first_completed = tick
                break
            self.assertFalse(sim.tasks["T-1"].completed)
        # T-1 completes exactly at tick 9, at the dropoff, with no route left.
        self.assertEqual(first_completed, 9)
        carrier = sim.robots["R-1"]
        task = sim.tasks["T-1"]
        self.assertTrue(task.completed)
        self.assertTrue(task.picked_up)
        self.assertEqual(carrier.position, DETOUR_DROPOFF)
        self.assertEqual(carrier.route, [])
        self.assertIsNone(carrier.task_id)
        # A finished task keeps its historical owner even after release.
        self.assertEqual(task.assigned_robot, "R-1")

    def test_detour_completes_the_delivery_within_budget(self) -> None:
        sim = self.run_detour()
        self.assertTrue(sim.tasks["T-1"].completed)
        # Six cells of shortest travel plus the one-cell sidestep detour.
        self.assertEqual(sim.robots["R-1"].distance_travelled, 9)
        self.assertEqual(sim.status()["paused_tasks"], [])

    def test_same_tick_assignment_still_pickups_before_delivering(self) -> None:
        # The carrier is hired during the first tick (assign_tasks) and
        # sidesteps in that very same tick; the pickup rules are unchanged.
        sim = self.run_detour(carrier_idle=True)
        task = sim.tasks["T-1"]
        self.assertEqual(task.assigned_robot, "R-1")
        self.assertTrue(task.picked_up)
        self.assertTrue(task.completed)
        self.assertEqual(sim.robots["R-1"].position, DETOUR_DROPOFF)

    def test_replay_physics_positions_and_mileage_agree(self) -> None:
        sim = self.run_detour()
        assert_physics_and_bookkeeping(self, sim)


class PickupThenSidestepTests(unittest.TestCase):
    """R-1 starts the tick standing on its pickup, next cell blocked by R-2."""

    def build(self, carrier_idle: bool = False) -> FleetSimulator:
        # 5x2 with (3,0) sealed: from the pickup (1,0) the only route to the
        # dropoff (4,0) bends down through row 1, and the busy R-2 blocks the
        # immediate next cell (2,0).
        grid = GridMap(5, 2, frozenset({(3, 0)}))
        route = [(2, 0), (2, 1), (3, 1), (4, 1), (4, 0)]
        carrier = (
            Robot("R-1", (1, 0))
            if carrier_idle
            else Robot("R-1", (1, 0), route=route, task_id="T-1")
        )
        robots = [
            carrier,
            Robot("R-2", (2, 0), route=[(1, 0), (0, 0)], task_id="T-2"),
        ]
        tasks = [
            Task(
                "T-1",
                (1, 0),
                (4, 0),
                assigned_robot=None if carrier_idle else "R-1",
                picked_up=False,
            ),
            Task("T-2", (2, 0), (0, 0), assigned_robot="R-2", picked_up=True),
        ]
        return FleetSimulator(grid, robots, tasks)

    def test_pickup_confirmed_then_exactly_one_sidestep(self) -> None:
        sim = self.build()
        event = sim.step()
        carrier = sim.robots["R-1"]
        task = sim.tasks["T-1"]
        # Goods are collected before the robot leaves the pickup cell.
        self.assertTrue(task.picked_up)
        self.assertEqual(task.assigned_robot, "R-1")
        self.assertEqual(carrier.task_id, "T-1")
        # The sidestep is a single adjacent move (down), never two cells.
        self.assertEqual(event["moved"].count("R-1"), 1)
        self.assertEqual(carrier.position, (1, 1))
        self.assertEqual(carrier.distance_travelled, 1)
        self.assertNotIn("T-1", event["completed"])

    def test_leaving_robot_is_not_empty_and_pickup_trip_not_replanned(self) -> None:
        sim = self.build()
        sim.step()
        carrier = sim.robots["R-1"]
        task = sim.tasks["T-1"]
        # Still bound and loaded: it is not released, reassigned or idle.
        self.assertEqual(carrier.task_id, "T-1")
        self.assertEqual(task.assigned_robot, "R-1")
        self.assertTrue(task.picked_up)
        # A loaded robot routes straight to the dropoff and never revisits the
        # pickup cell: the missed-pickup bookkeeping must not reschedule it.
        self.assertNotIn((1, 0), carrier.route)
        self.assertEqual(carrier.route[-1], (4, 0))
        for _ in range(3):
            sim.step()
        self.assertTrue(task.picked_up)
        self.assertEqual(task.assigned_robot, "R-1")

    def test_completion_recorded_only_after_real_delivery(self) -> None:
        sim = self.build()
        completed_sets = []
        for _ in range(5):
            event = sim.step()
            completed_sets.append(set(event["completed"]))
        # No early completion during pickup, sidestep or travel; T-1 appears
        # for the first time on the final tick at the dropoff.
        for earlier in completed_sets[:-1]:
            self.assertNotIn("T-1", earlier)
        self.assertEqual(completed_sets[-1], {"T-1", "T-2"})
        carrier = sim.robots["R-1"]
        task = sim.tasks["T-1"]
        self.assertTrue(task.completed)
        self.assertTrue(task.picked_up)
        self.assertEqual(carrier.position, (4, 0))
        self.assertEqual(carrier.route, [])

    def test_assignment_on_pickup_cell_collects_before_sidestepping(self) -> None:
        # Hired during this tick while already parked on the pickup: assignment
        # must collect immediately (second _confirm_pickups pass) and only then
        # sidestep -- never drive off as an empty car.
        sim = self.build(carrier_idle=True)
        event = sim.step()
        carrier = sim.robots["R-1"]
        task = sim.tasks["T-1"]
        self.assertEqual(task.assigned_robot, "R-1")
        self.assertTrue(task.picked_up)
        self.assertEqual(carrier.position, (1, 1))
        self.assertEqual(carrier.distance_travelled, 1)
        self.assertNotIn((1, 0), carrier.route)
        self.assertNotIn("T-1", event["completed"])
        for _ in range(4):
            sim.step()
        self.assertTrue(task.completed)
        self.assertEqual(carrier.position, (4, 0))

    def test_replay_physics_positions_and_mileage_agree(self) -> None:
        sim = self.build()
        for _ in range(5):
            sim.step()
        assert_physics_and_bookkeeping(self, sim)


class NoSideRoadBeforePickupTests(unittest.TestCase):
    def build_corridor(self) -> FleetSimulator:
        # Pure 4x1 corridor: no side cell exists. R-1 still has to collect at
        # (2,0) before delivering to (3,0); idle R-2 blocks (1,0).
        return FleetSimulator(
            GridMap(4, 1),
            [
                Robot("R-1", (0, 0), route=[(1, 0), (2, 0), (3, 0)], task_id="T-1"),
                Robot("R-2", (1, 0)),
            ],
            [Task("T-1", (2, 0), (3, 0), assigned_robot="R-1", picked_up=False)],
        )

    def test_task_points_are_reachable_despite_the_traffic(self) -> None:
        sim = self.build_corridor()
        # The jam is purely traffic: ignoring robots, every point is reachable.
        self.assertEqual(shortest_path(sim.grid, (0, 0), (2, 0)), [(1, 0), (2, 0)])
        self.assertEqual(shortest_path(sim.grid, (2, 0), (3, 0)), [(3, 0)])

    def test_waits_in_place_time_advances_without_mileage(self) -> None:
        sim = self.build_corridor()
        for _ in range(5):
            event = sim.step()
            self.assertEqual(event["moved"], [])
            self.assertEqual(sim.robots["R-1"].position, (0, 0))
            self.assertEqual(sim.robots["R-2"].position, (1, 0))
        self.assertEqual(sim.tick, 5)
        self.assertEqual(sim.robots["R-1"].distance_travelled, 0)
        self.assertEqual(sim.robots["R-2"].distance_travelled, 0)
        self.assertEqual(sim.metrics()["distance_total"], 0)

    def test_pre_pickup_cargo_and_ownership_preserved(self) -> None:
        sim = self.build_corridor()
        for _ in range(5):
            sim.step()
            task = sim.tasks["T-1"]
            self.assertFalse(task.picked_up)
            self.assertFalse(task.completed)
            self.assertEqual(task.assigned_robot, "R-1")
            self.assertEqual(sim.robots["R-1"].task_id, "T-1")
            self.assertEqual(sim.robots["R-1"].route, [(1, 0), (2, 0), (3, 0)])
            self.assertNotIn("T-1", sim.replay[-1]["completed"])
        self.assertIsNone(sim.robots["R-2"].task_id)

    def test_wait_is_traffic_not_map_unreachability(self) -> None:
        sim = self.build_corridor()
        for ticks in range(1, 4):
            sim.step()
            self.assertEqual(sim.status()["paused_tasks"], [])
            self.assertEqual(
                sim.status()["traffic_waits"],
                [{"robot_id": "R-1", "blocked_by": ["R-2"], "ticks": ticks}],
            )
            self.assertEqual(
                sim.metrics()["traffic_waits"], sim.status()["traffic_waits"]
            )

    def test_opening_a_side_road_resumes_and_finishes_the_delivery(self) -> None:
        # The side pocket starts sealed; waiting looks identical to the
        # corridor case. Reopening it lets the idle blocker yield, after which
        # the still-empty carrier collects and delivers in order.
        sim = FleetSimulator(
            GridMap(4, 2, frozenset({(1, 1), (2, 1), (3, 1)})),
            [
                Robot("R-1", (0, 0), route=[(1, 0), (2, 0), (3, 0)], task_id="T-1"),
                Robot("R-2", (1, 0)),
            ],
            [Task("T-1", (2, 0), (3, 0), assigned_robot="R-1", picked_up=False)],
        )
        sim.step()
        sim.step()
        self.assertEqual(sim.robots["R-1"].position, (0, 0))
        self.assertEqual(
            sim.status()["traffic_waits"],
            [{"robot_id": "R-1", "blocked_by": ["R-2"], "ticks": 2}],
        )
        sim.modify_obstacles(removed=[(1, 1)])
        event = sim.step()
        # R-2 yields into the pocket; the carrier advances one cell only.
        self.assertEqual(event["moved"], ["R-2", "R-1"])
        self.assertEqual(sim.robots["R-1"].position, (1, 0))
        self.assertFalse(sim.tasks["T-1"].picked_up)
        self.assertEqual(sim.status()["traffic_waits"], [])
        sim.step()
        self.assertEqual(sim.robots["R-1"].position, (2, 0))
        self.assertTrue(sim.tasks["T-1"].picked_up)
        self.assertFalse(sim.tasks["T-1"].completed)
        sim.step()
        self.assertTrue(sim.tasks["T-1"].completed)
        self.assertEqual(sim.robots["R-1"].position, (3, 0))
        self.assertEqual(sim.tasks["T-1"].assigned_robot, "R-1")
        # The yielding blocker stays idle and is never hired by the move.
        self.assertEqual(sim.robots["R-2"].position, (1, 1))
        self.assertIsNone(sim.robots["R-2"].task_id)
        assert_physics_and_bookkeeping(self, sim)


if __name__ == "__main__":
    unittest.main()
