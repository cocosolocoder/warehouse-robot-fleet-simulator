"""Regression coverage for yielding while a task's goods are not yet collected.

The existing yielding tests mostly start from robots that already carry their
goods. These scenarios cover the pickup-timing rules observed purely through
the public API while traffic avoidance changes the driven route:

* a detour may change the path, but it may not skip the pickup, and passing
  the dropoff early is not a delivery;
* a robot that starts a tick standing on its uncollected pickup confirms the
  pickup before it sidesteps away, keeps the goods and never gets a second
  pickup trip scheduled;
* with no safe side road the robot simply waits -- the task points stay
  reachable on the map, so this is traffic, not a map-unreachability pause;
* in every scenario robots move one orthogonal cell per tick, never share a
  cell, never swap cells within one tick, and the replay's positions, moved
  list, mileages and completion records match what physically happened.
"""

import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task, shortest_path


# ---------------------------------------------------------------------------
# Shared observation helpers (public API only)
# ---------------------------------------------------------------------------


def tick_frames(simulator: FleetSimulator) -> list[dict]:
    return [frame for frame in simulator.replay if frame.get("type") == "tick"]


def assert_frame_physically_valid(
    testcase: unittest.TestCase,
    simulator: FleetSimulator,
    previous: dict[str, tuple[int, int]] | None,
    frame: dict,
) -> dict[str, tuple[int, int]]:
    """One frame obeys the one-cell/no-overlap/no-swap/replay contract.

    Returns the frame's robot positions so they can serve as the predecessor
    of the next frame.
    """
    positions = {
        robot_id: tuple(cell) for robot_id, cell in frame["robots"].items()
    }
    moved = list(frame["moved"])

    # The moved list names current robots exactly once each.
    testcase.assertEqual(
        sorted(moved), sorted(set(moved)), "a robot appears twice in 'moved'"
    )
    testcase.assertTrue(set(moved) <= set(positions))

    if previous is not None:
        # 'moved' is exactly the set of robots whose position changed.
        changed = {
            robot_id
            for robot_id, position in positions.items()
            if previous[robot_id] != position
        }
        testcase.assertEqual(set(moved), changed)

        for robot_id in positions:
            before = previous[robot_id]
            after = positions[robot_id]
            distance = abs(after[0] - before[0]) + abs(after[1] - before[1])
            testcase.assertIn(distance, (0, 1))
            # A recorded move is exactly one orthogonal cell -- never a
            # second cell within the same tick.
            if robot_id in moved:
                testcase.assertEqual(distance, 1)

        # Two robots may not exchange cells within the same tick.
        ids = list(positions)
        for i, left in enumerate(ids):
            for right in ids[i + 1 :]:
                testcase.assertFalse(
                    positions[left] == previous[right]
                    and positions[right] == previous[left],
                    f"{left} and {right} swapped cells in one tick",
                )

    # No cell holds two robots at a tick boundary and nobody stands off the map.
    testcase.assertEqual(len(positions), len(set(positions.values())))
    for position in positions.values():
        testcase.assertTrue(simulator.grid.traversable(position))
    return positions


def assert_frame_matches_movement(
    testcase: unittest.TestCase,
    simulator: FleetSimulator,
    previous: dict[str, tuple[int, int]] | None,
    frame: dict,
    mileages_before: dict[str, int],
) -> dict[str, tuple[int, int]]:
    """Frame physical validity plus its mileage delta vs. the pre-tick state."""
    positions = assert_frame_physically_valid(testcase, simulator, previous, frame)
    for robot_id, robot in simulator.robots.items():
        delta = robot.distance_travelled - mileages_before[robot_id]
        testcase.assertEqual(delta, 1 if robot_id in frame["moved"] else 0)
    return positions


def assert_replay_physically_consistent(
    testcase: unittest.TestCase,
    simulator: FleetSimulator,
    initial_mileages: dict[str, int] | None = None,
) -> None:
    """Whole replay matches the movement that actually produced it.

    The first tick frame has no recorded predecessor, so only its cells,
    overlaps and ``moved`` ids are checked; from the second frame on every
    robot must have waited or taken one orthogonal step, no two robots may
    share a cell, and in-tick swaps are rejected. Each frame's ``moved`` list
    must name exactly the robots whose position changed. Each robot's current
    mileage must also equal its initial mileage plus the number of frames it
    appears in ``moved``.
    """
    if initial_mileages is None:
        initial_mileages = {robot_id: 0 for robot_id in simulator.robots}
    previous = None
    move_counts = {robot_id: 0 for robot_id in simulator.robots}
    for frame in tick_frames(simulator):
        previous = assert_frame_physically_valid(testcase, simulator, previous, frame)
        for robot_id in frame["moved"]:
            move_counts[robot_id] += 1
    for robot_id, robot in simulator.robots.items():
        testcase.assertEqual(
            robot.distance_travelled,
            initial_mileages[robot_id] + move_counts[robot_id],
        )


def step_and_check_replay(
    testcase: unittest.TestCase,
    simulator: FleetSimulator,
    previous: dict[str, tuple[int, int]] | None,
) -> tuple[dict, dict[str, tuple[int, int]]]:
    """Run one tick and verify the new frame against the pre-tick layout."""
    mileages_before = {
        robot_id: robot.distance_travelled
        for robot_id, robot in simulator.robots.items()
    }
    frame = simulator.step()
    positions = assert_frame_matches_movement(
        testcase, simulator, previous, frame, mileages_before
    )
    return frame, positions


# ---------------------------------------------------------------------------
# Scenario 1: blocked on the way to the pickup; the detour passes the dropoff
# ---------------------------------------------------------------------------


def detour_before_pickup_sim() -> FleetSimulator:
    # 4x2 with the top-right two cells walled off: the bottom row is the only
    # way around. R-1 starts empty, heading for its pickup at (3, 1); the
    # straight path runs along the top row through (1, 0), where busy R-2 is
    # parked head-on. R-1's detour dips to (0, 1) and then runs through (2, 1)
    # -- its own dropoff -- *before* reaching the pickup at (3, 1).
    return FleetSimulator(
        GridMap(4, 2, frozenset({(2, 0), (3, 0)})),
        [
            Robot(
                "R-1",
                (0, 0),
                route=[(1, 0), (1, 1), (2, 1), (3, 1), (2, 1)],
                task_id="T-1",
            ),
            Robot("R-2", (1, 0), route=[(0, 0)], task_id="T-2"),
        ],
        [
            Task("T-1", (3, 1), (2, 1), assigned_robot="R-1"),
            Task("T-2", (1, 1), (0, 0), assigned_robot="R-2", picked_up=True),
        ],
    )


class DetourBeforePickupTests(unittest.TestCase):
    def test_sidestep_detour_pickup_and_delivery_happen_in_order(self) -> None:
        sim = detour_before_pickup_sim()
        task = sim.tasks["T-1"]
        original_route = list(sim.robots["R-1"].route)
        previous = {
            robot_id: robot.position for robot_id, robot in sim.robots.items()
        }

        # Tick 1: R-1 sidesteps onto (0, 1), a cell its original route never
        # used, while R-2 drives on and finishes.
        frame, previous = step_and_check_replay(self, sim, previous)
        self.assertIn("R-1", frame["moved"])
        self.assertEqual(sim.robots["R-1"].position, (0, 1))
        self.assertNotIn((0, 1), original_route)

        # The replanned detour still owes the pickup before the dropoff:
        # the route ends at the dropoff and visits the pickup beforehand.
        route = sim.robots["R-1"].route
        self.assertEqual(route[-1], task.dropoff)
        self.assertIn(task.pickup, route)
        self.assertLess(route.index(task.pickup), len(route) - 1)

        observed_states: list[tuple[bool, bool]] = []
        for _ in range(2):  # ticks 2 and 3: closing in, tick 3 reaches dropoff
            frame, previous = step_and_check_replay(self, sim, previous)
            observed_states.append((task.picked_up, task.completed))
            self.assertEqual(frame["completed"], ["T-2"])

        # Tick 3 ends with the empty robot standing on its own dropoff cell.
        self.assertEqual(sim.robots["R-1"].position, task.dropoff)
        # Standing on the dropoff without goods is just passing through:
        # neither pickup nor completion may be reported.
        self.assertFalse(task.picked_up)
        self.assertFalse(task.completed)
        self.assertNotIn("T-1", sim.replay[-1]["completed"])
        # The same held for every pre-pickup tick.
        self.assertTrue(all(not pu and not done for pu, done in observed_states))

        # Tick 4: the step onto the pickup cell confirms the pickup.
        frame, previous = step_and_check_replay(self, sim, previous)
        self.assertEqual(sim.robots["R-1"].position, task.pickup)
        self.assertTrue(task.picked_up)
        self.assertFalse(task.completed)
        self.assertEqual(frame["completed"], ["T-2"])

        # Tick 5: carrying goods back to the dropoff finishes the task after
        # the remaining route is exhausted.
        frame, previous = step_and_check_replay(self, sim, previous)
        self.assertEqual(sim.robots["R-1"].position, task.dropoff)
        self.assertTrue(task.picked_up)
        self.assertTrue(task.completed)
        self.assertEqual(frame["completed"], ["T-1", "T-2"])
        self.assertEqual(sim.robots["R-1"].task_id, None)
        self.assertEqual(sim.robots["R-1"].distance_travelled, 5)

        # Completion appears for the first time exactly at that final frame;
        # passing the dropoff empty on tick 3 left no completion record.
        frames = tick_frames(sim)
        first_completed = next(
            index
            for index, frame in enumerate(frames)
            if "T-1" in frame["completed"]
        )
        self.assertEqual(first_completed, 4)  # zero-based frame index -> tick 5

    def test_task_keeps_its_robot_throughout_the_detour(self) -> None:
        sim = detour_before_pickup_sim()
        for _ in range(5):
            sim.step()
            task = sim.tasks["T-1"]
            if not task.completed:
                # Avoidance never releases or reassigns the task -- not even
                # after R-2 finishes its own work and becomes idle on tick 1.
                self.assertEqual(task.assigned_robot, "R-1")
                self.assertEqual(sim.robots["R-1"].task_id, "T-1")
                self.assertIsNone(sim.robots["R-2"].task_id)
        self.assertTrue(sim.tasks["T-1"].completed)
        self.assertEqual(sim.paused_tasks, set())
        assert_replay_physically_consistent(self, sim)

    def test_detour_finishes_the_delivery_within_budget(self) -> None:
        sim = detour_before_pickup_sim()
        for used in range(1, 12):
            sim.step()
            if all(task.completed for task in sim.tasks.values()):
                break
        else:
            self.fail("detoured delivery never completed")
        self.assertEqual(used, 5)
        self.assertTrue(all(task.completed for task in sim.tasks.values()))


# ---------------------------------------------------------------------------
# Scenario 2: tick starts on the uncollected pickup, the next cell is blocked
# ---------------------------------------------------------------------------


def pickup_sidestep_sim() -> FleetSimulator:
    # Open 4x2 map. R-1 starts the tick standing on its pickup (1, 0) with the
    # goods not yet confirmed; its next cell toward the dropoff (2, 0) is held
    # by busy R-2, which is heading the other way through (1, 0). The lower row
    # gives both robots room to avoid each other.
    return FleetSimulator(
        GridMap(4, 2),
        [
            Robot("R-1", (1, 0), route=[(2, 0), (3, 0)], task_id="T-1"),
            Robot("R-2", (2, 0), route=[(1, 0), (0, 0)], task_id="T-2"),
        ],
        [
            Task("T-1", (1, 0), (3, 0), assigned_robot="R-1"),
            Task("T-2", (2, 0), (0, 0), assigned_robot="R-2", picked_up=True),
        ],
    )


class PickupConfirmedBeforeSidestepTests(unittest.TestCase):
    def test_pickup_is_confirmed_then_one_sidestep_is_made(self) -> None:
        sim = pickup_sidestep_sim()
        task = sim.tasks["T-1"]
        positions_before = {
            robot_id: robot.position for robot_id, robot in sim.robots.items()
        }
        mileages_before = {
            robot_id: robot.distance_travelled
            for robot_id, robot in sim.robots.items()
        }
        frame = sim.step()

        # The pickup is confirmed before R-1 leaves the pickup cell, so the
        # sidestep is a loaded robot dodging, not an empty one driving off.
        self.assertTrue(task.picked_up)
        self.assertFalse(task.completed)
        self.assertEqual(task.assigned_robot, "R-1")
        self.assertEqual(sim.robots["R-1"].task_id, "T-1")

        # R-1's move is exactly one adjacent sidestep; it is listed once and
        # does not get a second cell in the same tick.
        self.assertEqual(frame["moved"].count("R-1"), 1)
        self.assertEqual(sim.robots["R-1"].position, (1, 1))
        self.assertEqual(
            abs(sim.robots["R-1"].position[0] - positions_before["R-1"][0])
            + abs(sim.robots["R-1"].position[1] - positions_before["R-1"][1]),
            1,
        )
        self.assertEqual(sim.robots["R-1"].distance_travelled, 1)

        # The replanned route runs straight to the dropoff: a loaded robot is
        # not sent on a second pickup trip, and the route must end at dropoff.
        route = sim.robots["R-1"].route
        self.assertEqual(route[-1], task.dropoff)
        # Nothing is completed just because pickup was confirmed.
        self.assertEqual(frame["completed"], [])
        assert_frame_matches_movement(
            self, sim, positions_before, frame, mileages_before
        )

    def test_goods_stay_on_board_and_delivery_completes_on_time(self) -> None:
        sim = pickup_sidestep_sim()
        task = sim.tasks["T-1"]
        # Ticks 1-5: the pickup is confirmed and R-1 sidesteps; R-2 finishes
        # and frees up; R-1 returns to the top row and drives to the dropoff.
        # Pickup must never be forgotten or re-done.
        for expected_tick in range(1, 6):
            frame = sim.step()
            self.assertEqual(sim.tick, expected_tick)
            self.assertTrue(task.picked_up)
            self.assertEqual(task.assigned_robot, "R-1")
            # Until the delivery actually lands, an idle R-2 must not steal a
            # task that still has an owner.
            if not task.completed:
                self.assertEqual(sim.robots["R-1"].task_id, "T-1")
        self.assertEqual(sim.robots["R-1"].position, task.dropoff)
        self.assertTrue(task.completed)
        # 1 sidestep + back onto the row + 2 delivery cells: no extra pickup
        # leg was scheduled, so the whole job costs four moves.
        self.assertEqual(sim.robots["R-1"].distance_travelled, 4)
        # The completion record exists only from the true delivery tick on.
        frames = tick_frames(sim)
        self.assertNotIn("T-1", frames[3]["completed"])
        self.assertIn("T-1", frames[4]["completed"])
        assert_replay_physically_consistent(self, sim)


# ---------------------------------------------------------------------------
# Scenario 3: no safe side road before pickup -- wait, never false-pause
# ---------------------------------------------------------------------------


def blocked_before_pickup_corridor() -> FleetSimulator:
    # Pure 6x1 corridor: there is no side cell anywhere. R-1 is still empty
    # and on its way to the pickup at (4, 0); idle R-2 is parked at (2, 0)
    # directly in its path.
    return FleetSimulator(
        GridMap(6, 1),
        [
            Robot("R-1", (1, 0), route=[(2, 0), (3, 0), (4, 0), (5, 0)],
                  task_id="T-1"),
            Robot("R-2", (2, 0)),
        ],
        [Task("T-1", (4, 0), (5, 0), assigned_robot="R-1")],
    )


class NoSideRoadBeforePickupTests(unittest.TestCase):
    def test_empty_robot_waits_in_place_without_pickup_or_pause(self) -> None:
        sim = blocked_before_pickup_corridor()
        task = sim.tasks["T-1"]
        robot = sim.robots["R-1"]
        start_position = robot.position
        start_route = list(robot.route)

        # Both task points are genuinely reachable on the map -- only the
        # traffic in front cannot clear.
        self.assertTrue(shortest_path(sim.grid, robot.position, task.pickup))
        self.assertTrue(shortest_path(sim.grid, task.pickup, task.dropoff))

        for expected in range(1, 6):
            frame = sim.step()
            self.assertEqual(frame["moved"], [])
            self.assertEqual(frame["completed"], [])
            # Time advances, the robot stays put and mileage does not grow.
            self.assertEqual(sim.tick, expected)
            self.assertEqual(robot.position, start_position)
            self.assertEqual(robot.distance_travelled, 0)
            self.assertEqual(robot.route, start_route)
            # Goods not yet collected: neither flag may flip while waiting.
            self.assertFalse(task.picked_up)
            self.assertFalse(task.completed)
            # Traffic, not an unreachable map: no pause, ownership unchanged.
            self.assertEqual(sim.paused_tasks, set())
            self.assertEqual(task.assigned_robot, "R-1")
            self.assertEqual(robot.task_id, "T-1")
            self.assertIsNone(sim.robots["R-2"].task_id)
            self.assertEqual(
                sim.status()["traffic_waits"],
                [{"robot_id": "R-1", "blocked_by": ["R-2"], "ticks": expected}],
            )

        # Every waiting frame records identical positions and no movements.
        frames = tick_frames(sim)
        for frame in frames:
            self.assertEqual(
                frame["robots"], {"R-1": [1, 0], "R-2": [2, 0]}
            )
        assert_replay_physically_consistent(self, sim)

    def test_waiting_empty_robot_completes_once_a_side_road_opens(self) -> None:
        # Same corridor, but a side pocket below (2, 0) exists and can be
        # reopened: the empty robot must only *wait* while traffic cannot
        # clear, then follow through, collect and deliver normally.
        sim = FleetSimulator(
            GridMap(6, 2, frozenset((x, 1) for x in range(6))),
            [
                Robot(
                    "R-1",
                    (1, 0),
                    route=[(2, 0), (3, 0), (4, 0), (5, 0)],
                    task_id="T-1",
                ),
                Robot("R-2", (2, 0)),
            ],
            [Task("T-1", (4, 0), (5, 0), assigned_robot="R-1")],
        )
        task = sim.tasks["T-1"]
        for _ in range(3):
            frame = sim.step()
            self.assertEqual(frame["moved"], [])
            self.assertFalse(task.picked_up)
        self.assertEqual(
            sim.status()["traffic_waits"],
            [{"robot_id": "R-1", "blocked_by": ["R-2"], "ticks": 3}],
        )
        self.assertEqual(sim.paused_tasks, set())

        sim.modify_obstacles(removed=[(2, 1)])
        # R-2 yields into the reopened pocket and R-1 follows in the same tick;
        # the wait record clears because R-1 moved.
        frame = sim.step()
        self.assertEqual(frame["moved"], ["R-2", "R-1"])
        self.assertEqual(sim.robots["R-2"].position, (2, 1))
        self.assertEqual(sim.robots["R-1"].position, (2, 0))
        self.assertFalse(task.picked_up)
        self.assertFalse(task.completed)
        self.assertEqual(sim.status()["traffic_waits"], [])

        # Continue to the pickup: goods are collected there, not earlier.
        sim.step()
        sim.step()
        self.assertEqual(sim.robots["R-1"].position, task.pickup)
        self.assertTrue(task.picked_up)
        self.assertFalse(task.completed)
        frame = sim.step()
        self.assertEqual(sim.robots["R-1"].position, task.dropoff)
        self.assertTrue(task.completed)
        self.assertIn("T-1", frame["completed"])
        self.assertEqual(sim.robots["R-1"].distance_travelled, 4)
        self.assertEqual(task.assigned_robot, "R-1")
        assert_replay_physically_consistent(self, sim)


if __name__ == "__main__":
    unittest.main()
