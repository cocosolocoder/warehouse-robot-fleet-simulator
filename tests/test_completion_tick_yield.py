"""Regression coverage for a robot that finishes on a cell another car wants.

When a task first appears in a tick frame's ``completed`` list, the robot that
executed it must still stand on the dropoff when that tick ends -- both when it
drove onto the dropoff that tick and when it was already standing there and
finished without moving. Until the tick ends it is therefore never pushed off
the cell by the automatic yielding rules, even though a free neighbouring cell
exists and another robot's remaining route runs through it. The hold lasts
exactly one tick: on the next tick the robot is an ordinary parked robot again
and yields normally, so the dropoff is never reserved permanently.

The bug this guards against let the finished robot yield on the completion tick
itself. The replay then recorded it away from the dropoff in the very frame in
which the task first appeared as completed, and the first-completion position
evidence checkpoint loading requires contradicted the saved layout -- normal
operation produced a checkpoint that could not be reloaded.
"""

import os
import tempfile
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


# 4 wide, 2 tall, no obstacles. (1, 1) is the side cell a parked robot on
# (1, 0) would normally be asked to yield onto:
#   y=0:  .  X  .  .
#   y=1:  .  .  .  .
PICKUP = DROPOFF = (1, 0)
SIDE_CELL = (1, 1)


def build_scenario() -> FleetSimulator:
    # R-1's preset route ends on the only task's coincident pickup/dropoff;
    # R-2's preset route runs straight through that cell afterwards. Both
    # start without a task, so the task can only be assigned once R-1's
    # preset route is exhausted.
    return FleetSimulator(
        GridMap(4, 2),
        [
            Robot("R-1", (0, 0), route=[(1, 0)]),
            Robot("R-2", (3, 0), route=[(2, 0), (1, 0), (0, 0)]),
        ],
        [Task("T-1", pickup=PICKUP, dropoff=DROPOFF)],
    )


def tick_frames(simulator: FleetSimulator) -> list[dict[str, object]]:
    return [frame for frame in simulator.replay if frame.get("type") == "tick"]


def round_trip(testcase: unittest.TestCase, simulator: FleetSimulator) -> FleetSimulator:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        simulator.save_checkpoint(path)
        return FleetSimulator.load_checkpoint(path)


class InPlaceCompletionHoldsTheDropoffTests(unittest.TestCase):
    def test_tick_one_preset_moves_leave_task_unassigned(self) -> None:
        simulator = build_scenario()
        event = simulator.step()
        self.assertEqual(event["moved"], ["R-1", "R-2"])
        self.assertEqual(event["robots"], {"R-1": [1, 0], "R-2": [2, 0]})
        self.assertEqual(event["completed"], [])
        task = simulator.tasks["T-1"]
        self.assertIsNone(task.assigned_robot)
        self.assertFalse(task.picked_up)
        self.assertFalse(task.completed)
        # R-1's preset route is exhausted, so it becomes available from the
        # next assignment pass; R-2 still drives its preset route.
        self.assertEqual(simulator.robots["R-1"].route, [])
        self.assertEqual(simulator.robots["R-2"].route, [(1, 0), (0, 0)])

    def test_completion_tick_finishes_in_place_without_moving_either_robot(self) -> None:
        simulator = build_scenario()
        simulator.step()
        event = simulator.step()

        r1, r2 = simulator.robots["R-1"], simulator.robots["R-2"]
        task = simulator.tasks["T-1"]
        # Assigned to the preset-finished R-1 and completed straight away.
        self.assertEqual(task.assigned_robot, "R-1")
        self.assertTrue(task.picked_up)
        self.assertTrue(task.completed)
        self.assertIsNone(r1.task_id)
        # The finisher ends the completion tick standing on the dropoff even
        # though the side cell (1, 1) is free and R-2 wants the cell; R-2 does
        # not enter it either.
        self.assertEqual(r1.position, DROPOFF)
        self.assertEqual(r2.position, (2, 0))
        self.assertEqual(event["moved"], [])
        self.assertEqual(event["robots"], {"R-1": [1, 0], "R-2": [2, 0]})
        self.assertEqual(event["completed"], ["T-1"])
        # In-place completion adds no mileage; neither robot drove this tick.
        self.assertEqual(r1.distance_travelled, 1)
        self.assertEqual(r2.distance_travelled, 1)
        self.assertEqual(simulator.metrics()["tasks_completed"], 1)
        self.assertEqual(simulator.metrics()["distance_total"], 2)
        # R-2's traffic wait points at the robot holding its next cell, R-1.
        self.assertEqual(
            simulator.status()["traffic_waits"],
            [{"robot_id": "R-2", "blocked_by": ["R-1"], "ticks": 1}],
        )

    def test_next_tick_yielding_resumes_and_wait_clears(self) -> None:
        simulator = build_scenario()
        simulator.step()
        simulator.step()
        event = simulator.step()

        r1, r2 = simulator.robots["R-1"], simulator.robots["R-2"]
        # The hold is one tick long: R-1 now yields like any parked robot and
        # R-2 follows it into the dropoff cell, each driving one cell.
        self.assertEqual(r1.position, SIDE_CELL)
        self.assertEqual(r2.position, DROPOFF)
        self.assertEqual(event["moved"], ["R-1", "R-2"])
        self.assertEqual(event["robots"], {"R-1": [1, 1], "R-2": [1, 0]})
        self.assertEqual(event["completed"], ["T-1"])
        self.assertEqual(r1.distance_travelled, 2)
        self.assertEqual(r2.distance_travelled, 2)
        # Moving cleared R-2's wait record.
        self.assertEqual(simulator.status()["traffic_waits"], [])
        # The finished task stays history only; R-1 is a free robot again,
        # not permanently bound to or occupying the dropoff.
        self.assertIsNone(r1.task_id)
        self.assertEqual(simulator.tasks["T-1"].assigned_robot, "R-1")
        self.assertTrue(simulator.tasks["T-1"].completed)

    def test_replay_moved_and_completion_lists_track_the_real_state(self) -> None:
        simulator = build_scenario()
        for _ in range(3):
            simulator.step()
        frames = tick_frames(simulator)
        self.assertEqual([frame["tick"] for frame in frames], [1, 2, 3])
        self.assertEqual(
            [frame["moved"] for frame in frames],
            [["R-1", "R-2"], [], ["R-1", "R-2"]],
        )
        self.assertEqual(
            [frame["completed"] for frame in frames],
            [[], ["T-1"], ["T-1"]],
        )
        # The first-completion frame itself keeps the finisher on the dropoff.
        self.assertEqual(frames[1]["robots"]["R-1"], list(DROPOFF))
        # Physical plausibility: one-cell moves, no overlaps.
        for frame in frames:
            positions = list(map(tuple, frame["robots"].values()))
            self.assertEqual(len(positions), len(set(positions)))

    def test_checkpoints_from_normal_operation_load_at_every_tick(self) -> None:
        simulator = build_scenario()
        simulator.step()
        loaded_one = round_trip(self, simulator)
        self.assertEqual(loaded_one.tasks["T-1"].completed, False)

        simulator.step()
        loaded_two = round_trip(self, simulator)
        # The exact document the old bug produced was rejected here: the
        # completion frame showed R-1 away from the dropoff.
        completion_frame = tick_frames(loaded_two)[1]
        self.assertEqual(completion_frame["robots"]["R-1"], list(DROPOFF))
        self.assertEqual(loaded_two.robots["R-1"].position, DROPOFF)
        self.assertTrue(loaded_two.tasks["T-1"].completed)
        self.assertEqual(loaded_two.metrics()["tasks_completed"], 1)

        simulator.step()
        loaded_three = round_trip(self, simulator)
        self.assertEqual(loaded_three.robots["R-1"].position, SIDE_CELL)
        self.assertEqual(loaded_three.robots["R-2"].position, DROPOFF)
        self.assertEqual(loaded_three.replay, simulator.replay)
        # Resuming the restored fleet keeps behaving identically.
        loaded_three.step()
        simulator.step()
        self.assertEqual(loaded_three.replay, simulator.replay)


class ArrivalCompletionHoldsTheDropoffTests(unittest.TestCase):
    """The same hold applies when the robot drives onto the dropoff to finish."""

    def build(self) -> FleetSimulator:
        # R-1 carries T-1 and its last step is onto the shared cell (1, 0);
        # R-2's taskless preset route immediately wants that same cell. The
        # side cell (1, 1) is open, so without the hold R-1 would be pushed
        # off the dropoff on its own completion tick.
        return FleetSimulator(
            GridMap(4, 2),
            [
                Robot("R-1", (0, 0), route=[(1, 0)], task_id="T-1"),
                Robot("R-2", (2, 0), route=[(1, 0), (0, 0)]),
            ],
            [
                Task(
                    "T-1",
                    pickup=(0, 0),
                    dropoff=(1, 0),
                    assigned_robot="R-1",
                    picked_up=True,
                )
            ],
        )

    def test_finisher_stays_on_the_dropoff_and_wait_is_recorded(self) -> None:
        simulator = self.build()
        event = simulator.step()
        r1, r2 = simulator.robots["R-1"], simulator.robots["R-2"]
        self.assertTrue(simulator.tasks["T-1"].completed)
        self.assertEqual(r1.position, DROPOFF)
        self.assertEqual(r2.position, (2, 0))
        # Only the finisher drove: the delivery step counts, the yielding
        # detour does not happen until next tick.
        self.assertEqual(event["moved"], ["R-1"])
        self.assertEqual(event["completed"], ["T-1"])
        self.assertEqual(r1.distance_travelled, 1)
        self.assertEqual(
            simulator.status()["traffic_waits"],
            [{"robot_id": "R-2", "blocked_by": ["R-1"], "ticks": 1}],
        )
        round_trip(self, simulator)

    def test_yield_happens_on_the_following_tick_only(self) -> None:
        simulator = self.build()
        simulator.step()
        event = simulator.step()
        self.assertEqual(event["moved"], ["R-1", "R-2"])
        self.assertEqual(simulator.robots["R-1"].position, SIDE_CELL)
        self.assertEqual(simulator.robots["R-2"].position, DROPOFF)
        self.assertEqual(simulator.status()["traffic_waits"], [])
        round_trip(self, simulator)


class FinishedHistoryNeverBlocksLaterMovementTests(unittest.TestCase):
    def test_historical_completion_does_not_freeze_the_robot(self) -> None:
        # R-1 finishes T-1 on tick 1 at (1, 0); nothing else needs the cell
        # that tick. From tick 2 on it is ordinary parked traffic and moves
        # when another robot asks it to.
        simulator = FleetSimulator(
            GridMap(4, 2),
            [
                Robot("R-1", (0, 0), route=[(1, 0)], task_id="T-1"),
                Robot("R-2", (3, 0)),
            ],
            [
                Task(
                    "T-1",
                    pickup=(0, 0),
                    dropoff=(1, 0),
                    assigned_robot="R-1",
                    picked_up=True,
                ),
                Task("T-2", pickup=(3, 0), dropoff=(0, 0)),
            ],
        )
        first = simulator.step()
        self.assertTrue(simulator.tasks["T-1"].completed)
        self.assertEqual(first["robots"]["R-1"], [1, 0])
        # Drive R-2's delivery all the way to (0, 0); R-1 yields on request.
        for _ in range(6):
            simulator.step()
        self.assertEqual(simulator.robots["R-2"].position, (0, 0))
        self.assertTrue(simulator.tasks["T-2"].completed)
        # R-1 was not pinned to the historical dropoff forever.
        self.assertNotEqual(simulator.robots["R-1"].position, DROPOFF)
        round_trip(self, simulator)


if __name__ == "__main__":
    unittest.main()
