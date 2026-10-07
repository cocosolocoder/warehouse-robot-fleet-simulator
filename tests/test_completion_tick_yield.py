"""Regression tests for a robot that finishes a task being pushed off the dropoff.

When a task first shows up in a tick's ``completed`` list, the robot that
executed it must still stand on that task's dropoff when the tick ends --
whether it walked to the dropoff to complete or was already standing there
and completed without moving. For the rest of that very tick the robot must
neither yield for another robot nor be routed around by the traffic rules,
even when a free side cell exists; the next tick normal idle and yielding
behaviour resumes. The stay is one tick only: an already completed task is
mere history and must never keep its delivery robot parked on the dropoff.

The observable channels all have to agree about that tick: the robot's
position and remaining route, the task flags and the robot binding, the
replay frame's ``moved``/``robots``/``completed`` lists, the traffic wait
report (the requesting robot waits, pointing at the delivery robot, until it
moves) and a checkpoint saved from the run -- which must pass the existing,
strict first-completion location validation rather than any relaxed rule.
"""

import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


def build_scenario() -> FleetSimulator:
    """The 4x2 scenario: R-1 walks one cell onto the shared point, R-2 follows.

    R-1's preset route only reaches (1,0); R-2's preset route walks
    (3,0) -> (2,0) -> (1,0) -> (0,0). The one waiting task has its pickup
    and dropoff both at (1,0), so it is assigned to R-1 only after R-1's
    preset route is exhausted and is then completed in place.
    """
    return FleetSimulator(
        GridMap(4, 2),
        [
            Robot("R-1", (0, 0), route=[(1, 0)]),
            Robot("R-2", (3, 0), route=[(2, 0), (1, 0), (0, 0)]),
        ],
        [Task("T-1", pickup=(1, 0), dropoff=(1, 0))],
    )


class InPlaceCompletionHoldsDropoffTests(unittest.TestCase):
    def test_first_tick_walks_the_preset_routes_without_assignment(self) -> None:
        simulator = build_scenario()
        event = simulator.step()

        self.assertEqual(event["moved"], ["R-1", "R-2"])
        self.assertEqual(event["completed"], [])
        self.assertEqual(simulator.robots["R-1"].position, (1, 0))
        self.assertEqual(simulator.robots["R-2"].position, (2, 0))
        # The task is still unassigned while the preset routes are in motion.
        self.assertIsNone(simulator.tasks["T-1"].assigned_robot)
        self.assertFalse(simulator.tasks["T-1"].picked_up)
        self.assertEqual(simulator.status()["traffic_waits"], [])

    def test_completion_tick_holds_the_dropoff_despite_the_request(self) -> None:
        simulator = build_scenario()
        simulator.step()
        event = simulator.step()

        task = simulator.tasks["T-1"]
        r1, r2 = simulator.robots["R-1"], simulator.robots["R-2"]
        # Picked up and completed on that tick, R-1 kept as the historical
        # delivery robot, and its current task released.
        self.assertTrue(task.picked_up)
        self.assertTrue(task.completed)
        self.assertEqual(task.assigned_robot, "R-1")
        self.assertIsNone(r1.task_id)
        # Both robots stay put even though the free side cell (1,1) exists:
        # the freshly delivered robot is not pushed off the dropoff, and the
        # requesting robot waits rather than swapping places with it.
        self.assertEqual(r1.position, (1, 0))
        self.assertEqual(r2.position, (2, 0))
        self.assertEqual(r1.route, [])
        self.assertEqual(r2.route, [(1, 0), (0, 0)])
        # The replay frame mirrors that settled layout exactly.
        self.assertEqual(event["moved"], [])
        self.assertEqual(
            event["robots"], {"R-1": [1, 0], "R-2": [2, 0]}
        )
        self.assertEqual(event["completed"], ["T-1"])
        # An in-place completion adds no mileage; waiting adds none either.
        self.assertEqual(r1.distance_travelled, 1)
        self.assertEqual(r2.distance_travelled, 1)
        # The requester's wait points at the robot holding its next cell.
        self.assertEqual(
            simulator.status()["traffic_waits"],
            [{"robot_id": "R-2", "blocked_by": ["R-1"], "ticks": 1}],
        )
        self.assertEqual(simulator.metrics()["tasks_completed"], 1)

    def test_next_tick_yields_normally_and_the_wait_clears(self) -> None:
        simulator = build_scenario()
        simulator.step()
        simulator.step()
        event = simulator.step()

        r1, r2 = simulator.robots["R-1"], simulator.robots["R-2"]
        # Ordinary idle/yield behaviour resumes: R-1 moves onto the free side
        # cell and R-2 follows into the just-vacated dropoff.
        self.assertEqual(r1.position, (1, 1))
        self.assertEqual(r2.position, (1, 0))
        self.assertEqual(r1.route, [])
        self.assertEqual(r2.route, [(0, 0)])
        self.assertEqual(event["moved"], ["R-1", "R-2"])
        self.assertEqual(
            event["robots"], {"R-1": [1, 1], "R-2": [1, 0]}
        )
        self.assertEqual(event["completed"], ["T-1"])
        # Each move counts exactly one cell of mileage for that tick.
        self.assertEqual(r1.distance_travelled, 2)
        self.assertEqual(r2.distance_travelled, 2)
        # Moving clears the wait.
        self.assertEqual(simulator.status()["traffic_waits"], [])

    def test_the_completion_robot_is_not_permanently_parked(self) -> None:
        simulator = build_scenario()
        for _ in range(4):
            simulator.step()

        r1, r2 = simulator.robots["R-1"], simulator.robots["R-2"]
        # R-2 finishes its preset route; R-1 is an ordinary idle robot now and
        # never reoccupies the dropoff merely because it once completed there.
        self.assertEqual(r2.position, (0, 0))
        self.assertEqual(r2.route, [])
        self.assertEqual(r1.position, (1, 1))
        self.assertIsNone(r1.task_id)
        self.assertEqual(simulator.status()["traffic_waits"], [])
        self.assertEqual(simulator.metrics()["tasks_completed"], 1)


class WalkToCompletionHoldsDropoffTests(unittest.TestCase):
    """The same stay rule when the robot has to walk onto the dropoff."""

    def build(self) -> FleetSimulator:
        # R-1 carries loaded goods one cell into the dropoff (2,0); R-2,
        # processed afterwards, wants that same cell next on its preset route.
        return FleetSimulator(
            GridMap(4, 2),
            [
                Robot("R-1", (1, 0), route=[(2, 0)], task_id="T-1"),
                Robot("R-2", (3, 0), route=[(2, 0), (1, 0)]),
            ],
            [
                Task(
                    "T-1",
                    pickup=(1, 0),
                    dropoff=(2, 0),
                    assigned_robot="R-1",
                    picked_up=True,
                )
            ],
        )

    def test_walking_in_to_complete_still_holds_the_cell_for_the_tick(self) -> None:
        simulator = self.build()
        event = simulator.step()

        r1, r2 = simulator.robots["R-1"], simulator.robots["R-2"]
        self.assertEqual(r1.position, (2, 0))
        self.assertEqual(r2.position, (3, 0))
        self.assertTrue(simulator.tasks["T-1"].completed)
        self.assertEqual(simulator.tasks["T-1"].assigned_robot, "R-1")
        self.assertIsNone(r1.task_id)
        # Only the delivery robot is listed as moved: the one-cell delivery
        # step. The following robot does not also push it aside in this tick.
        self.assertEqual(event["moved"], ["R-1"])
        self.assertEqual(
            event["robots"], {"R-1": [2, 0], "R-2": [3, 0]}
        )
        self.assertEqual(event["completed"], ["T-1"])
        self.assertEqual(r1.distance_travelled, 1)
        self.assertEqual(r2.distance_travelled, 0)
        self.assertEqual(
            simulator.status()["traffic_waits"],
            [{"robot_id": "R-2", "blocked_by": ["R-1"], "ticks": 1}],
        )

    def test_following_tick_yields_and_advances(self) -> None:
        simulator = self.build()
        simulator.step()
        event = simulator.step()

        self.assertEqual(event["moved"], ["R-1", "R-2"])
        self.assertEqual(
            event["robots"], {"R-1": [2, 1], "R-2": [2, 0]}
        )
        self.assertEqual(simulator.status()["traffic_waits"], [])
        self.assertEqual(simulator.robots["R-1"].distance_travelled, 2)
        self.assertEqual(simulator.robots["R-2"].distance_travelled, 1)


class RequesterBeforeFinisherTests(unittest.TestCase):
    """An earlier-processed requester waits even before the finisher completes."""

    def build(self) -> FleetSimulator:
        # "A" sorts before "B": A asks for (1,0) while B (later in the order)
        # is still task-bound with an empty route and completes in place.
        return FleetSimulator(
            GridMap(3, 2),
            [
                Robot("A", (2, 0), route=[(1, 0)]),
                Robot("B", (1, 0), task_id="TB"),
            ],
            [Task("TB", pickup=(1, 0), dropoff=(1, 0), assigned_robot="B")],
        )

    def test_requester_waits_on_the_completion_tick(self) -> None:
        simulator = self.build()
        event = simulator.step()

        self.assertEqual(event["moved"], [])
        self.assertEqual(event["robots"], {"A": [2, 0], "B": [1, 0]})
        self.assertEqual(event["completed"], ["TB"])
        self.assertTrue(simulator.tasks["TB"].completed)
        self.assertEqual(
            simulator.status()["traffic_waits"],
            [{"robot_id": "A", "blocked_by": ["B"], "ticks": 1}],
        )

    def test_next_tick_unblocks_normally(self) -> None:
        simulator = self.build()
        simulator.step()
        event = simulator.step()

        self.assertEqual(set(event["moved"]), {"A", "B"})
        self.assertEqual(event["robots"]["A"], [1, 0])
        self.assertNotEqual(event["robots"]["B"], [1, 0])
        self.assertEqual(simulator.status()["traffic_waits"], [])


class CompletionTickCheckpointTests(unittest.TestCase):
    def _run_to_completion_tick(self) -> FleetSimulator:
        simulator = build_scenario()
        simulator.step()
        simulator.step()
        return simulator

    def test_checkpoint_at_completion_tick_saves_and_loads(self) -> None:
        import os
        import tempfile

        simulator = self._run_to_completion_tick()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "fleet.json")
            simulator.save_checkpoint(path)
            restored = FleetSimulator.load_checkpoint(path)

        r1, r2 = restored.robots["R-1"], restored.robots["R-2"]
        self.assertEqual((r1.position, r1.route, r1.task_id), ((1, 0), [], None))
        self.assertEqual((r2.position, r2.route), ((2, 0), [(1, 0), (0, 0)]))
        self.assertEqual(
            (r1.distance_travelled, r2.distance_travelled), (1, 1)
        )
        task = restored.tasks["T-1"]
        self.assertTrue(task.completed)
        self.assertTrue(task.picked_up)
        self.assertEqual(task.assigned_robot, "R-1")
        # The wait survives the round trip as well.
        self.assertEqual(
            restored.status()["traffic_waits"],
            [{"robot_id": "R-2", "blocked_by": ["R-1"], "ticks": 1}],
        )

    def test_resumed_run_matches_an_uninterrupted_one(self) -> None:
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "fleet.json")
            interrupted = self._run_to_completion_tick()
            uninterrupted = build_scenario()
            uninterrupted.step()
            uninterrupted.step()
            interrupted.save_checkpoint(path)
            resumed = FleetSimulator.load_checkpoint(path)
            for _ in range(3):
                uninterrupted.step()
                resumed.step()

        self.assertEqual(resumed.tick, uninterrupted.tick)
        self.assertEqual(resumed.replay, uninterrupted.replay)
        for robot_id in uninterrupted.robots:
            self.assertEqual(
                resumed.robots[robot_id].position,
                uninterrupted.robots[robot_id].position,
            )
            self.assertEqual(
                resumed.robots[robot_id].route,
                uninterrupted.robots[robot_id].route,
            )
            self.assertEqual(
                resumed.robots[robot_id].distance_travelled,
                uninterrupted.robots[robot_id].distance_travelled,
            )
        self.assertEqual(
            resumed.status()["traffic_waits"],
            uninterrupted.status()["traffic_waits"],
        )

    def test_forged_frame_moving_the_finisher_is_still_rejected(self) -> None:
        # The fix must not come from relaxing first-completion evidence: a
        # hand-forged frame that parks the finisher on the side cell still
        # fails the existing strict location validation.
        import json
        import os
        import tempfile

        simulator = self._run_to_completion_tick()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "fleet.json")
            simulator.save_checkpoint(path)
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
            frame = document["replay"][-1]
            self.assertEqual(frame["tick"], 2)
            frame["robots"]["R-1"] = [1, 1]
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(document, handle)
            with self.assertRaises(ValueError) as context:
                FleetSimulator.load_checkpoint(path)
        self.assertIn("dropoff", str(context.exception))


if __name__ == "__main__":
    unittest.main()
