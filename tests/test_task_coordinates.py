"""Validation of task pickup/dropoff coordinates on directly constructed fleets.

A ``Task`` handed to ``FleetSimulator`` carries its pickup and dropoff in the
same coordinate shape a robot position uses. Construction must apply the same
strict coordinate rule: each point accepts only a list or tuple of two plain
integers (the spellings may be mixed), booleans and floats never count as
integers, and every accepted point becomes a fresh independent integer tuple.
Illegal points are rejected at construction -- never rounded, padded or
coerced -- with :class:`ValueError` naming the task and whether its pickup or
its dropoff is at fault, so a fleet can never accept a task that checkpoint
loading would later reject and list-shaped points can never raise
``TypeError`` while planning. The rule covers waiting, assigned, loaded and
completed tasks, and nothing is committed until the whole batch passes, so a
later bad point, ownership error or route error leaves every earlier object
exactly as passed in.
"""

import os
import tempfile
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


def make_sim(robots=(), tasks=(), grid=None):
    return FleetSimulator(
        GridMap(6, 4) if grid is None else grid, list(robots), list(tasks)
    )


def expect_rejected(testcase, tasks, *fragments, robots=(), grid=None):
    with testcase.assertRaises(ValueError) as context:
        make_sim(robots, tasks, grid)
    message = str(context.exception)
    for fragment in fragments:
        testcase.assertIn(fragment, message)
    return message


class MalformedTaskPointTests(unittest.TestCase):
    def test_scalars_and_non_coordinates_are_rejected(self) -> None:
        for bad in (True, False, 1, 1.0, 0.0, "10", None, {1: 2}, b"10"):
            expect_rejected(
                self, [Task("T-1", bad, (2, 0))], "T-1", "pickup"
            )
            expect_rejected(
                self, [Task("T-1", (0, 0), bad)], "T-1", "dropoff"
            )

    def test_wrong_lengths_are_rejected(self) -> None:
        for bad in ([1], (1,), [1, 2, 3], (1, 2, 3), [], ()):
            expect_rejected(
                self, [Task("T-1", bad, (2, 0))], "T-1", "pickup"
            )
            expect_rejected(
                self, [Task("T-1", (0, 0), bad)], "T-1", "dropoff"
            )

    def test_non_integer_components_are_rejected_without_coercion(self) -> None:
        for bad in (
            [True, 0],
            [0, False],
            [1.0, 2],
            [1, 2.0],
            ["1", 2],
            [1, "2"],
            [None, 0],
        ):
            message = expect_rejected(
                self, [Task("T-1", bad, (2, 0))], "T-1", "pickup"
            )
            self.assertNotIn("TypeError", message)
            message = expect_rejected(
                self, [Task("T-1", (0, 0), bad)], "T-1", "dropoff"
            )
            self.assertNotIn("TypeError", message)

    def test_float_that_equals_an_integer_is_still_rejected(self) -> None:
        expect_rejected(
            self, [Task("T-1", [1.0, 0], (2, 0))], "T-1", "pickup"
        )
        expect_rejected(
            self, [Task("T-1", (0, 0), [2.0, 0])], "T-1", "dropoff"
        )

    def test_rejection_names_task_and_which_point(self) -> None:
        message = expect_rejected(
            self,
            [
                Task("T-good", (0, 0), (1, 0)),
                Task("T-bad", (1, 0), [2, True]),
            ],
            "T-bad",
            "dropoff",
        )
        self.assertNotIn("T-good", message)
        self.assertNotIn("TypeError", message)
        message = expect_rejected(
            self, [Task("T-77", ["x", 0], (1, 0))], "T-77", "pickup"
        )
        self.assertNotIn("TypeError", message)

    def test_bad_points_raise_value_error_not_type_error(self) -> None:
        # A list point used to survive construction and only blow up later as
        # an unhashable-list TypeError while planning; an actually malformed
        # point must now fail up front with ValueError.
        for bad_pickup, bad_dropoff in ((True, (1, 0)), ((0, 0), 1.5)):
            with self.assertRaises(ValueError):
                make_sim(
                    [Robot("R-1", (0, 0))],
                    [Task("T-1", bad_pickup, bad_dropoff)],
                    grid=GridMap(3, 3),
                )


class PointNormalizationTests(unittest.TestCase):
    def test_list_and_tuple_become_independent_integer_tuples(self) -> None:
        for pickup, dropoff in (([1, 0], [2, 0]), ((1, 0), (2, 0))):
            task = Task("T-1", pickup, dropoff)
            sim = make_sim([Robot("R-1", (0, 0))], [task], grid=GridMap(3, 1))
            self.assertEqual(task.pickup, (1, 0))
            self.assertEqual(task.dropoff, (2, 0))
            self.assertIsInstance(task.pickup, tuple)
            self.assertIsInstance(task.dropoff, tuple)
            self.assertIsNot(task.pickup, pickup)
            self.assertIsNot(task.dropoff, dropoff)

    def test_spellings_may_be_mixed_within_one_task(self) -> None:
        task = Task("T-1", [1, 0], (2, 0))
        sim = make_sim([Robot("R-1", (0, 0))], [task], grid=GridMap(3, 1))
        self.assertEqual(sim.tasks["T-1"].pickup, (1, 0))
        self.assertEqual(sim.tasks["T-1"].dropoff, (2, 0))

    def test_list_and_tuple_points_behave_identically(self) -> None:
        list_sim = make_sim(
            [Robot("R-1", (0, 0))],
            [Task("T-1", [1, 0], [2, 0])],
            grid=GridMap(3, 1),
        )
        tuple_sim = make_sim(
            [Robot("R-1", (0, 0))],
            [Task("T-1", (1, 0), (2, 0))],
            grid=GridMap(3, 1),
        )
        for _ in range(3):
            list_sim.step()
            tuple_sim.step()
        self.assertTrue(list_sim.tasks["T-1"].completed)
        self.assertTrue(tuple_sim.tasks["T-1"].completed)
        self.assertEqual(
            [frame["robots"] for frame in list_sim.replay],
            [frame["robots"] for frame in tuple_sim.replay],
        )
        self.assertEqual(
            list_sim.robots["R-1"].distance_travelled,
            tuple_sim.robots["R-1"].distance_travelled,
        )

    def test_list_pickup_still_collects_and_tuple_dropoff_delivers(self) -> None:
        # [1, 0] and (1, 0) are one cell: a container mismatch must neither
        # skip the pickup nor block delivery confirmation (previously the
        # list points raised TypeError during assignment).
        task = Task("T-1", [1, 0], (2, 0))
        sim = make_sim([Robot("R-1", (0, 0))], [task], grid=GridMap(3, 1))
        sim.step()
        self.assertEqual(task.assigned_robot, "R-1")
        self.assertTrue(task.picked_up)
        sim.step()
        self.assertTrue(task.completed)
        self.assertIsNone(sim.robots["R-1"].task_id)

    def test_robot_standing_on_list_spelled_pickup_collects(self) -> None:
        # An already-bound robot starts on the pickup cell; the task spells it
        # as a list while the robot position is a tuple. The equality must
        # still hold so the goods are collected.
        robot = Robot("R-1", (1, 0), [(2, 0)], task_id="T-1")
        task = Task(
            "T-1", [1, 0], [2, 0], assigned_robot="R-1"
        )
        sim = make_sim([robot], [task], grid=GridMap(3, 1))
        sim.step()
        self.assertTrue(task.picked_up)
        sim.step()
        self.assertTrue(task.completed)

    def test_caller_lists_are_not_mutated_by_construction(self) -> None:
        pickup, dropoff = [1, 0], [2, 0]
        task = Task("T-1", pickup, dropoff)
        make_sim([Robot("R-1", (0, 0))], [task], grid=GridMap(3, 1))
        self.assertEqual(pickup, [1, 0])
        self.assertEqual(dropoff, [2, 0])
        self.assertIsNot(task.pickup, pickup)
        self.assertIsNot(task.dropoff, dropoff)

    def test_mutating_caller_lists_after_creation_changes_nothing(self) -> None:
        pickup, dropoff = [1, 0], [2, 0]
        task = Task("T-1", pickup, dropoff)
        sim = make_sim([Robot("R-1", (0, 0))], [task], grid=GridMap(3, 1))
        pickup[0] = 9
        pickup.append(5)
        dropoff[1] = 8
        self.assertEqual(sim.tasks["T-1"].pickup, (1, 0))
        self.assertEqual(sim.tasks["T-1"].dropoff, (2, 0))
        self.assertEqual(task.pickup, (1, 0))
        self.assertEqual(task.dropoff, (2, 0))


class RuleAppliesToEveryTaskStateTests(unittest.TestCase):
    def test_waiting_task_checked(self) -> None:
        expect_rejected(
            self, [Task("T-1", [1.0, 0], (2, 0))], "T-1", "pickup",
            robots=[Robot("R-1", (0, 0))], grid=GridMap(3, 1),
        )

    def test_assigned_task_heading_to_pickup_checked(self) -> None:
        task = Task(
            "T-1", [True, 0], (2, 0), assigned_robot="R-1"
        )
        robot = Robot("R-1", (0, 0), [(1, 0)], task_id="T-1")
        expect_rejected(
            self, [task], "T-1", "pickup",
            robots=[robot], grid=GridMap(3, 1),
        )

    def test_loaded_task_checked(self) -> None:
        task = Task(
            "T-1", (1, 0), [2.0, 0], assigned_robot="R-1", picked_up=True
        )
        robot = Robot("R-1", (0, 0), task_id="T-1")
        expect_rejected(
            self, [task], "T-1", "dropoff",
            robots=[robot], grid=GridMap(3, 1),
        )

    def test_completed_historical_task_checked(self) -> None:
        task = Task(
            "T-1",
            (1, 0),
            [False, 0],
            assigned_robot="R-gone",
            picked_up=True,
            completed=True,
        )
        expect_rejected(
            self, [task], "T-1", "dropoff",
            robots=[Robot("R-other", (0, 0))], grid=GridMap(3, 1),
        )

    def test_points_normalized_in_every_state(self) -> None:
        waiting = Task("TW", [0, 1], (0, 2))
        bound = Task(
            "TB", [1, 0], (2, 0), assigned_robot="R-1"
        )
        loaded = Task(
            "TL", [0, 0], (2, 0), assigned_robot="R-2", picked_up=True
        )
        done = Task(
            "TD", [0, 0], [1, 0],
            assigned_robot="R-gone", picked_up=True, completed=True,
        )
        robots = [
            Robot("R-1", (0, 0), [(1, 0)], task_id="TB"),
            Robot("R-2", (2, 0), task_id="TL"),
        ]
        sim = make_sim(
            robots, [waiting, bound, loaded, done], grid=GridMap(6, 4)
        )
        for task in sim.tasks.values():
            self.assertIsInstance(task.pickup, tuple)
            self.assertIsInstance(task.dropoff, tuple)


class AtomicRejectionTests(unittest.TestCase):
    def test_later_bad_point_leaves_earlier_objects_untouched(self) -> None:
        robot = Robot("R-1", [0, 0], [(0, 1)], task_id="T-1")
        first = Task(
            "T-1", [0, 1], [0, 2], assigned_robot="R-1", picked_up=True
        )
        bad = Task("T-2", (1, 0), True)
        with self.assertRaises(ValueError):
            make_sim([robot], [first, bad], grid=GridMap(6, 4))
        self.assertEqual(robot.position, [0, 0])
        self.assertIsInstance(robot.position, list)
        self.assertEqual(robot.route, [(0, 1)])
        self.assertEqual(robot.task_id, "T-1")
        self.assertEqual(robot.distance_travelled, 0)
        self.assertEqual(first.pickup, [0, 1])
        self.assertIsInstance(first.pickup, list)
        self.assertEqual(first.dropoff, [0, 2])
        self.assertEqual(first.assigned_robot, "R-1")
        self.assertTrue(first.picked_up)
        self.assertFalse(first.completed)
        self.assertEqual(bad.dropoff, True)

    def test_later_bad_pickup_leaves_earlier_dropoff_untouched(self) -> None:
        first = Task("T-1", [0, 1], [0, 2])
        bad = Task("T-2", [1.0, 0], (2, 0))
        with self.assertRaises(ValueError):
            make_sim([Robot("R-1", (0, 0))], [first, bad], grid=GridMap(6, 4))
        self.assertEqual(first.pickup, [0, 1])
        self.assertEqual(first.dropoff, [0, 2])
        self.assertIsInstance(first.pickup, list)
        self.assertIsInstance(first.dropoff, list)
        self.assertEqual(bad.pickup, [1.0, 0])

    def test_ownership_failure_leaves_task_coordinates_untouched(self) -> None:
        task = Task("T-1", [1, 1], [2, 2])
        robot = Robot("R-1", (0, 0), task_id="MISSING")
        with self.assertRaises(ValueError):
            make_sim([robot], [task], grid=GridMap(6, 6))
        self.assertEqual(task.pickup, [1, 1])
        self.assertEqual(task.dropoff, [2, 2])
        self.assertIsInstance(task.pickup, list)

    def test_route_failure_leaves_task_coordinates_untouched(self) -> None:
        task = Task("T-1", [1, 0], [2, 0])
        robot = Robot("R-1", (0, 0), [(9, 9)])
        with self.assertRaises(ValueError):
            make_sim([robot], [task], grid=GridMap(4, 4))
        self.assertEqual(task.pickup, [1, 0])
        self.assertEqual(task.dropoff, [2, 0])
        self.assertIsInstance(task.pickup, list)
        self.assertEqual(robot.route, [(9, 9)])

    def test_mileage_and_flags_survive_failed_batch(self) -> None:
        robot = Robot(
            "R-1", [2, 0], task_id="T-1", distance_travelled=7
        )
        task = Task(
            "T-1", [1, 0], (2, 0),
            assigned_robot="R-1", picked_up=True,
        )
        bad = Task("T-2", None, (0, 0))
        with self.assertRaises(ValueError):
            make_sim([robot], [task, bad], grid=GridMap(6, 4))
        self.assertEqual(robot.position, [2, 0])
        self.assertEqual(robot.distance_travelled, 7)
        self.assertEqual(robot.task_id, "T-1")
        self.assertTrue(task.picked_up)
        self.assertEqual(task.assigned_robot, "R-1")
        self.assertEqual(task.pickup, [1, 0])


class ReachabilityIsSeparateTests(unittest.TestCase):
    def test_well_formed_unreachable_task_keeps_waiting(self) -> None:
        # Validity of the coordinates says nothing about current reachability:
        # a pending task behind a wall simply is never selected, exactly as
        # before coordinate validation existed.
        wall = GridMap(3, 1, frozenset({(1, 0)}))
        task = Task("T-1", (2, 0), (0, 0))
        robot = Robot("R-1", (0, 0))
        sim = make_sim([robot], [task], grid=wall)
        sim.step()
        self.assertIsNone(task.assigned_robot)
        self.assertFalse(task.picked_up)
        self.assertEqual(robot.route, [])

    def test_loaded_task_with_closed_pickup_keeps_goods_and_owner(self) -> None:
        # The goods are already aboard; closing the pickup cell afterwards may
        # not strip the cargo or the binding, and the coordinates still pass.
        # The wall also leaves the dropoff unreachable, so the bound robot
        # simply waits without ever losing the task.
        grid = GridMap(3, 1, frozenset({(1, 0)}))
        robot = Robot("R-1", (2, 0), task_id="T-1")
        task = Task(
            "T-1", (1, 0), [0, 0], assigned_robot="R-1", picked_up=True
        )
        sim = make_sim([robot], [task], grid=grid)
        for _ in range(3):
            sim.step()
            self.assertTrue(task.picked_up)
            self.assertEqual(task.assigned_robot, "R-1")
            self.assertEqual(robot.task_id, "T-1")
            self.assertFalse(task.completed)
        self.assertEqual(task.pickup, (1, 0))
        self.assertEqual(task.dropoff, (0, 0))


class CheckpointConsistencyTests(unittest.TestCase):
    def test_list_points_round_trip_through_checkpoint(self) -> None:
        sim = make_sim(
            [Robot("R-1", (0, 0))],
            [Task("T-1", [1, 0], (2, 0))],
            grid=GridMap(3, 1),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "fleet.json")
            sim.save_checkpoint(path)
            restored = FleetSimulator.load_checkpoint(path)
        self.assertEqual(restored.tasks["T-1"].pickup, (1, 0))
        self.assertEqual(restored.tasks["T-1"].dropoff, (2, 0))

    def test_points_checkpoint_would_reject_never_enter_a_fleet(self) -> None:
        # Previously such a fleet saved fine and loading then failed; the
        # mismatch must now surface at construction instead.
        for pickup, dropoff in (
            ([True, 0], (1, 0)),
            ((0, 0), [1.0, 0]),
        ):
            with self.assertRaises(ValueError):
                make_sim(
                    [Robot("R-1", (0, 0))],
                    [Task("T-1", pickup, dropoff)],
                    grid=GridMap(3, 1),
                )


if __name__ == "__main__":
    unittest.main()
