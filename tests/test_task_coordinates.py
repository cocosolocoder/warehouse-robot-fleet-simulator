"""Validation of task pickup/dropoff points on directly constructed fleets.

Each task handed to ``FleetSimulator`` carries a ``pickup`` and a ``dropoff``
point. Construction must apply the same strict coordinate rule robot
positions and route waypoints use: each point is a list or tuple of exactly
two plain integers, the two spellings may be mixed, and an accepted point
becomes a fresh independent integer tuple. Illegal values are rejected at
construction (never rounded or coerced), with the error naming the task and
whether its pickup or dropoff is wrong. Nothing is committed until the whole
batch passes, so a malformed point on a later task -- or any ownership or
route failure -- leaves every caller object untouched.
"""

import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


GRID = GridMap(6, 3, frozenset({(3, 2)}))


def make_sim(robots=(), tasks=(), grid=None):
    return FleetSimulator(
        GRID if grid is None else grid, list(robots), list(tasks)
    )


def expect_rejected(testcase, task, *fragments, robots=(), tasks_extra=(), grid=None):
    with testcase.assertRaises(ValueError) as context:
        make_sim(robots, [task, *tasks_extra], grid)
    message = str(context.exception)
    for fragment in fragments:
        testcase.assertIn(fragment, message)
    return message


class MalformedTaskPointTests(unittest.TestCase):
    def test_pickup_booleans_floats_and_scalars_are_rejected(self) -> None:
        for bad in (True, False, 1, 1.0, 0.0, "00", None, {1: 2}, 5):
            expect_rejected(
                self, Task("T-1", bad, (2, 0)), "T-1", "pickup"
            )

    def test_dropoff_booleans_floats_and_scalars_are_rejected(self) -> None:
        for bad in (True, 1.0, 0.0, "xy", None, [1]):
            expect_rejected(
                self, Task("T-1", (0, 0), bad), "T-1", "dropoff"
            )

    def test_wrong_lengths_are_rejected(self) -> None:
        for bad in ([1], (1,), [1, 2, 3], (1, 2, 3), [], ()):
            expect_rejected(self, Task("T-1", bad, (2, 0)), "T-1", "pickup")
            expect_rejected(self, Task("T-1", (0, 0), bad), "T-1", "dropoff")

    def test_non_integer_components_are_rejected_without_coercion(self) -> None:
        for bad in (
            [True, 0],
            [0, False],
            (True, False),
            [1.0, 2],
            [1, 2.0],
            ["1", 2],
            [1, "2"],
            [None, 0],
        ):
            expect_rejected(self, Task("T-1", bad, (2, 0)), "T-1", "pickup")
            expect_rejected(self, Task("T-1", (0, 0), bad), "T-1", "dropoff")

    def test_a_float_that_looks_like_an_integer_is_still_rejected(self) -> None:
        expect_rejected(
            self, Task("T-1", [1.0, 0], (2, 0)), "T-1", "pickup"
        )

    def test_rejection_distinguishes_pickup_from_dropoff(self) -> None:
        message = expect_rejected(
            self, Task("T-7", (0, 0), [True, 0]), "T-7", "dropoff"
        )
        self.assertNotIn("pickup coordinates", message)
        message = expect_rejected(
            self, Task("T-8", [1.0, 0], (2, 0)), "T-8", "pickup"
        )
        self.assertNotIn("dropoff coordinates", message)

    def test_error_is_value_error_not_type_error(self) -> None:
        for bad in ([True, 0], [1.0, 0], [0, 0, 0], None, "00"):
            with self.assertRaises(ValueError):
                make_sim(tasks=[Task("T-1", bad, (2, 0))])

    def test_points_are_validated_in_every_task_state(self) -> None:
        # Waiting (unassigned), en route, already loaded and completed tasks
        # all have to pass the point rule.
        loaded_robot = Robot("R-1", (0, 0), task_id="T-load")
        loaded_bad = Task(
            "T-load", [True, 0], (2, 0), assigned_robot="R-1", picked_up=True
        )
        expect_rejected(self, loaded_bad, "T-load", "pickup", robots=[loaded_robot])
        done_bad = Task(
            "T-done", (0, 0), [1.0, 0],
            assigned_robot="R-9", picked_up=True, completed=True,
        )
        expect_rejected(self, done_bad, "T-done", "dropoff")


class TaskPointNormalizationTests(unittest.TestCase):
    def test_list_and_tuple_points_become_integer_tuples(self) -> None:
        task = Task("T-1", [1, 0], (2, 0))
        make_sim(tasks=[task])
        self.assertEqual(task.pickup, (1, 0))
        self.assertEqual(task.dropoff, (2, 0))
        self.assertIsInstance(task.pickup, tuple)
        self.assertIsInstance(task.dropoff, tuple)
        self.assertIs(type(task.pickup[0]), int)

    def test_spellings_may_be_mixed_between_and_within_tasks(self) -> None:
        first = Task("T-1", [1, 0], [2, 0])
        second = Task("T-2", (0, 1), (0, 2))
        make_sim(tasks=[first, second])
        self.assertEqual(first.pickup, (1, 0))
        self.assertEqual(first.dropoff, (2, 0))
        self.assertEqual(second.pickup, (0, 1))
        self.assertEqual(second.dropoff, (0, 2))

    def test_normalized_point_is_independent_of_caller_list(self) -> None:
        pickup = [1, 0]
        dropoff = [2, 0]
        task = Task("T-1", pickup, dropoff)
        make_sim(tasks=[task])
        pickup.append(9)
        pickup[0] = 5
        dropoff[0] = 4
        self.assertEqual(task.pickup, (1, 0))
        self.assertEqual(task.dropoff, (2, 0))

    def test_list_and_tuple_points_name_the_same_cells(self) -> None:
        # A robot spelled with a tuple position standing on a list-spelled
        # pickup must collect, and a list-spelled dropoff must complete.
        robot = Robot("R-1", (1, 0))
        task = Task("T-1", [1, 0], [2, 0])
        sim = make_sim([robot], [task], grid=GridMap(6, 3))
        sim.step()
        self.assertTrue(task.picked_up)
        while not task.completed:
            sim.step()
        self.assertTrue(task.completed)
        self.assertIsNone(sim.robots["R-1"].task_id)


class BatchAtomicityTests(unittest.TestCase):
    def test_later_bad_point_leaves_earlier_objects_untouched(self) -> None:
        robot = Robot(
            "R-1", [0, 0], route=[(1, 0)], task_id="T-1", distance_travelled=4
        )
        good = Task(
            "T-1", [0, 0], [1, 0], assigned_robot="R-1", picked_up=True
        )
        bad = Task("T-2", (1.0, 0), (2, 0))
        with self.assertRaises(ValueError):
            make_sim([robot], [good, bad])
        self.assertEqual(robot.position, [0, 0])
        self.assertIsInstance(robot.position, list)
        self.assertEqual(robot.route, [(1, 0)])
        self.assertEqual(robot.task_id, "T-1")
        self.assertEqual(robot.distance_travelled, 4)
        self.assertEqual(good.pickup, [0, 0])
        self.assertIsInstance(good.pickup, list)
        self.assertEqual(good.dropoff, [1, 0])
        self.assertEqual(good.assigned_robot, "R-1")
        self.assertTrue(good.picked_up)
        self.assertFalse(good.completed)

    def test_later_ownership_failure_leaves_task_points_untouched(self) -> None:
        robot = Robot("R-2", (0, 1), task_id="T-missing")
        good = Task("T-9", [0, 0], [1, 0])
        with self.assertRaises(ValueError):
            make_sim([robot], [good])
        self.assertEqual(good.pickup, [0, 0])
        self.assertIsInstance(good.pickup, list)
        self.assertEqual(good.dropoff, [1, 0])
        self.assertIsInstance(good.dropoff, list)


class ReachabilityUnaffectedTests(unittest.TestCase):
    def test_unreachable_waiting_task_is_still_accepted(self) -> None:
        split = GridMap(3, 1, frozenset({(1, 0)}))
        task = Task("T-1", (2, 0), (0, 0))
        sim = make_sim(tasks=[task], grid=split)
        self.assertIsNone(sim.tasks["T-1"].assigned_robot)
        self.assertFalse(task.picked_up)
        self.assertEqual(task.pickup, (2, 0))

    def test_loaded_task_with_closed_pickup_keeps_goods_and_owner(self) -> None:
        split = GridMap(3, 1, frozenset({(1, 0)}))
        robot = Robot("R-1", (0, 0), task_id="T-1")
        task = Task(
            "T-1", [2, 0], [0, 0], assigned_robot="R-1", picked_up=True
        )
        sim = make_sim([robot], [task], grid=split)
        self.assertTrue(task.picked_up)
        self.assertEqual(task.assigned_robot, "R-1")
        self.assertEqual(robot.task_id, "T-1")


if __name__ == "__main__":
    unittest.main()
