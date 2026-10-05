"""Accepted initial-route input forms all execute identically.

The remaining route handed to ``FleetSimulator`` may be a list or a tuple, and
each waypoint may itself be a list or tuple of two plain integers (the forms
may even be mixed within one route). Creation validates all of them without
touching the caller's objects, and once the fleet is accepted every form is
kept in one canonical internal representation, so the supplied form must not
change the per-step positions, consumed waypoints, mileage, replay frames or
task results. A tuple route used to fail with an ``AttributeError`` only after
the robot's first move had already changed its position and mileage, and a
list coordinate used to raise ``TypeError`` as soon as the cell was used as a
dict key -- both forms had passed creation validation.
"""

import copy
import unittest

from warehouse_fleet import FleetSimulator, GridMap, Robot, Task


# (0, 0) -> (1, 0) -> (2, 0) -> (2, 1): the same walk expressed with every
# accepted combination of outer container and coordinate container.
ROUTE_FORMS = {
    "list of tuples": [(1, 0), (2, 0), (2, 1)],
    "list of lists": [[1, 0], [2, 0], [2, 1]],
    "tuple of tuples": ((1, 0), (2, 0), (2, 1)),
    "tuple of lists": ([1, 0], [2, 0], [2, 1]),
    "mixed pair types": ([1, 0], (2, 0), [2, 1]),
}


def expected_run(simulator: FleetSimulator) -> list[tuple]:
    positions = []
    for _ in range(3):
        event = simulator.step()
        positions.append(
            (
                tuple(event["robots"]["R-1"]),
                list(simulator.robots["R-1"].route),
                simulator.robots["R-1"].distance_travelled,
                list(event["moved"]),
            )
        )
    return positions


class RouteFormEquivalenceTests(unittest.TestCase):
    def reference(self):
        simulator = FleetSimulator(
            GridMap(4, 3),
            [Robot("R-1", (0, 0), route=[(1, 0), (2, 0), (2, 1)])],
            [],
        )
        return simulator, expected_run(simulator)

    def test_every_form_reaches_the_same_cells_in_the_same_order(self) -> None:
        _, reference_run = self.reference()
        for label, route in ROUTE_FORMS.items():
            with self.subTest(form=label):
                simulator = FleetSimulator(
                    GridMap(4, 3), [Robot("R-1", (0, 0), route=route)], []
                )
                self.assertEqual(expected_run(simulator), reference_run)
                self.assertEqual(simulator.robots["R-1"].position, (2, 1))
                self.assertEqual(simulator.robots["R-1"].route, [])
                self.assertEqual(simulator.robots["R-1"].distance_travelled, 3)

    def test_every_step_consumes_only_the_walked_cell(self) -> None:
        for label, route in ROUTE_FORMS.items():
            with self.subTest(form=label):
                simulator = FleetSimulator(
                    GridMap(4, 3), [Robot("R-1", (0, 0), route=route)], []
                )
                expected_remaining = [
                    [(2, 0), (2, 1)],
                    [(2, 1)],
                    [],
                ]
                expected_positions = [(1, 0), (2, 0), (2, 1)]
                for tick in range(3):
                    event = simulator.step()
                    robot = simulator.robots["R-1"]
                    self.assertEqual(robot.position, expected_positions[tick])
                    self.assertEqual(robot.route, expected_remaining[tick])
                    self.assertEqual(robot.distance_travelled, tick + 1)
                    # One replay frame per step, matching the new position.
                    self.assertEqual(len(simulator.replay), tick + 1)
                    self.assertEqual(
                        tuple(event["robots"]["R-1"]), expected_positions[tick]
                    )
                    self.assertEqual(event["tick"], tick + 1)

    def test_every_form_records_identical_replay_frames(self) -> None:
        _, reference_run = self.reference()
        reference_simulator = FleetSimulator(
            GridMap(4, 3),
            [Robot("R-1", (0, 0), route=[(1, 0), (2, 0), (2, 1)])],
            [],
        )
        for _ in range(3):
            reference_simulator.step()
        for label, route in ROUTE_FORMS.items():
            with self.subTest(form=label):
                simulator = FleetSimulator(
                    GridMap(4, 3), [Robot("R-1", (0, 0), route=route)], []
                )
                for _ in range(3):
                    simulator.step()
                self.assertEqual(simulator.replay, reference_simulator.replay)

    def test_creation_never_drives_or_records_time(self) -> None:
        for label, route in ROUTE_FORMS.items():
            with self.subTest(form=label):
                caller_route = copy.deepcopy(route)
                simulator = FleetSimulator(
                    GridMap(4, 3), [Robot("R-1", (0, 0), route=route)], []
                )
                self.assertEqual(simulator.tick, 0)
                self.assertEqual(simulator.replay, [])
                self.assertEqual(simulator.robots["R-1"].position, (0, 0))
                self.assertEqual(simulator.robots["R-1"].distance_travelled, 0)
                # The internal store is canonical regardless of what was sent.
                self.assertEqual(
                    simulator.robots["R-1"].route,
                    [(1, 0), (2, 0), (2, 1)],
                )
                self.assertIsInstance(simulator.robots["R-1"].route, list)
                # The caller's own containers were never mutated.
                self.assertEqual(route, caller_route)

    def test_empty_tuple_means_no_remaining_waypoints(self) -> None:
        simulator = FleetSimulator(
            GridMap(2, 1), [Robot("R-1", (0, 0), route=())], []
        )
        self.assertEqual(simulator.robots["R-1"].route, [])
        event = simulator.step()
        self.assertEqual(event["moved"], [])
        self.assertEqual(simulator.robots["R-1"].position, (0, 0))
        self.assertEqual(simulator.robots["R-1"].distance_travelled, 0)


class BoundTaskRouteFormTests(unittest.TestCase):
    def test_pickup_delivery_timing_is_identical_across_forms(self) -> None:
        def run(route):
            simulator = FleetSimulator(
                GridMap(4, 2),
                [Robot("R-1", (0, 0), route=route, task_id="T-1")],
                [
                    Task(
                        "T-1",
                        (2, 0),
                        (2, 1),
                        assigned_robot="R-1",
                    )
                ],
            )
            timeline = []
            for _ in range(3):
                simulator.step()
                task = simulator.tasks["T-1"]
                timeline.append(
                    (
                        simulator.robots["R-1"].position,
                        task.picked_up,
                        task.completed,
                        simulator.robots["R-1"].task_id,
                    )
                )
            return timeline

        reference = run([(1, 0), (2, 0), (2, 1)])
        for label, route in ROUTE_FORMS.items():
            with self.subTest(form=label):
                self.assertEqual(run(route), reference)
        # Goods collected on reaching (2, 0), delivery one step later, and the
        # robot keeps the task binding until completion.
        self.assertEqual(
            reference,
            [
                ((1, 0), False, False, "T-1"),
                ((2, 0), True, False, "T-1"),
                ((2, 1), True, True, None),
            ],
        )

    def test_already_loaded_route_forms_finish_the_same_way(self) -> None:
        def run(route):
            simulator = FleetSimulator(
                GridMap(3, 1),
                [Robot("R-1", (0, 0), route=route, task_id="T-1")],
                [
                    Task(
                        "T-1",
                        (0, 0),
                        (2, 0),
                        assigned_robot="R-1",
                        picked_up=True,
                    )
                ],
            )
            simulator.step()
            simulator.step()
            return (
                simulator.robots["R-1"].position,
                simulator.robots["R-1"].distance_travelled,
                simulator.tasks["T-1"].completed,
            )

        reference = run([(1, 0), (2, 0)])
        for label, route in {
            "list of tuples": [(1, 0), (2, 0)],
            "list of lists": [[1, 0], [2, 0]],
            "tuple of tuples": ((1, 0), (2, 0)),
            "tuple of lists": ([1, 0], [2, 0]),
            "mixed": ([1, 0], (2, 0)),
        }.items():
            with self.subTest(form=label):
                self.assertEqual(run(route), reference)
                self.assertEqual(reference, ((2, 0), 2, True))


class WaitingWithAcceptedFormsTests(unittest.TestCase):
    def test_wait_consumes_no_waypoint_or_mileage_with_list_coordinates(self) -> None:
        # A (list-coordinate route) queues directly behind B, whose own route
        # (also list coordinates) leaves (1, 0) later in the first tick.
        simulator = FleetSimulator(
            GridMap(3, 1),
            [
                Robot("A", (0, 0), route=[[1, 0], [2, 0]]),
                Robot("B", (1, 0), route=[[2, 0]]),
            ],
            [],
        )
        event = simulator.step()
        self.assertEqual(event["moved"], ["B"])
        follower = simulator.robots["A"]
        self.assertEqual(follower.position, (0, 0))
        self.assertEqual(follower.route, [(1, 0), (2, 0)])
        self.assertEqual(follower.distance_travelled, 0)
        self.assertEqual(simulator.status()["traffic_waits"], [])

    def test_parked_blocker_yields_for_a_tuple_route_requester(self) -> None:
        simulator = FleetSimulator(
            GridMap(3, 2),
            [
                Robot("A", (0, 0), route=((1, 0), (2, 0))),
                Robot("B", (1, 0)),
            ],
            [],
        )
        event = simulator.step()
        self.assertEqual(event["moved"], ["B", "A"])
        self.assertEqual(simulator.robots["A"].position, (1, 0))
        self.assertEqual(simulator.robots["A"].route, [(2, 0)])
        self.assertEqual(simulator.robots["A"].distance_travelled, 1)


class DetourAndRevisitFormsTests(unittest.TestCase):
    def test_legal_detour_and_revisits_are_kept_verbatim(self) -> None:
        # Leaves the axis, revisits earlier cells (including the start) and is
        # in no way shortest; the waypoints and their order must survive
        # canonicalization and execution cell by cell.
        route = ([0, 1], [1, 1], (1, 0), (0, 0), (1, 0), [2, 0])
        simulator = FleetSimulator(
            GridMap(3, 2), [Robot("R-1", (0, 0), route=route)], []
        )
        canonical = [(0, 1), (1, 1), (1, 0), (0, 0), (1, 0), (2, 0)]
        self.assertEqual(simulator.robots["R-1"].route, canonical)
        visited = []
        for _ in canonical:
            event = simulator.step()
            visited.append(tuple(event["robots"]["R-1"]))
        self.assertEqual(visited, canonical)
        self.assertEqual(simulator.robots["R-1"].distance_travelled, len(canonical))


class IllegalRouteRejectionAcrossFormsTests(unittest.TestCase):
    def test_illegal_list_and_tuple_coordinates_still_rejected(self) -> None:
        bad_routes = (
            [[True, 0]],
            [(1, False)],
            [[1, "0"]],
            [[1, 0, 0]],
            [(2,)],
            [5],
            ["(1, 0)"],
            [None],
            [[3, 0]],                       # jump
            [[1, 0], [1, 0]],               # consecutive repeat
            [(0, 0), (1, 0)],               # first cell is current
            [(1, 1)],                       # diagonal
            [[9, 0]],                       # out of bounds
            [[0, 1]],                       # obstacle below
        )
        grid = GridMap(6, 2, frozenset({(0, 1)}))
        for route in bad_routes:
            with self.subTest(route=route):
                with self.assertRaises(ValueError) as context:
                    FleetSimulator(grid, [Robot("R-1", (0, 0), route=route)], [])
                self.assertIn("R-1", str(context.exception))
        # The same offences in a tuple outer container reject just the same.
        for route in (((True, 0),), ((3, 0),), ((1, 1),), ((9, 0),)):
            with self.subTest(route=route):
                with self.assertRaises(ValueError) as context:
                    FleetSimulator(
                        GridMap(6, 2), [Robot("R-1", (0, 0), route=route)], []
                    )
                self.assertIn("R-1", str(context.exception))

    def test_later_robot_failure_leaves_everything_untouched(self) -> None:
        robots = [
            Robot("R-1", (0, 0), route=[[1, 0], [2, 0]], task_id="T-1"),
            Robot("R-2", (0, 1), route=((2, 1),)),
        ]
        tasks = [
            Task("T-1", (0, 0), (2, 0), assigned_robot="R-1", picked_up=True)
        ]
        robots_before = copy.deepcopy(robots)
        tasks_before = copy.deepcopy(tasks)
        with self.assertRaises(ValueError) as context:
            FleetSimulator(GridMap(4, 2), robots, tasks)
        self.assertIn("R-2", str(context.exception))
        # The already-validated robot, its task and both caller routes (list
        # and tuple alike) are exactly as supplied.
        self.assertEqual(robots, robots_before)
        self.assertEqual(tasks, tasks_before)
        self.assertEqual(robots[0].route, [[1, 0], [2, 0]])
        self.assertEqual(robots[1].route, ((2, 1),))
        self.assertTrue(all(isinstance(cell, list) for cell in robots[0].route))

    def test_error_at_route_end_does_not_truncate_earlier_waypoints(self) -> None:
        route = ([1, 0], [2, 0], [2, 2])  # final waypoint is a jump
        with self.assertRaises(ValueError) as context:
            FleetSimulator(
                GridMap(4, 3), [Robot("R-late", (0, 0), route=route)], []
            )
        self.assertIn("R-late", str(context.exception))
        self.assertEqual(route, ([1, 0], [2, 0], [2, 2]))


if __name__ == "__main__":
    unittest.main()
