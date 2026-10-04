"""Regression tests for ``shortest_path``.

The function must return a *deterministic shortest feasible* route on the
current obstacle map, not merely a walk that reaches the goal. The checks here
therefore verify all of the following together:

* every step is an orthogonal unit move to an in-map, unblocked cell;
* the route runs from the cell after the start (the start is excluded) through
  the goal as its final cell;
* its length equals an independently computed shortest distance (an order
  agnostic breadth-first search), so a longer feasible detour fails;
* on maps with dead-end branches the branch never appears in the route;
* when several routes tie at the minimum length, the full coordinate sequence
  follows the stable up / left / right / down priority and does not depend on
  how the obstacle collection was constructed;
* same-cell endpoints return ``[]``, while invalid or unreachable endpoints
  raise ``ValueError`` so "cannot get there" is never confused with "already
  there".

Only the public ``GridMap`` / ``shortest_path`` entry points are used.
"""

import unittest
from collections import deque

from warehouse_fleet import GridMap, shortest_path

Position = tuple[int, int]

# up, left, right, down -- matches the documented stable neighbour priority.
PRIORITY_DIRECTIONS: tuple[Position, ...] = ((0, -1), (-1, 0), (1, 0), (0, 1))
# Neutral ordering for the reference oracle: the shortest distance must not
# depend on the implementation's neighbour priority.
NEUTRAL_DIRECTIONS: tuple[Position, ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))


def reference_distance(grid: GridMap, start: Position, goal: Position) -> int | None:
    """Shortest start->goal step count via an independent BFS, or ``None``.

    The oracle only asks the map which cells are traversable; its result is
    purely the BFS distance layer, so neighbour visit order cannot affect the
    value. ``None`` means at least one endpoint is invalid or unreachable.
    """
    if not grid.traversable(start) or not grid.traversable(goal):
        return None
    distance = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            return distance[current]
        x, y = current
        for dx, dy in NEUTRAL_DIRECTIONS:
            neighbor = (x + dx, y + dy)
            if grid.traversable(neighbor) and neighbor not in distance:
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
    return None


def assert_route_is_shortest(
    testcase: unittest.TestCase,
    grid: GridMap,
    start: Position,
    goal: Position,
    route: list[Position],
) -> None:
    """Feasibility, endpoint and strict optimality checks for one route."""
    testcase.assertNotIn(start, route, "the route must not include the start cell")
    testcase.assertEqual(
        len(route), len(set(route)), "a shortest route never revisits a cell"
    )
    walk = [start, *route]
    for cell in route:
        testcase.assertTrue(grid.contains(cell), f"route leaves the map at {cell}")
        testcase.assertTrue(
            grid.traversable(cell), f"route enters blocked/out-of-map cell {cell}"
        )
    for previous, cell in zip(walk, walk[1:]):
        testcase.assertEqual(
            abs(cell[0] - previous[0]) + abs(cell[1] - previous[1]),
            1,
            f"step from {previous} to {cell} is not an orthogonal unit move",
        )
    testcase.assertTrue(route, "the goal cell must be the route's final cell")
    testcase.assertEqual(route[-1], goal, "route does not finish at the goal")
    optimum = reference_distance(grid, start, goal)
    testcase.assertIsNotNone(optimum, "reference oracle could not reach the goal")
    testcase.assertEqual(
        len(route),
        optimum,
        "route reaches the goal but is longer than the shortest feasible route",
    )


class ShortestRouteGeometryTests(unittest.TestCase):
    """Maps where optimality means more than straight-line distance."""

    def test_route_round_wall_ignores_dead_end_pockets(self) -> None:
        # Vertical wall x == 2 for y0..3 leaves a single gap at (2, 4); the
        # only shortest route must travel down and around it. (1, 0) and
        # (3, 0) are dead-end pocket leaves off the corridor; entering either
        # can still reach the goal eventually, but only via a longer walk.
        #
        # # . # . #   y0   (1,0)/(3,0) pocket leaves, '#' blocked
        # S . # . G   y1   start (0,1), goal (4,1)
        # # . # . #   y2
        # . . # . .   y3
        # . . . . .   y4
        obstacles = frozenset(
            {(2, 0), (2, 1), (2, 2), (2, 3), (0, 0), (0, 2), (4, 0), (4, 2)}
        )
        grid = GridMap(5, 5, obstacles)
        start, goal = (0, 1), (4, 1)

        route = shortest_path(grid, start, goal)

        assert_route_is_shortest(self, grid, start, goal, route)
        self.assertEqual(
            route,
            [
                (1, 1), (1, 2), (1, 3), (1, 4),
                (2, 4),
                (3, 4), (3, 3), (3, 2), (3, 1),
                (4, 1),
            ],
        )
        # The route is a real detour (10 steps against a 4-step straight line),
        # and neither dead-end branch is used as a shortcut.
        manhattan = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
        self.assertGreater(len(route), manhattan)
        self.assertNotIn((1, 0), route)
        self.assertNotIn((3, 0), route)

    def test_winding_corridor_forces_route_away_from_the_goal_first(self) -> None:
        # The only feasible corridor from (0, 0) first heads *down* even
        # though the goal (4, 0) lies to the right and up; a straight-line
        # guess or a greedy "step toward the goal" walk cannot solve this.
        #
        # S # . . G    y0
        # . # . # .    y1
        # . . . # .    y2
        # # # # # .    y3
        obstacles = frozenset(
            {(1, 0), (1, 1), (3, 1), (3, 2),
             (0, 3), (1, 3), (2, 3), (3, 3)}
        )
        grid = GridMap(5, 4, obstacles)
        start, goal = (0, 0), (4, 0)

        route = shortest_path(grid, start, goal)

        assert_route_is_shortest(self, grid, start, goal, route)
        self.assertEqual(
            route,
            [(0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (3, 0), (4, 0)],
        )

    def test_feasible_but_longer_detour_is_rejected(self) -> None:
        # Wall x == 2 with equal gaps above and below the start/goal row. The
        # route through the upper gap is shortest; the walk that first detours
        # to the map edge is fully feasible but two steps longer and must not
        # be returned.
        obstacles = frozenset({(2, 0), (2, 2), (2, 4)})
        grid = GridMap(5, 5, obstacles)
        start, goal = (0, 2), (4, 2)

        route = shortest_path(grid, start, goal)

        assert_route_is_shortest(self, grid, start, goal, route)
        longer_walk = [
            (0, 1), (0, 0), (1, 0), (1, 1),
            (2, 1), (3, 1), (4, 1), (4, 2),
        ]
        self.assertTrue(
            all(grid.traversable(cell) for cell in longer_walk)
            and all(
                abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                for a, b in zip([start, *longer_walk], longer_walk)
            ),
            "test setup: the longer walk must be feasible",
        )
        self.assertEqual(longer_walk[-1], goal)
        self.assertLess(len(route), len(longer_walk))


class TieBreakingTests(unittest.TestCase):
    """Equal-length routes resolve to one stable full coordinate sequence."""

    def test_tied_routes_around_obstacles_take_the_upper_option(self) -> None:
        # Two equal-length gaps through a wall: going via the upper gap wins.
        wall = GridMap(5, 5, frozenset({(2, 0), (2, 2), (2, 4)}))
        upper_route = shortest_path(wall, (0, 2), (4, 2))
        assert_route_is_shortest(self, wall, (0, 2), (4, 2), upper_route)
        self.assertEqual(
            upper_route, [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (4, 2)]
        )

        # Same rule on a small detour around a single blocked centre cell.
        block = GridMap(3, 3, frozenset({(1, 1)}))
        around = shortest_path(block, (0, 1), (2, 1))
        assert_route_is_shortest(self, block, (0, 1), (2, 1), around)
        self.assertEqual(around, [(0, 0), (1, 0), (2, 0), (2, 1)])

    def test_pairwise_direction_priority_on_an_open_map(self) -> None:
        grid = GridMap(3, 3)
        cases = [
            ((0, 1), (1, 0), [(0, 0), (1, 0)]),   # up beats right
            ((1, 1), (0, 0), [(1, 0), (0, 0)]),   # up beats left
            ((1, 1), (0, 2), [(0, 1), (0, 2)]),   # left beats down
            ((1, 1), (2, 2), [(2, 1), (2, 2)]),   # right beats down
        ]
        for start, goal, expected in cases:
            with self.subTest(start=start, goal=goal):
                route = shortest_path(grid, start, goal)
                assert_route_is_shortest(self, grid, start, goal, route)
                self.assertEqual(route, expected)

    def test_tied_route_is_independent_of_obstacle_construction_order(self) -> None:
        obstacle_cells = {(2, 0), (2, 2), (2, 4)}
        orderings = [
            frozenset(obstacle_cells),
            frozenset(list(obstacle_cells)[::-1]),
            frozenset(sorted(obstacle_cells, reverse=True)),
            set(obstacle_cells),
        ]
        sequences = {
            tuple(shortest_path(GridMap(5, 5, obstacles), (0, 2), (4, 2)))
            for obstacles in orderings
        }
        self.assertEqual(
            sequences,
            {((0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (4, 2))},
        )


class NarrowMapTests(unittest.TestCase):
    """One-row / one-column maps must not find routes outside their strip."""

    def test_single_row_routes_stay_in_the_row(self) -> None:
        row = GridMap(5, 1)
        self.assertEqual(
            shortest_path(row, (0, 0), (4, 0)),
            [(1, 0), (2, 0), (3, 0), (4, 0)],
        )
        self.assertEqual(
            shortest_path(row, (4, 0), (0, 0)),
            [(3, 0), (2, 0), (1, 0), (0, 0)],
        )
        # A block past the goal is irrelevant; an edge cell has no neighbour
        # outside the map even when no obstacle marks the boundary.
        blocked_beyond = GridMap(5, 1, frozenset({(4, 0)}))
        self.assertEqual(shortest_path(blocked_beyond, (0, 0), (2, 0)),
                         [(1, 0), (2, 0)])
        with self.assertRaises(ValueError):
            shortest_path(GridMap(5, 1, frozenset({(2, 0)})), (0, 0), (4, 0))

    def test_single_column_routes_stay_in_the_column(self) -> None:
        column = GridMap(1, 5)
        self.assertEqual(
            shortest_path(column, (0, 0), (0, 4)),
            [(0, 1), (0, 2), (0, 3), (0, 4)],
        )
        self.assertEqual(
            shortest_path(column, (0, 4), (0, 0)),
            [(0, 3), (0, 2), (0, 1), (0, 0)],
        )
        blocked_beyond = GridMap(1, 5, frozenset({(0, 4)}))
        self.assertEqual(shortest_path(blocked_beyond, (0, 0), (0, 2)),
                         [(0, 1), (0, 2)])
        # A blocked column cannot be routed around "sideways" outside the map.
        with self.assertRaises(ValueError):
            shortest_path(GridMap(1, 5, frozenset({(0, 2)})), (0, 0), (0, 4))


class ZeroStepAndFailureTests(unittest.TestCase):
    """Zero-step success and the failure cases must stay clearly distinct."""

    def test_same_traversable_endpoint_returns_empty_route(self) -> None:
        self.assertEqual(shortest_path(GridMap(3, 3), (1, 1), (1, 1)), [])
        self.assertEqual(shortest_path(GridMap(1, 1), (0, 0), (0, 0)), [])

    def test_endpoint_outside_map_raises(self) -> None:
        grid = GridMap(3, 3)
        cases = [
            ((-1, 0), (2, 2)),
            ((3, 0), (2, 2)),
            ((0, 0), (0, 3)),
            ((0, 0), (3, 3)),
            # Even identical endpoints are validated: no shortcut past checks.
            ((3, 3), (3, 3)),
            ((-1, -1), (-1, -1)),
        ]
        for start, goal in cases:
            with self.subTest(start=start, goal=goal):
                with self.assertRaises(ValueError):
                    shortest_path(grid, start, goal)

    def test_endpoint_on_obstacle_raises(self) -> None:
        with self.assertRaises(ValueError):
            shortest_path(GridMap(3, 3, frozenset({(0, 0)})), (0, 0), (2, 2))
        with self.assertRaises(ValueError):
            shortest_path(GridMap(3, 3, frozenset({(2, 2)})), (0, 0), (2, 2))
        with self.assertRaises(ValueError):
            shortest_path(GridMap(3, 3, frozenset({(1, 1)})), (1, 1), (1, 1))

    def test_valid_but_disconnected_endpoints_raise(self) -> None:
        # Full wall splits the rectangle into two regions.
        wall = GridMap(5, 5, frozenset({(2, y) for y in range(5)}))
        with self.assertRaises(ValueError):
            shortest_path(wall, (0, 0), (4, 0))
        with self.assertRaises(ValueError):
            shortest_path(wall, (1, 1), (3, 1))
        # The same situation on a narrow strip: blocked ahead, nowhere to go.
        with self.assertRaises(ValueError):
            shortest_path(GridMap(5, 1, frozenset({(2, 0)})), (0, 0), (4, 0))
        with self.assertRaises(ValueError):
            shortest_path(GridMap(1, 5, frozenset({(0, 2)})), (0, 0), (0, 4))

    def test_unreachable_is_not_reported_as_a_zero_step_arrival(self) -> None:
        grid = GridMap(3, 3, frozenset({(1, 1)}))
        # The genuinely-already-there case returns [] ...
        self.assertEqual(shortest_path(grid, (0, 0), (0, 0)), [])
        # ... and no disconnected query may ever resolve to that same value.
        sealed = GridMap(5, 5, frozenset({(2, y) for y in range(5)}))
        with self.assertRaises(ValueError):
            shortest_path(sealed, (0, 0), (4, 0))


if __name__ == "__main__":
    unittest.main()
