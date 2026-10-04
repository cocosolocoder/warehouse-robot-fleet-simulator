"""Regression tests for the public ``shortest_path`` entry point.

The contract under protection is narrower and stricter than "a route that
eventually reaches the goal": on the current obstacle map the returned
sequence must be a *deterministic shortest* feasible route. Each test
therefore checks the whole coordinate sequence, not just its last cell and
its length:

* the result lists the cells after the start through the goal (the start
  itself is never included),
* every step moves one cell orthogonally onto an in-bounds, non-obstacle
  cell, so no diagonal jump or phantom route past a map edge is possible,
* the length matches an independently computed minimum (a separate BFS
  written here, rather than trusting the implementation under test),
* when several routes tie at the minimum, the full sequence still follows
  the stable up / left / right / down direction priority and does not change
  when the obstacle set is built in another order.

Error cases are kept distinct from the normal zero-step route: an invalid or
unreachable endpoint raises ``ValueError``, while start == goal on a
traversable cell returns ``[]``.
"""

import unittest
from collections import deque

from warehouse_fleet import GridMap, shortest_path

Position = tuple[int, int]

# Direction order promised by the pathfinder: up, left, right, down.
_STRAIGHT_LINE_DIRECTIONS = ((0, -1), (-1, 0), (1, 0), (0, 1))


def minimum_distance(grid: GridMap, start: Position, goal: Position) -> int | None:
    """Independent BFS over the rectangle; returns None when unreachable.

    Implemented from scratch (plain queue, explicit four-direction moves,
    direct bounds/obstacle checks) so a regression in the production search
    cannot hide behind the same adjacency helper returning the wrong answer.
    Only distances are compared, so neighbour order is irrelevant here.
    """
    if not grid.traversable(start) or not grid.traversable(goal):
        return None
    distances = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return distances[(x, y)]
        for dx, dy in _STRAIGHT_LINE_DIRECTIONS:
            neighbor = (x + dx, y + dy)
            if neighbor in distances or not grid.traversable(neighbor):
                continue
            distances[neighbor] = distances[(x, y)] + 1
            queue.append(neighbor)
    return None


def assert_is_shortest_route(
    testcase: unittest.TestCase,
    grid: GridMap,
    start: Position,
    goal: Position,
    path: list[Position],
) -> None:
    """Structural + minimality checks shared by every success-case test."""
    testcase.assertIsInstance(path, list)
    testcase.assertNotIn(start, path, "the start cell must not be part of the route")
    testcase.assertEqual(path[-1], goal, "the route must finish exactly at the goal")
    testcase.assertEqual(
        len(path), len(set(path)), "a shortest route never revisits a cell"
    )
    previous = start
    for cell in path:
        testcase.assertTrue(
            grid.contains(cell), f"route leaves the map at {cell}"
        )
        testcase.assertNotIn(cell, grid.obstacles, f"route crosses obstacle {cell}")
        testcase.assertEqual(
            abs(cell[0] - previous[0]) + abs(cell[1] - previous[1]),
            1,
            f"route must step orthogonally one cell, not {previous} -> {cell}",
        )
        previous = cell
    expected = minimum_distance(grid, start, goal)
    testcase.assertIsNotNone(expected, "fixture endpoints must be reachable")
    testcase.assertEqual(
        len(path),
        expected,
        f"route length {len(path)} is not the minimum {expected}",
    )


class ShortestRouteTests(unittest.TestCase):
    def test_open_rectangle_pins_full_diagonal_route(self) -> None:
        # Six routes tie at length 4; straight-line distance alone would
        # accept any of them, so pin the complete tie-broken sequence.
        grid = GridMap(3, 3)
        path = shortest_path(grid, (0, 0), (2, 2))
        self.assertEqual(path, [(1, 0), (2, 0), (2, 1), (2, 2)])
        assert_is_shortest_route(self, grid, (0, 0), (2, 2), path)

    def test_route_goes_all_the_way_around_a_wall(self) -> None:
        # A wall at (1, 0) blocks the straight three-step route; the minimum
        # detour dips one row down and back up.
        grid = GridMap(3, 3, frozenset({(1, 0)}))
        path = shortest_path(grid, (0, 0), (2, 0))
        self.assertEqual(path, [(0, 1), (1, 1), (2, 1), (2, 0)])
        assert_is_shortest_route(self, grid, (0, 0), (2, 0), path)

    def test_long_detour_around_nearly_closed_wall(self) -> None:
        # A vertical wall leaves only the bottom edge open: the route must
        # travel to row 4, cross there, and come back - length 12. A route
        # that merely reaches the goal along some walk is not enough.
        grid = GridMap(5, 5, frozenset({(2, 0), (2, 1), (2, 2), (2, 3)}))
        path = shortest_path(grid, (0, 0), (4, 0))
        self.assertEqual(
            path,
            [
                (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
                (2, 4), (3, 4), (3, 3), (3, 2), (3, 1),
                (3, 0), (4, 0),
            ],
        )
        assert_is_shortest_route(self, grid, (0, 0), (4, 0), path)

    def test_dead_end_branches_are_not_entered(self) -> None:
        # Row 1 is open only at column 2. The pockets at (0,0)-(1,0) and
        # (3,0)-(4,0) are dead ends attached to the start cell: a search
        # that merely finds *a* feasible walk could wander in and back out.
        # The shortest route stays on column 2 straight down.
        grid = GridMap(5, 4, frozenset({(0, 1), (1, 1), (3, 1), (4, 1)}))
        path = shortest_path(grid, (2, 0), (2, 3))
        self.assertEqual(path, [(2, 1), (2, 2), (2, 3)])
        assert_is_shortest_route(self, grid, (2, 0), (2, 3), path)

    def test_wall_with_two_equal_gaps_uses_stable_choice(self) -> None:
        # Going around the wall above or below costs the same six steps.
        # The up-first priority must pick the route over the top, both when
        # travelling left-to-right and right-to-left.
        grid = GridMap(5, 3, frozenset({(2, 1)}))
        left_to_right = shortest_path(grid, (0, 1), (4, 1))
        self.assertEqual(
            left_to_right,
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1)],
        )
        assert_is_shortest_route(self, grid, (0, 1), (4, 1), left_to_right)
        right_to_left = shortest_path(grid, (4, 1), (0, 1))
        self.assertEqual(
            right_to_left,
            [(4, 0), (3, 0), (2, 0), (1, 0), (0, 0), (0, 1)],
        )
        assert_is_shortest_route(self, grid, (4, 1), (0, 1), right_to_left)

    def test_two_equal_gaps_around_long_wall_pin_full_sequence(self) -> None:
        # A wall spans the middle of a 7x3 map with symmetric gaps at x = 0
        # and x = 6. From a centred start both detours are equal length;
        # up + left must win and the complete sequence is pinned.
        grid = GridMap(7, 3, frozenset({(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)}))
        path = shortest_path(grid, (3, 0), (3, 2))
        self.assertEqual(
            path,
            [(2, 0), (1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (3, 2)],
        )
        assert_is_shortest_route(self, grid, (3, 0), (3, 2), path)

    def test_equal_choice_is_independent_of_obstacle_construction_order(self) -> None:
        # frozenset construction order must not perturb which equal-length
        # route is returned: compare the full sequence, not just its length.
        obstacles = [(1, 2), (2, 2), (3, 2), (4, 2), (5, 2)]
        reference = shortest_path(GridMap(7, 5, frozenset(obstacles)), (1, 1), (1, 3))
        for reordered in (
            list(reversed(obstacles)),
            [(5, 2), (1, 2), (4, 2), (2, 2), (3, 2)],
            set(obstacles),
            frozenset({(3, 2)}) | frozenset(o for o in obstacles if o != (3, 2)),
        ):
            with self.subTest(order=reordered):
                grid = GridMap(7, 5, frozenset(reordered))
                self.assertEqual(
                    shortest_path(grid, (1, 1), (1, 3)),
                    reference,
                    "tie-broken route changed with obstacle set construction order",
                )
        # The pinned sequence threads the left gap around the wall.
        self.assertEqual(reference, [(0, 1), (0, 2), (0, 3), (1, 3)])

    def test_single_row_has_no_route_past_its_edges(self) -> None:
        grid = GridMap(5, 1)
        path = shortest_path(grid, (0, 0), (4, 0))
        self.assertEqual(path, [(1, 0), (2, 0), (3, 0), (4, 0)])
        assert_is_shortest_route(self, grid, (0, 0), (4, 0), path)

    def test_single_column_has_no_route_past_its_edges(self) -> None:
        grid = GridMap(1, 5)
        path = shortest_path(grid, (0, 0), (0, 4))
        self.assertEqual(path, [(0, 1), (0, 2), (0, 3), (0, 4)])
        assert_is_shortest_route(self, grid, (0, 0), (0, 4), path)

    def test_single_row_blocked_in_the_middle_is_unreachable(self) -> None:
        grid = GridMap(5, 1, frozenset({(2, 0)}))
        # No row above or below exists: an edge cell offers no detour.
        with self.assertRaises(ValueError):
            shortest_path(grid, (0, 0), (4, 0))

    def test_single_column_blocked_in_the_middle_is_unreachable(self) -> None:
        grid = GridMap(1, 5, frozenset({(0, 2)}))
        with self.assertRaises(ValueError):
            shortest_path(grid, (0, 0), (0, 4))


class ZeroStepRouteTests(unittest.TestCase):
    def test_same_traversable_cell_returns_empty_list(self) -> None:
        grid = GridMap(3, 3, frozenset({(1, 1)}))
        # A traversable corner next to an obstacle still counts as arrived.
        self.assertEqual(shortest_path(grid, (0, 0), (0, 0)), [])
        self.assertEqual(shortest_path(grid, (2, 2), (2, 2)), [])

    def test_empty_route_is_not_raised_for_unreachable_goal(self) -> None:
        # Callers must be able to trust [] as "already there": a goal in a
        # sealed-off region raises rather than returning [] or a partial path.
        grid = GridMap(3, 3, frozenset({(1, 0), (1, 1), (0, 2), (1, 2)}))
        with self.assertRaises(ValueError):
            shortest_path(grid, (0, 0), (2, 2))


class InvalidEndpointTests(unittest.TestCase):
    def test_start_outside_map_raises_value_error(self) -> None:
        grid = GridMap(3, 3)
        for outside in [(-1, 0), (0, -1), (3, 0), (0, 3)]:
            with self.subTest(cell=outside):
                with self.assertRaises(ValueError):
                    shortest_path(grid, outside, (0, 0))

    def test_goal_outside_map_raises_value_error(self) -> None:
        grid = GridMap(3, 3)
        for outside in [(-1, 1), (1, -1), (3, 1), (1, 3)]:
            with self.subTest(cell=outside):
                with self.assertRaises(ValueError):
                    shortest_path(grid, (0, 0), outside)

    def test_endpoint_on_obstacle_raises_value_error(self) -> None:
        grid = GridMap(3, 3, frozenset({(1, 1), (2, 2)}))
        with self.assertRaises(ValueError):
            shortest_path(grid, (1, 1), (0, 0))
        with self.assertRaises(ValueError):
            shortest_path(grid, (0, 0), (2, 2))

    def test_identical_endpoints_still_must_be_traversable(self) -> None:
        # start == goal must not short-circuit the traversable checks.
        grid = GridMap(3, 3, frozenset({(1, 1)}))
        with self.assertRaises(ValueError):
            shortest_path(grid, (1, 1), (1, 1))
        with self.assertRaises(ValueError):
            shortest_path(grid, (3, 3), (3, 3))
        with self.assertRaises(ValueError):
            shortest_path(grid, (-1, 0), (-1, 0))
        # And the zero-step success case stays distinct from those errors.
        self.assertEqual(shortest_path(grid, (0, 0), (0, 0)), [])


if __name__ == "__main__":
    unittest.main()
