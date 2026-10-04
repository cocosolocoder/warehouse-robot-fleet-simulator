"""Deterministic shortest-path search for four-way warehouse grids."""

from __future__ import annotations

from collections import deque

from .model import GridMap, Position, Task


def shortest_path(grid: GridMap, start: Position, goal: Position) -> list[Position]:
    """Return positions after *start* through *goal*, or raise when unreachable."""
    if not grid.traversable(start) or not grid.traversable(goal):
        raise ValueError("path endpoint is not traversable")
    if start == goal:
        return []
    queue = deque([start])
    previous: dict[Position, Position | None] = {start: None}
    while queue:
        current = queue.popleft()
        for neighbor in grid.neighbors(current):
            if neighbor in previous:
                continue
            previous[neighbor] = current
            if neighbor == goal:
                queue.clear()
                break
            queue.append(neighbor)
    if goal not in previous:
        raise ValueError(f"no route from {start} to {goal}")
    path: list[Position] = []
    cursor = goal
    while cursor != start:
        path.append(cursor)
        parent = previous[cursor]
        if parent is None:
            raise RuntimeError("broken predecessor chain")
        cursor = parent
    path.reverse()
    return path


def plan_task_route(grid: GridMap, task: Task, origin: Position) -> list[Position] | None:
    """Shortest route from *origin* that fulfils *task*, or None if unreachable.

    This is the single pickup/dropoff planning rule used by first assignment,
    rerouting after map edits, and checkpoint pause validation alike. Before
    the goods are collected the route runs origin -> pickup -> dropoff; a
    robot already standing on the pickup cell contributes an empty first leg
    (planning never changes the pickup or completion state itself). Once the
    goods are on board the route goes straight to the dropoff and never
    returns to the pickup cell, even if that cell was closed afterwards. The
    route holds only the cells after *origin*, so when position, pickup and
    dropoff all coincide the empty route is the valid result -- not to be
    confused with an unreachable task, which yields None.
    """
    try:
        if task.picked_up:
            # Goods are already on board: a reclosed pickup cell must not
            # pull the robot back.
            return shortest_path(grid, origin, task.dropoff)
        to_pickup = [] if origin == task.pickup else shortest_path(grid, origin, task.pickup)
        return to_pickup + shortest_path(grid, task.pickup, task.dropoff)
    except ValueError:
        return None

