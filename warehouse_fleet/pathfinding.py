"""Deterministic shortest-path search for four-way warehouse grids."""

from __future__ import annotations

from collections import deque

from .model import GridMap, Position


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

