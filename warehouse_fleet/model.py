"""Core immutable map and mutable fleet entities."""

from __future__ import annotations

from dataclasses import dataclass, field

Position = tuple[int, int]


@dataclass(frozen=True, slots=True)
class GridMap:
    width: int
    height: int
    obstacles: frozenset[Position] = frozenset()

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("map dimensions must be positive")
        if any(not self.contains(cell) for cell in self.obstacles):
            raise ValueError("obstacle lies outside the map")

    def contains(self, position: Position) -> bool:
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def traversable(self, position: Position) -> bool:
        return self.contains(position) and position not in self.obstacles

    def neighbors(self, position: Position) -> tuple[Position, ...]:
        x, y = position
        candidates = ((x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1))
        return tuple(candidate for candidate in candidates if self.traversable(candidate))


@dataclass(slots=True)
class Task:
    task_id: str
    pickup: Position
    dropoff: Position
    assigned_robot: str | None = None
    picked_up: bool = False
    completed: bool = False
    paused: bool = False


@dataclass(slots=True)
class Robot:
    robot_id: str
    position: Position
    route: list[Position] = field(default_factory=list)
    task_id: str | None = None
    distance_travelled: int = 0

    @property
    def idle(self) -> bool:
        return self.task_id is None


@dataclass(frozen=True, slots=True)
class MapChange:
    """A single effective map modification applied between two ticks.

    ``tick`` is the number of ticks already executed when the change took
    effect (so a change made right after tick ``t`` has effective tick
    ``t``). ``added`` and ``removed`` hold the cells that actually became
    blocked or open: duplicate submissions and no-ops (adding an already
    blocked cell, removing an already open one) are filtered out, so a
    change is only recorded when at least one cell really flips.
    """

    tick: int
    added: frozenset[Position]
    removed: frozenset[Position]

