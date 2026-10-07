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
        # Snapshot the caller's obstacle collection into a frozenset the
        # moment the map is created: a map keeps the content it was built
        # with, so mutating (or clearing) the original set afterwards never
        # rewrites this map, and the declared frozenset type genuinely holds
        # -- the map itself is never edited through the caller's container,
        # and construction never mutates the caller's collection either.
        object.__setattr__(self, "obstacles", frozenset(self.obstacles))
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

