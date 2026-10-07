"""Core immutable map and mutable fleet entities."""

from __future__ import annotations

from dataclasses import dataclass, field

Position = tuple[int, int]


@dataclass(frozen=True, slots=True)
class GridMap:
    """A fixed width/height grid with the obstacles captured at construction.

    ``obstacles`` accepts a set, frozenset or any other iterable of cells but
    always stores a fresh ``frozenset`` snapshot: keeping and mutating the
    original container afterwards never changes this map, and the stored
    attribute offers no mutation methods of its own.
    """

    width: int
    height: int
    obstacles: frozenset[Position] = frozenset()

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("map dimensions must be positive")
        # Snapshot the caller's obstacles as a fresh frozenset instead of
        # keeping the container it was built from. A map is the fixed layout
        # of its construction moment: a caller that keeps mutating its original
        # set afterwards -- adding, removing or clearing cells -- must never
        # edit an existing map, and the frozen type means the stored attribute
        # offers no add/remove/clear of its own. object.__setattr__ is required
        # because the dataclass itself is frozen. A set, frozenset or any other
        # iterable of cells is accepted, and building a map never writes back
        # to the caller's container; the same source reused for several maps
        # therefore leaves each map with the cells it saw at its own
        # construction.
        obstacles = frozenset(self.obstacles)
        if any(not self.contains(cell) for cell in obstacles):
            raise ValueError("obstacle lies outside the map")
        object.__setattr__(self, "obstacles", obstacles)

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

