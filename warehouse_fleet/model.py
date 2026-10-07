"""Core immutable map and mutable fleet entities."""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field

Position = tuple[int, int]


def _is_plain_int(value: object) -> bool:
    """Plain integers only: booleans are rejected despite being Python ints."""
    return isinstance(value, int) and not isinstance(value, bool)


def _describe_obstacle(value: object) -> str:
    """Stable, non-throwing rendering of one rejected obstacle entry."""
    try:
        return repr(value)
    except Exception:
        return type(value).__name__


@dataclass(frozen=True, slots=True)
class GridMap:
    width: int
    height: int
    obstacles: frozenset[Position] = frozenset()

    def __post_init__(self) -> None:
        # Dimensions must be plain positive integers. Nothing is coerced:
        # floats (even whole-valued ones), booleans, strings and None are
        # rejected at creation, because such a grid could be simulated yet
        # would never round-trip through a checkpoint -- the error must be
        # reported here, not surfaced as an unrecoverable saved run.
        if not _is_plain_int(self.width):
            raise ValueError(
                "map width must be a positive integer (booleans and floats are not accepted)"
            )
        if not _is_plain_int(self.height):
            raise ValueError(
                "map height must be a positive integer (booleans and floats are not accepted)"
            )
        if self.width <= 0:
            raise ValueError("map width must be greater than zero")
        if self.height <= 0:
            raise ValueError("map height must be greater than zero")

        # Validate every entry before snapshotting anything so a failed
        # construction never leaves behind a partially accepted obstacle set,
        # and never mutates the caller's collection: only ``frozenset`` is
        # invoked on the input, and the validated copy is published once.
        # A bare non-iterable (None, a single number, ...) fails the
        # container check instead of leaking an ``isinstance``/iteration
        # TypeError; shape and component errors name the offending entry.
        if not isinstance(self.obstacles, AbstractSet):
            raise ValueError(
                "map obstacles must be a set of (x, y) integer pairs, got "
                f"{_describe_obstacle(self.obstacles)!s}"
            )
        validated: set[Position] = set()
        for cell in self.obstacles:
            if (
                isinstance(cell, (str, bytes))
                or not isinstance(cell, (tuple, list))
                or len(cell) != 2
            ):
                raise ValueError(
                    f"invalid obstacle {_describe_obstacle(cell)!s}: "
                    "each obstacle must be an (x, y) pair of two integers"
                )
            x, y = cell
            if not (_is_plain_int(x) and _is_plain_int(y)):
                raise ValueError(
                    f"invalid obstacle {_describe_obstacle(cell)!s}: "
                    "obstacle coordinates must be plain integers, not booleans, "
                    "floats or other types"
                )
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError(
                    f"obstacle ({x}, {y}) lies outside the map "
                    f"(width {self.width}, height {self.height})"
                )
            validated.add((x, y))

        # Snapshot the caller's obstacle collection into a frozenset the
        # moment the map is created: a map keeps the content it was built
        # with, so mutating (or clearing) the original set afterwards never
        # rewrites this map, and the declared frozenset type genuinely holds
        # -- the map itself is never edited through the caller's container,
        # and construction never mutates the caller's collection either.
        object.__setattr__(self, "obstacles", frozenset(validated))

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

