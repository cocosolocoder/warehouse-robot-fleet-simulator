"""Core immutable map and mutable fleet entities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

Position = tuple[int, int]


def _is_plain_int(value: object) -> bool:
    """Whether *value* is a plain integer and not a boolean.

    Booleans are integers in Python, yet they must never act as grid
    dimensions or obstacle coordinates, and a float -- even one equal to an
    integer such as ``1.0`` -- is never silently converted. This mirrors the
    strict integer rule checkpoint loading applies to its map definitions, so
    a map built through the public Python interface can never carry values the
    checkpoint loader would later reject.
    """
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class GridMap:
    width: int
    height: int
    obstacles: frozenset[Position] = frozenset()

    def __post_init__(self) -> None:
        # Validate everything before any value commits: a rejected dimension
        # or obstacle raises before the snapshot below, so a failed request
        # never leaves a map holding only the cells checked before the bad one.
        if not _is_plain_int(self.width) or self.width <= 0:
            raise ValueError(
                f"grid width must be a positive integer, got {self.width!r}"
            )
        if not _is_plain_int(self.height) or self.height <= 0:
            raise ValueError(
                f"grid height must be a positive integer, got {self.height!r}"
            )
        cells = self._validate_obstacles(self.obstacles, self.width, self.height)
        # Snapshot the caller's obstacle collection into a frozenset the
        # moment the map is created: a map keeps the content it was built
        # with, so mutating (or clearing) the original set afterwards never
        # rewrites this map, and the declared frozenset type genuinely holds
        # -- the map itself is never edited through the caller's container,
        # and construction never mutates the caller's collection either.
        object.__setattr__(self, "obstacles", frozenset(cells))

    @staticmethod
    def _validate_obstacles(
        raw_obstacles: object, width: int, height: int
    ) -> list[Position]:
        """Validate *raw_obstacles* as in-bounds pairs of two plain integers.

        Each obstacle must be a tuple of exactly two plain integers lying
        inside the map:

        * a boolean component is rejected even though ``bool`` is an ``int``
          (``(True, 0)`` never names a cell), and a float component is never
          coerced (``(1.0, 0)`` is rejected even though it equals ``(1, 0)``);
        * strings, nulls, single numbers and lists used as *one obstacle* are
          rejected as that entry, as are pairs with missing or extra
          components -- the error names the offending obstacle (and its index
          in an ordered input) instead of leaking an unpacking or hashing
          :class:`TypeError`;
        * a cell outside ``0 <= x < width`` / ``0 <= y < height`` rejects the
          whole map -- it is never truncated, dropped or merely ignored.

        The outer collection may be a ``set``, ``frozenset``, ``list`` or
        ``tuple`` (or any other non-string iterable); an empty collection
        means no obstacles and a coordinate repeated by the caller counts
        once. Iteration is read-only -- nothing is inserted into the caller's
        container -- and results commit only once every entry has passed, so
        construction never mutates the caller's collection, on success or on
        failure. A set has no meaningful entry order, so the optional index
        hint is only added for list/tuple inputs.
        """
        if isinstance(raw_obstacles, (str, bytes)) or not isinstance(
            raw_obstacles, Iterable
        ):
            raise ValueError(
                "grid obstacles must be an iterable of (x, y) integer tuple "
                f"pairs, got {raw_obstacles!r}"
            )
        ordered = isinstance(raw_obstacles, (list, tuple))
        cells: list[Position] = []
        seen: set[Position] = set()
        for index, entry in enumerate(raw_obstacles):
            where = f" (obstacle {index})" if ordered else ""
            if not isinstance(entry, tuple):
                # Lists, strings, nulls and single numbers are not coordinate
                # tuples; naming the value (and its position in an ordered
                # input) lets the caller identify the bad obstacle directly.
                raise ValueError(
                    f"grid obstacle {entry!r}{where} must be an (x, y) "
                    "integer tuple pair"
                )
            if len(entry) != 2 or not all(
                _is_plain_int(component) for component in entry
            ):
                raise ValueError(
                    f"grid obstacle {entry!r}{where} must hold exactly two "
                    "plain integers -- booleans and floats are not accepted"
                )
            x, y = entry
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"grid obstacle ({x}, {y}){where} lies outside the "
                    f"{width}x{height} map: x must be in [0, {width - 1}] "
                    f"and y in [0, {height - 1}]"
                )
            if entry not in seen:
                seen.add(entry)
                cells.append(entry)
        return cells

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
