"""Task assignment, dynamic map edits and collision-aware tick execution.

Obstacles may be closed and reopened between ticks through
:meth:`FleetSimulator.modify_obstacles`. Every accepted edit is appended to a
deterministic change history and mirrored into the replay as a ``map_change``
event stamped with the tick at which it took effect. Tick frames themselves are
never rewritten, so historical frames keep the positions and completion
information recorded at the time.

Automatic yielding
------------------
When a robot's next waypoint is occupied by another robot, ``step()`` tries to
keep traffic flowing instead of waiting forever: a parked blocker (one with no
bound task *and* no remaining route) is asked to move onto a free side cell
that lies on no robot's planned route, and a busy robot facing a blocker that
cannot drive on may itself sidestep onto a free adjacent cell and replan its
(still shortest) route from there. A taskless robot that still carries a
remaining route is not parked: it drives that route cell by cell exactly like
a task-bound robot, is never pushed onto a side cell for someone else, and if
its own next cell is blocked it simply keeps its place and its unconsumed
waypoints -- an open side cell changes nothing. Only once such a route is
exhausted does the robot become a parked robot and yield by the side-cell
rule. Task assignment follows the same line: only a robot with no bound task
*and* no remaining route can receive a waiting task, so a taskless robot
still driving its preset route never has that route overwritten by a new
assignment -- being blocked by traffic along the way does not count as having
finished the route, and once the route runs out the robot joins the selection
at the next assignment pass, planned from wherever it stands then. A retreat that
only leads back to the same immovable blocker is never taken though: if the
replanned route immediately returns through the vacated cell, still passes the
blocker's cell and the blocker cannot use the opening to leave (a dead end or
corridor end with traffic parked beyond it), the robot simply waits -- extra
empty cells stretching out behind it change nothing. Nor does the blocker's
own side step automatically count as leaving: when that step only moves the
blocker onto a side cell whose replanned route returns it onto the corridor
to face the very same unyielding jam further ahead, the opening is no opening
and the retreat behind it is rejected as well. Yield moves count as
mileage, waiting does not. The task keeps its original robot, pickup and
completion rules are unchanged, and a robot paused by map unreachability never
takes part in yielding. If no safe side cell exists the robots simply wait:
time advances, nothing collides and nothing completes early. Waiting caused
purely by other robots is tracked per robot (consecutive ticks and the
blocking robot) and reported by :meth:`FleetSimulator.status` and
:meth:`FleetSimulator.metrics` as ``traffic_waits``, distinct from the
map-unreachability ``paused_tasks``. The record is judged against the
end-of-tick layout, not the moment the robot gave way: when a robot that was
asked to wait sees its next cell free at tick end (the blocker drove on or
made room later in the same tick), no wait is reported for that tick at all,
and ``blocked_by`` names whichever robot actually occupies the cell once the
tick settles. The counter clears the moment the robot moves again or the
blockage ends; an uninterrupted wait whose blocking robot merely changes
identity keeps accumulating with the new blocker, whereas a wait that was
interrupted and later recurs starts again at one. An accepted map edit
reconciles the records immediately: a robot paused by the new map or no
longer facing its recorded blocker on the replanned route loses its entry, an
entry still facing the same blocker keeps its count, and no edit ever creates
or increments one.

Initial task ownership
----------------------
A fleet can be created mid-work: a robot may already name its current task
with a remaining route, and a task may already be assigned or even loaded.
Construction validates these bindings up front instead of discovering them
while stepping. Every robot that names a task must name one that exists, is
still unfinished and is assigned back to that robot; every unfinished task
that names an owner must name an existing robot whose current task it is, and
a task with the goods collected may never lack that owner. One robot is
therefore never claimed by two unfinished tasks, and a robot may never be
bound to a completed task. A finished task is a fixed historical record and
must carry both the picked-up mark and the historical robot that finished it:
``assigned_robot`` of ``None`` means the record is missing, not an anonymous
completion, and the mark is never presumed -- not even when the task's pickup
point and dropoff are the same cell. Finished tasks only retain *historical*
ownership: their recorded robot need not be idle or name them back, may
already work a new task, may have finished several tasks, and may even be
absent from the fleet, so history never occupies a robot or enters new task
assignment. Unassigned, not-yet-collected tasks simply wait -- including ones
no robot can currently reach -- and are not an ownership error. Conflicts
raise :class:`ValueError` naming the robot or task and the contradiction
involved; nothing is rebound, unassigned or dropped, and the caller's objects
are left untouched. The very same ownership and completed-history rules
govern checkpoint loading, so a completion record one entry point accepts
can never be a record the other cannot reload.

Every robot's starting position is validated with the very same strict
coordinate rule its route waypoints use (:func:`_as_cell`): it must be a list
or tuple holding exactly two plain integers, and the two spellings may be
mixed freely within one batch. A boolean is never an integer here, and floats,
strings, missing or extra components and anything that is not a coordinate
pair all raise :class:`ValueError` naming the robot and its offending starting
position -- nothing is rounded, coerced or padded into a pair. An accepted
position is normalized to a fresh integer tuple (``[0, 0]`` and ``(0, 0)``
therefore create identical fleets that move and receive tasks identically),
and the copy means a caller that keeps mutating its original list after
construction can never change the created fleet. Positions must lie inside
the map and outside every obstacle, and no two robots may start on the same
cell: the comparison is made on the normalized coordinate values, so a list
and a tuple naming one cell still collide, and the error names both robots.
This covers every robot regardless of task state -- unbound, task-executing
and preset-route robots alike -- so a batch can never succeed by ignoring one
robot. Like the route check, position normalization commits nothing until the
entire batch (positions, ownership and routes) has passed: a failure on a
later robot's route or task binding leaves every earlier robot's original
position object, remaining route and task binding exactly as passed in.

Every task's pickup and dropoff obey that very same strict coordinate rule
(:func:`_as_cell`): each must be a list or tuple holding exactly two plain
integers, and the two spellings may be mixed freely within one batch or even
between a task's two points. A boolean is never an integer here -- not even as
one component -- and a float is never accepted, however neatly it prints
(``1.0`` included); strings, ``None``, missing or extra components and any
other non-coordinate input all raise :class:`ValueError` naming the task and
which of its points is wrong ("pickup" or "dropoff"), never rounded, coerced
or padded into a pair. An accepted point is normalized to a fresh integer
tuple, so ``[1, 0]`` and ``(1, 0)`` name one and the same cell -- a pickup
spelled one way can never be missed, nor a delivery fail to confirm, merely
because a robot's tuple position compares against a list -- and the copy
means a caller that keeps mutating its original coordinate lists after
construction can never change the adopted tasks. This is a shape rule only,
deliberately separate from reachability: no pickup or dropoff is checked
against the map here, so an unassigned task no robot can currently reach
keeps waiting exactly as before, and a loaded task whose pickup was later
closed keeps its goods and its owner. The rule covers every task regardless
of state -- waiting, en route to pickup, already carrying and already
completed alike. Like position normalization, it commits nothing until the
whole batch (task points, positions, ownership and routes) has passed: a
malformed point on a later task, or any later ownership or route failure,
leaves every earlier object -- coordinate containers, remaining routes,
bindings, mileage and pickup/completion marks -- exactly as passed in.

Every robot's remaining route is also validated as a walk on the given map,
whether or not the robot is bound to a task, and direct construction and
checkpoint recovery run the *same* :func:`_validate_route_walk` check, so the
walking rule is maintained in exactly one place. The route omits the robot's
current cell, so a non-empty route must begin with an up/down/left/right
neighbour of that cell and every later waypoint must be one orthogonal step
from the previous one; each waypoint must be in bounds and free of obstacles.
Diagonal steps, multi-cell jumps, two identical consecutive waypoints and a
first waypoint equal to the current position are all rejected, and waypoints
must be two plain integers (never booleans) in a list or tuple, with wrong
lengths or element types raising :class:`ValueError` rather than leaking
unpacking or hashing errors. Empty routes stay legal; a route need not be
shortest and may revisit cells. A route crossing another robot's current cell,
or several robots planning through the same cell, is not an error -- those
traffic conflicts are settled during execution. The direct interface keeps
its existing acceptance of a bound route's pickup/dropoff ordering and end
point; checkpoint loading additionally keeps its own stricter route checks
(including the task completion rules), its format and the map-paused
behaviour, under which a paused robot resumes with an empty route.

Direct construction always judges every waypoint against the map handed to
it, so a route crossing an initial obstacle is rejected no matter the robot's
task state. Checkpoint recovery owns exactly one relaxation, and only for a
version 2 document: a robot with no bound task keeps its remaining preset
waypoints when a later map edit closes the cell ahead -- it simply waits
before it -- so such a waypoint may be blocked on the saved map provided it
was traversable on the base map, some recorded change net-closes it and it is
still closed at save time. The set of cells this can possibly cover is
derived from the already validated ``base_grid`` plus ``map_changes``, never
from the route itself: a base-map obstacle and any blocked cell the history
cannot explain stay rejected, version 1 files (which carry no history) keep
the old rule, and a task-bound robot never qualifies. Out-of-bounds,
malformed and non-adjacent waypoints are rejected for every robot either
way. After reopening, the restored robot keeps driving the same cells in the
same order: waiting neither consumes a waypoint nor adds mileage, and the
route is never replanned into a shortcut.

Checkpoint file format
----------------------
``save_checkpoint`` writes versioned, UTF-8 encoded JSON. The current format
version is ``2``; version ``1`` files remain readable.

``version``
    Integer format version (``1`` or ``2``).
``grid``
    ``{"width": int, "height": int, "obstacles": [[x, y], ...]}`` describing
    the map dimensions and every currently blocked cell.
``tick``
    Non-negative integer: the number of ticks already executed. Saving and
    loading never advance this value.
``robots``
    List of robot states, each
    ``{"robot_id": str, "position": [x, y], "route": [[x, y], ...],
    "task_id": str | null, "distance_travelled": int}``. ``route`` holds the
    remaining waypoints after ``position`` (empty for a paused robot). For a
    taskless robot the route may head into a cell the change history has
    closed and the saved grid still shows blocked: the robot is simply waiting
    there with its preset waypoints intact, and loading restores that wait.
``tasks``
    List of task states, each
    ``{"task_id": str, "pickup": [x, y], "dropoff": [x, y],
    "assigned_robot": str | null, "picked_up": bool, "completed": bool}``.
    Completed tasks keep their historical ``assigned_robot``: a task saved as
    completed must also be marked picked up and name the robot that finished
    it (never null), and loading rejects a record missing either half with the
    same rule direct fleet construction enforces.
``replay``
    Ordered event list. Tick events are numbered consecutively from ``1``
    through ``tick`` and have the shape
    ``{"type": "tick", "tick": int, "moved": [robot_id, ...],
    "robots": {robot_id: [x, y], ...}, "completed": [task_id, ...]}``.
    ``completed`` is the cumulative set of tasks finished by the end of that
    tick: entries must name checkpoint tasks without repeats within one frame,
    a recorded completion must survive every later frame, and the final
    frame's set must equal the tasks saved with ``completed: true``.
    Map-change events sit between ticks (or after the last tick) and look like
    ``{"type": "map_change", "tick": int, "sequence": int,
    "added": [[x, y], ...], "removed": [[x, y], ...]}``. When ``tick`` is
    non-zero the final tick event's robot positions must match the current
    robot positions.
``map_changes`` (version 2 only)
    List mirroring the replay map-change events in the same order:
    ``{"tick": int, "sequence": int, "added": [...], "removed": [...]}``.
``base_grid`` (version 2 only)
    ``{"width", "height", "obstacles"}`` describing the map before any runtime
    edit. Replaying ``map_changes`` against it must reproduce the saved current
    ``grid``.
``traffic_waits`` (optional)
    List of ``{"robot_id": str, "blocked_by": [str, ...], "ticks": int}``
    entries recording how many consecutive ticks each robot has been held up
    purely by other robots. Absent in older files, which load with every
    wait counter at zero. Each entry must agree with the saved layout: the
    robot needs a remaining route and ``blocked_by`` must name exactly the
    robot occupying its next route cell -- a next cell nobody occupies, a
    listed blocker parked elsewhere, or extra names beside the real blocker
    all reject the document.

Version 1 documents store tick frames without a ``type`` field and have no
``map_changes`` or ``base_grid`` key; they load with their grid taken as the
baseline, as if no map edit ever happened.

The ordinary-tick-frame rules both versions share -- the replay container is
a list, every frame is an object, and tick frames are numbered consecutively
from 1 with a well-formed payload -- are owned by the single shared
:meth:`FleetSimulator._iter_replay_events` scan, so they are maintained in
exactly one place. Each version keeps its own handling on top: version 1
rejects non-tick frames and fills in the missing ``type`` on a copy, while
version 2 dispatches event types and enforces its map-change ordering.

Both map definitions are validated by the single shared
:meth:`FleetSimulator._load_map_definition` rule -- positive integer
dimensions, an obstacle list of integer ``[x, y]`` pairs inside the map --
with wording that names the map at fault, so a format or value rule is
maintained in exactly one place while each map keeps its own contents. The
cross-definition constraints (matching dimensions and a history that replays
``base_grid`` onto ``grid``) are checked separately and remain distinct
rejection reasons from a malformed definition.

Loading performs strict validation: malformed JSON, missing or wrongly typed
fields, unsupported versions, duplicate or inconsistent entities, out of
bounds/obstructed positions and routes, non-adjacent route steps, ownership
mismatches, routes that cannot finish the robot's bound task, traffic waits
that contradict the saved robot positions and remaining routes, broken replay
or map-change history, a completion history that contradicts the saved task
states, and a history that does not reproduce the saved grid all raise
:class:`ValueError`.

The single route-walk exception recovery allows is a preset-route robot that
is waiting out a dynamic closure. In a version 2 file a robot without a bound
task may keep waypoints that are blocked on the saved grid when the base map
had the cell open and the recorded change history explains its current,
still-effective closure (a cell closed and reopened before the save is open
again and therefore never covered). Such a robot was produced by normal
running -- closing a cell never strips a taskless robot of its remaining
route, the robot just waits before the cell -- so recovery restores it as is
and it drives on, same cells, same order, once the cell reopens. The
exemption cannot reach anything else: direct fleet construction has no
history and rejects routes through initial obstacles, a blocked waypoint the
history cannot account for (including every blocked waypoint in a version 1
document) is still rejected, and robots bound to a task never qualify --
their routes are replanned or their tasks paused at edit time and load under
the unchanged task rules. A robot standing on an obstacle and an out-of-map,
malformed or non-adjacent waypoint are rejected regardless of task state.

A task-bound route is judged against the task's pickup state, not just for
walkability. For every assigned, unfinished task that is not paused by map
unreachability, a non-empty remaining route must end at the dropoff, and when
the goods are not yet collected and the robot is not already standing on the
pickup cell, the route must still pass through that pickup cell -- passing the
dropoff early changes nothing as long as the route later collects and returns,
and avoidance detours need not match a replanned shortest route. An empty
route is accepted only when a loaded robot already stands at the dropoff, or
when position, pickup and dropoff all coincide; paused tasks keep their
empty-route exemption, and unassigned or completed tasks are not checked.

The recorded movement history is checked for physical plausibility, not just
for a matching final frame. In every tick frame each robot must stand on a
cell that is in bounds and traversable on the map of that moment, no two
robots may end a tick on the same cell, and from the second frame on a robot
may only have waited or stepped one orthogonal cell (two robots cannot swap
cells within one tick; following a robot into the cell it just vacated is
fine). A frame's ``moved`` list must reference current fleet robots without
duplicates and, whenever a previous frame exists, must name exactly the
robots whose positions changed, in any order. Version 2 map changes are
applied in their recorded order after the tick they are stamped on, so a
robot may be recorded in a cell that is closed later, may not appear in a
cell before it opens, and no change may add an obstacle onto a cell a tick
frame shows occupied -- not even when a later change of the same tick removes
it. Filesystem and permission failures propagate as :class:`OSError`.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict

from .model import GridMap, Position, Robot, Task
from .pathfinding import shortest_path

CHECKPOINT_VERSION = 2
_LEGACY_VERSIONS = frozenset({1})
_SUPPORTED_VERSIONS = frozenset({CHECKPOINT_VERSION}) | _LEGACY_VERSIONS


def _is_int(value: object) -> bool:
    """JSON booleans are ints in Python; checkpoints treat them as wrong."""
    return isinstance(value, int) and not isinstance(value, bool)


def _coordinate_pair(value: object, description: str, *, strict_list: bool) -> Position:
    """Validate *value* as a pair of two plain integers.

    Booleans are rejected even though they are Python ints, and wrong lengths
    or element types raise :class:`ValueError` instead of leaking unpacking or
    hashing errors. This is the one coordinate-shape rule shared by every
    caller; the two entry points only differ in the outer container spellings
    they accept:

    * strict JSON lists only (*strict_list* true, checkpoints, whose document
      shape is fixed and must not be relaxed because validation is shared);
    * lists or tuples alike (*strict_list* false, the direct Python interface,
      where the two spellings may be mixed freely).
    """
    if strict_list:
        # Checkpoints keep their single, fixed message for every shape problem
        # -- wrong type, wrong length or a non-integer (including a boolean) --
        # because the JSON document shape is fixed and shared validation must
        # not change what a malformed checkpoint reports.
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not all(_is_int(coord) for coord in value)
        ):
            raise ValueError(f"{description} must be a [x, y] integer pair")
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
            raise ValueError(f"{description} must be a [x, y] integer pair")
        if len(value) != 2:
            raise ValueError(f"{description} must be a [x, y] integer pair")
        if not all(_is_int(coord) for coord in value):
            raise ValueError(
                f"{description} coordinates must be integers, not booleans or other types"
            )
    return value[0], value[1]


def _as_cell(value: object, description: str) -> Position:
    """Validate a user supplied coordinate as a two-integer list or tuple."""
    return _coordinate_pair(value, description, strict_list=False)


def _as_pair(value: object, description: str) -> tuple[int, int]:
    """Validate a checkpoint coordinate as a two-integer JSON list."""
    return _coordinate_pair(value, description, strict_list=True)


def _task_route_waypoints(origin: Position, task: Task) -> list[Position]:
    """Required remaining visit points after *origin* for *task*.

    The same pickup/delivery rule is used by first-time assignment, rerouting
    after map changes (and the recovery/sidestep planning that shares it) and
    checkpoint validation, so the requirement lives in exactly one place:

    * Goods not yet collected: the route must still reach the pickup and then
      the dropoff. Standing exactly on the pickup makes the pickup leg empty --
      the leg contributes no waypoint -- but planning never collects or
      completes anything by itself.
    * Goods already collected: the robot routes straight to the dropoff, even
      when the old pickup cell was closed afterwards; it is never sent back.

    When *origin*, pickup and dropoff coincide the list is empty, which
    describes a feasible zero-step route and must not be confused with an
    unreachable one.
    """
    waypoints: list[Position] = []
    if not task.picked_up and origin != task.pickup:
        waypoints.append(task.pickup)
    last = waypoints[-1] if waypoints else origin
    if task.dropoff != last:
        waypoints.append(task.dropoff)
    return waypoints


def _plan_task_route(
    grid: GridMap, origin: Position, task: Task
) -> list[Position] | None:
    """Shortest feasible route through the task's required waypoints.

    The returned list holds the cells after *origin* only, concatenating the
    deterministic shortest legs between consecutive required points. It is
    ``None`` when any required point cannot be reached, and an empty list for a
    feasible route with zero steps (origin already at the only destination).
    """
    route: list[Position] = []
    current = origin
    for waypoint in _task_route_waypoints(origin, task):
        try:
            leg = shortest_path(grid, current, waypoint)
        except ValueError:
            return None
        route.extend(leg)
        current = waypoint
    return route


def _validate_route_walk(
    grid: GridMap,
    origin: Position,
    route: Sequence[object],
    *,
    robot_id: str,
    checkpoint: bool,
    blocked_cells: frozenset[Position] | None = None,
) -> list[Position]:
    """Validate one robot's remaining route as a cell-by-cell map walk.

    This is the single place that owns the walking rule shared by direct
    fleet construction and checkpoint recovery, so the same legality check is
    never edited twice. A route holds the cells still ahead of *origin*:

    * every waypoint must parse as two plain integers (booleans are rejected),
      and lie inside the map -- a cell in *blocked_cells* is the one kind of
      currently obstructed waypoint a version 2 checkpoint recovery may carry
      (see below), everything else off the traversable map is rejected;
    * the first waypoint must be an up/down/left/right neighbour of *origin*,
      so a first waypoint equal to the current cell is rejected;
    * every later waypoint must be exactly one orthogonal step from the
      previous one -- diagonal moves, multi-cell jumps and two identical
      consecutive waypoints are all rejected.

    Detours and revisits (including *origin*) are legal, and a route crossing
    another robot's cell is never a walk error: traffic is settled while
    stepping, not here. The walkability rule deliberately says nothing about
    the bound task's pickup/dropoff requirements; checkpoint loading keeps
    that separate, stricter business check. The result is a fresh mutable list
    of tuple cells, so validating never aliases or mutates the caller's
    container and a rejected input leaves every earlier object untouched.

    *blocked_cells* is honoured only for checkpoint recovery and names the
    cells the recorded map-change history explains: cells open on the base map
    that an accepted edit has since closed and that are still closed at save
    time. A taskless robot's preset route may legitimately end on such a cell
    -- the robot keeps its unconsumed waypoints and simply waits before it --
    so those waypoints walk as well as the open cells do. The exemption is
    deliberately narrow: only checkpoint loading ever supplies the set (direct
    construction leaves it ``None``), only a robot without a bound task may
    use it (the caller never passes the set for a task route), and the cells
    must be inside the map -- an out-of-bounds waypoint is rejected regardless.
    A blocked cell in the base map that no edit explains is never part of the
    set, so a route crossing an initial or unexplained obstacle stays illegal.

    The two entries keep their existing format and wording differences via
    *checkpoint*: direct construction (false) accepts a list or tuple route of
    list-or-tuple cells and reports ``route waypoint N`` with a dedicated
    message for a first waypoint on the current cell and for repeated
    consecutive waypoints; checkpoint loading (true) keeps the strict JSON
    list-of-lists format (``_as_pair``), says ``route entry`` and folds both
    equal-cell cases into its existing non-adjacency message. The checks
    performed are identical either way.
    """
    allowed_blocked = frozenset() if blocked_cells is None else blocked_cells
    waypoints: list[Position] = []
    previous = origin
    for index, cell in enumerate(route):
        if checkpoint:
            waypoint = _as_pair(
                cell, f"robot {robot_id!r} route entry {index}"
            )
        else:
            waypoint = _as_cell(
                cell, f"robot {robot_id!r} route waypoint {index}"
            )
        if not grid.contains(waypoint) or (
            waypoint in grid.obstacles and waypoint not in allowed_blocked
        ):
            if checkpoint:
                raise ValueError(
                    f"robot {robot_id!r} route entry {list(waypoint)} is "
                    "outside the map or inside an obstacle"
                )
            raise ValueError(
                f"robot {robot_id!r} route waypoint {index} "
                f"{list(waypoint)} is outside the map or inside an obstacle"
            )
        if waypoint == previous:
            if checkpoint:
                # The checkpoint wording treats an equal cell like any other
                # non-adjacent step, naming both cells of the offending move.
                raise ValueError(
                    f"robot {robot_id!r} route moves non-adjacently from "
                    f"{list(previous)} to {list(waypoint)}"
                )
            if index == 0:
                raise ValueError(
                    f"robot {robot_id!r} route waypoint 0 {list(waypoint)} "
                    "is the robot's current cell; the remaining route must "
                    "start with the next cell"
                )
            raise ValueError(
                f"robot {robot_id!r} route repeats cell {list(waypoint)} at "
                f"consecutive waypoints {index - 1} and {index}"
            )
        if (
            abs(waypoint[0] - previous[0])
            + abs(waypoint[1] - previous[1])
            != 1
        ):
            if checkpoint:
                raise ValueError(
                    f"robot {robot_id!r} route moves non-adjacently from "
                    f"{list(previous)} to {list(waypoint)}"
                )
            if index == 0:
                segment = (
                    f"from its current position {list(previous)} to route "
                    f"waypoint 0 {list(waypoint)}"
                )
            else:
                segment = (
                    f"from route waypoint {index - 1} {list(previous)} to "
                    f"waypoint {index} {list(waypoint)}"
                )
            raise ValueError(
                f"robot {robot_id!r} route moves non-adjacently {segment}: "
                "every step must move one orthogonal cell"
            )
        waypoints.append(waypoint)
        previous = waypoint
    return waypoints


class FleetSimulator:
    def __init__(
        self,
        grid: GridMap,
        robots: list[Robot],
        tasks: list[Task],
        *,
        _routes_prevalidated: bool = False,
    ) -> None:
        self.grid = grid
        # The map as built before any runtime edit; map-change history is
        # replayed against this baseline.
        self.base_grid = grid
        # Starting positions share the route coordinates' strict rule: only a
        # list or tuple of two plain integers is accepted (booleans, floats and
        # strings never act as coordinates), and every accepted position is
        # normalized to a fresh integer tuple before any state is committed.
        # This mirrors :meth:`_normalize_initial_routes`: all robots and tasks
        # stay exactly as the caller passed them until the whole batch --
        # positions, overlaps, ownership and routes -- has passed.
        normalized_positions = self._normalize_start_positions(grid, robots)
        # Task pickup/dropoff points share the positions' and route waypoints'
        # strict rule and are validated up front as well, so a batch that
        # smuggles in booleans, floats or malformed points fails at construction
        # -- not later, when assignment hashes the point or a checkpoint load
        # rejects it. Like the positions, the tuple copies are committed only
        # after the whole batch passes, keeping the caller's objects untouched.
        normalized_task_points = self._normalize_task_points(tasks)
        self.robots = {robot.robot_id: robot for robot in robots}
        self.tasks = {task.task_id: task for task in tasks}
        self.tick = 0
        self.replay: list[dict[str, object]] = []
        self.map_changes: list[dict[str, object]] = []
        self._map_sequence = 0
        self.paused_tasks: set[str] = set()
        # Consecutive per-robot waits caused purely by other robots:
        # robot_id -> {"blocked_by": [robot_id, ...], "ticks": int}.
        self._traffic_waits: dict[str, dict[str, object]] = {}
        self._validate_task_ownership(robots, tasks)
        if _routes_prevalidated:
            # Checkpoint recovery has already run the shared
            # :func:`_validate_route_walk` itself -- including its one narrow
            # checkpoint exemption for a taskless preset route waiting before
            # a history-explained closed cell -- so its already-normalized
            # routes must not be re-checked here against the saved grid, where
            # those legitimate waypoints look obstructed. Its robots are built
            # fresh during loading with validated integer-tuple positions too,
            # so nothing needs committing here. Every other constructor check
            # above still ran.
            return
        # Accepted positions and routes share one internal shape regardless of
        # how the caller spelled them: an integer tuple for the position and a
        # mutable list of tuple cells for the route. The normalized copies are
        # committed together only after every robot passed validation, so a
        # rejection -- even one buried in a later robot's route -- never touches
        # the caller's objects: a fleet whose routes fail is rejected with every
        # robot still holding the original position object and route. See
        # :meth:`_normalize_initial_routes` and
        # :meth:`_normalize_start_positions`.
        normalized_routes = self._normalize_initial_routes(
            grid, robots, normalized_positions
        )
        for robot, position, route in zip(robots, normalized_positions, normalized_routes):
            robot.position = position
            robot.route = route
        for task, (pickup, dropoff) in zip(tasks, normalized_task_points):
            task.pickup = pickup
            task.dropoff = dropoff

    @staticmethod
    def _normalize_start_positions(
        grid: GridMap, robots: Sequence[Robot]
    ) -> list[Position]:
        """Validate every robot's starting position and return tuple copies.

        The position rule is deliberately the same strict rule the route
        waypoints use (:func:`_as_cell`): a position must be a list or tuple
        (the two spellings may be mixed across one batch) holding exactly two
        plain integers -- booleans are rejected even though they are Python
        ints, and floats, strings, missing components, extra components and
        non-pair inputs are all refused with :class:`ValueError` naming the
        robot and its bad position, never rounded, coerced or padded.

        Every accepted position is additionally checked against the map: it
        must lie inside the grid and outside every obstacle. Two robots may
        not start on the same cell either; the comparison runs on normalized
        tuple values, so one robot spelling a cell as ``[x, y]`` and another
        as ``(x, y)`` is still a collision, and the error names both robots
        involved. Idle robots, task-bound robots and robots carrying preset
        routes are all checked -- no robot is skipped to make a batch pass.

        The check is read-only: it returns a fresh tuple per robot and commits
        nothing itself. The caller assigns the tuples to the robots only once
        the whole batch (routes and task bindings included) has passed, so a
        later rejection leaves the caller's positions untouched, and the tuple
        copy means mutating an originally passed-in list after construction
        can never change the created fleet.
        """
        normalized: list[Position] = []
        occupied: dict[Position, str] = {}
        for robot in robots:
            position = _as_cell(
                robot.position, f"robot {robot.robot_id!r} starting position"
            )
            if not grid.traversable(position):
                raise ValueError(
                    f"robot {robot.robot_id!r} starting position {list(position)} "
                    "is outside the map or inside an obstacle"
                )
            other_id = occupied.get(position)
            if other_id is not None:
                raise ValueError(
                    f"robot {robot.robot_id!r} starting position {list(position)} "
                    f"is already occupied by robot {other_id!r}: robots cannot "
                    "share an initial cell"
                )
            occupied[position] = robot.robot_id
            normalized.append(position)
        return normalized

    @staticmethod
    def _normalize_task_points(
        tasks: Sequence[Task],
    ) -> list[tuple[Position, Position]]:
        """Validate every task's pickup and dropoff and return tuple copies.

        Both points use the same strict rule as robot positions and route
        waypoints (:func:`_as_cell`): each must be a list or tuple holding
        exactly two plain integers. Lists and tuples may be mixed freely,
        between tasks and even between a task's pickup and dropoff; booleans
        are rejected even though they are Python ints, and floats (``1.0``
        included), strings, ``None``, missing or extra components and any
        other non-coordinate input are all refused with :class:`ValueError`
        naming the task id and the offending point (its pickup or dropoff) --
        nothing is ever rounded, coerced or padded into a pair.

        Accepted points are returned as fresh integer tuples. Normalization is
        what makes ``[1, 0]`` and ``(1, 0)`` the same cell for the rest of the
        engine: without it, tuple robot positions would compare unequal to
        list task points (so pickups and deliveries would never confirm) and
        list points could not be hashed during route planning. Only the
        coordinate *shape* is judged here -- points are deliberately never
        checked against the map, so a task whose point is currently
        unreachable or blocked keeps waiting (or, when already loaded, keeps
        its goods and owner) exactly as before. Every task is covered no
        matter its state: unassigned and waiting, heading to pickup, carrying
        goods, or already completed history.

        The check is read-only: it returns one ``(pickup, dropoff)`` tuple
        pair per task and commits nothing. The caller assigns them only once
        the whole batch (positions, ownership and routes included) has passed,
        so a malformed point on a later task -- or any later ownership or
        route failure -- leaves every earlier task's original coordinate
        containers, and every robot's position, route, binding and mileage,
        exactly as passed in.
        """
        normalized: list[tuple[Position, Position]] = []
        for task in tasks:
            pickup = _as_cell(task.pickup, f"task {task.task_id!r} pickup")
            dropoff = _as_cell(task.dropoff, f"task {task.task_id!r} dropoff")
            normalized.append((pickup, dropoff))
        return normalized

    @staticmethod
    def _validate_task_ownership(robots: Sequence[Robot], tasks: Sequence[Task]) -> None:
        """Reject impossible initial bindings between robots and tasks.

        The very same ownership requirements back direct construction and
        checkpoint loading, so they live in exactly one place. The mutual
        binding only governs work still in progress:

        * A robot that declares a task must name a task that exists, is not
          completed yet, and is assigned back to that same robot.
        * An unfinished task that names an owner must name an existing robot
          whose current task it is; an unfinished task with the goods already
          collected additionally must have an owner.
        * A robot can therefore never be claimed by two unfinished tasks: the
          second task's owner points at a robot that executes another task.
        * A completed task must carry the picked-up mark *and* a historical
          owner (a non-null ``assigned_robot``); see below.

        Completed tasks are history, not work in progress, but the history must
        still be coherent: a finished task must carry both the picked-up mark
        and the id of the robot that finished it -- ``None`` means "no record",
        never an anonymous completion, even when the pickup point and the
        dropoff coincide. Neither gap is repaired (the owner is not filled in,
        the pickup mark is not flipped, nothing is reassigned or dropped); the
        fleet is rejected as a whole with a :class:`ValueError` naming the task
        and the exact contradiction. The historical owner itself is bound by
        none of the in-progress rules: it need not be idle or name the task
        back, may already work a new task, may have finished several tasks, and
        may even be absent from the fleet, so historical records never occupy a
        robot or take part in new task assignment. Unassigned, not-yet-collected
        tasks simply wait -- even when no robot can reach them -- and are not an
        ownership error. Nothing here rewrites a binding or drops a task; an
        inconsistent input is rejected as a whole, leaving the caller's objects
        untouched.
        """
        robot_by_id = {robot.robot_id: robot for robot in robots}
        task_by_id = {task.task_id: task for task in tasks}
        for robot in robots:
            if robot.task_id is None:
                continue
            task = task_by_id.get(robot.task_id)
            if task is None:
                raise ValueError(
                    f"robot {robot.robot_id!r} executes unknown task "
                    f"{robot.task_id!r}"
                )
            if task.completed:
                raise ValueError(
                    f"robot {robot.robot_id!r} executes task {task.task_id!r} "
                    "that is already completed"
                )
            if task.assigned_robot != robot.robot_id:
                raise ValueError(
                    f"robot {robot.robot_id!r} executes task {task.task_id!r} "
                    f"but the task is assigned to {task.assigned_robot!r}"
                )
        for task in tasks:
            if task.completed:
                # A finished task is a fixed historical record: the goods must
                # have been collected and a historical finishing robot must be
                # named. The very same requirement backs direct construction
                # and checkpoint recovery, so a document that could never be
                # loaded again can never be built in the first place. This is
                # judged purely from the record -- even a task whose pickup
                # point equals its dropoff is not presumed collected -- and the
                # recorded robot is history only: it is never looked up, so it
                # may be retired, idle or already busy with another task, and
                # several completed tasks may name the same robot without ever
                # counting as occupying it.
                if not task.picked_up:
                    if task.assigned_robot is None:
                        raise ValueError(
                            f"task {task.task_id!r} is marked completed but was "
                            "never picked up and has no historical assigned robot"
                        )
                    raise ValueError(
                        f"task {task.task_id!r} is marked completed but never "
                        "picked up, even though historical robot "
                        f"{task.assigned_robot!r} is recorded for it"
                    )
                if task.assigned_robot is None:
                    raise ValueError(
                        f"task {task.task_id!r} is marked completed and picked up "
                        "but has no historical assigned robot"
                    )
                continue
            if task.picked_up and task.assigned_robot is None:
                raise ValueError(
                    f"task {task.task_id!r} is picked up but has no assigned robot"
                )
            if task.assigned_robot is not None:
                owner = robot_by_id.get(task.assigned_robot)
                if owner is None:
                    raise ValueError(
                        f"task {task.task_id!r} is assigned to unknown robot "
                        f"{task.assigned_robot!r}"
                    )
                if owner.task_id != task.task_id:
                    raise ValueError(
                        f"task {task.task_id!r} claims robot {owner.robot_id!r} "
                        f"but the robot executes {owner.task_id!r}"
                    )

    @staticmethod
    def _normalize_initial_routes(
        grid: GridMap,
        robots: Sequence[Robot],
        positions: Sequence[Position],
    ) -> list[list[Position]]:
        """Validate every remaining route and return normalized copies.

        Every robot is checked, idle or task-bound, via the single shared walk
        validator :func:`_validate_route_walk` -- the same coordinate shape,
        map-walkability and one-orthogonal-step rule checkpoint recovery uses
        -- so a walking rule never has to be changed in two places. The outer
        route may be a list or tuple and each waypoint may itself be a list or
        tuple of two plain integers -- booleans are not integers here, and
        wrong lengths or element types raise :class:`ValueError` instead of
        leaking unpacking or hashing failures. The forms may be mixed freely;
        only the coordinate order matters, not the container types.

        *positions* holds the already-normalized tuple starting positions from
        :meth:`_normalize_start_positions`, paired one-to-one with *robots*:
        every route is walked from the tuple form so a first waypoint equal to
        the start cell is recognized regardless of how the caller spelled
        either container.

        Accepted routes are returned as mutable lists of tuple cells, the one
        shape the rest of the engine relies on (head consumption with
        ``pop(0)`` and hashable cells for road and occupancy checks). Empty
        lists and tuples both normalize to an empty route. Routes need not be
        shortest paths and may revisit cells, and traffic (a route crossing
        another robot's cell, or routes sharing a cell) is left to execution:
        only walkability on the given map is judged.

        The check is read-only -- :func:`_validate_route_walk` builds fresh
        lists, and the caller commits them itself only once every robot has
        passed, so a rejected fleet is rejected as a whole: an error on a
        later robot or at the end of a route never truncates, pads, reorders
        or otherwise modifies an earlier robot or any caller object. The
        shared walk check says nothing about tasks; direct construction keeps
        accepting any walkable route, while checkpoint loading keeps its own
        stricter validation that also covers the pickup/dropoff rules.
        """
        normalized: list[list[Position]] = []
        for robot, position in zip(robots, positions):
            route = robot.route
            if not isinstance(route, (list, tuple)):
                raise ValueError(
                    f"robot {robot.robot_id!r} remaining route must be a list "
                    "of [x, y] waypoints"
                )
            normalized.append(
                _validate_route_walk(
                    grid,
                    position,
                    route,
                    robot_id=robot.robot_id,
                    checkpoint=False,
                )
            )
        return normalized

    # ------------------------------------------------------------------
    # Dynamic map edits
    # ------------------------------------------------------------------

    def modify_obstacles(
        self,
        added: Sequence[Position] = (),
        removed: Sequence[Position] = (),
    ) -> dict[str, object]:
        """Close and/or reopen cells without advancing the simulation.

        The whole request is validated before anything changes: coordinates
        must be two plain integers (booleans are rejected), lie inside the
        map, no cell may appear in both lists, and newly added obstacles may
        not cover a robot's current cell. Duplicate coordinates inside one
        batch are processed once. Adding an existing obstacle or removing a
        traversable cell is a no-op. Invalid requests raise :class:`ValueError`
        and leave every part of the state untouched.

        Assigned unfinished tasks keep their robot: routes are replanned
        against the new map, or, when a required point is unreachable, cleared
        while the task is suspended. No tick is recorded, no robot moves and no
        distance is accumulated.
        """
        add_cells = self._normalize_cells(added, "added")
        remove_cells = self._normalize_cells(removed, "removed")
        conflict = set(add_cells) & set(remove_cells)
        if conflict:
            cell = sorted(conflict)[0]
            raise ValueError(
                f"cell {list(cell)} is both added and removed in the same request"
            )
        occupied = {robot.position for robot in self.robots.values()}
        for cell in add_cells:
            if cell in occupied:
                raise ValueError(
                    f"cannot add obstacle at {list(cell)}: a robot currently occupies it"
                )
        return self._apply_map_change(add_cells, remove_cells)

    def _normalize_cells(self, cells: Sequence[Position], label: str) -> list[Position]:
        if isinstance(cells, (str, bytes)) or not isinstance(cells, (list, tuple)):
            raise ValueError(f"{label} cells must be provided as a list of [x, y] pairs")
        seen: set[Position] = set()
        ordered: list[Position] = []
        for index, cell in enumerate(cells):
            position = _as_cell(cell, f"{label} cell {index}")
            if not self.grid.contains(position):
                raise ValueError(f"{label} cell {list(position)} lies outside the map")
            if position in seen:
                # Batch-internal duplicates are handled once, not rejected.
                continue
            seen.add(position)
            ordered.append(position)
        return ordered

    def _apply_map_change(
        self, added: list[Position], removed: list[Position]
    ) -> dict[str, object]:
        obstacles = set(self.grid.obstacles)
        actual_added = [cell for cell in added if cell not in obstacles]
        obstacles.update(actual_added)
        actual_removed = [cell for cell in removed if cell in obstacles]
        obstacles.difference_update(actual_removed)
        if not actual_added and not actual_removed:
            return self._map_change_record(self._map_sequence, [], [])

        # Everything below only runs once the batch was fully validated and is
        # known to change the map.
        self.grid = GridMap(self.grid.width, self.grid.height, frozenset(obstacles))
        self._map_sequence += 1
        record = self._map_change_record(self._map_sequence, actual_added, actual_removed)
        self.map_changes.append(copy.deepcopy(record))
        self.replay.append(copy.deepcopy(record))
        self._reroute_after_map_change()
        self._reconcile_traffic_waits_after_map_change()
        return copy.deepcopy(record)

    def _map_change_record(
        self, sequence: int, added: list[Position], removed: list[Position]
    ) -> dict[str, object]:
        return {
            "type": "map_change",
            "tick": self.tick,
            "sequence": sequence,
            "added": [list(cell) for cell in added],
            "removed": [list(cell) for cell in removed],
        }

    def _plan_route(
        self, robot: Robot, task: Task, start: Position | None = None
    ) -> list[Position] | None:
        """Shortest current->pickup->dropoff route, or None if unreachable.

        *start* overrides the robot's current position, which is how sidestep
        candidates are evaluated without moving the robot first. The
        pickup/delivery rule itself is shared with first-time assignment and
        checkpoint validation via :func:`_plan_task_route`.
        """
        origin = robot.position if start is None else start
        return _plan_task_route(self.grid, origin, task)

    def _reroute_after_map_change(self) -> None:
        for task in self.tasks.values():
            if task.completed or task.assigned_robot is None:
                continue
            robot = self.robots[task.assigned_robot]
            if robot.task_id != task.task_id:
                continue
            route = self._plan_route(robot, task)
            if route is None:
                robot.route = []
                self.paused_tasks.add(task.task_id)
            else:
                # Only the route is refreshed here; pickup/delivery are
                # confirmed at step start (or were already recorded).
                robot.route = route
                self.paused_tasks.discard(task.task_id)

    def _recover_paused_tasks(self) -> None:
        """Resume tasks suspended by map unreachability where possible."""
        for task_id in sorted(self.paused_tasks):
            task = self.tasks.get(task_id)
            if task is None or task.completed or task.assigned_robot is None:
                self.paused_tasks.discard(task_id)
                continue
            robot = self.robots[task.assigned_robot]
            if robot.task_id != task.task_id:
                self.paused_tasks.discard(task_id)
                continue
            route = self._plan_route(robot, task)
            if route is None:
                robot.route = []
                continue
            robot.route = route
            self.paused_tasks.discard(task_id)
            self._finish_if_arrived(robot)

    # ------------------------------------------------------------------
    # Traffic conflicts and yielding
    # ------------------------------------------------------------------

    def _route_cells(self, exclude_id: str) -> set[Position]:
        """Cells every robot except *exclude_id* still plans to drive through."""
        cells: set[Position] = set()
        for robot in self.robots.values():
            if robot.robot_id != exclude_id:
                cells.update(robot.route)
        return cells

    def _safe_side_cells(
        self,
        robot: Robot,
        occupied: Mapping[Position, str],
        reserved: Mapping[Position, str],
        forbidden: Iterable[Position] = (),
    ) -> Iterator[Position]:
        """Adjacent cells *robot* may actually side-step onto this tick.

        The very same safety rule backs every side-step decision -- an idle
        robot yielding for someone, a busy robot dodging a blocker itself, and
        the hypothetical leave checks behind a retreat -- so the shared
        conditions live exactly here. A cell qualifies only when it:

        * lies inside the map with no obstacle (``GridMap.neighbors`` already
          skips anything else),
        * is held by no robot in the given (possibly hypothetical) layout,
        * was not reserved by a move that already happened this tick,
        * lies on no other robot's remaining route (the moving robot's own
          route is excluded), and
        * is not among *forbidden*: cells that are physically free but unsafe
          for the caller's scenario, such as the requester's own cell (a
          side-step there would swap the two robots within one tick) or the
          retreating requester's replanned path (parking there only moves the
          blockage onto a cell the requester must re-enter).

        Neighbours are yielded in the deterministic ``GridMap.neighbors``
        order (up, left, right, down); callers keep their own selection rules.
        """
        blocked = set(forbidden)
        blocked.update(self._route_cells(exclude_id=robot.robot_id))
        for cell in self.grid.neighbors(robot.position):
            if cell in blocked or cell in occupied or cell in reserved:
                continue
            yield cell

    def _idle_sidestep_cell(
        self,
        blocker: Robot,
        requester: Robot,
        occupied: dict[Position, str],
        reserved: dict[Position, str],
    ) -> Position | None:
        """Free side-cell a parked blocker can yield to, or None.

        Only a robot with no bound task and no remaining route is parked and
        asked to yield; callers must never route a taskless-but-moving robot
        here. The requester's own cell is forbidden (yielding onto it would
        swap the two robots within one tick); every other safety condition is
        shared via :meth:`_safe_side_cells`, so the blocker never trades one
        blockage for another. The first candidate in neighbour order wins.
        """
        return next(
            self._safe_side_cells(
                blocker, occupied, reserved, forbidden=(requester.position,)
            ),
            None,
        )

    def _self_sidestep(
        self,
        robot: Robot,
        task: Task | None,
        occupied: dict[Position, str],
        reserved: dict[Position, str],
        already_moved: frozenset[str] = frozenset(),
    ) -> tuple[Position, list[Position]] | None:
        """Best side cell plus replanned route for a blocked robot, or None.

        Only cells off every other robot's planned route qualify, and only if
        the task remains reachable from there. A candidate that merely retreats
        along the corridor only to face the very same blocker again on the next
        tick is rejected too: such a move records mileage and gets undone
        forever instead of making progress, even when more empty cells stretch
        out behind the robot. Candidates are ranked by replanned route length
        with the deterministic neighbor order (up, left, right, down) breaking
        ties. *already_moved* names robots that have used their one move of
        this tick and therefore cannot vacate a cell to resolve the block.
        """
        if task is None or task.completed:
            return None
        destination = robot.route[0]
        blocker_id = occupied.get(destination)
        if blocker_id is None:
            blocker_id = reserved.get(destination)
        best: tuple[Position, list[Position]] | None = None
        best_length = 0
        # Candidates arrive in neighbour order (up, left, right, down), so the
        # first candidate of the minimum replanned length automatically keeps
        # that order as the tie break.
        for cell in self._safe_side_cells(robot, occupied, reserved):
            route = self._plan_route(robot, task, start=cell)
            if route is None:
                continue
            if (
                blocker_id is not None
                and blocker_id != robot.robot_id
                and self._is_bounceback_retreat(
                    robot, cell, route, blocker_id, occupied, reserved, already_moved
                )
            ):
                # The replanned route immediately returns through the cell the
                # robot is vacating and the blocker cannot use that opening to
                # leave: the move would be undone on the next tick.
                continue
            if best is None or len(route) < best_length:
                best = (cell, route)
                best_length = len(route)
        return best

    def _is_bounceback_retreat(
        self,
        robot: Robot,
        side: Position,
        replanned: Sequence[Position],
        blocker_id: str,
        occupied: dict[Position, str],
        reserved: dict[Position, str],
        already_moved: frozenset[str],
    ) -> bool:
        """Whether stepping to *side* only sends the robot back to the block.

        A retreat is futile when the replanned route leads straight back through
        the robot's current cell and still runs through the blocker's cell (so
        the side move is undone next tick with the same blocker ahead) while
        the blocker cannot leave its own cell through the gap that opens up.
        A route that bypasses the blocker entirely is a genuine detour and is
        never rejected here.
        """
        if not replanned or replanned[0] != robot.position:
            return False
        blocker = self.robots.get(blocker_id)
        if blocker is None or blocker.position not in replanned:
            return False
        # Hypothetical end-of-tick layout: the requester has vacated its cell
        # and stands on the candidate side cell.
        hyp_occupied = dict(occupied)
        del hyp_occupied[robot.position]
        hyp_occupied[side] = robot.robot_id
        # The requester only gets back onto its route by driving through these
        # cells, so a blocker may only "leave" by driving on, never by parking
        # on the replanned route. The requester cannot make a second move
        # within this tick, and neither can a robot that already moved.
        blocked_path = frozenset(replanned)
        return not self._can_vacate(
            blocker,
            robot,
            hyp_occupied,
            reserved,
            frozenset({robot.robot_id}) | already_moved,
            blocked_path,
        )

    def _side_exit_cells(
        self,
        robot: Robot,
        task: Task,
        occupied: dict[Position, str],
        reserved: dict[Position, str],
        blocked_path: frozenset[Position],
    ) -> Iterator[tuple[Position, list[Position]]]:
        """Side cells a busy robot could escape to, with their replanned routes.

        Only cells off every other robot's planned route that keep the robot's
        task reachable are offered, in deterministic neighbour order. Whether
        such a cell is a genuine way out -- or just moves the robot aside for
        one tick before it is driven back onto the same blocked corridor -- is
        judged by the caller as it scans the candidates.
        """
        for side_cell in self._safe_side_cells(
            robot, occupied, reserved, forbidden=blocked_path
        ):
            route = self._plan_route(robot, task, start=side_cell)
            if route is not None:
                yield side_cell, route

    def _can_vacate(
        self,
        blocker: Robot,
        requester: Robot,
        occupied: dict[Position, str],
        reserved: dict[Position, str],
        checking: frozenset[str],
        blocked_path: frozenset[Position],
    ) -> bool:
        """Whether *blocker* can leave its cell this tick in the given layout.

        A parked blocker (no task and no remaining route) leaves only by
        yielding onto a safe side cell for the requesting robot; a moving
        blocker -- task-bound, or taskless but still driving a remaining route
        -- leaves by driving onto its next waypoint (possibly following a
        chain of robots that all move this tick); a task-bound one may also
        sidestep onto a free adjacent cell itself. Robots paused by map
        unreachability never move. *checking* names robots already on the
        dependency chain (including robots that used their one move of this
        tick), so meeting one again closes a cyclic dependency -- a head-on
        deadlock, or the requester itself -- and never proves an exit merely
        because every robot on the cycle still carries a route.
        *blocked_path* holds the cells the retreating requester must drive
        back through: another robot parking there only moves the blockage, so
        such sidesteps do not count as leaving (driving on along the cell on
        the blocker's own route does).

        A task-bound blocker's own side step is judged by the same standard as
        the requester's retreat: it counts as leaving only when the replanned
        route from the side cell does not lead straight back through the cell
        being vacated into another robot that cannot get out of the way
        either. A side cell the blocker must leave again on the next tick to
        re-face the very same unyielding jam further down the corridor is no
        way out, so it never justifies the retreat behind it.

        The follow chain can run through an arbitrarily long legal queue, so
        the search is an explicit stack rather than Python recursion: a queue
        of twelve hundred robots ending at a parked robot with no side cell
        must report "cannot vacate" from a normal ``step()`` instead of
        raising :class:`RecursionError`. Each frame is
        ``(robot, occupied, checking, blocked_path, resume)``: a nested
        hypothetical (the blocker having stepped aside) works with its own
        layout and no-go cells, so both travel with the frame. *resume* is
        ``None`` on a frame's first visit and afterwards holds the robot's
        not-yet-tried side-exit candidates as a stateful iterator. A frame
        examined for the first time first tries the free-waypoint and
        follow-chain rules; only when the chain ahead comes back empty does it
        resume at its own side-cell scan, continuing exactly where that scan
        stopped -- the order the single recursive call imposed.
        """
        frame = (blocker, occupied, checking, blocked_path, None)
        stack: list[
            tuple[
                Robot,
                dict[Position, str],
                frozenset[str],
                frozenset[Position],
                Iterator[tuple[Position, list[Position]]] | None,
            ]
        ] = [frame]
        while stack:
            current, cur_occupied, cur_checking, cur_blocked_path, resume = stack.pop()
            if current.robot_id in cur_checking:
                # The chain has folded back onto a robot it already depends
                # on: a cycle cannot vacate itself. The frame that followed
                # this one simply resumes with its own side-cell options.
                continue
            task = self.tasks[current.task_id] if current.task_id is not None else None
            if task is not None and task.task_id in self.paused_tasks:
                continue
            if not current.route:
                if task is not None:
                    # A busy robot with no route left only arrives/finishes this
                    # tick; it never drives or sidesteps for anyone.
                    continue
                # A parked robot (no task and no remaining route) leaves only by
                # yielding onto a safe side cell for the requesting robot; it does
                # so onto the first safe cell in neighbour order. When that
                # particular cell sits on the path the retreating requester must
                # re-enter, parking there only relocates the blockage instead of
                # clearing it, so it does not count as leaving.
                side = self._idle_sidestep_cell(current, requester, cur_occupied, reserved)
                if side is not None and side not in cur_blocked_path:
                    return True
                continue
            if resume is None:
                ahead = current.route[0]
                if (
                    self.grid.traversable(ahead)
                    and ahead not in cur_occupied
                    and ahead not in reserved
                ):
                    return True
                occupant_id = cur_occupied.get(ahead) or reserved.get(ahead)
                if (
                    occupant_id is not None
                    and occupant_id != current.robot_id
                    and occupant_id not in cur_checking
                ):
                    # The robot could follow the occupant once that robot
                    # vacates its cell this same tick. This frame resumes with
                    # its side-cell scan only after the chain ahead proves it
                    # cannot clear, mirroring the recursive call order.
                    stack.append(
                        (
                            current,
                            cur_occupied,
                            cur_checking,
                            cur_blocked_path,
                            self._side_exit_cells(
                                current, task, cur_occupied, reserved, cur_blocked_path
                            )
                            if task is not None
                            else iter(()),
                        )
                    )
                    stack.append(
                        (
                            self.robots[occupant_id],
                            cur_occupied,
                            cur_checking | {current.robot_id},
                            cur_blocked_path,
                            None,
                        )
                    )
                    continue
                resume = (
                    self._side_exit_cells(
                        current, task, cur_occupied, reserved, cur_blocked_path
                    )
                    if task is not None
                    else iter(())
                )
            if task is None:
                # A taskless robot that still drives a remaining route never
                # sidesteps off it -- not even to unblock a requester -- so a side
                # cell cannot count as a way out for it.
                continue
            # Otherwise the task-bound blocker might sidestep onto a free side
            # cell itself, but not onto a cell the retreating requester still has
            # to use -- parking there only moves the blockage.
            chained = False
            for side_cell, route in resume:
                if not route or route[0] != current.position:
                    # A genuine exit: the robot drives on elsewhere instead of
                    # returning onto the cell it just vacated.
                    return True
                # The replanned route leads straight back through the cell
                # being vacated. That still counts as leaving when the way
                # back is clear, or when whoever holds it can clear out in
                # turn -- but not when the robot would re-face a jam that has
                # no exit of its own, because then the side step is undone on
                # the next tick with nothing gained.
                hyp_occupied = dict(cur_occupied)
                del hyp_occupied[current.position]
                hyp_occupied[side_cell] = current.robot_id
                holder_id: str | None = None
                for cell in route[1:]:
                    holder = hyp_occupied.get(cell) or reserved.get(cell)
                    if holder is not None and holder != current.robot_id:
                        holder_id = holder
                        break
                if holder_id is None:
                    return True
                if holder_id in cur_checking:
                    continue
                stack.append(
                    (current, cur_occupied, cur_checking, cur_blocked_path, resume)
                )
                stack.append(
                    (
                        self.robots[holder_id],
                        hyp_occupied,
                        cur_checking | {current.robot_id},
                        frozenset(route),
                        None,
                    )
                )
                chained = True
                break
            if chained:
                continue
        return False

    def _update_traffic_waits(
        self,
        waited: dict[str, str],
        moved_ids: frozenset[str],
    ) -> None:
        """Keep only waits that still hold once the tick's moves have settled.

        *waited* maps every robot that stopped this tick purely because
        another robot held its next waypoint to the robot it saw there at the
        time; *moved_ids* names the robots that ended up moving. The mid-tick
        impression is not trusted on its own, because the blocker may itself
        drive on (or make room for yet another robot) later in the same tick.
        A record survives only when the robot neither moved nor paused and its
        next waypoint is occupied by another robot in the final layout; the
        reported blocker is whichever robot actually occupies that cell. Any
        other outcome -- the robot moved, its task paused, or the way ahead
        cleared -- deletes the entry. A wait that continues without a free tick
        in between keeps accumulating even when the blocker changes identity;
        once interrupted, a later blockage starts a fresh count at one.
        """
        occupied = {robot.position: robot.robot_id for robot in self.robots.values()}
        surviving: dict[str, dict[str, object]] = {}
        for robot_id in waited:
            if robot_id in moved_ids:
                continue
            robot = self.robots.get(robot_id)
            if robot is None or not robot.route:
                continue
            task = self.tasks.get(robot.task_id) if robot.task_id is not None else None
            if task is not None and task.task_id in self.paused_tasks:
                continue
            blocker_id = occupied.get(robot.route[0])
            if blocker_id is None or blocker_id == robot_id:
                continue
            entry = self._traffic_waits.get(robot_id)
            if entry is None:
                surviving[robot_id] = {"blocked_by": [blocker_id], "ticks": 1}
            else:
                surviving[robot_id] = {
                    "blocked_by": [blocker_id],
                    "ticks": int(entry["ticks"]) + 1,
                }
        self._traffic_waits = surviving

    def _reconcile_traffic_waits_after_map_change(self) -> None:
        """Drop wait records an accepted map edit has invalidated.

        A map change never moves a robot, so it can neither create a wait nor
        add to one; but it can end one. A robot whose task just paused on
        unreachability has no route left to wait on, and a robot whose
        replanned next cell is no longer held by the recorded blocker is not
        blocked by it anymore -- both records are removed. An entry whose
        robot still faces the very same blocker on its refreshed route keeps
        its count untouched, whatever else the edit changed.
        """
        occupied = {robot.position: robot.robot_id for robot in self.robots.values()}
        for robot_id, entry in list(self._traffic_waits.items()):
            robot = self.robots.get(robot_id)
            if robot is None or not robot.route:
                del self._traffic_waits[robot_id]
                continue
            if occupied.get(robot.route[0]) not in entry["blocked_by"]:
                del self._traffic_waits[robot_id]

    def _traffic_wait_report(self) -> list[dict[str, object]]:
        return [
            {
                "robot_id": robot_id,
                "blocked_by": list(entry["blocked_by"]),
                "ticks": entry["ticks"],
            }
            for robot_id, entry in sorted(self._traffic_waits.items())
        ]

    # ------------------------------------------------------------------
    # Normal execution
    # ------------------------------------------------------------------

    def assign_tasks(self) -> None:
        """Assign pending tasks to the nearest available robot using stable tie breaks.

        Only a robot with no bound task *and* no remaining route is available:
        a taskless robot still driving a preset route keeps that route and is
        never selected, even when it is the closest to the pickup -- being
        temporarily blocked by traffic does not count as having finished the
        route either. Skipping such a robot never touches its position,
        mileage, cargo state or remaining waypoints, and never makes other
        tasks wait: the next available robot is chosen by the usual shortest
        feasible route length with the robot id string as tie break.
        """
        pending = [task for task in self.tasks.values() if task.assigned_robot is None]
        for task in sorted(pending, key=lambda item: item.task_id):
            choices: list[tuple[int, str, list[tuple[int, int]]]] = []
            for robot in self.robots.values():
                if not robot.idle or robot.route:
                    continue
                route = _plan_task_route(self.grid, robot.position, task)
                if route is None:
                    continue
                choices.append((len(route), robot.robot_id, route))
            if not choices:
                # No robot can reach this task yet; it keeps waiting and does
                # not block assignment of other tasks.
                continue
            _, robot_id, route = min(choices)
            robot = self.robots[robot_id]
            robot.task_id = task.task_id
            robot.route = route
            task.assigned_robot = robot_id

    def _confirm_pickups(self) -> None:
        """A robot starting a tick on its pickup point collects first."""
        for robot in self.robots.values():
            if robot.task_id is None:
                continue
            task = self.tasks[robot.task_id]
            if not task.completed and not task.picked_up and robot.position == task.pickup:
                task.picked_up = True

    def step(self) -> dict[str, object]:
        self._confirm_pickups()
        self._recover_paused_tasks()
        self.assign_tasks()
        # A robot assigned while already standing on the pickup cell collects
        # immediately instead of driving off without the goods.
        self._confirm_pickups()
        occupied = {robot.position: robot.robot_id for robot in self.robots.values()}
        reserved: dict[tuple[int, int], str] = {}
        moved: list[str] = []
        moved_ids: set[str] = set()
        blocked_by: dict[str, str] = {}

        def relocate(robot: Robot, destination: tuple[int, int]) -> None:
            del occupied[robot.position]
            robot.position = destination
            robot.distance_travelled += 1
            occupied[destination] = robot.robot_id
            reserved[destination] = robot.robot_id
            moved.append(robot.robot_id)
            moved_ids.add(robot.robot_id)

        for robot in sorted(self.robots.values(), key=lambda item: item.robot_id):
            if robot.robot_id in moved_ids:
                # Already yielded this tick at another robot's request.
                continue
            task = self.tasks[robot.task_id] if robot.task_id is not None else None
            if task is not None and task.task_id in self.paused_tasks:
                # Paused by map unreachability: stays put, never yields.
                continue
            if not robot.route:
                self._finish_if_arrived(robot)
                continue
            destination = robot.route[0]
            if not self.grid.traversable(destination):
                # Defensive: routes are normally replanned on every map edit.
                if task is not None and not task.completed:
                    route = self._plan_route(robot, task)
                    if route is None:
                        robot.route = []
                        self.paused_tasks.add(task.task_id)
                    else:
                        robot.route = route
                continue
            blocker_id = occupied.get(destination)
            if blocker_id is None:
                blocker_id = reserved.get(destination)
            if blocker_id is None or blocker_id == robot.robot_id:
                relocate(robot, destination)
                robot.route.pop(0)
                self._finish_if_arrived(robot)
                continue
            # Another robot is in the way; this is distinct from a
            # map-unreachability pause and the task stays active.
            blocker = self.robots[blocker_id]
            if blocker_id not in moved_ids:
                if blocker.task_id is None and not blocker.route:
                    # Truly parked blocker: no bound task and no remaining
                    # route. Ask it to yield onto a free side cell, then take
                    # over the cell it vacated. A taskless robot that still has
                    # a remaining route is not parked: it drives that route
                    # cell by cell like any other moving robot and is never
                    # pushed off it for a requester.
                    side = self._idle_sidestep_cell(blocker, robot, occupied, reserved)
                    if side is not None:
                        relocate(blocker, side)
                        relocate(robot, destination)
                        robot.route.pop(0)
                        self._finish_if_arrived(robot)
                        continue
                if blocker.route:
                    ahead = blocker.route[0]
                    if (
                        self.grid.traversable(ahead)
                        and ahead not in occupied
                        and ahead not in reserved
                    ):
                        # The blocker is expected to drive on within this
                        # tick, so waiting one tick is cheaper than detouring.
                        blocked_by[robot.robot_id] = blocker_id
                        continue
            # The blocker cannot or will not move this tick: try to sidestep
            # onto a free adjacent cell and replan from there.
            side = self._self_sidestep(
                robot, task, occupied, reserved, frozenset(moved_ids)
            )
            if side is not None:
                cell, route = side
                relocate(robot, cell)
                robot.route = route
                self._finish_if_arrived(robot)
                continue
            blocked_by[robot.robot_id] = blocker_id
        self._update_traffic_waits(blocked_by, frozenset(moved_ids))
        self.tick += 1
        event: dict[str, object] = {
            "type": "tick",
            "tick": self.tick,
            "moved": moved,
            "robots": {robot_id: list(robot.position) for robot_id, robot in sorted(self.robots.items())},
            "completed": sorted(task.task_id for task in self.tasks.values() if task.completed),
        }
        self.replay.append(event)
        # Hand back an independent copy: trimming or editing the returned
        # frame must never rewrite the history recorded here.
        return copy.deepcopy(event)

    def _finish_if_arrived(self, robot: Robot) -> None:
        if robot.task_id is None:
            return
        task = self.tasks[robot.task_id]
        if task.task_id in self.paused_tasks:
            return
        if robot.position == task.pickup:
            task.picked_up = True
        if task.picked_up and robot.position == task.dropoff and not robot.route:
            task.completed = True
            self.paused_tasks.discard(task.task_id)
            robot.task_id = None

    def metrics(self) -> dict[str, object]:
        completed = sum(task.completed for task in self.tasks.values())
        return {
            "ticks": self.tick,
            "tasks_total": len(self.tasks),
            "tasks_completed": completed,
            "tasks_paused": sorted(self.paused_tasks),
            "completion_ratio": completed / len(self.tasks) if self.tasks else 1.0,
            "distance_total": sum(robot.distance_travelled for robot in self.robots.values()),
            "traffic_waits": self._traffic_wait_report(),
        }

    def status(self) -> dict[str, object]:
        """Obstacles, map-unreachability pauses and per-robot traffic waits."""
        return {
            "tick": self.tick,
            "width": self.grid.width,
            "height": self.grid.height,
            "obstacles": [list(cell) for cell in sorted(self.grid.obstacles)],
            "paused_tasks": sorted(self.paused_tasks),
            "traffic_waits": self._traffic_wait_report(),
        }

    def map_change_history(self) -> list[dict[str, object]]:
        """Effective time, added/removed cells and order of every actual edit."""
        return copy.deepcopy(self.map_changes)

    def snapshot(self) -> dict[str, object]:
        return {
            "tick": self.tick,
            "robots": [asdict(robot) for robot in sorted(self.robots.values(), key=lambda item: item.robot_id)],
            "tasks": [asdict(task) for task in sorted(self.tasks.values(), key=lambda item: item.task_id)],
            "metrics": self.metrics(),
            # An independent copy of the history: later steps and map edits
            # extend the simulator's own replay without growing snapshots
            # already handed out, and edits to a returned snapshot never
            # reach the recorded history.
            "replay": copy.deepcopy(self.replay),
        }

    # ------------------------------------------------------------------
    # Checkpoint persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | os.PathLike[str]) -> None:
        """Write the simulator state to *path* without changing this instance.

        The write is atomic: an existing destination keeps either its complete
        old contents or the complete new contents, and a brand-new destination
        is either fully created or left absent. Filesystem or permission errors
        raise :class:`OSError`.
        """
        document = {
            "version": CHECKPOINT_VERSION,
            "grid": {
                "width": self.grid.width,
                "height": self.grid.height,
                "obstacles": [list(cell) for cell in sorted(self.grid.obstacles)],
            },
            "base_grid": {
                "width": self.base_grid.width,
                "height": self.base_grid.height,
                "obstacles": [list(cell) for cell in sorted(self.base_grid.obstacles)],
            },
            "tick": self.tick,
            "robots": [
                {
                    "robot_id": robot.robot_id,
                    "position": list(robot.position),
                    "route": [list(cell) for cell in robot.route],
                    "task_id": robot.task_id,
                    "distance_travelled": robot.distance_travelled,
                }
                for robot in sorted(self.robots.values(), key=lambda item: item.robot_id)
            ],
            "tasks": [
                {
                    "task_id": task.task_id,
                    "pickup": list(task.pickup),
                    "dropoff": list(task.dropoff),
                    "assigned_robot": task.assigned_robot,
                    "picked_up": task.picked_up,
                    "completed": task.completed,
                }
                for task in sorted(self.tasks.values(), key=lambda item: item.task_id)
            ],
            "replay": copy.deepcopy(self.replay),
            "map_changes": copy.deepcopy(self.map_changes),
            "paused_tasks": sorted(self.paused_tasks),
            "traffic_waits": self._traffic_wait_report(),
        }
        payload = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")

        directory = os.path.dirname(os.path.abspath(os.fspath(path))) or os.curdir
        fd, tmp_name = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=directory)
        try:
            handle = os.fdopen(fd, "wb")
            try:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                handle.close()
            os.replace(tmp_name, path)
            self._fsync_directory(directory)
        except BaseException:
            # After a successful replace the temp name is already gone.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    @staticmethod
    def _fsync_directory(directory: str) -> None:
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        except OSError:
            pass
        finally:
            os.close(dir_fd)

    @classmethod
    def load_checkpoint(cls, path: str | os.PathLike[str]) -> FleetSimulator:
        """Load an independent simulator instance from *path*.

        The returned simulator shares no mutable state with the file or any
        other loaded instance. Invalid checkpoint content raises
        :class:`ValueError` explaining the reason; read/permission failures
        propagate as :class:`OSError`.
        """
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            raise
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"checkpoint is not valid JSON: {exc.msg}") from exc
        return cls._from_checkpoint(data)

    @classmethod
    def _from_checkpoint(cls, data: object) -> FleetSimulator:
        if not isinstance(data, dict):
            raise ValueError("checkpoint root must be a JSON object")

        version = data.get("version")
        if not _is_int(version):
            raise ValueError("checkpoint field 'version' must be an integer")
        if version not in _SUPPORTED_VERSIONS:
            raise ValueError(
                f"unsupported checkpoint version {version}; supported versions: "
                f"{sorted(_SUPPORTED_VERSIONS)}"
            )

        grid = cls._load_grid(data)
        tick = cls._load_tick(data)
        robot_records, task_records = cls._load_entities(data)

        map_changes: list[dict[str, object]] = []
        paused: set[str] = set()
        if version == 1:
            replay = cls._load_legacy_replay(data)
            base_grid = grid
            # Version 1 has no change history: every blocked waypoint keeps
            # being judged purely against the saved grid, exactly as before.
            history_closed: frozenset[Position] = frozenset()
        else:
            replay = cls._load_replay(data)
            map_changes = cls._load_map_changes(data)
            paused = cls._load_paused_tasks(data)
            base_grid = cls._load_base_grid(data)
            if (base_grid.width, base_grid.height) != (grid.width, grid.height):
                raise ValueError("base grid dimensions must match the current grid")
            cls._validate_map_history(base_grid, grid, map_changes)
            history_closed = cls._history_closed_cells(base_grid, map_changes)

        robots: list[Robot] = []
        for record in robot_records:
            position = _as_pair(record["position"], f"robot {record['robot_id']!r} position")
            if not grid.traversable(position):
                raise ValueError(
                    f"robot {record['robot_id']!r} position {list(position)} is outside "
                    "the map or inside an obstacle"
                )
            # Walkability uses the very same shared cell-by-cell rule as
            # direct construction (strict JSON list format and checkpoint
            # wording preserved); the task pickup/dropoff rules are a separate
            # check below. The only relaxation checkpoint recovery adds over
            # direct construction is for a robot with no bound task whose
            # preset route waits before a cell the recorded edits closed: such
            # a waypoint is unobstructed on the base map and still blocked on
            # the saved one, so the route keeps it instead of being rejected
            # as an illegal walk. A bound task route, an initial obstacle and
            # anything the history cannot explain stay strict.
            allowed_blocked = (
                history_closed
                if record["task_id"] is None
                else frozenset()
            )
            route: list[tuple[int, int]] = _validate_route_walk(
                grid,
                position,
                record["route"],
                robot_id=record["robot_id"],
                checkpoint=True,
                blocked_cells=allowed_blocked,
            )
            distance = record["distance_travelled"]
            if not _is_int(distance) or distance < 0:
                raise ValueError(
                    f"robot {record['robot_id']!r} distance_travelled must be a "
                    "non-negative integer"
                )
            robots.append(Robot(record["robot_id"], position, route, record["task_id"], distance))

        tasks: list[Task] = []
        for record in task_records:
            pickup = _as_pair(record["pickup"], f"task {record['task_id']!r} pickup")
            dropoff = _as_pair(record["dropoff"], f"task {record['task_id']!r} dropoff")
            tasks.append(
                Task(
                    record["task_id"],
                    pickup,
                    dropoff,
                    record["assigned_robot"],
                    record["picked_up"],
                    record["completed"],
                )
            )

        robot_by_id = {robot.robot_id: robot for robot in robots}
        task_by_id = {task.task_id: task for task in tasks}

        # Ownership and the completed-task history are judged with the exact
        # rules direct construction uses -- mutual bindings for work in
        # progress, a collected task that may never lack its carrier, and a
        # finished task that must keep its picked-up mark and historical owner
        # -- so a checkpoint can never load a completion that direct
        # construction would have rejected (or vice versa).
        cls._validate_task_ownership(robots, tasks)

        cls._validate_paused_state(paused, task_by_id, robot_by_id, grid)
        cls._validate_active_task_routes(paused, task_by_id, robot_by_id)
        traffic_waits = cls._load_traffic_waits(data, robot_by_id)

        replay_change_frames = cls._replay_map_change_frames(replay)
        if replay_change_frames != map_changes:
            raise ValueError("replay map-change frames do not match 'map_changes' history")

        simulator = cls(grid, robots, tasks, _routes_prevalidated=True)
        simulator.base_grid = base_grid
        simulator.tick = tick
        simulator.replay = copy.deepcopy(replay)
        simulator.map_changes = copy.deepcopy(map_changes)
        simulator._map_sequence = len(map_changes)
        simulator.paused_tasks = set(paused)
        simulator._traffic_waits = traffic_waits
        # Version 1 has no edit history: its saved grid is the baseline and
        # applies to every tick frame. Version 2 replays the recorded changes
        # alongside the frames so each frame is judged against its contemporary
        # map.
        cls._validate_replay_positions(replay, tick, robot_by_id, base_grid, map_changes)
        cls._validate_replay_completion(replay, tick, task_by_id)
        return simulator

    @staticmethod
    def _require_field(data: Mapping[str, object], field_name: str) -> object:
        if field_name not in data:
            raise ValueError(f"checkpoint is missing required field {field_name!r}")
        return data[field_name]

    @classmethod
    def _load_grid(cls, data: Mapping[str, object]) -> GridMap:
        return cls._load_map_definition(
            data, "grid", map_label="grid", obstacle_label="obstacle"
        )

    @classmethod
    def _load_base_grid(cls, data: Mapping[str, object]) -> GridMap:
        return cls._load_map_definition(
            data, "base_grid", map_label="base grid", obstacle_label="base obstacle"
        )

    @classmethod
    def _load_map_definition(
        cls,
        data: Mapping[str, object],
        field_name: str,
        *,
        map_label: str,
        obstacle_label: str,
    ) -> GridMap:
        """Validate one checkpoint map definition and build its grid.

        This is the single place that owns the map-definition rule shared by
        the current ``grid`` and the version 2 ``base_grid``, so a format or
        value rule never has to be changed in two places: the field must be an
        object with positive integer ``width``/``height`` (booleans are not
        integers) and an ``obstacles`` list of ``[x, y]`` integer pairs, each
        lying inside its own map. An empty obstacle list is legal, and a
        coordinate repeated within the list denotes a single obstacle. Nothing
        is coerced -- strings, floats and booleans are rejected, never
        converted -- and the file-format judgement is not relaxed just because
        a grid could be built from the values.

        The two definitions are validated independently and keep their own
        contents: neither map is ever copied from the other. Only the wording
        differs between the two entry points, via *map_label* (``grid`` /
        ``base grid``) and *obstacle_label* (``obstacle`` / ``base obstacle``),
        so an error still names the map and the exact dimension or obstacle
        entry at fault. Cross-definition rules -- matching dimensions and a
        history that replays the base grid onto the current one -- are checked
        separately afterwards and stay distinct rejection reasons.
        """
        raw = cls._require_field(data, field_name)
        if not isinstance(raw, dict):
            raise ValueError(f"checkpoint field {field_name!r} must be an object")
        width = raw.get("width")
        height = raw.get("height")
        if not _is_int(width) or width <= 0:
            raise ValueError(
                f"checkpoint {map_label} width must be a positive integer"
            )
        if not _is_int(height) or height <= 0:
            raise ValueError(
                f"checkpoint {map_label} height must be a positive integer"
            )
        raw_obstacles = raw.get("obstacles")
        if not isinstance(raw_obstacles, list):
            raise ValueError(f"checkpoint {map_label} 'obstacles' must be a list")
        obstacles: set[tuple[int, int]] = set()
        for index, cell in enumerate(raw_obstacles):
            obstacle = _as_pair(cell, f"{obstacle_label} entry {index}")
            if not (0 <= obstacle[0] < width and 0 <= obstacle[1] < height):
                raise ValueError(
                    f"{obstacle_label} {list(obstacle)} lies outside the map"
                )
            obstacles.add(obstacle)
        try:
            return GridMap(width, height, frozenset(obstacles))
        except ValueError as exc:
            raise ValueError(f"invalid checkpoint {map_label}: {exc}") from exc

    @classmethod
    def _load_tick(cls, data: Mapping[str, object]) -> int:
        tick = cls._require_field(data, "tick")
        if not _is_int(tick) or tick < 0:
            raise ValueError("checkpoint field 'tick' must be a non-negative integer")
        return tick

    @classmethod
    def _load_entities(
        cls, data: Mapping[str, object]
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        raw_robots = cls._require_field(data, "robots")
        raw_tasks = cls._require_field(data, "tasks")
        if not isinstance(raw_robots, list):
            raise ValueError("checkpoint field 'robots' must be a list")
        if not isinstance(raw_tasks, list):
            raise ValueError("checkpoint field 'tasks' must be a list")

        robot_records: list[dict[str, object]] = []
        robot_ids: set[str] = set()
        for index, record in enumerate(raw_robots):
            if not isinstance(record, dict):
                raise ValueError(f"robot entry {index} must be an object")
            robot_id = record.get("robot_id")
            if not isinstance(robot_id, str):
                raise ValueError(f"robot entry {index} field 'robot_id' must be a string")
            if robot_id in robot_ids:
                raise ValueError(f"duplicate robot id {robot_id!r} in checkpoint")
            robot_ids.add(robot_id)
            task_id = record.get("task_id")
            if task_id is not None and not isinstance(task_id, str):
                raise ValueError(f"robot {robot_id!r} field 'task_id' must be a string or null")
            if "position" not in record:
                raise ValueError(f"robot {robot_id!r} is missing field 'position'")
            route = record.get("route")
            if not isinstance(route, list):
                raise ValueError(f"robot {robot_id!r} field 'route' must be a list")
            if "distance_travelled" not in record:
                raise ValueError(f"robot {robot_id!r} is missing field 'distance_travelled'")
            robot_records.append(record)

        task_records: list[dict[str, object]] = []
        task_ids: set[str] = set()
        for index, record in enumerate(raw_tasks):
            if not isinstance(record, dict):
                raise ValueError(f"task entry {index} must be an object")
            task_id = record.get("task_id")
            if not isinstance(task_id, str):
                raise ValueError(f"task entry {index} field 'task_id' must be a string")
            if task_id in task_ids:
                raise ValueError(f"duplicate task id {task_id!r} in checkpoint")
            task_ids.add(task_id)
            assigned = record.get("assigned_robot")
            if assigned is not None and not isinstance(assigned, str):
                raise ValueError(
                    f"task {task_id!r} field 'assigned_robot' must be a string or null"
                )
            if not isinstance(record.get("picked_up"), bool):
                raise ValueError(f"task {task_id!r} field 'picked_up' must be a boolean")
            if not isinstance(record.get("completed"), bool):
                raise ValueError(f"task {task_id!r} field 'completed' must be a boolean")
            if "pickup" not in record:
                raise ValueError(f"task {task_id!r} is missing field 'pickup'")
            if "dropoff" not in record:
                raise ValueError(f"task {task_id!r} is missing field 'dropoff'")
            task_records.append(record)

        positions = [
            tuple(record["position"])
            for record in robot_records
            if isinstance(record["position"], list)
            and len(record["position"]) == 2
            and all(_is_int(value) for value in record["position"])
        ]
        if len(positions) != len(set(positions)):
            raise ValueError("checkpoint robots cannot share a position")
        return robot_records, task_records

    @classmethod
    def _load_change_cells(
        cls, raw: object, context: str, field_name: str
    ) -> list[tuple[int, int]]:
        if not isinstance(raw, list):
            raise ValueError(f"{context} field {field_name!r} must be a list")
        cells: list[tuple[int, int]] = []
        for index, cell in enumerate(raw):
            position = _as_pair(cell, f"{context} {field_name} entry {index}")
            cells.append(position)
        return cells

    @classmethod
    def _load_map_changes(cls, data: Mapping[str, object]) -> list[dict[str, object]]:
        raw = cls._require_field(data, "map_changes")
        if not isinstance(raw, list):
            raise ValueError("checkpoint field 'map_changes' must be a list")
        changes: list[dict[str, object]] = []
        last_tick = -1
        last_sequence = 0
        for index, frame in enumerate(raw):
            context = f"map change {index}"
            if not isinstance(frame, dict):
                raise ValueError(f"{context} must be an object")
            tick = frame.get("tick")
            sequence = frame.get("sequence")
            if not _is_int(tick) or tick < 0:
                raise ValueError(f"{context} field 'tick' must be a non-negative integer")
            if not _is_int(sequence) or sequence <= 0:
                raise ValueError(f"{context} field 'sequence' must be a positive integer")
            if sequence != index + 1:
                raise ValueError(
                    f"map change sequences must run consecutively from 1; change {index} "
                    f"has sequence {sequence}, expected {index + 1}"
                )
            if tick < last_tick:
                raise ValueError(
                    f"{context} tick {tick} is earlier than a preceding map change"
                )
            if tick == last_tick and sequence != last_sequence + 1:
                raise ValueError(f"{context} breaks consecutive sequencing at tick {tick}")
            added = cls._load_change_cells(frame.get("added"), context, "added")
            removed = cls._load_change_cells(frame.get("removed"), context, "removed")
            if not added and not removed:
                raise ValueError(f"{context} contains no actual obstacle changes")
            overlap = set(added) & set(removed)
            if overlap:
                cell = sorted(overlap)[0]
                raise ValueError(
                    f"{context} adds and removes the same cell {list(cell)}"
                )
            changes.append(
                {
                    "type": "map_change",
                    "tick": tick,
                    "sequence": sequence,
                    "added": [list(cell) for cell in added],
                    "removed": [list(cell) for cell in removed],
                }
            )
            last_tick = tick
            last_sequence = sequence
        return changes

    @classmethod
    def _validate_map_history(
        cls,
        base_grid: GridMap,
        saved_grid: GridMap,
        changes: Sequence[Mapping[str, object]],
    ) -> None:
        obstacles = set(base_grid.obstacles)
        width, height = saved_grid.width, saved_grid.height
        for frame in changes:
            for key in ("added", "removed"):
                for cell in frame[key]:
                    x, y = cell
                    if not (0 <= x < width and 0 <= y < height):
                        raise ValueError(
                            f"map change {frame['sequence']} cell {list(cell)} lies "
                            "outside the map"
                        )
            added = {tuple(cell) for cell in frame["added"]}
            removed = {tuple(cell) for cell in frame["removed"]}
            for cell in added:
                if cell in obstacles:
                    raise ValueError(
                        f"map change {frame['sequence']} adds obstacle {list(cell)} "
                        "that is already present according to the history"
                    )
            for cell in removed:
                if cell not in obstacles:
                    raise ValueError(
                        f"map change {frame['sequence']} removes obstacle {list(cell)} "
                        "that is absent according to the history"
                    )
            obstacles |= added
            obstacles -= removed
        if obstacles != set(saved_grid.obstacles):
            raise ValueError(
                "the map reproduced from 'map_changes' does not match the saved grid"
            )

    @staticmethod
    def _history_closed_cells(
        base_grid: GridMap,
        changes: Sequence[Mapping[str, object]],
    ) -> frozenset[Position]:
        """Cells open on the base map that the recorded edits net-close.

        The change history is replayed from *base_grid* (its validity and
        reproduction of the saved grid are checked separately by
        :meth:`_validate_map_history`), and the result holds exactly the cells
        a preset-route robot may legitimately still name while waiting: open
        in the initial map, closed later by an accepted edit, and still closed
        at save time. A cell closed and reopened before the save nets to open
        and is therefore absent, as is every base-map obstacle.
        """
        obstacles = set(base_grid.obstacles)
        for frame in changes:
            obstacles.update(tuple(cell) for cell in frame["added"])
            obstacles.difference_update(tuple(cell) for cell in frame["removed"])
        return frozenset(obstacles - set(base_grid.obstacles))

    @classmethod
    def _replay_map_change_frames(
        cls, replay: Sequence[Mapping[str, object]]
    ) -> list[dict[str, object]]:
        frames = [frame for frame in replay if frame.get("type") == "map_change"]
        normalized = [
            {
                "type": "map_change",
                "tick": frame["tick"],
                "sequence": frame["sequence"],
                "added": [list(cell) for cell in frame["added"]],
                "removed": [list(cell) for cell in frame["removed"]],
            }
            for frame in frames
        ]
        return normalized

    @classmethod
    def _load_paused_tasks(cls, data: Mapping[str, object]) -> set[str]:
        raw = cls._require_field(data, "paused_tasks")
        if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
            raise ValueError("checkpoint field 'paused_tasks' must be a list of strings")
        if len(raw) != len(set(raw)):
            raise ValueError("checkpoint field 'paused_tasks' contains duplicates")
        return set(raw)

    @classmethod
    def _load_traffic_waits(
        cls, data: Mapping[str, object], robot_by_id: Mapping[str, Robot]
    ) -> dict[str, dict[str, object]]:
        # Optional in every format version; older files simply lack the field
        # and their robots start with zero recorded waits.
        raw = data.get("traffic_waits", [])
        if not isinstance(raw, list):
            raise ValueError("checkpoint field 'traffic_waits' must be a list")
        occupied = {robot.position: robot.robot_id for robot in robot_by_id.values()}
        waits: dict[str, dict[str, object]] = {}
        for index, entry in enumerate(raw):
            context = f"traffic wait entry {index}"
            if not isinstance(entry, dict):
                raise ValueError(f"{context} must be an object")
            robot_id = entry.get("robot_id")
            if not isinstance(robot_id, str):
                raise ValueError(f"{context} field 'robot_id' must be a string")
            if robot_id not in robot_by_id:
                raise ValueError(f"{context} refers to unknown robot {robot_id!r}")
            if robot_id in waits:
                raise ValueError(f"duplicate traffic wait entry for robot {robot_id!r}")
            blocked_by = entry.get("blocked_by")
            if (
                not isinstance(blocked_by, list)
                or not blocked_by
                or not all(isinstance(value, str) for value in blocked_by)
            ):
                raise ValueError(
                    f"{context} field 'blocked_by' must be a non-empty list of strings"
                )
            if len(blocked_by) != len(set(blocked_by)):
                raise ValueError(f"{context} field 'blocked_by' contains duplicates")
            for blocker_id in blocked_by:
                if blocker_id == robot_id:
                    raise ValueError(f"{context} robot {robot_id!r} cannot block itself")
                if blocker_id not in robot_by_id:
                    raise ValueError(
                        f"{context} refers to unknown blocking robot {blocker_id!r}"
                    )
            ticks = entry.get("ticks")
            if not _is_int(ticks) or ticks <= 0:
                raise ValueError(f"{context} field 'ticks' must be a positive integer")
            if not robot_by_id[robot_id].route:
                raise ValueError(
                    f"{context} robot {robot_id!r} has no remaining route to wait on"
                )
            # The record must reflect the saved layout: the robot's next route
            # cell is held by exactly the robots named in 'blocked_by'. One
            # cell holds at most one robot, so the list must name precisely
            # the occupant of that cell -- a blocker parked elsewhere, an
            # empty next cell, or extra names alongside the real blocker all
            # make the whole checkpoint inconsistent.
            next_cell = robot_by_id[robot_id].route[0]
            occupant_id = occupied.get(next_cell)
            if occupant_id is None:
                raise ValueError(
                    f"{context} robot {robot_id!r} is recorded as blocked by "
                    f"{sorted(blocked_by)} but no robot occupies its next route "
                    f"cell {list(next_cell)}"
                )
            if list(blocked_by) != [occupant_id]:
                raise ValueError(
                    f"{context} robot {robot_id!r} has its next route cell "
                    f"{list(next_cell)} occupied by robot {occupant_id!r}, which "
                    f"does not match the recorded blockers {sorted(blocked_by)}"
                )
            waits[robot_id] = {"blocked_by": list(blocked_by), "ticks": ticks}
        return waits

    @staticmethod
    def _validate_paused_state(
        paused: set[str],
        task_by_id: Mapping[str, Task],
        robot_by_id: Mapping[str, Robot],
        grid: GridMap | None = None,
    ) -> None:
        for task_id in paused:
            task = task_by_id.get(task_id)
            if task is None:
                raise ValueError(f"paused task {task_id!r} is not among the checkpoint tasks")
            if task.completed:
                raise ValueError(f"paused task {task_id!r} is already completed")
            if task.assigned_robot is None:
                raise ValueError(f"paused task {task_id!r} has no assigned robot")
            owner = robot_by_id.get(task.assigned_robot)
            if owner is None or owner.task_id != task_id:
                raise ValueError(
                    f"paused task {task_id!r} is not bound to its assigned robot"
                )
            if owner.route:
                raise ValueError(
                    f"paused task {task_id!r} robot must have an empty remaining route"
                )
            if grid is not None:
                # A saved pause must reflect genuine unreachability on the
                # saved map; otherwise the history is inconsistent. The same
                # pickup-state-aware route rule used at runtime decides this.
                if _plan_task_route(grid, owner.position, task) is not None:
                    raise ValueError(
                        f"paused task {task_id!r} is actually reachable on the saved map"
                    )

    @staticmethod
    def _validate_active_task_routes(
        paused: set[str],
        task_by_id: Mapping[str, Task],
        robot_by_id: Mapping[str, Robot],
    ) -> None:
        """Reject routes that cannot finish the task each robot is bound to.

        Traversability and adjacency are checked elsewhere. This enforces the
        pickup/delivery business rules -- the very same waypoint requirement
        :func:`_task_route_waypoints` plans with -- for tasks that are
        assigned, unfinished and not paused by map unreachability: a non-empty
        route must end at the dropoff, and a robot that has not collected yet
        -- and is not already standing on the pickup cell -- must still pass
        through the pickup. Passing the dropoff early changes nothing; the
        route only has to reach the pickup and return, and saved avoidance
        detours need not match a replanned shortest route. Empty routes are
        legal only when no remaining waypoint is required: the loaded robot is
        already at the dropoff, or position, pickup and dropoff all coincide
        (the goods are collected on resume). Paused tasks keep their empty-route
        exemption, and unassigned or completed tasks are not judged as work in
        progress.
        """
        for task in task_by_id.values():
            if task.completed or task.assigned_robot is None:
                continue
            if task.task_id in paused:
                # Map-unreachability pauses legitimately carry an empty route
                # regardless of pickup state; the binding itself is all that
                # must survive.
                continue
            robot = robot_by_id.get(task.assigned_robot)
            if robot is None or robot.task_id != task.task_id:
                # Ownership mismatches of this kind are reported separately.
                continue
            route = robot.route
            label = (
                f"task {task.task_id!r} assigned to robot {robot.robot_id!r}"
            )
            required = _task_route_waypoints(robot.position, task)
            if route:
                if route[-1] != task.dropoff:
                    raise ValueError(
                        f"{label} has a remaining route that ends at "
                        f"{list(route[-1])} instead of its dropoff "
                        f"{list(task.dropoff)}"
                    )
                if task.pickup in required and task.pickup not in route:
                    raise ValueError(
                        f"{label} has not picked up the goods and its remaining "
                        f"route never reaches the pickup point "
                        f"{list(task.pickup)} on the way to its dropoff "
                        f"{list(task.dropoff)}"
                    )
            elif task.picked_up:
                if robot.position != task.dropoff:
                    raise ValueError(
                        f"{label} has an empty remaining route with the goods "
                        f"already picked up but the robot is at "
                        f"{list(robot.position)}, not at the dropoff "
                        f"{list(task.dropoff)}"
                    )
            elif required:
                raise ValueError(
                    f"{label} has an empty remaining route before pickup: the "
                    "task can only resume when the robot position, pickup "
                    f"{list(task.pickup)} and dropoff {list(task.dropoff)} all "
                    "coincide"
                )

    @classmethod
    def _iter_replay_events(
        cls, data: Mapping[str, object]
    ) -> Iterator[tuple[int, dict[str, object], object]]:
        """Yield ``(index, frame, frame_type)`` for every replay event in order.

        This is the single place that owns the ordinary-tick-frame rules both
        format versions share, so they are never maintained twice: the
        ``replay`` field must be a list, every frame must be an object, a
        missing ``type`` means an ordinary tick frame, and tick frames must be
        numbered consecutively from 1 -- map-change events sit between ticks
        without consuming a number -- with a well-formed payload (checked via
        the equally shared :meth:`_validate_tick_frame_payload`). Anything
        beyond that stays with the two loaders: version 1 rejects non-tick
        frames and normalizes the ``type`` field, version 2 keeps its own
        event-type dispatch and map-change ordering checks. The scan is lazy,
        so each frame is examined exactly when the consuming loader reaches it
        and the first problem is reported in the same order as before.
        """
        raw = cls._require_field(data, "replay")
        if not isinstance(raw, list):
            raise ValueError("checkpoint field 'replay' must be a list")
        tick_index = 0
        for index, frame in enumerate(raw):
            if not isinstance(frame, dict):
                raise ValueError(f"replay frame {index} must be an object")
            frame_type = frame.get("type", "tick")
            if frame_type == "tick":
                tick_index += 1
                frame_tick = frame.get("tick")
                if not _is_int(frame_tick):
                    raise ValueError(f"replay frame {index} field 'tick' must be an integer")
                if frame_tick != tick_index:
                    raise ValueError(
                        "replay tick numbers must run consecutively from 1; frame "
                        f"{index} has tick {frame_tick}, expected {tick_index}"
                    )
                cls._validate_tick_frame_payload(frame, frame_tick)
            yield index, frame, frame_type

    @classmethod
    def _load_replay(cls, data: Mapping[str, object]) -> list[dict[str, object]]:
        replay: list[dict[str, object]] = []
        tick_index = 0
        last_map_tick = -1
        last_map_sequence = 0
        for index, frame, frame_type in cls._iter_replay_events(data):
            if frame_type == "tick":
                tick_index += 1
                replay.append(frame)
            elif frame_type == "map_change":
                tick = frame.get("tick")
                sequence = frame.get("sequence")
                if not _is_int(tick) or tick < 0:
                    raise ValueError(f"replay frame {index} map change tick must be a non-negative integer")
                if not _is_int(sequence) or sequence <= 0:
                    raise ValueError(f"replay frame {index} map change sequence must be positive")
                if tick != tick_index:
                    raise ValueError(
                        f"replay map change {sequence} is stamped tick {tick} but appears "
                        f"after {tick_index} tick frames"
                    )
                if tick < last_map_tick or (
                    tick == last_map_tick and sequence != last_map_sequence + 1
                ):
                    raise ValueError(
                        f"replay map change {sequence} is out of order at tick {tick}"
                    )
                added = cls._load_change_cells(frame.get("added"), f"replay frame {index}", "added")
                removed = cls._load_change_cells(frame.get("removed"), f"replay frame {index}", "removed")
                replay.append(
                    {
                        "type": "map_change",
                        "tick": tick,
                        "sequence": sequence,
                        "added": [list(cell) for cell in added],
                        "removed": [list(cell) for cell in removed],
                    }
                )
                last_map_tick = tick
                last_map_sequence = sequence
            else:
                raise ValueError(f"replay frame {index} has unknown type {frame_type!r}")
        return replay

    @staticmethod
    def _validate_tick_frame_payload(frame: Mapping[str, object], frame_tick: int) -> None:
        moved = frame.get("moved")
        completed = frame.get("completed")
        robots = frame.get("robots")
        if not isinstance(moved, list) or not all(isinstance(value, str) for value in moved):
            raise ValueError(f"replay frame {frame_tick} field 'moved' must be a list of strings")
        if not isinstance(completed, list) or not all(
            isinstance(value, str) for value in completed
        ):
            raise ValueError(
                f"replay frame {frame_tick} field 'completed' must be a list of strings"
            )
        if not isinstance(robots, dict) or not all(
            isinstance(key, str) and isinstance(value, list) for key, value in robots.items()
        ):
            raise ValueError(
                f"replay frame {frame_tick} field 'robots' must be an object of "
                "robot id to [x, y]"
            )
        for robot_id, position in robots.items():
            _as_pair(position, f"replay frame {frame_tick} robot {robot_id!r} position")

    @classmethod
    def _load_legacy_replay(cls, data: Mapping[str, object]) -> list[dict[str, object]]:
        # The container, frame-object and consecutive-tick rules are the shared
        # ones in :meth:`_iter_replay_events`; version 1 only adds its own
        # event-type policy on top: map changes do not exist yet, and ordinary
        # frames may lack the ``type`` field, which is filled in on a copy so
        # the loaded document is never rewritten.
        replay: list[dict[str, object]] = []
        for index, frame, frame_type in cls._iter_replay_events(data):
            if frame_type != "tick":
                raise ValueError("version 1 checkpoints cannot contain map change frames")
            normalized = dict(frame)
            normalized["type"] = "tick"
            replay.append(normalized)
        return replay

    @classmethod
    def _validate_replay_positions(
        cls,
        replay: Sequence[Mapping[str, object]],
        tick: int,
        robot_by_id: Mapping[str, Robot],
        base_grid: GridMap,
        changes: Sequence[Mapping[str, object]] = (),
    ) -> None:
        """Reject history that the simulator could never have produced.

        Every tick frame is checked against the map in effect *during* that
        tick: positions must be in bounds and traversable, no two robots may
        share a cell, each robot may only wait or step one orthogonal cell
        between consecutive tick frames, two robots may not swap cells within
        one tick, and ``moved`` must list exactly the robots whose positions
        changed.

        Version 2 map changes take effect after the tick they are stamped on,
        in their recorded order: the frame for tick *t* is judged against the
        base grid plus every change stamped below *t*, then the changes stamped
        *t* are applied and may not add obstacles onto cells frame *t* shows a
        robot on -- even when a later change of the same tick removes the cell
        again. Tick-0 changes precede the first frame. Adjacency is always
        judged between neighboring tick frames; a map change between them never
        moves a robot.

        Version 1 files pass no changes, so the saved grid governs every frame.
        The first frame has no recorded predecessor: its cells, overlaps and
        ``moved`` ids are still checked, but neither the distance of the first
        moves nor the first ``moved`` list is second-guessed.
        """
        tick_frames = [frame for frame in replay if frame.get("type") == "tick"]
        if len(tick_frames) != tick:
            raise ValueError(
                f"replay has {len(tick_frames)} tick frames but the checkpoint is at tick {tick}"
            )

        expected_ids = set(robot_by_id)

        def frame_positions(frame: Mapping[str, object]) -> dict[str, tuple[int, int]]:
            raw = frame["robots"]
            assert isinstance(raw, dict)
            if set(raw) != expected_ids:
                raise ValueError(
                    f"replay frame {frame['tick']} robot ids do not match current robots"
                )
            return {robot_id: tuple(position) for robot_id, position in raw.items()}

        # A non-zero-tick history must end exactly where the saved robots stand
        # before any deeper history check runs.
        if tick > 0:
            last_positions = frame_positions(tick_frames[-1])
            for robot_id, position in last_positions.items():
                if position != robot_by_id[robot_id].position:
                    raise ValueError(
                        "last replay frame positions do not match current robot positions "
                        f"for robot {robot_id!r}"
                    )

        obstacles = set(base_grid.obstacles)
        width, height = base_grid.width, base_grid.height
        change_index = 0

        def apply_changes(stamp: int, occupants: Mapping[Position, str]) -> None:
            """Apply every change stamped *stamp*, rejecting adds on robots.

            *occupants* maps a cell to the robot sitting on it when that tick
            ended; it is empty for tick-0 edits once ticks have already run,
            because the unrecorded initial layout cannot be reconstructed.
            """
            nonlocal change_index
            while change_index < len(changes) and changes[change_index]["tick"] == stamp:
                change = changes[change_index]
                sequence = change["sequence"]
                added = {tuple(cell) for cell in change["added"]}
                for cell in added:
                    robot_id = occupants.get(cell)
                    if robot_id is not None:
                        when = (
                            "the initial layout before tick 1"
                            if stamp == 0
                            else f"tick frame {stamp}"
                        )
                        raise ValueError(
                            f"map change {sequence} (after tick {stamp}) adds obstacle "
                            f"{list(cell)} occupied by robot {robot_id!r} in {when}"
                        )
                obstacles.update(added)
                obstacles.difference_update(
                    {tuple(cell) for cell in change["removed"]}
                )
                change_index += 1

        # Tick-0 edits happened before the first recorded frame. When the
        # checkpoint is still at tick 0 the saved robot positions *are* the
        # layout of that moment; otherwise they cannot be reconstructed.
        initial_occupants = (
            {robot.position: robot.robot_id for robot in robot_by_id.values()}
            if tick == 0
            else {}
        )
        apply_changes(0, initial_occupants)

        previous: dict[str, tuple[int, int]] | None = None
        for frame in tick_frames:
            frame_tick = frame["tick"]
            # The grid here still reflects only changes stamped below this
            # tick -- exactly the map the robots moved on during this tick.
            positions = frame_positions(frame)
            grid = GridMap(width, height, frozenset(obstacles))

            for robot_id, position in positions.items():
                if not grid.contains(position):
                    raise ValueError(
                        f"replay frame {frame_tick} robot {robot_id!r} is at "
                        f"{list(position)}, outside the map in effect at tick {frame_tick}"
                    )
                if position in grid.obstacles:
                    raise ValueError(
                        f"replay frame {frame_tick} robot {robot_id!r} is at "
                        f"{list(position)}, an obstacle on the map in effect at tick "
                        f"{frame_tick}"
                    )

            cells_seen: dict[Position, str] = {}
            for robot_id, position in positions.items():
                occupant = cells_seen.get(position)
                if occupant is not None:
                    raise ValueError(
                        f"replay frame {frame_tick} robots {occupant!r} and {robot_id!r} "
                        f"both occupy cell {list(position)}"
                    )
                cells_seen[position] = robot_id

            moved = frame["moved"]
            assert isinstance(moved, list)
            moved_ids: set[str] = set()
            for robot_id in moved:
                if robot_id not in expected_ids:
                    raise ValueError(
                        f"replay frame {frame_tick} 'moved' lists unknown robot "
                        f"{robot_id!r}"
                    )
                if robot_id in moved_ids:
                    raise ValueError(
                        f"replay frame {frame_tick} 'moved' lists robot {robot_id!r} "
                        "more than once"
                    )
                moved_ids.add(robot_id)

            if previous is not None:
                actually_moved: set[str] = set()
                for robot_id, position in positions.items():
                    old = previous[robot_id]
                    distance = abs(position[0] - old[0]) + abs(position[1] - old[1])
                    if distance > 1:
                        raise ValueError(
                            f"replay frame {frame_tick} robot {robot_id!r} moves more "
                            f"than one cell, from {list(old)} to {list(position)}"
                        )
                    if distance == 1:
                        actually_moved.add(robot_id)
                mover_ids = sorted(actually_moved)
                for index, robot_id in enumerate(mover_ids):
                    for other_id in mover_ids[index + 1 :]:
                        if (
                            positions[robot_id] == previous[other_id]
                            and positions[other_id] == previous[robot_id]
                        ):
                            raise ValueError(
                                f"replay frame {frame_tick} robots {robot_id!r} and "
                                f"{other_id!r} swap cells "
                                f"{list(previous[robot_id])} and {list(positions[robot_id])} "
                                "within one tick"
                            )
                # Following another robot into the cell it just vacated is
                # legal: only a true two-way swap is rejected above, and the
                # recorded list need not be sorted by robot id.
                if moved_ids != actually_moved:
                    missing = sorted(actually_moved - moved_ids)
                    extra = sorted(moved_ids - actually_moved)
                    details = []
                    if missing:
                        details.append(f"movers missing from the list: {missing}")
                    if extra:
                        details.append(f"robots listed without moving: {extra}")
                    raise ValueError(
                        f"replay frame {frame_tick} 'moved' does not match the robots "
                        f"that changed position ({'; '.join(details)})"
                    )

            # The tick has ended: its own map changes now take effect on top of
            # the cells the frame records the robots on.
            apply_changes(frame_tick, cells_seen)
            previous = positions

    @classmethod
    def _validate_replay_completion(
        cls,
        replay: Sequence[Mapping[str, object]],
        tick: int,
        task_by_id: Mapping[str, Task],
    ) -> None:
        """Reject completion histories that contradict the saved task states.

        A tick frame's ``completed`` list is the cumulative set of tasks
        finished by the end of that tick, not the tasks finished within it:
        every entry must name a checkpoint task, no task may appear twice in
        one frame, and a task once recorded must stay recorded in every later
        frame -- a robot moving on to new work never erases the history. The
        order inside a frame carries no meaning, and repeating the same task
        across frames is the normal case. The last frame must therefore list
        exactly the tasks saved as completed; anything else means the restored
        completion count would differ from what the replay shows. Map-change
        events are not time steps and take no part in this check, so edits
        stamped after the final tick cannot excuse a mismatch. A checkpoint
        still at tick 0 has no frames to judge: it loads as saved, completed
        initial tasks included, and the first recorded frame may list
        completions without any earlier empty frame.
        """
        known_ids = set(task_by_id)
        previous: set[str] = set()
        last_completed: set[str] | None = None
        last_frame_tick = 0
        for frame in replay:
            if frame.get("type") != "tick":
                continue
            frame_tick = frame["tick"]
            entries = frame["completed"]
            assert isinstance(entries, list)
            seen: set[str] = set()
            for task_id in entries:
                if task_id not in known_ids:
                    raise ValueError(
                        f"replay frame {frame_tick} lists unknown completed task "
                        f"{task_id!r}"
                    )
                if task_id in seen:
                    raise ValueError(
                        f"replay frame {frame_tick} lists completed task "
                        f"{task_id!r} more than once"
                    )
                seen.add(task_id)
            dropped = previous - seen
            if dropped:
                task_id = sorted(dropped)[0]
                raise ValueError(
                    f"replay frame {frame_tick} drops task {task_id!r} from the "
                    "completed list; a recorded completion must survive every "
                    "later tick"
                )
            previous = seen
            last_completed = seen
            last_frame_tick = frame_tick

        if tick == 0 or last_completed is None:
            return
        expected = {task.task_id for task in task_by_id.values() if task.completed}
        if last_completed != expected:
            missing = sorted(expected - last_completed)
            extra = sorted(last_completed - expected)
            details = []
            if missing:
                details.append(f"completed tasks missing from the frame: {missing}")
            if extra:
                details.append(f"tasks listed but saved as unfinished: {extra}")
            raise ValueError(
                f"last replay frame (tick {last_frame_tick}) completed tasks do "
                f"not match the saved task states ({'; '.join(details)})"
            )
