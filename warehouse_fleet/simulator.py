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
keep traffic flowing instead of waiting forever: an idle blocker is asked to
move onto a free side cell that lies on no robot's planned route, and a busy
robot facing a blocker that cannot drive on may itself sidestep onto a free
adjacent cell and replan its (still shortest) route from there. Yield moves
count as mileage, waiting does not. The task keeps its original robot, pickup
and completion rules are unchanged, and a robot paused by map unreachability
never takes part in yielding. If no safe side cell exists the robots simply
wait: time advances, nothing collides and nothing completes early. Waiting
caused purely by other robots is tracked per robot (consecutive ticks and the
blocking robot) and reported by :meth:`FleetSimulator.status` and
:meth:`FleetSimulator.metrics` as ``traffic_waits``, distinct from the
map-unreachability ``paused_tasks``.

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
    remaining waypoints after ``position``.
``tasks``
    List of task states, each
    ``{"task_id": str, "pickup": [x, y], "dropoff": [x, y],
    "assigned_robot": str | null, "picked_up": bool, "completed": bool}``.
    Completed tasks keep their historical ``assigned_robot``.
``replay``
    Ordered event list. Tick events are numbered consecutively from ``1``
    through ``tick`` and have the shape
    ``{"type": "tick", "tick": int, "moved": [robot_id, ...],
    "robots": {robot_id: [x, y], ...}, "completed": [task_id, ...]}``.
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
    wait counter at zero.

Version 1 documents store tick frames without a ``type`` field and have no
``map_changes`` or ``base_grid`` key; they load with their grid taken as the
baseline, as if no map edit ever happened.

Loading performs strict validation: malformed JSON, missing or wrongly typed
fields, unsupported versions, duplicate or inconsistent entities, out of
bounds/obstructed positions and routes, non-adjacent route steps, ownership
mismatches, broken replay or map-change history, and a history that does not
reproduce the saved grid all raise :class:`ValueError`. Filesystem and
permission failures propagate as :class:`OSError`.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict

from .model import GridMap, Position, Robot, Task
from .pathfinding import shortest_path

CHECKPOINT_VERSION = 2
_LEGACY_VERSIONS = frozenset({1})
_SUPPORTED_VERSIONS = frozenset({CHECKPOINT_VERSION}) | _LEGACY_VERSIONS


def _is_int(value: object) -> bool:
    """JSON booleans are ints in Python; checkpoints treat them as wrong."""
    return isinstance(value, int) and not isinstance(value, bool)


def _as_cell(value: object, description: str) -> Position:
    """Validate a user supplied coordinate as a two-integer pair."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise ValueError(f"{description} must be a [x, y] integer pair")
    if len(value) != 2:
        raise ValueError(f"{description} must be a [x, y] integer pair")
    if not all(_is_int(coord) for coord in value):
        raise ValueError(f"{description} coordinates must be integers, not booleans or other types")
    return value[0], value[1]


def _as_pair(value: object, description: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(_is_int(coord) for coord in value)
    ):
        raise ValueError(f"{description} must be a [x, y] integer pair")
    return value[0], value[1]


class FleetSimulator:
    def __init__(self, grid: GridMap, robots: list[Robot], tasks: list[Task]) -> None:
        self.grid = grid
        # The map as built before any runtime edit; map-change history is
        # replayed against this baseline.
        self.base_grid = grid
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
        positions = [robot.position for robot in robots]
        if len(positions) != len(set(positions)):
            raise ValueError("robots cannot share an initial position")
        if any(not grid.traversable(position) for position in positions):
            raise ValueError("robot starts outside traversable map space")

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
        candidates are evaluated without moving the robot first.
        """
        origin = robot.position if start is None else start
        try:
            if not task.picked_up:
                if origin == task.pickup:
                    to_pickup: list[Position] = []
                else:
                    to_pickup = shortest_path(self.grid, origin, task.pickup)
                to_dropoff = shortest_path(self.grid, task.pickup, task.dropoff)
            else:
                # Goods are already on board: a reclosed pickup cell must not
                # pull the robot back.
                to_pickup = []
                to_dropoff = shortest_path(self.grid, origin, task.dropoff)
        except ValueError:
            return None
        return to_pickup + to_dropoff

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

    def _idle_sidestep_cell(
        self,
        blocker: Robot,
        requester: Robot,
        occupied: dict[Position, str],
        reserved: dict[Position, str],
    ) -> Position | None:
        """Free side-cell an idle blocker can yield to, or None.

        The target may not be the requester's own cell (that would swap the
        two robots within one tick) nor any cell another robot still plans to
        drive through, so the blocker never trades one blockage for another.
        """
        on_routes = self._route_cells(exclude_id=blocker.robot_id)
        for cell in self.grid.neighbors(blocker.position):
            if cell == requester.position:
                continue
            if cell in occupied or cell in reserved:
                continue
            if cell in on_routes:
                continue
            return cell
        return None

    def _self_sidestep(
        self,
        robot: Robot,
        task: Task | None,
        occupied: dict[Position, str],
        reserved: dict[Position, str],
    ) -> tuple[Position, list[Position]] | None:
        """Best side cell plus replanned route for a blocked robot, or None.

        Only cells off every other robot's planned route qualify, and only if
        the task remains reachable from there; stepping backwards along the
        corridor is never useful and would just oscillate. Candidates are
        ranked by replanned route length with the deterministic neighbor order
        (up, left, right, down) breaking ties.
        """
        if task is None or task.completed:
            return None
        on_routes = self._route_cells(exclude_id=robot.robot_id)
        best: tuple[int, int, Position, list[Position]] | None = None
        for index, cell in enumerate(self.grid.neighbors(robot.position)):
            if cell in occupied or cell in reserved or cell in on_routes:
                continue
            route = self._plan_route(robot, task, start=cell)
            if route is None:
                continue
            if best is None or (len(route), index) < (best[0], best[1]):
                best = (len(route), index, cell, route)
        if best is None:
            return None
        return best[2], best[3]

    def _update_traffic_waits(self, blocked_by: dict[str, str]) -> None:
        """Accumulate consecutive waits; any other outcome resets the counter."""
        for robot_id in list(self._traffic_waits):
            if robot_id not in blocked_by:
                del self._traffic_waits[robot_id]
        for robot_id, blocker_id in blocked_by.items():
            entry = self._traffic_waits.get(robot_id)
            if entry is None:
                self._traffic_waits[robot_id] = {"blocked_by": [blocker_id], "ticks": 1}
            else:
                entry["blocked_by"] = [blocker_id]
                entry["ticks"] = int(entry["ticks"]) + 1

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
        """Assign pending tasks to the nearest idle robot using stable tie breaks."""
        pending = [task for task in self.tasks.values() if task.assigned_robot is None]
        for task in sorted(pending, key=lambda item: item.task_id):
            choices: list[tuple[int, str, list[tuple[int, int]]]] = []
            for robot in self.robots.values():
                if not robot.idle:
                    continue
                try:
                    route = shortest_path(self.grid, robot.position, task.pickup)
                    route += shortest_path(self.grid, task.pickup, task.dropoff)
                except ValueError:
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
                if blocker.task_id is None:
                    # Idle blocker: ask it to yield onto a free side cell,
                    # then take over the cell it vacated.
                    side = self._idle_sidestep_cell(blocker, robot, occupied, reserved)
                    if side is not None:
                        relocate(blocker, side)
                        relocate(robot, destination)
                        robot.route.pop(0)
                        self._finish_if_arrived(robot)
                        continue
                elif blocker.route:
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
            side = self._self_sidestep(robot, task, occupied, reserved)
            if side is not None:
                cell, route = side
                relocate(robot, cell)
                robot.route = route
                self._finish_if_arrived(robot)
                continue
            blocked_by[robot.robot_id] = blocker_id
        self._update_traffic_waits(blocked_by)
        self.tick += 1
        event: dict[str, object] = {
            "type": "tick",
            "tick": self.tick,
            "moved": moved,
            "robots": {robot_id: list(robot.position) for robot_id, robot in sorted(self.robots.items())},
            "completed": sorted(task.task_id for task in self.tasks.values() if task.completed),
        }
        self.replay.append(event)
        return event

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
            "replay": self.replay,
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
        else:
            replay = cls._load_replay(data)
            map_changes = cls._load_map_changes(data)
            paused = cls._load_paused_tasks(data)
            base_grid = cls._load_base_grid(data)
            if (base_grid.width, base_grid.height) != (grid.width, grid.height):
                raise ValueError("base grid dimensions must match the current grid")
            cls._validate_map_history(base_grid, grid, map_changes)

        robots: list[Robot] = []
        for record in robot_records:
            position = _as_pair(record["position"], f"robot {record['robot_id']!r} position")
            if not grid.traversable(position):
                raise ValueError(
                    f"robot {record['robot_id']!r} position {list(position)} is outside "
                    "the map or inside an obstacle"
                )
            route: list[tuple[int, int]] = []
            previous = position
            for index, cell in enumerate(record["route"]):
                waypoint = _as_pair(
                    cell, f"robot {record['robot_id']!r} route entry {index}"
                )
                if not grid.traversable(waypoint):
                    raise ValueError(
                        f"robot {record['robot_id']!r} route entry {list(waypoint)} is "
                        "outside the map or inside an obstacle"
                    )
                if abs(waypoint[0] - previous[0]) + abs(waypoint[1] - previous[1]) != 1:
                    raise ValueError(
                        f"robot {record['robot_id']!r} route moves non-adjacently from "
                        f"{list(previous)} to {list(waypoint)}"
                    )
                route.append(waypoint)
                previous = waypoint
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

        for robot in robots:
            if robot.task_id is not None:
                task = task_by_id.get(robot.task_id)
                if task is None:
                    raise ValueError(
                        f"robot {robot.robot_id!r} executes unknown task {robot.task_id!r}"
                    )
                if task.completed:
                    raise ValueError(
                        f"robot {robot.robot_id!r} executes task {task.task_id!r} that is "
                        "already completed"
                    )
                if task.assigned_robot != robot.robot_id:
                    raise ValueError(
                        f"robot {robot.robot_id!r} executes task {task.task_id!r} but the "
                        f"task is assigned to {task.assigned_robot!r}"
                    )
        for task in tasks:
            if task.completed and not task.picked_up:
                raise ValueError(f"task {task.task_id!r} is completed but never picked up")
            if task.picked_up and task.assigned_robot is None:
                raise ValueError(
                    f"task {task.task_id!r} is picked up but has no assigned robot"
                )
            # Completed tasks retain historical ownership; their robots are free
            # to take new work, so the mutual binding only holds while a task is
            # still in progress.
            if not task.completed and task.assigned_robot is not None:
                owner = robot_by_id.get(task.assigned_robot)
                if owner is None:
                    raise ValueError(
                        f"task {task.task_id!r} is assigned to unknown robot "
                        f"{task.assigned_robot!r}"
                    )
                if owner.task_id != task.task_id:
                    raise ValueError(
                        f"task {task.task_id!r} claims robot {owner.robot_id!r} but the "
                        f"robot executes {owner.task_id!r}"
                    )

        cls._validate_paused_state(paused, task_by_id, robot_by_id, grid)
        traffic_waits = cls._load_traffic_waits(data, robot_by_id)

        replay_change_frames = cls._replay_map_change_frames(replay)
        if replay_change_frames != map_changes:
            raise ValueError("replay map-change frames do not match 'map_changes' history")

        simulator = cls(grid, robots, tasks)
        simulator.base_grid = base_grid
        simulator.tick = tick
        simulator.replay = copy.deepcopy(replay)
        simulator.map_changes = copy.deepcopy(map_changes)
        simulator._map_sequence = len(map_changes)
        simulator.paused_tasks = set(paused)
        simulator._traffic_waits = traffic_waits
        cls._validate_replay_positions(replay, tick, robot_by_id)
        return simulator

    @staticmethod
    def _require_field(data: Mapping[str, object], field_name: str) -> object:
        if field_name not in data:
            raise ValueError(f"checkpoint is missing required field {field_name!r}")
        return data[field_name]

    @classmethod
    def _load_grid(cls, data: Mapping[str, object]) -> GridMap:
        raw = cls._require_field(data, "grid")
        if not isinstance(raw, dict):
            raise ValueError("checkpoint field 'grid' must be an object")
        width = raw.get("width")
        height = raw.get("height")
        if not _is_int(width) or width <= 0:
            raise ValueError("checkpoint grid width must be a positive integer")
        if not _is_int(height) or height <= 0:
            raise ValueError("checkpoint grid height must be a positive integer")
        raw_obstacles = raw.get("obstacles")
        if not isinstance(raw_obstacles, list):
            raise ValueError("checkpoint grid 'obstacles' must be a list")
        obstacles: set[tuple[int, int]] = set()
        for index, cell in enumerate(raw_obstacles):
            obstacle = _as_pair(cell, f"obstacle entry {index}")
            if not (0 <= obstacle[0] < width and 0 <= obstacle[1] < height):
                raise ValueError(f"obstacle {list(obstacle)} lies outside the map")
            obstacles.add(obstacle)
        try:
            return GridMap(width, height, frozenset(obstacles))
        except ValueError as exc:
            raise ValueError(f"invalid checkpoint grid: {exc}") from exc

    @classmethod
    def _load_base_grid(cls, data: Mapping[str, object]) -> GridMap:
        raw = cls._require_field(data, "base_grid")
        if not isinstance(raw, dict):
            raise ValueError("checkpoint field 'base_grid' must be an object")
        width = raw.get("width")
        height = raw.get("height")
        if not _is_int(width) or width <= 0:
            raise ValueError("checkpoint base grid width must be a positive integer")
        if not _is_int(height) or height <= 0:
            raise ValueError("checkpoint base grid height must be a positive integer")
        raw_obstacles = raw.get("obstacles")
        if not isinstance(raw_obstacles, list):
            raise ValueError("checkpoint base grid 'obstacles' must be a list")
        obstacles: set[tuple[int, int]] = set()
        for index, cell in enumerate(raw_obstacles):
            obstacle = _as_pair(cell, f"base obstacle entry {index}")
            if not (0 <= obstacle[0] < width and 0 <= obstacle[1] < height):
                raise ValueError(f"base obstacle {list(obstacle)} lies outside the map")
            obstacles.add(obstacle)
        try:
            return GridMap(width, height, frozenset(obstacles))
        except ValueError as exc:
            raise ValueError(f"invalid checkpoint base grid: {exc}") from exc

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
                # saved map; otherwise the history is inconsistent.
                reachable = True
                try:
                    if not task.picked_up:
                        if owner.position != task.pickup:
                            shortest_path(grid, owner.position, task.pickup)
                        shortest_path(grid, task.pickup, task.dropoff)
                    else:
                        shortest_path(grid, owner.position, task.dropoff)
                except ValueError:
                    reachable = False
                if reachable:
                    raise ValueError(
                        f"paused task {task_id!r} is actually reachable on the saved map"
                    )

    @classmethod
    def _load_replay(cls, data: Mapping[str, object]) -> list[dict[str, object]]:
        raw = cls._require_field(data, "replay")
        if not isinstance(raw, list):
            raise ValueError("checkpoint field 'replay' must be a list")
        replay: list[dict[str, object]] = []
        tick_index = 0
        last_map_tick = -1
        last_map_sequence = 0
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
        raw = cls._require_field(data, "replay")
        if not isinstance(raw, list):
            raise ValueError("checkpoint field 'replay' must be a list")
        replay: list[dict[str, object]] = []
        for index, frame in enumerate(raw):
            if not isinstance(frame, dict):
                raise ValueError(f"replay frame {index} must be an object")
            frame_type = frame.get("type", "tick")
            if frame_type != "tick":
                raise ValueError("version 1 checkpoints cannot contain map change frames")
            frame_tick = frame.get("tick")
            if not _is_int(frame_tick):
                raise ValueError(f"replay frame {index} field 'tick' must be an integer")
            if frame_tick != index + 1:
                raise ValueError(
                    "replay tick numbers must run consecutively from 1; frame "
                    f"{index} has tick {frame_tick}, expected {index + 1}"
                )
            cls._validate_tick_frame_payload(frame, frame_tick)
            normalized = dict(frame)
            normalized["type"] = "tick"
            replay.append(normalized)
        return replay

    @staticmethod
    def _validate_replay_positions(
        replay: Sequence[Mapping[str, object]],
        tick: int,
        robot_by_id: Mapping[str, Robot],
    ) -> None:
        tick_frames = [frame for frame in replay if frame.get("type") == "tick"]
        if len(tick_frames) != tick:
            raise ValueError(
                f"replay has {len(tick_frames)} tick frames but the checkpoint is at tick {tick}"
            )
        expected_ids = set(robot_by_id)
        for frame in tick_frames:
            frame_positions = frame["robots"]
            assert isinstance(frame_positions, dict)
            if set(frame_positions) != expected_ids:
                raise ValueError(
                    f"replay frame {frame['tick']} robot ids do not match current robots"
                )
        if tick > 0:
            last = tick_frames[-1]
            assert isinstance(last["robots"], dict)
            for robot_id, position in last["robots"].items():
                if tuple(position) != robot_by_id[robot_id].position:
                    raise ValueError(
                        "last replay frame positions do not match current robot positions "
                        f"for robot {robot_id!r}"
                    )
