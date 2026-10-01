"""Task assignment and collision-aware tick execution.

Checkpoint file format
----------------------
``save_checkpoint`` writes versioned, UTF-8 encoded JSON. The current (and
first) format version is ``1``. A checkpoint document is a JSON object with
these keys:

``version``
    Integer format version, currently ``1``.
``grid``
    ``{"width": int, "height": int, "obstacles": [[x, y], ...]}`` describing
    the map dimensions and every blocked cell.
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
    List of per-tick event frames in execution order. Frames are numbered
    consecutively from ``1`` through ``tick`` and each has the shape
    ``{"tick": int, "moved": [robot_id, ...],
    "robots": {robot_id: [x, y], ...}, "completed": [task_id, ...]}``. When
    ``tick`` is non-zero the final frame's robot positions must match the
    current robot positions.

Loading performs strict validation: malformed JSON, missing or wrongly typed
fields, unsupported versions, duplicate or inconsistent entities, out of
bounds/obstructed positions and routes, non-adjacent route steps, ownership
mismatches, and broken replay history all raise :class:`ValueError`. Filesystem
and permission failures propagate as :class:`OSError`.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict

from .model import GridMap, MapChange, Robot, Task
from .pathfinding import shortest_path

CHECKPOINT_VERSION = 1
_SUPPORTED_VERSIONS = frozenset({CHECKPOINT_VERSION})


def _is_int(value: object) -> bool:
    """JSON booleans are ints in Python; checkpoints treat them as wrong."""
    return isinstance(value, int) and not isinstance(value, bool)


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
        self.robots = {robot.robot_id: robot for robot in robots}
        self.tasks = {task.task_id: task for task in tasks}
        self.tick = 0
        self.replay: list[dict[str, object]] = []
        self.map_history: list[MapChange] = []
        self._initial_obstacles = frozenset(grid.obstacles)
        positions = [robot.position for robot in robots]
        if len(positions) != len(set(positions)):
            raise ValueError("robots cannot share an initial position")
        if any(not grid.traversable(position) for position in positions):
            raise ValueError("robot starts outside traversable map space")

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
                continue
            _, robot_id, route = min(choices)
            robot = self.robots[robot_id]
            robot.task_id = task.task_id
            robot.route = route
            task.assigned_robot = robot_id

    # ------------------------------------------------------------------
    # In-run map modification
    # ------------------------------------------------------------------

    def modify_map(
        self,
        add_obstacles: Iterable[tuple[int, int]] = (),
        remove_obstacles: Iterable[tuple[int, int]] = (),
    ) -> None:
        """Close and/or reopen cells between two ticks.

        The batch is validated as a whole before anything is applied: every
        coordinate must be a pair of integers (booleans are not integers),
        lie inside the map, and no cell may appear on both sides. Adding an
        obstacle on a cell a robot currently occupies also rejects the whole
        batch. On rejection the map, routes, tasks, statistics and history
        are all left untouched.

        Duplicate coordinates within a batch are processed once. Adding an
        already blocked cell or removing an already open one is a no-op; a
        batch with no net effect changes nothing and is not recorded.

        An effective change flips the map immediately without advancing the
        clock, moving robots or adding mileage. Assigned, unfinished tasks
        keep their robot: a robot that has picked up replans to the dropoff
        only; a robot still heading to pickup replans from its current
        position via the pickup, confirming pickup first when it is already
        there. If any required segment or endpoint is unreachable the
        robot's remaining route is cleared and the task is paused (never
        released or marked complete) until a later change restores a path.
        """
        added = self._coerce_cells(add_obstacles, "add_obstacles")
        removed = self._coerce_cells(remove_obstacles, "remove_obstacles")
        added_set = frozenset(added)
        removed_set = frozenset(removed)
        if added_set & removed_set:
            raise ValueError(
                "cells "
                f"{sorted(added_set & removed_set)} are both added and removed"
            )
        for cell in added_set | removed_set:
            if not self.grid.contains(cell):
                raise ValueError(f"cell {list(cell)} lies outside the map")
        actual_added = added_set - self.grid.obstacles
        actual_removed = removed_set & self.grid.obstacles
        robot_positions = frozenset(robot.position for robot in self.robots.values())
        if actual_added & robot_positions:
            raise ValueError(
                "cannot add obstacles at "
                f"{sorted(actual_added & robot_positions)}: a robot occupies the cell"
            )
        if not actual_added and not actual_removed:
            return
        new_obstacles = (self.grid.obstacles - actual_removed) | actual_added
        self.grid = GridMap(self.grid.width, self.grid.height, frozenset(new_obstacles))
        self.map_history.append(
            MapChange(self.tick, frozenset(actual_added), frozenset(actual_removed))
        )
        self._replan_after_map_change()

    def _coerce_cells(
        self, value: object, description: str
    ) -> list[tuple[int, int]]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise ValueError(f"{description} must be an iterable of [x, y] pairs")
        cells: list[tuple[int, int]] = []
        for index, item in enumerate(value):
            if not isinstance(item, (list, tuple)) or len(item) != 2 or not all(
                _is_int(coord) for coord in item
            ):
                raise ValueError(
                    f"{description} entry {index} must be a [x, y] integer pair"
                )
            cells.append((item[0], item[1]))
        return cells

    def _replan_after_map_change(self) -> None:
        """Replan every assigned, unfinished task against the current map.

        Unreachable tasks are paused with their route cleared; they are never
        released or marked complete. Tasks already picked up only need the
        dropoff leg; the rest need the pickup leg too, confirming pickup
        immediately when the robot is already standing on it.
        """
        for task in self.tasks.values():
            if task.completed or task.assigned_robot is None:
                continue
            robot = self.robots[task.assigned_robot]
            task.paused = False
            robot.route = []
            try:
                if task.picked_up:
                    route = shortest_path(self.grid, robot.position, task.dropoff)
                elif robot.position == task.pickup:
                    task.picked_up = True
                    route = shortest_path(self.grid, robot.position, task.dropoff)
                else:
                    route = shortest_path(self.grid, robot.position, task.pickup)
                    route += shortest_path(self.grid, task.pickup, task.dropoff)
            except ValueError:
                task.paused = True
                robot.route = []
                continue
            robot.route = route

    def paused_task_ids(self) -> list[str]:
        """Task ids paused because the current map made their route unreachable."""
        return sorted(task.task_id for task in self.tasks.values() if task.paused)

    def current_obstacles(self) -> list[tuple[int, int]]:
        """Cells currently blocked, sorted for stable display."""
        return sorted(self.grid.obstacles)

    def step(self) -> dict[str, object]:
        self.assign_tasks()
        occupied = {robot.position for robot in self.robots.values()}
        reserved: set[tuple[int, int]] = set()
        moved: list[str] = []
        for robot in sorted(self.robots.values(), key=lambda item: item.robot_id):
            if not robot.route:
                self._finish_if_arrived(robot)
                continue
            destination = robot.route[0]
            if destination in reserved or (destination in occupied and destination != robot.position):
                continue
            occupied.remove(robot.position)
            robot.position = destination
            robot.route.pop(0)
            robot.distance_travelled += 1
            occupied.add(destination)
            reserved.add(destination)
            moved.append(robot.robot_id)
            self._finish_if_arrived(robot)
        self.tick += 1
        event = {
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
        if robot.position == task.pickup:
            task.picked_up = True
        if task.picked_up and robot.position == task.dropoff and not robot.route:
            task.completed = True
            robot.task_id = None

    def metrics(self) -> dict[str, object]:
        completed = sum(task.completed for task in self.tasks.values())
        return {
            "ticks": self.tick,
            "tasks_total": len(self.tasks),
            "tasks_completed": completed,
            "completion_ratio": completed / len(self.tasks) if self.tasks else 1.0,
            "distance_total": sum(robot.distance_travelled for robot in self.robots.values()),
        }

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
                "initial_obstacles": [
                    list(cell) for cell in sorted(self._initial_obstacles)
                ],
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
                    "paused": task.paused,
                }
                for task in sorted(self.tasks.values(), key=lambda item: item.task_id)
            ],
            "replay": copy.deepcopy(self.replay),
            "map_history": [
                {
                    "tick": change.tick,
                    "added": [list(cell) for cell in sorted(change.added)],
                    "removed": [list(cell) for cell in sorted(change.removed)],
                }
                for change in self.map_history
            ],
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
        replay = cls._load_replay(data)
        initial_obstacles = cls._load_initial_obstacles(data, grid)
        map_history = cls._load_map_history(data, initial_obstacles, grid, tick)

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
            paused = record.get("paused", False)
            if not isinstance(paused, bool):
                raise ValueError(
                    f"task {record['task_id']!r} field 'paused' must be a boolean"
                )
            tasks.append(
                Task(
                    record["task_id"],
                    pickup,
                    dropoff,
                    record["assigned_robot"],
                    record["picked_up"],
                    record["completed"],
                    paused,
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

        simulator = cls(grid, robots, tasks)
        simulator.tick = tick
        simulator.replay = copy.deepcopy(replay)
        simulator._initial_obstacles = frozenset(initial_obstacles)
        simulator.map_history = map_history
        cls._validate_replay_positions(replay, tick, robot_by_id)
        cls._validate_paused_tasks(tasks, robot_by_id)
        return simulator

    @staticmethod
    def _require_field(data: Mapping[str, object], field: str) -> object:
        if field not in data:
            raise ValueError(f"checkpoint is missing required field {field!r}")
        return data[field]

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
    def _load_initial_obstacles(
        cls, data: Mapping[str, object], current_grid: GridMap
    ) -> frozenset[tuple[int, int]]:
        raw_grid = data.get("grid")
        raw = raw_grid.get("initial_obstacles") if isinstance(raw_grid, dict) else None
        if raw is None:
            # Version 1 files predate map history: the map never changed.
            return frozenset(current_grid.obstacles)
        if not isinstance(raw, list):
            raise ValueError("checkpoint grid 'initial_obstacles' must be a list")
        obstacles: set[tuple[int, int]] = set()
        for index, cell in enumerate(raw):
            obstacle = _as_pair(cell, f"initial obstacle entry {index}")
            if not current_grid.contains(obstacle):
                raise ValueError(f"initial obstacle {list(obstacle)} lies outside the map")
            obstacles.add(obstacle)
        return frozenset(obstacles)

    @classmethod
    def _load_map_history(
        cls,
        data: Mapping[str, object],
        initial_obstacles: frozenset[tuple[int, int]],
        current_grid: GridMap,
        tick: int,
    ) -> list[MapChange]:
        raw = data.get("map_history")
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise ValueError("checkpoint field 'map_history' must be a list")
        history: list[MapChange] = []
        obstacles = set(initial_obstacles)
        previous_tick = -1
        for index, record in enumerate(raw):
            if not isinstance(record, dict):
                raise ValueError(f"map change {index} must be an object")
            change_tick = record.get("tick")
            if not _is_int(change_tick) or change_tick < 0:
                raise ValueError(
                    f"map change {index} field 'tick' must be a non-negative integer"
                )
            if change_tick > tick:
                raise ValueError(
                    f"map change {index} tick {change_tick} is after checkpoint tick {tick}"
                )
            if change_tick < previous_tick:
                raise ValueError(
                    f"map change {index} tick {change_tick} is before an earlier change"
                )
            previous_tick = change_tick
            added = cls._load_cell_list(record.get("added"), f"map change {index} added")
            removed = cls._load_cell_list(
                record.get("removed"), f"map change {index} removed"
            )
            added_set = frozenset(added)
            removed_set = frozenset(removed)
            if added_set & removed_set:
                raise ValueError(
                    f"map change {index} adds and removes the same cell "
                    f"{sorted(added_set & removed_set)}"
                )
            for cell in added_set | removed_set:
                if not current_grid.contains(cell):
                    raise ValueError(
                        f"map change {index} cell {list(cell)} is outside the map"
                    )
            for cell in added_set:
                if cell in obstacles:
                    raise ValueError(
                        f"map change {index} adds cell {list(cell)} that is already blocked"
                    )
            for cell in removed_set:
                if cell not in obstacles:
                    raise ValueError(
                        f"map change {index} removes cell {list(cell)} that is not blocked"
                    )
            obstacles = (obstacles - removed_set) | added_set
            history.append(MapChange(change_tick, added_set, removed_set))
        final_grid = GridMap(current_grid.width, current_grid.height, frozenset(obstacles))
        if final_grid != current_grid:
            raise ValueError("map history does not restore the saved map")
        return history

    @classmethod
    def _load_cell_list(
        cls, raw: object, description: str
    ) -> list[tuple[int, int]]:
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise ValueError(f"{description} must be a list")
        cells: list[tuple[int, int]] = []
        for index, cell in enumerate(raw):
            cells.append(_as_pair(cell, f"{description} entry {index}"))
        return cells

    @staticmethod
    def _validate_paused_tasks(
        tasks: Sequence[Task], robot_by_id: Mapping[str, Robot]
    ) -> None:
        for task in tasks:
            if not task.paused:
                continue
            if task.completed:
                raise ValueError(f"task {task.task_id!r} is paused but completed")
            if task.assigned_robot is None:
                raise ValueError(f"task {task.task_id!r} is paused but unassigned")
            owner = robot_by_id.get(task.assigned_robot)
            if owner is None or owner.task_id != task.task_id:
                raise ValueError(
                    f"task {task.task_id!r} is paused but its robot does not own it"
                )
            if owner.route:
                raise ValueError(
                    f"task {task.task_id!r} is paused but its robot still has a route"
                )

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
    def _load_replay(cls, data: Mapping[str, object]) -> list[dict[str, object]]:
        raw = cls._require_field(data, "replay")
        if not isinstance(raw, list):
            raise ValueError("checkpoint field 'replay' must be a list")
        replay: list[dict[str, object]] = []
        for index, frame in enumerate(raw):
            if not isinstance(frame, dict):
                raise ValueError(f"replay frame {index} must be an object")
            frame_tick = frame.get("tick")
            if not _is_int(frame_tick):
                raise ValueError(f"replay frame {index} field 'tick' must be an integer")
            if frame_tick != index + 1:
                raise ValueError(
                    "replay tick numbers must run consecutively from 1; frame "
                    f"{index} has tick {frame_tick}, expected {index + 1}"
                )
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
            replay.append(frame)
        return replay

    @staticmethod
    def _validate_replay_positions(
        replay: Sequence[Mapping[str, object]],
        tick: int,
        robot_by_id: Mapping[str, Robot],
    ) -> None:
        if len(replay) != tick:
            raise ValueError(
                f"replay has {len(replay)} frames but the checkpoint is at tick {tick}"
            )
        expected_ids = set(robot_by_id)
        for frame in replay:
            frame_positions = frame["robots"]
            assert isinstance(frame_positions, dict)
            if set(frame_positions) != expected_ids:
                raise ValueError(
                    f"replay frame {frame['tick']} robot ids do not match current robots"
                )
        if tick > 0:
            last = replay[-1]
            assert isinstance(last["robots"], dict)
            for robot_id, position in last["robots"].items():
                if tuple(position) != robot_by_id[robot_id].position:
                    raise ValueError(
                        "last replay frame positions do not match current robot positions "
                        f"for robot {robot_id!r}"
                    )
