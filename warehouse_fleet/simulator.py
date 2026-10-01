"""Task assignment and collision-aware tick execution."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict

from .model import GridMap, Robot, Task
from .pathfinding import shortest_path

CHECKPOINT_VERSION = 1


class FleetSimulator:
    def __init__(self, grid: GridMap, robots: list[Robot], tasks: list[Task]) -> None:
        self.grid = grid
        self.robots = {robot.robot_id: robot for robot in robots}
        self.tasks = {task.task_id: task for task in tasks}
        self.tick = 0
        self.replay: list[dict[str, object]] = []
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
        """Atomically write a versioned UTF-8 JSON checkpoint of this simulator.

        The simulator is not mutated: tick, robots, tasks, and replay are read
        but never changed. The write is atomic with respect to an existing
        target: a failure leaves either the previous complete file or no file at
        all, never a partially written checkpoint.
        """
        document = {
            "version": CHECKPOINT_VERSION,
            "grid": {
                "width": self.grid.width,
                "height": self.grid.height,
                "obstacles": [list(cell) for cell in sorted(self.grid.obstacles)],
            },
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
            "tick": self.tick,
            "replay": self.replay,
        }
        payload = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        _atomic_write(path, payload)

    @classmethod
    def load_checkpoint(cls, path: str | os.PathLike[str]) -> "FleetSimulator":
        """Load a checkpoint written by :meth:`save_checkpoint`.

        Returns a new, fully independent simulator at the saved tick. The
        returned instance shares no mutable state with the file or with other
        instances loaded from the same file. Raises ``ValueError`` for any
        malformed or inconsistent checkpoint and ``OSError`` for file access
        failures.
        """
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid checkpoint JSON: {exc.msg}") from exc
        return cls._from_checkpoint(data)

    @classmethod
    def _from_checkpoint(cls, data: object) -> "FleetSimulator":
        if not isinstance(data, dict):
            raise ValueError("checkpoint root must be a JSON object")

        version = _require(data, "version", int)
        if isinstance(version, bool) or version != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported checkpoint version: {version!r}")

        grid_data = _require(data, "grid", dict)
        width = _require(grid_data, "width", int)
        height = _require(grid_data, "height", int)
        if isinstance(width, bool) or isinstance(height, bool):
            raise ValueError("grid dimensions must be integers")
        obstacles_raw = _require(grid_data, "obstacles", list)
        obstacles = frozenset(_parse_position(cell, "obstacle") for cell in obstacles_raw)
        grid = GridMap(width, height, obstacles)

        robots_data = _require(data, "robots", list)
        robots: list[Robot] = []
        seen_robot_ids: set[str] = set()
        for index, entry in enumerate(robots_data):
            if not isinstance(entry, dict):
                raise ValueError(f"robots[{index}] must be an object")
            robot_id = _require(entry, "robot_id", str)
            if not robot_id:
                raise ValueError(f"robots[{index}].robot_id must be a non-empty string")
            if robot_id in seen_robot_ids:
                raise ValueError(f"duplicate robot id: {robot_id!r}")
            seen_robot_ids.add(robot_id)
            position = _parse_position(entry.get("position"), f"robots[{index}].position")
            route = [
                _parse_position(cell, f"robots[{index}].route[{route_index}]")
                for route_index, cell in enumerate(_require(entry, "route", list))
            ]
            task_id = entry.get("task_id")
            if task_id is not None and not isinstance(task_id, str):
                raise ValueError(f"robots[{index}].task_id must be a string or null")
            distance = _require(entry, "distance_travelled", int)
            if isinstance(distance, bool) or distance < 0:
                raise ValueError(f"robots[{index}].distance_travelled must be a non-negative integer")
            robots.append(Robot(robot_id, position, route, task_id, distance))

        tasks_data = _require(data, "tasks", list)
        tasks: list[Task] = []
        seen_task_ids: set[str] = set()
        for index, entry in enumerate(tasks_data):
            if not isinstance(entry, dict):
                raise ValueError(f"tasks[{index}] must be an object")
            task_id = _require(entry, "task_id", str)
            if not task_id:
                raise ValueError(f"tasks[{index}].task_id must be a non-empty string")
            if task_id in seen_task_ids:
                raise ValueError(f"duplicate task id: {task_id!r}")
            seen_task_ids.add(task_id)
            pickup = _parse_position(entry.get("pickup"), f"tasks[{index}].pickup")
            dropoff = _parse_position(entry.get("dropoff"), f"tasks[{index}].dropoff")
            assigned_robot = entry.get("assigned_robot")
            if assigned_robot is not None and not isinstance(assigned_robot, str):
                raise ValueError(f"tasks[{index}].assigned_robot must be a string or null")
            picked_up = _require(entry, "picked_up", bool)
            completed = _require(entry, "completed", bool)
            tasks.append(Task(task_id, pickup, dropoff, assigned_robot, picked_up, completed))

        tick = _require(data, "tick", int)
        if isinstance(tick, bool) or tick < 0:
            raise ValueError("tick must be a non-negative integer")

        replay_data = _require(data, "replay", list)
        replay = _parse_replay(replay_data, tick, seen_robot_ids, seen_task_ids)

        _validate_fleet_state(grid, robots, tasks, tick, replay)

        simulator = cls(grid, robots, tasks)
        simulator.tick = tick
        simulator.replay = replay
        return simulator


def _atomic_write(path: str | os.PathLike[str], payload: str) -> None:
    """Write *payload* to *path* atomically: temp file in the same directory, then replace."""
    target = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(target)
    descriptor, temporary = tempfile.mkstemp(prefix=".fleet-checkpoint-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _require(data: dict[str, object], key: str, kind: type) -> object:
    if key not in data:
        raise ValueError(f"missing required field: {key!r}")
    value = data[key]
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise ValueError(f"field {key!r} must be of type {kind.__name__}")
    return value


def _parse_position(value: object, description: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{description} must be a [x, y] pair")
    x, y = value
    if isinstance(x, bool) or not isinstance(x, int) or isinstance(y, bool) or not isinstance(y, int):
        raise ValueError(f"{description} coordinates must be integers")
    return (x, y)


def _parse_replay(
    replay_data: list[object],
    tick: int,
    robot_ids: set[str],
    task_ids: set[str],
) -> list[dict[str, object]]:
    if len(replay_data) != tick:
        raise ValueError(f"replay length {len(replay_data)} does not match tick {tick}")
    replay: list[dict[str, object]] = []
    for index, event in enumerate(replay_data):
        if not isinstance(event, dict):
            raise ValueError(f"replay[{index}] must be an object")
        event_tick = _require(event, "tick", int)
        if isinstance(event_tick, bool) or event_tick != index + 1:
            raise ValueError(f"replay[{index}].tick must be {index + 1}")
        moved = _require(event, "moved", list)
        for robot_id in moved:
            if not isinstance(robot_id, str):
                raise ValueError(f"replay[{index}].moved entries must be robot id strings")
            if robot_id not in robot_ids:
                raise ValueError(f"replay[{index}].moved references unknown robot: {robot_id!r}")
        positions = _require(event, "robots", dict)
        for robot_id, raw_position in positions.items():
            if robot_id not in robot_ids:
                raise ValueError(f"replay[{index}].robots references unknown robot: {robot_id!r}")
            _parse_position(raw_position, f"replay[{index}].robots[{robot_id}]")
        completed = _require(event, "completed", list)
        for task_id in completed:
            if not isinstance(task_id, str):
                raise ValueError(f"replay[{index}].completed entries must be task id strings")
            if task_id not in task_ids:
                raise ValueError(f"replay[{index}].completed references unknown task: {task_id!r}")
        replay.append(
            {
                "tick": event_tick,
                "moved": list(moved),
                "robots": dict(positions),
                "completed": list(completed),
            }
        )
    return replay


def _validate_fleet_state(
    grid: GridMap,
    robots: list[Robot],
    tasks: list[Task],
    tick: int,
    replay: list[dict[str, object]],
) -> None:
    robot_by_id = {robot.robot_id: robot for robot in robots}
    task_by_id = {task.task_id: task for task in tasks}

    seen_positions: set[tuple[int, int]] = set()
    for robot in robots:
        if not grid.traversable(robot.position):
            raise ValueError(f"robot {robot.robot_id!r} position is outside traversable map space")
        if robot.position in seen_positions:
            raise ValueError(f"robot {robot.robot_id!r} overlaps another robot")
        seen_positions.add(robot.position)
        previous = robot.position
        for index, cell in enumerate(robot.route):
            if not grid.traversable(cell):
                raise ValueError(f"robot {robot.robot_id!r} route[{index}] is outside traversable map space")
            if abs(cell[0] - previous[0]) + abs(cell[1] - previous[1]) != 1:
                raise ValueError(f"robot {robot.robot_id!r} route[{index}] is not adjacent to the previous move")
            previous = cell

    for task in tasks:
        # Task endpoints are not checked against the map: a task whose pickup
        # or dropoff is unreachable simply stays unassigned and waits, which is
        # a valid state the simulator itself produces.
        if task.assigned_robot is not None and task.assigned_robot not in robot_by_id:
            raise ValueError(
                f"task {task.task_id!r} is assigned to unknown robot: {task.assigned_robot!r}"
            )

    for robot in robots:
        if robot.task_id is not None and robot.task_id not in task_by_id:
            raise ValueError(f"robot {robot.robot_id!r} carries unknown task: {robot.task_id!r}")

    for task in tasks:
        # Completed tasks keep their historical owner, who may now be idle or
        # carry a different task; only in-progress ownership must be mutual.
        if task.assigned_robot is None or task.completed:
            continue
        robot = robot_by_id[task.assigned_robot]
        if robot.task_id != task.task_id:
            raise ValueError(
                f"task {task.task_id!r} ownership is inconsistent with robot {robot.robot_id!r}"
            )
    for robot in robots:
        if robot.task_id is None:
            continue
        task = task_by_id[robot.task_id]
        if task.assigned_robot != robot.robot_id:
            raise ValueError(
                f"robot {robot.robot_id!r} carries task {task.task_id!r} owned by another robot"
            )

    if tick > 0:
        last_frame = replay[-1]
        for robot in robots:
            raw_position = last_frame["robots"].get(robot.robot_id)
            if raw_position is None:
                raise ValueError(
                    f"replay last frame is missing robot {robot.robot_id!r} at tick {tick}"
                )
            frame_position = _parse_position(
                raw_position, f"replay[{tick - 1}].robots[{robot.robot_id}]"
            )
            if frame_position != robot.position:
                raise ValueError(
                    f"replay last frame position for robot {robot.robot_id!r} does not match current state"
                )
