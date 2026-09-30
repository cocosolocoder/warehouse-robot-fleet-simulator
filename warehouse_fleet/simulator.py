"""Task assignment and collision-aware tick execution."""

from __future__ import annotations

from dataclasses import asdict

from .model import GridMap, Robot, Task
from .pathfinding import shortest_path


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

