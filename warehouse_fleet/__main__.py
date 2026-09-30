"""Command-line demonstration for the local simulator."""

from __future__ import annotations

import argparse
import json

from .model import GridMap, Robot, Task
from .simulator import FleetSimulator


def demo(steps: int) -> dict[str, object]:
    grid = GridMap(8, 6, frozenset({(3, 1), (3, 2), (3, 3), (5, 4)}))
    robots = [Robot("R-01", (0, 0)), Robot("R-02", (7, 5))]
    tasks = [Task("T-100", (1, 4), (6, 0)), Task("T-200", (6, 5), (0, 2))]
    simulator = FleetSimulator(grid, robots, tasks)
    for _ in range(steps):
        simulator.step()
        if simulator.metrics()["tasks_completed"] == simulator.metrics()["tasks_total"]:
            break
    return simulator.snapshot()


def main() -> None:
    parser = argparse.ArgumentParser(prog="warehouse-fleet")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo_parser = subparsers.add_parser("demo", help="run the deterministic sample scenario")
    demo_parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be non-negative")
    print(json.dumps(demo(args.steps), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

