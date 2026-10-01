"""Command-line demonstration for the local simulator."""

from __future__ import annotations

import argparse
import json
import sys

from .model import GridMap, Robot, Task
from .simulator import FleetSimulator


def _build_demo_simulator() -> FleetSimulator:
    grid = GridMap(8, 6, frozenset({(3, 1), (3, 2), (3, 3), (5, 4)}))
    robots = [Robot("R-01", (0, 0)), Robot("R-02", (7, 5))]
    tasks = [Task("T-100", (1, 4), (6, 0)), Task("T-200", (6, 5), (0, 2))]
    return FleetSimulator(grid, robots, tasks)


def _all_tasks_complete(simulator: FleetSimulator) -> bool:
    metrics = simulator.metrics()
    return metrics["tasks_completed"] == metrics["tasks_total"]


def demo(steps: int, checkpoint: str | None = None) -> dict[str, object]:
    simulator = _build_demo_simulator()
    for _ in range(steps):
        simulator.step()
        if _all_tasks_complete(simulator):
            break
    snapshot = simulator.snapshot()
    if checkpoint is not None:
        simulator.save_checkpoint(checkpoint)
    return snapshot


def resume(path: str, steps: int, checkpoint: str | None = None) -> dict[str, object]:
    simulator = FleetSimulator.load_checkpoint(path)
    for _ in range(steps):
        if _all_tasks_complete(simulator):
            break
        simulator.step()
    snapshot = simulator.snapshot()
    if checkpoint is not None:
        simulator.save_checkpoint(checkpoint)
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(prog="warehouse-fleet")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo_parser = subparsers.add_parser("demo", help="run the deterministic sample scenario")
    demo_parser.add_argument("--steps", type=int, default=12)
    demo_parser.add_argument(
        "--checkpoint",
        default=None,
        help="optional path to save a checkpoint after the run",
    )
    resume_parser = subparsers.add_parser(
        "resume", help="continue a run from a checkpoint file"
    )
    resume_parser.add_argument("file", help="checkpoint file to resume from")
    resume_parser.add_argument("--steps", type=int, required=True, help="maximum ticks to run")
    resume_parser.add_argument(
        "--checkpoint",
        default=None,
        help="optional path to write progress to; when omitted the source file is only read",
    )
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be non-negative")
    try:
        if args.command == "demo":
            snapshot = demo(args.steps, args.checkpoint)
        else:
            snapshot = resume(args.file, args.steps, args.checkpoint)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
