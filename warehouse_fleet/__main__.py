"""Command-line demonstration for the local simulator."""

from __future__ import annotations

import argparse
import json
import sys

from .model import GridMap, Robot, Task
from .simulator import FleetSimulator


def _all_tasks_done(simulator: FleetSimulator) -> bool:
    metrics = simulator.metrics()
    return metrics["tasks_completed"] == metrics["tasks_total"]


def demo(steps: int, checkpoint: str | None = None) -> dict[str, object]:
    grid = GridMap(8, 6, frozenset({(3, 1), (3, 2), (3, 3), (5, 4)}))
    robots = [Robot("R-01", (0, 0)), Robot("R-02", (7, 5))]
    tasks = [Task("T-100", (1, 4), (6, 0)), Task("T-200", (6, 5), (0, 2))]
    simulator = FleetSimulator(grid, robots, tasks)
    for _ in range(steps):
        simulator.step()
        if _all_tasks_done(simulator):
            break
    if checkpoint is not None:
        simulator.save_checkpoint(checkpoint)
    return simulator.snapshot()


def resume(source: str, steps: int, checkpoint: str | None = None) -> dict[str, object]:
    simulator = FleetSimulator.load_checkpoint(source)
    for _ in range(steps):
        # Already-complete runs (including an empty task list) report state
        # without advancing another tick.
        if _all_tasks_done(simulator):
            break
        simulator.step()
        if _all_tasks_done(simulator):
            break
    if checkpoint is not None:
        simulator.save_checkpoint(checkpoint)
    return simulator.snapshot()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="warehouse-fleet")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="run the deterministic sample scenario")
    demo_parser.add_argument("--steps", type=int, default=12)
    demo_parser.add_argument(
        "--checkpoint",
        metavar="FILE",
        help="save simulator state to FILE after the run",
    )

    resume_parser = subparsers.add_parser("resume", help="continue a simulator from a checkpoint")
    resume_parser.add_argument("file", help="checkpoint file to resume from")
    resume_parser.add_argument("--steps", type=int, required=True, help="maximum ticks to run")
    resume_parser.add_argument(
        "--checkpoint",
        metavar="FILE",
        help="save updated state to FILE; defaults to leaving the source untouched",
    )

    args = parser.parse_args(argv)
    if args.steps < 0:
        parser.error("--steps must be non-negative")

    try:
        if args.command == "demo":
            snapshot = demo(args.steps, args.checkpoint)
        else:
            snapshot = resume(args.file, args.steps, args.checkpoint)
    except (ValueError, OSError) as exc:
        print(f"{args.command}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
