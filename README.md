# Warehouse Robot Fleet Simulator

A small, deterministic Python simulator for experimenting with warehouse maps,
robot fleets, task assignment, collision avoidance, route execution, replay, and
efficiency metrics. It runs locally and has no runtime dependencies outside the
Python standard library.

## Quick start

```bash
python3 -m warehouse_fleet demo --steps 12
```

The command builds a sample grid, assigns pickup-and-delivery tasks to available
robots, advances the fleet one tick at a time, and prints a JSON replay summary.

Run the tests with:

```bash
python3 -m unittest discover -s tests -v
```

The public Python API exposes `GridMap`, `Robot`, `Task`, `FleetSimulator`, and
`shortest_path` for programmatic scenarios.

## Saving and resuming progress

State can be saved after any tick and resumed later in a new process. From
Python, `FleetSimulator.save_checkpoint(path)` writes the state without
advancing the tick or mutating the simulator, and the class method
`FleetSimulator.load_checkpoint(path)` returns an independent simulator:
loading the same file twice gives two instances whose subsequent runs never
interfere with each other.

```python
simulator.save_checkpoint("fleet.json")
restored = FleetSimulator.load_checkpoint("fleet.json")
```

Writes are atomic: an existing checkpoint file always contains either the
complete old contents or the complete new contents, and a failed save to a new
path leaves no partial file behind. Filesystem and permission failures raise
`OSError`; malformed or inconsistent checkpoint content raises `ValueError`
with the reason.

The command line can save at the end of a run and continue from a checkpoint:

```bash
python3 -m warehouse_fleet demo --steps 6 --checkpoint fleet.json
python3 -m warehouse_fleet resume fleet.json --steps 20
```

`resume FILE --steps N` runs at most `N` more ticks and stops early once every
task is complete. When the checkpoint is already complete, or `N` is `0`, the
current state is printed without advancing. Add `--checkpoint OUT` to save the
updated state; without it the source file is left untouched, and naming the
same file updates it in place. Both commands print the same JSON snapshot and
report any load/save error on standard error with a non-zero exit code.

### Checkpoint file format (version 1)

Checkpoints are UTF-8 encoded JSON. The top-level object contains:

- `version` — integer format version, currently `1`.
- `grid` — `{"width", "height", "obstacles"}`, where obstacles is a list of
  `[x, y]` cells.
- `tick` — number of ticks already executed.
- `robots` — list of `{"robot_id", "position", "route", "task_id",
  "distance_travelled"}`; `position` is `[x, y]` and `route` is the remaining
  waypoints after the current position.
- `tasks` — list of `{"task_id", "pickup", "dropoff", "assigned_robot",
  "picked_up", "completed"}`. Completed tasks keep their historical
  `assigned_robot`; unassigned tasks use `null`.
- `replay` — per-tick frames `{"tick", "moved", "robots", "completed"}`,
  numbered consecutively from `1` through `tick`. In any non-zero-tick file the
  last frame's robot positions must equal the current robot positions.

Loading validates the entire document: corrupted JSON, missing or wrongly
typed fields, unsupported versions, duplicate robot or task ids, overlapping
or out-of-bounds/blocked robot positions, routes that leave the map, cross
obstacles, or make non-adjacent moves, task/robot ownership mismatches, and a
broken replay history are all rejected.


