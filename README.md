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

## Closing and reopening cells at runtime

Between ticks you may change the map with one batch call. The edit takes effect
immediately but neither advances the clock, moves a robot, nor adds mileage:

```python
simulator.modify_obstacles(added=[(3, 2), (3, 3)], removed=[(5, 4)])
```

Coordinates must be plain two-integer pairs (booleans are not integers). The
whole request is validated up front and rejected with `ValueError` if a
coordinate is out of bounds or malformed, a cell is both added and removed, or
a newly added obstacle covers a robot's current cell; on rejection the map,
routes, tasks, statistics and history are all left untouched. Duplicate
coordinates inside one batch are processed once; adding an already blocked
cell or removing a traversable cell is a no-op, and a batch with no actual
effect produces no change record.

After an accepted edit every assigned, unfinished task keeps its robot. A robot
that has not picked up replans a shortest feasible route from its current cell
through the pickup point to the dropoff (a robot already at the pickup point
collects first at the next step); a robot carrying goods routes straight to the
dropoff and never returns to a subsequently closed pickup cell. If any required
cell is unreachable, the robot's remaining route is cleared and the task is
*paused* — it is neither driven along a stale route, released, nor completed.
Unassigned tasks whose points are unreachable simply keep waiting and never
block other tasks. The next `step()` after cells reopen automatically resumes
paused driving and assigns waiting tasks; collision avoidance still applies and
robots never enter a new obstacle.

Queries and history:

```python
simulator.status()              # current obstacles + paused tasks + traffic waits
simulator.map_change_history()  # effective tick, adds/removes, order
```

`status()` distinguishes two kinds of interruption. A task is *paused* when the
map made its route infeasible (e.g. a required cell was closed); paused robots
stay in place and never yield. A robot is *traffic waiting* when another vehicle
blocks its desired move; `status()` lists each waiting robot with the blocking
vehicle's id and the number of consecutive ticks it has been stationary. The
wait count resets to zero as soon as the robot moves or the blockage clears.

## Automatic yielding

When a robot's desired move is blocked by another vehicle, it does not simply
wait forever. If a safe adjacent side cell is free — one that is traversable,
not occupied or reserved, not another robot's desired destination, and not on
any other robot's shortest path — the robot may step into it as a *yield* move.
After yielding the robot replans its route from the new position and continues.
Yield moves obey the same safety rules as every move: at most one adjacent cell
per tick, no two robots end on the same cell, no position swaps, and no passing
through obstacles. Yield moves count toward mileage; waiting does not.

When the blocking vehicle is idle (has no task), it is the idle vehicle that
yields to let the active robot pass, rather than the active robot stepping
backward into an oscillation. Idle vehicles that yield generate no task, stay
in the side cell once the way is clear, and remain available for new
assignments. Robots paused by map unreachability stay in place and never yield.

If no safe yield cell exists (e.g. the side cell is closed and the corridor is
narrow), the robots wait: `step()` still advances time and preserves tasks and
cargo, with no collisions or false completion. Reopening a closed side cell
lets the robots try to pass again on the next tick; stale yield routes are never
reused.

Every *actual* change is stored both in the dedicated history and in the
replay as a `map_change` event carrying the tick it took effect at, its global
sequence number (consecutive edits at the same tick keep their order), and the
added/removed cells; ordinary tick frames keep their original positions and
completion data and are never rewritten by later maps. Identical initial state
with an identical sequence of steps and edits always produces identical
routes, replay and statistics, independent of robot or task list ordering.

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

### Checkpoint file format (version 2)

Checkpoints are UTF-8 encoded JSON. The top-level object contains:

- `version` — integer format version, currently `2`. Version `1` files remain
  loadable and are interpreted as having no map-modification history.
- `grid` — `{"width", "height", "obstacles"}`, the map *currently in effect*,
  where obstacles is a list of `[x, y]` cells.
- `base_grid` — same shape as `grid`; the map before any runtime edit. Replaying
  the change history against it must reproduce `grid`.
- `tick` — number of ticks already executed.
- `robots` — list of `{"robot_id", "position", "route", "task_id",
  "distance_travelled", "wait_ticks"}`; `position` is `[x, y]` and `route` is
  the remaining waypoints after the current position (empty for a paused robot).
  `wait_ticks` is the number of consecutive ticks the robot has spent
  stationary because of traffic; it resets on move and is `0` when the robot is
  not waiting. Version 1 and older version 2 files without this field load with
  `wait_ticks` starting from `0`.
- `tasks` — list of `{"task_id", "pickup", "dropoff", "assigned_robot",
  "picked_up", "completed"}`. Completed tasks keep their historical
  `assigned_robot`; unassigned tasks use `null`.
- `paused_tasks` — ids of tasks whose assigned robot currently cannot reach a
  required point on the saved map.
- `map_changes` — every actual edit in order, each
  `{"tick", "sequence", "added", "removed"}`; `sequence` runs from 1 and
  consecutive edits at the same `tick` preserve their order.
- `replay` — ordered events. Tick frames are `{"type": "tick", "tick", "moved",
  "robots", "completed"}`, numbered consecutively from `1` through `tick`.
  Map-change frames are interspersed at their effective tick and have the shape
  `{"type": "map_change", "tick", "sequence", "added", "removed"}`. In any
  non-zero-tick file the last tick frame's robot positions must equal the
  current robot positions; historical frames are never overwritten by later
  maps.

Loading validates the entire document: corrupted JSON, missing or wrongly
typed fields, unsupported versions, duplicate robot or task ids, overlapping
or out-of-bounds/blocked robot positions, routes that leave the map, cross
obstacles, or make non-adjacent moves, task/robot ownership mismatches, a
broken replay history, out-of-bounds or add/remove-conflicting change records,
a change history that does not reproduce the saved grid, and an inconsistent
paused-task state are all rejected. Version 1 documents contain tick frames
without a `type` field and none of the edit-related keys; they load as if no
map edit had ever happened.


