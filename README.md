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
simulator.status()              # current obstacles + paused task ids + traffic waits
simulator.map_change_history()  # effective tick, adds/removes, order
```

`status()` reports tasks paused because the map made their route infeasible,
which is distinct from a robot briefly waiting for another robot to clear the
way. Every *actual* change is stored both in the dedicated history and in the
replay as a `map_change` event carrying the tick it took effect at, its global
sequence number (consecutive edits at the same tick keep their order), and the
added/removed cells; ordinary tick frames keep their original positions and
completion data and are never rewritten by later maps. Identical initial state
with an identical sequence of steps and edits always produces identical
routes, replay and statistics.

## Automatic yielding in traffic

When a robot's next waypoint is occupied by another robot, `step()` no longer
just waits: an idle blocker is asked to step onto a free neighbouring cell
that lies on no robot's planned route, and a busy robot whose blocker cannot
drive on may sidestep onto a free adjacent cell itself and replan its
(still shortest) route from there. Every robot still moves at most one
orthogonal cell per tick (or waits), robots never share a cell at the end of
a tick, never swap places within one tick, and never enter an obstacle.

A sidestep that cannot make progress is not taken. If the replanned route
steps straight back through the cell the robot is vacating, still runs past
the same blocker's cell, and the blocker cannot use the opening to get out of
the way (for example an idle robot parked at the end of a one-wide corridor
with no neighbouring cell to yield onto), the robot waits instead of
retreating and shuttling back to the identical blockage on the next tick.
More empty cells behind the waiting robot do not change this. A sidestep that
genuinely routes around the blocker, or a retreat the blocker immediately
drives into to clear the way, is still driven.

Yielding moves count as mileage; waiting does not. The task keeps its
original robot and the pickup rules are unchanged — goods are collected
before delivery, a loaded robot never returns to the pickup cell, and nothing
completes early because of a detour. Robots paused by map unreachability stay
put and never take part in yielding. An idle robot that yields is not hired
by doing so: it stays on its side cell once the way is clear and can still
pick up tasks later. If no safe side cell exists, both robots simply wait —
time advances, cargo and tasks are kept, and nothing collides or completes.

Traffic waits are reported separately from map pauses:

```python
simulator.status()["traffic_waits"]
# [{"robot_id": "R-01", "blocked_by": ["R-02"], "ticks": 3}, ...]
```

A robot's counter only grows while another robot blocks its next waypoint and
it does not move; any move or the blockage clearing resets it to zero. The
same report appears in `metrics()` (and therefore in the command-line
output), and it is saved into and restored from checkpoints — older
checkpoint files without it load with every counter at zero.


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
  "distance_travelled"}`; `position` is `[x, y]` and `route` is the remaining
  waypoints after the current position (empty for a paused robot).
- `tasks` — list of `{"task_id", "pickup", "dropoff", "assigned_robot",
  "picked_up", "completed"}`. Completed tasks keep their historical
  `assigned_robot`; unassigned tasks use `null`.
- `paused_tasks` — ids of tasks whose assigned robot currently cannot reach a
  required point on the saved map.
- `traffic_waits` — optional list of `{"robot_id", "blocked_by", "ticks"}`
  entries: robots currently held up by other robots, who blocks them, and for
  how many consecutive ticks. Older files without this field load with every
  counter at zero.
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
paused-task state are all rejected. So is a replay whose *history* could never
have happened, even when its final frame matches the saved robots: a robot out
of bounds or on an obstacle that existed at that tick, two robots sharing a
cell at a tick boundary, a jump of more than one orthogonal cell or a two-robot
cell swap between neighboring tick frames, and a `moved` list that names an
unknown/duplicated robot or does not exactly match the robots that changed
position (following another robot into the cell it just vacated is legal, and
the list need not be sorted). Map edits are judged against contemporary
positions: a cell a robot once passed through may be closed later without
invalidating the replay, a robot may not appear in a cell before it opens, and
an edit may not add an obstacle onto a cell occupied when that tick ended —
not even if another edit of the same tick removes it again. The first tick
frame has no recorded predecessor, so its cells, overlaps and `moved` ids are
still checked but the first moves are not second-guessed; zero-tick files,
empty fleets and all-waiting frames remain valid. Version 1 documents contain
tick frames without a `type` field and none of the edit-related keys; they
load as if no map edit had ever happened.


