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

## Taking over work already in progress

A fleet does not have to start idle. `FleetSimulator` accepts robots that are
already carrying goods partway along a route, together with their in-flight
tasks and any completed-task history, and continues the work from there:

```python
from warehouse_fleet import FleetSimulator, GridMap, Robot, Task

grid = GridMap(3, 3, obstacles=frozenset({(1, 0)}))

# R-1 collected T-1's goods earlier and is halfway to the dropoff. The
# remaining route lists the cells still ahead, next cell first; it heads
# straight to the dropoff and never returns to the pickup cell.
carrier = Robot(
    "R-1",
    position=(0, 0),
    route=[(0, 1), (1, 1), (2, 1), (2, 0)],
    task_id="T-1",
    distance_travelled=5,      # mileage already on the clock
)
in_flight = Task(
    "T-1",
    pickup=(0, 2),
    dropoff=(2, 0),
    assigned_robot="R-1",      # must name the carrier back
    picked_up=True,            # goods already collected
)
# A finished delivery kept as history. The robot that completed it may be
# retired, absent from this fleet, or busy with a new task.
history = Task(
    "T-0",
    pickup=(1, 2),
    dropoff=(2, 2),
    assigned_robot="R-9",      # historical record; R-9 is not in this fleet
    picked_up=True,
    completed=True,
)

simulator = FleetSimulator(grid, [carrier], [in_flight, history])

# Creation itself changes nothing: the clock is zero, the replay is empty,
# the robot has not moved, and the mileage and completion marks it came
# with are kept exactly as passed in.
assert simulator.tick == 0 and simulator.replay == []
assert carrier.position == (0, 0) and carrier.distance_travelled == 5
assert history.completed

for _ in range(4):             # one tick per step(); each actual move adds
    simulator.step()           # exactly one cell of mileage

assert carrier.position == (2, 0)      # route exhausted at the dropoff
assert carrier.distance_travelled == 9 # 5 kept + 4 cells driven
assert carrier.task_id is None         # delivery released the binding
assert in_flight.completed and in_flight.assigned_robot == "R-1"

simulator.metrics()
# {"ticks": 4, "tasks_total": 2, "tasks_completed": 2, ...}
# tasks_completed counts the preserved T-0 record as well as T-1.
```

Ownership of in-flight work is mutual: a robot's `task_id` must name an
existing, unfinished task whose `assigned_robot` names that robot back, and a
task whose goods are already collected (`picked_up=True`) may never lack its
carrier. Completed tasks are fixed historical records, not work in progress:
each must keep both the picked-up mark and the id of the robot that finished
it, but that robot is bound by nothing — it may be missing from the fleet,
idle, or already executing another task, and several records may name the
same robot. History never takes part in new task assignment and never
occupies a robot, while its completions still count in `metrics()`.

The remaining route describes a walk from the robot's current cell: the first
waypoint is the *next* cell (never the current one), every later waypoint is
exactly one orthogonal step from the previous one, and every waypoint lies
inside the map and outside the initial obstacles. Legal detours and revisits
of already-seen cells are fine; the route need not be shortest. The two entry
points judge a route differently: direct construction only checks that the
walk is legal — it never plans or pads the pickup and delivery legs for you —
while checkpoint loading additionally requires every in-flight task to be
completable along its robot's remaining route. Either way, accepting the
input is not a promise that the task will finish: hand in a route that
actually reaches the dropoff, as the example does, and the robot will drive
those cells in order, complete the task on arrival, and release the binding
while the task keeps its robot's id.

Inconsistent input is rejected at construction time with a `ValueError`
naming the robot or task involved — a robot bound to an unknown or already
completed task, a task claiming a robot that executes something else, a
collected task with no carrier, or a completion record missing its picked-up
mark or historical robot. A rejected fleet commits nothing: every robot and
task the caller passed in is left exactly as it was. The same ownership and
completed-history rules govern checkpoint loading, so a fleet one entry point
accepts is always a state the other can reload, and the in-flight robots
behave like any other fleet under map edits, traffic yielding, and
save/restore.

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

A robot without a bound task that is still driving a preset route is treated
differently: closing a cell on its remaining route neither clears nor replans
that route. The robot keeps every unconsumed waypoint, drives the cells still
open before the closure, and then simply waits in front of the closed cell —
waiting consumes no waypoint and adds no mileage. When the cell reopens it
continues the same route in the same order; it never picks a shortcut and
never skips the blocked waypoint. Such a waiting state saves and restores
through checkpoints (see below).

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
just waits: a parked blocker — one with no bound task *and* no remaining
route — is asked to step onto a free neighbouring cell that lies on no
robot's planned route, and a busy robot whose blocker cannot drive on may
sidestep onto a free adjacent cell itself and replan its (still shortest)
route from there. A taskless robot that still carries a remaining route is
not parked: it drives that route cell by cell just like a task-bound robot,
is never pushed off it for another robot, and when its own next cell is
occupied it simply stays put with its unconsumed waypoints — an open side
cell changes nothing. Only once that route is finished does it become a
parked robot that yields by the side-cell rule. Every robot still moves at
most one orthogonal cell per tick (or waits), robots never share a cell at
the end of a tick, never swap places within one tick, and never enter an
obstacle.

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

A robot's counter only grows while it stops for traffic and another robot
still blocks its next waypoint once the tick's moves have settled; any move or
the blockage clearing resets it to zero. A blocker that advances later in the
same tick is not a blockage at tick end, so that tick adds nothing. If the
wait never breaks but the robot holding the next cell merely changes, the
count continues with the new blocker; once the wait was actually interrupted,
a later blockage starts again at one. The
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
obstacles unexplained by the change history, make non-adjacent moves, or
cannot complete the robot's bound task, task/robot ownership mismatches, a
broken replay history, out-of-bounds or add/remove-conflicting change
records,
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
empty fleets and all-waiting frames remain valid. Beyond physical
plausibility, each assigned, unfinished, non-paused task must still be
completable along its robot's remaining route: a non-empty route has to end at
the dropoff, and a robot that has not collected yet — unless it is already
standing on the pickup cell — must still pass through the pickup (visiting the
dropoff early and leaving again is fine; detours need not match a replanned
shortest path). An empty route is accepted only for a loaded robot already at
the dropoff, or when position, pickup and dropoff all coincide; map-paused
tasks keep their empty-route exemption. Version 1 documents contain
tick frames without a `type` field and none of the edit-related keys; they
load as if no map edit had ever happened.

The one blocked-waypoint route a version 2 checkpoint accepts belongs to a
robot with no bound task that is waiting out a dynamic closure: the blocked
waypoint must have been traversable on the initial map, a valid map change
must have closed it, and it must still be closed in the saved grid — exactly
the state normal running produces when a preset route reaches a cell that was
later shut. Saving and loading preserve the position, the remaining waypoint
order, the mileage and the change history without advancing time or moving
the robot; on resume the robot finishes the still-open waypoints, waits
without consuming a waypoint or gaining mileage, and continues the original
route from the next cell once the way reopens, never detouring or skipping.
The exemption is narrow: creating a fleet directly still rejects a route
through an initial obstacle, a blocked waypoint that the change history
cannot explain is rejected, task-bound robots keep their reroute/pause rules,
and version 1 files keep the old requirement. A robot standing on an
obstacle, an out-of-map or non-integer waypoint, and a jump between waypoints
are rejected regardless.


