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

## Taking over in-flight work

A fleet does not have to start idle: `FleetSimulator` accepts robots and tasks
that are already mid-work, so a scenario can be handed over exactly where an
earlier run (or another tool) left off. Construction only validates and adopts
the state — it never advances the clock, moves a robot, or rewrites mileage or
completion flags:

```python
from warehouse_fleet import FleetSimulator, GridMap, Robot, Task

grid = GridMap(width=4, height=2, obstacles=frozenset({(1, 0)}))

# R-1 collected T-1's goods earlier and is mid-delivery, detouring around
# the obstacle at (1, 0) towards the dropoff at (3, 0).
carrier = Robot(
    "R-1",
    position=(0, 0),
    route=[(0, 1), (1, 1), (2, 1), (3, 1), (3, 0)],
    task_id="T-1",
    distance_travelled=5,
)
in_flight = Task("T-1", pickup=(0, 0), dropoff=(3, 0),
                 assigned_robot="R-1", picked_up=True)
# A finished delivery kept as history; R-9 need not be in this fleet.
history = Task("T-0", pickup=(2, 0), dropoff=(3, 1),
               assigned_robot="R-9", picked_up=True, completed=True)

simulator = FleetSimulator(grid, [carrier], [in_flight, history])
```

Right after construction the clock is zero, the replay is empty, no robot has
moved, and the mileage and task flags passed in stand unchanged:

```python
simulator.status()
# {'tick': 0, 'width': 4, 'height': 2, 'obstacles': [[1, 0]],
#  'paused_tasks': [], 'traffic_waits': []}
simulator.metrics()
# {'ticks': 0, 'tasks_total': 2, 'tasks_completed': 1, ...}  # T-0 already counts
```

Because T-1 is already picked up, its remaining route leads only to the
dropoff: the robot walks those cells one per tick and never returns to the
pickup cell. Each `step()` advances the clock by one, and each cell a robot
actually enters adds one to its mileage:

```python
for _ in range(5):
    simulator.step()

simulator.robots["R-1"].position            # (3, 0)
simulator.robots["R-1"].task_id             # None — released on delivery
simulator.robots["R-1"].distance_travelled  # 10 = 5 carried over + 5 driven
simulator.tasks["T-1"].completed            # True
simulator.tasks["T-1"].assigned_robot       # 'R-1' — kept as the finisher
simulator.metrics()["tasks_completed"]      # 2 — history plus the new delivery
```

Once the route is exhausted on the dropoff cell the task completes, the robot
is unbound from it (and becomes available for new assignments), and the task
keeps the id of the robot that finished it.

### Unique identifiers

Each robot id may appear at most once in the robot list and each task id at
most once in the task list of a construction request; the two lists have
independent namespaces, so a robot and a task may share one string. The
identifier alone decides — nothing about the rest of two records can make a
clash legal:

- two robots with the same id but different positions (or the same position),
  different idle state, preset routes or current tasks;
- two tasks with the same id but different pickup/dropoff points, owners or
  completion state — including one already finished beside one still waiting;
- the very same object, or an identical copy, passed in twice.

In every such case the whole request is rejected with `ValueError` — the
message says whether the clash is among the robots or the tasks and quotes the
duplicated id. No record is kept in preference to another, nothing is merged,
and nothing is renamed. The check is read-only and runs before any other
normalization commits, so even a duplicate at the end of either list leaves
every earlier `Robot` and `Task` exactly as supplied — positions, list-shaped
routes and points, bindings, mileage and completion flags all keep their
original containers and values.

A task's `assigned_robot` is only a reference, never a second robot entry:
several completed tasks may keep the same historical robot, which may even be
absent from the fleet. Conversely, tasks with distinct ids but identical
pickup/dropoff points are separate tasks — both are kept, assigned and counted
independently.

### Ownership of in-flight tasks

Work still in progress must be bound consistently in both directions: a robot
that names a current task must name one that exists, is unfinished, and is
assigned back to that same robot, and an unfinished task that names an owner
must name an existing robot whose current task it is. A task whose goods are
already collected may never lack its carrier — `picked_up=True` with
`assigned_robot=None` is rejected. One robot can therefore never be claimed by
two unfinished tasks.

A completed task is a fixed historical record, not work in progress. It must
carry both the picked-up mark and the id of the robot that finished it (a
`None` owner means the record is missing, not an anonymous completion), but
that robot is history only: it need not be in the fleet, may be idle, or may
already work another task. Historical records never occupy a robot and never
re-enter task assignment — but they do count towards `metrics()` completion,
as the `tasks_completed: 1` above shows. Unassigned, not-yet-collected tasks
simply wait and are not an ownership error.

Inconsistent input is rejected as a whole: binding a robot to a completed
task, pointing a robot at a task owned by someone else, or any other
contradiction above raises `ValueError` naming the robot or task involved.
Nothing is rebound, unassigned, or dropped, and the caller's `Robot` and
`Task` objects are left exactly as passed in — positions, routes and bindings
included.

```python
stray = Robot("R-2", position=(1, 1), task_id="T-0")  # T-0 is completed
FleetSimulator(grid, [carrier, stray], [in_flight, history])
# ValueError: robot 'R-2' executes task 'T-0' that is already completed
```

### Task coordinates

Every task's `pickup` and `dropoff` must be a list or tuple holding exactly
two plain integers, and the two spellings may be mixed freely within one task
or batch. Booleans are not integers, and a float is rejected even when it
equals an integer value (`1.0` is not `1`); strings, `None`, scalars, missing
or extra components, and any other non-coordinate input raise `ValueError`
naming the task and whether its pickup or dropoff is at fault. Nothing is
rounded, padded, or type-converted. Accepted points are normalized to fresh
integer tuples — `[1, 0]` and `(1, 0)` name the same cell, so container type
can never make a robot miss a pickup or fail a delivery — and the copy means
mutating a coordinate list you passed in after construction never changes the
fleet. The rule applies to tasks in every state (waiting, assigned, loaded,
completed) and, like positions, commits only once the whole batch passes, so a
later bad point, ownership conflict, or route error leaves every earlier
`Robot` and `Task` object — containers, routes, bindings, mileage and flags —
exactly as passed in. Coordinate validity is separate from reachability: a
well-formed task no robot can currently reach simply keeps waiting, and a
loaded task whose pickup cell has since been closed keeps the goods and its
owner.

### Remaining routes for in-flight tasks

A robot's remaining route omits its current cell: the first waypoint must be
an up/down/left/right neighbour of the current position, and every later
waypoint exactly one orthogonal step from the previous one. Every waypoint
must lie inside the map and outside the initial obstacles. The route need not
be shortest — legal detours and revisiting old cells are fine — and it is
judged purely as a walk: crossing another robot's cell is traffic to be
settled while stepping, not an error.

The two entry points differ in how much they promise. Direct construction
checks that the route is a legal walk on the given map; it does not fill in
missing pickup or dropoff legs for you, so accepting a route is not a
guarantee that the bound task will complete along it — the example above
works because its route was chosen to actually reach the dropoff.
`FleetSimulator.load_checkpoint` is stricter: every in-flight task must be
completable along its robot's remaining route (a non-empty route ends at the
dropoff, and a robot that has not collected yet must still pass the pickup).
Fleets built through either entry point then behave identically: map edits,
yielding and save/restore apply to them exactly as described below.

## Assigning tasks to robots

`FleetSimulator.assign_tasks()` binds waiting tasks to available robots. It is
also called once inside every `step()`, so a fleet left to run assigns work on
its own; call it directly when you want to assign (or inspect an assignment)
without advancing the simulation.

**Which tasks are considered.** Only tasks with no `assigned_robot` are
pending — assigned, loaded and completed tasks never re-enter assignment.
Pending tasks are processed one at a time in the **string order of their task
ids**: this is plain lexicographic text ordering, so `"T-10"` sorts before
`"T-2"` (the digits inside an id are not interpreted as a number), and the
order is well-defined only because task ids are unique within a fleet.

**Which robots are available.** A robot must have no bound task *and* no
remaining route. A taskless robot still driving a preset route is skipped —
even when it sits closest to the pickup — and keeps its waypoints, position,
mileage and cargo state untouched; only once that route is exhausted does the
robot join the selection at the next assignment pass, planned from wherever
it stands then. Being briefly blocked by traffic does not count as having
finished the route either.

**How the winner is chosen.** For each task, in turn, every available robot
plans a shortest feasible route on the *current* map from its own cell through
the pickup point to the dropoff point, and the robot with the shortest such
route wins. The measure is the length of an actual walkable path around
obstacles — never the straight-line distance between coordinates — so a robot
that looks nearer to the pickup can lose to one with a clear corridor. Equal
route lengths are broken by robot id in string order. Because tasks are
processed one by one, a robot bound to an earlier task is no longer available
for the next task of the same pass: this is greedy per-task assignment, not a
batch plan that minimizes the fleet's total travel.

**When no robot qualifies, the task simply waits.** If no available robot can
walk the full pickup-then-dropoff route, the task keeps `assigned_robot=None`,
occupies no robot, and does not prevent later tasks in the same pass from
being assigned. A task whose pickup is reachable but whose dropoff is not is
treated exactly the same — it is *not* assigned first in the hope that the
dropoff opens up later; the whole route must be feasible at assignment time.
(Separately, a robot paused by a map edit keeps its task; see
`status()["paused_tasks"]`.)

**What the route looks like.** The planned route stored on the robot omits
the robot's current cell and starts with the next cell to enter. A robot not
yet standing on the pickup point gets a route that passes through the pickup
and ends at the dropoff; a robot already on the pickup cell gets only the
dropoff leg. The route is a plan, not a promise of immediate execution.

**Assignment changes no physics.** `assign_tasks()` itself only sets
`task.assigned_robot`, `robot.task_id` and `robot.route`. It does not advance
the clock, move any robot, add mileage, or mark anything picked up or
completed — not even for a robot already standing on its pickup cell. The
pickup and delivery rules run inside `step()`: at the start of each tick a
robot standing on its pickup point collects the goods first and then drives,
and each cell actually entered adds one to `distance_travelled`.

### A complete assignment example

```python
from warehouse_fleet import FleetSimulator, GridMap, Robot, Task

#       x=0     x=1     x=2     x=3     x=4
# y=0   .       R-1     #       T-01 pk .
# y=1   T-02 dr .       #       T-01 dr .
# y=2   R-3     .       T-03 pk .       R-2
# T-02 picks up where R-1 stands; T-03's dropoff is the obstacle (2, 0).
grid = GridMap(width=5, height=3, obstacles=frozenset({(2, 0), (2, 1)}))

robots = [
    Robot("R-1", position=(1, 0)),
    Robot("R-2", position=(4, 2)),
    # No task, but still driving a preset route: not available yet.
    Robot("R-3", position=(0, 2), route=[(1, 2)]),
]
tasks = [
    Task("T-01", pickup=(3, 0), dropoff=(3, 1)),
    Task("T-02", pickup=(1, 0), dropoff=(0, 1)),
    Task("T-03", pickup=(2, 2), dropoff=(2, 0)),  # dropoff unreachable
]

simulator = FleetSimulator(grid, robots, tasks)
simulator.assign_tasks()
```

R-1 sits two straight-line cells from T-01's pickup while R-2 is three away,
yet the wall at column 2 forces R-1 to detour six steps to reach `(3, 0)`
(plus one to the dropoff, seven in total), while R-2 walks `(4, 2) → (4, 1) →
(4, 0) → (3, 0)` and on to the dropoff in four. Feasible route length, not
straight-line distance, decides, so T-01 goes to R-2. T-02 is processed next:
R-2 is now bound, R-3 is still driving its preset route and is skipped, so
R-1 — already standing on the pickup cell — takes it with only the dropoff
leg to drive. T-03's dropoff is the obstacle cell `(2, 0)`: no robot can
finish that route, so T-03 stays unassigned without blocking anything. Right
after `assign_tasks()` the clock is still zero, nothing has moved, no mileage
has accrued and no goods are marked collected — only ownership and routes
exist:

```python
simulator.tick                              # 0
simulator.tasks["T-01"].assigned_robot      # 'R-2'
simulator.tasks["T-02"].assigned_robot      # 'R-1'
simulator.tasks["T-03"].assigned_robot      # None — waits, blocks nothing
simulator.robots["R-2"].task_id             # 'T-01'
simulator.robots["R-2"].route               # [(4, 1), (4, 0), (3, 0), (3, 1)]
simulator.robots["R-1"].route               # [(0, 0), (0, 1)] — pickup leg empty
simulator.robots["R-3"].route               # [(1, 2)] — preset route untouched
simulator.robots["R-2"].distance_travelled  # 0
simulator.tasks["T-02"].picked_up           # False — not collected by assigning
```

R-2's route starts at the cell after its position, passes the pickup `(3, 0)`
and ends at the dropoff `(3, 1)` — checkable against the map above. Only
`step()` advances the world: at the start of the tick R-1, already on its
pickup cell, collects T-02's goods before driving off, and every robot
drives one route cell (or waits):

```python
simulator.step()
simulator.tick                              # 1
simulator.tasks["T-02"].picked_up           # True — collected before driving
simulator.robots["R-1"].position            # (0, 0)
simulator.robots["R-1"].distance_travelled  # 1
simulator.robots["R-2"].position            # (4, 1) — one route cell per tick
simulator.robots["R-3"].position            # (1, 2) — preset route now finished

simulator.step()
simulator.tasks["T-02"].completed           # True — R-1 reached (0, 1) and released
simulator.robots["R-1"].task_id             # None
```

With its preset route exhausted after the first step, R-3 is available at the
next assignment pass — here only the unreachable T-03 is left, so it stays
idle; had a reachable task been waiting, that pass would have considered R-3
from `(1, 2)` like any other free robot.

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


