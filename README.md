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

## Saving and resuming

A run can be saved to a versioned UTF-8 JSON file at any point after a tick and
resumed later, even after the process has exited:

```bash
python3 -m warehouse_fleet demo --steps 6 --checkpoint /tmp/run.json
python3 -m warehouse_fleet resume /tmp/run.json --steps 6
```

`demo` keeps its original output and early-stop behaviour; `--checkpoint` only
additionally saves progress after the run. `resume` loads a checkpoint, runs at
most `--steps` ticks (stopping early once every task is complete), and prints the
same snapshot JSON structure as `demo`. With `--steps 0` or an already-complete
fleet it prints the current state without advancing. Pass `--checkpoint` to
`resume` to write progress (the same path updates the file in place); when omitted
the source file is only read.

The Python API mirrors this:

```python
simulator.save_checkpoint("/tmp/run.json")            # does not mutate the simulator
restored = FleetSimulator.load_checkpoint("/tmp/run.json")  # independent instance
```

Saving and loading do not advance the tick. Two instances loaded from the same
file share no mutable state, so stepping one leaves the other untouched. Resuming
replays the exact per-tick output, task state, metrics, and replay of an
uninterrupted run; existing task assignments and remaining routes are preserved,
completed tasks keep their historical owner, and temporarily unreachable tasks
stay unassigned and waiting.

### Checkpoint file format

The file is UTF-8 JSON with a top-level `version` field (currently `1`). It is
written atomically: a temporary file in the same directory is fully written and
fsynced, then replaced over the target, so a failure or interrupted process
leaves either the previous complete file or no file at all — never a partial
checkpoint.

```json
{
  "version": 1,
  "grid": {
    "width": 8,
    "height": 6,
    "obstacles": [[3, 1], [3, 2], [3, 3], [5, 4]]
  },
  "robots": [
    {
      "robot_id": "R-01",
      "position": [0, 0],
      "route": [[1, 0], [1, 1]],
      "task_id": "T-100",
      "distance_travelled": 5
    }
  ],
  "tasks": [
    {
      "task_id": "T-100",
      "pickup": [1, 4],
      "dropoff": [6, 0],
      "assigned_robot": "R-01",
      "picked_up": true,
      "completed": false
    }
  ],
  "tick": 5,
  "replay": [
    {
      "tick": 1,
      "moved": ["R-01"],
      "robots": {"R-01": [1, 0]},
      "completed": []
    }
  ]
}
```

Coordinates are `[x, y]` integer pairs. `grid` holds the map dimensions and
obstacle cells. Each robot stores its id, current position, remaining route
(positions still to traverse), the id of the task it is currently carrying (or
`null`), and its cumulative distance travelled. Each task stores its id, pickup
and dropoff positions, the id of the robot it is assigned to (or `null`), and its
pickup/completion flags; completed tasks retain their historical `assigned_robot`.
`tick` is the number of ticks already executed, and `replay` is the per-tick log
with consecutive tick numbers from `1` to `tick`.

Loading rejects malformed JSON, missing or mistyped fields, unsupported versions,
duplicate robot or task ids, overlapping robots, robots out of bounds or on an
obstacle, routes that leave the map, cross obstacles, or make non-adjacent moves,
inconsistent task ownership, and a replay whose ticks are not consecutive from 1
or whose final frame does not match the current robot positions — all with a
`ValueError` explaining the reason. File access and permission failures raise
`OSError`. The CLI reports either error on stderr and exits non-zero without
printing a success snapshot.

