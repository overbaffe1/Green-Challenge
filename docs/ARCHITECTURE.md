# Architecture

## The contract

The simulator is a WebSocket client. It connects to `POLICY_PORT`, receives one
msgpack metadata message, then exchanges one request per control period:

```
server -> client   metadata (dict)
client -> server   {"obs": observation}
server -> client   {"actions_list": [row_0 ... row_19], "server_timing": {...}}
```

Images are RGB `uint8`, HWC or CHW. The model server rotates the head camera 90°
clockwise and both wrist cameras 180°, then resizes to 448×448; nothing upstream of
it may rotate them again.

The observation state is a dictionary, not a packed vector:

```python
{
  "legs_joint_pos": (12,), "torso_joint_pos": (13,),
  "finger_joint_pos": (16,),          # 6 left, 2 left wrist, 6 right, 2 right wrist
  "base_lin_vel": (3,), "base_ang_vel": (3,), "projected_gravity": (3,),
  "root_height": float, "root_quat": (4,),   # w x y z
  "reset": bool, "policy_seed": int,
}
```

Each reply row contains legs (12), torso (13), fingers (12), wrists (4), velocity
(6) and `base_command` (root height, roll, pitch, yaw). The simulator's finger
action term adds the robot's default finger pose, so every finger value on the wire
is **relative**; the model server emits absolute values and the boundary subtracts
the defaults.

## Coordinate systems

Three systems meet in one request, and mixing them up produces a server that looks
healthy and misses every grasp.

| System | Size | Where it appears | Notes |
| --- | --- | --- | --- |
| raw51 | 51 | Dataset `observation.state`, simulator state | legs 0–11, torso 12–24, wrists 25–28, fingers 29–40, attitude/height 41–44, velocity 45–50 |
| model50 | 50 | GreenVLA v1.1 input/output | raw 44 (global yaw) dropped; wrists/fingers reordered |
| wire | 51 | Per-tick command row | 12 legs, 13 torso, 12 fingers, 4 wrists, 6 velocity, 4 base |

`greenchallenge/layout.py` owns the maps and pins them with tests. The model treats
channels `0:25` and `34:50` as deltas and `25:34` as absolute; the adapter adds the
input state to the delta channels before the boundary ever sees a row.

`calibration.swap_wrists` (default on) swaps crank/roll inside each wrist, in both
directions. It is part of the checkpoint's contract, not a tuning knob: the packaged
launcher starts with `S0S1_SWAP_WRISTS=1`.

## The echo problem

The benchmark runs the simulator with `PROPRIO_CMD=all`. That setting does **not**
send the measured joints; it echoes the policy's own last wire command. A model
trained on measured state would therefore see commands as if they were measurements:
fingers would look 0.1 rad away from where they are, and the next prediction would
inherit that offset.

The boundary undoes it:

1. At an episode start (`reset=True`, or a clock that moves backwards) the simulator
   sends the measured pose. Nothing is decoded; that pose seats the limiter.
2. On every later request, the finger defaults are added back and an optional neck
   bias is removed.
3. `GC_PROPRIO=auto` (default in the boundary stack) performs the decode only when
   all 13 torso channels equal the last command within 1e-6 rad. Measured joints
   never coincide that precisely, so a simulator started without `PROPRIO_CMD=all`
   is not corrupted. `all` mirrors the packaged deployment; `measured` disables it.

## The action feed

A model reply is a *plan*, not a command stream. Row `j` describes the moment

```
t_j = t_obs + (j + 1) * row_step + lead        row_step = row_dt / speed
```

on the plan's own clock. The feed turns plans into commands for the 20 ticks between
requests (`period / control_dt = 0.080 / 0.004`):

1. **Scheduling** samples the active plan at each tick, linearly interpolating
   between rows. Before row 0 is due, the plan can only offer row 0; past its last
   row it freezes (unless expiry removes it).
2. **Blending** replaces the setpoint with the weighted mean of every live plan's
   forecast *for now*, `w = exp(-age / tau)`. Averaging forecasts cancels the
   disagreement between consecutive chunks without adding phase lag. The active
   plan never expires. At the defaults the blend peaks at 15 live plans.
3. **Limiting** caps velocity and acceleration against the command actually issued,
   using discrete braking distance so it settles instead of ringing. The default
   selection is both hands, both wrists and the right arm — 21 channels.

Numbers at `GC_SPEED=1.3`: row step 25.64 ms, plan horizon 1.312 s, blend life
1.20 s, request period 80 ms, reply 20 commands. Because the life is shorter than
the horizon, no plan ever contributes its frozen tail. At 1.0× the horizon is 1.70 s
and a stale plan would freeze inside the blend after 1.2 s; `expire_frozen` drops it
instead, which is the only behavioural difference from the reference feed at 1.0×.

### Why the request rate is fixed

The simulator requests every 80 ms and executes one command per 4 ms tick. Returning
more or fewer rows, or scheduling against a different tick, silently shifts every
command in time; the reference server announces `control_dt` in its metadata for
exactly this reason. The feed therefore derives everything from `control_dt`,
`period` and `t`, and refuses chunks that are empty, non-finite or not shaped
`(N, 51)`.

## Episode and instruction boundaries

* `reset=True` or a backwards clock: clear every plan, clear the echo history, seat
  the limiter on the measured pose.
* Instruction change (the simulator's subtask switches mid-episode): drop the old
  plans and re-seat the limiter on the decoded current state. Continuing to blend
  plans produced for the previous subtask is the one case where temporal ensembling
  is clearly wrong.
* Prompt priority: the simulator's `subtask` when `GC_PROMPT_FROM_SUBTASK=1`,
  otherwise `prompt`, otherwise the previous instruction.

## Supervision and failure semantics

The public server and the model are separate processes. The entrypoint waits for the
model's `/healthz` before binding the public port, forwards stdout/stderr, and exits
if either child exits. A healthy proxy in front of a dead model would keep answering
requests with stale commands, which is worse than a visible failure.

Inside the boundary, an upstream error is retried twice (reconnecting the WebSocket);
if it still fails, the error is returned to the simulator as a text frame and the
connection is closed. Rows containing NaN, Inf or wrong channel counts are rejected
rather than clamped into something plausible.

## Telemetry

The boundary writes one JSON object per request (`GC_TELEMETRY`, default
`/tmp/greenchallenge/tracking.jsonl`): simulation time, instruction, rows in/out,
live plans, frozen drops, adaptive rows, inference latency, command discontinuity at
the seam, requests/resets. It is enough to reconstruct tempo and clamping decisions
after a run without touching the simulator. Telemetry never blocks inference: a
write error disables the file and logs one warning.

The official stack writes the tracker's own log (`--log`), which records the last
original and last wire row per request.
