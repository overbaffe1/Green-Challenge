# Runtime configuration

The container reads environment variables at startup. The default mode uses the
organisers’ tracking server and action feed. Use `GC_STACK=boundary` to run this
repository’s action feed in front of the raw model server.

| Variable | Default | Purpose |
| --- | --- | --- |
| `POLICY_PORT` | `8999` | Simulator-facing WebSocket and `/healthz`. |
| `POLICY_RAW_PORT` | `8998` | Loopback port of the raw model server. |
| `CHECKPOINT_PATH` | `/ckpt` | Checkpoint directory baked into the image. |
| `GC_STACK` | `official` | `official` or `boundary`. |
| `GC_SPEED` | `1.3` | Playback rate. `1.0` restores the reference row interval. |
| `GC_ROW_DT` | `1/30` s | Trained interval between rows. Effective step is `GC_ROW_DT / GC_SPEED`. |
| `GC_TELEMETRY` | `/tmp/greenchallenge/tracking.jsonl` | Per-request trace. In `official`, this is the tracking server’s own log. |
| `GC_ADAPTER` | `none` | In `official`, `v1` enables the packaged joint-tracking correction. |
| `GC_NECK_COMPENSATION` | `0` | In `official`, enables the packaged neck compensation. |
| `GC_PROPRIO` | `auto` | In `boundary`, decode command echo with `all`, `measured` or `auto`. |
| `GC_PROMPT_FROM_SUBTASK` | `1` | In `boundary`, prefer the simulator’s atomic instruction. |
| `GC_EXPIRE_FROZEN` | `1` | In `boundary`, remove expired plans from the blend. |
| `GC_MAX_SPEED_*`, `GC_MAX_ACCEL_*` | Reference values | Velocity and acceleration limits in `boundary`. |
| `GC_ADAPTIVE_SPEED` | `1.0` (off) | In `boundary`, optionally vary playback rate around grasp/release events. |

`GC_SPEED` changes the timestamps of the predicted rows. The 80 ms request
period and the expected 20 commands at 4 ms remain the same. See the
[action feed](ARCHITECTURE.md#the-action-feed) for the model and
[`RuntimeConfig.from_env`](../greenchallenge/config.py) for all supported flags.

The entrypoint refuses to start with non-neutral fixed-pose or clock-scheduled
overrides such as `S0S1_SQUAT_POSE_JSON`; see [Compliance](COMPLIANCE.md).
For build and run commands, see the [GPU runbook](RUNBOOK.md).
