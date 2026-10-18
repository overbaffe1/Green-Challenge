# Compliance

The contest rules are short and the failure mode is a zero score, so each rule is
mapped to a concrete design decision and, where possible, to code that enforces it.

## The control signal comes from the model

| Requirement | How this submission satisfies it |
| --- | --- |
| Every command is produced by the model from the current observation | The boundary forwards the observation of the current request to the model and converts the rows it returns. It stores no trajectory, no episode script and no clock-indexed table. |
| No replay of recorded trajectories | There is no dataset, demonstration or episode file in the image, and no code path that reads one. The only persistent state is the feed's in-memory plans and the limiter's last issued command. |
| No fixed joint positions or commands substituted for, or applied on top of, model output | The feed schedules the model's rows in time, averages forecasts of consecutive model plans, and limits velocity/acceleration. It never adds an offset or overwrites a channel with a constant. The organiser's reference server ships the same three mechanisms; this implementation reproduces them (verified to 4 × 10⁻¹⁶ against the reference for a 1.3× schedule). |
| No time-indexed per-episode commands | Nothing in the package indexes by episode number or wall clock. The only clock is the simulator's own `t`, used to decide which predicted row is due now. |
| No disabling or substituting the model on any task subset | One code path for all three scenarios. Task identity is never read; the instruction and the observation are the only inputs. |

The limiter is seeded on the measured pose at an episode start. That is an initial
condition for the limiter, not a command: the first row sent to the simulator is
still the model's, shaped by the limiter exactly as the reference feed shapes it.

`GC_ADAPTIVE_SPEED` (boundary stack, off by default) is computed from the model's
own predicted finger rows, not from a clock or a scene lookup. It changes how fast
rows are played, never their values.

## The disqualification guard

`greenchallenge/config.py` refuses to start when any of these is set to a
non-neutral value:

```
S0S1_TORSO_PITCH_OFFSET   S0S1_ROOT_HEIGHT_OFFSET   S0S1_LEG_SQUAT_FRAC
S0S1_SQUAT_POSE_JSON      S0S1_ARM_OFFSETS          S0S1_FINGER_GAIN
S0S1_THUMB_PITCH_MIN      S0S1_PROMPT_SCHEDULE_JSON GC_SCRIPTED_TRAJECTORY
```

They are exactly the packaged overrides that replace model output with a fixed pose,
a squat, an arm offset or a timed instruction schedule. The check is deliberately
conservative: a typo that re-enables one of them fails the container at startup
instead of during an evaluation.

## Docker and runtime

| Requirement | Implementation |
| --- | --- |
| Self-contained, no internet at runtime | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, no download code. The image is built from the contest image that already bakes the checkpoint and the Qwen snapshot. |
| All weights and dependencies baked in | Nothing is mounted at run time; `CHECKPOINT_PATH=/ckpt` is inside the image. |
| ≤ 24 GB VRAM | Same model, same precision, same batch of one as the organisers' image; the boundary runs on CPU. |
| ≤ 50 GB image | Only a few hundred kilobytes are added to the base image. |
| `ENTRYPOINT ["python", "/app/docker/entrypoint.py"]` | Declared in `docker/Dockerfile`. |
| Ports from `POLICY_PORT` / `POLICY_RAW_PORT` | Read in `config.py`; the raw port is loopback and the public port is `0.0.0.0`. |
| Logs to stdout/stderr | Both children inherit the container's stdout/stderr; Python is unbuffered. |
| ≤ 4 h benchmark | Unchanged from the baseline; a 50-row chunk at 30 fps is ~1.7 s of plan and inference is a single forward pass per 80 ms. |

## Contest rules

* **One submission per day, best score counts.** The three-slot plan is in
  `docs/RUNBOOK.md`; each slot has one isolated change.
* **Private sharing outside the team is banned.** This repository is the team's
  workspace. Do not publish the image, the configuration or the measured results
  publicly before the contest ends.
* **No proprietary third-party technology.** Runtime dependencies are NumPy,
  msgpack and websockets, all already present in the contest image; the model and
  simulator belong to the organisers.
* **Simulator licence only during the contest.** Nothing from the simulator image
  is redistributed here; the submission image is derived from the organisers' own
  image at build time.

## Audit checklist before a submission

1. `git grep -nE "trajectory|replay|episode_[0-9]|scripted_pose"` returns only the
   compliance guard and documentation.
2. `make test` passes; the echo-decode and feed equivalence tests are the ones that
   would catch a semantics regression.
3. The container starts with `--network none` and answers `/healthz`.
4. The handshake metadata shows `robot=scripted_014814`, `action_horizon=50`,
   `action_sample_step=1`, `feed_by_server=true`, `control_dt=0.004`.
5. A `GC_SPEED=1.0` benchmark reproduces the reference score within noise.
6. The submitted image reference is a digest, not a mutable tag.
