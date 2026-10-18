# Green Challenge 2026 — submission stack and playbook

A submission for the [Green Challenge 2026](https://dsworks.ru/en/champ/aij26-green):
a whole-body VLA policy served over the benchmark's WebSocket protocol, packaged as
a Docker image.

The model is the organisers' GreenVLAv1.1 baseline. This repository contributes the
part around it that decides *when* each predicted row is executed, plus the tooling
to measure whether a change helped.

## Why this exists

The contest score is a raw sum over completed subtasks:

```
Speed_{i,g} = clamp(1 - t_{i,g} / T_max, 0, 1)
score       = Σ Speed_{i,g}
```

`T_max` is the episode timeout (52 s / 70 s / 35 s for the three scenarios), and an
incomplete subtask contributes 0. With the published run config — 8 episodes per
scenario and 4 + 5 + 2 subtasks — the maximum is **88**. The target for this work is
**> 68.14**, which is 77.4 % of that maximum: it requires both high completion and
fast execution.

The same checkpoint scores about 15.4 on the public partial leaderboard with the reference feed, 18.1 when the
execution layer replays the model's chunk 1.3× faster, and about 7 % success with no
feed at all. Execution quality is not a detail; it is a third of the problem.

## What is in the box

| Piece | Purpose |
| --- | --- |
| `greenchallenge/` | Execution layer: msgpack/WebSocket server, observation↔command calibration, action feed (scheduling, blending, limiting, expiry), telemetry. |
| `docker/` | Submission image: starts from the contest's `green-vla` image, keeps its checkpoint and CUDA stack, replaces the entrypoint. |
| `eval/` | Benchmark wrapper and a `results.json` report that decomposes the score and sweeps playback speed. |
| `training/` | Dataset validator and the fine-tuning recipe for the contest data. |
| `tests/`, `tools/` | CPU test suite and a two-hop smoke test that needs no GPU, checkpoint or simulator. |

## Two stacks, one contract

`GC_STACK=official` (default) is the organisers' own topology:

```
simulator ── ws://host:POLICY_PORT ──> serve_tracking ──> serve_policy (--feed-by-server)
                                      (wire boundary)      (model + action feed)
```

The only change from the baseline image is the feed's row interval, which is what
turns 1.0× into the measured 1.3×. Nothing about the model, the images or the
instructions changes.

`GC_STACK=boundary` runs `serve_policy` without its feed and puts this package's
execution layer on the public port. It is the experiment bench: the same scheduling
and limiting as the reference feed, plus plan expiry and optional adaptive speed.
The two-hop transport and timing are tested end to end on CPU with a scripted policy
(peak of 15 live plans, matching the reference feed's documented behaviour).

## Quick start

Полная последовательность команд для своей машины: [`docs/COMMANDS.md`](docs/COMMANDS.md).

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt

make test                      # the CPU test suite
python tools/smoke_test.py     # full transport + feed, scripted policy
```

Docker (on a machine that can reach the contest registry):

```bash
docker build -f docker/Dockerfile \
  --build-arg BASE_IMAGE='aij-green-challenge.cr.cloud.ru/green-vla@sha256:<digest>' \
  -t green-challenge:local .
docker run --gpus all --rm -p 8999:8999 \
  -e POLICY_PORT=8999 -e POLICY_RAW_PORT=8998 -e GC_SPEED=1.3 \
  green-challenge:local
```

The container answers `GET /healthz` on `POLICY_PORT` and the contest WebSocket
protocol on the same port.

## The speed knob

`GC_SPEED` scales the playback of every predicted chunk:

```
row_step = GC_ROW_DT / GC_SPEED        (default 1/30 / 1.3 = 1/39 s)
```

Rows keep their shape; only their timestamps are compressed. The feed's 0.080 s
request period and the 20-command reply are unchanged, so the simulator sees exactly
the same protocol.

* `1.0` reproduces the reference feed.
* `1.3` is the measured public optimum for this checkpoint (+17 % on the published
  partial leaderboard, and no loss of completion).
* Above ~1.42 the blend can outlive the chunk; `GC_EXPIRE_FROZEN=1` (default)
  removes stale plans instead of letting them freeze on their last row.
* `GC_ADAPTIVE_SPEED` (boundary stack) slows rows around detected grasps. The
  published experiment with a row-wise detector failed; use `net`, and only after
  an A/B run.

## Configuration

Everything is an environment variable; every default is the safe one.

| Variable | Default | Meaning |
| --- | --- | --- |
| `POLICY_PORT` | `8999` | Public port the simulator connects to. |
| `POLICY_RAW_PORT` | `8998` | Loopback port of the raw model server. |
| `CHECKPOINT_PATH` | `/ckpt` | Checkpoint directory baked into the image. |
| `GC_STACK` | `official` | `official` or `boundary`. |
| `GC_SPEED` | `1.3` | Chunk playback rate. |
| `GC_ROW_DT` | `1/30` | Trained row interval, before `GC_SPEED`. |
| `GC_TELEMETRY` | `/tmp/greenchallenge/tracking.jsonl` | Per-request telemetry (official stack writes the tracker's log). |
| `GC_ADAPTER` | `none` | `v1` enables the packaged learned joint-tracking correction. |
| `GC_NECK_COMPENSATION` | `0` | Packaged empirical neck bias. |
| `GC_PROPRIO` | `auto` | Echo decoding: `all`, `measured` or `auto`. |
| `GC_PROMPT_FROM_SUBTASK` | `1` | Use the simulator's atomic instruction. |
| `GC_MAX_SPEED_*`, `GC_MAX_ACCEL_*` | reference values | Joint limits of the feed. |
| `GC_ADAPTIVE_*` | off (`1.0`) | Optional speed modulation (boundary stack). |

The entrypoint refuses to start if a fixed-pose or clock-scheduled override is set
(`S0S1_TORSO_PITCH_OFFSET`, `S0S1_SQUAT_POSE_JSON`, `S0S1_PROMPT_SCHEDULE_JSON`, …).
Those are disqualifying under the rules; see `docs/COMPLIANCE.md`.

## Documentation

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the contract, the timing math, the feed.
* [`docs/EVALUATION.md`](docs/EVALUATION.md) — how the score is computed and how to A/B safely.
* [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — build, run, roll back, three-submission plan.
* [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md) — rules mapped to design decisions.
* [`docs/FINDINGS.md`](docs/FINDINGS.md) — what the official stack and public experiments imply.
* [`docs/SCORE.md`](docs/SCORE.md) — the completion/speed budget for the 68.14 target.
* [`docs/COMMANDS.md`](docs/COMMANDS.md) — copy-paste runbook from a bare GPU host to a submission.
* [`docs/TRAINING.md`](docs/TRAINING.md) — fine-tuning on `GreenChallengeData`.

## What is verified, and what is not

Verified in this sandbox (CPU only): the full test suite; NumPy/msgpack wire round-trip; the simulator's
three observation layouts; echo decoding; the feed against an independent
reimplementation of its formula and against the reference implementation for a
1.3× schedule (max difference 4 × 10⁻¹⁶); the two-hop WebSocket pipeline with a
scripted policy; configuration and the disqualification guard.

Not verified here: the real checkpoint, CUDA kernels, the Isaac Sim benchmark and
therefore the actual score. This sandbox has no GPU, no dataset and no access to
`huggingface.co`. The numbers quoted above are the organisers' and a public
participant's published measurements, not a local result.

## License

MIT for this code. The GreenVLA model, simulator and assets remain under their own
licenses and are not redistributed here.
