# Evaluation

## The metric, exactly

From the simulator's `scripts/results_aggregator.py`:

```
Speed_{i,g} = clamp(1 - t_{i,g} / T_max, 0, 1)   for completed subtasks
score       = Σ over episodes i and subtasks g of Speed_{i,g}
```

* `t_{i,g}` is the simulation-time duration of subtask `g` in episode `i`
  (`subtask_history`), not wall time.
* `T_max` is the task's timeout relation: 52 s (`dry_kitchen.e3.a2`),
  70 s (`sber_shop.e1.a2`), 35 s (`darkstore.e1.a2`).
* An incomplete subtask contributes 0. The episode-level `speed` and
  `success_rate` in `results.json` are diagnostics; the run `score` is the sum above.
* With the published run config the maximum is `8 × (4 + 5 + 2) = 88`.

Consequences:

1. **Completion is worth more than speed.** A subtask that never finishes loses its
   entire speed contribution, and the subtasks after it may not start.
2. **The same `T_max` covers every subtask of a scenario**, so a slow first subtask
   is expensive twice: it burns its own budget and delays the ones after it.
3. **A 1.3× playback multiplies every `(1 - Speed)` term by 0.77.** If a subtask took
   10 s of a 52 s budget (speed 0.81), it becomes 7.7 s (speed 0.85).
4. Scores are integers-ish at the extremes: zero completed subtasks is a zero score,
   regardless of how good the motion looked.

## Running the benchmark

On a GPU host with the GreenSim checkout and assets downloaded:

```bash
# terminal 1 — the policy under test
docker run --gpus all --rm --network host \
  -e POLICY_PORT=8999 -e POLICY_RAW_PORT=8998 -e GC_SPEED=1.3 \
  green-challenge:local

# terminal 2 — the benchmark
cd ~/green_challenge
make up RUN_BENCHMARK=1 POLICY_PORT=8999 \
  RUN_PATH=assets/runs/green_challenge.json \
  RESULTS_PATH=$PWD/results.json
```

`network_mode: host` in the simulator's compose file is why the policy is reachable
at `127.0.0.1:8999`. The simulator must run with its documented calibration:

```
PROPRIO_CMD=all  ARM_FF=0  ABS_HAND_TARGETS=1  CAMERA_FTHETA=1
START_POSE=canonical  SIM_SEED=42  ACTION_HORIZON=50
```

`vla_env.sh` in the simulator repository sets these; `run_with_vla.sh` and
`run_benchmark.sh` source it. If they are not in the simulator's actual environment
the policy receives measured joints while believing they are commands — the
symptom is a policy that twitches at the start of every episode and never settles.

A run takes up to the 4 h benchmark limit. The output lands in
`benchmark_logs/<run_name>_<timestamp>/results.json`, with one `reports/` JSON per
episode.

## Reading a result

```bash
python eval/report.py benchmark_logs/green_challenge_*/results.json --sweep
```

The report prints the run score, completion, and per-scenario score/success. The
`--sweep` table recomputes the score as if every completed subtask had been `k`
times faster. It is an **upper bound**: it ignores the effect of playback speed on
success, which only the A/B run measures.

Typical readings:

| Symptom | Likely cause |
| --- | --- |
| Subscore 0, `subtask_completed` 0 | The model is not receiving the instruction it expects, or the state decode is wrong. Check the handshake metadata and the tracker log. |
| First subtask completes, later ones stall | Prompt is not switching: `GC_PROMPT_FROM_SUBTASK` or the simulator's `subtask` field. |
| Everything completes but slowly | Playback speed or the feed's row interval. Sweep `GC_SPEED`. |
| Hands close but miss | Finger defaults / wrist swap mismatch. Compare the tracker log's `last_wire` against the simulator report. |
| Motion is smooth for 1.3 s then jerks | Blend life longer than the chunk horizon at that speed. Keep `life ≤ horizon` or enable expiry. |

## A/B protocol

Eight episodes per scenario is a small sample; a single run is not enough to accept
a change. Use this protocol for every candidate:

1. Same host, same simulator commit, same `SIM_SEED` for A and B.
2. Run A (baseline settings) and B (candidate) **alternating**, at least three times
   each. Do not run all A then all B: thermal state and cache warmth drift.
3. Compare the run `score`, then `subtask_completed` per scenario. A gain in score
   with a loss in completion is a fragile change; prefer the version that keeps
   completion first.
4. Keep the per-run `results.json` and the policy's telemetry next to each other;
   the timing of a regression is usually visible in both.

`eval/report.py a/results.json b/results.json --compare` prints the per-scenario
delta and completion counts.

## Guards before spending a submission slot

* `make test` and `python tools/smoke_test.py` pass on the CPU host.
* The image starts with no network (`--network none` is enough to prove there is no
  download at boot) and `/healthz` answers within the start timeout.
* A benchmark run with the *baseline* settings (`GC_SPEED=1.0`) reproduces the
  reference score within noise. If it does not, the image is not serving the same
  model and no other comparison is meaningful.
* One run with `GC_STACK=boundary` is compared against `official` before the
  boundary stack is considered for a submission.

## Cost of a wrong assumption

The contest allows one submission per day and the best score counts. A submission
that fails to start, or that serves the wrong semantics, scores zero for that day but
does not remove earlier results. The runbook therefore keeps a known-good image tag
and a one-variable rollback.
