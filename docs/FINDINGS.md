# Findings behind the design

Everything below comes from public sources, read on 2026-10-18. It is recorded here
because the numbers are easy to lose and they justify the defaults. None of it is a
local measurement: this sandbox has no GPU, no simulator and no access to
`huggingface.co`.

## The official stack

Source: [`greenvla/GreenSim`](https://github.com/greenvla/GreenSim) and the
`aij_contest_inference` branch of
[`greenvla/GreenVLA`](https://github.com/greenvla/GreenVLA).

* The contest entrypoint starts two processes:
  `serve_policy` on `127.0.0.1:$POLICY_RAW_PORT` with `--feed-by-server
  --feed.row-dt 1/30 --warmup-steps 2`, and `serve_tracking` on
  `0.0.0.0:$POLICY_PORT` with the packaged wire calibration.
* Default environment: `S0S1_PROMPT_FROM_SUBTASK=1`, `S0S1_SWAP_WRISTS=1`,
  `S0S1_GMM_GUARD_MODE=off`. The learned adapter and neck compensation are off.
* The simulator side runs `PROPRIO_CMD=all ARM_FF=0 ABS_HAND_TARGETS=1
  CAMERA_FTHETA=1 SIM_SEED=42 ACTION_HORIZON=50`, at 250 Hz (4 ms ticks).
* The checkpoint is GreenVLAv1.1: 50 state and 50 action channels, flow matching,
  10 integration steps, Qwen3.5-0.8B frozen backbone, 50-row chunk, sample step 1.
* Reply: 20 commands per request, one per 4 ms tick. Request period 80 ms.

The reference feed's own documentation gives the three mechanisms and their
measured defaults: scheduling on the plan's clock, blending forecasts with
`exp(-age/tau)`, and a velocity/acceleration limiter with discrete braking. At the
defaults the blend peaks at **15 live plans**. Our implementation reproduces this
number in the CPU smoke test, which is the cheapest end-to-end check that the
scheduling is right.

## Confirmed by the organisers

* The evaluation simulator runs with `PROPRIO_CMD=all`: the policy receives its
  own previous command, not the measured joints. (Answer in the contest thread,
  2 October.)
* There is no private leaderboard; the public one is final. The task set, scenes,
  objects and goals are fixed at 8 episodes per scenario; only `SIM_SEED` changes.
* Smoothing, filtering, joint-tracking calibration and handing the model's last
  command back into the state are allowed. Multiple models with routing, and two
  action experts with a learned visual switch, are allowed.
* One submission per day, best score counts, deadline 2026-10-20 23:59 MSK, and
  images must stay in the registry for possible recomputation.

## What the score rewards

Source: the simulator's `scripts/results_aggregator.py`, quoted in
`docs/EVALUATION.md`.

* `score = Σ clamp(1 - t_subtask / T_max)` over completed subtasks, a raw sum.
* With 8 episodes and 4 + 5 + 2 subtasks, the maximum is 88; the target 68.14 is
  77.4 % of it.
* Speed is not gated on success, but an incomplete subtask contributes zero and
  stops the ones after it.

## Public experiments with the same checkpoint

Source: [`fanot/miniature-dollop`](https://github.com/fanot/miniature-dollop), a
research repository by a participant (Russian; the private toolkit it references is
not published). Numbers are from the public partial leaderboard as reported there.

| Configuration | Score |
| --- | --- |
| Baseline | 15.445 |
| 1.15× playback | 15.635 |
| **1.30× playback** | **18.096** |
| Adaptive around events, 1.5× | 15.358 |
| Packaged tracking adapter `v1` | 14.936 |
| Fine-tune 1 | 14.0 |
| Fine-tune 2 | 15.6 |

Two conclusions shaped this repository:

1. **Playback rate is the reliable lever.** 1.3× is a 17 % gain with no loss of
   completion; the adapter and two fine-tunes did not beat it.
2. **Naive adaptive speed is a trap.** Their row-wise finger-travel detector fired on
   59.9 % of rows (2.3 % of demonstrations), so the plan was slowed almost
   everywhere. A windowed "net travel" detector fired on 2.8 % of rows at
   `span=4`, threshold 0.15 rad. That is why the default here is
   `GC_ADAPTIVE_SPEED=1.0` (off), the detector default is the conservative level
   crossing, and the recommendation is to A/B a net detector at 1.5–2.0 only after
   the 1.3× build has a measured score.

Their tempo analysis is also worth keeping: with a request period `s`, a row interval
`d`, lead `ℓ`, mean row index `ā` and mean plan age `δ`, the deficit between the
planned and executed tempo is

```
u/v = (δ·d + s·(ā − ℓ)) / (ā + δ·s)
```

A plan older than its horizon does not expire in the reference feed — it freezes on
its last row and drags the blend backwards. At 1.0× the horizon is 1.70 s and the
blend life 1.20 s, so it does not bite; above ~1.42× it can. `expire_frozen`
(boundary stack, on by default) drops those plans instead.

## Implications

* The default submission is the official stack at 1.3×. It is measured, it is the
  organisers' own code path, and it changes no model semantics.
* The custom boundary exists to make the *next* experiments cheap: expiry, adaptive
  speed, blend-skip on fingers, period changes. Each is one environment variable and
  one A/B run.
* Fine-tuning is the only route to the completion rate that 68+ requires; speed
  alone cannot close a 2× gap. `docs/TRAINING.md` is the path, and it is the least
  verified part of this repository.
* Completion first, speed second. A candidate that trades a completed subtask for a
  faster one loses: one subtask is worth up to 1.0 point, while 0.1× extra playback
  on an 11-subtask round is worth about 0.2 points.

## Caveats

* The leaderboard numbers above are from a *partial* evaluation, not the full
  88-point run; do not scale them to predict 68.
* The tempo formula and detector rates were measured on a planar stand-in
  (`TempoBench`), not on the Isaac Sim benchmark; they are hypotheses to test.
* The fork has changed shape more than once (16 interleaved finger channels vs
  12 + 4 split wrists). The boundary accepts both on input and always emits the
  12 + 4 wire form; this is tested.
