# What it takes to score above 68.14

The score is additive, so the requirement can be written down exactly.

Let `E_s` be the episodes per scenario, `N_s` the subtasks per episode, `c_s` the
fraction of subtasks completed and `v_s` their mean speed. Then

```
score ≈ Σ_s E_s · N_s · c_s · v_s
```

The evaluation runs 8 episodes per scenario (`E = 8`), with `N = 4, 5, 2` subtasks —
24 episodes and **88 subtask slots** in total. The public leaderboard is the final
one; there is no hidden test set with different scenes.
and the target is 77.4 % of the maximum.

## The completion/speed frontier

Assuming the same completion rate and mean speed in every scenario:

| completion | v = 0.70 | 0.75 | 0.80 | 0.85 | 0.90 |
| --- | --- | --- | --- | --- | --- |
| 100 % | 61.6 | 66.0 | 70.4 | 74.8 | 79.2 |
| 95 % | 58.5 | 62.7 | 66.9 | 71.1 | 75.2 |
| 90 % | 55.4 | 59.4 | 63.4 | 67.3 | 71.3 |
| 85 % | 52.4 | 56.1 | 59.8 | 63.6 | 67.3 |

Read it as a budget, not a prediction:

* **100 % completion needs a mean speed of 0.775** to reach 68.14.
* **90 % completion needs 0.86.**
* **85 % cannot reach it at all**, even at 0.90.
* A missed subtask costs up to 1.0 point *and* delays every subtask after it, so the
  table is optimistic about partial completion.

A mean speed of 0.775 means a mean subtask duration of `0.225 × T_max`:

| Scenario | `T_max` | Budget per subtask at 100 % completion |
| --- | --- | --- |
| `dry_kitchen.e3.a2` | 52 s | 11.7 s |
| `sber_shop.e1.a2` | 70 s | 15.8 s |
| `darkstore.e1.a2` | 35 s | 7.9 s |

These are whole-subtask budgets including approach, grasp, transport and release.
They are the numbers to look at in a report, not the run total.

## What each lever is worth

* **Completion** is worth up to one point per subtask, plus everything downstream.
  It is the only lever that can move 63 → 68 on its own.
* **Playback speed** multiplies `(1 - v)` by `1/k`. At `v ≈ 0.70`, 1.3× gives
  `v ≈ 0.77` — about +6 points over 88 slots if completion holds. The public
  partial-leaderboard measurement (15.445 → 18.096) is consistent with that.
* **Feed tuning** (blend life, expiry, clamp limits) moves tenths of a point. Worth
  doing after completion is stable, not before.
* **Adaptive speed** can add on top of 1.3× only if the detector is conservative.
  The published row-wise detector fired on ~60 % of rows and lost score; a net
  detector at 1.5–2.0 is the version worth an A/B.

## Order of operations

1. Reproduce the baseline at `GC_SPEED=1.0` and record per-scenario completion and
   score.
2. Submit `GC_SPEED=1.3` (slot 1). Confirm completion does not fall.
3. Fine-tune for completion; evaluate with the same protocol. A fine-tune that adds
   two completed subtasks per round is worth more than any feed change.
4. Re-apply 1.3× to the fine-tuned model; the two gains multiply.
5. Only then try adaptive speed or a second checkpoint as an ensemble.

A useful sanity rule: if a candidate raises the run score but lowers
`subtask_completed`, it is probably overfitting the timing of the episodes already
seen, and it will not survive a different seed.
