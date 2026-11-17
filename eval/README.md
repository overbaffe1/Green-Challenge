# Evaluation

`run_benchmark.sh` wraps the simulator's `make up` with the environment it expects
and copies `results.json` somewhere durable. `report.py` explains the score.

```bash
# policy container is already running on 127.0.0.1:8999
GREEN_SIM_DIR=~/green_challenge ./eval/run_benchmark.sh

python eval/report.py results.20261018T101500Z.json --sweep
python eval/report.py baseline/results.json candidate/results.json --compare
```

The `--sweep` table is an upper bound: it rescales completed subtask durations and
ignores the effect of speed on completion. The `--compare` view is the one that
decides a submission; it shows the per-scenario score delta and completion counts.
See `docs/EVALUATION.md` for the A/B protocol.
