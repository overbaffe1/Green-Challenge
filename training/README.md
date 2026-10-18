# Training

`validate_dataset.py` checks a downloaded LeRobot dataset against the benchmark
contract before any GPU time is spent:

```bash
python training/validate_dataset.py data
```

The fine-tuning recipe — data contract, language strategy, normalisation statistics,
robustness measures, packaging and evaluation — is in
[`docs/TRAINING.md`](../docs/TRAINING.md). It is written against the public GreenVLA
code and has not been executed in this repository.

Weights and datasets are not tracked in Git. `model/` (checkpoint + data config) is
the build context for `docker/Dockerfile.finetuned`.
