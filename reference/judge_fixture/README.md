# Submission-zip fixture (not the Green Challenge format)

Copied out of the 3.45 GB `submission.zip` that `artifacts/submission-zip` holds
(see `docs/ARTIFACTS.md`). The archive is a *demo fixture* for a judge that loads
model weights directly from a zip:

```
submission.zip
  smolvlm2/              SmolVLM2 weights, 2.03 GB
  action_expert/         SmolVLA policy (LeRobot format), 1.42 GB
  README.md, config.yaml, src/pipeline.py, annotations.jsonl, ...
                         stubs — "Judge Score uses only smolvlm2/ and action_expert/"
```

Why it is filed here: it shows that on this platform the *other* submission route
is a zip of `smolvlm2/` + `action_expert/`, where the judge rewrites
`vlm_model_name` to point inside the unpacked directory. The numbers in
`config.json` are LIBERO-shaped (state 8, action 7, 2×256×256 cameras, chunk 8,
`type: smolvla`), not the humanoid Green Challenge contract (state 51, action 50,
3×448×448, chunk 50). So this archive is **not** a Green Challenge submission and
must not be confused with one; it is kept as a format reference only.
