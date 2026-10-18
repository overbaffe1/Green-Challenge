# Runbook

## 0. Preconditions

* A GPU host with the NVIDIA driver and `nvidia-container-toolkit`.
* Access to `aij-green-challenge.cr.cloud.ru` (build time only; the image must run
  with no internet).
* The submitted image must be ≤ 50 GB, fit in 24 GB of VRAM on an RTX PRO 6000, and
  start `python /app/docker/entrypoint.py`.

## 1. Build the submission image

Pin the base image by digest, not by tag:

```bash
export BASE='aij-green-challenge.cr.cloud.ru/green-vla@sha256:<the digest from the task page>'
docker build -f docker/Dockerfile --build-arg BASE_IMAGE="$BASE" -t green-challenge:1.3 .
```

The Dockerfile copies only `greenchallenge/`, `docker/entrypoint.py` and `docs/`;
the checkpoint, Qwen cache, CUDA kernels and the `lerobot` fork stay exactly as the
organisers shipped them.

Check the size before pushing:

```bash
docker image inspect green-challenge:1.3 --format '{{.Size}}'   # bytes; must be < 50e9
```

## 2. Verify the image

```bash
# startup without a simulator
docker run --gpus all --rm --network none -p 8999:8999 \
  -e POLICY_PORT=8999 -e POLICY_RAW_PORT=8998 -e GC_SPEED=1.3 \
  green-challenge:1.3

# in another shell
curl -sf http://127.0.0.1:8999/healthz && echo OK
```

Expected log lines, in order:

```
greenchallenge 0.1.0
configuration: {... 'stack': 'official', 'speed': 1.3 ...}
starting raw policy: ... --feed-by-server --feed.row-dt 0.02564102564102564 ...
starting wire boundary: ...
ready: ws://0.0.0.0:8999
```

If the raw server exits, the container exits with code 1 and the traceback is on
stderr. There is deliberately no proxy-only healthy state.

Offline proof: `--network none` must still reach `/healthz`. Any attempt to reach
Hugging Face at boot fails in that mode; the image must not need it.

## 3. One benchmark run

```bash
cd ~/green_challenge
make up RUN_BENCHMARK=1 POLICY_PORT=8999 \
  RUN_PATH=assets/runs/green_challenge.json RESULTS_PATH=$PWD/results.json
python ../Green-Challenge/eval/report.py results.json --sweep
```

See `docs/EVALUATION.md` for the simulator environment (`PROPRIO_CMD=all`, etc.) and
the A/B protocol.

## 4. Rollback

The original entrypoint is preserved in the image as
`/app/docker/entrypoint.official.py`. To run the untouched baseline:

```bash
docker run --gpus all --rm -p 8999:8999 \
  --entrypoint python green-challenge:1.3 /app/docker/entrypoint.official.py
```

or set `GC_STACK=official` with `GC_SPEED=1.0`, which produces the reference feed.

Keep the previous image tag locally until the new submission has a recorded score.

## 5. Submission schedule

The rules allow one submission per calendar day from 2026-09-07 to 2026-10-20
inclusive, and the best score counts. With three days left there are three slots:

| Slot | Candidate | What it proves |
| --- | --- | --- |
| 1 (18 Oct) | `official` stack, `GC_SPEED=1.3` | The measured speed recipe runs in the contest harness without losing completion. |
| 2 (19 Oct) | Slot 1 ± one controlled change: adapter `v1`, or `boundary` if it won the A/B | A single isolated delta with a recorded A/B. |
| 3 (20 Oct) | Best measured configuration | No unmeasured changes; freeze the image digest and submit early in the day. |

A submission is a text file containing the full image reference. Do not put
credentials in it, and do not rebuild a tag that has already been submitted: the
registry digest is what gets evaluated.

## 6. Troubleshooting

| Symptom | Check |
| --- | --- |
| Container exits immediately | Child process traceback on stderr; usually a missing checkpoint or a CUDA/driver mismatch. |
| `/healthz` never answers | `GC_START_TIMEOUT` (default 600 s); the checkpoint load can take minutes on a cold cache. |
| Simulator connects but never gets a reply | `POLICY_PORT` must be the public port; the raw port is loopback by design. |
| Policy output looks frozen | The feed never engaged: check that the request `t` advances. A clock that does not advance freezes every plan on row 0. |
| Hands miss grasps after a checkpoint swap | Data config / `asset_id` mismatch, or `S0S1_SWAP_WRISTS` changed. |
| Score lower than baseline at 1.0× | The image is not serving the same checkpoint or the same data config; stop and compare handshake metadata. |
| `period is too short` at startup | `GC_PERIOD` and `GC_LEAD` conflict; restore the defaults. |
