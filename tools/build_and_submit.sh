#!/usr/bin/env bash
# Build the submission image, push it, and write submission/submission.txt
# with the resulting digest.
#
#   REGISTRY=cr.cloud.ru/<your-namespace> ./tools/build_and_submit.sh
#
# Environment:
#   REGISTRY      registry path without tag (required), e.g. cr.cloud.ru/team
#   TAG           tag (default: 1.3)
#   BASE_IMAGE    contest base image; defaults to the published baseline digest
#   GC_SPEED      playback rate baked into the image (default 1.3)
#   SKIP_PUSH     set to 1 to stop after the local build and the smoke test
#   DRY_RUN       set to 1 to print the commands without running them
#
# The script refuses to write a submission file until the registry actually
# serves the manifest: submitting during an unfinished push is one of the
# documented causes of "[ERROR] No such file or directory: /opt/enroot/...".
set -euo pipefail

cd "$(dirname "$0")/.."

REGISTRY="${REGISTRY:-}"
TAG="${TAG:-1.3}"
BASE_IMAGE="${BASE_IMAGE:-aij-green-challenge.cr.cloud.ru/green-vla@sha256:317dc67f49de511e9fd0fb97fb27fefd3cd95125655c1410ac3956505edd6037}"
GC_SPEED="${GC_SPEED:-1.3}"
IMAGE="${REGISTRY}:${TAG}"

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

run() {
  if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '  $ %s\n' "$*"
  else
    "$@"
  fi
}

[ -n "$REGISTRY" ] || die "set REGISTRY, e.g. REGISTRY=cr.cloud.ru/my-team ./tools/build_and_submit.sh"
IMAGE="${REGISTRY}:${TAG}"

# A dry run must work on a machine without Docker (CI, laptops) and still
# print the commands that would run.
if [ "${DRY_RUN:-0}" = "1" ]; then
  run() { printf '  $ %s\n' "$*"; }
  say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
  say "dry run"
  run docker build -f docker/Dockerfile --build-arg "BASE_IMAGE=$BASE_IMAGE" -t "$IMAGE" .
  run docker push "$IMAGE"
  run docker buildx imagetools inspect "$IMAGE"
  echo "  (submission/submission.txt would then be written with the pushed digest)"
  exit 0
fi

say "pre-flight"
command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "the docker daemon is not reachable (are you in the docker group?)"
if ! docker info --format '{{json .Runtimes}}' | grep -q nvidia; then
  printf 'warning: no nvidia runtime registered; the image will build but cannot run the model here\n'
fi
df -h . | awk 'NR==2 {printf "free disk: %s\n", $4}'

say "build $IMAGE"
run docker build \
  -f docker/Dockerfile \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --label "org.opencontainers.image.description=Green Challenge 2026 execution layer, GC_SPEED=$GC_SPEED" \
  -t "$IMAGE" .

say "smoke test (no network, no GPU needed for the import and config path)"
run docker run --rm --network none "$IMAGE" python - <<'PY'
import greenchallenge
from greenchallenge.config import RuntimeConfig
runtime = RuntimeConfig.from_env()
print("version:", greenchallenge.__version__)
print("stack:", runtime.stack, "speed:", runtime.feed.speed,
      "row_step:", round(runtime.feed.row_step(), 6))
assert abs(runtime.feed.row_step() - 1 / 39) < 1e-9
print("config ok")
PY

SIZE=$(docker image inspect --format '{{.Size}}' "$IMAGE")
printf 'image size: %s bytes (%s GB)\n' "$SIZE" "$(awk -v s="$SIZE" 'BEGIN {printf "%.1f", s/1e9}')"
if [ "$SIZE" -gt 50000000000 ]; then
  die "image exceeds the 50 GB contest limit"
fi

if [ "${SKIP_PUSH:-0}" = "1" ]; then
  say "SKIP_PUSH=1: stopping before the push"
  exit 0
fi

say "push"
run docker push "$IMAGE"

say "wait for the registry to serve the manifest"
# A tag is visible only after every layer has landed; poll instead of trusting
# `docker push`'s final line, which races with registry-side indexing.
for attempt in $(seq 1 60); do
  if docker buildx imagetools inspect "$IMAGE" >/dev/null 2>&1; then
    break
  fi
  if docker manifest inspect "$IMAGE" >/dev/null 2>&1; then
    break
  fi
  [ "$attempt" -eq 60 ] && die "the registry still does not serve $IMAGE after 10 minutes"
  sleep 10
done
echo "registry serves $IMAGE"

DIGEST=$(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE" | sed 's/.*@//')
[ -n "$DIGEST" ] || die "no RepoDigest: was the push authenticated?"

say "write submission/submission.txt"
python3 tools/submission_file.py make --ref "$DIGEST" --out submission/submission.txt --force
python3 tools/submission_file.py check

cat <<EOF

$(printf '\033[1m== done ==\033[0m')

image   : $IMAGE
digest  : $DIGEST
file    : submission/submission.txt

Upload submission/submission.txt to the contest page. It contains one line:
  $DIGEST

Keep $IMAGE in the registry until the contest ends: the organisers may
recompute submissions from the same digest.
EOF
