#!/usr/bin/env bash
# Reassemble the artefact that .github/workflows/fetch-artifact.yml stored on
# the artifacts branch, verify it, and optionally list its contents.
#
#   ./tools/fetch_parts.sh                  # -> ./submission.zip
#   ./tools/fetch_parts.sh /tmp/big.zip     # explicit destination
#   LIST=1 ./tools/fetch_parts.sh           # also print the zip listing
#
# Environment:
#   REPO    owner/repo      (default overbaffe1/Green-Challenge)
#   BRANCH  chunks branch   (default artifacts/submission-zip)
#   LIST    set to 1 to run `unzip -l` afterwards
set -euo pipefail

REPO="${REPO:-overbaffe1/Green-Challenge}"
BRANCH="${BRANCH:-artifacts/submission-zip}"
DEST="${1:-./submission.zip}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "cloning $REPO@$BRANCH (chunks only) into a temporary directory"
git clone --quiet --depth 1 --branch "$BRANCH" "https://github.com/$REPO.git" "$TMP/parts"

shopt -s nullglob
parts=("$TMP/parts"/submission.zip.part.*)
if [ ${#parts[@]} -eq 0 ]; then
  echo "no parts found on $BRANCH" >&2
  exit 1
fi
echo "reassembling ${#parts[@]} parts into $DEST"
cat "${parts[@]}" > "$DEST"

expected="$(sed -n 's/^sha256:[[:space:]]*//p' "$TMP/parts/MANIFEST.txt" | head -1)"
actual="$(sha256sum "$DEST" | awk '{print $1}')"
echo "expected sha256: ${expected:-<unknown>}"
echo "actual   sha256: $actual"
if [ -n "$expected" ] && [ "$expected" != "$actual" ]; then
  echo "CHECKSUM MISMATCH — the artefact is corrupted" >&2
  exit 1
fi
echo "size: $(stat -c%s "$DEST") bytes"

if [ "${LIST:-0}" = "1" ]; then
  unzip -l "$DEST" | head -60 || true
fi

echo "ok: $DEST"
