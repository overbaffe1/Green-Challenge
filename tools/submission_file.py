#!/usr/bin/env python3
"""Create and validate the submission file the contest platform expects.

The platform accepts a plain text file with the image reference. The organisers
are explicit about the format, and getting it wrong is expensive:

* a tag together with a digest ("mychrepo/green-vla:v1@sha256:...") fails with
  ``INFRA_ERROR / rtx_status: INTERNAL_ERROR`` a couple of minutes into the queue;
* publishing the full registry URL publicly is treated as revealing the
  solution location and is grounds for disqualification;
* the accepted form is the digest alone::

      @sha256:<64 lowercase hex>

Usage::

    python tools/submission_file.py make --ref myregistry.cr.cloud.ru/green-vla:1.3
    python tools/submission_file.py make --ref '@sha256:317dc67f...' --out submission/submission.txt
    python tools/submission_file.py check                      # validate submission/submission.txt
    python tools/submission_file.py check --allow-placeholder   # used by CI before a digest exists

``make`` pulls the digest from the local Docker daemon when given a tagged
reference, so a tag that was re-pushed after the build cannot be submitted by
accident.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DIGEST_RE = re.compile(r"^@sha256:[0-9a-f]{64}$")
REF_WITH_DIGEST_RE = re.compile(r"^(?:[A-Za-z0-9._:\-/]+)?(@sha256:[0-9a-f]{64})$")
REF_WITH_TAG_RE = re.compile(r"^([A-Za-z0-9._\-]+(?::[0-9]+)?/[A-Za-z0-9._\-/]+):([A-Za-z0-9._\-]+)$")
BARE_HEX_RE = re.compile(r"^(?:sha256:)?([0-9a-f]{64})$")

PLACEHOLDER = "@sha256:" + "0" * 64
#: The organisers' published baseline. Kept as a last-resort fallback, so
#: overwriting it with a real build is always allowed.
BASELINE = "@sha256:317dc67f49de511e9fd0fb97fb27fefd3cd95125655c1410ac3956505edd6037"
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "submission" / "submission.txt"


def normalise(ref: str) -> str:
    """Turn any accepted reference into ``@sha256:<digest>``."""
    ref = ref.strip()
    if DIGEST_RE.match(ref):
        return ref
    if REF_WITH_DIGEST_RE.match(ref):
        # A full path with a digest is accepted by the platform's validator but
        # leaks the registry location; keep only the digest.
        return REF_WITH_DIGEST_RE.match(ref).group(1)  # type: ignore[union-attr]
    match = BARE_HEX_RE.match(ref)
    if match:
        return "@sha256:" + match.group(1)
    raise SystemExit(
        f"not a usable reference: {ref!r}\n"
        "Expected '@sha256:<64 hex>', '<raw hex>' or a tagged image such as "
        "'registry/green-vla:1.3' (the tag is resolved through docker)."
    )


def resolve_tag(ref: str) -> str:
    """Ask Docker for the repo digest of a tagged image."""
    try:
        output = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", ref],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except FileNotFoundError:
        raise SystemExit("docker is not installed; pass '@sha256:<digest>' directly")
    except subprocess.CalledProcessError:
        raise SystemExit(
            f"docker knows no image {ref!r}. Push it first (docker push {ref}) and "
            "then run this again, or pass the digest explicitly."
        )
    digests = json.loads(output)
    if not digests:
        raise SystemExit(
            f"{ref!r} has no RepoDigest: it was built locally but never pushed. "
            "A local image cannot be submitted; push it to the contest registry first."
        )
    return digests[0]


def check(path: Path, allow_placeholder: bool) -> int:
    if not path.is_file():
        print(f"FAIL: {path} does not exist", file=sys.stderr)
        return 1
    raw = path.read_text(encoding="utf-8")
    text = raw.strip()
    problems = []
    if raw != text + ("\n" if raw.endswith("\n") else ""):
        problems.append("leading/trailing whitespace")
    if "\n" in text:
        problems.append(f"more than one line ({text.count(chr(10)) + 1} lines)")
    if text != text.lower():
        problems.append("uppercase characters (digests are lowercase)")
    if not DIGEST_RE.match(text):
        if text == PLACEHOLDER:
            if allow_placeholder:
                print(f"OK (placeholder): {path}")
                return 0
            problems.append("still the placeholder digest")
        elif "@sha256:" in text and ":" in text.split("@")[0]:
            problems.append("contains a tag together with the digest — remove ':tag'")
        elif "@sha256:" in text:
            problems.append("contains a registry path — the platform needs the digest only")
        else:
            problems.append("not of the form '@sha256:<64 lowercase hex>'")
    if problems:
        print(f"FAIL: {path}", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(f"  content: {text[:120]!r}", file=sys.stderr)
        return 1
    print(f"OK: {path} -> {text}")
    return 0


def make(ref: str, out: Path, force: bool) -> int:
    if REF_WITH_TAG_RE.match(ref):
        ref = resolve_tag(ref)
    digest = normalise(ref)
    out = out.resolve()
    if out.exists() and not force:
        current = out.read_text(encoding="utf-8").strip()
        if current not in (PLACEHOLDER, BASELINE) and current != digest:
            raise SystemExit(
                f"{out} already holds a different digest:\n  {current}\n"
                "Submissions that already ran must stay reproducible; pass --force "
                "only for a digest that has never been submitted."
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(digest + "\n", encoding="utf-8")
    print(f"wrote {out}: {digest}")
    return check(out, allow_placeholder=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_make = sub.add_parser("make", help="write the submission file")
    p_make.add_argument("--ref", required=True, help="image reference or @sha256 digest")
    p_make.add_argument("--out", type=Path, default=DEFAULT_PATH)
    p_make.add_argument("--force", action="store_true")

    p_check = sub.add_parser("check", help="validate the submission file")
    p_check.add_argument("--path", type=Path, default=DEFAULT_PATH)
    p_check.add_argument("--allow-placeholder", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "make":
        return make(args.ref, args.out, args.force)
    return check(args.path, args.allow_placeholder)


if __name__ == "__main__":
    sys.exit(main())
