#!/usr/bin/env python3
"""Inspect a LeRobot-format dataset before spending GPU time on it.

    python training/validate_dataset.py /path/to/GreenChallengeData

Prints the feature schema, camera names, episode/task counts and whether a
``subtask`` field exists.  Exits non-zero if the state or action dimensionality
does not match the benchmark contract, or if the three cameras are missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXPECTED_CAMERAS = {
    "head": ("cam_head", "top_head", "observation.images.cam_head"),
    "left_wrist": ("cam_left_wrist", "hand_left", "observation.images.cam_left_wrist"),
    "right_wrist": ("cam_right_wrist", "hand_right", "observation.images.cam_right_wrist"),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def shape_of(feature: dict[str, Any]) -> tuple[str, Any]:
    return feature.get("dtype", "?"), feature.get("shape", "?")


def find_feature(features: dict[str, Any], *names: str) -> tuple[str | None, dict | None]:
    for name in names:
        if name in features:
            return name, features[name]
    return None, None


def parquet_columns(data_dir: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return []
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        return []
    try:
        return list(pq.ParquetFile(files[0]).schema_arrow.names)
    except Exception:
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    root = args.dataset

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        print(f"not a LeRobot dataset: {info_path} is missing", file=sys.stderr)
        return 2
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    if not isinstance(features, dict) or not features:
        print("meta/info.json has no features schema", file=sys.stderr)
        return 2

    print(f"dataset:      {root}")
    print(f"codebase:     {info.get('codebase_version', '?')}")
    print(f"fps:          {info.get('fps', '?')}")
    print(f"episodes:     {info.get('total_episodes', '?')}")
    print(f"frames:       {info.get('total_frames', '?')}")
    print(f"tasks:        {info.get('total_tasks', '?')}")

    problems: list[str] = []

    print("\nfeatures:")
    for name in sorted(features):
        dtype, shape = shape_of(features[name])
        print(f"  {name:<42} {str(dtype):<10} {shape}")

    state_name, state = find_feature(features, "observation.state")
    action_name, action = find_feature(features, "action")
    for label, found, expected_dim in (
        ("observation.state", state, 51),
        ("action", action, 51),
    ):
        if found is None:
            problems.append(f"{label} is missing")
            continue
        dim = found.get("shape", [None])[-1]
        if dim not in (expected_dim, 52):
            problems.append(f"{label} has last dimension {dim}, expected {expected_dim} (or 52 with episode_progress)")
        elif label == "action" and dim == 52:
            print("\nnote: action has 52 columns; the last one is episode_progress and is not a motor command")

    print("\ncameras:")
    for role, candidates in EXPECTED_CAMERAS.items():
        name, feature = find_feature(features, *candidates)
        if feature is None:
            problems.append(f"camera {role} not found (looked for {candidates})")
            print(f"  {role:<12} MISSING")
        else:
            print(f"  {role:<12} {name}  {feature.get('shape')}")

    tasks = load_jsonl(root / "meta" / "tasks.jsonl")
    episodes = load_jsonl(root / "meta" / "episodes.jsonl")
    print(f"\ntasks.jsonl:    {len(tasks)} rows")
    if tasks[:1]:
        print(f"  example: {json.dumps(tasks[0], ensure_ascii=False)[:200]}")
    print(f"episodes.jsonl: {len(episodes)} rows")
    if episodes[:1]:
        print(f"  keys: {sorted(episodes[0])}")

    subtask_found = any("subtask" in row for row in episodes) or any(
        "subtask" in row for row in tasks
    )
    columns = parquet_columns(root / "data")
    if columns:
        print(f"\nparquet columns: {columns}")
        if any("subtask" in column for column in columns):
            subtask_found = True
    print(f"\nsubtask field: {'yes' if subtask_found else 'no'}")
    if not subtask_found:
        print("  -> either add a subtask column, or train with the full task")
        print("     instruction and set GC_PROMPT_FROM_SUBTASK=0 at inference")
        print("     (see docs/TRAINING.md, 'Language').")

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\ndataset layout matches the benchmark contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
