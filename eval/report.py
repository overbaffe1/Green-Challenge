#!/usr/bin/env python3
"""Read a GreenSim ``results.json`` and explain the score.

The contest score is a raw sum, not an average:

    Speed_{i,g} = clamp(1 - t_{i,g} / T_max, 0, 1)     for completed subtasks
    score       = sum over episodes i and subtasks g

so a run's score is dominated by *how many* subtasks finish and *how fast*.
This tool prints both, plus a "what if" table that scales every measured
subtask duration by a playback factor and recomputes the score.  The table is
an upper bound: it ignores the effect of speed on success, which is exactly the
effect the A/B benchmark has to measure.

Usage:
    python eval/report.py benchmark_logs/green_challenge_*/results.json
    python eval/report.py run_a/results.json run_b/results.json --compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def scenario_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, metrics in (results.get("subtasks") or {}).items():
        rows.append(
            {
                "scenario": name,
                "score": float(metrics.get("score", 0.0)),
                "completed": int(metrics.get("subtask_completed", 0)),
                "total": int(metrics.get("subtask_total", 0)),
                "success_rate": metrics.get("success_rate", metrics.get("success", "?")),
                "speed": metrics.get("speed", "?"),
                "goal_score": metrics.get("goals_score", "?"),
            }
        )
    return rows


def summarise(path: str | Path) -> dict[str, Any]:
    results = load(path)
    rows = scenario_rows(results)
    score = float(results.get("score", sum(r["score"] for r in rows)))
    completed = int(results.get("subtask_completed", sum(r["completed"] for r in rows)))
    total = int(results.get("subtask_total", sum(r["total"] for r in rows)))
    return {
        "path": str(path),
        "run_name": results.get("run_name", Path(path).parent.name),
        "score": score,
        "completed": completed,
        "total": total,
        "rows": rows,
        "raw": results,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"run:    {summary['run_name']}")
    print(f"file:   {summary['path']}")
    print(f"score:  {summary['score']:.4f}")
    if summary["total"]:
        print(
            f"subtasks: {summary['completed']}/{summary['total']} completed "
            f"({100.0 * summary['completed'] / summary['total']:.1f}%)"
        )
    print()
    header = f"{'scenario':<24}{'score':>10}{'subtasks':>12}{'success':>12}{'ep speed':>12}"
    print(header)
    print("-" * len(header))
    for row in summary["rows"]:
        print(
            f"{row['scenario']:<24}{row['score']:>10.4f}"
            f"{row['completed']:>7}/{row['total']:<5}{str(row['success_rate']):>12}"
            f"{str(row['speed']):>12}"
        )


def speed_table(results: dict[str, Any], factors: list[float]) -> list[tuple[float, float, int]]:
    """Recompute the score if every completed subtask were ``factor`` faster.

    ``results.json`` does not include per-subtask durations, so this uses
    each completed subtask's mean speed: ``t = (1-Speed)*T_max``, a playback
    factor ``k`` gives ``t_new = t/k`` and
    ``Speed_new = clamp(1 - (1-Speed)/k)``.
    """
    # The aggregator stores subtask speeds only in the score sum; reconstruct
    # the mean speed per scenario and treat each subtask as the average.  This
    # is exact for the *mean* and therefore for the reported totals.
    per_scenario = []
    for name, metrics in (results.get("subtasks") or {}).items():
        completed = int(metrics.get("subtask_completed", 0))
        score = float(metrics.get("score", 0.0))
        if completed:
            per_scenario.append((name, completed, score / completed))
    table = []
    for factor in factors:
        total = 0.0
        count = 0
        for _name, completed, mean_speed in per_scenario:
            new_speed = max(0.0, min(1.0, 1.0 - (1.0 - mean_speed) / factor))
            total += new_speed * completed
            count += completed
        table.append((factor, total, count))
    return table


def compare(a: dict[str, Any], b: dict[str, Any]) -> None:
    print(f"A: {a['run_name']}  score {a['score']:.4f}")
    print(f"B: {b['run_name']}  score {b['score']:.4f}")
    delta = b["score"] - a["score"]
    print(f"delta: {delta:+.4f} ({100.0 * delta / a['score'] if a['score'] else float('nan'):+.1f}%)")
    print()
    rows_a = {row["scenario"]: row for row in a["rows"]}
    rows_b = {row["scenario"]: row for row in b["rows"]}
    print(f"{'scenario':<24}{'A':>10}{'B':>10}{'delta':>10}{'A done':>10}{'B done':>10}")
    for name in sorted(set(rows_a) | set(rows_b)):
        ra, rb = rows_a.get(name), rows_b.get(name)
        print(
            f"{name:<24}{ra['score'] if ra else float('nan'):>10.4f}"
            f"{rb['score'] if rb else float('nan'):>10.4f}"
            f"{(rb['score'] if rb else 0) - (ra['score'] if ra else 0):>+10.4f}"
            f"{ra['completed'] if ra else 0:>10}{rb['completed'] if rb else 0:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="path(s) to results.json")
    parser.add_argument("--compare", action="store_true", help="compare the first two files")
    parser.add_argument("--sweep", action="store_true", help="print the playback-factor table")
    args = parser.parse_args()

    summaries = [summarise(path) for path in args.results]
    for summary in summaries:
        print_summary(summary)
        print()

    if args.sweep and summaries:
        print(f"{'playback':>10}{'score':>12}{'subtasks':>12}")
        for factor, score, count in speed_table(
            summaries[0]["raw"], [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]
        ):
            print(f"{factor:>10.1f}{score:>12.4f}{count:>12}")

    if args.compare and len(summaries) >= 2:
        print()
        compare(summaries[0], summaries[1])


if __name__ == "__main__":
    main()
