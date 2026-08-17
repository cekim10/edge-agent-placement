#!/usr/bin/env python3
"""Irreversible actions taken by a ScienceWorld agent, by stage placement.

Runs nothing: the harness already records every step it took, whether the
environment accepted it, and the score at the time. This reads existing run
directories.

The comparison AppWorld could not support is available here because the score is
graded. `irrev_per_point` is the one to read: irreversible actions divided by
the normalized score the episode actually reached. A tier that burns through
irreversible actions and ends up at the same score paid more for the same
result, and that ratio stays meaningful even when nothing is solved outright.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_agent.scienceworld_sideeffects import (  # noqa: E402
    episode_action_profile,
)


def load_run(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(run_dir.glob("*.jsonl")):
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if rows:
            runs[path.stem] = rows
    return runs


def summarise(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    profiles = [(row, episode_action_profile(row.get("steps", []))) for row in rows]
    scores = [float(row.get("normalized_score", 0.0) or 0.0) for row, _ in profiles]
    irreversible = [p["executed_irreversible"] for _, p in profiles]
    total_score = sum(scores)
    return {
        "label": label,
        "n": n,
        "avg_normalized_score": (sum(scores) / n) if n else 0.0,
        "success_rate": sum(1.0 for row, _ in profiles if row.get("success")) / n if n else 0.0,
        "avg_steps": sum(p["steps"] for _, p in profiles) / n if n else 0.0,
        "invalid_action_rate": (
            sum(p["invalid_actions"] for _, p in profiles)
            / max(1, sum(p["steps"] for _, p in profiles))
        ),
        "irrev_per_episode": (sum(irreversible) / n) if n else 0.0,
        "attempted_irrev_per_episode": (
            sum(p["attempted_irreversible"] for _, p in profiles) / n if n else 0.0
        ),
        # Irreversible actions spent per unit of score actually achieved. Guarded
        # against a zero-score run, where the ratio is undefined rather than 0.
        "irrev_per_point": (sum(irreversible) / total_score) if total_score > 0 else float("nan"),
        "episodes_with_irrev": (
            sum(1 for value in irreversible if value > 0) / n if n else 0.0
        ),
        "unclassified_per_episode": (
            sum(p["unclassified_actions"] for _, p in profiles) / n if n else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, nargs="*")
    args = parser.parse_args()

    run_dirs = list(args.run_dir)
    if not run_dirs:
        candidates = sorted((ROOT / "outputs").glob("scienceworld_observation1_*"))
        if not candidates:
            print("no scienceworld_observation1_* run directories found")
            return 1
        run_dirs = [candidates[-1]]

    runs: dict[str, list[dict[str, Any]]] = {}
    for directory in run_dirs:
        print(f"run: {directory}")
        for label, rows in load_run(directory).items():
            if label in runs:
                label = f"{label}@{directory.name}"
            runs[label] = rows
    if not runs:
        print("no .jsonl records found")
        return 1

    rows = [summarise(label, records) for label, records in runs.items()]
    rows.sort(key=lambda row: row["label"] != "all_cloud")

    header = (
        f"{'placement':<26}{'n':>4}{'score':>8}{'success':>9}{'steps':>7}"
        f"{'invalid':>9}{'irrev/ep':>10}{'irrev/pt':>10}{'unclass':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        ratio = row["irrev_per_point"]
        ratio_text = "  n/a" if ratio != ratio else f"{ratio:>10.2f}"
        print(
            f"{row['label']:<26}{row['n']:>4}{row['avg_normalized_score']:>8.3f}"
            f"{row['success_rate']:>9.3f}{row['avg_steps']:>7.1f}"
            f"{row['invalid_action_rate']:>9.3f}{row['irrev_per_episode']:>10.2f}"
            f"{ratio_text}{row['unclassified_per_episode']:>9.2f}"
        )

    # Which verbs actually produced the count. A single verb dominating means the
    # metric is measuring that verb, not "destruction" -- and `focus on` is a
    # judgement call, so it has to be visible rather than folded into a total.
    print("\nirreversible actions taken, by placement:")
    for label, records in runs.items():
        verbs: Counter[str] = Counter()
        for record in records:
            for action, count in episode_action_profile(
                record.get("steps", [])
            )["irreversible_multiset"].items():
                verbs[action.split()[0] if action.split() else action] += count
        total = sum(verbs.values())
        detail = ", ".join(f"{verb}={count}" for verb, count in verbs.most_common()) or "(none)"
        print(f"  {label:<26} total={total:<4} {detail}")

    unknown: Counter[str] = Counter()
    for records in runs.values():
        for record in records:
            unknown.update(episode_action_profile(record.get("steps", []))["unclassified_multiset"])
    if unknown:
        print("\nunclassified actions (extend VERB_CLASSES, then re-run):")
        for action, count in unknown.most_common(20):
            print(f"  {action}  x{count}")
    else:
        print("\nunclassified actions: none")

    out = run_dirs[-1] / "irreversible_actions.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
