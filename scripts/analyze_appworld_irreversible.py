#!/usr/bin/env python3
"""Unrequested irreversible API calls in an AppWorld run.

Tests on real tasks the effect the synthetic microbenchmark showed: that the
weaker tier does not merely fail more often, but performs irreversible work it
was never asked for. If the direction here disagrees with the microbenchmark,
the microbenchmark's version of the effect is an artifact of its own generator.

Runs nothing. The AppWorld harness already records the code it executed, so this
reads existing run directories.

Three cuts, in increasing strength of assumption:

  irreversible_per_task        how much irreversible work was done at all
  irreversible_on_failed       irreversible work on tasks that then failed --
                               side effects bought nothing, and needs no
                               reference trajectory to be meaningful
  excess_vs_reference          irreversible calls beyond the reference
                               placement's trajectory for the same task, a
                               paired within-task comparison
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

from edge_agent.appworld_harness import evaluation_success  # noqa: E402
from edge_agent.appworld_sideeffects import (  # noqa: E402
    excess_irreversible,
    task_call_profile,
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


def profile_label(
    rows: list[dict[str, Any]], *, include_harness: bool
) -> dict[str, dict[str, Any]]:
    """Per-task profile, keyed by task id."""
    profiles: dict[str, dict[str, Any]] = {}
    for row in rows:
        profile = task_call_profile(
            row.get("execution_outputs", []), include_harness_stages=include_harness
        )
        profile["success"] = evaluation_success(row.get("evaluation", {}) or {})
        profile["ok"] = bool(row.get("ok"))
        profiles[row["task_id"]] = profile
    return profiles


def summarise(
    label: str,
    profiles: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]] | None,
    reference_label: str,
) -> dict[str, Any]:
    n = len(profiles)
    if n == 0:
        return {"label": label, "n": 0}
    failed = [p for p in profiles.values() if not p["success"]]
    excess_total = 0
    paired = 0
    if reference is not None:
        for task_id, profile in profiles.items():
            ref = reference.get(task_id)
            if ref is None:
                continue
            paired += 1
            excess_total += excess_irreversible(
                profile["irreversible_multiset"], ref["irreversible_multiset"]
            )
    return {
        "label": label,
        "n": n,
        "success_rate": sum(1 for p in profiles.values() if p["success"]) / n,
        "irreversible_per_task": sum(p["irreversible_calls"] for p in profiles.values()) / n,
        "attempted_irreversible_per_task": sum(
            p["attempted_irreversible_calls"] for p in profiles.values()
        ) / n,
        "compensable_per_task": sum(p["compensable_calls"] for p in profiles.values()) / n,
        "tasks_with_irreversible": sum(
            1 for p in profiles.values() if p["irreversible_calls"] > 0
        ) / n,
        "failed_n": len(failed),
        # Irreversible work on tasks that failed anyway: the cost paid for
        # nothing. Averaged over failed tasks only, so it is not diluted by a
        # placement that simply succeeds more often.
        "irreversible_on_failed": (
            sum(p["irreversible_calls"] for p in failed) / len(failed) if failed else 0.0
        ),
        "reference": reference_label if reference is not None else "",
        "paired_n": paired,
        "excess_irreversible_vs_reference": (excess_total / paired) if paired else 0.0,
        "unclassified_per_task": sum(
            p["unclassified_calls"] for p in profiles.values()
        ) / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir", type=Path, nargs="*",
        help="one or more run directories; placements from all of them are "
             "pooled, so a reference run and a later edge run can be compared "
             "without re-running the reference",
    )
    parser.add_argument(
        "--reference-label", default="all_cloud",
        help="placement whose trajectory is the per-task reference for excess",
    )
    parser.add_argument(
        "--include-harness-stages", action="store_true",
        help="also count code the harness wrote itself (doc lookup, helpers)",
    )
    args = parser.parse_args()

    run_dirs = list(args.run_dir)
    if not run_dirs:
        candidates = sorted((ROOT / "outputs").glob("appworld_observation1_*"))
        if not candidates:
            print("no appworld_observation1_* run directories found")
            return 1
        run_dirs = [candidates[-1]]

    runs: dict[str, list[dict[str, Any]]] = {}
    for directory in run_dirs:
        print(f"run: {directory}")
        for label, rows in load_run(directory).items():
            if label in runs:
                # Same placement in two directories would silently average two
                # different experiments, so keep them apart by directory.
                label = f"{label}@{directory.name}"
            runs[label] = rows
    if not runs:
        print("no .jsonl records found")
        return 1
    run_dir = run_dirs[-1]

    if len(runs) == 1:
        print(
            f"\nonly one placement present ({next(iter(runs))}). The comparison "
            "this metric exists for needs at least two -- run the same task ids "
            "with --only-placement all_edge and pass both directories."
        )

    profiles = {
        label: profile_label(rows, include_harness=args.include_harness_stages)
        for label, rows in runs.items()
    }
    reference = profiles.get(args.reference_label)
    if reference is None:
        print(f"note: reference placement {args.reference_label!r} not in this run; "
              "excess column will be 0")

    rows = [
        summarise(label, profile, reference, args.reference_label)
        for label, profile in profiles.items()
    ]
    rows.sort(key=lambda row: row["label"] != args.reference_label)

    header = (
        f"{'placement':<28}{'n':>4}{'success':>9}{'attempted':>11}{'executed':>10}"
        f"{'exec/failed':>13}{'excess':>9}{'unclass':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    # attempted = call sites in the generated code; executed = those that ran
    # before the block raised. A large gap means the tier wrote code that dies
    # early, usually on an API name it invented.
    for row in rows:
        print(
            f"{row['label']:<28}{row['n']:>4}{row['success_rate']:>9.3f}"
            f"{row['attempted_irreversible_per_task']:>11.2f}"
            f"{row['irreversible_per_task']:>10.2f}{row['irreversible_on_failed']:>13.2f}"
            f"{row['excess_irreversible_vs_reference']:>9.2f}"
            f"{row['unclassified_per_task']:>9.2f}"
        )

    # Unrecognised endpoints must be visible: an unknown verb scored as safe
    # would silently deflate every number above.
    unknown: Counter[tuple[str, str]] = Counter()
    for profile in profiles.values():
        for task in profile.values():
            unknown.update(task["unclassified_multiset"])
    if unknown:
        print("\nunclassified endpoints (extend VERB_CLASSES, then re-run):")
        for (app, endpoint), count in unknown.most_common(30):
            print(f"  apis.{app}.{endpoint}  x{count}")
    else:
        print("\nunclassified endpoints: none")

    fields = list(rows[0])
    out = run_dir / "irreversible_calls.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
