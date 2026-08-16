#!/usr/bin/env python3
"""Inspect what actually went wrong in a micro_obs1 run.

Aggregate rates say a cell failed; they do not say whether the model reasoned
badly, the harness truncated the output, or the task spec and the ground truth
disagree. This prints the raw evidence for each of those.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(run_dir.glob("*.jsonl")):
        runs[path.stem] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
    return runs


def _ops_key(ops: list[dict[str, str]] | None) -> set[tuple[str, str, str, str]]:
    if not ops:
        return set()
    return {
        (op["op"], op["user_id"], op["resource"], op.get("role", "") or "")
        for op in ops
    }


def _fmt(ops: set[tuple[str, str, str, str]]) -> str:
    return ", ".join(sorted("/".join(op) for op in ops)) or "(none)"


def _diff_label(expected: set[Any], predicted: set[Any]) -> str:
    """Name the failure mode so patterns are countable rather than anecdotal."""
    if not predicted:
        return "empty_plan"
    if expected == predicted:
        return "match"
    exp_by = {op[0] for op in expected}
    pred_by = {op[0] for op in predicted}
    if exp_by != pred_by:
        return "wrong_op_type"
    if {op[1] for op in expected} != {op[1] for op in predicted}:
        return "wrong_user_id"
    if {op[2] for op in expected} != {op[2] for op in predicted}:
        return "wrong_resource"
    if {op[3] for op in expected} != {op[3] for op in predicted}:
        return "wrong_role"
    if len(expected) != len(predicted):
        return "wrong_op_count"
    return "other"


def report_schema(records: list[dict[str, Any]], label: str, limit: int) -> None:
    broken = [r for r in records if r["score"].get("schema_violation") or r["score"].get("no_output")]
    if not broken:
        return
    print(f"\n### {label}: {len(broken)}/{len(records)} schema violations")
    for record in broken[:limit]:
        stages = (record.get("workflow") or {}).get("stages", [])
        failed = next((s for s in stages if s.get("error")), None)
        print(f"  instance={record['instance']['instance_id']} error={record['score'].get('schema_error')}")
        if failed:
            raw = failed.get("output", "")
            print(f"    stage={failed['stage']} tier={failed['tier']} len={len(raw)}")
            print(f"    raw={raw[:400]!r}")


def report_plan(records: list[dict[str, Any]], label: str, limit: int) -> None:
    modes: Counter[str] = Counter()
    by_category: Counter[tuple[str, str]] = Counter()
    shown = 0
    lines: list[str] = []
    for record in records:
        if record["score"].get("schema_violation") or record["score"].get("no_output"):
            continue
        instance = record["instance"]
        expected = _ops_key(instance["expected_ops"])
        predicted = _ops_key((record.get("workflow") or {}).get("predicted_ops"))
        mode = _diff_label(expected, predicted)
        category = instance["classification"]["category"]
        modes[mode] += 1
        by_category[(category, mode)] += 1
        if mode != "match" and shown < limit:
            shown += 1
            predicted_classification = (record.get("workflow") or {}).get(
                "predicted_classification"
            )
            lines.append(
                f"  {instance['instance_id']} [{category}] mode={mode}\n"
                f"    request={instance['request'][:140]}\n"
                f"    classify_ok={record['score']['classification_correct']} "
                f"predicted_classification={json.dumps(predicted_classification, sort_keys=True)}\n"
                f"    expected  = {_fmt(expected)}\n"
                f"    predicted = {_fmt(predicted)}"
            )
    print(f"\n### {label}: plan failure modes")
    for mode, count in modes.most_common():
        print(f"  {mode:16s} {count}")
    print("  by category:")
    for (category, mode), count in sorted(by_category.items()):
        if mode != "match":
            print(f"    {category:20s} {mode:16s} {count}")
    for line in lines:
        print(line)


def report_cells(records: list[dict[str, Any]], label: str) -> None:
    """Failure modes broken out by difficulty cell.

    The aggregate cannot explain why a cell scores the way it does. If one cell
    fails through a different mechanism than another, the difficulty axis is not
    a single ordered thing and the grid should not be read as a gradient.
    """
    cells: dict[tuple[str, str], Counter[str]] = {}
    totals: Counter[tuple[str, str]] = Counter()
    destructive: dict[tuple[str, str], list[int]] = {}
    for record in records:
        instance = record["instance"]
        key = (instance["a_level"], instance["b_level"])
        totals[key] += 1
        destructive.setdefault(key, []).append(
            record["score"].get("unrequested_destructive_ops", 0)
        )
        if record["score"].get("schema_violation") or record["score"].get("no_output"):
            cells.setdefault(key, Counter())["schema_violation"] += 1
            continue
        expected = _ops_key(instance["expected_ops"])
        predicted = _ops_key((record.get("workflow") or {}).get("predicted_ops"))
        mode = _diff_label(expected, predicted)
        category = instance["classification"]["category"]
        tag = "match" if mode == "match" else f"{category}:{mode}"
        cells.setdefault(key, Counter())[tag] += 1

    print(f"\n### {label}: failure modes by cell")
    for key in sorted(cells):
        n = totals[key]
        modes = cells[key]
        matched = modes.get("match", 0)
        extra = destructive.get(key, [])
        mean_extra = sum(extra) / len(extra) if extra else 0.0
        print(
            f"  A {key[0]:<5} B {key[1]:<5} n={n:<4} exact={matched}/{n} "
            f"incorrect_irreversible_ops={mean_extra:.2f}"
        )
        for tag, count in modes.most_common():
            if tag == "match":
                continue
            print(f"      {tag:<40s} {count}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--limit", type=int, default=6, help="examples printed per label")
    parser.add_argument("--only-label", help="restrict to one placement label")
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        candidates = sorted((ROOT / "outputs").glob("micro_obs1_*"))
        if not candidates:
            print("no micro_obs1_* run directories found")
            return 1
        run_dir = candidates[-1]
    print(f"run: {run_dir}")

    runs = _load(run_dir)
    if not runs:
        print("no .jsonl records found")
        return 1
    for label, records in runs.items():
        if args.only_label and label != args.only_label:
            continue
        report_schema(records, label, args.limit)
        report_cells(records, label)
        report_plan(records, label, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
