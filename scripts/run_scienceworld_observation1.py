#!/usr/bin/env python3
"""Run Observation 1 on ScienceWorld."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_agent.client import build_client  # noqa: E402
from edge_agent.scienceworld_harness import (  # noqa: E402
    SCIENCEWORLD_STAGES,
    ScienceWorldMockClient,
    ScienceWorldTaskSpec,
    list_scienceworld_tasks,
    load_scienceworld_task_specs,
    mock_scienceworld_result,
    placement_for_edge_scienceworld_stage,
    placement_name,
    run_scienceworld_workflow,
)


def _parse_int_list(value: str) -> list[int]:
    items: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start, end = int(start_s), int(end_s)
            items.extend(range(start, end + 1))
        else:
            items.append(int(chunk))
    return items


def _summarize(label: str, placement: tuple[str, ...], rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage_latency = {}
    for stage in SCIENCEWORLD_STAGES:
        latencies = [
            stage_result["latency_s"]
            for row in rows
            for stage_result in row.get("stages", [])
            if stage_result["stage"] == stage
        ]
        stage_latency[stage] = mean(latencies) if latencies else 0.0
    steps = [row.get("steps_taken", 0) for row in rows]
    return {
        "label": label,
        "placement": list(placement),
        "n": len(rows),
        "ok_n": sum(1 for row in rows if row["ok"]),
        "error_n": sum(1 for row in rows if not row["ok"]),
        "success_rate": mean(1.0 if row.get("success") else 0.0 for row in rows) if rows else 0.0,
        "avg_score": mean(float(row.get("final_score", 0.0)) for row in rows) if rows else 0.0,
        "avg_normalized_score": mean(float(row.get("normalized_score", 0.0)) for row in rows) if rows else 0.0,
        "avg_steps": mean(float(step) for step in steps) if steps else 0.0,
        "invalid_action_rate": (
            sum(float(row.get("invalid_action_count", 0)) for row in rows)
            / max(1.0, sum(float(row.get("steps_taken", 0)) for row in rows))
        ),
        "mean_stage_latency_s": stage_latency,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", action="append", default=None, help="ScienceWorld task name; can be repeated")
    parser.add_argument("--task-index", action="append", type=int, default=None, help="Index into env.getTaskNames(); can be repeated")
    parser.add_argument("--variations", default="0-2", help="Comma/range list, e.g. 0-4 or 0,3,5")
    parser.add_argument("--simplification", default="easy")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--state-max-tokens", type=int, default=120)
    parser.add_argument("--plan-max-tokens", type=int, default=120)
    parser.add_argument("--action-max-tokens", type=int, default=48)
    parser.add_argument("--verify-max-tokens", type=int, default=64)
    parser.add_argument("--dump-prompts", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--only-placement",
        choices=["all_cloud", "all_edge", *SCIENCEWORLD_STAGES],
        default=None,
        help="Run only all_cloud, all_edge, or one edge-stage variant",
    )
    args = parser.parse_args()

    if args.list_tasks:
        for index, name in enumerate(list_scienceworld_tasks()):
            print(f"{index}\t{name}")
        return 0

    variations = _parse_int_list(args.variations)
    if args.mock:
        task_specs = [
            ScienceWorldTaskSpec(task_name="mock-task", variation=index, simplification=args.simplification)
            for index in variations
        ]
        task_specs = task_specs[: args.limit] if args.limit else task_specs
        client = ScienceWorldMockClient()
    else:
        task_specs = load_scienceworld_task_specs(
            task_names=args.task_name,
            task_indices=args.task_index,
            variations=variations,
            simplification=args.simplification,
            limit=args.limit,
        )
        client = build_client(
            mock=False,
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
            max_tokens_by_stage={
                "state_abstraction": args.state_max_tokens,
                "subgoal_planning": args.plan_max_tokens,
                "action_selection": args.action_max_tokens,
                "progress_verification": args.verify_max_tokens,
            },
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"scienceworld_observation1_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = run_dir / "prompts" if args.dump_prompts else None

    variants = [args.only_placement] if args.only_placement else ["all_cloud", *SCIENCEWORLD_STAGES, "all_edge"]
    metrics = []
    print(
        f"Loaded {len(task_specs)} ScienceWorld episodes; "
        f"tasks={sorted(set(spec.task_name for spec in task_specs))} "
        f"variations={[spec.variation for spec in task_specs]}",
        flush=True,
    )
    for edge_stage in variants:
        placement = placement_for_edge_scienceworld_stage(edge_stage)
        label = placement_name(placement)
        records_path = run_dir / f"{label}.jsonl"
        rows = []
        with records_path.open("w", encoding="utf-8") as handle:
            for index, task_spec in enumerate(task_specs, start=1):
                print(f"[{label}] episode {index}/{len(task_specs)} {task_spec.task_id}", flush=True)
                if args.mock:
                    result = mock_scienceworld_result(task_spec, placement)
                else:
                    result = run_scienceworld_workflow(
                        client=client,
                        task_spec=task_spec,
                        placement=placement,
                        max_steps=args.max_steps,
                        dump_prompt_dir=prompt_dir / label if prompt_dir is not None else None,
                    )
                row = result.to_dict()
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                error_suffix = f" error={row['error']}" if row.get("error") else ""
                print(
                    f"[{label}] episode {index}/{len(task_specs)} ok={row['ok']} "
                    f"success={row['success']} score={row['final_score']:.1f} "
                    f"steps={row['steps_taken']} invalid={row['invalid_action_count']}{error_suffix}",
                    flush=True,
                )
        summary = _summarize(label, placement, rows)
        metrics.append(summary)
        print(
            f"{label}: success_rate={summary['success_rate']:.3f} "
            f"avg_score={summary['avg_score']:.1f} "
            f"invalid_action_rate={summary['invalid_action_rate']:.3f} "
            f"ok={summary['ok_n']}/{summary['n']}",
            flush=True,
        )

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = run_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "n",
                "ok_n",
                "error_n",
                "success_rate",
                "avg_score",
                "avg_normalized_score",
                "avg_steps",
                "invalid_action_rate",
                "placement",
            ],
        )
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "label": metric["label"],
                    "n": metric["n"],
                    "ok_n": metric["ok_n"],
                    "error_n": metric["error_n"],
                    "success_rate": metric["success_rate"],
                    "avg_score": metric["avg_score"],
                    "avg_normalized_score": metric["avg_normalized_score"],
                    "avg_steps": metric["avg_steps"],
                    "invalid_action_rate": metric["invalid_action_rate"],
                    "placement": ",".join(metric["placement"]),
                }
            )
    print(f"Wrote {metrics_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
