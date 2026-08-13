#!/usr/bin/env python3
"""Run Observation 1 on AppWorld."""

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

from edge_agent.appworld_harness import (  # noqa: E402
    APPWORLD_STAGES,
    AppWorldMockClient,
    AppWorldStageResult,
    AppWorldTaskResult,
    evaluation_success,
    load_appworld_task_ids,
    placement_for_edge_appworld_stage,
    placement_name,
    run_appworld_workflow,
)
from edge_agent.client import build_client  # noqa: E402


def _mock_result(task_id: str, placement: tuple[str, ...]) -> AppWorldTaskResult:
    stages = [
        AppWorldStageResult(stage=stage, tier=tier, latency_s=0.0, output="{}")
        for stage, tier in zip(APPWORLD_STAGES, placement)
    ]
    return AppWorldTaskResult(
        task_id=task_id,
        placement=placement,
        ok=True,
        error="",
        stages=stages,
        generated_code="apis.supervisor.complete_task()",
        repair_code="",
        execution_outputs=[{"stage": "code_generation", "latency_s": 0.0, "code": "mock", "output": "mock"}],
        evaluation={"success": True, "mock": True},
    )


def _summarize(label: str, placement: tuple[str, ...], rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage_latency = {}
    for stage in APPWORLD_STAGES:
        latencies = [
            stage_result["latency_s"]
            for row in rows
            for stage_result in row.get("stages", [])
            if stage_result["stage"] == stage
        ]
        stage_latency[stage] = mean(latencies) if latencies else 0.0
    return {
        "label": label,
        "placement": list(placement),
        "n": len(rows),
        "ok_n": sum(1 for row in rows if row["ok"]),
        "error_n": sum(1 for row in rows if not row["ok"]),
        "success_rate": mean(1.0 if evaluation_success(row.get("evaluation", {})) else 0.0 for row in rows) if rows else 0.0,
        "completion_rate": mean(
            1.0 if "complete_task" in row.get("generated_code", "") or "complete_task" in row.get("repair_code", "") else 0.0
            for row in rows
        )
        if rows
        else 0.0,
        "mean_stage_latency_s": stage_latency,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="dev")
    parser.add_argument("--task-id", action="append", default=None, help="Run a specific task id; can be repeated")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--appworld-root", type=Path, default=None)
    parser.add_argument("--experiment-prefix", default="edge_agent_appworld")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--analysis-max-tokens", type=int, default=160)
    parser.add_argument("--api-plan-max-tokens", type=int, default=160)
    parser.add_argument("--code-max-tokens", type=int, default=384)
    parser.add_argument("--verify-max-tokens", type=int, default=256)
    parser.add_argument("--dump-prompts", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--only-placement",
        choices=["all_cloud", "all_edge", *APPWORLD_STAGES],
        default=None,
        help="Run only all_cloud, all_edge, or one edge-stage variant",
    )
    args = parser.parse_args()

    if args.mock:
        task_ids = args.task_id or [f"mock_{index:03d}" for index in range(1, args.limit + 1)]
        client = AppWorldMockClient()
    else:
        task_ids = load_appworld_task_ids(
            dataset_name=args.dataset_name,
            limit=args.limit,
            task_ids=args.task_id,
            appworld_root=args.appworld_root,
        )
        client = build_client(
            mock=False,
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
            max_tokens_by_stage={
                "task_analysis": args.analysis_max_tokens,
                "api_planning": args.api_plan_max_tokens,
                "code_generation": args.code_max_tokens,
                "execution_verification": args.verify_max_tokens,
            },
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"appworld_observation1_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = run_dir / "prompts" if args.dump_prompts else None

    variants = [args.only_placement] if args.only_placement else ["all_cloud", *APPWORLD_STAGES, "all_edge"]
    metrics = []
    print(f"Loaded {len(task_ids)} AppWorld tasks from dataset={args.dataset_name}", flush=True)
    for edge_stage in variants:
        placement = placement_for_edge_appworld_stage(edge_stage)
        label = placement_name(placement)
        records_path = run_dir / f"{label}.jsonl"
        rows = []
        experiment_name = f"{args.experiment_prefix}_{stamp}_{label}"
        with records_path.open("w", encoding="utf-8") as handle:
            for index, task_id in enumerate(task_ids, start=1):
                print(f"[{label}] task {index}/{len(task_ids)} {task_id}", flush=True)
                if args.mock:
                    result = _mock_result(task_id, placement)
                else:
                    result = run_appworld_workflow(
                        client=client,
                        task_id=task_id,
                        dataset_name=args.dataset_name,
                        placement=placement,
                        experiment_name=experiment_name,
                        appworld_root=args.appworld_root,
                        dump_prompt_dir=prompt_dir / label if prompt_dir is not None else None,
                    )
                row = result.to_dict()
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                success = evaluation_success(row.get("evaluation", {}))
                error_suffix = f" error={row['error']}" if row.get("error") else ""
                print(
                    f"[{label}] task {index}/{len(task_ids)} ok={row['ok']} success={success}{error_suffix}",
                    flush=True,
                )
        summary = _summarize(label, placement, rows)
        metrics.append(summary)
        print(
            f"{label}: success_rate={summary['success_rate']:.3f} "
            f"completion_rate={summary['completion_rate']:.3f} ok={summary['ok_n']}/{summary['n']}",
            flush=True,
        )

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = run_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "n", "ok_n", "error_n", "success_rate", "completion_rate", "placement"],
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
                    "completion_rate": metric["completion_rate"],
                    "placement": ",".join(metric["placement"]),
                }
            )
    print(f"Wrote {metrics_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
