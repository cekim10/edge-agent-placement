#!/usr/bin/env python3
"""Run Observation 1 on the real SWE-agent/test-repo GitHub repository."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_agent.client import build_client  # noqa: E402
from edge_agent.swe_repo import (  # noqa: E402
    SWE_STAGES,
    SWEMockClient,
    apply_patch,
    copy_repo_to_temp,
    load_swe_testrepo_tasks,
    placement_for_edge_swe_stage,
    placement_name,
    run_pytest,
    run_swe_workflow,
)


def _summarize(label: str, placement: tuple[str, ...], rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage_latency = {}
    for stage in SWE_STAGES:
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
        "patch_apply_rate": mean(1.0 if row["patch_apply"].get("applied") else 0.0 for row in rows) if rows else 0.0,
        "pass_rate": mean(1.0 if row["test"].get("passed") else 0.0 for row in rows) if rows else 0.0,
        "mean_stage_latency_s": stage_latency,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", type=Path, default=Path.home() / "test-repo")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--analysis-max-tokens", type=int, default=256)
    parser.add_argument("--patch-max-tokens", type=int, default=768)
    parser.add_argument("--repair-max-tokens", type=int, default=768)
    parser.add_argument("--test-timeout-s", type=float, default=20.0)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--only-placement",
        choices=["all_cloud", "all_edge", *SWE_STAGES],
        default=None,
        help="Run only all_cloud, all_edge, or one edge-stage variant",
    )
    args = parser.parse_args()

    tasks = load_swe_testrepo_tasks(args.repo_path, limit=args.limit)
    stage_max_tokens = {
        "issue_analysis": args.analysis_max_tokens,
        "patch_generation": args.patch_max_tokens,
        "test_repair": args.repair_max_tokens,
    }
    client = SWEMockClient() if args.mock else build_client(
        mock=False,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        max_tokens_by_stage=stage_max_tokens,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"swe_testrepo_observation1_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    variants = [args.only_placement] if args.only_placement else ["all_cloud", *SWE_STAGES, "all_edge"]
    metrics = []
    print(f"Loaded {len(tasks)} tasks from {args.repo_path}", flush=True)
    for edge_stage in variants:
        placement = placement_for_edge_swe_stage(edge_stage)
        label = placement_name(placement)
        records_path = run_dir / f"{label}.jsonl"
        rows = []
        with records_path.open("w", encoding="utf-8") as handle:
            for index, task in enumerate(tasks, start=1):
                print(f"[{label}] task {index}/{len(tasks)} {task.task_id}", flush=True)
                worktree = None
                try:
                    worktree = copy_repo_to_temp(args.repo_path)
                    stage_results, final_patch = run_swe_workflow(
                        client=client,
                        task=task,
                        repo_path=worktree,
                        placement=placement,
                    )
                    patch_result = apply_patch(worktree, final_patch)
                    test_result = run_pytest(worktree, timeout_s=args.test_timeout_s) if patch_result["applied"] else {
                        "passed": False,
                        "result": "patch_failed",
                        "latency_s": 0.0,
                        "stdout": "",
                        "stderr": patch_result.get("stderr", ""),
                    }
                    row = {
                        "task_id": task.task_id,
                        "placement": list(placement),
                        "ok": True,
                        "error": "",
                        "stages": [stage.to_dict() for stage in stage_results],
                        "final_patch": final_patch,
                        "patch_apply": patch_result,
                        "test": test_result,
                    }
                except Exception as exc:  # noqa: BLE001 - keep experiment running.
                    row = {
                        "task_id": task.task_id,
                        "placement": list(placement),
                        "ok": False,
                        "error": repr(exc),
                        "stages": [],
                        "final_patch": "",
                        "patch_apply": {"applied": False, "stdout": "", "stderr": ""},
                        "test": {"passed": False, "result": "workflow_error", "latency_s": 0.0, "stdout": "", "stderr": ""},
                    }
                finally:
                    if worktree is not None:
                        shutil.rmtree(worktree.parent, ignore_errors=True)
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{label}] task {index}/{len(tasks)} ok={row['ok']} "
                    f"applied={row['patch_apply'].get('applied')} "
                    f"passed={row['test'].get('passed')} result={row['test'].get('result')}",
                    flush=True,
                )
        summary = _summarize(label, placement, rows)
        metrics.append(summary)
        print(
            f"{label}: pass_rate={summary['pass_rate']:.3f} "
            f"patch_apply_rate={summary['patch_apply_rate']:.3f} "
            f"ok={summary['ok_n']}/{summary['n']}",
            flush=True,
        )

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = run_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "n", "ok_n", "error_n", "patch_apply_rate", "pass_rate", "placement"],
        )
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "label": metric["label"],
                    "n": metric["n"],
                    "ok_n": metric["ok_n"],
                    "error_n": metric["error_n"],
                    "patch_apply_rate": metric["patch_apply_rate"],
                    "pass_rate": metric["pass_rate"],
                    "placement": ",".join(metric["placement"]),
                }
            )

    print(f"Wrote {metrics_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
