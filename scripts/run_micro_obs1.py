#!/usr/bin/env python3
"""Run Observation 1 on the controlled access-control microbenchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_agent.client import build_client, mss_clamp  # noqa: E402
from edge_agent.microbench.generator import generate_instances  # noqa: E402
from edge_agent.microbench.scorer import aggregate, score_instance  # noqa: E402
from edge_agent.microbench.service import AccessControlService  # noqa: E402
from edge_agent.microbench.workflow import (  # noqa: E402
    MicroMockClient,
    placement_for_edge_stage,
    placement_name,
    run_micro_workflow,
)


PLACEMENT_CHOICES = ("all_cloud", "classify", "plan", "all_edge")


def _placements(only: str | None) -> list[tuple[str, str]]:
    if only:
        return [placement_for_edge_stage(only)]
    return [placement_for_edge_stage(choice) for choice in PLACEMENT_CHOICES]


def _load_instances(args: argparse.Namespace) -> list[Any]:
    instances = []
    for a_level in args.a_levels.split(","):
        for b_level in args.b_levels.split(","):
            instances.extend(
                generate_instances(
                    seed=args.seed,
                    count=args.instances_per_cell,
                    a_level=a_level.strip(),
                    b_level=b_level.strip(),
                )
            )
    return instances[: args.limit] if args.limit else instances


def _schema_error(workflow: dict[str, Any]) -> str | None:
    for stage in workflow["stages"]:
        if stage.get("error") is not None:
            return f"{stage['stage']}:{stage['error']}"
    return None


def run_label(
    *,
    label: str,
    placement: tuple[str, str],
    instances: list[Any],
    client: Any,
    output_dir: Path,
) -> dict[str, Any]:
    records = []
    path = output_dir / f"{label}.jsonl"
    service = AccessControlService()
    with path.open("w", encoding="utf-8") as handle:
        for index, instance in enumerate(instances, start=1):
            print(f"[{label}] instance {index}/{len(instances)} {instance.instance_id}", flush=True)
            try:
                workflow = run_micro_workflow(
                    client=client,
                    service=service,
                    instance=instance,
                    placement=placement,
                )
                score = score_instance(
                    expected_classification=instance.classification,
                    predicted_classification=workflow["predicted_classification"],
                    expected_ops=instance.expected_ops,
                    predicted_ops=workflow["predicted_ops"],
                    expected_final_state=instance.expected_final_state,
                    actual_final_state=workflow["final_state"],
                    schema_error=_schema_error(workflow),
                    output_error=_schema_error(workflow),
                )
                ok = not score["schema_violation"] and not score["no_output"]
                record = {
                    "ok": ok,
                    "instance": instance.to_dict(),
                    "placement": list(placement),
                    "workflow": workflow,
                    "score": score,
                }
                stage_summary = " ".join(
                    f"{stage['stage']}={stage['tier']}:{stage['latency_s']:.2f}s"
                    for stage in workflow["stages"]
                )
                print(
                    f"[{label}] success={score['end_to_end_success']} "
                    f"classify={score['classification_correct']} "
                    f"plan={score['plan_exact']} commit={score['commit_correct']} "
                    f"{stage_summary}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - experiment records failures.
                record = {
                    "ok": False,
                    "instance": instance.to_dict(),
                    "placement": list(placement),
                    "workflow": None,
                    "score": {
                        "classification_correct": False,
                        "plan_exact": False,
                        "plan_partial_precision": 0.0,
                        "plan_partial_recall": 0.0,
                        "commit_correct": False,
                        "end_to_end_success": False,
                        "schema_violation": False,
                        "schema_error": None,
                        "no_output": False,
                        "output_error": repr(exc),
                    },
                    "error": repr(exc),
                }
                print(f"[{label}] error={exc!r}", flush=True)
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics = {
        "label": label,
        "placement": list(placement),
        "records_path": str(path),
        **aggregate(records),
        "by_cell": _aggregate_by_cell(records),
    }
    return metrics


def _aggregate_by_cell(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        instance = record["instance"]
        key = (instance["a_level"], instance["b_level"])
        cells.setdefault(key, []).append(record)
    rows = []
    for (a_level, b_level), cell_records in sorted(cells.items()):
        rows.append({"a_level": a_level, "b_level": b_level, **aggregate(cell_records)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--a-levels", default="easy,hard")
    parser.add_argument("--b-levels", default="easy,hard")
    parser.add_argument("--instances-per-cell", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-placement", choices=PLACEMENT_CHOICES)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--classify-max-tokens", type=int, default=64)
    parser.add_argument("--plan-max-tokens", type=int, default=256)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    instances = _load_instances(args)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"micro_obs1_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    client = (
        MicroMockClient()
        if args.mock
        else build_client(
            mock=False,
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
            max_tokens_by_stage={
                "classify": args.classify_max_tokens,
                "plan": args.plan_max_tokens,
            },
        )
    )

    print(
        f"Loaded {len(instances)} generated instances; "
        f"a_levels={args.a_levels} b_levels={args.b_levels}",
        flush=True,
    )
    all_metrics = []
    started = time.perf_counter()
    for placement in _placements(args.only_placement):
        label = placement_name(placement)
        metrics = run_label(
            label=label,
            placement=placement,
            instances=instances,
            client=client,
            output_dir=output_dir,
        )
        all_metrics.append(metrics)
        print(
            f"{label}: e2e={metrics['end_to_end_success_rate']:.3f} "
            f"classify={metrics['classification_accuracy']:.3f} "
            f"plan={metrics['plan_exact_rate']:.3f} "
            f"commit={metrics['commit_success_rate']:.3f} "
            f"schema={metrics['schema_violation_rate']:.3f} n={metrics['n']}",
            flush=True,
        )

    metrics_json = output_dir / "metrics.json"
    metrics_csv = output_dir / "metrics.csv"
    cell_csv = output_dir / "metrics_by_cell.csv"
    metrics_json.write_text(json.dumps(all_metrics, indent=2) + "\n", encoding="utf-8")
    with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "label",
            "n",
            "ok_n",
            "error_n",
            "classification_accuracy",
            "plan_exact_rate",
            "commit_success_rate",
            "end_to_end_success_rate",
            "schema_violation_rate",
            "plan_partial_recall",
            "placement",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in all_metrics:
            row = {key: metrics[key] for key in fieldnames if key != "placement"}
            row["placement"] = ",".join(metrics["placement"])
            writer.writerow(row)
    with cell_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "label",
            "a_level",
            "b_level",
            "n",
            "ok_n",
            "error_n",
            "classification_accuracy",
            "plan_exact_rate",
            "commit_success_rate",
            "end_to_end_success_rate",
            "schema_violation_rate",
            "plan_partial_recall",
            "placement",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in all_metrics:
            for cell in metrics["by_cell"]:
                row = {key: cell[key] for key in fieldnames if key not in {"label", "placement"}}
                row["label"] = metrics["label"]
                row["placement"] = ",".join(metrics["placement"])
                writer.writerow(row)

    manifest = {
        "seed": args.seed,
        "a_levels": args.a_levels,
        "b_levels": args.b_levels,
        "instances_per_cell": args.instances_per_cell,
        "limit": args.limit,
        "tcp_mss_clamp": mss_clamp(),
        "elapsed_s": time.perf_counter() - started,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {metrics_json}")
    print(f"Wrote {metrics_csv}")
    print(f"Wrote {cell_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
