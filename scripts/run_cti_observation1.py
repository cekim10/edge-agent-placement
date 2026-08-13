#!/usr/bin/env python3
"""Run CTI-REALM Observation 1 stage-sensitivity experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_agent.client import build_client  # noqa: E402
from edge_agent.cti_realm import (  # noqa: E402
    CTI_STAGES,
    CTIMockClient,
    build_data_source_catalog,
    build_mitre_catalog,
    load_cti_realm_records,
    placement_for_edge_stage,
    placement_name,
    run_cti_workflow,
    score_cti_proxy,
)


def _resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.cti_data_dir:
        return args.cti_data_dir
    if args.aces_root:
        return args.aces_root / "domains" / "cti_realm" / "data"
    raise SystemExit("Pass --aces-root /path/to/ACESEvals or --cti-data-dir /path/to/domains/cti_realm/data")


def _summarize(label: str, placement: tuple[str, ...], rows: list[dict]) -> dict:
    scores = [row["score"] for row in rows]
    stage_latency = {}
    for stage in CTI_STAGES:
        latencies = [
            stage_result["latency_s"]
            for row in rows
            for stage_result in row["stages"]
            if stage_result["stage"] == stage
        ]
        stage_latency[stage] = mean(latencies) if latencies else 0.0
    return {
        "label": label,
        "placement": list(placement),
        "n": len(rows),
        "proxy_quality": mean(score["proxy_quality"] for score in scores) if scores else 0.0,
        "mitre_jaccard": mean(score["mitre_jaccard"] for score in scores) if scores else 0.0,
        "data_source_recall": mean(score["data_source_recall"] for score in scores) if scores else 0.0,
        "detection_term_recall": mean(score["detection_term_recall"] for score in scores) if scores else 0.0,
        "mean_stage_latency_s": stage_latency,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aces-root", type=Path, default=None)
    parser.add_argument("--cti-data-dir", type=Path, default=None)
    parser.add_argument("--dataset-size", type=int, choices=[25, 50], default=25)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--cti-max-tokens", type=int, default=64)
    parser.add_argument("--mitre-max-tokens", type=int, default=16)
    parser.add_argument("--data-source-max-tokens", type=int, default=32)
    parser.add_argument("--kql-max-tokens", type=int, default=16)
    parser.add_argument("--rule-max-tokens", type=int, default=16)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--mitre-use-prior",
        action="store_true",
        help="Include prior C0 output in C1 MITRE mapping; disabled by default to avoid vLLM stalls",
    )
    parser.add_argument(
        "--data-source-use-prior",
        action="store_true",
        help="Include prior C0/C1 output in C2 data-source discovery; disabled by default to avoid vLLM stalls",
    )
    parser.add_argument(
        "--proxy-final-from-c2",
        action="store_true",
        help="Run C0-C2 with LLMs and synthesize C3-C4 to avoid KQL/rule-generation stalls",
    )
    parser.add_argument(
        "--only-placement",
        choices=["all_cloud", *CTI_STAGES],
        default=None,
        help="Run only all_cloud or one edge-stage variant for smoke tests",
    )
    args = parser.parse_args()

    if args.max_tokens < 64:
        print(
            "WARNING: --max-tokens below 64 is intended only for timeout smoke tests; "
            "quality scores may be meaningless.",
            flush=True,
        )

    data_dir = _resolve_data_dir(args)
    all_records = load_cti_realm_records(data_dir, args.dataset_size, limit=0)
    data_source_catalog = build_data_source_catalog(all_records)
    mitre_catalog = build_mitre_catalog(all_records)
    records = all_records[: args.limit] if args.limit else all_records
    print(
        f"Loaded {len(records)} records; "
        f"data_source_catalog_size={len(data_source_catalog)} "
        f"mitre_catalog_size={len(mitre_catalog)}",
        flush=True,
    )
    stage_max_tokens = {
        "cti_analysis": args.cti_max_tokens,
        "mitre_mapping": args.mitre_max_tokens,
        "data_source_discovery": args.data_source_max_tokens,
        "kql_development": args.kql_max_tokens,
        "rule_generation": args.rule_max_tokens,
    }
    client = CTIMockClient() if args.mock else build_client(
        mock=False,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        max_tokens_by_stage=stage_max_tokens,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"cti_observation1_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    edge_stage_variants = [args.only_placement] if args.only_placement else ["all_cloud", *CTI_STAGES]
    metrics = []
    for edge_stage in edge_stage_variants:
        placement = placement_for_edge_stage(edge_stage)
        label = placement_name(placement)
        records_path = run_dir / f"{label}.jsonl"
        rows = []
        with records_path.open("w", encoding="utf-8") as handle:
            for index, record in enumerate(records, start=1):
                print(f"[{label}] task {index}/{len(records)} {record.task_id}", flush=True)
                stage_results = run_cti_workflow(
                    client=client,
                    record=record,
                    placement=placement,
                    proxy_final_from_c2=args.proxy_final_from_c2,
                    data_source_catalog=data_source_catalog,
                    mitre_catalog=mitre_catalog,
                    use_mitre_prior=args.mitre_use_prior,
                    use_data_source_prior=args.data_source_use_prior,
                )
                score = score_cti_proxy(record, stage_results)
                row = {
                    "task_id": record.task_id,
                    "platform": record.platform,
                    "placement": list(placement),
                    "stages": [stage.to_dict() for stage in stage_results],
                    "score": score,
                }
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{label}] task {index}/{len(records)} "
                    f"proxy_quality={score['proxy_quality']:.3f} "
                    f"mitre={score['mitre_jaccard']:.3f} "
                    f"data={score['data_source_recall']:.3f}",
                    flush=True,
                )
        summary = _summarize(label, placement, rows)
        metrics.append(summary)
        print(
            f"{label}: proxy_quality={summary['proxy_quality']:.3f} "
            f"mitre={summary['mitre_jaccard']:.3f} "
            f"data={summary['data_source_recall']:.3f} "
            f"detect_terms={summary['detection_term_recall']:.3f} "
            f"n={summary['n']}",
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
                "proxy_quality",
                "mitre_jaccard",
                "data_source_recall",
                "detection_term_recall",
                "placement",
            ],
        )
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "label": metric["label"],
                    "n": metric["n"],
                    "proxy_quality": metric["proxy_quality"],
                    "mitre_jaccard": metric["mitre_jaccard"],
                    "data_source_recall": metric["data_source_recall"],
                    "detection_term_recall": metric["detection_term_recall"],
                    "placement": ",".join(metric["placement"]),
                }
            )

    print(f"Wrote {metrics_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
