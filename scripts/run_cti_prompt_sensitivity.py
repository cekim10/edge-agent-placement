#!/usr/bin/env python3
"""Sweep CTI-REALM all-cloud prompt variants before placement experiments."""

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
from edge_agent.cti_realm import (  # noqa: E402
    CTI_STAGES,
    CTIMockClient,
    build_data_source_catalog,
    build_mitre_catalog,
    load_cti_realm_records,
    run_cti_workflow,
    score_cti_proxy,
)


VARIANTS = {
    "compact": {
        "mitre_name_style": "compact",
        "mitre_scope": "platform",
        "classifier_objective_chars": 300,
    },
    "platform_names": {
        "mitre_name_style": "full",
        "mitre_scope": "platform",
        "classifier_objective_chars": 700,
    },
    "full_names": {
        "mitre_name_style": "full",
        "mitre_scope": "global",
        "classifier_objective_chars": 700,
    },
    "ids": {
        "mitre_name_style": "ids",
        "mitre_scope": "platform",
        "classifier_objective_chars": 300,
    },
}


ZERO_SCORE = {
    "proxy_quality": 0.0,
    "mitre_jaccard": 0.0,
    "data_source_recall": 0.0,
    "detection_term_recall": 0.0,
}


def _resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.cti_data_dir:
        return args.cti_data_dir
    if args.aces_root:
        return args.aces_root / "domains" / "cti_realm" / "data"
    raise SystemExit("Pass --aces-root /path/to/ACESEvals or --cti-data-dir /path/to/domains/cti_realm/data")


def _parse_variants(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in variants if item not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants {unknown}; choose from {sorted(VARIANTS)}")
    return variants


def _mean(rows: list[dict[str, Any]], key: str, ok_only: bool = False) -> float:
    selected = [row for row in rows if (row["ok"] or not ok_only)]
    if not selected:
        return 0.0
    return mean(row["score"][key] for row in selected)


def _summarize(variant: str, config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_n = sum(1 for row in rows if row["ok"])
    return {
        "variant": variant,
        "n": len(rows),
        "ok_n": ok_n,
        "error_n": len(rows) - ok_n,
        "proxy_quality": _mean(rows, "proxy_quality"),
        "ok_proxy_quality": _mean(rows, "proxy_quality", ok_only=True),
        "mitre_jaccard": _mean(rows, "mitre_jaccard"),
        "data_source_recall": _mean(rows, "data_source_recall"),
        "detection_term_recall": _mean(rows, "detection_term_recall"),
        "mitre_name_style": config["mitre_name_style"],
        "mitre_scope": config["mitre_scope"],
        "classifier_objective_chars": config["classifier_objective_chars"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aces-root", type=Path, default=None)
    parser.add_argument("--cti-data-dir", type=Path, default=None)
    parser.add_argument("--dataset-size", type=int, choices=[25, 50], default=25)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--variants", default="compact,platform_names,full_names")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--cti-max-tokens", type=int, default=64)
    parser.add_argument("--mitre-max-tokens", type=int, default=16)
    parser.add_argument("--data-source-max-tokens", type=int, default=32)
    parser.add_argument("--kql-max-tokens", type=int, default=16)
    parser.add_argument("--rule-max-tokens", type=int, default=16)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mitre-use-prior", action="store_true")
    parser.add_argument("--data-source-use-prior", action="store_true")
    args = parser.parse_args()

    variant_names = _parse_variants(args.variants)
    data_dir = _resolve_data_dir(args)
    all_records = load_cti_realm_records(data_dir, args.dataset_size, limit=0)
    records = all_records[: args.limit] if args.limit else all_records
    data_source_catalog = build_data_source_catalog(all_records)
    platforms = sorted({record.platform for record in all_records})
    platform_records = {
        platform: [record for record in all_records if record.platform == platform]
        for platform in platforms
    }

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
    run_dir = args.output_dir / f"cti_prompt_sensitivity_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    placement = tuple("cloud" for _ in CTI_STAGES)

    print(
        f"Loaded {len(records)} records; variants={','.join(variant_names)} "
        f"data_source_catalog_size={len(data_source_catalog)}",
        flush=True,
    )

    summaries = []
    for variant in variant_names:
        config = VARIANTS[variant]
        global_mitre_catalog = build_mitre_catalog(all_records, name_style=config["mitre_name_style"])
        platform_mitre_catalogs = {
            platform: build_mitre_catalog(items, name_style=config["mitre_name_style"])
            for platform, items in platform_records.items()
        }
        platform_sizes = ",".join(
            f"{platform}:{len(catalog)}" for platform, catalog in sorted(platform_mitre_catalogs.items())
        )
        print(
            f"[{variant}] scope={config['mitre_scope']} "
            f"name_style={config['mitre_name_style']} "
            f"classifier_objective_chars={config['classifier_objective_chars']} "
            f"global_mitre_catalog_size={len(global_mitre_catalog)} "
            f"platform_mitre_catalog_sizes={platform_sizes}",
            flush=True,
        )

        records_path = run_dir / f"{variant}.jsonl"
        rows = []
        with records_path.open("w", encoding="utf-8") as handle:
            for index, record in enumerate(records, start=1):
                if config["mitre_scope"] == "platform":
                    mitre_catalog = platform_mitre_catalogs.get(record.platform, global_mitre_catalog)
                else:
                    mitre_catalog = global_mitre_catalog
                print(f"[{variant}] task {index}/{len(records)} {record.task_id}", flush=True)
                try:
                    stage_results = run_cti_workflow(
                        client=client,
                        record=record,
                        placement=placement,
                        proxy_final_from_c2=True,
                        data_source_catalog=data_source_catalog,
                        mitre_catalog=mitre_catalog,
                        use_mitre_prior=args.mitre_use_prior,
                        use_data_source_prior=args.data_source_use_prior,
                        classifier_objective_chars=config["classifier_objective_chars"],
                    )
                    score = score_cti_proxy(record, stage_results)
                    row = {
                        "task_id": record.task_id,
                        "platform": record.platform,
                        "variant": variant,
                        "ok": True,
                        "error": "",
                        "stages": [stage.to_dict() for stage in stage_results],
                        "score": score,
                    }
                    print(
                        f"[{variant}] task {index}/{len(records)} ok "
                        f"proxy_quality={score['proxy_quality']:.3f} "
                        f"mitre={score['mitre_jaccard']:.3f} "
                        f"data={score['data_source_recall']:.3f}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - experiment should continue after model timeout.
                    row = {
                        "task_id": record.task_id,
                        "platform": record.platform,
                        "variant": variant,
                        "ok": False,
                        "error": repr(exc),
                        "stages": [],
                        "score": dict(ZERO_SCORE),
                    }
                    print(f"[{variant}] task {index}/{len(records)} error={exc}", flush=True)
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
        summary = _summarize(variant, config, rows)
        summaries.append(summary)
        print(
            f"{variant}: ok={summary['ok_n']}/{summary['n']} "
            f"proxy_quality={summary['proxy_quality']:.3f} "
            f"ok_proxy_quality={summary['ok_proxy_quality']:.3f} "
            f"mitre={summary['mitre_jaccard']:.3f} "
            f"data={summary['data_source_recall']:.3f}",
            flush=True,
        )

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = run_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "n",
                "ok_n",
                "error_n",
                "proxy_quality",
                "ok_proxy_quality",
                "mitre_jaccard",
                "data_source_recall",
                "detection_term_recall",
                "mitre_name_style",
                "mitre_scope",
                "classifier_objective_chars",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)

    print(f"Wrote {metrics_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
