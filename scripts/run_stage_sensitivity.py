#!/usr/bin/env python3
"""Run placement sensitivity after the clean baseline passes."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_agent.client import build_client  # noqa: E402
from edge_agent.runner import load_incidents, print_metrics, run_experiment  # noqa: E402


PLACEMENTS = [
    ("cloud", "cloud", "cloud"),
    ("edge", "cloud", "cloud"),
    ("cloud", "edge", "cloud"),
    ("cloud", "cloud", "edge"),
    ("edge", "edge", "edge"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "incidents_distractor.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--context-mode", choices=["full_context", "summary_only"], default="full_context")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--mock", action="store_true", help="Use deterministic local mock instead of vLLM")
    args = parser.parse_args()

    incidents = load_incidents(args.data)
    client = build_client(mock=args.mock, timeout_s=args.timeout_s)
    run_dir = args.output_dir / f"stage_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    baseline_accuracy = None
    for placement in PLACEMENTS:
        label = "placement_" + "-".join(placement)
        metrics = run_experiment(
            incidents=incidents,
            client=client,
            placement=placement,
            context_mode=args.context_mode,
            output_dir=run_dir,
            label=label,
        )
        print_metrics(metrics)
        if placement == ("cloud", "cloud", "cloud"):
            baseline_accuracy = metrics["accuracy"]
        else:
            drop = (baseline_accuracy or 0.0) - metrics["accuracy"]
            print(f"  drop_vs_all_cloud={drop:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
