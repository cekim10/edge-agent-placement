#!/usr/bin/env python3
"""Compare full_context and summary_only state representations."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_agent.client import build_client  # noqa: E402
from edge_agent.runner import load_incidents, parse_placement, print_metrics, run_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "incidents_distractor.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--placement", default="cloud,cloud,cloud")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--mock", action="store_true", help="Use deterministic local mock instead of vLLM")
    args = parser.parse_args()

    incidents = load_incidents(args.data)
    client = build_client(mock=args.mock, timeout_s=args.timeout_s, max_tokens=args.max_tokens)
    placement = parse_placement(args.placement)
    run_dir = args.output_dir / f"context_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    full = run_experiment(
        incidents=incidents,
        client=client,
        placement=placement,
        context_mode="full_context",
        output_dir=run_dir,
        label="context_full_context",
    )
    print_metrics(full)

    summary = run_experiment(
        incidents=incidents,
        client=client,
        placement=placement,
        context_mode="summary_only",
        output_dir=run_dir,
        label="context_summary_only",
    )
    print_metrics(summary)
    print(f"accuracy_delta_full_minus_summary={full['accuracy'] - summary['accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
