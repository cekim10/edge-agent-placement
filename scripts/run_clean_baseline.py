#!/usr/bin/env python3
"""Run the required all-cloud clean baseline."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_agent.client import build_client  # noqa: E402
from edge_agent.runner import load_incidents, print_metrics, run_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "incidents_distractor.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--context-mode", choices=["full_context", "summary_only"], default="full_context")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--mock", action="store_true", help="Use deterministic local mock instead of vLLM")
    parser.add_argument("--no-fail", action="store_true", help="Do not return non-zero below threshold")
    args = parser.parse_args()

    incidents = load_incidents(args.data)
    client = build_client(mock=args.mock, timeout_s=args.timeout_s)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = run_experiment(
        incidents=incidents,
        client=client,
        placement=("cloud", "cloud", "cloud"),
        context_mode=args.context_mode,
        output_dir=args.output_dir / f"clean_baseline_{stamp}",
        label=f"clean_baseline_{args.context_mode}",
    )
    print_metrics(metrics)

    if metrics["accuracy"] < args.threshold:
        print(
            f"NO-GO: all-cloud accuracy {metrics['accuracy']:.3f} < {args.threshold:.3f}. "
            "Fix scorer/prompt/data before placement experiments."
        )
        return 0 if args.no_fail else 2

    print(f"GO: all-cloud accuracy {metrics['accuracy']:.3f} >= {args.threshold:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
