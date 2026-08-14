#!/usr/bin/env python3
"""Inspect a ScienceWorld Observation 1 output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _clip(text: object, max_chars: int) -> str:
    value = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, indent=2, default=str)
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "\n[TRUNCATED]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--label", default="all_cloud")
    parser.add_argument("--max-chars", type=int, default=1200)
    args = parser.parse_args()

    path = args.run_dir / f"{args.label}.jsonl"
    print(f"RUN {args.run_dir}")
    print(f"LABEL {args.label}")
    print()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        print("=" * 80)
        print(
            f"TASK {row['task_id']} ok={row['ok']} success={row['success']} "
            f"score={row['final_score']:.1f} steps={row['steps_taken']} "
            f"invalid={row['invalid_action_count']} error={row.get('error', '')}"
        )
        print("\n--- steps ---")
        for step in row.get("steps", []):
            print(
                f"{step['step_index']}. action={step['action']!r} "
                f"invalid={step['invalid_action']} score={step['score']:.1f} done={step['done']}"
            )
            print(_clip(step.get("observation", ""), 450))
        print("\n--- stages ---")
        for stage in row.get("stages", []):
            print(f"[step {stage['step_index']}] {stage['stage']} ({stage['tier']}, {stage['latency_s']:.2f}s)")
            print(_clip(stage["output"], args.max_chars))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
