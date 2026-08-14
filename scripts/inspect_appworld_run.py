#!/usr/bin/env python3
"""Inspect an AppWorld Observation 1 output directory."""

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
    parser.add_argument("--max-chars", type=int, default=1800)
    args = parser.parse_args()

    path = args.run_dir / f"{args.label}.jsonl"
    print(f"RUN {args.run_dir}")
    print(f"LABEL {args.label}")
    print()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        print("=" * 80)
        print(
            f"TASK {row['task_id']} ok={row['ok']} "
            f"success={row.get('evaluation', {}).get('success')} error={row.get('error', '')}"
        )
        print("\n--- stages ---")
        for stage in row.get("stages", []):
            print(f"{stage['stage']} ({stage['tier']}, {stage['latency_s']:.2f}s): {_clip(stage['output'], 500)}")
        print("\n--- generated_code ---")
        print(_clip(row.get("generated_code", ""), args.max_chars))
        print("\n--- repair_code ---")
        print(_clip(row.get("repair_code", ""), args.max_chars))
        print("\n--- execution_outputs ---")
        for output in row.get("execution_outputs", []):
            print(f"[{output.get('stage')}]")
            print(_clip(output.get("output", ""), args.max_chars))
        print("\n--- evaluation ---")
        print(_clip(row.get("evaluation", {}), args.max_chars))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
