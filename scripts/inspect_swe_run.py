#!/usr/bin/env python3
"""Summarize a SWE test-repo run JSONL for debugging baseline failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _latest_run(output_dir: Path) -> Path:
    runs = sorted(output_dir.glob("swe_testrepo_observation1_*"), reverse=True)
    if not runs:
        raise FileNotFoundError(f"No swe_testrepo_observation1_* runs under {output_dir}")
    return runs[0]


def _clip(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[TRUNCATED]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    parser.add_argument("--label", default="all_cloud")
    parser.add_argument("--max-chars", type=int, default=1800)
    args = parser.parse_args()

    run_dir = args.run_dir or _latest_run(args.output_dir)
    records_path = run_dir / f"{args.label}.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(records_path)

    print(f"RUN {run_dir}")
    print(f"LABEL {args.label}")
    for line in records_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        print("\n" + "=" * 80)
        print(
            f"TASK {row['task_id']} ok={row['ok']} "
            f"applied={row['patch_apply'].get('applied')} "
            f"passed={row['test'].get('passed')} result={row['test'].get('result')}"
        )
        if row.get("error"):
            print(f"ERROR: {row['error']}")
        for stage in row.get("stages", []):
            print(f"\n--- {stage['stage']} ({stage['tier']}, {stage['latency_s']:.2f}s) ---")
            print(_clip(stage.get("output", ""), args.max_chars))
        print("\n--- final_patch ---")
        print(_clip(row.get("final_patch", ""), args.max_chars))
        print("\n--- patch stderr ---")
        print(_clip(row.get("patch_apply", {}).get("stderr", ""), args.max_chars))
        print("\n--- validation ---")
        test = row.get("test", {})
        print(f"validation={test.get('validation')} command={test.get('command') or test.get('commands')}")
        print(_clip(test.get("stdout", ""), args.max_chars))
        print(_clip(test.get("stderr", ""), args.max_chars))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
