#!/usr/bin/env python3
"""Probe which SWE Stage-B prompt component triggers vLLM/Qwen hangs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_agent.client import build_client  # noqa: E402
from edge_agent.swe_repo import (  # noqa: E402
    SWEStageResult,
    _clip,
    _localized_repo_context,
    _repo_tree_context,
    _sanitize_prompt_text,
    load_swe_testrepo_tasks,
    messages_for_swe_stage,
)


SYSTEM = "You are a deterministic software engineering agent. Return exactly what is requested."


def _messages(kind: str, task, repo_path: Path) -> list[dict[str, str]]:
    issue = _clip(_sanitize_prompt_text(task.problem_statement), 900)
    instruction = (
        "Return a JSON object only with key edits. Each edit has keys file, find, replace. "
        "If unsure return an empty edits list."
    )
    if kind == "tiny":
        user = 'Return JSON only: {"edits":[]}'
    elif kind == "issue_only":
        user = f"{instruction}\n\nISSUE:\n{issue}"
    elif kind == "tree_only":
        user = f"{instruction}\n\nISSUE:\n{issue}\n\n{_repo_tree_context(repo_path)}"
    elif kind == "files_only":
        user = f"{instruction}\n\n{_localized_repo_context(repo_path, [], task.problem_statement)}"
    elif kind == "issue_files":
        user = f"{instruction}\n\nISSUE:\n{issue}\n\n{_localized_repo_context(repo_path, [], task.problem_statement)}"
    elif kind == "full_stage_b_no_prior":
        return messages_for_swe_stage(
            task=task,
            repo_path=repo_path,
            stage="patch_generation",
            results=[],
            include_prior=False,
        )
    elif kind == "full_stage_b_fake_prior":
        fake = SWEStageResult(
            stage="issue_analysis",
            tier="cloud",
            latency_s=0.0,
            output=json.dumps({"likely_files": []}),
        )
        return messages_for_swe_stage(
            task=task,
            repo_path=repo_path,
            stage="patch_generation",
            results=[fake],
            include_prior=True,
        )
    else:
        raise ValueError(f"unknown variant: {kind}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", type=Path, default=Path.home() / "test-repo")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--variants", default="tiny,issue_only,tree_only,files_only,issue_files,full_stage_b_no_prior")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    tasks = load_swe_testrepo_tasks(args.repo_path)
    task = tasks[args.task_index]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    client = build_client(mock=args.mock, timeout_s=args.timeout_s, max_tokens=args.max_tokens)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"swe_prompt_probe_{stamp}"
    prompt_dir = run_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"Task {task.task_id}; variants={','.join(variants)}", flush=True)
    for variant in variants:
        messages = _messages(variant, task, args.repo_path)
        prompt_chars = sum(len(message["content"]) for message in messages)
        prompt_path = prompt_dir / f"{variant}.json"
        prompt_path.write_text(
            json.dumps({"variant": variant, "messages": messages}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[{variant}] prompt_chars={prompt_chars}", flush=True)
        started = time.perf_counter()
        try:
            output = client.chat(tier="cloud", stage=f"probe_{variant}", messages=messages, incident=task.to_incident())
            latency_s = time.perf_counter() - started
            row = {
                "variant": variant,
                "ok": True,
                "error": "",
                "latency_s": latency_s,
                "prompt_chars": prompt_chars,
                "output_preview": output[:200],
            }
            print(f"[{variant}] ok latency_s={latency_s:.2f} output={output[:80]!r}", flush=True)
        except Exception as exc:  # noqa: BLE001 - probe should continue.
            latency_s = time.perf_counter() - started
            row = {
                "variant": variant,
                "ok": False,
                "error": repr(exc),
                "latency_s": latency_s,
                "prompt_chars": prompt_chars,
                "output_preview": "",
            }
            print(f"[{variant}] error={exc}", flush=True)
        rows.append(row)

    csv_path = run_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "ok", "error", "latency_s", "prompt_chars", "output_preview"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
