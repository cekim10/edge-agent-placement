#!/usr/bin/env python3
"""Three go/no-go measurements for verifying a plan operation by operation.

The idea under test is that verification need not be one remote verdict on a
whole plan. If the cloud verifier judged operations one at a time, the edge could
commit each as soon as its verdict arrived instead of waiting for all of them.
That would make the benefit independent of the commit cost, which matters here
because the measured commit cost is 0.65 ms and leaves nothing to hide behind.

It only works if three things hold, and all three are cheap to check.

1. latency
   Splitting is worth nothing if the 1.8 s verification is mostly fixed cost.
   The prompt carries the whole record table, so `k` per-operation calls could
   easily cost `k` times a whole-plan call rather than `1/k` of it. Both prompt
   orders are timed: the shipped one, where the operations sit in front of the
   tables, and one with the operations last so that judging several operations
   of the same plan shares a prefix the server can cache. The difference between
   those two is the whole engineering question.

2. accuracy
   Per-operation verdicts, combined by conjunction, are compared against the
   whole-plan verdict on the same plans, both scored against whether the plan
   was actually correct. A method that is faster and wronger is not a method.

3. plan-global residue
   Some violations are properties of the operation set, not of any operation.
   The clearest is omission: a revocation that removes two of three roles has no
   bad operation in it at all. Nothing judged per operation can see that, so this
   is a bound on the approach rather than a tuning problem. It is measured as the
   share of genuinely-wrong plans that every per-operation verdict approves.

Plans come from the recorded runs rather than being generated fresh, so the
operations judged here are the ones the models actually produced.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_agent.client import build_client  # noqa: E402
from edge_agent.microbench.generator import AccessInstance  # noqa: E402
from edge_agent.microbench.schemas import parse_json_object  # noqa: E402
from edge_agent.microbench.schemas import VERIFY_SCHEMA  # noqa: E402
from edge_agent.microbench.verifier import (  # noqa: E402
    _verify_op_prompt,
    _verify_prompt,
)
from edge_agent.microbench.workflow import record_view  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_plans(run_dirs: list[Path], min_ops: int) -> list[dict[str, Any]]:
    """Recorded plans with at least `min_ops` operations, de-duplicated."""
    seen: set[str] = set()
    plans: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        for path in sorted(glob.glob(str(run_dir / "*_speculative.jsonl"))):
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not record.get("ok"):
                    continue
                ops = record["workflow"]["predicted_ops"]
                if len(ops) < min_ops:
                    continue
                key = f"{record['instance_id']}:{json.dumps(ops, sort_keys=True)}"
                if key in seen:
                    continue
                seen.add(key)
                plans.append(record)
    return plans


def ask(client: Any, tier: str, messages: list[dict[str, str]]) -> tuple[bool, float, bool]:
    """(approved, seconds, parse_failed).

    A parse failure is reported rather than folded into a verdict. The workflow
    verifier fails open on unparseable output, which is the right call there --
    a broken auditor should not block the pipeline. Here it would be a
    measurement error: a per-operation prompt that the model answers badly would
    show up as a verifier that approves everything, and look like high accuracy
    on correct plans.
    """
    started = time.perf_counter()
    output = client.chat(
        tier=tier, stage="verify", messages=messages, guided_json=VERIFY_SCHEMA
    )
    latency = time.perf_counter() - started
    parsed, parse_error = parse_json_object(output)
    if parse_error is not None or parsed is None:
        return False, latency, True
    return bool(parsed.get("approved")), latency, False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, nargs="+")
    parser.add_argument("--plans", type=int, default=40)
    parser.add_argument("--min-ops", type=int, default=2,
                        help="only multi-op plans can be split")
    parser.add_argument("--tier", default="cloud")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--verify-max-tokens", type=int, default=256)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs" / "incremental_verification.json")
    args = parser.parse_args()

    plans = load_plans(args.run_dir, args.min_ops)[: args.plans]
    if not plans:
        print("no multi-op plans found in those runs")
        return 1
    client = build_client(
        mock=False, timeout_s=args.timeout_s, max_tokens=args.verify_max_tokens,
        max_tokens_by_stage={"verify": args.verify_max_tokens},
    )
    print(f"{len(plans)} plans, tier={args.tier}, "
          f"op counts {dict(Counter(len(p['workflow']['predicted_ops']) for p in plans))}",
          flush=True)

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(plans, start=1):
        instance = AccessInstance(**record["instance"])
        ops = record["workflow"]["predicted_ops"]
        view = record_view(instance)
        try:
            whole, t_whole, bad_whole = ask(
                client, args.tier, _verify_prompt(instance, ops, view)
            )
            _, t_whole_last, _ = ask(
                client, args.tier, _verify_prompt(instance, ops, view, ops_last=True)
            )
            per_op: list[bool] = []
            t_ops: list[float] = []
            bad_ops = 0
            for position, op in enumerate(ops, start=1):
                approved, latency, bad = ask(
                    client, args.tier,
                    _verify_op_prompt(instance, op, view, position, len(ops)),
                )
                per_op.append(approved)
                t_ops.append(latency)
                bad_ops += int(bad)
        except Exception as exc:  # noqa: BLE001 - a failed plan is data, not a crash
            print(f"  [{index}/{len(plans)}] {record['instance_id']} error={exc!r}",
                  flush=True)
            continue

        rows.append({
            "instance_id": record["instance_id"],
            "n_ops": len(ops),
            "plan_correct": bool(record["score"]["plan_correct"]),
            "whole_approved": whole,
            "per_op_approved": per_op,
            "conjunction": all(per_op),
            "t_whole": t_whole,
            "t_whole_ops_last": t_whole_last,
            "t_ops": t_ops,
            # What the edge would wait for before committing the first operation,
            # against what it waits for now.
            "t_first_op": t_ops[0],
            "parse_failed_whole": bad_whole,
            "parse_failed_ops": bad_ops,
        })
        print(f"  [{index}/{len(plans)}] {record['instance_id']} ops={len(ops)} "
              f"whole={whole} per_op={per_op} correct={rows[-1]['plan_correct']} "
              f"t_whole={t_whole:.2f}s t_first={t_ops[0]:.2f}s", flush=True)

    if not rows:
        print("every plan failed; nothing measured")
        return 1
    report(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")
    return 0


def report(rows: list[dict[str, Any]]) -> None:
    n = len(rows)
    whole = [r["t_whole"] for r in rows]
    whole_last = [r["t_whole_ops_last"] for r in rows]
    first = [r["t_first_op"] for r in rows]
    total_ops = [sum(r["t_ops"]) for r in rows]
    each = [t for r in rows for t in r["t_ops"]]

    print(f"\n=== 1. latency  (n={n} plans, {len(each)} operations) ===")
    print(f"  whole plan, ops first        {statistics.mean(whole):.3f} s")
    print(f"  whole plan, ops last         {statistics.mean(whole_last):.3f} s")
    print(f"  one operation                {statistics.mean(each):.3f} s")
    print(f"  all operations, serial       {statistics.mean(total_ops):.3f} s")
    print(f"  first operation only         {statistics.mean(first):.3f} s")
    ratio = statistics.mean(each) / statistics.mean(whole)
    print(f"\n  one op / whole plan          {ratio:.2f}x")
    print(f"  time to first commit         {statistics.mean(first):.3f} s vs "
          f"{statistics.mean(whole):.3f} s  "
          f"({(1 - statistics.mean(first)/statistics.mean(whole)):+.0%})")
    if statistics.mean(total_ops) > statistics.mean(whole):
        print(f"  NOTE serial per-op verification costs "
              f"{statistics.mean(total_ops)/statistics.mean(whole):.2f}x the whole "
              "plan in total work; only the first-commit time improves.")

    print(f"\n=== 2. accuracy ===")
    print(f"  {'method':<22}{'accuracy':>10}{'detection':>11}{'false reject':>14}")
    print("  " + "-" * 57)
    wrong = [r for r in rows if not r["plan_correct"]]
    right = [r for r in rows if r["plan_correct"]]
    for label, key in (("whole plan", "whole_approved"), ("per-op conjunction", "conjunction")):
        correct = sum(1 for r in rows if r[key] == r["plan_correct"])
        detect = (sum(1 for r in wrong if not r[key]) / len(wrong)) if wrong else float("nan")
        false_rej = (sum(1 for r in right if not r[key]) / len(right)) if right else float("nan")
        print(f"  {label:<22}{correct/n:>10.3f}{detect:>11.3f}{false_rej:>14.3f}")
    print(f"  (n wrong={len(wrong)}, n correct={len(right)})")
    bad_w = sum(1 for r in rows if r["parse_failed_whole"])
    bad_o = sum(r["parse_failed_ops"] for r in rows)
    if bad_w or bad_o:
        print(f"  unparseable verdicts: whole {bad_w}/{n}, per-op {bad_o}/{len(each)}"
              "  <- counted as reject; a large number here invalidates the row above")

    print(f"\n=== 3. plan-global residue ===")
    missed = [r for r in wrong if r["conjunction"]]
    caught_whole = [r for r in wrong if not r["whole_approved"]]
    print(f"  wrong plans every per-op verdict approved: "
          f"{len(missed)}/{len(wrong)}"
          + (f" = {len(missed)/len(wrong):.1%}" if wrong else ""))
    print(f"  of those, the whole-plan verdict caught:   "
          f"{sum(1 for r in missed if not r['whole_approved'])}/{len(missed)}")
    print(f"  (whole-plan detection over the same wrong plans: "
          f"{len(caught_whole)}/{len(wrong)})")
    if missed:
        print("  missed plans by op count:",
              dict(Counter(r["n_ops"] for r in missed)))


if __name__ == "__main__":
    raise SystemExit(main())
