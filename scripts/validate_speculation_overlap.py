#!/usr/bin/env python3
"""Check the assumption the speculative benefit rests on.

Sequential runs compose the speculative critical path as `max(commit, verify)`.
That is an assumption about what happens when the two are actually run at the
same time, and nothing had ever executed it. Contention on the HTTP client, the
GIL, or a server that serialises requests would all make the real overlap longer
than the max -- and the entire latency argument for speculation would shrink or
vanish.

Two things are checked, and they are independent:

  overlap   `t_overlap_measured` against `max(t_commit, t_verify)` on the same
            record. An overhead near zero means the composition is sound.

  state     per-instance state outcome of a concurrent run against a sequential
            one over the same instances. The verifier judges the plan, so the
            verdict cannot depend on whether the commit has landed; if the two
            runs disagree on any instance, that reasoning is wrong and the
            concurrent path has a race.

Pass the concurrent run alone to check the overlap, or both runs to also check
state equivalence.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_speculative(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Speculative records keyed by class and instance id."""
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(str(run_dir / "*_speculative.jsonl"))):
        recoverability = Path(path).name.replace("_speculative.jsonl", "")
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("ok"):
                continue
            records[f"{recoverability}/{record['instance_id']}"] = record
    return records


def report_overlap(records: dict[str, dict[str, Any]], label: str) -> bool:
    measured = [
        (
            record["components"]["t_overlap_measured"],
            max(record["components"]["t_commit"], record["components"]["t_verify"]),
        )
        for record in records.values()
        if record["components"].get("t_overlap_measured") is not None
    ]
    if not measured:
        print(f"{label}: no measured overlap; run with --concurrent-speculation")
        return False

    overheads = [actual - predicted for actual, predicted in measured]
    ratios = [actual / predicted for actual, predicted in measured if predicted > 0]
    print(f"\n{label}: overlap measured on {len(measured)} records")
    print(f"  mean max(c, v)        {statistics.mean(p for _a, p in measured):.4f} s")
    print(f"  mean measured overlap {statistics.mean(a for a, _p in measured):.4f} s")
    print(
        f"  overhead              mean {statistics.mean(overheads)*1000:+.1f} ms, "
        f"median {statistics.median(overheads)*1000:+.1f} ms, "
        f"p95 {sorted(overheads)[int(0.95 * (len(overheads) - 1))]*1000:+.1f} ms"
    )
    if ratios:
        print(
            f"  measured / max        mean {statistics.mean(ratios):.3f}, "
            f"max {max(ratios):.3f}"
        )
    worst = max(overheads)
    print(
        f"  worst case            {worst*1000:+.1f} ms"
        + ("  <- composition holds" if worst < 0.05 else "  <- composition optimistic")
    )
    return True


def report_state(
    concurrent: dict[str, dict[str, Any]], sequential: dict[str, dict[str, Any]]
) -> None:
    shared = sorted(set(concurrent) & set(sequential))
    if not shared:
        print("\nno shared instances between the two runs; state check skipped")
        return
    disagreements = [
        key
        for key in shared
        if concurrent[key]["score"]["state_outcome"]
        != sequential[key]["score"]["state_outcome"]
    ]
    print(f"\nstate equivalence over {len(shared)} shared instances")
    print(f"  identical outcome     {len(shared) - len(disagreements)}/{len(shared)}")
    if disagreements:
        print("  DISAGREEMENTS (the concurrent path is not order-independent):")
        for key in disagreements[:10]:
            print(
                f"    {key}: concurrent={concurrent[key]['score']['state_outcome']} "
                f"sequential={sequential[key]['score']['state_outcome']}"
            )
    else:
        print("  no disagreements: the verdict does not depend on commit ordering")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("concurrent_run", type=Path)
    parser.add_argument("sequential_run", type=Path, nargs="?")
    args = parser.parse_args()

    concurrent = load_speculative(args.concurrent_run)
    if not concurrent:
        print(f"no speculative records in {args.concurrent_run}")
        return 1
    report_overlap(concurrent, args.concurrent_run.name)

    if args.sequential_run is not None:
        sequential = load_speculative(args.sequential_run)
        if sequential:
            report_state(concurrent, sequential)
        else:
            print(f"\nno speculative records in {args.sequential_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
