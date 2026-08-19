#!/usr/bin/env python3
"""Compare Observation 3 runs: safety by class, and where the policies cross.

Takes one or more micro_obs3 run directories and prints, per recoverability
class, the safety outcome and the commit cost at which speculation stops losing.

Thresholds
----------
With `a` the approval rate, `c` the commit round trip, `v` the verification
latency (including any RTT it crosses), and `r` the rate at which compensation
succeeds once it runs, the compensation rate is `m = (1 - a) * r` -- a function
of `a`, not a constant. Writing it as a constant makes the tie look like `a = m`,
which is wrong.

    conservative = v + c * a
    speculative  = max(c, v) + c * m

  c <= v:  delta = c * (m - a) = c * [r - a(1 + r)]   ->  a* = r / (1 + r)
  c >  v:  delta = c * (1 + m - a) - v                ->  a* = 1 - v / (c(1+r))

The first has no `v` and no RTT in it: verification crosses the network under
both policies, so the crossing cancels and the tie point cannot move with RTT.
Only once the commit costs more than verification does speculation hide the
verification behind it, and only then does RTT enter -- lowering `a*`, i.e.
making speculation easier the further away the verifier is.

Both forms reproduce the bisection over measured per-record components to within
0.01 across commit costs of 0.5-6 s and RTT of 0-500 ms.

For an irreversible operation compensation refuses at no cost, so `r` and hence
`m` collapse toward 0, `a*` drops, and speculation looks *cheapest* exactly where
it is most damaging: failed recovery costs nothing in time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLASS_ORDER = ("reversible", "compensable", "irreversible")


def load(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    latency_path = run_dir / "latency_vs_rtt.csv"
    latency = []
    if latency_path.exists():
        with latency_path.open(encoding="utf-8", newline="") as handle:
            latency = list(csv.DictReader(handle))
    return {"dir": run_dir, "manifest": manifest, "metrics": metrics, "latency": latency}


def crossover_commit_ms(latency: list[dict[str, Any]], recoverability: str) -> float | None:
    """Smallest swept commit cost at which speculative beats conservative.

    Evaluated at rtt=0; the RTT term is identical for both policies, so the sign
    of the difference does not depend on it.
    """
    if not latency or "commit_ms" not in latency[0]:
        return None
    by_commit: dict[float, dict[str, float]] = {}
    for row in latency:
        if row["recoverability"] != recoverability or float(row["rtt_ms"]) != 0.0:
            continue
        by_commit.setdefault(float(row["commit_ms"]), {})[row["policy"]] = float(
            row["mean_latency_s"]
        )
    for commit_ms in sorted(by_commit):
        pair = by_commit[commit_ms]
        if "speculative" in pair and "conservative" in pair:
            if pair["speculative"] < pair["conservative"]:
                return commit_ms
    return None


def report(run: dict[str, Any]) -> None:
    manifest = run["manifest"]
    print(
        f"\n=== {run['dir'].name}  placement={manifest.get('placement', '?')} "
        f"inject_rate={manifest.get('inject_rate', '?')} "
        f"verifier={manifest.get('verifier', '?')}"
    )
    header = (
        f"{'class':<14}{'policy':<14}{'n':>4}{'plan_ok':>9}{'approved':>10}"
        f"{'damaged':>9}{'policy_dmg':>12}{'recovery':>10}"
    )
    print(header)
    print("-" * len(header))
    rows = {(r["recoverability"], r["policy"]): r for r in run["metrics"]}
    for recoverability in CLASS_ORDER:
        for policy in ("conservative", "speculative"):
            row = rows.get((recoverability, policy))
            if row is None:
                print(f"{recoverability:<14}{policy:<14}{'--':>4}  (not run)")
                continue
            recovery = float(row["recovery_success_rate"])
            recovery_text = "  n/a" if math.isnan(recovery) else f"{recovery:>10.2f}"
            print(
                f"{recoverability:<14}{policy:<14}{int(row['n']):>4}"
                f"{float(row['plan_correct_rate']):>9.2f}"
                f"{float(row['approved_rate']):>10.2f}"
                f"{float(row['state_damaged_rate']):>9.2f}"
                f"{float(row['policy_induced_damage_rate']):>12.2f}"
                f"{recovery_text}"
            )

    print(
        f"\n{'class':<14}{'approve a':>11}{'comp m':>9}{'predicted':>12}{'crossover':>12}"
        "\n(predicted holds while commit <= verify; crossover is measured at rtt=0)"
    )
    print("-" * 58)
    for recoverability in CLASS_ORDER:
        spec = rows.get((recoverability, "speculative"))
        if spec is None:
            continue
        n = int(spec["n"]) or 1
        a = float(spec["approved_rate"])
        invoked = int(spec["compensation_invoked_n"])
        recovery = float(spec["recovery_success_rate"])
        m = 0.0 if math.isnan(recovery) else invoked * recovery / n
        predicted = "speculative" if m < a else "conservative"
        crossing = crossover_commit_ms(run["latency"], recoverability)
        crossing_text = (
            "all swept" if crossing == 0 else
            f"{crossing:.0f} ms" if crossing is not None else "never"
        )
        print(
            f"{recoverability:<14}{a:>11.2f}{m:>9.2f}{predicted:>12}{crossing_text:>12}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, nargs="+")
    args = parser.parse_args()
    for run_dir in args.run_dir:
        if not (run_dir / "metrics.csv").exists():
            print(f"skipping {run_dir}: no metrics.csv")
            continue
        report(load(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
