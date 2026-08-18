#!/usr/bin/env python3
"""Compare Observation 3 runs: safety by class, and where the policies cross.

Takes one or more micro_obs3 run directories and prints, per recoverability
class, the safety outcome and the commit cost at which speculation stops losing.

The crossover is in commit cost, not RTT. Verification crosses the network under
both policies, so both curves have slope 1 in RTT and never cross; what
speculation hides is the commit. Algebraically, with `a` the approval rate and
`m` the rate at which compensation actually ran:

    speculative - conservative = commit * (m - a)        for commit <= verify

so speculation wins exactly when compensation runs less often than approval
does. For an irreversible operation compensation refuses at no cost, `m` is 0,
and speculation is *always* the faster policy -- which is the trap: it is fastest
because it never even tries to undo the damage it caused.
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
