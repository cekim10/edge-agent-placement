#!/usr/bin/env python3
"""Measure the commit round trip `c` and split it into its three terms.

`c` is the one parameter in the commit-barrier experiments that has only ever
been simulated. Everything the paper says about speculation is a statement about
`max(c, v)`, so without a real `c` the results say where the policies cross but
not what the crossing is worth. A crossing worth 20 ms and a crossing worth
2 s are different papers.

What is reported
----------------
Per deployment, the distribution of `c` as the client sees it, decomposed as

    c = network + service + durability

`service` and `durability` are timed by the server and returned in the response;
`network` is the remainder. The split matters because only `network` moves with
placement -- the other two are the same wherever the agent runs -- so it decides
how much of `c` the deployment can actually change.

Percentiles, not just the mean: the speculative critical path is `max(c, v)`, a
maximum, so its behaviour is set by the upper tail of `c` rather than its centre.

Latency is injected in the client, not with `tc netem`, because shaping the
interface needs root. That reproduces the delay but not queueing or loss, and a
WAN row here should be read as a floor.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_agent.microbench.generator import generate_instances  # noqa: E402
from edge_agent.microbench.service_http import (  # noqa: E402
    RemoteAccessControlService,
    _ClampedCommitConnection,
)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]


def measure(
    service: RemoteAccessControlService,
    instances: list[Any],
    *,
    repeats: int,
    injected_delay_s: float,
) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {
        "total": [], "server": [], "durability": [], "network": []
    }
    for _ in range(repeats):
        for instance in instances:
            service.reset(instance.initial_state)
            if injected_delay_s:
                time.sleep(injected_delay_s / 2)
            started = time.perf_counter()
            result = service.commit(instance.expected_ops)
            observed = time.perf_counter() - started
            if injected_delay_s:
                time.sleep(injected_delay_s / 2)
                observed += injected_delay_s
            if not result.applied:
                continue
            timings = service.last_timings
            samples["total"].append(observed)
            samples["server"].append(timings.total_s - timings.durability_s)
            samples["durability"].append(timings.durability_s)
            samples["network"].append(observed - timings.total_s)
    return samples


def report(label: str, samples: dict[str, list[float]], verify_s: float) -> None:
    total = samples["total"]
    if not total:
        print(f"{label}: no successful commits")
        return
    print(f"\n{label}  n={len(total)}")
    print(f"  {'term':<14}{'mean':>10}{'p50':>10}{'p95':>10}{'share':>9}")
    print("  " + "-" * 53)
    mean_total = statistics.mean(total)
    for term in ("network", "server", "durability"):
        values = samples[term]
        print(
            f"  {term:<14}{statistics.mean(values)*1000:>9.2f}m"
            f"{percentile(values, 0.50)*1000:>9.2f}m"
            f"{percentile(values, 0.95)*1000:>9.2f}m"
            f"{statistics.mean(values)/mean_total:>8.0%}"
        )
    print(
        f"  {'TOTAL c':<14}{mean_total*1000:>9.2f}m"
        f"{percentile(total, 0.50)*1000:>9.2f}m"
        f"{percentile(total, 0.95)*1000:>9.2f}m"
    )
    ratio = mean_total / verify_s if verify_s else float("nan")
    regime = "c <= v (commit hides behind verification)" if ratio <= 1 else \
             "c > v (verification hides behind the commit)"
    print(f"  c/v = {ratio:.3f} against v={verify_s:.3f}s   -> {regime}")
    # Speculation can only hide a cost that exists on the other side of the
    # overlap, so what it wins per approved request is bounded by the smaller of
    # the two, however large the other one gets.
    print(f"  prize per approved request = min(v, c) = {min(verify_s, mean_total)*1000:.1f} ms")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True,
                        help="host:port of a running service_http instance")
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--a-level", default="hard")
    parser.add_argument("--b-level", default="hard")
    parser.add_argument(
        "--verify-s", type=float, default=1.108,
        help="measured verification latency to compare against; the default is "
             "the mean over the n=150 all_cloud and edge_plan runs",
    )
    parser.add_argument(
        "--inject-delay-ms", default="0",
        help="comma-separated one-way delays to add in the client, standing in "
             "for links this cluster does not have",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    instances = generate_instances(
        seed=args.seed, count=args.instances,
        a_level=args.a_level, b_level=args.b_level,
    )
    service = RemoteAccessControlService(args.endpoint)

    results: dict[str, Any] = {"endpoint": args.endpoint, "verify_s": args.verify_s,
                               "deployments": {}}
    service.state()  # opens the connection so the clamp state is known
    print(
        f"endpoint {args.endpoint}   TCP_MSS_CLAMP applied: "
        f"{_ClampedCommitConnection.clamp_applied}"
    )
    results["mss_clamp_applied"] = _ClampedCommitConnection.clamp_applied
    for raw in str(args.inject_delay_ms).split(","):
        delay_ms = float(raw.strip() or 0)
        label = (f"{args.endpoint}" if delay_ms == 0
                 else f"{args.endpoint} +{delay_ms:.0f}ms injected")
        samples = measure(
            service, instances, repeats=args.repeats,
            injected_delay_s=delay_ms / 1000.0,
        )
        report(label, samples, args.verify_s)
        results["deployments"][label] = {
            term: {"mean": statistics.mean(v), "p50": percentile(v, 0.50),
                   "p95": percentile(v, 0.95), "n": len(v)}
            for term, v in samples.items() if v
        }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
