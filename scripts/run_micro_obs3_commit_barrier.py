#!/usr/bin/env python3
"""Observation 3: the commit barrier.

Sweeps two axes and holds everything else fixed: the recoverability class of the
operation, and whether the commit waits for verification.

    conservative  plan -> verify -> commit
    speculative   plan -> commit, verify in parallel, compensate on reject

What is measured and what is composed
-------------------------------------
State is measured. The speculative path really commits before the verdict, and
really calls `/compensate` when the verdict is no, so whether the world could be
put back is an observed fact per instance, not an assumption. Its three outcomes
are `correct`, `restored` and `damaged`.

Latency is composed from measured component times:

    conservative(rtt) = plan + (verify + rtt) + commit_if_approved
    speculative(rtt)  = plan + max(commit, verify + rtt) + compensate_if_rejected

`max` cannot be reached by the additive projection Observation 2 uses, but it is
still arithmetic over measured components: it assumes the overlap is perfect and
uncontended, and nothing else. Re-running per RTT would reproduce identical
verdicts at GPU cost, since neither the prompts nor the decisions depend on RTT.

Commit is not free. `--commit-latency-ms` is the cost of one round trip to the
service, and speculation can only hide a cost that exists; at zero it is a
no-op by construction. The value used is recorded in the manifest.

Irreversible operations have no speculative curve to compare against, because
`/compensate` fails by contract. Running speculation there anyway is the point
of `--force-speculative-irreversible`: it converts the missing curve into a
measured count of permanent damage.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_agent.client import build_client, mss_clamp  # noqa: E402
from edge_agent.microbench.generator import generate_instances  # noqa: E402
from edge_agent.microbench.injection import injected_ops_for, injection_for  # noqa: E402
from edge_agent.microbench.scorer import (  # noqa: E402
    aggregate_commit_barrier,
    score_commit_barrier,
)
from edge_agent.microbench.service import AccessControlService  # noqa: E402
from edge_agent.microbench.verifier import make_verifier  # noqa: E402
from edge_agent.microbench.workflow import (  # noqa: E402
    COMMIT_POLICIES,
    MicroMockClient,
    placement_for_edge_stage,
    record_view,
    run_commit_barrier_workflow,
)

RECOVERABILITY_CLASSES = ("reversible", "compensable", "irreversible")
DEFAULT_RTT_MS = "0,10,25,50,100,200,500"


def _components(workflow: dict[str, Any]) -> dict[str, float]:
    """Measured per-component times, and which of them crossed to the cloud."""
    stages = workflow["stages"]
    verify = workflow.get("verify") or {}
    compensation = workflow.get("compensation") or {}
    return {
        "t_stages": sum(float(s["latency_s"]) for s in stages),
        "stage_cloud_calls": sum(1 for s in stages if s["tier"] == "cloud"),
        "t_verify": float(verify.get("latency_s") or 0.0),
        "verify_crosses": 1.0 if verify.get("tier") == "cloud" else 0.0,
        "t_commit": float((workflow.get("commit") or {}).get("latency_s") or 0.0),
        "t_compensate": float(compensation.get("latency_s") or 0.0),
    }


def latency_at(components: dict[str, float], policy: str, rtt_s: float) -> float:
    """Compose end-to-end latency for one instance at a given RTT."""
    upstream = components["t_stages"] + components["stage_cloud_calls"] * rtt_s
    verify = components["t_verify"] + components["verify_crosses"] * rtt_s
    if policy == "conservative":
        return upstream + verify + components["t_commit"]
    return upstream + max(components["t_commit"], verify) + components["t_compensate"]


def select_instances(
    *, seed: int, per_class: int, a_level: str, b_level: str
) -> dict[str, list[Any]]:
    """Group generated instances by recoverability class.

    The generator cycles request categories, so asking for 3x and grouping gives
    the same instances each class would get on its own -- no separate seed per
    class, and therefore no draw variance between the classes being compared.
    """
    pool = generate_instances(
        seed=seed, count=per_class * 3 + 6, a_level=a_level, b_level=b_level
    )
    grouped: dict[str, list[Any]] = {name: [] for name in RECOVERABILITY_CLASSES}
    for instance in pool:
        bucket = grouped[instance.recoverability]
        if len(bucket) < per_class:
            bucket.append(instance)
    return grouped


def run_cell(
    *,
    recoverability: str,
    policy: str,
    instances: list[Any],
    client: Any,
    verifier_variant: str,
    placement: tuple[str, str],
    inject_rate: float,
    commit_latency_s: float,
    output_dir: Path,
) -> dict[str, Any]:
    label = f"{recoverability}_{policy}"
    path = output_dir / f"{label}.jsonl"
    service = AccessControlService(commit_latency_s=commit_latency_s)
    records: list[dict[str, Any]] = []

    with path.open("w", encoding="utf-8") as handle:
        for index, instance in enumerate(instances, start=1):
            violation = injection_for(instance, inject_rate)
            try:
                workflow = run_commit_barrier_workflow(
                    client=client,
                    service=service,
                    instance=instance,
                    placement=placement,
                    verifier=make_verifier(
                        verifier_variant,
                        client=None if verifier_variant in {"none", "rule_local"} else client,
                        instance=instance,
                        record_view=record_view(instance),
                    ),
                    policy=policy,
                    injected_ops=injected_ops_for(instance, violation),
                )
            except Exception as exc:  # noqa: BLE001 - experiment records failures.
                print(f"[{label}] {instance.instance_id} error={exc!r}", flush=True)
                handle.write(json.dumps({"ok": False, "instance_id": instance.instance_id,
                                         "error": repr(exc)}) + "\n")
                continue

            if workflow["predicted_ops"] is None:
                handle.write(json.dumps({"ok": False, "instance_id": instance.instance_id,
                                         "error": "no_plan"}) + "\n")
                continue

            score = score_commit_barrier(
                expected_ops=instance.expected_ops,
                predicted_ops=workflow["predicted_ops"],
                initial_state=instance.initial_state,
                expected_final_state=instance.expected_final_state,
                actual_final_state=workflow["final_state"],
                recoverability=recoverability,
                policy=policy,
                approved=bool(workflow["verify"]["approved"]),
                committed=bool(workflow["committed"]),
                compensation=workflow.get("compensation"),
            )
            components = _components(workflow)
            record = {
                "ok": True,
                "label": label,
                "instance_id": instance.instance_id,
                "injected": violation,
                "components": components,
                "workflow": workflow,
                "score": score,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{label}] {index}/{len(instances)} {instance.instance_id} "
                f"approved={score['approved']} state={score['state_outcome']} "
                f"violation={score['unrecoverable_violation']}",
                flush=True,
            )

    return {
        "label": label,
        "recoverability": recoverability,
        "policy": policy,
        "records_path": str(path),
        **aggregate_commit_barrier(records),
        "_records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--a-level", default="hard")
    parser.add_argument("--b-level", default="hard")
    parser.add_argument("--instances-per-class", type=int, default=20)
    parser.add_argument("--placement", default="all_edge",
                        choices=("all_cloud", "classify", "plan", "all_edge"))
    parser.add_argument("--verifier", default="llm_cloud",
                        help="held fixed; llm_cloud is the only variant that both "
                             "detects reliably and does not block most correct plans")
    parser.add_argument("--inject-rate", type=float, default=0.5)
    parser.add_argument("--commit-latency-ms", type=float, default=50.0)
    parser.add_argument("--rtt-ms", default=DEFAULT_RTT_MS)
    parser.add_argument(
        "--force-speculative-irreversible", action="store_true",
        help="run speculation on irreversible operations, which cannot be undone. "
             "Turns the absent curve into a measured count of permanent damage.",
    )
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--classify-max-tokens", type=int, default=128)
    parser.add_argument("--plan-max-tokens", type=int, default=512)
    parser.add_argument("--verify-max-tokens", type=int, default=256)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    grouped = select_instances(
        seed=args.seed,
        per_class=args.instances_per_class,
        a_level=args.a_level,
        b_level=args.b_level,
    )
    rtt_ms = [float(v) for v in args.rtt_ms.split(",") if v.strip()]
    commit_latency_s = args.commit_latency_ms / 1000.0
    placement = placement_for_edge_stage(args.placement)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"micro_obs3_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    client = (
        MicroMockClient()
        if args.mock
        else build_client(
            mock=False,
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
            max_tokens_by_stage={
                "classify": args.classify_max_tokens,
                "plan": args.plan_max_tokens,
                "verify": args.verify_max_tokens,
            },
        )
    )

    print(
        f"cell A={args.a_level} B={args.b_level}, placement={args.placement}, "
        f"verifier={args.verifier}, commit={args.commit_latency_ms}ms, "
        f"{args.instances_per_class}/class",
        flush=True,
    )
    started = time.perf_counter()
    cells: list[dict[str, Any]] = []
    for recoverability in RECOVERABILITY_CLASSES:
        for policy in COMMIT_POLICIES:
            if (
                recoverability == "irreversible"
                and policy == "speculative"
                and not args.force_speculative_irreversible
            ):
                print(
                    "[skip] irreversible + speculative: compensation is unsupported by "
                    "contract, so this configuration is unavailable rather than merely "
                    "worse. Pass --force-speculative-irreversible to measure the damage.",
                    flush=True,
                )
                continue
            cell = run_cell(
                recoverability=recoverability,
                policy=policy,
                instances=grouped[recoverability],
                client=client,
                verifier_variant=args.verifier,
                placement=placement,
                inject_rate=args.inject_rate,
                commit_latency_s=commit_latency_s,
                output_dir=output_dir,
            )
            cells.append(cell)
            print(
                f"{cell['label']}: state_correct={cell['state_correct_rate']:.3f} "
                f"damaged={cell['state_damaged_rate']:.3f} "
                f"unrecoverable={cell['unrecoverable_violation_rate']:.3f} "
                f"recovery={cell['recovery_success_rate']:.3f} n={cell['n']}",
                flush=True,
            )

    latency_rows = []
    for cell in cells:
        for rtt in rtt_ms:
            values = [
                latency_at(record["components"], cell["policy"], rtt / 1000.0)
                for record in cell["_records"]
            ]
            latency_rows.append(
                {
                    "recoverability": cell["recoverability"],
                    "policy": cell["policy"],
                    "rtt_ms": rtt,
                    "mean_latency_s": sum(values) / len(values) if values else 0.0,
                    "unrecoverable_violation_rate": cell["unrecoverable_violation_rate"],
                    "n": cell["n"],
                }
            )

    summary_fields = [
        "label", "recoverability", "policy", "n", "plan_correct_rate", "approved_rate",
        "commit_rate", "state_correct_rate", "state_damaged_rate",
        "unrecoverable_violation_rate", "rejected_n", "compensation_invoked_n",
        "recovery_success_rate",
        "mean_compensation_latency_s",
    ]
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for cell in cells:
            writer.writerow({key: cell[key] for key in summary_fields})
    with (output_dir / "latency_vs_rtt.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(latency_rows[0]))
        writer.writeheader()
        writer.writerows(latency_rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            [{k: v for k, v in cell.items() if k != "_records"} for cell in cells],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "a_level": args.a_level,
                "b_level": args.b_level,
                "instances_per_class": args.instances_per_class,
                "placement": args.placement,
                "verifier": args.verifier,
                "inject_rate": args.inject_rate,
                "commit_latency_ms": args.commit_latency_ms,
                "rtt_ms": rtt_ms,
                "force_speculative_irreversible": bool(args.force_speculative_irreversible),
                "tcp_mss_clamp": mss_clamp(),
                "latency_note": "state measured; latency composed from measured "
                                "components assuming perfect uncontended overlap",
                "elapsed_s": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
