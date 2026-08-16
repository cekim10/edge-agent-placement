#!/usr/bin/env python3
"""Observation 2: verification is a decision variable, not a fixed stage.

Difficulty and stage placement are held fixed here; the sweep is over which
verifier runs and where. Fixing them is deliberate -- Observation 1 already
swept difficulty, and a figure carries two dimensions, not five.

Latency and RTT
---------------
Accuracy does not depend on RTT: the prompts, the models and the temperature are
identical at every RTT, so re-running the sweep per RTT would burn GPU hours to
reproduce the same answers. Instead each configuration runs once, the real
per-call latency is measured, and the number of calls that crossed to the cloud
tier is recorded. For a strictly sequential pipeline the latency at a given RTT
is then exact arithmetic, not a model:

    latency(rtt) = measured_latency + cloud_calls * rtt

This is projection over measured data, not simulated model quality. Use
--inject-rtt-ms to have the client actually sleep instead, to confirm the
projection against a real run.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge_agent.client import build_client, mss_clamp  # noqa: E402
from edge_agent.microbench.generator import generate_instances  # noqa: E402
from edge_agent.microbench.scorer import (  # noqa: E402
    aggregate_verification,
    score_verification,
)
from edge_agent.microbench.service import AccessControlService  # noqa: E402
from edge_agent.microbench.verifier import VERIFIER_VARIANTS, make_verifier  # noqa: E402
from edge_agent.microbench.workflow import (  # noqa: E402
    MicroMockClient,
    placement_for_edge_stage,
    record_view,
    run_micro_workflow,
)

PLACEMENT_CHOICES = ("all_cloud", "classify", "plan", "all_edge")
INJECTION_CLASSES = ("natural", "wrong_subject", "policy_forbidden_role")
DEFAULT_RTT_MS = "0,10,25,50,100,200"


def injection_for(instance: Any, rate: float) -> str:
    """Pick this instance's plan source deterministically.

    Seeded from the instance id rather than its index so the choice does not
    correlate with the request category, which cycles by index.
    """
    if rate <= 0:
        return "natural"
    rng = random.Random(f"inject:{instance.instance_id}")
    if rng.random() >= rate:
        return "natural"
    return rng.choice(["wrong_subject", "policy_forbidden_role"])


def injected_ops_for(instance: Any, violation: str) -> list[dict[str, str]] | None:
    if violation == "natural":
        return None
    for plan in instance.invalid_plans:
        if plan["violation"] == violation:
            return list(plan["ops"])
    raise ValueError(f"instance has no invalid plan for {violation}")


def _latency(workflow: dict[str, Any]) -> tuple[float, int]:
    """Measured wall-clock for the pipeline, and how many calls hit the cloud."""
    total = sum(float(stage["latency_s"]) for stage in workflow["stages"])
    cloud_calls = sum(1 for stage in workflow["stages"] if stage["tier"] == "cloud")
    verify = workflow.get("verify")
    if verify is not None:
        total += float(verify["latency_s"])
        if verify["tier"] == "cloud":
            cloud_calls += 1
    return total, cloud_calls


def run_variant(
    *,
    variant: str,
    placement: tuple[str, str],
    instances: list[Any],
    client: Any,
    inject_rate: float,
    output_dir: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    service = AccessControlService()
    path = output_dir / f"verify_{variant}.jsonl"

    with path.open("w", encoding="utf-8") as handle:
        for index, instance in enumerate(instances, start=1):
            violation = injection_for(instance, inject_rate)
            try:
                verifier = make_verifier(
                    variant,
                    client=None if variant in {"none", "rule_local"} else client,
                    instance=instance,
                    record_view=record_view(instance),
                )
                workflow = run_micro_workflow(
                    client=client,
                    service=service,
                    instance=instance,
                    placement=placement,
                    verifier=verifier,
                    injected_ops=injected_ops_for(instance, violation),
                )
            except Exception as exc:  # noqa: BLE001 - experiment records failures.
                print(f"[{variant}] {instance.instance_id} error={exc!r}", flush=True)
                handle.write(
                    json.dumps({"ok": False, "instance_id": instance.instance_id,
                                "injected": violation, "error": repr(exc)}) + "\n"
                )
                continue

            if workflow["predicted_ops"] is None:
                # The plan stage never produced a usable op set, so there was
                # nothing to verify. Excluded from verifier rates and counted.
                handle.write(
                    json.dumps({"ok": False, "instance_id": instance.instance_id,
                                "injected": violation, "error": "no_plan"}) + "\n"
                )
                continue

            verify = workflow.get("verify") or {}
            score = score_verification(
                expected_ops=instance.expected_ops,
                predicted_ops=workflow["predicted_ops"],
                approved=bool(verify.get("approved", True)),
                committed=bool(workflow["committed"]),
                verifier_error=verify.get("error"),
            )
            latency, cloud_calls = _latency(workflow)
            record = {
                "ok": True,
                "variant": variant,
                "instance_id": instance.instance_id,
                "a_level": instance.a_level,
                "b_level": instance.b_level,
                "recoverability": instance.recoverability,
                "injected": violation,
                "latency_s": latency,
                "cloud_calls": cloud_calls,
                "workflow": workflow,
                "score": score,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{variant}] {index}/{len(instances)} {instance.instance_id} "
                f"inject={violation} plan_ok={score['plan_correct']} "
                f"approved={score['approved']} unsafe={score['unsafe_commit']} "
                f"{latency:.2f}s cloud_calls={cloud_calls}",
                flush=True,
            )

    metrics = {
        "variant": variant,
        "records_path": str(path),
        **aggregate_verification(records),
        "mean_latency_s": (
            sum(r["latency_s"] for r in records) / len(records) if records else 0.0
        ),
        "mean_cloud_calls": (
            sum(r["cloud_calls"] for r in records) / len(records) if records else 0.0
        ),
        "by_injection": _by_injection(records),
    }
    return metrics


def _by_injection(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for violation in INJECTION_CLASSES:
        subset = [r for r in records if r["injected"] == violation]
        if not subset:
            continue
        rows.append({"injected": violation, **aggregate_verification(subset)})
    return rows


def latency_projection(
    metrics: list[dict[str, Any]], rtt_ms: list[float]
) -> list[dict[str, Any]]:
    rows = []
    for entry in metrics:
        for rtt in rtt_ms:
            rows.append(
                {
                    "variant": entry["variant"],
                    "rtt_ms": rtt,
                    "projected_latency_s": entry["mean_latency_s"]
                    + entry["mean_cloud_calls"] * rtt / 1000.0,
                    "mean_cloud_calls": entry["mean_cloud_calls"],
                    "unsafe_commit_rate": entry["unsafe_commit_rate"],
                    "detection_recall": entry["detection_recall"],
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--a-level", default="hard")
    parser.add_argument("--b-level", default="hard")
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--placement", choices=PLACEMENT_CHOICES, default="all_edge")
    parser.add_argument("--variants", default=",".join(VERIFIER_VARIANTS))
    parser.add_argument(
        "--inject-rate", type=float, default=0.5,
        help="fraction of instances whose plan is replaced by a known-bad plan; "
             "without injection a competent planner leaves the verifier nothing to catch",
    )
    parser.add_argument("--rtt-ms", default=DEFAULT_RTT_MS)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--classify-max-tokens", type=int, default=128)
    parser.add_argument("--plan-max-tokens", type=int, default=512)
    parser.add_argument("--verify-max-tokens", type=int, default=128)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    instances = generate_instances(
        seed=args.seed,
        count=args.instances,
        a_level=args.a_level,
        b_level=args.b_level,
    )
    placement = placement_for_edge_stage(args.placement)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    rtt_ms = [float(v) for v in args.rtt_ms.split(",") if v.strip()]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"micro_obs2_{stamp}"
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
        f"{len(instances)} instances, cell A={args.a_level} B={args.b_level}, "
        f"placement={args.placement}, inject_rate={args.inject_rate}",
        flush=True,
    )
    started = time.perf_counter()
    all_metrics = []
    for variant in variants:
        metrics = run_variant(
            variant=variant,
            placement=placement,
            instances=instances,
            client=client,
            inject_rate=args.inject_rate,
            output_dir=output_dir,
        )
        all_metrics.append(metrics)
        print(
            f"{variant}: detection={metrics['detection_recall']:.3f} "
            f"false_reject={metrics['false_reject_rate']:.3f} "
            f"unsafe_commit={metrics['unsafe_commit_rate']:.3f} "
            f"latency={metrics['mean_latency_s']:.2f}s "
            f"cloud_calls={metrics['mean_cloud_calls']:.2f} n={metrics['n']}",
            flush=True,
        )

    (output_dir / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2) + "\n", encoding="utf-8"
    )

    summary_fields = [
        "variant", "n", "bad_plan_n", "good_plan_n", "plan_correct_rate",
        "approved_rate", "verifier_accuracy", "detection_recall",
        "false_reject_rate", "unsafe_commit_rate", "verifier_failure_rate",
        "mean_latency_s", "mean_cloud_calls",
    ]
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for metrics in all_metrics:
            writer.writerow({key: metrics[key] for key in summary_fields})

    with (output_dir / "detection_by_class.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ["variant", "injected", "n", "detection_recall", "false_reject_rate",
                  "unsafe_commit_rate"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metrics in all_metrics:
            for row in metrics["by_injection"]:
                writer.writerow(
                    {"variant": metrics["variant"], **{k: row[k] for k in fields[1:]}}
                )

    projection = latency_projection(all_metrics, rtt_ms)
    with (output_dir / "latency_vs_rtt.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(projection[0]))
        writer.writeheader()
        writer.writerows(projection)

    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "a_level": args.a_level,
                "b_level": args.b_level,
                "instances": args.instances,
                "placement": args.placement,
                "variants": variants,
                "inject_rate": args.inject_rate,
                "rtt_ms": rtt_ms,
                "tcp_mss_clamp": mss_clamp(),
                "latency_note": "latency(rtt) projected from measured latency and "
                                "cloud call count; accuracy is RTT-independent",
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
