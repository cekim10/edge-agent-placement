"""Experiment runner shared by scripts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .client import ChatClient
from .scorer import aggregate_scores, score_answer
from .workflow import ContextMode, Tier, run_workflow


Placement = tuple[Tier, Tier, Tier]


def load_incidents(path: Path) -> list[dict[str, Any]]:
    incidents = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "id" not in item or "incident" not in item or "expected" not in item:
                raise ValueError(f"{path}:{line_no} missing id, incident, or expected")
            incidents.append(item)
    return incidents


def parse_placement(value: str) -> Placement:
    parts = tuple(part.strip() for part in value.split(","))
    if len(parts) != 3 or any(part not in {"edge", "cloud"} for part in parts):
        raise ValueError("placement must look like cloud,edge,cloud")
    return parts  # type: ignore[return-value]


def placement_name(placement: Placement) -> str:
    return "-".join(placement)


def run_experiment(
    *,
    incidents: list[dict[str, Any]],
    client: ChatClient,
    placement: Placement,
    context_mode: ContextMode,
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"{label}.jsonl"
    metrics_path = output_dir / f"{label}.metrics.json"

    records = []
    scores = []
    started = time.perf_counter()
    with records_path.open("w", encoding="utf-8") as handle:
        for incident in incidents:
            workflow_result = run_workflow(
                client=client,
                incident=incident,
                placement=placement,
                context_mode=context_mode,
            )
            score = score_answer(incident, workflow_result.final_answer)
            scores.append(score)
            record = {
                "incident": {"id": incident["id"]},
                "workflow": workflow_result.to_dict(),
                "score": score.to_dict(),
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    stage_latency: dict[str, float] = {}
    for stage in ("triage", "diagnose", "report"):
        latencies = [
            stage_result["latency_s"]
            for record in records
            for stage_result in record["workflow"]["stages"]
            if stage_result["stage"] == stage
        ]
        stage_latency[stage] = sum(latencies) / len(latencies) if latencies else 0.0

    metrics = {
        "label": label,
        "placement": list(placement),
        "context_mode": context_mode,
        "elapsed_s": time.perf_counter() - started,
        "mean_stage_latency_s": stage_latency,
        **aggregate_scores(scores),
        "records_path": str(records_path),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def print_metrics(metrics: dict[str, Any]) -> None:
    print(
        f"{metrics['label']}: "
        f"accuracy={metrics['accuracy']:.3f} "
        f"partial={metrics['partial_score']:.3f} "
        f"avoid_rate={metrics['avoid_rate']:.3f} "
        f"n={metrics['n']} "
        f"elapsed_s={metrics['elapsed_s']:.2f}"
    )

