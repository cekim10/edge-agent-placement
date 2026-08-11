"""Fixed triage -> diagnose -> report workflow."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

from .client import ChatClient


Stage = Literal["triage", "diagnose", "report"]
Tier = Literal["edge", "cloud"]
ContextMode = Literal["full_context", "summary_only"]

STAGES: tuple[Stage, Stage, Stage] = ("triage", "diagnose", "report")

SYSTEM_PROMPT = (
    "You are a deterministic incident-response analyst. "
    "Use only the provided incident. Do not invent facts. "
    "Keep the answer concise and structured."
)

TRIAGE_PROMPT = """Stage: triage.
Extract the shortest useful state for diagnosis.
Return exactly:
TRIAGE_SUMMARY:
- user_impact:
- strongest_signals:
- likely_distractors:
- missing_checks:
"""

DIAGNOSE_PROMPT = """Stage: diagnose.
Identify the single most likely root cause. Separate evidence from distractors.
Return exactly:
DIAGNOSIS:
Root cause candidate:
Evidence:
Why distractors are not primary:
"""

REPORT_PROMPT = """Stage: report.
Write a final report. The root cause must be explicit and must not contain non-primary distractors.
Return exactly:
ROOT_CAUSE:
<one sentence>

EVIDENCE:
<2-4 bullets>

NOT_MAIN_CAUSE:
<comma-separated distractors>
"""


@dataclass(frozen=True)
class StageResult:
    stage: Stage
    tier: Tier
    latency_s: float
    output: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "tier": self.tier,
            "latency_s": self.latency_s,
            "output": self.output,
        }


@dataclass(frozen=True)
class WorkflowResult:
    incident_id: str
    placement: tuple[Tier, Tier, Tier]
    context_mode: ContextMode
    final_answer: str
    stages: list[StageResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "placement": list(self.placement),
            "context_mode": self.context_mode,
            "final_answer": self.final_answer,
            "stages": [stage.to_dict() for stage in self.stages],
        }


def _incident_block(incident: dict[str, Any]) -> str:
    return f"INCIDENT_ID: {incident['id']}\n\n{incident['incident']}"


def _summary_context(stage_outputs: list[StageResult]) -> str:
    if not stage_outputs:
        return ""
    latest = stage_outputs[-1]
    lines = [line.strip() for line in latest.output.splitlines() if line.strip()]
    return "\n".join(lines[:8])


def _messages_for_stage(
    *,
    incident: dict[str, Any],
    stage: Stage,
    context_mode: ContextMode,
    stage_outputs: list[StageResult],
) -> list[dict[str, str]]:
    if stage == "triage":
        user_content = f"{TRIAGE_PROMPT}\n\n{_incident_block(incident)}"
    elif stage == "diagnose":
        if context_mode == "full_context":
            prior = "\n\n".join(output.output for output in stage_outputs)
            user_content = f"{DIAGNOSE_PROMPT}\n\n{_incident_block(incident)}\n\nPRIOR_STAGE_OUTPUTS:\n{prior}"
        else:
            user_content = f"{DIAGNOSE_PROMPT}\n\nCOMPACT_STATE_FROM_TRIAGE:\n{_summary_context(stage_outputs)}"
    else:
        if context_mode == "full_context":
            prior = "\n\n".join(output.output for output in stage_outputs)
            user_content = f"{REPORT_PROMPT}\n\n{_incident_block(incident)}\n\nPRIOR_STAGE_OUTPUTS:\n{prior}"
        else:
            user_content = f"{REPORT_PROMPT}\n\nCOMPACT_STATE_FROM_DIAGNOSIS:\n{_summary_context(stage_outputs)}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def run_workflow(
    *,
    client: ChatClient,
    incident: dict[str, Any],
    placement: tuple[Tier, Tier, Tier],
    context_mode: ContextMode = "full_context",
) -> WorkflowResult:
    if len(placement) != len(STAGES):
        raise ValueError(f"placement must have {len(STAGES)} tiers")

    stage_results: list[StageResult] = []
    for stage, tier in zip(STAGES, placement):
        messages = _messages_for_stage(
            incident=incident,
            stage=stage,
            context_mode=context_mode,
            stage_outputs=stage_results,
        )
        started = time.perf_counter()
        output = client.chat(tier=tier, stage=stage, messages=messages, incident=incident)
        latency_s = time.perf_counter() - started
        stage_results.append(
            StageResult(stage=stage, tier=tier, latency_s=latency_s, output=output)
        )

    return WorkflowResult(
        incident_id=str(incident["id"]),
        placement=placement,
        context_mode=context_mode,
        final_answer=stage_results[-1].output,
        stages=stage_results,
    )

