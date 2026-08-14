"""Workflow runner for the access-control placement microbenchmark."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from edge_agent.client import ChatClient

from .generator import AccessInstance
from .schemas import CLASSIFY_SCHEMA, PLAN_SCHEMA, parse_json_object, validate_classification, validate_plan
from .service import AccessControlService


MICRO_STAGES = ("classify", "plan")


@dataclass(frozen=True)
class StageRecord:
    stage: str
    tier: str
    latency_s: float
    output: str
    parsed: dict[str, Any] | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "tier": self.tier,
            "latency_s": self.latency_s,
            "output": self.output,
            "parsed": self.parsed,
            "error": self.error,
        }


class MicroMockClient:
    """Deterministic smoke-test client.

    Cloud is oracle-like. Edge fails on hard classify/plan cells often enough to
    validate aggregation before using GPUs.
    """

    def chat(
        self,
        *,
        tier: str,
        stage: str,
        messages: list[dict[str, str]],
        incident: dict[str, Any] | None = None,
        guided_json: dict[str, Any] | None = None,
    ) -> str:
        del guided_json
        if incident is None:
            raise ValueError("MicroMockClient requires incident")
        instance = AccessInstance(**incident)
        if stage == "classify":
            if tier == "edge" and instance.a_level == "hard" and instance.instance_id.endswith(("0001", "0004", "0007")):
                wrong = dict(instance.classification)
                wrong["category"] = "grant_access" if wrong["category"] != "grant_access" else "revoke_access"
                return json.dumps(wrong)
            return json.dumps(instance.classification)
        if stage == "plan":
            if tier == "edge" and instance.b_level == "hard" and instance.instance_id.endswith(("0002", "0005", "0008")):
                return json.dumps(instance.invalid_plan["ops"][0])
            classification = _classification_from_messages(messages)
            return json.dumps(_op_from_classification(classification))
        raise ValueError(stage)


def placement_for_edge_stage(edge_stage: str | None) -> tuple[str, str]:
    if edge_stage is None or edge_stage == "all_cloud":
        return ("cloud", "cloud")
    if edge_stage == "all_edge":
        return ("edge", "edge")
    if edge_stage not in MICRO_STAGES:
        raise ValueError(f"unknown microbenchmark stage: {edge_stage}")
    return tuple("edge" if stage == edge_stage else "cloud" for stage in MICRO_STAGES)  # type: ignore[return-value]


def placement_name(placement: tuple[str, str]) -> str:
    if placement == ("cloud", "cloud"):
        return "all_cloud"
    if placement == ("edge", "edge"):
        return "all_edge"
    return "edge_" + "_".join(stage for stage, tier in zip(MICRO_STAGES, placement) if tier == "edge")


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _classification_from_messages(messages: list[dict[str, str]]) -> dict[str, Any]:
    marker = "CLASSIFICATION: "
    for message in messages:
        content = message.get("content", "")
        if marker in content:
            raw = content.split(marker, 1)[1].split("\n", 1)[0]
            return json.loads(raw)
    raise ValueError("classification missing from plan prompt")


def _op_from_classification(classification: dict[str, Any]) -> dict[str, str]:
    category_to_op = {
        "grant_access": "grant_role",
        "revoke_access": "revoke_role",
        "rotate_credential": "rotate_credential",
    }
    return {
        "op": category_to_op[str(classification["category"])],
        "user_id": str(classification["user_id"]),
        "resource": str(classification["resource"]),
        "role": str(classification["role"]),
    }


def _candidate_records(instance: AccessInstance, limit: int = 24) -> list[dict[str, str]]:
    target = {
        "user_id": instance.classification["user_id"],
        "resource": instance.classification["resource"],
        "role": instance.classification["role"],
    }
    records = [target]
    for row in instance.initial_state["roles"]:
        if row != target and row not in records:
            records.append(row)
        if len(records) >= limit:
            break
    return records


def _classify_prompt(instance: AccessInstance) -> list[dict[str, str]]:
    return _messages(
        "You classify access-control requests. Return JSON only.",
        "\n".join(
            [
                "Pick exactly one category: grant_access, revoke_access, rotate_credential.",
                "Return category, user_id, resource, role, confidence.",
                f"REQUEST: {instance.request}",
            ]
        ),
    )


def _plan_prompt(instance: AccessInstance, classification: dict[str, Any]) -> list[dict[str, str]]:
    category_to_op = {
        "grant_access": "grant_role",
        "revoke_access": "revoke_role",
        "rotate_credential": "rotate_credential",
    }
    return _messages(
        "You plan access-control mutations. Return JSON only.",
        "\n".join(
            [
                "Use only the supplied classification and candidate records.",
                "Return exactly one operation object with keys op, user_id, resource, role.",
                f"Allowed op for category: {category_to_op}",
                f"CLASSIFICATION: {json.dumps(classification, sort_keys=True)}",
                f"CANDIDATE_RECORDS: {json.dumps(_candidate_records(instance), sort_keys=True)}",
            ]
        ),
    )


def _call_json_stage(
    *,
    client: ChatClient,
    tier: str,
    stage: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    incident: dict[str, Any],
    validator: Any,
) -> StageRecord:
    started = time.perf_counter()
    output = client.chat(
        tier=tier,
        stage=stage,
        messages=messages,
        incident=incident,
        guided_json=schema,
    )
    latency = time.perf_counter() - started
    if not output.strip():
        return StageRecord(stage, tier, latency, output, None, "no_output")
    parsed, parse_error = parse_json_object(output)
    if parse_error is not None:
        return StageRecord(stage, tier, latency, output, None, parse_error)
    assert parsed is not None
    validation_error = validator(parsed)
    return StageRecord(stage, tier, latency, output, parsed, validation_error)


def run_micro_workflow(
    *,
    client: ChatClient,
    service: AccessControlService,
    instance: AccessInstance,
    placement: tuple[str, str],
) -> dict[str, Any]:
    incident = instance.to_dict()
    service.reset(instance.initial_state)
    stages: list[StageRecord] = []

    classify = _call_json_stage(
        client=client,
        tier=placement[0],
        stage="classify",
        messages=_classify_prompt(instance),
        schema=CLASSIFY_SCHEMA,
        incident=incident,
        validator=validate_classification,
    )
    stages.append(classify)
    if classify.error is not None or classify.parsed is None:
        return {
            "stages": [stage.to_dict() for stage in stages],
            "commit": {"applied": False, "error": "classification_failed"},
            "final_state": service.state(),
            "predicted_classification": classify.parsed,
            "predicted_ops": None,
        }

    plan = _call_json_stage(
        client=client,
        tier=placement[1],
        stage="plan",
        messages=_plan_prompt(instance, classify.parsed),
        schema=PLAN_SCHEMA,
        incident=incident,
        validator=validate_plan,
    )
    stages.append(plan)
    if plan.error is not None or plan.parsed is None:
        return {
            "stages": [stage.to_dict() for stage in stages],
            "commit": {"applied": False, "error": "plan_failed"},
            "final_state": service.state(),
            "predicted_classification": classify.parsed,
            "predicted_ops": None,
        }

    predicted_ops = [plan.parsed]
    commit = service.commit(predicted_ops)
    return {
        "stages": [stage.to_dict() for stage in stages],
        "commit": commit.__dict__,
        "final_state": service.state(),
        "predicted_classification": classify.parsed,
        "predicted_ops": predicted_ops,
    }
