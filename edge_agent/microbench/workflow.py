"""Workflow runner for the access-control placement microbenchmark.

The workflow is classify -> plan -> commit. Verification is not a stage here; it
is inserted between plan and commit as a decision variable by later experiments.

The plan prompt states the operation vocabulary and the meaning of each request
category, because without that the ground truth would be ambiguous. It does not
state which operation belongs to the classified category, does not order the
records so the answer comes first, and does not include the request text. What
remains for the model is the actual work: resolve the subject name to a user_id
and build the op set from the records.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any

from edge_agent.client import ChatClient

from .generator import AccessInstance
from .schemas import (
    CLASSIFY_SCHEMA,
    PLAN_SCHEMA,
    parse_json_object,
    validate_classification,
    validate_plan,
)
from .service import AccessControlService


MICRO_STAGES = ("classify", "plan")

_PLAN_SPEC = (
    "Operation vocabulary:",
    "  grant_role(user_id, resource, role)",
    "  revoke_role(user_id, resource, role)",
    "  rotate_credential(user_id, resource) -- carries no role field",
    "Request semantics:",
    "  grant_access adds exactly the role named in the classification.",
    "  revoke_access removes every role the subject currently holds on the resource:",
    "    emit one revoke_role op for each matching row in ASSIGNMENTS.",
    "  rotate_credential replaces the credential and changes no roles.",
    "The classification's role field applies to grant_access only; for the other",
    "categories it is empty and must not be carried into an operation.",
    "Resolve the subject's display name to a user_id using USERS; names may be similar.",
    "Use only the listed records. Return {\"ops\": [...]} with one entry per mutation.",
)


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
            if (
                tier == "edge"
                and instance.a_level == "hard"
                and instance.instance_id.endswith(("0001", "0004", "0007"))
            ):
                wrong = dict(instance.classification)
                wrong["category"] = (
                    "grant_access"
                    if wrong["category"] != "grant_access"
                    else "revoke_access"
                )
                return json.dumps(wrong)
            return json.dumps(instance.classification)
        if stage == "plan":
            classification = _classification_from_messages(messages)
            propagated = any(
                classification.get(key) != instance.classification.get(key)
                for key in ("category", "subject", "resource", "role")
            )
            edge_plan_miss = (
                tier == "edge"
                and instance.b_level == "hard"
                and instance.instance_id.endswith(("0002", "0005", "0008"))
            )
            if propagated or edge_plan_miss:
                return json.dumps({"ops": instance.invalid_plans[0]["ops"]})
            return json.dumps({"ops": instance.expected_ops})
        if stage == "verify":
            ops = _ops_from_messages(messages)
            correct = _canonical(ops) == _canonical(instance.expected_ops)
            # Cloud audits correctly. Edge misses a third of the bad plans, so
            # the aggregation distinguishes verifier tiers during smoke tests.
            if not correct and tier == "edge" and instance.instance_id.endswith(("0000", "0003", "0006")):
                correct = True
            return json.dumps(
                {"approved": correct, "reason": "mock_ok" if correct else "mock_mismatch"}
            )
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
    return "edge_" + "_".join(
        stage for stage, tier in zip(MICRO_STAGES, placement) if tier == "edge"
    )


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


def _ops_from_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    marker = "PROPOSED_OPS: "
    for message in messages:
        content = message.get("content", "")
        if marker in content:
            return json.loads(content.split(marker, 1)[1].split("\n", 1)[0])
    raise ValueError("proposed ops missing from verify prompt")


def _canonical(ops: list[dict[str, str]]) -> set[tuple[str, str, str, str]]:
    return {
        (op["op"], op["user_id"], op["resource"], op.get("role", "") or "")
        for op in ops
    }


def record_view(instance: AccessInstance) -> tuple[str, str]:
    """Render the record tables in a compact, deterministically shuffled order.

    The shuffle is seeded from the instance id so runs are reproducible while the
    ground-truth rows land in an arbitrary position rather than first.
    """
    rng = random.Random(f"view:{instance.instance_id}")
    users = list(instance.initial_state["users"])
    assignments = list(instance.initial_state["roles"])
    rng.shuffle(users)
    rng.shuffle(assignments)
    user_lines = "\n".join(
        f"{user['user_id']}={user['display_name']}" for user in users
    )
    assignment_lines = "\n".join(
        f"{row['user_id']}|{row['resource']}|{row['role']}" for row in assignments
    )
    return user_lines, assignment_lines


def _classify_prompt(instance: AccessInstance) -> list[dict[str, str]]:
    return _messages(
        "You classify access-control requests. Return JSON only.",
        "\n".join(
            [
                "Pick exactly one category: grant_access, revoke_access, rotate_credential.",
                "Report the person named in the request as subject, the resource the "
                "request acts on, and the role to grant.",
                "role must be the empty string unless the category is grant_access.",
                "Some requests mention a second resource only to say it must not change.",
                f"REQUEST: {instance.request}",
            ]
        ),
    )


def _plan_prompt(
    instance: AccessInstance, classification: dict[str, Any]
) -> list[dict[str, str]]:
    user_lines, assignment_lines = record_view(instance)
    return _messages(
        "You plan access-control mutations. Return JSON only.",
        "\n".join(
            [
                *_PLAN_SPEC,
                f"CLASSIFICATION: {json.dumps(classification, sort_keys=True)}",
                "USERS:",
                user_lines,
                "ASSIGNMENTS (user_id|resource|role):",
                assignment_lines,
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
    verifier: Any = None,
    injected_ops: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run classify -> plan -> [verify] -> commit for one instance.

    `verifier` is optional and gates the commit: a rejected plan is not applied.
    `injected_ops` replaces the planner's output after the plan call has been
    made, so controlled bad plans can be fed to the verifier without changing
    the pipeline's shape or its measured latency.
    """
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
            "verify": None,
            "committed": False,
            "final_state": service.state(),
            "predicted_classification": classify.parsed,
            "planned_ops": None,
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
            "verify": None,
            "committed": False,
            "final_state": service.state(),
            "predicted_classification": classify.parsed,
            "planned_ops": None,
            "predicted_ops": None,
        }

    planned_ops = list(plan.parsed["ops"])
    predicted_ops = list(injected_ops) if injected_ops is not None else planned_ops

    verify = verifier(predicted_ops) if verifier is not None else None
    if verify is not None and not verify.approved:
        return {
            "stages": [stage.to_dict() for stage in stages],
            "commit": {"applied": False, "error": "rejected_by_verifier"},
            "verify": verify.to_dict(),
            "committed": False,
            "final_state": service.state(),
            "predicted_classification": classify.parsed,
            "planned_ops": planned_ops,
            "predicted_ops": predicted_ops,
        }

    commit = service.commit(predicted_ops)
    return {
        "stages": [stage.to_dict() for stage in stages],
        "commit": commit.__dict__,
        "verify": verify.to_dict() if verify is not None else None,
        "committed": True,
        "final_state": service.state(),
        "predicted_classification": classify.parsed,
        "planned_ops": planned_ops,
        "predicted_ops": predicted_ops,
    }
