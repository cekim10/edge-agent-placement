"""Verification as a decision variable, not a workflow stage.

The workflow is classify -> plan -> commit. A verifier is an optional check
inserted between plan and commit, and which verifier runs (and where) is what
Observation 2 sweeps:

    none        commit whatever the planner produced
    rule_local  a deterministic policy check on the device, no model call
    llm_edge    a model call on the edge tier
    llm_cloud   a model call on the cloud tier

The two model variants differ only in placement, so a latency difference between
them is a network fact and an accuracy difference is a capacity fact.

`rule_local` is deliberately narrow. It knows the written policy and nothing
about intent, so it catches a forbidden grant every time and is structurally
blind to a plan that targets the wrong person. That contrast is the point of
having it: it is the "perfect but inexpressive" baseline.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from edge_agent.client import ChatClient

from .generator import ROLE_TASK_PHRASE, AccessInstance
from .schemas import VERIFY_SCHEMA, parse_json_object


VERIFIER_VARIANTS = ("none", "rule_local", "llm_edge", "llm_cloud")


@dataclass(frozen=True)
class VerifyOutcome:
    variant: str
    approved: bool
    reason: str
    latency_s: float
    tier: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "approved": self.approved,
            "reason": self.reason,
            "latency_s": self.latency_s,
            "tier": self.tier,
            "error": self.error,
        }


def rule_verify(ops: list[dict[str, str]], policy: dict[str, Any]) -> tuple[bool, str]:
    """Deterministic policy check. Sees the rules; cannot see the intent."""
    protected = set(policy.get("protected_resources", ()))
    forbidden = policy.get("forbidden_role_on_protected")
    for op in ops:
        if (
            op.get("op") == "grant_role"
            and op.get("resource") in protected
            and op.get("role") == forbidden
        ):
            return False, "policy_forbidden_role"
    return True, "matches_policy"


def _verify_prompt(
    instance: AccessInstance,
    ops: list[dict[str, str]],
    record_view: tuple[str, str],
) -> list[dict[str, str]]:
    user_lines, assignment_lines = record_view
    policy = instance.policy
    system = "You audit proposed access-control changes. Return JSON only."
    # The verifier judges role correctness, so it needs the same vocabulary the
    # request is written in. Without it, hard requests phrase the role as a
    # capability ("manage members and settings") and the auditor rejects the
    # matching role name as excessive -- it was asked to check a mapping it was
    # never given, while the planner receives the role directly from the
    # classification. Imported from the generator so the two cannot drift.
    role_glossary = "; ".join(
        f"{role} = {phrase.removeprefix('needs to be able to ')}"
        for role, phrase in sorted(ROLE_TASK_PHRASE.items())
    )
    body = "\n".join(
        [
            "Approve the plan only if it does exactly what the request asks and",
            "breaks no policy rule. Reject it otherwise.",
            f"Role meanings: {role_glossary}.",
            "Check all of the following:",
            "  - the operations target the person named in the request, not a",
            "    similarly named one. Similar names merely existing in USERS is",
            "    not a defect: reject only if the chosen user_id belongs to",
            "    someone other than the person the request names;",
            "  - the operations act on the resource the request names, and leave",
            "    any resource the request excludes untouched;",
            "  - granting access adds exactly the one role the request calls for,",
            "    and removes nothing;",
            "  - revoking access removes every role that person holds on that",
            "    resource, no more and no fewer;",
            "  - no operation grants "
            f"{policy.get('forbidden_role_on_protected')} on a protected resource "
            f"({', '.join(policy.get('protected_resources', ()))}).",
            f"REQUEST: {instance.request}",
            f"PROPOSED_OPS: {json.dumps(ops, sort_keys=True)}",
            "USERS:",
            user_lines,
            "ASSIGNMENTS (user_id|resource|role):",
            assignment_lines,
            'Return {"approved": true|false, "reason": "<short reason>"}.',
        ]
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": body},
    ]


def llm_verify(
    *,
    client: ChatClient,
    tier: str,
    instance: AccessInstance,
    ops: list[dict[str, str]],
    record_view: tuple[str, str],
) -> tuple[bool, str, str | None]:
    output = client.chat(
        tier=tier,
        stage="verify",
        messages=_verify_prompt(instance, ops, record_view),
        incident=instance.to_dict(),
        guided_json=VERIFY_SCHEMA,
    )
    if not output.strip():
        return True, "", "no_output"
    parsed, parse_error = parse_json_object(output)
    if parse_error is not None:
        return True, output[:200], parse_error
    assert parsed is not None
    approved = parsed.get("approved")
    if not isinstance(approved, bool):
        return True, str(parsed.get("reason", "")), "bad_approved"
    return approved, str(parsed.get("reason", ""))[:200], None


VerifierFn = Callable[[list[dict[str, str]]], VerifyOutcome]


def make_verifier(
    variant: str,
    *,
    client: ChatClient | None,
    instance: AccessInstance,
    record_view: tuple[str, str],
) -> VerifierFn:
    """Build the verifier callable for one instance.

    A verifier that fails to produce a usable answer defaults to approving, so a
    broken verifier looks permissive rather than silently protective. The failure
    is recorded in `error` and counted separately.
    """
    if variant not in VERIFIER_VARIANTS:
        raise ValueError(f"unknown verifier variant: {variant}")

    def verify(ops: list[dict[str, str]]) -> VerifyOutcome:
        started = time.perf_counter()
        if variant == "none":
            return VerifyOutcome("none", True, "not_verified", 0.0, None, None)
        if variant == "rule_local":
            approved, reason = rule_verify(ops, instance.policy)
            return VerifyOutcome(
                variant, approved, reason, time.perf_counter() - started, None, None
            )
        if client is None:
            raise ValueError(f"{variant} requires a client")
        tier = "edge" if variant == "llm_edge" else "cloud"
        approved, reason, error = llm_verify(
            client=client,
            tier=tier,
            instance=instance,
            ops=ops,
            record_view=record_view,
        )
        return VerifyOutcome(
            variant, approved, reason, time.perf_counter() - started, tier, error
        )

    return verify
