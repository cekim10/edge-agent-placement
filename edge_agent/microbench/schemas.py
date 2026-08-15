"""JSON schemas and validators for the access-control microbenchmark.

Validation here is deliberately structural only: it checks exactly what guided
decoding already enforces. Semantic mistakes (right shape, wrong content) must
fall through to the scorer as wrong answers, otherwise model error would be
counted as `schema_violation` and that metric would stop being a harness-bug
detector.
"""

from __future__ import annotations

from typing import Any


CATEGORIES = ("grant_access", "revoke_access", "rotate_credential")
OP_TYPES = ("grant_role", "revoke_role", "rotate_credential")
ROLES = ("reader", "writer", "admin")
RESOURCES = (
    "github",
    "staging_db",
    "prod_db",
    "billing",
    "ci_runner",
    "artifact_store",
    "vpn",
    "grafana",
)
RECOVERABILITY = ("reversible", "compensable", "irreversible")

# Roles that carry no meaning for rotate_credential are emitted as "".
ROLE_VALUES = [*ROLES, ""]


CLASSIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "subject": {"type": "string"},
        "resource": {"type": "string", "enum": list(RESOURCES)},
        "role": {"type": "string", "enum": ROLE_VALUES},
        "confidence": {"type": "number"},
    },
    "required": ["category", "subject", "resource", "role", "confidence"],
    "additionalProperties": False,
}


OP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": list(OP_TYPES)},
        "user_id": {"type": "string"},
        "resource": {"type": "string", "enum": list(RESOURCES)},
        "role": {"type": "string", "enum": ROLE_VALUES},
    },
    "required": ["op", "user_id", "resource", "role"],
    "additionalProperties": False,
}


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ops": {"type": "array", "items": OP_SCHEMA}},
    "required": ["ops"],
    "additionalProperties": False,
}


VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["approved", "reason"],
    "additionalProperties": False,
}


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a JSON object, allowing common markdown wrapping but no heuristics."""
    import json
    import re

    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.S)
        if match:
            stripped = match.group(1).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error:{exc.msg}"
    if not isinstance(value, dict):
        return None, "not_json_object"
    return value, None


def validate_classification(value: dict[str, Any]) -> str | None:
    if value.get("category") not in CATEGORIES:
        return "bad_category"
    subject = value.get("subject")
    if not isinstance(subject, str) or not subject.strip():
        return "bad_subject"
    if value.get("resource") not in RESOURCES:
        return "bad_resource"
    if value.get("role") not in ROLE_VALUES:
        return "bad_role"
    if not isinstance(value.get("confidence"), (int, float)):
        return "bad_confidence"
    return None


def validate_op(value: Any) -> str | None:
    if not isinstance(value, dict):
        return "op_not_object"
    if value.get("op") not in OP_TYPES:
        return "bad_op_type"
    user_id = value.get("user_id")
    if not isinstance(user_id, str) or not user_id.strip():
        return "bad_op_user_id"
    if value.get("resource") not in RESOURCES:
        return "bad_op_resource"
    if value.get("role") not in ROLE_VALUES:
        return "bad_op_role"
    return None


def validate_plan(value: dict[str, Any]) -> str | None:
    """Structural check only.

    An empty op list is *not* a schema violation: it is a wrong answer that the
    scorer will mark as such. Only a broken shape counts here.
    """
    ops = value.get("ops")
    if not isinstance(ops, list):
        return "ops_not_array"
    for op in ops:
        error = validate_op(op)
        if error is not None:
            return error
    return None


def canonical_op(op: dict[str, str]) -> tuple[str, str, str, str]:
    return (op["op"], op["user_id"], op["resource"], op["role"])


def canonical_ops(ops: list[dict[str, str]]) -> set[tuple[str, str, str, str]]:
    return {canonical_op(op) for op in ops}
