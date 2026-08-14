"""JSON schemas and validators for the access-control microbenchmark."""

from __future__ import annotations

from typing import Any


CATEGORIES = ("grant_access", "revoke_access", "rotate_credential")
OP_TYPES = ("grant_role", "revoke_role", "rotate_credential")
RECOVERABILITY = ("reversible", "compensable", "irreversible")


CLASSIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "user_id": {"type": "string"},
        "resource": {"type": "string"},
        "role": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["category", "user_id", "resource", "role", "confidence"],
    "additionalProperties": False,
}


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ops": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": list(OP_TYPES)},
                    "user_id": {"type": "string"},
                    "resource": {"type": "string"},
                    "role": {"type": "string"},
                },
                "required": ["op", "user_id", "resource", "role"],
                "additionalProperties": False,
            },
        }
    },
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
    for key in ("user_id", "resource", "role"):
        if not isinstance(value.get(key), str) or not value[key]:
            return f"bad_{key}"
    if not isinstance(value.get("confidence"), (int, float)):
        return "bad_confidence"
    return None


def validate_plan(value: dict[str, Any]) -> str | None:
    ops = value.get("ops")
    if not isinstance(ops, list):
        return "bad_ops"
    for op in ops:
        if not isinstance(op, dict):
            return "bad_op_item"
        if op.get("op") not in OP_TYPES:
            return "bad_op_type"
        for key in ("user_id", "resource", "role"):
            if not isinstance(op.get(key), str) or not op[key]:
                return f"bad_op_{key}"
    return None


def canonical_op(op: dict[str, str]) -> tuple[str, str, str, str]:
    return (op["op"], op["user_id"], op["resource"], op["role"])


def canonical_ops(ops: list[dict[str, str]]) -> set[tuple[str, str, str, str]]:
    return {canonical_op(op) for op in ops}

