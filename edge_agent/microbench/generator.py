"""Deterministic generator for access-control placement instances."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .schemas import RECOVERABILITY


@dataclass(frozen=True)
class AccessInstance:
    instance_id: str
    a_level: str
    b_level: str
    request: str
    initial_state: dict[str, Any]
    classification: dict[str, Any]
    expected_ops: list[dict[str, str]]
    expected_final_state: dict[str, Any]
    recoverability: str
    invalid_plan: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "a_level": self.a_level,
            "b_level": self.b_level,
            "request": self.request,
            "initial_state": self.initial_state,
            "classification": self.classification,
            "expected_ops": self.expected_ops,
            "expected_final_state": self.expected_final_state,
            "recoverability": self.recoverability,
            "invalid_plan": self.invalid_plan,
        }


def _records(rng: random.Random, count: int) -> list[dict[str, str]]:
    users = [f"user_{i:03d}" for i in range(max(8, count // 2))]
    resources = [f"repo_{i:02d}" for i in range(8)]
    roles = ["reader", "writer", "admin"]
    rows = []
    seen = set()
    while len(rows) < count:
        row = {
            "user_id": rng.choice(users),
            "resource": rng.choice(resources),
            "role": rng.choice(roles),
        }
        key = (row["user_id"], row["resource"], row["role"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def _apply_ops(state: dict[str, Any], ops: list[dict[str, str]]) -> dict[str, Any]:
    roles = {tuple(row.values()) for row in state["roles"]}
    credentials = set(state["credentials"])
    audit = list(state["audit"])
    for op in ops:
        role_key = (op["user_id"], op["resource"], op["role"])
        if op["op"] == "grant_role":
            roles.add(role_key)
            audit.append(f"grant:{':'.join(role_key)}")
        elif op["op"] == "revoke_role":
            roles.discard(role_key)
            audit.append(f"revoke:{':'.join(role_key)}")
        elif op["op"] == "rotate_credential":
            credentials.add(f"rotated:{op['user_id']}:{op['resource']}")
            audit.append(f"rotate:{op['user_id']}:{op['resource']}")
    return {
        "roles": [
            {"user_id": user_id, "resource": resource, "role": role}
            for user_id, resource, role in sorted(roles)
        ],
        "credentials": sorted(credentials),
        "audit": audit,
    }


def _request_text(
    *,
    category: str,
    user_id: str,
    resource: str,
    role: str,
    a_level: str,
    distractor_user: str,
    distractor_resource: str,
) -> str:
    if category == "grant_access":
        direct = f"Grant {role} access on {resource} to {user_id}."
        ambiguous = (
            f"{user_id} needs to work on {resource} as {role}. "
            f"Do not remove access from {distractor_user}; they are only mentioned for context."
        )
    elif category == "revoke_access":
        direct = f"Revoke {role} access on {resource} from {user_id}."
        ambiguous = (
            f"{user_id} should no longer have {role} on {resource}. "
            f"Ignore the unrelated request about granting {distractor_user} access to {distractor_resource}."
        )
    else:
        direct = f"Rotate the credential used by {user_id} for {resource}; keep role assignments unchanged."
        ambiguous = (
            f"The access entry for {user_id} on {resource} is not the problem. "
            "The secret itself was exposed, so update the credential and leave roles as they are."
        )
    return direct if a_level == "easy" else ambiguous


def generate_instances(
    *,
    seed: int,
    count: int,
    a_level: str,
    b_level: str,
) -> list[AccessInstance]:
    if a_level not in {"easy", "hard"}:
        raise ValueError("a_level must be easy or hard")
    if b_level not in {"easy", "hard"}:
        raise ValueError("b_level must be easy or hard")

    rng = random.Random(seed)
    categories = ["grant_access", "revoke_access", "rotate_credential"]
    instances: list[AccessInstance] = []
    record_count = 20 if b_level == "easy" else 120

    for index in range(count):
        rows = _records(rng, record_count)
        target = rng.choice(rows)
        distractor = rng.choice([row for row in rows if row != target])
        category = categories[index % len(categories)]
        op_type = {
            "grant_access": "grant_role",
            "revoke_access": "revoke_role",
            "rotate_credential": "rotate_credential",
        }[category]
        recoverability = {
            "grant_access": "reversible",
            "revoke_access": "compensable",
            "rotate_credential": "irreversible",
        }[category]
        op = {
            "op": op_type,
            "user_id": target["user_id"],
            "resource": target["resource"],
            "role": target["role"],
        }
        initial_state = {
            "roles": rows,
            "credentials": [f"active:{target['user_id']}:{target['resource']}"],
            "audit": [],
        }
        request = _request_text(
            category=category,
            user_id=target["user_id"],
            resource=target["resource"],
            role=target["role"],
            a_level=a_level,
            distractor_user=distractor["user_id"],
            distractor_resource=distractor["resource"],
        )
        wrong_op = {
            "op": "grant_role" if op_type != "grant_role" else "revoke_role",
            "user_id": distractor["user_id"],
            "resource": distractor["resource"],
            "role": distractor["role"],
        }
        instances.append(
            AccessInstance(
                instance_id=f"{a_level}_{b_level}_{seed}_{index:04d}",
                a_level=a_level,
                b_level=b_level,
                request=request,
                initial_state=initial_state,
                classification={
                    "category": category,
                    "user_id": target["user_id"],
                    "resource": target["resource"],
                    "role": target["role"],
                    "confidence": 1.0,
                },
                expected_ops=[op],
                expected_final_state=_apply_ops(initial_state, [op]),
                recoverability=recoverability,
                invalid_plan={"ops": [wrong_op], "violation": "wrong_target_or_action"},
            )
        )
    assert all(item.recoverability in RECOVERABILITY for item in instances)
    return instances

