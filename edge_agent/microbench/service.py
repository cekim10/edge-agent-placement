"""Stateful access-control service used as the commit target."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass
class CommitResult:
    applied: bool
    error: str | None = None
    compensation_supported: bool = False
    compensation_applied: bool = False


class AccessControlService:
    """Small stateful service with explicit compensation semantics."""

    def __init__(self) -> None:
        self._state: dict[str, Any] = {"roles": [], "credentials": [], "audit": []}

    def reset(self, state: dict[str, Any]) -> None:
        self._state = copy.deepcopy(state)

    def state(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    def commit(self, ops: list[dict[str, str]]) -> CommitResult:
        roles = {
            (row["user_id"], row["resource"], row["role"])
            for row in self._state.get("roles", [])
        }
        credentials = set(self._state.get("credentials", []))
        audit = list(self._state.get("audit", []))
        for op in ops:
            key = (op["user_id"], op["resource"], op["role"])
            if op["op"] == "grant_role":
                roles.add(key)
                audit.append(f"grant:{':'.join(key)}")
            elif op["op"] == "revoke_role":
                roles.discard(key)
                audit.append(f"revoke:{':'.join(key)}")
            elif op["op"] == "rotate_credential":
                credentials.add(f"rotated:{op['user_id']}:{op['resource']}")
                audit.append(f"rotate:{op['user_id']}:{op['resource']}")
            else:
                return CommitResult(applied=False, error=f"unknown_op:{op['op']}")
        self._state = {
            "roles": [
                {"user_id": user_id, "resource": resource, "role": role}
                for user_id, resource, role in sorted(roles)
            ],
            "credentials": sorted(credentials),
            "audit": audit,
        }
        return CommitResult(applied=True)

    def compensate(self, ops: list[dict[str, str]], recoverability: str) -> CommitResult:
        if recoverability == "irreversible":
            return CommitResult(
                applied=False,
                error="irreversible_compensation_not_supported",
                compensation_supported=False,
            )
        inverse = []
        for op in reversed(ops):
            if op["op"] == "grant_role":
                inverse.append({**op, "op": "revoke_role"})
            elif op["op"] == "revoke_role":
                inverse.append({**op, "op": "grant_role"})
            elif op["op"] == "rotate_credential":
                return CommitResult(
                    applied=False,
                    error="credential_rotation_compensation_not_supported",
                    compensation_supported=False,
                )
        result = self.commit(inverse)
        return CommitResult(
            applied=result.applied,
            error=result.error,
            compensation_supported=True,
            compensation_applied=result.applied,
        )

