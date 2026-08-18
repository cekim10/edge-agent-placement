"""Stateful access-control service used as the commit target."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CommitResult:
    applied: bool
    error: str | None = None
    compensation_supported: bool = False
    compensation_applied: bool = False
    latency_s: float = 0.0


class AccessControlService:
    """Small stateful service with explicit compensation semantics."""

    def __init__(self, commit_latency_s: float = 0.0) -> None:
        # What one round trip to the service costs. In-process it is zero, which
        # would make speculation pointless: overlapping commit with verification
        # can only hide a cost that exists. Declared as a deployment parameter
        # and recorded in the run manifest rather than left implicit at zero.
        self.commit_latency_s = commit_latency_s
        # What the last commit actually changed. Compensation must undo exactly
        # this, not the inverse of the requested operations: revoking a role the
        # subject never held is a no-op, and blindly granting it back would
        # invent a role that never existed -- compensation damaging the state it
        # was called to repair.
        self._last_delta: dict[str, set[Any]] = {}
        self._state: dict[str, Any] = {
            "users": [],
            "roles": [],
            "credentials": [],
            "audit": [],
        }

    def reset(self, state: dict[str, Any]) -> None:
        self._state = copy.deepcopy(state)

    def state(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    def commit(self, ops: list[dict[str, str]]) -> CommitResult:
        started = time.perf_counter()
        if self.commit_latency_s:
            time.sleep(self.commit_latency_s)
        before_roles = {
            (row["user_id"], row["resource"], row["role"])
            for row in self._state.get("roles", [])
        }
        before_credentials = set(self._state.get("credentials", []))
        roles = {
            (row["user_id"], row["resource"], row["role"])
            for row in self._state.get("roles", [])
        }
        credentials = set(self._state.get("credentials", []))
        audit = list(self._state.get("audit", []))
        for op in ops:
            key = (op["user_id"], op["resource"], op.get("role", ""))
            if op["op"] == "grant_role":
                roles.add(key)
                audit.append(f"grant:{':'.join(key)}")
            elif op["op"] == "revoke_role":
                roles.discard(key)
                audit.append(f"revoke:{':'.join(key)}")
            elif op["op"] == "rotate_credential":
                credentials.discard(f"active:{op['user_id']}:{op['resource']}")
                credentials.add(f"rotated:{op['user_id']}:{op['resource']}")
                audit.append(f"rotate:{op['user_id']}:{op['resource']}")
            else:
                return CommitResult(
                    applied=False,
                    error=f"unknown_op:{op['op']}",
                    latency_s=time.perf_counter() - started,
                )
        self._state = {
            "users": list(self._state.get("users", [])),
            "roles": [
                {"user_id": user_id, "resource": resource, "role": role}
                for user_id, resource, role in sorted(roles)
            ],
            "credentials": sorted(credentials),
            "audit": audit,
        }
        self._last_delta = {
            "roles_added": roles - before_roles,
            "roles_removed": before_roles - roles,
            "credentials_added": credentials - before_credentials,
            "credentials_removed": before_credentials - credentials,
        }
        return CommitResult(applied=True, latency_s=time.perf_counter() - started)

    def compensate(self, ops: list[dict[str, str]], recoverability: str) -> CommitResult:
        """Undo exactly what the last commit changed.

        Costs one more round trip to the service, which is the price speculation
        pays when the verdict comes back no.
        """
        started = time.perf_counter()
        # Feasibility is decided by the operations that were actually committed,
        # not by the request's recoverability label. The two can diverge: an
        # injected policy violation replaces a credential rotation with a role
        # grant, which is trivially reversible, and refusing it on the strength
        # of the label alone reported damage as unrecoverable when it was not.
        # `recoverability` is kept for the audit trail only.
        del recoverability
        if any(op["op"] == "rotate_credential" for op in ops):
            # Fails by contract and costs nothing: a rotated credential is
            # already outside this system's control, so there is no operation to
            # attempt and nothing to recover.
            return CommitResult(
                applied=False,
                error="irreversible_compensation_not_supported",
                compensation_supported=False,
                latency_s=time.perf_counter() - started,
            )

        delta = self._last_delta
        if not delta:
            return CommitResult(
                applied=False,
                error="nothing_to_compensate",
                compensation_supported=True,
                latency_s=time.perf_counter() - started,
            )

        if self.commit_latency_s:
            time.sleep(self.commit_latency_s)
        roles = {
            (row["user_id"], row["resource"], row["role"])
            for row in self._state.get("roles", [])
        }
        credentials = set(self._state.get("credentials", []))
        audit = list(self._state.get("audit", []))

        roles -= delta.get("roles_added", set())
        roles |= delta.get("roles_removed", set())
        credentials -= delta.get("credentials_added", set())
        credentials |= delta.get("credentials_removed", set())
        audit.append(f"compensate:{len(delta.get('roles_added', set()))}+"
                     f"{len(delta.get('roles_removed', set()))}")

        self._state = {
            "users": list(self._state.get("users", [])),
            "roles": [
                {"user_id": user_id, "resource": resource, "role": role}
                for user_id, resource, role in sorted(roles)
            ],
            "credentials": sorted(credentials),
            "audit": audit,
        }
        self._last_delta = {}
        return CommitResult(
            applied=True,
            compensation_supported=True,
            compensation_applied=True,
            latency_s=time.perf_counter() - started,
        )
