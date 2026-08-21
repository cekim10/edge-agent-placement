"""AWS Lambda handler speaking the same commit protocol as service_http.

Purpose
-------
Every commit cost measured so far sits three orders of magnitude below the point
where speculation pays. The gap is not a flaw in the experiment; it is that a
commit inside one cluster is sub-millisecond. The open question is whether any
real deployment puts a commit near the verification latency, and a cold-starting
serverless function is the most common way that actually happens: the agent's
external action is a function invocation, and the first invocation after idle
pays for an execution environment.

So this handler exists to measure one number -- the distribution of `c` when the
commit is a serverless call -- and it deliberately implements the same routes as
the local service so `measure_commit_cost.py` runs against it unchanged.

`cold` in the response
----------------------
A module-level flag is set once at import and cleared on the first invocation.
Import happens per execution environment, so `cold: true` marks exactly the
invocations that paid for a new environment. Reporting it beats inferring it
from the timings, which is circular when the timings are what is in question.

What this measurement does NOT include
--------------------------------------
Durability. The journal goes to /tmp, which is local to the execution
environment and disappears with it, so this is not a durable commit in the sense
the rest of the paper uses the word. Making it durable means a DynamoDB or S3
write, which is another network hop on top of the invocation. That means `c`
measured here is a *lower bound* on a production serverless commit, and the
paper should say so rather than presenting it as the finished figure.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

_COLD = True
_STATE: dict[str, Any] = {"users": [], "roles": [], "credentials": [], "audit": []}
_LAST_DELTA: dict[str, set[Any]] = {}
JOURNAL = "/tmp/commit_journal.log"


def _journal(kind: str, ops: list[dict[str, str]]) -> float:
    started = time.perf_counter()
    line = json.dumps({"kind": kind, "ops": ops, "at": time.time()}) + "\n"
    with open(JOURNAL, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    return time.perf_counter() - started


def _roles() -> set[tuple[str, str, str]]:
    return {(r["user_id"], r["resource"], r["role"]) for r in _STATE.get("roles", [])}


def _store(roles: set[tuple[str, str, str]], credentials: set[str], audit: list[str]) -> None:
    global _STATE
    _STATE = {
        "users": list(_STATE.get("users", [])),
        "roles": [
            {"user_id": u, "resource": r, "role": ro} for u, r, ro in sorted(roles)
        ],
        "credentials": sorted(credentials),
        "audit": audit,
    }


def _commit(ops: list[dict[str, str]]) -> dict[str, Any]:
    global _LAST_DELTA
    before_roles, before_credentials = _roles(), set(_STATE.get("credentials", []))
    roles, credentials = set(before_roles), set(before_credentials)
    audit = list(_STATE.get("audit", []))
    for op in ops:
        key = (op["user_id"], op["resource"], op.get("role", ""))
        if op["op"] == "grant_role":
            roles.add(key)
            audit.append("grant:" + ":".join(key))
        elif op["op"] == "revoke_role":
            roles.discard(key)
            audit.append("revoke:" + ":".join(key))
        elif op["op"] == "rotate_credential":
            credentials.discard(f"{op['user_id']}:{op['resource']}")
            credentials.add(f"{op['user_id']}:{op['resource']}:rotated")
            audit.append(f"rotate:{op['user_id']}:{op['resource']}")
        else:
            return {"applied": False, "error": f"unknown_op:{op['op']}"}
    _LAST_DELTA = {
        "roles_added": roles - before_roles,
        "roles_removed": before_roles - roles,
        "credentials_added": credentials - before_credentials,
        "credentials_removed": before_credentials - credentials,
    }
    _store(roles, credentials, audit)
    return {"applied": True, "error": None}


def _compensate(ops: list[dict[str, str]]) -> dict[str, Any]:
    global _LAST_DELTA
    if any(op["op"] == "rotate_credential" for op in ops):
        return {"applied": False, "error": "irreversible_compensation_not_supported",
                "compensation_supported": False, "compensation_applied": False}
    if not _LAST_DELTA:
        return {"applied": False, "error": "nothing_to_compensate",
                "compensation_supported": True, "compensation_applied": False}
    roles, credentials = _roles(), set(_STATE.get("credentials", []))
    audit = list(_STATE.get("audit", []))
    roles -= _LAST_DELTA.get("roles_added", set())
    roles |= _LAST_DELTA.get("roles_removed", set())
    credentials -= _LAST_DELTA.get("credentials_added", set())
    credentials |= _LAST_DELTA.get("credentials_removed", set())
    audit.append("compensate")
    _store(roles, credentials, audit)
    _LAST_DELTA = {}
    return {"applied": True, "error": None,
            "compensation_supported": True, "compensation_applied": True}


def _respond(payload: dict[str, Any], status: int = 200) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def handler(event: dict[str, Any], _context: Any = None) -> dict[str, Any]:
    global _COLD, _STATE, _LAST_DELTA
    started = time.perf_counter()
    cold, _COLD = _COLD, False

    path = (event.get("rawPath") or event.get("path") or "/").rstrip("/") or "/"
    body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        import base64
        body = base64.b64decode(body).decode("utf-8")
    try:
        payload = json.loads(body) if body else {}
    except ValueError:
        return _respond({"error": "bad_request"}, 400)
    read_s = time.perf_counter() - started

    if path == "/health":
        return _respond({"ok": True, "durable": False, "nagle_disabled": True,
                         "single_write_response": True, "serverless": True,
                         "cold": cold})
    if path == "/state":
        return _respond({"state": _STATE})
    if path == "/reset":
        _STATE = payload["state"]
        _LAST_DELTA = {}
        return _respond({"ok": True, "server_s": time.perf_counter() - started,
                         "cold": cold})

    ops = payload.get("ops", [])
    if path == "/commit":
        result = _commit(ops)
    elif path == "/compensate":
        result = _compensate(ops)
    else:
        return _respond({"error": "not_found"}, 404)

    durability_s = _journal(path[1:], ops) if result.get("applied") else 0.0
    return _respond({
        "applied": result.get("applied", False),
        "error": result.get("error"),
        "compensation_supported": result.get("compensation_supported", False),
        "compensation_applied": result.get("compensation_applied", False),
        "server_s": time.perf_counter() - started,
        "durability_s": durability_s,
        "read_s": read_s,
        "cold": cold,
    })
