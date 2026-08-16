"""Deterministic generator for access-control placement instances.

Two difficulty axes, frozen before any measurement (see
docs/microbenchmark_design.md):

`a_level` controls what the classify stage must recover from the request text.
  easy: the category, resource and role are stated directly.
  hard: the intent is implied ("their contract ended", "the secret leaked"), a
        second resource appears inside a negative clause, and the role must be
        inferred from a task description rather than named.

`b_level` controls what the plan stage must do with the record set.
  easy: 24 assignments, 12 users, revoke targets a single role.
  hard: 120 assignments, 40 users including three look-alike names, revoke
        targets two or three roles so the op set is not a single copy.

Identifiers never appear verbatim in the request: the request names a person
("Mina Park") and the plan stage must resolve that to a `user_id` from the
record table. That resolution is the reason look-alike names bite.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .schemas import RECOVERABILITY, RESOURCES, ROLES


PROTECTED_RESOURCES = ("prod_db", "billing")
FORBIDDEN_ROLE_ON_PROTECTED = "admin"

FIRST_NAMES = (
    "mina", "jonah", "priya", "omar", "lena", "david", "sara", "kai",
    "ines", "tomas", "yuki", "noor", "elena", "rafael", "hana", "peter",
)
LAST_NAMES = (
    "park", "reyes", "shah", "haddad", "novak", "kim", "oliveira",
    "tanaka", "weber", "costa", "lindqvist", "moreau",
)

ROLE_TASK_PHRASE = {
    "reader": "needs to be able to look at",
    "writer": "needs to be able to push changes to",
    "admin": "needs to be able to manage members and settings for",
}

CATEGORY_TO_OP = {
    "grant_access": "grant_role",
    "revoke_access": "revoke_role",
    "rotate_credential": "rotate_credential",
}

CATEGORY_TO_RECOVERABILITY = {
    "grant_access": "reversible",
    "revoke_access": "compensable",
    "rotate_credential": "irreversible",
}

CELL_SHAPE = {
    "easy": {"users": 12, "assignments": 24, "revoke_roles": (1,)},
    "hard": {"users": 40, "assignments": 120, "revoke_roles": (2, 3)},
}


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
    policy: dict[str, Any]
    invalid_plans: list[dict[str, Any]]

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
            "policy": self.policy,
            "invalid_plans": self.invalid_plans,
        }


def _user_pool(rng: random.Random, count: int) -> list[dict[str, str]]:
    combos: set[tuple[str, str]] = set()
    users: list[dict[str, str]] = []
    while len(users) < count:
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        if (first, last) in combos:
            continue
        combos.add((first, last))
        users.append(
            {
                "user_id": f"u{len(users):03d}",
                "display_name": f"{first.capitalize()} {last.capitalize()}",
            }
        )
    return users


def _lookalike_names(target_display: str, taken: set[str], rng: random.Random) -> list[str]:
    """Names close enough that a sloppy resolution picks the wrong user_id."""
    first, last = target_display.split(" ", 1)
    candidates = [
        f"{first} {last}e",
        f"{first} {rng.choice(LAST_NAMES).capitalize()}",
        f"{rng.choice(FIRST_NAMES).capitalize()} {last}",
    ]
    names = []
    for name in candidates:
        if name not in taken and name != target_display:
            taken.add(name)
            names.append(name)
    return names


def _assignments(
    rng: random.Random,
    users: list[dict[str, str]],
    count: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    while len(rows) < count:
        row = {
            "user_id": rng.choice(users)["user_id"],
            "resource": rng.choice(RESOURCES),
            "role": rng.choice(ROLES),
        }
        key = (row["user_id"], row["resource"], row["role"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def _set_roles_on(
    rows: list[dict[str, str]],
    *,
    user_id: str,
    resource: str,
    roles: list[str],
) -> list[dict[str, str]]:
    """Force exactly `roles` for (user, resource), leaving other rows alone."""
    kept = [
        row
        for row in rows
        if not (row["user_id"] == user_id and row["resource"] == resource)
    ]
    kept.extend(
        {"user_id": user_id, "resource": resource, "role": role} for role in roles
    )
    return kept


def _apply_ops(state: dict[str, Any], ops: list[dict[str, str]]) -> dict[str, Any]:
    roles = {
        (row["user_id"], row["resource"], row["role"]) for row in state["roles"]
    }
    credentials = set(state["credentials"])
    audit = list(state["audit"])
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
    return {
        "users": list(state["users"]),
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
    a_level: str,
    display_name: str,
    resource: str,
    role: str,
    other_display: str,
    other_resource: str,
) -> str:
    if category == "grant_access":
        if a_level == "easy":
            return f"Grant {display_name} {role} access on {resource}."
        return (
            f"{display_name} is rotating onto the platform team this week and "
            f"{ROLE_TASK_PHRASE[role]} {resource}. The separate request from "
            f"{other_display} about {other_resource} was already handled, so do "
            "not act on that one."
        )
    if category == "revoke_access":
        if a_level == "easy":
            return f"Revoke all of {display_name}'s access on {resource}."
        return (
            f"{display_name}'s contract ended on Friday, so they should not "
            f"retain anything on {resource} any more. Their {other_resource} "
            "access is owned by a different team and must stay exactly as it is."
        )
    if a_level == "easy":
        return (
            f"Rotate the credential {display_name} uses for {resource}; "
            "leave role assignments unchanged."
        )
    return (
        f"The secret {display_name} uses for {resource} was posted in a public "
        f"channel, so replace it. Their roles on {resource} and their "
        f"{other_resource} setup are both fine and must not change."
    )


def _build_instance(
    *,
    rng: random.Random,
    seed: int,
    index: int,
    a_level: str,
    b_level: str,
    category: str,
) -> AccessInstance:
    shape = CELL_SHAPE[b_level]
    lookalike_slots = 3 if b_level == "hard" else 0
    users = _user_pool(rng, shape["users"] - lookalike_slots)
    target = rng.choice(users)
    target_id = target["user_id"]
    display_name = target["display_name"]

    lookalikes: list[dict[str, str]] = []
    if lookalike_slots:
        taken = {user["display_name"] for user in users}
        for name in _lookalike_names(display_name, taken, rng):
            lookalikes.append(
                {"user_id": f"u{len(users) + len(lookalikes):03d}", "display_name": name}
            )
        users = users + lookalikes

    resource = rng.choice(RESOURCES)
    other_resource = rng.choice([item for item in RESOURCES if item != resource])
    other_user = rng.choice([user for user in users if user["user_id"] != target_id])

    rows = _assignments(rng, users, shape["assignments"])

    if category == "revoke_access":
        held = rng.sample(list(ROLES), rng.choice(shape["revoke_roles"]))
        rows = _set_roles_on(rows, user_id=target_id, resource=resource, roles=held)
        expected_ops = [
            {"op": "revoke_role", "user_id": target_id, "resource": resource, "role": role}
            for role in sorted(held)
        ]
        target_role = ""
    elif category == "grant_access":
        allowed = [role for role in ROLES]
        if resource in PROTECTED_RESOURCES:
            allowed = [role for role in allowed if role != FORBIDDEN_ROLE_ON_PROTECTED]
        target_role = rng.choice(allowed)
        # A grant that is already satisfied would leave the state unchanged and
        # make the commit check ambiguous, so clear that exact row.
        rows = [
            row
            for row in rows
            if not (
                row["user_id"] == target_id
                and row["resource"] == resource
                and row["role"] == target_role
            )
        ]
        expected_ops = [
            {
                "op": "grant_role",
                "user_id": target_id,
                "resource": resource,
                "role": target_role,
            }
        ]
    else:
        target_role = ""
        expected_ops = [
            {"op": "rotate_credential", "user_id": target_id, "resource": resource}
        ]

    # Look-alikes hold roles on the same resource, so resolving the wrong name
    # produces a different op set rather than an accidentally identical one.
    for extra in lookalikes:
        rows = _set_roles_on(
            rows,
            user_id=extra["user_id"],
            resource=resource,
            roles=[rng.choice(ROLES)],
        )

    initial_state = {
        "users": users,
        "roles": sorted(
            rows, key=lambda row: (row["user_id"], row["resource"], row["role"])
        ),
        "credentials": [f"active:{target_id}:{resource}"],
        "audit": [],
    }

    policy = {
        "protected_resources": list(PROTECTED_RESOURCES),
        "forbidden_role_on_protected": FORBIDDEN_ROLE_ON_PROTECTED,
    }

    # Two violation classes with different detectability. A rule checker sees
    # the policy breach but cannot see that the wrong person was targeted.
    wrong_subject = lookalikes[0]["user_id"] if lookalikes else other_user["user_id"]
    invalid_plans = [
        {
            "ops": [{**op, "user_id": wrong_subject} for op in expected_ops],
            "violation": "wrong_subject",
            "rule_detectable": False,
        },
        {
            "ops": [
                {
                    "op": "grant_role",
                    "user_id": target_id,
                    "resource": PROTECTED_RESOURCES[0],
                    "role": FORBIDDEN_ROLE_ON_PROTECTED,
                }
            ],
            "violation": "policy_forbidden_role",
            "rule_detectable": True,
        },
    ]

    return AccessInstance(
        instance_id=f"{a_level}_{b_level}_{seed}_{index:04d}",
        a_level=a_level,
        b_level=b_level,
        request=_request_text(
            category=category,
            a_level=a_level,
            display_name=display_name,
            resource=resource,
            role=target_role or rng.choice(ROLES),
            other_display=other_user["display_name"],
            other_resource=other_resource,
        ),
        initial_state=initial_state,
        classification={
            "category": category,
            "subject": display_name,
            "resource": resource,
            "role": target_role,
            "confidence": 1.0,
        },
        expected_ops=expected_ops,
        expected_final_state=_apply_ops(initial_state, expected_ops),
        recoverability=CATEGORY_TO_RECOVERABILITY[category],
        policy=policy,
        invalid_plans=invalid_plans,
    )


def generate_instances(
    *,
    seed: int,
    count: int,
    a_level: str,
    b_level: str,
) -> list[AccessInstance]:
    if a_level not in CELL_SHAPE:
        raise ValueError("a_level must be easy or hard")
    if b_level not in CELL_SHAPE:
        raise ValueError("b_level must be easy or hard")

    # Seeded on b_level only, deliberately. `a_level` changes nothing but the
    # wording of the request, so seeding on it too would redraw the users,
    # resources, records and target as well, and an A:easy-vs-A:hard comparison
    # would confound the wording change with a fresh sample. Excluding it makes
    # the A axis a *paired* comparison: the two cells are the same instances
    # asked two ways, which is both fairer and far lower variance at small n.
    # `b_level` must stay in the seed -- it changes the record set by design.
    rng = random.Random(f"{seed}:{b_level}")
    categories = list(CATEGORY_TO_OP)
    instances = [
        _build_instance(
            rng=rng,
            seed=seed,
            index=index,
            a_level=a_level,
            b_level=b_level,
            category=categories[index % len(categories)],
        )
        for index in range(count)
    ]
    assert all(item.recoverability in RECOVERABILITY for item in instances)
    return instances
