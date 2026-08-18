"""Deterministic selection of which plans get replaced by a known-bad one.

Shared by Observation 2 (verifier accuracy) and Observation 3 (commit barrier).
Both need negatives: a competent planner leaves a verifier nothing to catch, and
a verifier that never rejects never triggers compensation, so neither experiment
would have any content on the natural distribution alone.
"""

from __future__ import annotations

import random
from typing import Any


INJECTION_CLASSES = ("natural", "wrong_subject", "policy_forbidden_role")


BAD_PLAN_CLASSES = ("wrong_subject", "policy_forbidden_role")


def injection_for(
    instance: Any,
    rate: float,
    classes: tuple[str, ...] = BAD_PLAN_CLASSES,
) -> str:
    """Pick this instance's plan source deterministically.

    Seeded from the instance id rather than its index so the choice does not
    correlate with the request category, which cycles by index.

    `classes` narrows which kind of bad plan may be injected. Observation 3 uses
    `wrong_subject` only, because `policy_forbidden_role` swaps a credential
    rotation for a role grant -- the executed operation would then be reversible
    while the cell is labelled irreversible, and the cell would no longer test
    what it is named after.
    """
    if rate <= 0:
        return "natural"
    rng = random.Random(f"inject:{instance.instance_id}")
    if rng.random() >= rate:
        return "natural"
    return rng.choice(list(classes))


def injected_ops_for(instance: Any, violation: str) -> list[dict[str, str]] | None:
    if violation == "natural":
        return None
    for plan in instance.invalid_plans:
        if plan["violation"] == violation:
            return list(plan["ops"])
    raise ValueError(f"instance has no invalid plan for {violation}")
