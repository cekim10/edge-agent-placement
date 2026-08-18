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


def injection_for(instance: Any, rate: float) -> str:
    """Pick this instance's plan source deterministically.

    Seeded from the instance id rather than its index so the choice does not
    correlate with the request category, which cycles by index.
    """
    if rate <= 0:
        return "natural"
    rng = random.Random(f"inject:{instance.instance_id}")
    if rng.random() >= rate:
        return "natural"
    return rng.choice(["wrong_subject", "policy_forbidden_role"])


def injected_ops_for(instance: Any, violation: str) -> list[dict[str, str]] | None:
    if violation == "natural":
        return None
    for plan in instance.invalid_plans:
        if plan["violation"] == violation:
            return list(plan["ops"])
    raise ValueError(f"instance has no invalid plan for {violation}")
