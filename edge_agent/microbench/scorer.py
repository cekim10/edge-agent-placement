"""Deterministic scoring for the access-control microbenchmark."""

from __future__ import annotations

from typing import Any

from .schemas import canonical_ops


def _state_key(state: dict[str, Any]) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    roles = tuple(
        sorted((row["user_id"], row["resource"], row["role"]) for row in state.get("roles", []))
    )
    credentials = tuple(sorted(state.get("credentials", [])))
    return roles, credentials


def score_instance(
    *,
    expected_classification: dict[str, Any],
    predicted_classification: dict[str, Any] | None,
    expected_ops: list[dict[str, str]],
    predicted_ops: list[dict[str, str]] | None,
    expected_final_state: dict[str, Any],
    actual_final_state: dict[str, Any] | None,
    schema_error: str | None,
    output_error: str | None,
) -> dict[str, Any]:
    expected_op_set = canonical_ops(expected_ops)
    predicted_op_set = canonical_ops(predicted_ops or [])
    intersection = expected_op_set & predicted_op_set
    precision = len(intersection) / len(predicted_op_set) if predicted_op_set else 0.0
    recall = len(intersection) / len(expected_op_set) if expected_op_set else 1.0
    classification_correct = predicted_classification is not None and all(
        predicted_classification.get(key) == expected_classification.get(key)
        for key in ("category", "user_id", "resource", "role")
    )
    plan_exact = predicted_op_set == expected_op_set
    commit_correct = (
        actual_final_state is not None
        and _state_key(actual_final_state) == _state_key(expected_final_state)
    )
    return {
        "classification_correct": classification_correct,
        "plan_exact": plan_exact,
        "plan_partial_precision": precision,
        "plan_partial_recall": recall,
        "commit_correct": commit_correct,
        "end_to_end_success": classification_correct and plan_exact and commit_correct,
        "schema_violation": schema_error is not None,
        "schema_error": schema_error,
        "no_output": output_error == "no_output",
        "output_error": output_error,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "ok_n": 0,
            "error_n": 0,
            "classification_accuracy": 0.0,
            "plan_exact_rate": 0.0,
            "commit_success_rate": 0.0,
            "end_to_end_success_rate": 0.0,
            "schema_violation_rate": 0.0,
            "plan_partial_recall": 0.0,
        }

    def mean(key: str) -> float:
        return sum(float(record["score"][key]) for record in records) / n

    return {
        "n": n,
        "ok_n": sum(1 for record in records if record.get("ok")),
        "error_n": sum(1 for record in records if not record.get("ok")),
        "classification_accuracy": mean("classification_correct"),
        "plan_exact_rate": mean("plan_exact"),
        "commit_success_rate": mean("commit_correct"),
        "end_to_end_success_rate": mean("end_to_end_success"),
        "schema_violation_rate": mean("schema_violation"),
        "plan_partial_recall": mean("plan_partial_recall"),
    }
