"""Scoring utilities for incident root-cause reports.

Correctness is binary:
- every must_include synonym group must have at least one hit in the answer
- no avoid_main_cause term may appear in the root-cause section
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_HEADING_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?"
    r"(root[\s_-]*cause|main[\s_-]*cause|primary[\s_-]*cause|probable[\s_-]*cause|원인|근본\s*원인)"
    r"\s*:?\s*$"
)
_NEXT_HEADING_RE = re.compile(r"(?m)^\s*(?:#{1,6}\s*)?[A-Z가-힣][A-Z가-힣0-9 _/-]{1,40}:?\s*$")


@dataclass(frozen=True)
class ScoreResult:
    incident_id: str
    correct: bool
    accuracy: float
    partial_score: float
    must_hits: list[dict[str, Any]]
    avoid_hits: list[str]
    root_cause_section: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "partial_score": self.partial_score,
            "must_hits": self.must_hits,
            "avoid_hits": self.avoid_hits,
            "root_cause_section": self.root_cause_section,
        }


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9가-힣+./%-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains(text_norm: str, term: str) -> bool:
    term_norm = _normalize(term)
    if not term_norm:
        return False
    return f" {term_norm} " in f" {text_norm} "


def extract_root_cause_section(answer: str) -> str:
    """Return the root-cause section, falling back to the full answer.

    The fallback is intentional: if a model does not clearly separate sections,
    avoid_main_cause terms should still be able to fail the answer.
    """

    match = _HEADING_RE.search(answer)
    if not match:
        inline = re.search(
            r"(?is)(root[\s_-]*cause|main[\s_-]*cause|primary[\s_-]*cause|probable[\s_-]*cause)\s*:\s*(.+)",
            answer,
        )
        return inline.group(2).strip() if inline else answer.strip()

    start = match.end()
    rest = answer[start:]
    next_match = _NEXT_HEADING_RE.search(rest)
    end = start + next_match.start() if next_match else len(answer)
    return answer[start:end].strip()


def score_answer(incident: dict[str, Any], answer: str) -> ScoreResult:
    expected = incident.get("expected", {})
    must_groups: list[list[str]] = expected.get("must_include", [])
    avoid_terms: list[str] = expected.get("avoid_main_cause", [])

    answer_norm = _normalize(answer)
    root_cause = extract_root_cause_section(answer)
    root_norm = _normalize(root_cause)

    must_hits = []
    for group in must_groups:
        matched = [term for term in group if _contains(answer_norm, term)]
        must_hits.append(
            {
                "group": group,
                "hit": bool(matched),
                "matched": matched,
            }
        )

    avoid_hits = [term for term in avoid_terms if _contains(root_norm, term)]
    groups_hit = sum(1 for item in must_hits if item["hit"])
    partial_score = groups_hit / len(must_hits) if must_hits else 1.0
    correct = partial_score == 1.0 and not avoid_hits

    return ScoreResult(
        incident_id=str(incident.get("id", "")),
        correct=correct,
        accuracy=1.0 if correct else 0.0,
        partial_score=partial_score,
        must_hits=must_hits,
        avoid_hits=avoid_hits,
        root_cause_section=root_cause,
    )


def aggregate_scores(scores: list[ScoreResult]) -> dict[str, Any]:
    if not scores:
        return {
            "n": 0,
            "accuracy": 0.0,
            "partial_score": 0.0,
            "avoid_rate": 0.0,
        }

    return {
        "n": len(scores),
        "accuracy": sum(score.accuracy for score in scores) / len(scores),
        "partial_score": sum(score.partial_score for score in scores) / len(scores),
        "avoid_rate": sum(1 for score in scores if score.avoid_hits) / len(scores),
    }
