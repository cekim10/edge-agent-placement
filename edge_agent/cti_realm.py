"""CTI-REALM Observation 1 characterization harness.

This module intentionally avoids vendoring ACESEvals. It consumes the CTI-REALM
JSONL files downloaded by ACESEvals and runs a deterministic five-stage workflow
with edge/cloud model routing controlled per stage.

The score is a lightweight ground-truth proxy for early characterization, not
the official CTI-REALM C4 scorer. Use it to decide whether stage sensitivity is
worth pursuing, then validate strong results with the ACESEvals Docker/Kusto
environment and official checkpoint scorers.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import ChatClient


CTI_STAGES = (
    "cti_analysis",
    "mitre_mapping",
    "data_source_discovery",
    "kql_development",
    "rule_generation",
)

STAGE_LABELS = {
    "cti_analysis": "CTI analysis",
    "mitre_mapping": "MITRE mapping",
    "data_source_discovery": "Data-source discovery",
    "kql_development": "KQL development",
    "rule_generation": "Rule generation",
}

MAX_OBJECTIVE_CHARS = 1600
MAX_PRIOR_CHARS = 1200
MAX_PRIOR_STAGE_CHARS = 320


@dataclass(frozen=True)
class CTIRecord:
    task_id: str
    platform: str
    detection_objective: str
    expected_mitre_techniques: list[str]
    expected_data_sources: list[str]
    regex_patterns: dict[str, Any]

    def to_incident(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "platform": self.platform,
            "detection_objective": self.detection_objective,
            "expected_mitre_techniques": self.expected_mitre_techniques,
            "expected_data_sources": self.expected_data_sources,
            "regex_patterns": self.regex_patterns,
        }


@dataclass(frozen=True)
class CTIStageResult:
    stage: str
    tier: str
    latency_s: float
    output: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "tier": self.tier,
            "latency_s": self.latency_s,
            "output": self.output,
        }


def _platform_from_id(task_id: str) -> str:
    prefix = task_id.split("_", 1)[0]
    return {"linux": "Linux", "aks": "AKS", "cloud": "Cloud"}.get(prefix, prefix.capitalize())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_cti_realm_records(data_dir: Path, size: int, limit: int = 0) -> list[CTIRecord]:
    samples_path = data_dir / f"dataset_samples_stratified_{size}.jsonl"
    answers_path = data_dir / f"dataset_answers_stratified_{size}.jsonl"
    if not samples_path.exists() or not answers_path.exists():
        raise FileNotFoundError(
            "Missing CTI-REALM JSONL files. Run ACESEvals once first, or pass "
            "--cti-data-dir pointing at domains/cti_realm/data. Expected files: "
            f"{samples_path.name}, {answers_path.name}"
        )

    samples = {record["id"]: record for record in _load_jsonl(samples_path)}
    answers = {record["id"]: record for record in _load_jsonl(answers_path)}
    if set(samples) != set(answers):
        missing = sorted(set(samples) ^ set(answers))
        raise ValueError(f"CTI sample/answer IDs do not match: {missing[:10]}")

    records: list[CTIRecord] = []
    for task_id in sorted(samples):
        sample = samples[task_id]
        answer = answers[task_id]
        ground_truth = sample.get("ground_truth", {})
        records.append(
            CTIRecord(
                task_id=task_id,
                platform=_platform_from_id(task_id),
                detection_objective=answer.get("detection_description")
                or sample.get("detection_description")
                or sample.get("objective")
                or "",
                expected_mitre_techniques=list(ground_truth.get("mitre_techniques", [])),
                expected_data_sources=list(ground_truth.get("data_sources", [])),
                regex_patterns=dict(ground_truth.get("regex_patterns", {})),
            )
        )
    return records[:limit] if limit else records


def placement_for_edge_stage(edge_stage: str | None) -> tuple[str, ...]:
    if edge_stage is None or edge_stage == "all_cloud":
        return tuple("cloud" for _ in CTI_STAGES)
    if edge_stage not in CTI_STAGES:
        raise ValueError(f"unknown CTI stage: {edge_stage}")
    return tuple("edge" if stage == edge_stage else "cloud" for stage in CTI_STAGES)


def placement_name(placement: tuple[str, ...]) -> str:
    if all(tier == "cloud" for tier in placement):
        return "all_cloud"
    edge = [stage for stage, tier in zip(CTI_STAGES, placement) if tier == "edge"]
    return "edge_" + "_".join(edge)


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[TRUNCATED]"


def _system_prompt() -> str:
    return (
        "You are a deterministic security detection engineer. Use only the "
        "provided detection objective and prior stage outputs. Return compact JSON. "
        "Do not invent telemetry that was not provided."
    )


def _prior_block(stage_outputs: list[CTIStageResult]) -> str:
    if not stage_outputs:
        return "None"
    block = "\n\n".join(
        f"{result.stage.upper()} OUTPUT:\n{_clip(result.output, MAX_PRIOR_STAGE_CHARS)}"
        for result in stage_outputs
    )
    return _clip(block, MAX_PRIOR_CHARS)


def _messages_for_stage(record: CTIRecord, stage: str, stage_outputs: list[CTIStageResult]) -> list[dict[str, str]]:
    objective = _clip(record.detection_objective, MAX_OBJECTIVE_CHARS)
    platform = record.platform
    prior = _prior_block(stage_outputs)

    prompts = {
        "cti_analysis": (
            "Stage C0: CTI report analysis.\n"
            "Extract the threat behavior, affected platform, useful entities, "
            "and likely telemetry families.\n"
            "Return JSON with keys: threat_summary, entities, telemetry_hints, uncertainties."
        ),
        "mitre_mapping": (
            "Stage C1: MITRE technique mapping.\n"
            "Map the behavior to likely MITRE ATT&CK technique IDs and names.\n"
            "Return JSON with key mitre_techniques as a list of objects with id, name, rationale."
        ),
        "data_source_discovery": (
            "Stage C2: data-source discovery.\n"
            "Identify telemetry tables or data sources needed to detect this behavior.\n"
            "Return JSON with key data_sources as a list and explain why each source is needed."
        ),
        "kql_development": (
            "Stage C3: KQL development.\n"
            "Draft a KQL query for the detection. It should include relevant tables, filters, "
            "joins if needed, and projected evidence fields.\n"
            "Return JSON with keys kql_query, detection_logic, expected_matches."
        ),
        "rule_generation": (
            "Stage C4: detection rule generation.\n"
            "Produce the final detection artifact. Include MITRE technique IDs, data sources, "
            "a KQL query, and a Sigma-style rule summary.\n"
            "Return JSON with keys mitre_techniques, data_sources, kql_query, sigma_rule, rationale."
        ),
    }

    user = (
        f"{prompts[stage]}\n\n"
        f"PLATFORM: {platform}\n"
        f"DETECTION_OBJECTIVE:\n{objective}\n\n"
        f"PRIOR_STAGE_OUTPUTS:\n{prior}"
    )
    return [{"role": "system", "content": _system_prompt()}, {"role": "user", "content": user}]


def run_cti_workflow(
    *,
    client: ChatClient,
    record: CTIRecord,
    placement: tuple[str, ...],
) -> list[CTIStageResult]:
    if len(placement) != len(CTI_STAGES):
        raise ValueError(f"placement must have {len(CTI_STAGES)} tiers")
    incident = record.to_incident()
    results: list[CTIStageResult] = []
    for stage, tier in zip(CTI_STAGES, placement):
        messages = _messages_for_stage(record, stage, results)
        prompt_chars = sum(len(message["content"]) for message in messages)
        print(
            f"  stage_start stage={stage} tier={tier} prompt_chars={prompt_chars}",
            flush=True,
        )
        started = time.perf_counter()
        output = client.chat(
            tier=tier,
            stage=stage,
            messages=messages,
            incident=incident,
        )
        latency_s = time.perf_counter() - started
        print(
            f"  stage_done stage={stage} tier={tier} latency_s={latency_s:.2f} output_chars={len(output)}",
            flush=True,
        )
        results.append(
            CTIStageResult(
                stage=stage,
                tier=tier,
                latency_s=latency_s,
                output=output,
            )
        )
    return results


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9_.:-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _mitre_ids(text: str) -> set[str]:
    return {match.upper() for match in re.findall(r"\bT\d{4}(?:\.\d{3})?\b", text, flags=re.I)}


def _jaccard(expected: set[str], predicted: set[str]) -> float:
    if not expected:
        return 1.0
    if not predicted:
        return 0.0
    return len(expected & predicted) / len(expected | predicted)


def _contains_source(text_norm: str, source: str) -> bool:
    source_norm = _normalize(source)
    if not source_norm:
        return False
    return source_norm in text_norm


def _regex_terms(regex_patterns: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for key, value in regex_patterns.items():
        if isinstance(key, str):
            terms.append(key)
        if isinstance(value, str):
            terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9_./:-]{3,}", value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9_./:-]{3,}", item))
    ignored = {"true", "false", "null", "where", "project", "extend", "regex"}
    deduped = []
    seen = set()
    for term in terms:
        term_norm = _normalize(term)
        if len(term_norm) < 4 or term_norm in ignored or term_norm in seen:
            continue
        seen.add(term_norm)
        deduped.append(term)
    return deduped[:20]


def score_cti_proxy(record: CTIRecord, stage_results: list[CTIStageResult]) -> dict[str, Any]:
    final = stage_results[-1].output if stage_results else ""
    all_text = "\n".join(result.output for result in stage_results)
    final_norm = _normalize(final)
    all_norm = _normalize(all_text)

    expected_mitre = {item.upper() for item in record.expected_mitre_techniques}
    predicted_mitre = _mitre_ids(final) | _mitre_ids(all_text)
    mitre_jaccard = _jaccard(expected_mitre, predicted_mitre)

    data_hits = [
        source for source in record.expected_data_sources if _contains_source(final_norm, source) or _contains_source(all_norm, source)
    ]
    data_source_recall = len(data_hits) / len(record.expected_data_sources) if record.expected_data_sources else 1.0

    terms = _regex_terms(record.regex_patterns)
    term_hits = [term for term in terms if _normalize(term) in final_norm]
    detection_proxy = len(term_hits) / len(terms) if terms else (0.5 * mitre_jaccard + 0.5 * data_source_recall)

    proxy_quality = 0.30 * mitre_jaccard + 0.30 * data_source_recall + 0.40 * detection_proxy
    return {
        "task_id": record.task_id,
        "proxy_quality": proxy_quality,
        "mitre_jaccard": mitre_jaccard,
        "data_source_recall": data_source_recall,
        "detection_term_recall": detection_proxy,
        "expected_mitre_techniques": sorted(expected_mitre),
        "predicted_mitre_techniques": sorted(predicted_mitre),
        "expected_data_sources": record.expected_data_sources,
        "data_source_hits": data_hits,
        "detection_terms": terms,
        "detection_term_hits": term_hits,
    }


class CTIMockClient:
    """Deterministic local client for script validation without vLLM."""

    def chat(
        self,
        *,
        tier: str,
        stage: str,
        messages: list[dict[str, str]],
        incident: dict[str, Any] | None = None,
    ) -> str:
        del messages
        if incident is None:
            raise ValueError("CTIMockClient requires incident metadata")
        mitre = list(incident["expected_mitre_techniques"])
        sources = list(incident["expected_data_sources"])
        if tier == "edge" and stage in {"mitre_mapping", "rule_generation"} and len(mitre) > 1:
            mitre = mitre[:-1]
        if tier == "edge" and stage in {"data_source_discovery", "rule_generation"} and len(sources) > 1:
            sources = sources[:-1]

        if stage == "cti_analysis":
            return json.dumps(
                {
                    "threat_summary": incident["detection_objective"][:240],
                    "telemetry_hints": sources,
                }
            )
        if stage == "mitre_mapping":
            return json.dumps({"mitre_techniques": [{"id": item} for item in mitre]})
        if stage == "data_source_discovery":
            return json.dumps({"data_sources": sources})
        if stage == "kql_development":
            table = sources[0] if sources else "SecurityEvent"
            return json.dumps({"kql_query": f"{table} | take 10", "mitre_techniques": mitre})
        return json.dumps(
            {
                "mitre_techniques": mitre,
                "data_sources": sources,
                "kql_query": (sources[0] if sources else "SecurityEvent") + " | take 10",
                "sigma_rule": {"title": incident["id"], "tags": mitre},
            }
        )
