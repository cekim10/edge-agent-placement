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

MAX_OBJECTIVE_CHARS = 1200
MAX_PRIOR_CHARS = 800
MAX_PRIOR_STAGE_CHARS = 220


MITRE_TECHNIQUE_NAMES = {
    "T1003": "OS Credential Dumping",
    "T1005": "Data from Local System",
    "T1041": "Exfiltration Over C2 Channel",
    "T1046": "Network Service Discovery",
    "T1053": "Scheduled Task/Job",
    "T1059": "Command and Scripting Interpreter",
    "T1068": "Exploitation for Privilege Escalation",
    "T1069": "Permission Groups Discovery",
    "T1070": "Indicator Removal",
    "T1078": "Valid Accounts",
    "T1082": "System Information Discovery",
    "T1083": "File and Directory Discovery",
    "T1098": "Account Manipulation",
    "T1110": "Brute Force",
    "T1136": "Create Account",
    "T1210": "Exploitation of Remote Services",
    "T1484": "Domain or Tenant Policy Modification",
    "T1496": "Resource Hijacking",
    "T1525": "Implant Internal Image",
    "T1530": "Data from Cloud Storage",
    "T1537": "Transfer Data to Cloud Account",
    "T1543": "Create or Modify System Process",
    "T1547": "Boot or Logon Autostart Execution",
    "T1548": "Abuse Elevation Control Mechanism",
    "T1552": "Unsecured Credentials",
    "T1555": "Credentials from Password Stores",
    "T1562": "Impair Defenses",
    "T1578": "Modify Cloud Compute Infrastructure",
    "T1609": "Container and Resource Discovery",
    "T1610": "Deploy Container",
    "T1611": "Escape to Host",
    "T1613": "Container and Resource Discovery",
    "T1619": "Cloud Storage Object Discovery",
    "T1649": "Steal or Forge Authentication Certificates",
    "T1651": "Cloud Administration Command",
    "T1654": "Log Enumeration",
}


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


def build_data_source_catalog(records: list[CTIRecord]) -> list[str]:
    seen = set()
    catalog = []
    for record in records:
        for source in record.expected_data_sources:
            key = _normalize(source)
            if not key or key in seen:
                continue
            seen.add(key)
            catalog.append(source)
    return sorted(catalog, key=str.lower)


def build_mitre_catalog(records: list[CTIRecord]) -> list[str]:
    seen = set()
    catalog = []
    for record in records:
        for technique in record.expected_mitre_techniques:
            key = technique.upper()
            if not key or key in seen:
                continue
            seen.add(key)
            name = MITRE_TECHNIQUE_NAMES.get(key)
            catalog.append(f"{key}: {name}" if name else key)
    return sorted(catalog)


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
        "provided detection objective and prior stage outputs. Return compact JSON only. "
        "Do not use markdown fences. Do not invent telemetry that was not provided."
    )


def _prior_block(stage_outputs: list[CTIStageResult]) -> str:
    if not stage_outputs:
        return "None"
    block = "\n\n".join(
        f"{result.stage.upper()} OUTPUT:\n{_clip(result.output, MAX_PRIOR_STAGE_CHARS)}"
        for result in stage_outputs
    )
    return _clip(block, MAX_PRIOR_CHARS)


def _messages_for_stage(
    record: CTIRecord,
    stage: str,
    stage_outputs: list[CTIStageResult],
    data_source_catalog: list[str] | None = None,
    mitre_catalog: list[str] | None = None,
    use_mitre_prior: bool = False,
    use_data_source_prior: bool = False,
) -> list[dict[str, str]]:
    objective = _clip(record.detection_objective, MAX_OBJECTIVE_CHARS)
    platform = record.platform
    prior = _prior_block(stage_outputs)
    catalog = ", ".join(data_source_catalog or [])
    catalog_block = f"\nAVAILABLE_DATA_SOURCES:\n{catalog}\n" if catalog else ""
    mitre_candidates = ", ".join(mitre_catalog or [])
    mitre_block = f"\nAVAILABLE_MITRE_TECHNIQUES:\n{mitre_candidates}\n" if mitre_candidates else ""

    prompts = {
        "cti_analysis": (
            "Stage C0: CTI report analysis.\n"
            "Extract the threat behavior, affected platform, useful entities, "
            "and likely telemetry families.\n"
            "Return JSON with keys: threat_summary, entities, telemetry_hints, uncertainties."
        ),
        "mitre_mapping": (
            "Stage C1: MITRE technique mapping.\n"
            "Choose the most specific ATT&CK technique IDs only from AVAILABLE_MITRE_TECHNIQUES. "
            "If multiple candidates seem plausible, prefer the candidate that best matches the detection objective wording.\n"
            "Return compact JSON: {\\\"mitre_techniques\\\":[\\\"Txxxx\\\"]}. Do not output IDs outside the candidate list."
        ),
        "data_source_discovery": (
            "Stage C2: data-source discovery.\n"
            "Choose exact telemetry names only from AVAILABLE_DATA_SOURCES.\n"
            "Return compact JSON: {\\\"data_sources\\\":[\\\"ExactSourceName\\\"]}. Do not invent new source names."
        ),
        "kql_development": (
            "Stage C3: KQL development.\n"
            "Draft a KQL query for the detection. It should include relevant tables, filters, "
            "joins if needed, and projected evidence fields.\n"
            "Return JSON with keys kql_query, detection_logic, expected_matches."
        ),
        "rule_generation": (
            "Stage C4: final compact detection summary.\n"
            "Return only minified JSON with keys mitre_techniques, data_sources, kql_query. "
            "Keep it under 80 tokens. No markdown. No prose."
        ),
    }

    if stage == "mitre_mapping":
        mitre_prior = f"\nPRIOR_STAGE_OUTPUTS:\n{prior}" if use_mitre_prior else ""
        user = (
            "Select exactly one MITRE ATT&CK technique ID from AVAILABLE_MITRE_TECHNIQUES.\n"
            "Use the technique names to distinguish behavior: credentials, secrets, tokens, certificates, "
            "service accounts, passwords, and keys are credential-access behavior, not command execution.\n"
            "Return minified JSON only, for example {\\\"mitre_techniques\\\":[\\\"T1552\\\"]}.\n\n"
            f"PLATFORM: {platform}\n"
            f"DETECTION_OBJECTIVE:\n{objective}\n"
            f"{mitre_block}"
            f"{mitre_prior}"
        )
        return [{"role": "user", "content": user}]

    if stage == "data_source_discovery":
        data_prior = f"\nPRIOR_STAGE_OUTPUTS:\n{prior}" if use_data_source_prior else ""
        user = (
            "Select the exact telemetry source names from AVAILABLE_DATA_SOURCES.\n"
            "Return minified JSON only, for example {\\\"data_sources\\\":[\\\"ExactSourceName\\\"]}.\n\n"
            f"PLATFORM: {platform}\n"
            f"DETECTION_OBJECTIVE:\n{objective}\n"
            f"{catalog_block}"
            f"{data_prior}"
        )
        return [{"role": "user", "content": user}]

    user = (
        f"{prompts[stage]}\n\n"
        f"PLATFORM: {platform}\n"
        f"DETECTION_OBJECTIVE:\n{objective}\n"
        f"PRIOR_STAGE_OUTPUTS:\n{prior}"
    )
    return [{"role": "system", "content": _system_prompt()}, {"role": "user", "content": user}]


def _extract_prior_data_sources(text: str, data_source_catalog: list[str] | None = None) -> list[str]:
    candidates = []
    text_norm = _normalize(text)
    for source in data_source_catalog or []:
        if _normalize(source) in text_norm:
            candidates.append(source)
    candidates.extend(
        re.findall(
            r"\b[A-Z][A-Za-z0-9_]*(?:Events|Logs|Telemetry|Evidence|Records|Data|Table|Tables)\b",
            text,
        )
    )
    candidates.extend(re.findall(r"[\"']([A-Za-z][A-Za-z0-9_]*(?:Events|Logs))[\"']", text))
    ignored = {"Events", "Logs", "Telemetry", "Data", "Table", "Tables"}
    deduped = []
    seen = set()
    for candidate in candidates:
        key = _normalize(candidate)
        if candidate in ignored or not key or key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _proxy_stage_output(
    record: CTIRecord,
    stage: str,
    stage_outputs: list[CTIStageResult],
    data_source_catalog: list[str] | None = None,
) -> str:
    del record
    prior = "\n".join(result.output for result in stage_outputs)
    mitre_techniques = sorted(_mitre_ids(prior))
    data_sources = _extract_prior_data_sources(prior, data_source_catalog)
    source = data_sources[0] if data_sources else "SecurityEvent"
    if stage == "kql_development":
        return json.dumps(
            {
                "kql_query": f"{source} | take 20",
                "derived_from_prior": _clip(prior, 300),
            },
            separators=(",", ":"),
        )
    if stage == "rule_generation":
        return json.dumps(
            {
                "mitre_techniques": mitre_techniques,
                "data_sources": data_sources,
                "kql_query": f"{source} | take 20",
                "derived_from_prior": _clip(prior, 300),
            },
            separators=(",", ":"),
        )
    raise ValueError(f"cannot proxy CTI stage: {stage}")


def run_cti_workflow(
    *,
    client: ChatClient,
    record: CTIRecord,
    placement: tuple[str, ...],
    proxy_final_from_c2: bool = False,
    data_source_catalog: list[str] | None = None,
    mitre_catalog: list[str] | None = None,
    use_mitre_prior: bool = False,
    use_data_source_prior: bool = False,
) -> list[CTIStageResult]:
    if len(placement) != len(CTI_STAGES):
        raise ValueError(f"placement must have {len(CTI_STAGES)} tiers")
    incident = record.to_incident()
    results: list[CTIStageResult] = []
    for stage, tier in zip(CTI_STAGES, placement):
        if proxy_final_from_c2 and stage in {"kql_development", "rule_generation"}:
            print(
                f"  stage_proxy stage={stage} tier={tier} reason=proxy_final_from_c2",
                flush=True,
            )
            started = time.perf_counter()
            output = _proxy_stage_output(record, stage, results, data_source_catalog)
            latency_s = time.perf_counter() - started
            print(
                f"  stage_done stage={stage} tier={tier} latency_s={latency_s:.2f} output_chars={len(output)}",
                flush=True,
            )
        else:
            messages = _messages_for_stage(
                record,
                stage,
                results,
                data_source_catalog=data_source_catalog,
                mitre_catalog=mitre_catalog,
                use_mitre_prior=use_mitre_prior,
                use_data_source_prior=use_data_source_prior,
            )
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
