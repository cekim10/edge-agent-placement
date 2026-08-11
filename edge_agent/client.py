"""OpenAI-compatible client for vLLM endpoints plus a deterministic mock."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Endpoint:
    tier: str
    base_url: str
    model: str


DEFAULT_ENDPOINTS = {
    "edge": Endpoint(
        tier="edge",
        base_url=os.environ.get("EDGE_BASE_URL", "http://elves-01:8001/v1"),
        model=os.environ.get("EDGE_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
    ),
    "cloud": Endpoint(
        tier="cloud",
        base_url=os.environ.get("CLOUD_BASE_URL", "http://elves-02:8002/v1"),
        model=os.environ.get("CLOUD_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    ),
}


class ChatClient(Protocol):
    def chat(
        self,
        *,
        tier: str,
        stage: str,
        messages: list[dict[str, str]],
        incident: dict[str, Any] | None = None,
    ) -> str:
        ...


class VLLMClient:
    def __init__(
        self,
        endpoints: dict[str, Endpoint] | None = None,
        timeout_s: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> None:
        self.endpoints = endpoints or DEFAULT_ENDPOINTS
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(
        self,
        *,
        tier: str,
        stage: str,
        messages: list[dict[str, str]],
        incident: dict[str, Any] | None = None,
    ) -> str:
        del stage, incident
        endpoint = self.endpoints[tier]
        url = endpoint.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": endpoint.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": 1.0,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to call {tier} endpoint {url}: {exc}") from exc
        elapsed = time.perf_counter() - started
        if not body.get("choices"):
            raise RuntimeError(f"Empty response from {tier} endpoint after {elapsed:.2f}s")
        return body["choices"][0]["message"]["content"]


class MockClient:
    """Deterministic local client for validating scripts without GPUs.

    The mock is intentionally simple. Cloud behaves like an oracle, while edge
    drops information in stage-specific ways so sensitivity scripts produce
    non-identical outputs during local smoke tests.
    """

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
            raise ValueError("MockClient requires incident metadata")
        expected = incident["expected"]
        canonical_terms = [group[0] for group in expected["must_include"]]
        root_cause = " + ".join(canonical_terms)
        evidence = incident.get("mock_evidence", "Signals align with the expected cause.")

        if tier == "edge" and stage == "triage" and int(incident["id"].split("-")[-1]) % 3 == 0:
            canonical_terms = canonical_terms[:-1] or canonical_terms
        if tier == "edge" and stage == "diagnose" and int(incident["id"].split("-")[-1]) % 2 == 0:
            root_cause = incident["expected"]["avoid_main_cause"][0]
        if tier == "edge" and stage == "report" and int(incident["id"].split("-")[-1]) % 4 == 0:
            root_cause = canonical_terms[0]

        if stage == "triage":
            return (
                "TRIAGE_SUMMARY:\n"
                f"- likely_signals: {', '.join(canonical_terms)}\n"
                f"- distractors_to_check: {', '.join(expected['avoid_main_cause'])}\n"
            )
        if stage == "diagnose":
            return (
                "DIAGNOSIS:\n"
                f"Root cause candidate: {root_cause}\n"
                f"Evidence: {evidence}\n"
            )
        return (
            "ROOT_CAUSE:\n"
            f"{root_cause}\n\n"
            "EVIDENCE:\n"
            f"{evidence}\n\n"
            "NOT_MAIN_CAUSE:\n"
            f"{', '.join(expected['avoid_main_cause'])}\n"
        )


def build_client(mock: bool = False, timeout_s: float = 120.0) -> ChatClient:
    return MockClient() if mock else VLLMClient(timeout_s=timeout_s)
