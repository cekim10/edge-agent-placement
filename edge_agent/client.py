"""OpenAI-compatible client for vLLM endpoints plus a deterministic mock."""

from __future__ import annotations

import http.client
import json
import os
import socket
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
    api_kind: str = "chat"


DEFAULT_ENDPOINTS = {
    "edge": Endpoint(
        tier="edge",
        base_url=os.environ.get("EDGE_BASE_URL", "http://elves-01:8001/v1"),
        model=os.environ.get("EDGE_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
        api_kind=os.environ.get("EDGE_API_KIND", os.environ.get("API_KIND", "chat")),
    ),
    "cloud": Endpoint(
        tier="cloud",
        base_url=os.environ.get("CLOUD_BASE_URL", "http://elves-02:8002/v1"),
        model=os.environ.get("CLOUD_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        api_kind=os.environ.get("CLOUD_API_KIND", os.environ.get("API_KIND", "chat")),
    ),
}


def mss_clamp() -> int:
    """Outgoing TCP segment cap, or 0 to leave the kernel default alone.

    The cluster NICs advertise a 9000-byte MTU while the switch between the
    nodes only forwards 1500-byte frames, so any request that does not fit in a
    single segment is silently dropped and the connection hangs until it times
    out. Lowering TCP_MAXSEG before connect is the only fix available without
    root. Set TCP_MSS_CLAMP=0 once the network itself is corrected.
    """
    try:
        value = int(os.environ.get("TCP_MSS_CLAMP", "1400").strip())
    except ValueError:
        return 0
    return value if value > 0 else 0


class _ClampedHTTPConnection(http.client.HTTPConnection):
    mss: int = 0

    def connect(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if self.mss:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, self.mss)
        if isinstance(self.timeout, (int, float)):
            sock.settimeout(self.timeout)
        if self.source_address:
            sock.bind(self.source_address)
        sock.connect((self.host, self.port))
        self.sock = sock


class _ClampedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, mss: int) -> None:
        super().__init__()
        self._mss = mss

    def http_open(self, req: Any) -> Any:
        def factory(*args: Any, **kwargs: Any) -> _ClampedHTTPConnection:
            connection = _ClampedHTTPConnection(*args, **kwargs)
            connection.mss = self._mss
            return connection

        return self.do_open(factory, req)


class ChatClient(Protocol):
    def chat(
        self,
        *,
        tier: str,
        stage: str,
        messages: list[dict[str, str]],
        incident: dict[str, Any] | None = None,
        guided_json: dict[str, Any] | None = None,
    ) -> str:
        ...


class VLLMClient:
    def __init__(
        self,
        endpoints: dict[str, Endpoint] | None = None,
        timeout_s: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 256,
        max_tokens_by_stage: dict[str, int] | None = None,
    ) -> None:
        self.endpoints = endpoints or DEFAULT_ENDPOINTS
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tokens_by_stage = max_tokens_by_stage or {}
        self._tokenizers: dict[str, Any] = {}
        self.mss = mss_clamp()
        self._opener = (
            urllib.request.build_opener(_ClampedHTTPHandler(self.mss))
            if self.mss
            else urllib.request.build_opener()
        )
        if self.mss:
            print(f"[client] TCP_MAXSEG clamped to {self.mss} bytes", flush=True)

    def _tokenizer_for_model(self, model: str) -> Any:
        tokenizer = self._tokenizers.get(model)
        if tokenizer is not None:
            return tokenizer
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "CLOUD_API_KIND/EDGE_API_KIND=completions requires transformers so the client can "
                "apply the model chat template before calling /v1/completions. Install with: "
                "uv pip install transformers"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(model)
        self._tokenizers[model] = tokenizer
        return tokenizer

    def _prompt_from_messages(self, endpoint: Endpoint, messages: list[dict[str, str]]) -> str:
        tokenizer = self._tokenizer_for_model(endpoint.model)
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to apply chat template for {endpoint.model}: {exc}") from exc

    def _payload_for_request(
        self,
        *,
        endpoint: Endpoint,
        messages: list[dict[str, str]],
        request_max_tokens: int,
        guided_json: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any], int | None]:
        api_kind = endpoint.api_kind.strip().lower()
        if api_kind in {"chat", "chat_completions", "chat/completions"}:
            payload = {
                "model": endpoint.model,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": 1.0,
                "max_tokens": request_max_tokens,
            }
            if guided_json is not None:
                payload["guided_json"] = guided_json
            return (
                endpoint.base_url.rstrip("/") + "/chat/completions",
                payload,
                None,
            )
        if api_kind in {"completion", "completions", "text"}:
            prompt = self._prompt_from_messages(endpoint, messages)
            tokenizer = self._tokenizer_for_model(endpoint.model)
            prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
            payload = {
                "model": endpoint.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "top_p": 1.0,
                "max_tokens": request_max_tokens,
            }
            if guided_json is not None:
                payload["guided_json"] = guided_json
            return (
                endpoint.base_url.rstrip("/") + "/completions",
                payload,
                prompt_tokens,
            )
        raise ValueError(f"Unsupported api_kind for {endpoint.tier}: {endpoint.api_kind!r}")

    def chat(
        self,
        *,
        tier: str,
        stage: str,
        messages: list[dict[str, str]],
        incident: dict[str, Any] | None = None,
        guided_json: dict[str, Any] | None = None,
    ) -> str:
        del incident
        endpoint = self.endpoints[tier]
        request_max_tokens = self.max_tokens_by_stage.get(stage, self.max_tokens)
        url, payload, prompt_tokens = self._payload_for_request(
            endpoint=endpoint,
            messages=messages,
            request_max_tokens=request_max_tokens,
            guided_json=guided_json,
        )
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except socket.timeout as exc:
            raise RuntimeError(
                f"Timed out calling {tier} {stage} endpoint {url} after {self.timeout_s:.1f}s "
                f"with max_tokens={request_max_tokens}"
                f"{f' prompt_tokens={prompt_tokens}' if prompt_tokens is not None else ''}"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Timed out calling {tier} {stage} endpoint {url} after {self.timeout_s:.1f}s "
                f"with max_tokens={request_max_tokens}"
                f"{f' prompt_tokens={prompt_tokens}' if prompt_tokens is not None else ''}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to call {tier} {stage} endpoint {url}: {exc}") from exc
        elapsed = time.perf_counter() - started
        if not body.get("choices"):
            raise RuntimeError(f"Empty response from {tier} endpoint after {elapsed:.2f}s")
        choice = body["choices"][0]
        if "message" in choice:
            return choice["message"]["content"]
        return choice.get("text", "")


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
        guided_json: dict[str, Any] | None = None,
    ) -> str:
        del messages, guided_json
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


def build_client(
    mock: bool = False,
    timeout_s: float = 120.0,
    max_tokens: int = 256,
    max_tokens_by_stage: dict[str, int] | None = None,
) -> ChatClient:
    return MockClient() if mock else VLLMClient(
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        max_tokens_by_stage=max_tokens_by_stage,
    )
