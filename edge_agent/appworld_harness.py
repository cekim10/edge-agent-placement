"""AppWorld Observation 1 harness.

This module keeps AppWorld as an optional runtime dependency. Local py_compile
and mock runs work without installing AppWorld; real runs import AppWorld only
inside the adapter functions.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import ChatClient


APPWORLD_STAGES = ("task_analysis", "api_planning", "code_generation", "execution_verification")

MAX_INSTRUCTION_CHARS = 650
MAX_APP_DESCRIPTIONS_CHARS = 520
MAX_STAGE_OUTPUT_CHARS = 520
MAX_EXECUTION_OUTPUT_CHARS = 1000
MAX_EVALUATION_CHARS = 1000
MAX_API_DOCS_CHARS = 260


@dataclass(frozen=True)
class AppWorldTaskInfo:
    task_id: str
    dataset_name: str
    instruction: str
    supervisor: dict[str, Any]
    app_descriptions: dict[str, Any]
    api_docs_preview: str = ""

    def to_incident(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "dataset_name": self.dataset_name,
            "instruction": self.instruction,
        }


@dataclass(frozen=True)
class AppWorldStageResult:
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


@dataclass
class AppWorldTaskResult:
    task_id: str
    placement: tuple[str, ...]
    ok: bool
    error: str
    stages: list[AppWorldStageResult]
    generated_code: str
    repair_code: str
    execution_outputs: list[dict[str, Any]]
    evaluation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "placement": list(self.placement),
            "ok": self.ok,
            "error": self.error,
            "stages": [stage.to_dict() for stage in self.stages],
            "generated_code": self.generated_code,
            "repair_code": self.repair_code,
            "execution_outputs": self.execution_outputs,
            "evaluation": self.evaluation,
        }


def _clip(text: Any, max_chars: int) -> str:
    value = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str)
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "\n[TRUNCATED]"


def _json_loads_loose(text: str) -> Any:
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    candidates = fenced + [text]
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        start = candidate.find("{")
        end = candidate.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}


def _extract_code(text: str) -> str:
    fenced = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.S | re.I)
    candidate = fenced[-1].strip() if fenced else text.strip()
    if candidate.lower().startswith("python\n"):
        candidate = candidate.split("\n", 1)[1]
    return candidate.strip()


def _summarize_app_descriptions(app_descriptions: dict[str, Any]) -> str:
    lines = []
    for name, description in sorted(app_descriptions.items()):
        lines.append(f"- {name}: {_clip(description, 80)}")
    return _clip("\n".join(lines), MAX_APP_DESCRIPTIONS_CHARS)


def _selected_apps_from_results(results: list[AppWorldStageResult]) -> list[str]:
    apps: list[str] = []
    for result in results:
        parsed = _json_loads_loose(result.output)
        if not isinstance(parsed, dict):
            continue
        for key in ("selected_apps", "relevant_apps"):
            value = parsed.get(key, [])
            if isinstance(value, str):
                value = [value]
            if isinstance(value, list):
                apps.extend(str(item).strip() for item in value if str(item).strip())
    return list(dict.fromkeys(apps))[:4]


def _compact_prior(results: list[AppWorldStageResult]) -> str:
    compact: list[dict[str, Any]] = []
    for result in results:
        parsed = _json_loads_loose(result.output)
        if isinstance(parsed, dict):
            item = {
                key: parsed[key]
                for key in ("relevant_apps", "selected_apps", "api_needs", "execution_plan", "plan")
                if key in parsed
            }
            compact.append({"stage": result.stage, **item})
        else:
            compact.append({"stage": result.stage, "output": _clip(result.output, 160)})
    return _clip(compact, MAX_STAGE_OUTPUT_CHARS)


def _api_docs_preview(task: Any, selected_apps: list[str]) -> str:
    api_docs = getattr(task, "api_docs", None)
    if api_docs is None:
        return ""
    try:
        if hasattr(api_docs, "compress_parameters"):
            api_docs = api_docs.compress_parameters()
        if hasattr(api_docs, "remove_fields"):
            api_docs = api_docs.remove_fields(["path", "method"])
    except Exception:
        pass

    chunks = []
    for app_name in selected_apps[:4]:
        try:
            doc = getattr(api_docs, app_name)
        except Exception:
            continue
        chunks.append(f"{app_name} APIs:\n{_clip(doc, 130)}")
    return _clip("\n\n".join(chunks), MAX_API_DOCS_CHARS)


def _task_info_from_world(task_id: str, dataset_name: str, world: Any, results: list[AppWorldStageResult] | None = None) -> AppWorldTaskInfo:
    task = world.task
    selected_apps = _selected_apps_from_results(results or [])
    return AppWorldTaskInfo(
        task_id=task_id,
        dataset_name=dataset_name,
        instruction=str(getattr(task, "instruction", "")),
        supervisor=dict(getattr(task, "supervisor", {}) or {}),
        app_descriptions=dict(getattr(task, "app_descriptions", {}) or {}),
        api_docs_preview=_api_docs_preview(task, selected_apps),
    )


def import_appworld() -> tuple[Any, Any, Any]:
    try:
        from appworld import AppWorld, load_task_ids, update_root
    except ImportError as exc:
        raise RuntimeError(
            "AppWorld is not installed. On the GPU server run: "
            "uv pip install appworld && uv run appworld install && uv run appworld download data"
        ) from exc
    return AppWorld, load_task_ids, update_root


def load_appworld_task_ids(
    *,
    dataset_name: str,
    limit: int = 0,
    task_ids: list[str] | None = None,
    appworld_root: Path | None = None,
) -> list[str]:
    if task_ids:
        return task_ids[:limit] if limit else task_ids
    _AppWorld, load_task_ids, update_root = import_appworld()
    if appworld_root is not None:
        update_root(str(appworld_root))
    loaded = list(load_task_ids(dataset_name))
    return loaded[:limit] if limit else loaded


def placement_for_edge_appworld_stage(edge_stage: str | None) -> tuple[str, ...]:
    if edge_stage is None or edge_stage == "all_cloud":
        return tuple("cloud" for _ in APPWORLD_STAGES)
    if edge_stage == "all_edge":
        return tuple("edge" for _ in APPWORLD_STAGES)
    if edge_stage not in APPWORLD_STAGES:
        raise ValueError(f"unknown AppWorld stage: {edge_stage}")
    return tuple("edge" if stage == edge_stage else "cloud" for stage in APPWORLD_STAGES)


def placement_name(placement: tuple[str, ...]) -> str:
    if all(tier == "cloud" for tier in placement):
        return "all_cloud"
    if all(tier == "edge" for tier in placement):
        return "all_edge"
    edge = [stage for stage, tier in zip(APPWORLD_STAGES, placement) if tier == "edge"]
    return "edge_" + "_".join(edge)


def messages_for_appworld_stage(
    *,
    task_info: AppWorldTaskInfo,
    stage: str,
    results: list[AppWorldStageResult],
    execution_outputs: list[dict[str, Any]],
    evaluation: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    instruction = _clip(task_info.instruction, MAX_INSTRUCTION_CHARS)
    apps = _summarize_app_descriptions(task_info.app_descriptions)
    selected_apps = _selected_apps_from_results(results)
    selected_apps_text = ", ".join(selected_apps)
    prior = _compact_prior(results)
    exec_text = _clip(execution_outputs[-1]["output"], MAX_EXECUTION_OUTPUT_CHARS) if execution_outputs else ""
    eval_text = _clip(evaluation or {}, MAX_EVALUATION_CHARS) if evaluation else ""
    api_docs = _clip(task_info.api_docs_preview, MAX_API_DOCS_CHARS)
    supervisor = _clip(task_info.supervisor, 320)

    system = "You are a deterministic AppWorld agent."
    if stage == "task_analysis":
        user = (
            "Return JSON only: task_summary, relevant_apps, plan.\n\n"
            f"INSTRUCTION:\n{instruction}\n\nAPPS:\n{apps}"
        )
    elif stage == "api_planning":
        user = (
            "Return JSON only: selected_apps, api_needs, execution_plan. No code.\n\n"
            f"INSTRUCTION:\n{instruction}\n\nAPPS:\n{apps}\n\nRELEVANT_APPS:\n{selected_apps_text}"
        )
    elif stage == "code_generation":
        user = (
            "Return Python code only. Use apis.<app>.<api>(...). "
            "Call apis.supervisor.complete_task(...) when done.\n\n"
            f"TASK:\n{_clip(instruction, 360)}\n\nAPPS:\n{selected_apps_text}\n\nAPI_DOCS:\n{api_docs}\n\nPLAN:\n{_clip(prior, 100)}"
        )
    elif stage == "execution_verification":
        user = (
            "Return empty code if done; otherwise return one Python repair code block only.\n\n"
            f"TASK:\n{_clip(instruction, 420)}\n\nLAST_EXECUTION:\n{_clip(exec_text, 420)}\n\nEVALUATION:\n{_clip(eval_text, 360)}\n\nPLAN:\n{_clip(prior, 160)}"
        )
    else:
        raise ValueError(f"unknown AppWorld stage: {stage}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def preview_appworld_prompts(
    *,
    task_id: str,
    dataset_name: str,
    appworld_root: Path | None = None,
) -> dict[str, Any]:
    AppWorld, _load_task_ids, update_root = import_appworld()
    if appworld_root is not None:
        update_root(str(appworld_root))
    previews = []
    with AppWorld(task_id=task_id, experiment_name="edge_agent_appworld_prompt_preview") as world:
        results: list[AppWorldStageResult] = []
        execution_outputs: list[dict[str, Any]] = []
        evaluation: dict[str, Any] = {}
        for stage in APPWORLD_STAGES:
            task_info = _task_info_from_world(task_id, dataset_name, world, results)
            messages = messages_for_appworld_stage(
                task_info=task_info,
                stage=stage,
                results=results,
                execution_outputs=execution_outputs,
                evaluation=evaluation,
            )
            previews.append(
                {
                    "stage": stage,
                    "prompt_chars": sum(len(message["content"]) for message in messages),
                    "messages": messages,
                }
            )
            results.append(AppWorldStageResult(stage=stage, tier="cloud", latency_s=0.0, output="{}"))
    return {"task_id": task_id, "dataset_name": dataset_name, "stages": previews}


def _evaluation_to_dict(world: Any) -> dict[str, Any]:
    try:
        evaluation = world.evaluate()
        if hasattr(evaluation, "to_dict"):
            data = evaluation.to_dict()
        else:
            data = evaluation
    except Exception as exc:
        return {"success": False, "error": repr(exc)}
    if isinstance(data, dict):
        return data
    return {"raw": str(data)}


def evaluation_success(evaluation: dict[str, Any]) -> bool:
    if isinstance(evaluation.get("success"), bool):
        return bool(evaluation["success"])
    if isinstance(evaluation.get("passed"), bool):
        return bool(evaluation["passed"])
    fails = evaluation.get("fails")
    if isinstance(fails, list):
        return len(fails) == 0
    aggregate = evaluation.get("aggregate")
    if isinstance(aggregate, dict):
        for key in ("success", "tgc", "sgc"):
            value = aggregate.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value >= 1.0
    return False


def run_appworld_workflow(
    *,
    client: ChatClient,
    task_id: str,
    dataset_name: str,
    placement: tuple[str, ...],
    experiment_name: str,
    appworld_root: Path | None = None,
    dump_prompt_dir: Path | None = None,
) -> AppWorldTaskResult:
    if len(placement) != len(APPWORLD_STAGES):
        raise ValueError(f"placement must have {len(APPWORLD_STAGES)} tiers")

    AppWorld, _load_task_ids, update_root = import_appworld()
    if appworld_root is not None:
        update_root(str(appworld_root))

    stages: list[AppWorldStageResult] = []
    execution_outputs: list[dict[str, Any]] = []
    generated_code = ""
    repair_code = ""
    evaluation: dict[str, Any] = {}
    try:
        with AppWorld(task_id=task_id, experiment_name=experiment_name) as world:
            task_info = _task_info_from_world(task_id, dataset_name, world)
            for stage, tier in zip(APPWORLD_STAGES, placement):
                task_info = _task_info_from_world(task_id, dataset_name, world, stages)
                messages = messages_for_appworld_stage(
                    task_info=task_info,
                    stage=stage,
                    results=stages,
                    execution_outputs=execution_outputs,
                    evaluation=evaluation,
                )
                if dump_prompt_dir is not None:
                    dump_prompt_dir.mkdir(parents=True, exist_ok=True)
                    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
                    dump_path = dump_prompt_dir / f"{safe_task_id}_{stage}.json"
                    dump_path.write_text(
                        json.dumps({"stage": stage, "tier": tier, "messages": messages}, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                print(
                    f"  stage_start stage={stage} tier={tier} "
                    f"prompt_chars={sum(len(message['content']) for message in messages)}",
                    flush=True,
                )
                started = time.perf_counter()
                output = client.chat(tier=tier, stage=stage, messages=messages, incident=task_info.to_incident())
                latency_s = time.perf_counter() - started
                print(f"  stage_done stage={stage} tier={tier} latency_s={latency_s:.2f} output_chars={len(output)}", flush=True)
                stages.append(AppWorldStageResult(stage=stage, tier=tier, latency_s=latency_s, output=output))

                if stage == "code_generation":
                    generated_code = _extract_code(output)
                    exec_started = time.perf_counter()
                    exec_output = world.execute(generated_code)
                    execution_outputs.append(
                        {
                            "stage": stage,
                            "latency_s": time.perf_counter() - exec_started,
                            "code": generated_code,
                            "output": str(exec_output),
                        }
                    )
                    evaluation = _evaluation_to_dict(world)
                elif stage == "execution_verification":
                    repair_code = _extract_code(output)
                    if repair_code:
                        exec_started = time.perf_counter()
                        exec_output = world.execute(repair_code)
                        execution_outputs.append(
                            {
                                "stage": stage,
                                "latency_s": time.perf_counter() - exec_started,
                                "code": repair_code,
                                "output": str(exec_output),
                            }
                        )
                    evaluation = _evaluation_to_dict(world)
    except Exception as exc:
        return AppWorldTaskResult(
            task_id=task_id,
            placement=placement,
            ok=False,
            error=repr(exc),
            stages=stages,
            generated_code=generated_code,
            repair_code=repair_code,
            execution_outputs=execution_outputs,
            evaluation=evaluation,
        )

    return AppWorldTaskResult(
        task_id=task_id,
        placement=placement,
        ok=True,
        error="",
        stages=stages,
        generated_code=generated_code,
        repair_code=repair_code,
        execution_outputs=execution_outputs,
        evaluation=evaluation,
    )


class AppWorldMockClient:
    def chat(
        self,
        *,
        tier: str,
        stage: str,
        messages: list[dict[str, str]],
        incident: dict[str, Any] | None = None,
    ) -> str:
        del tier, messages, incident
        if stage == "task_analysis":
            return json.dumps({"task_summary": "mock", "relevant_apps": ["supervisor"], "plan": ["complete"]})
        if stage == "api_planning":
            return json.dumps({"selected_apps": ["supervisor"], "api_needs": [], "execution_plan": ["complete"]})
        if stage == "code_generation":
            return "apis.supervisor.complete_task()"
        return ""
