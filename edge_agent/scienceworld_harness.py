"""ScienceWorld Observation 1 harness.

ScienceWorld is optional at import time so local py_compile and mock runs do
not require Java or the benchmark package.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from statistics import mean
from typing import Any

from .client import ChatClient


SCIENCEWORLD_STAGES = (
    "state_abstraction",
    "subgoal_planning",
    "action_selection",
    "progress_verification",
)

MAX_TASK_CHARS = 240
MAX_OBS_CHARS = 360
MAX_HISTORY_CHARS = 260
MAX_STATE_CHARS = 220
MAX_PLAN_CHARS = 220
MAX_VERIFY_TASK_CHARS = 160
MAX_VERIFY_OBS_CHARS = 260
MAX_VERIFY_HISTORY_CHARS = 180
MAX_ACTION_TASK_CHARS = 110
MAX_ACTION_OBS_CHARS = 170
MAX_ACTION_STATE_CHARS = 90
MAX_ACTION_PLAN_CHARS = 90
MAX_VALID_ACTIONS = 8
MAX_VALID_CHARS = 220
REPEAT_ACTION_PENALTY = 10


@dataclass(frozen=True)
class ScienceWorldTaskSpec:
    task_name: str
    variation: int
    simplification: str

    @property
    def task_id(self) -> str:
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.task_name)
        safe_simplification = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.simplification or "none")
        return f"{safe_task}_v{self.variation}_{safe_simplification}"

    def to_incident(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "task_name": self.task_name,
            "variation": self.variation,
            "simplification": self.simplification,
        }


@dataclass(frozen=True)
class ScienceWorldStageResult:
    step_index: int
    stage: str
    tier: str
    latency_s: float
    output: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "stage": self.stage,
            "tier": self.tier,
            "latency_s": self.latency_s,
            "output": self.output,
        }


@dataclass(frozen=True)
class ScienceWorldStepResult:
    step_index: int
    action: str
    invalid_action: bool
    observation: str
    reward: float
    score: float
    done: bool
    valid_action_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "action": self.action,
            "invalid_action": self.invalid_action,
            "observation": self.observation,
            "reward": self.reward,
            "score": self.score,
            "done": self.done,
            "valid_action_count": self.valid_action_count,
        }


@dataclass
class ScienceWorldEpisodeResult:
    task_id: str
    task_name: str
    variation: int
    simplification: str
    placement: tuple[str, ...]
    ok: bool
    error: str
    success: bool
    final_score: float
    normalized_score: float
    steps_taken: int
    invalid_action_count: int
    stages: list[ScienceWorldStageResult]
    steps: list[ScienceWorldStepResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "variation": self.variation,
            "simplification": self.simplification,
            "placement": list(self.placement),
            "ok": self.ok,
            "error": self.error,
            "success": self.success,
            "final_score": self.final_score,
            "normalized_score": self.normalized_score,
            "steps_taken": self.steps_taken,
            "invalid_action_count": self.invalid_action_count,
            "stages": [stage.to_dict() for stage in self.stages],
            "steps": [step.to_dict() for step in self.steps],
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


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _info_score(info: dict[str, Any]) -> float:
    for key in ("score", "scoreNormalized", "score_normalized"):
        if key in info:
            score = _as_float(info[key])
            return score * 100.0 if key != "score" and score <= 1.0 else score
    return 0.0


def _valid_actions(info: dict[str, Any]) -> list[str]:
    valid = info.get("valid", info.get("validActions", info.get("admissible_commands", [])))
    if isinstance(valid, str):
        return [valid]
    if isinstance(valid, (list, tuple)):
        return [str(action) for action in valid if str(action).strip()]
    return []


def _short_history(steps: list[ScienceWorldStepResult]) -> str:
    lines = []
    for step in steps[-3:]:
        lines.append(
            f"{step.step_index}. action={step.action!r} score={step.score:.1f} "
            f"obs={_clip(step.observation, 80)}"
        )
    return _clip("\n".join(lines) if lines else "none", MAX_HISTORY_CHARS)


def _ranked_valid_actions(
    valid_actions: list[str],
    context: str = "",
    recent_actions: list[str] | None = None,
) -> list[str]:
    context_words = {
        word
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", context.lower())
        if len(word) >= 4
    }
    recent_set = {action.lower() for action in (recent_actions or [])}
    context_lower = context.lower()
    preferred_prefixes = (
        "look",
        "examine",
        "focus",
        "open",
        "close",
        "move",
        "go",
        "take",
        "pick",
        "put",
        "place",
        "drop",
        "use",
        "activate",
        "deactivate",
        "pour",
        "mix",
        "wait",
    )
    scored: list[tuple[int, int, str]] = []
    for index, action in enumerate(valid_actions):
        lowered = action.lower()
        score = 0
        if any(lowered.startswith(prefix) for prefix in preferred_prefixes):
            score += 4
        if any(word in lowered for word in context_words):
            score += 3
        if lowered == "look around":
            score += 5
        if lowered == "inventory":
            score -= 2
        if lowered in recent_set:
            score -= REPEAT_ACTION_PENALTY
        if "non-living" in context_lower:
            if any(word in lowered for word in ("orange", "apple", "banana", "plant", "animal", "person", "air")):
                score -= 6
            if any(word in lowered for word in ("picture", "box", "table", "chair", "door", "book", "key", "metal", "wood")):
                score += 6
        scored.append((-score, index, action))
    return [action for _score, _index, action in sorted(scored)[:MAX_VALID_ACTIONS]]


def _valid_actions_text(
    valid_actions: list[str],
    context: str = "",
    recent_actions: list[str] | None = None,
) -> str:
    shown = _ranked_valid_actions(valid_actions, context, recent_actions)
    lines = [f"{index}. {action}" for index, action in enumerate(shown, start=1)]
    if len(valid_actions) > len(shown):
        lines.append(f"... {len(valid_actions) - len(shown)} more actions omitted")
    return _clip("\n".join(lines), MAX_VALID_CHARS)


def _latest_stage_output(stages: list[ScienceWorldStageResult], stage: str) -> str:
    for result in reversed(stages):
        if result.stage == stage:
            return result.output
    return ""


def _parsed_stage(stages: list[ScienceWorldStageResult], stage: str) -> dict[str, Any]:
    parsed = _json_loads_loose(_latest_stage_output(stages, stage))
    return parsed if isinstance(parsed, dict) else {}


def _compact_stage_fields(stage_data: dict[str, Any], keys: tuple[str, ...], max_chars: int) -> str:
    parts = []
    for key in keys:
        value = stage_data.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value[:4])
        parts.append(f"{key}: {value}")
    return _clip("; ".join(parts), max_chars) if parts else "none"


def import_scienceworld() -> Any:
    try:
        from scienceworld import ScienceWorldEnv
    except ImportError as exc:
        raise RuntimeError(
            "ScienceWorld is not installed. On the GPU server run: "
            "uv venv --python 3.11 .venv-scienceworld && source .venv-scienceworld/bin/activate && "
            "uv pip install scienceworld"
        ) from exc
    return ScienceWorldEnv


def _make_env(max_steps: int) -> Any:
    del max_steps
    ScienceWorldEnv = import_scienceworld()
    for args in (("",), ()):
        try:
            return ScienceWorldEnv(*args)
        except TypeError:
            continue
    return ScienceWorldEnv()


def list_scienceworld_tasks() -> list[str]:
    env = _make_env(max_steps=1)
    try:
        return [str(name) for name in env.getTaskNames()]
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def load_scienceworld_task_specs(
    *,
    task_names: list[str] | None,
    task_indices: list[int] | None,
    variations: list[int],
    simplification: str,
    limit: int,
) -> list[ScienceWorldTaskSpec]:
    names: list[str] = []
    all_names: list[str] | None = None
    if task_names:
        names.extend(task_names)
    if task_indices:
        all_names = list_scienceworld_tasks()
        for index in task_indices:
            if index < 0 or index >= len(all_names):
                raise ValueError(f"task index {index} out of range; ScienceWorld exposes {len(all_names)} tasks")
            names.append(all_names[index])
    if not names:
        names = ["find-non-living-thing"]

    specs = [
        ScienceWorldTaskSpec(task_name=task_name, variation=variation, simplification=simplification)
        for task_name in names
        for variation in variations
    ]
    return specs[:limit] if limit else specs


def placement_for_edge_scienceworld_stage(edge_stage: str | None) -> tuple[str, ...]:
    if edge_stage is None or edge_stage == "all_cloud":
        return tuple("cloud" for _ in SCIENCEWORLD_STAGES)
    if edge_stage == "all_edge":
        return tuple("edge" for _ in SCIENCEWORLD_STAGES)
    if edge_stage not in SCIENCEWORLD_STAGES:
        raise ValueError(f"unknown ScienceWorld stage: {edge_stage}")
    return tuple("edge" if stage == edge_stage else "cloud" for stage in SCIENCEWORLD_STAGES)


def placement_name(placement: tuple[str, ...]) -> str:
    if all(tier == "cloud" for tier in placement):
        return "all_cloud"
    if all(tier == "edge" for tier in placement):
        return "all_edge"
    edge = [stage for stage, tier in zip(SCIENCEWORLD_STAGES, placement) if tier == "edge"]
    return "edge_" + "_".join(edge)


def messages_for_scienceworld_stage(
    *,
    task_description: str,
    observation: str,
    info: dict[str, Any],
    step_index: int,
    stage: str,
    stages: list[ScienceWorldStageResult],
    steps: list[ScienceWorldStepResult],
) -> list[dict[str, str]]:
    score = _info_score(info)
    valid = _valid_actions(info)
    history = _short_history(steps)
    recent_actions = [step.action for step in steps[-3:]]
    system = "You are a deterministic ScienceWorld agent. Return exactly the requested JSON."
    common = (
        f"TASK:\n{_clip(task_description, MAX_TASK_CHARS)}\n\n"
        f"STEP: {step_index}\nSCORE: {score:.1f}/100\n\n"
        f"OBSERVATION:\n{_clip(observation, MAX_OBS_CHARS)}\n\n"
        f"RECENT_HISTORY:\n{history}"
    )
    if stage == "state_abstraction":
        user = (
            "Return JSON only with keys state_summary, relevant_objects, progress, blockers.\n\n"
            f"{common}"
        )
    elif stage == "subgoal_planning":
        state = _clip(_parsed_stage(stages, "state_abstraction"), MAX_STATE_CHARS)
        user = (
            "Return JSON only with keys subgoal, plan, stop_condition. "
            "Plan for the next 1-3 environment actions.\n\n"
            f"{common}\n\nSTATE_ABSTRACTION:\n{state}"
        )
    elif stage == "action_selection":
        state = _compact_stage_fields(
            _parsed_stage(stages, "state_abstraction"),
            ("state_summary", "relevant_objects", "progress", "blockers"),
            MAX_ACTION_STATE_CHARS,
        )
        plan = _compact_stage_fields(
            _parsed_stage(stages, "subgoal_planning"),
            ("subgoal", "plan", "stop_condition"),
            MAX_ACTION_PLAN_CHARS,
        )
        action_context = f"{task_description}\n{observation}\n{state}\n{plan}"
        user = (
            "Pick the best next action by number. Return JSON only: {\"index\":N}.\n\n"
            f"TASK: {_clip(task_description, MAX_ACTION_TASK_CHARS)}\n"
            f"STEP: {step_index} SCORE: {score:.1f}/100\n"
            f"OBS: {_clip(observation, MAX_ACTION_OBS_CHARS)}\n"
            f"STATE: {state}\n"
            f"PLAN: {plan}\n"
            f"RECENT_ACTIONS: {', '.join(recent_actions) if recent_actions else 'none'}\n"
            "Avoid repeating an action unless score increased last time.\n"
            f"ACTIONS:\n{_valid_actions_text(valid, action_context, recent_actions)}"
        )
    elif stage == "progress_verification":
        action = _clip(steps[-1].action if steps else "", 120)
        last_obs = _clip(steps[-1].observation if steps else observation, MAX_VERIFY_OBS_CHARS)
        recent = _clip(_short_history(steps), MAX_VERIFY_HISTORY_CHARS)
        user = (
            "Return JSON only: {\"continue\":true|false,\"status\":\"...\"}. "
            "Set continue=false only if the task is complete or no useful action remains.\n\n"
            f"TASK:\n{_clip(task_description, MAX_VERIFY_TASK_CHARS)}\n\n"
            f"STEP: {step_index}\nSCORE: {_info_score(info):.1f}/100\nLAST_ACTION: {action}\n\n"
            f"LAST_OBSERVATION:\n{last_obs}\n\nRECENT_HISTORY:\n{recent}"
        )
    else:
        raise ValueError(f"unknown ScienceWorld stage: {stage}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _select_action(
    output: str,
    valid_actions: list[str],
    context: str = "",
    recent_actions: list[str] | None = None,
) -> tuple[str, bool]:
    ranked_actions = _ranked_valid_actions(valid_actions, context, recent_actions)
    parsed = _json_loads_loose(output)
    action = ""
    if isinstance(parsed, dict):
        action = str(parsed.get("action", "")).strip()
        index_value = parsed.get("action_index", parsed.get("index"))
        if isinstance(index_value, int) and 1 <= index_value <= len(ranked_actions):
            return ranked_actions[index_value - 1], False
        if isinstance(index_value, str) and index_value.strip().isdigit():
            index = int(index_value.strip())
            if 1 <= index <= len(ranked_actions):
                return ranked_actions[index - 1], False
    if not action:
        text = output.strip()
        number_match = re.search(r"\b(\d{1,3})\b", text)
        if number_match:
            index = int(number_match.group(1))
            if 1 <= index <= len(ranked_actions):
                return ranked_actions[index - 1], False
        for candidate in ranked_actions + valid_actions:
            if candidate and candidate in text:
                action = candidate
                break
    if action in valid_actions:
        return action, False
    lowered = action.lower()
    for candidate in ranked_actions + valid_actions:
        if lowered and lowered == candidate.lower():
            return candidate, False
    for fallback in ("look around", "inventory"):
        if fallback in valid_actions:
            return fallback, True
    return (valid_actions[0] if valid_actions else action or "look around"), True


def _should_continue(output: str, *, done: bool, score: float) -> bool:
    if done or score >= 100.0:
        return False
    parsed = _json_loads_loose(output)
    if isinstance(parsed, dict) and isinstance(parsed.get("continue"), bool):
        return bool(parsed["continue"])
    return True


def _task_description(env: Any) -> str:
    for name in ("get_task_description", "getTaskDescription"):
        fn = getattr(env, name, None)
        if callable(fn):
            try:
                return str(fn())
            except Exception:
                pass
    return ""


def _reset_env(env: Any, spec: ScienceWorldTaskSpec) -> tuple[str, dict[str, Any]]:
    env.load(spec.task_name, spec.variation, spec.simplification)
    reset_with_variation = getattr(env, "resetWithVariation", None)
    if callable(reset_with_variation):
        return reset_with_variation(spec.variation, spec.simplification)
    return env.reset()


def run_scienceworld_workflow(
    *,
    client: ChatClient,
    task_spec: ScienceWorldTaskSpec,
    placement: tuple[str, ...],
    max_steps: int,
    dump_prompt_dir: Any | None = None,
) -> ScienceWorldEpisodeResult:
    if len(placement) != len(SCIENCEWORLD_STAGES):
        raise ValueError(f"placement must have {len(SCIENCEWORLD_STAGES)} tiers")

    stages: list[ScienceWorldStageResult] = []
    steps: list[ScienceWorldStepResult] = []
    final_score = 0.0
    success = False
    env = None
    try:
        env = _make_env(max_steps=max_steps)
        observation, info = _reset_env(env, task_spec)
        task_description = _task_description(env)
        if not task_description:
            task_description = str(info.get("taskDesc", info.get("task_description", task_spec.task_name)))
        info = dict(info or {})
        final_score = _info_score(info)

        for step_index in range(1, max_steps + 1):
            valid_actions = _valid_actions(info)
            if not valid_actions:
                valid_actions = ["look around"]

            step_stage_results: list[ScienceWorldStageResult] = []
            for stage, tier in zip(SCIENCEWORLD_STAGES[:3], placement[:3]):
                messages = messages_for_scienceworld_stage(
                    task_description=task_description,
                    observation=observation,
                    info=info,
                    step_index=step_index,
                    stage=stage,
                    stages=stages + step_stage_results,
                    steps=steps,
                )
                if dump_prompt_dir is not None:
                    dump_prompt_dir.mkdir(parents=True, exist_ok=True)
                    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_spec.task_id)
                    dump_path = dump_prompt_dir / f"{safe_task_id}_step{step_index:02d}_{stage}.json"
                    dump_path.write_text(
                        json.dumps({"stage": stage, "tier": tier, "messages": messages}, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                print(
                    f"  stage_start step={step_index} stage={stage} tier={tier} "
                    f"prompt_chars={sum(len(message['content']) for message in messages)}",
                    flush=True,
                )
                started = time.perf_counter()
                output = client.chat(tier=tier, stage=stage, messages=messages, incident=task_spec.to_incident())
                latency_s = time.perf_counter() - started
                print(
                    f"  stage_done step={step_index} stage={stage} tier={tier} "
                    f"latency_s={latency_s:.2f} output_chars={len(output)}",
                    flush=True,
                )
                stage_result = ScienceWorldStageResult(
                    step_index=step_index,
                    stage=stage,
                    tier=tier,
                    latency_s=latency_s,
                    output=output,
                )
                step_stage_results.append(stage_result)

            action_output = step_stage_results[-1].output
            action_context = (
                f"{task_description}\n{observation}\n"
                f"{_latest_stage_output(step_stage_results, 'state_abstraction')}\n"
                f"{_latest_stage_output(step_stage_results, 'subgoal_planning')}"
            )
            recent_actions = [step.action for step in steps[-3:]]
            action, invalid = _select_action(action_output, valid_actions, action_context, recent_actions)
            next_observation, reward, done, next_info = env.step(action)
            next_info = dict(next_info or {})
            final_score = _info_score(next_info)
            steps.append(
                ScienceWorldStepResult(
                    step_index=step_index,
                    action=action,
                    invalid_action=invalid,
                    observation=str(next_observation),
                    reward=_as_float(reward),
                    score=final_score,
                    done=bool(done),
                    valid_action_count=len(valid_actions),
                )
            )
            stages.extend(step_stage_results)

            verifier_tier = placement[3]
            verify_messages = messages_for_scienceworld_stage(
                task_description=task_description,
                observation=str(next_observation),
                info=next_info,
                step_index=step_index,
                stage="progress_verification",
                stages=stages,
                steps=steps,
            )
            if dump_prompt_dir is not None:
                safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_spec.task_id)
                dump_path = dump_prompt_dir / f"{safe_task_id}_step{step_index:02d}_progress_verification.json"
                dump_path.write_text(
                    json.dumps(
                        {"stage": "progress_verification", "tier": verifier_tier, "messages": verify_messages},
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            print(
                f"  stage_start step={step_index} stage=progress_verification tier={verifier_tier} "
                f"prompt_chars={sum(len(message['content']) for message in verify_messages)}",
                flush=True,
            )
            started = time.perf_counter()
            verify_output = client.chat(
                tier=verifier_tier,
                stage="progress_verification",
                messages=verify_messages,
                incident=task_spec.to_incident(),
            )
            verify_latency_s = time.perf_counter() - started
            print(
                f"  stage_done step={step_index} stage=progress_verification tier={verifier_tier} "
                f"latency_s={verify_latency_s:.2f} output_chars={len(verify_output)} score={final_score:.1f}",
                flush=True,
            )
            stages.append(
                ScienceWorldStageResult(
                    step_index=step_index,
                    stage="progress_verification",
                    tier=verifier_tier,
                    latency_s=verify_latency_s,
                    output=verify_output,
                )
            )
            success = final_score >= 100.0
            if not _should_continue(verify_output, done=bool(done), score=final_score):
                break
            observation, info = str(next_observation), next_info
    except Exception as exc:
        return ScienceWorldEpisodeResult(
            task_id=task_spec.task_id,
            task_name=task_spec.task_name,
            variation=task_spec.variation,
            simplification=task_spec.simplification,
            placement=placement,
            ok=False,
            error=repr(exc),
            success=success,
            final_score=final_score,
            normalized_score=final_score / 100.0,
            steps_taken=len(steps),
            invalid_action_count=sum(1 for step in steps if step.invalid_action),
            stages=stages,
            steps=steps,
        )
    finally:
        if env is not None:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    return ScienceWorldEpisodeResult(
        task_id=task_spec.task_id,
        task_name=task_spec.task_name,
        variation=task_spec.variation,
        simplification=task_spec.simplification,
        placement=placement,
        ok=True,
        error="",
        success=success,
        final_score=final_score,
        normalized_score=final_score / 100.0,
        steps_taken=len(steps),
        invalid_action_count=sum(1 for step in steps if step.invalid_action),
        stages=stages,
        steps=steps,
    )


class ScienceWorldMockClient:
    def chat(
        self,
        *,
        tier: str,
        stage: str,
        messages: list[dict[str, str]],
        incident: dict[str, Any] | None = None,
    ) -> str:
        del tier, messages, incident
        if stage == "state_abstraction":
            return json.dumps({"state_summary": "mock room", "relevant_objects": ["box"], "progress": "starting", "blockers": []})
        if stage == "subgoal_planning":
            return json.dumps({"subgoal": "inspect room", "plan": ["look around"], "stop_condition": "score 100"})
        if stage == "action_selection":
            return json.dumps({"action": "look around"})
        if stage == "progress_verification":
            return json.dumps({"continue": False, "status": "mock done"})
        return "{}"


def mock_scienceworld_result(task_spec: ScienceWorldTaskSpec, placement: tuple[str, ...]) -> ScienceWorldEpisodeResult:
    stages = [
        ScienceWorldStageResult(step_index=1, stage=stage, tier=tier, latency_s=0.0, output="{}")
        for stage, tier in zip(SCIENCEWORLD_STAGES, placement)
    ]
    steps = [
        ScienceWorldStepResult(
            step_index=1,
            action="look around",
            invalid_action=False,
            observation="mock observation",
            reward=0.0,
            score=100.0,
            done=True,
            valid_action_count=1,
        )
    ]
    return ScienceWorldEpisodeResult(
        task_id=task_spec.task_id,
        task_name=task_spec.task_name,
        variation=task_spec.variation,
        simplification=task_spec.simplification,
        placement=placement,
        ok=True,
        error="",
        success=True,
        final_score=100.0,
        normalized_score=1.0,
        steps_taken=1,
        invalid_action_count=0,
        stages=stages,
        steps=steps,
    )
