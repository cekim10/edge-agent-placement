"""SWE-agent test-repo Observation 1 harness.

This uses a real GitHub repository workload rather than a synthetic code task:
https://github.com/SWE-agent/test-repo

Fixed stage workflow:

    issue_analysis -> patch_generation -> test_repair

The final quality metric is whether the produced patch applies and the copied
repository's pytest suite passes.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import ChatClient


SWE_STAGES = ("issue_analysis", "patch_generation", "test_repair")

MAX_ISSUE_CHARS = 900
MAX_TREE_CHARS = 1800
MAX_CONTEXT_CHARS = 4000
MAX_LOCALIZED_CONTEXT_CHARS = 2400
MAX_PRIOR_CHARS = 600
MAX_PATCH_CHARS = 12000


@dataclass(frozen=True)
class SWEIssueTask:
    task_id: str
    statement_path: Path
    problem_statement: str

    def to_incident(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "problem_statement": self.problem_statement,
        }


@dataclass(frozen=True)
class SWEStageResult:
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


def load_swe_testrepo_tasks(repo_path: Path, limit: int = 0) -> list[SWEIssueTask]:
    problem_dir = repo_path / "problem_statements"
    if not problem_dir.exists():
        raise FileNotFoundError(
            f"Missing {problem_dir}. Clone https://github.com/SWE-agent/test-repo "
            "and pass --repo-path /path/to/test-repo."
        )
    tasks = []
    for path in sorted(problem_dir.glob("*.md")):
        tasks.append(
            SWEIssueTask(
                task_id=path.stem,
                statement_path=path,
                problem_statement=path.read_text(encoding="utf-8"),
            )
        )
    return tasks[:limit] if limit else tasks


def placement_for_edge_swe_stage(edge_stage: str | None) -> tuple[str, ...]:
    if edge_stage is None or edge_stage == "all_cloud":
        return tuple("cloud" for _ in SWE_STAGES)
    if edge_stage == "all_edge":
        return tuple("edge" for _ in SWE_STAGES)
    if edge_stage not in SWE_STAGES:
        raise ValueError(f"unknown SWE stage: {edge_stage}")
    return tuple("edge" if stage == edge_stage else "cloud" for stage in SWE_STAGES)


def placement_name(placement: tuple[str, ...]) -> str:
    if all(tier == "cloud" for tier in placement):
        return "all_cloud"
    if all(tier == "edge" for tier in placement):
        return "all_edge"
    edge = [stage for stage, tier in zip(SWE_STAGES, placement) if tier == "edge"]
    return "edge_" + "_".join(edge)


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[TRUNCATED]"


def _sanitize_prompt_text(text: str) -> str:
    text = text.replace("```", "")
    text = text.replace("`", "'")
    return text


def _numbered_file_text(text: str, max_chars: int = 1600) -> str:
    numbered = "\n".join(f"L{index:03d}: {_sanitize_prompt_text(line)}" for index, line in enumerate(text.splitlines(), start=1))
    return _clip(numbered, max_chars)


def _issue_line_numbers(issue: str) -> list[int]:
    line_numbers = [int(match) for match in re.findall(r"line\s+(\d+)", issue, flags=re.I)]
    return [line for line in line_numbers if line > 0]


def _issue_code_lines(issue: str) -> list[str]:
    lines = []
    for raw in issue.splitlines():
        stripped = _sanitize_prompt_text(raw).strip()
        if not stripped or stripped.startswith(("File ", "^", "SyntaxError")):
            continue
        if stripped.startswith(("def ", "return ", "class ", "import ", "from ")):
            lines.append(stripped)
    return lines[:6]


def _numbered_file_snippet(text: str, issue: str, radius: int = 2, max_chars: int = 900) -> str:
    lines = text.splitlines()
    selected: set[int] = set()
    for line_no in _issue_line_numbers(issue):
        for index in range(max(1, line_no - radius), min(len(lines), line_no + radius) + 1):
            selected.add(index)
    for needle in _issue_code_lines(issue):
        for index, line in enumerate(lines, start=1):
            if needle in _sanitize_prompt_text(line):
                for near in range(max(1, index - radius), min(len(lines), index + radius) + 1):
                    selected.add(near)
    if not selected:
        selected = set(range(1, min(len(lines), 8) + 1))
    numbered = "\n".join(f"L{index:03d}: {_sanitize_prompt_text(lines[index - 1])}" for index in sorted(selected))
    return _clip(numbered, max_chars)


def _repo_tree(repo_path: Path) -> str:
    paths = []
    for path in sorted(repo_path.rglob("*")):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.is_file():
            paths.append(str(path.relative_to(repo_path)))
    return "\n".join(paths)


def _repo_tree_context(repo_path: Path) -> str:
    return _clip(f"REPO_TREE:\n{_repo_tree(repo_path)}", MAX_TREE_CHARS)


def _repo_context(repo_path: Path) -> str:
    chunks = [_repo_tree_context(repo_path)]
    for path in sorted(repo_path.rglob("*.py")):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(repo_path)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        chunks.append(f"FILE: {rel}\n{_numbered_file_text(text)}")
    return _clip("\n\n".join(chunks), MAX_CONTEXT_CHARS)


def _all_repo_files(repo_path: Path) -> list[str]:
    return [
        str(path.relative_to(repo_path))
        for path in sorted(repo_path.rglob("*"))
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
    ]


def _extract_likely_files(results: list[SWEStageResult], repo_path: Path, issue: str) -> list[str]:
    available = _all_repo_files(repo_path)
    available_set = set(available)
    candidates: list[str] = []
    if results:
        text = results[-1].output
        try:
            parsed = json.loads(text)
            likely = parsed.get("likely_files", [])
            if isinstance(likely, str):
                likely = [likely]
            if isinstance(likely, list):
                candidates.extend(str(item) for item in likely)
        except json.JSONDecodeError:
            pass
        candidates.extend(re.findall(r"[A-Za-z0-9_./-]+\.py", text))
    candidates.extend(path for path in available if path in issue)

    selected = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.strip().lstrip("./")
        matches = [candidate] if candidate in available_set else [path for path in available if path.endswith(candidate)]
        for match in matches:
            if match not in seen:
                seen.add(match)
                selected.append(match)
    if not selected:
        selected = [path for path in available if path.endswith(".py")][:4]
    return selected[:4]


def _localized_repo_context(repo_path: Path, results: list[SWEStageResult], issue: str) -> str:
    selected = _extract_likely_files(results, repo_path, issue)
    chunks = ["SELECTED_FILES:\n" + "\n".join(selected)]
    for rel in selected:
        path = repo_path / rel
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        chunks.append(f"FILE_SNIPPET: {rel}\n{_numbered_file_snippet(text, issue)}")
    return _clip("\n\n".join(chunks), MAX_LOCALIZED_CONTEXT_CHARS)


def _prior_block(results: list[SWEStageResult]) -> str:
    if not results:
        return "None"
    text = "\n\n".join(
        f"{result.stage.upper()} OUTPUT:\n{_clip(result.output, MAX_PRIOR_CHARS)}"
        for result in results
    )
    return _clip(text, MAX_PRIOR_CHARS)


def messages_for_swe_stage(
    *,
    task: SWEIssueTask,
    repo_path: Path,
    stage: str,
    results: list[SWEStageResult],
    include_prior: bool = True,
) -> list[dict[str, str]]:
    issue = _clip(_sanitize_prompt_text(task.problem_statement), MAX_ISSUE_CHARS)
    if stage == "issue_analysis":
        context = _repo_tree_context(repo_path)
    else:
        context = _localized_repo_context(repo_path, results, task.problem_statement)
    prior = _prior_block(results) if include_prior else "None"
    system = (
        "You are a deterministic software engineering agent working on a real "
        "GitHub repository. Follow the requested output format exactly."
    )
    if stage == "issue_analysis":
        user = (
            "Stage A: triage the GitHub issue using the repo tree. Return compact JSON "
            "with keys bug_summary, likely_files, fix_strategy, test_strategy. Do not write a patch.\n\n"
            f"ISSUE:\n{issue}\n\nREPOSITORY_CONTEXT:\n{context}"
        )
    elif stage == "patch_generation":
        user = (
            "Edit selected files. Return a JSON object only. The object has key edits. "
            "Each edit has keys file, find, replace. The find value must be exact text copied from a FILE_SNIPPET block. "
            "If unsure, return an empty edits list. No markdown.\n\n"
            f"ISSUE:\n{issue}\n\n{context}\n\nPRIOR:\n{prior}"
        )
    elif stage == "test_repair":
        user = (
            "Review the prior edit. Return a JSON object only with key edits. "
            "Keep or improve the edit. If unsure, return an empty edits list. No markdown.\n\n"
            f"ISSUE:\n{issue}\n\n{context}\n\nPRIOR:\n{prior}"
        )
    else:
        raise ValueError(f"unknown SWE stage: {stage}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_unified_diff(text: str) -> str:
    fenced = re.findall(r"```(?:diff|patch)?\s*(.*?)```", text, flags=re.S | re.I)
    candidate = fenced[-1].strip() if fenced else text.strip()
    for marker in ("diff --git ", "--- "):
        index = candidate.find(marker)
        if index >= 0:
            candidate = candidate[index:]
            break
    return _clip(candidate.rstrip() + "\n", MAX_PATCH_CHARS)


def _extract_json_object(text: str) -> dict[str, Any]:
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    candidates = fenced + [text]
    for candidate in candidates:
        candidate = candidate.strip()
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end < start:
            continue
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _edit_plan_to_diff(repo_path: Path, text: str) -> str:
    parsed = _extract_json_object(text)
    edits = parsed.get("edits", [])
    if isinstance(edits, dict):
        edits = [edits]
    if not isinstance(edits, list):
        return ""

    original_by_file: dict[str, str] = {}
    updated_by_file: dict[str, str] = {}
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        rel = str(edit.get("file", "")).strip().lstrip("./")
        find = edit.get("find", "")
        replace = edit.get("replace", "")
        if not rel or not isinstance(find, str) or not isinstance(replace, str):
            continue
        path = repo_path / rel
        if not path.exists() or not path.is_file():
            continue
        original = original_by_file.setdefault(rel, path.read_text(encoding="utf-8"))
        current = updated_by_file.get(rel, original)
        if find not in current:
            continue
        updated_by_file[rel] = current.replace(find, replace, 1)

    chunks: list[str] = []
    for rel, updated in updated_by_file.items():
        original = original_by_file[rel]
        if original == updated:
            continue
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
        body = "".join(diff)
        if body:
            chunks.append(f"diff --git a/{rel} b/{rel}\n" + body)
    return _clip("\n".join(chunks), MAX_PATCH_CHARS)


def extract_patch_or_edit_diff(text: str, repo_path: Path) -> str:
    if "diff --git " in text or re.search(r"(?m)^---\s+", text):
        return extract_unified_diff(text)
    return _edit_plan_to_diff(repo_path, text)


def copy_repo_to_temp(repo_path: Path) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="edge_agent_swe_repo_"))
    target = tmpdir / "repo"
    ignore = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc")
    shutil.copytree(repo_path, target, ignore=ignore)
    return target


def apply_patch(repo_path: Path, patch_text: str, timeout_s: float = 10.0) -> dict[str, Any]:
    if not patch_text.strip():
        return {"applied": False, "stdout": "", "stderr": "empty patch"}
    completed = subprocess.run(
        ["git", "apply", "--whitespace=fix", "-"],
        input=patch_text,
        cwd=repo_path,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    return {
        "applied": completed.returncode == 0,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }


def run_pytest(repo_path: Path, timeout_s: float = 20.0) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_path / "src") + os.pathsep + env.get("PYTHONPATH", "")
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            ["python3", "-m", "pytest", "-q"],
            cwd=repo_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        elapsed = time.perf_counter() - started
    except subprocess.TimeoutExpired as exc:
        return {
            "passed": False,
            "result": "timed_out",
            "latency_s": timeout_s,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }
    return {
        "passed": completed.returncode == 0,
        "result": "passed" if completed.returncode == 0 else "failed",
        "latency_s": elapsed,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def run_swe_workflow(
    *,
    client: ChatClient,
    task: SWEIssueTask,
    repo_path: Path,
    placement: tuple[str, ...],
    dump_prompt_dir: Path | None = None,
    include_prior: bool = True,
) -> tuple[list[SWEStageResult], str]:
    if len(placement) != len(SWE_STAGES):
        raise ValueError(f"placement must have {len(SWE_STAGES)} tiers")
    results: list[SWEStageResult] = []
    final_patch = ""
    incident = task.to_incident()
    for stage, tier in zip(SWE_STAGES, placement):
        messages = messages_for_swe_stage(
            task=task,
            repo_path=repo_path,
            stage=stage,
            results=results,
            include_prior=include_prior,
        )
        prompt_chars = sum(len(message["content"]) for message in messages)
        if dump_prompt_dir is not None:
            dump_prompt_dir.mkdir(parents=True, exist_ok=True)
            safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task.task_id)
            dump_path = dump_prompt_dir / f"{safe_task_id}_{stage}.json"
            dump_path.write_text(
                json.dumps({"stage": stage, "tier": tier, "messages": messages}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(f"  stage_start stage={stage} tier={tier} prompt_chars={prompt_chars}", flush=True)
        started = time.perf_counter()
        output = client.chat(tier=tier, stage=stage, messages=messages, incident=incident)
        latency_s = time.perf_counter() - started
        print(
            f"  stage_done stage={stage} tier={tier} latency_s={latency_s:.2f} output_chars={len(output)}",
            flush=True,
        )
        results.append(SWEStageResult(stage=stage, tier=tier, latency_s=latency_s, output=output))
        if stage in {"patch_generation", "test_repair"}:
            final_patch = extract_patch_or_edit_diff(output, repo_path)
    return results, final_patch


class SWEMockClient:
    """Local smoke-test mock. It does not solve issues."""

    def chat(
        self,
        *,
        tier: str,
        stage: str,
        messages: list[dict[str, str]],
        incident: dict[str, Any] | None = None,
    ) -> str:
        del messages, incident
        if stage == "issue_analysis":
            return json.dumps({"bug_summary": "mock", "likely_files": [], "fix_strategy": "none"})
        return ""
