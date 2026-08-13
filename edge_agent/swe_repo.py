"""SWE-agent test-repo Observation 1 harness.

This uses a real GitHub repository workload rather than a synthetic code task:
https://github.com/SWE-agent/test-repo

Fixed stage workflow:

    issue_analysis -> patch_generation -> test_repair

The final quality metric is whether the produced patch applies and the task
specific reproducer passes.
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

MAX_ISSUE_CHARS = 650
MAX_TREE_CHARS = 1800
MAX_CONTEXT_CHARS = 4000
MAX_LOCALIZED_CONTEXT_CHARS = 1500
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


def _issue_symbols(issue: str) -> list[str]:
    symbols = []
    for imported in re.findall(r"\bfrom\s+[A-Za-z_][A-Za-z0-9_.]*\s+import\s+([A-Za-z0-9_,\s]+)", issue):
        symbols.extend(item.strip() for item in imported.split(","))
    symbols.extend(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", issue))
    ignore = {"assert", "print", "return", "if", "for", "while", "with"}
    return list(dict.fromkeys(symbol for symbol in symbols if symbol and symbol not in ignore))


def _issue_traceback_files(issue: str, repo_path: Path) -> list[str]:
    candidates = []
    for match in re.findall(r'File\s+"([^"]+\.py)"', issue):
        normalized = match.replace("\\", "/").replace("/./", "/")
        for anchor in ("/tests/", "/src/"):
            if anchor in normalized:
                rel = normalized.split(anchor, 1)[1]
                rel = anchor.strip("/") + "/" + rel
                if (repo_path / rel).exists():
                    candidates.append(rel)
    return list(dict.fromkeys(candidates))


def _issue_import_files(issue: str, repo_path: Path) -> list[str]:
    modules = []
    modules.extend(re.findall(r"\bfrom\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import\b", issue))
    modules.extend(re.findall(r"\bimport\s+([A-Za-z_][A-Za-z0-9_.]*)\b", issue))
    candidates = []
    for module in modules:
        module_path = module.replace(".", "/")
        for rel in (f"src/{module_path}.py", f"{module_path}.py", f"src/{module_path}/__init__.py"):
            if (repo_path / rel).exists():
                candidates.append(rel)
    return list(dict.fromkeys(candidates))


def _numbered_file_snippet(text: str, issue: str, radius: int = 2, max_chars: int = 850) -> str:
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
    for symbol in _issue_symbols(issue):
        pattern = re.compile(rf"^\s*(def|class)\s+{re.escape(symbol)}\b")
        for index, line in enumerate(lines, start=1):
            if pattern.search(_sanitize_prompt_text(line)):
                for near in range(max(1, index - 1), min(len(lines), index + 10) + 1):
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
    candidates.extend(_issue_traceback_files(issue, repo_path))
    candidates.extend(_issue_import_files(issue, repo_path))
    candidates.extend(path for path in available if path in issue)
    if results:
        text = results[-1].output
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                likely = parsed.get("likely_files", [])
                if isinstance(likely, str):
                    likely = [likely]
                if isinstance(likely, list):
                    candidates.extend(str(item) for item in likely)
        except json.JSONDecodeError:
            pass
        candidates.extend(re.findall(r"[A-Za-z0-9_./-]+\.py", text))

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
    return selected[:2]


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
    prior = _prior_block(results) if include_prior and results else ""
    system = "You are a deterministic software engineering agent. Return exactly what is requested."
    if stage == "issue_analysis":
        user = (
            "Stage A: triage the GitHub issue using the repo tree. Return compact JSON "
            "with keys bug_summary, likely_files, fix_strategy, test_strategy. Do not write a patch.\n\n"
            f"ISSUE:\n{issue}\n\nREPOSITORY_CONTEXT:\n{context}"
        )
    elif stage == "patch_generation":
        prior_text = f"\n\nPRIOR:\n{prior}" if prior else ""
        user = (
            'Return JSON only: {"edits":[{"file":"path.py","line":1,"new":"replacement line"}]}. '
            "Use L001 as line number. One-line new value. Smallest edit. No whole-file rewrite. "
            'If unsure return {"edits":[]}.\n\n'
            f"ISSUE:\n{issue}\n\n{context}{prior_text}"
        )
    elif stage == "test_repair":
        prior_text = f"\n\nPRIOR:\n{prior}" if prior else ""
        user = (
            'Return JSON only: {"edits":[{"file":"path.py","line":1,"new":"replacement line"}]}. '
            "Use L001 as line number. One-line new value. Keep or improve prior edit. "
            'If unsure return {"edits":[]}.\n\n'
            f"ISSUE:\n{issue}\n\n{context}{prior_text}"
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


def _extract_json_value(text: str) -> Any:
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
        start = candidate.find("[")
        end = candidate.rfind("]")
        if 0 <= start < end:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}


def _extract_json_object(text: str) -> dict[str, Any]:
    parsed = _extract_json_value(text)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"edits": parsed}
    return {}


def _strip_prompt_line_numbers(text: str) -> str:
    return "\n".join(re.sub(r"^L\d{3}:\s?", "", line) for line in text.splitlines())


def _line_signature(text: str) -> str:
    text = _strip_prompt_line_numbers(text).strip()
    text = text[:-1].rstrip() if text.endswith(":") else text
    return re.sub(r"\s+", " ", text)


def _replacement_index(lines: list[str], requested_index: int, new_first_line: str) -> int | None:
    if 0 <= requested_index < len(lines):
        requested = _line_signature(lines[requested_index])
        replacement = _line_signature(new_first_line)
        if requested == replacement or requested in replacement or replacement in requested:
            return requested_index
    replacement = _line_signature(new_first_line)
    for index, line in enumerate(lines):
        if _line_signature(line) == replacement:
            return index
    return requested_index if 0 <= requested_index < len(lines) else None


def _preserve_first_line_indent(old_line: str, new_line: str) -> str:
    old_indent = re.match(r"\s*", old_line).group(0)
    return old_indent + new_line.strip()


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
        path = repo_path / rel
        if not rel or not path.exists() or not path.is_file():
            continue
        original = original_by_file.setdefault(rel, path.read_text(encoding="utf-8"))
        current = updated_by_file.get(rel, original)

        line_number = edit.get("line")
        new_line = edit.get("new")
        if isinstance(line_number, str):
            line_match = re.search(r"\d+", line_number)
            line_number = int(line_match.group(0)) if line_match else line_number
        if isinstance(line_number, int) and isinstance(new_line, str):
            current_lines = current.splitlines(keepends=True)
            new_lines = _strip_prompt_line_numbers(new_line).splitlines()
            if not new_lines:
                continue
            replacement_at = _replacement_index(current_lines, line_number - 1, new_lines[0])
            if replacement_at is not None:
                old_line = current_lines[replacement_at]
                ending = "\n" if old_line.endswith("\n") else ""
                new_lines[0] = _preserve_first_line_indent(old_line, new_lines[0])
                replacement = [line.rstrip("\n") + "\n" for line in new_lines]
                if replacement:
                    replacement[-1] = replacement[-1].rstrip("\n") + ending
                span = max(1, len(new_lines))
                current_lines[replacement_at : replacement_at + span] = replacement
                updated_by_file[rel] = "".join(current_lines)
                continue

        find = edit.get("find", "")
        replace = edit.get("replace", "")
        if not isinstance(find, str) or not isinstance(replace, str):
            continue
        find = _strip_prompt_line_numbers(find)
        replace = _strip_prompt_line_numbers(replace)
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


def _run_python_validation(
    repo_path: Path,
    args: list[str],
    *,
    timeout_s: float,
    kind: str,
) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_path / "src") + os.pathsep + env.get("PYTHONPATH", "")
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            args,
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
            "validation": kind,
            "command": args,
            "latency_s": timeout_s,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }
    return {
        "passed": completed.returncode == 0,
        "result": "passed" if completed.returncode == 0 else "failed",
        "validation": kind,
        "command": args,
        "latency_s": elapsed,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def _issue_python_blocks(issue: str) -> list[str]:
    blocks = re.findall(r"```python\s*(.*?)```", issue, flags=re.S | re.I)
    return [block.strip() for block in blocks if block.strip()]


def run_task_validation(repo_path: Path, task: SWEIssueTask, timeout_s: float = 20.0) -> dict[str, Any]:
    """Run the issue-specific reproducer instead of the repo-wide test suite."""
    traceback_files = _issue_traceback_files(task.problem_statement, repo_path)
    if traceback_files:
        results = [
            _run_python_validation(
                repo_path,
                ["python3", rel],
                timeout_s=timeout_s,
                kind="traceback_file",
            )
            for rel in traceback_files
        ]
        return {
            "passed": all(result["passed"] for result in results),
            "result": "passed" if all(result["passed"] for result in results) else "failed",
            "validation": "traceback_file",
            "commands": [result["command"] for result in results],
            "latency_s": sum(result["latency_s"] for result in results),
            "stdout": "\n".join(result["stdout"] for result in results)[-4000:],
            "stderr": "\n".join(result["stderr"] for result in results)[-4000:],
        }

    code_blocks = _issue_python_blocks(task.problem_statement)
    if code_blocks:
        code = "\n\n".join(code_blocks)
        return _run_python_validation(
            repo_path,
            ["python3", "-c", code],
            timeout_s=timeout_s,
            kind="python_block",
        )

    result = run_pytest(repo_path, timeout_s=timeout_s)
    result["validation"] = "pytest"
    return result


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
        if stage == "test_repair" and not include_prior:
            results.append(SWEStageResult(stage=stage, tier=tier, latency_s=0.0, output=""))
            continue
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
            candidate_patch = extract_patch_or_edit_diff(output, repo_path)
            if stage == "patch_generation" or (include_prior and candidate_patch.strip()):
                final_patch = candidate_patch
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
