# Observation 1: SWE-agent test-repo Stage Sensitivity

Goal:

```text
Agent workflow stages exhibit heterogeneous sensitivity to edge execution.
```

This Option-B workload uses a real GitHub repository:

```text
https://github.com/SWE-agent/test-repo
```

SWE-agent's own documentation uses this repository for individual issue-solving
examples. The harness consumes the repository's real `problem_statements/`,
source files, and tests.

Fixed workflow:

```text
issue_analysis -> patch_generation -> test_repair
```

Stage A sees only the repository tree and issue text for triage/localization.
Patch generation and test repair receive selected file names plus localized Python line snippets selected from Stage A `likely_files` and issue line numbers; the repository tree is not repeated in later stages. Stage B/C return compact edit JSON using either `file,line,new` or `file,find,replace`; the harness converts it to a unified diff before applying it.

Final quality:

```text
copy repo -> convert generated edit JSON to unified diff -> git apply -> run task-specific reproducer
```

`test_repair` is a real feedback stage: the harness applies the Stage B patch
to a temporary copy, runs the task-specific validation, and passes only the
patch/test failure summary to Stage C. It does not use hidden solutions or
task-specific oracle patches.

The validator uses the traceback file from the problem statement when present,
then fenced Python repro code, and falls back to `python3 -m pytest -q` only
when the issue does not contain a specific reproducer. This avoids penalizing
one issue for unrelated known bugs in the test repository.

## Setup

On the GPU server:

```bash
git clone https://github.com/SWE-agent/test-repo.git ~/test-repo
```

## Clean All-Cloud Baseline

```bash
CLOUD_API_KIND=completions \
CLOUD_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
python3 scripts/run_swe_testrepo_observation1.py \
  --repo-path ~/test-repo \
  --only-placement all_cloud \
  --timeout-s 120 \
  --max-tokens 256 \
  --analysis-max-tokens 96 \
  --patch-max-tokens 160 \
  --repair-max-tokens 160 \
  --dump-prompts
```

Do not run placement variants until all-cloud pass rate is meaningful.

## Stage Sensitivity

```bash
CLOUD_API_KIND=completions \
EDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
CLOUD_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
python3 scripts/run_swe_testrepo_observation1.py \
  --repo-path ~/test-repo \
  --timeout-s 120 \
  --max-tokens 256 \
  --analysis-max-tokens 96 \
  --patch-max-tokens 160 \
  --repair-max-tokens 160 \
  --dump-prompts
```

Placements:

```text
cloud,cloud,cloud
edge,cloud,cloud
cloud,edge,cloud
cloud,cloud,edge
edge,edge,edge
```

Interpretation target:

- `edge_patch_generation` should usually hurt more than `edge_issue_analysis`.
- `edge_test_repair` measures whether a weak final reviewer breaks or fails to
  repair otherwise plausible patches.


## Prompt Debugging

If a vLLM endpoint hangs on a specific stage, dump the exact OpenAI-compatible
messages:

```bash
python3 scripts/run_swe_testrepo_observation1.py \
  --repo-path ~/test-repo \
  --limit 1 \
  --only-placement all_cloud \
  --dump-prompts
```

Prompts are written under `outputs/swe_testrepo_observation1_<timestamp>/prompts/`.
Use `--no-prior` to test whether prior stage outputs are triggering a prompt-specific hang.


To isolate prompt-specific vLLM hangs in Stage B:

```bash
CLOUD_MODEL=Qwen/Qwen2.5-7B-Instruct \
python3 scripts/probe_swe_stage_prompt.py \
  --repo-path ~/test-repo \
  --timeout-s 30 \
  --max-tokens 1
```
