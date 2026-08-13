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
Patch generation and test repair receive localized Python line snippets selected from Stage A `likely_files` and issue line numbers. Stage B/C return compact edit JSON; the harness converts it to a unified diff before applying it.

Final quality:

```text
copy repo -> convert generated edit JSON to unified diff -> git apply -> run python3 -m pytest -q
```

## Setup

On the GPU server:

```bash
git clone https://github.com/SWE-agent/test-repo.git ~/test-repo
```

## Clean All-Cloud Baseline

```bash
CLOUD_MODEL=Qwen/Qwen2.5-14B-Instruct \
python3 scripts/run_swe_testrepo_observation1.py \
  --repo-path ~/test-repo \
  --only-placement all_cloud \
  --timeout-s 180 \
  --max-tokens 1024 \
  --analysis-max-tokens 256 \
  --patch-max-tokens 768 \
  --repair-max-tokens 768
```

Do not run placement variants until all-cloud pass rate is meaningful.

## Stage Sensitivity

```bash
CLOUD_MODEL=Qwen/Qwen2.5-14B-Instruct \
python3 scripts/run_swe_testrepo_observation1.py \
  --repo-path ~/test-repo \
  --timeout-s 180 \
  --max-tokens 1024 \
  --analysis-max-tokens 256 \
  --patch-max-tokens 768 \
  --repair-max-tokens 768
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
