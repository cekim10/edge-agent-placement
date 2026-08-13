# Observation 1: AppWorld Stage Sensitivity

Goal:

```text
Agent workflow stages exhibit heterogeneous sensitivity to edge execution.
```

This workload uses the real AppWorld benchmark:

```text
https://github.com/StonyBrookNLP/appworld
```

AppWorld provides a sandbox world of day-to-day apps and APIs. The official
task evaluator checks final database state and answer correctness, so this
avoids LLM-as-judge scoring.

Fixed workflow:

```text
task_analysis -> api_planning -> code_generation -> execution_verification
```

Interpretation:

- `task_analysis`: task understanding and high-level plan.
- `api_planning`: app/API selection before writing executable code.
- `code_generation`: executable `world.execute(...)` code.
- `execution_verification`: one repair step after execution/evaluation.

Final quality:

```text
AppWorld evaluator success rate
```

## Setup

On the GPU server, use a Python 3.11+ environment:

```bash
uv venv --python 3.11 .venv-appworld
source .venv-appworld/bin/activate
uv pip install appworld
appworld install
appworld download data
appworld verify tasks
```

If the `appworld` command is not on `PATH`, use:

```bash
python -m appworld.cli install
python -m appworld.cli download data
python -m appworld.cli verify tasks
```

## Local Harness Smoke

This does not require AppWorld:

```bash
python3 scripts/run_appworld_observation1.py --mock --limit 2 --only-placement all_cloud
```

## Clean All-Cloud Pilot

Start with a tiny dev-set pilot. Do not run placement variants until this is
stable.

First check prompt sizes without LLM calls:

```bash
python3 scripts/run_appworld_observation1.py \
  --appworld-root ~/appworld-pip \
  --dataset-name dev \
  --limit 3 \
  --dry-run-prompts
```

```bash
CLOUD_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ \
python3 scripts/run_appworld_observation1.py \
  --dataset-name dev \
  --limit 3 \
  --only-placement all_cloud \
  --timeout-s 180 \
  --analysis-max-tokens 160 \
  --api-plan-max-tokens 160 \
  --code-max-tokens 384 \
  --verify-max-tokens 256 \
  --dump-prompts
```

Go/no-go:

```text
all-cloud success_rate < 0.80: do not run placement variants
all-cloud success_rate >= 0.80: run stage sensitivity
```

## Stage Sensitivity

```bash
CLOUD_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ \
python3 scripts/run_appworld_observation1.py \
  --dataset-name dev \
  --limit 10 \
  --timeout-s 180 \
  --analysis-max-tokens 160 \
  --api-plan-max-tokens 160 \
  --code-max-tokens 384 \
  --verify-max-tokens 256 \
  --dump-prompts
```

Placements:

```text
cloud,cloud,cloud,cloud
edge,cloud,cloud,cloud
cloud,edge,cloud,cloud
cloud,cloud,edge,cloud
cloud,cloud,cloud,edge
edge,edge,edge,edge
```

Expected Observation 1 signal:

```text
Moving different stages to edge causes non-uniform final success-rate drops.
```

## Notes

This harness intentionally does not expose AppWorld ground-truth solution code
or evaluation code to the model. It uses `world.task.instruction`,
`world.task.supervisor`, `world.task.app_descriptions`, and a compact API-doc
preview derived from selected apps.
