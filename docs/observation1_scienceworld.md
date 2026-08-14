# Observation 1: ScienceWorld Stage Sensitivity

Goal:

```text
Agent workflow stages exhibit heterogeneous sensitivity to edge execution.
```

This workload uses the real ScienceWorld benchmark:

```text
https://github.com/allenai/ScienceWorld
```

ScienceWorld is a text environment with an objective environment score. The
harness runs a fixed agent loop:

```text
state_abstraction -> subgoal_planning -> action_selection -> progress_verification
```

Metrics:

```text
success_rate
avg_score
avg_normalized_score
avg_steps
invalid_action_rate
```

## Setup

On the GPU server:

```bash
cd ~/edge-agent-placement
uv venv --python 3.11 .venv-scienceworld
source .venv-scienceworld/bin/activate
uv pip install scienceworld
```

Java is required:

```bash
java -version
```

If Java is missing and you have sudo:

```bash
sudo apt-get update
sudo apt-get install -y default-jre
```

## Local Harness Smoke

This does not require ScienceWorld:

```bash
python3 scripts/run_scienceworld_observation1.py --mock --limit 2 --only-placement all_cloud
```

## Inspect Available Tasks

```bash
python3 scripts/run_scienceworld_observation1.py --list-tasks
```

The default task is:

```text
find-non-living-thing
```

## Clean All-Cloud Pilot

Do not run placement variants until all-cloud is stable.

```bash
CLOUD_MODEL=Qwen/Qwen2.5-14B-Instruct \
python3 scripts/run_scienceworld_observation1.py \
  --task-name find-non-living-thing \
  --variations 0-2 \
  --limit 3 \
  --max-steps 12 \
  --only-placement all_cloud \
  --timeout-s 120 \
  --state-max-tokens 120 \
  --plan-max-tokens 120 \
  --action-max-tokens 48 \
  --verify-max-tokens 64 \
  --dump-prompts
```

Go/no-go:

```text
all-cloud success_rate < 0.80: do not run placement variants
all-cloud success_rate >= 0.80: run stage sensitivity
```

Inspect:

```bash
python3 scripts/inspect_scienceworld_run.py \
  outputs/scienceworld_observation1_<timestamp> \
  --label all_cloud
```

## Stage Sensitivity

```bash
EDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
CLOUD_MODEL=Qwen/Qwen2.5-14B-Instruct \
python3 scripts/run_scienceworld_observation1.py \
  --task-name find-non-living-thing \
  --variations 0-4 \
  --limit 5 \
  --max-steps 12 \
  --timeout-s 120 \
  --state-max-tokens 120 \
  --plan-max-tokens 120 \
  --action-max-tokens 48 \
  --verify-max-tokens 64 \
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

Expected signal:

```text
Moving action_selection to edge should increase invalid_action_rate or reduce
avg_score more than moving progress_verification to edge.
```
