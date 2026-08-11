# edge-agent-placement

Minimal experiment harness for testing whether agent stage placement and state
representation change the quality-latency tradeoff.

Core workflow:

```text
triage -> diagnose -> report
```

Go/no-go starts with the clean all-cloud baseline:

```bash
cd edge-agent-placement
python3 scripts/run_clean_baseline.py
```

For local pipeline validation without vLLM servers:

```bash
python3 scripts/run_clean_baseline.py --mock
```

The clean baseline enforces `accuracy >= 0.80`. If it fails, do not run
placement experiments. Fix scorer, prompts, or data first.

## vLLM endpoints

Defaults match the planned server layout:

```bash
export EDGE_BASE_URL=http://elves-01:8001/v1
export EDGE_MODEL=Qwen/Qwen2.5-3B-Instruct
export CLOUD_BASE_URL=http://elves-02:8002/v1
export CLOUD_MODEL=Qwen/Qwen2.5-7B-Instruct
```

Edge:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 \
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 --port 8001 \
  --enforce-eager --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm
```

Cloud:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 \
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8002 \
  --enforce-eager --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm
```

## Experiments

Clean baseline:

```bash
python3 scripts/run_clean_baseline.py
python3 scripts/run_clean_baseline.py --timeout-s 10
python3 scripts/run_clean_baseline.py --timeout-s 180 --max-tokens 128
```

Stage sensitivity:

```bash
python3 scripts/run_stage_sensitivity.py
```

Context sensitivity:

```bash
python3 scripts/run_context_sensitivity.py
python3 scripts/run_context_sensitivity.py --placement cloud,edge,cloud
```

Outputs are written under `outputs/<run_name>/` as per-incident JSONL records
and aggregate metrics JSON.

## Scoring Contract

Each data row has:

- `must_include`: list of synonym groups. Every group must have at least one
  synonym hit somewhere in the final answer.
- `avoid_main_cause`: if any listed term appears in the `ROOT_CAUSE` section,
  the answer is incorrect.

Metrics:

- `accuracy`: binary exact pass rate
- `partial_score`: fraction of `must_include` groups hit
- `must_hits`: per-group matched synonyms
- `avoid_hits`: forbidden main-cause terms found in the root-cause section
