# Observation 1: CTI-REALM Stage Sensitivity

Goal:

```text
Observation 1: Agent workflow stages exhibit heterogeneous sensitivity to edge execution.
```

This harness uses CTI-REALM task metadata from ACESEvals and runs a deterministic
five-stage detection-engineering workflow:

```text
C0 CTI analysis
C1 MITRE mapping
C2 Data-source discovery
C3 KQL development
C4 Detection rule generation
```

Placements:

```text
cloud,cloud,cloud,cloud,cloud
edge,cloud,cloud,cloud,cloud
cloud,edge,cloud,cloud,cloud
cloud,cloud,edge,cloud,cloud
cloud,cloud,cloud,edge,cloud
cloud,cloud,cloud,cloud,edge
```

## Setup

Clone and initialize ACESEvals on the GPU server:

```bash
git clone https://github.com/microsoft/ACESEvals.git
cd ACESEvals
uv sync --all-extras
uv run inspect eval domains/cti_realm --model openai/azure/gpt-4.1 -T dataset=cti_realm_25 --limit 1
```

The first run downloads CTI-REALM data into:

```text
ACESEvals/domains/cti_realm/data/
```

## Run Observation 1 Prototype

From this repo:

```bash
python3 scripts/run_cti_observation1.py \
  --aces-root ~/ACESEvals \
  --dataset-size 25 \
  --limit 5 \
  --timeout-s 180 \
  --max-tokens 256
```

Fast server smoke test. This only checks that all five stages complete; do
not interpret the quality score from this run.

```bash
python3 scripts/run_cti_observation1.py \
  --aces-root ~/ACESEvals \
  --dataset-size 25 \
  --limit 1 \
  --only-placement all_cloud \
  --timeout-s 300 \
  --max-tokens 32
```

If the final rule-generation stage stalls, retry the smoke with a smaller
generation cap:

```bash
python3 scripts/run_cti_observation1.py \
  --aces-root ~/ACESEvals \
  --dataset-size 25 \
  --limit 1 \
  --only-placement all_cloud \
  --timeout-s 300 \
  --max-tokens 16
```

Quality run after smoke passes:

```bash
python3 scripts/run_cti_observation1.py \
  --aces-root ~/ACESEvals \
  --dataset-size 25 \
  --limit 1 \
  --only-placement all_cloud \
  --timeout-s 300 \
  --max-tokens 64 \
  --cti-max-tokens 64 \
  --mitre-max-tokens 16 \
  --data-source-max-tokens 32 \
  --proxy-final-from-c2
```

If all-cloud proxy quality is still near zero, stop and inspect the JSONL output
before running the full placement matrix. Observation 1 is only meaningful after
the all-cloud run produces non-zero MITRE/data-source hits.

The recommended Phase-A quality run uses `--proxy-final-from-c2`: C0-C2 are
model-generated, while C3/C4 are compact deterministic pass-through outputs.
Short JSON stages should use stage-specific token caps such as
`--mitre-max-tokens 16`. C1 also defaults to a direct objective-only classifier
prompt instead of consuming C0 output because some vLLM/Qwen runs stall on the
longer chained prompt. C2 similarly defaults to an objective-only data-source
classifier prompt; pass `--mitre-use-prior` or `--data-source-use-prior` only
when debugging those paths.
This avoids Qwen/vLLM stalls in long KQL/rule-generation completions while
preserving the MITRE/data-source scoring signal needed for the first
stage-sensitivity figure. The proxy finalizer does not read ground truth; it
only parses MITRE IDs and data-source names from prior model outputs. C1 is
given a platform-scoped dataset-wide MITRE technique catalog with stable
technique names and C2 is given a dataset-wide data-source catalog, analogous to
bounded analyst lookup/tool context; neither stage receives per-task ground
truth. Do not use this as the final CTI-REALM C4 result.

Local smoke test without vLLM:

```bash
python3 scripts/run_cti_observation1.py \
  --cti-data-dir ~/ACESEvals/domains/cti_realm/data \
  --dataset-size 25 \
  --limit 2 \
  --mock
```

Outputs:

```text
outputs/cti_observation1_<timestamp>/
  all_cloud.jsonl
  edge_cti_analysis.jsonl
  edge_mitre_mapping.jsonl
  edge_data_source_discovery.jsonl
  edge_kql_development.jsonl
  edge_rule_generation.jsonl
  metrics.json
  metrics.csv
```

## Metric Caveat

This is a Phase-A characterization prototype. It uses CTI-REALM ground-truth
MITRE techniques, data sources, and regex-pattern terms to compute a lightweight
proxy score:

```text
proxy_quality = 0.30 * MITRE Jaccard
              + 0.30 * data-source recall
              + 0.40 * detection-term recall
```

This is not the official CTI-REALM C4 score. Use this to detect whether
stage-level edge degradation exists. If the signal is strong, the next step is
to wire the same stage-routing idea into ACESEvals/Inspect so the paper figure
uses official C0-C4 checkpoint scores.
