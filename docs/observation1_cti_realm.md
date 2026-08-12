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
