# Controlled Access-Control Microbenchmark

This workload is intentionally a controlled microbenchmark, not a real-world
benchmark replacement. The generated request distribution is synthetic. The LLM
calls, stage placement, network path, commit side effects, and state validation
are measured.

## Claim Boundary

The benchmark separates two forms of recoverability:

1. Semantic error propagation before commit.
2. External side-effect recoverability after commit.

Observation 1 uses the first axis. The workflow is:

```text
classify -> plan -> commit
```

`classify` produces the request category and target. `plan` receives only that
stage state plus candidate records, then emits a set of operations. This makes a
bad classification propagate structurally: downstream stages cannot silently
recover by rereading the original request.

Verification is not a fixed workflow stage. Later experiments insert it between
`plan` and `commit` as a decision variable:

```text
none | rule_local | llm_edge | llm_cloud
```

## Operation Types

The generator emits three operation classes:

| Request class | Operation | Recoverability |
| --- | --- | --- |
| grant access | grant_role | reversible |
| revoke access | revoke_role | compensable |
| rotate credential | rotate_credential | irreversible |

For irreversible operations, `/compensate` must fail by contract. This is not a
missing feature; it is the condition that makes speculation unavailable.

## Initial Sweep

The first implementation runs only Observation 1 with no network emulation:

```text
A difficulty: easy, hard
B difficulty: easy, hard
instances/cell: 20
placements: all_cloud, edge_classify, edge_plan, all_edge
```

Primary metrics:

- `classification_accuracy`
- `plan_exact_rate`
- `commit_success_rate`
- `end_to_end_success_rate`
- `schema_violation_rate`

Secondary metric:

- `plan_partial_recall`

Do not tune the difficulty grid after seeing the result. If the phenomenon only
appears in some cells, report those cells rather than reshaping the workload.

## Running

Smoke test without GPUs:

```bash
python3 scripts/run_micro_obs1.py --mock --instances-per-cell 5
```

vLLM run with guided JSON enabled:

```bash
EDGE_MODEL=<edge-model> \
CLOUD_MODEL=<cloud-model> \
EDGE_API_KIND=completions \
CLOUD_API_KIND=completions \
python3 scripts/run_micro_obs1.py \
  --instances-per-cell 20 \
  --timeout-s 120 \
  --classify-max-tokens 64 \
  --plan-max-tokens 96
```

