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

`classify` produces the request category, subject and scope. `plan` receives only
that stage state plus the record tables, never the original request text. This
makes a bad classification propagate structurally: downstream stages cannot
silently recover by rereading the request.

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

## What Each Stage Must Actually Do

An earlier revision of this workload scored 100% in every cell for both models.
The cause was not difficulty but construct validity: the plan prompt contained
the category-to-operation mapping, the ground-truth record was always first in
the candidate list, and the request text spelled out the `user_id` verbatim. The
stages had no work left to do. The task content is now:

`classify` reads only the request and must recover:

- the category, which in hard cells is implied ("their contract ended",
  "the secret was posted in a public channel") rather than named;
- the resource, where hard cells also mention a second resource inside a
  negative clause that must not be acted on;
- the role, which in hard cells is described by task ("needs to be able to push
  changes to") rather than named.

`plan` reads the classification and the record tables, and must:

- resolve a display name to a `user_id`, against a user table that in hard cells
  contains look-alike names (`Noor Kim` vs `Noor Kime` vs `Priya Kim`);
- construct the op set, where `revoke_access` means every role the subject holds
  on that resource, so hard cells need two or three operations rather than one;
- work from a deterministically shuffled record list in which the ground-truth
  rows sit in arbitrary positions.

The plan prompt still states the operation vocabulary and what each category
means, because without that the ground truth would be ambiguous. It does not
state which operation belongs to the classified category.

### Measurement artifacts removed after the first real run

The first GPU run scored 0.55 end-to-end even on the cloud tier, and the failure
dump showed the drop was instrumentation rather than reasoning:

- `revoke_role` with an empty role accounted for 8 of 9 cloud failures. The
  classification carries `role: ""` for revoke, the op schema permitted `""`,
  and the model copied it instead of enumerating assignments. The op schema is
  now `anyOf` over a role-bearing shape (real role required) and a credential
  shape (no role field), so guided decoding cannot produce that op at all.
- Classify outputs on the 7B tier ran whitespace to `max_tokens` and were scored
  as schema violations. `guided_whitespace_pattern` now bounds it.
- Runaway plans were truncated by `max_tokens` and also landed in
  `schema_violation`. `maxItems` bounds the array so a runaway plan terminates
  and is scored as the wrong answer it is.
- `guided_whitespace_pattern` turned out to be accepted and silently ignored by
  the vLLM build in use: served output still arrives pretty-printed, at roughly
  40 tokens per operation. The op-count bound and a 512-token plan budget are
  what actually prevent truncation; the whitespace field is sent but not relied
  on. `GUIDED_DECODING_BACKEND` can be set to try a backend that honours it.

None of these changed the difficulty grid.

## Frozen Difficulty Grid

This grid is fixed before measurement. Results are reported for every cell,
including cells where no effect appears. If the phenomenon shows up in some
cells and not others, that is the result; it is not a reason to reshape the
workload.

| Axis | easy | hard |
| --- | --- | --- |
| A (classify) | category, resource and role stated directly; one resource mentioned | intent implied; second resource in a negative clause; role given as a task description |
| B (plan) | 12 users, 24 assignments, no look-alike names, revoke touches 1 role | 40 users incl. 3 look-alikes, 120 assignments, revoke touches 2-3 roles |

Placements: `all_cloud`, `edge_classify`, `edge_plan`, `all_edge`.

Primary metrics (binary):

- `classification_accuracy`
- `plan_exact_rate`
- `commit_success_rate`
- `end_to_end_success_rate`
- `schema_violation_rate`

Secondary metrics (continuous, for analysis only):

- `plan_partial_recall`
- `plan_partial_precision`
- `unrequested_destructive_rate`, `mean_unrequested_destructive_ops`

The destructive-op metrics exist because of what the first clean run showed: on
`grant_access` requests the 7B tier emitted the correct grant *and* a revoke
nobody asked for, while the 32B tier did not. That is the failure the
recoverability argument turns on -- the weaker tier does not merely miss, it
adds destructive work, and once committed a spurious revoke is compensable at
best and a spurious rotation is not recoverable at all. It is a finding, not an
artifact, and must not be prompted away.

`schema_violation_rate` must be 0 when guided decoding is on. A non-zero value is
a harness bug, not a model result. Validators check only what guided decoding
enforces, so a well-formed wrong answer is scored as wrong rather than as a
schema violation.

## Verifier Ground Truth

Each instance carries a policy and two deliberately invalid plans, so verifier
variants have something to catch and can be told apart:

| Violation | Detectable by rule checker | Why |
| --- | --- | --- |
| `wrong_subject` | no | the ops are policy-clean; only intent reveals the wrong person was targeted |
| `policy_forbidden_role` | yes | grants `admin` on a protected resource |

This is what keeps `rule_local` honest as a baseline: it should be near-perfect
on the second class and blind to the first.

## Observation 2: verification as a decision variable

Difficulty and stage placement are held fixed (defaults: cell A=hard/B=hard,
placement `all_edge`); the sweep is over which verifier runs and where.

| Variant | Where | What it can see |
| --- | --- | --- |
| `none` | -- | commit whatever the planner produced |
| `rule_local` | on device, no model call | the written policy only |
| `llm_edge` | edge tier | request, policy, proposed ops, record tables |
| `llm_cloud` | cloud tier | same as `llm_edge` |

`llm_edge` and `llm_cloud` differ only in placement, so latency between them is
a network fact and accuracy between them is a capacity fact.

### Where the negatives come from

A competent planner leaves a verifier almost nothing to catch, so verifier
accuracy measured on the natural distribution alone is close to meaningless.
`--inject-rate` replaces that fraction of plans with a generator-supplied
invalid plan, split evenly between the two violation classes. The plan call is
still made, so the pipeline's shape and measured latency are unchanged.

The label is the same either way: **a plan deserves approval exactly when its op
set equals the ground truth**. Injection only controls how many negatives there
are and what kind.

### Latency and RTT

Accuracy does not depend on RTT -- same prompts, same models, temperature 0 --
so the sweep runs once per variant and latency is projected:

```text
latency(rtt) = measured_latency + cloud_calls * rtt
```

This is arithmetic over measured per-call latency and a measured count of calls
that crossed to the cloud tier, exact for a sequential pipeline. It is not
simulated model quality. `tc netem` is not available on the cluster (no root),
so `--inject-rtt-ms` exists to validate the projection against real injected
delay rather than to replace it.

Under placement `all_edge` the classify and plan calls cost no RTT and
`llm_cloud` adds exactly one crossing, which is what makes a crossover appear:
`llm_cloud` buys detection accuracy at one RTT, and past some RTT that price
exceeds what the safety constraint is worth.

### Metrics

- `detection_recall` -- of the plans that should have been stopped, how many were
- `false_reject_rate` -- correct plans wrongly blocked
- `unsafe_commit_rate` -- the safety constraint is written against this
- `detection_by_class` -- `rule_local` should be near 1.0 on
  `policy_forbidden_role` and near 0.0 on `wrong_subject`; it is precise and
  inexpressive by construction, and that contrast is why it is in the sweep

A verifier that fails to return a usable answer defaults to approving, so a
broken verifier reads as permissive rather than silently protective.
`verifier_failure_rate` counts those separately.

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
  --timeout-s 180 \
  --classify-max-tokens 64 \
  --plan-max-tokens 192
```

Hard-cell plan prompts run to roughly 1.1k tokens, so serve both tiers with
`--max-model-len 4096` or larger.

Observation 2:

```bash
python3 scripts/run_micro_obs2_verification.py --mock --instances 12   # smoke test

EDGE_MODEL=<edge-model> CLOUD_MODEL=<cloud-model> \
EDGE_API_KIND=completions CLOUD_API_KIND=completions \
python3 scripts/run_micro_obs2_verification.py --instances 20 --inject-rate 0.5
```

`TCP_MSS_CLAMP` (default 1400) caps outgoing TCP segments because the cluster
NICs advertise a 9000-byte MTU while the switch between the nodes only forwards
1500-byte frames; without it any request larger than one segment hangs. The
value used is recorded in each run's `manifest.json`. Set it to 0 once the
network is fixed.
