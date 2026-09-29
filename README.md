# agent-observability

Structured token telemetry, a hard session budget, and deterministic waste / routing analysis for AI agents.
Stdlib-only Python (3.10+), SQLite ledger. **Agents emit telemetry through one gateway; code computes every
number; an LLM (optional) only explains the computed findings.**

```
agent ──► Gateway.call() ──► provider          Ledger (SQLite, append-only facts)
            │  reserve budget (pre-flight)       sessions · tasks · routing_decisions · model_calls
            │  record usage/cache/retry/latency  context_segments · tool_calls · quality_evals · budget_events
            └► commit / release budget
                                   │
        waste.py · metrics.py · routing.py · report.py · aggregate.py  ──►  analyzer_input.py ──► LLM (explains)
```

## Quick start

```python
from agent_observability import *
led = Ledger("ledger.db")
gw = Gateway(led, BudgetManager(led, BudgetConfig(limit=300_000)), pricing=my_price_table)
gw.start_session("s1"); gw.start_task("t1", "s1", task_type="refactor", complexity_est="medium")

ctx = ContextBuilder()                       # re-send the SAME segment objects each iteration
ctx.add("system", SYSTEM); ctx.add("user", request)
r = gw.call(CallRequest(session_id="s1", agent_id="a", model_id="m", segments=list(ctx.segments),
                        task_id="t1", iteration=1, prompt_version="v3"), provider_fn)
tcid, seg = gw.record_tool_call(r.call_id, "s1", "grep", args, raw_output, max_inject_tokens=4000)
ctx.add_segment(seg)
print(render_text(build_session_report(led, "s1")))
```

`provider_fn(request, max_output_tokens, policy) -> ProviderResponse(usage=Usage(...))`. Build `Usage` with
`usage_from_anthropic` / `usage_from_openai` (they differ on whether cached tokens are inside `input`).

`python3 examples/simulate.py` shows the whole loop; `python3 -m agent_observability report|trajectory|payload|aggregate|select --db ledger.db`.
Tests: `PYTHONPATH=.:tests python3 -m unittest discover -s tests`.

## UI

```bash
python3 examples/seed_demo.py demo.db                 # synthetic data (skip for your own ledger)
python3 -m agent_observability serve --db demo.db --open   # http://127.0.0.1:8765
```
Overview (fleet KPIs, recurring waste, flags, agent/model tables), Sessions, and a per-session page (budget gauge and
threshold chart, context-by-source per call, waste findings with cause/fix/risk, full call trajectory, routing). Dark/light,
"Live (5s)" refresh. Stdlib server bound to localhost; no build step, no CDN.

## Ingest API (any language) and auth

The server accepts JSON events, so agents in TypeScript, Go, Java etc. can report without Python.

```bash
export OBS_INGEST_TOKEN=$(python3 -m agent_observability token) OBS_READ_TOKEN=$(python3 -m agent_observability token)
python3 -m agent_observability serve --db ledger.db --host 0.0.0.0     # non-loopback refuses to start without tokens
curl -X POST localhost:8765/ingest -H "Authorization: Bearer $OBS_INGEST_TOKEN" -d '{"events":[
 {"type":"session_start","session_id":"s1","budget":{"limit":300000}},
 {"type":"task_start","task_id":"t1","session_id":"s1","task_type":"refactor"},
 {"type":"model_call","call_id":"c1","session_id":"s1","task_id":"t1","agent_id":"a","model_id":"m","iteration":1,
  "usage_format":"anthropic","usage":{"input_tokens":900,"cache_read_input_tokens":4000,"output_tokens":200},
  "segments":[{"kind":"system","text":"..."},{"kind":"tool_result","tokens":1200,"content_hash":"abc"}]},
 {"type":"tool_call","call_id":"c1","tool_name":"grep","args":{"q":"x"},"raw_output_tokens":5000,"injected_tokens":1200},
 {"type":"quality","task_id":"t1","method":"test","score":1},
 {"type":"task_end","task_id":"t1","outcome":"success"},{"type":"session_end","session_id":"s1"}]}'
```

Event types: `session_start`, `session_end`, `task_start`, `task_end`, `routing`, `model_call`, `tool_call`, `quality`.
Every event is idempotent on its id (`session_id`, `task_id`, `call_id`, `tool_call_id`, `decision_id`), so retrying is safe;
send `model_call` before its `tool_call`s (batches are processed in order). A batch returns per-event
`ok | duplicate | error` (HTTP 207 if any failed). Segments take `text` (tokens/hash/fingerprint computed, text discarded) or
`tokens` + optional `content_hash`/`fingerprint`. `usage_format` = `native` | `anthropic` | `openai`. Unknown sessions are created on demand.

**Enforcement vs reporting.** A reported call already happened, so `/ingest` only *accounts* (advisory: the reply carries the
level and policy). For a hard cap the caller must `POST /budget/reserve` first (`{session_id, est_input_tokens,
max_output_tokens}`), which answers 200 with `granted_max_output_tokens` and a `reservation_id`, or **429** when it won't fit.
Send that `reservation_id` on the `model_call` event; unused reservations expire (default 300s) or `POST /budget/release`.
`GET /api/budget/<session>` returns level/remaining/policy. Python callers can use `ObsClient` (`agent_observability.client`).

Auth: bearer tokens, `OBS_TOKEN` (both scopes) or separate `OBS_READ_TOKEN` (dashboard/API) and `OBS_INGEST_TOKEN`
(ingest/budget). Min 16 chars, constant-time compare, `401` missing/invalid, `403` wrong scope. The dashboard shell loads
without a token and prompts for a read token. Limits: no TLS (put it behind a proxy for anything off-machine), no per-tenant
isolation (any read token sees everything), 8 MB body / 1000 events per request.

## Budget (300k default)

Enforced in the gateway, atomically (`BEGIN IMMEDIATE`), shared by subagents through their root session.
Every call **reserves** `est_input × 1.05 + max_output` first; max output shrinks to fit; a call that cannot
fit is refused *before* it is sent. `budget.basis` = `billed` (default, all tokens) or `fresh` (excl. cache reads).

| Level | At | What the system does | What the agent should do |
|---|---|---|---|
| normal | <60% | log | work normally |
| warning | 60% | event + state line | no capability change; targeted tool calls |
| high | 80% | compact history, memoise repeat tools | finish the plan, no new broad exploration |
| critical | 90% | output cap, optional tools off, wrap-up | deliver now, report gaps, save checkpoint |
| hard_stop | 100% | reject all calls, `budget_exhausted` | resume from checkpoint in a new session |

Normal calls may only use `limit − finalization_reserve` (12k); `finalization=True` calls may use the rest so the
agent can always emit a checkpoint. **Waste is degraded first, capability last.** `BudgetPolicy` carries these flags;
the gateway applies the output cap, the agent applies compaction/memoisation (`ContextBuilder.compact/dedupe`, `ToolMemo`).

## What is measured vs inferred

* **Measured**: provider usage (fresh / cache read / cache write / output / reasoning-subset), retries and their chain,
  latency, cost (from *your* versioned price table; unknown model ⇒ NULL + "unpriced", never 0), segment sizes and hashes.
* **Estimated**: per-segment token split (client tokenizer). The residual vs provider input is stored as `est_error`
  and surfaced as a note when > 3%.
* **Heuristic (labelled)**: "was this context ever used" uses hashed distinctive-token fingerprints of context vs later
  outputs. It cannot see paraphrase: treat as a lead. A negative tool result can be valuable and still look unused.

## Waste categories

Every token is attributed to at most one category (findings never double count): `failed_retry_calls`,
`duplicated_context`, `repeated_tool_calls`, `unnecessary_tool_calls`, `irrelevant_context`, `redundant_context`,
`oversized_tool_output`, `oversized_conversation_history`, `context_accumulation`, `verbose_prompts`,
`excessive_iterations`, `unnecessary_output`, `poor_routing`, plus cost-only `cache_miss` and `overpowered_model`.
Each finding carries evidence ids, tokens, % of session, root cause, recommendation, expected-reduction *range*,
and quality risk. `required_context` = everything unclaimed. Baseline-driven detectors need
`compute_baselines(ledger)`; recompute regularly.

## Quality

Methods are ranked `test > assertion > human > user_signal > judge` and **never pooled**. A judge score must name a
judge model; `judge_human_agreement` checks calibration. Success (task outcome) and quality are separate. Quality can arrive
late — evals are append-only.

## Routing

`cheaper_model_evidence` recommends a cheaper model only if, **within the same task type and complexity bucket** and with
`min_n` samples per side, success *and* quality are statistically non-inferior (one-sided). Sparse quality labels ⇒
`insufficient_evidence`. Verdicts are observational (routing was not randomised) — confirm with replay/A-B.
`RoutingConfig.tiers` maps model → highest complexity it handles (simple=0 … large=3).

## Known limits (be honest with yourself)

* Subagent/parallel calls are ordered by ledger insertion, not causal time; use `parent_call_id` when you need the tree.
* Fingerprints are bottom-k samples: high-volume outputs (>2k distinct identifiers) may under-report references.
* `context_accumulation` and `oversized_conversation_history` savings assume state capture is correct — **validate any
  compaction by replaying sessions and comparing outcomes**, not by token savings.
* Content is never stored. If you need replay, keep prompts in your own (access-controlled) store keyed by `call_id`.
