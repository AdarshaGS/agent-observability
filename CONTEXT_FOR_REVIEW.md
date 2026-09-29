# agent-observability — current state (handoff for an external reviewer)

You are being asked to review this project and suggest what to add or change. Everything below is factual as of 2026-09-29. Where something is unverified or a known weakness, it is labelled. Please answer the questions in the last section, and keep suggestions concrete (file/module level), not generic.

## 1. What it is

A local, stdlib-only Python (3.10+, no third-party dependencies) service that records token usage for AI agents in a SQLite ledger, enforces an optional hard token budget, detects wasted tokens deterministically, and shows it in a small web UI. Design rule: **code computes every number; an LLM (optional) may only explain already-computed findings.** ~3,650 lines of Python in `agent_observability/`, one 415-line single-file HTML UI (no build step, no CDN), 59 unit tests (all passing).

Current real usage: it runs on the author's laptop (`--collect-claude-code`) and passively ingests Claude Code transcripts from `~/.claude/projects/**`. The live ledger holds ~56 sessions and ~5,950 model calls. **Every call is currently "unpriced"** (no price table is configured), so all dollar figures are opt-in estimates.

## 2. Architecture

```
agent ──► Gateway.call() ──► provider         Ledger (SQLite, WAL, append-only facts)
            │ reserve budget (pre-flight)       sessions · tasks · routing_decisions · model_calls
            │ record usage/cache/retry/latency  context_segments · tool_calls · quality_evals
            └► commit / release budget          budget_state/events/reservations · collector_files
                              │
   waste.py · metrics.py · routing.py · report.py · aggregate.py ──► analyzer_input.py ──► (optional) LLM explains
```

Three ways data gets in:
1. **In-process:** Python agents call `Gateway.call()` (wraps the provider call, reserves/commits budget, records usage).
2. **HTTP ingest:** `POST /ingest` with JSON events (any language). Event types: `session_start`, `session_end`, `task_start`, `task_end`, `routing`, `model_call`, `tool_call`, `quality`. Idempotent on ids; batch limit 1000 events / 8 MB.
3. **Pull collector (`collect_claude.py`):** on a timer, reads Claude Code transcript JSONL files (and `<session>/subagents/agent-*.jsonl`) and turns them into ledger events. Stores metadata only (token usage, model, timestamps, tool names, result sizes, hashes) — never prompt/response/tool text. Idempotent (call_id = API message id; unchanged files skipped). Subagent sessions are linked to their parent via `parent_session_id`.

### Modules (lines)
`ingest` 391 (event validation + handling) · `waste` 433 (detectors) · `gateway` 247 (call wrapper, `ContextBuilder`, `ToolMemo`) · `budget` 250 (atomic reservations, levels) · `server` 253 (stdlib `ThreadingHTTPServer`, routes, auth) · `metrics` 218 · `ledger` 207 (SQLite access) · `schema` 191 · `collect_claude` 187 · `aggregate` 184 (cross-session analysis) · `routing` 168 · `report` 135 · `models` 146 · `compare` 102 (new) · `cli` 103 · `pricing` 47 · `auth` 63 · `quality` 73 · `analyzer_input` 81 · `tokens` 54 · `stats` 50 · `client` 50.

## 3. Key behaviours

**Budget (default 300k tokens/session tree, shared by subagents via the root session).** Enforced atomically (`BEGIN IMMEDIATE`). Every call *reserves* `est_input × 1.05 + max_output` first; max output shrinks to fit; a call that can't fit is refused before sending. Levels: normal <60%, warning 60%, high 80% (compact/memoise), critical 90% (output cap, wrap-up), hard_stop 100% (reject). A 12k `finalization_reserve` lets an agent always emit a checkpoint. Basis is `billed` (all tokens) or `fresh` (excludes cache reads; the author's server runs `fresh`). Note: for collected Claude Code transcripts the budget is *accounting only* — the collector cannot block calls that already happened.

**Waste detection (deterministic; each token attributed to at most one category):** `failed_retry_calls`, `duplicated_context`, `repeated_tool_calls`, `unnecessary_tool_calls`, `irrelevant_context`, `redundant_context`, `oversized_tool_output`, `oversized_conversation_history`, `context_accumulation`, `verbose_prompts`, `excessive_iterations`, `unnecessary_output`, `poor_routing`, plus cost-only `cache_miss`, `overpowered_model`. Each finding has evidence ids, tokens, % of session, cause, recommendation, expected-reduction *range*, quality risk, and a `measured` vs `heuristic` label. "Was this context used" relies on hashed-token fingerprints (bottom-k samples): blind to paraphrase, treat as a lead.

**Routing:** `cheaper_model_evidence` recommends a cheaper model only if, within the same task type + complexity bucket and with `min_n` samples per side, success and quality are statistically non-inferior (one-sided). Observational, not randomised. Sparse quality labels ⇒ `insufficient_evidence`.

**Quality:** methods ranked `test > assertion > human > user_signal > judge`, never pooled; judge scores must name a judge model; append-only.

**Auth:** bearer tokens (`OBS_TOKEN`, or separate `OBS_READ_TOKEN` / `OBS_INGEST_TOKEN`, ≥16 chars, constant-time compare). Non-loopback bind refuses to start without tokens. No TLS, no per-tenant isolation.

**Pricing:** no prices hardcoded. A `PriceTable` you supply (per MTok, versioned); unknown model ⇒ cost NULL / "unpriced", never 0.

## 4. HTTP surface
`GET /` (UI) · `GET /healthz` · `GET /api/overview` · `GET /api/sessions` · `GET /api/session/<id>` · `GET /api/budget/<session>` · `GET /api/compare` (new) · `POST /ingest` · `POST /budget/reserve` · `POST /budget/release`. CLI: `serve`, `report`, `trajectory`, `payload`, `aggregate`, `select`, `token`, `compare-tag`, `compare`.

## 5. Newest addition: Compare (cohort comparison)
Purpose: compare token use/cost of work done **with subagents vs inline**, normalised per unit of work (per GitHub issue).
- Tables `cohorts(cohort_id, arm, label, units, unit_label, note)` and `cohort_sessions(cohort_id, session_id)`, created lazily.
- A cohort = one piece of work = root session(s) + all their subagent sessions. Tagged **manually** via `compare-tag`.
- `/api/compare` and UI page `#/compare` show per-unit calls, output, cache-read, billed tokens, est. cost, per-cohort table, and a fixed caveats box. Cost uses `compare_prices.json` (next to the ledger) if present, else a clearly-labelled **ASSUMED** placeholder Sonnet-class price ($3 / $15 / $0.30 / $3.75 per MTok in/out/cache-read/cache-write).

First result (a case study, NOT a controlled experiment): one milestone built with 2 subagents (3 issues) vs four earlier milestones built single-thread (27 issues), same model. Per issue, with agents vs inline: calls 32.7 vs 38.8 (−16%), output tokens 28.2k vs 21.1k (+33%), cache-read 3.1M vs 8.7M (−64%), billed tokens 3.2M vs 8.7M (−63%), est. cost ~$1.72 vs ~$3.11 (−45%, at placeholder prices). Interpretation: the gap comes from cache re-reads of a growing single-thread context; subagents keep exploration out of the main context. Confounders: different, larger baseline milestones; the agent run had a pre-written API contract; n=1 on the agent side; one baseline session (M15) failed to backfill.

## 6. Known limitations / honest gaps
- No real price table configured anywhere; all $ are estimates.
- Cohort tagging is manual; no automatic detection of "this session used agents".
- Subagent *type* (general-purpose/Explore/…) is not recorded, only `agent_id="subagent"`.
- Compare has no quality dimension (no check that cheaper output was equally correct) and no statistical treatment (n is tiny, no confidence intervals).
- Collector budget is accounting-only for Claude Code (cannot enforce).
- Subagent/parallel call ordering is by ledger insertion order, not causal time.
- `est_error` (client tokenizer vs provider) is stored and flagged >3%, but the estimator is a heuristic.
- SQLite with a threaded stdlib server: fine for one machine; a second process writing concurrently (I did a backfill this way) works via WAL but isn't a designed path.
- Fingerprint-based "unused context" detection under-reports for high-volume outputs (>2k distinct identifiers).
- UI is one hand-written HTML file; no pagination for large ledgers verified.
- No CI, no packaging beyond `pyproject.toml`, git repo has no commits yet.

## 7. Questions for you (please answer these)
1. What are the top 3 additions that would make the with-agents vs inline comparison scientifically credible (design of a matched A/B, metrics, statistics)? Be specific about what the tool should record/compute.
2. How should quality be captured so "cheaper" can't silently mean "worse" (which existing `quality_evals` fields to reuse, what new signals)?
3. Should cohort assignment be automated? Propose a rule using data already in the ledger (e.g. presence of subagent sessions under a root).
4. Which of the waste detectors are most likely to produce false positives on Claude Code transcripts, and how would you validate them?
5. What should be added for cost accuracy given prompt-cache behaviour (TTL, cache-write vs read pricing, per-model price versioning)?
6. Any correctness, security, or scalability risks in the design above that I should fix before relying on it? Rank by severity.
7. What is missing that a team running many agents would expect (alerts, per-agent-type breakdown, export, retention)?
