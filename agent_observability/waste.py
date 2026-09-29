"""Deterministic token-waste detection over the ledger.

Rules of the road:
  * Every token is attributed to AT MOST ONE waste category (detectors run in priority order and
    'claim' context-segment appearances / call-level tokens), so findings never double count.
  * Each finding is labelled `measured` (exact from telemetry) or `heuristic` (inferred). Heuristics
    that depend on 'was this context referenced' use hashed-token fingerprints: a lower bound on
    use / upper bound on waste, blind to paraphrase. Treat them as leads, not verdicts.
  * Waste is not a target: some redundancy is safety. Use per-task-type ceilings validated by replay.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .ledger import Ledger
from .models import STATIC_KINDS
from .stats import mean


@dataclass
class WasteConfig:
    tool_output_cap: int = 8_000        # tokens per injected tool result
    history_cap: int = 30_000           # history tokens per prompt before it counts as oversized
    min_tool_tokens: int = 200          # ignore tiny tool results for 'unnecessary'
    stale_window: int = 3               # calls a block may linger after its last use
    redundancy_jaccard: float = 0.8
    static_prompt_cap: int = 6_000      # system + tool_schema tokens per prompt
    min_cacheable: int = 1_024
    loop_repeat: int = 3                # identical tool signature in N consecutive calls => loop
    output_factor: float = 2.0          # output > factor * baseline p50 => excess
    est_error_alert: float = 0.03


# category -> (unit, confidence, root_cause, recommendation, reduction_low, reduction_high, quality_risk, risk_note)
RULES: dict[str, tuple] = {
    "failed_retry_calls": ("tokens", "measured",
        "Calls failed after consuming tokens (bad args/schema, timeouts, provider errors) and were retried or abandoned.",
        "Fix the failing tool/schema first; validate arguments before calling; use backoff; make calls idempotent.",
        0.5, 0.9, "low", "Removing avoidable failures does not change outputs."),
    "duplicated_context": ("tokens", "measured",
        "Identical content appears more than once in a single prompt (re-injected state, overlapping retrieval).",
        "Dedupe by content hash at context assembly; reference blocks by id.",
        0.9, 1.0, "low", "Exact duplicates carry no extra information."),
    "repeated_tool_calls": ("tokens", "measured",
        "Identical tool call re-issued with an unchanged result (no memo / prior result forgotten).",
        "Memoise by (tool, args, state token) and keep results in structured agent state.",
        0.8, 1.0, "low", "Only safe if the state token really changes when the underlying state changes."),
    "unnecessary_tool_calls": ("tokens", "heuristic",
        "Tool result was never referenced by any later output.",
        "Add planning before exploration; prefer cheap probe tools; tighten tool-use guidance.",
        0.3, 0.7, "medium", "A negative result can still steer the plan; fingerprints cannot see that."),
    "irrelevant_context": ("tokens", "heuristic",
        "Retrieved context was never referenced by any later output.",
        "Raise relevance threshold / rerank; lower top_k; retrieve once and keep in state.",
        0.4, 0.8, "medium", "Tighter retrieval can drop the one supporting passage; measure answer quality on replay."),
    "redundant_context": ("tokens", "heuristic",
        "Near-duplicate content in one prompt (overlapping retrievals or restated tool output).",
        "Merge/dedupe overlapping retrievals; diff tool outputs against the previous call.",
        0.6, 0.9, "low", "Near-duplicates are not identical; verify the dropped copy adds nothing."),
    "oversized_tool_output": ("tokens", "measured",
        "Tool returned more than the per-result cap.",
        "Return filtered / paged / field-selected data; truncate with a 'fetch more' affordance.",
        0.5, 0.9, "low", "Needed detail may sit in the truncated tail; provide a way to page."),
    "oversized_conversation_history": ("tokens", "heuristic",
        "History exceeds the cap with no compaction.",
        "Summarise old turns into structured state; PIN exact values (ids, paths, numbers, original user constraints).",
        0.4, 0.8, "high", "Summaries are lossy; a dropped constraint is the most damaging failure. Validate by replay."),
    "context_accumulation": ("tokens", "heuristic",
        "Tool/retrieval blocks stayed in context long after their last use.",
        "Evict raw results after use; keep extracted facts in agent_state.",
        0.5, 0.9, "medium", "High if agent_state fails to capture facts the model would later need."),
    "verbose_prompts": ("tokens", "heuristic",
        "Static system prompt + tool schemas exceed the cap on every call.",
        "Trim prompt, prune unused tool schemas, cache the static prefix.",
        0.3, 0.6, "medium", "Instruction removal can change behaviour; A/B on replay."),
    "excessive_iterations": ("tokens", "heuristic",
        "Iterations beyond the p90 for this task type, or the same tool call looped.",
        "Add a planning step, loop detection and explicit stop criteria.",
        0.4, 0.8, "medium", "Capping iterations can cut off verification on hard tasks."),
    "unnecessary_output": ("tokens", "heuristic",
        "Output far above the baseline for this task type.",
        "Constrain output format; lower reasoning budget ONLY for task types where quality is proven unaffected.",
        0.3, 0.7, "high", "Trimming reasoning on hard tasks silently reduces accuracy."),
    "poor_routing": ("tokens", "heuristic",
        "Attempts on a model that was later escalated away from: the work was paid for twice.",
        "Improve the complexity estimate; route this task profile straight to the stronger model.",
        0.5, 0.9, "low", "Routing earlier to the stronger model preserves quality."),
    "cache_miss": ("cost_only", "measured",
        "Static prefix not served from cache although caching is active (prefix changed, TTL lapsed, volatile content before the prefix).",
        "Put volatile content after the static prefix, keep tool order stable, refresh before TTL expiry.",
        0.7, 1.0, "low", "Dollar waste only under the 'billed' budget basis; tokens still count against the cap."),
}


@dataclass
class WasteFinding:
    category: str
    unit: str
    confidence: str
    tokens_wasted: int
    pct_of_session: float
    evidence: list[str]
    evidence_count: int
    root_cause: str
    recommendation: str
    expected_reduction_low: int
    expected_reduction_high: int
    quality_risk: str
    quality_risk_note: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["expected_reduction"] = dict(low=d.pop("expected_reduction_low"), high=d.pop("expected_reduction_high"))
        return d


@dataclass
class WasteAnalysis:
    session_id: str
    total_billed: int
    total_input: int
    total_output: int
    segment_tokens: int
    required_context_tokens: int
    unattributed_input_tokens: int
    est_error_mean_abs_pct: Optional[float]
    findings: list[WasteFinding]
    notes: list[str]
    coverage: dict

    def token_findings(self) -> list[WasteFinding]:
        return [f for f in self.findings if f.unit == "tokens"]

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["findings"] = [f.to_dict() for f in self.findings]
        return d


class _Acc:
    __slots__ = ("tokens", "evidence", "n", "details")

    def __init__(self):
        self.tokens, self.evidence, self.n, self.details = 0, [], 0, {}


def _thread(c: dict) -> tuple:
    return (c["session_id"], c["task_id"], c["agent_id"])


def build_reference_index(calls: list[dict], segs: dict[str, list[dict]]):
    """For every tool_result/retrieval segment appearance: was it referenced by any output at or
    after its first appearance, and when was it last used (thread-local call index)?"""
    live = [c for c in calls if c["status"] != "blocked_budget"]
    threads: dict[tuple, list[dict]] = defaultdict(list)
    for c in live:
        threads[_thread(c)].append(c)
    call_k: dict[str, int] = {}
    for cs in threads.values():
        for k, c in enumerate(cs):
            call_k[c["call_id"]] = k
    out_fp = {c["call_id"]: set(c["output_fingerprint"]) for c in live}
    groups: dict[tuple, list[tuple[int, dict]]] = defaultdict(list)
    for c in live:
        for s in segs.get(c["call_id"], []):
            if s["kind"] in ("tool_result", "retrieval"):
                groups[(_thread(c), s["kind"], s["content_hash"], s["source_ref"])].append((call_k[c["call_id"]], s))
    info: dict[str, dict] = {}
    for key, apps in groups.items():
        cs = threads[key[0]]
        fp: set = set()
        for _, s in apps:
            fp |= set(s["fingerprint"])
        first_k = min(k for k, _ in apps)
        assessable = bool(fp) and any(out_fp[cs[j]["call_id"]] for j in range(first_k, len(cs)))
        uses = [j for j in range(first_k, len(cs)) if fp & out_fp[cs[j]["call_id"]]] if assessable else []
        for k, s in apps:
            info[s["segment_id"]] = dict(assessable=assessable, referenced=(bool(uses) if assessable else None),
                                         last_use=(max(uses) if uses else None), k=k)
    return info, call_k, threads


def analyze_waste(ledger: Ledger, session_id: str, *, baselines: Optional[dict] = None,
                  config: Optional[WasteConfig] = None, include_children: bool = True,
                  persist_references: bool = True) -> WasteAnalysis:
    cfg = config or WasteConfig()
    baselines = baselines or {}
    ids = ledger.tree_ids(session_id) if include_children else [session_id]
    all_calls = ledger.calls_for(ids)
    calls = [c for c in all_calls if c["status"] != "blocked_budget"]
    segs = ledger.segments_for(ids)
    tools = ledger.tool_calls_for(ids)
    tasks = {t["task_id"]: t for t in ledger.tasks_for(ids)}
    info, call_k, threads = build_reference_index(calls, segs)

    billed_of = {c["call_id"]: c["input_fresh"] + c["cache_read"] + c["cache_write"] + c["output_tokens"] for c in calls}
    call_by_id = {c["call_id"]: c for c in calls}
    total_billed = sum(billed_of.values())
    total_input = sum(c["input_fresh"] + c["cache_read"] + c["cache_write"] for c in calls)
    total_output = sum(c["output_tokens"] for c in calls)
    seg_tokens = sum(s["tokens"] for c in calls for s in segs.get(c["call_id"], []))

    claimed: set[str] = set()
    call_claim: dict[str, int] = defaultdict(int)
    seg_claimed_tokens = 0
    acc: dict[str, _Acc] = defaultdict(_Acc)
    tool_by_id = {t["tool_call_id"]: t for t in tools}
    segs_by_source: dict[str, list[dict]] = defaultdict(list)
    for cid, ss in segs.items():
        for s in ss:
            if s["source_ref"]:
                segs_by_source[s["source_ref"]].append(s)

    def add(cat: str, tokens: int, ev: str, call_id: str, tool: Optional[str] = None) -> None:
        a = acc[cat]
        a.tokens += tokens
        a.n += 1
        if len(a.evidence) < 20:
            a.evidence.append(ev)
        c = call_by_id[call_id]
        for key, val in (("by_agent", c["agent_id"]), ("by_prompt_version", c["prompt_version"]), ("by_tool", tool)):
            if val is not None:
                d = a.details.setdefault(key, {})
                d[val] = d.get(val, 0) + tokens

    def claim_seg(s: dict, cat: str, tokens: Optional[int] = None) -> int:
        nonlocal seg_claimed_tokens
        if s["segment_id"] in claimed:
            return 0
        t = s["tokens"] if tokens is None else min(tokens, s["tokens"])
        t = min(t, billed_of[s["call_id"]] - call_claim[s["call_id"]])
        if t <= 0:
            return 0
        claimed.add(s["segment_id"])
        call_claim[s["call_id"]] += t
        seg_claimed_tokens += t
        tool = tool_by_id.get(s["source_ref"], {}).get("tool_name") if s["source_ref"] else None
        add(cat, t, f"seg:{s['segment_id']}", s["call_id"], tool)
        return t

    def claim_call(c: dict, cat: str, tokens: Optional[int] = None) -> int:
        rest = billed_of[c["call_id"]] - call_claim[c["call_id"]]
        t = rest if tokens is None else min(tokens, rest)
        if t <= 0:
            return 0
        call_claim[c["call_id"]] += t
        add(cat, t, f"call:{c['call_id']}", c["call_id"])
        return t

    # 1. failed / retried calls -------------------------------------------------------------
    for c in calls:
        if c["status"] == "failure":
            for s in segs.get(c["call_id"], []):
                claim_seg(s, "failed_retry_calls")
            claim_call(c, "failed_retry_calls")

    # 2. exact duplicates within one prompt -----------------------------------------------
    for c in calls:
        seen: set[str] = set()
        for s in segs.get(c["call_id"], []):
            if s["content_hash"] in seen:
                claim_seg(s, "duplicated_context")
            seen.add(s["content_hash"])

    # 3. repeated tool calls (same tool+args, unchanged result) ----------------------------
    seen_calls: set[tuple] = set()
    for t in tools:
        if t["status"] != "ok":
            continue
        key = (t["tool_name"], t["args_hash"], t["result_hash"])
        if key in seen_calls:
            for s in segs_by_source.get(t["tool_call_id"], []):
                claim_seg(s, "repeated_tool_calls")
        seen_calls.add(key)

    # 4. unnecessary tool calls / 5. irrelevant retrieval (both need reference info) ----------
    assessed = unassessable = 0
    for t in tools:
        if t["status"] != "ok" or t["injected_tokens"] < cfg.min_tool_tokens:
            continue
        for s in segs_by_source.get(t["tool_call_id"], []):
            i = info.get(s["segment_id"])
            if i is None or not i["assessable"]:
                unassessable += 1
                continue
            assessed += 1
            if not i["referenced"]:
                claim_seg(s, "unnecessary_tool_calls")
    for c in calls:
        for s in segs.get(c["call_id"], []):
            if s["kind"] == "retrieval":
                i = info.get(s["segment_id"])
                if i is None or not i["assessable"]:
                    unassessable += 1
                elif not i["referenced"]:
                    assessed += 1
                    claim_seg(s, "irrelevant_context")
                else:
                    assessed += 1

    # 6. near-duplicate (redundant) blocks in one prompt -----------------------------------
    for c in calls:
        cand = [s for s in segs.get(c["call_id"], []) if s["kind"] in ("tool_result", "retrieval")
                and s["fingerprint"] and s["segment_id"] not in claimed][:50]
        for a in range(len(cand)):
            for b in range(a + 1, len(cand)):
                x, y = cand[a], cand[b]
                if x["segment_id"] in claimed or y["segment_id"] in claimed or x["content_hash"] == y["content_hash"]:
                    continue
                fx, fy = set(x["fingerprint"]), set(y["fingerprint"])
                if len(fx | fy) and len(fx & fy) / len(fx | fy) >= cfg.redundancy_jaccard:
                    claim_seg(x if x["tokens"] <= y["tokens"] else y, "redundant_context")

    # 7. oversized tool output (excess over cap) -------------------------------------------
    for c in calls:
        for s in segs.get(c["call_id"], []):
            if s["kind"] == "tool_result" and s["tokens"] > cfg.tool_output_cap:
                claim_seg(s, "oversized_tool_output", s["tokens"] - cfg.tool_output_cap)

    # 8. oversized history --------------------------------------------------------------------
    for c in calls:
        hist = [s for s in segs.get(c["call_id"], []) if s["kind"] == "history"]
        excess = sum(s["tokens"] for s in hist) - cfg.history_cap
        for s in hist:
            if excess <= 0:
                break
            excess -= claim_seg(s, "oversized_conversation_history", min(s["tokens"], excess))

    # 9. context accumulation: carried well past last use ------------------------------------
    for c in calls:
        for s in segs.get(c["call_id"], []):
            i = info.get(s["segment_id"])
            if i and i["referenced"] and i["last_use"] is not None and i["k"] > i["last_use"] + cfg.stale_window:
                claim_seg(s, "context_accumulation")

    # 10. verbose static prompts ---------------------------------------------------------------
    for c in calls:
        stat = [s for s in segs.get(c["call_id"], []) if s["kind"] in STATIC_KINDS]
        excess = sum(s["tokens"] for s in stat) - cfg.static_prompt_cap
        for s in stat:
            if excess <= 0:
                break
            excess -= claim_seg(s, "verbose_prompts", min(s["tokens"], excess))

    # 11. excessive iterations: tool loops + baseline p90 ---------------------------------------
    tools_by_call: dict[str, list[dict]] = defaultdict(list)
    for t in tools:
        tools_by_call[t["call_id"]].append(t)
    loop_calls = 0
    for cs in threads.values():
        run, prev = 0, None
        for c in cs:
            sig = frozenset((t["tool_name"], t["args_hash"]) for t in tools_by_call.get(c["call_id"], []) if t["status"] != "memo_hit")
            run = run + 1 if sig and sig == prev else 1
            prev = sig
            if sig and run >= cfg.loop_repeat:
                loop_calls += claim_call(c, "excessive_iterations") > 0
    for c in calls:
        tt = tasks.get(c["task_id"], {}).get("task_type")
        p90 = (baselines.get(tt) or {}).get("iterations_p90")
        if p90 and c["iteration"] and c["iteration"] > p90:
            claim_call(c, "excessive_iterations")

    # 12. unnecessary output vs baseline -----------------------------------------------------------
    for c in calls:
        tt = tasks.get(c["task_id"], {}).get("task_type")
        p50 = (baselines.get(tt) or {}).get("output_p50")
        if p50 and c["output_tokens"] > cfg.output_factor * p50:
            claim_call(c, "unnecessary_output", int(c["output_tokens"] - cfg.output_factor * p50))

    # 13. poor routing: attempts on the model that was escalated away from ----------------------------
    for r in ledger.query(f"SELECT * FROM routing_decisions WHERE session_id IN ({','.join('?' * len(ids))}) "
                          "AND escalated_from IS NOT NULL", ids):
        for c in calls:
            if c["task_id"] == r["task_id"] and c["model_id"] == r["escalated_from"]:
                claim_call(c, "poor_routing")

    # 14. cache misses (cost-only; does not claim tokens) --------------------------------------------------
    notes: list[str] = []
    cache_active = any(c["cache_read"] or c["cache_write"] for c in calls)
    if not cache_active and calls:
        notes.append("No cache activity reported by any call: prefix caching is off or usage fields are not captured.")
    if cache_active:
        for cs in threads.values():
            for c in cs[1:]:
                static = 0
                for s in segs.get(c["call_id"], []):
                    if s["kind"] not in STATIC_KINDS:
                        break
                    static += s["tokens"]
                if static >= cfg.min_cacheable and c["cache_read"] < 0.5 * static:
                    add("cache_miss", int(static - c["cache_read"]), f"call:{c['call_id']}", c["call_id"])

    # persist later_referenced on tool_calls --------------------------------------------------------------
    if persist_references and tools:
        with ledger.transaction():
            for t in tools:
                ss = [info[s["segment_id"]] for s in segs_by_source.get(t["tool_call_id"], []) if s["segment_id"] in info]
                if not ss or not ss[0]["assessable"]:
                    val = None
                else:
                    val = int(any(i["referenced"] for i in ss))
                ledger.execute("UPDATE tool_calls SET later_referenced=? WHERE tool_call_id=?", (val, t["tool_call_id"]))

    # assemble findings ------------------------------------------------------------------------------------
    findings: list[WasteFinding] = []
    for cat, a in acc.items():
        unit, conf, cause, rec, lo, hi, risk, note = RULES[cat]
        findings.append(WasteFinding(
            category=cat, unit=unit, confidence=conf, tokens_wasted=a.tokens,
            pct_of_session=(a.tokens / total_billed) if total_billed else 0.0,
            evidence=a.evidence, evidence_count=a.n, root_cause=cause, recommendation=rec,
            expected_reduction_low=int(a.tokens * lo), expected_reduction_high=int(a.tokens * hi),
            quality_risk=risk, quality_risk_note=note, details=a.details))
    findings.sort(key=lambda f: f.tokens_wasted, reverse=True)

    errs = [abs(c["est_error"]) / (c["input_fresh"] + c["cache_read"] + c["cache_write"])
            for c in calls if c["est_error"] is not None and c.get("est_input_tokens") and (c["input_fresh"] + c["cache_read"] + c["cache_write"])]
    err_mean = mean(errs)
    if not errs and total_input:
        notes.append("No context segments were reported, so only call-level waste (retries, iterations, cache misses, tool "
                     "output size) can be measured; duplication / irrelevant-context analysis needs per-segment data.")
    if err_mean is not None and err_mean > cfg.est_error_alert:
        notes.append(f"Segment token estimates drift {err_mean:.1%} from provider input counts (> {cfg.est_error_alert:.0%}); "
                     "waste attribution is approximate until the estimator is calibrated.")
    return WasteAnalysis(
        session_id=session_id, total_billed=total_billed, total_input=total_input, total_output=total_output,
        segment_tokens=seg_tokens, required_context_tokens=seg_tokens - seg_claimed_tokens,
        unattributed_input_tokens=total_input - seg_tokens, est_error_mean_abs_pct=err_mean, findings=findings,
        notes=notes, coverage=dict(reference_assessed=assessed, reference_unassessable=unassessable))
