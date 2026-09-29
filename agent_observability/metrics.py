"""Deterministic metrics: session summaries, per-agent/model/task-type aggregates, context growth.

Conventions that avoid misleading numbers:
  * 'tokens per successful task' divides ALL tokens (failures and retries included) by successes.
  * Multi-agent tasks count once per participating agent (tokens split by agent, outcome shared).
  * Never compare across task types without a difficulty control; group by task_type.
  * Success and quality are separate. Quality is per-method, never pooled.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .ledger import Ledger
from .models import INFRA_ERROR_CLASSES
from .quality import quality_by_task, summarize
from .stats import mean, median, slope

BLOCKED = "blocked_budget"


def _billed(c: dict) -> int:
    return c["input_fresh"] + c["cache_read"] + c["cache_write"] + c["output_tokens"]


def _cost(c: dict) -> Optional[float]:
    if c["cost_in"] is None or c["cost_out"] is None:
        return None
    return c["cost_in"] + c["cost_out"]


def context_growth(ledger: Ledger, session_id: str, include_children: bool = True) -> list[dict]:
    """Per (task, agent) thread: input tokens per iteration and WHAT grew, by segment kind and by
    whether the block was new this call or carried forward. Answers 'why did 20k become 80k'."""
    ids = ledger.tree_ids(session_id) if include_children else [session_id]
    calls = [c for c in ledger.calls_for(ids) if c["status"] != BLOCKED and c["attempt_no"] == 1]
    segs = ledger.segments_for(ids)
    threads: dict[tuple, list[dict]] = defaultdict(list)
    for c in calls:
        threads[(c["session_id"], c["task_id"], c["agent_id"])].append(c)
    out = []
    for th, cs in threads.items():
        rows, prev_kinds = [], {}
        for c in cs:
            kinds: dict[str, int] = defaultdict(int)
            new_kinds: dict[str, int] = defaultdict(int)
            for s in segs.get(c["call_id"], []):
                kinds[s["kind"]] += s["tokens"]
                if s["origin_call_id"] == c["call_id"]:
                    new_kinds[s["kind"]] += s["tokens"]
            delta = {k: kinds[k] - prev_kinds.get(k, 0) for k in set(kinds) | set(prev_kinds)}
            rows.append(dict(call_id=c["call_id"], iteration=c["iteration"],
                             input_tokens=c["input_fresh"] + c["cache_read"] + c["cache_write"],
                             by_kind=dict(kinds), new_by_kind=dict(new_kinds), delta_by_kind=delta))
            prev_kinds = dict(kinds)
        ys = [r["input_tokens"] for r in rows]
        growth: dict[str, int] = defaultdict(int)
        for r in rows[1:]:
            for k, v in r["delta_by_kind"].items():
                growth[k] += v
        dominant = max(growth, key=growth.get) if growth else None
        out.append(dict(session_id=th[0], task_id=th[1], agent_id=th[2], slope_tokens_per_iteration=slope(ys),
                        first_input=ys[0] if ys else None, last_input=ys[-1] if ys else None,
                        growth_by_kind=dict(growth), dominant_growth_kind=dominant, iterations=rows))
    return out


def session_summary(ledger: Ledger, session_id: str, include_children: bool = True) -> dict:
    ids = ledger.tree_ids(session_id) if include_children else [session_id]
    all_calls = ledger.calls_for(ids)
    calls = [c for c in all_calls if c["status"] != BLOCKED]
    tasks = ledger.tasks_for(ids)
    tools = ledger.tool_calls_for(ids)
    q = quality_by_task(ledger, [t["task_id"] for t in tasks])
    costs = [_cost(c) for c in calls]
    unpriced = sum(1 for x in costs if x is None)
    outcomes = [t["outcome"] for t in tasks]
    models: dict[str, dict] = {}
    for c in calls:
        m = models.setdefault(c["model_id"], dict(calls=0, tokens=0, cost=0.0))
        m["calls"] += 1
        m["tokens"] += _billed(c)
        m["cost"] += _cost(c) or 0.0
    sess = ledger.one("SELECT * FROM sessions WHERE session_id=?", (session_id,))
    return dict(
        session_id=session_id, terminal_status=sess["terminal_status"] if sess else None,
        total_billed_tokens=sum(_billed(c) for c in calls),
        input_fresh=sum(c["input_fresh"] for c in calls), cache_read=sum(c["cache_read"] for c in calls),
        cache_write=sum(c["cache_write"] for c in calls), output=sum(c["output_tokens"] for c in calls),
        reasoning=sum(c["reasoning_tokens"] for c in calls),
        estimated_cost=sum(x for x in costs if x is not None), unpriced_calls=unpriced,
        models=models, model_calls=len(calls), blocked_calls=len(all_calls) - len(calls),
        iterations=sum(1 for c in calls if c["attempt_no"] == 1), retries=sum(1 for c in calls if c["attempt_no"] > 1),
        tool_calls=len([t for t in tools if t["status"] != "memo_hit"]),
        memo_hits=len([t for t in tools if t["status"] == "memo_hit"]),
        tasks=len(tasks), successes=sum(1 for o in outcomes if o == "success"),
        success=(all(o == "success" for o in outcomes) if outcomes and all(outcomes) else None),
        quality=summarize(q, len(tasks)), quality_by_task={t: dict(method=e["method"], score=e["score"], confidence=e["confidence"]) for t, e in q.items()},
        latency_ms_total=sum(c["latency_ms"] or 0 for c in calls))


def task_rows(ledger: Ledger, session_ids: Optional[list[str]] = None, *, since: Optional[float] = None,
              until: Optional[float] = None) -> list[dict]:
    """One row per (task, agent): all tokens spent by that agent on that task, plus the task outcome."""
    where, params = [], []
    if session_ids is not None:
        where.append(f"c.session_id IN ({','.join('?' * len(session_ids))})")
        params += session_ids
    if since is not None:
        where.append("c.ts >= ?")
        params.append(since)
    if until is not None:
        where.append("c.ts < ?")
        params.append(until)
    calls = ledger.query(f"SELECT c.* FROM model_calls c {'WHERE ' + ' AND '.join(where) if where else ''} ORDER BY c.seq", params)
    calls = [c for c in calls if c["status"] != BLOCKED and c["task_id"]]
    tids = sorted({c["task_id"] for c in calls})
    tmap = {}
    for i in range(0, len(tids), 500):
        chunk = tids[i:i + 500]
        for t in ledger.query(f"SELECT * FROM tasks WHERE task_id IN ({','.join('?' * len(chunk))})", chunk):
            tmap[t["task_id"]] = t
    q = quality_by_task(ledger, tids)
    tool_counts = {r["call_id"]: r["n"] for r in ledger.query("SELECT call_id, COUNT(*) n FROM tool_calls WHERE status!='memo_hit' GROUP BY call_id")}
    by: dict[tuple, list[dict]] = defaultdict(list)
    for c in calls:
        by[(c["task_id"], c["agent_id"])].append(c)
    rows = []
    for (tid, agent), cs in by.items():
        t = tmap.get(tid, {})
        inp = sum(c["input_fresh"] + c["cache_read"] + c["cache_write"] for c in cs)
        cache_r = sum(c["cache_read"] for c in cs)
        costs = [_cost(c) for c in cs]
        first = [c for c in cs if c["attempt_no"] == 1]
        ys = [c["input_fresh"] + c["cache_read"] + c["cache_write"] for c in first]
        model_counts: dict[str, int] = defaultdict(int)
        for c in cs:
            model_counts[c["model_id"]] += 1
        retries = [c for c in cs if c["attempt_no"] > 1]
        rows.append(dict(
            task_id=tid, agent_id=agent, task_type=t.get("task_type"), complexity=t.get("complexity_est"),
            outcome=t.get("outcome"), tokens=sum(_billed(c) for c in cs), input=inp,
            output=sum(c["output_tokens"] for c in cs), cache_read=cache_r,
            cost=None if any(x is None for x in costs) else sum(costs), iterations=len(first),
            tool_calls=sum(tool_counts.get(c["call_id"], 0) for c in cs), calls=len(cs), retries=len(retries),
            infra_retries=sum(1 for c in retries if _prev_err(cs, c) in INFRA_ERROR_CLASSES),
            failures=sum(1 for c in cs if c["status"] == "failure"), growth_slope=slope(ys) if len(ys) >= 3 else None,
            primary_model=max(model_counts, key=model_counts.get), models=dict(model_counts),
            prompt_version=cs[0]["prompt_version"], quality=(q.get(tid) or {}).get("score"),
            quality_method=(q.get(tid) or {}).get("method"), started=cs[0]["ts"]))
    return rows


def _prev_err(cs: list[dict], c: dict) -> Optional[str]:
    prev = next((p for p in cs if p["call_id"] == c["retry_of_call_id"]), None)
    return prev["error_class"] if prev else None


def aggregate_rows(rows: list[dict]) -> dict:
    """Metrics for a set of (task, agent) rows."""
    n = len(rows)
    if not n:
        return dict(tasks=0)
    succ = [r for r in rows if r["outcome"] == "success"]
    finished = [r for r in rows if r["outcome"]]
    tot_tokens = sum(r["tokens"] for r in rows)
    tot_in, tot_out = sum(r["input"] for r in rows), sum(r["output"] for r in rows)
    costs = [r["cost"] for r in rows if r["cost"] is not None]
    tot_cost = sum(costs) if len(costs) == n else None
    calls = sum(r["calls"] for r in rows)
    quals = [r for r in rows if r["quality"] is not None]
    per_method: dict[str, list[float]] = defaultdict(list)
    for r in quals:
        per_method[r["quality_method"]].append(r["quality"])
    success_rate = len(succ) / len(finished) if finished else None
    priced = len(costs) == n
    med_cost = median(costs) if priced else median([r["tokens"] for r in rows])
    # quality-adjusted efficiency: (mean quality x success rate) / median cost per task.
    # Only defined within ONE quality method; we use the most trusted method present.
    qa = None
    if per_method and success_rate is not None and med_cost:
        m = min(per_method, key=lambda k: ("test", "assertion", "human", "user_signal", "judge").index(k))
        qa = mean(per_method[m]) * success_rate / med_cost
    models: dict[str, dict] = defaultdict(lambda: dict(calls=0, tokens=0))
    for r in rows:
        for m, k in r["models"].items():
            models[m]["calls"] += k
        models[r["primary_model"]]["tokens"] += r["tokens"]
    return dict(
        tasks=n, finished=len(finished), successes=len(succ), success_rate=success_rate,
        failure_rate=(1 - success_rate) if success_rate is not None else None,
        avg_tokens_per_task=tot_tokens / n,
        tokens_per_successful_task=(tot_tokens / len(succ)) if succ else None,
        cost_per_successful_task=(tot_cost / len(succ)) if succ and tot_cost is not None else None,
        total_cost=tot_cost, input_output_ratio=(tot_in / tot_out) if tot_out else None,
        avg_iterations=mean([r["iterations"] for r in rows]), avg_tool_calls=mean([r["tool_calls"] for r in rows]),
        retry_rate=sum(r["retries"] for r in rows) / calls if calls else 0.0,
        infra_retry_rate=sum(r["infra_retries"] for r in rows) / calls if calls else 0.0,
        semantic_retry_rate=(sum(r["retries"] - r["infra_retries"] for r in rows)) / calls if calls else 0.0,
        call_failure_rate=sum(r["failures"] for r in rows) / calls if calls else 0.0,
        context_growth_tokens_per_iteration=mean([r["growth_slope"] for r in rows if r["growth_slope"] is not None]),
        cache_hit_ratio=(sum(r["cache_read"] for r in rows) / tot_in) if tot_in else None,
        quality=dict(coverage=len(quals) / n, by_method={m: dict(n=len(v), mean=mean(v)) for m, v in per_method.items()}),
        quality_adjusted_efficiency=qa, qa_unit="per_usd" if priced else "per_token", median_cost_per_task=med_cost,
        model_utilization={m: dict(calls=v["calls"], call_share=v["calls"] / calls if calls else 0.0,
                                   primary_task_tokens=v["tokens"]) for m, v in models.items()})


def agent_metrics(ledger: Ledger, *, group_by: str = "agent_id", task_type: Optional[str] = None,
                  since: Optional[float] = None, until: Optional[float] = None) -> dict[str, dict]:
    """group_by: agent_id | primary_model | task_type | prompt_version"""
    rows = task_rows(ledger, since=since, until=until)
    if task_type:
        rows = [r for r in rows if r["task_type"] == task_type]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r[group_by])].append(r)
    return {k: aggregate_rows(v) for k, v in groups.items()}
