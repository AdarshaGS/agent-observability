"""Per-session report: computed numbers first, narrative second.

Every figure comes from the ledger. An LLM may later *explain* this dict (see analyzer_input), but it
never reconstructs usage from raw logs.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .ledger import Ledger
from .metrics import context_growth, session_summary
from .pricing import PriceTable
from .routing import RoutingConfig, analyze_task_routing, overpowered_findings
from .waste import WasteConfig, analyze_waste

RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def top_consumers(ledger: Ledger, session_id: str, include_children: bool = True) -> list[dict]:
    """Several lenses on where tokens went. Lenses overlap (a tool result is both a 'context kind' and a
    'tool'), so do not add them together."""
    ids = ledger.tree_ids(session_id) if include_children else [session_id]
    calls = [c for c in ledger.calls_for(ids) if c["status"] != "blocked_budget"]
    segs = ledger.segments_for(ids)
    tools = {t["tool_call_id"]: t["tool_name"] for t in ledger.tool_calls_for(ids)}
    total = sum(c["input_fresh"] + c["cache_read"] + c["cache_write"] + c["output_tokens"] for c in calls) or 1
    by_kind: dict[str, int] = defaultdict(int)
    by_tool: dict[str, int] = defaultdict(int)
    for c in calls:
        for s in segs.get(c["call_id"], []):
            by_kind[s["kind"]] += s["tokens"]
            if s["source_ref"] in tools:
                by_tool[tools[s["source_ref"]]] += s["tokens"]
    out_tokens = sum(c["output_tokens"] for c in calls)
    rows = [dict(lens="context_kind", name=k, tokens=v) for k, v in by_kind.items()]
    rows += [dict(lens="tool_context", name=k, tokens=v) for k, v in by_tool.items()]
    rows.append(dict(lens="output", name="model_output", tokens=out_tokens))
    for c in sorted(calls, key=lambda c: -(c["input_fresh"] + c["cache_read"] + c["cache_write"] + c["output_tokens"]))[:3]:
        rows.append(dict(lens="call", name=f"{c['call_id']} (iter {c['iteration']}, {c['model_id']})",
                         tokens=c["input_fresh"] + c["cache_read"] + c["cache_write"] + c["output_tokens"]))
    for r in rows:
        r["pct_of_session"] = r["tokens"] / total
    return sorted(rows, key=lambda r: -r["tokens"])


def build_session_report(ledger: Ledger, session_id: str, *, baselines: Optional[dict] = None,
                         waste_config: Optional[WasteConfig] = None, routing_config: Optional[RoutingConfig] = None,
                         pricing: Optional[PriceTable] = None, include_children: bool = True) -> dict:
    summary = session_summary(ledger, session_id, include_children)
    waste = analyze_waste(ledger, session_id, baselines=baselines, config=waste_config,
                          include_children=include_children)
    findings = list(waste.findings)
    routing = []
    if routing_config is not None:
        ids = ledger.tree_ids(session_id) if include_children else [session_id]
        tasks = ledger.tasks_for(ids)
        for tt in sorted({t["task_type"] for t in tasks if t["task_type"]}):
            findings += overpowered_findings(ledger, routing_config, pricing, task_type=tt)
        for t in tasks:
            try:
                routing.append(analyze_task_routing(ledger, t["task_id"], routing_config, pricing))
            except KeyError:
                pass
    token_findings = [f for f in findings if f.unit == "tokens"]
    lo = sum(f.expected_reduction_low for f in token_findings)
    hi = sum(f.expected_reduction_high for f in token_findings)
    total = waste.total_billed or 1
    growth = [dict(g, iterations=[dict(iteration=r["iteration"], input_tokens=r["input_tokens"]) for r in g["iterations"]])
              for g in context_growth(ledger, session_id, include_children)]
    traj = ledger.trajectory(session_id, include_children)
    ranked = sorted(findings, key=lambda f: -f.tokens_wasted)
    causes: list[str] = []
    for f in ranked:
        if f.root_cause not in causes:
            causes.append(f.root_cause)
    return dict(
        summary=summary,
        top_consumers=top_consumers(ledger, session_id, include_children),
        waste=dict(total_billed=waste.total_billed, required_context_tokens=waste.required_context_tokens,
                   unattributed_input_tokens=waste.unattributed_input_tokens, est_error_mean_abs_pct=waste.est_error_mean_abs_pct,
                   notes=waste.notes, coverage=waste.coverage, findings=[f.to_dict() for f in ranked]),
        root_causes=causes,
        recommendations=[dict(category=f.category, action=f.recommendation, tokens=f.tokens_wasted,
                              expected_reduction=dict(low=f.expected_reduction_low, high=f.expected_reduction_high),
                              quality_risk=f.quality_risk, quality_risk_note=f.quality_risk_note) for f in ranked],
        estimated_savings=dict(tokens_low=lo, tokens_high=hi, pct_low=lo / total, pct_high=hi / total,
                               note="Range; categories are mutually exclusive so figures do not double count."),
        quality_risk=max((f.quality_risk for f in ranked if f.tokens_wasted or f.unit == "cost_only"),
                         key=RISK_ORDER.get, default="low"),
        context_growth=growth,
        routing=routing,
        budget=dict(events=traj["budget_events"],
                    trajectory=[dict(seq=s["call"]["seq"], iteration=s["call"]["iteration"], model=s["call"]["model_id"],
                                     status=s["call"]["status"], billed=s["billed"], cumulative=s["cumulative_billed"],
                                     budget_used_after=s["call"]["budget_used_after"]) for s in traj["steps"]]))


def _k(n: float) -> str:
    return f"{n / 1000:.1f}k" if abs(n) >= 1000 else f"{n:.0f}"


def render_text(r: dict) -> str:
    s, w = r["summary"], r["waste"]
    q = s["quality"]["by_method"]
    qtxt = ", ".join(f"{m}={v['mean']:.2f} (n={v['n']})" for m, v in q.items()) or "n/a"
    cost = f"${s['estimated_cost']:.4f}" + (f" (+{s['unpriced_calls']} unpriced calls)" if s["unpriced_calls"] else "")
    L = [f"Session            {s['session_id']}  [{s['terminal_status'] or 'open'}]",
         f"Total Tokens       {s['total_billed_tokens']:,}  (fresh {s['input_fresh']:,} | cache-read {s['cache_read']:,} | "
         f"cache-write {s['cache_write']:,} | out {s['output']:,})",
         f"Estimated Cost     {cost}",
         f"Model(s)           {', '.join(f'{m} x{v['calls']}' for m, v in s['models'].items())}",
         f"Iterations         {s['iterations']}  (retries {s['retries']}, blocked {s['blocked_calls']})",
         f"Tool Calls         {s['tool_calls']}  (memo hits {s['memo_hits']})",
         f"Success            {s['successes']}/{s['tasks']} tasks",
         f"Quality            {qtxt}  coverage {s['quality']['coverage']:.0%}", "", "Top Token Consumers"]
    for i, c in enumerate(r["top_consumers"][:3], 1):
        L.append(f"  {i}. [{c['lens']}] {c['name']}: {_k(c['tokens'])} ({c['pct_of_session']:.0%})")
    L += ["", "Top Waste Sources"]
    fs = [f for f in w["findings"] if f["tokens_wasted"] or f["unit"] == "cost_only"]
    for i, f in enumerate(fs[:3], 1):
        tag = "cost-only" if f["unit"] == "cost_only" else f"{_k(f['tokens_wasted'])} tokens, {f['pct_of_session']:.0%}"
        L.append(f"  {i}. {f['category']} - {tag} [{f['confidence']}]")
    if not fs:
        L.append("  none detected")
    L += ["", "Root Causes"] + [f"  - {c}" for c in r["root_causes"][:5]]
    L += ["", "Recommended Optimizations"]
    for x in r["recommendations"][:5]:
        L.append(f"  - {x['action']}  (risk: {x['quality_risk']})")
    e = r["estimated_savings"]
    L += ["", f"Estimated Token Savings  {_k(e['tokens_low'])} - {_k(e['tokens_high'])}  ({e['pct_low']:.0%} - {e['pct_high']:.0%} of session)",
          f"Quality Risk       {r['quality_risk']}"]
    for n in w["notes"]:
        L.append(f"Note: {n}")
    return "\n".join(L)
