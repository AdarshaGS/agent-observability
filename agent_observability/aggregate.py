"""Cross-session analysis: recurring waste, outlier agents, noisy tools, prompt regressions,
iteration hotspots, routing opportunities. All deterministic SQL/Python; the LLM only explains these."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from .ledger import Ledger
from .metrics import aggregate_rows, task_rows
from .pricing import PriceTable
from .routing import RoutingConfig, overpowered_findings
from .stats import median, percentile
from .waste import WasteConfig, analyze_waste


@dataclass
class AggregateConfig:
    min_tasks: int = 10
    outlier_ratio: float = 1.5           # tokens/success vs peer median
    prompt_token_regression: float = 0.15
    prompt_success_drop: float = 0.05
    tool_p95_cap: int = 8_000
    tool_min_yield: float = 0.25
    tool_max_error_rate: float = 0.20
    hotspot_p90_over_p50: float = 2.0
    hotspot_min_p90: int = 6


def compute_baselines(ledger: Ledger, min_n: int = 10) -> dict:
    """Per task_type baselines from SUCCESSFUL tasks, feeding the excessive-iteration and
    unnecessary-output detectors. Recompute periodically; stale baselines create false findings."""
    rows = [r for r in task_rows(ledger) if r["outcome"] == "success" and r["task_type"]]
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["task_type"]].append(r)
    out = {}
    for tt, rs in by.items():
        if len(rs) < min_n:
            continue
        out[tt] = dict(n=len(rs), iterations_p50=median([r["iterations"] for r in rs]),
                       iterations_p90=percentile([r["iterations"] for r in rs], 90),
                       output_p50=median([r["output"] / r["calls"] for r in rs if r["calls"]]),
                       tokens_p50=median([r["tokens"] for r in rs]))
    return out


def aggregate(ledger: Ledger, *, since: Optional[float] = None, until: Optional[float] = None,
              cfg: Optional[AggregateConfig] = None, waste_config: Optional[WasteConfig] = None,
              routing_config: Optional[RoutingConfig] = None, pricing: Optional[PriceTable] = None) -> dict:
    cfg = cfg or AggregateConfig()
    where, params = ["parent_session_id IS NULL"], []
    if since is not None:
        where.append("started_at >= ?")
        params.append(since)
    if until is not None:
        where.append("started_at < ?")
        params.append(until)
    roots = [r["session_id"] for r in ledger.query(f"SELECT session_id FROM sessions WHERE {' AND '.join(where)}", params)]
    baselines = compute_baselines(ledger, cfg.min_tasks)

    # --- recurring waste --------------------------------------------------------------
    cats: dict[str, dict] = {}
    total_billed = 0
    for sid in roots:
        wa = analyze_waste(ledger, sid, baselines=baselines, config=waste_config)
        total_billed += wa.total_billed
        for f in wa.findings:
            c = cats.setdefault(f.category, dict(tokens=0, sessions=0, unit=f.unit, confidence=f.confidence,
                                                 by_agent=defaultdict(int), by_prompt_version=defaultdict(int),
                                                 by_tool=defaultdict(int), recommendation=f.recommendation))
            c["tokens"] += f.tokens_wasted
            c["sessions"] += 1
            for key in ("by_agent", "by_prompt_version", "by_tool"):
                for k, v in f.details.get(key, {}).items():
                    c[key][k] += v
    waste_patterns = sorted(
        ({"category": k, **{kk: (dict(sorted(vv.items(), key=lambda kv: -kv[1])[:5]) if isinstance(vv, dict) else vv)
                            for kk, vv in v.items()}, "pct_of_all_tokens": v["tokens"] / total_billed if total_billed else 0.0,
          "sessions_affected": v["sessions"], "share_of_sessions": v["sessions"] / len(roots) if roots else 0.0}
         for k, v in cats.items()), key=lambda d: -d["tokens"])
    for d in waste_patterns:
        d.pop("sessions", None)

    # --- outlier agents (compare within task_type only) -----------------------------------------------------
    rows = task_rows(ledger, since=since, until=until)
    by_tt_agent: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_tt_agent[(r["task_type"], r["agent_id"])].append(r)
    per_tt: dict[str, dict[str, float]] = defaultdict(dict)
    for (tt, ag), rs in by_tt_agent.items():
        m = aggregate_rows(rs)
        if len(rs) >= cfg.min_tasks and m["tokens_per_successful_task"]:
            per_tt[tt][ag] = m["tokens_per_successful_task"]
    outliers = []
    for tt, agents in per_tt.items():
        if len(agents) < 2:
            continue
        for ag, v in agents.items():
            peers = [x for a, x in agents.items() if a != ag]
            med = median(peers)
            if med and v / med >= cfg.outlier_ratio:
                outliers.append(dict(task_type=tt, agent_id=ag, tokens_per_success=v, peer_median=med, ratio=v / med))
    outliers.sort(key=lambda d: -d["ratio"])

    # --- noisy tools ----------------------------------------------------------------------------------------------
    trows = ledger.query("SELECT tool_name, status, injected_tokens, raw_output_tokens, later_referenced FROM tool_calls WHERE status!='memo_hit'")
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for t in trows:
        by_tool[t["tool_name"]].append(t)
    noisy = []
    for name, ts in by_tool.items():
        if len(ts) < cfg.min_tasks:
            continue
        inj = [t["injected_tokens"] for t in ts]
        assessable = [t for t in ts if t["later_referenced"] is not None]
        yld = (sum(t["later_referenced"] for t in assessable) / len(assessable)) if assessable else None
        err = sum(1 for t in ts if t["status"] == "error") / len(ts)
        p95 = percentile(inj, 95)
        reasons = []
        if p95 > cfg.tool_p95_cap:
            reasons.append(f"p95 injected {p95:.0f} tokens")
        if yld is not None and yld < cfg.tool_min_yield:
            reasons.append(f"yield {yld:.0%}")
        if err > cfg.tool_max_error_rate:
            reasons.append(f"error rate {err:.0%}")
        if reasons:
            noisy.append(dict(tool=name, calls=len(ts), p50_injected=median(inj), p95_injected=p95, yield_=yld,
                              error_rate=err, total_injected=sum(inj), reasons=reasons,
                              suggestion="return structured, field-selected, paginated output with a size cap"))
    noisy.sort(key=lambda d: -d["total_injected"])

    # --- prompt-version regressions ------------------------------------------------------------------------------------
    regress = []
    by_ag_tt: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_ag_tt[(r["agent_id"], r["task_type"])][r["prompt_version"]].append(r)
    for (ag, tt), versions in by_ag_tt.items():
        ordered = sorted(versions.items(), key=lambda kv: min(r["started"] for r in kv[1]))
        for (v0, r0), (v1, r1) in zip(ordered, ordered[1:]):
            if len(r0) < cfg.min_tasks or len(r1) < cfg.min_tasks:
                continue
            m0, m1 = aggregate_rows(r0), aggregate_rows(r1)
            t0, t1 = m0["tokens_per_successful_task"], m1["tokens_per_successful_task"]
            s0, s1 = m0["success_rate"], m1["success_rate"]
            flags = []
            if t0 and t1 and t1 / t0 - 1 >= cfg.prompt_token_regression:
                flags.append(f"tokens/success +{t1 / t0 - 1:.0%}")
            if s0 is not None and s1 is not None and s0 - s1 >= cfg.prompt_success_drop:
                flags.append(f"success -{(s0 - s1) * 100:.0f}pt")
            if flags:
                regress.append(dict(agent_id=ag, task_type=tt, from_version=v0, to_version=v1, flags=flags))

    # --- iteration hotspots ----------------------------------------------------------------------------------------------------
    hot = []
    by_tt: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_tt[r["task_type"]].append(r)
    for tt, rs in by_tt.items():
        if len(rs) < cfg.min_tasks:
            continue
        its = [r["iterations"] for r in rs]
        p50, p90 = median(its), percentile(its, 90)
        if p90 >= cfg.hotspot_min_p90 and p50 and p90 / p50 >= cfg.hotspot_p90_over_p50:
            hot.append(dict(task_type=tt, tasks=len(rs), iterations_p50=p50, iterations_p90=p90,
                            failure_rate=1 - aggregate_rows(rs)["success_rate"] if aggregate_rows(rs)["success_rate"] is not None else None))

    opportunities = []
    if routing_config is not None:
        opportunities = [f.to_dict() for f in overpowered_findings(ledger, routing_config, pricing)]

    return dict(
        window=dict(since=since, until=until, sessions=len(roots), total_billed_tokens=total_billed),
        model_metrics=aggregate_by(rows, "primary_model"), agent_metrics=aggregate_by(rows, "agent_id"),
        waste_patterns=waste_patterns, outlier_agents=outliers, noisy_tools=noisy,
        prompt_regressions=regress, iteration_hotspots=hot, routing_opportunities=opportunities,
        baselines=baselines)


def aggregate_by(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r[key])].append(r)
    return {k: aggregate_rows(v) for k, v in groups.items()}
