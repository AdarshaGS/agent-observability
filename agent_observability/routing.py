"""Model-routing analysis.

Never recommends a cheaper model because it is cheaper. A downgrade is only 'supported' when, within
the SAME task type and complexity bucket, the alternative is statistically non-inferior on success AND
measured quality with enough samples. Observational data is confounded (harder tasks were routed to
stronger models), so 'supported' means 'worth a replay/A-B test', and the verdict says so.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .ledger import Ledger
from .metrics import task_rows
from .pricing import PriceTable
from .stats import mean, median, stdev, wilson
from .waste import WasteFinding

COMPLEXITY_RANK = {"simple": 0, "medium": 1, "complex": 2, "large": 3}


@dataclass
class RoutingConfig:
    min_n: int = 20                    # min tasks per side to say anything
    success_margin: float = 0.05       # tolerated success-rate loss
    quality_margin: float = 0.05       # tolerated mean-quality loss
    z: float = 1.645                   # one-sided 95%
    # model_id -> highest complexity it is expected to handle (simple=0 .. large=3)
    tiers: dict[str, int] = field(default_factory=dict)


def tier_of(model_id: str, cfg: RoutingConfig, pricing: Optional[PriceTable] = None) -> Optional[int]:
    if model_id in cfg.tiers:
        return cfg.tiers[model_id]
    if pricing and model_id in pricing.prices:            # fallback: price rank as a capability proxy
        order = sorted(pricing.prices, key=lambda m: pricing.prices[m].input + pricing.prices[m].output)
        return order.index(model_id)
    return None


def stratified_matrix(ledger: Ledger, task_type: Optional[str] = None) -> list[dict]:
    """(task_type x complexity x model) -> n, success, quality, cost. Compare only within a bucket."""
    rows = task_rows(ledger)
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["outcome"] and (task_type is None or r["task_type"] == task_type):
            by[(r["task_type"], r["complexity"], r["primary_model"])].append(r)
    out = []
    for (tt, cx, m), rs in sorted(by.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        k = sum(1 for r in rs if r["outcome"] == "success")
        lo, hi = wilson(k, len(rs))
        qs = [r["quality"] for r in rs if r["quality"] is not None]
        costs = [r["cost"] for r in rs if r["cost"] is not None]
        out.append(dict(task_type=tt, complexity=cx, model=m, n=len(rs), success_rate=k / len(rs),
                        success_ci=(lo, hi), mean_quality=mean(qs), quality_n=len(qs),
                        median_cost=median(costs) if len(costs) == len(rs) else None,
                        median_tokens=median([r["tokens"] for r in rs])))
    return out


def _pool(rows: list[dict], task_type, complexity, model) -> list[dict]:
    return [r for r in rows if r["outcome"] and r["task_type"] == task_type and r["complexity"] == complexity
            and r["primary_model"] == model]


def cheaper_model_evidence(ledger: Ledger, task_type: str, complexity: Optional[str], chosen: str, alt: str,
                           cfg: RoutingConfig, rows: Optional[list[dict]] = None) -> dict:
    rows = rows if rows is not None else task_rows(ledger)
    a, c = _pool(rows, task_type, complexity, alt), _pool(rows, task_type, complexity, chosen)
    ev = dict(task_type=task_type, complexity=complexity, chosen=chosen, alternative=alt, n_alt=len(a), n_chosen=len(c))
    if len(a) < cfg.min_n or len(c) < cfg.min_n:
        return {**ev, "verdict": "insufficient_evidence",
                "reason": f"need >= {cfg.min_n} tasks per side in this bucket; have alt={len(a)}, chosen={len(c)}"}
    pa = sum(r["outcome"] == "success" for r in a) / len(a)
    pc = sum(r["outcome"] == "success" for r in c) / len(c)
    se = math.sqrt(pa * (1 - pa) / len(a) + pc * (1 - pc) / len(c))
    lower = (pa - pc) - cfg.z * se
    ev.update(success_alt=pa, success_chosen=pc, success_diff_lower=lower)
    if lower <= -cfg.success_margin:
        return {**ev, "verdict": "not_supported", "reason": "alternative not shown non-inferior on success"}
    qa = [r["quality"] for r in a if r["quality_method"] and r["quality"] is not None]
    qc = [r["quality"] for r in c if r["quality"] is not None]
    if len(qa) < cfg.min_n or len(qc) < cfg.min_n:
        return {**ev, "verdict": "insufficient_evidence",
                "reason": "success is non-inferior but quality labels are too sparse to confirm; do not downgrade on success alone"}
    sa, sc = stdev(qa) or 0.0, stdev(qc) or 0.0
    qse = math.sqrt(sa ** 2 / len(qa) + sc ** 2 / len(qc))
    qlower = (mean(qa) - mean(qc)) - cfg.z * qse
    ev.update(quality_alt=mean(qa), quality_chosen=mean(qc), quality_diff_lower=qlower)
    if qlower <= -cfg.quality_margin:
        return {**ev, "verdict": "not_supported", "reason": "alternative not shown non-inferior on quality"}
    ca, cc = [r["cost"] for r in a if r["cost"] is not None], [r["cost"] for r in c if r["cost"] is not None]
    if ca and cc:
        ev["median_cost_alt"], ev["median_cost_chosen"] = median(ca), median(cc)
    return {**ev, "verdict": "supported_downgrade",
            "reason": "non-inferior on success and quality in this bucket. Observational (routing was not "
                      "randomised): confirm with replay or a small A/B before changing the router."}


def analyze_task_routing(ledger: Ledger, task_id: str, cfg: Optional[RoutingConfig] = None,
                         pricing: Optional[PriceTable] = None) -> dict:
    cfg = cfg or RoutingConfig()
    task = ledger.one("SELECT * FROM tasks WHERE task_id=?", (task_id,))
    if task is None:
        raise KeyError(task_id)
    decisions = ledger.query("SELECT * FROM routing_decisions WHERE task_id=? ORDER BY ts", (task_id,))
    calls = [c for c in ledger.query("SELECT * FROM model_calls WHERE task_id=? ORDER BY seq", (task_id,))
             if c["status"] != "blocked_budget"]
    by_model: dict[str, int] = defaultdict(int)
    for c in calls:
        by_model[c["model_id"]] += c["input_fresh"] + c["cache_read"] + c["cache_write"] + c["output_tokens"]
    chosen = decisions[-1]["chosen"] if decisions else (max(by_model, key=by_model.get) if by_model else None)
    escalations = [d for d in decisions if d["escalated_from"]]
    prior_spend = sum(by_model.get(d["escalated_from"], 0) for d in escalations)
    tier = tier_of(chosen, cfg, pricing) if chosen else None
    need = COMPLEXITY_RANK.get(task["complexity_est"] or "", None)
    # cfg.tiers is on the same scale as complexity: simple=0, medium=1, complex=2, large=3
    # (the highest complexity a model is expected to handle).
    verdict = "unknown"
    if escalations:
        verdict = "under_provisioned_first_choice"
    elif tier is not None and need is not None:
        verdict = "under_provisioned" if tier < need else ("over_provisioned_candidate" if tier > need else "appropriate")
    alts = []
    if chosen and decisions:
        cands = json.loads(decisions[-1]["candidates"] or "[]")
        rows = task_rows(ledger)
        for m in cands:
            t = tier_of(m, cfg, pricing)
            if m != chosen and t is not None and tier is not None and t < tier:
                alts.append(cheaper_model_evidence(ledger, task["task_type"], task["complexity_est"], chosen, m, cfg, rows))
    return dict(task_id=task_id, task_type=task["task_type"], complexity=task["complexity_est"], outcome=task["outcome"],
                chosen=chosen, reason_code=decisions[-1]["reason_code"] if decisions else None,
                features=json.loads(decisions[-1]["complexity_features"] or "{}") if decisions else {},
                appropriateness=verdict, tokens_by_model=dict(by_model),
                failure_driven_spend_tokens=prior_spend, escalated_from=[d["escalated_from"] for d in escalations],
                cheaper_alternatives=alts)


def overpowered_findings(ledger: Ledger, cfg: Optional[RoutingConfig] = None, pricing: Optional[PriceTable] = None,
                         task_type: Optional[str] = None) -> list[WasteFinding]:
    """Cost-only findings: task profiles where evidence supports a cheaper model."""
    cfg = cfg or RoutingConfig()
    rows = task_rows(ledger)
    buckets = {(r["task_type"], r["complexity"]) for r in rows if r["outcome"]}
    out = []
    for tt, cx in sorted(buckets, key=str):
        if task_type and tt != task_type:
            continue
        models = sorted({r["primary_model"] for r in rows if r["task_type"] == tt and r["complexity"] == cx}, key=lambda m: tier_of(m, cfg, pricing) or 0)
        for hi in models[1:]:
            for lo in models:
                if (tier_of(lo, cfg, pricing) or 0) >= (tier_of(hi, cfg, pricing) or 0):
                    continue
                ev = cheaper_model_evidence(ledger, tt, cx, hi, lo, cfg, rows)
                if ev["verdict"] == "supported_downgrade":
                    n = ev["n_chosen"]
                    out.append(WasteFinding(
                        category="overpowered_model", unit="cost_only", confidence="heuristic", tokens_wasted=0,
                        pct_of_session=0.0, evidence=[f"{tt}/{cx}: {hi} -> {lo}"], evidence_count=n,
                        root_cause=f"{hi} used for {tt}/{cx} tasks where {lo} is statistically non-inferior (n={ev['n_alt']}/{n}).",
                        recommendation=f"Replay or A/B route a slice of {tt}/{cx} to {lo}; keep {hi} as escalation target.",
                        expected_reduction_low=0, expected_reduction_high=0, quality_risk="medium",
                        quality_risk_note="Observational evidence; verify with randomised comparison.", details=ev))
    return out
