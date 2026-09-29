"""Compact, evidence-linked telemetry payload for the LLM analyzer.

The analyzer EXPLAINS deterministic findings; it must not recompute token usage. Analysis itself costs
tokens, so we sample: all failures, all budget-exhausted, all outliers, plus a random slice of the rest.
"""
from __future__ import annotations

import json
import random
from typing import Optional

from .ledger import Ledger

ANALYZER_SYSTEM_PROMPT = """You are a token-efficiency analyst for an AI-agent platform.
You receive STRUCTURED TELEMETRY computed deterministically from a ledger. Rules:
1. Use only numbers present in the payload. Never estimate or recompute token counts.
2. Every claim must cite evidence ids (call:/seg:/tool:) from the payload.
3. Findings labelled `heuristic` are leads, not facts: say what would confirm them.
4. Do not recommend a cheaper model or a smaller context merely because it is cheaper. State the quality
   risk and how to validate (replay / A-B) before any change.
5. Separate root-cause hypotheses from measured facts. If evidence is insufficient, say so.
Output: root causes (ranked), recommended optimizations with expected savings range and quality risk,
and what additional telemetry would remove remaining uncertainty."""


def select_sessions(ledger: Ledger, *, since: Optional[float] = None, until: Optional[float] = None,
                    random_fraction: float = 0.1, seed: int = 0, outlier_sessions: Optional[set] = None) -> list[dict]:
    where, params = ["parent_session_id IS NULL"], []
    if since is not None:
        where.append("started_at >= ?")
        params.append(since)
    if until is not None:
        where.append("started_at < ?")
        params.append(until)
    roots = ledger.query(f"SELECT session_id, terminal_status FROM sessions WHERE {' AND '.join(where)}", params)
    failed = {r["sid"] for r in ledger.query(
        """SELECT DISTINCT s.budget_root_id AS sid FROM tasks t JOIN sessions s ON s.session_id=t.session_id
           WHERE t.outcome IS NOT NULL AND t.outcome!='success'""")}
    picked: dict[str, str] = {}
    for r in roots:
        if r["terminal_status"] == "budget_exhausted":
            picked[r["session_id"]] = "budget_exhausted"
        elif r["session_id"] in failed:
            picked[r["session_id"]] = "failure"
        elif outlier_sessions and r["session_id"] in outlier_sessions:
            picked[r["session_id"]] = "outlier"
    rest = sorted(r["session_id"] for r in roots if r["session_id"] not in picked)
    rng = random.Random(seed)
    for sid in rng.sample(rest, k=min(len(rest), max(1, int(len(rest) * random_fraction))) if rest else 0):
        picked[sid] = "random_sample"
    return [dict(session_id=k, reason=v) for k, v in picked.items()]


def build_payload(report: dict, *, max_chars: int = 24_000, max_evidence: int = 8) -> str:
    """Trim a session report to a size the analyzer can afford. Drops detail, never evidence ids of kept findings."""
    s = report["summary"]
    payload = dict(
        session=dict(id=s["session_id"], status=s["terminal_status"], billed=s["total_billed_tokens"], cost=s["estimated_cost"],
                     models=s["models"], iterations=s["iterations"], retries=s["retries"], tool_calls=s["tool_calls"],
                     tasks=s["tasks"], successes=s["successes"], quality=s["quality"]["by_method"]),
        budget_events=[dict(level=e["level"], tokens_at=e["tokens_at"]) for e in report["budget"]["events"]],
        per_iteration=[dict(i=t["iteration"], model=t["model"], status=t["status"], billed=t["billed"], cum=t["cumulative"])
                       for t in report["budget"]["trajectory"]],
        context_growth=[dict(task=g["task_id"], agent=g["agent_id"], slope=g["slope_tokens_per_iteration"], first=g["first_input"],
                             last=g["last_input"], dominant=g["dominant_growth_kind"], growth_by_kind=g["growth_by_kind"])
                        for g in report["context_growth"]],
        top_consumers=report["top_consumers"][:6],
        findings=[dict(category=f["category"], unit=f["unit"], confidence=f["confidence"], tokens=f["tokens_wasted"],
                       pct=round(f["pct_of_session"], 4), evidence=f["evidence"][:max_evidence], root_cause=f["root_cause"],
                       risk=f["quality_risk"], details=f["details"]) for f in report["waste"]["findings"][:10]],
        routing=[dict(task=r["task_id"], chosen=r["chosen"], appropriateness=r["appropriateness"],
                      failure_driven_spend=r["failure_driven_spend_tokens"],
                      alternatives=[dict(model=a["alternative"], verdict=a["verdict"], reason=a["reason"]) for a in r["cheaper_alternatives"]])
                 for r in report["routing"]],
        estimated_savings=report["estimated_savings"], notes=report["waste"]["notes"], coverage=report["waste"]["coverage"])
    text = json.dumps(payload, separators=(",", ":"), default=str)
    if len(text) > max_chars:                       # shed the bulkiest, least essential parts first
        payload["per_iteration"] = payload["per_iteration"][:5] + payload["per_iteration"][-5:]
        payload["findings"] = payload["findings"][:5]
        text = json.dumps(payload, separators=(",", ":"), default=str)
    return text
