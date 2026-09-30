"""Cohort comparison: work done with subagents vs inline, normalised per unit of work (e.g. per issue).

Cohorts are hand-tagged (a cohort = one piece of work = one root session plus its subagent sessions).
Nothing here is randomised: different work units differ in size and difficulty, so treat the numbers as
an observational case study, not a controlled experiment. Token counts are measured; dollar figures are
only as good as the price table you supply (compare_prices.json next to the ledger, else an ASSUMED default).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .ledger import Ledger
from .pricing import ModelPrice

ARMS = ("with_agents", "inline")
ASSUMED = dict(input=3.0, output=15.0, cache_read=0.30, cache_write=3.75)   # $/MTok, placeholder Sonnet-class list price

DDL = """
CREATE TABLE IF NOT EXISTS cohorts(
  cohort_id TEXT PRIMARY KEY, arm TEXT NOT NULL, label TEXT NOT NULL,
  units INTEGER NOT NULL, unit_label TEXT NOT NULL DEFAULT 'issue', note TEXT);
CREATE TABLE IF NOT EXISTS cohort_sessions(
  cohort_id TEXT NOT NULL, session_id TEXT NOT NULL, PRIMARY KEY(cohort_id, session_id));
"""


def ensure(ledger: Ledger) -> None:
    for stmt in DDL.strip().split(";"):
        if stmt.strip():
            ledger.execute(stmt)


def tag(ledger: Ledger, cohort_id: str, arm: str, label: str, units: int, session_ids: list[str],
        unit_label: str = "issue", note: Optional[str] = None) -> dict:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if units <= 0:
        raise ValueError("units must be positive")
    ensure(ledger)
    missing = [s for s in session_ids if not ledger.one("SELECT 1 FROM sessions WHERE session_id=?", (s,))]
    if missing:
        raise ValueError(f"unknown sessions (not collected yet?): {missing}")
    ledger.execute("INSERT OR REPLACE INTO cohorts VALUES(?,?,?,?,?,?)", (cohort_id, arm, label, units, unit_label, note))
    ledger.execute("DELETE FROM cohort_sessions WHERE cohort_id=?", (cohort_id,))
    for s in session_ids:
        ledger.execute("INSERT INTO cohort_sessions VALUES(?,?)", (cohort_id, s))
    return dict(cohort_id=cohort_id, sessions=len(session_ids))


def _price(db_path: Optional[str]) -> tuple[ModelPrice, str]:
    if db_path:
        f = Path(db_path).with_name("compare_prices.json")
        if f.exists():
            d = json.loads(f.read_text())
            return ModelPrice(d["input"], d["output"], d.get("cache_read"), d.get("cache_write")), f"{f.name} ({d.get('version', 'unversioned')})"
    return ModelPrice(**ASSUMED), "ASSUMED placeholder list price - supply compare_prices.json for real numbers"


def _sum(ledger: Ledger, sids: list[str]) -> dict:
    if not sids:
        return dict(calls=0, fresh=0, cache_read=0, cache_write=0, output=0)
    q = ",".join("?" * len(sids))
    return ledger.one(f"SELECT COUNT(*) calls, COALESCE(SUM(input_fresh),0) fresh, COALESCE(SUM(cache_read),0) cache_read,"
                      f" COALESCE(SUM(cache_write),0) cache_write, COALESCE(SUM(output_tokens),0) output"
                      f" FROM model_calls WHERE session_id IN ({q})", sids)


def _cost(t: dict, p: ModelPrice) -> float:
    return (t["fresh"] * p.input + t["cache_read"] * (p.cache_read or p.input)
            + t["cache_write"] * (p.cache_write or p.input) + t["output"] * p.output) / 1e6


def compare(ledger: Ledger, db_path: Optional[str] = None) -> dict:
    ensure(ledger)
    price, basis = _price(db_path)
    cohorts = []
    for c in ledger.query("SELECT * FROM cohorts ORDER BY arm, cohort_id"):
        sids = [r["session_id"] for r in ledger.query("SELECT session_id FROM cohort_sessions WHERE cohort_id=?", (c["cohort_id"],))]
        subs = [r["session_id"] for r in ledger.query(
            f"SELECT session_id FROM sessions WHERE parent_session_id IN ({','.join('?' * len(sids)) or 'NULL'})", sids)] if sids else []
        main, sub = _sum(ledger, sids), _sum(ledger, [s for s in subs if s not in sids])
        tot = {k: main[k] + sub[k] for k in main}
        cohorts.append(dict(**c, sessions=len(sids), subagent_sessions=len(subs), main=main, sub=sub, total=tot,
                            est_cost=_cost(tot, price), sub_cost=_cost(sub, price)))
    arms = {}
    for arm in ARMS:
        cs = [c for c in cohorts if c["arm"] == arm]
        units = sum(c["units"] for c in cs)
        tot = {k: sum(c["total"][k] for c in cs) for k in ("calls", "fresh", "cache_read", "cache_write", "output")}
        cost = sum(c["est_cost"] for c in cs)
        arms[arm] = dict(cohorts=len(cs), units=units, total=tot, est_cost=cost,
                         per_unit=dict(calls=tot["calls"] / units, output=tot["output"] / units, cache_read=tot["cache_read"] / units,
                                       billed=sum(tot[k] for k in ("fresh", "cache_read", "cache_write", "output")) / units,
                                       est_cost=cost / units) if units else None)
    a, b = arms["with_agents"]["per_unit"], arms["inline"]["per_unit"]
    delta = {k: (a[k] / b[k] - 1) if b[k] else None for k in a} if a and b else None
    return dict(price_basis=basis, price=price.__dict__, cohorts=cohorts, arms=arms, delta_agents_vs_inline=delta,
                caveats=["Different work units, not the same work run twice: size/difficulty differ.",
                         "Cache-read tokens dominate billed volume; long single-thread sessions re-read a growing context every call.",
                         "Token counts are measured from provider usage; cost is only as accurate as the price basis above."])


def tag_custom_agents(ledger: Ledger, agent_names: set[str], since: float, min_calls: int = 5) -> dict:
    """Auto-cohorts per project: 'with_agents' = root sessions started at/after `since` that ran one of YOUR agents
    (built-in Explore etc. do not count); 'inline' = root sessions in the same window with no subagents at all.
    One root session = one unit. Only projects present in both arms are tagged. Replaces earlier auto cohorts."""
    ensure(ledger)
    ledger.execute("DELETE FROM cohort_sessions WHERE cohort_id IN (SELECT cohort_id FROM cohorts WHERE note LIKE 'auto%')")
    ledger.execute("DELETE FROM cohorts WHERE note LIKE 'auto%'")
    names = ",".join("?" * len(agent_names)) or "NULL"
    rows = ledger.query(f"""
        SELECT r.session_id sid, REPLACE(r.root_task,' · claude-vscode','') proj,
          (SELECT COUNT(*) FROM sessions s JOIN model_calls m ON m.session_id=s.session_id
             WHERE s.parent_session_id=r.session_id AND m.agent_id IN ({names})) custom,
          (SELECT COUNT(*) FROM sessions s WHERE s.parent_session_id=r.session_id) subs,
          (SELECT COUNT(*) FROM model_calls m WHERE m.session_id=r.session_id OR m.session_id IN
             (SELECT session_id FROM sessions WHERE parent_session_id=r.session_id)) calls
        FROM sessions r WHERE r.parent_session_id IS NULL AND r.started_at>=? AND COALESCE(r.root_task,'') NOT LIKE '%subagent%'""",
        [*sorted(agent_names), since])
    by: dict[str, dict[str, list[str]]] = {}
    for r in rows:
        if r["calls"] < min_calls:
            continue
        arm = "with_agents" if r["custom"] else "inline" if not r["subs"] else None
        if arm:
            by.setdefault(r["proj"], {}).setdefault(arm, []).append(r["sid"])
    tagged = {}
    for proj, d in by.items():
        if len(d) == 2:
            for arm, ids in d.items():
                tag(ledger, f"{proj}:{arm}", arm, f"{proj} ({arm})", len(ids), ids, "session",
                    "auto: custom agents only; 1 root session = 1 unit")
                tagged[f"{proj}:{arm}"] = len(ids)
    counts = {a: sum(1 for r in rows if r["calls"] >= min_calls and (r["custom"] if a == "with_agents" else not r["subs"]))
              for a in ARMS}
    return dict(tagged=tagged, sessions_with_your_agents=counts["with_agents"], inline_sessions=counts["inline"])
