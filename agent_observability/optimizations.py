"""Before-vs-after tracking for optimizations: did a change cut tokens WITHOUT hurting success or quality?

An optimization links two sets of root sessions (baseline, after). Metrics are recomputed live from the ledger
(late quality evals count), per task. A verdict of 'improvement' needs ALL of: tokens/task down by >= MIN_REDUCTION,
the drop beyond noise, success rate not down, quality measured with a shared method and not down, enough tasks.
Anything less gets an honest weaker label. Observational, not randomised: task mix can differ between arms.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Optional

from .compare import _cost, _price
from .ledger import Ledger
from .quality import quality_by_task, record_eval
from .stats import mean, stdev

MIN_REDUCTION = 0.05      # smaller moves are treated as noise
MAX_QUALITY_DROP = 0.02   # tolerated success/quality drop (absolute)
ARMS = ("baseline", "after")
BLOCKED = "blocked_budget"

DDL = """
CREATE TABLE IF NOT EXISTS optimizations(
  opt_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS optimization_sessions(
  opt_id TEXT NOT NULL, arm TEXT NOT NULL, session_id TEXT NOT NULL, PRIMARY KEY(opt_id, arm, session_id));
"""


def ensure(ledger: Ledger) -> None:
    for stmt in DDL.strip().split(";"):
        if stmt.strip():
            ledger.execute(stmt)


def add(ledger: Ledger, opt_id: str, name: str, baseline: list[str], after: list[str],
        description: Optional[str] = None) -> dict:
    ensure(ledger)
    if not baseline or not after:
        raise ValueError("need at least one baseline and one after session")
    if set(baseline) & set(after):
        raise ValueError("a session cannot be in both baseline and after")
    missing = [s for s in baseline + after if not ledger.one("SELECT 1 FROM sessions WHERE session_id=?", (s,))]
    if missing:
        raise ValueError(f"unknown sessions (not collected yet?): {missing}")
    ledger.execute("INSERT OR REPLACE INTO optimizations VALUES(?,?,?,?)", (opt_id, name, description, time.time()))
    ledger.execute("DELETE FROM optimization_sessions WHERE opt_id=?", (opt_id,))
    for arm, ids in (("baseline", baseline), ("after", after)):
        for s in ids:
            ledger.execute("INSERT INTO optimization_sessions VALUES(?,?,?)", (opt_id, arm, s))
    return dict(opt_id=opt_id, baseline=len(baseline), after=len(after))


RATING = "rating:"   # task_id prefix used by rate(): one rating per collected session that has no explicit tasks


def rate(ledger: Ledger, root_session_id: str, outcome: str, score: Optional[float] = None,
         method: str = "human") -> dict:
    """Mark a whole collected session (a unit of work with no task events) as success/failure, optionally scored."""
    if outcome not in ("success", "failure", "partial"):
        raise ValueError("outcome must be success, failure or partial")
    tid = RATING + root_session_id
    ledger.execute("DELETE FROM tasks WHERE task_id=?", (tid,))
    ledger.execute("DELETE FROM quality_evals WHERE task_id=?", (tid,))
    ledger.create_task(tid, root_session_id)
    ledger.finish_task(tid, outcome)
    if score is not None:
        record_eval(ledger, tid, method, score, session_id=root_session_id)
    return dict(session_id=root_session_id, outcome=outcome, score=score, method=method if score is not None else None)


def _arm(ledger: Ledger, roots: list[str], price) -> dict:
    """Unit of work = an explicit task, or the whole session tree when it has no task events (Claude Code transcripts)."""
    calls, tools, tasks, task_of = [], [], [], {}
    for r in roots:
        ids = ledger.tree_ids(r)
        rc = [c for c in ledger.calls_for(ids) if c["status"] != BLOCKED]
        rt = ledger.tasks_for(ids)
        real = {t["task_id"] for t in rt if not t["task_id"].startswith(RATING)}
        if any(c["task_id"] in real for c in rc):
            tasks += [t for t in rt if t["task_id"] in real]
        else:
            for c in rc:
                c["task_id"] = "unit:" + r
            tasks += [dict(task_id="unit:" + r, outcome=next((t["outcome"] for t in rt if t["task_id"] == RATING + r), None))]
        calls += rc
        task_of.update({c["call_id"]: c["task_id"] for c in rc})
        tools += ledger.tool_calls_for(ids)
    per: dict[str, dict] = {t["task_id"]: defaultdict(float) for t in tasks}
    for c in calls:
        p = per.get(c["task_id"])
        if p is None:
            continue
        p["input"] += c["input_fresh"] + c["cache_read"] + c["cache_write"]
        p["output"] += c["output_tokens"]
        p["cache_read"] += c["cache_read"]
        p["iterations"] += 1
        p["cost"] += _cost(dict(fresh=c["input_fresh"], cache_read=c["cache_read"], cache_write=c["cache_write"],
                                output=c["output_tokens"]), price)
    for t in tools:
        p = per.get(task_of.get(t["call_id"]))
        if p is not None:
            p["tool_calls"] += 1
            p["tool_result_tokens"] += t["injected_tokens"]
    n = len(tasks)
    rows = list(per.values())
    for p in rows:
        p["tokens"] = p["input"] + p["output"]
    tot = lambda k: sum(p[k] for p in rows)
    per_task = lambda k: tot(k) / n if n else None
    judged = [t for t in tasks if t["outcome"]]
    q = quality_by_task(ledger, [t["task_id"] if not t["task_id"].startswith("unit:") else RATING + t["task_id"][5:] for t in tasks])
    by_method: dict[str, list[float]] = defaultdict(list)
    for e in q.values():
        by_method[e["method"]].append(e["score"])
    return dict(
        tasks=n, sessions=len(roots),
        tokens_per_task=per_task("tokens"), input_tokens=per_task("input"), output_tokens=per_task("output"),
        tool_result_tokens=per_task("tool_result_tokens"), iterations=per_task("iterations"),
        tool_calls=per_task("tool_calls"), cost=per_task("cost"),
        cache_hit_rate=tot("cache_read") / tot("input") if tot("input") else None,
        success_rate=sum(t["outcome"] == "success" for t in judged) / len(judged) if judged else None,
        quality={m: dict(n=len(v), mean=sum(v) / len(v)) for m, v in by_method.items()},
        _token_samples=[p["tokens"] for p in rows])


def _delta(a: dict, b: dict, k: str) -> Optional[float]:
    """Relative change after vs baseline (negative = fewer)."""
    return (b[k] / a[k] - 1) if a.get(k) and b.get(k) is not None else None


def _verdict(a: dict, b: dict) -> tuple[str, str, str]:
    """-> (verdict, confidence, reason). Deliberately conservative."""
    red = -(_delta(a, b, "tokens_per_task") or 0.0)
    xa, xb = a["_token_samples"], b["_token_samples"]
    n = min(len(xa), len(xb))
    se = math.sqrt((stdev(xa) or 0) ** 2 / len(xa) + (stdev(xb) or 0) ** 2 / len(xb)) if n > 1 else None
    z = abs(mean(xa) - mean(xb)) / se if se else None
    beyond_noise = z is not None and z >= 2
    conf = "low" if n < 5 else "high" if n >= 20 and beyond_noise else "medium" if beyond_noise else "low"
    sa, sb = a["success_rate"], b["success_rate"]
    succ_drop = sa is not None and sb is not None and sb < sa - MAX_QUALITY_DROP
    shared = [m for m in a["quality"] if m in b["quality"]]
    q_drop = [m for m in shared if b["quality"][m]["mean"] < a["quality"][m]["mean"] - MAX_QUALITY_DROP]
    if red <= -MIN_REDUCTION:
        return "worse", conf, f"tokens/task rose {-red:.0%}"
    if red < MIN_REDUCTION:
        return "no_change", conf, f"tokens/task moved {-red:+.0%}, under the {MIN_REDUCTION:.0%} noise floor"
    if succ_drop or q_drop:
        what = "success rate" if succ_drop else f"quality ({', '.join(q_drop)})"
        return "quality_regression", conf, f"tokens/task down {red:.0%} but {what} dropped"
    if n < 5:
        return "inconclusive", "low", f"only {n} tasks in the smaller arm (need >= 5)"
    if not beyond_noise:
        return "inconclusive", "low", f"tokens/task down {red:.0%} but within run-to-run variation"
    if sa is None or sb is None:
        return "unverified", conf, "tokens down but task outcomes are not recorded, success unknown"
    if not shared:
        return "unverified", conf, "tokens down, success held, but no quality eval shared by both arms"
    return "improvement", conf, f"tokens/task down {red:.0%}, success and quality held"


def report(ledger: Ledger, db_path: Optional[str] = None) -> dict:
    ensure(ledger)
    price, basis = _price(db_path)
    out = []
    for o in ledger.query("SELECT * FROM optimizations ORDER BY created_at DESC"):
        roots = {arm: [r["session_id"] for r in ledger.query(
            "SELECT session_id FROM optimization_sessions WHERE opt_id=? AND arm=?", (o["opt_id"], arm))] for arm in ARMS}
        a, b = (_arm(ledger, roots[arm], price) for arm in ARMS)
        verdict, conf, reason = _verdict(a, b)
        keys = ("tokens_per_task", "input_tokens", "output_tokens", "cache_hit_rate", "tool_result_tokens",
                "iterations", "tool_calls", "cost")
        shared = [m for m in a["quality"] if m in b["quality"]]
        for x in (a, b):
            x.pop("_token_samples")
        out.append(dict(
            **o, baseline=a, after=b, verdict=verdict, confidence=conf, reason=reason,
            token_reduction=-(_delta(a, b, "tokens_per_task") or 0.0) if a["tasks"] and b["tasks"] else None,
            deltas={k: _delta(a, b, k) for k in keys},
            success_change=(b["success_rate"] - a["success_rate"]) if None not in (a["success_rate"], b["success_rate"]) else None,
            quality_change={m: b["quality"][m]["mean"] - a["quality"][m]["mean"] for m in shared}))
    return dict(price_basis=basis, optimizations=out,
                caveats=["Different tasks in each arm, not the same tasks run twice: mix and difficulty can differ.",
                         "Only 'improvement' means tokens fell with success and quality held; every other label is weaker.",
                         "Quality is compared per method, never pooled; no shared quality method means 'unverified'."])
