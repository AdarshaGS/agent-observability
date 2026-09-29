"""Quality evaluation storage. Scores from different methods are NEVER averaged together.

Trust order (highest first): test > assertion > human > user_signal > judge.
Quality can arrive late (a revert next week): evals are append-only and metrics recompute.
"""
from __future__ import annotations

import json
import time
from typing import Iterable, Optional

from .ledger import Ledger

TRUST_ORDER = ("test", "assertion", "human", "user_signal", "judge")


def record_eval(ledger: Ledger, task_id: str, method: str, score: float, *, session_id: Optional[str] = None,
                rubric_version: Optional[str] = None, judge_model: Optional[str] = None,
                confidence: Optional[float] = None, meta: Optional[dict] = None, now: Optional[float] = None) -> None:
    if method not in TRUST_ORDER:
        raise ValueError(f"method must be one of {TRUST_ORDER}")
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be in [0,1]")
    if method == "judge" and not judge_model:
        raise ValueError("judge evals must record judge_model (judge must differ from the task model)")
    if session_id is None:
        t = ledger.one("SELECT session_id FROM tasks WHERE task_id=?", (task_id,))
        session_id = t["session_id"] if t else None
    ledger.insert("quality_evals", dict(task_id=task_id, session_id=session_id, method=method, score=score,
                                        rubric_version=rubric_version, judge_model=judge_model,
                                        confidence=confidence, meta=json.dumps(meta) if meta else None,
                                        ts=now if now is not None else time.time()))


def best_eval(evals: list[dict]) -> Optional[dict]:
    """Highest-trust eval; latest wins within a method."""
    best = None
    for e in evals:
        rank = TRUST_ORDER.index(e["method"])
        if best is None or rank < best[0] or (rank == best[0] and e["eval_id"] > best[1]["eval_id"]):
            best = (rank, e)
    return best[1] if best else None


def quality_by_task(ledger: Ledger, task_ids: Iterable[str]) -> dict[str, dict]:
    ids = list(task_ids)
    if not ids:
        return {}
    rows = ledger.query(f"SELECT * FROM quality_evals WHERE task_id IN ({','.join('?' * len(ids))})", ids)
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["task_id"], []).append(r)
    return {t: b for t, ev in by.items() if (b := best_eval(ev))}


def summarize(quality: dict[str, dict], n_tasks: int) -> dict:
    """Per-method means (never pooled) plus coverage."""
    per: dict[str, list[float]] = {}
    for e in quality.values():
        per.setdefault(e["method"], []).append(e["score"])
    return dict(coverage=(len(quality) / n_tasks) if n_tasks else 0.0,
                by_method={m: dict(n=len(v), mean=sum(v) / len(v)) for m, v in per.items()})


def judge_human_agreement(ledger: Ledger) -> dict:
    """Calibration check: how far does the LLM judge sit from human scores on shared tasks?"""
    rows = ledger.query(
        """SELECT j.score AS j, h.score AS h FROM quality_evals j JOIN quality_evals h ON h.task_id=j.task_id
           WHERE j.method='judge' AND h.method='human'""")
    if not rows:
        return dict(n=0, mean_abs_diff=None, bias=None)
    diffs = [r["j"] - r["h"] for r in rows]
    return dict(n=len(rows), mean_abs_diff=sum(abs(d) for d in diffs) / len(diffs), bias=sum(diffs) / len(diffs))
