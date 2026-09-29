"""Language-agnostic event ingestion (JSON) plus remote budget reservations.

Any system, in any language, can report model calls by POSTing events. Events are idempotent on their ids
(retrying a POST never double counts). Message *content* is optional and never stored: if you send segment
`text`, the server computes token count, hash and a hashed-identifier fingerprint and discards the text.

Enforcement note: a reported call has already happened, so budget accounting on ingest is ADVISORY (the reply
tells the caller its level and policy). For a hard cap the caller must `reserve` before calling the model.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from typing import Any, Optional

from .budget import BudgetConfig, BudgetExhausted, BudgetManager, BudgetStatus, Reservation
from .gateway import Gateway, _canon
from .ledger import Ledger
from .models import (SEGMENT_KINDS, CallRequest, ProviderError, ProviderResponse, Segment, ToolCallRequest, Usage,
                     usage_from_anthropic, usage_from_openai)
from .pricing import PriceTable
from .quality import record_eval
from .tokens import HeuristicEstimator, TokenEstimator, content_hash, fingerprint

MAX_BATCH = 1000
MAX_SEGMENTS = 500
DEFAULT_RESERVATION_TTL = 300.0
OUTCOMES = ("success", "failure", "partial", "budget_exhausted", "abandoned")
SESSION_STATUSES = ("completed", "failed", "budget_exhausted", "abandoned")


class IngestError(ValueError):
    pass


def _uid() -> str:
    return uuid.uuid4().hex[:16]


def _str(ev: dict, key: str, required: bool = False, default: Optional[str] = None) -> Optional[str]:
    v = ev.get(key, default)
    if v is None:
        if required:
            raise IngestError(f"'{key}' is required")
        return None
    if not isinstance(v, str) or not v or len(v) > 256:
        raise IngestError(f"'{key}' must be a non-empty string (<=256 chars)")
    return v


def _nonneg(v: Any, name: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or v != int(v):
        raise IngestError(f"'{name}' must be a non-negative integer")
    return int(v)


def _num(v: Any, name: str) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
        raise IngestError(f"'{name}' must be a non-negative number")
    return float(v)


def budget_dict(st: BudgetStatus) -> dict:
    return dict(level=st.level, fraction=st.fraction, used=st.used, reserved=st.reserved, limit=st.limit,
                remaining=st.remaining, remaining_normal=st.remaining_normal,
                projected_calls_remaining=st.projected_calls_remaining, state_line=st.state_line(),
                policy=asdict(st.policy))


class Ingestor:
    def __init__(self, ledger: Ledger, budget: Optional[BudgetManager] = None, pricing: Optional[PriceTable] = None,
                 estimator: Optional[TokenEstimator] = None, clock=time.time):
        self.ledger = ledger
        self.budget = budget or BudgetManager(ledger)
        self.estimator = estimator or HeuristicEstimator()
        self.gw = Gateway(ledger, self.budget, pricing, self.estimator, clock=clock)
        self.clock = clock

    # ------------------------------------------------------------------ entry points
    def handle_batch(self, events: list) -> list[dict]:
        if len(events) > MAX_BATCH:
            raise IngestError(f"batch too large (max {MAX_BATCH} events)")
        return [self.handle(e) for e in events]

    def handle(self, ev: Any) -> dict:
        try:
            if not isinstance(ev, dict):
                raise IngestError("event must be a JSON object")
            handler = {"session_start": self._session_start, "session_end": self._session_end,
                       "task_start": self._task_start, "task_end": self._task_end, "routing": self._routing,
                       "model_call": self._model_call, "tool_call": self._tool_call,
                       "quality": self._quality}.get(ev.get("type"))
            if handler is None:
                raise IngestError(f"unknown event type {ev.get('type')!r}")
            return handler(ev)
        except IngestError as e:
            return dict(status="error", error=str(e))
        except KeyError as e:
            return dict(status="error", error=f"unknown reference: {e}")

    # ------------------------------------------------------------------ helpers
    def _session_exists(self, sid: str) -> bool:
        return self.ledger.one("SELECT 1 FROM sessions WHERE session_id=?", (sid,)) is not None

    def _budget_cfg(self, ev: dict) -> Optional[BudgetConfig]:
        b = ev.get("budget")
        if b is None:
            return None
        if not isinstance(b, dict):
            raise IngestError("'budget' must be an object")
        base = self.budget.config
        basis = b.get("basis", base.basis)
        if basis not in ("billed", "fresh"):
            raise IngestError("budget.basis must be 'billed' or 'fresh'")
        limit = _nonneg(b.get("limit", base.limit), "budget.limit")
        reserve = _nonneg(b.get("finalization_reserve", base.finalization_reserve), "budget.finalization_reserve")
        if limit <= 0 or reserve >= limit:
            raise IngestError("budget.limit must be > 0 and > finalization_reserve")
        return BudgetConfig(limit=limit, finalization_reserve=reserve, basis=basis)

    def _ensure_session(self, sid: str, ev: Optional[dict] = None) -> None:
        if self._session_exists(sid):
            return
        ev = ev if isinstance(ev, dict) else {}
        parent = _str(ev, "parent_session_id")
        if parent and not self._session_exists(parent):
            raise IngestError(f"parent_session_id {parent!r} does not exist")
        self.gw.start_session(sid, parent_session_id=parent, tenant=_str(ev, "tenant"), root_task=_str(ev, "root_task"),
                              budget=self._budget_cfg(ev))

    def status(self, session_id: str) -> dict:
        return budget_dict(self.budget.status(session_id))

    # ------------------------------------------------------------------ lifecycle events
    def _session_start(self, ev: dict) -> dict:
        sid = _str(ev, "session_id", required=True)
        if self._session_exists(sid):
            return dict(status="duplicate", session_id=sid)
        self._ensure_session(sid, ev)
        if ev.get("ts") is not None:                      # backfilled sessions keep their real start time
            self.ledger.execute("UPDATE sessions SET started_at=? WHERE session_id=?", (_num(ev["ts"], "ts"), sid))
        return dict(status="ok", session_id=sid, budget=self.status(sid))

    def _session_end(self, ev: dict) -> dict:
        sid = _str(ev, "session_id", required=True)
        st = ev.get("status", "completed")
        if st not in SESSION_STATUSES:
            raise IngestError(f"status must be one of {SESSION_STATUSES}")
        if not self._session_exists(sid):
            raise IngestError(f"unknown session {sid!r}")
        cp = ev.get("checkpoint")
        if cp is not None and not isinstance(cp, dict):
            raise IngestError("'checkpoint' must be an object")
        self.gw.end_session(sid, st, cp)
        if ev.get("ts") is not None:
            self.ledger.execute("UPDATE sessions SET ended_at=? WHERE session_id=?", (_num(ev["ts"], "ts"), sid))
        return dict(status="ok", session_id=sid)

    def _task_start(self, ev: dict) -> dict:
        tid, sid = _str(ev, "task_id", required=True), _str(ev, "session_id", required=True)
        if self.ledger.one("SELECT 1 FROM tasks WHERE task_id=?", (tid,)):
            return dict(status="duplicate", task_id=tid)
        self._ensure_session(sid)
        self.gw.start_task(tid, sid, _str(ev, "task_type"), _str(ev, "complexity_est"), _str(ev, "acceptance_ref"))
        return dict(status="ok", task_id=tid)

    def _task_end(self, ev: dict) -> dict:
        tid = _str(ev, "task_id", required=True)
        outcome = ev.get("outcome")
        if outcome not in OUTCOMES:
            raise IngestError(f"outcome must be one of {OUTCOMES}")
        if not self.ledger.one("SELECT 1 FROM tasks WHERE task_id=?", (tid,)):
            raise IngestError(f"unknown task {tid!r}")
        self.gw.finish_task(tid, outcome)
        return dict(status="ok", task_id=tid)

    def _routing(self, ev: dict) -> dict:
        did = _str(ev, "decision_id") or _uid()
        if self.ledger.one("SELECT 1 FROM routing_decisions WHERE decision_id=?", (did,)):
            return dict(status="duplicate", decision_id=did)
        sid = _str(ev, "session_id", required=True)
        self._ensure_session(sid)
        cands = ev.get("candidates") or []
        if not isinstance(cands, list) or not all(isinstance(c, str) for c in cands):
            raise IngestError("'candidates' must be a list of model ids")
        self.ledger.record_routing_decision(
            did, task_id=_str(ev, "task_id"), session_id=sid, chosen=_str(ev, "chosen", required=True), candidates=cands,
            reason_code=_str(ev, "reason_code"), complexity_features=ev.get("complexity_features") or {},
            router_version=_str(ev, "router_version"), escalated_from=_str(ev, "escalated_from"), now=self.clock())
        return dict(status="ok", decision_id=did)

    def _quality(self, ev: dict) -> dict:
        tid = _str(ev, "task_id", required=True)
        if not self.ledger.one("SELECT 1 FROM tasks WHERE task_id=?", (tid,)):
            raise IngestError(f"unknown task {tid!r}")
        score = ev.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise IngestError("'score' must be a number in [0,1]")
        try:
            record_eval(self.ledger, tid, ev.get("method", ""), float(score), rubric_version=_str(ev, "rubric_version"),
                        judge_model=_str(ev, "judge_model"), confidence=_num(ev.get("confidence"), "confidence"),
                        meta=ev.get("meta") if isinstance(ev.get("meta"), dict) else None)
        except ValueError as e:
            raise IngestError(str(e))
        return dict(status="ok", task_id=tid)

    # ------------------------------------------------------------------ model calls
    def _usage(self, ev: dict) -> Usage:
        u = ev.get("usage")
        if not isinstance(u, dict):
            raise IngestError("'usage' object is required")
        fmt = ev.get("usage_format", "native")
        if fmt == "native":
            usage = Usage(**{k: _nonneg(u.get(k, 0), f"usage.{k}") for k in
                             ("input_fresh", "output", "cache_read", "cache_write", "reasoning")})
        elif fmt in ("anthropic", "openai"):
            for k, v in u.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0:
                    raise IngestError(f"usage.{k} must be non-negative")
            usage = usage_from_anthropic(u) if fmt == "anthropic" else usage_from_openai(u)
        else:
            raise IngestError("usage_format must be native, anthropic or openai")
        if usage.reasoning > usage.output:
            raise IngestError("usage.reasoning is a subset of output and cannot exceed it")
        return usage

    def _segments(self, raw: Any) -> list[Segment]:
        if raw is None:
            return []
        if not isinstance(raw, list) or len(raw) > MAX_SEGMENTS:
            raise IngestError(f"'segments' must be a list (max {MAX_SEGMENTS})")
        out = []
        for i, s in enumerate(raw):
            if not isinstance(s, dict) or s.get("kind") not in SEGMENT_KINDS:
                raise IngestError(f"segments[{i}].kind must be one of {SEGMENT_KINDS}")
            kind, text = s["kind"], s.get("text")
            if text is not None:
                if not isinstance(text, str):
                    raise IngestError(f"segments[{i}].text must be a string")
                tokens = _nonneg(s["tokens"], "tokens") if "tokens" in s else self.estimator.count(text)
                chash = content_hash(text)
                fp = fingerprint(text) if kind in ("tool_result", "retrieval") else []
            else:
                if "tokens" not in s:
                    raise IngestError(f"segments[{i}] needs 'text' or 'tokens'")
                tokens = _nonneg(s["tokens"], f"segments[{i}].tokens")
                chash = s.get("content_hash") or f"unhashed:{_uid()}"        # no hash => cannot dedupe
                fp = s.get("fingerprint") or []
                if not isinstance(fp, list) or len(fp) > 64 or not all(isinstance(x, str) for x in fp):
                    raise IngestError(f"segments[{i}].fingerprint must be a list of <=64 strings")
            out.append(Segment(kind=kind, tokens=tokens, content_hash=str(chash)[:64], source_ref=_str(s, "source_ref"),
                               origin_call_id=_str(s, "origin_call_id"),
                               first_iteration=_nonneg(s["first_iteration"], "first_iteration") if "first_iteration" in s else None,
                               was_truncated=bool(s.get("was_truncated", False)), fingerprint=fp))
        return out

    def _model_call(self, ev: dict) -> dict:
        call_id = _str(ev, "call_id") or _uid()
        if self.ledger.one("SELECT 1 FROM model_calls WHERE call_id=?", (call_id,)):
            return dict(status="duplicate", call_id=call_id)
        sid, model = _str(ev, "session_id", required=True), _str(ev, "model_id", required=True)
        usage, segs = self._usage(ev), self._segments(ev.get("segments"))
        failed = ev.get("status", "success") == "failure"
        if ev.get("status", "success") not in ("success", "failure"):
            raise IngestError("status must be 'success' or 'failure'")
        tcs = ev.get("tool_calls")
        if tcs is not None and not (isinstance(tcs, list) and all(isinstance(t, dict) and isinstance(t.get("name"), str) for t in tcs)):
            raise IngestError("'tool_calls' must be a list of {name, args?}")
        n_tools = len(tcs) if tcs is not None else _nonneg(ev.get("tool_call_count", 0), "tool_call_count")
        out_text = ev.get("output_text") or ""
        if not isinstance(out_text, str):
            raise IngestError("'output_text' must be a string")
        latency = _num(ev.get("latency_ms"), "latency_ms") or 0.0
        req = CallRequest(session_id=sid, agent_id=_str(ev, "agent_id", default="unknown"), model_id=model, segments=segs,
                          task_id=_str(ev, "task_id"), iteration=_nonneg(ev.get("iteration", 1), "iteration"),
                          provider=_str(ev, "provider") or "", agent_version=_str(ev, "agent_version"),
                          prompt_version=_str(ev, "prompt_version"), routing_decision_id=_str(ev, "routing_decision_id"),
                          parent_call_id=_str(ev, "parent_call_id"), finalization=bool(ev.get("finalization", False)))
        err = ProviderError(str(ev.get("error_message") or "reported failure"), error_class=_str(ev, "error_class") or "unknown") if failed else None
        resp = None if failed else ProviderResponse(
            usage=usage, text=out_text, stop_reason=_str(ev, "stop_reason"),
            tool_calls=[ToolCallRequest(name=t["name"], args=t.get("args")) for t in tcs] if tcs is not None
            else [ToolCallRequest(name="?") for _ in range(n_tools)])
        if failed and out_text:
            resp = None
        with self.ledger.transaction():
            self._ensure_session(sid, ev.get("session"))
            rid = _str(ev, "reservation_id")
            commit = self._settle(rid, sid, usage)
            self.gw._record_call(req, call_id, _nonneg(ev.get("attempt_no", 1), "attempt_no"), _str(ev, "retry_of_call_id"),
                                 usage, resp, err, latency, sum(s.tokens for s in segs), commit.used_before, commit.used_after)
            if ev.get("ts") is not None:
                self.ledger.execute("UPDATE model_calls SET ts=? WHERE call_id=?", (_num(ev["ts"], "ts"), call_id))
            if ev.get("ttft_ms") is not None:
                self.ledger.execute("UPDATE model_calls SET ttft_ms=? WHERE call_id=?", (_num(ev["ttft_ms"], "ttft_ms"), call_id))
        return dict(status="ok", call_id=call_id, budget=budget_dict(commit.status))

    # ------------------------------------------------------------------ tool calls
    def _tool_call(self, ev: dict) -> dict:
        tcid = _str(ev, "tool_call_id") or _uid()
        if self.ledger.one("SELECT 1 FROM tool_calls WHERE tool_call_id=?", (tcid,)):
            return dict(status="duplicate", tool_call_id=tcid)
        call_id = _str(ev, "call_id", required=True)
        call = self.ledger.one("SELECT session_id FROM model_calls WHERE call_id=?", (call_id,))
        if call is None:
            raise IngestError(f"unknown call_id {call_id!r}: send the model_call event first (batches are processed in order)")
        status = ev.get("status", "ok")
        if status not in ("ok", "error", "memo_hit"):
            raise IngestError("status must be ok, error or memo_hit")
        raw, inj = ev.get("raw_output"), ev.get("injected_text")
        raw_tokens = _nonneg(ev["raw_output_tokens"], "raw_output_tokens") if "raw_output_tokens" in ev else (
            self.estimator.count(raw) if isinstance(raw, str) else 0)
        if "injected_tokens" in ev:
            inj_tokens = _nonneg(ev["injected_tokens"], "injected_tokens")
        elif isinstance(inj, str):
            inj_tokens = self.estimator.count(inj)
        else:
            inj_tokens = raw_tokens
        basis = inj if isinstance(inj, str) else raw
        result_hash = _str(ev, "result_hash") or (content_hash(basis) if isinstance(basis, str) else None)
        args_hash = _str(ev, "args_hash") or (content_hash(_canon(ev["args"])) if "args" in ev else None)
        self.ledger.insert("tool_calls", dict(
            tool_call_id=tcid, call_id=call_id, session_id=call["session_id"], tool_name=_str(ev, "tool_name", required=True),
            args_hash=args_hash, status=status, error_class=_str(ev, "error_class"), retry_of=_str(ev, "retry_of"),
            latency_ms=_num(ev.get("latency_ms"), "latency_ms"), raw_output_tokens=raw_tokens, injected_tokens=inj_tokens,
            was_truncated=int(bool(ev.get("was_truncated", inj_tokens < raw_tokens))), result_hash=result_hash))
        return dict(status="ok", tool_call_id=tcid)

    # ------------------------------------------------------------------ remote budget reservations
    def sweep_reservations(self) -> int:
        """Release reservations whose caller vanished (crash, timeout, server restart)."""
        n = 0
        with self.ledger.transaction():
            for r in self.ledger.query("SELECT * FROM budget_reservations WHERE expires_at < ?", (self.clock(),)):
                self.ledger.execute("UPDATE budget_state SET reserved=MAX(reserved-?,0) WHERE root_id=?", (r["amount"], r["root_id"]))
                self.ledger.execute("DELETE FROM budget_reservations WHERE reservation_id=?", (r["reservation_id"],))
                n += 1
        return n

    def reserve(self, ev: dict) -> tuple[int, dict]:
        """(http_status, body). 429 when the budget cannot fit the call."""
        self.sweep_reservations()
        try:
            sid = _str(ev, "session_id", required=True)
            est_in = _nonneg(ev.get("est_input_tokens"), "est_input_tokens")
            max_out = _nonneg(ev.get("max_output_tokens"), "max_output_tokens")
            ttl = _num(ev.get("ttl_seconds", DEFAULT_RESERVATION_TTL), "ttl_seconds") or DEFAULT_RESERVATION_TTL
            ttl = min(ttl, 3600.0)
            with self.ledger.transaction():
                self._ensure_session(sid, ev.get("session"))
                res = self.budget.reserve(sid, est_in, max_out, bool(ev.get("finalization", False)))
                rid = _uid()
                self.ledger.insert("budget_reservations", dict(
                    reservation_id=rid, root_id=res.root_id, session_id=sid, amount=res.amount, est_input=res.est_input,
                    granted_output=res.granted_output, finalization=int(res.finalization), expires_at=self.clock() + ttl))
            return 200, dict(reservation_id=rid, granted_max_output_tokens=res.granted_output, expires_in_seconds=ttl,
                             budget=budget_dict(res.status_before))
        except IngestError as e:
            return 400, dict(error=str(e))
        except BudgetExhausted as e:
            return 429, dict(error="budget_exhausted", reason=e.reason, finalization_available=e.finalization_available,
                             budget=budget_dict(e.status))

    def release(self, ev: dict) -> tuple[int, dict]:
        try:
            rid = _str(ev, "reservation_id", required=True)
        except IngestError as e:
            return 400, dict(error=str(e))
        with self.ledger.transaction():
            r = self.ledger.one("SELECT * FROM budget_reservations WHERE reservation_id=?", (rid,))
            if r is None:
                return 404, dict(error="unknown or expired reservation")
            self.ledger.execute("UPDATE budget_state SET reserved=MAX(reserved-?,0) WHERE root_id=?", (r["amount"], r["root_id"]))
            self.ledger.execute("DELETE FROM budget_reservations WHERE reservation_id=?", (rid,))
        return 200, dict(status="released")

    def _settle(self, reservation_id: Optional[str], session_id: str, usage: Usage):
        """Commit against a held reservation if given, else account the spend directly."""
        if reservation_id:
            r = self.ledger.one("SELECT * FROM budget_reservations WHERE reservation_id=?", (reservation_id,))
            if r is not None and r["session_id"] == session_id:
                self.ledger.execute("DELETE FROM budget_reservations WHERE reservation_id=?", (reservation_id,))
                res = Reservation(r["root_id"], session_id, r["amount"], r["est_input"], r["granted_output"],
                                  bool(r["finalization"]), self.budget.status(session_id))
                return self.budget.commit(res, usage) if usage.billed else self.budget.release(res)
            # expired/unknown reservation: still record the real spend so the ledger stays truthful
        return self.budget.record_usage(session_id, usage)
