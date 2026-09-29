"""The single choke point every model call goes through.

Agents never write telemetry by hand: the gateway captures provider usage, context
composition, retries, latency, cost and budget state, and enforces the session budget.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable, Optional

from .budget import BudgetConfig, BudgetExhausted, BudgetManager, BudgetStatus
from .ledger import Ledger
from .models import (CallRequest, CallResult, ProviderError, ProviderResponse, Segment, ToolCallRequest, Usage)
from .pricing import PriceTable
from .tokens import HeuristicEstimator, TokenEstimator, content_hash, fingerprint

ProviderFn = Callable[[CallRequest, int, Any], ProviderResponse]


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _canon(obj: Any) -> str:
    return obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


class ContextBuilder:
    """Tracks the segments of an agent's context across iterations.

    Re-sending the same Segment objects preserves origin/age so context growth can be
    attributed to the call that introduced each block.
    """

    def __init__(self, estimator: Optional[TokenEstimator] = None, fingerprint_kinds=("tool_result", "retrieval")):
        self.estimator = estimator or HeuristicEstimator()
        self.segments: list[Segment] = []
        self._fp_kinds = fingerprint_kinds

    def add(self, kind: str, text: str, source_ref: Optional[str] = None, was_truncated: bool = False) -> Segment:
        seg = Segment(kind=kind, tokens=self.estimator.count(text), content_hash=content_hash(text),
                      source_ref=source_ref, was_truncated=was_truncated,
                      fingerprint=fingerprint(text) if kind in self._fp_kinds else [])
        self.segments.append(seg)
        return seg

    def add_segment(self, seg: Segment) -> Segment:
        self.segments.append(seg)
        return seg

    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.segments)

    def dedupe(self) -> int:
        """Drop later segments whose (kind, content_hash) already appeared. Returns tokens removed."""
        seen, keep, removed = set(), [], 0
        for s in self.segments:
            k = (s.kind, s.content_hash)
            if k in seen:
                removed += s.tokens
                continue
            seen.add(k)
            keep.append(s)
        self.segments = keep
        return removed

    def evict(self, predicate: Callable[[Segment], bool]) -> int:
        keep = [s for s in self.segments if not predicate(s)]
        removed = self.total_tokens() - sum(s.tokens for s in keep)
        self.segments = keep
        return removed

    def compact(self, kind: str, summary_text: str) -> Segment:
        """Replace every segment of `kind` with one summary. LOSSY: pin exact values (ids, paths,
        numbers, original user constraints) in a separate agent_state segment BEFORE compacting."""
        self.segments = [s for s in self.segments if s.kind != kind]
        return self.add(kind, summary_text)


class ToolMemo:
    """Memoise tool results by (tool, args, freshness token). Args alone are NOT enough: a file
    edited or test re-run changes the answer, so the caller supplies `state_token`."""

    def __init__(self):
        self._cache: dict[tuple, Any] = {}

    def get(self, tool: str, args: Any, state_token: str = "") -> Optional[Any]:
        return self._cache.get((tool, content_hash(_canon(args)), state_token))

    def put(self, tool: str, args: Any, value: Any, state_token: str = "") -> None:
        self._cache[(tool, content_hash(_canon(args)), state_token)] = value


class Gateway:
    def __init__(self, ledger: Ledger, budget: Optional[BudgetManager] = None, pricing: Optional[PriceTable] = None,
                 estimator: Optional[TokenEstimator] = None, clock=time.time, monotonic=time.monotonic):
        self.ledger = ledger
        self.budget = budget or BudgetManager(ledger)
        self.pricing = pricing or PriceTable()
        self.estimator = estimator or HeuristicEstimator()
        self.clock = clock
        self.monotonic = monotonic

    # ------------------------------------------------------------ session / task lifecycle
    def start_session(self, session_id: str, *, parent_session_id: Optional[str] = None, tenant: Optional[str] = None,
                      root_task: Optional[str] = None, budget: Optional[BudgetConfig] = None) -> str:
        root = self.ledger.create_session(session_id, parent_session_id=parent_session_id, tenant=tenant,
                                          root_task=root_task, now=self.clock())
        if root == session_id:                          # subagents inherit their root's budget
            self.budget.open_root(root, budget)
        return root

    def end_session(self, session_id: str, status: str = "completed", checkpoint: Optional[dict] = None) -> None:
        if checkpoint:
            self.ledger.save_checkpoint(session_id, checkpoint)
        self.ledger.end_session(session_id, status, now=self.clock())

    def start_task(self, task_id: str, session_id: str, task_type: Optional[str] = None,
                   complexity_est: Optional[str] = None, acceptance_ref: Optional[str] = None) -> None:
        self.ledger.create_task(task_id, session_id, task_type, complexity_est, acceptance_ref, now=self.clock())

    def finish_task(self, task_id: str, outcome: str) -> None:
        self.ledger.finish_task(task_id, outcome, now=self.clock())

    # ------------------------------------------------------------ the model call
    def call(self, req: CallRequest, provider_fn: ProviderFn) -> CallResult:
        retry_of: Optional[str] = None
        attempt = 1
        while True:
            call_id = _new_id()
            est_input = sum(s.tokens for s in req.segments)
            try:
                res = self.budget.reserve(req.session_id, est_input, req.max_output_tokens, req.finalization)
            except BudgetExhausted as e:
                self._record_blocked(req, call_id, attempt, retry_of, e)
                if e.reason == "hard_stop":
                    self.ledger.set_terminal_status(req.session_id, "budget_exhausted")
                raise
            policy = res.status_before.policy
            max_out = res.granted_output
            if policy.output_cap and not req.finalization:
                max_out = min(max_out, policy.output_cap)

            t0 = self.monotonic()
            resp: Optional[ProviderResponse] = None
            err: Optional[ProviderError] = None
            try:
                resp = provider_fn(req, max_out, policy)
                usage = resp.usage
            except ProviderError as e:
                err, usage = e, (e.usage or Usage())
            except Exception as e:  # unknown provider failure: record, do not retry
                err = ProviderError(str(e), error_class=type(e).__name__, retryable=False)
                usage = Usage()
            latency_ms = (self.monotonic() - t0) * 1000.0

            commit = self.budget.commit(res, usage) if usage.billed else self.budget.release(res)
            self._record_call(req, call_id, attempt, retry_of, usage, resp, err, latency_ms, est_input,
                              commit.used_before, commit.used_after)
            if err is None:
                return CallResult(call_id, resp, usage, attempt, commit.status)
            if err.retryable and attempt < req.max_attempts:
                retry_of, attempt = call_id, attempt + 1
                continue
            err.call_id = call_id
            raise err

    def _write_segments(self, req: CallRequest, call_id: str) -> None:
        for pos, s in enumerate(req.segments):
            if s.origin_call_id is None:
                s.origin_call_id = call_id
            if s.first_iteration is None:
                s.first_iteration = req.iteration
            self.ledger.insert("context_segments", dict(
                segment_id=f"{call_id}:{pos}", call_id=call_id, session_id=req.session_id, kind=s.kind,
                tokens=s.tokens, content_hash=s.content_hash, source_ref=s.source_ref, origin_call_id=s.origin_call_id,
                position=pos, age_iterations=max(req.iteration - s.first_iteration, 0),
                was_truncated=int(s.was_truncated), fingerprint=json.dumps(s.fingerprint) if s.fingerprint else None))

    def _record_call(self, req, call_id, attempt, retry_of, usage: Usage, resp, err, latency_ms, est_input,
                     used_before, used_after) -> None:
        cost_in, cost_out = self.pricing.cost(req.model_id, usage)
        out_fp: list[str] = []
        if resp is not None:
            args_text = " ".join(_canon(tc.args) for tc in resp.tool_calls if tc.args is not None)
            out_fp = fingerprint(f"{resp.text} {args_text}", limit=2048)
        self.ledger.insert("model_calls", dict(
            call_id=call_id, session_id=req.session_id, task_id=req.task_id, agent_id=req.agent_id,
            agent_version=req.agent_version, prompt_version=req.prompt_version, iteration=req.iteration,
            parent_call_id=req.parent_call_id, retry_of_call_id=retry_of, attempt_no=attempt, model_id=req.model_id,
            provider=req.provider, routing_decision_id=req.routing_decision_id, input_fresh=usage.input_fresh,
            cache_read=usage.cache_read, cache_write=usage.cache_write, output_tokens=usage.output,
            reasoning_tokens=usage.reasoning, est_input_tokens=est_input,
            est_error=(usage.input_total - est_input) if usage.input_total else None,
            cost_in=cost_in, cost_out=cost_out, price_table_version=self.pricing.version, latency_ms=latency_ms,
            ttft_ms=getattr(resp, "ttft_ms", None), stop_reason=getattr(resp, "stop_reason", None),
            status="failure" if err else "success", error_class=err.error_class if err else None,
            budget_used_before=used_before, budget_used_after=used_after,
            tool_call_count=len(resp.tool_calls) if resp else 0, finalization=int(req.finalization),
            output_fingerprint=json.dumps(out_fp) if out_fp else None, ts=self.clock()))
        if usage.billed:                       # failed-but-billed calls still consumed this context
            self._write_segments(req, call_id)

    def _record_blocked(self, req, call_id, attempt, retry_of, e: BudgetExhausted) -> None:
        self.ledger.insert("model_calls", dict(
            call_id=call_id, session_id=req.session_id, task_id=req.task_id, agent_id=req.agent_id,
            agent_version=req.agent_version, prompt_version=req.prompt_version, iteration=req.iteration,
            retry_of_call_id=retry_of, attempt_no=attempt, model_id=req.model_id, provider=req.provider,
            est_input_tokens=sum(s.tokens for s in req.segments), status="blocked_budget", error_class=e.reason,
            budget_used_before=e.status.used, budget_used_after=e.status.used,
            finalization=int(req.finalization), ts=self.clock()))

    # ------------------------------------------------------------ tools
    def record_tool_call(self, call_id: str, session_id: str, tool_name: str, args: Any, raw_output: str, *,
                         injected_text: Optional[str] = None, max_inject_tokens: Optional[int] = None,
                         status: str = "ok", error_class: Optional[str] = None, retry_of: Optional[str] = None,
                         latency_ms: Optional[float] = None, tool_call_id: Optional[str] = None
                         ) -> tuple[str, Segment]:
        """Log a tool call and return (tool_call_id, tool_result Segment) ready to add to context.

        raw_output vs injected_text separates 'what the tool produced' from 'what the model saw'.
        If only max_inject_tokens is given, output is truncated to that budget (flagged truncated).
        """
        tcid = tool_call_id or _new_id()
        raw_tokens = self.estimator.count(raw_output)
        text, truncated = raw_output if injected_text is None else injected_text, injected_text is not None and injected_text != raw_output
        if injected_text is None and max_inject_tokens is not None and raw_tokens > max_inject_tokens:
            cpt = len(raw_output) / max(raw_tokens, 1)
            text, truncated = raw_output[: int(max_inject_tokens * cpt)], True
        seg = Segment(kind="tool_result", tokens=self.estimator.count(text), content_hash=content_hash(text),
                      source_ref=tcid, was_truncated=truncated, fingerprint=fingerprint(text))
        self.ledger.insert("tool_calls", dict(
            tool_call_id=tcid, call_id=call_id, session_id=session_id, tool_name=tool_name,
            args_hash=content_hash(_canon(args)), status=status, error_class=error_class, retry_of=retry_of,
            latency_ms=latency_ms, raw_output_tokens=raw_tokens, injected_tokens=seg.tokens,
            was_truncated=int(truncated), result_hash=seg.content_hash))
        return tcid, seg

    def record_memo_hit(self, call_id: str, session_id: str, tool_name: str, args: Any) -> str:
        """A duplicate tool call avoided via ToolMemo (kept in the ledger for observability)."""
        tcid = _new_id()
        self.ledger.insert("tool_calls", dict(
            tool_call_id=tcid, call_id=call_id, session_id=session_id, tool_name=tool_name,
            args_hash=content_hash(_canon(args)), status="memo_hit"))
        return tcid
