import itertools

from agent_observability import (BudgetConfig, BudgetManager, CallRequest, ContextBuilder, Gateway, Ledger,
                                 ProviderResponse, Usage)
from agent_observability.tokens import HeuristicEstimator

_ids = itertools.count(1)


def words(prefix: str, n: int) -> str:
    """n distinctive tokens (each >= 6 chars so they fingerprint)."""
    return " ".join(f"{prefix}_{i:06d}" for i in range(n))


def make_gateway(budget: BudgetConfig = None, pricing=None):
    led = Ledger()
    est = HeuristicEstimator(chars_per_token=1.0)          # tokens == characters: easy arithmetic
    gw = Gateway(led, BudgetManager(led, budget or BudgetConfig()), pricing=pricing, estimator=est)
    return led, gw


def provider(out_text="", out_tokens=10, tool_calls=(), cache_read=0, stop="end_turn", reasoning=0):
    def fn(req, max_out, policy):
        fresh = sum(s.tokens for s in req.segments) - cache_read
        return ProviderResponse(usage=Usage(input_fresh=fresh, output=min(out_tokens, max_out), cache_read=cache_read,
                                            reasoning=reasoning),
                                text=out_text, stop_reason=stop, tool_calls=list(tool_calls))
    return fn


class Sim:
    """Drives a fake agent through the gateway."""

    def __init__(self, gw, session="S1", task="T1", agent="a1", model="m-big", task_type="fix",
                 complexity="medium", prompt_version="p1", budget=None, parent=None):
        self.gw, self.led = gw, gw.ledger
        self.session, self.task, self.agent, self.model = session, task, agent, model
        self.prompt_version = prompt_version
        self.iteration = 0
        gw.start_session(session, parent_session_id=parent, budget=budget)
        gw.start_task(task, session, task_type, complexity)
        self.ctx = ContextBuilder(gw.estimator)

    def step(self, out_text="", out_tokens=10, tool_calls=(), provider_fn=None, finalization=False, max_attempts=1,
             max_output_tokens=4096, model=None):
        self.iteration += 1
        req = CallRequest(session_id=self.session, agent_id=self.agent, model_id=model or self.model,
                          segments=list(self.ctx.segments), task_id=self.task, iteration=self.iteration,
                          prompt_version=self.prompt_version, finalization=finalization, max_attempts=max_attempts,
                          max_output_tokens=max_output_tokens)
        return self.gw.call(req, provider_fn or provider(out_text, out_tokens, tool_calls))

    def tool(self, call_id, name, args, output, **kw):
        tcid, seg = self.gw.record_tool_call(call_id, self.session, name, args, output, **kw)
        return tcid, self.ctx.add_segment(seg)

    def finish(self, outcome="success"):
        self.gw.finish_task(self.task, outcome)
        self.gw.end_session(self.session, "completed" if outcome == "success" else "failed")
