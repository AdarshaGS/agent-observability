"""Populate a ledger with varied synthetic sessions so the UI has something to show.

    python3 examples/seed_demo.py demo.db && python3 -m agent_observability serve --db demo.db --open

Synthetic data only: agents, tokens and prices are made up.
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_observability import (BudgetConfig, BudgetExhausted, BudgetManager, CallRequest, ContextBuilder, Gateway, Ledger,
                                 ModelPrice, PriceTable, ProviderError, ProviderResponse, Usage, record_eval)
from agent_observability.tokens import HeuristicEstimator


def words(prefix, n):
    return " ".join(f"{prefix}_{i:06d}" for i in range(n))


def main(db: str) -> None:
    rng = random.Random(7)
    led = Ledger(db)
    prices = PriceTable("EXAMPLE-PLACEHOLDER-PRICES")
    prices.register("large-model", ModelPrice(input=15.0, output=75.0, cache_read=1.5, cache_write=18.75))
    prices.register("small-model", ModelPrice(input=1.0, output=5.0, cache_read=0.1, cache_write=1.25))
    now = time.time()
    clock = {"t": now - 3 * 86400}
    gw = Gateway(led, BudgetManager(led, BudgetConfig(limit=300_000)), pricing=prices,
                 estimator=HeuristicEstimator(4.0), clock=lambda: clock["t"])

    def provider(text, out, cache_read=0, fail_first=False, state={"n": 0}):
        def fn(req, max_out, policy):
            total = sum(s.tokens for s in req.segments)
            cr = min(cache_read, total)
            return ProviderResponse(usage=Usage(input_fresh=total - cr, cache_read=cr, output=min(out, max_out)), text=text)
        return fn

    profiles = [  # agent, model, task_type, complexity, sys tokens, iterations, tool size, waste behaviour
        ("triage-bot", "small-model", "classify", "simple", 900, (1, 3), 300, None),
        ("triage-bot", "large-model", "classify", "simple", 900, (1, 3), 300, None),
        ("code-agent", "large-model", "refactor", "medium", 6000, (5, 9), 2500, "leaky"),
        ("code-agent", "large-model", "refactor", "medium", 6000, (5, 9), 12000, "oversized"),
        ("research-agent", "large-model", "research", "complex", 3000, (6, 14), 4000, "loop"),
    ]
    n = 0
    for rep in range(6):
        for agent, model, ttype, cx, sys_tok, its, tool_size, waste in profiles:
            n += 1
            sid, tid = f"sess-{n:03d}", f"task-{n:03d}"
            clock["t"] += rng.uniform(600, 4000)
            gw.start_session(sid, root_task=f"{ttype} job {n}")
            gw.start_task(tid, sid, ttype, cx)
            ctx = ContextBuilder(gw.estimator)
            ctx.add("system", "instructions " * (sys_tok * 4 // 13))
            ctx.add("tool_schema", "schema " * (sys_tok // 3))
            ctx.add("user", "please handle the request " * 8)
            iterations = rng.randint(*its)
            outcome = "success" if rng.random() > (0.35 if cx == "complex" else 0.12) else "failure"
            if n == 26:                       # one runaway session that exhausts the budget
                iterations, tool_size, waste = 60, 15000, "oversized"
            try:
                for i in range(1, iterations + 1):
                    text = words(f"ref{n}x{i}", 12)
                    fn = provider(text, rng.randint(150, 700) if waste != "loop" else rng.randint(600, 1600),
                                  cache_read=(sys_tok + sys_tok // 3) if i > 1 and rng.random() > 0.15 else 0)
                    try:
                        r = gw.call(CallRequest(session_id=sid, agent_id=agent, model_id=model, segments=list(ctx.segments),
                                                task_id=tid, iteration=i, prompt_version="v2" if rep >= 3 else "v1",
                                                max_attempts=2), fn)
                    except ProviderError:
                        continue
                    if r.budget.policy.dedupe_tool_calls:
                        ctx.dedupe()
                    if r.budget.policy.compact_history:
                        ctx.evict(lambda s: s.kind == "tool_result" and s.first_iteration is not None and i - s.first_iteration > 2)
                    out = words(f"res{n}x{i}", max(tool_size // 13, 5))
                    tool_args = {"q": "same query"} if waste == "loop" and i % 3 else {"q": f"q{i}"}
                    _, seg = gw.record_tool_call(r.call_id, sid, "search_repo" if agent != "triage-bot" else "lookup",
                                                 tool_args, out, latency_ms=rng.uniform(30, 600))
                    ctx.add_segment(seg)
                    if waste == "leaky" and i == 2:                    # duplicated retrieval block
                        ctx.add("retrieval", words("doc", 300))
                        ctx.add("retrieval", words("doc", 300))
            except BudgetExhausted:
                outcome = "budget_exhausted"
            gw.finish_task(tid, outcome)
            gw.end_session(sid, "budget_exhausted" if outcome == "budget_exhausted" else
                           "completed" if outcome == "success" else "failed")
            record_eval(led, tid, "test", 1.0 if outcome == "success" else rng.choice([0.0, 0.3]), session_id=sid)
    print(f"seeded {n} sessions into {db}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo.db")
