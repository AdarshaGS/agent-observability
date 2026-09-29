"""Simulate a deliberately wasteful agent session and print the report.

    python3 examples/simulate.py [ledger.db]

Shows the intended integration: the agent only talks to `Gateway`; telemetry, budget enforcement and
analysis follow from that. Replace `fake_provider` with a real SDK call and return `Usage` via
`usage_from_anthropic(resp.usage)` / `usage_from_openai(resp.usage)`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_observability import (BudgetConfig, BudgetExhausted, BudgetManager, CallRequest, ContextBuilder, Gateway,
                                 Ledger, ModelPrice, PriceTable, ProviderResponse, RoutingConfig, ToolMemo, Usage,
                                 build_session_report, record_eval, render_text)


def words(prefix, n):
    return " ".join(f"{prefix}_{i:06d}" for i in range(n))


def main(db: str = ":memory:") -> None:
    led = Ledger(db)
    prices = PriceTable("EXAMPLE-PLACEHOLDER-PRICES")          # illustrative only; load your real table
    prices.register("large-model", ModelPrice(input=15.0, output=75.0, cache_read=1.5, cache_write=18.75))
    prices.register("small-model", ModelPrice(input=1.0, output=5.0))
    gw = Gateway(led, BudgetManager(led, BudgetConfig(limit=300_000)), pricing=prices)

    gw.start_session("demo", root_task="refactor billing module")
    gw.start_task("demo-t1", "demo", task_type="refactor", complexity_est="medium")
    ctx = ContextBuilder(gw.estimator)
    ctx.add("system", "You are a careful refactoring agent. " * 200)
    ctx.add("tool_schema", "tool schema json " * 400)
    ctx.add("user", "Refactor the billing module to remove duplicated tax logic.")
    memo = ToolMemo()

    def fake_provider(text, out, tools=0):
        def fn(req, max_out, policy):
            fresh = sum(s.tokens for s in req.segments)
            return ProviderResponse(usage=Usage(input_fresh=fresh, output=min(out, max_out)), text=text)
        return fn

    big_listing = words("billing_file", 1500)                  # oversized tool output
    try:
        for i in range(1, 9):
            r = gw.call(CallRequest(session_id="demo", agent_id="refactor-agent", model_id="large-model",
                                    segments=list(ctx.segments), task_id="demo-t1", iteration=i,
                                    prompt_version="v3"), fake_provider("continuing analysis " * 20, 600))
            if r.budget.policy.inject_state_line:
                print(r.budget.state_line())
            if r.budget.policy.dedupe_tool_calls:
                ctx.dedupe()
            if memo.get("list_dir", {"path": "billing/"}) is None:      # first call only; later ones hit the memo
                _, seg = gw.record_tool_call(r.call_id, "demo", "list_dir", {"path": "billing/"}, big_listing)
                ctx.add_segment(seg)
                memo.put("list_dir", {"path": "billing/"}, seg)
            else:
                gw.record_memo_hit(r.call_id, "demo", "list_dir", {"path": "billing/"})
    except BudgetExhausted as e:
        print("stopped:", e)
    gw.finish_task("demo-t1", "success")
    gw.end_session("demo", "completed")
    record_eval(led, "demo-t1", "test", 1.0, rubric_version="unit-tests-v1")

    report = build_session_report(led, "demo", routing_config=RoutingConfig(tiers={"large-model": 2, "small-model": 1}),
                                  pricing=prices)
    print()
    print(render_text(report))


if __name__ == "__main__":
    main(*(sys.argv[1:2] or [":memory:"]))
