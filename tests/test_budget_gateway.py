import threading
import unittest

from agent_observability import (BudgetConfig, BudgetExhausted, CallRequest, ProviderError, Usage,
                                 usage_from_anthropic, usage_from_openai)
from helpers import Sim, make_gateway, provider


class UsageAdapters(unittest.TestCase):
    def test_anthropic_input_excludes_cache(self):
        u = usage_from_anthropic({"input_tokens": 100, "output_tokens": 40, "cache_read_input_tokens": 900,
                                  "cache_creation_input_tokens": 50})
        self.assertEqual((u.input_fresh, u.cache_read, u.cache_write, u.input_total, u.billed), (100, 900, 50, 1050, 1090))

    def test_openai_prompt_includes_cached(self):
        u = usage_from_openai({"prompt_tokens": 1000, "completion_tokens": 30,
                               "prompt_tokens_details": {"cached_tokens": 800},
                               "completion_tokens_details": {"reasoning_tokens": 10}})
        self.assertEqual((u.input_fresh, u.cache_read, u.output, u.reasoning), (200, 800, 30, 10))

    def test_budget_basis(self):
        u = Usage(input_fresh=100, cache_read=900, output=50)
        self.assertEqual(u.counted("billed"), 1050)
        self.assertEqual(u.counted("fresh"), 150)


class BudgetTests(unittest.TestCase):
    def test_thresholds_fire_once_in_order_and_hard_stop(self):
        led, gw = make_gateway(BudgetConfig(limit=1000, finalization_reserve=50, min_output_tokens=1,
                                            input_safety_margin=1.0))
        sim = Sim(gw)
        sim.ctx.add("system", "x" * 90)
        levels = [sim.step(out_tokens=10).budget.level for _ in range(9)]      # 100 tokens/call
        self.assertEqual(levels[5], "warning")                                  # 600 used = 60%
        self.assertEqual(levels[7], "high")                                     # 800 = 80%
        self.assertEqual(levels[8], "critical")                                 # 900 = 90%
        with self.assertRaises(BudgetExhausted) as cm:                          # 900 + reserve: normal calls blocked
            sim.step(out_tokens=10)
        self.assertEqual(cm.exception.reason, "reserve_only")
        sim.step(out_tokens=100, finalization=True)                             # wrap-up spends the reserve
        with self.assertRaises(BudgetExhausted) as cm:
            sim.step(out_tokens=10, finalization=True)
        self.assertEqual(cm.exception.reason, "hard_stop")
        events = [e["level"] for e in led.query("SELECT level FROM budget_events ORDER BY id")]
        self.assertEqual(events, ["warning", "high", "critical", "hard_stop"])
        self.assertEqual(led.one("SELECT terminal_status t FROM sessions")["t"], "budget_exhausted")
        self.assertLessEqual(gw.budget.status("S1").used, 1000)

    def test_reserve_only_then_finalization_allowed(self):
        led, gw = make_gateway(BudgetConfig(limit=1000, finalization_reserve=200, min_output_tokens=10))
        sim = Sim(gw)
        sim.ctx.add("system", "x" * 350)
        sim.step(out_tokens=100)                               # used 450; normal ceiling = 800
        with self.assertRaises(BudgetExhausted) as cm:
            sim.step(out_tokens=100)                           # est_in 368 > 350 available, fits inside 1000
        self.assertEqual(cm.exception.reason, "reserve_only")
        r = sim.step(out_tokens=50, finalization=True)         # wrap-up may use the reserve
        self.assertLessEqual(r.budget.used, 1000)
        self.assertEqual(len(led.query("SELECT * FROM model_calls WHERE status='blocked_budget'")), 1)

    def test_preflight_shrinks_max_output_never_overshoots(self):
        led, gw = make_gateway(BudgetConfig(limit=1000, finalization_reserve=0, min_output_tokens=1, input_safety_margin=1.0))
        sim = Sim(gw)
        sim.ctx.add("system", "x" * 900)
        seen = {}

        def fn(req, max_out, policy):
            seen["max_out"] = max_out
            from agent_observability import ProviderResponse
            return ProviderResponse(usage=Usage(input_fresh=900, output=max_out))
        sim.step(provider_fn=fn, max_output_tokens=4096)
        self.assertEqual(seen["max_out"], 100)
        self.assertLessEqual(gw.budget.status("S1").used, 1000)

    def test_subagents_share_root_budget(self):
        led, gw = make_gateway(BudgetConfig(limit=1000, finalization_reserve=0, min_output_tokens=1))
        parent = Sim(gw, session="P", task="TP")
        parent.ctx.add("system", "x" * 300)
        parent.step(out_tokens=100)
        child = Sim(gw, session="C", task="TC", parent="P")
        child.ctx.add("system", "y" * 400)
        child.step(out_tokens=100)
        self.assertEqual(gw.budget.status("P").used, 900)
        self.assertEqual(gw.budget.status("C").used, 900)
        with self.assertRaises(BudgetExhausted):
            parent.step(out_tokens=100)

    def test_concurrent_reservations_never_exceed_limit(self):
        led, gw = make_gateway(BudgetConfig(limit=5000, finalization_reserve=0, min_output_tokens=1, input_safety_margin=1.0))
        gw.start_session("C")
        errors, ok = [], []

        def worker():
            from agent_observability import ContextBuilder
            ctx = ContextBuilder(gw.estimator)
            ctx.add("system", "x" * 300)
            for i in range(10):
                try:
                    gw.call(CallRequest(session_id="C", agent_id="a", model_id="m", segments=list(ctx.segments),
                                        iteration=i + 1, max_output_tokens=200), provider(out_tokens=200))
                    ok.append(1)
                except BudgetExhausted:
                    errors.append(1)
        ts = [threading.Thread(target=worker) for _ in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        st = gw.budget.status("C")
        self.assertLessEqual(st.used, 5000)
        self.assertEqual(st.reserved, 0)
        self.assertTrue(errors)


class GatewayTests(unittest.TestCase):
    def test_retry_chain_and_failed_tokens_count(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        sim.ctx.add("system", "x" * 200)
        n = {"i": 0}

        def flaky(req, max_out, policy):
            n["i"] += 1
            if n["i"] < 3:
                raise ProviderError("429", error_class="rate_limit", retryable=True,
                                    usage=Usage(input_fresh=200, output=0))
            return provider("ok", 20)(req, max_out, policy)
        r = sim.step(provider_fn=flaky, max_attempts=3)
        rows = led.query("SELECT call_id, attempt_no, retry_of_call_id, status, error_class FROM model_calls ORDER BY seq")
        self.assertEqual([x["attempt_no"] for x in rows], [1, 2, 3])
        self.assertIsNone(rows[0]["retry_of_call_id"])
        self.assertEqual(rows[1]["retry_of_call_id"], rows[0]["call_id"])
        self.assertEqual(rows[2]["retry_of_call_id"], rows[1]["call_id"])
        self.assertEqual([x["status"] for x in rows], ["failure", "failure", "success"])
        self.assertEqual(r.attempts, 3)
        self.assertEqual(gw.budget.status("S1").used, 200 + 200 + 220)       # failed attempts were billed

    def test_unbilled_failure_releases_reservation(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        sim.ctx.add("system", "x" * 100)

        def boom(req, max_out, policy):
            raise ProviderError("down", error_class="connection", retryable=False)
        with self.assertRaises(ProviderError):
            sim.step(provider_fn=boom)
        st = gw.budget.status("S1")
        self.assertEqual((st.used, st.reserved), (0, 0))

    def test_trajectory_reconstructs_session(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        sim.ctx.add("system", "s" * 300)
        sim.ctx.add("user", "u" * 100)
        r1 = sim.step(out_tokens=50)
        sim.tool(r1.call_id, "search", {"q": "a"}, "r" * 400)
        sim.step(out_tokens=60)
        traj = led.trajectory("S1")
        self.assertEqual(len(traj["steps"]), 2)
        self.assertEqual(traj["steps"][1]["cumulative_billed"], 450 + 860)
        self.assertEqual(traj["total_billed"], gw.budget.status("S1").used)
        self.assertEqual(traj["steps"][1]["call"]["budget_used_after"], traj["total_billed"])
        kinds = [s["kind"] for s in traj["steps"][1]["segments"]]
        self.assertEqual(kinds, ["system", "user", "tool_result"])
        self.assertEqual(traj["steps"][0]["tool_calls"][0]["tool_name"], "search")

    def test_estimate_error_recorded(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        sim.ctx.add("system", "x" * 100)

        def off(req, max_out, policy):
            from agent_observability import ProviderResponse
            return ProviderResponse(usage=Usage(input_fresh=110, output=5))
        sim.step(provider_fn=off)
        self.assertEqual(led.one("SELECT est_error FROM model_calls")["est_error"], 10)

    def test_segment_origin_and_age_carry_over(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        sim.ctx.add("system", "s" * 100)
        c1 = sim.step()
        sim.ctx.add("user", "u" * 50)
        sim.step()
        sim.step()
        rows = led.query("SELECT kind, origin_call_id, age_iterations FROM context_segments ORDER BY call_id, position")
        sys_rows = [r for r in rows if r["kind"] == "system"]
        self.assertTrue(all(r["origin_call_id"] == c1.call_id for r in sys_rows))
        self.assertEqual(sorted(r["age_iterations"] for r in sys_rows), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
