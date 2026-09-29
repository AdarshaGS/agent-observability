import unittest

from agent_observability import ProviderError, ProviderResponse, Usage, WasteConfig, analyze_waste
from helpers import Sim, make_gateway, provider, words


def by_cat(wa):
    return {f.category: f for f in wa.findings}


class WasteDetectors(unittest.TestCase):
    def test_duplicated_context(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        t = words("dup", 30)
        sim.ctx.add("tool_result", t)
        sim.ctx.add("tool_result", t)
        sim.step(out_text=t)
        f = by_cat(analyze_waste(led, "S1"))["duplicated_context"]
        self.assertEqual(f.tokens_wasted, len(t))
        self.assertEqual(f.confidence, "measured")

    def test_repeated_tool_call(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        sim.ctx.add("system", "s" * 100)
        R = words("rep", 40)
        c1 = sim.step()
        _, seg = sim.tool(c1.call_id, "search", {"q": "a"}, R)
        c2 = sim.step(out_text=R)
        sim.ctx.evict(lambda s: s is seg)
        sim.tool(c2.call_id, "search", {"q": "a"}, R)          # same args, unchanged result
        sim.step(out_text=R)
        wa = analyze_waste(led, "S1")
        f = by_cat(wa)["repeated_tool_calls"]
        self.assertEqual(f.tokens_wasted, len(R))
        self.assertNotIn("unnecessary_tool_calls", by_cat(wa))

    def test_unnecessary_vs_referenced_tool_calls_and_persisted_flag(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        sim.ctx.add("system", "s" * 100)
        R1, R2 = words("unn", 50), words("use", 50)
        c1 = sim.step()
        t1, _ = sim.tool(c1.call_id, "grep", {"q": 1}, R1)
        t2, _ = sim.tool(c1.call_id, "read", {"f": 2}, R2)
        sim.step(out_text=R2)
        wa = analyze_waste(led, "S1")
        self.assertEqual(by_cat(wa)["unnecessary_tool_calls"].tokens_wasted, len(R1))
        flags = {r["tool_call_id"]: r["later_referenced"] for r in led.query("SELECT tool_call_id, later_referenced FROM tool_calls")}
        self.assertEqual((flags[t1], flags[t2]), (0, 1))

    def test_failed_retry_calls(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        sim.ctx.add("system", "s" * 200)
        n = {"i": 0}

        def flaky(req, max_out, policy):
            n["i"] += 1
            if n["i"] == 1:
                raise ProviderError("bad args", error_class="invalid_request", retryable=True,
                                    usage=Usage(input_fresh=200, output=0))
            return provider("ok", 10)(req, max_out, policy)
        sim.step(provider_fn=flaky, max_attempts=2)
        f = by_cat(analyze_waste(led, "S1"))["failed_retry_calls"]
        self.assertEqual(f.tokens_wasted, 200)

    def test_oversized_tool_output_counts_only_the_excess(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        R = words("big", 300)
        c1 = sim.step()
        sim.tool(c1.call_id, "dump", {}, R)
        sim.step(out_text=R)
        f = by_cat(analyze_waste(led, "S1", config=WasteConfig(tool_output_cap=1000)))["oversized_tool_output"]
        self.assertEqual(f.tokens_wasted, len(R) - 1000)

    def test_context_accumulation_after_last_use(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        sim.ctx.add("system", "s" * 50)
        R = words("acc", 30)
        c1 = sim.step()
        sim.tool(c1.call_id, "read", {}, R)
        sim.step(out_text=R)                                  # k=1: last use
        for _ in range(6):                                    # k=2..7 carry R unused
            sim.step(out_text="continuing work")
        f = by_cat(analyze_waste(led, "S1"))["context_accumulation"]
        self.assertEqual(f.tokens_wasted, 3 * len(R))         # k=5,6,7 exceed last_use + window(3)

    def test_irrelevant_retrieval(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        r = words("ret", 20)
        sim.ctx.add("retrieval", r)
        sim.step(out_text="answer without citing")
        self.assertEqual(by_cat(analyze_waste(led, "S1"))["irrelevant_context"].tokens_wasted, len(r))

    def test_redundant_near_duplicate_retrievals(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        base = words("nd", 40)
        sim.ctx.add("retrieval", base)
        sim.ctx.add("retrieval", base + " extra_token_zz")
        sim.step(out_text=base)
        f = by_cat(analyze_waste(led, "S1"))["redundant_context"]
        self.assertEqual(f.tokens_wasted, len(base))

    def test_poor_routing_counts_attempts_on_escalated_model(self):
        led, gw = make_gateway()
        sim = Sim(gw, model="m-small")
        sim.ctx.add("system", "s" * 300)
        sim.step(out_tokens=20, model="m-small")
        led.record_routing_decision("d2", task_id="T1", session_id="S1", chosen="m-big", candidates=["m-small", "m-big"],
                                    reason_code="escalate_after_failure", escalated_from="m-small")
        sim.step(out_tokens=20, model="m-big")
        f = by_cat(analyze_waste(led, "S1"))["poor_routing"]
        self.assertEqual(f.tokens_wasted, 320)

    def test_cache_miss_is_cost_only(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        sim.ctx.add("system", "s" * 2000)

        def cached(read, write):
            def fn(req, max_out, policy):
                total = sum(s.tokens for s in req.segments)
                return ProviderResponse(usage=Usage(input_fresh=total - read - write, cache_read=read, cache_write=write, output=10))
            return fn
        sim.step(provider_fn=cached(0, 2000))
        sim.step(provider_fn=cached(2000, 0))
        sim.step(provider_fn=cached(0, 0))                    # prefix cache lost
        f = by_cat(analyze_waste(led, "S1"))["cache_miss"]
        self.assertEqual((f.unit, f.tokens_wasted), ("cost_only", 2000))

    def test_tool_loop_flagged_as_excessive_iterations(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        sim.ctx.add("system", "s" * 100)
        for _ in range(4):
            r = sim.step(out_text="thinking")
            sim.tool(r.call_id, "ls", {"p": "."}, "same listing output here")
        f = by_cat(analyze_waste(led, "S1"))["excessive_iterations"]
        self.assertGreater(f.tokens_wasted, 0)

    def test_baseline_driven_output_and_iteration_findings(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        sim.ctx.add("system", "s" * 100)
        for _ in range(3):
            sim.step(out_tokens=500)
        base = {"fix": {"iterations_p90": 2, "output_p50": 100}}
        cats = by_cat(analyze_waste(led, "S1", baselines=base))
        self.assertIn("excessive_iterations", cats)
        self.assertIn("unnecessary_output", cats)

    def test_categories_are_exclusive_and_bounded(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        sim.ctx.add("system", "s" * 500)
        T = words("mix", 60)
        sim.ctx.add("tool_result", T)
        sim.ctx.add("tool_result", T)
        sim.ctx.add("retrieval", words("noise", 30))
        c1 = sim.step(out_text="x")
        sim.tool(c1.call_id, "grep", {"a": 1}, words("zzz", 80))
        for _ in range(4):
            sim.step(out_text="no references here")
        wa = analyze_waste(led, "S1", config=WasteConfig(tool_output_cap=100))
        total = sum(f.tokens_wasted for f in wa.token_findings())
        self.assertLessEqual(total, wa.total_billed)
        self.assertEqual(wa.required_context_tokens, wa.segment_tokens - sum(
            f.tokens_wasted for f in wa.token_findings() if f.category not in ("failed_retry_calls", "excessive_iterations",
                                                                             "unnecessary_output", "poor_routing")))
        # nothing counted twice: per-segment claims come from a single category
        seen = set()
        for f in wa.token_findings():
            for e in f.evidence:
                self.assertNotIn(e, seen)
                seen.add(e)

    def test_estimate_drift_note(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        sim.ctx.add("system", "s" * 100)
        sim.step(provider_fn=lambda req, m, p: ProviderResponse(usage=Usage(input_fresh=150, output=5)))
        wa = analyze_waste(led, "S1")
        self.assertTrue(any("drift" in n for n in wa.notes))
        self.assertEqual(wa.unattributed_input_tokens, 50)


if __name__ == "__main__":
    unittest.main()
