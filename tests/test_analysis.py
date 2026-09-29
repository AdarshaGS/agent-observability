import json
import unittest

from agent_observability import (AggregateConfig, PriceTable, ModelPrice, RoutingConfig, agent_metrics, aggregate,
                                 analyze_task_routing, build_payload, build_session_report, cheaper_model_evidence,
                                 compute_baselines, context_growth, record_eval, render_text, select_sessions,
                                 session_summary, stratified_matrix)
from agent_observability.quality import judge_human_agreement
from helpers import Sim, make_gateway, words

_n = iter(range(1, 10_000))


def quick(gw, *, agent="a1", model="m-big", outcome="success", quality=None, method="test", iterations=2,
          tokens=1000, task_type="fix", complexity="medium", prompt_version="p1", tool_tokens=0, out_tokens=10):
    n = next(_n)
    sim = Sim(gw, session=f"S{n}", task=f"T{n}", agent=agent, model=model, task_type=task_type,
              complexity=complexity, prompt_version=prompt_version)
    sim.ctx.add("system", "x" * tokens)
    for _ in range(iterations):
        r = sim.step(out_tokens=out_tokens)
        if tool_tokens:
            sim.tool(r.call_id, "big_tool", {"i": n}, "z" * tool_tokens)
    sim.finish(outcome)
    if quality is not None:
        record_eval(gw.ledger, f"T{n}", method, quality)
    return f"S{n}", f"T{n}"


class MetricsTests(unittest.TestCase):
    def test_tokens_per_success_includes_failed_task_tokens(self):
        led, gw = make_gateway()
        for _ in range(4):
            quick(gw, tokens=1000, iterations=1)                    # 1010 tokens each
        quick(gw, tokens=1000, iterations=1, outcome="failure")     # 1010 wasted on a failure
        m = agent_metrics(led)["a1"]
        self.assertEqual(m["tasks"], 5)
        self.assertAlmostEqual(m["success_rate"], 0.8)
        self.assertAlmostEqual(m["tokens_per_successful_task"], 5 * 1010 / 4)
        self.assertAlmostEqual(m["avg_tokens_per_task"], 1010)
        self.assertAlmostEqual(m["failure_rate"], 0.2)

    def test_quality_methods_never_pooled_and_qa_efficiency(self):
        led, gw = make_gateway()
        for _ in range(3):
            quick(gw, quality=1.0, method="test")
        for _ in range(3):
            quick(gw, quality=0.2, method="human")
        q = agent_metrics(led)["a1"]["quality"]
        self.assertEqual(set(q["by_method"]), {"test", "human"})
        self.assertAlmostEqual(q["by_method"]["test"]["mean"], 1.0)
        self.assertAlmostEqual(q["by_method"]["human"]["mean"], 0.2)

    def test_judge_requires_model_and_agreement_metric(self):
        led, gw = make_gateway()
        _, t = quick(gw)
        with self.assertRaises(ValueError):
            record_eval(led, t, "judge", 0.9)
        record_eval(led, t, "judge", 0.9, judge_model="j1")
        record_eval(led, t, "human", 0.7)
        a = judge_human_agreement(led)
        self.assertAlmostEqual(a["bias"], 0.2)

    def test_cost_uses_price_table_and_flags_unpriced(self):
        pt = PriceTable("v1")
        pt.register("m-big", ModelPrice(input=10.0, output=30.0))
        led, gw = make_gateway(pricing=pt)
        s1, _ = quick(gw, model="m-big", tokens=1000, iterations=1, out_tokens=100)
        s2, _ = quick(gw, model="m-unknown", iterations=1)
        a, b = session_summary(led, s1), session_summary(led, s2)
        self.assertAlmostEqual(a["estimated_cost"], (1000 * 10 + 100 * 30) / 1e6)
        self.assertEqual((a["unpriced_calls"], b["unpriced_calls"]), (0, 1))

    def test_context_growth_attributes_delta_to_kind_and_origin(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        sim.ctx.add("system", "s" * 1000)
        for i in range(4):
            r = sim.step()
            sim.tool(r.call_id, "read", {"i": i}, words(f"t{i}", 100))
        g = context_growth(led, "S1")[0]
        self.assertGreater(g["slope_tokens_per_iteration"], 0)
        self.assertEqual(g["dominant_growth_kind"], "tool_result")
        last = g["iterations"][-1]
        self.assertEqual(last["new_by_kind"].get("tool_result"), len(words("t2", 100)))    # only the new block is 'new'
        self.assertEqual(last["by_kind"]["tool_result"], 3 * len(words("t0", 100)))       # t0..t2 carried


class RoutingTests(unittest.TestCase):
    def fill(self, gw, model, n, success, quality, task_type="fix", complexity="medium"):
        for i in range(n):
            ok = i < success
            quick(gw, model=model, outcome="success" if ok else "failure", quality=quality[i % len(quality)] if quality else None,
                  task_type=task_type, complexity=complexity, iterations=1)

    def test_downgrade_supported_only_with_non_inferiority_on_success_and_quality(self):
        led, gw = make_gateway()
        self.fill(gw, "m-big", 30, 30, [0.9])
        self.fill(gw, "m-small", 30, 30, [0.88, 0.92])
        ev = cheaper_model_evidence(led, "fix", "medium", "m-big", "m-small", RoutingConfig())
        self.assertEqual(ev["verdict"], "supported_downgrade")
        self.assertIn("Observational", ev["reason"])

    def test_cheaper_alone_is_never_enough(self):
        led, gw = make_gateway()
        self.fill(gw, "m-big", 30, 30, [0.9])
        self.fill(gw, "m-small", 30, 18, [0.9])                    # 60% success
        self.assertEqual(cheaper_model_evidence(led, "fix", "medium", "m-big", "m-small", RoutingConfig())["verdict"], "not_supported")

    def test_success_ok_but_no_quality_labels_is_insufficient(self):
        led, gw = make_gateway()
        self.fill(gw, "m-big", 30, 30, [0.9])
        self.fill(gw, "m-small", 30, 30, None)
        ev = cheaper_model_evidence(led, "fix", "medium", "m-big", "m-small", RoutingConfig())
        self.assertEqual(ev["verdict"], "insufficient_evidence")

    def test_quality_regression_blocks_downgrade(self):
        led, gw = make_gateway()
        self.fill(gw, "m-big", 30, 30, [0.9, 0.92, 0.88])
        self.fill(gw, "m-small", 30, 30, [0.6, 0.65, 0.7])
        self.assertEqual(cheaper_model_evidence(led, "fix", "medium", "m-big", "m-small", RoutingConfig())["verdict"], "not_supported")

    def test_small_samples_and_buckets_are_respected(self):
        led, gw = make_gateway()
        self.fill(gw, "m-big", 30, 30, [0.9])
        self.fill(gw, "m-small", 5, 5, [0.9])
        self.assertEqual(cheaper_model_evidence(led, "fix", "medium", "m-big", "m-small", RoutingConfig())["verdict"], "insufficient_evidence")
        self.fill(gw, "m-small", 30, 30, [0.9], complexity="simple")   # evidence from another bucket must not count
        self.assertEqual(cheaper_model_evidence(led, "fix", "medium", "m-big", "m-small", RoutingConfig())["verdict"], "insufficient_evidence")

    def test_task_routing_flags_escalation_and_failure_driven_spend(self):
        led, gw = make_gateway()
        sim = Sim(gw, model="m-small", complexity="complex")
        led.record_routing_decision("d1", task_id="T1", session_id="S1", chosen="m-small", candidates=["m-small", "m-big"],
                                    reason_code="default_cheap")
        sim.ctx.add("system", "s" * 500)
        sim.step(out_tokens=20, model="m-small")
        led.record_routing_decision("d2", task_id="T1", session_id="S1", chosen="m-big", candidates=["m-small", "m-big"],
                                    reason_code="escalate_after_failure", escalated_from="m-small")
        sim.step(out_tokens=20, model="m-big")
        sim.finish()
        r = analyze_task_routing(led, "T1", RoutingConfig(tiers={"m-small": 1, "m-big": 2}))
        self.assertEqual(r["appropriateness"], "under_provisioned_first_choice")
        self.assertEqual(r["failure_driven_spend_tokens"], 520)
        self.assertEqual(r["chosen"], "m-big")

    def test_stratified_matrix_groups_by_bucket_and_model(self):
        led, gw = make_gateway()
        self.fill(gw, "m-big", 5, 5, [0.9])
        self.fill(gw, "m-small", 5, 4, [0.8], complexity="simple")
        keys = {(r["complexity"], r["model"]) for r in stratified_matrix(led)}
        self.assertEqual(keys, {("medium", "m-big"), ("simple", "m-small")})


class ReportAggregateTests(unittest.TestCase):
    def test_session_report_end_to_end_and_render(self):
        led, gw = make_gateway()
        sim = Sim(gw)
        sim.ctx.add("system", "s" * 500)
        T = words("dup", 40)
        sim.ctx.add("tool_result", T)
        sim.ctx.add("tool_result", T)
        sim.step(out_text=T)
        sim.finish()
        record_eval(led, "T1", "test", 1.0)
        r = build_session_report(led, "S1")
        self.assertEqual(r["summary"]["successes"], 1)
        self.assertEqual(r["waste"]["findings"][0]["category"], "duplicated_context")
        self.assertEqual(r["quality_risk"], "low")
        self.assertGreater(r["estimated_savings"]["tokens_high"], 0)
        text = render_text(r)
        for h in ("Total Tokens", "Top Token Consumers", "Top Waste Sources", "Root Causes", "Estimated Token Savings", "Quality Risk"):
            self.assertIn(h, text)
        json.dumps(r, default=str)

    def test_payload_is_compact_json_and_selection_includes_failures(self):
        led, gw = make_gateway()
        s_ok, _ = quick(gw)
        s_bad, _ = quick(gw, outcome="failure")
        payload = json.loads(build_payload(build_session_report(led, s_ok), max_chars=6000))
        self.assertIn("findings", payload)
        picked = {p["session_id"]: p["reason"] for p in select_sessions(led, random_fraction=0.0)}
        self.assertEqual(picked.get(s_bad), "failure")

    def test_aggregate_flags_outlier_agent_noisy_tool_regression_and_hotspot(self):
        led, gw = make_gateway()
        for _ in range(12):
            quick(gw, agent="lean", tokens=1000, iterations=2, prompt_version="p1")
            quick(gw, agent="lean2", tokens=1000, iterations=2, prompt_version="p1")
            quick(gw, agent="fat", tokens=4000, iterations=2, prompt_version="p1")
        for _ in range(12):                                                    # p2 costs more than p1 for 'lean'
            quick(gw, agent="lean", tokens=2000, iterations=2, prompt_version="p2")
        for i in range(12):                                                    # iteration hotspot task type
            quick(gw, agent="h", task_type="research", iterations=2 if i < 9 else 10, tokens=200)
        for _ in range(12):
            quick(gw, agent="tooly", task_type="scrape", iterations=2, tokens=100, tool_tokens=9000)
        rep = aggregate(led, cfg=AggregateConfig(min_tasks=10))
        self.assertIn("fat", {o["agent_id"] for o in rep["outlier_agents"]})
        self.assertNotIn("lean2", {o["agent_id"] for o in rep["outlier_agents"]})
        self.assertIn("big_tool", {t["tool"] for t in rep["noisy_tools"]})
        self.assertTrue(any(r["agent_id"] == "lean" and r["to_version"] == "p2" for r in rep["prompt_regressions"]))
        self.assertIn("research", {h["task_type"] for h in rep["iteration_hotspots"]})
        self.assertTrue(rep["waste_patterns"])

    def test_baselines(self):
        led, gw = make_gateway()
        for i in range(12):
            quick(gw, iterations=2 + (i % 3))
        b = compute_baselines(led)["fix"]
        self.assertEqual(b["n"], 12)
        self.assertGreaterEqual(b["iterations_p90"], b["iterations_p50"])


if __name__ == "__main__":
    unittest.main()
