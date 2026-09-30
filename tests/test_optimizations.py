import unittest

from agent_observability.optimizations import add, auto_rate, rate, report
from agent_observability.quality import record_eval
from helpers import Sim, make_gateway


class AutoRateTest(unittest.TestCase):
    def test_proxy_scores_errors_and_never_overwrites_manual(self):
        led, gw = make_gateway()
        for sid, bad in (("s1", 1), ("s2", 0)):
            sim = Sim(gw, session=sid, task="T" + sid)
            r = sim.step(out_tokens=10)
            for i in range(6):
                led.insert("tool_calls", dict(tool_call_id=f"{sid}t{i}", call_id=led.one(
                    "SELECT call_id FROM model_calls WHERE session_id=? LIMIT 1", (sid,))["call_id"], session_id=sid,
                    tool_name="Bash", status="error" if i < bad else "ok"))
        self.assertEqual(auto_rate(led), 2)
        sc = {r["session_id"]: r["score"] for r in led.query("SELECT session_id, score FROM quality_evals")}
        self.assertAlmostEqual(sc["s1"], 1 - 1 / 6, places=3)
        self.assertEqual(sc["s2"], 1.0)
        self.assertEqual(auto_rate(led), 0)                       # idempotent
        rate(led, "s1", "failure", 0.1, "human")
        led.insert("tool_calls", dict(tool_call_id="x", call_id=led.one("SELECT call_id FROM model_calls WHERE session_id='s1'")["call_id"],
                                      session_id="s1", tool_name="Bash", status="ok"))
        auto_rate(led)
        self.assertEqual(led.one("SELECT score FROM quality_evals WHERE session_id='s1'")["score"], 0.1)


class OptimizationTest(unittest.TestCase):
    def _arm(self, gw, prefix, n, calls, outcome="success", score=1.0):
        ids = []
        for i in range(n):
            sid = f"{prefix}{i}"
            sim = Sim(gw, session=sid, task="T" + sid)
            sim.ctx.add("system", "x" * 400)
            for _ in range(calls + i % 2):       # slight variance
                sim.step(out_tokens=10)
            sim.finish(outcome)
            record_eval(gw.ledger, "T" + sid, "test", score)
            ids.append(sid)
        return ids

    def _verdict(self, after_calls, after_outcome="success", after_score=1.0, n=6):
        led, gw = make_gateway()
        base = self._arm(gw, "b", n, 10)
        after = self._arm(gw, "a", n, after_calls, after_outcome, after_score)
        add(led, "o", "opt", base, after)
        return report(led)["optimizations"][0]

    def test_improvement_only_when_all_metrics_hold(self):
        o = self._verdict(4)
        self.assertEqual(o["verdict"], "improvement")
        self.assertGreater(o["token_reduction"], 0.3)

    def test_quality_or_success_drop_blocks_improvement(self):
        self.assertEqual(self._verdict(4, after_score=0.5)["verdict"], "quality_regression")
        self.assertEqual(self._verdict(4, after_outcome="failure")["verdict"], "quality_regression")

    def test_worse_no_change_and_small_sample(self):
        self.assertEqual(self._verdict(15)["verdict"], "worse")
        self.assertEqual(self._verdict(10)["verdict"], "no_change")
        self.assertEqual(self._verdict(4, n=2)["verdict"], "inconclusive")

    def test_rated_session_without_task_events(self):
        led, gw = make_gateway()
        for arm, calls, score in (("b", 10, 0.9), ("a", 4, 0.9)):
            for i in range(6):
                sid = f"{arm}{i}"
                gw.start_session(sid)
                # raw calls with task_id=None, like collected transcripts
                for k in range(calls + i % 2):
                    from agent_observability import CallRequest, ContextBuilder
                    ctx = ContextBuilder(gw.estimator)
                    ctx.add("system", "x" * 400)
                    from helpers import provider
                    gw.call(CallRequest(session_id=sid, agent_id="a", model_id="m", segments=list(ctx.segments),
                                        iteration=k + 1), provider(out_tokens=10))
                rate(led, sid, "success", score)
        add(led, "o", "opt", [f"b{i}" for i in range(6)], [f"a{i}" for i in range(6)])
        o = report(led)["optimizations"][0]
        self.assertEqual(o["baseline"]["tasks"], 6)
        self.assertEqual(o["verdict"], "improvement")
        with self.assertRaises(ValueError):
            rate(led, "b0", "maybe")

    def test_validation(self):
        led, gw = make_gateway()
        ids = self._arm(gw, "b", 1, 2)
        with self.assertRaises(ValueError):
            add(led, "o", "n", ids, ids)
        with self.assertRaises(ValueError):
            add(led, "o", "n", ids, ["missing"])


if __name__ == "__main__":
    unittest.main()
