import unittest

from agent_observability.compare import compare, tag, tag_custom_agents
from helpers import Sim, make_gateway


class CompareTest(unittest.TestCase):
    def _session(self, gw, sid, calls):
        sim = Sim(gw, session=sid, task="T" + sid, agent="a", model="m", task_type="fix", complexity="medium", prompt_version="p")
        sim.ctx.add("system", "x" * 400)
        for _ in range(calls):
            sim.step(out_tokens=10)
        sim.finish("success")

    def test_per_unit_normalisation_and_validation(self):
        led, gw = make_gateway()
        self._session(gw, "A", 2)
        self._session(gw, "B", 6)
        tag(led, "ca", "with_agents", "agents", 2, ["A"])
        tag(led, "cb", "inline", "inline", 3, ["B"])
        out = compare(led)
        self.assertEqual(out["arms"]["with_agents"]["total"]["calls"], 2)
        self.assertAlmostEqual(out["arms"]["with_agents"]["per_unit"]["calls"], 1.0)
        self.assertAlmostEqual(out["arms"]["inline"]["per_unit"]["calls"], 2.0)
        self.assertAlmostEqual(out["delta_agents_vs_inline"]["calls"], -0.5)
        with self.assertRaises(ValueError):
            tag(led, "bad", "sideways", "x", 1, ["A"])
        with self.assertRaises(ValueError):
            tag(led, "bad", "inline", "x", 1, ["missing"])
        self.assertIn("ASSUMED", out["price_basis"])

    def test_custom_agents_only(self):
        led, gw = make_gateway()
        self._session(gw, "R1", 6)                       # inline
        self._session(gw, "R2", 6)                       # uses a built-in subagent: excluded from both arms
        self._session(gw, "R3", 6)                       # uses a custom agent
        for root, agent in (("R2", "Explore"), ("R3", "bug-fixer")):
            self._session_child(gw, root, agent)
        out = tag_custom_agents(led, {"bug-fixer"}, since=0)
        self.assertEqual((out["sessions_with_your_agents"], out["inline_sessions"]), (1, 1))

    def _session_child(self, gw, root, agent):
        sim = Sim(gw, session=root + "c", task="T" + root + "c", agent=agent, model="m", parent=root)
        sim.ctx.add("system", "x" * 400)
        sim.step(out_tokens=10)
        sim.finish("success")


if __name__ == "__main__":
    unittest.main()
