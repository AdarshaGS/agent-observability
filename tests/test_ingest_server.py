import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from agent_observability import Ledger
from agent_observability.auth import Auth, check_bind_security
from agent_observability.ingest import Ingestor
from agent_observability.server import Api, make_handler

TOKEN = "t" * 24
RTOKEN = "r" * 24
ITOKEN = "i" * 24


def call_ev(cid, sid="s1", **kw):
    ev = dict(type="model_call", call_id=cid, session_id=sid, model_id="m", agent_id="a", task_id="t1",
              usage={"input_fresh": 1000, "output": 200}, segments=[{"kind": "system", "text": "you are helpful"},
                                                                     {"kind": "user", "tokens": 50}])
    ev.update(kw)
    return ev


class IngestorTests(unittest.TestCase):
    def setUp(self):
        self.led = Ledger()
        self.ing = Ingestor(self.led)

    def test_full_flow_and_idempotency(self):
        r = self.ing.handle_batch([
            dict(type="session_start", session_id="s1", budget={"limit": 10_000, "finalization_reserve": 1000}),
            dict(type="task_start", task_id="t1", session_id="s1", task_type="x"),
            call_ev("c1"), call_ev("c1"),
            dict(type="tool_call", call_id="c1", tool_name="grep", args={"q": 1}, raw_output_tokens=900, injected_tokens=300),
            dict(type="quality", task_id="t1", method="test", score=1),
            dict(type="task_end", task_id="t1", outcome="success"),
            dict(type="session_end", session_id="s1")])
        self.assertEqual([x["status"] for x in r],
                         ["ok", "ok", "ok", "duplicate", "ok", "ok", "ok", "ok"])
        st = self.ing.status("s1")
        self.assertEqual(st["used"], 1200)
        self.assertEqual(st["limit"], 10_000)
        self.assertEqual(self.led.one("SELECT COUNT(*) n FROM model_calls")["n"], 1)
        self.assertEqual(self.led.one("SELECT COUNT(*) n FROM context_segments")["n"], 2)

    def test_validation_errors_do_not_leave_partial_rows(self):
        r = self.ing.handle(call_ev("bad", usage={"input_fresh": -5, "output": 1}))
        self.assertEqual(r["status"], "error")
        r = self.ing.handle(call_ev("bad2", segments=[{"kind": "nope", "tokens": 1}]))
        self.assertEqual(r["status"], "error")
        r = self.ing.handle(call_ev("bad3", usage={"output": 5, "reasoning": 9}))
        self.assertEqual(r["status"], "error")
        self.assertEqual(self.led.one("SELECT COUNT(*) n FROM sessions")["n"], 0)
        self.assertEqual(self.ing.handle({"type": "wat"})["status"], "error")
        self.assertEqual(self.ing.handle("x")["status"], "error")

    def test_text_is_hashed_not_stored(self):
        self.ing.handle(call_ev("c1", segments=[{"kind": "tool_result", "text": "SECRET-PAYLOAD alphaBeta_gamma"}]))
        dump = json.dumps(self.led.query("SELECT * FROM context_segments"))
        self.assertNotIn("SECRET-PAYLOAD", dump)

    def test_provider_usage_formats(self):
        self.ing.handle(call_ev("a", usage_format="anthropic",
                                usage={"input_tokens": 100, "cache_read_input_tokens": 400, "output_tokens": 10}))
        self.ing.handle(call_ev("b", usage_format="openai", usage={
            "prompt_tokens": 500, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 400}}))
        rows = {r["call_id"]: r for r in self.led.query("SELECT * FROM model_calls")}
        self.assertEqual((rows["a"]["input_fresh"], rows["a"]["cache_read"]), (100, 400))
        self.assertEqual((rows["b"]["input_fresh"], rows["b"]["cache_read"]), (100, 400))

    def test_reserve_enforces_cap_and_commit_releases(self):
        self.ing.handle(dict(type="session_start", session_id="s1", budget={"limit": 10_000, "finalization_reserve": 1000}))
        code, ok = self.ing.reserve(dict(session_id="s1", est_input_tokens=2000, max_output_tokens=2000))
        self.assertEqual(code, 200)
        self.assertGreater(self.ing.status("s1")["reserved"], 0)
        self.ing.handle(call_ev("c1", reservation_id=ok["reservation_id"]))
        self.assertEqual(self.ing.status("s1")["reserved"], 0)
        self.assertEqual(self.ing.status("s1")["used"], 1200)
        code, body = self.ing.reserve(dict(session_id="s1", est_input_tokens=20_000, max_output_tokens=100))
        self.assertEqual((code, body["error"]), (429, "budget_exhausted"))

    def test_expired_reservation_is_swept(self):
        t = [1000.0]
        ing = Ingestor(Ledger(), clock=lambda: t[0])
        ing.handle(dict(type="session_start", session_id="s1"))
        code, _ = ing.reserve(dict(session_id="s1", est_input_tokens=1000, max_output_tokens=1000, ttl_seconds=10))
        self.assertEqual(code, 200)
        self.assertGreater(ing.status("s1")["reserved"], 0)
        t[0] += 11
        self.assertEqual(ing.sweep_reservations(), 1)
        self.assertEqual(ing.status("s1")["reserved"], 0)

    def test_release(self):
        self.ing.handle(dict(type="session_start", session_id="s1"))
        _, r = self.ing.reserve(dict(session_id="s1", est_input_tokens=1000, max_output_tokens=1000))
        self.assertEqual(self.ing.release(dict(reservation_id=r["reservation_id"]))[0], 200)
        self.assertEqual(self.ing.status("s1")["reserved"], 0)
        self.assertEqual(self.ing.release(dict(reservation_id=r["reservation_id"]))[0], 404)


class AuthTests(unittest.TestCase):
    def test_scopes(self):
        a = Auth(read=RTOKEN, ingest=ITOKEN)
        self.assertEqual(a.check(None, {"read"}), 401)
        self.assertEqual(a.check("Bearer nope", {"read"}), 401)
        self.assertEqual(a.check("Bearer " + RTOKEN, {"read"}), 200)
        self.assertEqual(a.check("Bearer " + RTOKEN, {"ingest"}), 403)
        self.assertEqual(a.check("bearer " + ITOKEN, {"ingest"}), 200)
        self.assertEqual(Auth().check(None, {"read"}), 200)

    def test_short_token_and_bind_rules(self):
        with self.assertRaises(ValueError):
            Auth(both="short")
        with self.assertRaises(ValueError):
            check_bind_security("0.0.0.0", Auth())
        check_bind_security("0.0.0.0", Auth(both=TOKEN))
        check_bind_security("127.0.0.1", Auth())


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = Api(Ledger())
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.api, Auth(read=RTOKEN, ingest=ITOKEN)))
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def req(self, method, path, body=None, token=None):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(self.base + path, data=data, method=method,
                                   headers={"Authorization": f"Bearer {token}"} if token else {})
        try:
            with urllib.request.urlopen(r) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            finally:
                e.close()

    def test_end_to_end(self):
        self.assertEqual(self.req("GET", "/healthz")[0], 200)
        self.assertEqual(self.req("POST", "/ingest", call_ev("h1"))[0], 401)
        self.assertEqual(self.req("POST", "/ingest", call_ev("h1"), RTOKEN)[0], 403)
        self.assertEqual(self.req("GET", "/api/sessions")[0], 401)
        self.assertEqual(self.req("GET", "/api/sessions", token=ITOKEN)[0], 403)
        code, out = self.req("POST", "/ingest", {"events": [call_ev("h1", sid="hs"), call_ev("h1", sid="hs"), {"type": "x"}]}, ITOKEN)
        self.assertEqual((code, out["accepted"], out["duplicates"], out["errors"]), (207, 1, 1, 1))
        code, sess = self.req("GET", "/api/sessions", token=RTOKEN)
        self.assertEqual(code, 200)
        self.assertIn("hs", [s["session_id"] for s in sess])
        code, b = self.req("GET", "/api/budget/hs", token=ITOKEN)
        self.assertEqual((code, b["used"]), (200, 1200))
        code, r = self.req("POST", "/budget/reserve", dict(session_id="hs", est_input_tokens=999_999, max_output_tokens=10), ITOKEN)
        self.assertEqual(code, 429)
        self.assertEqual(self.req("POST", "/ingest", None, ITOKEN)[0], 400)


class WindowTests(unittest.TestCase):
    def test_since_until_filter_sessions_and_overview(self):
        t = [1000.0]
        led = Ledger()
        ing = Ingestor(led, clock=lambda: t[0])
        for i, ts in enumerate((1000.0, 2000.0, 3000.0)):
            t[0] = ts
            ing.handle(call_ev(f"c{i}", sid=f"s{i}"))
        api = Api(led)
        ids = lambda q: sorted(s["session_id"] for s in api.route("/api/sessions", q))
        self.assertEqual(ids({}), ["s0", "s1", "s2"])
        self.assertEqual(ids({"since": ["1500"], "until": ["3000"]}), ["s1"])     # [since, until)
        self.assertEqual(ids({"since": ["2000"]}), ["s1", "s2"])
        self.assertEqual(api.route("/api/overview", {"since": ["1500"], "until": ["3000"]})["window"]["sessions"], 1)
        with self.assertRaises(ValueError):
            api.route("/api/sessions", {"since": ["5"], "until": ["5"]})
        with self.assertRaises(ValueError):
            api.route("/api/sessions", {"since": ["abc"]})


if __name__ == "__main__":
    unittest.main()
