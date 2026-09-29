import json
import tempfile
import time
import unittest
from pathlib import Path

from agent_observability import Ledger
from agent_observability.collect_claude import ClaudeCodeCollector
from agent_observability.ingest import Ingestor


def line(**kw):
    return json.dumps(kw) + "\n"


def asst(mid, ts, out, blocks, sid="S1"):
    return line(type="assistant", sessionId=sid, cwd="/w/proj", entrypoint="cli", timestamp=ts, uuid=mid + ts,
                message=dict(id=mid, model="claude-x", stop_reason="tool_use", content=blocks,
                             usage=dict(input_tokens=3, cache_read_input_tokens=1000, cache_creation_input_tokens=200,
                                        output_tokens=out, output_tokens_details=dict(thinking_tokens=out // 2))))


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "p" / "S1" / "subagents").mkdir(parents=True)
        self.f = self.root / "p" / "S1.jsonl"
        self.f.write_text(
            asst("m1", "2026-09-29T10:00:00Z", 100, [{"type": "thinking"}])
            + asst("m1", "2026-09-29T10:00:01Z", 100, [{"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"c": "SECRET"}}])
            + line(type="user", sessionId="S1", timestamp="2026-09-29T10:00:02Z",
                   message=dict(role="user", content=[{"type": "tool_result", "tool_use_id": "tu1", "content": "SECRET output " * 100}]))
            + asst("m2", "2026-09-29T10:01:00Z", 50, [{"type": "text", "text": "SECRET"}]))
        (self.root / "p" / "S1" / "subagents" / "agent-abc.jsonl").write_text(asst("s1", "2026-09-29T10:00:30Z", 10, [], sid="S1"))
        self.led = Ledger()
        self.col = ClaudeCodeCollector(Ingestor(self.led), self.root, lookback_days=10_000)

    def tearDown(self):
        self.tmp.cleanup()

    def test_collect_dedupes_lines_and_links_subagents(self):
        st = self.col.run_once()
        self.assertEqual((st["sessions"], st["calls"], st["tool_calls"], st["errors"]), (2, 3, 1, 0))
        c = self.led.one("SELECT * FROM model_calls WHERE call_id='m1'")
        self.assertEqual((c["input_fresh"], c["cache_read"], c["cache_write"], c["output_tokens"], c["reasoning_tokens"]),
                         (3, 1000, 200, 100, 50))
        sub = self.led.one("SELECT * FROM sessions WHERE session_id='agent-abc'")
        self.assertEqual(sub["parent_session_id"], "S1")
        self.assertEqual(self.led.root_of("agent-abc"), "S1")
        self.assertEqual(self.led.one("SELECT root_task FROM sessions WHERE session_id='S1'")["root_task"], "proj · cli")
        self.assertGreater(self.led.one("SELECT raw_output_tokens r FROM tool_calls")["r"], 100)

    def test_real_timestamps_and_no_content_stored(self):
        self.col.run_once()
        self.assertEqual(self.led.one("SELECT started_at FROM sessions WHERE session_id='S1'")["started_at"], 1790676000.0)
        dump = json.dumps([self.led.query(f"SELECT * FROM {t}") for t in ("sessions", "model_calls", "tool_calls")])
        self.assertNotIn("SECRET", dump)

    def test_incremental_and_idempotent(self):
        self.col.run_once()
        self.assertEqual(self.col.run_once()["skipped"], 2)
        with open(self.f, "a") as fh:
            fh.write(asst("m3", "2026-09-29T10:02:00Z", 7, [{"type": "text", "text": "x"}]))
        st = self.col.run_once()
        self.assertEqual((st["files"], st["calls"]), (1, 1))
        self.assertEqual(self.led.one("SELECT COUNT(*) n FROM model_calls")["n"], 4)
        self.assertEqual(self.led.one("SELECT used FROM budget_state WHERE root_id='S1'")["used"] > 0, True)

    def test_lookback_excludes_old_files(self):
        old = time.time() - 5 * 86400
        import os
        os.utime(self.f, (old, old))
        col = ClaudeCodeCollector(Ingestor(self.led), self.root, lookback_days=1)
        self.assertEqual(col.run_once()["sessions"], 1)          # only the untouched-today subagent file


if __name__ == "__main__":
    unittest.main()
