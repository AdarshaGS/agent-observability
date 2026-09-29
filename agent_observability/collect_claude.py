"""Pull collector: turns Claude Code session transcripts into ledger events. No agent changes needed.

Reads ~/.claude/projects/<project>/<session>.jsonl (+ <session>/subagents/agent-*.jsonl) on a timer.
Only metadata is kept: per-API-call token usage, model, timestamps, tool names, result *sizes* and hashes.
Prompt / response / tool text is never stored. Idempotent: an unchanged file is skipped, and re-reading a changed
one adds only new calls (call_id = the API message id). One API call is written as several transcript lines
(one per content block) with the same message id, so lines are merged by id.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .budget import BudgetConfig
from .ingest import Ingestor
from .tokens import HeuristicEstimator, content_hash

DEFAULT_ROOT = Path.home() / ".claude" / "projects"
IDLE_SECONDS = 1800          # a transcript untouched this long is treated as a finished session


def _ts(v) -> Optional[float]:
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _text_len(content) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(b.get("text", "")) if isinstance(b, dict) else len(str(b)) for b in content)
    return len(str(content or ""))


def parse_transcript(path: Path, estimator) -> dict:
    """-> {"session_id","cwd","entrypoint","first_ts","last_ts","messages":[...]} from one transcript file."""
    msgs: dict[str, dict] = {}
    order: list[str] = []
    results: dict[str, dict] = {}
    meta = dict(session_id=None, cwd=None, entrypoint=None, first_ts=None, last_ts=None)
    with open(path, "rb") as fh:
        for raw in fh:
            try:
                d = json.loads(raw)
            except ValueError:
                continue                                   # partially written trailing line
            ts = _ts(d.get("timestamp"))
            if ts:
                meta["first_ts"] = meta["first_ts"] or ts
                meta["last_ts"] = max(meta["last_ts"] or 0, ts)
            for k, src in (("session_id", "sessionId"), ("cwd", "cwd"), ("entrypoint", "entrypoint")):
                meta[k] = meta[k] or d.get(src)
            m = d.get("message")
            if d.get("type") == "assistant" and isinstance(m, dict) and isinstance(m.get("usage"), dict):
                mid = m.get("id") or d.get("uuid")
                cur = msgs.get(mid)
                if cur is None:
                    cur = msgs[mid] = dict(id=mid, model=m.get("model"), ts=ts, stop=m.get("stop_reason"),
                                           usage={}, tools=[])
                    order.append(mid)
                u = m["usage"]
                th = (u.get("output_tokens_details") or {}).get("thinking_tokens", 0) or 0
                for key, val in (("input_fresh", u.get("input_tokens", 0)), ("cache_read", u.get("cache_read_input_tokens", 0)),
                                 ("cache_write", u.get("cache_creation_input_tokens", 0)), ("output", u.get("output_tokens", 0)),
                                 ("reasoning", th)):
                    cur["usage"][key] = max(cur["usage"].get(key, 0), int(val or 0))
                cur["stop"] = m.get("stop_reason") or cur["stop"]
                for b in m.get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id") not in {t["id"] for t in cur["tools"]}:
                        cur["tools"].append(dict(id=b["id"], name=b.get("name", "?"),
                                                 args_hash=content_hash(json.dumps(b.get("input"), sort_keys=True, default=str))))
            elif d.get("type") == "user" and isinstance(m, dict) and isinstance(m.get("content"), list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        results[b.get("tool_use_id")] = dict(tokens=estimator.count("x" * _text_len(b.get("content"))),
                                                             error=bool(b.get("is_error")),
                                                             hash=content_hash(json.dumps(b.get("content"), default=str)))
    meta["messages"] = [msgs[i] for i in order]
    meta["results"] = results
    return meta


class ClaudeCodeCollector:
    def __init__(self, ingestor: Ingestor, root: Path = DEFAULT_ROOT, lookback_days: float = 1.0,
                 budget: Optional[BudgetConfig] = None, clock=time.time):
        self.ing, self.root, self.lookback = ingestor, Path(root).expanduser(), lookback_days
        self.budget, self.clock = budget, clock
        self.est = HeuristicEstimator()
        self._stop = threading.Event()
        self.last_stats: dict = {}

    def _files(self) -> list[Path]:
        cutoff = self.clock() - self.lookback * 86400
        found = []
        for p in self.root.glob("*/*.jsonl"):
            found.append((0, p))
        for p in self.root.glob("*/*/subagents/agent-*.jsonl"):
            found.append((1, p))                          # parents first, so subagents can attach to them
        out = [(d, p) for d, p in found if p.stat().st_mtime >= cutoff]
        return [p for _, p in sorted(out, key=lambda x: (x[0], str(x[1])))]

    def run_once(self) -> dict:
        stats = dict(files=0, skipped=0, sessions=0, calls=0, tool_calls=0, errors=0)
        for path in self._files():
            try:
                st = path.stat()
                row = self.ing.ledger.one("SELECT size, mtime FROM collector_files WHERE path=?", (str(path),))
                if row and row["size"] == st.st_size and row["mtime"] == st.st_mtime:
                    stats["skipped"] += 1
                    continue
                self._sync(path, st, stats)
                stats["files"] += 1
            except Exception:
                stats["errors"] += 1
        if stats["files"]:
            self.ing.sweep_reservations()
        self.last_stats = dict(stats, at=self.clock())
        return stats

    def _sync(self, path: Path, st, stats: dict) -> None:
        t = parse_transcript(path, self.est)
        is_sub = path.parent.name == "subagents"
        sid = path.stem if is_sub else (t["session_id"] or path.stem)
        parent = path.parent.parent.name if is_sub else None
        if parent and not self.ing._session_exists(parent):
            parent = None
        project = os.path.basename(t["cwd"] or "") or path.parent.name
        events = [dict(type="session_start", session_id=sid, parent_session_id=parent, ts=t["first_ts"] or st.st_mtime,
                       root_task=f"{project} · {t['entrypoint'] or 'claude-code'}" + (" · subagent" if is_sub else ""),
                       **({"budget": dict(limit=self.budget.limit, finalization_reserve=self.budget.finalization_reserve,
                                          basis=self.budget.basis)} if self.budget and not parent else {}))]
        known = {r["call_id"] for r in self.ing.ledger.query("SELECT call_id FROM model_calls WHERE session_id=?", (sid,))}
        known_tools = {r["tool_call_id"] for r in self.ing.ledger.query("SELECT tool_call_id FROM tool_calls WHERE session_id=?", (sid,))}
        for i, m in enumerate(t["messages"], 1):
            u = m["usage"]
            if not any(u.values()) or m["model"] in (None, "<synthetic>"):
                continue
            u["reasoning"] = min(u.get("reasoning", 0), u.get("output", 0))
            if m["id"] not in known:
                events.append(dict(type="model_call", call_id=m["id"], session_id=sid, agent_id="subagent" if is_sub else "claude-code",
                                   model_id=m["model"], iteration=i, ts=m["ts"], usage=u, stop_reason=m["stop"],
                                   provider="anthropic", tool_call_count=len(m["tools"])))
            for tool in m["tools"]:
                if tool["id"] in known_tools:
                    continue
                r = t["results"].get(tool["id"])
                events.append(dict(type="tool_call", tool_call_id=tool["id"], call_id=m["id"], tool_name=tool["name"],
                                   args_hash=tool["args_hash"], status="error" if r and r["error"] else "ok",
                                   raw_output_tokens=r["tokens"] if r else 0, injected_tokens=r["tokens"] if r else 0,
                                   result_hash=r["hash"] if r else None))
        if t["last_ts"] and self.clock() - st.st_mtime > IDLE_SECONDS:
            events.append(dict(type="session_end", session_id=sid, status="completed", ts=t["last_ts"]))
        for e, r in zip(events, self.ing.handle_batch(events)):
            if r["status"] == "error":
                stats["errors"] += 1
            elif r["status"] == "ok":
                stats["calls"] += e["type"] == "model_call"
                stats["tool_calls"] += e["type"] == "tool_call"
                stats["sessions"] += e["type"] == "session_start"
        self.ing.ledger.execute(
            "INSERT OR REPLACE INTO collector_files(path,size,mtime,session_id,synced_at) VALUES(?,?,?,?,?)",
            (str(path), st.st_size, st.st_mtime, sid, self.clock()))

    # -- background loop -------------------------------------------------------------
    def start(self, interval: float = 30.0, on_pass=None) -> threading.Thread:
        def loop():
            while not self._stop.is_set():
                try:
                    stats = self.run_once()
                    if on_pass and stats["files"]:
                        on_pass()
                except Exception:
                    pass
                self._stop.wait(interval)
        th = threading.Thread(target=loop, name="claude-code-collector", daemon=True)
        th.start()
        return th

    def stop(self) -> None:
        self._stop.set()
