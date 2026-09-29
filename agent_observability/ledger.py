"""SQLite-backed append-only ledger + trajectory reconstruction."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterable, Optional

from .schema import SCHEMA, SCHEMA_VERSION


def _marks(n: int) -> str:
    return ",".join("?" * n)


class Ledger:
    def __init__(self, path: str = ":memory:"):
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._in_tx = False
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))

    # ------------------------------------------------------------------ plumbing
    @contextmanager
    def transaction(self):
        """Serialised write transaction (BEGIN IMMEDIATE => safe across processes too)."""
        with self._lock:
            if self._in_tx:
                yield
                return
            self._conn.execute("BEGIN IMMEDIATE")
            self._in_tx = True
            try:
                yield
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            finally:
                self._in_tx = False

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, tuple(params))

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, tuple(params)).fetchall()]

    def one(self, sql: str, params: Iterable[Any] = ()) -> Optional[dict]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def insert(self, table: str, row: dict, or_ignore: bool = False) -> int:
        cols = list(row)
        verb = "INSERT OR IGNORE" if or_ignore else "INSERT"
        cur = self.execute(f"{verb} INTO {table}({','.join(cols)}) VALUES({_marks(len(cols))})",
                           [row[c] for c in cols])
        return cur.lastrowid

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ sessions / tasks
    def create_session(self, session_id: str, *, parent_session_id: Optional[str] = None,
                       tenant: Optional[str] = None, root_task: Optional[str] = None,
                       now: Optional[float] = None) -> str:
        """Returns the budget root id (children share their root's budget)."""
        root = session_id
        if parent_session_id:
            p = self.one("SELECT budget_root_id FROM sessions WHERE session_id=?", (parent_session_id,))
            if p is None:
                raise KeyError(f"unknown parent session {parent_session_id}")
            root = p["budget_root_id"]
        self.insert("sessions", dict(session_id=session_id, parent_session_id=parent_session_id,
                                     budget_root_id=root, tenant=tenant, root_task=root_task,
                                     started_at=now if now is not None else time.time()))
        return root

    def end_session(self, session_id: str, status: str, now: Optional[float] = None) -> None:
        self.execute("UPDATE sessions SET ended_at=?, terminal_status=COALESCE(terminal_status, ?) WHERE session_id=?",
                     (now if now is not None else time.time(), status, session_id))

    def set_terminal_status(self, session_id: str, status: str) -> None:
        self.execute("UPDATE sessions SET terminal_status=? WHERE session_id=?", (status, session_id))

    def save_checkpoint(self, session_id: str, checkpoint: dict) -> None:
        self.execute("UPDATE sessions SET checkpoint=? WHERE session_id=?", (json.dumps(checkpoint), session_id))

    def root_of(self, session_id: str) -> str:
        r = self.one("SELECT budget_root_id FROM sessions WHERE session_id=?", (session_id,))
        if r is None:
            raise KeyError(f"unknown session {session_id}")
        return r["budget_root_id"]

    def tree_ids(self, session_id: str) -> list[str]:
        """The session and all its descendants (subagents)."""
        rows = self.query(
            """WITH RECURSIVE t(id) AS (
                   SELECT session_id FROM sessions WHERE session_id=?
                   UNION ALL
                   SELECT s.session_id FROM sessions s JOIN t ON s.parent_session_id=t.id)
               SELECT id FROM t""", (session_id,))
        return [r["id"] for r in rows]

    def create_task(self, task_id: str, session_id: str, task_type: Optional[str] = None,
                    complexity_est: Optional[str] = None, acceptance_ref: Optional[str] = None,
                    now: Optional[float] = None, meta: Optional[dict] = None) -> None:
        self.insert("tasks", dict(task_id=task_id, session_id=session_id, task_type=task_type,
                                  complexity_est=complexity_est, acceptance_ref=acceptance_ref,
                                  started_at=now if now is not None else time.time(),
                                  meta=json.dumps(meta) if meta else None))

    def finish_task(self, task_id: str, outcome: str, now: Optional[float] = None) -> None:
        self.execute("UPDATE tasks SET outcome=?, ended_at=? WHERE task_id=?",
                     (outcome, now if now is not None else time.time(), task_id))

    def record_routing_decision(self, decision_id: str, *, task_id: Optional[str], session_id: str,
                                chosen: str, candidates: Optional[list[str]] = None,
                                reason_code: Optional[str] = None,
                                complexity_features: Optional[dict] = None,
                                router_version: Optional[str] = None,
                                escalated_from: Optional[str] = None,
                                now: Optional[float] = None) -> str:
        self.insert("routing_decisions", dict(
            decision_id=decision_id, task_id=task_id, session_id=session_id,
            candidates=json.dumps(candidates or []), chosen=chosen, reason_code=reason_code,
            complexity_features=json.dumps(complexity_features or {}), router_version=router_version,
            escalated_from=escalated_from, ts=now if now is not None else time.time()))
        return decision_id

    # ------------------------------------------------------------------ reads used by analyzers
    def calls_for(self, session_ids: list[str]) -> list[dict]:
        rows = self.query(f"SELECT * FROM model_calls WHERE session_id IN ({_marks(len(session_ids))}) ORDER BY seq",
                          session_ids)
        for r in rows:
            r["output_fingerprint"] = json.loads(r["output_fingerprint"]) if r["output_fingerprint"] else []
        return rows

    def segments_for(self, session_ids: list[str]) -> dict[str, list[dict]]:
        rows = self.query(
            f"SELECT * FROM context_segments WHERE session_id IN ({_marks(len(session_ids))}) ORDER BY call_id, position",
            session_ids)
        out: dict[str, list[dict]] = {}
        for r in rows:
            r["fingerprint"] = json.loads(r["fingerprint"]) if r["fingerprint"] else []
            out.setdefault(r["call_id"], []).append(r)
        return out

    def tool_calls_for(self, session_ids: list[str]) -> list[dict]:
        return self.query(
            f"""SELECT t.*, c.seq AS call_seq FROM tool_calls t JOIN model_calls c ON c.call_id=t.call_id
                WHERE t.session_id IN ({_marks(len(session_ids))}) ORDER BY c.seq, t.rowid""", session_ids)

    def tasks_for(self, session_ids: list[str]) -> list[dict]:
        return self.query(f"SELECT * FROM tasks WHERE session_id IN ({_marks(len(session_ids))}) ORDER BY started_at, task_id",
                          session_ids)

    # ------------------------------------------------------------------ trajectory
    def trajectory(self, session_id: str, include_children: bool = True) -> dict:
        """Complete, ordered reconstruction of a session: every call, its context segments,
        its tool calls, cumulative token counts, budget events, routing and quality."""
        ids = self.tree_ids(session_id) if include_children else [session_id]
        calls = self.calls_for(ids)
        segs = self.segments_for(ids)
        tools_by_call: dict[str, list[dict]] = {}
        for t in self.tool_calls_for(ids):
            tools_by_call.setdefault(t["call_id"], []).append(t)
        cum = 0
        steps = []
        for c in calls:
            billed = c["input_fresh"] + c["cache_read"] + c["cache_write"] + c["output_tokens"]
            cum += billed
            c = dict(c)
            c.pop("output_fingerprint", None)
            steps.append(dict(call=c, billed=billed, cumulative_billed=cum,
                              segments=[{k: v for k, v in s.items() if k != "fingerprint"}
                                        for s in segs.get(c["call_id"], [])],
                              tool_calls=tools_by_call.get(c["call_id"], [])))
        root = self.root_of(session_id)
        return dict(
            session=self.one("SELECT * FROM sessions WHERE session_id=?", (session_id,)),
            sessions=ids, steps=steps, total_billed=cum,
            budget_events=self.query("SELECT * FROM budget_events WHERE root_id=? ORDER BY id", (root,)),
            tasks=self.tasks_for(ids),
            routing=self.query(f"SELECT * FROM routing_decisions WHERE session_id IN ({_marks(len(ids))}) ORDER BY ts", ids),
            quality=self.query(f"SELECT * FROM quality_evals WHERE session_id IN ({_marks(len(ids))}) ORDER BY eval_id", ids),
        )
