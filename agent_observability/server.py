"""Local read-mostly web UI + JSON API over a ledger.

    python3 -m agent_observability serve --db ledger.db [--port 8765] [--open]

Binds to 127.0.0.1 by default. Non-loopback binds require tokens (see auth.py).

  GET  /                       dashboard shell (no data; the page asks for a token on 401)
  GET  /healthz                liveness
  GET  /api/...                dashboard data                                   [read]
  GET  /api/budget/<session>   budget status for a session tree                 [ingest or read]
  POST /ingest                 one event, a list, or {"events":[...]}           [ingest]
  POST /budget/reserve         reserve tokens before a call (429 if it won't fit) [ingest]
  POST /budget/release         hand back an unused reservation                  [ingest]
"""
from __future__ import annotations

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .aggregate import aggregate
from .auth import Auth, check_bind_security
from .ingest import IngestError, Ingestor, MAX_BATCH
from .ledger import Ledger
from .metrics import context_growth, session_summary
from .report import build_session_report, render_text

UI_FILE = Path(__file__).parent / "ui" / "index.html"
MAX_BODY = 8 * 1024 * 1024
_CACHE_TTL = 10.0


class Api:
    def __init__(self, ledger: Ledger, db_path: str = None):
        self.ledger = ledger
        self.db_path = db_path
        self.ingestor = Ingestor(ledger)
        self._cache: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def window(query: dict) -> tuple:
        """?since=&until= as epoch seconds; the window is [since, until) on session start time."""
        out = []
        for k in ("since", "until"):
            v = (query.get(k) or [None])[0]
            try:
                out.append(float(v) if v not in (None, "") else None)
            except ValueError:
                raise ValueError(f"'{k}' must be epoch seconds")
        if None not in out and out[0] >= out[1]:
            raise ValueError("'since' must be before 'until'")
        return tuple(out)

    def _cached(self, key: str, fn):
        with self._lock:
            hit = self._cache.get(key)
            if hit and time.time() - hit[0] < _CACHE_TTL:
                return hit[1]
        val = fn()
        with self._lock:
            self._cache[key] = (time.time(), val)
        return val

    def sessions(self, since=None, until=None) -> list[dict]:
        where, params = ["parent_session_id IS NULL"], []
        if since is not None:
            where.append("started_at >= ?")
            params.append(since)
        if until is not None:
            where.append("started_at < ?")
            params.append(until)
        rows = self.ledger.query(
            "SELECT session_id, started_at, ended_at, terminal_status, tenant, root_task FROM sessions "
            f"WHERE {' AND '.join(where)} ORDER BY started_at DESC LIMIT 500", params)
        out = []
        for r in rows:
            s = session_summary(self.ledger, r["session_id"])
            b = self.ledger.one("SELECT budget_limit, used FROM budget_state WHERE root_id=?", (r["session_id"],)) or {}
            q = next(iter(s["quality"]["by_method"].values()), None)
            out.append(dict(r, tokens=s["total_billed_tokens"], cost=s["estimated_cost"], unpriced=s["unpriced_calls"],
                            models=list(s["models"]), iterations=s["iterations"], tool_calls=s["tool_calls"],
                            tasks=s["tasks"], successes=s["successes"], quality=q["mean"] if q else None,
                            budget_limit=b.get("budget_limit"), budget_used=b.get("used"),
                            budget_fraction=(b["used"] / b["budget_limit"]) if b.get("budget_limit") else None))
        return out

    def session(self, session_id: str) -> dict:
        if not self.ledger.one("SELECT 1 FROM sessions WHERE session_id=?", (session_id,)):
            raise KeyError(session_id)
        report = build_session_report(self.ledger, session_id)
        traj = self.ledger.trajectory(session_id)
        b = self.ledger.one("SELECT * FROM budget_state WHERE root_id=?", (self.ledger.root_of(session_id),))
        return dict(report=report, text=render_text(report), budget=b,
                    growth=context_growth(self.ledger, session_id),
                    steps=[dict(billed=s["billed"], cumulative=s["cumulative_billed"], call=s["call"],
                                segments=s["segments"], tool_calls=s["tool_calls"]) for s in traj["steps"]],
                    tasks=traj["tasks"], quality=traj["quality"])

    def overview(self, since=None, until=None) -> dict:
        return self._cached(f"overview:{since}:{until}", lambda: aggregate(self.ledger, since=since, until=until))

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()

    def route(self, path: str, query: dict = None):
        query = query or {}
        parts = [unquote(p) for p in path.strip("/").split("/")]
        if len(parts) == 3 and parts[:2] == ["api", "budget"]:
            self.ingestor.sweep_reservations()
            return self.ingestor.status(parts[2])
        if parts == ["api", "sessions"]:
            since, until = self.window(query)
            return self._cached(f"sessions:{since}:{until}", lambda: self.sessions(since, until))
        if parts == ["api", "compare"]:
            from .compare import compare
            return self._cached("compare", lambda: compare(self.ledger, self.db_path))
        if parts == ["api", "overview"]:
            return self.overview(*self.window(query))
        if len(parts) == 3 and parts[:2] == ["api", "session"]:
            return self.session(parts[2])
        raise LookupError(path)


def make_handler(api: Api, auth: Auth = None):
    auth = auth or Auth()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra=None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj, extra=None) -> None:
            self._send(code, json.dumps(obj, default=str).encode(), "application/json", extra)

        def _authorized(self, need: set) -> bool:
            code = auth.check(self.headers.get("Authorization"), need)
            if code == 200:
                return True
            self._json(code, {"error": "unauthorized" if code == 401 else "forbidden: token lacks scope"},
                       {"WWW-Authenticate": 'Bearer realm="agent-observability"'} if code == 401 else None)
            return False

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            if path in ("/", "/index.html"):
                return self._send(200, UI_FILE.read_bytes(), "text/html; charset=utf-8")
            if path == "/healthz":
                return self._json(200, {"status": "ok"})
            if path.startswith("/api/"):
                need = {"read", "ingest"} if path.startswith("/api/budget/") else {"read"}
                if not self._authorized(need):
                    return
                try:
                    return self._json(200, api.route(path, parse_qs(parsed.query)))
                except ValueError as e:
                    return self._json(400, {"error": str(e)})
                except (KeyError, LookupError):
                    return self._json(404, {"error": "not found"})
                except Exception as e:  # surface analysis errors to the UI instead of hanging
                    return self._json(500, {"error": str(e)})
            self._send(404, b"not found", "text/plain")

        def _body(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n <= 0:
                self._json(400, {"error": "a JSON body with Content-Length is required"})
                return None
            if n > MAX_BODY:
                self._json(413, {"error": f"body too large (max {MAX_BODY} bytes)"})
                return None
            try:
                return json.loads(self.rfile.read(n))
            except (ValueError, UnicodeDecodeError):
                self._json(400, {"error": "invalid JSON"})
                return None

        def do_POST(self):
            path = urlparse(self.path).path
            if path not in ("/ingest", "/budget/reserve", "/budget/release"):
                return self._send(404, b"not found", "text/plain")
            if not self._authorized({"ingest"}):
                return
            body = self._body()
            if body is None:
                return
            try:
                if path == "/ingest":
                    events = body.get("events") if isinstance(body, dict) and "events" in body else body
                    events = events if isinstance(events, list) else [events]
                    if len(events) > MAX_BATCH:
                        return self._json(413, {"error": f"batch too large (max {MAX_BATCH})"})
                    results = api.ingestor.handle_batch(events)
                    api.invalidate()
                    errors = sum(r["status"] == "error" for r in results)
                    return self._json(200 if not errors else 207,
                                      {"accepted": sum(r["status"] == "ok" for r in results),
                                       "duplicates": sum(r["status"] == "duplicate" for r in results),
                                       "errors": errors, "results": results})
                if not isinstance(body, dict):
                    return self._json(400, {"error": "body must be a JSON object"})
                code, obj = (api.ingestor.reserve if path == "/budget/reserve" else api.ingestor.release)(body)
                api.invalidate()
                self._json(code, obj, {"Retry-After": "0"} if code == 429 else None)
            except Exception as e:
                self._json(500, {"error": str(e)})

    return Handler


def serve(db: str, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False,
          auth: Auth = None, insecure: bool = False, collect_claude: str = None, collect_interval: float = 30.0,
          lookback_days: float = 1.0, budget=None) -> None:
    auth = auth or Auth.from_args()
    check_bind_security(host, auth, insecure)
    api = Api(Ledger(db), db)
    if collect_claude is not None:
        from .collect_claude import DEFAULT_ROOT, ClaudeCodeCollector
        col = ClaudeCodeCollector(api.ingestor, collect_claude or DEFAULT_ROOT, lookback_days, budget)
        first = col.run_once()                                   # backfill before serving, so the first page load has data
        api.invalidate()
        print(f"Collector: {col.root}  lookback {lookback_days}d  first pass: {first}")
        col.start(collect_interval, on_pass=api.invalidate)
    httpd = ThreadingHTTPServer((host, port), make_handler(api, auth))
    url = f"http://{host}:{port}/"
    print(f"Token observability UI: {url}   (ledger: {db}, auth: {'on' if auth.enabled else 'off'})  Ctrl+C to stop")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
