"""python -m agent_observability <command> --db ledger.db"""
from __future__ import annotations

import argparse
import json
import sys

from .aggregate import aggregate
from .analyzer_input import build_payload, select_sessions
from .ledger import Ledger
from .report import build_session_report, render_text


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="agent_observability")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("report", "trajectory", "payload"):
        s = sub.add_parser(name)
        s.add_argument("--db", required=True)
        s.add_argument("--session", required=True)
        s.add_argument("--json", action="store_true")
    for name in ("aggregate", "select"):
        s = sub.add_parser(name)
        s.add_argument("--db", required=True)
        s.add_argument("--since", type=float)
        s.add_argument("--until", type=float)
    sv = sub.add_parser("serve")
    sv.add_argument("--db", required=True)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--open", action="store_true")
    sv.add_argument("--token", help="one token with read+ingest scope (or env OBS_TOKEN)")
    sv.add_argument("--read-token", help="dashboard/API only (or env OBS_READ_TOKEN)")
    sv.add_argument("--ingest-token", help="POST /ingest and /budget/* only (or env OBS_INGEST_TOKEN)")
    sv.add_argument("--insecure", action="store_true", help="allow a non-loopback bind without tokens")
    sv.add_argument("--collect-claude-code", nargs="?", const="", metavar="DIR",
                    help="pull Claude Code session transcripts (default dir ~/.claude/projects) on a timer")
    sv.add_argument("--collect-interval", type=float, default=30.0, help="seconds between passes")
    sv.add_argument("--lookback-days", type=float, default=1.0, help="only transcripts modified within this many days (1 = today)")
    sv.add_argument("--budget-limit", type=int, help="token budget for collected sessions (default 300000)")
    sv.add_argument("--budget-basis", choices=("billed", "fresh"), help="billed = all tokens, fresh = excl. cache reads")
    ct = sub.add_parser("compare-tag", help="tag root sessions (+ their subagents) as one cohort for the with-agents vs inline comparison")
    ct.add_argument("--db", required=True)
    ct.add_argument("--cohort", required=True)
    ct.add_argument("--arm", required=True, choices=("with_agents", "inline"))
    ct.add_argument("--label", required=True)
    ct.add_argument("--units", type=int, required=True, help="work units delivered (e.g. issues closed)")
    ct.add_argument("--unit-label", default="issue")
    ct.add_argument("--note")
    ct.add_argument("--sessions", nargs="+", required=True, help="root session ids or unique prefixes")
    cs = sub.add_parser("compare", help="print the with-agents vs inline comparison as JSON")
    cs.add_argument("--db", required=True)
    sub.add_parser("token", help="print a fresh random token")
    a = p.parse_args(argv)
    if a.cmd == "token":
        from .auth import generate_token
        print(generate_token())
        return 0
    if a.cmd == "serve":
        from .auth import Auth
        from .server import serve
        try:
            from .budget import BudgetConfig
            base = BudgetConfig()
            budget = BudgetConfig(limit=a.budget_limit or base.limit, basis=a.budget_basis or base.basis)
            serve(a.db, a.host, a.port, a.open, Auth.from_args(a.token, a.read_token, a.ingest_token), a.insecure,
                  a.collect_claude_code, a.collect_interval, a.lookback_days, budget)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        return 0
    led = Ledger(a.db)
    if a.cmd == "compare-tag":
        from .compare import tag
        ids = []
        for s in a.sessions:
            rows = led.query("SELECT session_id FROM sessions WHERE session_id LIKE ? AND parent_session_id IS NULL", (s + "%",))
            if len(rows) != 1:
                print(f"error: {s!r} matches {len(rows)} root sessions", file=sys.stderr)
                return 2
            ids.append(rows[0]["session_id"])
        print(json.dumps(tag(led, a.cohort, a.arm, a.label, a.units, ids, a.unit_label, a.note)))
        return 0
    if a.cmd == "compare":
        from .compare import compare
        print(json.dumps(compare(led, a.db), indent=2, default=str))
        return 0
    if a.cmd == "report":
        r = build_session_report(led, a.session)
        print(json.dumps(r, indent=2, default=str) if a.json else render_text(r))
    elif a.cmd == "trajectory":
        print(json.dumps(led.trajectory(a.session), indent=2, default=str))
    elif a.cmd == "payload":
        print(build_payload(build_session_report(led, a.session)))
    elif a.cmd == "aggregate":
        print(json.dumps(aggregate(led, since=a.since, until=a.until), indent=2, default=str))
    elif a.cmd == "select":
        print(json.dumps(select_sessions(led, since=a.since, until=a.until), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
