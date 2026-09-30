#!/usr/bin/env python3
"""Claude Code UserPromptSubmit hook: remind Claude of the search order when a code index exists in the project.

Injects one short line of context; it cannot force Claude to comply. Silent when there is no index (no wasted tokens).
"""
import json
import sys
from pathlib import Path

try:
    cwd = Path(json.load(sys.stdin).get("cwd", "."))
    tools = [t for d, t in ((".codegraph", "codegraph explore"), ("graphify-out", "graphify")) if (cwd / d).exists()]
    if tools:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext":
            f"Search order: 1) {' / '.join(tools)} first for code questions. 2) Only if that is not enough, delegate "
            "broader code search to the explorer agent and GitHub search to github-scout. 3) Don't grep/read many files in the main thread."}}))
except Exception:
    pass
