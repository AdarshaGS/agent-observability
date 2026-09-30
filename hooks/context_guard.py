#!/usr/bin/env python3
"""Claude Code Stop hook: after every response show how much context is used vs the per-model cap.

Context of the latest call = input + cache read + cache write tokens from the transcript's last assistant usage.
Edit CAPS to change limits (first substring match on the model id wins). WARN_AT is the "start wrapping up" line.
"""
import json
import sys
from pathlib import Path

CAPS = {"opus": 300_000, "sonnet": 300_000, "haiku": 300_000}
DEFAULT_CAP = 300_000
WARN_AT = 150_000

try:
    ev = json.load(sys.stdin)
    lines = Path(ev["transcript_path"]).read_bytes()[-400_000:].splitlines()
    for line in reversed(lines):
        try:
            m = json.loads(line).get("message") or {}
        except ValueError:
            continue
        u = m.get("usage")
        if u and m.get("model") not in (None, "<synthetic>"):
            break
    else:
        sys.exit(0)
    ctx = u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
    cap = next((v for k, v in CAPS.items() if k in m["model"]), DEFAULT_CAP)
    msg = f"context {ctx // 1000}k / {cap // 1000}k cap ({ctx * 100 // cap}%, {(cap - ctx) // 1000}k left)"
    if ctx > cap:
        msg += " - OVER CAP: save a handoff note, start a fresh session"
    elif ctx >= WARN_AT:
        msg += " - getting heavy: finish this piece, then hand off to a fresh session"
    print(json.dumps({"systemMessage": msg}))
except Exception:
    pass          # a guard must never break the session
