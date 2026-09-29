"""Tiny stdlib client for the ingest server. Other languages: POST the same JSON (see README "Ingest API")."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional


class BudgetDenied(Exception):
    def __init__(self, body: dict):
        super().__init__(f"budget denied: {body.get('reason')}")
        self.body = body


class ObsClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8765", token: Optional[str] = None, timeout: float = 10.0):
        self.base, self.token, self.timeout = base_url.rstrip("/"), token, timeout

    def _post(self, path: str, body) -> tuple[int, dict]:
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST", headers={
            "Content-Type": "application/json", **({"Authorization": f"Bearer {self.token}"} if self.token else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            finally:
                e.close()

    def ingest(self, *events: dict) -> dict:
        code, out = self._post("/ingest", {"events": list(events)})
        if code not in (200, 207):
            raise RuntimeError(f"ingest failed ({code}): {out}")
        return out

    def reserve(self, session_id: str, est_input_tokens: int, max_output_tokens: int, **kw) -> dict:
        """Call BEFORE the model call. Raises BudgetDenied on 429. Pass the returned reservation_id and
        granted_max_output_tokens into your call, then send the model_call event with reservation_id."""
        code, out = self._post("/budget/reserve", dict(session_id=session_id, est_input_tokens=est_input_tokens,
                                                        max_output_tokens=max_output_tokens, **kw))
        if code == 429:
            raise BudgetDenied(out)
        if code != 200:
            raise RuntimeError(f"reserve failed ({code}): {out}")
        return out

    def release(self, reservation_id: str) -> None:
        self._post("/budget/release", {"reservation_id": reservation_id})
