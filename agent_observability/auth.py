"""Bearer-token auth with two scopes.

  read    -> dashboard + /api/* (usage metadata is sensitive)
  ingest  -> POST /ingest, /budget/reserve, /budget/release (and GET /api/budget/<id>)

No tokens configured => open, which the server only allows on loopback. Tokens must be >= 16 chars and are
compared in constant time. There is no per-tenant isolation: any valid `read` token sees every session.
"""
from __future__ import annotations

import hmac
import os
import secrets
from typing import Optional

MIN_TOKEN_LEN = 16
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def generate_token() -> str:
    return secrets.token_urlsafe(24)


class Auth:
    def __init__(self, read: Optional[str] = None, ingest: Optional[str] = None, both: Optional[str] = None):
        self._tokens: list[tuple[str, frozenset]] = []
        for tok, scopes in ((both, {"read", "ingest"}), (read, {"read"}), (ingest, {"ingest"})):
            if tok:
                if len(tok) < MIN_TOKEN_LEN:
                    raise ValueError(f"tokens must be at least {MIN_TOKEN_LEN} characters (try `python3 -m agent_observability token`)")
                self._tokens.append((tok, frozenset(scopes)))

    @classmethod
    def from_args(cls, token=None, read_token=None, ingest_token=None) -> "Auth":
        env = os.environ.get
        return cls(read=read_token or env("OBS_READ_TOKEN"), ingest=ingest_token or env("OBS_INGEST_TOKEN"),
                   both=token or env("OBS_TOKEN"))

    @property
    def enabled(self) -> bool:
        return bool(self._tokens)

    def check(self, authorization_header: Optional[str], need: set) -> int:
        """Returns 200 (ok), 401 (missing/invalid token) or 403 (valid token, wrong scope)."""
        if not self.enabled:
            return 200
        header = authorization_header or ""
        presented = header[7:].strip() if header[:7].lower() == "bearer " else ""
        if not presented:
            return 401
        matched: Optional[frozenset] = None
        for tok, scopes in self._tokens:                       # no early exit: constant-ish time
            if hmac.compare_digest(tok.encode(), presented.encode()):
                matched = scopes if matched is None else matched | scopes
        if matched is None:
            return 401
        return 200 if matched & need else 403


def check_bind_security(host: str, auth: Auth, insecure: bool = False) -> None:
    if host not in LOOPBACK and not auth.enabled and not insecure:
        raise ValueError(f"refusing to bind {host} without authentication. Set --token (or --read-token and "
                         "--ingest-token), or pass --insecure if a trusted proxy in front handles auth.")
