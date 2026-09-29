"""Client-side token estimation, content hashing and privacy-preserving fingerprints.

Provider-reported usage is the source of truth for billing. The estimator only splits the
*input* into segments; the residual (provider - sum(segments)) is stored as `est_error`
so drift is visible instead of silently corrupting the waste analysis.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class TokenEstimator(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicEstimator:
    """~chars/4. Swap in a real tokenizer (e.g. tiktoken) by implementing `count`."""

    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self.chars_per_token))


_WS = re.compile(r"\s+")


def content_hash(text: str) -> str:
    """Whitespace-normalised hash so trivially reformatted duplicates still match."""
    return hashlib.sha256(_WS.sub(" ", text.strip()).encode("utf-8")).hexdigest()[:16]


_TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-./:@]{5,}")


def _h(tok: str) -> str:
    return hashlib.blake2b(tok.lower().encode("utf-8"), digest_size=4).hexdigest()


def fingerprint(text: str, limit: int = 64) -> list[str]:
    """Bottom-`limit` hashes of distinctive tokens (identifiers, paths, ids, long words).

    Bottom-k sampling is consistent across texts, so a later output that contains a sampled
    token also contains its hash. Used to test 'was this context ever referenced' without
    storing any content. Approximate by design: paraphrased use is invisible.
    """
    hashes = {_h(t) for t in _TOKEN.findall(text)}
    return sorted(hashes)[:limit]
