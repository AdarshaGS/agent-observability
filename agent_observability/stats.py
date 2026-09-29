"""Tiny stdlib statistics helpers."""
from __future__ import annotations

import math
from typing import Optional, Sequence


def mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def median(xs: Sequence[float]) -> Optional[float]:
    return percentile(xs, 50)


def percentile(xs: Sequence[float], p: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def stdev(xs: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def slope(ys: Sequence[float], xs: Optional[Sequence[float]] = None) -> Optional[float]:
    """Least-squares slope of ys against xs (default 0..n-1)."""
    n = len(ys)
    if n < 2:
        return None
    xs = list(xs) if xs is not None else list(range(n))
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return None if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
