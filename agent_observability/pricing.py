"""Price tables. No prices are hardcoded: supply your own (per million tokens) and version them.

Unknown model => cost is NULL and reported as 'unpriced', never silently zero.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from .models import Usage


@dataclass(frozen=True)
class ModelPrice:
    input: float                 # $/MTok uncached input
    output: float                # $/MTok output (incl. reasoning)
    cache_read: Optional[float] = None
    cache_write: Optional[float] = None


@dataclass
class PriceTable:
    version: str = "unpriced"
    prices: dict[str, ModelPrice] = field(default_factory=dict)

    def register(self, model_id: str, price: ModelPrice) -> None:
        self.prices[model_id] = price

    def cost(self, model_id: str, usage: Usage) -> tuple[Optional[float], Optional[float]]:
        p = self.prices.get(model_id)
        if p is None:
            return None, None
        cr = p.cache_read if p.cache_read is not None else p.input
        cw = p.cache_write if p.cache_write is not None else p.input
        cost_in = (usage.input_fresh * p.input + usage.cache_read * cr + usage.cache_write * cw) / 1e6
        cost_out = usage.output * p.output / 1e6
        return cost_in, cost_out

    @classmethod
    def from_json(cls, path: str) -> "PriceTable":
        with open(path) as f:
            d = json.load(f)
        t = cls(version=d.get("version", "unversioned"))
        for m, p in d.get("models", {}).items():
            t.register(m, ModelPrice(p["input"], p["output"], p.get("cache_read"), p.get("cache_write")))
        return t
