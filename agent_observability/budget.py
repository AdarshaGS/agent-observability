"""Hard session token budget with atomic pre-flight reservation.

Enforced in the gateway (not in the prompt). Subagent sessions share their root's budget.

Levels (fraction of limit consumed):
    <60% normal | 60% warning | 80% high | 90% critical | 100% hard_stop

Normal calls may only use (limit - finalization_reserve). The reserve exists so the agent can
always run one wrap-up call and emit a resumable checkpoint instead of dying mid-edit.
Degrade WASTE first (compaction, dedupe, memoisation); degrade CAPABILITY last.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from .ledger import Ledger
from .models import Usage

LEVELS = ("normal", "warning", "high", "critical", "hard_stop")
THRESHOLDS = ((0.60, "warning"), (0.80, "high"), (0.90, "critical"), (1.00, "hard_stop"))


@dataclass(frozen=True)
class BudgetConfig:
    limit: int = 300_000
    finalization_reserve: int = 12_000
    basis: str = "billed"              # billed: all tokens; fresh: excludes cache reads
    input_safety_margin: float = 1.05  # inflate input estimate to absorb tokenizer drift
    min_output_tokens: int = 256       # below this a call is not worth making
    critical_output_cap: int = 4096    # per-call output cap at >=90% (not applied to finalization)


@dataclass(frozen=True)
class BudgetPolicy:
    level: str
    inject_state_line: bool = False
    compact_history: bool = False
    dedupe_tool_calls: bool = False
    disable_optional_tools: bool = False
    wrap_up: bool = False
    output_cap: Optional[int] = None
    guidance: str = ""


def policy_for(level: str, cfg: BudgetConfig) -> BudgetPolicy:
    if level == "normal":
        return BudgetPolicy(level, guidance="Work normally.")
    if level == "warning":
        return BudgetPolicy(level, inject_state_line=True,
                            guidance="No capability change. Prefer targeted tool calls; stop speculative exploration.")
    if level == "high":
        return BudgetPolicy(level, inject_state_line=True, compact_history=True, dedupe_tool_calls=True,
                            guidance="Auto-compact stale history/tool results into structured state (pin exact "
                                     "values and original requirements). Memoise repeat tool calls. Finish the "
                                     "current plan; no new broad exploration.")
    if level == "critical":
        return BudgetPolicy(level, inject_state_line=True, compact_history=True, dedupe_tool_calls=True,
                            disable_optional_tools=True, wrap_up=True, output_cap=cfg.critical_output_cap,
                            guidance="Stop exploring. Produce the deliverable now; report what is incomplete "
                                     "and emit a resumable checkpoint. Optional tools disabled.")
    return BudgetPolicy("hard_stop", inject_state_line=True, wrap_up=True,
                        guidance="Hard stop. Only a finalization call (checkpoint) is allowed, if reserve remains.")


@dataclass(frozen=True)
class BudgetStatus:
    root_id: str
    limit: int
    used: int
    reserved: int
    finalization_reserve: int
    level: str
    policy: BudgetPolicy
    projected_calls_remaining: Optional[float] = None

    @property
    def fraction(self) -> float:
        return self.used / self.limit if self.limit else 1.0

    @property
    def remaining(self) -> int:
        return max(self.limit - self.used, 0)

    @property
    def remaining_normal(self) -> int:
        return max(self.limit - self.finalization_reserve - self.used - self.reserved, 0)

    def state_line(self) -> str:
        eta = "" if self.projected_calls_remaining is None else f" ~{self.projected_calls_remaining:.0f} calls left"
        return (f"[budget {self.fraction:.0%} used ({self.used:,}/{self.limit:,}) level={self.level}{eta}]")


class BudgetExhausted(Exception):
    def __init__(self, status: BudgetStatus, reason: str, needed: int = 0):
        super().__init__(f"token budget exhausted ({reason}): {status.state_line()}")
        self.status = status
        self.reason = reason                       # hard_stop | reserve_only | insufficient
        self.needed = needed
        # A finalization (checkpoint) call may still be possible inside the reserve.
        self.finalization_available = status.used < status.limit and status.remaining >= 1


@dataclass
class Reservation:
    root_id: str
    session_id: str
    amount: int
    est_input: int
    granted_output: int
    finalization: bool
    status_before: BudgetStatus
    open: bool = True


@dataclass
class CommitResult:
    used_before: int
    used_after: int
    fired: list[tuple[float, str]] = field(default_factory=list)
    status: Optional[BudgetStatus] = None


def level_for(fraction: float) -> str:
    level = "normal"
    for t, name in THRESHOLDS:
        if fraction >= t:
            level = name
    return level


class BudgetManager:
    def __init__(self, ledger: Ledger, config: Optional[BudgetConfig] = None, clock=time.time):
        self.ledger = ledger
        self.config = config or BudgetConfig()
        self.clock = clock

    # -- setup ---------------------------------------------------------------------
    def open_root(self, root_id: str, config: Optional[BudgetConfig] = None) -> None:
        c = config or self.config
        self.ledger.insert("budget_state", dict(root_id=root_id, budget_limit=c.limit,
                                                finalization_reserve=c.finalization_reserve, basis=c.basis),
                           or_ignore=True)

    # -- reads ---------------------------------------------------------------------
    def _row(self, root_id: str) -> dict:
        r = self.ledger.one("SELECT * FROM budget_state WHERE root_id=?", (root_id,))
        if r is None:
            raise KeyError(f"no budget opened for root {root_id}")
        return r

    def _status(self, row: dict, with_projection: bool = False) -> BudgetStatus:
        used, limit = row["used"], row["budget_limit"]
        level = level_for(used / limit if limit else 1.0)
        proj = None
        if with_projection:
            recent = self.ledger.query(
                """SELECT c.input_fresh, c.cache_read, c.cache_write, c.output_tokens FROM model_calls c
                   JOIN sessions s ON s.session_id=c.session_id
                   WHERE s.budget_root_id=? AND c.status!='blocked_budget' ORDER BY c.seq DESC LIMIT 3""",
                (row["root_id"],))
            if recent:
                per = [Usage(r["input_fresh"], r["output_tokens"], r["cache_read"], r["cache_write"]).counted(row["basis"])
                       for r in recent]
                avg = sum(per) / len(per)
                if avg > 0:
                    proj = max(limit - used, 0) / avg
        cfg = BudgetConfig(limit=limit, finalization_reserve=row["finalization_reserve"], basis=row["basis"],
                           input_safety_margin=self.config.input_safety_margin,
                           min_output_tokens=self.config.min_output_tokens,
                           critical_output_cap=self.config.critical_output_cap)
        return BudgetStatus(row["root_id"], limit, used, row["reserved"], row["finalization_reserve"], level,
                            policy_for(level, cfg), proj)

    def status(self, session_id: str) -> BudgetStatus:
        return self._status(self._row(self.ledger.root_of(session_id)), with_projection=True)

    # -- reserve / commit / release ----------------------------------------------------
    def reserve(self, session_id: str, est_input: int, requested_output: int,
                finalization: bool = False) -> Reservation:
        """Atomically reserve worst-case tokens BEFORE the call. Shrinks max output to fit;
        raises BudgetExhausted if even the minimum useful call does not fit."""
        root = self.ledger.root_of(session_id)
        cfg = self.config
        with self.ledger.transaction():
            row = self._row(root)
            status = self._status(row)
            used, reserved, limit = row["used"], row["reserved"], row["budget_limit"]
            if used >= limit:
                raise BudgetExhausted(status, "hard_stop")
            ceiling = limit if finalization else limit - row["finalization_reserve"]
            avail = ceiling - used - reserved
            est_in = math.ceil(est_input * cfg.input_safety_margin)
            granted = min(requested_output, avail - est_in)
            if granted < cfg.min_output_tokens:
                if not finalization and (limit - used - reserved) >= est_in + cfg.min_output_tokens:
                    raise BudgetExhausted(status, "reserve_only", est_in + cfg.min_output_tokens)
                raise BudgetExhausted(status, "insufficient", est_in + cfg.min_output_tokens)
            amount = est_in + granted
            self.ledger.execute("UPDATE budget_state SET reserved=reserved+? WHERE root_id=?", (amount, root))
            return Reservation(root, session_id, amount, est_in, granted, finalization, status)

    def _close(self, res: Reservation, usage: Optional[Usage]) -> CommitResult:
        if not res.open:
            raise RuntimeError("reservation already settled")
        res.open = False
        with self.ledger.transaction():
            row = self._row(res.root_id)
            counted = usage.counted(row["basis"]) if usage else 0
            before = row["used"]
            after = before + counted
            self.ledger.execute(
                "UPDATE budget_state SET used=?, reserved=MAX(reserved-?,0) WHERE root_id=?",
                (after, res.amount, res.root_id))
            fired: list[tuple[float, str]] = []
            limit = row["budget_limit"]
            for t, name in THRESHOLDS:
                if after >= t * limit:
                    cur = self.ledger.execute(
                        "INSERT OR IGNORE INTO budget_events(root_id,session_id,level,threshold,tokens_at,action_taken,ts)"
                        " VALUES(?,?,?,?,?,?,?)",
                        (res.root_id, res.session_id, name, t, after,
                         policy_for(name, self.config).guidance, self.clock()))
                    if cur.rowcount:
                        fired.append((t, name))
            if after > limit:
                self.ledger.execute(
                    "INSERT OR IGNORE INTO budget_events(root_id,session_id,level,threshold,tokens_at,action_taken,ts)"
                    " VALUES(?,?,?,?,?,?,?)",
                    (res.root_id, res.session_id, "overshoot", None, after,
                     "estimate under-shot actual usage; tune input_safety_margin", self.clock()))
            new_row = self._row(res.root_id)
            return CommitResult(before, after, fired, self._status(new_row, with_projection=True))

    def record_usage(self, session_id: str, usage: Usage) -> CommitResult:
        """Account tokens that were already spent (no pre-flight). Used for externally reported calls:
        the budget is then advisory for that caller unless it reserves first."""
        root = self.ledger.root_of(session_id)
        with self.ledger.transaction():
            res = Reservation(root, session_id, 0, 0, 0, False, self._status(self._row(root)))
            return self._close(res, usage)

    def commit(self, res: Reservation, usage: Usage) -> CommitResult:
        return self._close(res, usage)

    def release(self, res: Reservation) -> CommitResult:
        """Call failed before any tokens were billed."""
        return self._close(res, None)
