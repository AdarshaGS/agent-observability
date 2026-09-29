"""Plain data types shared across the package."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

SEGMENT_KINDS = ("system", "user", "history", "retrieval", "tool_result", "agent_state", "tool_schema")
STATIC_KINDS = ("system", "tool_schema")


def _get(obj: Any, name: str, default: int = 0) -> int:
    if obj is None:
        return default
    if isinstance(obj, dict):
        v = obj.get(name, default)
    else:
        v = getattr(obj, name, default)
    return int(v or 0)


@dataclass(frozen=True)
class Usage:
    """Provider-reported token usage for ONE model call.

    input_fresh  = uncached input tokens
    cache_read   = input tokens served from cache
    cache_write  = input tokens written to cache
    output       = all billed output tokens (INCLUDES reasoning/thinking)
    reasoning    = reasoning tokens, reported as a subset of `output` for analysis only
    """

    input_fresh: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    reasoning: int = 0

    @property
    def input_total(self) -> int:
        return self.input_fresh + self.cache_read + self.cache_write

    @property
    def billed(self) -> int:
        return self.input_total + self.output

    def counted(self, basis: str) -> int:
        """Tokens counted against the budget under `basis` ('billed' or 'fresh')."""
        return self.billed if basis == "billed" else self.billed - self.cache_read


def usage_from_anthropic(u: Any) -> Usage:
    """Anthropic `usage`: input_tokens EXCLUDES cache reads/writes; thinking is inside output_tokens."""
    return Usage(
        input_fresh=_get(u, "input_tokens"),
        output=_get(u, "output_tokens"),
        cache_read=_get(u, "cache_read_input_tokens"),
        cache_write=_get(u, "cache_creation_input_tokens"),
    )


def usage_from_openai(u: Any) -> Usage:
    """OpenAI `usage`: prompt_tokens INCLUDES cached tokens; completion_tokens includes reasoning."""
    prompt = _get(u, "prompt_tokens")
    details = u.get("prompt_tokens_details") if isinstance(u, dict) else getattr(u, "prompt_tokens_details", None)
    cdetails = u.get("completion_tokens_details") if isinstance(u, dict) else getattr(u, "completion_tokens_details", None)
    cached = _get(details, "cached_tokens")
    return Usage(
        input_fresh=max(prompt - cached, 0),
        output=_get(u, "completion_tokens"),
        cache_read=cached,
        reasoning=_get(cdetails, "reasoning_tokens"),
    )


@dataclass
class Segment:
    """One block of input context. Re-send the SAME object on later iterations so
    origin_call_id / first_iteration carry over and growth can be attributed."""

    kind: str
    tokens: int
    content_hash: str
    source_ref: Optional[str] = None
    origin_call_id: Optional[str] = None
    first_iteration: Optional[int] = None
    was_truncated: bool = False
    fingerprint: list[str] = field(default_factory=list)


@dataclass
class ToolCallRequest:
    name: str
    args: Any = None
    tool_call_id: Optional[str] = None


@dataclass
class ProviderResponse:
    usage: Usage
    text: str = ""
    stop_reason: Optional[str] = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    ttft_ms: Optional[float] = None
    raw: Any = None


class ProviderError(Exception):
    """Raise from a provider function. `usage` should be set if the provider billed tokens anyway."""

    def __init__(self, message: str = "", error_class: str = "unknown", retryable: bool = False,
                 usage: Optional[Usage] = None):
        super().__init__(message)
        self.error_class = error_class
        self.retryable = retryable
        self.usage = usage
        self.call_id: Optional[str] = None


INFRA_ERROR_CLASSES = frozenset({"rate_limit", "timeout", "server_error", "overloaded", "connection"})


@dataclass
class CallRequest:
    session_id: str
    agent_id: str
    model_id: str
    segments: list[Segment]
    task_id: Optional[str] = None
    iteration: int = 1
    max_output_tokens: int = 4096
    provider: str = ""
    agent_version: Optional[str] = None
    prompt_version: Optional[str] = None
    routing_decision_id: Optional[str] = None
    parent_call_id: Optional[str] = None
    finalization: bool = False
    max_attempts: int = 1


@dataclass
class CallResult:
    call_id: str
    response: ProviderResponse
    usage: Usage
    attempts: int
    budget: Any  # BudgetStatus after the call
