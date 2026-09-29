"""Agent token observability: structured telemetry, budget enforcement, deterministic waste/routing analysis."""
from .aggregate import AggregateConfig, aggregate, compute_baselines
from .analyzer_input import ANALYZER_SYSTEM_PROMPT, build_payload, select_sessions
from .budget import (BudgetConfig, BudgetExhausted, BudgetManager, BudgetPolicy, BudgetStatus)
from .gateway import ContextBuilder, Gateway, ToolMemo
from .ledger import Ledger
from .metrics import agent_metrics, context_growth, session_summary
from .models import (CallRequest, CallResult, ProviderError, ProviderResponse, Segment, ToolCallRequest, Usage,
                     usage_from_anthropic, usage_from_openai)
from .pricing import ModelPrice, PriceTable
from .quality import record_eval
from .report import build_session_report, render_text
from .routing import RoutingConfig, analyze_task_routing, cheaper_model_evidence, stratified_matrix
from .waste import WasteConfig, analyze_waste

__all__ = [n for n in dir() if not n.startswith("_")]
