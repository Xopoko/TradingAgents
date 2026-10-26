"""Fail-closed TradingAgents analysis for the isolated Bybit demo monitor.

The language model supplies a typed opinion. The exchange quote, its timestamp,
the account balances, and the execution policy remain deterministic inputs.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal
from math import isfinite
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.agents.structured import strict_structured_outputs
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.portfolio import PortfolioContext, Position

from .contracts import AnalysisDecision, ManagedPortfolio, SpotSnapshot

MAX_QUOTE_AGE = timedelta(seconds=120)
MAX_FUTURE_SKEW = timedelta(seconds=5)
DECISION_VALIDITY = timedelta(minutes=15)
ANALYSIS_TICKER = "BTC-USD"


class AnalysisInputError(ValueError):
    """The exchange snapshot is not usable as analysis input."""


class AnalysisOutputError(ValueError):
    """The graph did not produce a proven, typed final opinion."""


class _DemoPortfolioContext(PortfolioContext):
    capital_limit_usdt: Decimal
    quote_bid: Decimal
    quote_ask: Decimal
    quote_at: datetime

    def render(self, ticker: str) -> str:
        return (
            super().render(ticker)
            + "\nBybit DEMO managed sub-portfolio only, quoted in USDT. "
            "Analyze BTCUSDT spot for an hourly decision; BTC-USD data is a "
            "reference market, not the executable USDT quote. No borrowing, "
            "short selling, derivatives or leverage. Sell only held BTC. "
            "Never use other account balances to increase this allocation."
            f"\nManaged capital allocation limit: {self.capital_limit_usdt} USDT."
            f"\nExchange quote: bid {self.quote_bid} USDT, ask {self.quote_ask} USDT; "
            f"source timestamp {self.quote_at.isoformat()}."
        )


def _graph_factory(**kwargs: Any) -> Any:
    # Importing a monitor contract must not initialize LLM clients or adapters.
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    return TradingAgentsGraph(**kwargs)


def analysis_config(state_root: Path) -> dict[str, Any]:
    """Pin the unattended run and isolate every graph-owned persistence path."""
    root = Path(state_root).resolve() / "analysis"
    config = deepcopy(DEFAULT_CONFIG)
    config.update(
        llm_provider="chatgpt",
        quick_think_provider="chatgpt",
        deep_think_provider="chatgpt",
        quick_think_llm="gpt-6-luna",
        deep_think_llm="gpt-6-sol",
        backend_url=None,
        quick_think_backend_url=None,
        deep_think_backend_url=None,
        output_language="Russian",
        max_debate_rounds=1,
        max_risk_discuss_rounds=1,
        max_tool_rounds=8,
        max_recur_limit=100,
        checkpoint_enabled=False,
        llm_max_retries=1,
        results_dir=str(root / "results"),
        data_cache_dir=str(root / "cache"),
        memory_log_path=str(root / "memory" / "trading_memory.md"),
    )
    return config


def _aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise AnalysisInputError(f"{name} must have a timezone")


def _typed_decision(state: Any) -> PortfolioDecision:
    if not isinstance(state, dict):
        raise AnalysisOutputError("graph state must be a mapping")
    decision = state.get("final_decision")
    if not isinstance(decision, PortfolioDecision):
        raise AnalysisOutputError("final_decision must be a typed PortfolioDecision")
    # Check the enum rather than letting a constructed/mutated model smuggle a
    # string or REVIEW sentinel through an otherwise valid Pydantic instance.
    if not isinstance(decision.rating, PortfolioRating):
        raise AnalysisOutputError("final decision rating must be a PortfolioRating")
    if state.get("final_rating") != decision.rating.value:
        raise AnalysisOutputError("final_rating disagrees with typed final decision")
    if not isinstance(decision.executive_summary, str) or not decision.executive_summary.strip():
        raise AnalysisOutputError("final decision summary is empty")
    return decision


def _validate_inputs(snapshot: SpotSnapshot, portfolio: ManagedPortfolio, now: datetime) -> None:
    _aware(now, "now")
    if not isinstance(snapshot, SpotSnapshot) or not isinstance(portfolio, ManagedPortfolio):
        raise AnalysisInputError("snapshot and portfolio must be validated contracts")
    # Revalidate even model_construct/model_copy inputs rather than assuming the
    # caller cannot bypass assignment validation on a Pydantic instance.
    try:
        SpotSnapshot.model_validate(snapshot.model_dump())
        ManagedPortfolio.model_validate(portfolio.model_dump())
    except ValueError as exc:
        raise AnalysisInputError("snapshot or managed portfolio is invalid") from exc
    for name in ("quote_at", "observed_at"):
        stamp = getattr(snapshot, name)
        _aware(stamp, name)
        age = now - stamp
        if age > MAX_QUOTE_AGE or age < -MAX_FUTURE_SKEW:
            raise AnalysisInputError(f"{name} is stale or ahead of the clock")
    if snapshot.btc_borrowed or snapshot.usdt_borrowed:
        raise AnalysisInputError("borrowed balances are incompatible with spot-only analysis")
    if portfolio.btc_quantity > snapshot.btc_available or portfolio.cash_usdt > snapshot.usdt_available:
        raise AnalysisInputError("managed portfolio exceeds available demo balances")
    if not all(isfinite(float(value)) for value in (portfolio.cash_usdt, portfolio.btc_quantity)):
        raise AnalysisInputError("managed portfolio cannot be represented by the analysis context")


def analyze(
    snapshot: SpotSnapshot,
    portfolio: ManagedPortfolio,
    state_root: Path,
    now: datetime,
) -> AnalysisDecision:
    """Run the real graph with typed output and deterministic quote provenance.

    ``now`` is the caller's aware start time. Monotonic elapsed time supplies the
    completion timestamp, including when the graph takes several minutes. The
    execution caller must fetch a fresh quote before creating any order.
    """
    started = monotonic()
    _validate_inputs(snapshot, portfolio, now)
    context = _DemoPortfolioContext(
        cash=float(portfolio.cash_usdt),
        currency="USDT",
        positions=[Position(ticker=ANALYSIS_TICKER, quantity=float(portfolio.btc_quantity))],
        capital_limit_usdt=portfolio.capital_limit_usdt,
        quote_bid=snapshot.bid,
        quote_ask=snapshot.ask,
        quote_at=snapshot.quote_at,
    )
    # Binding happens during graph construction; invocation happens inside
    # LangGraph workers. Both must inherit the fail-closed context.
    with strict_structured_outputs():
        graph = _graph_factory(
            selected_analysts=("market", "social", "news"),
            debug=False,
            config=analysis_config(state_root),
        )
        state, _signal = graph.propagate(
            ANALYSIS_TICKER,
            now.date().isoformat(),
            asset_type="crypto",
            portfolio=context,
        )
        decision = _typed_decision(state)
    completed_at = now + timedelta(seconds=monotonic() - started)
    return AnalysisDecision(
        decision_id=str(uuid4()),
        rating=decision.rating,
        analyzed_at=completed_at,
        market_observed_at=snapshot.quote_at,
        expires_at=completed_at + DECISION_VALIDITY,
        summary=decision.executive_summary.strip(),
        source="structured",
    )
