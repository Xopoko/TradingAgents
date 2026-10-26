"""Typed TradingAgents opinions for the isolated Ethereum WBTC/USDT fork."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.agents.structured import strict_structured_outputs
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.portfolio import PortfolioContext, Position

from .contracts import AnalysisDecision, ForkSnapshot, ManagedPortfolio
from .profile import BTC_SCALE, USDT_SCALE

ANALYSIS_TICKER = "BTC-USD"
MAX_SNAPSHOT_AGE = timedelta(seconds=120)
MAX_FUTURE_SKEW = timedelta(seconds=5)
DECISION_VALIDITY = timedelta(minutes=15)


class AnalysisInputError(ValueError):
    """The fork snapshot or managed allocation is not usable."""


class AnalysisOutputError(ValueError):
    """The graph did not return a validated final portfolio opinion."""


class _ForkPortfolioContext(PortfolioContext):
    managed_cash_units: int
    managed_btc_units: int
    capital_limit_usdt: Decimal
    snapshot: ForkSnapshot

    def render(self, ticker: str) -> str:
        snapshot = self.snapshot
        return (
            "LOCAL ETHEREUM FORK ONLY: chain_id=31337, source_chain_id=1. "
            "No mainnet execution. WBTC is the Ethereum representation of BTC; "
            "it is not native Bitcoin. Analyze WBTC/USDT spot for an hourly decision. "
            "Yahoo BTC-USD provides daily reference-market analysis, not an executable "
            "WBTC/USDT quote and not an intraday-bar strategy."
            f"\nManaged WBTC: {Decimal(self.managed_btc_units) / BTC_SCALE} "
            f"({self.managed_btc_units} base units; 8 decimals)."
            f"\nManaged cash: {Decimal(self.managed_cash_units) / USDT_SCALE} USDT "
            f"({self.managed_cash_units} base units; 6 decimals)."
            f"\nAvailable wallet WBTC: {Decimal(snapshot.btc_units) / BTC_SCALE} "
            f"({snapshot.btc_units} base units)."
            f"\nAvailable wallet USDT: {Decimal(snapshot.cash_units) / USDT_SCALE} "
            f"({snapshot.cash_units} base units)."
            f"\nManaged capital limit: {self.capital_limit_usdt} USDT. "
            "Only managed balances belong to this strategy. "
            "No leverage, debt, borrowing, short sales or derivatives. "
            "Sell only managed WBTC; never increase the allocation using other wallet assets."
            f"\nLocal fork pool spot price: {snapshot.btc_price_usdt} USDT per WBTC; "
            f"observed at {snapshot.observed_at.isoformat()}, block {snapshot.block_number} "
            f"({snapshot.block_hash}). This price is for valuation, not an executable swap "
            "quote. Execution requires a separate fresh quote and deterministic policy checks."
        )


def _graph_factory(**kwargs: Any) -> Any:
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    return TradingAgentsGraph(**kwargs)


def analysis_config(state_root: Path) -> dict[str, Any]:
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


def _validate_inputs(
    snapshot: ForkSnapshot,
    portfolio: ManagedPortfolio,
    now: datetime,
) -> tuple[ForkSnapshot, ManagedPortfolio]:
    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise AnalysisInputError("now must have a timezone")
    if not isinstance(snapshot, ForkSnapshot) or not isinstance(portfolio, ManagedPortfolio):
        raise AnalysisInputError("snapshot and portfolio must be validated contracts")
    try:
        # Validate dictionaries, since model_copy/model_construct can bypass
        # validation on an existing instance, including its nested identity.
        snapshot = ForkSnapshot.model_validate(snapshot.model_dump())
        portfolio = ManagedPortfolio.model_validate(portfolio.model_dump())
    except (ValueError, TypeError) as exc:
        raise AnalysisInputError("invalid fork snapshot or managed portfolio") from exc
    age = now - snapshot.observed_at
    if age > MAX_SNAPSHOT_AGE or age < -MAX_FUTURE_SKEW:
        raise AnalysisInputError("snapshot is stale or ahead of the clock")
    if portfolio.cash_units > snapshot.cash_units or portfolio.btc_units > snapshot.btc_units:
        raise AnalysisInputError("managed portfolio exceeds available token balances")
    return snapshot, portfolio


def _typed_decision(state: Any) -> PortfolioDecision:
    if not isinstance(state, dict):
        raise AnalysisOutputError("graph state must be a mapping")
    decision = state.get("final_decision")
    if not isinstance(decision, PortfolioDecision):
        raise AnalysisOutputError("final_decision must be a typed PortfolioDecision")
    if not isinstance(decision.rating, PortfolioRating):
        raise AnalysisOutputError("final rating must be a PortfolioRating")
    if state.get("final_rating") != decision.rating.value:
        raise AnalysisOutputError("final_rating disagrees with typed decision")
    if not isinstance(decision.executive_summary, str) or not decision.executive_summary.strip():
        raise AnalysisOutputError("final decision summary is empty")
    return decision


def analyze(
    snapshot: ForkSnapshot,
    portfolio: ManagedPortfolio,
    state_root: Path,
    now: datetime,
) -> AnalysisDecision:
    """Return a typed opinion; this function never quotes or executes a swap."""
    started = monotonic()
    snapshot, portfolio = _validate_inputs(snapshot, portfolio, now)
    context = _ForkPortfolioContext(
        cash=float(Decimal(portfolio.cash_units) / USDT_SCALE),
        currency="USDT",
        positions=[
            Position(
                ticker=ANALYSIS_TICKER,
                quantity=float(Decimal(portfolio.btc_units) / BTC_SCALE),
            )
        ],
        managed_cash_units=portfolio.cash_units,
        managed_btc_units=portfolio.btc_units,
        capital_limit_usdt=portfolio.capital_limit_usdt,
        snapshot=snapshot,
    )
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
        expires_at=completed_at + DECISION_VALIDITY,
        summary=decision.executive_summary.strip(),
        source="structured",
    )
