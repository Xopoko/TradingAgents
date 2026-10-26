"""Offline checks for the local-fork TradingAgents analysis boundary."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.agents.state import AgentState
from tradingagents.agents.structured import (
    StructuredOutputError,
    bind_structured,
    invoke_structured,
)
from tradingagents.graph.propagation import Propagator
from tradingagents.integrations.onchain import analysis
from tradingagents.integrations.onchain.contracts import (
    ForkIdentity,
    ForkSnapshot,
    ManagedPortfolio,
)

NOW = datetime(2026, 10, 26, 12, tzinfo=UTC)


class WrongDecision(BaseModel):
    rating: str = "Buy"


def _decision():
    return PortfolioDecision(
        rating=PortfolioRating.OVERWEIGHT,
        executive_summary=" Осторожно увеличить долю WBTC. ",
        investment_thesis="Данные поддерживают увеличение доли.",
    )


@pytest.fixture
def snapshot():
    return ForkSnapshot(
        identity=ForkIdentity(
            instance_id="offline-fork",
            fork_block_number=20_000_000,
            fork_block_hash="0x" + "1" * 64,
            wallet_address="0x" + "2" * 40,
        ),
        observed_at=NOW - timedelta(seconds=10),
        block_number=20_000_010,
        block_hash="0x" + "3" * 64,
        cash_units=50_000_000_000,
        btc_units=100_000_000,
        native_wei=10**18,
        btc_price_usdt="60000.1234567890123456789",
    )


@pytest.fixture
def portfolio():
    return ManagedPortfolio(cash_units=180_000_000, btc_units=200_000)


@pytest.fixture
def fake_graph(monkeypatch):
    graph = MagicMock()
    graph.propagate.return_value = (
        {"final_decision": _decision(), "final_rating": "Overweight"},
        "Sell",  # The legacy string cannot override the typed PM decision.
    )
    factory = MagicMock(return_value=graph)
    monkeypatch.setattr(analysis, "_graph_factory", factory)
    return factory, graph


def test_model_and_storage_settings_are_pinned_and_independent(tmp_path):
    config = analysis.analysis_config(tmp_path)
    assert [
        config[key] for key in ("llm_provider", "quick_think_provider", "deep_think_provider")
    ] == ["chatgpt"] * 3
    assert config["quick_think_llm"] == "gpt-6-luna"
    assert config["deep_think_llm"] == "gpt-6-sol"
    assert config["output_language"] == "Russian"
    assert config["max_debate_rounds"] == config["max_risk_discuss_rounds"] == 1
    assert config["max_tool_rounds"] == 8
    assert config["checkpoint_enabled"] is False
    for key in ("results_dir", "data_cache_dir", "memory_log_path"):
        assert str(tmp_path.resolve() / "analysis") in config[key]
    for key in ("backend_url", "quick_think_backend_url", "deep_think_backend_url"):
        assert config[key] is None
    config["data_vendors"]["news_data"] = "modified"
    assert analysis.analysis_config(tmp_path)["data_vendors"]["news_data"] != "modified"


def test_analyze_uses_managed_units_and_labels_fork_price_as_nonexecutable(
    snapshot,
    portfolio,
    fake_graph,
    tmp_path,
    monkeypatch,
):
    factory, graph = fake_graph
    times = iter([100.0, 220.0])
    monkeypatch.setattr(analysis, "monotonic", lambda: next(times))
    decision = analysis.analyze(snapshot, portfolio, tmp_path, NOW)

    assert decision.rating is PortfolioRating.OVERWEIGHT
    assert decision.source == "structured"
    assert decision.summary == _decision().executive_summary.strip()
    assert decision.analyzed_at == NOW + timedelta(seconds=120)
    assert decision.expires_at == decision.analyzed_at + timedelta(minutes=15)
    assert decision.decision_id
    assert factory.call_args.kwargs["selected_analysts"] == ("market", "social", "news")
    assert factory.call_args.kwargs["debug"] is False
    assert graph.propagate.call_args.args == ("BTC-USD", "2026-10-26")
    assert graph.propagate.call_args.kwargs["asset_type"] == "crypto"
    context = graph.propagate.call_args.kwargs["portfolio"]
    assert context.cash == 180
    assert context.currency == "USDT"
    assert context.position_in("BTC-USD").quantity == 0.002
    assert context.position_in("BTC-USD").average_price is None
    rendered = context.render("BTC-USD")
    for required in (
        "LOCAL ETHEREUM FORK ONLY",
        "chain_id=31337",
        "source_chain_id=1",
        "Ethereum representation of BTC",
        "Managed WBTC: 0.002",
        "200000 base units; 8 decimals",
        "Managed cash: 180 USDT",
        "180000000 base units; 6 decimals",
        "daily reference-market",
        "not an executable",
        str(snapshot.btc_price_usdt),
        snapshot.observed_at.isoformat(),
        "No leverage, debt, borrowing",
        "Managed capital limit: 300 USDT",
        "Available wallet WBTC: 1",
    ):
        assert required in rendered
    assert "Bybit" not in rendered
    assert "BTCUSDT" not in rendered
    assert "DEMO" not in rendered


def test_empty_wbtc_holdings_are_explicit(snapshot, fake_graph, tmp_path):
    analysis.analyze(snapshot, ManagedPortfolio(), tmp_path, NOW)
    context = fake_graph[1].propagate.call_args.kwargs["portfolio"]
    assert context.position_in("BTC-USD").quantity == 0
    assert "Managed WBTC: 0 (0 base units; 8 decimals)" in context.render("BTC-USD")


@pytest.mark.parametrize("delta", [timedelta(seconds=-121), timedelta(seconds=6)])
def test_stale_or_future_snapshot_prevents_graph_construction(
    snapshot,
    portfolio,
    fake_graph,
    tmp_path,
    delta,
):
    with pytest.raises(analysis.AnalysisInputError, match="stale or ahead"):
        analysis.analyze(
            snapshot.model_copy(update={"observed_at": NOW + delta}), portfolio, tmp_path, NOW
        )
    fake_graph[0].assert_not_called()


@pytest.mark.parametrize(
    "updates",
    [
        {"btc_price_usdt": Decimal("NaN")},
        {"btc_price_usdt": Decimal("0")},
        {"cash_units": -1},
        {"btc_units": True},
        {"observed_at": NOW.replace(tzinfo=None)},
    ],
)
def test_invalid_snapshot_copies_are_rejected(snapshot, portfolio, fake_graph, tmp_path, updates):
    with pytest.raises(analysis.AnalysisInputError):
        analysis.analyze(snapshot.model_copy(update=updates), portfolio, tmp_path, NOW)
    fake_graph[0].assert_not_called()


def test_wrong_chain_and_unbacked_portfolio_are_rejected(snapshot, portfolio, fake_graph, tmp_path):
    identity = snapshot.identity.model_copy(update={"chain_id": 1})
    with pytest.raises(analysis.AnalysisInputError):
        analysis.analyze(
            snapshot.model_copy(update={"identity": identity}), portfolio, tmp_path, NOW
        )
    with pytest.raises(analysis.AnalysisInputError, match="exceeds"):
        analysis.analyze(
            snapshot,
            portfolio.model_copy(update={"btc_units": snapshot.btc_units + 1}),
            tmp_path,
            NOW,
        )
    with pytest.raises(analysis.AnalysisInputError, match="timezone"):
        analysis.analyze(snapshot, portfolio, tmp_path, NOW.replace(tzinfo=None))
    fake_graph[0].assert_not_called()


@pytest.mark.parametrize("typed", [None, "Rating: Buy", {}, WrongDecision()])
def test_untyped_final_decision_cannot_authorize_a_result(
    snapshot,
    portfolio,
    fake_graph,
    tmp_path,
    typed,
):
    fake_graph[1].propagate.return_value = ({"final_decision": typed, "final_rating": "Buy"}, "Buy")
    with pytest.raises(analysis.AnalysisOutputError, match="typed PortfolioDecision"):
        analysis.analyze(snapshot, portfolio, tmp_path, NOW)


@pytest.mark.parametrize("rating", [None, "REVIEW", "Sell"])
def test_missing_review_or_disagreeing_rating_is_rejected(
    snapshot,
    portfolio,
    fake_graph,
    tmp_path,
    rating,
):
    fake_graph[1].propagate.return_value = (
        {"final_decision": _decision(), "final_rating": rating},
        "Buy",
    )
    with pytest.raises(analysis.AnalysisOutputError, match="disagrees"):
        analysis.analyze(snapshot, portfolio, tmp_path, NOW)


def test_unvalidated_enum_and_blank_summary_are_rejected():
    decision = _decision().model_copy(update={"rating": "Overweight"})
    with pytest.raises(analysis.AnalysisOutputError, match="PortfolioRating"):
        analysis._typed_decision({"final_decision": decision, "final_rating": "Overweight"})
    decision = _decision().model_copy(update={"executive_summary": " "})
    with pytest.raises(analysis.AnalysisOutputError, match="summary"):
        analysis._typed_decision({"final_decision": decision, "final_rating": "Overweight"})


@pytest.mark.parametrize("phase", ["construction", "execution"])
def test_strict_mode_wraps_construction_and_execution_and_does_not_leak(
    snapshot,
    portfolio,
    tmp_path,
    monkeypatch,
    phase,
):
    model = MagicMock()
    model.with_structured_output.side_effect = NotImplementedError

    def factory(**kwargs):
        if phase == "construction":
            bind_structured(model, PortfolioDecision, "Portfolio Manager")
        graph = MagicMock()
        graph.propagate.side_effect = lambda *args, **kwargs: invoke_structured(
            None, "prompt", "Portfolio Manager"
        )
        return graph

    monkeypatch.setattr(analysis, "_graph_factory", factory)
    with pytest.raises(StructuredOutputError):
        analysis.analyze(snapshot, portfolio, tmp_path, NOW)
    assert invoke_structured(None, "prompt", "Portfolio Manager") is None
    model.invoke.assert_not_called()


@pytest.mark.parametrize("invalid", [False, True])
def test_real_langgraph_preserves_typed_output_and_strict_context(
    snapshot,
    portfolio,
    tmp_path,
    monkeypatch,
    invalid,
):
    model = MagicMock()
    model.with_structured_output.return_value.invoke.return_value = _decision()
    if invalid:
        model.with_structured_output.return_value.invoke.side_effect = ValueError("bad schema")

    class OfflineGraph:
        def __init__(self, **kwargs):
            graph = StateGraph(AgentState)
            graph.add_node("pm", create_portfolio_manager(model))
            graph.add_edge(START, "pm")
            graph.add_edge("pm", END)
            self.graph = graph.compile()

        def propagate(self, ticker, trade_date, asset_type, portfolio):
            state = Propagator().create_initial_state(
                ticker,
                trade_date,
                asset_type=asset_type,
                portfolio_context=portfolio.render(ticker),
            )
            state.update(investment_plan="offline", trader_investment_plan="offline")
            result = self.graph.invoke(state)
            return result, result.get("final_rating")

    monkeypatch.setattr(analysis, "_graph_factory", OfflineGraph)
    if invalid:
        with pytest.raises(StructuredOutputError):
            analysis.analyze(snapshot, portfolio, tmp_path, NOW)
    else:
        result = analysis.analyze(snapshot, portfolio, tmp_path, NOW)
        assert result.rating is PortfolioRating.OVERWEIGHT
        assert result.source == "structured"
    model.invoke.assert_not_called()
