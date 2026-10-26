"""Offline evidence for the demo monitor's LLM boundary."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.agents.state import AgentState
from tradingagents.agents.structured import StructuredOutputError, strict_structured_outputs
from tradingagents.graph.propagation import Propagator
from tradingagents.integrations.bybit_demo import analysis
from tradingagents.integrations.bybit_demo.contracts import ManagedPortfolio, SpotSnapshot

NOW = datetime(2026, 10, 26, 12, tzinfo=UTC)


class WrongDecision(BaseModel):
    rating: str = "Buy"


def _decision():
    return PortfolioDecision(
        rating=PortfolioRating.OVERWEIGHT,
        executive_summary="Осторожно увеличить долю BTC.",
        investment_thesis="Наблюдаемые данные поддерживают решение.",
    )


def test_configuration_pins_chatgpt_and_isolates_storage(tmp_path):
    config = analysis.analysis_config(tmp_path)
    assert [config[k] for k in ("llm_provider", "quick_think_provider", "deep_think_provider")] == ["chatgpt"] * 3
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


@pytest.mark.parametrize("decision", [None, {}, "Rating: Buy", WrongDecision()])
def test_untyped_final_decision_is_rejected(decision):
    with pytest.raises(analysis.AnalysisOutputError, match="typed PortfolioDecision"):
        analysis._typed_decision({"final_decision": decision, "final_rating": "Buy"})


@pytest.mark.parametrize("rating", [None, "REVIEW", "Buy"])
def test_missing_review_or_disagreeing_rating_is_rejected(rating):
    with pytest.raises(analysis.AnalysisOutputError, match="disagrees"):
        analysis._typed_decision({"final_decision": _decision(), "final_rating": rating})


def test_constructed_invalid_enum_and_blank_summary_are_rejected():
    decision = _decision().model_copy(update={"rating": "Overweight"})
    with pytest.raises(analysis.AnalysisOutputError, match="PortfolioRating"):
        analysis._typed_decision({"final_decision": decision, "final_rating": "Overweight"})
    decision = _decision().model_copy(update={"executive_summary": "  "})
    with pytest.raises(analysis.AnalysisOutputError, match="summary"):
        analysis._typed_decision({"final_decision": decision, "final_rating": "Overweight"})


def _compiled_manager(model):
    """Keep real nested, parallel LangGraph dispatch across the strict boundary."""
    manager = create_portfolio_manager(model)
    inner = StateGraph(AgentState)
    inner.add_node("pm", manager)
    inner.add_edge(START, "pm")
    inner.add_edge("pm", END)
    outer = StateGraph(AgentState)
    outer.add_node("nested", inner.compile())
    outer.add_node("parallel", lambda state: {})
    outer.add_edge(START, "nested")
    outer.add_edge(START, "parallel")
    outer.add_edge(["nested", "parallel"], END)
    return outer.compile()


def _initial_state():
    state = Propagator().create_initial_state("BTC-USD", "2026-10-26", asset_type="crypto")
    state.update(investment_plan="plan", trader_investment_plan="plan")
    return state


def test_langgraph_preserves_strict_context_into_nested_parallel_nodes():
    model = MagicMock()
    model.with_structured_output.return_value.invoke.side_effect = ValueError("invalid output")
    with strict_structured_outputs():
        graph = _compiled_manager(model)
        with pytest.raises(StructuredOutputError):
            graph.invoke(_initial_state())
    model.invoke.assert_not_called()


def test_typed_pm_decision_survives_langgraph_state():
    model = MagicMock()
    model.with_structured_output.return_value.invoke.return_value = _decision()
    with strict_structured_outputs():
        graph = _compiled_manager(model)
        state = graph.invoke(_initial_state())
    assert analysis._typed_decision(state) == _decision()
    model.invoke.assert_not_called()


def test_wrong_typed_pm_schema_is_rejected_without_freetext():
    model = MagicMock()
    model.with_structured_output.return_value.invoke.return_value = WrongDecision()
    with strict_structured_outputs():
        graph = _compiled_manager(model)
        with pytest.raises(StructuredOutputError, match="PortfolioDecision"):
            graph.invoke(_initial_state())
    model.invoke.assert_not_called()


@pytest.fixture
def snapshot():
    return SpotSnapshot(
        observed_at=NOW,
        quote_at=NOW - timedelta(seconds=10),
        bid="60000",
        ask="60001",
        btc_available="2",
        usdt_available="50000",
        quantity_step="0.00001",
        price_step="0.01",
        min_quantity="0.00001",
        min_notional="5",
    )


@pytest.fixture
def portfolio():
    return ManagedPortfolio(cash_usdt="180", btc_quantity="0.002")


def test_analyze_passes_only_ringfenced_portfolio_and_returns_provenance(
    snapshot, portfolio, tmp_path, monkeypatch
):
    graph = MagicMock()
    graph.propagate.return_value = (
        {"final_decision": _decision(), "final_rating": "Overweight"},
        # A contradictory legacy signal cannot replace the typed PM output.
        "Sell",
    )
    factory = MagicMock(return_value=graph)
    monkeypatch.setattr(analysis, "_graph_factory", factory)
    elapsed = iter([100.0, 220.0])
    monkeypatch.setattr(analysis, "monotonic", lambda: next(elapsed))

    result = analysis.analyze(snapshot, portfolio, tmp_path, NOW)

    assert result.rating is PortfolioRating.OVERWEIGHT
    assert result.summary == _decision().executive_summary
    assert result.source == "structured"
    assert result.market_observed_at == snapshot.quote_at
    assert result.analyzed_at == NOW + timedelta(seconds=120)
    assert result.expires_at == result.analyzed_at + timedelta(minutes=15)
    assert result.decision_id
    kwargs = factory.call_args.kwargs
    assert kwargs["selected_analysts"] == ("market", "social", "news")
    assert kwargs["debug"] is False
    assert graph.propagate.call_args.args == ("BTC-USD", NOW.date().isoformat())
    assert graph.propagate.call_args.kwargs["asset_type"] == "crypto"
    passed = graph.propagate.call_args.kwargs["portfolio"]
    assert passed.cash == 180
    assert passed.currency == "USDT"
    assert passed.position_in("BTC-USD").quantity == 0.002
    assert passed.position_in("BTC-USD").average_price is None
    rendered = passed.render("BTC-USD")
    assert "No borrowing" in rendered
    assert "hourly" in rendered
    assert snapshot.quote_at.isoformat() in rendered
    assert "50000" not in rendered


@pytest.mark.parametrize("field", ["quote_at", "observed_at"])
@pytest.mark.parametrize("delta", [timedelta(seconds=-121), timedelta(seconds=6)])
def test_stale_or_future_snapshot_never_constructs_graph(
    snapshot, portfolio, tmp_path, monkeypatch, field, delta
):
    snapshot = snapshot.model_copy(update={field: NOW + delta})
    factory = MagicMock()
    monkeypatch.setattr(analysis, "_graph_factory", factory)
    with pytest.raises(analysis.AnalysisInputError, match=field):
        analysis.analyze(snapshot, portfolio, tmp_path, NOW)
    factory.assert_not_called()


@pytest.mark.parametrize(
    "updates",
    [
        {"bid": Decimal("NaN")},
        {"bid": Decimal("0")},
        {"ask": Decimal("1")},
        {"environment": "live"},
        {"symbol": "ETHUSDT"},
        {"quote_at": NOW.replace(tzinfo=None)},
    ],
)
def test_invalid_constructed_snapshot_is_rejected(snapshot, portfolio, tmp_path, updates):
    with pytest.raises(analysis.AnalysisInputError):
        analysis.analyze(snapshot.model_copy(update=updates), portfolio, tmp_path, NOW)


def test_borrowing_and_unbacked_allocation_are_rejected(snapshot, portfolio, tmp_path):
    with pytest.raises(analysis.AnalysisInputError, match="borrowed"):
        analysis.analyze(snapshot.model_copy(update={"btc_borrowed": Decimal("0.1")}), portfolio, tmp_path, NOW)
    with pytest.raises(analysis.AnalysisInputError, match="exceeds"):
        analysis.analyze(snapshot, portfolio.model_copy(update={"cash_usdt": Decimal("60000")}), tmp_path, NOW)
    with pytest.raises(analysis.AnalysisInputError, match="timezone"):
        analysis.analyze(snapshot, portfolio, tmp_path, NOW.replace(tzinfo=None))


@pytest.mark.parametrize("phase", ["construction", "execution"])
def test_analyze_wraps_construction_and_execution_in_strict_context(
    snapshot, portfolio, tmp_path, monkeypatch, phase
):
    from tradingagents.agents.structured import bind_structured, invoke_structured

    model = MagicMock()
    model.with_structured_output.side_effect = NotImplementedError

    def factory(**kwargs):
        if phase == "construction":
            bind_structured(model, PortfolioDecision, "Portfolio Manager")
        graph = MagicMock()
        graph.propagate.side_effect = lambda *args, **kwargs: invoke_structured(None, "prompt", "Portfolio Manager")
        return graph

    monkeypatch.setattr(analysis, "_graph_factory", factory)
    with pytest.raises(StructuredOutputError):
        analysis.analyze(snapshot, portfolio, tmp_path, NOW)
    # No strict flag leaks into interactive callers after any failure.
    assert invoke_structured(None, "prompt", "Portfolio Manager") is None
    model.invoke.assert_not_called()
