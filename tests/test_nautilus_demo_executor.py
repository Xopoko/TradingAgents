"""Offline checks for execution boundaries and uncertain venue effects."""

import asyncio
import socket
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tradingagents.integrations.bybit_demo.contracts import (
    GatewayRequest,
    OrderIntent,
    SpotSnapshot,
)
from tradingagents.integrations.bybit_demo.nautilus_executor import (
    DemoBoundaryError,
    NautilusBybitDemoExecutor,
    _receipt_from_report,
    handle_request,
    main,
    validate_environment,
    validate_intent,
)

_SOCKET_CONNECT = socket.socket.connect
_SOCKETPAIR = socket.socketpair


@pytest.fixture(autouse=True)
def _allow_asyncio_internal_loopback_socketpair(monkeypatch):
    """Windows asyncio needs its local wakeup pair; external networking stays blocked."""
    def local_pair(*args, **kwargs):
        with monkeypatch.context() as scoped:
            scoped.setattr(socket.socket, "connect", _SOCKET_CONNECT)
            return _SOCKETPAIR(*args, **kwargs)

    monkeypatch.setattr(socket, "socketpair", local_pair)


def snapshot(**updates):
    now = datetime.now(UTC)
    values = {
        "observed_at": now, "quote_at": now, "bid": "100000", "ask": "100001",
        "btc_available": "1", "usdt_available": "1000", "quantity_step": "0.000001",
        "price_step": "0.01", "min_quantity": "0.000001", "min_notional": "5",
        "max_quantity": "100",
    }
    return SpotSnapshot.model_validate(values | updates)


def intent(**updates):
    values = {
        "client_order_id": "ta-demo-unit-1", "decision_id": "unit-1", "side": "Buy",
        "quantity": "0.001", "limit_price": "100001.00", "max_notional_usdt": "300",
        "expires_at": datetime.now(UTC) + timedelta(minutes=1),
    }
    return OrderIntent.model_validate(values | updates)


def test_valid_demo_cash_ioc_intent():
    validate_intent(intent(), snapshot())


@pytest.mark.parametrize("field", ["btc_borrowed", "usdt_borrowed"])
def test_existing_borrow_blocks_all_submission(field):
    with pytest.raises(DemoBoundaryError, match="existing_spot_borrow"):
        validate_intent(intent(), snapshot(**{field: "0.000001"}))


def test_sell_cannot_borrow_from_inherited_account():
    with pytest.raises(DemoBoundaryError, match="insufficient_unborrowed_btc"):
        validate_intent(intent(side="Sell"), snapshot(btc_available="0"))


def test_buy_leaves_fee_cash_and_cannot_borrow():
    with pytest.raises(DemoBoundaryError, match="insufficient_unborrowed_usdt"):
        validate_intent(intent(), snapshot(usdt_available="100.01"))


@pytest.mark.parametrize("updates,code", [
    ({"quantity": "0.0010001"}, "quantity_precision"),
    ({"limit_price": "100001.001"}, "price_precision"),
    ({"quantity": "0.000001"}, "below_minimum_notional"),
    ({"max_notional_usdt": "1"}, "intent_notional_limit"),
    ({"expires_at": datetime.now(UTC) - timedelta(seconds=1)}, "intent_expired"),
])
def test_invalid_order_blocks_before_engine(updates, code):
    with pytest.raises(DemoBoundaryError, match=code):
        validate_intent(intent(**updates), snapshot())


def test_stale_quote_and_open_order_block_execution():
    with pytest.raises(DemoBoundaryError, match="stale_quote"):
        validate_intent(intent(), snapshot(quote_at=datetime.now(UTC) - timedelta(minutes=1)))
    with pytest.raises(DemoBoundaryError, match="existing_open_spot_order"):
        validate_intent(intent(), snapshot(open_order_ids=["another-order"]))


@pytest.mark.parametrize("key", ["BYBIT_ENVIRONMENT", "BYBIT_BASE_URL", "BYBIT_PROXY_URL"])
def test_environment_cannot_redirect_to_mainnet(key):
    with pytest.raises(DemoBoundaryError, match="override_forbidden"):
        validate_environment({key: "https://api.bybit.com"})


def test_nonterminal_venue_acknowledgement_is_not_fill():
    report = SimpleNamespace(order_status="ACCEPTED", filled_qty="0", avg_px=None,
                             venue_order_id="venue-123")
    receipt = _receipt_from_report(report, "ta-demo-unit-1")
    assert receipt.status == "open"
    assert receipt.filled_quantity == 0


def test_native_cython_status_enum_is_not_treated_as_integer():
    pytest.importorskip("nautilus_trader")
    from nautilus_trader.model.enums import OrderStatus

    report = SimpleNamespace(order_status=OrderStatus.CANCELED, filled_qty="0", avg_px=None,
                             venue_order_id="venue-123")
    assert _receipt_from_report(report, "ta-demo-unit-1").status == "canceled"


def test_reconciliation_error_is_unknown_and_never_resubmits():
    executor = NautilusBybitDemoExecutor()
    executor._query_report = AsyncMock(side_effect=ValueError("private signed response"))
    executor._strategy = SimpleNamespace(submit_order=AsyncMock())
    receipt = asyncio.run(executor.reconcile("ta-demo-unit-1"))
    assert receipt.status == "unknown"
    assert "private" not in receipt.model_dump_json()
    executor._strategy.submit_order.assert_not_called()


def test_native_json_failure_uses_read_only_fallback_without_submitting(monkeypatch):
    from tradingagents.integrations.bybit_demo import nautilus_executor as module
    from tradingagents.integrations.bybit_demo.contracts import ExecutionReceipt

    expected = ExecutionReceipt(
        client_order_id="ta-demo-unit-1", status="filled", venue_order_id="venue-123",
        filled_quantity="0.001", filled_quote="100", fee_btc="0.000001",
        observed_at=datetime.now(UTC),
    )
    queried = []

    def fallback(key):
        queried.append(key)
        return expected

    monkeypatch.setattr(module, "reconcile_demo_order", fallback)
    executor = NautilusBybitDemoExecutor()
    executor._query_report = AsyncMock(side_effect=ValueError("JSON invalid type"))
    executor._strategy = SimpleNamespace(submit_order=AsyncMock())
    assert asyncio.run(executor.reconcile("ta-demo-unit-1")) == expected
    assert queried == ["ta-demo-unit-1"]
    executor._strategy.submit_order.assert_not_called()


def test_attempted_order_only_reconciles_even_when_absent():
    executor = NautilusBybitDemoExecutor()
    executor._attempted.add("ta-demo-unit-1")
    executor._query_report = AsyncMock(return_value=None)
    executor.snapshot = AsyncMock(side_effect=AssertionError("must not plan another trade"))
    receipt = asyncio.run(executor.execute_once(intent()))
    assert receipt.status == "unknown"
    executor.snapshot.assert_not_called()


def test_failure_response_does_not_include_exception_or_credentials(monkeypatch):
    async def fail(_self):
        raise RuntimeError("secret-key-value signed private URL")

    monkeypatch.setattr(NautilusBybitDemoExecutor, "start", fail)
    response = handle_request(GatewayRequest(operation="snapshot"))
    assert not response.ok
    assert response.error_code == "executor_failed"
    assert "secret-key-value" not in response.model_dump_json()


def test_cli_response_file_is_exact_typed_json(monkeypatch, tmp_path, capsys):
    import io
    import sys

    from tradingagents.integrations.bybit_demo import nautilus_executor as module
    from tradingagents.integrations.bybit_demo.contracts import GatewayResponse

    destination = tmp_path / "response.json"
    response = GatewayResponse(ok=True, snapshot=snapshot())
    monkeypatch.setattr(module, "handle_request", lambda request: response)
    monkeypatch.setattr(sys, "argv", ["executor", "--response-file", str(destination)])
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"operation":"snapshot"}'))
    assert main() == 0
    assert GatewayResponse.model_validate_json(destination.read_text(encoding="utf-8")) == response
    assert capsys.readouterr().out == ""
    assert list(tmp_path.iterdir()) == [destination]


def test_unknown_fill_fees_cannot_claim_terminal_execution():
    executor = NautilusBybitDemoExecutor()
    executor._exec_client = SimpleNamespace(
        pyo3_account_id="BYBIT-UNIFIED",
        _http_client=SimpleNamespace(request_fill_reports=AsyncMock(return_value=[])),
    )
    report = SimpleNamespace(order_status="FILLED", filled_qty="0.001", avg_px="100000",
                             venue_order_id="venue-123")
    pytest.importorskip("nautilus_trader")
    receipt = asyncio.run(executor._complete_receipt(report, "ta-demo-unit-1"))
    assert receipt.status == "unknown"
    assert receipt.safe_message == "execution_fees_unconfirmed"


def test_fill_quantity_quote_and_commissions_reconcile_exactly():
    pytest.importorskip("nautilus_trader")
    from nautilus_trader.core import nautilus_pyo3 as native

    fee = native.Money.from_str("0.000001 BTC")
    fills = [SimpleNamespace(venue_order_id="venue-123", trade_id="fill-1",
                             last_qty="0.001", last_px="100000", commission=fee)]
    executor = NautilusBybitDemoExecutor()
    executor._exec_client = SimpleNamespace(pyo3_account_id="BYBIT-UNIFIED",
        _http_client=SimpleNamespace(request_fill_reports=AsyncMock(return_value=fills + fills)))
    report = SimpleNamespace(order_status="FILLED", filled_qty="0.001", avg_px="100000",
                             venue_order_id="venue-123")
    receipt = asyncio.run(executor._complete_receipt(report, "ta-demo-unit-1"))
    assert receipt.status == "filled"
    assert receipt.filled_quantity == Decimal("0.001")
    assert receipt.filled_quote == Decimal("100")
    assert receipt.fee_btc == Decimal("0.000001")
    assert receipt.fee_usdt == 0


def test_installed_native_adapter_configuration_is_demo_spot_only(monkeypatch):
    pytest.importorskip("nautilus_trader")
    from nautilus_trader.adapters.bybit import BybitEnvironment, BybitProductType

    monkeypatch.setenv("BYBIT_DEMO_API_KEY", "unit-test-placeholder-key")
    monkeypatch.setenv("BYBIT_DEMO_API_SECRET", "unit-test-placeholder-secret")

    async def build():
        executor = NautilusBybitDemoExecutor()
        executor._build_node()  # Construction only: no connect/start/network.
        cfg = executor._node._config.exec_clients["BYBIT"]
        assert cfg.environment == BybitEnvironment.DEMO
        assert cfg.product_types == (BybitProductType.SPOT,)
        assert cfg.base_url_http == "https://api-demo.bybit.com"
        assert cfg.base_url_ws_private == "wss://stream-demo.bybit.com/v5/private"
        assert cfg.base_url_ws_trade == "wss://stream-demo.bybit.com/v5/trade"
        assert cfg.max_retries == 0
        assert cfg.auto_repay_spot_borrows is False
        assert executor._node._config.logging.log_level == "OFF"
        assert executor._node._config.logging.print_config is False
        return executor

    executor = asyncio.run(build())
    executor.dispose()
