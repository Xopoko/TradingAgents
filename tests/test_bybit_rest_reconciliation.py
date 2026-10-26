"""Offline evidence checks for the read-only Bybit DEMO reconciliation fallback."""

import hashlib
import hmac
import json
from decimal import Decimal
from urllib.parse import parse_qs, urlsplit

import pytest

from tradingagents.integrations.bybit_demo.rest_reconciliation import DemoRestReconciler

CLIENT_ID = "ta-demo-offline-reconcile"


def order(**changes):
    return {
        "symbol": "BTCUSDT", "orderId": "venue-offline", "orderLinkId": CLIENT_ID,
        "orderStatus": "Filled", "qty": "0.000118", "cumExecQty": "0.000118",
        # Confirmed live SDK mismatch is irrelevant to the consumed contract.
        "smpGroup": "0", **changes,
    }


def fill(**changes):
    return {
        "symbol": "BTCUSDT", "orderId": "venue-offline", "orderLinkId": CLIENT_ID,
        "execId": "trade-offline", "execQty": "0.000118", "execPrice": "80000.1",
        "execFee": "0.000000118", "feeCurrency": "BTC", "execType": "Trade", **changes,
    }


def page(rows, cursor="", **changes):
    return {"retCode": 0, "result": {"category": "spot", "list": rows,
                                    "nextPageCursor": cursor}, **changes}


class Response:
    def __init__(self, payload, status=200):
        self.status_code = status
        self.payload = json.dumps(payload).encode()
        self.closed = False

    def iter_content(self, chunk_size):
        yield self.payload

    def close(self):
        self.closed = True


class Session:
    def __init__(self, payloads):
        self.responses = [Response(payload) for payload in payloads]
        self.calls = []
        self.trust_env = True

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses[len(self.calls) - 1]

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _placeholder_credentials(monkeypatch):
    monkeypatch.setenv("BYBIT_DEMO_API_KEY", "offline-placeholder-key")
    monkeypatch.setenv("BYBIT_DEMO_API_SECRET", "offline-placeholder-secret")


def reconcile(payloads):
    session = Session(payloads)
    return DemoRestReconciler(session=session).reconcile(CLIENT_ID), session


def test_filled_receipt_ignores_sdk_incompatible_string_smp_group():
    receipt, session = reconcile([page([order()]), page([fill()])])
    assert receipt.status == "filled"
    assert receipt.filled_quantity == Decimal("0.000118")
    assert receipt.filled_quote == Decimal("9.4400118")
    assert receipt.fee_btc == Decimal("0.000000118")
    assert receipt.fee_usdt == 0
    assert all(response.closed for response in session.responses)


def test_transport_is_fixed_demo_get_no_redirect_or_env_proxy(monkeypatch):
    monkeypatch.setattr("time.time_ns", lambda: 1234567890000000)
    _, session = reconcile([page([order()]), page([fill()])])
    assert session.trust_env is False
    for url, kwargs in session.calls:
        parsed = urlsplit(url)
        assert parsed.scheme == "https" and parsed.netloc == "api-demo.bybit.com"
        query = parse_qs(parsed.query)
        assert query["category"] == ["spot"]
        assert query["symbol"] == ["BTCUSDT"]
        assert query["orderLinkId"] == [CLIENT_ID]
        assert kwargs["allow_redirects"] is False
        headers = kwargs["headers"]
        material = headers["X-BAPI-TIMESTAMP"] + "offline-placeholder-key5000" + parsed.query
        expected = hmac.new(b"offline-placeholder-secret", material.encode(), hashlib.sha256).hexdigest()
        assert headers["X-BAPI-SIGN"] == expected
    execution_query = parse_qs(urlsplit(session.calls[1][0]).query)
    assert execution_query["orderId"] == ["venue-offline"]


def test_history_is_used_if_realtime_is_empty():
    receipt, session = reconcile([page([]), page([order()]), page([fill()])])
    assert receipt.status == "filled"
    assert urlsplit(session.calls[1][0]).path == "/v5/order/history"


def test_exact_order_execution_scope_and_cursor_recover_older_fill():
    first = fill(execId="first", execQty="0.000018", execFee="0.000000018")
    older = fill(execId="older", execQty="0.000100", execFee="0.000000100")
    receipt, session = reconcile([
        page([order()]), page([first] * 100, "older-page"), page([older]),
    ])
    assert receipt.status == "filled"
    assert receipt.filled_quantity == Decimal("0.000118")
    assert receipt.fee_btc == Decimal("0.000000118")
    query = parse_qs(urlsplit(session.calls[2][0]).query)
    assert query["cursor"] == ["older-page"]
    assert query["orderId"] == ["venue-offline"]


@pytest.mark.parametrize("row", [
    fill(execFee="-0.1"), fill(feeCurrency="ETH"), fill(execQty=0.000118),
    fill(execPrice="NaN"), fill(execQty="0.000117"), fill(symbol="ETHUSDT"),
    fill(orderId="wrong-order"), fill(orderLinkId="ta-demo-wrong-order"),
    {key: value for key, value in fill().items() if key != "feeCurrency"},
])
def test_uncertain_execution_evidence_has_zero_unknown_totals(row):
    receipt, _ = reconcile([page([order()]), page([row])])
    assert receipt.status == "unknown"
    assert receipt.filled_quantity == receipt.filled_quote == receipt.fee_btc == receipt.fee_usdt == 0


def test_missing_order_never_authorizes_retry():
    receipt, _ = reconcile([page([]), page([])])
    assert receipt.status == "unknown"


def test_redirect_and_api_error_are_unknown():
    session = Session([page([order()])])
    session.responses[0].status_code = 302
    assert DemoRestReconciler(session=session).reconcile(CLIENT_ID).status == "unknown"
    assert len(session.calls) == 1
    receipt, _ = reconcile([page([], retCode=10003)])
    assert receipt.status == "unknown"


def test_conflicting_duplicate_execution_is_unknown():
    receipt, _ = reconcile([page([order()]), page([fill(), fill(execFee="1")])])
    assert receipt.status == "unknown"


def test_pagination_is_bounded_and_repeated_cursor_is_unknown():
    receipt, session = reconcile([page([], "same"), page([], "same")])
    assert receipt.status == "unknown"
    assert len(session.calls) == 2
    receipt, session = reconcile([page([], str(i)) for i in range(10)])
    assert receipt.status == "unknown"
    assert len(session.calls) == 10


def test_partial_ioc_cancellation_preserves_authoritative_fills():
    receipt, _ = reconcile([page([order(orderStatus="PartiallyFilledCanceled", qty="0.001")]), page([fill()])])
    assert receipt.status == "canceled"
    assert receipt.filled_quantity == Decimal("0.000118")


def test_unexpected_url_or_effect_endpoint_never_reaches_transport():
    session = Session([])
    reconciler = DemoRestReconciler(session=session)
    for endpoint in ("/v5/order/create", "https://api.bybit.com/v5/order/realtime"):
        with pytest.raises(ValueError, match="read_boundary_violation"):
            reconciler._get(endpoint, {"category": "spot", "symbol": "BTCUSDT", "orderLinkId": CLIENT_ID})
    assert not session.calls


def test_exception_and_secret_text_do_not_escape_receipt():
    session = Session([])
    session.get = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("private-auth-material"))
    receipt = DemoRestReconciler(session=session).reconcile(CLIENT_ID)
    assert receipt.status == "unknown"
    assert "private-auth-material" not in receipt.model_dump_json()
