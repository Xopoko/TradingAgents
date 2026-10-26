"""Read-only DEMO reconciliation for venue fields the pinned SDK cannot decode.

NautilusTrader 1.231.0 expects integer ``smpGroup``; actual Bybit V5 returns a
string. This module ignores irrelevant fields and strictly validates the order,
fills and fees it consumes. It cannot submit, cancel, borrow, transfer or retry
orders. Unknown results never constitute proof an order was not submitted.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any, Literal
from urllib.parse import urlencode

import requests
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from .contracts import ExecutionReceipt

DEMO_ORIGIN = "https://api-demo.bybit.com"
GET_PATHS = frozenset({"/v5/order/realtime", "/v5/order/history", "/v5/execution/list"})
MAX_PAGES = 10
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def _decimal_string(value: Any) -> Decimal:
    if not isinstance(value, str) or not value:
        raise ValueError("decimal_string_required")
    number = Decimal(value)
    if not number.is_finite():
        raise ValueError("finite_decimal_required")
    return number


Amount = Annotated[Decimal, BeforeValidator(_decimal_string), Field(ge=0, allow_inf_nan=False)]
PositiveAmount = Annotated[Decimal, BeforeValidator(_decimal_string), Field(gt=0, allow_inf_nan=False)]


class _VenueModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, populate_by_name=True)


class _Order(_VenueModel):
    symbol: Literal["BTCUSDT"]
    order_id: str = Field(alias="orderId", min_length=1, max_length=128)
    client_order_id: str = Field(alias="orderLinkId", min_length=1, max_length=36)
    status: Literal["Created", "New", "Untriggered", "Triggered", "PartiallyFilled",
                    "Filled", "Cancelled", "Canceled", "PartiallyFilledCanceled",
                    "Rejected", "Deactivated"] = Field(alias="orderStatus")
    quantity: PositiveAmount = Field(alias="qty")
    filled_quantity: Amount = Field(alias="cumExecQty")


class _Fill(_VenueModel):
    symbol: Literal["BTCUSDT"]
    order_id: str = Field(alias="orderId", min_length=1, max_length=128)
    client_order_id: str = Field(alias="orderLinkId", min_length=1, max_length=36)
    trade_id: str = Field(alias="execId", min_length=1, max_length=128)
    quantity: PositiveAmount = Field(alias="execQty")
    price: PositiveAmount = Field(alias="execPrice")
    fee: Amount = Field(alias="execFee")
    fee_currency: Literal["BTC", "USDT"] = Field(alias="feeCurrency")
    execution_type: Literal["Trade"] = Field(alias="execType")


class _Page(_VenueModel):
    category: Literal["spot"]
    rows: list[dict[str, Any]] = Field(alias="list", max_length=1000)
    cursor: str = Field(default="", alias="nextPageCursor", max_length=4096)


class _Envelope(_VenueModel):
    code: int = Field(alias="retCode", strict=True)
    result: _Page


def _unknown(client_order_id: str, code: str = "venue_state_unconfirmed") -> ExecutionReceipt:
    return ExecutionReceipt(client_order_id=client_order_id, status="unknown",
                            observed_at=datetime.now(UTC), safe_message=code)


class DemoRestReconciler:
    """Strictly scoped authenticated GET transport; secret values never leave it."""

    def __init__(self, *, session: requests.Session | None = None, timeout_seconds: int = 10):
        self._session = session if session is not None else requests.Session()
        self._session.trust_env = False
        self._timeout_seconds = timeout_seconds
        self._deadline = time.monotonic() + 35

    def close(self) -> None:
        self._session.close()

    def _get(self, path: str, params: dict[str, str]) -> _Page:
        if (path not in GET_PATHS or params.get("category") != "spot"
                or params.get("symbol") != "BTCUSDT"
                or not re.fullmatch(r"ta-demo-[A-Za-z0-9-]{1,28}", params.get("orderLinkId", ""))
                or not set(params) <= {"category", "symbol", "orderLinkId", "orderId", "limit", "cursor"}):
            raise ValueError("read_boundary_violation")
        key = os.environ.get("BYBIT_DEMO_API_KEY")
        secret = os.environ.get("BYBIT_DEMO_API_SECRET")
        if not key or not secret or key.startswith("op://") or secret.startswith("op://"):
            raise ValueError("demo_credentials_missing")
        query = urlencode(params)
        timestamp = str(time.time_ns() // 1_000_000)
        window = "5000"
        signature = hmac.new(secret.encode(), (timestamp + key + window + query).encode(),
                             hashlib.sha256).hexdigest()
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("reconciliation_deadline")
        response = self._session.get(
            DEMO_ORIGIN + path + "?" + query,
            headers={"X-BAPI-API-KEY": key, "X-BAPI-TIMESTAMP": timestamp,
                     "X-BAPI-RECV-WINDOW": window, "X-BAPI-SIGN": signature,
                     "X-BAPI-SIGN-TYPE": "2"},
            timeout=min(self._timeout_seconds, remaining), allow_redirects=False, stream=True,
        )
        try:
            if response.status_code != 200:
                raise ValueError("venue_http_unconfirmed")
            data = bytearray()
            for chunk in response.iter_content(chunk_size=65536):
                if time.monotonic() >= self._deadline:
                    raise ValueError("reconciliation_deadline")
                data.extend(chunk)
                if len(data) > MAX_RESPONSE_BYTES:
                    raise ValueError("venue_response_too_large")
            envelope = _Envelope.model_validate(json.loads(data))
            if envelope.code != 0:
                raise ValueError("venue_api_unconfirmed")
            return envelope.result
        finally:
            response.close()

    def _pages(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        cursor = ""
        seen: set[str] = set()
        for _ in range(MAX_PAGES):
            page = self._get(path, {**params, "limit": "100", **({"cursor": cursor} if cursor else {})})
            rows.extend(page.rows)
            if not page.cursor:
                return rows
            if page.cursor in seen:
                raise ValueError("repeated_page_cursor")
            seen.add(page.cursor)
            cursor = page.cursor
        raise ValueError("venue_pagination_limit")

    def reconcile(self, client_order_id: str) -> ExecutionReceipt:
        """Return only verified cumulative fills; all failures have zero unknown totals."""
        if not re.fullmatch(r"ta-demo-[A-Za-z0-9-]{1,28}", client_order_id):
            return _unknown(client_order_id, "invalid_client_order_id")
        try:
            params = {"category": "spot", "symbol": "BTCUSDT", "orderLinkId": client_order_id}
            orders: dict[str, _Order] = {}
            for path in ("/v5/order/realtime", "/v5/order/history"):
                for row in self._pages(path, params):
                    order = _Order.model_validate(row)
                    if order.client_order_id != client_order_id:
                        raise ValueError("order_identity_mismatch")
                    prior = orders.get(order.order_id)
                    if prior is not None and prior != order:
                        raise ValueError("order_report_conflict")
                    orders[order.order_id] = order
                if orders:
                    break
            if len(orders) != 1:
                return _unknown(client_order_id)
            order = next(iter(orders.values()))
            if order.filled_quantity > order.quantity:
                raise ValueError("fill_exceeds_order")
            if order.status == "Filled" and order.filled_quantity != order.quantity:
                raise ValueError("incomplete_filled_status")
            statuses = {"Filled": "filled", "Cancelled": "canceled", "Canceled": "canceled",
                        "PartiallyFilledCanceled": "canceled", "Deactivated": "canceled",
                        "Rejected": "rejected", "PartiallyFilled": "partially_filled"}
            status = statuses.get(order.status, "open")
            if status == "rejected" and order.filled_quantity > 0:
                status = "canceled"
            fills: dict[str, _Fill] = {}
            if order.filled_quantity:
                for row in self._pages("/v5/execution/list", {**params, "orderId": order.order_id}):
                    fill = _Fill.model_validate(row)
                    if fill.client_order_id != client_order_id or fill.order_id != order.order_id:
                        raise ValueError("execution_identity_mismatch")
                    prior = fills.get(fill.trade_id)
                    if prior is not None and prior != fill:
                        raise ValueError("execution_report_conflict")
                    fills[fill.trade_id] = fill
                if sum((fill.quantity for fill in fills.values()), Decimal(0)) != order.filled_quantity:
                    raise ValueError("execution_quantity_mismatch")
            return ExecutionReceipt(
                client_order_id=client_order_id, status=status, venue_order_id=order.order_id,
                filled_quantity=order.filled_quantity,
                filled_quote=sum((fill.quantity * fill.price for fill in fills.values()), Decimal(0)),
                fee_btc=sum((fill.fee for fill in fills.values() if fill.fee_currency == "BTC"), Decimal(0)),
                fee_usdt=sum((fill.fee for fill in fills.values() if fill.fee_currency == "USDT"), Decimal(0)),
                observed_at=datetime.now(UTC),
            )
        except Exception:
            return _unknown(client_order_id, "rest_reconciliation_unconfirmed")


def reconcile_demo_order(client_order_id: str) -> ExecutionReceipt:
    reconciler = DemoRestReconciler()
    try:
        return reconciler.reconcile(client_order_id)
    finally:
        reconciler.close()
