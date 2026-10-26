"""Validated process boundaries. Decimal amounts serialize as JSON strings."""
from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from tradingagents.agents.schemas import PortfolioRating


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("*", mode="before")
    @classmethod
    def reject_float(cls, value):
        if isinstance(value, float):
            raise ValueError("Use decimal strings for monetary values")
        return value


class SpotSnapshot(Contract):
    environment: Literal["demo"] = "demo"
    symbol: Literal["BTCUSDT"] = "BTCUSDT"
    observed_at: AwareDatetime
    quote_at: AwareDatetime
    bid: Decimal = Field(gt=0, allow_inf_nan=False)
    ask: Decimal = Field(gt=0, allow_inf_nan=False)
    btc_available: Decimal = Field(ge=0, allow_inf_nan=False)
    usdt_available: Decimal = Field(ge=0, allow_inf_nan=False)
    btc_borrowed: Decimal = Field(default=Decimal("0"), ge=0, allow_inf_nan=False)
    usdt_borrowed: Decimal = Field(default=Decimal("0"), ge=0, allow_inf_nan=False)
    quantity_step: Decimal = Field(gt=0, allow_inf_nan=False)
    price_step: Decimal = Field(gt=0, allow_inf_nan=False)
    min_quantity: Decimal = Field(ge=0, allow_inf_nan=False)
    min_notional: Decimal = Field(gt=0, allow_inf_nan=False)
    max_quantity: Decimal | None = Field(default=None, gt=0, allow_inf_nan=False)
    open_order_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_spread(self):
        if self.ask < self.bid:
            raise ValueError("Crossed bid/ask")
        return self


class ManagedPortfolio(Contract):
    cash_usdt: Decimal = Field(ge=0, allow_inf_nan=False)
    btc_quantity: Decimal = Field(ge=0, allow_inf_nan=False)
    capital_limit_usdt: Decimal = Field(default=Decimal("300"), gt=0, allow_inf_nan=False)


class AnalysisDecision(Contract):
    decision_id: str = Field(min_length=1, max_length=100)
    symbol: Literal["BTCUSDT"] = "BTCUSDT"
    rating: PortfolioRating
    analyzed_at: AwareDatetime
    market_observed_at: AwareDatetime
    expires_at: AwareDatetime
    summary: str = Field(min_length=1, max_length=16000)
    source: Literal["structured"] = "structured"
    report_path: str | None = None

    @model_validator(mode="after")
    def valid_time_window(self):
        if self.expires_at <= self.analyzed_at:
            raise ValueError("Decision must have a positive validity window")
        return self


class OrderIntent(Contract):
    environment: Literal["demo"] = "demo"
    symbol: Literal["BTCUSDT"] = "BTCUSDT"
    client_order_id: str = Field(pattern=r"^ta-demo-[a-zA-Z0-9-]{1,28}$", max_length=36)
    decision_id: str = Field(min_length=1, max_length=100)
    side: Literal["Buy", "Sell"]
    quantity: Decimal = Field(gt=0, allow_inf_nan=False)
    limit_price: Decimal = Field(gt=0, allow_inf_nan=False)
    max_notional_usdt: Decimal = Field(gt=0, allow_inf_nan=False)
    expires_at: AwareDatetime
    order_type: Literal["LIMIT"] = "LIMIT"
    time_in_force: Literal["IOC"] = "IOC"
    is_leverage: Literal[False] = False


class ExecutionReceipt(Contract):
    environment: Literal["demo"] = "demo"
    symbol: Literal["BTCUSDT"] = "BTCUSDT"
    client_order_id: str
    status: Literal["filled", "partially_filled", "canceled", "rejected", "not_found", "unknown", "open"]
    venue_order_id: str | None = None
    filled_quantity: Decimal = Field(default=Decimal("0"), ge=0, allow_inf_nan=False)
    filled_quote: Decimal = Field(default=Decimal("0"), ge=0, allow_inf_nan=False)
    fee_usdt: Decimal = Field(default=Decimal("0"), ge=0, allow_inf_nan=False)
    fee_btc: Decimal = Field(default=Decimal("0"), ge=0, allow_inf_nan=False)
    observed_at: AwareDatetime
    safe_message: str | None = Field(default=None, max_length=500)


class GatewayRequest(Contract):
    operation: Literal["snapshot", "execute", "reconcile"]
    intent: OrderIntent | None = None
    client_order_id: str | None = None

    @model_validator(mode="after")
    def fields_match_operation(self):
        if (self.operation == "execute") != (self.intent is not None):
            raise ValueError("Only execute requires an intent")
        if (self.operation == "reconcile") != (self.client_order_id is not None):
            raise ValueError("Only reconcile requires a client order ID")
        return self


class GatewayResponse(Contract):
    ok: bool
    snapshot: SpotSnapshot | None = None
    receipt: ExecutionReceipt | None = None
    error_code: str | None = None

    @model_validator(mode="after")
    def valid_envelope(self):
        results = int(self.snapshot is not None) + int(self.receipt is not None)
        if self.ok and (results != 1 or self.error_code is not None):
            raise ValueError("Success requires exactly one result")
        if not self.ok and (results != 0 or not self.error_code):
            raise ValueError("Failure requires only an error code")
        return self
