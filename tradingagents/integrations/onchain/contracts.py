"""Process and persistence contracts for an isolated Ethereum fork.

Token amounts are integer base units. Money used for valuation is a decimal
string. No model accepts a signing key or a mainnet execution mode.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from tradingagents.agents.schemas import PortfolioRating

Address = Annotated[str, StringConstraints(pattern=r"^0x[0-9a-fA-F]{40}$")]
Hash = Annotated[str, StringConstraints(pattern=r"^0x[0-9a-fA-F]{64}$")]
Units = Annotated[int, Field(ge=0, le=2**256 - 1, strict=True)]
PositiveUnits = Annotated[int, Field(gt=0, le=2**256 - 1, strict=True)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("*", mode="before")
    @classmethod
    def reject_float(cls, value):
        if isinstance(value, float):
            raise ValueError("floating_point_forbidden")
        return value


class ForkIdentity(Contract):
    environment: Literal["local-fork"] = "local-fork"
    profile: Literal["ethereum-wbtc-usdt"] = "ethereum-wbtc-usdt"
    chain_id: Literal[31337] = 31337
    source_chain_id: Literal[1] = 1
    instance_id: str = Field(min_length=1, max_length=256)
    fork_block_number: int = Field(ge=1, strict=True)
    fork_block_hash: Hash
    wallet_address: Address


class ForkSnapshot(Contract):
    identity: ForkIdentity
    observed_at: AwareDatetime
    block_number: int = Field(ge=1, strict=True)
    block_hash: Hash
    cash_units: Units
    btc_units: Units
    native_wei: Units
    btc_price_usdt: Decimal = Field(gt=0, allow_inf_nan=False)


class QuoteRequest(Contract):
    side: Literal["Buy", "Sell"]
    amount_in_units: PositiveUnits
    slippage_bps: int = Field(default=50, ge=1, le=50, strict=True)


class SwapQuote(Contract):
    identity: ForkIdentity
    quote_id: str = Field(min_length=1, max_length=128)
    side: Literal["Buy", "Sell"]
    token_in: Address
    token_out: Address
    router: Address
    amount_in_units: PositiveUnits
    expected_out_units: PositiveUnits
    minimum_out_units: PositiveUnits
    slippage_bps: int = Field(ge=1, le=50, strict=True)
    pool_fee: Literal[100, 500, 3000, 10000]
    quoted_block_number: int = Field(ge=1, strict=True)
    quoted_block_hash: Hash
    quoted_at: AwareDatetime
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def bounds(self):
        if self.minimum_out_units > self.expected_out_units or self.expires_at <= self.quoted_at:
            raise ValueError("invalid_quote_bounds")
        if self.minimum_out_units * 10000 < self.expected_out_units * (10000 - self.slippage_bps):
            raise ValueError("slippage_bound_exceeded")
        return self


class AnalysisDecision(Contract):
    decision_id: str = Field(min_length=1, max_length=128)
    rating: PortfolioRating
    analyzed_at: AwareDatetime
    expires_at: AwareDatetime
    summary: str = Field(min_length=1, max_length=16000)
    source: Literal["structured", "fixture"] = "structured"

    @model_validator(mode="after")
    def window(self):
        if self.expires_at <= self.analyzed_at:
            raise ValueError("invalid_decision_window")
        return self


class ManagedPortfolio(Contract):
    cash_units: Units = 300_000_000
    btc_units: Units = 0
    capital_limit_usdt: Decimal = Field(default=Decimal("300"), gt=0, le=300, allow_inf_nan=False)


class SwapIntent(Contract):
    intent_id: str = Field(pattern=r"^ta-fork-[a-f0-9]{32}$")
    decision_id: str = Field(min_length=1, max_length=128)
    cycle_id: str = Field(min_length=1, max_length=128)
    quote: SwapQuote
    created_at: AwareDatetime
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def window(self):
        if self.expires_at > self.quote.expires_at or self.expires_at <= self.created_at:
            raise ValueError("invalid_intent_window")
        return self


class ExecutionReceipt(Contract):
    environment: Literal["local-fork"] = "local-fork"
    intent_id: str = Field(pattern=r"^ta-fork-[a-f0-9]{32}$")
    identity: ForkIdentity
    status: Literal["confirmed", "reverted", "pending", "unknown"]
    transaction_hashes: list[Hash] = Field(default_factory=list, max_length=8)
    amount_in_units: Units = 0
    amount_out_units: Units = 0
    gas_paid_wei: Units = 0
    confirmed_block_number: int | None = Field(default=None, ge=1, strict=True)
    confirmed_block_hash: Hash | None = None
    observed_at: AwareDatetime
    safe_error_code: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def terminal_evidence(self):
        if self.status == "confirmed" and (
            not self.transaction_hashes or not self.amount_in_units or not self.amount_out_units
            or self.confirmed_block_number is None or self.confirmed_block_hash is None
        ):
            raise ValueError("confirmed_execution_requires_evidence")
        if self.status in {"pending", "unknown"} and (self.amount_in_units or self.amount_out_units):
            raise ValueError("uncertain_execution_cannot_establish_amounts")
        return self


class GatewayRequest(Contract):
    operation: Literal["snapshot", "quote", "execute", "reconcile"]
    quote_request: QuoteRequest | None = None
    intent: SwapIntent | None = None
    intent_id: str | None = None

    @model_validator(mode="after")
    def operation_fields(self):
        if (self.operation == "quote") != (self.quote_request is not None):
            raise ValueError("quote_request_mismatch")
        if (self.operation == "execute") != (self.intent is not None):
            raise ValueError("intent_mismatch")
        if (self.operation == "reconcile") != (self.intent_id is not None):
            raise ValueError("intent_id_mismatch")
        return self


class GatewayResponse(Contract):
    ok: bool = Field(strict=True)
    snapshot: ForkSnapshot | None = None
    quote: SwapQuote | None = None
    receipt: ExecutionReceipt | None = None
    safe_error_code: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def envelope(self):
        count = sum(x is not None for x in (self.snapshot, self.quote, self.receipt))
        if self.ok and (count != 1 or self.safe_error_code is not None):
            raise ValueError("invalid_success_envelope")
        if not self.ok and (count or not self.safe_error_code):
            raise ValueError("invalid_error_envelope")
        return self
