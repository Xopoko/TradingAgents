"""Deterministic integer-unit policy and ownership ledger for one local fork.

USDT has six decimals and WBTC eight. ETH pays gas separately: the fixed native
reserve is 0.005 ETH, never converted to an invented dollar amount. The worker
must also enforce its transaction gas budget. Persist binding and reservations
before sending any transaction; unknown effects are reconciled, never retried.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from decimal import ROUND_FLOOR, Decimal, localcontext
from typing import Literal

from pydantic import Field, model_validator

from tradingagents.agents.schemas import PortfolioRating

from .contracts import (
    AnalysisDecision,
    Contract,
    ExecutionReceipt,
    ForkIdentity,
    ForkSnapshot,
    ManagedPortfolio,
    QuoteRequest,
    SwapIntent,
    SwapQuote,
    Units,
)
from .profile import BTC_SCALE, MAX_ORDER_USDT, MAX_SLIPPAGE_BPS, ROUTER, USDT, USDT_SCALE, WBTC

MIN_NATIVE_WEI = 5_000_000_000_000_000
TERMINAL = frozenset({"confirmed", "reverted"})
ALLOCATIONS = {
    PortfolioRating.BUY: Decimal("0.50"),
    PortfolioRating.OVERWEIGHT: Decimal("0.25"),
    PortfolioRating.UNDERWEIGHT: Decimal("0.10"),
    PortfolioRating.SELL: Decimal("0"),
}


class PolicyBlocked(ValueError):
    """A safe reason code for an invalid or unsafe policy boundary."""


class PolicyConfig(Contract):
    capital_cap_usdt: Decimal = Field(default=Decimal("300"), gt=0, le=300, allow_inf_nan=False)
    max_order_usdt: Decimal = Field(default=MAX_ORDER_USDT, gt=0, le=150, allow_inf_nan=False)
    slippage_bps: int = Field(default=MAX_SLIPPAGE_BPS, ge=1, le=50, strict=True)
    min_native_wei: int = Field(default=MIN_NATIVE_WEI, ge=MIN_NATIVE_WEI, le=2**256 - 1, strict=True)
    snapshot_max_age_seconds: int = Field(default=120, gt=0, le=120, strict=True)
    decision_max_age_seconds: int = Field(default=3600, gt=0, le=3600, strict=True)
    quote_max_age_seconds: int = Field(default=120, gt=0, le=120, strict=True)


def _identity_data(identity: ForkIdentity) -> dict:
    data = identity.model_dump()
    data["wallet_address"] = identity.wallet_address.lower()
    data["fork_block_hash"] = identity.fork_block_hash.lower()
    return data


def _same_identity(left: ForkIdentity, right: ForkIdentity) -> bool:
    return _identity_data(left) == _identity_data(right)


def intent_id(decision_id: str, cycle_id: str, identity: ForkIdentity) -> str:
    payload = {"decision_id": decision_id, "cycle_id": cycle_id, "identity": _identity_data(identity)}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"ta-fork-{digest[:32]}"


def _quote_profile(quote: SwapQuote) -> None:
    expected_in, expected_out = (USDT, WBTC) if quote.side == "Buy" else (WBTC, USDT)
    if (quote.token_in.lower() != expected_in or quote.token_out.lower() != expected_out
            or quote.router.lower() != ROUTER):
        raise PolicyBlocked("quote_profile_mismatch")


def _receipt_bounds(intent: SwapIntent, receipt: ExecutionReceipt) -> None:
    if not _same_identity(intent.quote.identity, receipt.identity):
        raise PolicyBlocked("receipt_identity_mismatch")
    if receipt.status == "reverted" and (receipt.amount_in_units or receipt.amount_out_units):
        raise PolicyBlocked("reverted_swap_has_token_amounts")
    if receipt.status == "confirmed":
        if receipt.amount_in_units != intent.quote.amount_in_units:
            raise PolicyBlocked("confirmed_input_mismatch")
        if receipt.amount_out_units < intent.quote.minimum_out_units:
            raise PolicyBlocked("confirmed_output_below_minimum")
        if receipt.confirmed_block_number < intent.quote.quoted_block_number:
            raise PolicyBlocked("receipt_block_precedes_quote")
    if receipt.gas_paid_wei and not receipt.transaction_hashes:
        raise PolicyBlocked("gas_requires_transaction_evidence")


class ManagedBook(ManagedPortfolio):
    """Cash and WBTC owned by the bot, bound permanently to one fork instance."""

    version: Literal[1] = 1
    revision: int = Field(default=0, ge=0, strict=True)
    identity: ForkIdentity | None = None
    pending_intent: SwapIntent | None = None
    intent_history: dict[str, SwapIntent] = Field(default_factory=dict)
    applied_receipts: dict[str, ExecutionReceipt] = Field(default_factory=dict)
    processed_cycle_ids: list[str] = Field(default_factory=list)
    processed_decision_ids: list[str] = Field(default_factory=list)
    gas_paid_wei: Units = 0

    @model_validator(mode="after")
    def consistent_history(self):
        if self.identity is None and (self.pending_intent or self.intent_history or self.applied_receipts):
            raise ValueError("unbound_book_has_orders")
        for key, intent in self.intent_history.items():
            _quote_profile(intent.quote)
            if key != intent.intent_id or key != intent_id(intent.decision_id, intent.cycle_id, intent.quote.identity):
                raise ValueError("invalid_intent_history_key")
            if not _same_identity(self.identity, intent.quote.identity):
                raise ValueError("historical_identity_mismatch")
            if intent.cycle_id not in self.processed_cycle_ids or intent.decision_id not in self.processed_decision_ids:
                raise ValueError("historical_cycle_not_reserved")
        for key, receipt in self.applied_receipts.items():
            if key != receipt.intent_id or key not in self.intent_history:
                raise ValueError("receipt_without_reserved_intent")
            _receipt_bounds(self.intent_history[key], receipt)
        if self.gas_paid_wei != sum(receipt.gas_paid_wei for receipt in self.applied_receipts.values()):
            raise ValueError("gas_ledger_mismatch")
        if self.pending_intent is not None:
            if self.intent_history.get(self.pending_intent.intent_id) != self.pending_intent:
                raise ValueError("pending_intent_missing_from_history")
            receipt = self.applied_receipts.get(self.pending_intent.intent_id)
            if receipt is not None and receipt.status in TERMINAL:
                raise ValueError("terminal_intent_still_pending")
        for key in self.intent_history:
            receipt = self.applied_receipts.get(key)
            if ((receipt is None or receipt.status not in TERMINAL)
                    and (self.pending_intent is None or self.pending_intent.intent_id != key)):
                raise ValueError("unresolved_intent_not_pending")
        return self


def _validated(model, value):
    try:
        return model.model_validate(value.model_dump())
    except (ValueError, TypeError, AttributeError) as exc:
        raise PolicyBlocked(f"invalid_{model.__name__}") from exc


def _updated(book: ManagedBook, **updates) -> ManagedBook:
    return ManagedBook.model_validate({**book.model_dump(), **updates, "revision": book.revision + 1})


def _aware(now: datetime) -> None:
    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise PolicyBlocked("timezone_aware_now_required")


def _fresh(stamp: datetime, now: datetime, seconds: int, name: str) -> None:
    if not timedelta(0) <= now - stamp <= timedelta(seconds=seconds):
        raise PolicyBlocked(f"{name}_not_fresh")


def _snapshot_bounds(snapshot: ForkSnapshot, book: ManagedPortfolio) -> None:
    if snapshot.block_number < snapshot.identity.fork_block_number:
        raise PolicyBlocked("snapshot_precedes_fork")
    if (snapshot.block_number == snapshot.identity.fork_block_number
            and snapshot.block_hash.lower() != snapshot.identity.fork_block_hash.lower()):
        raise PolicyBlocked("snapshot_fork_hash_mismatch")
    if book.cash_units > snapshot.cash_units or book.btc_units > snapshot.btc_units:
        raise PolicyBlocked("managed_funds_unavailable")
    if isinstance(book, ManagedBook):
        if book.identity is None:
            raise PolicyBlocked("book_identity_not_bound")
        if not _same_identity(book.identity, snapshot.identity):
            raise PolicyBlocked("fork_identity_changed")


def bind_book(book: ManagedBook, snapshot: ForkSnapshot) -> ManagedBook:
    """Bind the initial ledger without adopting any additional wallet tokens.

    Binding never resets cash, imports existing wallet WBTC, or accepts a new
    fork instance after a restart. The caller must persist the returned book.
    """
    book = _validated(ManagedBook, book)
    snapshot = _validated(ForkSnapshot, snapshot)
    bound = _updated(book, identity=snapshot.identity) if book.identity is None else book
    _snapshot_bounds(snapshot, bound)
    return bound


def _inputs(decision, snapshot, book, now, cycle_id, config):
    _aware(now)
    if not isinstance(cycle_id, str) or not cycle_id.strip() or len(cycle_id) > 128:
        raise PolicyBlocked("invalid_cycle_id")
    decision = _validated(AnalysisDecision, decision)
    snapshot = _validated(ForkSnapshot, snapshot)
    book = _validated(ManagedBook if isinstance(book, ManagedBook) else ManagedPortfolio, book)
    config = _validated(PolicyConfig, config or PolicyConfig())
    _snapshot_bounds(snapshot, book)
    _fresh(snapshot.observed_at, now, config.snapshot_max_age_seconds, "snapshot")
    _fresh(decision.analyzed_at, now, config.decision_max_age_seconds, "decision")
    if now >= decision.expires_at or decision.expires_at - decision.analyzed_at > timedelta(minutes=15):
        raise PolicyBlocked("invalid_decision_expiry")
    if snapshot.native_wei < config.min_native_wei:
        raise PolicyBlocked("native_gas_reserve_unavailable")
    if isinstance(book, ManagedBook) and book.pending_intent is not None:
        raise PolicyBlocked("unresolved_intent")
    return decision, snapshot, book, config


def plan_quote_request(
    decision: AnalysisDecision, snapshot: ForkSnapshot, book: ManagedPortfolio,
    now: datetime, cycle_id: str, config: PolicyConfig | None = None,
) -> QuoteRequest | None:
    """Size input tokens in base units; None means Hold, duplicate, or dust."""
    decision, snapshot, book, config = _inputs(decision, snapshot, book, now, cycle_id, config)
    if (isinstance(book, ManagedBook)
            and (cycle_id in book.processed_cycle_ids or decision.decision_id in book.processed_decision_ids)):
        return None
    if decision.rating == PortfolioRating.HOLD:
        return None
    # uint256-scale balances need more precision than Decimal's default 28.
    with localcontext() as context:
        context.prec = 100
        cash = Decimal(book.cash_units) / USDT_SCALE
        btc_value = Decimal(book.btc_units) / BTC_SCALE * snapshot.btc_price_usdt
        capital = min(cash + btc_value, book.capital_limit_usdt, config.capital_cap_usdt)
        delta = capital * ALLOCATIONS[decision.rating] - btc_value
        if delta > 0:
            side = "Buy"
            amount = min(delta, config.max_order_usdt, cash) * USDT_SCALE
        elif delta < 0:
            side = "Sell"
            if decision.rating == PortfolioRating.SELL and btc_value <= config.max_order_usdt:
                # The target is exactly zero: use the known token balance,
                # avoiding a value/price round trip which can lose one unit.
                amount = Decimal(book.btc_units)
            else:
                amount = min(-delta, config.max_order_usdt) / snapshot.btc_price_usdt * BTC_SCALE
            amount = min(amount, Decimal(book.btc_units), Decimal(snapshot.btc_units))
        else:
            return None
        units = int(amount.to_integral_value(rounding=ROUND_FLOOR))
    if units <= 0:
        return None
    return QuoteRequest(side=side, amount_in_units=units, slippage_bps=config.slippage_bps)


def build_intent(
    decision: AnalysisDecision, snapshot: ForkSnapshot, book: ManagedPortfolio,
    quote: SwapQuote, now: datetime, cycle_id: str, config: PolicyConfig | None = None,
) -> SwapIntent:
    """Bind an exact requested amount to a fresh, fixed-profile quote."""
    decision, snapshot, book, config = _inputs(decision, snapshot, book, now, cycle_id, config)
    quote = _validated(SwapQuote, quote)
    request = plan_quote_request(decision, snapshot, book, now, cycle_id, config)
    if request is None:
        raise PolicyBlocked("no_swap_required")
    if not _same_identity(snapshot.identity, quote.identity):
        raise PolicyBlocked("quote_identity_mismatch")
    _quote_profile(quote)
    if (quote.side != request.side or quote.amount_in_units != request.amount_in_units
            or quote.slippage_bps != request.slippage_bps):
        raise PolicyBlocked("quote_request_mismatch")
    if (quote.quoted_block_number != snapshot.block_number
            or quote.quoted_block_hash.lower() != snapshot.block_hash.lower()):
        raise PolicyBlocked("quote_block_mismatch")
    _fresh(quote.quoted_at, now, config.quote_max_age_seconds, "quote")
    if quote.expires_at - quote.quoted_at > timedelta(seconds=config.quote_max_age_seconds):
        raise PolicyBlocked("quote_ttl_exceeded")
    minimum = (quote.expected_out_units * (10000 - request.slippage_bps) + 9999) // 10000
    if quote.minimum_out_units < minimum:
        raise PolicyBlocked("quote_slippage_exceeded")
    expiry = min(quote.expires_at, decision.expires_at,
                 snapshot.observed_at + timedelta(seconds=config.snapshot_max_age_seconds))
    if expiry <= now:
        raise PolicyBlocked("quote_expired")
    return SwapIntent(
        intent_id=intent_id(decision.decision_id, cycle_id, snapshot.identity),
        decision_id=decision.decision_id, cycle_id=cycle_id, quote=quote,
        created_at=now, expires_at=expiry,
    )


def reserve_intent(book: ManagedBook, intent: SwapIntent) -> ManagedBook:
    """Return a new reservation which the caller must fsync before execution."""
    book = _validated(ManagedBook, book)
    intent = _validated(SwapIntent, intent)
    _quote_profile(intent.quote)
    if book.identity is None or not _same_identity(book.identity, intent.quote.identity):
        raise PolicyBlocked("intent_identity_mismatch")
    if book.pending_intent is not None:
        raise PolicyBlocked("unresolved_intent")
    if (intent.intent_id in book.intent_history or intent.decision_id in book.processed_decision_ids
            or intent.cycle_id in book.processed_cycle_ids):
        raise PolicyBlocked("intent_already_reserved")
    if intent.intent_id != intent_id(intent.decision_id, intent.cycle_id, book.identity):
        raise PolicyBlocked("intent_id_mismatch")
    available = book.cash_units if intent.quote.side == "Buy" else book.btc_units
    if intent.quote.amount_in_units > available:
        raise PolicyBlocked("intent_exceeds_managed_funds")
    return _updated(
        book, pending_intent=intent,
        intent_history={**book.intent_history, intent.intent_id: intent},
        processed_cycle_ids=[*book.processed_cycle_ids, intent.cycle_id],
        processed_decision_ids=[*book.processed_decision_ids, intent.decision_id],
    )


def apply_receipt(book: ManagedBook, receipt: ExecutionReceipt) -> ManagedBook:
    """Apply integer token deltas and cumulative native gas exactly once."""
    book = _validated(ManagedBook, book)
    receipt = _validated(ExecutionReceipt, receipt)
    intent = book.intent_history.get(receipt.intent_id)
    if intent is None:
        raise PolicyBlocked("receipt_without_reserved_intent")
    _receipt_bounds(intent, receipt)
    previous = book.applied_receipts.get(receipt.intent_id)
    if previous is not None and previous.status in TERMINAL:
        immutable = ("status", "amount_in_units", "amount_out_units", "gas_paid_wei",
                     "transaction_hashes", "confirmed_block_number", "confirmed_block_hash")
        if any(getattr(previous, name) != getattr(receipt, name) for name in immutable):
            raise PolicyBlocked("terminal_receipt_changed")
        return book
    previous_gas = previous.gas_paid_wei if previous is not None else 0
    if receipt.gas_paid_wei < previous_gas:
        raise PolicyBlocked("cumulative_gas_regressed")
    if previous is not None and not set(previous.transaction_hashes).issubset(receipt.transaction_hashes):
        raise PolicyBlocked("transaction_history_regressed")
    cash, btc = book.cash_units, book.btc_units
    if receipt.status == "confirmed":
        if intent.quote.side == "Buy":
            cash -= receipt.amount_in_units
            btc += receipt.amount_out_units
        else:
            btc -= receipt.amount_in_units
            cash += receipt.amount_out_units
    if cash < 0 or btc < 0:
        raise PolicyBlocked("receipt_overdraws_managed_funds")
    if previous is not None and previous == receipt:
        return book
    return _updated(
        book, cash_units=cash, btc_units=btc,
        gas_paid_wei=book.gas_paid_wei + receipt.gas_paid_wei - previous_gas,
        applied_receipts={**book.applied_receipts, receipt.intent_id: receipt},
        pending_intent=None if receipt.status in TERMINAL else intent,
    )
