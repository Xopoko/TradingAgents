"""Deterministic demo spot sizing and a serializable, bot-owned sub-ledger.

The caller must atomically persist ``reserve_intent`` before submitting it. All
receipt amounts are cumulative venue totals, never incremental fill events.
An uncertain order remains reserved until a known terminal venue result arrives.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from hashlib import sha256
from typing import Literal

from pydantic import Field, model_validator

from tradingagents.agents.schemas import PortfolioRating

from .contracts import (
    AnalysisDecision,
    Contract,
    ExecutionReceipt,
    ManagedPortfolio,
    OrderIntent,
    SpotSnapshot,
)

ZERO = Decimal("0")
ONE = Decimal("1")
TERMINAL = frozenset({"filled", "canceled", "rejected"})
TOTAL_FIELDS = ("filled_quantity", "filled_quote", "fee_usdt", "fee_btc")
ALLOCATIONS = {
    PortfolioRating.BUY: Decimal("0.50"),
    PortfolioRating.OVERWEIGHT: Decimal("0.25"),
    PortfolioRating.UNDERWEIGHT: Decimal("0.10"),
    PortfolioRating.SELL: ZERO,
}


class PolicyBlocked(ValueError):
    """An unsafe or inconsistent input prevents order execution."""


class PolicyConfig(Contract):
    capital_cap_usdt: Decimal = Field(default=Decimal("300"), gt=0, allow_inf_nan=False)
    max_order_notional_usdt: Decimal = Field(default=Decimal("150"), gt=0, allow_inf_nan=False)
    fee_buffer: Decimal = Field(default=Decimal("0.002"), ge=0, lt=1, allow_inf_nan=False)
    slippage_bound: Decimal = Field(default=Decimal("0.002"), ge=0, lt=1, allow_inf_nan=False)
    snapshot_max_age_seconds: int = Field(default=120, gt=0, strict=True)
    decision_max_age_seconds: int = Field(default=3600, gt=0, strict=True)


class ManagedBook(ManagedPortfolio):
    """Virtual cash and BTC acquired by this bot only; never import account BTC.

    ``version`` describes the persisted schema; ``revision`` changes on updates.
    Order history intentionally survives terminal receipts and process restarts.
    """

    version: Literal[1] = 1
    revision: int = Field(default=0, ge=0, strict=True)
    cash_usdt: Decimal = Field(default=Decimal("300"), ge=0, allow_inf_nan=False)
    btc_quantity: Decimal = Field(default=ZERO, ge=0, allow_inf_nan=False)
    pending_intent: OrderIntent | None = None
    intent_history: dict[str, OrderIntent] = Field(default_factory=dict)
    applied_receipts: dict[str, ExecutionReceipt] = Field(default_factory=dict)
    processed_decision_ids: list[str] = Field(default_factory=list)
    processed_cycle_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_history(self):
        for key, intent in self.intent_history.items():
            if key != intent.client_order_id:
                raise ValueError("Intent history key mismatch")
        for key, receipt in self.applied_receipts.items():
            if key != receipt.client_order_id or key not in self.intent_history:
                raise ValueError("Receipt does not belong to a reserved intent")
        if self.pending_intent is not None:
            key = self.pending_intent.client_order_id
            if self.intent_history.get(key) != self.pending_intent:
                raise ValueError("Pending intent must be in history")
            receipt = self.applied_receipts.get(key)
            if receipt is not None and receipt.status in TERMINAL:
                raise ValueError("Terminal order cannot remain pending")
        for key in self.intent_history:
            receipt = self.applied_receipts.get(key)
            if (receipt is None or receipt.status not in TERMINAL) and (
                self.pending_intent is None or self.pending_intent.client_order_id != key
            ):
                raise ValueError("Unresolved historical intent must remain pending")
        return self


def _validated(model, value):
    try:
        return model.model_validate(value.model_dump())
    except (ValueError, AttributeError, TypeError) as exc:
        raise PolicyBlocked(f"Invalid {model.__name__}") from exc


def _updated(book: ManagedBook, **updates) -> ManagedBook:
    data = book.model_dump()
    data.update(updates, revision=book.revision + 1)
    return ManagedBook.model_validate(data)


def _step(value: Decimal, step: Decimal, rounding: str) -> Decimal:
    return (value / step).to_integral_value(rounding=rounding) * step


def _fresh(stamp: datetime, now: datetime, max_age: int, name: str) -> None:
    age = now - stamp
    if age < timedelta(0) or age > timedelta(seconds=max_age):
        raise PolicyBlocked(f"{name} is stale or in the future")


def client_order_id(decision_id: str, cycle_id: str) -> str:
    """A stable 36-character venue ID; duplicate cycles are also ledger-blocked."""
    digest = sha256(f"{len(cycle_id)}:{cycle_id}{decision_id}".encode()).hexdigest()[:28]
    return f"ta-demo-{digest}"


def plan_order(
    decision: AnalysisDecision,
    snapshot: SpotSnapshot,
    portfolio: ManagedPortfolio,
    now: datetime,
    cycle_id: str,
    config: PolicyConfig | None = None,
) -> OrderIntent | None:
    """Return a bounded LIMIT IOC intent, or None for Hold/duplicate/dust.

    Unsafe freshness, debt, unresolved orders, or balances raise PolicyBlocked.
    The managed portfolio is authoritative for ownership, the snapshot for funds
    available at the venue. Neither can increase the other's spending allowance.
    """
    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise PolicyBlocked("now must be timezone-aware")
    if not isinstance(cycle_id, str) or not cycle_id.strip():
        raise PolicyBlocked("cycle_id is required")
    decision = _validated(AnalysisDecision, decision)
    snapshot = _validated(SpotSnapshot, snapshot)
    config = _validated(PolicyConfig, config or PolicyConfig())
    book = _validated(ManagedBook if isinstance(portfolio, ManagedBook) else ManagedPortfolio, portfolio)
    if isinstance(book, ManagedBook):
        if book.pending_intent is not None:
            raise PolicyBlocked("Unresolved reserved order")
        if decision.decision_id in book.processed_decision_ids or cycle_id in book.processed_cycle_ids:
            return None
    for field in ("observed_at", "quote_at"):
        _fresh(getattr(snapshot, field), now, config.snapshot_max_age_seconds, field)
    for field in ("analyzed_at", "market_observed_at"):
        _fresh(getattr(decision, field), now, config.decision_max_age_seconds, field)
    if now >= decision.expires_at or decision.market_observed_at > decision.analyzed_at:
        raise PolicyBlocked("Decision validity/provenance window is invalid")
    if snapshot.btc_borrowed or snapshot.usdt_borrowed:
        raise PolicyBlocked("Borrowed balances are forbidden")
    if snapshot.open_order_ids:
        raise PolicyBlocked("Preexisting open orders")
    if book.cash_usdt > snapshot.usdt_available or book.btc_quantity > snapshot.btc_available:
        raise PolicyBlocked("Managed balances exceed actual available balances")
    if decision.rating == PortfolioRating.HOLD:
        return None

    mark = (snapshot.bid + snapshot.ask) / Decimal("2")
    equity = book.cash_usdt + book.btc_quantity * mark
    capital = min(equity, book.capital_limit_usdt, config.capital_cap_usdt)
    target_quantity = capital * ALLOCATIONS[decision.rating] / mark
    delta = target_quantity - book.btc_quantity
    if delta == ZERO:
        return None
    side = "Buy" if delta > ZERO else "Sell"
    if side == "Buy":
        price = _step(snapshot.ask * (ONE + config.slippage_bound), snapshot.price_step, ROUND_FLOOR)
        if price <= ZERO:
            return None
        cash = min(book.cash_usdt, snapshot.usdt_available)
        quantity = min(delta, cash / (price * (ONE + config.fee_buffer)))
    else:
        price = _step(snapshot.bid * (ONE - config.slippage_bound), snapshot.price_step, ROUND_CEILING)
        available = min(book.btc_quantity, snapshot.btc_available)
        quantity = min(-delta, available / (ONE + config.fee_buffer))
    quantity = min(quantity, config.max_order_notional_usdt / price)
    if snapshot.max_quantity is not None:
        quantity = min(quantity, snapshot.max_quantity)
    quantity = _step(quantity, snapshot.quantity_step, ROUND_FLOOR)
    notional = quantity * price
    if quantity <= ZERO or quantity < snapshot.min_quantity or notional < snapshot.min_notional:
        return None
    return OrderIntent(
        client_order_id=client_order_id(decision.decision_id, cycle_id),
        decision_id=decision.decision_id,
        side=side,
        quantity=quantity,
        limit_price=price,
        max_notional_usdt=notional,
        expires_at=min(
            decision.expires_at,
            snapshot.observed_at + timedelta(seconds=config.snapshot_max_age_seconds),
            snapshot.quote_at + timedelta(seconds=config.snapshot_max_age_seconds),
        ),
    )


def reserve_intent(book: ManagedBook, intent: OrderIntent, cycle_id: str) -> ManagedBook:
    """Return the reservation that MUST be durably written before submission."""
    book = _validated(ManagedBook, book)
    intent = _validated(OrderIntent, intent)
    if not isinstance(cycle_id, str) or not cycle_id.strip():
        raise PolicyBlocked("cycle_id is required")
    key = intent.client_order_id
    if book.pending_intent is not None:
        raise PolicyBlocked("An order is already pending")
    if (key in book.intent_history or intent.decision_id in book.processed_decision_ids
            or cycle_id in book.processed_cycle_ids):
        raise PolicyBlocked("Order, decision or cycle has already been reserved")
    if key != client_order_id(intent.decision_id, cycle_id):
        raise PolicyBlocked("Intent ID does not match decision/cycle")
    if intent.quantity * intent.limit_price > intent.max_notional_usdt:
        raise PolicyBlocked("Intent exceeds its notional bound")
    if (intent.side == "Buy" and intent.max_notional_usdt > book.cash_usdt
            or intent.side == "Sell" and intent.quantity > book.btc_quantity):
        raise PolicyBlocked("Intent exceeds managed funds")
    return _updated(
        book, pending_intent=intent,
        intent_history={**book.intent_history, key: intent},
        processed_decision_ids=[*book.processed_decision_ids, intent.decision_id],
        processed_cycle_ids=[*book.processed_cycle_ids, cycle_id],
    )


def apply_receipt(book: ManagedBook, receipt: ExecutionReceipt) -> ManagedBook:
    """Apply confirmed cumulative fills exactly once, retaining uncertain orders.

    A terminal canceled receipt may contain an IOC partial fill. The distinct
    partially_filled status does not prove terminality and retains the reserve.
    Unknown/not_found never authorize a retry, even after a process restart.
    """
    book = _validated(ManagedBook, book)
    receipt = _validated(ExecutionReceipt, receipt)
    key = receipt.client_order_id
    intent = book.intent_history.get(key)
    if intent is None:
        raise PolicyBlocked("Receipt has no reserved intent")
    previous = book.applied_receipts.get(key)
    if receipt.status in {"unknown", "not_found"}:
        if any(getattr(receipt, name) for name in TOTAL_FIELDS):
            raise PolicyBlocked("Uncertain receipt cannot establish fill totals")
        if previous is not None:
            # A lookup failure does not erase earlier confirmed cumulative fills.
            receipt = ExecutionReceipt.model_validate({
                **receipt.model_dump(),
                **{name: getattr(previous, name) for name in TOTAL_FIELDS},
            })
    totals = {name: getattr(receipt, name) for name in TOTAL_FIELDS}
    old = {name: getattr(previous, name) if previous else ZERO for name in TOTAL_FIELDS}
    if previous is not None and previous.status in TERMINAL:
        if totals != old:
            raise PolicyBlocked("Terminal receipt totals changed; manual reconciliation required")
        return book
    if any(totals[name] < old[name] for name in TOTAL_FIELDS):
        raise PolicyBlocked("Cumulative receipt totals regressed")
    if receipt.filled_quantity > intent.quantity:
        raise PolicyBlocked("Fill exceeds reserved quantity")
    if bool(receipt.filled_quantity) != bool(receipt.filled_quote):
        raise PolicyBlocked("Fill quantity and quote must both be confirmed")
    if receipt.filled_quantity == ZERO and (receipt.fee_btc or receipt.fee_usdt):
        raise PolicyBlocked("Fees without fills")
    if receipt.status == "filled" and receipt.filled_quantity != intent.quantity:
        raise PolicyBlocked("Filled status must confirm the whole reserved quantity")
    if receipt.filled_quantity:
        bound = receipt.filled_quantity * intent.limit_price
        if intent.side == "Buy" and receipt.filled_quote > bound:
            raise PolicyBlocked("Buy fill exceeds limit price")
        if intent.side == "Sell" and receipt.filled_quote < bound:
            raise PolicyBlocked("Sell fill is below limit price")
    delta = {name: totals[name] - old[name] for name in TOTAL_FIELDS}
    if intent.side == "Buy":
        cash = book.cash_usdt - delta["filled_quote"] - delta["fee_usdt"]
        btc = book.btc_quantity + delta["filled_quantity"] - delta["fee_btc"]
    else:
        cash = book.cash_usdt + delta["filled_quote"] - delta["fee_usdt"]
        btc = book.btc_quantity - delta["filled_quantity"] - delta["fee_btc"]
    if cash < ZERO or btc < ZERO:
        raise PolicyBlocked("Receipt would overdraw the managed ledger")
    if previous is not None and receipt == previous:
        return book
    pending = None if receipt.status in TERMINAL else intent
    return _updated(
        book, cash_usdt=cash, btc_quantity=btc,
        pending_intent=pending,
        applied_receipts={**book.applied_receipts, key: receipt},
    )
