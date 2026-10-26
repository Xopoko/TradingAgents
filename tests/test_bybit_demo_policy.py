"""Offline regression tests for demo sizing and durable ownership accounting."""
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from tradingagents.integrations.bybit_demo.contracts import (
    AnalysisDecision,
    ExecutionReceipt,
    ManagedPortfolio,
    SpotSnapshot,
)
from tradingagents.integrations.bybit_demo.policy import (
    ManagedBook,
    PolicyBlocked,
    PolicyConfig,
    apply_receipt,
    plan_order,
    reserve_intent,
)

D = Decimal
NOW = datetime(2026, 10, 26, 12, tzinfo=UTC)
CYCLE = "2026-10-26T12"


def snapshot(**updates):
    fields = {
        "observed_at": NOW, "quote_at": NOW, "bid": "100000", "ask": "100000",
        "btc_available": "10", "usdt_available": "100000", "quantity_step": "0.000001",
        "price_step": "0.1", "min_quantity": "0.000001", "min_notional": "5",
    }
    fields.update(updates)
    return SpotSnapshot(**fields)


def decision(rating="Buy", **updates):
    fields = {
        "decision_id": "decision-one", "rating": rating, "analyzed_at": NOW,
        "market_observed_at": NOW, "expires_at": NOW + timedelta(minutes=15),
        "summary": "Typed opinion, not an execution instruction.",
    }
    fields.update(updates)
    return AnalysisDecision(**fields)


def order_and_book(book=None, rating="Buy", config=None):
    book = book or ManagedBook()
    intent = plan_order(decision(rating), snapshot(), book, NOW, CYCLE, config)
    assert intent is not None
    return intent, reserve_intent(book, intent, CYCLE)


def receipt(intent, status="filled", **updates):
    fields = {
        "client_order_id": intent.client_order_id, "status": status,
        "filled_quantity": intent.quantity, "filled_quote": intent.quantity * intent.limit_price,
        "observed_at": NOW,
    }
    fields.update(updates)
    return ExecutionReceipt(**fields)


def test_initial_book_does_not_adopt_demo_account_btc():
    book = ManagedBook()
    assert book.cash_usdt == D("300") and book.btc_quantity == 0
    assert plan_order(decision("Sell"), snapshot(), book, NOW, CYCLE) is None
    assert book == ManagedBook.model_validate_json(book.model_dump_json())


@pytest.mark.parametrize("rating, allocation", [("Buy", "0.5"), ("Overweight", "0.25"), ("Underweight", "0.1")])
def test_target_allocation_and_limit_order_bounds(rating, allocation):
    order = plan_order(decision(rating), snapshot(), ManagedBook(), NOW, CYCLE)
    assert order.side == "Buy"
    assert order.quantity * D("100000") <= D("300") * D(allocation)
    assert order.max_notional_usdt <= D("150")
    assert order.limit_price <= D("100000") * D("1.002")
    assert order.quantity % D("0.000001") == 0
    assert order.limit_price % D("0.1") == 0
    assert order.order_type == "LIMIT" and order.time_in_force == "IOC"
    assert order.is_leverage is False


def test_hold_and_already_at_target_never_order():
    assert plan_order(decision("Hold"), snapshot(), ManagedBook(), NOW, CYCLE) is None
    # Exactly 50% of 300 USDT equity is already held by this bot.
    at_target = ManagedBook(cash_usdt="150", btc_quantity="0.0015")
    assert plan_order(decision(), snapshot(), at_target, NOW, CYCLE) is None


def test_repeated_buy_after_fill_does_not_add_another_allocation():
    intent, book = order_and_book()
    book = apply_receipt(book, receipt(intent, fee_usdt="0.15"))
    next_decision = decision(decision_id="decision-two")
    # Remaining rounding/slippage/fee adjustment is below venue minimum notional.
    assert plan_order(next_decision, snapshot(), book, NOW, "next-cycle") is None
    assert book.cash_usdt < D("151")


def test_fees_are_reserved_with_tiny_cash_and_both_fee_currencies():
    config = PolicyConfig(capital_cap_usdt="1000", max_order_notional_usdt="1000")
    book = ManagedBook(cash_usdt="10", btc_quantity="0", capital_limit_usdt="1000")
    intent = plan_order(decision(), snapshot(min_notional="1"), book, NOW, CYCLE, config)
    assert intent.max_notional_usdt * D("1.002") <= book.cash_usdt
    reserved = reserve_intent(book, intent, CYCLE)
    filled = apply_receipt(reserved, receipt(intent, fee_usdt="0.001", fee_btc="0.000001"))
    assert filled.cash_usdt == book.cash_usdt - intent.max_notional_usdt - D("0.001")
    assert filled.btc_quantity == intent.quantity - D("0.000001")


def test_sell_uses_bot_btc_only_and_reserves_base_fee():
    initial = ManagedBook(cash_usdt="100", btc_quantity="0.001")
    intent, book = order_and_book(initial, "Sell")
    assert intent.side == "Sell"
    assert intent.quantity * D("1.002") <= initial.btc_quantity
    assert intent.quantity < snapshot().btc_available
    book = apply_receipt(book, receipt(intent, fee_btc="0.000001", fee_usdt="0.05"))
    assert book.btc_quantity == initial.btc_quantity - intent.quantity - D("0.000001")
    assert book.cash_usdt == initial.cash_usdt + intent.max_notional_usdt - D("0.05")


@pytest.mark.parametrize("changes", [
    {"observed_at": NOW - timedelta(seconds=121)},
    {"quote_at": NOW - timedelta(seconds=121)},
    {"quote_at": NOW + timedelta(microseconds=1)},
    {"btc_borrowed": "0.000001"}, {"usdt_borrowed": "0.01"},
    {"usdt_available": "299"}, {"open_order_ids": ["foreign-order"]},
])
def test_unsafe_snapshot_blocks(changes):
    with pytest.raises(PolicyBlocked):
        plan_order(decision(), snapshot(**changes), ManagedBook(), NOW, CYCLE)


def test_actual_btc_cannot_be_less_than_managed_btc():
    with pytest.raises(PolicyBlocked):
        plan_order(decision("Sell"), snapshot(btc_available="0"), ManagedBook(btc_quantity="0.001"), NOW, CYCLE)


@pytest.mark.parametrize("changes", [
    {"expires_at": NOW},
    {"analyzed_at": NOW + timedelta(seconds=1)},
    {"market_observed_at": NOW + timedelta(seconds=1)},
    {"market_observed_at": NOW - timedelta(seconds=3601)},
])
def test_stale_expired_or_future_decision_blocks(changes):
    if changes.get("expires_at") == NOW:
        changes["analyzed_at"] = NOW - timedelta(seconds=1)
        changes["market_observed_at"] = NOW - timedelta(seconds=1)
    with pytest.raises(PolicyBlocked):
        plan_order(decision(**changes), snapshot(), ManagedBook(), NOW, CYCLE)


def test_boundary_age_and_decimal_step_rounding():
    market = snapshot(quote_at=NOW - timedelta(seconds=120), price_step="3", quantity_step="0.000007")
    order = plan_order(decision(), market, ManagedBook(), NOW, CYCLE)
    assert order.quantity % D("0.000007") == 0
    assert order.limit_price % D("3") == 0
    assert order.limit_price <= market.ask * D("1.002")
    assert order.expires_at == NOW


def test_maximum_quantity_and_minimum_trade_filters():
    small = plan_order(decision(), snapshot(max_quantity="0.0001"), ManagedBook(), NOW, CYCLE)
    assert small.quantity == D("0.0001")
    assert plan_order(decision(), snapshot(min_quantity="0.1"), ManagedBook(), NOW, CYCLE) is None
    assert plan_order(decision(), snapshot(min_notional="151"), ManagedBook(), NOW, CYCLE) is None


def test_capital_cap_and_available_cash_prevent_overspend():
    book = ManagedBook(cash_usdt="10000", capital_limit_usdt="10000")
    order = plan_order(decision(), snapshot(), book, NOW, CYCLE)
    assert order.max_notional_usdt <= D("150")
    assert order.max_notional_usdt * D("1.002") <= book.cash_usdt


def test_intent_reservation_is_pure_durable_and_duplicate_safe():
    initial = ManagedBook()
    intent, reserved = order_and_book(initial)
    assert initial.pending_intent is None
    assert reserved.pending_intent == intent
    assert reserved.revision == 1
    restored = ManagedBook.model_validate_json(reserved.model_dump_json())
    with pytest.raises(PolicyBlocked):
        plan_order(decision(), snapshot(), restored, NOW, "another-cycle")
    finished = apply_receipt(restored, receipt(intent))
    assert plan_order(decision(), snapshot(), finished, NOW, "another-cycle") is None
    assert plan_order(decision(decision_id="fresh"), snapshot(), finished, NOW, CYCLE) is None
    with pytest.raises(PolicyBlocked):
        reserve_intent(finished, intent, CYCLE)
    assert plan_order(decision(), snapshot(), initial, NOW, CYCLE).client_order_id == intent.client_order_id


def test_duplicate_receipt_applies_fill_and_fees_exactly_once():
    intent, book = order_and_book()
    fill = receipt(intent, fee_usdt="0.1", fee_btc="0.000001")
    book = apply_receipt(book, fill)
    again = apply_receipt(ManagedBook.model_validate_json(book.model_dump_json()), fill)
    assert again == book
    assert again.revision == book.revision
    assert again.pending_intent is None


def test_cumulative_partial_fills_then_terminal_cancel():
    intent, reserved = order_and_book()
    qty = intent.quantity / D("2")
    partial = receipt(intent, "partially_filled", filled_quantity=qty,
                      filled_quote=qty * intent.limit_price, fee_usdt="0.01", fee_btc="0.000001")
    book = apply_receipt(reserved, partial)
    assert book.pending_intent == intent
    assert apply_receipt(book, partial) == book
    canceled = receipt(intent, "canceled", fee_usdt="0.02", fee_btc="0.000002")
    done = apply_receipt(book, canceled)
    assert done.pending_intent is None
    assert done.cash_usdt == D("300") - intent.max_notional_usdt - D("0.02")
    assert done.btc_quantity == intent.quantity - D("0.000002")


@pytest.mark.parametrize("status", ["unknown", "open", "not_found", "partially_filled"])
def test_unknown_and_nonterminal_states_remain_pending(status):
    intent, reserved = order_and_book()
    uncertain = receipt(intent, status, filled_quantity="0", filled_quote="0")
    book = apply_receipt(reserved, uncertain)
    assert book.pending_intent == intent and book.cash_usdt == D("300")
    with pytest.raises(PolicyBlocked):
        plan_order(decision(decision_id="fresh"), snapshot(), book, NOW, "next")
    not_found = receipt(intent, "not_found", filled_quantity="0", filled_quote="0")
    book = apply_receipt(book, not_found)
    assert book.pending_intent == intent


def test_lookup_failure_does_not_erase_confirmed_partial_fills():
    intent, reserved = order_and_book()
    partial = receipt(intent, "partially_filled", filled_quantity="0.0001", filled_quote="10.02")
    book = apply_receipt(reserved, partial)
    failed_lookup = receipt(intent, "not_found", filled_quantity="0", filled_quote="0")
    after = apply_receipt(book, failed_lookup)
    assert after.cash_usdt == book.cash_usdt and after.btc_quantity == book.btc_quantity
    assert after.pending_intent == intent
    done = apply_receipt(after, receipt(intent))
    assert done.cash_usdt == D("300") - intent.max_notional_usdt


@pytest.mark.parametrize("status", ["canceled", "rejected"])
def test_known_zero_terminal_clears_pending_without_fabricating_holdings(status):
    intent, reserved = order_and_book()
    done = apply_receipt(reserved, receipt(intent, status, filled_quantity="0", filled_quote="0"))
    assert done.pending_intent is None
    assert done.cash_usdt == D("300") and done.btc_quantity == 0


@pytest.mark.parametrize("changes", [
    {"filled_quantity": "1"}, {"filled_quote": "999"},
    {"filled_quote": "0"}, {"fee_usdt": "301"}, {"fee_btc": "1"},
])
def test_impossible_fill_blocks_without_mutating_book(changes):
    intent, reserved = order_and_book()
    with pytest.raises(PolicyBlocked):
        apply_receipt(reserved, receipt(intent, **changes))
    assert reserved.cash_usdt == D("300") and reserved.btc_quantity == 0


def test_receipt_for_unreserved_order_and_regressing_totals_fail_closed():
    intent, reserved = order_and_book()
    with pytest.raises(PolicyBlocked):
        apply_receipt(ManagedBook(), receipt(intent))
    partial = receipt(intent, "partially_filled", filled_quantity="0.0001", filled_quote="10.02")
    book = apply_receipt(reserved, partial)
    with pytest.raises(PolicyBlocked):
        apply_receipt(book, receipt(intent, "canceled", filled_quantity="0", filled_quote="0"))


def test_monetary_floats_and_bypassed_validation_are_rejected():
    with pytest.raises(ValidationError):
        PolicyConfig(fee_buffer=0.002)
    with pytest.raises(ValidationError):
        ManagedBook(cash_usdt=300.0)
    invalid = ManagedPortfolio.model_construct(cash_usdt=D("-1"), btc_quantity=D("0"), capital_limit_usdt=D("300"))
    with pytest.raises(PolicyBlocked):
        plan_order(decision(), snapshot(), invalid, NOW, CYCLE)


def test_invalid_persisted_unresolved_intent_cannot_drop_pending():
    _, book = order_and_book()
    broken = book.model_dump()
    broken["pending_intent"] = None
    with pytest.raises(ValidationError):
        ManagedBook.model_validate(broken)
