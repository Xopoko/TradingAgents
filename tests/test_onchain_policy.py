"""Offline integer accounting, identity, and quote-boundary regression tests."""
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from tradingagents.integrations.onchain.contracts import (
    AnalysisDecision,
    ExecutionReceipt,
    ForkIdentity,
    ForkSnapshot,
    ManagedPortfolio,
    SwapQuote,
)
from tradingagents.integrations.onchain.policy import (
    MIN_NATIVE_WEI,
    ManagedBook,
    PolicyBlocked,
    PolicyConfig,
    apply_receipt,
    bind_book,
    build_intent,
    plan_quote_request,
    reserve_intent,
)
from tradingagents.integrations.onchain.profile import ROUTER, USDT, WBTC

NOW = datetime(2026, 10, 26, 10, tzinfo=UTC)
CYCLE = "2026-10-26T10:00Z"
HASH = "0x" + "a" * 64
TX = "0x" + "b" * 64
OTHER_TX = "0x" + "c" * 64


def identity(**updates):
    fields = {"instance_id": "test-fork-one", "fork_block_number": 100,
                  "fork_block_hash": HASH, "wallet_address": "0x" + "1" * 40}
    fields.update(updates)
    return ForkIdentity(**fields)


def snapshot(**updates):
    fields = {"identity": identity(), "observed_at": NOW, "block_number": 100,
                  "block_hash": HASH, "cash_units": 1_000_000_000, "btc_units": 100_000_000,
                  "native_wei": 10**18, "btc_price_usdt": "100000"}
    fields.update(updates)
    return ForkSnapshot(**fields)


def decision(rating="Buy", **updates):
    fields = {"decision_id": "structured-one", "rating": rating, "analyzed_at": NOW,
                  "expires_at": NOW + timedelta(minutes=15), "summary": "Strict opinion."}
    fields.update(updates)
    return AnalysisDecision(**fields)


def book(**updates):
    return bind_book(ManagedBook(**updates), snapshot())


def quote(request=None, **updates):
    request = request or plan_quote_request(decision(), snapshot(), book(), NOW, CYCLE)
    buying = request.side == "Buy"
    expected = request.amount_in_units // 1000 if buying else request.amount_in_units * 1000
    fields = {
        "identity": identity(), "quote_id": "test-quote", "side": request.side,
        "token_in": USDT if buying else WBTC, "token_out": WBTC if buying else USDT,
        "router": ROUTER, "amount_in_units": request.amount_in_units,
        "expected_out_units": expected,
        "minimum_out_units": (expected * (10000 - request.slippage_bps) + 9999) // 10000,
        "slippage_bps": request.slippage_bps, "pool_fee": 500, "quoted_block_number": 100,
        "quoted_block_hash": HASH, "quoted_at": NOW, "expires_at": NOW + timedelta(seconds=120),
    }
    fields.update(updates)
    return SwapQuote(**fields)


def reserved(initial=None, rating="Buy"):
    initial = initial or book()
    analysis = decision(rating)
    request = plan_quote_request(analysis, snapshot(), initial, NOW, CYCLE)
    intent = build_intent(analysis, snapshot(), initial, quote(request), NOW, CYCLE)
    return intent, reserve_intent(initial, intent)


def receipt(intent, status="confirmed", **updates):
    fields = {
        "identity": identity(), "intent_id": intent.intent_id, "status": status,
        "transaction_hashes": [TX], "observed_at": NOW,
        "amount_in_units": intent.quote.amount_in_units if status == "confirmed" else 0,
        "amount_out_units": intent.quote.expected_out_units if status == "confirmed" else 0,
        "gas_paid_wei": 100_000_000_000_000,
        "confirmed_block_number": 101 if status in {"confirmed", "reverted"} else None,
        "confirmed_block_hash": OTHER_TX if status in {"confirmed", "reverted"} else None,
    }
    fields.update(updates)
    return ExecutionReceipt(**fields)


def test_initial_binding_never_adopts_wallet_wbtc_or_additional_cash():
    initial = ManagedBook()
    assert initial.identity is None and initial.btc_units == 0
    bound = bind_book(initial, snapshot())
    assert initial.identity is None
    assert bound.identity == identity()
    assert bound.cash_units == 300_000_000 and bound.btc_units == 0
    assert bind_book(bound, snapshot()) == bound
    assert ManagedBook.model_validate_json(bound.model_dump_json()) == bound
    assert plan_quote_request(decision("Sell"), snapshot(), bound, NOW, CYCLE) is None


@pytest.mark.parametrize("changes", [
    {"instance_id": "restarted-fork"}, {"wallet_address": "0x" + "2" * 40},
    {"fork_block_number": 99}, {"fork_block_hash": OTHER_TX},
])
def test_bound_identity_never_follows_another_instance_wallet_or_fork(changes):
    changed = snapshot(identity=identity(**changes), block_number=101)
    with pytest.raises(PolicyBlocked, match="fork_identity_changed"):
        bind_book(book(), changed)
    with pytest.raises(PolicyBlocked, match="fork_identity_changed"):
        plan_quote_request(decision(), changed, book(), NOW, CYCLE)


def test_unbound_managed_book_cannot_plan_or_reserve():
    with pytest.raises(PolicyBlocked, match="book_identity_not_bound"):
        plan_quote_request(decision(), snapshot(), ManagedBook(), NOW, CYCLE)
    intent, unused = reserved()
    with pytest.raises(PolicyBlocked, match="intent_identity_mismatch"):
        reserve_intent(ManagedBook(), intent)


@pytest.mark.parametrize("rating, units", [("Buy", 150_000_000), ("Overweight", 75_000_000), ("Underweight", 30_000_000)])
def test_target_ratings_use_usdt_base_units(rating, units):
    request = plan_quote_request(decision(rating), snapshot(), book(), NOW, CYCLE)
    assert request.side == "Buy" and request.amount_in_units == units
    assert type(request.amount_in_units) is int and request.slippage_bps == 50


def test_hold_and_repeated_buy_at_target_do_not_trade():
    assert plan_quote_request(decision("Hold"), snapshot(), book(), NOW, CYCLE) is None
    initial = book(cash_units=150_000_000, btc_units=150_000)
    assert plan_quote_request(decision(), snapshot(), initial, NOW, CYCLE) is None
    intent, pending = reserved()
    done = apply_receipt(pending, receipt(intent))
    assert plan_quote_request(decision(decision_id="next"), snapshot(), done, NOW, "next-hour") is None


def test_sell_respects_owned_tokens_and_order_cap():
    initial = book(cash_units=100_000_000, btc_units=100_000)
    request = plan_quote_request(decision("Sell"), snapshot(), initial, NOW, CYCLE)
    assert request.side == "Sell" and request.amount_in_units == 100_000
    assert request.amount_in_units < snapshot().btc_units
    larger = book(cash_units=0, btc_units=500_000)
    request = plan_quote_request(decision("Sell"), snapshot(), larger, NOW, CYCLE)
    assert request.amount_in_units == 150_000


def test_decimal_price_integer_rounding_never_oversizes():
    snap = snapshot(btc_price_usdt="99999.123456789")
    initial = book(cash_units=0, btc_units=500_000)
    request = plan_quote_request(decision("Sell"), snap, initial, NOW, CYCLE)
    value = Decimal(request.amount_in_units) / Decimal(100_000_000) * snap.btc_price_usdt
    assert value <= Decimal("150")
    assert (Decimal(request.amount_in_units + 1) / Decimal(100_000_000) * snap.btc_price_usdt) > Decimal("150")


@pytest.mark.parametrize("price", [
    "80240.123456789", "99999.123456789", "123456.789123456789",
    "80240." + "9876543210" * 12,
])
@pytest.mark.parametrize("owned_units", [1, 12461, 120_000])
def test_full_sell_below_order_cap_uses_exact_owned_units(price, owned_units):
    initial = book(cash_units=0, btc_units=owned_units)
    snap = snapshot(btc_price_usdt=price)
    request = plan_quote_request(decision("Sell"), snap, initial, NOW, CYCLE)
    assert request is not None and request.side == "Sell"
    assert request.amount_in_units == owned_units


def test_full_sell_at_exact_cap_uses_all_units():
    initial = book(cash_units=0, btc_units=150_000)
    request = plan_quote_request(decision("Sell"), snapshot(), initial, NOW, CYCLE)
    assert request.amount_in_units == initial.btc_units


@pytest.mark.parametrize("price", ["100000", "99999.123456789", "80240.123456789"])
def test_full_sell_above_cap_remains_clipped_by_notional(price):
    initial = book(cash_units=0, btc_units=1_000_000)
    snap = snapshot(btc_price_usdt=price)
    request = plan_quote_request(decision("Sell"), snap, initial, NOW, CYCLE)
    value = Decimal(request.amount_in_units) / Decimal(100_000_000) * snap.btc_price_usdt
    next_value = Decimal(request.amount_in_units + 1) / Decimal(100_000_000) * snap.btc_price_usdt
    assert request.amount_in_units < initial.btc_units
    assert value <= Decimal("150") < next_value


@pytest.mark.parametrize("changes", [
    {"observed_at": NOW - timedelta(seconds=121)},
    {"observed_at": NOW + timedelta(microseconds=1)},
    {"native_wei": MIN_NATIVE_WEI - 1}, {"cash_units": 299_999_999},
    {"block_number": 99}, {"block_hash": OTHER_TX},
])
def test_snapshot_safety_failures_block(changes):
    with pytest.raises(PolicyBlocked):
        plan_quote_request(decision(), snapshot(**changes), book(), NOW, CYCLE)


def test_unavailable_owned_btc_cannot_be_sold():
    with pytest.raises(PolicyBlocked, match="managed_funds_unavailable"):
        plan_quote_request(decision("Sell"), snapshot(btc_units=0), book(btc_units=1000), NOW, CYCLE)


@pytest.mark.parametrize("changes", [
    {"analyzed_at": NOW + timedelta(seconds=1)},
    {"analyzed_at": NOW - timedelta(seconds=3601)},
    {"expires_at": NOW + timedelta(minutes=16)},
    {"analyzed_at": NOW - timedelta(minutes=15), "expires_at": NOW},
])
def test_future_stale_expired_or_long_lived_decisions_block(changes):
    with pytest.raises(PolicyBlocked):
        plan_quote_request(decision(**changes), snapshot(), book(), NOW, CYCLE)


def test_plain_portfolio_projection_can_size_without_persisted_history():
    request = plan_quote_request(decision(), snapshot(), ManagedPortfolio(), NOW, CYCLE)
    assert request.amount_in_units == 150_000_000


def test_intent_id_determinism_expiry_and_ceil_minimum_output():
    request = plan_quote_request(decision(), snapshot(), book(), NOW, CYCLE)
    quoted = quote(request, expected_out_units=101, minimum_out_units=101)
    intent = build_intent(decision(), snapshot(), book(), quoted, NOW, CYCLE)
    assert len(intent.intent_id) == 40 and intent.intent_id.startswith("ta-fork-")
    assert intent == build_intent(decision(), snapshot(), book(), quoted, NOW, CYCLE)
    assert intent.expires_at == quoted.expires_at
    # floor(101 * .995) = 100 violates the required ceiling, so no unsafe
    # one-base-unit shortfall can enter the executable intent.
    with pytest.raises(ValidationError):
        quote(request, expected_out_units=101, minimum_out_units=100)
    bypassed = quoted.model_copy(update={"minimum_out_units": 100})
    with pytest.raises(PolicyBlocked):
        build_intent(decision(), snapshot(), book(), bypassed, NOW, CYCLE)


@pytest.mark.parametrize("changes", [
    {"side": "Sell"}, {"amount_in_units": 149_999_999},
    {"token_in": WBTC}, {"token_out": USDT}, {"router": "0x" + "3" * 40},
    {"identity": identity(instance_id="foreign")}, {"quoted_block_number": 101},
    {"quoted_block_hash": OTHER_TX}, {"quoted_at": NOW + timedelta(seconds=1)},
    {"quoted_at": NOW - timedelta(seconds=121)},
    {"expires_at": NOW + timedelta(seconds=121)},
    {"slippage_bps": 49, "minimum_out_units": 149265},
])
def test_quote_must_match_exact_request_profile_identity_and_fresh_block(changes):
    with pytest.raises(PolicyBlocked):
        build_intent(decision(), snapshot(), book(), quote(**changes), NOW, CYCLE)


def test_quote_expiry_does_not_get_extended_by_planning():
    snap = snapshot(observed_at=NOW - timedelta(seconds=119))
    intent = build_intent(decision(), snap, book(), quote(), NOW, CYCLE)
    assert intent.expires_at == NOW + timedelta(seconds=1)
    with pytest.raises(PolicyBlocked, match="quote_expired"):
        build_intent(decision(), snapshot(), book(), quote(quoted_at=NOW - timedelta(seconds=1), expires_at=NOW), NOW, CYCLE)


def test_reservation_is_pure_and_duplicate_cycle_or_decision_is_blocked():
    initial = book()
    intent, pending = reserved(initial)
    assert initial.pending_intent is None
    assert pending.pending_intent == intent
    assert ManagedBook.model_validate_json(pending.model_dump_json()) == pending
    with pytest.raises(PolicyBlocked, match="unresolved_intent"):
        plan_quote_request(decision(decision_id="new"), snapshot(), pending, NOW, "new-hour")
    with pytest.raises(PolicyBlocked):
        reserve_intent(pending, intent)
    done = apply_receipt(pending, receipt(intent))
    assert plan_quote_request(decision(), snapshot(), done, NOW, "other-hour") is None
    assert plan_quote_request(decision(decision_id="other"), snapshot(), done, NOW, CYCLE) is None


@pytest.mark.parametrize("rating, initial", [
    ("Buy", {}), ("Sell", {"cash_units": 100_000_000, "btc_units": 100_000}),
])
def test_confirmed_actual_token_amounts_and_native_gas_account_exactly_once(rating, initial):
    start = book(**initial)
    intent, pending = reserved(start, rating)
    filled = receipt(intent)
    done = apply_receipt(pending, filled)
    assert done.pending_intent is None
    assert done.gas_paid_wei == filled.gas_paid_wei
    if rating == "Buy":
        assert done.cash_units == start.cash_units - filled.amount_in_units
        assert done.btc_units == start.btc_units + filled.amount_out_units
    else:
        assert done.btc_units == start.btc_units - filled.amount_in_units
        assert done.cash_units == start.cash_units + filled.amount_out_units
    assert apply_receipt(ManagedBook.model_validate_json(done.model_dump_json()), filled) == done


@pytest.mark.parametrize("status", ["pending", "unknown"])
def test_uncertain_effect_keeps_reservation_and_never_changes_token_ownership(status):
    intent, pending = reserved()
    uncertain = receipt(intent, status, gas_paid_wei=0, transaction_hashes=[])
    updated = apply_receipt(pending, uncertain)
    assert updated.pending_intent == intent
    assert updated.cash_units == pending.cash_units and updated.btc_units == pending.btc_units
    assert apply_receipt(updated, uncertain) == updated
    with pytest.raises(PolicyBlocked):
        reserve_intent(updated, intent)


def test_confirmed_approval_gas_accumulates_without_usd_valuation_or_double_counting():
    intent, pending = reserved()
    approval = receipt(intent, "pending", gas_paid_wei=50_000, transaction_hashes=[TX])
    partial = apply_receipt(pending, approval)
    assert partial.gas_paid_wei == 50_000
    assert partial.cash_units == pending.cash_units and partial.btc_units == pending.btc_units
    final = receipt(intent, gas_paid_wei=80_000, transaction_hashes=[TX, OTHER_TX])
    done = apply_receipt(partial, final)
    assert done.gas_paid_wei == 80_000
    assert done.cash_units == 300_000_000 - intent.quote.amount_in_units
    assert apply_receipt(done, final) == done


def test_reverted_swap_spends_only_actual_native_gas():
    intent, pending = reserved()
    reverted = receipt(intent, "reverted", gas_paid_wei=100_000)
    done = apply_receipt(pending, reverted)
    assert done.pending_intent is None
    assert done.cash_units == 300_000_000 and done.btc_units == 0
    assert done.gas_paid_wei == 100_000
    assert apply_receipt(done, reverted) == done
    invalid = receipt(intent, "reverted", amount_in_units=1)
    with pytest.raises(PolicyBlocked, match="reverted_swap_has_token_amounts"):
        apply_receipt(pending, invalid)


@pytest.mark.parametrize("changes", [
    {"identity": identity(instance_id="restart")}, {"amount_in_units": 1},
    {"amount_out_units": 1}, {"confirmed_block_number": 99},
])
def test_invalid_confirmed_receipts_never_mutate_book(changes):
    intent, pending = reserved()
    with pytest.raises(PolicyBlocked):
        apply_receipt(pending, receipt(intent, **changes))
    assert pending.cash_units == 300_000_000 and pending.btc_units == 0


def test_terminal_receipt_changes_and_nonterminal_regressions_block():
    intent, pending = reserved()
    terminal = receipt(intent)
    done = apply_receipt(pending, terminal)
    with pytest.raises(PolicyBlocked, match="terminal_receipt_changed"):
        apply_receipt(done, receipt(intent, gas_paid_wei=terminal.gas_paid_wei + 1))
    first = apply_receipt(pending, receipt(intent, "pending", gas_paid_wei=10))
    with pytest.raises(PolicyBlocked, match="cumulative_gas_regressed"):
        apply_receipt(first, receipt(intent, "unknown", gas_paid_wei=9))
    with pytest.raises(PolicyBlocked, match="transaction_history_regressed"):
        apply_receipt(first, receipt(intent, "unknown", gas_paid_wei=10, transaction_hashes=[OTHER_TX]))


def test_receipt_without_reservation_is_not_an_import_route():
    intent, unused = reserved()
    with pytest.raises(PolicyBlocked, match="receipt_without_reserved_intent"):
        apply_receipt(book(), receipt(intent))


def test_corrupted_history_cannot_drop_pending_or_change_ownership_identity():
    intent, pending = reserved()
    data = pending.model_dump()
    data["pending_intent"] = None
    with pytest.raises(ValidationError):
        ManagedBook.model_validate(data)
    data = pending.model_dump()
    data["identity"] = identity(instance_id="foreign").model_dump()
    with pytest.raises(ValidationError):
        ManagedBook.model_validate(data)
    data = pending.model_dump()
    data["intent_history"][intent.intent_id]["quote"]["router"] = "0x" + "3" * 40
    with pytest.raises(ValidationError):
        ManagedBook.model_validate(data)


def test_strict_integer_units_decimal_money_and_config_caps():
    for data in ({"cash_units": 1.0}, {"cash_units": "1"}, {"btc_units": True}):
        with pytest.raises(ValidationError):
            ManagedBook(**data)
    with pytest.raises(ValidationError):
        PolicyConfig(capital_cap_usdt=300.0)
    with pytest.raises(ValidationError):
        PolicyConfig(max_order_usdt="151")
    with pytest.raises(ValidationError):
        PolicyConfig(slippage_bps=51)
    with pytest.raises(ValidationError):
        PolicyConfig(min_native_wei=MIN_NATIVE_WEI - 1)
    invalid = snapshot().model_copy(update={"cash_units": -1})
    with pytest.raises(PolicyBlocked):
        plan_quote_request(decision(), invalid, book(), NOW, CYCLE)
