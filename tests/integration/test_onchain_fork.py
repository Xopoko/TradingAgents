"""Opt-in real-bytecode swap acceptance on a verified disposable Anvil fork.

Requires an otherwise unused fork account and an explicit persistent journal
directory. Never runs in the default offline test suite.
"""
from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tradingagents.agents.schemas import PortfolioRating
from tradingagents.integrations.onchain.contracts import AnalysisDecision, GatewayRequest
from tradingagents.integrations.onchain.policy import (
    ManagedBook,
    apply_receipt,
    bind_book,
    build_intent,
    plan_quote_request,
    reserve_intent,
)
from tradingagents.integrations.onchain.profile import USDT, WBTC
from tradingagents.integrations.onchain.rpc import ForkRPC
from tradingagents.integrations.onchain.runner import SubprocessGateway
from tradingagents.integrations.onchain.storage import read_model, write_model

pytestmark = pytest.mark.integration


def _seed_empty_test_account(rpc: ForkRPC) -> None:
    """Discover and change only an empty account balance on the verified fork."""
    from web3 import Web3

    identity = rpc.identity()
    wallet = identity.wallet_address
    cash = rpc.balance(USDT, wallet)
    assert rpc.balance(WBTC, wallet) == 0, "Use a fresh disposable fork account"
    if cash == 20_000_000:
        return
    assert cash == 0, "Do not reset an existing account to fund this test"
    marker = 123_456_789
    slot_hash = None
    for slot in range(20):
        saved = rpc.call("evm_snapshot", [])
        candidate = Web3.to_hex(Web3.keccak(bytes.fromhex(wallet[2:]).rjust(32, b"\0") + slot.to_bytes(32, "big")))
        try:
            rpc.assert_identity(identity)
            rpc.call("anvil_setStorageAt", [USDT, candidate, "0x" + marker.to_bytes(32, "big").hex()])
            if rpc.balance(USDT, wallet) == marker:
                slot_hash = candidate
        finally:
            assert rpc.call("evm_revert", [saved]) is True
        if slot_hash is not None:
            break
    assert slot_hash is not None, "No supported USDT balance slot was found"
    rpc.assert_identity(identity)
    rpc.call("anvil_setStorageAt", [USDT, slot_hash, "0x" + (20_000_000).to_bytes(32, "big").hex()])
    assert rpc.balance(USDT, wallet) == 20_000_000


def test_real_uniswap_buy_sell_and_restart_reconciliation():
    rpc_url = os.environ.get("ONCHAIN_FORK_RPC")
    state_path = os.environ.get("ONCHAIN_FORK_TEST_STATE")
    if not rpc_url or not state_path:
        pytest.skip("Set ONCHAIN_FORK_RPC and a persistent ONCHAIN_FORK_TEST_STATE")
    pytest.importorskip("almanak")
    root = Path(state_path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rpc = ForkRPC(rpc_url)
    identity = rpc.identity()
    book_path = root / "book.json"
    if not book_path.exists():
        _seed_empty_test_account(rpc)
        book = ManagedBook(cash_units=20_000_000, capital_limit_usdt="20")
    else:
        book = read_model(book_path, ManagedBook)
    gateway = SubprocessGateway(root, rpc_url=rpc_url, worker_python=sys.executable)

    def request(value):
        response = gateway(value)
        assert response.ok, response.safe_error_code
        return response

    if book.pending_intent is not None:
        receipt = request(GatewayRequest(operation="reconcile", intent_id=book.pending_intent.intent_id)).receipt
        assert receipt.status == "confirmed", receipt.safe_error_code
        book = apply_receipt(book, receipt)
        write_model(book_path, book)

    for label, rating in (("buy", PortfolioRating.BUY), ("sell", PortfolioRating.SELL)):
        cycle = f"acceptance-{label}"
        if cycle in book.processed_cycle_ids:
            continue
        snapshot = request(GatewayRequest(operation="snapshot")).snapshot
        book = bind_book(book, snapshot)
        write_model(book_path, book)
        now = datetime.now(UTC)
        decision = AnalysisDecision(
            decision_id=f"acceptance-{identity.instance_id}-{label}", rating=rating,
            analyzed_at=now, expires_at=now + timedelta(minutes=15),
            summary="Explicit local-fork acceptance fixture, not a model recommendation.",
            source="fixture",
        )
        quote_request = plan_quote_request(decision, snapshot, book, now, cycle)
        assert quote_request is not None
        quote = request(GatewayRequest(operation="quote", quote_request=quote_request)).quote
        intent = build_intent(decision, snapshot, book, quote, datetime.now(UTC), cycle)
        book = reserve_intent(book, intent)
        write_model(book_path, book)
        receipt = request(GatewayRequest(operation="execute", intent=intent)).receipt
        write_model(root / f"{label}-receipt.json", receipt)
        assert receipt.status == "confirmed", receipt.safe_error_code
        assert receipt.amount_in_units == intent.quote.amount_in_units
        assert receipt.amount_out_units >= intent.quote.minimum_out_units
        assert receipt.gas_paid_wei > 0
        assert receipt.confirmed_block_hash is not None
        book = apply_receipt(book, receipt)
        write_model(book_path, book)
        assert book.pending_intent is None
        count = rpc.call("eth_getTransactionCount", [identity.wallet_address, "latest"])
        # A newly launched worker receives the same execution request. It must
        # recover the receipt instead of broadcasting another transaction.
        replay = request(GatewayRequest(operation="execute", intent=intent)).receipt
        assert replay == receipt or (
            replay.status == receipt.status
            and replay.transaction_hashes == receipt.transaction_hashes
            and replay.amount_in_units == receipt.amount_in_units
            and replay.amount_out_units == receipt.amount_out_units
            and replay.gas_paid_wei == receipt.gas_paid_wei
        )
        assert rpc.call("eth_getTransactionCount", [identity.wallet_address, "latest"]) == count
        assert apply_receipt(book, replay) == book

    assert len(book.applied_receipts) == 2
    assert book.btc_units == 0
    assert 19_000_000 < book.cash_units <= 20_000_000
    assert book.cash_units == rpc.balance(USDT, identity.wallet_address)
    assert book.btc_units == rpc.balance(WBTC, identity.wallet_address)
