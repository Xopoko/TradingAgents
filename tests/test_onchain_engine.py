"""Offline safety tests: bounded ABI, write-ahead state, and receipt recovery."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from tradingagents.integrations.onchain import engine
from tradingagents.integrations.onchain.contracts import (
    ForkIdentity,
    GatewayRequest,
    QuoteRequest,
    SwapIntent,
    SwapQuote,
)
from tradingagents.integrations.onchain.profile import ROUTER, USDT, WBTC
from tradingagents.integrations.onchain.rpc import ForkError

HASH = "0x" + "ab" * 32
BLOCK = "0x" + "bc" * 32
WALLET = "0x" + "12" * 20
POOL = "0x" + "34" * 20
IDENTITY = ForkIdentity(
    instance_id="unit-anvil", fork_block_number=100, fork_block_hash=BLOCK, wallet_address=WALLET
)


def quote():
    now = datetime.now(UTC)
    return SwapQuote(
        identity=IDENTITY,
        quote_id="a" * 32,
        side="Buy",
        token_in=USDT,
        token_out=WBTC,
        router=ROUTER,
        amount_in_units=10_000_000,
        expected_out_units=10_000,
        minimum_out_units=9950,
        slippage_bps=50,
        pool_fee=3000,
        quoted_block_number=100,
        quoted_block_hash=BLOCK,
        quoted_at=now,
        expires_at=now + timedelta(seconds=120),
    )


def intent():
    result = quote()
    return SwapIntent(
        intent_id="ta-fork-" + "b" * 32,
        decision_id="unit-decision",
        cycle_id="unit-cycle",
        quote=result,
        created_at=result.quoted_at,
        expires_at=result.expires_at,
    )


def swap_data(trade):
    values = (
        USDT,
        WBTC,
        trade.quote.pool_fee,
        WALLET,
        trade.quote.amount_in_units,
        trade.quote.minimum_out_units,
        0,
    )
    return "0x04e45aaf" + "".join(engine._word(x) for x in values)


def tx_payload(trade):
    return {
        "to": ROUTER,
        "data": engine._deadline_wrap(swap_data(trade), int(trade.expires_at.timestamp())),
        "value": 0,
    }


class FakeRPC:
    def __init__(self, _url=None):
        self.url = "http://127.0.0.1:8545"
        self.cash, self.btc, self.native = 300_000_000, 0, 10**18
        self.receipts = {}
        self.transactions = {}

    def identity(self):
        return IDENTITY

    def assert_identity(self, expected):
        if expected != IDENTITY:
            raise ForkError("fork_identity_changed")

    def balance(self, token, wallet):
        assert wallet == WALLET
        return self.cash if token == USDT else self.btc

    def native_balance(self, wallet):
        return self.native

    def call(self, method, params):
        if method == "eth_getTransactionReceipt":
            return self.receipts.get(params[0])
        if method == "eth_getTransactionByHash":
            return self.transactions.get(params[0])
        if method == "eth_getBlockByNumber":
            return {"number": "0x65", "hash": BLOCK}
        raise AssertionError(method)

    def record(self, trade, *, status=1, logs=True):
        payload = tx_payload(trade)
        self.transactions[HASH] = {
            "hash": HASH,
            "to": ROUTER,
            "input": payload["data"],
            "value": "0x0",
            "nonce": "0x0",
            "from": WALLET,
            "chainId": hex(31337),
        }
        topic_wallet = "0x" + engine._word(WALLET)
        topic_other = "0x" + engine._word(POOL)
        entries = (
            [
                {
                    "address": USDT,
                    "topics": [engine._TRANSFER, topic_wallet, topic_other],
                    "data": hex(10_000_000),
                },
                {
                    "address": WBTC,
                    "topics": [engine._TRANSFER, topic_other, topic_wallet],
                    "data": hex(10_000),
                },
                {
                    "address": POOL,
                    "topics": [engine._SWAP, topic_other, topic_wallet],
                    "data": "0x"
                    + "".join(engine._word(x) for x in ((2**256 - 10_000), 10_000_000, 1, 1, 0)),
                },
            ]
            if logs
            else []
        )
        self.receipts[HASH] = {
            "transactionHash": HASH,
            "blockNumber": "0x65",
            "blockHash": BLOCK,
            "status": hex(status),
            "gasUsed": "0x5208",
            "effectiveGasPrice": "0x2",
            "logs": entries,
        }
        if status:
            self.cash -= 10_000_000
            self.btc += 10_000


def setup(monkeypatch, tmp_path):
    rpc = FakeRPC()
    monkeypatch.setattr(engine, "ForkRPC", lambda _url: rpc)
    monkeypatch.setattr(engine, "_pool", lambda _rpc, _fee: POOL)
    monkeypatch.setattr(engine, "_pool_price", lambda _rpc, _pool: Decimal("100000"))

    # Mock SDK coroutines below have no awaits. Do not construct a Windows
    # Proactor socket pair under the suite's intentional socket prohibition.
    def run_immediate(coroutine):
        try:
            coroutine.send(None)
        except StopIteration as complete:
            return complete.value
        raise AssertionError("mock coroutine unexpectedly suspended")

    monkeypatch.setattr(engine.asyncio, "run", run_immediate)
    engine._bind(tmp_path, IDENTITY, create=True)
    engine._write(tmp_path / "intent-index.json", {})
    trade = intent()
    engine._write(
        tmp_path / "quotes" / (trade.quote.quote_id + ".json"), trade.quote.model_dump(mode="json")
    )
    return rpc, trade


def prepared(trade):
    payload = tx_payload(trade)
    return engine.TxRecord(
        fingerprint=engine._digest(payload),
        target=ROUTER,
        data=payload["data"],
        value=0,
        tx_hash=HASH,
        nonce=0,
        attempted=True,
    )


def test_bounded_exact_swap_tightens_only_minimum():
    trade = intent()
    data = swap_data(trade)
    low = data[:-128] + engine._word(9949) + engine._word(0)
    bounded = engine._bounded_calldata([{"to": ROUTER, "data": low, "value": "0"}], trade, WALLET)
    assert bounded[0]["data"] == data


@pytest.mark.parametrize(
    "bad", ["spender", "unlimited", "recipient", "token", "native", "selector"]
)
def test_bounded_calldata_refuses_unsafe_payloads(bad):
    trade = intent()
    data = swap_data(trade)
    transaction = {"to": ROUTER, "data": data, "value": 0}
    if bad in {"spender", "unlimited"}:
        spender = POOL if bad == "spender" else ROUTER
        amount = 2**256 - 1 if bad == "unlimited" else trade.quote.amount_in_units
        transaction = {
            "to": USDT,
            "data": "0x095ea7b3" + engine._word(spender) + engine._word(amount),
            "value": 0,
        }
    elif bad == "recipient":
        transaction["data"] = data[: 10 + 3 * 64] + engine._word(POOL) + data[10 + 4 * 64 :]
    elif bad == "token":
        transaction["data"] = data[:10] + engine._word(POOL) + data[74:]
    elif bad == "native":
        transaction["value"] = 1
    else:
        transaction["data"] = "0xa9059cbb" + data[10:]
    with pytest.raises(ForkError):
        engine._bounded_calldata([transaction], trade, WALLET)


def test_multicall_deadline_is_canonical_one_call():
    trade = intent()
    raw = engine._deadline_wrap(swap_data(trade), 1234)
    words = raw[10:]
    assert raw.startswith("0x5ae401dc")
    assert [int(words[i : i + 64], 16) for i in range(0, 320, 64)] == [1234, 64, 1, 32, 228]
    assert words[320 : 320 + 456] == swap_data(trade)[2:]


@pytest.mark.parametrize(
    ("allowance", "expected"), [(0, [10_000_000]), (1, [0, 10_000_000]), (10_000_000, [])]
)
def test_exact_approval_adapter_preserves_usdt_reset_and_exact_amount(allowance, expected):
    rpc = SimpleNamespace(eth_call=lambda _token, _data: "0x" + engine._word(allowance))
    trade = intent()
    txs = engine._exact_approval_payloads(rpc, trade.quote, WALLET, USDT, ROUTER, 10_000_000)
    assert [engine._decode("0x" + tx["data"][10:])[1] for tx in txs] == expected
    for tx in txs:
        assert engine._decode("0x" + tx["data"][10:])[0] == int(ROUTER, 16)
    # The independent final guard sees only exact approvals, never SDK's 11m.
    engine._bounded_calldata(
        txs + [{"to": ROUTER, "data": swap_data(trade), "value": 0}], trade, WALLET
    )


def test_exact_approval_adapter_refuses_unknown_allowance_or_wrong_request():
    trade = intent()
    rpc = SimpleNamespace(eth_call=lambda _token, _data: "0x")
    with pytest.raises(ForkError, match="invalid_contract_response"):
        engine._exact_approval_payloads(rpc, trade.quote, WALLET, USDT, ROUTER, 10_000_000)
    with pytest.raises(ForkError, match="approval_request_mismatch"):
        engine._exact_approval_payloads(rpc, trade.quote, WALLET, USDT, POOL, 10_000_000)
    with pytest.raises(ForkError, match="approval_request_mismatch"):
        engine._exact_approval_payloads(rpc, trade.quote, WALLET, USDT, ROUTER, 11_000_000)


def gas_tx(gas=100_000, fee=1_000_000_000, priority=0):
    return SimpleNamespace(gas_limit=gas, max_fee_per_gas=fee, max_priority_fee_per_gas=priority)


def test_gas_budget_checks_final_bundle_and_reserve_at_boundaries():
    txs = [gas_tx(1_000_000, 4_000_000_000)]
    assert engine._validate_gas_budget(txs, 5_000_000_000_000_000) == engine.MAX_BUNDLE_GAS_WEI
    assert engine._validate_gas_budget([gas_tx(), gas_tx()], 10**18) == 200_000_000_000_000
    with pytest.raises(ForkError, match="native_gas_reserve_insufficient"):
        engine._validate_gas_budget(txs, 5_000_000_000_000_000 - 1)
    with pytest.raises(ForkError, match="gas_bundle_budget_exceeded"):
        engine._validate_gas_budget([gas_tx(100_000, 25_000_000_000)] * 2, 10**18)


@pytest.mark.parametrize(
    ("tx", "error"),
    [
        (gas_tx(0), "gas_limit_exceeded"),
        (gas_tx(1_000_001), "gas_limit_exceeded"),
        (gas_tx(True), "gas_field_invalid"),
        (gas_tx(fee=True), "gas_field_invalid"),
        (gas_tx(priority=True), "gas_field_invalid"),
        (gas_tx(fee=-1), "gas_fee_cap_exceeded"),
        (gas_tx(fee=50_000_000_001), "gas_fee_cap_exceeded"),
        (gas_tx(priority=-1), "gas_priority_invalid"),
        (gas_tx(priority=1_000_000_001), "gas_priority_invalid"),
    ],
)
def test_gas_budget_rejects_malformed_or_excessive_fields(tx, error):
    with pytest.raises(ForkError, match=error):
        engine._validate_gas_budget([tx], 10**18)


def test_quote_uses_executable_tiers_and_ceiling_bound(monkeypatch, tmp_path):
    rpc = FakeRPC()
    rpc.block = lambda: {"number": "0x64", "hash": BLOCK}

    def pool(_rpc, fee):
        if fee == 100:
            raise ForkError("pool_unavailable")
        return POOL

    monkeypatch.setattr(engine, "_pool", pool)
    monkeypatch.setattr(
        engine, "_quote_amount", lambda _rpc, _a, _b, _c, fee: {500: 1, 3000: 101, 10000: 99}[fee]
    )
    result = engine._quote(rpc, IDENTITY, QuoteRequest(side="Buy", amount_in_units=1000), tmp_path)
    assert result.pool_fee == 3000
    assert result.expected_out_units == result.minimum_out_units == 101
    assert (result.expires_at - result.quoted_at).total_seconds() == 120
    assert (tmp_path / "quotes" / (result.quote_id + ".json")).is_file()


def test_execute_replay_only_reconciles_and_reports_real_units(monkeypatch, tmp_path):
    rpc, trade = setup(monkeypatch, tmp_path)
    calls = []

    async def run(_rpc, journal, path):
        calls.append(1)
        journal.transactions.append(prepared(trade))
        engine._save_journal(path, journal)
        rpc.record(trade)

    monkeypatch.setattr(engine, "_run_sdk", run)
    request = GatewayRequest(operation="execute", intent=trade)
    first = engine.handle_request(request, tmp_path, rpc.url)
    second = engine.handle_request(request, tmp_path, rpc.url)
    assert first.ok and second.ok
    assert first.receipt.status == second.receipt.status == "confirmed"
    assert first.receipt.amount_in_units == 10_000_000
    assert first.receipt.amount_out_units == 10_000
    assert first.receipt.gas_paid_wei == 42000
    assert calls == [1]


def test_timeout_journal_blocks_rebroadcast_then_reconciles(monkeypatch, tmp_path):
    rpc, trade = setup(monkeypatch, tmp_path)
    calls = []

    async def run(_rpc, journal, path):
        calls.append(1)
        journal.transactions.append(prepared(trade))
        engine._save_journal(path, journal)
        raise TimeoutError()

    monkeypatch.setattr(engine, "_run_sdk", run)
    request = GatewayRequest(operation="execute", intent=trade)
    first = engine.handle_request(request, tmp_path, rpc.url)
    second = engine.handle_request(request, tmp_path, rpc.url)
    assert first.receipt.status == second.receipt.status == "pending"
    assert first.receipt.amount_in_units == 0
    assert calls == [1]
    rpc.record(trade)
    recovered = engine.handle_request(
        GatewayRequest(operation="reconcile", intent_id=trade.intent_id), tmp_path, rpc.url
    )
    assert recovered.receipt.status == "confirmed"
    assert calls == [1]


def test_lost_worker_intent_journal_fails_closed(monkeypatch, tmp_path):
    rpc, trade = setup(monkeypatch, tmp_path)
    engine._write(
        tmp_path / "intent-index.json", {trade.intent_id: {"quote_id": trade.quote.quote_id}}
    )
    monkeypatch.setattr(engine, "_run_sdk", lambda *args: pytest.fail("must not execute"))
    response = engine.handle_request(
        GatewayRequest(operation="execute", intent=trade), tmp_path, rpc.url
    )
    assert not response.ok and response.safe_error_code == "intent_journal_missing"


def test_missing_journal_reconcile_never_executes(monkeypatch, tmp_path):
    rpc, trade = setup(monkeypatch, tmp_path)
    response = engine.handle_request(
        GatewayRequest(operation="reconcile", intent_id=trade.intent_id), tmp_path, rpc.url
    )
    assert not response.ok and response.safe_error_code == "journal_missing_or_invalid"


def test_reverted_receipt_does_not_manufacture_swap_amounts(monkeypatch, tmp_path):
    rpc, trade = setup(monkeypatch, tmp_path)

    async def run(_rpc, journal, path):
        journal.transactions.append(prepared(trade))
        engine._save_journal(path, journal)
        rpc.record(trade, status=0, logs=False)

    monkeypatch.setattr(engine, "_run_sdk", run)
    result = engine.handle_request(
        GatewayRequest(operation="execute", intent=trade), tmp_path, rpc.url
    )
    assert result.receipt.status == "reverted"
    assert result.receipt.amount_in_units == result.receipt.amount_out_units == 0
    assert result.receipt.gas_paid_wei == 42000


def test_receipt_payload_mismatch_is_refused(monkeypatch, tmp_path):
    rpc, trade = setup(monkeypatch, tmp_path)

    async def run(_rpc, journal, path):
        journal.transactions.append(prepared(trade))
        engine._save_journal(path, journal)
        rpc.record(trade)
        rpc.transactions[HASH]["input"] = "0x00"

    monkeypatch.setattr(engine, "_run_sdk", run)
    result = engine.handle_request(
        GatewayRequest(operation="execute", intent=trade), tmp_path, rpc.url
    )
    assert not result.ok and result.safe_error_code == "receipt_payload_mismatch"


def test_worker_lock_is_not_removed_or_stolen(monkeypatch, tmp_path):
    rpc, _trade = setup(monkeypatch, tmp_path)
    with engine._lock(tmp_path):
        result = engine.handle_request(GatewayRequest(operation="snapshot"), tmp_path, rpc.url)
        assert not result.ok and result.safe_error_code == "worker_locked"
    # A subsequent process can acquire without removing stale PID files.
    with engine._lock(tmp_path):
        pass
