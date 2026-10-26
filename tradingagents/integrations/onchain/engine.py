"""Isolated Anvil execution worker. There is deliberately no public-chain mode.

The SDK compiler and execution pipeline are pinned. Our adapter supplies an
unlocked Anvil signer, local chain ID, bounded calldata and a write-ahead journal.
The journal never contains keys or raw signed transactions.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import sys
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from pydantic import Field

from .contracts import (
    Address,
    Contract,
    ExecutionReceipt,
    ForkIdentity,
    ForkSnapshot,
    GatewayRequest,
    GatewayResponse,
    Hash,
    QuoteRequest,
    SwapIntent,
    SwapQuote,
    Units,
)
from .profile import ALMANAK_VERSION, FACTORY, LOCAL_CHAIN_ID, QUOTER, ROUTER, USDT, WBTC
from .rpc import ForkError, ForkRPC

_FEES = (100, 500, 3000, 10000)
_TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
_SWAP = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
_ID = re.compile(r"^ta-fork-[a-f0-9]{32}$")
MAX_TRANSACTION_GAS = 1_000_000
MAX_FEE_PER_GAS_WEI = 50_000_000_000
MAX_BUNDLE_GAS_WEI = 4_000_000_000_000_000
NATIVE_RESERVE_WEI = 1_000_000_000_000_000


def _now() -> datetime:
    return datetime.now(UTC)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "worker.lock"
    # OS locks release on process termination. An O_EXCL sentinel would strand
    # reconciliation permanently after the very crash this journal addresses.
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ForkError("worker_locked") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ForkError("journal_missing_or_invalid") from None


def _word(value: int | str) -> str:
    if isinstance(value, str):
        if not re.fullmatch(r"0x[0-9a-fA-F]{40}", value):
            raise ForkError("invalid_address")
        return value[2:].lower().zfill(64)
    if isinstance(value, bool) or not 0 <= value < 2**256:
        raise ForkError("invalid_integer")
    return f"{value:064x}"


def _decode(data: str) -> list[int]:
    if not isinstance(data, str) or not re.fullmatch(r"0x(?:[0-9a-fA-F]{64})+", data):
        raise ForkError("invalid_contract_response")
    return [int(data[i : i + 64], 16) for i in range(2, len(data), 64)]


def _pool(rpc: ForkRPC, fee: int) -> str:
    answer = _decode(rpc.eth_call(FACTORY, "0x1698ee82" + _word(WBTC) + _word(USDT) + _word(fee)))
    address = "0x" + f"{answer[0]:040x}"
    if len(address) != 42 or answer[0] == 0:
        raise ForkError("pool_unavailable")
    if not rpc.code(address) or _decode(rpc.eth_call(address, "0x1a686502"))[0] <= 0:
        raise ForkError("pool_liquidity_unavailable")
    tokens = [
        _decode(rpc.eth_call(address, selector))[0] for selector in ("0x0dfe1681", "0xd21220a7")
    ]
    if tokens != sorted((int(WBTC, 16), int(USDT, 16))):
        raise ForkError("pool_identity_mismatch")
    if _decode(rpc.eth_call(address, "0xddca3f43"))[0] != fee:
        raise ForkError("pool_fee_mismatch")
    return address


def _pool_price(rpc: ForkRPC, pool: str) -> Decimal:
    sqrt_price = _decode(rpc.eth_call(pool, "0x3850c7bd"))[0]
    with localcontext() as ctx:
        ctx.prec = 100
        price = Decimal(sqrt_price) ** 2 / Decimal(2**192) * Decimal(100)
    if price <= 0:
        raise ForkError("pool_price_invalid")
    return price


def _quote_amount(rpc: ForkRPC, token_in: str, token_out: str, units: int, fee: int) -> int:
    calldata = "0xc6a5026a" + "".join(_word(v) for v in (token_in, token_out, units, fee, 0))
    return _decode(rpc.eth_call(QUOTER, calldata))[0]


def _balances(rpc: ForkRPC, identity: ForkIdentity) -> dict[str, int]:
    wallet = identity.wallet_address
    return {
        "cash": rpc.balance(USDT, wallet),
        "btc": rpc.balance(WBTC, wallet),
        "native": rpc.native_balance(wallet),
    }


def _bind(root: Path, identity: ForkIdentity, *, create: bool) -> None:
    path = root / "identity.json"
    if path.exists():
        try:
            saved = ForkIdentity.model_validate(_read(path))
        except ValueError:
            raise ForkError("journal_identity_invalid") from None
        if saved != identity:
            raise ForkError("journal_identity_mismatch")
    elif create:
        if any((root / name).exists() for name in ("intents", "quotes")):
            raise ForkError("journal_identity_missing")
        _write(path, identity.model_dump(mode="json"))
    else:
        raise ForkError("journal_identity_missing")


def _snapshot(rpc: ForkRPC, identity: ForkIdentity) -> ForkSnapshot:
    pool = None
    for fee in _FEES:
        try:
            pool = _pool(rpc, fee)
            break
        except ForkError:
            continue
    if pool is None:
        raise ForkError("no_usable_pool")
    block = rpc.block()
    balances = _balances(rpc, identity)
    return ForkSnapshot(
        identity=identity,
        observed_at=_now(),
        block_number=int(block["number"], 16),
        block_hash=block["hash"],
        cash_units=balances["cash"],
        btc_units=balances["btc"],
        native_wei=balances["native"],
        btc_price_usdt=_pool_price(rpc, pool),
    )


def _quote(rpc: ForkRPC, identity: ForkIdentity, request: QuoteRequest, root: Path) -> SwapQuote:
    token_in, token_out = (USDT, WBTC) if request.side == "Buy" else (WBTC, USDT)
    candidates = []
    for fee in _FEES:
        try:
            _pool(rpc, fee)
            output = _quote_amount(rpc, token_in, token_out, request.amount_in_units, fee)
            if output > 0:
                candidates.append((output, fee))
        except ForkError:
            continue
    if not candidates:
        raise ForkError("no_executable_quote")
    output, fee = max(candidates)
    minimum = (output * (10000 - request.slippage_bps) + 9999) // 10000
    block, now = rpc.block(), _now()
    quote = SwapQuote(
        identity=identity,
        quote_id=uuid.uuid4().hex,
        side=request.side,
        token_in=token_in,
        token_out=token_out,
        router=ROUTER,
        amount_in_units=request.amount_in_units,
        expected_out_units=output,
        minimum_out_units=minimum,
        slippage_bps=request.slippage_bps,
        pool_fee=fee,
        quoted_block_number=int(block["number"], 16),
        quoted_block_hash=block["hash"],
        quoted_at=now,
        expires_at=now + timedelta(seconds=120),
    )
    _write(root / "quotes" / (quote.quote_id + ".json"), quote.model_dump(mode="json"))
    return quote


class TxRecord(Contract):
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    target: Address
    data: str = Field(pattern=r"^0x(?:[0-9a-fA-F]{2})*$")
    value: Units = 0
    tx_hash: Hash | None = None
    nonce: int | None = Field(default=None, ge=0, strict=True)
    attempted: bool = False


class IntentJournal(Contract):
    version: Literal[1] = 1
    identity: ForkIdentity
    intent: SwapIntent
    intent_digest: str
    before: dict[str, Units]
    pool: Address
    transactions: list[TxRecord] = Field(default_factory=list)
    phase: Literal[
        "preparing", "prepared", "broadcasting", "confirmed", "reverted", "pending", "unknown"
    ] = "preparing"
    failure_code: str | None = None
    receipt: ExecutionReceipt | None = None


def _journal_path(root: Path, intent_id: str) -> Path:
    if not _ID.fullmatch(intent_id):
        raise ForkError("invalid_intent_id")
    return root / "intents" / (intent_id + ".json")


def _load_journal(path: Path) -> IntentJournal:
    try:
        journal = IntentJournal.model_validate(_read(path))
        if journal.version != 1 or journal.intent_digest != _digest(
            journal.intent.model_dump(mode="json")
        ):
            raise ValueError()
        for tx in journal.transactions:
            if tx.fingerprint != _digest({"to": tx.target, "data": tx.data, "value": tx.value}):
                raise ValueError()
        return journal
    except ValueError:
        raise ForkError("journal_invalid") from None


def _save_journal(path: Path, journal: IntentJournal) -> None:
    _write(path, journal.model_dump(mode="json"))


def _bounded_calldata(transactions: list[dict], intent: SwapIntent, wallet: str) -> list[dict]:
    """Allow only exact approvals and one SDK single-pool exact-input swap."""
    quote = intent.quote
    output, swap_count, approvals = [], 0, []
    for transaction in transactions:
        tx = dict(transaction)
        target = str(tx.get("to", "")).lower()
        data = str(tx.get("data", "")).lower()
        if int(tx.get("value", 0)) != 0:
            raise ForkError("native_value_forbidden")
        if target == quote.token_in.lower() and data.startswith("0x095ea7b3"):
            args = _decode("0x" + data[10:])
            if (
                len(args) != 2
                or args[0] != int(ROUTER, 16)
                or args[1] not in (0, quote.amount_in_units)
            ):
                raise ForkError("approval_contract_mismatch")
            approvals.append(args[1])
        elif target == ROUTER and data.startswith("0x04e45aaf"):
            args = _decode("0x" + data[10:])
            expected = [
                int(quote.token_in, 16),
                int(quote.token_out, 16),
                quote.pool_fee,
                int(wallet, 16),
                quote.amount_in_units,
            ]
            if len(args) != 7 or args[:5] != expected or args[6] != 0 or args[5] <= 0:
                raise ForkError("swap_contract_mismatch")
            if args[5] < quote.minimum_out_units:
                # SDK rounds down; our contract uses a ceiling. Tighten, never relax.
                args[5] = quote.minimum_out_units
            tx["data"] = "0x04e45aaf" + "".join(_word(v) for v in args)
            swap_count += 1
        else:
            raise ForkError("transaction_target_or_selector_forbidden")
        output.append(tx)
    if (
        swap_count != 1
        or len(output) > 3
        or approvals not in ([], [quote.amount_in_units], [0, quote.amount_in_units])
    ):
        raise ForkError("bundle_shape_forbidden")
    if str(output[-1].get("to", "")).lower() != ROUTER:
        raise ForkError("swap_must_be_last")
    return output


def _deadline_wrap(data: str, deadline: int) -> str:
    """Canonical ABI multicall(uint256,bytes[]) containing exactly one call."""
    payload = data[2:]
    padded = payload.ljust(((len(payload) + 63) // 64) * 64, "0")
    return (
        "0x5ae401dc"
        + _word(deadline)
        + _word(64)
        + _word(1)
        + _word(32)
        + _word(len(payload) // 2)
        + padded
    )


def _exact_approval_payloads(
    rpc: ForkRPC, quote: SwapQuote, wallet: str, token: str, spender: str, amount: int
) -> list[dict]:
    """Replace the pinned SDK's 10% approval buffer with our exact spend.

    This adapter is intentionally limited to the single declared swap. Unknown
    allowance reads fail closed; USDT changes reset an existing allowance first.
    The final calldata guard independently checks the resulting transactions.
    """
    if (
        token.lower() != quote.token_in.lower()
        or spender.lower() != ROUTER
        or type(amount) is not int
        or amount != quote.amount_in_units
    ):
        raise ForkError("approval_request_mismatch")
    answer = _decode(rpc.eth_call(token, "0xdd62ed3e" + _word(wallet) + _word(ROUTER)))
    if len(answer) != 1:
        raise ForkError("allowance_response_invalid")
    allowance = answer[0]
    if allowance >= amount:
        return []
    values = [0, amount] if token.lower() == USDT and allowance > 0 else [amount]
    return [
        {
            "to": token,
            "value": 0,
            "data": "0x095ea7b3" + _word(ROUTER) + _word(value),
            "description": "Exact swap allowance" if value else "Reset swap allowance",
            "tx_type": "approve" if value else "approve_reset",
        }
        for value in values
    ]


def _validate_gas_budget(txs: list, native_wei: int) -> int:
    """Check the final unsigned fields, before any transaction is signed."""
    if type(native_wei) is not int or native_wei < 0 or not txs:
        raise ForkError("gas_budget_input_invalid")
    maximum_cost = 0
    for tx in txs:
        gas, fee, priority = tx.gas_limit, tx.max_fee_per_gas, tx.max_priority_fee_per_gas
        if any(type(value) is not int for value in (gas, fee, priority)):
            raise ForkError("gas_field_invalid")
        if not 0 < gas <= MAX_TRANSACTION_GAS:
            raise ForkError("gas_limit_exceeded")
        if not 0 <= fee <= MAX_FEE_PER_GAS_WEI:
            raise ForkError("gas_fee_cap_exceeded")
        if not 0 <= priority <= fee:
            raise ForkError("gas_priority_invalid")
        maximum_cost += gas * fee
    if maximum_cost > MAX_BUNDLE_GAS_WEI:
        raise ForkError("gas_bundle_budget_exceeded")
    if native_wei < maximum_cost + NATIVE_RESERVE_WEI:
        raise ForkError("native_gas_reserve_insufficient")
    return maximum_cost


class _Observer:
    """Gateway-shaped reader backed exclusively by our guarded local transport."""

    def __init__(self, rpc: ForkRPC):
        self._rpc = rpc
        self.rpc = self
        self.config = SimpleNamespace(timeout=15)
        self.is_connected = True

    def eth_call(
        self, chain, to, data, *, block=None, raise_on_error=False, from_address=None, value=0
    ):
        del chain, raise_on_error
        transaction = {"to": to, "data": data}
        if from_address:
            transaction["from"] = from_address
        if value:
            transaction["value"] = hex(value)
        return self._rpc.call(
            "eth_call", [transaction, hex(block) if block is not None else "latest"]
        )

    def block_number(self, chain):
        return int(self._rpc.block()["number"], 16)

    def read(self, *, chain, target, payload, block_number=None):
        return bytes.fromhex(
            self.eth_call(chain, target.reference, "0x" + payload.hex(), block=block_number)[2:]
        )

    def code(self, *, chain, target, block_number=None):
        return bytes.fromhex(
            self._rpc.call(
                "eth_getCode", [target.reference, hex(block_number) if block_number else "latest"]
            )[2:]
        )

    def Call(self, request, timeout=None):
        del timeout
        try:
            result = self._rpc.call(request.method, json.loads(request.params or "[]"))
            return SimpleNamespace(success=True, result=json.dumps(result), error="")
        except ForkError:
            return SimpleNamespace(success=False, result="", error="guarded_rpc_failed")


async def _run_sdk(rpc: ForkRPC, journal: IntentJournal, path: Path) -> None:
    """Imports remain lazy so the root bot environment never needs this SDK."""
    from importlib.metadata import version

    from almanak.framework.execution.interfaces import SignedTransaction, Signer
    from almanak.framework.execution.orchestrator import ExecutionContext, ExecutionOrchestrator
    from almanak.framework.execution.simulator.local import LocalSimulator
    from almanak.framework.execution.submitter.public import PublicMempoolSubmitter
    from almanak.framework.intents import SwapIntent as SDKSwapIntent
    from almanak.framework.intents.compiler import IntentCompiler, IntentCompilerConfig
    from almanak.framework.intents.compiler_constants import get_gas_estimate
    from almanak.framework.intents.compiler_models import TransactionData
    from web3 import Web3

    if version("almanak") != ALMANAK_VERSION:
        raise ForkError("sdk_version_mismatch")
    identity, quote = journal.identity, journal.intent.quote
    observer = _Observer(rpc)

    class AnvilSigner(Signer):
        @property
        def address(self):
            return identity.wallet_address

        async def sign_batch(self, txs, chain):
            rpc.assert_identity(identity)
            _validate_gas_budget(txs, rpc.native_balance(self.address))
            return await super().sign_batch(txs, chain)

        async def sign(self, tx, chain):
            del chain
            rpc.assert_identity(identity)
            if tx.chain_id != LOCAL_CHAIN_ID:
                raise ForkError("signing_chain_mismatch")
            transaction = {
                "from": self.address,
                "to": tx.to,
                "data": tx.data,
                "value": hex(tx.value),
                "gas": hex(tx.gas_limit),
                "nonce": hex(tx.nonce),
                "chainId": hex(LOCAL_CHAIN_ID),
                "type": "0x2",
                "maxFeePerGas": hex(tx.max_fee_per_gas),
                "maxPriorityFeePerGas": hex(tx.max_priority_fee_per_gas),
            }
            signed = rpc.call("eth_signTransaction", [transaction])
            raw = signed.get("raw") if isinstance(signed, dict) else signed
            if not isinstance(raw, str) or not re.fullmatch(r"0x[0-9a-fA-F]+", raw):
                raise ForkError("anvil_signing_unavailable")
            tx_hash = Web3.to_hex(Web3.keccak(hexstr=raw))
            return SignedTransaction(raw_tx=raw, tx_hash=tx_hash, unsigned_tx=tx)

    class ForkOrchestrator(ExecutionOrchestrator):
        async def _build_unsigned_transactions(self, bundle, context):
            rpc.assert_identity(identity)
            transactions = await super()._build_unsigned_transactions(bundle, context)
            result = []
            for tx in transactions:
                data = tx.data
                if tx.to.lower() == ROUTER:
                    data = _deadline_wrap(data, int(journal.intent.expires_at.timestamp()))
                result.append(replace(tx, chain_id=LOCAL_CHAIN_ID, data=data))
            return result

    class ExactApprovalCompiler(IntentCompiler):
        def _build_approve_tx(self, token_address, spender, amount):
            rpc.assert_identity(identity)
            payloads = _exact_approval_payloads(
                rpc, quote, identity.wallet_address, token_address, spender, amount
            )
            return [
                TransactionData(**payload, gas_estimate=get_gas_estimate("ethereum", "approve"))
                for payload in payloads
            ]

    class JournalSubmitter(PublicMempoolSubmitter):
        async def _submit_single(self, signed_tx):
            rpc.assert_identity(identity)
            if _now() >= journal.intent.expires_at:
                raise ForkError("intent_expired_before_broadcast")
            tx = signed_tx.unsigned_tx
            fingerprint = _digest({"to": tx.to.lower(), "data": tx.data.lower(), "value": tx.value})
            matches = [entry for entry in journal.transactions if entry.fingerprint == fingerprint]
            if len(matches) != 1 or matches[0].attempted:
                raise ForkError("journal_transaction_mismatch")
            entry = matches[0]
            entry.tx_hash, entry.nonce, entry.attempted = signed_tx.tx_hash, tx.nonce, True
            journal.phase = "broadcasting"
            _save_journal(path, journal)  # fsync + replace BEFORE the network side effect.
            # No SDK retry/resubmit: uncertain outcome is reconciled by hash.
            return await super()._submit_single(signed_tx)

    compiler = ExactApprovalCompiler(
        chain="ethereum",
        wallet_address=identity.wallet_address,
        rpc_url=rpc.url,
        gateway_client=None,
        price_oracle={"USDT": Decimal(1), "WBTC": _pool_price(rpc, journal.pool)},
        config=IntentCompilerConfig(
            allow_placeholder_prices=False,
            offline_discovery=True,
            managed_fork=True,
            swap_pool_selection_mode="fixed",
            fixed_swap_fee_tier=quote.pool_fee,
        ),
    )
    amount_scale = Decimal(1_000_000 if quote.side == "Buy" else 100_000_000)
    sdk_intent = SDKSwapIntent(
        from_token="USDT" if quote.side == "Buy" else "WBTC",
        to_token="WBTC" if quote.side == "Buy" else "USDT",
        amount=Decimal(quote.amount_in_units) / amount_scale,
        max_slippage=Decimal(quote.slippage_bps) / 10000,
        protocol="uniswap_v3",
        chain="ethereum",
    )
    compiled = compiler.compile(sdk_intent)
    if compiled.status.value != "SUCCESS" or compiled.action_bundle is None:
        raise ForkError("sdk_compilation_failed")
    bundle = compiled.action_bundle
    bounded = _bounded_calldata(bundle.transactions, journal.intent, identity.wallet_address)
    # ActionBundle is a dataclass in the pinned SDK.
    swap_args = _decode("0x" + bounded[-1]["data"][10:])
    metadata = {
        **bundle.metadata,
        "min_amount_out": str(swap_args[5]),
        "deadline": int(journal.intent.expires_at.timestamp()),
    }
    bundle = replace(bundle, transactions=bounded, metadata=metadata)
    for tx in bounded:
        data = tx["data"].lower()
        if tx["to"].lower() == ROUTER:
            data = _deadline_wrap(data, int(journal.intent.expires_at.timestamp()))
        payload = {"to": tx["to"].lower(), "data": data, "value": int(tx.get("value", 0))}
        journal.transactions.append(
            TxRecord(
                fingerprint=_digest(payload),
                target=payload["to"],
                data=data,
                value=payload["value"],
            )
        )
    journal.phase = "prepared"
    _save_journal(path, journal)
    signer = AnvilSigner()
    submitter = JournalSubmitter(
        rpc_url=rpc.url, max_retries=0, timeout_seconds=25, chain="ethereum"
    )
    orchestrator = ForkOrchestrator(
        signer=signer,
        submitter=submitter,
        simulator=LocalSimulator(rpc_url=rpc.url),
        chain="ethereum",
        rpc_url=rpc.url,
        tx_timeout_seconds=25,
        managed_fork=True,
        operation_observer_factory=lambda: observer,
    )
    context = ExecutionContext(
        wallet_address=signer.address,
        chain="ethereum",
        simulation_enabled=True,
        intent_id=journal.intent.intent_id,
        cycle_id=journal.intent.cycle_id,
        protocol="uniswap_v3",
    )
    result = await orchestrator.execute(bundle, context)
    if not result.success:
        phase = getattr(result.error_phase, "value", "execution").lower()
        raise ForkError("sdk_" + phase + "_not_confirmed")


def _transfer_amount(receipts: list[dict], token: str, wallet: str, *, outgoing: bool) -> int:
    amount = 0
    for receipt in receipts:
        for log in receipt.get("logs", []):
            topics = log.get("topics", [])
            if (
                log.get("address", "").lower() != token.lower()
                or len(topics) != 3
                or topics[0].lower() != _TRANSFER
            ):
                continue
            account = topics[1 if outgoing else 2][-40:].lower()
            if account == wallet[2:].lower():
                amount += int(log["data"], 16)
    return amount


def _reconcile(rpc: ForkRPC, journal: IntentJournal, path: Path) -> ExecutionReceipt:
    rpc.assert_identity(journal.identity)
    if journal.receipt is not None and journal.receipt.status in {"confirmed", "reverted"}:
        # Revalidate canonical receipt blocks below even for terminal journals.
        pass
    receipts, hashes = [], []
    missing = False
    for tx in journal.transactions:
        if not tx.attempted:
            continue
        if tx.tx_hash is None or tx.nonce is None:
            raise ForkError("journal_hash_missing")
        hashes.append(tx.tx_hash)
        receipt = rpc.call("eth_getTransactionReceipt", [tx.tx_hash])
        if receipt is None:
            missing = True
            continue
        transaction = rpc.call("eth_getTransactionByHash", [tx.tx_hash])
        if transaction is None or transaction.get("hash", "").lower() != tx.tx_hash.lower():
            raise ForkError("receipt_transaction_missing")
        payload = {
            "to": transaction.get("to", "").lower(),
            "data": transaction.get("input", "").lower(),
            "value": int(transaction.get("value", "0x0"), 16),
        }
        if (
            _digest(payload) != tx.fingerprint
            or int(transaction["nonce"], 16) != tx.nonce
            or transaction.get("from", "").lower() != journal.identity.wallet_address.lower()
            or int(transaction.get("chainId", "0x0"), 16) != LOCAL_CHAIN_ID
        ):
            raise ForkError("receipt_payload_mismatch")
        block = rpc.call("eth_getBlockByNumber", [receipt["blockNumber"], False])
        if (
            not block
            or block["hash"].lower() != receipt["blockHash"].lower()
            or receipt.get("transactionHash", "").lower() != tx.tx_hash.lower()
        ):
            raise ForkError("receipt_block_mismatch")
        receipts.append(receipt)
    gas = sum(int(r["gasUsed"], 16) * int(r["effectiveGasPrice"], 16) for r in receipts)
    kwargs = {
        "intent_id": journal.intent.intent_id,
        "identity": journal.identity,
        "transaction_hashes": hashes,
        "gas_paid_wei": gas,
        "observed_at": _now(),
    }
    if missing:
        result = ExecutionReceipt(status="pending", safe_error_code="receipt_pending", **kwargs)
    elif any(int(r["status"], 16) == 0 for r in receipts):
        result = ExecutionReceipt(
            status="reverted", safe_error_code="transaction_reverted", **kwargs
        )
    elif receipts and len(receipts) == len(journal.transactions):
        quote, wallet = journal.intent.quote, journal.identity.wallet_address
        incoming = _transfer_amount(receipts, quote.token_out, wallet, outgoing=False)
        outgoing = _transfer_amount(receipts, quote.token_in, wallet, outgoing=True)
        balances = _balances(rpc, journal.identity)
        input_key, output_key = ("cash", "btc") if quote.side == "Buy" else ("btc", "cash")
        swap_logs = [
            log
            for r in receipts
            for log in r.get("logs", [])
            if log.get("address", "").lower() == journal.pool.lower()
            and log.get("topics")
            and log["topics"][0].lower() == _SWAP
        ]
        # On subsequent reconciliation balances may have changed through later intents;
        # initial confirmation requires them, terminal re-check relies on immutable logs.
        balance_valid = (
            journal.before[input_key] - balances[input_key] == outgoing
            and balances[output_key] - journal.before[output_key] == incoming
        )
        previously_confirmed = journal.receipt is not None and journal.receipt.status == "confirmed"
        if (
            outgoing != quote.amount_in_units
            or incoming < quote.minimum_out_units
            or len(swap_logs) != 1
            or (not previously_confirmed and not balance_valid)
        ):
            raise ForkError("swap_evidence_mismatch")
        event = swap_logs[0]
        amounts = _decode(event["data"])
        signed = [v - 2**256 if v >= 2**255 else v for v in amounts[:2]]
        expected_amounts = [-incoming, outgoing] if quote.side == "Buy" else [outgoing, -incoming]
        if (
            len(amounts) != 5
            or signed != expected_amounts
            or len(event["topics"]) != 3
            or event["topics"][2][-40:].lower() != wallet[2:].lower()
        ):
            raise ForkError("swap_event_amounts_mismatch")
        last = receipts[-1]
        result = ExecutionReceipt(
            status="confirmed",
            amount_in_units=outgoing,
            amount_out_units=incoming,
            confirmed_block_number=int(last["blockNumber"], 16),
            confirmed_block_hash=last["blockHash"],
            **kwargs,
        )
    else:
        result = ExecutionReceipt(
            status="unknown",
            safe_error_code=journal.failure_code or "partial_bundle_or_no_broadcast",
            **kwargs,
        )
    journal.receipt, journal.phase = result, result.status
    _save_journal(path, journal)
    return result


def handle_request(request: GatewayRequest, state_root: Path, rpc_url: str) -> GatewayResponse:
    """One durable operation. Existing intent IDs can only reconcile."""
    try:
        rpc = ForkRPC(rpc_url)
        with _lock(state_root):
            identity = rpc.identity()
            _bind(state_root, identity, create=request.operation in {"snapshot", "quote"})
            index_path = state_root / "intent-index.json"
            if not index_path.exists():
                if (
                    request.operation in {"execute", "reconcile"}
                    or (state_root / "intents").exists()
                ):
                    raise ForkError("intent_index_missing")
                _write(index_path, {})
            if request.operation == "snapshot":
                return GatewayResponse(ok=True, snapshot=_snapshot(rpc, identity))
            if request.operation == "quote":
                return GatewayResponse(
                    ok=True, quote=_quote(rpc, identity, request.quote_request, state_root)
                )
            if request.operation == "reconcile":
                path = _journal_path(state_root, request.intent_id)
                journal = _load_journal(path)
                if journal.identity != identity:
                    raise ForkError("journal_identity_mismatch")
                return GatewayResponse(ok=True, receipt=_reconcile(rpc, journal, path))
            intent = request.intent
            path = _journal_path(state_root, intent.intent_id)
            index = _read(index_path)
            if not isinstance(index, dict):
                raise ForkError("intent_index_invalid")
            if path.exists():
                journal = _load_journal(path)
                if (
                    journal.intent_digest != _digest(intent.model_dump(mode="json"))
                    or journal.identity != identity
                ):
                    raise ForkError("intent_replay_mismatch")
                return GatewayResponse(ok=True, receipt=_reconcile(rpc, journal, path))
            if intent.intent_id in index:
                raise ForkError("intent_journal_missing")
            for prior_id in index:
                prior = _load_journal(_journal_path(state_root, prior_id))
                if prior.receipt is None or prior.receipt.status not in {"confirmed", "reverted"}:
                    raise ForkError("prior_intent_unresolved")
            if intent.quote.quote_id in [
                entry.get("quote_id") for entry in index.values() if isinstance(entry, dict)
            ]:
                raise ForkError("quote_already_consumed")
            if intent.quote.identity != identity or _now() >= intent.expires_at:
                raise ForkError("intent_identity_or_expiry_mismatch")
            if not re.fullmatch(r"[a-f0-9]{32}", intent.quote.quote_id):
                raise ForkError("quote_id_invalid")
            saved_quote = SwapQuote.model_validate(
                _read(state_root / "quotes" / (intent.quote.quote_id + ".json"))
            )
            if saved_quote != intent.quote:
                raise ForkError("quote_not_issued_by_worker")
            pool = _pool(rpc, intent.quote.pool_fee)
            value = Decimal(intent.quote.amount_in_units) / (
                1_000_000 if intent.quote.side == "Buy" else 100_000_000
            )
            if intent.quote.side == "Sell":
                value *= _pool_price(rpc, pool)
            if value > Decimal(150):
                raise ForkError("order_notional_limit")
            journal = IntentJournal(
                identity=identity,
                intent=intent,
                intent_digest=_digest(intent.model_dump(mode="json")),
                before=_balances(rpc, identity),
                pool=pool,
            )
            index[intent.intent_id] = {
                "intent_digest": journal.intent_digest,
                "quote_id": intent.quote.quote_id,
            }
            _write(index_path, index)
            _save_journal(path, journal)
            try:
                asyncio.run(_run_sdk(rpc, journal, path))
            except Exception as error:
                # After a signing/submission failure the only safe action is receipt
                # reconciliation. Do not print SDK exceptions containing raw requests.
                journal.failure_code = (
                    error.code
                    if isinstance(error, ForkError)
                    else "sdk_" + type(error).__name__.lower()
                )
                _save_journal(path, journal)
            return GatewayResponse(ok=True, receipt=_reconcile(rpc, journal, path))
    except ForkError as error:
        return GatewayResponse(ok=False, safe_error_code=error.code)
    except Exception:
        return GatewayResponse(ok=False, safe_error_code="worker_internal_error")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--rpc-url", required=True)
    parser.add_argument("--response-file", required=True, type=Path)
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    try:
        request = GatewayRequest.model_validate_json(sys.stdin.read(1_000_001))
        response = handle_request(request, args.state_root, args.rpc_url)
    except Exception:
        response = GatewayResponse(ok=False, safe_error_code="invalid_worker_request")
    _write(args.response_file, response.model_dump(mode="json"))
    return (
        0
        if response.ok
        and (response.receipt is None or response.receipt.status in {"confirmed", "reverted"})
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
