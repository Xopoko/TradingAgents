"""Restart and effect-boundary checks using typed, entirely offline fork fixtures."""

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.schemas import PortfolioRating
from tradingagents.integrations.onchain import runner
from tradingagents.integrations.onchain.contracts import (
    AnalysisDecision,
    ExecutionReceipt,
    ForkIdentity,
    ForkSnapshot,
    GatewayResponse,
    ManagedPortfolio,
    SwapQuote,
)
from tradingagents.integrations.onchain.policy import ManagedBook
from tradingagents.integrations.onchain.profile import ROUTER, USDT, WBTC
from tradingagents.integrations.onchain.storage import read_model, write_model

NOW = datetime(2026, 10, 26, 12, tzinfo=UTC)
TX_HASH = "0x" + "a" * 64
PRIVATE_ERROR = "private-worker-error-must-never-be-printed"


def _identity():
    return ForkIdentity(
        instance_id="offline-fork",
        fork_block_number=20_000_000,
        fork_block_hash="0x" + "1" * 64,
        wallet_address="0x" + "2" * 40,
    )


def _snapshot(now=NOW, identity=None):
    return ForkSnapshot(
        identity=identity or _identity(),
        observed_at=now,
        block_number=20_000_010,
        block_hash="0x" + "3" * 64,
        cash_units=50_000_000_000,
        btc_units=100_000_000,
        native_wei=10**18,
        btc_price_usdt="60000",
    )


def _decision(now=NOW, rating=PortfolioRating.BUY):
    return AnalysisDecision(
        decision_id="offline-decision",
        rating=rating,
        analyzed_at=now,
        expires_at=now + timedelta(minutes=15),
        summary="Explicit offline fixture; no model recommendation.",
        source="fixture",
    )


def _receipt(intent, status="confirmed", gas=45_000, now=NOW):
    terminal = status == "confirmed"
    return ExecutionReceipt(
        intent_id=intent.intent_id,
        identity=intent.quote.identity,
        status=status,
        transaction_hashes=[TX_HASH] if status != "unknown" else [],
        amount_in_units=intent.quote.amount_in_units if terminal else 0,
        amount_out_units=intent.quote.expected_out_units if terminal else 0,
        gas_paid_wei=gas if status != "unknown" else 0,
        confirmed_block_number=20_000_011 if terminal else None,
        confirmed_block_hash="0x" + "4" * 64 if terminal else None,
        observed_at=now,
    )


class FakeGateway:
    def __init__(self, *, now=NOW, snapshots=None, execute_status="confirmed", hook=None):
        self.now = now
        self.snapshots = list(snapshots or [_snapshot(now)])
        self.execute_status = execute_status
        self.hook = hook
        self.calls = []
        self.intent = None
        self.reconcile_receipt = None
        self.snapshot_calls = 0

    def __call__(self, request):
        self.calls.append(request)
        if self.hook:
            self.hook(request)
        if request.operation == "snapshot":
            snapshot = self.snapshots[min(self.snapshot_calls, len(self.snapshots) - 1)]
            self.snapshot_calls += 1
            return GatewayResponse(ok=True, snapshot=snapshot)
        if request.operation == "quote":
            requested = request.quote_request
            buy = requested.side == "Buy"
            amount_out = (
                requested.amount_in_units * 100 // 60_000
                if buy
                else requested.amount_in_units * 60_000 // 100
            )
            snapshot = self.snapshots[-1]
            return GatewayResponse(
                ok=True,
                quote=SwapQuote(
                    identity=snapshot.identity,
                    quote_id="offline-quote",
                    side=requested.side,
                    token_in=USDT if buy else WBTC,
                    token_out=WBTC if buy else USDT,
                    router=ROUTER,
                    amount_in_units=requested.amount_in_units,
                    expected_out_units=amount_out,
                    minimum_out_units=(amount_out * (10000 - requested.slippage_bps) + 9999)
                    // 10000,
                    slippage_bps=requested.slippage_bps,
                    pool_fee=3000,
                    quoted_block_number=snapshot.block_number,
                    quoted_block_hash=snapshot.block_hash,
                    quoted_at=self.now,
                    expires_at=self.now + timedelta(seconds=60),
                ),
            )
        if request.operation == "execute":
            self.intent = request.intent
            return GatewayResponse(
                ok=True, receipt=_receipt(self.intent, self.execute_status, now=self.now)
            )
        if request.operation == "reconcile":
            assert self.reconcile_receipt is not None
            return GatewayResponse(ok=True, receipt=self.reconcile_receipt)
        raise AssertionError("unexpected gateway operation")


def _operations(gateway):
    return [request.operation for request in gateway.calls]


def _book(root):
    return read_model(root / "book.json", ManagedBook)


def test_reservation_is_persisted_before_execute_and_confirmed_amounts_are_applied(tmp_path):
    def inspect_reservation(request):
        if request.operation == "execute":
            saved = _book(tmp_path)
            assert saved.pending_intent == request.intent
            assert saved.intent_history[request.intent.intent_id] == request.intent
            assert request.intent.cycle_id in saved.processed_cycle_ids
            runtime = read_model(tmp_path / "runtime.json", runner.Runtime)
            assert runtime.pending_cycle == request.intent.cycle_id
            assert runtime.last_decision.source == "fixture"

    gateway = FakeGateway(hook=inspect_reservation)
    analyzer = MagicMock(return_value=_decision())
    result = runner.run_once(tmp_path, gateway=gateway, analyzer=analyzer, now=NOW)

    assert result.status == "confirmed"
    assert result.notify is True
    assert _operations(gateway) == ["snapshot", "snapshot", "quote", "execute"]
    saved = _book(tmp_path)
    assert saved.pending_intent is None
    assert saved.cash_units == 150_000_000
    assert saved.btc_units == 250_000
    assert saved.gas_paid_wei == 45_000
    assert len(saved.applied_receipts) == 1
    assert saved.identity == _identity()
    assert saved.btc_units != _snapshot().btc_units


def test_production_analyzer_receives_only_clean_managed_portfolio(tmp_path):
    def inspect_portfolio(snapshot, portfolio, root, now):
        assert type(portfolio) is ManagedPortfolio
        assert ManagedPortfolio.model_validate(portfolio.model_dump()) == portfolio
        assert portfolio.btc_units == 0
        assert portfolio.cash_units == 300_000_000
        return _decision(now, PortfolioRating.HOLD)

    result = runner.run_once(tmp_path, gateway=FakeGateway(), analyzer=inspect_portfolio, now=NOW)
    assert result.status == "no_order"


@pytest.mark.parametrize("status", ["pending", "unknown"])
def test_restart_only_reconciles_existing_intent_until_terminal(tmp_path, status):
    first = FakeGateway(execute_status=status)
    result = runner.run_once(tmp_path, gateway=first, analyzer=lambda *args: _decision(), now=NOW)
    assert result.status == "pending"
    reserved = _book(tmp_path).pending_intent
    assert reserved is not None
    analyzer = MagicMock(side_effect=AssertionError("pending execution must not rerun analysis"))

    restarted = FakeGateway(now=NOW + timedelta(hours=1))
    restarted.reconcile_receipt = _receipt(reserved, "pending", gas=50_000, now=restarted.now)
    pending = runner.run_once(tmp_path, gateway=restarted, analyzer=analyzer, now=restarted.now)
    assert pending.status == "pending"
    assert _operations(restarted) == ["reconcile"]
    assert restarted.calls[0].intent_id == reserved.intent_id
    assert _book(tmp_path).cash_units == 300_000_000
    assert _book(tmp_path).btc_units == 0
    assert _book(tmp_path).gas_paid_wei == 50_000
    repeated = runner.run_once(tmp_path, gateway=restarted, analyzer=analyzer, now=restarted.now)
    assert repeated.status == "pending"
    assert _operations(restarted) == ["reconcile", "reconcile"]
    assert _book(tmp_path).gas_paid_wei == 50_000

    terminal = FakeGateway(now=NOW + timedelta(hours=2))
    terminal.reconcile_receipt = _receipt(reserved, gas=80_000, now=terminal.now)
    confirmed = runner.run_once(tmp_path, gateway=terminal, analyzer=analyzer, now=terminal.now)
    assert confirmed.status == "confirmed"
    assert _operations(terminal) == ["reconcile"]
    assert _book(tmp_path).pending_intent is None
    assert _book(tmp_path).cash_units == 150_000_000
    assert _book(tmp_path).btc_units == 250_000
    assert _book(tmp_path).gas_paid_wei == 80_000
    analyzer.assert_not_called()


@pytest.mark.parametrize(
    "rating, expected", [(PortfolioRating.BUY, "confirmed"), (PortfolioRating.HOLD, "no_order")]
)
def test_completed_hour_is_not_repeated_and_does_not_contact_dependencies(
    tmp_path, rating, expected
):
    gateway = FakeGateway()
    analyzer = MagicMock(return_value=_decision(rating=rating))
    first = runner.run_once(tmp_path, gateway=gateway, analyzer=analyzer, now=NOW)
    assert first.status == expected
    calls = len(gateway.calls)
    second = runner.run_once(
        tmp_path, gateway=gateway, analyzer=analyzer, now=NOW + timedelta(minutes=30)
    )
    assert second.status == "already_completed"
    assert len(gateway.calls) == calls
    analyzer.assert_called_once()
    if rating is PortfolioRating.HOLD:
        assert _operations(gateway) == ["snapshot", "snapshot"]


def test_missing_ledger_with_existing_runtime_is_not_reset(tmp_path):
    write_model(tmp_path / "runtime.json", runner.Runtime(completed_cycles=["previous-cycle"]))
    before = (tmp_path / "runtime.json").read_bytes()
    gateway, analyzer = MagicMock(), MagicMock()
    result = runner.run_once(tmp_path, gateway=gateway, analyzer=analyzer, now=NOW)
    assert result.status == "error"
    assert result.safe_error_code == "ledger_missing"
    assert not (tmp_path / "book.json").exists()
    assert (tmp_path / "runtime.json").read_bytes() == before
    gateway.assert_not_called()
    analyzer.assert_not_called()


@pytest.mark.parametrize("filename", ["book.json", "runtime.json", "control.json"])
def test_corrupt_state_stops_without_overwriting_it(tmp_path, filename):
    path = tmp_path / filename
    path.write_text("{corrupt-state", encoding="utf-8")
    gateway, analyzer = MagicMock(), MagicMock()
    result = runner.run_once(tmp_path, gateway=gateway, analyzer=analyzer, now=NOW)
    assert result.status == "error"
    assert result.safe_error_code == "state_corrupt"
    assert path.read_text(encoding="utf-8") == "{corrupt-state"
    gateway.assert_not_called()
    analyzer.assert_not_called()


def test_disable_during_analysis_uses_separate_lock_and_stops_before_more_worker_calls(tmp_path):
    gateway = FakeGateway()

    def analyzer(*args):
        assert runner.set_enabled(tmp_path, False).status == "disabled"
        return _decision()

    result = runner.run_once(tmp_path, gateway=gateway, analyzer=analyzer, now=NOW)
    assert result.status == "disabled"
    assert _operations(gateway) == ["snapshot"]
    assert _book(tmp_path).pending_intent is None


def test_disable_after_reservation_preserves_pending_and_never_submits(tmp_path, monkeypatch):
    original_write = runner.write_model

    def disable_on_reservation(path, model):
        original_write(path, model)
        if Path(path).name == "book.json" and model.pending_intent is not None:
            runner.set_enabled(tmp_path, False)

    monkeypatch.setattr(runner, "write_model", disable_on_reservation)
    gateway = FakeGateway()
    result = runner.run_once(tmp_path, gateway=gateway, analyzer=lambda *args: _decision(), now=NOW)
    assert result.status == "disabled"
    assert result.safe_error_code == "execution_not_submitted"
    assert _operations(gateway) == ["snapshot", "snapshot", "quote"]
    assert _book(tmp_path).pending_intent is not None
    assert read_model(tmp_path / "runtime.json", runner.Runtime).pending_cycle == runner._cycle(NOW)
    # A locally blocked submission is not a venue receipt. Re-enabling must
    # reconcile the old reservation rather than inventing permission to resend.
    monkeypatch.setattr(runner, "write_model", original_write)
    runner.set_enabled(tmp_path, True)
    reserved = _book(tmp_path).pending_intent
    restarted = FakeGateway()
    restarted.reconcile_receipt = _receipt(reserved, status="unknown")
    analyzer = MagicMock()
    again = runner.run_once(tmp_path, gateway=restarted, analyzer=analyzer, now=NOW)
    assert again.status == "pending"
    assert _operations(restarted) == ["reconcile"]
    analyzer.assert_not_called()


@pytest.mark.parametrize("when", ["initial", "fresh"])
def test_fork_identity_change_blocks_execution_and_preserves_binding(tmp_path, when):
    initial = FakeGateway()
    assert (
        runner.run_once(
            tmp_path,
            gateway=initial,
            analyzer=lambda *args: _decision(rating=PortfolioRating.HOLD),
            now=NOW,
        ).status
        == "no_order"
    )
    next_time = NOW + timedelta(hours=1)
    changed = _snapshot(next_time, _identity().model_copy(update={"instance_id": "different-fork"}))
    unchanged = _snapshot(next_time)
    gateway = FakeGateway(
        now=next_time, snapshots=[changed] if when == "initial" else [unchanged, changed]
    )
    analyzer = MagicMock(return_value=_decision(next_time))
    result = runner.run_once(tmp_path, gateway=gateway, analyzer=analyzer, now=next_time)
    assert result.status == "error"
    assert result.safe_error_code == "unsafe_policy"
    assert _book(tmp_path).identity == _identity()
    assert "execute" not in _operations(gateway)
    assert "quote" not in _operations(gateway)
    if when == "initial":
        analyzer.assert_not_called()


def test_wrong_typed_worker_payload_is_rejected(tmp_path):
    # A well-typed envelope with the wrong operation result is still unusable.
    gateway = MagicMock(return_value=GatewayResponse(ok=True, receipt=_receipt_for_unknown()))
    analyzer = MagicMock()
    result = runner.run_once(tmp_path, gateway=gateway, analyzer=analyzer, now=NOW)
    assert result.safe_error_code == "worker_protocol_error"
    analyzer.assert_not_called()


def _receipt_for_unknown():
    return ExecutionReceipt(
        intent_id="ta-fork-" + "a" * 32,
        identity=_identity(),
        status="unknown",
        observed_at=NOW,
    )


@pytest.mark.parametrize(
    "payload", [None, "invalid-json", "x" * 131073], ids=["missing", "invalid", "oversized"]
)
def test_subprocess_stdout_cannot_replace_missing_or_malformed_protocol(
    tmp_path, monkeypatch, capsys, payload
):
    valid_stdout = GatewayResponse(ok=True, snapshot=_snapshot()).model_dump_json()

    def fake_run(command, **kwargs):
        assert kwargs["stdout"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.DEVNULL
        assert kwargs["check"] is False
        if payload is not None:
            Path(command[command.index("--response-file") + 1]).write_text(
                payload, encoding="utf-8"
            )
        return subprocess.CompletedProcess(command, 0, stdout=valid_stdout, stderr=PRIVATE_ERROR)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.run_once(
        tmp_path, gateway=runner.SubprocessGateway(tmp_path), analyzer=MagicMock(), now=NOW
    )
    assert result.status == "error"
    assert result.safe_error_code == "worker_protocol_error"
    assert PRIVATE_ERROR not in result.model_dump_json()
    output = capsys.readouterr()
    assert PRIVATE_ERROR not in output.out + output.err


@pytest.mark.parametrize("kind", ["exception", "envelope"])
def test_worker_errors_do_not_escape_through_cli_output(tmp_path, monkeypatch, capsys, kind):
    gateway = MagicMock()
    if kind == "exception":
        gateway.side_effect = RuntimeError(PRIVATE_ERROR)
    else:
        gateway.return_value = GatewayResponse(ok=False, safe_error_code=PRIVATE_ERROR)
    monkeypatch.setattr(runner, "SubprocessGateway", lambda *args, **kwargs: gateway)
    exit_code = runner.main(["--root", str(tmp_path), "once"])
    output = capsys.readouterr()
    assert exit_code == 1
    assert PRIVATE_ERROR not in output.out + output.err
    result = json.loads(output.out)
    assert result["status"] == "error"
    assert result["safe_error_code"] == "worker_failed"


def test_plain_dictionary_response_is_not_treated_as_a_validated_worker_contract(tmp_path):
    gateway = MagicMock(return_value=GatewayResponse(ok=True, snapshot=_snapshot()).model_dump())
    analyzer = MagicMock()
    result = runner.run_once(tmp_path, gateway=gateway, analyzer=analyzer, now=NOW)
    assert result.status == "error"
    assert result.safe_error_code == "worker_protocol_error"
    analyzer.assert_not_called()


def test_status_reads_only_local_state_without_initializing_a_worker(tmp_path, monkeypatch):
    factory = MagicMock()
    monkeypatch.setattr(runner, "SubprocessGateway", factory)
    result = runner.status(tmp_path)
    assert result.status == "status"
    assert not (tmp_path / "book.json").exists()
    assert not (tmp_path / "runtime.json").exists()
    factory.assert_not_called()
