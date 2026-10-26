"""No-network regression coverage for hourly cycle crash boundaries."""
import json
import subprocess
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from tradingagents.dataflows.files import locked
from tradingagents.integrations.bybit_demo.contracts import (
    AnalysisDecision,
    ExecutionReceipt,
    GatewayResponse,
    ManagedPortfolio,
    SpotSnapshot,
)
from tradingagents.integrations.bybit_demo.policy import ManagedBook
from tradingagents.integrations.bybit_demo.runner import (
    CREDENTIAL_NAMES,
    RunnerFailure,
    RuntimeState,
    SubprocessGateway,
    main,
    read_status,
    run_once,
    set_enabled,
)

NOW = datetime(2026, 10, 26, 12, 10, tzinfo=UTC)


def snapshot(at=NOW):
    return SpotSnapshot(
        observed_at=at, quote_at=at, bid="100000", ask="100000",
        btc_available="10", usdt_available="10000", quantity_step="0.000001",
        price_step="0.1", min_quantity="0.000001", min_notional="5",
    )


def analyzer(rating="Buy"):
    def analyze(snap, portfolio, root, now):
        assert type(portfolio) is ManagedPortfolio
        return AnalysisDecision(
            decision_id="opinion-one", rating=rating, analyzed_at=now,
            market_observed_at=snap.quote_at, expires_at=now + timedelta(minutes=15),
            summary="Strict structured result.",
        )
    return analyze


class Gateway:
    def __init__(self, root, *, execute="filled", reconcile="filled"):
        self.root = root
        self.execute_status = execute
        self.reconcile_status = reconcile
        self.calls = []
        self.intent = None

    def __call__(self, request):
        self.calls.append(request.operation)
        if request.operation == "snapshot":
            return GatewayResponse(ok=True, snapshot=snapshot())
        if request.operation == "execute":
            # Submission must observe the actual fsynced on-disk reservation.
            disk = ManagedBook.model_validate_json((self.root / "book.json").read_text())
            assert disk.pending_intent == request.intent
            state = RuntimeState.model_validate_json((self.root / "runtime.json").read_text())
            assert state.pending_cycle_id is not None
            self.intent = request.intent
            if self.execute_status == "timeout":
                raise TimeoutError("secret provider exception that must never escape")
            status = self.execute_status
        else:
            disk = ManagedBook.model_validate_json((self.root / "book.json").read_text())
            self.intent = disk.pending_intent
            assert request.client_order_id == self.intent.client_order_id
            status = self.reconcile_status
        filled = status == "filled"
        return GatewayResponse(ok=True, receipt=ExecutionReceipt(
            client_order_id=self.intent.client_order_id, status=status,
            filled_quantity=self.intent.quantity if filled else "0",
            filled_quote=self.intent.quantity * self.intent.limit_price if filled else "0",
            observed_at=NOW,
        ))


def never(*args):
    pytest.fail("An external dependency must not be invoked")


def test_reservation_precedes_submit_and_fill_persists(tmp_path):
    gateway = Gateway(tmp_path)
    result = run_once(tmp_path, gateway=gateway, analyzer=analyzer(), now=NOW)
    assert result.status == "filled" and result.notify
    assert gateway.calls == ["snapshot", "snapshot", "execute"]
    book = ManagedBook.model_validate_json((tmp_path / "book.json").read_text())
    assert book.pending_intent is None
    assert book.btc_quantity == result.filled_quantity
    assert book.cash_usdt == result.managed_cash_usdt
    assert result.environment == "demo"


def test_same_hour_dedupes_without_model_or_exchange_calls(tmp_path):
    run_once(tmp_path, gateway=Gateway(tmp_path), analyzer=analyzer(), now=NOW)
    result = run_once(tmp_path, gateway=never, analyzer=never, now=NOW + timedelta(minutes=5))
    assert result.status == "already_completed" and not result.notify


def test_timeout_after_reservation_only_reconciles_next_run(tmp_path):
    gateway = Gateway(tmp_path, execute="timeout")
    failed = run_once(tmp_path, gateway=gateway, analyzer=analyzer(), now=NOW)
    assert failed.status == "error" and failed.safe_error_code == "gateway_timeout"
    assert failed.notify
    disk = ManagedBook.model_validate_json((tmp_path / "book.json").read_text())
    assert disk.pending_intent is not None and disk.btc_quantity == 0
    assert "secret" not in failed.model_dump_json()
    gateway.calls.clear()
    recovered = run_once(tmp_path, gateway=gateway, analyzer=never, now=NOW)
    assert gateway.calls == ["reconcile"]
    assert recovered.status == "filled" and recovered.notify
    assert run_once(tmp_path, gateway=never, analyzer=never, now=NOW).status == "already_completed"


def test_unchanged_unknown_blocks_without_notification_spam(tmp_path):
    gateway = Gateway(tmp_path, execute="unknown", reconcile="not_found")
    first = run_once(tmp_path, gateway=gateway, analyzer=analyzer(), now=NOW)
    assert first.status == "pending" and first.notify
    for _unused in range(2):
        gateway.calls.clear()
        again = run_once(tmp_path, gateway=gateway, analyzer=never, now=NOW)
        assert gateway.calls == ["reconcile"]
        assert again.status == "pending" and not again.notify
    assert ManagedBook.model_validate_json((tmp_path / "book.json").read_text()).pending_intent is not None


def test_new_hour_with_pending_only_reconciles_old_order(tmp_path):
    gateway = Gateway(tmp_path, execute="unknown", reconcile="filled")
    run_once(tmp_path, gateway=gateway, analyzer=analyzer(), now=NOW)
    gateway.calls.clear()
    recovered = run_once(tmp_path, gateway=gateway, analyzer=never, now=NOW + timedelta(hours=1))
    assert gateway.calls == ["reconcile"] and recovered.status == "filled"
    runtime = RuntimeState.model_validate_json((tmp_path / "runtime.json").read_text())
    assert runtime.completed_cycle_ids == ["2026-10-26T12:00Z"]


@pytest.mark.parametrize("name", ["book.json", "runtime.json", "control.json"])
def test_corrupt_state_preserved_and_no_dependencies_called(tmp_path, name):
    malformed = '{"never reset this":'
    path = tmp_path / name
    path.write_text(malformed)
    result = run_once(tmp_path, gateway=never, analyzer=never, now=NOW)
    assert result.status == "error" and result.safe_error_code == "state_corrupt"
    assert path.read_text() == malformed


def test_no_order_marks_hour_complete_and_retains_zero_owned_btc(tmp_path):
    gateway = Gateway(tmp_path)
    result = run_once(tmp_path, gateway=gateway, analyzer=analyzer("Hold"), now=NOW)
    assert result.status == "no_order" and not result.notify
    assert gateway.calls == ["snapshot", "snapshot"]
    assert result.managed_btc_quantity == 0
    assert run_once(tmp_path, gateway=never, analyzer=never, now=NOW).status == "already_completed"


def test_disable_prevents_snapshot_analysis_and_reconciliation(tmp_path):
    gateway = Gateway(tmp_path, execute="unknown")
    run_once(tmp_path, gateway=gateway, analyzer=analyzer(), now=NOW)
    assert set_enabled(tmp_path, False).status == "disabled"
    result = run_once(tmp_path, gateway=never, analyzer=never, now=NOW)
    assert result.status == "disabled" and not result.notify
    assert set_enabled(tmp_path, True).status == "enabled"
    assert read_status(tmp_path, now=NOW).order_id is not None


@pytest.mark.parametrize("execution_status", ["filled", "unknown"])
@pytest.mark.parametrize("disabled", [False, True])
def test_missing_established_book_never_reinitializes_cash_or_runs_dependencies(tmp_path, execution_status, disabled):
    gateway = Gateway(tmp_path, execute=execution_status)
    run_once(tmp_path, gateway=gateway, analyzer=analyzer(), now=NOW)
    runtime_before = (tmp_path / "runtime.json").read_bytes()
    (tmp_path / "book.json").unlink()
    if disabled:
        set_enabled(tmp_path, False)

    result = run_once(tmp_path, gateway=never, analyzer=never, now=NOW + timedelta(hours=1))
    assert result.status == "error" and result.safe_error_code == "ledger_missing"
    assert result.managed_cash_usdt is None
    assert not (tmp_path / "book.json").exists()
    assert (tmp_path / "runtime.json").read_bytes() == runtime_before
    assert read_status(tmp_path, now=NOW).safe_error_code == "ledger_missing"


def test_disable_during_analysis_is_immediate_and_leaves_hour_incomplete(tmp_path):
    gateway = Gateway(tmp_path)

    def disabling_analyzer(*args):
        assert set_enabled(tmp_path, False).status == "disabled"
        return analyzer()(*args)

    result = run_once(tmp_path, gateway=gateway, analyzer=disabling_analyzer, now=NOW)
    assert result.status == "disabled"
    assert gateway.calls == ["snapshot"]
    assert json.loads((tmp_path / "control.json").read_text())["enabled"] is False
    book = ManagedBook.model_validate_json((tmp_path / "book.json").read_text())
    assert book.pending_intent is None and not book.processed_cycle_ids
    assert not (tmp_path / "runtime.json").exists()

    assert set_enabled(tmp_path, True).status == "enabled"
    resumed = run_once(tmp_path, gateway=gateway, analyzer=analyzer(), now=NOW)
    assert resumed.status == "filled"


def test_disable_during_refreshed_snapshot_prevents_reservation(tmp_path):
    gateway = Gateway(tmp_path)

    def disabling_gateway(request):
        response = gateway(request)
        if gateway.calls == ["snapshot", "snapshot"]:
            assert set_enabled(tmp_path, False).status == "disabled"
        return response

    result = run_once(tmp_path, gateway=disabling_gateway, analyzer=analyzer(), now=NOW)
    assert result.status == "disabled"
    assert gateway.calls == ["snapshot", "snapshot"]
    assert ManagedBook.model_validate_json((tmp_path / "book.json").read_text()).pending_intent is None


def test_disable_after_reservation_retains_pending_without_submission(tmp_path, monkeypatch):
    import tradingagents.integrations.bybit_demo.runner as runner
    original = runner._atomic_write

    def disable_after_reserve(path, model):
        original(path, model)
        if isinstance(model, ManagedBook) and model.pending_intent is not None:
            assert set_enabled(tmp_path, False).status == "disabled"

    monkeypatch.setattr(runner, "_atomic_write", disable_after_reserve)
    gateway = Gateway(tmp_path, reconcile="not_found")
    result = run_once(tmp_path, gateway=gateway, analyzer=analyzer(), now=NOW)
    assert result.status == "disabled" and result.safe_error_code == "execution_not_submitted"
    assert gateway.calls == ["snapshot", "snapshot"]
    book = ManagedBook.model_validate_json((tmp_path / "book.json").read_text())
    assert book.pending_intent is not None and not book.applied_receipts
    runtime = RuntimeState.model_validate_json((tmp_path / "runtime.json").read_text())
    assert runtime.completed_cycle_ids == []
    assert runtime.pending_cycle_id == "2026-10-26T12:00Z"

    monkeypatch.setattr(runner, "_atomic_write", original)
    set_enabled(tmp_path, True)
    gateway.calls.clear()
    resumed = run_once(tmp_path, gateway=gateway, analyzer=never, now=NOW)
    assert resumed.status == "pending" and gateway.calls == ["reconcile"]


def test_control_corruption_during_analysis_fails_closed(tmp_path):
    gateway = Gateway(tmp_path)

    def corrupting_analyzer(*args):
        (tmp_path / "control.json").write_text('{"enabled":')
        return analyzer()(*args)

    result = run_once(tmp_path, gateway=gateway, analyzer=corrupting_analyzer, now=NOW)
    assert result.status == "error" and result.safe_error_code == "state_corrupt"
    assert gateway.calls == ["snapshot"]
    assert ManagedBook.model_validate_json((tmp_path / "book.json").read_text()).pending_intent is None
    assert (tmp_path / "control.json").read_text() == '{"enabled":'


def test_nonblocking_crossprocess_lock_skips_work(tmp_path):
    with locked(tmp_path / "runtime", wait=False) as acquired:
        assert acquired
        result = run_once(tmp_path, gateway=never, analyzer=never, now=NOW)
    assert result.status == "already_running" and not result.notify


def test_untyped_model_text_never_parsed_or_submitted(tmp_path):
    gateway = Gateway(tmp_path)
    result = run_once(tmp_path, gateway=gateway, analyzer=lambda *args: '{"rating":"Buy"}', now=NOW)
    assert result.status == "error" and result.safe_error_code == "analysis_failed"
    assert gateway.calls == ["snapshot"]
    assert ManagedBook.model_validate_json((tmp_path / "book.json").read_text()).pending_intent is None


def test_wrong_receipt_id_keeps_reserved_order_pending(tmp_path):
    gateway = Gateway(tmp_path)

    def wrong(request):
        response = gateway(request)
        if request.operation == "execute":
            response.receipt.client_order_id = "not-the-reserved-order"
        return response

    result = run_once(tmp_path, gateway=wrong, analyzer=analyzer(), now=NOW)
    assert result.safe_error_code == "gateway_response_mismatch"
    assert ManagedBook.model_validate_json((tmp_path / "book.json").read_text()).pending_intent is not None


def test_expired_after_persist_never_submits_and_stays_reconcilable(tmp_path):
    calls = 0

    def advancing():
        nonlocal calls
        calls += 1
        return NOW if calls < 5 else NOW + timedelta(minutes=3)

    gateway = Gateway(tmp_path)
    result = run_once(tmp_path, gateway=gateway, analyzer=analyzer(), now=advancing)
    assert result.safe_error_code == "intent_expired"
    assert gateway.calls == ["snapshot", "snapshot"]
    assert ManagedBook.model_validate_json((tmp_path / "book.json").read_text()).pending_intent is not None


def test_status_and_cli_are_read_only_and_emit_typed_json(tmp_path, capsys):
    assert read_status(tmp_path, now=NOW).managed_cash_usdt == Decimal("300")
    assert not (tmp_path / "book.json").exists()
    assert main(["--root", str(tmp_path), "status"]) == 0
    assert json.loads(capsys.readouterr().out)["environment"] == "demo"
    assert not (tmp_path / "book.json").exists()


def test_stale_initial_snapshot_never_reaches_model(tmp_path):
    def stale(request):
        return GatewayResponse(ok=True, snapshot=snapshot(NOW - timedelta(seconds=121)))

    result = run_once(tmp_path, gateway=stale, analyzer=never, now=NOW)
    assert result.safe_error_code == "snapshot_not_fresh"


@pytest.fixture
def no_credentials(monkeypatch):
    for key in CREDENTIAL_NAMES:
        monkeypatch.delenv(key, raising=False)


def test_missing_credentials_safe_failure_without_subprocess(tmp_path, monkeypatch, no_credentials):
    monkeypatch.setattr(subprocess, "run", never)
    from tradingagents.integrations.bybit_demo.contracts import GatewayRequest
    with pytest.raises(RunnerFailure, match="demo_credentials_missing"):
        SubprocessGateway(tmp_path)(GatewayRequest(operation="snapshot"))


def test_plaintext_or_extra_reference_keys_rejected(tmp_path, monkeypatch, no_credentials):
    monkeypatch.setattr(subprocess, "run", never)
    path = tmp_path / "credentials.env"
    for invalid in (
        "BYBIT_DEMO_API_KEY=plaintext\nBYBIT_DEMO_API_SECRET=plaintext\n",
        "BYBIT_DEMO_API_KEY=op://vault/item/key\nBYBIT_DEMO_API_SECRET=op://vault/item/secret\nOTHER=op://vault/item/other\n",
    ):
        path.write_text(invalid)
        with pytest.raises(RunnerFailure, match="demo_credentials_invalid"):
            SubprocessGateway(tmp_path)._command(tmp_path / "response.json")


def test_op_reference_file_command_contains_no_reference_values(tmp_path, monkeypatch, no_credentials):
    monkeypatch.setattr("tradingagents.integrations.bybit_demo.runner.shutil.which", lambda name: "op")
    (tmp_path / "credentials.env").write_text(
        "BYBIT_DEMO_API_KEY=op://vault/item/key\nBYBIT_DEMO_API_SECRET=op://vault/item/secret\n")
    command = SubprocessGateway(tmp_path)._command(tmp_path / "response.json")
    assert command[:3] == ["op", "run", "--env-file"]
    assert not any("op://" in argument for argument in command)


@pytest.mark.parametrize("quote", ['"', "'", ""])
def test_reference_vault_and_item_spaces_with_dotenv_quoting(tmp_path, monkeypatch, no_credentials, quote):
    monkeypatch.setattr("tradingagents.integrations.bybit_demo.runner.shutil.which", lambda name: "op")
    (tmp_path / "credentials.env").write_text(
        f"BYBIT_DEMO_API_KEY={quote}op://Demo Vault/Demo Item/API_KEY{quote}\n"
        f"BYBIT_DEMO_API_SECRET={quote}op://Demo Vault/Demo Item/API_SECRET{quote}\n")
    command = SubprocessGateway(tmp_path)._command(tmp_path / "response.json")
    assert command[:3] == ["op", "run", "--env-file"]
    assert not any("op://" in argument for argument in command)


@pytest.mark.parametrize("invalid", [
    "op://vault//key", "op:///item/key", "op://vault/item/", "op://vault/item/key?x=y",
    "op://vault/item/key#fragment", "op://vault/item/key\x00", "op://vault/item/key\t",
    "op://vault/item/key\v", "op://vault/item/key\nextra", "op://vault/${ITEM}/key",
    '"op://vault/item/key', "'op://vault/item/key\"", "plaintext",
])
def test_invalid_reference_shapes_rejected(tmp_path, monkeypatch, no_credentials, invalid):
    monkeypatch.setattr(subprocess, "run", never)
    (tmp_path / "credentials.env").write_text(
        f"BYBIT_DEMO_API_KEY={invalid}\nBYBIT_DEMO_API_SECRET=op://vault/item/secret\n")
    with pytest.raises(RunnerFailure, match="demo_credentials_invalid"):
        SubprocessGateway(tmp_path)._command(tmp_path / "response.json")


def test_duplicate_reference_names_rejected(tmp_path, monkeypatch, no_credentials):
    monkeypatch.setattr(subprocess, "run", never)
    (tmp_path / "credentials.env").write_text(
        "BYBIT_DEMO_API_KEY=op://vault/item/key\n"
        "BYBIT_DEMO_API_KEY=op://vault/item/other\n"
        "BYBIT_DEMO_API_SECRET=op://vault/item/secret\n")
    with pytest.raises(RunnerFailure, match="demo_credentials_invalid"):
        SubprocessGateway(tmp_path)._command(tmp_path / "response.json")


def test_gateway_uses_response_file_only_not_free_text_stdout(tmp_path, monkeypatch, no_credentials):
    from tradingagents.integrations.bybit_demo.contracts import GatewayRequest
    for key in CREDENTIAL_NAMES:
        monkeypatch.setenv(key, "synthetic-test-value")

    def fake_run(command, **kwargs):
        assert kwargs["stdout"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.DEVNULL
        assert kwargs["timeout"] == 180
        request = GatewayRequest.model_validate_json(kwargs["input"])
        assert request.operation == "snapshot"
        response_path = Path(command[command.index("--response-file") + 1])
        assert response_path.is_absolute()
        # A response wrapped in arbitrary native output must fail strict parsing.
        response_path.write_text("debug text\n" + GatewayResponse(ok=True, snapshot=snapshot()).model_dump_json())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RunnerFailure, match="gateway_protocol_error"):
        SubprocessGateway(tmp_path)(GatewayRequest(operation="snapshot"))


def test_atomic_reservation_failure_never_calls_execute(tmp_path, monkeypatch):
    import tradingagents.integrations.bybit_demo.runner as runner
    original = runner._atomic_write

    def fail_reservation(path, model):
        if isinstance(model, ManagedBook) and model.pending_intent is not None:
            raise OSError("private local error")
        original(path, model)

    monkeypatch.setattr(runner, "_atomic_write", fail_reservation)
    gateway = Gateway(tmp_path)
    result = run_once(tmp_path, gateway=gateway, analyzer=analyzer(), now=NOW)
    assert result.status == "error"
    assert gateway.calls == ["snapshot", "snapshot"]
    assert ManagedBook.model_validate_json((tmp_path / "book.json").read_text()).pending_intent is None
