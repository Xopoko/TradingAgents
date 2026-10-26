"""One bounded, restart-safe hourly demo cycle; scheduling is external."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field

from tradingagents.agents.schemas import PortfolioRating
from tradingagents.dataflows.files import locked

from .contracts import (
    AnalysisDecision,
    Contract,
    GatewayRequest,
    GatewayResponse,
    ManagedPortfolio,
)
from .policy import ManagedBook, PolicyBlocked, apply_receipt, plan_order, reserve_intent

DEFAULT_ROOT = Path.home() / ".tradingagents" / "bybit-demo"
CREDENTIAL_NAMES = frozenset({"BYBIT_DEMO_API_KEY", "BYBIT_DEMO_API_SECRET"})


class RunResult(Contract):
    status: Literal[
        "disabled", "already_running", "already_completed", "no_order", "pending",
        "filled", "terminal", "error", "status", "enabled",
    ]
    environment: Literal["demo"] = "demo"
    cycle_id: str
    rating: PortfolioRating | None = None
    notify: bool = False
    safe_error_code: str | None = None
    order_id: str | None = None
    filled_quantity: Decimal = Field(default=Decimal("0"), ge=0, allow_inf_nan=False)
    managed_cash_usdt: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    managed_btc_quantity: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)


class RuntimeState(Contract):
    version: Literal[1] = 1
    completed_cycle_ids: list[str] = Field(default_factory=list)
    pending_cycle_id: str | None = None
    last_decision: AnalysisDecision | None = None
    last_result: RunResult | None = None
    last_notification_signature: str | None = None


class ControlState(Contract):
    version: Literal[1] = 1
    enabled: bool = Field(default=True, strict=True)


class RunnerFailure(ValueError):
    """Carries a fixed safe code, never subprocess/provider exception text."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _atomic_write(path: Path, model: Contract) -> None:
    """Replace whole JSON after fsync; unlike cache helpers, failures propagate."""
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}-", suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            stream.write(model.model_dump_json(indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _read(path: Path, model, default):
    if not path.exists():
        return default
    try:
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("state too large")
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise RunnerFailure("state_corrupt") from exc


def _read_book(root: Path) -> ManagedBook:
    if not (root / "book.json").exists() and (root / "runtime.json").exists():
        # Runtime progress proves this is an established ledger. Recreating its
        # cash allocation would lose ownership and pending-order evidence.
        raise RunnerFailure("ledger_missing")
    return _read(root / "book.json", ManagedBook, ManagedBook())


def _clock(now: datetime | Callable[[], datetime] | None) -> Callable[[], datetime]:
    clock = (lambda: datetime.now(UTC)) if now is None else now if callable(now) else lambda: now

    def aware() -> datetime:
        value = clock()
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise RunnerFailure("invalid_clock")
        return value.astimezone(UTC)

    return aware


def _cycle(now: datetime) -> str:
    return now.astimezone(UTC).strftime("%Y-%m-%dT%H:00Z")


class SubprocessGateway:
    """Private stdio and a strict response file isolate native adapter output.

    Credentials are inherited, or injected by official ``op run`` from an
    explicitly provisioned reference-only file. No discovery, authentication,
    secret retrieval, or credential creation is performed here.
    """

    def __init__(self, root: Path, *, timeout_seconds: int = 180):
        self.root = Path(root).resolve()
        self.timeout_seconds = timeout_seconds

    def _command(self, response_path: Path) -> list[str]:
        command = [sys.executable, "-m", "tradingagents.integrations.bybit_demo.nautilus_executor",
                   "--response-file", str(response_path)]
        present = {key for key in CREDENTIAL_NAMES if os.environ.get(key)}
        if present == CREDENTIAL_NAMES:
            if any(os.environ[key].startswith("op://") for key in CREDENTIAL_NAMES):
                raise RunnerFailure("demo_credentials_invalid")
            return command
        # Mixing a partial plaintext environment with injected references is
        # ambiguous; prepare either complete route, never silently combine them.
        if present:
            raise RunnerFailure("demo_credentials_missing")
        references = self.root / "credentials.env"
        if not references.is_file():
            raise RunnerFailure("demo_credentials_missing")
        try:
            if references.stat().st_size > 8192:
                raise ValueError("oversized credential reference file")
            entries: dict[str, str] = {}
            for line in references.read_text(encoding="utf-8").split("\n"):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                if any(ord(character) < 32 or ord(character) == 127 for character in line):
                    raise ValueError("control character in credential reference")
                key, separator, value = line.partition("=")
                key, value = key.strip(), value.strip()
                # Ordinary dotenv quoting is permitted. Strip exactly one
                # matching pair; do not expand variables or interpret escapes.
                if value.startswith(("'", '"')):
                    if len(value) < 2 or value[-1] != value[0]:
                        raise ValueError("unbalanced credential reference quotes")
                    value = value[1:-1]
                segments = value[5:].split("/") if value.startswith("op://") else []
                if (not separator or key not in CREDENTIAL_NAMES or key in entries
                        or len(segments) not in (3, 4) or any(not part.strip() for part in segments)
                        or any(character in value for character in ('"', "'", "?", "#", "\\", "$"))):
                    raise ValueError("invalid credential reference")
                entries[key] = value
            if entries.keys() != CREDENTIAL_NAMES:
                raise ValueError("missing reference")
        except (OSError, ValueError) as exc:
            raise RunnerFailure("demo_credentials_invalid") from exc
        executable = shutil.which("op")
        if executable is None:
            raise RunnerFailure("credential_injector_unavailable")
        return [executable, "run", "--env-file", str(references), "--", *command]

    def __call__(self, request: GatewayRequest) -> GatewayResponse:
        request = GatewayRequest.model_validate(request.model_dump())
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".gateway-", dir=self.root) as directory:
            response_path = Path(directory).resolve() / "response.json"
            command = self._command(response_path)
            try:
                # Native stdout/stderr may contain private payloads: discard them
                # without decoding, logging, or using them as a fallback protocol.
                subprocess.run(
                    command, input=request.model_dump_json().encode("utf-8"),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=self.timeout_seconds, check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            except subprocess.TimeoutExpired as exc:
                raise RunnerFailure("gateway_timeout") from exc
            except OSError as exc:
                raise RunnerFailure("gateway_unavailable") from exc
            try:
                if response_path.stat().st_size > 131072:
                    raise ValueError("response too large")
                return GatewayResponse.model_validate_json(response_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RunnerFailure("gateway_protocol_error") from exc


def _request(gateway, request: GatewayRequest) -> GatewayResponse:
    try:
        response = gateway(request)
        if not isinstance(response, GatewayResponse):
            raise RunnerFailure("gateway_protocol_error")
        response = GatewayResponse.model_validate(response.model_dump())
    except RunnerFailure:
        raise
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        raise RunnerFailure("gateway_timeout") from exc
    except Exception as exc:
        raise RunnerFailure("gateway_failed") from exc
    if not response.ok:
        # Even a malformed/provider-controlled error string cannot enter logs.
        raise RunnerFailure("gateway_failed")
    if request.operation == "snapshot":
        if response.snapshot is None:
            raise RunnerFailure("gateway_protocol_error")
    else:
        key = request.intent.client_order_id if request.intent is not None else request.client_order_id
        if response.receipt is None or response.receipt.client_order_id != key:
            raise RunnerFailure("gateway_response_mismatch")
    return response


def _result(status: str, cycle: str, book: ManagedBook | None = None, **fields) -> RunResult:
    if book is not None:
        fields.update(managed_cash_usdt=book.cash_usdt, managed_btc_quantity=book.btc_quantity)
    return RunResult(status=status, cycle_id=cycle, **fields)


def _analysis_snapshot(snapshot, book: ManagedBook, now: datetime) -> None:
    """Reject unusable inputs before spending time on a model analysis."""
    for stamp in (snapshot.observed_at, snapshot.quote_at):
        if not timedelta(0) <= now - stamp <= timedelta(seconds=120):
            raise RunnerFailure("snapshot_not_fresh")
    if snapshot.btc_borrowed or snapshot.usdt_borrowed:
        raise RunnerFailure("borrowed_balance_forbidden")
    if snapshot.open_order_ids:
        raise RunnerFailure("preexisting_open_orders")
    if book.cash_usdt > snapshot.usdt_available or book.btc_quantity > snapshot.btc_available:
        raise RunnerFailure("managed_funds_unavailable")


def _persist_result(root: Path, runtime: RuntimeState, result: RunResult,
                    *, complete: str | None = None, pending_cycle: str | None = None,
                    decision: AnalysisDecision | None = None, new_fill: bool = False) -> RunResult:
    signature = None
    if result.safe_error_code:
        signature = hashlib.sha256(json.dumps({
            "error": result.safe_error_code, "order": result.order_id,
        }, sort_keys=True).encode()).hexdigest()
    notify = new_fill or (signature is not None and signature != runtime.last_notification_signature)
    result = RunResult.model_validate({**result.model_dump(), "notify": notify})
    completed = list(runtime.completed_cycle_ids)
    if complete is not None and complete not in completed:
        completed.append(complete)
    # UTC hourly cycles stay monotonic in normal operation. Retain all completed
    # IDs so an old process clock cannot replay a historic no-order analysis.
    updated = RuntimeState(
        completed_cycle_ids=completed, pending_cycle_id=pending_cycle,
        last_decision=decision or runtime.last_decision, last_result=result,
        last_notification_signature=signature,
    )
    _atomic_write(root / "runtime.json", updated)
    return result


def _default_analyzer(snapshot, portfolio, root, now):
    from .analysis import analyze

    # Graph/debug text is not part of the CLI protocol. Reports are saved by the
    # graph in its isolated state root; ordinary stdout/stderr stay private.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return analyze(snapshot, portfolio, root, now)


def run_once(
    root: Path = DEFAULT_ROOT, *, gateway=None, analyzer=None,
    now: datetime | Callable[[], datetime] | None = None,
) -> RunResult:
    """Perform one cycle, or only reconcile a previously reserved intent.

    Test doubles use ``gateway(GatewayRequest) -> GatewayResponse`` and
    ``analyzer(snapshot, ManagedPortfolio, root, aware_now) -> AnalysisDecision``.
    Neither dependency is called when disabled, locked, corrupt, or deduplicated.
    """
    root = Path(root).resolve()
    clock = _clock(now)
    try:
        cycle = _cycle(clock())
    except RunnerFailure as exc:
        return _result("error", "invalid", safe_error_code=exc.code, notify=True)
    book: ManagedBook | None = None
    runtime = RuntimeState()
    try:
        root.mkdir(parents=True, exist_ok=True)
        with locked(root / "runtime", wait=False) as acquired:
            if not acquired:
                return _result("already_running", cycle)
            try:
                control = _read(root / "control.json", ControlState, ControlState())
                runtime = _read(root / "runtime.json", RuntimeState, RuntimeState())
                book = _read_book(root)
                if not control.enabled:
                    return _result("disabled", cycle, book)
                if not (root / "book.json").exists():
                    _atomic_write(root / "book.json", book)
                gateway = gateway or SubprocessGateway(root)
                analyzer = analyzer or _default_analyzer

                if book.pending_intent is not None:
                    intent = book.pending_intent
                    response = _request(gateway, GatewayRequest(
                        operation="reconcile", client_order_id=intent.client_order_id,
                    ))
                    updated = apply_receipt(book, response.receipt)
                    changed = updated.btc_quantity != book.btc_quantity or updated.cash_usdt != book.cash_usdt
                    _atomic_write(root / "book.json", updated)
                    book = updated
                    pending_cycle = runtime.pending_cycle_id
                    status = "pending" if book.pending_intent else "filled" if response.receipt.filled_quantity else "terminal"
                    result = _result(status, cycle, book, order_id=intent.client_order_id,
                                     filled_quantity=response.receipt.filled_quantity,
                                     safe_error_code="execution_unconfirmed" if book.pending_intent else
                                     "order_rejected" if response.receipt.status == "rejected" else None)
                    return _persist_result(root, runtime, result,
                                           complete=pending_cycle if not book.pending_intent else None,
                                           pending_cycle=pending_cycle if book.pending_intent else None,
                                           new_fill=changed)
                if cycle in runtime.completed_cycle_ids or cycle in book.processed_cycle_ids:
                    return _result("already_completed", cycle, book)

                first = _request(gateway, GatewayRequest(operation="snapshot")).snapshot
                _analysis_snapshot(first, book, clock())
                portfolio = ManagedPortfolio(cash_usdt=book.cash_usdt, btc_quantity=book.btc_quantity,
                                             capital_limit_usdt=book.capital_limit_usdt)
                try:
                    decision = analyzer(first, portfolio, root, clock())
                    if not isinstance(decision, AnalysisDecision):
                        raise ValueError("analysis must be a typed decision")
                    decision = AnalysisDecision.model_validate(decision.model_dump())
                except Exception as exc:
                    raise RunnerFailure("analysis_failed") from exc
                if not _read(root / "control.json", ControlState, ControlState()).enabled:
                    return _result("disabled", cycle, book, rating=decision.rating)
                fresh = _request(gateway, GatewayRequest(operation="snapshot")).snapshot
                intent = plan_order(decision, fresh, book, clock(), cycle)
                if not _read(root / "control.json", ControlState, ControlState()).enabled:
                    return _result("disabled", cycle, book, rating=decision.rating)
                if intent is None:
                    return _persist_result(root, runtime, _result("no_order", cycle, book, rating=decision.rating),
                                           complete=cycle, decision=decision)
                # Persist contextual progress first, then the authoritative book.
                # Either crash boundary prevents an unsafe repeat submission.
                runtime = RuntimeState.model_validate({**runtime.model_dump(),
                    "pending_cycle_id": cycle, "last_decision": decision})
                _atomic_write(root / "runtime.json", runtime)
                book = reserve_intent(book, intent, cycle)
                _atomic_write(root / "book.json", book)
                # Verify the exact persisted reservation, including its expiry,
                # before handing it to the only component allowed to submit.
                persisted = _read(root / "book.json", ManagedBook, None)
                if persisted is None or persisted.pending_intent != intent:
                    raise RunnerFailure("reservation_not_durable")
                if clock() >= intent.expires_at:
                    raise RunnerFailure("intent_expired")
                if not _read(root / "control.json", ControlState, ControlState()).enabled:
                    # Preserve the durable reservation. Even a local refusal is
                    # not a venue receipt and cannot authorize automatic resend.
                    return _persist_result(root, runtime, _result(
                        "disabled", cycle, book, rating=decision.rating,
                        order_id=intent.client_order_id, safe_error_code="execution_not_submitted",
                    ), pending_cycle=cycle, decision=decision)
                response = _request(gateway, GatewayRequest(operation="execute", intent=intent))
                updated = apply_receipt(book, response.receipt)
                changed = updated.cash_usdt != book.cash_usdt or updated.btc_quantity != book.btc_quantity
                _atomic_write(root / "book.json", updated)
                book = updated
                status = "pending" if book.pending_intent else "filled" if response.receipt.filled_quantity else "terminal"
                result = _result(status, cycle, book, rating=decision.rating,
                                 order_id=intent.client_order_id, filled_quantity=response.receipt.filled_quantity,
                                 safe_error_code="execution_unconfirmed" if book.pending_intent else
                                 "order_rejected" if response.receipt.status == "rejected" else None)
                return _persist_result(root, runtime, result,
                                       complete=cycle if not book.pending_intent else None,
                                       pending_cycle=cycle if book.pending_intent else None,
                                       decision=decision, new_fill=changed)
            except Exception as exc:
                code = exc.code if isinstance(exc, RunnerFailure) else "unsafe_policy" if isinstance(exc, PolicyBlocked) else "cycle_failed"
                result = _result("error", cycle, book, safe_error_code=code,
                                 order_id=book.pending_intent.client_order_id if book and book.pending_intent else None)
                # Never overwrite malformed state with an empty/default ledger.
                if code in {"state_corrupt", "ledger_missing"}:
                    return result.model_copy(update={"notify": True})
                try:
                    return _persist_result(root, runtime, result, pending_cycle=runtime.pending_cycle_id)
                except OSError:
                    return result.model_copy(update={"notify": True, "safe_error_code": "state_write_failed"})
    except OSError:
        return _result("error", cycle, book, safe_error_code="state_io_failed", notify=True)


def read_status(root: Path = DEFAULT_ROOT, *, now: datetime | None = None) -> RunResult:
    """Read local state without initializing it or touching exchange/model APIs."""
    root = Path(root).resolve()
    cycle = _cycle(_clock(now)())
    try:
        control = _read(root / "control.json", ControlState, ControlState())
        runtime = _read(root / "runtime.json", RuntimeState, RuntimeState())
        book = _read_book(root)
        return _result("status" if control.enabled else "disabled", cycle, book,
                       order_id=book.pending_intent.client_order_id if book.pending_intent else None,
                       rating=runtime.last_decision.rating if runtime.last_decision else None)
    except RunnerFailure as exc:
        return _result("error", cycle, safe_error_code=exc.code)


def set_enabled(root: Path, enabled: bool) -> RunResult:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cycle = _cycle(datetime.now(UTC))
    # An active cycle owns the runtime lock through potentially long analysis.
    # The independent short control lock lets disable take effect meanwhile.
    with locked(root / "control", wait=True) as acquired:
        if not acquired:
            return _result("already_running", cycle)
        _atomic_write(root / "control.json", ControlState(enabled=enabled))
    return _result("enabled" if enabled else "disabled", cycle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bybit DEMO BTC spot hourly monitor")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("command", choices=("once", "status", "disable", "enable"))
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "once":
            result = run_once(arguments.root)
        elif arguments.command == "status":
            result = read_status(arguments.root)
        else:
            result = set_enabled(arguments.root, arguments.command == "enable")
    except Exception:
        result = _result("error", _cycle(datetime.now(UTC)), safe_error_code="state_io_failed")
    print(result.model_dump_json())
    return 1 if result.status == "error" else 0
