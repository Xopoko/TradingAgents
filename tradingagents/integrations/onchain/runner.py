"""One restart-safe TradingAgents cycle on a strictly local EVM fork."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field

from tradingagents.agents.schemas import PortfolioRating
from tradingagents.dataflows.files import locked

from .contracts import AnalysisDecision, Contract, GatewayRequest, GatewayResponse, ManagedPortfolio
from .policy import (
    ManagedBook,
    PolicyBlocked,
    apply_receipt,
    bind_book,
    build_intent,
    plan_quote_request,
    reserve_intent,
)
from .storage import StateError, read_model, write_model

DEFAULT_ROOT = Path.home() / ".tradingagents" / "onchain-fork"
DEFAULT_RPC = "http://127.0.0.1:18547"


class RunResult(Contract):
    environment: Literal["local-fork"] = "local-fork"
    status: Literal[
        "status", "enabled", "disabled", "already_running", "already_completed",
        "no_order", "confirmed", "reverted", "pending", "error",
    ]
    cycle_id: str
    rating: PortfolioRating | None = None
    notify: bool = Field(default=False, strict=True)
    safe_error_code: str | None = None
    intent_id: str | None = None
    cash_units: int | None = Field(default=None, ge=0, strict=True)
    btc_units: int | None = Field(default=None, ge=0, strict=True)


class Control(Contract):
    enabled: bool = Field(default=True, strict=True)


class Runtime(Contract):
    completed_cycles: list[str] = Field(default_factory=list)
    pending_cycle: str | None = None
    last_decision: AnalysisDecision | None = None
    last_result: RunResult | None = None
    notification_signature: str | None = None


class RunnerError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class SubprocessGateway:
    """The SDK runs in a separately provisioned Python environment.

    stdout/stderr are never treated as a protocol or echoed. Worker identity,
    nonce and transaction journals remain under this same persistent root.
    """

    def __init__(self, root: Path, *, rpc_url: str = DEFAULT_RPC,
                 worker_python: Path | str | None = None, timeout_seconds: int = 300):
        self.root = Path(root).resolve()
        self.rpc_url = rpc_url
        self.worker_python = str(worker_python or sys.executable)
        self.timeout_seconds = timeout_seconds

    def __call__(self, request: GatewayRequest) -> GatewayResponse:
        request = GatewayRequest.model_validate(request.model_dump())
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".worker-response-", dir=self.root) as directory:
            response_file = Path(directory).resolve() / "response.json"
            command = [
                self.worker_python, "-m", "tradingagents.integrations.onchain.engine",
                "--rpc-url", self.rpc_url, "--state-root", str(self.root / "worker"),
                "--response-file", str(response_file),
            ]
            try:
                subprocess.run(
                    command, input=request.model_dump_json().encode("utf-8"),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=self.timeout_seconds, check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            except subprocess.TimeoutExpired as exc:
                raise RunnerError("worker_timeout") from exc
            except OSError as exc:
                raise RunnerError("worker_unavailable") from exc
            try:
                if response_file.stat().st_size > 131072:
                    raise ValueError("oversized_response")
                return GatewayResponse.model_validate_json(response_file.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RunnerError("worker_protocol_error") from exc


def _now() -> datetime:
    return datetime.now(UTC)


def _cycle(now: datetime) -> str:
    if now.utcoffset() is None:
        raise RunnerError("invalid_clock")
    return now.astimezone(UTC).strftime("%Y-%m-%dT%H:00Z")


def _book(root: Path) -> ManagedBook:
    if not (root / "book.json").exists() and (root / "runtime.json").exists():
        raise RunnerError("ledger_missing")
    return read_model(root / "book.json", ManagedBook, ManagedBook())


def _enabled(root: Path) -> bool:
    return read_model(root / "control.json", Control, Control()).enabled


def _result(status: str, cycle: str, book: ManagedBook | None = None, **fields) -> RunResult:
    if book is not None:
        fields.update(cash_units=book.cash_units, btc_units=book.btc_units)
    return RunResult(status=status, cycle_id=cycle, **fields)


def _request(gateway, request: GatewayRequest) -> GatewayResponse:
    try:
        response = gateway(request)
        if not isinstance(response, GatewayResponse):
            raise RunnerError("worker_protocol_error")
        response = GatewayResponse.model_validate(response.model_dump())
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError("worker_failed") from exc
    if not response.ok:
        # Do not forward provider-controlled exception/error text to the terminal.
        raise RunnerError("worker_failed")
    expected = "snapshot" if request.operation == "snapshot" else "quote" if request.operation == "quote" else "receipt"
    payload = getattr(response, expected)
    if payload is None:
        raise RunnerError("worker_protocol_error")
    if expected == "receipt":
        key = request.intent.intent_id if request.intent is not None else request.intent_id
        if payload.intent_id != key:
            raise RunnerError("worker_identity_mismatch")
    return response


def _persist(root: Path, runtime: Runtime, result: RunResult, *,
             complete: str | None = None, pending: str | None = None,
             decision: AnalysisDecision | None = None, changed: bool = False) -> RunResult:
    signature = hashlib.sha256(f"{result.safe_error_code}|{result.intent_id}".encode()).hexdigest() if result.safe_error_code else None
    result = result.model_copy(update={"notify": changed or (signature is not None and signature != runtime.notification_signature)})
    completed = list(runtime.completed_cycles)
    if complete is not None and complete not in completed:
        completed.append(complete)
    write_model(root / "runtime.json", Runtime(
        completed_cycles=completed, pending_cycle=pending,
        last_decision=decision or runtime.last_decision, last_result=result,
        notification_signature=signature,
    ))
    return result


def _analyze(snapshot, portfolio, root, now):
    from .analysis import analyze

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return analyze(snapshot, portfolio, root, now)


def run_once(root: Path = DEFAULT_ROOT, *, gateway=None, analyzer=None,
             now: datetime | Callable[[], datetime] | None = None) -> RunResult:
    root = Path(root).resolve()
    clock = _now if now is None else now if callable(now) else lambda: now
    book: ManagedBook | None = None
    runtime = Runtime()
    cycle = "invalid"
    try:
        cycle = _cycle(clock())
        root.mkdir(parents=True, exist_ok=True)
        with locked(root / "runtime", wait=False) as acquired:
            if not acquired:
                return _result("already_running", cycle)
            try:
                runtime = read_model(root / "runtime.json", Runtime, Runtime())
                book = _book(root)
                if not _enabled(root):
                    return _result("disabled", cycle, book)
                if not (root / "book.json").exists():
                    write_model(root / "book.json", book)
                gateway = gateway or SubprocessGateway(root)
                analyzer = analyzer or _analyze
                if book.pending_intent is not None:
                    intent = book.pending_intent
                    receipt = _request(gateway, GatewayRequest(operation="reconcile", intent_id=intent.intent_id)).receipt
                    updated = apply_receipt(book, receipt)
                    changed = updated.cash_units != book.cash_units or updated.btc_units != book.btc_units
                    write_model(root / "book.json", updated)
                    book = updated
                    pending = book.pending_intent is not None
                    return _persist(root, runtime, _result(
                        "pending" if pending else receipt.status, cycle, book, intent_id=intent.intent_id,
                        safe_error_code="execution_unconfirmed" if pending else "swap_reverted" if receipt.status == "reverted" else None,
                    ), complete=runtime.pending_cycle if not pending else None,
                       pending=runtime.pending_cycle if pending else None, changed=changed)
                if cycle in runtime.completed_cycles or cycle in book.processed_cycle_ids:
                    return _result("already_completed", cycle, book)
                initial = _request(gateway, GatewayRequest(operation="snapshot")).snapshot
                book = bind_book(book, initial)
                write_model(root / "book.json", book)
                try:
                    portfolio = ManagedPortfolio(
                        cash_units=book.cash_units, btc_units=book.btc_units,
                        capital_limit_usdt=book.capital_limit_usdt,
                    )
                    decision = analyzer(initial, portfolio, root, clock())
                    if not isinstance(decision, AnalysisDecision):
                        raise ValueError("typed_decision_required")
                    decision = AnalysisDecision.model_validate(decision.model_dump())
                except Exception as exc:
                    raise RunnerError("analysis_failed") from exc
                if not _enabled(root):
                    return _result("disabled", cycle, book, rating=decision.rating)
                fresh = _request(gateway, GatewayRequest(operation="snapshot")).snapshot
                book = bind_book(book, fresh)
                quote_request = plan_quote_request(decision, fresh, book, clock(), cycle)
                if not _enabled(root):
                    return _result("disabled", cycle, book, rating=decision.rating)
                if quote_request is None:
                    return _persist(root, runtime, _result("no_order", cycle, book, rating=decision.rating),
                                    complete=cycle, decision=decision)
                quote = _request(gateway, GatewayRequest(operation="quote", quote_request=quote_request)).quote
                intent = build_intent(decision, fresh, book, quote, clock(), cycle)
                if not _enabled(root):
                    return _result("disabled", cycle, book, rating=decision.rating)
                runtime = runtime.model_copy(update={"pending_cycle": cycle, "last_decision": decision})
                write_model(root / "runtime.json", runtime)
                book = reserve_intent(book, intent)
                write_model(root / "book.json", book)
                saved = read_model(root / "book.json", ManagedBook)
                if saved is None or saved.pending_intent != intent:
                    raise RunnerError("reservation_not_durable")
                if not _enabled(root):
                    return _persist(root, runtime, _result("disabled", cycle, book,
                                    intent_id=intent.intent_id, safe_error_code="execution_not_submitted"),
                                    pending=cycle, decision=decision)
                if clock() >= intent.expires_at:
                    raise RunnerError("intent_expired")
                receipt = _request(gateway, GatewayRequest(operation="execute", intent=intent)).receipt
                updated = apply_receipt(book, receipt)
                changed = updated.cash_units != book.cash_units or updated.btc_units != book.btc_units
                write_model(root / "book.json", updated)
                book = updated
                pending = book.pending_intent is not None
                return _persist(root, runtime, _result(
                    "pending" if pending else receipt.status, cycle, book, rating=decision.rating,
                    intent_id=intent.intent_id,
                    safe_error_code="execution_unconfirmed" if pending else "swap_reverted" if receipt.status == "reverted" else None,
                ), complete=cycle if not pending else None, pending=cycle if pending else None,
                   decision=decision, changed=changed)
            except Exception as exc:
                code = exc.code if isinstance(exc, RunnerError) else "state_corrupt" if isinstance(exc, StateError) else "unsafe_policy" if isinstance(exc, PolicyBlocked) else "cycle_failed"
                result = _result("error", cycle, book, safe_error_code=code,
                                 intent_id=book.pending_intent.intent_id if book and book.pending_intent else None)
                if code in {"state_corrupt", "ledger_missing"}:
                    return result.model_copy(update={"notify": True})
                return _persist(root, runtime, result, pending=runtime.pending_cycle)
    except (OSError, StateError):
        return _result("error", cycle, book, safe_error_code="state_io_failed", notify=True)
    except RunnerError as exc:
        return _result("error", cycle, book, safe_error_code=exc.code, notify=True)


def status(root: Path = DEFAULT_ROOT) -> RunResult:
    try:
        root = Path(root).resolve()
        book = _book(root)
        runtime = read_model(root / "runtime.json", Runtime, Runtime())
        return _result("status" if _enabled(root) else "disabled", _cycle(_now()), book,
                       rating=runtime.last_decision.rating if runtime.last_decision else None,
                       intent_id=book.pending_intent.intent_id if book.pending_intent else None)
    except RunnerError as exc:
        return _result("error", _cycle(_now()), safe_error_code=exc.code)
    except StateError:
        return _result("error", _cycle(_now()), safe_error_code="state_corrupt")


def set_enabled(root: Path, enabled: bool) -> RunResult:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / "control", wait=True) as acquired:
        if not acquired:
            return _result("already_running", _cycle(_now()))
        write_model(root / "control.json", Control(enabled=enabled))
    return _result("enabled" if enabled else "disabled", _cycle(_now()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TradingAgents isolated WBTC/USDT fork trader")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--rpc-url", default=DEFAULT_RPC)
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument("--fixture-rating", choices=[x.value for x in PortfolioRating])
    parser.add_argument("command", choices=("status", "once", "enable", "disable", "snapshot"))
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            result = status(args.root)
        elif args.command in {"enable", "disable"}:
            result = set_enabled(args.root, args.command == "enable")
        else:
            gateway = SubprocessGateway(args.root, rpc_url=args.rpc_url, worker_python=args.worker_python)
            if args.command == "snapshot":
                response = gateway(GatewayRequest(operation="snapshot"))
                print(response.model_dump_json())
                return 0 if response.ok else 1
            analyzer = None
            if args.fixture_rating:
                def fixture(snapshot, portfolio, root, now):
                    return AnalysisDecision(
                        decision_id=str(uuid4()), rating=PortfolioRating(args.fixture_rating),
                        analyzed_at=now, expires_at=now + timedelta(minutes=15),
                        summary="Explicit local-fork execution fixture; no model recommendation.", source="fixture",
                    )
                analyzer = fixture
            result = run_once(args.root, gateway=gateway, analyzer=analyzer)
    except Exception:
        result = _result("error", _cycle(_now()), safe_error_code="command_failed", notify=True)
    print(result.model_dump_json())
    return 1 if result.status == "error" else 0
