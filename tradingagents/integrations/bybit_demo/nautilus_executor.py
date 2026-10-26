"""Fail-closed, one-order Bybit DEMO gateway using NautilusTrader 1.231.0.

All order submission passes through the Nautilus risk/execution engines and the
official Bybit adapter. Read-only order reconciliation has a typed DEMO REST
fallback for the pinned SDK's incompatible smpGroup response field. Account
reads remain native. No credentials or private venue payloads are
included in receipts. The caller must durably record intent BEFORE execute_once;
after an unknown result it must use reconcile, never submit the intent again.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import re
import sys
import tempfile
import time
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from importlib.metadata import version
from pathlib import Path
from typing import Any

from tradingagents.integrations.bybit_demo.contracts import (
    ExecutionReceipt,
    GatewayRequest,
    GatewayResponse,
    OrderIntent,
    SpotSnapshot,
)
from tradingagents.integrations.bybit_demo.rest_reconciliation import reconcile_demo_order

NAUTILUS_VERSION = "1.231.0"
INSTRUMENT_ID = "BTCUSDT-SPOT.BYBIT"
DEMO_HTTP = "https://api-demo.bybit.com"
DEMO_PRIVATE_WS = "wss://stream-demo.bybit.com/v5/private"
# DEMO does not implement WS trading. Pin its unused URL away from mainnet too.
DEMO_TRADE_WS = "wss://stream-demo.bybit.com/v5/trade"
PUBLIC_SPOT_WS = "wss://stream.bybit.com/v5/public/spot"
PUBLIC_HTTP = "https://api.bybit.com"


class DemoBoundaryError(ValueError):
    """A fixed demo/spot/cash boundary was violated."""


def decimal_value(value: Any, *, nonnegative: bool = True) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise DemoBoundaryError("invalid_decimal") from None
    if not result.is_finite() or (nonnegative and result < 0):
        raise DemoBoundaryError("invalid_decimal")
    return result


def validate_client_order_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,36}", value):
        raise DemoBoundaryError("invalid_client_order_id")
    return value


def validate_environment(environ: dict[str, str] | os._Environ[str]) -> None:
    """No URL/proxy/environment input can redirect the fixed integration."""
    forbidden = (
        "BYBIT_BASE_URL", "BYBIT_BASE_URL_HTTP", "BYBIT_BASE_URL_WS_PRIVATE",
        "BYBIT_BASE_URL_WS_TRADE", "BYBIT_HTTP_URL", "BYBIT_WS_URL",
        "BYBIT_PROXY_URL", "BYBIT_ENVIRONMENT", "BYBIT_TESTNET", "BYBIT_DEMO",
    )
    if any(environ.get(key) for key in forbidden):
        raise DemoBoundaryError("bybit_environment_override_forbidden")


def validate_intent(intent: OrderIntent, snapshot: SpotSnapshot) -> None:
    """Enforce no borrowing and enforce venue precision before engine submission."""
    if snapshot.environment != "demo" or snapshot.symbol != "BTCUSDT":
        raise DemoBoundaryError("wrong_snapshot_boundary")
    if intent.environment != "demo" or intent.symbol != "BTCUSDT" or intent.is_leverage is not False:
        raise DemoBoundaryError("wrong_intent_boundary")
    if intent.order_type != "LIMIT" or intent.time_in_force != "IOC":
        raise DemoBoundaryError("unsafe_order_type")
    if intent.expires_at <= datetime.now(UTC):
        raise DemoBoundaryError("intent_expired")
    if intent.quantity * intent.limit_price > intent.max_notional_usdt:
        raise DemoBoundaryError("intent_notional_limit")
    if snapshot.btc_borrowed != 0 or snapshot.usdt_borrowed != 0:
        raise DemoBoundaryError("existing_spot_borrow")
    if snapshot.open_order_ids:
        raise DemoBoundaryError("existing_open_spot_order")
    if snapshot.quantity_step <= 0 or snapshot.price_step <= 0:
        raise DemoBoundaryError("invalid_instrument_increment")
    if intent.quantity % snapshot.quantity_step:
        raise DemoBoundaryError("quantity_precision")
    if intent.limit_price % snapshot.price_step:
        raise DemoBoundaryError("price_precision")
    if intent.quantity < snapshot.min_quantity:
        raise DemoBoundaryError("below_minimum_quantity")
    if intent.quantity * intent.limit_price < snapshot.min_notional:
        raise DemoBoundaryError("below_minimum_notional")
    if snapshot.max_quantity is not None and intent.quantity > snapshot.max_quantity:
        raise DemoBoundaryError("above_maximum_quantity")
    # Leave 1% cash for trading fees; a limit IOC bounds the spend itself.
    if intent.side == "Buy" and intent.quantity * intent.limit_price > snapshot.usdt_available * Decimal("0.99"):
        raise DemoBoundaryError("insufficient_unborrowed_usdt")
    if intent.side == "Sell" and intent.quantity > snapshot.btc_available:
        raise DemoBoundaryError("insufficient_unborrowed_btc")
    age = (datetime.now(UTC) - snapshot.quote_at).total_seconds()
    if age < -5 or age > 30:
        raise DemoBoundaryError("stale_quote")


def safe_json(value: Any) -> str:
    return value.model_dump_json()


def _number(value: Any) -> Decimal:
    return decimal_value(value.as_decimal() if hasattr(value, "as_decimal") else value)


def _receipt_from_report(report: Any, client_order_id: str) -> ExecutionReceipt:
    """A venue report is authoritative; nonterminal reports remain UNKNOWN."""
    status = getattr(report.order_status, "name", str(report.order_status)).split(".")[-1].upper()
    statuses = {"FILLED": "filled", "CANCELED": "canceled", "REJECTED": "rejected",
                "EXPIRED": "canceled", "PARTIALLY_FILLED": "partially_filled",
                "ACCEPTED": "open", "PENDING_CANCEL": "open", "PENDING_UPDATE": "open"}
    filled = _number(report.filled_qty)
    average = getattr(report, "avg_px", None)
    return ExecutionReceipt(
        client_order_id=client_order_id,
        status=statuses.get(status, "unknown"),
        venue_order_id=str(report.venue_order_id) if report.venue_order_id else None,
        filled_quantity=filled,
        filled_quote=filled * _number(average) if average is not None else Decimal(0),
        observed_at=datetime.now(UTC),
        safe_message=None if status in statuses else "venue_order_nonterminal",
    )


class NautilusBybitDemoExecutor:
    """One process, one engine, fixed DEMO cash BTC/USDT boundary.

    ``start``/``close`` must run on the same asyncio loop. ``dispose`` is called
    after the loop has stopped because Nautilus node.dispose owns its loop.
    """

    def __init__(self, *, timeout_secs: float = 45.0) -> None:
        if not 1 <= timeout_secs <= 180:
            raise DemoBoundaryError("invalid_timeout")
        self.timeout_secs = timeout_secs
        self._node: Any = None
        self._strategy: Any = None
        self._exec_client: Any = None
        self._run_task: asyncio.Task | None = None
        self._attempted: set[str] = set()
        self._lock = asyncio.Lock()
        self._ready = asyncio.Event()

    def _build_node(self) -> None:
        validate_environment(os.environ)
        if version("nautilus_trader") != NAUTILUS_VERSION:
            raise DemoBoundaryError("nautilus_version_mismatch")
        api_key = os.environ.get("BYBIT_DEMO_API_KEY")
        api_secret = os.environ.get("BYBIT_DEMO_API_SECRET")
        if not api_key or not api_secret:
            raise DemoBoundaryError("demo_credentials_missing")

        from nautilus_trader.adapters.bybit import (
            BYBIT,
            BybitDataClientConfig,
            BybitEnvironment,
            BybitExecClientConfig,
            BybitLiveDataClientFactory,
            BybitLiveExecClientFactory,
            BybitProductType,
        )
        from nautilus_trader.config import (
            InstrumentProviderConfig,
            LiveExecEngineConfig,
            LoggingConfig,
            TradingNodeConfig,
        )
        from nautilus_trader.core import nautilus_pyo3
        from nautilus_trader.live.node import TradingNode
        from nautilus_trader.model.identifiers import InstrumentId, TraderId
        from nautilus_trader.trading.strategy import Strategy

        environment = BybitEnvironment.DEMO
        product_types = (BybitProductType.SPOT,)
        # Verify SDK resolution; a demo flag alone cannot authorize an endpoint.
        if nautilus_pyo3.get_bybit_http_base_url(environment) != DEMO_HTTP:
            raise DemoBoundaryError("sdk_demo_http_mismatch")
        if nautilus_pyo3.get_bybit_ws_url_public(BybitProductType.SPOT, environment) != PUBLIC_SPOT_WS:
            raise DemoBoundaryError("sdk_public_ws_mismatch")
        if nautilus_pyo3.get_bybit_ws_url_private(environment) != DEMO_PRIVATE_WS:
            raise DemoBoundaryError("sdk_private_ws_mismatch")
        instrument_id = InstrumentId.from_str(INSTRUMENT_ID)
        provider = InstrumentProviderConfig(load_all=False, load_ids=frozenset([instrument_id]))
        owner = self

        class GatewayStrategy(Strategy):
            def on_start(self) -> None:
                self.subscribe_quote_ticks(instrument_id)
                owner._ready.set()

        class DemoExecFactory(BybitLiveExecClientFactory):
            @staticmethod
            def create(loop, name, config, msgbus, cache, clock):
                if (
                    config.environment != BybitEnvironment.DEMO
                    or config.product_types != product_types
                    or config.base_url_http != DEMO_HTTP
                    or config.base_url_ws_private != DEMO_PRIVATE_WS
                    or config.base_url_ws_trade != DEMO_TRADE_WS
                    or config.max_retries != 0
                    or config.auto_repay_spot_borrows
                    or config.proxy_url is not None
                ):
                    raise DemoBoundaryError("unsafe_execution_client_config")
                client = BybitLiveExecClientFactory.create(loop, name, config, msgbus, cache, clock)
                if client._http_client.base_url != DEMO_HTTP:
                    raise DemoBoundaryError("unsafe_execution_http_endpoint")
                owner._exec_client = client
                return client

        config = TradingNodeConfig(
            trader_id=TraderId("DEMO-001"),
            # LIVE engines reject bypass_logging=True even for venue DEMO.
            logging=LoggingConfig(log_level="OFF", log_components_only=True,
                                  log_component_levels={}, print_config=False, use_tracing=False),
            exec_engine=LiveExecEngineConfig(
                reconciliation=True,
                reconciliation_instrument_ids=[instrument_id],
                generate_missing_orders=False,
                filter_position_reports=True,
                open_check_interval_secs=3.0,
                open_check_open_only=False,
                graceful_shutdown_on_exception=True,
            ),
            data_clients={BYBIT: BybitDataClientConfig(
                environment=environment,
                product_types=product_types,
                api_key=api_key,
                api_secret=api_secret,
                instrument_provider=provider,
                # Keep None: pinned upstream also feeds this field to the public
                # WS constructor. The verified DEMO defaults select correct URLs.
                base_url_http=None,
                max_retries=0,
                update_instruments_interval_mins=None,
                instrument_status_poll_secs=None,
            )},
            exec_clients={BYBIT: BybitExecClientConfig(
                environment=environment,
                product_types=product_types,
                api_key=api_key,
                api_secret=api_secret,
                instrument_provider=provider,
                base_url_http=DEMO_HTTP,
                base_url_ws_private=DEMO_PRIVATE_WS,
                base_url_ws_trade=DEMO_TRADE_WS,
                use_spot_position_reports=False,
                auto_repay_spot_borrows=False,
                max_retries=0,
            )},
            timeout_connection=self.timeout_secs,
            timeout_reconciliation=self.timeout_secs,
            timeout_portfolio=self.timeout_secs,
            timeout_disconnection=10.0,
            timeout_post_stop=0.1,
        )
        self._node = TradingNode(config=config, loop=asyncio.get_running_loop())
        self._strategy = GatewayStrategy()
        self._node.trader.add_strategy(self._strategy)
        self._node.add_data_client_factory(BYBIT, BybitLiveDataClientFactory)
        self._node.add_exec_client_factory(BYBIT, DemoExecFactory)
        self._node.build()

    async def start(self) -> None:
        self._build_node()
        self._run_task = asyncio.create_task(self._node.run_async())
        ready = asyncio.create_task(self._ready.wait())
        done, _ = await asyncio.wait([ready, self._run_task], timeout=self.timeout_secs * 3,
                                     return_when=asyncio.FIRST_COMPLETED)
        if ready not in done:
            ready.cancel()
            raise DemoBoundaryError("nautilus_startup_failed")

    async def close(self) -> None:
        if self._node is not None:
            await self._node.stop_async()
        if self._run_task is not None:
            self._run_task.cancel()
            await asyncio.gather(self._run_task, return_exceptions=True)

    def dispose(self) -> None:
        if self._node is not None:
            self._node.dispose()

    async def snapshot(self) -> SpotSnapshot:
        if not self._ready.is_set() or self._exec_client is None:
            raise DemoBoundaryError("executor_not_ready")
        from nautilus_trader.model.identifiers import InstrumentId

        instrument_id = InstrumentId.from_str(INSTRUMENT_ID)
        deadline = asyncio.get_running_loop().time() + self.timeout_secs
        while (quote := self._node.cache.quote_tick(instrument_id)) is None:
            if asyncio.get_running_loop().time() >= deadline:
                raise DemoBoundaryError("quote_unavailable")
            await asyncio.sleep(0.1)
        await asyncio.wait_for(self._exec_client._update_account_state(), self.timeout_secs)
        borrow_btc, borrow_usdt = await asyncio.wait_for(asyncio.gather(
            self._exec_client._http_client.get_spot_borrow_amount("BTC"),
            self._exec_client._http_client.get_spot_borrow_amount("USDT"),
        ), self.timeout_secs)
        account = self._node.cache.account(self._exec_client.account_id)
        if account is None:
            raise DemoBoundaryError("account_unavailable")
        balances = {str(currency): balance for currency, balance in account.balances().items()}

        def balance(coin: str, field: str) -> Decimal:
            item = balances.get(coin)
            return _number(getattr(item, field)) if item is not None else Decimal(0)

        instrument = self._node.cache.instrument(instrument_id)
        if instrument is None or instrument.min_notional is None:
            raise DemoBoundaryError("instrument_limits_unavailable")
        quote = self._node.cache.quote_tick(instrument_id)
        result = SpotSnapshot(
            btc_available=balance("BTC", "free"), usdt_available=balance("USDT", "free"),
            btc_borrowed=_number(borrow_btc), usdt_borrowed=_number(borrow_usdt),
            bid=_number(quote.bid_price), ask=_number(quote.ask_price),
            quantity_step=_number(instrument.size_increment),
            price_step=_number(instrument.price_increment),
            min_quantity=_number(instrument.min_quantity) if instrument.min_quantity else Decimal(0),
            min_notional=_number(instrument.min_notional),
            max_quantity=_number(instrument.max_quantity) if instrument.max_quantity else None,
            quote_at=datetime.fromtimestamp(quote.ts_event / 1_000_000_000, UTC),
            observed_at=datetime.now(UTC),
            open_order_ids=[str(order.client_order_id) for order in
                            self._node.cache.orders_open(instrument_id=instrument_id)],
        )
        if result.bid <= 0 or result.ask < result.bid:
            raise DemoBoundaryError("invalid_market_quote")
        return result

    async def _query_report(self, client_order_id: str) -> Any:
        from nautilus_trader.core.uuid import UUID4
        from nautilus_trader.execution.messages import GenerateOrderStatusReport
        from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId

        command = GenerateOrderStatusReport(
            instrument_id=InstrumentId.from_str(INSTRUMENT_ID),
            client_order_id=ClientOrderId(client_order_id), venue_order_id=None,
            command_id=UUID4(), ts_init=time.time_ns(),
        )
        # Adapter deliberately catches query errors and returns None. This is
        # acceptable only for reconciliation (UNKNOWN), never proof of absence.
        return await asyncio.wait_for(
            self._exec_client.generate_order_status_report(command), self.timeout_secs,
        )

    async def reconcile(self, client_order_id: str) -> ExecutionReceipt:
        validate_client_order_id(client_order_id)
        try:
            report = await self._query_report(client_order_id)
        except Exception:
            report = None
        if report is not None:
            try:
                receipt = await self._complete_receipt(report, client_order_id)
                if receipt.status != "unknown":
                    return receipt
            except Exception:
                pass
        # Read-only fallback never authorizes submission or changes preflight
        # behavior. It returns zero unknown totals on absent/incomplete evidence.
        try:
            return await asyncio.to_thread(reconcile_demo_order, client_order_id)
        except Exception:
            return ExecutionReceipt(client_order_id=client_order_id, status="unknown",
                                    observed_at=datetime.now(UTC), safe_message="venue_state_unconfirmed")

    async def _complete_receipt(self, report: Any, client_order_id: str) -> ExecutionReceipt:
        receipt = _receipt_from_report(report, client_order_id)
        if receipt.filled_quantity == 0:
            return receipt
        from nautilus_trader.core import nautilus_pyo3

        try:
            fills = await asyncio.wait_for(self._exec_client._http_client.request_fill_reports(
                account_id=self._exec_client.pyo3_account_id,
                product_type=nautilus_pyo3.BybitProductType.SPOT,
                instrument_id=nautilus_pyo3.InstrumentId.from_str(INSTRUMENT_ID),
                limit=100,
            ), self.timeout_secs)
            quantity = Decimal(0)
            quote = Decimal(0)
            fees = {"BTC": Decimal(0), "USDT": Decimal(0)}
            seen: set[str] = set()
            for fill in fills:
                if str(fill.venue_order_id) != receipt.venue_order_id:
                    continue
                trade_id = str(fill.trade_id)
                if trade_id in seen:
                    continue
                seen.add(trade_id)
                quantity += _number(fill.last_qty)
                quote += _number(fill.last_qty) * _number(fill.last_px)
                currency = str(fill.commission.currency)
                if currency not in fees:
                    raise DemoBoundaryError("unexpected_fee_currency")
                # The shared contract cannot represent rebates: fail closed
                # rather than silently inventing a zero fee.
                fees[currency] += decimal_value(fill.commission.as_decimal())
            if quantity != receipt.filled_quantity:
                raise DemoBoundaryError("incomplete_fill_reconciliation")
            return receipt.model_copy(update={"filled_quote": quote, "fee_btc": fees["BTC"],
                                               "fee_usdt": fees["USDT"]})
        except Exception:
            return receipt.model_copy(update={"status": "unknown", "safe_message": "execution_fees_unconfirmed"})

    async def execute_once(self, intent: OrderIntent) -> ExecutionReceipt:
        async with self._lock:
            if intent.client_order_id in self._attempted:
                return await self.reconcile(intent.client_order_id)
            snapshot = await self.snapshot()
            validate_intent(intent, snapshot)
            # Check through the native client directly: unlike the adapter report
            # helper, errors must propagate and cannot masquerade as absence.
            from nautilus_trader.core import nautilus_pyo3
            from nautilus_trader.model.enums import OrderSide, TimeInForce
            from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId
            from nautilus_trader.model.objects import Price, Quantity

            existing = await asyncio.wait_for(self._exec_client._http_client.query_order(
                account_id=self._exec_client.pyo3_account_id,
                product_type=nautilus_pyo3.BybitProductType.SPOT,
                instrument_id=nautilus_pyo3.InstrumentId.from_str(INSTRUMENT_ID),
                client_order_id=nautilus_pyo3.ClientOrderId(intent.client_order_id),
            ), self.timeout_secs)
            if existing is not None:
                return await self._complete_receipt(existing, intent.client_order_id)
            validate_intent(intent, snapshot)
            order = self._strategy.order_factory.limit(
                instrument_id=InstrumentId.from_str(INSTRUMENT_ID),
                order_side=OrderSide.BUY if intent.side == "Buy" else OrderSide.SELL,
                quantity=Quantity.from_str(str(intent.quantity)),
                price=Price.from_str(str(intent.limit_price)),
                time_in_force=TimeInForce.IOC,
                post_only=False, reduce_only=False, quote_quantity=False,
                client_order_id=ClientOrderId(intent.client_order_id),
            )
            # Mark before submission. Exceptions and transport timeouts never
            # clear this marker and never trigger another submit.
            self._attempted.add(intent.client_order_id)
            try:
                self._strategy.submit_order(order, params={"is_leverage": False})
            except Exception:
                return ExecutionReceipt(client_order_id=intent.client_order_id, status="unknown",
                                        observed_at=datetime.now(UTC), safe_message="submission_effect_unconfirmed")
            deadline = asyncio.get_running_loop().time() + self.timeout_secs
            while asyncio.get_running_loop().time() < deadline:
                # DENIED is the engine's pre-submission refusal, never infer this
                # from a missing venue order after a transport error.
                cached = self._node.cache.order(order.client_order_id)
                if cached is not None and getattr(cached.status, "name", "") == "DENIED":
                    return ExecutionReceipt(client_order_id=intent.client_order_id, status="rejected",
                                            observed_at=datetime.now(UTC), safe_message="engine_order_denied")
                receipt = await self.reconcile(intent.client_order_id)
                if receipt.status in ("filled", "canceled", "rejected"):
                    return receipt
                await asyncio.sleep(0.5)
            return ExecutionReceipt(client_order_id=intent.client_order_id, status="unknown",
                                    observed_at=datetime.now(UTC), safe_message="terminal_execution_unconfirmed")


async def _run_request(executor: NautilusBybitDemoExecutor, request: GatewayRequest) -> GatewayResponse:
    if request.operation == "snapshot":
        return GatewayResponse(ok=True, snapshot=await executor.snapshot())
    if request.operation == "reconcile":
        return GatewayResponse(ok=True, receipt=await executor.reconcile(request.client_order_id))
    return GatewayResponse(ok=True, receipt=await executor.execute_once(request.intent))


@contextlib.contextmanager
def _quiet_native_output():
    """Discard native library output; the process protocol emits only safe JSON.

    Windows Rust stdio uses Win32 standard handles, Python/C extensions use CRT
    descriptors. Redirect both for the entire native lifecycle, including stop.
    This gateway is intended to run in its dedicated subprocess.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    win_handles = None
    with open(os.devnull, "w") as sink:
        if os.name == "nt":
            import ctypes
            import msvcrt

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetStdHandle.argtypes = [ctypes.c_uint32]
            kernel.GetStdHandle.restype = ctypes.c_void_p
            kernel.SetStdHandle.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
            kernel.SetStdHandle.restype = ctypes.c_int
            ids = [0xFFFFFFF5, 0xFFFFFFF4]
            win_handles = [(item, kernel.GetStdHandle(item)) for item in ids]
            for item in ids:
                kernel.SetStdHandle(item, msvcrt.get_osfhandle(sink.fileno()))
        try:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for descriptor, previous in zip((1, 2), saved, strict=True):
                os.dup2(previous, descriptor)
                os.close(previous)
            if win_handles is not None:
                for item, previous in win_handles:
                    kernel.SetStdHandle(item, previous)


def handle_request(request: GatewayRequest) -> GatewayResponse:
    with _quiet_native_output():
        return _handle_request_quiet(request)


def _handle_request_quiet(request: GatewayRequest) -> GatewayResponse:
    executor = NautilusBybitDemoExecutor()

    async def run() -> GatewayResponse:
        try:
            await executor.start()
            return await _run_request(executor, request)
        finally:
            await executor.close()

    try:
        # Revalidate even model_construct/model_copy callers before any I/O.
        request = GatewayRequest.model_validate(request.model_dump())
        return asyncio.run(run())
    except Exception as exc:
        reason = str(exc) if isinstance(exc, DemoBoundaryError) else "executor_failed"
        return GatewayResponse(ok=False, error_code=reason)
    finally:
        with contextlib.suppress(Exception):
            executor.dispose()


def main() -> int:
    """Read one strict JSON request from stdin and emit one sanitized JSON result."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--response-file", type=Path)
    arguments = parser.parse_args()
    response_file = arguments.response_file
    try:
        if response_file is not None and not response_file.is_absolute():
            raise DemoBoundaryError("response_path_must_be_absolute")
        raw = sys.stdin.read(16385)
        if len(raw) > 16384:
            raise DemoBoundaryError("request_too_large")
        request = GatewayRequest.model_validate_json(raw)
        result = handle_request(request)
    except Exception as exc:
        # Do not emit arbitrary exception strings: venue errors may contain
        # signed URLs, auth headers or full private response bodies.
        reason = str(exc) if isinstance(exc, DemoBoundaryError) else "executor_failed"
        result = GatewayResponse(ok=False, error_code=reason)
    if response_file is None:
        print(safe_json(result), flush=True)
    else:
        try:
            _write_response_file(response_file, result)
        except Exception:
            # No path, OS error, credential or venue text can escape the process.
            print(GatewayResponse(ok=False, error_code="response_write_failed").model_dump_json(), flush=True)
            return 2
    return 0 if result.ok else 2


def _write_response_file(path: Path, response: GatewayResponse) -> None:
    if not path.is_absolute():
        raise DemoBoundaryError("response_path_must_be_absolute")
    name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".bybit-demo-response-", suffix=".tmp", delete=False) as stream:
            name = stream.name
            stream.write(response.model_dump_json())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        name = None
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
