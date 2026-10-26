"""Bounded JSON-RPC access to a literal loopback Anvil Ethereum fork.

This adapter never accepts credentials, resolves hostnames, follows redirects,
or inherits HTTP proxy settings. Node error text is deliberately not exposed.
"""

from __future__ import annotations

import json
import re
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from pydantic import ValidationError

from .contracts import ForkIdentity
from .profile import FACTORY, LOCAL_CHAIN_ID, QUOTER, ROUTER, SOURCE_CHAIN_ID, USDT, WBTC

_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}\Z")
_HASH = re.compile(r"0x[0-9a-fA-F]{64}\Z")
_DATA = re.compile(r"0x(?:[0-9a-fA-F]{2})*\Z")
_QUANTITY = re.compile(r"0x(?:0|[1-9a-fA-F][0-9a-fA-F]*)\Z")
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_PUBLIC_ETHEREUM_RPC = "https://ethereum-rpc.publicnode.com"


class ForkError(RuntimeError):
    """An error whose public representation contains only a safe local code."""

    def __init__(self, code: str):
        self.code = code if re.fullmatch(r"[a-z][a-z0-9_]{0,99}", code) else "fork_error"
        super().__init__(self.code)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ForkError("rpc_redirect_forbidden")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("non_json_number")


def _quantity(value, *, metadata: bool = False) -> int:
    if metadata and type(value) is int and value >= 0:
        number = value
    elif isinstance(value, str) and _QUANTITY.fullmatch(value):
        number = int(value, 16)
    else:
        raise ForkError("rpc_invalid_response")
    if number > 2**256 - 1:
        raise ForkError("rpc_invalid_response")
    return number


def _address(value: str) -> str:
    if not isinstance(value, str) or not _ADDRESS.fullmatch(value):
        raise ForkError("rpc_invalid_address")
    return value.lower()


def _hash(value) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ForkError("rpc_invalid_response")
    return value.lower()


def _public_fork_proof(url: str, number: int, expected_hash: str, opener) -> None:
    """Two fixed read calls only; this transport has no transaction API."""
    if url not in {_PUBLIC_ETHEREUM_RPC, _PUBLIC_ETHEREUM_RPC + "/"}:
        raise ForkError("fork_source_url_forbidden")
    try:
        answers = []
        for request_id, (method, params) in enumerate(
            (
                ("eth_chainId", []),
                ("eth_getBlockByNumber", [hex(number), False]),
            ),
            start=1,
        ):
            body = json.dumps(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            ).encode()
            request = Request(
                _PUBLIC_ETHEREUM_RPC,
                body,
                {
                    "Content-Type": "application/json",
                    "User-Agent": "TradingAgents fork verifier",
                },
                method="POST",
            )
            with opener.open(request, timeout=20) as response:
                if response.status != 200:
                    raise ValueError()
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                raise ValueError()
            result = json.loads(raw, object_pairs_hook=_object, parse_constant=_reject_constant)
            if (
                not isinstance(result, dict)
                or result.get("jsonrpc") != "2.0"
                or type(result.get("id")) is not int
                or result["id"] != request_id
                or "result" not in result
                or "error" in result
            ):
                raise ValueError()
            answers.append(result["result"])
        if _quantity(answers[0]) != SOURCE_CHAIN_ID:
            raise ValueError()
        block = answers[1]
        if (
            not isinstance(block, dict)
            or _quantity(block.get("number")) != number
            or _hash(block.get("hash")) != expected_hash
        ):
            raise ValueError()
    except (ForkError, OSError, HTTPException, ValueError, TypeError, UnicodeError, RecursionError):
        raise ForkError("fork_source_unverified") from None


class ForkRPC:
    def __init__(self, url: str):
        try:
            if not isinstance(url, str) or any(c.isspace() for c in url):
                raise ValueError
            parsed = urlsplit(url)
            if (
                parsed.scheme not in {"http", "https"}
                or parsed.hostname not in {"127.0.0.1", "::1"}
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path not in {"", "/"}
                or any(c in url for c in "?#\\@")
                or (parsed.port is not None and not 1 <= parsed.port <= 65535)
            ):
                raise ValueError
        except (ValueError, TypeError):
            raise ForkError("rpc_url_forbidden") from None
        self.url = url
        self._next_id = 0
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())
        self._proof_opener = build_opener(ProxyHandler({}), _NoRedirect())
        self._verified_source = None

    def call(self, method: str, params: list):
        if not isinstance(method, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,99}", method):
            raise ForkError("rpc_invalid_request")
        if not isinstance(params, list):
            raise ForkError("rpc_invalid_request")
        self._next_id += 1
        request_id = self._next_id
        try:
            body = json.dumps(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, OverflowError):
            raise ForkError("rpc_invalid_request") from None
        request = Request(self.url, body, {"Content-Type": "application/json"}, method="POST")
        try:
            with self._opener.open(request, timeout=30) as response:
                if response.status != 200:
                    raise ForkError("rpc_http_error")
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except ForkError:
            raise
        except HTTPError:
            raise ForkError("rpc_http_error") from None
        except (URLError, OSError, TimeoutError, ValueError, HTTPException):
            raise ForkError("rpc_unavailable") from None
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ForkError("rpc_response_too_large")
        try:
            result = json.loads(raw, object_pairs_hook=_object, parse_constant=_reject_constant)
        except (ValueError, UnicodeError, RecursionError):
            raise ForkError("rpc_invalid_response") from None
        if (
            not isinstance(result, dict)
            or result.get("jsonrpc") != "2.0"
            or type(result.get("id")) is not int
            or result["id"] != request_id
            or (("result" in result) == ("error" in result))
        ):
            raise ForkError("rpc_invalid_response")
        if "error" in result:
            error = result["error"]
            if (
                not isinstance(error, dict)
                or type(error.get("code")) is not int
                or not isinstance(error.get("message"), str)
            ):
                raise ForkError("rpc_invalid_response")
            raise ForkError("rpc_error")
        return result["result"]

    def code(self, address: str) -> str:
        result = self.call("eth_getCode", [_address(address), "latest"])
        if not isinstance(result, str) or not _DATA.fullmatch(result):
            raise ForkError("rpc_invalid_response")
        return result

    def eth_call(self, to: str, data: str) -> str:
        if not isinstance(data, str) or not _DATA.fullmatch(data):
            raise ForkError("rpc_invalid_request")
        result = self.call("eth_call", [{"to": _address(to), "data": data}, "latest"])
        if not isinstance(result, str) or not _DATA.fullmatch(result):
            raise ForkError("rpc_invalid_response")
        return result

    def balance(self, token: str, wallet: str) -> int:
        result = self.eth_call(token, "0x70a08231" + _address(wallet)[2:].rjust(64, "0"))
        if len(result) != 66:
            raise ForkError("rpc_invalid_response")
        return int(result, 16)

    def native_balance(self, wallet: str) -> int:
        return _quantity(self.call("eth_getBalance", [_address(wallet), "latest"]))

    def block(self) -> dict:
        block = self.call("eth_getBlockByNumber", ["latest", False])
        self._validate_block(block)
        return block

    @staticmethod
    def _validate_block(block) -> None:
        if not isinstance(block, dict) or _quantity(block.get("number")) < 1:
            raise ForkError("rpc_invalid_response")
        _hash(block.get("hash"))

    def identity(self) -> ForkIdentity:
        version = self.call("web3_clientVersion", [])
        if not isinstance(version, str) or "anvil" not in version.lower():
            raise ForkError("fork_not_anvil")
        if _quantity(self.call("eth_chainId", [])) != LOCAL_CHAIN_ID:
            raise ForkError("fork_chain_mismatch")
        metadata = self.call("anvil_metadata", [])
        if not isinstance(metadata, dict) or not isinstance(metadata.get("forkedNetwork"), dict):
            raise ForkError("fork_metadata_invalid")
        network = metadata["forkedNetwork"]
        if _quantity(metadata.get("chainId"), metadata=True) != LOCAL_CHAIN_ID:
            raise ForkError("fork_chain_mismatch")
        if _quantity(network.get("chainId"), metadata=True) not in {
            SOURCE_CHAIN_ID,
            LOCAL_CHAIN_ID,
        }:
            raise ForkError("fork_source_mismatch")
        instance_id = _hash(metadata.get("instanceId"))
        fork_number = _quantity(network.get("forkBlockNumber"), metadata=True)
        fork_hash = _hash(network.get("forkBlockHash"))
        if fork_number < 1:
            raise ForkError("fork_metadata_invalid")
        # Anvil 1.7.1 reports the overridden local chain ID in forkedNetwork.
        # Prove the origin independently; never reinterpret 31337 as Ethereum.
        info = self.call("anvil_nodeInfo", [])
        if not isinstance(info, dict) or not isinstance(info.get("forkConfig"), dict):
            raise ForkError("fork_source_unverified")
        source_url = info["forkConfig"].get("forkUrl")
        if source_url not in {_PUBLIC_ETHEREUM_RPC, _PUBLIC_ETHEREUM_RPC + "/"}:
            raise ForkError("fork_source_url_forbidden")
        source = (fork_number, fork_hash, source_url.rstrip("/"))
        if source != self._verified_source:
            _public_fork_proof(source_url, fork_number, fork_hash, self._proof_opener)
            self._verified_source = source
        fork_block = self.call("eth_getBlockByNumber", [hex(fork_number), False])
        self._validate_block(fork_block)
        if _quantity(fork_block["number"]) != fork_number or _hash(fork_block["hash"]) != fork_hash:
            raise ForkError("fork_block_mismatch")
        accounts = self.call("eth_accounts", [])
        if not isinstance(accounts, list) or not accounts:
            raise ForkError("fork_wallet_missing")
        wallet = _address(accounts[0])
        for contract in (USDT, WBTC, FACTORY, QUOTER, ROUTER):
            if self.code(contract) == "0x":
                raise ForkError("fork_contract_missing")
        for token, decimals in ((USDT, 6), (WBTC, 8)):
            value = self.eth_call(token, "0x313ce567")
            if len(value) != 66 or int(value, 16) != decimals:
                raise ForkError("fork_token_decimals_mismatch")
        # A reset during the multi-call inspection must not produce a mixed identity.
        final_metadata = self.call("anvil_metadata", [])
        if (
            not isinstance(final_metadata, dict)
            or final_metadata.get("instanceId") != metadata["instanceId"]
        ):
            raise ForkError("fork_identity_changed")
        if (
            final_metadata.get("forkedNetwork") != network
            or final_metadata.get("chainId") != metadata["chainId"]
        ):
            raise ForkError("fork_identity_changed")
        try:
            return ForkIdentity(
                instance_id=instance_id,
                fork_block_number=fork_number,
                fork_block_hash=fork_hash,
                wallet_address=wallet,
            )
        except ValidationError:
            raise ForkError("fork_metadata_invalid") from None

    def assert_identity(self, expected: ForkIdentity) -> None:
        if not isinstance(expected, ForkIdentity) or self.identity() != expected:
            raise ForkError("fork_identity_changed")
