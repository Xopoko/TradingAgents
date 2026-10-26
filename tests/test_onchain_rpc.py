"""Offline adversarial checks for the local-fork RPC boundary."""

import json
from urllib.error import URLError

import pytest

from tradingagents.integrations.onchain.profile import USDT, WBTC
from tradingagents.integrations.onchain.rpc import ForkError, ForkRPC, _NoRedirect

WALLET = "0x" + "12" * 20
BLOCK_HASH = "0x" + "ab" * 32
INSTANCE_ID = "0x" + "cd" * 32


class Response:
    status = 200

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, limit):
        return self.body[:limit]


class Opener:
    def __init__(self, transform=None):
        self.requests = []
        self.transform = transform or (lambda value: value)

    def open(self, request, timeout):
        value = json.loads(request.data)
        self.requests.append(value)
        return Response(
            json.dumps(
                self.transform(
                    {
                        "jsonrpc": "2.0",
                        "id": value["id"],
                        "result": "0x1",
                    }
                )
            ).encode()
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://mainnet.example",
        "http://localhost:8545",
        "http://127.0.0.2:8545",
        "http://2130706433:8545",
        "http://127.1:8545",
        "http://[::ffff:127.0.0.1]:8545",
        "http://user:secret@127.0.0.1:8545",
        "http://127.0.0.1:8545/?secret",
        "http://127.0.0.1:8545/#secret",
        "http://127.0.0.1:8545/?",
        "http://127.0.0.1:8545/path",
        " http://127.0.0.1:8545",
        "file:///tmp/rpc",
        "http://127.0.0.1:0",
        "http://127.0.0.1:65536",
        "http://127.0.0.1:bad",
    ],
)
def test_forbids_nonliteral_loopback_and_url_secrets(url):
    with pytest.raises(ForkError, match="^rpc_url_forbidden$"):
        ForkRPC(url)


@pytest.mark.parametrize("url", ["http://127.0.0.1:8545", "http://[::1]:8545/"])
def test_accepts_literal_loopback(url):
    assert ForkRPC(url).url == url


def test_rpc_request_and_unique_response_ids():
    rpc = ForkRPC("http://127.0.0.1:8545")
    opener = Opener()
    rpc._opener = opener
    assert rpc.call("eth_chainId", []) == "0x1"
    assert rpc.call("eth_chainId", []) == "0x1"
    assert [request["id"] for request in opener.requests] == [1, 2]
    assert opener.requests[0] == {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_chainId",
        "params": [],
    }


@pytest.mark.parametrize(
    "transform",
    [
        lambda x: {**x, "id": 55},
        lambda x: {**x, "id": True},
        lambda x: {**x, "jsonrpc": "1.0"},
        lambda x: {**x, "error": None},
        lambda x: {"jsonrpc": "2.0", "id": x["id"]},
        lambda x: {"jsonrpc": "2.0", "id": x["id"], "error": "private error"},
        lambda x: {"jsonrpc": "2.0", "id": x["id"], "error": {"code": True, "message": "private"}},
    ],
)
def test_rejects_invalid_envelope(transform):
    rpc = ForkRPC("http://127.0.0.1:8545")
    rpc._opener = Opener(transform)
    with pytest.raises(ForkError, match="^rpc_invalid_response$"):
        rpc.call("eth_chainId", [])


def test_rpc_errors_never_expose_provider_message():
    rpc = ForkRPC("http://127.0.0.1:8545")
    rpc._opener = Opener(
        lambda x: {
            "jsonrpc": "2.0",
            "id": x["id"],
            "error": {"code": -32000, "message": "secret provider URL", "data": "secret"},
        }
    )
    with pytest.raises(ForkError, match="^rpc_error$") as caught:
        rpc.call("eth_chainId", [])
    assert caught.value.code == "rpc_error"


def test_transport_errors_are_sanitized():
    class BrokenOpener:
        def open(self, *args, **kwargs):
            raise URLError("secret provider URL")

    rpc = ForkRPC("http://127.0.0.1:8545")
    rpc._opener = BrokenOpener()
    with pytest.raises(ForkError, match="^rpc_unavailable$"):
        rpc.call("eth_chainId", [])


@pytest.mark.parametrize(
    "body",
    [
        b'{"jsonrpc":"2.0","id":1,"id":1,"result":1}',
        b'{"jsonrpc":"2.0","id":1,"result":NaN}',
        b"not JSON",
        b"x" * (2 * 1024 * 1024 + 1),
    ],
    ids=["duplicate-key", "nan", "non-json", "oversized"],
)
def test_malformed_or_unbounded_response(body):
    class RawOpener:
        def open(self, *args, **kwargs):
            return Response(body)

    rpc = ForkRPC("http://127.0.0.1:8545")
    rpc._opener = RawOpener()
    with pytest.raises(ForkError):
        rpc.call("eth_chainId", [])


def test_redirect_is_rejected_before_opening_destination():
    with pytest.raises(ForkError, match="^rpc_redirect_forbidden$"):
        _NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere.example")


class ForkFixture(ForkRPC):
    def __init__(self):
        super().__init__("http://127.0.0.1:8545")
        self.version = "anvil/v1.0"
        self.chain = "0x7a69"
        self.metadata = {
            "chainId": 31337,
            "instanceId": INSTANCE_ID,
            "forkedNetwork": {
                "chainId": 1,
                "forkBlockNumber": 20000000,
                "forkBlockHash": BLOCK_HASH,
            },
        }
        self.fork_hash = BLOCK_HASH
        self.accounts = [WALLET]
        self.missing_contract = None
        self.decimals = {USDT: 6, WBTC: 8}
        self.calls = []
        self.reset = False
        self.source_url = "https://ethereum-rpc.publicnode.com"
        self._proof_opener = ProofOpener()

    def call(self, method, params):
        self.calls.append((method, params))
        if method == "web3_clientVersion":
            return self.version
        if method == "eth_chainId":
            return self.chain
        if method == "anvil_metadata":
            value = json.loads(json.dumps(self.metadata))
            if self.reset and sum(name == method for name, _ in self.calls) > 1:
                value["instanceId"] = "0x" + "ef" * 32
            return value
        if method == "eth_accounts":
            return self.accounts
        if method == "anvil_nodeInfo":
            return {"forkConfig": {"forkUrl": self.source_url, "forkBlockNumber": 20000000}}
        if method == "eth_getBlockByNumber":
            return {"number": hex(20000000), "hash": self.fork_hash}
        if method == "eth_getCode":
            return "0x" if params[0] == self.missing_contract else "0x6001"
        if method == "eth_call":
            address = params[0]["to"]
            value = self.decimals[address] if params[0]["data"] == "0x313ce567" else 99
            return "0x" + format(value, "064x")
        if method == "eth_getBalance":
            return "0x64"
        raise AssertionError(method)


class ProofOpener:
    def __init__(self, chain="0x1", block_hash=BLOCK_HASH, failure=None):
        self.requests = []
        self.chain, self.block_hash, self.failure = chain, block_hash, failure

    def open(self, request, timeout):
        if self.failure:
            raise self.failure
        value = json.loads(request.data)
        self.requests.append(value)
        assert value["method"] in {"eth_chainId", "eth_getBlockByNumber"}
        assert request.full_url == "https://ethereum-rpc.publicnode.com"
        assert request.get_header("User-agent") == "TradingAgents fork verifier"
        result = (
            self.chain
            if value["method"] == "eth_chainId"
            else {"number": hex(20000000), "hash": self.block_hash}
        )
        return Response(
            json.dumps({"jsonrpc": "2.0", "id": value["id"], "result": result}).encode()
        )


def test_identity_and_base_unit_reads():
    rpc = ForkFixture()
    identity = rpc.identity()
    assert identity.chain_id == 31337
    assert identity.source_chain_id == 1
    assert identity.wallet_address == WALLET
    assert identity.fork_block_hash == BLOCK_HASH
    assert identity.instance_id == INSTANCE_ID
    assert identity.fork_block_number == 20000000
    rpc.assert_identity(identity)
    assert len(rpc._proof_opener.requests) == 2  # immutable source proof cached
    assert rpc.balance(USDT, WALLET) == 99
    assert rpc.native_balance(WALLET) == 100
    assert rpc.block()["hash"] == BLOCK_HASH


def test_anvil_171_overridden_source_id_still_requires_public_proof():
    rpc = ForkFixture()
    rpc.metadata["forkedNetwork"]["chainId"] = 31337
    assert rpc.identity().source_chain_id == 1
    assert [value["method"] for value in rpc._proof_opener.requests] == [
        "eth_chainId",
        "eth_getBlockByNumber",
    ]


@pytest.mark.parametrize(
    "opener",
    [
        ProofOpener(chain="0xa4b1"),
        ProofOpener(block_hash="0x" + "ef" * 32),
        ProofOpener(failure=URLError("private provider text")),
        ProofOpener(failure=ForkError("rpc_redirect_forbidden")),
    ],
    ids=["chain-mismatch", "hash-mismatch", "unavailable", "redirect"],
)
def test_public_source_proof_fails_closed(opener):
    rpc = ForkFixture()
    rpc._proof_opener = opener
    with pytest.raises(ForkError, match="^fork_source_unverified$"):
        rpc.identity()


@pytest.mark.parametrize(
    "url",
    [
        "https://ethereum-rpc.publicnode.com?x=1",
        "http://ethereum-rpc.publicnode.com",
        "https://elsewhere.example",
        "https://ethereum-rpc.publicnode.com/other",
        "https://secret@ethereum-rpc.publicnode.com",
        "https://ethereum-rpc.publicnode.com:443",
    ],
)
def test_unknown_public_source_url_never_requested(url):
    rpc = ForkFixture()
    rpc.source_url = url
    with pytest.raises(ForkError, match="^fork_source_url_forbidden$"):
        rpc.identity()
    assert not rpc._proof_opener.requests


def test_metadata_accepts_anvil_hex_quantities():
    rpc = ForkFixture()
    rpc.metadata["chainId"] = "0x7a69"
    rpc.metadata["forkedNetwork"]["chainId"] = "0x1"
    rpc.metadata["forkedNetwork"]["forkBlockNumber"] = hex(20000000)
    assert rpc.identity().fork_block_number == 20000000


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("version", "Geth", "fork_not_anvil"),
        ("chain", "0x1", "fork_chain_mismatch"),
        ("fork_hash", "0x" + "ef" * 32, "fork_block_mismatch"),
        ("accounts", [], "fork_wallet_missing"),
        ("missing_contract", USDT, "fork_contract_missing"),
        ("reset", True, "fork_identity_changed"),
    ],
)
def test_identity_rejects_wrong_environment(field, value, error):
    rpc = ForkFixture()
    setattr(rpc, field, value)
    with pytest.raises(ForkError, match=f"^{error}$"):
        rpc.identity()


def test_identity_rejects_wrong_source_decimals_and_stale_expected():
    rpc = ForkFixture()
    identity = rpc.identity()
    rpc.metadata["forkedNetwork"]["chainId"] = 42161
    with pytest.raises(ForkError, match="^fork_source_mismatch$"):
        rpc.identity()
    rpc.metadata["forkedNetwork"]["chainId"] = 1
    rpc.decimals[USDT] = 18
    with pytest.raises(ForkError, match="^fork_token_decimals_mismatch$"):
        rpc.identity()
    rpc.decimals[USDT] = 6
    rpc.metadata["instanceId"] = "0x" + "ef" * 32
    with pytest.raises(ForkError, match="^fork_identity_changed$"):
        rpc.assert_identity(identity)
