# WBTC/USDT trading on a local Ethereum fork

This integration connects TradingAgents decisions to Almanak and Uniswap V3 on
an isolated Anvil fork. It supports **local-fork execution only**. There is no
mainnet execution flag, wallet import, key store, bridge, withdrawal or leverage
operation. The existing Bybit DEMO integration has separate state and scheduling.

WBTC is an Ethereum token representing BTC. This route does not exchange native
Bitcoin. Its quote token is Ethereum USDT with six decimals; WBTC has eight.
Choosing the eventual network and custody route for real funds is a separate step.

## Install an isolated worker

The worker uses Almanak 2.29.2 pinned to commit
`39cfabb780d297f5f90e2f3ee68215dfb382b5f2`. Use a dedicated Python 3.12 or 3.13
environment; leave the running Bybit environment alone. From the repository root,
on Linux:

```sh
uv venv --python 3.12 .venv/onchain-worker
uv pip install --python .venv/onchain-worker/bin/python -e . -r requirements/onchain-worker.txt
```

On Windows the worker executable is
`.venv/onchain-worker/Scripts/python.exe`. The main CLI can use its existing
Python environment and call that worker with `--worker-python PATH`. The worker
needs access to the same source and persistent state directory.

## Start a disposable fork

Use Anvil 1.7.1 on a free **loopback-only** port. Read a public Ethereum block
number and hash first, then pin that block. For example, after installing the
official `@foundry-rs/anvil@1.7.1` package in a disposable tools directory:

```sh
anvil --host 127.0.0.1 --port 18547 --chain-id 31337 \
  --fork-chain-id 1 --fork-url https://ethereum-rpc.publicnode.com \
  --fork-block-number BLOCK_NUMBER --quiet --no-storage-caching
```

The worker requires local chain ID 31337, an Anvil identity, the pinned fork
block, and the expected contract bytecode. It accepts only literal loopback RPC
addresses without credentials, redirects or environment proxies. It verifies the
fork source against the fixed public Ethereum read endpoint. Anvil 1.7.1 can
report the local chain ID in its source metadata; this is accepted only after
independent Ethereum chain/block verification. An unavailable source fails closed.

With the public endpoint, capture a latest block and start the acceptance test
immediately. Historical account-state reads can require archive access even when
block-header reads still work; Anvil 1.7.1 may exit on that provider refusal. A
failed old fork is not recovered by resetting its journal. Preserve its evidence
and use a new disposable fork plus a separate journal for a new test case.

The unlocked accounts belong to this disposable node. Signing uses
`eth_signTransaction` inside Anvil; no private key or seed enters the program's
configuration, output or journal. Test token funding belongs exclusively to the
opt-in fork acceptance test, after the same node verification.

## CLI

```sh
tradingagents-onchain status
tradingagents-onchain --worker-python .venv/onchain-worker/bin/python snapshot
tradingagents-onchain --worker-python .venv/onchain-worker/bin/python once
tradingagents-onchain disable
tradingagents-onchain enable
```

Global options precede the command. `--rpc-url` defaults to
`http://127.0.0.1:18547`; `--root` defaults to `~/.tradingagents/onchain-fork`.
`status` reads only local files. `snapshot` is read-only. `once` may perform a
swap **on the verified local fork**. It does not create a recurring schedule.
`disable` also takes effect while analysis runs; already broadcast transactions
are preserved for reconciliation. `enable` does not execute an order.

An explicit `--fixture-rating Buy` (or another valid rating) replaces the model
only for local-fork testing. Such decisions are saved as `source=fixture` and
are not reported as model recommendations. Normal analysis requires the separate
ChatGPT subscription login on the machine running the analysis. Never copy an
authentication store or wallet material into the fork workspace.

## Decision and execution

TradingAgents analyses BTC-USD reference data, news and sentiment, debates the
thesis and risk, then returns a validated `PortfolioDecision`. Prompts identify
the execution asset as WBTC on a local Ethereum fork and the valuation currency
as USDT. Daily reference data is not an executable DEX quote. No text fallback
can authorize a swap.

Deterministic policy starts with **300 virtual USDT and zero managed WBTC**.
It binds the book to the exact fork instance, source block and unlocked account;
it does not adopt existing token balances. The wallet must actually contain the
managed balances on the fork.

| Rating | Target WBTC share |
| --- | --- |
| Buy | 50% |
| Overweight | 25% |
| Hold | Keep current holdings |
| Underweight | 10% |
| Sell | 0% |

Orders are limited to 150 USDT equivalent. Policy requires fresh snapshots,
unexpired decisions and quotes, slippage at most 50 basis points, exact integer
token amounts, and at least 0.005 native test ETH as a gas reserve. Native gas is
accounted separately in wei; it is not assigned an invented USDT price.

The worker discovers an actual WBTC/USDT pool through the canonical factory and
compares quotes. It validates compiled approvals and the exact swap's tokens,
router, recipient, amount and minimum output. Approvals are bounded to the
required amount. The swap is simulated and has an onchain deadline.
Before signing, each transaction is limited to 1,000,000 gas and 50 gwei maximum
fee per gas. The bundle's maximum gas cost cannot exceed 0.004 test ETH, and at
least another 0.001 test ETH must remain available. Actual gas is still charged
from the confirmed receipt rather than that ceiling.

Narrow project-owned adapters make the pinned SDK suitable for this fork:
unsigned transaction chain IDs are changed to 31337 only after node verification,
the Router02 swap is wrapped in a deadline-bearing multicall, and the unlocked
Anvil signer supplies signatures. The compiler's default 10% approval buffer is
replaced with an exact-amount approval; USDT's allowance reset is handled when
needed. No dependency or global chain registry is patched.

The main intent is persisted before execution. The worker journals transaction
hashes and nonces before broadcast. A timeout or unknown result never authorizes
a new submission: subsequent runs only reconcile the saved identity and hashes.
Receipts must agree with canonical blocks, actual transfer logs and token balance
changes; successful approval alone is not successful execution. Actual gas is
recorded from confirmed receipts. Corrupt or missing established journals and
fork restarts fail closed; do not delete state to bypass those checks.

## Verification and boundaries

Offline tests exercise schema rejection, integer sizing, gas and slippage limits,
strict model output, durable state, node and payload guards, pending recovery,
duplicate execution and interruption. The opt-in fork acceptance test exercises
real deployed Uniswap bytecode copied into Anvil, with disposable token balances.
Neither offline tests nor a fork receipt proves mainnet execution or a funded
wallet is ready. No onchain hourly automation is installed by these commands.

For an otherwise unused fork account, the acceptance test creates 20 disposable
USDT, buys approximately 10 USDT of WBTC and sells the acquired tokens back. It
also repeats each saved intent through a fresh worker and checks that the nonce
does not increase. Keep the test journal outside the repository:

```sh
ONCHAIN_FORK_RPC=http://127.0.0.1:18547 \
ONCHAIN_FORK_TEST_STATE="$HOME/.tradingagents/onchain-fork-acceptance" \
  .venv/onchain-worker/bin/python -m pytest -m integration -q tests/integration/test_onchain_fork.py
```

Install pytest in the isolated worker environment to run this test. Existing
nonzero balances are not overwritten; unresolved transactions and journals are
retained. Start a fresh disposable fork for a fresh acceptance account.

The source addresses are verified against [Uniswap's Ethereum deployments](https://developers.uniswap.org/docs/protocols/v3/deployments/v3-ethereum-deployments),
[WBTC transparency](https://wbtc.network/transparency) and
[Tether's supported protocols](https://tether.to/en/supported-protocols/).
The direct SDK route is based on [Almanak's pinned source](https://github.com/almanak-co/sdk/tree/39cfabb780d297f5f90e2f3ee68215dfb382b5f2).
