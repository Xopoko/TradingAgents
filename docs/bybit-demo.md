# Bybit DEMO BTC spot monitor

This optional integration runs one TradingAgents analysis and at most one bounded BTCUSDT spot order per UTC hour. Orders use Bybit DEMO through NautilusTrader; they never fall back to production or testnet. Installing the command does not connect an account or create an hourly schedule.

## Install and authenticate

Use Python 3.12–3.14 for the optional execution adapter. The core package still supports Python 3.11. From the project directory:

```sh
python -m pip install -e ".[bybit-demo]"
```

The extra pins stable `nautilus_trader==1.231.0`; the integration does not support the v2 release-candidate API.

All submissions pass through NautilusTrader's risk and execution engines. When the pinned adapter cannot parse a current Bybit order or commission response, reconciliation uses a separate, typed, read-only Bybit V5 client. That client permits only DEMO order and execution lookups; it cannot place or cancel orders. Filled quantities and actual BTC/USDT commissions must agree before the journal accepts a terminal fill. Missing or inconsistent evidence remains pending.

TradingAgents uses your existing ChatGPT subscription session. Complete its separate browser sign-in if needed:

```sh
tradingagents login chatgpt
```

The monitor explicitly selects the `chatgpt` provider for both model tiers: `gpt-6-luna` for quick work and `gpt-6-sol` for deep work. Reports are in Russian. Subscription sign-in does not authenticate Bybit.

Create an HMAC API key from **Bybit Demo Trading**. Select the narrowest available trading permissions and leave asset transfers disabled. Some DEMO accounts expose only the combined Unified Trading permission; in that case the executor still enforces its own BTC spot-only boundary. Store the key and secret through the protected user-controlled 1Password route. Never paste credentials into chat, terminal arguments, logs, source files, or this guide. No withdrawal permission is needed.

The executor accepts the complete pair of process environment variables `BYBIT_DEMO_API_KEY` and `BYBIT_DEMO_API_SECRET`, supplied through an approved secret injector. Alternatively, create a **reference-only** `credentials.env` in the state directory, with this shape:

```dotenv
BYBIT_DEMO_API_KEY=op://vault/item/field
BYBIT_DEMO_API_SECRET=op://vault/item/field
```

Replace those placeholders with the distinct references to the stored key and secret fields. The runner invokes official `op run --env-file … -- …` to inject them only into the executor process; 1Password must already be available and authorized for the scheduled user. Do not put plaintext credentials in that file. A partial environment pair is rejected rather than combined with references.

## Run and control

The default state directory is `~/.tradingagents/bybit-demo`. Use `--root PATH` before the subcommand to select a different directory consistently.

```sh
tradingagents-bybit-demo status
tradingagents-bybit-demo once
tradingagents-bybit-demo disable
tradingagents-bybit-demo enable
```

`status` reads local state without contacting Bybit or ChatGPT. `once` runs a single cycle and prints a typed JSON result. A fresh state directory is enabled by default, so `once` can submit a DEMO order when credentials, analysis, and policy checks succeed. `enable` allows future cycles; it does not execute one or install a schedule. `disable` also works while analysis is running: the cycle checks the control flag again before reserving and submitting an order. Orders already submitted are not canceled. If disable arrives after a reservation was saved, that reservation is retained; enabling later permits reconciliation only, without automatic resubmission.

Hourly scheduling belongs to an external scheduler. Configure it to invoke `once` using the same environment and state directory. No scheduler is installed by these commands. A process lock and persisted UTC-hour identifiers prevent overlapping or repeated completed cycles.

## Portfolio and execution rules

The bot starts with a virtual allocation of **300 demo USDT and zero bot-owned BTC**. The account must actually have enough available demo USDT. Preloaded BTC in the DEMO account is not imported into the bot ledger or sold by the bot. The allocation is an accounting boundary, not a separate Bybit subaccount.

The typed Portfolio Manager rating sets the target BTC share of managed capital, bounded by current managed equity and the 300-USDT cap:

| Rating | Target BTC share |
| --- | --- |
| Buy | 50% |
| Overweight | 25% |
| Hold | Keep current holdings; no order |
| Underweight | 10% |
| Sell | 0% |

These are target allocations, so a rating can require either buying or selling to rebalance. Each order is capped at **150 USDT** and uses **LIMIT IOC**, with a **0.2% price bound** and a **0.2% fee buffer** for sizing. Exchange minimums and quantity/price increments can result in no order. Partial fills are possible; confirmed actual USDT and BTC fees are deducted from the managed ledger.

The runner requires a quote and account snapshot no older than **120 seconds** and fetches them again after analysis, immediately before sizing. A typed analysis expires 15 minutes after completion. Orders use available cash and bot-owned BTC only. Borrowed balances and preexisting open orders block trading. Margin, leverage, short selling, derivatives, transfers, withdrawals, and production fallback are outside the integration.

An intent is persisted before submission. An unknown or pending execution is reconciled using its existing client order ID before any further analysis or submission. Do not clear the ledger or retry the order under a new ID to resolve uncertainty. Corrupt persisted state, or a missing book in an already initialized state directory, blocks execution; do not delete or reset it to bypass the error. Preserve it and reconcile against the venue first.

## Analysis and evidence

Market, social, and news analysts run with one debate round, one risk round, and at most eight tool rounds per analyst. Strict structured output is required during graph construction and execution. The final `PortfolioDecision` must survive graph propagation and agree with `final_rating`; free-text ratings and the `REVIEW` fallback cannot authorize an order.

Yahoo `BTC-USD` data provides the daily reference-market analysis. Actual orders and deterministic sizing use timestamped Bybit `BTCUSDT` bid/ask quotes. Running the analysis hourly is periodic reevaluation; it does not make the daily Yahoo data an intraday-bar strategy. USD reference prices are not executable USDT quotes.

Analysis results, cache, and memory live under the selected state directory's `analysis/` subtree. The managed book and pending-order state live alongside it. Keep this directory across restarts.

Offline tests verify contracts, sizing, ledger reconciliation, restart handling, and strict graph output. They do not prove account access, DEMO connectivity, or exchange fills. Establish those separately from fresh execution receipts; installing the package or passing tests is not evidence that an account is connected.
