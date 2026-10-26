"""Public contract identities for the initial isolated fork profile."""
from decimal import Decimal

PROFILE = "ethereum-wbtc-usdt"
LOCAL_CHAIN_ID = 31337
SOURCE_CHAIN_ID = 1
WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"
USDT = "0xdac17f958d2ee523a2206206994597c13d831ec7"
ROUTER = "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45"
FACTORY = "0x1f98431c8ad98523631ae4a59f267346ea31f984"
QUOTER = "0x61ffe014ba17989e743c5f6cb21bf9697530b21e"
BTC_SCALE = Decimal(100_000_000)
USDT_SCALE = Decimal(1_000_000)
MAX_ORDER_USDT = Decimal("150")
MAX_SLIPPAGE_BPS = 50
ALMANAK_VERSION = "2.29.2"
ALMANAK_COMMIT = "39cfabb780d297f5f90e2f3ee68215dfb382b5f2"
