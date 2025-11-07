# Lighter Grid Trading Bot

Automated grid trading bot for Lighter.xyz DEX that places buy/sell orders in a grid pattern with auto-refill and smart cleanup.

## Features

- **Grid Trading Strategy** - BUY LOW → SELL HIGH with configurable spacing
- **Auto-Refill** - Replaces filled orders automatically
- **Smart Cleanup** - Cancels orders and closes positions on exit (Ctrl+C)
- **Multi-Market Support** - 90+ markets (see `Market_id.txt`)
- **Dynamic Decimals** - Auto-detects price precision per market
- **Adjustable Grid Spacing** - Configure profit per cycle (0.5% to 5%+)

## Quick Start

### 1. Install Dependencies
```bash
pip3 install lighter-python python-dotenv requests
```

### 2. Configure .env File
```bash
# Lighter API Credentials
API_KEY_PRIVATE_KEY=0xYOUR_PRIVATE_KEY_HERE
ACCOUNT_INDEX=312892
API_KEY_INDEX=2
BASE_URL=https://mainnet.zklighter.elliot.ai

# Grid Bot Settings
MARKET_INDEX=9              # See Market_id.txt for all markets
DIRECTION=NEUTRAL           # LONG / NEUTRAL / SHORT
LEVERAGE=2                  # 1-25x
GRID_COUNT=10               # Number of grid levels
INVESTMENT_USDC=100         # Investment amount in USDC
GRID_SPACING_PERCENT=1.0    # Spacing between grids (default: 1.0% = 1% profit per cycle)
```

### 3. Run the Bot
```bash
python3 grid.py
```

Press `Ctrl+C` to stop (auto-cleanup enabled)

## Configuration

### Market Selection
See `Market_id.txt` for all 90+ available markets (ETH, BTC, SOL, DOGE, WLD, AVAX, meme coins, DeFi tokens, etc.)

### Order Size Calculator
**Minimum order size: $15-20 per order**

Formula: `Investment Needed = (Min Order Size × Grid Count) / Leverage`

Example:
- 10 grids at 2x leverage = $100 investment → $20 per order ✓
- 20 grids at 2x leverage = $200 investment → $20 per order ✓

## How It Works

1. **Setup** - Connects to Lighter, fetches market price and decimals
2. **Place Grid** - Creates buy orders below and sell orders above market price
3. **Monitor** - Checks orders every 2 seconds
4. **Refill** - When BUY fills → place SELL higher | When SELL fills → place BUY lower (by GRID_SPACING_PERCENT)
5. **Cleanup** - On Ctrl+C, cancels all orders and closes positions

**Grid Spacing Examples:**
- `GRID_SPACING_PERCENT=0.5` → 0.5% spacing (tighter grid, more trades, less profit per cycle)
- `GRID_SPACING_PERCENT=1.0` → 1% spacing (default, balanced)
- `GRID_SPACING_PERCENT=2.0` → 2% spacing (wider grid, fewer trades, more profit per cycle)

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Order price flagged as accidental | Increase `INVESTMENT_USDC` or decrease `GRID_COUNT` |
| Only SELL orders appear | BUY orders rejected (too small) - increase investment |
| Private key does not match | Check `API_KEY_INDEX` (usually 0, 1, or 2) |
| Rate limit exceeded | Keep `GRID_COUNT` under 40 |

## Security
- **NEVER share `.env`** - contains your private key
- Start with small amounts ($50-100)
- Monitor regularly

## Resources
- Lighter Docs: https://docs.lighter.xyz
- API Docs: https://apidocs.lighter.xyz
- Discord: https://discord.gg/lighter

---

**Trade responsibly. Never risk more than you can afford to lose.**
