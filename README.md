# Trading Bot V2 — Quantitative Crypto Backtesting System

A professional-grade, research-driven backtesting framework for intraday crypto trading strategies. Zero external dependencies. Designed to reveal whether an edge exists, not to overfit.

---

## Architecture

```
trading-bot-v2/
├── src/
│   ├── data/
│   │   └── loader.py          # DataLoader, Candle, CSV parsing, train/val/test split
│   ├── strategy/
│   │   ├── base.py            # BaseStrategy, Signal dataclass
│   │   ├── indicators.py      # RSI, Bollinger Bands, ATR, SMA, EMA (pure Python)
│   │   └── mean_reversion.py  # V1: RSI + BB mean reversion (long only)
│   ├── backtest/
│   │   └── engine.py          # BacktestEngine, TradeRecord, position state machine
│   ├── risk/
│   │   └── manager.py         # RiskManager: position sizing, daily loss limit, cooldown
│   ├── analytics/
│   │   └── performance.py     # PerformanceAnalytics, PerformanceMetrics
│   └── runner/
│       └── experiment.py      # StandardRunner, WalkForwardRunner, ParameterSweepRunner
├── scripts/
│   ├── fetch_data.py          # Download OHLCV from Binance (no auth required)
│   ├── generate_sample_data.py # Generate synthetic data for testing
│   └── run_backtest.py        # Main entry point
├── data/                      # CSV files (gitignored)
└── results/                   # Output trade logs and equity curves
```

---

## Quick Start

### 1. Get data (one of two options)

**Option A: Download real data from Binance**
```bash
python3 scripts/fetch_data.py --symbol BTCUSDT --interval 15m --days 730
# Output: data/BTCUSDT_15m.csv
```

**Option B: Generate synthetic data for testing**
```bash
python3 scripts/generate_sample_data.py --bars 25000
# Output: data/BTCUSDT_15m_sample.csv
```

### 2. Run backtest

```bash
# Basic run (train/val/test split)
python3 scripts/run_backtest.py --data data/BTCUSDT_15m.csv

# With walk-forward validation
python3 scripts/run_backtest.py --data data/BTCUSDT_15m.csv --walk-forward

# With parameter sweep (training data only)
python3 scripts/run_backtest.py --data data/BTCUSDT_15m.csv --sweep

# Custom parameters
python3 scripts/run_backtest.py --data data/BTCUSDT_15m.csv \
    --equity 10000 --risk 0.01 --rsi-oversold 28 --r-multiple 2.5

# Date range filter
python3 scripts/run_backtest.py --data data/BTCUSDT_15m.csv \
    --start 2022-01-01 --end 2024-01-01
```

---

## V1 Strategy: Mean Reversion

**Market**: BTCUSDT 15-minute candles  
**Direction**: Long only (v1)

### Entry Conditions (all must be true)

1. **RSI(14) < 30** — momentum is oversold
2. **Close ≤ Lower Bollinger Band(20, 2σ)** — price at statistical extreme
3. **Recent drop ≥ 1%** (optional, configurable) — confirms momentum, not noise

### Position Management

| Parameter | Default | Meaning |
|-----------|---------|---------|
| Stop Loss | Swing low (5 bars) - 0.2% buffer | Invalidation level if reversion fails |
| Take Profit | Entry + 2R | Fixed 2:1 reward/risk |
| Cooldown | 3 bars after exit | Prevent revenge trading |
| Risk/trade | 1% of equity | Fixed fractional sizing |

### Why Mean Reversion?

Mean reversion works in ranging markets when two independent indicators confirm an extreme simultaneously. The dual confirmation (RSI + BB lower band) reduces false entries versus using either signal alone. The stop below recent swing low respects market structure — a break of that low genuinely invalidates the trade thesis.

---

## Validation System

### Anti-Overfitting Design

| Practice | How It's Implemented |
|----------|----------------------|
| No lookahead bias | Strategy only reads `candles[0..index]` — enforced at the API level |
| No bar-close lookahead | Fills execute at the **next bar's open + slippage**, not the signal bar's close |
| Temporal data split | Train/Val/Test split respects time order — never shuffled |
| Walk-forward testing | Multiple rolling OOS windows, aggregated results |
| Separate sweep/test | Parameter sweep runs only on **training** data, never test |
| Realistic costs | Fee (0.1%/side) + slippage (0.05%/fill) applied at every trade |

### Train / Val / Test Split (60/20/20)

```
|--- 60% Training ---|-- 20% Val --|-- 20% Test --|
         ↑                  ↑            ↑
   Strategy development   Tune      Final answer
   Parameter ranges       thresholds  (look once)
```

**Rule**: Never touch the test set until you're done developing. Looking at test performance multiple times and adjusting based on it is data snooping.

### Walk-Forward Testing

Rolls a train/test window forward through time:
```
Fold 1: [train 0-8000]  test [8000-10000]
Fold 2: [train 2000-10000]  test [10000-12000]
Fold 3: [train 4000-12000]  test [12000-14000]
...
```
Aggregated results across all OOS windows are more statistically robust than any single test period.

---

## Risk Management

All sizing is based on **risk amount**, not notional:

```
risk_amount  = equity × risk_pct_per_trade  (default 1%)
risk_per_unit = entry_price - stop_loss
position_size = risk_amount / risk_per_unit
```

If the trade hits stop loss exactly, you lose exactly 1% of equity — no more.

| Guard | Default |
|-------|---------|
| Max risk per trade | 2% (hard cap) |
| Daily loss limit | 5% — no new trades for the rest of the day |
| Trade cooldown | 3 bars after any exit |
| Max leverage | 1x (no leverage in v1) |

---

## Output Files

After a backtest run, `results/<experiment_name>/` contains:

| File | Contents |
|------|----------|
| `train_trades.csv` | Per-trade log for training period |
| `val_trades.csv` | Per-trade log for validation period |
| `test_trades.csv` | Per-trade log for test period |
| `*_equity.csv` | Bar-by-bar equity curve (for charting) |
| `summary.csv` | Side-by-side metrics comparison |

### Trade Log Columns

`trade_id, direction, entry_time, entry_price, exit_time, exit_price, exit_reason, stop_loss, take_profit, position_size, entry_fee, exit_fee, gross_pnl, net_pnl, pnl_r, mfe_dollar, mae_dollar, equity_before, equity_after, signal_reason`

### Key Metrics Explained

| Metric | Good threshold | Meaning |
|--------|---------------|---------|
| Profit Factor | > 1.5 | Gross profit / gross loss |
| Win Rate | > 40% at 2R | % of trades that are winners |
| Avg R Multiple | > 0 | Average PnL in risk units |
| Max Drawdown | < 20% | Largest peak-to-trough equity loss |
| Sharpe Ratio | > 1.0 | Risk-adjusted return (annualized) |

---

## Adding New Strategies

1. Create `src/strategy/my_strategy.py`
2. Subclass `BaseStrategy` and implement `compute_signal(candles, index)`
3. Use only `candles[0..index]` — never beyond
4. Precompute indicators once (cache on `id(candles)`) for performance
5. Return a `Signal(direction, entry_price, stop_loss, take_profit)` or `None`

```python
from src.strategy.base import BaseStrategy, Signal

class BreakoutStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20):
        super().__init__(name="Breakout_v1")
        self.lookback = lookback

    def compute_signal(self, candles, index):
        if index < self.lookback:
            return None
        # Your logic here — only use candles[0..index]
        ...
```

6. Pass it to `StandardRunner.run(candles, strategy)`

---

## Roadmap

- [ ] Regime detection (trending vs ranging) to gate strategy entries
- [ ] Breakout strategy (ATR channel breakout)
- [ ] Trend-following strategy (EMA crossover + ADX filter)
- [ ] Multi-symbol support
- [ ] Live trading execution layer (Binance API integration)
- [ ] Visualization module (equity curve, trade markers)
- [ ] Monte Carlo simulation for drawdown estimation
