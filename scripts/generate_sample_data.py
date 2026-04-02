"""
Generate synthetic BTCUSDT 15-minute OHLCV data for testing.

The generated data simulates realistic crypto price action including:
- Long-term trend with mean reversion
- Volatility clustering (GARCH-like: high vol begets high vol)
- Occasional flash crashes and spikes
- Realistic OHLC construction (high/low relative to open/close)
- Volume correlated with volatility

This is NOT for strategy development (you'd overfit synthetic data).
It's purely to test the system components work end-to-end before
fetching real data from Binance.

Run:
    python scripts/generate_sample_data.py
    # Generates data/BTCUSDT_15m_sample.csv
"""

import csv
import math
import random
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_btcusdt_15m(
    n_bars: int = 20_000,
    start_price: float = 25_000.0,
    start_date: datetime = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    seed: int = 42,
) -> list:
    """
    Generate synthetic 15m OHLCV bars with semi-realistic price dynamics.

    Returns list of (timestamp, open, high, low, close, volume) tuples.
    """
    random.seed(seed)

    bars = []
    price = start_price
    ts = start_date
    interval = timedelta(minutes=15)

    # Volatility state (GARCH-like)
    vol_base = 0.003        # Base volatility per bar (0.3%)
    vol = vol_base
    vol_decay = 0.95        # Volatility mean-reversion speed
    vol_shock_scale = 0.002 # Size of vol shocks

    # Trend state
    trend = 0.0
    trend_strength = 0.0002  # Small directional drift per bar
    trend_change_prob = 0.002  # Probability of trend reversal per bar

    for i in range(n_bars):
        # --- Update volatility ---
        vol_shock = abs(random.gauss(0, vol_shock_scale))
        vol = vol * vol_decay + vol_base * (1 - vol_decay) + vol_shock

        # --- Update trend ---
        if random.random() < trend_change_prob:
            trend = random.gauss(0, trend_strength * 5)

        # --- Occasional regime events ---
        event_multiplier = 1.0
        if random.random() < 0.001:  # 0.1% chance of flash crash
            event_multiplier = random.uniform(-0.05, -0.02)
            vol *= 3
        elif random.random() < 0.001:  # 0.1% chance of spike
            event_multiplier = random.uniform(0.02, 0.05)
            vol *= 2

        # --- Generate close price ---
        bar_return = trend + random.gauss(0, vol)
        if event_multiplier != 1.0:
            bar_return += event_multiplier
        # Mean reversion: if price drifts far from a floating anchor, pull back
        # Use log-price mean reversion to keep the process stationary
        log_deviation = math.log(price / start_price)
        mean_reversion = -log_deviation * 0.005  # 0.5% pull per 1.0 log deviation
        bar_return += mean_reversion

        close_price = price * (1 + bar_return)
        close_price = max(close_price, start_price * 0.1)  # Floor at 10% of start

        open_price = price

        # --- Generate high/low ---
        # High and low are generated relative to the open-close range
        range_size = abs(close_price - open_price)
        extra_range = vol * price * random.uniform(0.3, 1.5)

        if close_price >= open_price:
            # Bullish bar: wick up is larger
            high = max(open_price, close_price) + extra_range * random.uniform(0.5, 1.0)
            low = min(open_price, close_price) - extra_range * random.uniform(0.1, 0.5)
        else:
            # Bearish bar: wick down is larger
            high = max(open_price, close_price) + extra_range * random.uniform(0.1, 0.5)
            low = min(open_price, close_price) - extra_range * random.uniform(0.5, 1.0)

        # Ensure OHLC consistency
        high = max(open_price, close_price, high)
        low = min(open_price, close_price, low)
        low = max(low, close_price * 0.001)  # Floor

        # --- Generate volume ---
        # Volume is higher when volatility is high
        base_volume = 50.0  # Base BTC volume per 15m bar
        volume = base_volume * (1 + vol / vol_base) * random.uniform(0.5, 2.0)
        volume = max(volume, 0.1)

        bars.append((
            ts.strftime("%Y-%m-%d %H:%M:%S"),
            round(open_price, 2),
            round(high, 2),
            round(low, 2),
            round(close_price, 2),
            round(volume, 4),
        ))

        # Advance
        price = close_price
        ts += interval

    return bars


def save_csv(bars: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        writer.writerows(bars)
    print(f"Saved {len(bars)} bars to {output_path}")
    print(f"  First bar: {bars[0][0]}  open={bars[0][1]}")
    print(f"  Last bar:  {bars[-1][0]}  close={bars[-1][3]}")
    print(f"  Price range: ${min(b[4] for b in bars):,.0f} - ${max(b[2] for b in bars):,.0f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic OHLCV data")
    parser.add_argument("--bars", type=int, default=20_000, help="Number of bars (default: 20000 ≈ 208 days)")
    parser.add_argument("--start-price", type=float, default=25_000.0)
    parser.add_argument("--output", default="data/BTCUSDT_15m_sample.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    output_path = project_root / args.output

    print(f"Generating {args.bars} synthetic 15m bars...")
    bars = generate_btcusdt_15m(
        n_bars=args.bars,
        start_price=args.start_price,
        seed=args.seed,
    )
    save_csv(bars, output_path)
    print(f"\nRun backtest with:")
    print(f"  python scripts/run_backtest.py --data {args.output}")
