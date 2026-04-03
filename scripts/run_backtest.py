"""
Main entry point: Run a full backtest with train/val/test split and walk-forward.

Strategy: Momentum Continuation (TrendPullbackStrategy)
  - EMA50 / EMA200 trend filter
  - Impulse bar detection (range > 1.5×ATR)
  - 1-3 bar pullback, then resume entry
  - Stop 1.2×ATR, target 2.5R

Usage:
    # Standard train/val/test
    python scripts/run_backtest.py --data data/BTCUSDT_15m.csv

    # Walk-forward validation
    python scripts/run_backtest.py --data data/BTCUSDT_15m.csv --walk-forward

    # Both modes
    python scripts/run_backtest.py --data data/BTCUSDT_15m.csv --walk-forward

    # Custom parameters
    python scripts/run_backtest.py --data data/BTCUSDT_15m.csv \\
        --impulse-atr-mult 1.5 --r-multiple 2.5 --atr-stop-mult 1.2

    # Date range filter
    python scripts/run_backtest.py --data data/BTCUSDT_15m.csv \\
        --start 2022-01-01 --end 2024-01-01
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader

# ── Strategy under evaluation ────────────────────────────────────────────────
# This is the ONLY strategy used throughout this research script.
# There is no fallback, no factory, no default to a different strategy.
from src.strategy.compression_breakout import CompressionBreakoutStrategy
# ─────────────────────────────────────────────────────────────────────────────

from src.backtest.engine import BacktestEngine
from src.risk.manager import RiskManager
from src.analytics.performance import PerformanceAnalytics
from src.runner.experiment import (
    ExperimentConfig,
    StandardRunner,
    WalkForwardRunner,
)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        logging.getLogger("src.backtest.engine").setLevel(logging.WARNING)
        logging.getLogger("src.strategy.trend_pullback").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compression Breakout strategy backtest on BTCUSDT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data
    parser.add_argument("--data", default="data/BTCUSDT_15m.csv",
                        help="Path to OHLCV CSV (default: data/BTCUSDT_15m.csv)")
    parser.add_argument("--start", help="Start date filter YYYY-MM-DD")
    parser.add_argument("--end",   help="End date filter YYYY-MM-DD")

    # Experiment
    parser.add_argument("--name",     default="compression_breakout_v2",
                        help="Experiment name (used for output directory)")
    parser.add_argument("--equity",   type=float, default=10_000.0,
                        help="Initial equity in USD (default: 10000)")
    parser.add_argument("--risk",     type=float, default=0.01,
                        help="Risk per trade as decimal (default: 0.01 = 1%%)")
    parser.add_argument("--fee",      type=float, default=0.001,
                        help="Taker fee per side (default: 0.001 = 0.1%%)")
    parser.add_argument("--slippage", type=float, default=0.0005,
                        help="Slippage per fill (default: 0.0005 = 0.05%%)")

    # Strategy parameters
    parser.add_argument("--impulse-atr-mult", type=float, default=1.5,
                        help="Impulse: candle range must exceed N×ATR (default: 1.5)")
    parser.add_argument("--atr-stop-mult",    type=float, default=2.0,
                        help="Stop loss = N×ATR from entry (default: 2.0)")
    parser.add_argument("--r-multiple",       type=float, default=2.5,
                        help="Take profit R multiple (default: 2.5)")

    parser.add_argument("--atr-min-pct",      type=float, default=0.003,
                        help="Minimum ATR/close for volatility filter (default: 0.003)")
    
    parser.add_argument("--compression-lookback", type=int, default=8,
                        help="Bars in the tight pre-breakout range")
    parser.add_argument("--structure-lookback", type=int, default=24,
                        help="Broader lookback used to measure relative compression")

    parser.add_argument("--compression-ratio-threshold", type=float, default=0.60,
                        help="Recent range / prior range must be <= this value")
    parser.add_argument("--compression-max-atr-multiple", type=float, default=2.2,
                        help="Reject setups where the recent base is already too wide in ATR terms")

    parser.add_argument("--breakout-buffer-atr", type=float, default=0.05,
                        help="ATR buffer added beyond the squeeze range for breakout confirmation")
    parser.add_argument("--breakout-min-range-atr", type=float, default=0.90,
                        help="Breakout bar must have at least this many ATRs of total range")
    parser.add_argument("--breakout-close-location-min", type=float, default=0.65,
                        help="For longs, close must finish in top X%% of bar; inverse for shorts")

    parser.add_argument("--atr-expansion-lookback", type=int, default=5,
                        help="Compare current ATR to ATR N bars ago")
    parser.add_argument("--atr-expansion-min-ratio", type=float, default=1.00,
                        help="Current ATR must be at least this multiple of ATR N bars ago")

    # Validation modes
    parser.add_argument("--walk-forward", action="store_true",
                        help="Also run walk-forward validation")

    # Output
    parser.add_argument("--output-dir", default="results",
                        help="Output base directory (default: results)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")

    return parser.parse_args()


def print_strategy_banner(strategy) -> None:
    """Print an unambiguous confirmation of which strategy is running."""
    print()
    print("=" * 60)
    print("  STRATEGY CONFIRMATION")
    print(f"  Class : {strategy.__class__.__name__}")
    print(f"  Name  : {strategy.name}")
    print(f"  Module: src/strategy/compression_breakout.py")
    print("  Parameters:")
    for k, v in strategy.parameter_dict().items():
        print(f"    {k:<25} {v}")
    print("=" * 60)
    print()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    data_path = Path(args.data)
    if not data_path.is_absolute():
        project_root = Path(__file__).parent.parent
        data_path = project_root / data_path

    if not data_path.exists():
        print(f"\nERROR: Data file not found: {data_path}")
        print("Fetch real data with:")
        print("  python scripts/fetch_data.py --symbol BTCUSDT --interval 15m --days 730")
        sys.exit(1)

    loader = DataLoader(strict=True)

    start_date = (
        datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.start else None
    )
    end_date = (
        datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end else None
    )

    print(f"\nLoading data from: {data_path}")
    candles = loader.load(data_path, start_date=start_date, end_date=end_date)
    print(f"  {len(candles)} candles: "
          f"{candles[0].timestamp.date()} → {candles[-1].timestamp.date()}")

    # ----------------------------------------------------------------
    # Build strategy — TrendPullbackStrategy only
    # ----------------------------------------------------------------
    strategy = CompressionBreakoutStrategy(
        compression_lookback=args.compression_lookback,
        structure_lookback=args.structure_lookback,

        compression_ratio_threshold=args.compression_ratio_threshold,
        compression_max_atr_multiple=args.compression_max_atr_multiple,

        breakout_buffer_atr=args.breakout_buffer_atr,
        breakout_min_range_atr=args.breakout_min_range_atr,
        breakout_close_location_min=args.breakout_close_location_min,

        atr_expansion_lookback=args.atr_expansion_lookback,
        atr_expansion_min_ratio=args.atr_expansion_min_ratio,

        atr_stop_mult=args.atr_stop_mult,
        r_multiple=args.r_multiple,
        atr_min_pct=args.atr_min_pct,
    )

    print_strategy_banner(strategy)

    # ----------------------------------------------------------------
    # Experiment config
    # ----------------------------------------------------------------
    config = ExperimentConfig(
        name              = args.name,
        initial_equity    = args.equity,
        fee_rate          = args.fee,
        slippage_pct      = args.slippage,
        risk_pct_per_trade= args.risk,
        cooldown_bars     = 5,             # fixed for momentum continuation
        output_dir        = str(Path(__file__).parent.parent / args.output_dir),
    )

    # ----------------------------------------------------------------
    # Walk-forward (optional, run first so standard run always executes)
    # ----------------------------------------------------------------
    if args.walk_forward:
        print("=" * 60)
        print("  WALK-FORWARD VALIDATION")
        print("=" * 60)
        wf_runner = WalkForwardRunner(config)
        wf_runner.run(
            candles,
            strategy,
            train_bars = 8000,   # ~83 days of 15m bars
            test_bars  = 2000,   # ~20 days per fold
            step_bars  = 2000,
        )

    # ----------------------------------------------------------------
    # Standard train / validation / test backtest (always runs)
    # ----------------------------------------------------------------
    print("=" * 60)
    print("  STANDARD BACKTEST: Train / Validation / Test")
    print("=" * 60)
    runner = StandardRunner(config)
    runner.run(candles, strategy, train_ratio=0.6, val_ratio=0.2)

    out = Path(config.output_dir) / config.name
    print(f"\nOutputs saved to: {out}/")
    print("  Standard:      train/val/test _trades.csv, _equity.csv, _research.csv")
    print("  Summary:       summary.csv")
    if args.walk_forward:
        print("  Walk-forward:  walk_forward/fold_*_research.csv, fold_summary.csv")


if __name__ == "__main__":
    main()
