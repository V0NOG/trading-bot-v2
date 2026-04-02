"""
Experiment Runner: Orchestrates backtests across time windows and parameters.

Two main runners:

1. StandardRunner:
   - Runs a single backtest on train/val/test splits
   - Prints metrics for each split
   - Saves trade logs and equity curves

2. WalkForwardRunner:
   - Rolls a train+test window forward through time
   - Each fold: train on N months, test on M months
   - Aggregates out-of-sample trades across all folds
   - This is the primary anti-overfitting validation method

3. ParameterSweepRunner:
   - Runs the same strategy with different parameter combinations
   - Reports metrics for each combination
   - WARNING: Only run sweeps on the TRAINING set.
     Never select parameters based on test set performance.
     That is data snooping.

Walk-Forward Testing Rationale:
    A single train/test split can get lucky or unlucky depending on
    which period you happen to test on. Walk-forward testing:
    - Uses multiple non-overlapping test periods
    - Each test period is truly out-of-sample (never seen during "optimization")
    - Aggregated results are more statistically robust
    - Reveals if strategy performance degrades over time (concept drift)
"""

import logging
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.data.loader import DataLoader, Candle
from src.strategy.base import BaseStrategy
from src.backtest.engine import BacktestEngine, BacktestResult
from src.risk.manager import RiskManager
from src.analytics.performance import PerformanceAnalytics, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a backtest experiment."""
    name: str
    initial_equity: float = 10_000.0
    fee_rate: float = 0.001          # 0.1% taker fee
    slippage_pct: float = 0.0005     # 0.05% slippage
    risk_pct_per_trade: float = 0.01 # 1% risk per trade
    cooldown_bars: int = 3
    use_daily_loss_limit: bool = True
    max_daily_loss_pct: float = 0.05
    output_dir: str = "results"


class StandardRunner:
    """
    Runs a single experiment with train/val/test splits and reports metrics.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.analytics = PerformanceAnalytics()

    def run(
        self,
        candles: List[Candle],
        strategy: BaseStrategy,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ) -> dict[str, PerformanceMetrics]:
        """
        Run backtest on train/val/test splits.

        Args:
            candles: Full sorted candle list.
            strategy: Strategy to test.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.

        Returns:
            Dict with keys "train", "val", "test" → PerformanceMetrics.
        """
        loader = DataLoader()
        train, val, test = loader.split(candles, train_ratio, val_ratio)

        splits = [
            ("train", train),
            ("val", val),
            ("test", test),
        ]

        results: dict[str, PerformanceMetrics] = {}
        output_dir = Path(self.config.output_dir) / self.config.name
        output_dir.mkdir(parents=True, exist_ok=True)

        for label, split_candles in splits:
            if not split_candles:
                logger.warning(f"Empty {label} split, skipping")
                continue

            logger.info(
                f"Running {label} backtest: {len(split_candles)} bars "
                f"({split_candles[0].timestamp.date()} to "
                f"{split_candles[-1].timestamp.date()})"
            )

            engine = BacktestEngine(
                initial_equity=self.config.initial_equity,
                fee_rate=self.config.fee_rate,
                slippage_pct=self.config.slippage_pct,
            )
            risk_mgr = RiskManager(
                risk_pct_per_trade=self.config.risk_pct_per_trade,
                cooldown_bars=self.config.cooldown_bars,
                use_daily_loss_limit=self.config.use_daily_loss_limit,
                max_daily_loss_pct=self.config.max_daily_loss_pct,
            )

            result = engine.run(split_candles, strategy, risk_mgr)
            metrics = self.analytics.compute(
                result, label=label, strategy_name=strategy.name
            )
            results[label] = metrics

            # Save outputs
            self.analytics.save_trade_log(
                result, output_dir / f"{label}_trades.csv", label=label
            )
            self.analytics.save_equity_curve(
                result, output_dir / f"{label}_equity.csv"
            )
            self.analytics.save_research_log(
                result, output_dir / f"{label}_research.csv", label=label
            )

            print(metrics)

        # Print comparison table
        if len(results) > 1:
            print(f"\n{'='*60}")
            print(f"  COMPARISON: Train vs Val vs Test")
            self.analytics.print_comparison_table(list(results.values()))

        # Save summary
        self._save_summary(results, output_dir)

        return results

    def _save_summary(
        self,
        results: dict[str, PerformanceMetrics],
        output_dir: Path,
    ) -> None:
        filepath = output_dir / "summary.csv"
        rows = [m.to_dict() for m in results.values()]
        if not rows:
            return
        all_keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, restval="")
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Summary saved: {filepath}")


class WalkForwardRunner:
    """
    Walk-forward tester: rolls train/test window through time.

    Example with train_bars=5000, test_bars=1000, step_bars=1000:

        Fold 1: train [0..5000),    test [5000..6000)
        Fold 2: train [1000..6000), test [6000..7000)
        Fold 3: train [2000..7000), test [7000..8000)
        ...

    The step controls how far the window advances each fold.
    Anchored mode (anchored=True): training window grows from index 0 instead
    of sliding. Useful when more data is always better for the strategy.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.analytics = PerformanceAnalytics()

    def run(
        self,
        candles: List[Candle],
        strategy: BaseStrategy,
        train_bars: int = 5000,
        test_bars: int = 1000,
        step_bars: int = 1000,
        anchored: bool = False,
    ) -> dict:
        """
        Run walk-forward backtests and aggregate out-of-sample results.

        Args:
            candles: Full sorted candle list.
            strategy: Strategy instance.
            train_bars: Number of bars in training window.
            test_bars: Number of bars in each test window.
            step_bars: How far to advance the window each fold.
            anchored: If True, training always starts at index 0 (expanding window).

        Returns:
            Dict with fold metrics and aggregated OOS metrics.
        """
        n = len(candles)
        fold_results: List[PerformanceMetrics] = []
        all_oos_trades = []

        output_dir = Path(self.config.output_dir) / self.config.name / "walk_forward"
        output_dir.mkdir(parents=True, exist_ok=True)

        fold = 0
        train_start = 0
        train_end = train_bars

        while train_end + test_bars <= n:
            test_start = train_end
            test_end = test_start + test_bars

            train_slice = candles[train_start:train_end]
            test_slice = candles[test_start:test_end]

            fold += 1
            logger.info(
                f"Walk-forward fold {fold}: "
                f"train [{train_start}:{train_end}] "
                f"({train_slice[0].timestamp.date()} to "
                f"{train_slice[-1].timestamp.date()}), "
                f"test [{test_start}:{test_end}] "
                f"({test_slice[0].timestamp.date()} to "
                f"{test_slice[-1].timestamp.date()})"
            )

            # Run test fold (OOS)
            engine = BacktestEngine(
                initial_equity=self.config.initial_equity,
                fee_rate=self.config.fee_rate,
                slippage_pct=self.config.slippage_pct,
            )
            risk_mgr = RiskManager(
                risk_pct_per_trade=self.config.risk_pct_per_trade,
                cooldown_bars=self.config.cooldown_bars,
                use_daily_loss_limit=self.config.use_daily_loss_limit,
                max_daily_loss_pct=self.config.max_daily_loss_pct,
            )

            result = engine.run(test_slice, strategy, risk_mgr)
            metrics = self.analytics.compute(
                result,
                label=f"fold_{fold}_oos",
                strategy_name=strategy.name,
            )
            fold_results.append(metrics)
            all_oos_trades.extend(result.trades)
            # Save per-fold research log
            self.analytics.save_research_log(
                result,
                output_dir / f"fold_{fold}_research.csv",
                label=f"fold_{fold}_oos",
            )

            print(f"\n[Fold {fold}] OOS: "
                  f"trades={metrics.total_trades} "
                  f"win_rate={metrics.win_rate:.1%} "
                  f"profit_factor={metrics.profit_factor:.2f} "
                  f"return={metrics.total_return:+.1%}")

            # Advance window
            if anchored:
                train_end += step_bars
            else:
                train_start += step_bars
                train_end += step_bars

        if not fold_results:
            logger.warning("Walk-forward: no folds completed (insufficient data)")
            return {}

        # --- Aggregate OOS stats ---
        print(f"\n{'='*60}")
        print(f"  WALK-FORWARD SUMMARY: {fold} folds, {len(all_oos_trades)} total OOS trades")
        print(f"{'='*60}")
        self.analytics.print_comparison_table(fold_results)

        # Save fold summary
        rows = [m.to_dict() for m in fold_results]
        if rows:
            filepath = output_dir / "fold_summary.csv"
            # Compute the union of all keys across all fold rows so that optional
            # keys (e.g. exit_take_profit absent in folds with zero TPs) don't
            # crash DictWriter. restval="" writes an empty cell for missing keys.
            all_keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys, restval="")
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"Walk-forward fold summary saved: {filepath}")

        return {
            "folds": fold_results,
            "total_oos_trades": len(all_oos_trades),
            "avg_win_rate": sum(m.win_rate for m in fold_results) / len(fold_results),
            "avg_profit_factor": sum(m.profit_factor for m in fold_results) / len(fold_results),
        }


class ParameterSweepRunner:
    """
    Runs the same strategy with varying parameters to find robust settings.

    IMPORTANT: Only sweep on training data. Never use test/validation
    performance to select parameters — that is lookahead bias at the
    experiment level (data snooping / multiple comparison bias).

    Use this to understand the parameter sensitivity landscape, not to
    pick the "best" parameters for live trading without OOS validation.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.analytics = PerformanceAnalytics()

    def run(
        self,
        candles: List[Candle],
        strategy_factory,  # Callable that returns a strategy given **params
        param_grid: List[dict],
        label: str = "sweep",
    ) -> List[dict]:
        """
        Run backtest for each parameter combination.

        Args:
            candles: Candle list (should be TRAINING data only).
            strategy_factory: Callable(**params) → BaseStrategy.
            param_grid: List of parameter dicts to try.
            label: Label prefix for output files.

        Returns:
            List of result dicts with params + metrics.
        """
        results = []
        output_dir = Path(self.config.output_dir) / self.config.name / "sweep"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, params in enumerate(param_grid):
            try:
                strategy = strategy_factory(**params)
            except Exception as e:
                logger.warning(f"Sweep {i}: failed to create strategy with {params}: {e}")
                continue

            engine = BacktestEngine(
                initial_equity=self.config.initial_equity,
                fee_rate=self.config.fee_rate,
                slippage_pct=self.config.slippage_pct,
            )
            risk_mgr = RiskManager(
                risk_pct_per_trade=self.config.risk_pct_per_trade,
            )

            result = engine.run(candles, strategy, risk_mgr)
            metrics = self.analytics.compute(
                result, label=f"{label}_{i}", strategy_name=strategy.name
            )

            row = {**params, **metrics.to_dict()}
            results.append(row)

            logger.info(
                f"Sweep {i+1}/{len(param_grid)}: {params} → "
                f"trades={metrics.total_trades} "
                f"pf={metrics.profit_factor:.2f} "
                f"wr={metrics.win_rate:.1%}"
            )

        # Save sweep results
        if results:
            filepath = output_dir / f"{label}_results.csv"
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            logger.info(f"Sweep results saved: {filepath}")

        return results
