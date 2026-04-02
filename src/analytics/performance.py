"""
Analytics Engine: Computes performance metrics from backtest results.

Metrics computed:
- Total return
- CAGR (annualized return)
- Win rate
- Profit factor
- Average R multiple (net)
- Average win / average loss
- Max drawdown (absolute and %)
- Sharpe ratio (annualized)
- Sortino ratio (annualized)
- Number of trades
- Avg trade duration (bars)

Per-trade analysis:
- MFE/MAE distributions
- Exit reason breakdown

These metrics are intentionally computed from out-of-sample results only
during formal validation. The runner controls which data goes where.
"""

import math
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.backtest.engine import BacktestResult, TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Complete performance summary for a backtest run.

    All return metrics are in decimal (0.10 = 10%).
    All dollar metrics are in quote currency (USDT).
    """
    # --- Overview ---
    total_trades: int
    initial_equity: float
    final_equity: float
    total_return: float              # (final - initial) / initial

    # --- Trade statistics ---
    win_count: int
    loss_count: int
    win_rate: float                  # wins / total
    avg_win: float                   # Average net PnL of winning trades
    avg_loss: float                  # Average net PnL of losing trades (negative)
    profit_factor: float             # gross_profit / gross_loss
    avg_r_multiple: float            # Average PnL in R multiples
    expectancy_per_trade: float      # Expected PnL per trade in $

    # --- Risk ---
    max_drawdown_pct: float          # Max peak-to-trough % drawdown
    max_drawdown_dollar: float       # Max drawdown in $

    # --- Risk-adjusted ---
    sharpe_ratio: float              # Annualized (bar-level returns)
    sortino_ratio: float             # Annualized (downside deviation)

    # --- Time ---
    bars_in_market: int              # Total bars with open position
    exposure_pct: float              # bars_in_market / total_bars

    # --- Breakdown ---
    exit_breakdown: dict = field(default_factory=dict)  # {"stop_loss": N, "take_profit": N, ...}

    # --- Optional ---
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    strategy_name: str = ""
    label: str = ""  # e.g., "train", "validation", "test"

    def __str__(self) -> str:
        lines = [
            f"",
            f"{'='*60}",
            f"  Performance Report: {self.label or self.strategy_name}",
            f"  Period: {self.period_start} → {self.period_end}",
            f"{'='*60}",
            f"  Trades:          {self.total_trades}",
            f"  Win Rate:        {self.win_rate:.1%}  ({self.win_count}W / {self.loss_count}L)",
            f"  Profit Factor:   {self.profit_factor:.2f}",
            f"  Avg R Multiple:  {self.avg_r_multiple:+.2f}R",
            f"  Expectancy:      ${self.expectancy_per_trade:+.2f} per trade",
            f"",
            f"  Total Return:    {self.total_return:+.2%}",
            f"  Final Equity:    ${self.final_equity:,.2f}  (from ${self.initial_equity:,.2f})",
            f"  Max Drawdown:    {self.max_drawdown_pct:.1%}  (${self.max_drawdown_dollar:,.2f})",
            f"",
            f"  Sharpe Ratio:    {self.sharpe_ratio:.2f}",
            f"  Sortino Ratio:   {self.sortino_ratio:.2f}",
            f"",
            f"  Avg Win:         ${self.avg_win:+.2f}",
            f"  Avg Loss:        ${self.avg_loss:+.2f}",
            f"  Exposure:        {self.exposure_pct:.1%}",
            f"",
            f"  Exit Reasons:",
        ]
        for reason, count in sorted(self.exit_breakdown.items(), key=lambda x: -x[1]):
            lines.append(f"    {reason:<20} {count}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Flat dict for CSV export or comparison table."""
        return {
            "label": self.label,
            "strategy": self.strategy_name,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "avg_r_multiple": round(self.avg_r_multiple, 4),
            "expectancy": round(self.expectancy_per_trade, 2),
            "total_return_pct": round(self.total_return * 100, 2),
            "final_equity": round(self.final_equity, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "exposure_pct": round(self.exposure_pct, 4),
            **{f"exit_{k}": v for k, v in self.exit_breakdown.items()},
        }


class PerformanceAnalytics:
    """
    Computes PerformanceMetrics from a BacktestResult.

    Usage:
        analytics = PerformanceAnalytics()
        metrics = analytics.compute(result, label="test")
        print(metrics)
        analytics.save_trade_log(result, "results/trades.csv")
    """

    # Assume 15-minute bars: 4 bars/hour * 24 * 365 = 35,040 bars/year
    BARS_PER_YEAR = 35_040

    def compute(
        self,
        result: BacktestResult,
        label: str = "",
        strategy_name: str = "",
    ) -> PerformanceMetrics:
        """
        Compute all performance metrics from a BacktestResult.

        Args:
            result: Output from BacktestEngine.run().
            label: Descriptive label (e.g., "train", "val", "test").
            strategy_name: Name of the strategy for reporting.

        Returns:
            PerformanceMetrics dataclass with all metrics populated.
        """
        trades = result.trades
        equity_curve = result.equity_curve
        n_bars = len(equity_curve)

        # --- Period dates ---
        period_start = str(result.candles[0].timestamp.date()) if result.candles else None
        period_end = str(result.candles[-1].timestamp.date()) if result.candles else None

        # --- Basic trade stats ---
        total_trades = len(trades)
        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0.0

        avg_win = sum(t.net_pnl for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t.net_pnl for t in losses) / len(losses) if losses else 0.0

        gross_profit = sum(t.gross_pnl for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.gross_pnl for t in losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_r_multiple = (
            sum(t.pnl_r for t in trades) / total_trades if total_trades > 0 else 0.0
        )
        expectancy_per_trade = (
            sum(t.net_pnl for t in trades) / total_trades if total_trades > 0 else 0.0
        )

        # --- Return ---
        total_return = (result.final_equity - result.initial_equity) / result.initial_equity

        # --- Drawdown from equity curve ---
        max_dd_pct, max_dd_dollar = _compute_max_drawdown(equity_curve)

        # --- Sharpe / Sortino (bar-level returns, annualized) ---
        sharpe, sortino = _compute_risk_adjusted(equity_curve, self.BARS_PER_YEAR)

        # --- Exposure ---
        bars_in_market = _compute_bars_in_market(result)
        exposure_pct = bars_in_market / n_bars if n_bars > 0 else 0.0

        # --- Exit breakdown ---
        exit_breakdown: dict[str, int] = {}
        for t in trades:
            exit_breakdown[t.exit_reason] = exit_breakdown.get(t.exit_reason, 0) + 1

        return PerformanceMetrics(
            total_trades=total_trades,
            initial_equity=result.initial_equity,
            final_equity=result.final_equity,
            total_return=total_return,
            win_count=win_count,
            loss_count=loss_count,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_r_multiple=avg_r_multiple,
            expectancy_per_trade=expectancy_per_trade,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_dollar=max_dd_dollar,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            bars_in_market=bars_in_market,
            exposure_pct=exposure_pct,
            exit_breakdown=exit_breakdown,
            period_start=period_start,
            period_end=period_end,
            strategy_name=strategy_name,
            label=label,
        )

    def save_trade_log(
        self,
        result: BacktestResult,
        filepath: str | Path,
        label: str = "",
    ) -> None:
        """
        Save per-trade log to CSV.

        Columns include all TradeRecord fields plus derived metrics.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if not result.trades:
            logger.warning(f"No trades to save to {filepath}")
            return

        fieldnames = [
            "trade_id", "label", "direction",
            "entry_time", "entry_price", "entry_signal_price",
            "exit_time", "exit_price", "exit_reason",
            "stop_loss", "take_profit",
            "position_size",
            "entry_fee", "exit_fee", "total_fees",
            "gross_pnl", "net_pnl", "pnl_r",
            "mfe_dollar", "mae_dollar",
            "equity_before", "equity_after",
            "signal_reason",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in result.trades:
                writer.writerow({
                    "trade_id": t.trade_id,
                    "label": label,
                    "direction": t.direction,
                    "entry_time": t.entry_time,
                    "entry_price": round(t.entry_price, 4),
                    "entry_signal_price": round(t.entry_signal_price, 4),
                    "exit_time": t.exit_time,
                    "exit_price": round(t.exit_price, 4),
                    "exit_reason": t.exit_reason,
                    "stop_loss": round(t.stop_loss, 4),
                    "take_profit": round(t.take_profit, 4),
                    "position_size": round(t.position_size, 8),
                    "entry_fee": round(t.entry_fee, 4),
                    "exit_fee": round(t.exit_fee, 4),
                    "total_fees": round(t.entry_fee + t.exit_fee, 4),
                    "gross_pnl": round(t.gross_pnl, 4),
                    "net_pnl": round(t.net_pnl, 4),
                    "pnl_r": round(t.pnl_r, 4),
                    "mfe_dollar": round(t.max_favorable_excursion, 4),
                    "mae_dollar": round(t.max_adverse_excursion, 4),
                    "equity_before": round(t.equity_before, 4),
                    "equity_after": round(t.equity_after, 4),
                    "signal_reason": t.signal_reason,
                })

        logger.info(f"Trade log saved: {filepath} ({len(result.trades)} trades)")

    def save_equity_curve(
        self,
        result: BacktestResult,
        filepath: str | Path,
    ) -> None:
        """Save equity curve to CSV for charting."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "equity"])
            for i, (candle, equity) in enumerate(
                zip(result.candles, result.equity_curve)
            ):
                writer.writerow([candle.timestamp, round(equity, 4)])

        logger.info(f"Equity curve saved: {filepath}")

    def save_research_log(
        self,
        result: BacktestResult,
        filepath: str | Path,
        label: str = "",
    ) -> None:
        """
        Save a trade-level research CSV for manual behaviour analysis.

        Includes all standard trade fields plus every key from entry_context
        (the strategy's metadata dict captured at entry time). The union of
        all context keys across all trades is computed so folds with different
        keys don't crash or silently drop columns.

        Suitable for studying: why setups worked, pullback depth distribution,
        ATR regime at entry, long vs short performance breakdown, etc.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if not result.trades:
            logger.warning(f"No trades to save to {filepath}")
            return

        # Fixed columns that every trade has
        fixed_cols = [
            "trade_id", "label", "direction",
            "entry_time", "exit_time",
            "entry_price", "exit_price", "exit_reason",
            "stop_loss", "take_profit",
            "net_pnl", "pnl_r",
            "mfe_dollar", "mae_dollar",
            "total_fees",
        ]

        # Collect the union of all entry_context keys across all trades,
        # preserving insertion order (Python 3.7+ dict guarantee).
        context_keys: list[str] = list(
            dict.fromkeys(
                k for t in result.trades for k in t.entry_context.keys()
            )
        )

        fieldnames = fixed_cols + context_keys

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            for t in result.trades:
                row: dict = {
                    "trade_id":    t.trade_id,
                    "label":       label,
                    "direction":   t.direction,
                    "entry_time":  t.entry_time,
                    "exit_time":   t.exit_time,
                    "entry_price": round(t.entry_price, 4),
                    "exit_price":  round(t.exit_price, 4),
                    "exit_reason": t.exit_reason,
                    "stop_loss":   round(t.stop_loss, 4),
                    "take_profit": round(t.take_profit, 4),
                    "net_pnl":     round(t.net_pnl, 4),
                    "pnl_r":       round(t.pnl_r, 4),
                    "mfe_dollar":  round(t.max_favorable_excursion, 4),
                    "mae_dollar":  round(t.max_adverse_excursion, 4),
                    "total_fees":  round(t.entry_fee + t.exit_fee, 4),
                }
                row.update(t.entry_context)
                writer.writerow(row)

        logger.info(f"Research log saved: {filepath} ({len(result.trades)} trades)")

    def print_comparison_table(self, metrics_list: list[PerformanceMetrics]) -> None:
        """Print a side-by-side comparison table for multiple runs."""
        if not metrics_list:
            return

        keys = ["label", "total_trades", "win_rate", "profit_factor",
                "avg_r_multiple", "total_return_pct", "max_drawdown_pct",
                "sharpe_ratio"]

        # Header
        header = f"{'Metric':<25}" + "".join(f"{m.label or m.strategy_name:>15}" for m in metrics_list)
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))

        rows = {
            "Trades":           lambda m: m.total_trades,
            "Win Rate":         lambda m: f"{m.win_rate:.1%}",
            "Profit Factor":    lambda m: f"{m.profit_factor:.2f}",
            "Avg R":            lambda m: f"{m.avg_r_multiple:+.2f}",
            "Total Return":     lambda m: f"{m.total_return:+.1%}",
            "Max Drawdown":     lambda m: f"{m.max_drawdown_pct:.1%}",
            "Sharpe":           lambda m: f"{m.sharpe_ratio:.2f}",
            "Sortino":          lambda m: f"{m.sortino_ratio:.2f}",
            "Exposure":         lambda m: f"{m.exposure_pct:.1%}",
        }

        for row_name, fn in rows.items():
            row = f"{row_name:<25}" + "".join(f"{str(fn(m)):>15}" for m in metrics_list)
            print(row)

        print("=" * len(header))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_max_drawdown(equity_curve: List[float]) -> tuple[float, float]:
    """
    Compute maximum peak-to-trough drawdown.

    Returns (max_drawdown_pct, max_drawdown_dollar).
    Drawdown percentage is relative to the peak equity.
    """
    if not equity_curve:
        return 0.0, 0.0

    peak = equity_curve[0]
    max_dd_pct = 0.0
    max_dd_dollar = 0.0

    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd_dollar = peak - eq
        dd_pct = dd_dollar / peak if peak > 0 else 0.0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_dollar = dd_dollar

    return max_dd_pct, max_dd_dollar


def _compute_risk_adjusted(
    equity_curve: List[float],
    bars_per_year: int,
    risk_free_rate: float = 0.04,  # 4% annual risk-free rate
) -> tuple[float, float]:
    """
    Compute annualized Sharpe and Sortino ratios from bar-level equity.

    Sharpe  = (mean_return - rf_per_bar) / std_return * sqrt(bars_per_year)
    Sortino = (mean_return - rf_per_bar) / downside_std * sqrt(bars_per_year)
    """
    if len(equity_curve) < 2:
        return 0.0, 0.0

    # Bar-level returns
    returns = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i - 1] > 0:
            r = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            returns.append(r)

    if len(returns) < 2:
        return 0.0, 0.0

    rf_per_bar = risk_free_rate / bars_per_year
    mean_r = sum(returns) / len(returns)
    excess = mean_r - rf_per_bar

    variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    std = math.sqrt(variance)

    if std == 0:
        sharpe = 0.0
    else:
        sharpe = (excess / std) * math.sqrt(bars_per_year)

    # Sortino: downside deviation only (returns below 0)
    downside_returns = [r for r in returns if r < 0]
    if not downside_returns:
        sortino = float("inf")
    else:
        down_variance = sum(r ** 2 for r in downside_returns) / len(returns)
        down_std = math.sqrt(down_variance)
        sortino = (excess / down_std) * math.sqrt(bars_per_year) if down_std > 0 else 0.0

    return round(sharpe, 4), round(sortino, 4)


def _compute_bars_in_market(result: BacktestResult) -> int:
    """
    Count bars where a position was open.

    We approximate using trade entry/exit times matched against candles.
    This is an estimation — exact tracking would require per-bar position state.
    """
    if not result.trades or not result.candles:
        return 0

    total = 0
    for trade in result.trades:
        entry_ts = trade.entry_time
        exit_ts = trade.exit_time
        for candle in result.candles:
            if entry_ts <= candle.timestamp <= exit_ts:
                total += 1
    return total
