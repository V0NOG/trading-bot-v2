"""
V1 Strategy: Mean Reversion on 15-minute BTCUSDT

Logic:
    ENTRY (Long only):
        1. RSI(14) < rsi_oversold (default 30) — momentum oversold
        2. Close <= lower Bollinger Band (20, 2.0) — price at statistical extreme
        3. Optional: close dropped >= drop_threshold% in recent lookback bars
        4. No active position
        5. Not in cooldown period after last trade exit

    STOP LOSS:
        - Below the lowest low of the last 'sl_lookback' bars minus a buffer
        - If that level is unreasonably far, fall back to fixed % stop
        - Enforces a minimum stop distance to avoid tiny stops getting noise-triggered

    TAKE PROFIT:
        - Entry + (risk_distance * r_multiple) — fixed R multiple target
        - Default: 2R

Rationale:
    Mean reversion works in ranging markets when RSI and BB confirm
    extreme conditions simultaneously. The dual confirmation (RSI + BB touch)
    reduces false entries compared to using either signal alone.

    The stop below recent swing low respects market structure. A break of that
    low invalidates the reversion hypothesis.

Performance note:
    Indicators are precomputed once over the full candle list on first call,
    then cached. This makes the backtester O(n) per run instead of O(n²).
    The cache is keyed by the candle list id, so separate runs on different
    candle slices each get their own cache.

Parameters:
    All configurable — designed for parameter sweeps in the experiment runner.
"""

import math
import logging
from typing import List, Optional

from src.data.loader import Candle
from src.strategy.base import BaseStrategy, Signal
from src.strategy.indicators import rsi, bollinger_bands, rolling_min_low

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion: enter long when price is oversold (RSI + BB lower band).

    This is a V1 long-only strategy. Short entries are intentionally excluded
    because crypto has a structural long bias and short entries during
    trending markets produce catastrophic drawdowns.
    """

    def __init__(
        self,
        # RSI parameters
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        # Bollinger Band parameters
        bb_period: int = 20,
        bb_std: float = 2.0,
        # Stop loss parameters
        sl_lookback: int = 5,           # Bars to look back for swing low
        sl_buffer_pct: float = 0.002,   # Extra buffer below swing low (0.2%)
        sl_max_pct: float = 0.03,       # Max allowed stop distance (3%) — fallback protection
        sl_min_pct: float = 0.003,      # Min stop distance (0.3%) — avoid trivial stops
        # Take profit
        r_multiple: float = 2.0,        # TP = entry + r_multiple * risk_distance
        # Entry filter: optional recent drop threshold
        use_drop_filter: bool = True,
        drop_lookback: int = 4,         # Bars to look back for drop
        drop_threshold_pct: float = 0.01,  # Minimum drop to qualify (1%)
        # Minimum lookback before first signal
        warmup_bars: int = 30,
        name: str = "MeanReversion_v1",
    ):
        super().__init__(name=name)
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.sl_lookback = sl_lookback
        self.sl_buffer_pct = sl_buffer_pct
        self.sl_max_pct = sl_max_pct
        self.sl_min_pct = sl_min_pct
        self.r_multiple = r_multiple
        self.use_drop_filter = use_drop_filter
        self.drop_lookback = drop_lookback
        self.drop_threshold_pct = drop_threshold_pct

        # Warmup: bars needed before any indicator is valid
        # RSI needs rsi_period+1 bars, BB needs bb_period bars
        self.warmup_bars = max(warmup_bars, rsi_period + 1, bb_period, sl_lookback + 1)

        # Indicator cache: keyed by id(candles) to handle multiple datasets
        # Stores precomputed arrays for the full candle list
        self._cache_key: Optional[int] = None
        self._rsi_vals: List[float] = []
        self._bb_lower: List[float] = []
        self._sw_lows: List[float] = []
        self._closes: List[float] = []

    def compute_signal(
        self,
        candles: List[Candle],
        index: int,
    ) -> Optional[Signal]:
        """
        Evaluate entry conditions for the bar at `index`.

        Uses only candles[0..index] — no lookahead.

        The signal is computed on bar CLOSE, meaning we enter at the NEXT
        bar's open in the backtester (no bar-close lookahead for fill).
        Wait — actually we enter at THIS bar's close (market order on close),
        which means we know the close price when we signal.

        This is a standard assumption: you see the bar complete and
        submit a market order that fills at or near the close price.
        Slippage in the backtester accounts for imperfect fills.
        """
        # Not enough data for indicators
        if index < self.warmup_bars:
            return None

        # --- Precompute indicators over full candle list (cached) ---
        # Keyed by the identity of the candle list object. When a new backtest
        # starts with a different candle slice, the cache is invalidated.
        cache_key = id(candles)
        if cache_key != self._cache_key:
            closes_full = [c.close for c in candles]
            lows_full = [c.low for c in candles]
            self._rsi_vals = rsi(closes_full, period=self.rsi_period)
            _, _, bb_lower_full = bollinger_bands(
                closes_full, period=self.bb_period, num_std=self.bb_std
            )
            self._bb_lower = bb_lower_full
            self._sw_lows = rolling_min_low(lows_full, lookback=self.sl_lookback)
            self._closes = closes_full
            self._cache_key = cache_key

        current_rsi = self._rsi_vals[index]
        current_bb_lower = self._bb_lower[index]
        current_close = self._closes[index]
        recent_swing_low = self._sw_lows[index]

        # Skip if any indicator is NaN (insufficient data)
        if any(math.isnan(v) for v in [current_rsi, current_bb_lower, recent_swing_low]):
            return None

        # --- Condition 1: RSI oversold ---
        if current_rsi >= self.rsi_oversold:
            return None

        # --- Condition 2: Price at or below lower Bollinger Band ---
        # Allow a tiny tolerance (0.01%) so we catch touches, not just breaches
        bb_tolerance = current_bb_lower * 0.0001
        if current_close > current_bb_lower + bb_tolerance:
            return None

        # --- Condition 3 (optional): Recent price drop ---
        if self.use_drop_filter and index >= self.drop_lookback:
            lookback_close = self._closes[index - self.drop_lookback]
            if lookback_close > 0:
                drop_pct = (lookback_close - current_close) / lookback_close
                if drop_pct < self.drop_threshold_pct:
                    return None
            # If lookback_close is 0 (bad data), skip the filter gracefully

        # --- Entry price: current bar close + slippage applied by backtester ---
        entry_price = current_close

        # --- Stop Loss: below recent swing low with buffer ---
        stop_loss = recent_swing_low * (1 - self.sl_buffer_pct)

        # Validate stop distance
        sl_distance = entry_price - stop_loss
        sl_pct = sl_distance / entry_price

        # If swing low stop is too tight, widen to minimum
        if sl_pct < self.sl_min_pct:
            stop_loss = entry_price * (1 - self.sl_min_pct)
            sl_distance = entry_price - stop_loss

        # If swing low stop is too wide (e.g., volatile crash), fall back to fixed %
        if sl_pct > self.sl_max_pct:
            stop_loss = entry_price * (1 - self.sl_max_pct)
            sl_distance = entry_price - stop_loss

        # Final sanity check
        if stop_loss <= 0 or stop_loss >= entry_price:
            return None

        # --- Take Profit: fixed R multiple ---
        take_profit = entry_price + sl_distance * self.r_multiple

        reason = (
            f"RSI={current_rsi:.1f} BB_lower={current_bb_lower:.1f} "
            f"close={entry_price:.1f} swing_low={recent_swing_low:.1f}"
        )

        return Signal(
            direction="long",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
        )

    def parameter_dict(self) -> dict:
        """Return current parameters as a dict for logging/comparison."""
        return {
            "rsi_period": self.rsi_period,
            "rsi_oversold": self.rsi_oversold,
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "sl_lookback": self.sl_lookback,
            "sl_buffer_pct": self.sl_buffer_pct,
            "sl_max_pct": self.sl_max_pct,
            "r_multiple": self.r_multiple,
            "use_drop_filter": self.use_drop_filter,
            "drop_threshold_pct": self.drop_threshold_pct,
        }
