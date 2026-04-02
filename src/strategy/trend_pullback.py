"""
Momentum Continuation Strategy (15m BTCUSDT)

Concept:
    A strong impulse candle shows that real directional conviction exists.
    Price then retraces briefly (1–3 candles) as short-term traders take
    profit. When price resumes in the impulse direction, the original move
    is likely to continue. We enter on that resumption.

Setup sequence (Long example):
    1. Impulse bar: bullish candle (close > open), range > 1.5×ATR, EMA50 > EMA200
    2. Pullback phase: 1–3 candles where each close < previous close (price drifts down)
    3. Resume bar: close > previous candle's high → entry signal

    Short is the mirror image.

State machine (per-instance, resets on new candle list):

    IDLE
      │  impulse detected (range > 1.5×ATR, trend aligned, candle body direction)
      ▼
    WAITING
      │  each bearish candle (for long) → pullback_count += 1
      │  pullback_count > max_pullback_bars → back to IDLE (setup expired)
      │  bars since impulse > max_setup_bars → back to IDLE (timeout)
      │  pullback_count >= 1 AND close > prev_high (long) → SIGNAL, back to IDLE
      ▼
    IDLE  (after signal or expiry)

Filters applied every bar before the state machine:
    - ATR(14) / close > atr_min_pct (single-bar volatility)
    - Rolling 10-bar avg range / close > atr_min_pct (flat-market gate)
    Both must pass or compute_signal returns None immediately.

Stop / Target:
    Stop loss : entry ± 1.2 × ATR
    Take profit: entry ± stop_distance × r_multiple (default 2.5R)

Important implementation note:
    Indicators are precomputed once per candle list (cached on id(candles)).
    State machine variables (state, pullback_count, impulse_bar_index) are
    also reset when a new candle list is detected, so each backtest fold
    starts from a clean slate without the strategy holding ghost state from
    a prior run.
"""

import math
import logging
from typing import List, Optional

from src.data.loader import Candle
from src.strategy.base import BaseStrategy, Signal
from src.strategy.indicators import ema, atr

logger = logging.getLogger(__name__)

# State labels — plain strings to keep debugging readable in logs
_IDLE    = "idle"
_WAITING = "waiting"


class TrendPullbackStrategy(BaseStrategy):

    def __init__(
        self,
        # Trend filter
        ema_fast: int = 50,
        ema_slow: int = 200,
        # Impulse detection
        atr_period: int = 14,
        impulse_atr_mult: float = 1.6,    # Candle range must exceed this × ATR
        # Pullback constraints
        min_pullback_bars: int = 1,        # Must see at least this many pullback candles
        max_pullback_bars: int = 4,        # Abandon setup if more than this many pullback candles
        max_setup_bars: int = 8,           # Hard timeout: bars since impulse (index-based)
        # Impulse quality guard: reject exhaustion candles that are too large
        max_impulse_atr_mult: float = 2.8,
        # Volatility filters
        atr_min_pct: float = 0.003,        # ATR/close must exceed 0.3%
        flat_lookback: int = 10,           # Rolling window for flat-market filter
        # Regime filter: EMA50 slope must be steep enough to confirm trend
        ema_slope_lookback: int = 10,      # Bars to look back for EMA slope measurement
        ema_slope_min_pct: float = 0.0015,  # EMA50 must move 0.1% in slope_lookback bars
        # Stop / target
        atr_stop_mult: float = 1.8,        # Stop = entry ± atr_stop_mult × ATR
        r_multiple: float = 2.0,
        # Warmup: EMA200 needs the most bars
        warmup_bars: int = 210,
        name: str = "MomentumContinuation_v2",

        atr_contraction_lookback: int = 10,
        atr_contraction_threshold: float = -0.05,

        max_pullback_atr_pct: float = 1.2,

        min_ema_spread_delta_pct: float = 0.0005,

        regime_filter_enabled: bool = True,
        min_trend_strength_pct: float = 0.003,
    ):
        super().__init__(name=name)
        self.ema_fast         = ema_fast
        self.ema_slow         = ema_slow
        self.atr_period       = atr_period
        self.impulse_atr_mult = impulse_atr_mult
        self.min_pullback_bars = min_pullback_bars
        self.max_pullback_bars = max_pullback_bars
        self.max_setup_bars   = max_setup_bars
        self.max_impulse_atr_mult = max_impulse_atr_mult
        self.atr_min_pct          = atr_min_pct
        self.flat_lookback        = flat_lookback
        self.ema_slope_lookback   = ema_slope_lookback
        self.ema_slope_min_pct    = ema_slope_min_pct
        self.atr_stop_mult        = atr_stop_mult
        self.r_multiple           = r_multiple
        self.warmup_bars          = max(warmup_bars, ema_slow + 1, atr_period + 1, flat_lookback, ema_slope_lookback + 1)
        self.min_ema_spread_delta_pct = min_ema_spread_delta_pct
        self.atr_contraction_lookback = atr_contraction_lookback
        self.atr_contraction_threshold = atr_contraction_threshold
        self.max_pullback_atr_pct = max_pullback_atr_pct

        # --- Indicator cache (reset when candle list changes) ---
        self._cache_key: Optional[int] = None
        self._ema_fast_vals: List[float] = []
        self._ema_slow_vals: List[float] = []
        self._atr_vals: List[float] = []
        self._avg_range_10: List[float] = []  # Rolling avg (high-low) for flat filter
        self._closes: List[float] = []
        self._opens:  List[float] = []
        self._highs:  List[float] = []
        self._lows:   List[float] = []

        # --- State machine (reset when candle list changes) ---
        self._state: str = _IDLE
        self._impulse_direction: Optional[str] = None
        self._impulse_bar_index: int = -1
        self._pullback_count: int = 0

        self.regime_filter_enabled = regime_filter_enabled
        self.min_trend_strength_pct = min_trend_strength_pct

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_signal(
        self,
        candles: List[Candle],
        index: int,
    ) -> Optional[Signal]:
        """
        Evaluate entry conditions for the bar at `index`.

        Uses only candles[0..index] — no lookahead.
        Signal fires on bar close; backtester fills on next bar open + slippage.
        """
        if index < self.warmup_bars:
            return None

        # --- Rebuild indicators and reset state on new candle list ---
        cache_key = id(candles)
        if cache_key != self._cache_key:
            self._precompute(candles)
            self._reset_state()
            self._cache_key = cache_key

        # --- Read current bar values ---
        ema_f  = self._ema_fast_vals[index]
        ema_s  = self._ema_slow_vals[index]
        atr_v  = self._atr_vals[index]
        avg_r  = self._avg_range_10[index]
        close  = self._closes[index]
        open_  = self._opens[index]
        high   = self._highs[index]
        low    = self._lows[index]

        if any(math.isnan(v) for v in [ema_f, ema_s, atr_v, avg_r]):
            return None

        # --- Filter 1: single-bar ATR volatility ---
        if atr_v / close < self.atr_min_pct:
            self._reset_state()   # Don't hold stale setup through low-vol periods
            return None

        # --- Filter 2: flat-market gate (rolling 10-bar avg range) ---
        if avg_r / close < self.atr_min_pct:
            self._reset_state()
            return None

        # --- State machine ---
        if self._state == _IDLE:
            return self._handle_idle(index, ema_f, ema_s, atr_v, close, open_)

        if self._state == _WAITING:
            return self._handle_waiting(index, atr_v, close)

        return None  # Unreachable but explicit
    
    
    def _get_regime(self, index: int) -> str:
        """
        Classify the current market regime using cached indicators.

        Returns:
            "bull"    -> strong enough bullish trend
            "bear"    -> strong enough bearish trend
            "neutral" -> no clear tradeable regime
        """
        if index < max(self.ema_slow, 20):
            return "neutral"

        fast = self._ema_fast_vals[index]
        slow = self._ema_slow_vals[index]
        atr_v = self._atr_vals[index]
        close = self._closes[index]

        if any(math.isnan(v) for v in [fast, slow, atr_v]) or close <= 0:
            return "neutral"

        trend_strength_pct = abs(fast - slow) / close
        atr_pct = atr_v / close

        if atr_pct < self.atr_min_pct:
            return "neutral"

        if self.regime_filter_enabled and trend_strength_pct < self.min_trend_strength_pct:
            return "neutral"

        if fast > slow:
            return "bull"
        if fast < slow:
            return "bear"

        return "neutral"

    def parameter_dict(self) -> dict:
        return {
            "ema_fast":                 self.ema_fast,
            "ema_slow":                 self.ema_slow,
            "impulse_atr_mult":         self.impulse_atr_mult,
            "min_pullback_bars":        self.min_pullback_bars,
            "max_pullback_bars":        self.max_pullback_bars,
            "max_setup_bars":           self.max_setup_bars,
            "atr_min_pct":              self.atr_min_pct,
            "ema_slope_lookback":       self.ema_slope_lookback,
            "ema_slope_min_pct":        self.ema_slope_min_pct,
            "atr_stop_mult":            self.atr_stop_mult,
            "r_multiple":               self.r_multiple,
            "regime_filter_enabled":    self.regime_filter_enabled,
            "min_trend_strength_pct":   self.min_trend_strength_pct,
            "max_impulse_atr_mult": self.max_impulse_atr_mult,
            "min_ema_spread_delta_pct": self.min_ema_spread_delta_pct,
            "atr_contraction_lookback": self.atr_contraction_lookback,
            "atr_contraction_threshold": self.atr_contraction_threshold,
            "max_pullback_atr_pct": self.max_pullback_atr_pct,
        }

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_idle(
        self,
        index: int,
        ema_f: float,
        ema_s: float,
        atr_v: float,
        close: float,
        open_: float,
    ) -> Optional[Signal]:
        """
        IDLE: scan current bar for a qualifying impulse.

        Impulse conditions:
          - Candle range (high - low) > impulse_atr_mult × ATR
          - Candle body direction matches trend (bullish body in uptrend,
            bearish body in downtrend)

        We never signal on the impulse bar itself — that would be chasing.
        The setup begins watching for a pullback starting on the next bar.
        """
        regime = self._get_regime(index)

        # --- NEW: strong trend regime filter ---
        ema_f = self._ema_fast_vals[index]
        ema_s = self._ema_slow_vals[index]
        close = self._closes[index]

        trend_strength = abs(ema_f - ema_s) / close

        # Require STRONG trend (not just trending)
        if trend_strength < 0.006:   # <-- key change (was ~0.003)
            return None

        candle_range = self._highs[index] - self._lows[index]
        impulse_range_atr = candle_range / atr_v if atr_v > 0 else 0.0

        # Must be a real impulse, but reject exhaustion spikes that are too extended.
        if impulse_range_atr <= self.impulse_atr_mult:
            return None

        if impulse_range_atr >= self.max_impulse_atr_mult:
            logger.debug(
                f"Bar {index}: Rejecting impulse as exhaustion "
                f"(range={impulse_range_atr:.2f}x ATR >= max {self.max_impulse_atr_mult:.2f}x)"
            )
            return None
        
        body_size = abs(close - open_)
        if body_size < 0.5 * candle_range:
            return None
        
        # Volatility expansion filter: require ATR to be expanding vs a few bars ago
        atr_ref_index = index - 5
        if atr_ref_index < 0:
            return None

        atr_prev = self._atr_vals[atr_ref_index]
        if math.isnan(atr_prev):
            return None

        if atr_v <= atr_prev:
            return None
        
        # Trend persistence filter: fast EMA must still be advancing
        ema_persist_index = index - 3
        if ema_persist_index < 0:
            return None

        ema_fast_prev = self._ema_fast_vals[ema_persist_index]
        if math.isnan(ema_fast_prev):
            return None

        # EMA50 slope: compare current EMA50 to its value N bars ago
        slope_ref_index = index - self.ema_slope_lookback

        # --- NEW: EMA spread expansion filter (trend strengthening) ---
        spread_now = abs(ema_f - ema_s)

        spread_ref_index = index - 5
        if spread_ref_index < 0:
            return None

        ema_f_prev = self._ema_fast_vals[spread_ref_index]
        ema_s_prev = self._ema_slow_vals[spread_ref_index]

        if math.isnan(ema_f_prev) or math.isnan(ema_s_prev):
            return None

        spread_prev = abs(ema_f_prev - ema_s_prev)

        if spread_prev == 0:
            return None

        spread_delta_pct = (spread_now - spread_prev) / spread_prev

        if spread_delta_pct < self.min_ema_spread_delta_pct:
            return None
        
        # --- NEW: ATR contraction filter (avoid compressing markets) ---
        atr_ref_index = index - self.atr_contraction_lookback
        if atr_ref_index < 0:
            return None

        atr_now = atr_v
        atr_prev = self._atr_vals[atr_ref_index]

        if math.isnan(atr_prev) or atr_prev == 0:
            return None

        atr_change_pct = (atr_now - atr_prev) / atr_prev

        # If ATR is contracting significantly → skip
        if atr_change_pct < self.atr_contraction_threshold:
            return None

        if slope_ref_index < 0:
            return None
        ema_f_ref = self._ema_fast_vals[slope_ref_index]
        if math.isnan(ema_f_ref) or ema_f_ref == 0:
            return None
        slope_pct = (ema_f - ema_f_ref) / ema_f_ref  # Positive = rising, negative = falling

        # Bullish impulse in uptrend with rising EMA50
        if close > open_ and ema_f > ema_s:
            if regime != "bull":
                return None
            if ema_f <= ema_fast_prev:
                return None
            if slope_pct < self.ema_slope_min_pct:
                return None
            self._state             = _WAITING
            self._impulse_direction = "long"
            self._impulse_bar_index = index
            self._pullback_count    = 0
            logger.debug(
                f"Bar {index}: LONG impulse detected "
                f"range={candle_range:.1f} > {self.impulse_atr_mult}×ATR={atr_v:.1f} "
                f"ema_slope={slope_pct*100:.3f}% regime={regime}"
            )
            return None

        # Bearish impulse in downtrend with falling EMA50
        if close < open_ and ema_f < ema_s:
            if regime != "bear":
                return None
            if ema_f >= ema_fast_prev:
                return None
            if slope_pct > -self.ema_slope_min_pct:
                return None
            self._state             = _WAITING
            self._impulse_direction = "short"
            self._impulse_bar_index = index
            self._pullback_count    = 0
            logger.debug(
                f"Bar {index}: SHORT impulse detected "
                f"range={candle_range:.1f} > {self.impulse_atr_mult}×ATR={atr_v:.1f} "
                f"ema_slope={slope_pct*100:.3f}% regime={regime}"
            )
            return None

        return None

    def _handle_waiting(
        self,
        index: int,
        atr_v: float,
        close: float,
    ) -> Optional[Signal]:
        """
        WAITING: track pullback bars and watch for resume.

        Timeout: if too many bars have elapsed since the impulse, the
        momentum window has closed — abandon the setup.

        Pullback (long setup): current close < previous close (price drifted down).
        Pullback (short setup): current close > previous close (price drifted up).

        Resume trigger (long): current close > previous bar's high.
        Resume trigger (short): current close < previous bar's low.

        Once pullback_count >= min_pullback_bars AND resume fires → Signal.
        """
        # Timeout check (bar-index based so cooldown gaps don't distort it)
        bars_elapsed = index - self._impulse_bar_index
        if bars_elapsed > self.max_setup_bars:
            logger.debug(f"Bar {index}: Setup timed out after {bars_elapsed} bars")
            self._reset_state()
            return None

        prev_close = self._closes[index - 1]
        prev_high  = self._highs[index - 1]
        prev_low   = self._lows[index - 1]

        direction  = self._impulse_direction
        stop_dist  = self.atr_stop_mult * atr_v

        # Snapshot EMA values at this bar for metadata (readable from cached arrays)
        ema_f_now = self._ema_fast_vals[index]
        ema_s_now = self._ema_slow_vals[index]

        if direction == "long":
            is_pullback = close < prev_close
            impulse_high = self._highs[self._impulse_bar_index]
            is_resume = close > prev_close and close > (prev_high * 0.995)

            # --- NEW: Pullback depth filter (structure protection) ---
            impulse_high = self._highs[self._impulse_bar_index]

            # Only evaluate true pullback bars AFTER the impulse bar
            if index > self._impulse_bar_index:
                pullback_low = min(self._lows[self._impulse_bar_index + 1:index + 1])
                pullback_depth = impulse_high - pullback_low

                # If pullback is too deep relative to ATR → structure likely broken
                if pullback_depth > (self.max_pullback_atr_pct * atr_v):
                    self._reset_state()
                    return None

            if is_resume and self._pullback_count >= self.min_pullback_bars:
                # --- NEW: breakout strength filter ---
                prev_high = self._highs[index - 1]
                breakout_strength = (close - prev_high) / atr_v

                # Require meaningful break above previous high
                if breakout_strength < 0.2:
                    return None
                
                if close < impulse_high:
                    return None
                body_size = abs(close - self._opens[index])
                if body_size < 0.3 * atr_v:
                    return None
                pullback_count_snapshot  = self._pullback_count
                bars_since_impulse       = index - self._impulse_bar_index
                self._reset_state()
                stop_loss   = close - stop_dist
                take_profit = close + stop_dist * self.r_multiple
                if stop_loss <= 0 or stop_loss >= close:
                    return None
                return Signal(
                    direction="long",
                    entry_price=close,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=(
                        f"LONG resume above prev_high={prev_high:.1f} | "
                        f"pullbacks={pullback_count_snapshot} "
                        f"ATR%={atr_v/close*100:.2f}%"
                    ),
                    metadata={
                        "impulse_direction":        "long",
                        "pullback_count_at_entry":  pullback_count_snapshot,
                        "bars_since_impulse":       bars_since_impulse,
                        "atr_at_entry":             round(atr_v, 4),
                        "atr_pct_at_entry":         round(atr_v / close * 100, 4),
                        "ema_fast_at_entry":        round(ema_f_now, 2),
                        "ema_slow_at_entry":        round(ema_s_now, 2),
                    },
                )

            if is_pullback:
                self._pullback_count += 1
                if self._pullback_count > self.max_pullback_bars:
                    logger.debug(f"Bar {index}: Too many pullback bars, abandoning long setup")
                    self._reset_state()

        elif direction == "short":
            is_pullback = close > prev_close
            impulse_low = self._lows[self._impulse_bar_index]
            is_resume = close < prev_close and close < (prev_low * 1.005)

            # --- NEW: Pullback depth filter ---
            impulse_low = self._lows[self._impulse_bar_index]

            # Only evaluate true pullback bars AFTER the impulse bar
            if index > self._impulse_bar_index:
                pullback_high = max(self._highs[self._impulse_bar_index + 1:index + 1])
                pullback_depth = pullback_high - impulse_low

                if pullback_depth > (self.max_pullback_atr_pct * atr_v):
                    self._reset_state()
                    return None

            if is_resume and self._pullback_count >= self.min_pullback_bars:
                # --- NEW: breakout strength filter ---
                prev_low = self._lows[index - 1]
                breakout_strength = (prev_low - close) / atr_v

                if breakout_strength < 0.2:
                    return None
                if close > impulse_low:
                    return None
                body_size = abs(close - self._opens[index])
                if body_size < 0.3 * atr_v:
                    return None
                pullback_count_snapshot  = self._pullback_count
                bars_since_impulse       = index - self._impulse_bar_index
                self._reset_state()
                stop_loss   = close + stop_dist
                take_profit = close - stop_dist * self.r_multiple
                if take_profit <= 0:
                    return None
                return Signal(
                    direction="short",
                    entry_price=close,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=(
                        f"SHORT resume below prev_low={prev_low:.1f} | "
                        f"pullbacks={pullback_count_snapshot} "
                        f"ATR%={atr_v/close*100:.2f}%"
                    ),
                    metadata={
                        "impulse_direction":        "short",
                        "pullback_count_at_entry":  pullback_count_snapshot,
                        "bars_since_impulse":       bars_since_impulse,
                        "atr_at_entry":             round(atr_v, 4),
                        "atr_pct_at_entry":         round(atr_v / close * 100, 4),
                        "ema_fast_at_entry":        round(ema_f_now, 2),
                        "ema_slow_at_entry":        round(ema_s_now, 2),
                    },
                )

            if is_pullback:
                self._pullback_count += 1
                if self._pullback_count > self.max_pullback_bars:
                    logger.debug(f"Bar {index}: Too many pullback bars, abandoning short setup")
                    self._reset_state()

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _precompute(self, candles: List[Candle]) -> None:
        """Compute all indicator arrays over the full candle list."""
        n      = len(candles)
        closes = [c.close for c in candles]
        opens  = [c.open  for c in candles]
        highs  = [c.high  for c in candles]
        lows   = [c.low   for c in candles]

        self._closes = closes
        self._opens  = opens
        self._highs  = highs
        self._lows   = lows

        self._ema_fast_vals = ema(closes, period=self.ema_fast)
        self._ema_slow_vals = ema(closes, period=self.ema_slow)
        self._atr_vals      = atr(highs, lows, closes, period=self.atr_period)

        # Rolling average of (high-low) over flat_lookback bars, including current bar.
        # Used to detect flat/dead markets where no candle in the window had any range.
        lb = self.flat_lookback
        avg_range = [float("nan")] * n
        for i in range(lb - 1, n):
            avg_range[i] = sum(
                highs[j] - lows[j] for j in range(i - lb + 1, i + 1)
            ) / lb
        self._avg_range_10 = avg_range

    def _reset_state(self) -> None:
        """Return state machine to idle. Called on setup expiry, signal, or new dataset."""
        self._state             = _IDLE
        self._impulse_direction = None
        self._impulse_bar_index = -1
        self._pullback_count    = 0
