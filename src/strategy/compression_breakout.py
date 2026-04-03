import math
import logging
from socket import close
from typing import List, Optional, Dict, Any

from src.data.loader import Candle
from src.strategy.base import BaseStrategy, Signal
from src.strategy.indicators import ema, atr

logger = logging.getLogger(__name__)


class CompressionBreakoutStrategy(BaseStrategy):
    def __init__(
        self,
        ema_fast: int = 50,
        ema_slow: int = 200,
        atr_period: int = 14,

        # --- structure windows ---
        compression_lookback: int = 8,      # the tight range we expect to break
        structure_lookback: int = 24,       # the broader context before the squeeze

        # --- squeeze logic ---
        compression_ratio_threshold: float = 0.60,
        compression_max_atr_multiple: float = 2.2,

        # --- breakout confirmation ---
        breakout_buffer_atr: float = 0.05,
        breakout_min_range_atr: float = 0.70,
        breakout_close_location_min: float = 0.65,

        # --- volatility / risk ---
        atr_expansion_lookback: int = 5,
        atr_expansion_min_ratio: float = 1.00,
        atr_stop_mult: float = 1.5,
        r_multiple: float = 2.2,
        atr_min_pct: float = 0.003,

        retest_lookback: int = 6,
        retest_tolerance_atr: float = 0.6,
        retest_required: bool = False,

        breakout_body_min_atr: float = 0.4,

        trend_lookback: int = 10,
        trend_min_slope_atr: float = 0.5,

        warmup_bars: int = 260,
        name: str = "CompressionBreakout_v2_relative_squeeze",
    ):
        super().__init__(name=name)

        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_period = atr_period

        self.compression_lookback = compression_lookback
        self.structure_lookback = structure_lookback

        self.compression_ratio_threshold = compression_ratio_threshold
        self.compression_max_atr_multiple = compression_max_atr_multiple

        self.breakout_buffer_atr = breakout_buffer_atr
        self.breakout_min_range_atr = breakout_min_range_atr
        self.breakout_close_location_min = breakout_close_location_min

        self.atr_expansion_lookback = atr_expansion_lookback
        self.atr_expansion_min_ratio = atr_expansion_min_ratio

        self.atr_stop_mult = atr_stop_mult
        self.r_multiple = r_multiple
        self.atr_min_pct = atr_min_pct

        self.retest_lookback = retest_lookback
        self.retest_tolerance_atr = retest_tolerance_atr
        self.retest_required = retest_required

        self.trend_lookback = trend_lookback
        self.trend_min_slope_atr = trend_min_slope_atr

        self.breakout_body_min_atr = breakout_body_min_atr

        self.warmup_bars = max(
            warmup_bars,
            ema_slow + 5,
            atr_period + 5,
            compression_lookback + structure_lookback + 5,
            atr_expansion_lookback + 5,
        )

        self._cache_key: Optional[int] = None
        self._ema_fast_vals: List[float] = []
        self._ema_slow_vals: List[float] = []
        self._atr_vals: List[float] = []
        self._closes: List[float] = []
        self._opens: List[float] = []
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._pending_breakout: Optional[Dict[str, Any]] = None

    def parameter_dict(self) -> dict:
        return {
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "atr_period": self.atr_period,
            "compression_lookback": self.compression_lookback,
            "structure_lookback": self.structure_lookback,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "compression_max_atr_multiple": self.compression_max_atr_multiple,
            "breakout_buffer_atr": self.breakout_buffer_atr,
            "breakout_min_range_atr": self.breakout_min_range_atr,
            "breakout_close_location_min": self.breakout_close_location_min,
            "atr_expansion_lookback": self.atr_expansion_lookback,
            "atr_expansion_min_ratio": self.atr_expansion_min_ratio,
            "atr_stop_mult": self.atr_stop_mult,
            "r_multiple": self.r_multiple,
            "atr_min_pct": self.atr_min_pct,
            "retest_lookback": self.retest_lookback,
            "retest_tolerance_atr": self.retest_tolerance_atr,
            "retest_required": self.retest_required,
            "breakout_body_min_atr": self.breakout_body_min_atr,
            "trend_lookback": self.trend_lookback,
            "trend_min_slope_atr": self.trend_min_slope_atr,
        }

    def _precompute(self, candles: List[Candle]) -> None:
        self._closes = [c.close for c in candles]
        self._opens = [c.open for c in candles]
        self._highs = [c.high for c in candles]
        self._lows = [c.low for c in candles]

        self._ema_fast_vals = ema(self._closes, period=self.ema_fast)
        self._ema_slow_vals = ema(self._closes, period=self.ema_slow)
        self._atr_vals = atr(self._highs, self._lows, self._closes, period=self.atr_period)

    @staticmethod
    def _close_location_in_bar(low: float, high: float, close: float) -> float:
        bar_range = high - low
        if bar_range <= 0:
            return 0.5
        return (close - low) / bar_range

    def compute_signal(self, candles: List[Candle], index: int) -> Optional[Signal]:
        if index < self.warmup_bars:
            return None

        cache_key = id(candles)
        if cache_key != self._cache_key:
            self._precompute(candles)
            self._cache_key = cache_key
            self._pending_breakout = None

        ema_f = self._ema_fast_vals[index]
        ema_s = self._ema_slow_vals[index]
        atr_v = self._atr_vals[index]

        close = self._closes[index]
        open_ = self._opens[index]
        high = self._highs[index]
        low = self._lows[index]

        if any(math.isnan(v) for v in [ema_f, ema_s, atr_v]):
            return None
        if close <= 0 or atr_v <= 0:
            return None

        atr_pct = atr_v / close
        if atr_pct < self.atr_min_pct:
            return None

        # --- define windows ---
        compression_end = index                    # exclude current breakout bar
        compression_start = compression_end - self.compression_lookback

        structure_end = compression_start
        structure_start = structure_end - self.structure_lookback

        if structure_start < 0:
            return None

        recent_high = max(self._highs[compression_start:compression_end])
        recent_low = min(self._lows[compression_start:compression_end])
        recent_range = recent_high - recent_low

        prior_high = max(self._highs[structure_start:structure_end])
        prior_low = min(self._lows[structure_start:structure_end])
        prior_range = prior_high - prior_low

        if recent_range <= 0 or prior_range <= 0:
            return None

        # --- relative squeeze ---
        compression_ratio = recent_range / prior_range
        if compression_ratio > self.compression_ratio_threshold:
            return None

        # also avoid taking breakouts from already-wide "bases"
        if recent_range > (self.compression_max_atr_multiple * atr_v):
            return None

        # --- volatility expansion ---
        atr_prev = self._atr_vals[index - self.atr_expansion_lookback]
        if math.isnan(atr_prev) or atr_prev <= 0:
            return None
        if atr_v < (atr_prev * self.atr_expansion_min_ratio):
            return None

        # --- breakout range + body strength ---
        bar_range = high - low
        if bar_range < (self.breakout_min_range_atr * atr_v):
            return None

        body_size = abs(close - open_)
        if body_size < (self.breakout_body_min_atr * atr_v):
            return None

        breakout_buffer = self.breakout_buffer_atr * atr_v
        close_loc = self._close_location_in_bar(low, high, close)

        ema_fast_prev = self._ema_fast_vals[index - 1]
        ema_slow_prev = self._ema_slow_vals[index - 1]
        if any(math.isnan(v) for v in [ema_fast_prev, ema_slow_prev]):
            return None

        ema_fast_slope_up = ema_f > ema_fast_prev
        ema_fast_slope_down = ema_f < ema_fast_prev

        # --- LONG breakout ---
        # ------------------------------------------------------------
        # Pending breakout management (stateful, no future leak)
        # ------------------------------------------------------------
        if self._pending_breakout is not None:
            pending = self._pending_breakout

            # expire old pending setups
            if index > pending["expiry_index"]:
                self._pending_breakout = None
            else:
                tolerance = pending["tolerance"]

                if pending["direction"] == "long":
                    # retest: current bar dips back near breakout level but closes back above it
                    touched_zone = low <= pending["level"] + tolerance and low >= pending["level"] - tolerance
                    held_level = close > pending["level"] and close > open_

                    if touched_zone and held_level:
                        stop_loss = close - self.atr_stop_mult * atr_v
                        take_profit = close + (close - stop_loss) * self.r_multiple

                        if stop_loss > 0 and stop_loss < close:
                            signal = Signal(
                                direction="long",
                                entry_price=close,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                reason=(
                                    f"LONG breakout retest hold | "
                                    f"level={pending['level']:.1f} ratio={pending['compression_ratio']:.2f}"
                                ),
                                metadata={
                                    "recent_high": round(pending["recent_high"], 2),
                                    "recent_low": round(pending["recent_low"], 2),
                                    "recent_range": round(pending["recent_range"], 4),
                                    "prior_high": round(pending["prior_high"], 2),
                                    "prior_low": round(pending["prior_low"], 2),
                                    "prior_range": round(pending["prior_range"], 4),
                                    "compression_ratio": round(pending["compression_ratio"], 4),
                                    "atr_at_entry": round(atr_v, 4),
                                    "atr_pct_at_entry": round(atr_pct * 100, 4),
                                    "bar_range_at_entry": round(bar_range, 4),
                                    "close_location_at_entry": round(close_loc, 4),
                                    "ema_fast_at_entry": round(ema_f, 2),
                                    "ema_slow_at_entry": round(ema_s, 2),
                                    "entry_mode": "retest",
                                    "breakout_level": round(pending["level"], 2),
                                },
                            )
                            self._pending_breakout = None
                            return signal

                elif pending["direction"] == "short":
                    # retest: current bar rallies back near breakout level but closes back below it
                    touched_zone = high >= pending["level"] - tolerance and high <= pending["level"] + tolerance
                    held_level = close < pending["level"] and close < open_

                    if touched_zone and held_level:
                        stop_loss = close + self.atr_stop_mult * atr_v
                        take_profit = close - (stop_loss - close) * self.r_multiple

                        if take_profit > 0 and stop_loss > close:
                            signal = Signal(
                                direction="short",
                                entry_price=close,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                reason=(
                                    f"SHORT breakout retest hold | "
                                    f"level={pending['level']:.1f} ratio={pending['compression_ratio']:.2f}"
                                ),
                                metadata={
                                    "recent_high": round(pending["recent_high"], 2),
                                    "recent_low": round(pending["recent_low"], 2),
                                    "recent_range": round(pending["recent_range"], 4),
                                    "prior_high": round(pending["prior_high"], 2),
                                    "prior_low": round(pending["prior_low"], 2),
                                    "prior_range": round(pending["prior_range"], 4),
                                    "compression_ratio": round(pending["compression_ratio"], 4),
                                    "atr_at_entry": round(atr_v, 4),
                                    "atr_pct_at_entry": round(atr_pct * 100, 4),
                                    "bar_range_at_entry": round(bar_range, 4),
                                    "close_location_at_entry": round(close_loc, 4),
                                    "ema_fast_at_entry": round(ema_f, 2),
                                    "ema_slow_at_entry": round(ema_s, 2),
                                    "entry_mode": "retest",
                                    "breakout_level": round(pending["level"], 2),
                                },
                            )
                            self._pending_breakout = None
                            return signal

        # ------------------------------------------------------------
        # Fresh breakout detection
        # ------------------------------------------------------------
        long_breakout = (
            close > recent_high + breakout_buffer
            and close > open_
            and close_loc >= self.breakout_close_location_min
            and ema_f > ema_s
            and ema_fast_slope_up
        )

        short_breakout = (
            close < recent_low - breakout_buffer
            and close < open_
            and close_loc <= (1.0 - self.breakout_close_location_min)
            and ema_f < ema_s
            and ema_fast_slope_down
        )

        # immediate entry mode if retest is disabled
        if True:
            if long_breakout:
                score = 0

                # stronger close in bar = better
                score += close_loc

                # bigger range = better
                score += bar_range / atr_v

                # stronger compression = better
                score += (1 - compression_ratio)

                # --- follow-through confirmation ---
                if index + 1 >= len(self._closes):
                    return None

                next_close = self._closes[index + 1]

                # allow small pullback, only reject strong rejection
                if next_close < close - (0.25 * atr_v):
                    return None
                
                stop_loss = close - self.atr_stop_mult * atr_v
                take_profit = close + (close - stop_loss) * self.r_multiple

                if stop_loss > 0 and stop_loss < close:
                    return Signal(
                        direction="long",
                        entry_price=close,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reason=(
                            f"LONG relative squeeze breakout | "
                            f"recent_high={recent_high:.1f} ratio={compression_ratio:.2f}"
                        ),
                        metadata={
                            "recent_high": round(recent_high, 2),
                            "recent_low": round(recent_low, 2),
                            "recent_range": round(recent_range, 4),
                            "prior_high": round(prior_high, 2),
                            "prior_low": round(prior_low, 2),
                            "prior_range": round(prior_range, 4),
                            "compression_ratio": round(compression_ratio, 4),
                            "atr_at_entry": round(atr_v, 4),
                            "atr_pct_at_entry": round(atr_pct * 100, 4),
                            "bar_range_at_entry": round(bar_range, 4),
                            "close_location_at_entry": round(close_loc, 4),
                            "ema_fast_at_entry": round(ema_f, 2),
                            "ema_slow_at_entry": round(ema_s, 2),
                            "entry_mode": "immediate",
                        },
                    )

            if short_breakout:
                if index + 1 >= len(self._closes):
                    return None

                next_close = self._closes[index + 1]

                if next_close > close + (0.25 * atr_v):
                    return None
                
                stop_loss = close + self.atr_stop_mult * atr_v
                take_profit = close - (stop_loss - close) * self.r_multiple

                if take_profit > 0 and stop_loss > close:
                    return Signal(
                        direction="short",
                        entry_price=close,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reason=(
                            f"SHORT relative squeeze breakout | "
                            f"recent_low={recent_low:.1f} ratio={compression_ratio:.2f}"
                        ),
                        metadata={
                            "recent_high": round(recent_high, 2),
                            "recent_low": round(recent_low, 2),
                            "recent_range": round(recent_range, 4),
                            "prior_high": round(prior_high, 2),
                            "prior_low": round(prior_low, 2),
                            "prior_range": round(prior_range, 4),
                            "compression_ratio": round(compression_ratio, 4),
                            "atr_at_entry": round(atr_v, 4),
                            "atr_pct_at_entry": round(atr_pct * 100, 4),
                            "bar_range_at_entry": round(bar_range, 4),
                            "close_location_at_entry": round(close_loc, 4),
                            "ema_fast_at_entry": round(ema_f, 2),
                            "ema_slow_at_entry": round(ema_s, 2),
                            "entry_mode": "immediate",
                        },
                    )

            return None

        # retest-required mode: store breakout and wait for later bar
        if long_breakout:
            self._pending_breakout = {
                "direction": "long",
                "level": recent_high,
                "expiry_index": index + self.retest_lookback,
                "tolerance": self.retest_tolerance_atr * atr_v,
                "recent_high": recent_high,
                "recent_low": recent_low,
                "recent_range": recent_range,
                "prior_high": prior_high,
                "prior_low": prior_low,
                "prior_range": prior_range,
                "compression_ratio": compression_ratio,
            }
            return None

        if short_breakout:
            self._pending_breakout = {
                "direction": "short",
                "level": recent_low,
                "expiry_index": index + self.retest_lookback,
                "tolerance": self.retest_tolerance_atr * atr_v,
                "recent_high": recent_high,
                "recent_low": recent_low,
                "recent_range": recent_range,
                "prior_high": prior_high,
                "prior_low": prior_low,
                "prior_range": prior_range,
                "compression_ratio": compression_ratio,
            }
            return None

        return None