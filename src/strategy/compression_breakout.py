import math
import logging
from typing import List, Optional

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

        compression_lookback: int = 12,
        compression_atr_threshold: float = 0.8,   # range must be <= N * ATR
        breakout_buffer_atr: float = 0.10,        # breakout must exceed range edge by N * ATR
        breakout_min_body_atr: float = 0.40,      # breakout candle body must be meaningful
        atr_expansion_lookback: int = 5,          # current ATR must be > ATR N bars ago

        atr_stop_mult: float = 1.5,
        r_multiple: float = 2.0,

        atr_min_pct: float = 0.003,
        min_trend_strength_pct: float = 0.003,

        warmup_bars: int = 210,
        name: str = "CompressionBreakout_v1",
    ):
        super().__init__(name=name)

        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_period = atr_period

        self.compression_lookback = compression_lookback
        self.compression_atr_threshold = compression_atr_threshold
        self.breakout_buffer_atr = breakout_buffer_atr
        self.breakout_min_body_atr = breakout_min_body_atr
        self.atr_expansion_lookback = atr_expansion_lookback

        self.atr_stop_mult = atr_stop_mult
        self.r_multiple = r_multiple

        self.atr_min_pct = atr_min_pct
        self.min_trend_strength_pct = min_trend_strength_pct

        self.warmup_bars = max(
            warmup_bars,
            ema_slow + 1,
            atr_period + 1,
            compression_lookback + 2,
            atr_expansion_lookback + 2,
        )

        self._cache_key: Optional[int] = None
        self._ema_fast_vals: List[float] = []
        self._ema_slow_vals: List[float] = []
        self._atr_vals: List[float] = []
        self._closes: List[float] = []
        self._opens: List[float] = []
        self._highs: List[float] = []
        self._lows: List[float] = []

    def parameter_dict(self) -> dict:
        return {
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "atr_period": self.atr_period,
            "compression_lookback": self.compression_lookback,
            "compression_atr_threshold": self.compression_atr_threshold,
            "breakout_buffer_atr": self.breakout_buffer_atr,
            "breakout_min_body_atr": self.breakout_min_body_atr,
            "atr_expansion_lookback": self.atr_expansion_lookback,
            "atr_stop_mult": self.atr_stop_mult,
            "r_multiple": self.r_multiple,
            "atr_min_pct": self.atr_min_pct,
            "min_trend_strength_pct": self.min_trend_strength_pct,
        }

    def _precompute(self, candles: List[Candle]) -> None:
        closes = [c.close for c in candles]
        opens = [c.open for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]

        self._closes = closes
        self._opens = opens
        self._highs = highs
        self._lows = lows

        self._ema_fast_vals = ema(closes, period=self.ema_fast)
        self._ema_slow_vals = ema(closes, period=self.ema_slow)
        self._atr_vals = atr(highs, lows, closes, period=self.atr_period)

    def compute_signal(
        self,
        candles: List[Candle],
        index: int,
    ) -> Optional[Signal]:
        if index < self.warmup_bars:
            return None

        cache_key = id(candles)
        if cache_key != self._cache_key:
            self._precompute(candles)
            self._cache_key = cache_key

        ema_f = self._ema_fast_vals[index]
        ema_s = self._ema_slow_vals[index]
        atr_v = self._atr_vals[index]
        close = self._closes[index]
        open_ = self._opens[index]
        high = self._highs[index]
        low = self._lows[index]

        if any(math.isnan(v) for v in [ema_f, ema_s, atr_v]):
            return None

        if close <= 0:
            return None

        atr_pct = atr_v / close
        if atr_pct < self.atr_min_pct:
            return None

        trend_strength = abs(ema_f - ema_s) / close
        # if trend_strength < self.min_trend_strength_pct:
        #     return None

        start = index - self.compression_lookback
        end = index  # exclude current breakout bar from compression window
        if start < 0:
            return None

        range_high = max(self._highs[start:end])
        range_low = min(self._lows[start:end])
        compression_range = range_high - range_low

        # Compression: prior range must be relatively tight
        if compression_range > (self.compression_atr_threshold * atr_v):
            return None

        body_size = abs(close - open_)
        # if body_size < (0.1 * atr_v):
        #     return None

        breakout_buffer = self.breakout_buffer_atr * atr_v

        atr_prev = self._atr_vals[index - self.atr_expansion_lookback]

        if atr_v < (0.9 * atr_prev):
            return None

        # Long breakout
        if (
            close > range_high + breakout_buffer
            and close > open_
            and ema_f > ema_s
        ):
            stop_loss = close - self.atr_stop_mult * atr_v
            take_profit = close + (close - stop_loss) * self.r_multiple
            if stop_loss <= 0 or stop_loss >= close:
                return None

            return Signal(
                direction="long",
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=(
                    f"LONG breakout above compression range | "
                    f"range_high={range_high:.1f} ATR%={atr_pct*100:.2f}%"
                ),
                metadata={
                    "range_high": round(range_high, 2),
                    "range_low": round(range_low, 2),
                    "compression_range": round(compression_range, 4),
                    "atr_at_entry": round(atr_v, 4),
                    "atr_pct_at_entry": round(atr_pct * 100, 4),
                    "ema_fast_at_entry": round(ema_f, 2),
                    "ema_slow_at_entry": round(ema_s, 2),
                },
            )

        # Short breakout
        if (
            close < range_low - breakout_buffer
            and close < open_
            and ema_f < ema_s
        ):
            stop_loss = close + self.atr_stop_mult * atr_v
            take_profit = close - (stop_loss - close) * self.r_multiple
            if take_profit <= 0:
                return None

            return Signal(
                direction="short",
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=(
                    f"SHORT breakout below compression range | "
                    f"range_low={range_low:.1f} ATR%={atr_pct*100:.2f}%"
                ),
                metadata={
                    "range_high": round(range_high, 2),
                    "range_low": round(range_low, 2),
                    "compression_range": round(compression_range, 4),
                    "atr_at_entry": round(atr_v, 4),
                    "atr_pct_at_entry": round(atr_pct * 100, 4),
                    "ema_fast_at_entry": round(ema_f, 2),
                    "ema_slow_at_entry": round(ema_s, 2),
                },
            )

        return None