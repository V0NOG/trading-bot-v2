"""
Technical indicator calculations.

All indicators are implemented as pure functions that operate on plain lists
of floats. They accept a 'period' argument and always return a list of the
same length as the input, with NaN (float('nan')) for bars where there is
insufficient lookback data.

IMPORTANT: All indicators operate strictly on data ending at index i.
The caller must pass only candles[0..i] to avoid lookahead bias.
"""

import math
from typing import List, Tuple


def sma(values: List[float], period: int) -> List[float]:
    """
    Simple Moving Average.

    Returns list of same length as input. First (period-1) values are NaN.
    """
    result = [float("nan")] * len(values)
    for i in range(period - 1, len(values)):
        result[i] = sum(values[i - period + 1 : i + 1]) / period
    return result


def ema(values: List[float], period: int) -> List[float]:
    """
    Exponential Moving Average (Wilder's smoothing).

    Uses standard EMA formula: EMA_t = price * k + EMA_{t-1} * (1-k)
    where k = 2 / (period + 1).

    Seeded from the first SMA of length 'period'.
    """
    result = [float("nan")] * len(values)
    k = 2.0 / (period + 1)

    # Seed: first EMA value is the SMA of the first 'period' bars
    if len(values) < period:
        return result

    seed = sum(values[:period]) / period
    result[period - 1] = seed

    for i in range(period, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)

    return result


def rsi(closes: List[float], period: int = 14) -> List[float]:
    """
    Relative Strength Index (Wilder's smoothing).

    Classic RSI formula using exponential smoothing for avg gain/loss.
    Returns values in [0, 100]. NaN for first 'period' bars.

    Oversold < 30, Overbought > 70.
    """
    result = [float("nan")] * len(closes)
    if len(closes) < period + 1:
        return result

    # Calculate gains and losses
    gains = [0.0] * len(closes)
    losses = [0.0] * len(closes)
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        if delta > 0:
            gains[i] = delta
        else:
            losses[i] = abs(delta)

    # Seed with simple average over first 'period' intervals
    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period

    # First RSI value at index 'period'
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100 - (100 / (1 + rs))

    # Wilder's smoothing for subsequent values
    for i in range(period + 1, len(closes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100 - (100 / (1 + rs))

    return result


def bollinger_bands(
    closes: List[float],
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Bollinger Bands: (upper, middle, lower).

    middle = SMA(period)
    upper = middle + num_std * std_dev
    lower = middle - num_std * std_dev

    Standard deviation uses population std (ddof=0), consistent with
    most charting platforms.

    Returns three lists of the same length as input (NaN for insufficient data).
    """
    n = len(closes)
    upper = [float("nan")] * n
    middle = [float("nan")] * n
    lower = [float("nan")] * n

    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        mean = sum(window) / period
        variance = sum((x - mean) ** 2 for x in window) / period
        std = math.sqrt(variance)

        middle[i] = mean
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std

    return upper, middle, lower


def atr(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14,
) -> List[float]:
    """
    Average True Range (Wilder's smoothing).

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = Wilder's EMA of TR with given period.

    Used for volatility-adjusted stop loss placement.
    """
    n = len(closes)
    result = [float("nan")] * n

    if n < period + 1:
        return result

    # Calculate true range for each bar (starting from bar 1)
    tr = [0.0] * n
    for i in range(1, n):
        high_low = highs[i] - lows[i]
        high_prev = abs(highs[i] - closes[i - 1])
        low_prev = abs(lows[i] - closes[i - 1])
        tr[i] = max(high_low, high_prev, low_prev)

    # Seed ATR with simple average
    seed = sum(tr[1 : period + 1]) / period
    result[period] = seed

    # Wilder's smoothing
    for i in range(period + 1, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return result


def swing_low(
    lows: List[float],
    lookback: int = 5,
) -> List[float]:
    """
    Swing low: the lowest low in the previous 'lookback' bars (not including current).

    Used for trailing stop loss placement below recent structure.
    Returns NaN for bars with insufficient lookback.
    """
    n = len(lows)
    result = [float("nan")] * n
    for i in range(lookback, n):
        result[i] = min(lows[i - lookback : i])  # Lookback only, not current bar
    return result


def rolling_min_low(lows: List[float], lookback: int) -> List[float]:
    """
    Lowest low over the last 'lookback' bars including current bar.
    Used for stop placement below recent low.
    """
    n = len(lows)
    result = [float("nan")] * n
    for i in range(lookback - 1, n):
        result[i] = min(lows[i - lookback + 1 : i + 1])
    return result
