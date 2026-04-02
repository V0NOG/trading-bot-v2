"""
Strategy Layer: Base interface that all strategies must implement.

Design goals:
- Strategies are PURE FUNCTIONS over a window of candles — no side effects.
- The strategy sees only candles[0..i] when computing signal for candle i.
  This is the primary lookahead-bias prevention mechanism.
- Each strategy returns a Signal dataclass or None (no trade).

Adding a new strategy:
    class MyStrategy(BaseStrategy):
        def compute_signal(self, candles, index):
            ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from src.data.loader import Candle


@dataclass
class Signal:
    """
    Output of a strategy for a given bar.

    All prices are in quote currency (USDT for BTCUSDT).
    stop_loss and take_profit are absolute price levels, not distances.
    """
    direction: str          # "long" or "short" (v1: long only)
    entry_price: float      # Expected fill price (close of signal bar)
    stop_loss: float        # Absolute price level for stop
    take_profit: float      # Absolute price level for TP
    reason: str = ""        # Human-readable label for debugging
    metadata: dict = field(default_factory=dict)  # Machine-readable entry context

    def __post_init__(self):
        if self.direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got '{self.direction}'")
        if self.direction == "long":
            if self.stop_loss >= self.entry_price:
                raise ValueError(
                    f"Long stop_loss ({self.stop_loss}) must be below entry ({self.entry_price})"
                )
            if self.take_profit <= self.entry_price:
                raise ValueError(
                    f"Long take_profit ({self.take_profit}) must be above entry ({self.entry_price})"
                )
        else:
            if self.stop_loss <= self.entry_price:
                raise ValueError(
                    f"Short stop_loss ({self.stop_loss}) must be above entry ({self.entry_price})"
                )
            if self.take_profit >= self.entry_price:
                raise ValueError(
                    f"Short take_profit ({self.take_profit}) must be below entry ({self.entry_price})"
                )

    @property
    def risk_distance(self) -> float:
        """Absolute distance from entry to stop loss (always positive)."""
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_distance(self) -> float:
        """Absolute distance from entry to take profit (always positive)."""
        return abs(self.take_profit - self.entry_price)

    @property
    def r_multiple_target(self) -> float:
        """Target R multiple (reward / risk). e.g., 2.0 = 2R trade."""
        if self.risk_distance == 0:
            return 0.0
        return self.reward_distance / self.risk_distance


class BaseStrategy(ABC):
    """
    Abstract base for all trading strategies.

    Subclasses implement compute_signal() only.
    The backtester calls this for each bar, passing the full candle list
    and the current index — the strategy must NOT read beyond index.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute_signal(
        self,
        candles: List[Candle],
        index: int,
    ) -> Optional[Signal]:
        """
        Given candles[0..index], return a Signal or None.

        CRITICAL: Must only use candles[0] through candles[index].
        Never access candles[index+1] or later — that is lookahead bias.

        Args:
            candles: Full list of candles (only look back from index).
            index: Current bar index (the bar we're deciding for).

        Returns:
            Signal if entry conditions are met, else None.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
