"""
Risk Management Layer.

Responsibilities:
1. Position sizing: how many units to buy given account size and risk parameters
2. Trade gate: should we take this trade? (daily loss limit, cooldown, etc.)
3. All sizing is based on RISK AMOUNT, not notional — standard quant practice

Position Sizing Formula (fixed fractional / risk-based):
    risk_amount = account_equity * risk_pct_per_trade
    position_size = risk_amount / (entry_price - stop_loss)

    This means: if the trade hits stop loss, we lose exactly risk_pct_per_trade
    of current equity — no more, no less.

Daily Loss Limit:
    If cumulative daily loss exceeds max_daily_loss_pct of equity at day start,
    no new trades are opened that day. Resets at UTC midnight.

Cooldown:
    After any exit (win or loss), no new entry for 'cooldown_bars' bars.
    This prevents revenge trading and re-entry into choppy conditions.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from src.data.loader import Candle
from src.strategy.base import Signal

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Stateful risk manager — tracks daily PnL and cooldown state.

    Must be reset between backtests (call reset()).
    """

    def __init__(
        self,
        risk_pct_per_trade: float = 0.01,    # 1% of equity risked per trade
        max_risk_pct_per_trade: float = 0.02, # Hard cap: never risk more than 2%
        max_daily_loss_pct: float = 0.05,     # Stop trading for the day after 5% loss
        cooldown_bars: int = 3,               # Bars to wait after trade exit
        use_daily_loss_limit: bool = True,
        use_cooldown: bool = True,
    ):
        if risk_pct_per_trade <= 0 or risk_pct_per_trade > 1:
            raise ValueError("risk_pct_per_trade must be in (0, 1]")
        if max_risk_pct_per_trade < risk_pct_per_trade:
            raise ValueError("max_risk_pct_per_trade must be >= risk_pct_per_trade")

        self.risk_pct_per_trade = risk_pct_per_trade
        self.max_risk_pct_per_trade = max_risk_pct_per_trade
        self.max_daily_loss_pct = max_daily_loss_pct
        self.cooldown_bars = cooldown_bars
        self.use_daily_loss_limit = use_daily_loss_limit
        self.use_cooldown = use_cooldown

        self.reset()

    def reset(self):
        """Reset all stateful tracking. Call before each backtest run."""
        self._cooldown_bars_remaining = 0
        self._daily_equity_start: Optional[float] = None
        self._daily_loss: float = 0.0
        self._current_day: Optional[datetime] = None
        self._trading_halted_today: bool = False

    def can_trade(self, candle: Candle, equity: float) -> tuple[bool, str]:
        """
        Check whether we're allowed to open a new position on this bar.

        Args:
            candle: Current bar (used for timestamp).
            equity: Current account equity.

        Returns:
            (allowed: bool, reason: str)
        """
        # Cooldown check
        if self.use_cooldown and self._cooldown_bars_remaining > 0:
            return False, f"Cooldown ({self._cooldown_bars_remaining} bars remaining)"

        # Daily loss limit check
        if self.use_daily_loss_limit:
            self._update_day(candle, equity)
            if self._trading_halted_today:
                return False, "Daily loss limit reached"

        return True, ""

    def size_position(
        self,
        signal: Signal,
        equity: float,
    ) -> float:
        """
        Calculate position size in base currency units (e.g., BTC).

        Uses fixed fractional risk: risk_amount / risk_per_unit

        The risk_pct is capped at max_risk_pct_per_trade as a hard guardrail.

        Args:
            signal: Entry signal with entry_price and stop_loss.
            equity: Current account equity in quote currency.

        Returns:
            Position size in base currency (BTC). 0 if trade should be skipped.
        """
        risk_pct = min(self.risk_pct_per_trade, self.max_risk_pct_per_trade)
        risk_amount = equity * risk_pct  # USDT at risk

        # Risk per unit is always positive: distance from entry to stop
        risk_per_unit = abs(signal.entry_price - signal.stop_loss)
        if risk_per_unit <= 0:
            logger.warning("Invalid signal: risk_per_unit <= 0, skipping")
            return 0.0

        position_size = risk_amount / risk_per_unit

        # Sanity check: notional value shouldn't exceed full equity
        # (would mean leverage; v1 is no-leverage)
        notional = position_size * signal.entry_price
        if notional > equity:
            # Cap at 100% of equity (no leverage)
            position_size = equity / signal.entry_price
            logger.debug(
                f"Position capped at 100% equity: size={position_size:.6f} BTC "
                f"notional={notional:.2f} > equity={equity:.2f}"
            )

        return position_size

    def notify_trade_closed(self, pnl: float, equity: float, candle: Candle):
        """
        Called by the backtester when a trade is closed.

        Updates cooldown counter and daily loss tracking.
        """
        # Start cooldown
        if self.use_cooldown:
            self._cooldown_bars_remaining = self.cooldown_bars

        # Update daily loss
        if self.use_daily_loss_limit and pnl < 0:
            self._daily_loss += abs(pnl)
            loss_pct = self._daily_loss / (self._daily_equity_start or equity)
            if loss_pct >= self.max_daily_loss_pct:
                self._trading_halted_today = True
                logger.info(
                    f"Daily loss limit hit: {loss_pct:.1%} >= {self.max_daily_loss_pct:.1%}. "
                    f"No more trades today."
                )

    def tick(self):
        """
        Called once per bar to decrement cooldown counter.
        Must be called AFTER can_trade() for the current bar.
        """
        if self._cooldown_bars_remaining > 0:
            self._cooldown_bars_remaining -= 1

    def _update_day(self, candle: Candle, equity: float):
        """Track day boundary and reset daily state at UTC midnight."""
        candle_day = candle.timestamp.astimezone(timezone.utc).date()

        if self._current_day is None or candle_day != self._current_day:
            # New day: reset daily tracking
            self._current_day = candle_day
            self._daily_equity_start = equity
            self._daily_loss = 0.0
            self._trading_halted_today = False
