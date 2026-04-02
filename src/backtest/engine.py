"""
Backtesting Engine: Simulates trading candle-by-candle with realistic fills.

Core design principles:
- Single-pass forward simulation — no lookahead
- One position at a time
- Position state machine: FLAT → OPEN → FLAT
- Fees and slippage applied at fill time
- Stop loss and take profit checked intra-bar (on high/low, not just close)
- Detailed per-trade records for post-analysis

Fill Assumptions:
    Entry: filled at signal bar's CLOSE + slippage
        (Represents a market order submitted at bar close, filled at open+spread of next)
        We simplify to: fill = close * (1 + slippage_pct) for long entries

    Actually, for realism, we fill on the NEXT bar's OPEN with slippage.
    This is the most conservative and realistic approach — you can't fill
    at the candle that generated your signal.

    Stop Loss: triggered when bar LOW crosses stop price
        Fill = stop_loss * (1 - slippage_pct) — gapped through in slippage direction

    Take Profit: triggered when bar HIGH crosses TP price
        Fill = take_profit * (1 - slippage_pct) — slight slippage on TP too

    Both SL and TP can trigger on the same bar (gap opens). SL takes precedence
    (conservative assumption).

Fee Model:
    - Taker fee applied at entry and exit
    - Fee = notional * fee_rate
    - Total round-trip fee ≈ 2 * fee_rate * notional
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from src.data.loader import Candle
from src.strategy.base import BaseStrategy, Signal
from src.risk.manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """
    Complete record of a single completed trade.

    All prices in quote currency (USDT).
    MFE/MAE are in absolute price units, not PnL.
    """
    trade_id: int
    direction: str              # "long" or "short"

    entry_time: datetime
    entry_price: float          # Actual fill price (after slippage)
    entry_signal_price: float   # Signal price (before slippage) — for analysis

    exit_time: datetime
    exit_price: float           # Actual fill price (after slippage)
    exit_reason: str            # "stop_loss", "take_profit", "end_of_data"

    stop_loss: float
    take_profit: float
    position_size: float        # Base currency (BTC)

    entry_fee: float            # Fee paid on entry (USDT)
    exit_fee: float             # Fee paid on exit (USDT)

    gross_pnl: float            # PnL before fees
    net_pnl: float              # PnL after fees
    pnl_r: float                # PnL in R multiples (net_pnl / initial_risk)

    max_favorable_excursion: float   # Best unrealized profit during trade (USDT)
    max_adverse_excursion: float     # Worst unrealized loss during trade (USDT)

    equity_before: float        # Account equity before trade
    equity_after: float         # Account equity after trade

    signal_reason: str = ""     # Debug info from strategy
    entry_context: dict = field(default_factory=dict)  # Strategy metadata at entry

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    @property
    def duration_bars(self) -> int:
        """Number of bars the trade was open (approximate from timestamps)."""
        return 0  # Filled in by engine


@dataclass
class BacktestState:
    """Mutable state during backtesting — position tracking."""
    in_position: bool = False
    entry_bar_index: int = -1
    entry_time: Optional[datetime] = None
    entry_price: float = 0.0
    entry_signal_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size: float = 0.0
    entry_fee: float = 0.0
    current_signal: Optional[Signal] = None
    equity_before_trade: float = 0.0

    # MFE/MAE tracking
    best_price: float = 0.0     # Highest price seen since entry (long)
    worst_price: float = float("inf")  # Lowest price seen since entry (long)


@dataclass
class BacktestResult:
    """Output of a complete backtest run."""
    trades: List[TradeRecord]
    equity_curve: List[float]       # Equity at end of each bar
    candles: List[Candle]           # Reference to input data
    initial_equity: float
    final_equity: float
    config: dict = field(default_factory=dict)


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Processes candles one at a time. For each bar:
    1. Check if current position hits SL or TP (intra-bar check)
    2. If flat, ask strategy for signal
    3. If signal received, size position and enter on NEXT bar open
    4. Record equity

    The "enter on next bar" approach prevents lookahead bias at the fill level.
    We know the signal bar closed, but we don't know the next bar's open —
    the backtester uses next_bar.open + slippage as the fill price.
    """

    def __init__(
        self,
        initial_equity: float = 10_000.0,
        fee_rate: float = 0.001,          # 0.1% per side (taker fee)
        slippage_pct: float = 0.0005,     # 0.05% slippage per fill
    ):
        if initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if fee_rate < 0 or fee_rate > 0.05:
            raise ValueError("fee_rate out of reasonable range")
        if slippage_pct < 0:
            raise ValueError("slippage_pct must be non-negative")

        self.initial_equity = initial_equity
        self.fee_rate = fee_rate
        self.slippage_pct = slippage_pct

    def run(
        self,
        candles: List[Candle],
        strategy: BaseStrategy,
        risk_manager: RiskManager,
    ) -> BacktestResult:
        """
        Run a full backtest over the provided candles.

        Args:
            candles: Sorted list of OHLCV bars (ascending by time).
            strategy: Initialized strategy instance.
            risk_manager: Initialized risk manager (will be reset before run).

        Returns:
            BacktestResult with all trades and equity curve.
        """
        if len(candles) < 2:
            raise ValueError("Need at least 2 candles to backtest")

        risk_manager.reset()

        equity = self.initial_equity
        equity_curve = []
        trades: List[TradeRecord] = []
        state = BacktestState()
        trade_counter = 0

        # Pending signal: strategy fired on bar i, we enter on bar i+1
        pending_signal: Optional[Signal] = None
        pending_equity_before: float = 0.0

        for i, candle in enumerate(candles):

            # ----------------------------------------------------------------
            # 1. Handle pending entry (from signal on previous bar)
            # ----------------------------------------------------------------
            if pending_signal is not None and not state.in_position:
                signal = pending_signal
                pending_signal = None

                # Fill at current bar's OPEN ± slippage
                # Long: pay slightly more; Short: receive slightly less
                if signal.direction == "long":
                    fill_price = candle.open * (1 + self.slippage_pct)
                else:
                    fill_price = candle.open * (1 - self.slippage_pct)

                # Check that fill price hasn't blown through stop or TP on the gap open
                if signal.direction == "long" and fill_price <= signal.stop_loss:
                    logger.debug(
                        f"Bar {i}: Long entry cancelled — open {candle.open:.2f} gapped "
                        f"through stop {signal.stop_loss:.2f}"
                    )
                elif signal.direction == "short" and fill_price >= signal.stop_loss:
                    logger.debug(
                        f"Bar {i}: Short entry cancelled — open {candle.open:.2f} gapped "
                        f"through stop {signal.stop_loss:.2f}"
                    )
                elif signal.direction == "long" and fill_price >= signal.take_profit:
                    logger.debug(
                        f"Bar {i}: Long entry cancelled — open {candle.open:.2f} gapped "
                        f"through TP {signal.take_profit:.2f}"
                    )
                elif signal.direction == "short" and fill_price <= signal.take_profit:
                    logger.debug(
                        f"Bar {i}: Short entry cancelled — open {candle.open:.2f} gapped "
                        f"through TP {signal.take_profit:.2f}"
                    )
                else:
                    # Size the position based on actual fill price
                    # (risk_manager uses signal.entry_price for sizing, then we adjust)
                    pos_size = risk_manager.size_position(signal, pending_equity_before)

                    if pos_size > 0:
                        entry_fee = pos_size * fill_price * self.fee_rate
                        equity -= entry_fee  # Fee deducted at entry

                        state.in_position = True
                        state.entry_bar_index = i
                        state.entry_time = candle.timestamp
                        state.entry_price = fill_price
                        state.entry_signal_price = signal.entry_price
                        state.stop_loss = signal.stop_loss
                        state.take_profit = signal.take_profit
                        state.position_size = pos_size
                        state.entry_fee = entry_fee
                        state.current_signal = signal
                        state.equity_before_trade = pending_equity_before
                        state.best_price = fill_price
                        state.worst_price = fill_price

                        logger.debug(
                            f"Bar {i} ENTRY: {candle.timestamp} fill={fill_price:.2f} "
                            f"sl={signal.stop_loss:.2f} tp={signal.take_profit:.2f} "
                            f"size={pos_size:.6f} fee={entry_fee:.2f}"
                        )

            # ----------------------------------------------------------------
            # 2. Manage open position: check SL/TP, update MFE/MAE
            # ----------------------------------------------------------------
            if state.in_position:
                is_long = state.current_signal.direction == "long"

                # Update best/worst price seen (for MFE/MAE)
                state.best_price = max(state.best_price, candle.high)
                state.worst_price = min(state.worst_price, candle.low)

                exit_price = None
                exit_reason = None

                if is_long:
                    # Long: SL triggers on low crossing down, TP on high crossing up
                    if candle.low <= state.stop_loss:
                        exit_price = state.stop_loss * (1 - self.slippage_pct)
                        exit_reason = "stop_loss"
                    elif candle.high >= state.take_profit:
                        exit_price = state.take_profit * (1 - self.slippage_pct)
                        exit_reason = "take_profit"
                else:
                    # Short: SL triggers on high crossing up, TP on low crossing down
                    if candle.high >= state.stop_loss:
                        exit_price = state.stop_loss * (1 + self.slippage_pct)
                        exit_reason = "stop_loss"
                    elif candle.low <= state.take_profit:
                        exit_price = state.take_profit * (1 + self.slippage_pct)
                        exit_reason = "take_profit"

                # Last bar — close any open position at close price
                if exit_price is None and i == len(candles) - 1:
                    if is_long:
                        exit_price = candle.close * (1 - self.slippage_pct)
                    else:
                        exit_price = candle.close * (1 + self.slippage_pct)
                    exit_reason = "end_of_data"

                if exit_price is not None:
                    trade_counter += 1
                    trade = self._close_position(
                        state=state,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        exit_time=candle.timestamp,
                        equity=equity,
                        trade_id=trade_counter,
                        entry_bar=i,
                    )
                    equity = trade.equity_after
                    trades.append(trade)

                    # Notify risk manager
                    risk_manager.notify_trade_closed(trade.net_pnl, equity, candle)

                    # Reset position state
                    state = BacktestState()

                    logger.debug(
                        f"Bar {i} EXIT ({exit_reason}): {candle.timestamp} "
                        f"fill={exit_price:.2f} net_pnl={trade.net_pnl:.2f} "
                        f"R={trade.pnl_r:.2f}"
                    )

            # ----------------------------------------------------------------
            # 3. If flat, ask strategy for new signal
            # ----------------------------------------------------------------
            if not state.in_position and pending_signal is None:
                # Check risk manager gates (cooldown, daily loss limit)
                allowed, gate_reason = risk_manager.can_trade(candle, equity)

                if allowed:
                    # Strategy evaluates candles[0..i] — no lookahead
                    signal = strategy.compute_signal(candles, i)
                    if signal is not None:
                        pending_signal = signal
                        pending_equity_before = equity
                        logger.debug(
                            f"Bar {i} SIGNAL: {candle.timestamp} {signal.direction} "
                            f"entry={signal.entry_price:.2f} ({signal.reason})"
                        )

            # ----------------------------------------------------------------
            # 4. Tick risk manager (decrement cooldown)
            # ----------------------------------------------------------------
            risk_manager.tick()

            # ----------------------------------------------------------------
            # 5. Record equity
            # ----------------------------------------------------------------
            # Mark-to-market: include unrealized PnL of open position
            unrealized_pnl = 0.0
            if state.in_position:
                if state.current_signal.direction == "long":
                    unrealized_pnl = (candle.close - state.entry_price) * state.position_size
                else:
                    unrealized_pnl = (state.entry_price - candle.close) * state.position_size

            equity_curve.append(equity + unrealized_pnl)

        logger.info(
            f"Backtest complete: {len(trades)} trades, "
            f"final equity={equity:.2f} "
            f"({(equity/self.initial_equity - 1)*100:.1f}% return)"
        )

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            candles=candles,
            initial_equity=self.initial_equity,
            final_equity=equity,
        )

    def _close_position(
        self,
        state: BacktestState,
        exit_price: float,
        exit_reason: str,
        exit_time: datetime,
        equity: float,
        trade_id: int,
        entry_bar: int,
    ) -> TradeRecord:
        """Compute and return a completed TradeRecord."""
        pos = state.position_size
        entry = state.entry_price
        sl = state.stop_loss
        tp = state.take_profit
        direction = state.current_signal.direction
        is_long = direction == "long"

        # Gross PnL: signed by direction
        if is_long:
            gross_pnl = (exit_price - entry) * pos
        else:
            gross_pnl = (entry - exit_price) * pos

        # Exit fee (always positive, based on notional)
        exit_fee = pos * exit_price * self.fee_rate

        # Net PnL (entry fee already deducted from equity at entry)
        net_pnl = gross_pnl - exit_fee

        equity_after = equity + gross_pnl - exit_fee

        # R multiple: net_pnl / initial_risk_in_dollar_terms
        if is_long:
            initial_risk = (entry - sl) * pos   # Positive: entry above stop
        else:
            initial_risk = (sl - entry) * pos   # Positive: stop above entry
        pnl_r = net_pnl / initial_risk if initial_risk > 0 else 0.0

        # MFE/MAE in dollar terms (direction-aware)
        if is_long:
            mfe = (state.best_price - entry) * pos
            mae = (state.worst_price - entry) * pos
        else:
            mfe = (entry - state.worst_price) * pos   # worst_price = lowest low = best for short MFE
            mae = (entry - state.best_price) * pos    # best_price = highest high = worst for short

        return TradeRecord(
            trade_id=trade_id,
            direction=direction,
            entry_time=state.entry_time or exit_time,
            entry_price=entry,
            entry_signal_price=state.entry_signal_price,
            exit_time=exit_time,
            exit_price=exit_price,
            exit_reason=exit_reason,
            stop_loss=sl,
            take_profit=tp,
            position_size=pos,
            entry_fee=state.entry_fee,
            exit_fee=exit_fee,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            pnl_r=pnl_r,
            max_favorable_excursion=mfe,
            max_adverse_excursion=mae,
            equity_before=state.equity_before_trade,
            equity_after=equity_after,
            signal_reason=state.current_signal.reason if state.current_signal else "",
            entry_context=state.current_signal.metadata if state.current_signal else {},
        )
