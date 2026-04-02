"""
Data Layer: Loading and validating OHLCV data from CSV files.

Expected CSV format:
    timestamp,open,high,low,close,volume
    2023-01-01 00:00:00,16500.0,16550.0,16480.0,16530.0,123.45
    ...

No lookahead bias is possible at this layer — we simply load historical data.
All timestamps are parsed to UTC-aware datetimes to avoid timezone bugs.
"""

import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Candle:
    """A single OHLCV bar. Immutable by convention — never mutate after creation."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self):
        # Enforce basic OHLC sanity. Data errors here mean silent PnL bugs later.
        if not (self.low <= self.open <= self.high):
            raise ValueError(
                f"OHLC violation at {self.timestamp}: "
                f"open={self.open} not in [{self.low}, {self.high}]"
            )
        if not (self.low <= self.close <= self.high):
            raise ValueError(
                f"OHLC violation at {self.timestamp}: "
                f"close={self.close} not in [{self.low}, {self.high}]"
            )
        if self.low > self.high:
            raise ValueError(
                f"Low > High at {self.timestamp}: low={self.low}, high={self.high}"
            )
        if self.volume < 0:
            raise ValueError(f"Negative volume at {self.timestamp}: {self.volume}")


class DataLoader:
    """
    Loads OHLCV data from CSV, cleans it, validates it, and slices by date.

    Design decisions:
    - Strict validation on load rather than lazy — fail loudly on bad data
    - Returns plain list of Candle objects — no pandas dependency for the core engine
    - Date slicing uses closed-left, closed-right interval [start, end]
    """

    # Columns we accept (case-insensitive header matching)
    REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}

    def __init__(self, strict: bool = True):
        """
        Args:
            strict: If True, raises on any data quality issue. If False, skips bad rows.
        """
        self.strict = strict

    def load(
        self,
        filepath: str | Path,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Candle]:
        """
        Load candles from a CSV file, optionally filtered to [start_date, end_date].

        Args:
            filepath: Path to CSV file.
            start_date: Inclusive start date filter (UTC).
            end_date: Inclusive end date filter (UTC).

        Returns:
            Sorted list of Candle objects (ascending by timestamp).

        Raises:
            FileNotFoundError: If CSV not found.
            ValueError: If CSV format is invalid or data fails validation (strict=True).
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        logger.info(f"Loading data from {filepath}")
        candles = []
        skipped = 0

        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Normalize headers to lowercase
            if reader.fieldnames is None:
                raise ValueError("CSV file appears empty or has no header row.")
            normalized_headers = {h.strip().lower() for h in reader.fieldnames}

            missing = self.REQUIRED_COLUMNS - normalized_headers
            if missing:
                raise ValueError(f"CSV missing required columns: {missing}")

            for row_num, raw_row in enumerate(reader, start=2):
                # Normalize keys
                row = {k.strip().lower(): v.strip() for k, v in raw_row.items()}
                try:
                    candle = self._parse_row(row)

                    # Apply date filter before building objects we don't need
                    if start_date and candle.timestamp < _to_utc(start_date):
                        continue
                    if end_date and candle.timestamp > _to_utc(end_date):
                        continue

                    candles.append(candle)

                except (ValueError, KeyError) as e:
                    if self.strict:
                        raise ValueError(f"Row {row_num}: {e}") from e
                    else:
                        logger.warning(f"Skipping row {row_num}: {e}")
                        skipped += 1

        if skipped > 0:
            logger.warning(f"Skipped {skipped} invalid rows from {filepath.name}")

        if not candles:
            raise ValueError(
                f"No candles loaded from {filepath.name} "
                f"(after date filter: {start_date} to {end_date})"
            )

        # Sort by timestamp — never assume CSV is sorted
        candles.sort(key=lambda c: c.timestamp)

        # Detect and warn about gaps (important for 15m data)
        self._check_continuity(candles)

        logger.info(
            f"Loaded {len(candles)} candles: "
            f"{candles[0].timestamp} → {candles[-1].timestamp}"
        )
        return candles

    def _parse_row(self, row: dict) -> Candle:
        """Parse a single CSV row dict into a Candle."""
        ts_raw = row["timestamp"]

        # Try multiple timestamp formats
        ts = _parse_timestamp(ts_raw)

        return Candle(
            timestamp=ts,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )

    def _check_continuity(self, candles: List[Candle]) -> None:
        """
        Warn about gaps in the data. Gaps > 2x the inferred bar interval
        suggest missing data that could cause inflated indicator lookback periods.
        """
        if len(candles) < 2:
            return

        # Infer bar interval from first few bars
        intervals = []
        for i in range(1, min(10, len(candles))):
            delta = (candles[i].timestamp - candles[i - 1].timestamp).total_seconds()
            intervals.append(delta)
        median_interval = sorted(intervals)[len(intervals) // 2]

        gap_count = 0
        for i in range(1, len(candles)):
            delta = (candles[i].timestamp - candles[i - 1].timestamp).total_seconds()
            if delta > median_interval * 2:
                gap_count += 1
                if gap_count <= 5:  # Don't spam logs with hundreds of gap warnings
                    logger.warning(
                        f"Data gap: {candles[i-1].timestamp} → {candles[i].timestamp} "
                        f"({delta/60:.0f} min gap)"
                    )

        if gap_count > 5:
            logger.warning(f"... and {gap_count - 5} more data gaps.")

    def split(
        self,
        candles: List[Candle],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ) -> Tuple[List[Candle], List[Candle], List[Candle]]:
        """
        Time-based train/validation/test split.

        Ratios must sum to <= 1.0. Remaining data goes to test.
        This is a TEMPORAL split — never shuffle time-series data.

        Args:
            candles: Full sorted candle list.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.

        Returns:
            (train, validation, test) candle lists.
        """
        if train_ratio + val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1.0")

        n = len(candles)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train = candles[:train_end]
        val = candles[train_end:val_end]
        test = candles[val_end:]

        logger.info(
            f"Data split: train={len(train)} ({candles[0].timestamp.date()} to "
            f"{candles[train_end-1].timestamp.date()}), "
            f"val={len(val)}, test={len(test)}"
        )
        return train, val, test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TIMESTAMP_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S+00:00",
    "%Y-%m-%dT%H:%M:%S+00:00",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
]


def _parse_timestamp(raw: str) -> datetime:
    """Try multiple common formats; return UTC-aware datetime."""
    raw = raw.strip()

    # Already ISO 8601 with offset
    if "+" in raw[10:] or raw.endswith("Z"):
        # Let Python parse it; replace Z with +00:00 for compatibility
        raw_iso = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(raw_iso)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass

    # Unix timestamp (integer or float)
    try:
        ts_float = float(raw)
        # Heuristic: if > 1e10 it's likely milliseconds
        if ts_float > 1e10:
            ts_float /= 1000
        return datetime.fromtimestamp(ts_float, tz=timezone.utc)
    except ValueError:
        pass

    # Try explicit formats
    for fmt in _TIMESTAMP_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            # Assume UTC if no timezone info
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    raise ValueError(f"Cannot parse timestamp: '{raw}'")


def _to_utc(dt: datetime) -> datetime:
    """Ensure a datetime is UTC-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
