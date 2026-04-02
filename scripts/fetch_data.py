"""
Fetch historical OHLCV data from Binance public API (no auth required).

Usage:
    python scripts/fetch_data.py --symbol BTCUSDT --interval 15m --days 730
    python scripts/fetch_data.py --symbol BTCUSDT --interval 15m --start 2022-01-01 --end 2024-01-01

Output:
    data/BTCUSDT_15m.csv

Binance klines endpoint returns up to 1000 bars per request.
We paginate automatically to fetch the full requested range.

Rate limits: Binance allows ~1200 requests/minute without auth.
We add a small delay between requests to be polite.
"""

import csv
import json
import time
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path
import argparse
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_BARS_PER_REQUEST = 1000
REQUEST_DELAY_SECONDS = 0.2  # Be polite to Binance

# Interval to milliseconds mapping
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list:
    """
    Fetch all klines between start_ms and end_ms from Binance.
    Handles pagination automatically.

    Args:
        symbol: e.g. "BTCUSDT"
        interval: e.g. "15m"
        start_ms: Start timestamp in milliseconds (UTC)
        end_ms: End timestamp in milliseconds (UTC)

    Returns:
        List of kline lists (raw Binance format).
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}. Choose from {list(INTERVAL_MS)}")

    all_klines = []
    current_start = start_ms
    request_count = 0

    print(f"Fetching {symbol} {interval} from "
          f"{datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).date()} to "
          f"{datetime.fromtimestamp(end_ms/1000, tz=timezone.utc).date()}")

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": MAX_BARS_PER_REQUEST,
        }

        url = BINANCE_KLINES_URL + "?" + urllib.parse.urlencode(params)

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            print(f"HTTP error {e.code}: {e.reason}")
            raise
        except urllib.error.URLError as e:
            print(f"Connection error: {e.reason}")
            raise

        if not data:
            break

        all_klines.extend(data)
        request_count += 1

        # Advance to the timestamp after the last fetched bar
        last_open_time = data[-1][0]
        current_start = last_open_time + INTERVAL_MS[interval]

        bars_fetched = len(all_klines)
        progress_pct = min(100, (last_open_time - start_ms) / (end_ms - start_ms) * 100)
        print(f"  Fetched {bars_fetched} bars ({progress_pct:.0f}%)...", end="\r")

        if len(data) < MAX_BARS_PER_REQUEST:
            # We got fewer bars than the limit — we've reached the end
            break

        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nTotal: {len(all_klines)} bars fetched in {request_count} requests")
    return all_klines


def klines_to_csv(klines: list, output_path: Path, symbol: str, interval: str) -> None:
    """
    Convert raw Binance kline format to our CSV format.

    Binance kline format:
    [open_time, open, high, low, close, volume, close_time, ...]
    Indices 0-5 are what we need.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])

        for kline in klines:
            open_time_ms = int(kline[0])
            ts = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc)
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")

            writer.writerow([
                ts_str,
                kline[1],  # open
                kline[2],  # high
                kline[3],  # low
                kline[4],  # close
                kline[5],  # volume
            ])

    print(f"Saved {len(klines)} bars to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch OHLCV data from Binance")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair (default: BTCUSDT)")
    parser.add_argument("--interval", default="15m", help="Bar interval (default: 15m)")
    parser.add_argument("--days", type=int, default=730, help="Days of history (default: 730)")
    parser.add_argument("--start", help="Start date YYYY-MM-DD (overrides --days)")
    parser.add_argument("--end", help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--output", help="Output CSV path (default: data/SYMBOL_INTERVAL.csv)")
    args = parser.parse_args()

    # Determine date range
    now = datetime.now(tz=timezone.utc)

    if args.end:
        end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_dt = now

    if args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start_dt = end_dt - timedelta(days=args.days)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        project_root = Path(__file__).parent.parent
        output_path = project_root / "data" / f"{args.symbol}_{args.interval}.csv"

    # Fetch and save
    klines = fetch_klines(args.symbol, args.interval, start_ms, end_ms)
    if klines:
        klines_to_csv(klines, output_path, args.symbol, args.interval)
    else:
        print("No data fetched.")
        sys.exit(1)


if __name__ == "__main__":
    main()
