#!/usr/bin/env python3

import os
import threading
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone, timedelta
import yaml
from binance.spot import Spot

# === CONFIG ===

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

PAIR = "SOLUSDT"  # Binance native symbol format
OUT_PATH = "data/dataset_1m.parquet"
LOOKBACK_DAYS = 365 * 5  # 5 years
INTERVAL = "1m"

os.makedirs("data", exist_ok=True)

# === SETUP BINANCE CLIENT ===

client = Spot()

# === LOGGING ===

def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}")

# === FETCH FUNCTION ===

def fetch_chunk(symbol, start_ms, end_ms, results, thread_id):
    log(f"[Thread-{thread_id}] Fetching from {datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)} to {datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)}")

    all_data = []
    while start_ms < end_ms:
        try:
            klines = client.klines(symbol, INTERVAL, startTime=start_ms, endTime=end_ms, limit=1000)
            if not klines:
                break
            df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time",
                                               "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore"])
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.astype({
                "open": "float",
                "high": "float",
                "low": "float",
                "close": "float",
                "volume": "float"
            })
            all_data.append(df)
            start_ms = int(df["timestamp"].max().timestamp() * 1000) + 60_000
            time.sleep(0.2)  # Safety sleep
        except Exception as e:
            log(f"[Thread-{thread_id}] Error: {e}")
            time.sleep(1)
    if all_data:
        results.append(pd.concat(all_data))

# === GAP VERIFICATION & HEALING ===

def verify_and_fill_gaps(df):
    expected = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="1min", tz="UTC")
    actual = df["timestamp"]
    missing = expected.difference(actual)

    if missing.empty:
        log("✅ No gaps detected. Dataset is complete.")
        return df

    log(f"[WARNING] {len(missing)} gaps detected. Auto-filling...")

    gap_results = []
    threads = []
    gap_ranges = []
    for ts in missing:
        start = int(ts.timestamp() * 1000)
        end = start + 60_000
        gap_ranges.append((start, end))

    for idx, (start, end) in enumerate(gap_ranges):
        t = threading.Thread(target=fetch_chunk, args=(PAIR, start, end, gap_results, f"gap-{idx}"))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if gap_results:
        df_filled = pd.concat([df] + gap_results).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        log(f"✅ Gaps filled. New total candles: {len(df_filled)}")
        return df_filled

    log("⚠️ Gap filling returned no data. Check API limits.")
    return df

# === MAIN PROCESS ===

def main():
    now = datetime.now(timezone.utc)
    since_ms = int((now - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)
    latest_ms = int(now.timestamp() * 1000)

    df_existing = pd.read_parquet(OUT_PATH) if os.path.exists(OUT_PATH) else pd.DataFrame()
    if not df_existing.empty:
        since_ms = int(df_existing["timestamp"].max().timestamp() * 1000) + 60_000
        log(f"Resuming from existing dataset. Start time: {datetime.utcfromtimestamp(since_ms / 1000)}")

    # Split into parallel intervals (threads)
    interval_ms = 24 * 60 * 60 * 1000  # 1 day in milliseconds
    time_splits = list(range(since_ms, latest_ms, interval_ms))
    ranges = [(start, min(start + interval_ms, latest_ms)) for start in time_splits]

    results = []
    threads = []

    for idx, (start, end) in enumerate(ranges):
        t = threading.Thread(target=fetch_chunk, args=(PAIR, start, end, results, idx))
        threads.append(t)
        t.start()
        time.sleep(0.05)  # Small delay to prevent burst API limits

    for t in threads:
        t.join()

    if not results:
        log("No new data fetched. Exiting.")
        return

    df_new = pd.concat(results).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    # Merge with existing data
    if not df_existing.empty:
        df_all = pd.concat([df_existing, df_new]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    else:
        df_all = df_new

    # Verify and auto-fill gaps
    df_all = verify_and_fill_gaps(df_all)

    df_all.to_parquet(OUT_PATH, index=False, compression="snappy")
    log(f"✅ Dataset saved: {OUT_PATH} — Total candles: {len(df_all)}")

if __name__ == "__main__":
    main()
