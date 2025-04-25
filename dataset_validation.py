#!/usr/bin/env python3

import os
import threading
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone, timedelta
import yaml
from binance.spot import Spot

# === CONFIGURATION ===
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

PAIR_USDC = "SOLUSDC"
PAIR_USDT = "SOLUSDT"
OUT_PATH = "data/validation_dataset.parquet"
BASE_DATASET = cfg['paths']['unified']['base']
LOOKBACK_HOURS = 48
INTERVAL = "1m"

os.makedirs("data", exist_ok=True)

client = Spot()

# === LOGGING FUNCTION ===
def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}")

# === FETCH FUNCTION ===
def fetch_ohlcv(symbol, since_ms, until_ms, thread_id, only_volume=False):
    log(f"[{symbol}][Thread-{thread_id}] Fetching from {datetime.fromtimestamp(since_ms / 1000, timezone.utc)} to {datetime.fromtimestamp(until_ms / 1000, timezone.utc)}")
    all_data = []

    while since_ms < until_ms:
        try:
            klines = client.klines(symbol, INTERVAL, startTime=since_ms, endTime=until_ms, limit=1000)
            if not klines:
                break
            df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume",
                                               "close_time", "quote_asset_volume", "number_of_trades",
                                               "taker_buy_base", "taker_buy_quote", "ignore"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["volume"] = df["volume"].astype(float)

            if only_volume:
                df = df[["timestamp", "volume"]]
            else:
                df = df[["timestamp", "open", "high", "low", "close", "volume"]]
                df = df.astype({
                    "open": "float",
                    "high": "float",
                    "low": "float",
                    "close": "float",
                    "volume": "float"
                })

            all_data.append(df)
            since_ms = int(df["timestamp"].max().timestamp() * 1000) + 60_000
            time.sleep(0.2)
        except Exception as e:
            log(f"[{symbol}][Thread-{thread_id}] Error: {e}")
            time.sleep(1)

    return pd.concat(all_data) if all_data else pd.DataFrame()

# === GAP FILLING (SOL/USDC candles only) ===
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

    gap_ranges = [(int(ts.timestamp() * 1000), int(ts.timestamp() * 1000) + 60_000) for ts in missing]

    def gap_worker(symbol, start_ms, end_ms):
        gap_df = fetch_ohlcv(symbol, start_ms, end_ms, thread_id=f"gap-{symbol}")
        if not gap_df.empty:
            gap_results.append(gap_df)

    for start_ms, end_ms in gap_ranges:
        t = threading.Thread(target=gap_worker, args=(PAIR_USDC, start_ms, end_ms))
        threads.append(t)
        t.start()
        time.sleep(0.01)

    for t in threads:
        t.join()

    if gap_results:
        df_filled = pd.concat([df] + gap_results).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        log(f"✅ Gaps filled. New total candles: {len(df_filled)}")
        return df_filled

    log("⚠️ Gap filling returned no data. Check API limits or network.")
    return df

# === MAIN PROCESS ===
def main():
    if os.path.exists(BASE_DATASET):
        df_base = pd.read_parquet(BASE_DATASET)
        since = df_base["timestamp"].max() if not df_base.empty else datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
        log(f"Base dataset latest timestamp: {since}")
    else:
        log("Base dataset not found. Using safety lookback window.")
        since = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)

    now = datetime.now(timezone.utc)
    since_ms = int(since.timestamp() * 1000) + 60_000
    until_ms = int(now.timestamp() * 1000)

    results = {}

    def worker(symbol, key, only_volume=False):
        results[key] = fetch_ohlcv(symbol, since_ms, until_ms, thread_id=key, only_volume=only_volume)

    threads = [
        threading.Thread(target=worker, args=(PAIR_USDC, 'usdc', False)),
        threading.Thread(target=worker, args=(PAIR_USDT, 'usdt', True))
    ]

    for t in threads:
        t.start()
        time.sleep(0.05)

    for t in threads:
        t.join()

    df_usdc = results.get('usdc', pd.DataFrame())
    df_usdt_volume = results.get('usdt', pd.DataFrame())

    if df_usdc.empty:
        log("No new SOL/USDC data fetched. Exiting.")
        return

    df_usdc = verify_and_fill_gaps(df_usdc)

    if df_usdt_volume.empty:
        log("No SOL/USDT volume data fetched. Merging without volume_usdt.")
        df_usdc["volume_usdt"] = np.nan
    else:
        df_merged = pd.merge(df_usdc, df_usdt_volume.rename(columns={"volume": "volume_usdt"}), on="timestamp", how="left")
        df_usdc = df_merged

    df_usdc["total_volume"] = df_usdc["volume"] + df_usdc["volume_usdt"]

    df_usdc.to_parquet(OUT_PATH, index=False, compression="snappy")
    log(f"✅ Validation dataset saved: {OUT_PATH} — Total candles: {len(df_usdc)}")

if __name__ == "__main__":
    main()
