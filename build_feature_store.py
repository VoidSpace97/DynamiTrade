#!/usr/bin/env python3
"""
Build Feature Store — Binance BTC/USDC OHLC + Aggregated Volume (BTC/USDC + BTC/USDT)
Compute indicators and store in Parquet, with incremental update.
"""

import time
import os
import pandas as pd
import ccxt
import yaml
from datetime import datetime, timezone, timedelta
from trading_indicators import compute_technical_indicators

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

SYMBOL_BASE = "BTC/USDC"
SYMBOL_VOLUME_EXTRA = "BTC/USDT"
PARQUET_PATH = "data/feature_store.parquet"
OHLCV_PARQUET_PATH = "data/ohlcv_data.parquet"
TIMEFRAME = '1m'
BATCH_LIMIT = 1000
LOOKBACK_DAYS = 365 * 5  # 5 years

os.makedirs("data", exist_ok=True)

exchange = ccxt.binance({'enableRateLimit': True})


def log(message):
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f"[{timestamp}] {message}")


# ----------------------------------------------------------------------
# LOAD EXISTING RAW OHLCV DATA
# ----------------------------------------------------------------------

def load_existing_ohlcv():
    if os.path.exists(OHLCV_PARQUET_PATH):
        df_existing = pd.read_parquet(OHLCV_PARQUET_PATH)
        log(f"[LOAD] Existing OHLCV data found: {len(df_existing)} rows.")
        return df_existing
    else:
        log("[LOAD] No existing OHLCV data found. Starting fresh.")
        return pd.DataFrame()


# ----------------------------------------------------------------------
# FETCH OHLCV DATA
# ----------------------------------------------------------------------

def fetch_ohlcv(symbol, since_ms):
    all_data = []

    log(f"[FETCH] {symbol} starting from: {datetime.fromtimestamp(since_ms / 1000, timezone.utc)}")

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since_ms, limit=BATCH_LIMIT)

        if not ohlcv:
            break

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        all_data.append(df)

        since_ms = ohlcv[-1][0] + 60_000  # advance by 1 min

        if len(ohlcv) < BATCH_LIMIT:
            break  # last batch

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data).drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    log(f"[FETCHED] {symbol}: {len(combined)} rows.")
    return combined


# ----------------------------------------------------------------------
# MAIN PROCESS
# ----------------------------------------------------------------------

def main():
    # Step 1: Load existing raw OHLCV data
    df_existing = load_existing_ohlcv()

    # Step 2: Determine fetch starting point
    if df_existing.empty:
        since = exchange.parse8601(
            (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%dT%H:%M:%SZ')
        )
    else:
        latest_timestamp = df_existing["timestamp"].max()
        since = int(latest_timestamp.timestamp() * 1000) + 60_000  # next candle

    # Step 3: Fetch new data
    df_usdc = fetch_ohlcv(SYMBOL_BASE, since)
    df_usdt = fetch_ohlcv(SYMBOL_VOLUME_EXTRA, since)

    if df_usdc.empty:
        log("[INFO] No new BTC/USDC data to process. Exiting.")
        return

    if df_usdt.empty:
        log("[INFO] No new BTC/USDT data. Proceeding with BTC/USDC only.")

    # Step 4: Merge volumes and ensure full OHLCV structure
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df_merged = df_usdc[required_columns].copy()

    if not df_usdt.empty:
        df_usdt_aligned = df_usdt.set_index("timestamp").reindex(df_merged["timestamp"]).fillna(0).reset_index()
        df_merged["volume"] += df_usdt_aligned["volume"]

    assert all(col in df_merged.columns for col in required_columns), "[ERROR] Missing required columns after merge."

    log(f"[MERGE] Combined new data: {len(df_merged)} rows.")

    # Step 5: Update and save OHLCV data
    if not df_existing.empty:
        df_combined = pd.concat([df_existing, df_merged]).drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    else:
        df_combined = df_merged

    log(f"[TOTAL] Combined OHLCV dataset size: {len(df_combined)} rows.")

    df_combined.to_parquet(OHLCV_PARQUET_PATH, index=False, compression="snappy")
    log(f"[SAVE] OHLCV data updated at: {OHLCV_PARQUET_PATH}")

    # Step 6: Compute indicators on clean OHLCV data
    df_features, _, _ = compute_technical_indicators(
        df_combined, horizon=20, buy_mult=2.0, sell_mult=2.0, compute_labels=True, raw_df=df_combined
    )
    df_features = df_features.dropna().reset_index(drop=True)

    log(f"[FEATURES] After computing indicators: {len(df_features)} rows.")

    # Step 7: Save feature store
    df_features.to_parquet(PARQUET_PATH, index=False, compression="snappy")
    log(f"[SAVE] Feature store updated at: {PARQUET_PATH}")
    log(f"[COMPLETE] Final row count: {len(df_features)} ✅")


# ----------------------------------------------------------------------
# CONTINUOUS MODE
# ----------------------------------------------------------------------

def continuous_update(interval_seconds=60):
    while True:
        main()
        time.sleep(interval_seconds)


# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--realtime", action="store_true", help="Run in continuous realtime mode")
    args = parser.parse_args()

    if args.realtime:
        continuous_update()
    else:
        main()
