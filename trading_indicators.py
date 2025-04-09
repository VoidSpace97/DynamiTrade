#!/usr/bin/env python3
"""
Trading Indicators & Feature Engineering
Multi-timeframe indicators: 1m, 5m, 15m, 1H, 4H, 1D
"""

import pandas as pd
import numpy as np
import yaml
from ta.volume import VolumeWeightedAveragePrice, MFIIndicator
from ta.trend import PSARIndicator
from ta.volatility import KeltnerChannel, AverageTrueRange
from sklearn.utils import resample

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

FEATURE_CONF = config['model']['features']

# ------------------------------------------------------------------------------
# UTILITY INDICATORS
# ------------------------------------------------------------------------------

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_tema(series, window=14):
    ema1 = series.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    return 3 * (ema1 - ema2) + ema3

def compute_obv(df):
    direction = np.sign(df['close'].diff().fillna(0))
    return (direction * df['volume']).fillna(0).cumsum()

def compute_candle_features(df):
    df["body"] = (df["close"] - df["open"]).abs()
    df["upper_shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_shadow"] = df[["close", "open"]].min(axis=1) - df["low"]
    return df

# ------------------------------------------------------------------------------
# MULTI-TIMEFRAME FEATURE GENERATOR
# ------------------------------------------------------------------------------

def compute_timeframe_features(df, rule, prefix):
    df_tf = df.resample(rule, on='timestamp').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()

    df_tf[f'{prefix}_sma_5'] = df_tf['close'].rolling(5).mean()
    df_tf[f'{prefix}_rsi_14'] = compute_rsi(df_tf['close'])
    df_tf[f'{prefix}_atr_14'] = AverageTrueRange(df_tf['high'], df_tf['low'], df_tf['close'], window=14).average_true_range()
    df_tf[f'{prefix}_momentum_10'] = df_tf['close'].diff(10)
    df_tf[f'{prefix}_trend_slope'] = df_tf['close'].rolling(3).mean().diff()
    df_tf[f'{prefix}_volatility_10'] = df_tf['close'].rolling(10).std()

    df_tf = compute_candle_features(df_tf)
    df_tf = df_tf.rename(columns={'timestamp': f'{prefix}_timestamp'})
    drop_columns = ['open', 'high', 'low', 'close', 'volume', 'body', 'upper_shadow', 'lower_shadow']
    df_tf = df_tf.drop(columns=[col for col in drop_columns if col in df_tf.columns])

    return df_tf

# ------------------------------------------------------------------------------
# MAIN TECHNICAL INDICATORS
# ------------------------------------------------------------------------------
def compute_technical_indicators(df, horizon, buy_mult=2.0, sell_mult=2.0, compute_labels=True, raw_df=None):
    df = df.copy().reset_index(drop=True)

    # Base timeframe indicators (1m)
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['momentum_10'] = df['close'].diff(10)

    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    df['bb_width'] = (2 * std_20) / sma_20

    df['rsi_14'] = compute_rsi(df['close'])
    df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)

    df['close_lag1'] = df['close'].shift(1)
    df['pct_change_1'] = df['close'].pct_change() * 100
    df['volatility_10'] = df['close'].rolling(10).std()
    df['vol_regime'] = (df['volatility_10'] > df['volatility_10'].median()).astype(int)

    df['obv'] = compute_obv(df)
    df['atr_14'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['vwap_14'] = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=14).volume_weighted_average_price()
    df['mfi_14'] = MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=14).money_flow_index()
    df['tema_14'] = compute_tema(df['close'])
    df['kc_width'] = KeltnerChannel(df['high'], df['low'], df['close'], window=20, window_atr=10).keltner_channel_wband()
    df['psar'] = PSARIndicator(df['high'], df['low'], df['close'], step=0.02, max_step=0.2).psar()

    df['hour'] = df['timestamp'].dt.hour
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['volatility_regime'] = df['atr_14'] / df['close']

    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['cum_volume'] = df['volume'].cumsum()
    df['cum_vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['cum_volume']
    df['vwap_distance'] = (df['close'] - df['cum_vwap']) / df['cum_vwap']

    df = compute_candle_features(df)

    # Multi-timeframe indicators
    timeframes = {'5min': 'tf5m', '15min': 'tf15m', '1h': 'tf1h', '4h': 'tf4h', '1D': 'tf1d'}
    source_df = raw_df if raw_df is not None else df
    for rule, prefix in timeframes.items():
        df_tf = compute_timeframe_features(source_df, rule, prefix)
        df = pd.merge_asof(
            df.sort_values('timestamp'),
            df_tf.sort_values(f'{prefix}_timestamp'),
            left_on='timestamp',
            right_on=f'{prefix}_timestamp',
            direction='backward',
            suffixes=('', f'_{prefix}'))

    before_rows = len(df)
    df = df.dropna().reset_index(drop=True)
    after_rows = len(df)
    print(f"[CLEAN] Dropped {before_rows - after_rows} rows due to incomplete features.")

    # Labels
    if compute_labels:
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
        vol = df['close'].pct_change().rolling(20).std()
        df['label'] = 1
        df.loc[df['future_return'] > vol * buy_mult, 'label'] = 2
        df.loc[df['future_return'] < -vol * sell_mult, 'label'] = 0

    return df, df.get('future_return'), df.get('timestamp')

# ------------------------------------------------------------------------------
# TRAINING DATA
# ------------------------------------------------------------------------------

def balance_classes(df):
    majority = df[df['target'] == 1]
    minority_buy = df[df['target'] == 2]
    minority_sell = df[df['target'] == 0]

    n = max(len(minority_buy), len(minority_sell))
    n_majority = min(len(majority), n)

    return pd.concat([
        resample(majority, replace=False, n_samples=n_majority, random_state=42),
        resample(minority_buy, replace=True, n_samples=n, random_state=42),
        resample(minority_sell, replace=True, n_samples=n, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

def prepare_training_data(symbol, threshold_buy, threshold_sell, horizon, buy_mult=2.0, sell_mult=2.0):
    df = pd.read_parquet("data/feature_store.parquet")
    df = df[df['timestamp'].notnull()].copy()
    if df.empty:
        raise ValueError("[ERROR] No OHLCV data available.")

    df["target"] = df["label"]
    df.dropna(inplace=True)

    df = balance_classes(df)
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'target', 'label', 'future_return']]
    return df, df["target"], feature_cols

# ------------------------------------------------------------------------------
# LIVE INFERENCE FEATURES
# ------------------------------------------------------------------------------

def prepare_live_features(symbol, horizon, return_df=False):
    df = pd.read_parquet("data/feature_store.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True).tail(1500).copy()
    df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'close_time'], inplace=True, errors='ignore')
    latest = df.iloc[-1:].copy()
    feature_cols = [col for col in latest.columns if col not in ['timestamp', 'future_return', 'target']]
    X_live = latest[feature_cols].values
    if return_df:
        return X_live, feature_cols, df
    return X_live, feature_cols
