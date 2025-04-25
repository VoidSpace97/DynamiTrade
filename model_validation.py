import argparse
import os
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb

# --- CONFIG ---
parser = argparse.ArgumentParser()
parser.add_argument("--study_name", type=str, required=True, help="Subfolder under study_results/ (e.g., SOL4)")
args = parser.parse_args()

STUDY_DIR = os.path.join("study_results", args.study_name)
VALIDATION_DATA_PATH = "data/validation_dataset.parquet"
INITIAL_BALANCE = 100.0
MAX_HOLD = 24
STOP_LOSS = 0.025
FEE = 0.001

# --- Load Models and Parameters ---
with open(os.path.join(STUDY_DIR, "best_params.yaml"), "r") as f:
    params = yaml.safe_load(f)

clf = lgb.Booster(model_file=os.path.join(STUDY_DIR, "best_classifier.txt"))
reg = lgb.Booster(model_file=os.path.join(STUDY_DIR, "best_regressor.txt"))

# --- Feature Engineering ---
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def add_indicators(df):
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['rsi_14'] = compute_rsi(df['close'])
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    return df

def assign_market_regime(df, adx_period=14, adx_threshold=25):
    delta_high = df['high'].diff()
    delta_low = df['low'].diff()
    up = delta_high.where(delta_high > delta_low, 0.0)
    down = -delta_low.where(delta_low > delta_high, 0.0)
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    tr_smooth = tr.rolling(adx_period).mean()
    plus_dm = up.rolling(adx_period).sum()
    minus_dm = down.rolling(adx_period).sum()
    plus_di = 100 * (plus_dm / tr_smooth)
    minus_di = 100 * (minus_dm / tr_smooth)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(adx_period).mean()
    df['adx'] = adx
    df['market_regime'] = 'consolidation'
    df.loc[df['adx'] > adx_threshold, 'market_regime'] = 'trend'
    return df

# --- Simulation Function ---
def simulate_trading(y_cls, p_cls, price_series, conf, timestamps, regimes,
                     base_size, conf_thresh, pred_ret, return_mean, return_std,
                     max_hold=24, stop_loss=0.025, fee=0.001):
    balance = 100.0
    log = []
    pnl, wins, trades = 0.0, 0, 0
    cumulative_pnl = 0.0

    holding = False
    entry_index = None
    entry_time = None
    entry_price = None
    target_price = None
    hold_size = 0.0

    for i in range(len(y_cls)):
        current_ts = timestamps[i]
        pred_class = p_cls[i]
        pred_conf = conf[i]
        regime = regimes[i]
        price_now = price_series[i]

        size_scale = 1.2 if regime == 'trend' else 0.8
        conf_scale = 0.9 if regime == 'trend' else 1.1

        if not holding and pred_class == 2 and pred_conf >= conf_thresh * conf_scale:
            size = base_size * size_scale
            size = min(size, 1.0)
            denorm_ret = pred_ret[i] * return_std + return_mean
            target_price = price_now * np.exp(denorm_ret)

            holding = True
            entry_index = i
            entry_time = current_ts
            entry_price = price_now
            hold_size = size

        elif holding:
            price_delta = (price_now - entry_price) / entry_price
            exit = False
            reason = None

            if price_now >= target_price:
                exit = True
                reason = "target_hit"
            elif i - entry_index >= max_hold:
                exit = True
                reason = "timeout"
            elif price_delta <= -stop_loss:
                exit = True
                reason = "stop_loss"

            if exit:
                net_ret = (price_now / entry_price - 1) - 2 * fee
                realized_pnl = hold_size * net_ret
                balance *= (1 + realized_pnl)
                pnl += net_ret
                cumulative_pnl += realized_pnl
                wins += int(net_ret > 0)
                trades += 1

                log.append({
                    'entry_time': entry_time,
                    'exit_time': current_ts,
                    'hold_duration': i - entry_index,
                    'exit_reason': reason,
                    'entry_price': entry_price,
                    'exit_price': price_now,
                    'price_return': price_delta,
                    'trade_pnl': realized_pnl,
                    'cumulative_pnl': cumulative_pnl,
                    'balance': balance,
                    'confidence': pred_conf,
                    'regime': regime
                })

                holding = False
                entry_index = None
                entry_price = None
                target_price = None
                hold_size = 0.0

    return {
        'final_balance': balance,
        'total_pnl': pnl,
        'win_rate': wins / trades if trades > 0 else 0.0,
        'trades': trades,
        'log': pd.DataFrame(log)
    }

# --- Load & Process Data ---
df = pd.read_parquet(VALIDATION_DATA_PATH)
df.set_index('timestamp', inplace=True)
df = df.resample('1h').agg({
    'open': 'first', 'high': 'max', 'low': 'min',
    'close': 'last', 'volume': 'sum'
}).dropna().reset_index()

df = add_indicators(df)
df = assign_market_regime(df)
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df['volatility'] = df['log_return'].rolling(10).std()
df.dropna(inplace=True)

horizon = params['horizon']
df['future'] = df['close'].shift(-horizon)
df['future_return'] = np.log(df['future'] / df['close']).rolling(3).mean()
df['return_mean'] = df['future_return'].mean()
df['return_std'] = df['future_return'].std() + 1e-8
df['future_return'] = (df['future_return'] - df['return_mean']) / df['return_std']
df.dropna(inplace=True)

features = [c for c in df.columns if c not in ['timestamp', 'future_return', 'market_regime', 'future']]
X = df[features]
y_cls = np.zeros(len(df))  # dummy
clf_probs = clf.predict(X)
reg_preds = reg.predict(X)
conf = clf_probs.max(axis=1)
pred_cls = clf_probs.argmax(axis=1)

conf_thresh = np.quantile(conf, params['conf_thresh_q'])

metrics = simulate_trading(
    y_cls=y_cls,
    p_cls=pred_cls,
    price_series=df['close'].values,
    conf=conf,
    timestamps=df['timestamp'].values,
    regimes=df['market_regime'].values,
    base_size=params['base_position_size'],
    conf_thresh=conf_thresh,
    pred_ret=reg_preds,
    return_mean=df['return_mean'].iloc[0],
    return_std=df['return_std'].iloc[0],
    max_hold=MAX_HOLD,
    stop_loss=STOP_LOSS,
    fee=FEE
)

# --- Output ---
print(f"Final Balance: ${metrics['final_balance']:.2f}")
print(f"Total Trades: {metrics['trades']}")
print(f"Win Rate: {metrics['win_rate']:.2%}")
metrics['log'].to_csv("validation_trade_log.csv", index=False)
