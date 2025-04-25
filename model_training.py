import os
import yaml
import optuna
import logging
import lightgbm as lgb
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import TimeSeriesSplit
from optuna.samplers import TPESampler
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# Logging
logging.basicConfig(filename='training_reinforced.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Globals
STUDY_DIR = None
GLOBAL_DF = None
BEST_SCORE = -np.inf

# --- Indicators ---
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def add_indicators(df):
    df['ema_12'] = df['close'].ewm(span=720).mean()
    df['ema_26'] = df['close'].ewm(span=1560).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=540).mean()
    df['rsi_14'] = compute_rsi(df['close'], window=840)
    df['bb_mid'] = df['close'].rolling(1200).mean()
    df['bb_std'] = df['close'].rolling(1200).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    return df

def assign_market_regime(df, adx_period=14, adx_threshold=25):
    # Calculate ADX
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

    # Classify regime
    df['market_regime'] = 'consolidation'
    df.loc[df['adx'] > adx_threshold, 'market_regime'] = 'trend'

    df['market_regime'] = df['market_regime'].shift(1)  # Prevent leakage
    return df

# --- Feature Engineering ---
def create_features(df):
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_return'].rolling(600).std()
    df['momentum'] = df['close'] / df['close'].shift(90) - 1
    df['price_range'] = (df['high'] - df['low']) / df['open']
    df['trend_strength'] = df['close'].rolling(600).apply(lambda x: np.mean(np.sign(x.diff())), raw=False)

    # Efficient lag feature construction
    lags = list(range(60, 121, 10)) + list(range(150, 241, 30)) + list(range(300, 361, 60))
    lagged_data = {
        f'return_lag_{lag}': df['log_return'].shift(lag)
        for lag in lags
    }
    lagged_data.update({
        f'vol_lag_{lag}': df['volatility'].shift(lag)
        for lag in lags
    })
    lagged_df = pd.DataFrame(lagged_data, index=df.index)
    df = pd.concat([df, lagged_df], axis=1)

    return df


# --- Simulated Trading ---
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

        # Regime-based scaling
        if regime == 'trend':
            size_scale = 1.2
            conf_scale = 0.9
        else:
            size_scale = 0.8
            conf_scale = 1.1

        # --- ENTRY ---
        if not holding and pred_class == 2:  # BUY signal
            if pred_conf >= conf_thresh * conf_scale:
                size = base_size * size_scale

                denorm_ret = pred_ret[i] * return_std + return_mean
                target_price = price_now * np.exp(denorm_ret)

                holding = True
                entry_index = i
                entry_time = current_ts
                entry_price = price_now
                hold_size = size

        # --- EXIT ---
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

        # Log current state
        log.append({
            'timestamp': current_ts,
            'position': 1 if holding else 0,
            'status': 'open' if not holding and pred_class == 2 else 'hold',
            'hold_duration': i - entry_index if holding else 0,
            'balance': balance,
            'confidence': pred_conf,
            'price': price_now,
            'regime': regime
        })

    return {
        'final_balance': balance,
        'total_pnl': pnl,
        'win_rate': wins / trades if trades > 0 else 0.0,
        'trades': trades,
        'log': pd.DataFrame(log)
    }

# --- Objective ---
def objective(trial):
    global GLOBAL_DF, STUDY_DIR, BEST_SCORE

    df = GLOBAL_DF.copy()
    horizon = trial.suggest_int("horizon", 32, 240)
    volatility_factor = trial.suggest_float("volatility_factor", 0.5, 2.0)
    df['future'] = df['close'].shift(-horizon)
    base_size = trial.suggest_float("base_position_size", 0.75, 1.2)
    adx_threshold = trial.suggest_int("adx_threshold", 60, 130)
    adx_period = trial.suggest_int("adx_period", 40, 120)
    df = assign_market_regime(df, adx_period=adx_period, adx_threshold=adx_threshold)

    df['future_return'] = np.log(df['future'] / df['close']).rolling(3).mean()
    df['future_return'] = df['future_return'].clip(lower=-0.1, upper=0.1)
    df['return_mean'] = df['future_return'].mean()
    df['return_std'] = df['future_return'].std() + 1e-8
    df['future_return'] = (df['future_return'] - df['return_mean']) / df['return_std']
    ql, qh = df['future_return'].quantile([0.35, 0.65])
    df['label'] = 1  # Default HOLD

    trend_mask = df['market_regime'] == 'trend'
    cons_mask = df['market_regime'] == 'consolidation'
    df.loc[trend_mask & (df['future_return'] > qh), 'label'] = 2
    df.loc[trend_mask & (df['future_return'] < ql), 'label'] = 0
    df.loc[cons_mask & (df['future_return'] > qh + df['volatility'] * volatility_factor), 'label'] = 2
    df.loc[cons_mask & (df['future_return'] < ql - df['volatility'] * volatility_factor), 'label'] = 0

    df.dropna(inplace=True)

    features = [c for c in df.columns if c not in ['timestamp', 'future_return', 'label', 'market_regime']]
    df['regime_flag'] = df['market_regime'].map({'trend': 1, 'consolidation': 0})
    X = df[features]
    y_cls = df['label']
    y_reg = df['future_return']
    class_freq = y_cls.value_counts(normalize=True).reindex([0, 1, 2], fill_value=1e-8)
    weights = y_cls.map(class_freq.rdiv(1.0))

    clf_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'learning_rate': trial.suggest_float("clf_lr", 0.005, 0.05),
        'num_leaves': trial.suggest_int("clf_leaves", 64, 256),
        'feature_fraction': trial.suggest_float("clf_ff", 0.5, 0.9),
        'bagging_fraction': trial.suggest_float("clf_bf", 0.5, 0.9),
        'bagging_freq': trial.suggest_int("clf_bf_freq", 1, 5),
        'lambda_l1': trial.suggest_float("clf_l1", 0.0, 5.0),
        'lambda_l2': trial.suggest_float("clf_l2", 0.0, 5.0),
        'max_depth': trial.suggest_int("clf_depth", 3, 7),
        'verbosity': -1,
        'seed': 42
    }

    reg_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': trial.suggest_float("reg_lr", 0.005, 0.05),
        'num_leaves': trial.suggest_int("reg_leaves", 64, 256),
        'feature_fraction': trial.suggest_float("reg_ff", 0.5, 0.9),
        'bagging_fraction': trial.suggest_float("reg_bf", 0.5, 0.9),
        'bagging_freq': trial.suggest_int("reg_bf_freq", 1, 5),
        'lambda_l1': trial.suggest_float("reg_l1", 0.0, 5.0),
        'lambda_l2': trial.suggest_float("reg_l2", 0.0, 5.0),
        'max_depth': trial.suggest_int("reg_depth", 3, 7),
        'verbosity': -1,
        'seed': 42
    }

    cv = TimeSeriesSplit(n_splits=10)
    balances = []

    for fold, (train, test) in enumerate(cv.split(X)):
        X_tr, X_te = X.iloc[train], X.iloc[test]
        y_cls_tr, y_cls_te = y_cls.iloc[train], y_cls.iloc[test]
        y_reg_tr, y_reg_te = y_reg.iloc[train], y_reg.iloc[test]
        w_tr = weights.iloc[train]

        clf = lgb.train(clf_params, lgb.Dataset(X_tr, label=y_cls_tr, weight=w_tr),
                        valid_sets=[lgb.Dataset(X_te, label=y_cls_te)],
                        num_boost_round=300, callbacks=[lgb.early_stopping(30)])
        reg = lgb.train(reg_params, lgb.Dataset(X_tr, label=y_reg_tr),
                        valid_sets=[lgb.Dataset(X_te, label=y_reg_te)],
                        num_boost_round=300, callbacks=[lgb.early_stopping(30)])

        pred_cls_probs = clf.predict(X_te)
        pred_cls = np.argmax(pred_cls_probs, axis=1)
        conf = pred_cls_probs.max(axis=1)
        pred_ret = reg.predict(X_te)
        pred_ret = np.clip(pred_ret, -0.1, 0.1)

        conf_thresh = np.quantile(conf, trial.suggest_float("conf_q", 0.55, 0.72))

        timestamps = df.index[test].values
        regimes = df.iloc[test]['market_regime'].values
        prices = df.iloc[test]['close'].values
        return_mean = y_reg.iloc[test].mean()
        return_std = y_reg.iloc[test].std() + 1e-8

        metrics = simulate_trading(
            y_cls=y_cls.iloc[test].values,
            p_cls=pred_cls,
            price_series=prices,
            conf=conf,
            timestamps=timestamps,
            regimes=regimes,
            base_size=base_size,
            conf_thresh=conf_thresh,
            pred_ret=pred_ret,
            return_mean=return_mean,
            return_std=return_std
        )

        balances.append(metrics['final_balance'])

    log_growth = np.mean(np.log(np.array(balances) / 100))
    std_growth = np.std(np.log(np.array(balances) / 100))
    sharpe = log_growth / (std_growth + 1e-8)

    final_retention = np.mean(np.array(balances)) / 100
    peak_balance = np.max(np.array(balances))
    final_balance = np.mean(np.array(balances))
    drawdown_penalty = max(0, 1 - (final_balance / (peak_balance + 1e-8)))

    composite = (
        log_growth
        + 0.1 * sharpe
        + 0.2 * np.log(final_retention + 1e-8)
        - 0.2 * drawdown_penalty
    )

    print(f"Trial {trial.number} Results:")
    print(f"  Balances: {balances}")
    print(f"  Log Growth: {log_growth}")
    print(f"  Sharpe: {sharpe}")
    print(f"  Composite Score: {composite}")

    if composite > BEST_SCORE:
        BEST_SCORE = composite
        with open(os.path.join(STUDY_DIR, 'best_params.yaml'), 'w') as f:
            yaml.dump(trial.params, f)
        
        # Save best trade log
        metrics['log'].to_csv(os.path.join(STUDY_DIR, 'best_trade_log.csv'), index=False)
        
        # Save best models
        clf.save_model(os.path.join(STUDY_DIR, 'best_classifier.txt'))
        reg.save_model(os.path.join(STUDY_DIR, 'best_regressor.txt'))

    trial_log_path = os.path.join(STUDY_DIR, f"trial_{trial.number}_log.csv")
    metrics['log'].to_csv(trial_log_path, index=False)

    return composite

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--study_name", type=str, default="adaptive_model")
    parser.add_argument("--study_dir", type=str, default="study_results")
    args = parser.parse_args()

    STUDY_DIR = os.path.join(args.study_dir, args.study_name)
    os.makedirs(STUDY_DIR, exist_ok=True)

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    path = config['paths']['unified']['base']
    df = pd.read_parquet(path)
    df.set_index('timestamp', inplace=True)

    df = add_indicators(df)
    df = create_features(df)

    df.dropna(inplace=True)

    GLOBAL_DF = df

    study = optuna.create_study(direction="maximize", study_name=args.study_name,
                                sampler=TPESampler(multivariate=True, group=True))
    study.optimize(objective, n_trials=args.trials)

    print("Best Trial:", study.best_trial.value)
    for k, v in study.best_trial.params.items():
        print(f"{k}: {v}")
