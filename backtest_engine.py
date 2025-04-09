#!/usr/bin/env python3

"""
Backtest Engine — Executes historical backtests on trained models.
Outputs trade results and performance metrics.
Supports manual backtest and checkpoint backtests during training.
"""

import os
import yaml
import pandas as pd
import lightgbm as lgb
import inspect
from datetime import datetime, timezone
from trading_indicators import prepare_training_data
from reporting import generate_report_from_df

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

SYMBOL = config['model']['target_symbol']
MIN_AMOUNT = 1e-5
RISK_PCT = 0.05
TAKER_FEE_RATE = 0.0004
SLIPPAGE_BPS = 5 / 10000

# ------------------------------------------------------------------------------
# UTILITY: Dynamic Output Directory
# ------------------------------------------------------------------------------

def determine_output_dir(study_name):
    caller = inspect.stack()[2].function
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')

    if caller == 'objective':
        return os.path.join("optuna_experiments", study_name, f"checkpoint_{timestamp}")
    else:
        return os.path.join("csv", f"backtest_{study_name}_{timestamp}")

# ------------------------------------------------------------------------------
# BACKTEST FUNCTION
# ------------------------------------------------------------------------------

def run_backtest(model_path, feature_path, threshold_buy, threshold_sell, output_dir=None):
    print("[INFO] Starting backtest...")

    if output_dir is None:
        study_name = os.path.basename(os.path.dirname(model_path))
        output_dir = determine_output_dir(study_name)

    os.makedirs(output_dir, exist_ok=True)

    # Load model and feature list
    model = lgb.Booster(model_file=model_path)
    with open(feature_path, "r") as f:
        trained_features = [line.strip() for line in f if line.strip()]

    # Prepare and validate data
    df, _, _ = prepare_training_data(
        symbol=SYMBOL,
        threshold_buy=threshold_buy,
        threshold_sell=threshold_sell,
        horizon=20,
        buy_mult=2.0,
        sell_mult=2.0
    )
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    trained_features = df[trained_features].select_dtypes(include=['number']).columns.tolist()
    
    if not trained_features:
        raise ValueError("Trained feature list is empty. Check top_features.txt.")
    if missing := set(trained_features) - set(df.columns):
        raise ValueError(f"Backtest data missing features: {missing}")

    # Sort chronologically — CRITICAL
    df.sort_values(by="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Inference
    preds = model.predict(df[trained_features].values)
    df['predicted_class'] = preds.argmax(axis=1)
    df['confidence'] = preds.max(axis=1)

    # Exit rule preparation
    df['price'] = df['close']
    df['atr'] = df['atr_14'].fillna(1.0)
    df['tp'] = df['price'] + 2 * df['atr']
    df['sl'] = df['price'] - 1.5 * df['atr']
    df['entry_signal'] = (df['predicted_class'] == 2) & (df['confidence'] >= threshold_buy)

    trades = []
    in_position = False

    entry_signals_count = df['entry_signal'].sum()
    print(f"[INFO] Entry signals detected: {entry_signals_count}")


    for idx, row in df[df['entry_signal']].iterrows():
        if in_position:
            continue

        entry_price = row['price'] * (1 + SLIPPAGE_BPS)
        size = (1000.0 * RISK_PCT) / entry_price
        if size < MIN_AMOUNT:
            continue

        # Simulate forward path from this point (positional safety)
        sub_df = df.iloc[idx:].copy().reset_index(drop=True)
        exit_cond = (sub_df['price'] >= row['tp']) | (sub_df['price'] <= row['sl'])
        exit_pos = exit_cond.idxmax() if exit_cond.any() else sub_df.index[-1]
        exit_idx = idx + exit_pos

        exit_price = df.loc[exit_idx, 'price'] * (1 - SLIPPAGE_BPS)
        pnl = (exit_price - entry_price) * size
        fee_entry = entry_price * TAKER_FEE_RATE * size
        fee_exit = exit_price * TAKER_FEE_RATE * size

        trades.append({
            "entry_time": row["timestamp"],
            "entry_price": entry_price,
            "side": "long",
            "size": size,
            "exit_time": df.loc[exit_idx, "timestamp"],
            "exit_price": exit_price,
            "pnl": pnl,
            "fees": fee_entry + fee_exit,
            "trade_amount_usdc": 1000.0 * RISK_PCT
        })

        in_position = False

    trades_df = pd.DataFrame(trades)
    print(f"[INFO] Total trades executed: {len(trades_df)}")

    if not trades_df.empty:
        invalid_trades = trades_df[trades_df["exit_time"] < trades_df["entry_time"]]
        if not invalid_trades.empty:
            print(f"[FATAL] {len(invalid_trades)} invalid trades (exit before entry).")
            raise ValueError("Corrupted backtest detected: exit_time precedes entry_time.")

        # Proceed with report generation and export
        print("[INFO] Generating performance report...")
        report_metrics = generate_report_from_df(trades_df, save_reports=True, output_dir=output_dir)
        print(f"[REPORT] Total PnL: {report_metrics['total_pnl']:.2f} | "
            f"Sharpe: {report_metrics['sharpe_ratio']:.2f} | "
            f"Win Rate: {report_metrics['win_rate'] * 100:.2f}% | "
            f"Max Drawdown: {report_metrics['max_drawdown'] * 100:.2f}%")

        session_id = f"backtest_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        trades_df.to_csv(os.path.join(output_dir, f"{session_id}_trades.csv"), index=False)
        print(f"[DONE] Backtest complete. Trades saved.")
    else:
        print("[WARNING] No trades executed. No report generated.")

    return trades_df

# ------------------------------------------------------------------------------
# CLI ENTRY POINT
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", type=str, required=True, help="Study name (directory)")
    args = parser.parse_args()

    study_dir = os.path.join("optuna_experiments", args.study_name)
    model_path = os.path.join(study_dir, "pruned_model.txt")
    feature_path = os.path.join(study_dir, "top_features.txt")
    thresholds_path = os.path.join(study_dir, "best_trial.txt")

    # Load thresholds
    threshold_buy = threshold_sell = 0.6  # Defaults
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r") as f:
            for line in f:
                if "threshold_buy" in line:
                    threshold_buy = float(line.split(":")[1].strip())
                elif "threshold_sell" in line:
                    threshold_sell = float(line.split(":")[1].strip())

    run_backtest(
        model_path=model_path,
        feature_path=feature_path,
        threshold_buy=threshold_buy,
        threshold_sell=threshold_sell,
        output_dir=None  # Dynamically determined
    )
