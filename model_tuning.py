#!/usr/bin/env python3

"""
Model Tuning — Hyperparameter optimization using Optuna for LightGBM.
Includes integrated periodic backtesting and performance-driven optimization.
"""

import os
import optuna
import lightgbm as lgb
import yaml
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import optuna.visualization.matplotlib as optuna_vis
from trading_indicators import prepare_training_data
from backtest_engine import run_backtest
from reporting import generate_report_from_df

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

SYMBOL = config['model']['target_symbol']

# Global state
STUDY_DIR = None
global_trial_counter = 0

# ------------------------------------------------------------------------------
# FEATURE IMPORTANCE PRUNING FUNCTION
# ------------------------------------------------------------------------------

def prune_features(model_path, feature_list_path, output_feature_path, importance_threshold=0.95):
    import pandas as pd
    import lightgbm as lgb

    print("[INFO] Starting feature importance pruning...")

    # Load model
    model = lgb.Booster(model_file=model_path)

    # Extract importances
    importances = model.feature_importance(importance_type='gain')
    features = model.feature_name()

    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    # Calculate cumulative importance
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()

    # Select features under threshold
    selected_features = importance_df[importance_df['cumulative_importance'] <= importance_threshold]['feature'].tolist()

    if not selected_features:
        raise ValueError("[ERROR] No features selected after pruning. Lower your threshold.")

    # Save selected features
    with open(output_feature_path, "w") as f:
        for feat in selected_features:
            f.write(f"{feat}\n")

    print(f"[INFO] Pruning complete. {len(selected_features)} features retained out of {len(features)}.")
    print(f"[INFO] Pruned feature list saved to {output_feature_path}")

# ------------------------------------------------------------------------------
# OBJECTIVE FUNCTION FOR OPTUNA
# ------------------------------------------------------------------------------

def objective(trial):
    global global_trial_counter
    global_trial_counter += 1
    elapsed_seconds = int(time.time() - start_time)
    elapsed_minutes = elapsed_seconds // 60
    elapsed_secs = elapsed_seconds % 60

    # Estimate ETA to next checkpoint
    trials_to_next_checkpoint = ((global_trial_counter // BACKTEST_INTERVAL) + 1) * BACKTEST_INTERVAL - global_trial_counter
    avg_time_per_trial = elapsed_seconds / max(global_trial_counter, 1)
    eta_seconds = int(trials_to_next_checkpoint * avg_time_per_trial)
    eta_minutes = eta_seconds // 60
    eta_secs = eta_seconds % 60

    print("-" * 60)
    print(f"[INFO] Trial {global_trial_counter}")
    print(f"[TIME] Elapsed: {elapsed_minutes} min {elapsed_secs} sec")
    print(f"[CHECKPOINT] Next checkpoint at trial {((global_trial_counter // BACKTEST_INTERVAL) + 1) * BACKTEST_INTERVAL} (ETA: {eta_minutes} min {eta_secs} sec)")
    print("-" * 60)

    # Hyperparameter space — dynamic, correct
    threshold_buy = trial.suggest_float("threshold_buy", 0.6, 0.9)
    threshold_sell = trial.suggest_float("threshold_sell", 0.1, 0.4)
    horizon = trial.suggest_int("horizon", 1500, 3000)
    buy_mult = trial.suggest_float("buy_mult", 1.5, 3.0)
    sell_mult = trial.suggest_float("sell_mult", 1.5, 3.0)

    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2)
    num_leaves = trial.suggest_int("num_leaves", 31, 300)
    feature_fraction = trial.suggest_float("feature_fraction", 0.6, 1.0)
    bagging_fraction = trial.suggest_float("bagging_fraction", 0.6, 1.0)
    bagging_freq = trial.suggest_int("bagging_freq", 1, 10)

    # Prepare data
    df, target, feature_cols = prepare_training_data(
        symbol=SYMBOL,
        threshold_buy=threshold_buy,
        threshold_sell=threshold_sell,
        horizon=horizon,
        buy_mult=buy_mult,
        sell_mult=sell_mult
    )

    class_weights = {0: 2.0, 1: 1.0, 2: 2.0}
    weights = target.map(class_weights).values

    numeric_feature_cols = df[feature_cols].select_dtypes(include=['number']).columns.tolist()
    dtrain = lgb.Dataset(df[numeric_feature_cols], label=target, weight=weights)

    # Model parameters (dynamic)
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'seed': 42
    }

    # Train model
    model = lgb.train(params, dtrain, valid_sets=[dtrain], callbacks=[lgb.log_evaluation(period=0)])

    # Save model and features
    model_path = os.path.join(STUDY_DIR, "pruned_model.txt")
    model.save_model(model_path)

    feature_path = os.path.join(STUDY_DIR, "top_features.txt")
    with open(feature_path, "w") as f:
        for feat in numeric_feature_cols:
            f.write(f"{feat}\n")

    # Save trial hyperparameters
    best_trial_path = os.path.join(STUDY_DIR, "best_trial.txt")
    with open(best_trial_path, "w") as f:
        f.write(f"threshold_buy: {threshold_buy}\n")
        f.write(f"threshold_sell: {threshold_sell}\n")

    # In-training evaluation
    preds = model.predict(df[numeric_feature_cols])
    df["pred_class"] = preds.argmax(axis=1)
    accuracy = (df["pred_class"] == df["target"]).mean()

    # Periodic checkpoint backtest
    metrics = None
    if global_trial_counter % BACKTEST_INTERVAL == 0:
        print(f"\n[CHECKPOINT] Running backtest at trial {global_trial_counter}...")
        trades_df = run_backtest(
            model_path=model_path,
            feature_path=feature_path,
            threshold_buy=threshold_buy,
            threshold_sell=threshold_sell,
            output_dir=os.path.join(STUDY_DIR, f"checkpoint_trial_{global_trial_counter}")
        )

        if not trades_df.empty:
            metrics = generate_report_from_df(trades_df, save_reports=True)
            print(f"[CHECKPOINT METRICS] PnL: {metrics['total_pnl']:.2f} | "
                  f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
                  f"Win Rate: {metrics['win_rate']*100:.2f}% | "
                  f"Max Drawdown: {metrics['max_drawdown']*100:.2f}% | "
                  f"Profit Factor: {metrics['profit_factor']:.2f}")
        else:
            print("[WARNING] No trades generated during checkpoint backtest.")

    # Composite score
    if metrics:
        fee_penalty = metrics['total_fees'] / max(abs(metrics['gross_pnl']), 1e-8)
        score = (
            0.4 * metrics['sharpe_ratio'] +
            0.3 * metrics['profit_factor'] +
            0.2 * metrics['win_rate'] -
            0.1 * abs(metrics['max_drawdown']) -
            0.2 * fee_penalty
        )
    else:
        score = accuracy  # Fallback

    return score

# ------------------------------------------------------------------------------
# MAIN TUNING FUNCTION
# ------------------------------------------------------------------------------

def run_tuning(study_name: str, n_trials: int = 50):
    global STUDY_DIR
    global BACKTEST_INTERVAL
    global start_time

    start_time = time.time()
    BACKTEST_INTERVAL = max(1, n_trials // 4)
    print(f"[START] Optuna tuning: {study_name} | Trials: {n_trials}")
    print(f"[INFO] Backtesting will occur every {BACKTEST_INTERVAL} trials (~quarter intervals).")
    STUDY_DIR = os.path.join("optuna_experiments", study_name)
    os.makedirs(STUDY_DIR, exist_ok=True)

    storage_path = f"sqlite:///{STUDY_DIR}/optuna_study.db"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True
    )

    study.optimize(objective, n_trials=n_trials)

    # After study complete, prune features based on final model
    model_path = os.path.join(STUDY_DIR, "pruned_model.txt")
    feature_list_path = os.path.join(STUDY_DIR, "top_features.txt")
    pruned_feature_list_path = os.path.join(STUDY_DIR, "top_features_pruned.txt")

    prune_features(model_path, feature_list_path, pruned_feature_list_path, importance_threshold=0.95)

    # Export Optuna trials
    df_trials = study.trials_dataframe()
    df_trials.to_csv(os.path.join(STUDY_DIR, "optuna_trials.csv"), index=False)
    print(f"[EXPORT] Trials exported: {os.path.join(STUDY_DIR, 'optuna_trials.csv')}")

    # Export optimization history
    fig = optuna_vis.plot_optimization_history(study).figure
    fig.savefig(os.path.join(STUDY_DIR, "optuna_history.png"))
    plt.close(fig)
    print(f"[EXPORT] Optimization history plot saved.")

    print(f"[BEST VALUE] Composite Score: {study.best_value:.5f}")
    print(f"[BEST PARAMS] {study.best_params}")

# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", type=str, required=True, help="Study name")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()

    run_tuning(study_name=args.study_name, n_trials=args.n_trials)
