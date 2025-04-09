#!/usr/bin/env python3

"""
Inference Engine — Real-time and Batch prediction for trading engine.
Loads the trained model and outputs predictions continuously or on historical data.
"""

import os
import time
import yaml
import pandas as pd
import argparse
from datetime import datetime, timezone
import lightgbm as lgb
from trading_indicators import prepare_live_features, prepare_training_data

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

SYMBOL = config['model']['target_symbol']

# ------------------------------------------------------------------------------
# INFERENCE FUNCTION
# ------------------------------------------------------------------------------

def run_inference(run_id, study_name, mode):
    print(f"[INFO] Initializing Inference Engine — Mode: {mode.upper()}")

    study_dir = os.path.join("optuna_experiments", study_name)
    model_path = os.path.join(study_dir, "pruned_model.txt")
    feature_list_path = os.path.join(study_dir, "top_features.txt")
    thresholds_path = os.path.join(study_dir, "best_trial.txt")

    # Validate paths
    for path in [model_path, feature_list_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file missing: {path}")

    # Load model
    model = lgb.Booster(model_file=model_path)

    # Load features
    with open(feature_list_path, "r") as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    if not feature_cols:
        raise ValueError("Feature list is empty — check top_features.txt")

    os.makedirs(f"csv/{run_id}", exist_ok=True)
    predictions_path = f"csv/{run_id}/predictions.csv"
    metadata_path = f"csv/{run_id}/metadata.txt"

    # Write metadata
    with open(metadata_path, "w") as f:
        f.write(f"run_id: {run_id}\n")
        f.write(f"symbol: {SYMBOL}\n")
        f.write(f"features: {len(feature_cols)}\n")
        f.write(f"timestamp: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"mode: {mode}\n")

    print(f"[INFO] Model and features loaded successfully.")
    print(f"[INFO] Feature count: {len(feature_cols)}")

    # Load thresholds (optional future extension)
    threshold_buy = threshold_sell = 0.6
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r") as f:
            for line in f:
                if "threshold_buy" in line:
                    threshold_buy = float(line.split(":")[1].strip())
                elif "threshold_sell" in line:
                    threshold_sell = float(line.split(":")[1].strip())

    if mode == "batch":
        print(f"[INFO] Starting batch inference...")
        df_full, _, _ = prepare_training_data(
            symbol=SYMBOL,
            threshold_buy=threshold_buy,
            threshold_sell=threshold_sell,
            horizon=20,
            buy_mult=2.0,
            sell_mult=2.0
        )

        feature_cols_final = [col for col in df_full.columns if col in feature_cols]

        if not feature_cols_final:
            raise ValueError("No matching features found for inference — check feature preparation pipeline.")

        X_batch = df_full[feature_cols_final].values
        predictions = []

        for idx, row in df_full.iterrows():
            features = X_batch[idx].reshape(1, -1)
            y_proba = model.predict(features)[0]
            y_pred = int(y_proba.argmax())
            conf = float(y_proba[y_pred])
            timestamp = row["timestamp"]
            price = row.get("close_lag1", row.get("close", None))

            predictions.append({
                "timestamp": timestamp,
                "symbol": SYMBOL,
                "predicted_class": y_pred,
                "prediction_prob": conf,
                "price": price
            })

        df_pred = pd.DataFrame(predictions)

        if len(df_pred) < 100:
            print(f"[WARNING] Batch inference generated only {len(df_pred)} rows — check data quality.")

        df_pred.to_csv(predictions_path, index=False)
        print(f"[INFO] Batch predictions saved: {predictions_path}")
        print(f"[INFO] Total predictions: {len(df_pred)} ✅")
        return

    # -------------------------------------------
    # REALTIME MODE
    # -------------------------------------------

    print(f"[INFO] Starting realtime inference loop...")
    last_timestamp = None
    heartbeat_counter = 0

    try:
        while True:
            try:
                X_live, _, df_live = prepare_live_features(SYMBOL, horizon=20, return_df=True)

                if X_live.shape[1] != model.num_feature():
                    raise ValueError(f"Feature mismatch: model expects {model.num_feature()}, got {X_live.shape[1]}")

                y_proba = model.predict(X_live)[0]
                y_pred = int(y_proba.argmax())
                conf = float(y_proba[y_pred])
                timestamp = df_live["timestamp"].iloc[-1]
                price = df_live["close_lag1"].iloc[-1]

                if last_timestamp is None or timestamp > last_timestamp:
                    row = {
                        "timestamp": timestamp,
                        "symbol": SYMBOL,
                        "predicted_class": y_pred,
                        "prediction_prob": conf,
                        "price": price
                    }
                    df_row = pd.DataFrame([row])
                    mode_write = 'a' if os.path.exists(predictions_path) else 'w'
                    df_row.to_csv(predictions_path, mode=mode_write, header=not os.path.exists(predictions_path), index=False)

                    print(f"[PREDICTION] {timestamp} — Class: {y_pred} | Confidence: {conf:.4f} | Price: {price}")
                    last_timestamp = timestamp

                heartbeat_counter += 1
                if heartbeat_counter >= 30:
                    print(f"[HEARTBEAT] Inference engine alive @ {datetime.now(timezone.utc).isoformat()}")
                    heartbeat_counter = 0

            except Exception as e:
                print(f"[ERROR] Inference loop error: {e}")
                time.sleep(5)

            time.sleep(1)

    except KeyboardInterrupt:
        print("[INFO] Inference engine terminated by user.")

# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="Unique session identifier")
    parser.add_argument("--study_name", type=str, required=True, help="Study directory name")
    parser.add_argument("--mode", type=str, choices=["realtime", "batch"], default="realtime", help="Mode: realtime or batch")
    args = parser.parse_args()

    run_inference(run_id=args.run_id, study_name=args.study_name, mode=args.mode)
