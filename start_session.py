#!/usr/bin/env python3

"""
Start Session — Full Pipeline Orchestrator
Executes: Backfill → Realtime Ingestion → Inference → Trading Engine
"""

import subprocess
import sys
import os
import time
import yaml
from datetime import datetime, timezone

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

ENV = config.get("env", "paper")
SYMBOL = config['model']['target_symbol']

# ------------------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------------------

def log(msg):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {msg}")

def launch_process(command, name, log_file=None):
    log(f"[LAUNCH] Starting {name}...")
    stdout = stderr = None
    if log_file:
        stdout = open(log_file, 'a')
        stderr = stdout
        log(f"[LOGGING] {name} output redirected to {log_file}")
    process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
    return process

def wait_for_file(file_path, check_interval=0.5, timeout=120):
    log(f"[WAIT] Waiting for file: {file_path}")
    waited = 0
    while not (os.path.exists(file_path) and os.path.getsize(file_path) > 0):
        time.sleep(check_interval)
        waited += check_interval
        if waited >= timeout:
            raise TimeoutError(f"Timeout waiting for {file_path}")
    log(f"[FOUND] File available: {file_path}")

# ------------------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", type=str, required=True, help="Study directory under optuna_experiments/")
    args = parser.parse_args()

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{ENV}_session_{timestamp_str}"

    log(f"[SESSION] Starting full pipeline with run_id={run_id}")

    try:
        # Step 1: Backfill historical data
        log("[STEP 1] Backfilling historical data...")
        backfill_cmd = [
            sys.executable, "build_feature_store.py",
            ]
        subprocess.run(backfill_cmd, check=True)
        log("[STEP 1] Backfill completed successfully.")

        # Step 2: Start realtime ingestion
        ingestion_log = f"logs/{run_id}_ingestion.log"
        ingestion_proc = launch_process([
            sys.executable, "build_feature_store.py",
            "--realtime"
        ], name="Realtime Feature Store Update", log_file=ingestion_log)

        # Step 3: Start inference engine
        inference_log = f"logs/{run_id}_inference.log"
        inference_proc = launch_process([
            sys.executable, "inference.py",
            "--run_id", run_id,
            "--study_name", args.study_name
        ], name="Inference Engine", log_file=inference_log)

        # Step 4: Wait for inference metadata to confirm readiness
        metadata_path = f"csv/{run_id}/metadata.txt"
        wait_for_file(metadata_path)

        # Step 5: Start trading engine
        trading_log = f"logs/{run_id}_trading.log"
        trading_proc = launch_process([
            sys.executable, "trading_engine.py",
            "--run_id", run_id,
            "--study_name", args.study_name
        ], name="Trading Engine", log_file=trading_log)

        log("[SESSION] All processes launched successfully. Monitoring...")

        # Wait for subprocesses to finish
        while True:
            if ingestion_proc.poll() is not None:
                log("[WARNING] Realtime ingestion process terminated.")
                break
            if inference_proc.poll() is not None:
                log("[WARNING] Inference process terminated.")
                break
            if trading_proc.poll() is not None:
                log("[WARNING] Trading engine process terminated.")
                break
            time.sleep(5)

    except Exception as e:
        log(f"[ERROR] Session failed: {e}")
    finally:
        # Ensure all processes are terminated cleanly
        for proc, name in [(ingestion_proc, "Ingestion"), (inference_proc, "Inference"), (trading_proc, "Trading")]:
            if proc and proc.poll() is None:
                log(f"[CLEANUP] Terminating {name} process...")
                proc.terminate()
                proc.wait()

        log("[SESSION] Pipeline session completed and cleaned up.")

# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    main()
