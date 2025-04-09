#!/usr/bin/env python3

"""
Trading Engine — Unified for Paper / Testnet / Live
Executes trades based on model predictions with dynamic thresholds and robust handling.
"""

import os
import time
import math
import json
import logging
import yaml
import ccxt
import pandas as pd
import argparse
from datetime import datetime, timezone
from trading_indicators import prepare_live_features
from ccxt.base.errors import NetworkError, ExchangeError, DDoSProtection, RequestTimeout

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

ENV = config.get("env", "paper")
SYMBOL = config['model']['target_symbol']
RISK_PCT = config['trading']['risk_percentage']
TP_PCT = config['trading'].get('take_profit_pct', 3.0)
SL_PCT = config['trading'].get('stop_loss_pct', 1.5)

API_KEY, API_SECRET = None, None
if ENV == "live":
    API_KEY = config['binance']['mainnet']['api_key']
    API_SECRET = config['binance']['mainnet']['api_secret']
elif ENV == "testnet":
    API_KEY = config['binance']['testnet']['api_key']
    API_SECRET = config['binance']['testnet']['api_secret']

exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
})

if ENV == "testnet":
    exchange.set_sandbox_mode(True)
    print("[CONFIG] Testnet mode active.")
elif ENV == "live":
    print("[CONFIG] Live trading mode active.")
else:
    exchange = None
    print("[CONFIG] Paper trading mode active (no real exchange).")

# ------------------------------------------------------------------------------
# INITIALIZE VIRTUAL PORTFOLIO (Paper mode)
# ------------------------------------------------------------------------------

portfolio = {
    "USDC": config['trading'].get('initial_usdc', 1000),
    "BASE": config['trading'].get('initial_sol', 0)
}

position_state = {
    "in_position": False,
    "entry_price": None,
    "entry_time": None,
    "side": None,
    "tp_price": None,
    "sl_price": None
}

# ------------------------------------------------------------------------------
# LOGGER SETUP
# ------------------------------------------------------------------------------

logger = logging.getLogger("trade_logger")
logger.setLevel(logging.INFO)
os.makedirs("logs", exist_ok=True)

# ------------------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------------------

def setup_logger(run_id):
    fh = logging.FileHandler(f"logs/{run_id}_trading.log")
    formatter = logging.Formatter('%(asctime)s UTC — %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def log_trade(event_type, message, context=None):
    entry = f"[{event_type.upper()}] {message}"
    if context:
        entry += f" | Context: {json.dumps(context, default=str)}"
    logger.info(entry)

def calculate_position_size(conf, threshold, volatility, base_risk=0.10, max_risk=0.30):
    conf_margin = max(conf - threshold, 0)
    confidence_factor = 1 / (1 + math.exp(-10 * conf_margin))
    volatility_factor = min(1.0, 1.5 / max(volatility, 1e-6))
    scaled_risk = base_risk + (max_risk - base_risk) * confidence_factor * volatility_factor
    return round(min(scaled_risk, max_risk), 4)

def get_portfolio():
    if ENV == "paper":
        return portfolio
    try:
        balance = exchange.fetch_balance()
        base_asset = SYMBOL.split('/')[0]
        return {
            "USDC": balance['total'].get('USDC', 0),
            "BASE": balance['total'].get(base_asset, 0)
        }
    except Exception as e:
        print(f"[ERROR] Portfolio fetch error: {e}")
        return {"USDC": 0, "BASE": 0}

def get_current_price(run_id):
    if ENV == "paper":
        df = pd.read_csv(f"csv/{run_id}/predictions.csv")
        latest_price = df[df["symbol"] == SYMBOL].sort_values(by="timestamp").iloc[-1]["price"]
        return latest_price

    max_retries = 5
    for attempt in range(max_retries):
        try:
            ticker = exchange.fetch_ticker(SYMBOL)
            return ticker['last']
        except (NetworkError, ExchangeError, DDoSProtection, RequestTimeout) as e:
            wait = 2 ** attempt
            print(f"[WARNING] Price fetch error: {e} — Retrying in {wait} sec...")
            time.sleep(wait)
        except Exception as e:
            print(f"[ERROR] Critical price fetch failure: {e}")
            break
    raise RuntimeError("Failed to fetch current price after retries.")

def place_order(side: str, amount: float):
    MIN_AMOUNT = 0.00001
    if amount < MIN_AMOUNT:
        print(f"[WARNING] Order size {amount} below minimum precision.")
        return None

    if ENV == "paper":
        print(f"[PAPER] {side.upper()} {amount} units")
        return {"status": "paper_order", "side": side, "amount": amount}

    for attempt in range(5):
        try:
            if side == "buy":
                return exchange.create_market_buy_order(SYMBOL, amount)
            elif side == "sell":
                return exchange.create_market_sell_order(SYMBOL, amount)
        except (NetworkError, ExchangeError, DDoSProtection, RequestTimeout) as e:
            wait = 2 ** attempt
            print(f"[WARNING] Order {side.upper()} failed: {e} — Retrying in {wait} sec...")
            time.sleep(wait)
        except Exception as e:
            print(f"[ERROR] Unexpected order error: {e}")
            break
    print(f"[ERROR] Order {side.upper()} failed after multiple attempts.")
    return None

def record_trade(run_id, entry_time, side, entry_price, size, pnl=0, exit_time=None, exit_price=None):
    trades_path = f"csv/{run_id}/paper_trades.csv"
    trade_data = {
        "entry_time": entry_time,
        "exit_time": exit_time or entry_time,
        "side": side,
        "entry_price": entry_price,
        "exit_price": exit_price or entry_price,
        "size": size,
        "pnl": pnl,
        "trade_amount_usdc": size * entry_price
    }
    df_trade = pd.DataFrame([trade_data])
    mode = 'a' if os.path.exists(trades_path) else 'w'
    df_trade.to_csv(trades_path, mode=mode, header=not os.path.exists(trades_path), index=False)

def dynamic_threshold(df_predictions, base_threshold=0.6, window=100):
    if df_predictions.empty or len(df_predictions) < window:
        return base_threshold
    recent_conf = df_predictions['prediction_prob'].tail(window)
    mean_conf = recent_conf.mean()
    std_conf = recent_conf.std()
    return max(min(mean_conf + (0.5 * std_conf), 0.9), 0.3)

def adjust_threshold_by_volatility(atr_value):
    if atr_value > 300:
        return 0.1
    elif atr_value < 100:
        return -0.1
    return 0.0

def get_latest_prediction(run_id):
    pred_path = f"csv/{run_id}/predictions.csv"
    if not os.path.exists(pred_path):
        return None, None, None
    try:
        df = pd.read_csv(pred_path)
        df = df[df["symbol"] == SYMBOL]
        if df.empty:
            return None, None, None
        row = df.sort_values(by="timestamp", ascending=False).iloc[0]
        return pd.to_datetime(row["timestamp"]), int(row["predicted_class"]), float(row["prediction_prob"])
    except Exception as e:
        print(f"[ERROR] Prediction read failure: {e}")
        return None, None, None

# ------------------------------------------------------------------------------
# TRADE LOGIC
# ------------------------------------------------------------------------------

def trade_if_signal(run_id, threshold_buy, threshold_sell, signal, conf, pred_time):
    price = get_current_price(run_id)
    portfolio = get_portfolio()
    _, _, df_live = prepare_live_features(SYMBOL, horizon=20, return_df=True)
    atr = df_live["atr_14"].iloc[-1]

    predictions_path = f"csv/{run_id}/predictions.csv"
    df_predictions = pd.read_csv(predictions_path) if os.path.exists(predictions_path) else pd.DataFrame()

    dynamic_buy = dynamic_threshold(df_predictions, threshold_buy) + adjust_threshold_by_volatility(atr)
    dynamic_sell = dynamic_threshold(df_predictions, threshold_sell) + adjust_threshold_by_volatility(atr)

    dynamic_buy = min(max(dynamic_buy, 0.3), 0.9)
    dynamic_sell = min(max(dynamic_sell, 0.05), 0.9)

    log_trade("threshold", "Dynamic thresholds", {
        "adjusted_buy": dynamic_buy,
        "adjusted_sell": dynamic_sell,
        "ATR": atr,
        "signal": signal,
        "confidence": conf,
        "in_position": position_state['in_position']
    })

    base_asset = SYMBOL.split('/')[0]

    if signal == 2 and conf >= dynamic_buy and not position_state["in_position"]:
        usdc_balance = portfolio["USDC"]
        trade_amount = usdc_balance * RISK_PCT
        size = trade_amount / price
        response = place_order("buy", round(size, 6))
        if response:
            position_state.update({
                "in_position": True,
                "entry_price": price,
                "entry_time": pred_time,
                "side": "long",
                "tp_price": price + 2 * atr,
                "sl_price": price - 1.5 * atr
            })
            record_trade(run_id, pred_time, "long", price, size)
            log_trade("entry", f"Long {size:.4f} {base_asset} @ {price:.2f}")
            if ENV == "paper":
                portfolio["USDC"] -= trade_amount
                portfolio["BASE"] += size

    elif signal == 0 and conf >= dynamic_sell and not position_state["in_position"]:
        base_balance = portfolio["BASE"]
        if base_balance <= 0:
            print(f"[INFO] No base asset to sell. Skipping.")
            return
        response = place_order("sell", round(base_balance, 6))
        if response:
            position_state.update({
                "in_position": False,
                "entry_price": None,
                "entry_time": None,
                "side": None,
                "tp_price": None,
                "sl_price": None
            })
            log_trade("exit", f"Short {base_balance:.4f} {base_asset} @ {price:.2f}")
            if ENV == "paper":
                portfolio["USDC"] += base_balance * price
                portfolio["BASE"] = 0

def check_exit_conditions(run_id):
    if not position_state["in_position"]:
        return

    price = get_current_price(run_id)
    side = position_state["side"]
    base_asset = SYMBOL.split('/')[0]

    tp_hit = price >= position_state["tp_price"]
    sl_hit = price <= position_state["sl_price"]

    if tp_hit or sl_hit:
        amount = portfolio["BASE"]
        place_order("sell", round(amount, 6))
        log_trade("exit", f"{side.upper()} exit at {price:.2f}")

        if ENV == "paper":
            portfolio["USDC"] += amount * price
            portfolio["BASE"] = 0

        record_trade(
            run_id,
            position_state["entry_time"],
            side,
            position_state["entry_price"],
            amount,
            exit_time=datetime.now(timezone.utc),
            exit_price=price,
            pnl=(price - position_state["entry_price"]) * amount
        )
        position_state["in_position"] = False

# ------------------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--study_name", type=str, required=True)
    args = parser.parse_args()

    run_id = args.run_id
    setup_logger(run_id)

    thresholds_path = f"optuna_experiments/{args.study_name}/best_trial.txt"
    threshold_buy, threshold_sell = 0.6, 0.6

    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r") as f:
            for line in f:
                if "threshold_buy" in line:
                    threshold_buy = float(line.split(":")[1].strip())
                elif "threshold_sell" in line:
                    threshold_sell = float(line.split(":")[1].strip())

    print(f"=== Trading Engine Initialized | {ENV.upper()} MODE ===")
    print(f"Thresholds: Buy={threshold_buy}, Sell={threshold_sell}")

    last_timestamp = None
    try:
        while True:
            check_exit_conditions(run_id)
            timestamp, signal, conf = get_latest_prediction(run_id)
            if timestamp and (last_timestamp is None or timestamp > last_timestamp):
                trade_if_signal(run_id, threshold_buy, threshold_sell, signal, conf, timestamp)
                last_timestamp = timestamp
            time.sleep(1)
    except KeyboardInterrupt:
        print("[INFO] Terminated by user.")
    except Exception as e:
        print(f"[FATAL] Unexpected error: {e}")
