# ALGORITHMIC TRADING SYSTEM — CONSOLIDATED PIPELINE v2.1

## Overview

Modular, real-time algorithmic trading system featuring:
- ✅ Optuna-tuned LightGBM models with feature pruning
- ✅ Feature store creation from live Binance OHLCV data (BTC/USDC + BTC/USDT)
- ✅ Multi-timeframe technical indicator engineering
- ✅ Live inference & signal-based trading with volatility-adjusted thresholds
- ✅ Dynamic SL/TP execution (ATR-based) with paper/testnet/live modes
- ✅ PostgreSQL-compatible data structure (via Parquet and streaming)
- ✅ Full logging, CI/CD-ready JSON metrics, Streamlit dashboard

---

## Project Structure

| Module                    | Description                                           |
|---------------------------|-------------------------------------------------------|
| `build_feature_store.py`  | Historical + realtime OHLCV fetching, feature computation |
| `trading_indicators.py`   | Technical indicators, multi-timeframe features, label generation |
| `model_tuning.py`         | Optuna LightGBM training with checkpoint backtesting |
| `backtest_engine.py`      | Historical strategy backtesting using trained model |
| `inference.py`            | Batch or live prediction from saved model |
| `trading_engine.py`       | TP/SL dynamic trading logic for paper/testnet/live |
| `start_session.py`        | Orchestration pipeline: data → inference → trading |
| `reporting.py`            | Performance metrics and visualization generation |
| `dashboard.py`            | Live Streamlit dashboard for model, trades, metrics |

---

## Installation

```bash
git clone <repo_url>
cd <repo>
pip install -r requirements.txt
```

Ensure Binance API credentials and environment settings are configured in `config.yaml`.

---

## Configuration (`config.yaml`)

Define:
- Binance API keys (testnet or live)
- Target symbol (e.g. BTC/USDC)
- Environment: `paper`, `testnet`, `live`
- Trading parameters: initial capital, TP/SL %, risk %, etc.

---

## Workflow Overview

### 1. Build Feature Store (Backfill + Indicators)

```bash
python build_feature_store.py
```

Or run in continuous realtime mode:

```bash
python build_feature_store.py --realtime
```

---

### 2. Tune Model (LightGBM + Optuna + Checkpoint Backtests)

```bash
python model_tuning.py --study_name BTC_1 --n_trials 50
```

Artifacts saved to `optuna_experiments/BTC_1/`:
- `pruned_model.txt`, `top_features.txt`, `best_trial.txt`
- Checkpoint backtests + visualizations
- Optimization plots, trial logs, and metrics

---

### 3. Run Manual Backtest (Evaluate Offline)

```bash
python backtest_engine.py --study_name BTC_1
```

Outputs:
- `csv/backtest_BTC_1_<timestamp>/backtest_report.csv`, `report_metrics.json`, trades
- Visuals: equity, drawdown, Sharpe, gross vs net

---

### 4. Full Live/Paper/Testnet Session (Auto-Ingestion + Inference + Trading)

```bash
python start_session.py --study_name BTC_1
```

Processes launched:
- Realtime feature ingestion
- Inference loop (predictions logged)
- Signal-driven trading engine

Logs and CSVs saved under `csv/<run_id>/`, `logs/`, and `report_metrics.json`.

---

### 5. Launch Monitoring Dashboard

```bash
streamlit run dashboard.py
```

- Optuna study visualization
- Live trades, PnL, equity curves
- Backtest metrics & checkpoint comparisons

---

## Command Reference

| Action                         | Command
|--------------------------------|-------------------------------------------|
| Build Feature Store (Backfill) | `python build_feature_store.py` |
| Build Feature Store (Realtime) | `python build_feature_store.py --realtime` |
| Model Tuning                   | `python model_tuning.py --study_name BTC_1 --n_trials 50` |
| Manual Backtest                | `python backtest_engine.py --study_name BTC_1` |
| Inference (Batch)              | `python inference.py --run_id test_run --study_name BTC_1 --mode batch` |
| Inference (Live)               | `python inference.py --run_id live_run --study_name BTC_1 --mode realtime` |
| Trading Engine                 | `python trading_engine.py --run_id live_run --study_name BTC_1` |
| Full Session (Pipeline)        | `python start_session.py --study_name BTC_1` |
| Monitoring Dashboard           | `streamlit run dashboard.py` |
| Performance Report             | `python reporting.py --trades <path_to_trades.csv>` |

---

## Notes

- All data and logs are session-tracked via unique `run_id`
- In `paper` mode, trades are simulated; no keys needed
- In `testnet`, real API calls are sandboxed
- In `live`, real market orders are executed — use with caution
- Feature store is saved as `data/feature_store.parquet`

---

## License

**Private Research Use Only.**  
For access or licensing, contact the system maintainer.