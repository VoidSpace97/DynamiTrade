#!/usr/bin/env python3

"""
Dashboard — Visual Monitor for ML Spot Trader
Visualizes Optuna Experiments, Manual Backtests, and Training Checkpoints.
"""

import os
import streamlit as st
import pandas as pd
import json
import optuna
from datetime import datetime

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

REFRESH_INTERVAL = 60  # seconds
DATA_DIR = "csv"
OPTUNA_DIR = "optuna_experiments"

# ------------------------------------------------------------------------------
# PAGE SETTINGS
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="ML Spot Dashboard",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("🧩 ML Spot Trading Dashboard")
st.markdown("Monitor Optuna experiments, backtests, and performance reports in real time.")
st.caption(f"Auto-refresh every {REFRESH_INTERVAL} seconds.")

# ------------------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------------------

def list_optuna_experiments():
    return [d for d in os.listdir(OPTUNA_DIR) if os.path.isdir(os.path.join(OPTUNA_DIR, d))]

def list_backtest_sessions():
    return [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

def list_checkpoints(study_name):
    study_dir = os.path.join(OPTUNA_DIR, study_name)
    if not os.path.exists(study_dir):
        return []
    return [d for d in os.listdir(study_dir) if d.startswith("checkpoint_") and os.path.isdir(os.path.join(study_dir, d))]

def load_optuna_study(study_name):
    db_path = os.path.join(OPTUNA_DIR, study_name, "optuna_study.db")
    if os.path.exists(db_path):
        storage_url = f"sqlite:///{db_path}"
        try:
            return optuna.load_study(study_name=study_name, storage=storage_url)
        except Exception as e:
            st.error(f"Error loading Optuna study: {e}")
    return None

def load_report_metrics(folder_path):
    path = os.path.join(folder_path, "report_metrics.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def load_trades(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith("_trades.csv"):
            return pd.read_csv(os.path.join(folder_path, file))
    return pd.DataFrame()

def load_equity_curve(folder_path):
    path = os.path.join(folder_path, "equity_curve.png")
    if os.path.exists(path):
        return path
    return None

def load_optuna_history_image(study_name):
    path = os.path.join(OPTUNA_DIR, study_name, "optuna_history.png")
    if os.path.exists(path):
        return path
    return None

# ------------------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Balance & Equity",
    "📈 Trade Analysis",
    "🔮 Model Insights",
    "⚙️ Features & Alerts",
    "🧩 Experiments & Backtests"
])

# ------------------------------------------------------------------------------
# TAB: Experiments & Backtests
# ------------------------------------------------------------------------------

with tab5:
    st.header("🧩 Experiments & Backtests Overview")

    # ---------------- Optuna Experiments ----------------
    st.subheader("⚙️ Optuna Experiments")
    optuna_experiments = list_optuna_experiments()

    if optuna_experiments:
        selected_study = st.selectbox("Select Optuna Study", sorted(optuna_experiments, reverse=True))
        study = load_optuna_study(selected_study)

        if study:
            df_trials = study.trials_dataframe()
            if not df_trials.empty:
                st.markdown(f"#### Trials for `{selected_study}`")
                st.dataframe(df_trials)

                best_trial = study.best_trial
                st.markdown("**Best Trial Parameters:**")
                st.json(best_trial.params)

                st.markdown("**Best Trial Metrics:**")
                st.json({
                    "Value": best_trial.value,
                    "Number": best_trial.number,
                    "User Attributes": best_trial.user_attrs
                })

                if any(t.intermediate_values for t in study.trials):
                    st.markdown("**Intermediate Values:**")
                    for trial in study.trials:
                        if trial.intermediate_values:
                            st.write(f"Trial {trial.number}: {trial.intermediate_values}")

            image_path = load_optuna_history_image(selected_study)
            if image_path:
                st.image(image_path, caption="Optimization History", use_column_width=True)

        else:
            st.warning("No Optuna study database found for this study.")

    else:
        st.warning("No Optuna experiments found.")

    st.markdown("---")

    # ---------------- Checkpoint Backtests (Training Time) ----------------
    st.subheader("🧩 Checkpoint Backtests (During Training)")

    if optuna_experiments:
        checkpoints = list_checkpoints(selected_study)

        if checkpoints:
            selected_checkpoint = st.selectbox("Select Checkpoint Backtest", sorted(checkpoints, reverse=True))
            checkpoint_path = os.path.join(OPTUNA_DIR, selected_study, selected_checkpoint)

            checkpoint_report = load_report_metrics(checkpoint_path)
            checkpoint_trades = load_trades(checkpoint_path)
            checkpoint_equity = load_equity_curve(checkpoint_path)

            if checkpoint_report:
                st.markdown("#### Checkpoint Report Metrics")
                st.json(checkpoint_report)

            if checkpoint_equity:
                st.markdown("#### Equity Curve")
                st.image(checkpoint_equity, use_column_width=True)

            if not checkpoint_trades.empty:
                st.markdown("#### Trade History")
                st.dataframe(checkpoint_trades)

        else:
            st.info("No checkpoint backtests found in this study folder.")

    st.markdown("---")

    # ---------------- Manual Backtests ----------------
    st.subheader("📦 Manual Backtests")

    backtest_sessions = list_backtest_sessions()
    if backtest_sessions:
        selected_session = st.selectbox("Select Manual Backtest", sorted(backtest_sessions, reverse=True))
        session_path = os.path.join(DATA_DIR, selected_session)

        report = load_report_metrics(session_path)
        trades = load_trades(session_path)
        equity_image = load_equity_curve(session_path)

        if report:
            st.markdown("#### Backtest Report Metrics")
            st.json(report)

        if equity_image:
            st.markdown("#### Equity Curve")
            st.image(equity_image, use_column_width=True)

        if not trades.empty:
            st.markdown("#### Trade History")
            st.dataframe(trades)

    else:
        st.info("No manual backtest sessions found.")

# ------------------------------------------------------------------------------
# PLACEHOLDER TABS (For future expansion)
# ------------------------------------------------------------------------------

with tab1:
    st.info("📊 Balance & Equity — (Future: Connect live balances)")

with tab2:
    st.info("🔍 Trade Analysis — (Future: Advanced trade analytics)")

with tab3:
    st.info("🧠 Model Insights — (Future: Feature importance, confusion matrix)")

with tab4:
    st.info("⚙️ Features & Alerts — (Future: Live alerts, anomaly detection)")

# ------------------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------------------

st.markdown("---")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
