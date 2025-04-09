import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import json
from datetime import datetime, timezone

# ------------------------------------------------------------------------------
# CORE METRICS CALCULATIONS
# ------------------------------------------------------------------------------

def calculate_drawdown(equity_curve):
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    end_idx = drawdown.idxmin()
    start_idx = equity_curve[:end_idx].idxmax() if end_idx != 0 else 0
    duration = end_idx - start_idx
    return drawdown, max_drawdown, duration

def calculate_rolling_sharpe(pnl_series, window=10):
    rolling_mean = pnl_series.rolling(window).mean()
    rolling_std = pnl_series.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)
    return rolling_sharpe

# ------------------------------------------------------------------------------
# REPORT GENERATION
# ------------------------------------------------------------------------------

def generate_report_from_df(df, save_reports=False, output_dir="./csv"):
    df["cum_pnl"] = df["pnl"].cumsum()
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df = df.sort_values(by="exit_time").reset_index(drop=True)

    total_trades = len(df)
    total_pnl = df["pnl"].sum()
    returns = df["pnl"]

    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    avg_win = wins["pnl"].mean() if not wins.empty else 0
    avg_loss = losses["pnl"].mean() if not losses.empty else 0
    profit_factor = abs(wins["pnl"].sum() / losses["pnl"].sum()) if not losses.empty else float('inf')
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    drawdown_series, max_drawdown, drawdown_duration = calculate_drawdown(df["cum_pnl"])
    rolling_sharpe = calculate_rolling_sharpe(df["pnl"])

    total_fees = df["fees"].sum() if "fees" in df.columns else 0
    df["gross_pnl"] = df["pnl"] + df.get("fees", 0)
    df["cum_gross_pnl"] = df["gross_pnl"].cumsum()
    gross_pnl = df["gross_pnl"].sum()

    metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_trades": total_trades,
        "gross_pnl": float(gross_pnl),
        "total_fees": float(total_fees),
        "total_pnl": float(total_pnl),
        "sharpe_ratio": float(sharpe),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "expectancy": float(expectancy),
        "max_drawdown": float(max_drawdown),
        "drawdown_duration": int(drawdown_duration)
    }

    if save_reports:
        os.makedirs(output_dir, exist_ok=True)

        # Export DataFrame
        df.to_csv(os.path.join(output_dir, "backtest_report.csv"), index=False)

        # Export Metrics as JSON
        metrics_file = os.path.join(output_dir, "report_metrics.json")
        with open(metrics_file, "w") as f_json:
            json.dump(metrics, f_json, indent=4)
        print(f"[EXPORT] Metrics JSON saved: {metrics_file}")

        # Plots
        plot_series(df, drawdown_series, rolling_sharpe, output_dir)

        print(f"[EXPORT] Report saved to {output_dir}")

    return metrics

# ------------------------------------------------------------------------------
# PLOTTING UTILITIES
# ------------------------------------------------------------------------------

def plot_series(df, drawdown_series, rolling_sharpe, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(df["exit_time"], df["cum_pnl"], label='Equity Curve')
    plt.title("Equity Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "equity_curve.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["exit_time"], drawdown_series, color='red', label='Drawdown')
    plt.title("Drawdown Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "drawdown_curve.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["exit_time"], rolling_sharpe, color='green', label='Rolling Sharpe Ratio')
    plt.title("Rolling Sharpe Ratio")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "rolling_sharpe.png"))
    plt.close()

    if "fees" in df.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(df["exit_time"], df["fees"].cumsum(), label='Cumulative Fees', color='orange')
        plt.title("Cumulative Fees Over Time")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "cumulative_fees.png"))
        plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["exit_time"], df["cum_gross_pnl"], label='Gross Equity Curve', linestyle='--')
    plt.plot(df["exit_time"], df["cum_pnl"], label='Net Equity Curve', color='green')
    plt.title("Gross vs Net PnL Over Time")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "gross_vs_net_pnl.png"))
    plt.close()

    print(f"[EXPORT] Plots saved to {output_dir}")

# ------------------------------------------------------------------------------
# CLI ENTRY POINT (Optional)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=str, required=True, help="CSV file with trade history")
    args = parser.parse_args()

    if not os.path.exists(args.trades):
        raise FileNotFoundError(f"Trade file not found: {args.trades}")

    df = pd.read_csv(args.trades)
    generate_report_from_df(df, save_reports=True, output_dir=os.path.dirname(args.trades))
