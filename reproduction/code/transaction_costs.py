import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

PREDICTIONS_FILE = "../results/baseline_results/test_predictions_rank_model.csv"
OUTPUT_DIR = "../results/transaction_costs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HORIZON = 20
TOP_N = 10
BOTTOM_N = 20
PERIODS_PER_YEAR = 12
RISK_FREE_RATE = 0.04

TRANSACTION_COST_BPS_LIST = [0, 5, 10, 25, 50]


def monthly_rebalance_dates(data):
    dates = data[["date"]].drop_duplicates().sort_values("date")
    dates["year_month"] = dates["date"].dt.to_period("M")
    return dates.groupby("year_month")["date"].min().tolist()


def max_drawdown(equity):
    peak = equity.cummax()
    return (equity / peak - 1).min()


def performance_metrics(returns, benchmark_returns=None):
    returns = returns.dropna()
    equity = (1 + returns).cumprod()

    years = len(returns) / PERIODS_PER_YEAR
    annual_return = returns.mean() * PERIODS_PER_YEAR
    annual_vol = returns.std() * np.sqrt(PERIODS_PER_YEAR)

    out = {
        "total_return": equity.iloc[-1] - 1,
        "CAGR": equity.iloc[-1] ** (1 / years) - 1,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": (annual_return - RISK_FREE_RATE) / annual_vol if annual_vol != 0 else np.nan,
        "max_drawdown": max_drawdown(equity),
        "n_periods": len(returns),
    }

    if benchmark_returns is not None:
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ["strategy", "spy"]
        excess = aligned["strategy"] - aligned["spy"]

        x = aligned["spy"].values
        y = aligned["strategy"].values

        beta = np.cov(y, x)[0, 1] / np.var(x)
        alpha_periodic = y.mean() - beta * x.mean()

        out["annual_excess_return_vs_spy"] = excess.mean() * PERIODS_PER_YEAR
        out["top_minus_spy_p_value"] = ttest_1samp(excess, 0).pvalue
        out["beta_vs_spy"] = beta
        out["alpha_vs_spy"] = alpha_periodic * PERIODS_PER_YEAR

    return out


pred = pd.read_csv(PREDICTIONS_FILE)
pred["date"] = pd.to_datetime(pred["date"])
pred = pred.sort_values(["date", "ticker"])

future_col = f"future_return_{HORIZON}d"
spy_col = f"spy_future_return_{HORIZON}d"

rows = []

prev_end_weights = {}

for date in monthly_rebalance_dates(pred):
    g = pred[pred["date"] == date].copy()
    g = g.dropna(subset=["predicted_return_score", future_col])

    if len(g) < TOP_N + BOTTOM_N:
        continue

    g = g.sort_values("predicted_return_score", ascending=False)
    top = g.head(TOP_N).copy()
    bottom = g.tail(BOTTOM_N).copy()

    selected = top["ticker"].tolist()

    new_weights = {ticker: 1 / TOP_N for ticker in selected}

    all_tickers = set(prev_end_weights.keys()).union(new_weights.keys())

    turnover = sum(
        abs(new_weights.get(t, 0.0) - prev_end_weights.get(t, 0.0))
        for t in all_tickers
    )

    gross_return = top[future_col].mean()
    spy_return = g[spy_col].iloc[0]
    bottom_return = bottom[future_col].mean()

    for cost_bps in TRANSACTION_COST_BPS_LIST:
        cost_rate = cost_bps / 10000
        transaction_cost = cost_rate * turnover
        net_return = gross_return - transaction_cost

        rows.append({
            "date": date,
            "cost_bps": cost_bps,
            "gross_return": gross_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "net_return": net_return,
            "spy_return": spy_return,
            "bottom20_return": bottom_return,
            "top_minus_bottom_net": net_return - bottom_return,
            "top_minus_spy_net": net_return - spy_return,
            "top10_tickers": ",".join(selected),
        })

    # Update old weights after the realized holding-period returns.
    # This gives the weights just before the next rebalance.
    realized = top.set_index("ticker")[future_col].to_dict()
    end_values = {
        ticker: weight * (1 + realized[ticker])
        for ticker, weight in new_weights.items()
    }
    total_end_value = sum(end_values.values())

    prev_end_weights = {
        ticker: value / total_end_value
        for ticker, value in end_values.items()
    }


tc = pd.DataFrame(rows)
tc.to_csv(os.path.join(OUTPUT_DIR, "transaction_cost_monthly_returns.csv"), index=False)

perf_rows = []

for cost_bps, g in tc.groupby("cost_bps"):
    g = g.sort_values("date")

    metrics = performance_metrics(
        g["net_return"],
        benchmark_returns=g["spy_return"]
    )

    metrics["cost_bps"] = cost_bps
    metrics["avg_turnover"] = g["turnover"].mean()
    metrics["avg_monthly_transaction_cost"] = g["transaction_cost"].mean()
    metrics["annualized_transaction_cost"] = g["transaction_cost"].mean() * PERIODS_PER_YEAR

    metrics["top_minus_bottom_p_value"] = ttest_1samp(
        g["top_minus_bottom_net"], 0
    ).pvalue

    metrics["top_minus_spy_p_value"] = ttest_1samp(
        g["top_minus_spy_net"], 0
    ).pvalue

    perf_rows.append(metrics)

perf = pd.DataFrame(perf_rows)

ordered_cols = [
    "cost_bps",
    "total_return",
    "CAGR",
    "annual_return",
    "annual_volatility",
    "sharpe",
    "max_drawdown",
    "annual_excess_return_vs_spy",
    "alpha_vs_spy",
    "top_minus_spy_p_value",
    "top_minus_bottom_p_value",
    "avg_turnover",
    "avg_monthly_transaction_cost",
    "annualized_transaction_cost",
    "n_periods",
]

for col in ordered_cols:
    if col not in perf.columns:
        perf[col] = np.nan

perf = perf[ordered_cols]
perf.to_csv(os.path.join(OUTPUT_DIR, "transaction_cost_performance_summary.csv"), index=False)

print("\nTransaction cost robustness:")
print(perf.round(4))
print(f"\nSaved results in: {OUTPUT_DIR}")