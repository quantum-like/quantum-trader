import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import ttest_1samp

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "../data_bloomberg"
PREDICTIONS_FILE = "../results/baseline_results/test_predictions_rank_model.csv"
HMM_REGIMES_FILE = "../data/hmm_4_state_regimes_expanding.csv"

OUTPUT_DIR = "../results/portfolio_optimization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HORIZON = 20
TOP_N = 10
BOTTOM_N = 20
PERIODS_PER_YEAR = 12
RISK_FREE_RATE = 0.04

LOOKBACK_DAYS = 756
MIN_OBS_STATIC = 126
MIN_OBS_REGIME = 60
COV_SHRINKAGE = 0.20
RIDGE = 1e-8

MAX_WEIGHT = 0.25
RISK_AVERSION = 4.0
ALPHA_SCORE_SCALE = 0.02

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ============================================================
# HELPERS
# ============================================================

def monthly_rebalance_dates(data):
    dates = data[["date"]].drop_duplicates().sort_values("date")
    dates["year_month"] = dates["date"].dt.to_period("M")
    return dates.groupby("year_month")["date"].min().tolist()


def load_close_matrix(tickers):
    close_parts = []

    for ticker in sorted(set(tickers)):
        path = os.path.join(DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        if "date" not in df.columns or "close" not in df.columns:
            continue

        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "close"]].sort_values("date")
        df = df.rename(columns={"close": ticker})
        close_parts.append(df.set_index("date"))

    if not close_parts:
        raise RuntimeError("No close price files loaded. Check DATA_DIR.")

    close = pd.concat(close_parts, axis=1).sort_index()
    daily_returns = close.pct_change().replace([np.inf, -np.inf], np.nan)
    return close, daily_returns


def shrink_cov(cov, shrinkage=COV_SHRINKAGE):
    cov = cov.copy()
    diag = pd.DataFrame(
        np.diag(np.diag(cov.values)),
        index=cov.index,
        columns=cov.columns
    )

    shrunk = (1 - shrinkage) * cov + shrinkage * diag
    arr = shrunk.to_numpy(copy=True)
    arr[np.diag_indices_from(arr)] += RIDGE

    return pd.DataFrame(arr, index=cov.index, columns=cov.columns)


def make_alpha_scores(g):
    scores = g["predicted_return_score"].astype(float).values

    if len(scores) <= 1 or np.nanstd(scores) == 0:
        return np.zeros(len(scores))

    z = (scores - np.nanmean(scores)) / np.nanstd(scores)
    z = np.clip(z, -2.0, 2.0)

    return ALPHA_SCORE_SCALE * z


def estimate_static_cov(daily_returns, tickers, date):
    hist = daily_returns.loc[daily_returns.index < date, tickers].tail(LOOKBACK_DAYS)
    hist = hist.dropna(how="any")

    if len(hist) < MIN_OBS_STATIC:
        hist = daily_returns.loc[daily_returns.index < date, tickers].dropna(how="any")

    if len(hist) < 20:
        return None, len(hist)

    cov = hist.cov() * HORIZON
    cov = shrink_cov(cov)
    return cov, len(hist)


def estimate_hmm_regime_cov(daily_returns, regimes, tickers, date):
    regime_row = regimes[regimes["date"] <= date].tail(1)

    if regime_row.empty:
        return None, np.nan, 0, "no_regime"

    current_regime = int(regime_row["hmm_regime"].iloc[0])

    hist = daily_returns.loc[daily_returns.index < date, tickers].tail(LOOKBACK_DAYS)
    hist = hist.dropna(how="any")

    if hist.empty:
        return None, current_regime, 0, "no_history"

    regime_hist = regimes[regimes["date"] < date][["date", "hmm_regime"]].copy()
    regime_hist = regime_hist.set_index("date")

    joined = hist.join(regime_hist, how="left")
    same_regime = joined[joined["hmm_regime"] == current_regime].drop(columns=["hmm_regime"])

    if len(same_regime) >= MIN_OBS_REGIME:
        cov = same_regime.cov() * HORIZON
        cov = shrink_cov(cov)
        return cov, current_regime, len(same_regime), "hmm_same_regime_cov"

    cov, n_obs = estimate_static_cov(daily_returns, tickers, date)
    return cov, current_regime, len(same_regime), "fallback_static_cov"


def min_variance_weights(cov, max_weight=MAX_WEIGHT):
    if cov is None:
        return None

    tickers = list(cov.index)
    n = len(tickers)
    Sigma = cov.values

    if n == 0 or not np.all(np.isfinite(Sigma)):
        return None

    def objective(w):
        return float(w @ Sigma @ w)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, max_weight) for _ in range(n)]
    w0 = np.repeat(1 / n, n)

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10}
    )

    if not res.success or not np.all(np.isfinite(res.x)):
        return w0

    w = np.maximum(res.x, 0)
    return w / w.sum()


def score_mean_variance_weights(mu, cov, max_weight=MAX_WEIGHT, risk_aversion=RISK_AVERSION):
    if cov is None:
        return None

    n = len(mu)
    Sigma = cov.values
    mu = np.asarray(mu, dtype=float)

    if n == 0 or not np.all(np.isfinite(Sigma)) or not np.all(np.isfinite(mu)):
        return None

    def objective(w):
        expected_score = w @ mu
        risk = w @ Sigma @ w
        return -float(expected_score - risk_aversion * risk)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, max_weight) for _ in range(n)]
    w0 = np.repeat(1 / n, n)

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10}
    )

    if not res.success or not np.all(np.isfinite(res.x)):
        return w0

    w = np.maximum(res.x, 0)
    return w / w.sum()


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
        out["excess_vs_spy_p_value"] = ttest_1samp(excess, 0).pvalue
        out["beta_vs_spy"] = beta
        out["alpha_vs_spy"] = alpha_periodic * PERIODS_PER_YEAR

    return out


# ============================================================
# LOAD DATA
# ============================================================

pred = pd.read_csv(PREDICTIONS_FILE)
pred["date"] = pd.to_datetime(pred["date"])
pred = pred.sort_values(["date", "ticker"])

future_col = f"future_return_{HORIZON}d"
spy_col = f"spy_future_return_{HORIZON}d"

required = ["date", "ticker", "predicted_return_score", future_col, spy_col]
missing = [c for c in required if c not in pred.columns]
if missing:
    raise ValueError(f"Predictions file missing columns: {missing}")

hmm = pd.read_csv(HMM_REGIMES_FILE)
hmm.columns = [c.lower().strip() for c in hmm.columns]
hmm["date"] = pd.to_datetime(hmm["date"])

if "hmm_regime" not in hmm.columns:
    if "regime" in hmm.columns:
        hmm = hmm.rename(columns={"regime": "hmm_regime"})
    elif "predicted_regime" in hmm.columns:
        hmm = hmm.rename(columns={"predicted_regime": "hmm_regime"})
    else:
        raise ValueError("Could not find HMM regime column.")

hmm = hmm[["date", "hmm_regime"]].dropna().sort_values("date")

all_tickers = pred["ticker"].unique().tolist()
_, daily_returns = load_close_matrix(all_tickers)


# ============================================================
# BACKTEST
# ============================================================

rows = []
weight_rows = []

for date in monthly_rebalance_dates(pred):
    g_all = pred[pred["date"] == date].copy()
    g_all = g_all.dropna(subset=["predicted_return_score", future_col])

    if len(g_all) < TOP_N + BOTTOM_N:
        continue

    g_all = g_all.sort_values("predicted_return_score", ascending=False)
    top = g_all.head(TOP_N).copy()
    bottom = g_all.tail(BOTTOM_N).copy()

    tickers = [t for t in top["ticker"].tolist() if t in daily_returns.columns]
    top = top[top["ticker"].isin(tickers)].copy()

    if len(top) < 2:
        continue

    top = top.set_index("ticker").loc[tickers].reset_index()

    actual_returns = top[future_col].values
    spy_return = float(g_all[spy_col].iloc[0])
    bottom20_return = bottom[future_col].mean()

    mu = make_alpha_scores(top)

    static_cov, static_nobs = estimate_static_cov(daily_returns, tickers, date)
    hmm_cov, regime, hmm_nobs, hmm_cov_source = estimate_hmm_regime_cov(
        daily_returns, hmm, tickers, date
    )

    ew_w = np.repeat(1 / len(top), len(top))

    static_minvar_w = min_variance_weights(static_cov)
    if static_minvar_w is None:
        static_minvar_w = ew_w.copy()

    static_scoremv_w = score_mean_variance_weights(mu, static_cov)
    if static_scoremv_w is None:
        static_scoremv_w = ew_w.copy()

    hmm_minvar_w = min_variance_weights(hmm_cov)
    if hmm_minvar_w is None:
        hmm_minvar_w = ew_w.copy()

    hmm_scoremv_w = score_mean_variance_weights(mu, hmm_cov)
    if hmm_scoremv_w is None:
        hmm_scoremv_w = ew_w.copy()

    strategy_weights = {
        "EqualWeight_Top10": ew_w,
        "Static_MinVar_Top10": static_minvar_w,
        "Static_ScoreMV_Top10": static_scoremv_w,
        "HMM_MinVar_Top10": hmm_minvar_w,
        "HMM_ScoreMV_Top10": hmm_scoremv_w,
    }

    row = {
        "date": date,
        "hmm_regime": regime,
        "hmm_cov_source": hmm_cov_source,
        "static_cov_nobs": static_nobs,
        "hmm_same_regime_nobs": hmm_nobs,
        "spy_return": spy_return,
        "bottom20_return": bottom20_return,
    }

    for name, weights in strategy_weights.items():
        row[name] = float(weights @ actual_returns)

        for ticker, weight, realized_ret, score in zip(
            top["ticker"].values,
            weights,
            actual_returns,
            top["predicted_return_score"].values
        ):
            weight_rows.append({
                "date": date,
                "strategy": name,
                "ticker": ticker,
                "weight": weight,
                "realized_future_return": realized_ret,
                "predicted_return_score": score,
                "hmm_regime": regime,
            })

    rows.append(row)

portfolio = pd.DataFrame(rows).sort_values("date")
weights = pd.DataFrame(weight_rows).sort_values(["date", "strategy", "ticker"])

portfolio.to_csv(os.path.join(OUTPUT_DIR, "optimized_portfolio_returns.csv"), index=False)
weights.to_csv(os.path.join(OUTPUT_DIR, "optimized_portfolio_weights.csv"), index=False)


# ============================================================
# METRICS
# ============================================================

portfolio_i = portfolio.set_index("date")

strategy_cols = [
    "EqualWeight_Top10",
    "Static_MinVar_Top10",
    "Static_ScoreMV_Top10",
    "HMM_MinVar_Top10",
    "HMM_ScoreMV_Top10",
]

perf_rows = []

for name in strategy_cols:
    metrics = performance_metrics(
        portfolio_i[name],
        benchmark_returns=portfolio_i["spy_return"]
    )
    metrics["strategy"] = name
    perf_rows.append(metrics)

spy_metrics = performance_metrics(portfolio_i["spy_return"])
spy_metrics["strategy"] = "SPY"
perf_rows.append(spy_metrics)

bottom_metrics = performance_metrics(
    portfolio_i["bottom20_return"],
    benchmark_returns=portfolio_i["spy_return"]
)
bottom_metrics["strategy"] = "Bottom20_Predicted"
perf_rows.append(bottom_metrics)

performance = pd.DataFrame(perf_rows)

ordered = [
    "strategy", "total_return", "CAGR", "annual_return", "annual_volatility",
    "sharpe", "max_drawdown", "annual_excess_return_vs_spy",
    "excess_vs_spy_p_value", "beta_vs_spy", "alpha_vs_spy", "n_periods"
]
for c in ordered:
    if c not in performance.columns:
        performance[c] = np.nan

performance = performance[ordered]
performance.to_csv(os.path.join(OUTPUT_DIR, "optimized_portfolio_performance_metrics.csv"), index=False)


# ============================================================
# STATISTICAL COMPARISONS
# ============================================================

comparison_rows = []

for name in strategy_cols:
    returns = portfolio[name]

    comparison_rows.append({
        "strategy": name,
        "top_minus_bottom_p_value": ttest_1samp(
            returns - portfolio["bottom20_return"], 0
        ).pvalue,
        "top_minus_spy_p_value": ttest_1samp(
            returns - portfolio["spy_return"], 0
        ).pvalue,
        "strategy_minus_equal_weight_p_value": (
            np.nan if name == "EqualWeight_Top10"
            else ttest_1samp(returns - portfolio["EqualWeight_Top10"], 0).pvalue
        ),
        "avg_monthly_return": returns.mean(),
        "avg_monthly_excess_vs_equal_weight": (
            np.nan if name == "EqualWeight_Top10"
            else (returns - portfolio["EqualWeight_Top10"]).mean()
        ),
        "annualized_excess_vs_equal_weight": (
            np.nan if name == "EqualWeight_Top10"
            else (returns - portfolio["EqualWeight_Top10"]).mean() * PERIODS_PER_YEAR
        ),
        "percent_months_beats_equal_weight": (
            np.nan if name == "EqualWeight_Top10"
            else (returns > portfolio["EqualWeight_Top10"]).mean()
        ),
    })

comparison = pd.DataFrame(comparison_rows)
comparison.to_csv(os.path.join(OUTPUT_DIR, "optimization_comparison_tests.csv"), index=False)


# ============================================================
# WEIGHT SUMMARY
# ============================================================

weight_summary = (
    weights.groupby("strategy")["weight"]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
)

max_weight_by_month = (
    weights.groupby(["strategy", "date"])["weight"]
    .max()
    .groupby("strategy")
    .mean()
    .reset_index(name="avg_monthly_max_weight")
)

weight_summary = weight_summary.merge(max_weight_by_month, on="strategy", how="left")
weight_summary.to_csv(os.path.join(OUTPUT_DIR, "weight_summary.csv"), index=False)


# ============================================================
# REGIME CONDITIONAL OPTIMIZATION PERFORMANCE
# ============================================================

regime_rows = []

for regime, g in portfolio.groupby("hmm_regime"):
    for name in strategy_cols:
        regime_rows.append({
            "hmm_regime": regime,
            "strategy": name,
            "n_months": len(g),
            "avg_monthly_return": g[name].mean(),
            "annualized_return": g[name].mean() * PERIODS_PER_YEAR,
            "avg_excess_vs_spy": (g[name] - g["spy_return"]).mean(),
            "annualized_excess_vs_spy": (g[name] - g["spy_return"]).mean() * PERIODS_PER_YEAR,
            "avg_excess_vs_equal_weight": (
                np.nan if name == "EqualWeight_Top10"
                else (g[name] - g["EqualWeight_Top10"]).mean()
            ),
        })

regime_perf = pd.DataFrame(regime_rows)
regime_perf.to_csv(os.path.join(OUTPUT_DIR, "regime_conditional_optimization_performance.csv"), index=False)


# ============================================================
# YEARLY RETURNS
# ============================================================

portfolio["year"] = portfolio["date"].dt.year
yearly_rows = []

for year, g in portfolio.groupby("year"):
    row = {"year": year}
    for name in strategy_cols:
        row[name] = (1 + g[name]).prod() - 1
    row["SPY"] = (1 + g["spy_return"]).prod() - 1
    yearly_rows.append(row)

yearly = pd.DataFrame(yearly_rows)
yearly.to_csv(os.path.join(OUTPUT_DIR, "optimized_portfolio_yearly_returns.csv"), index=False)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\nFinal portfolio optimization experiment")
print("=======================================")
print(f"Predictions file: {PREDICTIONS_FILE}")
print(f"HMM regimes file: {HMM_REGIMES_FILE}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Rebalance months: {len(portfolio)}")

print("\nPerformance:")
print(performance.round(4))

print("\nComparison tests:")
print(comparison.round(4))

print("\nWeight summary:")
print(weight_summary.round(4))

print("\nRegime conditional optimization performance:")
print(regime_perf.round(4))