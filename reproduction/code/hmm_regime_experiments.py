# Regime-aware forecasting experiments (paper Tables 10, 11, 12, 13).
# The public replication repo does not include this script; it is reconstructed
# from the paper's Section 4.5 description on top of baseline.py:
#   (a) HMM regime as one-hot predictive features added to the baseline XGBoost
#   (b) One XGBoost model per HMM regime (train on regime subsample, predict with
#       the model matching the regime prevailing at each date)
#   (c) Baseline strategy performance conditional on the prevailing regime
#   (d) Paired tests of (a) and (b) against the baseline with Newey-West HAC p-values
#
# Uses the same data layout, feature set, seeds and portfolio construction as baseline.py.

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, ttest_1samp
from xgboost import XGBRegressor
import statsmodels.api as sm

DATA_DIR = "../data_bloomberg"
FUNDAMENTALS_DIR = "../data_bloomberg_fundamentals"
MACRO_DIR = "../data_bloomberg_macro"
HMM_REGIMES_FILE = "../data/hmm_4_state_regimes_expanding.csv"
BASELINE_PREDICTIONS = "../results/baseline_results/test_predictions_rank_model.csv"
USE_FUNDAMENTALS = True

OUTPUT_DIR = "../results/hmm_regime_experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HORIZON = 20
TOP_N = 10
BOTTOM_N = 20
RISK_FREE_RATE = 0.04
PERIODS_PER_YEAR = 12

TRAIN_START = "2005-01-01"
TRAIN_END   = "2018-12-31"
VAL_START   = "2019-01-01"
VAL_END     = "2020-12-31"
TEST_START  = "2021-01-01"
TEST_END    = "2025-12-31"

RANDOM_STATE = 42
SEEDS = [1, 10, 42, 123, 500, 999, 2025]

TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO",
    "JPM","V","TSLA","XOM","UNH","MA","COST","HD","PG","NFLX",
    "JNJ","CRM","ABBV","BAC","KO","WMT","CVX","MRK","AMD","PEP",
    "ADBE","TMO","LIN","MCD","CSCO","ACN","ABT","ORCL","WFC","QCOM",
    "INTU","IBM","GE","TXN","DHR","AMAT","VZ","CAT","PM","NOW",
    "ISRG","DIS","NEE","RTX","PFE","GS","UBER","SPGI","UNP","LOW",
    "HON","BKNG","T","PGR","BLK","ELV","SYK","LMT","TJX","VRTX",
    "MS","C","MDT","ADI","MU","PANW","CB","ETN","REGN",
    "BSX","PLD","DE","SCHW","KLAC","ADP","CI","AMT","LRCX","COP",
    "GILD","MDLZ","ZTS","SBUX","TMUS","SO","UPS","MO","CME",
    "ICE","EQIX","DUK","NKE","SHW","CL","APH","WM","MCK","CDNS",
    "SNPS","PH","ANET","CMG","TT","ORLY","TDG","EOG","ITW","USB",
    "PNC","AON","HCA","MCO","MSI","BDX","APD","EMR","GD","FDX",
    "CTAS","MMM","MAR","ROP","NOC","ECL","AJG","COF","TFC","CSX",
    "NSC","HLT","GM","FCX","SLB","NXPI","WELL","PSA","AEP","O",
    "TRV","AZO","ROST","SPG","PCAR","SRE","DLR","AFL","MET","OKE",
    "PSX","KMB","D","ALL","VLO","URI","TEL","F","BK","AMP",
    "AIG","KMI","LHX","FIS","COR","PAYX","MNST","DOW","HUM","PRU",
    "OXY","FAST","KR","NEM","CTVA","ODFL","PEG","YUM","RSG","VRSK",
    "EXC","LEN","KDP","IR","GEHC","KHC","EA","CHTR","PPG",
    "XEL","DD","HAL","BIIB","GLW","HPQ","DLTR","GPN","MTD"
]

# ============================================================
# PANEL CONSTRUCTION (identical to baseline.py)
# ============================================================

def load_ohlcv(path):
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[required]


def load_fundamentals(ticker):
    path = os.path.join(FUNDAMENTALS_DIR, f"{ticker}_fundamentals.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    expected = ["date", "market_cap", "pe_ratio", "pb_ratio", "roe", "debt_to_equity"]
    df = df[expected].copy()
    for col in expected:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["log_market_cap"] = np.log(df["market_cap"].replace(0, np.nan))
    fund_cols = ["market_cap", "pe_ratio", "pb_ratio", "roe", "debt_to_equity", "log_market_cap"]
    df[fund_cols] = df[fund_cols].ffill()
    return df


def add_stock_features(df):
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    for w in [1, 2, 5, 10, 20, 60, 120, 252]:
        df[f"ret_{w}d"] = close.pct_change(w)

    df["momentum_3m"] = df["ret_60d"]
    df["momentum_6m"] = df["ret_120d"]
    df["momentum_12m"] = df["ret_252d"]
    df["momentum_12m_skip_1m"] = close.shift(20) / close.shift(252) - 1

    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_200"] = close.rolling(200).mean()

    df["price_sma20"] = close / df["sma_20"] - 1
    df["price_sma50"] = close / df["sma_50"] - 1
    df["price_sma200"] = close / df["sma_200"] - 1
    df["sma20_sma50"] = df["sma_20"] / df["sma_50"] - 1

    daily_ret = close.pct_change()
    for w in [5, 10, 20, 60]:
        df[f"vol_{w}d"] = daily_ret.rolling(w).std() * np.sqrt(252)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr_14_pct"] = tr.rolling(14).mean() / close

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["rsi_14"] = 100 - 100 / (1 + rs)

    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    df["stoch_k"] = 100 * (close - low_14) / (high_14 - low_14)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["williams_r"] = -100 * (high_14 - close) / (high_14 - low_14)

    df["volume_change_1d"] = volume.pct_change()
    df["volume_sma20_ratio"] = volume / volume.rolling(20).mean()

    df[f"future_return_{HORIZON}d"] = close.shift(-HORIZON) / close - 1
    return df


spy = load_ohlcv(os.path.join(DATA_DIR, "SPY.csv"))
spy = spy.rename(columns={
    "open": "spy_open", "high": "spy_high", "low": "spy_low",
    "close": "spy_close", "volume": "spy_volume"
})
spy["spy_ret_1d"] = spy["spy_close"].pct_change(1)
for w in [5, 10, 20, 60, 120, 252]:
    spy[f"spy_ret_{w}d"] = spy["spy_close"].pct_change(w)
for w in [5, 10, 20, 60]:
    spy[f"spy_vol_{w}d"] = spy["spy_ret_1d"].rolling(w).std() * np.sqrt(252)
spy[f"spy_future_return_{HORIZON}d"] = spy["spy_close"].shift(-HORIZON) / spy["spy_close"] - 1

vix = pd.read_csv(os.path.join(DATA_DIR, "VIX.csv"))
vix.columns = [c.lower().strip() for c in vix.columns]
vix["date"] = pd.to_datetime(vix["date"])
if "close" in vix.columns:
    vix = vix.rename(columns={"close": "vix_close"})
vix = vix[["date", "vix_close"]].sort_values("date")
vix["vix_change_5d"] = vix["vix_close"].pct_change(5)
vix["vix_change_20d"] = vix["vix_close"].pct_change(20)

market = spy.merge(vix, on="date", how="left")

macro = pd.read_csv(os.path.join(MACRO_DIR, "macro_features.csv"))
macro["date"] = pd.to_datetime(macro["date"])
macro = macro.sort_values("date")

market = pd.merge_asof(
    market.sort_values("date"), macro.sort_values("date"),
    on="date", direction="backward"
)

parts = []
for ticker in TICKERS:
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        continue
    try:
        stock = load_ohlcv(path)
        stock = add_stock_features(stock)
        stock["ticker"] = ticker
        merged = stock.merge(market, on="date", how="inner")

        if USE_FUNDAMENTALS:
            fundamentals = load_fundamentals(ticker)
            if fundamentals is not None:
                merged = pd.merge_asof(
                    merged.sort_values("date"),
                    fundamentals.sort_values("date"),
                    on="date", direction="backward"
                )
                fund_cols_tmp = ["log_market_cap", "pe_ratio", "pb_ratio", "roe", "debt_to_equity"]
                merged[fund_cols_tmp] = merged[fund_cols_tmp].ffill()

        daily_stock_ret = merged["close"].pct_change()
        daily_spy_ret = merged["spy_close"].pct_change()
        for w in [20, 60]:
            merged[f"beta_{w}d"] = (
                daily_stock_ret.rolling(w).cov(daily_spy_ret)
                / daily_spy_ret.rolling(w).var()
            )
            merged[f"corr_spy_{w}d"] = daily_stock_ret.rolling(w).corr(daily_spy_ret)
        for w in [5, 10, 20, 60, 120, 252]:
            merged[f"rel_strength_{w}d"] = merged[f"ret_{w}d"] - merged[f"spy_ret_{w}d"]
        merged["rel_momentum_6m"] = merged["rel_strength_120d"]
        merged["rel_momentum_12m"] = merged["rel_strength_252d"]
        parts.append(merged)
    except Exception:
        continue

panel = pd.concat(parts, ignore_index=True)
panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

panel["future_return_rank_pct"] = (
    panel.groupby("date")[f"future_return_{HORIZON}d"].rank(pct=True)
)

fundamental_features = [
    "log_market_cap", "pe_ratio", "pb_ratio", "roe", "debt_to_equity"
] if USE_FUNDAMENTALS else []

original_features = [
    "ret_1d", "ret_2d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "rel_strength_5d", "rel_strength_10d", "rel_strength_20d", "rel_strength_60d",
    "price_sma20", "price_sma50", "price_sma200", "sma20_sma50",
    "vol_5d", "vol_10d", "vol_20d", "vol_60d",
    "atr_14_pct",
    "macd", "macd_signal", "macd_hist",
    "rsi_14", "stoch_k", "stoch_d", "williams_r",
    "volume_change_1d", "volume_sma20_ratio",
    "spy_ret_1d", "spy_ret_5d", "spy_ret_10d", "spy_ret_20d", "spy_ret_60d",
    "spy_vol_5d", "spy_vol_10d", "spy_vol_20d", "spy_vol_60d",
    "vix_close", "vix_change_5d", "vix_change_20d",
    "beta_20d", "beta_60d", "corr_spy_20d", "corr_spy_60d"
]

momentum_features = [
    "ret_120d", "ret_252d",
    "momentum_3m", "momentum_6m", "momentum_12m", "momentum_12m_skip_1m",
    "rel_strength_120d", "rel_strength_252d",
    "rel_momentum_6m", "rel_momentum_12m",
    "spy_ret_120d", "spy_ret_252d"
]

cross_asset_features = [
    "us_agg_bond_ret_20d", "us_agg_bond_ret_60d",
    "dxy_ret_20d", "dxy_ret_60d",
    "gold_ret_20d", "gold_ret_60d",
    "oil_ret_20d", "oil_ret_60d",
]

numeric_features = original_features + momentum_features + fundamental_features + cross_asset_features
numeric_features = list(dict.fromkeys(numeric_features))

panel = panel.replace([np.inf, -np.inf], np.nan)

if USE_FUNDAMENTALS:
    for col in fundamental_features:
        if col in panel.columns:
            panel[col] = panel.groupby("ticker")[col].ffill()
            panel[col] = panel.groupby("date")[col].transform(lambda x: x.fillna(x.median()))

available_features = [c for c in numeric_features if c in panel.columns]

panel = panel.dropna(subset=available_features + [
    f"future_return_{HORIZON}d",
    f"spy_future_return_{HORIZON}d",
    "future_return_rank_pct"
])

# ============================================================
# MERGE EXPANDING-WINDOW HMM REGIMES
# ============================================================

hmm = pd.read_csv(HMM_REGIMES_FILE)
hmm.columns = [c.lower().strip() for c in hmm.columns]
hmm["date"] = pd.to_datetime(hmm["date"])
hmm = hmm.dropna(subset=["hmm_regime"])[["date", "hmm_regime"]].sort_values("date")
hmm["hmm_regime"] = hmm["hmm_regime"].astype(int)

panel = pd.merge_asof(
    panel.sort_values("date"), hmm,
    on="date", direction="backward"
)
panel = panel.dropna(subset=["hmm_regime"])
panel["hmm_regime"] = panel["hmm_regime"].astype(int)
panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

N_REGIMES = 4
regime_dummies = [f"hmm_regime_{k}" for k in range(N_REGIMES)]
for k in range(N_REGIMES):
    panel[f"hmm_regime_{k}"] = (panel["hmm_regime"] == k).astype(float)

# ============================================================
# SPLITS
# ============================================================

train = panel[(panel["date"] >= TRAIN_START) & (panel["date"] <= TRAIN_END)]
val = panel[(panel["date"] >= VAL_START) & (panel["date"] <= VAL_END)]
test = panel[(panel["date"] >= TEST_START) & (panel["date"] <= TEST_END)]

y_train = train["future_return_rank_pct"]

def make_xgb(seed):
    return XGBRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=2.0, objective="reg:squarederror",
        random_state=seed, n_jobs=-1
    )

# ------------------------------------------------------------
# (a) HMM regime one-hot as additional features
# ------------------------------------------------------------

feat_hmm = available_features + regime_dummies
test_preds = []
for seed in SEEDS:
    m = make_xgb(seed)
    m.fit(train[feat_hmm], y_train)
    test_preds.append(m.predict(test[feat_hmm]))
hmm_feature_pred = np.mean(test_preds, axis=0)

# ------------------------------------------------------------
# (b) One model per regime
# ------------------------------------------------------------

per_regime_models = {}
for k in range(N_REGIMES):
    sub = train[train["hmm_regime"] == k]
    if len(sub) < 1000:
        per_regime_models[k] = None
        continue
    seed_models = []
    for seed in SEEDS:
        m = make_xgb(seed)
        m.fit(sub[available_features], sub["future_return_rank_pct"])
        seed_models.append(m)
    per_regime_models[k] = seed_models

# Global fallback for regimes with too little training data (only trained if needed)
global_models = []
if any(per_regime_models[k] is None and (test["hmm_regime"] == k).any() for k in range(N_REGIMES)):
    for seed in SEEDS:
        m = make_xgb(seed)
        m.fit(train[available_features], y_train)
        global_models.append(m)

per_regime_pred = np.full(len(test), np.nan)
test_reset = test.reset_index(drop=True)
for k in range(N_REGIMES):
    mask = (test_reset["hmm_regime"] == k).values
    if mask.sum() == 0:
        continue
    models_k = per_regime_models[k] if per_regime_models[k] is not None else global_models
    X_k = test_reset.loc[mask, available_features]
    per_regime_pred[mask] = np.mean([m.predict(X_k) for m in models_k], axis=0)

# ============================================================
# EVALUATION (same portfolio construction as baseline.py)
# ============================================================

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
        out["beta_vs_spy"] = beta
        out["alpha_vs_spy"] = alpha_periodic * PERIODS_PER_YEAR
    return out


def build_top_bottom_portfolios(data, score_col):
    rows = []
    for date in monthly_rebalance_dates(data):
        g = data[data["date"] == date].copy()
        g = g.dropna(subset=[score_col, f"future_return_{HORIZON}d"])
        if len(g) < TOP_N + BOTTOM_N:
            continue
        g = g.sort_values(score_col, ascending=False)
        top = g.head(TOP_N)
        bottom = g.tail(BOTTOM_N)
        n_random_trials = 1000
        random_return = np.mean([
            g.sample(n=TOP_N, random_state=RANDOM_STATE + i)[f"future_return_{HORIZON}d"].mean()
            for i in range(n_random_trials)
        ])
        spearman_ic = spearmanr(g[score_col], g[f"future_return_{HORIZON}d"]).correlation
        rows.append({
            "date": date,
            "top10_return": top[f"future_return_{HORIZON}d"].mean(),
            "bottom20_return": bottom[f"future_return_{HORIZON}d"].mean(),
            "random_top10_return": random_return,
            "top_minus_bottom": top[f"future_return_{HORIZON}d"].mean() - bottom[f"future_return_{HORIZON}d"].mean(),
            "spy_return": g[f"spy_future_return_{HORIZON}d"].iloc[0],
            "spearman_ic": spearman_ic,
            "hmm_regime": int(g["hmm_regime"].iloc[0]),
        })
    return pd.DataFrame(rows).sort_values("date")


def diagnostics(portfolio):
    return {
        "top_minus_bottom_p": ttest_1samp(portfolio["top_minus_bottom"], 0).pvalue,
        "top_minus_random_p": ttest_1samp(portfolio["top10_return"] - portfolio["random_top10_return"], 0).pvalue,
        "top_minus_spy_p": ttest_1samp(portfolio["top10_return"] - portfolio["spy_return"], 0).pvalue,
        "spearman_ic_mean": portfolio["spearman_ic"].mean(),
        "spearman_ic_p": ttest_1samp(portfolio["spearman_ic"].dropna(), 0).pvalue,
    }


results = {}
portfolios = {}

for name, pred_vec in [("hmm_as_feature", hmm_feature_pred), ("one_model_per_regime", per_regime_pred)]:
    scored = test.copy()
    scored["predicted_return_score"] = pred_vec
    port = build_top_bottom_portfolios(scored, "predicted_return_score")
    portfolios[name] = port
    port.to_csv(os.path.join(OUTPUT_DIR, f"portfolio_{name}.csv"), index=False)
    metrics = performance_metrics(port.set_index("date")["top10_return"],
                                  benchmark_returns=port.set_index("date")["spy_return"])
    metrics.update(diagnostics(port))
    results[name] = metrics

# ------------------------------------------------------------
# (c) Baseline conditional on regime + (d) paired HAC tests
# ------------------------------------------------------------

baseline_pred = pd.read_csv(BASELINE_PREDICTIONS)
baseline_pred["date"] = pd.to_datetime(baseline_pred["date"])
needed = ["date", "ticker", "predicted_return_score",
          f"future_return_{HORIZON}d", f"spy_future_return_{HORIZON}d"]
baseline_scored = baseline_pred[needed].merge(
    test[["date", "ticker", "hmm_regime"]], on=["date", "ticker"], how="inner"
)
baseline_port = build_top_bottom_portfolios(baseline_scored, "predicted_return_score")
baseline_port.to_csv(os.path.join(OUTPUT_DIR, "portfolio_baseline_with_regime.csv"), index=False)

regime_rows = []
for regime, g in baseline_port.groupby("hmm_regime"):
    eq = (1 + g["top10_return"]).cumprod()
    years = len(g) / PERIODS_PER_YEAR
    ann_ret = g["top10_return"].mean() * PERIODS_PER_YEAR
    ann_vol = g["top10_return"].std() * np.sqrt(PERIODS_PER_YEAR)
    regime_rows.append({
        "hmm_regime": regime,
        "n_months": len(g),
        "CAGR": eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "sharpe": (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else np.nan,
        "avg_ic": g["spearman_ic"].mean(),
    })
pd.DataFrame(regime_rows).to_csv(os.path.join(OUTPUT_DIR, "baseline_regime_conditional.csv"), index=False)


def hac_paired_test(diff_series, lags=3):
    diff = pd.Series(diff_series).dropna()
    X = np.ones((len(diff), 1))
    model = sm.OLS(diff.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(model.params[0]), float(model.pvalues[0])


paired_rows = []
base_i = baseline_port.set_index("date")
for name, port in portfolios.items():
    alt_i = port.set_index("date")
    joined = base_i.join(alt_i, lsuffix="_base", rsuffix="_alt", how="inner")
    for spread, bcol, acol in [
        ("top10_return", "top10_return_base", "top10_return_alt"),
        ("top_minus_spy", None, None),
        ("top_minus_bottom", "top_minus_bottom_base", "top_minus_bottom_alt"),
        ("top_minus_random", None, None),
    ]:
        if spread == "top_minus_spy":
            diff = (joined["top10_return_alt"] - joined["spy_return_alt"]) - \
                   (joined["top10_return_base"] - joined["spy_return_base"])
        elif spread == "top_minus_random":
            diff = (joined["top10_return_alt"] - joined["random_top10_return_alt"]) - \
                   (joined["top10_return_base"] - joined["random_top10_return_base"])
        else:
            diff = joined[acol] - joined[bcol]
        mean_diff, p = hac_paired_test(diff)
        paired_rows.append({
            "alternative": name,
            "spread": spread,
            "annualized_diff": mean_diff * PERIODS_PER_YEAR,
            "hac_p_value": p,
        })

paired = pd.DataFrame(paired_rows)
paired.to_csv(os.path.join(OUTPUT_DIR, "paired_tests_vs_baseline.csv"), index=False)

summary = pd.DataFrame(results).T
summary.to_csv(os.path.join(OUTPUT_DIR, "regime_experiment_metrics.csv"))

print("\nRegime experiment metrics:")
print(summary.round(4))
print("\nBaseline conditional on regime:")
print(pd.DataFrame(regime_rows).round(4))
print("\nPaired HAC tests vs baseline:")
print(paired.round(4))
