# Clean final model
#   1) build daily stock panel
#   2) train XGBoost model
#   3) rank stock
#   4) equal-weight Top 10 monthly portfolio

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, ttest_1samp, pearsonr
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import shap


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "../data_bloomberg"
FUNDAMENTALS_DIR = "../data_bloomberg_fundamentals"
MACRO_DIR = "../data_bloomberg_macro"
USE_FUNDAMENTALS = True

OUTPUT_DIR = "../results/baseline_results"
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

# remove mega-cap / superstar stocks
# REMOVE = [
#     "NVDA", "MSFT", "AAPL", "META", "AMZN",
#     "AVGO", "GOOGL", "GOOG", "TSLA", "LLY",
#     "NFLX", "COST", "AMD", "ADBE", "CRM",
#     "ORCL", "NOW", "PANW", "ANET", "CDNS",
#     "SNPS", "BKNG", "CMG", "KLAC", "LRCX"
# ]

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

# In case REMOVE is not defined
remove_set = set(globals().get("REMOVE", []))
TICKERS = [t for t in TICKERS if t not in remove_set]

# ============================================================
# DATA LOADING AND FEATURE ENGINEERING
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

    # Original short/medium returns, plus long-momentum windows.
    for w in [1, 2, 5, 10, 20, 60, 120, 252]:
        df[f"ret_{w}d"] = close.pct_change(w)

    # Long-term momentum features. These use only past prices.
    df["momentum_3m"] = df["ret_60d"]
    df["momentum_6m"] = df["ret_120d"]
    df["momentum_12m"] = df["ret_252d"]
    # 12-month momentum skipping the most recent 1 month, common in factor models.
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


# ============================================================
# MARKET DATA
# ============================================================

spy = load_ohlcv(os.path.join(DATA_DIR, "SPY.csv"))
spy = spy.rename(columns={
    "open": "spy_open",
    "high": "spy_high",
    "low": "spy_low",
    "close": "spy_close",
    "volume": "spy_volume"
})

spy["spy_ret_1d"] = spy["spy_close"].pct_change(1)
for w in [5, 10, 20, 60, 120, 252]:
    spy[f"spy_ret_{w}d"] = spy["spy_close"].pct_change(w)

for w in [5, 10, 20, 60]:
    spy[f"spy_vol_{w}d"] = spy["spy_ret_1d"].rolling(w).std() * np.sqrt(252)

spy[f"spy_future_return_{HORIZON}d"] = (
    spy["spy_close"].shift(-HORIZON) / spy["spy_close"] - 1
)

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
    market.sort_values("date"),
    macro.sort_values("date"),
    on="date",
    direction="backward"
)
# ============================================================
# BUILD PANEL
# ============================================================

parts = []
skipped = []

for ticker in TICKERS:
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        skipped.append((ticker, "file not found"))
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
                    on="date",
                    direction="backward"
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

    except Exception as e:
        skipped.append((ticker, str(e)))

if not parts:
    raise RuntimeError("No stock files loaded. Check DATA_DIR and ticker files.")

panel = pd.concat(parts, ignore_index=True)
panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

panel["future_return_rank_pct"] = (
    panel.groupby("date")[f"future_return_{HORIZON}d"].rank(pct=True)
)
panel[f"relative_future_return_{HORIZON}d"] = (
    panel[f"future_return_{HORIZON}d"]
    - panel[f"spy_future_return_{HORIZON}d"]
)

# ============================================================
# FEATURE LIST
# ============================================================

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

rates_features = [
    "us10y", "us2y", "us3m",
    "yield_spread_10y_2y", "yield_spread_10y_3m",
    "us10y_change_20d", "us10y_change_60d",
    "yield_spread_10y_2y_change_20d",
    "yield_spread_10y_2y_change_60d",
]

risk_features = [
    "spy_tr_drawdown_252d",
    "spy_tr_vol_20d", "spy_tr_vol_60d",
    "vix_change_20d", "vix_change_60d",
]

cross_asset_features = [
    "us_agg_bond_ret_20d", "us_agg_bond_ret_60d",
    "dxy_ret_20d", "dxy_ret_60d",
    "gold_ret_20d", "gold_ret_60d",
    "oil_ret_20d", "oil_ret_60d",
]

macro_cycle_features = [
    "fed_funds", "fed_funds_change_20d", "fed_funds_change_60d",
    "cpi_yoy", "cpi_yoy_change_60d",
    "unemployment", "unemployment_change_60d",
    "ism_pmi", "ism_pmi_change_60d",
    "consumer_confidence", "consumer_confidence_change_60d",
]

numeric_features = original_features + momentum_features + fundamental_features + cross_asset_features
# Remove accidental duplicates while preserving order.
numeric_features = list(dict.fromkeys(numeric_features))

panel = panel.replace([np.inf, -np.inf], np.nan)

if USE_FUNDAMENTALS:
    for col in fundamental_features:
        if col in panel.columns:
            panel[col] = panel.groupby("ticker")[col].ffill()
            panel[col] = panel.groupby("date")[col].transform(lambda x: x.fillna(x.median()))


available_features = [c for c in numeric_features if c in panel.columns]
missing_features = [c for c in numeric_features if c not in panel.columns]

panel = panel.dropna(subset=available_features + [
    f"future_return_{HORIZON}d",
    f"spy_future_return_{HORIZON}d",
    f"relative_future_return_{HORIZON}d",
    "future_return_rank_pct"
])

panel_model = panel.copy()
feature_cols = available_features


# ============================================================
# SPLIT
# ============================================================

train = panel_model[(panel_model["date"] >= TRAIN_START) & (panel_model["date"] <= TRAIN_END)]
val = panel_model[(panel_model["date"] >= VAL_START) & (panel_model["date"] <= VAL_END)]
test = panel_model[(panel_model["date"] >= TEST_START) & (panel_model["date"] <= TEST_END)]

X_train = train[feature_cols]
y_train = train["future_return_rank_pct"]
X_val = val[feature_cols]
y_val = val["future_return_rank_pct"]
X_test = test[feature_cols]
y_test = test["future_return_rank_pct"]

# ============================================================
# MODEL TRAINING
# ============================================================

SEEDS = [1, 10, 42, 123, 500, 999, 2025]

val_preds = []
test_preds = []
models = []

for seed in SEEDS:

    model = XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    val_preds.append(model.predict(X_val))
    test_preds.append(model.predict(X_test))

    models.append(model)

val_pred = np.mean(val_preds, axis=0)
test_pred = np.mean(test_preds, axis=0)

model = models[0]


# ============================================================
# PREDICTIONS
# ============================================================

predictions = test[[
    "date", "ticker",
    f"future_return_{HORIZON}d",
    f"spy_future_return_{HORIZON}d",
    "future_return_rank_pct"
] + feature_cols].copy()

predictions["predicted_return_score"] = test_pred
predictions.to_csv(os.path.join(OUTPUT_DIR, "test_predictions_rank_model.csv"), index=False)


#SHAP ANALYSIS

# Use one trained XGBoost model from the ensemble
shap_model = models[0]

# Sample test data to avoid memory issues
X_shap = X_test.sample(n=min(5000, len(X_test)), random_state=42)

explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(X_shap)

# 1. Global feature importance plot
shap.summary_plot(
    shap_values,
    X_shap,
    feature_names=feature_cols,
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_plot.png"), dpi=300, bbox_inches="tight")
plt.close()

# 2. Bar plot of mean absolute SHAP values
shap.summary_plot(
    shap_values,
    X_shap,
    feature_names=feature_cols,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar_importance.png"), dpi=300, bbox_inches="tight")
plt.close()

# 3. Save SHAP importance table
shap_importance = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

shap_importance.to_csv(
    os.path.join(OUTPUT_DIR, "shap_feature_importance.csv"),
    index=False
)

print("\nTop SHAP features:")
print(shap_importance.head(20))


# ============================================================
# PORTFOLIO FUNCTIONS
# ============================================================

def monthly_rebalance_dates(data):
    dates = data[["date"]].drop_duplicates().sort_values("date")
    dates["year_month"] = dates["date"].dt.to_period("M")
    return dates.groupby("year_month")["date"].min().tolist()


def max_drawdown(equity):
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return drawdown.min()


def performance_metrics(returns, benchmark_returns=None):
    returns = returns.dropna()
    equity = (1 + returns).cumprod()

    total_return = equity.iloc[-1] - 1
    years = len(returns) / PERIODS_PER_YEAR
    cagr = equity.iloc[-1] ** (1 / years) - 1

    annual_return = returns.mean() * PERIODS_PER_YEAR
    annual_vol = returns.std() * np.sqrt(PERIODS_PER_YEAR)
    sharpe = (annual_return - RISK_FREE_RATE) / annual_vol if annual_vol != 0 else np.nan
    mdd = max_drawdown(equity)

    out = {
        "total_return": total_return,
        "CAGR": cagr,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "n_periods": len(returns)
    }

    if benchmark_returns is not None:
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ["strategy", "spy"]
        excess = aligned["strategy"] - aligned["spy"]
        out["annual_excess_return_vs_spy"] = excess.mean() * PERIODS_PER_YEAR

        x = aligned["spy"].values
        y = aligned["strategy"].values
        beta = np.cov(y, x)[0, 1] / np.var(x)
        alpha_periodic = y.mean() - beta * x.mean()
        out["beta_vs_spy"] = beta
        out["alpha_vs_spy"] = alpha_periodic * PERIODS_PER_YEAR

    return out


def build_top_bottom_portfolios(data, score_col="predicted_return_score"):
    rows = []
    dates = monthly_rebalance_dates(data)

    for date in dates:
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

        actual_top = set(g.sort_values(f"future_return_{HORIZON}d", ascending=False).head(TOP_N)["ticker"])
        predicted_top = set(top["ticker"])
        hits = len(actual_top.intersection(predicted_top))

        spearman_ic = spearmanr(g[score_col], g[f"future_return_{HORIZON}d"]).correlation

        rows.append({
            "date": date,
            "top10_return": top[f"future_return_{HORIZON}d"].mean(),
            "bottom20_return": bottom[f"future_return_{HORIZON}d"].mean(),
            "random_top10_return": random_return,
            "top_minus_bottom": top[f"future_return_{HORIZON}d"].mean() - bottom[f"future_return_{HORIZON}d"].mean(),
            "spy_return": g[f"spy_future_return_{HORIZON}d"].iloc[0],
            "oracle_return": g.sort_values(f"future_return_{HORIZON}d", ascending=False).head(TOP_N)[f"future_return_{HORIZON}d"].mean(),
            "top10_hits": hits,
            "top10_hit_rate": hits / TOP_N,
            "spearman_ic": spearman_ic,
            "top10_tickers": ",".join(top["ticker"].tolist()),
            "bottom20_tickers": ",".join(bottom["ticker"].tolist())
        })

    return pd.DataFrame(rows).sort_values("date")


portfolio = build_top_bottom_portfolios(predictions, score_col="predicted_return_score")
portfolio.to_csv(os.path.join(OUTPUT_DIR, "monthly_rank_model_portfolio_returns.csv"), index=False)
portfolio_i = portfolio.set_index("date")


# ============================================================
# METRICS AND DIAGNOSTICS
# ============================================================

performance_rows = []
for name, col in [
    ("RankModel_Top10", "top10_return"),
    ("SPY", "spy_return"),
    ("bottom20_Predicted", "bottom20_return"),
    ("TopMinusBottom", "top_minus_bottom"),
    ("Oracle_Top10", "oracle_return")
]:
    if name == "TopMinusBottom":
        metrics = performance_metrics(portfolio_i[col])
    else:
        metrics = performance_metrics(portfolio_i[col], benchmark_returns=portfolio_i["spy_return"])
    metrics["portfolio"] = name
    performance_rows.append(metrics)

performance_df = pd.DataFrame(performance_rows)
ordered_cols = [
    "portfolio", "total_return", "CAGR", "annual_return", "annual_volatility", "sharpe",
    "max_drawdown", "annual_excess_return_vs_spy", "beta_vs_spy", "alpha_vs_spy", "n_periods"
]
for c in ordered_cols:
    if c not in performance_df.columns:
        performance_df[c] = np.nan
performance_df = performance_df[ordered_cols]
performance_df.to_csv(os.path.join(OUTPUT_DIR, "portfolio_performance_metrics.csv"), index=False)

summary = {
    "n_months": len(portfolio),
    "pearson_corr": pearsonr(predictions["predicted_return_score"], predictions[f"future_return_{HORIZON}d"])[0],
    "pearson_p_value": pearsonr(predictions["predicted_return_score"], predictions[f"future_return_{HORIZON}d"])[1],
    "r2": r2_score(predictions[f"future_return_{HORIZON}d"], predictions["predicted_return_score"]),
    "avg_top10_return": portfolio["top10_return"].mean(),
    "avg_bottom20_return": portfolio["bottom20_return"].mean(),
    "avg_top_minus_bottom": portfolio["top_minus_bottom"].mean(),
    "annualized_top_minus_bottom": portfolio["top_minus_bottom"].mean() * PERIODS_PER_YEAR,
    "top_minus_bottom_t_stat": ttest_1samp(portfolio["top_minus_bottom"], 0).statistic,
    "top_minus_bottom_p_value": ttest_1samp(portfolio["top_minus_bottom"], 0).pvalue,
    "avg_top10_excess_vs_random": (portfolio["top10_return"] - portfolio["random_top10_return"]).mean(),
    "annualized_top10_excess_vs_random": (portfolio["top10_return"] - portfolio["random_top10_return"]).mean() * PERIODS_PER_YEAR,
    "top_minus_random_p_value": ttest_1samp(portfolio["top10_return"] - portfolio["random_top10_return"], 0).pvalue,
    "avg_top10_excess_vs_spy": (portfolio["top10_return"] - portfolio["spy_return"]).mean(),
    "annualized_top10_excess_vs_spy": (portfolio["top10_return"] - portfolio["spy_return"]).mean() * PERIODS_PER_YEAR,
    "top_minus_spy_p_value": ttest_1samp(portfolio["top10_return"] - portfolio["spy_return"], 0).pvalue,
    "percent_months_top_beats_bottom": (portfolio["top_minus_bottom"] > 0).mean(),
    "percent_months_top_beats_spy": (portfolio["top10_return"] > portfolio["spy_return"]).mean(),
    "avg_spearman_ic": portfolio["spearman_ic"].mean(),
    "median_spearman_ic": portfolio["spearman_ic"].median(),
    "percent_positive_ic": (portfolio["spearman_ic"] > 0).mean(),
    "spearman_ic_t_stat": ttest_1samp(portfolio["spearman_ic"].dropna(), 0).statistic,
    "spearman_ic_p_value": ttest_1samp(portfolio["spearman_ic"].dropna(), 0).pvalue,
    "avg_top10_hits": portfolio["top10_hits"].mean(),
    "avg_top10_hit_rate": portfolio["top10_hit_rate"].mean(),
    "n_train_rows": len(train),
    "n_val_rows": len(val),
    "n_test_rows": len(test),
    "n_features": len(feature_cols),
    "n_stocks_loaded": panel["ticker"].nunique()
}

diagnostics_df = pd.DataFrame([summary])
diagnostics_df.to_csv(os.path.join(OUTPUT_DIR, "ranking_diagnostics_summary.csv"), index=False)

# Validation/test IC by daily date.
def daily_ic(df, pred_col, target_col):
    return df.groupby("date").apply(lambda x: spearmanr(x[pred_col], x[target_col]).correlation)

val_predictions = val[["date", f"future_return_{HORIZON}d"]].copy()
val_predictions["predicted_return_score"] = val_pred
val_ic_series = daily_ic(val_predictions, "predicted_return_score", f"future_return_{HORIZON}d")
val_ic_series.to_csv(os.path.join(OUTPUT_DIR, "validation_daily_ic.csv"), header=["spearman_ic"])

test_ic_series = daily_ic(predictions, "predicted_return_score", f"future_return_{HORIZON}d")
test_ic_series.to_csv(os.path.join(OUTPUT_DIR, "test_daily_ic.csv"), header=["spearman_ic"])

monthly_ic = portfolio[["date", "spearman_ic", "top_minus_bottom", "top10_return", "bottom20_return", "spy_return"]].copy()
monthly_ic.to_csv(os.path.join(OUTPUT_DIR, "monthly_ic_and_spread.csv"), index=False)

# Yearly returns.
yearly_rows = []
portfolio["year"] = portfolio["date"].dt.year
for year, g in portfolio.groupby("year"):
    yearly_rows.append({
        "year": year,
        "rank_model_return": (1 + g["top10_return"]).prod() - 1,
        "spy_return": (1 + g["spy_return"]).prod() - 1,
        "bottom20_return": (1 + g["bottom20_return"]).prod() - 1,
        "top_minus_bottom_return": (1 + g["top_minus_bottom"]).prod() - 1,
        "excess_vs_spy": (1 + g["top10_return"]).prod() - (1 + g["spy_return"]).prod()
    })
yearly_df = pd.DataFrame(yearly_rows)
yearly_df.to_csv(os.path.join(OUTPUT_DIR, "yearly_returns.csv"), index=False)

# Yearly IC based on monthly rebalance dates and full daily prediction panel.
yearly_ic_monthly = portfolio.groupby(portfolio["date"].dt.year)["spearman_ic"].mean().reset_index()
yearly_ic_monthly.columns = ["year", "mean_monthly_rebalance_ic"]
yearly_ic_daily = predictions.groupby(predictions["date"].dt.year).apply(
    lambda g: spearmanr(g["predicted_return_score"], g[f"future_return_{HORIZON}d"]).correlation
).reset_index()
yearly_ic_daily.columns = ["year", "daily_panel_ic"]
yearly_ic = yearly_ic_monthly.merge(yearly_ic_daily, on="year", how="outer")
yearly_ic.to_csv(os.path.join(OUTPUT_DIR, "yearly_ic.csv"), index=False)

# Decile analysis at monthly rebalance dates.
decile_rows = []
for date in monthly_rebalance_dates(predictions):
    g = predictions[predictions["date"] == date].copy()
    if len(g) < 50:
        continue
    g["predicted_decile"] = pd.qcut(g["predicted_return_score"], 10, labels=False, duplicates="drop") + 1
    for decile, group in g.groupby("predicted_decile"):
        decile_rows.append({
            "date": date,
            "year": date.year,
            "predicted_decile": decile,
            "future_return": group[f"future_return_{HORIZON}d"].mean()
        })

deciles = pd.DataFrame(decile_rows)
deciles.to_csv(os.path.join(OUTPUT_DIR, "decile_returns_by_rebalance_date.csv"), index=False)

decile_summary = deciles.groupby("predicted_decile")["future_return"].agg(["mean", "std", "count"]).reset_index()
decile_summary["annualized_mean_return"] = decile_summary["mean"] * PERIODS_PER_YEAR
decile_summary.to_csv(os.path.join(OUTPUT_DIR, "decile_return_summary.csv"), index=False)

decile_by_year = deciles.groupby(["year", "predicted_decile"])["future_return"].mean().reset_index()
decile_by_year["annualized_mean_return"] = decile_by_year["future_return"] * PERIODS_PER_YEAR
decile_by_year.to_csv(os.path.join(OUTPUT_DIR, "decile_return_summary_by_year.csv"), index=False)

# Feature importance.
feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
feature_importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_rank_model.csv"), index=False)

# Ticker concentration.
selected = []
for tickers in portfolio["top10_tickers"]:
    selected.extend(tickers.split(","))
concentration = pd.Series(selected).value_counts().reset_index()
concentration.columns = ["ticker", "selection_count"]
concentration["selection_frequency"] = concentration["selection_count"] / len(portfolio)
concentration["portfolio_slot_share"] = concentration["selection_count"] / len(selected)
concentration.to_csv(os.path.join(OUTPUT_DIR, "ticker_concentration.csv"), index=False)

# Top/bottom selected stock details.
selection_rows = []
for date in portfolio["date"]:
    g = predictions[predictions["date"] == date].sort_values("predicted_return_score", ascending=False).copy()
    top = g.head(TOP_N).copy()
    bottom = g.tail(TOP_N).copy()
    top["bucket"] = "TOP"
    bottom["bucket"] = "BOTTOM"
    selection_rows.append(pd.concat([top, bottom], axis=0))
selection_details = pd.concat(selection_rows, ignore_index=True)
selection_details.to_csv(os.path.join(OUTPUT_DIR, "top_bottom_stock_details.csv"), index=False)


# ============================================================
# PLOTS
# ============================================================

def save_cumulative_returns_plot():
    plt.figure(figsize=(12, 7))
    series = {
        "Rank Model Top 10": portfolio_i["top10_return"],
        "SPY": portfolio_i["spy_return"],
        "Bottom 20 Predicted": portfolio_i["bottom20_return"]
    }
    for name, returns in series.items():
        equity = (1 + returns.dropna()).cumprod()
        plt.plot(equity.index, equity.values, label=name)
    plt.title("Cumulative Portfolio Performance")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "cumulative_portfolio_performance.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_yearly_returns_plot():
    x = np.arange(len(yearly_df))
    width = 0.22
    plt.figure(figsize=(12, 7))
    plt.bar(x - width, yearly_df["rank_model_return"], width, label="Rank Model Top 10")
    plt.bar(x, yearly_df["spy_return"], width, label="SPY")
    plt.axhline(0, linewidth=1)
    plt.xticks(x, yearly_df["year"].astype(str))
    plt.ylabel("Annual Return")
    plt.title("Yearly Returns: Rank Model vs Benchmarks")
    plt.legend()
    plt.grid(True, axis="y")
    plt.savefig(os.path.join(OUTPUT_DIR, "yearly_returns_vs_benchmarks.png"), dpi=300, bbox_inches="tight")
    plt.close()

def save_top_bottom_plot():
    plt.figure(figsize=(12, 7))
    for col, label in [
        ("top10_return", "Top 10 Predicted"),
        ("bottom20_return", "Bottom 10 Predicted"),
        ("top_minus_bottom", "Top - Bottom Spread")
    ]:
        equity = (1 + portfolio_i[col].dropna()).cumprod()
        plt.plot(equity.index, equity.values, label=label)
    plt.title("Top 10 vs Bottom 10 Ranking Test")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "top_vs_bottom_ranking_test.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_decile_plot(df, filename, title):
    plt.figure(figsize=(10, 6))
    plt.bar(df["predicted_decile"], df["annualized_mean_return"])
    plt.xlabel("Predicted score decile, 1 = lowest, 10 = highest")
    plt.ylabel("Annualized average return")
    plt.title(title)
    plt.grid(True, axis="y")
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


def save_monthly_ic_plot():
    plt.figure(figsize=(12, 5))
    plt.plot(monthly_ic["date"], monthly_ic["spearman_ic"], marker="o")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Monthly Information Coefficient at Rebalance Dates")
    plt.xlabel("Date")
    plt.ylabel("Spearman IC")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "monthly_information_coefficient.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_feature_importance_plot():
    top = feature_importance.head(30).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 9))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Feature importance")
    plt.title("Top 30 XGBoost Feature Importances")
    plt.grid(True, axis="x")
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_top30.png"), dpi=300, bbox_inches="tight")
    plt.close()


save_cumulative_returns_plot()
save_yearly_returns_plot()
save_top_bottom_plot()
save_decile_plot(decile_summary, "decile_returns.png", "Future Returns by Predicted Score Decile")
for year in sorted(decile_by_year["year"].unique()):
    save_decile_plot(
        decile_by_year[decile_by_year["year"] == year],
        f"decile_returns_{year}.png",
        f"Future Returns by Predicted Score Decile: {year}"
    )
save_monthly_ic_plot()
save_feature_importance_plot()


# ============================================================
# FINAL CONSOLE SUMMARY
# ============================================================

print("\nFinal model: Bloomberg fundamentals + long momentum")
print("================================================")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Stocks loaded: {panel['ticker'].nunique()} | Features: {len(feature_cols)}")
print(f"Train rows: {len(train)} | Validation rows: {len(val)} | Test rows: {len(test)}")
if skipped:
    print(f"Skipped tickers: {len(skipped)}")
if missing_features:
    print(f"Missing requested features ignored: {missing_features}")

print("\nPortfolio performance:")
print(performance_df.round(4))

print("\nRanking diagnostics:")
print(diagnostics_df.T.round(4))

print("\nMost important saved files:")
for fname in [
    "portfolio_performance_metrics.csv",
    "ranking_diagnostics_summary.csv",
    "monthly_rank_model_portfolio_returns.csv",
    "yearly_returns.csv",
    "yearly_ic.csv",
    "decile_return_summary.csv",
    "decile_return_summary_by_year.csv",
    "feature_importance_rank_model.csv",
    "ticker_concentration.csv",
    "cumulative_portfolio_performance.png",
    "yearly_returns_vs_benchmarks.png",
    "decile_returns.png",
    "monthly_information_coefficient.png",
    "feature_importance_top30.png"
]:
    print(f"- {fname}")
