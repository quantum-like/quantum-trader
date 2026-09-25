import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

INPUT_PATH = "../data/merged_market_data.csv"
OUTPUT_DIR = "../data"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "hmm_4_state_regimes_expanding.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_START = "2005-01-01"
FIRST_PREDICT_DATE = "2008-01-01"
N_COMPONENTS = 4
MIN_TRAIN_OBS = 500      # about 3 years of trading days
RETRAIN_EVERY = 20      # refit roughly monthly, much faster than daily

df = pd.read_csv(INPUT_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")

df["log_return"] = np.log(df["sp500_close"] / df["sp500_close"].shift(1))

df["volatility_20d"] = (
    df["log_return"]
    .rolling(20)
    .std()
    * np.sqrt(252)
)

df["momentum_60d"] = np.log(df["sp500_close"] / df["sp500_close"].shift(60))

rolling_max = df["sp500_close"].rolling(252).max()
df["drawdown"] = df["sp500_close"] / rolling_max - 1

df["vix_level"] = df["vix_close"]

df["yield_spread_10y_2y"] = df["us10y_yield"] - df["us2y_yield"]

feature_cols = [
    "log_return",
    "volatility_20d",
    "momentum_60d",
    "drawdown",
    "vix_level",
    "yield_spread_10y_2y"
]

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=feature_cols).copy()

dates = df.index.to_list()

rows = []
last_model = None
last_scaler = None

for i, current_date in enumerate(dates):

    if current_date < pd.to_datetime(FIRST_PREDICT_DATE):
        continue

    train_data = df.loc[:current_date, feature_cols].copy()

    if len(train_data) < MIN_TRAIN_OBS:
        rows.append({
            "date": current_date,
            "hmm_regime": np.nan,
            "n_train_obs": len(train_data)
        })
        continue

    should_refit = (
        last_model is None
        or i % RETRAIN_EVERY == 0
    )

    if should_refit:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_data)

        model = GaussianHMM(
            n_components=N_COMPONENTS,
            covariance_type="full",
            n_iter=2000,
            random_state=42,
            min_covar=1e-3
        )

        model.fit(X_train_scaled)

        last_model = model
        last_scaler = scaler

    X_until_now_scaled = last_scaler.transform(train_data)
    regime_today = int(last_model.predict(X_until_now_scaled)[-1])

    rows.append({
        "date": current_date,
        "hmm_regime": regime_today,
        "n_train_obs": len(train_data)
    })

    if len(rows) % 250 == 0:
        print(f"Processed {len(rows)} dates. Current date: {current_date.date()}")

regimes = pd.DataFrame(rows)
regimes.to_csv(OUTPUT_PATH, index=False)

# ============================================================
# REGIME CHARACTERISTICS AND ECONOMIC LABELLING
# ============================================================

regime_chars = (
    regimes
    .dropna(subset=["hmm_regime"])
    .copy()
)

regime_chars["date"] = pd.to_datetime(regime_chars["date"])
regime_chars["hmm_regime"] = regime_chars["hmm_regime"].astype(int)

# df already contains the HMM input features and has date as index.
feature_data = df[feature_cols].reset_index().copy()

regime_chars = pd.merge(
    regime_chars,
    feature_data,
    on="date",
    how="left"
)

summary = (
    regime_chars
    .groupby("hmm_regime")
    .agg(
        n_obs=("date", "count"),
        start_date=("date", "min"),
        end_date=("date", "max"),
        avg_log_return=("log_return", "mean"),
        ann_log_return=("log_return", lambda x: x.mean() * 252),
        avg_volatility_20d=("volatility_20d", "mean"),
        avg_momentum_60d=("momentum_60d", "mean"),
        avg_drawdown=("drawdown", "mean"),
        worst_drawdown=("drawdown", "min"),
        avg_vix=("vix_level", "mean"),
        avg_yield_spread_10y_2y=("yield_spread_10y_2y", "mean"),
    )
    .reset_index()
)

# Higher stress_score = more crisis-like.
summary["stress_score"] = (
    summary["avg_volatility_20d"].rank(pct=True)
    + summary["avg_vix"].rank(pct=True)
    + (-summary["avg_drawdown"]).rank(pct=True)
    + (-summary["avg_momentum_60d"]).rank(pct=True)
)

# Higher bull_score = more bull-market-like.
summary["bull_score"] = (
    summary["ann_log_return"].rank(pct=True)
    + summary["avg_momentum_60d"].rank(pct=True)
    + (-summary["avg_volatility_20d"]).rank(pct=True)
    + (-summary["avg_vix"]).rank(pct=True)
    + summary["avg_drawdown"].rank(pct=True)
)

# Assign simple economic labels.
summary["economic_label"] = "Unlabelled"

crisis_regime = summary.loc[summary["stress_score"].idxmax(), "hmm_regime"]
bull_regime = summary.loc[summary["bull_score"].idxmax(), "hmm_regime"]

summary.loc[summary["hmm_regime"] == crisis_regime, "economic_label"] = "Crisis / Extreme Stress"
summary.loc[summary["hmm_regime"] == bull_regime, "economic_label"] = "Bull Market"

remaining = summary[summary["economic_label"] == "Unlabelled"].copy()

if len(remaining) == 2:
    # Of the remaining regimes, the one with higher stress is labelled Correction/Bear.
    bear_regime = remaining.loc[remaining["stress_score"].idxmax(), "hmm_regime"]
    normal_regime = remaining.loc[remaining["stress_score"].idxmin(), "hmm_regime"]

    summary.loc[summary["hmm_regime"] == bear_regime, "economic_label"] = "Correction / Bear Market"
    summary.loc[summary["hmm_regime"] == normal_regime, "economic_label"] = "Normal Expansion"

summary = summary.sort_values("hmm_regime")

CHARACTERISTICS_PATH = os.path.join(OUTPUT_DIR, "hmm_4_state_regime_characteristics_expanding.csv")
summary.to_csv(CHARACTERISTICS_PATH, index=False)

print("\nExpanding-window HMM regime characteristics:")
print(summary.round(4))

print("\nSaved regime characteristics:")
print(CHARACTERISTICS_PATH)

print("\nSaved expanding-window HMM regimes:")
print(OUTPUT_PATH)
print(regimes.head())
print(regimes.tail())