import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM


# 1. Load merged Bloomberg data

df = pd.read_csv("../data_HMM/merged_market_data.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df = df.set_index("date")

# 2. Feature engineering

# Daily log return
df["log_return"] = np.log(
    df["sp500_close"] /
    df["sp500_close"].shift(1)
)

# 20-day rolling volatility
df["volatility_20d"] = (
    df["log_return"]
    .rolling(20)
    .std()
    * np.sqrt(252)
)

# 60-day momentum
df["momentum_60d"] = np.log(
    df["sp500_close"] /
    df["sp500_close"].shift(60)
)

# Drawdown
rolling_max = df["sp500_close"].rolling(252).max()

df["drawdown"] = (
    df["sp500_close"] /
    rolling_max
    - 1
)

# VIX level
df["vix_level"] = df["vix_close"]

# Yield spread
df["yield_spread_10y_2y"] = (
    df["us10y_yield"]
    - df["us2y_yield"]
)

# Clean
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()


# 3. Features used by final HMM

feature_cols = [
    "log_return",
    "volatility_20d",
    "momentum_60d",
    "drawdown",
    "vix_level",
    "yield_spread_10y_2y"
]

X = df[feature_cols]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# 4. Fit 4-state Gaussian HMM

model = GaussianHMM(
    n_components=4,
    covariance_type="full",
    n_iter=2000,
    random_state=42,
    min_covar=1e-3
)

model.fit(X_scaled)

df["regime"] = model.predict(X_scaled)

# 5. Regime interpretation and labelled plot

def regime_summary(df_model):
    summary = df_model.groupby("regime").apply(lambda x: pd.Series({
        "annual_return": x["log_return"].mean() * 252,
        "annual_volatility": x["log_return"].std() * np.sqrt(252),
        "avg_rolling_volatility": x["volatility_20d"].mean(),
        "avg_drawdown": x["drawdown"].mean(),
        "avg_vix": x["vix_level"].mean(),
        "avg_momentum_60d": x["momentum_60d"].mean(),
        "avg_yield_spread": x["yield_spread_10y_2y"].mean(),
        "count": len(x)
    }))
    return summary


summary = regime_summary(df)

# Automatically infer economic regime labels from the data
def assign_regime_labels(summary):
    labels = {}

    # Crisis: highest VIX / highest volatility / deepest drawdown
    crisis_regime = (
        summary["avg_vix"]
        .rank(ascending=False)
        + summary["annual_volatility"].rank(ascending=False)
        + summary["avg_drawdown"].rank(ascending=True)
    ).idxmin()

    labels[crisis_regime] = "Crisis / Extreme Stress"

    remaining = summary.drop(index=crisis_regime)

    # Bull market: strongest returns and momentum, low drawdowns
    bull_regime = (
        remaining["annual_return"].rank(ascending=False)
        + remaining["avg_momentum_60d"].rank(ascending=False)
        + remaining["avg_drawdown"].rank(ascending=False)
    ).idxmin()

    labels[bull_regime] = "Bull Market"

    remaining = remaining.drop(index=bull_regime)

    # Correction / bear market: weakest return and momentum among remaining
    bear_regime = (
        remaining["annual_return"].rank(ascending=True)
        + remaining["avg_momentum_60d"].rank(ascending=True)
        + remaining["avg_drawdown"].rank(ascending=True)
    ).idxmin()

    labels[bear_regime] = "Correction / Bear Market"

    remaining = remaining.drop(index=bear_regime)

    # Remaining state is the normal/intermediate regime
    normal_regime = remaining.index[0]
    labels[normal_regime] = "Normal Expansion"

    return labels


regime_names = assign_regime_labels(summary)

summary["interpreted_label"] = summary.index.map(regime_names)

print("\nRegime summary:")
print(summary)

print("\nAutomatically inferred regime labels:")
for regime, label in regime_names.items():
    print(f"Regime {regime}: {label}")


plt.figure(figsize=(14, 6))

for regime in sorted(df["regime"].unique()):
    subset = df[df["regime"] == regime]

    plt.scatter(
        subset.index,
        subset["sp500_close"],
        s=6,
        alpha=0.7,
        label=f"Regime {regime}: {regime_names[regime]}"
    )

plt.title("S&P 500 Regimes Detected by 4-State Gaussian HMM")
plt.xlabel("Date")
plt.ylabel("S&P 500 Close")
plt.legend()
plt.tight_layout()

plt.savefig(
    "../results/full_hmm/regimes_price_plot_4_states_labelled.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 6. Save regime labels

regimes = (
    df.reset_index()[["date", "regime"]]
)

regimes.to_csv(
    "../results/full_hmm/hmm_4_state_regimes.csv",
    index=False
)

regimes.to_excel(
    "../results/full_hmm/hmm_4_state_regimes.xlsx",
    index=False
)

print("\nSaved:")
print("../results/full_hmm/hmm_4_state_regimes.csv")
print("../results/full_hmm/hmm_4_state_regimes.xlsx")