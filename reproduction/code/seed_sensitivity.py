# Seed-sensitivity check (not in the paper; raised by another replication):
# retrain the 7-seed XGBoost ensemble with alternative seed sets on the same
# panel and re-run the Top10 evaluation. Reuses baseline.py's panel and
# feature construction verbatim by executing its source up to model training.

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp
from xgboost import XGBRegressor

src = open("baseline.py").read()
head = src.split("SEEDS = [")[0]
ns = {}
exec(compile(head, "baseline_head", "exec"), ns)
train, test = ns["train"], ns["test"]
feature_cols = ns["feature_cols"]
HORIZON, TOP_N, BOTTOM_N = ns["HORIZON"], ns["TOP_N"], ns["BOTTOM_N"]

OUT = "../results/seed_sensitivity"
os.makedirs(OUT, exist_ok=True)

SEED_SETS = {
    "paper_seeds": [1, 10, 42, 123, 500, 999, 2025],
    "set_A": [7, 77, 777, 3, 33, 333, 2026],
    "set_B": [11, 22, 44, 88, 176, 352, 704],
    "set_C": [101, 202, 303, 404, 505, 606, 707],
}

X_train, y_train = train[feature_cols], train["future_return_rank_pct"]
X_test = test[feature_cols]


def monthly_rebalance_dates(data):
    d = data[["date"]].drop_duplicates().sort_values("date")
    d["ym"] = d["date"].dt.to_period("M")
    return d.groupby("ym")["date"].min().tolist()


def evaluate(scored):
    rows = []
    for date in monthly_rebalance_dates(scored):
        g = scored[scored["date"] == date].dropna(subset=["score", f"future_return_{HORIZON}d"])
        if len(g) < TOP_N + BOTTOM_N:
            continue
        g = g.sort_values("score", ascending=False)
        top, bottom = g.head(TOP_N), g.tail(BOTTOM_N)
        rnd = np.mean([g.sample(n=TOP_N, random_state=42 + i)[f"future_return_{HORIZON}d"].mean() for i in range(1000)])
        rows.append({"date": date, "top": top[f"future_return_{HORIZON}d"].mean(),
                     "bottom": bottom[f"future_return_{HORIZON}d"].mean(), "rnd": rnd,
                     "spy": g[f"spy_future_return_{HORIZON}d"].iloc[0],
                     "ic": spearmanr(g["score"], g[f"future_return_{HORIZON}d"]).correlation})
    r = pd.DataFrame(rows)
    eq = (1 + r["top"]).cumprod()
    ann, vol = r["top"].mean() * 12, r["top"].std() * np.sqrt(12)
    return {
        "CAGR": eq.iloc[-1] ** (12 / len(r)) - 1, "sharpe": (ann - 0.04) / vol,
        "max_drawdown": (eq / eq.cummax() - 1).min(),
        "top_minus_spy_p": ttest_1samp(r["top"] - r["spy"], 0).pvalue,
        "top_minus_random_p": ttest_1samp(r["top"] - r["rnd"], 0).pvalue,
        "top_minus_bottom_p": ttest_1samp(r["top"] - r["bottom"], 0).pvalue,
        "avg_ic": r["ic"].mean(), "ic_p": ttest_1samp(r["ic"].dropna(), 0).pvalue,
    }, r


results = []
for name, seeds in SEED_SETS.items():
    if name == "paper_seeds":
        base = pd.read_csv("../results/baseline_results/test_predictions_rank_model.csv",
                           usecols=["date", "ticker", "predicted_return_score"], parse_dates=["date"])
        scored = test[["date", "ticker", f"future_return_{HORIZON}d", f"spy_future_return_{HORIZON}d"]].merge(
            base, on=["date", "ticker"]).rename(columns={"predicted_return_score": "score"})
    else:
        preds = []
        for s in seeds:
            m = XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=10, reg_lambda=2.0,
                             objective="reg:squarederror", random_state=s, n_jobs=-1)
            m.fit(X_train, y_train)
            preds.append(m.predict(X_test))
        scored = test[["date", "ticker", f"future_return_{HORIZON}d", f"spy_future_return_{HORIZON}d"]].copy()
        scored["score"] = np.mean(preds, axis=0)
    metrics, monthly = evaluate(scored)
    metrics["seed_set"] = name
    metrics["seeds"] = str(seeds)
    results.append(metrics)
    monthly.to_csv(f"{OUT}/monthly_{name}.csv", index=False)
    pd.DataFrame(results).to_csv(f"{OUT}/seed_sensitivity_summary.csv", index=False)
    print(name, {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}, flush=True)

print(pd.DataFrame(results).round(4).to_string(index=False))
