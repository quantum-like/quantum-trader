# Assemble reproduced metrics next to the paper's reported numbers.
# Writes ../results/comparison_with_paper.json and ../results/comparison_with_paper.md

import json
import os
import numpy as np
import pandas as pd

R = "../results"
paper = json.load(open("../paper_reference.json"))
out = {"headline": {}, "diagnostics": {}, "yearly": [], "models": {}, "regime": {},
       "regime_conditional": {}, "optimization": {}, "transaction_costs": {},
       "paired_tests": {}, "sample": {}}


def f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


# ---------------- baseline ----------------
perf = pd.read_csv(f"{R}/baseline_results/portfolio_performance_metrics.csv").set_index("portfolio")
diag = pd.read_csv(f"{R}/baseline_results/ranking_diagnostics_summary.csv").iloc[0]
p1 = paper["table1_performance_2021_2025"]
for mine, theirs in [("RankModel_Top10", "RankModel_Top10"), ("SPY", "SPY"),
                     ("bottom20_Predicted", "Bottom20_Predicted"), ("TopMinusBottom", "TopMinusBottom")]:
    row = perf.loc[mine]
    pr = p1[theirs]
    out["headline"][theirs] = {
        "total_return": {"paper": pr.get("total_return"), "repro": f(row["total_return"])},
        "CAGR": {"paper": pr.get("CAGR"), "repro": f(row["CAGR"])},
        "annual_return": {"paper": pr.get("annual_return"), "repro": f(row["annual_return"])},
        "volatility": {"paper": pr.get("volatility"), "repro": f(row["annual_volatility"])},
        "sharpe": {"paper": pr.get("sharpe"), "repro": f(row["sharpe"])},
        "max_drawdown": {"paper": pr.get("max_drawdown"), "repro": f(row["max_drawdown"])},
        "excess_vs_spy": {"paper": pr.get("excess_vs_spy"), "repro": f(row.get("annual_excess_return_vs_spy"))},
        "beta_vs_spy": {"paper": pr.get("beta_vs_spy"), "repro": f(row.get("beta_vs_spy"))},
        "spy_alpha": {"paper": pr.get("spy_alpha"), "repro": f(row.get("alpha_vs_spy"))},
    }

p2 = paper["table2_ranking_diagnostics"]
dmap = {
    "avg_top10_monthly_return": "avg_top10_return",
    "avg_bottom20_monthly_return": "avg_bottom20_return",
    "avg_top_minus_bottom": "avg_top_minus_bottom",
    "annualized_top_minus_bottom": "annualized_top_minus_bottom",
    "top_minus_bottom_p": "top_minus_bottom_p_value",
    "annualized_top10_excess_vs_random": "annualized_top10_excess_vs_random",
    "top_minus_random_p": "top_minus_random_p_value",
    "annualized_top10_excess_vs_spy": "annualized_top10_excess_vs_spy",
    "top_minus_spy_p": "top_minus_spy_p_value",
    "avg_spearman_ic": "avg_spearman_ic",
    "spearman_ic_p": "spearman_ic_p_value",
    "pct_months_top_beats_spy": "percent_months_top_beats_spy",
    "pct_positive_ic": "percent_positive_ic",
}
for k, col in dmap.items():
    out["diagnostics"][k] = {"paper": p2[k], "repro": f(diag[col])}

out["sample"] = {
    "train_rows": {"paper": 654009, "repro": int(diag["n_train_rows"])},
    "val_rows": {"paper": 102527, "repro": int(diag["n_val_rows"])},
    "test_rows": {"paper": 253665, "repro": int(diag["n_test_rows"])},
    "features": {"paper": 68, "repro": int(diag["n_features"])},
    "stocks": {"paper": 206, "repro": int(diag["n_stocks_loaded"])},
    "test_months": {"paper": 60, "repro": int(diag["n_months"])},
}

yearly = pd.read_csv(f"{R}/baseline_results/yearly_returns.csv")
out["yearly"] = [{"year": int(r.year), "top10": f(r.rank_model_return), "spy": f(r.spy_return),
                  "bottom20": f(r.bottom20_return)} for r in yearly.itertuples()]

# ---------------- alternative models ----------------
p3 = paper["table3_model_comparison"]
out["models"]["XGBoost"] = {
    "paper": p3["XGBoost"],
    "repro": {"CAGR": f(perf.loc["RankModel_Top10", "CAGR"]), "sharpe": f(perf.loc["RankModel_Top10", "sharpe"]),
              "tb_p": f(diag["top_minus_bottom_p_value"]), "tr_p": f(diag["top_minus_random_p_value"]),
              "ts_p": f(diag["top_minus_spy_p_value"]), "ic_p": f(diag["spearman_ic_p_value"])},
}
for key, name in [("rf", "RandomForest"), ("lgbm", "LightGBM"), ("extra_trees", "ExtraTrees"),
                  ("ridge", "Ridge"), ("nn", "NeuralNetwork")]:
    d = f"{R}/model_comparison_{key}"
    if os.path.exists(f"{d}/portfolio_performance_metrics.csv") and os.path.exists(f"{d}/ranking_diagnostics_summary.csv"):
        pm = pd.read_csv(f"{d}/portfolio_performance_metrics.csv").set_index("portfolio")
        dg = pd.read_csv(f"{d}/ranking_diagnostics_summary.csv").iloc[0]
        out["models"][name] = {
            "paper": p3[name],
            "repro": {"CAGR": f(pm.loc["RankModel_Top10", "CAGR"]), "sharpe": f(pm.loc["RankModel_Top10", "sharpe"]),
                      "tb_p": f(dg["top_minus_bottom_p_value"]), "tr_p": f(dg["top_minus_random_p_value"]),
                      "ts_p": f(dg["top_minus_spy_p_value"]), "ic_p": f(dg["spearman_ic_p_value"])},
        }
    else:
        out["models"][name] = {"paper": p3[name], "repro": None}

# ---------------- regime experiments ----------------
rd = f"{R}/hmm_regime_experiments"
if os.path.exists(f"{rd}/regime_experiment_metrics.csv"):
    rm = pd.read_csv(f"{rd}/regime_experiment_metrics.csv", index_col=0)
    for mine, theirs, pkey in [("hmm_as_feature", "hmm_as_feature", "table10_hmm_as_feature"),
                               ("one_model_per_regime", "one_model_per_regime", "table11_one_model_per_regime")]:
        row = rm.loc[mine]
        pp = paper[pkey]["hmm_as_feature"] if pkey == "table10_hmm_as_feature" else paper[pkey]
        out["regime"][theirs] = {
            "paper": pp,
            "repro": {"CAGR": f(row["CAGR"]), "sharpe": f(row["sharpe"]), "tb_p": f(row["top_minus_bottom_p"]),
                      "tr_p": f(row["top_minus_random_p"]), "ts_p": f(row["top_minus_spy_p"]),
                      "ic_p": f(row["spearman_ic_p"]), "avg_ic": f(row["spearman_ic_mean"])},
        }
    rc = pd.read_csv(f"{rd}/baseline_regime_conditional.csv")
    chars = pd.read_csv("../data/hmm_4_state_regime_characteristics_expanding.csv")
    labels = dict(zip(chars["hmm_regime"], chars["economic_label"]))
    out["regime_conditional"] = {
        "paper": paper["table12_regime_conditional_baseline"],
        "repro": [{"regime": int(r.hmm_regime), "label": labels.get(int(r.hmm_regime), "?"),
                   "n_months": int(r.n_months), "CAGR": f(r.CAGR), "sharpe": f(r.sharpe), "avg_ic": f(r.avg_ic)}
                  for r in rc.itertuples()],
    }
    pt = pd.read_csv(f"{rd}/paired_tests_vs_baseline.csv")
    out["paired_tests"]["regime"] = {
        "paper": paper["table13_paired_tests_vs_baseline"],
        "repro": {alt: {sp: {"ann_diff": f(g2["annualized_diff"].iloc[0]), "hac_p": f(g2["hac_p_value"].iloc[0])}
                        for sp, g2 in g.groupby("spread")}
                  for alt, g in pt.groupby("alternative")},
    }

# ---------------- portfolio optimization ----------------
po = pd.read_csv(f"{R}/portfolio_optimization/optimized_portfolio_performance_metrics.csv").set_index("strategy")
cmp = pd.read_csv(f"{R}/portfolio_optimization/optimization_comparison_tests.csv").set_index("strategy")
p14 = paper["table14_portfolio_optimization"]
p15 = paper["table15_paired_optimization_vs_ew"]
for s in ["EqualWeight_Top10", "Static_MinVar_Top10", "Static_ScoreMV_Top10", "HMM_MinVar_Top10", "HMM_ScoreMV_Top10"]:
    short = s.replace("_Top10", "")
    out["optimization"][s] = {
        "paper": p14[s],
        "repro": {"CAGR": f(po.loc[s, "CAGR"]), "sharpe": f(po.loc[s, "sharpe"]),
                  "tb_p": f(cmp.loc[s, "top_minus_bottom_p_value"]), "ts_p": f(cmp.loc[s, "top_minus_spy_p_value"]),
                  "spy_alpha": f(po.loc[s, "alpha_vs_spy"]), "max_drawdown": f(po.loc[s, "max_drawdown"])},
        "paired_vs_ew": {
            "paper": p15.get(short),
            "repro": None if s == "EqualWeight_Top10" else {
                "ann_diff": f(cmp.loc[s, "annualized_excess_vs_equal_weight"]),
                "p": f(cmp.loc[s, "strategy_minus_equal_weight_p_value"])},
        },
    }

# ---------------- transaction costs ----------------
tc = pd.read_csv(f"{R}/transaction_costs/transaction_cost_performance_summary.csv").set_index("cost_bps")
p20 = paper["table20_transaction_costs"]
out["transaction_costs"] = {
    "avg_monthly_turnover": {"paper": p20["avg_monthly_turnover"], "repro": f(tc["avg_turnover"].iloc[0])},
    "rows": [{"bps": int(b), "paper": p20[f"bps{int(b)}"],
              "repro": {"CAGR": f(tc.loc[b, "CAGR"]), "sharpe": f(tc.loc[b, "sharpe"]),
                        "spy_alpha": f(tc.loc[b, "alpha_vs_spy"]), "ts_p": f(tc.loc[b, "top_minus_spy_p_value"]),
                        "tb_p": f(tc.loc[b, "top_minus_bottom_p_value"])}}
             for b in tc.index]
}

# ---------------- SHAP top features ----------------
shap = pd.read_csv(f"{R}/baseline_results/shap_feature_importance.csv").head(15)
out["paper_table13"] = paper["table13_paired_tests_vs_baseline"]
out["shap_top15"] = [{"feature": r.feature, "mean_abs_shap": f(r.mean_abs_shap)} for r in shap.itertuples()]

# ---------------- monthly series for charts ----------------
m = pd.read_csv(f"{R}/baseline_results/monthly_rank_model_portfolio_returns.csv", parse_dates=["date"])
out["monthly"] = [{"date": d.strftime("%Y-%m-%d"), "top10": f(a), "spy": f(b), "bottom20": f(c), "ic": f(i)}
                  for d, a, b, c, i in zip(m["date"], m["top10_return"], m["spy_return"], m["bottom20_return"], m["spearman_ic"])]
dec = pd.read_csv(f"{R}/baseline_results/decile_return_summary.csv")
out["deciles"] = [{"decile": int(r.predicted_decile), "annualized_mean_return": f(r.annualized_mean_return)} for r in dec.itertuples()]

with open(f"{R}/comparison_with_paper.json", "w") as fh:
    json.dump(out, fh, indent=1)


# ---------------- markdown summary ----------------
def pct(x):
    return "—" if x is None else f"{100 * x:.2f}%"


def num(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


lines = ["# Reproduction vs paper", "", "## Table 1 — Out-of-sample portfolio performance, 2021–2025", "",
         "| Portfolio | Metric | Paper | Reproduced |", "|---|---|---|---|"]
for port, mets in out["headline"].items():
    for k, v in mets.items():
        if v["paper"] is None:
            continue
        fmt = num if k in ("sharpe", "beta_vs_spy", "total_return") else pct
        lines.append(f"| {port} | {k} | {fmt(v['paper'])} | {fmt(v['repro'])} |")
lines += ["", "## Table 2 — Ranking diagnostics", "", "| Diagnostic | Paper | Reproduced |", "|---|---|---|"]
for k, v in out["diagnostics"].items():
    fmt = num if ("_p" in k or "ic" in k and "pct" not in k) else pct
    if k in ("avg_spearman_ic",):
        fmt = lambda x: num(x, 4)
    lines.append(f"| {k} | {fmt(v['paper'])} | {fmt(v['repro'])} |")
lines += ["", "## Table 3 — Alternative models (CAGR / Sharpe / Top-SPY p)", "", "| Model | Paper | Reproduced |", "|---|---|---|"]
for name, v in out["models"].items():
    pp = v["paper"]
    rr = v["repro"]
    lines.append(f"| {name} | {pct(pp['CAGR'])} / {num(pp['sharpe'])} / {num(pp['ts_p'], 4)} | " +
                 ("pending" if rr is None else f"{pct(rr['CAGR'])} / {num(rr['sharpe'])} / {num(rr['ts_p'], 4)}") + " |")
if out["regime"]:
    lines += ["", "## Tables 10–11 — Regime-aware forecasting (CAGR / Sharpe / IC p)", "", "| Specification | Paper | Reproduced |", "|---|---|---|"]
    for name, v in out["regime"].items():
        pp, rr = v["paper"], v["repro"]
        lines.append(f"| {name} | {pct(pp['CAGR'])} / {num(pp['sharpe'])} / {num(pp['ic_p'], 4)} | {pct(rr['CAGR'])} / {num(rr['sharpe'])} / {num(rr['ic_p'], 4)} |")
lines += ["", "## Table 14 — Portfolio optimization (CAGR / Sharpe / SPY alpha)", "", "| Strategy | Paper | Reproduced |", "|---|---|---|"]
for s, v in out["optimization"].items():
    pp, rr = v["paper"], v["repro"]
    lines.append(f"| {s} | {pct(pp['CAGR'])} / {num(pp['sharpe'])} / {pct(pp['spy_alpha'])} | {pct(rr['CAGR'])} / {num(rr['sharpe'])} / {pct(rr['spy_alpha'])} |")
lines += ["", "## Table 20 — Transaction costs (CAGR / Sharpe / Top-SPY p)", "",
          f"Average monthly turnover: paper {pct(out['transaction_costs']['avg_monthly_turnover']['paper'])}, reproduced {pct(out['transaction_costs']['avg_monthly_turnover']['repro'])}", "",
          "| Cost | Paper | Reproduced |", "|---|---|---|"]
for r in out["transaction_costs"]["rows"]:
    pp, rr = r["paper"], r["repro"]
    lines.append(f"| {r['bps']} bps | {pct(pp['CAGR'])} / {num(pp['sharpe'])} / {num(pp['ts_p'], 4)} | {pct(rr['CAGR'])} / {num(rr['sharpe'])} / {num(rr['ts_p'], 4)} |")
lines += ["", "## Yearly returns (Top10 / SPY)", "", "| Year | Top10 | SPY | Bottom20 |", "|---|---|---|---|"]
for y in out["yearly"]:
    lines.append(f"| {y['year']} | {pct(y['top10'])} | {pct(y['spy'])} | {pct(y['bottom20'])} |")

with open(f"{R}/comparison_with_paper.md", "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))
