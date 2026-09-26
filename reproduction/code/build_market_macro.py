# Build the market-level inputs:
#   ../data/merged_market_data.csv and ../data_HMM/merged_market_data.csv
#       (date, sp500_close, vix_close, us10y_yield, us2y_yield) for the HMM
#   ../data_bloomberg_macro/macro_features.csv
#       cross-asset features (final baseline) + rates features + partial
#       macro-cycle features (for the feature-ablation robustness runs)
#
# Sources (all public GitHub mirrors, verified):
#   ^GSPC close        Jackscuster/fx-data
#   VIX                datasets/finance-vix (already staged as data_bloomberg/VIX.csv)
#   Treasury yields    fujiapple852/yield (daily par yield curve, = FRED DGS*)
#   AGG adj close      giacomomaggiore/margin-investing-etfs (dividend-adjusted,
#                      so pct changes are total returns)
#   DXY                whyyouj/market-sentiment-analysis (2000-2024) +
#                      kittycapital/btc-asset-correlation (2025-2026); same
#                      Yahoo DX-Y.NYB series, identical on overlap
#   Gold               tomelam/PortfolioAnalyzer (LBMA USD daily)
#   WTI oil            datasets/oil-prices (EIA spot)
#   ^SP500TR           eklyb94-blip/asset-allocation-dashboard
#   CPI/UNRATE/FEDFUNDS  shergin/malevich FRED copies (monthly)
#
# Note: ISM PMI and consumer confidence (used only in the paper's rejected
# "+macro-cycle" ablation) have no accessible public mirror; that ablation
# runs with the remaining macro-cycle features and is flagged in the report.

import os
import numpy as np
import pandas as pd

RAW = "../raw_sources"
START, END = "2003-01-01", "2026-03-31"

os.makedirs("../data", exist_ok=True)
os.makedirs("../data_HMM", exist_ok=True)
os.makedirs("../data_bloomberg_macro", exist_ok=True)


def clip(df):
    return df[(df["date"] >= START) & (df["date"] <= END)].reset_index(drop=True)


# ---------------- HMM market data ----------------

gspc = pd.read_csv(f"{RAW}/misc/gspc.csv")
gspc.columns = ["date", "sp500_close"]
gspc["date"] = pd.to_datetime(gspc["date"])
gspc = clip(gspc.sort_values("date").dropna())

vix = pd.read_csv("../data_bloomberg/VIX.csv")
vix.columns = [c.lower() for c in vix.columns]
vix = vix.rename(columns={"close": "vix_close"})[["date", "vix_close"]]
vix["date"] = pd.to_datetime(vix["date"])

yields = pd.read_csv(f"{RAW}/misc/treasury_yields.csv")
yields.columns = [c.strip() for c in yields.columns]
yields["date"] = pd.to_datetime(yields["date"])
yields = yields.sort_values("date")
for c in ["3M", "2Y", "10Y"]:
    yields[c] = pd.to_numeric(yields[c], errors="coerce")
yields[["3M", "2Y", "10Y"]] = yields[["3M", "2Y", "10Y"]].ffill()
yc = yields.rename(columns={"10Y": "us10y_yield", "2Y": "us2y_yield", "3M": "us3m_yield"})
yc = yc[["date", "us10y_yield", "us2y_yield", "us3m_yield"]]

merged = gspc.merge(vix, on="date", how="left")
merged = pd.merge_asof(merged.sort_values("date"), yc.sort_values("date"),
                       on="date", direction="backward")
merged = merged.dropna(subset=["sp500_close", "vix_close", "us10y_yield", "us2y_yield"])
merged[["date", "sp500_close", "vix_close", "us10y_yield", "us2y_yield"]].to_csv(
    "../data/merged_market_data.csv", index=False)
merged[["date", "sp500_close", "vix_close", "us10y_yield", "us2y_yield"]].to_csv(
    "../data_HMM/merged_market_data.csv", index=False)
print("merged_market_data:", len(merged), merged["date"].min().date(), "->", merged["date"].max().date())


# ---------------- macro / cross-asset features ----------------

def series_from(path, date_col, val_col, name, sep=",", rename_cols=None):
    df = pd.read_csv(path, sep=sep)
    if rename_cols:
        df.columns = rename_cols
    df[date_col] = pd.to_datetime(df[date_col].astype(str).str.slice(0, 10))
    df = df[[date_col, val_col]].rename(columns={date_col: "date", val_col: name})
    df[name] = pd.to_numeric(df[name], errors="coerce")
    return df.dropna().sort_values("date").reset_index(drop=True)


agg = series_from(f"{RAW}/misc/agg.csv", "date", "adj close", "agg")
gold = series_from(f"{RAW}/misc/gold_lbma.csv", "Date", "Close", "gold")
oil = series_from(f"{RAW}/misc/wti_daily.csv", "Date", "Price", "oil")

dxy_a = series_from(f"{RAW}/misc/dxy_2000_2024.csv", "Date", "DXY", "dxy")
dxy_b = series_from(f"{RAW}/misc/dxy_2014_2026.csv", "Date", "Close", "dxy")
cut = dxy_a["date"].max()
overlap = dxy_a.merge(dxy_b, on="date", suffixes=("_a", "_b"))
dev = (overlap["dxy_a"] / overlap["dxy_b"] - 1).abs().max()
print(f"DXY splice: {len(overlap)} overlap days, max deviation {dev:.6f}")
dxy = pd.concat([dxy_a, dxy_b[dxy_b["date"] > cut]], ignore_index=True).sort_values("date")

sptr = pd.read_csv(f"{RAW}/misc/sp500tr.csv", encoding="utf-8-sig")
sptr.columns = ["date", "sp500tr"]
sptr["date"] = pd.to_datetime(sptr["date"])
sptr = sptr.dropna().sort_values("date").reset_index(drop=True)

frames = []
for df, name in [(agg, "us_agg_bond"), (dxy, "dxy"), (gold, "gold"), (oil, "oil")]:
    df = clip(df)
    val = df.columns[1]
    for w in [20, 60]:
        df[f"{name}_ret_{w}d"] = df[val].pct_change(w)
    frames.append(df.drop(columns=[val]).set_index("date"))

# rates features (paper's "+rates" ablation)
yr = clip(yc.copy()).set_index("date")
yr = yr.rename(columns={"us10y_yield": "us10y", "us2y_yield": "us2y", "us3m_yield": "us3m"})
yr["yield_spread_10y_2y"] = yr["us10y"] - yr["us2y"]
yr["yield_spread_10y_3m"] = yr["us10y"] - yr["us3m"]
yr["us10y_change_20d"] = yr["us10y"].diff(20)
yr["us10y_change_60d"] = yr["us10y"].diff(60)
yr["yield_spread_10y_2y_change_20d"] = yr["yield_spread_10y_2y"].diff(20)
yr["yield_spread_10y_2y_change_60d"] = yr["yield_spread_10y_2y"].diff(60)
frames.append(yr)

# risk features (spy_tr_* from the total-return index)
st = clip(sptr).set_index("date")
st_ret = st["sp500tr"].pct_change()
risk = pd.DataFrame(index=st.index)
risk["spy_tr_drawdown_252d"] = st["sp500tr"] / st["sp500tr"].rolling(252).max() - 1
risk["spy_tr_vol_20d"] = st_ret.rolling(20).std() * np.sqrt(252)
risk["spy_tr_vol_60d"] = st_ret.rolling(60).std() * np.sqrt(252)
frames.append(risk)

vix2 = vix.set_index("date")
frames.append(pd.DataFrame({"vix_change_60d": vix2["vix_close"].pct_change(60)}))

# macro-cycle (monthly FRED, daily-ffilled; partial: no ISM PMI / consumer conf.)
def fred_monthly(fname, col, name):
    df = pd.read_csv(f"{RAW}/misc/{fname}")
    df.columns = ["date", name]
    df["date"] = pd.to_datetime(df["date"])
    df[name] = pd.to_numeric(df[name], errors="coerce")
    return df.dropna().sort_values("date").set_index("date")

cpi = fred_monthly("cpi.csv", "CPIAUCSL", "cpi_level")
cpi["cpi_yoy"] = cpi["cpi_level"].pct_change(12)
unrate = fred_monthly("unrate.csv", "UNRATE", "unemployment")
ff = fred_monthly("fedfunds.csv", "FEDFUNDS", "fed_funds")

macro_m = cpi[["cpi_yoy"]].join(unrate, how="outer").join(ff, how="outer")

# assemble on the union of dates, ffill, then keep NYSE (SPY) dates
spy_dates = pd.read_csv("../data_bloomberg/SPY.csv", usecols=["date"], parse_dates=["date"])["date"]
idx = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames]) | set(spy_dates)))
out = pd.DataFrame(index=idx)
for f in frames:
    out = out.join(f, how="left")
out = out.join(macro_m, how="left")
out = out.sort_index().ffill()

out["fed_funds_change_20d"] = out["fed_funds"].diff(20)
out["fed_funds_change_60d"] = out["fed_funds"].diff(60)
out["cpi_yoy_change_60d"] = out["cpi_yoy"].diff(60)
out["unemployment_change_60d"] = out["unemployment"].diff(60)

out = out.loc[out.index.isin(set(spy_dates))]
out.index.name = "date"
out = out.reset_index()
out = clip(out)
out.to_csv("../data_bloomberg_macro/macro_features.csv", index=False)

print("macro_features:", len(out), out["date"].min().date(), "->", out["date"].max().date())
print("columns:", out.columns.tolist())
print("null counts (2004+):")
recent = out[out["date"] >= "2004-06-01"]
print(recent.isna().sum()[recent.isna().sum() > 0])
