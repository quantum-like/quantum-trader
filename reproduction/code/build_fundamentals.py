# Build daily point-in-time fundamentals per ticker:
#   ../data_bloomberg_fundamentals/{TICKER}_fundamentals.csv
#   columns: date, market_cap, pe_ratio, pb_ratio, roe, debt_to_equity
#
# Three stitched point-in-time layers (SEC-derived, no Bloomberg access):
#   1. stockpup quarterly (1993-2019): backfill before XBRL. as-of date =
#      quarter end + 45 calendar days (typical 10-Q lag; the paper's Table 18
#      shows a 90-day lag changes little).
#   2. GTSF sec_parser XBRL per filing (2009 - Feb 2024): filing_date is the
#      as-of date (true point-in-time).
#   3. SEC Financial Statement Data Sets 2024q1-2025q2 (via LFS mirror):
#      `filed` is the as-of date; un-dimensioned facts only.
#   After mid-2025 the last observation is forward-filled (as baseline.py
#   does anyway); the Jan-2026 tail only needs prices.
#
# Market cap methodology ("anchor and scale", split-safe):
#   Every source reports as-filed shares outstanding, so consecutive filings
#   let us detect split factors as jumps snapped to standard ratios. All
#   shares are converted to the 2026 basis (the basis of the brownbear price
#   snapshot). Daily market cap = shares_2026(latest filing <= t) x
#   split-adjusted close in 2026 basis:
#     - t >= 2019-01-02: brownbear 'Close' (split-adjusted, not dividend-adj)
#     - t <  2019-01-02: Deamoner raw close / (product of split factors dated
#       after t), with exact pre-2019 split dates recovered from one-day raw
#       price drops matching the detected factor inside the filing interval.
#   P/E = mcap / trailing-4-quarter net income (NaN if ttm NI <= 0)
#   P/B = mcap / stockholders equity (NaN if equity <= 0)
#   ROE = ttm NI / equity;  D/E = long-term debt / equity
#
# Validation: ROE and D/E (split-invariant) are compared against stockpup's
# own published ratio columns; market caps are spot-checked against known
# values; per-ticker logs land in ../data_support/fundamentals_log.json.

import json
import os
import re
import numpy as np
import pandas as pd

RAW = "../raw_sources"
OUT = "../data_bloomberg_fundamentals"
SUPPORT = "../data_support"
SEAM = pd.Timestamp("2019-01-02")
os.makedirs(OUT, exist_ok=True)
os.makedirs(SUPPORT, exist_ok=True)

code = open("baseline.py").read()
TICKERS = re.findall(r'"([A-Z.\-]+)"', code.split("TICKERS = [")[1].split("]")[0])

SPLIT_CANDIDATES = [1.5, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 28, 50]
SPLIT_CANDIDATES = SPLIT_CANDIDATES + [1 / x for x in SPLIT_CANDIDATES]

# Companies sharing a CIK / share-class twins: reuse the sibling's file.
FUND_ALIAS = {"GOOGL": "GOOG"}

# ------------------------------------------------------------------
# Load FSDS (2024q1-2025q2) once, keyed by cik
# ------------------------------------------------------------------

TAGS_INSTANT = ["StockholdersEquity", "Liabilities", "LongTermDebtNoncurrent",
                "LongTermDebt", "LongTermDebtCurrent",
                "EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"]
TAGS_DUR = ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss",
            "WeightedAverageNumberOfDilutedSharesOutstanding"]

def load_fsds():
    frames = []
    for q in ["2024q1", "2024q2", "2024q3", "2024q4", "2025q1", "2025q2"]:
        sub = pd.read_parquet(f"{RAW}/fsds/{q}_sub.parquet",
                              columns=["adsh", "cik", "form", "period", "filed"])
        sub = sub[sub["form"].isin(["10-K", "10-Q"])]
        num = pd.read_parquet(f"{RAW}/fsds/{q}_num.parquet",
                              columns=["adsh", "tag", "ddate", "qtrs", "segments", "coreg", "value"])
        num = num[num["tag"].isin(TAGS_INSTANT + TAGS_DUR)]
        num = num[num["segments"].isna() & num["coreg"].isna()]
        m = num.merge(sub, on="adsh", how="inner")
        frames.append(m)
    fsds = pd.concat(frames, ignore_index=True)
    fsds["ddate"] = pd.to_datetime(fsds["ddate"], format="%Y%m%d")
    fsds["filed"] = pd.to_datetime(fsds["filed"], format="%Y%m%d")
    fsds["period"] = pd.to_datetime(fsds["period"], format="%Y%m%d")
    return fsds


def load_cik_map():
    with open(f"{RAW}/misc/company_tickers.json") as f:
        m = json.load(f)
    # file format: {"AAPL": 320193, ...} or {"0": {"cik_str":..,"ticker":..}}
    first = next(iter(m.values()))
    if isinstance(first, dict):
        return {v["ticker"].upper().replace("-", "."): int(v["cik_str"]) for v in m.values()}
    return {k.upper().replace("-", "."): int(v) for k, v in m.items()}


def fsds_filings(fsds, cik):
    """Per-filing rows (as_of, shares, equity, ni_q, ltd) from FSDS for one cik."""
    g = fsds[fsds["cik"] == cik]
    if g.empty:
        return pd.DataFrame()
    rows = []
    for adsh, f in g.groupby("adsh"):
        filed = f["filed"].iloc[0]
        period = f["period"].iloc[0]

        def instant(tags):
            c = f[(f["tag"].isin(tags)) & (f["qtrs"] == 0)]
            if c.empty:
                return np.nan
            c = c.sort_values("ddate")
            return c["value"].iloc[-1]

        # quarterly NI: duration fact ending at period, qtrs==1 (tag fallbacks)
        ni_q = np.nan
        ni_a = np.nan
        for tag in ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"]:
            c = f[(f["tag"] == tag) & (f["qtrs"] == 1) & (f["ddate"] == period)]
            if len(c):
                ni_q = c["value"].iloc[-1]
                break
        for tag in ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"]:
            c = f[(f["tag"] == tag) & (f["qtrs"] == 4) & (f["ddate"] == period)]
            if len(c):
                ni_a = c["value"].iloc[-1]
                break
        if f["form"].iloc[0] != "10-K":
            ni_a = np.nan

        shares = instant(["EntityCommonStockSharesOutstanding"])
        if not np.isfinite(shares):
            shares = instant(["CommonStockSharesOutstanding"])
        if not np.isfinite(shares):
            wa = f[(f["tag"] == "WeightedAverageNumberOfDilutedSharesOutstanding") & (f["qtrs"] == 1)]
            shares = wa["value"].iloc[-1] if len(wa) else np.nan
        equity = instant(["StockholdersEquity"])
        ltd_nc = instant(["LongTermDebtNoncurrent"])
        ltd_c = instant(["LongTermDebtCurrent"])
        ltd_plain = instant(["LongTermDebt"])
        if np.isfinite(ltd_nc):
            ltd = ltd_nc + (ltd_c if np.isfinite(ltd_c) else 0.0)
        else:
            ltd = ltd_plain
        rows.append({"as_of": filed, "period_end": period, "shares": shares,
                     "equity": equity, "ni_q": ni_q, "ni_a": ni_a, "ltd": ltd, "src": "fsds"})
    return pd.DataFrame(rows).sort_values("as_of")


# ------------------------------------------------------------------
# GTSF per-filing extraction
# ------------------------------------------------------------------

def gtsf_filings(t):
    path = f"{RAW}/fundamentals/gtsf/{t}.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    g = pd.read_csv(path)
    g["filing_date"] = pd.to_datetime(g["filing_date"])
    g = g.sort_values("filing_date")

    def col(name):
        return pd.to_numeric(g[name], errors="coerce") if name in g.columns else pd.Series(np.nan, index=g.index)

    # choose the NI tag with best coverage for this company, fill gaps from others
    ni_candidates = ["NetIncomeLoss_USD", "NetIncomeLossAvailableToCommonStockholdersBasic_USD", "ProfitLoss_USD"]
    ni_cov = {c: col(c).notna().sum() for c in ni_candidates}
    ni_order = sorted(ni_candidates, key=lambda c: -ni_cov[c])
    ni_series = col(ni_order[0])
    for c in ni_order[1:]:
        ni_series = ni_series.fillna(col(c))

    shares = col("EntityCommonStockSharesOutstanding_shares")
    shares = shares.fillna(col("CommonStockSharesOutstanding_shares"))
    shares = shares.fillna(col("WeightedAverageNumberOfDilutedSharesOutstanding_shares"))
    shares = shares.fillna(col("WeightedAverageNumberOfSharesOutstandingBasic_shares"))
    equity = col("StockholdersEquity_USD")
    ni = ni_series
    ltd_nc = col("LongTermDebtNoncurrent_USD")
    ltd_c = col("LongTermDebtCurrent_USD")
    ltd_plain = col("LongTermDebt_USD")
    ltd = ltd_nc + ltd_c.fillna(0.0)
    ltd = ltd.fillna(ltd_plain)

    out = pd.DataFrame({
        "as_of": g["filing_date"], "annual": g["annual"].astype(bool),
        "shares": shares, "equity": equity, "ni_raw": ni, "ltd": ltd,
    })
    out["ni_a"] = out["ni_raw"].where(out["annual"])

    # 10-Q rows carry quarterly NI; 10-K rows carry fiscal-year totals.
    # Convert annual rows to Q4 = annual - previous three quarterly values.
    ni_q = []
    recent_q = []
    for _, r in out.iterrows():
        if not r["annual"]:
            ni_q.append(r["ni_raw"])
            recent_q.append(r["ni_raw"])
        else:
            prev3 = [x for x in recent_q[-3:] if np.isfinite(x)]
            q4 = r["ni_raw"] - sum(prev3) if np.isfinite(r["ni_raw"]) and len(prev3) == 3 else np.nan
            ni_q.append(q4)
            recent_q = []
    out["ni_q"] = ni_q
    out["src"] = "gtsf"
    return out[["as_of", "shares", "equity", "ni_q", "ni_a", "ltd", "src"]]


# ------------------------------------------------------------------
# stockpup extraction (pre-XBRL backfill)
# ------------------------------------------------------------------

def stockpup_filings(t):
    path = f"{RAW}/fundamentals/stockpup/{t}.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    s = pd.read_csv(path)
    s["Quarter end"] = pd.to_datetime(s["Quarter end"])
    s = s.sort_values("Quarter end")
    num = lambda c: pd.to_numeric(s[c], errors="coerce")
    out = pd.DataFrame({
        "as_of": s["Quarter end"] + pd.Timedelta(days=45),
        "period_end": s["Quarter end"],
        "shares": num("Shares"),
        "equity": num("Shareholders equity"),
        "ni_q": num("Earnings"),
        "ni_a": np.nan,
        "ltd": num("Long-term debt"),
        "roe_ref": num("ROE"),
        "dte_ref": num("Long-term debt to equity ratio"),
        "src": "stockpup",
    })
    return out


# ------------------------------------------------------------------
# split detection + basis conversion
# ------------------------------------------------------------------

# Known splits with effective dates on/after 2019-01-01 (through the brownbear
# snapshot of Jul 2026). Filing-based detection cannot see events after the
# last filing (~mid-2025) and cannot distinguish splits from M&A share
# issuance without raw prices, which do not exist post-2019 in our sources.
POST2019_SPLITS = {
    "AAPL": [("2020-08-31", 4)],
    "TSLA": [("2020-08-31", 5), ("2022-08-25", 3)],
    "NVDA": [("2021-07-20", 4), ("2024-06-10", 10)],
    "AMZN": [("2022-06-06", 20)],
    "GOOGL": [("2022-07-18", 20)],
    "GOOG": [("2022-07-18", 20)],
    "SHW": [("2021-04-01", 3)],
    "ISRG": [("2021-10-05", 3)],
    "CMG": [("2024-06-26", 50)],
    "WMT": [("2024-02-26", 3)],
    "AVGO": [("2024-07-15", 10)],
    "LRCX": [("2024-10-03", 10)],
    "CTAS": [("2024-09-11", 4)],
    "APH": [("2021-03-19", 2)],
    "CSX": [("2021-06-28", 3)],
    "NEE": [("2020-10-26", 4)],
    "GE": [("2021-08-02", 0.125)],
    "PANW": [("2022-09-14", 3), ("2024-12-16", 2)],
    "MNST": [("2023-03-28", 2)],
    "FAST": [("2019-05-23", 2)],
    "ORLY": [("2025-06-10", 15)],
    "ODFL": [("2022-03-25", 2)],
}


def detect_splits(filings):
    """Return list of (index_after, factor) where shares jump by a split ratio."""
    s = filings["shares"].values
    events = []
    for i in range(1, len(s)):
        a, b = s[i - 1], s[i]
        if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
            continue
        r = b / a
        if 0.7 < r < 1.4:
            continue
        cand = min(SPLIT_CANDIDATES, key=lambda c: abs(np.log(r / c)))
        if abs(np.log(r / cand)) < 0.06:
            events.append((i, cand))
    return events


def find_split_date(raw_prices, lo, hi, factor):
    """Exact split date from a one-day raw-price drop of ~1/factor in (lo, hi]."""
    seg = raw_prices[(raw_prices["date"] > lo) & (raw_prices["date"] <= hi)].copy()
    if len(seg) < 2:
        return None
    r = seg["raw_close"].values[1:] / seg["raw_close"].values[:-1]
    target = 1.0 / factor
    best = np.argmin(np.abs(np.log(r / target)))
    if abs(np.log(r[best] / target)) < 0.12:
        return seg["date"].values[1 + best]
    return None


# ------------------------------------------------------------------
# main per-ticker build
# ------------------------------------------------------------------

def build_ticker(t, fsds, cik_map, prices, support):
    tf = FUND_ALIAS.get(t, t)
    layers = []
    sp = stockpup_filings(tf)
    gt = gtsf_filings(tf)
    cik = cik_map.get(tf.replace("-", "."))
    fs = fsds_filings(fsds, cik) if cik else pd.DataFrame()

    gt_start = gt["as_of"].min() if len(gt) else pd.Timestamp("2100-01-01")
    fs_start = fs["as_of"].min() if len(fs) else pd.Timestamp("2100-01-01")
    if len(sp):
        layers.append(sp[sp["as_of"] < gt_start][["as_of", "shares", "equity", "ni_q", "ni_a", "ltd", "src"]])
    if len(gt):
        layers.append(gt[gt["as_of"] < fs_start])
    if len(fs):
        layers.append(fs[["as_of", "shares", "equity", "ni_q", "ni_a", "ltd", "src"]])
    if not layers:
        return None, {"status": "no_fundamentals"}

    fil = pd.concat(layers, ignore_index=True).sort_values("as_of").reset_index(drop=True)
    fil = fil.dropna(subset=["shares", "equity"], how="all")
    if len(fil) < 4:
        return None, {"status": "too_few_filings", "n": len(fil)}

    fil["shares"] = fil["shares"].ffill()
    fil["equity"] = fil["equity"].ffill()
    fil["ltd"] = fil["ltd"].ffill()
    # For FSDS annual rows with a FY total but no Q4 fact, derive Q4 from the
    # previous three quarterly values (GTSF annual rows are handled upstream).
    ni_q = fil["ni_q"].tolist()
    ni_a = fil["ni_a"].tolist()
    for i in range(len(fil)):
        if np.isfinite(ni_a[i]) and not np.isfinite(ni_q[i]) and i >= 3:
            prev3 = [x for x in ni_q[i - 3:i] if np.isfinite(x)]
            if len(prev3) == 3:
                ni_q[i] = ni_a[i] - sum(prev3)
    # trailing-4-quarter NI: anchored at each annual total, rolled forward
    # quarter by quarter, with a plain 4-quarter sum as fallback
    ttm = [np.nan] * len(fil)
    for i in range(len(fil)):
        if np.isfinite(ni_a[i]):
            ttm[i] = ni_a[i]
        elif i >= 1 and np.isfinite(ttm[i - 1]) and i >= 4 and np.isfinite(ni_q[i]) and np.isfinite(ni_q[i - 4]):
            ttm[i] = ttm[i - 1] + ni_q[i] - ni_q[i - 4]
        elif i >= 3 and all(np.isfinite(ni_q[j]) for j in range(i - 3, i + 1)):
            ttm[i] = sum(ni_q[j] for j in range(i - 3, i + 1))
    fil["ni_ttm"] = ttm
    fil = fil.dropna(subset=["shares"])

    fil = fil.reset_index(drop=True)
    px = prices[prices["ticker"] == t].sort_values("date")
    if px.empty:
        return None, {"status": "no_prices"}
    pre = px[px["date"] < SEAM].copy()
    post = px[px["date"] >= SEAM].copy()

    # --- collect split events as (date, factor) ---
    # Pre-2019: shares-jump candidates verified by a one-day raw-price drop of
    # the same factor (a jump without a matching price drop is M&A share
    # issuance, i.e. a real share-count change, and is kept in `shares` as-is).
    # Post-2019: whitelist of known splits (raw prices no longer exist to
    # verify against, and events after the last filing are invisible anyway).
    split_events = []
    detected_log, rejected_log = [], []
    for idx, f in detect_splits(fil):
        lo = fil["as_of"].iloc[idx - 1]
        hi = fil["as_of"].iloc[idx]
        if hi < SEAM and len(pre):
            d = find_split_date(pre, lo - pd.Timedelta(days=120), hi, f)
            if d is not None:
                split_events.append((pd.Timestamp(d), f))
                detected_log.append((str(pd.Timestamp(d).date()), f, "price-verified"))
            else:
                rejected_log.append((str(lo.date()), str(hi.date()), f, "no price drop -> issuance"))
        # events straddling/after the seam are handled by whitelist + residual
    for d, f in POST2019_SPLITS.get(t, []):
        split_events.append((pd.Timestamp(d), f))
        detected_log.append((d, f, "whitelist"))

    # --- price basis in 2026 (brownbear-Close) terms ---
    pre_factor = np.ones(len(pre))
    for d, f in split_events:
        pre_factor[pre["date"].values < np.datetime64(d)] /= f
    pre["p2026"] = pre["raw_close"] * pre_factor
    post["p2026"] = post["split_adj_close"]

    # --- seam residual: catches splits after the last filing (e.g. 2025-2026
    # events absent from every filing source) and Yahoo spinoff price
    # adjustments. A residual that snaps to a standard ratio is a missed
    # split (applies to shares too); otherwise it is a price-basis-only
    # correction (spinoff), leaving share counts untouched. ---
    seam_resid = np.nan
    resid_kind = "none"
    a = pre.dropna(subset=["p2026"])
    b = post.dropna(subset=["p2026"])
    if len(a) >= 5 and len(b) >= 5:
        # median over a short window to average out daily returns
        ratio = b["p2026"].head(5).median() / a["p2026"].tail(5).median()
        seam_resid = float(ratio)
        if abs(np.log(ratio)) > np.log(1.10):
            inv = 1.0 / ratio
            cand = min(SPLIT_CANDIDATES, key=lambda c: abs(np.log(inv / c)))
            if abs(np.log(inv / cand)) < 0.05:
                # missed split after the last filing: adjust prices and shares
                split_events.append((pd.Timestamp("2026-07-31"), cand))
                detected_log.append(("~post-filing", cand, f"seam-residual snap ({ratio:.4f})"))
                pre["p2026"] = pre["p2026"] / cand
            else:
                # spinoff/price-only adjustment: enforce continuity
                pre["p2026"] = pre["p2026"] * ratio
                resid_kind = f"price-continuity x{ratio:.4f}"
        else:
            resid_kind = "ok"

    # --- shares in 2026 basis: multiply by factors of split events dated
    # after each filing ---
    ev_dates = np.array([np.datetime64(d) for d, _ in split_events])
    ev_factors = np.array([f for _, f in split_events])
    shares_2026 = fil["shares"].values.astype(float).copy()
    for i, asof in enumerate(fil["as_of"].values):
        if len(ev_dates):
            mask = ev_dates > asof
            shares_2026[i] *= ev_factors[mask].prod()
    fil["shares_2026"] = shares_2026

    pxx = pd.concat([pre[["date", "p2026"]], post[["date", "p2026"]]], ignore_index=True)
    pxx = pxx.dropna().sort_values("date")

    # final continuity check
    seam_check = np.nan
    a = pre.dropna(subset=["p2026"])
    b = post.dropna(subset=["p2026"])
    if len(a) and len(b):
        seam_check = float(b["p2026"].iloc[0] / a["p2026"].iloc[-1] - 1)

    daily = pd.merge_asof(pxx, fil[["as_of", "shares_2026", "equity", "ltd", "ni_ttm"]]
                          .rename(columns={"as_of": "date"}).sort_values("date"),
                          on="date", direction="backward")
    daily = daily.dropna(subset=["shares_2026"])
    daily["market_cap"] = daily["p2026"] * daily["shares_2026"]
    eq = daily["equity"].where(daily["equity"] > 0)
    ni = daily["ni_ttm"]
    daily["pe_ratio"] = (daily["market_cap"] / ni.where(ni > 0))
    daily["pb_ratio"] = daily["market_cap"] / eq
    daily["roe"] = ni / eq
    daily["debt_to_equity"] = daily["ltd"] / eq

    out = daily[["date", "market_cap", "pe_ratio", "pb_ratio", "roe", "debt_to_equity"]]

    # validation vs stockpup reference ratios (split-invariant)
    val = {}
    if len(sp) and "roe_ref" in sp.columns:
        ref = sp.dropna(subset=["roe_ref"])
        if len(ref) > 8:
            m = pd.merge_asof(ref[["as_of", "roe_ref", "dte_ref"]].sort_values("as_of"),
                              fil[["as_of", "ni_ttm", "equity", "ltd"]].sort_values("as_of"),
                              on="as_of", direction="backward")
            m = m.dropna(subset=["ni_ttm", "equity"])
            m = m[m["equity"] > 0]
            if len(m) > 8:
                my_roe = m["ni_ttm"] / m["equity"]
                val["roe_corr_vs_stockpup"] = float(np.corrcoef(my_roe, m["roe_ref"])[0, 1])

    log = {
        "status": "ok",
        "n_filings": int(len(fil)),
        "first_asof": str(fil["as_of"].min().date()),
        "last_asof": str(fil["as_of"].max().date()),
        "sources": fil["src"].value_counts().to_dict(),
        "splits_applied": detected_log,
        "jumps_rejected_as_issuance": rejected_log,
        "seam_residual": seam_resid,
        "residual_treatment": resid_kind,
        "seam_price_gap": seam_check,
        "mcap_last": float(daily["market_cap"].iloc[-1]),
        "mcap_last_date": str(daily["date"].iloc[-1].date()),
        **val,
    }
    return out, log


if __name__ == "__main__":
    fsds = load_fsds()
    cik_map = load_cik_map()
    # manual CIK fixes for renames the (dated) mapping file may miss
    cik_map.setdefault("COR", 1140859)   # Cencora (AmerisourceBergen)
    cik_map.setdefault("ELV", 1156039)   # Elevance (Anthem)
    cik_map.setdefault("TT", 1466258)    # Trane Technologies (Ingersoln-Rand plc)
    cik_map.setdefault("IR", 1699150)    # Ingersoll Rand Inc (Gardner Denver)
    cik_map.setdefault("META", 1326801)
    cik_map.setdefault("GEHC", 1932393)

    prices = pd.read_csv(f"{SUPPORT}/raw_close_store.csv", parse_dates=["date"])

    logs = {}
    for t in TICKERS:
        try:
            out, log = build_ticker(t, fsds, cik_map, prices, SUPPORT)
        except Exception as e:
            out, log = None, {"status": f"ERROR: {e}"}
        logs[t] = log
        if out is not None and len(out) > 50:
            out.to_csv(f"{OUT}/{t}_fundamentals.csv", index=False)

    with open(f"{SUPPORT}/fundamentals_log.json", "w") as f:
        json.dump(logs, f, indent=1, default=str)

    ok = [t for t, l in logs.items() if l.get("status") == "ok"]
    bad = {t: l["status"] for t, l in logs.items() if l.get("status") != "ok"}
    print(f"Fundamentals built for {len(ok)}/{len(TICKERS)} tickers")
    print("Problems:", bad)
    corr = [l.get("roe_corr_vs_stockpup") for l in logs.values() if l.get("roe_corr_vs_stockpup") is not None]
    print(f"ROE corr vs stockpup reference: median {np.median(corr):.3f}, min {min(corr):.3f} (n={len(corr)})")
    seams = [abs(l["seam_price_gap"]) for l in logs.values() if isinstance(l.get("seam_price_gap"), float) and np.isfinite(l.get("seam_price_gap"))]
    print(f"Seam price-basis gap: median {np.median(seams):.4f}, max {max(seams):.4f}")
    big_seam = {t: l["seam_price_gap"] for t, l in logs.items()
                if isinstance(l.get("seam_price_gap"), float) and np.isfinite(l.get("seam_price_gap")) and abs(l["seam_price_gap"]) > 0.06}
    print("Large seam gaps (>6%):", big_seam)
