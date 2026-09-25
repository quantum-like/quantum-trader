# Build data_bloomberg/{TICKER}.csv (+ SPY.csv) from downloaded GitHub mirrors.
#
# Splice logic per ticker:
#   - Pre-2019 segment: Deamoner snapshot (raw OHLC + dividend/split adjusted
#     'adjclose' as of 2019-04-18). Adjusted OHLC = raw OHLC * (adjclose/close).
#   - 2019+ segment: brownbear symbol-cache (Yahoo download of ~Jul 2026:
#     'Close' split-adjusted, 'Adj Close' split+dividend adjusted).
#     Adjusted OHLC = OHLC * (AdjClose/Close).
#   - The two adjusted series differ by a constant (dividend-adjustment
#     horizon), estimated as the median ratio over the 2019-01..2019-04
#     overlap and applied to the pre-2019 segment. Ratio dispersion over the
#     overlap is a per-ticker sanity check that both files describe the same
#     security.
#   - Volume is rescaled across the seam the same way (share-basis of the
#     frozen snapshot differs by post-2019 split factors).
#   - Fallback for tickers missing from brownbear: ArjunDivecha/Kronos
#     Universe500 (fully adjusted close, 2015 -> 2026-04).
#
# SPY: reconstructed from raw OHLCV (ctj01/regime-discovery-cpp) plus the SPY
# dividend history, using Yahoo's backward adjustment-factor convention; this
# is the paper's "SPY adjusted using total-return information". Cross-checked
# against brownbear's SPY Adj Close.
#
# Also writes ../data_support/raw_close_store.csv with per-ticker raw
# (as-traded) closes pre-2019 and split-adjusted closes 2019+ for the
# fundamentals builder.

import json
import os
import numpy as np
import pandas as pd

RAW = "../raw_sources"
OUT = "../data_bloomberg"
SUPPORT = "../data_support"
START = "2003-06-01"   # 252-day lookback buffer before the 2005 train start
END = "2026-02-28"     # 20-trading-day forward window beyond 2025-12-31

os.makedirs(OUT, exist_ok=True)
os.makedirs(SUPPORT, exist_ok=True)

TICKERS = sorted(os.path.splitext(f)[0] for f in os.listdir(f"{RAW}/brownbear")) if os.path.isdir(f"{RAW}/brownbear") else []
ALL_TICKERS = sorted(set(
    [os.path.splitext(f)[0] for f in os.listdir(f"{RAW}/deamoner")] +
    [os.path.splitext(f)[0] for f in os.listdir(f"{RAW}/brownbear")] +
    [os.path.splitext(f)[0] for f in os.listdir(f"{RAW}/kronos")]
))
ALL_TICKERS = [t for t in ALL_TICKERS if t != "SPY"]

SEAM = pd.Timestamp("2019-01-02")


def load_deamoner(t):
    path = f"{RAW}/deamoner/{t}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["close", "adjclose"])
    df = df[df["close"] > 0]
    f = df["adjclose"] / df["close"]
    out = pd.DataFrame({
        "date": df["date"],
        "open": df["open"] * f,
        "high": df["high"] * f,
        "low": df["low"] * f,
        "close": df["adjclose"],
        "volume": df["volume"],
        "raw_close": df["close"],
    })
    return out


def load_brownbear(t):
    path = f"{RAW}/brownbear/{t}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns={"adj close": "adjclose"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["close", "adjclose"])
    df = df[df["close"] > 0]
    f = df["adjclose"] / df["close"]
    out = pd.DataFrame({
        "date": df["date"],
        "open": df["open"] * f,
        "high": df["high"] * f,
        "low": df["low"] * f,
        "close": df["adjclose"],
        "volume": df["volume"],
        "split_adj_close": df["close"],   # split-adjusted (not div-adjusted), 2026 basis
    })
    return out


def load_kronos(t):
    path = f"{RAW}/kronos/{t}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["close"])
    out = df[["date", "open", "high", "low", "close", "volume"]].copy()
    out["split_adj_close"] = np.nan
    return out


def splice(old, new, ticker, log):
    """Scale `old` (pre-seam) onto `new`'s adjustment basis and concatenate."""
    if new is None and old is None:
        return None
    if new is None:
        log["segments"] = "deamoner-only"
        return old.drop(columns=["raw_close"], errors="ignore"), old
    if old is None:
        log["segments"] = "recent-only"
        return new.drop(columns=["split_adj_close"], errors="ignore"), new

    ov = old.merge(new, on="date", suffixes=("_o", "_n"))
    ov = ov[(ov["close_o"] > 0) & (ov["close_n"] > 0)]
    if len(ov) < 10:
        log["segments"] = "recent-only(no-overlap)"
        log["overlap_days"] = len(ov)
        return new.drop(columns=["split_adj_close"], errors="ignore"), new

    ratio = ov["close_n"] / ov["close_o"]
    scale = ratio.median()
    disp = float((ratio / scale - 1).abs().max())
    vol_ratio = (ov["volume_n"] / ov["volume_o"].replace(0, np.nan)).median()
    if not np.isfinite(vol_ratio) or vol_ratio <= 0:
        vol_ratio = 1.0

    log.update({
        "segments": "spliced",
        "overlap_days": int(len(ov)),
        "price_scale": float(scale),
        "max_ratio_dispersion": disp,
        "volume_scale": float(vol_ratio),
    })

    old_part = old[old["date"] < new["date"].min()].copy()
    for c in ["open", "high", "low", "close"]:
        old_part[c] = old_part[c] * scale
    old_part["volume"] = old_part["volume"] * vol_ratio

    merged = pd.concat([
        old_part[["date", "open", "high", "low", "close", "volume"]],
        new[["date", "open", "high", "low", "close", "volume"]],
    ], ignore_index=True).sort_values("date").reset_index(drop=True)

    # support frame: raw close pre-seam, split-adjusted close post-seam
    support = pd.concat([
        old_part.assign(raw_close=old.loc[old["date"] < new["date"].min(), "raw_close"].values
                        if "raw_close" in old.columns else np.nan)[["date", "raw_close"]],
        new[["date", "split_adj_close"]],
    ], ignore_index=True).sort_values("date")
    return merged, support


def build_spy():
    raw = pd.read_csv(f"{RAW}/misc/spy_raw.csv")
    raw.columns = [c.lower().strip() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").reset_index(drop=True)

    div = pd.read_csv(f"{RAW}/misc/spy_dividends.csv")
    div.columns = [c.lower().strip() for c in div.columns]
    div["date"] = pd.to_datetime(div["date"]).dt.normalize()
    div = div.sort_values("date")

    # Yahoo backward adjustment: at each ex-date, all prices strictly before
    # the ex-date are multiplied by (1 - div / close_{exdate-1}).
    raw = raw.set_index("date")
    factor = pd.Series(1.0, index=raw.index)
    for _, row in div.iterrows():
        exd = row["date"]
        prev = raw.index[raw.index < exd]
        if len(prev) == 0 or exd > raw.index.max():
            continue
        prev_close = raw.loc[prev[-1], "close"]
        k = 1.0 - row["dividend"] / prev_close
        factor.loc[factor.index < exd] *= k

    adj = raw.copy()
    for c in ["open", "high", "low", "close"]:
        adj[c] = raw[c] * factor
    adj = adj.reset_index()

    # Cross-check against brownbear SPY Adj Close (constant offset expected)
    bb = load_brownbear("SPY")
    check = {}
    if bb is not None:
        m = adj.merge(bb[["date", "close"]], on="date", suffixes=("", "_bb"))
        r = m["close_bb"] / m["close"]
        check = {
            "const_ratio_median": float(r.median()),
            "max_dev_from_const": float((r / r.median() - 1).abs().max()),
            "n_overlap": int(len(m)),
        }
    return adj[["date", "open", "high", "low", "close", "volume"]], check


if __name__ == "__main__":
    logs = {}
    support_frames = []
    spy, spy_check = build_spy()
    spy = spy[(spy["date"] >= START) & (spy["date"] <= END)]
    spy.to_csv(f"{OUT}/SPY.csv", index=False)
    logs["SPY"] = {"rows": len(spy), "first": str(spy['date'].min().date()),
                   "last": str(spy['date'].max().date()), "adj_check": spy_check}

    for t in ALL_TICKERS:
        log = {}
        old = load_deamoner(t)
        new = load_brownbear(t)
        if new is None:
            new = load_kronos(t)
            if new is not None:
                log["recent_source"] = "kronos"
        res = splice(old, new, t, log)
        if res is None:
            logs[t] = {"status": "MISSING"}
            continue
        merged, support = res
        merged = merged[(merged["date"] >= START) & (merged["date"] <= END)]
        merged = merged.drop_duplicates(subset="date").sort_values("date")
        if len(merged) < 100:
            logs[t] = {"status": "TOO_SHORT", "rows": len(merged)}
            continue

        ret = merged["close"].pct_change()
        log.update({
            "status": "ok",
            "rows": int(len(merged)),
            "first": str(merged["date"].min().date()),
            "last": str(merged["date"].max().date()),
            "max_abs_daily_ret": float(ret.abs().max()),
        })
        merged.to_csv(f"{OUT}/{t}.csv", index=False)
        sup = support.copy()
        sup["ticker"] = t
        support_frames.append(sup)
        logs[t] = log

    pd.concat(support_frames, ignore_index=True).to_csv(
        f"{SUPPORT}/raw_close_store.csv", index=False)

    with open(f"{SUPPORT}/build_price_panel_log.json", "w") as f:
        json.dump(logs, f, indent=1)

    ok = [t for t, l in logs.items() if l.get("status") == "ok" or t == "SPY"]
    disp_flags = {t: l for t, l in logs.items()
                  if l.get("max_ratio_dispersion", 0) > 0.002}
    ret_flags = {t: l["max_abs_daily_ret"] for t, l in logs.items()
                 if l.get("max_abs_daily_ret", 0) > 0.55}
    print(f"Built {len(ok)} tickers (incl. SPY). SPY adj check: {spy_check}")
    print(f"Splice-dispersion flags (>0.2%): {list(disp_flags)}")
    print(f"Extreme daily return flags (>55%): {ret_flags}")
    missing = [t for t, l in logs.items() if l.get("status") in ("MISSING", "TOO_SHORT")]
    print(f"Missing/short: {missing}")
