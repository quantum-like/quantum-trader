# Download all raw source files from public GitHub mirrors.
#
# The original paper uses Bloomberg data that cannot be redistributed. This
# session's network only reaches GitHub, so every series is sourced from
# public GitHub repositories that mirror Yahoo Finance / LBMA / EIA /
# Treasury.gov / SEC EDGAR data. Sources were verified by direct fetches
# (date ranges, adjustment conventions) before being used here.
#
# Layout produced:
#   ../raw_sources/deamoner/{TICKER}.csv     per-ticker full history -> 2019-04-18
#   ../raw_sources/brownbear/{TICKER}.csv    per-ticker 2019-01-02 -> 2026-07-30
#   ../raw_sources/kronos/{TICKER}.csv       fallback recent segment 2015 -> 2026-04
#   ../raw_sources/misc/...                  SPY raw, SPY dividends, GSPC, SP500TR,
#                                            VIX, gold, oil, DXY(2), AGG, yields,
#                                            CPI, UNRATE, FEDFUNDS
#   ../raw_sources/fundamentals/gtsf/{TICKER}.csv   SEC XBRL point-in-time 2009-2024
#   ../raw_sources/fundamentals/stockpup/{TICKER}.csv  quarterly 1993-2019
#   ../raw_sources/manifest.json             url, status, size, sha256 per file

import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

RAW = "../raw_sources"

# Universe from the replication repo's baseline.py (201 tickers)
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

# Current symbol -> symbol in the frozen Apr-2019 Deamoner snapshot.
# Covers renames/mergers between Apr 2019 and 2026 (the snapshot predates them),
# plus dot/dash share-class conventions.
DEAMONER_RENAME = {
    "META": "FB",       # renamed 2022
    "BRK-B": "BRK.B",   # dot class in Deamoner
    "ELV": "ANTM",      # Anthem -> Elevance 2022
    "RTX": "UTX",       # United Technologies -> Raytheon Technologies 2020
    "TT": "IR",         # Ingersoll-Rand plc -> Trane Technologies 2020
    "IR": "GDI",        # Gardner Denver -> Ingersoll Rand Inc 2020
    "LHX": "HRS",       # Harris -> L3Harris 2019-06
    "COR": "ABC",       # AmerisourceBergen -> Cencora 2023
    "TFC": "BBT",       # BB&T -> Truist 2019-12
    "DD": "DWDP",       # DowDuPont -> DuPont 2019-06 (partial history 2017+)
}
# GTSF SEC XBRL parser uses dots for share classes
GTSF_RENAME = {"BRK-B": "BRK.B"}
# stockpup snapshot (~2019) uses pre-rename symbols
STOCKPUP_RENAME = dict(DEAMONER_RENAME)

JOBS = []

def add(url, dest):
    JOBS.append((url, dest))

RAWGH = "https://raw.githubusercontent.com"

for t in TICKERS + ["SPY"]:
    d = DEAMONER_RENAME.get(t, t)
    add(f"{RAWGH}/Deamoner/ultimate-stock-machine-learning-training-dataset/master/full_history/{d}.csv",
        f"{RAW}/deamoner/{t}.csv")
    add(f"{RAWGH}/fja05680/brownbear/master/symbol-cache/{t}.csv",
        f"{RAW}/brownbear/{t}.csv")
    add(f"{RAWGH}/ArjunDivecha/Kronos/master/data/Universe500/{t}.csv",
        f"{RAW}/kronos/{t}.csv")

for t in TICKERS:
    g = GTSF_RENAME.get(t, t)
    add(f"{RAWGH}/GTSF-Quantitative-Sector/sec_parser/master/sec/data/processed/{g}.csv",
        f"{RAW}/fundamentals/gtsf/{t}.csv")
    s = STOCKPUP_RENAME.get(t, t)
    add(f"{RAWGH}/michaelcho1/stockpupr/master/company/{s}.csv",
        f"{RAW}/fundamentals/stockpup/{t}.csv")

MISC = {
    "spy_raw.csv": "ctj01/regime-discovery-cpp/HEAD/data/spy.csv",
    "spy_dividends.csv": "Gabe-Soler/dev_LS_gamma/HEAD/src/data/spy_dividends.csv",
    "spy_adjusted_crosscheck.csv": "nathberg1/ma-crossover-backtester/HEAD/data/spy.csv",
    "gspc.csv": "Jackscuster/fx-data/HEAD/data/gspc.csv",
    "sp500tr.csv": "eklyb94-blip/asset-allocation-dashboard/HEAD/sp500tr_history.csv",
    "vix_daily.csv": "datasets/finance-vix/main/data/vix-daily.csv",
    "gold_lbma.csv": "tomelam/PortfolioAnalyzer/main/data/reference/gold_lbma_usd_daily.csv",
    "wti_daily.csv": "datasets/oil-prices/main/data/wti-daily.csv",
    "dxy_2000_2024.csv": "whyyouj/market-sentiment-analysis/main/gold_historical/data/dxy.csv",
    "dxy_2014_2026.csv": "kittycapital/btc-asset-correlation/main/data/DXY.csv",
    "agg.csv": "giacomomaggiore/margin-investing-etfs/main/data/AGG.csv",
    "agg_crosscheck.csv": "ram-ki/101_formulaic_alphas/master/data/AGG.csv",
    "treasury_yields.csv": "fujiapple852/yield/master/data/us_treasury_yield_curve_history.csv",
    "cpi.csv": "shergin/malevich/main/demos/fred/data/cpi.csv",
    "unrate.csv": "shergin/malevich/main/demos/fred/data/unrate.csv",
    "fedfunds.csv": "shergin/malevich/main/demos/fred/data/fedfunds.csv",
    "company_tickers.json": "Camelket/pysec_downloader/master/resources/company_tickers.json",
}
for name, path in MISC.items():
    add(f"{RAWGH}/{path}", f"{RAW}/misc/{name}")

# SEC Financial Statement Data Sets (LFS -> media endpoint), for 2024-2025 fundamentals
MEDIA = "https://media.githubusercontent.com/media"
for q in ["2024q1", "2024q2", "2024q3", "2024q4", "2025q1", "2025q2"]:
    for part in ["sub", "num"]:
        add(f"{MEDIA}/Jesse3141/stock_data/main/small_data/parquet/quarter/{q}.zip/{part}.txt.parquet",
            f"{RAW}/fsds/{q}_{part}.parquet")


def fetch(job):
    url, dest = job
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return (url, dest, "cached", os.path.getsize(dest))
    r = subprocess.run(
        ["curl", "-sSL", "-m", "300", "-o", dest, "-w", "%{http_code}", url],
        capture_output=True, text=True
    )
    code = r.stdout.strip()
    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    if code != "200" or size == 0:
        if os.path.exists(dest):
            os.remove(dest)
        return (url, dest, f"FAIL:{code}", 0)
    # LFS pointer sanity check
    with open(dest, "rb") as f:
        head = f.read(64)
    if head.startswith(b"version https://git-lfs"):
        os.remove(dest)
        return (url, dest, "FAIL:lfs-pointer", 0)
    return (url, dest, "ok", size)


if __name__ == "__main__":
    results = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for res in ex.map(fetch, JOBS):
            results.append(res)
            if len(results) % 100 == 0:
                print(f"{len(results)}/{len(JOBS)} done")

    manifest = []
    for url, dest, status, size in results:
        entry = {"url": url, "dest": dest, "status": status, "size": size}
        if status in ("ok", "cached") and os.path.exists(dest):
            h = hashlib.sha256()
            with open(dest, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            entry["sha256"] = h.hexdigest()
        manifest.append(entry)

    os.makedirs(RAW, exist_ok=True)
    with open(f"{RAW}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)

    fails = [m for m in manifest if m["status"].startswith("FAIL")]
    print(f"\nTotal: {len(manifest)}, failed: {len(fails)}")
    for m in fails:
        print("  FAIL", m["url"], m["status"])
