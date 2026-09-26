# Reproduction: "Do Market Regimes Improve Machine-Learning Stock Ranking?"

Reproduction of Pascual Miralles & Alfeus (2026), SSRN 7200998, using the
authors' public replication code
([carlospascualmiralles/carlospascualmiralles-RegimeAware_ML_PO](https://github.com/carlospascualmiralles/carlospascualmiralles-RegimeAware_ML_PO))
with freely available substitute data. The paper's Bloomberg inputs cannot be
redistributed, and this environment can only reach GitHub, so every input
series was rebuilt from public GitHub mirrors of Yahoo Finance / LBMA / EIA /
Treasury.gov / SEC EDGAR data.

## Pipeline

```
code/data_download.py        # fetch ~1,000 source files (manifest with sha256)
code/build_price_panel.py    # per-ticker adjusted OHLCV 2003-2026 (two-source splice)
code/build_market_macro.py   # HMM market data + cross-asset/rates/macro features
code/build_fundamentals.py   # daily point-in-time market cap, P/E, P/B, ROE, D/E
code/expanding_hmm_regimes.py    # authors' script, unmodified
code/baseline.py                 # authors' script, unmodified
code/baseline_different_models.py# authors' script (MODEL_TYPE now a CLI arg)
code/hmm_regime_experiments.py   # reconstructed from paper Sec. 4.5 (not in public repo)
code/portfolio_optimization.py   # authors' script, unmodified
code/transaction_costs.py        # authors' script, unmodified
```

## Data substitutions (Bloomberg -> public mirrors)

| Series | Source |
|---|---|
| Stock OHLCV 2003-2018 | `Deamoner/ultimate-stock-machine-learning-training-dataset` (Yahoo snapshot, Apr 2019) |
| Stock OHLCV 2019-2026 | `fja05680/brownbear` symbol-cache (Yahoo snapshot, Jul 2026); `ArjunDivecha/Kronos` fallback |
| SPY (total-return adjusted) | raw SPY (`ctj01/regime-discovery-cpp`) + dividend history (`Gabe-Soler/dev_LS_gamma`), Yahoo-convention reconstruction; matches an independent adjusted file to a constant within 0.26% |
| VIX | `datasets/finance-vix` (CBOE) |
| S&P 500 index (HMM) | `Jackscuster/fx-data` |
| Treasury yields 3M/2Y/10Y | `fujiapple852/yield` (Treasury.gov daily par curve, = FRED DGS*) |
| US agg bond return | AGG dividend-adjusted close, `giacomomaggiore/margin-investing-etfs` |
| Dollar index | DX-Y.NYB from `whyyouj/market-sentiment-analysis` + `kittycapital/btc-asset-correlation` (identical on overlap) |
| Gold | LBMA USD daily, `tomelam/PortfolioAnalyzer` |
| WTI oil | `datasets/oil-prices` (EIA) |
| CPI / unemployment / fed funds | FRED copies in `shergin/malevich` |
| Fundamentals 2004-2008 | stockpup quarterly mirror `michaelcho1/stockpupr` |
| Fundamentals 2009-2024 | SEC XBRL point-in-time per filing, `GTSF-Quantitative-Sector/sec_parser` |
| Fundamentals 2024-2025 | SEC Financial Statement Data Sets, `Jesse3141/stock_data` (2024q1-2025q2) |

The two price segments are spliced on their 2019-01..2019-04 overlap
(median ratio; dispersion checked per ticker). Fundamentals are true
point-in-time (filing dates; stockpup rows lagged 45 days). Market cap uses
an anchor-and-scale construction that is immune to split-basis errors;
splits are price-verified pre-2019, whitelisted post-2019, with a
seam-residual check that catches post-mid-2025 splits absent from all filing
sources. See headers of the build scripts for details.

## Known deviations from the paper

1. **Data vendor.** Yahoo-derived prices vs Bloomberg adjusted OHLCV; SEC
   filings vs Bloomberg fundamentals. Fundamental definitions are
   approximations (e.g. D/E = long-term debt / common equity; negative-equity
   names have undefined P/B / ROE / D/E and fall back to the pipeline's
   cross-sectional median fill).
2. **Universe edges.** BK (Bank of NY Mellon) lacks a 2019+ price source and
   drops out of the panel after April 2019. DD has no pre-2017 history
   (DowDuPont). A handful of renamed tickers (META<-FB, ELV<-ANTM, RTX<-UTX,
   TT<-IR, IR<-GDI, LHX<-HRS, COR<-ABC, TFC<-BBT) are mapped explicitly.
3. **FF5+MOM alpha not reproduced.** Kenneth French's library is unreachable
   from this environment and no public mirror extends past 2018. The SPY
   single-factor alpha (the paper's headline) is reproduced.
4. **Feature ablation "+macro-cycle"** runs without ISM PMI and consumer
   confidence (no accessible public source); flagged where reported.
5. **Walk-forward retraining** (paper Table 21) is reproduced only if compute
   allows; it retrains the 7-seed ensemble at each of 60 rebalances.

Everything else — universe list, feature engineering, target construction,
splits, hyperparameters, seeds, portfolio construction, statistical tests —
follows the authors' code exactly.
