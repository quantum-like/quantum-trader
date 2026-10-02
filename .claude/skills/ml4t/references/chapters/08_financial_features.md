# Chapter 8: Financial Feature Engineering

> Chapter 8 turns a trading narrative into a documented feature specification and then builds, aligns, selects and stress-tests the features. Its grammar is a three-step filter (horizon alignment, driver hypothesis, role separation) plus three construction knobs (reference frame, representation, aggregation), applied to price/volume, microstructure, cross-instrument, options-implied, fundamental, macro and calendar families. The position it argues: feature design is hypothesis design, not indicator collecting; slow data is more dangerous than fast data because lags, revisions and repeated values fabricate evidence; and every interaction, lookback and regime split is another test that must enter the multiple-testing accounting. Selection (HAC-IC, BH-FDR, bootstrap stability), robustness (breadth of the near-optimal region, RAS) and event studies are the search-control tools; breadth is reasoned about with IR ~ IC x sqrt(BR). Source: `08_financial_features/` README and notebooks 01-07 plus `case_study_feature_summary` (book PDF unavailable; wording of 8.1/8.5 is from README summaries).

## When to use this reference

- Building or reviewing features from OHLCV (returns, momentum, MA distance, volatility, volume, regime indicators) for any instrument.
- Choosing a volatility estimator (close-to-close vs Parkinson vs Garman-Klass vs Yang-Zhang vs ATR) for a given holding horizon.
- Computing intraday microstructure features from trades or ITCH (Kyle lambda, Amihud, Roll spread, OFI, trade intensity) and deciding which may be predictors vs feasibility inputs.
- Deriving futures carry/roll yield, term-structure slope/curvature, rolling beta, residual momentum, lead-lag or cross-sectional relative value.
- Extracting options-implied features (ATM IV, risk reversal, IV term slope, variance risk premium) from an options surface.
- Aligning fundamentals (XBRL) or macro (FRED) series to a daily panel point-in-time (ASOF joins, publication lags, calendar-day grids).
- Encoding calendar phase or time-to-event (sin/cos, capped countdown).
- Reducing a large candidate feature set (IC ranking, correlation filter, clustering, BH-FDR, bootstrap stability, LightGBM importance).
- Running a lookback/parameter sweep, regime-conditional IC, or signal x state interactions and judging robustness.
- Validating a signal with an event study (market model, CAAR, post-event window).
- Auditing a feature pipeline for leakage (full-sample z-scores, contemporaneous flow, period-end fundamentals, holdout contamination).
- Comparing feature inventories across case studies or reasoning about breadth vs IC.

## Core ideas (the why)

- **Feature design is hypothesis design.** Three-step filter: (1) horizon alignment: the feature's lookback/decay must match the label and execution horizon; (2) driver hypothesis: name the economic driver the feature proxies; (3) role separation: is it a signal, a state variable, or a feasibility/cost input?
- **Three construction knobs**: reference frame (what is subtracted or divided), representation, aggregation. Reference-frame choices change the hypothesis; smoothing and window length only control noise. Keep the two kinds apart when searching.
- **Signal vs state.** Signals carry directional expectation; states (vol regime, VIX regime, calendar phase, options-implied quantities, time-to-event) condition how signals behave. States enter marginally, as interactions, or as conditioning variables; the question is always whether the interaction adds incremental information.
- Every feature family encodes a specific economic claim, operates at a specific horizon, and fails recognizably when costs, latency or regime shifts are ignored.
- **Construction choices are part of the hypothesis** for cross-instrument features (maturity alignment, peer-set definition, options surface policy) and must be versioned with the features.
- **Slow data is more dangerous than fast data.** Reporting lags, revisions and repeated values create fake evidence easily; point-in-time correctness is the binding constraint of section 8.4.
- **Limits of direct aggregation (8.5).** Deterministic rolling transforms suffice for many features; latent states, conditional dynamics, cycle strength and path shape need fitted models (Chapter 9: HMM, Markov-switching GARCH, learned representations).
- **Interactions multiply degrees of freedom.** Signal x state interactions often give the practical improvement, but each is another test and must enter the searched-set count (8.6).
- **Efficiency vs target.** A more efficient estimator of the wrong quantity is still the wrong quantity: Parkinson/GK estimate within-session vol, not close-to-close daily vol.
- **"Illiquid" has to name a measure.** Kyle ratio and Amihud rank names by different raw quantities; a feasibility overlay must state which one it gates on and why.
- **Robustness is the breadth of the near-optimal region** on the response surface, not a scalar mean/std ratio. A single tall value surrounded by poor neighbours is the shape that does not survive new data.
- A ranked sweep is informative about shape even when no level is distinguishable from zero; ranking never licenses the claim that the top window carries an edge.
- In a signal-triggered event study the pre-event rise is the event definition showing up in the chart; only post-day-zero behaviour was unknown when the signal fired.
- **Registry for production, manual code for teaching.** Use `ml4t.engineer` (120 pre-built features) in pipelines; write manual implementations to understand the economics.
- **Fundamental Law**: IR ~ IC x sqrt(BR). Per-bet IC does not rise with universe size; IR does, because a 100x wider universe compensates a 10x smaller IC. BR means independent bets: 3,000 co-moving equities supply far fewer than 3,000.
- **IC is diagnostic, not a selector.** Configurations are chosen on validation backtest Sharpe; displaying a family's best IC is fine, choosing what to trade or fit from it is data mining.
- **Inventory and evaluation are separate layers.** The cross-case-study summary reads upstream registry results; IC, HAC significance and BH-FDR survival live in each case study's `13_model_analysis.py` to avoid a second source of truth.

## Method recipes (the how)

### Feature specification (section 8.1 grammar)

Write this down before computing anything:

| Field | Content |
|---|---|
| Horizon | Lookback/decay matched to the label and execution horizon |
| Driver | Named economic mechanism the feature proxies |
| Role | Signal / state / feasibility (cost) input |
| Reference frame | What is subtracted or divided (hypothesis-changing) |
| Representation, aggregation | Level, change, z-score, rank; smoothing, window (noise-control) |
| Timing | Lag assumption (when is the value knowable?) |
| Failure mode | How it breaks under costs, latency, regime shift |

### Returns and momentum (`01_price_volume_features`)

| Feature | Formula | Notes |
|---|---|---|
| Simple return | `(P_t - P_{t-h}) / P_{t-h}` | |
| Log return | `ln(P_t / P_{t-h})` | Use when additivity across time matters |
| Skip-1 momentum | `P_{t-1} / P_{t-21} - 1` | Drops the last day to avoid bid-ask bounce; default for short-horizon, daily-rebalanced strategies |
| Cumulative | `sum_{i=0..h} r_{t-i}` | |
| Overnight / intraday | `Open_t / Close_{t-1} - 1`; `Close_t / Open_t - 1` | |
| Short-term reversal | `-r_1d` | |
| Vol-scaled momentum | `r_21d / sigma_21d`, then cross-sectional percentile rank | Reshuffles ranks: high-vol high-momentum assets drop, steady trenders rise |

### Trend, reversal and MA distance

- MA distance, vol-scaled: `(P_t - SMA_21) / ATR_21` with `ml4t.engineer.features.volatility.atr`. Dollar and ATR-unit versions share sign and zero crossings but differ in where the extremes fall.
- Directional persistence: `dist-to-ma-atr = (P_t - MA_50) / ATR_14`; `|value| > 3` = extreme extension; usable as contrarian signal and as state.
- Rolling regression slope: OLS over a window with `x = [0..n-1]`, `beta = Cov(x, P) / Var(x)`, normalized by mean price: `rolling_regression_slope(df, price_col="close", period=21)`.

### Volatility estimators and vol state

| Estimator | Formula | Measures | Asymptotic efficiency vs CC | Use for |
|---|---|---|---|---|
| Close-to-close | `sigma_CC = sqrt(252) * std(r_t)` | Daily close-to-close vol | 1x | Overnight-held positions (simple, unbiased target) |
| Parkinson | `sigma^2 = (1 / (4 ln 2)) (ln H - ln L)^2` | Within-session vol | ~5x | Intraday range, execution risk |
| Garman-Klass | `sigma^2 = 0.5 (ln H - ln L)^2 - (2 ln 2 - 1)(ln C - ln O)^2` | Within-session vol | ~7x | Intraday range, execution risk |
| Yang-Zhang | Adds an explicit overnight term (exact formula not in notes) | Same quantity as CC, less noise | ~8-14x | Daily returns, overnight-held positions |
| ATR | Average true range in price units | A range, not return vol | n/a | Stops and normalization only |

Efficiency figures hold under driftless GBM with no overnight gap. Default `period=21`, annualized. Notebook-local `parkinson_vol(period=21) -> pl.Expr`, `garman_klass_vol(period=21) -> pl.Expr`; library `realized_volatility`, `yang_zhang_volatility`, `volatility_of_volatility` (signatures/defaults not shown in notes).

Vol state variables:
- Vol ratio short/long (expansion/contraction; windows not shown).
- Vol percentile over a 252-day history (preferred for granularity); vol decile (binned, for evaluation slicing).
- Vol-of-vol `volatility_of_volatility` (second moment; detects unstable regimes).

### Price-derived regime indicators (`ml4t.engineer.features.regime`)

| Function | Reading |
|---|---|
| `variance_ratio` | Lo-MacKinlay (1988) rolling; > 1 trending, < 1 mean-reverting |
| `fractal_efficiency` | 1 = straight line (trend), 0 = noise |
| `trend_intensity_index` | ADX-like trend strength |

Parameters (windows, lag q) not shown in notes.

### Volume and liquidity (daily)

- Dollar volume `V * Close`; relative volume `V_t / SMA_21(V)`; log-volume z-score (standardize `log V`); VWAP distance `(P_t - VWAP_t) / VWAP_t`.
- Write threshold rules against the log-volume z-score, not relative-volume ratios; the same numeric cut selects different day sets and does not transfer across instruments. Never use raw volume.

### Cross-sectional normalization

All cross-sectional statistics are computed per timestamp with `.over("timestamp")`:
- Rank `rank(f_{t,i}) / N_t` (outlier-robust default).
- Z-score `(f - mu_t) / sigma_t`; robust variant uses median/MAD.
- Relative-value z: `(r_a - rbar_t) / sigma_t` (21-day momentum in the example).

### Risk features and fractional differencing

- `ml4t.engineer.features.risk`: VaR (loss threshold at configured level), CVaR (expected loss beyond VaR), downside deviation (Sortino denominator), tail ratio (right tail / left tail). Confidence level default not shown.
- Fractional differencing `ml4t.engineer.features.fdiff.ffdiff`: default order around the middle of (0, 1), bracketed by computing two orders; truncated weights mean `d=1` only approximates `pct_change()`.

### Microstructure features (`02_microstructure_features`, NASDAQ ITCH)

Bars: `aggregate_to_bars(trades, interval="5m", stock_col="stock", price_col="price", volume_col="shares")` builds OHLCV with a per-trade buy/sell split via `effective_tick_rule` (Lee-Ready) applied before aggregation. Common intervals 1m/5m/15m intraday, daily for cross-sectional work. Loader raises if ITCH is missing (no synthetic fallback).

| Feature | Formula / implementation | Role and timing |
|---|---|---|
| Kyle lambda (installed) | `kyle_lambda(method="ratio")` = `mean_t(|r_t| / (V_t / Vbar_t))`, volume relative to its own rolling mean; unsigned. `method="regression"` raises `NotImplementedError`. Kyle (1985) proper = slope of `r_t = lambda S_t + eps` on signed flow | Feasibility (contemporaneous OK; size and filter) |
| Amihud | `(1/N) sum_t |r_t| / DollarVolume_t x 1e6` (return per million dollars) | Feasibility (contemporaneous OK) |
| Roll spread | `2 sqrt(-Cov(dP_t, dP_{t-1}))` | Cost-model input |
| OFI | `(V_buy - V_sell) / (V_buy + V_sell)` per bar from tick-rule-classified trades | Alpha; lag >= 1 bar (`shift(1)`) when used as predictor |
| Trade intensity | Activity regime (definition not shown) | Context; contemporaneous OK |
| Composite liquidity score | z-score each of three illiquidity metrics, sum | Ex-post description only when full-sample moments are used; not lookahead-safe |

Helpers: Pearson CI via Fisher-z with `z_crit=1.96`; cross-stock ordering check compares Kyle ordering vs median `|r|` and Amihud ordering vs median `|r|` per million dollars, using the median of per-bar ratios, not the ratio of medians.

### Carry, term structure, beta, residual momentum, lead-lag (`03_structural_cross_instrument_features`)

| Feature | Formula | Defaults / notes |
|---|---|---|
| Carry / roll yield | `carry = (F_near - F_far) / F_near * 365 / dT` | Positive = backwardation, negative = contango; `compute_roll_yield` pivots tenors; requires paired tenors |
| Term-structure slope (3 tenors) | `(F_0 - F_2) / F_0` | |
| Curvature | `F_0 - 2 F_1 + F_2` | |
| Rolling beta | `beta = Cov(r_a, r_m) / Var(r_m)` via `ml4t.engineer.features.cross_asset.beta_to_market` | 21d for fast regime, 63d for a stable estimate |
| Residual momentum | `r_resid = r_a - beta * r_m` | |
| Lead-lag | `corr(SPY_t, Sector_{t+lag})` | Positive lag-1 suggests SPY leads only if not a staleness artifact |
| Relative-value z | `(r_a - rbar_t) / sigma_t` cross-sectionally | |

Memory: this notebook peaks at ~7.4 GB RSS on the AlgoSeek options surface; use >= 8 GB RAM.

### Options-implied features (AlgoSeek S&P 500 options)

| Feature | Function and defaults | Reading |
|---|---|---|
| ATM IV | `compute_atm_iv(df, dte_range=(25, 35), moneyness_range=(0.98, 1.02))`; calls by moneyness, nearest strike per day; delta never read | Level of implied vol |
| Risk reversal | `RR_25d = IV_put - IV_call`; `compute_risk_reversal(df, dte_range=(25, 35), otm_put_range=(0.93, 0.97), otm_call_range=(1.03, 1.07))` | Positive = crash fear |
| IV term slope | `IV_short / IV_long`; `compute_iv_term_slope(df, short_dte=(8, 30), long_dte=(60, 180), moneyness_range=(0.98, 1.02))` | > 1 inverted (near-term stress), < 1 normal |
| VRP | `VRP_t = IV_30atm,t - RV_20,t`; `compute_vrp(iv_df, equity_df, rv_window=20)`, RV from split-adjusted closes | Same-day spread, a state; VRP < 0 is extreme and informative |

Options features are states, not signals. The seller's realized P&L needs horizon-matched `IV_t` vs RV over the following 30 days, not trailing RV20.

### Fundamentals: value/quality with ASOF alignment (`04_fundamentals_macro_calendar`)

- Value: book-to-market, earnings yield (inverse P/E), cash-flow yield, all over `market_cap`. Quality: ROE, ROA, accruals ratio (`compute_quality_factors`; exact accruals formula not shown).
- Source: SEC XBRL via `load_fundamentals()`. The notebook scaffolds `market_cap = 2 x book_value`, so its factor values are not real; production uses `load_firm_characteristics()` in `data/equities/loader.py` and the `us_firm_characteristics` case study.
- ASOF alignment: sort both frames by `[symbol, date]`, then `join_asof(left_on="timestamp", right_on="announcement_date", by="symbol")`, or `align_factors_to_daily(factor_df, daily_dates, announcement_col="announcement_date")`. The result is a step function that changes only on announcement dates.

### Macro features (FRED) and regimes

- Series: `t10y2y`, `vixcls`, credit spreads. FRED is a calendar-day grid (weekends/holidays included): 250 rows ~ 8 months. Declare windows in calendar days and assert grid spacing.
- `create_macro_trend_features(df, cols, windows=[21, 63, 252], conservative_lag=0)` (rolling z-scores etc.).
- `create_monthly_features(df, monthly_cols, conservative_lag=30)`: YoY/3m changes computed on the forward-filled series, with a 30-day publication lag.
- `create_relative_value_features`; `create_risk_regime_features`: VIX regime first bucket `vixcls < 15` (upper buckets inferred as 15-25 and > 25, not confirmed); credit regime from spread levels (thresholds not shown).
- Yield-curve slope `t10y2y`: EMA smoothing plus rolling z-score (span/window values not shown); z > +2 unusually steep, z < -2 inversion. VIX z-score +2 = elevated fear.

### Calendar encodings

- Cyclical: `x_sin = sin(2 pi m / 12)`, `x_cos = cos(2 pi m / 12)` via `ml4t.engineer.features.ml.cyclical_encode`; same for day-of-week.
- Time-to-event: `d = min(T_next - t, H_MAX)`, `H_MAX = 63` trading days; sawtooth resetting to the cap; proximity buckets pre-2d, pre-5d, normal, far. Encode phase and proximity only, never post-event realized outcomes. The notebook uses a synthetic earnings calendar.

### Signal x state interactions (`06_robustness_sensitivity`)

| Template | Construction | What it changes |
|---|---|---|
| Gating | Zero the signal in the high-vol regime | Active sample, turnover, capacity (costs active days) |
| Scaling | `signal / vol` | Position size |
| Conditional | IC per regime | Separate testable hypotheses |

Conditioning variable: 42-day realized vol on SPY, tercile thresholds from expanding-window percentiles (never a variable derived from the signal). `compute_regime_ic(prices_df, regime_df, symbols, lookback=126, forward_horizon=20)`. Build interactions only if the IC range across regimes >= `REGIME_IC_RANGE_MIN = 0.04`, and add each interaction to the searched-set count.

### Feature selection pipeline (`05_feature_selection`)

Input: `case_studies/etfs/features/financial.parquet` (prerequisite: run `case_studies/etfs/03_financial_features`), pre-holdout rows only per `setup.yaml` `evaluation.holdout_start`; `START_DATE = "2006-01-01"`; label `fwd_return_1m`.

| Step | Method | Function / threshold |
|---|---|---|
| 1 | Sanitize | Replace non-finite feature values with nulls first |
| 2 | Cross-sectional IC | Spearman per date, sorted by timestamp; i.i.d. t and Newey-West HAC t (regress IC series on a constant); drop features whose daily IC is undefined on most dates or has no variance |
| 3 | Correlation filter | `filter_correlated_features(corr_matrix, feature_names, ic_scores, threshold=CORR_THRESHOLD)`; greedily drop the lower-`|IC|` member of each pair with `|r| > 0.9` |
| 4 | Clustering | Complete linkage on correlation distance, `N_CLUSTERS = 10`; representative = member with largest `|IC|` |
| 5 | IC threshold | `|IC| >= IC_THRESHOLD = 0.01` |
| 6 | BH-FDR | `benjamini_hochberg_fdr` at `FDR_ALPHA = 0.05` on p-values from the Newey-West t-stats |
| 7 | Bootstrap stability | `bootstrap_ic(df, feature_cols, return_col="fwd_return_1m", n_bootstrap=50, sample_frac=0.8)`, rows with replacement; keep if sign consistency (max of strictly-positive share, strictly-negative share) >= `STABILITY_MIN_SIGN_CONSISTENCY_PCT = 80.0`; survivors replace `final_features` everywhere downstream |
| 8 | ML importance | `analyze_ml_importance` (LightGBM MDI and PFI); features high in both IC and ML importance are the strongest candidates |
| 9 | Verify | Print max pairwise `|corr|` of the selected set against `CORR_THRESHOLD` |

Output: selected feature list for Chapter 9 and two exported parquet files (names not stated). Whether the IC threshold applies before or after BH-FDR in the exported list is not stated.

### Robustness and sensitivity sweeps (`06_robustness_sensitivity`)

- IC helpers: `compute_ic_hac_stats`, ICIR = mean IC / std IC; `compute_momentum_ic_series(prices_df, symbols, lookback, forward_horizon=20)`.
- Sweep one knob (lookback) only; `compute_robust_region(sweep_results, metric="icir", threshold_pct=0.90)` = share of parameters within 90% of the peak. A single-point region is a no-go.
- RAS correction (`ml4t.diagnostic.evaluation.stats`) deflates the best IC for the number of effectively independent tests given correlated neighbours (function names/args not listed).
- Implementation variants: five momentum variants at `LOOKBACK = 63`, each scored with `_compute_variant_ic` (HAC IC); variant names not in notes (inference: raw, skip-1, log, vol-scaled, residual or rank).
- Data: `START_DATE = "2015-01-01"`, `END_DATE = "2024-01-01"` clamped to the holdout boundary; `FORWARD_HORIZON = 20`.

### Event studies (`07_event_studies`)

- Events: `generate_momentum_breakout_events(prices, lookback=20, min_gap_days=21)` (new 20-day highs; narrative says a 30-day gap was used).
- `compute_event_study(returns_df, benchmark_df, events_df, estimation_window=(-60, -6), event_window=(-5, 10), min_estimation_obs=30)`: market model `R_i = alpha + beta R_m` per event on the estimation window; `AR_t = R_actual - (alpha + beta R_market)`; AAR per relative day; CAAR = cumulative AAR; `SE(CAAR_t) = sqrt(sum_{s<=t} sigma_s^2 / n_s)` (variances add; not a rolling SE).
- Align by integer offset on a single wide shared date index, never by date-label lookup.
- `Z_CRIT = 1.96`; derive `CONF_LEVEL` in a later cell so Papermill overrides propagate.
- Library: `EventStudyAnalysis` with `EventConfig` / `WindowSettings` adds the mean-adjusted model, BMP robust variance (Boehmer et al. 1991), Corrado rank test and event-clustering handling.
- `validate_signal_with_event_study(returns_df, benchmark_df, signal_df, signal_column="signal", threshold=2.0, event_window=(-5, 10))`.
- Always report full-window and post-event-only (day 0 onward) rows separately; compare mean vs median, skewness and share of positive events.
- Data: `START_DATE = "2018-01-01"`, `END_DATE = "2024-01-01"`, `SYMBOLS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]`.

### Cross-case-study feature inventory and breadth vs IC (`case_study_feature_summary`)

- Reads each case study's `<case_dir>/features/financial.parquet` schema only (`pl.scan_parquet(...).collect_schema()`), the registry's best IC per family, and `DATASET_META` universe sizes; writes nothing.
- Panel state `resolve_feature_panel(id) -> (state, path)`: `readable` (panel exists) / `awaiting` (`features/` exists, no panel) / `unreachable` (`features/` absent). `refuse_a_partial_view` raises `RuntimeError` on any `unreachable` unless `ML4T_OUTPUT_DIR` is set.
- Feature count: schema columns minus `_ID_COLS = {"timestamp", "symbol", "product", "stock_id", "instrument_id", "date", "asset"}`; family = `name.split("_")[0]` if `_` in name else `"other"`.
- Family heatmap: recurring prefixes (breadth >= 2) sorted ascending by `(breadth, total_count, name)`; singletons (breadth == 1) in a companion bar; color `zmin=0`, `zmax = nanpercentile(cells, 90)`; zero cells blank.
- Breadth vs IC: `retired = frozenset().union(*(_retired_prediction_hashes(cs) for cs in CASE_STUDIES))`; `load_best_ic_per_family(exclude_prediction_hashes=retired)`; best row per case study by `ic_mean`; `estimated_ir = |IC| * sqrt(entities)`, skip if `entities == 0` or `|ic_mean| == 0`; bubble marker `sizemode="area"`, `MAX_MARKER_DIAMETER = 60`, `sizeref = 2 * max(IR) / 60**2`, `sizemin=4`.
- Universe sizes used as BR: etfs 99 (daily, 21d), crypto_perps_funding 21 (8-hourly, 8h), nasdaq100_microstructure 114 (15-min, 15m), sp500_equity_option_analytics 638 (daily, 5d), us_firm_characteristics 2483 (monthly, 1m), fx_pairs 20 (daily, 1d), cme_futures 30 (daily, 5d), sp500_options 612 (daily, dh-10d), us_equities_panel 3199 (daily, 1d).
- Sequencing: case-study feature notebooks -> `13_model_analysis.py` (IC, HAC, BH-FDR) -> registry -> this summary -> `09_model_based_features/case_study_temporal_summary`.

## Guardrails and pitfalls

Ordering and leakage
- **Unsorted data before rolling ops** — windows silently misalign. Sort once at load: `df.sort(["symbol", "timestamp"])`; assert grid spacing (04 asserts the FRED daily grid).
- **Full-sample z-score or rank** — uses future cross-sectional moments. Use `.over("timestamp")` for every cross-sectional statistic; never `pl.col(x).zscore()` on the whole column.
- **Survivorship in the universe** — current constituents only inflate results. Use point-in-time constituent lists.
- **Selection or sweeps seeing the holdout** — development decisions contaminate the test set. Read `evaluation.holdout_start` from the case study's `setup.yaml`, clamp `END_DATE`, compute labels from already-filtered prices.
- **Composite liquidity score with full-sample z-scores** — carries post-bar information into every bar. Use expanding-window percentiles, rolling z-scores, or build features inside walk-forward folds (`case_studies/*/data/features/`).

Microstructure
- **Contemporaneous OFI (or any flow feature) as predictor** — same-bar correlation is mostly the definition of an up bar. `shift(1)`; compare same-bar vs lagged correlation and treat the gap as the size of the look-ahead bias.
- **Tick rule on bar close instead of per trade** — OFI collapses to the sign of the bar return (+/-1). Classify at trade level inside `aggregate_to_bars`, then sum buy/sell volume.
- **Spread or depth computed from order flow** — cancellations and executions are not in arrival flow; the book has memory. Reconstruct LOB state (Chapter 3); keep flow and state in separate feature classes.
- **Kyle ratio and Amihud treated as interchangeable** — they rank by different quantities (typical `|r|` vs `|r|` per dollar), can invert orderings and scale differently with volume units. Multiply volumes by a constant and check which moves; name the measure in the feasibility overlay; never port thresholds between them.
- **`kyle_lambda(method="regression")`** — raises `NotImplementedError`; the installed default `"ratio"` has no sign. Do not describe it as the Kyle (1985) slope.
- **Single ITCH session** — one day, one venue cannot establish general behaviour. Read printed spans; widen samples before quoting point estimates.

Volatility and volume
- **Parkinson/GK to size overnight positions** — they estimate within-session vol and read low vs close-to-close. Use close-to-close or Yang-Zhang for daily returns; Parkinson/GK for intraday range and execution risk.
- **ATR as a volatility estimator** — it is a dollar range. Use ATR for stops and normalization only.
- **Ratio thresholds on relative volume** — the same numeric cut selects different days than a log-volume z-score and does not transfer across instruments. Write rules on the log-volume z-score; measure the overlap of the two selections.
- **Naming a rolling median a "percentile rank"** — different transforms; be precise in feature names.

Cross-instrument and options
- **Lead-lag from stale closes** — a less liquid ETF closing before SPY fakes a lead. Verify the leading instrument actually traded at the lead timestamp.
- **Missing deferred contracts** — carry becomes NaN. Require paired tenors; handle rolls explicitly (see `cme_futures`).
- **Reading carry sign from panel heights** — panels have different scales. Count signs / read printed shares.
- **Changing options quote-selection policy mid-backtest** (delta convention, interpolation, maturity mapping) — contaminates every options-derived feature. Version construction choices with the features; keep surface policy fixed.
- **Reading VRP (IV30 - trailing RV20) as a vol-selling return** — windows barely overlap and no position is held. Compute horizon-matched `IV_t` vs following-30-day RV; treat the spread as a state; report share positive, mean and median together (left skew makes mean and median disagree on sign); check every symbol is above 50% positive before claiming "VRP is positive".

Slow data (fundamentals, macro, calendar)
- **Period-end dates for fundamentals** — data was not knowable then. ASOF join on `announcement_date`, both frames sorted by `[symbol, date]`, `by="symbol"`; sorting by date only is wrong.
- **Forward-filled quarterly data inflates sample size** — each observation repeats for ~a quarter of trading days. Compute the empirical inflation factor; do not treat daily rows as independent.
- **Macro publication lag and revisions** — monthly data arrives 2-4 weeks late and initial estimates are revised. `conservative_lag=30` for monthly series; vintage-aware availability rules; limit forward-fill; compute YoY/3m changes on the forward-filled series.
- **Copying window lengths from price bars to FRED grids** — FRED has calendar days, so 250 rows ~ 8 months. Declare windows in calendar days; assert spacing.
- **Scaffolded `market_cap = 2 x book_value`** — book-to-market becomes constant with no cross-section. Use `load_firm_characteristics()` and the `us_firm_characteristics` case study.
- **Integer month encoding** — implies Dec > Jan. Use sin/cos; encode phase and proximity, never post-event realized outcomes.

Selection statistics
- **NaN vs null in Polars** — `drop_nulls` keeps NaN, which corrupts `pl.corr` and misgroups clusters; `pl.corr` returns NaN on constant cross-sections. Replace non-finite with null up front; filter daily ICs on finiteness.
- **Unordered `group_by` output** — Newey-West on an unordered IC series is meaningless. Sort the per-date IC series by timestamp.
- **Pooled IC** — conflates time-series drift with cross-sectional predictive power. Compute IC per date, then average.
- **i.i.d. t-stat on daily IC** — ICs are serially correlated (overlapping information sets, slow common factors). Feed Newey-West HAC t-stats to BH-FDR.
- **No multiple-testing correction** — a share of null features equal to alpha looks significant by chance. BH-FDR at `FDR_ALPHA = 0.05`.
- **Ward linkage on correlation distance** — Ward assumes Euclidean geometry. Use complete (or average) linkage; complete avoids chaining on moderately correlated panels.
- **Stability cut on share of positive IC** — contradicts `|IC|`-based selection; a reliable negative edge is a feature. Cut on sign consistency (max of positive share, negative share) at 80%; derive the reported sign from counts, not mean IC.
- **Row-wise bootstrap pooled across dates** — breaks cross-sectional structure. Block-bootstrap by date in production; treat the pooled version as a quick filter.
- **A cut that only changes a printed table** — exported selection disagrees with the notebook. Survivors must replace `final_features` before model fit, importances, correlation check and exports.

Sweeps and interactions
- **Changing several parameters at once** — nothing is attributable. One knob at a time; combine per-knob choices afterwards.
- **Picking the best lookback from a sweep** — the maximum of N correlated estimates is biased upward. Apply RAS; require a broad near-optimal region (>= 90% of peak), not a single point.
- **Conditioning on a variable derived from the signal** — circular; fabricates regime dependence. Pre-defined clean conditioning variable (42-day SPY RV) with expanding-window tercile thresholds.
- **Three regimes = three tests with chosen split points** — sign flips across regimes are sample claims. Require IC range >= 0.04 before building interactions; add interactions to the searched-set count.
- **"Gating avoids drawdowns" read as a profitable signal** — gating may only lift a negative IC toward zero. Check the level of both series; account for lost active days (breadth vs lift).

Event studies
- **Pre-event drift in signal-triggered studies** — events are selected on the pre-event path. Report the post-event window separately every time.
- **Extending the CAAR window until the band excludes zero** — the band widens mechanically; this is a search. Fix the event window in advance.
- **Event clustering on the same day** — cross-sectional correlation inflates t-stats. Portfolio-level returns, clustered SEs, or one event per day; library BMP/Corrado tests.
- **Overlapping estimation and event windows** — contaminates the normal-return estimate. Minimum gap 20-30 days per symbol (`min_gap_days=21` in code, 30 in text), shorter estimation windows, or calendar-time portfolios.
- **Confounding events (earnings, macro)** — wrong attribution. Screen, matched controls, subsamples.
- **Event windows located by date label** — duplicate or missing dates misalign windows. Integer offsets on a single shared wide date index.
- **~100 breakouts on 5 ETFs** — wide CIs mean "widen the sample", not a result.

Notebook and infrastructure
- **Papermill parameter injection** — anything derived inside the tagged cell runs on defaults. Derive dependent constants (`CONF_LEVEL`) in a later cell.
- **03 notebook memory** — ~7.4 GB RSS on the options surface; use >= 8 GB RAM.

Cross-case-study summary
- **Partial-view denominator** — counts and "which prefixes look universal" over a subset state a wrong denominator silently. `refuse_a_partial_view` raises on any `unreachable` case study when `ML4T_OUTPUT_DIR` is unset; distinguish `awaiting` (legitimate) from `unreachable`.
- **Retired prediction sets winning the IC comparison** — the coverage bar is a population maximum, so a superseded generation that scored every decision day clears it and wins on IC; estimated IRs inherit the error. Pass `exclude_prediction_hashes=_retired` into `load_best_ic_per_family`; filtering afterwards deletes the family instead of falling back to the runner-up. Measured 2026-09-18: 3 of 9 case studies were topped by a retired set when omitted.
- **Prefix-as-taxonomy over-count** — ~100 prefixes across nine markets overstate distinct ideas (asset-specific measures; naming drift such as `bb`/`bollinger`, `r12`/`r36`/`past`). Keep breadth >= 2 in the heatmap; collapse singletons to a count; read the axis as prefixes; enforce consistent naming upstream (inference).
- **Feature-space size inflates the multiple-testing burden** — more features = more tests to survive BH-FDR. Rely on upstream HAC + BH-FDR survival counts in `13_model_analysis.py`, not raw IC.
- **IC as a selector** — configurations are chosen on validation backtest Sharpe. Only display best-IC rows.
- **Comparing ICs scored on different days** — not comparable. `load_model_ic` keeps maximum-coverage prediction sets per (split, family, label); check `ic_n_days` equal and load rows by `prediction_hash` for same-days certainty.
- **Estimated IR is arithmetic, not evidence** — `IR = IC * sqrt(N)` applied to the plotted IC. Confirm with measured backtest Sharpe/IR.
- **Breadth means independent bets** — correlated universes supply far fewer; discount BR for cross-sectional correlation before trusting IR (inference on mechanism; the notebook names the caution without a formula).
- **Plotly marker size as diameter** — area scales with IR^2. Use `sizemode="area"` with `sizeref = 2 * max / d_max**2`.
- **Heatmap palette flattened by a catch-all bucket** — winsorize color at the 90th percentile while printing exact counts.
- **Schema introspection** — identifier columns counted as features inflate counts. Subtract `_ID_COLS`; names without underscore fall into `other` (inference).
- **Two sources of truth** — recomputing IC/FDR in the summary could diverge from the per-case-study analysis. Summary reads registry results only.

## Decision rules and defaults

| Decision | Rule / default |
|---|---|
| Feature spec | Horizon aligned to label; named driver; role (signal / state / feasibility); reference frame, representation, aggregation declared; lag and failure mode written down |
| Returns | Skip-1 for short-horizon daily-rebalanced strategies; log returns when additivity matters |
| Volatility | Parkinson or GK for intraday/session questions; close-to-close or Yang-Zhang (same quantity, ~8-14x efficiency) for overnight-held positions; `period=21`, annualized |
| Vol state | 252-day percentile over decile for granularity; decile only for evaluation slicing |
| Volume | Thresholds on log-volume z-scores; relative or dollar volume, never raw |
| Cross-sectional | Ranks for outlier robustness; median/MAD z-scores; always `.over("timestamp")` |
| Fractional differencing | Order around the middle of (0, 1) |
| Microstructure timing | OFI = alpha, lag >= 1 bar; Kyle lambda and Amihud = feasibility (contemporaneous, size/filter); trade intensity = context (contemporaneous); Roll spread = cost model input |
| Bars | `aggregate_to_bars(interval="5m")`; tick rule per trade; 1m/5m/15m intraday, daily cross-sectional |
| Beta window | 21d fast regime, 63d stable |
| Options | ATM dte 25-35, moneyness 0.98-1.02; RR puts 0.93-0.97, calls 1.03-1.07; term slope short 8-30 vs long 60-180 dte; VRP `rv_window=20`; term slope > 1 = inverted; VRP < 0 extreme; options features are states |
| Lead-lag | Tradeable only if the leading instrument traded at the lead timestamp |
| Fundamentals | ASOF on `announcement_date`, sorted `[symbol, date]`, `by="symbol"` |
| Macro | Lag monthly data 30+ days (`conservative_lag=30`); windows `[21, 63, 252]` for trend features in calendar days; VIX z > +2 elevated fear; yield-curve z > +2 steep, < -2 inversion; VIX regime first bucket < 15 (15-25, > 25 inferred) |
| Calendar | sin/cos encoding; time-to-event cap `H_MAX = 63`; buckets pre-2d, pre-5d, normal, far |
| Selection order | sanitize non-finite -> per-date IC with HAC t -> corr filter `|r| > 0.9` -> complete-linkage clusters (10) with max-`|IC|` representative -> `|IC| >= 0.01` -> BH-FDR alpha 0.05 -> bootstrap 50 x 80% rows, sign consistency >= 80% -> LightGBM MDI/PFI cross-check -> verify max pairwise `|corr|` vs 0.9 |
| Holdout | Read `evaluation.holdout_start` from `setup.yaml`; clamp `END_DATE`; labels from filtered prices |
| Robustness go/no-go | Robust region = parameters with ICIR >= 90% of peak; single-point region = no-go; apply RAS to the peak; one knob at a time |
| Regime interactions | Only if IC range across regimes >= `REGIME_IC_RANGE_MIN = 0.04`; conditioning variable 42d SPY RV, expanding-window terciles; count every interaction as a test |
| Event study | Estimation (-60, -6), event (-5, +10), `min_estimation_obs=30`, min gap 21-30 days, `Z_CRIT = 1.96`, signal `threshold=2.0`; fix windows in advance; read post-event rows only for signal-triggered events; size against aggregates where mean ~ median, skew ~ 0, share positive well above 0.5 |
| Breadth vs IC | `IR = |IC| * sqrt(entities)`; 100x wider universe compensates 10x smaller IC; IC 0.01 on 3,000 bets (IR ~0.55) beats IC 0.03 on 20 (IR ~0.13); discount BR for correlation |
| Registry comparisons | Always pass `exclude_prediction_hashes` (union of `_retired_prediction_hashes(cs)`); `split="validation"`; choose configurations on validation backtest Sharpe, not IC |
| Summary prerequisites | Every case study must have `features/financial.parquet`; any `unreachable` without `ML4T_OUTPUT_DIR` -> abort |
| Production | Registry/library (`compute_features`) for pipelines; manual code for understanding |

## Code patterns and APIs

Imports
```python
from ml4t.engineer import compute_features                       # names (defaults) or dicts (custom params)
from ml4t.engineer.core.registry import get_registry             # 120 features: formula, parameters, input type, description
from ml4t.engineer.features.volatility import atr, realized_volatility, yang_zhang_volatility, volatility_of_volatility
from ml4t.engineer.features.regime import fractal_efficiency, trend_intensity_index, variance_ratio
from ml4t.engineer.features.risk import ...                      # VaR, CVaR, downside deviation, tail ratio
from ml4t.engineer.features.fdiff import ffdiff
from ml4t.engineer.features.microstructure import effective_tick_rule, kyle_lambda, amihud_illiquidity, roll_spread_estimator, trade_intensity
from ml4t.engineer.features.cross_asset import beta_to_market
from ml4t.engineer.features.ml import cyclical_encode
from ml4t.diagnostic.metrics import pooled_ic, compute_ic_hac_stats, analyze_ml_importance
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr   # plus RAS functions
from ml4t.diagnostic.evaluation import EventStudyAnalysis
from ml4t.diagnostic.config import EventConfig
from ml4t.diagnostic.config.event_config import WindowSettings
from utils.style import ...                                      # registers the ml4t Plotly template; COLORS, show_plotly_with_alt
```

Point-in-time cross-sectional z-score (never a whole-column `zscore()`)
```python
(pl.col("mom") - pl.col("mom").mean().over("timestamp")) / pl.col("mom").std().over("timestamp")
```

ASOF fundamentals join
```python
daily_df.sort(["symbol", "timestamp"]).join_asof(
    fundamental_df.sort(["symbol", "announcement_date"]),
    left_on="timestamp", right_on="announcement_date", by="symbol")
```

Lag flow features before using them as predictors
```python
df["signal"] = df["ofi"].shift(1)
```

Schema-only feature count and family prefix
```python
schema = pl.scan_parquet(path).collect_schema()
names = [c for c in schema.names() if c not in _ID_COLS]
fam = lambda n: n.split("_")[0] if "_" in n else "other"
```

Best IC per case study with retirement exclusion
```python
retired = frozenset().union(*(_retired_prediction_hashes(cs) for cs in CASE_STUDIES))
df = load_best_ic_per_family(exclude_prediction_hashes=retired)
best = df.sort("ic_mean", descending=True, nulls_last=True).group_by("case_study").first()
```

Plotly bubble area encoding
```python
sizeref = 2.0 * max(vals) / (MAX_D ** 2)
go.Scatter(marker=dict(size=vals, sizemode="area", sizeref=sizeref, sizemin=4))
```

Implementation rules (notebook 01): sort once at load; put all transforms in a single `with_columns` for parallelism; vol-scale for comparability; library for production; `.over("timestamp")` for cross-sectional statistics.

Repo helpers and config keys
- `case_studies/etfs/setup.yaml` -> `evaluation.holdout_start`; `case_studies/etfs/features/financial.parquet`; `case_studies/*/data/features/` (walk-forward feature construction; current layout writes to `<case_dir>/features/`).
- `data/equities/loader.py::load_firm_characteristics()`; `load_fundamentals()` (SEC XBRL).
- `case_studies.utils.analytics`: `DATASET_META` (`frequency`, `entities`, `horizon`), `CASE_STUDY_IDS`, `PRIMARY_LABELS`, `SHORT_NAMES`, `load_best_ic_per_family(families=None, *, split="validation", case_studies=None, use_primary_label=True, exclude_prediction_hashes=None) -> pl.DataFrame` (columns `case_study, display_name, family, config_name, label, ic_mean, ic_n_days, prediction_hash`), `load_model_ic(..., require_full_coverage=True, exclude_prediction_hashes=...)` (emits `coverage_enforced`; registries predating the coverage backfill do not enforce coverage).
- `case_studies.utils.paired_metrics._retired_prediction_hashes(case_study) -> set` (shared with `populate_paired_metrics` and `20_strategy_synthesis/01_aggregate_synthesis`).
- `utils.paths.get_case_study_dir(id) -> Path`, `utils.paths.display_path(path)`.
- Env var `ML4T_OUTPUT_DIR` redirects case-study outputs to a scratch root (pytest sets it).
- Run: `uv run python 08_financial_features/<notebook>.py`; test mode `uv run pytest tests/test_chapter_notebooks.py -v -k "08_financial_features"`; Docker image `ml4t`.

Notebook constants
| Notebook | Constants |
|---|---|
| 01 | `START_DATE="2015-01-01"` |
| 04 | `H_MAX=63` |
| 05 | `START_DATE="2006-01-01"`, `N_BOOTSTRAP=50`, `BOOTSTRAP_SAMPLE_FRAC=0.8`, `CORR_THRESHOLD=0.9`, `IC_THRESHOLD=0.01`, `FDR_ALPHA=0.05`, `STABILITY_MIN_SIGN_CONSISTENCY_PCT=80.0`, `N_CLUSTERS=10` |
| 06 | `START_DATE="2015-01-01"`, `END_DATE="2024-01-01"`, `ROBUST_THRESHOLD_PCT=0.90`, `REGIME_IC_RANGE_MIN=0.04`, `FORWARD_HORIZON=20`, `LOOKBACK=63` |
| 07 | `START_DATE="2018-01-01"`, `END_DATE="2024-01-01"`, `Z_CRIT=1.96`, `SYMBOLS=["SPY","QQQ","IWM","TLT","GLD"]` |
| summary | `START_DATE=None` (full dataset), `MAX_MARKER_DIAMETER=60` |

## Evidence from the book

Runtime numbers are printed by the notebooks and mostly absent from the notes; qualitative results stated in the text:

- Skip-1 vs plain 21d momentum (ETFs, 2015+): correlation close enough to one that they carry very similar information at this horizon.
- MA distance in dollars vs ATR units: same sign and zero crossings, but the largest positive distance falls in different months.
- Volatility estimators (ETF, last two years): close-to-close reaches the highest peak and is most jagged but is not the highest line most days; Yang-Zhang is highest on the majority of days; Parkinson and GK are never highest and usually lowest. On full history Parkinson/GK land on the measured within-session reference vol (not below CC by noise); YZ tracks CC through the peak.
- Relative volume vs log-volume z-score: the same numeric cut selects overlapping but different day sets; the relative-volume level at a z cut is a range, not a number.
- Vol-scaled momentum reshuffles cross-sectional ranks: high-vol high-momentum assets drop, steady trenders rise.
- ITCH single session: Kyle ratio and Amihud correlations are negative (Kyle falls through the afternoon, Amihud peaks midday); multiplying volume by a constant leaves Kyle unchanged and scales Amihud by the reciprocal; across three names the two liquidity orderings partly reverse, each matching its raw quantity.
- OFI: same-bar correlation with returns is larger and its 95% Fisher interval excludes zero; the lagged correlation's interval spans zero; the gap is the size of the look-ahead bias.
- Carry snapshot (CME, 30 products): CL closer to backwardation than GC on this sample; received asset-class stories (CL tightness, GC contango, ZN coupon vs financing, ES dividends vs financing) must be checked against printed shares, not assumed.
- VRP (S&P 500 options): positive on most days with mean below median (left-skewed, occasional deep negative excursions); share of positive days not uniform across names.
- Fundamentals scaffolding: book-to-market constant across rows; earnings yield varies across quarters.
- Yield-curve inversions: several shaded episodes last a single day (invisible at 900 px over 25 years).
- Feature selection (ETF case study, pre-holdout): clustering typically removes more redundancy than the pairwise filter (max surviving `|corr|` well below 0.9 when it worked).
- Lookback sweep (ETF momentum, 20-day forward): robust region is a single point; no lookback has mean IC distinguishable from zero (all error bars cross zero, peak included).
- Regime-conditional IC (42d SPY RV terciles, lookback 126): mean IC highest in low-vol, smaller in normal, negative in high-vol; only low-vol significant; IC range clears 0.04 comfortably.
- Gating: gated rolling-rank IC sits above raw through most of the window, especially at raw troughs, but both are below zero most of the time (momentum IC negative more often than not on this panel/span); gating costs active days.
- Event study (~100 momentum breakouts on 5 ETFs, 2018-2024): CAAR rises before day 0 by construction, then flat to gently declining inside a widening band; full-window and post-event-only CAR rows disagree about the sign.
- Retired-generation contamination (measured 2026-09-18, `exclude_prediction_hashes` omitted): cme_futures retired 0.0443 vs live 0.0430; fx_pairs retired 0.0150 vs live 0.0149; us_equities_panel retired `gbm/leaves_63_huber` 0.0343 vs live `gbm/leaves_63_mae` 0.0311 (reported configuration changed too); all three estimated IRs were wrong by the corresponding amount. Implied live IRs (inference): cme_futures ~0.24, fx_pairs ~0.07, us_equities_panel ~1.76, so the 3,199-name universe reaches ~7x the IR of CME futures on a lower IC.
- Feature-count figure: bars broadly similar; tallest = CME futures, shortest = crypto perpetuals.
- Family heatmap: volatility, return, momentum, Sharpe and a residual catch-all prefix recur across most case studies; implied-vol and variance-premium prefixes only in the two options-bearing case studies; singleton prefixes present in each case study with panels, ETF and futures studies carrying the most (alt text written when 7 case studies had panels, inference).
- Breadth-vs-IC scatter: no trend between `|IC|` and universe size (~20 to several thousand instruments); highest `|IC|` from a ~100-instrument case study (ETFs 99 or NASDAQ-100 114, inference); two of the widest universes sit at middling/low IC; largest IR circles at the widest universes.

## Related references

- `chapters/03_market_microstructure.md` — LOB reconstruction for state features (spread, depth) that cannot be derived from order flow.
- `chapters/06_strategy_definition.md` — holdout rule (`02_cv_foundations`) that governs `evaluation.holdout_start` clamping in selection and sweeps.
- `chapters/07_defining_the_learning_task.md` — forward-return labels (`fwd_return_1m`) and the `10_ml4t_library_ecosystem` registry tour (config-driven batch RSI/MACD/Bollinger).
- `chapters/09_model_based_features.md` — HMM, Markov-switching GARCH, learned representations; consumes the selected feature list; `case_study_temporal_summary` companion.
- `chapters/11_ml_pipeline.md` — `us_firm_characteristics` panel (Chen-Pelger-Zhu 2020) and production `load_firm_characteristics()`.
- `chapters/12_gradient_boosting.md` — LightGBM MDI/PFI importance used in the selection cross-check.
- `chapters/20_strategy_synthesis.md` — `01_aggregate_synthesis` shares the `_retired_prediction_hashes` helper and registry conventions.
- `chapters/02_financial_data_universe.md` — point-in-time constituent lists and survivorship (inference on placement).
- `case_studies/etfs.md` — `03_financial_features`, 100-ETF universe, residual-momentum IC test; input to notebook 05.
- `case_studies/cme_futures.md` — 30 products, roll handling and paired tenors for carry.
- `case_studies/nasdaq100_microstructure.md` — ITCH-based microstructure features at 15-min horizon.
- `case_studies/sp500_equity_option_analytics.md` — aggregate Greeks as OI proxy; options-implied features on equities.
- `case_studies/sp500_options.md` — straddles; VRP and surface-policy versioning.
- `case_studies/us_firm_characteristics.md` — real fundamentals cross-section replacing the scaffolded `market_cap`.
- `case_studies/crypto_perps_funding.md` — 8-hour funding as a carry-like structural feature.
- `case_studies/fx_pairs.md` — 20-pair universe, carry features; narrow-breadth example.
- `case_studies/us_equities_panel.md` — 3,199-name universe; widest-breadth example in the breadth-vs-IC view.
- `libraries/ml4t_engineer.md` — feature registry (`compute_features`, `get_registry`), volatility/regime/risk/microstructure/cross_asset/ml modules.
- `libraries/ml4t_diagnostic.md` — IC/HAC stats, BH-FDR, RAS, `EventStudyAnalysis`, `analyze_ml_importance`.
- `libraries/ml4t_data.md` — loaders for ETF, ITCH, CME, AlgoSeek, XBRL and FRED inputs (inference).
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `companion_repo.md`, `workflow.md` — cross-cutting versions of the rules above.
- Further reading:
  - Kyle (1985) Continuous Auctions and Insider Trading; Amihud (2002) Illiquidity and stock returns; Cont et al. (2014) Price Impact of Order Book Events; Easley et al. (2021) Microstructure in the Machine Age; Lee-Ready tick rule.
  - Parkinson (1980); Garman & Klass (1980); Yang & Zhang (2000) drift-independent volatility.
  - Lo & MacKinlay (1988) variance ratio; Mandelbrot fractal efficiency; Lopez de Prado (2018) Advances in Financial Machine Learning.
  - Jegadeesh & Titman (1993); Asness et al. (2013) Value and Momentum Everywhere; Levine & Pedersen (2016) Which Trend Is Your Friend?; Hurst et al., A Century of Evidence on Trend-Following; Novy-Marx (2015); Fama & French (1992/1993); Piotroski (2000); Cochrane (2011).
  - Carr & Wu (2009) Variance Risk Premiums; Ang & Timmermann (2011) Regime Changes.
  - Harvey, Liu & Zhu (2016) ...and the Cross-Section of Expected Returns; Kakushadze et al. (2015) 101 Formulaic Alphas; Meinshausen & Buhlmann (2010) stability selection; Paleologo (2025) Elements of Quantitative Investing.
  - MacKinlay (1997) event studies; Boehmer, Musumeci & Poulsen (1991) BMP test; Corrado rank test; Grinold's Fundamental Law of Active Management.

## Glossary

- **Horizon alignment** — matching a feature's lookback/decay to the label and execution horizon.
- **Driver hypothesis** — the named economic mechanism a feature proxies.
- **Role separation** — classifying a feature as signal, state, or feasibility/cost input.
- **Signal feature** — carries directional expectation about forward returns.
- **State variable** — conditions how signals behave; not traded directly.
- **Feasibility feature** — contemporaneous liquidity/cost input for sizing and filtering (Kyle lambda, Amihud).
- **Flow vs state feature** — events aggregated over a window (trades, volume imbalance) vs a snapshot (spread, depth).
- **Reference frame / representation / aggregation** — the three construction knobs; only the first changes the hypothesis.
- **Skip-1 momentum** — return excluding the most recent day to avoid bid-ask bounce.
- **ATR** — average true range in price units; normalization and stops, not a vol estimator.
- **Parkinson / Garman-Klass / Yang-Zhang** — range-based vol estimators; first two within-session, YZ includes overnight.
- **Estimator efficiency** — asymptotic variance ratio vs close-to-close under driftless GBM.
- **Variance ratio** — Lo-MacKinlay random-walk test as a rolling feature; > 1 trending.
- **Fractal efficiency** — path straightness, 1 trend, 0 noise.
- **Kyle lambda (ratio)** — mean `|r|` / relative volume; unsigned price-impact proxy.
- **Amihud** — mean `|r|` per million dollars traded.
- **Roll spread** — `2 sqrt(-Cov(dP_t, dP_{t-1}))`, implied bid-ask spread.
- **OFI** — order flow imbalance `(V_buy - V_sell) / (V_buy + V_sell)` from tick-rule-classified trades.
- **Tick rule (Lee-Ready)** — classify trade side from the sign of the price change.
- **Carry / roll yield** — annualized near-minus-far futures spread; backwardation positive, contango negative.
- **Residual momentum** — return minus beta times market return.
- **Risk reversal** — 25-delta put IV minus call IV; crash-fear skew.
- **IV term slope** — short-dated / long-dated ATM IV; > 1 inverted.
- **VRP** — implied minus trailing realized vol (here IV30 - RV20); a state.
- **Surface policy** — options quote-selection conventions (delta, interpolation, maturity mapping).
- **ASOF join** — align slow data to daily rows using the last value announced at or before each date.
- **Publication lag** — delay between a macro period and its release; handled with `conservative_lag`.
- **Cyclical encoding** — sin/cos mapping of periodic calendar variables.
- **Time-to-event** — capped countdown to the next scheduled event (`H_MAX = 63`).
- **IC** — per-date cross-sectional Spearman correlation of feature and forward return; `ic_mean` in the registry.
- **ICIR** — mean IC / std IC.
- **Newey-West / HAC t-stat** — t-statistic with autocorrelation-robust standard errors.
- **BH-FDR** — Benjamini-Hochberg false discovery rate control.
- **Stability selection** — bootstrap test of IC sign consistency.
- **MDI / PFI** — mean decrease in impurity / permutation feature importance.
- **Response surface** — metric as a function of a swept parameter.
- **Robust region** — parameters within `ROBUST_THRESHOLD_PCT` of the peak.
- **RAS** — correlation-aware multiple-testing deflation for parameter snooping.
- **Gating / scaling / conditional** — the three signal x state interaction templates.
- **AR / AAR / CAR / CAAR** — abnormal return, average AR per relative day, cumulative AR per event, cumulative average AR.
- **BMP test / Corrado test** — event-induced-variance-robust test / non-parametric rank test.
- **Fundamental Law** — IR ~ IC x sqrt(BR), BR = number of independent bets.
- **Feature family** — name prefix before the first underscore (`mom_21` -> `mom`); no underscore -> `other`.
- **Breadth (of a family)** — number of case studies in which the family has at least one feature; recurring >= 2, singleton == 1.
- **Panel state** — `readable` / `awaiting` / `unreachable` for `features/financial.parquet`.
- **Coverage bar** — restriction to maximum-coverage prediction sets per (split, family, label).
- **Prediction hash** — identity of a prediction set in the registry; carries its split; used for tie-breaking and exact-row loading.
- **Retired prediction set** — superseded model generation; excluded inside the loader via `exclude_prediction_hashes`.
- **`ML4T_OUTPUT_DIR`** — env var redirecting case-study outputs to a scratch root (pytest).
