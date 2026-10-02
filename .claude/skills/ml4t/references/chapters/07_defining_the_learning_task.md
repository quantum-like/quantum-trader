# Chapter 7: Defining the Learning Task
> Chapter 7 turns raw-but-validated market data into a learning task that can be evaluated honestly: split-aware preprocessing (every scaler, imputer, encoder and winsor bound fitted on the training fold only and refit per walk-forward fold), execution-consistent labels (fixed-horizon, percentile, triple-barrier, trend-scanning) whose anchor sits where the fill happens, a univariate triage ladder (correct at decision time -> associated -> well-shaped -> economically feasible) with IC inference that respects overlap (HAC, block bootstrap, block permutation), search accounting that corrects the best-of-N for selection (BH-FDR, Holm, RAS, DSR, PBO, MinTRL), and mechanism-plausibility checks that reject confounded or timing-artifact features early. The position it argues: the payoff of this discipline is comparability, auditability and leakage protection, not cosmetic cleanliness; a maximum is not an estimate; and a label describes a trade, so it must be built the way the trade is executed. Chapter notebooks live under `07_defining_the_learning_task/` (run `uv run python 07_defining_the_learning_task/<notebook>.py`; test mode `uv run pytest tests/test_chapter_notebooks.py -v -k "07_defining_the_learning_task"`; Docker image `ml4t`; outputs under `get_chapter_dir(7) / "output"`).

## When to use this reference
- Running a first data-health survey on a new panel (index integrity, duplicates, coverage, outliers, calendar anomalies) before any cleaning.
- Building or auditing a preprocessing pipeline: winsorization, scaling, imputation, categorical encoding, gap handling, as-of joins of mixed-frequency tables, walk-forward refits.
- Choosing and configuring a label: fixed-horizon return/binary, time-series or cross-sectional percentile, triple barrier (fixed or ATR), trend scanning, meta-labeling and bet sizing.
- Calibrating take-profit / stop-loss widths or ATR multiples from MFE/MAE instead of round numbers.
- Evaluating a single factor: IC, ICIR, quantile spread, monotonicity, horizon and decay, turnover and break-even cost, fold-level stability, permutation nulls, AUC for binary labels.
- Putting a standard error, HAC t-statistic, bootstrap CI, effective sample size or minimum track record on an IC series with overlapping labels.
- Screening many candidate factors or parameter variants and deciding which survive BH-FDR, Holm, Rademacher Anti-Serum, Harvey t > 3, DSR or PBO.
- Checking whether a surviving feature-label association is a shared-endpoint artifact, shared-driver confound, regime-mixing effect or collider-induced correlation.
- Deciding between ml4t-engineer's feature registry and manual Polars code, and writing point-in-time-safe Polars transformations.

## Core ideas (the why)

### 7.1 Preprocessing
- Data quality determines model quality: diagnose first (no mutation), then clean with an audit trail, then fit.
- A check must be able to fail: sorting by (symbol, time) before testing monotonicity returns True for any input.
- Column coverage is not panel coverage: null counts only see rows that exist; whole missing rows/months are invisible until you reindex onto a continuous time range and compare against an independently defined expected universe.
- Detect events from the column that records them (`split_ratio`), not from a threshold on a derived quantity whose bounds you have not checked: a raw return is bounded below by -1, so `|r| > 1` is one-sided by construction.
- US-equity missingness is overwhelmingly entry/exit (listing/delisting): 7.1 is a universe-construction problem, not an imputation problem; survivorship-free panels must be rebuilt as of each decision date.
- Preprocessing parameters (mean/std, median/mode, encoder vocabulary, winsor bounds) are learned on training data only and refit at each walk-forward fold boundary; the train/test boundary is what gives evaluation its meaning. The per-feature shift is small but accumulates across dozens of features and inflates Sharpe/IC.
- Financial returns reject normality (fat tails, asymmetry) -> robust scaling and winsorization, not plain z-scoring; prices carry a unit root, returns do not -> features and labels in returns/differences.

### 7.2 Labels
- Labels must be execution-consistent: the anchor (close-to-close vs next-open) nets to zero on average and changes every individual label; a model scored on a price it could not transact at is leakage.
- Triple-barrier labels describe the trade, not the market: mass pinned to two barrier values, the rest of the path discarded. Right when the strategy really exits at those thresholds; wrong when forecasting magnitude.
- Cross-sectional percentile labels fix class balance by construction and push all variation into the cut point (a few percent in a quiet month, tens of percent in a dislocation): stable classes, unstable economic meaning.
- Overlapping labels cut the effective sample size by roughly the horizon; uniqueness weights and the sequential bootstrap tilt against overlap but cannot manufacture independent observations.
- Trend-scanning's t-statistic is a regression on price levels with autocorrelated residuals; it rejects under its own null at almost any threshold. Use the sign/horizon, not the t.
- Barrier widths are measured, not chosen: a width is a property of one instrument at one horizon; regime conditioning matters more than instrument choice.

### 7.3 Triage and inference
- Factor research is an auditable screening process (correct at decision time -> associated -> well-shaped -> economically feasible), not a hunt for one attractive statistic.
- An IC has no meaning without its standard error (sigma_IC / sqrt(T)); a near-zero ICIR means the mean sits inside the daily dispersion. Read sign before magnitude: a strongly ordered signal pointing the wrong way is inverted, not weak.
- Pooled IC vs fold IC: a gap between statistics computed over different dates is evidence of nothing; first recompute the pooled statistic on the fold dates only (period effect vs aggregation effect).
- A null must be as dependent as the data: permute on the same dates as the observed statistic and in blocks of the label horizon.
- Naive t-stats on autocorrelated IC overstate significance; the HAC bandwidth must look at the label horizon and signal persistence, not just T. Statistical significance is separate from tradeability after costs.

### 7.4 Search accounting
- A maximum is not an estimate: under the null every factor IC is centred on zero and E[max IC] ~ sqrt(2 ln N) x sigma_IC. Reporting the selected factor's IC as its IC is the whole selection-bias problem; nothing was mismeasured, the selection step is simply missing from the number.
- Factor-zoo arithmetic (Harvey, Liu, Zhu 2016): with several hundred published factors, expected false discoveries = candidate count x alpha, double digits even if no factor is real -> t > 3.0 for a new factor instead of t > 2.0.
- Corrections take p-values as given: HAC first (widens one test's interval), then multiplicity (sets the threshold that test must clear given all the others). Both apply, separately.
- FDR vs FWER are different promises: BH controls E[FP / discoveries] <= alpha (screening many factors); Holm controls P(any FP) <= alpha (deploying real capital).
- Correlated candidates are fewer effective tests: 100 lookback variants of one factor are not 100 independent trials; Rademacher complexity measures the realized search size and yields a sharper bound than independence.
- Separate exploration from confirmation: screen on the first portion of data, promote on fold stability rather than peak performance, confirm on held-out data with the reduced set.
- DSR, PBO and MinTRL answer different questions: is the best Sharpe of N genuinely positive given the null's expected maximum; does the IS-best rank in the bottom half OOS; how long a record is needed before a Sharpe is trustworthy (lengthened by non-normality and by the search). PBO's reference point is 0.5, not 0.
- The FWER correction prices the search through trial dispersion: `variance_trials = 0` switches the correction off.

### 7.5 Mechanism plausibility
- Mechanism checks are triage, not causal identification: one feature-outcome pair at a time, blind to multivariate confounding, unable to separate confounding from effect modification (Ch15). REVISE is the expected outcome; where/when a feature works matters more than the label.
- State the mechanism first (confounder, mediator, collider) so the checks have an interpretation; never condition on a variable caused by both feature and label.
- Horizon selection is a modeling decision that changes which features look informative: cross-asset momentum needs 6-12 month lookbacks and monthly+ horizons; short-term features are noise at every horizon on ETFs.
- Shared-endpoint artifacts are provable without data (a feature ending at t and a label starting at t share p_t), so failing the shift check means re-measure, not disproof.
- Library for production, manual code for teaching and custom factors.

## Method recipes (the how)

Notebook prerequisite chain: 01 is the entry point (needs no prior notebook) and establishes the ETF shape that 10 reuses; 06_ic_inference supplies the per-factor HAC p-values that 07_multiple_testing consumes; 05/06/07 feed 08_causal_sanity_checks. Each notebook is a Jupytext percent-format `.py` (`07_defining_the_learning_task/0X_*.py`) paired with a `.ipynb`.

### Recipe 1: Data-quality diagnostics (`07_defining_the_learning_task/01_data_quality_diagnostics`)
Survey all seven datasets (ETFs, US equities, crypto perps, crypto premium index, CME futures, FX pairs, firm characteristics); emit diagnostic dicts/figures only, no mutation. ~9 GB RSS (>= 16 GB RAM). Entry point of the pipeline.

| Function | Purpose | Defaults |
|---|---|---|
| `check_index_integrity(df, time_col="timestamp", symbol_col="symbol")` | dtype, per-symbol monotonicity in arrival order, uniqueness of (date, symbol) | - |
| `check_duplicates(df, key_cols, value_cols=None)` | exact and near-duplicates (same key, different values) | - |
| `coverage_report(df, time_col, symbol_col, value_cols)` | missingness by field, asset, period | - |
| `coverage_heatmap(df, time_col, symbol_col="symbol", value_col="close", max_symbols=50, order_by="first_observation"\|"symbol")` | (plotly Figure, matrix); reindex months onto a continuous range so an all-missing month is a blank row; order by first observation to expose the listing staircase and delisting edge | max_symbols 50 |
| `distribution_summary(df, fields, extreme_threshold=5.0)` | stats + extreme flags | \|z\| > 5.0 |
| `outlier_flags(df, price_cols, volume_col="volume", return_col, max_return_threshold=2.0, high_spike_ratio=1.5, low_spike_ratio=0.5)` | domain violations (negative price/volume) + single-bar spike reversals | 2.0 / 1.5 / 0.5 |
| `rag_status(passed, warning_threshold, value)` | Red/Amber/Green for the summary dashboard | - |
| `ml4t.diagnostic.evaluation.distribution.tests.jarque_bera_test` | joint skew/kurtosis normality test | - |
| `ml4t.diagnostic.evaluation.stationarity.analyze_stationarity` | ADF/KPSS/PP consensus (KPSS p-values from a lookup table clip at the bounds -> `InterpolationWarning` silenced; formal treatment Ch9) | - |

Calendar facts to encode in checks:
- Crypto perps/premium: 8-hour bars at 00:00/08:00/16:00 UTC, 24/7; mid-panel delistings cause coverage drops.
- FX: 4-hour bars; market closes Fri 17:00 EST, reopens Sun 17:00 EST (= 21:00-22:00 UTC depending on DST). Sunday bars at/after reopen are valid; Saturday and Sunday-daytime bars are anomalies.
- CME futures: session-aligned daily bars with roll continuity; `load_cme_futures` returns three rows per session (tenors 0, 1, 2).
- Firm characteristics: Chen-Pelger-Zhu monthly panel, 46 characteristics + `ret`, `timestamp`, `split`; rows filtered upstream by the paper's availability rules (do not reconstruct dropped rows).
- US equities: start at `US_EQUITIES_START_DATE`, ship a `split_ratio` column, survivorship-free. ETFs: 100 Yahoo ETFs across 9 asset classes (outliers are adjustment artifacts).

### Recipe 2: Cleaning and split-aware preprocessing (`07_defining_the_learning_task/02_preprocessing_pipeline`)
~12 GB RSS (>= 24 GB RAM). Output is in-memory cleaned frames only; case studies clean via their loaders.

Sequence: diagnose -> dedupe -> domain filters -> extreme-return filter -> spike detection -> winsorize (train bounds) -> scale/impute/encode (train fit) -> refit per fold -> serialize.

| Step | Function | Defaults |
|---|---|---|
| Dedupe | `remove_duplicates(df, key_cols, strategy="keep_last"\|"keep_first"\|"drop_all")` -> (df, n_removed) | keep_last |
| Gaps | `fill_expected_gaps(df, time_col, symbol_col, method="flag_only"\|"ffill"\|"interpolate", max_gap_days=5)` | flag_only, 5 days |
| Domain rules | `apply_domain_filters(df, rules: dict[str, Callable[[pl.Expr], pl.Expr]])` -> (df, counts per rule) | exact rules dict not in notes |
| Spikes | `spike_filter(df, price_col="close", threshold=0.5, symbol_col="symbol", time_col="timestamp", action="flag"\|"remove"\|"replace")` | 0.5, flag |
| Winsorize | `winsorize_panel(df, fields, limits=(0.01, 0.99), by_period=True, period_col="timestamp")`; clips per period (cross-sectional); tails become point masses at the bounds | (0.01, 0.99) |
| Fit / transform | `SplitAwarePreprocessor.fit(train_df)` / `.transform(df)` / `.fit_transform(train_df)` / `.save(path)` / `.load(path)`: learns mean/std, median/mode, encoder vocabulary on train only (teaching class, pickle). Production: `ml4t.engineer.preprocessing.StandardScaler` (same semantics; tiny sample-vs-population std difference) | - |
| Encoding | categorical encoder fitted on train with `handle_unknown="ignore"` -> unseen test categories map to all-zero vectors | - |

US-equities deep-clean order: (1) drop penny stocks (< $1); (2) domain filters (negative price/volume, high < low, etc.); (3) remove implausibly large one-day moves (`MAX_ABS_DAILY_RETURN = 2.0`, i.e. |r| > 2, well above the -1 lower bound so only upward jumps can fire; most removed rows are reverse-split days per `split_ratio`); (4) spike detection. Winsorize before any scaler fit. ETFs need only a lighter cleanup. Walk-forward: refit the preprocessor at every fold boundary on an expanding window; parameters drift with regimes.

### Recipe 3: Cross-dataset alignment and Polars patterns (`02_preprocessing_pipeline`, `10_ml4t_library_ecosystem`)
- Same frequency (crypto spot/perp 8-hour bars): inner join on exact timestamps; basis = premium x 100 (percentage points).
- Daily prices to monthly characteristics (as-of logic): `daily.with_columns(month=pl.col("timestamp").dt.truncate("1mo")).join(monthly, left_on=["permno","month"], right_on=["permno","timestamp"])`.
- General point-in-time join: `df.join_asof(other, on="date", by="symbol", strategy="backward")` with both sides sorted. Never `forward`.
- Put every transformation in ONE `with_columns()` with `.over("symbol")` windows (chained calls lose Polars parallelism); `pl.scan_parquet(path).filter(...).collect()` for predicate pushdown.
- Sort on native timestamps, never on stringified dates.

### Recipe 4: Label methods (`07_defining_the_learning_task/03_label_methods`; `ml4t.engineer.labeling`, `ml4t.engineer.config.labeling.LabelingConfig`)
Teaching labels on real ETF data (SPY single-asset; full universe cross-sectional). Production labels live in each case study's `02_labels` notebook, reading horizon and label set from `config/setup.yaml` via `utils.artifact_specs.resolve_label_horizon(case_study_id, label, setup)` (reads `definition.horizon` from the label spec, then `setup.labels.horizons[label]`, else the CV buffer).

| Method | Config | Use when | Caveat |
|---|---|---|---|
| Fixed horizon | `LabelingConfig.fixed_horizon(horizon=10, return_method="returns"\|"binary"\|"log_returns", threshold=None)`; binary = +1 if r > 0 else -1 | factor timing (monthly) -> fixed horizon; stat arb (intraday) -> binary; forecasting magnitude -> returns (regression) | binary inherits sample drift (majority class wins without learning) |
| Time-series percentile | threshold = top quartile of the trailing year for one instrument; fields `percentile_window`, `n_bins` (defaults not visible) | single-asset thresholding | threshold moves with trailing vol AND drift (correlation with vol well below 1) |
| Cross-sectional percentile | rank assets at each t, label top/bottom quintiles | equity/ETF rotation, cross-sectional ranking | point-in-time by construction; cut point drifts (few % vs tens of %) |
| Triple barrier | `LabelingConfig.triple_barrier(upper_barrier=0.02, lower_barrier=0.01, max_holding_period=20, side=1, trailing_stop=False)`; upper -> +1, lower -> -1, time barrier -> sign/return at expiry | active trading where stops are part of the strategy | defaults are placeholders; calibrate from MFE/MAE; close-only mode flatters the trade |
| ATR triple barrier | `LabelingConfig.atr_barrier(atr_tp_multiple=2.0, atr_sl_multiple=1.0, atr_period=14, max_holding_period=20, side=1, trailing_stop=False)` + `atr_triple_barrier_labels(data, config=cfg)`; or per-row barrier columns passed as strings to `LabelingConfig.triple_barrier` (notebook 03 route; both below) | daily strategies with changing volatility | measured p75 stop for SPY/ES > 3x ATR, so 1x is hit by ordinary movement |
| Trend scanning | `LabelingConfig.trend_scanning(min_horizon=5, max_horizon=20, t_value_threshold=2.0, step=1)`: regress price level on time over windows 5..20 (16 candidates), keep max \|t\| | trend following, sign/horizon only | the t is not a significance statistic |
| Meta-labeling | primary model gives +1/-1; triple barrier says whether each signal paid; secondary model learns when/how much; size with `compute_bet_size(probability, method="sigmoid"\|"linear"\|"discrete", scale=1.0, threshold=0.5)` | sizing on calibrated probabilities | case studies do not use it (one model, one horizon, sizing via allocator) |

Other `LabelingConfig` fields: `price_col="close"`, `timestamp_col="timestamp"`, `group_col`, `data_contract`, `weight_scheme`, `weight_decay_rate` (option values only partially visible; inference from field names).

ATR barriers, two routes (there is no `LabelingConfig.atr_triple_barrier`; calling it raises AttributeError). (a) Library: `LabelingConfig.atr_barrier(...)` sets `method="atr_barrier"`; label with `ml4t.engineer.labeling.atr_triple_barrier_labels(data, atr_tp_multiple=None, atr_sl_multiple=None, atr_period=None, max_holding_bars=None, side=None, price_col=None, timestamp_col=None, group_col=None, trailing_stop=False, *, config=None, contract=None)`. (b) Notebook 03 (the chapter's executed variant): compute `true_range`, `atr_dollar = true_range.rolling_mean(14)`, then `upper_barrier_pct = 1.0 * atr_dollar / close` and `lower_barrier_pct = 0.5 * atr_dollar / close`, and pass the column names as strings: `LabelingConfig.triple_barrier(upper_barrier="upper_barrier_pct", lower_barrier="lower_barrier_pct", max_holding_period=20, side=1)` with plain `triple_barrier_labels(spy_atr, config=atr_config, price_col="close", timestamp_col="timestamp")`.

Bet-size formulas: linear = 2(p - 0.5); sigmoid = 2 / (1 + exp(-scale (p - 0.5))) - 1; discrete = 1 if p > threshold else 0. Calibrate probabilities first.

Anchor alignment (decisive per label; the label gate "anchor matches execution"): close-to-close (decide at close t, measure close_t -> close_{t+H}) vs open-anchored (decide at close t, fill at open_{t+1}, measure from there). Mean difference ~0, std ~ one daily move. Build the label where `decision.execution_delay` in `config/setup.yaml` says the fill happens; `h` counts sessions and the frame must be sorted by `symbol, timestamp`:

```python
h = 21
df = df.sort("symbol", "timestamp").with_columns(
    # next_bar_open, exit at the open h sessions after entry (open-to-open)
    (pl.col("open").shift(-(h + 1)) / pl.col("open").shift(-1) - 1).over("symbol").alias(f"fwd_ret_{h}d"),
    # next_bar_open, exit at the close of session t+h (open-to-close; the sp500_equity_option_analytics form)
    (pl.col("close").shift(-h) / pl.col("open").shift(-1) - 1).over("symbol").alias(f"fwd_ret_{h}d_oc"),
    # close-to-close proxy (what the etfs / fx_pairs / cme_futures / us_equities_panel 02_labels write)
    (pl.col("close").shift(-h) / pl.col("close") - 1).over("symbol").alias(f"fwd_ret_{h}d_cc"),
)
# library route to the open-to-open label: label open_t -> open_{t+h}, then move it onto the decision row
lab = fixed_time_horizon_labels(df, horizon=h, method="returns", price_col="open", group_col="symbol", timestamp_col="timestamp")
lab = lab.with_columns(pl.col(f"label_return_{h}p").shift(-1).over("symbol").alias(f"fwd_ret_{h}d"))
```

What the case studies declare (`decision.snapshot` / `execution_delay`) and what their `02_labels` write: etfs, us_equities_panel, us_firm_characteristics and fx_pairs declare `next_bar_open` but write close-to-close labels (`close.shift(-h) / close - 1`; the ETF audit record says `anchor: "adjusted close at t"` and names "close-to-close is not the backtest's next-open execution" as a known limitation); cme_futures declares `monday_open` and writes settlement-to-settlement on `adj_close`; sp500_equity_option_analytics declares `monday_open` and is the one open-anchored label (entry `adj_open.shift(-1)`, exit `adj_close.shift(-h)`); sp500_options `next_session_close`; crypto_perps_funding `at_funding_timestamp` (close of the bar completing at t); nasdaq100_microstructure `1_bar` (entry and exit at the VWAP of the next bar and of bar t+H). When no open series exists, or when you keep a close-to-close label, record it as a proxy in the label audit (`anchor`, `resolves`) and let the backtest fill at the next open so the gap is measured there (Ch16 / Ch18), never left implicit.

Triple-barrier engine: `triple_barrier_labels(data, config, price_col=None, high_col=None, low_col=None, timestamp_col=None, group_col=None, calculate_uniqueness=False, uniqueness_weight_scheme="returns_uniqueness"|"uniqueness_only"|"returns_only"|"equal", contract=None, open_col=None)`. With high/low/open: touch detected on the bar range, gap-through executed at the open. Without: close-only test, trade booked at the barrier price not the crossing close. Diagnose with `label_diagnostics(df, label_col, timestamp_col="timestamp", title_prefix="")` (distribution stability + class balance over time).

Trend-scanning null: demean SPY daily log returns and shuffle (keeps dispersion/shape, removes drift and ordering). Bonferroni alpha/k moves rejection by a few points only; the binding problem is autocorrelated residuals.

### Recipe 5: Uniqueness weights and sequential bootstrap (AFML Ch4; `03_label_methods`)
- Label i is alive over bars [i, i+H]; w_i = mean over its span of 1 / c(u), where c(u) = number of concurrent labels at bar u. Compute per symbol, not across the cross-section.
- N_eff = sum w ~ N/H for every-bar fixed-horizon labels; `measure_n_eff(n_labels, h)` -> (mean uniqueness, N_eff).
- Set `calculate_uniqueness=True` in `triple_barrier_labels` and pick `uniqueness_weight_scheme`.
- Sequential bootstrap draws tilt toward less-overlapped labels; expect nearly identical uniqueness histograms with a small positive mean shift. It does not remove the overlap.

### Recipe 6: MFE/MAE barrier calibration (`07_defining_the_learning_task/04_maximum_favorable_adverse_excursion`)
Long entry at the close of bar t, horizon H; the window opens at t+1.
- MFE(t) = max_{u in [t+1, t+H]} (high_u / entry_t - 1); MAE(t) = max_{u in [t+1, t+H]} (1 - low_u / entry_t); both >= 0; short reverses highs/lows.
- `compute_mfe_mae(prices, timestamp_col, close_col, horizon_bars, high_col="high", low_col="low", side=1|-1, unit="pct"|"decimal")` builds H shifted columns (shift -1..-H) and takes `max_horizontal` / `min_horizontal` (O(n x H) memory). Large-H alternative: `pl.col("high").shift(-1).reverse().rolling_max(H).reverse() / pl.col("close") - 1` (the `shift(-1)` is mandatory).
- True range TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|); Wilder ATR_t = (n-1)/n ATR_{t-1} + 1/n TR_t (EMA, alpha = 1/n). `compute_atr(prices, timestamp_col, high_col, low_col, close_col, period=14, unit="pct"|"price")`; library `ml4t.engineer.features.volatility.atr` (panel-aware TA-Lib ATR replacement).
- Calibration: take-profit from the MFE quantile and stop from the MAE quantile at the same percentile (p50/p75/p90); validate by running `triple_barrier_labels` and inspecting the hit-type distribution.
- ATR multiple for a volatility-scaled stop: `atr_scaled_stop_multiple(mfe_mae, atr, quantile=0.75)` = p75 of the per-entry ratio MAE(t)/ATR(t), NOT p75(MAE)/mean(ATR).
- Regime conditioning: excursions by volatility tercile. Multi-horizon: `ml4t.diagnostic.evaluation.excursion.analyze_excursions` (close-based, signed MAE -> negative `mae_p50_pct`; the manual block uses high/low, unsigned, so magnitudes differ).
- Histograms share axes; limit = the `HIST_CLIP_Q = 0.995` quantile of the wider series.
- Reference calibrations: SPY daily over 21 sessions; BTC premium index hourly bars over 8 hours (one funding cycle); ES CME daily. Filter CME to one tenor before sorting and shifting. Output: `mfe_mae_summary.json` (the chapter's barrier-width references).

### Recipe 7: Single-factor triage (`07_defining_the_learning_task/05_signal_evaluation`; `ml4t.diagnostic.signal.analyze_signal`)
Placeholder factor: 21-day ETF momentum. Order of screens: correctness at decision time -> association -> shape -> feasibility.
1. Correctness screens (four): coverage (fraction of (date, asset) with non-null factor, overall and per date); staleness (change rate vs definition-implied update frequency; price momentum should change every session); timing/lag consistency; mask alignment (critical for fundamental/third-party data, Ch8-10). Exact thresholds not in notes.
2. Association: IC_t = Spearman(signal_t, return_{t->t+h}) across assets. `cross_sectional_ic` = per-date Spearman then mean (exposes IC IR, t, p); `pooled_ic` = one global Spearman over all (date, asset) rows (conflates good days with within-day ranking skill). Ch14 standardizes on cross-sectional. SE(mean IC) ~ sigma_IC / sqrt(T); ICIR = mean IC / sigma_IC; t ~ ICIR x sqrt(T) if serially uncorrelated, HAC t otherwise. `ic_band(ic)` maps |mean IC| to literature bands (declared table; starting points only).
3. Shape: spread = top minus bottom quantile mean return; monotonicity (library) = Spearman rho between quantile rank and mean return in [-1, 1]; the fold section uses the fraction of upward adjacent steps in [0, 1] (different scales, named apart). `monotonicity_band(value)` scores magnitude; read sign first.
4. Horizon: `periods=(1, 5, 21)`; 21-day forward returns on consecutive days share 20 days of overlap. IC decay on a finer grid; draw the half-peak marker only for a crossing after the peak, none if still rising.
5. Feasibility: cost-adjusted IC (Grinold) IC_net ~ IC - c x turnover / E[r], c = round-trip cost; break-even = compare the top-bottom spread to a conservative round-trip cost at the rebalance frequency; capacity = recompute IC by liquidity bucket (Ch8).
6. Fold-aware IC: `create_walk_forward_splits(dates, n_splits=5, min_train_pct=0.2, test_periods=63)` expanding window; IC on test dates only per fold; report the distribution of fold stats plus the fold-date span and coverage share. Decompose: fold-date IC vs full-sample IC isolates the period effect; fold mean vs fold-date IC isolates the aggregation effect.
7. Within-time permutation: permute asset labels on the fold dates only (a null over ~5000 dates is ~sqrt(10) ~ 3x too narrow vs ~500); block permutation = one relabeling per 21-session block (label horizon) held fixed across the block; compare with block = 1 (independent within-date shuffle) on identical dates.
8. Binary labels: ROC AUC, PR AUC (rare positives), confusion matrix at a threshold; hand-built ROC/PR sweeps (the sklearn helper fails under the project pin); `ml4t.diagnostic.evaluation.binary_metrics.binary_classification_report()` gives Wilson CIs + tests; compute AUC per fold as with IC.
Writes a factor scorecard (global + fold stats) and a compact NumPy artifact for the IC time-series figure.

### Recipe 8: IC inference (`07_defining_the_learning_task/06_ic_inference`)
1. Dependence: `compute_acf(series, nlags=20)`; library `ml4t.diagnostic.evaluation.autocorrelation.analyze_autocorrelation` (ACF, PACF, Ljung-Box joint test up to lag L).
2. Newey-West HAC: sigma^2_HAC = gamma_0 + 2 sum_{j=1..L} w_j gamma_j, w_j = 1 - j/(L+1) (Bartlett). Automatic L = floor(4 (T/100)^(2/9)) (= 9 on the chapter series); with `label_horizon=h` the library uses L = max(h-1, auto). h-1 is a conservative horizon-aware bandwidth justified by the measured ACF (outside the white-noise band until ~lag 16), not a theorem. `compute_ic_hac_stats(ic_series, ic_col="ic", maxlags=None, label_horizon=None, kernel="bartlett", use_correction=True, allow_naive_fallback=False) -> ICHACStats` (keys `t_stat`, `p_value`); omitting `label_horizon` raises a `UserWarning` (anti-conservative bandwidth).
3. Effective sample size: T_eff = T x (sigma_naive / sigma_HAC)^2.
4. Block bootstrap: `block_bootstrap_ic_ci(ic_series, n_bootstrap=1000, block_length=21, alpha=0.05, random_state=42)` on `arch.bootstrap.StationaryBootstrap(21, ic_series)`; `bs.conf_int(np.mean, 1000, method="percentile")`. Moving-block = fixed length; stationary = geometric random lengths. The HAC CI and the block-bootstrap CI should overlay. `ml4t.diagnostic.evaluation.stats.stationary_bootstrap_ic(signals_t, returns_t, n_samples=1000)` (defined in `evaluation/stats/bootstrap.py`; `hac_standard_errors.py` only re-imports it) is a cross-sectional bootstrap for ONE date's IC; the time-series question needs the block bootstrap on the IC series.
5. Practical vs statistical significance: minimum detectable IC = from the HAC SE at the stated confidence (no fixed convention).
6. Track-record planning: T_required = (z_{alpha/2} sigma_IC / IC_target)^2 x HAC_inflation^2; `min_track_record_ic(target_ic, ic_std, confidence=0.95, hac_inflation=1.5)`. Writes an IC inference report.

### Recipe 9: Search accounting and multiple testing (`07_defining_the_learning_task/07_multiple_testing`; `ml4t.diagnostic.evaluation.stats`)
Pipeline order: HAC p-values -> BH (or Holm) -> report adjusted p-values with discoveries -> RAS if candidates are parameter variants -> log a JSON discovery report (keys `n_factors_tested`, `sample_periods`, `alpha`, `methods.{naive: threshold "t > 2.0", harvey: threshold "t > 3.0", benjamini_hochberg: {alpha, discoveries, true_positives, ...}, holm_bonferroni: {alpha, discoveries, true_positives}}`, Rademacher analysis, `search_set` sizes). SEED=42 via `set_global_seeds`.

| Tool | Question | Call | Reading |
|---|---|---|---|
| Expected max under null | how big is the best of N noise ICs | `expected_max_null = sqrt(2 * ln(n_factors)) * ic_std` | compare the selected IC to this, not to zero |
| BH-FDR | screening many factors | `benjamini_hochberg_fdr(p_values, alpha=0.05, return_details=True)`; manual: sort p, largest k with p(k) <= (k/n) alpha, reject 1..k | E[FP / discoveries] <= alpha |
| Holm-Bonferroni | no false positive tolerable | `holm_bonferroni(p_values, alpha=0.05)` -> dict with `"rejected"` boolean array | P(any FP) <= alpha; step-down |
| Harvey-Liu-Zhu | new factor claim | `HARVEY_THRESHOLDS = (("traditional", 2.0, "the level a single test would use"), ("modern", 3.0, "accounts for the factors already searched"), ("strict", 3.5, "for a paper claiming a new factor"))`, declared in code and printed so the applying cells cannot drift from prose; `SINGLE_TEST_ALPHA = 0.05` | t > 3.0 vs single-test t > 2.0 (3.5 for a paper claiming a new factor) |
| Rademacher Anti-Serum | correlated variants | `R_hat = rademacher_complexity(ic_matrix, n_simulations=5000, random_state=42)` on a (T periods x N factors) per-period IC matrix (library default `n_simulations=10000`); `ras_ic_adjustment(observed_ic, complexity, n_samples=T, delta=0.05, kappa=1.0, return_result=True)` -> `RASResult` (adjusted_values, observed_values, complexity, data_snooping_penalty, estimation_error, n_significant, significant_mask, massart_bound, complexity_ratio) | lower bound theta_N >= theta_hat_N - 2 R_hat - 2 kappa sqrt(log(2/delta)/T); significant iff adjusted IC > 0; Massart = sqrt(2 ln N / T); `complexity_ratio` = R_hat_norm / Massart (near 1 independent, well below 1 correlated) |
| RAS for Sharpe | best strategy among correlated ones | `ras_sharpe_adjustment(observed_sharpe, complexity, n_samples, n_strategies, delta=0.05)` | - |
| Two-pass | avoid double-dipping | exploration on the first `int(n_periods * 0.8)` periods with BH alpha=0.10, promote survivors; confirmation on the remaining 20% with BH alpha=0.05 on the promoted set only; report `search_set: [n_total, n_promoted]` | promote on fold stability, not peak |
| DSR | is the best Sharpe of N real | `deflated_sharpe_ratio(returns, frequency="daily", benchmark_sharpe=0.0, confidence_level=0.95, periods_per_year=None, skewness=None, excess_kurtosis=None, autocorrelation=None, effective_trials=None, correlation_method=None, min_k_eff=1.0)`; `deflated_sharpe_ratio_from_statistics(observed_sharpe, n_samples, n_trials=1, variance_trials=0.0, ...)`; `compute_expected_max_sharpe(n_trials, variance_trials)`; `effective_number_of_trials(returns, method="effective_rank", random_state=None)` (also Marchenko-Pastur, clustering) | `DSRResult.probability` must exceed `confidence_level` (0.95); compare annualized SR_hat with annualized `expected_max_sharpe` |
| PBO | does the IS-best collapse OOS | `compute_pbo(is_performance, oos_performance)` on IS/OOS matrices -> `PBOResult` (pbo, pbo_pct, n_combinations, n_strategies, is_best_rank_oos_median, is_best_rank_oos_mean, degradation_mean, degradation_std) | no-edge selection sits at 0.5; reassurance = near 0 with IS-best near the top OOS; full CPCV (S groups, C(S, S/2) combos) in Ch16 |
| MinTRL | how long a record is needed | `compute_min_trl(returns=None, observed_sharpe=None, target_sharpe=0.0, confidence_level=0.95, frequency="daily", periods_per_year=None, skewness=None, excess_kurtosis=None, autocorrelation=None)` -> `MinTRLResult` (min_trl, min_trl_years, current_samples, has_adequate_sample, deficit, deficit_years, observed_sharpe, target_sharpe, confidence_level, the moments skewness / excess_kurtosis / autocorrelation, frequency, periods_per_year); `min_trl_fwer(observed_sharpe, n_trials, variance_trials, target_sharpe=0.0, confidence_level=0.95, frequency="daily", skewness=0.0, excess_kurtosis=0.0, autocorrelation=0.0)` with `variance_trials` from the DSR search actually run | read across a row (search width), not down a column; `never` = required record grows faster than evidence accrues |
| One call | after per-factor test results exist | `multiple_testing_summary(test_results, method="benjamini_hochberg", alpha=0.05)` | `test_results` shape not shown |
| Related | Sharpe inference | `probabilistic_sharpe_ratio(observed_sharpe, benchmark_sharpe=0.0, n_samples=1, skewness=0.0, kurtosis=3.0, return_components=False)`; `compute_sharpe_variance(sharpe, n_samples, skewness, kurtosis, autocorrelation, n_trials=1)` | - |

RAS units discipline: compute R_hat twice, on standardized ICs (each column divided by its own per-period IC std) for the Massart ratio and on raw ICs for the deduction. kappa bounds the per-period observation (Spearman support [-1, 1]) -> KAPPA = 1.0 (the library default kappa = 0.02 is NOT what the notebook uses); `kappa_empirical = max|ic_matrix|` is run only as a sensitivity calculation with no significance read off it. Print `data_snooping_penalty` (2 R_hat) and `estimation_error` (2 kappa sqrt(log(2/delta)/T)) from the result rather than asserting which dominates (it moves with kappa, N, T).

`DSRResult` fields: probability, is_significant, z_score, p_value, sharpe_ratio (per period), sharpe_ratio_annualized, benchmark_sharpe, n_samples, n_trials, n_trials_raw, n_trials_effective, correlation_method, min_k_eff, frequency, periods_per_year, skewness, excess_kurtosis (Fisher: normal = 0), autocorrelation, expected_max_sharpe (per period), deflated_sharpe (per period), variance_trials, min_trl, min_trl_years (both can be inf), has_adequate_sample, confidence_level. Decide on `is_significant` (probability > confidence_level), report `n_trials_effective` as the search size actually charged, and pass `variance_trials` on to `min_trl_fwer`.

HAC `label_horizon` feeding the corrections: `NON_OVERLAPPING` (= 1) for independently drawn synthetic periods; the forward-return window otherwise (`ETF_LABEL_HORIZON = 5` for the ETF search; 5/21/63 for the scan).

### Recipe 10: Causal sanity checks (`07_defining_the_learning_task/08_causal_sanity_checks`, ~6 min)
Outcome per feature: PROCEED / REVISE (expected default) / STOP. Data: ~92 non-bond ETFs over 14 years, VIX from FRED macro data for regimes, IEF retained as the Treasury control. Fail loudly at load if IEF/VIX are absent (an inner join with a missing control empties the panel silently).
1. Feature x horizon scan: 10 features in four families (`SCAN_FEATURES`: momentum `mom_21d`, `mom_63d`, `mom_126d`, `mom_252d`, `mom_12_1` = return t-252 to t-21, `adj_mom_126d` = mom_126d / vol_126d; reversal `rev_1d`, `rev_5d`; trend `dist_200ma` = distance from the 200-day MA; volatility `rvol_ratio` = 20d / 60d realized vol) x `SCAN_HORIZONS` 5d/21d/63d (`fwd_5d`, `fwd_21d`, `fwd_63d`) = 30 tests. Per cell `compute_cross_sectional_ic(df, feature_col, outcome_col, min_obs=20)` -> (mean IC, t, IC series), then `compute_ic_hac_stats` at the horizon, then BH across all 30 cells; stars mark BH decisions, not |t| > 2. The deep-dive joins on the feature column (not the display label) and carries the grid-wide BH decision next to the raw p (`_bh_for(feat_col)`).
2. Shifted-label check: with p = log P, rev1d_t = p_{t-1} - p_t and fwd5d_t = p_{t+5} - p_t share -p_t. Re-measure with the label p_{t+6} - p_{t+1} (same 5-day holding, no shared endpoint). If the cell depends on which five sessions are used, withhold belief. Four undistinguished explanations: shared-endpoint noise, a real t->t+1 effect dropped, the added t+5->t+6 session, sampling variation.
3. State the mechanism as a DAG (confounder / mediator / collider) before running the checks.
4. Timing placebo: `run_timing_placebo(df, feature_col, outcome_col="forward_return", lags=None)` shifts the feature backward by increasing lags and recomputes HAC IC. A lookback-L feature shifted by Delta shares ~(L - Delta)/L of its inputs (mechanical persistence floor): judge decay at lags beyond L (grid extended to 252d for 12-1 momentum, lookback ~231 trading days). Diagnostic, not a gate; IC should peak at lag 0.
5. Shared-driver check: `run_shared_driver_check(df, feature_col, baseline_ic, n_permutations=200, seed=42)`. Treasury arm: does the feature's cross-sectional mean predict IEF 21-day forward returns (rolling Spearman rank correlation over a `WINDOW = 63`-session quarterly window, then HAC)? Permutation arm: `block_permutation_null(df, feature_col, baseline_ic, seed, n_permutations)` blocked at the label horizon, p = (r+1)/(B+1), finest resolvable p = 1/(B+1). Only the Treasury arm decides the outcome; report the permutation beside it. Fast path: pre-standardize ranks per date so permuted Spearman IC = dot(z_x, z_y)/n (> 10x faster).
6. Regime heterogeneity: `run_regime_heterogeneity(df, feature_col, outcome_col="forward_return", vix_low=15, vix_high=22)`; the executed run passes the constants `VIX_LOW_THRESHOLD = 15` / `VIX_HIGH_THRESHOLD = 22` (also the signature defaults), which the notebook describes as tercile cutoffs; the reading is robust to a median split or quartiles. STOP only if an opposite-sign partition has HAC |t| > 2 AND the unconditional IC is itself not significant (a priori rule); stable sign with large magnitude variation -> CAUTION / REVISE.
7. Collider simulation: X (momentum) and Y (forward return) independent; Z = beta1 X + beta2 Y + eps (fund flows); conditioning on high Z induces a negative X-Y correlation.
8. Event-time alignment (concept only here; applied in Ch8 fundamentals and Ch9 regime transitions): IC should concentrate where the mechanism predicts. Red flags: IC peaking before the event (anticipation/leakage); symmetric pre/post IC (confound).
Writes the figure 7.10 artifact (`write_figure_7_10_artifact()`), the scan heatmap and the scorecard.

### Recipe 11: ml4t-engineer feature registry and `analyze_signal` handoff (`07_defining_the_learning_task/10_ml4t_library_ecosystem`)
- `compute_features(df, features=[names])` (registry defaults), `features=[{"name": ..., "params": {...}}]`, or `compute_features(df, config_path="features.yaml")`. One output column per feature per call; for several horizons of one indicator call once per parameter set, suffix, then join.
- `get_registry()` entries carry defaults, inputs, description and a closed-form formula for ~25% of the catalog. Bollinger takes upper and lower deviations separately (TA-Lib); RSI uses Wilder's smoothing (TA-Lib), not an EWM span, so library and EWM-span RSI differ.
- YAML config: `features: [{name: rsi, params: {period: 14}}, {name: macd, params: {fast: 12, slow: 26, signal: 9}}, {name: atr, params: {period: 14}}]`.
- Library vs manual: standard indicators / production / TA-Lib cross-validation -> ml4t-engineer (120+ registered features, validated against TA-Lib); custom alpha factors / pedagogy / non-standard variants -> manual code.

## Guardrails and pitfalls

Preprocessing
- **Sort-then-check monotonicity** — a check that cannot fail hides misordered rows. Test time order per symbol in arrival order; confirm the check fires on a deliberately shuffled frame.
- **Null counts as coverage** — whole missing rows/months never register as nulls. Reindex onto a continuous period range, count all-missing periods, compare the asset set to an independent expected universe.
- **`|r| > 1` as corporate-action detector** — returns are bounded at -1, forward splits never fire, most flagged rows are not splits. Use `split_ratio`; check bounds before any two-sided rule.
- **Deleting flagged extreme returns blindly** — they are not the events they look like. Investigate; use the threshold only as a filter for implausible upward jumps before scaler fitting, never as a detector.
- **Fitting scalers/imputers/encoders on full data** — leaks future statistics; small per feature, accumulates across dozens, inflates Sharpe/IC. `SplitAwarePreprocessor` / `ml4t.engineer.preprocessing` fit on train only; refit at each walk-forward fold.
- **Winsor bounds from the full sample** — same leakage channel. Bounds from the training period, by-period cross-sectional; remember it replaces tails with point masses and loses "how extreme".
- **Unseen categories at test time** — crash or silent misencoding. Encoder fitted on train with `handle_unknown="ignore"`.
- **Naive join of monthly characteristics to daily prices** — future characteristic updates leak. As-of join on `dt.truncate("1mo")`; inner join on exact timestamps only for same-frequency panels; `join_asof(strategy="backward")` on sorted keys, never `forward`.
- **Z-scoring fat-tailed returns** — Jarque-Bera rejects normality; a few days drag the scaler. Robust scaling + winsorization; stationarity check on returns, not levels.
- **Calendar misreads** — FX Sunday reopen bars (>= 21:00 UTC) look like weekend anomalies; crypto has none. Filter only Saturday and Sunday-daytime FX bars; verify crypto bars sit on 00/08/16 UTC.
- **Reconstructing dropped firm-month rows** — the upstream coverage rule is the authors' design. Treat the panel as given; describe gaps as "missing due to observed coverage rules".
- **Stringified-date sorting** — lexicographic order is correct only for zero-padded ISO. Sort on native timestamps.
- **Chained `with_columns`** — loses Polars parallelism. One call with `.over()` windows.

Labels
- **Anchor mismatch** (label close-to-close, execution next open) — scores the model on an untransactable price; invisible in the mean, decisive per label. Anchor where the fill happens (next open for EOD signals).
- **Close-only triple barrier in production** — touch detected only on closes and booked at the barrier price; gains understated, losses hidden (asymmetric, larger on the stop side). Pass `high_col`, `low_col`, `open_col`; to size the effect run a second labeling, do not reprice the first.
- **Reading triple-barrier labels as market description** — a -1% stop on a path that fell 28% discards the path. Barriers only when the strategy really exits there; fixed-horizon returns to forecast magnitude.
- **Class balance read as informativeness** — barrier labels balance because the stop (1%) is tighter than the TP (2%); fixed-horizon binary inherits sample drift. Compare against the majority-class baseline; check balance over time with `label_diagnostics`.
- **Cross-sectional cut-point drift** — constant class balance hides a target whose economic meaning changes. Plot the cut-point return series alongside class proportions.
- **Overlapping labels treated as independent** — N_eff ~ N/H; the loss is dominated by high-concurrency periods. Uniqueness weights (`calculate_uniqueness=True`), sequential bootstrap, N_eff per symbol; never pretend they removed the overlap.
- **Trend-scanning t as significance** — price-level regression residuals are autocorrelated; a driftless shuffled path rejects as often as SPY; Bonferroni fixes only the selection part. Use sign/horizon; calibrate any t against a demeaned-shuffle permutation null, not a t table.
- **Sizing on uncalibrated probabilities** — AUC/accuracy-optimized classifiers rank but are not calibrated. Calibrate before `compute_bet_size`.
- **MFE/MAE window including bar t** — counts pre-entry movement, stretches H to H+1 bars, inflates both sides unevenly. Shifts -1..-H; `shift(-1)` before the reverse rolling max.
- **CME multi-tenor rows in forward windows** — `shift(-k)` steps across contracts, inflating excursions exported to `mfe_mae_summary.json`. Filter to one tenor before sorting and shifting.
- **Round-number barriers** ("TP = 2x ATR, SL = 1x ATR", "2% crypto") — measured p75 stop for SPY/ES is > 3x ATR; a 1x stop is hit by ordinary movement. Calibrate from MFE/MAE quantiles per instrument and horizon; recompute on each data refresh.
- **Aggregate-ratio ATR multiple** — p75(MAE)/mean(ATR) ignores co-movement of ATR and excursion. Use p75 of per-entry MAE(t)/ATR(t).
- **Independent-axis histograms** — the narrower distribution is drawn as wide as the broader one; fake findings. Shared bins/axes; overlay on one axis.

Triage and inference
- **Pooled IC** — mixes which days were good with within-day ranking; regime-sensitive; hides regime mixing (a reversal's halves cancel). Cross-sectional IC per date; report IC, ICIR, HAC t, CI; partition by regime.
- **Headline IC without SE** — the same number can be clearly positive or indistinguishable from zero depending on sigma_IC and T. Always report CI/t and dispersion.
- **Monotonicity magnitude without sign** — an inverted ladder means shorting the wrong leg. Read sign first; check the spread sign at every horizon.
- **IC rising with horizon** — overlap autocorrelates IC and understates SE. HAC t via `analyze_signal`; be skeptical of monotone-in-horizon IC.
- **Half-life marker from the shortest horizon** — a rising profile reads as backwards decay. Mark only a crossing after the peak; none if still rising.
- **Fold scheme covering only the front of the panel** — `min_train_pct=0.2`, 5 folds x 63 days left the last decade unevaluated; flattering windows are the default outcome. Print fold-date span and coverage share; recompute pooled IC on fold dates before interpreting any gap.
- **Permutation null on the wrong dates or iid within date** — ~3x too narrow (sample size) and ~10x too narrow (dependence). Permute fold dates only; block relabeling per label horizon; compare block = 1 vs block = h widths.
- **Naive t on the IC series** — the iid assumption understates SE; an iid bootstrap is invalid. Newey-West with `label_horizon`; stationary block bootstrap with block = horizon.
- **Automatic HAC bandwidth** — L = 9 from T alone ignores h = 21 and signal persistence; the library warns. Pass `label_horizon`; verify the HAC CI overlays the block-bootstrap CI.
- **Short track records** — small IC plus autocorrelation needs large T; 1-2 years rarely enough. `min_track_record_ic`; report the minimum detectable IC from the HAC SE.
- **Significance as tradeability** — a factor can pass the HAC test and lose after costs. Break-even and cost models (Ch16-18); liquidity-bucket IC (Ch8).
- **Single raw indicator as a signal** — 21-day momentum IC ~0 on ETFs. Feature engineering and model combination (Ch8-12).

Search accounting
- **Selection bias** — the max of N null ICs ~ sqrt(2 ln N) sigma_IC looks like skill. Never report the selected IC uncorrected; apply BH/Holm/RAS or DSR; log the search-set size.
- **Naive p-values into corrections** — the dependence problem passes straight through. Use `compute_ic_hac_stats` p-values with the real `label_horizon`.
- **Wrong `label_horizon`** — omitting it infers the bandwidth from sample size and emits an anti-conservative warning (right for overlapping labels, wrong for independent ones). Pass `NON_OVERLAPPING` (1) for independent periods, the actual horizon (5, 21, 63) for daily-sampled multi-day returns.
- **Mixing RAS scales** — a standardized R_hat deducted from an IC rejects everything regardless of data. Massart ratio on standardized ICs; deduction with R_hat in IC units; print both.
- **kappa from the averaged IC** — the per-period Spearman IC spans [-1, 1]; a small kappa invalidates Hoeffding. KAPPA = 1.0 for inference; the empirical max only as sensitivity.
- **Data-dependent kappa** — no coverage guarantee. Fix the bound before seeing data.
- **Narrated term ordering** — prose outlives the rerun. Print `data_snooping_penalty` and `estimation_error` from the result.
- **Double-dipping the search set** — exploration-pass discovery rates inflate. 80/20 exploration (alpha 0.10) then confirmation on held-out data with the promoted set (alpha 0.05); promote on fold stability.
- **Treating 100 variants as 100 trials** — over-penalizes. Rademacher ratio; `effective_number_of_trials` for Sharpe. Conversely, never drop a Rademacher-correlated set to independence; report `complexity_ratio` in the discovery report.
- **DSR scale mismatch** — `sharpe_ratio`, `expected_max_sharpe`, `deflated_sharpe` are per period while `sharpe_ratio_annualized` is annualized; a sqrt(252) factor sits between adjacent rows. Annualize all rows before tabulating.
- **PBO misread** — a no-edge selection sits AT 0.5, and sampling error on 50 combos is wide. Want PBO near 0 and the IS-best near the top OOS; use full CPCV (Ch16).
- **Sharpe track record too short** — low Sharpes need > 10 years unsearched and `never` after a handful of candidates. `compute_min_trl` / `min_trl_fwer` with the search's `variance_trials` before trusting.

Mechanism checks
- **Reading |t| > 2 off a grid** — 30 cells yield 1-2 false positives by chance; the raw reading doubled the discoveries. BH across the whole grid; star corrected cells.
- **Post-selection raw p** — picking feature + horizon by looking at the scan must be paid for. Carry the grid-wide BH decision alongside the raw p in the deep-dive.
- **Shared endpoint between feature and label** — noise in p_t enters both with the same sign: spurious covariance by construction. Shift the label one session; treat failing cells as re-measure, not disproof.
- **Inner join with a missing control series** — empties the panel silently; every downstream statistic is computed on nothing. Fail loudly at load if IEF/VIX are absent.
- **Label join on display names** — scan and deep-dive spell features differently; silent empty join. Join on the feature column.
- **Gating shared-driver on the permutation arm** — flags "confound" when the real problem is no signal. Decide on the Treasury arm only; report the permutation beside it.
- **Permutation p printed as 0** — the observed assignment is itself one arrangement. p = (r+1)/(B+1); print the resolution 1/(B+1).
- **Rolling-window IC series in the Treasury arm** — consecutive quarterly windows share all but one day; the effective sample is tiny. HAC; read borderline t with care.
- **Mechanical placebo persistence** — shifted rolling features share (L - Delta)/L inputs. Diagnostic, not a gate; weight lags beyond the lookback (252d for 12-1).
- **Low-power regime sign flips** — small wrong-sign ICs in subsamples are noise. STOP only if opposite-sign |t| > 2 AND the unconditional IC is not significant.
- **Collider conditioning** — conditioning on a common effect (fund flows) manufactures association. Draw the DAG first; never condition on variables caused by both feature and label.
- **Bivariate checks miss multivariate confounders** — the most dangerous confounders are other features. Ch15 partial IC, DML, sensitivity analysis.

## Decision rules and defaults

| Area | Rule / default |
|---|---|
| Resources | 01 needs >= 16 GB RAM, 02 >= 24 GB; 08 takes ~6 min (200-permutation null) |
| Diagnostics | extreme flag \|z\| > 5.0; `max_return_threshold` 2.0; spike ratios high 1.5 / low 0.5; heatmap `max_symbols` 50 ordered by first observation |
| Cleaning | duplicates `keep_last`; gaps `flag_only`, `max_gap_days` 5; spike threshold 0.5 (flag); winsorize (0.01, 0.99) by period; penny-stock cutoff $1 |
| Which datasets need cleaning | US equities: active deep clean; FX: weekend filter (Saturday + Sunday-daytime bars only); ETFs, crypto, CME, firm characteristics: usable directly |
| Preprocessing sequence | diagnose -> dedupe -> domain filters -> extreme-return filter -> spike detection -> winsorize (train bounds) -> scale/impute/encode (train fit) -> refit per fold -> serialize |
| Label by strategy type | factor timing (monthly) -> fixed horizon; stat arb (intraday) -> fixed-horizon binary; cross-sectional ranking / equity-ETF rotation -> cross-sectional percentile; active trading with stops -> triple barrier (ATR); trend following -> trend scanning, sign only |
| Label checklist | anchor matches execution (next open for EOD signals: `open.shift(-(h+1)) / open.shift(-1) - 1` per symbol, Recipe 4; a close-to-close label is a named proxy); triple barrier only when stops are part of the strategy; give the barrier engine high/low/open in production; ATR barriers when vol changes |
| Triple-barrier placeholders | TP 0.02, SL 0.01, 20 bars; ATR variant (`LabelingConfig.atr_barrier`) 2.0x / 1.0x, period 14 (notebook 03's string-column run used 1.0x / 0.5x); always calibrate from MFE/MAE at matched percentiles (p50/p75/p90) per instrument and horizon |
| Barrier calibration | fixed-percentage barriers only where the holding period is short and externally fixed (crypto funding cycle, 8 h); daily strategies need ATR scaling; stop multiple = p75 of per-entry MAE/ATR; recompute on each data refresh |
| Trend scanning | windows 5..20, step 1, t threshold 2.0 (library default); never read the t as significance |
| Horizon source | `resolve_label_horizon(case_study_id, label, setup)`: `definition.horizon` -> `setup.labels.horizons[label]` -> CV buffer |
| `analyze_signal` defaults | `periods=(1, 5, 21)`, `quantiles=5`, `ic_method="spearman"`, `filter_zscore=3.0`, `min_assets=10`, `autocorrelation_lags=10` |
| Fold evaluation | 5 expanding-window folds, `min_train_pct=0.2`, `test_periods=63`; report fold distribution, fold-date span, coverage share; recompute pooled IC on fold dates before comparing |
| Permutation null | same dates as the statistic; block length = label horizon (21); compare with block 1 |
| IC inference | HAC with `label_horizon=h` (L = max(h-1, auto)); block bootstrap n = 1000-2000, block 21, alpha 0.05; both CIs should overlay; report T_eff; minimum detectable IC from the HAC SE; plan T_required with `hac_inflation` ~1.5 |
| Feasibility go/no-go | spread sign correct AND spread > conservative round-trip cost at the rebalance frequency; top-k turnover acceptable (monthly full-book turnover already erodes most equity alpha); IC by liquidity bucket (Ch8) |
| Binary threshold | high precision / low recall for fewer, higher-win-rate trades; high recall for opportunity capture; pick by costs and capacity; AUC per fold |
| Multiple-testing alphas | BH, Holm, RAS delta = 0.05; exploration alpha 0.10 on the first 80% of periods, confirmation alpha 0.05 on the last 20% with the promoted set only |
| Harvey threshold | t > 3.0 for a new factor (vs single-test t > 2.0 / alpha 0.05) |
| Correction choice | screening many factors (IC) -> BH-FDR or Holm; correlated parameter variants -> RAS; selecting the best strategy (Sharpe) -> DSR; validating backtests -> PBO + DSR; planning data collection -> MinTRL; FWER (Holm) when one false positive is unacceptable (real capital) |
| RAS | KAPPA = 1.0, delta = 0.05, `n_simulations` = 5000 (lib default 10000), `random_state` = 42; significant iff adjusted IC > 0; `complexity_ratio` near 1 = independent, well below 1 = correlated |
| HAC `label_horizon` | 1 (`NON_OVERLAPPING`) for independent periods; forward-return window length otherwise (5 for the ETF search; 5/21/63 for the scan) |
| Multiple-testing pipeline | HAC p-values -> BH or Holm -> adjusted p-values with discoveries -> RAS if parameter variants -> JSON discovery report |
| DSR / PBO / MinTRL reading | DSR probability > 0.95 to be significant, all rows annualized; PBO compared to 0.5, reassuring = near 0 with IS-best near the top OOS; MinTRL read across a row (search width), `never` = unconfirmable |
| Scan reading | significance = BH-corrected decision across the whole grid, not \|t\| > 2 |
| Shift check | if a cell depends on which five sessions are used, withhold belief and re-measure |
| Timing placebo | diagnostic only; IC should peak at lag 0; judge decay at lags beyond lookback L (grid to 252d for 12-1) |
| Shared-driver | outcome from the Treasury (IEF) arm only; permutation p = (r+1)/(B+1), B = 200 |
| Regime | VIX terciles, run as the constants `VIX_LOW_THRESHOLD = 15` / `VIX_HIGH_THRESHOLD = 22` (also the function defaults); STOP iff opposite-sign partition HAC \|t\| > 2 AND unconditional IC not significant; stable sign with large magnitude variation -> CAUTION / REVISE |
| Triage outcomes | PROCEED / REVISE (expected default) / STOP; 12-1 momentum = REVISE, 1-day reversal = STOP |
| Event-time | IC before the event or symmetric around it = red flag |
| Library vs manual | standard indicators / production / TA-Lib cross-validation -> ml4t-engineer; custom alpha factors / pedagogy / non-standard variants -> manual |
| Multi-horizon features | one `compute_features` call per parameter set, suffix, join |

## Code patterns and APIs

Imports and helpers (companion repo):
- Loaders (`data` package exports): `from data import load_etfs, load_us_equities, load_crypto_perps, load_crypto_premium, load_cme_futures, load_fx_pairs, load_firm_characteristics` (01; CME returns tenors 0/1/2 per session); `load_macro` (08, VIX from FRED).
- Notebook-local, not importable from `data` or `utils`: `US_EQUITIES_START_DATE = "1970-01-01"`, `ensure_symbol_alias(df)`, `filter_from_start(df, time_col, start_value)` (defined in both 01 and 02); `load_dataset_safely(loader_func, *args, **kwargs)` and `quiet_ml4t_logging()` (01 only); `ETF_START_DATE = "2010-01-01"` (07). Copy them into your own script rather than importing.
- Utilities: `from utils.paths import get_chapter_dir`; `from utils.reproducibility import set_global_seeds`; `from utils.style import COLORS, show_plotly_with_alt` (activates the ml4t Plotly template; 02 imports `show_with_alt`); `utils/artifact_specs.py` (`resolve_label_horizon`, `load_label_spec`, `resolve_label_buffer`, `resolve_market_semantics`); `case_studies/<id>/config/setup.yaml` (`labels.horizons`); `case_studies/<id>/02_labels`.
- ml4t-engineer: `ml4t.engineer.preprocessing.StandardScaler`; `ml4t.engineer.config.labeling.LabelingConfig` (classmethods `fixed_horizon`, `triple_barrier`, `atr_barrier`, `trend_scanning`; no `atr_triple_barrier`); `ml4t.engineer.labeling.triple_barrier_labels`, `ml4t.engineer.labeling.atr_triple_barrier_labels`; `ml4t.engineer.labeling.fixed_time_horizon_labels(data, horizon, method, price_col, group_col, timestamp_col)` -> `label_return_{h}p` (shift `-1` per symbol for the next-open anchor); `ml4t.engineer.labeling.meta_labels.compute_bet_size`; `ml4t.engineer.features.volatility.atr`; `from ml4t.engineer import compute_features`; `from ml4t.engineer.core.registry import get_registry`.
- ml4t-diagnostic: `ml4t.diagnostic.signal.analyze_signal`; `ml4t.diagnostic.metrics.cross_sectional_ic`, `pooled_ic`, `compute_ic_hac_stats`; `from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr, compute_min_trl, compute_pbo, deflated_sharpe_ratio, holm_bonferroni, min_trl_fwer, multiple_testing_summary, rademacher_complexity` (plus `ras_ic_adjustment`, inferred from result fields); `ml4t.diagnostic.evaluation.stats.stationary_bootstrap_ic` (source `evaluation/stats/bootstrap.py`); `ml4t.diagnostic.evaluation.autocorrelation.analyze_autocorrelation`; `ml4t.diagnostic.evaluation.binary_metrics.binary_classification_report`; `ml4t.diagnostic.evaluation.excursion.analyze_excursions`; `ml4t.diagnostic.evaluation.distribution.tests.jarque_bera_test`; `ml4t.diagnostic.evaluation.stationarity.analyze_stationarity`.
- External: `arch.bootstrap.StationaryBootstrap`; Polars; Plotly; Papermill.
- Library sources: `libs/src/ml4t_engineer/ml4t/engineer/labeling/triple_barrier.py`, `.../labeling/atr_barriers.py`, `.../labeling/meta_labels.py`, `.../config/labeling.py`; `libs/src/ml4t_diagnostic/ml4t/diagnostic/metrics/ic_inference.py`, `.../signal/core.py`; `libs/src/ml4t_diagnostic/ml4t/diagnostic/evaluation/stats/{false_discovery_rate, rademacher_adjustment, deflated_sharpe_ratio, backtest_overfitting, minimum_track_record, effective_trials, sharpe_inference, hac_standard_errors, bootstrap, reality_check, moments}.py`.

`analyze_signal` signature: `analyze_signal(factor, prices, *, periods, quantiles, filter_zscore, quantile_method, ic_method, compute_turnover_flag=True, autocorrelation_lags, min_assets, factor_col="factor", date_col="date", asset_col="asset", price_col="price") -> SignalResult` with `.ic`, `.ic_ir`, `.quantile_returns`, `.spread`, `.monotonicity`, `.turnover`, `.summary()`; aligns panels via ASOF joins. Library column defaults are date/asset/price while notebook panels use timestamp/symbol/close (inference: the notebook renames before the call).

```python
# IC inference on an overlapping-label IC series
hac = compute_ic_hac_stats(ic_series, label_horizon=21)   # hac["t_stat"], hac["p_value"]
bs = StationaryBootstrap(21, ic_series)
ci = bs.conf_int(np.mean, 1000, method="percentile")

# large-H MFE without the entry bar (per symbol: wrap in .over("symbol"))
mfe = pl.col("high").shift(-1).reverse().rolling_max(H).reverse() / pl.col("close") - 1

# BH / Holm on HAC p-values
hac = compute_ic_hac_stats(ic_series, label_horizon=5)      # one p-value per factor
bh = benjamini_hochberg_fdr(p_values, alpha=0.05, return_details=True)
holm = holm_bonferroni(p_values, alpha=0.05); n_sig = holm["rejected"].sum()

# Rademacher Anti-Serum, both scales
R_norm = rademacher_complexity(ic_matrix / ic_matrix.std(0), n_simulations=5000, random_state=42)
ratio = R_norm / np.sqrt(2 * np.log(N) / T)                 # Massart comparison
R_hat = rademacher_complexity(ic_matrix, n_simulations=5000, random_state=42)
ras = ras_ic_adjustment(mean_ic, R_hat, n_samples=T, delta=0.05, kappa=1.0, return_result=True)
significant = ras.adjusted_values > 0

# Polars point-in-time patterns
df = df.with_columns(pl.col(x).rolling_mean(n).over("symbol"), ...)        # one call
df = df.join_asof(other, on="date", by="symbol", strategy="backward")      # both sorted
df = pl.scan_parquet(path).filter(...).collect()
```

Notebook constants (07): SEED=42, N_FACTORS=100, N_PERIODS=252, N_ASSETS=50, N_FIGURE_SIMS=200, N_RAD_SIMS=5000, N_FACTORS_ZOO=300, N_TRUE_ZOO=15, N_PERIODS_ZOO=1260, N_ASSETS_ZOO=100, ETF_START_DATE="2010-01-01", ETF_LABEL_HORIZON=5, NON_OVERLAPPING=1, N_RAD_ETF=5000, N_STRATEGIES_DSR=50, N_DAYS_DSR=756, N_STRAT_PBO=20, N_COMBOS_PBO=50, TREASURY_SYMS=["IEF","TLT","SHY","AGG","BND","TIP","GOVT","BNDX","VGSH"]. Helpers in 08: `compute_cross_sectional_ic`, `block_permutation_null`, `run_timing_placebo`, `run_shared_driver_check`, `run_regime_heterogeneity`, `_contiguous_groups`, `_figure_7_10_*`, `write_figure_7_10_artifact`; in 07: `_vectorized_rank_ic`, `_vectorized_ic_with_pvals`, `write_figure_7_6_artifact`, `_fdr(fp, total)`.

## Evidence from the book
Exact numeric outputs (IC values, fold spans, quantile spreads, MFE/MAE quantiles, ATR multiples, N_eff, TP/FP counts, DSR probability, PBO estimate) are printed at notebook runtime (`mfe_mae_summary.json`, factor scorecard, IC report, discovery report); only qualitative readings are reproduced here.

Diagnostics and preprocessing (01, 02)
- All seven datasets pass index integrity; market-data coverage is high; firm characteristics are column-complete by construction. US equities carry outlier flags that need domain filters and spike review.
- US equities: every row flagged by the return threshold is an upward move (smallest flagged return positive); the filter catches most reverse splits and essentially no forward splits, although forward splits outnumber reverse ~8:1; most flagged rows are not splits. Interior gaps are rare; missingness is entry/exit. Heatmap: staircase of listings, broken top edge of delistings, no full-width blank months.
- SPY daily returns: Jarque-Bera p ~ 0 (excess kurtosis); prices non-stationary, returns stationary. FX: almost every weekend timestamp is a valid Sunday reopen bar.
- Winsorization at (0.01, 0.99) leaves the bulk untouched and turns the tails into point masses at the bounds; the tall spike at exactly zero return is a raw-panel property (unchanged closes).
- Leakage demo: fit-on-full vs fit-on-train differ by a small mean shift on synthetic data; manual scaler and `ml4t-engineer` `StandardScaler` agree up to sample/population std.

Labels and barriers (03, 04)
- Anchor alignment on SPY: mean close-to-close minus next-open difference indistinguishable from zero; std of the same order as a daily move.
- Time-series percentile threshold correlates positively but well below 1 with trailing vol (drift also moves it).
- Triple barrier on SPY (TP 2%, SL 1%, 20 days, close-only): mass at the two barrier values; the booked-vs-trigger-close gap runs the wrong way on both sides and is larger on the stop side (labels flatter the trade). Both barrier variants near balanced; the ATR variant almost never reaches the time barrier; fixed-horizon binary skews "up".
- Sequential bootstrap vs plain: nearly identical uniqueness histograms, small positive mean shift.
- Trend scanning on a demeaned-shuffled SPY path rejects almost every bar at a rate matching real SPY with similar median |t|; Bonferroni shifts the rejection rate by a few percentage points; the longest window is selected most often.
- MFE/MAE: SPY median MFE > median MAE but p90 MAE > p90 MFE (adverse tail heavier); measured p75 stop > 3x ATR (vs the conventional 1x); BTC over one 8-hour cycle is symmetric (medians and p75 agree to two decimals) and an order of magnitude smaller than daily instruments over a month (horizon effect); ES adverse excursions are wider than SPY's at both quantiles; the high-volatility tercile carries ~2x the excursion of the low tercile (larger than any SPY-ES gap); the MFE/MAE scatter shows no diagonal, so no barrier rectangle separates good from bad trades; favorable percentiles cluster, adverse spread out. Library (close-based, signed) magnitudes run narrower than manual high/low.

Triage and inference (05, 06)
- 21-day ETF momentum (`periods=(1,5,21)`, quintiles): full-sample cross-sectional IC within a few thousandths of zero, HAC t well under 1, ICIR ~0; quintile spread negative and monotonicity negative at every horizon (near-perfect inverted ladder at short horizons) -> a reversal signal; horizon-IC profile still rising at 21 days, no half-life marker.
- Fold-level IC (5 folds covering only the front ~500 dates) positive and outside every block permutation; decomposition shows the full-sample-vs-fold gap is entirely a period effect, the aggregation term rounds to zero. The block-1 null is several times narrower than the 21-block null on identical dates; both p-values on the floor.
- IC series ACF declines smoothly, reaching the white-noise band near the label horizon (~lag 16 outside the band); automatic NW bandwidth 9 vs horizon-aware 20; the HAC CI with L = 20 overlays the 21-block stationary-bootstrap CI; both contain zero.

Multiple testing (07)
- Noise panel (100 factors, 252 periods, 50 assets): naive p < 0.05 expects 5 false discoveries; BH-FDR and Holm at 0.05 expect 0; the sorted p-curve sits above the BH line everywhere. All 100 RAS lower bounds stay below zero although many raw ICs are positive; here the estimation term (kappa = 1, T = 252) dominates the search term and the total deduction is many times the largest IC; on the standardized scale the search penalty alone exceeded the largest IC by an order of magnitude.
- Factor zoo (300 factors, 15 true, 5 years, 100 assets): naive t > 2 yields a large false-positive block; Harvey t > 3 keeps most true positives with far fewer FPs; BH and Holm compared on TP/FP (counts printed in-notebook). Two-pass: exploration (80%, alpha 0.10) promotes a subset; confirmation (20%, alpha 0.05) on the reduced set.
- ETF search (13 features, `FEAT_COLS`: `mom_5d/10d/20d/40d/60d/120d`; `rev_1d/3d/5d` = negated lagged return and its 3/5-day means; `rvol_10d/20d` = annualized realized vol; `vratio_5_20`, `vratio_20_60` = rolling-mean volume ratios; 5-day horizon; from 2010; Treasuries excluded): 0 of 13 survive BH-FDR at 0.05 (and Holm); the two largest ICs are realized-volatility features and are not small in absolute terms; the Rademacher ratio is well below 1 because of six correlated momentum lookbacks.
- DSR (50 noise strategies, 756 days): the best annualized Sharpe looks respectable; the expected max under the null is barely below it; the residual deflated excess is small; the DSR probability is well short of 0.95.
- PBO (20 strategies, 50 combos, strategy 0 with an IS-only advantage): PBO ~0.5; the median OOS rank of the IS-best is dead centre.
- MinTRL table: the lowest-Sharpe row needs > 10 years daily with no search and `never` after a handful of candidates; the middle row is confirmable in ~2 years unsearched, longer than a career after a handful, `never` after 100; only the highest-Sharpe row stays inside a working career across all search widths.

Causal sanity checks (08) and library tour (10)
- Scan (10 features x 3 horizons, ~92 ETFs, 14 years): raw |t| > 2 credits twice as many cells as BH; BH survivors are close to the null's expected count; all survivors sit at the 5-day horizon (252d momentum, 12-1 momentum, 1-day reversal). The largest ICs in the grid are 21-day momentum cells but they do not clear BH (t in the low twos). At 63d, distance from the 200-day MA has the largest t. Short momentum, risk-adjusted momentum, 5-day reversal and the volatility ratio never reach t = 2. Weaker than Jegadeesh-Titman 1993 / Asness et al. 2013 would suggest.
- Shift check: moving the 5-day label by one session roughly halves the 1-day reversal t (significant -> not).
- 12-1 momentum at 21d: largest IC and smaller raw p, but does not clear grid-wide BH; naive t > HAC t. Timing: IC peaks at lag 0, decays gradually, residual at lag 252d. Treasury IC not significant. Regime: positive in all three VIX terciles, ~5x magnitude variation, only the calm regime significant (momentum crash, Daniel and Moskowitz 2016) -> CAUTION -> overall REVISE.
- 1-day reversal at 21d: IC indistinguishable from zero at all lags and from the block-permutation null; sign flip across VIX regimes (negative low-VIX, positive high-VIX, both |t| > 2) cancelling to ~0 unconditionally -> STOP.
- Collider simulation: conditioning on high fund flows induces a negative momentum-return correlation from independent X, Y.
- Library tour: the 21-day momentum factor has near-zero IC on the ETF universe with t below conventional significance; library RSI (Wilder) differs from EWM-span RSI.

## Related references
- `chapters/06_strategy_definition.md` — §6.3 leakage-aware splitting / walk-forward CV; trial accounting that this chapter's search corrections consume.
- `chapters/08_financial_features.md` — feature pipelines, liquidity-bucket capacity check, fundamental-data timing/mask screens, event-time alignment with fundamentals.
- `chapters/09_model_based_features.md` — formal stationarity tests (ADF, KPSS, Phillips-Perron), regime transitions for event-time alignment.
- `chapters/10_text_feature_engineering.md` — timing and mask-alignment screens for third-party text data.
- `chapters/11_ml_pipeline.md`, `chapters/12_gradient_boosting.md` — multivariate feature evaluation and regime-conditional modeling after univariate triage.
- `chapters/14_latent_factors.md` — standardizes on cross-sectional IC.
- `chapters/15_causal_estimation.md` — formal identification, DML, partial IC, sensitivity analysis beyond the bivariate checks here.
- `chapters/16_strategy_simulation.md` — CPCV for PBO, execution assumptions, open-to-open labels.
- `chapters/17_portfolio_construction.md`, `chapters/18_transaction_costs.md` — cost models and break-even checks behind the feasibility screen.
- `chapters/20_strategy_synthesis.md` — position sizing that consumes the rich triple-barrier output.
- `chapters/02_financial_data_universe.md` — the seven datasets served by ml4t-data.
- `case_studies/etfs.md`, `case_studies/crypto_perps_funding.md`, `case_studies/nasdaq100_microstructure.md`, `case_studies/sp500_equity_option_analytics.md`, `case_studies/cme_futures.md`, `case_studies/sp500_options.md`, `case_studies/us_equities_panel.md`, `case_studies/us_firm_characteristics.md`, `case_studies/fx_pairs.md` — each case study's `02_labels` notebook applies these label methods with horizons from `config/setup.yaml`.
- `libraries/ml4t_data.md` — loaders and dataset calendar facts.
- `libraries/ml4t_engineer.md` — preprocessing, `LabelingConfig`, `triple_barrier_labels`, ATR, feature registry.
- `libraries/ml4t_diagnostic.md` — `analyze_signal`, IC metrics, HAC, bootstrap, multiple-testing, excursion, binary metrics, stationarity.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `workflow.md`, `companion_repo.md` — cross-cutting summaries that aggregate this chapter's rules.
- Further reading:
  - Lopez de Prado, Advances in Financial Machine Learning (2018), Ch3 labeling/meta-labeling, Ch4 sample weights and sequential bootstrap.
  - Sweeney, Campaign Trading (1996) — MFE/MAE.
  - Grinold and Kahn, Active Portfolio Management (2000) — cost-adjusted IC.
  - Newey and West (1986) — HAC covariance.
  - Holm (1979); Benjamini and Hochberg (1995) — FWER and FDR control.
  - Bailey and Lopez de Prado (2014) — Deflated Sharpe Ratio; Bailey et al. — Probability of Backtest Overfitting (2015 SSRN working paper as cited in the chapter README; 2017 Journal of Computational Finance as cited in notebook 07).
  - Harvey, Liu and Zhu (2016); Hou et al. (2020); McLean and Pontiff (2016); Chen and Zimmermann (2021); Jensen et al. (2022) — factor zoo and replication.
  - Jegadeesh and Titman (1993); Asness, Moskowitz and Pedersen (2013); Daniel and Moskowitz (2016) — momentum and momentum crashes.
  - Glasserman et al. (2025), "Does Overnight News Explain Overnight Returns?"; Cinelli and Hazlett (2020), "Making sense of sensitivity: extending omitted variable bias" (JRSS-B); Pearl (2019), "The seven tools of causal inference, with reflections on machine learning" (CACM) — titles per the chapter README.

## Glossary
- Index integrity — correct dtype, per-symbol monotone time, unique (date, symbol).
- Near-duplicate — same key, different values.
- Survivorship-free — delisted assets remain in the panel; the universe must be rebuilt per decision date.
- Winsorization — clipping values beyond percentile bounds onto those bounds.
- Split-aware preprocessing — parameters learned on train only, applied to validation/test, refit per fold.
- As-of join — each observation sees the most recent prior snapshot of a lower-frequency table (`join_asof`, `strategy="backward"`).
- Anchor — the price/time at which a label's return measurement starts (close vs next open).
- Fixed-horizon label — forward return (raw/log/binary) over H bars.
- Cross-sectional percentile label — top/bottom quantile by rank across assets at each date.
- Triple barrier — first of take-profit, stop or time barrier decides the label.
- Uniqueness / N_eff — average inverse concurrency of a label; sum of uniqueness weights (~N/H).
- Sequential bootstrap — resampling that favours less-overlapped labels.
- Trend scanning — pick the window (5-20) with max |t| of a price-level regression; use its sign.
- Meta-labeling — secondary model predicting whether a primary directional signal pays, used for sizing.
- MFE / MAE — maximum favorable / adverse excursion from entry over [t+1, t+H].
- ATR — Wilder-smoothed true range (alpha = 1/n, n = 14).
- IC / ICIR — per-date Spearman between signal and forward return; mean/std of the IC series.
- Pooled IC — one Spearman over all (date, asset) pairs.
- Monotonicity — Spearman rho of quantile rank vs mean return (library) or fraction of upward steps (fold section).
- Spread — top minus bottom quantile mean return.
- Break-even cost — round-trip cost at which the spread is fully consumed.
- Correctness screens — coverage, staleness, timing/lag consistency, mask alignment.
- HAC / Newey-West — autocorrelation-robust variance with Bartlett weights up to lag L.
- Bandwidth L — HAC lag truncation; horizon-aware L = max(h-1, floor(4 (T/100)^(2/9))).
- T_eff — T x (sigma_naive / sigma_HAC)^2.
- Block / stationary bootstrap — resample contiguous blocks (fixed or geometric length) to keep dependence.
- Within-time (block) permutation — relabel assets per block of h dates on the evaluated dates to build a null.
- Selection bias — inflation of the best observed statistic because the selection step is not in the estimate.
- FDR / FWER — expected proportion of false discoveries among rejections (BH) / probability of at least one false discovery (Holm).
- Rademacher complexity (R_hat) — expected max correlation of the hypothesis class with random signs; measures effective search size.
- RAS (Rademacher Anti-Serum) — lower bound theta_hat - 2 R_hat - 2 kappa sqrt(log(2/delta)/T).
- Massart bound — sqrt(2 ln N / T), max of N standardized means; benchmark for independence.
- kappa — a priori bound on per-period observations for the Hoeffding term (1.0 for Spearman IC).
- Harvey threshold — t > 3.0 for new factor discovery.
- DSR / expected_max_sharpe / variance_trials — probability the observed Sharpe exceeds the expected max of N null trials; how good the best of N no-skill strategies would look; dispersion of trial Sharpes that drives the correction.
- PBO / CPCV — probability the IS-best ranks in the bottom half OOS; combinatorial purged CV with S groups and C(S, S/2) splits.
- MinTRL — minimum track record length for Sharpe significance; `never` when unattainable.
- NON_OVERLAPPING — `label_horizon=1` sentinel for independently drawn periods.
- Exploration / confirmation — two-pass screening (alpha 0.10 on 80%) then held-out confirmation (alpha 0.05).
- Timing placebo — shift the feature by lags and watch IC decay; diagnostic of information aging.
- Shared-driver check — test whether the feature predicts an outcome it should not (Treasury IEF).
- Block permutation null — permute labels in blocks the size of the label horizon.
- Regime heterogeneity — IC by VIX tercile; sign flip vs magnitude variation.
- Confounder / mediator / collider — common cause / intermediate variable on the causal path / common effect; conditioning on a collider opens a spurious path.
- Shared-endpoint artifact — feature ending at t and label starting at t share p_t.
- 12-1 momentum — return from t-252 to t-21, skipping the most recent month.
- Momentum crash — momentum underperformance in high-volatility regimes.
- PROCEED / REVISE / STOP — triage outcomes of the plausibility scorecard.
- Event-time alignment — IC should concentrate in the window the mechanism predicts.
- Wilder's smoothing — TA-Lib RSI/ATR smoothing, distinct from EWM span.
