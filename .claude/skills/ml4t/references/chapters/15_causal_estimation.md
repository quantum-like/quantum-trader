# Chapter 15: Causal Machine Learning

> Chapter 7 (§7.5) gave bivariate causal diagnostics (DAGs, confounder/mediator/collider roles, single-feature plausibility checks); this chapter supplies the multivariate estimation machinery: Double Machine Learning (DML) for continuous treatments, Bayesian Structural Time-Series (BSTS) for discrete events, and causal discovery (PCMCI, NOTEARS, VAR-LiNGAM, Granger) for unknown structure. Its position: a sophisticated estimator cannot rescue an identification failure, so credible work starts with treatment, outcome, estimand and adjustment set, runs a refutation battery, and ends by stating which claims survive. On a stacked panel every notion of "nearby" (standard errors, CV folds, permutation blocks, temporal halves) must be counted in decision times, not rows. A causal estimate is a statement about confounding in the training period, not a forecast: predictive power and causal effect are distinct objects, discovery output is a hypothesis, and the chapter's cross-case-study evidence (§15.7) shows confounding bias is pervasive.

## When to use this reference

- Estimating the effect of a continuous signal or factor (momentum, carry, funding-premium z-score, IV-RV spread, signed volume share) on forward returns after controlling for confounders (DML, post-double-selection LASSO).
- Deciding whether a predictive feature's association is causal, or how much confounding bias a naive OLS slope carries (naive-vs-DML bias %).
- Measuring the impact of a dated discrete event (FOMC announcement, listing, macro release) on a price series against a data-driven counterfactual (BSTS / `tfcausalimpact`).
- Discovering lead-lag or contemporaneous structure among a small panel of series without a prior DAG (PCMCI, NOTEARS, VAR-LiNGAM, Granger with BH-FDR).
- Running or reading refutation tests: placebo treatment, placebo date, negative-control outcomes, block permutation, partial-R2 sensitivity, OOS drift.
- Computing standard errors on overlapping forward labels in a stacked panel (Newey-West HAC, Driscoll-Kraay `hac-groupsum`).
- Testing regime heterogeneity of an effect (subgroup ATE vs interaction term vs CATE) or scaling positions by CATE signal-to-noise.
- Testing whether a candidate factor adds explanatory power beyond a factor zoo of PCA controls (PDS LASSO, held-out outcome).
- Reading or writing `case_studies/{cs}/12_causal_dml.py` outputs and the causal registry (`run_log/registry.db`, table `causal_runs`).
- Evaluating a causal-discovery benchmark claim (ADIA Lab challenge) or any result scored on simulated DAGs.

## Core ideas (the why)

- **Identification before estimation.** "If the estimand is not identified from the data at hand, no estimator recovers it. A flexible model fitted to a confounded comparison returns a precise answer to the wrong question." Credible work: treatment, outcome, estimand, adjustment set -> validation/refutation -> state which claims survive.
- **Estimand vs estimator.** The estimand is the target quantity; the estimator is the procedure. Two libraries disagree either because of different estimators for the same estimand (an accuracy question) or different estimands (not comparable). Binarizing a continuous treatment (CausalML) changes the estimand from the marginal effect E[Y|do(T=t+1)]-E[Y|do(T=t)] to a high-vs-low contrast E[Y|do(T=1)]-E[Y|do(T=0)].
- **Effect estimation and structure discovery do not compete.** EconML/DoWhy/CausalML start from a structure; Tigramite/causal-learn/NOTEARS start from data. Discovery precedes estimation and yields hypotheses, not truth; the downstream estimator inherits whatever discovery got wrong.
- **The outcome is part of the design, not a reporting choice.** Same treatment (extreme premium), same adjustment set, same estimator, two outcomes: forward returns (broad, indirect sentiment/risk channel, many backdoor paths) vs premium reversion (direct arbitrage-pressure channel, short path). Credibility is bounded by how much of the world sits between treatment and outcome. A mechanism-near outcome is easier to defend and easier to trivialize (part of reversion is plain mean reversion because treatment and outcome share the premium series).
- **DML is an adjusted estimate, not a free lunch.** Interpretable causally only if pre-treatment controls are sufficient, positivity holds, there is no interference, and the DAG is correct. DML reduces sensitivity to nuisance-model error (Neyman orthogonality is first-order only) but does not solve unobserved confounding, simultaneity, interference, or bad-control bias.
- **The sign of confounding bias is a result, not a rule.** On ETF momentum the controls move the slope one way; on the crypto premium the other way. Naive smaller than DML => confounders mask; naive larger => confounders inflate. Bias-vs-DML = (naive - DML)/DML measures what the controls carried, not what remains causal.
- **Decision times, not rows.** On a stacked panel every correction with a notion of "nearby" (SE, CV folds, permutation blocks, temporal subsets) must count in decision times. Rows sorted by date put different entities adjacent; 52,000 ETF-days are not 52,000 independent observations.
- **The standard error decides what an estimate can support more than the estimator does.** Three intervals (EconML iid `ate_interval`, Driscoll-Kraay, permutation null) disagree, and the spread is the price of the panel, not a defect.
- **Refutation checks fail for different reasons and are blind to each other.** Placebo-date catches effects that do not depend on timing; negative control catches adjustment that leaves residual association; OOS drift catches a fit to one period; the robustness value says how weak an unseen confounder could be and still erase the effect. Checks weaken claims; they never prove them.
- **A causal estimate is not a forecast.** It is a statement about confounding in the training period, conditional on adequate controls. A holdout backtest is not a test of it; a correctly estimated regime effect can still earn nothing.
- **Discovery != identification.** Every discovered edge (PCMCI link, NOTEARS edge, VAR-LiNGAM edge, Granger pair) is statistical evidence, not proof of causality; nothing in notebooks 07/08 is tested out of sample, so no trading interpretation is permitted. PCMCI reduces bias from *observed* common drivers; it cannot remove hidden confounding.
- **Method choice changes the object being selected.** NOTEARS estimates contemporaneous structure; VAR-LiNGAM lagged and instantaneous structural edges; Granger directed predictive pairs; PCMCI source-target-lag links. Edge counts are not interchangeable; their spread shows how conclusions depend on estimand and identifying assumptions.
- **A null discovery result does not prove market efficiency.** It is equally compatible with low power, aggregation, nonlinearity (linear ParCorr), or regime change.
- **ADIA is amortized inference under a known simulator family**, not evidence that "AI can discover causal structure". Training labels remove the identifiability bottleneck, the task (1-hop role classification) is narrower than DAG recovery up to Markov equivalence, and simulators leak causal order through variance (varsortability, Reisach et al. 2021). ADIA deliberately excludes the hard parts of finance: unmeasured confounding, measurement error, nonstationarity/regime change, feedback loops and simultaneity, selection effects, time-order constraints and mixed frequencies.
- **Case-study causal evidence is an identification diagnostic, not a trading signal.** Effects across nine panels have different units; compare dimensionless HAC t-statistics. The HAC interval and the block-permutation refutation answer different questions and are reported as two separate decisions. A refutation pass does not remove the unconfoundedness assumption (SP500 Options: treatment and outcome both depend on the IV surface).
- **BSTS controls are "plausible", not "unaffected".** Passing spillover/placebo diagnostics does not establish that the controls were unaffected by the event.
- **Missing data are part of the estimand.** Zero-filling pre-inception ETF returns manufactures returns; use a balanced long-history panel.

## Method recipes (the how)

### Library and method selection (Table 15.1, `15_causal_estimation/01_library_overview`)

| Graph known? | Treatment | Method | Library |
|---|---|---|---|
| Yes | Binary or continuous | Backdoor adjustment | DoWhy |
| Yes | Continuous | DML | EconML (`LinearDML`, `CausalForestDML`) |
| Yes | Binary, point in time (dated event) | BSTS counterfactual | `tfcausalimpact` / tfp-causalimpact |
| Yes | Binary/discrete uplift | Metalearners (S/T/X/R) | CausalML (not used in chapter) |
| No | n/a | PCMCI (lagged links) | Tigramite |
| No | n/a | VAR-LiNGAM (lagged + contemporaneous, non-Gaussian residuals) | causal-learn |
| No | n/a | NOTEARS (contemporaneous, cross-section) | hand-implemented in `08_neural_causal_discovery` |

- The bottom half splits contemporaneous vs lagged structure, not cross-section vs time series. Granger is the predictive baseline, never a causal claim.
- Common pattern: build the graph in DoWhy, estimate with an EconML estimator (DoWhy can call EconML directly).
- EconML slots: `W` = controls for residualization (confounders); `X` = effect modifiers (heterogeneity). Plain confounders placed in `X` still yield one ATE (nothing visibly breaks) but it is the wrong habit. `cv=3` random splits are valid only for i.i.d. rows.
- Notebook 01: 1,000 i.i.d. synthetic rows with known `TRUE_ATE`; treatment `momentum`, outcome `returns`, confounders `volatility`, `regime`. Reports EconML and DoWhy estimates with CI and a coverage-of-truth column. Read coverage before error; a single draw ranks nothing.

### Causal Design Contract and sequencing (fill before estimating)

| Field | Content |
|---|---|
| Unit | row of the panel (entity x decision time) |
| Treatment | continuous signal or dated indicator, timing closed before t |
| Outcome | forward window opening after t; prefer the mechanism-near outcome, then discount mechanical overlap |
| Controls (W) | pre-treatment confounders, every rolling window `shift(1)` |
| Effect modifiers (X) | regime / heterogeneity variables (only if CATE is wanted) |
| Identification assumption | backdoor adjustment set from the DAG; alternatives (IV, diff-in-diff, regression discontinuity, event-study counterfactual) when backdoor is not credible |
| Main failure modes | unobserved confounding, bad controls, interference, simultaneity, outcome in the span of controls |
| Estimand | e.g. marginal effect E[Y|do(T=t+1)]-E[Y|do(T=t)], subgroup ATE, CATE, cumulative abnormal log return |

Sequence: association check (AR(1) regression, reversion-rate counts) -> DAG + `identify_effect` -> estimate -> HAC / Driscoll-Kraay SE -> refute -> sensitivity -> OOS drift -> only then size positions.

### DoWhy backdoor workflow with validation battery (`15_causal_estimation/02_dowhy_causal_graph`)

Setting: BTC 8h bars, `load_crypto_premium(frequency="8h")` + `load_crypto_perps(frequency="1h")` (from `data/crypto/download.py`); one treatment, two outcomes.

Feature timing (every rolling window gets `shift(1)`):

| Variable | Window | Role |
|---|---|---|
| `premium_ma`, `premium_std` | t-168h..t (21 bars = 7 d) | z-score inputs |
| `return_24h`, `volatility_24h` | t-24h..t | adjustment set (backdoor) |
| `extreme_high_premium = premium_zscore > 2` (`EXTREME_THRESHOLD = 2.0`) | at t | treatment (directional, high-only, to avoid cancellation from pooling extremes) |
| `fwd_return_24h`, `fwd_premium_change` | t..t+24h (3 bars = `HORIZON_BARS`) | outcomes |
| `past_return_48h`, `past_premium_change_48h` | 24h windows ending 48h before t | negative-control outcomes |

Steps:
1. Pre-check association: AR(1) regression of forward premium change on z-score plus reversion-rate counts.
2. Specify the DAG string; `CausalModel(data, treatment, outcome, graph=dot_string)`; `identify_effect()` applies the backdoor criterion -> adjustment set `{return_24h, volatility_24h}`.
3. `fit_and_estimate(df, outcome_col, graph)` -> ATE via `backdoor.linear_regression` (iid SEs).
4. `estimate_backdoor_ols_hac(df, outcome_col, treatment_col="extreme_high_premium", controls=CONTROLS, maxlags=HORIZON_BARS)` -> Newey-West HAC with lags = outcome horizon (3 bars). Print the SE inflation ratio.
5. `run_refutations(model, estimand, estimate, n_sims=20)` (config `REFUTATION_SIMULATIONS = 50`): placebo-treatment refuter (permute treatment) and random-common-cause refuter (add an independent draw to the adjustment set); report as ratios to the original estimate. Reseed the numpy legacy global RNG (`SEED = 42`) before each refutation block.
6. `run_sensitivity(model, estimand, estimate, benchmark_covariates)` with `simulation_method="linear-partial-R2"` (Cinelli & Hazlett 2020): robustness value = partial R2 an unobserved confounder needs with both treatment and outcome (after controls) to drive the estimate to zero; `benchmark_common_causes` = observed controls expresses the hypothetical confounder as a multiple of existing ones. `bias_adjusted_fraction(analyzer, unadjusted_estimate)` traces the estimate fraction along r2tu = r2yu (curves start at 1, cross zero at the robustness value). The robustness value depends on the fit's algebra (t^2/df = R^2/(1-R^2)), not on the SE choice, so HAC leaves it unchanged; omit the "robustness value at a significance level" because it rests on the wrong iid SE.
7. Placebo-date test: shift the treatment forward 21 bars (7 days); the effect should go to ~0; read as a ratio to the original.
8. Negative-control outcomes: DoWhy returns exactly 0 by construction (the graph omits the edge), so measure residual association instead by OLS+HAC of the past outcome on treatment + controls; the coefficient should be negligible relative to the headline effect.
9. OOS drift: temporal train/test split; purge training rows within the last `HORIZON_BARS` of the cutoff (`PURGE_HOURS = 24`); cast the split boundary to the data's timestamp dtype. Read drift beside the HAC SE, not against a fixed cutoff.
10. End with a four-diagnostic comparison table per outcome (OOS drift, placebo date, negative control, robustness value) and state which checks pass/fail plus the confounding strength that would overturn the estimate. Robustness value near zero => not separable from unseen confounding.

### Double ML on a panel (`03_econml_dml`, `04_dml_crypto_regime`)

Three steps: (1) Y_hat = g(X) -> Y_tilde = Y - Y_hat; (2) T_hat = m(X) -> T_tilde = T - T_hat; (3) regress Y_tilde ~ theta T_tilde. theta is Neyman-orthogonal: first-order insensitive to nuisance error, NOT guaranteed to agree across nuisance choices.

| Setting | 03 (ETF panel, `CASE_STUDY_ID="etfs"`, ~52,000 ETF-days) | 04 (19 perps, `CASE_STUDY_ID="crypto_perps_funding"`) |
|---|---|---|
| Treatment -> outcome | `skip_recent_6_1` -> `fwd_ret_21d` | `premium_zscore_14d` -> `fwd_ret_8h` |
| Data | `load_modeling_dataset()` (Ch8 + Ch9 features + labels) | same loader |
| `HAC_LAGS` | `FORWARD_HORIZON = 21` (overlapping label) | `TREATMENT_WINDOW_DAYS * BARS_PER_DAY = 14*3 = 42` (treatment input window; 8h label does not overlap) |
| `BLOCK_SIZE` | 21 (= horizon) | sweep `BLOCK_SIZES = [3, 21, 42]` (1/7/14 days), headline 21 (one treatment half-life) |
| Embargo | `EMBARGO_PCT = 0.01` | `EMBARGO_PERIODS = 3` (24h) |
| `MAX_SAMPLES` | 50,000 (temporal subsample keeping whole dates) | 30,000 |

The four DML choices (03):
1. **Cross-fitting with purge + embargo over decision times.** Use `WalkForwardCV` (`ml4t.diagnostic.splitters`) with purging (drop training rows whose label window overlaps the test window) and embargo (gap after the test window). Not `KFold` / `TimeSeriesSplit`. On a panel build folds over unique decision times then expand to rows (`panel_folds(n_splits)`); `WalkForwardCV` counts `label_horizon` and embargo in the positions it receives, so fed raw rows it would purge a fraction of one bar. `CV_FOLDS = 5`.
2. **Driscoll-Kraay SE.** Aggregate the score by decision time, then apply a Newey-West kernel on that time series: statsmodels `cov_type="hac-groupsum"` with `time=`; helper `driscoll_kraay(endog, exog, times, maxlags=HAC_LAGS)`. Bandwidth rule: `HAC_LAGS = max(label overlap, treatment input window)`.
3. **Block permutation within entity, t-statistic comparison** (see next recipe).
4. **Nuisance-model sweep.** Linear vs gradient-boosting (LightGBM) learners; the spread across specs is the honest width of the finding.

- Manual DML: `manual_dml_timeseries(Y, T, X, n_folds=5, embargo=21, model_y=None, model_t=None, return_residuals=False, hac_maxlags=None, horizon=None, groups=None, thread_limit=DML_THREAD_LIMIT)` from `case_studies/utils/causal.py`. With `groups` it builds folds over decision times and reports Driscoll-Kraay; without, both are by row.
- EconML comparison: `LinearDML(model_y=..., model_t=..., cv=<folds>)` cross-fits on whatever folds it is handed; pass the decision-time folds expanded to rows; `.fit(Y, T, X=None, W=controls)`, `.ate()`, `.ate_interval()` (iid interval).
- Compare naive OLS, EconML `LinearDML`, manual DML; report the three intervals (iid, Driscoll-Kraay, permutation null) side by side.

### DML refutation battery

1. **Temporal placebo**: regress Y on the lead of T (reverse-causality check). For a persistent treatment (6-1 momentum barely moves in 21 days) a ratio ~1 is expected and refutes nothing; note persistence in the write-up and rely on block permutation.
2. **Block permutation**: `block_permute(arr, block_size, rng=None, groups=None, units=None, expected_step=None, gap_tolerance=None)`. Pass `groups` (symbol codes) and `units` (date codes) so blocks run along each entity's own ordered trading days. Each placebo re-runs the full estimator with the same fold count and embargo. Compare t-statistics, not effect sizes (permuted treatment is no longer explained by controls -> larger residual variance -> mechanically smaller raw effect). `N_PLACEBO_PERMUTATIONS = 100`; p = (#placebo |t| >= observed + 1)/(n + 1); print the floor 1/(n+1) beside it (100 perms -> 0.0099). Not an FDR. Require `PERMUTATION_MIN_SUCCESS = max(10, 0.5*N)` successful placebos. Read the count (and placebo mean/sd) before the z-score; the z-score assumes a normal centred on the placebo mean, and the null here is not centred at zero (demeaning by symbol or by date does not fix it; cause unresolved by the author).
3. **Subset stability**: split at a decision time (whole dates), compare halves.
4. Block-size rule: headline block = one treatment half-life (7 d = 21 bars at 8h); longest block = the treatment's full input window (14 d = 42 bars).

Helpers (`case_studies/utils/causal.py`): `empirical_permutation_p(placebo_effects, observed_effect)`, `classify_refutation(empirical_p, n_successful=None)`, `placebo_request_is_on_the_boundary(n_placebo)`, `_assert_placebo_permutation_possible`, `embargo_from_buffer`, `observation_step(frame, date_col="timestamp")`.

### Regime heterogeneity: subgroup ATE vs interaction vs CATE (`04_dml_crypto_regime`)

- Subgroup ATEs: `regime_effect(mask, name)` cross-fits (nuisance included) inside a regime, then places residuals back on the FULL time grid with zeros elsewhere before Driscoll-Kraay (zero rows move neither X'X nor X'y but keep calendar gaps between episodes). Print the filtered-grid SE beside the full-grid SE.
- Regime difference: never add two subgroup variances (not independent). Fit one regression on the residualized full sample: T, regime, T x regime (regime main effect included so the interaction is identified against regime intercepts). Report the interaction coefficient with its SE.
- Never report a ratio of two regime estimates that are each small relative to their SE.
- Terminology: regime-stratified fits are subgroup ATEs, not CATEs; CATE = one heterogeneity model (Causal Forest, X-learner).
- Diagnostic: print the residual share of treatment variance after controls (04: controls leave under a tenth); the permuted-treatment residual variance is an order of magnitude larger.

### Causal position sizing with CausalForestDML (`05_momentum_causal_trading`)

1. Regime: SPY 63-day vol with pre-holdout thresholds, `assign_volatility_regime(features_df, date_col, splits)`, 3 buckets low/mid/high (threshold quantiles not visible in notes).
2. Split: train = all walk-forward folds except the last validation period; test = last validation period (`mds.splits`, `train_end`/`val_start`); `build_walk_forward_cv_splits(train_dates, valid_mask)` -> EconML-compatible splits; no sklearn fallback. Thresholds, CATEs and scalings are frozen before the test period.
3. Fit `LinearDML` (ATE) and `CausalForestDML` (CATE with `X=` regime) on training only. `DML_NUISANCE_ESTIMATORS = 100`, `CF_MODEL_ESTIMATORS = 100`, `CF_FOREST_ESTIMATORS = 500`. Cache: `get_output_dir(15, "momentum_causal_trading")/dml_artifacts_{cs}_{label}_{tag}.json`, schema `"v3_xw_wfcv"`, `RETRAIN=True` default; the cache records X and W columns and refuses itself if they changed (cannot detect estimator-internal changes).
4. `compute_regime_scaling(cate_by_regime, cate_std_by_regime=None)` -> scaling from CATE signal-to-noise (exact formula inferred from description), shrunk toward neutral by `SHRINKAGE = 0.5`, clipped to `[MIN_SCALING_FLOOR = 0.5, MAX_SCALING_CAP = 1.5]`.
5. Baselines: `NAIVE_SCALING = {low 1.0, mid 1.0, high 1.0}`; `SIMPLE_HEURISTIC = {low_vol 1.2, mid_vol 1.0, high_vol 0.6}`.
6. Backtest: `backtest_momentum_strategy(df, regime_scaling, n_quantiles=5, cost_bps=0, treatment_col="skip_recent_6_1", outcome_col="fwd_ret_21d")`: quintiles within each date (`_assign_quintiles`); gross-normalized long-short weights within date (`_assign_base_weights`) so regime scaling changes gross exposure; return = sum(weight x outcome), turnover = sum |dw|, cost = turnover x bps (`_compute_portfolio_returns`). `TRANSACTION_COST_BPS = 10`; `COST_SCENARIOS = [5, 10, 15, 20]`. `compute_metrics` = Sharpe of overlapping 21-day forward returns. `cross_sectional_ic` = mean over dates of Spearman(momentum, fwd return), `MIN_IC_NAMES = 20`.
7. Verdict rule: the causal rule must beat `SIMPLE_HEURISTIC` net of 10 bps (and across the cost sweep) or the machinery bought nothing. Report IC sign stability train vs holdout.

### BSTS event study (`06_fed_announcement_bsts`)

- Model: `tfcausalimpact` (TFP port of Google R CausalImpact; Brodersen et al. 2015). Requires the `ml4t-py312` image and `/opt/bsts/bin/python` (NumPy 1 / pandas 2.2 stack): `docker compose --profile py312 run --rm py312 /opt/bsts/bin/python 15_causal_estimation/06_fed_announcement_bsts.py`.
- Target and controls are daily LOG RETURNS; cumulative effect = sum of post-period point effects = cumulative abnormal log return. A levels spec would sum log-point-days (wrong estimand).
- Windows: `PRE_PERIOD_DAYS = 60` (~3 months: learn control relations, shorter than ~6-month Fed cycles), `POST_PERIOD_DAYS = 20` (~1 month: 2-3 weeks of bond repricing). `_compute_analysis_window` by trading-day index position. Sensitivity sweep pre {45, 60, 90} x post {10, 20, 30} left to the reader (MCMC bottleneck).
- Pipeline: `run_event_study(data, target, controls, event_date, pre_days=60, post_days=20)` -> `_run_bsts(...)` -> `CausalImpact(data, pre_period, post_period)` -> `_extract_tfp_effect_stats(impact_data, post_start)` normalizes column names across versions (`TFP_EFFECT_COLUMNS`); significance = posterior credible interval excludes zero.
- Controls: `TARGET_ETF = "IEF"`, `CONTROL_ETFS = ["VEA", "EFA", "DBC"]` (two international + one commodity; categories matter more than tickers). Avoid SPY/VNQ (react to the same announcement). Data 2020-01-01..2024-06-01, `SEED = 42`, `MAX_FOMC_EVENTS = 0`, `MAX_PLACEBO_DATES = 0` (0 = no cap, inference).
- Spillover: `validate_control_spillover(data, controls, event_date, pre_days=60, post_days=20)` runs each control as target with the other controls as predictors; flags if its post-window interval excludes zero. Detects only a DIFFERENTIAL response among controls, not a common dollar/global-risk shock. More conservative: non-US sovereign-bond ETFs, or a local-level model with no ETF controls and a longer pre-window.
- Placebo: 12 non-FOMC dates; `post_window_events(date, index, post_days, event_dates)` screens post-windows against the 4 `FOMC_EVENTS` only (FOMC meets every 6-8 weeks; windows are 20 days). Report as a placebo false-positive rate with the surviving-date count as denominator; not an FDR, not a Type I estimate.

### PCMCI time-series discovery (`07_tigramite_time_series`, `08_neural_causal_discovery`)

| Parameter | 07 (SPY, IEF, GLD, VIX daily) | 08 (SPY, QQQ, IWM, TLT, GLD, EEM, XLF; 2015-01-01..2024-06-01) |
|---|---|---|
| `MAX_LAG` (tau_max) | 5 (predeclared: weekly response window) | 3 |
| `pc_alpha` | `None` (Tigramite auto-select) | 0.05 |
| FDR | `fdr_method="fdr_bh"` inside `run_pcmci` | external `multipletests(lagged_p.reshape(-1), alpha=0.05, method="fdr_bh")` across all source-target-lag p-values |
| `ALPHA_LEVEL` | 0.05 | 0.05 |
| Bootstrap | `N_BOOTSTRAP = 100`, `BLOCK_SIZE = 20`, `STABILITY_THRESHOLD = 0.5`, `LAG_CONTEXT = 2 * MAX_LAG` = 10 | `N_BOOTSTRAP = 100`, `BLOCK_SIZE = 20` |
| Other | `N_SAMPLES = 500`, `SEED = 42`, `GRANGER_MAX_LAG = 3` | `N_SAMPLES = 2000`, `GRANGER_MAX_LAG = 5` |

Algorithm:
1. PC1 (condition selection): for each target Y_t, start from all X_{t-tau}, tau = 1..tau_max; iteratively remove candidates conditionally independent of Y_t given the strongest remaining parents; `pc_alpha` sets the removal level (regularization, NOT a multiple-testing correction).
2. MCI: for every remaining pair test X_{t-tau} _||_ Y_t | parents(Y_t) \ {X_{t-tau}} U parents(X_{t-tau}); conditioning on both sides' parents removes autocorrelation-driven false positives.
3. FDR: Benjamini-Hochberg across all MCI p-values; keep links with adjusted p < `ALPHA_LEVEL`. With `fdr_method="fdr_bh"`, the returned `p_matrix` already holds BH-adjusted p-values; `val_matrix[i, j, tau]` holds the partial correlation (effect size).
4. Report each surviving link as (source, target, lag, partial correlation).
5. Block bootstrap: `block_bootstrap_blocks(values, block_size, rng, context=LAG_CONTEXT)` draws 20-row blocks with replacement and stacks them as *separate Tigramite datasets* with lag context, so the per-dataset `2 * tau_max` cut (first ten rows of every twenty-row block) stops lagged tests straddling block seams. Each resample reruns `PCMCI(... ParCorr(significance="analytic"), verbosity=0).run_pcmci(tau_max=MAX_LAG, pc_alpha=None, fdr_method="fdr_bh")`; stability = share of 100 resamples recovering the edge; keep edges with stability >= 0.5.
- `compute_acf(series, max_lag=20)` is descriptive only; the own-series ACF never sets `MAX_LAG`.
- Test: `ParCorr(significance="analytic")` is linear; nonlinear/state-dependent links are missed (say so in the interpretation).

### NOTEARS (`08_neural_causal_discovery`, own implementation)

- Objective: min_W (1/2n)||X - XW||_F^2 + lambda ||W||_1 s.t. h(W) = tr(e^{W o W}) - d = 0; h(W) = 0 iff W is a DAG. W[i, j] = edge i -> j.
- `notears_linear(X, lambda1=0.1, max_iter=100, h_tol=1e-8, rho_max=1e16, w_threshold=0.3)`; augmented Lagrangian with `_solve_notears_subproblem(w_est, d, loss, h, rho, alpha, lambda1)` (objective = loss + 0.5 rho h^2 + alpha h + lambda1 sum w; L-BFGS-B over positive/negative parts of W, bounds >= 0, zero diagonal):
  1. Initialize W = 0, rho = 1, alpha = 0, h = +inf.
  2. Solve the smooth subproblem.
  3. If h(W_new) > 0.25 h_prev, multiply rho by 10 and re-solve while rho < rho_max; else accept.
  4. alpha += rho h(W_new); stop when h <= h_tol, rho >= rho_max or max_iter reached.
  5. Threshold |W| < w_threshold to zero. (Steps 1-4 follow Zheng et al. 2018; the exact rho schedule is an inference.)
- Notebook settings: `NOTEARS_LAMBDA1 = 0.05` (function default 0.1), `NOTEARS_MAX_ITER = 100`, `BOOTSTRAP_MAX_ITER = 30` (cheaper refits), `N_BOOTSTRAP = 100`, `BLOCK_SIZE = 20`; keep raw returns so each bootstrap replicate refits its own scaler. `block_bootstrap_indices(n, block_size=20)`; `extract_edges(W, labels, threshold=0.0)`.
- Synthetic validation first (linear equal-variance SEM, seed 42): X0 = e; X1 = 0.8 X0 + e; X2 = 0.7 X0 + 0.8 X1 + e; X3 = 0.9 X2 + e; X4 = 0.8 X3 + e; `TRUE_EDGES = {(0,1),(1,2),(0,2),(2,3),(3,4)}`; `notears_linear(X_syn, lambda1=0.05, max_iter=50)`; score precision, sensitivity (recall), F1. Orientation is identifiable here only because variance accumulates along the causal order (varsortability).
- Consensus rule: an edge is robust only if present in the full-sample graph AND bootstrap frequency >= 0.5 ("CONSENSUS" vs "sample-sensitive").

### VAR-LiNGAM (`08_neural_causal_discovery`)

1. Fit a VAR for lagged relationships; 2. ICA / DirectLiNGAM on the residuals for instantaneous causal order; 3. identification via non-Gaussianity.
- Predictive screen: `ridge_var_screen(X, threshold=0.1)` = one-lag `Ridge(alpha=1.0)` VAR, coefficients thresholded at 0.1 (does not solve the ICA permutation/scaling problem).
- Library: `var_lingam_library(X, lags=1, threshold=0.1)` -> `VARLiNGAM(lags=lags, criterion=None, prune=True, random_state=SEED)` returning (B0 instantaneous, B_lag lagged). Overlap between screen and library edges = agreement between a predictive screen and a structural specification.

### Granger causality with BH-FDR (`07`, `08`)

- 07: `simple_granger_test(X, Y, max_lag=5)`, pairwise F-test at each lag order (`GRANGER_MAX_LAG = 3` in the run); BH across all pair x lag p-values at once.
- 08: 7 assets -> 42 directed pairs; one joint test per pair: `grangercausalitytests(data, maxlag=5)`, `joint_p = result[5][0]["ssr_ftest"][1]`; BH across the 42 p-values via `multipletests(..., method="fdr_bh", alpha=0.05)`.
- Never report the largest F over lag orders without correction; that *is* the lag-selection weakness, performed rather than described. Both Granger and PCMCI get BH for a fair comparison.

### Supervised role classification, ADIA surrogate (`09_adia_causal_benchmark`)

- Config: `N_DATASETS = 2000`, `N_SAMPLES = 1000`, `N_SPLITS = 5`, `N_ESTIMATORS = 500`, `SEED = 42`; `generate_random_dag(n_vars, edge_prob=0.3)` (training generator `edge_prob=0.35`; random topological permutation hides order from column index; X -> Y edge always enforced per ADIA convention); `generate_data_from_dag(W, order, n_samples=1000, noise_scale=1.0)` linear Gaussian SEM; `generate_training_dataset(n_datasets=1000, min_vars=5, max_vars=8, n_samples=1000)`.
- Role labels `classify_variable(W, x_idx, y_idx, z_idx)` -> 0 Confounder (X <- Z -> Y), 1 Collider (X -> Z <- Y), 2 Mediator (X -> Z -> Y), 3 Independent, 4 Cause of X, 5 Consequence of X (Z != Y), 6 Cause of Y (Z != X), 7 Consequence of Y.
- Features (column positions excluded): correlation stats (Pearson, Spearman, partial correlations via `compute_partial_correlation(data, i, j, conditioning_set)`), CI p-values (`ci_test_pvalue` = Fisher-z, p = 2 Phi_bar(|z|), returns 1.0 on degenerate input), regression diagnostics, distribution shape (variance, skewness, kurtosis). Pass a pandas DataFrame to LightGBM so `feature_names_in_` carries real names.
- Baseline `ci_heuristic_baseline(features)` at alpha 0.05: both marginal p > a -> Independent; both < a and p(X,Y|Z) > p_zx p_zy -> Confounder; both < a and p(Z,X|Y) > a and p(Z,Y|X) > a -> Mediator; p_zx < a and p(Z,Y|X) > a -> Cause of X; p_zy < a and p(Z,X|Y) > a -> Cause of Y; p_zx < a -> Consequence of X; p_zy < a -> Consequence of Y; else Independent. Not the PC algorithm.
- Supervised: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` with group = dataset id; `lgb.LGBMClassifier(n_estimators=500, learning_rate=0.1, num_leaves=31, deterministic=True, n_jobs=1, drop_seed=SEED, ...)` on unscaled features; score = `balanced_accuracy_score`; importance = mean +/- std of gain share across the five fold models. Import `lightgbm` before `scikit-learn`.
- Reference bar: 0.7670 balanced accuracy (top ADIA entry) vs ~0.4 for classical constraint-based baselines in the official challenge. Treat the score as within-simulator-family only.

### Cross-case-study causal registry (`10_case_study_insights`)

- Source: each case study's `run_log/registry.db` (SQLite, table `causal_runs`), read-only; immutable URI only when `registry_readonly_uri` decides the tree is unwritable; `PRAGMA integrity_check` first.
- `_load_causal_runs(case_study) -> pl.DataFrame` selects `causal_hash, label, treatment, confounders_json, n_folds, ..., confounding_bias_pct, refutation_p, created_at` excluding every hash in `SELECT supersedes_hash FROM causal_runs WHERE supersedes_hash IS NOT NULL`; rejects duplicate live labels. Authoritative three-condition rule (supersession + current identity version + requested execution tier): `case_studies/utils/registry/store.py::current_causal_identities`.
- `_enrich_causal` derives the HAC t-statistic, CI, confounder count, and `refutation_class = "Passes" if refutation_p < 0.05 else "Fails"`. `SIG_T = 1.96`. Bias% = (theta_naive - theta_DML)/|theta_DML| x 100; sign reversal reported separately; "large bias" at |Bias%| > 50; median |bias| reported.
- Cross-tab: {HAC clears (|t| > 1.96), HAC overlaps zero} x {Passes, Fails}; forest plot of HAC statistics; horizon plot via `HORIZON_DAYS` (fwd_ret_5m 5/390, 15m 15/390, 60m 60/390, 8h 1/3, 24h 1.0, 1d 1.0, 5d/risk_adj_5d/dh_5d 5.0, 10d/dh_10d 10.0, 21d/1m/1m_win/ret_to_expiry 21.0; unknown label -> RuntimeError).
- `TREATMENT_LABELS`: etfs "Momentum (skip-recent 6m/1m)"; crypto_perps_funding "Premium z-score (14d)"; cme_futures "Carry (%)"; fx_pairs "FX momentum (skip-recent)"; us_equities_panel "12-1 momentum"; us_firm_characteristics "12-2 momentum"; sp500_equity_option_analytics "IV-RV spread"; nasdaq100_microstructure "Signed volume share"; sp500_options "Variance risk premium (21d)".
- Print loaded vs existing case-study counts side by side; divide by the loaded count. Stamp provenance (registry hash, read time) on inherited `refutation_p`.

### Post-double-selection LASSO factor zoo (`11_factor_zoo_validation`)

- Config: `START_DATE 2006-01-01`, `END_DATE 2024-12-31`, `MIN_OBSERVATIONS 252`, `N_PCA_FACTORS 10`, `N_CV_SPLITS 5`, `N_ALPHAS 60`, `SIGNIFICANCE_LEVEL 0.05`, `MAX_SYMBOLS 0`, `OUTCOME_SYMBOL "SPY"`, `WARMUP 60`, `SEED 42`; `HAC_LAGS = max(1, int(T ** (1/3)))` after warm-up. Depends on `14_latent_factors/01_pca_equity_sectors` and `03_econml_dml`.
- Panel: SPY excluded from the factor universe and held out as outcome (breaks the tautology of an outcome in the column span of controls); keep only long-history symbols observed on every date (no zero-fill); assert no nulls/non-finite.
- Controls: `StandardScaler` -> `PCA(n_components=10, random_state=42)` on the post-warm-up sample (explicitly in-sample).
- Candidates: four lagged managed portfolios (information to t-1, return at t; quintile long-short via 80th/20th percentiles): momentum_20d, momentum_60d, low_vol (60-day std; long low-vol, short high), mean_reversion (long recent losers, short winners; lookback not visible, likely 20d, inference). Drop zeros before `WARMUP`. Sharpe ratios descriptive only.
- Naive: `ols_hac_test(outcome, design, hac_lags)` = OLS with intercept, `cov_type="HAC", cov_kwds={"maxlags": hac_lags}`.
- Selection: `fit_lasso_selector(target, pca_inputs)` = `GridSearchCV` over a fixed 60-alpha grid of `StandardScaler -> PCA(10) -> StandardScaler -> Lasso(max_iter=20_000, random_state=42)` with `cv=TimeSeriesSplit(n_splits=5)` (expanding; scaler and PCA refit per fold); returns full-sample support.
- `double_selection_test(outcome, candidate, controls, pca_inputs, control_names, hac_lags)`: selection 1 = PCs predicting SPY; selection 2 = PCs predicting the candidate; the union enters an intercept-inclusive OLS-HAC regression of SPY on candidate + union; Newey-West CI at two-sided 5%. A candidate "adds" only if its PDS HAC interval excludes zero. Report both selections' breadth, not just the union.

## Guardrails and pitfalls

Identification and timing
- **Outcome in the span of controls / mechanically linked to treatment** — tautological or trivial effect (reversion built from the same premium series). Use a held-out outcome (factor zoo); flag mechanism-near outcomes and discount for ordinary mean reversion.
- **Confounders computed after treatment time** — bad controls / lookahead leakage. `shift(1)` every rolling window; draw a timing diagram with confounders closing before t and outcomes opening after t; fill the Causal Design Contract.
- **Pooling extreme-high and extreme-low treatments** — effects cancel. Use a directional indicator (z > 2).
- **Bad controls** — a control affected by the treatment (premium-derived controls) blocks part of the effect. Keep W strictly pre-treatment.
- **Reading the holdout as a test of the causal estimate** — the estimate is a statement about training-period confounding, not a forecast. Report IC sign stability train vs holdout, compare against a zero-cost heuristic, sweep costs.
- **Lookahead in managed portfolios** — ranking at t with t's returns. Use information through t-1, record the return at t; drop warm-up zeros before inference.
- **Estimand mismatch** — factor-mean test vs conditional loading; time-series ETF loading vs Feng-Giglio-Xiu cross-sectional SDF loading. Hold the target fixed (same outcome, same HAC estimator); state that no SDF or causal estimand is claimed.

Inference on panels
- **iid standard errors on overlapping forward outcomes** — SE too small, false significance. Newey-West HAC with maxlags = outcome horizon (02: 3 bars) or treatment window (04: 42 bars); print the SE inflation ratio.
- **`cov_type="HAC"` on a date-sorted panel** — the kernel runs across the cross-section (21 rows = a fifth of one day). Use Driscoll-Kraay `cov_type="hac-groupsum"` with `time`; always pass decision-time arrays.
- **CV folds / `block_permute` / temporal halves counted in rows on a panel** — folds cut through a cross-section; "blocks" become a handful of entities on one day (an iid shuffle, too easy to pass). Build over unique timestamps then expand to rows; pass `groups`/`units` to `block_permute`; cut halves at a decision time.
- **Training labels realized inside the test window** — leakage across the split. Purge the last `HORIZON_BARS` before the cutoff; `WalkForwardCV` purging + embargo (`EMBARGO_PCT = 0.01` / `EMBARGO_PERIODS = 3`).
- **Nuisance-model dependence of the DML point estimate** — orthogonality is first-order only. Sweep linear vs gradient boosting; report the spread as the finding's width.
- **Serial dependence in factor-zoo inference** — iid mean tests and HC1 understate uncertainty. Intercept-inclusive OLS with Newey-West HAC, bandwidth T^(1/3).

Refutation
- **Placebo fitted with a different fold count/embargo than the estimate** — different estimator, null centred elsewhere. Each placebo re-runs the identical estimator config.
- **Comparing raw placebo effects to the observed effect** — permuted T is free of controls -> larger residual variance -> mechanically smaller placebo effects -> inflated significance (biased toward "Passes"). Compare t-statistics. Moving 04 to t-stats took its permutation p from the floor to mid-null.
- **Reporting p = 0 from finite permutations** — no finite n establishes zero. Plus-one correction; print the floor 1/(n+1); require `PERMUTATION_MIN_SUCCESS = max(10, 0.5*N)` successful placebos.
- **Treating the permutation test as calibrated** — exchangeability fails when persistent confounders predict T: on 12 zero-effect synthetic panels (AR(1) confounders, 40 placebo draws) the studentized test rejected 5/12 at 5% (raw-effect 11/12; 11/12 observed t-stats negative). Read it as "comparison against a shuffle", not a significance level; cross-check with the HAC CI.
- **z-score vs permutation count disagreement** — the placebo null is not centred at zero. Read the count and placebo mean/sd first; the z-score describes a distribution the test has not explained.
- **Temporal placebo on a persistent treatment** — the lead of T is nearly T; ratio ~1 refutes nothing. Rely on block permutation; note persistence.
- **Refutation "Fails" over-read** — only says the statistic is indistinguishable from its permutation null (short sample or unbiased estimate). Keep HAC and refutation as two separate decisions in a cross-tab.
- **Refutation pass != unconfoundedness** — the permutation tests the registered null, not the identification assumption (SP500 Options: treatment and outcome both depend on the IV surface). State the assumption explicitly per panel.
- **DoWhy negative-control estimate = 0 by construction** — the graph omits the edge, so it says nothing about the data. Measure residual association with OLS+HAC on the same adjustment set; treat as a placebo diagnostic.
- **DoWhy RNG state bleeding across refutation blocks** — the second outcome's draws depend on the first's consumption. Reseed the numpy legacy global stream (`SEED = 42`) before each block.
- **Judging coverage from one synthetic draw** — a 95% interval misses 1 in 20 even when correct; two estimators on the same sample have correlated errors. Read coverage before error; ranking needs repeated samples.

Regimes and sizing
- **Adding subgroup variances for a regime difference** — subgroups are not independent draws and lack an unbroken time grid. Interaction term in one regression on the residualized full sample; SE from the full grid with zero rows.
- **Ratio of two noisy regime estimates** — signs not pinned down, quotient uninformative. Report the interaction coefficient with SE.
- **Noisy CATE producing large positions** — overfitted heterogeneity. Shrink 0.5 toward neutral, clip [0.5, 1.5].
- **Strategy ranking that changes across cost assumptions** — the result is about costs, not strategies. Sweep `COST_SCENARIOS = [5, 10, 15, 20]` bps before believing a ranking.
- **Regime scaling cancelling out of exposure-weighted returns** — without gross normalization a market-wide scale only changes turnover costs. Gross-normalize long-short weights within each date.
- **Trusting a DML cache** — cached numbers reflect an old estimator. `RETRAIN=True` for published renders; the cache refuses itself on X/W column changes; set False only while iterating on downstream strategy code.

BSTS
- **Controls affected by the event (spillover)** — the counterfactual absorbs part of the effect. Control-as-target test; prefer international + commodity; more conservative: non-US sovereign-bond ETFs or a local-level model with no ETF controls and a longer pre-window. The spillover test misses a common shock to all controls.
- **Placebo dates contaminated by other macro releases** (CPI, payrolls, refunding, FOMC minutes) or an FOMC inside the 20-day post window — a "false positive" may be a true effect from the wrong origin. Screen post-windows against `FOMC_EVENTS`; publication grade needs a macro-calendar exclusion list.
- **Daily bars for an intraday announcement** — event-day close-to-close mixes pre/post trading. Declare it part of the estimand; estimates remain model-dependent.
- **Single BSTS window setting** — untested sensitivity. Sweep pre {45, 60, 90}, post {10, 20, 30}.
- **BSTS placebo rate from 12 dates** — one flag moves the rate by a large step. Report as a coarse placebo false-positive rate with surviving-date denominator, not a Type I estimate or FDR.
- **Levels instead of log returns** — sums log-point-days, the wrong estimand. Model daily log returns; cumulative effect = cumulative abnormal log return.

Causal discovery
- **Hidden confounding in discovery** — PCMCI conditions only on observed panel members; macro factors driving two assets create spurious links. Treat every edge as a hypothesis; require economic rationale and DML/BSTS effect estimation before use.
- **Multiple testing across links/lags** — 4-7 assets x lags x pairs yields dozens-hundreds of tests; raw p < 0.05 lists are mostly false discoveries. `fdr_method="fdr_bh"` in `run_pcmci` or `multipletests(method="fdr_bh")` over all source-target-lag p-values; BH for Granger too.
- **Confusing `pc_alpha` with FDR** — `pc_alpha` is PC1 regularization, not an MCI correction. Set `pc_alpha=None` and rely on the BH-adjusted `p_matrix`.
- **Letting the ACF set `MAX_LAG`** — own-series decay says nothing about cross-series delays; shortening the horizon post hoc is data snooping. Predeclare `MAX_LAG` (5 trading days) from economics.
- **Lag-selection sensitivity in Granger** — reporting max F over lag orders inflates significance. Compute p per lag order and BH-correct the whole set, or one joint test at a fixed max lag (lag 5, `ssr_ftest`).
- **Sample-specific edges** — single fits on noisy returns produce unstable graphs. Block bootstrap (`BLOCK_SIZE` 20, `N_BOOTSTRAP` 100); keep edges with stability >= 0.5 and (NOTEARS) also present in the full-sample fit.
- **Bootstrap breaking lag structure** — naive row resampling destroys autocorrelation; concatenated blocks let lagged tests straddle seams. Stack blocks as separate Tigramite datasets with `LAG_CONTEXT = 2 * tau_max`.
- **Non-stationarity / regime change** — constant edge weights are invalid across regimes. Split temporally; discover on train, validate on test; report sample dependence.
- **Hyperparameter sensitivity** (lambda, `w_threshold`, 0.1 cutoffs) — different lambda gives different NOTEARS graphs; ridge/LiNGAM edges depend on the 0.1 cutoff. Report sensitivity; use bootstrap consensus.
- **Linear ParCorr misses nonlinear/state-dependent links** — null results may reflect the test choice. Say so; consider nonlinear CI tests (not demonstrated).
- **Over-reading null results as efficiency** — low power, aggregation, nonlinearity, regime change are alternative explanations. Report a balanced interpretation.

Benchmarks and supervised discovery
- **Varsortability / simulator leakage** — variance grows along causal order in common SEM simulators, so methods and supervised models read order from marginal variance. Randomize topological order, exclude column positions from features, treat synthetic benchmark scores as within-family only.
- **Group leakage in supervised discovery CV** — nodes from the same DAG share a dataset; random K-fold leaks. `StratifiedGroupKFold` with group = dataset id; no cross-fold preprocessing state.
- **Accuracy on imbalanced roles** — independent nodes dominate; raw accuracy rewards majority voting. Use balanced accuracy; inspect per-role accuracy and the row-normalized confusion matrix.
- **Feature importance misread as causal importance** — gain importance shows classifier shortcuts, not graph structure. Report fold mean +/- std and label it as signature evidence.
- **Omitting the X -> Y edge in the simulator** — changes the dependence structure and invalidates comparison with ADIA. Always enforce the treatment -> outcome edge.
- **OpenMP runtime race (macOS ARM64)** — importing scikit-learn before lightgbm makes the next multithreaded LightGBM fit segfault with no traceback. Import `lightgbm` first at module top (pre-commit hook enforces).
- **Non-reproducible boosting** — thread scheduling changes histograms. `deterministic=True`, `n_jobs=1`, fixed seeds including `drop_seed`.

Registry and data hygiene
- **Synthetic fallback when real data are missing** — publishes indistinguishable fake numbers under real headings. Fail loudly; real data only (07 load failure is fatal; `ML4T_DATA_PATH`).
- **Comparing effects across panels by magnitude** — units differ (bp, %, z). Compare HAC t-statistics; native units only in tables.
- **Label count != horizon count** — two labels can share a horizon. Map labels to trading days (`HORIZON_DAYS`); fail on unknown labels rather than silently dropping.
- **Superseded registry rows** — refits write new rows with `supersedes_hash`; a naive SELECT returns duplicates. Exclude superseded hashes; fail on duplicate live labels, missing registry, or failed `PRAGMA integrity_check`; distinguish "no causal row" (stage not run) from a broken registry.
- **SQLite `immutable=1` on a live directory** — skips locking and never reads the WAL, so the integrity check certifies a pre-WAL file and the query reads a stale snapshot. Use immutable only for an unwritable bundle (`registry_readonly_uri` decides).
- **Partial coverage read as complete** — charts over N loaded case studies look like all nine. Print loaded vs existing counts; divide by the loaded count.
- **Inherited `refutation_p`** — only as good as the run that wrote it. Stamp provenance (registry hash, read time). Note: at the time of writing the corrected t-stat runner had NOT landed in this checkout's `case_studies/utils/causal.py` (registries shown were refit elsewhere).
- **Zero-filling pre-inception returns** — manufactures returns and distorts PCA/LASSO. Balanced long-history panel; executable null/finite assertions.
- **Survivorship in the ETF universe (11)** — the curated list is not point-in-time. Label as an in-sample teaching exercise; production needs a point-in-time universe. (Survivorship/PIT handling in ETF/crypto DML panels via `load_modeling_dataset` is an inference; not detailed in the notes.)
- **Leakage inside LASSO selection** — full-sample scaler/PCA or alphas tuned on the full target leak the future into early folds. Expanding `TimeSeriesSplit(5)`, refit scaler + PCA per fold, fixed 60-alpha grid.
- **Double selection degenerating to "control for everything"** — if the outcome LASSO keeps all 10 PCs the union is the full basis regardless of the candidate equation. Report both selections' breadth.

## Decision rules and defaults

Method choice
- Effect given a known structure -> EconML / DoWhy / CausalML; structure unknown -> Tigramite / causal-learn / NOTEARS first, then estimate.
- Continuous treatment -> EconML DML or DoWhy; binary/discrete uplift -> CausalML or EconML; single dated event -> BSTS.
- Lagged lead-lag structure in a small panel (<= ~10 series), linear dependence acceptable -> PCMCI with ParCorr, predeclare tau_max. Contemporaneous structure, many variables, sparse differentiable estimate -> NOTEARS, always bootstrap and threshold. Lagged + instantaneous structural edges with non-Gaussian returns -> VAR-LiNGAM, lags = 1. Quick pairwise predictive screen -> Granger (one joint test per pair at a fixed max lag + BH), never a causal claim. Many labeled graphs from a known simulator -> supervised role classification; do not expect transfer to real markets.
- Choose the outcome before the estimator; prefer the mechanism-near outcome, then discount for mechanical overlap.
- Confounders before treatment, treatment before outcome; purge the training window by the outcome horizon.
- Regime difference: one interaction model, never two subgroup fits subtracted, never a ratio.

Go/no-go gates
- Causal claim from DML (§15.7): HAC significance (|t| > `SIG_T` = 1.96) AND block-permutation refutation survival (`refutation_p` < 0.05), reported as two separate decisions; neither alone certifies identification; flag any naive-vs-DML sign reversal qualitatively regardless of size; |Bias%| > 50 = large bias.
- Discovered edge entering a strategy: bootstrap stability >= 0.5, present in the full-sample fit, FDR-significant, agreed by >= 2 independent methods (NOTEARS, VAR-LiNGAM, Granger), economically motivated, replicated out of sample (discover on train, validate on test), magnitude estimated with DML/BSTS; otherwise it stays a hypothesis.
- Causal sizing: scaling must beat `SIMPLE_HEURISTIC` {1.2, 1.0, 0.6} net of 10 bps (sweep 5-20 bps) or the machinery bought nothing.
- BSTS: an interval excluding zero is conditional on no spillover; run control-as-target and the 12-date placebo every time; screen placebo post-windows for events.
- Factor zoo: a candidate adds to SPY only if its PDS HAC interval excludes zero.
- Supervised discovery applies only in the "special setting": many labeled dataset-DAG pairs from the same DGP, role-classification scoring, no unmeasured confounding/measurement error/nonstationarity.

Defaults

| Parameter | Default | Where |
|---|---|---|
| `SEED` | 42 | all notebooks |
| `CV_FOLDS` | 5 | 03/04 DML cross-fitting |
| `N_PLACEBO_PERMUTATIONS` | 100 (floor 1/101 = 0.0099) | 03/04 |
| `PERMUTATION_MIN_SUCCESS` | max(10, 0.5 N) = 50 | 03/04 |
| `BLOCK_SIZE` | label horizon (21) or sweep [3, 21, 42]; headline = one treatment half-life, longest = full input window | 03/04 |
| `HAC_LAGS` | max(label overlap, treatment input window): 21 (03), 42 (04), 3 bars (02), T^(1/3) (11) | |
| `EMBARGO_PCT` / `EMBARGO_PERIODS` | 0.01 / 3 (24h) | 03 / 04 |
| `MAX_SAMPLES` | 50,000 (03) / 30,000 (04), whole dates | |
| `PURGE_HOURS` / `HORIZON_BARS` | 24 / 3 | 02 |
| `EXTREME_THRESHOLD` | 2.0 (z-score) | 02 |
| `REFUTATION_SIMULATIONS` | 50 (`n_sims=20` in helper call) | 02 |
| Placebo-date shift | 21 bars (7 d) | 02 |
| `SHRINKAGE`, floor/cap | 0.5, [0.5, 1.5] | 05 |
| `TRANSACTION_COST_BPS`, `COST_SCENARIOS` | 10, [5, 10, 15, 20] | 05 |
| `n_quantiles`, `MIN_IC_NAMES` | 5, 20 | 05 |
| `DML_NUISANCE_ESTIMATORS`, `CF_MODEL_ESTIMATORS`, `CF_FOREST_ESTIMATORS` | 100, 100, 500 | 05 |
| `PRE_PERIOD_DAYS`, `POST_PERIOD_DAYS` | 60, 20 trading days; sweep {45,60,90} x {10,20,30} | 06 |
| `CONTROL_ETFS` | VEA, EFA, DBC (2 international + 1 commodity) | 06 |
| Placebo dates | 12 non-FOMC dates | 06 |
| PCMCI | tau_max 5 (07) / 3 (08); `pc_alpha` None / 0.05; `fdr_bh`; `ALPHA_LEVEL` 0.05; ParCorr analytic | 07/08 |
| Block bootstrap | `N_BOOTSTRAP` 100, `BLOCK_SIZE` 20, `STABILITY_THRESHOLD` 0.5, `LAG_CONTEXT` 2 tau_max | 07/08 |
| NOTEARS | lambda1 0.05 (fn default 0.1), `w_threshold` 0.3, `h_tol` 1e-8, `rho_max` 1e16, `max_iter` 100 (30 bootstrap) | 08 |
| VAR-LiNGAM / ridge screen | lags 1, prune True, threshold 0.1; Ridge alpha 1.0, threshold 0.1 | 08 |
| Granger | 07: per-lag F up to lag 3 + BH; 08: joint `ssr_ftest` at lag 5, 42 pairs, BH 0.05 | 07/08 |
| ADIA surrogate | 2000 datasets, 5-8 vars, edge_prob 0.35, 1000 rows; 5-fold `StratifiedGroupKFold`; LightGBM 500 trees, lr 0.1, 31 leaves; balanced accuracy | 09 |
| Registry | `SIG_T` 1.96; refutation cut 0.05; large-bias cut 50% | 10 |
| Factor zoo | 10 PCs, 5 expanding splits, 60 alphas, 5% two-sided HAC, `WARMUP` 60, `MIN_OBSERVATIONS` 252, quintiles 80/20 | 11 |

Sequencing: 01 -> 02 -> 03 -> 04 -> 05; 06 separately (py312 image); 07 -> 08 -> 09 -> 10 (apply the "hypothesis" posture to case studies); 11 after `14_latent_factors/01_pca_equity_sectors` and `03_econml_dml`. `08` runs ~40 min.

## Code patterns and APIs

Run: `uv run python 15_causal_estimation/<nb>.py`; test mode `uv run pytest tests/test_chapter_notebooks.py -v -k "15_causal_estimation"`; headless `MPLBACKEND=Agg PLOTLY_RENDERER=json ...`. Data loaders: `load_modeling_dataset()` (params `CASE_STUDY_ID`, `PRIMARY_LABEL`, `MAX_SYMBOLS=0`, `SYMBOL_SUBSET=[]`, `RUN_TAG="full"`; `mds.splits` carries `train_end`/`val_start`), `load_crypto_premium(frequency="8h")`, `load_crypto_perps(frequency="1h")`, `get_output_dir(15, "momentum_causal_trading")`, env `ML4T_DATA_PATH`. Plotting: `import utils.style`; `from utils.style import COLORS, apply_ml4t_style, show_with_alt, add_message_title`.

```python
# Panel-safe DML: folds over decision times, Driscoll-Kraay SE, studentized block permutation
from ml4t.diagnostic.splitters import WalkForwardCV          # purge (label_horizon) + embargo
from case_studies.utils.causal import manual_dml_timeseries, block_permute, empirical_permutation_p
import statsmodels.api as sm

res = manual_dml_timeseries(Y, T, X, n_folds=5, embargo=21, hac_maxlags=HAC_LAGS,
                            horizon=21, groups=date_codes)    # groups -> decision-time folds + DK SE
ols = sm.OLS(endog, exog).fit(cov_type="hac-groupsum",
                              cov_kwds={"time": time_index, "maxlags": HAC_LAGS})
perm_T = block_permute(T, block_size=BLOCK_SIZE, rng=rng, groups=symbol_codes, units=date_codes)
# re-run the identical estimator on perm_T; collect t_placebo
p = (np.sum(np.abs(t_placebo) >= abs(t_obs)) + 1) / (n + 1)   # floor 1/(n+1)
```

```python
# EconML
from econml.dml import LinearDML, CausalForestDML
est = LinearDML(model_y=model_y, model_t=model_t, cv=decision_time_folds_expanded_to_rows)
est.fit(Y, T, X=None, W=controls); est.ate(); est.ate_interval()   # iid interval; report beside DK
cf = CausalForestDML(...); cf.fit(Y, T, X=regime_modifiers, W=controls)  # CATE
```

```python
# DoWhy
model = CausalModel(data, treatment, outcome, graph=dot_string)
estimand = model.identify_effect()                       # backdoor adjustment set
estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
model.refute_estimate(estimand, estimate, method_name="placebo_treatment_refuter", num_simulations=...)
model.refute_estimate(estimand, estimate, method_name="random_common_cause", num_simulations=...)
# sensitivity: simulation_method="linear-partial-R2", benchmark_common_causes=[...]
# (method-name strings reconstructed from prose and the standard DoWhy API)
```

```python
# PCMCI with BH-adjusted p-values
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
parcorr = ParCorr(significance="analytic")
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)
results = pcmci.run_pcmci(tau_max=MAX_LAG, pc_alpha=None, fdr_method="fdr_bh")
# results["p_matrix"][i, j, tau] is BH-adjusted; results["val_matrix"] = partial correlation
```

```python
# Granger with BH; VAR-LiNGAM
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests
joint_p = grangercausalitytests(data, maxlag=5)[5][0]["ssr_ftest"][1]
reject, p_adj, *_ = multipletests(p_values, alpha=0.05, method="fdr_bh")
from causallearn.search.FCMBased.lingam import VARLiNGAM
B0, B_lag = var_lingam_library(X, lags=1, threshold=0.1)   # VARLiNGAM(lags=1, criterion=None, prune=True, random_state=SEED)
```

```python
# BSTS (run under /opt/bsts/bin/python, ml4t-py312 image); log returns in, cumulative abnormal log return out
ci = CausalImpact(data, pre_period, post_period)
stats = _extract_tfp_effect_stats(ci.inferences, post_start)   # TFP_EFFECT_COLUMNS normalizes names
```

```sql
-- Live causal registry rows (10_case_study_insights)
SELECT causal_hash, label, treatment, confounders_json, n_folds, ..., confounding_bias_pct,
       refutation_p, created_at FROM causal_runs
WHERE causal_hash NOT IN (SELECT supersedes_hash FROM causal_runs WHERE supersedes_hash IS NOT NULL)
```

Other named helpers: `case_studies/utils/causal.py` (2,223 lines): `run_dml_analysis`, `format_dml_summary(results)`, `resolve_causal_request(study, request)`, `run_resolved_causal_request`, `register_causal_run` (writes the registry), `DMLResearchContext`, `DML_THREAD_LIMIT`, `classify_refutation`, `embargo_from_buffer`, `observation_step` (semantics of the request/registry functions are inferred from names). Notebook-local: `panel_folds`, `driscoll_kraay`, `regime_effect`, `assign_volatility_regime`, `build_walk_forward_cv_splits`, `compute_regime_scaling`, `backtest_momentum_strategy`, `compute_metrics`, `cross_sectional_ic`, `run_event_study`, `validate_control_spillover`, `post_window_events`, `compute_acf`, `block_bootstrap_blocks`, `simple_granger_test`, `notears_linear`, `_notears_objectives`, `_solve_notears_subproblem`, `ridge_var_screen`, `var_lingam_library`, `block_bootstrap_indices`, `extract_edges`, `create_causal_graph_viz`, `generate_random_dag`, `classify_variable`, `generate_data_from_dag`, `extract_features`, `ci_heuristic_baseline`, `_load_causal_runs`, `_enrich_causal`, `ols_hac_test`, `fit_lasso_selector`, `double_selection_test`; polars `pl.col("label").replace_strict(HORIZON_DAYS, default=None)`. Warnings silenced in 02 by category/module (DoWhy unescaped backslashes, pydot/pyparsing renames, DoWhy `robustness_value_func` dead branch); keep convergence warnings visible.

## Evidence from the book

Numeric ATEs, SEs, robustness values, p-values, Sharpe ratios, IC values and discovered edges were computed in stripped cells; only qualitative directions are visible in the notes.

- `01_library_overview` (synthetic, n = 1,000): EconML and DoWhy intervals reported with coverage of `TRUE_ATE`; precision is set by the small independent variation of `momentum` after partialling out confounders; a single draw ranks nothing.
- `02_dowhy_causal_graph` (BTC 8h, train 2019-01-01 to 2023-06-30): association "not subtle"; ATEs for the two outcomes differ by a factor of ~5 and in units; both drift substantially OOS; the negative control shows residual association on the returns outcome (backdoor path partly open); the robustness value is larger for reversion; the reversion claim is "the more defensible of two claims that both rest on a short sample of one asset". None of the four diagnostics establishes causality; the mechanism argument does the work.
- `03_econml_dml` (ETF panel, ~52,000 ETF-days): controls shift the momentum slope (direction printed in the notebook); Driscoll-Kraay inflates SE relative to iid; temporal placebo ratio ~1 (uninformative); permutation null not centred at zero; three intervals disagree; the point estimate moves with nuisance flexibility.
- Synthetic calibration (carried in `10`): 12 panels with true effect 0 and AR(1) confounders, 40 placebo draws: studentized block permutation rejects 5/12 at 5%; raw-effect comparison rejects 11/12; 11/12 observed t-stats negative.
- `04_dml_crypto_regime` (19 perps): controls explain most treatment variance (residual share < 1/10); permuted-treatment residual variance an order of magnitude larger; both regime ATEs small relative to SE; the interaction gives the difference with an SE; moving to t-stat comparison took the permutation p from the floor to mid-null.
- `05_momentum_causal_trading` (ETF holdout): IC sign flips between training and holdout; in-sample improvement does not carry to holdout; causal scaling reduces degradation when training heterogeneity persists but cannot rescue an inverted signal; where the causal rule does not beat `SIMPLE_HEURISTIC`, the machinery bought nothing.
- `06_fed_announcement_bsts` (IEF, 4 FOMC dates): the corrected log-return run is not flagged by control-as-target or placebo diagnostics; estimates remain model-dependent due to daily timing and possible weak common spillover.
- `07_tigramite_time_series` (SPY, IEF, GLD, VIX; tau_max 5): null result, no lagged edge stable above the 0.5 robustness threshold; daily returns show limited own-series persistence; linear ParCorr may miss nonlinear/state-dependent relationships.
- `08_neural_causal_discovery`: NOTEARS recovers the 5-edge synthetic chain (precision/recall/F1 reported) because variance accumulates along the causal order; on the 7-ETF panel FDR "materially thins the graphs" for both PCMCI and Granger; method outputs diverge from zero to dozens of edges; only NOTEARS edges in the full-sample fit with >= 50% bootstrap frequency are called consensus.
- `09_adia_causal_benchmark`: supervised LightGBM beats the CI heuristic on balanced accuracy across most roles (out-of-fold); weakest roles remain the observationally equivalent ones; reference 0.7670 (top ADIA entry) vs ~0.4 for classical constraint-based baselines.
- `10_case_study_insights` (nine panels): naive OLS and DML signs differ on some panels; median |bias%| reported; confounding bias is pervasive; predictive power and causal effect are distinct; the refutation column was biased toward "Passes" before the t-stat fix.
- `11_factor_zoo_validation`: naive SPY loadings on all four managed factors shrink toward zero once 10 PCs enter and their HAC intervals include zero; the outcome LASSO retains the full 10-PC basis, so the test reduces to a conservative all-PC spanning test; the four factors are spanned in-sample. Not a replication of Feng-Giglio-Xiu (2020).

## Related references

- `chapters/07_defining_the_learning_task.md` — §7.5 DAGs, confounder/mediator/collider roles and single-feature plausibility checks that this chapter generalizes.
- `chapters/08_financial_features.md` — ETF momentum (`skip_recent_6_1`) and crypto premium z-score treatments and controls.
- `chapters/09_model_based_features.md` — temporal features that enter `load_modeling_dataset()` as controls.
- `chapters/11_ml_pipeline.md` — `WalkForwardCV`, purging and embargo used for cross-fitting.
- `chapters/12_gradient_boosting.md` — LightGBM nuisance learners and the ADIA role classifier (determinism, import order).
- `chapters/14_latent_factors.md` — PCA controls for the factor zoo (`14_latent_factors/01_pca_equity_sectors`) and regime detection.
- `chapters/16_strategy_simulation.md` — evaluating the causal-sizing backtest and holdout reads.
- `chapters/17_portfolio_construction.md` — weighting factors by causal confidence (downstream of §15.8).
- `chapters/18_transaction_costs.md` — cost sweeps that decide whether causal scaling survives.
- `chapters/19_risk_management.md` — effect uncertainty in position sizing; event risk.
- `chapters/02_financial_data_universe.md`, `chapters/03_market_microstructure.md` — ETF OHLCV, universe and point-in-time issues behind the panels.
- `chapters/04_fundamental_alternative_data.md`, `chapters/05_synthetic_data.md` — macro/VIX data for discovery panels; synthetic DGPs and the simulator-leakage lesson.
- `case_studies/etfs.md` — momentum DML, CATE sizing (`12_causal_dml.py`).
- `case_studies/crypto_perps_funding.md` — premium z-score DML, regime interaction, DoWhy two-outcome study.
- `case_studies/cme_futures.md`, `case_studies/fx_pairs.md`, `case_studies/us_equities_panel.md`, `case_studies/us_firm_characteristics.md`, `case_studies/sp500_equity_option_analytics.md`, `case_studies/nasdaq100_microstructure.md`, `case_studies/sp500_options.md` — the other registered treatments in the §15.7 cross-tab.
- `libraries/ml4t_diagnostic.md` — `WalkForwardCV`, HAC inference and refutation utilities.
- `libraries/ml4t_backtest.md` — simulating the regime-scaled long-short strategy with costs.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `workflow.md`, `companion_repo.md` — cross-cutting index entries for this chapter.
- Further reading: Chernozhukov et al. 2018 (DML); Belloni et al. 2012/2014 (post-double-selection); Feng, Giglio & Xiu 2020 (factor zoo); Brodersen et al. 2015 (BSTS/CausalImpact); Cinelli & Hazlett 2020 (partial-R2 sensitivity); Imbens & Angrist 1994 (LATE); Lopez de Prado 2018 (purging/embargo), 2022 (causal factor investing); Bailey & Lopez de Prado 2014 (deflated Sharpe); Runge et al. 2019 (PCMCI); Zheng et al. 2018 (NOTEARS); Hyvarinen et al. 2010 (VAR-LiNGAM); Shojaie & Fox 2022 (Granger review); Reisach et al. 2021 (simulated DAG benchmarks gameable); Olivetti et al. 2026 (ADIA causal challenge); Spirtes et al. 2000 (PC).

## Glossary

- **Estimand / estimator** — the causal quantity targeted (e.g. marginal effect E[Y|do(T=t+1)]-E[Y|do(T=t)]) / the procedure producing a number for it.
- **Backdoor criterion / adjustment set** — rule choosing observed variables that block non-causal paths from treatment to outcome (here `{return_24h, volatility_24h}`).
- **W vs X (EconML)** — controls for residualization vs effect modifiers for heterogeneity.
- **ATE / subgroup ATE / CATE** — average effect; same estimand inside a stratum; covariate-conditional effect from one heterogeneity model.
- **DML** — double/debiased ML: residualize Y and T on controls, regress residuals; Neyman-orthogonal score.
- **Cross-fitting / purging / embargo** — nuisance models fitted on folds excluding the residualized rows / dropping training rows whose label window overlaps the test window / gap after the test window.
- **Driscoll-Kraay SE** — Newey-West on time-aggregated panel scores (`hac-groupsum`).
- **Block permutation / studentized permutation test** — shuffle treatment in contiguous time blocks within entity / compare t-statistics rather than effects to cancel residual-variance differences.
- **Temporal placebo / placebo-date test / negative-control outcome** — regress outcome on the lead of treatment / shift treatment by several days / pre-treatment variable the treatment cannot cause (residual association = confounding or leakage).
- **Robustness value** — partial R2 an unobserved confounder needs with treatment and outcome to zero the estimate.
- **Bad control** — a control affected by the treatment.
- **Post-double-selection (PDS)** — LASSO on outcome and on treatment; union of selected controls enters the final OLS.
- **Managed portfolio** — long-short quintile portfolio rebalanced with information through t-1.
- **BSTS / spillover / cumulative abnormal log return** — Bayesian structural time series counterfactual from controls' pre-period relation / a control that itself responds to the event / sum of post-period point effects on daily log returns.
- **Premium / funding rate** — (perp - spot)/spot; periodic payment closing the gap; high premium -> longs pay shorts.
- **Information coefficient (cross-sectional)** — mean over dates of Spearman(signal, forward return).
- **Gross-normalized weights** — long-short weights scaled so gross exposure is 1 within a date.
- **PCMCI / pc_alpha / MCI test / ParCorr** — PC1 parent selection + momentary conditional-independence tests for lagged links / PC1 regularization level (None = auto), not FDR / CI test of X_{t-tau} -> Y_t given both parents / linear partial-correlation CI test.
- **fdr_bh** — Benjamini-Hochberg false-discovery-rate adjustment.
- **Granger causality** — lagged X improves prediction of Y (pairwise F-test); predictive, not causal.
- **NOTEARS / augmented Lagrangian** — continuous DAG learning with h(W) = tr(e^{W o W}) - d = 0 and l1 penalty / penalty-plus-multiplier scheme (rho, alpha) enforcing h(W) = 0.
- **VAR-LiNGAM** — VAR for lags + DirectLiNGAM (ICA, non-Gaussianity) on residuals for instantaneous order.
- **Block bootstrap / edge stability** — resample contiguous blocks (20 rows) to preserve autocorrelation / share of resamples recovering an edge; consensus at >= 0.5.
- **Causal sufficiency / Markov equivalence class** — no unmeasured common causes (assumed by PC/PCMCI) / DAGs with identical conditional independences.
- **Varsortability / amortized inference** — degree to which marginal-variance ordering matches causal order in simulated data / learning a direct mapping from data to structure labels from many simulated examples.
- **Balanced accuracy** — mean per-class recall.
- **supersedes_hash / immutable=1** — registry pointer from a refit row to the row it replaces / SQLite URI flag safe only for unwritable bundles.
