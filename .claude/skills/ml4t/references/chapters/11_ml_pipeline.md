# Chapter 11: The ML Pipeline

> Chapter 11 turns the "linear baseline" into a full research protocol: it moves from econometric inference (unbiased parameter recovery, hypothesis tests, robust standard errors) to prediction (stable out-of-sample rankings scored by cross-sectional IC). It fits leakage-safe Ridge / LASSO / Elastic Net / logistic models on the ETF panel with per-fold winsorization and standardization, tunes them with walk-forward and nested CV, interprets them with exact linear SHAP, wraps them in conformal prediction intervals, and compares the linear family across all nine case studies. The position it argues: with many, correlated, noisy features the right tools are shrinkage plus out-of-sample evaluation, not OLS plus p-values; walk-forward folds support model comparison but are not a holdout; and a positive IC is not a tradable edge until turnover and costs have been charged. Later model chapters (12-14) must beat this baseline before added complexity is justified.

## When to use this reference

- Fitting any linear or logistic model on a panel of assets x dates (Ridge, LASSO, Elastic Net, `LogisticRegression`) for return or direction prediction.
- Deciding between regression (continuous returns), binary direction, or tercile classification for a strategy.
- Choosing a regularization strength: alpha grids, `alpha_max` for LASSO, `l1_ratio`, logistic `C`; reading an alpha sweep or a stability table.
- Designing hyperparameter optimization on time-series data: single-loop vs nested walk-forward CV, Optuna with `WalkForwardCV`, purge/embargo inside the inner loop.
- Auditing an OLS or panel regression for singular design, overlapping-label autocorrelation, cross-sectional dependence, or the wrong standard-error estimator (HC3 vs clustered vs Driscoll-Kraay vs HAC).
- Reporting IC with an honest t-stat (HAC with lag = label horizon) and reading OOS R² against zero.
- Interpreting a model with SHAP: sign checks, right-vs-wrong high-conviction splits, concentration, cross-fold and bootstrap stability.
- Calibrating classifier probabilities chronologically (Platt scaling with purged expanding splits) before confidence-based sizing.
- Building prediction intervals with split conformal, CQR, or ACI on a panel with a 21-day label and checking conditional (regime) coverage before sizing positions off interval width.
- Comparing model results stored in a case-study `run_log/registry.db` (tie resolution, retired generations, chronological HAC recompute, sign consistency).
- Checking whether a ranking signal survives turnover and transaction costs (gross vs net Sharpe) before a production backtest.
- Writing or validating model caches (signature-keyed, not existence-keyed).

## Core ideas (the why)

- **Inference and prediction answer different questions** (Breiman's two cultures; Simonian 2024). Breusch-Pagan, autocorrelation, Jarque-Bera, VIF tell you whether inference about parameters is reliable; none measures out-of-sample ranking accuracy. The trading question "does this model rank future returns accurately?" is answered by OOS evaluation, not hypothesis tests.
- **Large samples make tiny effects "significant."** Hundreds of thousands of panel rows yield many small p-values while R² stays very low. Statistical significance is not economic magnitude.
- **Robust standard errors repair exactly one failure** (non-spherical errors). They leave point estimates unchanged, cannot fix misspecification (omitted variables, wrong functional form, failed exogeneity), and do not improve predictions. "A robust SE on a misspecified model is a more careful statement about the wrong quantity."
- **A panel statistic measures what its row order says it measures.** Durbin-Watson / Breusch-Godfrey on a date-ordered panel report cross-sectional dependence under the name of serial correlation; plain Newey-West lags run across assets, not time. Regroup by asset, count lags on the session grid, and pick a covariance estimator that assumes independence only where diagnostics found none.
- **Shrinkage buys variance reduction with bias.** OLS on many correlated, weakly predictive features is exactly the condition overfitting needs. Ridge, LASSO and Elastic Net encode priors about diffuse vs sparse signal. The alpha sweep's shape (flat, rising, falling) shows where the trade stops paying; the turn is a property of the data, so re-sweep on every dataset.
- **LASSO answers "which features to keep"; Ridge answers "how much to believe all of them."** Elastic Net can zero coefficients but its L2 part keeps correlated groups together instead of picking one arbitrarily. LASSO selection is unstable under correlated inputs (features flicker on/off across folds).
- **IC (Spearman rank correlation within each date) is the metric for cross-sectional return prediction** because it is invariant to level shifts and scale. R² is not and can go negative out of sample even when the ranking is correct.
- **A sweep's own maximum is biased upward by the picking.** Nothing is held back to measure the gap. Nested CV separates "how well can the model be made to fit this fold" from "how well does the procedure generalize" (Cawley & Talbot 2010). The gap is a property of the search, not of Ridge.
- **Walk-forward validation is not a final holdout.** Folds support model comparison and diagnostics; a claim about what a strategy would have earned needs data held back from every choice, including the choice of C/alpha.
- **Direction is often more tractable than magnitude** because most trading decisions reduce to long/short/flat, but once the output is a probability, calibration, class imbalance, thresholds and turnover become first-order.
- **Discrimination and calibration are separate properties.** A model can rank correctly yet be uniformly over-confident, or be well calibrated and uninformative. The calibration curve, not the model family, decides whether confidence-based sizing is defensible. The 0.5 threshold is a convention; leaving it default "is making the cost-of-error choice by not making it."
- **Interpretability is part of validation, not cosmetics.** Attribution explains what a model did, never whether it was right: a confident wrong prediction decomposes as cleanly as a confident correct one. Global mean |SHAP| is magnitude, not evidence of skill.
- **Two kinds of instability**: cross-fold variation (a regime story) vs within-fold bootstrap uncertainty (a sample-size story). A feature can look solid on one and fail the other.
- **Recency weighting is a bet** that the recent past resembles the near future more than the distant past; its cost is effective sample size. After a structural break it is what lets a model respond at all; on one fold the sign of its effect proves nothing.
- **Conformal intervals are valid under exchangeability alone**, which walk-forward evaluation on non-stationary returns violates; under-coverage is the expected direction. Coverage validity follows from calibration-set exchangeability, not training-set size.
- **The marginal coverage number is the least interesting one.** All three conformal variants land near target marginally but over-cover in quiet terciles and under-cover badly in the high-volatility tercile. The root cause is a calibration set pooled across regimes; the fix is regime-conditional calibration, not a different variant.
- **"Online" must mean what the trading problem allows.** With a 21-day label, ACI feedback is 21 sessions stale by construction; it can only track drift slower than the label resolves, and no step size changes that.
- **IC and net Sharpe are different measurements.** Costs are charged at every rebalance while ranking is free; a signal can rank well and finish behind a rule with no signal. Regularization controls coefficient magnitude, not position change: a penalized fit is not a low-turnover fit.
- **Cross-study comparison must fail closed**: a tie on the ranking metric that differs on any measured column is a real ambiguity and raises; row order never decides which configuration a section reports.

## Method recipes (the how)

### Shared data, splits and per-fold preprocessing (all notebooks)

| Item | Value |
|---|---|
| Features | `case_studies/etfs/features/financial.parquet` (Ch8 output); momentum windows [5,10,21,42,63,126,189,252], volatility [21,63,126,252], `skip_recent` 21, drawdown [63,126], volume [21,63] |
| Labels | `case_studies/etfs/labels/fwd_ret_21d.parquet` (Ch7); `primary: fwd_ret_21d`, `buffer: 21D`; variant `fwd_ret_5d` with buffer `5D` |
| Config | `case_studies/etfs/config/setup.yaml`: `n_assets: 100`, `eligibility_rule: point_in_time_adv_10m_annual`, `eligibility_file: eligibility.csv`, decision cadence `monthly_month_end`, snapshot `close`, `execution_delay: next_bar_open` |
| Splits | `generate_cv_splits(df, case_study_id="etfs", label_buffer="21D", date_col="timestamp")` -> 8 rolling train/validation windows with purge gap (NB08 states 10y train, 1y validation, annual step) |
| Columns | `timestamp`, `symbol`, `fwd_ret_21d`; `META_COLS = {"timestamp", "symbol", target}`; `FEATURE_COLS = sorted(c for c in df.columns if c not in META_COLS)` |
| Seed | `SEED = 42` everywhere |

Steps:
1. Join features and labels on `(timestamp, symbol)`. Drop rows with any null feature; never zero-fill (zero is a real value: flat 252d return, zero vol, mid-range oscillator). Dropped rows are early warm-up history of a few symbols.
2. Inspect panel shape: unbalanced (ETFs enter over the sample; some symbols resume after gaps of a year or more); features come in pervasively correlated families.
3. Per fold, fit preprocessing on training rows only: `winsorize_train_test(X_train, X_test, lower=1.0, upper=99.0)` (clip at training 1st/99th percentiles), then `StandardScaler()`.
4. Score with `evaluate_predictions(y_true, y_pred, dates, symbols)` -> mean cross-sectional IC (`utils.modeling.cross_sectional_ic_mean`), RMSE, R². Dates below the minimum cross-section or with constant predictions are dropped.
5. No synthetic fallback: if input files are missing, run the upstream Ch7/Ch8 notebooks.

### OLS inferential toolkit (11.1, `11_ml_pipeline/01_ols_inference`)

1. Drop the 5 exact linear combinations before OLS: `mom_accel_short`, `mom_accel_medium`, `mom_accel_long`, `skip_recent_6_1`, `skip_recent_12_1` (e.g. `skip_recent_12_1 = ret_252d - ret_21d`). Keeping them makes X singular and robust covariances uncomputable.
2. Fit statsmodels OLS on the first walk-forward fold's training rows (inference is in-sample); hold the validation fold for OOS comparison.
3. Gauss-Markov battery. Only two assumptions are testable from residuals: spherical errors (Breusch-Pagan + autocorrelation) and no perfect multicollinearity (VIF). Jarque-Bera normality is not Gauss-Markov; it matters only for exact finite-sample t/F and is least consequential at this n.
4. Residual autocorrelation: regroup residuals per symbol in time order; `LAGS = (1, 5, 21)`; count lags on the session grid, not row position (`lagged_correlation(values, sessions, lag)`, `gap_aware_durbin_watson(values, sessions)` pair only rows exactly `lag` sessions apart). Skip symbols with fewer than `max(LAGS)+2` rows. Report median autocorrelation and share positive per lag. Lag 21 = label horizon: consecutive daily rows share 20 of 21 outcome days, so autocorrelation out to lag 20 is by construction.
5. VIF: > 5 concern, > 10 severe. `VIF_MAX_ROWS = 0` (0 = all rows; otherwise random subsample with `default_rng(SEED)`).
6. Robust SE comparison, reporting the ratio SE_robust / SE_OLS per feature (1 = no dependence found; 3 = OLS overstated precision threefold):

| Estimator | Independence it assumes | Fit for an overlapping-label panel? |
|---|---|---|
| OLS | all pairs uncorrelated, equal variance | no |
| HC3 | heteroscedasticity only, no cross-observation correlation | insufficient |
| Cluster by date | same-date pairs may correlate | insufficient (misses same-symbol serial correlation) |
| Two-way cluster (date and symbol) | adds same-symbol pairs | insufficient (misses cross-symbol, cross-date correlation within the 21-day window) |
| Driscoll-Kraay | arbitrary cross-sectional correlation within a window of nearby dates + Newey-West over dates; lag = label horizon (21) | use this |
| HAC / Newey-West | single time series | only after collapsing to one series per date (the IC series) |

Also listed: WLS, GLS/FGLS, Fama-MacBeth.

7. Coefficient plot: rank by |coef| with Driscoll-Kraay CI (`CONF_Z` multiplier, value not captured); bars crossing zero = sign not pinned down.
8. OOS IC: `cross_sectional_ic_series` (Spearman within each date) -> mean, IR = mean/std, t-stat via `compute_ic_hac_stats` with lag = label horizon. Show the naive t = mean/(std/sqrt(n)) beside it; it overstates by ~sqrt(overlap). Drop dates with NaN (constant predictions -> undefined Spearman) and null (too few symbols); in polars these are distinct and one NaN poisons the mean.
9. OOS R² = 1 - SS_res/SS_tot; in-sample with intercept it is >= 0 by construction; OOS it can be < 0 because coefficients amplified noise or the intercept was calibrated to the training-period mean (level error). Read against zero.

### Regularization paths (11.2, `11_ml_pipeline/02_regularization_paths`)

| Method | Penalty | Encodes | Grid / default | Prefer when |
|---|---|---|---|---|
| OLS | none | reference | - | baseline only |
| Ridge | L2 | diffuse signal; all coefficients shrink together, none exactly zero | `RIDGE_ALPHAS = np.logspace(-2, 9, 23)`; select alpha by highest mean IC across folds | correlated feature families, stable rankings wanted |
| LASSO | L1 | sparse signal; feature selection | `alpha_max = max|X0ᵀ y0| / n` on the first fold's standardized training data; `LASSO_ALPHAS = np.logspace(log10(alpha_max), log10(0.01*alpha_max), 10)`; track `n_nonzero` per alpha | you need "which features to keep" and accept selection instability |
| Elastic Net | L1 + L2 (`l1_ratio`: 1 = LASSO, 0 = Ridge) | sparse but keeps correlated groups together | hold alpha at the LASSO-selected value, sweep `l1_ratio`; joint tuning only via nested CV (NB04) | correlated groups should survive or die together |

Steps:
1. Use the 8 shared folds; per-fold winsorize + standardize (above).
2. `cross_validate(model_class, model_params)` -> `(fold metrics, fold predictions, fold coefficients)`; every fold learns preprocessing and model state from its training rows only.
3. Ridge: plot IC and ICIR vs alpha and the coefficient path. Read the sweep's shape: flat = penalty doing nothing; rising = variance reduction exceeding bias cost; falling = coefficients crushed, ranking lost. Pick inside the rising/plateau zone, not at the global maximum.
4. LASSO: fold x feature binary heatmap of surviving features; `sklearn.linear_model.lasso_path` on the first fold for the warm-started coefficient path (plot top 10 by peak magnitude).
5. Model comparison: mean IC +/- std across folds; settle ordering with paired fold-by-fold IC differences, not overlapping error bars.
6. Rank stability: apply the fold-t and fold-t+1 models to the same test set, `scipy.stats.spearmanr` of predictions; high rho = stable rankings / low implied turnover. Production: `ml4t.diagnostic.signal.compute_turnover()`.
7. Cache: `get_chapter_dir(11)/models/02_regularization_paths/cv_results.joblib` (`RETRAIN=False`) stores fitted models, scalers, `alpha_max`, and the sample-weight record.

### Sample weighting: recency and uniqueness

- Recency: w_t = exp(-lambda (T - t)) with age in **sessions**, not rows (~100 rows share each date); `HALF_LIFE_SESSIONS = 252`, `lam = ln2 / 252`; rescale weights to mean 1.
- Effective sample size (Kish): N_eff = (sum w)² / sum w² (scale-invariant, unlike sum w). Report N_eff and N_eff / N before adopting.
- Compare `Ridge(alpha=best_ridge_alpha)` with and without `sample_weight` on the last fold; `sample_weight` exists on all sklearn estimators (logistic in NB03, boosting in Ch12).
- The NB02 cache records `n_train`, `n_sessions`, `half_life`, `n_eff`, `ic_uw`, `ic_w`, `n_eff_kind="kish"`; a cache lacking any key or with a non-Kish `n_eff_kind` is recomputed (formula-version stamping as cache invalidation).
- Uniqueness (H-bar) weighting needs triple-barrier concurrency from Ch7; not applicable to ETF forward returns; applied in per-case-study `case_studies/{cs}/06_linear.py` runners where relevant.

### Loss functions (`SGDRegressor`)

- `SGDRegressor(loss="squared_error"|"huber"|"epsilon_insensitive", penalty="l2", alpha=sgd_alpha, sample_weight=...)`; `sgd_alpha = best_ridge_alpha / n_sgd` (SGD alpha scale differs from closed-form Ridge; `n_sgd` not captured).
- Huber: quadratic inside the threshold, linear beyond. Epsilon-insensitive ("MAE (ε-insensitive)"): ignores residuals < ε, linear outside.
- Read the ordering only; SGD ICs are not comparable with closed-form Ridge. Heavy-tailed returns -> prefer Huber (or epsilon-insensitive) over squared error; Ch12 uses the LightGBM Huber objective.

### Alpha landscape, stability table and 3σ diagnostic (`11_ml_pipeline/04_nested_cv_hpo` Part A)

1. `ALPHAS = np.logspace(-2, 9, ALPHA_GRID_POINTS=23)`, `LABEL_HORIZON = 21`.
2. `evaluate_alpha_grid(X, y, df_cv, alphas, n_splits, test_size, label_horizon)` uses `WalkForwardCV(n_splits=5, test_size=200, label_horizon=21, embargo_size=10)`; returns per-fold IC per alpha; heatmap with a diverging symmetric colour scale around 0. (These 5 x 200-session folds are distinct from the 8 setup.yaml folds.)
3. Stability table per alpha: `cv = std_ic / |mean_ic|`; best alpha = argmax mean_ic; plateau = alphas with mean_ic >= 0.9 x best (if best > 0; else >= 1.1 x best); median CV < 0.5 "stable", < 1.0 "moderate", else "unstable". Rows: Best alpha, Mean IC, Std IC, Plateau size, Plateau range, Median CV, Stability.
4. 3σ diagnostic over the grid's fold-mean ICs: `gap_sigma = (max - median) / std`; `> 3.0` -> "WARNING: likely overfitting". It measures across-alpha dispersion; the stability CV measures within-alpha across-fold dispersion; they can disagree, so compute both.
5. A plateau spanning orders of magnitude means the exact maximum is noise: pick inside the plateau or ensemble.

### Single-loop vs nested HPO with Optuna (`11_ml_pipeline/04_nested_cv_hpo` Part B)

| Parameter | Value |
|---|---|
| `N_SPLITS` (outer) | 5 |
| `TEST_SIZE` | 200 sessions |
| outer embargo | `embargo_size=10` |
| `N_INNER` | 3 |
| inner `WalkForwardCV` | `n_splits=n_inner, test_size=max(label_horizon, inner_sessions // (n_inner + 2)), label_horizon=21, embargo_size=21`, on the outer training window only |
| `N_TRIALS` | 20 in the notebook (function defaults 15) |
| search space | `trial.suggest_float("alpha", 0.01, 1e9, log=True)` |

- `single_loop_cv(X, y, df_cv, n_splits=5, n_trials=15, label_horizon=21, test_size=200)`: objective maximizes IC on the outer test fold itself (biased).
- `nested_cv(X, y, df_cv, n_outer=5, n_inner=3, n_trials=15, label_horizon=21, test_size=200)`: inner loop selects alpha, outer fold scores it; the test fold is never seen by the search.
- Record folds where every trial is pruned (no date with defined IC) as missing (`_selected_alpha` -> NaN); never drop them, so the fold-by-fold pairing stays aligned.
- Read the selected-alpha spread across folds before the mean ICs: swings over orders of magnitude = the search tracks fold noise; clustered selections = a stable property of the data.
- Caches: `models/04_nested_cv_hpo/grid_results{ARTIFACT_TAG}.joblib`, `cv_results{ARTIFACT_TAG}.joblib`.

### Logistic direction prediction (11.3, `11_ml_pipeline/03_logistic_classification`)

1. Target: `direction = 1 if fwd_ret_21d > 0 else 0`; check class balance (expect moderate positive skew over a long bull market). Constants: `LABEL_HORIZON_SESSIONS = 21`, `OUTER_LABEL_BUFFER = "21D"`, `MODEL_MAX_ITER = 1000`.
2. Solver by penalty (`build_logistic_model(l1_ratio, C)`):

| `l1_ratio` | Penalty | Solver | Notes |
|---|---|---|---|
| 0.0 | L2 | `lbfgs` | default; multinomial-capable |
| 1.0 | L1 | `liblinear` | binary, coordinate descent |
| else | elasticnet | `saga` | large n, supports `sample_weight` |

3. `C` is the inverse penalty strength (small C = strong shrinkage); sweep several orders of magnitude in both directions. `cross_validate_logistic(l1_ratio=0.0, C=1.0)` defaults; `fit_logistic_fold(train_idx, test_idx, l1_ratio, C)` -> `(metrics, predictions, coefficients, fitted state)`.
4. Metrics via `evaluate_classification`: accuracy, precision, recall, F1, AUC-ROC, log-loss. Fit a majority-class baseline fold by fold from training rows only; no-information references are AUC 0.5 and accuracy = majority rate (well above 0.5 here).
5. `class_weight='balanced'` vs none on the latest fold: moves recall/F1 (threshold-dependent), leaves AUC largely alone.
6. Pool OOS predictions across folds -> confusion matrix at 0.5, ROC, PR curve, `calibration_curve(..., n_bins=10)`.
7. Platt scaling: `CalibratedClassifierCV(base_model, method="sigmoid", cv=expanding_calibration_splits(dates, n_splits=3, gap_sessions=21))`; splits are expanding date blocks with starts at `linspace(0.55, 0.85)` of unique dates, complete timestamp groups kept together, last training label matures strictly before the first calibration timestamp. Compare original vs corrected curves on the latest outer fold with the same fold-local scaler; verify the corrected curve is closer to the diagonal (Platt can fit noise).
8. Hit rate by confidence: confidence = |p - 0.5|; quintile bins `np.quantile(confidence, linspace(0,1,6))`; accuracy must rise monotonically before confidence sizing.
9. Probability-to-position conversions on the last fold: threshold-based, probability-weighted, rank-based (exact rules not captured; full framework in Ch17).
10. Multinomial extension: tercile edges `np.quantile(y_ret_train, [1/3, 2/3])`, `np.digitize` (0 bottom, 1 mid, 2 top), lbfgs; compare per-class recall against the 1/3 random reference.
11. Cache: `get_output_dir(11, "03_logistic_classification")/cv_results.joblib` with `CACHE_SCHEMA_VERSION = 2` and hashed contracts (see cache recipe).

### SHAP interpretation (11.4, `11_ml_pipeline/05_shap_analysis`)

1. Model: `Ridge(alpha=1.0, random_state=42)` on the last fold (deliberately light penalty to spread attribution; not the predictive optimum).
2. `masker = shap.maskers.Independent(X_train, max_samples=len(X_train))`; `explainer = shap.LinearExplainer(model, masker)`; `explainer(X_test)` -> `Explanation` with `.values`, `.base_values`, `.data` (SHAP v0.50+).
3. Verify the identity φ_j^(i) = β_j (x_j^(i) - x̄_j); it holds exactly only with the full training set as background (the masker default of 100 rows disagrees in the second decimal).
4. Layers: beeswarm + mean |φ| bar; `EXPECTED_SIGNS` sign check (momentum +, volatility -); dependence plot on the top feature that varies on the fold; waterfalls for a correct and a wrong large positive prediction (candidates above the 95th percentile of ŷ).
5. Decision-relevant split: conviction = |ŷ|; high-magnitude = top 20% (`threshold = np.percentile(conviction, 80)`); Right = sign(ŷ) = sign(y), Wrong otherwise. Read the right/wrong count first. Compute mean |φ| per group, the wrong/right ratio (> 1 = feature leaned on hardest when wrong), the mean signed φ difference, and median-row waterfalls per group.
6. Concentration: `max_frac = max|φ| / sum|φ|` per row; flag `> 0.60`; print mean and 95th percentile of max_frac. A cut that never fires is the wrong cut: aggregate the top few features, compare across models, check per fold.
7. Cross-fold stability: Ridge on all 8 folds, mean |φ| trajectory per feature. Production: `ml4t.diagnostic.evaluation.compute_shap_importance()`.
8. Within-fold bootstrap: `N_BOOT = 200`, `TOP_K_BOOT = 10`; resample test rows with replacement, recompute mean |φ_j|, percentile interval; bootstrap the difference (leader minus each other feature) on the same resamples; claim a ranking only where the difference interval excludes zero.
9. Output `output/05_shap_analysis/shap_arrays.npz` (SHAP values, test matrix, predictions, outcomes, base value, feature names, per-fold importance matrix; source of Figure 11.3). Refit if the cached fold count < current or the test-fold row count differs from the cached `y_pred` length.

### Conformal prediction intervals (11.5, `11_ml_pipeline/06_conformal_prediction`)

Shared setup: per walk-forward fold, split the training window 80/20 into model-training / calibration; date-level `TRAIN_SUBSAMPLE = 0.25` applied equally to all three methods for parity (`papermill -p TRAIN_SUBSAMPLE 1` to compare with full history); `TARGET_COVERAGE = 0.90` (alpha_target = 0.10); `RIDGE_ALPHA = 1.0`; `RANDOM_SEED = 42`; `LABEL_HORIZON_SESSIONS = 21`.

| Method | Base / scores | Interval | Width | Use when |
|---|---|---|---|---|
| Split conformal (SC) | `fit_split_conformal(X_train, y_train, X_cal, y_cal, ridge_alpha=1.0)`; s_i = \|y_i - ŷ_i\| on calibration | half-width = the ceil((n+1)(1-alpha))-th smallest calibration score via `utils.modeling.conformal_quantile`; unbounded when rank > n; `predict_split_conformal(state, X, coverage=0.90)` | one width per fold | exchangeability plausible; cheapest (one Ridge fit) |
| CQR | two `QuantileRegressor(quantile=q, alpha=0.01, solver="highs")` at q = alpha/2 and 1 - alpha/2 (inference); s_i = max(q̂_lo(x_i) - y_i, y_i - q̂_hi(x_i)); conformal quantile added to both bounds; `fit_cqr(...)`, `predict_cqr(state, X)` | per observation | adaptive | width must adapt per observation (position sizing); LP is O(n³): `MAX_QR_SAMPLES = 20_000` |
| ACI | same base and scores as SC; `predict_aci_adaptive(state, X, y_true, dates, coverage=0.90, gamma=0.01, delay_sessions=21)` | α̂_{t+1} = α̂_t + γ(α_target - 1{y_t ∉ C_t}), clipped to [0.001, 0.999] | one width for the whole cross-section per decision date | distribution shift expected and delayed feedback available; tracks drift no faster than the label horizon |

ACI panel rules: one α̂_t per decision date; the indicator generalizes to that date's cross-sectional miscoverage rate; the update on d_i uses the miss rate of d_{i-21} only; the first 21 dates of each fold run at target alpha (warm-up). Miss -> alpha falls -> higher residual quantile -> wider next interval. Larger gamma tracks a shift sooner but makes coverage noisier.

Evaluation:
1. `run_conformal_evaluation(features, targets, cv_splits, dates, coverage=0.90)`; cache `models/06_conformal_prediction/conformal_results{ARTIFACT_TAG}.joblib` guarded by `_cache_is_usable` against `notebook_cache_signature(...)`.
2. `compute_calibration_metrics(results)` -> coverage, `coverage_gap = actual - TARGET_COVERAGE`, `mean_width`, `std_width`, `fold_std`; "closest" = argmin |coverage_gap|. Read in order: coverage_gap (negative = exchangeability violation), mean_width (what coverage cost), std_width and fold_std (constant width means coverage moves across folds instead), then the tercile table.
3. Multi-level calibration: `target_levels = np.arange(0.1, 1.0, 0.1)`; requested vs empirical coverage; above diagonal = conservative, below = under-coverage; persisted to `output/11/06_conformal_prediction/calibration.parquet`.
4. Width dynamics: fold-level mean width over time with a VIX overlay (axes not shared; compare shapes). Conditional coverage: stratify by terciles of realized |return| per method.
5. Position-size bridge: `weight = np.clip(median_width / own_width, 0, 3)` (cap at 3x median). Downstream: Ch19 uncertainty-aware sizing.

### Cross-case-study comparison (11.6, `11_ml_pipeline/07_case_study_insights`)

Data: each case study's `run_log/registry.db` populated by `case_studies/{cs}/06_linear.py` (FAMILY="linear"); tables `prediction_metrics`, fold metrics, `official_populations`; OOF prediction artifacts; per-fold pickled pipelines under the training run's `models/` directory. Per-case-study deep dives: `case_studies/{cs}/13_model_analysis.py`.

1. Eligibility (`load_complete_metrics(case_study, *, label=None)`): drop predictions with any null fold IC; require all `FINITE_METRIC_FIELDS` finite (`ic_mean_daily, ic_se_hac, ic_ci_lo, ic_ci_hi, ic_t_hac, ic_p_hac, ic_n_days, ic_hac_lag`); keep only `ic_n_days == max over label`; drop generations retired via `official_populations`; preserve `training_hash` / `prediction_hash` identities.
2. Rank by `ic_mean_daily` (`collect_complete_rank1`, `select_unique_best(frame, groups)`). Ties (`resolve_generation_tie`): identical except `TIE_IDENTITY_EXCLUDED = {training_hash, prediction_hash}` -> collapse (refit duplicate); identical except `TIE_NAME_ONLY = {config_name}` -> keep first alphabetically, carry the rest in `tied_configs` (`report_name_only_ties`); any measured difference -> raise.
3. HAC recompute: `_daily_ic_values(row, target_column)` loads chronological daily IC from the exact prediction artifact; `_bartlett_hac(values, lag, case_study)` with `lag = min(stored ic_hac_lag, n_days - 1)` returns mean, `ic_se_hac`, `ic_ci_lo/hi = mean -/+ critical x SE`, `ic_t_hac`, `ic_p_hac = 2 t.sf(|t|, n_days-1)`; `replace_with_chronological_hac(selected)` replaces selected rows only. Significance: CI excludes zero (filled markers, |t_HAC| > 2; 95% level inferred).
4. Regularization comparison: `family_from_config(expr)` -> OLS/Ridge/Lasso/ElasticNet; `regularization_metric(frame, case_studies, column)` leaves gaps (not zeros) for missing families; Ridge path = mean daily IC vs alpha with HAC band (flat = benign conditioning; a peak matters only if large relative to the band).
5. Persistence: 3-month rolling mean of daily IC on the common validation window (ETFs and FX; Figure 11.8); per-fold IC and ICIR = |mean IC| / std(IC) are secondary.
6. Multi-label horizon: `regression_labels(cs)`; `HORIZON_DAYS` via `pl.col("label").replace_strict(HORIZON_DAYS, default=None)`.
7. Metric symmetry: Direction A = classification model's `ic_mean_daily` (`task_type='classification'`) vs continuous return; Direction B = regression OOF scores as unweighted mean daily AUC vs binary direction on dates with both classes (`_load_binary_label`, `_aligned_direction_scores`, `_mean_daily_auc -> (auc, n_days)`, `_direction_b_auc`). Ternary labels (NASDAQ-100 `fwd_dir_15m`, crypto `fwd_dir_8h_3c`) excluded.
8. Coefficients: `load_coefficients(cs, training_hash, config_name)` reads only `coef_` and `feature_names` from pickled pipelines (standardized features); silence only `InconsistentVersionWarning`; raise when coefficient count != feature-name count. Sign consistency = share of non-zero folds (|coef| > `ZERO_TOL = 1e-10`) agreeing in sign, all-zero features excluded, `n_active` reported, `SIGN_STABLE_MIN = 0.8`. Lasso sparsity = fraction of zero coefficients at the highest-IC Lasso alpha. Top-N overlap: rank by mean |fold coef|, `TOP_N = 10`, pairwise `jaccard(a, b)`; report the shared feature names of the most similar pair.

### Pedagogical ML backtest (11.6, `11_ml_pipeline/08_ml_backtest_intro`)

| Constant | Value |
|---|---|
| folds | 8 x (10y train, 1y validation), annual step, `label_buffer="21D"` (evaluation section of ETF `setup.yaml`) |
| models | `Ridge(alpha=1.0)`; `LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")`; momentum = rank on `ret_126d`; equal weight 1/N |
| portfolio | month-end rebalance; `TOP_N = 10` equal-weight long-only via `rank_top_n(assets, scores, top_n)` |
| costs | `COST_BPS = 10` per side; `TRADING_DAYS_PER_YEAR = 252` |
| exclusions | `EXCLUDE = {"timestamp","symbol","regime","fwd_ret_21d"}` |
| IC | per rebalance date via `cross_sectional_ic_series`, smoothed over 12 rebalances (`_ic_series(signal_col)`) |

Steps: fit once per fold, predict the whole validation window; forward-fill weights to daily; r_p,t = sum_i w_i,t r_i,t; one-sided turnover at each rebalance; net = gross - cost rate x fraction traded; `max_drawdown(cum_returns)`; summarize annualized gross/net Sharpe, return, volatility, max drawdown, average annual turnover. Return drag = cost rate x fraction traded; Sharpe drag = return drag / strategy vol. Not a production backtest: no execution modeling, slippage or risk management (Ch16-18); regularization untuned (tune in NB04).

### Signature-keyed model caches

- Key caches on content, never on file existence: NB03 `DATA_CONTRACT` (input content hashes + cleaning / symbol limits / row and feature order), `SPLIT_CONTRACT` (label_horizon_sessions, fold state), `MODEL_CONTRACT` (optimization, selection, source implementation, library versions), `TRAINING_CONTRACT`, `INPUT_SIGNATURE`; `assess_cache(cached, expected_signature)` -> missing keys + signature match; shared digests (file content hash, array hash, canonical contract hash) in `utils.modeling`.
- NB06: `NEED_TRAINING = RETRAIN or not RESULTS_PATH.exists()`; `_cache_is_usable(cached) -> (bool, reason)` against `CACHE_SIGNATURE = notebook_cache_signature(...)` over notebook source, input files, cleaned arrays and fit settings; mismatch -> refit.
- `ARTIFACT_TAG = ""` suffixes cache names so a reduced test run never overwrites the full-data cache; `RETRAIN=True` forces a refit.

## Guardrails and pitfalls

- **Singular design from derived features** — exact linear combinations (`skip_recent_*`, `mom_accel_*` are differences of raw returns) break no-perfect-multicollinearity and make robust covariances uncomputable / drop composites before OLS; VIF > 10 flags the rest.
- **Overlapping labels induce autocorrelation** — a 21-day label shares 20/21 days between consecutive rows; naive SEs, t-stats and IC t-stats overstate significance by ~sqrt(overlap) / HAC with lag = label horizon on the per-date IC series; Driscoll-Kraay with lag 21 on the panel; purge/embargo = label horizon in every split.
- **Row-order statistics on a panel** — DW/BG/Newey-West on a date-ordered panel measure cross-sectional dependence / regroup residuals per symbol; count lags on the session grid; collapse to one series per date before HAC.
- **Unbalanced panel gaps** — pairing rows by position files gap-crossing pairs as "lag 1"; pooled fits over-weight later years; date-clustered SEs average over a narrower early cross-section / session-indexed lag pairing; inspect the symbol-history staircase; age weights by session.
- **Robust SEs as a cure-all** — they fix variance only; misspecification leaves coefficients inconsistent and predictions unchanged / treat them as inference tools; judge models by OOS IC.
- **Reading R² out of sample** — intercept level shift and noise-amplifying coefficients drive R² < 0 even with correct ranking / use IC; read R² against zero.
- **NaN vs null in IC aggregation** — constant predictions give NaN Spearman, thin dates give null; one NaN makes the mean NaN; polars drops nulls but keeps NaNs / drop both (`cross_sectional_ic_mean` does).
- **Standardization / winsorization leakage** — percentiles or scaler statistics fitted on test rows leak the future distribution / fit `winsorize_train_test` (1st/99th) and `StandardScaler` on training rows per fold.
- **Reusing a hyperparameter across datasets** — where the sweep turns depends on how much signal the features carry / sweep 1e-2..1e9 on every dataset; read the shape, not the max.
- **Reporting the maximum of a sweep** — selecting and scoring on the same folds inflates the score / nested CV; compare single-loop vs nested mean IC; watch selected-alpha spread.
- **Inner loop without purge/embargo** — the inner split has the same overlapping label; an inner loop producing no folds leaves Optuna returning whatever the sampler tried first / inner `WalkForwardCV` with `label_horizon=21`, `embargo_size=21`, test_size >= label horizon; record pruned folds as missing, never drop.
- **Validation noise dwarfing the mean** — median CV > 1 means no alpha ranking from these folds is reliable / stability table; 3σ gap > 3 = overfitting to validation noise; prefer plateau or ensemble over the exact maximum.
- **Smooth-across-alphas but noisy-across-folds landscape** — an orderly curve invites a confident choice backed by thin evidence / compute both dispersion diagnostics; they can disagree.
- **LASSO selection instability** — with correlated inputs, which feature survives depends on the training window / fold x feature survival heatmap; keep only features present in all folds, or use Elastic Net.
- **Sample weighting that discards data** — recency decay can leave a fraction of the rows (Kish N_eff) / report N_eff and N_eff/N; age by session; never judge on one fold.
- **Zero-filling missing features** — zero is a real value (flat return, zero vol, mid oscillator) / drop rows with nulls.
- **Accuracy as headline for direction models** — majority rate is well above 0.5 on ETFs in a long expansion; a model can beat AUC 0.5 and lose to the accuracy baseline / fit a per-fold majority baseline; report AUC, PR, baseline-relative accuracy.
- **Default 0.5 threshold** — probabilities cluster near the base rate so everything goes to the majority class / choose the threshold from error costs; evaluate across thresholds (ROC/PR).
- **Calibration leakage** — `CalibratedClassifierCV` internal folds default to non-chronological splits / expanding chronological splits with a 21-session purge; verify the corrected curve is closer to the diagonal.
- **Position sizing from uncalibrated probabilities** — sharp-but-miscalibrated probabilities mis-size bets / reliability diagram; hit rate by confidence quintile must rise monotonically.
- **Walk-forward folds treated as a holdout** — every choice (C, alpha, threshold) was made on them; NB07 persistence estimates are validation-only / reserve an untouched stretch for any P&L claim; never report validation results as holdout.
- **Stale or existence-keyed model caches** — refreshed features change fold row counts; edited code leaves old numbers under new source while the .py/.ipynb provenance stamp looks current / signature-keyed caches (`assess_cache`, `_cache_is_usable`); `RETRAIN=True` to force.
- **SHAP magnitude read as skill** — mean |φ| says a feature moved predictions, not that it helped / sign check vs `EXPECTED_SIGNS`; right-vs-wrong split on top-20% conviction; wrong/right ratio > 1 flags misleading features.
- **Model wrong on most of its largest bets** — then SHAP profiles diagnose what misleads it, not skill / read the right/wrong count before the profiles.
- **SHAP background subsampling** — `maskers.Independent` default 100 rows breaks the exact linear identity / `max_samples=len(X_train)`; re-run the closed-form check whenever toolchain, masker or version changes.
- **Concentration threshold that never fires** — Ridge spreads attribution but does not bound one feature's share / read flagged count vs mean and 95th pct of max_frac; aggregate top features; check per fold (regime folds can flip it).
- **Comparing one feature's bootstrap interval to another's point estimate** — ignores the correlation that decides the ranking / bootstrap the difference on the same resamples; claim ranking only when the interval excludes zero.
- **Survivorship in the ETF universe** — `setup.yaml` states the 100 ETFs were chosen backward-looking and the $10M ADV threshold is not inflation-adjusted / point-in-time eligibility via `eligibility.csv`; treat absolute performance with caution.
- **Positive IC != profit** — turnover and costs charged at every rebalance; monthly top-N re-pick trades most of the book; regularization does not lower turnover / rank stability as proxy; `compute_turnover()`; report gross and net Sharpe and turnover; compare to equal weight (gross ≈ net); put costs in the objective (Ch17-18).
- **Sharpe from vol rather than signal** — long-only top-N inherits universe beta; lower realized vol raises Sharpe without return; at equal turnover the steadier strategy loses the larger Sharpe / read return, vol, drawdown and IC separately; compare return drag and Sharpe drag separately.
- **Label leakage across fold boundary in a backtest** — the 21-day label resolves inside the validation window / `label_buffer="21D"` purge.
- **Interpolated conformal quantile** — a slightly narrower width consumes the finite-sample margin the ceiling guarantees / select the exact rank ceil((n+1)(1-alpha)) via `conformal_quantile`.
- **Calibration set too small for requested coverage** — rank exceeds n; no score can certify the level / return an unbounded interval and say so; do not size.
- **Within-date ACI updates on a panel** — one ETF's realized return would set another's interval at the same timestamp / one α̂_t per decision date updated with that date's cross-sectional miss rate.
- **Feeding a 21-day label back before it resolves (ACI)** — look-ahead however "online" the loop looks / `delay_sessions=21`; accept the 21-date warm-up at target alpha.
- **Trusting marginal coverage** — it averages over-coverage in quiet terciles with severe under-coverage in the high-vol tercile / report coverage stratified by regime; expect calibration below the diagonal under walk-forward; apply regime-conditional calibration.
- **Contaminated regime proxy** — the |return| tercile is built from the outcome the interval covers (direction robust, magnitudes not) / in production stratify on ex-ante state (trailing realized vol, VIX).
- **Inverse-width sizing on miscalibrated intervals** — takes the largest positions relative to true uncertainty in the highest-vol regime / check conditional coverage first; cap at 3x median; integrate with vol and correlation forecasts (Ch19).
- **Reading ACI alpha level as current miss rate** — level reflects how much pooled scores needed adjusting / read direction as the lagged coverage record; read level against the calibration quantile it selects.
- **Subsampled training history (CQR)** — coverage unaffected but widths and tercile gaps wider than full-data fits / `TRAIN_SUBSAMPLE=0.25` applied to all three methods for parity; rerun at 1 to compare.
- **Incomplete, retired or name-colliding registry rows** — partially populated results rank spuriously; retired refit generations are bit-identical duplicates; the same `config_name` can carry different parameters / completeness filters; `official_populations`; collapse only rows identical except the two hashes; raise on any measured difference; never `max` / `ORDER BY created_at`.
- **Fold-order-dependent stored HAC intervals** — Newey-West on unsorted daily IC gives the wrong SE / recompute Bartlett HAC on chronologically sorted daily IC with `lag = min(stored lag, n_days-1)`.
- **Hand-written comparison column lists** — drift from `METRICS_QUERY` (a first draft omitted `ic_mean`, `ic_std`) / define exclusions ("all except the two identity hashes").
- **Unpickling fold models across sklearn versions** — `InconsistentVersionWarning`; `predict` not trusted across the gap / read only `coef_` and `feature_names`; raise on count mismatch.
- **Counting zero coefficients as sign agreement** — rewards fits that select nothing / exclude |coef| <= 1e-10 and all-zero features; report `n_active`; threshold 0.8.
- **ICIR with few folds** and **full-period IC hiding regimes** — sample-starved; a positive mean can come from a few stretches / read alongside mean daily IC and per-fold distribution; 3-month rolling IC (both ETF and FX paths cross zero).
- **Low Top-N Jaccard misread as panel-specificity** — could be different feature libraries, and specificity cannot be claimed from one panel / report shared feature names of the most similar pair; say when the comparison is missing.
- **Ridge-path "optimum"** — neighbouring alphas are usually statistically indistinguishable / plot HAC bands; act only if the change is large relative to uncertainty.

## Decision rules and defaults

| Decision | Rule / default |
|---|---|
| Regression vs classification | Continuous returns when position size scales with ŷ (w_i ∝ ŷ_i) or when ranking the cross-section; binary direction when decisions reduce to long/short/flat; ternary terciles when magnitude ordering matters but a point forecast is too noisy. Keep the same folds whichever you choose. |
| VIF | > 5 concern, > 10 severe |
| Autocorrelation lags | 1 (yesterday), 5 (week), 21 (= label horizon where overlap ends) |
| SE estimator on an overlapping-label panel | Driscoll-Kraay with lag = label horizon; HC3 and one- or two-way clustering insufficient; HAC only on the per-date collapsed series |
| IC reporting | mean, IR = mean/std, HAC t-stat with lag = label horizon; read IC against its HAC p-value, not its sign |
| Preprocessing | winsorize at training 1st/99th percentiles, then standardize; refit both per fold |
| Ridge grid | 23 points log-spaced 1e-2..1e9; pick inside the rising/plateau zone |
| LASSO grid | 10 points from `alpha_max` down to 0.01 x `alpha_max` |
| Elastic Net | fix alpha at LASSO's best, sweep `l1_ratio`; tune jointly only via nested CV |
| Recency weighting | half-life 252 sessions; compute Kish N_eff and N_eff/N before adopting |
| Loss | heavy-tailed returns -> Huber (or epsilon-insensitive) over squared error |
| Method comparison | paired per-fold IC differences; overlapping error bars are not evidence either way |
| Alpha landscape (cheap insurance before HPO) | plateau = within 90% of best IC; median CV < 0.5 stable / < 1.0 moderate / >= 1.0 unstable; `gap_sigma` > 3 = overfitting to validation noise; plateau spanning orders of magnitude -> pick inside plateau or ensemble |
| Nested CV | outer 5 folds, test_size 200, label_horizon 21, embargo 10; inner 3 folds, embargo 21, test_size = max(21, inner_sessions // 5); Optuna 20 trials, alpha in [0.01, 1e9] log-uniform |
| Selected-alpha spread | orders-of-magnitude swings across folds = search tracks noise; clustered = stable property of data; read before mean ICs |
| Logistic | lbfgs for L2, liblinear for pure L1 binary, saga for ElasticNet / very large n; `max_iter=1000`; C sweep several orders of magnitude both ways |
| Classification references | AUC vs 0.5; accuracy vs per-fold majority rate; multinomial recall vs 1/3 |
| Platt scaling | 3 expanding chronological blocks starting at 55%-85% of unique dates, 21-session gap |
| Confidence sizing | defensible only if accuracy rises monotonically across confidence quintiles and the reliability curve tracks the diagonal |
| SHAP | full training set as background; top-20% \|ŷ\| = decision-relevant; concentration flag at 60%; 200 bootstrap resamples over top 10 features; claim rankings only from difference intervals excluding zero; sign violations persistent across folds warrant investigation (single-fold flips can be Ridge credit splitting) |
| Conformal method | SC when exchangeability plausible and one width per fold suffices; CQR when width must adapt per observation; ACI when shift expected and delayed feedback available; none fixes conditional under-coverage, regime-conditional calibration does |
| Conformal defaults | coverage 0.90; 80/20 train/calibration per fold; gamma 0.01; delay 21 sessions; alpha clip [0.001, 0.999]; `QuantileRegressor(alpha=0.01, solver="highs")`; levels 0.1..0.9; weight cap 3x median width |
| Go/no-go for sizing off intervals | coverage near target within each ex-ante regime bucket -> proceed; high-vol bucket far below target -> regime-conditional calibration first; rank k = ceil((n+1)(1-alpha)) > n -> unbounded interval, do not size |
| Registry selection (NB07) | eligible = no null fold IC, finite `FINITE_METRIC_FIELDS`, `ic_n_days == max` per label, not retired; rank by `ic_mean_daily`; ties collapse only if identical except hashes or except `config_name`; else raise; significance = CI excludes zero (\|t_HAC\| > 2); sign-stable >= 0.8 among non-zero folds; `TOP_N = 10`; HAC lag = min(stored lag, n_days-1), p from t with n_days-1 df |
| Backtest sanity (NB08) | 8 folds (10y/1y, annual step, 21D buffer); month-end rebalance; top-10 equal-weight long-only; 10 bps per side; IC smoothing 12 rebalances; report gross and net Sharpe, return, vol, drawdown, turnover separately |

Sequence (chapter order): (1) inspect panel shape and feature families -> (2) drop composites and null rows -> (3) walk-forward splits with 21D buffer -> (4) per-fold winsorize + scale -> (5) OLS baseline -> (6) alpha landscape -> (7) nested HPO -> (8) paired comparison + rank stability -> (9) SHAP sign / stability / right-vs-wrong -> (10) conformal intervals with conditional coverage -> (11) cross-study comparison -> (12) tradability net of costs -> (13) untouched holdout for any P&L claim. Downstream: Ch16 production backtest -> Ch17 turnover-constrained construction -> Ch18 cost modeling -> Ch19 uncertainty-aware sizing.

## Code patterns and APIs

| Library / module | API |
|---|---|
| `ml4t.diagnostic.metrics` | `cross_sectional_ic_series`, `compute_ic_hac_stats` (lag = label horizon) |
| `ml4t.diagnostic.splitters` | `WalkForwardCV(n_splits, test_size, label_horizon, embargo_size)` |
| `ml4t.diagnostic.signal` | `compute_turnover()` |
| `ml4t.diagnostic.evaluation` | `compute_shap_importance()` |
| `utils.modeling` | `cross_sectional_ic_mean(y_true, y_pred, dates, symbols)`, `generate_cv_splits(df, case_study_id, label_buffer, date_col)`, `conformal_quantile`, cache digests |
| `utils.style` | `COLORS`, `ml4t_diverging`, `show_with_alt` |
| path helpers | `get_case_study_dir("etfs")`, `get_chapter_dir(11)`, `get_output_dir(11, "<nb>")`, `data.load_etfs()` |
| sklearn | `Ridge(alpha, random_state=42)`, `Lasso`, `ElasticNet(alpha, l1_ratio)`, `SGDRegressor`, `lasso_path`, `StandardScaler`, `LogisticRegression(penalty, C, solver, class_weight, max_iter=1000)`, `CalibratedClassifierCV(base, method="sigmoid", cv=<chronological splits>)`, `calibration_curve(y, p, n_bins=10)`, `QuantileRegressor(quantile, alpha=0.01, solver="highs")`, `sklearn.exceptions.InconsistentVersionWarning` |
| statsmodels / scipy / optuna / shap | OLS with `cov_type` HC3 / cluster / Driscoll-Kraay; `spearmanr`; `trial.suggest_float("alpha", 0.01, 1e9, log=True)`; `shap.maskers.Independent`, `shap.LinearExplainer` |
| registry (`run_log/registry.db`) | `prediction_metrics` (ic_mean_daily, ic_se_hac, ic_ci_lo, ic_ci_hi, ic_t_hac, ic_p_hac, ic_n_days, ic_hac_lag, task_type); fold metrics (fold_id, ic, ic_std, n_entities, rmse, mae); `official_populations`; identities `training_hash`, `prediction_hash`, `config_name` |
| test-mode knobs | `MAX_SYMBOLS`, `MAX_CV_FOLDS` / `CV_MAX_SPLITS` / `CV_MAX_TRIALS`, `VIF_MAX_ROWS`, `MAX_FOLDS` (0 = unlimited); `ARTIFACT_TAG`; `RETRAIN` |

Running: `uv run python 11_ml_pipeline/<notebook>.py`; test mode `uv run pytest tests/test_chapter_notebooks.py -v -k "11_ml_pipeline"`; Docker image `ml4t`.

```python
# Kish effective sample size for recency weights (age in sessions, not rows)
lam = np.log(2.0) / HALF_LIFE_SESSIONS            # 252
w = np.exp(-lam * (T - session_idx)); w /= w.mean()
n_eff = w.sum() ** 2 / (w ** 2).sum()
```

```python
# LASSO grid anchored at alpha_max (standardized X0, y0 from the first fold)
alpha_max = float(np.max(np.abs(X0.T @ y0)) / len(y0))
LASSO_ALPHAS = np.logspace(np.log10(alpha_max), np.log10(0.01 * alpha_max), 10)
```

```python
# Exact linear SHAP with full-training background
masker = shap.maskers.Independent(X_train, max_samples=len(X_train))
exp = shap.LinearExplainer(model, masker)(X_test)   # .values, .base_values, .data
assert np.allclose(exp.values, model.coef_ * (X_test - X_train.mean(0)))
```

```python
# 3-sigma validation-overfitting flag over grid/trial scores
gap_sigma = (ics.max() - np.median(ics)) / ics.std()
flag = gap_sigma > 3.0
```

```python
# ACI update: once per decision date, using the miss rate of the date 21 sessions earlier
alpha_t = float(np.clip(alpha_t + gamma * (target_alpha - miss_rate), 0.001, 0.999))
# Position-size bridge from interval width
position_weight = np.clip(median_width / interval_width, 0, 3)   # cap at 3x median
```

```python
# Signature-keyed cache guard
NEED_TRAINING = RETRAIN or not RESULTS_PATH.exists()
ok, reason = _cache_is_usable(cached)   # signature over source, inputs, arrays, settings
```

```python
# Registry completeness and significance (polars)
df.with_columns(_max_n_days=pl.col("ic_n_days").max().over("label")) \
  .filter(pl.col("ic_n_days") == pl.col("_max_n_days"))
rank1.filter((pl.col("ic_ci_lo") > 0) | (pl.col("ic_ci_hi") < 0))
frame.pivot(on="reg_family", index="short_name", values="ic_mean_daily")
pl.col("label").replace_strict(HORIZON_DAYS, default=None).cast(pl.Float64)
```

Repo paths: `11_ml_pipeline/0{1..8}_*.py` (Jupytext); caches under `models/02_regularization_paths/`, `models/04_nested_cv_hpo/`, `models/06_conformal_prediction/`, `output/11/03_logistic_classification/`; `output/05_shap_analysis/shap_arrays.npz`; `output/11/06_conformal_prediction/calibration.parquet`; `case_studies/{cs}/06_linear.py`, `case_studies/{cs}/13_model_analysis.py`, `case_studies/etfs/setup.yaml`.

## Evidence from the book

Exact IC, AUC, coverage, width, alpha and Sharpe values are rendered at run time and were not in the notes; findings below are qualitative.

- **`01_ols_inference`** (first fold, hundreds of thousands of rows): many small p-values with very low R²; Breusch-Pagan rejects (volatility clustering); Jarque-Bera rejects (fat tails); many features VIF > 10; per-symbol residual autocorrelation strong through lag 20 by construction; three symbols resume after gaps >= 1 year. Robust SE ratios grow HC3 < cluster-by-date < two-way < Driscoll-Kraay, up to ~3x OLS; the largest coefficients are mostly indistinguishable from zero under Driscoll-Kraay. Naive IC t-stat overstates vs HAC; OOS R² can be negative.
- **`02_regularization_paths`** (8 folds): Ridge sweep shows flat -> improving -> over-regularized zones; LASSO reaches IC comparable to Ridge while zeroing a substantial fraction of features; feature count falls over a range where validation IC does not (sparsity nearly free there, but dropped contributions are absorbed by correlated survivors); surviving LASSO features flicker across folds; Ridge rank correlation between consecutive fold models is high (stable rankings, low implied turnover). Error bars across methods are wider than the gaps between them. Recency weighting leaves N_eff a fraction of the fold; the sign of the IC change on one fold is not conclusive.
- **`03_logistic_classification`**: positive 21-day ETF returns are more common, so majority baseline accuracy is well above 0.5; ROC sits just above the diagonal, AUC barely above 0.5; calibration curve nearly flat; confusion matrix at 0.5 lopsided to the majority class; the AUC-optimal L1 fit retains most features; tercile recalls asymmetric. Conclusion: a workflow demonstration, not a deployable directional edge.
- **`04_nested_cv_hpo`** (5 outer folds): the single-loop score is the more favourable of each pair; nested is the honest procedure estimate; single-loop alpha swings across orders of magnitude between folds while nested selection moves smoothly; where bars are equal both picked the same alpha. The landscape can be smooth across alphas while fold CV is high.
- **`05_shap_analysis`** (last fold, Ridge alpha=1.0): the closed-form identity holds to float precision with full background; Ridge credit-splitting causes occasional sign flips among correlated features; the 60% concentration flag may fire on no rows; cross-fold importance trajectories and within-fold bootstrap bands can disagree on which features are "stable".
- **`06_conformal_prediction`** (99 ETFs, 90% target, TRAIN_SUBSAMPLE=0.25): all three methods land near target marginally and all under-cover; the multi-level calibration curve sits below the diagonal. CQR buys coverage more cheaply with per-observation width; SC fixes width per fold so coverage moves across folds instead. Stratified by |return| tercile, all three cover almost everything in the two quiet terciles and fall far short in the high-vol tercile; CQR and ACI narrow but do not remove the shortfall, at the cost of width elsewhere. The ACI alpha path is a lagged record of realized coverage; its settled level need not equal nominal alpha.
- **`07_case_study_insights`**: both rolling-IC paths (ETFs and FX, common validation window) cross zero, so no full-period average describes every regime (Figure 11.8). `crypto_perps_funding` at `fwd_ret_24h`: `ols`, `ridge_a0.001`, `ridge_a0.01` tie exactly (Ridge at inert shrinkage reproduces OLS). Across Ridge paths, most neighbouring alphas are statistically indistinguishable (HAC bands overlap).
- **`08_ml_backtest_intro`** (same 8 folds): IC ordering and net-Sharpe ordering of Ridge / Logistic / momentum / equal-weight differ. Equal weight's gross and net Sharpe are nearly identical (trades only 1/N drift); monthly top-10 re-pick trades most of the book; at 10 bps per side, cost drag separates high-turnover ML strategies from low-turnover baselines. A near-zero or negative IC signal can still post a respectable net Sharpe via universe return and lower vol.
- **Reader uncertainty carried from the notes**: exact setup.yaml fold window lengths for the 8 Ch11 folds; `CONF_Z`; the exact Driscoll-Kraay statsmodels call; the logistic C grid and `l1_ratio` grid; `n_sgd`; the minimum cross-section for `cross_sectional_ic_mean`; `EXPECTED_SIGNS` contents beyond momentum +, volatility -; the three concrete position-conversion rules; the field list inside `notebook_cache_signature`; which module exports `PRIMARY_LABELS` and the full `HORIZON_DAYS`; whether the |return| tercile is pooled or per fold; whether the NB08 logistic signal is P(positive 21-day return) (inference); the 95% level behind `critical` and the CQR alpha/2 per-tail mapping (inferences).

## Related references

- `chapters/07_defining_the_learning_task.md` — `fwd_ret_21d` labels, label buffer, triple-barrier concurrency for uniqueness weights, direction and tercile labels.
- `chapters/08_financial_features.md` — the ETF feature families (momentum, volatility, `skip_recent`, `mom_accel` composites) whose collinearity drives every diagnostic here.
- `chapters/09_model_based_features.md` — regime and model-derived features used for ex-ante regime-conditional calibration.
- `chapters/12_gradient_boosting.md` — the next model family that must beat this baseline; LightGBM Huber objective, TreeExplainer SHAP, nested CV in higher-dimensional space.
- `chapters/13_dl_time_series.md` — KernelSHAP for deep models; same folds and IC protocol.
- `chapters/14_latent_factors.md` — alternative model family compared against the linear baseline.
- `chapters/16_strategy_simulation.md` — production backtesting that replaces the NB08 pedagogical loop.
- `chapters/17_portfolio_construction.md` — probability-to-position framework, turnover-constrained construction, turnover-adjusted evaluation.
- `chapters/18_transaction_costs.md` — cost modeling that NB08 approximates with 10 bps per side.
- `chapters/19_risk_management.md` — uncertainty-aware sizing from conformal widths with vol and correlation forecasts.
- `chapters/26_mlops_governance.md` — cache signatures, registries, generations and coverage monitoring in production.
- `case_studies/etfs.md` — the panel every Ch11 notebook runs on (100 ETFs, setup.yaml, survivorship caveat).
- `case_studies/fx_pairs.md` and `case_studies/crypto_perps_funding.md` — rolling-IC and Ridge/OLS tie evidence in NB07.
- `case_studies/nasdaq100_microstructure.md` — ternary `fwd_dir_15m` label excluded from the metric-symmetry check.
- `libraries/ml4t_diagnostic.md` — `WalkForwardCV`, `cross_sectional_ic_series`, `compute_ic_hac_stats`, `compute_turnover`, `compute_shap_importance`.
- `libraries/ml4t_models.md` — linear-family runners (`06_linear.py`) and registry conventions.
- `workflow.md`, `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `companion_repo.md` — cross-cutting indexes that this chapter's rules feed.
- Further reading: Breiman 2001 (two cultures); Simonian 2024 (econometrics vs ML); Hastie, Tibshirani & Friedman 2009 (ESL); Tibshirani 1996 (LASSO); Zou & Hastie 2005 (Elastic Net); Huber 1964; Cawley & Talbot 2010 (selection bias); Akiba et al. 2019 (Optuna); Lundberg & Lee 2017 (SHAP); Shapley/Kuhn 1953; Kumar et al. 2020 and Aas et al. 2021 (Shapley with dependent features); Niculescu-Mizil & Caruana 2005 (calibration); Gu, Kelly & Xiu 2020 (asset pricing via ML); Angelopoulos & Bates 2022, Papadopoulos 2002, Lei et al. 2017, Romano et al. 2019 (CQR), Gibbs & Candès 2021/2023 (ACI), Barber et al. 2023, Tibshirani et al. 2020, Sun & Yu 2024/2025 (conformal); O'Donovan & Yu 2024 (transaction costs).

## Glossary

- **BLUE** — Best Linear Unbiased Estimator; OLS under the four Gauss-Markov assumptions (linearity, strict exogeneity, no perfect multicollinearity, spherical errors).
- **Spherical errors** — constant variance and no correlation across observations.
- **Breusch-Pagan / Jarque-Bera** — heteroscedasticity test / residual normality test (the latter is not Gauss-Markov).
- **VIF** — variance inflation factor; > 5 concern, > 10 severe.
- **HC3 / clustered / Driscoll-Kraay / HAC (Newey-West)** — covariance estimators admitting, respectively, heteroscedasticity only; within-cluster correlation; arbitrary cross-sectional plus serial correlation up to a lag; serial correlation in a single series.
- **Fama-MacBeth** — cross-sectional regression per date, inference from the time series of estimates.
- **IC / ICIR (IR)** — Spearman rank correlation of predictions and realized returns within a date / mean IC divided by its std.
- **Walk-forward CV / purge / embargo** — chronological rolling splits / removing training rows whose labels overlap the test window / extra gap after it.
- **Winsorization** — clipping at training percentiles (1st/99th here).
- **Ridge / LASSO / Elastic Net / `l1_ratio`** — L2 / L1 / mixed penalties; `l1_ratio` sets the mix.
- **alpha_max** — smallest LASSO alpha that zeros every coefficient, max|Xᵀy|/n.
- **Regularization path** — coefficient trajectories as the penalty varies.
- **Kish effective sample size** — (sum w)²/sum w²; row-equivalent of a weighted fit.
- **Huber / epsilon-insensitive loss** — quadratic then linear / zero inside a band then linear.
- **Nested CV / selection bias / 3σ heuristic** — inner loop selects, outer scores / upward bias from choosing and scoring on the same data / flag when the top score exceeds the median by > 3 inter-configuration SDs.
- **C** — inverse regularization strength in logistic regression.
- **Calibration vs discrimination** — stated probabilities match observed frequencies vs ranking ability (AUC).
- **Platt scaling** — sigmoid fitted to raw scores (`CalibratedClassifierCV(method="sigmoid")`).
- **SHAP / masker (background) / concentration risk / decision-relevant predictions** — Shapley attributions, φ_j = β_j (x_j - x̄_j) for linear models / reference distribution SHAP integrates over / one feature's share of a row's |SHAP| (flag > 60%) / top 20% by |ŷ|.
- **Nonconformity score** — |residual| for SC; max lower/upper quantile exceedance for CQR.
- **Split conformal / CQR / ACI** — fixed half-width from a calibration set / quantile regressors plus conformal correction (adaptive width) / online update of α̂_t with step gamma.
- **Coverage gap / conditional coverage / regime-conditional calibration** — actual minus target / coverage within a stratum / separate conformal correction per ex-ante state.
- **Exchangeability** — joint distribution invariant to permutation; the assumption behind the marginal guarantee.
- **Warm-up** — first 21 dates of an ACI fold at target alpha because no label has resolved.
- **Complete result / generation / name-only tie** — registry row with no null fold IC and maximum validation-day coverage / refit under a corrected input artifact, superseding prior rows via `official_populations` / rows identical on every statistic, differing only in `config_name`.
- **Sign consistency** — share of non-zero folds agreeing on a coefficient's sign.
- **Direction A / B** — classification score as ranker (IC) / regression score as classifier (mean daily AUC).
- **One-sided turnover / net Sharpe** — sum of absolute weight changes at a rebalance / Sharpe after charging `COST_BPS` per side on turnover.
