# Chapter 12: Gradient Boosting and Advanced Tabular Models

> Chapter 12 takes you from single trees through Random Forests (which fix variance, not bias) to gradient boosting (sequential error correction that attacks bias), then through everything needed to run a GBM responsibly on a financial panel: choosing among XGBoost / LightGBM / CatBoost / sklearn HistGB on engineering grounds, picking objectives (MAE vs MSE, LambdaMART, monotone constraints), tuning with Optuna under walk-forward validation with pruning and a declared holdout, explaining with TreeSHAP while respecting instability and the Rashomon effect, and wrapping point forecasts in conformal intervals whose coverage you must verify. Its position: GBMs remain the default for financial tabular prediction; the library matters less than the loss and the tuning protocol; a single split with no interval ranks nothing; and the cross-case-study view (12.6) is a computed, coverage-matched reading of LightGBM across the nine case studies against the linear and TabM baselines, with inference attached to the daily IC series rather than fold summaries, cross-family deltas kept descriptive, and gaps drawn as gaps. These notes were extracted from the companion repo (`12_gradient_boosting/`), not the book PDF; numeric results print at run time and are recorded here only as directions.

## When to use this reference

- Fitting or comparing a GBM (LightGBM / XGBoost / CatBoost / sklearn HistGradientBoosting) or a Random Forest on a cross-sectional return panel.
- Choosing the loss or objective for a return model: MSE vs MAE vs Huber regression, `lambdarank` for top-k selection, binary direction labels.
- Deciding whether to impose a monotone constraint on a feature.
- Designing Optuna HPO for a GBM: search space, early stopping, IC-based pruning, single-fold vs walk-forward objective, trial budget, SQLite persistence, holdout acid test.
- Deciding grid search vs Bayesian (TPE) search, or adding turnover as a second objective (NSGA-II Pareto front).
- Reusing hyperparameters tuned on one asset class (ETFs) for another (crypto, CME futures).
- Explaining a tree model: native importance vs SHAP, MDI/PFI/SHAP consensus, interactions and H-statistic, walk-forward SHAP drift, SHAP-based feature selection, stakeholder reporting.
- Producing prediction intervals for a GBM (split conformal, quantile regression, CQR, ACI) with purged calibration, and sizing exposure from interval width.
- Judging whether tabular deep learning (torch MLP, TabM, TabPFN, TabR) is worth trying against a GBM, and how to report a paired walk-forward comparison.
- Running GBMs on GPU vs CPU, and reproducing a fit exactly.
- Reading or extending the cross-case-study GBM insight notebook: rank-one selection from the registry, HAC confidence intervals on daily IC, loss/depth/checkpoint grids, holdout decay, GBM-vs-linear and GBM-vs-TabM comparisons, classification/regression symmetry, feature-rank shift.

## Core ideas (the why)

- Trees suit financial tabular data because they learn nonlinear, threshold-driven structure linear models miss. RF averaging reduces variance but "leaves systematic bias where it is"; boosting's sequential error correction attacks bias, so a GBM must beat an RF baseline to justify its complexity (12.1).
- GBMs remain the default for most financial tabular problems; TabPFN / TabM / TabR are judged by a regime-based decision framework (temporal shift, production constraints), "not hype" (12.3).
- A single split with no interval does not rank models. Read two numbers against each other: the between-model spread on test and one model's validation-to-test move; when the second is larger, a ranking is reading noise (`01_ensemble_foundations`, `02_gbm_comparison`).
- Rank IC and R² answer different questions (ordering of the cross-section vs distance to realized return) and can order the same models differently; which one matters is a strategy decision made before the comparison.
- Early stopping selects complexity from data; a fixed epoch or tree count is "a hyperparameter nobody measured".
- Native feature importances are not comparable across libraries even after rescaling (impurity, gain, split count, prediction-value change are different definitions). SHAP is the one definition shared by every model.
- Loss is a first-class hyperparameter. MAE/L1 beats MSE on heavy-tailed returns because squared error chases a few large residuals at the expense of the cross-sectional ranking that IC rewards. `loss_type` was the single most important hyperparameter in every cross-library study, and the spread of test IC across libraries is smaller than the MSE-to-MAE gap: the library matters less than the tuning.
- Validation overfitting: one validation window barely constrains a search over eight hyperparameters; the selected config is partly a fit to that window's noise. The trial budget is a parameter to choose, not to maximize.
- An undefined IC (near-constant predictions, too-narrow cross-section) is not a bad score and not a smaller sample; how a search treats unscorable folds decides what it selects.
- A monotone constraint is regularization when the direction is right and a silent way of dropping a feature when it is not: the cheap way to satisfy a non-increasing constraint is to flatten the feature.
- Ranking (LambdaMART) and regression objectives answer different questions: ranking for top-k selection within a rebalance, regression for a magnitude that sizes a position. An NDCG advantage for a model trained on a ranking surrogate is "the setup working as arranged".
- Grid vs Bayesian is decided by the space, not the sampler: on a grid small enough to exhaust, enumeration wins by construction; continuous spaces are where TPE reaches configurations a grid cannot represent.
- Paired evaluation: when models share folds, the fold-to-fold swing is shared and cancels in the per-fold difference. Compare differences, their sign frequency and size, not overlapping error bars.
- Explanations are not unique (instability) and are model-specific (Rashomon). SHAP reflects fitting patterns, not ground truth; never claim causation.
- Conformal's finite-sample guarantee assumes exchangeability; financial residuals shift, so empirical coverage must always be checked. Wider intervals are not better covered.
- Cross-asset hyperparameter transfer is fragile; when feature distributions differ, asset-specific tuning is mandatory.
- GPU and CPU fits are related but distinct estimators (different histogram precision, different splits, IC differences larger than rounding); GPU is not bitwise reproducible.
- Cross-case-study reading (12.6): the headline metric is average daily cross-sectional Spearman IC with a HAC confidence interval on the chronological daily series. Per-fold IC is a stability diagnostic; validation-to-holdout decay is a generalization diagnostic; neither is an uncertainty estimator. Comparability must be earned (same folds, same number of scored days), gaps are drawn as gaps, narrative is computed from the rows just produced, and ambiguity (tied candidates) is excluded rather than resolved arbitrarily.
- Rank shifts between GBM and linear importances describe how two families use a shared feature library; they do not establish that a promoted feature causes the GBM-minus-linear performance difference (12.6). Importance is measured on the selected checkpoint, not the full saved booster.

## Method recipes (the how)

### Section map (companion repo `12_gradient_boosting/`)

Run with `uv run python 12_gradient_boosting/<notebook>.py`; test mode `uv run pytest tests/test_chapter_notebooks.py -v -k "12_gradient_boosting"` (reduced data via Papermill). Docker image `ml4t` unless noted; GPU notebooks: `docker compose run --rm ml4t-gpu python 12_gradient_boosting/05_cross_library_hpo.py`.

| Section | Notebook | Does | Data | Runtime / image |
|---|---|---|---|---|
| 12.1 | `01_ensemble_foundations` | RF vs XGBoost / LightGBM / CatBoost; GBM-over-bagging gap; spread across libraries | Chen-Pelger-Zhu (2020) stock-month panel via `load_firm_characteristics`, predefined temporal splits | ~6 min, `ml4t` |
| 12.2 | `02_gbm_comparison` | 4 libraries x 3 presets x {CPU, GPU}; 5M-row scale benchmark; monotone constraint + SHAP dependence; LambdaMART demo | ETF panel (99 symbols, ~251K rows, 71 features, 227K training rows), one walk-forward fold; US Equities Panel (~5M training rows) | ~45 min, ~52 GB peak RAM, `ml4t-gpu` |
| 12.3 | `03_dl_vs_gbm` | Walk-forward IC and train time: LightGBM vs torch MLP vs TabM vs TabPFN; reports per-fold IC, train time, `best_iter` / `best_epoch`, learning curves, paired per-fold differences | `case_studies/etfs/features/` + Ch11 artifacts, 8 paired folds | GPU recommended, `ml4t-gpu` |
| 12.4 | `04_optuna_tuning` | Single-fold Optuna HPO with early stopping + IC pruning; fANOVA; walk-forward HPO; holdout acid test; reports best params, default-vs-tuned table (`n_trees`), single-fold vs walk-forward holdout table | ETF features, pre-holdout dates with embargo, declared holdout | ~11 min, `ml4t` |
| 12.4 | `05_cross_library_hpo` | Optuna for XGBoost / LightGBM / CatBoost at identical budgets, `loss_type` categorical; writes a SQLite study per library; reports best-valid / best-test IC, per-loss bars, fANOVA importance incl. `loss_type` | CPZ panel, held-out 2000-2016 | ~25 min, GPU recommended, `ml4t-gpu` |
| 12.4 | `06_optuna_multi_asset` | NSGA-II multi-objective (IC vs turnover); cross-asset transfer ETF -> crypto / CME futures; reports single-objective baseline, Pareto front, transfer-gap table with efficiency ratio | ETF primary; crypto, CME futures on shared columns | `ml4t` |
| 12.4 | `07_hpo_comparison` | Grid vs TPE at equal budget on discrete then continuous space; validation-overfitting ladder; reports efficiency curve, val-vs-holdout IC across `TRIAL_BUDGETS`, final holdout for the two picks | ETF features (`MAX_SYMBOLS` caps universe; 0 = full) | ~15 min, `ml4t` |
| 12.5 | `08_shap_analysis` | TreeSHAP global + local, MDI/PFI/SHAP consensus, interactions, H-statistic, drift, top-k selection | ETF features, LightGBM, walk-forward folds | `ml4t` |
| 12.5 | `09_xai_limitations` | Explanation instability pairs; Rashomon across 3 LightGBM seeds + RF; stakeholder guidance | ETF features | - |
| 12.5 | `10_shap_nlp_sentiment` | Token-level SHAP on FinBERT-tone; narrowed-vs-widened perturbation | pinned FinBERT-tone; 5 probe + 10 constructed sentences (teaching sample) | GPU recommended, runs on CPU |
| 12.5 | `11_conformal_gbm` | Split conformal, QR, CQR, ACI for LightGBM with purged calibration; coverage by asset class; width-based sizing | `case_studies/{etfs,crypto_perps_funding,cme_futures}/features/`, canonical pre-holdout folds, `setup.yaml` horizon | `ml4t`, deterministic CPU |
| 12.6 | `12_case_study_insights` | Daily-pooled IC with HAC; loss/depth/leaf grids; checkpoint peaks; holdout decay; Linear / GBM / TabM | each CS `run_log/registry.db` (no training) | `ml4t` |

### Ensemble baselines: RF vs boosting (`01_ensemble_foundations`)

1. Load the CPZ panel with `load_firm_characteristics` and use its predefined temporal splits.
2. Fit sklearn RF, XGBoost, LightGBM, CatBoost with hyperparameters following Gu, Kelly & Xiu (2020) asset-pricing guidelines (exact values not in the notes).
3. Score with `evaluate_model`: Spearman rank IC (primary), R², MSE on validation and test.
4. Read the between-model spread on test against one model's val-to-test move before claiming an order.

Library cheat sheet (cross-notebook: `01` fits RF, XGBoost, LightGBM, CatBoost; sklearn HistGradientBoosting enters only in `02_gbm_comparison`'s benchmark grid):

| Library | Complexity control / mechanics | API idiosyncrasies | Native importance (rescale to share of own total before side-by-side plots) |
|---|---|---|---|
| sklearn RF | bagging; variance reduction only | - | normalized gain (sums to 1) |
| XGBoost | L1/L2 on leaf weights, second-order gradients, early stopping picks iterations | `n_estimators`, `max_depth`, `reg_lambda`, `colsample_bytree` (per tree), `n_jobs`; GPU `device="cuda", tree_method="hist"` | normalized gain (sums to 1) |
| LightGBM | leaf-wise growth + histogram binning; `num_leaves` (not `max_depth`) is the primary complexity control | `n_estimators`, `num_leaves`, `reg_lambda`, `n_jobs`; GPU `device="cuda"` (needs `-DUSE_CUDA=1` build; `ml4t-gpu` image) | raw split counts (sum = total splits) |
| CatBoost | symmetric (oblivious) trees: all nodes at a depth share one split, so inference is a bitwise lookup | `iterations` (not `n_estimators`), `depth` (not `max_depth`), `l2_leaf_reg` (not `reg_lambda`), `colsample_bylevel` (per level), `thread_count` (not `n_jobs`), `task_type="GPU"` | prediction-value change (sums to 100) |
| sklearn HistGradientBoosting (`02` only) | histogram GBM; threads via `OMP_NUM_THREADS` | no GPU | - |

### Library benchmark and engineering choice (`02_gbm_comparison`)

- Grid: {HistGB, XGBoost, LightGBM, CatBoost} x presets {light, medium, heavy} x {CPU, GPU}. Presets come from `load_gbm_config(name)` in `case_studies/utils/gbm.py` (repo-verified; the notes did not see the module) with library-neutral keys `n_trees`, `max_depth`, `lr`, `l2`, `subsample`, `colsample`, `min_leaf` mapped per library via `PARAM_NAMES`: light = 200 trees / depth 4 / lr 0.10, medium = 500 / 6 / 0.05, heavy = 1000 / 8 / 0.01, all with `l2=1.0`, `subsample=0.8`, `colsample=0.8`, `min_leaf=20` (values from the repo, not the notes; separate from the case-study LightGBM presets in `case_studies/config/lgb/`).
- Threads: `N_JOBS = 8` (`n_jobs=8` for XGBoost/LightGBM, `thread_count=8` for CatBoost, `OMP_NUM_THREADS=8` for sklearn). Split decisions are thread-count independent; only wall time moves.
- GPU probe `detect_gpu_capabilities()` fits 2-tree models: `XGBRegressor(n_estimators=2, device="cuda", tree_method="hist")`, `LGBMRegressor(n_estimators=2, device="cuda")`, `CatBoostRegressor(iterations=2, task_type="GPU", allow_writing_files=False)`.
- Measure: IC, wall-clock train time, `predict()` latency and rows/sec (CPU vs GPU), RSS delta around `fit` via `get_rss_mb` (a crude lower bound).
- Scale benchmark on the ~5M-row US Equities Panel; CPU rows at `N_JOBS`.

### Objectives and constraints

**Regression loss.** Default to MAE/L1 for return regression; treat `loss_type` in {MSE, MAE} as a categorical hyperparameter when budget allows (`05_cross_library_hpo`). The case-study GBM grid also includes Huber (`12_case_study_insights`).

**Learning to rank (LambdaMART, `02_gbm_comparison`).**
1. Query groups = dates. Relevance = `min(int(pct_rank * 5), 4)` (`np.minimum((pct * 5).astype(int), 4)`): 5 graded levels 0-4 from the forward-return percentile within each date.
2. `lgb.train({"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10], "learning_rate": 0.05, "num_leaves": 31}, train_set_with_group)`; regression baseline `{"objective": "regression", "learning_rate": 0.05, "num_leaves": 31}`.
3. Score both with `mean_group_ndcg(y_true_rel, y_score, groups, k=10)` (sklearn `ndcg_score`, k capped at group size) AND cross-sectional IC on raw returns. Judge by IC and by strategy need, across folds.

**Monotone constraints (`02_gbm_comparison`).** Fit unconstrained vs `LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, monotone_constraints=mc)` with `mc` a list of {-1, 0, +1} per feature (`-1` non-increasing on the highest-importance feature in the demo). Compare SHAP dependence panels on one vertical scale and test ICs. Impose only when the direction is known to be right and SHAP dependence bends rather than flattens across folds.

### Model-comparison protocol and deep tabular alternatives (`03_dl_vs_gbm`)

| Step | Rule |
|---|---|
| Validation slice | the last `VAL_FRACTION = 0.2` of each fold's train window, in chronological order, becomes validation (`_get_fold_data(split, val_frac=VAL_FRACTION)`; value from the repo); the test slice is never touched by any early-stopping signal |
| LightGBM | `eval_set=[(X_val, y_val)]` + `lgb.early_stopping(30)`; report `best_iter` |
| MLP floor | `TorchMLPModel` 64-32 ReLU, Adam, L1 loss; `train_torch_mlp(..., epochs=200, lr=1e-3, patience=20)` early stop on val-loss plateau |
| TabM | `TabMModel(n_features, hidden_dim=64, n_members=8)`: shared MLP backbone with M rank-one scaling adapters trained jointly (members overfit, average generalizes); `train_tabm(..., epochs=200, lr=1e-3)` restores best-val checkpoint, reports `best_epoch` |
| TabPFN | zero-shot, no training or validation; gated weights need `TABPFN_TOKEN` (register at https://ux.priorlabs.ai, see `.env.example`); free for research/evaluation, not commercial; notebook skips it if missing |
| Loss | all trainable models use L1/MAE; the notebook's stated rationale is that MAE had the highest IC on 8 of 9 regression-primary case studies, citing `12_case_study_insights` 3a (a count that notebook recomputes at run time, so treat 8/9 as NB03's rationale, not a fixed result) |
| Reporting | per-fold paired differences: mean, spread, sign frequency, described not tested (folds overlap in time) |

### Optuna HPO for LightGBM (`04_optuna_tuning`)

Constants: `N_TRIALS = 50` (book suggests low hundreds), `LABEL_HORIZON = 21` (matches `fwd_ret_21d`), `IC_MIN_OBS = 10`, `SEED = 42` (every chapter notebook uses 42; repo-verified). The objective sets the `n_estimators` ceiling at 500 (repo). Embargo: `val_embargo_cutoff = pre_holdout_dates[-(LABEL_HORIZON + 1)]` (the `-LABEL_HORIZON` index lands on the holdout's first day).

| Hyperparameter | Range | Sampling |
|---|---|---|
| `max_depth` | 2-8 | int |
| `num_leaves` | 8-64 | int |
| `min_child_samples` | 5-50 | int |
| `learning_rate` | 0.01-0.2 | float, log |
| `subsample` | 0.5-1.0 | float |
| `colsample_bytree` | 0.5-1.0 | float |
| `reg_alpha` | 1e-4-10 | float, log |
| `reg_lambda` | 1e-4-10 | float, log |
| `n_estimators` | set high | `lgb.early_stopping(50, verbose=False)` picks the count |

1. Sampler `TPESampler(seed=SEED)`; pruner `MedianPruner(n_startup_trials=5, n_warmup_steps=10)`; `optuna.create_study(direction="maximize", ...)`.
2. Pruning callback `ICPruningCallback(trial, X_eval, y_eval, dates_eval, symbols_eval, report_every=20)` reports validation cross-sectional IC every 20 rounds (`optuna_integration.LightGBMPruningCallback` only supports minimization metrics). Body (verified in the repo; the notes inferred it): on `(env.iteration + 1) % report_every == 0`, predict on the eval set and compute `cross_sectional_ic_mean`; if the IC is non-finite, return without reporting (an undefined IC is not a score); otherwise `trial.report(ic, step=env.iteration)` and raise `optuna.TrialPruned` if `trial.should_prune()`. The callback is defined inline in `04_optuna_tuning.py`, as are `cross_sectional_ic_mean`, `prepare_fold_data`, `fold_is_scorable`, `widest_cross_section` and `walkforward_objective`.
3. Compare against the default baseline `LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)`; the comparison table carries an `n_trees` column. Inspect it.
4. fANOVA importance: variance decomposition of the validation objective across hyperparameters.
5. Walk-forward objective (recommended default): `prepare_fold_data(fold_idx)`; drop folds where no validation date has >= `IC_MIN_OBS` names (`fold_is_scorable`) BEFORE the search; `widest_cross_section(dates_va, y_hat)` detects tied predictions; a trial that cannot rank one fold is pruned (no value), never scored at -1; objective = mean IC over the same scored folds for every trial. Cost scales with the number of folds, offset by pruning.
6. Acid test: score both tuned sets (single-fold, walk-forward) once on the declared holdout.

### Cross-library HPO (`05_cross_library_hpo`)

- One study per library; `loss_type = suggest_categorical(["mse", "mae"])` mapped to XGBoost `reg:squarederror` / `reg:absoluteerror`, LightGBM `regression` / `mae`, CatBoost `RMSE` / `MAE`; `n_estimators` (CatBoost `iterations`) via `suggest_int(100, 1000)` with `EARLY_STOPPING_ROUNDS = 50`; `N_TRIALS = 50`, `N_WARMUP_TRIALS = 5` -> `TPESampler(seed=SEED, n_startup_trials=5)`; `MedianPruner(n_startup_trials=5, n_warmup_steps=10)`; SQLite storage via `create_study(study_name, direction="maximize")`. Delete the prior DB at run start because `load_if_exists=True` appends trials. (Constants and objective strings are repo-verified; the notes listed them as unrecorded.)
- `build_model(lib_code, params)` reconstructs the estimator; `build_and_evaluate(...)` returns `(model, valid_ic, test_ic)`.
- Per-loss analysis picks each library's best MSE and best MAE config (conditional best, not matched). GPU for XGBoost / CatBoost detected at startup; the deterministic CPU recipe lives in `02_gbm_comparison`.
- fANOVA hyperparameter importance per library via `optuna.importance.get_param_importances(study)` (top 6 parameters each), with `loss_type` included as a parameter: this is where "`loss_type` is the single most important hyperparameter in every study" is established, not in `04`.
- XGBoost / CatBoost search spaces (repo, not notes): XGBoost `max_depth` 2-10, `learning_rate` 0.005-0.3 log, `subsample` / `colsample_bytree` 0.5-1.0, `min_child_weight` 1-300, `reg_alpha` / `reg_lambda` 1e-5-100 log; CatBoost `depth` 2-10, `learning_rate` 0.005-0.3 log, `l2_leaf_reg` 1e-5-100 log, `random_strength` 0-10, `bagging_temperature` 0-1.

### Multi-objective HPO and cross-asset transfer (`06_optuna_multi_asset`)

- Objectives: IC (maximize) and turnover (minimize), with `compute_turnover(predictions)` = mean absolute change in predictions (cost proxy); `multi_objective` returns `(ic, turnover)`; `optuna.create_study(directions=["maximize", "minimize"], sampler=NSGAIISampler(seed=SEED))` (sampler settings repo-verified; `N_TRIALS = 50`); `study.best_trials` is the Pareto set. The single-objective star may be off the front. `compute_turnover` and `make_asset_objective` are defined inline in `06_optuna_multi_asset.py`.
- Transfer: score the best ETF config on crypto / CME futures using shared features; tune per asset with `make_asset_objective(...)`; efficiency = transferred IC / asset-specific IC (ETF row = 100% by construction, tautological).

### Grid vs Optuna at equal budget (`07_hpo_comparison`)

1. `PARAM_GRID` (repo values; the notes did not record them): `n_estimators` [50, 100, 200], `learning_rate` [0.01, 0.05, 0.1], `max_depth` [3, 5, 7], `num_leaves` [15, 31] = 54 grid points; grid budget = `len(ParameterGrid(PARAM_GRID))`.
2. Give Optuna the same trial count on the same categorical space (TPE samples with replacement, so fewer distinct points), then a continuous space (`optuna_continuous_objective`).
3. `TRIAL_BUDGETS = [10, 25, 50, 75]` ladder (repo): fresh study from the same seed per budget; the validation column is a running max (cannot fall); the holdout column is the only informative one. Reports an efficiency curve and val-vs-holdout IC across the budgets.
4. Score only the two selected configs (grid pick, Optuna pick) on the holdout; section 10 refits on train + validation.

### TreeSHAP workflow (`08_shap_analysis`)

- `shap.TreeExplainer(model)` gives exact Shapley values in O(T L D²) (T trees, L leaves, D depth): `shap_values`, `shap_interaction_values`; beeswarm, mean |SHAP|, waterfall, dependence plots (exact calls are inferences).
- Decomposition: f(x) = E[f(X)] + sum_j phi_j^main + sum_{j<k} phi_jk^interaction; diagonal = main effects, off-diagonal = pairwise interactions (regime-conditional signal). Friedman-Popescu H-statistic approximates the fraction of a pair's joint effect due to interaction, in [0, 1]; the "meaningful" level is a convention.
- Consensus: MDI (tree splits; fast, biased to high cardinality), PFI (shuffle; model-agnostic, captures interactions), SHAP (game-theoretic; consistent, local). Keep features that all three agree on.
- Drift: per-feature mean |SHAP| per walk-forward fold; max % change vs the fold-0 baseline; flag > `DRIFT_THRESHOLD = 50` percent (a convention; value from the repo); rank only features whose fold-0 baseline clears `0.05 * max(baseline)` (repo `importance_floor`).
- Selection: rank by mean |SHAP|, retrain on top-k, score cross-sectional IC on the fold's validation window (`selection_df`). Watch for NaN IC from macro-only subsets.
- Figure artifact: `write_figure_12_7_artifact()` and `_build_beeswarm_panel(cs_id, label)` freeze compact SHAP arrays (ETFs + SP500 Options) so formatting changes do not retrain GBMs.

### XAI limits and stakeholder reporting (`09_xai_limitations`, `10_shap_nlp_sentiment`)

- Instability: find pairs of samples with near-identical predictions and compare their top-3 SHAP contributors.
- Rashomon: 3 LightGBM seeds + 1 RF; compare val-IC spread and attribution rankings.
- Token-level NLP SHAP: `predict_proba(texts)` batches FinBERT-tone inference, softmax, reorders to `LABEL_ORDER` = Negative, Neutral, Positive from checkpoint label metadata; `shap.maskers.Text` with `shap.Explainer(predict_proba, masker)` (inference) hides token groups; values are probability points vs the explainer baseline; a controlled perturbation swaps one token (narrowed -> widened).

| Stakeholder | Deliver | Caveat to state |
|---|---|---|
| Risk manager | SHAP summary | correlation is not causation |
| Trader | importance rank | varies with retraining |
| Regulator | model documentation with confidence intervals | - |
| Quant researcher | full SHAP + interactions | check seed stability |

### Conformal intervals for GBM regression (`11_conformal_gbm`)

| Method | Recipe | Needs | When |
|---|---|---|---|
| Split conformal | split proper-train / calibration; fit f_hat; R_i = \|Y_i - f_hat(X_i)\| on calibration; k = min(ceil((n_cal + 1)(1 - alpha)), n_cal); q_hat = R_(k) (exact order statistic via `conformal_order_statistic(scores, alpha)`, not an interpolating quantile); interval f_hat(x) ± q_hat; P(Y in C) >= 1 - alpha under exchangeability | any point model | default wrapper; symmetric widths |
| Quantile regression (QR) | `train_quantile_model(X, y, alpha, n_rounds=200)` for lower/upper quantiles | quantile estimator | narrowest, undercovers, no finite-sample guarantee: do not use alone |
| CQR | s_i = max(q_l(x_i) - y_i, y_i - q_u(x_i)); q_cqr = Q_{1-alpha}(s); C(x) = [q_l(x) - q_cqr, q_u(x) + q_cqr] | quantile estimator | asymmetric intervals; only method hitting target on the purged 2023 ETF fold |
| ACI | `compute_adaptive_intervals(y_pred, y_true, dates, cal_scores, alpha=0.10, gamma=0.01, window=250, label_horizon_steps=21)`: online update of the miscoverage target from a rolling residual buffer; a label enters only after its horizon elapses | any point model | shifting regimes; widest in the ETF test and still missed |

- Every helper in this subsection (`ConformalRegressor`, `LGBWrapper`, `lgb_model_factory`, `deterministic_lgb_parameters`, `train_quantile_model`, `conformal_order_statistic`, `compute_adaptive_intervals`, `fixed_budget_exposure`, `chronological_calibration_masks`, `embargo_steps_from_buffer`, `outer_purge_steps`, `evaluate_conformal`, `run_conformal_splits`, `summarize_conformal_results`, `mean_cross_sectional_ic`, plus `replace_temporal_state` / `build_fold_arrays`) is defined inline in `12_gradient_boosting/11_conformal_gbm.py` and must be copied, not imported; the notebook imports only `cross_sectional_ic_series` and the canonical folds via `utils.modeling.{fold_temporal_frame, load_modeling_dataset, temporal_fold_index}`. The existing `case_studies/utils/conformal.py` is a different API (`walk_forward_conformal_coverage`, `compute_conformal_widths`, `load_conformal_widths`, `coverage_summary`, ...) for the registered case-study pipeline (repo-verified).
- Wrapper: `ConformalRegressor(model_factory, alpha=0.10, cal_frac=0.2)`; `.fit(X, y, dates, label_buffer)`; `.predict(X)` -> `(pred, lower, upper)`. Model factory `lgb_model_factory()` -> `LGBWrapper(n_rounds=200)` on `deterministic_lgb_parameters(objective, alpha=None)`: fixed CPU config (repo) `gbdt`, `num_leaves=31`, `learning_rate=0.05`, `feature_fraction=0.8`, all four seeds = `SEED`, `deterministic=True`, `force_col_wise=True`, `num_threads=NUM_THREADS`, `device_type="cpu"`, plus `alpha` for quantile objectives.
- Constants (repo; the notes did not record them): `CAL_FRACTION = 0.2`, `REFERENCE_INTERVAL_WIDTH = 0.20`, `SEED = 42`, `MAX_SYMBOLS = 0` (all symbols).
- Purged calibration: calibration = final `CAL_FRACTION` of unique training timestamps; `embargo_steps_from_buffer(label_buffer, dates)` converts the horizon to calendar steps; `chronological_calibration_masks(dates, cal_frac, embargo_steps)` removes >= one label horizon before calibration and keeps each panel date wholly on one side. Verify the outer split carries the `setup.yaml` horizon with `outer_purge_steps(unique_dates, train_end, val_start)`.
- Evaluation: `evaluate_conformal(fold_data, label_buffer, alpha=0.10, n_splits=5)`, `run_conformal_splits`, `summarize_conformal_results`; IC via `mean_cross_sectional_ic` on unique timestamp-entity keys (composite key keeps CME product and position).
- Sizing: Relative Exposure = B / Interval Width with a fixed ex-ante `REFERENCE_INTERVAL_WIDTH` budget (`fixed_budget_exposure(widths, budget)`); illustrative, Ch19 integrates into Kelly.

### Cross-case-study insight pipeline (`12_case_study_insights`, Section 12.6)

Prerequisites: each case study's GBM sweep notebook (`case_studies/{cs}/07_gbm.py` in eight case studies, `06_gbm.py` in `us_firm_characteristics`; repo-verified) has populated `case_studies/{cs}/run_log/registry.db` for family `gbm`; the per-CS `08_tabular_dl.py` (`07_tabular_dl.py` in `us_firm_characteristics`; absent in `nasdaq100_microstructure`) built on `case_studies/utils/tabular_dl.py` adds `tabular_dl` (TabM) rows; Ch11's pipeline adds family `linear`. The registry, training dirs and label parquets are generated by the case-study pipeline and are not shipped in the checkout. Per-CS deep dives live in the per-CS `NN_model_analysis.py` (the notes say `13_model_analysis.py`; the number varies by case study, e.g. 13 for etfs, 10 for `us_firm_characteristics`). The notebook trains nothing and writes nothing; every "Computed reading" is derived from the rows just computed. Parameters: `FAMILY = "gbm"`, `BASELINE_FAMILY = "linear"`, `SEED = 42` via `set_global_seeds`.

GBM grid summarised: 4 leaf profiles (7 / 15 / 31 / 63) x 3 regression losses (MSE / MAE / Huber), each scored at 10 checkpoints over 50-500 trees (step 50, inference); direction labels add a binary-logistic variant. Config names look like `leaves_31_huber` or `default_binary`; decode with `parse_gbm_config` (`rsplit("_", 1)`; suffix mse/mae/huber -> regression loss; `_binary` -> `loss="binary"`, `objective_kind="classification"`; `leaves_<n>` -> `leaves=n`; else loss="unknown").

| Step (notebook section) | Helper | Rule |
|---|---|---|
| 1 Coverage | per-CS table | columns `case_study, primary_label, gbm_labels, gbm_configs_primary, tabm_configs, linear_present` |
| Rank-one selection | `select_rank1(metrics, folds, *, expected_fold_ids)` | required metric columns `prediction_hash, training_hash, config_name, ic_mean_daily, ic_n_days, ic_se_hac, ic_ci_lo, ic_ci_hi, ic_t_hac, ic_p_hac, ic_hac_lag`, fold columns `prediction_hash, fold_id, ic` (missing -> `RegistrySelectionError`); eligible = finite metrics, `ic_n_days > 0`, fold ids exactly equal to the declared set with no duplicates, every fold IC finite; comparable = eligible rows at max `ic_n_days`; winner = max `ic_mean_daily`; none -> `RegistrySelectionError("no finite candidate covers every declared fold")` |
| Per-CS collection | `collect_rank1_per_cs(case_studies, family, label_resolver=None)` | label from `PRIMARY_LABELS[cs]`; `resolve_expected_fold_ids(folds, n_folds)`; adds `case_study`, `short_name`, `covers_fold_grid`; a CS with no rows is omitted; `n_folds <= 0` raises; `IncomparableFoldGeometryError` is re-raised with the CS/family/label prefix |
| 2 Forest | `plot_cross_cs_forest(rank1, family=, title=)` | filled marker when \|t_HAC\| > 2 (CI excludes zero), open otherwise; table columns `short_name, label, config_name, trees (=checkpoint_value), ic, ci_lo, ci_hi, t_hac, n_days`; `clear_zero` = `ic_ci_lo > 0 or ic_ci_hi < 0` |
| 3a Loss | polars best-per-group on `grid_regression` | bars width 0.26, asymmetric yerr (ic - ci_lo, ci_hi - ic); mse neutral, mae blue, huber copper; `loss_top_per_cs` counts wins |
| 3b Depth | max `ic_mean_daily` per (short_name, leaves) | diverging colormap `ml4t_diverging()`, symmetric `vmin=-vmax`, vmax = max \|IC\| (fallback 0.05); `depth_spread = max_ic - min_ic`; compare spread to the HAC interval before claiming a depth preference |
| 3c Checkpoints | `collect_gbm_checkpoint_trajectories(rank1)` | reads `get_training_dir(cs, json.loads(spec_json)) / "learning_curves.parquet"` filtered `config == config_name`; mean `ic_mean`/`ic_std` by `iteration`; `argmax_iter` = first iteration at max; `argmax_iter <= 150` = early peak; `== max observed` = budget boundary (consider a larger budget, inference); missing curve -> `RegistrySelectionError` / re-run the sweep |
| 4a Per-fold IC | `collect_fold_ic_per_cs(rank1)` | `n_folds, median, std, pct_positive`; box plot `widths=0.55`; count CSs with `pct_positive > 0.5` |
| 4b Holdout decay | `HOLDOUT_QUERY` (`load_selected_holdout`) | join `prediction_metrics -> prediction_sets -> training_runs` with `split='holdout'`; exactly one valid row, else `RuntimeError("Ambiguous holdout rows ...")`; none -> explicit gap; dumbbell: validation "o" blue, holdout "D" amber; no aggregate decay claim when coverage is sparse |
| 5a GBM vs linear | inner join rank1 frames on `case_study` | `delta = gbm_ic - linear_ic`; rows with `gbm_days != linear_days` go to "Excluded for unequal coverage"; descriptive only |
| 5b By label | `collect_multi_label_per_cs(CASE_STUDY_IDS, family=, labels=callable)` | `regression_labels` = labels starting `fwd_ret_`, not containing `spot`, not in `HORIZON_EXCLUSIONS`; same equal-days filter |
| 6a Horizon | `HORIZON_DAYS` map | plot only CSs with >= 2 mapped horizons; log x-axis; HAC band `fill_between(lo, hi, alpha=0.12)`; sort on (horizon_days, label) |
| 6b Symmetry | `discover_symmetry_pairs(case_studies, family)` -> `(pairs, skipped)` | pairs from `setup.yaml` `labels.classification_eval_label`; both labels need `split='validation'` rows (a pair lacking them is silently skipped: not a defect); direction domain must be a subset of {0, 1}; Direction A = classification score's `ic_mean_daily` vs continuous return (+ native `auc_mean_daily`, `auc_ci_lo/hi`); Direction B = `gbm_direction_b_auc` -> `roc_auc_score(y_dir, y_score)` on predictions joined to the binary label on (`timestamp` cast `pl.Datetime("ms")`, `symbol` cast `pl.Utf8`), both classes required; build the table with the full `SYMMETRY_TABLE_SCHEMA` |
| 7a Rank shift | `load_gbm_feature_importance(cs, training_hash, config_name, top_n=50, num_iteration=checkpoint_value)` vs `_load_linear_importance` | GBM: iterate the per-fold LightGBM booster `.txt` files under `booster_dir(case_dir, training_hash)`, fold id parsed from the stem after `fold`, truncate to `num_iteration`, gain importance normalised by the per-(config, fold) max (`RegistrySelectionError` on any zero fold max), keep `top_features_by_gain`; linear = mean \|coef\| across `fold_*.joblib`, drop <= `ZERO_TOL = 1e-12`; `rank_shift = linear_rank - gbm_rank` (positive = GBM promotion), ties by name; per-CS summary `n_common_features, median_abs_shift, max_gbm_promotion, max_linear_promotion`; show `min(15, n)` features; only `IMPORTANCE_CASES = ["etfs", "sp500_options", "us_firm_characteristics", "us_equities_panel"]`; a rank shift describes how two families use a shared feature library, not why GBM beats or trails linear |
| 7b Rank stability | `feature_rank_stability(cs)` | `top_n=30`, keep top 10 by mean gain, pivot feature x `fold_id`, `drop_nulls`; need >= 2 folds and >= 3 features; `pairwise_rank_correlation` with `rankdata(-values, method="average")` and `np.corrcoef`; returns (mean, min) |
| 7c Three-way | left join linear and `tabular_dl` rank1 | `lin_ic` / `tabm_ic` null unless `ic_n_days == gbm_days`; sum masks per family cell; bars width 0.27 Linear blue, GBM amber, TabM copper |

`HORIZON_DAYS`: `fwd_ret_5m`=5/390, `fwd_ret_15m`=15/390, `fwd_ret_60m`=60/390, `fwd_ret_8h`=1/3, `fwd_ret_24h`=1, `fwd_ret_1d`=1, `fwd_ret_5d`=5, `fwd_ret_10d`=10, `fwd_ret_21d`=21, `fwd_ret_1m`=21, `fwd_ret_3m`=63, `fwd_ret_1m_win`=21, `fwd_ret_risk_adj_5d`=5. `HORIZON_EXCLUSIONS` (provisional registry): NASDAQ-100 GBM `fwd_ret_5m` (two tied rank-one candidates), S&P equity-option linear `fwd_ret_risk_adj_5d` (five ties); primary labels are never excluded. `NON_FEATURE_COLS = {timestamp, symbol, stock_id, product, position, instrument_id}`.

## Guardrails and pitfalls

- **Single-fold ranking** — between-model gaps are smaller than one model's val-to-test move, so an ordering is noise. Print both numbers; make accuracy claims only on multi-fold CV with tuned params; use paired per-fold differences.
- **Validation overfitting in HPO** — one window barely constrains 8 hyperparameters; holdout can flatter or punish at random. Use an averaged walk-forward objective, a budget in the low hundreds (or the 50-100 convention) chosen by walk-forward validation, and score the holdout once. The `07_hpo_comparison` ladder had holdout IC highest at the shortest budget.
- **Degenerate early-stopped model** — a config that stops after one tree gives near-flat predictions whose validation IC rests on tiny rank differences. Inspect the `n_trees` column; treat undefined IC as no value (prune), never as -1 or a partial average.
- **Unscorable folds** — a date with < `IC_MIN_OBS` names or tied predictions has no ranking; scoring at -1 moves a 4-fold mean by a quarter; averaging over scored folds only rewards one easy fold. Filter data-unscorable folds before the search (`fold_is_scorable`); score every trial on the same folds; prune trial-unscorable cases.
- **Label-horizon leakage into validation or holdout** — a 21-day forward label resolves inside the next window. Embargo cutoff at `-(LABEL_HORIZON + 1)`; purged calibration with embargo >= horizon; verify the outer split with `outer_purge_steps` rather than assuming it.
- **Row-offset splits on panels** — a row-offset cut splits one decision date between train and validation. Mask on dates (timestamp groups); sort before conversion.
- **Early-stopping signal touching the test slice** — inflates reported IC. Carve validation chronologically from the train window; keep test strictly out.
- **Native importance comparison** — gain is biased to features with more candidate splits, split counts to continuous features, seeds reorder correlated substitutes, and each library sums to a different total. Rescale to share of own total only for size; use SHAP for cross-model attribution; check MDI/PFI/SHAP consensus.
- **SHAP as truth** — instability (same prediction, different top-3) and Rashomon (seed changes val IC more than model family). Report confidence intervals on importance, check stability across seeds and architectures, name the method, never claim causation.
- **Drift percentages on near-zero baselines** — a tiny absolute shift over a near-zero denominator inflates to spurious drift. Rank drift only among features whose fold-0 mean |SHAP| clears a fraction of the max; `DRIFT_THRESHOLD` is a convention.
- **Macro-only features in cross-sectional models** — features identical across symbols on a date (yield-curve slope / z-score, regime duration, frac-diff QQQ / VNQ) predict identically for every name, so cross-sectional IC is NaN. Ensure cross-sectional signal comes from features that vary across assets; watch for NaN IC in top-k selection.
- **Monotone constraints on noisy features** — the constraint is satisfied by flattening the feature (SHAP collapses to a narrow band), silently dropping it. Compare constrained vs unconstrained SHAP dependence on one vertical scale; score across folds before imposing a directional prior.
- **Ranking-metric self-confirmation** — LambdaMART wins NDCG by construction. Judge by IC on raw returns and by strategy need (top-k vs sized magnitude), across folds.
- **GPU/CPU non-equivalence** — FP32 histogram binning on GPU vs CPU changes split points; parallel summation order is not reproducible. Never treat a GPU rerun as a reproduction; for exact reruns use deterministic CPU settings with a fixed thread count (recipe in `02_gbm_comparison`).
- **LightGBM CUDA availability** — the PyPI wheel is CPU/OpenCL only; under `uv`, `detect_gpu_capabilities()` reports `lightgbm_cuda=False` and GPU rows silently vanish; CUDA LightGBM is FP64-only so consumer GPUs gain little. Run in the `ml4t-gpu` (or `rapids`) image; if the GPU panel does not list all of xgboost / lightgbm / catboost, do not record GPU numbers.
- **RSS-delta memory panel** — misses native/CUDA allocations and allocator/gc timing; most cells read zero. Use a profiler or peak RSS to size hardware.
- **GPU `predict()` for batch inference** — launch overhead; speedups smaller than for training and can be < 1x for small batches. Default to CPU inference at case-study row counts.
- **Optuna SQLite persistence** — `load_if_exists=True` appends trials on rerun, so counts and best configs creep. Delete the prior DB at run start; use PostgreSQL/MySQL for parallel workers.
- **Conditional-best loss comparison** — best-MSE and best-MAE configs differ in depth, lr, regularization and sampler effort, so the gap is "how far the search got under each loss", not an isolated loss effect; a library appears only if both losses completed a trial. Read the printed loss distribution; interpret as the cost of fixing loss by prior.
- **Cross-asset transfer** — ETF-tuned params collapse to ~0 (slightly negative) val IC on CME futures; crypto had only 3 overlapping features (different feature pipeline); an efficiency ratio of two small numbers flips sign when transferred IC crosses zero. Tune per asset class; ignore the tautological ETF row.
- **Turnover-blind HPO** — the last basis points of IC need disproportionate turnover, unprofitable after costs. Run multi-objective (IC, turnover) and pick from the Pareto front by cost tolerance.
- **Wall-time comparisons between search methods** — single uncontrolled runs on a shared machine fitting different configs; do not read as cost benchmarks.
- **Conformal coverage assumed** — exchangeability fails under temporal shift; no asset class hit nominal coverage across purged folds. Always report empirical coverage on time-ordered validation; width is not coverage.
- **QR without conformalization** — narrowest intervals, undercovers, no finite-sample guarantee. Use CQR when asymmetric intervals are wanted.
- **Interpolating quantile for the conformal correction** — can select the next rank and break the finite-sample guarantee. Select the k-th sorted score directly (`conformal_order_statistic`).
- **ACI label maturity** — updating with an unmatured forward label uses unavailable information. Residuals enter the buffer only after `label_horizon_steps`.
- **Position sizing with cross-fold widths** — widths observed in another period leak. Use a fixed ex-ante `REFERENCE_INTERVAL_WIDTH` budget; exposure depends only on each fold's own calibration width.
- **NLP attribution overreach** — token attribution cannot prove leak-free training data, is local to model/masker/input, and a 10-sentence curated sample is not a vocabulary. Run adversarial perturbation checks; keep leakage checks, representative validation and economic testing separate.
- **Intervals on paired differences across overlapping folds** — adjacent overlapping windows are not independent draws; fold spread is not the uncertainty of the mean. Describe mean, spread, sign frequency; do not present as a test.
- **Grid-vs-Optuna instability** — an earlier run on an earlier ETF artifact vintage ranked them in the opposite order. The finding is instability; rely on an untouched holdout plus walk-forward HPO.
- **Comparing IC across unequal evaluation windows (12.6)** — a shorter window can win on IC without a better model. `select_rank1` keeps only candidates at max `ic_n_days`; cross-family joins keep only `gbm_days == other_days`; mismatches go to an "Excluded / Masked for unequal coverage" table, never silently plotted. Never substitute coverage from another label or validation span.
- **Ranking candidates that did not report every declared fold** — partial fold panels are not comparable; the fold-id set must equal `resolve_expected_fold_ids`, no duplicates, all fold ICs finite.
- **Partial-grid points on a shared horizon axis** — a partial-grid point beside a full-grid one reads as one quantity moving with horizon (`covers_fold_grid`, `MAX_FOLDS`); currently only `us_equities_panel/12_dl_weekly` has non-zero `MAX_FOLDS`; if any gbm/linear notebook gets a fold reduction, apply Ch13's retain-and-report split first.
- **Fold summaries as uncertainty** — std of per-fold IC is not an estimator; folds are few. Cite `ic_ci_lo/hi`, `ic_t_hac` from the daily series; show per-fold IC as a stability box plot only.
- **Over-reading the depth heatmap** — at financial SNR the leaf-profile IC range sits within the HAC interval; a flat panel means the depth knob has no resolution, not that depth is irrelevant everywhere. Compare `depth_spread` to the panel's CI before claiming a depth preference.
- **Confidence claims on cross-family deltas** — no registered paired daily-difference estimator exists; label GBM-minus-linear and TabM-vs-GBM deltas "descriptive".
- **Rank shift read as cause** — a feature promoted by the GBM relative to the linear model shows how the two families use the shared feature library; it does not explain the GBM-minus-linear IC delta. Report `n_common_features` and the shift table; do not attribute the performance gap to promoted features.
- **Hand-written literal lists drift** — the Ch12 copy of the symmetry-pair list silently lost `us_firm_characteristics` in the 2026-07-31 chapter-tree restore while still reporting a full count. Discover pairs from `setup.yaml` + registry; print every skipped candidate with its reason in the same sentence as the count.
- **Binary AUC on a ternary direction label** — multi-class AUC is a different metric. Measure the label domain on disk; skip unless it is a subset of {0, 1} (crypto `fwd_dir_8h_3c` is ternary; `fwd_class_1m` is binary despite an old comment).
- **Tied rank-one candidates** — arbitrary resolution changes the figure run to run; drop the cell (`HORIZON_EXCLUSIONS`) rather than break ties.
- **Non-deterministic tie-breaking** — set iteration order, per-process string hashing and double argsort produced different ranks from identical registries. Sort ties by feature name; `rankdata(method="average")`; sort horizon lines on (horizon_days, label); head/tail on (rank_shift, feature).
- **Importance from the full booster instead of the selected checkpoint** — training saves every round; selection picks one. Pass `num_iteration=checkpoint_value` to `load_gbm_feature_importance`.
- **Sparse linear baselines in rank comparison** — zero lasso coefficients fill ranks with name-broken ties; drop mean |coef| <= 1e-12 and report `n_common_features`.
- **Unpickling fold estimators across sklearn versions** — `InconsistentVersionWarning` noise and unsafe prediction; read `coef_` only, never predict; suppress the warning locally in `warnings.catch_warnings()`; a coef/feature-name length mismatch raises `RuntimeError` naming the fold file.
- **CUDA runtime precedence** — `ml4t.diagnostic` dlopens cudart; torch's bundled runtime must win. `import torch` before `ml4t.diagnostic` (same pattern as `case_studies/utils/model_analysis.py`).
- **LightGBM "X does not have valid feature names" warning** — a fit on an array with `eval_set` records synthetic names; filter that one message from `sklearn.utils.validation`, not the whole category.
- **Registry mutation during analysis** — open SQLite read-only: `sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)`.
- **Empty discovery result crashing before skip reasons print** — an empty frame with partial `schema_overrides` has no string columns (`ColumnNotFoundError`). Build with the full `SYMMETRY_TABLE_SCHEMA`; an empty table is a valid answer; raise only when no pair qualified, with the skips in the message.
- **Figures assembled across cells** — Jupyter captures partial figures; build each figure in one cell or helper (`add_loss_bars`, `plot_holdout_decay`).

## Decision rules and defaults

| Decision | Rule / default |
|---|---|
| Library | pick on engineering (latency, memory, categorical handling, deployment), not single-fold IC. LightGBM typically fastest on CPU at moderate threads; GPU helps XGBoost and CatBoost most at large tree counts and ~5M rows, LightGBM least (FP64 CUDA); CatBoost often fastest inference at moderate tree counts; sklearn HistGB has no GPU |
| Loss | MAE/L1 for return regression (NB03's rationale cites 8/9 case studies from the run-time 12.6 count; MAE wins in all three libraries in `05`); make `loss_type` categorical if budget allows |
| Objective | `lambdarank` for top-k selection inside a rebalance; regression when magnitude sizes positions |
| Monotone constraint | only when the direction is known to be right; confirm across folds that SHAP dependence bends rather than flattens |
| HPO method | grid for <= 4 parameters with <= 3 values each; Optuna (TPE) for > 4 parameters or continuous ranges |
| Trial budget | 50-100 trials is the common convention; book suggests starting in the low hundreds; beyond that marginal gain shrinks and validation overfitting grows; notebook `N_TRIALS = 50` |
| LightGBM tuning strategy | fix `learning_rate` low (0.01-0.2 log), set `n_estimators` high, let `early_stopping(50)` choose rounds; let Optuna trade tree structure (`num_leaves` 8-64, `max_depth` 2-8, `min_child_samples` 5-50) against regularization (`reg_alpha`, `reg_lambda` 1e-4-10 log; `subsample`, `colsample_bytree` 0.5-1.0); regularization often has the largest out-of-sample effect in low-SNR regimes |
| Pruning | `MedianPruner(n_startup_trials=5, n_warmup_steps=10)`, IC reported every 20 rounds; noisier objectives prune more |
| Early-stopping patience | 30 rounds (`03`), 50 (`04`, `05`); MLP patience 20 epochs, epoch cap 200 |
| HPO objective | walk-forward (averaged over folds ending before the holdout) is the recommended default |
| Scorability | a date counts only with >= `IC_MIN_OBS = 10` names; a fold with no such date is removed before the search |
| Embargo | `LABEL_HORIZON = 21` trading days; cutoff index `-(LABEL_HORIZON + 1)` |
| Default-vs-tuned sanity | check `n_trees`; a 1-tree model is not evidence either way |
| Holdout discipline | every walk-forward fold ends before the declared holdout; score the holdout once, after selection |
| Model comparison | same folds for every model; report paired per-fold difference, spread, sign frequency; within noise, choose by refit cost (LightGBM is an order of magnitude cheaper than TabM per fold) |
| DL vs GBM | minimal MLP is a floor; TabPFN is a cheap zero-shot probe before tuning (needs token); TabM only when its paired IC advantage keeps sign and refit cost is affordable |
| Multi-objective | when turnover matters, choose from the Pareto front by transaction-cost tolerance; the single-objective best may be dominated |
| Cross-asset | never "tune once, deploy everywhere"; asset-specific tuning when feature distributions differ |
| SHAP workflow | MDI/PFI/SHAP consensus for robust features; interactions compared to adjacent folds, not priors; drift flagged above `DRIFT_THRESHOLD` (50 percent in `08`) only on features whose fold-0 baseline clears 5 percent of the max; validate across seeds and families before pruning or allocating on attributions |
| Conformal | alpha = 0.10, cal_frac = 0.2, n_splits = 5, n_rounds = 200; ACI gamma = 0.01, window = 250, label_horizon_steps = 21; split conformal / ACI wrap any point model, QR / CQR need quantile estimators; prefer CQR when residual regimes shift; always verify empirical coverage |
| Reproducibility | deterministic CPU with fixed thread count for exact reruns; GPU for speed only |
| Threads | `N_JOBS = 8` across libraries for benchmarking |
| 12.6 significance | \|t_HAC\| > 2 (CI excludes zero) -> filled marker; positive-fold majority `pct_positive > 0.5` |
| 12.6 checkpoints | early peak `argmax_iter <= 150`; budget-boundary peak `argmax_iter == max observed` |
| 12.6 cross-family go/no-go | proceed only when `ic_n_days` is equal; else exclude/mask and report |
| 12.6 horizon plot | >= 2 mapped horizons per CS; single-horizon panels get no trend claim |
| 12.6 symmetry pair | declared in `setup.yaml`, both labels have validation predictions, direction domain subset of {0, 1}; native AUC "clears chance" when `auc_ci_lo > 0.5`; Direction B judged by \|pooled OOF AUC - 0.5\| |
| 12.6 importance diagnostics | rank shift `top_n=50`, plot `min(15, n)`; stability `top_n=30` -> top 10, >= 2 folds, >= 3 features; linear drop \|coef\| <= 1e-12; exactly one valid holdout row per config |
| 12.6 sequencing | run each CS's GBM sweep (`07_gbm.py`; `06_gbm.py` in `us_firm_characteristics`) + its `tabular_dl` notebook + the Ch11 linear pipeline -> registry populated -> insight notebook; re-run the boosting sweep if learning curves are missing |

## Code patterns and APIs

- IC: `from ml4t.diagnostic.metrics import cross_sectional_ic_series` (imported by notebooks 02, 03, 04, 06, 07, 08, 09 and 11; `05` and `10` do not use it), wrapped locally as `cross_sectional_ic_mean(y_true, y_pred, dates, symbols)` returning the mean per-date Spearman IC.
- Data: `load_firm_characteristics` (`data/__init__.py`) for the CPZ panel; ETF features at `case_studies/etfs/features/`; multi-asset at `case_studies/{etfs,crypto_perps_funding,cme_futures}/features/` (generated by the case-study pipelines, not shipped); label horizon in each `setup.yaml`; fold state via `replace_temporal_state(mds, temporal, fold_id)` and `build_fold_arrays(mds, split, temporal)` (inline in `11`); presets via `load_gbm_config("light"|"medium"|"heavy")` from `case_studies/utils/gbm.py`.
- Notebook-local helpers (defined inline, copy rather than import): `11_conformal_gbm.py` holds every conformal helper listed above; `04_optuna_tuning.py` holds `ICPruningCallback`, `prepare_fold_data`, `fold_is_scorable`, `widest_cross_section`, `cross_sectional_ic_mean`; `02_gbm_comparison.py` holds `detect_gpu_capabilities`, `get_rss_mb`, `mean_group_ndcg`; `06_optuna_multi_asset.py` holds `compute_turnover`, `make_asset_objective`.
- GPU flags: XGBoost `device="cuda", tree_method="hist"`; LightGBM `device="cuda"`; CatBoost `task_type="GPU"`.

Optuna objective skeleton (`04_optuna_tuning`):
```python
params = {"max_depth": trial.suggest_int("max_depth", 2, 8),
          "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
          "num_leaves": trial.suggest_int("num_leaves", 8, 64),
          "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
          "subsample": trial.suggest_float("subsample", 0.5, 1.0),
          "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
          "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
          "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True)}
callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0),
             ICPruningCallback(trial, X_val, y_val, dates_val, symbols_val, report_every=20)]
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED),
                            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10))
```
- Persistent study: `create_study(study_name, direction="maximize")` with SQLite storage (delete the DB first). Multi-objective: `optuna.create_study(directions=["maximize", "minimize"], sampler=NSGAIISampler(seed=SEED))`, objective returns `(ic, turnover)`, `study.best_trials` = Pareto set. Grid: `sklearn.model_selection.ParameterGrid(PARAM_GRID)`.
- LambdaMART: `lgb.train({"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10], "learning_rate": 0.05, "num_leaves": 31}, train_set_with_group)`; relevance `np.minimum((pct * 5).astype(int), 4)`.
- Monotone: `lgb.LGBMRegressor(..., monotone_constraints=mc)`.
- SHAP: `shap.TreeExplainer(model)`; `shap_values`, `shap_interaction_values`; `shap.Explainer(predict_proba, shap.maskers.Text(...))` for FinBERT (inferred calls).
- Conformal (all inline in `11_conformal_gbm.py`): `ConformalRegressor(lgb_model_factory, alpha=0.10, cal_frac=0.2).fit(X, y, dates, label_buffer="21D").predict(X_new)`; `conformal_order_statistic(scores, alpha)`; `compute_adaptive_intervals(y_pred, y_true, dates, cal_scores, alpha=0.10, gamma=0.01, window=250, label_horizon_steps=21)`; `fixed_budget_exposure(widths, budget)`.

Cross-case-study helpers (`case_studies/utils/insight_chapter.py`): `SYMMETRY_TABLE_SCHEMA`, `select_rank1`, `resolve_expected_fold_ids`, `canonical_fold_ids`, `collect_rank1_per_cs`, `collect_grid_per_cs`, `collect_multi_label_per_cs`, `collect_fold_ic_per_cs`, `collect_gbm_checkpoint_trajectories`, `collect_checkpoint_fold_trajectories`, `discover_symmetry_pairs`, `load_gbm_feature_importance`, `parse_gbm_config`, `plot_cross_cs_forest`, `plot_per_fold_violin`, `plot_rolling_daily_ic`, `plot_multi_label_horizon`, `compare_ic_on_shared_timestamps`, `conformal_coverage_for_selected_prediction`. Also `case_studies.utils.analytics` (`CASE_STUDY_IDS`, `PRIMARY_LABELS`, `SHORT_NAMES`), `case_studies.utils.model_analysis` (`load_metrics_from_registry(cs, families=[...])`, `load_predictions(cs, family=, label=, config_name=, checkpoint_value=, split=)`), `case_studies.utils.registry` (`get_training_dir(case_study, spec)`, defined in `registry/store.py` and exported from the package), `case_studies.utils.booster_paths.booster_dir(case_dir, training_hash)`, `case_studies.utils.gbm_importance.top_features_by_gain(df, top_n)` (the notes placed the last two under `registry`; `insight_chapter.py` imports them from these two modules), `utils.paths.get_case_study_dir(cs)`, `utils.reproducibility.set_global_seeds`, `utils.style` (`COLORS`, `ml4t_diverging()`, `ml4t_palette(n, categorical=True)`, `show_with_alt(fig, alt_text)`).

`SYMMETRY_TABLE_SCHEMA` columns: `short_name, reg_label, dir_label, cls_config, cls_score_ic, cls_score_ic_lo, cls_score_ic_hi, cls_score_ic_t, cls_score_auc, cls_score_auc_lo, cls_score_auc_hi, reg_config, reg_score_auc, n_b, n_b_days` (Ch11 fills `n_b_days`, Ch12 does not).

| Registry object | Columns / path |
|---|---|
| `prediction_metrics` | `prediction_hash, ic_mean_daily, ic_ci_lo, ic_ci_hi, ic_t_hac, ic_n_days, ic_se_hac, ic_p_hac, ic_hac_lag, auc_mean_daily, auc_ci_lo, auc_ci_hi` |
| `prediction_sets` | `prediction_hash, training_hash, split in {validation, holdout}` |
| `training_runs` | `training_hash, family, label, config_name, spec_json` |
| Paths (run-time artifacts generated by the case-study pipeline, not shipped; only `config/setup.yaml` is in the checkout) | `case_studies/{cs}/run_log/registry.db`; `run_log/training/{training_hash}/models/fold_*.joblib` (payload keys `feature_names`, `model`); per-fold LightGBM booster `.txt` files under `booster_dir(case_dir, training_hash)`; `{training_dir}/learning_curves.parquet` (`config, iteration, ic_mean, ic_std`); `labels/{label}.parquet` (`timestamp, symbol, <label>`); `config/setup.yaml` -> `labels.classification_eval_label` |

```python
# best config per (case study, loss)
loss_best = (grid_regression.sort("ic_mean_daily", descending=True, nulls_last=True)
             .unique(subset=["case_study", "loss"], keep="first"))
# coverage mask for a cross-family comparison
lin_ic = pl.when(pl.col("lin_days") == pl.col("gbm_days")).then(pl.col("lin_ic")).otherwise(None)
# checkpoint-truncated gain importance
model = lgb.Booster(model_file=path)
if k < model.num_trees():
    model = lgb.Booster(model_str=model.model_to_string(num_iteration=k))
imp = model.feature_importance(importance_type="gain")
# tie-aware fold rank correlation
ranks = [rankdata(-v, method="average") for v in fold_arrays]
pairs = [np.corrcoef(ranks[i], ranks[j])[0, 1]
         for i in range(len(ranks)) for j in range(i + 1, len(ranks))]
```
```sql
-- holdout lookup for a selected config
SELECT p.prediction_hash, t.training_hash, t.config_name,
       pm.ic_mean_daily, pm.ic_ci_lo, pm.ic_ci_hi, pm.ic_n_days
FROM prediction_metrics pm
JOIN prediction_sets p ON pm.prediction_hash = p.prediction_hash
JOIN training_runs t ON p.training_hash = t.training_hash
WHERE t.family = ? AND t.label = ? AND t.config_name = ? AND p.split = 'holdout'
```
- Horizon mapping: `pl.col("label").replace_strict(HORIZON_DAYS, default=None).cast(pl.Float64)`.

Not recorded in the notes (still open after the repo check): the Gu-Kelly-Xiu-guided RF / GBM hyperparameters in `01` beyond tree counts and depths (repo: RF 300 trees / depth 6; XGBoost 1000-tree ceiling / depth 4 with early stopping; LightGBM 300 / depth 4 / 31 leaves; CatBoost 300 iterations / depth 4); the exact SHAP explainer calls in `08`-`10`; the TabPFN evaluation details in `03`; the HAC lag rule and how `n_folds` is declared in the registry; whether holdout retrains exist for any CS in the current registry; and all numeric results (ICs, times, coverage, best hyperparameters), which print at run time.

## Evidence from the book

Numeric results print at run time; the notes record directions only.

- `01_ensemble_foundations` (CPZ panel, one split): boosted models and the RF baseline are separated by less than the validation-to-test drop; XGBoost early-stopped well below its ceiling; rank IC and R² ordered models differently; importance concentration differs by library.
- `02_gbm_comparison` (ETF single fold, untuned 21-day return regression): every library and preset lands at negative test IC; CPU vs CUDA ICs differ (FP32 binning); LightGBM fastest on CPU; GPU speedup largest for CatBoost and XGBoost at larger tree counts; at 227K rows GPU launch overhead competes with kernel time; at ~5M rows LightGBM CUDA gains least; GPU `predict()` speedup can be < 1x; RSS deltas mostly zero; the `-1` monotone constraint flattened the top feature's SHAP into a narrow band with close test ICs; LambdaMART beat regression on NDCG@10 (by construction) with an inconclusive IC comparison on one fold.
- `03_dl_vs_gbm` (ETF, 8 walk-forward folds): error bars overlap heavily; LightGBM trains in a fraction of TabM's time and stops well short of its tree budget on every fold; validation loss flattens within tens of epochs / hundreds of trees; MSE LightGBM fits normally but has lower test IC, negative on fold 0 where MAE is positive; the minimal MLP is a floor; TabPFN optional.
- `04_optuna_tuning` (ETF): fANOVA concentrated on essentially one parameter (most configs early-stop before penalties or sampling matter); the single-fold search selected a config that early-stops after one tree with near-flat predictions; walk-forward HPO costs many times the single-fold search; tuned-vs-untuned holdout margins are small.
- `05_cross_library_hpo` (CPZ, holdout 2000-2016): `loss_type` is the single most important hyperparameter in every study; MAE wins in all three libraries; the spread of test IC across libraries is smaller than the MSE-to-MAE gap.
- `06_optuna_multi_asset`: Pareto-front curvature shows the last few bps of IC need disproportionate turnover; ETF-tuned params give near-zero / slightly negative val IC on CME futures vs clearly positive asset-specific IC; crypto excluded (3 overlapping features).
- `07_hpo_comparison` (ETF): Optuna cannot beat the exhaustive grid on the same categorical space (can only tie); holdout IC is already highest at the shortest budget in the ladder; an earlier artifact vintage reversed the grid-vs-Optuna order.
- `08_shap_analysis` (ETF fold): MDI/PFI/SHAP agreement is partial; every SHAP-ranked top-k subset has validation IC near zero and non-monotonic in k; the top-5 subset (yield-curve slope and z-score, regime duration, frac-diff QQQ / VNQ) gives NaN IC because it is macro-only.
- `09_xai_limitations`: three LightGBM seeds span a wider val-IC range than separates any of them from the RF; top-3 contributors differ completely for near-identical predictions.
- `10_shap_nlp_sentiment`: the narrowed -> widened swap shows locally plausible attributions coexisting with counterintuitive nearby predictions.
- `11_conformal_gbm`: across purged walk-forward folds no asset class hits nominal coverage (one overshoots, others undershoot); the asset with the best coverage is not the one with the highest IC; on the purged 2023 ETF fold split conformal undercovers most, QR is narrowest and undercovers, CQR is the only method reaching target, ACI is widest and still misses.
- `12_case_study_insights` (shape of the computed findings; the notebook carries no fixed IC values and every count is recomputed from the registry at run time): count of CSs whose GBM HAC interval excludes zero; loss-function win counts across regression-primary CSs (the "MAE tops 8 of 9" figure is `03_dl_vs_gbm`'s stated rationale citing this cell, not a value fixed here); the CS with the widest leaf-profile spread; how many of nine panels peak by 150 trees vs at the budget boundary; positive-fold-majority count; holdout coverage count (no aggregate decay claim when sparse); GBM-higher count among matched-coverage pairs; Direction A / native AUC / Direction B counts; rank-shift and stability ranges; TabM missing / masked / above-GBM lists. Expected qualitative pattern from the section text: GBM promotes interaction / regime features, linear promotes monotonic predictors; depth often shows little resolution at financial SNR. Book Table 12.4 reports per symmetry pair the classification score IC, the classifier's own AUC and the regression score's AUC (the native AUC column was unreadable by the selector until 2026-09-18). Known fixed facts: nine case studies; `IMPORTANCE_CASES` has four entries (the only CSs with both saved boosters and linear fold models); crypto declares `fwd_dir_8h` and ternary `fwd_dir_8h_3c` onto `fwd_ret_8h`; `us_firm_characteristics` pairs (`fwd_ret_1m`, `fwd_class_1m`) with binary domain.

## Related references

- `chapters/11_ml_pipeline.md` — modeling setup, walk-forward folds, SHAP introduction and dependence plots (11.4), the linear baseline family and its symmetry table (`n_b_days` column).
- `chapters/07_defining_the_learning_task.md` — forward-return and direction labels, label horizons in `setup.yaml`, classification vs regression evaluation.
- `chapters/08_financial_features.md` — the feature library the GBMs consume and the conformal inputs; macro vs cross-sectional features.
- `chapters/10_text_feature_engineering.md` — FinBERT-tone sentiment features explained with token-level SHAP.
- `chapters/13_dl_time_series.md` — temporal deep-learning sibling; retain-and-report split on `covers_fold_grid`; `us_equities_panel/12_dl_weekly` `MAX_FOLDS`.
- `chapters/14_latent_factors.md` — IPCA, RP-PCA, CAE, SDF-GAN on the same CPZ benchmark.
- `chapters/19_risk_management.md` — position sizing with uncertainty; Kelly with conformal interval widths.
- `chapters/20_strategy_synthesis.md` — cross-model synthesis using GBM baselines.
- `chapters/26_mlops_governance.md` — production monitoring of SHAP drift (the notes cite "Ch23 production monitoring"; in this index monitoring lives here, inference).
- `chapters/18_transaction_costs.md` — turnover as a cost proxy in multi-objective HPO (`06`); inferred link, not a cross-reference the chapter notes make.
- `case_studies/etfs.md` — primary panel for notebooks 02-04, 06-09, 11.
- `case_studies/us_firm_characteristics.md` — Chen-Pelger-Zhu firm-characteristics panel (01, 05); binary `fwd_class_1m` symmetry pair.
- `case_studies/us_equities_panel.md` — ~5M-row scale benchmark; `12_dl_weekly` fold reduction.
- `case_studies/sp500_options.md` — Figure 12.7 SHAP beeswarm artifact; `IMPORTANCE_CASES` member.
- `case_studies/crypto_perps_funding.md`, `case_studies/cme_futures.md` — conformal coverage by asset class; cross-asset transfer targets; ternary `fwd_dir_8h_3c`.
- `case_studies/nasdaq100_microstructure.md`, `case_studies/sp500_equity_option_analytics.md` — `HORIZON_EXCLUSIONS` tie cases.
- `libraries/ml4t_diagnostic.md` — `cross_sectional_ic_series`, HAC IC inference; import order with torch.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `workflow.md`, `companion_repo.md` — cross-cutting indexes.
- Further reading: Friedman 2001 (GBM); Breiman 2001 (RF; Two Cultures); Chen & Guestrin 2016 (XGBoost); Ke et al. 2017 (LightGBM); Prokhorenkova et al. 2019 (CatBoost); Lundberg & Lee 2017, Lundberg et al. 2020 (SHAP / TreeSHAP); Friedman & Popescu 2008 (H-statistic); Hollmann et al. 2025 (TabPFN); Grinsztajn et al. 2025 (Scaling TabPFN); Gorishniy et al. 2024 (TabR), 2025 (TabM); Holzmüller et al. 2024 (Better by Default); Rubachev et al. 2024 (TabReD); Erickson et al. 2020 (AutoGluon), 2025 (TabArena); Ye et al. 2024 (TALENT); O'Donovan & Yu 2024 (option anomalies); Deb et al. 2002 (NSGA-II); Akiba et al. 2019 (Optuna); Bergstra & Bengio 2012 (random search); Vovk et al. 2005, Lei et al. 2018 (conformal); Gu, Kelly & Xiu 2020 (hyperparameter guidance); Chen, Pelger & Zhu 2020 (panel).

## Glossary

| Term | Meaning |
|---|---|
| Bagging | averaging independently trained trees (RF); reduces variance, not bias |
| Boosting | sequential fitting of trees to residual errors; reduces bias |
| Oblivious (symmetric) tree | CatBoost tree where all nodes at a depth share one split; bitwise inference |
| Leaf-wise growth | LightGBM grows the leaf with largest loss reduction; `num_leaves` controls complexity |
| Histogram binning | bucketing feature values before split search; GPU typically FP32, LightGBM CUDA FP64 |
| Rank IC / cross-sectional IC | Spearman correlation between predictions and realized returns within a date, averaged over dates |
| HAC interval | heteroskedasticity-and-autocorrelation-consistent CI on the daily IC series (`ic_se_hac`, `ic_hac_lag`) |
| NDCG@k | normalized discounted cumulative gain over the top-k of a ranked query group |
| LambdaMART | gradient-boosted learning-to-rank with `lambdarank` objective and query groups |
| Monotone constraint | forces a feature's effect to be non-decreasing (+1) or non-increasing (-1) |
| TPE | Tree-structured Parzen Estimator; models densities of good vs poor hyperparameters |
| MedianPruner | stops a trial whose intermediate value is below the median of prior trials at that step |
| Define-by-run | Optuna API where the search space is built inside the objective via `suggest_*` |
| fANOVA | functional ANOVA decomposing objective variance across hyperparameters |
| NSGA-II / Pareto front | elitist multi-objective genetic algorithm / non-dominated solutions where improving one objective worsens another |
| Validation overfitting | selecting a configuration that fits the validation window's noise |
| Walk-forward HPO | objective averaged over temporal folds, each ending before the holdout |
| Embargo / purge | removing observations within one label horizon of a split boundary |
| TreeSHAP | exact Shapley values for tree ensembles in O(T L D²) |
| SHAP interaction values / H-statistic | decomposition into main effects and pairwise interactions / Friedman-Popescu interaction strength in [0, 1] |
| MDI / PFI | mean decrease in impurity / permutation feature importance |
| Rashomon effect | many models with similar performance but different attributions |
| Explanation instability | near-identical predictions with different SHAP profiles |
| Drift (SHAP) | change in per-feature mean \|SHAP\| across walk-forward folds |
| TabM | MLP with M rank-one adapters trained jointly as an ensemble (registry family `tabular_dl`) |
| TabPFN | prior-data fitted network; zero-shot tabular foundation model (gated weights) |
| Split conformal / CQR / ACI | calibrated absolute residuals for finite-sample intervals / conformalized quantile regression (asymmetric) / adaptive conformal inference (online miscoverage update) |
| Exchangeability | distributional symmetry assumption behind the conformal guarantee |
| Coverage | empirical fraction of outcomes inside predicted intervals vs nominal 1 - alpha |
| Turnover (proxy) | mean absolute change in predictions between periods |
| Rank-one configuration | max-IC candidate among those covering all declared folds and the maximum day count |
| Coverage (`ic_n_days`) | number of scored days in a prediction set; must match for any cross-family comparison |
| `covers_fold_grid` | flag that a selection's folds span the CS's full modelling grid |
| Checkpoint | boosting iteration (tree count) at which a config's predictions were scored |
| Validation-to-holdout decay | validation-fold IC minus nested-holdout IC of the selected config |
| Direction A / Direction B | classification score evaluated as IC against the continuous return / regression score evaluated as AUC against the binary direction label |
| Native AUC | classifier's AUC against its own direction label (`auc_mean_daily`) |
| Rank shift | `linear_rank - gbm_rank` on the common feature set; positive = GBM promotion |
| Gain importance | LightGBM split-gain importance, normalised per fold by its max (`importance_norm`) |
| Primary label / short name | per-CS headline target from `PRIMARY_LABELS` / display alias from `SHORT_NAMES` |
