# Case study: S&P 500 equity + option analytics (634 stocks, daily, fwd_ret_5d)

> Trades S&P 500 *equities* (never options) on information read off their listed options: IV level, skew, term structure and the variance risk premium (VRP), joined to price momentum and realized vol, 2017-2021, weekly Friday-close decisions filled at Monday open. The question is not "do options contain information" but whether it survives point-in-time feature engineering (one-session option lag, pinned GARCH windows), weekly portfolio construction, 13 bps round-trip costs, risk overlays and the 2020 regime change. The headline lesson is that option features are forecasts of a distribution's *width*, not its *mean*: every family ranks the risk-adjusted label (`fwd_ret_risk_adj_5d`) above the raw return, and on the primary label `fwd_ret_5d` no family clears zero. The second lesson is procedural: every checkpoint of every model on every declared label is registered, IC decides nothing, and selection happens exactly once, on validation backtest Sharpe after costs in `14_backtest`. Report a weak result as a result; a screen that only ever finds signal is not a screen.

Repo path for every notebook below: `case_studies/sp500_equity_option_analytics/{notebook}.py` (`.ipynb` siblings). Config: `case_studies/sp500_equity_option_analytics/config/setup.yaml` plus `config/training/{label}.yaml` menus and `config/backtest/base.yaml`.

## Setup contract

| Item | Value (from `config/setup.yaml` unless noted) |
|---|---|
| Identity | `strategy_id: sp500_equity_option_analytics`, `setup_version: v1` |
| Universe | `universe.n_assets: 633`, `eligibility_rule: sp500_with_options`; current-constituent roster (survivorship bias acknowledged everywhere). Counts differ by source: README/config 633, unit title 634, `04_model_based_features` reports 624 securities actually carrying a surface. |
| Data | `ML4T_DATA_PATH` must hold `equities/market/sp500/daily_bars.parquet` (AlgoSeek, ships in repo at `data/equities/market/sp500/daily_bars.parquet`; cite algoseek.com) and `equities/market/sp500/options_surface_daily.parquet` (materialized by `data/equities/market/sp500/materialize_options.py`). Missing data raises with acquisition instructions. |
| History | 2017-01-03 to 2021-12-31, 1,259 NYSE sessions; fold 0 training opens 2018-01-04 (~250 sessions of run-up precede it) |
| Decision cadence | `decision.cadence: weekly_friday_close`, `snapshot: friday_16:00_et`, `execution_delay: monday_open`, `iv_feature_lag: 1_day`; `cadence_by_label: {fwd_ret_10d: biweekly, fwd_dir_10d: biweekly}` (a 10d position on a weekly grid would overlap the quantity being measured) |
| Labels | `labels.primary: fwd_ret_5d`; variants `fwd_ret_10d`, `fwd_ret_risk_adj_5d`, `fwd_dir_5d`, `fwd_dir_10d`; `horizons` 5D/10D/5D/5D/10D; `buffer: 10D` (primary train-to-validation, deliberately conservative); `variant_buffers` 10D/5D/5D/10D; `rebalance_step: 1` for all five (`ceil(horizon / cadence)` read against the per-label cadence); `classification_eval_label: {fwd_dir_5d: fwd_ret_5d, fwd_dir_10d: fwd_ret_10d}` |
| Walk-forward | `evaluation.n_splits: 2`, `train_size: 2Y`, `val_size: 1Y`, expanding window; validation windows 2019 and 2020; 10-session embargo (README); `holdout_start: 2021-01-01`, `holdout_end: 2021-12-31`; `calendar: NYSE`, `periods_per_year: 252` |
| Cost model | `costs.class: material`, `model: percentage` (bps regime = production headline), `components: [spread, commission, market_impact]`, `per_leg_cost_bps_range: [3, 10]`, `round_trip_cost_bps: 13` (6.5 x 2 legs), `per_share: 0.0035` (IBKR Pro Tiered top tier; exploratory regime only) |
| Execution engine | `execution.initial_cash: 1_000_000`, `share_type: integer`, `allocator_lookback: 63` bars; all three are spec-hash inputs, read via `get_backtest_config()` |
| Strategy mapping | `mapping.class: long_only_rank_and_rebalance`, `position_state_space: long_only`, `entry_logic: rank_by_iv_signal`, `sizing: equal_weight` |
| Benchmark | Full-universe 1/N equal weight with `backtest.rebalance.benchmark` thresholds 0/0; strategy profile `default: {min_weight_change: 0.005, min_trade_value: 100.0}` (skip only when both are below threshold) |
| Selection funnel | `backtest.sweep.top_n_predictions: {signal: 0, allocation: 10, cost_sensitivity: 1, risk_overlay: 1}`; `top_k_grid: [5, 10, 20]` for every label; `expensive_allocators_skip: false` |
| Allocators | `score_weighted`, `inverse_vol`, `risk_parity`, `mvo_ledoit_wolf` (`lookback: 126`), `hrp`, `conformal_weighted`; no `max_weight` cap; equal weight is the baseline, never an allocator |
| Cost grids | `cost_grid_bps: [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]` (11); `cost_grid_half_spread_usd: [0.0, 0.005, 0.01, 0.025, 0.05, 0.10]` (6); together the "17-point cost surface" (union, inference) |
| Risk controls | `risk_controls.position`: `stop_loss` 0.03/0.05/0.10/0.15; `trailing_stop` 0.01/0.02/0.03/0.05/0.10/0.15/0.20; `time_exit` bars 10/20/40 (14 predeclared) |
| GBM | `modeling.gbm: {libraries: [lightgbm], preset: default, device: cpu, max_bin: 255}` |
| Latent factors | `persistent_entities: true`, `device: cuda` (11a/11b override to cpu); `ipca: {max_iter: 10000, factor_ridge: 0.01, gamma_ridge: 0.01}`; `sdf: {checkpoint_epochs: [256, 512, 768, 1024], beta_checkpoint_epochs: [256], beta_default_checkpoint: 256}`; `sae: {batch_size: 10000}` |
| Model-based features | `model_based.gjr_garch: {burnin: 252, refit_every: 21, mean: Constant, vol: GARCH, p: 1, o: 1, q: 1, dist: Normal}`; schedule-bounded, carries no fold column |
| Causal | `causal: {treatment: ivrv_spread, treatment_window: 20, confounders: [rv_20, mom_21d, skew_rr_30_25d], method: walk_forward_dml}` |
| Holdout status | 2021 holdout was observed on an earlier IPCA lineage that the v3.1 rebuild superseded; notebooks report validation evidence and leave OOS efficacy unresolved |

### Running

1. `uv sync --frozen`; set `ML4T_DATA_PATH`.
2. Run in order: `01_feasibility_analysis 02_labels 03_financial_features 04_model_based_features` → `05_evaluation 06_linear 07_gbm 08_tabular_dl 09_dl_lstm 10_dl_patchtst` → `11a_pca 11b_ipca 11c_conditional_autoencoder 11d_stochastic_discount_factor 11e_supervised_autoencoder 11_latent_factors` → `12_causal_dml 13_model_analysis 14_backtest 15_portfolio_management 16_risk_management 17_costs` → `18_holdout_predictions 19_holdout_backtest 20_strategy_analysis`.
3. Several-hour workload on CUDA; CPU works but materially slower. Completed hashes are reused (cache hits reported).
4. Source of truth: `run_log/registry.db` + content-addressed `run_log/training/`, `run_log/predictions/`, `run_log/backtest/`. Legacy `results/*.json` unused. README restates no numbers.
5. Bundles: `uv run python scripts/download_artifacts.py --cs sp500_equity_option_analytics` fetches `v3.1.0-artifacts` (current); `v3.0.0-artifacts` is the first generation; hashes were re-keyed and do not resolve across bundles.

### Training menus (`config/training/{label}.yaml`) and notebook constants

| Label | `linear` | `gbm` | `deep_learning` | `tabular_dl` | `latent_factors` | `causal_dml` |
|---|---|---|---|---|---|---|
| `fwd_ret_5d` | 29 (1 OLS + 12 ridge + 8 lasso + 8 enet) | 15 | `nlinear`, `lstm_h64`, `patchtst` | `tabm_s/m/l` | `pca, ipca, cae, sdf, sae` | `dml` |
| `fwd_ret_10d`, `fwd_ret_risk_adj_5d` | 29 | 15 | `nlinear`, `lstm_h64` | `tabm_s/m/l` | all five | - |
| `fwd_dir_5d`, `fwd_dir_10d` | 10 logistic | 5 `*_binary` | - | - | - | - |

Menu names resolve to presets in `case_studies/config/{model_type}/`; latent-factor presets live library-side (`config/pca/pca.yaml`, `config/ipca/ipca.yaml`, `config/cae/cae.yaml`). `LABELS` narrows a run (diagnostic, not canonical). Direction labels are secondary variants, not a full sweep.

| Notebook | Constants and grids |
|---|---|
| `02_labels` | `RV_WINDOW = 20`; hole rule `ONE_SESSION = pl.col("session").diff().over("sec_id") == 1` |
| `03_financial_features` | `IV_LAG = 1`, `DECISION_CYCLE = 5`, `LEADING_BUDGET = 20 + 1 = 21`, `WARMUP_END` at index of max lookback (252), `ZERO_FLOOR = 0.001`, `PRICE_FLOOR = 1e-8`, redundancy `CUT = 0.7`; Garman-Klass coefficient `2·ln2 − 1`, annualize by `sqrt(252)` |
| `04_model_based_features` | `MIN_OBS = 252` (same value as `burnin`; whether one gate or two is not explicit), `MAX_SYMBOLS = None`, `MIN_CROSS_SECTION = min(10, n_symbols)`, `MIN_IC_DATES = 10`; `GARCH_KW = {k: schedule[k] for k in ("mean","vol","p","o","q","dist")}`; outputs `garch_cond_vol`, `garch_ivrv_spread`, `garch_vol_surprise` |
| `05_evaluation` | `HAC_MAXLAGS = 5` (label horizon), `COVERAGE_MIN = 0.70`, `STALENESS_MAX = 0.50`, `FDR_ALPHA = 0.05`, `SIGN_CONSISTENCY_MIN = 0.60`, `IC_THRESHOLD = 0.005`, `REDUNDANCY_CUT = 0.70`, `N_QUANTILES = 5`, `MIN_CROSS_SECTION_DEFAULT = 20` (capped by loaded universe), `MIN_SESSIONS_FOR_INFERENCE = 20`, `IC_ROLLING_WINDOW = 63`, `MAX_ABS_LABEL_RETURN = 3.0`, `RANKED_FEATURES = 25`, `RANKED_PAIRS = 20`; these are the screen's own judgment, not config |
| `06_linear` | ridge α ∈ {0.001 … 1e7} (12 values); lasso/enet `f` ∈ {0.015, 0.03, 0.08, 0.2, 0.35, 0.5, 0.7, 0.85}; logistic none / L2 C ∈ {0.001, 0.01, 0.1, 1, 10, 100} / L1 C ∈ {0.001, 0.01, 0.1}; one prediction set per config; `EXECUTION_TIER = "canonical"`, `POPULATION_NAME = ""` |
| `07_gbm` | `num_leaves` ∈ {default, 7, 15, 31, 63} x loss ∈ {mse, mae, huber} = 15 regression configs (5 `*_binary` for direction); fixed learning rate; 10 checkpoints per run → 150 candidates per regression label, 50 per direction label; `checkpoint_dominates` diagnostic |
| `08_tabular_dl` | `tabm_s/m/l` step width and `n_members` together across a factor of four at fixed dropout, lr, batch; 200 epochs, checkpoint every 25 → 8 models per config → 24 candidates per label; `PUBLISHED_DEVICE = "cuda"` |
| `09`/`10` sequence | `lookback` = 60 trading days (a stock needs 60 prior observations); `patchtst`: `patch_size`, `d_model`, `n_heads`, `n_layers`, 100 epochs / `checkpoint_interval` 5 → 20 sets; 09 assumed to share the schedule (inference); LSTM hidden 64 (inference from name) |
| `11a`-`11e` | PCA/IPCA `checkpoint_interval: 0` → one set at checkpoint 0 (never invent intermediate checkpoints); CAE `n_epochs: 50`, `checkpoint_interval: 5`, 5 factors, `(32,)`, lr 1e-3; SAE bottleneck 96, `(896, 448, 448, 256)`, lr 1e-4, loss weight = library default; SDF three budgets, conditional epoch = unconditional budget + itself |
| `12_causal_dml` | `N_PLACEBO`, `CV_FOLDS`, `MAX_SAMPLES` (0 = resolve from request); `SUPERSEDES_CAUSAL = "18f777683c40"`; 100 placebo refits in production; register only after finite effect + HAC SE |
| `14_backtest` | `TOP_K = 0` (= smallest of `top_k_grid`), `SPLIT = "validation"`, `FORCE_REBACKTEST`, `TOP_N_PREDICTIONS`, `MAX_SYMBOLS`, `SEED = 42`, `set_global_seeds(SEED)`; `PERSISTENCE_WEEKS = 8` and `PRICE_CUTS = [50, 100, 250]` in feasibility/cost diagnostics |

## Pipeline stages

| Stage | Notebook | Chapter | What it does | What it writes |
|---|---|---|---|---|
| Feasibility | `01_feasibility_analysis` | 6 | Options coverage, universe count on decision dates, ordering persistence (`PERSISTENCE_WEEKS = 8`), weekly cadence choice, cost headroom vs typical open-to-open move | Nothing |
| Labels | `02_labels` | 7 | Rebuild tradable prices via `adj_factor` inside `sec_id`; 5d/10d forward return (Friday close → Monday open entry), risk-adjusted (`RV_WINDOW = 20`), direction labels with null guard; fold boundaries; effective-sample-size diagnostics | `labels/{fwd_ret_5d,fwd_ret_10d,fwd_ret_risk_adj_5d,fwd_dir_5d,fwd_dir_10d}.parquet` + `.digest.json` |
| Financial features | `03_financial_features` | 8 | Reindex sparse surface onto each security's session grid, lag option columns by `IV_LAG = 1`, forward-fill 5 sessions, compute eight declared families, within-date ranks, seal test | `features/financial.parquet` |
| Model-based features | `04_model_based_features` | 9 | Forward-only GJR-GARCH(1,1), burn-in 252, refit 21, pinned single-start vintage; paired VRP-denominator test | `features/model_based.parquet` |
| Evaluation | `05_evaluation` | 7-9 | Coverage, staleness, daily Spearman IC, HAC + bootstrap bands, BH-FDR, Table 7.2 triage | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` |
| Linear | `06_linear` | 11 | OLS/ridge/lasso/enet + logistic baselines across the label panel | Runs + prediction sets in `run_log/registry.db`; coefficients `run_log/training/{hash}/`; scores `run_log/predictions/{hash}/` |
| GBM | `07_gbm` | 12 | LightGBM 15-config grid, 10 checkpoints each, fold-complete validation predictions | Boosters, `learning_curves.parquet`, `fold_metrics.parquet` under `run_log/training/{hash}/` |
| Tabular DL | `08_tabular_dl` | 12 | TabM `tabm_s/m/l` ensembles on combined equity + option panel; 200 epochs, checkpoint every 25 | Checkpoints under `run_log/training/tabular_dl/` |
| LSTM | `09_dl_lstm` | 13 | `nlinear` control and `lstm_h64` on 60-day windows of surface + price | Checkpoints under `run_log/training/deep_learning/` |
| PatchTST | `10_dl_patchtst` | 13 | `patchtst` (primary label only), 100 epochs / every 5 = 20 prediction sets | Checkpoints under `run_log/training/deep_learning/` |
| Latent-factor index | `11_latent_factors` | 14 | Checks sibling `MODEL_NAME` union covers every `latent_factors:` menu entry; side-by-side peak validation IC | Nothing (reads registry) |
| PCA | `11a_pca` | 14 | Unconditioned factors from the return panel; `checkpoint_interval: 0`; balanced panel | One run + one prediction set per label |
| IPCA | `11b_ipca` | 14 | Exposures linear in option-surface characteristics, ALS; ragged panel | One run + one prediction set per label |
| Conditional AE | `11c_conditional_autoencoder` | 14 | IPCA with network map; 5 factors, hidden `(32,)`, lr 1e-3, 50 epochs / every 5 | 10 prediction sets per label |
| SDF | `11d_stochastic_discount_factor` | 14 | Adversarial pricing kernel (Chen, Pelger, Zhu 2021); only member hashing a macro panel | One run + prediction set per label and checkpoint |
| Supervised AE | `11e_supervised_autoencoder` | 14 | CAE + forward-return loss; bottleneck 96, encoder `(896, 448, 448, 256)`, lr 1e-4, 50/5 | 10 prediction sets per label |
| Causal DML | `12_causal_dml` | 15 | `ivrv_spread` → `fwd_ret_5d` with walk-forward DML, Driscoll-Kraay SE, naive OLS, block-permutation null (100 placebo refits) | One row in registry `causal_runs` |
| Model analysis | `13_model_analysis` | 11-15 | Full-coverage families by daily IC with HAC intervals; pairwise rank correlations; deciles; conformal coverage; count of CI-credible label-family pairs | Nothing (reads registry via `Study.at`) |
| Backtest | `14_backtest` | 16 | Equal-weight top-k ∈ {5,10,20} per registered prediction set; coverage filter; bootstrap Sharpe; effective-rank DSR; two-fold PBO; **the only selection step** | `daily_returns.parquet`, `weights.parquet`, `trades.parquet`, `fills.parquet`, `equity.parquet`, `portfolio_state.parquet`, `spec.json` under `run_log/backtest/{hash}/` |
| Allocation | `15_portfolio_management` | 17 | Six allocators on the top-10 EW configs across `top_k_grid` | One backtest run per allocator, same layout |
| Risk | `16_risk_management` | 19 | 14 predeclared overlays on top-1 lineage per label; paired block-bootstrap Sharpe difference vs parent; freezes candidate field | One run per overlay, same layout |
| Costs | `17_costs` | 18 | Replays best risk-stage config over the 17-point cost surface with block-bootstrap bands | One run per cost level, same layout |
| Holdout predictions | `18_holdout_predictions` | 20 | Refit selected config on all pre-2021 history; publish only the selected checkpoint | One training run + one prediction set |
| Holdout backtest | `19_holdout_backtest` | 20 | Selected spec cloned, re-pointed, run unchanged once | One run at `stage='holdout'` |
| Strategy assessment | `20_strategy_analysis` | 20 | Funnel reconstruction; which claims the holdout supports; counts holdout fits per window; no gate | Nothing |

Downstream consumer: `20_strategy_synthesis/02_feature_evaluation.py` reads the nine case studies' `triage_ledger.parquet`.

## Design decisions and why

| Decision | Why |
|---|---|
| Option-derived families carry `lag: 1`; equity-price families `lag: 0` | A surface summary is stamped at the close it summarizes and cannot be read before it; the equity close *is* the decision snapshot |
| Friday-close decision, Monday-open fill | Shares have an opening auction; a Friday-close feature must never receive a Friday-close fill (same-bar fill) |
| Weekly cadence (biweekly for 10d labels) | 01 §B.4: most cross-sectional ordering survives one week (daily would re-pay spread for an unchanged cross-section) but decays within a few weeks (monthly acts on a stale ranking) |
| Risk-adjusted label expected to rank best | IV, skew, term structure, VRP are forecasts of width, asymmetry, horizon and price of width, none of the mean; prediction: mean/width label beats raw return, and 05/06/07/08 all land there |
| Within-date percentile ranks (`features.ranked`) | A rank strategy acts on the ordering, not the level; a rank pooled across dates answers a different question with the same arithmetic |
| Persistence measured between two cross-sectional orderings k dates apart | Autocorrelation inside one stock's history answers only for the always-quoted minority and discards the >2/3 of names absent on narrow dates |
| GARCH schedule in `setup.yaml`, not the notebook | A fitted feature's estimation window is part of its definition; a rule cannot leak by being fitted, a parameter can; fix the window before any model runs and assert it |
| `burnin: 252` not 504 | 504 drops coverage 76.5% → 54.7%, non-emitting securities 57 → 103, and eats 159,463 fold-0 training sessions (592 securities) vs 22,130 (121); 252 is paid out of the pre-fold run-up |
| `o: 1` (GJR, not GARCH) | Negative returns raise next-session variance more; on single-name equity under an option book that asymmetry is what quotes are priced around; `sp500_options` declares the same |
| `garch_ivrv_spread` replaces `rv_20` memory with a forecast | Compare a forecast against a forecast; then test the *paired difference* of daily ICs directly rather than inferring from two separate tests |
| Every checkpoint is a prediction set; selection only in `14_backtest` on validation backtest Sharpe | A checkpoint is part of a configuration; choosing best epoch after seeing curves is selection on the answer; IC can lead and lose after costs and turnover |
| Every family fitted on every declared label | Evidence can live on a non-traded label; label-routed allocation exists only because nothing selected a single label upstream |
| Declared, not searched, hyperparameters (factor counts, ridge, patch size, SDF budgets) | A search over validation IC is what the notebooks are arranged to avoid |
| Controls that do a job: `nlinear` for LSTM, PCA for conditioned factor models | Any gain is worth only its distance from the control; architectures with memory frequently fail to beat `nlinear` |
| Long-only equal weight baseline | Keeps borrow/locate out of a signal-focused example; avoids stacking a second optimization on the one under test |
| `initial_cash: 1_000_000`, `share_type: integer` | At $100k, top_k=20 → $5k/name x integer rounding x high-priced S&P tail → min 27 trades over 4y; engine almost never traded (`memory/feedback_2026_05_15_equity_sizing_invalidated.md`) |
| `mvo_ledoit_wolf` lookback 126 (vs 63) | With N/K < ~2.5 at top_k=20 Ledoit-Wolf shrinkage degenerates to the identity target |
| No `max_weight` cap | Prior 0.40 was already a loosened 0.20 workaround; restoring it pushes moment allocators back toward equal weight (`memory/feedback_max_weight_caps_intentionally_absent.md`) |
| `conformal_weighted` added | Was absent while `etfs`, `cme_futures`, `fx_pairs` declared it; `13_model_analysis` measures the coverage its widths come from, so running it tests whether calibration is good enough to size with |
| `max_bin: 255` on CPU | 63 is the GPU default carried over by mistake; it quartered bin resolution for every CPU fit |
| `sae.batch_size: 10000` | `SAEConfig.batch_size=None` = one batch of ~250,000 rows, which does not fit a 24 GB card |
| `causal.treatment_window: 20` derived from `realized_vol[0]` | Placebo blocks shorter than the spread's 20-session realized leg destroy the serial dependence the refutation preserves and yield a fake p-value |
| `latent_factors.device: cuda` declared | Otherwise `preferred_latent_device()` makes training identity a property of the host |
| bps regime headline, per-share exploratory | Per-share half-spread on split-adjusted prices conflates split adjustment with realized friction |
| No holdout lock; selection reproducible from registry | A once-only lock preserved a stale answer (superseded IPCA lineage); protection is that selection never consults the holdout, not that it runs once |

## Market-specific guardrails

### Data, labels, panel timing

| Guardrail | Why | Do this |
|---|---|---|
| Vendor sentinel values | A ranking sorts on a placeholder written for a failed IV solve; `drop_nulls` leaves it | Find what the file does with failure before computing; normalise at the loader so every reader gets one answer |
| Universe count cycles with the expiration calendar | Membership follows a listing calendar, not liquidity; >2/3 of names absent on narrow dates; all-session averages hide it | Count the universe on decision dates against `BREADTH_FLOOR = max(top_k_grid) = 20` |
| Corporate actions entering as returns | Splits, spin-offs, ticker reassignment leave raw prices intact and returns meaningless | Use `adj_factor` (splits + dividends → total return) inside `sec_id`; every window complete and within one security: `pl.col("session").diff().over("sec_id") == 1`; reconciliation must balance |
| Null comparison yields a confident class | `null > 0` is false, not null; naive direction labels write a class into unfillable rows | Explicit null guard when deriving `fwd_dir_*` from `classification_eval_label` |
| Holdout bleed through forward labels | A row observed before holdout whose outcome resolves inside it is a holdout row | Usable boundary = `holdout_start` − horizon in sessions (`fold_boundary_date`, NYSE calendar); diagnostics cut at the same point |
| Overlapping forward windows overstate evidence | Consecutive 5d labels share four of five days | `effective_sample_size`, `panel_autocorrelation`; HAC SE with `maxlags = horizon`; purge gap set by the forward window, not the effective count |
| Shifting a sparse panel by rows | One row back can be a month old; a 5-session fill spent on five rows spanning far more | `on_session_grid` inside `sec_id` first, then `.shift(IV_LAG)` and 5-session forward fill |
| Carry-forward crossing a ticker change | A dead security's last IV carried into whoever picked up the ticker | Apply lag and fill `.over("sec_id")` |
| Same-bar fill | Friday-close features receiving a Friday-close fill | Decision at close, order at next open; seeded random-signal smoke test (`SEED = 42`) catches gross engine bias only, not research-choice bias |
| Survivorship of current-constituent roster | 2021 holdout on today's membership excludes every departed company; optimistic by an unmeasured amount | State population scope on every result; never generalize to the index-membership process |

### Features

| Guardrail | Why | Do this |
|---|---|---|
| Whole-sample transforms (leakage by fit) | A z-score or scaler fitted across the sample reads the future | Seal test: rebuild the matrix with later dates withheld and diff every shared value |
| Fitted-feature leakage via estimation window and recursion inputs | Values inside the window are retrospective; libraries derive starting variance, scaling, clip bounds from the whole series handed to them | Schedule burn-in/refit in config; pin recursion inputs to the window; assert values do not move when later observations are deleted; score only inference-span dates |
| Degenerate GARCH on short windows | alpha + gamma < 0 for 19.0% of securities on the first 252-session block vs 1.8% walk average | `MIN_OBS = 252` floor; never fit below a year; report both rates |
| Horizon mismatch in `garch_ivrv_spread` | 1-session-ahead forecast subtracted from a 30-day IV | Flag as limitation; a term-structure-aware forecast would be needed |
| Normal innovations | Equity tails are heavier; persistence a little high, big moves over-scored as surprises | Acknowledge; consider a fat-tailed `dist` |
| Stale/carried values look like quotes | Carry-forward makes a value stale not missing; quality columns describe the last solve, not its age | Staleness screen `STALENESS_MAX = 0.50` (share of rows repeating prior value) |
| Uneven per-feature coverage | Failed solves leave gaps the fill only partly closes; option columns never reach price-column coverage | `COVERAGE_MIN = 0.70`; missing-value handling is a modelling choice |
| Thin cross-sections | Rank correlation over too few names is meaningless | Drop sessions with < `MIN_CROSS_SECTION` (20, capped by loaded universe) rather than averaging them in |
| z-score unbounded as own dispersion → 0 | IV dynamics family failure mode | `ZERO_FLOOR = 0.001`; `risk_adjusted_vol_floor: 0.01` in the risk-adjusted momentum denominator |
| IV read within a maturity window, not fixed tenor | An expiration change moves the series alongside the vol it measures | Acknowledged limitation of DTE-bucket selection |
| Pooled vs within-date ranks | Different question, same arithmetic | Percentile within the decision date (`over("timestamp")`) |
| No price-only ablation | Every model from Ch 11 on is fit on option + price columns together | Cannot attribute value to options vs free price history; add a second fit with option columns withheld before claiming it |

### Evaluation and model selection

| Guardrail | Why | Do this |
|---|---|---|
| Multiple testing across the feature set | N features at 5% → 0.05·N false positives | BH-FDR at `FDR_ALPHA = 0.05` over the *declared* searched set (fixed by generation rules, not what looked promising); record the arm in ledger `note` |
| Exploration arm promotes without confirmation | Fold agreement + effect size on two folds is the weakest evidence | Read `note`; promoted count can exceed FDR-cleared count |
| Screening validation windows, not the panel | Walk-forward spends early sessions training; GARCH columns exist only inside validation windows | Report fold by fold; do not expect the pooled average to match fold stats |
| Degenerate predictions inflate IC | A model predicting the same value for every stock contributes no IC on those dates; IC over a self-selected subset is not a metric (seen with aggressive L1 and networks) | Read `ic_n_days` coverage beside IC |
| Checkpoint dominates configuration | Where median within-run IC range exceeds grid spread, the leader was chosen by where its run happened to be | `checkpoint_dominates`, `epoch_against_capacity`; register every checkpoint |
| Best-epoch comparison is post-hoc selection | `ic_high` at each config's own `epoch_at_high` picks the epoch after seeing results | Treat `epoch_at_high` as a learning-curve diagnostic; compare only in `14_backtest` |
| Lookback silently shrinks the sample | 60-step windows drop early fold rows and short-lived names | Read `eligible_rows`; cross-family IC comparison needs the coverage column |
| Single-label population (`patchtst`) | Cannot separate architecture effect from target effect | Add a second preset/label under `deep_learning` before generalizing |
| PCA vs IPCA sample mismatch | IPCA scores ~10% more rows (shorter-lived, less liquid names) | Read the IC gap as a bound both ways; to estimate, score IPCA on PCA's balanced subset |
| PCA label rows not on a common sample | `prepare_panel_data` fills from each label; a 10-day window runs out earlier | Read label gaps as a ranking of what was fitted, not a horizon measurement |
| CAE vs SAE is not a controlled comparison | Four simultaneous differences (bottleneck 5 vs 96, hidden `(32,)` vs `(896,448,448,256)`, lr 1e-3 vs 1e-4, loss) | Match bottleneck, hidden units and lr in presets before attributing to supervision |
| SDF checkpoint-value sign tests | Packed phases put a validation-chosen state on `0`; `< 0` keeps the dangerous state, `<= 0` drops a sibling's only checkpoint | Identify checkpoint kind via the library's named constants only |
| Validation-chosen states entering the population | "Epoch where validation Sharpe was highest" wins a validation-Sharpe contest by construction | Schedule from physical epochs only; a guard fails the catalog if one appears |
| Macro panel moves SDF identity alone | SDF is the only member hashing a macro panel | Check identities before concluding SDF drifted |
| No convergence check (SDF) | Declared budgets run and stop; settled and unsettled look identical | Inspect epoch curves; wandering is normal for adversarial training |
| Stochastic non-reproducibility | CAE/SAE/SDF reruns reproduce identity, not the third decimal | Compare at interval level; carry training identities |
| Device nondeterminism | GPU and CPU sum gradients in different orders → different weights | Device inside training identity; separate population per device (`PUBLISHED_DEVICE = "cuda"` for TabM) |
| Validation folds read many times | Every number in 05-13 is on windows re-read throughout | Treat as diagnostics; 2021 holdout spent once in Ch 20 |
| Two folds, one of them 2020 | Half the evidence is a year where IV across every name moved together; cannot separate weak feature set from unrepresentative window | Caveat every fold-level claim; "enough to say an effect is not large, not enough to characterise one" |
| Never treat IC/HAC as a significance test | Daily ICs on overlapping returns are serially dependent; intervals are wide | Select on backtest Sharpe with effective-rank DSR |
| A split family can silently lose a model | A menu entry with no notebook is never fitted; the family just looks smaller | `11_latent_factors` asserts `declared <= claimed`; raise on empty glob |
| Causal identification is untestable | Conditional ignorability, overlap, SUTVA; refutation is evidence about timing, not confounding | Report DML beside naive OLS and permutation null; never use as a ranking signal |
| Causal diagnostics consuming the holdout | Outcome windows reaching past the cutoff leak the final window | The request drops every observation whose outcome could reach past `holdout_end` |
| Wrong `SUPERSEDES_CAUSAL` | Registration refuses after the fit and 100 placebo refits are paid; stranded rows (no `identity_version`, e.g. `b47bd0ec208a`) cannot be named | Resolve current canonical identity from the registry (`18f777683c40`); `run-production-notebook.sh` passes no overrides |

### Backtest, allocation, risk, costs, holdout

| Guardrail | Why | Do this |
|---|---|---|
| Partial pipeline runs | Greedy funnel is valid only after all predictions and all EW baselines exist | Never skip a family silently or start downstream from a partial baseline |
| Config changes invalidate hashes | `execution.initial_cash`, `share_type`, `allocator_lookback`, per-allocator overrides are spec-hash inputs | Expect every `backtest_hash` to change; read via `get_backtest_config()`, never a local constant |
| Partial-coverage prediction sets | A set measured on fewer validation dates is not comparable | Remove before ranking; eligibility = coverage of the canonical validation window; extra dates earn nothing |
| Cross-label selection | Picking the best across labels selects the label as well as the model | Each label keeps its own lineage; primary leader from primary candidates only |
| Sharpe clearing zero has not cleared the search | Many correlated variants per family x label | Effective-rank DSR; two-fold PBO is too coarse either way |
| Equal weight re-entering the allocator menu | It is the baseline every allocator is read against | `15_portfolio_management` raises if it reappears |
| Conformal allocator inherits miscalibration | `conformal_weighted` reads interval width; out-of-time under-coverage measured in 13 | Read its result beside the coverage number |
| MAE-calibrated risk thresholds | Learned from the same validation paths used to score them | Excluded in v3.1; only predeclared thresholds compete |
| Re-deriving selection downstream | Four stages each re-selecting makes the holdout config a rule, not a record | Freeze the field as an immutable candidate set in `16` |
| Cost band is not selection-adjusted | Block-bootstrap band conditions on the chosen lineage, ignores signal/allocation selection | Read as sensitivity; respect the lower bound's zero crossing, not the point path's |
| Per-share cost on split-adjusted prices | Flat dollar half-spread conflates split adjustment with friction; universe spans three orders of magnitude in price | Exploratory only; `PRICE_CUTS = [50, 100, 250]` buckets; name-level execution data would measure real cost |
| Under-capitalised integer sizing | $100k x top_k=20 x integer shares x high-priced tail → engine almost never trades | `initial_cash: 1_000_000`, `share_type: integer` |
| Holdout lock preserving a stale answer | A once-only lock evaluated a config the pipeline no longer selects | No lock; count holdout fits per window; re-run 18/19 on the selected config; keep superseded rows |
| Holdout refit matched by hash | A different training interval is a different training identity | Match training specs field by field |
| Validation-to-holdout gap read as a test | Validation is a search maximum; holdout is one noisy period | Report both with intervals; make no deployment claim |
| `ML4T_OUTPUT_DIR` redirection | `Study.activate()` rewrites the variable and clears caches; globbing from the data dir finds nothing | Read sources from the repo; open canonical read-only with `Study.at`; `Study.regenerate` refuses unless `features`, `labels`, `run_log` are symlinks |
| `WORKSPACE` read on one tier only (#1100) | A canonical run passed while reporting on the published registry | Read `WORKSPACE` at both tiers; preview requires a workspace; `_refuse_preview_activation` stops reduced runs from publishing |
| Resolved overwriting requested | A resolved value under the parameter-cell name hides the request | Keep request and resolved names distinct; `resolved = injected or declared` |

## Results and lessons

Caveat on everything below: two validation folds (2019, 2020); ICs in 06-08 are daily averages without serial-dependence adjustment; the current-lineage holdout is unresolved; numeric Sharpe, DSR, PBO, conformal coverage and the winning lineage/allocator/top_k/risk rule are not in the notes.

| Stage | Finding |
|---|---|
| 01 feasibility | Most cross-sectional ordering survives one week, substantially gone within a few weeks → weekly cadence. Universe count swings monthly with the expiration calendar; >2/3 of names absent on narrow dates. Cost failure criterion: typical open-to-open move ≤ round-trip cost (not triggered for S&P 500 names). Options-vs-price value never settled. |
| 04 GARCH burn-in | 252 vs 504: coverage 76.5% vs 54.7%; non-emitting securities 57 vs 103; fold-0 training sessions consumed 22,130 (121 securities) vs 159,463 (592). Degenerate fits (alpha + gamma < 0): 19.0% first block, 1.8% walk average. |
| 04 VRP denominator swap | 184,299 rows, 497 decision dates: `ivrv_spread` (rv_20 memory) mean IC −0.0053, HAC t −0.44; `garch_ivrv_spread` mean IC −0.0032, HAC t −0.37; daily ICs correlate −0.25; paired difference +0.0021, HAC t 0.13 → indistinguishable from no change. None of `garch_cond_vol`, `garch_ivrv_spread`, `garch_vol_surprise` clears FDR standalone; largest `garch_cond_vol` IC −0.0065, HAC t −0.34. |
| 05 screen | A weak screen: naive vs HAC vs BH significant counts printed with an "inflation" factor; exact PROCEED/STOP/REVISE counts live in `triage_ledger.parquet`. |
| 06 linear | `fwd_ret_risk_adj_5d`: every config's IC above zero; `fwd_ret_5d`: none above zero, same menu, features and folds. Ten orders of magnitude of ridge shrinkage never flips a label's sign. Within-label spread (`best_ic` − `worst_ic`) exceeds the gap to the next label. Aggressive L1 produced constant predictions. |
| 07 GBM | Share of candidates clearing zero differs by label on an identical grid; one label's own candidates span more than the leaders across all five labels. MSE/MAE/Huber interleave (no Huber > MAE > MSE ordering): daily equity tails are mild, so the "prefer Huber" rule (short straddle, perps) does not apply. Two families agree on which target the features rank → the finding is about the target. |
| 08 TabM | Read the capacity chart for whether IC moves at all, not which end leads; useful question is whether within-run (epoch) spread matches across-config spread. No intervals reported. |
| 11 latent factors | Scored rows IPCA vs PCA: 194,813 vs 177,682 (`fwd_ret_5d`), 192,139 vs 175,360 (`fwd_ret_10d`), 194,748 vs 177,769 (`fwd_ret_risk_adj_5d`). CAE/SAE panel `ragged train=493/475, max_N=503`. CAE reconstruction error falls with epochs by construction; IC need not follow. |
| 12 causal | Current canonical row `18f777683c40`; stranded capped fit `b47bd0ec208a` (2026-08-26, 37,240 rows = 9.9% of panel, no `identity_version`); refutation re-based on the HAC t-statistic 2026-09-10. Effect is not a ranking signal. |
| 13 model analysis | Primary label `fwd_ret_5d`: no family clears zero; point estimates sit within their SEs, so ordering is not a result. Corrected v3.1 registry: PCA clears zero on `fwd_ret_10d` and `fwd_ret_risk_adj_5d`; families have CI-credible estimates on different labels → label-routed allocation. Pairwise rank correlations: families not redundant; GBM and linear moderately similar; structural vs supervised rankings genuinely differ. |
| 14-17 funnel | Stage chart shows 1-3 points depending on whether allocation and risk improved; intervals wide at every stage; the point path moves more than uncertainty narrows. Low pairwise correlation among weak signals favours label-routed allocation, not a uniform ensemble. |
| 19-20 holdout | 2021 is one draw with a wide interval; not a deployment claim; a validation-to-holdout drop has at least two sufficient explanations (selection bias, different market) that one window cannot separate. |

**Lessons an agent should carry:** (1) the label decides the sign, the grid decides the magnitude; (2) IC decides nothing, backtest Sharpe after costs selects once; (3) a window costs rows, so every cross-family comparison needs a coverage column; (4) more capacity is not more information: CAE sees exactly what IPCA saw; (5) reconstruction is not forecasting; (6) a weak result reported with intervals is the deliverable.

## How to adapt this pattern to a new dataset

1. **Write the setup contract first** (`config/setup.yaml`): universe and `eligibility_rule`, `decision.cadence` + `snapshot` + `execution_delay`, per-input `lag` (fixed by when each input becomes knowable, not by convenience), label `horizons`, `buffer`, `variant_buffers`, `rebalance_step = ceil(horizon / cadence)` against per-label cadence, `evaluation` splits and holdout, `costs`, `execution`, sweep grids. Derive every notebook constant from it (`PRIMARY_LABEL = setup["labels"]["primary"]`, `IV_LAG = int(setup["decision"]["iv_feature_lag"].split("_")[0])`, `BREADTH_FLOOR = max(top_k_grid)`), never from a hard-coded list.
2. **Feasibility before research** (`01`): count the universe on decision dates against `BREADTH_FLOOR`; measure ordering persistence as correlation between cross-sectional orderings k dates apart with a null band (not per-name autocorrelation); set cadence where persistence survives one step and decays within a few; stop if typical move ≤ round-trip cost.
3. **Labels** (`02`): rebuild tradable prices with the adjustment factor inside a persistent entity id; entry/exit prices a strategy could actually trade at; complete windows inside one security (session-diff == 1 check); explicit null guard for direction labels; cut diagnostics at `holdout_start − horizon`; report `effective_sample_size`; write `.digest.json` sidecars via `write_artifact`.
4. **Features** (`03`): declare families in the register (`pattern`, `role`, `hypothesis`, `inputs`, `lookback`, `lag`, `frame`, `representation`, `failure_mode`); reindex sparse inputs onto the entity's own session grid (`on_session_grid`) before lag and forward fill; apply lag/fill `.over(entity)`; within-date percentiles for ranked columns; floors (`ZERO_FLOOR`, `PRICE_FLOOR`, `risk_adjusted_vol_floor`); warmup audit to `max lookback`; redundancy clusters at `CUT = 0.7`; run the seal test.
5. **Model-based features** (`04`): declare `burnin`, `refit_every`, spec in config; pin scaling, starting variance and clip bounds to the estimation window; assert values do not move when later observations are deleted; score only inference-span dates; test variants as a paired daily-IC difference with HAC.
6. **Screen** (`05`): per-session Spearman IC with `min_cross_section` ≥ 20; HAC `maxlags = horizon`; bootstrap `horizon + 1`; BH-FDR at 0.05 over the declared set; apply Table 7.2: PROCEED if `fdr_sig`; else PROCEED if `sign_consistency ≥ 0.60` and `|mean IC| ≥ 0.005`; STOP if coverage < 0.70 or staleness > 0.50; REVISE otherwise. Write `triage_ledger.parquet` with the arm in `note`.
7. **Train every declared family on every declared label** (`06`-`12`) from `config/training/{label}.yaml`; register every checkpoint as its own prediction set under a device-specific population; keep controls (`nlinear`, PCA) in the menu; declare hyperparameters, do not search them on validation IC; keep request vs resolved parameter names distinct.
8. **Compare with intervals** (`13`): HAC IC intervals, pairwise rank correlations, deciles, conformal coverage; count CI-credible label-family pairs; make no ordering claim inside SEs.
9. **Select once** (`14`): EW top-k over `top_k_grid` for every eligible (full-coverage) prediction set per label; random-signal smoke test; bootstrap Sharpe, effective-rank DSR, PBO; each label keeps its own lineage.
10. **Funnel** (`15`-`17`): top-10 → allocators (judge by method average across `top_k` and the per-`top_k` curve; gain = allocated Sharpe − same lineage EW Sharpe); top-1 → predeclared risk overlays (claim only if the paired block-bootstrap interval excludes zero, else advance the parent; freeze the field) → cost surface (claim must respect the lower bound's zero crossing; smooth decay is the finding, a cliff is a warning).
11. **Holdout** (`18`-`20`): refit the single selected config on all pre-holdout history ending at `holdout_start − buffer`; publish only the selected checkpoint; run the spec unchanged once; match specs field by field; report holdout beside validation with intervals; no deployment claim; keep superseded rows and count holdout fits per window.
12. **Record the limits** you did not close: price-only ablation absent, survivorship, two folds, fixed-tenor IV, Normal innovations, horizon mismatch.

### Key APIs and code patterns

| Area | Names |
|---|---|
| Config access | `setup["labels"]["primary"]`, `setup["labels"]["horizons"][label].rstrip("D")`, `setup["decision"]["iv_feature_lag"].split("_")[0]`, `setup["features"]["windows"]["iv_forward_fill"]`, `setup["backtest"]["sweep"]["top_k_grid"]`, `get_backtest_config()`, `get_top_n_predictions(case_study, stage)`, `families_from_config(setup)` → family objects with `.lookback`, `.pattern` |
| Loaders | `load_sp500_daily_bars()`, `load_sp500_options_surface()`, `load_modeling_dataset` (financial + model_based features for every model notebook) |
| Calendar | `from ml4t.diagnostic.splitters.calendar import TradingCalendar`; `TradingCalendar("NYSE").trading_days_between(start, end)` |
| IC / FDR | `from ml4t.diagnostic.metrics import compute_ic_hac_stats, cross_sectional_ic_series`; `compute_ic_uncertainty(series, horizon=HAC_MAXLAGS + 1, ic_col="ic")` → `ci_boot_lower/upper`; `from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr` → `adjusted_p_values`, `rejected`, `n_rejected` |
| Case-study helpers | `case_studies.utils.feasibility`; `artifact_digest` (`value_digest`, `write_artifact`, `read_digest`); `artifact_quality` (`quality_report`, `render_quality_report`); `label_diagnostics` (`effective_sample_size`, `panel_autocorrelation`); `feature_engineering` (`on_session_grid`); `cv_window` (`fold_boundary_date`, `modeling_fold_boundaries`; reads each label file's own timeline); `temporal` (GARCH scheduling); `warning_policy.apply_notebook_warning_policy`; `latent_factors/adapter.py` (computation `.runtime`, `.numerical_runtime`); `case_studies.research` (06/07/08 training-population API) |
| Study / registry | `open_study`, `Study.at` (read-only canonical), `Study.activate()` (rewrites `ML4T_OUTPUT_DIR`, links `config`, `labels`, `features`), `Study.regenerate` (symlink-only), `get_case_study_dir`, `_refuse_preview_activation`, `CASE_DIR`; prediction sets keyed on `(training identity, checkpoint)` |
| Parameter cells (tag `["parameters"]`) | `CASE_STUDY_ID`, `EXECUTION_TIER`, `WORKSPACE`, `PRIMARY_LABEL`, `MAX_SYMBOLS`, `CV_FOLDS`, `MAX_SAMPLES`, `N_PLACEBO`, `FORCE_RETRAIN`, `SUPERSEDES_CAUSAL`; backtest `LABEL`, `SPLIT`, `TOP_K`, `FORCE_REBACKTEST`, `TOP_N_PREDICTIONS`, `SEED` |
| Latent-factor runner | `prepare_panel_data`, `run_pca_fold`, `run_ipca_fold`, `run_cae_fold`, `run_sae_fold` (discards `n_factors`; `bottleneck_dim` sets width, so log label `sae (K=5)` is misleading), `persistent_entities`, `preferred_latent_device()` |
| Causal | shared causal request; `current_causal_identities`, `causal_supersedes`, `CAUSAL_RUNNER_VERSION`, `confounding_bias_pct`; `run-production-notebook.sh` |
| Polars panel idioms | keys `["timestamp", "symbol"]`, entity `sec_id`; within-date percentile `over("timestamp")`; lag/fill `over("sec_id")` |
| Triage ledger columns (partly inferred) | `feature`, `family`, `coverage`, `staleness`, `mean_ic`, `hac_t`, `p`, `fdr_p`, `fdr_sig`, `sign_consistency`, `monotonicity`, `decision`, `note` |

Derive timing constants from config, never from a hard-coded list:

```python
PRIMARY_LABEL = setup["labels"]["primary"]
LABEL_HORIZON_SESSIONS = int(str(setup["labels"]["horizons"][PRIMARY_LABEL]).rstrip("Dd"))
HAC_MAXLAGS = LABEL_HORIZON_SESSIONS            # HAC bandwidth = label horizon
IV_LAG = int(setup["decision"]["iv_feature_lag"].split("_")[0])   # "1_day" -> 1
BREADTH_FLOOR = max(setup["backtest"]["sweep"]["top_k_grid"][PRIMARY_LABEL])
```

Per-feature IC with overlap-aware uncertainty and search-wide FDR (argument names beyond `min_cross_section` inferred from the 05 call site):

```python
ic_df = cross_sectional_ic_series(panel, feature, label, min_cross_section=MIN_CROSS_SECTION)
stats = compute_ic_hac_stats(ic_df, ic_col="ic", maxlags=HAC_MAXLAGS)
bands = compute_ic_uncertainty(ic_df, horizon=HAC_MAXLAGS + 1, ic_col="ic")
fdr = benjamini_hochberg_fdr(p_values, alpha=FDR_ALPHA, return_details=True)
```

Family coverage check (`11_latent_factors`): read sources from the repo, not the data dir.

```python
declared = {m for menu in menus for m in menu["latent_factors"]}
claimed = {read_MODEL_NAME(p) for p in glob(REPO / "11?_*.py")}
assert claimed, "empty glob"           # distinct from "all unclaimed"
assert declared <= claimed, declared - claimed
```

Checkpoint kind: use the library's named constants for SDF phases; never `checkpoint_value < 0` or `<= 0`. Request vs resolved: `resolved = injected or declared`, under a different name from the parameter cell.

## Related references

- `chapters/06_strategy_definition.md` — feasibility (§6.2-6.6), search accounting and run log (§6.7) behind `01` and the registry.
- `chapters/07_defining_the_learning_task.md` — label construction, univariate evaluation, Table 7.2 triage rule, multiple testing (`02`, `05`).
- `chapters/08_financial_features.md` — IV, skew, term-structure, VRP, momentum, realized-vol families and the session-grid/lag procedure (`03`).
- `chapters/09_model_based_features.md` — GJR-GARCH schedule, pinned windows, inference span; chapter notebook `09_model_based_features/08_garch_volatility.ipynb` (`04`).
- `chapters/11_ml_pipeline.md` — regularized linear/logistic baselines and training-population API (`06`).
- `chapters/12_gradient_boosting.md` — LightGBM grid, checkpoints, TabM (`07`, `08`).
- `chapters/13_dl_time_series.md` — `nlinear`, LSTM, PatchTST, lookback row accounting (`09`, `10`).
- `chapters/14_latent_factors.md` — PCA/IPCA/CAE/SDF/SAE, reconstruction vs forecasting (`11a`-`11e`).
- `chapters/15_causal_estimation.md` — walk-forward DML, placebo block design (`12`).
- `chapters/16_strategy_simulation.md` — EW baseline, coverage-aware selection, DSR/PBO (`14`).
- `chapters/17_portfolio_construction.md` — allocator sweep, Ledoit-Wolf degeneracy, conformal weighting (`15`).
- `chapters/18_transaction_costs.md` — bps vs per-share regimes, cost-surface bands (`17`).
- `chapters/19_risk_management.md` — predeclared overlays, paired bootstrap (`16`).
- `chapters/20_strategy_synthesis.md` — holdout discipline, funnel reconstruction, cross-case-study feature evaluation (`18`-`20`).
- `case_studies/sp500_options.md` — sibling that trades the options and declares the same `o: 1` GJR deviation.
- `case_studies/etfs.md`, `case_studies/cme_futures.md`, `case_studies/fx_pairs.md` — siblings that declared `conformal_weighted` first.
- `case_studies/us_equities_panel.md`, `case_studies/us_firm_characteristics.md` — other weekly/daily equity cross-section panels with the same funnel.
- `libraries/ml4t_diagnostic.md` — `compute_ic_hac_stats`, `cross_sectional_ic_series`, `compute_ic_uncertainty`, `benjamini_hochberg_fdr`, `TradingCalendar`.
- `libraries/ml4t_backtest.md` — engine config (`initial_cash`, `share_type`, rebalance thresholds), allocators, risk overlays.
- `libraries/ml4t_engineer.md`, `libraries/ml4t_models.md` — feature and model-family building blocks.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `workflow.md`, `companion_repo.md`, `glossary.md` — cross-cutting summaries this file feeds.
- Further reading:
  - Chen, Pelger & Zhu (2021), adversarial stochastic discount factor (`11d`).
  - Repo issue #1100 (`WORKSPACE` read on one tier).
  - `memory/feedback_2026_05_15_equity_sizing_invalidated.md`, `memory/feedback_max_weight_caps_intentionally_absent.md` (config rationale).

## Glossary

| Term | Meaning here |
|---|---|
| IV surface | Implied vol summarized per name per session across delta (`atm 0.50`, `25d 0.25`, `10d 0.10`) and DTE buckets (`7d [5,10]`, `30d [25,35]`, `90d [80,110]`) |
| `iv_30_atm` | 30-day ATM implied vol; the feasibility ranking column and baseline for the IC audit |
| `skew_rr_30_25d` | 30-day 25-delta risk reversal (put IV − call IV) |
| `ivrv_spread` / VRP | `iv_30_atm − rv_20` in vol points; forward-looking month vs backward-looking month |
| `garch_ivrv_spread` | IV minus GJR-GARCH one-session-ahead conditional vol (horizon mismatch acknowledged) |
| GJR-GARCH(1,1) | GARCH with asymmetry term `o: 1`; negative shocks raise variance more |
| Garman-Klass | Range-based vol estimator, coefficient `2·ln2 − 1`, annualized by `sqrt(252)`; blind to overnight gaps |
| Skip-month momentum | Return t−252 → t−21, dropping the reversing recent month |
| Within-date percentile | Rank in (0,100) across names on one decision date; `features.ranked` names |
| `sec_id` / `adj_factor` | Entity id persisting across ticker changes / split-and-dividend adjustment factor |
| Lookback / lag | Daily bars back from the decision timestamp (warmup floor) / sessions until the input is knowable (1 for option families, 0 for price families) |
| IC | Per-session Spearman rank correlation between feature or prediction and forward return, averaged |
| HAC | Newey-West SE with `maxlags = label horizon` (5) |
| BH-FDR | Benjamini-Hochberg at α = 0.05 over the declared searched set |
| Confirmation vs exploration arm | PROCEED via `fdr_sig` vs via `sign_consistency ≥ 0.60` and `|IC| ≥ 0.005` |
| Staleness | Share of a column's rows repeating the prior value (carry-forward) |
| Seal test | Rebuild features with later dates withheld and diff shared rows |
| Inference span | Dates after a fitted feature's estimation window (truly point-in-time) |
| Checkpoint | A scored intermediate state (tree count or epoch) registered as its own prediction set; solved estimators publish checkpoint 0 |
| Population | Set of prediction identities a backtest selects over; device-specific |
| Training identity | Hash of the training spec; a holdout refit has a different one |
| Coverage (prediction set) | Share of canonical validation dates a prediction set scores; eligibility for ranking |
| `eligible_rows` | Rows surviving the sequence lookback (60) requirement |
| `epoch_at_high` / `ic_high` | Checkpoint and value of peak validation IC; diagnostic only |
| `nlinear` | Window minus its last observation, one linear map; memoryless sequence control |
| PatchTST | Transformer over contiguous patches of a 60-step window |
| TabM | Weight-sharing ensemble MLP: shared two-layer backbone, per-member scaling vectors and heads |
| IPCA / CAE / SAE / SDF | Instrumented PCA (linear characteristic map) / conditional autoencoder (network map) / supervised autoencoder (label in loss) / adversarial pricing kernel |
| DML | Double/debiased ML with cross-fitting and embargo; Driscoll-Kraay SE; `confounding_bias_pct = (OLS − DML)/DML` |
| Greedy funnel | signal (all) → allocation (top-10) → risk/cost (top-1) → holdout (one config) |
| Lineage | A label's chain of selected signal → allocation → risk → cost rows |
| Effective-rank DSR / PBO | Deflated Sharpe counting correlated variants within family x label / two-fold probability of backtest overfitting |
| Canonical / preview tier | Production vs reduced run; preview needs `WORKSPACE` and cannot publish |
| Content-addressed hash | Spec hash keying every training, prediction and backtest artifact; re-keyed between `v3.0.0` and `v3.1.0` bundles |
