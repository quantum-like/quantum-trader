# Case study: Crypto perpetual futures funding (19 pairs, 8-hourly, fwd_ret_8h)

> Binance USD-M perpetuals, 19 names, decisions at the three daily funding settlements (00/08/16 UTC), 2020-2025, two walk-forward folds and a 2024-25 holdout. The premium index is read as a *crowding* signal and ranked cross-sectionally; funding is never collected as carry, it is a feature upstream and a position-signed cash flow settled inside the backtest engine. The study is the book's smallest cross-section and highest non-intraday cadence, so every stage is about completed-bar timing, declared populations, and the uncertainty of two validation years. Headline lesson: the validation winner (Sharpe 1.668, 95% CI [0.42, 2.87], PSR p = 0.0049) was the best of 2,807 candidates; every deflated Sharpe (raw/MP/ER) went negative, and the holdout replay returned Sharpe -0.832 with an -82% drawdown. The deflation, not the interval, was the honest reading.

Repo root for every path below: `case_studies/crypto_perps_funding/`. Run from the repository root with `uv run python case_studies/crypto_perps_funding/NN_*.py`; 07-10 need CUDA. Download published results instead of training: `uv run python scripts/download_artifacts.py --cs crypto_perps_funding` (fetches `v3.1.0-artifacts`; `v3.0.0-artifacts` is a separate generation, hashes do not resolve across bundles). The README states no results; `19_strategy_analysis` reads them from `run_log/registry.db`.

## Setup contract

Everything below is in `config/setup.yaml` (`strategy_id: crypto_perps_funding`, `setup_version: v1`); notebooks read it, never restate it.

| Key | Value | Notes |
|---|---|---|
| `universe.symbols` | AAVE, ADA, APT, ATOM, AVAX, BNB, BTC, COMP, DOGE, DOT, ETH, INJ, LINK, MKR, NEAR, SOL, SUI, UNI, XRP (all `*USDT`) | `n_assets: 19`; `eligibility_rule: top_perps_by_volume`; unbalanced panel, assets enter at listing, no backfill. Fixed list chosen knowing which perps stayed listed: survivorship acknowledged in 01-03, not fixed |
| `decision` | `cadence: 8_hour_funding_aligned`, `snapshot: pre_funding_timestamp`, `execution_delay: at_funding_timestamp` | Information cutoff is the pre-funding snapshot; fill is at the funding bar itself (same bar) |
| `execution` | `initial_cash: 100_000`, `share_type: fractional`, `allocator_lookback: 240` | 240 8h bars ~ 80 days; applies to inverse_vol, risk_parity, hrp, mvo_ledoit_wolf. Read via `get_backtest_config()`; never declare a local `INITIAL_CASH`. Changing cash or share_type invalidates every `backtest_hash` |
| `mapping` | `class: long_short_funding_aligned`, `position_state_space: long_short`, `entry_logic: threshold_or_rank_based`, `sizing: equal_weight_or_risk_parity` | `position_state_space` is what makes `get_backtest_config(...).long_short` true |
| `costs` | `class: material`, `components: [taker_fee, maker_fee]`, `fee_schedule: {taker_bps: 4, maker_bps: 2}` | Majors (BTC, ETH, BNB, SOL, XRP) clear maker, alts taker (used by Ch18 spread estimation and 12's cost comparison). The backtest charges the taker tier flat: `config/backtest/base.yaml` has `commission: {model: percentage, rate: 0.0004}` and `slippage: {model: percentage, rate: 0.0001}`, `allow_short_selling: true`, `execution_mode: same_bar`, `calendar: crypto`, `data_frequency: irregular` |
| `backtest.rebalance.default` | `min_weight_change: 0.005`, `min_trade_value: 100.0` | A trade is skipped when weight change < 0.005 AND notional < $100. `benchmark: {0.0, 0.0}` so 1/N rebalances at all |
| `backtest.sweep.top_n_predictions` | `signal: 0` (all), `allocation: 10`, `risk_overlay: 1`, `cost_sensitivity: 1` | Read via `get_top_n_predictions(case_study, stage)`; `expensive_allocators_skip: false` |
| `backtest.sweep.top_k_grid` / `quantile_grid` | `[3, 5, 10]` / `[5]` for all four labels | Tradeable cross-section per decision (measured 2026-08-23): 9 at p10, 18 median, 19 at p90. k=5 is 28% of the book, k=10 56%, k=3 17% supplies the concentrated side. k=10 is infeasible long-short (needs 20 names) and is dropped at run time |
| `backtest.sweep.allocators` | score_weighted, inverse_vol, risk_parity, mvo_ledoit_wolf, hrp, conformal_weighted | Equal weight is the baseline, not listed. No max-weight cap |
| `backtest.sweep.cost_grid_bps` | `[0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]` | Round trip on traded notional, half commission / half slippage. Production 4 + 1 = 5 bps is a grid cell |
| `backtest.sweep.risk_controls.position` (14) | stop_loss {3, 5, 10, 15}%; trailing_stop {1, 2, 3, 5, 10, 15, 20}%; time_exit {10, 20, 40} bars | No portfolio-level controls declared. Grid fixed so search width is a stated property |
| `evaluation` | `n_splits: 2`, `train_size: 2Y`, `val_size: 1Y`, `holdout_start: '2024-01-01'`, `holdout_end: '2025-12-31'`, `calendar: crypto`, `periods_per_year: 365` | Validation 2022-01-01 to 2023-12-31 (two one-year folds, 729-730 daily periods each). Fold 0 is the most recent development fold |
| `labels` | `primary: fwd_ret_8h`, `buffer: 8H`; variants `fwd_ret_24h` (buffer 24H), `fwd_dir_8h`, `fwd_dir_8h_3c` (8H) | `rebalance_step: {fwd_ret_8h: 1, fwd_ret_24h: 3, fwd_dir_8h: 1, fwd_dir_8h_3c: 1}` (ceil(24/8) = 3 so holdings do not overlap). `classification_eval_label` maps both direction labels to `fwd_ret_8h` (IC against the continuous return; AUC/accuracy/log loss against the binary label). Leave the buffer string as `"8H"`: 144 registered training runs hash it; `normalize_label_buffer` handles pandas' deprecated "H" |
| `features` | `bar_hours: 8`, `majors`, `ranked: premium_index_close`, `redundancy_cut: 0.7`, `windows`, `clip`, `families` | Windows in 8h bars, map key = emitted suffix. See Design decisions |
| `model_based` | `min_train_bars: 500`; `garch: {refit_every: 21, vol_zscore_window: 90, zscore_clip: 10.0}`; `hmm: {n_states: 2, refit_every: 63, n_restarts: 10, tol: 1.0e-4}`; `incremental_ic: {min_cross_section: 10, min_decision_times: 20}` | 500 settlements ~ 5.5 months burn-in, not a fold condition |
| `modeling.gbm` | `libraries: [lightgbm]`, `preset: default`, `device: cpu`, `max_bin: 255` | 63 was the GPU default carried over by mistake; 255 is LightGBM's CPU default |
| `causal` | `treatment: premium_zscore_14d`, `confounders: [price_vol_14d, funding_rate, premium_dev_mean_14d]`, `method: walk_forward_dml` | |

Benchmark: full-universe 1/N equal weight (benchmark rebalance thresholds 0/0). Selection reference at every stage is the equal-weight baseline on the same prediction set, never the rest of the grid.

Training menus (`config/training/{label}.yaml`): `fwd_ret_8h` = ols + ridge_a{1e-3..1e7} (12) + lasso_f{0.015, 0.03, 0.08, 0.2, 0.35, 0.5, 0.7, 0.85} + enet_f{same 8}; gbm {default, leaves_7, leaves_15, leaves_31, leaves_63} x {mse, mae, huber} = 15; deep_learning nlinear, lstm_h64, tcn; tabular_dl tabm_s/m/l; causal_dml dml. `fwd_ret_24h` same minus tcn. `fwd_dir_8h` = logistic_none, logistic_l2_C{0.001..100} (6), logistic_l1_C{0.01..100} (5; `logistic_l1_C0.001` omitted because it zeroes every coefficient and predicts a constant), `_binary` gbm presets, nlinear/lstm_h64 declared but NOT runnable (sequence runner is regression-only), tabm, NO causal_dml (DML nuisances are `HistGradientBoostingRegressor`, no classification branch; declaring it was the trap in ml4t/agent-workspace#396). `fwd_dir_8h_3c` keeps `logistic_l1_C0.001` and uses `_multiclass` presets. Editing a preset under `case_studies/config/{model_type}/` creates a new registry identity beside the old one.

## Pipeline stages

| Stage | Notebook | Chapter | What it does | What it writes |
|---|---|---|---|---|
| Feasibility | `01_feasibility_analysis` | 6 | Breadth at the funding timestamp vs floor 2 x max top-k = 20; exceedance of \|8h move\| vs round trip (maker 4 bps, taker 8 bps); premium ACF to 63 lags and integrated autocorrelation time; fold timeline via `generate_cv_splits`; kill conditions | Nothing (contract fixed in setup.yaml) |
| Labels | `02_labels` | 7.2 | `timestamp + 8h` availability clock; forward return `C_{t+h}/C_t - 1` only on exact grid landing; null-propagating binary label; development-tercile 3-class label; window reconciliation; ESS; baseline premium-z IC with HAC | `labels/{fwd_ret_8h,fwd_ret_24h,fwd_dir_8h,fwd_dir_8h_3c}.parquet` + `.digest.json` |
| Financial features | `03_financial_features` | 8 | 39 features in six families (carry, mean_reversion, momentum, volatility, cross_sectional, regime) from premium index and official funding; warmup audit; holdout-withheld rebuild check; redundancy and persistence plots | `features/financial.parquet` + sidecar |
| Model-based features | `04_model_based_features` | 9.3, 9.5 | Per-symbol GJR-GARCH(1,1) and market-wide 2-state HMM on a refit schedule (burn-in 500, refit 21/63, frozen at holdout); filtered probabilities; incremental IC with BH | `features/model_based.parquet` (5 features, keyed `(timestamp, symbol)`, no fold column) + sidecar |
| Evaluation | `05_evaluation` | 7.3, 7.4 | Quality, per-settlement Spearman IC with HAC (3 lags), fold sign consistency, BH-FDR at 0.05, quintile profile, redundancy; triage ledger (screening, not selection) | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` |
| Linear | `06_linear` | 11 | OLS/Ridge/Lasso/ElasticNet/logistic surfaces per fold (train-median impute, train-standardize); population `crypto_perps_funding-linear-validation-v1` | training runs + prediction sets in `run_log/registry.db`; coefficients `run_log/training/{hash}/`; scores `run_log/predictions/{hash}/` |
| GBM | `07_gbm` | 12.2, 12.3 | LightGBM 500 iterations, checkpoint every 50 (10 sets per config); NaN left in place; learning curves | boosters, `learning_curves.parquet`, `fold_metrics.parquet` under `run_log/training/{hash}/` |
| Tabular DL | `08_tabular_dl` | 12 (README; header says 18) | TabM s/m/l, 200 epochs, checkpoint every 25 (8 sets per config), `class_weight: balanced` per fold, CUDA | checkpoints `run_log/training/tabular_dl/` |
| LSTM | `09_dl_lstm` | 13 (README; header says 19) | NLinear + LSTM(2 x 64) on 60-settlement windows, gap policy drops windows crossing missing settlements, 100 epochs, checkpoint every 5 (20 sets) | checkpoints `run_log/training/deep_learning/` |
| TCN | `10_dl_tcn` | 13 | Causal dilated TCN (kernel 3, dilations [1,2,4,8], 32 channels, receptive field 61 >= 60), regression labels only | same |
| Causal DML | `11_causal_dml` | 15 | Walk-forward DML: treatment `premium_zscore_14d`, 3 confounders, 5 embargoed folds, HAC t, 100 block-permutation placebos (block 42) | one row in registry `causal_runs` |
| Model analysis | `12_model_analysis` | 12-15 | Every checkpoint of every family on one validation panel; causal result reported beside, never among | Nothing (reads registry) |
| Backtest | `13_backtest` | 16 | Equal-weight top-3 / top-5 / quintile long-short for all 677 prediction sets with funding settled in the engine; decision clock + coverage gate | per run: `daily_returns/weights/trades/fills/equity/portfolio_state.parquet`, `spec.json` under `run_log/backtest/{hash}/`; 12 populations `crypto-signal-{label}-{rule}-v1`; candidate sets `crypto-signal-{label}` |
| Portfolio | `14_portfolio_management` | 17 | Six allocators on top-10 `(family, config_name)` per label, paired vs equal-weight baseline | 72 populations `crypto-allocation-{label}-{scheme}-{allocator}-v1`; candidate sets `crypto-signal-allocation-{label}` (traded every fold) |
| Risk | `15_risk_management` | 19 | 14 position overlays on the top-1 per label, paired on common support | 56 populations `crypto-risk-{label}-{control}-v2`; candidate sets `crypto-final-validation-{label}` |
| Costs | `16_costs` | 18 | 11-level cost curve on the post-overlay winner, funding P&L beside execution cost, breakeven interpolation. Runs AFTER 15 | 44 populations `crypto-cost-{label}-{level}bps-v1`; no candidate set (selects nothing) |
| Holdout predictions | `17_holdout_predictions` | 20 | Refit the carrier on history ending `holdout_start - 24H`, predict 2024-25; one generation readable at a time | one `training_runs` + one `prediction_sets` row at `split='holdout'` |
| Holdout backtest | `18_holdout_backtest` | 20 | Replay the selected strategy (sizing, overlay, costs settled) with funding on holdout prices; guard = prospective backtest hash | one `backtest_runs` row at `stage='holdout'` |
| Synthesis | `19_strategy_analysis` | 20 | One selection across labels; bootstrap CI, PSR, DSR raw/MP/ER, RAS, min-TRL, funding attribution, holdout comparison | pool `crypto-final-selection` (2,807 members); otherwise reads registry |

Sequencing: 01 -> 02 -> 03 -> 04 -> 05 -> 06..11 (any order, all four labels per run) -> 12 (needs complete populations) -> 13 -> 14 -> 15 -> 16 -> 17 -> 18 -> 19. 17/18 open the holdout once, on the configuration 19 reports.

## Design decisions and why

### Data, clock and labels (01-02)

| Decision | Why |
|---|---|
| Read the premium index as crowding, not as carry to collect | A perp whose longs pay heavily is one borrowed money leans on; the bet is the next price move, ranked cross-sectionally, held to the next settlement. The label is a close-to-close return; "a strategy that collects funding is a separate construction on the funding-rate series, and this label cannot measure it" |
| Decide only at settlements | Cadence is an information schedule, not a hyperparameter: a new premium observation exists only when a period settles; deciding between settlements reads the same premium twice and pays twice |
| Advance every bar by one bar (`AVAILABLE = timestamp + 8h`) before any shift/join/filter | Binance stamps klines at open; a midnight row reports a close/premium known only 8h later. Assert stamps in {0, 8, 16} |
| Funding `calc_time` is NOT shifted | It is the settlement instant. Observability check in 03: corr(funding_rate, log return of the interval ending at its stamp) > corr with the following interval |
| Breadth measured at decision moments, floor `2 * max(top_k_grid)` = 20 | A ranking on 19 names needs many contracts quoting at once; breadth never reaches 20 on any of 4,382 development timestamps, so k=10 cannot fill both legs anywhere |
| Label on the price series, then join predictors | Premium index is absent at some settlements where the contract traded; labelling on a joined frame would null valid labels |
| Horizon = hours / 8 settlements (`fwd_ret_8h`: 1, `fwd_ret_24h`: 3); require `timestamp.shift(-h).over(symbol) == timestamp + 8h*h` | Never label across a hole; no eligibility filter before the shift |
| Binary label `(ret > 0).cast(Int8)`; 3-class at pooled development terciles p33/p67 | `when/then/otherwise` fabricates "down" for null windows; terciles cut on `_label_end < HOLDOUT_START`, pooled not per fold |
| Reconcile `labelled + tail + grid_hole == height` per label; direction null counts == primary null count | Silent failure modes caught by assertions that must balance, not prose |
| ESS via `effective_sample_size(dev, horizon=h, bar_col="slot")` over `h` return intervals | 8h label: N_eff == N (66,604); 24h: 22,201 of 66,550 (0.334 ~ 1/3). A closed-bar form returning N/2 for the one-period case is wrong |
| HAC (Newey-West) on IC series with `maxlags = 3` (one funding day) | Disjoint is not independent: the 8h label has zero overlap but non-zero serial dependence; horizon-derived bandwidth would be zero |
| Two folds, 2Y/1Y, fold 0 most recent; purge gap = label buffer; holdout 2024-25 read once | Integrated autocorrelation 37 periods -> at most 30 independent premium observations per contract per year; two folds is the binding constraint throughout |

### Features (03-04)

- Windows declared once in `features.windows` (8h bars) and bound: `premium_momentum {8h:1, 24h:3, 72h:9, 168h:21, 336h:42, 720h:90}`, `premium_volatility {3, 9, 21, 42}`, `premium_zscore`/`premium_dev_mean {7d:21, 14d:42}`, `premium_quantile {21, 42, 90}`, `premium_rsi {3, 9}`, `price_volatility {21, 42}`, `premium_persistence {21}`, `premium_regime {9}`, `funding_zscore`/`funding_half_life {42}`, `funding_change {3}`, `funding_cashflow {7d: 7}` (time window, days). Clips: zscore 10, vol_ratio 10, ar1 0.999, half_life [0.5, 100]. Longest window 90 sets the leading budget.
- Family register (lookback / lag / role): carry 43 / 0 / signal; mean_reversion 90 / 0 / signal; momentum 90 / 0 / signal; volatility 43 / 0 / state; cross_sectional 1 / 0 / state; regime 10 / 0 / state. State the timing contract before coding the feature; `warmup_audit` and the timing figure read these numbers. (Actual longest windows 42/90/90/42/9/1, consistent with lookback = window + 1, not stated explicitly.)
- Shared primitives (`case_studies/utils/feature_engineering.py`): `EPS = 1e-8`; `rolling_zscore = (x - mean) / std.clip(lower=EPS)`; `cross_sectional_percentile = rank(method="min") / (count + 1) * 100`; `trailing_volatility = rolling_std * sqrt(1095)` (`BARS_PER_YEAR = 365*24/8`).
- Half-life (`funding_half_life_14d`): AR(1) slope with 42-bar trailing moments, `rho` clipped +-0.999, `-ln2 / ln|rho|` clipped [0.5, 100]; guard the denominator on `_var > 0` (an EPS floor on a bps-variance sits inside the distribution body); `rho == 0` -> floor 0.5. Two wrong versions (`clip_variance`, `zero_to_window`) kept as switchable oracles and asserted against.
- RSI with fixed-window means of up/down changes, deliberately NOT Wilder's recursive `ml4t.engineer.features.momentum.rsi`: recursion propagates the first gap into every later value.
- `cum_positive_funding_7d`: time-window `rolling(period="7d", closed="right")` on the funding series' own clock, because Binance shortens intervals to 2h/4h and 7 days can hold > 21 settlements.
- Assembly order: per-symbol features -> drop rows with any null per-symbol feature -> keep timestamps with `len.over(timestamp) >= MIN_CROSS_SECTION = 2` -> THEN cross-sectional features (`premium_vs_median`, `premium_xs_zscore`, `xs_funding_dispersion`, `premium_rank`). Exclude raw OHLCV and `premium_index_close` (except as `premium_level`) from the matrix.
- Seals: `warmup_audit(trailing, {col: window}, entity="symbol")` on the pre-gate panel; `assert_values_agree(built.filter(< holdout), build_features(panel.filter(< holdout)), columns=feature_cols, keys=[timestamp, symbol])` (value vs null counts as a difference). `write_artifact(..., inputs={"load_crypto_perps": value_digest(prices), "load_funding_rates": value_digest(funding)})`.
- Model-based features: two leakage channels (conditioning set and parameters). Fix = refit schedule via `refit_boundaries(n_obs, burnin=500, refit_every)` and `walk_forward_feature(X, timestamps, burnin, refit_every, fit, apply, n_features, freeze_after=count(index < holdout_start), on_fit_error="skip")`, expanding window, every row's parameters from strictly earlier settlements; frozen at holdout. One value per settlement whichever fold selects it, so no fold column.
- GJR-GARCH(1,1), Student-t, `arch_model(returns*100, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="StudentsT")`, needs >= 100 returns; take `backcast` and `variance_bounds` from the TRAINING fit (arch's `fix()` re-derives them from the handed array, measured drift up to 0.19% on arch 8.0.0); advance one step `h_{t+1} = w + (a + g*1[e_t<0]) e_t^2 + b h_t` (library `h_t` discards the newest return); emit `garch_cond_vol = sqrt(forecast)/100`, `garch_asymmetry = g`, `garch_vol_zscore` (90-bar within-symbol z, clipped +-10). Refit every 21.
- HMM: observation per settlement `(xs_mean_funding_bps, xs_std_funding_bps)` over >= 2 contracts; `fit_hmm_restarts(train, n_states=2, random_state=0, n_restarts=10, tol=1e-4)` with k-means init, `covariance_type="full"`, `n_iter=200`, inside `threadpool_limits(1)`; `sort_states_by_variance` (state 0 calm, 1 stress); `filtered_state_probs` forward recursion (P(z_t | x_{1:t}), never `predict_proba`'s smoothed posterior). Refit every 63. Emits `hmm_regime_prob_calm/stress` broadcast to every symbol.
- Holdout fold for 04 from `build_holdout_cv`, `train_end = holdout_start - widest declared buffer (24H = 3 settlements)` so one fold serves every label.

### Evaluation and models (05-12)

- 05 settings: `MIN_CROSS_SECTION = 10`, `MIN_IC_PERIODS = 20`, `HAC_MAXLAGS = 3`, `IC_THRESHOLD = 0.005`, `N_QUANTILES = 5`, `REDUNDANCY_CUT = 0.7`. `quality_pass = coverage >= 0.90 and overall_unique >= 2` (asserted). Three different minimum cross-sections by design: 02 baseline uses half the median (8), 04 and 05 use 10; do not conflate.
- Triage rule: STOP on quality failure; REVISE "conditioning_not_cross_sectional" when IC undefined; PROCEED if BH-significant at 0.05; else PROCEED if `sign_consistency == 1.0 and |mean_ic| >= 0.005` (exploration arm); else REVISE "weak_univariate". Ledger decisions are records; all 44 features go to every model (screening is not selection).
- Linear: per fold median-impute and standardize on training only; Lasso/ElasticNet via `alpha_frac` (fraction of each fold's own `alpha_max`, so the grid is a path); Ridge alpha spans 10 orders of magnitude; `logistic_l1_CX` = saga, `tol=0.001`, `max_iter=200`. Count non-zero coefficients from `fitted_states()`: past `alpha_frac = 0.7` one column remains and IC is identical across configs (one model reported several times).
- GBM presets (`case_studies/config/lgb/`): `default_*` carry only `objective` and `seed: 42` (library defaults: 31 leaves, lr 0.1); `leaves_{7,15,31,63}_*` add `bagging_fraction 0.8, bagging_freq 1, feature_fraction 0.7, lambda_l1 0.5, lambda_l2 5.0, learning_rate 0.05, min_child_samples 50`; huber `huber_alpha_scale: 0.5` (0.5 x training-label std per fold). Leave NaN in place (tree routes missing; imputing invents an observation). Compare at fixed iteration; report (across-config spread) / (within-config checkpoint range).
- Every checkpoint is a registered prediction set: gbm 10, tabular_dl 8, deep_learning 20, linear 1. Collapsing to a per-config best is choosing after seeing the answer and flatters whichever family saves most often.
- Populations declared before the first fit (`freeze_official_model_population` -> `crypto-validation-predictions-v1`, 677 sets), immutable, superseded explicitly; identity binds device policy and seed (GPU kernels reorder float reductions).
- Sequence models: lookback 60, gap policy `exclude_windows_crossing_missing_expected_periods`, `batch_size 2048`, `dropout 0.1`; TCN receptive field `1 + 2(k-1)*sum(d)` must be >= lookback (61 >= 60); left-pad / right-trim for causality; batch norm pools across windows in a batch.
- DML (`case_studies/utils/causal.py::manual_dml_timeseries`): chronological folds with embargo from the label buffer; nuisances `HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42)`; residual-on-residual OLS; HAC bandwidth `max(h-1, n^(1/3))` capped at `n/2`; placebo block permutation within symbol, `block_size = max(buffer_steps, treatment_window_steps) = 42`, `n_placebo: 100`, p floor 1/101; `CAUSAL_RUNNER_VERSION = 2` is the identity's source component. Compare `spec["computation"]` only on re-run. Report beside, never among, predictive metrics.

### Strategy funnel (13-19)

- Equal weight is a measuring instrument: it contributes no information, so a difference between two baselines is a difference between two rankings. Each later stage changes exactly one spec field (allocation, risk, cost) against the same rankings, timestamps and funding; "a comparison that changes the model and the sizing measures neither."
- Funding is settled inside the engine, position-signed, BEFORE the same timestamp's fills (`economic_cashflows: {"funding": "position_signed_before_same_timestamp_fills"}`); `funding_rates_for_prices(prices)` semi-joins official settlements to the exact `(symbol, timestamp)` keys the price panel spans; `value_digest(funding_rates)` goes into `input_identity`. Reconstructing funding afterwards would create two series under one hash. Vectorized mode refuses funding; crypto runs the event engine.
- Decision clock on the panel's settlement index (`features/financial.parquet` unique timestamps with `slot`), not the calendar: the panel has a 57-day premium-index outage from 2021-08-27 that forward returns (prices only) do not. Four moments must agree with the label: information cutoff (pre-funding), fill (at funding timestamp, same bar), holding period (8h or 24h), next decision (`rebalance_step` 1 or 3).
- Read inputs by frozen name (`CandidateSet.one`, `OfficialPopulation`), never by registry stage/split filters, which resurrect retired generations (a rendered run once reported 346 backtests per rule after executing 262).
- Selection is validation backtest Sharpe (annualized at 365) at every stage; ties by identity hash; ruined paths last; never IC/AUC/log loss. On perps the IC-vs-Sharpe gap has a cause: funding is paid by whoever holds at settlement, so a signal that flips often pays repeatedly.
- Allocation advances ten distinct `(family, config_name)` pairs with `checkpoints_per_config=1` (the checkpoint is part of the configuration, not a knob). A sizing rule only redistributes what the ranking selected, so sizing comes after selection.
- Paired difference is the only honest stage increment: join on `(prediction_hash, entry_rule)` in 14, on `baseline_key` (spec minus `chapter`, `metadata.chapter`, `metadata.preset_path`, `strategy.risk`) in 15; compare on common support (`rank_returns_on_common_support`, `periods_per_year=365`); report `sessions_flattened`.
- Candidate sets admit only results whose `traded_folds` (non-zero days in `daily_returns.parquet` intersected with fold windows) equal every declared fold: a flat day books 0.0, so an allocator that sat out a losing year once "won" its stage.
- Overlay width 1 parent per label (14 x 10 would be 140 readings of one validation year). An overlay never adds a position; on a funding instrument it pays twice (the exit trade and the stopped funding stream). Thresholds fixed in setup.yaml: a stop calibrated on validation prices is a fitted parameter.
- Cost sensitivity selects nothing and prices the post-overlay winner (`selected_final_result` reads `crypto-final-validation-{label}`); cme_futures showed 1.209 pre-overlay vs 1.274 post-risk when the wrong configuration was priced. Breakeven = linear interpolation at the first sign flip; `None` if it never crosses ("still above zero at 50").
- Holdout: configuration fixed before 17; refit on `build_holdout_training_spec` (one fold, train end `holdout_start - 24H`, `_rekey_holdout_spec` recomputes `FOLD_DERIVED_FIELDS`); raise if the holdout training hash equals the validation one; `REPLACE_HOLDOUT=False` default, one readable generation per window, delete-and-rerun rather than "spent". 18's guard is the prospective `backtest_run_status(...).backtest_hash`, asserted equal after the run.
- Synthesis: pool across labels with `comparison_contract={"comparable_fields": HORIZON_DEPENDENT_PROTOCOL_FIELDS}` = `("cv", "feature_artifacts", "label_artifact")`; `compute_backtest_uncertainty` (stationary block bootstrap, `n_boot=2000`, `seed=0`, block = `max(rebalance_step, horizon floor)`, Lo SE, HAC, PSR via `deflated_sharpe_ratio` on one series) and `compute_cohort_metrics` (DSR raw / Marchenko-Pastur / effective-rank K, `expected_max_sharpe_*`, `min_trl_periods_*`, RAS with `rademacher_complexity(n_simulations=2000)`, optional White RC and CSCV PBO). Raise if `aligned_periods < shortest member` or cohort leader != registered selection.

## Market-specific guardrails

Timing and data
- Advance Binance bar-open stamps by 8h once, before any shift/join/filter; assert hours in {0, 8, 16}. Do not shift funding `calc_time`.
- Build labels on the perpetual's own bars; join the premium index afterwards. Never filter before shifting; require `label_end == t + h`.
- Fixed-row windows are wrong on a variable settlement clock (2h/4h intervals): use time-window rolling for funding sums.
- A sequence window of 60 adjacent rows can span > 60 settlements: use the declared gap policy, quote `eligible_rows` not panel height.
- One missing premium observation costs a whole window (90 rows); declare leading/interior/trailing causes in `quality_report` and print the unexplained residual (1,068 keys, 1.0%) rather than explain it away.
- Timestamp units differ: deep_learning prediction sets carry `us`, every other family `ms`; a join on mismatched dtype returns empty with no error. Cast at every join site with `on_clock_dtype`; `dt.replace_time_zone("UTC")` before casting a naive column. Never re-key the registry: `value_digest` is unit-sensitive and `training_hash` would move for every sequence run.
- Polars: `walk_forward_feature` burn-in is NaN, not null (`drop_nulls` keeps it) -> `pl.DataFrame(..., nan_to_null=True)`; `is_in(series.implode())`; pass an explicit schema for columns null for 300 rows then string; `pl.concat(how="vertical_relaxed")` for Null/Int64 checkpoint columns.

Features and leakage
- No warmup fallbacks (`warmup_audit`); no whole-sample fitted transforms (`assert_values_agree` on a holdout-withheld rebuild); no raw close or `premium_index_close` beside the label.
- Cross-sectional stats only after the null gate and `>= 2` names; percentile = `rank / (count + 1) * 100` (top of 3 and top of 19 must not both read 100 on an unbalanced panel).
- EPS floors must be in the variable's units; guard on positivity for bps-scale variances.
- Model-based features: refit schedule with seals (`n_refits <= len(scheduled)`, `last_fit_ends < holdout_start`); test by deleting the last year and checking every surviving value is unchanged. Filtered, not smoothed, HMM probabilities; states sorted by covariance trace; `threadpool_limits(1)` (five thread counts gave five log-likelihoods). GARCH backcast/bounds from the training fit; advance one step. `freeze_after` at the holdout; append the holdout fold and assert it has rows with every column non-null somewhere.
- Key temporal artifacts on `(timestamp, symbol)` only; assert `load_modeling_dataset(...).temporal_by_fold is None` and feature count 39 + 5 = 44.

Evaluation
- IC only on settlements with >= 10 contracts (186 of 2,189 validation decisions fall below); report the count scored. `.filter(ic.is_finite())`: market-level features (regime probs, `funding_session`, `xs_funding_dispersion`) have no within-settlement spread and return NaN.
- Sort before `compute_ic_hac_stats` (polars `group_by` order is arbitrary); use `cross_sectional_ic_series` which sorts.
- BH at 0.05 across the searched set (40 features in 05, 5 in 04); count identical-ordering features (4 premium features = 1 ordering) as conservative over-counting.
- Judge `full_coverage` within label (`fwd_ret_24h` runs out of window earlier); AUC undefined on 540 of 2,189 timestamps where all contracts move one way.
- Drop `superseded_members(study, member_kind="prediction")`: `identity_status == "current"` names the schema version, not lineage. Do not intersect keys across families; score each row on its own fold's declared settlements.

Backtest funnel
- `FundingSettlementLedger.metrics()` raises unless `funding_settlements == rate_count`; settle each timestamp exactly once; funding digest in `input_identity` so funding data drift is a different result.
- `holding_slots` raises on a decision at a timestamp the panel does not hold. Do not assert `advance == step` (tautology for cross-sectional families, false by construction for sequence families); report `panel_settlements_skipped` and rely on `complete` plus `check_prediction_coverage(..., decision_axis=CLOCK timestamps, raise_on_gap=False)`. A `gap_policy` excuses only `missing_sessions`; raise `CoverageError` on `missing_fold`, `undeclared_fold`, `out_of_window`, `unaccounted_window`. `ValueError` if `PREVIEW_FOLDS and CANONICAL_RUN`.
- `get_entry_schemes_for` drops infeasible k (`2k > tradeable` or `k >= tradeable`), raises if none remain or if the ranked width is narrower than the panel; print requested vs run. Per-decision fill is `min(k, n/2)`: top-5 narrows at 186 decisions, top-3 at 93; a name can overstate the book.
- `CANONICAL_RUN = EXECUTION_TIER == "canonical"` (tier alone; a workspace only decides where results go). `STORAGE_ROOT = study.storage_root(tier)`, not `study.root`. Preview tier cannot create official populations; preview rankings require `num_trades > 0` (a never-trading book reports Sharpe 0.0 and wins); cap previews per label with `group_by("label").head(limit)`. `top_n_cap`: `0 -> None` (all), negative -> ValueError.
- Risk specs must be `{"name", "position_rules": [{"type", "threshold"|"bars"}]}` via `as_risk_spec`; a flat mapping installs nothing yet hashes as a distinct result (56 such rows in a retired generation). Exclude `overlay_without_rule`; raise if every control is inert. Do not filter overlays by equal `traded_folds` (removes exactly the controls that fire); count flattened sessions as `(overlay == 0.0) & (baseline != 0.0)`.
- Load prices with the allocator's warmup at 14/15/16/18 (`strategy_warmup_periods` -> `load_backtest_prices_for(warmup_periods=...)`; moment allocators 240, score/conformal 0) or the first weeks trade on median-imputed fallback weights.
- Cost siblings: `_non_cost_projection` must equal the chosen result's; carry `risk` explicitly; read cells by `cost_runs` hashes and expect exactly `len(labels) * len(cost_grid)` = 44 rows. The grid is uniform (cannot reproduce the maker/alt split); read it as a sensitivity axis. Funding is not on the cost axis: report `funding_pnl` beside execution cost.
- Holdout: training ends `holdout_start - widest buffer (24H)`, refused to default to zero; rebuild `input_identity` from holdout frames (never clone validation digests); pass `funding_rates_for_prices(prices)` to `run_backtest`; `resolve_solvent_carrier` raises on `max_drawdown <= -1.0` (a long-short book with no margin call compounds through zero) and on a retired conformal `calibration_version != "walk_forward_v3"`; never fall through to the runner-up (fix the sweep or pin in `CARRIER_PINS`). Lineage by `resolve_solvent_carrier(...)["holdout_backtest_hash"]`, never by `stage='holdout'` alone; raise on an empty holdout query.
- Supersedes etiquette: leave `SUPERSEDES*` empty on first runs; when a refusal names a predecessor hash, paste it; resolve through `population_supersedes` / `candidate_set_supersedes` / `causal_supersedes` / `supersedes_for_run`; `"live"` resolves to the current tip for backtest populations. `CandidateSet.create` and `OfficialPopulation.create` refuse preview members.
- Do not infer which configurations advance without running the notebook: five case studies use `resolve_best_predictions` and four restate the rule differently (agent-workspace#1204).
- Known limitations, declared not searched: fixed fee tier per symbol (`cost_tier_alt` does not follow tier changes and does not set backtest costs); uniform controls across contracts; no portfolio-level controls; one allocator lookback; survivorship in the 19-name list.

## Results and lessons

Feasibility and labels (01-02)
- Breadth 2 -> 7 (spring 2020) -> 16 (end 2020) -> 19 (2023); never 20 on any of 4,382 timestamps. Median |8h move| 138 bps = 17x the taker round trip (8 bps); 96.3% of moves exceed it. Integrated autocorrelation 37 periods -> <= 30 independent premium observations per contract per year (of 1,095). Kill conditions: fees erase gross return; premium stops predicting by next settlement; venue changes funding formula/cap/interval; equal-weight cross-section beats it on Sharpe.
- `fwd_ret_8h` dev std 0.03417, skew 1.31; `fwd_ret_24h` std 0.06424 (~sqrt(3) wider), skew 8.20; 24h lag-1 autocorrelation 0.673, lag-3 -0.020. Baseline premium z-score IC -0.0352 (reversal), naive t -7.90, Newey-West (3) t -7.92 over 3,655 settlements. Coverage 99.98% / 99.93%; no winsorization.

Features and evaluation (03-05)
- 39 features, 99,877 rows, digest `873623a4af13b3fc`; 8,421 of 108,298 bars discarded, mostly from 537 missing premium observations; 23 redundancy clusters at 0.7 (largest a 7-column normalized premium block). Half-life distribution piles at the lower clip.
- 04: `garch_cond_vol` is the only model-based feature clearing BH, with negative IC (agitated perps underperform next 8h) beside `price_vol_14d`; vol z-score and asymmetry do not clear; regime probabilities unrankable by construction.
- 05: 40 of 44 scored on <= 2,003 of 2,189 settlements; 30 clear BH; 34 same sign in both folds; largest |IC| = -0.041 on `price_vol_7d`; the twenty strongest are all negative; HAC slightly strengthens the strongest (negative dependence at the funding cycle). Ledger: 32 PROCEED (30 FDR, 2 exploration), 12 REVISE (4 unrankable). 45 pairs |rho| > 0.7.

Models (06-12)
- Linear: direction labels reach an order of magnitude higher IC than return labels (`fwd_ret_8h` best a fraction of a thousandth, most of the grid below zero; `fwd_dir_8h` whole grid above zero at a few hundredths, AUC a little above 0.5). The L1 path keeps a volatility column, never a premium column; single-column configs post the largest-magnitude IC, negative.
- GBM closes the size-vs-direction gap (every config above zero at end of training; weakest GBM return config beats anything linear reached). Loss ordering Huber > MAE > MSE on both return labels (`fwd_ret_8h` excess kurtosis ~25). Median peak checkpoint 100-150 of 500; median config ends below where it started; checkpoint choice moves the answer by a third to two thirds of the model choice. Author's summary: direction, not magnitude; 400 candidates on two years.
- 08-11 report nothing in-notebook (selection deferred to 13); DML `refutation_n_successful` is None on this run so no `refutation_class` is published.

Funnel (13-16)
- Population 677 sets (262 / 242 / 86 / 87 by label); 2,031 signal backtests; clock: `panel_settlements_skipped = 0` on all six grids, 2,189 decisions (8h) / 2,187 (24h); breadth < 10 names for 186 decisions.
- Signal baselines: direction labels lose everywhere (`fwd_dir_8h` median Sharpe -2.7 to -3.1, max -0.92, 0 above zero, turnover 4.1-4.5, 9-14k trades; `fwd_dir_8h_3c` median -1.6 to -1.8). `fwd_ret_8h` median -1.9 to -2.3, max 0.49 (quintile), 11 of 262 above zero. `fwd_ret_24h` is the only label with positive medians: top-3 median 0.22 (max 1.53, 151/242 above zero, turnover 2.83), top-5 median 0.14 (max 1.31). Distributions straddle zero in every panel; no entry rule or label separates.
- Allocation: 720 backtests, 720 paired, 0 unpaired. Median paired Sharpe change: mvo_ledoit_wolf +0.061 (64/120 improved, turnover 1.38), conformal_weighted -0.018 (turnover 0.044), score_weighted -0.039, inverse_vol -0.132, risk_parity -0.188, hrp -0.324 (5/120 above zero). Best/worst change about +-2.0-2.3 for every allocator: the spread dwarfs the median effect.
- Overlay parents: `fwd_ret_24h` linear `enet_f0.03` equal weight 1.556 (DD -0.467); `fwd_ret_8h` linear `lasso_f0.2` equal weight 0.487 (DD -0.707); `fwd_dir_8h` gbm `default_binary` + mvo_ledoit_wolf 0.399; `fwd_dir_8h_3c` gbm `default_multiclass` + mvo 0.485. 56 overlays, 730 shared sessions, 24 flattened at least one session, 5 inert. Tight stops are destructive: on `fwd_ret_8h` `trailing_1pct` Sharpe -3.39 (change -3.87, +1,060 trades), `trailing_2pct` -1.93, `trailing_5pct` -1.06, `stop_loss_3pct` -0.81 (+264 trades). Stage: best 1.668, median 0.31, worst -3.39, 39 of 56 above zero. An overlay helped three labels and none helped the fourth.
- Cost curves (Sharpe at 0 bps -> breakeven): `fwd_ret_8h` 1.13 -> 8.6 bps; `fwd_ret_24h` 2.07 -> 19.4 bps; `fwd_dir_8h` 0.81 -> 7.1 bps; `fwd_dir_8h_3c` 0.97 -> 7.8 bps. Production 5 bps leaves ~2-3 bps margin for three labels, ~14 bps for the 24h winner; at 50 bps every label is -3.9 to -7.8. Execution cost across the grid $110k-$150k on $100k capital. Oddity: `fwd_dir_8h` rises from 0.81 at 0 bps to 1.18 at 1 bps with trades 3,829 -> 2,080 ((inference) thresholds and cost-dependent equity change the trade set). Funding P&L is not flat across the grid despite the prose: ranges $2.4k-$50k ((inference) notional scales with cost-dependent equity).

Selection and holdout (19, 17-18)
- Pool 2,807 (signal 2,031 best 1.556 / median -1.55 / 467 above zero; allocation 720 best 0.901 / median -0.33; risk 56). Selected: `fwd_ret_24h`, linear `enet_f0.03`, checkpoint `final`, `quintile_long_short`, equal weight, overlay `time_exit_20`; Sharpe 1.668, max DD -0.469, total return +416% on $100k, 2,746 trades; funding P&L +$21,932 over 38,473 settlements; commission $95,653; slippage $23,913.
- Confidence (rendered): 95% bootstrap CI [0.42, 2.87]; PSR p = 0.0049; DSR raw K -0.131, MP -0.048, ER -0.051; RAS 1.18; min-TRL (ER) 298 periods. (Prose quotes an earlier generation: 1.57, [0.27, 2.80], p 0.0107, DSR -0.15, RAS 1.09, min-TRL 374.) "The largest of 2,807 draws lands near 1.6 whether or not any of them has an edge." Every uncorrected statistic said real; every DSR said not.
- Holdout: refit training `918aeaf5a811`, prediction set `9adfb97d8db7` (40,821 rows, 2,190 timestamps, 19 names); backtest `ee7c119e1009`: Sharpe -0.832 over 731 periods, CAGR -44.8%, max DD -82.26%, win rate 47%, vs validation 1.668. Read the sign, not the magnitude: validation carries the selection maximum, the holdout its own sampling error, and the drawdown reflects leverage and the absence of any stop beyond `time_exit_20`.
- Lessons: (1) on 19 names and two folds the within-panel Sharpe spread (+-2) is wider than any model, allocator or overlay effect; (2) interval and p-value answer "is this series non-zero", deflation answers "is the best of K better than the best of K worthless strategies", and only the second matched the holdout; (3) RAS penalizes class complexity, DSR penalizes draws; on a pool this size the count dominates; (4) a strategy whose return is mostly funding is a carry strategy and flips with the rate's sign; here funding was +$22k against $120k of execution cost, so the book was a price bet paying to trade.

## How to adapt this pattern to a new dataset

1. Write `setup.yaml` first: universe, cadence (`decision.snapshot` / `execution_delay`), labels with buffers and `rebalance_step = ceil(horizon / bar)`, fold geometry, holdout window, `periods_per_year` for the calendar, fee schedule, rebalance thresholds, sweep grids (`top_k_grid`, `quantile_grid`, allocators, `cost_grid_bps`, `risk_controls`), feature windows and family register with lookback/lag, model-based burn-in and refit cadences. Notebooks read it; never type a window or cash constant locally.
2. Establish the availability clock: find where the provider stamps bars (open vs close) and advance once before any shift/join/filter; assert stamps land on the expected grid. Check each exogenous series' observability (correlate with the interval ending vs following its stamp).
3. Feasibility before labels: breadth at decision moments vs `2 * max(top_k)`; exceedance of |move| vs round trip; ACF -> integrated autocorrelation time -> independent observations per year; `generate_cv_splits` with the declared buffer. Record kill conditions.
4. Labels on the price series alone: exact-grid forward return, null-propagating direction labels, development-tercile classes, full reconciliation (`labelled + tail + grid_hole == height`), diagnostics cut on the label endpoint, `effective_sample_size` over `h` return intervals, HAC IC with bandwidth = one trading day in bars.
5. Features from the declared register: per-symbol -> null gate -> minimum cross-section -> cross-sectional; `rank / (count + 1)`; time-window rolling on irregular clocks; fixed-window RSI across gaps; `warmup_audit`; holdout-withheld rebuild with `assert_values_agree`; `quality_report` with declared leading/interior/trailing budgets; write with `write_artifact` and input digests.
6. Model-based features on a refit schedule (`refit_boundaries`, `walk_forward_feature`, `freeze_after` at holdout), keyed `(timestamp, entity)` with no fold column; filtered HMM probabilities, variance-sorted states, pinned thread pool; GARCH seeds carried from the training fit, one-step-ahead variance; BH across the model-based set.
7. Univariate evaluation as a ledger, not a filter: `MIN_CROSS_SECTION`, `MIN_IC_PERIODS`, HAC, fold sign consistency, BH at 0.05, quintile profile, redundancy clusters; keep all features in the training frame.
8. Declare the model menu per label (`config/training/{label}.yaml`), omit configurations the runner cannot execute (classification x sequence, classification x DML), freeze the official population before fitting, register every checkpoint, thread `SUPERSEDES_*` through parameters.
9. Baseline grid: equal-weight top-k and quantile long-short for every prediction set, infeasible k dropped by `get_entry_schemes_for`; decision clock on the feature panel; coverage gate with the panel as `decision_axis`; venue cash flows (funding, borrow, dividends) settled inside the engine and digested into `input_identity`.
10. Funnel with declared widths: allocation on top-N distinct configurations (one checkpoint each) paired to baselines; overlays on the top-1 with fixed thresholds, paired on common support; cost curve on the post-overlay winner with every non-cost field audited equal; admit only results that traded every fold.
11. Holdout once: refit with training end = `holdout_start - widest buffer`, one readable generation, rebuilt input digests, same cash flows; guard on the prospective backtest hash.
12. Synthesis: freeze the pool, then report K distinct trials, selected Sharpe, bootstrap CI, PSR, expected max Sharpe under the null, DSR raw/MP/ER, RAS, min-TRL, cash-flow attribution and the holdout sign. If DSR < 0 the selection is not evidence regardless of the CI.

## Related references

- `chapters/06_strategy_definition.md` -- feasibility counts, search accounting, kill conditions, run log (01).
- `chapters/07_defining_the_learning_task.md` -- forward-return labels, ESS/average uniqueness, HAC IC, BH-FDR, triage (02, 05).
- `chapters/08_financial_features.md` -- timing contracts, warmup audit, redundancy and persistence (03).
- `chapters/09_model_based_features.md` -- GARCH and HMM on a refit schedule, filtered vs smoothed (04).
- `chapters/11_ml_pipeline.md` -- per-fold imputation/standardization, regularization paths, populations (06).
- `chapters/12_gradient_boosting.md` -- LightGBM presets, checkpoints as models, loss choice on heavy tails (07).
- `chapters/13_dl_time_series.md` -- TabM, NLinear/LSTM/TCN sequence contracts, gap policy (08-10).
- `chapters/15_causal_estimation.md` -- walk-forward DML, block-permutation placebo (11).
- `chapters/16_strategy_simulation.md` -- decision clock, equal-weight baseline, engine cash flows (13).
- `chapters/17_portfolio_construction.md` -- six allocators, paired differences, traded-fold admission (14).
- `chapters/18_transaction_costs.md` -- cost curve, breakeven, funding vs execution (16).
- `chapters/19_risk_management.md` -- position overlays, common-support pairing, inert-control guard (15).
- `chapters/20_strategy_synthesis.md` -- holdout protocol, bootstrap CI, PSR, DSR, RAS, min-TRL (17-19).
- `chapters/03_market_microstructure.md` -- perpetual mechanics, premium index, funding formula.
- `chapters/02_financial_data_universe.md` -- availability clocks, unbalanced panels, survivorship.
- `case_studies/cme_futures.md` -- ARIMA refit-schedule precedent; the 1.209 vs 1.274 cost-stage example.
- `case_studies/etfs.md` -- HMM restart/tolerance measurements; same DML spec check.
- `case_studies/fx_pairs.md` -- HMM tol 1e-4 chosen independently; inline shortlist rule.
- `case_studies/nasdaq100_microstructure.md` -- HAR refit per bar; warmup over-allocation incident.
- `case_studies/us_firm_characteristics.md` -- retired conformal calibration incident; zero-horizon labels.
- `case_studies/sp500_options.md` -- `max_samples` cap flipping a DML sign.
- `libraries/ml4t_backtest.md` -- engine, `ml4t.backtest.risk` (StopLoss, TrailingStop, TimeExit, RuleChain), broker hooks the funding ledger patches.
- `libraries/ml4t_diagnostic.md` -- `compute_ic_hac_stats`, `cross_sectional_ic_series`, `benjamini_hochberg_fdr`, `deflated_sharpe_ratio`, `effective_number_of_trials`, `rademacher_complexity`.
- `libraries/ml4t_engineer.md` -- `percentile_rank_features`; why `momentum.rsi` is avoided across gaps.
- `libraries/ml4t_data.md` -- `load_crypto_perps`, Binance Vision funding archives.
- `guardrails.md` -- the cross-cutting leakage, multiple-testing and population rules this study instantiates.
- `decision_rules.md` -- selection on backtest Sharpe, funnel widths, DSR < 0 rule.
- `workflow.md` -- stage ordering 01-19 and what each writes.
- `evidence.md` -- the headline numbers above in the cross-study table.
- `companion_repo.md` -- `case_studies/research`, `case_studies/utils`, registry layout, `RUN_LOG.md#identity`.
- `glossary.md` -- shared vocabulary.

Further reading
- Binance Vision USD-M monthly funding-rate archives (`https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{YYYY-MM}.zip`).
- Newey-West HAC standard errors; Benjamini-Hochberg FDR; Lo / Lopez de Prado Sharpe standard error, probabilistic and deflated Sharpe ratio, minimum track-record length; Rademacher-adjusted Sharpe; White's reality check; CSCV probability of backtest overfitting; stationary block bootstrap; Ledoit-Wolf shrinkage; hierarchical risk parity; split conformal prediction; GJR-GARCH; Gaussian HMM (hmmlearn); double machine learning.
- Repo issues: ml4t/agent-workspace#396 (declared-but-unrunnable causal menu), #740 (sklearn floor 1.7), #1204 (shortlist-rule divergence).

## Glossary

- **Perpetual future** -- futures contract with no expiry, anchored to spot by funding rather than delivery.
- **Funding rate / settlement** -- 8-hourly payment between longs and shorts proportional to the perp-index gap; positive rate = longs pay shorts; a transfer, not a venue fee.
- **Premium index** -- exchange-published perp-vs-spot gap; `premium_index_close` at bar close; its outage creates the 57-day panel hole.
- **Availability clock** -- bar-open timestamp advanced one bar so it names when the bar is known.
- **Decision clock / slot** -- the panel's numbered settlement index on which decisions sit (holes preserved), as opposed to the calendar.
- **Rebalance step** -- schedule slots between decisions so holdings do not overlap (1 for 8h labels, 3 for 24h).
- **Breadth** -- contracts quoting at a decision timestamp; floor = 2 x largest top-k.
- **Exceedance curve** -- fraction of |moves| at least as large as each threshold; round trip = fee paid twice.
- **Integrated autocorrelation time** -- `1 + 2 * sum` of initial positive ACF pair sums; periods per independent observation.
- **Label buffer / purge gap** -- stretch between training end and evaluation start covering the horizon (8H primary, 24H widest).
- **Average uniqueness / N_eff** -- weighting of a row by the share of its forward window no concurrent label spans.
- **IC** -- per-settlement Spearman rank correlation between a score and the forward return, averaged over settlements.
- **HAC / Newey-West** -- serial-dependence-robust standard error; bandwidth in settlements (3 = one funding day).
- **BH-FDR** -- Benjamini-Hochberg control of the expected false-discovery share among features declared significant.
- **Triage ledger** -- per-feature PROCEED/REVISE/STOP record; a record, not a filter.
- **Redundancy cut** -- |Spearman| threshold (0.7) above which two features count as one ordering.
- **Timing contract** -- a family's lookback and lag declared before coding; **warmup audit** checks no value precedes its window.
- **Seal** -- executable assertion that a construction reads neither the holdout nor the future.
- **Refit schedule / burn-in** -- burn-in (500), fit, emit, refit every k (21 GARCH, 63 HMM), expanding window, frozen at holdout.
- **Filtered vs smoothed probability** -- P(z_t | x_{1:t}) vs P(z_t | x_{1:T}); only the filtered one is a usable feature.
- **GJR-GARCH(1,1)** -- GARCH with an extra term for negative shocks; persistence = alpha + beta + gamma/2.
- **Population (OfficialPopulation)** -- named, immutable, declared-before-fit set of result hashes; superseded explicitly.
- **Candidate set** -- named, immutable set of backtest identities admitted to a comparison (`crypto-signal-{label}`, `crypto-signal-allocation-{label}`, `crypto-final-validation-{label}`).
- **Generation / supersedes** -- a re-declared population or set must name the one it replaces.
- **Checkpoint** -- registered prediction identity at an iteration/epoch (gbm 10, tabular_dl 8, deep_learning 20, linear 1); part of the configuration.
- **Complete** -- registry flag: written-key digest equals declared-eligible digest with no bad scores.
- **Coverage gate** -- `check_prediction_coverage` against declared folds/sessions; gap kinds `missing_sessions`, `missing_fold`, `undeclared_fold`, `out_of_window`, `unaccounted_window`.
- **Gap policy** -- `exclude_windows_crossing_missing_expected_periods`, declared by sequence families.
- **Execution tier / workspace** -- canonical (shared registry, named pools) vs preview (isolated, reduced); a workspace only decides where results go.
- **Entry rule / scheme** -- `equal_weight_top_k` (ew_top3, ew_top5) or `quintile_long_short`; **feasible** = `2k <= tradeable`, distinct from filled at every decision.
- **Equal-weight baseline** -- `stage='signal'` 1/N book; the measuring instrument.
- **Allocator / moment-based allocator** -- sizing rule on selected positions; inverse_vol, risk_parity, hrp, mvo read `allocator_lookback` bars (warm-up 240; score/conformal 0).
- **Conformal width** -- quantile of past absolute errors per contract; `1/width` sizing, lag `h = max(1, k)`, `alpha=0.2`, `calibration_version="walk_forward_v3"`.
- **Traded folds** -- folds with at least one non-zero return day; admission requires all declared folds.
- **Paired difference / common support** -- variant minus its own baseline differing in one spec field, on the exact timestamp intersection.
- **Sessions flattened** -- sessions where the baseline booked a return and the overlay exactly zero.
- **Inert control** -- overlay leaving Sharpe, drawdown and trades unchanged.
- **Cost level / breakeven / turnover** -- round-trip bps split half commission half slippage; interpolated zero crossing; fraction of book replaced per rebalance.
- **Carrier / solvent** -- validation rank-1 configuration downstream notebooks run; `max_drawdown <= -1.0` is insolvent.
- **Holdout generation** -- one refit or one backtest readable on the holdout window at a time.
- **PSR / DSR / RAS / min-TRL** -- probabilistic Sharpe p-value for one series; deflated Sharpe vs expected max of K trials (raw, Marchenko-Pastur, effective-rank K); Rademacher-adjusted lower bound; periods to reach 5% significance.
- **`HORIZON_DEPENDENT_PROTOCOL_FIELDS`** -- `("cv", "feature_artifacts", "label_artifact")`, the only fields allowed to differ across labels in one pool.
- **`input_identity` / `economic_cashflows`** -- digests of data a run read (prices, funding); spec field `funding: position_signed_before_same_timestamp_fills`.
- **alpha_frac / huber_alpha_scale** -- L1 penalty as a fraction of the fold's alpha_max; Huber threshold as 0.5 x training-label std.
