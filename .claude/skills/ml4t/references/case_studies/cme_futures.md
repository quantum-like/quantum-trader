# Case study: CME futures (30 products, daily, fwd_ret_5d)

> Daily Databento settlements on 30 CME futures across 7 sectors (2011-2025), traded as a weekly long-short carry-ranked cross-section with Friday-close decisions and Monday-open fills. The market teaches three things the equity case studies cannot: a futures return splits into spot and roll so two price series (`adj_close` for returns, `raw_close` for curve quantities) do two different jobs; cross-sectional ranking across incomparable return distributions (treasuries vs energy, 6.8x label std) is itself a modelling choice; and fitted features (ARIMA, HMM) leak through their estimation window unless bounded by a refit schedule rather than a walk-forward fold. Headline lesson (README): portfolio Sharpe comes from magnitude at the top of the cross-section, not from average IC, so the family that ranks best on IC and the family the selection rule carries need not be the same one. The holdout is two years of weekly decisions, "a window short enough that the interval around anything it estimates is wide by construction." Repo path: `case_studies/cme_futures/`.

## Setup contract

Source of truth: `case_studies/cme_futures/config/setup.yaml` (`strategy_id: cme_futures`, `setup_version: v1`). Ch16-19 notebooks read execution values via `get_backtest_config()`; never declare a local `INITIAL_CASH` or `share_type`. Changing `initial_cash` or `share_type` invalidates every `backtest_hash`.

| Area | Key | Value | Why |
|---|---|---|---|
| Universe | `universe.product_groups` | equity_index [ES NQ YM RTY]; treasuries [ZN ZB ZF ZT]; energy [CL NG HO RB]; metals [GC SI HG PL]; currencies [6E 6J 6B 6A 6C 6S]; agriculture [ZC ZS ZW ZM ZL]; livestock [LE HE GF]; `n_products: 30` | Front-month continuous contracts, ratio back-adjusted; loader carries tenors 0, 1, 2 |
| Universe | `universe.excluded_sessions` | `[2020-02-28, 2014-06-13]` | One clearing venue's settlement file missing (18 and 12 products respectively; the two sets partition the universe). Eleven stage-03 levels are within-date percentiles; "a percentile taken over one venue's products is not comparable with one taken over thirty" |
| Decision | `decision` | `cadence: weekly_friday_close`, `snapshot: settlement_price`, `execution_delay: monday_open` | Decision date = last session of each ISO week (Thursday on holiday weeks) |
| Execution | `execution.initial_cash` | `10_000_000` | Per-position budget 5% x cash at k=10 long-short = $500k clears ES (~$347k holdout notional); NQ Dec 2025 (~$513k) is the one residual integer-rounding footnote |
| Execution | `execution.share_type` | `integer` | Futures contracts are whole numbers: `target_notional / contract_notional -> integer contracts` |
| Execution | `execution.allocator_lookback` | `63` | 3 months of daily bars, applied uniformly to inverse_vol, risk_parity, hrp, mvo_ledoit_wolf; `allow_leverage` lives in `config/backtest/base.yaml` |
| Mapping | `mapping` | `class: long_short_carry_rank`, `position_state_space: long_short`, `entry_logic: rank_by_carry_or_momentum`, `sizing: equal_risk_or_notional` | |
| Costs | `costs` | `class: material`, `components: [commission, spread, roll_slippage]`, `commission_per_contract: 2.0`, `spread_ticks: {liquid: 1, illiquid: 2}`, `illiquid_products: [LE, GF, ZL]` | Live cattle, feeder cattle, soybean oil quote wider |
| Backtest | `backtest.rebalance.default` | `min_weight_change: 0.005`, `min_trade_value: 100.0`; `benchmark: {0.0, 0.0}` | Skip a rebalance when weight change < 0.005 AND notional < $100; benchmark profile disables both so full-universe 1/N rebalances at all |
| Sweep | `backtest.sweep.top_n_predictions` | `signal: 0` (all), `allocation: 10`, `cost_sensitivity: 1`, `risk_overlay: 1` | Read via `get_top_n_predictions(case_study, stage)`; `expensive_allocators_skip: false` |
| Sweep | `backtest.sweep.top_k_grid` | `fwd_ret_5d: [5, 10]`, `fwd_ret_21d: [5, 10]` | Cap 10 = floor(30/3); k=20 long-short = 40 positions from 30 products -> ~zero trades |
| Sweep | `backtest.sweep.allocators` | score_weighted, inverse_vol, risk_parity, mvo_ledoit_wolf, hrp, conformal_weighted | **No equal_weight** (see guardrails: hash collision with the Ch16 baseline) |
| Sweep | `backtest.sweep.cost_grid_bps` | `[0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]` | All-in bps, split 50/50 commission/slippage |
| Sweep | `backtest.sweep.risk_controls.position` | stop_loss 3/5/10/15%; trailing_stop 1/2/3/5/10/15/20%; time_exit 10/20/40 bars (14 rules) | No `risk_controls.portfolio` declared |
| Evaluation | `evaluation` | `n_splits: 5`, `train_size: 8Y`, `val_size: 1Y`, `holdout_start: 2024-01-01`, `holdout_end: 2025-12-31`, `calendar: CME`, `periods_per_year: 252` | Last validation window ends 2023-12-21; purge = label horizon |
| Labels | `labels` | `primary: fwd_ret_5d`, `buffer: 5D`, `variants: [fwd_ret_21d]`, `variant_buffers: {fwd_ret_21d: 21D}`, `rebalance_step: {fwd_ret_5d: 1, fwd_ret_21d: 3}` | ceil(21/7)=3 schedule slots so holding periods do not overlap in the vectorized backtest |
| Modeling | `modeling.gbm` | `libraries: [lightgbm]`, `preset: default`, `device: cpu`, `max_bin: 255` | 63 is LightGBM's GPU default and had been carried over, quantizing every CPU fit into a quarter of the bins |
| Modeling | `modeling.latent_factors` | `persistent_entities: true`, `device: cuda` (declared, not inferred), `model_kwargs.sdf.checkpoint_epochs: [256, 512, 768, 1024]`, `beta_checkpoint_epochs: [256]`, `beta_default_checkpoint: 256` | Device enters `computation.runtime` inside the hashed identity; 10b uses cuda, 10a overrides to cpu |
| Model-based | `model_based.arima` | `burnin: 252`, `refit_every: 21`, `order: [2, 0, 1]` | Declared order replaces AutoARIMA (34 orders in 240 searches); d=0 by construction on a bounded z-score |
| Model-based | `model_based.hmm` | `n_states: 2`, `burnin: 504`, `refit_every: 63` | Backwardation/contango chain on book-average carry; burn-in paid once per notebook |
| Causal | `causal` | `treatment: carry_pct`, `treatment_window: 1`, `confounders: [vol_21d, momentum_composite, carry_rank]`, `method: walk_forward_dml` | Treatment window derived from the formula (two same-timestamp prices); label buffer sizes the placebo block |
| Engine | `config/backtest/base.yaml` | `allow_short_selling: true`, `allow_leverage: true`, `execution_price: open`, `execution_mode: next_bar`, commission `percentage 0.0005`, slippage `percentage 0.0002`, `calendar: CME`, `timezone: UTC`, `data_frequency: daily` | |

**Features register** (`features.*`): windows momentum [21, 63, 126, 252] (each with a Sharpe twin), short_return 5, volatility [21, 63, 126], skip_recent 21, skip_start 252, carry_smoothing 21, carry_zscore [63, 126], carry_momentum [5, 21], trend_sign [63, 126, 252], moving_average [21, 50, 200], high 252, low 126, rsi 14, yang_zhang 21, variance_ratio {horizon: 5, window: 63}. `ranked` maps source -> percentile name (ret_21d->mom_rank_21d ... carry_pct->carry_rank, curve_curvature_norm->curvature_rank, vol_21d->vol_rank, rsi_14->rsi_rank). `composites`: momentum_composite = mean(mom_rank_63d/126d/252d); sharpe_composite = mean(sharpe_rank_63d/126d/252d). `thresholds`: carry_regime_band 0.01, signal_long 60, signal_short 40, roll_proximity_day 25. `seasonal_sectors: [agriculture, energy]`. Eight `families` (term structure lookback 146; momentum, risk-adjusted momentum, trend and range, cross-sectional position, composite 252; volatility and regime 126; calendar 1), every `lag: 0`, each with role signal/state, hypothesis and failure_mode.

**Training menus** (`config/training/fwd_ret_5d.yaml` = `fwd_ret_21d.yaml`): linear ols + ridge_a{0.001 .. 1e7, 11 values} + lasso_f{0.5, 0.2, 0.08, 0.03, 0.015, 0.35, 0.7, 0.85} + enet_f{same 8} = 28; gbm {default, leaves_7, leaves_15, leaves_31, leaves_63} x {mse, mae, huber} = 15; deep_learning nlinear, lstm_h64; tabular_dl tabm_s, tabm_m, tabm_l; latent_factors pca, sdf; causal_dml dml. Presets live in `case_studies/config/{ols,ridge,lasso,elastic_net,lgb,tabm,lstm,nlinear,pca,sdf,dml}/`.

**Preset and threshold defaults** (read from `setup.yaml` and the preset directories; never retype in prose):

| Component | Defaults |
|---|---|
| GBM `default_*` | `lightgbm.Booster`, `max_iterations: 500`, `checkpoint_interval: 50`, `objective: regression | l1 | huber`, `seed: 42`; huber adds `huber_alpha_scale: 0.5` (fraction of training label std, per fold) |
| GBM `leaves_{7,15,31,63}_*` | `num_leaves: N`, `learning_rate: 0.05`, `max_depth: -1`, `min_child_samples: 50`, `feature_fraction: 0.7`, `bagging_fraction: 0.8`, `bagging_freq: 1`, `lambda_l1: 0.5`, `lambda_l2: 5.0`, `seed: 42`; `max_bin: 255` |
| Linear | `ols: LinearRegression`; `ridge_a{alpha}`; `lasso_f{alpha_frac}: {Lasso, max_iter: 5000}`; `enet_f{alpha_frac}: {ElasticNet, l1_ratio: 0.5, max_iter: 5000}` |
| TabM | `tabm_s {hidden_dim 64, n_members 4, dropout 0.1, n_epochs 200, batch_size 4096, checkpoint_interval 25}`; `tabm_l {hidden_dim 256, n_members 16}`; `PUBLISHED_DEVICE = "cuda"` |
| Sequence | `lstm_h64 {hidden_size 64, n_layers 2, lookback 60, dropout 0.1, n_epochs 100, batch_size 2048, checkpoint_interval 5}`; `nlinear {lookback 60, same schedule}`; MC dropout declared in identity |
| Latent | `pca {n_factors: 5, checkpoint_interval: 0}` (cpu); `sdf {n_factors 5, n_epochs_unc 256, n_epochs_moment 64, n_epochs_cond 1024, burn_in_epochs 0, beta_n_epochs 256}` (cuda) |
| DML `dml` | `seed = RANDOM_SEED`, `n_folds 5`, `n_placebo 100`, `max_samples 0` (canonical); `HistGradientBoostingRegressor` nuisances; `DML_THREAD_LIMIT 1`; `REFUTATION_ALPHA 0.05`; `MIN_PLACEBO_DRAWS 10`; `GAP_TOLERANCE_CADENCES 4`; min rows `(n_folds + 1) * 50 + n_folds * embargo`; treatment/outcome std > 1e-10 (inference: the `dml` config file was not located; these are code fallbacks) |
| 04 constants | `SEED 42`, `FFT_WINDOW 252`, `FFT_TARGET_PERIODS [63, 126]`, FFT truncation probe `t = 352`, `HOLD_LAST_SETTLE_SESSIONS 2`, `MIN_FORWARD_SMOOTHED_SEPARATION 0.1`, `HELD_RUN_SESSIONS 21`, `FDR_ALPHA 0.05`, ARIMA eligibility `>= burnin + 30`, `ProcessPoolExecutor(max_workers=min(n, cpu_count-1), mp_context=fork)` |
| 05 thresholds | `HAC_MAXLAGS 5`, `FDR_ALPHA 0.05`, `NAIVE_T 1.96`, `REDUNDANCY_CUT 0.7`, `MIN_SIGN_CONSISTENCY 0.60`, `IC_THRESHOLD 0.008`, `N_QUANTILES 5`, `MIN_COVERAGE 0.70`, `MAX_STALENESS 0.50`, `MAX_ABS_RETURN 1.0`, `MAX_ABS_FEATURE 1e6`, `CROSS_SECTION_TARGET 10`, `MIN_IC_DATES 20`, `MIN_FOLD_DATES 5`, `ROLLING_DAYS 126`, `CORRELATION_SAMPLE_SESSIONS 200`, `SPARSE_PCT 5.0` |
| Conformal | `HOLDOUT_CONFORMAL_EMBARGO_STEPS` 5 (fwd_ret_5d) / 21 (fwd_ret_21d); sizing lag `max(1, horizon)`; `alpha 0.2`; `min_calibration_n 30` |
| Uncertainty | Paired bootstrap minimum n: 21 daily, 12 weekly, 6 monthly; `INSOLVENT_MAX_DRAWDOWN -1.0`; `STAGE_BASELINE {signal: equal_weight, allocation: signal_leader, risk_overlay: allocation_leader}` |

**Margin model** (README): `ContractSpec.margin_pct = (initial, maintenance)` from CME maintenance rates at 2025-12-31 settlements in `data/futures/market/futures_specs.yaml`; engine applies the ratio to each bar's notional. 8 products not in the CSV (ES/NQ/YM/RTY, CL/NG/HO/RB) use SPAN-style initial 5% equity_index, 8% energy, maintenance = initial/1.10. Stable-pct drift (anchored vs start-window implied): ES 4.54% vs 28.61%, NQ 4.54% vs 44.64%, GC 6.28% vs 13.82%, ZN 1.67% vs 1.86%, CL 7.27% vs 2.22%, ZC 4.43% vs 2.70%, NG 7.27% vs 0.13%. 2011 under-margins equity indices and over-margins crashed commodities; confined to absolute levels 2011-2015; relative comparisons and the holdout unaffected.

**Eligibility**: declared ⊆ data and archive ⊆ declared (two-way roster assertion); breadth floor `2 * max(top_k_grid[primary])` = 20 products per decision date; a session is scored only if ≥ half the usual product count (02) / ≥ 10 products (04, 05); a product enters ARIMA only with ≥ burnin + 30 pre-holdout sessions; FFT needs ≥ 302 observations.

## Pipeline stages

Run from repo root in numeric order (`uv run python case_studies/cme_futures/NN_*.py`; 10a, 10b before 10). Bundles: `v3.1.0-artifacts` (current; `uv run python scripts/download_artifacts.py --cs cme_futures`) and `v3.0.0-artifacts`; hashes do not cross bundles.

| Stage | Notebook | Chapter | What it does | What it writes |
|---|---|---|---|---|
| Feasibility | `01_feasibility_analysis` | 6 | Breadth per decision date, spread per product from contract specs, move-to-spread exceedance, carry ACF, declared folds. Fits nothing | nothing |
| Labels | `02_labels` | 7 | 5d and 21d forward returns on `adj_close`, hole rule, reconciliation, ESS, carry baseline IC | `labels/fwd_ret_5d.parquet`, `labels/fwd_ret_21d.parquet` + `.digest.json` sidecars |
| Features | `03_financial_features` | 8 | 62 features: term structure, momentum, vol, trend/range, calendar, within-date percentiles, composites; warmup audit; holdout-withheld rebuild | `features/financial.parquet` + sidecar |
| Temporal | `04_model_based_features` | 9 | ARIMA(2,0,1) walk on carry z-score per product; rolling FFT; 2-state HMM on book-average carry (filtered, scheduled, frozen at holdout) | `features/model_based.parquet` + sidecar (no fold column) |
| Evaluation | `05_evaluation` | 7-9 | Univariate IC screen with HAC + BH-FDR, sign consistency, quantile profiles, redundancy, triage | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` |
| Linear | `06_linear` | 11 | OLS/Ridge/Lasso/ENet on both labels (28 configs) | runs + prediction sets in `run_log/registry.db`; coefficients `run_log/training/{hash}/`, scores `run_log/predictions/{hash}/` |
| GBM | `07_gbm` | 12 | LightGBM 5x3 grid, 10 checkpoints each | runs + prediction sets; boosters, `learning_curves.parquet`, `fold_metrics.parquet` |
| Tabular DL | `08_tabular_dl` | 12 | TabM s/m/l | runs + prediction sets; checkpoints `run_log/training/tabular_dl/` |
| LSTM | `09_dl_lstm` | 13 | NLinear and LSTM (lookback 60) | runs + prediction sets; checkpoints `run_log/training/deep_learning/` |
| Latent index | `10_latent_factors` | 14 | Index of 10a/10b | nothing (reads registry) |
| PCA | `10a_pca` | 14 | Return-panel PCA, 5 factors, CPU | runs + prediction sets |
| SDF | `10b_stochastic_discount_factor` | 14 | Neural SDF with characteristics, 5 factors, cuda | runs + prediction sets |
| Causal DML | `11_causal_dml` | 15 | Effect of `carry_pct` on each horizon; never enters the funnel | one row per label in registry `causal_runs` |
| Model analysis | `12_model_analysis` | 11-15 | IC diagnostics per (family, config, label, checkpoint); menu coverage; carry autocorrelation | nothing |
| Backtest | `13_backtest` | 16 | Equal-weight top-k long/short baseline over every prediction set and checkpoint | one backtest per (prediction, top_k); population `cme_futures-signal-validation-v1`; sets `cme_futures-signal-<label>-v1`; `daily_returns/weights/trades/fills/equity/portfolio_state.parquet`, `spec.json` under `run_log/backtest/{hash}/` |
| Portfolio | `14_portfolio_management` | 17 | 6 allocators on top-10 distinct (family, config_name) | population `cme_futures-allocation-validation-v1`; sets `cme_futures-allocation-<label>-v1` |
| Risk | `15_risk_management` | 19 | 14 position overlays on the per-label rank-1 of signal∪allocation | population `cme_futures-risk-validation-v1`; sets `cme_futures-risk-<label>-v1`, `cme_futures-pre-overlay-<label>-v1` |
| Costs | `16_costs` | 18 | 11-point all-in cost grid on the shipped configuration, overlay carried; diagnostic only | population `cme_futures-cost-validation-v1` |
| Holdout predictions | `17_holdout_predictions` | 20 | Refit the carrier on pre-2024 history | training run whose CV declares the holdout fold; prediction set at `split='holdout'` |
| Holdout backtest | `18_holdout_backtest` | 20 | Replay the carrier's own spec on holdout predictions | one backtest at `stage='holdout'` |
| Strategy analysis | `19_strategy_analysis` | 20 | Freeze final sets, deflated Sharpe, paired bootstrap, holdout comparison | `results/strategy_assessment.json`, tearsheet HTML; rebuilds `cohort_metrics`, `backtest_paired_metrics` |

Model populations (`MODEL_POPULATION_NAMES`): `cme_futures-linear-validation-v1`, `cme_futures-gbm-validation-v1`, `cme_futures-tabular_dl-validation-v1`, `cme_futures-deep_learning-validation-v1`, `cme_futures-pca-validation-v1`, `cme_futures-sdf-validation-v1`; final sets `cme_futures-final-validation-<label>-v1`, `cme_futures-final-selection-v1`. Registry tables: `training_runs`, `prediction_sets(prediction_hash, training_hash, split, checkpoint_kind, checkpoint_value)`, `backtest_runs(backtest_hash, prediction_hash, spec_json, stage)`, `backtest_metrics`, `candidate_sets`/`candidate_set_names`, `cohort_metrics`, `backtest_paired_metrics`, `causal_runs`.

Sequencing: 11 after 03/04/05; 12 after 06-10b; 13 -> 14 -> 15 -> 16 -> 17 -> 18 -> 19 (risk before costs because costs price the post-overlay winner). Registry `stage` vocabulary: `signal`, `allocation`, `risk_overlay`, `cost_sensitivity`, `holdout`; candidate-set vocabulary: `signal`, `allocation`, `risk`, `pre-overlay`, `final-validation` (`REGISTRY_STAGE_BY_SET_STAGE` maps them).

## Design decisions and why

| Decision | Choice | Reason / evidence |
|---|---|---|
| Signal thesis | Carry = `(c0 - c1)/c0 * 12` from raw front and second settlements; rank the 30 products, long top vs short bottom | Carry is observable today, not forecast: backwardation earns the roll, contango pays it. "A bet on a difference that is already observable" |
| Two price series | `adj_close` (ratio back-adjusted) for returns/labels; `raw_close` for carry, curvature, slope | A return across a roll on raw prices books the basis gap as profit; differencing two adjusted series measures accumulated roll history |
| Ranking frame | Every percentile within `(timestamp, position)`; carry additionally within sector (`carry_rank_sector`) | Three tenors per product-date; ranking across positions "would put ES front month and ES third month in one ordering and call the difference information"; unconditional carry rank is dominated by sector membership |
| Cross-sectional vs pooled loss | Rank-based evaluation, within-sector normalization | Within-sector 5d label std 0.0082 (treasuries) to 0.0559 (energy), 6.8x; a pooled-loss model "treats a large move in a treasury note as noise" |
| Label | Close-to-close `A_{t+h}/A_t - 1` on the product's own sessions; sample every session; `position == 0` kept before any shift | 5x rows at the cost of overlap; N_eff ≈ N/h. Filtering survivors before shifting would count rows, not sessions |
| Development window | Filter on `_label_end = timestamp.shift(-h).over(product) < HOLDOUT_START` | Decide what a diagnostic may see by the date the outcome is known, not the observation date |
| Fitted features | Refit schedule (burn-in, fit, apply for `refit_every`, refit) with `freeze_after` at the last pre-holdout index; no fold column | A per-fold fit is causal for validation rows but not training rows (earliest training rows carry parameters estimated from 8 years of their own future). Nothing raised before the 2026-09-04 fix |
| HMM posterior | `filtered_state_probs` (forward recursion P(z_t | x_{1:t})), not `predict_proba` (smoothed) | Smoothed values move when later data arrives; the truncation test passed at 4.3e-12 even with the leak, so the replacement check requires forward-vs-smoothed max gap > 0.1 (observed 0.74) |
| ARIMA order | Declared (2,0,1), weights-only refit every 21 sessions | AutoARIMA: 34 orders across 240 searches, no product held one order, modal (2,0,1) 11.7%; "the information criterion moving with the sample, not structure being found" |
| ARIMA walk | One walk per product over its own history (not per fold) | The shared CV call validated `n_windows` against RTY (listed 2017-07-10), truncating every product; last period's training rows 99.87% empty. One walk: 86,204 forecasts vs 123,870 overlapping |
| HMM input | Book-average carry on a complete date x product grid, forward-fill limit 2 sessions (`HOLD_LAST_SETTLE_SESSIONS = 2`, measured pre-holdout) | Sectors keep different holiday calendars; an average over whoever settled jumps for non-curve reasons |
| Multiple testing | HAC t (maxlags = horizon 5) then BH-FDR at 0.05; print uncorrected / HAC / FDR counts; screen both horizons | "State the size of the search next to the p-value"; the search across horizons is the one FDR cannot see |
| Sign consistency | Score agreement with the feature's own pooled sign, ≥ 0.60 | Strongest associations here are negative; counting positive windows makes a reliable inverse predictor unpromotable |
| Selection | Validation backtest Sharpe in 13 over published populations; IC never selects | Repeated in every model notebook. "A better story about why a model should work is not evidence that it does" |
| Checkpoints | Every GBM iteration (50..500), TabM/LSTM epoch, SDF epoch is its own prediction set | 15 configs x 10 checkpoints = 150 candidates per label; reporting the leading row "would be reporting the maximum of ten numbers as though it were one"; declaring them puts the choice in the funnel the deflated Sharpe divides by |
| GBM objective | Huber (`huber_alpha_scale: 0.5` of training label std) / MAE / MSE compared | "An objective is a claim about which errors matter"; heavy-tailed commodity returns let one move carry hundreds of times the weight under squared error |
| GBM missing values | Left in place for LightGBM | Imputing a median "would hand the model an observation nobody made" |
| Network scaling | Scaler refit inside each training fold (TabM adapter) | Global mean/std leak validation information through two numbers that never appear in output |
| Linear penalties | Ridge alpha by powers of ten; Lasso/ENet by `alpha_frac` of each fold's own `alpha_max` | One declared value means the same thing on every fold |
| Latent factors | 5 factors declared; PCA sees returns only (characteristics `del`-ed), SDF sees returns and characteristics | Are the directions along which returns vary most also those along which they are compensated? A question about this panel |
| Causal question | DML of `carry_pct` on each horizon with confounders [vol_21d, momentum_composite, carry_rank] | Carry is the traded signal; a positive backtest Sharpe shows carry predicted returns, "says nothing about whether carry was the cause" |
| Equal-weight baseline | Ch16 signal stage; absent from the allocator menu | "The absence of an allocation method, not one of them"; quoting it would be "quoting the control arm as the finding" |
| Allocation shortlist | Top 10 distinct `(family, config_name)` by equal-weight Sharpe, keeping the winning checkpoint and top_k | Running every allocator on every baseline row multiplies trials and inflates K |
| Allocator lookback, stop levels | From `setup.yaml`, never from validation Sharpe | Choosing them by validation Sharpe fits the parameter to the path it is assessed on |
| Overlay parent | Per-label rank-1 of signal∪allocation; one rule per request | An overlay on a weaker parent measures it against a strategy that would not ship; stacked rules may exit on the same moves |
| Costs | Select frictionlessly, then sweep costs on the fixed winner with its overlay carried | "Price, don't cost"; the output is a breakeven level, not a net Sharpe; a strategy that dies at a few bps was "a description of a data artifact" |
| Holdout | Refit on pre-window history sealed by the widest buffer (21D); one refitted generation; one backtest; find by derivation, never by search | Only a refit is out of sample; the new training hash is the evidence |
| Fold boundary | `StateTransitionPolicy(fold_boundary="continue")` | Positions carry across the 4 fold boundaries (~1.5% of decisions inherit ≤ one week of exposure); flattening with same-bar execution would be implausible |
| Identity | Content-addressed hashes; device, seed, preview reductions, checkpoint in the hash; immutable populations created before running | GPU and CPU sums reach different weights; a changed membership under the same name is refused unless the superseded hash is named |

## Market-specific guardrails

**Data and prices**
- Use `adj_close` for returns, `raw_close` for carry/curvature/slope; 02 and 03 state it per column.
- Null (never clip) non-positive prices once at load: `when(col > 0).then(col)`; a deferred CL contract settled below zero on 2020-04-20. Guard every division.
- Read `tick_size`/`tick_value` from `futures_specs.yaml`; assert `quote_scale = tick * contract_size / tick_value ∈ {1, 100}`; settlements are calculated values that sit off the tick grid, so an inferred tick understates the spread.
- Detect a missing settlement file by shape (absent at every tenor, whole sectors by venue, neighbours normal); declare in `universe.excluded_sessions` and drop before computing anything.
- Count breadth on `max(session_date)` per truncated week (Thursday on holiday weeks), compare with `2 x max top_k` = 20.
- Divide each move by its own product's spread (0.43 bps NQ to 5.80 bps ZC) so break-even is 1 everywhere; one cost line over raw returns hides an order-of-magnitude range.
- Compute ACF per entity then average (`panel_acf`); one concatenated series measures the 29 joins between products.
- Treat `x12` carry as a scale factor, not an annual rate (only energy lists monthly); compare across sectors via `carry_rank_sector`. The roll-week flag (`day >= 25`) is a calendar approximation.

**Labels**
- Keep `position == 0` before any shift; compute `from_end` and `session` on the complete series.
- Hole rule: `MAX_SESSION_GAP_DAYS = 5`; null where `_holes.shift(-h) - _holes != 0`; reconcile `labelled + tail + hole + no-anchor == height`; no window spans more than `ceil(7h/5) + 7` calendar days; dtype Float64.
- Seal diagnostics on outcome date: `_label_end < HOLDOUT_START` (02), `LAST_SCORABLE_DECISION_DATE = pre_holdout_sessions[-(h+1)]` (04), per-product `shift(-h) < HOLDOUT_START` (05).
- Report `effective_sample_size` (N_eff ≈ N/h) and use HAC with `label_horizon`; `.sort(date_col)` before `compute_ic_hac_stats` (Newey-West does not sort).

**Features**
- Rank within `[timestamp, position]`; screen the front month only; `fill_nan(None)` on library NaNs (`rsi_14`, `vol_yz_21d`, `vr_63d`) because polars ranks NaN at the top.
- `ls_signal` is null where the score is null (flat and no reading differ).
- Clip vol ratios at 10, carry z-score at ±5, `+10` in the `risk_adj_score` denominator, `EPS = 1e-8` in `rolling_zscore`/`trailing_*`.
- Rebuild with the holdout withheld and `assert_values_agree` (value vs null counts); `warmup_audit` leading nulls = declared lookback; audit term-structure columns with `entity="product"` on `term_structure(bars)`, price columns with `entity=["product","position"]`.
- Fitted features: schedule, not fold; `refit_boundaries` puts every emitted index at or after its block's `fit_end`; assert every `fit_through < HOLDOUT_START` **and** that holdout-dated rows are populated (an empty holdout satisfies "no leak" vacuously and makes a holdout retrain fit on nulls).
- ARIMA: `apply(y, refit=False)` and assert params moved by 0; never `forecast(h)`; declare the order.
- HMM: `threadpool_limits(limits=1)`, constant seed, `sort_states_by_mean`, name the feature for the higher-carry state; measure `HOLD_LAST_SETTLE_SESSIONS` on pre-holdout data; filter `timestamp <= HOLDOUT_END` so a refreshed price file cannot widen the artifact.
- Check that a leak check can fail: the truncation test is nearly empty under a refit schedule.
- Fingerprint values, not names: `.digest.json` sidecar with content digest, row count, keys, writer, input digests, estimation schedule.

**Evaluation and models**
- Gate 0 `validate_modeling_inputs` on the pre-holdout span only (`fail_on_critical=True`); coverage ≥ 0.70; staleness ≤ 0.50; `|return| <= 1.0`, `|feature| <= 1e6`.
- Quantile buckets within each session, never pooled; regime features (constant across products) cannot be rank-screened.
- Print per-fold training coverage (`SPARSE_PCT = 5`) before fitting; a sparse feature median-imputed to zero sits inert and nothing raises.
- Judge `full_coverage` against each label's own maximum scorable dates; partial-coverage ICs (lasso_f0.85, lasso_f0.7 at 5d) are self-selected samples. Carry `ic_n_days`; `ic_t = ic_mean / (ic_std / sqrt(n_folds_ic))`.
- Compare GBM configs at the final iteration; read `comparison.ratio` to see whether the checkpoint or the model does the ranking.
- Declare `device`; a narrowed catalog or another device needs its own `POPULATION_NAME`; previews need `WORKSPACE` and `PREVIEW_REDUCTIONS`; `RuntimeError` on any `complete == False` row.
- Latent factors: fit within each training fold; complete validation key set required (a missing product changes every component); SDF convergence is not enforced (only IPCA is).

**Causal (11)**
- Missing confounders are an error (a different estimand, not a degraded one). Declare `causal.treatment_window` from the formula; canonical refuses without it.
- Placebo block = `max(label_buffer_steps, treatment_window_steps)`; compare block length to measured autocorrelation (0.52 at lag 5 means the 5-session block still under-preserves; 0.14 at 21).
- Compare HAC t-statistics, not raw placebo effects (`var(T_res)` inflates 11.4x under permutation; p 0.0164 vs 0.5902 on a true-zero synthetic); `CAUSAL_RUNNER_VERSION = 2`; a row with `refutation_placebo_t_json` NULL is on the old scale.
- Plus-one p-value, floor `1/(n+1)`; `Underpowered` when ≤ 19 successful draws; `empirical_permutation_p` raises on non-finite inputs; record `placebo_frozen_fraction`.
- HAC bandwidth ≥ horizon - 1 with the outcome horizon, not the CV buffer; a `failed` covariance is never relabeled HC0 and makes the result incomplete.
- `DML_THREAD_LIMIT = 1`; restart check `restarted.hash == result.hash`; one live identity per label (`SUPERSEDES_CAUSAL`).
- Count buffers in observations per entity (`embargo_from_buffer(buffer, observed_step=cadence)`); "21D" as a Timedelta ≈ 15 trading days silently leaks.

**Backtest funnel (13-16)**
- Snapshot expected identities before running (`OfficialPopulation.create` before execution; `require_complete()`); a silently skipped configuration leaves "a right-looking number computed over a set nobody can reconstruct".
- No equal_weight allocator: `stage` is not in `backtest_hash`, so the row collides with its baseline parent (48 rows measured lost).
- `top_n_cap(0) -> None` (all); a positive `limit` the pool cannot fill raises; declare every papermill override in the parameters cell.
- Moment-based allocators rebuild estimates per decision from strictly prior history; warmup prefix is not scored.
- Risk overlays: configured list is the population; one rule per request; `_build_position_rules` raises on unknown types.
- Costs: all-in grid value split 50/50; the backtest charges the total; fail on any missing grid point; refuse a null cost axis. Flat cost across 30 products is a deliberate simplification: linear in turnover, a lower bound on scale costs; do not read breakeven as capacity.
- Price one strategy via `resolve_solvent_carrier` with its overlay carried; never per label or pre-overlay.

**Holdout and selection (17-19)**
- Refuse if `model_run.training.hash == carrier["training_hash"]` (no refit); raise if another refitted generation exists with a different `(training_hash, checkpoint)`; retire via the registry lifecycle, never a switch.
- Find the holdout prediction set by derived `(training_hash, checkpoint)`, never by best holdout Sharpe; `_resolve_holdout_self_backtest` raises on ambiguity.
- Guard on the full prospective `backtest_hash` from `backtest_run_status` (a field-by-field guard missed cost and calibration; a reconstructed hash once disagreed: `f23ff90cf518` vs `b2acfd5420c8`).
- Restore `contract_specs`, `futures_market`, `prices` digests (on the `symbol`-keyed frame) and `entity_contract` in the cloned spec; otherwise P&L is in wrong units.
- HRP warmup 63 bars of pre-window prices; `compute_hrp_weights` falls back to equal weight on short history and reads as decay.
- Conformal: embargo 5 / 21 observations at the holdout boundary; `ensure_conformal_calibration_identity` writes `input_identity.conformal_holdout_embargo_steps`; write widths only after the guard passes.
- Rank with `resolve_solvent_carrier` everywhere; re-rank on common support whenever a conformal candidate is present; ruined rows sort last; refuse `max_drawdown <= -1.0`; never fall through to the runner-up.
- Scope `compute_and_register(prediction_hashes=pool)` so retired generations do not inflate K; pass the resolved `carrier` to `populate_paired_metrics(replace_all=True)`.
- `num_trades` is NULL on the vectorized path: report none, not zero. Declare supersedes as `"live"` for candidate sets; `CARRIER_PINS` is empty and must be re-derived after a rebuild.

## Results and lessons

Numbers below are from the digest; the README deliberately restates nothing ("a number copied into prose stays correct only until the next rebuild"). Read `19_strategy_analysis` and the registry for current values.

| Stage | Finding |
|---|---|
| Feasibility (01) | 29 products until RTY lists 2017-07-10, then 30; breadth below 20 on 7 of 678 rebalance dates (5 Good Fridays, 2 missing-file Fridays; low of 8). Median round-trip spread 1.13 bps; median absolute 5-session move 121.2 bps (~100x); 0.991 of moves beat their own spread. Contract value ~$24k (ZC) to ~$340k (NQ); leverage ZT ~158x, silver ~8x. Narrowest purge 5 sessions |
| Labels (02) | Dev std 0.03138 (5d) vs 0.06407 (21d), ratio 2.04 vs sqrt(21/5)=2.05. Cross-product dispersion peaks 3.9% (2020) vs 2.7% median year. ESS 97,921 -> 19,611 (0.2003 vs 0.2000); 21d 97,393 -> 4,669 (0.0479). ACF 0.797 at lag 1 -> -0.004 at lag 5. Carry baseline IC 0.0069 over 3,337 sessions, naive t 1.61, HAC (8 lags) t 0.87, p 0.387: **not separable from zero**. Coverage 99.85% / 99.39%; one hole (ten-day livestock break Feb 2012) |
| Features (03) | 62 features, 310,866 rows, warmup ends 2011-12-21, 1,912 rows dropped, 25 redundancy clusters at 0.7. Carry's ordering is the least stable across rebalances (rank autocorrelation ~0.6 vs vol_21d ~1); raw carry ACF ~0.27 over ten sessions; z-score reaches zero within 20 |
| Model-based (04) | 86,204 ARIMA forecasts; HMM first value 2013-04-09, 44 estimates pre-holdout, forward-vs-smoothed separation 0.74, runs of roughly a trading month. Screen: 7 of 9 rankable; IC -0.031 (`fft_energy_63d`, HAC t -2.10) to 0.016 (`fft_dominant_period`, t 1.47); one clears |t|>1.96 alone, **none clears BH across seven** |
| Screen (05) | Few features clear FDR ("a property of the measurement"); most PROCEED via the exploration route. The carry family runs **negative** over validation windows; strongest member is the 5-day change in carry (a product whose carry just rose did worse the following week). Open readings: premium absent/reversed, or level and change carry different information |
| Linear (06) | Almost nothing above zero; what does is Lasso/ENet at alpha_frac 0.5 or 0.7; no Ridge config above zero at either horizon (peaks at alpha 1e6 / 1e7, grid edge). Ordering strong L1 > weak L1 > Ridge > OLS; `ic_std` an order of magnitude above any `ic_mean`; 21d best "fifteen times" the 5d best but mostly the coverage filter |
| GBM (07) | Every 5d config positive at the final iteration, weakest a multiple of the best linear 5d; nothing above zero at 21d: **horizons reverse between families**. Huber > MAE > MSE at 5d; MSE last at both. Median config ends near where it started; interior peaks common at 21d |
| 08-10b | No numeric results in-notebook; populations published for 13 |
| Causal (11) | Carry autocorrelation 0.52 at lag 5, 0.14 at lag 21, ~0 at lag 63 (slower than AR(1): 0.38 / 0.02). Refutation p under t-stat comparison: fwd_ret_5d 0.5545, fwd_ret_21d 0.2673, both `Fails` (old raw-effect scale: 0.0396, 0.0099 = floor). Author: the refutation column "carries no evidence here"; read the DML point estimate and HAC SE (values printed at run time, not in the digest) |
| Funnel (19, fwd_ret_21d) | signal 496 -> allocation 60 -> risk overlay 14; median Sharpe -0.392 -> 0.192 -> 1.010 by selection alone. 14 overlays on one parent span 0.322 to 1.274: the position rule alone is responsible for that range |
| Selection | Raw Sharpe ranking names `latent_factors/sdf` on `fwd_ret_21d` (1.274); the canonical resolver names **`gbm/leaves_31_mse` on `fwd_ret_5d`** (raw 1.236 -> 1.294 on the 1,270 common-support sessions), allocated by **`hrp`**. Validation figure is "the maximum of a ranking over more than a thousand backtests" |
| Holdout | ~100 weekly observations; Sharpe/CAGR/drawdown printed by 18, not in the digest. Validation-vs-holdout gap is selection plus sampling error, not decay; intervals via deflated Sharpe (`cohort_metrics`) and paired bootstrap `val_rank1_self` |
| Cost survival | Breakeven read off the 11-point curve on the shipped configuration; values not in the digest (inference: consult `19_strategy_analysis` cost-curve cell) |

Lessons: (1) the IC leader (SDF, 21d) and the shipped strategy (GBM, 5d, HRP) differ, as the README predicts; (2) the horizon a family reads is a property of the family (GBM 5d, L1-linear 21d); (3) a squared-error objective was last at both horizons on this heavy-tailed panel; (4) the carry premium the thesis assumes ran negative in validation and its univariate IC is indistinguishable from zero, yet a multivariate nonlinear model on carry plus momentum/vol state still ranked; (5) funnel medians rise by selection, so compare only within a row; (6) two historical defects, a validation-fitted holdout generation and a per-fold HMM leak, both passed every local check, which is why the guards above exist.

## How to adapt this pattern to a new dataset

1. Write `setup.yaml` first: universe groups, excluded sessions (by shape, not by thinness), decision cadence/snapshot/execution delay, `initial_cash` sized so per-position budget at the largest top_k clears the largest notional, `share_type`, `allocator_lookback`, cost components with per-product spread tiers, `top_k_grid` capped at floor(n/3) for long-short, allocator menu without equal_weight, cost grid, risk rules, feature windows and families register, model_based schedules, evaluation folds, labels with buffers and `rebalance_step = ceil(horizon/cadence_days)`, causal treatment with its construction window.
2. Build the two price series: back-adjusted for returns, raw for curve quantities; null non-positive prices at load; load contract specs (`tick_size`, `tick_value`, `multiplier`, margins) from an exchange spec file and assert the quote scale.
3. Run feasibility with no model: breadth per decision date vs `2 x max top_k`; per-product spread in bps; exceedance of own-spread at each horizon (kill-switch if the spread exceeds a typical move); `panel_acf` of the signal at the rebalance lag; `generate_cv_splits` on the whole sample and assert `len == n_splits`, `last_val < holdout_start`, `purge >= horizon`.
4. Build labels on the entity's own sessions: keep the traded tenor before shifting, hole rule, four assertions, calendar-span tolerance, `_label_end` seal, ESS, baseline IC of the thesis signal with HAC; write with `write_artifact` and a digest sidecar; `quality_report` against the expected key universe.
5. Build formula features from the register; rank within `[timestamp, position]` (and within sector where membership dominates); `fill_nan(None)` before ranking; composites with null-not-zero; `warmup_audit`; holdout-withheld rebuild with `assert_values_agree`; redundancy clusters at 0.7; persistence out to 4 decision cycles.
6. Build fitted features with `walk_forward_feature(burnin, refit_every, fit, apply, freeze_after=n_pre_holdout)`: declare the model order, filter forward (never smooth), one walk per entity over its own history, deterministic threads and seed, assert `fit_through < holdout_start` and holdout rows populated, no fold column, estimation schedule in the sidecar; make sure the leak check can fail.
7. Screen univariately on the union of validation windows, front tenor only: gates (coverage 0.70, staleness 0.50), per-date Spearman IC with HAC (maxlags = horizon), BH-FDR 0.05, sign consistency ≥ 0.60 against the feature's own sign, |IC| ≥ 0.008, within-session quantile profiles; repeat per horizon; write the triage ledger with the route in `note`.
8. Fit every family from a declared menu: resolve the plan before fitting (feature_count, eligible rows, folds constant within a label; validation inside development); impute/scale inside the fold; publish every checkpoint; `RuntimeError` on incomplete rows; declare device and seed; run the causal estimate as a separate, non-competing request.
9. Run the funnel in order with `get_*` readers from `sweep_config`: equal-weight top-k baseline on everything -> top-N distinct configs x allocators -> one rule at a time on the per-label rank-1 -> cost grid on the resolved carrier with overlay; freeze populations before running; `require_complete()`.
10. Holdout once: `build_holdout_training_spec` sealed by the widest buffer, refit (new training hash), derive the prediction set, hash-guard the single backtest, restore futures contract specs and price digests, warm up moment allocators; then `compute_and_register` and `populate_paired_metrics` scoped to the pool; report kill gates (validation CI lower bound ≥ 0; holdout-vs-EW diff CI not strongly negative) with `no_data` never coerced to pass.
11. Re-derive every hash by rule after a rebuild; never hardcode; declare supersedes as `"live"`.

### Key APIs and patterns

- Data and config: `from data import load_cme_futures` (columns `product, session_date, tenor, raw_close, adj_open/high/low/close, volume`; rename to `timestamp`/`position`); `utils.artifact_specs.load_setup_config`, `resolve_label_horizon`, `resolve_label_buffer`; `utils.cv_splits.generate_cv_splits(dataset, case_study_id, label_buffer, date_col)` wrapping `ml4t.diagnostic.splitters.WalkForwardCV` / `WalkForwardConfig(..., fold_direction="backward")`.
- Diagnostics: `ml4t.diagnostic.metrics.compute_ic_hac_stats(series, ic_col="ic", maxlags=None, label_horizon=None, kernel="bartlett")` -> `mean_ic, hac_se, t_stat, p_value, n_periods, effective_lags, naive_se, naive_t_stat, used_naive_fallback`; `cross_sectional_ic_series(..., method="spearman", min_obs)`; `ml4t.diagnostic.evaluation.stats.benjamini_hochberg_fdr(p_values, alpha=0.05, return_details=True)`.
- Engineer: `ml4t.engineer.features.momentum.rsi`, `regime.variance_ratio`, `volatility.yang_zhang_volatility` (polars expressions with `.over(ENTITY)`).
- Case-study utils: `feasibility` (`exceedance_curve, fold_timeline, panel_acf`); `label_diagnostics` (`effective_sample_size, panel_autocorrelation`); `artifact_digest` (`value_digest, write_artifact`); `artifact_quality` (`quality_report, render_quality_report, label_universe`); `feature_engineering` (`momentum_volatility_block, rolling_zscore, cross_sectional_percentile, warmup_audit, assert_values_agree, family_coverage, quantile_profile`); `temporal` (`walk_forward_feature, refit_boundaries, filtered_state_probs, arima_one_step_forecast, fit_hmm_kmeans_init, sort_states_by_mean, write_model_based`); `data_quality.validate_modeling_inputs`; `backtest_loaders.resolve_rebalance_timestamps`.
- Research: `case_studies.research` (`open_study, load_model_configs, model_requests, resolved_model_plan, run_model_population, population_supersedes, CausalResult, OfficialPopulation, require_declared_menu_coverage`); `research_workflow` (`ALL_LABELS, MODEL_POPULATION_NAMES, load_futures_price_path, run_official_backtest_requests, create_label_candidate_sets, rank_by_validation_sharpe, shortlist_signal_configurations, pre_overlay_results, final_selection_candidate_set, selection_catalog`); `sweep_config` (`get_top_n_predictions, top_n_cap, get_top_k_values_for, get_allocators, get_cost_grid_bps, get_position_risk_controls`); `strategy_analysis` (`resolve_solvent_carrier, select_holdout_self_backtest, training_run_fitted_for_the_holdout`); `research.holdout.build_holdout_training_spec`; `research.models.reconstruct_locked_model_request`; `backtest_runner.run_backtest`; `cohort_metrics.compute_and_register`; `paired_metrics.populate_paired_metrics`.
- Engine: `ml4t.backtest.risk` (`StopLoss(pct)`, `TrailingStop(pct)`, `TimeExit(max_bars)`, `RuleChain`); `Engine.from_config(contract_specs=...)` populates the broker's `margin_pct_schedule`.

```python
# Label with guarded denominator and hole rule (02); seal on outcome date
base = pl.when(pl.col("adj_close") > 0).then(pl.col("adj_close"))
holes_ahead = pl.col("_holes").shift(-h).over("product") - pl.col("_holes")
label = pl.when(holes_ahead == 0).then(pl.col("adj_close").shift(-h).over("product") / base - 1)
dev = (df.with_columns(pl.col("timestamp").shift(-h).over("product").alias("_label_end"))
         .filter(pl.col("_label_end") < HOLDOUT_START))

# Within-date, within-position percentile (03)
pct = (pl.col(src).rank(method="min").over(["timestamp", "position"])
       / (pl.col(src).count().over(["timestamp", "position"]) + 1) * 100)

# Scheduled, frozen-at-holdout fitted feature (04)
out = walk_forward_feature(X, timestamps=dates, burnin=252, refit_every=21,
                           fit=lambda tr: ARIMA(tr[:, 0], order=(2, 0, 1)).fit(),
                           apply=lambda m, pre: arima_one_step_forecast(m, pre).reshape(-1, 1),
                           n_features=1, freeze_after=n_pre_holdout)

# Strategy request rows by stage (13-16)
{"request_name": f"{prediction_hash}-equal-weight-k{top_k}", "label": label,
 "signal": {"method": "equal_weight_top_k", "top_k": top_k},
 "allocation": None, "risk": None, "costs": None, "chapter": "ch16"}
# ch17: "allocation": {"method": "hrp", "vol_window": 63}
# ch19: "risk": {"position_rules": [{"type": "stop_loss", "threshold": 0.05}]}
# ch18: "costs": {"commission_bps": c/2, "slippage_bps": c/2}

# Resolve the shipped configuration, refit for the holdout, guard the single backtest (16-18)
carrier = resolve_solvent_carrier("cme_futures")
holdout_spec = build_holdout_training_spec(study, validation_spec, timeline=timeline, case_study="cme_futures")
request = reconstruct_locked_model_request(study, holdout_spec, checkpoint_kind=K, checkpoint_value=V)
model_run = request.run()
if model_run.training.hash == carrier["training_hash"]: raise RuntimeError("did not refit")
prospective_hash = backtest_run_status(CS, HOLDOUT_PREDICTION_HASH, spec).backtest_hash
if {r["backtest_hash"] for r in _registered_holdout_backtests(CASE_DIR, HOLDOUT_PREDICTION_HASH)} - {prospective_hash}:
    raise RuntimeError("the holdout window already carries a backtest of a different configuration")
result = run_backtest(CS, HOLDOUT_PREDICTION_HASH, spec, prices=prices, predictions=predictions,
                      label=LABEL, register=True, initial_cash=..., calendar=..., contract_specs=contract_specs)
assert result.backtest_hash == prospective_hash
```

## Related references

- `chapters/06_strategy_definition.md`: feasibility without a model, breadth, spread exceedance, run log.
- `chapters/07_defining_the_learning_task.md`: forward-return labels, ESS/uniqueness, HAC IC, BH-FDR, Table 7.2 triage.
- `chapters/08_financial_features.md`: momentum/vol/trend families, within-date percentiles, redundancy, warmup audits.
- `chapters/09_model_based_features.md`: ARIMA/HMM refit schedules, filtered vs smoothed posteriors, `walk_forward_feature`.
- `chapters/11_ml_pipeline.md`: resolve-then-fit, fold-scoped imputation and scaling, populations.
- `chapters/12_gradient_boosting.md`: LightGBM objectives, checkpoints as configurations, TabM.
- `chapters/13_dl_time_series.md`: NLinear/LSTM windows that stop at fold and purge boundaries.
- `chapters/14_latent_factors.md`: PCA vs SDF, fold-fitted factors.
- `chapters/15_causal_estimation.md`: DML, block-permutation refutation on t-stats, estimand discipline.
- `chapters/16_strategy_simulation.md`: equal-weight top-k baseline, fold-boundary policy, vectorized rebalance.
- `chapters/17_portfolio_construction.md`: inverse-vol, risk parity, HRP, Ledoit-Wolf MVO, conformal sizing, allocator lookback.
- `chapters/18_transaction_costs.md`: price-don't-cost, breakeven curves, flat-cost caveat.
- `chapters/19_risk_management.md`: stops on mean-reverting signals, one rule per request.
- `chapters/20_strategy_synthesis.md`: holdout refit, deflated Sharpe, paired bootstrap, kill gates.
- `chapters/02_financial_data_universe.md`: continuous futures construction (`02_financial_data_universe/06_futures_continuous`), ratio back-adjustment.
- `case_studies/etfs.md`, `case_studies/fx_pairs.md`: same HMM refit cadence (63) and holdout backtest pattern.
- `case_studies/us_equities_panel.md`: batch runner that holds one fold set (pass unresolved requests there).
- `case_studies/crypto_perps_funding.md`: same 10.9x `var(T_res)` inflation under placebo permutation.
- `case_studies/sp500_equity_option_analytics.md`: DML `max_samples` note.
- `libraries/ml4t_diagnostic.md`: `WalkForwardCV`, `compute_ic_hac_stats`, `cross_sectional_ic_series`, `benjamini_hochberg_fdr`.
- `libraries/ml4t_engineer.md`: `rsi`, `variance_ratio`, `yang_zhang_volatility`.
- `libraries/ml4t_backtest.md`: `Engine.from_config(contract_specs=...)`, margin schedule, `StopLoss`/`TrailingStop`/`TimeExit`/`RuleChain`.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `workflow.md`, `companion_repo.md`, `glossary.md`: cross-cutting rules this file instantiates.
- Further reading:
  - Chernozhukov et al. (2017), double/debiased machine learning.
  - López de Prado (2018), walk-forward with embargo, CSCV/PBO; Bailey & López de Prado, deflated Sharpe ratio.
  - Davison & Hinkley (1997); Phipson & Smyth (2010), plus-one permutation p-values.
  - Newey-West and Driscoll-Kraay HAC covariance; White's Reality Check; Ledoit-Wolf shrinkage; hierarchical risk parity; split conformal prediction.
  - CME historical-margins page and CME Datamine catalog F001 (period-specific SPAN margins).

## Glossary

- **Front month / tenor / position**: contract closest to expiry (`position` 0); 1 and 2 are the second and third tenors. Only the front is traded and labelled.
- **Carry (`carry_pct`)**: `(c0 - c1)/c0 * 12` from raw front and second settlements; positive in backwardation (rolling earns), negative in contango (rolling pays). `x12` is a scale factor, not an annual rate.
- **Curvature**: `(c0 - 2c1 + c2)/c1`.
- **Ratio back-adjustment / `adj_close` / `raw_close` / `cum_ratio`**: rescaled continuous series for returns; exchange settlement for curve quantities; cumulative ratio whose change marks a roll.
- **Settlement price**: calculated end-of-session value that may sit off the tick grid.
- **Quote scale**: `tick * contract_size / tick_value`, 1 or 100 (cents-quoted grains/livestock, percent-of-face treasuries).
- **Margin pct / SPAN**: initial and maintenance deposit as a fraction of notional; maintenance ≈ initial/1.10; leverage = 1/margin_pct.
- **Decision date**: last session of each ISO week (`resolve_rebalance_timestamps(ts, "weekly_friday_close", calendar="CME")`).
- **Breadth**: products quoted on a decision date; floor 20.
- **Exceedance curve**: fraction of moves at least a given multiple of the product's own spread.
- **Hole**: session spacing > 5 calendar days inside a product's series.
- **Development window**: pre-holdout sessions whose label resolves before the holdout (`_label_end < HOLDOUT_START`).
- **ESS / average uniqueness**: sum of uniqueness weights; ≈ N/h for overlapping h-session labels.
- **IC / ICIR / `ic_t`**: per-session Spearman between feature or prediction and forward return, averaged; mean over std across validation windows; `ic_mean / (ic_std / sqrt(n_folds_ic))`.
- **HAC / Newey-West / Driscoll-Kraay**: autocorrelation-consistent SE with Bartlett weights; lag = `max(horizon-1, floor(4(T/100)^(2/9)))`.
- **Sign consistency**: share of validation windows whose mean IC has the pooled sign.
- **PROCEED / REVISE / STOP**: Table 7.2 triage; confirmation (FDR) vs exploration (stable + above floor) routes; only STOP judges the column.
- **Register / timing contract**: per-family pattern, role (signal/state), hypothesis, inputs, lookback, lag, frame, representation, failure_mode in `setup.yaml`.
- **Seal**: withholding the holdout changes no feature value (`assert_values_agree`).
- **Fitted feature / refit schedule / `freeze_after`**: model-output feature bounded by burn-in, `refit_every` and a freeze at the last pre-holdout index.
- **Filtered vs smoothed posterior**: P(z_t | x_{1:t}) via forward recursion vs P(z_t | x_{1:T}) via `predict_proba`.
- **Digest sidecar**: `.digest.json` with content digest, row count, keys, writer, input digests, estimation schedule.
- **Population / candidate set / supersedes**: immutable named list of identities created before a run; per-label pool of backtest hashes (`cme_futures-<stage>-<label>-v1`); declaration of the generation a re-run retires (`"live"` = current head).
- **Execution tier / workspace / preview reductions**: canonical (published) vs preview (reduced, workspace-only, reductions in the hash).
- **Checkpoint**: a training state (GBM iteration, NN epoch, SDF epoch) published as its own prediction set.
- **`alpha_frac`**: Lasso/ENet penalty as a fraction of the fold's `alpha_max`.
- **TabM / NLinear / MC dropout**: rank-1-adapter tabular ensemble; last-value-normalized linear window map; dropout at inference, declared in identity.
- **SDF / PCA / latent factor**: pricing kernel m with E[m R] constant across assets; variance directions of the return panel; 5 factors declared.
- **DML / estimand / embargo / block permutation / plus-one p / Underpowered**: cross-fitted residualization effect estimate; full specification; observation gap between folds; within-product contiguous placebo on HAC t-stats; `(1 + extreme)/(1 + n)`; ≤ 19 draws.
- **Signal / allocation / risk overlay / cost sensitivity stages**: equal-weight top-k baseline; alternative sizing on the shortlist; one exit rule on the rank-1 parent; all-in bps grid on the carrier.
- **top_k / integer-share rule / allocator lookback / warmup**: products per leg (5, 10); whole contracts; 63 bars for moment allocators, supplied as unscored prefix.
- **Carrier**: the validation rank-1 configuration with spec and drawdown, resolved by `resolve_solvent_carrier`; refused if ruined (`max_drawdown <= -1.0`).
- **Common-support re-ranking**: Sharpe recomputed over timestamps every candidate prices (required when conformal candidates abstain).
- **Holdout vintage**: ARIMA/HMM values frozen at the last pre-window estimate and rolled across 2024-2025.
- **Fold-derived fields**: `expected_prediction_keys`, `effective_params_by_fold`, `resolved_fold_digest`; recomputed for the holdout fold.
- **StateTransitionPolicy**: `fold_boundary` / `temporal_gap` in {continue, reset, liquidate}.
- **Conformal width / sizing lag / embargo**: per-prediction residual quantile for inverse-uncertainty sizing; `max(1, horizon)`; 5 / 21 observations at the holdout boundary.
- **DSR / PBO / paired block bootstrap / `val_rank1_self` / kill gate**: trial-count-corrected Sharpe; CSCV overfitting probability; interval on Sharpe differences; selected validation series vs its own holdout replay; CI-bound go/no-go.
- **Degenerate prediction set**: one with a constant-prediction fold; excluded from trials.
- **`source` (computed/reused)**: whether a backtest ran or was served from the registry.
