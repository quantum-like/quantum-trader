# Case study: NASDAQ-100 intraday microstructure (114 stocks, 15-min, fwd_ret_15m)

> The highest-frequency case in the book: AlgoSeek TAQ-derived 1-minute NBBO and trade-location bars for point-in-time NASDAQ-100 members (2020-01-02 to 2021-12-31), decided every 15 minutes on a 15-minute forward midpoint return. At this horizon a price move (median absolute 15-min move 15.8 bps) and the cost of capturing it (measured median round trip 6.16 bps) are the same order of magnitude, so a cost figure wrong by 2x changes the sign of the answer, and more of the feasibility notebook is cost measurement than anything else. The featured configuration collapses out of sample on the full 115-name universe (holdout Sharpe -0.42) but is positive on the 50 cheapest-to-trade names, so the cost-feasible screen is load-bearing. The closing lesson is that choosing among one family's configurations on a validation window does not carry to a later window while the family average does, which is why the case study carries a declared mean-forecast GBM ensemble instead of a single winner. Headline: diagnose the cost problem, screen the universe, treat model selection as estimation under uncertainty.

Universe-count caveat: the unit title says 114 stocks; `setup.yaml` declares `n_assets: 115`; the part-2 digest says 113 names carry prices and predictions. Unresolved from the notes; assert the config's 115 against the archive (as `01_feasibility_analysis` does) and report what survives.

Repo root for every path below: `case_studies/nasdaq100_microstructure/`.

## Setup contract

Source: `config/setup.yaml` (`strategy_id: nasdaq100_microstructure`, `setup_version: v1`) and `README.md`.

| Item | Contract |
|---|---|
| Data | AlgoSeek TAQ-derived minute bars with full NBBO + trade-location schema, 2020-01-02 to 2021-12-31; loader `load_nasdaq100_bars()`; feasibility wraps it as `load_bars(frequency="1m")` (midpoint attached, stopped short of `holdout_start`) and `load_bars(CADENCE)` for decision bars |
| Observation grid | ONE MINUTE. `decision.bar_frequency: 15_minute` is how often the strategy may act, NOT the bar size; `03_financial_features` keeps every 15th observation to build the decision schedule |
| Universe | 115 point-in-time NASDAQ-100 members (`universe.symbols`, `n_assets: 115`), `eligibility_rule: nasdaq100_membership_with_pre_holdout_session`. Archive carries 123 symbols; 8 first quote after 2021-07-01 and are excluded. Leavers end where they left: AAL and WLTW stop 2020-05-01 |
| Cost-feasible universe | `universe.cost_feasible.validation` / `.holdout`: 50 names each, cheapest by round-trip proxy, frozen per split. Validation list profiled on quote bars 2020-01-01 to 2020-06-30 (strictly before the validation window); holdout list from the pre-holdout `liquidity_profile.parquet` (< 2021-07-01). Selected by `backtest.sweep.universe_filter: cost_feasible`; the split-specific list resolves at backtest time from the prediction set's split; the spec hash carries only the filter name |
| Labels | `labels.primary: fwd_ret_15m`; `variants: [fwd_ret_5m, fwd_ret_60m, fwd_dir_15m]`; `horizons` declared explicitly (15min/5min/60min/15min); `buffer: 16min`; `variant_buffers` fwd_ret_5m 6min, fwd_ret_60m 61min, fwd_dir_15m 16min; `rebalance_step: 1` for every label; `classification_eval_label: fwd_dir_15m -> fwd_ret_15m` |
| Decision cadence | `cadence_by_label`: fwd_ret_5m 5_minute, fwd_ret_15m 15_minute, fwd_dir_15m 15_minute, fwd_ret_60m 60_minute; `decision_snapshot: bar_close`; `execution_delay: 1_bar`; `execution_price_assumption: next_bar_open_or_vwap` |
| Engine price feed | 1 minute (`config/backtest/base.yaml::calendar.data_frequency: 1m`); a position is watched every minute between decisions, which is what stops and take-profits are measured on |
| CV | `evaluation`: `n_splits: 2`, `train_size: 6M`, `val_size: 6M`, `calendar: NYSE`, `periods_per_year: 252` (daily MTM despite intraday cadence); 13,094 decision bars from 2020-01-02; last validation ends 2021-06-30 |
| Holdout | `holdout_start: '2021-07-01'`, `holdout_end: '2021-12-31'`; untouched until notebooks 18-20 |
| Cost model | `costs.class: dominant`; `model: per_share_plus_spread`; `per_share: 0.0035` (IBKR Pro Tiered top tier, <=300K sh/mo); `minimum: 0.0`; `asset_spreads_source: liquidity_profile.parquet`; `asset_spreads_column: median_half_spread_usd`; `default_half_spread_usd: 0.025` (universe p75 fallback); `spread_convention: half_spread`; `friction_floor_bps: 5`; spreads from AlgoSeek NBBO close quotes 2020-01-02 to 2021-07-01 |
| Execution (spec-hash inputs) | `execution.initial_cash: 1_000_000`; `share_type: integer`; `allocator_lookback: 520` (rows of the 1-minute price frame, ~1.3 sessions, NOT 20 trading days); read via `get_backtest_config()`; any change invalidates every `backtest_hash` |
| Mapping | `mapping.class: intraday_rank_and_trade`; `position_state_space: long_short`; `entry_logic: rank_or_threshold`; `sizing: dollar_neutral_or_beta_neutral` |
| Benchmark | Full-universe equal weight (1/N) under the `backtest.rebalance.benchmark` profile (`min_weight_change: 0.0`, `min_trade_value: 0.0`, so it rebalances at all); paired day-by-day comparison in `20_strategy_analysis` |
| Rebalance skip | `backtest.rebalance.default`: skip when weight change < `0.005` AND trade notional < `100.0` USD |
| Features | `features.storage_dtype: float32`; `windows` (minute bars, session-bounded): `fast: 5`, `decision: 15`, `slow: 30` (realized-vol horizon, Amihud average), `hour: 60` (Kyle's lambda regression, slow regime average), `ewma_half_life: 30`, `stale_cap: 5`, `edge_block: 30`; `carrier: signed_vol_share_15m` |
| Model-based | `model_based.har`: `components: [5, 15, 60]`, `fit_window: 120`, `refit_every: 1`, `min_train_obs: 20`; `spectrum`: `window: 60`, `low_frequency_period: 20`; `signature`: `window: 30` |
| Modeling | `modeling.dl.device: gpu`, `train_sequence_stride_horizons: 1` (mutually exclusive with `max_train_sequences`); `modeling.gbm`: `libraries: [lightgbm]`, `preset: default`, `device: cpu`, `max_bin: 255` |
| Ensemble | `ensemble.member_family: gbm`, `max_num_leaves: 31`, `method: mean_forecast`, `config_name: gbm_mean_leaves31`, `checkpoint: last`, `featured_scheme: slot_l_lq90_s10_h480_noexit_b210`, `universe_filter: cost_feasible` |
| Causal | `causal.treatment: signed_vol_share`, `treatment_window: 1`, `confounders: [rel_spread_close, rv_5m, r1m]`, `method: walk_forward_dml` |
| Sweep | `backtest.sweep.top_n_predictions`: signal 0, allocation 10, cost_sensitivity 1, risk_overlay 1; `expensive_allocators_skip: true`; `top_k_grid` [5, 10, 20] for all four labels; `allocators` equal_weight, score_weighted, inverse_vol (no max_weight cap); `cost_grid_bps: [0,1,2,3,5,7,10,15,20,30,50]`; `cost_grid_half_spread_usd: [0.0,0.005,0.01,0.025,0.05,0.10]`; `cadence_sweep: [15_minute, 30_minute, 1_hour, 4_hour]` (first must equal `decision.bar_frequency`) |
| Running | `uv run python case_studies/nasdaq100_microstructure/NN_name.py` from repo root, 01..20 in order; results bundle `uv run python scripts/download_artifacts.py --cs nasdaq100_microstructure` (`v3.1.0-artifacts` current, `v3.0.0-artifacts` original; hashes re-keyed between bundles, do not resolve across them) |
| Run log | `run_log/registry.db` keyed by hash of the producing spec; artifacts under `run_log/training/`, `run_log/predictions/`, `run_log/backtest/`; README restates no results (registry is rebuilt on re-derivation) |

Feature families (`features.families`; lookback/lag in minute bars; frame per symbol-session, trailing; each in level and cross-sectional z-score representation):

| Family | Pattern | Role | Lookback/lag | Inputs | Declared failure mode |
|---|---|---|---|---|---|
| quote_liquidity | `rel_spread_*`, `depth_imb*`, `quote_rate*` | state | 60/0 | NBBO close quotes, quote update count | carried-forward quote reports a stale spread as a live calm state |
| microprice | `microprice_dev*` | signal | 15/0 | NBBO close prices and sizes (Stoikov 2018) | tenths of a cent on a wide book, rounding artifact on a one-tick book; not comparable until ranked |
| volatility | `r1m*`, `rv_*`, `quote_range*` | state | 60/0 | quote midpoint and extremes | a single bad quote dominates a short-window variance |
| order_flow | `signed_vol_share*`, `tick_imb_share*`, `trade_to_mid*`, `trades_per_1k_shares*`, `cross_locked_share*` | signal | 60/1 | AlgoSeek trade-location and tick buckets, total volume | stale quote attributes volume to the wrong side |
| price_impact | `dollar_vol*`, `illiq*`, `kyle_lambda*`, `trade_range*` | state | 60/1 | consolidated trade prices, dollars across venues | Amihud undefined on zero-volume bars; rolling average carries the gap |
| hidden_liquidity | `finra_share_60m*` | state | 60/1 | FINRA/TRF volume share | TRF prints delayed; not public at bar close unless lagged |
| session_clock | `time_since_open`, `time_to_close`, `is_first_30m`, `is_last_30m` | state | 0/0 | exchange calendar (scheduled open/close) | measured from realized bar count it is a quantity nobody had at the time |

Matrix: 16,098,877 rows x 88 features on the 1-minute grid; float64 = 10.9 GB dataset / 58.9 GB peak per GBM fold; float32 = 5.6 GB / 38.0 GB. 66 features reach the IC triage; `is_first_30m` is identically zero (the hour of warm-up removes the opening block).

## Pipeline stages

| Stage | Notebook | Chapter | What it does | What it writes |
|---|---|---|---|---|
| Feasibility | `01_feasibility_analysis` | Ch6 §6.2-6.6 | Per-symbol round-trip cost, breadth per decision bar vs `BREADTH_FLOOR` 40, move clearance by horizon, return persistence, walk-forward demo; asserts `universe.symbols` against the archive both ways; fits no model | `liquidity_profile.parquet` (115 symbols: `median_half_spread_bps`, `median_half_spread_usd`, `mean_price`, `rt_cost_bps_median`, `cost_rank`) |
| Labels | `02_labels` | Ch7 §7.2 | 5/15/60-min forward midpoint returns + 15-min direction as an execution convention (entry = next-bar VWAP, exit H later) resolved by timestamp | `labels/fwd_ret_5m.parquet`, `labels/fwd_ret_15m.parquet`, `labels/fwd_ret_60m.parquet`, `labels/fwd_dir_15m.parquet`, each with `.digest.json` |
| Features | `03_financial_features` | Ch8 §8.1-8.6 | 7 microstructure families, session-bounded windows, lag 1 on late-published inputs, level + cross-sectional z-score; draws the carrier in F2/F3/F6 | `features/financial.parquet` + `.digest.json` |
| Model-based features | `04_model_based_features` | Ch9 | HAR variance regression (refit every bar), rolling Fourier, depth-2 path signature | `features/model_based.parquet` (no `fold` column) + `.digest.json` |
| Evaluation | `05_evaluation` | Ch7 §7.3-7.4, Ch8 §8.6 | Univariate IC triage of 66 features: per-timestamp rank correlation, one obs per horizon, Newey-West over one session, coverage/repetition/monotonicity/redundancy screens, multiple-testing adjustment; PROCEED/REVISE/STOP | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` |
| Linear | `06_linear` | Ch11 §11.2 | OLS/ridge/lasso/enet (16 configs) and logistic (13) on the richest feature space in the book | runs + prediction sets in `run_log/registry.db`; coefficients `run_log/training/{hash}/`; scores `run_log/predictions/{hash}/` |
| GBM | `07_gbm` | Ch12 §12.2-12.3 | LightGBM on 13M+ samples, 15 configs x 10 tree-count checkpoints; `trees_effect`, `full_coverage` | boosters, `learning_curves.parquet`, `fold_metrics.parquet` under `run_log/training/{hash}/` |
| NLinear | `08_dl_nlinear` | Ch13 | 60-obs window, subtract last value, single linear layer: isolates the gain from seeing a window | checkpoints under `run_log/training/deep_learning/` |
| LSTM | `09_dl_lstm` | Ch13 | `SEQUENCE_CONFIG = "lstm_h64"` (hidden 64, inference from name); 20 epoch checkpoints, each its own prediction set | same layout, own population |
| TCN | `10_dl_tcn` | Ch13 | `SEQUENCE_CONFIG = "tcn"`, causal dilated convolutions | same layout |
| PatchTST | `11_dl_patchtst` | Ch13 | `SEQUENCE_CONFIG = "patchtst"`, patch attention; population `nasdaq100_microstructure-patchtst-validation-v1` | same layout |
| Causal DML | `12_causal_dml` | Ch15 §15.6 | Does `signed_vol_share` cause `fwd_ret_15m`? Walk-forward DML, NW SE, naive comparison, block-permutation refutation | one row in registry `causal_runs` (primary label only) |
| Model analysis | `13_model_analysis` | -- (bridges Ch11-15 and Ch16-20) | Cross-family IC comparison on full-coverage representatives, fold stability, conformal coverage; opens via `read_only_study` | nothing |
| Backtest | `14_backtest` | Ch16 §16.4-16.8 | Random-signal plumbing test; two-pass signal sweep (3,663 backtests); selects on validation backtest Sharpe; DSR / family / IC-to-Sharpe via `BacktestExplorer` | per prediction set x entry scheme: `daily_returns.parquet`, `weights.parquet`, `trades.parquet`, `fills.parquet`, `equity.parquet`, `portfolio_state.parquet`, `spec.json` under `run_log/backtest/{hash}/` |
| Portfolio | `15_portfolio_management` | Ch17 §17.2-17.8 | Top-10 predictions x `TOP_K_VALUES` x 3 fast allocators on the full universe, every 15-min bar; `BUDGET_SECONDS 3600` | one run per allocator, same layout |
| Risk | `16_risk_management` | Ch19 §19.3-19.6 | Stop-loss / trailing (incl. MAE/MFE-calibrated) / time-exit overlays on top-1 of {signal+allocation}; evaluated on the 1-min watch clock | one run per overlay, same layout |
| Costs | `17_costs` | Ch18 §18.2-18.5 | bps cost grid on top-1 of each pre-cost stage; full-vs-cost-feasible contrast; cadence x per-share cost heatmap; LAST stage that selects | one run per cost level, same layout |
| Holdout predictions | `18_holdout_predictions` | Ch20 | Refit `resolve_solvent_carrier` config with training ending one label buffer before 2021-07-01; asserts new training hash | one training run, one prediction set `split='holdout'` |
| Holdout backtest | `19_holdout_backtest` | Ch20 | One backtest with settled concentration, allocator, overlay, cadence, cost; asserts `registered_stage == "holdout"` | one run at `stage='holdout'` |
| Strategy analysis | `20_strategy_analysis` | Ch20 §20.1 | Block-bootstrap intervals, paired comparison vs equal-weight benchmark, gates, lineage; `PERIODS_PER_YEAR = 252` | `results/strategy_assessment.json`, `20_strategy_synthesis/output/nasdaq100_microstructure/nasdaq100_microstructure_tearsheet.html` |

Stage constants: `STAGE_SEQUENCE` = signal, allocation, risk_overlay, cost_sensitivity; `17_costs` defines `PRE_COST_STAGES = STAGE_SEQUENCE minus "cost_sensitivity"`. Prediction population: 741 admissible sets = 226 per continuous label (3 DL families x 20 epochs + 15 GBM x 10 checkpoints + 16 linear) + 63 for `fwd_dir_15m`.

## Design decisions and why

| Decision | Setting | Why |
|---|---|---|
| Measure cost per symbol, not one average | `liquidity_profile.parquet`; `rt_cost_bps = 2*(per_share/mean_price)*1e4 + 2*median_half_spread_bps` | A spread is cents, so bps cost differs across the cross-section; at 15 min the error is the size of the quantity measured |
| Cost-feasible screen, frozen per split | top-50 by `rt_cost_bps`, `_build_cost_feasible_universe.py` (`TOP_N=50`) | Full universe collapses out of sample (holdout -0.42); screening on the traded window would be in-sample liquidity. A tradability screen carries no return information |
| Friction floor is the optimistic end | `friction_floor_bps: 5` vs measured median round trip 6.16 bps | Quoted spread + commission is a floor; impact and queue position enter only at `17_costs` |
| Midpoint returns everywhere | drop crossed / non-positive quotes | Traded closes add bid-ask bounce, an artifact of which side traded |
| Execution convention label | entry = VWAP of the bar after the decision, exit = H past entry, resolved by timestamp not row shift | A label at t consumes a quote at t+H+1; hence `buffer = H+1` |
| Declare horizons explicitly | `labels.horizons` | `resolve_label_horizon` falls back to the buffer; without the declaration `fwd_ret_15m` becomes a 16-minute return everyone still calls fifteen |
| One slot = one holding period | `rebalance_step: 1`, `cadence_by_label` | `resolve_rebalance_timestamps` now builds the schedule from the declared cadence (previously raw panel timestamps, so three labels traded every minute, agent-workspace#187) |
| Session-bounded windows | `frame: per symbol-session, trailing`; each session restarts warm-up | A window spanning the overnight gap measures the gap, which on a minute grid dwarfs the signal |
| Session from scheduled close | `session_clock` from the exchange calendar | Realized bar count is unknowable until session end; vendor pads early-close days to the full grid |
| Lag 1 on late-published families | `order_flow`, `price_impact`, `hidden_liquidity` `lag: 1` | TRF prints arrive ~10 s late; deferring a 60-min average by one bar costs nothing |
| Stale-quote cap | `stale_cap: 5` | Beyond 5 carried-forward bars the spread is nulled; bars inside the cap are accepted as-is |
| Two representations per family | level + cross-sectional z-score ranked within the minute | Microprice deviation and similar are not comparable across names until ranked |
| float32 storage, this case study only | `storage_dtype: float32` | 10.9 GB -> 5.6 GB; LightGBM bins to uint8 anyway; smaller case studies keep float64 |
| Refit HAR every bar | `refit_every: 1`, `fit_window: 120`, `min_train_obs: 20` | The estimation window is part of the feature; coefficients for bar t come from a regression ending t-1, so no fold column is needed |
| Linear first, ridge as the answer | 16-config penalty grid | Linear IC is what features give on their own; collinear order-flow views want a dense penalty; the ridge peak-to-OLS gap shows what multicollinearity buried |
| GBM grid over leaves x loss | {default, 7, 15, 31, 63} x {mse, mae, huber}; CPU `max_bin: 255` | Expected interaction imbalance x spread; grid measures whether the interaction is worth a greedy splitter's instability on near-copy features |
| Checkpoint = part of the config | 10 GBM, 20 DL checkpoints each registered | Keeping the best iteration reports the max of N numbers as one |
| NLinear before LSTM/TCN/PatchTST | 60-obs window, subtract last value, linear head | Separates the gain from seeing a window from the gain from architecture |
| DL stride = one horizon | `train_sequence_stride_horizons: 1` | Consecutive windows carry non-overlapping labels; replaces `max_train_sequences: 750000` (a window every 5.4 min, 2.8x re-presentation, agent-workspace#1015) |
| Select on validation backtest Sharpe, never IC | `top_n_predictions.signal: 0`; `mechanism_top_n: 8` by pass-1 Sharpe | `load_prediction_index` orders `ic_mean DESC`; IC measures ranking, not tradability after costs and turnover |
| Two-pass sweep | pass 1: 741 x [ew_top5, ew_top10, ew_top20] on cost_feasible = 2,223; pass 2: top-8 per label (32) x 42 arms = 1,344 + 32 x 3 reference arms on full = 96; total 3,663 ~ 32 h at 31.8 s | Full cross (117 arms x 2 universes x 741) = 173,394 backtests ~ 1,500 h; the arm count was the problem, not the prediction count |
| Slot mechanism as the signal stage | `slot_persistent_signal_exit`: `long_q` [0.90, 0.95, 0.99], `max_slots` [5, 10, 20], `hold_bars` [120, 480] minute bars, `exit_signal_q` [null, 0.50], `bars_per_day_grid` [210], `lookback_days: 21`, `pred_freshness_max_min: 14`, long_only | Slots ARE the allocation (Ch17 skipped for slot rows); slot x long_short dropped because a slot book is single-direction |
| Long-short possible via `eq_w_topk` | `top_k` [5, 10, 20] x `direction` [long_only, long_short] | Large, heavily traded names: the short leg borrows about as cheaply as the long; holding both ends cancels the market move a cross-sectional score cannot predict |
| Only fast allocators | `expensive_allocators_skip: true` | Covariance estimation on a 1.3M-bar rebalance schedule is prohibitive and degenerate (Sharpe -7 to -10); a 6-month equivalent needs ~3,300 bars |
| Ensemble declared in config | `ensemble` block, 12 of 15 GBM presets (num_leaves <= 31), `checkpoint: last`, one featured scheme | The member rule IS the object; members resolve from the registry; one arm only so it is not one more config to select among |
| `initial_cash: 1_000_000` | with `share_type: integer` | 100k gave avg Sharpe -11 across 339 runs from rounding on BKNG (~$5k), NFLX/AVGO ($700-1.5k) |
| Holdout = refit, read once | `18` refits with `train_end` one `label_buffer` before `val_start`; `19` one backtest | Scoring the validation-fitted model on holdout is circular; evaluating a second configuration spends a second holdout observation |
| Daily MTM annualization | `periods_per_year: 252` | Sharpe annualized at daily grain independently of the 1-min engine feed or 15-min cadence |

Linear and GBM menus (`config/training/{label}.yaml`; presets in `case_studies/config/{model_type}/`; comment out lines to skip):

| Label | Linear | GBM | DL |
|---|---|---|---|
| fwd_ret_5m | `ols`, `ridge_a{0.001..10000000}` (11), `lasso_a{0.01,0.1}`, `enet_a{0.01,0.1}` | 15: leaves {default, 7, 15, 31, 63} x loss {mse, mae, huber} | `nlinear`, `lstm_h64`, `tcn` (no patchtst) |
| fwd_ret_15m, fwd_ret_60m | same 16 | same 15 | `nlinear`, `lstm_h64`, `tcn`, `patchtst` |
| fwd_dir_15m | `logistic_none`, `logistic_l2_C{0.001..100}`, `logistic_l1_C{0.001..100}` (13) | `default_multiclass`, `leaves_{7,15,31,63}_multiclass` | none (sequence runner is regression-only; agent-workspace#396) |

Huber threshold derives from each fold's own label spread. `default` = LightGBM's 31 leaves with no `lambda_l1`/`lambda_l2`/`bagging_fraction`/`feature_fraction`.

## Market-specific guardrails

Data grid and sessions

- Measure the observation grid from timestamp spacing; never divide a duration by `decision.bar_frequency`. Reading it as the bar size makes embargoes, permutation blocks and period counts 15x too small with nothing in the result showing it (`allocator_lookback: 520` is 520 minutes).
- Bound every trailing window by the symbol-session; restart warm-up each session; assert warm-up rows (feasibility section D.2).
- Bound sessions by the exchange's published schedule before building forward windows; drop bars stamped after the scheduled close (vendor padding on early-close days passes uniform-spacing checks and feeds padded exit prices to the last `horizon` genuine bars).
- Compute all returns from the NBBO midpoint; drop crossed and non-positive quotes.
- Tolerate at most `stale_cap: 5` consecutive stale-quote bars, then null; trade location against a stale quote attributes volume to the wrong side.
- Lag `order_flow`, `price_impact`, `hidden_liquidity` by one bar (TRF prints are delayed).
- Sanity-check quote extremes before trusting short-window variance features (inference); a single bad quote reports turbulence where there was a data error.
- Do not rename a single-bar |return|/dollar-volume ratio as Amihud; call the library estimator (a window average that skips zero-volume bars).
- Eligibility: a name must have at least one session before `holdout_start`; `01_feasibility` asserts the declared list against the archive in both directions and stops on mismatch. Names first quoting inside the holdout would trade at a never-measured cost.
- Run production with `MAX_SYMBOLS=0`; a capped run ranks cross-sectional statistics within a different universe and computes a different quantity, not a smaller one (CI reads capped matrices for shape only).

Labels and leakage

- Set the purge to H+1 (16/6/61/16 min), not H: entry at next-bar VWAP and exit H later consumes t+H+1.
- Declare `labels.horizons` explicitly so `resolve_label_horizon` cannot fall back to the buffer.
- Seal the holdout at `holdout_endpoint_cutoff` = holdout_start minus one horizon (computed, not typed), within session; `18` raises if the validation window closes at or after holdout open.
- Resolve walk-forward windows FIRST and route every diagnostic through one validation-only slice: `features/model_based.parquet` carries no fold column, so any readout from the raw frame spans the holdout.
- Price the overlap: on a 1-minute matrix each 15-min label is seen 15 times; sample IC one observation per horizon, use Newey-West over one session of lags, report the effective row count (t-stat inflation factor ~3).
- Trade one bar after the scored bar (`execution_delay: 1_bar`, `next_bar_open_or_vwap`); trading the scored close restates the signal's own price.
- Prove feature causality by rebuilding the matrix on a panel that stops early and requiring exact agreement on shared rows (costs ~3 symbols' worth of runtime; cannot pass by accident).
- Derive model-based feature splits from the same frame the consumer uses; several horizons mean several sealed gaps; check coverage where the artifact is written.

Evaluation and multiple testing

- Sort the per-date IC series by date before Newey-West; Polars `group_by` returns arbitrary order and NW on a permutation of time looks like good news.
- Require >= 10 names per timestamp and >= 20 observations per feature series before reading an IC.
- Fix the multiple-testing search set by the rules that generated the features, not by what the screen found.
- Expect IC in the thousandths at intraday horizons (daily equity signals report hundredths); check `by_label` whether IC rises with horizon.
- Known triage limits: the fold-agreement screen counts positive-IC folds only (sign-flip before screening, inference); the repetition screen fails discrete/flag columns, so interpret STOP on session flags accordingly.
- Nothing downstream filters on the triage ledger; only `20_strategy_synthesis/02_feature_evaluation.py` reads it.

Models and model selection

- Never select on IC: keep `top_n_predictions.signal: 0`; `load_prediction_index` ends `ORDER BY m.ic_mean DESC NULLS LAST`, so a positive N screens on rank correlation before any backtest.
- Read `ic_n_days` / `full_coverage` before `ic_mean`: aggressive L1 configs and some GBM checkpoints predict near-constants at some decision times and the average is over a self-selected sample; figures draw full-coverage configs only.
- Treat every checkpoint as its own configuration; never keep a config's best iteration; ensemble members enter at `checkpoint: last`.
- Compare rankings across `by_label` before transferring a config between 5/15/60-min horizons; where orderings disagree the grid measures the horizon.
- Use one representative = (config, checkpoint) per family, restricted to full decision-day coverage, before ranking in `13`; overlapping IC intervals on two folds = not separated.
- Check conformal coverage on a later window separately; residual scale may not transport.
- Keep `causal_runs` out of any predictive ranking or family coverage; different estimand and scale.
- GBM on CPU uses `max_bin: 255`, not the GPU default 63 carried over.
- DL: declare the device (`modeling.dl.device: gpu`; override only via notebook `DEVICE`, which is then registered); never combine `train_sequence_stride_horizons` with `max_train_sequences`; DL and DML menus must not list `fwd_dir_15m` (`deep_learning.py:374` refuses non-regression labels; DML nuisances are regressors).
- Sequence families score fewer rows than tabular ones (a prediction needs a full 60-obs window behind it); say so in comparisons.
- Declare the ensemble in `setup.yaml` (`member_family`, `max_num_leaves`, `method`, `checkpoint`); resolve members from the registry; never assemble it ad hoc or write a member count into config.
- DML: `causal.treatment_window: 1` so the placebo block covers the treatment's construction window; print `block_size_basis`; report parametric SE and permutation p-value both and distrust the parametric one when they disagree; apply sample caps only as declared preview reductions (`max_symbols`, `cv_folds`, `max_samples`, `n_placebo`) that write to `.preview/<case>` under a different identity.
- HAR variance forecasts can extrapolate negative; measure the share (section C.1) and model log-variance as the remedy. Model-based aggregation windows still cross session boundaries in this version (measured at the top of section C, not fixed).
- Accept that `is_first_30m` is dead (section E asserts it); the busiest block of the day is not representable after the hour of warm-up.

Backtest, allocation, risk, costs

- Run the plumbing test first: random scores through the same engine must be clearly negative after costs; profit means the pipeline (fill timing, join, cost model) is the alpha.
- Rank predictions on `baseline_universe: cost_feasible`; the full universe is a reference arm only (never a rank-1 / cohort / DSR candidate).
- Read `get_backtest_config()` for `initial_cash`, `share_type`, `allocator_lookback`; never declare a local constant; a change invalidates every `backtest_hash`.
- `hold_bars`, `bars_per_day` and risk `time_exit` bars count rows of the 1-minute price grid (the old 8/16/32 and 14 silently meant 8-32 minutes).
- Read a linear slot count as an upper bound: entries are ranked and admitted up to free capacity, so only `long_q: 0.90` binds on linear predictions (4-5 distinct outcomes per 36 rows; DL 36/36; GBM 35.5).
- Skip covariance-based allocators (risk_parity, mvo_ledoit_wolf, hrp) at 15-min cadence; no allocator rescues an every-bar strategy because allocators set sizes, not turnover.
- Sweep only position-level risk rules; keep `MaxDrawdownLimit` / `DailyLossLimit` as governance (a permanent halt truncates the series and yields zero-std Sharpe artifacts). Read the gradient across thresholds, not one row; tight stops at short cadence fire on ordinary fluctuation and each exit is an unrequested round trip.
- Vary universe and cadence one at a time (`17` §4 vs §5); compare the same predictions on both sides of the screen (`inner join on (prediction_hash, arm)`).
- Read the cadence x cost sweep as a cross-cadence comparison under one simplified cost shape (one $/share split evenly into commission and slippage); production is fixed per-share commission plus measured per-asset half-spread.
- Always report the breakeven cost with any result; at 15 min the cost decides the sign.
- Open the study correctly: `read_only_study` in 13 and 20; writing notebooks (14-19) call `open_study` before binding `CASE_DIR` or reading the registry (`Study.activate()` rewrites `ML4T_OUTPUT_DIR` and clears caches; a preview tier opened the wrong way reported 30 -> 0 prediction sets while claiming success).

Holdout and reporting

- Sequence 17 (selects) -> 18 (refit; assert holdout training hash != validation training hash) -> 19 (one backtest; assert `registered_stage == "holdout"`) -> 20. Nothing is chosen after 17.
- Re-resolve `resolve_solvent_carrier` and the training identity in each of 17/18/19; a copied hash goes stale on the next sweep rebuild.
- 18/19 refuse a different configuration; re-running the same one is idempotent; a changed selection goes through registry lifecycle (`holdout_generations_to_retire`), never row deletion. `registered_holdout_generations` flags "VALIDATION-FITTED - not out of sample".
- An ensemble carrier is refused at holdout: its adapter lacks `rekey_holdout_spec`, `reconstruct_locked_request`, `validate_locked_run` (every member would need refitting).
- Compare strategies by paired day-by-day differences with a block bootstrap over contiguous day blocks (`compute_bootstrap_ci`); never subtract summary Sharpes.
- Deflate the validation maximum (DSR) in 20; the top of 3,663 backtests is partly chance and the holdout is one 6-month measurement; state the gate (`gate2_holdout_diff_not_excludes_zero_negatively`) so it can fail.
- Validation folds are read many times; treat every 06/07 IC as a diagnostic, not evidence.

## Results and lessons

Numbers available from the notes (exact cost-sweep cells, breakeven levels, DSR values and the positive cost-feasible holdout Sharpe live in the registry / bundle, not in the digest):

| Quantity | Finding |
|---|---|
| Breadth per decision bar | declared universe 101-103 symbols; cost-feasible 46-50; the 40 needed by the largest portfolio (2 sides x top_k 20) available at every one of 9,778 decision bars |
| Round-trip cost | measured median 6.16 bps across the universe, above the 5 bps friction floor (floor = optimistic end) |
| Move clearance | median absolute midpoint move 9.1 bps (5 min), 15.8 bps (15 min), 30.6 bps (60 min) vs ~6 bps round trip |
| Return persistence | within-symbol, within-session autocorrelation of midpoint returns: negative result; "buy what just went up" ruled out; anything predictive must come from order flow and quotes |
| Overlap correction | HAC / effective-count correction deflates the label diagnostic t-statistic by nearly 3x |
| Feature matrix | 16,098,877 x 88; 66 triaged; `is_first_30m` identically zero |
| Linear | IC a few thousandths at 15 min; ridge curve rises then falls with alpha; optimal alpha differs by horizon; aggressive L1 posts high raw IC with incomplete coverage |
| GBM | expected spread x imbalance interaction; whether interior checkpoints win is in `trees_effect` (values in registry) |
| DL rows | fold 0 4,066,041 and fold 1 3,997,571 training rows across 115 symbols; primary label ~271,000 windows on fold 0 at stride 15 |
| Prediction population | 741 admissible sets (226 per continuous label, 63 for `fwd_dir_15m`); one backtest 31.8 s (29.2-36.3 s) on the 10.1M-row price frame |
| Plumbing test | random signal clearly negative (pays the spread every rebalance), as required |
| Universe effect | featured config holdout Sharpe -0.42 on the full 115-name universe; positive on the cost-feasible 50; `17` §4 prints `d_sharpe = avg_sharpe_screened - avg_sharpe_full` and `trade_ratio` |
| Allocators | every allocator in the full-universe every-bar sweep deeply negative; risk_parity / mvo_ledoit_wolf / hrp Sharpe -7 to -10 on every label; equal_weight ties at the top |
| Capital sizing | `initial_cash` 100k gave avg Sharpe -11 across 339 signal runs; restored to 1M on 2026-05-16 |
| Slot grid distinctness | DL 36/36 distinct outcomes, GBM 35.5/36, linear 4-5/36 (288 linear rows carry 36 distinguishable results; measured 2026-09-24) |
| Cadence x cost | slower cadence spreads entry cost over longer holds (publication heatmap; cell values in registry) |
| Ensemble | 12 of 15 GBM members per continuous label; configuration choice on validation does not carry to holdout, the family average does |
| Validation vs holdout | validation figure = maximum over >1,000 backtests; holdout = one 6-month measurement; 2020-2021 contained unusual conditions; two folds give wide IC intervals |

Lessons

- Cost measurement is the analysis: at 15 min, per-trade edge and the spread are the same order; the cost-feasible screen, not model choice, decided the holdout sign.
- Turnover is a strategy decision: at this cadence two trading rules on one ordering differ more than two models; ranking quality does not map to traded outcome.
- Allocation and risk overlays are second-order to turnover here; allocators decide sizes, overlays buy a tighter loss distribution by trading more.
- The ensemble buys stability, not strength; model selection is estimation under uncertainty.
- Kill conditions declared in advance (feasibility C.2): no feature IC distinguishable from zero across folds; expected gross edge below the measured round trip; signal decay inside the one-bar execution delay.

## How to adapt this pattern to a new dataset

1. Pin the observation grid and the decision cadence as two separate facts; measure the grid from timestamp spacing and store the cadence in `decision.bar_frequency` / `cadence_by_label`. Size every embargo, permutation block, lookback and hold count in grid rows.
2. Build a per-symbol liquidity profile first (`median_half_spread_bps`, `mean_price`, `rt_cost_bps`), sorted ascending with a `cost_rank`; keep a `default_half_spread_usd` fallback at the universe p75. Use the reusable Polars pattern:
   ```python
   rt_cost = 2 * pl.col("median_half_spread_bps") + 2e4 * PER_SHARE_USD / pl.col("mean_price")
   liquidity_profile = (bars.group_by("symbol")
       .agg(median_half_spread_bps=(pl.col("half_spread_usd")/pl.col("mid")*1e4).median(), ...)
       .with_columns(rt_cost_bps_median=rt_cost).sort("rt_cost_bps_median"))
   ```
3. Plot clearance curves per horizon: `m{h} = h{h} / rt_cost_bps_median`, exceedance on a log axis, break-even at 1. If the median move does not clear the round trip by a comfortable multiple, slow the cadence before building features.
4. Check breadth per decision bar against `2 x max top_k`; revise `universe.symbols` when breadth falls under the sweep's positions on either leg.
5. Test return persistence within symbol-session; a negative result means the signal must come from fields the return summarizes away.
6. Freeze a cost-feasible list per split from data strictly before that split; select it by a named `universe_filter` so the spec hash carries the name, not the list.
7. Write labels as an execution convention resolved by timestamp (entry next-bar VWAP, exit H later); set `buffer = H+1`; declare `horizons` explicitly; set `rebalance_step` so one slot is one holding period; seal diagnostics at holdout_start minus horizon within session.
8. Declare feature families with `pattern`, `role`, `lookback`, `lag`, `frame`, `representation`, `failure_mode`; bound windows by symbol-session; lag any input published late; cap stale quotes; add a cross-sectional z-score twin for scale-dependent features; use midpoint returns.
9. Choose `storage_dtype` from measured memory (float32 when the float64 dataset exceeds what a fold can hold).
10. Refit model-based features at the data cadence (`refit_every: 1`) with a declared `fit_window` and `min_train_obs`; then no fold column is needed. Prove causality by rebuilding on an early-stopped panel and requiring exact agreement.
11. Run univariate IC triage with the overlap priced (one obs per horizon, NW over one session, sorted by date, min 10 names / 20 obs); fix the multiple-testing search set in advance; carry PROCEED and REVISE, do not filter downstream on the ledger.
12. Fit linear first (ridge for collinear features), then GBM over leaves x loss with CPU `max_bin: 255`, then NLinear before heavier sequence models; register every checkpoint; set DL stride to one horizon; declare the device.
13. Keep `top_n_predictions.signal: 0`; rank on validation backtest Sharpe on the cost-feasible universe; if the arm grid is unaffordable, use a two-pass design (`baseline_schemes` on all predictions, `mechanism_top_n` per label x the mechanism grid, canonical reference arms on the full universe).
14. Run the random-signal plumbing test before reading any backtest.
15. Set `initial_cash` so integer-share rounding cannot zero out high-priced names; read it from `get_backtest_config()`.
16. Skip covariance allocators at intraday cadence (`expensive_allocators_skip: true`); sweep only position-level overlays; keep drawdown / daily-loss limits as governance.
17. Price costs last: bps grid, per-share x cadence heatmap, full-vs-screened contrast with the same predictions on both sides; report the breakeven cost.
18. If the lesson is model uncertainty, declare an ensemble rule in config (`member_family`, `max_num_leaves`, `method`, `checkpoint: last`, one `featured_scheme`) and let members resolve from the registry.
19. Refit the carrier for holdout with training ending one buffer before holdout open; assert a new training hash; backtest once at `stage='holdout'`; report block-bootstrap intervals, paired comparison vs benchmark, DSR, and a pre-stated gate.
20. Write kill conditions before fitting: IC indistinguishable from zero across folds; gross edge below measured round trip; signal decays inside the execution delay.

Key APIs and config helpers (homes marked "inference" are from import blocks, not confirmed):

| API | Role |
|---|---|
| `load_nasdaq100_bars()`, `load_bars(frequency="1m")` | AlgoSeek bars with midpoint, stopped short of `holdout_start` |
| `utils.modeling.load_modeling_dataset` | primary label + features from `06_linear` on |
| `get_backtest_config()` | `initial_cash`, `share_type`, `allocator_lookback`, `.cadence`, `.long_short` |
| `get_top_n_predictions(case, stage)`, `get_universe_filters_for(case)` (first = canonical `cost_feasible`), `get_top_k_values_for(case, label, n_assets)`, `get_checkpoints_per_config(case)`, `get_cost_grid_bps`, `get_cost_grid_half_spread_usd`, `get_cadence_sweep`, `get_entry_schemes_for` | `case_studies/utils/sweep_config.py` (inference) |
| `load_prediction_index` | ends `ORDER BY m.ic_mean DESC NULLS LAST` |
| `resolve_label_horizon`, `resolve_rebalance_timestamps` | horizon fallback to buffer; schedule from `cadence_by_label` |
| `load_backtest_prices_for(case, label, split=..., max_symbols=...)`, `_compute_rolling_vol` -> `rolling_std(vol_window)`, `traded_universe_declaration(prices)` | 60-second price frame |
| `case_studies/utils/slot_strategy.py` | slot mechanism; sub-params under `selection_method_config` |
| `case_studies/utils/deep_learning.py::resolve_model_request` | sequence runner, regression only |
| `case_studies/utils/causal.py` | DML nuisances as `HistGradientBoostingRegressor` |
| `_build_cost_feasible_universe.py::top50_from_existing()` | reads `symbol, median_half_spread_bps, mean_price` |
| `BacktestExplorer`, `backtest_run_status(case, prediction_hash, spec).backtest_hash`, `read_predictions(case, hash)` | registry readers |
| `open_study` / `Study.activate()` / `Study.regenerate` / `read_only_study` -> `Study.at(root)`, `get_case_study_dir(case)`, env `ML4T_OUTPUT_DIR`, preview root `.preview/<case>` | study lifecycle |
| `calibrate_trailing_stops(prices)`; `MaxDrawdownLimit`, `DailyLossLimit`; spec `portfolio_limits=[{"type":..,"threshold":..}]` (`rule_key = "bars"` for time_exit) | risk overlays |
| `resolve_solvent_carrier(case)` -> `label, family, config_name, val_backtest_hash, val_prediction_hash` + checkpoint fields ("solvent" meaning inferred) | carrier for holdout |
| `build_holdout_training_spec(...)`, `training_hash_from_spec(spec)`, `holdout_generations_to_retire(CASE_DIR, this_generation=...)`, `registered_holdout_generations(CASE_DIR)` | `case_studies/research/holdout.py` |
| `holdout_conformal_embargo_steps(case, label)`, `compute_holdout_conformal_widths(...)`, `ensure_conformal_calibration_identity(spec, holdout_embargo_steps=...)` | only if `allocation.method == "conformal_weighted"` |
| `compute_bootstrap_ci`, `gate2_holdout_diff_not_excludes_zero_negatively`, `get_output_dir(20, case)` | strategy analysis |

Cadence x cost spec pattern (`17_costs` section 5):

```python
spec["strategy"]["signal"] = {"method": "equal_weight_top_k", "top_k": 20}
spec["strategy"]["rebalance"]["cadence"] = cadence           # 15_minute|30_minute|1_hour|4_hour
spec["backtest_config"]["commission"]["model"] = "per_share"
spec["backtest_config"]["commission"]["per_share"] = cost_ps / 2   # other half as slippage
spec["cadence_sweep"] = True
```

Notebook parameters: `MAX_SYMBOLS` (0 = all), `LABELS = []` (every label declaring the family), `DEVICE`, `EXECUTION_TIER="canonical"`, `SPLIT="validation"`, `TOP_K=0` (first feasible), `FORCE_REBACKTEST=False`, `MAX_ENTRY_SCHEMES=0`, `ENSEMBLE_MEMBER_COUNT=None`, `BUDGET_SECONDS=3600`, `MAX_RISK_VARIANTS=0`, causal `RANDOM_SEED=42`, `CV_FOLDS=0`, `MAX_SAMPLES=0`, `N_PLACEBO=0`. Docker image for 06/07: `ml4t`.

## Related references

- `chapters/06_strategy_definition.md` - feasibility: cost measurement, breadth, clearance, kill conditions, run-log accounting.
- `chapters/07_defining_the_learning_task.md` - forward-return labels, H+1 purge, overlap, IC triage and multiple testing.
- `chapters/03_market_microstructure.md` - NBBO, midpoint returns, trade classification, microprice, spread and impact.
- `chapters/08_financial_features.md` - the seven microstructure families, session-bounded windows, lagging late data.
- `chapters/09_model_based_features.md` - HAR, rolling spectrum, path signatures refit every bar.
- `chapters/11_ml_pipeline.md` - regularized linear baseline and why ridge wins on collinear features.
- `chapters/12_gradient_boosting.md` - LightGBM grid, checkpoints as configurations, coverage.
- `chapters/13_dl_time_series.md` - NLinear, LSTM, TCN, PatchTST; window and stride.
- `chapters/15_causal_estimation.md` - walk-forward DML, placebo blocks, treatment window.
- `chapters/16_strategy_simulation.md` - plumbing test, two-pass sweep, selection on backtest Sharpe.
- `chapters/17_portfolio_construction.md` - why covariance allocators fail at intraday cadence.
- `chapters/18_transaction_costs.md` - cost decomposition, breakeven, cadence x cost, screen contrast.
- `chapters/19_risk_management.md` - position overlays on the watch clock; kill switches as governance.
- `chapters/20_strategy_synthesis.md` - holdout refit, paired bootstrap comparison, DSR, gates.
- `chapters/01_process_is_edge.md` - registry, hashes, read-once holdout discipline.
- `case_studies/sp500_options.md` - the `liquid` universe screen this case study's `cost_feasible` mirrors.
- `case_studies/us_equities_panel.md` - daily-cadence equity counterpart for comparing IC scale (hundredths vs thousandths).
- `libraries/ml4t_backtest.md` - engine, cost presets, `get_backtest_config`, rebalance thresholds.
- `libraries/ml4t_engineer.md` - feature estimators (Amihud, Kyle's lambda, realized vol).
- `libraries/ml4t_diagnostic.md` - IC with Newey-West, DSR, block bootstrap, conformal coverage.
- `libraries/ml4t_models.md` - linear / GBM / sequence runners and checkpoints.
- `libraries/ml4t_data.md` - bar loading, calendars, point-in-time universes.
- `workflow.md`, `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `companion_repo.md` - cross-cutting procedure, the general guardrail list, selection rules, evidence table, and repo layout.
- Further reading:
  - Stoikov (2018), microprice.
  - Corsi, HAR model of realized variance.
  - Amihud illiquidity measure.
  - Kyle's lambda (price impact per unit signed order flow).
  - Lee-Ready trade classification.
  - Newey-West HAC standard errors.
  - Path signatures (depth-2 cross terms).
  - Double machine learning (cross-fitted residual-on-residual estimation).
  - Deflated Sharpe ratio; block bootstrap for paired strategy comparison.

## Glossary

- Round-trip cost proxy - `2*(per_share/mean_price)*1e4 + 2*median_half_spread_bps`, in bps, per symbol.
- Cost-feasible universe - top-50 cheapest-to-trade names by the proxy, frozen per split, no look-ahead.
- Friction floor - `friction_floor_bps: 5`, the optimistic end of the declared round-trip cost.
- Decision bar - a timestamp on the 15-minute decision schedule; one common observation across the panel.
- Watch clock vs decision clock - 1-minute engine price feed vs the 15-minute decision cadence.
- Execution convention - label defined by entry (next-bar VWAP) and exit (H after entry), resolved by timestamp.
- Buffer / purge - H+1 bars sealed between train and validation.
- holdout_endpoint_cutoff - last admissible decision time = holdout start minus one horizon.
- Embargo - rows removed between train and validation, sized from the label horizon in grid rows.
- Symbol-session - the entity trailing windows are bounded by (one symbol within one trading session).
- Stale cap - consecutive carried-forward quote bars tolerated (5) before nulling.
- Carrier (feature) - the thesis feature `signed_vol_share_15m`.
- Carrier (strategy) - the configuration carried through holdout, from `resolve_solvent_carrier`.
- Microprice - depth-weighted price leaning toward the thin side of the book.
- Kyle's lambda - price impact per unit of signed order flow (60-min regression).
- Amihud illiquidity - window average of |return| / dollar volume.
- HAR - heterogeneous autoregression of realized variance over components [5, 15, 60] bars.
- Checkpoint - a scored point (tree count or epoch) along one fit; part of the configuration identity.
- Population - the group of prediction sets one DL notebook publishes together.
- Stride - `train_sequence_stride_horizons`: spacing of DL training windows in label horizons.
- Coverage (`ic_n_days`, `full_coverage`) - number / flag of decision times at which a config produced a usable IC.
- Effective row count - independent observations after pricing forward-window overlap.
- Triage ledger - per-feature PROCEED / REVISE / STOP record with evidence.
- Slot strategy - fixed weight-per-slot book, per-symbol rolling-percentile entry, max hold, optional signal exit.
- Mean-forecast ensemble - `gbm_mean_leaves31`: average of member predictions at their last checkpoint.
- Expensive allocators - risk_parity, hrp, mvo_ledoit_wolf (covariance-based), skipped here.
- Plumbing test - random-signal backtest that must not be profitable.
- Breakeven cost - cost level at which Sharpe crosses zero on the decay curve.
- Walk-forward DML - double machine learning with nuisance models refit per fold.
- Refutation (placebo) - block-permuting treatment timing to test the estimator; block >= treatment window.
- Tier - canonical (full, registered) vs preview (reduced, root `.preview/<case>`).
- read_only_study - registry opener that does not `Study.activate()`.
- Kill switch - portfolio-level permanent halt (`MaxDrawdownLimit`, `DailyLossLimit`); governance, never swept.
- Paired comparison / block bootstrap - day-by-day difference of two strategies, resampled in contiguous day blocks.
- DSR - deflated Sharpe ratio correcting for the number of trials.
