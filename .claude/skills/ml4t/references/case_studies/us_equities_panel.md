# Case study: US equities panel (~3,200 stocks, daily, fwd_ret_1d)

> The broadest cross-sectional workflow in the book: daily OHLCV for ~3,200 US stocks (NYSE/NASDAQ/AMEX, 1990 to 2018-Q1) traded as a daily long-short top-K ranker under the Fundamental Law of Active Management frame (small per-stock edge, breadth compensates). The pipeline is unusually long because the universe is unusually large (22 notebooks; 16 walk-forward folds of 10Y train / 1Y validation, the most folds of any case study; three label horizons) and exists to test whether a weak gross signal survives point-in-time eligibility, estimation-schedule leakage controls, multiple-testing accounting, turnover, material era-dependent costs and a sealed 2016-2018 holdout. Headline lesson: breadth buys *precision*, not *size*. Averaging IC over a wide panel and long history gives a tight interval around a small number; the selected configuration beat its equal-weight benchmark on validation only by a Sharpe difference whose bootstrap interval spans zero, and on a 3,000-name panel turnover, not ranking accuracy, is the binding constraint. Every stage reads `config/setup.yaml` as the single source of truth; results live only in `run_log/registry.db` and are reported by `22_strategy_analysis`, never in prose.

Repo root for every path below: `case_studies/us_equities_panel/`. Run each stage from repo root with `uv run python case_studies/us_equities_panel/NN_name.py` (Jupytext `.py` / `.ipynb` pairs side by side). Published artifact bundle: `uv run python scripts/download_artifacts.py --cs us_equities_panel` fetches `v3.1.0-artifacts` (current); `v3.0.0-artifacts` is the original generation and its content-addressed hashes do not resolve in 3.1.

## Setup contract

Source: `config/setup.yaml` (`strategy_id: us_equities_panel`, `setup_version: v1`) and the README. Every notebook loads it via `SETUP = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())` with `CASE_DIR = get_case_study_dir("us_equities_panel")`. Shared helpers: `resolve_label_horizon(CASE_STUDY_ID, PRIMARY_LABEL, SETUP)` (horizon sets the holdout cut-back and NW bandwidth), `load_us_equities()` returning the adjusted daily panel with `READ_COLS = ["symbol","timestamp","close","volume","adj_close","adj_volume"]`, `get_backtest_config()` for `execution.*`, `CandidateSet.one(name)` to reopen a frozen set. Changing any `execution.*` value or per-allocator override invalidates every existing `backtest_hash`.

| Item | Contract |
|---|---|
| Universe | ~3,200 US stocks (`universe.n_assets: 3199`), daily OHLCV from NASDAQ Data Link, `START_DATE="1990-01-01"`, `END_DATE="2018-03-31"` (= `holdout_end`). `START_DATE` is kept early on purpose: the screen needs a month of trailing volume before admitting any stock, and the 02 dispersion (Section E) and rank-correlation (Section G) diagnostics need a wide cross-section per session. Archive runs 1962-01-02 to 2018-03-27; only known absent NYSE session is `KNOWN_ABSENT_SESSIONS=[2017-11-08]`. |
| Eligibility (point-in-time, per decision date) | printed `close > 5.0` (`MIN_PRICE`) AND `adv_21d > 1_000_000` USD (`MIN_ADV_USD`, `ADV_WINDOW=21`) AND 21 consecutive sessions of volume (`adv_covered`). Polars form (02/03): `ELIGIBLE = pl.col("adv_covered") & (pl.col("close") > MIN_PRICE) & (pl.col("adv_21d") > MIN_ADV_USD)` with `ADV_COVERED = pl.col("session") - pl.col("session").shift(ADV_WINDOW - 1) == ADV_WINDOW - 1`. Screen on printed `close`/`volume`; compute returns from `adj_close`. Breadth floor `BREADTH_FLOOR = 2 * max(top_k_grid[primary]) = 100` eligible names per date; sessions below it are not evidence. |
| Decision / execution | `decision.cadence: daily_close`, `snapshot: close`, `execution_delay: next_bar_open` (sort at close, fill at next open). The label is close-to-close but the engine fills at the next open; that gap is unmeasured in 02 and only 19 (Ch18) measures costs on actual trades. |
| Labels | `labels.primary: fwd_ret_1d` (`buffer: 1D`); variants `fwd_ret_5d` (5D), `fwd_ret_21d` (21D). Close-to-close forward return counted in NYSE sessions. `rebalance_step` 1/5/21 thins the vectorized backtest so holds do not overlap. |
| CV | `evaluation.n_splits: 16`, `train_size: 10Y`, `val_size: 1Y`, `calendar: NYSE`, `periods_per_year: 252`. Validation window begins 2000-01-12. Folds are resolved from the label file (`labels/fwd_ret_*.parquet` and its `.digest.json` sidecar) so 04, 05 and the model stages agree on fold k; never hand-written. A `config/cv_config.json` sits beside `setup.yaml` but no notebook in this case study reads it (its keys are `test_size`/`test_start`, not the `val_size`/`holdout_start` vocabulary above); it is not the fold source. |
| Holdout | `holdout_start: '2016-01-01'`, `holdout_end: '2018-03-31'` (~2.25 years). Sealed on the label *endpoint*: the usable development boundary is `holdout_start` minus horizon, in sessions. Opened once, by notebooks 20-22 only. |
| Strategy mapping | `mapping.class: long_short_decile_rebalance`, `position_state_space: long_short`, `entry_logic: decile_sort_long_top_short_bottom`, `sizing: equal_weight_within_decile`. Dollar-neutral, equal weight within decile. |
| Execution engine | `execution.initial_cash: 1_000_000`, `share_type: integer`, `allocator_lookback: 63` bars (3 months; IV/RP/HRP fallback). Read via `get_backtest_config()`; never declare a local `INITIAL_CASH`. |
| Cost model (headline) | `costs.class: material`, `model: percentage`, `components: [spread, commission, market_impact, borrow_cost]`, `per_leg_cost_bps_range: [5, 20]` (loader uses midpoint 12.5 bps per leg; `COST_BPS` = 25 bps round trip, inference from code). Borrow ~50 bps/yr on the short leg. |
| Cost model (documentation-only) | `era_dependent`: pre-decimalization (before 2001-01-29; the `setup.yaml` note cites tick size 1/16 = $0.0625) [15, 30] bps/leg; post [5, 15]. Declared, not applied: validation starts 2000-01-12 so pre-decimal is ~6.5% of the window (feasibility B.4). |
| Cost model (exploratory) | `per_share: 0.0035` (IBKR Pro Tiered top tier) used only by the per-share regime of `19_costs`; `HALF_SPREAD_USD` = median of positive half-spread grid = $0.025 (inference). |
| Rebalance thresholds | `backtest.rebalance.default`: skip when `min_weight_change < 0.005` AND trade notional `< 100.0`. `benchmark` profile: 0.0 / 0.0 so full-universe 1/N rebalances at all. |
| Benchmark | Equal-weight long-short top-k/bottom-k book from `16_backtest` is the baseline every allocator and overlay is read against; full-universe equal weight uses the `benchmark` rebalance profile. |
| Sweep funnel | `backtest.sweep.top_n_predictions`: `signal: 0` (all predictions), `allocation: 10` (top-10 configs by equal-weight Sharpe), `cost_sensitivity: 1`, `risk_overlay: 1` (per label). Read with `get_top_n_predictions(case_study, stage)`. |
| Concentration | `top_k_grid`: `[20, 50]` for all three labels (wider than ETFs because ~3,000 names). |
| Allocators | `equal_weight` (run in 16, excluded from 17), `score_weighted` (capital proportional to prediction magnitude), `inverse_vol`, `risk_parity` (inverse-vol with a steeper exponent, approximating equal risk contribution without a covariance matrix), `mvo_ledoit_wolf` (`lookback: 126`), `hrp`, `conformal_weighted` (weight = 1 / interval width, width floored at the 1st percentile of that date's own cross-section); `expensive_allocators_skip: false`; no `max_weight` cap (intentional). |
| Cost grids | `cost_grid_bps: [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]`; `cost_grid_half_spread_usd: [0.0, 0.005, 0.01, 0.025, 0.05, 0.10]`. |
| Risk controls (14) | `stop_loss_{3,5,10,15}pct` (0.03/0.05/0.10/0.15), `trailing_{1,2,3,5,10,15,20}pct` (0.01..0.20), `time_exit_{10,20,40}` bars. |
| Model-based schedule | `model_based.regime`: `n_clusters: 2`, `window: 21`, `overlap: 5` (window step 16; burn-in 756 sessions = 46 windows), `burnin: 756`, `refit_every: 63`; the daily market-level distribution needs `XS_MIN_STOCKS=50` names; `SEED=42`. `model_based.garch`: `burnin: 504`, `refit_every: 63`, `mean: Constant`, `vol: GARCH`, `p: 1`, `o: 0`, `q: 1`, `dist: Normal`. 04 Papermill cell: `START_DATE`, `MAX_FOLDS=0`, `MAX_SYMBOLS=0` (via `top_entities`, longest-history symbols; does not reduce the panel), `XS_MIN_STOCKS=50`, `GARCH_MIN_OBS=0`, `REGIME_MIN_OBS=0` (0 = declared burn-in), `SEED=42`. |
| Modeling | `modeling.gbm`: `libraries: [lightgbm]`, `preset: default`, `device: cpu`, `max_bin: 255`. `modeling.latent_factors`: `persistent_entities: true`; nested under `modeling.latent_factors.model_kwargs`: `ipca` `max_iter: 10000`, `factor_ridge: 0.01`, `gamma_ridge: 0.01`; `sdf` `n_epochs_unc: 128`, `n_epochs_moment: 32`, `n_epochs_cond: 512`, `burn_in_epochs: 32`, `checkpoint_epochs: [128, 256, 384, 512]` (conditional-relative; published as global epochs 128..640), `beta_checkpoint_epochs: [256]`, `beta_default_checkpoint: 256`. |
| Causal | `causal.treatment: past_ret_12m_skip`, `treatment_window: 252`, `confounders: [vol_21d, illiq_rank, volume_ratio]`, `method: walk_forward_dml`. DML preset (`case_studies/config/dml/`): `n_folds: 5` nuisance folds, `n_placebo: 100`, `seed: 42`, `max_samples: 50000` (a second preset file carries 250000; which applies is not stated). |
| Kill conditions (declared in prose, not code-gated) | `ic_floor: 0.01` (cross-sectional IC < 0.01 across all features), `edge_to_cost_floor: 1.2` (net Sharpe / cost ratio < 1.2x), `micro_cap_concentration: 0.5` (>50% of alpha in bottom ADV quintile = untradeable), `net_sharpe_floor: 0.3` (net Sharpe after borrow < 0.3). |
| Training menus | `config/training/{fwd_ret_1d,fwd_ret_5d,fwd_ret_21d}.yaml`; each name resolves to a preset under `case_studies/config/{model_type}/`; comment a line out to skip a config. `fwd_ret_1d.yaml`: linear 16, gbm 15, deep_learning `nlinear`/`lstm_h64`/`tsmixer`, tabular_dl `tabm_s/m/l`, latent_factors `pca`/`ipca`, causal_dml `dml`. `fwd_ret_5d.yaml`: same plus deep_learning `nbeats_weekly`, no latent_factors block. `fwd_ret_21d.yaml`: 1d minus latent_factors. A family is fitted for a label only if that label's menu declares it; `LABELS=[]` in 06-10 means every label whose menu declares the family. |

## Pipeline stages

| Stage | Notebook | Chapter | What it does | What it writes |
|---|---|---|---|---|
| Feasibility | `01_feasibility_analysis` | Ch6 (6.2-6.6) | Point-in-time breadth per decision date (B.2), bps vs per-share cost against the price distribution (B.3), within-stock lag-1 autocorrelation (B.4), fraction of 1-day moves larger than round-trip cost (B.5), fold check, design decisions C.1-C.3. Fits no model. | Nothing |
| Labels | `02_labels` | Ch7 (7.2) | 1/5/21-session close-to-close forward returns on the NYSE session index; tradability screen applied *after* labels; reconciliation of unlabelled rows; effective-observation count under overlap; 12-1 momentum baseline (join `close_sessions_back(21, "_skip")` and `close_sessions_back(252, "_start")` on `["symbol","session"]`, then `_skip_close / _start_close.clip(lower_bound=1e-8) - 1`). | `labels/fwd_ret_{1d,5d,21d}.parquet` + `.digest.json` sidecars (record `MARKET_DATA_DIGEST`) |
| Features | `03_financial_features` | Ch8 (8.2) | 62 cross-sectional price-derived features in 8 families from `MOMENTUM_HORIZONS=[5,10,21,42,63,126,189,252]`, `VOLATILITY_HORIZONS=[21,63,126,252]`, `MA_HORIZONS=[10,20,50,100,200]`; clips at construction (vol ratios upper 10; Sharpe-like `ret_{h}d / vol_{v}d` with vol floor 0.01 clipped to [-10, 10]; 52-week ratios `adj_close / rolling_max(adj_high, 252)` in [0.1, 1.0] and `adj_close / rolling_min(adj_low, 252)` in [1, 10]; denominators `.clip(lower_bound=1e-8)`); eligibility screen; per-date winsorization at `WINSOR_LOWER, WINSOR_UPPER = 0.01, 0.99` (values from the 03 source, not the notes); IC scoring on the sealed development window. | `features/financial.parquet` |
| Temporal | `04_model_based_features` | Ch9 (9.1, 9.3, 9.5) | Walk-forward Wasserstein regime distance, FFD price, per-stock GARCH(1,1) conditional vol, all on an estimation schedule with `freeze_after`. | `features/model_based.parquet` + digest sidecar; joined to the 03 matrix on `(symbol, timestamp)` by `utils/modeling.py::load_modeling_dataset` for every model notebook |
| Evaluation | `05_evaluation` | Ch7 (7.3, 7.4) | Univariate feature-label rank IC across the panel, Newey-West CIs, FDR, per-fold stability, quintile returns, redundancy. | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` (ledger read by `20_strategy_synthesis/02_feature_evaluation`) |
| Linear | `06_linear` | Ch11 (11.2), Ch6 6.7 | OLS / Ridge / LASSO / ElasticNet penalty grid (16 configs per label: `ols`, 11 ridge alphas 0.001..1e7, lasso 0.01/0.1, enet 0.01/0.1); diagnostic set = `ols` at its only checkpoint; one prediction frame is 7M+ rows (~225 MB), so the menu is planned, not resolved, because one config costs minutes. The control everything more expensive is read against. | Training runs + validation prediction sets in `run_log/registry.db`; coefficients `run_log/training/{hash}/`, scores `run_log/predictions/{hash}/` |
| GBM | `07_gbm` | Ch12 (12.2, 12.3) | LightGBM grid `{default, leaves_7/15/31/63} x {mse, mae, huber}` = 15 configs, scored at 10 checkpoints; Huber threshold derived from each fold's own label spread. | Boosters, `learning_curves.parquet`, `fold_metrics.parquet` under `run_log/training/{hash}/` |
| Tabular DL | `08_tabular_dl` | Ch12 (12.3) | TabM weight-sharing ensemble: shared 2-layer backbone, each member owns a per-hidden-unit scaling vector and its own head, predictions averaged. Capacity ladder `tabm_s/m/l` = hidden 64/128/256 with 4/8/16 members (two dials move together, so not separable). 200 epochs, checkpoint every 25 -> 8 checkpoints x 3 = 24 scoreable models. | Checkpoints `run_log/training/tabular_dl/` |
| NLinear | `09_dl_nlinear` | Ch13 | Minimal sequence baseline: 60-session windows, last-value normalization, 100 epochs, checkpoint every 5. | Checkpoints `run_log/training/deep_learning/` |
| LSTM | `10_dl_lstm` | Ch13 | 2-layer LSTM, hidden 64 (`lstm_h64`), same windows and schedule. | Same layout |
| TSMixer | `11_dl_tsmixer` | Ch13 | Time-mixing + feature-mixing on the 60-session window (`n_blocks: 2`, `hidden_dim: 32`). 12 calls it "offered rather than run" (declined at 60-80 GPU hours); whether it ran canonically in the released study is ambiguous, so check the registry for `tsmixer` rows before assuming absence. | Same layout when run; presence in the published bundle unverified |
| Weekly DL | `12_dl_weekly` | Ch13 (13.9, Table 13.5) | `fwd_ret_5d` on a Friday-only grid: direct regression (`lstm_h64`, `nlinear`) vs one-step forecasting (Darts `NBEATSModel`). A reduced experiment: `N_EPOCHS=50` (the Friday grid holds a fifth of the rows), `MAX_FOLDS=4` of 16 (evenly spaced, always including the earliest and latest), `LOOKBACK=12` Fridays, `MAX_TRAIN_SEQUENCES` lower than the daily notebooks, `MAX_SYMBOLS=0`, `FORCE_RETRAIN` off unless identity changed invisibly; 10 checkpoints per fit. Weekly N-BEATS preset: `decision_cadence: weekly_friday`, `darts_input_chunk_length: 12`, `darts_output_chunk_length: 2` (notebook text says 1; unresolved), `darts_target: lagged_label`, `hidden_size: 128`, `n_blocks: 3`, `n_layers: 4`. Excluded from the daily backtest pool. | Training runs + prediction sets on a Friday grid |
| Latent factors index | `13_latent_factors` | Ch14 | Checks 13a/13b published one row per label per model with matching `cv_identity`. | Nothing |
| PCA | `13a_pca` | Ch14 | 5 factors from the return panel; factors and loadings fitted inside each fold's training window. Preset `case_studies/config/pca/pca.yaml` (`n_factors: 5`, `checkpoint_interval: 0`); 13b reads `case_studies/config/ipca/ipca.yaml`. Run values: `LABELS` (empty = primary label only), `OVERRIDES`, `EXECUTION_TIER`, `PREVIEW_FOLD_IDS`, `PREVIEW_MAX_SYMBOLS`. | One training run + one prediction set per label |
| IPCA | `13b_ipca` | Ch14 | 5 factors; loadings = shared mapping from stage-03 characteristics; refuses non-converged folds. | Same |
| Causal DML | `14_causal_dml` | Ch15 (15.6) | Does `past_ret_12m_skip` cause forward return after removing `vol_21d`, `illiq_rank`, `volume_ratio`? `walk_forward_dml` with embargo, HAC SE, block-permutation refutation. | One causal result row in `run_log/registry.db` (dedicated `causal_runs` table; table name from `case_studies/utils/causal.py`, not the notes) with `naive_effect`, `dml_effect`, `confounding_bias_pct`; never a prediction set |
| Model analysis | `15_model_analysis` | Ch11-16 | Completeness of every named set, IC with overlap-aware intervals, per-fold stability, pairwise redundancy, conformal calibration, causal section apart. Selects nothing. | Nothing |
| Backtest | `16_backtest` | Ch16 (16.4-16.8) | Every population member x every top-k into an equal-weight long-short book; gross of costs. Selection begins here. | One backtest per member and k, frozen as a named baseline set per label, under `run_log/backtest/{hash}/` (filenames `daily_returns/trades/fills/equity/portfolio_state/weights.parquet` + `spec.json` are from `case_studies/utils/registry/registration.py`, not the notes) |
| Portfolio | `17_portfolio_management` | Ch17 (17.2-17.8) | Names/dates/checkpoint fixed; sweeps six non-equal-weight allocators over the top-10 shortlist. | One backtest per config x allocator, frozen as a named allocation set per label |
| Risk | `18_risk_management` | Ch19 (19.3-19.6) | 14 overlays over the fixed rank-1 configuration from baseline/allocation stages. | One backtest per control, frozen as a risk-overlay set per label, plus the union population selection runs over |
| Costs | `19_costs` | Ch18 (18.2-18.5) | Cost sweep (bps headline, per-share exploratory) on one fixed configuration per label. Both tiers resolve the study through `open_study` (reads labels/features/results in place, redirects writes for preview); the source configuration is picked in-notebook with `get_top_n_predictions(CASE_STUDY_ID, "cost_sensitivity")` sorting by `(sharpe desc, backtest_hash asc)`, not via `resolve_solvent_carrier` (verified in `19_costs.py`; the notes say otherwise). | One backtest per cost level, registered outside the selection population |
| Holdout predictions | `20_holdout_predictions` | Ch20 (20.2) | Refit the selected configuration (via `resolve_solvent_carrier`) on history ending one label horizon before `holdout_start`. Refuses a second configuration. | One training run + one prediction set at `split='holdout'` |
| Holdout backtest | `21_holdout_backtest` | Ch20 | Trade holdout predictions with fixed k, allocator, overlay and cost. Finds the holdout prediction set by re-deriving the holdout training identity plus the selected checkpoint, never by search. `conformal_weighted` needs a calibration embargo on this panel (width calibrated from errors the model has already seen), set by a declaration with one horizon per label; the admission guard is written on the backtest hash (20's guard is on the model identity + checkpoint). Prints validation vs holdout pair without interpreting. | One backtest at `stage='holdout'` |
| Strategy analysis | `22_strategy_analysis` | Ch16-20 | Reopens the immutable validation set, applies the selection rule, verifies holdout lineage by full hash, paired-bootstrap comparisons. | `backtest_paired_metrics` table only (notebook header: "and nothing else"; it refits nothing and registers no backtest). The README table additionally lists `results/strategy_assessment.json` and a `template="full"` tearsheet at `20_strategy_synthesis/output/us_equities_panel/us_equities_panel_tearsheet.html` (gitignored, absent from the checkout); neither string appears in the 22 source, so treat the README entry as stale and the paths as not readable. |

Sequencing: 01 -> 02 -> 03 -> 04 -> 05 -> 06..13b -> 14 -> 15 -> 16 (select on validation Sharpe) -> 17 -> 18 -> 19 -> 20 -> 21 -> 22. Model notebooks (06-13b) write under named prediction sets declared before fitting and select nothing.

## Design decisions and why

| Decision | Choice | Why (evidence) |
|---|---|---|
| Feasibility before modeling | 01 fits no model | Ask whether breadth, cost scale, autocorrelation and fold fit can support a strategy at all before fitting anything. |
| Rebalance cadence (C.1) | Daily | 1-day moves are several times larger than the 25 bps round trip, so cost does not force a slower cadence. Lag-1 within-stock autocorrelation is near zero (no fast-fading tendency), so also label at 5d/21d. |
| Cost regime | bps headline, per-share exploratory | Universe spans two orders of magnitude in price; a fixed cents-per-share is wrong at the cheap end by the same factor, and per-share on split/dividend-adjusted prices is not the charge that would have been paid. |
| Flat bps across eras | `era_dependent` declared, not applied | Validation starts 2000-01-12; pre-decimal era is ~6.5% of the window. |
| Portfolio form (C.3) | Long top decile, short bottom, equal weight within decile | Uses the whole ordering; keeps ordering's contribution separable from any covariance estimate so Ch17 can compare allocators with names fixed. |
| Labels first, screen second | Labels on the complete series, then filter rows | The forward step must count market sessions, not surviving rows. |
| Session index | NYSE exchange calendar, not row counts | Halts, suspensions and late listings break the row = session identity; stray prints on non-market dates widen every rolling window. |
| Least expressive model first | 06 linear, 09 NLinear as controls | Everything more expensive is read against them. |
| Every checkpoint is a candidate | GBM 10, TabM 8, sequence 20 (daily) / 10 (weekly) | Prevents reporting "the maximum of N numbers as though it were one". |
| Selection metric | Validation backtest Sharpe in 16, never IC in model notebooks | IC measures ranking correctness, not money after costs and turnover; turnover is the binding constraint at 3,000 names. |
| Selection rule | Highest validation backtest Sharpe, ties by backtest hash; IC, cost sensitivity and holdout excluded | One sentence, applied to the immutable frozen set; ranks on a point estimate and does not consult the interval. |
| Top-k | `[20, 50]` both backtested | Top-k is a strategy axis (concentration, turnover, capacity), not a tuning axis. |
| Equal weight baseline | Run in 16 before any allocator | Adds the least estimation; hardest to beat by accident; covariance allocators have far fewer observations per parameter on a broad cross-section. |
| `mvo_ledoit_wolf` lookback | 126 (not 63) | At top_k=50 with 63 bars N/K is too small and shrinkage collapses to the identity target; 126 gives N/K >= 2.5. |
| No `max_weight` cap | Intentionally absent | Capping near top_k forces equal weight and defeats the allocator comparison. Rationale recorded in a `setup.yaml` comment; the note it cites is not shipped. |
| `initial_cash` | 1,000,000 with integer shares | The 2026-05-15 drop to 100k produced catastrophic degenerate output (integer rounding x high-priced names at top_k=50); restored 2026-05-16. Rationale recorded in a `setup.yaml` comment; the note it cites is not shipped. |
| Regime model | 2 Wasserstein clusters on the cross-sectional median return, 21-session windows overlapping 5 | Coarsest split separating calm from falling/turbulent months (momentum-crash literature); matches `etfs`, whose regime model reads a market-level series of the same shape. |
| GARCH spec | GARCH(1,1)-Normal, `o: 0` | Leverage term is least reliably identified on a broad cross-section of single stocks in a 2-year window; index case studies (`sp500_options`) fit `o: 1`. |
| GARCH refit cadence | Quarterly (63) | Fitted persistence on US equities moves slowly between refits; faster cadence multiplies ~183,000 fits without moving emitted vol (04 Section 4b). |
| FFD order | `FFD_D=0.4` (equity-class default), grid `[0,0.1,0.2,0.3,0.4,0.5,0.7,1.0]`, `FFD_THRESHOLD=1e-5` | Section 3 measures memory kept vs stationarity gained; no schedule needed because weights follow from d alone. |
| GBM `max_bin` | 255 on CPU | 63 was the GPU default carried over; every CPU fit was quantizing into a quarter of the bins. |
| TSMixer capacity | `hidden_dim: 32`, `n_blocks: 2` (smaller than LSTM state) | Unbroken 60-session windows are far fewer than rows; that count bounds the capacity worth fitting. Reading rule: compare TSMixer against the LSTM, not against NLinear; what is tested is repeated explicit nonlinear feature mixing alternated with time mixing (both other models combine features once, in one place). |
| Weekly reframing | Friday-only grid for `fwd_ret_5d`, 4 of 16 folds, 50 epochs | One-step vs multi-step is a property of the sampling grid; N-BEATS on a daily grid would compound its own output four times. Subsampling throws away four fifths of rows, so the experiment is reduced (`MAX_FOLDS=4`, `N_EPOCHS=50`); Table 13.5 is a 4-fold, 50-epoch experiment and must be read as such. What generalises is the reframing, not this panel's numbers. |
| Latent factors on primary label only | `pca`, `ipca` declared in `fwd_ret_1d.yaml` only | Most expensive fits; removing from a menu is what narrows 13a/13b. SDF schedule down-tuned from paper defaults (256/64/1024) to fit an overnight budget. |
| DML block length | `treatment_window: 252` | Derived from the treatment's construction (`close.shift(21)/close.shift(252) - 1`); a shorter placebo block destroys the serial dependence the refutation must preserve. |
| Cost sweep fixes its strategy first | Rank-1 across baseline/allocation/risk per label | A sweep allowed to reorder the field would choose a strategy on the cost assumption. |
| Holdout refit boundary | Training ends a full label horizon before `holdout_start` | Zero gap leaks the label window; identity must differ from the validation training hash or raise. |

## Market-specific guardrails

### Data, universe and labels

| Pitfall | Why it bites here | What to do |
|---|---|---|
| Whole-sample liquidity screen | Admits exactly the stocks that stayed liquid (survivorship); backtest measures an impossible choice | Apply `close > 5` and `adv_21d > 1M` on a trailing 21-session window per decision date (01 B.2); count eligible names against `BREADTH_FLOOR=100` |
| Adjusted price in the screen | An adjusted past price embeds later splits/dividends | Screen on printed `close`/`volume`; compute returns from `adj_close` |
| Unadjusted returns | A 2-for-1 split reads as a 50% fall | Returns only from the adjusted series |
| Row-count windows | Fixed rows back = fixed sessions only if the stock traded every session | Index by NYSE session number; screen the archive on the calendar first |
| Screen before label | Forward step counts surviving rows; horizon silently spans removed rows | Compute labels on the complete series, then filter |
| Unlabelled rows dropped silently | Window off the end of history, missed session, label crossing symbols | Reconciliation table of reasons that must sum to frame height (02) |
| Holdout leakage via label endpoint | A row observed before 2016-01-01 whose return resolves inside is a holdout row | Cut development diagnostics at `holdout_start` minus horizon in sessions (03/05 seal on the label endpoint) |
| Overlapping 5d/21d windows | Rows share forward windows; evidence overstated | Effective observation count and overlap-aware SE; the 1d label must return the row count unchanged as the arithmetic check |
| Stacked-panel autocorrelation | Measures joins between stocks | Compute lag-1 autocorrelation per stock, then average (01 B.4) |
| Suspension gaps in per-symbol transforms | shift, FFD convolution and GARCH recursion read a months-long gap as consecutive sessions | 02 Section D measures frequency; 04 does not segment (known limitation); sequence models window only over unbroken runs |
| Mid-window untradability | 21-session label checked at its two ends only | Known limitation: a stock untradable mid-window still carries the label. Do not claim mid-window tradability for the 5d/21d books; disclose it in the write-up; when adapting, screen every session of the window if the engine supports it |
| Thresholds not inflation-adjusted | $5 / $1M stricter in 1990 than 2018 | Known limitation: report that early-sample breadth is understated by the fixed screen; when adapting, consider deflating thresholds or screening on a trailing cross-sectional quantile instead of a fixed dollar level |
| Close-to-close label vs next-open fill | The label measures close-to-close; the engine fills at `next_bar_open`; the gap is unmeasured in 02 | Keep `execution_delay: next_bar_open` declared in `setup.yaml`; only 19 (Ch18) measures costs on actual trades; never read label returns as receivable |
| Digest mismatch between stages | 02 and 04 panels must be the same data | Assert `MARKET_DATA_DIGEST` from the label sidecar in 04 Section 1 (`value_digest`, `read_digest(path)["inputs"]`) |

### Features and univariate evaluation

| Pitfall | Why it bites here | What to do |
|---|---|---|
| Winsorizing against the full sample | Bounds estimated on future dates; panel dispersion more than doubles between quietest and loudest year | `winsorize_features` clips at per-date cross-sectional quantiles `WINSOR_LOWER/UPPER`, measured before applying |
| Order of operations | Ranks computed before the screen mix ineligible names | Per-symbol shifts/rolling on the complete series -> eligibility screen -> cross-sectional ranks |
| Newey-West on unsorted IC series | `partition_by` returns groups in no order | Sort the IC series by date before NW; bandwidth = label horizon |
| Confusing FDR with NW | One corrects for feature count, the other for one feature's persistence | Apply both (`FDR_ALPHA=0.05`), report separately; on a 1-session label NW costs almost nothing while FDR still bites |
| Candidate count != test count | A feature and its within-session rank give identical IC; FDR counts two | Read corrected p-values against the redundancy section (`REDUNDANT_CORR=0.7`) |
| Pooled quintile bins | Another period's distribution sets this period's boundaries; raw levels carry drift | Assign quintiles within decision time, net out the session mean |
| Fold stability by positive-fold count | A steady inverse predictor scores zero | Count folds agreeing with the feature's own sign |
| Rank correlation over market-level constants | Undefined; helper returns a tie-breaking artifact | Measure distinct-value count per date; set constant columns aside (04 Section 7) |
| Redundancy on levels | Finds duplicated construction, not duplicated information; bar moves with the panel | Treat the triage ledger as a screen; let 06+ choose |
| Single-horizon screen | A longer-hold association is invisible on the 1d label | 05 scores the primary label only; compare 1d/5d/21d orderings in 06/07 |
| Fundamental Law breadth arithmetic | `REBALANCES_PER_YEAR=252` as breadth multiplier assumes independent bets | Read as an upper bound |

### Model-based features (estimation schedules)

| Pitfall | Why it bites here | What to do |
|---|---|---|
| Fitted-parameter leakage | Parameters fitted once on the whole sample carry holdout into every row ("a schedule bounds an estimate; a fold selects rows") | Burn-in (504 GARCH / 756 regime), fit on strict prefix, emit block, refit every 63; `freeze_after` before holdout (prices may be emitted into holdout, labels never) |
| `arch_model(...).fix(params)` non-causal | Residuals, backcast seed and variance bounds derive from whatever array is passed | Produce seed and bounds where only earlier returns are in scope and pass them with coefficients |
| Capped `MAX_SYMBOLS` in 04 vs 05 | Different frames select different names; unfitted stocks silently carry market-level vol via coalesce | Leave both caps at 0 or verify identical name sets |
| Regime state labels across refits | State numbers are not comparable between fits | Prefer `wass_dist_ratio` over the state label |
| Regime blind to tail widening | Clustering on the cross-sectional median reads a centre shift only | Known limitation; pair with dispersion/volatility features (inference) |
| Fold grid vs regimes | 16 one-year folds show a regime-dependent feature as scattered folds, not two states | Known limitation; the regime feature exists for interaction/timing use |

### Model notebooks and the registry

| Pitfall | Why it bites here | What to do |
|---|---|---|
| IC without coverage | Aggressive L1 zeros all features on some folds, predicts a constant, scores no dates; a TabM run can complete and score zero dates when stocks do not overlap in time | Read `ic_n_days` / `full_coverage`; assert scored dates > 0 |
| Best checkpoint post hoc | Reports max of 10/8/20 numbers as one; bias does not cancel across architectures | Register every checkpoint as its own candidate; select once in 16 |
| Learning-rate confound in GBM grid | `default` profile runs at 2x the ladder's learning rate | Compare `leaves_*` rows among themselves; read `trees_effect` before ranking |
| Carrying a penalty across horizons | Grid measures the horizon, not the estimator | Compare orderings across 1d/5d/21d before transferring |
| Preview compared to canonical | A reduced run is a different measurement | `EXECUTION_TIER=preview` must declare >= 1 reduction carried in identity; cannot reach a holdout decision |
| Preview shortening epochs | Fewer epochs is a different model, not an early read | `SEQUENCE_PREVIEW_FIELDS` admits universe, fold subset, sequence cap only; `n_epochs` goes through `COMMON_OVERRIDES`, which moves identity and publishes nothing canonical |
| Narrowed run publishing canonical names | A name must not mean two member sets; 15 raises, 16 would silently backtest a narrower chain | `narrows_declared_catalog` guard; freeze candidate sets per `(label, family)` |
| Comparing whatever finished | A half-failed family looks cheaper | Declare named sets before fitting; check completeness before scoring |
| Identity not covering feature set | A run served as "already complete" could be a different experiment | Identity must cover feature lineage, parameters and CV interval |
| Window across a gap | A listing/halt/delisting jump reads as one day's move | Build windows only over unbroken 60-session runs; report example counts per fold (far fewer than rows) |
| Weekly predictions in daily pool | Friday-only series scored on a fifth of dates compares dates, not models | 12's sets are excluded from the pool 16 ranks |
| Hand-written walk-forward windows | Free to leak | Resolve folds from the label file |
| Factors on the whole sample | Computed partly from returns later predicted | Fit factors, loadings and the IPCA mapping inside each fold's training window |
| `n_factors: 5` read as validated | Nothing tunes it | Results show what 5 gives, not that it is right; do not name factors beyond "first looks like the market" |
| PCA vs IPCA across labels/CV | Measures label or window, not model | Both rows at every label with identical `cv_identity` (13) |
| IPCA convergence read as goodness | Stable estimate is not a correct mapping | Runner refuses non-converged folds; convergence is a precondition only; check stage-03 feature freshness |
| DML residualising one side | Nuisance error leaks into the effect at first order | Residualise both outcome and treatment; fit nuisance walk-forward with embargo |
| Row-wise permutation refutation | Breaks the serial dependence it should preserve | Permute in within-stock blocks of `treatment_window: 252`; `n_placebo: 100` |
| Uncorrected causal interval | Overlapping forward returns are serially dependent | HAC SE; register only with finite effect and finite HAC SE |
| Small p-value read as an assumption check | Permutation p-value and HAC interval check the estimator, not the three untestable assumptions: no unmeasured confounder, overlap, no interference (the least comfortable in a single market, where flows into momentum names are a channel between stocks) | State the three assumptions beside the estimate; a missing confounder is missing from the permutation too; the estimate is one specification on the development sample |
| Causal estimate beside predictive scores | Produces no ranking, cannot be backtested | Registered separately; 15 reads it in its own section |

### Backtest, allocation, risk, costs, holdout

| Pitfall | Why it bites here | What to do |
|---|---|---|
| Selecting on IC before backtesting | Turnover invisible | Backtest every member of every population x every k |
| Measuring models under a clever allocator | Confounds sizing with ranking | Equal-weight baseline first (16), allocators second (17) |
| Shortlist on equal-weight Sharpe before allocators | An allocator that rescues a buried model cannot be found | Acknowledged design bound; do not claim allocator comparisons are exhaustive |
| Shared allocator lookback | Starves allocators needing more history | 63 default, 126 for `mvo_ledoit_wolf`; one window per allocator declared, not swept |
| Conformal width from a later date | Lookahead in sizing | Floor width at the 1st percentile of the same date's cross-section; coverage is a diagnostic, not a promise (residuals cluster in vol, share a market factor); a width systematically too narrow oversizes `conformal_weighted` positions |
| Two correlated models as two signals | Portfolio holds one signal at twice the weight | Pairwise prediction correlation on shared dates with panel keys intact; watch joins that multiply rows |
| Overlay matching the unprotected book everywhere | Controls may never have reached the engine | Check `risk_triggers > 0`; declared-but-uninstalled stops the notebook; installed-and-never-triggered is a valid result |
| Overlay helping at one threshold only | A feature of the sample | Sweep tightest to loosest with the unprotected line beside it; the loosest setting is still a control. Read the time exit first: an overlay only ever removes (truncates both tails), and the time exit watches no price, so where it moves results as much as a stop does, the stops were shortening the holding period rather than responding to losses |
| Intra-bar breach | Controls act on bar closes only | Acknowledged limitation: state in the write-up that stops may fire later than live; do not quote stop thresholds as realised loss limits |
| Overlay adding trades | More turnover charged later | Costs applied in 19 after overlays |
| Gross numbers read as receivable | Daily long-short across 3,000 names has large friction | Nothing before 19 is net |
| No capacity constraint | Positions sized regardless of volume | Acknowledged limitation: report alpha share by ADV quintile against the `micro_cap_concentration: 0.5` kill condition; do not claim capacity; when adapting, size against ADV |
| Flat borrow fee | Borrow rises on the names most people short | Known limitation; ~50 bps/yr; `net_sharpe_floor: 0.3` after borrow |
| Listed re-pricing pool | A missing `risk_overlay` stage still resolves to the allocation parent and prices a different strategy | Derive the pool from `STAGE_SEQUENCE` minus the terminal stage |
| Break-even read as tradability | Real friction varies by name, size, time of day | A high break-even clears a floor only |
| Holdout scored with validation-fitted model | Circular | Refit on history ending one horizon before `holdout_start`; raise if identity equals the validation hash |
| Second configuration on the holdout | Spends the window again | 20/21 refuse a different configuration; reruns of the same one are served from the registry |
| Passing a hash between notebooks | Agrees only until the sweep is rebuilt | Resolve via `resolve_solvent_carrier` in 20/21/22 (19 selects its own sources under `open_study` with `get_top_n_predictions`; the notes list 19 as a resolver user but its source does not import it); derive holdout identity deterministically, never search |
| Conformal width calibrated on seen errors in the holdout | `conformal_weighted` sizes by a width calibrated from residuals the model already saw | 21 applies a calibration embargo declared per label (one horizon each); the guard admitting a run to the holdout is written on the backtest hash |
| Nothing after the holdout | Archive ends 2018-03-27; the holdout is the latest history | Known limitation: say so in the write-up; there is no post-holdout check, and a longer archive means a new holdout declaration, not an extension of this one |
| Resolver matching on name only | A refit under changed features keeps its name | 22 rebuilds the expected holdout spec and compares the full hash |
| Validation-to-holdout gap as decay | Holdout ~2.25 years; validation Sharpe is a max over a large sweep; validation spans two regimes, holdout one; model shape differs (all prior years vs rolling 10Y) | Paired bootstrap intervals; never pool observations across the boundary |
| Drawdown read uniformly | For the selected run it is decline from peak with margin never near short maintenance; across the sweep it usually means capital lost | Interpret per run |
| Registry row order changing selection | Non-reproducible choice | Select from the immutable frozen set reopened through the registry |
| Integer shares with small capital | Degenerate output at top_k=50 | Keep `initial_cash: 1_000_000` |

## Results and lessons

Numbers are deliberately absent from the README and the extracted notes: the registry is rebuilt on every re-derivation, so `22_strategy_analysis` reading `run_log/registry.db` (or the `v3.1.0-artifacts` bundle) is the only source for IC levels, validation/holdout Sharpe, bootstrap CIs, DML effect, break-even cost and Table 13.5 values. Qualitative findings that are recoverable:

| Stage | Finding |
|---|---|
| 01 Feasibility | 1-day moves are several times the 25 bps round trip (daily cadence affordable). Within-stock lag-1 autocorrelation near zero. Per-share cost unsupportable across a price range spanning two orders of magnitude. |
| 02 Labels | Only known absent NYSE session in the archive is 2017-11-08. |
| 03 Features | 62 features; families split on direction, the negative side is larger; composite, liquidity and Sharpe-like families extend to the positive side. Panel dispersion more than doubles between regimes. Families highly correlated by construction (block structure). |
| 04 Temporal | 3,160 GARCH series fitted, ~58 refits average, 110 max, ~183,000 fits; persistence moves slowly, so quarterly cadence suffices. Most emitted model-based columns are market-level constants. |
| 05 Evaluation | Breadth gives tight intervals around small IC: significance is not magnitude. |
| 06 Linear | Ridge IC curve is flat -> rises -> falls with penalty; height above the left end = signal buried by collinearity. Aggressive L1 posts high raw IC but fails coverage (`ic_n_days`). |
| 07 GBM | Whether best checkpoints are interior (early stopping binds) is read from `trees_effect`; MAE/Huber leading is a statement about label tails, not fit quality. |
| 08 TabM | A complete run can score zero dates; the assertion refuses it. |
| 11 TSMixer | Declined at 60-80 GPU hours; the notebook carries no outputs and 12 calls it "offered rather than run". Whether it ran canonically in the released study is ambiguous; verify in the registry rather than assume. |
| 12 Weekly | Table 13.5 is a 4-fold (of 16), 50-epoch experiment. Each table row is a max over ten checkpoints; the weekly-vs-daily comparison is looser twice over (different decision dates; max vs mean). |
| 14 DML | Momentum predicts returns throughout the panel; `dml_effect`, `naive_effect`, `confounding_bias_pct` computed, values in registry only. |
| 17 Allocation | Equal weight is hard to beat on this cross-section; covariance allocators' estimation error does not pay for itself (qualitative). |
| 20 Holdout | The validation leader is not always the risk-overlay run; the winning stage is printed, not assumed. |
| 22 Assessment | The selected configuration was never distinguishable from its equal-weight benchmark on validation: Sharpe difference positive with a bootstrap interval spanning zero. Validation Sharpe spans two regimes in which the strategy behaves oppositely; the holdout lies wholly within the second. Registered drawdown for the selected run is decline from peak, not capital lost. |

Lessons an agent should carry:
- Breadth tightens the interval around IC; it does not raise IC. Report magnitude with the interval, and apply the kill conditions (`ic_floor 0.01`, `edge_to_cost_floor 1.2`, `micro_cap_concentration 0.5`, `net_sharpe_floor 0.3`).
- Turnover is the binding constraint on a daily 3,000-name long-short; select on validation backtest Sharpe, never on IC.
- Equal weight is the benchmark to beat; a sizing method that does not beat it did not pay for its estimation error.
- The holdout removes one circularity but is not a fresh test: the configuration was chosen on validation and the window is spent once.

## How to adapt this pattern to a new dataset

1. Write `config/setup.yaml` first and read every constant from it (`decision.*`, `universe.n_assets`, `execution.*`, `mapping.*`, `costs.*`, `backtest.*`, `evaluation.*`, `labels.*`, `model_based.*`, `modeling.*`, `causal.*`, `kill_conditions.*`). Never duplicate a constant in a notebook.
2. Run a feasibility notebook that fits no model: point-in-time breadth per decision date vs `2 * max(top_k)`, cost regime (bps vs per-share) against the price distribution, within-entity lag-1 autocorrelation (per entity, then average), fraction of moves larger than round-trip cost, fold check. Declare kill conditions here.
3. Build a session index from the exchange calendar; screen the archive on the calendar; record known absent sessions.
4. Compute labels on the complete adjusted series by session count, *then* apply the eligibility screen on printed prices. Reconcile unlabelled rows to the frame height. Verify the 1-step label's effective observation count equals the row count.
5. Seal the holdout on the label endpoint: development diagnostics stop at `holdout_start` minus horizon in sessions; `freeze_after` for every fitted feature.
6. Engineer features per symbol on the complete series -> screen -> cross-sectional ranks; winsorize at per-date quantiles measured before applying; clip ratio denominators at 1e-8 and bound ratio families.
7. Put every fitted feature (regime, GARCH, anything with parameters) on an estimation schedule: burn-in, fit on strict prefix, emit block, refit every N; hand recursions only earlier data (`.fix(params)` with causal seed/bounds). Assert the feature panel digest equals the label digest.
8. Evaluate univariately with per-date rank IC, time-sorted NW SE (bandwidth = horizon), FDR at 0.05, per-fold sign-agreement, within-date quintiles, redundancy at 0.7. Treat the ledger as a screen.
9. Declare training menus per label (`config/training/{label}.yaml` -> presets); declare named prediction sets before fitting; register every checkpoint as a candidate; set `EXECUTION_TIER` and run a `CONFIG_NAMES` subset first to prove plumbing (`COMMON_OVERRIDES` applies to all selected configs, `CONFIG_OVERRIDES` to one config with precedence, `DIAGNOSTIC_CONFIG_NAMES` names the diagnostic set; a preview is `CONFIG_NAMES` subset + `COMMON_OVERRIDES={"n_epochs": ...}` + `EXECUTION_TIER="preview"` with at least one declared `SEQUENCE_PREVIEW_FIELDS` reduction).
10. Fit the least expressive model first (linear, NLinear) as the control; read coverage (`ic_n_days`, `full_coverage`) before any ranking; compare 1d/5d/21d orderings before transferring a penalty.
11. Keep causal estimates out of prediction sets; use walk-forward DML with embargo, HAC SE, block permutation with block length derived from the treatment's construction window.
12. Check completeness of every named set, then IC with overlap-aware intervals, per-fold stability, pairwise redundancy on shared dates, conformal calibration as a diagnostic.
13. Backtest every member x every top-k with equal weight, gross of costs; freeze the baseline set; select on validation Sharpe with ties by hash.
14. Sweep allocators on the top-N shortlist with names fixed; declare lookback per allocator (ensure N/K >= 2.5 for Ledoit-Wolf); no `max_weight` cap; use `initial_cash` large enough for integer shares.
15. Sweep risk overlays tightest to loosest on the fixed rank-1 configuration; verify `risk_triggers > 0`; read the time exit first: where it moves results as much as a stop does, the stops were shortening the holding period, not responding to losses.
16. Fix one configuration per label, then sweep costs (bps headline, per-share exploratory) outside the selection population; derive the pool from `STAGE_SEQUENCE` minus the terminal stage; report break-even as a floor.
17. Refit the selected configuration on history ending one horizon before the holdout; refuse a second configuration; resolve via a shared resolver, not a passed hash; if the allocator sizes by a conformal width, declare a calibration embargo per label before the holdout backtest; verify lineage by full hash; compare with paired bootstrap; never pool across the boundary.
18. Report results only from the registry; the README describes construction, never numbers.

## Related references

- `chapters/06_strategy_definition.md` — feasibility (6.2-6.6), search accounting and run logging (6.7).
- `chapters/07_defining_the_learning_task.md` — forward-return labels on a session calendar, univariate IC, NW/FDR search accounting.
- `chapters/08_financial_features.md` — the 62-feature cross-sectional panel and per-date winsorization.
- `chapters/09_model_based_features.md` — FFD, GARCH and Wasserstein regime features on estimation schedules.
- `chapters/11_ml_pipeline.md` — regularized linear grid as the control model.
- `chapters/12_gradient_boosting.md` — LightGBM grid, checkpoints, TabM.
- `chapters/13_dl_time_series.md` — NLinear, LSTM, TSMixer, weekly N-BEATS reframing (13.9, Table 13.5).
- `chapters/14_latent_factors.md` — PCA vs IPCA, SDF schedule.
- `chapters/15_causal_estimation.md` — walk-forward DML, HAC, block permutation (15.6).
- `chapters/16_strategy_simulation.md` — equal-weight long-short backtest and selection on validation Sharpe.
- `chapters/17_portfolio_construction.md` — allocator sweep, Ledoit-Wolf and HRP lookbacks.
- `chapters/18_transaction_costs.md` — bps vs per-share regimes, break-even cost.
- `chapters/19_risk_management.md` — stop/trailing/time-exit overlays and `risk_triggers`.
- `chapters/20_strategy_synthesis.md` — holdout refit, lineage verification, paired bootstrap.
- `case_studies/etfs.md` — regime model reads a market-level series of the same shape; ~100 names vs ~3,000 here.
- `case_studies/sp500_options.md`, `case_studies/sp500_equity_option_analytics.md` — fit GJR (`o: 1`) on index series where the leverage term is identifiable.
- `libraries/ml4t_engineer.md` — `ml4t.engineer.features.fdiff.ffdiff`, `get_ffd_weights`.
- `libraries/ml4t_backtest.md` — engine settings behind `execution.*` and the backtest artifact layout (inference).
- `libraries/ml4t_diagnostic.md` — IC, NW and FDR evaluation helpers (inference).
- `workflow.md` — the Stage 0-13 map (data through monitoring) that this case study's 22 notebooks instantiate; its section 4 places the nine case studies on that map.
- `guardrails.md`, `decision_rules.md` — cross-case versions of the rules above.
- `evidence.md` — where this case study's findings sit beside the other eight.
- `companion_repo.md` — registry layout, artifact bundles, `uv run` conventions.
- `glossary.md` — shared definitions.
- Further reading:
  - Fundamental Law of Active Management (breadth x skill framing).
  - Momentum-crash literature (two-state regime split).
  - Newey-West HAC standard errors; Benjamini-Hochberg FDR (implied by `FDR_ALPHA`).
  - Ledoit-Wolf covariance shrinkage; hierarchical risk parity (HRP); instrumented PCA (IPCA).
  - TSMixer, N-BEATS (Darts `NBEATSModel`), TabM; double machine learning; split conformal prediction.
  - IBKR Pro Tiered commission schedule (source of `per_share: 0.0035`).

## Glossary

| Term | Meaning here |
|---|---|
| Point-in-time universe | Stocks eligible on a date using only information before that date (`close > 5`, `adv_21d > 1M`, 21 covered sessions). |
| ADV | Trailing 21-session mean of printed volume x price (USD). |
| Breadth floor | >= 100 eligible names per decision date (2 x top_k 50). |
| Session index | NYSE exchange-calendar position; windows and labels count sessions, not rows. |
| Buffer | Gap between train and validation equal to the label horizon (1D/5D/21D). |
| Holdout seal | Development boundary = `holdout_start` minus horizon in sessions; a label's endpoint, not its observation date. |
| Effective observation count | Row count deflated for overlapping forward windows; unchanged for the 1d label. |
| IC | Per-date cross-sectional rank correlation between feature or prediction and forward return, averaged over dates. |
| Newey-West SE | Autocorrelation-robust SE of the time-sorted IC series; bandwidth = horizon. |
| FDR | False discovery rate over the number of features tested (`FDR_ALPHA=0.05`). |
| Winsorization | Clipping at per-date cross-sectional quantiles, bounds measured before applying. |
| Estimation schedule | Burn-in, fit on strict prefix, emit block, refit every N sessions; `freeze_after` stops re-estimation before holdout. |
| `wass_dist_ratio` | Wasserstein distance from the recent 21-session return-distribution window to the nearer of 2 centroids divided by the distance to the farther one (definition from the 04 source, not the notes): near 0 when squarely inside one state, near 1 when equidistant, so it carries how certain the match is, not which state; comparable across refits, unlike the state label. |
| FFD | Fractional differencing of price at `d=0.4` keeping memory while gaining stationarity. |
| `garch_cond_vol` | Per-stock GARCH(1,1)-Normal conditional volatility emitted on the schedule; market-level fit for stocks below burn-in. |
| Candidate set / named prediction set | Frozen membership per `(label, family)` declared before fitting that a run must fill completely. |
| Population | Immutable list of every member a run fitted across labels. |
| Checkpoint | Registered model state at a tree count or epoch; each is its own candidate. |
| Coverage (`ic_n_days`, `full_coverage`) | Number of dates a configuration actually scored. |
| `EXECUTION_TIER` | `canonical` (full panel, published schedule) vs `preview` (declared reduction in identity; cannot reach holdout). |
| Training identity | Hash over feature lineage, model parameters and CV interval that decides registry reuse. |
| `cv_identity` | Row key that must match across compared latent-factor rows. |
| Window | 60 consecutive sessions (daily) or 12 Fridays (weekly) of one stock's features used as one example. |
| TabM / NLinear / TSMixer | Weight-sharing ensemble MLP / per-feature linear map after last-value normalization / alternating time-mixing and feature-mixing MLPs. |
| IPCA | Instrumented PCA: loadings as one shared function of characteristics. |
| DML | Residualise outcome and treatment on confounders, regress residual on residual; `confounding_bias_pct = (naive_effect - dml_effect) / abs(dml_effect) * 100`. |
| `STAGE_SEQUENCE` | Ordered backtest stages: signal, allocation, risk overlay, cost sensitivity. |
| `resolve_solvent_carrier` | Shared resolver for the selected configuration imported by 20/21/22 (19 picks its sources with `get_top_n_predictions` under `open_study`). |
| `open_study` | Study handle used by 19's canonical and preview tiers: reads labels, features and results in place, redirects writes for preview. |
| `risk_triggers` | Count of overlay firings; proof a control acted. |
| Break-even cost | Uniform friction (bps) at which validation Sharpe crosses zero. |
| Selection rule | Highest validation backtest Sharpe, ties by backtest hash; IC, costs and holdout excluded. |
| `backtest_paired_metrics` | Table of paired-bootstrap comparisons between two existing return series written by 22. |
| Kill condition | Declared no-go threshold: IC 0.01, edge/cost 1.2, micro-cap 50%, net Sharpe 0.3. |
| Spec hash / `backtest_hash` | Content address of a run's specification; invalidated by any `execution.*` change. |
