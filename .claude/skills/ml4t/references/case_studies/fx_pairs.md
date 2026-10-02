# Case study: FX pairs (20 G10 pairs, daily, fwd_ret_1d)

> Daily OANDA spot bars (consolidated from 4H bars on the CME_FX 5PM rollover) for 20 G10 pairs, 2011-01-03 to 2025-12-31 (3,874 sessions), used to test whether momentum (and, nominally, carry) signals produce tradeable long-short alpha at a one-day horizon. FX is "structurally challenging": spreads are tight (1-8 bps) so execution is cheap, but the cross-section is tiny and dependent (20 pairs quoted among 8 currencies; participation ratio 5.27 effective bets), so rank IC counts one dollar bet twenty times and predictions must be judged through a portfolio backtest, not IC. The pipeline is a study in hypothesis revision (short-horizon momentum -> multi-horizon -> full strategy assessment) and in search accounting: 28 linear + 15 GBM x 10 checkpoints + 3 TabM + 3 sequence-DL configs per label, selected exactly once on validation backtest Sharpe, perturbed by allocator, risk overlay and cost as one-field siblings of a sealed parent, then spent once on a 2024-2025 holdout and judged on interval evidence. Headline lesson: at the traded horizon the signal is "small and positive" for heavily penalized linear models and absent for GBM ("does not rank this cross-section in either direction"); the durable content is the discipline (sealed cohorts, one-field siblings, selection re-resolved by registry query, no holdout lock) that stops a weak signal from being flattered by a large grid.

## Setup contract

Source: `case_studies/fx_pairs/config/setup.yaml` (`strategy_id: fx_pairs`, `setup_version: v1`) and `case_studies/fx_pairs/README.md`.

| Item | Contract |
|---|---|
| Universe (`universe.symbols`, `n_assets: 20`) | AUD_JPY, AUD_NZD, AUD_USD, CAD_JPY, CHF_JPY, EUR_AUD, EUR_CAD, EUR_CHF, EUR_GBP, EUR_JPY, EUR_USD, GBP_AUD, GBP_CHF, GBP_JPY, GBP_USD, NZD_JPY, NZD_USD, USD_CAD, USD_CHF, USD_JPY. 7 dollar pairs (AUD_USD, EUR_USD, GBP_USD, NZD_USD, USD_CAD, USD_CHF, USD_JPY), 13 crosses; 8 currencies; USD and JPY each in 7 of 20 pairs. Fixed list chosen knowing which pairs are liquid today (survivorship disclosed, not fixed). |
| Data | OANDA 4H spot bars consolidated to daily OHLCV; data root `ML4T_DATA_PATH/fx/` per `data/fx/README.md`; loader `load_fx_pairs()`. Panel 2011-01-03 to 2025-12-31, 3,874 sessions, every pair quotes on all of them (README says history 2005-2025; the 2005 start probably refers to raw daily files, not the 4H panel the notebooks run - uncertain). Volume is tick count at one venue: it enters only through the daily range, never as participation. The spot feed carries no rate differential, so there is NO carry feature despite the README row and `entry_logic` naming carry. |
| Decision cadence (`decision.*`) | `cadence: daily_ny_close`, `snapshot: ny_5pm_close`, `execution_delay: next_bar_open`, `session_calendar: CME_FX` (implements the 5PM rollover; assigns each 4H bar to its session; used by 02 and 04 for aggregation). Distinct from `evaluation.calendar: FX`, the calendar the CV splitter counts train/val windows on. |
| Labels (`labels.*`) | `primary: fwd_ret_1d`, `buffer: 1D`; `variants: [fwd_ret_5d, fwd_ret_21d]`; `variant_buffers: fwd_ret_5d: 5D, fwd_ret_21d: 21D` (buffer = gap between training and validation and the gap held open in front of the holdout); `rebalance_step: fwd_ret_1d 1, fwd_ret_5d 5, fwd_ret_21d 21` (vectorized-backtest thinning so holding periods do not overlap). Close-to-close forward return over 1/5/21 sessions within one pair. |
| CV (`evaluation.*`) | `n_splits: 8`, `train_size: P5Y`, `val_size: P1Y`, `calendar: FX`, `periods_per_year: 252`. Each fold = 5-year training window + the 1-year stretch it did not see; only those stretches are scored. Earliest validation window opens 2016-01-05, last ends 2023-12-28; 3,355 development decision dates. Oldest fold: 1,289 training sessions on the price panel (1,269 on the dollar-factor series). |
| Holdout | `holdout_start: '2024-01-01'`, `holdout_end: '2025-12-31'`. Spent once, against the whole frozen candidate set. Refit training window ends 21 observations (the widest configured buffer, `fwd_ret_21d: 21D`) before the first holdout session, counted on the FX observation grid, not in calendar days. |
| Mapping (`mapping.*`) | `class: long_short_rank_rebalance`, `position_state_space: long_short`, `entry_logic: rank_by_momentum_or_carry`, `sizing: equal_weight`. `backtest.sweep.top_k_grid`: `[5, 10]` for all three labels (k long + k short; `allow_short_selling=true` at backtest config). Breadth floor = 20 quoting pairs = 2 x max k; satisfied on all 3,355 decision dates. |
| Execution (`execution.*`, read via `get_backtest_config()`) | `initial_cash: 100_000` (IBKR retail cohort default), `share_type: integer` (units of base currency), `allocator_lookback: 63` (3 months of daily bars, applied uniformly to inverse_vol, risk_parity, hrp, mvo_ledoit_wolf). Never declare a local `INITIAL_CASH`; cash and share_type are spec-hash inputs, so changing them invalidates every `backtest_hash`. |
| Cost model (`costs.*`) | `class: material`, `components: [spread, swap_points]`, `spread_bps.major_pairs: [1, 3]`, `spread_bps.cross_pairs: [3, 8]`. Backtest charges one aggregate proportional (bps) rate per traded leg; `components` is a Ch18 taxonomy, swap points are declared but never priced. 01 charges the wide end of each group's range (3 bps dollar pairs, 8 bps crosses) per crossing; whether "round trip" doubles it is unverified (inference). |
| Rebalance (`backtest.rebalance`) | `default: min_weight_change 0.005, min_trade_value 100.0` (skip only when BOTH are below); `benchmark: 0.0 / 0.0` so the full-universe 1/N benchmark rebalances at all. |
| Benchmark | Full-universe 1/N equal weight (benchmark rebalance profile). Within the sweep, equal-weight long-short top-k is the baseline every allocation / risk / cost delta is attributed against. |
| Sweep controls (`backtest.sweep`, via `get_top_n_predictions(case_study, stage)`) | `top_n_predictions: signal 0` (all predictions), `allocation 10` (top-10 model configs by equal-weight baseline Sharpe per label), `cost_sensitivity 1`, `risk_overlay 1` (top-1 per label). `expensive_allocators_skip: false`. `allocators`: `score_weighted`, `inverse_vol`, `risk_parity`, `mvo_ledoit_wolf`, `hrp`, `conformal_weighted` (equal weight deliberately absent; no max-weight cap; no per-allocator lookback). `cost_grid_bps: [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]`. `risk_controls.position`: `stop_loss` 0.03/0.05/0.10/0.15, `trailing_stop` 0.01/0.02/0.03/0.05/0.10/0.15/0.20, `time_exit` 10/20/40 bars = 14 overlays; no portfolio-level block. |
| Eligibility rules | A candidate must be: a complete validation prediction set (`full_coverage = true`), a generation in force (lineage via `superseded_members`, not `identity_status`), solvent (`_ruined(returns)` false; `require_solvent=True`), canonical tier (previews with declared reductions never join an official population), ranked on common support, and not a cost-sensitivity row. Sequence models additionally need gap-safe lookback windows. |
| Modeling (`modeling.gbm`) | `libraries: [lightgbm]`, `preset: default`, `device: cpu`, `max_bin: 255` (LightGBM CPU default; 63 is the GPU default that had been wrongly carried over). |
| Causal (`causal.*`) | `treatment: mom_skip_recent`, `treatment_window: 252`, `confounders: [vol_gk_21d, vol_gk_63d, zscore_21d]`, `method: walk_forward_dml`. |

Run from repo root with `uv run python case_studies/fx_pairs/NN_*.py` for 01..19 in order (Python 3.14, `uv sync`). Notebooks 13-16 default `RUN_SWEEP=False` (report the registered result, no rerun). `uv run python scripts/download_artifacts.py --cs fx_pairs` fetches bundle `v3.1.0-artifacts` (current); `v3.0.0-artifacts` is the first-published generation; the 3.1 rebuild re-keyed every content-addressed hash, so never mix hashes across bundles. `run_log/registry.db` records every training run, prediction set and backtest by a hash of its producing specification; artifacts sit under `run_log/training/`, `run_log/predictions/`, `run_log/backtest/`.

## Pipeline stages

All notebooks live under `case_studies/fx_pairs/`.

| Stage | Notebook | Chapter | What it does | What it writes |
|---|---|---|---|---|
| Feasibility | `01_feasibility_analysis` | Ch6 (6.2-6.6) | Breadth at the daily decision, participation ratio of the return-correlation spectrum, wide-end round-trip cost per pair, move-to-cost by horizon, within-pair return autocorrelation, declared walk-forward folds | nothing (reads 4H bars + `config/setup.yaml`) |
| Labels | `02_labels` | Ch7 (7.2) | Aggregate 4H -> CME_FX sessions, then 1/5/21-session close-to-close forward returns with assertions; effective-sample count; pre-declared-sign baseline IC | `labels/fwd_ret_1d.parquet`, `labels/fwd_ret_5d.parquet`, `labels/fwd_ret_21d.parquet` |
| Features | `03_financial_features` | Ch8 (8.2-8.4) | Momentum, risk-adjusted momentum, mean reversion, volatility, range/drawdown, oscillator/trend, dollar factor, cross-sectional rank; matrix reading (scale, dispersion, redundancy tree cut 0.7, decay to 63 bars) | `features/financial.parquet` + `.digest.json` sidecar |
| Temporal | `04_model_based_features` | Ch9 (9.2 Kalman, 9.3 ARIMA, 9.5 HMM) | Walk-forward Kalman local linear trend per pair, 2-state HMM on the dollar factor, ARIMA(1,0,1) one-step surprise | `features/model_based.parquet` (10 columns: 5 Kalman, 2 HMM/dollar-regime, 3 ARIMA; keys `timestamp`, `symbol`; no `fold` column) |
| Evaluation | `05_evaluation` | Ch7 (7.3, 7.4) | Per-session Spearman IC of every column vs next-session return on validation windows only; HAC mean; BH FDR; triage decision per column | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` (read by `20_strategy_synthesis/02_feature_evaluation.py`) |
| Linear | `06_linear` | Ch11 (+ Ch6 6.7 run log) | OLS, Ridge, LASSO, ElasticNet penalty sweep (28 configs) fitted on the union of all three labels | training runs + prediction sets in `run_log/registry.db`; coefficients `run_log/training/{hash}/`, scores `run_log/predictions/{hash}/` |
| GBM | `07_gbm` | Ch12 (12.2, 12.3) | LightGBM: 15 configs (capacity x loss) x 10 checkpoints = 150 candidates per label | training runs + one prediction set per config AND checkpoint; boosters, `learning_curves.parquet`, `fold_metrics.parquet` under `run_log/training/{hash}/` |
| Tabular DL | `08_tabular_dl` | Ch12 (12.3) | TabM `tabm_s/m/l` via the shared TabM runner | checkpoints under `run_log/training/tabular_dl/`; every declared epoch checkpoint is a separate prediction set |
| TCN | `09_dl_tcn` | Ch13 (13.2, 13.4) | Dilated causal convolutions | checkpoints under `run_log/training/deep_learning/` |
| NLinear | `10_dl_nlinear` | Ch13 (13.4) | Tests whether FX dynamics are approximately linear (subtracts last observed level over lookback) | same |
| LSTM | `10a_dl_lstm` | Ch13 (13.4) | Recurrent member `lstm_h64` | same |
| Causal DML | `11_causal_dml` | Ch15 (15.4) | Does momentum cause future returns or reflect overshooting? Walk-forward DML with block placebo | one row in registry `causal_runs` |
| Model Analysis | `12_model_analysis` | Ch11-15 | Cross-model daily rank IC, fold stability, bucket shape, chronological conformal coverage, prediction agreement, share of signal that is the single USD exposure | nothing (reads registry; decides nothing) |
| Backtest | `13_backtest` | Ch16 | Equal-weight long-short top-k (k in {5, 10}) for every complete prediction set x checkpoint; the ONLY selection stage | one backtest per prediction set x entry scheme: `daily_returns`, `weights`, `trades`, `fills`, `equity`, `portfolio_state` parquet + `spec.json` under `run_log/backtest/{hash}/`; registered with `stage='signal'` |
| Portfolio | `14_portfolio_management` | Ch17 | Six allocators on top-10 configs per label, lookback 63 | one backtest per allocation method (allocation population) |
| Risk | `15_risk_management` | Ch19 | 14 position overlays on the rank-1 of {signal, allocation} per label; freezes the final candidate set | one backtest per overlay (risk_overlay population) |
| Costs | `16_costs` | Ch18 | 11-point per-leg cost grid on the rank-1 of {signal, allocation, risk_overlay} per label | one backtest per cost level (frozen; excluded from selection) |
| Holdout Predictions | `17_holdout_predictions` | Ch16-20 | Resolve the carrier, refit it on everything up to holdout minus 21-observation buffer | holdout predictions + selected checkpoint only |
| Holdout Backtest | `18_holdout_backtest` | Ch16-20 | Replay the registered spec; only prediction set and price-frame identity change | one holdout backtest |
| Strategy Analysis | `19_strategy_analysis` | Ch20 | Reproduce the selection from the frozen set, verify holdout lineage, report validation + holdout with interval and paired evidence | registry rows `cohort_metrics`, `backtest_paired_metrics`; no files; fails if 17/18 are missing |

## Design decisions and why

| Decision | Choice | Why |
|---|---|---|
| Cadence | Rebalance daily on the NY 5PM close, not on the 4H grid | A 4H move clears its round trip "about as often as not"; a 1-day move far more often. The 4H grid also drops pairs the daily snapshot keeps. |
| Session boundary | Aggregate on `decision.session_calendar: CME_FX` before computing anything | UTC-date aggregation moves ~a third of every day into the wrong row; every forward return inherits it. Assert the boundary (03 Section B); do not describe it. |
| Position space | Long-short, k long + k short | A pair is already a relative price: the sold side needs no borrow and costs what the bought side costs; holding both sides spends the whole breadth. |
| Baseline sizing | Equal weight in 13 | "The choice not to choose a size": isolates the ranking's contribution so allocation / risk / cost stages can attribute their deltas. |
| Labels | Three horizons, lead with `fwd_ret_1d` (the traded, weakest one) | The one-day horizon is where the strategy trades; the case study "leads with its weakest label" on purpose. Three labels agreeing on a shape is stronger evidence than one because they share only features and folds. |
| Feasibility stop rules | Three results would stop it: typical move < round-trip cost; no ordering-return relationship at any horizon; result entirely a USD directional bet | "A feasibility study is only useful if some result would have stopped it." Each is measured where its evidence exists (01, 05/Ch7, 12). |
| Hypothesis signs | Write down the sign each hypothesis predicts BEFORE measuring | The same trailing return supports momentum or reversal by coefficient sign; a coefficient indistinguishable from zero disconfirms neither and leaves the number to beat at zero. |
| Feature register | Every window lives in `setup.yaml::features`; 03 binds, never retypes; `lag: 0` throughout | Every input is a spot bar on the tape at the 5PM snapshot. `lookback` = longest trailing chain = floor the warmup audit holds each column to. |
| Fitted features | Bounded by declared burn-in + refit schedule (`model_based.*`), NOT by CV folds; no `fold` column | Fitting per fold gives training rows parameters from their own future while validation rows get past-only parameters; model fitted on one version of the column, scored on another, nothing raises. One row per pair-session serves every fold and label. |
| HMM output | Filtered probabilities via `case_studies.utils.temporal.filtered_state_probs` | "Take the forward answer, not the more accurate one": `predict_proba` smooths over the whole series. |
| Holdout parameters | Frozen at the last pre-holdout estimate (up to two years stale) | Deliberate: declared staleness is the cost of a forward-only feature. |
| Evaluation output | A decision per column with evidence (`PROCEED`/`REVISE`/`STOP` + `note`), not a shortlist; ledger does not gate what 06 may use | A shortlist discards the reasons the next stage needs. 06 takes the full feature matrix. |
| Linear fitting | One run fits the union of labels; pre-fit binding check that every config sees the same pairs, dates, folds | Isolates the horizon effect with everything else fixed; readers can run their own configs into a private copy of the run log and compare on the same footing. |
| GBM checkpoints | Every checkpoint is a separate prediction set and part of the identity; `interior_peaks` reported but never argued from | "The checkpoint is a decision of the same order as the model"; choosing a stop after seeing validation folds is selection. 15 x 10 = 150 candidates per label is the trial count. |
| GBM `max_bin` | 255 | CPU default; 63 (GPU default) had quantized the design matrix into a quarter of the bins. |
| DL runs | Preprocessing fitted inside each training fold; canonical vs preview tier; epochs/lookback/batch are identity-changing overrides | Previews must declare reductions, take an isolated workspace, and cannot join an official population. |
| Causal evidence | DML row never enters the selection funnel | Causal evidence is a different kind of evidence; its value is that the market-structure narrative is true, not that the model trades. |
| Selection point | ONLY `13_backtest`, on validation backtest Sharpe | IC rankings in 06/07/08 and diagnostics in 12 "decide nothing"; IC ignores costs and turnover. Best-of-grid Sharpe is high partly because the grid is large. |
| Allocation stage | Top-10 configs per label, one slot per configuration (not per result), six allocators, shared lookback 63, no cap, equal weight excluded | Checkpoint-rich families would otherwise crowd selection; a cap would hide each allocator's natural concentration profile; listing equal weight again would double-count the baseline. Leaf-by-leaf spec diff vs the baseline sibling. |
| Sweep populations | Immutable expected population written before any member runs | Survivor-only sweeps flatter themselves; failed jobs leave visible gaps. |
| Risk stage | Seal {signal, allocation}; one parent per label; one rule per sibling; freeze the full risk population | A stop is the first path-dependent stage; "best" is only a statement about a fixed set; two fields changed cannot be subtracted. |
| Cost stage | Swept LAST, over {signal, allocation, risk_overlay}; perturbation, not a choice; rows frozen and excluded | A strategy allowed to compete on its cost assumption wins by assuming costs away; the curve is only informative about the strategy actually traded. |
| Cost regime | Proportional (bps per leg), never per-unit | FX spreads are pips, a fraction of the rate; a flat per-unit charge would wrongly assume spread scales with level. |
| Holdout conduct | No lock, no seal, no gate; refit (17) and backtest (18) are separate notebooks; selection re-resolved by registry query in 17/18/19 | A research lock made the holdout a one-shot transaction, uncorrectable after a bug fix. Passing the choice through a file or parameter adds a way for notebooks to disagree. Interpretability is the reader's conduct ("do not run it 500 times"). |
| Holdout lineage | Matched by configuration (family, config name, label, checkpoint), not by training hash or family+name | A genuine retrain has a different hash; several training specs share family+name. |
| Reporting | Intervals reported, never converted; prose says "baseline (equal-weight) Sharpe", never "signal Sharpe" | A zero-spanning interval is not a sign. |

### Decision rules and thresholds

| Rule | Value |
|---|---|
| Feature screen order (05) | (1) close holdout gap: usable boundary = `holdout_start` minus label horizon, per instrument; (2) coverage >= `COVERAGE_FLOOR = 0.70` and staleness <= `STALENESS_CEILING = 0.50`, else STOP; (3) per-session Spearman IC, HAC mean; (4) fold-direction agreement >= `STABILITY_THRESHOLD = 0.60` in the column's own sign; (5) BH FDR `alpha=0.05` over the whole declared set with HAC p-values; (6) PROCEED if `fdr_significant`, or exploration route if stability >= 0.60 AND |mean IC| >= `IC_THRESHOLD = 0.005`; market-level columns (constant across pairs on >= `MARKET_LEVEL_SHARE = 0.90` of sessions) -> REVISE |
| Other 05 constants | `N_QUANTILES = 5`, `N_SHAPE_PANELS = 6`, `N_LEADERS = 8`, `MIN_FOLD_DAYS = 5`, `MIN_SESSIONS = 20`, `MIN_PERIODS = max(3, n_symbols // 4)`, `MAX_SYMBOLS = 0`, `MAX_FOLDS = 0` (0 = no reduction); `REDUNDANCY_CUT` and `LABEL_HORIZONS` read from setup.yaml |
| IC floor meaning | 0.005 is an order of magnitude below what an FX ranking strategy needs to earn back spread: clearing it is a floor, not a pass |
| Redundancy / persistence | Cut the rank-correlation tree at `redundancy_cut: 0.7`; read autocorrelation out to `persistence_horizon: 63`; a feature whose ordering decays inside the rebalance cycle cannot be traded at that cadence; a feature with no cross-sectional spread cannot rank |
| Row retention | `null_policy_carrier: zscore_126d` (longest chain, lookback 378 = 252 z-window + 126 horizon) |
| Model-based burn-ins (sessions) | Kalman `burnin 252`, `refit_every 63`, `maxiter 300` (first value 2011-12-22; costs oldest fold 251 of 1,289); HMM `n_states 2`, `burnin 504`, `refit_every 63`, `n_restarts 10`, `stability_rel_tol 0.001` (dollar factor starts 2011-02-01, first regime 2013-01-15 = 504 of 1,269); ARIMA `order [1, 0, 1]`, `burnin 252`, `refit_every 21` (first value 2011-12-23; costs 252 of 1,289). Burn-in paid from the oldest fold's training window, never a validation window. |
| Linear | Prefer heavy penalties; at every horizon the IC-vs-alpha curve crosses zero mid-grid; `best_config` differs per label, so "best" means best on that label |
| GBM | Squared error last at every horizon; Huber leads 1d and 21d, MAE leads 5d; 500 trees is not a tuned quantity; carry checkpoints forward rather than choosing a stop |
| Selection sequencing | 13 (all predictions, k in {5, 10}, equal weight) -> 14 (top-10 configs per label, 6 allocators) -> 15 (top-1 of {signal, allocation}, 14 overlays) -> 16 (top-1 of {signal, allocation, risk_overlay}, 11 cost levels) -> 17/18 (holdout once) -> 19 |
| Parent / carrier rule | argmax validation Sharpe; tie -> deterministic backtest identity (preserves the simpler equal-weight spec); solvent; generation in force; ranked on common support; cost rows excluded |
| Overlay gap (16) | best-overlay Sharpe minus best-un-overlaid Sharpe; negative -> controls did not help on that label -> parent comes from signal/allocation; print the parent's stage |
| Cost reading | Read the zero-Sharpe breakeven against the quoted band (majors 1-3 bps, crosses 3-8 bps); the Sharpe at 0 bps or at any single rate is not the decision quantity; most of 0-50 is deliberately implausible |
| Rebalance skip | Skip only if weight change < 0.005 AND trade notional < 100.0; benchmark profile 0 / 0 |
| Analysis gates (inference from helper names) | Gate 1: validation Sharpe CI lower bound >= 0 (`gate1_validation_sharpe_geq_zero`); Gate 2: holdout-minus-validation (or paired) difference interval not wholly below zero (`gate2_holdout_diff_not_excludes_zero_negatively`); `ci_status` is three-tier (wholly above / spans / wholly below zero) |
| Reduced runs | `START_DATE`, `MAX_SYMBOLS`, `MAX_FOLDS`, `*_OVERRIDE` (0 = use setup.yaml) create previews; previews never publish official populations |

## Market-specific guardrails

### Data, sessions and labels

| Guardrail | Why | How |
|---|---|---|
| Aggregate on the CME_FX 5PM rollover, never on UTC date | ~a third of every day lands in the wrong row; every forward return inherits it | `decision.session_calendar`; assert the boundary in 03 Section B |
| No stale "last bar" snapshots | Falling back to each instrument's last observed bar fills a missing price with an earlier one; the panel stops being one instant wide and the decision timestamp silently moves | Derive the decision timestamp from the calendar, check it against the declared close, count quoting instruments at that exact moment (01) |
| Instrument count is not bet count | 20 pairs among 8 currencies = 5.27 effective bets (accounting identities such as EUR_USD x GBP_USD -> EUR_GBP); IC counts one dollar bet 20 times | `pr = eigenvalues.sum()**2 / (eigenvalues**2).sum()` on the return-correlation spectrum; judge via portfolio backtest; 12 measures the USD-exposure share |
| Collinear features vs dependent cross-section | Regularization fixes collinear columns; nothing fixes linked rows | Route predictions through `13_backtest`, not IC |
| Autocorrelation within each instrument, then average | Stacked-panel autocorrelation measures the joins between instruments | 01 computes lags 1..N per pair and averages |
| Forward-return labels assert, not describe | Incomplete windows, cross-instrument returns and quoting gaps all produce plausible numbers | 02 asserts complete forward window, same instrument, no quoting gap |
| Effective-sample sanity check | A weighting that counts the anchor session halves the count at 1 day and reads as a refinement | At the 1-day horizon effective count must equal row count |
| Close-to-close label vs next-open fill | Label and fill price differ; the gap is unmeasured in 02 | Backtest fills at `next_bar_open`; keep the difference in mind reading IC vs Sharpe |
| Holdout boundary by label END date | A row observed before 2024-01-01 whose outcome resolves inside the holdout is a holdout row | Usable boundary = `holdout_start` minus horizon on each instrument's own sessions; 05 closes this before any statistic |
| Overlapping holding periods | A 21-day label traded daily stacks 21 overlapping positions and misstates turnover and risk | `labels.rebalance_step` 1 / 5 / 21 |
| Universe survivorship | 20 pairs chosen knowing today's liquidity | No in-notebook fix; disclose |

### Features (financial and model-based)

| Guardrail | Why | How |
|---|---|---|
| Fitted transforms must not read the whole sample | A z-score or scaler estimated over the full panel reads the future | Rebuild the panel with later dates removed and compare value-by-value (03) |
| No fold-wise fitting of model-based features | Training rows get parameters from their own future, validation rows do not; nothing raises | Declared burn-in + refit schedule; no `fold` column; parameters from strictly earlier sessions |
| Filtered, not smoothed, HMM probabilities | `predict_proba` conditions on the whole series | `filtered_state_probs` forward recursion; verify by truncation |
| Truncation test | Reproducibility needs more than a seed when numerics are multithreaded | Delete the tail, re-run the recursion, earlier values must not move (04) |
| Refit staleness is a declared cost | Estimates are held fixed between refits (quarter for Kalman/HMM, month for ARIMA); holdout params frozen up to 2 years | Accept; print burn-in session counts (04 Section 4) |
| Imputed burn-in rows | A model whose training window opens at panel start fits on imputed values | `null_policy_carrier` governs retention; 04 prints affected sessions |
| `kalman_smoothness` is not a price signal | It is a function of noise parameters and session index only | Treat as a per-block near-constant state, not a cross-sectional signal |
| Dollar factor is a signed average, not an estimated factor | Says nothing about EUR or JPY blocs | Keep it in the `state` role |
| Volume is tick count at one venue | Not participation | Enters only via daily range |
| Sequence windows must not cross gaps | Positional indexing hides a missing observation inside a lookback | Gap-safe eligibility grid; validation primed with earlier observable rows only; coverage may be smaller than the raw panel but exact |

### Evaluation and modeling

| Guardrail | Why | How |
|---|---|---|
| Market-level columns cannot rank | HMM dollar-regime columns take the same value for every pair | `MARKET_LEVEL_SHARE = 0.90` screen -> REVISE; test differently |
| Screen empty / frozen columns BEFORE inference | They produce p-values that look like the others | Coverage >= 0.70 and staleness <= 0.50, else STOP |
| Multiple testing over ~63 columns | A handful of small p-values is expected by chance | Declare the search; `benjamini_hochberg_fdr(hac_p_values, alpha=0.05, return_details=True)` over the whole set; label exploration-route PROCEEDs in the ledger `note` |
| Serially dependent daily IC | One day's agreement resembles the next; overlapping 5d/21d windows worsen it | `compute_ic_hac_stats(series, ic_col="ic", label_horizon=H)` (horizon sets bandwidth); record `naive_t` and `hac_t`; report fold agreement |
| Cross-sectional dependence inflates effective N | Each daily IC is from 20 non-independent pairs | Acknowledged uncorrected limitation in 06/07; rely on holdout and portfolio evidence |
| Write IC series first, read it back | Every figure/average must come from the stored copy | 05 writes `ic_timeseries.parquet` as soon as computed |
| Unregularized / weak-penalty linear fits rank backwards | OLS chases relationships that reverse out of sample | Heavy penalties; read the IC-vs-alpha curve crossing zero mid-grid |
| Partial-coverage configs are not comparable | Aggressive L1 yields constant predictions on some dates -> no rank correlation -> smaller sample, higher IC | Flag `full_coverage = false`; exclude from charts |
| Never select a GBM stopping point on validation curves | Choosing after seeing validation folds; the max of a serially correlated path piles up at the ends | Carry every checkpoint into 13 as part of the identity; report `interior_peaks` descriptively |
| Never report the best of 150 GBM candidates as one experiment | Max of 150 noisy numbers | Count configs x checkpoints x portfolio sizes as the trial count |
| Capacity is not free | A tree that conditions one feature on another can discover a condition that held in training and not after; negative OOS IC is what a discovered-and-reversed relationship looks like | Read near-zero negative IC as "does not rank", large negative as "reliably wrong" |
| Rank correlation is a diagnostic, not a checkpoint-selection rule | Selection belongs to 13 | 08 and 12 record it in the catalog only |
| Placebo blocks must span the treatment window | Blocks of 1/5/21 on a 252-session column destroy serial dependence, narrow the null, p-value reads stronger than evidence | `causal.treatment_window: 252` declared, not inferred; register DML only after finite effect + HAC SE; outcomes reaching the holdout excluded |

### Selection, sweeps, risk and costs

| Guardrail | Why | How |
|---|---|---|
| Never select on IC | IC ignores costs and turnover; a signal that ranks well but churns loses to spreads | Select only on validation backtest Sharpe in 13 |
| `identity_status == "current"` is not "still published" | It names the schema version; after a refit both generations look complete and retired predictions still backtest correctly | Filter by lineage via `superseded_members`; print the excluded count; `population_supersedes` on republish; `OfficialPopulation.create` refuses a changed member list under an existing name otherwise |
| Read the frozen population from 13; do not rebuild the catalog in 14 | Convention drift -> two sides never measured on the same set | 14 reads the exact frozen population |
| One field per sibling | Changing allocator / stop / cost AND anything else is a valid backtest of a different strategy; nothing looks wrong | Leaf-by-leaf spec diff vs the baseline sibling; identity audit (model, checkpoint, signal, allocation, execution, costs, rebalance rule, account config, price-frame identity) |
| Seal the cohort before ranking | A cohort that can gain members after selection lets a later run change the choice with no record | 15 seals {signal, allocation}; 19 reproduces the selection from the immutable set or fails |
| Same parent for risk and cost siblings | Otherwise a cost result and a risk result describe different strategies | 15 and 16 share the cohort and the `resolve_canonical_rank1_lineage` rule; print the parent's stage |
| Cost rows never enter a selection cohort | The ranking would report the kindest cost assumption, not the best strategy | Cost variants are descendants, excluded in 15/16/17/18/19 |
| Insolvent paths rank below solvent ones | A ruined path can show a high point Sharpe | `_ruined(returns)`; `require_solvent=True` raises on an insolvent or unmeasured carrier |
| Deterministic tie-break | Exact Sharpe ties happen by construction (two allocator rows reproducing the equal-weight baseline); `ORDER BY` alone is silently wrong | Tie-break on backtest identity |
| Rank on common support | A candidate scored over fewer days must not rank beside those that scored all of it | `rank_backtests_on_common_support`; conformal candidates with different coverage take a dedicated path (inference) |
| Stop level chosen on validation is in-sample | The improvement is a fitted number | Treat as a hypothesis; the holdout "is allowed to answer no"; report the number of variants tried beside the improvement |
| Stops are path-dependent | Their effect depends on what the bars carry (intrabar path) and the fill model | Know which price fields the fill model reads; do not compare stop results across bar/fill configurations |
| No portfolio-level controls exist | Nothing restrains the aggregate book | A drawdown cap or daily loss limit would be a separate declared control |
| Proportional costs only | A per-unit charge assumes spread scales with rate level | bps per leg; per-share only where nominal prices are stable or spreads come from quote data |
| The cost curve is a turnover measurement | A single-rate Sharpe hides turnover; a constant rate cannot show that turnover rises when spreads widen | Read the breakeven vs the quoted band; treat it as a validation-period bound |
| Swap/financing is unpriced | `components` is a taxonomy; a position against the rate differential costs more than anything charged | Report as limitation; Ch18 (16) perturbs only the aggregate |
| Hash invalidation | `initial_cash`, `share_type` are spec-hash inputs; v3.0.0 vs v3.1.0 re-keyed | Read via `get_backtest_config()`; never mix hashes across bundles |

### Holdout and analysis

| Guardrail | Why | How |
|---|---|---|
| Use the WIDEST label buffer (21D) for the refit window, even when the selected label is `fwd_ret_1d` | The same training window serves whichever label selection landed on; a 1D buffer leaves ~three weeks of `fwd_ret_21d` outcomes reachable from training (inference on exact count) | Training window ends 21 observations before the first holdout session, counted in observations (21 FX sessions = ~29-31 calendar days) |
| Do not assume the holdout grid | Three labels on one daily grid is exactly the coincidence that lets an unchecked assumption survive | Read the label from the selected lineage; step the interval back along that label's observation grid |
| Validation folds are read many times | Every IC in 06/07/12/13/14 reuses the same data | Spend the holdout once, at the end, against the whole candidate set |
| Repeated holdout looks are uninterpretable | Software cannot prevent "run 500 times and quote the best" | Conduct rule: ask once; `refuse_a_second_look` and `_refuse_a_selection_disagreement` refuse when the registered holdout disagrees with the resolver |
| No research lock | A one-shot transaction cannot be corrected after a bug fix | Retrain-predict-backtest is re-runnable; selection is protected by being upstream and frozen |
| Refit and backtest in separate notebooks | A backtest failure in one transaction discards the model half | 17 writes predictions only; 18 backtests only |
| Re-resolve, never hand over | A file or parameter lets two notebooks disagree without either being wrong | 17/18/19 each call `resolve_solvent_carrier` |
| Match holdout lineage by configuration | A retrain has a new training hash; family+name are shared by several specs | Match on family, configuration name, label, checkpoint (`is_refit_of`, `holdout_refit_status`) |
| Name a missing refit explicitly | An empty query reads as "no strategy" rather than "17 has not run" | 18 raises with the missing result named; `status.found` |
| Nothing on the holdout revises the selection | Any revision turns the holdout into another validation fold | 19 writes nothing and only reads |
| Never convert a zero-spanning interval into a claim | It manufactures a sign the evidence does not carry | Computed sentences report only intervals wholly above or below zero |
| Preview results stay outside cohorts | Preview mode resolves a reduced prediction catalog; rows are not comparable | Tier and reductions must agree; isolated workspace and identity |

## Results and lessons

Numbers are from the notes' digest of the notebook narratives; the README deliberately states no results because the registry is rebuilt on re-derivation. `19_strategy_analysis` and `run_log/registry.db` (bundle `v3.1.0-artifacts`) are the source of truth for Sharpe, breakeven and gate outcomes.

| Stage | Finding |
|---|---|
| Feasibility (01) | All 20 pairs quote on every one of 3,355 development decision dates; participation ratio 5.27 (~5.3 effective bets, about a quarter of the nominal count); 8 folds generated, last validation window ends 2023-12-28. |
| Feature triage (05) | 63 columns -> 27 PROCEED, 32 REVISE, 4 STOP. ALL 27 PROCEEDs came through the exploration route (stability >= 0.60 and |IC| >= 0.005); none cleared BH FDR at alpha 0.05. The next stage therefore starts from candidates, not findings. |
| Linear (06) | OLS and weakly penalized Ridge rank the cross-section backwards on all three labels; IC crosses zero mid-grid and peaks well up the penalty range. 10 of 28 one-day configs end above zero; one-day IC spread top-to-bottom is "a few thousandths". Best IC at 21d is several times the best at 1d, but the 21d grid's worst-to-best range exceeds the best-21d-minus-best-1d gap (penalty matters more than horizon). Rank correlation between horizon orderings is well above 0 and well below 1. 5d aggressive-L1 configs post higher IC on fewer dates and are excluded (`full_coverage = false`). Honest summary at the traded horizon: "small and positive". |
| GBM (07) | Only 2 of 15 one-day configs end above zero, both by < 0.002. At 1d and 21d the median config ends below where it started and two thirds end lower; at 5d median change is zero to five decimals. `interior_peaks` 7, 8, 8 of 15 (descriptive only). 21d is well ahead on both measures but 1d and 5d are level, so the horizon effect does not reproduce cleanly. MSE is bottom at all three labels and finishes above zero only once; Huber leads 1d/21d, MAE leads 5d. The 1d label has by far the heaviest tails yet the objective gap is widest at 5d and narrowest at 21d (tail mechanism not supported). Config spread vs checkpoint range ratio ~1-3 by label (near 1 at 21d: the checkpoint matters as much as the config). Honest summary: GBM "does not rank this cross-section in either direction". |
| Tabular DL / TCN / NLinear / LSTM (08-10a) | Numeric results not in the notes. Structural: every epoch checkpoint is a separate prediction set; hyperparameters (TabM sizes, TCN dilation/lookback, LSTM depth, epochs, batch) not visible in the digest. |
| Causal DML (11) | Registered only after finite effect + HAC SE and 252-block placebo; numeric effect not in the notes. Never a backtest candidate. |
| Backtest / allocation (13, 14) | Sharpe tables deliberately not restated (registry is the source). 13 notes best-of-grid validation Sharpe is high partly because the grid is large: 28 + 150 + TabM + DL checkpoints, each at k in {5, 10}, per label. |
| Risk (15) | No numbers in the notes. 16 prints per label the stage the parent came from and the overlay gap; the author states a negative gap is the measurement that controls did not help, implying this occurs for at least some labels (inference). The 15 markdown mentions "four fixed stop-loss levels and five trailing stops" while setup.yaml declares 4 + 7 + 3 rules (unresolved discrepancy). |
| Costs (16) | No breakeven numbers in the notes. The 0-50 bps grid is deliberately mostly implausible; read the zero crossing against 1-3 bps (majors) / 3-8 bps (crosses). Swap points unpriced. |
| Holdout / analysis (17-19) | No gate outcomes in the notes. `strategy_analysis.py` records that two allocator rows can tie the rank-1 Sharpe exactly (tie-break keeps the equal-weight spec) and a dated note (2026-09-07) that `deep_learning/tcn` on `fwd_ret_21d` had two backtests tied at Sharpe; whether these refer to FX is not visible. The author's stated expectation: a stop's validation improvement may not survive the holdout. |

Lessons an agent should carry:

- A dependent cross-section (5.3 bets from 20 names) means IC is inflated by construction; the backtest is the arbiter and even it is noisy.
- At a one-day FX horizon the only positive evidence came from heavily penalized linear models, with IC spread of thousandths; trees found nothing and lost capacity to reversed relationships.
- Checkpoint choice is as large a source of variation as model configuration for GBM (ratio near 1 at 21d).
- Univariate PROCEEDs by the exploration route are candidates, not findings; search accounting (63 columns, 150 GBM candidates, 14 overlays, 11 cost points, three labels) is the denominator of every later claim.
- Cost survival is decided by where the cost curve crosses zero relative to the 1-8 bps band, never by the 0-bps Sharpe.

## How to adapt this pattern to a new dataset

1. Write `config/setup.yaml` first: `universe`, `decision` (cadence, snapshot, `execution_delay`, `session_calendar`), `execution` (`initial_cash`, `share_type`, `allocator_lookback`), `mapping`, `costs`, `backtest.rebalance` and `backtest.sweep`, `features` (windows, `ranked`, `null_policy_carrier`, `persistence_horizon`, `redundancy_cut`, `families` with `lookback` and `lag`), `model_based` (burn-in, refit cadence, restarts), `evaluation`, `labels` (primary, variants, buffers, `rebalance_step`), `modeling`, `causal`. Notebooks bind, never retype.
2. Pick a session calendar that matches the venue's rollover (CME_FX for 24h FX) and a separate evaluation calendar for the CV splitter; aggregate intraday bars to sessions BEFORE labels or features; assert the boundary.
3. Run feasibility with stop rules declared in advance: breadth at the exact decision timestamp (floor = 2 x max top_k), participation ratio of the correlation spectrum, move-to-cost per horizon at the wide end of the quoted spread (`move * 1e4 / cost_bps`), within-instrument autocorrelation averaged across instruments.
4. Build labels with assertions (complete window, same instrument, no gap); check the 1-period effective-sample count equals the row count; pre-declare the sign each hypothesis predicts.
5. Register features in families with role (`signal` vs `state`), lookback floor and `lag`; cut redundancy at 0.7; read persistence to the rebalance horizon; retain rows from the longest-chain carrier. Exclude venue-specific volume from participation features.
6. Fit model-based features on a declared burn-in + refit schedule (Kalman 252/63/300, HMM 504/63/10 restarts/1e-3, ARIMA 252/21/(1,0,1) are the FX defaults), parameters strictly from earlier sessions, filtered not smoothed, no fold column; verify by truncation; freeze parameters for the holdout.
7. Evaluate univariately on validation windows only: close the holdout gap by label horizon per instrument; coverage >= 0.70, staleness <= 0.50; HAC-SE mean IC with bandwidth from the label horizon; fold agreement >= 0.60; BH FDR at 0.05 over the declared set; exploration route needs |IC| >= 0.005; market-level columns to REVISE. Write a ledger with `note`, not a shortlist.
8. Train the declared menus (`config/training/{label}.yaml`): linear penalty sweep on the union of labels with a pre-fit binding check and `full_coverage` flag; GBM capacity x loss x checkpoint with every checkpoint published; TabM and sequence models with fold-internal preprocessing, gap-safe eligibility and preview/canonical tiers; DML with a placebo block that spans the treatment's construction window.
9. Select ONCE in the baseline backtest on validation Sharpe (equal-weight long-short top-k over every complete prediction set x checkpoint x k); never on IC; filter retired generations with `superseded_members`; write the expected population before running members.
10. Allocation: top-N configs per label, one slot per configuration, vary only the allocator (shared lookback, no cap, equal weight excluded); leaf-by-leaf spec diff vs the baseline sibling.
11. Risk: seal {signal, allocation}; one parent per label by `resolve_solvent_carrier` rule; one predeclared position rule per sibling; freeze the final candidate set and report its size as the trial count. Add portfolio-level controls as separate declared entries if needed (FX has none).
12. Costs LAST: parent from {signal, allocation, risk_overlay}; print its stage and the overlay gap; sweep a proportional per-leg grid; read the breakeven against the quoted band; keep cost rows out of every selection. Use per-unit costs only where nominal prices are stable or spreads come from quotes.
13. Holdout: re-resolve the carrier by registry query in each notebook; step the window back along the selected label's observation grid by the WIDEST configured buffer; refit (predictions only) and backtest (spec with only prediction set and price-frame identity swapped) in separate notebooks; match lineage by configuration; ask once; let the holdout say no.
14. Analysis: reproduce the selection from the frozen set or fail; report three-tier CI status and paired evidence; Gate 1 (validation Sharpe CI lower bound >= 0) and Gate 2 (holdout difference interval not wholly below zero); write registry rows (`cohort_metrics`, `backtest_paired_metrics`), never prose numbers in a README.
15. Keep `initial_cash` / `share_type` out of notebooks (`get_backtest_config()`), never mix content-addressed hashes across artifact bundles, and say "baseline (equal-weight) Sharpe", never "signal Sharpe".

## Related references

- `chapters/06_strategy_definition.md` - feasibility stop rules, breadth, participation ratio, move-to-cost, run-log and search accounting (6.7).
- `chapters/07_defining_the_learning_task.md` - forward-return labels, univariate IC triage, HAC SE, BH FDR, holdout-gap arithmetic.
- `chapters/08_financial_features.md` - momentum / volatility / mean-reversion / dollar-factor families, redundancy and persistence reading.
- `chapters/09_model_based_features.md` - walk-forward Kalman, HMM (filtered vs smoothed), ARIMA surprise, burn-in and refit schedules.
- `chapters/11_ml_pipeline.md` - linear penalty sweep, binding checks, prediction-set populations.
- `chapters/12_gradient_boosting.md` - LightGBM capacity x loss grid, checkpoints as identity, `max_bin`, TabM.
- `chapters/13_dl_time_series.md` - TCN, NLinear, LSTM, gap-safe sequence eligibility.
- `chapters/15_causal_estimation.md` - walk-forward DML, block placebo sized to the treatment window.
- `chapters/16_strategy_simulation.md` - equal-weight long-short baseline, selection on validation Sharpe, `superseded_members`.
- `chapters/17_portfolio_construction.md` - the six allocators, shared lookback, no-cap comparison.
- `chapters/18_transaction_costs.md` - proportional bps cost regime, cost curve as turnover measurement, breakeven reading.
- `chapters/19_risk_management.md` - position-level stops, trailing stops, time exits; path dependence.
- `chapters/20_strategy_synthesis.md` - sealed cohorts, holdout conduct, interval evidence, gates; `20_strategy_synthesis/02_feature_evaluation.py` reads this ledger.
- `case_studies/cme_futures.md` - same daily market-level HMM shape and ARIMA burn-in.
- `case_studies/etfs.md` - same quarterly HMM refit cadence.
- `case_studies/nasdaq100_microstructure.md` - intraday case where `expensive_allocators_skip` matters.
- `case_studies/us_equities_panel.md` - per-share cost regime to contrast with FX bps.
- `libraries/ml4t_diagnostic.md` - `compute_ic_hac_stats`, `compute_ic_uncertainty`, `benjamini_hochberg_fdr`.
- `libraries/ml4t_backtest.md` - FX backtest engine, rebalance thresholds, allocators, risk overlays.
- `libraries/ml4t_engineer.md` - feature construction patterns bound by the register.
- `libraries/ml4t_models.md` - LightGBM, TabM, sequence runners, Kalman/HMM/ARIMA.
- `workflow.md` - stage order and selection sequencing.
- `guardrails.md` - leakage, multiple testing, holdout conduct in general form.
- `decision_rules.md` - thresholds reused across case studies.
- `evidence.md` - how to cite registry-sourced numbers.
- `companion_repo.md` - `uv run`, `ML4T_DATA_PATH`, `scripts/download_artifacts.py`, registry layout.
- `glossary.md` - shared terms.
- Further reading: none in the notes (no literature citations were extracted for this case study).

## Glossary

- Participation ratio - `(sum lambda)^2 / sum lambda^2` of correlation eigenvalues; effective number of independent bets (5.27 here).
- Session calendar (CME_FX) - venue calendar whose 5PM NY rollover assigns 4H bars to trading days.
- Evaluation calendar (FX) - calendar the walk-forward splitter counts windows on.
- Move-to-cost - price move in bps divided by the pair's assumed round-trip cost in bps.
- Coverage - share of sessions a column holds a value, from its first filled session (floor 0.70).
- Staleness - share of consecutive observations identical to the previous one (ceiling 0.50).
- Exploration route - PROCEED on fold stability >= 0.60 and |IC| >= 0.005 without FDR confirmation.
- Triage ledger - one decision per column (PROCEED / REVISE / STOP) with evidence and a `note` of the route.
- Market-level column - value constant across pairs on >= 0.90 of sessions; cannot rank.
- Null-policy carrier - the longest-chain column (`zscore_126d`) whose first valid bar sets row retention.
- Burn-in - sessions a fitted feature spends before its first estimate; carries no value.
- Refit cadence - sessions between parameter re-estimations (63 Kalman/HMM, 21 ARIMA).
- Filtered state probability - P(state | data up to t); smoothed uses the whole series.
- `kalman_smoothness` - inverse of level-estimate uncertainty; depends on noise parameters and session index only.
- Dollar factor - signed equal-weight average return of the 7 USD pairs; a proxy, not an estimated factor.
- Checkpoint - a scored point along a training run; part of model identity.
- Population - named, immutable set of identities (predictions or backtests) in the registry.
- `superseded_members` - identities retired by a later generation; the lineage filter (vs `identity_status`, a schema-version marker).
- Baseline backtest - equal-weight long-short top-k run registered with `stage='signal'`; "baseline (equal-weight) Sharpe".
- Parent strategy / carrier - per-label validation rank-1 from which cost and risk siblings descend; what the holdout refit carries forward (`resolve_solvent_carrier`).
- Sibling - a variant of a parent differing in exactly one declared field (allocator, risk rule, cost rate).
- Identity audit - check that all non-varied identity fields match between parent and sibling.
- Cohort / candidate set - frozen population of registered validation backtests a selection is made over; sealed before ranking.
- Solvent - a backtest whose equity path was not ruined (`_ruined`); insolvent runs rank below all solvent ones.
- Common support - shared observation set over which candidates are ranked.
- Position risk control - `stop_loss` (fixed % from entry), `trailing_stop` (% from running peak), `time_exit` (bars held); applied per position.
- Cost sensitivity - perturbation of the aggregate per-leg rate over `cost_grid_bps`; frozen, excluded from selection.
- Breakeven - cost rate at which the strategy's Sharpe crosses zero on the cost curve.
- Swap points - overnight financing component of FX cost; taxonomy item, never priced.
- Label buffer - observations between the last training label stamp and holdout start; the holdout refit uses the widest (21D).
- Observation grid - the panel's actual sessions; the holdout interval is stepped along it, not the calendar.
- Price-frame identity - names the price frame a backtest reads; the only thing besides the prediction set that changes for the holdout backtest.
- Research lock - abandoned design that pre-registered the holdout lineage as a one-shot transaction.
- CI status - three-tier classification: wholly above zero, spans zero, wholly below zero.
- Gate 1 / Gate 2 - validation Sharpe CI lower bound >= 0; holdout difference interval not wholly below zero (inference from helper names).
- Block placebo - refutation that permutes the treatment in blocks (252) to keep serial dependence.
- Cross-fitting - nuisance model never scores a row it was fitted on; the temporal embargo extends it across time.
- Preview tier - reduced run (`START_DATE`, `MAX_SYMBOLS`, `MAX_FOLDS`) with declared reductions and isolated identity; cannot join an official population.
