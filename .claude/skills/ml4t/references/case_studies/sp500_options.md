# Case study: S&P 500 options straddles (weekly Friday entry, held to expiry, ret_to_expiry)

> This case study trades options directly: a short ATM straddle on an S&P 500 constituent, written on the last available session of each ISO week, held to expiry (HTM) and delta-hedged daily. It is built around O'Donovan & Yu (2024): of 24 widely-cited single-name option-return predictors, 17 have significant gross long-short returns and none survive realistic transaction costs in the standard one-month delta-hedged framing. The headline lesson is methodological: an equity-style bps-of-notional cost model understates option spread cost by one to two orders of magnitude, so the case study switches to a premium-denominated cost framework and a hold-to-maturity engine as the cost mitigation itself, then quantifies what survives. Under full per-leg HTM cost accounting the signal is statistically null (IC and Sharpe both straddle zero), and that null result, reported with intervals and without "winner" language, is the case study's contribution to the Ch20 synthesis theme "cost model validity".

Repo path: `case_studies/sp500_options/` (paired `.py` jupytext + `.ipynb`; helpers `_htm_backtest.py`, `_ic_diagnostics.py`, `_label_artifacts.py`, `_straddle_moves.py`, `_underlying_returns.py`; `config/setup.yaml`, `config/backtest/base.yaml`, `config/training/{label}.yaml`; `benchmark/`).

## Setup contract

Source of truth: `case_studies/sp500_options/config/setup.yaml` (`strategy_id: sp500_options`, `setup_version: v1`). Notebooks bind these values; never retype them as local constants.

| Area | Key(s) | Value | Notes |
|---|---|---|---|
| Universe | `universe.underlying` / `strategy` / `n_assets` | `sp500_constituents` / `atm_straddle` / 627 | `n_assets` is display-only (includes membership churn); the backtest counts assets from data. Dev period: 605 stocks quoted, 599 on at least one decision date. |
| History | README | 2017-2021 | Holdout is 2021. |
| Data | `ML4T_DATA_PATH` | Materialized AlgoSeek S&P 500 straddles + daily underlying bars | One end-of-session quote per contract per day (`LastBidPrice`/`LastAskPrice`/`LastMidPrice`), no open/high/low. Missing licensed data fails at the loader boundary. |
| Decision cadence | `decision.entry_cadence` / `entry_time` | `weekly_friday` / `friday_close` | Entry = last available session of each ISO week (Thursday when Friday is a holiday). |
| Execution delay | `decision.execution_delay` | `next_session_close` | Maps via `backtest_presets._EXECUTION_MODE_BY_DELAY` to `next_bar` (same mode `MONDAY_OPEN` mapped to). Every fill is a close one session after the signal. |
| Legacy decision fields | `decision.holding_period_days` / `exit_time` | 10 / `10_days_after_entry_or_expiry` | Appear to be leftovers from the dropped 10d labels; effect on the HTM path unclear (uncertain). |
| Hedge | `decision.hedge_cadence`; `hedging_protocol.*` | `daily_close`; `hedge_instrument: underlying_stock`, `hedge_frequency: daily_close`, `delta_threshold: 0.1`, `gamma_hedging: false`, `vega_hedging: false` | `delta_threshold` read by `case_studies.utils.backtest_runner`; rehedge at close when net delta breaches 0.1. |
| Execution defaults | `execution.initial_cash` / `share_type` / `allocator_lookback` | `100_000` / `integer` / 63 | Cash and share_type are spec-hash inputs only (HTM uses fractional cohort weights); changing them invalidates every `backtest_hash`. 63 = ~3 months of daily underlying bars, applied uniformly to inverse_vol, risk_parity, hrp, mvo_ledoit_wolf. |
| Mapping | `mapping.*` | `class: systematic_straddle_sell`, `position_state_space: short_straddle_hedged`, `entry_logic: sell_atm_straddle_weekly`, `sizing: equal_premium_capital_with_fixed_cohort_fraction` | Equal premium capital within a cohort, then 1/n_roll per concurrent cohort; no fixed-vega target. 01 describes sizing as scaling by IV sensitivity (inference: equal premium capital ~ equal vega-ish exposure; ambiguous). |
| Labels | `labels.primary` / `buffer` / `buffer_unit` | `ret_to_expiry` / `35D` / `calendar` | Only calendar-anchored label in the nine case studies (default is `sessions` via `utils.artifact_specs.resolve_label_buffer_unit`). |
| Diagnostic labels | `labels.variants` / `variant_buffers` | `[]` / `{fwd_ret_5d: 5D, fwd_ret_10d: 10D, fwd_ret_dh_5d: 5D, fwd_ret_dh_10d: 10D}` | Declared so the written label set equals the declared set and `resolve_label_buffer`/`resolve_label_horizon` resolve; never in the sweep. Read by `03_financial_features`, `05_evaluation` (contrast) and `90_ic_diagnostic`; `fwd_ret_dh_10d` IC also orders champion configs in the training menus. |
| Rebalance step | `labels.rebalance_step.ret_to_expiry` | 5 | ceil(30/7) for weekly_friday + 30-day DTE; used by non-equal-weight allocation before HTM dispatch. |
| CV | `evaluation.n_splits` / `train_size` / `val_size` | 2 / `2Y` / `1Y` | Expanding single-window folds validating on 2019 and 2020. The last `val_end` is the last session before `holdout_start − labels.buffer` (35 calendar days; `utils/cv_splits.py::_purge_holdout_touching_validation`), and 01 asserts it equals its own hand-computed outcome seal. Read the date from that printout: 01's results cell types 2020-11-10, which its D.1 comment identifies as the sessions-counted seal. |
| Holdout | `evaluation.holdout_start` / `holdout_end` | `2021-01-01` / `2021-12-31` | Reported once; refit required (16). |
| Calendar | `evaluation.calendar` / `periods_per_year` | `NYSE` / 252 | Daily MTM over overlapping weekly cohorts. |
| Cost model | `costs.class` | `dominant` | Read directly by `_htm_backtest.py`; `commission.rate` and `slippage.rate` in `config/backtest/base.yaml` are inert here. |
| Option spread | `costs.components.option_spread.estimate_pct_of_premium` | `[2.0, 5.0]` | Entry-side only under HTM. |
| Hedge spread | `hedge_spread.estimate_bps_of_notional` / `hedges_per_holding` | 0.5 / 10 | Charged each session the hedge trades. |
| Commission | `commission.option_per_contract` / `equity_per_share` | 0.65 / 0.005 | Per leg / per share. |
| Margin | `margin_opportunity_cost.margin_pct_of_notional` / `opportunity_cost_annual_pct` | `[15, 20]` / 5.0 | Documented only; not charged inside the engine. |
| Cost note | `costs.cost_dominance_note` | "Total costs often exceed the expected VRP edge." | |
| Rebalance thresholds | `backtest.rebalance.default` / `benchmark` | `min_weight_change: 0.005`, `min_trade_value: 100.0` / both `0.0` | Skip when weight change < min AND trade notional < min; benchmark profile disabled so 1/N rebalances at all. |
| Canonical universe | `backtest.sweep.universe_filter` | `liquid` | Bottom 20% half-spread per rebalance date; `full` lives only in the Ch18 cascade. Spec field `strategy.signal.universe_filter` in {None (full), 'liquid'}. |
| Stage iteration | `backtest.sweep.top_n_predictions` | `signal: 0` (all), `allocation: 10`, `cost_sensitivity: 1`, `risk_overlay: 1` | Read via `get_top_n_predictions(case_study, stage)`. `expensive_allocators_skip: false`. |
| Concentration | `backtest.sweep.top_k_grid.ret_to_expiry` | `[5, 10, 20]` | Long-only equal-weight top-k; short sign handled by the label, not `long_short=True`. |
| Allocators | `backtest.sweep.allocators` | score_weighted, inverse_vol, risk_parity, mvo_ledoit_wolf, hrp, conformal_weighted | equal_weight omitted (it is the signal-stage baseline). conformal_weighted sizes by conformal interval width generated on demand from the prediction set. |
| Cost cascade | `backtest.sweep.htm_cost_cascade` | `cost_fractions: [0.203, 0.5, 0.75, 1.0]`, `universes: [full, liquid]`, `liquid_quantile: 0.20`, `top_k: 20` | 0.203 = Heston et al. 2023 "algo" anchor (Muravyev-Pearson 2020 $0.026/$0.128 ATM ratio); 0.75 ~ population-average effective/quoted; 1.0 = full quote crossed. Quintile is stricter than O&Y's bottom four deciles (~40%). Dispatched inline by Ch18, not via `run_backtest()` cost sweep. |
| Benchmark | 18 / `benchmark/` | Equal-weight (EW) universe benchmark | Overlaid on headline metrics; holdout-vs-EW paired comparison. |
| Model-based | `model_based.garch` | `burnin: 252`, `refit_every: 21`, `mean: Constant`, `vol: GARCH`, `p: 1`, `o: 1`, `q: 1`, `dist: Normal`, `rescale: true` | `o: 1` = GJR (declared deviation from shared GARCH(1,1)-Normal; same as `sp500_equity_option_analytics`). Expanding refits. |
| | `model_based.stochastic_volatility` | `burnin: 252`, `refit_every: 63`, `calibration_window: 252` | Rolling window (sampler cost is one latent state per observation). |
| Modeling | `modeling.gbm` | `libraries: [lightgbm]`, `preset: default`, `device: cpu`, `max_bin: 255` | 255 = LightGBM CPU default; 63 (GPU default) had been carried over and quantized into a quarter of the bins. |
| | `modeling.dl.device` | `cuda` | Enters training identity via `sequence_identity_params`; read by `research_workflow.declared_dl_device`. |
| Causal | `causal.*` | `treatment: vrp_21d`, `treatment_window: 21`, `confounders: [rv_21d, vrp_mom_5d, spread_pctl]`, `method: walk_forward_dml` | Placebo block length must cover the realized leg's 21 sessions. |

**Eligibility rules**
- A feature row is kept iff `vrp_21d` and `rv_21d` are measurable (`features.null_policy`); all other long-lookback columns ship as null where unsupported.
- Windows count on the underlying's NYSE session grid; a window produces a value only when >= `min_observations_fraction: 0.8` of its sessions carry a straddle quote.
- GARCH/SV values exist only after a 252-session burn-in per security-identity segment.
- Sequence models (NLinear, LSTM, PatchTST) need 60 sessions of symbol history, so they form a separate eligibility group with strictly fewer rows.
- Canonical backtests trade only the liquid quintile; `full` appears only in the cost cascade.
- Holdout exclusion is by label END (settlement) date, not signal date: 02 sets `END_OF[ret_to_expiry] = expiration` and keeps development rows with `_label_end < HOLDOUT_START`; a trade opened before 2021 that expires inside it is a holdout trade.

**Feature config (`features`, bound by 03)**

| Key | Value |
|---|---|
| `target_dte` / `hold_sessions` | 30 / 21 (~30 calendar days ~ 21 NYSE sessions; `target_dte` also puts `instr_dte` on [0, 1]) |
| `windows.underlying_return` | `[1, 5, 10, 21]` |
| `windows.realized_volatility` | `[5, 10, 21, 42, 63]` |
| `windows.volume_zscore` | 20 |
| `windows.instrument_return` / `instrument_cost_momentum` | `[1, 5]` / 5 |
| `windows.vrp` / `vrp_reference` / `vrp_zscore` / `vrp_momentum` | `[5, 10, 21, 42, 63]` / 21 / 252 / `[5, 10]` |
| `windows.iv_zscore` / `iv_momentum` | `[63, 252]` / `[5, 10, 21]` |
| `thresholds.vega_floor` / `realized_volatility_floor` | 0.001 / 0.01 (1% annualized floor of IV/RV ratio) |
| `ranked` (source -> shipped) | `vrp_21d -> vrp_21d_pctl`, `iv_atm -> iv_atm_pctl`, `instr_rel_spread -> spread_pctl`, `iv_rv_ratio -> iv_rv_ratio_pctl` |
| `metadata` (not features) | `underlying_price, instr_mid, instr_bid, instr_ask` |

## Pipeline stages

Run from repo root: `uv run python case_studies/sp500_options/NN_*.py`. Order: 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09 -> 09a -> 09b -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16 -> 17 -> 18 -> 90. Results bundle: `uv run python scripts/download_artifacts.py --cs sp500_options` (fetches `v3.1.0-artifacts`; `v3.0.0-artifacts` is as-first-published; hashes do not resolve across bundles).

| Stage | Notebook | Chapter | What it does | What it writes |
|---|---|---|---|---|
| Setup | `case_studies/sp500_options/01_feasibility_analysis` | Ch6 | Universe breadth, round-trip cost vs premium, premium persistence, fold structure. Reads straddle panel, raw chains, daily prices, IV summary, `setup.yaml`. | Nothing |
| Labels | `02_labels` | Ch7 | Prices the same contract at entry and exit from the raw chain; hedge path from underlying closes; reconciles unlabelled rows; effective count + overlap-corrected SE; one-signal VRP baseline. | `labels/{ret_to_expiry,fwd_ret_5d,fwd_ret_10d,fwd_ret_dh_5d,fwd_ret_dh_10d}.parquet` + `.digest.json`; intermediates `labels/contract_returns.parquet`, `labels/hedge_path.parquet` (via `_label_artifacts.py`) |
| Features | `03_financial_features` | Ch8 | VRP, IV surface, skew, Greeks, cross-sectional percentiles; session-grid windows; holdout-invariance check; F6 decay test. `START_DATE` only (CI override), no symbol cap. | `features/financial.parquet` + `.digest.json` |
| Temporal | `04_model_based_features` | Ch9 | Walk-forward GJR-GARCH + particle-filtered SV on refit schedules (no `fold` column). | `features/model_based.parquet` + JSON sidecar |
| Evaluation | `05_evaluation` | Ch7-9 | Per-date Spearman IC screen, overlap-corrected SE, BH-FDR, PROCEED/STOP/REVISE triage, redundancy section. | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` |
| Linear | `06_linear` | Ch11 | Ridge / Lasso / ENet / OLS per label from `config/training/{label}.yaml`. | Training runs + prediction sets in `run_log/registry.db`; coefficients `run_log/training/{hash}/`; scores `run_log/predictions/{hash}/` |
| GBM | `07_gbm` | Ch12 | LightGBM regression and classification; 15 configs x 10 checkpoints. | Boosters, `learning_curves.parquet`, `feature_importance.parquet` under `run_log/training/{hash}/`; own artifact writer, no `fold_metrics.parquet` |
| Tabular DL | `08_tabular_dl` | Ch12 | TabM rank-1 adapter MLP; every epoch checkpoint a candidate. | Checkpoints under `run_log/training/tabular_dl/` |
| Deep Learning | `09_deep_learning` | Ch13 | Index notebook: declares the 3-model sequence population, fits NLinear member. | Reads registry; NLinear checkpoint rows in the population |
| LSTM | `09a_lstm` | Ch13 | Fits the declared LSTM member (60-session lookback). CUDA required (~12 min RTX 3090). | Checkpoint predictions in the sequence population; `run_log/training/deep_learning/` |
| PatchTST | `09b_patchtst` | Ch13 | Fits PatchTST member; verifies the NLinear + LSTM + PatchTST population is complete (~67 min RTX 3090). | Same layout; population becomes readable downstream |
| Causal DML | `10_causal_dml` | Ch15 | Walk-forward DML effect of `vrp_21d` on `ret_to_expiry`; HAC SE; block-permuted placebo on the t-statistic. | One `CausalResult` row in registry `causal_runs`; retires `d034b82943c5` via `SUPERSEDES_CAUSAL` |
| Model Analysis | `11_model_analysis` | -- | Completeness audit per eligibility contract; one shared CV identity; IC + causal tables side by side. Selects nothing. | Nothing (reads registry) |
| Backtest | `12_backtest` | Ch16 | HTM cohort engine, equal-weight top-k over `top_k_grid`, liquid universe only; decision artifact per request; named baseline population. | `daily_returns.parquet`, `weights.parquet`, `spec.json` under `run_log/backtest/{hash}/` (no trade/fill ledger) |
| Portfolio | `13_portfolio_management` | Ch17 | Freezes baseline candidate set; advances top-10 distinct configs; varies only the allocator. | One backtest run per allocator; allocation population + strategy candidate set |
| Risk | `14_risk_management` | Ch19 | Proves the HTM path refuses a target-weight overlay; store unchanged before/after. | Nothing |
| Costs | `15_costs` | Ch18 | HTM cost cascade: one representative per family x {full, liquid} x `cost_fractions` at `top_k: 20`. Diagnostic only. | `evaluation/htm_cost_sensitivity.parquet`; one run per cost cell (`daily_returns.parquet`, `spec.json`; no weights); population `sp500-options-cost-sensitivity-validation-v1` |
| Holdout Predictions | `16_holdout_predictions` | Ch20 | Refits the selected config (via `resolve_solvent_carrier`) with training ending `labels.buffer` before 2021 under its own training identity. | One training run + one `split='holdout'` prediction set |
| Holdout Backtest | `17_holdout_backtest` | Ch20 | One backtest with every knob fixed from earlier stages; idempotent re-run; different config refused. | One run at `stage='holdout'`, decision artifact, population `sp500_options-holdout-ret_to_expiry` |
| Strategy Analysis | `18_strategy_analysis` | Ch20 §20.1 | DSR over the 12 grid, 95% block-bootstrap CIs, paired holdout closure, fold stability, cost-cascade reading, FF5+MOM regression, universal gates. | `results/strategy_assessment.json`; derived tables `backtest_paired_metrics`, `cohort_metrics` (replaced, not appended); tear sheet skipped (no `trades.parquet`) |
| Appendix | `90_ic_diagnostic` | -- | Preview-only mechanism study on `fwd_ret_dh_10d`: label-specific folds, feature ablation, lag decay, return decomposition, training-only dimensionality. | Nothing (no registry rows, no population, no backtest) |

Stage-ordering rule for the backtest block: 12 -> 13 -> 14 -> 15 -> 16 -> 17 -> 18. Costs (15) runs after 14 so "the last stage to select is the last stage to run"; nothing downstream of 15 may re-rank.

## Design decisions and why

| Decision | Choice | Why |
|---|---|---|
| Position | Short ATM straddle, HTM, daily delta hedge | O'Donovan & Yu's first cost mitigation: no exit-side option trade, so the round-trip spread becomes a one-sided entry cost; the option leg accrues to intrinsic at expiry; daily hedging captures the VRP. |
| Cost unit | % of premium, fraction of quoted half-spread | "Express a cost in the unit the position earns in." A 4% premium = 40 bps of notional, which an equity sweep reads as comfortably positive; bps-of-notional ranks the universe by volatility, not by cost to trade. |
| Cost crossings | Entry leg only (HTM); entry + exit + 2x commission for fixed-horizon labels | "Charge the crossings the strategy actually makes." Cash settlement is not a trade; the two cost lines sit an order of magnitude apart. |
| Engine | Dedicated `_htm_backtest.py::_run_htm_daily_mtm`; `_run_vectorized` for other labels | Shared `ml4t-backtest` assumes one position per symbol with continuous reallocation; it does not model overlap, paired option + hedge legs, or daily premium MTM. |
| Cohorts | Up to 5 concurrent weekly cohorts at 1/N_ROLL capital; weights normalized within cohort | Weekly entry at ~30-day DTE overlaps ~5 cohorts; portfolio P&L = weighted sum of per-cohort daily MTM. |
| Label | `ret_to_expiry` as the only strategy label; 5d/10d variants diagnostic | Vectorized path treats 5d/10d forward returns as daily returns (fwd_ret_10d Sharpe ~6.5, non-credible); dropped 2026-05-17. |
| Label construction | Look up the held contract in the raw chain at entry and exit | The "30-day ATM row" is a different contract each session; a shifted price column reports the change of contract as profit. |
| Buffer | 35 calendar days (`buffer_unit: calendar`) | Label horizon is calendar time (expiry); reading 35D as sessions purges ~7 weeks where 5 is needed (data loss). |
| Windows | Counted on the underlying's session grid with `min_observations_fraction: 0.8` | `shift(21)` on an intermittently listed instrument = 21 quotes, not 21 sessions; the two differ on nearly half the panel. |
| Null policy | `[vrp_21d, rv_21d]` only | Requiring a long-lookback column is a hidden liquidity screen that drops symbols quoted in bursts. |
| Model-based features | Refit schedule (`burnin`, `refit_every`) instead of fold-bounded fitting; no `fold` column | Fold-bounded fitting makes training rows non-causal while validation rows are causal, invisible in values. "A refit schedule, not a fold, should bound an estimate." |
| GJR (`o: 1`) | Declared deviation from shared GARCH(1,1)-Normal | A fall raises next-session variance more than a rise; index options are priced around that asymmetry. |
| SV `sigma_eta` | One market-level parameter estimated from a 10-symbol pool, refit quarterly (63) vs GARCH monthly (21) | It is a property of the market, not the asset; each refit costs a 4-chain MCMC per pool symbol. Rolling 252-session calibration because sampler cost is one latent state per observation. |
| Hand-written GARCH forward pass | Write the recursion from training-window quantities | Handing a fitted result back to `arch` to filter a longer series silently recomputes initialization and clipping bounds (leak with no symptom). |
| Loss functions (GBM) | mse / mae / huber compared at fixed features and folds | Short straddle is capped above, unbounded below; squared error is dominated by the left tail (a -9000% loss weighs ~200,000x a -20% loss) while IC treats them as adjacent. |
| Checkpoints | Publish every boosting iteration / epoch as its own candidate | A checkpoint is part of the configuration identity; keeping only the best is an unaccounted selection. 12 chooses and counts the 150 GBM candidates. |
| Device | `modeling.dl.device: cuda` pinned in config | GPU vs CPU accumulate in different order and reach different weights; the device is part of the training identity. |
| Selection criterion | Validation backtest Sharpe in `12_backtest` (equal-weight top-k, liquid) | A short straddle earns its premium whether or not the ranking was right; the spread can decide the outcome alone. IC in 05-11 is diagnostic only. |
| Populations | Declared by name before any member is fitted; `SUPERSEDES_*` for republish | A hash ties a request to the machine that produced it; a named immutable population cannot quietly gain or lose a member. |
| Equal weight in 12 | Model decides which symbols, never how much | Deliberate null; 13 varies only the allocator so any difference is attributable to one field. |
| Risk overlay | Refused by the execution path (14) | Cohort-normalized weights renormalize any scale-down; no cash leg to move into. Refuse, do not silently discard. |
| Cost sweep | Diagnostic population, never a candidate; runs after every selecting stage | Choosing a cost assumption after seeing results "has become a search". |
| Holdout | Refit with training ending one option cycle before 2021; carry the rank-1 config even with negative validation Sharpe | Scoring a validation-fitted model on the holdout is circular (it had already happened once here); picking by sign would be the selection the holdout exists to make honest. |
| Causal refutation | Compare placebo HAC t-statistics, not effects | Block-permuting the treatment inflates the second-stage denominator and narrows the placebo distribution (bias toward "passed"). |
| Results in prose | README restates no results | The registry is rebuilt on re-derivation; `18_strategy_analysis` resolves and prints the selected configuration. |

## Market-specific guardrails

Format: WHAT to avoid / WHY / HOW the case study handles it.

**Costs and engine**

| WHAT | WHY | HOW |
|---|---|---|
| Equity bps-of-notional cost model on options | Understates option spread cost by 1-2 orders of magnitude; ranks universe by volatility | `estimate_pct_of_premium`; 01 B.3/B.5 measure round-trip vs premium; sweep `cost_fractions` |
| Charging two spread crossings on an HTM position | Cash settlement needs no closing trade; cost lines differ by an order of magnitude | HTM engine charges entry-leg option half-spread only; fixed-horizon labels would need entry + exit half-spread + 2x commission |
| Shared vectorized backtester on `ret_to_expiry` | Assumes one position per symbol, no overlap, no paired legs, no premium MTM | Dispatch through `_run_htm_daily_mtm`; 5 cohorts at 1/5 capital |
| 5d/10d labels on the vectorized path | Treats forward returns as daily returns; Sharpe ~6.5 | `variants: []`; diagnostics only (rationale recorded in a `setup.yaml` comment under `labels` and `top_k_grid`; the issue note it cites is not shipped) |
| `total_slippage` NULL read as "no costs charged" | Vectorized path records no cost totals (NULL on all 879 rows, verified 2026-09-12) | Per-leg costs are applied inside the HTM engine; never infer cost from that column |
| Vectorized/equity-style Sharpes as anchors | Cannot anchor an HTM run | Only HTM-dispatched `ret_to_expiry` backtests enter registry comparisons |
| Cost curve read as pure entry cost | Contracts whose chain ends early are bought back at last ask and pay exit spread + commission under the same `option_spread_fraction` | State this when reading the grid |
| Spread fractions mistaken for achieved fills | They are assumptions calibrated from published execution studies | The curve says what the result would be under each assumption, not which holds |
| Margin opportunity cost omitted | 15-20% of notional tied up at ~5%/yr, not in per-leg accounting | Documented in `costs.components.margin_opportunity_cost`; `cost_dominance_note` |
| Mid-to-mid label pricing | Not what a trader receives | Entry half-spread and commission swept in 15; hedge rebalanced at close, charged nothing in the label itself |
| Feasibility moves are unhedged marks (01 B.3) | Daily rehedging removes part of what a seller collects | 02 builds the hedged outcome; feasibility cost comparison is an upper bound |
| Changing `execution.initial_cash` / `share_type` | Spec-hash inputs; invalidates every backtest hash | Single source of truth in `setup.yaml` via `get_backtest_config()`; never declare local constants |

**Labels, folds and holdout**

| WHAT | WHY | HOW |
|---|---|---|
| Label by shifting the "30-day ATM straddle" column | That row is a different contract most days | Look up the held contract in the raw chain at entry and exit (`_label_artifacts.py`); say that `instr_ret_*` is not a return anyone held |
| Holdout boundary by signal date, or by "decisions at or before `holdout_start − h − 1` sessions" | `ret_to_expiry` settles on each contract's own expiration, a calendar date that varies by contract, so the fixed-horizon shortcut is not equivalent to the settlement rule; a straddle sold in early December expires in January | Assign every row by its label END date: 02 `END_OF[ret_to_expiry] = expiration`, `_label_end < HOLDOUT_START`; the development seal is the last session before `holdout_start − 35 calendar days` (the longest a trade stays open), computed by hand in 01 and asserted equal to the last fold's `val_end`; only the fixed-horizon diagnostic variants use a session-grid `_end_{h}d` |
| Boundary declared on a non-session | `holdout_start: '2021-01-01'` is New Year's Day, not an NYSE session | The repo survives it because every boundary test is strict (`_label_end < HOLDOUT_START`, `timestamp < cutoff`; the sessions path uses `searchsorted(boundary, side="left")`), so 2021-01-04 is the first holdout session; snap your own boundaries to sessions so a `<` vs `<=` convention cannot move a row |
| Holdout scored before its last cohort settles | A cohort entered in late December 2021 would expire in January 2022, after `holdout_end` and past the 2017-2021 raw panel; a position that is simply cut off is neither settled nor valued | `_htm_backtest.py` loads quotes from `entry_min.year` through `exp_max.year`, follows every cohort to cash settlement at intrinsic or to a buy-back at the last quoted ask when its quote history ends early, and refuses to publish a run in which any contract ends with neither ("option lifecycle does not end every selected contract"), so a contract with no quotes to its expiration is refused, not cut off; the holdout is therefore scored once, after the last holdout decision plus the 35D horizon cap, under an unresolved-position rule fixed in code rather than improvised (scoring-date reading is inference from the engine; 16/17 do not print it) |
| Purge buffer unit mismatch | 35 calendar days read as 35 sessions purges ~7 weeks (data loss) | `buffer_unit: calendar` declared explicitly |
| Overlapping holding windows inflate evidence | Consecutive positions share most of a month-long life; plain SE too narrow (factor 4.33) | Effective count (02); overlap-corrected SE with lags from `hold_sessions` (04 F, 05); then FDR |
| Unlabelled rows left unexplained | Silent drops hide data defects | Reconcile every unlabelled row to exactly one cause: missing exit quote, never-quoted premium, hole in hedge path, expiry past panel end |
| Two folds with 2020 as one | Half the validation evidence is one vol event | Read magnitudes as a two-fold average; claim only patterns consistent across the grid with a mechanism |
| Summaries computed over holdout rows | A number computed over holdout is a number read off it | Filter value-derived statistics (mean, spread, median, null-count) to the development period; show only shape (row/symbol/date counts) whole |
| Holdout scored with the validation-fitted model | Circular; already happened once here | Refit with training ending `labels.buffer` before the window; assert training identity differs; raise otherwise |
| Label resolution leaking into holdout | `ret_to_expiry` resolves at expiry, not a fixed horizon | Buffer is a full option cycle (`labels.buffer: 35D`) |
| Holdout re-used for selection | One year of weekly cohorts cannot separate decay from an ordinary year | 2021 reported once; different configuration refused in 16/17; second looks go through the registry lifecycle |
| Picking the holdout configuration by its sign | Rank-1 has negative validation Sharpe | Carry it anyway; the holdout checks whether the reading holds |
| Ambiguous holdout prediction sets | Registry has held holdout sets belonging to no refit | Derive from the re-derived training identity + selected checkpoint; never search |

**Features**

| WHAT | WHY | HOW |
|---|---|---|
| Windows counted on straddle rows | `shift(21)` = 21 quotes, not 21 sessions, on nearly half the panel | Reindex onto the underlying session grid; `min_observations_fraction: 0.8`; 03 C.4 reports coverage under both rules |
| Null policy as a hidden liquidity screen | Requiring `iv_mom_10d` drops every symbol quoted in short bursts | `null_policy: [vrp_21d, rv_21d]` only; long-lookback columns null where unsupported; 03 prints what stricter policies cost |
| Min-observation rule trades comparability for coverage | A 252-session z-score from 202 quotes is not the same statistic as one from 252 | 0.8 is the declared bound; 03 C.4 prints coverage under this rule vs every-session |
| Returns spanning a security-identity change | A corporate action reads as a move; volume z-score restarts (volume unadjusted for splits) | Compute rv/returns per security-identity segment |
| IV solver non-convergence treated as signal | An unconverged IV level is a solver artefact | `qc_*` flags carried as negative controls; a control that predicts means contamination |
| Thin-date cross-sectional percentiles | An ordering of a few dozen names moves for reasons the level did not | Count universe on decision dates (77-469 names; < 100 on 4 of 209 dates, 2020-03-06 to 2020-04-09) |
| Feasibility IV series not fixed-tenor (01 B.4) | Nearest-the-money within a maturity bucket; a change of expiration moves the series | Read persistence as indicative; contract-indexed labels build the hedged outcome |
| Term structure and wings | Only one 30-day ATM straddle per symbol-session is materialized | Unobservable; `iv_skew_atm` is the call-put IV residual at one strike, not a smile |
| Counting a monotone transform as a second finding | Univariate rank screen cannot distinguish a feature from its transform | Redundancy section (05 S5); defer keep decision to multivariate modelling |
| Univariate screen read as model input | Blind to interactions and costs | Modelling notebooks take the whole panel; Ch16-18 measure after-cost returns; the screen is a reading order, not an input to 06 |
| Holdout-touching transforms in the panel | Any transform fitted across the sample leaks | Rebuild the panel with later dates withheld and compare every value; equality is the pass condition (03) |

**Model-based features**

| WHAT | WHY | HOW |
|---|---|---|
| Parameters estimated from the row's own future | Training rows non-causal, validation rows causal, indistinguishable in values | Refit schedule (`burnin: 252`, `refit_every: 21`/63) bounds every row; one value per (symbol, session); no `fold` column; staircase figure as the check |
| Library forward filtering on a longer series | `arch` silently recomputes initialization and clipping | Write the GARCH recursion out by hand from training-window quantities |
| Unconverged MCMC draws | Look identical to converged ones | Gate on R-hat, bulk ESS, tail ESS, divergence count; retry longer; fail the run rather than lower the bar |
| Vol-model premium columns read as current | GARCH/SV see only underlying returns; they respond after the underlying moves | Treat `forecast - IV` columns as lagging; null where no straddle quote; the model must handle that |
| Refit cadence assumed useful | Refits may buy nothing | Refit-jump diagnostic: jump across a refit session vs ordinary one-day move within a block |

**Evaluation, multiple testing and modelling**

| WHAT | WHY | HOW |
|---|---|---|
| Multiple testing across ~50 candidates | Plain threshold passes 13 of 49, overlap-corrected 3, FDR none | BH at `FDR_ALPHA = 0.05`; state searched set and label before reading a p-value; label diagnostics as such so the set does not grow silently |
| Squared-error loss on a capped-above, unbounded-below target | Tail observations dominate; IC gives nothing back | Compare mae/huber at fixed features/folds; winsorizing/ranking/robust loss only as explicit, separately-declared experiments |
| Selecting on IC | Ranking can be right and the strategy ruined by tail or spread; IC and Sharpe orderings routinely differ | Select on validation backtest Sharpe in 12; IC in 05-11 decides nothing |
| Keeping only the best epoch/iteration | Unaccounted selection | Publish every checkpoint; let 12 select and count all candidates |
| Device discovered from hardware | Same notebook gets different identities on different machines | `modeling.dl.device: cuda` pinned; CPU runs need `DEVICE="cpu"` and a new `POPULATION_NAME`; the mix is refused, not warned |
| CPU run substituted for a skipped GPU model | Silently fits a different population under the published name | Retain the stored artifact or document the skip |
| Notebook restating the label list | Diverges silently when `setup.yaml` changes | `declared_labels` = intersection of `setup.yaml` labels and `config/training/{label}.yaml` menus |
| Fold-pooled IC without interval (08) | No uncertainty | 11 compares predictions with uncertainty |
| Post-hoc population membership | A family added or dropped after results biases every comparison | Declare the population in one notebook (09) before fitting; members cannot extend it; missing members raise |
| Resolving by hash | Ties a run to the machine; cannot be declared ahead | Resolve populations and candidate sets by name; previews select by label |
| Partial checkpoint predictions | A model scored on a subset looks better for non-model reasons | Completeness check against each registered eligibility contract; partial = refused |
| Mixing eligibility groups | Sequence models (60 sessions) score fewer rows and symbols | 11 audits one row per eligibility group; read diagnostics within a group only |
| Comparing across CV identities | Different folds saw different training data | All four populations must share one CV identity before any table is read |
| Reading causal and predictive tables as confirming each other | They answer different questions | Agreement is not evidence; disagreement is not a defect |
| Placebo refutation on effect size | Permuted treatment frees it from controls; residual variance (the denominator) inflates; placebo distribution too narrow (old p = 0.0099 for t = 1.17) | Compare the HAC t-statistic; a p at the 1/100 floor for a weak effect is the signature |
| Placebo blocks shorter than the treatment window | Destroys serial dependence the refutation must preserve | `causal.treatment_window: 21` declared |
| Two canonical causal identities | `CausalResult.one` cannot resolve | Name the retired identity in `SUPERSEDES_CAUSAL` |
| Preview results leaking into canonical artifacts | Reduced runs cannot authorize a selection | Previews freeze nothing, select by label, never enter populations; 90 writes no registry rows |

**Backtest, portfolio, risk and analysis**

| WHAT | WHY | HOW |
|---|---|---|
| Registry drift in a comparison | A stray re-run adds a member, a failed one drops one | Immutable named populations fixed before the first backtest; `SUPERSEDES_*` for intentional republish |
| Checkpoint sweeps crowding a shortlist | One model occupies every slot | Count distinct model configurations, not rows (`top_n_cap`) |
| Confounded allocator comparison | Moving concentration or universe at the same time | Vary exactly one field per stage; pair against the equal-weight baseline on common support |
| Covariance allocators sized on the wrong instrument | They read underlying equity returns, not straddle returns | Known limitation; read allocator results accordingly |
| Risk overlay silently discarded | Cohort-normalized weights renormalize any scale-down; no cash leg | Execution path refuses a risk block; 14 proves store unchanged |
| Universe thins and spread widens in the same weeks (2020-03 to 2020-04) | Those are the weeks a short-vol book is most exposed | Ch19 measures a run of large moves against premium collected; neither feasibility figure separates stopped-quoting from failed-quality |
| Cost sweep entering selection | Choosing a favourable cost assumption after the fact is a search | Cost variants never join the candidate set; 15 runs after every selecting stage |
| Optimistic maximum from a large sweep | Rank-1 of a grid is biased upward | DSR deflated over the full declared `12_backtest` grid size K; under-deflation (K from a smaller stage) is the failure the check catches |
| Point-estimate ordering read as persistence | No interval attached | Paired block-bootstrap CI and p-value before stating decay or benchmark superiority |
| Stale derived tables | A moved selected configuration leaves predecessor rows looking current | `backtest_paired_metrics` and `cohort_metrics` replaced, not appended |
| Prose naming the selected configuration | Changes whenever the sweep is rebuilt | 18 resolves and prints it from the registry; only the selection contract is fixed in text |
| Directional leakage in the hedge | 0.10 net-delta band with daily rehedge leaks some direction | FF5+MOM regression (low R^2) shows what survives; read alpha with block-bootstrap CI |
| A stage that produces nothing claims so by narration | Unverifiable | Prove against registry contents before and after (14) |
| "Winner / champion / verdict" language | The signal is statistically null | "Holdout decay reading" only |

## Results and lessons

All numbers are from the development period or validation unless stated; the registry is rebuilt on re-derivation, so treat them as the published-bundle reading, not constants. Holdout, DSR, gate and paired-bootstrap values are printed at run time by 18 and were not in the digest.

**Feasibility (01)**
- 605 stocks quoted; 599 on at least one decision date; 77-469 names carry a straddle on a decision date; count < 100 on 4 of 209 dates (2020-03-06 to 2020-04-09).
- Cheapest fifth crosses at <= 0.0887 of premium. One crossing = 4.44% of premium at the cheapest fifth (inside the assumed 2-5%) vs 5.92% at the median stock (above it).
- Within-stock VRP persists a week later (supports weekly cadence); cross-stock ordering stability deferred to Ch7.
- 2 folds; last validation ends at the calendar outcome seal (Setup contract, CV row); the 2020-11-10 typed in 01's results prose is the sessions-counted date.
- Go/no-go: a typical premium move must exceed the one-crossing cost; a decision date must carry >= 5 x `top_k` names (100 for top_k 20) in the cheapest fifth; walk-forward split must stop one label horizon before holdout. Three things that would send the design back: cost failure (01 B.5, Ch18), no ranking-return relationship (Ch7), tail losses exceeding premium (Ch19).

**Labels (02)**
- The one-signal VRP baseline points the opposite way to the hypothesis on this universe (sign, not size, is the finding).
- Overlap-corrected SE turns a decisive-looking t-stat into one not separable from zero.

**Features (03) and model-based (04)**
- Session-grid vs row-count windows differ on nearly half the panel.
- Panel has 1,238 sessions; median symbol 373, 10th percentile 71 (may be sample-specific). Coverage (segments long enough for the 252 burn-in) is the binding constraint; the optimizer converges on almost every block.
- Median GARCH persistence rises and its spread narrows as the expanding window grows; refit jumps are visible per session instead of hidden across folds.

**Screen (05)**
- 49 of 51 candidates reach the daily measurement (2 `qc_*` flags constant); 13 clear a plain threshold, 3 after overlap correction (factor 4.33), 0 after family-wide FDR.
- Largest mean daily rank IC 0.0349 on the 21-session underlying return, adjusted p 0.263.

**Models (06-09, 11)**

| Family | Outcome |
|---|---|
| Linear (06) | No configuration clears zero IC; strongest Ridge shrinkage comes closest (curve rises toward zero as penalty grows = no ranking to defend). Target shape, not features, explains it. Menu champions by `fwd_ret_dh_10d` IC: ridge_a1000.0, ridge_a10000.0, ridge_a100.0. |
| GBM (07) | Every squared-error config ends below zero and below every Huber and MAE config (bottom five). Huber and MAE interleave; leading Huber and MAE tie; two Huber just below zero, four of five MAE above zero. Capacity orders nothing. 10 of 15 configs peak at checkpoint 1 or 2 (first fifth of iterations) then drift. Across-config IC spread at fixed length ~3x the within-config spread across checkpoints. Linear squared-error lands near the GBM squared-error block. Menu champions: leaves_31_mae, leaves_7_huber, leaves_7_mae, leaves_15_mae, default_mae. |
| Tabular DL (08) | TabM sizes tabm_s, tabm_m, tabm_l (menu order tabm_s, tabm_l, tabm_m); fold-pooled IC shown without interval. |
| Sequence (09, 09a, 09b) | NLinear, lstm_h64, patchtst (menu order patchtst, lstm_h64, nlinear). Not expected to win: far more freedom to fit noise on a few hundred names; losing to GBM is a result about the data. |
| Validation rank-1 (11/18) | Daily-pooled IC essentially zero (HAC CI straddles zero); Sharpe CI straddles zero spanning roughly +-1; the two pictures agree. |

**Causal DML (10)**: N = 166,105; VRP effect on `ret_to_expiry` = 0.4098; HAC SE = 0.3509; t = 1.17; p = 0.243 under either statistic. The old effect-based refutation p = 0.0099 (floor of 100 draws) was a spurious pass; the t-statistic refutation was adopted. Retired identity `d034b82943c5` is the same fit as its replacement.

**Backtest, allocation, costs (12-15)**
- Three of the model families produce zero positive-Sharpe baseline rows on `ret_to_expiry`.
- The liquid-universe pin excludes higher-Sharpe full-surface configurations from selection by construction.
- Cost story (config comment): full universe -> uneconomic; liquid quintile -> marginal-but-positive after costs; "Total costs often exceed the expected VRP edge."
- 879 registered backtest rows carry `total_slippage` NULL because the vectorized path records no cost totals, not because nothing was charged.
- Allocator comparison in 13 reads underlying equity covariance (`allocator_lookback: 63`), so option exposure is sized only indirectly; no allocator is reported as winning in the notes.
- Risk overlay (14): refused by the HTM path; nothing registered.

**Holdout and closure (16-18)**
- Selected (cross-stage rank-1 after the liquid pin) configuration has negative validation Sharpe; carried into the holdout unchanged.
- Fold stability: two fold Sharpes straddle zero, both inside the wide bootstrap CI; heavy tails and negative skew (long stretches of small premium decay punctuated by large losses).
- Factor regression FF5+MOM: low R^2; the 0.10 delta band leaks some directional exposure.
- Holdout (2021): paired val -> holdout decay and holdout-vs-EW reported with CI95 and p; one year of weekly cohorts cannot separate decay from an ordinary year.
- Gates: `gate1_validation_sharpe_geq_zero` (validation Sharpe CI lower bound >= 0) and `gate2_holdout_diff_not_excludes_zero_negatively`; pass/fail serialized, not promoted to a conclusion.
- Overall: statistically null signal under HTM full per-leg cost accounting. Lessons: (1) cost unit must match the return unit; (2) IC and squared-error loss measure nearly disjoint things on a long-tailed capped target; (3) overlap and FDR corrections remove every screened feature; (4) report null results with intervals and carry the honest rank-1 into the holdout.

## How to adapt this pattern to a new dataset

Use this checklist for any derivative or premium-denominated strategy (options, variance swaps, any held-to-settlement contract), and for any long-tailed target judged on a rank metric.

1. **Declare the setup once.** Put universe, cadence, hedge protocol, costs, folds, buffer, feature windows and device in one `setup.yaml`; read through `get_backtest_config()` and bind in notebooks; never redeclare `INITIAL_CASH`, `share_type` or the label list locally.
2. **Express costs in the unit the position earns in.** For options: spread as % of premium / fraction of quoted half-spread (`estimate_pct_of_premium`, `cost_fractions`); separate hedge-leg cost in bps of notional with `hedges_per_holding`; per-contract and per-share commissions; document margin opportunity cost even if not charged.
3. **Charge only the crossings the strategy makes.** HTM = entry leg only (plus forced buy-back when a chain ends early); fixed-horizon exits = entry + exit half-spread + 2x commission. Never let a vectorized path treat multi-day forward returns as daily returns.
4. **Build labels as a round trip in one contract.** Price the held contract at entry and exit from the raw chain; build the hedge path from underlying closes; reconcile every unlabelled row to one cause; exclude holdout by label END (settlement) date, never by decision date minus a fixed horizon when the horizon varies; declare `buffer_unit: calendar` when the horizon is calendar time.
5. **Count windows on the session grid**, not on instrument rows; set `min_observations_fraction` (0.8) and report coverage under both rules; compute returns per security-identity segment; keep `null_policy` to the thesis columns only.
6. **Bound fitted features by a refit schedule** (`burnin`, `refit_every`, expanding vs rolling), not by fold; write the forward recursion by hand; gate MCMC on R-hat / ESS / divergences; share market-level parameters from a pool.
7. **Screen with overlap and FDR corrections.** Per-date Spearman IC; overlap-corrected SE with lags = holding period; `COVERAGE_MIN = 0.70`, `STALENESS_MAX = 0.50`, `FDR_ALPHA = 0.05`, `SIGN_CONSISTENCY_MIN = 0.60`, `IC_THRESHOLD = 0.01`; record PROCEED/STOP/REVISE route in the ledger `note`; read decay out to 2x `hold_sessions`; carry `qc_*` negative controls; run the holdout-invariance recompute.
8. **Match the loss to the payoff.** For capped-above, unbounded-below targets compare mse/mae/huber at fixed features and folds; declare any winsorizing or ranking as a separate experiment; use `max_bin: 255` on CPU LightGBM.
9. **Publish every checkpoint and pin the device.** Checkpoint and device are part of the training identity; CPU fits go under their own `POPULATION_NAME`; never substitute a CPU run for a skipped GPU model.
10. **Declare populations by name before fitting** (09 for sequence models; 12 for baselines; 13 for candidates; 15 for costs); refuse incomplete or mixed-device populations; audit completeness per eligibility contract and a single CV identity before reading a table.
11. **Write a dedicated engine when the shared one cannot represent the position** (overlapping cohorts, paired legs, daily premium MTM). Publish a decision artifact (exact contracts on exact dates) before accounting; audit `sessions_held` against the validation calendar.
12. **Vary one field per stage**: concentration (`top_k_grid`) in 12 with equal weight; allocator only in 13 on the top-N distinct configurations; risk in 14 (prove the store unchanged if the overlay cannot apply); costs in 15 after every selecting stage, never as a candidate.
13. **Select on validation backtest Sharpe**, after a declared universe pin, never on IC; carry the rank-1 into the holdout even if negative; refit with training ending `labels.buffer` before the window and assert the training identity changed; score the holdout once, after the last holdout decision plus the horizon cap, with the rule for positions that cannot settle inside the window (follow to settlement, buy back at last ask, or refuse the run) written before the run.
14. **Close with intervals**: DSR with K = the full selection grid; 95% stationary block-bootstrap CI on every metric; paired bootstrap for decay and benchmark comparison; FF5+MOM alpha with `block_size=20`; serialize gates; replace derived tables; no winner language for a null.
15. **Causal check**: declare `treatment_window` from the construction window; refute on the HAC t-statistic with block-permuted treatment (100 draws); treat agreement with predictive tables as not evidence.

**Key APIs and patterns**
- Loaders: `load_sp500_options_straddles()`, `load_sp500_options_straddles_raw()`, `load_sp500_daily_bars()`, `load_backtest_prices_for(CASE_STUDY, PRIMARY_LABEL, split="validation")`.
- Config: `backtest_loaders.get_backtest_config`, `case_studies.utils.sweep_config.{get_top_k_values_for, get_universe_filters_for}`, `get_top_n_predictions(case_study, stage)`, `top_n_cap(n)`, `get_htm_cost_cascade(case_study)`, `utils.artifact_specs.{resolve_label_buffer_unit, resolve_label_buffer, resolve_label_horizon}`, `research_workflow.declared_dl_device`, `sequence_identity_params`.
- Backtest prep: `option_decision_dates(study, prediction_hashes, prices=, signal={"universe_filter": "liquid"})`, `option_trade_calendar(decision_dates)`, `strategy_request_frame(rows)`, `preview_prediction_candidates(...)`.
- Request row (12):
  ```python
  {"request_name": f"{h}-top{k}-liquid", "prediction_hash": h, "label": label, "top_k": k,
   "signal": {"method": "equal_weight_top_k", "top_k": k, "direction": "long_only",
              "long_short": False, "universe_filter": "liquid"},
   "allocation": None, "risk": None, "costs": None, "chapter": "ch16"}
  ```
  Cost request (15): `signal["option_spread_fraction"] = fraction`; `signal["universe_filter"] = "liquid"` or `signal.pop("universe_filter")` for `full`; `request_name = f"{family}-{universe}-spread-{fraction:g}"`.
- Private population (08):
  ```python
  study = open_study("sp500_options", workspace="~/ml4t-experiments")
  configs = load_model_configs(study, "tabular_dl", config_names=["tabm_s"])
  requests = model_requests(study, configs, overrides={"device": "cuda"})
  resolved = tuple(request.resolve() for request in requests)
  execution, population = run_model_population(study, resolved, population_name="my-tabm-v1")
  ```
  Add presets as new files under `case_studies/config/{model_type}/` (e.g. `tabm/tabm_xl.yaml`); editing an existing preset changes identity and registers a new row beside the old.
- Supersession constants (all `str = ""`, passed as `declared=... or None`): `SUPERSEDES_BASELINE_POPULATION`, `SUPERSEDES_BASELINE_CANDIDATES`, `SUPERSEDES_ALLOCATION_POPULATION`, `SUPERSEDES_STRATEGY_CANDIDATES`, `SUPERSEDES_COST_POPULATION`, `SUPERSEDES_CAUSAL`.
- Candidate/causal/holdout: `baseline_candidates.best_validation_sharpe()`, `CausalResult.one(label)`, `resolve_solvent_carrier(...)` (shared by 15/16/17; "solvent" = equity never reached zero, inference).
- Analysis (18): `compute_bootstrap_ci(..., block_size=20)`, `ci_status(lo, hi)`, `gate1_validation_sharpe_geq_zero`, `gate2_holdout_diff_not_excludes_zero_negatively`, `gate_passes`, `fmt_gate`; tables `backtest_paired_metrics`, `cohort_metrics`.
- Speed knobs: 02 `START_DATE`, `MAX_SYMBOLS` (both unset; thinning degrades per-session dispersion and rank stats); 03 `START_DATE` only; 15 `PREVIEW_COST_FRACTIONS`, `PREVIEW_UNIVERSES = ("liquid",)`; DL `DEVICE="cpu"` + `POPULATION_NAME`.
- Registry: `run_log/registry.db` (training runs, prediction sets, backtests, `causal_runs`); artifacts under `run_log/training/{hash}/`, `run_log/predictions/{hash}/`, `run_log/backtest/{hash}/`; populations resolved by name (e.g. `sp500_options-holdout-ret_to_expiry`, `sp500-options-cost-sensitivity-validation-v1`).

**Sequenced reproduction checklist (09a onward; go/no-go at each step)**
1. Confirm 09 declared the sequence population; run 09a then 09b; no-go if 09b's three-family completeness check fails.
2. Run 10 on the full pre-holdout population; set `SUPERSEDES_CAUSAL` to the retired hash when republishing.
3. Run 11; no-go if any population is incomplete or the four CV identities differ; read IC within eligibility group only.
4. Run 12; no-go if `get_universe_filters_for` != `["liquid"]`, if no straddle resolves inside the validation calendar, or if any published row lacks a Sharpe.
5. Run 13; freeze the candidate set; advance `top_n_predictions.allocation` distinct configurations; no-go if a pair has no Sharpe on common support.
6. Run 14; verify the risk request set is empty and the store is unchanged.
7. Run 15; one representative per family (max validation Sharpe, ties by backtest identity) at the cascade `top_k`; results stay out of the candidate set.
8. Run 16; no-go if the holdout training hash equals the validation one.
9. Run 17; no-go if the configuration differs from the one 15/16 resolved.
10. Run 18; check DSR K equals the 12 grid size, every metric carries a CI, both gates serialized before writing the assessment.
11. 90 is optional and never gates anything.

**Reader uncertainty carried forward**: exact HTM engine mechanics (per-cohort MTM, N_ROLL derivation, `delta_threshold` trigger, `rebalance_step` interaction with allocators); the precise `ret_to_expiry` denominator; whether sizing is literally vega-scaled; the overlap-correction estimator (Newey-West vs block bootstrap) and exact lag count; Lasso/ENet `f` semantics (assumed fraction of alpha_max) and l1_ratio; SV MCMC target acceptance and retry settings; sequence-model hyperparameters and checkpoint schedule; DML nuisance model; allocator estimators in 13; whether `bootstrap_block_length`/`bootstrap_n` equal the 20-day factor block.

## Related references

- `chapters/06_strategy_definition.md` -- feasibility go/no-go, search accounting and run logging (01).
- `chapters/07_defining_the_learning_task.md` -- contract-indexed labels, overlap-corrected SE, univariate IC screen, BH-FDR (02, 05).
- `chapters/08_financial_features.md` -- VRP, IV surface and session-grid feature construction, search control (03).
- `chapters/09_model_based_features.md` -- GJR-GARCH and stochastic-vol refit schedules, hand-written forward pass (04).
- `chapters/11_ml_pipeline.md` -- Ridge/Lasso/ENet menus and binding declarations to data (06).
- `chapters/12_gradient_boosting.md` -- LightGBM loss comparison, checkpoints as candidates, TabM (07, 08).
- `chapters/13_dl_time_series.md` -- NLinear/LSTM/PatchTST sequence population, device identity (09, 09a, 09b).
- `chapters/15_causal_estimation.md` -- walk-forward DML, HAC t-statistic placebo refutation (10).
- `chapters/16_strategy_simulation.md` -- HTM cohort engine, decision artifacts, equal-weight null (12).
- `chapters/17_portfolio_construction.md` -- allocator comparison on frozen candidates (13).
- `chapters/18_transaction_costs.md` -- premium-denominated costs, O'Donovan & Yu cascade (15).
- `chapters/19_risk_management.md` -- why a target-weight overlay cannot act on a cohort book (14).
- `chapters/20_strategy_synthesis.md` -- holdout refit, DSR, paired bootstrap, gates, "cost model validity" theme (16-18).
- `chapters/03_market_microstructure.md` -- bid-ask, effective vs quoted spread, liquidity quintiles.
- `case_studies/sp500_equity_option_analytics.md` -- sibling study; options as side information; same GJR deviation.
- `case_studies/us_equities_panel.md` -- the standard equity bps-of-notional cost sweep this case study contrasts against.
- `case_studies/crypto_perps_funding.md` -- another carry/premium-harvesting book with cost dominance.
- `libraries/ml4t_backtest.md` -- the shared engine and why it is not used for `ret_to_expiry`.
- `libraries/ml4t_engineer.md` -- feature families and session-grid windows.
- `libraries/ml4t_diagnostic.md` -- IC, DSR, bootstrap CI tooling.
- `guardrails.md` -- cross-cutting leakage, overlap and multiple-testing rules.
- `decision_rules.md` -- selection on backtest Sharpe, population declaration, stage ordering.
- `evidence.md` -- cross-case-study results table.
- `workflow.md` -- stage sequence shared by the nine case studies.
- `companion_repo.md` -- repo layout, registry, artifact bundles.
- `glossary.md` -- shared definitions.
- Further reading:
  - O'Donovan, J. & Yu, J. (2024). *A Transaction Cost Perspective on Option Anomalies.*
  - Muravyev, D. & Pearson, N. (2020). Effective vs quoted option spreads ($0.026/$0.128 ATM ratio).
  - Heston, S. et al. (2023). "Algo" execution case and "spread < 10%" filter.
  - Fama-French 5 factors + momentum (factor regression in 18).

## Glossary

- **Straddle** -- one call and one put on the same underlying, strike and expiry; ATM when the strike is nearest spot.
- **Short straddle** -- selling both legs; earns at most the premium, loses without bound.
- **HTM (hold-to-maturity)** -- holding to expiry so there is no exit-side option trade; makes the spread a one-sided entry cost.
- **HTM daily-MTM cohort engine** -- `_run_htm_daily_mtm`; values legs daily, settles at intrinsic, hedges delta.
- **Cohort** -- straddles entered on one weekly decision date; up to 5 overlap at 1/N_ROLL capital each.
- **Daily MTM** -- per-cohort daily mark-to-market of premium plus hedge P&L, summed to portfolio returns.
- **Delta hedge / delta threshold** -- offset share exposure with the underlying at each close; rehedge when |net delta| breaches 0.1.
- **VRP (variance risk premium)** -- implied minus subsequently realized variance; here `vrp_21d = iv_atm - rv_21d`; the DML treatment.
- **Half-spread** -- half the bid-ask gap; cost of crossing once; `cost_fractions` scale how much is paid.
- **`option_spread_fraction`** -- share of the quoted option half-spread paid on entry (and on forced exit).
- **Liquid universe / quintile** -- bottom 20% of names by half-spread (as a share of premium) per rebalance date.
- **Cost cascade rungs** -- rung 2 = full universe; rung 3 = liquid subset (O'Donovan & Yu framing).
- **`ret_to_expiry`** -- return of a short straddle from entry to its own contract's expiry, delta-hedged daily; the only strategy label.
- **`fwd_ret_{5,10}d`, `fwd_ret_dh_{5,10}d`** -- fixed-horizon raw and delta-hedged forward returns; diagnostic only.
- **Label buffer** -- purge separation between training and validation; 35 calendar days here.
- **Effective count** -- independent observations implied by overlapping holding windows.
- **IC** -- per-date Spearman rank correlation between a feature (or prediction) and the label, averaged over dates.
- **FDR / BH** -- Benjamini-Hochberg family-wide adjustment at alpha 0.05.
- **PROCEED / STOP / REVISE** -- triage decisions in `evaluation/triage_ledger.parquet`.
- **GJR-GARCH** -- GARCH(1,1) with extra weight (`o: 1`) on negative squared returns.
- **Stochastic volatility (SV)** -- log-variance follows an unobserved random walk; `sigma_eta` is its step sd.
- **Particle filter** -- carries ~1000 candidate latent-vol values forward, reweighting by each day's return.
- **Burn-in / refit schedule** -- sessions before a segment's first fit (252) and between refits (21 GARCH, 63 SV).
- **Checkpoint** -- model state at a boosting iteration or epoch; each is its own candidate.
- **TabM** -- MLP ensemble with shared trunk and per-member rank-1 scaling vectors plus own output heads.
- **Training identity** -- hash of everything that determines a fit, including device and checkpoint.
- **Population** -- named, immutable group of training runs / prediction sets / backtests fixed before execution.
- **Candidate set** -- hashed, frozen list of backtests a selection may range over.
- **Eligibility contract / group** -- registered rows a prediction set must cover exactly; populations sharing the same eligible rows (sequence vs cross-sectional).
- **`SUPERSEDES_*`** -- constant naming the retired identity when an artifact is republished.
- **Spec hash** -- content address of a backtest spec; cash and share_type are inputs.
- **`qc_*`** -- solver-convergence quality flags carried as negative controls.
- **Walk-forward DML** -- double ML run fold-by-fold for the causal effect of VRP on delta-hedged returns.
- **HAC** -- heteroskedasticity-and-autocorrelation-consistent covariance.
- **DSR** -- Deflated Sharpe Ratio corrected for K trials.
- **Paired bootstrap / `backtest_paired_metrics`** -- difference metrics with CI95 and p between two return series.
- **Gate 1 / Gate 2** -- validation-Sharpe CI lower bound >= 0; holdout-vs-EW paired CI not excluding zero negatively.
- **Preview tier** -- reduced run that writes nothing canonical.
- **Solvent carrier** -- the selected configuration resolved by `resolve_solvent_carrier` (inference: equity never reached zero).
