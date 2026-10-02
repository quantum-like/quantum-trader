# Case study: US firm characteristics panel (~2,500 stocks, monthly, fwd_ret_1m)

> Anonymized monthly firm characteristics released with Chen, Pelger and Zhu (2020) via NASDAQ Data Link: 46 provider rank-encoded characteristics plus 11 same-row composites/interactions (57 features), ~2,500 stocks, 1996–2016, a one-month forward-return label, equal-weight dollar-neutral long-short top-k books. It is the book's most feature-rich fundamental panel and the natural home of the latent-factor family (IPCA, CAE, SDF, SAE). Headline lesson: on a heavy-tailed monthly cross-section the **target** (raw vs winsorized vs median-split) and the **loss** (mse vs mae vs huber) move IC more than the feature set, the penalty strength or the model capacity, and a family counts as a signal only when its HAC interval excludes zero *and* its decile spread clears the 2 x [5, 20] bps round trip. A 12-month holdout cannot separate decay from an ordinary year, so every holdout reading carries an interval, and because the trial count is only known at the end of the selection funnel, Sharpe deflation happens once, in `17_strategy_analysis`. Repo path: `case_studies/us_firm_characteristics/`.

## Setup contract

Source of truth: `case_studies/us_firm_characteristics/config/setup.yaml` (`strategy_id: us_firm_characteristics`, `setup_version: v2`) plus `config/training/{fwd_ret_1m,fwd_ret_1m_win,fwd_class_1m}.yaml` (per-label model menus) and `config/cv_config.json` (fold boundaries written by `02_labels`). Every notebook reads periods, labels and holdout from `setup.yaml`; never retype a threshold.

| Item | Contract |
|---|---|
| Universe | `universe.inclusion_rule: complete_characteristic_case` (authors keep firm-months with all 46 characteristics); `identifiers: anonymous_split_scoped_firm_axis` (ids persist only within a released tensor block, no cross-block mapping); `n_assets: 2500` is display metadata, the backtest counts assets from data. README lists price > $5 and ADV > $1M, but the release carries no prices, volumes or share counts, so no liquidity screen is applied in-repo (inference: provider construction). |
| Data | 804,530 rows, 46 characteristics, provider cross-sectional rank in [-0.5, 0.5]; feature window `features.window.start: 1990-01-01`, `end: 2016-12-31` (release itself starts two decades earlier); development window 1996–2015, 312 month-ends. |
| Decision cadence | `decision.cadence: monthly_month_end`, `snapshot: month_end_close`, `execution_delay: next_bar_open`, `characteristic_availability: provider_standard_conventions`, `yearly_update: end_of_june` (annual accounting published end-June against December fiscal year end = 6-month lag), `monthly_update: month_end_for_next_month`. |
| Alignment | The release pairs characteristics observed at close of month t-1 with the return over month t and dates the row by month t; a row's return is already realised on its own timestamp. |
| Mapping | `mapping.class: long_short_top_k_rebalance`, `position_state_space: long_short`, `entry_logic: rank_top_k_long_bottom_k_short`, `sizing: equal_weight_within_leg`. |
| Execution | `execution.initial_cash: 1_000_000`, `share_type: integer`, `allocator_lookback: 12` (monthly bars). Read via `get_backtest_config()`; cash and share_type are spec-hash inputs, changing them invalidates every `backtest_hash`. |
| Costs | `costs.class: material`, `components: [spread, commission, market_impact, borrow_cost]`, `per_leg_cost_bps_range: [5, 20]`, `era_note`: pre-2001 spreads 15–30 bps, post-2001 (decimalization) 5–15 bps (recorded, not applied). Round trip = 2 x per leg, i.e. 10–40 bps per rebalance (inference from the printed string). Declared commission/slippage levels not visible in the notes. |
| CV | `evaluation.n_splits: 10`, `train_size: 10YE`, `val_size: 1YE`, `labels.buffer: 1M` between a training window and the validation window after it, `calendar: null` (monthly returns; calendar-aware splitting needs daily data), `periods_per_year: 12`. |
| Holdout | `holdout_start: '2016-01-01'`, `holdout_end: '2016-12-31'`; `rebalance_step: 1` gives 12 monthly decisions. Untouched until `15_holdout_predictions` / `16_holdout_backtest`. |
| Benchmark | Equal-weight universe over the holdout window (decay test 2 in `17`); `backtest.rebalance.benchmark` profile with `min_weight_change: 0.0`, `min_trade_value: 0.0`; signal floor = raw provider-ranked `r12_2` momentum IC (+0.04398); engine floor = random-ranking plumbing test. No named index benchmark appears in the notes. |
| Eligibility at run time | Breadth >= 2 x top_k on every rebalance date (100 firms at top_k=50); `get_entry_schemes_for` drops any declared k the cross-section cannot fill and raises if none is feasible; IC months need `min_obs` = half the median month's firm count (`02`) / `MIN_CROSS_SECTION = 30` (`04`). |

### Labels (`labels` block)

| Label | Definition | Buffer | Horizon | `rebalance_step` | Scored against |
|---|---|---|---|---|---|
| `fwd_ret_1m` (primary) | Total return over the month after the decision | 1M | 0D | 1 | Itself |
| `fwd_ret_1m_win` | Each month's cross-section clipped at its own 1st/99th percentiles | 1M | 0D | 1 | Itself |
| `fwd_class_1m` | Up/down at the month's own median | 1M | 0D | 1 | IC vs `fwd_ret_1m` (`classification_eval_label`); AUC/accuracy/log_loss vs the binary label |

`horizons` (0D) is what `generate_cv_splits` seals the last validation fold on; `buffer` (1M) is a separate, deliberately conservative train/validation gap. The HAC correction is told the label span (1M, `LABEL_SPAN_MONTHS = int(str(LABEL_BUFFER).rstrip("Mm"))`), not the horizon.

### Feature register (`features.families`; `lookback`/`lag` in months)

| Family | `pattern` | Role | Lookback | Lag |
|---|---|---|---|---|
| value | `BEME\|E2P\|CF2P\|D2P\|S2P\|A2ME` | signal | 12 | 6 |
| quality | `PROF\|ROE\|ROA\|OP\|PM\|PCM\|RNA` | signal | 12 | 6 |
| investment | `Investment\|NOA\|DPI2A\|NI\|OA\|AC` | signal | 24 | 6 |
| momentum | `r12_2\|r2_1\|r12_7\|r36_13\|ST_REV\|LT_Rev\|SUV\|Rel2High` | signal | 36 | 0 |
| risk | `Beta\|MktBeta\|IdioVol\|Resid_Var\|Variance\|Spread\|LTurnover\|LME` | state (LME behaves as a signal) | 12 | 0 |
| other | `Q\|C\|CF\|AT\|ATO\|CTO\|D2A\|FC2Y\|Lev\|OL\|SGA2S` | signal | 12 | 6 |
| composite accounting | `composite_value\|composite_quality\|composite_value_quality` (equal-weight mean of member ranks) | signal | 12 | 6 |
| composite investment | `composite_investment` | signal | 24 | 6 |
| composite momentum | `composite_momentum` (mean of r12_2, r12_7) | signal | 12 | 0 |
| interaction accounting | `interaction_value_x_quality\|interaction_value_x_roe` (BEME x PROF/ROE, product of centred ranks) | signal | 12 | 6 |
| composite mixed | `composite_value_momentum\|composite_quality_momentum` | signal | 18 | 0 |
| interaction mixed | `interaction_size_x_value` (LME x BEME) | signal | 18 | 0 |
| interaction momentum | `interaction_momentum_x_ivol` (r12_2 x IdioVol) | signal | 12 | 0 |

Mixed constructions span everything they read (lookback 18, lag 0): the accounting member was published six months earlier, the price member is current, so the composite is knowable at the decision timestamp. An earlier version gave them lag 6 and `plot_timing_contract` drew them as reading nothing from the last six months, which they do through the momentum member.

### Sweep and modeling blocks

| Key | Value |
|---|---|
| `backtest.rebalance.default` | `min_weight_change: 0.005`, `min_trade_value: 100.0` (at top_k=50 the EW weight is 2%, above the 0.5% threshold) |
| `backtest.sweep.top_n_predictions` | `signal: 0` (all), `allocation: 10`, `risk_overlay: 1`; cost sensitivity takes no entry (runs the carrier) |
| `backtest.sweep.top_k_grid` | `[5, 10, 20, 50]` for each of the three labels; `percentile_grid` / `quantile_grid` commented out |
| `backtest.sweep.allocators` | `{name: score_weighted}`, `{name: conformal_weighted}`; no max-weight cap; `expensive_allocators_skip: false` |
| `backtest.sweep.cost_grid_bps` | `[0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]` per leg |
| `backtest.sweep.risk_controls.position` | stop_loss {3, 5, 10, 15}%, trailing_stop {1, 2, 3, 5, 10, 15, 20}%, time_exit {10, 20, 40} bars (declared, not applicable here) |
| `modeling.gbm` | `libraries: [lightgbm]`, `preset: default`, `device: cpu`, `max_bin: 255`, `num_threads: 8` |
| `modeling.tabular_dl` | `device: cuda`, `num_threads: 8` (both part of the training identity) |
| `modeling.latent_factors` | `persistent_entities: true`, `device: cuda`, `num_threads: 8`, `deterministic_algorithms: true`; `macro_series: [dff, dgs1, dgs2, dgs3, dgs5, dgs7, dgs10, dgs20, dgs30, t10y2y, vixcls]`, `macro_availability_lag_days: 1`; `model_kwargs.ipca: {max_iter: 10000, factor_ridge: 0.01, gamma_ridge: 0.01}`; `model_kwargs.sdf: {checkpoint_epochs: [256, 512, 768, 1024], beta_checkpoint_epochs: [256], beta_default_checkpoint: 256}` |
| `causal` | `treatment: r12_2`, `treatment_window: 12`, `confounders: [Beta, IdioVol, LME, Variance]`, `method: walk_forward_dml` |

Per-label menu keys: `linear`, `gbm`, `tabular_dl`, `latent_factors` (every label declares `ipca, cae, sdf, sae`), `causal_dml` (regression labels only; ml4t/agent-workspace#396). Entries are preset names under `case_studies/config/{model_type}/` (e.g. `case_studies/config/cae/cae.yaml`); comment out a line to skip a config.

## Pipeline stages

Run from repo root: `uv run python case_studies/us_firm_characteristics/NN_*.py`; run 08a–08d before `08_latent_factors`. Fetch results without training: `uv run python scripts/download_artifacts.py --cs us_firm_characteristics` (`v3.1.0-artifacts`, current; `v3.0.0-artifacts` as first published; hashes do not cross-resolve).

| Stage | Notebook | Chapter | What it does | What it writes |
|---|---|---|---|---|
| Feasibility | `01_feasibility_analysis` | Ch6 §6.2–6.6 | Rank-encoding assertion, tenure distribution, breadth per rebalance date, exceedance curve of monthly moves vs round-trip band, within-firm rank persistence, fold fit vs `setup.yaml`. Fits no model. | Nothing |
| Labels | `02_labels` | Ch7 §7.2 | `fwd_ret_1m`, per-month winsorized variant, median-split class; alignment check via `ST_REV`; null-coverage assertion; uniqueness; baseline `r12_2` IC with HAC SE | `labels/prices.parquet`, `labels/fwd_ret_1m.parquet`, `labels/fwd_ret_1m_win.parquet`, `labels/fwd_class_1m.parquet`, `config/cv_config.json`, JSON sidecar per parquet |
| Features | `03_financial_features` | Ch8 §8.4 | 57 characteristics; same-row composites and interactions; timing contract; rebuild-without-holdout equality; 1:1 key join | `features/financial.parquet`, `features/feature_doc.json`, `financial.parquet.digest.json` |
| Evaluation | `04_evaluation` | Ch7 §7.3–7.4, Ch8 §8.6, Ch9 | Per-feature monthly Spearman IC, HAC SE, BH-FDR, per-fold sign consistency, redundancy clustering, triage | `evaluation/triage_ledger.parquet` (one row per characteristic with decision), `evaluation/ic_timeseries.parquet`, JSON digest sidecars |
| Linear | `05_linear` | Ch11 | OLS/Ridge/LASSO/ElasticNet/logistic, all three labels, per-fold fits | Training runs and prediction sets in `run_log/registry.db`; coefficients `run_log/training/{hash}/`; scores `run_log/predictions/{hash}/` |
| GBM | `06_gbm` | Ch12 §12.2 | LightGBM, objective x `num_leaves` grid, 10 checkpoints per run | Boosters, `learning_curves.parquet`, `fold_metrics.parquet` under `run_log/training/{hash}/`; prediction set per checkpoint |
| Tabular DL | `07_tabular_dl` | Ch12 §12.3 | TabM rank-1 adapter MLP ensemble, capacity grid | Checkpoints under `run_log/training/tabular_dl/`; prediction set per declared epoch |
| Latent index | `08_latent_factors` | Ch14 | Summarizes 08a–08d; checks every label's menu declares `ipca, cae, sdf, sae` | Nothing (reads registry) |
| IPCA | `08a_ipca` | Ch14 §14.5 | Instrumented PCA, characteristic-conditioned linear loadings, ALS to convergence | Training runs and prediction sets (converged map registered as one checkpoint) |
| CAE | `08b_conditional_autoencoder` | Ch14 §14.6 | Network map from characteristics to exposures; config `case_studies/config/cae/cae.yaml` | Training runs and prediction sets per epoch checkpoint |
| SDF | `08c_stochastic_discount_factor` | Ch14 §14.7 | Neural SDF, unconditional then conditional stage, adversarial moment network, macro context | Training runs and prediction sets at global epochs 256..1280 |
| SAE | `08d_supervised_autoencoder` | Ch14 §14.7 | Return-supervised bottleneck (`n_factors` wide); family control | Training runs and prediction sets |
| Causal | `09_causal_dml` | Ch15 | Does `r12_2` cause next-month return given Beta, IdioVol, LME, Variance; walk-forward DML, Driscoll-Kraay SE, block placebo | One row in registry table `causal_runs` |
| Model analysis | `10_model_analysis` | n/a | Family-leader HAC intervals, fold record, prediction correlations, decile spread vs cost, label variants, DML two gates, conformal coverage | Nothing (reads registry) |
| Backtest | `11_backtest` | Ch16 §16.4–16.8 | Plumbing test; every prediction set x every entry scheme, equal weight; per-strategy DSR | One run per (prediction set, scheme): `daily_returns.parquet`, `weights.parquet`, `spec.json` under `run_log/backtest/{hash}/`; no trade/fill ledger |
| Portfolio | `12_portfolio_management` | Ch17 §17.2–17.8 | Top-10 predictions x top_k grid x {score_weighted, conformal_weighted} | One backtest run per allocation cell |
| Risk | `13_risk_management` | Ch19 §19.3–19.6, 19.8 | Selects parent run; states why no position control applies; registry query returns empty | Nothing (by design) |
| Costs | `14_costs` | Ch18 §18.2–18.5 | Carrier re-run at each `cost_grid_bps` level; Sharpe-vs-cost slope per allocator | One backtest run per cost level |
| Holdout predictions | `15_holdout_predictions` | Ch20 | Refit carrier on history ending before holdout; retire defective generations | One training run + one prediction set keyed to the derived holdout fold |
| Holdout backtest | `16_holdout_backtest` | Ch20 | One run with allocator, concentration, cadence and cost fixed; conformal widths from validation residuals | One backtest run at `stage='holdout'` |
| Strategy analysis | `17_strategy_analysis` | Ch20 §20.1 | Block-bootstrap Sharpe CI, PSR/DSR with cohort K, decay pairs, factor attribution with 500-portfolio placebo, extended micro-cap cost grid, selection statement | Nothing (tear sheet takes its no-trades branch) |

Only `10` and `17` interpret; every other notebook states what its run produced and stops. Registry: `run_log/registry.db`, each training run, prediction set and backtest keyed by a hash of its spec; populations are "one training run per configuration (and label), one complete validation prediction set per (configuration, label, checkpoint)".

## Design decisions and why

| Decision | Choice | Why |
|---|---|---|
| Cadence | Monthly, month-end; no faster rebalance | Release is monthly; accounting ranks barely move month to month, so a faster book mostly pays to trade sort noise. |
| Book | Long-short top-k, dollar-neutral | A long-only top book moves with the market as much as with the ordering. |
| Sizing | Equal weight within leg (baseline) | A covariance-based weighting folds an estimate into the result; the ordering's contribution can no longer be separated; split-scoped ids do not reliably supply per-firm histories. Equal weight discards the within-set ranking on the argument it is too noisy to size with; `12` tests that argument. |
| Label variants | Raw, per-month 1st/99th clip, median split | Squared-error fits are steered by the extreme right tail (raw kurtosis 335.36 vs 6.47 winsorized) while IC counts a tripled firm once among thousands. Cross-sectional thresholds use only the month's own information, so no per-fold refit. |
| Baseline floor | Raw provider-ranked `r12_2` IC fixed in `02` before features exist | Comparison floor must be named before search; it is the causal treatment, not the strongest signal. |
| Feature construction | Same-row only; no fitted transform | `03` fits nothing and reads no other row; timing contract demonstrated by rebuild-without-holdout equality, not prose. |
| Feature triage | Effect size + sign consistency, not just FDR | With ~2,500 firms x ~300 months tiny ICs clear any p-value threshold. |
| GBM test design | Hold the target fixed, vary objective (mse/mae/huber) | Cleaner test of the tail diagnosis than changing the target; `num_leaves` barely moves it. |
| Checkpoints | Register every checkpoint as its own prediction set (GBM 10 per run; TabM/CAE/SDF/SAE per declared epoch; IPCA converged ALS map as one) | A checkpoint is part of a configuration; best-of-N post hoc is selection. Selection happens only in `11` on validation backtest Sharpe. |
| Latent family | Two pairs: IPCA vs CAE (linear vs network exposure map); SDF vs SAE (pricing vs direct prediction); SAE is the control | What SAE fails to achieve is what the factor structure is worth. Number of factors is declared, not read off the data. |
| PCA | Absent | Needs the same firm across the full sample (split-scoped ids cannot promise it) and uses characteristics nowhere; IPCA is the linear characteristic-sorted baseline. |
| Causal DML | Development-stage robustness diagnostic for momentum, not a selection input | Conditioning on four risk characteristics does not establish ignorability, overlap or no-interference. |
| DML preset | `dml_250k`; canonical request resolves full population (`max_samples: 0`) | A fleet-default row cap spent on whole decision months leaves under two years of history on a ~2,500-wide panel. |
| Allocators | Only `score_weighted` and `conformal_weighted`; `hrp`, `mvo_ledoit_wolf` excluded by identification; `inverse_vol`, `risk_parity` excluded by decision | 12 monthly bars give a covariance of rank <= 11; only top_k=5 (10 names) is identified. `get_allocators` injects the case-study lookback 12, overriding `compute_mvo_weights`' 126 default; shrinkage does not manufacture observations. Supplied `Variance`, `IdioVol`, `Resid_Var`, `Beta`, `MktBeta` in `labels/prices.parquet` could drive a lookback-free allocator (not implemented). |
| Cash | `1_000_000`, integer shares | At $100k, top_k=50 long-short holds ~$10K per leg per name, below realistic round-trip granularity. |
| GBM binning | `max_bin: 255` on CPU | 63 (GPU default) had been carried over, quantizing the design matrix into a quarter of the bins and losing split points. |
| Device | `device` and `num_threads` inside TabM / latent-factor training identity | CUDA and CPU arithmetic differ; same config hashes differently. |
| Macro context (SDF only) | Unrevised daily series with `macro_availability_lag_days: 1` | A finalized quarterly figure was not readable at decision date; same-close data would leak. |
| Cost model | Draw the band [5, 20] bps per leg (x2 round trip), not a midpoint; sweep [0..50] | State when a cost is assumed; a midpoint claims precision the data cannot support. |
| Risk overlays | None registered | A stop needs a price between rebalances; the vectorized path holds one return per rebalance; anonymous ids make a price join impossible. Permanent, from the release, not the engine. |
| Portfolio limits | Governance, not Sharpe variants | Sweeping a cap keeps whichever was loosest. |
| Holdout | Calendar 2016, one year | 10 x (10Y + 1M + 1Y) from 1996 leaves one year; widening starves folds. Carry sampling error explicitly. |
| Carrier resolution | `resolve_solvent_carrier` re-run in 14, 15, 16, 17 | A hash copied between notebooks agrees only until the sweep is rebuilt. |
| Deflation | Per-strategy DSR in `11`; cohort DSR once in `17` | K is unknown until the last stage. |

## Market-specific guardrails

### Data, encoding and alignment
- Assert at load that characteristics are ranks in [-0.5, 0.5]; reading a rank as a level yields plausible numbers to the end.
- Read alignment out of the data: correlate `ST_REV` with the label at candidate lags (+0.904 previous row, -0.028 own row). The panel is already shifted; shifting again fails silently.
- Assert the label is null exactly on each entity's last period; a valid-row count cannot tell a fabricated tail from a complete one.
- Assert the provider's complete-case guarantee instead of imposing a drop rule; dropping rows on whichever columns a frame lists first is an invisible screen.
- Assert a one-to-one join between features and labels on (`timestamp`, `symbol`); flattening an array offers two positions that look like ids.
- Describe the universe by tenure: a few long-lived firms supply most rows, and the typical member's history decides which statistics can be computed.
- Count breadth on the dates the strategy acts, against 2 x top_k; a sample average hides thin stretches.
- Survivorship, delisting and dividend conventions are inherited from the provider and cannot be audited (ids persist only within a block); state as a limitation, never claim survivorship-clean.
- Close-to-open gap: the label is close-to-close, fills are at next open, the release has no open price; the cost band is the only cushion.

### Features and timing
- Only same-row constructions; register every column in exactly one family `pattern`; recompute composites from declared members by a second route and assert each released column is claimed exactly once (a nullity check on a complete-case panel passes whatever the composites contain).
- Rebuild features with the holdout withheld and assert pre-holdout values are equal (`03` §D.3).
- Mixed composites/interactions get lookback 18 / lag 0; mis-declaring the lag misdraws `plot_timing_contract`.
- Say what an audit can reach; a passing nullity check is not evidence of window compliance.
- Cross-sectional thresholds (month's own median/percentiles) may be applied to the whole sample; any time-axis threshold or fitted scaler must be re-estimated inside each training window.

### Evaluation and multiple testing
- Compute IC per month then average; pooling weights months by firm count and mixes market moves with ranking. Set quantile boundaries within month.
- Use Newey-West HAC SE told the label span (1M); baseline naive t +6.89 vs HAC t +6.14 on 5 Bartlett lags.
- Thin months: `min_obs` = half the median firm count (`02`); `MIN_CROSS_SECTION = 30`, `MIN_IC_MONTHS = 20`, `MIN_FOLD_MONTHS = 5` (`04`).
- BH-FDR at `FDR_ALPHA = 0.05` over a candidate set fixed by `03`; report search size beside every p-value; never add a candidate after seeing results.
- Cluster near-duplicates on 1-|corr| at `REDUNDANCY_CUT = 0.70` and name one representative (informational only; three variance measures would otherwise clear FDR three times).
- Model-notebook ICs (`05`–`08`) are averaged monthly rank correlations with no HAC adjustment: exposition only. Use `04`'s HAC inference for per-feature claims and `10`'s `ic_se_hac` for family claims.
- State the axis each grid explores before reading a "best": the linear sweep varies penalty only; the TabM grid varies capacity only (objective fixed by the runner).

### Modeling
- Heavy-tailed target: compare raw vs `fwd_ret_1m_win` and mse vs mae/huber (`by_label`, `by_objective` frames) before blaming features or penalty.
- Register every checkpoint; never keep best-of-10 post hoc (reports the max of ten as one number).
- Read family leaders by `covers_zero` first; a leader whose HAC interval covers zero is indistinguishable from no signal.
- Print `n_configs` per family and discount wide-grid families: the leader is chosen by the statistic the CI surrounds and the CI does not correct for it.
- Treat latent-factor IC as a separate evidence kind: convergence diagnostics can be healthy while IC intervals cover zero; "a model that prices well and ranks badly is not a contradiction".
- Check the leader prediction correlation matrix before averaging families; strongly correlated pairs are one view twice.
- Attach a signal to the label it holds on; a winsorized-only signal is a claim about the bulk, not the raw cross-section.
- Keep `causal_dml` off classification menus (nuisance models are regressors); all six `causal_runs` rows are on `fwd_ret_1m`.
- Placebo block = `treatment_window: 12` (declared, derived from r12_2's construction); shorter blocks destroy the serial dependence the refutation preserves.
- Compare placebo HAC t-statistics, not effects; p cannot fall below 1/(permutations+1), so a floor value is a bound; block permutation disturbs timing, not confounding. If the DML estimate is not separable from zero, the placebo says little either way.

### Backtest, allocation and costs
- Plumbing test: random ranking through the identical path (`method: "score_weighted_top_k"`, `TOP_K = 20`), PASS iff |Sharpe| < `PLUMBING_SHARPE_TOLERANCE = 1.5`; print the tolerance.
- Select on validation backtest Sharpe after costs, not IC; monthly rebalance across thousands of names is where turnover bites, and the smallest, widest-`Spread` firms are least tradable.
- Moment-based allocators need observations > names held; with 12 monthly bars only top_k=5 is identified.
- Read `insolvent` first; an allocator whose runs all went insolvent has null `avg_sharpe` and is excluded; `resolve_solvent_carrier` refuses insolvent carriers.
- Scope conformal reads by generation/carrier, not allocator name: `walk_forward_v2` and `walk_forward_v3` sweeps on one prediction set both sit under `conformal_weighted` (22 rows drawn as one line).
- Cost sensitivity is about slope, not height; height carries every prior selection. The declared level must sit inside the grid. The cost curve does not say which strategy would have been selected under another cost assumption.
- A single cost assumption flatters the more active allocator (score/conformal re-size monthly); sweep in `14` and run the micro-cap-realistic grid in `17`.
- Never simulate stops on the vectorized monthly path; never sweep gross-exposure or per-name caps as Sharpe variants.
- `MAX_SYMBOLS` is not a scale knob here (only trims the price panel for the rebalance calendar); keep `MAX_SYMBOLS = 0` or run under `EXECUTION_TIER='preview'` with a `WORKSPACE`.
- Open the study before resolving any path or building `BacktestExplorer`; the preview tier rewrites `ML4T_OUTPUT_DIR` process-wide and an earlier `CASE_DIR` points at the released registry while writes go to the preview one.

### Holdout and reporting
- Holdout 2016 untouched until `15`/`16`; validation folds are read many times before `05` and are not out-of-sample.
- The refit must produce a training hash different from the validation `carrier['training_hash']` or raise "did not refit"; retire rows flagged VALIDATION-FITTED (`refitted=False`). The registry actually contained such a set.
- Derive `holdout_training_hash` + checkpoint and look up by identity; never search the registry for the holdout prediction set.
- Calibrate conformal widths on validation residuals only with `holdout_conformal_embargo_steps`; `ensure_conformal_calibration_identity` adds `CALIBRATION_VERSION` and embargo steps to the spec so a stale calibration cannot be reused silently.
- Fix predictions, allocator, concentration, cadence and charge upstream of `16`; every open knob is a decision after seeing the holdout.
- Twelve observations: an interval covering zero means "the holdout did not resolve the comparison", not decay and not persistence. The holdout is re-runnable (delete the generation and regenerate), not spent; exactly one readable generation at a time.
- Factor loading is not alpha: read each beta against the 500-portfolio placebo interval; only betas outside it are attributable to selection.
- Trial count K excludes superseded generations but includes all label variants (~2,300 backtests added by `fwd_ret_1m_win` and `fwd_class_1m`).
- Read results from the registry via `17`; never copy numbers into prose; never cross-resolve hashes between `v3.0.0` and `v3.1.0` bundles.
- Expect "nothing registered" from `13` and the no-trades branch in `17`; when excluding a method, record the real constraint and the evidence (an earlier comment blamed unstable identities; `r36_13` non-null on all 804,530 rows refutes it).

## Results and lessons

Numbers below are what the notes carry; results cells for `04`–`17` (model ICs, Sharpes, decay magnitudes, which allocator won) live in the executed notebooks and the registry and are not restated.

### Data and label facts (`01`, `02`)

| Quantity | Value |
|---|---|
| Rows in release | 804,530 (`r36_13` non-null on all) |
| Development firm-months | 780,882 = 780,882 effective observations (uniqueness 1.0000) |
| Development month-ends | 312; minimum cross-section 1,293 firms in the baseline computation |
| `ST_REV` rank corr with label | +0.904 previous row, -0.028 own row |
| Label autocorrelation | -0.0227 at lag 1, -0.0045 at lag 12 (short-term reversal in returns, not label construction) |
| Raw `fwd_ret_1m` | std 0.17430, kurtosis 335.36 |
| `fwd_ret_1m_win` | std 0.14850, kurtosis 6.47; clipping touches ~1/50 of rows |
| Cross-firm dispersion | peak 22.4% (2000), trough 11.8% (2013), median year 15.3%; widens in stress, drifting down since 2000 |
| `fwd_class_1m` base rate | 0.438–0.500 monthly, mean 0.498; up to 0.084 of a month's firms tied at the median (ties appear to go to class 0, inference) |
| Baseline `r12_2` IC | mean +0.04398, HAC SE 0.00716, naive t +6.89, HAC t +6.14 (5 Bartlett lags) |

### Model-family lessons (`04`–`10`)

| Stage | Finding |
|---|---|
| `04_evaluation` | The single-feature screen "found what the literature would lead you to expect"; promotions can exceed BH-significant count because the two PROCEED routes are alternatives; where `IC_THRESHOLD = 0.01` sits in the IC distribution is reported so the reader sees whether it binds. |
| `05_linear` | On raw `fwd_ret_1m` the full 28-config linear grid "produced nothing"; the same 28 configs land on the opposite side of zero on `fwd_ret_1m_win`; the ten-decade Ridge sweep (1e-3..1e7) moves IC less than the label choice; curves are flat over most of the range and fall at the largest penalties on every target. The target, not features or penalty, decides the result. |
| `06_gbm` | Objectives (mse < mae < huber expected order) separate the grid on the raw label while `num_leaves` barely moves it; huber threshold derived per fold from the label spread. Trees supply interactions without naming them. |
| `07_tabular_dl` | Capacity-only grid (`tabm_s` 64/4, `tabm_m` 128/8, `tabm_l` 256/16 hidden units/members) under squared error; narrative: a nonlinear model extracts something the linear grid did not (numbers in `10`). |
| `08a`–`08d` | Scored on an axis they were not fitted for; structural convergence can be healthy while IC intervals cover zero. |
| `09_causal_dml` | Reports a DML coefficient with Driscoll-Kraay SE, naive-OLS gap on identical out-of-fold rows, and a block-permutation p; six `causal_runs` rows, all on `fwd_ret_1m`; outcome not in the digest. |
| `10_model_analysis` | The family-leader table printed a nonzero count of intervals covering zero; point estimates would have supported an ordering the intervals do not. Winsorizing changed which families clear zero; tree families mostly already had that robustness. |

### Funnel lessons (`11`–`17`)
- Plumbing: random ranking passed |Sharpe| < 1.5 (the only standalone result of `11`).
- Backtest surface holds hundreds of configurations; allocation adds its own grid; both counts feed K.
- Risk stage: zero registered runs by design.
- Cost curve: slope reported per allocator, declared level mid-grid; defect found and fixed where two conformal generations were drawn as one line.
- Holdout: a prior generation was validation-fitted and replaced by `15`; `16` prints the validation-vs-holdout Sharpe pair; `17` finds the one-year interval too short to resolve decay in at least some rows (status column). The selected configuration's registered Sharpe differs from the resolver's common-support figure (the latter re-ranks the conformal field on shared timestamps).
- The lineage resolver without a solvency check once picked a retired conformal generation and `17` §6 raised "Incomplete holdout closure pairs"; fixed by `resolve_solvent_carrier`.
- Which allocator (score vs conformal) and which top_k (`RANK1_TOP_K`) won is read from the registry, not stated in the notes; "either can win".

## How to adapt this pattern to a new dataset

### Feasibility (`01`), before any model
1. Assert encoding at load (ranks in [-0.5, 0.5] vs levels).
2. Tenure distribution: how long members stay; which firms supply the rows.
3. Breadth on each rebalance date >= 2 x top_k.
4. Exceedance curve of |move| vs round-trip band 2 x per-leg range; stop rule: typical move smaller than the round trip sends the design back.
5. Within-entity rank autocorrelation of the sort (drop gapped entities, then average); if ranks barely move, do not rebalance faster.
6. Folds fit the sample (n_splits x (train + buffer + val)) and leave the holdout unread.

### Labels (`02`)
7. Read alignment out of the data with a return-restating column at candidate lags; never shift a pre-shifted panel.
8. Assert nulls only on each entity's last period.
9. Use cross-sectional thresholds (no per-fold refit); re-estimate anything time-axis inside each training window.
10. Compute uniqueness / effective N; a 1-period label at matching sampling is 1.0.
11. Fix a named baseline IC with HAC SE (`cross_sectional_ic_series` -> `compute_ic_hac_stats`) before features exist; `min_obs` = half the median cross-section.
12. Commit fold boundaries to `config/cv_config.json`; declare `horizons` (what seals the last fold) separately from `buffer`.

### Features (`03`)
13. Same-row constructions only; declare every column in exactly one `features.families` pattern with `role`, `lookback`, `lag`; mixed-timing constructions get their own row spanning everything they read.
14. Recompute composites from declared members by a second route; rebuild without holdout and assert equality; assert the 1:1 key join; render with `register_frame` / `plot_timing_contract`.

### Evaluation (`04`)
15. Per-period IC averaged over periods; HAC SE told the label span; BH-FDR over a fixed candidate set.
16. Triage with `MIN_COVERAGE = 0.70`, `MIN_CROSS_SECTION = 30`, `MIN_IC_MONTHS = 20`, `MIN_FOLD_MONTHS = 5`, `FDR_ALPHA = 0.05`, `IC_THRESHOLD = 0.01`, `MIN_SIGN_CONSISTENCY = 0.60`, `REDUNDANCY_CUT = 0.70`, `N_QUANTILES = 5`: PROCEED (confirmation) if BH q <= 0.05; PROCEED (exploration) if sign consistency >= 0.60 and |IC| >= 0.01; STOP if coverage < 0.70; else REVISE (not rejection). Bind each judgement once and print it as a statement.
17. Write `evaluation/triage_ledger.parquet` so `20_strategy_synthesis/02_feature_evaluation` can read it alongside the other case studies.

### Models (`05`–`09`)
18. Run all label variants through one menu (`config/training/{label}.yaml`); keep `causal_dml` on continuous labels only; omit `pca` when identities are split-scoped.
19. Test target vs fit: raw vs winsorized label (`by_label`), mse vs mae vs huber at fixed target (`by_objective`, `checkpoint_vs_grid`).
20. Register every checkpoint as its own prediction set; put `device` and `num_threads` in the training identity; use LightGBM `max_bin: 255` on CPU.
21. Declare `n_factors`; publish the least-structured member (SAE) as the control; feed SDF only unrevised daily macro series with a 1-day availability lag.
22. Treat DML as a diagnostic: `treatment_window` equal to the treatment's construction span; size the row cap for the panel width; read coefficient vs Driscoll-Kraay SE, then naive-OLS gap, then block-permutation t-stat placebo.

### Selection funnel (`10`–`17`)
23. Family leaders: `ci = ic_mean_daily ± 1.96 * ic_se_hac`, filter `ic_se_hac > 0`, read `covers_zero`, print `n_configs`; forward a family iff `interval_excludes_zero & spread_bps > round_trip_lo` (`spread_bps = (top - bottom) * 10_000`); also report `clears_high_cost`.
24. Plumbing test first; then every prediction set x `top_k_grid` at equal weight; drop any k the cross-section cannot fill.
25. Allocation: top-10 predictions x k grid x declared allocators; equal weight is the baseline row, never an allocation method; check identification (observations > names held) before declaring moment-based allocators.
26. Risk parent = max validation Sharpe over best(11) and best(12); on a vectorized path register nothing.
27. Cost grid halved into `commission_bps = cost_bps / 2`, `slippage_bps = cost_bps / 2`; declared level inside the grid; read slope only.
28. Holdout: `build_holdout_training_spec` -> `training_hash_from_spec`; assert the hash differs from validation; one readable generation; `REPLACE_HOLDOUT=True` only to supersede a different configuration; conformal widths from validation residuals with embargo; fix every knob upstream.
29. `17`: block-bootstrap CI, PSR, cohort DSR with K = every non-superseded trial (including label variants), paired decay tests (strategy vs self across the split, then vs EW universe), Layer 1 factor regression + Layer 2 500-portfolio placebo, extended micro-cap cost grid; papermill overrides in exactly one cell.
30. Re-resolve the carrier (`resolve_solvent_carrier`) in every downstream notebook; read results from the registry, never from prose.

## Related references

- `chapters/06_strategy_definition.md`: feasibility study (§6.2–6.6), search accounting and run log (§6.7) that `01` and the registry implement.
- `chapters/07_defining_the_learning_task.md`: labels (§7.2), single-feature evaluation (§7.3), multiple-testing pass (§7.4) behind `02` and `04`.
- `chapters/08_financial_features.md`: contextual slow-moving features (§8.4), representatives of near-identical features (§8.6).
- `chapters/09_model_based_features.md`: evaluation context cited for `04` (Ch7–9).
- `chapters/11_ml_pipeline.md`: walk-forward fitting, menus and the registry pattern used by `05`.
- `chapters/12_gradient_boosting.md`: LightGBM objectives/tuning (§12.2) and TabM for tabular data (§12.3).
- `chapters/14_latent_factors.md`: IPCA (§14.5), CAE (§14.6), SDF and SAE (§14.7).
- `chapters/15_causal_estimation.md`: walk-forward DML, Driscoll-Kraay SE, block placebo.
- `chapters/16_strategy_simulation.md`: vectorized long-short backtest, plumbing test, per-strategy DSR (§16.4–16.8).
- `chapters/17_portfolio_construction.md`: allocator sweep, score vs conformal weighting, identification limits (§17.2–17.8).
- `chapters/18_transaction_costs.md`: cost-sensitivity slope, era-dependent spreads (§18.2–18.5).
- `chapters/19_risk_management.md`: why overlays are not representable on a vectorized monthly path (§19.3–19.6, 19.8).
- `chapters/20_strategy_synthesis.md`: holdout refit, closure, deflation with cohort K, cross-case aggregation (nb01, nb04, nb05).
- `case_studies/us_equities_panel.md`: the daily, public-identifier equity panel where overlays and `MAX_SYMBOLS` do apply.
- `case_studies/etfs.md`: another slow-cadence cross-section with the same `setup.yaml` schema.
- `libraries/ml4t_diagnostic.md`: IC/HAC helpers, FDR, DSR/PSR, conformal coverage.
- `libraries/ml4t_backtest.md`: `run_backtest`, allocators, cost model, registry hashes.
- `libraries/ml4t_models.md`: GBM, TabM and latent-factor runners.
- `workflow.md`: the 17-stage sequence this study instantiates.
- `guardrails.md`: cross-cutting leakage, multiple-testing and holdout rules.
- `decision_rules.md`: triage, tradeability and selection thresholds in one place.
- `evidence.md`: where this study's findings sit among the nine.
- `companion_repo.md`: run order, artifact bundles, registry layout.
- `glossary.md`: shared vocabulary.

Further reading:
- Chen, Pelger and Zhu (2020): the panel and the neural SDF.
- Gu, Kelly and Xiu (2020): empirical asset pricing via machine learning; conditional autoencoder.
- Kelly, Pruitt and Su: instrumented PCA.
- Benjamini and Hochberg: FDR control. Newey and West: HAC standard errors. Driscoll and Kraay: panel-robust standard errors.
- Bailey and Lopez de Prado: deflated Sharpe ratio (inference on attribution).
- Vovk et al. (2005); Lei et al.: conformal prediction finite-sample coverage.
- TabM: rank-1 adapter MLP ensembles. Ledoit and Wolf: covariance shrinkage.

## Glossary

- IC: Spearman rank correlation between signal and forward return across firms in one month, averaged over months.
- HAC / Newey-West SE: autocorrelation-consistent standard error on the monthly IC series, Bartlett lags chosen by rule (5 at a 1-month span).
- BH / FDR: Benjamini-Hochberg false-discovery-rate control across the feature candidate set.
- Exceedance curve: fraction of firm-months whose |move| is at least x, read where it crosses the round-trip band.
- Winsorized label: per-month cross-sectional clip at the 1st/99th percentiles.
- Uniqueness / effective N: share of a label's window not shared with neighbours; 1.0 for a 1-month label at monthly sampling.
- Timing contract: per-family `lookback`/`lag` declaration of what a feature reads and when it is knowable.
- Composite: equal-weight mean of member ranks. Interaction: product of two centred ranks; sign encodes agreement.
- Triage ledger: one row per characteristic with PROCEED/REVISE/STOP and which route fired.
- TabM: shared-backbone MLP ensemble with rank-1 per-member adapters and per-member heads.
- IPCA: exposures linear in same-month characteristics, fitted jointly with factor returns by ALS.
- CAE: IPCA structure with a network mapping characteristics to exposures.
- SDF: firm-month weights that price the cross-section under E[m R] = 0; adversarial moment network picks the worst-priced test assets.
- SAE: characteristics -> `n_factors` bottleneck -> return, no factor interpretation; the family's control.
- DML: residualize treatment and outcome with nuisance models (`HistGradientBoostingRegressor`) fit on earlier months, regress residuals.
- Driscoll-Kraay SE: panel SE robust to dependence over time and across firms in the same month.
- Block permutation: placebo that permutes the treatment in 12-month blocks to preserve its serial dependence.
- Split-scoped identifier: anonymous firm id persistent only within one released tensor block.
- Checkpoint: a registered model state at an iteration/epoch, treated as its own configuration.
- Entry scheme: top k long, bottom k short, equal weight. Allocator: `score_weighted` or `conformal_weighted`; equal weight is the baseline, not an allocator.
- Conformal width: interval sized so 1 - alpha (alpha default 0.2) of past errors fall inside, calibrated per firm on data the model did not fit.
- Carrier: the single selected configuration (validation rank-1, solvent, common-support ranked) that 14–17 all resolve.
- Training identity / hash: deterministic hash of the training spec; a holdout refit must produce a new one.
- Generation: one registered prediction set for a window; one holdout generation readable at a time.
- Plumbing test: random-ranking backtest proving the engine adds no edge.
- DSR / PSR: deflated Sharpe (probability the observed Sharpe beats the expected max of K trials given T, skew, kurtosis) / probabilistic Sharpe.
- covers_zero: flag that the 95% HAC interval includes zero.
- Decile spread (bps): top minus bottom decile mean forward return x 10,000. Round trip: twice the per-leg cost.
- Vectorized forward-return path: one weight vector per rebalance times the realised monthly return; no intra-period prices, no trade ledger.
- Maximum adverse excursion: furthest a position moved against its direction while held.
- Placebo benchmark (`17`): 500 random dollar-neutral portfolios giving a null interval per factor beta.
- Holdout closure: paired comparison of the selected strategy across the validation/holdout split and against the EW universe.
- Common support: timestamps every conformal candidate shares; the resolver re-ranks on it.
- Preview tier / workspace: execution mode that rewrites `ML4T_OUTPUT_DIR` to a scratch registry.
- Decimalization: 2001 move to decimal ticks; pre-2001 spreads 15–30 bps, post 5–15 bps.

### Reader uncertainty carried from the notes
- 46 released vs 57 total features: 57 = 46 + 11 constructed columns (inference).
- README's price > $5 / ADV > $1M screen is presumed provider construction, not applied in-repo (inference).
- `n_factors`, CAE epoch budget, TabM epoch checkpoints/learning rates, Newey-West lag rule, placebo permutation count, declared commission/slippage levels, `holdout_conformal_embargo_steps`, block-bootstrap parameters, the Layer 1 factor model, the extended micro-cap cost grid levels, the `dsr_mp` / `dsr_er` definitions and how `get_top_n_predictions` picks n are not visible in the notes.
- Whether `03` applies an explicit 6-month shift or relies on the provider convention is not visible.
