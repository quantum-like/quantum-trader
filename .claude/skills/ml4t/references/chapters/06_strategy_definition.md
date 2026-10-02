# Chapter 6: Strategy Research Framework
> A trading strategy is an executable decision process, not a signal or a model: it has a decision schedule, an admissible information set, a score-to-position mapping, constraints and costs, and it must be defined at decision time and evaluated as if it were live. This chapter is the methodological spine of the book: place the idea on a strategy map (family = feasibility filter, source of edge = durability filter), freeze a versioned trading setup, define "better" economically with model, signal and strategy diagnostics kept apart, run a walk-forward evaluation protocol with label purge, feature embargo and a sealed holdout, establish a narrow baseline before widening the search, and log every trial so the search stays countable. The nine case studies (`case_studies/*/config/setup.yaml`) are the concrete instances of that protocol.

## When to use this reference
- Starting any new strategy idea: you need to name its family, source of edge, counterparty and failure mode before touching data.
- Writing or reviewing a trading setup: universe rules, decision cadence, admissible information, mapping from scores to positions, constraints, costs.
- Designing or auditing a train/validation/holdout split for time series (walk-forward, purge, embargo, nested walk-forward, CPCV).
- Deciding whether a change is "tuning" or a new strategy version.
- Setting up run logging / trial accounting so later significance claims can be deflated by the number of trials.
- Reading or writing a `setup.yaml`-style configuration (`universe`, `decision`, `mapping`, `costs`, `labels`, `evaluation`, `backtest.sweep` blocks).
- Running a first exploratory sort (quintiles, era split, rank autocorrelation) to decide whether an idea deserves modeling.

## Core ideas (the why)
- **Research loop must mimic the live loop.** Every quantity used in research must be stated in decision-time terms; otherwise historical tests say nothing about future behavior.
- **Strategy map first.** Classify an idea by (a) strategy family (can it be implemented in this market structure?), (b) source of edge (will it persist after costs, constraints, competition?), (c) dominant feasibility constraints and failure modes. The first question for any idea: *who is on the other side, and why are they happy to trade with us?*
- **Source-of-edge taxonomy: SLOW / WRONG / RISK.** SLOW: information diffuses or capital moves with a lag. WRONG: behavioral under/over-reaction, performance chasing. RISK: paid to bear a risk others shed. Momentum is a SLOW/WRONG blend ("others are a step behind"), not a risk premium, so its predicted failure modes are sharp reversals at turning points and decay as the edge becomes widely known.
- **Two clocks.** Signal horizon (e.g., 12-month lookback) and holding horizon (e.g., 1 month) are separate design choices; keep them explicit.
- **Versioned setup = comparability.** Changing parameters inside fixed invariants is tuning the same strategy; changing tradability rules, decision schedule, score-to-trade mapping, constraints or cost treatment is a new mechanics version that must be recorded as such.
- **Define "better" economically, in three separate roles.** Model diagnostics (fit/accuracy), signal diagnostics (does the score order forward returns?), strategy outcomes (P&L after mapping, constraints, costs). Optimizing every micro-decision directly on simulated P&L is one of the easiest ways to overfit a backtest.
- **Decision-time admissibility is the CV design constraint.** At decision time t, any sample whose label, feature or selection criterion depends on information unavailable at t is excluded from training.
- **Out-of-sample is a protocol, not a slogan.** Chronology, buffers, a sealed holdout, and governance separating model selection (validation) from final performance estimation (holdout).
- **Narrow baseline first.** A deliberately narrow setup with timing, coverage and trading-intensity sanity checks rules out brittle setups early and earns the right to widen features or model class.
- **Countable search.** After iteration, a performance claim is credible only if you know what was tried, what was selected and what was reserved for confirmation. Trial taxonomy: strategy > trial family > trial > run, logged automatically.
- **The setup is a file, not a notebook constant.** In the companion repo every case study's invariants live in one `config/setup.yaml` that notebooks read at runtime; a threshold retyped in code is a second source of truth and drifts (`case_studies/RUN_LOG.md` lists the known exceptions). The same file feeds the evaluation protocol (`get_cv_config`), the cost model and the run-log identity, which is what makes "versioned setup" operational rather than aspirational.
- **EDA is not a backtest.** A quintile sort is visual confirmation that there is something to model plus a set of cautions. A real mechanism is "lumpy but present": still visible when the sample is cut by time. An effect that lives entirely in one period is an artifact or a dead regime. Report results in the horizon the design uses (a monthly sort yields a monthly return; compounding to annual assumes the edge recurs every month).

## Method recipes (the how)

### Recipe 1: Idea triage and strategy map
1. Name the strategy family (e.g., cross-sectional rank-and-rebalance, carry, funding arbitrage, intraday order-flow, straddle selling).
2. Name the source of edge (SLOW / WRONG / RISK) and the counterparty.
3. Write the predicted failure mode (momentum: turning-point reversals, decay; carry: crash risk; intraday: cost cliff).
4. List dominant feasibility constraints: cost class, capacity, data availability, holding horizon vs signal decay.
5. Only then run the footprint EDA (Recipe 2).

### Recipe 2: Footprint EDA with a quintile conditional-return sort (`06_strategy_definition/01_where_ideas_come_from`)
Defaults from the ETF example: `LOOKBACK = 12` months, `SKIP = 1` (12-1 momentum skips the last month because short-term reversal is a different mechanism with opposite sign), `MIN_NAMES = 30` eligible names per month, monthly cadence, 5 buckets, equal-weight buckets, equal-weight universe as benchmark, era boundary `2016-01-01`.

Panel construction: take the last close per ETF per month from the daily data (`load_etfs().select("timestamp", "symbol", "close")`); `close` is split/dividend-adjusted, so simple returns are total returns. A row is eligible only when both the signal and the outcome exist for it, and then only in months with at least `MIN_NAMES` such rows.

```python
mom = (pl.col("close").shift(SKIP) / pl.col("close").shift(LOOKBACK) - 1).over("symbol")
fwd = (pl.col("close").shift(-1) / pl.col("close") - 1).over("symbol")
elig = df.filter(pl.len().over("timestamp") >= MIN_NAMES)
rank = pl.col("momentum").rank("ordinal").over("timestamp")   # sort by (timestamp, momentum, symbol) first
```
- Cut ranks at `1 + k(n-1)/5` (equivalent to pandas `qcut` on the ranks; balanced buckets, ties never merge buckets); Q1 = worst past year, Q5 = best.
- Report gross (%/month) and net of benchmark (bucket minus EW universe, bps/month). Net of benchmark strips beta; it is **not** net of trading costs.
- Spread = Q5 − Q1, identical whether computed gross or net of benchmark (the benchmark cancels); tradable long-short series `Q5 − Q1`; `ls_t = mean / std * sqrt(n_months)`. Read the t-stat before the spread.
- Read the sort for monotonicity (all buckets order), magnitude (economically material), shape (smooth premium-like vs tail-concentrated event-like).
- Era stress test: split at the boundary, report Q1_net, Q5_net, spread, t, months per era; move the boundary to test the stability of any fade.
- Turnover proxy: month-over-month Spearman rank autocorrelation of the signal (join adjacent months on symbol, keep pairs with ≥ `MIN_NAMES` survivors, `rank("average")`, `pl.corr` per pair, average). High autocorrelation → persistent ranking → modest turnover and costs.

### Recipe 3: Freeze the trading setup (the invariants, §6.3)
Write down and version, before the first model run. The invariant list is the book's (§6.3, known here through the README summary); the key paths are read from the nine `case_studies/*/config/setup.yaml` files (2026-10-02 checkout) and are **not uniform** across case studies, so read the target study's own file before copying a shape. Every path below is `block.key`; nothing in this table is a top-level key except `strategy_id` and `setup_version`.

| Invariant | What to state | setup.yaml block / keys (companion repo) |
|---|---|---|
| Tradable universe and eligibility | per-period membership rule, floors such as `MIN_NAMES`, point-in-time eligibility file | `universe.n_assets` (8 of 9; cme_futures uses `universe.n_products: 30`); the member list is `universe.assets` (etfs), `universe.symbols` (crypto, fx, nasdaq100) or `universe.product_groups` (cme); `universe.eligibility_rule` in 4 of 9 (etfs `point_in_time_adv_10m_annual`, crypto `top_perps_by_volume`, nasdaq100, sp500_equity_option_analytics `sp500_with_options`); `universe.eligibility_file: eligibility.csv` only in etfs |
| Decision schedule | cadence, decision snapshot, execution delay, per-label cadence, rebalance step | `decision.cadence` (7 of 9: `monthly_month_end`, `daily_close`, `daily_ny_close`, `weekly_friday_close`, `8_hour_funding_aligned`); `decision.snapshot` (`close`, `month_end_close`, `settlement_price`, `ny_5pm_close`, `friday_16:00_et`, `pre_funding_timestamp`); `decision.execution_delay` (`next_bar_open`, `monday_open`, `at_funding_timestamp`, `1_bar`, `next_session_close`); `decision.cadence_by_label` (etfs, nasdaq100, sp500_equity_option_analytics). Two exceptions: nasdaq100 declares `decision.bar_frequency: 15_minute` + `decision.decision_snapshot: bar_close`; sp500_options declares `decision.entry_cadence: weekly_friday`, `entry_time: friday_close`, `holding_period_days: 10`, `hedge_cadence: daily_close`. Rebalance step is `labels.rebalance_step`, one entry per label, required |
| Admissible information | publication lags, rolling-only transforms and thresholds | feature/label construction (Ch7, Ch8); where a study declares it: `decision.iv_feature_lag: 1_day` (sp500_equity_option_analytics), `decision.characteristic_availability` / `yearly_update: end_of_june` / `monthly_update` (us_firm_characteristics) |
| Score-to-position mapping | setup class, entry logic, sizing; top-k / quantile grids | `mapping.class` (`long_only_rank_and_rebalance`, `long_short_top_k_rebalance`, `long_short_decile_rebalance`, `long_short_rank_rebalance`, `long_short_carry_rank`, `long_short_funding_aligned`, `intraday_rank_and_trade`, `systematic_straddle_sell`); `mapping.entry_logic` (e.g. `rank_selection_top_n`, `decile_sort_long_top_short_bottom`, `rank_by_carry_or_momentum`, `sell_atm_straddle_weekly`); `mapping.sizing` (`equal_weight`, `equal_weight_within_leg`, `dollar_neutral_or_beta_neutral`, `equal_premium_capital_with_fixed_cohort_fraction`). The grids are **not** beside the class: `backtest.sweep.top_k_grid` keyed by label (etfs `[5, 10, 20]`, us_equities_panel `[20, 50]`), `backtest.sweep.quantile_grid` (crypto only), `backtest.sweep.htm_cost_cascade.top_k: 20` (sp500_options) |
| Constraints | long-only vs long-short, shorting, leverage | `mapping.position_state_space`: `long_only` (etfs, sp500_equity_option_analytics), `long_short` (six studies), `short_straddle_hedged` (sp500_options). Shorting and leverage flags are in `config/backtest/base.yaml` (`account.allow_short_selling`; `allow_leverage: true` declared only by cme_futures). No `max_weight` cap is declared anywhere. The `execution` block holds only `initial_cash`, `share_type` (`integer` / `fractional`) and `allocator_lookback`, never constraints |
| Material cost components | in the market's own units: bps per leg, ticks, per-share plus half-spread, fraction of option premium | `costs.class` (`material` / `dominant`) everywhere; then market-specific: `costs.model: per_share_plus_spread` + `costs.per_share: 0.0035` + `costs.default_half_spread_usd` + `costs.spread_convention: half_spread` (etfs, nasdaq100); `costs.model: percentage` + `costs.per_leg_cost_bps_range` (sp500_equity_option_analytics `[3, 10]`, both equity panels `[5, 20]`); `costs.spread_bps` per leg (fx: majors `[1, 3]`, crosses `[3, 8]`); `costs.commission_per_contract: 2.0` + `costs.spread_ticks` (cme); `costs.fee_schedule` taker 4 / maker 2 bps (crypto); `costs.components` dict + `cost_dominance_note` (sp500_options) |
| Evaluation protocol | folds, train/val windows, label buffer, calendar, holdout dates | `evaluation.n_splits`, `train_size`, `val_size`, `holdout_start`, `holdout_end`, `calendar`, `periods_per_year`; the purge comes from `labels.buffer` (primary label) and `labels.variant_buffers` |

Any later change to tradability, schedule, mapping, constraints or cost treatment bumps `setup_version` (v1 everywhere except `us_firm_characteristics` v2); a parameter change inside the invariants does not. A full block skeleton is under Code patterns.

### Recipe 4: Separate the three metric roles (§6.4)
| Role | Question | Typical metrics | Used for |
|---|---|---|---|
| Model diagnostics | Did the model fit? | loss, R², accuracy, calibration | debugging, early stopping |
| Signal diagnostics | Does the score order forward returns? | cross-sectional IC, ICIR, quantile spread, HAC t | selecting features and models |
| Strategy outcomes | What is left after mapping, constraints, costs? | net Sharpe, drawdown, turnover, cost sensitivity | confirming a strategy, never for micro-tuning |

### Recipe 5: Walk-forward evaluation protocol (§6.5, `02_cv_foundations`)
Dataset roles: training (fit), validation (select hyperparameters, compare models, during development), holdout (final unbiased estimate, opened once at the end).

Five leakage channels and what prevents each:

| Channel | Example | Prevention |
|---|---|---|
| Label leakage | 21-day forward return overlaps validation | label buffer (purge) |
| Standardization leakage | z-scoring with full-sample mean/std | expanding-window or training-window transforms |
| Threshold leakage | percentile labels from the full sample | rolling percentile thresholds |
| Survivorship leakage | training only on survivors | rebuild the eligible universe per period |
| Point-in-time leakage | fundamentals reported after the decision date | conservative publication lag |

The CV splitters handle only the first channel; the other four live in feature, label and universe construction.

Steps:
1. Never shuffle. `KFold(shuffle=True)` trains on the future to predict the past.
2. Choose expanding (older data still relevant: `WalkForwardCV(n_splits=5, test_size=252, expanding=True)`, all history, growing train) or rolling (regimes change: `train_size=1260, expanding=False`, fixed 5-year window, stale data dropped). With a rolling window longer than early history, early folds are silently clipped to expanding folds; check each fold's actual train length.
3. Purge: training end ≤ validation start − h, with h the label horizon counted in **trading sessions** of the market calendar. 21 sessions ≈ 30 calendar days; a 21-*calendar*-day purge removes only ~14 sessions and leaks ~7. Purged rows are discarded entirely: they enter neither training nor validation. Check `purged_sessions = va[0] - tr[-1] - 1 == h` (the index gap is h + 1).
4. Embargo: when training can follow a validation block (CPCV, k-fold), drop training samples within one feature lookback after the block. Zero embargo is correct only in pure walk-forward.
5. Nested walk-forward when hyperparameters are tuned: inner walk-forward selects λ* on the pre-test years, outer loop advances the test year and re-tunes. The notebook's illustration: outer fold 1 tunes on 2019–2023 (five inner folds, each training from 2014 to the year before its validation year) and tests 2024; outer fold 2 tunes on 2020–2024 and tests 2025. If λ*₁ ≠ λ*₂ the hyperparameter is unstable and a single tuned value should not go to production.
6. CPCV for robustness: N contiguous blocks, hold out k, purge/embargo at every boundary, assemble validation predictions into complete paths. N=6, k=2 → C(6,2)=15 splits, k·C(N,k)/N = 5 paths, each path made of N/k = 3 splits, each block validated 5 times (once per path). Paths are built by round-robin 1-factorization (fix one block, rotate the rest); the notebook's occupancy plot encodes 0 = purged/embargoed, 1 = train, 2 = validation. Similar Sharpe across paths → robust; wide variance → path-dependent/overfit. PBO = fraction of paths whose in-sample rank fails out of sample. The notebook defers the full CPCV/PBO treatment to the book's Ch17; in this skill it is `chapters/16_strategy_simulation.md`.

| Method | Paths | Label buffer | Feature buffer | Use |
|---|---|---|---|---|
| Walk-forward | 1 | yes | no (train precedes val) | standard evaluation |
| Nested walk-forward | 1 per test year | yes | no | multi-year test with retuning |
| CPCV | multiple | yes | yes | robustness, PBO |

### Recipe 6: Baseline checkpoint (§6.6)
Before widening features or model class, run one deliberately narrow configuration (one label, one feature family, one model, the setup's default mapping) and pass three sanity checks. The book names the checks, timing, coverage and trading intensity, and frames the narrow baseline as governance ("earn the right" to widen); their exact definitions and thresholds in the book's prose were available only through the README summary, so the operational proxies below are this reference's, not the book's.
1. **Timing** — decisions and fills on the right bars, no lookahead. Proxy: for every trade assert the fill bar is strictly after the decision bar (`decision.execution_delay`: `next_bar_open`, `1_bar`, `monday_open`, `next_session_close`), that features at decision time use only data up to `decision.snapshot`, and that `labels.buffer` is at least the label horizon.
2. **Coverage** — predictions exist for the intended universe and periods. Proxy: count predicted rows per `(fold_id, timestamp)` against the eligible universe and the fold boundaries from `get_cv_config`. In the companion registry the eligibility manifest (which rows per fold must be predicted) is part of the run identity; `training_run_status(cs, spec).partial`, `evaluate_prediction_coverage` and `resolve_best_predictions(..., coverage_window="canonical")` surface partial coverage, and only complete members rank. `03_case_study_overview`'s `compute_coverage(results)` / `plot_coverage` (Figure 6.5) draw the spans each protocol implies from `holdout_start/end`, `n_splits`, `train_size`, `val_size`, not raw data availability.
3. **Trading intensity** — turnover consistent with the cost class and horizon. Proxy: signal rank autocorrelation (Recipe 2) read against `labels.rebalance_step` and `decision.cadence`; `avg_turnover`, `num_trades`, `total_commission`, `total_slippage` in the registry's `backtest_metrics`. A dominant-cost setup (nasdaq100, sp500_options) that re-ranks every bar fails here before any model is widened.
4. Only when all three pass, widen one axis at a time (features, then model class), each widening logged as a new trial family (Recipe 7).

### Recipe 7: Search accounting and run logging (§6.7)
- Taxonomy: strategy > trial family > trial > run. A trial family is one hypothesis with variants; a run is one fitted configuration. The book's requirement (§6.7, known through the README summary): after iteration you must be able to say what was tried, what was selected and what was reserved for confirmation. The book's trial-family rules and run-log schema were not visible to this reference; what follows is the companion repo's implementation.
- Reference implementation: the per-case-study run log documented in `case_studies/RUN_LOG.md` and summarised in `companion_repo.md` (section "case_studies/research package and the run-log registry"). It lives at `case_studies/{id}/run_log/` (`registry.db` SQLite index plus `training/`, `predictions/`, `backtest/` artifact directories) but is **not checked in**: fetch it with `uv run python scripts/download_artifacts.py --cs <id>` (release `v3.1.0-artifacts`, installed read-only with a `run_log/.release` marker), or make a writable copy with `uv run python scripts/create_experiment.py --cs <id> --output PATH`.
- What the registry records (RUN_LOG.md "Identity"): every entry is content-addressed, 12 hex chars of SHA-256 over the canonical JSON of its resolved specification, chained `training_hash` → `prediction_hash` → `backtest_hash`. Hashed: label name + label-file digest, feature-file digests + exact column list, every fold's train/validation start and end, estimator class + per-fold effective parameters, the eligibility manifest with row and fold counts, seed (`DEFAULT_SEED=42`), declared runner and preprocessing versions, preview reductions. Deliberately not hashed: notebook, git commit, write time, device. Tables: `training_runs`, `prediction_sets` (`split` ∈ {validation, holdout}, `checkpoint_kind`), `prediction_metrics` / `fold_metrics` (IC per fold, HAC t), `backtest_runs` / `backtest_metrics` / `backtest_fold_metrics`, `causal_runs`.
- Query entry points, `from case_studies.utils.registry import ...`: `load_training_runs(cs, family=)`, `load_prediction_sets`, `load_prediction_metrics(cs, prediction_hash=)`, `load_backtest_runs` / `load_backtest_metrics` / `load_backtest_fold_metrics`, `read_training_spec`, `read_predictions`, `resolve_best_predictions(cs, label, split=, top_n=10)`, `resolve_best_backtest_runs(top_n=3)`. Reporting notebooks open with `read_only_study(cs)`, never `open_study`.
- "Reserved for confirmation" in code: `evaluation.holdout_start/end` are required; holdout prediction sets are a separate `split`; scoring the holdout twice raises `HoldoutWindowSpent` unless earlier generations are named in `retiring=`; populations are declared before execution, so a half-finished sweep cannot pose as complete.
- Report search size alongside results so Deflated Sharpe and PBO are computed from the real number of trials: the count is every registered run (training runs, prediction sets per checkpoint, backtest stages) per label, not the subset you show (Ch7 for DSR; `chapters/16_strategy_simulation.md` for PBO on CPCV paths).

## Guardrails and pitfalls
Each bullet is **pitfall** — cause. Fix.

Evaluation protocol
- **Label leakage at the fold boundary** — forward labels of horizon h near the boundary embed validation prices and inflate validation scores. Purge h sessions (`label_horizon` in `WalkForwardCV` / `CombinatorialCV`); assert `va[0] - tr[-1] - 1 == h`.
- **Splitter defaults do not purge** — `WalkForwardCV` defaults (library source) are `label_horizon=0`, `embargo_size=None`, `gap=0`, `calendar=None`, `expanding=True`, so a bare `WalkForwardCV(n_splits=5)` is an unpurged expanding split counted in rows. Always pass `label_horizon` and `calendar`; pass `expanding=False` with `train_size` when you mean rolling.
- **Calendar-day purge** — 21 calendar days ≈ 14 sessions, so ~7 sessions still leak. Pass the market calendar (`calendar="XNYS"`; CME, FX, crypto in the case studies) so buffers count sessions; verify with `exchange_calendars`.
- **Missing embargo in CPCV / k-fold** — backward-looking features of post-validation training rows consume validation prices. Set `embargo_size ≥ feature lookback`; zero only in pure walk-forward.
- **Shuffled k-fold on time series** — trains on the future to predict the past (Bergmeir et al. 2018; Kohavi 1995). Chronological walk-forward only.
- **Standardization, threshold, survivorship, point-in-time leakage** — splitters do not prevent these. Training-window transforms, rolling thresholds, per-period universes, publication lags (Ch2, Ch4, Ch7, Ch8).
- **Holdout contamination** — any development decision made with the holdout destroys the unbiased estimate. Set `holdout_start`/`holdout_end` at the start; open once, for confirmation only (the companion registry raises `HoldoutWindowSpent` on a second look).
- **Single-path luck** — one walk-forward path can be a lucky sequence. CPCV paths; compare Sharpe across paths; PBO.
- **Hyperparameter staleness** — λ tuned once on old years may be wrong later. Nested walk-forward; carry one value forward only if it is stable across outer folds.
- **Rolling window longer than history** — early folds are clipped to expanding without warning, so train size is not constant. Check each fold's train length; expect the fixed size only from the first fold whose window fits.
- **Off-by-one in buffer reporting** — the index gap is h + 1 (22 for a 21-session purge) and is easy to quote as the buffer. Compute `purged_sessions = va[0] - tr[-1] - 1`.
- **Horizon mismatch** — purge gap ≠ label horizon leaves leakage. `label_horizon` = forward-return horizon (21 sessions for 1-month labels); in the case studies `labels.buffer` ≥ the primary horizon, and the widest buffer over primary + variants governs the holdout gap.
- **Dataset heterogeneity** — long histories (1990+) span regimes; short ones (crypto 2020+, intraday) limit fold depth. Match train window and fold count to history (6M–10Y, 2–16 folds); state stationarity assumptions.

Search discipline
- **Uncounted search** — the best of many trials is inflated. Trial taxonomy + automatic logging; deflate by trials tried, not trials reported.
- **Optimizing on portfolio outcomes** — micro-tuning on simulated P&L overfits. Tune on model and signal diagnostics; confirm on strategy outcomes.
- **Mechanics drift masquerading as tuning** — silently changing schedule, mapping, constraints or costs breaks comparability. Version the setup (`setup_version`); classify every change as parameter tuning or mechanics change.
- **Thresholds retyped in notebooks** — a constant copied out of `setup.yaml` is a second source of truth and drifts (RUN_LOG.md lists the known exceptions, e.g. `LIQUID_QUANTILE = 0.20` hardcoded in sp500_options). Read the yaml at runtime.
- **Baseline skipped** — broad searches on an invalid or implausible setup waste effort and inflate trial counts. Baseline checkpoint first.

Footprint EDA
- **Beta mistaken for signal** — gross quintile returns all sit near the EW benchmark; most of what buckets earn is market drift. Report net-of-benchmark alongside gross.
- **"Net" ambiguity** — net of benchmark is not net of costs (notebook 01 subtracts zero bps). Say which; defer cost deduction to the setup's cost model.
- **Spurious significance in a lumpy sample** — the ETF long-short t-stat is not significant in the full sample or either era; a two-decade average hides regime shifts. Print t before spread; era split; account for overlap and effective sample size (Ch7).
- **Annualizing a monthly sort** — compounding to the 12th power assumes the edge recurs every month, which the era split contradicts. Report in the design's horizon.
- **Edge decay and momentum crashes** — price-based edges narrow as they become known (McLean & Pontiff 2016); momentum reverses sharply at turning points (Daniel & Moskowitz 2016). Era stress tests; name the failure mode in advance; never rely on full-sample averages.
- **Short-term reversal contamination** — the latest month carries the opposite-signed liquidity-provision effect. `SKIP = 1`.
- **Unstable ranks** — run-to-run tie-breaking makes quintiles non-reproducible. Sort by `(timestamp, signal, symbol)`; ordinal ranks; cut at 1 + k(n-1)/5.
- **Thin cross-sections** — few names give noisy buckets. `MIN_NAMES` floor per period (and per adjacent-month pair for the autocorrelation).
- **Turnover blindness** — a signal that re-ranks every period is erased by costs. Measure rank autocorrelation before modeling; declare cost components in the setup.
- **Dominant-cost regimes** — at 15-minute cadence or in options (spread quoted against premium) friction is first-order. Require unusually strong signals or longer holding periods.

## Decision rules and defaults
| Decision | Rule / default | Notes |
|---|---|---|
| Idea go/no-go before modeling | monotonic sort across all buckets, material spread, premium-like shape (smooth) rather than event-like (tail-concentrated), read t before spread, survives an era split, high rank autocorrelation | "lumpy but present" passes; "only in one period" fails |
| Momentum footprint defaults | LOOKBACK 12, SKIP 1, MIN_NAMES 30, monthly, quintiles, EW buckets, EW benchmark, era boundary 2016-01-01 | ETF example |
| Dataset roles | train / validation / holdout; holdout opened once | non-negotiable in every case study |
| Walk-forward example defaults | `n_splits=5`, `test_size=252`, `train_size=1260` (rolling), `label_horizon=21`, `calendar="XNYS"`, `fold_direction="forward"` | from `02_cv_foundations` |
| Expanding vs rolling | expanding if old data still relevant; rolling if regimes change | check clipped early folds |
| Purge | = label horizon, in sessions | 21 sessions for 1-month labels |
| Embargo | = feature lookback; only when training can follow validation | CPCV, k-fold |
| Nested walk-forward | whenever hyperparameters are tuned and the test spans several years | carry λ forward only if stable across outer folds |
| CPCV | N=6, k=2 → 15 splits, 5 paths | robustness and PBO |
| Cost class | dominant → need exceptionally strong signals, longer horizons; material → choose horizon by signal decay vs cost hurdle | NASDAQ-100 intraday and S&P 500 options are dominant; the other seven material |
| Case-study protocol ranges | train 6M (intraday) to 10Y (ETFs, panels); folds 2 to 16; holdout 1 year where the data end (options 2021, firm characteristics 2016) vs 2 years (2024–2025) elsewhere, except us_equities_panel at 2016-01-01..2018-03-31 (2.25 years) | the "1 vs 2 years" line is the notebook's simplification; the table below has the exact dates; see `case_studies/*.md` |
| Setup versioning | any change to tradability, schedule, mapping, constraints or material costs = new `setup_version` | parameter tuning does not |
| Sequencing | EDA footprint → freeze setup → economic objective + metric roles → leak-free walk-forward with sealed holdout → narrow baseline (timing, coverage, intensity) → logged, countable search → Ch7 label/overlap/N_eff checks | |

Case-study protocols as recorded in each `config/setup.yaml` (2026-10-02 checkout; universe sizes are `universe.n_assets` / `n_products`, which is what notebook 03 reads) with the chapter track from notebook 03's `CHAPTER_TRACKS`:

| Case study (asset class, N) | Cadence | `mapping.class` | Costs | Folds / train / val | Holdout | Calendar | Track |
|---|---|---|---|---|---|---|---|
| etfs (multi-asset ETFs, 100) | monthly month-end close, next open | long_only_rank_and_rebalance | material, per_share_plus_spread (half spread) | 8 / 10Y / 1Y | 2024-01-01..2025-12-31 | NYSE | Ch6–Ch21 |
| crypto_perps_funding (crypto, 19) | 8-hour funding-aligned, at funding timestamp | long_short_funding_aligned | material, taker 4 / maker 2 bps | 2 / 2Y / 1Y | 2024-01-01..2025-12-31 | crypto | Ch6–Ch12 |
| nasdaq100_microstructure (equities, `n_assets: 115`; notebook prose says 114) | 15-minute bar close, 1-bar delay | intraday_rank_and_trade | dominant, per_share_plus_spread, 5 bps friction floor | 2 / 6M / 6M | 2021-07-01..2021-12-31 | NYSE | Ch6–Ch12 |
| sp500_equity_option_analytics (hybrid: equities traded on options-derived features, 633) | weekly Friday 16:00 ET, Monday open | long_only_rank_and_rebalance | material, percentage 3–10 bps per leg | 2 / 2Y / 1Y | 2021-01-01..2021-12-31 | NYSE | Ch6–Ch21 |
| us_firm_characteristics (equities, ~2,500; setup v2; data 1990-01-01..2016-12-31) | monthly month-end close, next open | long_short_top_k_rebalance | material, 5–20 bps per leg (15–30 pre-2001) | 10 / 10YE / 1YE | 2016-01-01..2016-12-31 | null (monthly returns; calendar-aware splitting needs daily data) | Ch6–Ch14 |
| fx_pairs (FX, 20) | daily NY 5pm close, next open | long_short_rank_rebalance | material, 1–3 bps majors / 3–8 bps crosses per leg | 8 / P5Y / P1Y | 2024-01-01..2025-12-31 | FX | Ch6–Ch17 |
| cme_futures (futures, 30 products) | weekly Friday settlement, Monday open | long_short_carry_rank | material, $2 per contract + 1–2 spread ticks | 5 / 8Y / 1Y | 2024-01-01..2025-12-31 | CME | Ch6–Ch17 |
| sp500_options (options, 627 ATM straddle books; README prose ~600–630) | weekly Friday entry at close, 10-day hold, daily-close delta hedge | systematic_straddle_sell (`short_straddle_hedged`; cascade `top_k: 20`) | dominant, spread quoted against premium | 2 / 2Y / 1Y | 2021-01-01..2021-12-31 | NYSE | Ch6–Ch21 |
| us_equities_panel (equities, 3,199) | daily close, next open | long_short_decile_rebalance | material, percentage 5–20 bps per leg | 16 / 10Y / 1Y | 2016-01-01..2018-03-31 | NYSE | Ch6–Ch14 |

Equities dominate the landscape (three pure equity studies plus the hybrid). Tracks tell you which downstream chapter references apply: the dominant-cost nasdaq100 stops at Ch12 with crypto, the two equity panels at Ch14, fx and cme at Ch17, and etfs, sp500_equity_option_analytics and sp500_options (the hedging case) run to Ch21.

## Code patterns and APIs
- Splitters: `from ml4t.diagnostic.splitters import WalkForwardCV, CombinatorialCV`; `from ml4t.diagnostic.splitters.config import WalkForwardConfig`.
```python
WalkForwardCV(n_splits=5, test_size=252, expanding=True)                 # expanding: all history, growing train
cv = WalkForwardCV(n_splits=5, test_size=252, train_size=1260, expanding=False,
                   label_horizon=21, calendar="XNYS")                     # rolling 5y window, 21-session purge
for train_idx, val_idx in cv.split(df_dates): ...                         # splits = list(cv.split(df_dates))
CombinatorialCV(n_groups=6, n_test_groups=2, label_horizon=5, embargo_size=2)
WalkForwardConfig(n_splits=5, test_size=252, train_size=1260, label_horizon=21,
                  fold_direction="forward", calendar_id="NYSE").model_dump()
```
  Library defaults (`walk_forward.py`): `n_splits=5`, `test_size=None`, `train_size=None` (expanding), `gap=0`, `label_horizon=0`, `embargo_size=None`, `expanding=True`, `calendar=None`, `fold_direction="forward"`; `CombinatorialCV` defaults `n_groups=8`, `n_test_groups=2`, `label_horizon=0`. `WalkForwardConfig` accepts `label_buffer` / `feature_buffer` as aliases for `label_horizon` / `embargo_td`.
- Case-study protocol loading: `from utils.modeling import get_cv_config; get_cv_config("etfs")` → `WalkForwardConfig` from `case_studies/etfs/config/setup.yaml` (8 splits, 10-year rolling train, 1-year test, 21-session label buffer, holdout 2024–2025).
- Data: `from data import load_etfs` (parquet under `ML4T_DATA_PATH/etfs/market/`, fetched once with `python data/etfs/market/download.py`).
- Calendar sessions: `import exchange_calendars as xcals` (`XNYS`, 2014–2025 in the notebook); `from math import comb` for C(N,k).
- Long-short t-stat: `ls.mean() / ls.std() * np.sqrt(len(ls))`. Rank autocorrelation: join month `mi` with `mi+1` on symbol, `rank("average").over("mi")`, `group_by("mi").agg(pl.corr("r_curr", "r_prev"))`, mean. Era bucketing: `pl.when(pl.col("timestamp") < pl.date(2016, 1, 1)).then(pl.lit("2006–2015")).otherwise(pl.lit("2016–2025"))`.
- setup.yaml block skeleton (etfs values; comments give the variants seen across the nine files; block paths verified against the checkout):
```yaml
strategy_id: etfs
setup_version: v1                 # v2 only for us_firm_characteristics
universe:
  n_assets: 100                   # cme_futures: n_products: 30
  assets: [ACWI, AGG, ...]        # or symbols: [...] (crypto, fx, nasdaq100) / product_groups: {...} (cme)
  eligibility_rule: point_in_time_adv_10m_annual   # 4 of 9 declare one
  eligibility_file: eligibility.csv                # etfs only
decision:
  cadence: monthly_month_end      # nasdaq100: bar_frequency: 15_minute + decision_snapshot: bar_close
  snapshot: close                 # sp500_options: entry_cadence: weekly_friday, entry_time: friday_close,
  execution_delay: next_bar_open  #                holding_period_days: 10, hedge_cadence: daily_close
  cadence_by_label: {fwd_ret_5d: weekly_friday_close}
mapping:
  class: long_only_rank_and_rebalance
  position_state_space: long_only # long_short | short_straddle_hedged
  entry_logic: rank_selection_top_n
  sizing: equal_weight
execution: {initial_cash: 100_000, share_type: integer, allocator_lookback: 63}   # no constraints here
costs:
  class: material                 # dominant (nasdaq100, sp500_options)
  model: per_share_plus_spread    # everything below is market-specific; read the study's own file
  per_share: 0.0035
  default_half_spread_usd: 0.02
  spread_convention: half_spread
labels:
  primary: fwd_ret_21d
  buffer: 21D                     # purge for the primary label; sp500_options: 35D, buffer_unit: calendar
  variants: [fwd_ret_5d]
  variant_buffers: {fwd_ret_5d: 5D}
  rebalance_step: {fwd_ret_21d: 1, fwd_ret_5d: 1}   # one entry per label, required, never inferred
  # horizons: {...} and classification_eval_label: {fwd_dir_15m: fwd_ret_15m} where classification labels exist
backtest:
  rebalance: {default: {min_weight_change: 0.005, min_trade_value: 100.0}}
  sweep:
    top_k_grid: {fwd_ret_21d: [5, 10, 20]}   # quantile_grid (crypto); cadence_sweep: [15_minute, 30_minute, 1_hour, 4_hour]
                                             # + signal_nasdaq100.bars_per_day_grid: [210] (nasdaq100); htm_cost_cascade.top_k: 20 (sp500_options)
evaluation:
  n_splits: 8
  train_size: 10Y                 # also 2Y, 6M, 8Y, P5Y, 10YE
  val_size: 1Y                    # P1Y (fx_pairs), 6M (nasdaq100), 1YE (us_firm_characteristics)
  holdout_start: '2024-01-01'
  holdout_end: '2025-12-31'
  calendar: NYSE                  # CME, FX, crypto, null
  periods_per_year: 252           # 365 crypto, 12 us_firm_characteristics
features: {regime_threshold: 0.005}            # etfs (10y-2y yield spread regime); crypto: bar_hours: 8
modeling: {gbm: {libraries: [lightgbm], preset: default, device: cpu, max_bin: 255}}   # latent_factors.model_kwargs where used
causal: {method: walk_forward_dml, treatment: skip_recent_6_1, confounders: [vol_21d, vol_126d, regime, yield_curve_slope]}
```
- Notebook 03 helpers: `_normalize_setup_yaml(case_id, cfg)` turns a yaml into a summary / costs / diagnostics / techniques dict (reads `ev["train_size"]`, `ev["val_size"]`, `ev["n_splits"]`, holdout dates, `costs["class"]`, `mapping["class"]`, `mapping["entry_logic"]`); `load_setup_results()`; `print_costs(block, indent)`; `compute_coverage(results)` and `plot_coverage(coverage_data)` (Figure 6.5); `freq_map` normalises cadence strings; static maps `DISPLAY_NAMES`, `CHAPTER_TRACKS`, `_infer_asset_class(case_id)`.
- Running: `uv run python 06_strategy_definition/<notebook>.py` from the repo root (Docker image `ml4t`); reduced-data test mode through Papermill: `uv run pytest tests/test_chapter_notebooks.py -v -k "06_strategy_definition"`. Seeds via `utils.reproducibility.set_global_seeds` (`SEED` feeds the sklearn `KFold` anti-pattern demo); plotting via `utils.style` (`COLORS`, `show_with_alt`, `show_plotly_with_alt`). None of the three notebooks writes to disk. `06_strategy_definition/exploration.md` is a 2026-01 codebase analysis of an earlier chapter layout (different notebook set) and was not used for this reference.
- Repo paths: `06_strategy_definition/01_where_ideas_come_from.py`, `02_cv_foundations.py`, `03_case_study_overview.py`; `case_studies/{id}/config/setup.yaml`; `case_studies/*/01_feasibility_analysis.py` applies the protocol per strategy; `case_studies/RUN_LOG.md` and `case_studies/utils/registry/` for the run log (Recipe 7).

## Evidence from the book
- ETF 12-1 momentum, 100 ETFs, monthly, ~2006–2025: gross quintiles sit close to the equal-weight benchmark (mostly beta from a universe that drifted up); net of market the lower buckets are negative and the upper positive, ordering roughly with momentum, and the tilt is top-concentrated ("worth engineering, not enough to trade on its own"); the top-minus-bottom spread is small on the full sample and the tradable Q5−Q1 series is **not** statistically significant on the full sample or in either era; the net spread is wide in 2006–2015 and much narrower in 2016–2025, which is the information-edge decay the SLOW/WRONG mechanism predicts, but a coarse two-bucket split makes the reading only suggestive; signal rank autocorrelation is high (persistent ranking, modest turnover), with zero bps of cost subtracted. Exact printed numbers are computed at runtime (not in the notes).
- CV mechanics on NYSE 2014–2025: with `train_size=1260` the first two of five rolling folds are clipped to expanding folds and only folds 3–5 hold the full window; a 21-session purge spans ~30 calendar days; a 21-calendar-day purge leaks 7 sessions; CPCV with N=6, k=2 gives 15 splits and 5 paths, and the demo buffers 5 + 2 samples at each boundary against blocks of 84 (`N_VIZ = 504` split into 6 blocks).
- Case-study landscape: 7 material-cost vs 2 dominant-cost studies; equities dominate (three pure equity studies plus the sp500_equity_option_analytics hybrid); FX has the tightest quoted spreads (single-digit bps per leg, majors tighter than crosses), which is what allows daily decisions; option spreads quoted against premium make cost the binding constraint; universe sizes 19 to 3,199; cadences 15-minute to weekly; every case study reserves a holdout ("non-negotiable") and declares `causal.method: walk_forward_dml`; dominant-cost studies carry the shorter tracks (nasdaq100 to Ch12; sp500_options is the exception at Ch21), material-cost ones run to Ch14, Ch17 or Ch21.

## Related references
- `chapters/01_process_is_edge.md` — the evidence boundary and workflow this chapter operationalizes.
- `chapters/07_defining_the_learning_task.md` — label construction, overlap and effective sample size, IC inference, multiple testing after the search is counted.
- `chapters/08_financial_features.md` — admissible-information rules for features (publication lags, train-only fitting).
- `chapters/16_strategy_simulation.md` — backtest protocol, Deflated Sharpe, PBO on CPCV paths.
- `chapters/17_portfolio_construction.md` — allocator comparison under the same protocol.
- `chapters/18_transaction_costs.md` — cost classes and horizon feasibility.
- `chapters/20_strategy_synthesis.md` — what the nine protocols produced end to end.
- `case_studies/etfs.md` and the other eight case-study files — the concrete setup contracts.
- `libraries/ml4t_diagnostic.md` — `WalkForwardCV`, `CombinatorialCV`, `WalkForwardConfig`.
- `companion_repo.md` — section "case_studies/research package and the run-log registry" (registry query API, hashed identity inputs, holdout lock), "Configuration system" (`setup.yaml` schema), `scripts/download_artifacts.py`, `utils.modeling.get_cv_config`.
- `workflow.md`, `guardrails.md`, `decision_rules.md` — cross-cutting views.
- Further reading: López de Prado (2018) *Advances in Financial Machine Learning* ch. 7 (purging, embargo, CPCV); Bailey & López de Prado (2014) Deflated Sharpe Ratio; Bailey, Borwein, López de Prado & Zhu (2015) Probability of Backtest Overfitting; Bergmeir et al. (2018) and Kohavi (1995) on CV validity; Bates et al. (2021) on what CV estimates; Jegadeesh & Titman (1993), Asness et al. (2013) Value and Momentum Everywhere, Moskowitz et al. (2011) Time Series Momentum, Hurst et al. A Century of Evidence on Trend-Following, Daniel & Moskowitz (2016) Momentum Crashes, McLean & Pontiff (2016); Paleologo (2025) *Elements of Quantitative Investing*.

## Glossary
- **Decision-time admissibility** — only information available at decision time t may enter training for predictions at t.
- **Strategy family** — structural class of a strategy; a feasibility filter.
- **Source of edge (SLOW / WRONG / RISK)** — economic reason a strategy pays; a durability filter.
- **Trading setup** — versioned bundle of tradability rules, decision schedule, admissible information, score-to-trade mapping, constraints and material costs.
- **Mechanics change** — alteration of any setup invariant; a new setup version, not a tuned trial.
- **Model / signal / strategy diagnostics** — fit quality / ordering of forward returns / P&L after mapping and costs.
- **Label buffer (purge)** — exclusion of training samples within one label horizon before validation start.
- **Feature buffer (embargo)** — exclusion of training samples within one feature lookback after a validation block.
- **Walk-forward CV** — chronological folds, training precedes validation; expanding or rolling.
- **Nested walk-forward** — outer loop over test windows, inner walk-forward for hyperparameter selection.
- **CPCV** — combinatorial purged cross-validation: C(N,k) splits assembled into k·C(N,k)/N complete backtest paths.
- **PBO** — probability of backtest overfitting: share of paths whose in-sample rank fails out of sample.
- **Holdout** — sealed final-confirmation period, opened once.
- **12-1 momentum** — return from 12 months ago to 1 month ago.
- **Net of market** — bucket return minus equal-weight universe return; beta removed, costs not deducted.
- **Rank autocorrelation** — period-over-period Spearman correlation of signal ranks; a turnover proxy.
- **Era stress test** — rerun the sort on time-split subsamples to separate footprint from artifact.
- **Cost class** — dominant (costs first-order) vs material (costs matter but do not preclude trading).
- **Trial taxonomy** — strategy > trial family > trial > run; the unit of countable search. A trial family is one hypothesis with variants; a run is one fitted configuration.
- **Baseline checkpoint** — narrow initial setup validated with timing, coverage and trading-intensity checks.
- **Run log / registry** — the companion repo's content-addressed experiment archive (`case_studies/{id}/run_log/`, a released artifact): SQLite index plus training / prediction / backtest artifacts, each keyed by a SHA-256 hash of its resolved configuration.
- **Prediction coverage** — the fraction of the eligible `(symbol, timestamp, fold)` rows for which a prediction exists; part of the run identity in the companion registry, and the second baseline check.
- **Chapter track** — the range of book chapters a case study is carried through (Ch6–Ch12 to Ch6–Ch21); tells you which downstream references apply to it.
