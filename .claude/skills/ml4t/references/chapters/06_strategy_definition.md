# Chapter 6: Strategy Research Framework
> A trading strategy is an executable decision process, not a signal or a model: it has a decision schedule, an admissible information set, a score-to-position mapping, constraints and costs, and it must be defined at decision time and evaluated as if it were live. This chapter is the methodological spine of the book: place the idea on a strategy map (family = feasibility filter, source of edge = durability filter), freeze a versioned trading setup, define "better" economically with model, signal and strategy diagnostics kept apart, run a walk-forward evaluation protocol with label purge, feature embargo and a sealed holdout, establish a narrow baseline before widening the search, and log every trial so the search stays countable. The nine case studies (`case_studies/*/config/setup.yaml`) are the concrete instances of that protocol.

## When to use this reference
- Starting any new strategy idea: you need to name its family, source of edge, counterparty and failure mode before touching data.
- Writing or reviewing a trading setup: universe rules, decision cadence, admissible information, mapping from scores to positions, constraints, costs.
- Designing or auditing a train/validation/holdout split for time series (walk-forward, purge, embargo, nested walk-forward, CPCV).
- Deciding whether a change is "tuning" or a new strategy version.
- Setting up run logging / trial accounting so later significance claims can be deflated by the number of trials.
- Reading or writing a `setup.yaml`-style configuration (`evaluation`, `costs`, `labels`, `universe` blocks).
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

```python
mom = (pl.col("close").shift(SKIP) / pl.col("close").shift(LOOKBACK) - 1).over("symbol")
fwd = (pl.col("close").shift(-1) / pl.col("close") - 1).over("symbol")
elig = df.filter(pl.len().over("timestamp") >= MIN_NAMES)
rank = pl.col("momentum").rank("ordinal").over("timestamp")   # sort by (timestamp, momentum, symbol) first
```
- Cut ranks at `1 + k(n-1)/5` (balanced buckets; ties never merge buckets); Q1 = worst past year, Q5 = best.
- Report gross (%/month) and net of benchmark (bucket minus EW universe, bps/month). Net of benchmark strips beta; it is **not** net of trading costs.
- Spread = Q5 − Q1; tradable long-short series `Q5 − Q1`; `ls_t = mean / std * sqrt(n_months)`. Read the t-stat before the spread.
- Read the sort for monotonicity (all buckets order), magnitude (economically material), shape (smooth premium-like vs tail-concentrated event-like).
- Era stress test: split at the boundary, report Q1_net, Q5_net, spread, t, months per era; move the boundary to test the stability of any fade.
- Turnover proxy: month-over-month Spearman rank autocorrelation of the signal (join adjacent months on symbol, keep pairs with ≥ `MIN_NAMES` survivors, `rank("average")`, `pl.corr` per pair, average). High autocorrelation → persistent ranking → modest turnover and costs.

### Recipe 3: Freeze the trading setup (the invariants, §6.3)
Write down and version, before the first model run:

| Invariant | What to state | setup.yaml keys (companion repo) |
|---|---|---|
| Tradable universe and eligibility | per-period membership rule, floors such as `MIN_NAMES`, point-in-time eligibility file | `universe.assets`, `universe.eligibility_rule`, `eligibility_file` |
| Decision schedule | cadence, decision snapshot (`close`, `bar_close`), execution delay, rebalance step | `decision.cadence`, `decision.snapshot`, `decision.execution_delay`, `labels.rebalance_step`, `cadence_by_label` |
| Admissible information | publication lags, rolling-only transforms and thresholds | feature/label construction (Ch7, Ch8) |
| Score-to-position mapping | setup class (`long_only_rank_and_rebalance`, `long_short_top_k_rebalance`, `long_short_decile_rebalance`, `systematic_straddle_sell`, `intraday_rank_and_trade`), `top_k`, quantiles | setup `class`, `top_k_grid`, `quantile_grid` |
| Constraints | long-only vs long-short, position limits, leverage | `execution.*`, allocator config |
| Material cost components | in the market's own units: bps per leg, ticks, per-share plus half-spread, fraction of option premium | `costs.class` (`material` / `dominant`), `costs.model`, `spread_convention`, `spread_bps`, `commission_per_contract`, `per_share`, `default_half_spread_usd` |
| Evaluation protocol | folds, train/val windows, label horizon, calendar, holdout dates | `evaluation.n_splits`, `train_size`, `val_size`, `holdout_start`, `holdout_end`, `calendar` |

Any later change to tradability, schedule, mapping, constraints or cost treatment bumps `setup_version`; a parameter change inside the invariants does not.

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
2. Choose expanding (older data still relevant) or rolling (regimes change) walk-forward. With a rolling window longer than early history, early folds are silently clipped to expanding folds; check each fold's actual train length.
3. Purge: training end ≤ validation start − h, with h the label horizon counted in **trading sessions** of the market calendar. 21 sessions ≈ 30 calendar days; a 21-*calendar*-day purge removes only ~14 sessions and leaks ~7. Check `purged_sessions = va[0] - tr[-1] - 1 == h` (the index gap is h + 1).
4. Embargo: when training can follow a validation block (CPCV, k-fold), drop training samples within one feature lookback after the block. Zero embargo is correct only in pure walk-forward.
5. Nested walk-forward when hyperparameters are tuned: inner walk-forward selects λ* on the pre-test years, outer loop advances the test year and re-tunes. If λ* differs across outer folds, the hyperparameter is unstable and a single tuned value should not go to production.
6. CPCV for robustness: N contiguous blocks, hold out k, purge/embargo at every boundary, assemble validation predictions into complete paths. N=6, k=2 → C(6,2)=15 splits, k·C(N,k)/N = 5 paths, each block validated 5 times. Similar Sharpe across paths → robust; wide variance → path-dependent/overfit. PBO = fraction of paths whose in-sample rank fails out of sample (full treatment Ch16/17 references).

| Method | Paths | Label buffer | Feature buffer | Use |
|---|---|---|---|---|
| Walk-forward | 1 | yes | no (train precedes val) | standard evaluation |
| Nested walk-forward | 1 per test year | yes | no | multi-year test with retuning |
| CPCV | multiple | yes | yes | robustness, PBO |

### Recipe 6: Baseline checkpoint (§6.6)
Before widening features or model class, run a deliberately narrow configuration and pass three checks: **timing** (decisions and fills on the right bars, no lookahead), **coverage** (predictions exist for the intended universe and periods; cf. the prediction-coverage timeline of `03_case_study_overview`), **trading intensity** (turnover consistent with the cost class and horizon).

### Recipe 7: Search accounting and run logging (§6.7)
- Taxonomy: strategy > trial family > trial > run. A trial family is one hypothesis with variants; a run is one fitted configuration.
- Log automatically: configuration (content-addressed hash), data vintage, fold geometry, metrics per fold, selection decisions, what was reserved for confirmation. The companion repo's run log (`case_studies/{id}/run_log/registry.db`, see `companion_repo.md`) is the reference implementation.
- Report search size alongside results so Deflated Sharpe / BH-FDR / PBO can be computed from the real number of trials (Ch7, Ch16 references).

## Guardrails and pitfalls
- **Label leakage at the fold boundary** — forward labels of horizon h near the boundary embed validation prices and inflate validation scores / purge h sessions (`label_horizon` in `WalkForwardCV` / `CombinatorialCV`); assert `va[0] - tr[-1] - 1 == h`.
- **Calendar-day purge** — 21 calendar days ≈ 14 sessions, ~7 sessions still leak / pass the market calendar (`calendar="XNYS"`, CME, FX, crypto) so buffers count sessions.
- **Missing embargo in CPCV / k-fold** — backward-looking features of post-validation training rows consume validation prices / `embargo_size ≥ feature lookback`; zero only in pure walk-forward.
- **Shuffled k-fold on time series** — trains on the future to predict the past / chronological walk-forward only.
- **Standardization, threshold, survivorship, point-in-time leakage** — splitters do not prevent these / training-window transforms, rolling thresholds, per-period universes, publication lags (Ch2, Ch4, Ch7, Ch8).
- **Holdout contamination** — any development decision made with the holdout destroys the unbiased estimate / set `holdout_start`/`holdout_end` at the start; open once, for confirmation only.
- **Single-path luck** — one walk-forward path can be a lucky sequence / CPCV paths, PBO.
- **Hyperparameter staleness** — λ tuned once on old years may be wrong later / nested walk-forward; require stability across outer folds.
- **Uncounted search** — the best of many trials is inflated / trial taxonomy + automatic logging; deflate by trials tried, not trials reported.
- **Optimizing on portfolio outcomes** — micro-tuning on simulated P&L overfits / tune on model and signal diagnostics, confirm on strategy outcomes.
- **Mechanics drift masquerading as tuning** — silently changing schedule, mapping, constraints or costs breaks comparability / version the setup; classify every change.
- **Beta mistaken for signal** — gross quintile returns all sit near the EW benchmark / report net-of-benchmark alongside gross.
- **"Net" ambiguity** — net of benchmark is not net of costs / say which; defer cost deduction to the setup's cost model.
- **Spurious significance in a lumpy sample** — the ETF long-short t-stat is not significant in the full sample or either era / print t before spread; era split; account for overlap and effective sample size (Ch7).
- **Annualizing a monthly sort** — compounding assumes the edge recurs every month / report in the design's horizon.
- **Edge decay and momentum crashes** — price-based edges narrow as they become known; momentum reverses at turning points / era stress tests; name the failure mode in advance; never rely on full-sample averages.
- **Short-term reversal contamination** — the latest month carries the opposite-signed liquidity-provision effect / `SKIP = 1`.
- **Unstable ranks** — run-to-run tie-breaking makes quintiles non-reproducible / sort by `(timestamp, signal, symbol)` and use ordinal ranks.
- **Thin cross-sections** — few names → noisy buckets / `MIN_NAMES` floor per period.
- **Turnover blindness** — a signal that re-ranks every period is erased by costs / measure rank autocorrelation before modeling; declare cost components in the setup.
- **Dominant-cost regimes** — at 15-minute cadence or in options (spread vs premium) friction is first-order / require unusually strong signals or longer holding periods.
- **Rolling window longer than history** — early folds are clipped to expanding without warning / check each fold's train length.
- **Horizon mismatch** — purge gap ≠ label horizon leaves leakage / `label_horizon` = forward-return horizon.
- **Baseline skipped** — broad searches on an invalid setup waste effort and inflate trial counts / baseline checkpoint first.
- **Dataset heterogeneity** — long histories span regimes; short ones (crypto, intraday) limit fold depth / match train window and fold count to history (6M–10Y, 2–16 folds) and state stationarity assumptions.

## Decision rules and defaults
| Decision | Rule / default | Notes |
|---|---|---|
| Idea go/no-go before modeling | monotonic sort, material spread, read t before spread, survives an era split, high rank autocorrelation | "lumpy but present" passes; "only in one period" fails |
| Momentum footprint defaults | LOOKBACK 12, SKIP 1, MIN_NAMES 30, monthly, quintiles, EW buckets, EW benchmark, era boundary 2016-01-01 | ETF example |
| Dataset roles | train / validation / holdout; holdout opened once | non-negotiable in every case study |
| Walk-forward example defaults | `n_splits=5`, `test_size=252`, `train_size=1260` (rolling), `label_horizon=21`, `calendar="XNYS"`, `fold_direction="forward"` | from `02_cv_foundations` |
| Expanding vs rolling | expanding if old data still relevant; rolling if regimes change | check clipped early folds |
| Purge | = label horizon, in sessions | 21 sessions for 1-month labels |
| Embargo | = feature lookback; only when training can follow validation | CPCV, k-fold |
| Nested walk-forward | whenever hyperparameters are tuned and the test spans several years | carry λ forward only if stable across outer folds |
| CPCV | N=6, k=2 → 15 splits, 5 paths | robustness and PBO |
| Cost class | dominant → need exceptionally strong signals, longer horizons; material → choose horizon by signal decay vs cost hurdle | NASDAQ-100 intraday and S&P 500 options are dominant; the other seven material |
| Case-study protocol ranges | train 6M (intraday) to 10Y (ETFs, panels); folds 2 to 16; holdout 1 year (options, firm characteristics) to 2 years (2024–2025) | see `case_studies/*.md` |
| Setup versioning | any change to tradability, schedule, mapping, constraints or material costs = new `setup_version` | parameter tuning does not |
| Sequencing | EDA footprint → freeze setup → economic objective + metric roles → leak-free walk-forward with sealed holdout → narrow baseline (timing, coverage, intensity) → logged, countable search → Ch7 label/overlap/N_eff checks | |

Case-study protocols as recorded in `setup.yaml` (names, cadence, setup class, cost class, folds, train window, holdout, calendar):

| Case study | Cadence | Setup class | Costs | Folds / train | Holdout | Calendar |
|---|---|---|---|---|---|---|
| etfs (100) | monthly month-end | long_only_rank_and_rebalance | material, per_share_plus_spread (half spread) | 8 / 10Y (1Y val) | 2024-01-01..2025-12-31 | NYSE |
| crypto_perps_funding (19) | 8-hour funding-aligned | long_short_funding_aligned | material | 2 / 2Y | 2024..2025 | crypto |
| nasdaq100_microstructure (114) | 15-minute bar close | intraday_rank_and_trade | dominant, per_share_plus_spread | 2 / 6M | 2021-07-01..2021-12-31 | NYSE |
| sp500_equity_option_analytics (~630) | weekly Friday close | long_only_rank_and_rebalance | material, percentage bps | 2 / 2Y | 2021 | NYSE |
| us_firm_characteristics (~2,500; setup v2) | monthly | long_short_top_k_rebalance | material | 10 / 10YE | 2016 | none (monthly) |
| fx_pairs (20) | daily NY close | long_short_rank_rebalance | material, single-digit bps per leg | 8 / 5Y | 2024..2025 | FX |
| cme_futures (30) | weekly Friday close | long_short_carry_rank | material, $2 per contract + spread ticks | 5 / 8Y | 2024..2025 | CME |
| sp500_options | daily | systematic_straddle_sell, top_k 20 | dominant, spread vs premium | 2 / 2Y | 2021 | NYSE |
| us_equities_panel (3,199) | daily close | long_short_decile_rebalance | material, percentage | 16 / 10Y | 2016-01-01..2018-03-31 | NYSE |

## Code patterns and APIs
- Splitters: `from ml4t.diagnostic.splitters import WalkForwardCV, CombinatorialCV`; `from ml4t.diagnostic.splitters.config import WalkForwardConfig`.
```python
cv = WalkForwardCV(n_splits=5, test_size=252, train_size=1260, expanding=False,
                   label_horizon=21, calendar="XNYS")
for train_idx, val_idx in cv.split(df_dates): ...
CombinatorialCV(n_groups=6, n_test_groups=2, label_horizon=5, embargo_size=2)
WalkForwardConfig(n_splits=5, test_size=252, train_size=1260, label_horizon=21,
                  fold_direction="forward", calendar_id="NYSE").model_dump()
```
- Case-study protocol loading: `from utils.modeling import get_cv_config; get_cv_config("etfs")` → `WalkForwardConfig` from `case_studies/etfs/config/setup.yaml` (8 splits, 10-year rolling train, 1-year test, 21-session label buffer, holdout 2024–2025).
- Data: `from data import load_etfs` (parquet under `ML4T_DATA_PATH/etfs/market/`).
- Calendar sessions: `import exchange_calendars as xcals` (`XNYS`).
- Long-short t-stat: `ls.mean() / ls.std() * np.sqrt(len(ls))`.
- setup.yaml evaluation block:
```yaml
evaluation:
  n_splits: 8
  train_size: 10Y        # also 2Y, 6M, 8Y, P5Y, 10YE
  val_size: 1Y
  holdout_start: '2024-01-01'
  holdout_end: '2025-12-31'
  calendar: NYSE         # CME, FX, crypto, null
```
- Repo paths: `06_strategy_definition/01_where_ideas_come_from.py`, `02_cv_foundations.py`, `03_case_study_overview.py`; `case_studies/*/01_feasibility_analysis.py` applies the protocol per strategy; tests: `uv run pytest tests/test_chapter_notebooks.py -k 06_strategy_definition`.

## Evidence from the book
- ETF 12-1 momentum, 100 ETFs, monthly, ~2006–2025: gross quintiles sit close to the equal-weight benchmark (mostly beta); net of market the buckets order roughly with momentum and the tilt is top-concentrated ("worth engineering, not enough to trade on its own"); the tradable Q5−Q1 series is **not** statistically significant on the full sample or in either era; the net spread is wide in 2006–2015 and much narrower in 2016–2025, consistent with information-edge decay; signal rank autocorrelation is high (persistent ranking, modest turnover). Exact printed numbers are computed at runtime (not in the notes).
- CV mechanics on NYSE 2014–2025: with `train_size=1260` the first two of five rolling folds are clipped to expanding folds; a 21-session purge spans ~30 calendar days; a 21-calendar-day purge leaks 7 sessions; CPCV with N=6, k=2 gives 15 splits and 5 paths, buffered 5+2 samples per boundary against 84-sample blocks.
- Case-study landscape: 7 material-cost vs 2 dominant-cost studies; FX has the tightest spreads (single-digit bps per leg), which is what allows daily decisions; option spreads quoted against premium make cost the binding constraint; universe sizes 19 to 3,199; cadences 15-minute to weekly; every case study reserves a holdout and uses `method: walk_forward_dml` in its modeling block.

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
- `companion_repo.md` — run log / registry, `setup.yaml` schema, `utils.modeling.get_cv_config`.
- `workflow.md`, `guardrails.md`, `decision_rules.md` — cross-cutting views.
- Further reading: López de Prado (2018) *Advances in Financial Machine Learning* ch. 7 (purging, embargo, CPCV); Bailey & López de Prado (2014) Deflated Sharpe Ratio; Bailey, Borwein, López de Prado & Zhu (2015) Probability of Backtest Overfitting; Bergmeir et al. (2018) and Kohavi (1995) on CV validity; Bates et al. (2021) on what CV estimates; Jegadeesh & Titman (1993), Asness et al. (2013), Moskowitz et al. (2011), Daniel & Moskowitz (2016), McLean & Pontiff (2016); Paleologo (2025) *Elements of Quantitative Investing*.

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
- **Trial taxonomy** — strategy > trial family > trial > run; the unit of countable search.
- **Baseline checkpoint** — narrow initial setup validated with timing, coverage and trading-intensity checks.
