# Chapter 26: MLOps and Governance

> Chapter 26 covers what happens after Chapter 25 takes a model live: every deployed model decays (regimes shift, competitors find the same signals, learned relationships erode), so profitability depends on how fast decay is detected and how safely the system responds. It argues for a three-layer governance model — detection (rolling metrics, PSI/K-S drift diagnostics, online ADWIN-style and DDM detectors), response (shadow mode, capital-capped A/B, staged rollout behind a promotion gate fixed before evaluation), and automated safety (multi-level circuit breakers with a closed/open/half-open lifecycle) — built on an MLOps stack (feature stores, content-addressed registries, experiment tracking, CI/CD) that is right-sized to team maturity, not adopted wholesale. The organizing diagnostic is the failure taxonomy of §26.1: technical failures (same inputs, different outputs) and statistical failures (same outputs, no longer predictive) need different evidence and different responses, and alert tables are escalation lists, never retraining triggers. All seven notebooks run on real `us_equities_panel` artifacts (holdout boundary, stored prediction streams, feature panels, SQLite registry); only the latency stream in `04_circuit_breakers` is synthetic.

## When to use this reference

- Building or reviewing a live-monitoring dashboard for a deployed model (PSI per feature, prediction-score density, rolling IC / hit rate / MSE) and choosing alert thresholds.
- Deciding whether a degraded status means "data broke", "market moved" or "model decayed", and what to do about it (never retrain from an alert directly).
- Designing sequential/online drift detectors (two-window mean shift, DDM-style bad-day monitor), their calibration windows, cooldowns, and how to read alerts against a market turbulence proxy.
- Evaluating a candidate model against an incumbent in shadow mode and writing a promotion gate (minimum Sharpe improvement, observation window, signal correlation, position agreement, drawdown ratio) before looking at results.
- Implementing circuit breakers (drawdown, daily loss, consecutive losses, latency, or any new halt rule) with half-open recovery, a `BreakerManager`, and a full transition log.
- Setting up a feature store (by hand in Polars or with Feast): feature views, entity key, event timestamp, TTL, point-in-time offline join with holdout guard, as-of online retrieval, training-serving skew measurement, parity checks.
- Turning a case-study run-log registry (`training_runs` -> `prediction_sets` -> `backtest_runs`) into a searchable experiment catalog, tracing a run back to its artifacts, and logging it to MLflow with ranking-parity assertions.
- Reading a registry safely (read-only URIs, config names not hashes, fold geometry derived rather than typed) in any notebook that reviews stored results.
- Separating a ranking metric (validation IC) from the selection metric (validation backtest Sharpe at `stage='signal'`) and accounting for unequal experiment budgets across model families.
- Auditing any post-deployment workflow for the failure modes catalogued here (validation-set monitoring, timer-closed breakers, `>` vs `<=` as-of joins, missing TTL, pooled rank correlations, top-N sampling bias).

## Core ideas (the why)

- **Failure taxonomy (§26.1).** Technical failure = verification problem (same inputs produce different outputs; pipeline divergence). Statistical failure = performance problem (same outputs no longer predict returns; decay). Conflating them wastes time and capital; each needs its own diagnostic and response workflow.
- **Three candidate causes for a degraded status** — data broke, market moved, model decayed — need different responses and are told apart by different evidence. Retraining on a broken feed fits the model to the break. Alert tables and detector queues are evidence packages / escalation lists, never retraining triggers.
- **Validate before measuring.** A broken join or null-filled column produces the same distribution shift a regime change does, and the two call for opposite responses (fix the feed vs retrain).
- **PSI and K-S answer different questions.** PSI says how much a distribution moved (no null distribution; thresholds are conventions, not significance levels). K-S says whether the gap exceeds sampling noise (has a null, but at tens of thousands of daily observations rejects on differences too small to act on). Read together, never either alone.
- **Distribution drift and performance decay are separate measurements** that move independently: track input distributions, output distribution and rolling signal quality, and read disagreements between them.
- **Marginal (one-feature-at-a-time) comparisons bound what disagreement can mean.** Features moving while predictions stay still is consistent with light weighting or offsetting moves. Predictions moving while features stay still is the more urgent direction: either something outside the monitored set moved, or the relationships between features changed (invisible to marginal tests).
- **Measure drift only over sessions the model was scored on out of sample** (walk-forward validation windows + holdout). Comparing against training sessions measures the training set. Monitoring a validation set "would measure the window the model was chosen to fit and call the result drift" — only a holdout prediction set will do for the dashboard.
- **Sequential detectors must be calibrated on a period they are not judged over**; calibrating on the monitored data lets the detector tune itself to the shift it is meant to find.
- **Choose the detector for the failure you care about.** A two-window mean-shift test finds errors that grow in size; a bad-day frequency monitor finds a model that is wrong slightly more often. Neither is more sensitive in general; more alerts means more sensitive, not better.
- **An alert is not a degradation label.** An independent market measure (turbulence proxy) is what makes "model broke" vs "market moved" distinguishable at all. An alert cluster on a turbulent stretch is a coincidence in time, not a cause; it says check sizing before checking the model.
- **Shadow mode is the cheapest evaluation on unseen data and is paid for in time**; the window must be long enough for statistics to mean something. Holding a candidate in shadow is a complete outcome, not a deferral.
- **Promotion rules must be written down before evaluation.** A threshold chosen after seeing the result is not a threshold. Roles (incumbent/candidate) are assigned before any statistic: "a challenger picked after looking at the numbers is not a challenger, it is a result."
- **The gate is a conjunction of five criteria, not a score.** Any single criterion can be met by a model that should not be deployed: Sharpe bought with risk; Sharpe inside estimation noise; a genuine improvement from a strategy that is no longer the one the desk signed off on. Replacing with a different strategy is a decision for a person, not a Sharpe comparison.
- **A promotion gate does not ask whether the incumbent should be running at all**; that is a separate question the gate is the wrong instrument for.
- **Circuit breakers: switching off is automatic, switching back on is not.** Half-open is a trial, not a verdict. A breaker that closes on a timer resumes trading into whatever stopped it; one requiring manual close needs a person awake.
- **Simple conditions that fail in different circumstances beat one elaborate rule**: single shock -> daily loss; slow bleed -> drawdown; broken signal -> streak; degraded system -> latency. The infrastructure breaker must share the control plane: a slow system trades on stale prices, which no market-risk rule sees.
- **Log every transition, not every trip.** Half-open probes and closes are what let an operator reconstruct a sequence afterwards.
- **A feature store answers one question in two places** (what were this entity's features at this moment: millions of past moments for training, one moment now for serving). When the two answers come from different code they drift apart — training-serving skew, the failure the component exists to prevent.
- **A feature store is configured, not written.** A wrong timestamp field, entity key or TTL produces a store that runs, answers every query, and answers wrongly. Check it against an implementation whose rules you can read. The library buys shared infrastructure (registry, online store, multi-team isolation), not correctness.
- **Guards should refuse rather than degrade quietly** (holdout boundary guard, book-construction checks, coverage checks, stream-difference checks).
- **Pin configuration names, not prediction hashes.** Hashes are content-addressed and move on every refit; a notebook pinning a hash stops running the next time anything upstream changes.
- **Derive fold-dependent dates from the case study's own geometry rather than typing them.** A typed date is a claim about where a fold boundary falls and lands silently in the wrong place after a rebuild.
- **An experiment tracker answers three questions after the fact**: what was run, what came out, can it be reproduced. A metrics table cannot answer the third; a row that resolves to its specification and artifacts is "the difference between a record and a claim". A run log must carry five things: provenance, data and evaluation protocol, configuration, artifacts, decision gates applied.
- **Content-addressed identifiers make the log idempotent.** Run id = hash of canonical spec; re-running lands on the existing row or produces a different hash — "there is no third case where two rows describe the same run". Configuration changes appear as new identifiers, never edits.
- **The catalog ranks; selection happens elsewhere.** Ordering by IC describes the predictions; the pipeline selects on validation backtest Sharpe because a model can rank well and still produce a portfolio nobody would hold once turnover and costs are charged. A tracker that presents one ordering "invites it to become the other".
- **Record the experiment budget, not only the results.** The maximum of more draws is higher even when nothing improved; a best-per-family table hides how many draws each family had. Scores within a family are not independent draws (shared features, folds, fitted state); the score distribution says what to check, the check is a held-out evaluation, and the holdout is spent once.
- **Tool vs discipline.** MLflow supplies interface (`log_params`/`log_metrics`/`log_artifact`/`search_runs`, web UI) and infrastructure (persistent shared store, multi-user, access control). Trust — deterministic hashing, a manifest that resolves, a recorded specification — comes from the pipeline that writes the runs.

## Method recipes (the how)

### Failure triage order (§26.1-26.2)

1. Check data integrity first: run `validate_feature_data` / `DataFrameValidator` on the feature panel before computing any statistic (notebook `26_mlops_governance/01_drift_monitoring`).
2. Check market conditions second: place alerts against the turbulence proxy (`02_online_drift_detection`).
3. Only then consider model decay; act through the shadow/promotion workflow (`03_safe_model_rollout`) and only after data integrity is confirmed.
4. Never retrain directly from an alert.

### Drift dashboard: PSI, K-S and rolling metrics (`01_drift_monitoring`, §26.2-26.3)

Four panels (Figure 26.2): PSI per feature; launch-vs-latest prediction-score density; rolling IC; rolling hit rate. Requires a holdout prediction set (the registry "does not yet hold" one at time of writing; the notebook stops if the requested label lacks one).

**PSI** — `compute_psi(reference, current, n_bins=10, epsilon=1e-6) -> (psi, bin_psi)`:

```python
inner = np.linspace(ref.min(), ref.max(), n_bins + 1)[1:-1]   # bins from the reference only
edges = np.concatenate([[-np.inf], inner, [np.inf]])
ref_pct = np.histogram(ref, edges)[0] / len(ref) + eps
cur_pct = np.histogram(cur, edges)[0] / len(cur) + eps
psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
```

**K-S** — SciPy two-sample Kolmogorov-Smirnov: largest gap between empirical CDFs, p-value under the single-distribution null.

**Status rules**

| Series | ALERT | WATCH | OK |
|---|---|---|---|
| Feature | `psi >= 0.25` | `psi >= 0.10` OR (`psi >= 0.02` AND `ks_pvalue < 0.05`) | otherwise |
| Prediction score | `psi >= 0.25` | `psi >= 0.10` (no K-S term) | otherwise |
| Rolling IC / hit rate | absolute drop from launch baseline `>= 0.01` | drop `>= 0.005` | otherwise |
| Rolling MSE | relative increase `>= 10%` | `>= 5%` | otherwise |

- Reference for features = model's last validation window (`LAST_VALIDATION_SPAN = max(VALIDATION_SPANS)`); current = most recent `LOOKBACK_DAYS=63` sessions inside holdout.
- Reference for predictions = first 63 holdout sessions (predictions exist only inside holdout). Only the direction of the two movements is comparable, not the size.
- Rolling performance: launch baseline = first 63 holdout sessions; subsequent rolling 63-session windows; columns `rolling_ic_63`, `rolling_hit_rate_63`, `rolling_mse_63`; `MIN_ROLLING_SESSIONS = LOOKBACK_DAYS // 3 = 21`.
- IC = per-session Spearman rank correlation between scores and realized returns (what a long-short book uses); hit rate = share of names whose direction was called right.
- Use absolute drops for IC and hit rate (a relative change on an IC near zero is meaningless) and relative increases for MSE (squared error has no natural scale). The alert table shows both bounds per row because prediction-PSI bounds are fixed conventions while rolling-metric bounds are offsets from the metric's own launch baseline.
- Output: `DriftMetric` dataclass (name, psi, ks_stat, ks_pvalue, reference_mean, ..., status); `summarize_feature_drift(reference_frame, current_frame, feature_columns) -> list[DriftMetric]`; dashboard arrays persisted to `get_output_dir(26, "figure_26_2")`, plus feature drift summary and alert table.
- Whether hit rate reuses `IC_WATCH_DROP`/`IC_ALERT_DROP` or has its own constants is unverified (notes uncertainty). Monitored `FEATURE_COLUMNS` are likely the seven features of `05_feast_feature_store` (inference).

### Evaluated-session restriction (used by 01, 05, 05b)

- Walk-forward validation windows + holdout are the only sessions with out-of-sample numbers.
- `evaluation_spans()` returns (validation windows, holdout); `folds_by_coverage(path, windows)` pairs each window with the artifact fold covering it by date; `window_terms(clipped)` builds one predicate per window, OR-ed into a single `pl.DataFrame.filter`.
- `model_based.parquet` may carry a `fold` column (one row per stock-date per fold: apply window and fold together) or not (date range alone selects). Detect with `FOLD_KEYED_ARTIFACT = "fold" in pl.scan_parquet(path).collect_schema().names()`.

### Online sequential detectors (`02_online_drift_detection`, §26.3)

Inputs: one calendar year (2015) of real validation prediction error for `linear/ols` vs `linear/ridge_a1000000.0`, resolved by config name through the registry; liquid universe `get_liquid_universe(top_n=200)` by prior-year average dollar volume; sessions with fewer than `MIN_ASSETS_PER_DATE=10` names dropped. Holdout never read.

**Error dating**: a score formed at session t is scored against session t+1's return; carry the error forward to the session on which the outcome becomes available and index every statistic on that date.

**Two-window mean shift (ADWIN-style)** — `TwoWindowMeanShift(window_size=21, sensitivity=1.4, cooldown_days=21).update(value) -> bool`; input = error magnitude per session:
1. Append value; if `cooldown_remaining > 0`: decrement, return False.
2. Need `>= 2 * window_size` observations; recent = last W, prior = the W before.
3. `pooled = sqrt(var_prior(ddof=1)/W + var_recent(ddof=1)/W + 1e-12)`; `stat = |mean_recent - mean_prior| / pooled` (two-sample t statistic).
4. Alert if `stat >= sensitivity`; set cooldown = `cooldown_days` so one sustained shift yields one alert.

**Bad-day frequency monitor (DDM, Gama et al. 2004)** — `BadDayFrequencyMonitor(min_samples=20, warning_level=1.5, drift_level=2.0).update(error: bool) -> "normal"|"warning"|"drift"`:
1. `n += 1; errors += error`; if `n < min_samples` return "normal".
2. `p = errors/n`; `s = sqrt(max(p(1-p)/n, 1e-12))`; if `p + s < p_min + s_min` then `(p_min, s_min) = (p, s)`.
3. "drift" if `p + s >= p_min + 2.0*s_min`; "warning" if `p + s >= p_min + 1.5*s_min`. `reset()` zeroes n, errors and sets `p_min = s_min = inf`.
4. Because the standard error shrinks with n, the same absolute rise is significant later in the stream (an unusual first month is weaker evidence than the same stretch in month six).

**Bad-day definition**: `threshold = max(calibration["direction_error_rate"].mean() + BAD_DAY_MARGIN (0.01), BAD_DAY_FLOOR (0.52))`; a session is bad when `direction_error_rate > threshold`. The floor stops a model already wrong more often than not during calibration from setting a bar it meets by standing still.

**Calibration**: first `CALIBRATION_DAYS=63` sessions of 2015 feed each detector with no alerts; the remainder is monitored. Reset per model stream via `run_adwin_stream`, `run_ddm_stream`, `run_detectors(daily_errors, stress_starts)`.

**Turbulence proxy**: `rolling_vol_21d` = annualized rolling std (`STRESS_VOL_WINDOW=21`) of the median daily return across the liquid universe; `stress_threshold = calibration_market["rolling_vol_21d"].quantile(0.80)` on the PRIOR year (`CALIBRATION_YEAR=2014`), held fixed through the monitored year.

**Signed lead/lag**: `nearest_stress_lag(alert_dates, stress_dates) -> float | None` — per alert, distance to the nearest stress-episode start in either direction, sign kept: positive = episode came after alert (detector led); negative = alert fired after the episode began. Summary row per (config, detector): `alert_count`, `first_alert`, `median_lag_days`.

**Coverage and difference assertions**: both streams start and end together and cover ~252 sessions (`MIN_MONITORED_SESSIONS=240`); bad-day rates must differ on a meaningful share of sessions (adjacent ridge penalties agree to five decimal places, so use the unpenalized vs heavily penalized pair).

| Detector | Finds | Prefer when |
|---|---|---|
| Two-window mean shift | errors that grow in size | you expect gradual magnitude drift |
| Bad-day frequency monitor | model wrong slightly more often | you expect directional hit-rate erosion |

### Shadow mode and promotion gate (`03_safe_model_rollout`, §26.4)

Sequence: assign roles (INCUMBENT `ols`, CANDIDATE `ridge_a100.0`) -> open registry read-only -> verify prediction parquet on disk -> load both streams over an identical interval -> `validate_coverage` -> build identical books with own-turnover cost -> describe (historical IC 2010-2014, descriptive only) -> score shadow window -> compute signal correlation and position agreement -> apply gate -> read decision off the table.

**Book** — `build_portfolio_stream(predictions, top_k=100, cost_bps)`: each session, `TOP_K` highest scores long at 1/TOP_K, `TOP_K` lowest short at 1/TOP_K (dollar neutral); cost charged on turnover against that model's own previous book; cost = midpoint of the configured per-leg bps range from `setup.yaml` (`load_cost_bps`). Raises on: fewer than `2*TOP_K` ranked names; repeated name within a leg; name in both legs.

**Metrics**: `annualized_sharpe = sqrt(252) * mean / std(ddof=1)` (0.0 if std == 0); `max_drawdown = ((1+r).cumprod() / cummax - 1).min()`; signal correlation = per-session Spearman between the two models' scores, averaged (not pooled); position agreement = |names both hold| / |names either holds| per session, averaged.

**Promotion gate** — `PromotionCriteria` dataclass; all five must hold (shadow window = first `SHADOW_SESSIONS=63` sessions both models cover in ROLLOUT 2015; label `fwd_ret_1d` between adjusted closes):

| # | Criterion | Default | Why |
|---|---|---|---|
| 1 | `sharpe_improvement >= MIN_SHARPE_IMPROVEMENT` | 0.20 | minimum effect, not merely positive: 63-session annualized Sharpe SE ~ sqrt(252/63) ~ 2 |
| 2 | observed sessions `>= MIN_OBSERVATION_SESSIONS` | 63 | statistics need a window |
| 3 | `signal_correlation >= MIN_SIGNAL_CORRELATION` | 0.30 | candidate is still the same kind of signal |
| 4 | `mean_position_agreement >= MIN_POSITION_AGREEMENT` | 0.30 | candidate is still trading the same kind of book |
| 5 | `drawdown_ratio = cand MDD / inc MDD <= MAX_DRAWDOWN_RATIO` | 1.25 | Sharpe not bought with risk |

Go/no-go: promote only if all five rows pass. Any failure = candidate stays in shadow with no capital until it clears or is withdrawn; nothing to stage, nothing to A/B. After the gate: A/B with capped capital, then staged rollout with tested rollback (§26.4; not implemented in the notebook). Thresholds are defensible in shape, not calibrated; a desk sets them from its own promotion history.

### Circuit breakers (`04_circuit_breakers`, §26.5)

States: `CLOSED` (trading), `OPEN` (halted), `HALF_OPEN` (trial). `advance_breaker_state(breaker, event_time, **kwargs)`:
1. If OPEN and `event_time - trip_time >= recovery_timeout` -> HALF_OPEN ("Recovery timeout elapsed"), return (trading permitted this tick BEFORE the condition is checked).
2. Else `should_trip, reason, value, threshold = check_condition(**kwargs)`: trip from HALF_OPEN -> OPEN "Recovery failed: {reason}" (fresh full timeout); trip from CLOSED -> OPEN; no trip and HALF_OPEN -> CLOSED "Recovery successful".
3. `CircuitBreaker(name, recovery_timeout=timedelta(hours=1), on_trip=None, on_transition=None)` with `update()`, `reset()` (manual -> CLOSED), `is_open()`, `allows_trading()`. Concrete breakers override only `check_condition`.

| Breaker | Name | Trip condition | Default limit | Recovery timeout |
|---|---|---|---|---|
| `DrawdownBreaker` | `drawdown_10%` | `DD_t = (peak - value)/peak > max_drawdown`, peak = running max | 0.10 | 4 h |
| `DailyLossBreaker` | `daily_loss_2%` | `daily_pnl <= -max_daily_loss` from that session's opening value; needs `reset_day(current_value)` at every session start | 0.02 | 1 h |
| `ConsecutiveLossBreaker` | `consecutive_5` | `record_trade(pnl)`; `consecutive_losses >= max_consecutive` | 5 | 30 min |
| `LatencyBreaker` | `latency_100ms` | `record_latency(ms)`; `mean(latency_history[-LATENCY_WINDOW(10):]) > max_latency_ms`; history capped at `LATENCY_HISTORY=100` | 100.0 ms | 5 min |

Timeouts differ because conditions clear at different speeds. `BreakerManager(on_any_trip)`: `add_breaker`, `check_all(**kwargs) -> bool` (True only when every breaker allows trading; adding a breaker can only make the system more cautious), `get_status()`, `reset_all()`. `make_trip_callback(manager, breaker)` and `make_transition_logger(manager)` wrap callbacks so the manager logs every transition; `alert_handler` emits the first trip per breaker only. Other §26.5 breakers (weekly drawdown, position size, sector, volatility, spread, intraday move) are the same state machine with a different `check_condition`.

### Feature store by hand (`05_feast_feature_store`, §26.6)

- `FeatureViewSpec`: entity key `symbol`, event timestamp `timestamp`, TTL, feature columns, source path. Two views: financial `["past_ret_21d", "vol_21d", "rsi_14", "sharpe_21d"]`; model-based `["garch_cond_vol", "ffd_log_price", "ffd_log_volume"]` (fitted per fold).
- Servable date range = union of validation windows + holdout.
- `load_training_events(start, end)` -> (symbol, date, label) rows. `load_model_vintage(start, end, columns, assets)` selects the fold vintage valid at each decision date (by date, not id).
- `offline_join(events)`: attach features carrying exactly the decision date (known after that close; label = return from that session to the next; position acts no earlier than the next bar); drop rows missing any feature and print the count.
- `latest_snapshot(source_path, as_of_date, assets, columns)`: everything `<=` cutoff, sorted, last row per name (as-of retrieval). `leaked_snapshot`: first row `>` cutoff within `SKEW_SEARCH_DAYS=7` — measures what one session of look-ahead does to the served vector.
- Derived dates: training window = tail (`TRAINING_LOOKBACK_DAYS=91`) of the last validation window; `AS_OF_DATE` = first holdout session; guard asserts training end < holdout start and start < end, placed after derivation so hand overrides are also checked.
- `sample_assets(n=8)` by average dollar volume over `LIQUIDITY_WINDOW_DAYS=21` / `LIQUIDITY_RANK_DAYS=30` tail of the training window.
- Lineage per view: file, row count, date range, distinct (entity, timestamp) key count (rows > keys flags duplicates).

### Feast end to end with parity check (`05b_feast_live`, §26.6)

1. Declare: `Entity` (symbol), `FileSource(path=..., timestamp_field="timestamp")`, `FeatureView(name, entities=[symbol], ttl=timedelta(days=FEATURE_TTL_DAYS), schema=[Field(...)], source=...)`, `store = FeatureStore(repo_path=feast_tmp)`; `store.apply([entity, views...])` (schema validated at registration).
2. `feature_store.yaml` (written with `yaml.dump`): project `ml4t_feature_store`, provider `local`, registry `{path: <tmp>/registry.db}`, online_store `{type: sqlite, path: <tmp>/online.db}`, offline_store `{type: file}`, `entity_key_serialization_version: 3`.
3. Source prep: temp Parquet copies with `Date` cast to datetime over the window padded by `SOURCE_PAD_DAYS=30` each side; model-based copy = ONE filter over the union of evaluation windows (a per-window concat double-counts overlapping dates).
4. Offline: `store.get_historical_features(entity_df=(symbol, event_timestamp) pairs, features=[...]).to_df()` returns the most recent values at or before each timestamp within TTL. Online shape reproduced with one timestamp per entity (`get_online_features` needs a materialization job + online store, not stood up here).
5. Parity: restrict events to keys both sources hold (print dropped counts split by missing source); `polars_offline_join()` reproduces 05; assert per-feature `|diff| <= PARITY_TOLERANCE=1e-10`.
6. Registry introspection (`store.list_feature_views()` / `store.list_entities()`, exact names inferred) = library version of 05's lineage table. Temp dir deleted at the end.

### Experiment tracking: registry as tracker + MLflow parity (`06_mlflow_experiments`, §26.6)

Sequence: read registry by hand -> build catalog -> add backtest evidence -> trace manifest -> inspect distributions and budgets -> log to MLflow -> query -> assert parity -> delete store.

1. **Hash-stability demo**: `example_hash = compute_hash(canonical_json(example_config))`; recompute and compare equal. Schema listed by scanning `REGISTRY_SCHEMA_SQL` for `CREATE TABLE` / `CREATE INDEX` lines.
2. **Catalog** (SQL): `training_runs tr JOIN prediction_sets ps ON training_hash LEFT JOIN prediction_metrics pm ON prediction_hash WHERE tr.label IN (?, ?) AND ps.split = 'validation' AND pm.ic_mean_daily IS NOT NULL AND pm.ic_n_days > 0 ORDER BY tr.label, tr.family, pm.ic_mean_daily DESC`; columns `training_hash, family, label, config_name, created_at, prediction_hash, split, ic_mean`.
3. **best_validation**: sort by `ic_mean` desc -> `groupby(["label","family"], as_index=False).first()` -> one row per family-label. **family_counts** = `run_catalog.groupby(["label","family"]).size()` = experiment budget, shown beside it.
4. **Backtest evidence**: `JOIN backtest_metrics bm ON br.backtest_hash = bm.backtest_hash` with `br.stage = 'signal'` (equal-weight baseline); select `family, label, config_name, backtest_hash, sharpe, cagr, max_drawdown, total_return`; sort by `sharpe` desc.
5. **Manifest**: from `best_validation.iloc[0]` resolve `run_log/training/<training_hash>/spec.json` and `run_log/predictions/<prediction_hash>/predictions.parquet` with `exists` flags; read spec keys `family, config_name, label, seed, identity_version, execution_tier`; `pl.read_parquet(...).head(5)`. Any `False` = reproducibility claim failed.
6. **Figure**: ECDF step curve per family (ordered by median `ic_mean`, legend `"<family> (n=<count>)"`) — chosen over a strip plot because families differ by an order of magnitude in run count; `barh` of top `BACKTEST_PANEL_ROWS=6` Sharpe bars labelled `config_name + " / " + backtest_hash[:4]`.
7. **MLflow**: `shutil.rmtree(MLFLOW_DIR)` first; `mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}")`; `mlflow.set_experiment(CASE_STUDY_ID)`. Sample = all group leaders first, then highest-IC remainder, truncated to `MAX_MLFLOW_RUNS=50`; `assert leader_hashes.issubset(...)`. `log_catalog_to_mlflow` skips None/NaN IC, logs params `{family, label, config_name}` as `str`, metric `ic_mean_daily` as `float`, artifact `spec.json` if it exists; `assert n_logged == len(catalog_sample)`.
8. **Query**: `mlflow.search_runs(experiment_names=[CASE_STUDY_ID], order_by=["metrics.ic_mean_daily DESC"])`; `filter_string="params.family = 'gbm'"`.
9. **Parity**: `mlflow_best` = groupby (`params.label`, `params.family`) first after sort; rename to `training_hash, family, label, ic_mean`; `merge(manual_best, on=["family","label"], how="outer", indicator=True)`; `assert (_merge == "both").all()`; `ic_match = |diff| < PARITY_TOLERANCE=1e-8`; `hash_match = training_hash_mlflow == training_hash_manual`; assert both.
10. Cleanup: `shutil.rmtree(MLFLOW_DIR, ignore_errors=True)`.

## Guardrails and pitfalls

### Detection (01, 02)

- **Monitoring on a validation set instead of holdout** — validation sessions are the ones the model was selected on, so "drift" there measures the selection window. Read LABEL from the registry and require a holdout prediction set; stop if it is missing.
- **Measuring drift over training sessions** — compares a live window against the training set. Restrict panels via `evaluation_spans()`, `folds_by_coverage`, `window_terms`; apply fold and window together when a `fold` column exists.
- **Pipeline failure diagnosed as market drift** — a broken join or null-filled column produces the same shift as a regime change and the response is opposite; retraining on a broken feed fits the break. Run `validate_feature_data` / `DataFrameValidator` before any statistic.
- **PSI thresholds read as significance levels** — PSI has no null distribution. Treat 0.10 / 0.25 as conventions that rank features; use K-S for the noise question.
- **K-S alone at production sample sizes** — tens of thousands of observations reject on unactionable differences; a monitor that always alerts is one nobody reads. K-S only lifts a feature to WATCH when `psi >= PSI_KS_FLOOR = 0.02`.
- **Relative thresholds on near-zero metrics** — a relative change on IC near zero is meaningless; squared error has no natural scale. Absolute drops for IC and hit rate (0.005 / 0.01), relative increases for MSE (5% / 10%).
- **PSI bins from the reference range** — current values outside the reference support pile into the outer `-inf/+inf` bins and PSI understates the move. Known limitation; inspect `bin_psi` and support ranges (inference).
- **Over-reading marginal drift disagreements** — feature moved + output still can be light weighting or cancellation; output moved + features still can be an inter-feature relationship change invisible to marginal tests. Treat as a checklist (feature dependence, model sensitivity per input, upstream changes), not a diagnosis.
- **Comparing feature-PSI and prediction-PSI magnitudes** — different reference periods (last validation window vs first 63 holdout sessions). Compare direction only.
- **Detector calibrated on monitored data** — tunes itself to the shift it should find. First 63 sessions calibrate with no alerts; turbulence threshold set on the prior year's 80th percentile.
- **Comparing detectors across identical streams** — adjacent ridge penalties agree to five decimal places; noise reads as a finding. Assert bad-day rates differ on a meaningful share of sessions; use the unpenalized vs heavily penalized pair.
- **Empty or thin streams passing checks** — a filtered-out config leaves nothing to inspect; with 2-3 names a daily rank correlation is +/-1 by construction. Assert both configs survive filters; `MIN_ASSETS_PER_DATE=10`; coverage ~252 sessions (`>= 240`), same start/end.
- **Dating errors by the prediction session** — puts every alert one session earlier than a desk could raise it; lead-lag inherits the shift. Carry each error forward to the session its outcome becomes known.
- **Alert flapping on a sustained shift** — one shift walking through two windows raises a run of alerts. `ADWIN_COOLDOWN=21` sessions after an alert.
- **Daily-repeated tests fire on chance** — 1.5/2.0 sigma repeated daily produces chance alerts. Treat alerts as a review queue, not a finding; a false-alarm rate needs many periods.
- **Absolute lead-lag distances** — hide early warning vs late confirmation, the only thing timing measures. Keep the sign in `nearest_stress_lag`.
- **Turbulence proxy read as a regime label** — it sees the size of moves, not changes in feature-return relationships; quiet regime shifts are invisible. Use it to order checks, not to diagnose.

### Response (03)

- **Promoting on any positive Sharpe difference** — 63-session Sharpe SE ~ 2, so it promotes on noise about half the time. `MIN_SHARPE_IMPROVEMENT=0.20` and `MIN_OBSERVATION_SESSIONS=63`.
- **Thresholds set after seeing results** — not a threshold; the notebook's order is the only thing keeping rule and result apart. Fix all five criteria in the parameters cell; assign roles before any statistic.
- **Sharpe improvement bought with risk** — a candidate can earn Sharpe by taking more drawdown. `drawdown_ratio <= 1.25`.
- **Promoting a different strategy as an "upgrade"** — changes what the desk is exposed to; a person's decision. Signal correlation `>= 0.30` and position agreement `>= 0.30`.
- **Gross vs net comparison** — a candidate that trades more shows better gross and worse net. Charge each model for its own turnover at the midpoint cost.
- **Long-short book silently collapsing** — fewer than `2*TOP_K` ranked names makes legs overlap, the book loses dollar neutrality, spreads ~0 and the gate reads "no edge"; duplicate names overwrite weights. Raise on `< 2*TOP_K` names, duplicates in a leg, a name in both legs.
- **Pooled instead of per-session rank correlation** — pooled is dominated by day-level co-movement; two models can track levels and disagree on within-session ordering, the only thing a long-short book reads. Compute Spearman per session and average.
- **Label/execution timing mismatch** — `fwd_ret_1d` is close-to-close (adjusted closes) but the case study decides at close and enters at next open; the overnight move differs per name, so the Sharpe difference is a difference of proxies. Name it as a proxy; a true measurement needs open prices and a label built from them.
- **Scoring a prediction on the bar it was formed from** — credits the model with a price it could not trade. Pair score at close t with return t -> t+1 only.
- **Negative-to-less-negative Sharpe "improvement"** — still an improvement in ratio but no evidence either model deserves capital. The gate is not an instrument for "should the incumbent run"; ask that separately.

### Automated safety (04)

- **Breaker closing on a timer** — resumes trading into whatever tripped it. Timeout -> HALF_OPEN; the next observation closes it or re-opens for a full timeout.
- **Daily-loss breaker without session reset** — becomes the drawdown rule with a different number. Call `reset_day(current_value)` at every session start.
- **Single slow round trip halting trading** — one outlier is not degradation. Average the last `LATENCY_WINDOW=10` measurements before comparing to 100 ms.
- **Trip-only alert streams** — lose half-open probes and closes needed for post-mortems. `make_transition_logger` appends every transition to the manager log.
- **Daily-step simulation vs sub-day timeouts** — all four timeouts (4h, 1h, 30m, 5m) elapse before the next daily step, so every OPEN breaker retries daily and the drawdown breaker trips ~every two days (31 trips). That is the cycle, not policy. Distinguishing timeouts needs a minute-stepped simulation; intraday breakers fire on the path, not the session total.
- **Counterfactual account path** — SPY keeps being tracked after a halt, so post-halt returns are not money anyone made. Read the figure as "what would have happened"; nothing propagates a halt into later inputs, so co-occurrence is two rules on one session, never causation.
- **Uncalibrated limits** — 10% / 2% / 5 / 100 ms are plausible, not calibrated; what a desk can absorb depends on capital and investors, not on any calculation here.

### Infrastructure (05, 05b, 06)

- **Training-serving skew** — offline exact-date join and online as-of lookup computed by different code drift apart; the model is served what it was never trained on. Both must return identical vectors for a covered date; measure it rather than reason about it.
- **One wrong comparison operator (`>` instead of `<=`)** — a batch job writing tomorrow's features tonight plus an unfiltered serving query = look-ahead; one session moves the whole feature vector, not one feature. `latest_snapshot` uses `<=`; `<` would drop the current session's own values.
- **Missing TTL** — a store without TTL hands back last week's value as today's. Declare TTL (1 session for daily features in 05; 05 declares but does not enforce it — a real store rejects the stale read).
- **TTL too long or too short** — too short empties legitimate queries; too long answers with data that no longer describes the moment; parity on events with own-date rows cannot catch a long TTL. Test with events lacking their own row at ages either side of the declared limit (a separate test from parity).
- **Offline join reaching past the holdout** — the training set looks ordinary, evaluation means nothing. Assert training window ends before holdout opens and runs forward; refuse, do not degrade.
- **Silent imputation in a feature store** — hides its own gaps. Drop rows missing any feature and print the dropped count.
- **Duplicate values per (entity, timestamp) key** — the store's answer depends on which row it picked; the join fans out or picks a vintage fitted on later data. Lineage records distinct key count; select fold vintage by date, not by id.
- **Per-window concat of model-based copies** — double-counts any date two windows cover, silently. One filter over the union of windows.
- **Parity compared on dates with no own row** — Feast carries forward within TTL, the exact-key join returns nothing; disagreement is a property of the question (embargo gaps between validation windows). Restrict events to keys both sources hold; print dropped counts split by source.
- **Parity tolerance too loose** — the failure is a wrong row (off by the day's move), not rounding. `PARITY_TOLERANCE=1e-10`; compare per feature; assert so the notebook stops.
- **Reading parity agreement as correctness** — it shows both paths made the same vintage selection, not that the selection is right. The fold-by-date pairing is the separate argument for correctness.
- **Stale MLflow tracking store** — a store left by an interrupted run still answers `search_runs`. `shutil.rmtree(MLFLOW_DIR, ignore_errors=True)` before `set_tracking_uri`; keep the store under `get_output_dir(26, ...)`, never beside source; delete at the end.
- **Top-N sampling bias in the logged subset** — a (family, label) group whose best run falls outside the global top 50 reaches the manual catalog but not MLflow. Put every group leader in first, fill with highest-IC leftovers, truncate; assert leaders are included.
- **Silent group loss in the parity join** — an inner join drops a group present on one side and the check passes vacuously. `merge(how="outer", indicator=True)` and `assert (_merge == "both").all()`.
- **Agreeing on the metric but not the run** — "agreeing on the number while disagreeing on which run produced it is the more interesting failure". Assert `hash_match` and `ic_match` together.
- **Ranking metric mistaken for the selection metric** — deploying the top-IC configuration. Rank by `ic_mean_daily`; select on validation backtest Sharpe from `backtest_metrics`; keep the two tables and questions separate.
- **Cross-stage comparison** — stages other than `signal` vary an allocator, cost assumption or risk overlay. `WHERE br.stage = 'signal'` for any model-vs-model comparison.
- **Multiple testing through unequal experiment budgets** — the maximum of more draws is higher even when nothing improved. Display `family_counts` beside `best_validation`; the count exposes the asymmetry even though it does not quantify the inflation (depends on configuration correlation).
- **Correlated configurations within a family** — a tight cluster can be several views of one overfit; an isolated high score can be a real improvement. Read the per-family ECDF to decide what to check; the check is a held-out evaluation, and the holdout is spent once.
- **Reproducibility claim without artifacts** — a metric row cannot answer "can it be reproduced". Build a manifest with `exists` flags; any `False` fails the claim.
- **Duplicate or editable run rows** — sequence numbers, timestamps or user names as ids let duplicates accumulate and hide config changes as edits. Primary keys = `compute_hash(canonical_json(spec))`.
- **Non-canonical serialization before hashing** — key order or whitespace differences give different hashes for the same spec. Always canonicalize; verify by recomputing and comparing equal.
- **Pooling across return horizons** — `fwd_ret_1d` and `fwd_ret_5d` are different tasks with different IC scales. Group by `(label, family)` everywhere.
- **Fitted-but-never-backtested runs invisible** — a prediction set with no backtest row is a model never evaluated as a strategy. Join to `backtest_runs` and report missing rows as a gap.
- **Untyped MLflow logging** — params are uninterpreted strings; only metrics can be ordered and filtered. `log_params` for `family`, `label`, `config_name`; `log_metrics` for `ic_mean_daily`.
- **NaN or empty metrics** — corrupt ranking and logging. SQL filters `ic_mean_daily IS NOT NULL AND ic_n_days > 0`; logger skips None/NaN; `fillna(0)` only for the Sharpe-bar display.
- **Blanket warning suppression** — hides convergence and numerical warnings. Filter `category=FutureWarning, module="mlflow"` only; raise only `logging.getLogger("mlflow")` to WARNING.
- **Ephemeral single-user store mistaken for a tracker** — persistence, multi-user and access control are most of why MLflow is chosen. Production: persistent shared store, log one run as each training job finishes, no `MAX_MLFLOW_RUNS` cap.
- **Over-reading the parity check** — proves only that the query returns the catalog it was given, on one case study with one metric; says nothing about whether IC is the right ranking metric.

### Registry hygiene (all notebooks)

- **Pinning prediction hashes** — content-addressed hashes change on every refit. Pin config names (`ols`, `ridge_a100.0`, `ridge_a1000000.0`); resolve via `resolve_prediction_hash` / `load_prediction_identity`.
- **Registry row without parquet on disk** — role resolves, failure surfaces later where the cause is invisible. `load_prediction_identity` checks the prediction parquet exists.
- **Writing to the registry while reviewing** — review must not mutate the result record. SQLite URI `mode=ro` + `PRAGMA query_only`.
- **`immutable=1` on a live registry** — skips locking and ignores the WAL, so reads return superseded rows or missing tables. Use it only for the downloaded, unwritable artifact bundle (verified by `scripts/download_artifacts.py`); `registry_readonly_uri` chooses by checking whether the directory is writable.
- **Typed fold-boundary dates** — a rebuild with different folds moves every boundary while the literal stays. `REFERENCE_START`, `TRAINING_START/END`, `AS_OF_DATE` default `None` and derive from `setup.yaml` fold geometry; guard asserts run after derivation.
- **Suppressed figures in executed notebooks** — `MPLBACKEND=Agg` or `PLOTLY_RENDERER=json` removes figures the notebook must carry. Do not set them.
- **Per-case-study registries** — each case study has its own `run_log/registry.db`; a registry-wide view must reconcile across them (stated limitation).

## Decision rules and defaults

**Triage**: degraded status -> data integrity (01) -> market conditions (02 turbulence) -> model decay; act via 03 only after data integrity is confirmed. Never retrain directly from an alert. Alert cluster on a turbulent stretch -> check strategy sizing for those conditions first; cluster in a quiet stretch -> volatility is ruled out, data and model remain. Two configs tripping the same detector on the same sessions -> something both see moved (does not rule out joint decay of models sharing features/history).

**Drift monitoring (01) defaults**

| Parameter | Default | Note |
|---|---|---|
| `LOOKBACK_DAYS` | 63 | a quarter: long enough for rolling IC to mean something, short enough to notice within a cycle |
| `PSI_WATCH` / `PSI_ALERT` | 0.10 / 0.25 | conventions, not significance |
| `KS_WATCH_PVALUE` | 0.05 | |
| `PSI_KS_FLOOR` | 0.02 | K-S only lifts to WATCH above this PSI |
| `IC_WATCH_DROP` / `IC_ALERT_DROP` | 0.005 / 0.01 | absolute |
| `MSE_WATCH_INCREASE` / `MSE_ALERT_INCREASE` | 0.05 / 0.10 | relative |
| `n_bins` / `epsilon` | 10 / 1e-6 | |
| `MIN_ROLLING_SESSIONS` | 21 | `LOOKBACK_DAYS // 3` |
| `SEED` | 42 | |

**Online detection (02) defaults**: VALIDATION 2015-01-01..2015-12-31; `CALIBRATION_DAYS=63`; `CALIBRATION_YEAR=2014`; configs `ols` vs `ridge_a1000000.0`; `TOP_N_LIQUID=200`; `MIN_ASSETS_PER_DATE=10`; `ADWIN_WINDOW=21`, `ADWIN_SENSITIVITY=1.4` (pooled SEs), `ADWIN_COOLDOWN=21`; `DDM_MIN_SAMPLES=20`, `DDM_WARNING_SIGMA=1.5`, `DDM_DRIFT_SIGMA=2.0`; `BAD_DAY_MARGIN=0.01`, `BAD_DAY_FLOOR=0.52`; `STRESS_VOL_WINDOW=21`, `STRESS_QUANTILE=0.80`; `MIN_MONITORED_SESSIONS=240`; 252 sessions/year. Use the mean-shift detector for errors that grow in size; the bad-day monitor for a model wrong slightly more often.

**Promotion gate (03) defaults**: INCUMBENT `ols`, CANDIDATE `ridge_a100.0`; HISTORICAL 2010-01-01..2014-12-31 (descriptive only); ROLLOUT 2015-01-01..2015-12-31; `SHADOW_SESSIONS=63`; `TOP_K=100` (needs `>= 200` ranked names/session); `MIN_SHARPE_IMPROVEMENT=0.20`; `MIN_OBSERVATION_SESSIONS=63`; `MIN_SIGNAL_CORRELATION=0.30`; `MIN_POSITION_AGREEMENT=0.30`; `MAX_DRAWDOWN_RATIO=1.25`; `MIN_SESSIONS_PER_YEAR=240`. Promote only if all five pass; otherwise the candidate stays in shadow with no capital. After the gate: capital-capped A/B -> staged rollout -> tested rollback.

**Circuit breakers (04) defaults**: `MAX_DRAWDOWN=0.10` (from running peak), `MAX_DAILY_LOSS=0.02` (from session open), `MAX_CONSECUTIVE_LOSSES=5`, `MAX_LATENCY_MS=100.0` (mean of last 10); recovery drawdown 4 h, daily loss 1 h, consecutive 30 min, latency 5 min; `LATENCY_HISTORY=100`; sim: SPY from 2020-01-02, `N_STEPS=100`, `INITIAL_VALUE=100000`; latency exponential mean 20 ms -> 150 ms after 80% of run; `SEED=42` set immediately before the loop. The engine asks one question: `check_all` -> trade only if every breaker allows it. Market-driven and infrastructure-driven halts need different operator responses; start from the timeline panel (which breakers open when).

**Feature store (05/05b) defaults**: `TRAINING_LOOKBACK_DAYS=91`; `N_SAMPLE_ASSETS=8`; `FEATURE_TTL_DAYS=1` (05, daily features) / 2 (05b, lets Feast carry the previous session forward); `LIQUIDITY_WINDOW_DAYS=21`; `LIQUIDITY_RANK_DAYS=30`; `SKEW_SEARCH_DAYS=7`; `SOURCE_PAD_DAYS=30`; `PARITY_TOLERANCE=1e-10`. Feature-view checklist: entity key, event timestamp, TTL, feature columns (typed in Feast), source lineage (file, rows, date range, distinct keys). Contract: offline join takes values known at each decision timestamp; online retrieval takes the last values known at the decision moment by the same rule; the registry says where both came from. Choose Feast when you need a registry, an online store or multi-team isolation; otherwise direct Parquet retrievals do the same work with no infrastructure.

**Experiment tracking (06) defaults**: labels `fwd_ret_1d`, `fwd_ret_5d`; `MAX_MLFLOW_RUNS=50`; `PARITY_TOLERANCE=1e-8`; `BACKTEST_PANEL_ROWS=6`; `HASH_PREFIX_CHARS=4`; MLflow experiment name = `CASE_STUDY_ID`; run name = `training_hash`. Catalog eligibility: `split='validation'`, `ic_mean_daily IS NOT NULL`, `ic_n_days > 0`. Rank by IC; select on validation backtest Sharpe; compare models only within `stage='signal'`. Group by (label, family), `.first()` after sorting by IC desc, show `family_counts` beside it. Parity go/no-go: every (family, label) group on both sides, top-run hash identical, `|dIC| < 1e-8`. Plot: ECDF when group sizes differ by an order of magnitude; strip plot otherwise (inference). Run-log checklist: provenance; data + evaluation protocol; configuration; artifacts; decision gates; three hashed lineage levels, each carrying its parent id.

**Reproduce-one-run procedure**: pick row -> `training_hash`, `prediction_hash`; resolve `run_log/training/<training_hash>/spec.json` (confirm exists); resolve `run_log/predictions/<prediction_hash>/predictions.parquet` (confirm exists); read spec (`family`, `config_name`, `label`, `seed`, `identity_version`, `execution_tier`); read first rows of predictions with polars; any missing file = claim failed.

## Code patterns and APIs

**Repo paths**: `26_mlops_governance/{01_drift_monitoring, 02_online_drift_detection, 03_safe_model_rollout, 04_circuit_breakers, 05_feast_feature_store, 05b_feast_live, 06_mlflow_experiments}.py` (Jupytext percent) and `.ipynb` twins. Run: `uv run python 26_mlops_governance/01_drift_monitoring.py`; test: `uv run pytest tests/test_chapter_notebooks.py -v -k "26_mlops_governance"`; execute in place: `uv run jupyter nbconvert --to notebook --execute --inplace ...`. Docker image `ml4t` for 01/02/04/05/05b/06; 03 runs on local `uv`, CPU-only. Install Feast: `pip install 'feast>=0.40'` (`[mlops]` extra); MLflow 3.x.

**Case-study layout**: `CASE_DIR = get_case_study_dir("us_equities_panel")`; `CASE_DIR/config/setup.yaml` (holdout window, primary label, per-leg cost range, fold geometry); `CASE_DIR/run_log/registry.db` (SQLite); `CASE_DIR/run_log/training/<training_hash>/spec.json`; `CASE_DIR/run_log/predictions/<prediction_hash>/predictions.parquet`; `CASE_DIR/features/model_based.parquet` (optional `fold` column); financial feature parquet. `CODE_ROOT = CASE_DIR.parent.parent`. Env override `ML4T_REGISTRY_SNAPSHOT` for the registry path (03). Outputs via `get_output_dir(26, "figure_26_2")` / `get_output_dir(26, "mlflow_tracking")`.

**Registry schema** (`case_studies/utils/registry/store.py`, `REGISTRY_SCHEMA_SQL`): `training_runs(training_hash PK, family, label, config_name, spec_json, created_at, git_commit, entry_point, started_at, elapsed_s, runtime_json, identity_version, execution_tier)`; `training_identity_migrations`; `prediction_sets(prediction_hash PK, training_hash FK, checkpoint_value, checkpoint_kind, split, created_at)`; `prediction_coverage(... n_duplicates, n_missing, n_extra, n_null, n_non_finite, status)`; `prediction_metrics(prediction_hash PK, ic_mean, ic_std, ic_t, n_folds, pct_positive, task_type, accuracy, ..., direction_label_error)`; `fold_metrics`; `backtest_runs(backtest_hash PK, prediction_hash FK, spec_json, stage, created_at, git_commit, ...)`; `backtest_metrics(backtest_hash PK, sharpe, sortino, total_return, max_drawdown, cagr, volatility, calmar, omega, stability, tail_ratio, win_rate, kurtosis, skewness, var_95, cvar_95, n_periods, num_trades, total_commission, total_slippage, avg_turnover)`; `backtest_fold_metrics`; `causal_runs` (DML effects, Driscoll-Kraay / Newey-West HAC SEs; not read here). Model names stored as `linear/{config}`. Families: `deep_learning`, `latent_factors`, `linear`, `tabular_dl`, `gbm`. Note: the notebook filters on `pm.ic_mean_daily` / `pm.ic_n_days`, which the `CREATE TABLE` lists as `ic_mean` etc. — likely a schema migration (inference, unresolved).

**Registry access**: `from case_studies.utils.registry import REGISTRY_SCHEMA_SQL, canonical_json, compute_hash`; `open_registry_readonly(path) -> sqlite3.Connection` (URI `mode=ro`, `PRAGMA query_only`; `registry_readonly_uri` adds `immutable=1` only for an unwritable bundle); `load_holdout_window(setup_path) -> (start, end, label)`; `load_holdout_prediction_hash(registry_path, preferred, required) -> (label, hash, model_family)`; `resolve_prediction_hash(config_name) -> str`; `_load_predictions_from_hash(run_hash, model_label) -> pl.LazyFrame`; `load_prediction_identity(config_name) -> PredictionIdentity`; `load_run_predictions(run_hash, model_label)`; `load_predictions(start, end) -> pd.DataFrame`; `validate_coverage(frame, start, end)`; `load_cost_bps(setup_path)`. Prediction schema: canonical `symbol`, `timestamp`, scores, label `fwd_ret_1d`; concatenate streams with a model label column.

```python
def query_table(query: str, params: tuple[object, ...] = ()) -> pd.DataFrame:
    with sqlite3.connect(REGISTRY_PATH) as conn:
        return pd.read_sql_query(query, conn, params=params)
```

**Validation**: `from ml4t.diagnostic.validation import DataFrameValidator`; `validate_feature_data(features_df: pl.DataFrame, required_columns: list[str]) -> None`.

**Evaluated-session filter**: `evaluation_spans()`, `folds_by_coverage(path, windows)`, `window_terms(clipped) -> list[pl.Expr]`; `load_temporal_panel(start, end, columns)`; `load_feature_panel(start, end, feature_columns)` joins financial + model-based parquets.

**Detectors**: `TwoWindowMeanShift(window_size, sensitivity, cooldown_days).update(value) -> bool`; `BadDayFrequencyMonitor(min_samples, warning_level, drift_level).update(error: bool) -> str`, `.reset()`; `DetectorSummary` dataclass; `run_adwin_stream(model_name, calibration, monitoring)`, `run_ddm_stream(...)` -> (alert_dates, records); `run_detectors(daily_errors, stress_starts)`; `nearest_stress_lag(alert_dates, stress_dates)`; `get_liquid_universe(top_n)`. Production would use a library implementation such as river's ADWIN/DDM (inference).

**Rollout**: `PredictionIdentity`, `build_portfolio_stream(predictions, top_k, cost_bps) -> pd.DataFrame`, `annualized_sharpe(returns)`, `max_drawdown(returns)`, `PromotionCriteria` dataclass, `draw_shadow_panel(ax)`, `draw_hurdle_panel(ax)`.

**Breakers**: `BreakerState(Enum)` CLOSED/OPEN/HALF_OPEN; `BreakerEvent(reason, value, threshold, time)`; `transition_breaker(breaker, new_state, reason, value, threshold, event_time)`; `advance_breaker_state(breaker, event_time, **kwargs)`; `CircuitBreaker(ABC).check_condition(portfolio_value=None, **kwargs) -> (should_trip, reason, value, threshold)`; `DrawdownBreaker`, `DailyLossBreaker.reset_day(value)`, `ConsecutiveLossBreaker.record_trade(pnl)`, `LatencyBreaker.record_latency(ms)`; `BreakerManager.add_breaker / check_all / get_status / reset_all`; `make_trip_callback(manager, breaker)`, `make_transition_logger(manager)`, `alert_handler(event)`. New rule = subclass + `check_condition` only.

```python
manager.add_breaker(DrawdownBreaker(name=DRAWDOWN_BREAKER, max_drawdown=MAX_DRAWDOWN,
                                    recovery_timeout=timedelta(hours=DRAWDOWN_RECOVERY_HOURS)))
ok = manager.check_all(portfolio_value=v, event_time=t)   # trade only if True
```

**Feature store by hand**: `FeatureViewSpec`, `load_training_events(start, end)`, `load_model_vintage(start, end, columns, assets=None)`, `offline_join(events)`, `sample_assets(n_assets)`, `latest_snapshot(source_path, as_of_date, assets, columns)`, `latest_model_snapshot(as_of_date, assets)`, `leaked_snapshot(as_of_date, assets)`. As-of pattern: filter `timestamp <= as_of`, sort, take last row per `symbol`.

**Feast**: `from feast import Entity, FileSource, FeatureView, FeatureStore` (+ `Field`); `FileSource(path=..., timestamp_field="timestamp")`; `FeatureView(name, entities=[symbol], ttl=timedelta(days=FEATURE_TTL_DAYS), schema=[Field(...)], source=...)`; `store = FeatureStore(repo_path=feast_tmp)`; `store.apply([...])`; `store.get_historical_features(entity_df, features).to_df()`; `store.list_feature_views()` / `store.list_entities()` (inference on exact names); `polars_offline_join()`.

**MLflow**: `mlflow.set_tracking_uri("sqlite:///.../mlflow.db")`, `set_experiment(name)` (`.name`, `.experiment_id`), `start_run(run_name=)`, `log_params(dict[str, str])`, `log_metrics(dict[str, float])`, `log_artifact(path)`, `search_runs(experiment_names=[...], filter_string=..., order_by=["metrics.<m> DESC"])` -> pandas with `run_id`, `params.*`, `metrics.*`, `tags.mlflow.runName`.

```python
with mlflow.start_run(run_name=row.training_hash):
    mlflow.log_params({"family": str(row.family), "label": str(row.label)})
    mlflow.log_metrics({"ic_mean_daily": float(row.ic_mean)})
    if spec_path.exists():
        mlflow.log_artifact(str(spec_path))
```

Parity pattern: outer `merge(on=[group keys], how="outer", indicator=True)` -> assert `_merge == "both"` everywhere, assert identity column equal, assert `|metric diff| < tolerance`.

**Plot helpers**: `utils.style.COLORS["positive"]`, `FIGSIZE["dual_h"]`, `add_message_title(ax, title, subtitle=)`, `ml4t_palette(n, categorical=True)`, `show_with_alt(fig, alt_text)`, `zero_line(ax, axis="x")`; `FAMILY_DISPLAY` maps family ids to labels. Tests: `tests/test_registry_completeness.py`, `tests/test_backtest_rekey.py`, `tests/fixtures/seed_results.py`.

## Evidence from the book

- **03 shadow window (63 sessions, 2015)**: signal correlation and position agreement between `ols` and `ridge_a100.0` both "very high" — the candidate is a small adjustment to the incumbent, so criteria 3 and 4 do no work (their purpose is a candidate from a different family). Both books lost money over the window (negative Sharpes); a Sharpe improvement between two negative Sharpes is not evidence either model deserves capital. Pass/fail is read off the table at run time; the digest does not fix the outcome.
- **Adjacent ridge penalties** in the case-study grid produce predictions agreeing to about five decimal places (motivates 02's stream-difference check and 03's choice of pair).
- **04 SPY 2020-01-02 + 100 sessions**: the drawdown breaker recorded 31 trips; SPY stayed beyond the 10% limit for most of the window and, because every recovery timeout (4h..5m) is shorter than the 24h step, the breaker re-opens the day after each recovery attempt, about every two days — the daily-step artifact, not policy. The synthetic latency breaker trips after 80% of the run when the rolling mean crosses 100 ms. Both runs end halted; market and infrastructure halts arrive at different times.
- **05 look-ahead**: one session of look-ahead (`>` vs `<=`) moves every feature of the served vector for the eight sample names; bars near the axis are not zero (RSI runs 0-100 while annualized vol is a small decimal, so the chart ranks moves rather than comparing them).
- **05b parity**: Feast's point-in-time join and the hand-written exact-key join agree per feature within 1e-10 on the restricted event set (assertion passes or the notebook stops). Dropped events split by missing source: model features missing = the embargo gap the restriction exists for; financial features missing = a different problem.
- **06 parity**: designed to pass by construction — MLflow's `search_runs` returns the same top run per (family, label) with the same IC within 1e-8 as the hand-built SQL catalog. Interpretation: the tracker adds interface and infrastructure; determinism and traceability came from the pipeline. Families differ by roughly an order of magnitude in run count.
- **01**: no numeric results; the notebook could not run because the registry lacked a holdout prediction set. **02**: summary-table structure only (alert_count, first_alert, median_lag_days per config x detector), no numbers.
- Not implemented in any notebook (README level only): backtest-to-live realization ratios and dashboards (§26.2), SHAP-based feature drift (§26.3), A/B and staged rollout gates / rollback procedures (§26.4), position/sector/volatility breakers and override discipline (§26.5), data versioning, model registries beyond the case-study registry, CI/CD (§26.6).

## Related references

- `chapters/25_live_trading.md` — deployment verification that precedes this monitoring workflow; technical-failure diagnostics.
- `chapters/11_ml_pipeline.md` — registry schema (`training_runs`, `prediction_sets`, `backtest_runs`), walk-forward folds and the validation/holdout geometry this chapter reads.
- `chapters/16_strategy_simulation.md` — baseline backtest metrics (`stage='signal'`) that 06 reads and that decide selection.
- `chapters/19_risk_management.md` — drawdown and limit framework reused by the circuit breakers.
- `chapters/07_defining_the_learning_task.md` — `fwd_ret_1d` / `fwd_ret_5d` labels and the close-to-close vs next-open timing proxy.
- `chapters/08_financial_features.md` and `chapters/09_model_based_features.md` — the financial and model-based (GARCH, FFD) feature views served by the feature store.
- `chapters/12_gradient_boosting.md`, `chapters/13_dl_time_series.md`, `chapters/14_latent_factors.md` — the model families catalogued in the registry.
- `chapters/15_causal_estimation.md` — `causal_runs` table (DML, HAC standard errors) present in the registry schema.
- `chapters/27_systematic_edge.md` — reflection on the full pipeline as one continuously validated system.
- `case_studies/us_equities_panel.md` — the artifacts every notebook here consumes.
- `libraries/ml4t_diagnostic.md` — `DataFrameValidator`, metrics and registry interface.
- `libraries/ml4t_data.md` — loaders and point-in-time data access.
- `libraries/ml4t_live.md` — live execution layer the breakers and monitors sit on.
- `libraries/ml4t_backtest.md` — backtest runs and metrics in the registry.
- `guardrails.md`, `decision_rules.md`, `workflow.md`, `evidence.md`, `glossary.md`, `companion_repo.md` — cross-cutting indexes.

**Further reading**
- Bailey & Lopez de Prado 2014, "The Deflated Sharpe Ratio".
- Bifet & Gavalda 2007, "Learning from Time-Changing Data with Adaptive Windowing" (ADWIN).
- Federal Reserve SR 11-7 (2011), Supervisory Guidance on Model Risk Management.
- Capponi, Huang, Sidaoui, Wang & Zou 2025, "The Nonstationarity-Complexity Tradeoff in Return Prediction".
- Gama, Medas, Castillo & Rodrigues 2004, "Learning with Drift Detection" (DDM).
- Harvey, Liu & Zhu 2016, "...and the Cross-Section of Expected Returns".
- Hinder, Vaquet & Hammer 2023, concept drift survey.
- Korn, Moller & Schwehm 2022, "Drawdown Measures: Are They All the Same?".
- Lopez de Prado, Lipton & Zoonekynd 2025, "How to Use the Sharpe Ratio".
- Lu et al. 2018, "Learning under Concept Drift: A Review".
- Lundberg & Lee 2017, SHAP.
- McLean & Pontiff 2016, "Does Academic Research Destroy Stock Return Predictability?".
- Paleyes, Urma & Lawrence 2023, "Challenges in Deploying Machine Learning".
- Sculley et al. 2015, "Hidden Technical Debt in Machine Learning Systems".
- Studer et al. 2021, CRISP-ML(Q).
- Varma 2025, "The False Promise of Drawdown Rules".

## Glossary

- **Technical failure** — verification problem: same inputs produce different outputs (pipeline divergence).
- **Statistical failure** — performance problem: same outputs no longer predict returns (decay).
- **Holdout** — history reserved and never read while models are chosen; the offline stand-in for live data.
- **Validation window** — sessions a walk-forward fold is scored on just after its cut-off.
- **PSI** — population stability index: sum over reference-defined bins of (cur% - ref%) * ln(cur%/ref%); no null distribution.
- **K-S test** — two-sample Kolmogorov-Smirnov: max gap between empirical CDFs with a p-value under the one-distribution null.
- **Information coefficient (IC)** — per-session Spearman rank correlation between scores and realized returns; `ic_mean_daily` = mean of daily cross-sectional IC.
- **Hit rate** — share of names whose direction the model called correctly.
- **Sequential detector** — reads one observation at a time and decides after each whether the stream still looks like its calibration.
- **ADWIN-style / two-window mean shift** — alerts when adjacent windows' means separate by more than pooled standard error times a sensitivity.
- **DDM** — Drift Detection Method: tracks running error rate p and SE s, keeps the minimum p+s, warns/drifts when p+s exceeds p_min + k*s_min.
- **Bad day** — session whose directional error rate exceeds max(calibration mean + 0.01, 0.52).
- **Turbulence proxy** — annualized rolling std (21 sessions) of the cross-sectional median daily return; stress when above the prior-year 80th percentile.
- **Signed lead/lag** — alert-to-nearest-stress distance with sign: positive = detector led, negative = alert after the episode began.
- **Shadow mode** — candidate runs on live data, trades recorded, no capital.
- **Promotion gate** — conjunction of fixed criteria a candidate must clear to leave shadow mode.
- **Signal correlation** — per-session Spearman between two models' scores, averaged.
- **Position agreement** — intersection over union of names held by two books.
- **Drawdown ratio** — candidate max drawdown / incumbent max drawdown.
- **Realization ratio** — backtest-to-live performance comparison (§26.2; not implemented in the notebooks).
- **Circuit breaker** — rule that switches a system off automatically when a measurement crosses a limit and back on only after a defined test.
- **Closed / open / half-open** — trading / halted / trial state entered after timeout where the next observation closes or re-opens.
- **Recovery timeout** — time an OPEN breaker waits before moving to HALF_OPEN.
- **Counterfactual account path** — equity series that keeps tracking the market after a halt; not realized P&L.
- **Feature store** — component answering "what were this entity's features at this moment" for both training and serving.
- **Training-serving skew** — divergence between offline and online feature answers computed by different code.
- **Feature view** — declaration of one feature table: entity key, event timestamp, TTL, feature columns.
- **Entity key** — column naming what each row is about (`symbol`).
- **Event timestamp** — column saying when a row's values were true (`timestamp`).
- **Time to live (TTL)** — how long a value stays servable after its timestamp; turns a missing row into an error instead of a stale answer.
- **Point-in-time join** — attach to each (entity, timestamp) the features known at exactly that moment (or latest within TTL).
- **As-of retrieval** — latest value at or before a moment, per entity.
- **Vintage** — the per-fold fitted version of a model-based feature; selected by date, not id.
- **Lineage** — where a served value came from: file, row count, date range, distinct entity-timestamp keys.
- **Content-addressed identifier / prediction hash** — registry key that is the hash of the canonical specification that produced it; changes on refit.
- **Canonical JSON** — deterministic serialization (stable key order/bytes) so equal specs hash equal.
- **Idempotent run log** — re-running lands on the existing row or creates a new hash; never a duplicate row.
- **Training run / prediction set / backtest run** — the three lineage levels (`training_hash` -> `prediction_hash` -> `backtest_hash`), each carrying its parent id.
- **Stage `signal`** — the equal-weight baseline backtest; other stages add allocator, cost or risk overlays.
- **Family** — model class grouping (`gbm`, `linear`, `deep_learning`, `tabular_dl`, `latent_factors`).
- **Experiment budget** — number of configurations tried per family and label.
- **Manifest** — table mapping a run to its artifact paths with existence flags.
- **Ranking parity** — agreement between two trackers on each group's top run identity and metric.
- **Params vs metrics (MLflow)** — uninterpreted strings vs orderable/filterable numbers; `filter_string` = SQL-like predicate over `params.*` / `metrics.*`.
- **ECDF** — empirical cumulative distribution; share of runs at or below each value.
- **Three-layer governance model** — detection, response, automated safety.
