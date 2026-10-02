# The ML4T workflow: research to production, end to end
> The book runs one process across 27 chapters and nine markets: build point-in-time-correct data, define a strategy as a decision process, turn data into labels and features under decision-time admissibility, learn in a model cascade that starts linear, cross an *evidence boundary* into backtests, portfolios, costs and risk overlays, confirm once on a sealed holdout, deploy the same code that was researched, and monitor until the edge decays and the strategy is retrained, paused or retired. This file is the map: for each stage, what it must answer, what it produces, its go/no-go gate, and which reference to open.

## When to use this reference
- Any multi-stage task (a new strategy, a pipeline audit, "where does this step belong"): place the task on the stage map, read that stage's gate, then open the chapter file it names.
- Deciding whether a result may move to the next stage, or whether a change is a tuned trial or a new setup version.
- This file is the map, not the catalog: numeric thresholds live in `decision_rules.md`, the full guardrail list in `guardrails.md`, vocabulary in `glossary.md`, and what the nine case studies found in `evidence.md`.

## 1. The thesis
- **The process is the edge.** Markets change (structural breaks, regimes, data drift, concept drift); any single signal decays. Durable performance comes from a repeatable process that keeps generating candidates and, more importantly, keeps rejecting bad ones (`chapters/01_process_is_edge.md`, `chapters/27_systematic_edge.md`).
- **Evidence boundary.** Exploration (features, models, allocators, overlays, cost headroom) happens on training and validation folds with every trial logged. Confirmation happens once, on a sealed holdout, with everything frozen. A second look at the holdout turns it into validation data.
- **Falsification, not demonstration.** A backtest is an attempt to break the strategy under realistic assumptions (timing, fills, costs, regimes, search size); what survives is what gets reported (`chapters/16_strategy_simulation.md`).
- **Three defenses against the researcher's own biases** (ch27): falsifiable hypotheses stated before testing; rigorous out-of-sample evaluation; statistical corrections for the number of trials.
- **Four separate evaluation questions** (ch20): signal quality (does the score order returns?), portfolio translation (does ranking become Sharpe?), cost survival (does the edge exceed the cost gate set by the instrument?), temporal stability (does it hold out of sample?). A strategy can pass one and fail the next; the funnel is the story.
- **Absence of a measurement is not a measurement of zero.** Report what was not attempted as "not attempted", never as a failure or a pass.
- **Three metric roles, never conflated** (ch06 §6.4): model diagnostics (did it fit), signal diagnostics (does the score order forward returns), strategy outcomes (P&L after mapping, constraints and costs). Tune on the first two, confirm on the third; optimizing every micro-decision directly on simulated portfolio outcomes is one of the easiest ways to overfit a backtest.
- **Regimes are a risk lens, not a timing signal** (ch01 §1.4): they earn their place by identifying adverse environments and connecting them to predefined risk actions; whole-sample labels describe, they do not predict (go/no-go rule under Stage 3).
- **Independents must manufacture governance** (ch01 §1.5): institutions get friction and review for free; a solo researcher substitutes documentation, checkpoints and explicit stop criteria written before results, plus reusable infrastructure (registry, protocol, parity tests) that compounds research quality over time.

## 2. Stage-by-stage walkthrough

Each stage lists: purpose · inputs → outputs · questions it must answer · gate · references.

The book's own coarse label is the "5-Stage ML4T Workflow" (ch27 calls it the alpha-factory blueprint; that chapter's README does not spell the stage names out). The companion repo's library table is the nearest concrete mapping, and the numbered stages below refine it rather than replace it: Data (`ml4t-data`) → Stage 0; Signal (`ml4t-engineer`) → Stages 1–4; Evaluation (`ml4t-diagnostic`) → Stages 4–5 and 11; Models (`ml4t-models`) → Stage 6; Strategy (`ml4t-backtest`) → Stages 7–10; Deployment (`ml4t-live`) → Stages 12–13.

### Stage 0. Data infrastructure (Ch2–Ch4)
- Purpose: a point-in-time-correct, survivorship-free, identifier-consistent data layer that the rest of the workflow can trust.
- Inputs → outputs: vendor/raw files → validated, adjusted, as-of-queryable datasets (Parquet/Hive storage, DuckDB for analytics; `ml4t-data` loaders), data-quality report, dataset cards.
- Must answer: What does each timestamp mean (event, disclosure, extraction)? Which revision of each value was visible when? Which corporate actions and identifier changes are handled, and how? Is the universe defined per period?
- Gate: data-quality framework passes (OHLC invariants, coverage, PIT validation, survivorship test); storage fits the access pattern.
- References: `chapters/02_financial_data_universe.md`, `chapters/03_market_microstructure.md` (intraday data, bars, LOB), `chapters/04_fundamental_alternative_data.md` (bitemporal fundamentals, entity resolution, macro release timing, alt-data due diligence), `libraries/ml4t_data.md`, `companion_repo.md` (data loaders, `ML4T_DATA_PATH`).

### Stage 1. Strategy framing and trading setup (Ch6)
- Purpose: turn an idea into a testable economic hypothesis and a versioned decision process.
- Inputs → outputs: idea → strategy map (family, source of edge SLOW/WRONG/RISK, counterparty, failure mode) + frozen setup (`setup.yaml`: universe and eligibility, decision cadence and snapshot, execution delay, score-to-position mapping, constraints, material cost components, evaluation protocol with holdout dates) + footprint EDA (quintile sort, era split, rank autocorrelation).
- Must answer: Who is on the other side? Is the footprint monotonic, material, "lumpy but present" across eras? Is the ranking persistent enough for the cost class? Two clocks: which signal horizon feeds which holding horizon? Read the t-stat before the spread, and report in the horizon the design uses (a monthly sort is a monthly return; do not annualize it).
- Gate: EDA go/no-go passed; setup invariants written and versioned, with the comparability rule stated: any later change to tradability, decision schedule, score-to-trade mapping, constraints or cost treatment is a new `setup_version`, not a tuned trial; holdout sealed.
- References: `chapters/06_strategy_definition.md`, `case_studies/*.md` (setup contracts as templates).

### Stage 2. Labels (Ch7)
- Purpose: define the prediction target in execution-consistent terms.
- Inputs → outputs: validated prices + setup → label panel sealed on the label's outcome date (fixed-horizon forward returns anchored where the fill happens, triple-barrier/ATR barriers calibrated from MFE/MAE, trend-scanning, cross-sectional percentile labels with rolling cut-points), plus diagnostics (overlap, effective sample size ≈ N/h, class balance over time, implied trading intensity).
- Must answer: Which price does the fill happen at (close-to-close vs next-open)? How much do labels overlap? What turnover and breakeven cost does the label imply?
- Gate: label anchor matches execution; N_eff reported; barrier parameters calibrated, not round numbers.
- References: `chapters/07_defining_the_learning_task.md`, `libraries/ml4t_engineer.md`.

### Stage 3. Features (Ch8–Ch10)
- Purpose: translate driver hypotheses into feature specifications with explicit horizon alignment and roles (signal vs state).
- Inputs → outputs: prices, fundamentals, macro, text → feature panels: price-derived families (trend, reversal, volatility, liquidity, microstructure), structural/cross-instrument (carry, term structure, relative value, options-implied), contextual slow-moving (fundamentals, macro, calendar, PIT-lagged), model-based (stationarity diagnostics, Kalman, spectral, GARCH/HAR, HMM regime probabilities, uncertainty), text features (lexicon, TF-IDF, embeddings, FinBERT sentiment, event extraction).
- Must answer: What economic driver, at what horizon, in what role? Is every fitted feature refit inside the fold with filtered (not smoothed) states? Is slow data lagged by its publication delay? How many feature degrees of freedom were searched? Is a regime feature a risk lens or a timing signal? Whole-sample regime labels are lookahead; a label may be acted on only if it is fitted forward in time, built from inputs available at the decision date (initial-release vintages plus publication lag) and persists well beyond the rebalance cost horizon (an HMM with transition structure, not a mixture, for persistence); macro regimes confirm, they do not warn.
- Gate: features pass the admissibility audit (holdout-withheld rebuild agrees; leading nulls match declared lookbacks); search counted, meaning the feature degrees of freedom (features × horizons × configurations) are declared before results and carried into the BH-FDR / DSR denominators; within-family deduplication done.
- References: `chapters/08_financial_features.md`, `chapters/09_model_based_features.md`, `chapters/10_text_feature_engineering.md`, `libraries/ml4t_engineer.md`.

### Stage 4. Univariate evaluation and triage (Ch7 §7.3–7.5, Ch8 §8.6)
- Purpose: screen features before modeling, in an auditable ledger.
- Inputs → outputs: feature + label panel → triage ledger per feature: cross-sectional IC per date (mean, ICIR, HAC t, CI), quantile spread and monotonicity, sign consistency across folds, coverage/staleness, feasibility (turnover, breakeven cost, capacity), BH-FDR-adjusted p-value, decision PROCEED / REVISE / STOP.
- Must answer: Is the association real after HAC inference and FDR control over the searched set? Is it economically feasible? Is it a mechanism or a confounded proxy (causal sanity checks)?
- Gate: PROCEED only via FDR significance (BH at α = 0.05) or stability above a per-case effect-size floor (|mean IC| 0.003–0.01 with fold sign consistency); feasibility screen passed. Read the ledger as a screen on features, not a forecast of strategy survival: zero PROCEED features can still yield a usable model (joint prediction ≠ univariate IC), so an empty PROCEED set does not stop the cascade; PROCEED rates are comparable only within a case study (floors differ by 3x); BH-FDR is conservative under correlated menus (overlapping windows, related vol measures), which is why the stability route exists.
- References: `chapters/07_defining_the_learning_task.md`, `chapters/08_financial_features.md`, `libraries/ml4t_diagnostic.md`.

### Stage 5. Evaluation protocol (Ch6 §6.5)
- Purpose: a leak-free, chronological protocol that separates model selection from final estimation.
- Inputs → outputs: label horizon, feature lookbacks, available history and the holdout dates in `setup.yaml` → walk-forward folds (expanding or rolling) with label purge = horizon in sessions, feature embargo where training can follow validation, nested walk-forward for tuning, CPCV paths for robustness, sealed holdout window; all as configuration (`WalkForwardConfig`, `evaluation` block of `setup.yaml`).
- Must answer: Expanding or rolling (expanding if old data is still relevant, rolling if regimes change), and is the rolling window actually full (early folds are clipped to expanding when history is shorter than `train_size`; check each fold's train length)? Is the purge counted in sessions, not calendar days (pass `calendar="XNYS"`; a 21-calendar-day purge is about 14 NYSE sessions and leaves 7 sessions of leakage)? If nested tuning gives a different λ* in each outer fold, which value goes forward (none: the hyperparameter is unstable)? Which leakage channels does the splitter cover (only label leakage; standardization, threshold, survivorship and point-in-time leakage are handled in feature, label and universe construction, not by the CV)?
- Gate: purge arithmetic verified (`purged_sessions = va[0] - tr[-1] - 1 == h`); calendar-aware; holdout dates declared before any model run.
- References: `chapters/06_strategy_definition.md`, `libraries/ml4t_diagnostic.md`, `guardrails.md`, `decision_rules.md`.

### Stage 5b. Baseline checkpoint (Ch6 §6.6)
- Purpose: earn the right to widen the search. A deliberately narrow configuration (narrow in features and model class, using the setup's own score-to-position mapping) is run end to end through the protocol before any feature or family expansion; skipping it spends broad searches on an invalid setup and inflates trial counts.
- Inputs → outputs: frozen setup + protocol + a minimal feature and label panel → one registered baseline run with three sanity checks: timing (decisions and fills on the right bars, no lookahead), coverage (predictions exist for the intended universe and periods; cf. the prediction-coverage timeline, Figure 6.5), trading intensity (turnover consistent with the cost class and horizon).
- Must answer: Do fills land one execution delay after the decision snapshot? Does every asset-period the setup declares receive a prediction? Is implied turnover what the label horizon and cost class imply?
- Gate: all three checks pass; only then widen features or model class. A failed check is a setup or pipeline bug, not a modeling result.
- References: `chapters/06_strategy_definition.md` (Recipe 6), `companion_repo.md`.

### Stage 6. Model cascade (Ch11–Ch15)
- Purpose: learn the mapping from features to labels, with each family earning its complexity against the previous one on the same folds and days.
- Order: regularized linear baseline (Ridge/LASSO/Elastic Net; logistic for direction) → gradient boosting (LightGBM/XGBoost/CatBoost with Optuna, walk-forward HPO, monotone constraints, learning-to-rank where ranking is the task) → tabular deep models (TabM, TabPFN) → sequence models only after a shuffle diagnostic shows ordering carries signal (NLinear/DLinear, LSTM, TCN, TSMixer, PatchTST, iTransformer, Mamba) → latent factors (PCA, IPCA, RP-PCA, conditional/supervised autoencoders, SDF) → causal estimation (DML, BSTS, causal discovery) as a credibility filter.
- Outputs: registered training runs and prediction sets per fold and checkpoint; model analysis (IC with HAC inference, fold stability, checkpoint sensitivity, SHAP for plausibility and drift, conformal intervals for uncertainty).
- Must answer: Does the family beat the linear baseline on the same folds and days? Is the top configuration a thick cluster (many configs within one fold-SE) or a thin tail draw? Are hyperparameters stable across outer folds? Were micro-decisions (loss, penalty, stopping point, checkpoint) tuned on model and signal diagnostics, with strategy outcomes used only to confirm?
- Gate: positive validation IC with inference; stability bundle acceptable, meaning ICIR reported, positive-fold share close to n_folds, small checkpoint sensitivity, and a thick top (rank-1 to rank-10 validation Sharpe spread small relative to fold-SE = std(per-fold Sharpe) / √n_folds); no selection on the holdout.
- References: `chapters/11_ml_pipeline.md`, `chapters/12_gradient_boosting.md`, `chapters/13_dl_time_series.md`, `chapters/14_latent_factors.md`, `chapters/15_causal_estimation.md`, `libraries/ml4t_models.md`, `libraries/ml4t_diagnostic.md`, `evidence.md`.

### Stage 7. Backtest (Ch16)
- Purpose: convert predictions into a trading protocol and try to break it.
- Outputs: written protocol (signal timing, execution delay, fills, rounding, rebalancing mode, sizing, constraints, costs, benchmark); backtest runs (returns, weights, trades, fills, equity) registered per prediction set and entry scheme; report stack (gross and net, Sharpe with SE, drawdown, turnover, baseline comparison, regime slices, cost sensitivity); search-aware inference (Deflated Sharpe, PBO on CPCV paths, Rademacher Anti-Serum).
- Must answer: Is the non-ML baseline beaten on an identical protocol? How many strategies were looked at before this one? Does the engine choice change the answer (vectorized vs event-driven parity)?
- Gate: gross Sharpe > 0 against the equal-weight baseline on an identical protocol, with the paired stationary block bootstrap on the Sharpe difference meeting all three: CI excludes zero, `prob_challenger_wins` close to 1, small two-sided p-value (defaults n_boot=2000, seed=42, block length from `rebalance_step` and never below the label horizon; minimum series length 6 monthly / 12 weekly / 21 daily or faster). A CI straddling zero is "within block-bootstrap sampling error", not a failure. DSR credible for the trials run.
- References: `chapters/16_strategy_simulation.md`, `libraries/ml4t_backtest.md`.

### Stage 8. Portfolio construction (Ch17)
- Purpose: turn scores into weights without manufacturing a signal.
- Outputs: allocator term sheet; backtests per allocator on the same prediction (equal weight, inverse vol, score-weighted, risk parity, shrinkage MVO, HRP, conformal sizing, fractional Kelly; deep allocators evaluated against heuristics); allocator metrics (information ratio, active share, HHI, risk contributions, turnover and leverage stability).
- Must answer: Does any allocator beat equal weight broadly (more than half of them), or only one (selection bias)? Is the best–worst spread wide (construction is load-bearing, and the choice was made on validation) or tight (then decide on turnover, capacity and explicability rather than Sharpe)?
- Gate: allocator chosen on validation before the holdout; uplift breadth reported.
- References: `chapters/17_portfolio_construction.md`.

### Stage 9. Transaction costs and capacity (Ch18)
- Purpose: treat costs as a workflow constraint set by the instrument, not a backtest adjustment.
- Outputs: cost taxonomy for the asset class (explicit, implicit, capacity); spread estimates; impact calibration (linear, square-root, Kyle's lambda, participation of ADV); cost sweep; breakeven cost and cost-margin ratio; alpha-to-go and capacity estimate; execution-algorithm choice (TWAP/VWAP/participation/Almgren–Chriss); kill criteria tied to realized costs (TCA).
- Must answer: Net Sharpe at the cost the strategy actually assumes (interpolated at that cost, bracketing grid points reported)? Breakeven relative to assumed cost (`cost_margin_ratio` = breakeven / max(assumed, 1): ≥ 10 very robust, ≥ 3 robust, ≥ 1.5 marginal, otherwise fragile, < 1 does not survive; a curve still positive at the grid ceiling is censored, not a breakeven)? Capacity: at what AUM does impact equal the gross alpha, and where is the participation ceiling? Deployable AUM is the smaller of the two.
- Gate: net Sharpe at assumed cost > 0 with headroom; for spread-dominated instruments (single-name options, intraday equities) an executable-quote backtest, not a bps-of-notional sweep.
- References: `chapters/18_transaction_costs.md`.

### Stage 10. Risk management (Ch19)
- Purpose: make the strategy deployable: limits, overlays, kill switches and governance defined in advance and leak-free.
- Outputs: VaR/CVaR (historical, parametric, Cornish–Fisher, regime-conditional) with backtests; drawdown depth/duration/recovery; factor, sector, macro exposure decomposition; stress tests (historical replay, hypothetical shocks, reverse); adaptive controls (vol targeting, exposure caps, turnover tightening, position exits calibrated from in-sample MAE/MFE); kill switches and escalation rules; drift monitoring.
- Must answer: Which exposures are intended? What does the strategy do in the worst historical regime? Does an overlay improve Sharpe *and* drawdown out of sample, or is it insurance at a Sharpe cost?
- Gate: default is no overlay; adopt one only from the win-win quadrant confirmed out of sample, judged by the population of configurations, not best-of-sweep. Reading rules: compare by the single highest-Sharpe overlay row (a name such as `trailing_3pct` covers many parameterizations); record which baseline the delta is against (allocation stage, or the signal-stage fallback); do not rank overlays on small max-drawdown differences, since max drawdown is a single-path statistic with no interval.
- References: `chapters/19_risk_management.md`.

### Stage 11. Holdout and strategy synthesis (Ch20)
- Purpose: confirm once, then read the whole funnel diagnostically.
- Outputs: holdout predictions and holdout backtest for the single selected specification (signal + allocation + overlay + costs, retrained on history ending before the holdout; if the retrained holdout yields no usable backtest, walk to the next-highest validation Sharpe with a usable holdout and classify "Holdout not available" separately from "Holdout collapse"); cumulative gate record in pipeline order: IC > 0 → validation ML Sharpe of the selected configuration > 0 (read from its backtest, not from the risk-stage baseline, which is null where the risk stage does not apply) → net Sharpe at actual cost > 0, or cost n/a (option-native accounting) → holdout Sharpe > 0 → managed Sharpe > 0, or risk n/a (HTM and vectorized paths), and evidence resolved; drop only on a genuine negative at an applicable stage, and tally gates as passed / applicable; holdout decay = (val − holdout)/val, read through three mechanisms (prediction-quality drift, portfolio-translation drift, structural break or regime change); exclusion taxonomy (signal invalidity / implementation infeasibility / evidence-quality failure); next-iteration priorities with content: label refinement (horizon, winsorization, classification vs regression), domain features (order flow for NASDAQ-100, carry dynamics for CME, funding structure for crypto), focused tuning with larger budgets only on families that survived the holdout gate, simple-average ensembles guided by inter-family prediction correlation, strategy design (sector constraints, regime conditioning, dynamic sizing, multi-horizon blending).
- Must answer: Validation vs holdout Sharpe of the *same* configuration? Holdout IC and holdout Sharpe both positive? Does the CI exclude zero, or is the result statistically unresolved? Are the comparisons like with like: never validation max-IC (a maximum over a sweep) against holdout IC ("the figure manufactured the decay it was meant to measure"); `ic_mean` and `ic_best` on separate axes; IC compared within a label row, not across labels or instruments?
- Gate: holdout Sharpe > 0 with CI; decay < 50% "modest"; |worst drawdown| ≤ 50% (beyond that the outcome is "unacceptable drawdown", an implementation-infeasibility exclusion). A CI spanning zero is "statistically unresolved", not "failed".
- References: `chapters/20_strategy_synthesis.md`, `evidence.md`, `case_studies/*.md`.

### Stage 12. Deployment (Ch25)
- Purpose: run the researched strategy live without a second pipeline.
- Outputs: unified engine (same strategy class under `ml4t.backtest.Engine` and `ml4t.live.LiveEngine`), broker integration (Interactive Brokers and Alpaca adapters in `ml4t.live`; OKX data + Alpaca execution for crypto) plus a QuantConnect/LEAN prediction bridge (predictions exported as JSON for the platform to trade; not a live broker adapter), `SafeBroker` limits (order size, position, daily loss, rate, asset restrictions, kill switch persisted across restarts, shadow mode), order lifecycle state machine with idempotent recovery and reconciliation, staged pipeline-parity tests (data → features → predictions → sizing → orders), pre-flight checklist and staged rollout.
- Must answer: Which parts of a run were real (broker connection, fills, latency, rejections) and which simulated? How does the deployment artefact differ from the research artefact (hyperparameters cross over, trained weights do not; the live feature subset is what the broker feed can compute)? Does the deployment universe match the research universe, and is the gap written down?
- Gate: parity tests pass (unequal signal counts fail before any field comparison; identity fields exact, tolerance only on named float fields); shadow mode on a real broker connection for 1–2 weeks with signals matching backtest expectations, then paper for 2–4 weeks under the same `SafeBroker` configuration; operator checklist complete.
- References: `chapters/25_live_trading.md`, `libraries/ml4t_live.md`.

### Stage 13. Monitoring, governance and the feedback loop (Ch26, Ch19 §19.8)
- Purpose: detect decay and fail safely.
- Outputs: failure taxonomy (technical divergence vs statistical decay); rolling metrics and backtest-to-live realization ratio; drift detectors (PSI, K-S, SHAP drift, ADWIN-style and DDM online detectors) calibrated on a period they are not judged on; safe rollout (shadow → capital-capped A/B → staged rollout with pre-set promotion criteria and tested rollback); multi-level circuit breakers with a half-open state; feature store with PIT joins; experiment tracking/registry.
- Must answer: Is a divergence technical (same inputs, different outputs: feed, pipeline, broker) or statistical (same outputs, no longer predictive)? Was the promotion gate written before the shadow window was read? What trips which breaker, and who may reset it?
- Gate: promotion criteria (minimum Sharpe improvement, observation window, signal correlation, position agreement, drawdown ratio) fixed before evaluation; drift thresholds calibrated on a period they are not judged on; rollback tested; technical failure ruled out before any statistical-decay response.
- Decision loop: retrain (statistical decay with intact pipeline), pause (anomaly, breaker trip), retire (edge gone after re-research). Each outcome feeds back into Stage 1 as a new hypothesis.
- References: `chapters/26_mlops_governance.md`, `chapters/19_risk_management.md`.

### Cross-cutting tracks (use when relevant)
- Synthetic data for robustness (Ch5): classical baselines first; evaluate fidelity, utility (train-synthetic-test-real) and privacy before trusting a generator (`chapters/05_synthetic_data.md`).
- Reinforcement learning (Ch21): only for control problems with clear objectives and tight feedback (execution, market making, hedging); benchmark against TWAP/Almgren–Chriss/delta hedging; mind the sim-to-real gap (`chapters/21_rl_execution_hedging.md`).
- Generative AI (Ch22–Ch24; the failure modes the book names for it are leakage, hallucination and workflow bloat): RAG grounded in point-in-time filings with citations and separate retrieval/synthesis evaluation; knowledge graphs only when questions are multi-hop or temporal, with the three-timestamp model; agents read-only, with typed state, tool contracts, replay and proper scoring rules (`chapters/22_rag_financial_research.md`, `chapters/23_knowledge_graphs.md`, `chapters/24_autonomous_agents.md`).

## 3. Run log and registry discipline (Ch6 §6.7, companion repo)
- Trial taxonomy: strategy > trial family (one hypothesis with variants) > trial > run (one fitted configuration). Search is countable only in these units: "how many trials" means trials logged in the family, not configurations the researcher remembers.
- Configuration flows from broad to narrow (paths relative to the repo root): `case_studies/{id}/config/setup.yaml` (the problem: universe, cadence, costs, labels, evaluation, holdout) → `case_studies/{id}/config/training/{label}.yaml` (which presets to train per family) → `case_studies/config/{family}/{preset}.yaml` (hyperparameters, shared across case studies) → resolver binds data files, fold dates and rows → `SHA-256(canonical_json(identity))[:12]` names the run.
- Three hashed lineage levels, each carrying its parent id: training run → prediction set → backtest run, plus a `causal_runs` side table and supporting tables (fold metrics, paired metrics, candidate sets, official populations); every artifact (coefficients, boosters, checkpoints, predictions, returns, weights, trades, fills, equity, spec) lives under the hash.
- Rules the book enforces: nothing is anonymous or re-derivable only by re-running; populations are immutable and content-addressed; the identity includes the seed, the execution tier (canonical vs preview) and every preview reduction, so a changed seed or a preview run is a different run; the holdout is registered as `split='holdout'` and evaluated once; micro-decisions are tuned on model and signal diagnostics, the strategy is selected on validation backtest Sharpe with costs (never on IC alone), and strategy outcomes confirm rather than tune; any change to tradability, decision schedule, score-to-trade mapping, constraints or cost treatment is a new `setup_version`, not a tuned trial, because it breaks comparability; retired or superseded runs are recorded, not deleted.
- What to keep for every trial: full configuration hash, data vintage, fold geometry, per-fold metrics, selection decisions, what was reserved for confirmation, and the count of trials in the family (feeds DSR / BH-FDR / PBO).
- Reference implementation: `companion_repo.md`. The registry `case_studies/{id}/run_log/registry.db` is generated, not shipped: `run_log/` is created by running the pipeline or installed read-only from the artifact release (`scripts/download_artifacts.py`); `scripts/create_experiment.py` makes a writable experiment copy (`run_log` is in its `GENERATED_DIRS`); the `case_studies/research` package writes it; `BacktestExplorer` queries it.

## 4. How the nine case studies instantiate the workflow
| Case study | Market / cadence | Label | Cost class | Protocol (folds / train / val) | Holdout | Headline lesson |
|---|---|---|---|---|---|---|
| etfs | 100 multi-asset ETFs, monthly month-end | fwd_ret_21d | material (per-share + half spread) | 8 / 10Y / 1Y | 2024–2025 | broadest family comparison; IC and Sharpe leaders differ; allocation mediates prediction quality |
| crypto_perps_funding | 19 perpetuals, 8-hourly funding-aligned | fwd_ret_8h | material | 2 / 2Y / 1Y | 2024–2025 | funding-rate structure as the mechanism; decisions on the 8-hour funding clock (its model notebooks require CUDA) |
| nasdaq100_microstructure | 115 stocks (`n_assets` in `setup.yaml`; the Ch6 overview notebook prints 114), 15-minute | fwd_ret_15m | dominant | 2 / 6M / 6M | H2 2021 | order-flow signals meet the cost cliff; intervals cross zero |
| sp500_equity_option_analytics | ~630 stocks, weekly Friday close | fwd_ret_5d | material (bps) | 2 / 2Y / 1Y | 2021 | options-implied features for equity selection |
| us_firm_characteristics | ~2,500 stocks, monthly | fwd_ret_1m | material | 10 / 10YE / 1YE | 2016 | home of the latent-factor family (IPCA, CAE, SAE, SDF); target and loss choice move IC more than features or capacity |
| fx_pairs | 20 G10 pairs, daily NY close | fwd_ret_1d | material (single-digit bps) | 8 / 5Y / 1Y | 2024–2025 | tightest spreads allow daily decisions; momentum (carry is named in the setup, but the spot feed carries no rate differential); judged on interval evidence |
| cme_futures | 30 products, weekly Friday close | fwd_ret_5d | material (per contract + ticks) | 5 / 8Y / 1Y | 2024–2025 | carry/term structure; two price series (adjusted for returns, raw for carry) |
| sp500_options | S&P 500 ATM straddles on 627 underlyings; weekly Friday entry, daily-close delta hedge | ret_to_expiry (hold-to-maturity; `fwd_ret_dh_10d` and the other fixed-horizon labels are diagnostic only, not in the sweep) | dominant (spread vs premium) | 2 / 2Y / 1Y | 2021 | net-negative under realistic option costs; executable-quote accounting |
| us_equities_panel | ~3,200 stocks, daily | fwd_ret_1d | material (percentage) | 16 / 10Y / 1Y | 2016–Q1 2018 | broad cross-section; drift and MLOps notebooks run on its artifacts |

Each case study runs the same numbered stages: feasibility → labels → financial features → model-based features → evaluation → linear → GBM → tabular DL → sequence DL → latent factors → causal DML → model analysis → backtest → portfolio → risk → costs → holdout predictions → holdout backtest → strategy analysis (`case_studies/*.md`).

## 5. Common ways the workflow breaks, and the book's response
| Failure | Mechanism | Response |
|---|---|---|
| Lookahead through preprocessing | scalers, winsor bounds, imputers, thresholds, regime models fitted on the full sample | fit on the training fold only, refit per fold, filtered not smoothed states; holdout-withheld rebuild must agree |
| Label overlap treated as independence | N_eff ≈ N/h; HAC-free t-stats too large | purge = horizon; HAC / block bootstrap; report N_eff |
| Survivorship and point-in-time errors | today's constituents, revised macro, restated fundamentals | per-period universes, initial-release vintages, publication lags, bitemporal storage |
| Selection optimism | best of a sweep reported as the estimate | log trials; DSR / BH-FDR / PBO; thick-vs-thin top cluster; paired bootstrap vs equal weight |
| IC-to-Sharpe decoupling | turnover, breadth, cadence and allocation sit between ranking and P&L | select on validation backtest Sharpe with costs; never deploy on IC |
| Cost model mismatch | bps-of-notional on spread-dominated instruments | executable quotes, option-native accounting, three-label decomposition |
| Overlay overfitting | best-of-sweep overlay cuts both tails and trades more | default no overlay; population statistics; win-win quadrant out of sample |
| Holdout contamination | selection after seeing holdout; ex-post ensemble "rescue" | freeze everything before the holdout; fix ensembles before scoring |
| Engine disagreement | fills, cash release, rounding, rebalance mode | protocol parity tests; one field at a time |
| Research-production divergence | two pipelines drift | unified engine, pipeline parity tests, shadow mode |
| Silent decay | regimes shift, competitors arrive | drift detectors with pre-set thresholds, circuit breakers, retrain/pause/retire loop |
| Partial or stale aggregation | a subset run stamped as production; typed-in constants | refuse partial runs; compute every number from the registry |
| Regime label used as a timing signal | whole-sample GMM/HMM labels are informed by later months; macro labels arrive with or after the event | risk lens tied to predefined actions; actionable only if fitted forward, built from initial-release vintages with publication lag, and persistent well beyond the rebalance cost horizon |
| Tuning on portfolio outcomes | every micro-decision optimized on simulated P&L | three metric roles: tune on model and signal diagnostics, confirm on strategy outcomes |
| Mechanics drift masquerading as tuning | tradability, schedule, mapping, constraints or cost treatment changed inside a "tuned" trial | new `setup_version`; classify every change as parameter tuning vs mechanics change |
| Manufactured holdout decay | validation max-IC over a sweep compared with one holdout IC | same configuration validation vs holdout; holdout IC vs holdout Sharpe; `ic_mean` and `ic_best` on separate axes |

## 6. Minimal sequence for a new strategy (practitioner checklist)
1. Write the hypothesis as a falsifiable statement with the disconfirming outcome named before testing, plus source of edge, counterparty and failure mode (Stage 1).
2. Freeze `setup.yaml`-style invariants including holdout dates and cost model (Stage 1).
3. Build PIT-correct data; run the data-quality framework (Stage 0).
4. Define execution-consistent labels; report overlap and N_eff (Stage 2).
5. Build feature families with declared lookbacks and roles; refit fitted features per fold (Stage 3).
6. Triage features with IC + HAC + BH-FDR + feasibility; keep the ledger (Stage 4).
7. Run the baseline checkpoint: a narrow configuration through the walk-forward protocol with timing, coverage and trading-intensity checks; register it (Stages 5–5b).
8. Fit the linear baseline, then escalate families only where they beat it on the same folds and days; log every trial (Stage 6).
9. Backtest with a written protocol against a non-ML baseline; deflate for trials (Stage 7).
10. Compare allocators on the same prediction; choose before the holdout (Stage 8).
11. Sweep costs at the assumed cost; compute breakeven ratio and capacity (Stage 9).
12. Add overlays only from the win-win quadrant; define kill criteria (Stage 10).
13. Open the holdout once; classify the outcome; write next-iteration priorities (Stage 11).
14. Deploy through the unified engine with parity tests, shadow mode and staged rollout (Stage 12).
15. Monitor drift and realization ratio; retrain, pause or retire by pre-set rules (Stage 13).

## Related references
- `guardrails.md` — pre-flight checklist and the stage-grouped guardrail catalog behind every gate above.
- `decision_rules.md` — the numeric thresholds this map cites (purge and embargo, FDR alpha and effect-size floors, bootstrap defaults, cost-resilience ladder, overlay quadrant, promotion gates).
- `evidence.md` — what the nine case studies found at each stage; `case_studies/*.md` — setup contracts and stage tables to copy.
- `companion_repo.md` — registry, config layers and scripts; `glossary.md` — vocabulary (trial family, spine, rung, evidence boundary, PROCEED).
- `chapters/01_process_is_edge.md`, `chapters/06_strategy_definition.md`, `chapters/20_strategy_synthesis.md`, `chapters/27_systematic_edge.md` — the chapters this map is distilled from.
