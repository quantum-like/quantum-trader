# The ML4T workflow: research to production, end to end
> The book runs one process across 27 chapters and nine markets: build point-in-time-correct data, define a strategy as a decision process, turn data into labels and features under decision-time admissibility, learn in a model cascade that starts linear, cross an *evidence boundary* into backtests, portfolios, costs and risk overlays, confirm once on a sealed holdout, deploy the same code that was researched, and monitor until the edge decays and the strategy is retrained, paused or retired. This file is the map: for each stage, what it must answer, what it produces, its go/no-go gate, and which reference to open.

## 1. The thesis
- **The process is the edge.** Markets change (structural breaks, regimes, data drift, concept drift); any single signal decays. Durable performance comes from a repeatable process that keeps generating candidates and, more importantly, keeps rejecting bad ones (`chapters/01_process_is_edge.md`, `chapters/27_systematic_edge.md`).
- **Evidence boundary.** Exploration (features, models, allocators, overlays, cost headroom) happens on training and validation folds with every trial logged. Confirmation happens once, on a sealed holdout, with everything frozen. A second look at the holdout turns it into validation data.
- **Falsification, not demonstration.** A backtest is an attempt to break the strategy under realistic assumptions (timing, fills, costs, regimes, search size); what survives is what gets reported (`chapters/16_strategy_simulation.md`).
- **Three defenses against the researcher's own biases** (ch27): falsifiable hypotheses stated before testing; rigorous out-of-sample evaluation; statistical corrections for the number of trials.
- **Four separate evaluation questions** (ch20): signal quality (does the score order returns?), portfolio translation (does ranking become Sharpe?), cost survival (does the edge exceed the cost gate set by the instrument?), temporal stability (does it hold out of sample?). A strategy can pass one and fail the next; the funnel is the story.
- **Absence of a measurement is not a measurement of zero.** Report what was not attempted as "not attempted", never as a failure or a pass.

## 2. Stage-by-stage walkthrough

Each stage lists: purpose · inputs → outputs · questions it must answer · gate · references.

### Stage 0. Data infrastructure (Ch2–Ch4)
- Purpose: a point-in-time-correct, survivorship-free, identifier-consistent data layer that the rest of the workflow can trust.
- Inputs → outputs: vendor/raw files → validated, adjusted, as-of-queryable datasets (Parquet/Hive storage, DuckDB for analytics; `ml4t-data` loaders), data-quality report, dataset cards.
- Must answer: What does each timestamp mean (event, disclosure, extraction)? Which revision of each value was visible when? Which corporate actions and identifier changes are handled, and how? Is the universe defined per period?
- Gate: data-quality framework passes (OHLC invariants, coverage, PIT validation, survivorship test); storage fits the access pattern.
- References: `chapters/02_financial_data_universe.md`, `chapters/03_market_microstructure.md` (intraday data, bars, LOB), `chapters/04_fundamental_alternative_data.md` (bitemporal fundamentals, entity resolution, macro release timing, alt-data due diligence), `libraries/ml4t_data.md`, `companion_repo.md` (data loaders, `ML4T_DATA_PATH`).

### Stage 1. Strategy framing and trading setup (Ch6)
- Purpose: turn an idea into a testable economic hypothesis and a versioned decision process.
- Inputs → outputs: idea → strategy map (family, source of edge SLOW/WRONG/RISK, counterparty, failure mode) + frozen setup (`setup.yaml`: universe and eligibility, decision cadence and snapshot, execution delay, score-to-position mapping, constraints, material cost components, evaluation protocol with holdout dates) + footprint EDA (quintile sort, era split, rank autocorrelation).
- Must answer: Who is on the other side? Is the footprint monotonic, material, "lumpy but present" across eras? Is the ranking persistent enough for the cost class?
- Gate: EDA go/no-go passed; setup invariants written and versioned; holdout sealed.
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
- Must answer: What economic driver, at what horizon, in what role? Is every fitted feature refit inside the fold with filtered (not smoothed) states? Is slow data lagged by its publication delay? How many feature degrees of freedom were searched?
- Gate: features pass the admissibility audit (holdout-withheld rebuild agrees; leading nulls match declared lookbacks); search counted; within-family deduplication done.
- References: `chapters/08_financial_features.md`, `chapters/09_model_based_features.md`, `chapters/10_text_feature_engineering.md`, `libraries/ml4t_engineer.md`.

### Stage 4. Univariate evaluation and triage (Ch7 §7.3–7.5, Ch8 §8.6)
- Purpose: screen features before modeling, in an auditable ledger.
- Inputs → outputs: feature + label panel → triage ledger per feature: cross-sectional IC per date (mean, ICIR, HAC t, CI), quantile spread and monotonicity, sign consistency across folds, coverage/staleness, feasibility (turnover, breakeven cost, capacity), BH-FDR-adjusted p-value, decision PROCEED / REVISE / STOP.
- Must answer: Is the association real after HAC inference and FDR control over the searched set? Is it economically feasible? Is it a mechanism or a confounded proxy (causal sanity checks)?
- Gate: PROCEED only via FDR significance or stability above an effect-size floor; feasibility screen passed.
- References: `chapters/07_defining_the_learning_task.md`, `chapters/08_financial_features.md`, `libraries/ml4t_diagnostic.md`.

### Stage 5. Evaluation protocol (Ch6 §6.5)
- Purpose: a leak-free, chronological protocol that separates model selection from final estimation.
- Outputs: walk-forward folds (expanding or rolling) with label purge = horizon in sessions, feature embargo where training can follow validation, nested walk-forward for tuning, CPCV paths for robustness, sealed holdout window; all as configuration (`WalkForwardConfig`, `evaluation` block of `setup.yaml`).
- Gate: purge arithmetic verified; calendar-aware; holdout dates declared before any model run.
- References: `chapters/06_strategy_definition.md`, `libraries/ml4t_diagnostic.md`, `guardrails.md`.

### Stage 6. Model cascade (Ch11–Ch15)
- Purpose: learn the mapping from features to labels, with each family earning its complexity against the previous one on the same folds and days.
- Order: regularized linear baseline (Ridge/LASSO/Elastic Net; logistic for direction) → gradient boosting (LightGBM/XGBoost/CatBoost with Optuna, walk-forward HPO, monotone constraints, learning-to-rank where ranking is the task) → tabular deep models (TabM, TabPFN) → sequence models only after a shuffle diagnostic shows ordering carries signal (NLinear/DLinear, LSTM, TCN, TSMixer, PatchTST, iTransformer, Mamba) → latent factors (PCA, IPCA, RP-PCA, conditional/supervised autoencoders, SDF) → causal estimation (DML, BSTS, causal discovery) as a credibility filter.
- Outputs: registered training runs and prediction sets per fold and checkpoint; model analysis (IC with HAC inference, fold stability, checkpoint sensitivity, SHAP for plausibility and drift, conformal intervals for uncertainty).
- Must answer: Does the family beat the linear baseline on the same folds and days? Is the top configuration a thick cluster (many configs within one fold-SE) or a thin tail draw? Are hyperparameters stable across outer folds?
- Gate: positive validation IC with inference; stability bundle acceptable; no selection on the holdout.
- References: `chapters/11_ml_pipeline.md`, `chapters/12_gradient_boosting.md`, `chapters/13_dl_time_series.md`, `chapters/14_latent_factors.md`, `chapters/15_causal_estimation.md`, `libraries/ml4t_models.md`, `libraries/ml4t_diagnostic.md`, `evidence.md`.

### Stage 7. Backtest (Ch16)
- Purpose: convert predictions into a trading protocol and try to break it.
- Outputs: written protocol (signal timing, execution delay, fills, rounding, rebalancing mode, sizing, constraints, costs, benchmark); backtest runs (returns, weights, trades, fills, equity) registered per prediction set and entry scheme; report stack (gross and net, Sharpe with SE, drawdown, turnover, baseline comparison, regime slices, cost sensitivity); search-aware inference (Deflated Sharpe, PBO on CPCV paths, Rademacher Anti-Serum).
- Must answer: Is the non-ML baseline beaten on an identical protocol? How many strategies were looked at before this one? Does the engine choice change the answer (vectorized vs event-driven parity)?
- Gate: gross Sharpe > 0 against baseline with a paired bootstrap CI that excludes zero; DSR credible for the trials run.
- References: `chapters/16_strategy_simulation.md`, `libraries/ml4t_backtest.md`.

### Stage 8. Portfolio construction (Ch17)
- Purpose: turn scores into weights without manufacturing a signal.
- Outputs: allocator term sheet; backtests per allocator on the same prediction (equal weight, inverse vol, score-weighted, risk parity, shrinkage MVO, HRP, conformal sizing, fractional Kelly; deep allocators evaluated against heuristics); allocator metrics (information ratio, active share, HHI, risk contributions, turnover and leverage stability).
- Must answer: Does any allocator beat equal weight broadly (more than half of them), or only one (selection bias)? Is the best–worst spread wide (construction is load-bearing)?
- Gate: allocator chosen on validation before the holdout; uplift breadth reported.
- References: `chapters/17_portfolio_construction.md`.

### Stage 9. Transaction costs and capacity (Ch18)
- Purpose: treat costs as a workflow constraint set by the instrument, not a backtest adjustment.
- Outputs: cost taxonomy for the asset class (explicit, implicit, capacity); spread estimates; impact calibration (linear, square-root, Kyle's lambda, participation of ADV); cost sweep; breakeven cost and cost-margin ratio; alpha-to-go and capacity estimate; execution-algorithm choice (TWAP/VWAP/participation/Almgren–Chriss); kill criteria tied to realized costs (TCA).
- Must answer: Net Sharpe at the cost the strategy actually assumes? Breakeven relative to assumed cost (ratio ≥ 3 robust, < 1 does not survive)? At what AUM does impact eat half the alpha?
- Gate: net Sharpe at assumed cost > 0 with headroom; for spread-dominated instruments (single-name options, intraday equities) an executable-quote backtest, not a bps-of-notional sweep.
- References: `chapters/18_transaction_costs.md`.

### Stage 10. Risk management (Ch19)
- Purpose: make the strategy deployable: limits, overlays, kill switches and governance defined in advance and leak-free.
- Outputs: VaR/CVaR (historical, parametric, Cornish–Fisher, regime-conditional) with backtests; drawdown depth/duration/recovery; factor, sector, macro exposure decomposition; stress tests (historical replay, hypothetical shocks, reverse); adaptive controls (vol targeting, exposure caps, turnover tightening, position exits calibrated from in-sample MAE/MFE); kill switches and escalation rules; drift monitoring.
- Must answer: Which exposures are intended? What does the strategy do in the worst historical regime? Does an overlay improve Sharpe *and* drawdown out of sample, or is it insurance at a Sharpe cost?
- Gate: default is no overlay; adopt one only from the win-win quadrant confirmed out of sample, judged by the population of configurations, not best-of-sweep.
- References: `chapters/19_risk_management.md`.

### Stage 11. Holdout and strategy synthesis (Ch20)
- Purpose: confirm once, then read the whole funnel diagnostically.
- Outputs: holdout predictions and holdout backtest for the single selected specification (signal + allocation + overlay + costs, retrained on history ending before the holdout); cumulative gate record (IC > 0 → validation ML Sharpe > 0 → net Sharpe at actual cost > 0 → holdout Sharpe > 0 → managed Sharpe > 0 and evidence resolved); holdout decay = (val − holdout)/val; exclusion taxonomy (signal invalidity / implementation infeasibility / evidence-quality failure); next-iteration priorities.
- Must answer: Validation vs holdout Sharpe of the *same* configuration? Holdout IC and holdout Sharpe both positive? Does the CI exclude zero, or is the result statistically unresolved?
- Gate: holdout Sharpe > 0 with CI; decay < 50% "modest"; drawdown tolerable. A CI spanning zero is "unresolved", not "failed".
- References: `chapters/20_strategy_synthesis.md`, `evidence.md`, `case_studies/*.md`.

### Stage 12. Deployment (Ch25)
- Purpose: run the researched strategy live without a second pipeline.
- Outputs: unified engine (same strategy class under `ml4t.backtest.Engine` and `ml4t.live.LiveEngine`), broker integration (Interactive Brokers, Alpaca, managed platforms such as QuantConnect), `SafeBroker` limits (order size, position, daily loss, rate, asset restrictions, kill switch persisted across restarts, shadow mode), order lifecycle state machine with idempotent recovery and reconciliation, staged pipeline-parity tests (data → features → predictions → sizing → orders), pre-flight checklist and staged rollout.
- Gate: parity tests pass; shadow trading matches the reference tape; operator checklist complete.
- References: `chapters/25_live_trading.md`, `libraries/ml4t_live.md`.

### Stage 13. Monitoring, governance and the feedback loop (Ch26, Ch19 §19.8)
- Purpose: detect decay and fail safely.
- Outputs: failure taxonomy (technical divergence vs statistical decay); rolling metrics and backtest-to-live realization ratio; drift detectors (PSI, K-S, SHAP drift, ADWIN-style and DDM online detectors) calibrated on a period they are not judged on; safe rollout (shadow → capital-capped A/B → staged rollout with pre-set promotion criteria and tested rollback); multi-level circuit breakers with a half-open state; feature store with PIT joins; experiment tracking/registry.
- Decision loop: retrain (statistical decay with intact pipeline), pause (anomaly, breaker trip), retire (edge gone after re-research). Each outcome feeds back into Stage 1 as a new hypothesis.
- References: `chapters/26_mlops_governance.md`, `chapters/19_risk_management.md`.

### Cross-cutting tracks (use when relevant)
- Synthetic data for robustness (Ch5): classical baselines first; evaluate fidelity, utility (train-synthetic-test-real) and privacy before trusting a generator (`chapters/05_synthetic_data.md`).
- Reinforcement learning (Ch21): only for control problems with clear objectives and tight feedback (execution, market making, hedging); benchmark against TWAP/Almgren–Chriss/delta hedging; mind the sim-to-real gap (`chapters/21_rl_execution_hedging.md`).
- Generative AI (Ch22–Ch24): RAG grounded in point-in-time filings with citations and separate retrieval/synthesis evaluation; knowledge graphs only when questions are multi-hop or temporal, with the three-timestamp model; agents read-only, with typed state, tool contracts, replay and proper scoring rules (`chapters/22_rag_financial_research.md`, `chapters/23_knowledge_graphs.md`, `chapters/24_autonomous_agents.md`).

## 3. Run log and registry discipline (Ch6 §6.7, companion repo)
- Configuration flows from broad to narrow: `config/setup.yaml` (case study) → `config/training/{label}.yaml` (which presets to train) → `case_studies/config/{family}/{preset}.yaml` (hyperparameters) → resolver binds data files, fold dates and rows → `SHA-256(canonical_json(identity))[:12]` names the run.
- Three-level entity model: training run → prediction set → backtest run, plus a causal-runs side table; every artifact (coefficients, boosters, checkpoints, predictions, returns, weights, trades, fills, equity, spec) lives under the hash.
- Rules the book enforces: nothing is anonymous or re-derivable only by re-running; populations are immutable and content-addressed; a changed device, seed or preview reduction changes the identity; the holdout is registered as `split='holdout'` and evaluated once; selection happens on validation backtest Sharpe, never on IC alone; retired or superseded runs are recorded, not deleted.
- What to keep for every trial: full configuration hash, data vintage, fold geometry, per-fold metrics, selection decisions, what was reserved for confirmation, and the count of trials in the family (feeds DSR / BH-FDR / PBO).
- Reference implementation: `companion_repo.md` (`case_studies/{id}/run_log/registry.db`, `case_studies/research` package, `scripts/create_experiment.py` for writable experiment copies, `BacktestExplorer` for queries).

## 4. How the nine case studies instantiate the workflow
| Case study | Market / cadence | Label | Cost class | Protocol (folds / train) | Holdout | Headline lesson |
|---|---|---|---|---|---|---|
| etfs | 100 multi-asset ETFs, monthly month-end | fwd_ret_21d | material (per-share + half spread) | 8 / 10Y | 2024–2025 | broadest family comparison; IC and Sharpe leaders differ; allocation mediates prediction quality |
| crypto_perps_funding | 19 perpetuals, 8-hourly funding-aligned | fwd_ret_8h | material | 2 / 2Y | 2024–2025 | funding-rate structure as the mechanism; GPU-only model stages |
| nasdaq100_microstructure | 114 stocks, 15-minute | fwd_ret_15m | dominant | 2 / 6M | H2 2021 | order-flow signals meet the cost cliff; intervals cross zero |
| sp500_equity_option_analytics | ~630 stocks, weekly | fwd_ret_5d | material (bps) | 2 / 2Y | 2021 | options-implied features for equity selection |
| us_firm_characteristics | ~2,500 stocks, monthly | fwd_ret_1m | material | 10 / 10Y | 2016 | latent factors, IPCA, autoencoders, SDF on a characteristics panel |
| fx_pairs | 20 G10 pairs, daily | fwd_ret_1d | material (single-digit bps) | 8 / 5Y | 2024–2025 | tightest spreads allow daily decisions; carry and momentum |
| cme_futures | 30 products, weekly Friday close | fwd_ret_5d | material (per contract + ticks) | 5 / 8Y | 2024–2025 | carry/term structure; two price series (adjusted for returns, raw for carry) |
| sp500_options | S&P 500 straddles, daily | fwd_ret_dh_10d / hold-to-maturity | dominant (spread vs premium) | 2 / 2Y | 2021 | net-negative under realistic option costs; executable-quote accounting |
| us_equities_panel | ~3,200 stocks, daily | fwd_ret_1d | material (percentage) | 16 / 10Y | 2016–Q1 2018 | broad cross-section; drift and MLOps notebooks run on its artifacts |

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

## 6. Minimal sequence for a new strategy (practitioner checklist)
1. Write the hypothesis, source of edge, counterparty and failure mode (Stage 1).
2. Freeze `setup.yaml`-style invariants including holdout dates and cost model (Stage 1).
3. Build PIT-correct data; run the data-quality framework (Stage 0).
4. Define execution-consistent labels; report overlap and N_eff (Stage 2).
5. Build feature families with declared lookbacks and roles; refit fitted features per fold (Stage 3).
6. Triage features with IC + HAC + BH-FDR + feasibility; keep the ledger (Stage 4).
7. Fit the linear baseline on walk-forward folds; register it (Stages 5–6).
8. Escalate families only where they beat the baseline on the same folds; log every trial (Stage 6).
9. Backtest with a written protocol against a non-ML baseline; deflate for trials (Stage 7).
10. Compare allocators on the same prediction; choose before the holdout (Stage 8).
11. Sweep costs at the assumed cost; compute breakeven ratio and capacity (Stage 9).
12. Add overlays only from the win-win quadrant; define kill criteria (Stage 10).
13. Open the holdout once; classify the outcome; write next-iteration priorities (Stage 11).
14. Deploy through the unified engine with parity tests, shadow mode and staged rollout (Stage 12).
15. Monitor drift and realization ratio; retrain, pause or retire by pre-set rules (Stage 13).
