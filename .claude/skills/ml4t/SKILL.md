---
name: ml4t
description: "Machine Learning for Trading (Stefan Jansen, 3rd ed., 2026) distilled into an operating procedure for quant research: the end-to-end ML4T workflow, its leakage / lookahead / multiple-testing guardrails, decision rules, nine case studies and the six ml4t-* Python libraries. Use whenever a task touches systematic or algorithmic trading research in any form: market, fundamental, alternative or prediction-market data pipelines and point-in-time correctness; bars and microstructure; labels and feature engineering; walk-forward or purged CV; IC, BH-FDR, Deflated Sharpe, PBO; linear, GBM, deep, latent-factor or causal return models; backtesting; portfolio construction; transaction costs and capacity; risk, VaR/CVaR, kill switches; live trading, MLOps, drift; RL for execution; RAG or LLM agents for financial research. Trigger even when the user never says ML4T or the book, and for narrow asks such as reviewing a backtest, adding costs, designing a CV split, or auditing a ranking model for leakage."
---

# ML4T: Machine Learning for Trading as an operating procedure

This skill turns Stefan Jansen's *Machine Learning for Trading, 3rd edition* (27 chapters, nine case
studies, six `ml4t-*` libraries) into something you can execute. It was built from the book's
companion repository (`stefan-jansen/machine-learning-for-trading`: chapter READMEs, Jupytext
notebooks, case-study pipelines) and the published library sources, not from the PDF, so cite
notebooks by repo path and treat numbers as "as of the repository", which the author says supersedes
the printed book where they differ.

The book's thesis, which this skill enforces: **the process is the edge.** A trading result is
credible only if it was produced by a protocol that (1) used only information available at decision
time, (2) kept exploration and confirmation on opposite sides of an *evidence boundary*, (3) counted
every trial it ran, (4) charged realistic costs, and (5) survived an honest attempt to break it. A
backtest is a falsification test, not a demo.

## How to use this skill

All paths below are relative to this skill's directory; inside `references/`, every "Related
references" entry is relative to `references/`.

1. Place the task on the workflow map below and read the stage's gate questions.
2. Open the primary reference for that stage (bold in the table) before writing code or conclusions;
   open secondary files only for the recipe or API you need. Each chapter file is self-contained
   (recipes, guardrails, defaults, APIs, evidence); the cross-cutting catalogs are large and are for
   lookup, not sequential reading. Read one file at a time: start with its "When to use this
   reference" block, then Grep for the recipe, guardrail or API you need.
3. Run `references/preflight.md` (the 35-question pre-flight checklist) before trusting or
   reporting any research number. Fix protocol problems before tuning anything. The catalog behind
   each question is indexed by `references/guardrails.md`, which maps each section to one of three
   part files (data and features; evaluation and backtest; portfolio, costs, risk and live); open
   one part and Grep it for the item.
4. When a method decision comes up (which label, which split, which model family first, which
   allocator, which cost model), look it up in `references/decision_rules.md` and cite the rule.
5. When you need to say "what usually works", use `references/evidence.md` as priors, with the
   case-study context attached. Do not generalize a single case study.
6. When implementing, prefer the `ml4t-*` library primitives (see `references/libraries/`) or the
   companion repo's `utils/` and `case_studies/utils/` patterns over ad-hoc code; clone the repo when
   you need the exact notebook (`git clone --depth 1 https://github.com/stefan-jansen/machine-learning-for-trading.git`).

## The workflow map (what to open at each stage)

Stage numbers follow `references/workflow.md`, which has the full purpose / inputs / outputs / gate
text for each stage. The bold file in each row is the primary reference; the others are secondary. Library files are listed where their APIs are the default implementation.

| Stage | Gate questions you must be able to answer | References |
|---|---|---|
| 0. Data infrastructure (Ch2-4) | Point-in-time correct, survivorship-free universe, adjustments and identifiers time-valid? Feed, session boundaries and bar type match the hypothesis? Option chains: IV convergence codes and Greeks validated? Full gate: `references/workflow.md` Stage 0. | **`references/chapters/02_financial_data_universe.md`**, `references/chapters/03_market_microstructure.md`, `references/chapters/04_fundamental_alternative_data.md`, `references/libraries/ml4t_data.md`, `references/libraries/ml4t_engineer.md` (bars), `references/companion_repo.md` |
| 1. Strategy framing and trading setup (Ch6, Ch27) | Source of edge (SLOW / WRONG / RISK) and failure modes named? Pre-registered hypothesis with its disconfirming result written down? Versioned trading setup (universe rule, decision schedule, admissible information, score-to-position map, cost class, benchmark)? Full gate: Stage 1. | `references/workflow.md`, **`references/chapters/06_strategy_definition.md`**, `references/chapters/27_systematic_edge.md`, `references/chapters/01_process_is_edge.md` |
| 2. Labels (Ch7) | Horizon and anchor execution-consistent (next open, not the signal close)? Overlap handled (purge, uniqueness weights, HAC)? Implied turnover and breakeven cost known? Full gate: Stage 2. | **`references/chapters/07_defining_the_learning_task.md`**, `references/libraries/ml4t_engineer.md` |
| 3. Features (Ch8-10) | Driver hypothesis, horizon alignment and role (signal vs state) per feature? Fitted features refit inside folds with filtered states; slow data lagged by publication? Search degrees of freedom counted? Full gate: Stage 3. | **`references/chapters/08_financial_features.md`**, `references/chapters/09_model_based_features.md`, `references/chapters/10_text_feature_engineering.md`, `references/libraries/ml4t_engineer.md` |
| 4. Univariate evaluation and triage (Ch7, Ch8) | Per-feature ledger (IC, ICIR, HAC t-stat, quantile spread, sign consistency, feasibility) with BH-FDR over the whole searched set and a PROCEED / REVISE / STOP decision, read as a feature screen rather than a strategy forecast? Full gate: Stage 4. | **`references/chapters/07_defining_the_learning_task.md`**, `references/chapters/08_financial_features.md`, `references/libraries/ml4t_diagnostic.md` |
| 5. Evaluation protocol (Ch6) | Calendar-aware walk-forward with purge = horizon in sessions, embargo where training can follow the test block, nested tuning or CPCV as needed? Holdout dates declared before any model run; trials logged? Full gate: Stage 5. | **`references/chapters/06_strategy_definition.md`**, `references/libraries/ml4t_diagnostic.md`, `references/guardrails.md`, `references/decision_rules.md` |
| 5b. Baseline checkpoint (Ch6) | One narrow configuration run end to end and registered; timing, coverage and trading-intensity checks passed before widening the search? Full gate: Stage 5b. | **`references/chapters/06_strategy_definition.md`**, `references/companion_repo.md` |
| 6. Model cascade (Ch11-15) | Linear baseline first; does the next family beat it on the same folds and days with HAC inference? Uncertainty attached; sequence models only after a shuffle diagnostic? Key associations checked as mechanism vs confounded proxy? Full gate: Stage 6. | **`references/chapters/11_ml_pipeline.md`**, `references/chapters/12_gradient_boosting.md`, `references/chapters/13_dl_time_series.md`, `references/chapters/14_latent_factors.md`, `references/chapters/15_causal_estimation.md`, `references/libraries/ml4t_models.md`, `references/libraries/ml4t_diagnostic.md` |
| 7. Backtest (Ch16) | Protocol written (timing, fills, sizing, costs, constraints, benchmark); vectorized and event-driven engines agree; gross and net, turnover, drawdown, regime slices and cost sensitivity reported; Sharpe deflated for the trials run (DSR / PBO / RAS)? Full gate: Stage 7. | **`references/chapters/16_strategy_simulation.md`**, `references/libraries/ml4t_backtest.md`, `references/libraries/ml4t_diagnostic.md` |
| 8. Portfolio construction (Ch17) | Allocator beats equal weight / inverse vol on identical inputs; concentration, turnover and leverage stable; allocator chosen before holdout? Full gate: Stage 8. | **`references/chapters/17_portfolio_construction.md`**, `references/libraries/ml4t_models.md` (portfolio learners) |
| 9. Costs and capacity (Ch18) | Spread + impact modeled for the asset class and venue; breakeven cost, minimum required edge and capacity (where impact eats half the alpha) reported; execution schedule chosen? Full gate: Stage 9. | **`references/chapters/18_transaction_costs.md`**, `references/libraries/ml4t_backtest.md` (cost models) |
| 10. Risk (Ch19) | VaR / CVaR with Kupiec backtests and path risk measured; exposures decomposed; stress tests run; overlays leakage-free and precommitted; kill criteria written as rules with thresholds, windows and reset logic? Full gate: Stage 10. | **`references/chapters/19_risk_management.md`**, `references/precommitted_controls.md`, `references/libraries/ml4t_backtest.md` (risk rules) |
| 11. Holdout and synthesis (Ch20, Ch27) | Signal quality, portfolio translation, cost survival and temporal stability judged separately; holdout read once with everything frozen; promotion gate passed or the idea retired with its reason logged? Full gate: Stage 11. | **`references/chapters/20_strategy_synthesis.md`**, `references/chapters/27_systematic_edge.md`, `references/evidence.md`, `references/case_studies/*.md` |
| 12. Deployment (Ch25) | Same strategy code in backtest, paper and live; order state machine, reconciliation, SafeBroker limits, staged rollout and pre-flight checks in place? Full gate: Stage 12. | **`references/chapters/25_live_trading.md`**, `references/libraries/ml4t_live.md`, `references/libraries/ml4t_backtest.md` |
| 13. Monitoring and governance (Ch26, Ch19) | Technical divergence vs statistical decay separated; drift detectors calibrated; promotion gate (63-session shadow window) and rollback tested; circuit breakers with half-open recovery; research log fed back into Stage 1? Full gate: Stage 13. | **`references/chapters/26_mlops_governance.md`**, `references/chapters/19_risk_management.md`, `references/libraries/ml4t_diagnostic.md` (drift), `references/libraries/ml4t_live.md` |
| Cross-cutting: advanced AI | RL only for control problems with clear objectives (execution, market making, hedging); RAG grounded in filings with citation checks; graphs only when questions are multi-hop; agents read-only with typed state, tool contracts and replay | **`references/chapters/21_rl_execution_hedging.md`**, `references/chapters/22_rag_financial_research.md`, `references/chapters/23_knowledge_graphs.md`, `references/chapters/24_autonomous_agents.md` |
| Cross-cutting: synthetic data | Fidelity, utility (TSTR) and privacy evaluated before use; classical baselines first | **`references/chapters/05_synthetic_data.md`** |

The nine case studies (`references/case_studies/`) show the same pipeline on ETFs, crypto perpetuals,
intraday NASDAQ-100, S&P 500 equity + options, firm-characteristics panels, FX, CME futures, S&P 500
options straddles, and a broad US equity panel. Open the one closest to the user's market before
designing a new pipeline; its setup contract and stage table are the template.

## Non-negotiable guardrails (the short list)

The full catalog is indexed by `references/guardrails.md` (410 items in three part files); the pre-flight questions are `references/preflight.md`. These are the ones that invalidate
results most often; treat a violation as a bug to fix before any other work.

1. **Decision-time admissibility.** Every feature, label, scaler, imputer, regime model, universe
   membership and hyperparameter must be computable from information available at the decision
   timestamp. Fit on the training fold only; use filtered (not smoothed) states; lag slow data by its
   publication delay; use point-in-time universes, not today's constituents.
2. **Purge and embargo.** With a label horizon of *h* sessions, purge every training row whose label
   window overlaps the test window (purge = *h* sessions on the market calendar, verified with
   `va[0] - tr[-1] - 1 == h`; for variable horizons purge by label-interval overlap, not by a count).
   Pure forward walk-forward needs no embargo. Wherever training rows can follow the test block
   (CPCV, k-fold, nested inner loops) purge the overlap in both directions and add an embargo that
   covers the longer of the feature lookback and the maximum label horizon (the library convention
   `embargo_pct` of about 0.01 of the sample is the shorthand when lookbacks are short relative to the
   sample). Overlapping labels also shrink effective sample size: use HAC or block-bootstrap inference.
3. **Walk-forward, chronological, calendar-aware splits.** No shuffled K-fold on time series. Nested
   walk-forward when hyperparameters are tuned; CPCV when you need a distribution of paths.
4. **One sealed holdout, opened once.** Everything (model, allocator, overlay, costs) is frozen before
   the holdout window is read. A second look makes it validation data.
5. **Count the search.** Define the searched set before looking at results; log every trial; report
   BH-FDR / Deflated Sharpe Ratio / PBO against the number of trials actually run, not the one winner.
6. **Linear baseline first.** A regularized linear model on the same folds is the bar every other
   family must clear; a non-ML rule (equal weight, momentum, 60/40) is the bar every strategy must clear.
7. **IC is not Sharpe.** Prediction quality and trading quality decouple through turnover, breadth,
   costs and allocation; never select a deployable strategy on IC alone.
8. **Costs are a workflow constraint, not an afterthought.** Model spread plus impact for the asset
   class; report breakeven cost and capacity; sweep costs; intraday edges usually die on the cost cliff.
9. **Regimes are a risk lens, not a timing signal.** Use regime probabilities as soft conditioning
   features refit inside folds; do not train one model per hard-switched regime without evidence.
10. **Report the whole stack.** Gross and net returns, Sharpe with its standard error, drawdown and
    recovery, turnover, baseline comparison, cost sensitivity and regime slices belong together.
11. **Backtest protocol in writing.** Signal timing, execution delay (next-bar open is the default),
    fill rules, rounding, rebalancing mode, sizing, constraints, benchmark. Engine disagreements are
    protocol disagreements until proven otherwise.
12. **Reproducible run log.** Content-address every run by its full configuration; register training
    runs, prediction sets and backtests; never overwrite results silently.

## Working patterns for common requests

- **"Audit this backtest / model / notebook."** Walk `references/preflight.md` in order; for each
  violation give location, mechanism of inflation, and the fix; run `assert_causal(extract, panel)`
  (`references/libraries/ml4t_diagnostic.md` §Leakage audit) on every feature extractor before reading
  any IC (it catches smoothed states, full-sample scalers and PCA, not same-timestamp label leakage,
  which the purge assert covers); then re-estimate with purged walk-forward, costs and DSR before
  offering a verdict. Detail: `references/chapters/16_strategy_simulation.md` §"Guardrails" and
  `references/chapters/06_strategy_definition.md`.
- **"Design labels / features for X."** Start from the driver hypothesis and horizon; pick the label by
  how predictions become trades (`references/chapters/07_defining_the_learning_task.md`); build the five price-derived families plus
  structural, contextual and model-based features (`references/chapters/08_financial_features.md`, `references/chapters/09_model_based_features.md`); evaluate univariately
  (IC, quantiles, HAC inference, feasibility screen) before any model.
- **"Which model should I use?"** Follow the cascade in `references/workflow.md`: Ridge/LASSO/Elastic
  Net → LightGBM/XGBoost/CatBoost with Optuna and walk-forward HPO → TabM/TabPFN → sequence models
  only if a shuffle diagnostic shows order matters → latent factors / causal DML for structure. Check
  `references/evidence.md` for where each family actually won.
- **"Turn predictions into a portfolio."** Baselines first (equal weight, inverse vol, score weight),
  then shrinkage MVO, HRP, conformal sizing, fractional Kelly; compare on identical inputs; pick before
  holdout (`references/chapters/17_portfolio_construction.md`). Add overlays (vol targeting, exposure caps, stops calibrated on in-sample
  MAE/MFE) and kill switches (`references/chapters/19_risk_management.md`); make them precommitted with the pattern in `references/precommitted_controls.md`.
- **"Add costs."** Taxonomy → spread estimate → impact calibration (square-root, participation of ADV)
  → cost sweep → breakeven and capacity → alpha-to-go and kill criteria (`references/chapters/18_transaction_costs.md`).
- **"Take it live."** Unified engine (same strategy class in `ml4t.backtest` and `ml4t.live`), SafeBroker
  limits, order state machine, pipeline parity tests, shadow mode, staged rollout, drift monitoring
  and circuit breakers (`references/chapters/25_live_trading.md`, `references/chapters/26_mlops_governance.md`).
- **"Is this idea worth researching?" / "Should we promote this strategy?"** Write the pre-registration
  (rationale, source of edge, the disconfirming result) before any test, then run the idea promotion
  gate and frontier allocation in `references/chapters/27_systematic_edge.md`; keep the research log of rejected
  ideas. A strategy is promoted on the full reporting stack and the holdout, never on one number.
- **Writing for readers who do not have this skill.** Deliverables (audits, protocols, code
  docstrings) cite the book chapter, the companion-repo notebook path, or the library function with
  its package name (`ml4t.diagnostic.compute_ic_hac_stats`); never cite `.claude/skills/...` paths,
  "Recipe 5" or "guardrails item 16", which an outside reader cannot resolve. Apply a stage only if
  its object exists: no feature screen means no BH-FDR over features, no sweep means no PBO; say
  which stages were skipped and why.
- **"Use an LLM / agent for research."** Ground in retrieved, point-in-time filings with citations;
  evaluate retrieval and synthesis separately; typed state, tool contracts, replay, read-only
  boundary; score forecasts with proper scoring rules (`references/chapters/22_rag_financial_research.md`, `references/chapters/23_knowledge_graphs.md`, `references/chapters/24_autonomous_agents.md`).

## Applying the skill in this repository

`reproduction/` reproduces a regime-aware ML stock-ranking paper (XGBoost on a daily panel of large US
stocks, 20-day forward return, fixed 2005-2018 / 2019-2020 / 2021-2025 split, monthly equal-weight
Top-10). The ML4T stages map directly: Ch2/4 (point-in-time fundamentals, survivorship in the ticker
list), Ch7 (overlapping 20-day labels, purge/embargo, IC inference), Ch6 (walk-forward with purge and
embargo instead of a single split, trial accounting), Ch7 (BH-FDR on the feature screen), Ch9 (HMM regime
features refit inside folds, filtered not smoothed), Ch12 (GBM tuning and SHAP), Ch16 (backtest protocol,
DSR/PBO for the model/seed search), Ch17-19 (allocation, costs, overlays), Ch20 (holdout discipline).
Repository conventions to keep: Sharpe uses `RISK_FREE_RATE = 0.04` (see `reproduction/code/baseline.py`
and `transaction_costs.py`), turnover is reported as the sum of absolute weight changes over all names
(two-sided; halve it for one-way), the `random_top10` baseline is numerically the equal-weight universe
return, and the 2021-2025 window has already been read by several scripts, so a sealed holdout must be
declared outside it or labelled as exposed. When asked to extend or validate that work, run the audit
pattern above first.

## Reference index

Cross-cutting (open these first for any multi-stage task):
- `references/workflow.md` — the end-to-end process: 14 stages with purpose, outputs, gate questions and which file to open; run-log discipline; how the nine case studies instantiate it; common failure modes.
- `references/preflight.md` — the 35 pre-flight questions, ordered by stage; answer every one before reporting a number.
- `references/guardrails.md` — index of the 410-item guardrail catalog (tag legend, section map by workflow stage) pointing to `references/guardrails_data_features.md` (§1-5), `references/guardrails_evaluation_backtest.md` (§6-9) and `references/guardrails_portfolio_live.md` (§10-16); each item gives what, why, and how to detect or prevent.
- `references/decision_rules.md` — numeric defaults, thresholds and sequencing rules in tables, with sources.
- `references/evidence.md` — what the nine case studies found; what generalizes and what did not.
- `references/precommitted_controls.md` — the lag rule for overlay inputs, re-entry bounds and longest-flat-run test for kill rules, cadence parity, the precommitment pattern (freeze date, declared grid, in-sample flag, sweep rank) and the VaR/ES backtest ladder; short, with runnable snippets.
- `references/glossary.md` — vocabulary and abbreviations.
- `references/companion_repo.md` — repo layout, install and run, `utils/` and `case_studies/utils` APIs, registry identity scheme, config layers, data loaders, scripts, Docker; "where to find code for X".

Chapters (`references/chapters/`), each with when-to-use triggers, recipes, guardrails, defaults, APIs, evidence:
- `references/chapters/01_process_is_edge.md` — the workflow thesis and evidence boundary; GMM regime detection on factor and macro data: cluster selection, naming, validation, perturbation.
- `references/chapters/02_financial_data_universe.md` — dataset conventions, corporate-action and futures-roll adjustment, session bars, point-in-time joins, data-quality framework, survivorship, provider stitching, options-chain analytics (IV convergence codes, Black-Scholes Greeks validation, straddle selection), storage and engine choice.
- `references/chapters/03_market_microstructure.md` — feed selection (L1/L2/L3/TAQ), ITCH/DataBento/IEX order-book reconstruction, order-flow features and markouts, Lee–Ready, bar sampling and calibration, jump detection, intraday data-quality invariants.
- `references/chapters/04_fundamental_alternative_data.md` — point-in-time fundamentals (XBRL as-of, Form 4/13F), entity resolution, macro/COT/on-chain alignment, alt-data four-question evaluation, prediction markets (Kalshi, Polymarket), filing-text extraction.
- `references/chapters/05_synthetic_data.md` — classical simulators and bootstraps, GAN/diffusion/LLM/DP generators, and the fidelity–utility–privacy validation protocol.
- `references/chapters/06_strategy_definition.md` — strategy map and source of edge, versioned trading setup, metric roles, walk-forward with purge/embargo, nested CV and CPCV, baseline checkpoint, trial accounting; all nine setup contracts.
- `references/chapters/07_defining_the_learning_task.md` — split-aware preprocessing, execution-consistent labels (fixed-horizon, percentile, triple-barrier, trend-scanning, MFE/MAE calibration), meta-labeling and bet sizing, uniqueness weights and the sequential bootstrap, IC triage with HAC inference, multiple-testing corrections, causal sanity checks.
- `references/chapters/08_financial_features.md` — feature-spec grammar; price, microstructure, cross-instrument, options, fundamental, macro and calendar recipes; HAC-IC + BH-FDR selection; robustness sweeps; event studies; breadth vs IC.
- `references/chapters/09_model_based_features.md` — stationarity diagnostics, breaks, fractional differencing, pairs and cointegration with filtered hedge ratios, Kalman, spectral, signatures, ARIMA/GARCH/HAR, uncertainty, HMM and Wasserstein regimes, panel features, point-in-time refit rules.
- `references/chapters/10_text_feature_engineering.md` — TF-IDF, Word2Vec, asset embeddings, FinBERT, NER, and the point-in-time workflow from news and filings to screened factors.
- `references/chapters/11_ml_pipeline.md` — leakage-safe linear baselines: OLS diagnostics, Ridge/LASSO/Elastic Net, nested CV, logistic calibration, linear SHAP, conformal intervals, IC vs net Sharpe.
- `references/chapters/12_gradient_boosting.md` — GBM selection and objectives (MSE/MAE/Huber, `lambdarank` for top-k, monotone constraints), Optuna walk-forward HPO, TreeSHAP and its limits, conformal intervals, TabPFN/TabM, cross-case GBM evidence.
- `references/chapters/13_dl_time_series.md` — RNN, N-BEATS, linear vs transformer debate, PatchTST, iTransformer, TCN, TSMixer, Mamba, image CNNs, foundation models, uncertainty calibration, when depth helps.
- `references/chapters/14_latent_factors.md` — PCA and eigenportfolios, yield-curve factors, IPCA, RP-PCA, conditional and supervised autoencoders, adversarial SDF, the three-stage forecast adapter.
- `references/chapters/15_causal_estimation.md` — DoWhy backdoor validation, panel-safe DML with block permutation, BSTS event studies, PCMCI/NOTEARS/VAR-LiNGAM discovery, post-double-selection factor zoo.
- `references/chapters/16_strategy_simulation.md` — backtesting as falsification: protocol spec, vectorized vs event-driven engines and parity, baseline, reporting stack, regime and cost diagnostics, Sharpe inference, DSR/PBO/RAS.
- `references/chapters/17_portfolio_construction.md` — allocator metrics, baseline allocators, MVO instability and shrinkage, Kelly, HRP, conformal sizing, fair allocator comparison, deep allocators.
- `references/chapters/18_transaction_costs.md` — cost taxonomy and units, OHLCV spread estimation, square-root impact and capacity, TWAP/VWAP/Almgren–Chriss/RL execution, ml4t cost models, breakeven turnover, cost cliff, TCA.
- `references/chapters/19_risk_management.md` — VaR/CVaR with Kupiec backtests, drawdowns, exits and sizing, factor and SHAP attribution, stress tests, drift monitoring, deep hedging, `ml4t.backtest.risk` rules and kill switches.
- `references/chapters/20_strategy_synthesis.md` — selection rule, rank-1 cluster, paired block bootstrap vs equal weight, cumulative gate funnel, exclusion taxonomy, cost breakeven, overlay quadrant, holdout decay.
- `references/chapters/21_rl_execution_hedging.md` — MDP design, GARCH-calibrated simulators, DQN/PPO/A2C/SAC, paired benchmarking vs TWAP and Almgren–Chriss, market making, deep hedging, IRL, sim-to-real guardrails.
- `references/chapters/22_rag_financial_research.md` — point-in-time filings ingestion, embeddings, hybrid retrieval and re-ranking, cited generation, four-failure-mode evaluation, security.
- `references/chapters/23_knowledge_graphs.md` — LLM extraction with identity/schema/provenance contracts, deterministic Graph RAG, graph-derived leakage-safe features, three-timestamp temporal integrity, network portfolios.
- `references/chapters/24_autonomous_agents.md` — read-only forecasting agents: ReAct, tool contracts, typed state and gates, aggregation, debate, supervisor, proper scoring, security, the research operator.
- `references/chapters/25_live_trading.md` — unified strategy/engine parity, IB/Alpaca/OKX/QuantConnect integration, order state machine, SafeBroker controls, staged rollout.
- `references/chapters/26_mlops_governance.md` — drift monitoring (PSI/K-S, ADWIN/DDM), shadow-mode promotion gate, circuit breakers, feature stores, registry as experiment tracker.
- `references/chapters/27_systematic_edge.md` — process-as-edge thesis, hypothesis pre-registration, idea promotion gate, frontier allocation, research-log practice, cognitive-bias defenses.

Case studies (`references/case_studies/`), each with setup contract, stage table, design decisions, guardrails, results and how to adapt:
- `references/case_studies/etfs.md` — long-only monthly ETF rotation; the broadest model-family comparison.
- `references/case_studies/crypto_perps_funding.md` — crypto perpetuals on the 8-hour funding clock; availability-clock shift.
- `references/case_studies/nasdaq100_microstructure.md` — intraday NASDAQ-100 bars; the cost floor and cost-feasible screens.
- `references/case_studies/sp500_equity_option_analytics.md` — equity ranking on option-surface features.
- `references/case_studies/us_firm_characteristics.md` — monthly firm-characteristics panel; latent factors.
- `references/case_studies/fx_pairs.md` — small dependent FX cross-section; session calendars.
- `references/case_studies/cme_futures.md` — futures carry, two price series, scheduled refits.
- `references/case_studies/sp500_options.md` — delta-hedged straddles, premium-denominated costs, an honest null result.
- `references/case_studies/us_equities_panel.md` — broad daily US long-short panel; eligibility screens.

Libraries (`references/libraries/`), each with install, API map, data contracts, built-in guardrails, usage patterns and defaults. All six share contracts from `ml4t-specs` (`FeedSpec`, `ArtifactSpec`, Lifecycle V1), a dependency rather than a documented library here:
- `references/libraries/ml4t_data.md` — 24 advertised providers plus a mock, canonical UTC Polars OHLCV, point-in-time guardrails, Hive storage, continuous futures.
- `references/libraries/ml4t_engineer.md` — feature registry, alternative bars, labels, train-only scalers, dataset builder.
- `references/libraries/ml4t_diagnostic.md` — look-ahead audits (`assert_causal`), purged walk-forward and CPCV splitters, IC with HAC, DSR/PBO, drift, tearsheets.
- `references/libraries/ml4t_models.md` — latent factors, SDF, supervised autoencoder, portfolio learners.
- `references/libraries/ml4t_backtest.md` — Engine, DataFeed, Strategy, BacktestConfig, execution and cost models, risk rules.
- `references/libraries/ml4t_live.md` — LiveEngine, SafeBroker, IB/Alpaca adapters, shadow-to-live promotion.

## Provenance and limits

Built on 2026-10-02 from the companion repository at that date (3rd-edition chapter READMEs and
Jupytext notebooks) and the PyPI releases ml4t-data 0.2.0, ml4t-engineer 0.1.6, ml4t-models 0.1.4,
ml4t-diagnostic 0.1.8, ml4t-backtest 0.1.12, ml4t-live 0.1.2. The book's prose was not available;
where a reference says "(inference)" the point was reconstructed from code and notebook text. Numbers
quoted from notebooks are the repository's current values and can move when notebooks are re-run.
References were compiled from per-chapter reading notes ("the notes" / "the digest"); a paragraph headed
"Reader uncertainty" lists points the note-taker could not confirm from code and that should be verified
before relying on them. Validation prompts and expectations live in `evals/evals.json`; the benchmark
results are summarised in `evals/README.md`.
