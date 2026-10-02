---
name: ml4t
description: "Machine Learning for Trading (Stefan Jansen, 3rd ed., 2026) distilled into an operating procedure for quant research: the end-to-end ML4T workflow, its leakage / lookahead / multiple-testing guardrails, decision rules, the nine case studies' evidence, and the six ml4t Python libraries. Use this skill whenever a task touches systematic or algorithmic trading research in any form: market, fundamental or alternative data pipelines; point-in-time correctness; bars and microstructure; labels (forward returns, triple barrier, trend scanning); feature engineering (momentum, volatility, regimes, HMM/GARCH/Kalman, text/NLP); walk-forward or purged cross-validation; IC and signal evaluation; multiple testing (BH-FDR, Deflated Sharpe, PBO); linear, gradient-boosting, deep time-series, latent-factor or causal models for returns; backtesting; portfolio construction (MVO, HRP, Kelly, risk parity); transaction costs, impact and capacity; risk management, VaR/CVaR, stops, kill switches; live trading, MLOps, drift; RL for execution/hedging; RAG, knowledge graphs or LLM agents for financial research. Trigger even when the user does not say 'ML4T' or 'the book', and even for narrow asks such as reviewing a backtest, adding costs to a strategy, designing a CV split, or auditing a stock-ranking model for leakage."
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

1. Place the task on the workflow map below and read the stage's gate questions.
2. Open the matching reference file(s) under `references/` before writing code or conclusions.
   Each chapter file is self-contained (recipes, guardrails, defaults, APIs, evidence); cross-cutting
   files consolidate guardrails, numeric rules, evidence and vocabulary across chapters.
3. Run the pre-flight checklist in `references/guardrails.md` before trusting or reporting any
   research number. Fix protocol problems before tuning anything.
4. When a method decision comes up (which label, which split, which model family first, which
   allocator, which cost model), look it up in `references/decision_rules.md` and cite the rule.
5. When you need to say "what usually works", use `references/evidence.md` as priors, with the
   case-study context attached. Do not generalize a single case study.
6. When implementing, prefer the `ml4t-*` library primitives (see `references/libraries/`) or the
   companion repo's `utils/` and `case_studies/utils/` patterns over ad-hoc code; clone the repo when
   you need the exact notebook (`git clone --depth 1 https://github.com/stefan-jansen/machine-learning-for-trading.git`).

## The workflow map (what to open at each stage)

Stage numbers follow `references/workflow.md`, which has the full purpose / inputs / outputs / gate
text for each stage. Library files are listed where their APIs are the default implementation.

| Stage | Gate questions you must be able to answer | References |
|---|---|---|
| 0. Data infrastructure (Ch2-4) | Point-in-time correct? Survivorship-free universe? Corporate actions and futures rolls adjusted the right way? Identifier mapping time-valid? Right feed (L1/L2/L3/TAQ), session boundaries, bar type (time / tick / volume / dollar / imbalance) for the hypothesis? Option chains: IV convergence codes checked, Greeks validated, same-contract vs chained returns? Slow data (fundamentals, macro, alt data, prediction markets) aligned by publication time? | `chapters/02_financial_data_universe.md`, `chapters/03_market_microstructure.md`, `chapters/04_fundamental_alternative_data.md`, `libraries/ml4t_data.md`, `libraries/ml4t_engineer.md` (bars), `companion_repo.md` |
| 1. Strategy framing and trading setup (Ch6, Ch27) | Strategy family and source of edge (SLOW / WRONG / RISK)? Feasibility constraints and failure modes? Pre-registered falsifiable hypothesis with the disconfirming outcome written down? Versioned trading setup (universe rule, decision schedule, admissible information, score-to-position map, cost class, benchmark)? | `references/workflow.md`, `chapters/06_strategy_definition.md`, `chapters/27_systematic_edge.md`, `chapters/01_process_is_edge.md` |
| 2. Labels (Ch7) | Execution-consistent horizon and anchor (close-to-close vs next-open)? Overlap handled (purge, uniqueness weights, HAC)? Implied trading intensity and breakeven cost known? Meta-labeling or bet sizing needed? | `chapters/07_defining_the_learning_task.md`, `libraries/ml4t_engineer.md` |
| 3. Features (Ch8-10) | Each feature has a driver hypothesis, horizon alignment and role (signal vs state)? Fitted features (scalers, HMM, Kalman, PCA, embeddings) refit inside folds with filtered states? Slow data lagged by publication? Search degrees of freedom counted? | `chapters/08_financial_features.md`, `chapters/09_model_based_features.md`, `chapters/10_text_feature_engineering.md`, `libraries/ml4t_engineer.md` |
| 4. Univariate evaluation and triage (Ch7, Ch8) | Per-feature ledger with cross-sectional IC, ICIR, HAC t-stat, quantile spread, sign consistency across folds, feasibility (turnover, breakeven cost, capacity)? BH-FDR applied to the whole searched set? PROCEED / REVISE / STOP decision recorded, read as a feature screen, not a strategy forecast? | `chapters/07_defining_the_learning_task.md`, `chapters/08_financial_features.md`, `libraries/ml4t_diagnostic.md` |
| 5. Evaluation protocol (Ch6) | Walk-forward (expanding or rolling, calendar-aware) with label purge = horizon in sessions and feature embargo? Nested walk-forward when tuning; CPCV when a path distribution is needed? Sealed holdout dates declared before any model run? Trials logged and counted? | `chapters/06_strategy_definition.md`, `libraries/ml4t_diagnostic.md`, `guardrails.md`, `decision_rules.md` |
| 5b. Baseline checkpoint (Ch6) | One narrow configuration run end to end through the protocol and registered? Timing (fills one execution delay after the decision), coverage (every declared asset-period predicted) and trading intensity (turnover consistent with horizon and cost class) checked before widening the search? | `chapters/06_strategy_definition.md`, `companion_repo.md` |
| 6. Model cascade (Ch11-15) | Linear baseline first; does the next family beat it on the same folds and days? IC with HAC inference, fold stability, checkpoint sensitivity? Uncertainty (conformal) attached? Sequence models only after a shuffle diagnostic? Is a key association a mechanism or a confounded proxy (treatment / outcome / estimand / adjustment set stated, refutations survived)? | `chapters/11_ml_pipeline.md`, `chapters/12_gradient_boosting.md`, `chapters/13_dl_time_series.md`, `chapters/14_latent_factors.md`, `chapters/15_causal_estimation.md`, `libraries/ml4t_models.md`, `libraries/ml4t_diagnostic.md` |
| 7. Backtest (Ch16) | Protocol written (timing, fills, sizing, costs, constraints, benchmark)? Vectorized and event-driven engines agree? Gross and net, turnover, drawdown, regime slices, cost sensitivity reported? Sharpe deflated for the number of trials (DSR / PBO / RAS)? | `chapters/16_strategy_simulation.md`, `libraries/ml4t_backtest.md`, `libraries/ml4t_diagnostic.md` |
| 8. Portfolio construction (Ch17) | Does the allocator beat equal weight / inverse vol on identical inputs? Concentration, turnover and leverage stable? Allocator chosen before holdout? | `chapters/17_portfolio_construction.md`, `libraries/ml4t_models.md` (portfolio learners) |
| 9. Costs and capacity (Ch18) | Spread + impact modeled for this asset class and venue? Breakeven cost, minimum required edge, capacity at which impact eats half the alpha? Execution schedule (TWAP / VWAP / Almgren-Chriss) chosen? | `chapters/18_transaction_costs.md`, `libraries/ml4t_backtest.md` (cost models) |
| 10. Risk (Ch19) | VaR / CVaR with Kupiec backtests and path risk measured? Exposures decomposed? Stress tests run? Overlays leakage-free and precommitted? Kill criteria written as rules? | `chapters/19_risk_management.md`, `libraries/ml4t_backtest.md` (risk rules) |
| 11. Holdout and synthesis (Ch20, Ch27) | Signal quality, portfolio translation, cost survival and temporal stability judged separately? Holdout evaluated once with everything frozen? Promotion gate passed or the idea retired with the reason logged? Next-iteration priorities named? | `chapters/20_strategy_synthesis.md`, `chapters/27_systematic_edge.md`, `evidence.md`, `case_studies/*.md` |
| 12. Deployment (Ch25) | Same strategy code in backtest, paper and live? Order state machine, reconciliation, SafeBroker limits, staged rollout, pre-flight checks? | `chapters/25_live_trading.md`, `libraries/ml4t_live.md`, `libraries/ml4t_backtest.md` |
| 13. Monitoring and governance (Ch26, Ch19) | Technical divergence vs statistical decay separated? Drift detectors calibrated (PSI / K-S, ADWIN / DDM)? Promotion gates and rollback tested? Circuit breakers with half-open recovery? Research log fed back into Stage 1? | `chapters/26_mlops_governance.md`, `chapters/19_risk_management.md`, `libraries/ml4t_diagnostic.md` (drift), `libraries/ml4t_live.md` |
| Cross-cutting: advanced AI | RL only for control problems with clear objectives (execution, market making, hedging); RAG grounded in filings with citation checks; graphs only when questions are multi-hop; agents read-only with typed state, tool contracts and replay | `chapters/21_rl_execution_hedging.md`, `chapters/22_rag_financial_research.md`, `chapters/23_knowledge_graphs.md`, `chapters/24_autonomous_agents.md` |
| Cross-cutting: synthetic data | Fidelity, utility (TSTR) and privacy evaluated before use; classical baselines first | `chapters/05_synthetic_data.md` |

The nine case studies (`references/case_studies/`) show the same pipeline on ETFs, crypto perpetuals,
intraday NASDAQ-100, S&P 500 equity + options, firm-characteristics panels, FX, CME futures, S&P 500
options straddles, and a broad US equity panel. Open the one closest to the user's market before
designing a new pipeline; its setup contract and stage table are the template.

## Non-negotiable guardrails (the short list)

The full, stage-grouped checklist is `references/guardrails.md`. These are the ones that invalidate
results most often; treat a violation as a bug to fix before any other work.

1. **Decision-time admissibility.** Every feature, label, scaler, imputer, regime model, universe
   membership and hyperparameter must be computable from information available at the decision
   timestamp. Fit on the training fold only; use filtered (not smoothed) states; lag slow data by its
   publication delay; use point-in-time universes, not today's constituents.
2. **Purge and embargo.** With a label horizon of *h* bars, remove training rows whose label window
   overlaps the test window (purge ≥ *h*) and add a feature-side embargo after the test block.
   Overlapping labels also shrink effective sample size: use HAC or block-bootstrap inference.
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

- **"Audit this backtest / model / notebook."** Walk the guardrails pre-flight list in order; for each
  violation give location, mechanism of inflation, and the fix; then re-estimate with purged
  walk-forward, costs and DSR before offering a verdict. See `chapters/16_strategy_simulation.md`
  §"Guardrails" and `chapters/06_strategy_definition.md`.
- **"Design labels / features for X."** Start from the driver hypothesis and horizon; pick the label by
  how predictions become trades (`chapters/07_...`); build the five price-derived families plus
  structural, contextual and model-based features (`chapters/08_...`, `09_...`); evaluate univariately
  (IC, quantiles, HAC inference, feasibility screen) before any model.
- **"Which model should I use?"** Follow the cascade in `references/workflow.md`: Ridge/LASSO/Elastic
  Net → LightGBM/XGBoost/CatBoost with Optuna and walk-forward HPO → TabM/TabPFN → sequence models
  only if a shuffle diagnostic shows order matters → latent factors / causal DML for structure. Check
  `references/evidence.md` for where each family actually won.
- **"Turn predictions into a portfolio."** Baselines first (equal weight, inverse vol, score weight),
  then shrinkage MVO, HRP, conformal sizing, fractional Kelly; compare on identical inputs; pick before
  holdout (`chapters/17_...`). Add overlays (vol targeting, exposure caps, stops calibrated on in-sample
  MAE/MFE) and kill switches (`chapters/19_...`).
- **"Add costs."** Taxonomy → spread estimate → impact calibration (square-root, participation of ADV)
  → cost sweep → breakeven and capacity → alpha-to-go and kill criteria (`chapters/18_...`).
- **"Take it live."** Unified engine (same strategy class in `ml4t.backtest` and `ml4t.live`), SafeBroker
  limits, order state machine, pipeline parity tests, shadow mode, staged rollout, drift monitoring
  and circuit breakers (`chapters/25_...`, `26_...`).
- **"Is this idea worth researching?" / "Should we promote this strategy?"** Write the pre-registration
  (rationale, source of edge, the disconfirming result) before any test, then run the idea promotion
  gate and frontier allocation in `chapters/27_systematic_edge.md`; keep the research log of rejected
  ideas. A strategy is promoted on the full reporting stack and the holdout, never on one number.
- **"Use an LLM / agent for research."** Ground in retrieved, point-in-time filings with citations;
  evaluate retrieval and synthesis separately; typed state, tool contracts, replay, read-only
  boundary; score forecasts with proper scoring rules (`chapters/22_...`, `23_...`, `24_...`).

## Applying the skill in this repository

`reproduction/` reproduces a regime-aware ML stock-ranking paper (XGBoost on a daily panel of large US
stocks, 20-day forward return, fixed 2005-2018 / 2019-2020 / 2021-2025 split, monthly equal-weight
Top-10). The ML4T stages map directly: Ch2/4 (point-in-time fundamentals, survivorship in the ticker
list), Ch7 (overlapping 20-day labels, purge/embargo, IC inference), Ch6 (walk-forward with purge and
embargo instead of a single split, trial accounting), Ch7 (BH-FDR on the feature screen), Ch9 (HMM regime
features refit inside folds, filtered not smoothed), Ch12 (GBM tuning and SHAP), Ch16 (backtest protocol,
DSR/PBO for the model/seed search), Ch17-19 (allocation, costs, overlays), Ch20 (holdout discipline).
When asked to extend or validate that work, run the audit pattern above first.

## Reference index

Cross-cutting (open these first for any multi-stage task):
- `references/workflow.md` — the end-to-end process: 14 stages with purpose, outputs, gate questions and which file to open; run-log discipline; how the nine case studies instantiate it; common failure modes.
- `references/guardrails.md` — pre-flight checklist and the stage-grouped catalog of every guardrail with detection and prevention.
- `references/decision_rules.md` — numeric defaults, thresholds and sequencing rules in tables, with sources.
- `references/evidence.md` — what the nine case studies found; what generalizes and what did not.
- `references/glossary.md` — vocabulary and abbreviations.
- `references/companion_repo.md` — repo layout, install and run, `utils/` and `case_studies/utils` APIs, registry identity scheme, config layers, data loaders, scripts, Docker; "where to find code for X".

Chapters (`references/chapters/`), each with when-to-use triggers, recipes, guardrails, defaults, APIs, evidence:
- `01_process_is_edge.md` — the workflow thesis and evidence boundary; GMM regime detection on factor and macro data: cluster selection, naming, validation, perturbation.
- `02_financial_data_universe.md` — dataset conventions, corporate-action and futures-roll adjustment, session bars, point-in-time joins, data-quality framework, survivorship, provider stitching, options-chain analytics (IV convergence codes, Black-Scholes Greeks validation, straddle selection), storage and engine choice.
- `03_market_microstructure.md` — feed selection (L1/L2/L3/TAQ), ITCH/DataBento/IEX order-book reconstruction, order-flow features and markouts, Lee–Ready, bar sampling and calibration, jump detection, intraday data-quality invariants.
- `04_fundamental_alternative_data.md` — point-in-time fundamentals (XBRL as-of, Form 4/13F), entity resolution, macro/COT/on-chain alignment, alt-data four-question evaluation, prediction markets (Kalshi, Polymarket), filing-text extraction.
- `05_synthetic_data.md` — classical simulators and bootstraps, GAN/diffusion/LLM/DP generators, and the fidelity–utility–privacy validation protocol.
- `06_strategy_definition.md` — strategy map and source of edge, versioned trading setup, metric roles, walk-forward with purge/embargo, nested CV and CPCV, baseline checkpoint, trial accounting; all nine setup contracts.
- `07_defining_the_learning_task.md` — split-aware preprocessing, execution-consistent labels (fixed-horizon, percentile, triple-barrier, trend-scanning, MFE/MAE calibration), meta-labeling and bet sizing, uniqueness weights and the sequential bootstrap, IC triage with HAC inference, multiple-testing corrections, causal sanity checks.
- `08_financial_features.md` — feature-spec grammar; price, microstructure, cross-instrument, options, fundamental, macro and calendar recipes; HAC-IC + BH-FDR selection; robustness sweeps; event studies; breadth vs IC.
- `09_model_based_features.md` — stationarity diagnostics, breaks, fractional differencing, pairs and cointegration with filtered hedge ratios, Kalman, spectral, signatures, ARIMA/GARCH/HAR, uncertainty, HMM and Wasserstein regimes, panel features, point-in-time refit rules.
- `10_text_feature_engineering.md` — TF-IDF, Word2Vec, asset embeddings, FinBERT, NER, and the point-in-time workflow from news and filings to screened factors.
- `11_ml_pipeline.md` — leakage-safe linear baselines: OLS diagnostics, Ridge/LASSO/Elastic Net, nested CV, logistic calibration, linear SHAP, conformal intervals, IC vs net Sharpe.
- `12_gradient_boosting.md` — GBM selection and objectives (MSE/MAE/Huber, `lambdarank` for top-k, monotone constraints), Optuna walk-forward HPO, TreeSHAP and its limits, conformal intervals, TabPFN/TabM, cross-case GBM evidence.
- `13_dl_time_series.md` — RNN, N-BEATS, linear vs transformer debate, PatchTST, iTransformer, TCN, TSMixer, Mamba, image CNNs, foundation models, uncertainty calibration, when depth helps.
- `14_latent_factors.md` — PCA and eigenportfolios, yield-curve factors, IPCA, RP-PCA, conditional and supervised autoencoders, adversarial SDF, the three-stage forecast adapter.
- `15_causal_estimation.md` — DoWhy backdoor validation, panel-safe DML with block permutation, BSTS event studies, PCMCI/NOTEARS/VAR-LiNGAM discovery, post-double-selection factor zoo.
- `16_strategy_simulation.md` — backtesting as falsification: protocol spec, vectorized vs event-driven engines and parity, baseline, reporting stack, regime and cost diagnostics, Sharpe inference, DSR/PBO/RAS.
- `17_portfolio_construction.md` — allocator metrics, baseline allocators, MVO instability and shrinkage, Kelly, HRP, conformal sizing, fair allocator comparison, deep allocators.
- `18_transaction_costs.md` — cost taxonomy and units, OHLCV spread estimation, square-root impact and capacity, TWAP/VWAP/Almgren–Chriss/RL execution, ml4t cost models, breakeven turnover, cost cliff, TCA.
- `19_risk_management.md` — VaR/CVaR with Kupiec backtests, drawdowns, exits and sizing, factor and SHAP attribution, stress tests, drift monitoring, deep hedging, `ml4t.backtest.risk` rules and kill switches.
- `20_strategy_synthesis.md` — selection rule, rank-1 cluster, paired block bootstrap vs equal weight, cumulative gate funnel, exclusion taxonomy, cost breakeven, overlay quadrant, holdout decay.
- `21_rl_execution_hedging.md` — MDP design, GARCH-calibrated simulators, DQN/PPO/A2C/SAC, paired benchmarking vs TWAP and Almgren–Chriss, market making, deep hedging, IRL, sim-to-real guardrails.
- `22_rag_financial_research.md` — point-in-time filings ingestion, embeddings, hybrid retrieval and re-ranking, cited generation, four-failure-mode evaluation, security.
- `23_knowledge_graphs.md` — LLM extraction with identity/schema/provenance contracts, deterministic Graph RAG, graph-derived leakage-safe features, three-timestamp temporal integrity, network portfolios.
- `24_autonomous_agents.md` — read-only forecasting agents: ReAct, tool contracts, typed state and gates, aggregation, debate, supervisor, proper scoring, security, the research operator.
- `25_live_trading.md` — unified strategy/engine parity, IB/Alpaca/OKX/QuantConnect integration, order state machine, SafeBroker controls, staged rollout.
- `26_mlops_governance.md` — drift monitoring (PSI/K-S, ADWIN/DDM), shadow-mode promotion gate, circuit breakers, feature stores, registry as experiment tracker.
- `27_systematic_edge.md` — process-as-edge thesis, hypothesis pre-registration, idea promotion gate, frontier allocation, research-log practice, cognitive-bias defenses.

Case studies (`references/case_studies/`): `etfs.md` (long-only monthly rotation, broadest family comparison), `crypto_perps_funding.md` (8-hour funding clock, availability-clock shift), `nasdaq100_microstructure.md` (intraday cost floor and cost-feasible screens), `sp500_equity_option_analytics.md` (equity ranking on option-surface features), `us_firm_characteristics.md` (monthly characteristics panel, latent factors), `fx_pairs.md` (small dependent cross-section, session calendars), `cme_futures.md` (carry, two price series, scheduled refits), `sp500_options.md` (premium-denominated costs, honest null result), `us_equities_panel.md` (broad daily long-short, eligibility screens).

Libraries (`references/libraries/`): `ml4t_data.md` (23 providers, canonical UTC Polars OHLCV, PIT guardrails, Hive storage, continuous futures), `ml4t_engineer.md` (feature registry, alternative bars, labels, train-only scalers, dataset builder), `ml4t_diagnostic.md` (look-ahead audits, purged CV, IC/HAC, DSR/PBO, drift, tearsheets), `ml4t_models.md` (latent factors, SDF, supervised autoencoder, portfolio learners), `ml4t_backtest.md` (Engine, DataFeed, Strategy, BacktestConfig, execution and cost models, risk rules), `ml4t_live.md` (LiveEngine, SafeBroker, IB/Alpaca adapters, shadow-to-live promotion). All six share artifact and data contracts from `ml4t-specs` (`FeedSpec`, `ArtifactSpec`, Lifecycle V1), which is a dependency rather than a documented library here.

- `evals/evals.json` — test prompts used to validate this skill.

## Provenance and limits

Built on 2026-10-02 from the companion repository at that date (3rd-edition chapter READMEs and
Jupytext notebooks) and the PyPI releases ml4t-data 0.2.0, ml4t-engineer 0.1.6, ml4t-models 0.1.4,
ml4t-diagnostic 0.1.8, ml4t-backtest 0.1.12, ml4t-live 0.1.2. The book's prose was not available;
where a reference says "(inference)" the point was reconstructed from code and notebook text. Numbers
quoted from notebooks are the repository's current values and can move when notebooks are re-run.
