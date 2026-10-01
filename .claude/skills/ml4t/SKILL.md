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

| Stage | Gate questions you must be able to answer | References |
|---|---|---|
| 0. Frame the idea | Strategy family, source of edge (SLOW / WRONG / RISK taxonomy), feasibility constraints, failure modes? Written trading setup (universe rule, decision schedule, admissible info, score-to-position map, costs that count)? | `references/workflow.md`, `chapters/06_strategy_definition.md`, `chapters/01_process_is_edge.md` |
| 1. Data infrastructure | Point-in-time correct? Survivorship-free universe? Corporate actions adjusted the right way? Identifier mapping time-valid? Storage fits access pattern? | `chapters/02_financial_data_universe.md`, `chapters/04_fundamental_alternative_data.md`, `libraries/ml4t_data.md`, `companion_repo.md` |
| 2. Microstructure and bars | Right data product (L1/L2/L3/TAQ)? Session boundaries and timestamps right? Bar type matches the hypothesis (time / tick / volume / dollar / imbalance)? Trade classification needed? | `chapters/03_market_microstructure.md`, `libraries/ml4t_engineer.md` |
| 3. Labels | Execution-consistent horizon and anchor (close-to-close vs next-open)? Overlap handled? Implied trading intensity and breakeven cost known? | `chapters/07_defining_the_learning_task.md`, `libraries/ml4t_engineer.md` |
| 4. Features | Each feature has a driver hypothesis, horizon alignment and role (signal vs state)? Fitted features refit inside folds? Slow data lagged by publication? Search degrees of freedom counted? | `chapters/08_financial_features.md`, `chapters/09_model_based_features.md`, `chapters/10_text_feature_engineering.md` |
| 5. Evaluation protocol | Walk-forward with label purge and feature embargo? Sealed holdout touched once? Trials logged and counted? FDR / DSR correction applied to the searched set? | `chapters/06_strategy_definition.md`, `chapters/07_defining_the_learning_task.md`, `guardrails.md`, `decision_rules.md` |
| 6. Model cascade | Linear baseline first; does the next family beat it on the same folds and days? IC with HAC inference, fold stability, checkpoint sensitivity? Uncertainty (conformal) attached? | `chapters/11_ml_pipeline.md`, `chapters/12_gradient_boosting.md`, `chapters/13_dl_time_series.md`, `chapters/14_latent_factors.md`, `libraries/ml4t_models.md`, `libraries/ml4t_diagnostic.md` |
| 7. Causal credibility | Is the association a mechanism or a confounded proxy? Treatment / outcome / estimand / adjustment set stated? Refutations survived? | `chapters/15_causal_estimation.md`, `chapters/07_defining_the_learning_task.md` |
| 8. Backtest | Protocol written (timing, fills, sizing, costs, constraints, benchmark)? Gross and net, turnover, drawdown, regime slices, cost sensitivity reported? Sharpe deflated for the number of trials? | `chapters/16_strategy_simulation.md`, `libraries/ml4t_backtest.md` |
| 9. Portfolio construction | Does the allocator beat equal weight / inverse vol on identical inputs? Concentration, turnover and leverage stable? Allocator chosen before holdout? | `chapters/17_portfolio_construction.md` |
| 10. Costs and capacity | Spread + impact modeled for this asset class? Breakeven cost, minimum required edge, capacity at which impact eats half the alpha? | `chapters/18_transaction_costs.md` |
| 11. Risk | VaR/CVaR and path risk measured? Exposures decomposed? Stress tests run? Overlays leakage-free and precommitted? Kill criteria written? | `chapters/19_risk_management.md` |
| 12. Synthesis and holdout | Signal quality, portfolio translation, cost survival and temporal stability judged separately? Holdout evaluated once with everything frozen? Next-iteration priorities named? | `chapters/20_strategy_synthesis.md`, `evidence.md`, `case_studies/*.md` |
| 13. Deployment | Same strategy code in backtest, paper and live? Order state machine, reconciliation, staged rollout, pre-flight checks? | `chapters/25_live_trading.md`, `libraries/ml4t_live.md` |
| 14. Monitoring and governance | Technical divergence vs statistical decay separated? Drift detectors calibrated? Promotion gates and rollback tested? Circuit breakers with half-open recovery? | `chapters/26_mlops_governance.md` |
| Advanced AI (when relevant) | RL only for control problems with clear objectives (execution, market making, hedging); RAG grounded in filings with citation checks; graphs only when questions are multi-hop; agents read-only with typed state, tool contracts and replay | `chapters/21_rl_execution_hedging.md`, `chapters/22_rag_financial_research.md`, `chapters/23_knowledge_graphs.md`, `chapters/24_autonomous_agents.md` |
| Synthetic data | Fidelity, utility (TSTR) and privacy evaluated before use; classical baselines first | `chapters/05_synthetic_data.md` |

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
- **"Use an LLM / agent for research."** Ground in retrieved, point-in-time filings with citations;
  evaluate retrieval and synthesis separately; typed state, tool contracts, replay, read-only
  boundary; score forecasts with proper scoring rules (`chapters/22_...`, `23_...`, `24_...`).

## Applying the skill in this repository

`reproduction/` reproduces a regime-aware ML stock-ranking paper (XGBoost on a daily panel of large US
stocks, 20-day forward return, fixed 2005-2018 / 2019-2020 / 2021-2025 split, monthly equal-weight
Top-10). The ML4T stages map directly: Ch2/4 (point-in-time fundamentals, survivorship in the ticker
list), Ch7 (overlapping 20-day labels, purge/embargo, IC inference), Ch9 (HMM regime features refit
inside folds, filtered not smoothed), Ch12 (GBM tuning and SHAP), Ch16 (walk-forward instead of a single
split, DSR for the model/seed search), Ch17-19 (allocation, costs, overlays), Ch20 (holdout discipline).
When asked to extend or validate that work, run the audit pattern above first.

## Reference index

- `references/workflow.md` — the end-to-end process, gates, artifacts, run-log discipline.
- `references/guardrails.md` — pre-flight checklist and the full stage-grouped guardrail catalog.
- `references/decision_rules.md` — numeric defaults, thresholds and sequencing rules with sources.
- `references/evidence.md` — what the nine case studies found; what generalizes and what did not.
- `references/glossary.md` — vocabulary and abbreviations.
- `references/chapters/NN_slug.md` — one file per chapter (01 … 27), named after the repo directories.
- `references/case_studies/<name>.md` — one file per case study: setup contract, stage table, results, how to adapt.
- `references/libraries/ml4t_<pkg>.md` — API maps for ml4t-data, -engineer, -models, -diagnostic, -backtest, -live.
- `references/companion_repo.md` — repository layout, utils, research package, registry, configs, scripts, Docker.
- `evals/evals.json` — test prompts used to validate this skill.
