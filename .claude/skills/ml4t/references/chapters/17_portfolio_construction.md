# Chapter 17: Portfolio Construction

> Good forecasts are not yet portfolios. This chapter formalizes allocation as the step that combines expected returns (mu), a covariance estimate (Sigma), admissible-risk constraints, leverage and rebalancing into weights, and argues that this layer can amplify a weak edge or destroy a strong one through concentration, unstable sizing and excess turnover. Its position: simple baselines (equal weight, inverse volatility, risk parity) are demanding benchmarks; MVO is dominated by its worst-estimated input (mu), so "estimating less" (min variance, risk parity, HRP) and shrinkage (Ledoit-Wolf) are regularizers rather than cosmetics; Kelly is a growth argument that only works with an explicit leverage cap. Allocators must be compared on identical inputs and protocol, selected once before the holdout opens, and ranked only after the execution bridge (vectorized -> zero-cost engine -> cost-aware engine). End-to-end deep allocators (LSTM / VLSTM / DeePM trained on a differentiable Sharpe loss) consume a different input stream and are therefore judged against the heuristics, never head-to-head with MVO.

## When to use this reference

- Turning a model ranking or forward-return prediction into target weights (top-K selection + sizing rule).
- Choosing among equal weight, inverse vol, risk parity/ERC, min variance, max Sharpe, min CDaR, HRP for a given universe and sample length.
- Debugging extreme, unstable or sign-flipping optimizer weights; checking covariance condition number, N/T and redundancy before trusting an optimizer.
- Producing a portfolio performance report / tear sheet (Sharpe, Sortino, Calmar, IR, alpha/beta, drawdowns, capture, VaR/CVaR, rolling metrics) with `ml4t-diagnostic`.
- Sizing leverage with Kelly (binary, continuous, multi-asset) or fractional Kelly; computing gross exposure and `min_wealth_multiple`.
- Measuring concentration and diversification: risk contributions, effective positions (1/HHI), largest weight.
- Designing a fair allocator comparison (selection region vs holdout, Sharpe SE, paired block bootstrap, regime slices) and avoiding "try allocators until one works".
- Using PyPortfolioOpt, Riskfolio-Lib or skfolio, and reconciling their defaults/units.
- Sizing positions from per-entity conformal interval widths (Mondrian split conformal).
- Training or auditing an end-to-end DL allocator (differentiable Sharpe, in-loss turnover cost, VSN, SoftMin regime robustness) and its cost-grid sensitivity.
- Deciding whether published factor premia (value, momentum, carry, defensive, TSMOM) are real and diversify each other.
- Replaying frozen or scheduled target weights through `ml4t-backtest` to price timing and costs.

## Core ideas (the why)

- **Allocation is its own research stage** (17.2): it needs an allocator "term sheet" (inputs, estimation windows, constraints, rebalancing), leakage controls, matched estimation windows and a strict separation of prediction from sizing, otherwise the allocation layer becomes hidden overfitting. (Exact term-sheet content is known only from the README; see uncertainty note in Evidence.)
- **Simple baselines are serious competitors** (17.4; DeMiguel et al. 2009): if a sophisticated allocator cannot reliably beat equal weight / inverse vol / risk parity on identical inputs, its estimation burden is unjustified.
- **Estimating less is a strategy**: EW estimates nothing; IV one variance per asset; ERC/risk parity and min variance the covariance but no returns; HRP correlations (for the tree) plus cluster variances; max Sharpe and Kelly need mu and Sigma. Each gives up the ability to express a view in exchange for not having to hold one; the test window prices that trade.
- **The Markowitz curse**: the MVO solution is dominated by the input estimated worst. Endpoint-CAGR expected returns rest on two prices per asset while the covariance uses thousands of observations per pair, yet the optimizer is most sensitive to mu; max-Sharpe weights concentrate into a handful of names and rarely stay there. Inversion amplifies the smallest eigenvalues, which the sample estimates worst; sample covariance is singular once N > T and ill-conditioned well before.
- **A high condition number is a statement about redundancy, not risk**: it grows when assets move together (nine sector funds beside the broad-market funds that contain them). N/T is one contributor, not the decider: low N/T over highly correlated assets can be worse conditioned than higher N/T over independent ones.
- **Dropping mu buys stability, not diversification**: min variance and min CDaR concentrate harder than max Sharpe because minimizing a risk number is as much a corner solution as maximizing a ratio. Only a diversification requirement inside the objective (risk parity) keeps weights spread. HRP is not concentration-free either: successive inverse-variance splits pile weight onto low-variance assets (bonds, gold). "Risk-based is not the same thing as diversified."
- **Shrinkage is a deliberate bias-for-variance trade**: Ledoit-Wolf is wrong in a known direction but less wrong sample to sample, and inversion punishes variance far more than bias. Two responses to the curse: never invert (HRP) or invert after pulling toward a well-conditioned target (shrinkage).
- **Risk parity equalizes risk shares, not capital, and only against the covariance it was fitted on**; min CDaR optimizes the tail it was shown. Neither property is guaranteed out of sample.
- **Kelly answers a growth question, not a survival question**: it maximizes expected log terminal wealth with no term for ruin, borrowing limit or funding rate, and scales directly with an unestimable mu. Practical use is "the formula plus a leverage constraint", and the constraint does at least as much work as the formula. Fractional Kelly is a cap, not a fix. Kelly and max-Sharpe share the direction Sigma^-1 mu; MVO renormalizes to sum to one and throws the length away, Kelly keeps the length as leverage.
- **One ratio cannot describe a path**: each headline metric is defined by its denominator (Sharpe: total vol; Sortino: downside vol; Calmar: worst drawdown; Omega: mass of losses; tail ratio: extremes; alpha: nothing; IR: active vol; capture: benchmark return by direction). No threshold makes any one "good"; read them together. Drawdown, not volatility, drives abandonment; recovery time decides holdability.
- **Report the cost-aware run beside the paper one**: timing and fees are separate mechanisms; an allocation ranking computed without them ranks portfolios nobody can hold. The frontier is a picture of the sample, not a menu; only estimate-on-one-window, score-on-another says anything about the future.
- **A return source is useful to an allocator because it pays when the others do not**: two factors negatively correlated for a structural reason (value buys what fell, momentum buys what rose) are worth more than two with high individual Sharpe. Separate three questions: is the premium real (t-stat), do the factors diversify (correlation + mechanism), does it pay in crises (describable with ex-post windows, not testable).
- **"The API is not the method"**: matched objectives across libraries agree to solver tolerance; material differences come from defaults, units and penalties (turnover, L2), which are part of the objective.
- **Select once, before the holdout opens**: reading the holdout table and picking the top row turns the holdout into a second selection window.
- **Conformal sizing bets on residual-dispersion stability**; uncertainty changes position size but cannot manufacture direction. If IC is near zero, inverse-width weighting rescales noise.
- **End-to-end DL removes the prediction step, not the estimation problem**: the loss (differentiable Sharpe, optionally net of turnover), not the architecture, is the contribution. It entangles prediction error, sizing, turnover and exposure in one loss surface and cannot separate signal failure from allocation failure. A learned allocator must beat EW and IV to have earned its machinery. "The cost grid is the number to read before the table." A single seeded run is a diagnostic, not an effect size.

## Method recipes (the how)

### Allocation workflow and term sheet (17.2)

1. Declare the allocator's inputs (mu source, Sigma estimator, windows), constraints (long-only, budget, leverage cap, per-name caps), rebalancing frequency and hurdle rate `RISK_FREE_RATE` per experiment (02 uses 0.04, 03 uses 0, 04 uses 0.02; treat the hurdle as a declared parameter, not a chapter default).
2. Match estimation windows to the evaluation protocol: everything through `TRAIN_END` enters the fit; nothing after. Freeze weights, score later.
3. Keep prediction (ranking) and sizing (weights) as separate, independently auditable steps.
4. Run the pre-optimizer checklist and the execution bridge (below) before reading any ranking.

### Portfolio evaluation report (`01_portfolio_metrics`, 17.3)

| Step | What to compute | Default / note |
|---|---|---|
| Inputs | Daily strategy returns, benchmark returns | `BENCHMARK_SYMBOL="SPY"`, `RISK_FREE_RATE=0.0` annual, `PERIODS_PER_YEAR=252`; returns from `case_studies/etfs/run_log/backtest/<hash>/daily_returns.parquet` chosen by `resolve_best_backtest_runs(...)` (highest validation Sharpe, `ETF_LABEL="fwd_ret_21d"`, `BACKTEST_HASH=None` auto-picks; 12-char hash pins) |
| Headline table | Annualized return, vol, Sharpe, Sortino, Calmar, max DD, alpha, beta, IR; plus Omega, tail ratio, up/down capture | Sharpe = (mu - r_f)/sigma; raising r_f by Delta costs Delta/sigma (more for low-vol series, can reorder strategies); IR invariant to common r_f; `periodic_sortino_ratio` subtracts the rate before downside deviation (only moves ratio down) |
| Annualization | sqrt(periods) for vol/ratios; compounded return raised to the count | 252 for daily equities; 365 inflates vol-based ratios by sqrt(365/252) ~ +20% |
| Rolling metrics | `ROLLING_WINDOWS=[21, 63, 252]` | 21-session Sharpe SE ~ the ratio itself; 252 reacts months late; read all three together |
| Drawdowns | `compute_drawdown_analysis` -> episodes (depth, sessions to trough, sessions to recovery); `DrawdownResult.underwater_curve` | Hand-rolled `wealth=(1+r).cumprod(); dd=wealth/wealth.cummax()-1` asserted equal to library |
| Monthly/annual tables | Compound daily returns to month/year ends; year-by-month heatmap | Answers "how much came from how few periods" |
| Benchmark regression | beta = slope, alpha = annualized intercept; TE = vol(strategy - benchmark); IR = mean active / TE | Labels: beta < 0.8 "materially less exposed", > 1.2 "materially more" (conventional); IR * sqrt(T years) ~ t-stat |
| Capture ratios | Compound both to month ends; up capture = mean strategy return in benchmark-up months / mean benchmark return in those months; down likewise | Use ratio of means; library `up_down_capture` divides wealth factors and inverts the down reading (>1 for a strategy losing half as much); frequency-dependent |
| VaR / CVaR | `value_at_risk(returns, confidence=0.95/0.99)`, `conditional_var(returns, confidence=0.99)` | Empirical quantiles; CVaR = mean loss beyond VaR; quote both |
| Event analysis | `analyze_period(returns, benchmark, start, end)` over hand-named windows | Seen in source: "COVID Crash (2020)" 2020-02-19..2020-03-23; "COVID Recovery" 2020-03-23..2020-08-31; "2022 Bear Market" 2022-01-03..2022-10-12 (two more windows unconfirmed) |
| Stability | R^2 of a line fitted to cumulative value vs time | Path shape, not return; read beside drawdown table |
| Assembly | `add_metric(category, name, value)` into return / risk / ratio / benchmark-relative; `create_portfolio_dashboard()` -> `tear_sheet.show()`; `save_html(..., include_plotlyjs="cdn")`; `combine_figures_to_html(figures + narrative)` | Pull figures by name to set title and alt text; `tear_sheet.show()` publishes PNGs without alt text |

The registered backtest artifact already includes commission, slippage and next-bar execution: do not subtract costs again.

### Baseline allocators (17.4)

| Allocator | Weights | Estimates | Notes |
|---|---|---|---|
| Equal weight | 1/N | nothing | Benchmark every other allocator must beat; `equal_weights(n)` |
| Inverse volatility | w_i ∝ 1/sigma_i, normalized | one variance per asset | Ignores correlations; `inverse_volatility_weights(returns)`; windows used: 63 (09, 12), 21 (11, 13) |
| Equal risk contribution / risk parity | solve RC_i = 1/N, RC_i = w_i (Sigma w)_i / (w^T Sigma w) | full Sigma, no mu | `equal_risk_contribution(cov)` minimizes sum_i (w_i (Sigma w)_i - w^T Sigma w / N)^2; Riskfolio `port.rp_optimization(...)`; differs from IV by accounting for correlations |
| Volatility targeting | w_{i,t} = VOL_TARGET_ANN / (sqrt(252) * max(vol_{i,t}, 1e-4) * N) * signal_{i,t} | rolling vol (63d) | `VOL_TARGET_ANN=0.15`; the 1/N keeps the book at the target rather than sqrt(N) times it (12) |
| Score weighting | w_i = yhat_i / sum_j yhat_j over top-K with yhat > 0 | prediction scores | (07) |
| Conformal sizing | w_i = (1/Delta_i) / sum_j (1/Delta_j) | per-entity interval widths | (07; recipe below) |
| Kelly sizing | f* = Sigma^-1 mu, scaled by a multiple and capped | mu and Sigma | (04; recipe below) |

Under identical-asset estimates (equal mu, Sigma ∝ I) EW, IV and ERC coincide with min variance on the frontier; their distance below it is a property of the estimates, not a general one.

### Mean-variance optimization from scratch (`02_mean_variance_optimization`, 17.5)

1. Inputs: mu_i = CAGR from first and last price (`annualize_returns_from_prices(prices_df, symbols)`); Sigma = annualized covariance of all daily observations. Settings: `N_PORTFOLIOS=10000`, `SEED=42`, `TRAIN_END="2019-12-31"`, `RISK_FREE_RATE=0.04` (scenario hurdle), `COMMISSION_RATE=SLIPPAGE_RATE=0.0005`. Universe ~30 ETFs (equities by region/sector incl. nine US sector funds plus broad-market funds, government and corporate credit, real assets) -> 435 covariances.
2. Compute kappa(Sigma) = sigma_max/sigma_min and plot the log eigenvalue spectrum before optimizing; kappa bounds the amplification of relative input error through inversion.
3. Feasible region: `simulate_portfolios(returns, cov, n_portfolios, rf_rate=0.0)` draws Dirichlet weights with `alpha=np.full(n_assets, 0.05)` (small alpha -> edges of the bullet; large alpha -> near equal weight); plot 5000 of 10000.
4. Solve with `scipy.optimize.minimize(method="SLSQP")`, bounds [0,1], sum-to-one constraint: `optimize_max_sharpe(returns, cov, rf)`, `optimize_min_vol(returns, cov)`; helpers `portfolio_return`, `portfolio_volatility`, `portfolio_sharpe(weights, returns, cov, rf)`, `neg_sharpe`. Markowitz dual: min w^T Sigma w s.t. w^T mu = mu_PF, 1^T w = 1.
5. Validate: `validate_optimization_result(result, *, label, lower_bound=0.0, upper_bound=1.0, target_return=None, expected_returns=None, erc_cov=None, tolerance=1e-6)`.
6. Frontier: `compute_efficient_frontier(returns, cov, min_vol_weights, n_points=50)` (long-only upper frontier from the global minimum variance upward).
7. Compare five portfolios: Max Sharpe, Min Volatility, Equal Weight, Inverse Vol, ERC. Report effective positions and largest weight.
8. Train/test: estimate all inputs through `TRAIN_END`, freeze weights, rebalance daily to target over the test window; choose nothing from the test result.
9. Factor attribution of the frozen max-Sharpe test returns: `FactorAnalysis` + `load_fama_french_5factor` (Mkt-RF, SMB, HML, RMW, CMA), HAC standard errors; intercept = per-day factor-adjusted alpha with t-stat; loadings describe test-window co-movement only.
10. Execution bridge (three rows; recipe below).

Random-portfolio sanity check: if the optimizer lands only marginally above a broad Dirichlet sample, MVO's extra precision buys little. Allowing shorts makes concentration considerably worse; long-only and fully-invested constraints are what keep it "as mild as it appears".

### Robust allocation via Riskfolio-Lib (`03_robust_optimization`, 17.5)

- Universe (11 ETFs -> 55 covariances): SPY, QQQ, IWM; EFA, EEM; AGG, TLT, HYG; GLD, VNQ, DBC. `START_DATE="2015-01-01"`, `END_DATE="2023-12-31"`, `TRAIN_END="2019-12-31"`.
- Train `returns.loc[:TRAIN_END]`, test `returns.index > TRAIN_END`; only training rows enter the correlation map and every fit.
- Moments: `port = rp.Portfolio(returns=fit_returns); port.assets_stats(method_mu="hist", method_cov="ledoit")`; `rf=0` throughout.

| Method | Riskfolio call | Needs |
|---|---|---|
| Max Sharpe | `port.optimization(model="Classic", rm="MV", obj="Sharpe", rf=0, hist=True)` | mu + Sigma |
| Min Variance | `port.optimization(model="Classic", rm="MV", obj="MinRisk")` | Sigma |
| Risk Parity | `port.rp_optimization(model="Classic", rm="MV", rf=0, hist=True)` | Sigma |
| Min CDaR | `port.optimization(model="Classic", rm="CDaR", obj="MinRisk")` | realized path |
| HRP | `rp.HCPortfolio(returns=fit_returns).optimization(model="HRP", codependence="pearson", rm="MV", rf=0, linkage="ward", max_k=10, leaf_order=True, method_cov="ledoit")` | correlations + cluster variances |
| Equal Weight | 1/N, no solver | nothing |

- `validate_weights(method, weights)`: symbol coverage, finiteness, long-only bounds, budget = 1.
- Diagnostics: effective positions = 1/sum(w_i^2); `compute_risk_contribution(weights, covariance)` = w_i (Sigma w)_i / (w^T Sigma w) (shares sum to one; the "weight times derivative of variance" form carries a factor two and sums to two). Check against the same Ledoit-Wolf covariance that produced the risk-parity weights; diagonal-covariance oracle: inverse-vol weights must contribute exactly 1/N each.
- `backtest_portfolio(weights, evaluation_returns)` applies frozen symbol-keyed weights; `compute_drawdown(returns)`; rolling Sharpe 252-day window, `min_periods=126`, sqrt(252) (a stability diagnostic, not a selection criterion).
- Riskfolio reaches CVXPY through an interface that changed across versions: check compatibility up front and report a mismatch as an environment problem.

### Kelly criterion (`04_kelly_criterion`, 17.4)

| Setting | Formula / function | Notes |
|---|---|---|
| Binary wager | f* = (bp - (1-p))/b = edge/odds; g(f) = p log(1+bf) + (1-p) log(1-f) | `compute_kelly_fraction(win_prob, odds=1.0)`, `compute_growth_rate(win_prob, fraction, odds=1.0)`, `simulate_wealth_paths(outcomes, fraction, odds=1.0, initial_wealth=100.0)`, `terminal_wealth_distribution(n_trials, win_prob, fraction, odds=1.0, initial_wealth=100.0)`; fractions grid `np.linspace(0.01, 0.99, 100)`; multiples `[0.25, 0.5, 1.0, 1.5, 2.0]` on one shared outcome sequence (paired); plot median, 25-75 and 5-95 bands (`add_percentile_bands`). Only this form is exact. |
| Continuous, single asset | f* ~ (mu - r_f)/sigma^2 from log(1+x) ~ x - x^2/2 | `kelly_fraction_continuous(mean_return, std_return, risk_free=0.0)`; `RISK_FREE_RATE=0.02`; SPY |
| Empirical growth | annualized mean log(1 + r_f_daily + f (r - r_f_daily)) | `empirical_growth_rate(returns, fraction, risk_free=0.0)`; `optimal_kelly_empirical(returns, risk_free=0.0)` constrains wealth > 0 for every observed return (historical support, not a guarantee) |
| Rolling Kelly | 5-year window = 1260 sessions | `ROLLING_WINDOW_YEARS=5`; polars `rolling_mean`/`rolling_std`; shorter windows widen swings, longer flatten without becoming actionable |
| Multi-asset | f* = Sigma^-1 mu | `np.linalg.solve(annual_cov, annual_excess_returns)` (never an explicit inverse); shrunk variant `np.linalg.solve(shrunk_cov, ...)`; 8-ETF set SPY QQQ IWM EFA EEM TLT GLD VNQ, `START_DATE="2010-01-01"`, `END_DATE="2024-12-01"`, `TRAIN_END="2019-12-31"` |
| Fractional table | multiples `[0.25, 0.5, 1.0]` on the raw solution, no renormalization | Columns: gross exposure (sum of absolute weights), net exposure, Sharpe, `min_wealth_multiple`; rows ordered by gross exposure, not Sharpe; equal weight at gross 1 as reference |

Assumptions the derivation makes, in order: (1) odds known (in markets p, b are estimated; the solution divides by estimated variance and scales with estimated mean); (2) the bet repeats many times with a stake that is a fraction of wealth (a long-run argument; ruin-proof property lost once leverage exceeds one); (3) leveraged returns small enough for the second-order expansion (accuracy depends on the size of leveraged returns entering the log, not on normality); (4) borrowing free and unlimited (no funding rate, gross cap or liquidation rule). Gross vs net: 3x long / 2x short has gross 5, net +1; 2.5x / 2x has gross 5, net ~flat. Only 1/gross is the uniform adverse move that ends the account. Half-Kelly or less is the practical default.

### Factor allocation evidence (`05_factor_allocation_evidence`, 17.4)

1. Load monthly decimal returns: `FamaFrenchProvider` (FF3/FF5/MOM, 1926-present; divides percent data by 100), `AQRFactorProvider` (QMJ, BAB, VME 1972-; TSMOM 1985-; Century of Factor Premia 1926-2024; one-time `AQRFactorProvider.download()`, ~50 MB Excel; returns decimals as published). `dropna()`. `Mkt-RF` is excess; `HML`, `SMB`, `MOM` are long-short (already excess).
2. Significance: `calculate_mean_return_tstat(returns, periods_per_year=12)` (Newey-West) against `CONVENTIONAL_T_THRESHOLD=2.0` and `DISCOVERY_T_THRESHOLD=3.0` (Harvey, Liu & Zhu 2016). Applies to the mean-return t-stat, not Sharpe significance.
3. Sharpe uncertainty: `compute_autocorrelation_adjustment(returns, max_lag) -> (list[float], float)` inside `calculate_sharpe_stats(returns, periods_per_year=12)` (Lo-style diagnostic: rescales monthly estimate and SE to annual, inflates for autocorrelation); `calculate_max_drawdown(returns)`.
4. Diversification: value-momentum correlation across VME sleeves (`VALLS_VME_XX90` / `MOMLS_VME_XX90`; buckets `EQ`, `FX`, `FI`, `COM`; regions `US90`, `UK90`, `ROE90`, `JP90`; "Everywhere" uses bundled `VAL`/`MOM`). Correlation heatmap on the common window (shortest series), lower triangle, diverging scale centred at zero.
5. Crisis table: signed returns over ex-post `CRISES` windows and static NBER `RECESSIONS` through 2020; blanks = history not started. Describes, does not test.
6. SMB decay split at 1981 publication year: descriptive only (three explanations fit).
7. Summary: `append_summary_row(summary_stats, factor, source, returns, stats_dict)` for the risk-return map; three printed counts (factors clearing t=2, t=3, crisis windows with TSMOM > 0).
8. Beta: BAB levers low-beta leg to beta 1 and de-levers high-beta leg to 1, so net beta is zero by construction, realized can differ; correlation is not beta; run the exposure regression from 01.

### Hierarchical Risk Parity (`06_hierarchical_risk_parity`, 17.6)

1. Distance d_ij = sqrt((1 - rho_ij)/2) via `correlation_distance(corr)`; `cluster_assets(returns, method="ward")` (SciPy linkage; merge heights are Ward distances).
2. `get_quasi_diagonal_order(link) -> list[int]` leaf order; reorder covariance to reveal blocks. Only the leaf order reaches the allocation.
3. `cluster_variance(cov, indices)` = w'Σw of the block at inverse-variance weights; `recursive_bisection(cov, sorted_idx)` halves the ordered list by count at every level (not at tree cluster boundaries), allocates each half inversely to its cluster variance, shares multiply down the tree. `hrp_portfolio(returns) -> np.ndarray`; `trace_bisection(cov, sorted_idx, columns, levels=2)` prints halves, annualized cluster variances and split shares.
4. Baselines: `inverse_volatility_weights`, `equal_weights`, `minimum_variance_weights(returns)` = Σ^-1 1 / (1'Σ^-1 1), clip negatives, renormalize; `mvo_shrinkage_weights(returns)` = same with Ledoit-Wolf.
5. Walk-forward with a Chapter 12 ETF GBM signal: `LOOKBACK=252`, `TOP_N=5` of 15, `REBALANCE_FREQ="M"`, `MIN_HISTORY=60`; `START_DATE="2010-01-01"`, `END_DATE="2024-01-01"`, `SEED=42`; predictions resolved from `case_studies/etfs/run_log/registry.db` (best validation-IC GBM) at `<etf_case_dir>/run_log/predictions/<prediction_hash>/predictions.parquet`. `select_top_assets(date, full_returns, predictions, top_n)` as-of joins the latest prediction per symbol; `allocate_selected_assets(...)` validates long-only weights with an equal-weight fallback and a per-method counter on singular covariance; `rebalance_target(...)` returns None (prior target unchanged) when history/signal unavailable; `walk_forward_backtest(returns, allocation_fn, predictions, lookback, rebalance_freq, top_n, min_history, method)` holds positions between month-end decisions, weights drift, target effective next bar.
6. Execution bridge: `_weights_wide_to_long` -> timestamp/symbol/weight with zero targets kept (explicit liquidations); `ScheduledWeightStrategy(weights_long, allow_short)`; commissions, slippage, next-bar fills; synthetic OHLC = daily close (accounting/timing bridge, not fill quality).

HRP's case is N approaching or exceeding T; with 5 selected assets over 252 observations the covariance is well-conditioned and HRP's property is not exercised.

### Conformal position sizing (`07_conformal_position_sizing`, 17.4)

- Settings: `CONFORMAL_ALPHA=0.20` (80% interval), `TOP_K_ETF=20`, `TOP_K_CME=10`, `HORIZON_ETF=21`, `HORIZON_CME=5`, `TRADING_DAYS_PER_YEAR=252`. Inputs: registry best-IC GBM validation panels (`REGISTRY_ROOTS`, `BEST_GBM`, read-only; `ML4T_OUTPUT_DIR` may point at the teaching-registry overlay); entity column `symbol` for both (CME roots `ES`, `CL`, `6E`).
- Width: Delta = 2 * r_(ceil((n+1)(1-alpha))), rank capped at n (`conformal_order_stat(residuals, alpha)`). `mondrian_widths(df, id_col, horizon, alpha)`: per entity per fold, calibration residuals from chronologically prior folds whose origin + horizon is strictly earlier than fold k's first timestamp (order by timestamps, not fold IDs); earliest fold excluded. Normalize widths by case-study median for cross-study comparison.
- Rules (long-only, sum to one, top-K by signed yhat restricted to yhat > 0): equal 1/K; conformal w_i ∝ 1/Delta_i; score w_i ∝ yhat_i. Fewer than K positive -> smaller selection; none -> skip timestamp.
- Schedule: `nonoverlap_schedule(df, horizon)` keeps every h-th timestamp globally after dropping the first fold, no reset at fold boundaries; r_t = sum_i w_{i,t} y_{i,t} from `actual`; cohort held to maturity.
- Metrics: `build_portfolio_returns(df, widths, id_col, top_k, horizon)`; `compute_turnover(weights_long, id_col, weight_col)` = mean one-way turnover = half L1 change; `metric_block(rets, weights, id_col, horizon)` annualizes by 252/h (sqrt(12) for h=21, sqrt(50.4) for h=5); `allocator_table(metrics)`.
- Signal check first: daily cross-sectional Spearman IC via `extract_signal_ic_series`, `compute_ic_hac_stats` with >= h-1 lags; `resolve_best_prediction(case_study, label)` excludes degenerate folds and ranks by pooled IC; `load_predictions` rejects malformed keys.

### Library comparison (`08_library_comparison`, 17.7)

- Settings: `START_DATE="2018-01-01"`, `TRAIN_END="2021-12-31"`, `END_DATE="2024-12-01"`, `TRADING_DAYS=252`, `SEED=42`, `MAX_SYMBOLS=0`; `RISK_FREE_RATE=0.04`, `RISK_FREE_RATE_DAILY=0.04/252`; `CVAR_CONFIDENCE=0.95`; `FRONTIER_POINTS=50`; `COMMISSION_RATE=SLIPPAGE_RATE=0.0005`; `TRANSACTION_COST_PENALTY=0.01`; `L2_GAMMA=0.5`; `ACTIVE_WEIGHT_THRESHOLD=0.001`; `WEIGHT_TOLERANCE=1e-5`; `COMMON_WEIGHT_TOLERANCE=5e-4`; `COMMON_OBJECTIVE_TOLERANCE=1e-7`; `MAX_SHARPE_EXCESS_TOLERANCE=1e-12`; `INFEASIBLE_MAX_SHARPE_POLICY="cash"`; `CV_TRAIN_SIZE=252`, `CV_TEST_SIZE=63`.
- Units: estimate daily arithmetic moments once; PyPortfolioOpt gets annualized moments, Riskfolio-Lib and skfolio daily moments and a daily hurdle; annualize stats in the notebook.
- `max_sharpe_regime(expected_returns, risk_free_rate) -> str`: long-only max Sharpe needs >= 1 positive expected excess return, else hold cash at the hurdle, logged `cash_precheck` (economic policy, not a fallback objective); applied before every skfolio walk-forward fold.
- Hygiene: `align_weights(weights, name) -> pd.Series` canonical symbol order + validation; `assert_optimal_status(problem, name)` exact optimal cvxpy status; `suppress_riskfolio_cvxpy_star_warning()`, `suppress_ppo_max_sharpe_objective_warning()` scoped to one message/category/module; `run_riskfolio(operation, name)` captures statuses; `common_empirical_prior() -> EmpiricalPrior`; `negative_common_sharpe(weights)`, `empirical_cvar(weights)` as independent checks of the shared objective.
- Runs: PyPortfolioOpt Max Sharpe, Min Volatility, CVaR, HRP, Ledoit-Wolf shrinkage, turnover-penalized and L2-regularized pairs; Riskfolio Max Sharpe under MV / MDD / CDaR, CVaR min-risk, risk parity, frontiers (`frontier_to_df`; risk codes MV, MAD, MSV, GMD, KT, SKT; CVaR, EVaR, RLVaR, WR; MDD, ADD, CDaR, EDaR, RLDaR, UCI); skfolio mean-risk, HRP, `WalkForward` (252/63) inside the training window.
- Test: 14 frozen allocations + EW; metrics, growth paths, risk-return map, rank heatmap, HHI vs EW; one allocation replayed via `DailyTargetWeightStrategy`; first test return is NEXT_BAR warm-up, excluded from both paths.

| Library | Pick when |
|---|---|
| PyPortfolioOpt | readable classic MVO with explicit penalties (turnover, L2) |
| Riskfolio-Lib | sweeping many risk measures via string codes |
| skfolio | portfolio construction as a stage in a sklearn pipeline (Pipeline, GridSearchCV, WalkForward, CPCV) |

### Controlled allocator comparison (`09_allocator_comparison`, 17.7)

- Dates: `START_DATE="2016-01-01"`, `END_DATE="2024-01-01"`, `BACKTEST_START="2018-01-01"`, `SELECTION_END="2021-12-31"`, `HOLDOUT_START="2022-01-01"`. Signal: `LABEL_HORIZON=5`, `SIGNAL_LOOKBACK=252`, `REBALANCE_FREQ=21`, `MOMENTUM_WINDOWS=[21,63,126]`, `MOVING_AVERAGE_WINDOWS=[21,63]`, `VOLATILITY_WINDOW=21`. Book: `MAX_SIDE_POSITIONS=10`; `TOP_N=BOTTOM_N=min(10, universe/4)`. `ALLOCATION_WINDOW=252`; `TURNOVER_INCLUDE_INITIAL=True`; `COMMISSION_RATE=0.001`, `SLIPPAGE_RATE=0.0005`; `CI_STANDARD_ERRORS=1.96`; `SEED=42`. Universe: 39 ETFs (US_EQUITIES, INTERNATIONAL, FIXED_INCOME, ALTERNATIVES, SECTORS; "36-ETF protocol" after coverage, inference) via `fetch_etf_data`.
- Signal: `generate_ml_signals(prices, lookback=252, horizon=5, top_n=10, bottom_n=10, rebalance_freq=21)` walk-forward cross-sectional Ridge; label = cumulative return t..t+5; a row enters a fit only after its label end date is observed (5-day purge).
- Allocators: `equal_weight_allocation(selected, n_assets)`; `inverse_vol_allocation(returns, selected, window=63)`; `mvo_lw_allocation(returns, selected, window=252)`; `hrp_allocation(returns, selected, window=252)`.
- Backtest: `run_backtest(returns_df, signals_df, allocation_fn, allocation_name, window=252) -> dict` (returns, dates, cumulative, turnover, positions, metrics); `compute_rebalance_turnover(weights, previous)` = ½ Σ|w_t - w_{t-1}| over the union of symbols; `average_rebalance_turnover(list, include_initial=True)`; `slice_signals(signals_df, start, end)` inclusive.
- Inference: `annualized_sharpe_se(sharpe, n, periods_per_year=252)` = sqrt((P + SR^2/2)/T); `build_comparison_table(period_results)`. For a gap, Var(S1)+Var(S2)-2Cov(S1,S2) -> paired block bootstrap (resample date blocks, recompute both Sharpes on the same dates).
- Regime diagnostic: threshold = median trailing 63-day SPY vol over the selection region, frozen, applied to holdout.
- Sequence: select allocator once on pre-holdout data -> describe holdout -> regime slice -> `PortfolioTearSheet` (SPY benchmark; `.show()` / `.save_html(path)`; `style_diagnostic_figures`) and engine replay (`AllocatorComparisonStrategy`, NEXT_BAR, 10 bp + 5 bp) for the pre-selected allocator only.

### Execution bridge (shared by 02, 03, 06, 08, 09)

Always three rows: (1) vectorized close-to-close daily rebalance; (2) engine with next-bar execution on actual OHLCV at zero cost (`CommissionType.NONE`, `SlippageType.NONE`); (3) cost-aware engine (`CommissionType.PERCENTAGE`, `commission_rate=0.0005`; `SlippageType.PERCENTAGE`, `slippage_rate=0.0005`), `initial_cash=100_000.0`. Attribute row1->row2 to timing and row2->row3 to fees; never the whole gap to fees. `run_daily_target_engine(*, cost_aware, return_column)`.

### End-to-end deep allocators (11, 12, 13; 17.8)

| Setting | `11_dl_portfolio_allocation` | `12_vlstm_portfolio` | `13_deepm_regime_robust` |
|---|---|---|---|
| Universe | 29 ETFs (US_EQUITY 6, INTERNATIONAL 4, FIXED_INCOME 7, ALTERNATIVES 4, SECTORS 8) | same | same, 5 groups (`us_equity`, `intl_equity`, `fixed_income`, `alternatives`, `sectors`) |
| Features | log returns `HORIZONS=[1,5,21,63]` + rolling vol | `ret_1d, ret_5d, ret_21d, ret_63d, vol_21d` (F=5) | `build_feature_panel`: vol-normalized ret 1/21/63/252, MACD (8,24)(16,48)(32,96), z-scores 21/252, MAD clip 252, existence flags |
| `SEQ_LEN` | 63 | 63 | 84 |
| Model | `LSTMPortfolioNet(n_features, n_assets, hidden_dim=64)`: LSTM -> linear -> softmax (long-only, sum 1) | `VLSTM`: VSN -> shared per-asset LSTM -> post GRN -> linear -> tanh signal; `D_MODEL=32`, `LSTM_HIDDEN=64`, `DROPOUT=0.0` | `DeepmPolicy`: FiLM context, V-VSN, LSTM, cross-sectional attention with Directed Delay (`cross_attention_lag=1`), macro-graph attention; `D_MODEL=64` (paper 128), `n_heads=4`, `dropout=0.3` |
| Positions | softmax weights | w = 0.15/(sqrt(252)*max(vol63,1e-4)*N) * signal (long-short, vol-targeted) | risk signal * `panel.vol_scale`, normalized to sum abs(w) = 1 at evaluation |
| Loss | `differentiable_sharpe_loss(weights, fwd_returns, annualization=252.0, eps=1e-8)`, gross | `-pooled_sharpe(net)` with 5 bps in-loss cost, `COST_WEIGHT=1.0` | `-SR_pool - 0.1 * SoftMin_0.2({SR_b})`, per-asset 5/10 bps, `gamma_cost=0.5` |
| Training | `N_EPOCHS=200`, whole window as one batch | Adam lr 1e-3, wd 1e-5, cosine, grad clip 1.0, one step per epoch, 200 epochs, chronological chunks `batch_size=32, shuffle=False` | lr 1e-4, wd 1e-4, `max_grad_norm=1.0`, `max_iters=500`, `eval_every=25`, windows `shuffle=True, drop_last=True` |
| Selection | validation-peak checkpoint | max validation pooled Sharpe | validation-Sharpe peak, early stopping patience 50 after burn-in 50, EMA alpha 0.45, min delta 0.001 (inference) |
| Baselines | EW, IV (21d) | EW, IV (63d), long-only sum-to-1, all charged 5 bps | EW, IV (21d), same net-return function |
| Extras | HHI over time, realized turnover | signal histogram, `signal_at_bounds`, VSN weights vs 20% uniform, cost grid [0,5,10,20,50] bps | regime table (SPY 21d vol median), drawdown panel -> `drawdowns.parquet`, no-SoftMin ablation |

Shared pipeline: `load_etfs()` -> pivot `close` -> keep dates with >= 80% asset coverage (`dropna(thresh=max(1, int(n*0.8)))`) -> (12) `ffill().dropna()` / (13) no ffill, existence features -> returns `pct_change(fill_method=None)` -> labels `fwd_returns = returns.shift(-1)` -> 60/20/20 chronological split (`train_end=int(0.6n)`, `val_end=int(0.8n)`); test decisions end at `dates[-2]`.

VLSTM two-pass exact pooled-Sharpe gradient: (1) no-grad chronological pass collects endpoint net returns r (prepending the predecessor endpoint to every chunk after the first so turnover is continuous); `g = d(-Sharpe(r))/dr` by autograd on a detached leaf; (2) second pass recomputes each chunk with grad and calls `torch.sum(net_r * g[start:stop]).backward()`; then `clip_grad_norm_(1.0)`, `optimizer.step()`. Requires `DROPOUT=0.0` and an unshuffled loader. `pooled_sharpe(returns, ann_factor=252.0, eps=1e-6)`.

DeePM run order: determinism block -> universe + groups -> 80% coverage panel -> `build_feature_panel(prices, FeatureConfig(vol_span=63, ret_horizons=(1,21,63,252), macd_pairs=((8,24),(16,48),(32,96)), zscore_windows=(21,252), clip_window=252, include_existence=True))` -> `build_macro_adjacency(assets, asset_to_group, cross_edges=[("us_equity","sectors"),("fixed_income","alternatives")])` -> `adjacency_to_attn_mask` -> date split -> `DeepmWindowDataset(panel, seq_len, start_date, end_date)` x3 -> `COST_BPS` (5 bps; 10 bps for SLV, DBC, EEM, VWO, HYG, EMB) -> `build_static_metadata(assets, asset_to_group, asset_to_cost_bps)` -> `ModelConfig(...)` -> `DeepmPolicy(n_assets, n_features, n_groups, adjacency_mask, cfg)`; snapshot `initial_state` -> `TrainingConfig(seq_len=84, burn_in=21, batch_size=32, learning_rate=1e-4, weight_decay=1e-4, max_grad_norm=1.0, gamma_cost=0.5, softmin_tau=0.2, softmin_lambda=0.1, max_iters=500, eval_every=25, metric_ema_alpha=0.45, metric_min_delta=0.001, early_stopping_patience=50, early_stopping_burn_in_iters=50)` -> `train_model(model, train_loader, val_loader, static_meta, cfg) -> (best_state, history)` -> ablation: new policy, `load_state_dict(initial_state)`, same config with `softmin_lambda=0.0` -> `infer_risk_weights_rolling(model, panel, static_meta, seq_len, batch_size=256, device)` -> `portfolio_returns_from_weights(weights_df, returns_df, mask, cost_bps)` for DeePM, no-SoftMin, EW, IV -> `compute_perf_metrics` ("Ann. Mean Return" = arithmetic mean * 252, explicitly not compounded; N/A below 10 observations) -> regime table (calm = SPY 21d ann. vol <= test-period median; crisis > median; `Gap = Calm - Crisis`) -> drawdown panel to `get_output_dir(17, "deepm_regime_robust") / "drawdowns.parquet"`.

## Guardrails and pitfalls

- **Selection bias in the reported strategy** — ranking candidates by validation Sharpe then reporting the winner's statistics overstates them / treat the report as a diagnostic of a chosen artifact; keep a holdout untouched by selection.
- **In-sample frontier read as evidence** — every point is where the optimizer would have put you given the data it saw / estimate through `TRAIN_END`, freeze, score later; never pick an allocator or parameter from the test result; do not difference the full-sample and train/test tables (they do not share weights).
- **Ill-conditioned covariance** — inversion divides by near-zero eigenvalues from redundant assets / compute kappa(Sigma) and the eigenvalue spectrum first; shrink (Ledoit-Wolf); narrow the universe (11 assets = 55 covariances vs 30 = 435); check conditioning, not only N/T.
- **Expected-return noise dominates MVO** — endpoint CAGR rests on two prices per asset / prefer allocators that drop mu (min var, risk parity, HRP, EW) or shrink; measure concentration with effective positions.
- **Corner solutions from pure risk minimizers** — min variance and min CDaR load into the lowest-risk asset as hard as max Sharpe into the highest-return one / check effective positions and largest weight; add a diversification requirement (risk parity) or constraints.
- **Risk contributions and drawdown targets do not transfer** — risk parity equalizes shares only against the fitted covariance; min CDaR optimizes the tail it saw / recompute risk contributions on the test covariance; read test drawdowns against the others before claiming protection.
- **Wrong risk-contribution formula** — w_i d(var)/dw_i / var carries a factor two and sums to two / use w_i (Sigma w)_i / (w^T Sigma w), assert shares sum to one, verify with the diagonal oracle (IV gives 1/N each).
- **Mistaking HRP for diversified** — inverse-variance splits concentrate on low-variance assets (a majority in one bond fund on 15 ETFs) / trace bisection splits, report HHI, pair with exposure constraints.
- **Testing HRP in the wrong regime** — 5 assets over 252 observations is well-conditioned; HRP's case is N near or above T / compute N/T and conditioning before interpreting a ranking.
- **Solver output silently invalid** — SLSQP/CVXPY can return infeasible, NaN or budget-violating weights / `validate_optimization_result(... tolerance=1e-6)`, `validate_weights`, `assert_optimal_status` (exact optimal cvxpy status); pre-check Riskfolio/CVXPY versions; never blanket-suppress warnings (scope by message, category, module).
- **Max Sharpe with no positive expected excess return** — the ratio solver is ill-posed / `max_sharpe_regime` precheck -> all cash, logged `cash_precheck`.
- **Library unit and container mismatch** — PyPortfolioOpt wants annualized moments, Riskfolio/skfolio daily; weight containers differ / estimate once in daily units, convert at each boundary, `align_weights` to canonical order, verify matched objectives agree to `COMMON_OBJECTIVE_TOLERANCE=1e-7` and weights to `WEIGHT_TOLERANCE=1e-5`; label mismatch in heatmaps: drive axes and values from one symbol order.
- **Walk-forward CV inside the training window mistaken for a test** — every fold sits inside training / keep the test window untouched; call CV a stability demonstration.
- **Holdout used as a second selection stage** — nothing is left to test the choice / select exactly once before `HOLDOUT_START`; mark "Selected pre-holdout"; never feed holdout ordering into the tear sheet or engine; freeze regime thresholds (median 63-day SPY vol) on the selection region.
- **IID Sharpe SE read as a comparison** — allocators share assets and dates (Cov != 0) and returns are serially dependent / paired block bootstrap of the Sharpe difference; SE ~ 1/sqrt(years): a few tenths over a few years is not a separation; IR * sqrt(T) as a t-stat.
- **One split, one universe, one window** — a different split refits everything and could reorder the table; no confidence intervals / report orderings as descriptions of one period; never re-tune on the result.
- **Survivorship / current-vintage ETF universes** — fixed lists chosen because the funds exist today; factor series are revisable and dated by return month / present as allocator mechanics or historical evidence, not survivorship-free tests; show coverage/inception tables; use point-in-time membership for a real test.
- **Selection signal ranked on validation (06, 07)** — allocation measured on a sample the selection already saw / label as validation diagnostics; require an untouched holdout.
- **Paper vs implementable results** — vectorized rebalancing ignores next-bar timing and costs / run the three-row execution bridge; attribute vectorized->zero-cost to timing, zero-cost->cost-aware to fees; isolate costs with two otherwise identical engine runs; exclude the NEXT_BAR warm-up return from both paths.
- **Double-counting costs** — the registered backtest artifact already includes commission, slippage and next-bar execution / do not subtract again in the metrics notebook.
- **Frozen weights with daily rebalance understate turnover** — real allocations re-estimate periodically, adding turnover and fresh estimation error / note as limitation; use walk-forward re-estimation (06, 09).
- **Turnover misread** — it bundles re-selection and sizing; equal weight is not a floor / separate turnover from entering/leaving names vs retained names; price it (Ch18); gross Sharpe gains bought with turnover are unproven until priced.
- **Zero risk-free rate flatters ratios** — over 2022-23 cash paid / set `RISK_FREE_RATE` to a cash rate; a common rate can reorder strategies (lower-vol series loses more); IR is unaffected.
- **Hand-picked stress windows and publication-year splits** — chosen after the fact / label as description, never as a test of insurance or decay.
- **Empirical VaR/CVaR as forecasts** — the 1-in-100 quantile rests on a few dozen points and never extrapolates past the worst day / quote both; do not read as forecasts.
- **Compounded-wealth capture ratio** — library `up_down_capture` inverts the down reading / use ratio of mean monthly returns, compound to month ends first.
- **Stability R^2 misread** — a line over eight years absorbs a multi-month drawdown / read beside the drawdown table.
- **Sharpe hides ruin in leveraged series** — mean/std ignore a wealth path that went to ~0 / compute `min_wealth_multiple` on the compounded path; order Kelly tables by gross exposure.
- **Kelly leverage unbounded** — f* = Sigma^-1 mu has no budget, funding rate, cap or liquidation rule; fractional Kelly only scales it / always impose an explicit leverage constraint; compute gross exposure and its reciprocal.
- **Second-order Kelly breaks under leverage** — dropped terms grow with leveraged return size / use the empirical growth optimizer with the positivity constraint; keep leveraged returns small.
- **Rolling Kelly instability** — mu's SE falls only with sqrt(window); 5-year SPY fractions swing across an unactionable range / treat Kelly as a direction and a cap; shrink or cap.
- **Percent vs decimal units** — mixing rescales every Sharpe and t-stat by 100 / `FamaFrenchProvider` divides by 100, `AQRFactorProvider` does not; keep the unit sanity check.
- **Conventional t=2 on published factors** — survivors were selected on that bar / apply t >= 3 (Harvey) to the Newey-West mean-return t-stat, not to Sharpe; scale the bar to candidates tried; use Newey-West / Lo-style adjustments for serial dependence.
- **Correlation is not beta; dollar-neutral is not beta-neutral** — correlation is scaled by both vols; equal dollars with unequal betas leave net beta / run the exposure regression; compute correlations on a common window only.
- **Long factor histories read as achievable** — gross of financing, turnover, capacity, tax / treat as evidence about published returns; cost it in Ch16/Ch18.
- **Lookahead or pooling in conformal calibration** — residuals with unobserved labels leak; one width per fold collapses to equal weight / per-entity (Mondrian) pool of prior folds with origin + horizon strictly before fold start; exclude the first fold; check width dispersion; confirm IC > 0 (HAC t-stat) before sizing by uncertainty.
- **Overlapping forward-return labels** — inflate Sharpe, break drawdown accounting, manufacture IC precision / global every-h-th schedule with no fold-boundary reset; annualize by 252/h; HAC with >= h-1 lags.
- **Singular covariance in walk-forward** — inversion fails or yields garbage / validate weights; visible equal-weight fallback with per-method counter; `MIN_HISTORY=60`; `rebalance_target` returns None to keep the prior target.
- **Label leakage across DL partitions and via the last date** — a window whose scored endpoint lies in the next partition contaminates validation; the last date has no realized return / assert endpoint sets disjoint and `val.indices[0]==train_end` (VLSTM) or whole windows inside partitions (DeePM); decisions end at `dates[-2]`.
- **Non-deterministic GPU training** — cuDNN/cuBLAS pick kernels by timing, so ablations carry two changes / `set_global_seeds(42)`, `torch.use_deterministic_algorithms(True, warn_only=True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=":4096:8"`.
- **Stochastic layers or shuffled loaders with the two-pass loss** — recomputation would differ from the reference pass; cost needs consecutive decisions / `DROPOUT=0.0`, `shuffle=False, drop_last=False`; `prepend_predecessor` so chunk boundaries do not charge spurious entry trades.
- **Overfitting to training Sharpe** — train Sharpe keeps rising after validation flattens / checkpoint at the validation peak; early stopping; read trends not single dips.
- **Unfair ablation and single-seed inference** — building the ablation after training moves the RNG; one seed cannot size an effect / snapshot `initial_state` before training; change only `softmin_lambda`; repeat seeds before claiming an effect.
- **Cost-assumption dependence and gross vs net tables** — an allocator that wins at 5 bps may lose at 20; 11 reports gross, 12 net; IV baselines differ (21d vs 63d) / run the cost grid [0,5,10,20,50]; if the ranking reorders between 0 and 20 bps no advantage is established; compare only via the zero-cost row; flat bps are a proxy for Ch18 cost models.
- **Signal collapse and vol-driven churn** — a signal piled at zero is "declined to trade", not an allocation; rolling-vol jumps masquerade as model indecision / histogram of the signal, `signal_at_bounds`; plot signal and vol separately; price persistent position change.
- **Regime-gap misreading** — SoftMin penalizes the weakest training windows, never holdout SPY vol; the calm-crisis gap can widen while the objective improves / read Gap as a holdout diagnostic, not the objective's score.
- **Leverage convention mismatch in evaluation** — raw vol-scaled weights have arbitrary gross exposure / normalize all allocators to sum abs(w) = 1 per date through one net-return function (DeePM); note that VLSTM compares vol-targeted long-short against long-only sum-to-1 baselines.
- **NaN objectives and sparse history** — degenerate windows give NaN Sharpe; late-listed assets shift weights / skip NaN evaluations and raise if no checkpoint; 80% coverage gate; zero weights where the forward return is NaN; metrics N/A below 10 observations.
- **Allocator-selection overfitting (17.7)** — trying allocators until one works is multiple testing / identical inputs and protocol, count degrees of freedom, select before the holdout opens.

## Decision rules and defaults

| Rule | Default |
|---|---|
| Which allocator needs what | Max Sharpe / Kelly: mu + Sigma; Min Variance, Risk Parity, HRP: Sigma only; Min CDaR: realized path; EW: nothing (the benchmark) |
| Go/no-go for a sophisticated allocator | Must beat EW / IV / risk parity on identical inputs and protocol, net of the execution bridge |
| Use HRP when | N approaches or exceeds T (sample covariance singular at N > T); with N=5, T=252 its property is not exercised; shrinkage is the alternative when still inverting |
| Covariance / mu for optimizers | Ledoit-Wolf (`method_cov="ledoit"`); training sample mean (`method_mu="hist"`) |
| Hurdle rate | Declared per experiment: 0.04 (02, 08), 0 (03), 0.02 (04); set to a cash rate for real reports |
| Periods per year | 252 daily; 12 monthly (factor data); 252/h for non-overlapping h-day cohorts |
| Train/test split (02/03/04) | `TRAIN_END="2019-12-31"`, `SEED=42`; 08: `TRAIN_END="2021-12-31"`; 09: `SELECTION_END="2021-12-31"`, `HOLDOUT_START="2022-01-01"`; DL: 60/20/20 chronological |
| Execution bridge | Three rows; 5 bp commission + 5 bp slippage (09 replay: 10 bp + 5 bp); `initial_cash=100_000`; exclude NEXT_BAR warm-up |
| Concentration metrics | Effective positions = 1/sum w^2, largest weight, HHI vs 1/N; risk contributions summing to one |
| Rolling windows | 21 / 63 / 252 sessions (report); 252 with `min_periods=126` for rolling Sharpe diagnostic |
| Beta labels / IR | beta < 0.8 less exposed, > 1.2 more; t ~ IR * sqrt(T years), need ~2 |
| Sharpe SE | IID: sqrt((252 + SR^2/2)/T); ~1/sqrt(years); a Sharpe of 1 is unremarkable daily and hard at quarterly holding; gaps via paired block bootstrap; CI multiplier 1.96 |
| Capture ratios | Monthly frequency, ratio of means |
| VaR confidence | 0.95 and 0.99, empirical; CVaR confidence 0.95 in 08 |
| Dirichlet exploration | alpha = 0.05 for the bullet's edges |
| Kelly | Binary edge/odds; continuous (mu - r_f)/sigma^2; multi-asset Sigma^-1 mu; multiple <= 0.5 plus explicit leverage cap; rolling window 5 years (1260 sessions); order tables by gross exposure; inspect `min_wealth_multiple` |
| Factor significance | NW mean-return t >= 2.0 conventional, >= 3.0 discovery; scale to candidates tried; diversification claim needs common-window negative correlation + structural mechanism + replication (VME: 8 markets) + crisis table as description only |
| 06 walk-forward | LOOKBACK 252, TOP_N 5 of 15, month-end rebalance, MIN_HISTORY 60, equal-weight fallback counted |
| 07 conformal | alpha 0.20; K = 20 (ETF, h=21) / 10 (CME, h=5); exclude first fold; HAC lags >= h-1; confirm IC > 0 then width dispersion before sizing |
| 08 library | rf 4% (daily 0.04/252); 50 frontier points; turnover penalty 0.01; L2 gamma 0.5; active weight 0.001; tolerances 1e-5 / 5e-4 / 1e-7 / 1e-12; infeasible Max Sharpe -> cash; CV 252/63 |
| 09 comparison | signal lookback 252, horizon 5 (purged), rebalance 21; <= 10 names per side and <= universe/4; IV window 63; MVO-LW and HRP window 252; regime threshold = median trailing 63-day SPY vol on selection region, frozen |
| Turnover convention | one-way = ½ Σ|Δw|; 09 includes the initial allocation from cash |
| Simulation choice | Weight-based (vectorized) for method selection / isolating allocation effects; `ml4t-backtest` for execution costs, fill diagnostics, validation, path to live |
| DL defaults | 11: SEQ_LEN 63, HIDDEN 64, 200 epochs, eps 1e-8, IV window 21. 12: SEQ_LEN 63, D_MODEL 32, LSTM_HIDDEN 64, DROPOUT 0, LR 1e-3, WD 1e-5, BATCH 32, 200 epochs, VOL_TARGET_ANN 0.15, VOL_LOOKBACK 63, 5 bps, COST_WEIGHT 1.0. 13: SEQ_LEN 84, D_MODEL 64, heads 4, dropout 0.3, lr 1e-4, wd 1e-4, clip 1.0, gamma_cost 0.5, tau 0.2, lambda 0.1, burn_in 21, 500 iters, eval every 25, patience 50 / burn-in 50, EMA 0.45, min delta 0.001, inference batch 256 |
| DL costs | 5 bps liquid ETFs; 10 bps SLV, DBC, EEM, VWO, HYG, EMB; stress grid 0/5/10/20/50 |
| DL data gates | keep a date only with >= 80% asset coverage; decisions never include the final date; Sharpe only with >= 10 days |
| DL go/no-go | Must clear BOTH EW and IV net of in-loss cost; ranking survives the cost grid to 20 bps; signal uses the tanh range; VSN weights concentrate (near-uniform 20% = not paying for itself); position changes spike on regime change rather than staying high; HHI above EW -> add exposure constraints; compare DeePM vs no-SoftMin in both the held-out and regime tables; one-seed gaps are hypotheses |
| VLSTM vs LSTM vs DeePM | Add VSN for feature-level interpretability when selection weights concentrate; DeePM when regime robustness and cross-asset structure are the goal and ~2x the configuration surface is affordable |

Chapter sequencing: metrics (01) -> MVO instability (02) -> shrinkage / risk-only objectives (03) -> Kelly (04) -> factor evidence (05) -> HRP (06) -> conformal sizing (07) -> library cross-check (08) -> fair comparison with pre-holdout selection (09) -> DL allocators (11 -> 12 -> 13) -> Ch18 costs -> Ch20 `05_portfolio_allocation`.

Pre-optimizer checklist: (1) kappa(Sigma) and eigenvalue spectrum; (2) count redundant pairs; (3) decide whether mu is needed at all; (4) Ledoit-Wolf if an inverse is taken; (5) validate solver output (finite, bounds, budget, tolerance 1e-6); (6) effective positions and largest weight; (7) freeze at `TRAIN_END`, score on test; (8) three-row execution bridge.

Report-reading checklist: compare the dashboard's metrics object to hand-computed values; read 21/63/252 rolling Sharpe together; stability R^2 beside drawdowns; VaR and CVaR together; state the horizon with every IR and Sharpe; titles and alt text on pulled figures; `include_plotlyjs="cdn"` unless offline.

Kelly checklist: full-Kelly leverage, gross exposure and its reciprocal; apply a multiple <= 0.5 and an explicit cap; recompute on rolling windows before acting; order by gross exposure; inspect `min_wealth_multiple`.

DL pre-run checklist: pin RNGs and GPU kernels, snapshot `initial_state` if ablating; 80% coverage panel; partition asserts; loader order matches cost accounting; print cost assumptions next to training settings; keep validation-peak checkpoint and inspect train/val divergence; evaluate every allocator through ONE net-return function with identical cost schedule and leverage; cost grid and behavioural diagnostics before the headline Sharpe; single-seed results as diagnostics; persist figure data to parquet.

## Code patterns and APIs

Imports and entry points:

```python
from data import load_etfs                      # polars: timestamp, symbol, close
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, ml4t_palette, ml4t_diverging, show_with_alt, show_plotly_with_alt, zero_line, add_message_title
from utils.paths import get_output_dir          # get_output_dir(17, "deepm_regime_robust")
from ml4t.diagnostic.evaluation import PortfolioAnalysis           # PortfolioAnalysis(strategy_returns, benchmark_returns, dates, risk_free_rate=0.0, periods_per_year=252)  (signature inferred)
from ml4t.diagnostic.evaluation.portfolio_analysis import (compute_drawdown_analysis, DrawdownResult, up_down_capture, value_at_risk, conditional_var, create_portfolio_dashboard, annual_return, annual_volatility, max_drawdown)
from ml4t.diagnostic.metrics import sharpe_ratio, sortino_ratio, periodic_sortino_ratio
from ml4t.diagnostic.metrics.ic_inference import compute_ic_hac_stats
from ml4t.diagnostic.signal.signal_ic import extract_signal_ic_series
from ml4t.diagnostic.evaluation.factor import FactorAnalysis, load_fama_french_5factor
from ml4t.diagnostic.visualization import combine_figures_to_html, save_html, create_portfolio_dashboard
from ml4t.data.providers import AQRFactorProvider, FamaFrenchProvider
from ml4t.backtest import Engine, Strategy, CommissionType
from ml4t.backtest.config import SlippageType
from ml4t.backtest.execution.rebalancer import RebalanceConfig, TargetWeightExecutor
import riskfolio as rp
```

Frozen-target strategy adapter and cost toggle (02, 03, 08):

```python
class DailyTargetWeightStrategy(Strategy):
    def __init__(self, target_weights: dict[str, float], allow_short: bool):
        self.executor = TargetWeightExecutor(config=RebalanceConfig(...))   # fields not visible
    def on_data(self, timestamp, data, context, broker):
        ...  # submit the same frozen targets every bar (NEXT_BAR fills)

commission_type = CommissionType.PERCENTAGE if cost_aware else CommissionType.NONE
commission_rate = 0.0005 if cost_aware else 0.0
slippage_type = SlippageType.PERCENTAGE if cost_aware else SlippageType.NONE
slippage_rate = 0.0005 if cost_aware else 0.0
```

Other adapters: `ScheduledWeightStrategy(weights_long, allow_short)` (06; long frame from `_weights_wide_to_long`, zeros kept); `AllocatorComparisonStrategy(assets, signals_df, allocation_fn, allocation_name, returns_df, allocation_window)` (09).

SciPy MVO (call shape inferred):

```python
res = scipy.optimize.minimize(neg_sharpe, w0, args=(returns, cov, rf), method="SLSQP",
                              bounds=[(0, 1)] * n, constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
validate_optimization_result(res, label="max_sharpe", tolerance=1e-6)
```

Risk contribution, effective positions, drawdown, rolling Sharpe, Kelly:

```python
rc = w * (Sigma @ w) / (w @ Sigma @ w); assert abs(rc.sum() - 1) < 1e-9
eff_positions = 1 / (w ** 2).sum()
wealth = (1 + r).cumprod(); dd = wealth / wealth.cummax() - 1
roll_sharpe = r.rolling(252, min_periods=126).mean() / r.rolling(252, min_periods=126).std() * np.sqrt(252)
f_star = np.linalg.solve(annual_cov, annual_excess_returns)      # multi-asset Kelly; gross = np.abs(f_star).sum()
sharpe_se = np.sqrt((252 + sharpe ** 2 / 2) / n_observations)
```

HRP core (06) and split-conformal width (07):

```python
dist = np.sqrt((1 - corr) / 2)
link = scipy.cluster.hierarchy.linkage(squareform(dist), method="ward")
order = get_quasi_diagonal_order(link)
w = recursive_bisection(cov, order)        # halves by count, inverse cluster-variance split

n = len(res); k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
q = np.sort(np.abs(res))[k - 1]; width = 2 * q
```

Riskfolio-Lib (03): `rp.Portfolio(returns=df).assets_stats(method_mu="hist", method_cov="ledoit")`; `port.optimization(model="Classic", rm="MV"|"CDaR", obj="Sharpe"|"MinRisk", rf=0, hist=True)`; `port.rp_optimization(model="Classic", rm="MV", rf=0, hist=True)`; `rp.HCPortfolio(returns=df).optimization(model="HRP", codependence="pearson", rm="MV", rf=0, linkage="ward", max_k=10, leaf_order=True, method_cov="ledoit")`. PyPortfolioOpt (08): `EfficientFrontier.max_sharpe()`, `.min_volatility()`, `portfolio_performance()`, turnover / L2 objective penalties, `CovarianceShrinkage` (Ledoit-Wolf), `HRPOpt`. skfolio: `MeanRisk`, `HierarchicalRiskParity`, `EmpiricalPrior`, `WalkForward`, CPCV. `sklearn.covariance.LedoitWolf` for 06/09 (inference). cvxpy: check `cp.Problem` status exactly.

Registry lookups: `case_studies/etfs/run_log/registry.db` (SQLite, read-only); backtest returns `run_log/backtest/<hash>/daily_returns.parquet`, spec `run_log/backtest/<hash>/spec.json`; predictions `run_log/predictions/<prediction_hash>/predictions.parquet` (columns incl. `symbol`, prediction, `actual`, fold); `ML4T_OUTPUT_DIR` overrides the output root.

DL patterns (11-13):

```python
# determinism block
set_global_seeds(SEED); torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# differentiable Sharpe loss (11)
r_p = (weights * fwd_returns).sum(-1)
loss = -(np.sqrt(252) * r_p.mean()) / torch.sqrt(r_p.var() + 1e-8)

# exact pooled-Sharpe gradient without a full graph (12)
ref = collect_endpoint_returns(loader, ds)                 # no-grad pass
ref_var = ref.detach().requires_grad_(True)
g = torch.autograd.grad(-pooled_sharpe(ref_var), ref_var)[0].detach()
for chunk: net_r = ...forward with grad...; torch.sum(net_r * g[start:stop]).backward()
clip_grad_norm_(model.parameters(), 1.0); optimizer.step()

# vol-targeted positions and net return with turnover cost (12)
w = (VOL_TARGET_ANN / (np.maximum(vol, 1e-4) * np.sqrt(252) * N)) * signal
prev = np.vstack([np.zeros(N), w[:-1]]); gross = (w * fwd).sum(1)
cost = np.abs(w - prev).sum(1) * bps / 1e4; net = gross - cost
# per-asset schedule (13)
cost_rates = pd.Series(cost_bps).reindex(cols).fillna(0) / 1e4
costs = turnover.mul(cost_rates, axis=1).sum(axis=1)
```

`deepm` package (`17_portfolio_construction/deepm/`): `configs.FeatureConfig / ModelConfig / TrainingConfig`; `dataset.DeepmWindowDataset`, `dataset.build_static_metadata`; `features.build_feature_panel`; `graph.build_macro_adjacency`, `graph.adjacency_to_attn_mask`; `inference.infer_risk_weights_rolling`; `model.DeepmPolicy`; `train.train_model`. Polars at the boundary: `etf_data.filter(pl.col("symbol").is_in(UNIVERSE)).pivot(on="symbol", index="timestamp", values="close").sort("timestamp").to_pandas().set_index("timestamp")`; write back with `pl.from_pandas(df).write_parquet(path)`. Papermill: `# %% tags=["parameters"]` cell with production defaults (`MAX_SYMBOLS=0`, `N_EPOCHS`/`MAX_ITERS` overridable); results cell `# %% [markdown] tags=["results"]`.

Run: `uv run python 17_portfolio_construction/<notebook>.py`; tests `uv run pytest tests/test_chapter_notebooks.py -v -k "17_portfolio_construction"`; 11-13 use `docker compose run --rm ml4t-gpu python 17_portfolio_construction/12_vlstm_portfolio.py` (CPU works, slowly).

## Evidence from the book

Numeric outputs are produced at run time and were not stored in the sources; findings below are qualitative. The book PDF was unavailable, so 17.1, 17.2, 17.7 and 17.8 prose is known only from README summaries (the exact term-sheet content and the degrees-of-freedom accounting in 17.7 are not visible).

- `02_mean_variance_optimization` (30 ETFs, train through 2019-12-31, 4% hurdle): average pairwise correlation well below one, yet the condition number is large because of redundant sector/broad-market pairs; max Sharpe concentrates into a handful of funds while EW, IV and ERC hold all of them; min volatility sits at the left tip and can show a negative Sharpe when its return falls below the hurdle; the heuristics sit inside the in-sample frontier. Allowing shorts worsens concentration considerably. FF5 regression of the frozen max-Sharpe test returns describes co-movement over that window only.
- `03_robust_optimization` (11 ETFs, 2015-2023, train through 2019): effective-position ordering is "not the one the previous notebook would suggest" — min variance and min CDaR hold fewer effective positions than max Sharpe; risk parity spreads capital most evenly, HRP next, EW assumes the answer. Risk-parity contributions hold within a small tolerance in sample against the Ledoit-Wolf covariance; diagonal oracle gives 1/3 each to machine precision. Cost-aware risk parity prints the annualized return given up to 5 bp + 5 bp.
- `04_kelly_criterion`: coin-toss paths spread widely at full and above-full Kelly; in the multi-asset test window the full-Kelly path reaches a `min_wealth_multiple` printed in scientific notation (effectively ruined) while still showing a respectable Sharpe — the reason tables are ordered by gross exposure. The rolling 5-year SPY Kelly fraction swings across an unactionable range. Every path stays strictly above zero on observed data (arithmetic, not survival).
- `05_factor_allocation_evidence`: value, momentum, carry and defensive show positive gross premia back to 1926, most predating publication; value-momentum correlation is negative in every VME sleeve and in US equities; raising the bar from t=2 to t=3 disqualifies fewer of these (most-replicated) factors than of the literature at large; SMB is weaker after its 1981 publication on an ex-post split; crisis table: market negative by definition, momentum mixed (2009 momentum crash), value generally negative (2020 especially), QMJ varies, TSMOM positive in most but not all windows (1985 start leaves earlier crises unmeasured).
- `06_hierarchical_risk_parity`: static HRP on 15 ETFs puts a majority weight in a single bond fund via two successive inverse-variance splits (bonds and gold dominate). The walk-forward ranking across EW / IV / HRP / LW min-variance is a fact about this run (5 assets, 252 observations, well-conditioned), not evidence about the under-determined regime; engine replay preserves the ranking and risk shape with costs applied.
- `07_conformal_position_sizing`: three sizing rules on one selection differ by weights alone; the ETF (h=21) and CME (h=5) studies differ in signal strength (HAC t on mean IC), which governs whether inverse-width sizing can matter.
- `08_library_comparison`: matched objectives across PyPortfolioOpt, Riskfolio-Lib and skfolio agree to solver tolerance; routing one allocation through the engine moves its Sharpe (timing + fills + costs jointly); the turnover penalty cuts trading distance; the L2 penalty widens the number of positions held.
- `09_allocator_comparison`: four allocators on one signal give four results; most Sharpe gaps sit inside the IID SE; allocators that estimate a covariance (MVO, HRP) pay for the estimate; the pre-selected allocator and holdout numbers are runtime outputs.
- `11_dl_portfolio_allocation`: the LSTM shows persistent concentration in a small ETF subset (structural bias, not tactical tilts) and emergent turnover with no cost penalty; whether it beat EW/IV is printed at runtime.
- `12_vlstm_portfolio`: figure titles record "the training curve keeps rising after validation Sharpe stops" and "position changes cluster low with occasional rebalancing spikes" (inference from titles); an advantage that does not survive both heuristics net of cost means the complexity is not earning its keep; a near-uniform VSN distribution means selection added little.
- `13_deepm_regime_robust`: the SoftMin-vs-no-SoftMin gap is evidence for this seeded run only; the calm-crisis gap can move either way without contradicting the objective; per-component ablations (FiLM, V-VSN, Directed Delay) are not run; macro-prior density is printed as a share of 29^2 = 841 links; drawdown panel feeds book figure 17.9.
- Common caveats: one split, one universe, current-vintage ETFs, no confidence intervals, frozen weights with daily rebalance (02/03/04/08); fixed ex-post universes (06: 15, 08/09: lists, 11-13: 29).

## Related references

- `chapters/16_strategy_simulation.md` — the backtest report these metrics extend; registered run-log artifacts; turnover cost modelling.
- `chapters/18_transaction_costs.md` — prices the turnover every allocator table reports gross; capacity and cost-grid affordability.
- `chapters/19_risk_management.md` — drawdown control, VaR/CVaR, leverage limits that the Kelly cap and vol targeting hand off to.
- `chapters/20_strategy_synthesis.md` — cross-case-study allocator comparison (`05_portfolio_allocation`).
- `chapters/11_ml_pipeline.md` — conformal prediction basics (`06_conformal_prediction`) behind notebook 07; walk-forward / purged CV.
- `chapters/12_gradient_boosting.md` — the ETF GBM predictions and registry that 06 and 07 select from.
- `chapters/13_dl_time_series.md` — LSTM, TFT building blocks (GRN, VSN) reused by 11-13.
- `chapters/14_latent_factors.md` — factor structure as a covariance regularizer; Fama-French attribution.
- `chapters/07_defining_the_learning_task.md` — forward-return labels and horizons (`fwd_ret_21d`, `fwd_ret_5d`) consumed as allocation inputs.
- `chapters/08_financial_features.md` — momentum, moving-average and volatility features of the 09 Ridge signal and the DL feature panels.
- `chapters/27_systematic_edge.md` — why estimating less and pre-holdout selection protect a thin edge.
- `case_studies/etfs.md` — the universe, registry and backtests every notebook here uses.
- `case_studies/cme_futures.md` — the second conformal-sizing panel (h=5, K=10).
- `libraries/ml4t_diagnostic.md` — `PortfolioAnalysis`, tear sheets, drawdown, VaR, factor analysis, IC inference.
- `libraries/ml4t_backtest.md` — `Strategy`, `TargetWeightExecutor`, `RebalanceConfig`, commission/slippage types, NEXT_BAR.
- `libraries/ml4t_data.md` — `FamaFrenchProvider`, `AQRFactorProvider`, ETF loaders.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `workflow.md`, `companion_repo.md` — cross-cutting indexes this chapter feeds.
- Further reading:
  - Markowitz (1952) Portfolio selection; DeMiguel, Garlappi & Uppal (2009) 1/N; Ledoit & Wolf (2003) shrinkage; Maillard, Roncalli & Teiletche (2008) ERC; Hurst (2010) risk parity.
  - Lopez de Prado (2016) HRP and robust efficient frontier; Raffinot (2016); Antonov et al. (2024) HRP theory; Marti et al. (2021) correlation clustering review; Tan & Zohren (2025) large covariance CV; Jensen et al. (2024) implementable efficient frontier; Wang et al. (2025) ML meets Markowitz.
  - Grinold & Kahn (2000) active management / IR; Lo (2002) statistics of Sharpe ratios; Korn et al. (2022) and Alankar et al. (2023) drawdowns; Joubert et al. (2024) enhanced backtesting; French (2024) long/short sizing; Paleologo (2025).
  - Asness, Moskowitz & Pedersen (2013) Value and Momentum Everywhere; Harvey, Liu & Zhu (2016); Ilmanen et al. (2021); Moskowitz, Ooi & Pedersen (2012) TSMOM; Asness et al. (2017) factor timing; Shu & Mulvey (2025) and Ang & Bekaert (2002) regimes.
  - Zhang, Zohren & Roberts (2020) deep portfolio optimization; Lim et al. (2021) TFT; Liu & Zohren (2023); Wood, Roberts & Zohren (2026) DeePM; Saly-Kaufmann et al. (2026) DL benchmark.

## Glossary

- **Allocator term sheet** — documented specification of an allocator's inputs, windows, constraints and rebalancing (17.2).
- **Markowitz curse** — MVO amplifies small estimation differences into large weight differences through covariance inversion.
- **Condition number kappa(Sigma)** — largest/smallest eigenvalue; bounds error amplification under inversion; grows with redundancy.
- **Efficient frontier** — upper boundary of the feasible risk/return region (Markowitz bullet); a picture of the sample.
- **Max-Sharpe / minimum-variance portfolio** — highest excess return per unit vol at a hurdle / lowest-variance long-only portfolio (independent of mu).
- **Equal risk contribution / risk parity** — weights such that each asset's share of portfolio variance is 1/N.
- **Risk contribution** — w_i (Sigma w)_i / (w^T Sigma w); shares sum to one.
- **Inverse volatility** — w_i ∝ 1/sigma_i; ignores correlations.
- **Ledoit-Wolf shrinkage** — pulling the sample covariance toward a well-conditioned target before inversion.
- **HRP** — hierarchical risk parity: correlation-distance clustering, quasi-diagonalization, recursive bisection with inverse cluster-variance splits; never inverts.
- **Quasi-diagonalization / cluster variance** — reordering the covariance by dendrogram leaf order / w'Σw of a block at inverse-variance weights.
- **CDaR** — conditional drawdown at risk: average of the worst drawdowns on a path.
- **Effective positions / HHI** — 1/sum w_i^2 (inverse Herfindahl) / sum w_i^2 vs the 1/N reference.
- **Gross / net exposure** — sum of absolute positions (its reciprocal is the uniform adverse move that ends the account) / longs minus shorts.
- **Kelly fraction** — bet size maximizing expected log terminal wealth; edge/odds (binary), (mu - r_f)/sigma^2 (continuous), Sigma^-1 mu (multi-asset). **Fractional / half-Kelly** — fixed multiple of it. **min_wealth_multiple** — lowest point of the compounded wealth path relative to start.
- **Volatility targeting** — scaling each position by sigma* / (sqrt(252) * sigma_hat * N) so each contributes equal ex-ante risk.
- **Drawdown / underwater curve** — loss from a cumulative peak to the trough before recovery / that loss on every date.
- **Calmar / Sortino / Omega / tail ratio** — return over max drawdown / excess return over downside deviation / probability-weighted gains over losses / largest gains vs worst losses.
- **Alpha / beta / tracking error / information ratio** — annualized intercept / slope on the benchmark / vol of active returns / mean active return over TE.
- **Up / down capture** — strategy's mean return in benchmark-up (down) months over the benchmark's.
- **VaR / CVaR** — empirical loss quantile / mean loss beyond it.
- **Stability** — R^2 of a linear fit to the cumulative value curve.
- **Execution bridge** — three-row comparison: vectorized, zero-cost engine, cost-aware engine. **NEXT_BAR** — engine fill at the bar after the decision.
- **Dirichlet distribution** — non-negative vectors summing to one; concentration parameter controls spread.
- **Crisis alpha / pre-discovery evidence / Harvey discovery threshold** — positive return in an equity drawdown / factor returns predating publication (not pre-registered) / t >= 3 on the mean-return t-stat under multiple testing.
- **Newey-West / HAC** — heteroskedasticity-and-autocorrelation-consistent standard errors.
- **Long-short / dollar-neutral / beta-neutral** — holds both sides / equal-size legs / market exposures cancel.
- **Mondrian split conformal / horizon embargo / inverse-width weighting** — per-entity conformal calibration / excluding residuals whose labels were not observable by fold start / w_i ∝ 1/Delta_i.
- **One-way turnover** — ½ Σ|w_t - w_{t-1}|.
- **cash_precheck** — hold cash at the hurdle when no expected excess return is positive.
- **Walk-forward / CPCV** — sequential and combinatorial purged cross-validation (skfolio built-ins).
- **Paired block bootstrap** — resample date blocks and recompute both statistics on the same blocks to estimate the uncertainty of their difference.
- **Differentiable Sharpe loss / pooled Sharpe** — negative annualized Sharpe of portfolio returns optimized by backprop / mean/std over all decision endpoints flattened.
- **GRN / GLU / VSN / VLSTM** — gated residual network (ELU MLP, GLU gate, residual, LayerNorm) / a * sigmoid(b) / softmax over per-feature GRN embeddings / VSN + shared per-asset LSTM + GRN + tanh head.
- **Endpoint / predecessor endpoint / two-pass gradient** — last date of a feature window (the scored decision) / prior decision prepended for continuous turnover / no-grad global gradient then chunked exact backward.
- **Cost grid** — post-hoc Sharpe recomputation at several cost levels on fixed weights.
- **DeePM / SoftMin objective / FiLM / V-VSN / Directed Delay / macro graph prior / static metadata** — regime-robust deep portfolio policy / -SR_pool - lambda * SoftMin_tau({SR_b}) over rolling training windows / feature-wise linear modulation by context / DeePM's variable selection / cross-sectional attention with one-step lag / binary adjacency as attention mask / per-asset ids, group ids, cost bps.
- **Regime split / Gap** — test days above/below the median SPY 21-day annualized vol / Calm Sharpe minus Crisis Sharpe (a holdout diagnostic).
- **Ablation** — same architecture, data, seed and initial weights with one mechanism switched off.
