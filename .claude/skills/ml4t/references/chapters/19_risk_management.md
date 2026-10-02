# Chapter 19: Risk Management

> Risk management is part of system design, not post-hoc reporting: a backtest winner is not deployable until its limits, escalation rules, drift monitoring and governance artifacts are defined in advance, auditable and point-in-time safe (README 19.1, 19.8). The chapter equips you to measure the tail (VaR/CVaR four ways plus a Kupiec coverage backtest), measure path risk (drawdown depth/duration/recovery), decompose exposures (CAPM/FF3/FF5 with HAC, lagged attribution, trade-level SHAP), stress a book (crisis replay, simultaneous shock vectors, kurtosis-matched Student-t Monte Carlo), and then build adaptive controls (stops, trailing/ATR exits, sizing, drift monitors, learned hedges, `ml4t.backtest.risk` rules and kill switches) that use only information available at decision time. Its recurring position: a risk number is a frequency claim that must be backtested; every conditional or adaptive input must be lagged; combining rules only tightens; costs must be priced before any exit or sizing claim; and a limit class enforces a threshold but never decides one.

## When to use this reference

- Computing or reporting VaR / CVaR (historical, parametric, Cornish-Fisher, Student-t Monte Carlo) and validating the coverage with a Kupiec backtest.
- Building regime-conditional tail estimates, rolling tail charts, or a diversification-benefit split by volatility state without leaking the regime label.
- Choosing or scoring volatility forecasts (rolling variance, EWMA, GARCH) for position sizing — picking QLIKE over MSE.
- Measuring drawdown depth, duration and time-to-recovery, or comparing strategies with similar CVaR but different path risk.
- Designing, testing or comparing exit rules (fixed stop/target, trailing, ATR, ML-driven, hybrid) and deciding same-bar ordering.
- Sizing positions by stop distance (fixed fractional), inverse volatility, fractional Kelly, scale-out tranches or conviction multipliers; calibrating stops from MAE/MFE excursions.
- Decomposing a portfolio into market/size/value/profitability/investment exposures, running rolling betas, HAC-robust attribution, residual clustering, or covariance conditioning.
- Explaining why the worst trades of an ML strategy failed (TreeSHAP forensics via `TradeShapAnalyzer`).
- Running stress tests: historical crisis replay across several windows, hand-authored simultaneous shock scenarios, Student-t Monte Carlo, weight sensitivity, regime-conditional statistics.
- Setting up production drift monitoring (PSI, Wasserstein, domain classifier) with alert thresholds and a retraining rule that requires performance degradation.
- Building a two-model entry/exit architecture or a deep-hedging policy with a differentiable CVaR objective.
- Wiring `ml4t.backtest.risk` position rules (`StopLoss`, `TrailingStop`, `RuleChain`, ...) and portfolio limits / kill switches (`MaxDrawdownLimit`, `DailyLossLimit`, exposure limits) into an Engine strategy, or sweeping rule widths on a calibration window and evaluating once.

## Core ideas (the why)

- **A risk number is a claim about frequency.** A VaR never backtested is an assertion. Count exceptions and test whether the count matches the stated coverage (Kupiec). Kupiec only counts; clustered exceptions can pass it.
- **VaR is a quantile and nothing more.** CVaR (expected shortfall) averages the tail beyond it, is subadditive (coherent), and the VaR/CVaR gap widens with confidence exactly where a limit binds. Report severity as well as frequency.
- **Correcting a normal for skew/kurtosis is not free.** Cornish-Fisher is a truncated polynomial, unstable with large excess kurtosis; it needs a coverage backtest, never automatic preference. Prefer a kurtosis-matched Student-t when excess kurtosis is large.
- **Any conditional risk number must use a state label known before the return it labels** (prior-close volatility, expanding point-in-time thresholds, minimum history). A label that peeks makes the control unusable.
- **Average diversification benefit is dominated by calm periods.** Correlations rise in crises; check the benefit survives in the high-volatility state. A VaR benefit is an observation; a CVaR benefit is a guarantee (subadditivity).
- **Match the loss function to the decision's asymmetry.** Under-forecasting volatility leaves a position too large into a bad day, so QLIKE beats symmetric MSE for risk work.
- **Drawdowns measure path risk, not point loss.** Identical CVaR can produce very different drawdown experiences; strategies fail by losing money in ways allocators and operators cannot tolerate.
- **Test exit rules on entries with no edge** (random seeded entries); signal-driven entries flatter or damn the rule. Separate the deterministic part (exit-type composition given the path) from the sampled part (mean return with large SE).
- **Scale stops to what the instrument does (ATR units)** so a rule is portable; a fixed-percentage stop is a different rule in a calm month than a violent one.
- **Declare same-bar ordering.** Target-first manufactures winners; the book's convention is stop-first (conservative reading of a bar whose internal sequence is unknown). Order inside a `RuleChain` is policy: first non-HOLD wins.
- **Combining rules inherits the tightest one.** Overlays can only shorten trades; read exit-type composition, treat return change as noise.
- **Size from the loss you will accept, not the gain you hope for**: stop distance and risk budget solve for shares; round to whole shares before reporting risk. Lag every input to an adaptive control (vol window ends t-1).
- **Name a heuristic by what it does.** Inverse-volatility ignores correlations and is not risk parity.
- **Separation is not profit.** MAE/MFE breach-rate discrimination says nothing about PnL net of costs. Choose thresholds on one window, freeze, test on a later one.
- **Full-sample contemporaneous betas leak future betas into the past.** Deployable attribution uses lagged betas (`lag=1`). With correlated factors attribution is not unique (rotation ambiguity): declare an economic basis. Residual correlation is evidence of omitted structure, not proof of a factor.
- **SHAP attributes a prediction, not an outcome.** Failure clusters are hypotheses checked outside the clustering; join explanations to the decision row by stable trade ID. Reproducibility is a property of the backend (single CPU thread), not the seed.
- **Replay more than one crisis.** 2008 (growth shock) and 2022 (rate shock) reversed which assets protected. Write scenarios as simultaneous shocks to every asset; shocking one holding keeps the diversification a crisis removes. One event never sizes a limit.
- **A Monte Carlo must match tails to data and state what it omits** (serial dependence, vol clustering, regime shifts, changing dependence). It maps moments into a tail range, not a forecast; keep the deepest quantile in proportion to the paths behind it.
- **Drift is a comparison, so it inherits its baseline.** A calm reference makes ordinary variation look like drift; a reference containing a crisis absorbs the next one. Covariate drift (P(X)) != failure (P(Y|X)); none of PSI/Wasserstein/domain AUC see concept drift. Monitor inputs AND outcomes; retraining on every alert is a loop, not a control.
- **Univariate and multivariate checks see different things.** Features can each stay in range while the joint distribution moves; a domain classifier sees that, PSI cannot. Use blocked folds for the domain classifier, else it detects autocorrelation, not drift.
- **Monitoring cadence must match market speed.** ETF drift unfolds over quarters, crypto in days; row frequency is granularity, not cadence.
- **A feature built from another model must be produced out of sample** (purged expanding-window OOF), or the downstream model learns an accuracy production never supplies. A better AUC is not a better strategy; report trade-path metrics too. Check what a combined OR rule actually fires on.
- **A risk measure must be differentiable to be an objective.** Rockafellar-Uryasev replaces the CVaR quantile with a learned threshold w; report the empirical measure, not the objective's value. Charge costs over the whole lifecycle including liquidation. Training and benchmarking on the same generator (GBM) is the least informative comparison.
- **Read mean, dispersion and tail together.** A tail-minimizing hedge may accept more variance elsewhere; reporting one statistic alone conceals the trade. Emergent behaviour (trading less near the delta benchmark) is an association until identified, not a no-trade band.
- **Position rules and portfolio limits are different controls; neither substitutes.** Limit gross and net exposure separately. A limit class enforces a threshold; the value comes from a risk mandate, and escalation/approval/reinstatement live outside the library (§19.8).
- **Calibrate before evaluating; evaluate once; keep the operational contract explicit** (completed closes, NEXT_BAR execution, trading-bar schedules, fail-closed panels). Write only artifacts a named consumer reads.

## Method recipes (the how)

README section map (19.x -> notebooks): 19.1 turning a backtest winner into a tradable system -> 01, 06; 19.2 practical risk taxonomy -> 04, 06; 19.3 VaR/CVaR -> 01; 19.4 drawdowns, path risk, time-to-recovery -> 01, 02, 03, 10; 19.5 factor, sector and macro exposures -> 04, 05; 19.6 stress testing and scenarios -> 06; 19.7 adaptive controls without leakage (vol targeting, exposure caps, turnover tightening, position-level exits) -> 02, 07, 08, 09, 11; 19.8 kill switches and governance -> 07, 10. Cross-case-study risk-overlay comparison: ch20 `20_strategy_synthesis/07_regime_risk`.

**19.2 risk taxonomy**: market, factor, leverage, concentration, liquidity, model and operational risk, each mapped to an observable proxy and a control. The notes carry only this summary; the per-category proxy/control detail sits in the book text, with 04 (factor) and 06 (market) as the worked notebooks.

Reading order ("Next" links): 01 -> 06; 02 -> 03 -> 04 -> 05 -> 06 -> 07. Progression (nb11): measurement (01) -> exits and sizing (02-03) -> exposures, SHAP, stress (04-06) -> monitoring (07) -> learned controls (08-09) -> library controls (10) -> calibrated controls (11). Output contracts via `get_output_dir(19, name)`: `"var_cvar"` (two tables behind the printed figures; file names not stated), `"exit_strategies"`, `"position_sizing_mae_mfe"` (Figs 19.4, 19.8), `"ml_exit_signals"` (Fig 19.5 quintile table), `"risk_sweep"` (Figs 19.6-19.7 panels, one-time evaluation table); the other notebooks write figures or summaries with no stated consumer.

### VaR and CVaR four ways, with coverage backtest (`19_risk_management/01_var_cvar`)

Config: `SYMBOL="SPY"`, `PORTFOLIO_SYMBOLS=["SPY","AGG","GLD","EFA","EEM"]`, 2006-01-01 to 2024-12-31 (deliberately pre-2008), `CONFIDENCE_LEVELS=[0.90,0.95,0.99]`, `BACKTEST_CONFIDENCE=0.95`, `ROLLING_WINDOW=252`, `N_MC_SIMULATIONS=10_000`, `SEED=42`.

| Method | VaR_a | CVaR_a | Use when |
|---|---|---|---|
| Historical | `-q_{1-a}(r)` | `E[L \| L >= VaR_a]` | The method 01 backtests (`backtest_var(..., method="historical")`); cannot exceed worst loss in window — include a crisis |
| Parametric (Gaussian) | `-mu + sigma * Phi^{-1}(1-a)` | `-mu + sigma * phi(Phi^{-1}(1-a)) / a` | Gaussian baseline; Student-t CVaR sits above it at 99% on SPY (01) |
| Cornish-Fisher | Gaussian quantile adjusted for sample skew/kurtosis | — | Only if its coverage backtest passes; unstable with large kurtosis |
| Monte Carlo Student-t | `monte_carlo_var(returns, confidence=0.95, n_simulations=N_MC_SIMULATIONS, horizon=1, seed=SEED)` samples a fitted t | closed form under fitted t | Preferred with large excess kurtosis |
| Cantelli bound | `P(X - mu >= k sigma) <= 1/(1+k^2)` | — | Conservative stress benchmark when distributional assumptions are suspect; needs only finite mean/variance |

Steps:
1. Run `analyze_distribution` and `analyze_tails` (`ml4t.diagnostic.evaluation.distribution`): moments, normality tests, Hill estimator, QQ. Use them to pick parametric / CF / Student-t.
2. Compute VaR and CVaR at 90/95/99% side by side; report the CVaR/VaR ratio.
3. Backtest: `backtest_var(returns, window=252, confidence=0.95, method="historical") -> dict` rolls VaR against one-step losses; `kupiec_test(exceptions, expected_rate) -> (LR stat, p)` under a binomial null. Exception ratio = realized rate / target; want it near 1.
4. Inspect exception timing for clustering (Kupiec ignores it); (inference) add an independence test.
5. Never scale 1-day to 10-day by sqrt(10): it assumes independence that exception clustering contradicts. Estimate the horizon directly.

### Rolling, regime-conditional tail and diversification benefit (`01_var_cvar`)

- Rolling VaR/CVaR: trailing 252 daily unannualized returns (`ROLLING_WINDOW=252`); lower panel plots CVaR/VaR ratio.
- Regime-conditional tail: label each day by **prior-close** 63-day volatility (`REGIME_WINDOW=63`) against **expanding point-in-time tercile thresholds** built from lagged observations only; require `REGIME_MIN_HISTORY=252` days before the first split.
- Diversification: `portfolio_tail_risk(returns_matrix, weights, confidence=0.95)` compares empirical portfolio VaR/CVaR with weighted stand-alone estimates (equal weights, five ETFs); then split the benefit by the same lagged volatility state.
- Drawdown: `drawdown_path(returns) -> (wealth, running peak, drawdown %)`; plot with zero at the top.

### Volatility-forecast scoring (`01_var_cvar`; deeper GARCH in ch9)

- Candidates: 21-day rolling variance (`ROLLING_VAR_WINDOW=21` is this rolling-*variance* window, `rolling(21).var().shift(1)`, not a rolling-VaR window); EWMA `h_t = 0.94*h_{t-1} + 0.06*r_{t-1}^2` (`EWMA_LAMBDA=0.94`, RiskMetrics); GARCH(1,1) fitted strictly before `FORECAST_EVALUATION_START="2020-01-02"`, evaluated OOS from that date.
- Proxy = squared daily return (noisy; small gaps are not decisive).
- `qlike_loss(forecast_var, proxy_var)` = `mean(log(h_hat) + sigma^2/h_hat)`; `variance_mse(...)` = `mean((h_hat - sigma^2)^2)`. Rank by QLIKE for sizing; QLIKE and MSE can rank the three forecasts differently.

### Exit rules on random entries (`02_exit_strategies`)

Config: SPY OHLCV 2018-01-01 to 2024-01-01 (spans 2020 drawdown), `N_TRADES=100` random uniform entries, `SEED=42`, `LABEL_HORIZON=5`, `ADVERSE_MOVE_THRESHOLD=-0.02`, `TEST_FRACTION=0.30`.

- ATR = `max(H-L, |H-prevC|, |L-prevC|)` so gaps count. RSI = avg up-move / avg down-move on 0-100.
- Fill oracle `long_barrier_fill(open, high, low, stop, take_profit) -> (price, type) | None`: resting orders from entry close; gaps fill at next open; intraday touches fill at the barrier; both touched -> **stop first**.
- `summarize_trades(results)`: mean return, SE = sd/sqrt(n), win rate, count. A gap smaller than a couple of SEs is not a result.

| Rule | Config / simulator | Holding cap | Notes |
|---|---|---|---|
| Fixed | `FixedExitConfig` (loss %, target %); `simulate_fixed_exits(..., max_holding_days=20)` | 20 | Exit types stop-out / target / timeout; histograms on common bins |
| Trailing | `TrailingStopConfig` (initial protection vs trail from high-water mark); `trailing_stop_fill(open, low, stop)`; `simulate_trailing_stops(..., max_holding_days=50)` | 50 | Apply today's resting stop before a new high tightens tomorrow's stop |
| ATR | `ATRStopConfig` (ATR multiples down/up); `simulate_atr_exits(..., atr, ..., max_holding_days=30)` | 30 | Rescales both barriers by current ATR; SD of stop distance shows within-config adaptation |
| ML | nine features (returns over four horizons, realized vol, RSI, distance from two MAs, volume vs recent avg); label = close fell > 2% over next 5 sessions; `ml_exit_triggered(proba_by_ts, idx, exit_threshold, bar_timestamps)`; `simulate_ml_exits(..., exit_threshold=0.5, max_holding_days=30)` | 30 | Chronological split, purge 5 rows before boundary, MDI on expanding folds; probabilities keyed by timestamp and shifted one session (close t -> open t+1); bars without an OOS signal never trigger; the exit threshold trades coverage against signal quality |
| Hybrid | `HybridExitConfig`; `hybrid_bar_fill(open, high, low, stop, tp, ml_exit, stop_type)` evaluates barriers in declared priority; `simulate_one_hybrid_exit` owns event ordering; `simulate_hybrid_exits(..., max_holding_days=30)` | 30 | Inherits the tightest rule |

- Compare only within a section: fixed/trailing/ATR run on all entries, ML/hybrid on test-period entries only.
- `BarrierAnalysis` (`ml4t.diagnostic.evaluation`): TP-hit / SL-hit / timeout rates per signal quintile. Reading: Q5 higher TP rate -> scale up strong signals; Q1 higher SL rate -> cut weak signals; no quintile difference -> improve the signal; high timeout rate -> barriers too tight or signal too slow.
- `MAEMFEAnalyzer` (`ml4t.backtest.analytics`) derives thresholds from `Trade` objects of the representative ATR x2.0 trades; in-sample candidates only.
- Library intro: `PositionRule` protocol base for custom rules; `RuleChain` (ordered), `AllOf`/`AnyOf`; `StopLoss`, `TakeProfit`, `TrailingStop`, `TighteningTrailingStop`, `VolatilityStop`, `VolatilityTrailingStop`, `TimeExit`, `SignalExit` — behaviours in the nb10 recipe, evaluation pattern under Code patterns.

### Position sizing and MAE/MFE stop calibration (`03_position_sizing_mae_mfe`)

Config: `SYMBOLS=["SPY","QQQ","IWM","TLT"]`, 2018-01-01 to 2024-01-01, `CALIBRATION_END="2021-12-31"`, `HOLDING_PERIOD=20`, `MAX_HOLDING_DAYS=30`, entries spaced beyond the longest holding period, 30-bar embargo after the calibration boundary.

| Method | Formula / API | Defaults | Caveat |
|---|---|---|---|
| Fixed fractional | `shares = V*r / \|P_entry - P_stop\|`; `calculate_fixed_fractional_size(portfolio_value, entry_price, stop_price, config) -> dict` with `FixedFractionalConfig(risk_per_trade=0.01, max_position_pct=0.25)` | 1% of equity risked per trade, 25% concentration cap (repo-verified; the notes did not show these defaults) | Integer shares; cap binds for tight stops, risk budget for wide stops; recompute realized stop risk after rounding |
| Inverse volatility | `w~_i = 1/sigma_{i,t-1}`, `w_i = w~_i / sum_j w~_j`; `calculate_inverse_volatility_weights(asset_volatility)` | vol window ends t-1 | Long-only, sums to one, no covariance; not risk parity |
| Kelly | ch17 `04_kelly_criterion` | fractional 1/2 or 1/4 | Estimation error |
| Scale-out | `ScaleOutConfig`; `simulate_scale_out(prices, entry_idx, entry_price, config, stop_loss_pct=0.02, max_holding_days=MAX_HOLDING_DAYS)`; `simulate_entry_set(prices, entries, config)` | 1/3 at first target, 1/3 at second, remainder with stop to break-even after a target fires (single ratchet, not trailing) | Close-based stop checked before targets; three tranches pay 3x fixed costs |
| Conviction | `calculate_conviction_based_size(base_position_pct, conviction_score in [0,1], min_multiplier=0.2, max_multiplier=2.0)` | 0.2x-2.0x | Bounded deterministic map; does not supply or validate the score |
| Conformal | `w_t = w_base * Delta_median / Delta_t` (interval width) | — | Implementation in ch17 `07_conformal_position_sizing`; basics ch11 |

Sizing sequence: stop distance -> risk fraction r -> `shares = V*r/|entry-stop|` -> concentration cap -> integer shares -> recompute realized stop risk.

MAE/MFE protocol:
1. `calculate_mae_mfe(lows, highs, entry_idx, exit_idx, entry_price)` from bars **after** entry (exclude the entry bar); `build_trade_excursions(entries, split, prices, lows, highs, timestamps) -> pl.DataFrame`.
2. `analyze_stop_separation(excursions, stop_distances, split)` = share of eventual losers whose MAE breached d minus share of winners that breached d. Choose d on calibration (to 2021-12-31), 30-bar embargo, freeze, test on future; accept only if future separation is close.
3. `build_excursion_percentiles(prices, horizons, split)`: close-path percentiles by horizon, calibration (solid) vs future (dashed).
4. Treat the result as a prior, not an optimum; then re-run trades with the rule and costs under a frozen PnL objective.

### Factor exposure decomposition (`04_factor_exposure`)

Config: 2010-01-01 to 2024-01-01; `ROLLING_WINDOW=252`; `COVARIANCE_WINDOW=63`; `HAC_LAGS=5` (Newey-West); `ATTRIBUTION_LAG=1`; `RESIDUAL_CORR_THRESHOLD=0.15`; `FF3_COLUMNS=["Mkt-RF","SMB","HML","RF"]`, FF5 adds `"RMW","CMA"`. Data: local canonical Fama-French parquets; SPY, QQQ, IWM, VTV, VUG.

1. Regressions (statsmodels OLS, HAC lags 5 everywhere): CAPM `R_i - R_f = alpha + beta(R_m - R_f) + eps` (`capm_regression`); FF3 adds SMB, HML (`ff3_regression`); FF5 adds RMW, CMA (`ff5_regression`).
2. Rolling betas: `rolling_factor_regression(returns, factors, window=252)`, labelled at the next session.
3. Attribution: `factor_attribution(returns, factors)` uses full-sample contemporaneous betas (teaching only). Deployable: `compute_return_attribution(lag=1)`.
4. Uncertainty: delta-method interval from HAC SE of beta and HAC SE of the factor mean (`hac_mean_standard_error(values, maxlags=5)`, intercept-only HAC regression); it omits their covariance — block bootstrap when decision-relevant.
5. Residual clustering: flag pairs with |residual corr| > 0.15 (a screen, not a test); find an economic driver and test OOS.
6. Conditioning: `rolling_covariance_condition(return_frame, window=63)` compares condition number of trailing sample covariance vs Ledoit-Wolf on the same prior 63 sessions; shrink before inversion when it persistently lowers the condition number. Mahalanobis spikes mix tails and estimation error.
7. Rotation: a fixed orthogonal rotation leaves fitted values and total factor contribution identical but relabels components (centered PCA would lose factor means). Declare a basis or hierarchical rule.

Library (`ml4t.diagnostic.evaluation`): `FactorData.from_dataframe()` / `FactorData.from_fama_french("ff3")`; `compute_factor_model(hac=True)`; `compute_rolling_exposures()` (adds sign consistency, max step change); `compute_return_attribution(lag=1)`; `compute_risk_attribution()` (Euler variance decomposition with Ledoit-Wolf). Library defaults are Andrews-rule bandwidth ("well above 5" here) and a 63-session rolling window — pass `HAC_LAGS` and `ROLLING_WINDOW` explicitly to reproduce the manual results.

### Trade-level SHAP forensics (`05_trade_shap_diagnostics`)

Config: 2006-01-01 to 2025-12-31; `N_ESTIMATORS=100`; `MAX_BIN=255`; `TRAIN_FRACTION=0.70`; `WORST_N=20`; `MIN_EXPECTED_RETURN_BPS=5` (0.0005); `STRESS_VIX_LEVEL=25`; `MOMENTUM_WINDOW=20`; `VOLATILITY_WINDOW=20`; `VOLUME_WINDOW=60`; `SEED=42`; `LGB_DEVICE="cpu"`; `FEATURE_COLS=["momentum","volatility","volume_zscore","regime","yield_slope"]`; `LABEL_HORIZON=1`; `QUANTITY=100`.

1. Panel: single-asset SPY (`load_etfs`) + fixed FRED snapshot (`load_macro`: 10Y-2Y slope, VIX); predictors lagged one session; label = close-to-close return t to next session; purge one row before the boundary; LightGBM regressor, one CPU thread, device read back from the booster (unavailable backend raises at `fit()`).
2. Trades: market-on-close at t when forecast > 5 bps; exit next observed close; exact timestamps.
3. `TradeAnalysis` extracts worst trades; TreeSHAP with additivity check; SHAP vectors indexed by exit timestamp (`TradeRecord.timestamp` is the exit field) — join by stable trade ID and use entry-decision features.
4. `TradeShapAnalyzer.explain_worst_trades()`: align timestamps -> extract SHAP -> hierarchical clustering on normalized SHAP vectors -> characterize (separation scores, FDR-corrected feature tests as annotations, fallback to top-ranked features) -> generate hypotheses. Plot signed SHAP distributions (bps) ordered by mean |SHAP|.
5. Check the forecast gap: worst trades chosen by realized loss mix genuine failures with unlucky correct calls.

### Stress testing (`06_stress_testing`)

Config: `SEED=42`; `N_SIMULATIONS=10_000`; `HORIZON_DAYS=20`; `CONFIDENCE_LEVELS=(0.95,0.99,0.999)`; `REGIME_LOOKBACK=60`; `BEAR_TREND_THRESHOLD=-0.05`; `BULL_TREND_THRESHOLD=0.10`; `HIGH_VOL_MULTIPLE=1.3`. Data: `data.load_etfs()` SPY, EFA, EEM, AGG, TLT, GLD, VNQ, 2007-01-01 to 2024-01-01.

Portfolios: 60/40 {SPY .60, AGG .40}; All Weather {SPY .30, TLT .40, GLD .15, VNQ .15}; Aggressive Equity {SPY .60, EFA .20, EEM .20}; Defensive {AGG .40, TLT .30, GLD .20, SPY .10}.

**Historical replay** (close endpoints; returns compounded over (start close, end close]):

| Window | Dates |
|---|---|
| 2008 GFC | 2008-09-02 .. 2009-03-09 |
| 2010 Flash Crash | 2010-05-05 .. 2010-05-07 |
| 2011 Debt Downgrade | 2011-08-03 .. 2011-08-08 |
| 2015 China Devaluation | 2015-08-17 .. 2015-08-25 |
| 2018 Q4 | 2018-10-01 .. 2018-12-24 |
| 2020 COVID | 2020-02-19 .. 2020-03-23 |
| 2022 Tightening | 2022-01-03 .. 2022-10-12 |

`max_drawdown(returns_array)` (most negative point vs running peak, the single definition used); `aggregate_portfolio_returns` (complete returns, weights sum to one, constant daily rebalance before costs); `slice_stress_window` validates endpoints are observed sessions; `analyze_stress_period`; `compare_portfolios_stress` (one row per portfolio-window).

**Hypothetical scenarios** (`StressScenario(name, shocks={sym: pct}, description)`; `apply_scenario` = weighted sum of simultaneous shocks):

| Scenario | SPY | EFA | EEM | VNQ | AGG | TLT | GLD |
|---|---|---|---|---|---|---|---|
| Equity Crash (-30%) | -.30 | -.35 | -.40 | -.25 | +.02 | +.10 | +.05 |
| Rising Rates (+200bps) | -.10 | -.08 | -.15 | -.20 | -.10 | -.25 | (not visible) |
| Stagflation | -.20 | -.25 | -.30 | -.15 | -.05 | -.10 | +.20 |
| Deflation Crisis | -.25 | -.30 | -.35 | -.30 | +.05 | +.20 | -.10 |
| EM Crisis | -.10 | -.15 | -.40 | -.05 | +.02 | +.05 | (not visible) |

**Student-t Monte Carlo** on the portfolio-return series: fit full-sample mean, variance, excess kurtosis kappa; `nu = 6/kappa + 4`; scale `s = sigma*sqrt((nu-2)/nu)` so `Var = s^2 nu/(nu-2)` matches; iid daily draws, compounded; raise if a path implies simple return below -100%. `student_t_parameters`, `simulate_student_t_paths(mu, sigma, excess_kurtosis, n_simulations, horizon_days, seed)`, `summarize_simulated_returns` (VaR = lower quantile of cumulative return; CVaR = mean of paths at or below the first level, 0.95), `monte_carlo_stress_test(returns, weights, n_simulations=10_000, horizon_days=20, ...)`. State omitted mechanisms; report the 99.9% quantile as an order of magnitude (about 10 of 10,000 paths).

**Sensitivity**: `sensitivity_analysis(returns, base_weights, asset_to_vary, weight_range, adjust_asset)` varies equity weight in 60/40; return, vol, risk-adjusted, drawdown on separate panels.

**Regime classifier** `classify_regime(returns, lookback=60)`: trailing vol = std of last 60 SPY returns * sqrt(252) read at t-1; trend = mean(returns t-60..t-1) * 252; vol_reference = expanding median of prior trailing vols (needs >= 60). Bear if trend < -0.05; else Bull if trend > 0.10 and vol < median; else High Vol if vol > 1.3 * median; else Calm. `analyze_by_regime` reports one-day conditional metrics only (no max drawdown on concatenated days).

Summary report per allocation: worst historical window, worst scenario, MC tail metrics.

### Drift detection and the production monitor (`07_drift_detection`)

Params: `SEED=42`, `PSI_WARNING_THRESHOLD=0.10`, `PSI_CRITICAL_THRESHOLD=0.25`, `DOMAIN_AUC_WARNING_THRESHOLD=0.60`, `DOMAIN_AUC_CRITICAL_THRESHOLD=0.70`, `DOMAIN_CV_TIME_BLOCKS=20`, `LGB_DEVICE="cpu"`. All four thresholds are rules of thumb; calibrate against a period with known drift before they page anyone. Data: ETF feature panel (20/60-day momentum, 20-day return vol, 14-day RSI, 20-day volume ratio; calm 2017 vs stressed 2020; Q4-2019 baseline vs four quarters of 2020) and crypto perps + premium panel (hourly; hand-picked month-long episodes). Measures covariate drift only.

| Method | Formula / mechanics | Alert bands | Strength / weakness |
|---|---|---|---|
| PSI | `sum_i (p_i^actual - p_i^expected) * ln(p_i^actual / p_i^expected)` over quantile bins fixed on the reference, `n_bins=10` | < 0.10 stable; 0.10-0.25 warning; >= 0.25 critical (left-inclusive) | Regulatory/quick checks; sensitive to binning, ignores ordering; read the bin breakdown (`plot_psi_breakdown(result: PSIResult, feature_name)`): a bin that empties or fills contributes far more than one that shifts slightly, and the same total arises from a uniform shift and from one bin emptying |
| Wasserstein / EMD | `W_p(P,Q) = (inf_gamma integral \|\|x-y\|\|^p dgamma)^(1/p)`; bin-free, order-aware | descriptive: half the reference std; no permutation p-value (autocorrelated rows) | Continuous features, interpretable; one feature at a time (`plot_wasserstein_comparison(reference, current, result: WassersteinResult, feature_name)`) |
| Domain classifier | label reference=0 / current=1, LightGBM one thread, out-of-fold AUC over contiguous relative-time blocks (`relative_time_blocks` assigns 20 blocks as groups; `StratifiedGroupKFold(n_splits=5, shuffle=False)` turns them into five blocked folds — repo-verified); all-row fit only ranks features | 0.5 none; >= 0.60 warning; >= 0.70 critical | Multivariate drift and interactions; needs more data, less interpretable; `domain_classifier(reference, current, features, warning_threshold=0.60, critical_threshold=0.70) -> DomainClassifierResult` |

Drift taxonomy: covariate P(X) (check performance, then recalibrate/retrain); concept P(Y|X) (new architecture/features; needs labels); prior P(Y) (regime-aware models).

Unified: `checked_drift_analysis(reference, current, features, psi_warning_threshold=0.10, psi_critical_threshold=0.25, domain_auc_warning_threshold=0.60, domain_auc_critical_threshold=0.70) -> DriftSummaryResult` wraps library `analyze_drift`; requires both univariate methods to complete and both to flag a feature; multivariate alert separate; fails rather than returning a degraded partial result. `finalize_drift_summary(summary, domain)` attaches the domain result.

Alert policy: `policy_alert_level(value, warning_threshold, critical_threshold)` (left-inclusive, shared by PSI and AUC); `combined_alert_level(...)` returns the more severe; `alert_count_condition(statuses, critical_required=2, warning_required=3)` counts anywhere in the lookback (any-N, not consecutive).

Production loop: `DriftMonitorConfig` (configuration separate from execution); `ProductionDriftMonitor(reference_df, config)` with `check_drift(current_df, timestamp) -> dict`, `get_dashboard_data() -> pd.DataFrame`, `should_retrain(performance_degraded: bool, lookback: int = 5) -> (bool, str)`; returns False until `len(drift_history) >= lookback`; retrain only if performance degraded AND (>= 2 critical or >= 3 warning in the last `lookback` checks). Feature helpers: `etf_feature_panel(prices)` (per-symbol rolling features, only current and prior observations), `etf_window(start, end, cols=None)`, `crypto_feature_panel()`, `crypto_window(start, end)` (complete rows only).

### Two-model ML exit signals (`08_ml_exit_signals`; Figure 19.5)

Params: `SEED=42`, `LGB_DEVICE="cpu"`, `FORWARD_HOURS=24`, `N_OOF_FOLDS=5`, `N_IMPORTANCE_REPEATS=5`, `TEST_FRACTION=0.20`, `ENTRY_QUANTILE=0.95`, `ENTRY_PROBABILITY_THRESHOLD=0.5`, `PROBABILITY_BIN_WIDTH=0.02`, `ENTRY_CONFIDENCE_DROP=0.30`; BTCUSDT, ETHUSDT, SOLUSDT hourly perps from 2023-01-01.

1. `create_features(df, forward_hours=24)`: momentum, volatility, mean-reversion, volume in one polars frame shared by both models; `FEATURE_COLS_ENHANCED = FEATURE_COLS + ["entry_prediction"]`.
2. `add_labels(frame, entry_threshold)`: entry = forward 24h return above the 95th-percentile threshold estimated on **training rows only** (rare event by construction); exit = forward 24h return < 0.
3. Chronological split across all symbols, last 20% test, purge `FORWARD_HOURS` at the boundary.
4. Entry model `make_classifier(seed)` (pinned LightGBM); `predict_positive_probability(model, features)` uses the booster directly.
5. OOF stacking feature: `chronological_oof_entry_probabilities(frame, feature_cols, n_folds=5, label_horizon_hours=24, seed)` — expanding-window folds, fit rows purged by the label horizon before each contiguous validation block (`chronological_fold_rows`), fold-local rare-event cutoff (`fold_entry_labels`); test probabilities from the final entry model.
6. Basic exit model on exactly the OOF-covered rows; enhanced exit model adds `entry_prediction`.
7. Exit rules: (a) exit model says exit; (b) entry probability < 0.30; (c) either. Inspect per-clause fire rates before combining.
8. `simulate_strategy(symbols, timestamps, returns, entry_signals, exit_signals, max_hold=24)`, `build_trade`, `summarize_trades`: entry at close t earns t->t+1; exit at close u does not earn u->u+1; k-bar hold = k one-step returns; per-symbol paths; open positions closed at last observed close; compounded simple returns; no annualization of unequal horizons; explicit row index carried through regrouping.
9. `repeated_gain_importance(features, labels, feature_names, seed, repeats=5)`; quintiles of test probability -> exhaustive outcome shares (descriptive cuts).

### Deep hedging with a differentiable CVaR (`09_deep_hedging`; Buehler et al. 2019; Docker `ml4t-gpu`)

Setup: sell a European call, receive premium p0, hedge at t0..t_{n-1}. `PL_T = p0 - Z + sum_k delta_k (S_{k+1} - S_k) - C_T(delta)`, `Z = max(S_T - K, 0)`; `C_T` charges initial trade, every rebalance and liquidation of the final position at S_T (cash-settled call).

Params: `N_PATHS=50_000`, `N_STEPS=30`, `S0=100`, `STRIKE=100`, `MU=0.05`, `SIGMA=0.20`, `R=0.0`, `DT=1/252`, `COST_RATE=0.001` (10 bp), `CVAR_TAIL_PROBABILITY=0.05`, `HIDDEN_SIZE=32`, `N_EPOCHS=100`, `LR=5e-3`, `BATCH_SIZE=2048`, `SEED=42`, `SWEEP_SEED=7_042`, `COST_SWEEP_EPOCHS=10`, `COST_SWEEP_MAX_TRAIN_PATHS=10_000`, `TRAIN_FRACTION=0.70`, `VALIDATION_FRACTION=0.15` (test 0.15), `SIGNIFICANT_TRADE_THRESHOLD=0.01`.

1. Paths: `simulate_gbm_paths(s0, mu, sigma, n_paths, n_steps, dt, seed)`; benchmark `bs_price_delta(spot, strike, tau, sigma, r=0.0)`.
2. Information set `prepare_info(paths, strike, sigma, n_steps, dt)`: log-moneyness, normalized time-to-maturity, implied-vol proxy (constant under GBM), log return; `N_FEATURES=4`.
3. Policy `DeepHedger(n_steps, n_features, hidden_size=32, max_position=1.5)`: one MLP per timestep (`make_step_network`), previous position fed forward (semi-recurrent, no BPTT beyond the position).
4. Objective `CVaRLoss(tail_probability=0.05)`: `CVaR_{1-q}(L) = min_w [w + E[(L-w)_+]/q]`, `L = -PL_T`, `w` an `nn.Parameter` trained jointly by Adam; gradient clipping.
5. Protocol: `reset_torch_training_state(seed)` before each model; `train_one_epoch(model, criterion, optimizer, tensors, cost_rate, generator)` with an explicit CUDA generator (the runner supplies the cuBLAS/hash settings; strict deterministic mode, fail-closed on CUDA); train from scratch (no cache); validation only monitors the frozen schedule and carries the cost sweep (no epoch selection); test touched once. No purge needed on independent draws (needed on real data).
6. Evaluate with `evaluate_hedger(...)` = empirical sample `compute_cvar(pnl, tail_probability=0.05)`, never the learned w; `pnl_stats`; `compute_hedge_pnl(positions, paths, payoff, premium, cost_rate)`.
7. Cost sweep `train_cost_level(cost_rate)` (10 epochs / 10k paths, directional only) with matched delta trade count via `mean_significant_trades(positions, threshold=0.01)`; diagnostics `empirical_cdf` (magnified worst decile), `conditional_inaction_diagnostic(positions, benchmark_delta, threshold)`.

### `ml4t.backtest.risk` position rules and portfolio limits (`10_ml4t_backtest_risk_demo`)

Position layer (`ml4t.backtest.risk.position`): rule takes `PositionState`, returns `PositionAction` in {HOLD, EXIT_FULL, EXIT_PARTIAL, ADJUST_STOP} with fill price and reason; ATR/signals via `PositionState.context`.

| Rule | Constructor (nb10 demo argument) | Behaviour |
|---|---|---|
| `StopLoss` | `pct=0.05` | Measured from entry; never moves |
| `TakeProfit` | `pct=0.10` | Static target |
| `TimeExit` | `max_bars=20` | Holding cap |
| `TrailingStop` | `pct=0.05` (3% in chains) | From high-water mark through the prior completed bar; ratchets up only |
| `TighteningTrailingStop` | `schedule=[(0.0, 0.05), (0.10, 0.03), (0.20, 0.02)]`: (unrealized return reached, trail pct) pairs (library docstring demo; nb10's values not visible) | Highest reached threshold sets the trail from the water mark, below the lowest threshold the loosest entry applies; trail narrows as profit grows |
| `VolatilityStop` / `VolatilityTrailingStop` | ATR multiples | Current-ATR mode widens with ATR; a live position holds one instance that remembers entry ATR |
| `SignalExit` | signed signal threshold | Signal below negative threshold exits a long |
| `ScaledExit` | `targets=[(0.05, 0.25), (0.10, 0.33), (0.15, 0.50)]`: (return reached, fraction of the CURRENT remaining position) (library docstring demo; nb10's values not visible) | Each threshold fires once as `EXIT_PARTIAL`; stateful: one instance per position or per asset via `set_position_rules(rule, asset=...)`, `reset()` before reuse |

Composition: `RuleChain([...])` first non-HOLD wins in declared order; `AllOf` = AND (e.g. `TakeProfit(pct=0.01)` + `TimeExit(max_bars=5)` = gain >= 1% AND held >= 5 bars); `AnyOf` = OR (alias of `RuleChain`).

Portfolio layer (`PortfolioState` from equity, initial_equity, high_water_mark, positions, daily_pnl):

| Limit | Constructor (nb10 demo argument) | Action |
|---|---|---|
| `MaxDrawdownLimit` | `max_drawdown=0.20, warn_threshold=0.15` | Warn, then liquidate |
| `DailyLossLimit` | `max_daily_loss_pct=0.02` | vs current equity (tightens as the book shrinks); straight to liquidation, no warning band |
| `MaxPositionsLimit` | `max_positions=10` | Halt new trading |
| `GrossExposureLimit` | `max_gross_exposure=1.5` | sum of |positions| |
| `NetExposureLimit` | `max_net_exposure=0.10, min_net_exposure=-0.10` | signed sum |

Library defaults for a bare constructor (repo-verified in `ml4t/backtest/risk/portfolio/limits.py`; not in the notes): `MaxDrawdownLimit(max_drawdown=0.20, warn_threshold=None)`, `DailyLossLimit(max_daily_loss_pct=0.02)`, `MaxPositionsLimit(max_positions=10)`, `GrossExposureLimit(max_gross_exposure=1.0)`, `NetExposureLimit(max_net_exposure=1.0, min_net_exposure=-1.0)`. The 1.5 gross, +-0.10 net and 15% warn values in the table are what notebook 10 passes explicitly; `GrossExposureLimit()` yields 1.0, not 1.5.

Demo contract (`first_trigger(frame, rule_name, rule)`, SPY 2020, `N_BARS=252`): entry after first close of 2020; entry bar never tested; current-bar OHLC observable to active stops; water marks advance only after a bar completes without an exit. Rules compared on the path: StopLoss 5%, TrailingStop 3%, TakeProfit 15%. Priority grid on constructed states: `RuleChain([StopLoss(0.05), TakeProfit(0.10), TimeExit(20)])`; escalation surface: `MaxDrawdownLimit(20%, warn 15%)` x `DailyLossLimit(2%)`, reporting the more severe action. In production the backtester reconstructs `PortfolioState` from the broker every bar — test the halt-and-resume path. Not a backtest.

**Halt and re-entry in backtests** (guardrails §9 "Adaptive logic and halts"; `RiskManager` verified in `risk/portfolio/manager.py`): keep portfolio limits as governance, never as sweep variants. Once a limit returns `halt` or `liquidate`, `rm.is_halted` stays True and `can_open_position()` False across sessions until `rm.reset_halt()` (which also re-arms the one-shot liquidation), so a run without a reset reports a book that never trades again. Write the re-entry rule before the run and implement it in `on_data`: after `liquidate` equity is frozen, so only a time rule (N sessions) or a lagged market-state rule (the regime label of `06`) can re-enter; after `halt` (positions kept) `rm.current_drawdown < warn_threshold` works. Report halted sessions beside the overlay's Sharpe, count flattened sessions as `(overlay == 0) & (baseline != 0)`, and leave them inside the reported drawdown (equity stays at the halted level; a post-halt tracked index is not money anyone made).

```python
def on_data(self, timestamp, data, context, broker):
    self.rm.update(broker.get_account_value(), {a: p.market_value for a, p in broker.get_positions().items()},
                   timestamp, broker=broker)
    if self.rm.is_halted:
        self.halted_sessions += 1                                            # a flattened session in the overlay row
        if self.halted_sessions < REENTRY_SESSIONS: return                    # re-entry rule fixed before the run (demo: 21 sessions)
        self.rm.reset_halt(); self.halted_sessions = 0; self.reentries += 1   # log every re-entry
    ...                                                                       # normal sizing
```

Kill criteria as rules: one row per rule with layer, measured quantity and window, threshold with its source, action and reset / re-entry. The populated sheet is `decision_rules.md` §14c; its numbers come from four layers this chapter alone does not hold: research no-go thresholds (`us_equities_panel` `kill_conditions`: IC 0.01, edge-to-cost 1.2, micro-cap 0.5, net Sharpe 0.3 after borrow), backtest overlays (`MaxDrawdownLimit` / `DailyLossLimit` above), live `SafeBroker` limits (`libraries/ml4t_live.md`: `max_daily_loss`, `max_drawdown_pct`) and ch26 breakers with half-open recovery (`chapters/26_mlops_governance.md`).

### Vol targeting, exposure caps and turnover tightening (19.7 adaptive controls; `ml4t.backtest` wiring, no dedicated notebook)

No library rule implements vol targeting (`ml4t.backtest.risk` holds position rules and portfolio limits only): it is a weight transform applied before `TargetWeightExecutor`, with every input lagged one session (pre-flight 30(c)). Defaults are conventions, not fitted values: target 10-15% annualized (ch17 `12` uses `VOL_TARGET_ANN=0.15`, `VOL_LOOKBACK=63`), lookback 21 or 63 sessions, leverage cap 1.0 on a cash account and 1.0-1.5 on a margin account (`allow_leverage=True`, else the gatekeeper rejects the levered orders).

```python
VOL_TARGET_ANN, VOL_LOOKBACK, MAX_LEVERAGE = 0.15, 63, 1.5                   # conventions; state them in the term sheet
port = (w.shift(1) * r).sum(axis=1)                                           # un-scaled weights w (date x asset) x returns r, ch16 NB06 identity
realized = port.rolling(VOL_LOOKBACK).std() * np.sqrt(252)
scale = (VOL_TARGET_ANN / realized).shift(1).clip(0.0, MAX_LEVERAGE)         # shift(1): the scale for t uses returns through t-1
context_df = pl.DataFrame({"timestamp": scale.index, "vol_scale": scale.values})   # DataFeed(prices_df=..., signals_df=..., context_df=context_df)
# once, in on_start: self.executor = TargetWeightExecutor(RebalanceConfig(max_gross_leverage=MAX_LEVERAGE, min_weight_change=0.01, min_trade_value=100.0))
# in on_data at a rebalance timestamp, w_t = the model's un-scaled weights for this timestamp:
s = context.get("vol_scale"); s = 0.0 if s is None or np.isnan(s) else s     # warmup: no scale, no position
self.executor.execute({a: s * w_t[a] for a in w_t}, data, broker, timestamp=timestamp)
```

- Per-asset variant (ch17 `12`): `w_i = VOL_TARGET_ANN / (sqrt(252) * max(vol_{i,t-1}, 1e-4) * N) * signal_i`; the 1/N keeps the book at the target rather than sqrt(N) times it.
- Exposure caps: `RebalanceConfig(max_gross_leverage=...)` rescales the whole vector when `sum|w|` exceeds it and `max_single_weight` clips one name; `GrossExposureLimit` / `NetExposureLimit` are the governance check on the realized book, not the sizing rule.
- Turnover tightening: sweep `min_weight_change` (0.005-0.02), `min_trade_value` and the cadence (`RebalanceSchedule.fixed_n_sessions(n)`) on the calibration window only; read turnover from fills and net Sharpe against the ch18 break-even curve.
- Evaluate each control as an overlay row against its own un-scaled parent through the same engine (flattened sessions `(overlay == 0) & (baseline != 0)`); deploy only from the win-win quadrant (Sharpe up and drawdown down) confirmed out of sample (`chapters/20_strategy_synthesis.md`, `07_regime_risk`).

### Systematic risk sweep (`11_systematic_risk_sweep`; Figures 19.6-19.7; capstone)

Params: `START_DATE="2019-01-02"`, `END_DATE="2023-12-31"`, `CALIBRATION_END="2021-12-31"`, `N_BARS=1260`, `LOOKBACK_BARS=60`, `CADENCE_BARS=21`, `INITIAL_CASH=100_000`, `SYMBOLS=["SPY","QQQ","IWM","XLF","EEM"]`, `top_n=3` (present-day universe, not point-in-time membership).

1. Pre-flight: fail closed on schema, duplicate keys, null OHLC, incomplete daily snapshots.
2. `build_momentum_weights(prices, schedule: list[date], lookback, top_n=3) -> dict[date, dict[str, float]]`: trailing returns within symbol; explicit decision-timestamp schedule (trading-bar cadence, never calendar days).
3. `SweepStrategy(Strategy)(weights, rules=None)` submits targets on scheduled bars; `ExecutionMode.NEXT_BAR`; `TargetWeightExecutor` / `RebalanceConfig` rebalance.
4. `run_risk_sweep(prices, weights, rules=None, initial_cash=100_000) -> dict`: Calmar = library CAGR / |max drawdown|; trade stats over **closed trades only** (library `num_trades`/`win_rate` include partly unwound and open trades).
5. `run_1d_sweep(rule_class, levels, name)` + `add_sweep_panel` (Sharpe solid, closed-trade count dotted); `run_rule_grid(left_levels, right_levels, right_rule) -> (sharpe_matrix, calmar_matrix)` for StopLoss x TakeProfit and StopLoss x TrailingStop; `add_surface` baseline-relative heatmaps with one pooled symmetric scale per metric; `apply_surface_axes`.
6. `MAEMFEAnalyzer` percentiles on closed calibration trades -> stop/target priors (second method; small-sample priors, `mae_mfe.num_trades`).
7. Freeze three candidates (1D pick, joint-grid pick by calibration Calmar, MAE/MFE pick); evaluate once on 2022-2023 vs no-rule baseline; rank by CAGR-based Calmar; report closed-trade counts and win rates alongside equity-curve metrics, never instead.

### Parameters not visible in the source notes

Treat these as unknown rather than as values stated elsewhere in this file: the default percentages inside `FixedExitConfig`, `TrailingStopConfig`, `ATRStopConfig`, `HybridExitConfig` and `ScaleOutConfig`; the stop-distance grid passed to `analyze_stop_separation`; GLD shocks for the Rising Rates and EM Crisis scenarios; `MAEMFEAnalyzer`'s threshold method and the `BarrierAnalysis` call signature; the `TighteningTrailingStop` schedule and `ScaledExit` targets/fractions in notebook 10 (the nb10 table shows the argument shapes with the library docstring demos instead); notebook 11's sweep level lists and MAE/MFE percentile choices; notebook 09's cost-sweep basis-point levels; `DriftMonitorConfig` field names and the Wasserstein threshold parameter name; which exit model and rule won on trade-path metrics in 08. `FixedFractionalConfig` defaults were also absent from the notes and were read from the repo (1% risk per trade, 25% cap), as were the library limit defaults above.

## Guardrails and pitfalls

- **Unbacktested VaR** — publishing VaR without exception counting; it is a frequency claim. Run `backtest_var` + Kupiec at the stated level; exception ratio near 1.
- **Kupiec only counts** — ignores exception timing; clustered exceptions pass. Inspect clustering over time; (inference) add an independence test.
- **Historical VaR floor** — cannot exceed the worst loss in window; understates after calm periods. Compare with Student-t/CVaR; include a crisis.
- **Cornish-Fisher instability** — truncated expansion with large kurtosis can worsen coverage. Never prefer automatically; require a coverage backtest.
- **Sqrt-time scaling** — 10-day VaR = 1-day * sqrt(10) needs independence; exception clustering contradicts it. Estimate the horizon directly.
- **Regime-label leakage** — same-day return or full-sample thresholds define the regime. Use prior-close vol, expanding thresholds, minimum history (252 days in 01, 60 in 06).
- **Average-only diversification** — calm periods dominate; correlations rise in crises. Split by lagged volatility state.
- **Symmetric volatility loss** — ranking forecasts by MSE; under-forecasting is the dangerous side. Use QLIKE; the squared-return proxy is noisy.
- **Entry contamination of exit tests** — signal-driven entries flatter or damn the rule. Use random seeded entries; report SE of each mean.
- **Same-bar ambiguity** — bar touches stop and target; target-first manufactures winners. Stop-first; gaps fill at next open; state the order as policy.
- **ML exit timing contract** — positional mapping, no purge, no delay evaluates exits on the wrong date silently. Chronological split, purge `LABEL_HORIZON` rows, shift one session, key by timestamp.
- **Overlay tightening** — adding rules to a hybrid inherits the tightest rule. Read exit-type composition; treat return change as noise.
- **No costs in exit/sizing sims** — barrier-price fills, no slippage/spread/partials; a 3-tranche scale-out pays 3x fixed costs, enough to reverse a comparison; rules differ mainly in exit frequency. Price with ch18 before any claim (02, 03, 08, 10, 11 all flag this).
- **Cross-section incomparability** — fixed/trailing/ATR on all entries vs ML/hybrid on test entries only. Compare within a section.
- **Current-ATR vs entry-ATR** — stop distance breathes with the market, changing the rule. Live positions hold one `VolatilityStop` instance remembering entry ATR.
- **Unlagged sizing inputs** — day-t weight from day-t volatility, known only after close. Window ends the session before.
- **Entry-bar excursions** — entry bar's high/low in MAE/MFE inflates excursions and apparent stop hits. Measure from the bar after a close entry.
- **Unrounded shares** — fractional counts overstate risk-budget precision. Integer shares; recompute value and stop risk.
- **Separation != profit** — MAE discrimination read as stop value has no PnL or costs. Re-run trades with the rule and costs under a frozen objective.
- **In-sample stop thresholds** — the best distance on the trades used to pick it is no evidence. Calibrate to 2021-12-31, 30-bar embargo, test on the future window.
- **Single future window** — one calibration/future pair can agree by chance. Treat the distance as a prior, not an optimum.
- **Misnamed heuristics** — inverse-vol called risk parity implies covariance work not done. Name by construction; use a covariance-aware allocator when correlations are high.
- **Unvalidated conviction scores** — sizing on a score untied to outcomes gives confident wrong-direction sizing. Bounds 0.2x-2.0x; validate the score first.
- **Opaque conformal artifacts** — loading an unavailable prediction artifact cannot support a result. Use the versioned ch17 implementation with its data contract.
- **Contemporaneous betas in attribution** — full-sample betas applied to every date let future betas inflate explanation. `compute_return_attribution(lag=1)`; rolling betas labelled next session.
- **Unadjusted SEs** — plain OLS SEs give t-stats too large. HAC lags 5 everywhere (library calls too).
- **Delta-method omission** — the interval ignores beta-mean covariance. Block bootstrap when decision-relevant.
- **Library-default mismatch** — manual vs library at different HAC bandwidth/window misattributes differences. Pass every shared setting explicitly.
- **Residual screen as test** — |corr| > 0.15 has no null. Find an economic driver; test OOS.
- **Mahalanobis misread** — spikes mix tails and estimation error. Monitor the condition number; Ledoit-Wolf before inversion.
- **Attribution ambiguity** — correlated factors; rotation relabels contributions. Declare a basis.
- **Factor sample bias** — 2010-2024 is one expansion plus one short shock. State coverage.
- **SHAP joined on the wrong row** — exit-date or positional join explains another decision silently. Stable trade ID; SHAP from entry-decision features.
- **Worst trades chosen by outcome** — selection by realized loss mixes failures with unlucky correct calls. Check forecast gap; clusters are hypotheses.
- **Non-point-in-time macro** — finalized FRED snapshot is not vintage. No macro conclusions; PIT vintages for backtests.
- **Unscreened tiny forecasts** — trading any forecast > 0 fills the worst-trade list with untaken trades. 5 bps buffer; better, real costs (620 round trips would pay far more).
- **Backend-dependent reproducibility** — CUDA LightGBM with the same seeds varies; a silent device fallback leaves no trace. Single CPU thread; assert device read back from booster equals `LGB_DEVICE`.
- **Round-number thresholds** — VIX > 25, trend +-5%/10%, 1.3x median, tercile splits are sample-relative, not fitted. State as diagnostics.
- **Single-crisis stress** — one window tests one mechanism. Replay several windows across growth and rate shocks.
- **Single-asset shock** — shocking one holding preserves the diversification a crisis removes. Simultaneous shock vectors to every asset.
- **Hand-dated windows** — a few weeks moves every return. Validate observed-session endpoints; (inference) test boundary sensitivity.
- **Free daily rebalancing in crises** — constant weights at no cost trades most in the least liquid conditions. Apply ch18 costs; treat as implementability upper bound.
- **Gaussian Monte Carlo** — normal draws understate every tail. Kurtosis-matched Student-t with variance-preserving scale; list omitted mechanisms.
- **Deepest-quantile noise** — 99.9% on 10,000 paths rests on ~10 paths and moves between seeds. Report as order of magnitude or draw more.
- **Regime max drawdown** — drawdown on concatenated regime days invents a path. One-day conditional metrics only.
- **Stale artifacts** — outputs nobody reads go stale unnoticed. Write only for a named consumer.
- **Uncalibrated drift thresholds** — 0.10/0.25 PSI and 0.60/0.70 AUC are conventions; untested they alert constantly or never. Calibrate on a period with known drift; set threshold and window size together.
- **Retraining on every drift alert** — shortens the window, raises variance, guarantees the next alert. Require performance degradation plus >= 2 critical or >= 3 warning in the last 5 checks.
- **Treating covariate drift as concept drift** — PSI/Wasserstein/domain AUC never see P(Y|X). Join drift to realized performance; "no drift + performance down" means check concept drift.
- **PSI on tiny samples** — a few hundred rows move every value between adjacent windows. Size windows jointly with thresholds; prefer longer windows at the row frequency.
- **PSI binning dependence** — a different bin count moves every reported value. Fix quantile bins on the reference (`n_bins=10`), state them, read the breakdown.
- **Random CV folds in the domain classifier** — neighbouring rows let it memorize periods and score high AUC with no drift. Contiguous relative-time blocks (`DOMAIN_CV_TIME_BLOCKS=20`), all symbols at one timestamp in one fold; the all-row fit only ranks features, never alerts.
- **Permutation p-values on autocorrelated rows** — overstate precision. Descriptive threshold (half the reference std); settle borderline cases with another window.
- **Hand-picked drift episodes** — crisis-centred windows show what measures do, not how often they fire. Estimate the false-alarm rate on unselected rolling windows before setting production thresholds.
- **Baseline contamination** — a crisis inside the reference absorbs the next. Choose and document a defensible reference window.
- **Monitoring cadence slower than the market** — quarterly review learns about a days-long crypto collapse too late. Match cadence to regime speed.
- **In-sample stacking feature** — an exit model trained on in-sample entry probabilities learns an accuracy production never supplies. Purged expanding-window OOF predictions with fold-local label thresholds.
- **Missing purge at fold boundaries** — a 24h label on a fold's last row reads into the next fold. Purge `FORWARD_HOURS` at the train/test split and at every OOF fold boundary.
- **Threshold refit on test rows** — quantiles refit on test measure nothing. Estimate the 0.95 entry quantile on training rows only; apply unchanged.
- **AUC as the strategy metric** — ranking quality and thresholded trade-path returns diverge. Report both; read the trade-level result.
- **Dominant clause in an OR rule** — "either" collapses to the clause that fires every bar (p < 0.30 for a rare-event model). Inspect per-clause fire rates before combining; same for `AnyOf`/`RuleChain`.
- **Single fixed test interval** — third-decimal AUC gaps are not separable; no sampling-error estimate. More windows, or report uncertainty before naming a winner.
- **Execution-timing leakage in exit backtests** — earning the exit bar or missing the entry bar misstates returns. Entry close t earns t->t+1; exit close u earns nothing after; k-bar hold = k returns; close open positions at the last observed close.
- **Row misalignment after regrouping by symbol** — re-sorting pairs predictions with other symbols' bars silently. Carry an explicit row index through the regroup.
- **Quoting the OCE objective as CVaR** — the minimized value is an expectation over a learned w. Report empirical sample CVaR on validation/test.
- **Omitting liquidation cost** — rewards carrying inventory and biases the comparison against a benchmark that closes. Charge opening, every rebalance and terminal liquidation.
- **Epoch selection on validation** — turns validation into a second training set. Freeze the schedule (100 epochs); validation only monitors and carries the cost sweep; touch test once.
- **Benchmarking a learned policy on its own training generator** — GBM lacks jumps, vol clustering and widening spreads, exactly where a learned hedge would differ. Validate on market data strictly later than fitting; GBM results are a sanity check.
- **Unconstrained policy outputs** — a network emits a position for any input. Wrap with hard exposure, stop-loss and daily-loss limits outside the network; log every input and position; refit only on a stated trigger.
- **Fixed proportional costs** — real costs rise with size and volatility, when a hedge trades most. Treat 10 bp flat as optimistic.
- **Wrong rule order in a chain** — target-before-stop records a double-breach bar as a profitable exit. Stop first; state the order as policy.
- **Reusing a stateful `ScaledExit`** — it remembers fired targets and skips them for the next position. One instance per position; reset before reuse.
- **Trailing stop reading the current bar's high** — sets the water mark it then tests with information the order did not have. High-water mark through the prior completed bar; advance marks only after a bar completes without exit.
- **Net-only exposure limit** — a market-neutral book sits near zero net while levering gross. Limit gross and net separately.
- **Limit class as governance** — the class enforces a number; who is told, who overrides and what must be true to resume are outside it. Write escalation, approval and reinstatement procedures (§19.8).
- **Limits checked on a snapshot** — nothing exercises post-breach behaviour. Test halt-and-resume with `PortfolioState` rebuilt from the broker every bar.
- **Win rate inflated by open trades** — library `num_trades`/`win_rate` include partly unwound and open positions. Compute over closed trades only.
- **Calibrating on the evaluation window** — selection on the reported window is a fit, not a test. Calibrate 2019-2021, freeze, evaluate once on 2022-2023; do not revisit.
- **Multiple-testing overreach in sweeps** — grids of widths plus three selection methods are an unadjusted comparison. Read the one-time evaluation as one realized outcome, not as inference that a rule is universally optimal.
- **Survivorship in the ETF universe** — the present-day five-ETF panel is not point-in-time membership. Use for rule-calibration lessons only, not historical selection claims.
- **Calendar days mixed with trading bars** — schedules drift between decision and execution. Pass explicit decision timestamps (21-trading-bar cadence).
- **Incomplete panels handled inside the executor** — missing assets become silent exceptions. Fail closed before the backtest on schema, duplicates, nulls, incomplete snapshots.
- **Small-sample MAE/MFE percentiles** — few closed calibration trades give noisy levels. Use as priors seeding a search.

## Decision rules and defaults

| Topic | Rule / default |
|---|---|
| Confidence levels | Report 90/95/99% side by side; backtest at 95%; stress MC at 95/99/99.9% |
| Windows | Rolling VaR 252; regime vol 63; 252 days minimum before tercile split; EWMA lambda 0.94; rolling variance 21; MC 10,000 paths, 20-day horizon |
| Tail model | Run `analyze_distribution`/`analyze_tails` first; Student-t with large excess kurtosis; Cornish-Fisher only if coverage backtest passes; CVaR when severity or aggregation guarantees matter |
| Volatility forecasts | Rank by QLIKE, not MSE, for sizing |
| Exit-rule guide | Trend following -> `TrailingStop` + `TimeExit`; mean reversion -> `StopLoss` + `TakeProfit` (symmetric); high volatility -> `VolatilityStop` + `TighteningTrailingStop`; time-sensitive -> `StopLoss` + `TimeExit` (short `max_bars`); conservative -> `AllOf(TimeExit, TrailingStop)`; aggressive profits -> `TighteningTrailingStop` |
| Adaptive sizing (conventions) | vol target 0.10-0.15 annualized, lookback 21/63 ending t-1, scale clipped to [0, 1.0-1.5] and `RebalanceConfig(max_gross_leverage=...)`; turnover via `min_weight_change` 0.005-0.02 / `min_trade_value` / cadence sweeps on calibration only; deploy from the win-win quadrant |
| Rule-chain order | stop -> tightening trail -> target -> time cap; first non-HOLD wins; `RuleChain`/`AnyOf` for "any ends the trade", `AllOf` for "exit only when all hold" |
| Simulation vs library | Manual numpy for mechanics/custom logic/prototyping; `ml4t.backtest.risk` for production (composable, Engine integration, exercised by case-study pipelines) |
| Notebook holding caps | fixed 20 bars, trailing 50, ATR 30, ML 30, hybrid 30; ML threshold 0.5; adverse label -2% over 5 sessions; test fraction 30%; 100 trades |
| Sizing | stop distance -> r -> `V*r/\|entry-stop\|` -> cap -> integer shares -> recompute; fractional Kelly 1/2 or 1/4; conviction 0.2-2.0x; scale-out 1/3, 1/3, remainder with break-even stop and `stop_loss_pct=0.02` (the 2% stop is `simulate_scale_out`'s argument, not a fixed-fractional default) |
| MAE/MFE protocol | calibration to 2021-12-31, 30-bar embargo, entries spaced beyond max holding, freeze distance, accept only if future separation is close, then cost-aware PnL |
| Factor regressions | HAC lags 5; rolling 252; covariance 63; attribution lag 1; residual screen 0.15; Ledoit-Wolf when it persistently lowers condition number |
| Trade SHAP | forecast >= 5 bps to trade; `WORST_N=20`; VIX > 25 stressed; 70% train |
| Stress governance | Replay all seven windows (growth and rate shocks); summary table per allocation (worst window, worst scenario, MC tail); one event never sizes a limit |
| Regime classifier | lookback 60; Bear < -5% annualized trend; Bull > +10% with vol below expanding median; High Vol > 1.3x median; else Calm |
| Drift bands | PSI < 0.10 stable, 0.10-0.25 warning, >= 0.25 critical; domain AUC 0.5 none, >= 0.60 warning, >= 0.70 critical; combined = more severe; Wasserstein descriptive = half reference std; unified flag requires both PSI and Wasserstein |
| Retraining rule | `should_retrain` True only if `performance_degraded` AND (critical >= 2 OR warning >= 3) in last `lookback` (default 5; demo 4) |
| Drift action matrix | Drift + performance OK -> monitor; Drift + down -> any-N rule; No drift + down -> check concept drift; No drift + OK -> keep monitoring |
| Drift triage | One dominant domain-classifier feature -> start there; diffuse -> broader regime review; concentrated PSI bins -> regime shift; uniform small deviations -> sampling noise |
| Drift workflow | Scheduled review of a month-long current window vs reference; row frequency = granularity; investigate on warning/critical; retrain only on count + degradation |
| Drift checklist | Defensible reference window; read the bin breakdown; add a multivariate check; time-blocked folds; cadence matched to market speed; drift AND degradation before retraining |
| ML exit sequencing | chronological split + 24h purge -> train-only 0.95 entry quantile -> 5-fold purged expanding OOF entry probabilities -> basic exit on OOF rows -> enhanced (+`entry_prediction`) -> compare AUC AND trade-path -> enter when p > 0.5 -> exits: model / p < 0.30 / either; `max_hold=24` |
| Deep hedging protocol | 70/15/15 on 50k GBM paths; 30 steps, dt 1/252; q 0.05; 10 bp; 100 epochs, Adam 5e-3, batch 2048, clipping; cap 1.5; significant trade > 0.01; validation never picks an epoch; test once; cost sweep directional only |
| Deployable learned policy | Log every input and position; validate on strictly later market data; refit only on a stated trigger; hard exposure, stop-loss, daily-loss limits outside the network |
| Portfolio-limit values | nb10 demo arguments: `MaxDrawdownLimit` warn 15% / liquidate 20% (layered example: 10% / 15%); `DailyLossLimit` 2% of current equity; `MaxPositionsLimit` 10 (layered 20); `GrossExposureLimit` 1.5 (layered 1.0); `NetExposureLimit` [-0.10, +0.10]. Library bare-constructor defaults (repo-verified): drawdown 0.20 with no warn band, daily loss 0.02, positions 10, gross 1.0, net [-1.0, +1.0]. Neither list is a mandate; the threshold comes from the risk mandate |
| Position-rule demo arguments (nb10) | StopLoss 5%, TakeProfit 10%, TimeExit 20, TrailingStop 5% (3% in chains), TakeProfit 15% for the SPY trigger comparison; layered: 3% stop, tightening trail, 30% target, 60-bar cap |
| Sweep protocol | Calibrate 2019-01-02 to 2021-12-31 -> 1D sweeps -> joint grids by calibration Calmar -> MAE/MFE priors -> freeze three -> evaluate once on 2022-2023 vs baseline -> rank by CAGR-based Calmar; closed-trade counts alongside, never instead |
| Momentum contract (11) | 60-bar lookback, top-3 equal weight of 5 ETFs, rebalance every 21 trading bars, signal on completed close, `NEXT_BAR`, 100k cash |
| Go/no-go (inference from 19.1/19.8) | No deployment without pre-defined limits, escalation rules, drift monitoring and written governance artifacts, all point-in-time safe and auditable |

## Code patterns and APIs

Imports:
- `from ml4t.diagnostic.evaluation.distribution import analyze_distribution, analyze_tails`
- `from ml4t.diagnostic.evaluation import BarrierAnalysis, TradeAnalysis, TradeShapAnalyzer`; `from ml4t.diagnostic.config import TradeConfig`; `from ml4t.diagnostic.config.trade_analysis_config import ExtractionSettings`; `from ml4t.diagnostic.integration.backtest_contract import TradeRecord` (timestamp = exit)
- Factor API (`ml4t.diagnostic.evaluation`): `FactorData.from_dataframe()`, `FactorData.from_fama_french("ff3")`, `compute_factor_model(hac=True)`, `compute_rolling_exposures()`, `compute_return_attribution(lag=1)`, `compute_risk_attribution()`
- `from ml4t.diagnostic.evaluation.drift import analyze_drift` + `PSIResult`, `WassersteinResult`, `DomainClassifierResult`, `DriftSummaryResult`
- `from ml4t.backtest import BacktestConfig, DataFeed, Engine, ExecutionMode, Strategy`; `from ml4t.backtest.risk import StopLoss, TakeProfit, TimeExit, TrailingStop, TighteningTrailingStop, VolatilityStop, VolatilityTrailingStop, SignalExit, ScaledExit, RuleChain, AllOf, AnyOf, MaxDrawdownLimit, DailyLossLimit, MaxPositionsLimit, GrossExposureLimit, NetExposureLimit, PositionState, PositionAction, PortfolioState, ActionType, PositionRule` (position rules in `ml4t.backtest.risk.position`; `ActionType` and `PositionRule` are exported by the package `__init__`, repo-verified); `from ml4t.backtest.analytics.trades import MAEMFEAnalyzer`; `from ml4t.backtest.types import Trade`; `from ml4t.backtest.execution.rebalancer import RebalanceConfig, TargetWeightExecutor`
- Data/style: `data.load_etfs()`, `load_macro`, `load_crypto_perps`, `load_crypto_premium`, `get_output_dir(19, "<name>")`; `from utils.style import COLORS, ml4t_palette, ml4t_diverging, show_plotly_with_alt`
- Frames: polars `pl.DataFrame` for trade/excursion/stress frames (03, 06, 08); pandas + statsmodels (04); LightGBM `device="cpu"`, one thread, fixed seeds (05, 07, 08); PyTorch strict deterministic mode, fail-closed on CUDA (09)

Engine integration, the sweep's pattern (`Strategy` hooks `on_start(self, broker)` and `on_data(self, timestamp, data, context, broker)`): register the chain with the broker once and let the Engine evaluate it every bar.

```python
class SweepStrategy(Strategy):
    def __init__(self, weights, rules=None):
        self.rules, self.weights = rules, weights
        self.executor = TargetWeightExecutor(config=RebalanceConfig(...))
    def on_start(self, broker):
        if self.rules is not None:
            broker.set_position_rules(self.rules)   # e.g. RuleChain([StopLoss(pct=0.05), TakeProfit(pct=0.10), TimeExit(max_bars=20)])
    def on_data(self, timestamp, data, context, broker):
        if timestamp in self.weights:               # explicit decision-timestamp schedule, trading-bar cadence
            self.executor.execute(self.weights[timestamp], data, broker)
```

Standalone rule evaluation (field list verified against `ml4t/backtest/risk/types.py`; the notebooks' `create_position_state(...)` helper fills exactly these): a rule takes a `PositionState` and returns a `PositionAction(action, pct, stop_price, fill_price, reason)`. Test `action.action != ActionType.HOLD` (nb02's `should_exit` helper does this); `PositionAction` has no `should_exit` attribute and `PositionState` has no `high_since_entry` field, so the Engine sketch in notebook 02's commented block does not run.

```python
state = PositionState(asset="SPY", side="long", entry_price=100.0, current_price=97.0,
                      quantity=100.0, initial_quantity=100.0, unrealized_pnl=-300.0, unrealized_return=-0.03,
                      bars_held=3, high_water_mark=101.0, low_water_mark=97.0,      # required fields end here
                      bar_open=97.5, bar_high=98.0, bar_low=96.5, context={"atr": 1.8, "signal": -0.4})
action = rule.evaluate(state)
if action.action != ActionType.HOLD:        # EXIT_FULL / EXIT_PARTIAL (action.pct) / ADJUST_STOP (action.stop_price)
    ...
```

Custom rules implement the `PositionRule` protocol (`evaluate(state) -> PositionAction`), the base the notebooks name for the library rules. Demo helpers: `create_position_state(symbol="SPY", side="long", entry_price=100.0, current_price=100.0, bars_held=0, bar_open/high/low=None, high_water_mark=None, low_water_mark=None)`, `create_portfolio_state(equity=100000, initial_equity=100000, high_water_mark=100000, num_positions=5, positions=None, daily_pnl=0)`.

Layered configuration:

```python
rules = RuleChain([StopLoss(pct=0.03), TighteningTrailingStop(...), TakeProfit(pct=0.30), TimeExit(max_bars=60)])
limits = [MaxDrawdownLimit(max_drawdown=0.15, warn_threshold=0.10), DailyLossLimit(max_daily_loss_pct=0.02),
          MaxPositionsLimit(max_positions=20), GrossExposureLimit(max_gross_exposure=1.0)]
```

Any-N alert count in a lookback:

```python
recent = history[-lookback:]
crit = sum(s == "CRITICAL" for s in recent); warn = sum(s == "WARNING" for s in recent)
retrain = performance_degraded and (crit >= 2 or warn >= 3)
```

Differentiable CVaR (OCE): `loss = w + relu(-pnl - w).mean() / q` with `w` an `nn.Parameter` trained by the same Adam optimizer as the policy.

Student-t MC: `nu = 6/kappa + 4`; `s = sigma*sqrt((nu-2)/nu)`; draws `t(nu)*s + mu` iid; compound; VaR = quantile; CVaR = mean below.

Scenario: `StressScenario(name, shocks={sym: pct}, description)`; impact = `sum(w_i * shock_i)`.

Purged expanding OOF for a stacked feature (per fold): fit rows = timestamp <= validation_start - label_horizon; labels from the fold's own fit-return quantile; predict on the contiguous validation block; place predictions by explicit row index.

LightGBM backend check: read the fitted booster's device parameter and assert it equals `LGB_DEVICE`; one thread; `SEED=42`.

Drift notebook helpers (teaching code): `policy_alert_level`, `combined_alert_level`, `alert_count_condition`, `relative_time_blocks`, `fit_domain_model`, `domain_auc_interpretation`, `domain_result_metadata`, `build_domain_classifier_result`, `domain_classifier`, `checked_drift_analysis`, `finalize_drift_summary`, `summarize_drift_check(summary, config, timestamp)`, `DriftMonitorConfig`, `ProductionDriftMonitor`, `psi_distribution_bars`, `psi_contribution_bar`, `empirical_cdf`, `wasserstein_histograms`, `wasserstein_cdf_traces`.

Run: `uv run python 19_risk_management/<notebook>.py`; tests `uv run pytest tests/test_chapter_notebooks.py -v -k "19_risk_management"`; Docker `ml4t` (`ml4t-gpu` for 09).

## Evidence from the book

Numeric results (VaR/CVaR levels, exception counts, p-values, state-split benefits, stress losses, MC quantiles, betas, AUCs, PSI values, sweep winners) are computed at run time and absent from the notes; the findings below are the qualitative statements the notebooks make.

- `01_var_cvar`: CVaR > VaR at every level, gap widening with confidence; Student-t CVaR above Gaussian at 99%. Cornish-Fisher backtest no better than the plain normal on SPY. Exception clustering visible in the rolling backtest. Equal-weight five-ETF diversification benefit positive in sample, split by state. QLIKE and MSE can rank the three volatility forecasts differently.
- `02_exit_strategies`: exit-type composition shifts reliably with stop width; mean-return differences are within SE at 100 trades. ML exit classifier OOS AUC only slightly above chance (timing-contract demonstration).
- `03_position_sizing_mae_mfe`: scale-out vs full exit on the future sample reports a computed mean difference with no costs; three tranches pay 3x fixed costs.
- `04_factor_exposure`: with HAC lags 5 and window 252 matched, the library static regression reproduces the manual one exactly (coefficients, SEs, t-stats); Ledoit-Wolf risk decomposition is a different estimate. Rolling betas drift over 2010-2024.
- `05_trade_shap_diagnostics`: 620 overnight SPY round trips; forecast gaps confirm genuine model failures; top mean-|SHAP| features contribute in opposite directions across failures.
- `06_stress_testing`: in 2008 non-equity assets protected; in 2022 those same assets were the loss source. Defensive mixes lose less in equity crashes but are not uniformly best under rate shocks. Gaussian MC understates every tail vs kurtosis-matched Student-t; the 99.9% quantile moves between seeds.
- `07_drift_detection`: ETF 2017 vs 2020 is a genuine covariate shift (20-day return volatility roughly doubles, momentum widens, RSI and volume ratio stay closer); largest PSI on features with both location and dispersion change; PSI, Wasserstein and domain classifier agree; the 2020 volatility CDF is shifted upward across most of the support. Quarterly sequence (Q4-2019 reference): drift arrives in Q1 2020 and partly reverses while the monitor stays elevated. Crypto crisis window produces PSI orders of magnitude above 0.25 (log colour scale required); drift arrives in days. Domain-classifier top features overlap but do not match PSI/Wasserstein exactly (interactions).
- `08_ml_exit_signals`: forward 24h crypto returns are heavy in both tails; the rare entry label makes probabilities low almost everywhere and the > 0.5 filter admits almost no bars (sparse, mostly flat policy). Enhanced-vs-basic exit AUC gain is in the third decimal; the "either" rule collapses to the confidence-drop clause (p < 0.30 holds on almost every bar). Lesson: an architectural change can fail on AUC and still matter operationally.
- `09_deep_hedging`: learned hedge vs delta hedge reported as computed text; the learned policy trades less when close to the delta benchmark (association resembling a no-trade band); cost sweep shows an endpoint change in tail loss across bp levels without a monotonicity claim. Delta hedging is at its strongest because paths come from its own model.
- `10_ml4t_backtest_risk_demo`: on the SPY 2020 path StopLoss 5%, TrailingStop 3%, TakeProfit 15% each fire at different dates/fill prices; the 20-bar time exit overlaps price rules and `RuleChain` returns the earlier-declared rule; escalation surface reproduces class contracts (warn 15%, liquidate 20%, daily loss straight to liquidation at 2%).
- `11_systematic_risk_sweep`: the winner by CAGR-based Calmar on 2022-2023 and baseline Calmar are run-time values (`winner["configuration"]`, `winner["calmar"]`, `baseline_evaluation["calmar"]`); MAE/MFE levels rest on few closed calibration trades; configurations with few closed trades can still shape the equity curve because rules act on open positions.

## Related references

- `chapters/09_model_based_features.md` — GARCH volatility models behind the forecast scoring in 01.
- `chapters/08_financial_features.md` — feature engineering and distribution diagnostics; prerequisite for drift monitoring (07).
- `chapters/11_ml_pipeline.md` — purged/embargoed cross-validation and conformal basics reused by ML exits, OOF stacking and conformal sizing.
- `chapters/16_strategy_simulation.md` — `ml4t.backtest` Engine, `ExecutionMode.NEXT_BAR`, trade accounting; prerequisite for 10 and 11.
- `chapters/17_portfolio_construction.md` — Kelly (`04_kelly_criterion`), conformal position sizing (`07_conformal_position_sizing`), covariance-aware allocation vs inverse-vol.
- `chapters/18_transaction_costs.md` — cost model required before any exit, scale-out, rebalancing or sweep claim.
- `chapters/20_strategy_synthesis.md` — cross-case-study risk-overlay comparison (`20_strategy_synthesis/07_regime_risk`).
- `chapters/21_rl_execution_hedging.md` — downstream of deep hedging: exploration and credit assignment for sequential policies.
- `libraries/ml4t_backtest.md` — `ml4t.backtest.risk` rules and limits, `MAEMFEAnalyzer`, `TargetWeightExecutor`.
- `libraries/ml4t_diagnostic.md` — `analyze_distribution`/`analyze_tails`, factor API, `BarrierAnalysis`, `TradeShapAnalyzer`, drift toolkit.
- `case_studies/etfs.md` — the ETF panels used by 01, 03, 04, 06, 07, 11.
- `case_studies/crypto_perps_funding.md` — hourly perps and premium index used by 07 and 08.
- Not cross-referenced by the chapter notes; skill-internal pointers described per `SKILL.md`: `chapters/25_live_trading.md` (SafeBroker controls, staged rollout), `chapters/26_mlops_governance.md` (drift monitoring, promotion gates, circuit breakers), `libraries/ml4t_live.md` (LiveEngine, SafeBroker, shadow-to-live promotion).
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `workflow.md`, `companion_repo.md` — cross-cutting indexes.
- Further reading:
  - Rockafellar & Uryasev (2000) — CVaR; the OCE representation used as the deep-hedging objective.
  - Buehler, Gonon, Teichmann & Wood (2019), Deep hedging, Quantitative Finance 19(8).
  - Moreira & Muir (2017), Volatility-managed portfolios; Patton (2011) and Hansen & Lunde (2006) on volatility loss functions; Bollerslev (1986); Cont (2001).
  - Daniel & Moskowitz (2016), Momentum crashes; Ang & Timmermann (2011) and Khang (2022) on regimes; Shu & Mulvey (2025).
  - Alankar et al. (2023) and Varma (2025) on drawdown rules; Zumbach & Zumbach (2025) historical stress tests; Bianchi et al. (2023) fat tails; Martin et al. (2024) minimum downside risk; Browne et al. (2023).
  - Whalley & Wilmott (1997); Brixton et al. (2022) stock-bond correlation; Hurst (2010) risk parity; Schwert (1989).
  - Fed SR 11-7 (model risk); Lopez de Prado (2018); Paleologo (2025); Harvey et al. (2022); Lundberg & Lee (2017) SHAP.

## Glossary

- **VaR** — loss not exceeded with probability a over one day; `-q_{1-a}` of returns.
- **CVaR / expected shortfall** — mean loss given VaR is breached; subadditive, coherent.
- **Cornish-Fisher** — skew/kurtosis correction to the Gaussian quantile (truncated expansion).
- **Cantelli inequality** — distribution-free one-sided tail bound `1/(1+k^2)`.
- **Kupiec POF test** — likelihood-ratio test of exception count vs expected rate.
- **QLIKE** — asymmetric volatility-forecast loss `log(h) + sigma^2/h`.
- **EWMA (RiskMetrics)** — `h_t = lambda*h_{t-1} + (1-lambda)*r_{t-1}^2`, lambda 0.94.
- **Drawdown** — wealth vs running peak (negative %); max drawdown; time-to-recovery.
- **Calmar ratio** — CAGR divided by absolute maximum drawdown.
- **MAE / MFE** — maximum adverse / favorable excursion of a trade, from the bar after entry; percentiles seed stop/target priors.
- **ATR** — `max(H-L, |H-prevC|, |L-prevC|)`.
- **Purge / embargo** — drop training rows overlapping test labels; gap bars after the boundary.
- **Fixed fractional sizing** — `shares = V*r / |entry - stop|`.
- **Inverse-volatility weights** — `1/sigma` normalized; not risk parity.
- **Scale-out** — tranche exits with stop ratcheted to break-even.
- **HAC / Newey-West** — autocorrelation-robust standard errors (lags 5 here).
- **Ledoit-Wolf** — covariance shrinkage; lowers condition number before inversion.
- **Rotation ambiguity** — orthogonal re-basis of correlated factors relabels contributions.
- **TreeSHAP** — exact Shapley attributions for trees; base + sum = prediction.
- **Kurtosis-matched Student-t** — `nu = 6/kappa + 4`, scale `sigma*sqrt((nu-2)/nu)`.
- **Regime (06)** — Bear / Bull / High Vol / Calm from lagged 60-day trend and vol vs expanding median.
- **Covariate / concept / prior drift** — change in P(X) / P(Y|X) / P(Y).
- **PSI** — sum over bins of `(actual - expected) * ln(actual/expected)`; bands 0.10 / 0.25.
- **Wasserstein distance (EMD)** — minimum transport work between distributions; bin-free, order-aware.
- **Domain classifier** — classifier separating reference (0) from current (1); OOF AUC 0.5 means no detectable drift.
- **Blocked (relative-time) CV** — contiguous time-block folds so neighbouring rows never sit in opposite folds.
- **Any-N-in-lookback rule** — retraining trigger counting alerts anywhere in the last N checks.
- **Reference window** — the baseline every drift number is compared to.
- **OOF stacking feature** — a model's prediction for a row from a model that never saw it.
- **Entry confidence drop** — exit when the entry model's probability falls below 0.30.
- **Close-to-close trade accounting** — entry at close t earns t->t+1; exit at close u earns nothing beyond u.
- **Deep hedging** — learning hedge positions by minimizing a risk measure of terminal PnL under costs.
- **Semi-recurrent hedger** — one MLP per timestep taking (information, previous position).
- **Rockafellar-Uryasev / OCE** — `CVaR_{1-q}(L) = min_w [w + E[(L-w)_+]/q]`; makes CVaR differentiable.
- **No-transaction band** — region around the frictionless target where trading is suboptimal under proportional costs.
- **PositionState / PositionAction** — one position's state and the rule verdict (HOLD, EXIT_FULL, EXIT_PARTIAL, ADJUST_STOP).
- **PortfolioState** — equity, high-water mark, positions, daily PnL snapshot used by portfolio limits.
- **RuleChain / AllOf / AnyOf** — first-non-HOLD-wins chain / AND / OR (AnyOf aliases RuleChain).
- **High-water mark** — highest price since entry through the prior completed bar.
- **Gross vs net exposure** — sum of absolute position sizes vs signed sum.
- **Kill switch** — portfolio-level rule that halts trading or liquidates (drawdown, daily loss).
- **NEXT_BAR execution** — orders from a close-t signal fill on the following bar.
