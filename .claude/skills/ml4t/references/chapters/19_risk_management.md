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
- **Position rules and portfolio limits are different controls; neither substitutes.** Limit gross and net exposure separately. A limit class enforces a threshold; the value comes from a risk mandate, and escalation/approval/reinstatement live outside the library (§19.8).
- **Calibrate before evaluating; evaluate once; keep the operational contract explicit** (completed closes, NEXT_BAR execution, trading-bar schedules, fail-closed panels). Write only artifacts a named consumer reads.

## Method recipes (the how)

### VaR and CVaR four ways, with coverage backtest (`19_risk_management/01_var_cvar`)

Config: `SYMBOL="SPY"`, `PORTFOLIO_SYMBOLS=["SPY","AGG","GLD","EFA","EEM"]`, 2006-01-01 to 2024-12-31 (deliberately pre-2008), `CONFIDENCE_LEVELS=[0.90,0.95,0.99]`, `BACKTEST_CONFIDENCE=0.95`, `ROLLING_WINDOW=252`, `N_MC_SIMULATIONS=10_000`, `SEED=42`.

| Method | VaR_a | CVaR_a | Use when |
|---|---|---|---|
| Historical | `-q_{1-a}(r)` | `E[L \| L >= VaR_a]` | Default; cannot exceed worst loss in window — include a crisis |
| Parametric (Gaussian) | `-mu + sigma * Phi^{-1}(1-a)` | `-mu + sigma * phi(Phi^{-1}(1-a)) / a` | Benchmark only; understates fat tails |
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

- Rolling VaR/CVaR: trailing 252 daily unannualized returns (`ROLLING_VAR_WINDOW=21` for the short variant); lower panel plots CVaR/VaR ratio.
- Regime-conditional tail: label each day by **prior-close** 63-day volatility (`REGIME_WINDOW=63`) against **expanding point-in-time tercile thresholds** built from lagged observations only; require `REGIME_MIN_HISTORY=252` days before the first split.
- Diversification: `portfolio_tail_risk(returns_matrix, weights, confidence=0.95)` compares empirical portfolio VaR/CVaR with weighted stand-alone estimates (equal weights, five ETFs); then split the benefit by the same lagged volatility state.
- Drawdown: `drawdown_path(returns) -> (wealth, running peak, drawdown %)`; plot with zero at the top.

### Volatility-forecast scoring (`01_var_cvar`; deeper GARCH in ch9)

- Candidates: 21-day rolling variance; EWMA `h_t = 0.94*h_{t-1} + 0.06*r_{t-1}^2` (`EWMA_LAMBDA=0.94`, RiskMetrics); GARCH(1,1) fitted strictly before `FORECAST_EVALUATION_START="2020-01-02"`, evaluated OOS from that date.
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
| ML | nine features (returns over four horizons, realized vol, RSI, distance from two MAs, volume vs recent avg); label = close fell > 2% over next 5 sessions; `ml_exit_triggered(proba_by_ts, idx, exit_threshold, bar_timestamps)`; `simulate_ml_exits(..., exit_threshold=0.5, max_holding_days=30)` | 30 | Chronological split, purge 5 rows before boundary, MDI on expanding folds; probabilities keyed by timestamp and shifted one session (close t -> open t+1); bars without an OOS signal never trigger |
| Hybrid | `HybridExitConfig`; `hybrid_bar_fill(open, high, low, stop, tp, ml_exit, stop_type)` evaluates barriers in declared priority; `simulate_one_hybrid_exit` owns event ordering; `simulate_hybrid_exits(..., max_holding_days=30)` | 30 | Inherits the tightest rule |

- Compare only within a section: fixed/trailing/ATR run on all entries, ML/hybrid on test-period entries only.
- `BarrierAnalysis` (`ml4t.diagnostic.evaluation`): TP-hit / SL-hit / timeout rates per signal quintile. Reading: Q5 higher TP rate -> scale up strong signals; Q1 higher SL rate -> cut weak signals; no quintile difference -> improve the signal; high timeout rate -> barriers too tight or signal too slow.
- `MAEMFEAnalyzer` (`ml4t.backtest.analytics`) derives thresholds from `Trade` objects of the representative ATR x2.0 trades; in-sample candidates only.

### Position sizing and MAE/MFE stop calibration (`03_position_sizing_mae_mfe`)

Config: `SYMBOLS=["SPY","QQQ","IWM","TLT"]`, 2018-01-01 to 2024-01-01, `CALIBRATION_END="2021-12-31"`, `HOLDING_PERIOD=20`, `MAX_HOLDING_DAYS=30`, entries spaced beyond the longest holding period, 30-bar embargo after the calibration boundary.

| Method | Formula / API | Defaults | Caveat |
|---|---|---|---|
| Fixed fractional | `shares = V*r / \|P_entry - P_stop\|`; `calculate_fixed_fractional_size(portfolio_value, entry_price, stop_price, config) -> dict` with `FixedFractionalConfig` (risk fraction, concentration cap) | default stop 2% | Integer shares; cap binds for tight stops, risk budget for wide stops; recompute realized stop risk after rounding |
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
| PSI | `sum_i (p_i^actual - p_i^expected) * ln(p_i^actual / p_i^expected)` over quantile bins fixed on the reference, `n_bins=10` | < 0.10 stable; 0.10-0.25 warning; >= 0.25 critical (left-inclusive) | Regulatory/quick checks; sensitive to binning, ignores ordering; read the bin breakdown (`plot_psi_breakdown(result: PSIResult, feature_name)`) |
| Wasserstein / EMD | `W_p(P,Q) = (inf_gamma integral \|\|x-y\|\|^p dgamma)^(1/p)`; bin-free, order-aware | descriptive: half the reference std; no permutation p-value (autocorrelated rows) | Continuous features, interpretable; one feature at a time (`plot_wasserstein_comparison(reference, current, result: WassersteinResult, feature_name)`) |
| Domain classifier | label reference=0 / current=1, LightGBM one thread, out-of-fold AUC over contiguous relative-time blocks (`relative_time_blocks`, 20 blocks; digest says five blocked folds — (inference) 20 blocks grouped into 5 folds); all-row fit only ranks features | 0.5 none; >= 0.60 warning; >= 0.70 critical | Multivariate drift and interactions; needs more data, less interpretable; `domain_classifier(reference, current, features, warning_threshold=0.60, critical_threshold=0.70) -> DomainClassifierResult` |

Drift taxonomy: covariate P(X) (check performance, then recalibrate/retrain); concept P(Y|X) (new architecture/features; needs labels); prior P(Y) (regime-aware models).

Unified: `checked_drift_analysis(reference, current, features, psi_warning_threshold=0.10, psi_critical_threshold=0.25, domain_auc_warning_threshold=0.60, domain_auc_critical_threshold=0.70) -> DriftSummaryResult` wraps library `analyze_drift`; requires both univariate methods to complete and both to flag a feature; multivariate alert separate; fails rather than returning a degraded partial result. `finalize_drift_summary(summary, domain)` attaches the domain result.

Alert policy: `policy_alert_level(value, warning_threshold, critical_threshold)` (left-inclusive, shared by PSI and AUC); `combined_alert_level(...)` returns the more severe; `alert_count_condition(statuses, critical_required=2, warning_required=3)` counts anywhere in the lookback (any-N, not consecutive).

Production loop: `DriftMonitorConfig` (configuration separate from execution); `ProductionDriftMonitor(reference_df, config)` with `check_drift(current_df, timestamp) -> dict`, `get_dashboard_data() -> pd.DataFrame`, `should_retrain(performance_degraded: bool, lookback: int = 5) -> (bool, str)`; returns False until `len(drift_history) >= lookback`; retrain only if performance degraded AND (>= 2 critical or >= 3 warning in the last `lookback` checks). Feature helpers: `etf_feature_panel(prices)` (per-symbol rolling features, only current and prior observations), `etf_window(start, end, cols=None)`, `crypto_feature_panel()`, `crypto_window(start, end)` (complete rows only).

### Two-model ML exit signals (`08_ml_exit_signals`; Figure 19.5)

Params: `SEED=42`, `FORWARD_HOURS=24`, `N_OOF_FOLDS=5`, `N_IMPORTANCE_REPEATS=5`, `TEST_FRACTION=0.20`, `ENTRY_QUANTILE=0.95`, `ENTRY_PROBABILITY_THRESHOLD=0.5`, `PROBABILITY_BIN_WIDTH=0.02`, `ENTRY_CONFIDENCE_DROP=0.30`; BTCUSDT, ETHUSDT, SOLUSDT hourly perps from 2023-01-01.

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

Setup: sell a European call, receive premium p0, hedge at t0..t_{n-1}. `PL_T = p0 - Z + sum_k delta_k (S_{k+1} - S_k) - C_T(delta)`, `Z = max(S_T - K, 0)`; `C_T` charges initial trade, every rebalance and liquidation of the final position at S_T.

Params: `N_PATHS=50_000`, `N_STEPS=30`, `S0=100`, `STRIKE=100`, `MU=0.05`, `SIGMA=0.20`, `R=0.0`, `DT=1/252`, `COST_RATE=0.001` (10 bp), `CVAR_TAIL_PROBABILITY=0.05`, `HIDDEN_SIZE=32`, `N_EPOCHS=100`, `LR=5e-3`, `BATCH_SIZE=2048`, `SEED=42`, `SWEEP_SEED=7_042`, `COST_SWEEP_EPOCHS=10`, `COST_SWEEP_MAX_TRAIN_PATHS=10_000`, `TRAIN_FRACTION=0.70`, `VALIDATION_FRACTION=0.15` (test 0.15), `SIGNIFICANT_TRADE_THRESHOLD=0.01`.

1. Paths: `simulate_gbm_paths(s0, mu, sigma, n_paths, n_steps, dt, seed)`; benchmark `bs_price_delta(spot, strike, tau, sigma, r=0.0)`.
2. Information set `prepare_info(paths, strike, sigma, n_steps, dt)`: log-moneyness, normalized time-to-maturity, implied-vol proxy (constant under GBM), log return; `N_FEATURES=4`.
3. Policy `DeepHedger(n_steps, n_features, hidden_size=32, max_position=1.5)`: one MLP per timestep (`make_step_network`), previous position fed forward (semi-recurrent, no BPTT beyond the position).
4. Objective `CVaRLoss(tail_probability=0.05)`: `CVaR_{1-q}(L) = min_w [w + E[(L-w)_+]/q]`, `L = -PL_T`, `w` an `nn.Parameter` trained jointly by Adam; gradient clipping.
5. Protocol: `reset_torch_training_state(seed)`; train from scratch (no cache); validation only monitors the frozen schedule and carries the cost sweep (no epoch selection); test touched once. No purge needed on independent draws (needed on real data).
6. Evaluate with `evaluate_hedger(...)` = empirical sample `compute_cvar(pnl, tail_probability=0.05)`, never the learned w; `pnl_stats`; `compute_hedge_pnl(positions, paths, payoff, premium, cost_rate)`.
7. Cost sweep `train_cost_level(cost_rate)` (10 epochs / 10k paths, directional only) with matched delta trade count via `mean_significant_trades(positions, threshold=0.01)`; diagnostics `empirical_cdf` (magnified worst decile), `conditional_inaction_diagnostic(positions, benchmark_delta, threshold)`.

### `ml4t.backtest.risk` position rules and portfolio limits (`10_ml4t_backtest_risk_demo`)

Position layer (`ml4t.backtest.risk.position`): rule takes `PositionState`, returns `PositionAction` in {HOLD, EXIT_FULL, EXIT_PARTIAL, ADJUST_STOP} with fill price and reason; ATR/signals via `PositionState.context`.

| Rule | Constructor (demo default) | Behaviour |
|---|---|---|
| `StopLoss` | `pct=0.05` | Measured from entry; never moves |
| `TakeProfit` | `pct=0.10` | Static target |
| `TimeExit` | `max_bars=20` | Holding cap |
| `TrailingStop` | `pct=0.05` (3% in chains) | From high-water mark through the prior completed bar; ratchets up only |
| `TighteningTrailingStop` | schedule of (return reached, trail width) pairs (values not visible) | Trail narrows as profit grows |
| `VolatilityStop` / `VolatilityTrailingStop` | ATR multiples | Current-ATR mode widens with ATR; a live position holds one instance that remembers entry ATR |
| `SignalExit` | signed signal threshold | Signal below negative threshold exits a long |
| `ScaledExit` | fraction of remainder per target (values not visible) | Stateful: one instance per position, reset before reuse |

Composition: `RuleChain([...])` first non-HOLD wins in declared order; `AllOf` = AND (e.g. `TakeProfit(pct=0.01)` + `TimeExit(max_bars=5)` = gain >= 1% AND held >= 5 bars); `AnyOf` = OR (alias of `RuleChain`).

Portfolio layer (`PortfolioState` from equity, initial_equity, high_water_mark, positions, daily_pnl):

| Limit | Constructor (demo default) | Action |
|---|---|---|
| `MaxDrawdownLimit` | `max_drawdown=0.20, warn_threshold=0.15` | Warn, then liquidate |
| `DailyLossLimit` | `max_daily_loss_pct=0.02` | vs current equity (tightens as the book shrinks); straight to liquidation, no warning band |
| `MaxPositionsLimit` | `max_positions=10` | Halt new trading |
| `GrossExposureLimit` | `max_gross_exposure=1.5` | sum of |positions| |
| `NetExposureLimit` | `max_net_exposure=0.10, min_net_exposure=-0.10` | signed sum |

Demo contract (`first_trigger(frame, rule_name, rule)`, SPY 2020, `N_BARS=252`): entry after first close of 2020; entry bar never tested; current-bar OHLC observable to active stops; water marks advance only after a bar completes without an exit. In production the backtester reconstructs `PortfolioState` from the broker every bar — test the halt-and-resume path. Not a backtest.

### Systematic risk sweep (`11_systematic_risk_sweep`; Figures 19.6-19.7; capstone)

Params: `START_DATE="2019-01-02"`, `END_DATE="2023-12-31"`, `CALIBRATION_END="2021-12-31"`, `N_BARS=1260`, `LOOKBACK_BARS=60`, `CADENCE_BARS=21`, `INITIAL_CASH=100_000`, `SYMBOLS=["SPY","QQQ","IWM","XLF","EEM"]`, `top_n=3` (present-day universe, not point-in-time membership).

1. Pre-flight: fail closed on schema, duplicate keys, null OHLC, incomplete daily snapshots.
2. `build_momentum_weights(prices, schedule: list[date], lookback, top_n=3) -> dict[date, dict[str, float]]`: trailing returns within symbol; explicit decision-timestamp schedule (trading-bar cadence, never calendar days).
3. `SweepStrategy(Strategy)(weights, rules=None)` submits targets on scheduled bars; `ExecutionMode.NEXT_BAR`; `TargetWeightExecutor` / `RebalanceConfig` rebalance.
4. `run_risk_sweep(prices, weights, rules=None, initial_cash=100_000) -> dict`: Calmar = library CAGR / |max drawdown|; trade stats over **closed trades only** (library `num_trades`/`win_rate` include partly unwound and open trades).
5. `run_1d_sweep(rule_class, levels, name)` + `add_sweep_panel` (Sharpe solid, closed-trade count dotted); `run_rule_grid(left_levels, right_levels, right_rule) -> (sharpe_matrix, calmar_matrix)` for StopLoss x TakeProfit and StopLoss x TrailingStop; `add_surface` baseline-relative heatmaps with one pooled symmetric scale per metric; `apply_surface_axes`.
6. `MAEMFEAnalyzer` percentiles on closed calibration trades -> stop/target priors (second method; small-sample priors, `mae_mfe.num_trades`).
7. Freeze three candidates (1D pick, joint-grid pick by calibration Calmar, MAE/MFE pick); evaluate once on 2022-2023 vs no-rule baseline; rank by CAGR-based Calmar; report closed-trade counts and win rates alongside equity-curve metrics, never instead.

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
- **In-sample thresholds / single future window** — best distance on the trades used to pick it; one calibration/future pair can agree by chance. Calibrate to 2021-12-31, 30-bar embargo, future test; treat as prior.
- **Misnamed heuristics** — inverse-vol called risk parity implies covariance work not done. Name by construction; use a covariance-aware allocator when correlations are high.
- **Unvalidated conviction scores** — sizing on a score untied to outcomes gives confident wrong-direction sizing. Bounds 0.2x-2.0x; validate the score first.
- **Opaque conformal artifacts** — loading an unavailable prediction artifact cannot support a result. Use the versioned ch17 implementation with its data contract.
- **Contemporaneous betas in attribution** — full-sample betas applied to every date let future betas inflate explanation. `compute_return_attribution(lag=1)`; rolling betas labelled next session.
- **Unadjusted SEs / delta-method omission** — plain OLS SEs give t-stats too large; the interval ignores beta-mean covariance. HAC lags 5 everywhere (library calls too); block bootstrap when decision-relevant.
- **Library-default mismatch** — manual vs library at different HAC bandwidth/window misattributes differences. Pass every shared setting explicitly.
- **Residual screen as test / Mahalanobis misread** — |corr| > 0.15 has no null; spikes mix tails and estimation error. Find an economic driver, test OOS; monitor condition number; Ledoit-Wolf before inversion.
- **Attribution ambiguity** — correlated factors; rotation relabels contributions. Declare a basis.
- **Factor sample bias** — 2010-2024 is one expansion plus one short shock. State coverage.
- **SHAP joined on the wrong row** — exit-date or positional join explains another decision silently. Stable trade ID; SHAP from entry-decision features.
- **Worst trades chosen by outcome** — selection by realized loss mixes failures with unlucky correct calls. Check forecast gap; clusters are hypotheses.
- **Non-point-in-time macro** — finalized FRED snapshot is not vintage. No macro conclusions; PIT vintages for backtests.
- **Unscreened tiny forecasts** — trading any forecast > 0 fills the worst-trade list with untaken trades. 5 bps buffer; better, real costs (620 round trips would pay far more).
- **Backend-dependent reproducibility** — CUDA LightGBM with the same seeds varies; a silent device fallback leaves no trace. Single CPU thread; assert device read back from booster equals `LGB_DEVICE`.
- **Round-number thresholds** — VIX > 25, trend +-5%/10%, 1.3x median, tercile splits are sample-relative, not fitted. State as diagnostics.
- **Single-crisis / single-asset stress** — one window tests one mechanism; shocking one holding preserves diversification a crisis removes. Several windows across growth and rate shocks; simultaneous shock vectors.
- **Hand-dated windows** — a few weeks moves every return. Validate observed-session endpoints; (inference) test boundary sensitivity.
- **Free daily rebalancing in crises** — constant weights at no cost trades most in the least liquid conditions. Apply ch18 costs; treat as implementability upper bound.
- **Gaussian Monte Carlo** — normal draws understate every tail. Kurtosis-matched Student-t with variance-preserving scale; list omitted mechanisms.
- **Deepest-quantile noise** — 99.9% on 10,000 paths rests on ~10 paths and moves between seeds. Report as order of magnitude or draw more.
- **Regime max drawdown** — drawdown on concatenated regime days invents a path. One-day conditional metrics only.
- **Stale artifacts** — outputs nobody reads go stale unnoticed. Write only for a named consumer.
- **Uncalibrated drift thresholds** — 0.10/0.25 PSI and 0.60/0.70 AUC are conventions; untested they alert constantly or never. Calibrate on a period with known drift; set threshold and window size together.
- **Retraining on every drift alert** — shortens the window, raises variance, guarantees the next alert. Require performance degradation plus >= 2 critical or >= 3 warning in the last 5 checks.
- **Treating covariate drift as concept drift** — PSI/Wasserstein/domain AUC never see P(Y|X). Join drift to realized performance; "no drift + performance down" means check concept drift.
- **PSI on tiny samples / binning dependence** — a few hundred rows or a different bin count moves every value. Size windows with thresholds; fix quantile bins on the reference (`n_bins=10`); read the breakdown.
- **Random CV folds in the domain classifier** — neighbouring rows let it memorize periods and score high AUC with no drift. Contiguous relative-time blocks (`DOMAIN_CV_TIME_BLOCKS=20`), all symbols at one timestamp in one fold; the all-row fit only ranks features, never alerts.
- **Permutation p-values on autocorrelated rows** — overstate precision. Descriptive threshold (half the reference std); settle borderline cases with another window.
- **Hand-picked drift episodes / baseline contamination** — crisis-centred windows show what measures do, not how often they fire; a crisis inside the reference absorbs the next. Estimate false-alarm rate on unselected rolling windows; document a defensible reference.
- **Monitoring cadence slower than the market** — quarterly review learns about a days-long crypto collapse too late. Match cadence to regime speed.
- **In-sample stacking feature / missing fold purge / threshold refit on test** — an exit model trained on in-sample entry probabilities learns an accuracy production never supplies; a 24h label on a fold's last row reads into the next fold; quantiles refit on test measure nothing. Purged expanding OOF, fold-local thresholds, train-only 0.95 quantile applied unchanged.
- **AUC as the strategy metric / dominant OR clause / single test interval** — ranking and trade-path returns diverge; "either" collapses to the clause that fires every bar; third-decimal AUC gaps are not separable on one interval. Report both metrics, inspect per-clause fire rates, use more windows or report uncertainty.
- **Execution-timing leakage / row misalignment in exit backtests** — earning the exit bar or missing the entry bar misstates returns; regrouping by symbol pairs predictions with other bars. Entry close t earns t->t+1, exit close u earns nothing after; carry an explicit row index.
- **Quoting the OCE objective as CVaR / omitting liquidation cost / epoch selection on validation** — the minimized value is an expectation over learned w; uncharged liquidation rewards carrying inventory; epoch selection makes validation a second training set. Empirical CVaR on validation/test; charge open, rebalances and liquidation; freeze 100 epochs, touch test once.
- **Benchmarking a learned policy on its own training generator / unconstrained outputs / flat costs** — GBM lacks jumps and vol clustering; a network emits a position for any input; real costs rise with size and volatility. Validate on strictly later market data; wrap with hard exposure, stop-loss and daily-loss limits outside the network; log every input and position; treat 10 bp flat as optimistic.
- **Wrong rule order / reusing stateful `ScaledExit` / trailing stop on current-bar high** — target-before-stop records a double-breach bar as a win; `ScaledExit` remembers fired targets; a water mark set from the current high uses information the order did not have. Stop first; one instance per position; high-water mark through the prior completed bar.
- **Net-only exposure limit / limit class as governance / limits checked on a snapshot** — market-neutral books lever gross; a class enforces a number but not escalation; a snapshot never exercises post-breach behaviour. Limit gross and net; write escalation, approval and reinstatement procedures; test halt-and-resume with `PortfolioState` rebuilt every bar.
- **Win rate inflated by open trades** — library `num_trades`/`win_rate` include partly unwound and open positions. Compute over closed trades only.
- **Calibrating on the evaluation window / multiple-testing overreach / survivorship / calendar-vs-bar mixing / incomplete panels** — selection on the reported window is a fit; grids plus three selection methods are an unadjusted comparison; the five-ETF panel is present-day; calendar schedules drift from trading bars; missing assets become silent exceptions. Calibrate 2019-2021, evaluate once on 2022-2023; read the result as one realized outcome; use for rule-calibration lessons only; pass explicit decision timestamps; fail closed before the backtest.
- **Small-sample MAE/MFE percentiles** — few closed calibration trades give noisy levels. Use as priors seeding a search.

## Decision rules and defaults

| Topic | Rule / default |
|---|---|
| Confidence levels | Report 90/95/99% side by side; backtest at 95%; stress MC at 95/99/99.9% |
| Windows | Rolling VaR 252; regime vol 63; 252 days minimum before tercile split; EWMA lambda 0.94; rolling variance 21; MC 10,000 paths, 20-day horizon |
| Tail model | Run `analyze_distribution`/`analyze_tails` first; Student-t with large excess kurtosis; Cornish-Fisher only if coverage backtest passes; CVaR when severity or aggregation guarantees matter |
| Volatility forecasts | Rank by QLIKE, not MSE, for sizing |
| Exit-rule guide | Trend following -> `TrailingStop` + `TimeExit`; mean reversion -> `StopLoss` + `TakeProfit` (symmetric); high volatility -> `VolatilityStop` + `TighteningTrailingStop`; time-sensitive -> `StopLoss` + `TimeExit` (short `max_bars`); conservative -> `AllOf(TimeExit, TrailingStop)`; aggressive profits -> `TighteningTrailingStop` |
| Rule-chain order | stop -> tightening trail -> target -> time cap; first non-HOLD wins; `RuleChain`/`AnyOf` for "any ends the trade", `AllOf` for "exit only when all hold" |
| Simulation vs library | Manual numpy for mechanics/custom logic/prototyping; `ml4t.backtest.risk` for production (composable, Engine integration, exercised by case-study pipelines) |
| Notebook holding caps | fixed 20 bars, trailing 50, ATR 30, ML 30, hybrid 30; ML threshold 0.5; adverse label -2% over 5 sessions; test fraction 30%; 100 trades |
| Sizing | stop distance -> r -> `V*r/\|entry-stop\|` -> cap -> integer shares -> recompute; fractional Kelly 1/2 or 1/4; conviction 0.2-2.0x; scale-out 1/3, 1/3, remainder with break-even stop; default stop 2% |
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
| ML exit sequencing | chronological split + 24h purge -> train-only 0.95 entry quantile -> 5-fold purged expanding OOF entry probabilities -> basic exit on OOF rows -> enhanced (+`entry_prediction`) -> compare AUC AND trade-path -> enter when p > 0.5 -> exits: model / p < 0.30 / either; `max_hold=24` |
| Deep hedging protocol | 70/15/15 on 50k GBM paths; 30 steps, dt 1/252; q 0.05; 10 bp; 100 epochs, Adam 5e-3, batch 2048, clipping; cap 1.5; significant trade > 0.01; validation never picks an epoch; test once; cost sweep directional only |
| Deployable learned policy | Log every input and position; validate on strictly later market data; refit only on a stated trigger; hard exposure, stop-loss, daily-loss limits outside the network |
| Portfolio-limit defaults | `MaxDrawdownLimit` warn 15% / liquidate 20% (layered: 10% / 15%); `DailyLossLimit` 2% of current equity; `MaxPositionsLimit` 10 (layered 20); `GrossExposureLimit` 1.5 (layered 1.0); `NetExposureLimit` [-0.10, +0.10] |
| Position-rule demo defaults | StopLoss 5%, TakeProfit 10%, TimeExit 20, TrailingStop 5% (3% in chains), TakeProfit 15% for the SPY trigger comparison; layered: 3% stop, tightening trail, 30% target, 60-bar cap |
| Sweep protocol | Calibrate 2019-01-02 to 2021-12-31 -> 1D sweeps -> joint grids by calibration Calmar -> MAE/MFE priors -> freeze three -> evaluate once on 2022-2023 vs baseline -> rank by CAGR-based Calmar; closed-trade counts alongside, never instead |
| Momentum contract (11) | 60-bar lookback, top-3 equal weight of 5 ETFs, rebalance every 21 trading bars, signal on completed close, `NEXT_BAR`, 100k cash |
| Go/no-go (inference from 19.1/19.8) | No deployment without pre-defined limits, escalation rules, drift monitoring and written governance artifacts, all point-in-time safe and auditable |

## Code patterns and APIs

Imports:
- `from ml4t.diagnostic.evaluation.distribution import analyze_distribution, analyze_tails`
- `from ml4t.diagnostic.evaluation import BarrierAnalysis, TradeAnalysis, TradeShapAnalyzer`; `from ml4t.diagnostic.config import TradeConfig`; `from ml4t.diagnostic.config.trade_analysis_config import ExtractionSettings`; `from ml4t.diagnostic.integration.backtest_contract import TradeRecord` (timestamp = exit)
- Factor API (`ml4t.diagnostic.evaluation`): `FactorData.from_dataframe()`, `FactorData.from_fama_french("ff3")`, `compute_factor_model(hac=True)`, `compute_rolling_exposures()`, `compute_return_attribution(lag=1)`, `compute_risk_attribution()`
- `from ml4t.diagnostic.evaluation.drift import analyze_drift` + `PSIResult`, `WassersteinResult`, `DomainClassifierResult`, `DriftSummaryResult`
- `from ml4t.backtest import BacktestConfig, DataFeed, Engine, ExecutionMode, Strategy`; `from ml4t.backtest.risk import StopLoss, TakeProfit, TimeExit, TrailingStop, TighteningTrailingStop, VolatilityStop, VolatilityTrailingStop, SignalExit, ScaledExit, RuleChain, AllOf, AnyOf, MaxDrawdownLimit, DailyLossLimit, MaxPositionsLimit, GrossExposureLimit, NetExposureLimit, PositionState, PositionAction, PortfolioState` (position rules in `ml4t.backtest.risk.position`); `from ml4t.backtest.analytics.trades import MAEMFEAnalyzer`; `from ml4t.backtest.types import Trade`; `from ml4t.backtest.execution.rebalancer import RebalanceConfig, TargetWeightExecutor`
- Data/style: `data.load_etfs()`, `load_macro`, `load_crypto_perps`, `load_crypto_premium`, `get_output_dir(19, "<name>")`; `from utils.style import COLORS, ml4t_palette, ml4t_diverging, show_plotly_with_alt`
- Frames: polars `pl.DataFrame` for trade/excursion/stress frames (03, 06, 08); pandas + statsmodels (04); LightGBM `device="cpu"`, one thread, fixed seeds (05, 07, 08); PyTorch strict deterministic mode, fail-closed on CUDA (09)

Exit rules inside `Strategy.on_data` (or Engine `position_rules`):

```python
self.exit_rules = RuleChain([StopLoss(pct=STOP_LOSS_PCT), TakeProfit(pct=TAKE_PROFIT_PCT),
                             TimeExit(max_bars=MAX_HOLDING_BARS)])
for position in broker.positions.values():
    state = PositionState(entry_price=position.avg_price, current_price=data[position.asset].close,
                          entry_time=position.entry_time, current_time=timestamp,
                          high_since_entry=position.high_water_mark, bars_held=position.bars_held)
    if self.exit_rules.evaluate(state).should_exit:
        broker.close_position(position.asset)
```

Standalone rule evaluation: `risk.PositionState(entry_price, current_price, quantity, high_water_mark, bars_held, context={...})` then `rule.evaluate(state)`. Demo helpers: `create_position_state(symbol="SPY", side="long", entry_price=100.0, current_price=100.0, bars_held=0, bar_open/high/low=None, high_water_mark=None, low_water_mark=None)`, `create_portfolio_state(equity=100000, initial_equity=100000, high_water_mark=100000, num_positions=5, positions=None, daily_pnl=0)`.

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
- `chapters/25_live_trading.md` — where kill switches, escalation and reinstatement procedures are operated.
- `chapters/26_mlops_governance.md` — drift monitoring cadence, retraining triggers and governance artifacts (SR 11-7).
- `libraries/ml4t_backtest.md` — `ml4t.backtest.risk` rules and limits, `MAEMFEAnalyzer`, `TargetWeightExecutor`.
- `libraries/ml4t_diagnostic.md` — `analyze_distribution`/`analyze_tails`, factor API, `BarrierAnalysis`, `TradeShapAnalyzer`, drift toolkit.
- `libraries/ml4t_live.md` — live enforcement of portfolio limits.
- `case_studies/etfs.md` — the ETF panels used by 01, 03, 04, 06, 07, 11.
- `case_studies/crypto_perps_funding.md` — hourly perps and premium index used by 07 and 08.
- `case_studies/sp500_options.md` — option hedging context for 09.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `workflow.md`, `companion_repo.md` — cross-cutting indexes.
- Further reading:
  - Rockafellar & Uryasev (2000), Optimization of conditional value-at-risk — CVaR and its OCE representation.
  - Buehler, Gonon, Teichmann & Wood (2019), Deep hedging, Quantitative Finance 19(8).
  - Moreira & Muir (2017), Volatility-managed portfolios; Patton (2011) and Hansen & Lunde (2006) on volatility loss functions; Bollerslev (1986) GARCH; Cont (2001) stylized facts.
  - Daniel & Moskowitz (2016), Momentum crashes; Ang & Timmermann (2011) and Khang (2022) on regimes; Shu & Mulvey (2025) regime-switching allocation.
  - Alankar et al. (2023) and Varma (2025) on drawdown rules; Zumbach & Zumbach (2025) historical stress tests; Bianchi et al. (2023) fat tails; Martin et al. (2024) minimum downside risk; Browne et al. (2023).
  - Whalley & Wilmott (1997) hedging with transaction costs; Brixton et al. (2022) stock-bond correlation; Hurst (2010) risk parity; Schwert (1989).
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
