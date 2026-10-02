# Chapter 9: Model-Based Feature Extraction

> Chapter 8 built features by aggregating observed data with deterministic transforms; this chapter builds them from **fitted procedures**: coefficients, persistence measures, latent states, filtered estimates, innovations, forecasts, conditional variances, regime probabilities and posterior uncertainty. The recipe never changes: fit a procedure on training data and use its outputs as columns. The chapter is organised by extraction method (diagnostics, signal transforms, volatility models, uncertainty, regimes, panel) because one fitted procedure yields several feature kinds (Kalman: level, trend, innovation, uncertainty; GARCH: conditional vol and persistence; regime model: probabilities, durations, transitions). Its position: point-in-time is the invariant (every estimate must be computable when it is dated, refit inside walk-forward, versioned with the features it produced); filtered beats smoothed without exception; the mean of returns is nearly unpredictable while the variance is; and uncertainty and regime outputs are conditioning features, not stand-alone signals.

## When to use this reference

- Deciding whether a price, return or volatility series needs differencing (ADF + KPSS read jointly, fractional-differencing order) before a model sees it.
- Turning diagnostics into columns: rolling stationarity statistics, detected break dates, persistence estimates, CUSUM/MOSUM monitoring.
- Extracting features from a Kalman filter, ARIMA, GARCH/EGARCH, HAR, Hurst, stochastic-volatility sampler, HMM or Wasserstein clustering and needing to know which outputs are causal.
- Auditing a feature pipeline for look-ahead: smoothed states, `predict_proba`/Viterbi, full-sample fits, whole-sample scalers, window labels written back, FFD order searched on the evaluation sample, full-sample hedge ratios.
- Choosing a volatility estimator (close-to-close vs Parkinson/Garman-Klass/Rogers-Satchell/Yang-Zhang vs intraday realized variance) and a horizon structure (HAR daily/weekly/monthly).
- Feeding a regime probability to a downstream model (single column vs mixture of experts) with the regime model refit inside every fold.
- Building pair features (Engle-Granger/Johansen, filtered hedge ratio, half-life, z-score entry) or cross-sectional ranks, benchmark-relative ratios and universe breadth/dispersion.
- Using spectral, wavelet or path-signature representations and needing the causal, non-redundant version.
- Naming model-based columns by family token or reading a case study's `features/model_based.parquet` schema.
- Running `09_model_based_features/*.py` (signatures need the `ml4t-py312` Docker image).

## Core ideas (the why)

- **Point-in-time is the invariant.** Applies identically to a rolling ADF p-value, a PELT break date, an FFD order, Kalman noise variances, a wavelet component, GARCH parameters, HMM transition matrices and cluster centroids. Every one is a fitted object; every later use is a claim about the data it was fitted to.
- **Causality is a property of a value's position, not of the construction.** A filtered GARCH or ARIMA series is causal only on the block after the training split; HMM filtered probabilities are causal in inference but not in parameters when the model was fitted on the whole sample (`07`, `08`, `11`).
- **Filtered, never smoothed.** Smoothers (Kalman RTS, HMM `predict_proba`, Viterbi, wavelet DWT) revise the past with later observations: better estimates of the past that "cannot be used as a feature at all". Smoothed regime probabilities rise *before* a transition because they have seen it (`04`, `05`, `11`, `13`).
- **Diagnostics play two roles**: they decide preprocessing (difference? model volatility?) and become columns when recomputed on a rolling window (`01`).
- **A test only gives evidence against its null.** ADF (null unit root) and KPSS (null stationary) must be read jointly; neither alone separates "stationary" from "the sample does not settle it" (`01`).
- **Differencing is not free.** First differencing discards all level information; fractional differencing keeps as much level as stationarity allows. Fix the order per asset class: a fixed order is "not optimal for any one symbol and not estimated from anything, which is what makes it safe" (`03`).
- **A break detected on the full sample is not a feature; it is look-ahead.** Refit forward; expect step functions between refits and revisions of past break positions (`02`).
- **Neither a single-break test nor a segmentation reports the "true number" of breaks.** Zivot-Andrews tests a hypothesis and dates one break; segmentation minimises a cost under a constraint you chose. A mean-shift cost on a price level finds level shifts, not crashes (`02`).
- **A supervised break detector inherits its positive class from its labels.** Synthetic labels define the class by construction; transfer to market data fails on calibration while the statistics still rank windows sensibly. Way out: feed the family statistics downstream instead of the probability (`02`).
- **Kalman: the settings are the model and their ratio is the behaviour** (Q state movement vs R observation noise). A window length is what a fixed-weight method has instead; a slope with no window still has a horizon set by the fitted settings (`04`).
- **Check what a derived feature reduces to before adding it.** Spectral energy = window variance x constant (Parseval); interval ratio = exp(2 z s) duplicates `log_forecast_std`; `expected_duration` is the state label wearing a unit (`05`, `10`, `11`).
- **A flat spectrum is a finding.** Daily index returns are near white noise; spectral columns inform about departures. A tradable weekly cycle in a liquid index is reason to suspect the calculation (`05`).
- **A forecast is a column, not an answer.** It must be computable on the session it is stamped on, must *vary* over the sessions it is scored on, and must carry something other columns do not; accuracy is secondary (`07`).
- **The mean of returns is nearly unpredictable; the variance is predictable.** ARIMA on daily returns finds almost nothing; GARCH finds persistence just below one for essentially every liquid asset. Volatility clustering (Ljung-Box on squares, ARCH-LM) is what makes it forecastable (`01`, `07`, `08`).
- **One symbol is one draw.** Walk through SPY, then read a *distribution* across the panel (quartiles, medians, share above zero), remembering ~100 ETFs with overlapping holdings are far fewer effective observations (`07`, `08`, `09`).
- **Select on one criterion, report on another, keep them separate** (AIC on train vs IC on test). Report both converged and non-converged medians rather than dropping failed fits, which selects on optimiser ease (`07`).
- **Signatures keep the geometry summary statistics discard** (level two = signed area), worth it only where path shape is the signal (`06`). **HAR's three coefficients are readable horizon weights**; GARCH gives one decay rate (`09`). **Roughness is a finding about markets and a poor feature**: Hurst on log-vol near 0.1 everywhere separates nothing (`09`).
- **An uncertainty estimate is a second feature, free with the first, only if it moves for reasons the point estimate does not.** A fixed-order ARIMA interval width tracks the level: "a volatility feature wearing an uncertainty label" (`10`).
- **Rules that need no fitting set the bar.** VIX > 20 and price vs 200-day MA are transparent, immediate and free of estimation risk; a fitted regime model must beat them at something (`11`).
- **Soft conditioning over hard switching.** Regime information works better as a conditioning feature than as a switch between models because the regime call is least certain exactly where it matters most; a misrouted session in a mixture of experts reaches a model that never saw its kind (`13`).
- **A window is a distribution.** Sorting a return window discards order on purpose; Wasserstein reads the whole shape (where losses sit), not just two moments. An unsupervised method always reports a clean separation; without ground truth a separation statistic says the algorithm ran (`12`).
- **Temporal features need cross-sectional context.** A conditional volatility of a quarter means one thing for a utility and another for a biotech: rank across the universe, measure against the benchmark, pair, aggregate (`14`).
- **Cointegration is a claim about drift, not co-movement**; a hedge ratio used to hold a position has to be filtered; a window length is a parameter and leaks like one; price the position that was held, not the column that was plotted (`14`).
- **A schema read is an inventory, not a claim**: column names say what was built, never what predicted anything (`case_study_temporal_summary`).

## Method recipes (the how)

Sequencing across the chapter: `01` diagnostics -> `02` locate breaks -> `03` difference fractionally -> `04`/`05`/`06` signal transforms -> `07` mean model (finds nothing) -> `08` variance model -> `09` multi-horizon + roughness -> `10` uncertainty -> `11`/`12`/`13` regimes -> `14` panel -> summary. Every notebook ends with a "features this notebook produces" table carrying a **Causal** column and a "Known limitations" block.

### Visual diagnostics and rolling stationarity features (`09_model_based_features/01_visual_diagnostics`)

1. Plot four panels before any test: price level (does the level wander), returns (bursts of large moves = clustering), VIX (mean-reverting), histogram vs normal with the same mean/sd.
2. Run ADF (null = unit root; `autolag="AIC"`; `nobs` = observations left after lags) and KPSS (null = level-stationary; p-value interpolated from a table bounded at 0.01 and 0.10, so read 0.01 as "at most 1%" and 0.10 as "at least 10%"). Both with constant, no trend, read at 5%.
3. Read them jointly:

| ADF rejects | KPSS rejects | Reading |
|---|---|---|
| yes | no | stationary |
| no | yes | unit root |
| yes | yes | disagreement: diagnose before transforming (trend-stationary -> refit both with trend term; structural break -> `02`; strong persistence without unit root -> tests inaccurate at nominal size) |
| no | no | sample does not settle it |

4. `analyze_stationarity` (ml4t.diagnostic) adds Phillips-Perron (ADF null, different HAC correction) and returns a consensus label: `strong_` (3/3 agree), `likely_` (2/3), `inconclusive` (even split); agreement score = share voting with the majority (1.0 or 2/3). The label collapses the "both reject" disagreement into a majority call; read it beside the per-test statistics.
5. Correlogram `plot_correlogram(series, claim, lags=40)` (series + rolling mean, Q-Q, ACF, PACF; shaded band = 5% insignificance). Slow positive ACF decay -> wandering level; PACF cut-off after p -> AR(p); ACF cut-off after q -> MA(q). `analyze_autocorrelation` proposes an ARIMA order (starting point for `07`).
6. Ljung-Box pooled over `LJUNG_BOX_LAGS = [5, 10, 20, 40]` on raw and on squared returns: raw rejection = sign not quite independent; squared rejection = *size* of next move predictable (volatility clustering). `analyze_distribution` -> four moments + Jarque-Bera (excess kurtosis vs normal = 0). `arch_lm_test` regresses squared returns on lags; rejection -> conditionally heteroscedastic -> GARCH (`08`).
7. Rolling features: `ROLLING_WINDOW = 252`, `ROLLING_STEP = 5`; each window ends the session *before* the stamp date; take critical values from each test's own output (they depend on sample size and lag order). Output `rolling_df`: `adf_statistic`, `adf_pvalue`, `kpss_statistic`, `kpss_pvalue`, `stationarity_regime`.

### Structural breaks: tests, segmentation, causal columns, monitoring (`02_structural_breaks`)

Returns = log close difference x 100 (log so multi-day return = sum of daily). Synthetic AR series use `SEED=42`.

| Tool | Call | What it answers | Caveat |
|---|---|---|---|
| Zivot-Andrews | `ZivotAndrews(series, lags=12, trend="ct", trim=0.15)` | one endogenous break in intercept and trend; null = unit root with no break | failing to reject leaves open: unit root, or more than one break |
| PELT | `rpt.Pelt(model="l2", min_size=60).fit(x).predict(pen=np.log(len(x)) * x.var())` | penalised exact segmentation (BIC-shaped penalty: variance puts it in cost units, log grows slowly) | on price level neither 2008 nor 2020 gets a line: squared error around segment mean on a price that rose several hundred percent dwarfs a drawdown |
| Binseg | `rpt.Binseg(model="l2", min_size=60).fit(x).predict(n_bkps=4)` | greedy recursive with a requested count | count is your constraint, not a finding |

`MIN_SEGMENT = 60` (~one quarter). Segment the series whose property you care about (returns or volatility for crashes, not price level).

**Causal break features** `causal_break_features(series, index)`: refit PELT on a trailing window, `LOOKBACK = 1260` (~5 years), `REFIT_EVERY = 63` (~quarterly), `LEVEL_WINDOW = 252` sessions either side averaged for the level shift. Columns: `sessions_since_break` (censored at lookback: "none in five years", not "none ever"), `level_shift_usd` (unadjusted price units; scale before cross-sample use), `breaks_in_lookback` (comparable across refits because every refit reads the same number of sessions; not comparable with a full-sample count priced against 19 years of variance). Columns are step functions between refits; break positions can move at the next refit (running estimate, not a record of events).

**Monitoring** `cusum_and_mosum(series, burn_in, bandwidths)`: reference mean and scale estimated once from `BURN_IN = 252` sessions and never re-estimated (that is what makes them causal). CUSUM = cumulative deviations from the reference mean (drifts permanently after a shift; a second shift is read against the moved baseline). MOSUM: M_t = (1/(sigma_hat sqrt(h))) sum_{i=t-h+1..t}(x_i - xbar_burnin), `MOSUM_BANDWIDTHS = [20, 50]`; a temporary excursion leaves once the window slides past; a permanent shift Delta moves the expectation to Delta sqrt(h)/sigma_hat and stays. Short bandwidth = sooner with more false crossings; long = steadier, later. No default bandwidth and no threshold drawn: set from how long a shift must persist before you would act and from the false-alarm cost.

**Supervised detector (ADIA Lab framing)**: `generate_break_series(n, break_type, magnitude=1.0, seed)` with `N_EXAMPLES = 200`, `SERIES_LENGTH = 500`, boundary at midpoint, half null, half from `BREAK_TYPES = ["mean_shift", "var_shift", "trend_shift", "autocorr_shift"]`, `AR_PHI_BEFORE = 0.2`; autocorr shift keeps marginal variance at 1 by scaling innovations by sqrt(1 - phi^2) and starts from a draw with stationary variance (a restart at zero gives a transient at the boundary). Five feature families, each `f(series, boundary) -> dict`, concatenated by `compute_all_break_features` (`FEATURE_FAMILIES` map):

| Family | Statistics | Note |
|---|---|---|
| `location_shift_features` | Welch t on raw and absolute values at `FISHER_WINDOWS = [50, 100, 250]`, combined by Fisher's method | overlapping windows -> not calibrated; monotone feature only |
| `scale_shift_features` | variance-ratio F (exact only under normality), Levene (mean absolute deviations), Fligner-Killeen (ranked deviations) | read the last two on financial data; keep all three so the classifier learns disagreement |
| `distribution_shift_features(..., n_bins=50)` | KS, Jensen-Shannon (`scipy.spatial.distance.jensenshannon`), Hellinger (bounded), Wasserstein (unbounded, data units) | |
| `dependence_shift_features` | lag-one autocorrelation of values and of squared values, before vs after | |
| `alignment_features` | distance of a free PELT estimate (`min_size=20`, `pen=np.log(n)*series.var()`) from the candidate boundary | |

Classifier: gradient-boosted trees, `CV_FOLDS = 5`, out-of-fold probabilities for ROC; family importance = sum of split counts per family (each family carries at most ~1/4 with four break kinds in equal proportion). Market transfer test `features_around(series, center_date, label)` with `PERIOD_SESSIONS = 250` each side over `PERIODS` 2008-09-15, 2020-03-11, 2011-08-05 and quiet 2015/2017/2019-06-15.

**Polars rolling equivalents** (`ml4t.engineer.features.statistics`, `ENGINEER_WINDOWS = {"short": 50, "long": 100}`): `coefficient_of_variation` on *absolute* returns (daily mean ~0 makes the raw ratio explode), `rolling_kl_divergence("returns", window=100)`, `rolling_wasserstein("returns", window=100)`, `rolling_drift("returns", window=100)`. Each compares the recent window to the one before; usable on any session without refitting.

### Fractional differencing (`03_fractional_differencing`)

- Weights of (1-L)^d: w_0 = 1, w_k = -w_{k-1} (d-k+1)/k. d=1 -> [1,-1]; d=2 -> [1,-2,1]; 0<d<1 never terminates, every weight after w_0 negative and decaying, lag-1 weight = -d. Transform = today minus a decaying weighted average of the entire past.
- Truncate at `FFD_THRESHOLD = 1e-4`; surviving weights = fixed window width. Apply to **log** prices (linear filter; scale dependence reduced, not zero). Untruncated weights sum to zero for any d>0; truncation leaves a small positive remainder, so the output carries that fraction of the level ("memory") and is only approximately stationary.
- Boundary conventions: `ffdiff` is **boundary-partial** (no nulls, early rows from a shorter filter). **Full-window** (Lopez de Prado 2018) nulls the first `width - 1` rows = **sample loss**; use this for features and publish the validity mask. `ffd_full_window(series, d, threshold=FFD_THRESHOLD) -> dict` reports diagnostics.
- Grid `D_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]`: report sample loss, ADF p-value, correlation with log price per order. Lower d = more memory (higher correlation), weaker rejection, AND wider window: memory paid for twice.
- **Fixed order per asset class** `ASSET_CLASS_D = {"equities": 0.4, "fixed_income": 0.5, "crypto": 0.5, "commodities": 0.4, "fx": 0.35}` (panel `ETF_ASSETS`: SPY, QQQ, IWM, EFA, EEM equities; TLT fixed income; GLD commodities). Persistent series get a higher order.
- Forward search only when unavoidable: `search_d_on_training(series, train_end)` over `SEARCH_GRID = [0.1 .. 1.0 step 0.1]`, `TRAIN_FRACTION = 0.8`, smallest order rejecting a unit root inside the training cut.
- Library (`ml4t.engineer.features.fdiff`): `get_ffd_weights`, `ffdiff` (boundary-partial), `find_optimal_d`, `fdiff_diagnostics`; both helpers run boundary-partial with no validity mask. Output: `ffd` column + validity mask; downstream filters on the mask.

### Kalman filter: local linear trend and hedge ratio (`04_kalman_filter`)

- Model on the price level: state x_t = [level, slope], F = [[1,1],[0,1]], w ~ N(0, Q); y_t = H x_t + v_t, H = [1, 0], v ~ N(0, R). Recursion: predict (advance state and covariance, add Q) -> innovate (innovation = y_t - H x_hat_{t|t-1}; unpredictable if the model is right; diagnostic and feature) -> update (gain K = state uncertainty / total prediction uncertainty; estimate += K x innovation; covariance shrinks). `LIKELIHOOD_BURN_IN = 20` sessions dropped from the log-likelihood (vague starting covariance dominates).
- `kalman_local_linear(prices, observation_noise, level_noise, slope_noise, burn_in=0) -> dict` (noise settings scalar or per-session arrays).
- Hand-set smoothing direction `HAND_SET = {"observation_noise": 1.0, "level_noise": 0.01, "slope_noise": 0.001}`. Under constant Q, R the covariance converges to a fixed point, the gain converges, the filter becomes fixed linear, and the uncertainty feature carries no information until the settings change.
- ML fit: `negative_log_likelihood(log_params, series)` over log variances (positivity), start `LOG_START = np.log([1.0, 0.01, 0.001])`, fitted on `TRAIN_FRACTION = 0.6`. Compare settings by level-vs-close gap (dollars) and log-likelihood.
- Walk-forward: `REFIT_WINDOW = 504`, `REFIT_STEP = 63`, bounded optimiser `LOG_BOUNDS = [(-5, 5), (-5, 5), (-15, 5)]` (slope variance may go much smaller). Readable quantity = observation/model variance ratio: >1 tracks price, <1 smooths. `schedule_from_refits(refit_frame, index, column)` holds each session's settings from the most recent completed refit; sessions before the first refit are excluded, not filtered with a guess.
- Features: `level`, `slope`, `innovation`, `uncertainty` (filtered only). Baselines: EMA and rolling OLS slope over `BASELINE_WINDOW = 20`; IC vs `FORWARD_SESSIONS = 5` forward return scored only on sessions with prior-fitted settings.
- Hedge ratio: state = [beta_t, alpha_t], random-walk transition, observation row carries x_t. `DELTA = 1e-3`, Q = delta/(1-delta) I; `PAIR_SYMBOL = "QQQ"`; `PAIR_OBS_VAR` = residual variance of a static OLS of SPY on QQQ fitted on `HEDGE_BURN_IN = 252` sessions (nothing reported inside burn-in); compare against rolling regression `HEDGE_WINDOW = 60`. `kalman_hedge_ratio(x, y, delta, obs_var) -> dict`; spread band from an **expanding** standard deviation. Output columns `beta` (+ spread).

### Spectral and wavelet features (`05_spectral_features`)

- Wavelets `wavelet_decompose(signal, wavelet, level)`: `WAVELET_FAMILIES = {"db6", "sym6", "coif3"}`, `DEFAULT_WAVELET = "db6"`, `DECOMPOSITION_LEVELS = 5` (D1 ~2-4 sessions, D2 4-8, D3 8-16, D4 16-32, D5 32-64, A5 slower). Components add back to the series; their variances give variance share per scale. **Research only**: the DWT runs forward and backward, so every component is non-causal. Sort scales explicitly (alphabetical puts `A` before `D`).
- Causal substitute: `SCALE_TO_WINDOW = {"D1": 3, "D2": 5, "D3": 10, "D4": 21, "D5": 63, "A5": 126}`; check = correlation between trailing rolling sd over the window and trailing mean |component| (`trailing(values, window, statistic)`, NaN until the window fits).
- Rolling FFT `rolling_fft_features(signal, window, target_periods)`: `FFT_WINDOW = 63`, `TARGET_PERIODS = [5, 21, 63]`, `SLOW_BINS = 2`. Outputs: spectral energy, dominant period, spectral entropy (normalised by max for the bin count -> [0,1]), low-frequency ratio (share in the two slowest bins), `energy_period_*` (raw power in the three bins around each target period). `rfft` returns each +/- pair once; half-weight the Nyquist bin at even window lengths so the total equals window energy.
- Parseval: spectral energy = window variance x constant (half the squared window length here); dropped from the feature table. Divide `energy_period_*` by the window total or it is volatility three times.
- Welch PSD: `WELCH_SEGMENT = 128`, `WELCH_OVERLAP = 64`. Spectrogram `rolling_welch_psd(signal, window=252, step=5, segment=64)` (64 = slowest resolvable period).
- Feature table: `FEATURE_WINDOWS = [21, 63, 126]` x {`spectral_entropy`, `dominant_period`, `low_freq_ratio`}; IC = Pearson vs `FORWARD_SESSIONS = 5`. Use returns, not prices (price spectrum is dominated by trend).
- Library (`ml4t.engineer.features.ml`): `rolling_entropy` (Shannon entropy in bits of the histogram of values in the window; `ENTROPY_WINDOW = 50`, `ENTROPY_BINS = 10`, max = log2(10) ~ 3.32 bits (inference)); `fourier_features` = sin/cos pairs of the **row index** at multiples of a base period (`SEASONAL_PERIOD = 252`, `SEASONAL_COMPONENTS = 3`; default base period 390 = US session minutes, so set it on daily data). Not a measurement of the series; a calendar basis for a linear model.

### Path signatures (`06_path_signatures`; Docker `ml4t-py312`)

- Signature of a d-dim path: iterated integrals S^{i1..ik}. Level 1 = total change per coordinate (d terms, already in any feature set); level 2 = d^2 terms, S^{i,j} - S^{j,i} = 2 x signed (Levy) area, unreachable coordinate by coordinate; level k costs d^k. `esig` includes the level-0 constant: depth-K signature has 1 + sum_{k<=K} d^k entries (d=3, K=3 -> 40; d=5, K=3 -> 156); the (i then j) level-two term sits at index `1 + d + i*d + j`; `esig.sigkeys` prints labels. Helpers `level_two(signature, first, second, dimension)`, `signed_area(...)`.
- **Log-signature** (`signature_features(paths, depth, log_form=True)`): same information, one term per independent direction, no redundant columns. Properties: reparameterisation invariance (window length and uneven spacing irrelevant), uniqueness up to truncation.
- Paths `build_paths(frame, window_size, horizon)`: coordinates (time rising 0->1, price/window open, volume/window open), `WINDOW_SIZE = 20`, `FORECAST_HORIZON = 5`, `SIGNATURE_DEPTH = 3`, `PATH_DIMENSION = 3`, `SYMBOLS = ["SPY", "QQQ", "IWM", "EFA"]`. The time coordinate is mandatory (without it a rise-and-return window equals a flat one at level one). The time-price level-two term = 2 x signed area between path and chord; it does not say *when* the move happened (timing appears at level three). Normalise every coordinate against the window's own first observation so the feature is computable from the session's close.
- Split `split_indices(n)` with purge `PURGE = WINDOW_SIZE + FORECAST_HORIZON - 1` = 24 at every boundary. `evaluate(features, targets, splits)`: fixed GBT, reports R^2 and rank IC; `FEATURE_SETS = ["hand-built", "log-signature", "both"]` vs `shape_features(paths)` (seven summaries: return, variability, direction change, distance travelled, distance from extremes). Importance: impurity, read per-column mean (log-signature brings ~2x the columns). `TOP_TERMS = 15`; `FLAT_RETURN = 0.002` picks two "ended flat" SPY windows that travelled furthest up and down by midpoint.
- Cross-asset: coordinates = time + one cumulative return per symbol, `CROSS_ASSET_DEPTH = 2`, `CASE_STEPS = 40`; `two_step_path(first_move, second_move)` shows orientation = order x direction; `cross_asset_paths(frame, symbols, window_size)`.

### ARIMA as feature extractor (`07_arima_features`)

- On returns d = 0 throughout (ADF confirms). ACF proposes q, PACF proposes p; on daily returns both inside the band nearly everywhere -> very low order. `correlogram(series, claim)` with `ACF_LAGS = 20`; `symbol_frame(symbol)` returns closes and returns indexed by session.
- Multi-step `forecast(h)` decays geometrically to the unconditional mean within a few steps -> constant column -> **excluded**. One-step construction: **filter** (parameters fixed from the training block, recursion run forward; one fit) vs **refit on expanding window** (one fit per session). Notebook filters via `case_studies.utils.temporal.arima_one_step_forecast(fitted, y_prefix)`, which passes `apply(endog, refit=False)` and asserts parameters did not move.
- Order grid over small (p, q); `fit_recording_convergence(series, order)` catches `ConvergenceWarning` into a grid column; lowest AIC among **converged** fits wins (`TEST_START = "2024-01-01"`). Each order is refit on train, filtered across the series, scored by Spearman IC on the test block (RMSE also reported but dominated by target variance).
- Panel `one_step_column(symbol)`: `PANEL_ORDER = (1, 0, 0)`, `MINIMUM_SESSIONS = 252`, `MAX_SYMBOLS = 0` (all); report the IC distribution and both medians (converged vs not).
- Features: one-step forecast, fitted order, residual (realised minus one-step forecast; what a volatility model reads next). All causal on the post-split block.

### GARCH conditional volatility (`08_garch_volatility`)

- Scale returns by `RETURN_SCALE = 100` so `arch`'s MLE is well conditioned (alpha, beta unchanged; omega scales by 10^4); outputs are percent per session, annualise by sqrt(252). Precheck with `arch_lm_test` (reject H0 before fitting); visual: returns, squared returns, rolling std `ROLLING_WINDOW = 21`. `MINIMUM_SESSIONS = 500`.
- GARCH(1,1): sigma^2_t = omega + alpha eps^2_{t-1} + beta sigma^2_{t-1}; alpha = reaction, beta = memory, persistence = alpha + beta must be < 1 for a long-run level omega/(1 - alpha - beta) and a half-life. Equivalent to an EWMA of squared returns with decay beta plus floor omega. EGARCH models log variance with a sign term (asymmetry: falls raise vol more than rises).
- Diagnostics on standardized residuals (return / sigma_t): squared residuals should lose autocorrelation; the distribution is typically fat-tailed -> repair the innovation distribution (Student-t), not the variance equation. VaR under normal: mean + sigma_t x quantile at `CONFIDENCE_LEVELS = [0.95, 0.99]`; count exceedances at every session against the promised rate.
- Walk-forward feature: fit on `TRAIN_FRACTION = 0.7`, then `case_studies.utils.temporal.garch11_conditional_volatility(returns, *, mu, omega, alpha, beta, backcast, gamma=0.0, bounds=None)` runs the recursion forward under fitted parameters; use the post-split block only.
- Library expressions (`ml4t.engineer.features.volatility`): `realized_volatility`, `garch_forecast` take **returns**; `ewma_volatility(close="close", span=120, normalize=False, mu=-5.0, sigma=2.0)`, `volatility_of_volatility`, `volatility_percentile_rank` (lookback default 252) take a **price** and difference internally. Notebook: `EWMA_SPAN = 120`, `PERCENTILE_WINDOW = 60`, `PERCENTILE_LOOKBACK = 252`. `garch_forecast` has a default omega that must be overridden with the fitted value.
- Features: filtered conditional volatility (causal on the held-out block only), standardized residual, volatility percentile rank (bounded, regime-comparable), volatility of volatility, persistence (one per symbol; descriptive when fitted on the whole history).

### Range estimators, HAR and rough volatility (`09_har_rough_volatility`)

- Intraday realized variance RV_t = sum of squared minute log returns from **last-trade** prices (VWAP smooths and understates), bars 09:30-16:00 only, the opening minute keeps its own open-to-close return, session open/close read from raw bars, overnight pairs validated as consecutive sessions on the `EXCHANGE = "XNYS"` calendar, `MINIMUM_BARS_PER_SESSION = 360`; `INTRADAY_SYMBOL = "AAPL"` (two years), `DAILY_SYMBOL = "SPY"`. Flag any one-session |log return| > `IMPLAUSIBLE_RETURN = 0.25` as a split (AAPL 4-for-1 found).
- Decomposition: E[r^2_cc] = E[r^2_on] + E[r^2_oc] + 2 E[r_on r_oc]; the aggregate open-to-close square differs from the sum of minute squares by all cross-products. A position held through the session is exposed to the aggregate; one traded inside it to the sum.
- Daily estimators `ESTIMATORS = ["close_to_close", "parkinson", "garman_klass", "rogers_satchell"]` + Yang-Zhang, compared by dispersion around own trailing average (`ROLLING_WINDOW = 20`). Library `garman_klass_volatility(open, high, low, close, period=20, annualize=True, trading_periods=252)` and siblings average variances before the square root. Yang-Zhang = intraday + overnight (targets the whole day, sits above the others); Parkinson/GK/RS target the session; close-to-close targets the day and puts many sessions at zero.
- HAR (Corsi 2009): RV_{t+1} = c + b_d RV^(d)_t + b_w RV^(w)_t + b_m RV^(m)_t + e, `HORIZONS = {"daily": 1, "weekly": 5, "monthly": 22}`, regressors lagged; `har_frame(frame)`, `fit_har(frame)` = OLS with Newey-West at `NEWEY_WEST_LAGS = 22`. Four SE variants (textbook, HC3, NW0, NW22): NW22/NW0 isolates the lag window, HC3/textbook isolates heteroscedasticity. Out-of-sample vs GARCH: both fit on `TRAIN_FRACTION = 0.7`, print mean forecast beside mean target, compare **correlation** (level-invariant). Rolling coefficients `REFIT_WINDOW = 504`, `REFIT_STEP = 22`; term-structure ratio = daily/monthly (>1: recent more volatile than the month), smoothed over a month.
- Hurst: `rescaled_range(series, minimum=20, maximum=None)`; DFA `detrended_fluctuation(series, minimum=10, maximum=None, order=1)` (drift-robust); `window_sizes(n, minimum, maximum)` geometric with `WINDOW_GROWTH = 1.5`, `MINIMUM_WINDOW = 10`; both report R^2 of the log-log fit; rolling `HURST_WINDOW = 252`, `HURST_STEP = 10`. Inputs: returns directly; volatility as **increments of log-volatility**. Library `hurst_exponent` (Polars) takes a **price** column: a third, different series.
- Panel `panel_row(symbol)` for symbols with >= `MINIMUM_PANEL_SESSIONS = 600`. Features: `daily`, `weekly`, `monthly` lagged range-vol averages; HAR coefficients; term-structure ratio; Hurst on returns (feature-worthy); Hurst on log-vol (nearly constant). Minute-bar RV is not a feature (2 symbols, 2 years).

### Uncertainty features (`10_uncertainty_features`; ~3 min runtime)

- Stochastic volatility: log sigma_t = mu_h + phi (log sigma_{t-1} - mu_h) + sigma_eta eta_t with Student-t observations (nu estimated). `fit_stochastic_volatility(observations)` samples the posterior (`N_DRAWS = 1000`, `N_TUNE = 2000`, `N_CHAINS = 2`; PyMC-style NUTS inferred). Refit every `REFIT_INTERVAL = 63` on the preceding `TRAIN_DAYS = 252`. Target = Garman-Klass vol with `VOLATILITY_WINDOW = 21`, returns in percent.
- Features per refit: posterior means of phi (`persistence`) and sigma_eta (`vol_of_vol`) (`FEATURE_PARAMETERS`), plus `posterior_std`, `interval_width`, `relative_width` of the filtered volatility at the last training session. The level itself goes stale within the quarter; take the level from the GARCH filter instead.
- Go/no-go: zero divergences, R-hat <= `R_HAT_CEILING = 1.01`, ESS >= `EFFECTIVE_SAMPLE_FLOOR = 200` on `DIAGNOSTIC_PARAMETERS = ["phi", "sigma_eta", "mu_h", "nu"]`; then `sampling_error_row(name, column)`: max Monte Carlo SE across refits vs between-refit spread; a ratio that is a meaningful fraction of one makes the column unreliable at that resolution. Remedies: non-centered parameterisation, much longer chains (draws scale with the square of the error reduction), or a state-space sampler.
- ARIMA prediction interval on **log** GK volatility: `ARIMA_WINDOW = 252`, order chosen by information criterion on the first window and frozen, `CONFIDENCE = 95`, `Z_SCORE = 1.959964`, `NIXTLA_FREQUENCY = "B"`. Back-transform: e^mu = median, e^{mu + s^2/2} = mean; exponentiate the endpoints for exact coverage; check empirical coverage vs 95%. Interval ratio = exp(2 z s) is a function of `log_forecast_std` alone and original-scale width = ratio x level, so both are dropped. Keep `log_forecast_std` and `median_forecast` (every session).

### HMM regimes (`11_hmm_regimes`)

- Baseline rules: VIX > `VOLATILITY_INDEX_THRESHOLD = 20` (stress) and price below the `TREND_WINDOW = 200` MA (downtrend); keep only rows where all series are defined (a comparison with a missing MA is False, not null). Inputs to the HMM: log return and its `VOLATILITY_WINDOW = 21` rolling std.
- Forward algorithm (filtered P(state_t | o_{1:t})): alpha_1(k) = pi_k b_k(o_1); alpha_t(k) = b_k(o_t) sum_j alpha_{t-1}(j) A_jk; normalise each step. `forward_algorithm(observations, transition, means, deviations, initial)`; toy `sample_toy_path(rng)` (2 states, 10 steps). hmmlearn `predict_proba` is smoothed and `predict` is Viterbi, so filtered probabilities come from `case_studies.utils.temporal.filtered_state_probs(model, X)`.
- Three estimation problems: (1) local optima -> `fit_hmm_restarts(X, *, n_states=2, random_state=42, n_iter=200, tol=0.01, n_restarts=1, reject_unstable_rel_tol=None)` over `N_INITS = 10`, or `fit_hmm_kmeans_init(X, n_states=2, random_state=42, n_iter=200, tol=0.01)`; (2) state count -> BIC with `parameter_count(n_states, n_series)` over `STATE_COUNTS = [2, 3, 4]` ((K-1) initial + K(K-1) transition + K d means + K d(d+1)/2 covariances (inference)), then judgement -> `N_STATES = 2`; (3) label switching -> `sort_states_by_variance(model)` / `sort_states_by_mean(model, dim=0)` + `relabel_states(states, probs, order)`.
- Features: `filtered_stressed` (P of the higher-variance state), `state` = argmax of filtered probabilities (never `predict`), `state_entropy` (the model's own uncertainty; a downstream model can learn to ignore the probability when entropy is high), `expected_duration` = 1/(1 - A_kk), `switching_stressed` from a statsmodels Markov switching AR (second specification; agreement between two differently specified models beats either's confidence), `stressed_by_index`, `below_average` (fully causal, nothing estimated).
- Rule-based trend classifier `market_regime_classifier` (ADX + choppiness; -1 = range-bound) with `INDICATOR_WINDOWS = {"choppiness": 14, "hurst": 100, "efficiency": 20, "trend": 30}`; it classifies trend, not volatility. Carry both labels.

### Wasserstein regime clustering (`12_wasserstein_regimes`; Horvath et al. 2021)

1. Lift: `lift_stream(returns, window_len=21, overlap=5) -> LiftedStream` (raw windows, sorted windows, start index); step = 16 sessions. Longer window = better distribution estimate, later reaction; overlap buys windows, not information.
2. Distance: for equal-size sorted samples W_p = ((1/n) sum |a_(i) - b_(i)|^p)^(1/p); sorting is the whole computation. Barycenter: p=1 quantile-wise median, p=2 quantile-wise mean; `WASSERSTEIN_P = 1.0`. `wasserstein_distance_1d`, `distances_to_centroid`, `wasserstein_barycenter_1d`.
3. Cluster: `WassersteinKMeans1D(n_clusters=2, p=1.0, n_init=10, max_iter=100, tol=1e-4, random_state)`; `fit(sorted_segments) -> ClusteringResult`; `predict(sorted_segments, centroids)` for centroids fitted elsewhere; k-means++ seeding under W_p, Lloyd iterations, keep lowest inertia over `N_INIT = 10`.
4. Baseline: `moment_features(segments, n_moments)` (raw moments scaled by 1/k!, then standardized), `fit_moment_kmeans` and `fit_moment_mixture` -> `MomentClustering.predict(segments, n_moments)`; `N_MOMENTS = 4` for the benchmark, `N_MOMENT_BASELINE = 2` for `tail_divergence` (inference).
5. Relabel by `order_by_dispersion(segments, labels, n_clusters)` so the highest number is the most volatile (same rule as `11`).
6. Validate on simulation `simulate_two_regime_stream` (`N_STEPS = 2000`, `N_SWITCHES = 6`, `CALM = {mu 0.0005, sigma 0.01}`, `STRESSED = {mu -0.0003, sigma 0.025}`, `SEED = 42`; window truth = majority regime) with adjusted Rand plus within/between MMD (`maximum_mean_discrepancy` with `median_kernel_width(sample, rng, n_pairs=2000)`, `bootstrap_discrepancy(..., n_bootstrap=200, sample_size=200)`); silhouette deliberately not reported.
7. Stamp with `label_at_window_close(starts, window_len, labels, n_sessions)`: a window over t..t+h-1 is complete at close of t+h-1; hold until the next close; NaN before the first close.
8. Forward fit on the index (daily log returns 1980-2024): fit centroids on the first `TRAIN_FRACTION = 0.6` of windows, `predict` the rest, drop the fitted block. Evaluate only post-split: earn each label against the *next* session's return, hold one session, no costs. Distribution distance between clusters via `compute_wasserstein_distance` (permutation-calibrated threshold) and `compute_psi` (`ml4t.diagnostic.evaluation.drift`).
9. Columns: `wasserstein_cluster`, `cluster_distance` (large = environment the fitted block did not contain), `tail_divergence` (1 where Wasserstein and two-moment assignments disagree).

### Regime as a feature vs mixture of experts (`13_regime_as_feature`)

- Data: SPY closes joined with VIX, 2005-01-01..2024-06-30. `BASE_FEATURES = ["returns", "volatility", "momentum_short", "momentum_long", "index_deviation"]` (`VOLATILITY_WINDOW = 21`, `MOMENTUM_WINDOWS = (21, 63)`, VIX relative to its recent level); target = sum of next `FORECAST_HORIZON = 5` returns.
- Per fold of `TimeSeriesSplit(n_splits=5, gap=5)` (inference on kwargs): fit a 2-state HMM on training returns + volatility from `N_RESTARTS = 3` starts (keep highest likelihood; a restart column counts failures) -> renumber by variance so `VOLATILE_STATE = 1` -> forward-filter through the test block with training parameters -> refit scaler on train -> fit model -> score. Result: five `probability_of_the_volatile_state` columns that disagree on shared sessions.
- Designs: baseline (base features); regime as feature (+ filtered probability); mixture of experts (one model per hard state, test sessions routed by own state, a state with < `MINIMUM_REGIME_SESSIONS = 20` training rows falls back to the other). `ESTIMATORS = {"ridge": Ridge(alpha=1.0), "gradient boosting": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)}`; `evaluate_single_model(estimator, features, regime_column)`, `evaluate_mixture_of_experts(estimator, features)`. Paired difference per fold: Var(A-B) = Var(A) + Var(B) - 2 Cov(A,B); direction and magnitude only. Last-fold diagnostics: error by state vs the target's own std per state; impurity importance.
- Polars catalog block: `CATALOG_WINDOW = 20`, `HURST_WINDOW = 100`, `CHOPPINESS_WINDOW = 14`, `ENTROPY_WINDOW = 50`; `regime_conditional_features("returns", "trend_regime", regime_values=[-1, 0, 1])` -> `feat_bear`, `feat_neutral`, `feat_bull` (feature x indicator); names assume directional states but `market_regime_classifier` values mean range-bound/transitional/trending.

### Pairwise and cross-sectional panel features (`14_panel_features`)

- Data: `load_etfs` 2015-2024, `MAX_SYMBOLS = 0`; pair `("XLE", "USO")` else `("SPY", "QQQ")`; `MINIMUM_SESSIONS = 252` joint sessions.
- Cointegration (full sample, a pre-trading screen, not a feature): Engle-Granger `coint(dependent, independent, trend="c")[1]` (asymmetric in roles) and `adfuller(spread_static, autolag="AIC")`; Johansen `coint_johansen(X, det_order=0, k_ar_diff=1)` (symmetric, counts stationary combinations; implied ratio from the leading eigenvector is meaningless when none exists). `SIGNIFICANCE = 0.05`.
- Hedge ratio: static `LinearRegression().fit(independent, dependent).coef_[0]`; filtered via filterpy `KalmanFilter(dim_x=2, dim_z=1)`, state [intercept, ratio], `F = I`, `x0 = [0, 1]`, `P0 = I`, `R = [[MEASUREMENT_NOISE = 1e-3]]`, `Q = I * PROCESS_NOISE = 1e-5`, `H_t = [[1, independent_t]]`, `predict()` then `update(dependent_t)` each session. Tuning is Q/R: larger Q tracks structural change sooner but chases noise.
- Half-life: AR(1) in change form delta s_t = a + phi s_{t-1} + e; phi < 0 -> half-life = -ln 2 / ln(1 + phi); phi >= 0 -> none. `estimate_half_life(spread)` on the first `TRAIN_FRACTION = 0.5` only. Signal window `LOOKBACK = max(int(2 * half_life_train), MINIMUM_LOOKBACK = 20)`; z = (spread - rolling mean)/rolling std; enter long at z <= -`ENTRY_THRESHOLD = 2.0`, exit when z crosses back through 0; step the position forward one session at a time. P&L over t = dP_t - beta_{t-1} dQ_t over committed capital P_{t-1} + |beta_{t-1}| Q_{t-1} (inference on timing); no costs, slippage or borrow.
- Screen `CANDIDATE_PAIRS`: GLD/SLV, XLE/USO, QQQ/SMH, SPY/VTI, TLT/IEF, EEM/VWO; both tests at 0.05, half-life per pair.
- Library pairwise (`ml4t.engineer.features.cross_asset`, inference on module): `rolling_correlation("first_return", "second_return", window=60)`, `beta_to_market("second_return", "first_return", window=60)` (returns), `co_integration_score("first_close", "second_close", window=120)` (prices), `correlation_regime_indicator("correlation")` -> dict of `corr_regime*` flags. Rolling scores describe a window; they do not replace the full-sample tests. `LIBRARY_PAIR = ("GLD", "SLV")`.
- Cross-sectional transforms per date over `RANK_SYMBOLS` (SPY, QQQ, IWM, EFA, EEM, TLT, GLD, XLE, XLF, XLV), momentum over `FEATURE_WINDOW = 60`:

| Transform | Keeps | Discards | Use when |
|---|---|---|---|
| rank (integer) | ordering | distances; bounded by asset count | "which assets" |
| percentile (rank/count) | ordering, comparable across universe sizes | distances; manufactures distinctions between near-identical assets | comparing universes |
| z-score | distances, outliers | robustness | "how much" |

- Benchmark-relative (`BENCHMARK = "SPY"`, `COMPARED = "XLE"`): subtract additive quantities (momentum), divide scale quantities (volatility / SPY volatility; >1 = moved more than the market).
- Universe aggregates of a per-asset measure ranked within trailing `REGIME_WINDOW = 252` via `rolling_rank`: mean (level), breadth (fraction above `STRESS_THRESHOLD = 0.7`), dispersion (cross-asset std). Breadth near 1 with dispersion near 0 = universe-wide stress; high dispersion = a split universe.

### Case-study schema inventory (`case_study_temporal_summary`)

- `pl.scan_parquet(get_case_study_dir(cs) / "features/model_based.parquet").collect_schema()` reads the footer only, for `CASE_STUDIES` = etfs, crypto_perps_funding, nasdaq100_microstructure, sp500_equity_option_analytics, us_firm_characteristics, fx_pairs, cme_futures, sp500_options, us_equities_panel. Exclude `IDENTIFIER_COLUMNS = {timestamp, date, symbol, product, asset, stock_id, instrument_id}`; attribute each column to every `FAMILY_TOKENS` key its name contains (multi-counting allowed); read the unmatched count first; list `SAMPLE_NAMES = 8` names per artifact. A fresh checkout reports an empty inventory (artifacts are pipeline outputs, not committed).

### Feature columns by notebook and causal status

| Notebook | Columns | Causal? |
|---|---|---|
| 01 | `adf_statistic`, `adf_pvalue`, `kpss_statistic`, `kpss_pvalue`, `stationarity_regime` (252/5 rolling) | yes, window ends before stamp |
| 02 | `sessions_since_break`, `level_shift_usd`, `breaks_in_lookback`; CUSUM/MOSUM; rolling KL/Wasserstein/drift/CV | yes (refit forward / fixed burn-in) |
| 03 | `ffd` + validity mask | yes with fixed order and full-window mask |
| 04 | `level`, `slope`, `innovation`, `uncertainty`, `beta` | filtered only; after first refit |
| 05 | `spectral_entropy`, `dominant_period`, `low_freq_ratio`, normalised `energy_period_*` at 21/63/126 | yes; wavelet components no |
| 06 | log-signature terms, time-price and price-volume level-two terms, cross-asset level-two | yes (window-normalised) |
| 07 | one-step forecast, fitted order, residual | post-split block only |
| 08 | filtered conditional vol, standardized residual, vol percentile rank, vol-of-vol, persistence | vol post-split only; rank/vov throughout; persistence one per symbol |
| 09 | `daily`/`weekly`/`monthly` RV averages, HAR coefficients (504/22), term-structure ratio, rolling Hurst | yes; log-vol Hurst nearly constant |
| 10 | `persistence`, `vol_of_vol`, `posterior_std`, `interval_width`, `relative_width` (per refit); `log_forecast_std`, `median_forecast` (daily) | per refit where MC-SE << spread |
| 11 | `filtered_stressed`, `state`, `state_entropy`, `expected_duration`, `switching_stressed`, `stressed_by_index`, `below_average` | inference yes, parameters no unless refit per fold; rules fully causal |
| 12 | `wasserstein_cluster`, `cluster_distance`, `tail_divergence` | post-train block, stamped at window close |
| 13 | `probability_of_the_volatile_state` per fold | yes (fold-local fit, forward filter) |
| 14 | filtered hedge ratio, spread z, ranks/percentiles/z-scores, benchmark-relative, breadth/dispersion | yes; cointegration tests are a screen, not a feature |

## Guardrails and pitfalls

Look-ahead and stamping
- **Full-sample break detection as a feature** — PELT on the whole series marks 2012 with twelve unseen years; refit on trailing `LOOKBACK=1260` every `REFIT_EVERY=63` and stamp from the last completed refit; expect step functions and revisions.
- **Rolling-window stamp alignment** — a window containing the stamp date leaks that session; end each window the session before the stamp (`01`); each refit uses sessions up to the refit date only (`02`, `04`).
- **Smoothed Kalman states, wavelet components, HMM `predict_proba`/Viterbi as features** — all read future observations; smoothed regime probabilities rise before transitions and get transition sessions right for free; use filtered states, `filtered_state_probs`, argmax of filtered probabilities; measure the share of sessions where filtered and smoothed disagree; use the DWT only to pick scales then build rolling features at `SCALE_TO_WINDOW` lengths.
- **FFD order searched on the evaluation sample** — even a "single predetermined step up the grid" triggered by a full-sample test is the test data choosing; fix `ASSET_CLASS_D` or `search_d_on_training` on the 80% cut.
- **Kalman variances / GARCH / ARIMA / HMM / centroids fitted on the whole sample** — settings are fitted objects; the filtered series inside the training block used parameters estimated from those values; fit on the training cut, refit forward, mark and use only the post-split block; refit the HMM per fold (`case_studies/etfs/04_model_based_features.py`); fit centroids on the first 60% of windows and drop the fitted block.
- **`apply(endog, refit=True)` on the prediction block** — in-sample fit masquerading as a forecast; use `arima_one_step_forecast` (passes `refit=False`, asserts unchanged parameters).
- **Hedge-ratio observation variance or regression ratio from the whole pair** — later prices enter the gain that produced every earlier ratio; the full-sample number was unavailable on every date but the last; set `PAIR_OBS_VAR` from a static OLS on `HEDGE_BURN_IN=252`, filter the ratio, hold beta_{t-1}; use an expanding spread band.
- **CUSUM/MOSUM reference re-estimated** — not causal; fix mean and scale on `BURN_IN=252`.
- **Window label written back across covered sessions** — puts returns up to t+h-1 on session t; use `label_at_window_close`; verify the first label index >= `WINDOW_LEN - 1`.
- **Earning a regime label against the same session's return** — the label is known only at that close; earn against the next session.
- **Signal window length chosen from the whole sample** — a length is a parameter and leaks invisibly; estimate half-life on the first 50% only.
- **Ranking an asset's volatility over its whole history** — places today against years not yet happened; use `rolling_rank` with trailing `REGIME_WINDOW=252`.
- **No gap between training and test blocks / scaler fitted on the whole sample** — targets reach `FORECAST_HORIZON` ahead; use `gap=FORECAST_HORIZON` and refit the scaler per fold on the training block.
- **Overlapping-window leakage in signatures** — training targets drawn from test sessions; purge `WINDOW_SIZE + FORECAST_HORIZON - 1` = 24 rows at every boundary; normalise coordinates against the window's own first observation.
- **Overlapping windows read as independent draws** — 5-session forward windows share four of five days; rolling refits overlap; standard errors are "several times too small"; claim no significance from IC or consecutive rolling test values.

Estimation
- **Boundary-partial FFD values in a feature column** — silent regime change at the start; null the first `width - 1` rows and publish the mask and sample loss. **`find_optimal_d` / `fdiff_diagnostics`** run boundary-partial with no mask and on the nine-year sample picked an order whose window exceeded the sample; check implied width against sample length.
- **Weak identification of level vs slope variance (Kalman)** — the likelihood surface is nearly flat along that trade; use `LOG_BOUNDS` and report how many refits sit on the observation-variance floor (the bound is the only thing between the filter and the identity function: a modelling choice, not a result).
- **Uncertainty feature under fixed Kalman settings / constant-variance assumption** — converges to a constant; prefer time-varying observation noise from volatility models.
- **Both-reject ADF/KPSS read as "trend" / KPSS p-value boundaries / break inside a test window** — diagnose (trend term, break via `02`, persistence) before transforming; 0.01/0.10 are table limits; a break inside a window gives rejection without location.
- **Mean-shift segmentation on price level to find crashes** — level drift dominates drawdowns; segment returns/volatility with the right cost.
- **Synthetic-label detector calibration on market data** — 500 constant-variance steps with one change vs real windows with a dozen comparable changes: nearly every real window scores positive; label real history against dated events or feed the family statistics (or `ml4t.engineer` rolling stats at 50/100) directly.
- **Fisher-combined p-values from overlapping windows / F-test on returns / `coefficient_of_variation` on raw returns** — not calibrated (monotone feature only); exact only under normality (read Levene, Fligner); mean ~0 explodes the ratio (use absolute returns).
- **Silencing `ConvergenceWarning` / dropping non-converged fits** — AIC from a non-converged fit is not comparable; catch it into a grid column and report both medians.
- **Fitting `arch` on fractional returns / without an ARCH effect / persistence >= 1** — badly conditioned MLE (multiply by 100); nothing to model (run `arch_lm_test` first); long-run variance and half-life undefined though the one-step recursion remains valid.
- **Normal innovations with fat-tailed residuals** — VaR exceedances above the promised rate, worse at 99%; GARCH-normal standardized residuals are still heavy-tailed; switch to Student-t.
- **VWAP minute prices / session open from a reduced frame / overnight pairs by bar counts / unadjusted prices across a split** — VWAP understates RV materially; the open becomes the second minute; a missing weekday looks like a holiday; a 4-for-1 split reads as -75%; use last-trade prices, raw-bar open/close, the XNYS calendar, and flag |log return| > 0.25.
- **Textbook OLS SEs on a volatility regression / negative HAR coefficient read as "that horizon predicts less" / refitting HAR daily** — heteroscedastic, autocorrelated residuals (HC3 + NW22); usually collinearity, coefficients sum to ~1 (read the three together); daily variation is estimation noise (504/22).
- **Hurst without its R^2 / library `hurst_exponent` on prices vs manual on returns or log-vol** — a slope through points not on a line estimates nothing; a few hundredths between methods is noise; three different series give three answers.
- **Reading a posterior mean when the sampler failed / reading the nu posterior mean / back-transforming a log-scale sd** — divergences, R-hat > 1.01 or ESS < 200 means the mean is not the posterior's; nu is bounded below at 2 so a broad right tail means "normal not ruled out"; nothing is a standard deviation after exponentiation, exponentiate endpoints only.
- **Carrying the SV posterior level between quarterly refits** — by quarter-end it describes a session three months old; carry parameters per refit, take the level from GARCH.
- **Single EM start / unsorted HMM states or clusters across refits / state count by in-sample BIC** — stops at the first local maximum (10 restarts or k-means init); state 0 changes meaning (sort by variance or `order_by_dispersion`, assert the higher label has higher spread); more states always find more clustering (state the judgement: two, calm/stressed).
- **Kernel width far from the data scale / comparing clusterings by inertia or silhouette** — MMD collapses; each flatters the method that optimizes in its geometry; median heuristic from 2000 pairs (print sigma) and MMD plus adjusted Rand where truth exists.
- **Treating correlation as cointegration / trading the Johansen ratio with no cointegration / screening six pairs at 0.05 / rolling cointegration score as a test** — daily co-movement with roll-cost drift never comes back; the leading eigenvector is the least badly behaved direction in a system with none; family-wise false rejection far above 0.05 (demand it hold on unseen data); a window score describes a window.

Feature semantics and units
- **Spectral energy beside rolling variance / raw `energy_period_*` / raw entropy across window lengths / even-window Nyquist bin / spectrum of prices / window spanning a vol break** — same column twice (Parseval); volatility three times (divide by total); grows with bin count (normalise); data-dependent total (half-weight); trend-dominated (use returns); averages two regimes (stationarity prerequisite).
- **`fourier_features` default period 390 and name confusion with `rolling_entropy`** — minute-bar default is wrong on daily data (pass 252); neither is spectral entropy.
- **Comparing Kalman slope IC against 20-session OLS slope IC** — different horizons; fix the horizon first.
- **`level_shift_usd` across decades / `breaks_in_lookback` vs full-sample count / quarterly refit blind spot / monitoring thresholds** — scale price units; compare only equal-length refits; a break just after a refit is invisible for a quarter (shorten `REFIT_EVERY` or pair with MOSUM); thresholds depend on false-alarm cost.
- **Multi-step ARIMA forecast / ranking return models on RMSE / selecting on the reported metric** — decays to the mean (one-step only); a constant forecast wins RMSE and has undefined IC (use rank IC); select on AIC (train), report IC (test).
- **Persistence as a cross-sectional feature / raw volatility level as conditioning / unsmoothed term-structure ratio / `expected_duration` per session** — every liquid symbol sits just below one; levels move an order of magnitude 2012-2020 (prefer percentile rank or the HAR ratio); a single session dominates the numerator (smooth over a month); as many distinct values as states.
- **Mixing estimators with different targets / ranking HAR vs GARCH by squared error / GARCH long-horizon forecasts** — day-target vs session-target levels differ by construction; level difference is charged as inaccuracy (compare correlations); beyond one step the forecast is mean reversion the notebook never validates.
- **Interval width as an uncertainty feature** — scales with the level; keep `log_forecast_std` and `median_forecast`, run the coverage check.
- **Full signature instead of log-signature / depth too deep / omitting the time coordinate / reading cross-asset level-two sign as "who led"** — redundant columns; d^K growth; rise-and-return collapses onto flat; orientation mixes order with direction (report counts of positive areas).
- **Passing returns to a price-reading Polars expression (or vice versa)** — a plausible column with the wrong meaning and no error. Returns: `realized_volatility`, `garch_forecast`, `rolling_entropy`, `rolling_correlation`, `beta_to_market`. Prices: `ewma_volatility`, `volatility_of_volatility`, `volatility_percentile_rank`, `hurst_exponent`, `choppiness_index`, `market_regime_classifier`, `volatility_regime_probability`, `co_integration_score`. `garch_forecast`'s default omega must be overridden.
- **Misreading `regime_conditional_features` names / counting non-zero interaction values as regime counts / sparse interaction columns** — `feat_bear/neutral/bull` sit on unsigned -1/0/1 (assert the pairing); a zero return undercounts; the classifier calls almost every session transitional, so report session counts per regime.
- **Comparing a missing moving average / collapsing volatility and trend regimes / reading agreement rate between regime labels** — null comparison is False (drop warm-up rows); carry both labels; most agreement is both saying "no" (read co-occurrence against independence).
- **`tail_divergence` disagreement rate near 0 or 0.5 / trusting a clean unsupervised separation / mixed windows at switches** — constant or unrelated (print the rate); separation says the algorithm ran (validate downstream, carry `cluster_distance`); expect a lag of a window plus a step.
- **Splicing out cash sessions / averaging conditional and rule statistics** — per-time statistics overstated; name each column for its stream; compare rules against rules on the same sessions.
- **Reading a single averaged error with no spread / error by regime as a model finding / near-zero regime importance as "uninformative" / mixture of experts with thin regimes** — fold-to-fold error moves ~10x more than design-to-design; the target is more dispersed in the volatile state by about as much; `volatility` already carries the information; require 20 training rows per expert.
- **Differencing the spread or the two funds' returns for P&L / uncosted spread backtest / months-long half-life with a 2-sigma band** — mixes price moves with ratio changes; equal-dollar is not the share hedge; returns are upper bounds (short leg assumed free); a couple of entries held through most sessions is a slow directional bet.
- **Treating percentiles as distances / subtracting a benchmark's volatility** — evenly spaced by construction; market stress multiplies (divide).
- **Treating ~100 ETFs as 100 votes / no costs, turnover or capacity anywhere in this chapter** — overlapping holdings; an IC difference says nothing about tradability (evaluate in the backtest pipeline).
- **Reading a schema inventory as a contribution / growing `FAMILY_TOKENS` until nothing is unmatched / empty inventory as "no features"** — names say nothing about predictive value; the check becomes a record of whatever exists; the stage may not have run.

## Decision rules and defaults

| Method | Defaults and thresholds |
|---|---|
| Stationarity | ADF + KPSS at 5%, constant, no trend; 4-row matrix; diagnose double rejection before transforming; rolling window 252, step 5, stamp the session after the window closes |
| Autocorrelation / vol clustering | Ljung-Box lags 5/10/20/40 on raw and squared; squared rejection or ARCH-LM rejection -> GARCH |
| Zivot-Andrews | `lags=12`, `trend="ct"`, `trim=0.15`; non-rejection ambiguous -> segmentation |
| Segmentation | `model="l2"`, `min_size=60`, PELT `pen=log(n)*var` or Binseg `n_bkps=4`; segment the series whose property you care about |
| Causal breaks | `LOOKBACK=1260`, `REFIT_EVERY=63`, `LEVEL_WINDOW=252`; report censoring; scale `level_shift_usd` |
| Monitoring | `BURN_IN=252` fixed reference; bandwidths 20/50 shown, choose from required persistence; thresholds from false-alarm cost |
| Break classifier | 200 synthetic series of length 500, GBM, 5-fold CV; prefer feeding family statistics or rolling stats (50/100) over the probability |
| FFD | `FFD_THRESHOLD=1e-4`; grid 0.1-0.6; equities 0.4, fixed income 0.5, crypto 0.5, commodities 0.4, FX 0.35; full-window mask; if searching: grid 0.1-1.0 on 80% train, smallest passing d |
| Kalman trend | ML fit on 60% train; walk-forward 504/63 with bounds [(-5,5),(-5,5),(-15,5)]; likelihood burn-in 20; filtered only; exclude pre-first-refit sessions; baseline window 20 |
| Kalman hedge | `DELTA=1e-3` (nb 04) or `R=1e-3`, `Q=1e-5*I` (nb 14, Q/R = 0.01); obs var from 252-session burn-in OLS; compare vs 60-session rolling OLS; expanding spread sd |
| Spectral | returns only; feature windows 21/63/126; FFT window 63, targets 5/21/63, `SLOW_BINS=2`; normalise entropy; drop energy; divide `energy_period_*`; Welch 128/64; spectrogram 252/5/64; wavelet db6, 5 levels, scale->window {3,5,10,21,63,126} |
| `rolling_entropy` / `fourier_features` | window 50, bins 10 / period 252, 3 components on daily data |
| Signatures | depth 2 log form to start; include time coordinate; d=3 depth 3 = 40 raw terms; purge = window + horizon - 1; 20 sessions is the short end of useful windows; only where path shape is the signal |
| ARIMA | d=0 on returns; small (p,q) grid; lowest AIC among converged; filter one step with `refit=False`; panel order (1,0,0); >= 252 sessions |
| GARCH | returns x100; ARCH-LM must reject; >= 500 sessions; alpha+beta < 1 to quote long-run level; VaR at 95/99 with exceedance counts; train 0.7; percentile rank 60-window/252 lookback; EWMA span 120; Student-t innovations |
| Volatility estimator | range-based (GK) over close-to-close for daily; Yang-Zhang when the whole day is the target; never mix day- and session-target estimators; RV from last-trade minute bars with >= 360 bars/session |
| HAR | horizons 1/5/22; Newey-West lags 22; refit 504/22; term-structure ratio smoothed over a month; compare with GARCH by correlation only |
| Hurst | geometric windows from 10 growing x1.5; rolling 252, step 10; report R^2; returns ~0.5 (feature), log-vol ~0.1 (not a feature); panel >= 600 sessions |
| MCMC | 0 divergences, R-hat <= 1.01, ESS >= 200, MC-SE / between-refit spread well below one; refit 63 on 252; 2 chains x 1000 draws after 2000 tune |
| Uncertainty interval | fit on log vol; order by IC on first 252 window, frozen; check 95% coverage; keep `log_forecast_std` + `median_forecast` only |
| HMM | baselines VIX > 20 and price < 200-day MA; 2 states, 10 restarts (3 per fold in nb 13), 200 EM iterations, tol 0.01; sort by variance; filtered probabilities + entropy + duration; refit per fold; trend classifier alongside |
| Wasserstein | window 21, overlap 5 (step 16), 2 clusters, p=1, 10 inits, 100 iterations, tol 1e-4, train 0.6; MMD 200 bootstraps of 200, sigma from 2000 pairs; carry `tail_divergence` only when its rate is strictly between ~0 and ~0.5 |
| Regime as feature | horizon 5, vol window 21, momentum 21/63, 5 splits with gap 5, Ridge alpha 1.0, GBR 100 trees depth 3; expect R^2 <= 0; single model with a regime column when regimes are thin, mixture of experts only with >= 20 training rows per state |
| Catalog expressions | hurst period 100, choppiness 14, realized vol 20, `garch_forecast(alpha=0.1, beta=0.85, horizon=1)`, entropy 50/10, `volatility_regime_probability` period 20 |
| Pairs | both tests at 0.05, Johansen and Engle-Granger ideally agreeing, half-life of a few sessions not months, >= 252 joint sessions; `LOOKBACK=max(int(2*half_life_train), 20)`; enter z <= -2, exit on crossing the mean; hold beta_{t-1}; correlation/beta window 60, cointegration score 120 |
| Cross-sectional | rank for "which", z-score for "how much", percentile across universe sizes; subtract additive, divide scale; `REGIME_WINDOW=252`, `STRESS_THRESHOLD=0.7`; carry breadth and dispersion together |
| IC evaluation | forward 5 sessions, Pearson (01/04/05) or Spearman (07); only sessions with prior-fitted settings; read sign before size; no significance from overlapping windows |

X-vs-Y choices:
- **Filter vs refit (ARIMA/GARCH)**: filter under train-block parameters when the model is small and parameters barely move; refit on an expanding window only when dynamics genuinely change. Both are causal; whole-sample or `refit=True` on the prediction block is not.
- **HAR vs GARCH**: HAR for readable horizon weights, a cheap OLS fit and a range-based target; GARCH for close-to-close variance or a standard risk model. Tracking quality is similar.
- **Aggregate open-to-close square vs sum of minute squares**: held through the session -> aggregate; traded inside it -> sum.
- **Signature vs simple feature**: if the signal is in the level or total return, a signature is a longer way of writing a number already available.
- **Sampler posterior width vs ARIMA interval width**: the posterior width changes because the data changes what is knowable (quarterly, minutes per fit); the ARIMA interval is free every session but nearly constant in relative terms.
- **Two HMM states vs more**: two keeps labels interpretable across refits; more fit better in sample without being actionable.
- **Volatility regime vs trend regime**: carry both separately.
- **Clustering vs rolling-std threshold**: with two clusters they agree except near boundaries; clustering buys a construction that extends (three clusters, skew-driven regimes) with no threshold to choose.

## Code patterns and APIs

Imports
- `ml4t.diagnostic.evaluation.stationarity.analyze_stationarity`; `.autocorrelation.analyze_autocorrelation`; `.distribution.analyze_distribution`; `.volatility.arch_lm_test`; `ml4t.diagnostic.evaluation.drift.compute_psi, compute_wasserstein_distance`; `ml4t.diagnostic.metrics.pooled_ic`; `ml4t.diagnostic.logging.configure_logging, LogLevel`.
- `ml4t.engineer.features.statistics`: `coefficient_of_variation`, `rolling_drift`, `rolling_kl_divergence`, `rolling_wasserstein`. `ml4t.engineer.features.fdiff`: `get_ffd_weights`, `ffdiff`, `find_optimal_d`, `fdiff_diagnostics`. `ml4t.engineer.features.ml`: `fourier_features`, `rolling_entropy`, `regime_conditional_features`. `ml4t.engineer.features.volatility`: `garman_klass_volatility`, `ewma_volatility`, `bollinger_bands(close, period=20, nbdevup=2.0, nbdevdn=2.0)`, `conditional_volatility_ratio(returns, threshold=0.0, period=20)`, `realized_volatility`, `garch_forecast`, `volatility_of_volatility`, `volatility_percentile_rank`, parkinson, rogers_satchell, yang_zhang, close_to_close. `ml4t.engineer.features.regime`: `choppiness_index`, `variance_ratio`, `fractal_efficiency(close, period=10)`, `hurst_exponent`, `trend_intensity_index(close, period=30)`, `market_regime_classifier`, `volatility_regime_probability`. `ml4t.engineer.features.cross_asset` (inference): `rolling_correlation`, `beta_to_market`, `co_integration_score`, `correlation_regime_indicator`. `ml4t.engineer.logging.setup_logging`.
- `case_studies/utils/temporal.py`: `arima_one_step_forecast(fitted, y_prefix)`, `garch11_conditional_volatility(...)`, `filtered_state_probs(model, X)`, `sort_states_by_variance`, `sort_states_by_mean(model, dim=0)`, `relabel_states(states, probs, order)`, `fit_hmm_kmeans_init`, `fit_hmm_restarts`, `refit_boundaries(n_obs, burnin, refit_every)`, `walk_forward_feature(X, *, timestamps, burnin, refit_every, fit, apply, n_features, window=None, freeze_after=None, on_fit_error="raise", apply_scope="prefix")`, `write_model_based(frame, path, *, keys, feature_columns, time_column, written_by, fold_column="fold", expected_folds=None, inputs=None, metadata=None)`, `fold_feature_geometry(...)`, `chain_worker_pool(n_threads)`, `lift_stream`, `wasserstein_distance_1d`, `wasserstein_barycenter_1d`, `fit_wasserstein_kmeans(sorted_segments, n_clusters=2, max_iter=50, n_init=5, random_state=42)`.
- Third party: statsmodels (`adfuller(autolag="AIC")`, KPSS, Ljung-Box, `ZivotAndrews`, ARIMA `.apply(endog, refit=False)`, Markov switching AR, `coint`, `coint_johansen`), `ruptures` (`Pelt`, `Binseg`), scipy (Welch t, Levene, Fligner, KS, `jensenshannon`, Wasserstein, Welch PSD), `pywt`, `numpy.fft.rfft`, `esig` (`sigkeys`), `arch` (`ConvergenceWarning`), `hmmlearn.GaussianHMM`, filterpy `KalmanFilter`, sklearn (`TimeSeriesSplit`, `Ridge`, `GradientBoostingRegressor`, `LinearRegression`), PyMC-style NUTS and Nixtla-style ARIMA (inferred).

Causal refit schedule (nb 02/04):
```python
for end in range(LOOKBACK, n, REFIT_EVERY):
    history = series[end - LOOKBACK:end]            # nothing after the refit date
    penalty = np.log(len(history)) * history.var()
    found = rpt.Pelt(model="l2", min_size=MIN_SEGMENT).fit(history).predict(pen=penalty)
    # columns hold until the next refit (step function)
```
Full-window FFD mask:
```python
w = get_ffd_weights(d, threshold=FFD_THRESHOLD); width = len(w)
ffd = ffdiff(log_close, d, threshold=FFD_THRESHOLD)   # boundary-partial
valid = np.arange(len(ffd)) >= width - 1              # sample loss = width - 1
```
Settings carried forward: `schedule_from_refits(refit_frame, index, column)` -> one value per session from the most recent refit; `kalman_local_linear` accepts per-session noise arrays.
Walk-forward volatility: fit `arch` on the train block -> `garch11_conditional_volatility(returns, mu=, omega=, alpha=, beta=, backcast=)` across the full series -> use the post-split block only.
Point-in-time regime per fold: `fit_hmm_restarts(train)` -> `sort_states_by_variance` -> `relabel_states` -> `filtered_state_probs(model, X_fold)`; forward recursion `alpha_t = b(o_t) * (alpha_{t-1} @ A)`, normalise each step.
esig level-two index: `idx = 1 + d + i*d + j`; purge: `PURGE = WINDOW_SIZE + FORECAST_HORIZON - 1`.
Polars catalog (nb 13):
```python
df.with_columns(
    hurst=hurst_exponent("close", period=100),
    choppiness=choppiness_index("high", "low", "close", period=14),
    trend_regime=market_regime_classifier("high", "low", "close", "volume"),
    realized=realized_volatility("returns", period=20),
    garch=garch_forecast("returns", horizon=1, alpha=0.1, beta=0.85),
    entropy=rolling_entropy("returns", window=50, n_bins=10),
).with_columns(**volatility_regime_probability("close", period=20))
conditional = regime_conditional_features("returns", "trend_regime", regime_values=[-1, 0, 1])
df = df.with_columns(**conditional)   # assert sorted(conditional) == sorted(REGIME_COLUMNS)
```
Pairwise (nb 14): `rolling_correlation(r1, r2, window=60)`, `beta_to_market(asset_return, market_return, window=60)`, `co_integration_score(p1, p2, window=120)`, `df.with_columns(**correlation_regime_indicator("correlation"))`, flags via `pl.col("^corr_regime.*$")`.
Kalman hedge ratio (filterpy):
```python
f = KalmanFilter(dim_x=2, dim_z=1); f.F = np.eye(2); f.x = np.array([[0.0], [1.0]])
f.P = np.eye(2); f.R = np.array([[1e-3]]); f.Q = np.eye(2) * 1e-5
for t in range(n):
    f.H = np.array([[1.0, independent[t]]]); f.predict(); f.update(np.array([[dependent[t]]]))
    ratio[t] = f.x[1, 0]
```
Naming convention: `FAMILY_TOKENS = {"garch", "hmm", "regime", "kalman", "arima", "ffd", "spectral", "bayesian", "har", "hurst"}` as substrings of model-based column names; schema read `pl.scan_parquet(path).collect_schema()`.
Running: `uv run python 09_model_based_features/<notebook>.py`; tests `uv run pytest tests/test_chapter_notebooks.py -v -k "09_model_based_features"`; signatures `docker compose --profile py312 run --rm py312 python 09_model_based_features/06_path_signatures.py`.

## Evidence from the book

Numeric outputs are mostly not in the notes (the notebooks print them at run time); directions and qualitative results are recorded.
- **01**: SPY price has a unit root (ADF fails, KPSS rejects); returns stationary; VIX mean-reverting (long-run level "in the teens"). Return ACF/PACF near zero at all lags; Q-Q bends at both tails; large excess kurtosis; Ljung-Box rejects on squared returns; ARCH-LM rejects. SPY ETF history opens 2006.
- **02**: Segmentation on price level marks neither 2008 nor 2020; PELT and Binseg disagree on count by construction. Causal refit finds breaks later than the full-sample fit and produces far less variable columns. CUSUM excursions in 2008 and 2020 never return; MOSUM returns to zero. Classifier scores well on synthetic data but on six market dates nearly all windows look positive while distributional statistics still rank crisis windows further apart.
- **03**: Correlation with level, ADF p-value and sample loss all fall as d rises; the ADF crossing lies inside 0.1-0.6; the 80% training search selected a different d than the full sample; `find_optimal_d` chose an order whose window exceeded the sample; some panel symbols do not test stationary at their class order.
- **04**: ML fit moves far from hand-set smoothing (observation noise ~100x level noise): the close is near an exact observation of a random-walk-like level; the walk-forward ratio moves by orders of magnitude over six years and late windows hit the bound floor; the bounded fit puts almost all movement in level so the filter slope holds sign for months while the 20-session OLS slope crosses zero repeatedly; IC vs 5-day forward return negative (short-horizon reversal).
- **05**: Daily SPY returns near white noise: entropy near max, low-frequency ratio near the white-noise share, dominant period jumps randomly; Welch PSD flat with time-varying level; spectrogram shows vertical bands after 2020 persisting ~1 year, no horizontal bands. Spectral energy vs rolling variance: correlation 1, ratio = half the squared window length. Slow wavelet components nearly identical across families; `rolling_entropy` mean near log2(10).
- **06**: Out-of-sample R^2 near or below zero for every symbol and feature set; the count of symbols where log-signature terms raised IC is the result, with no general ranking; cross-asset SPY/QQQ level-two areas split ~50/50 averaging to ~0.
- **07**: Multi-step forecast collapses to one number; lowest-RMSE order is the constant (undefined IC); panel one-step IC quartiles span zero with median slightly above; about one sixth of fits did not converge and their median IC sits nearer zero.
- **08**: SPY and panel persistence just below one (some at the boundary); squared standardized residuals lose autocorrelation; residuals exceed 3 sigma several times more often than normal; VaR exceedances above promised rates at both levels, worse at 99%; filtered and in-sample conditional vol agree on the held-out block; GARCH, realized and EWMA(120) agree on where volatile periods are, disagree on decay.
- **09**: VWAP-based RV understates variance by a non-small amount; AAPL overnight square is a large fraction of the daily square, cross-term small; exactly one implausible return = the 4-for-1 split; close-to-close is the most dispersed estimator with many zero sessions, Yang-Zhang sits highest; HAR coefficients unchanged across SE variants; HAR tracks realized vol about as well as GARCH by correlation; Hurst on returns ~0.5 with symbols on both sides, on log-vol increments ~0.1 for every symbol (distributions do not overlap); HAR coefficients vary more across symbols than either exponent.
- **10**: phi and sigma_eta have lower ESS than directly pinned parameters and MC-SE a meaningful fraction of their quarter-to-quarter spread; the ARIMA width ratio's spread is a small fraction of its mean (model uncertainty nearly constant) while original-unit width tracks the level; stationarity tests disagree on d; only four refits a year.
- **11**: Log-likelihood spread across 10 EM starts quantifies local optima; BIC prefers more than 2 states (notebook keeps 2); filtered vs smoothed disagree exactly at the edges of stressed periods; HMM, switching AR, VIX > 20 and MA rules measure different things; volatility-regime label and the trend classifier's -1 co-occur less than independence.
- **12**: On the two-regime GBM simulation all three methods separate clusters in MMD; GMM on 4 moments lands close to Wasserstein k-means in adjusted Rand, k-means on moments far from both (geometry, not information). The Wasserstein split divides on spread with overlapping means. On the index, the volatile-cluster share differs between fitted and later blocks; conditional means differ between clusters; label vs rolling std agree except near boundaries; `tail_divergence` rate between 0 and 0.5.
- **13**: Error moves ~an order of magnitude more between folds than between designs; R^2 negative or near zero for ridge and GBR; error in the volatile state larger by about the target's dispersion; regime-probability importance near zero because `volatility` is already a feature; `market_regime_classifier` calls almost all SPY sessions transitional.
- **14**: XLE/USO is not cointegrated by either test; Johansen ratio far from the regression ratio; half-lives in months; the ~1-year window with a 2-sigma band opens a couple of positions and holds through more than half the sessions. Six-pair screen: none clears Engle-Granger, exactly one clears Johansen alone, every half-life in months. Percentiles evenly spaced while z-scores bunch; XLE/SPY volatility ratio near 1 in a market-wide selloff; breadth near 1 with dispersion near 0 in universe-wide stress.

## Related references

- `chapters/08_financial_features.md` — the deterministic features this chapter extends; range-estimator efficiency.
- `chapters/07_defining_the_learning_task.md` — forward-return targets and horizons the IC checks use.
- `chapters/06_strategy_definition.md` — CV foundations: why overlapping windows need purge gaps.
- `chapters/10_text_feature_engineering.md` — same point-in-time logic applied to embeddings (when the model was trained).
- `chapters/11_ml_pipeline.md`, `chapters/12_gradient_boosting.md` — downstream pipelines that receive the Polars catalog; tuning the GBT learner used in 06/13.
- `chapters/13_dl_time_series.md`, `chapters/14_latent_factors.md` — richer latent-state and factor extraction.
- `chapters/16_strategy_simulation.md` — where IC differences meet costs, turnover and capacity; regime features beside cross-sectional signals.
- `chapters/17_portfolio_construction.md` — Bayesian Sharpe ratio; position size depending on uncertainty.
- `chapters/19_risk_management.md` — VaR exceedance logic and volatility-managed sizing.
- `case_studies/etfs.md` — per-fold HMM refit (`case_studies/etfs/04_model_based_features.py`).
- `case_studies/nasdaq100_microstructure.md` — minute-bar realized variance inputs.
- `case_studies/crypto_perps_funding.md`, `case_studies/sp500_equity_option_analytics.md`, `case_studies/us_firm_characteristics.md`, `case_studies/fx_pairs.md`, `case_studies/cme_futures.md`, `case_studies/sp500_options.md`, `case_studies/us_equities_panel.md` — their `features/model_based.parquet` stages are what the summary notebook inventories.
- `libraries/ml4t_engineer.md` — features.statistics, fdiff, ml, volatility, regime, cross_asset expressions and input contracts.
- `libraries/ml4t_diagnostic.md` — stationarity, autocorrelation, distribution, volatility, drift evaluation and `pooled_ic`.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `workflow.md`, `companion_repo.md` — cross-cutting indices.
- Further reading: Dickey-Fuller (1979); Zivot-Andrews (1992); Bai-Perron (1998); Lopez de Prado (2018); Kalman (1960); rlabbe Kalman book; Engle (1983); Bollerslev (1986); Nelson (1991); Glosten et al. (1993); Corsi (2009); Gatheral et al. (2014); Parkinson (1980); Garman-Klass (1980); Yang-Zhang (2000); Hamilton (1989); Ang-Bekaert (2002); Ang-Timmermann (2011); Horvath et al. (2021); Uysal-Mulvey (2021); Shu-Mulvey (2025); Hoffman-Gelman (2011); Betancourt (2018); Taylor-Shephard (2005); Marra (2023); Moreira-Muir (2017); Engle-Granger (1987); Johansen-Juselius (1990); Chevyrev et al. (2026); Sun-Yu (2025); Hyndman FPP; ADIA Lab structural break challenge.

## Glossary

- **Stationary / unit root** — moments independent of sample position / today = yesterday + non-decaying shock.
- **ADF, KPSS, Phillips-Perron** — unit-root tests (ADF, PP null = unit root; KPSS null = stationary).
- **ACF / PACF, correlogram, Ljung-Box, ARCH-LM, Jarque-Bera** — autocorrelation functions and the joint tests on them; ARCH-LM regresses squared returns on lags; JB tests normality from skew and kurtosis.
- **Structural break, Zivot-Andrews, PELT / Binseg, CUSUM / MOSUM, burn-in** — regime-change date; one-break unit-root test; penalised exact / greedy segmentation; cumulative / moving sums against a fixed burn-in reference.
- **Fisher's method, Welch t, Levene, Fligner-Killeen, KS, Jensen-Shannon, Hellinger, Wasserstein** — p-value combination and two-sample statistics used as break features.
- **Fractional differencing, boundary-partial vs full-window, sample loss** — (1-L)^d weights truncated at a threshold; keep early rows with a shorter filter vs null them (width - 1 rows lost).
- **Local linear trend, innovation, Kalman gain, filtered vs smoothed, hedge ratio** — state [level, slope]; observation minus prediction; share of innovation believed; uses data to t vs future data; units of one asset held against another.
- **Wavelet decomposition, spectral entropy, dominant period, low-frequency ratio, Parseval, Welch's method** — multi-scale components (non-causal DWT); evenness of power; strongest bin; share in slowest bins; total power = sum of squared deviations; averaged periodograms.
- **`rolling_entropy`, `fourier_features`** — binned distributional entropy; calendar sin/cos basis of the row index.
- **Path signature, log-signature, signed (Levy) area, reparameterisation invariance, purge gap** — iterated integrals; minimal form; loop orientation; invariance to traversal speed; rows dropped at split boundaries.
- **Filtering vs smoothing (ARIMA/GARCH/HMM), information criterion, information coefficient** — forward recursion under fixed parameters vs whole-sample inference; penalised likelihood; correlation of a column with the forward target.
- **Persistence, half-life, standardized residual, VaR, realized variance, overnight return** — alpha + beta; sessions for half a shock to decay; return / sigma_t; loss exceeded with probability 1 - confidence; sum of squared intraday returns; close-to-next-open return.
- **Garman-Klass / Parkinson / Rogers-Satchell / Yang-Zhang, HAR, Newey-West, HC3** — range-based estimators (YZ includes overnight); heterogeneous autoregression at 1/5/22; HAC and leverage-robust covariances.
- **Hurst exponent, rescaled range / DFA, rough volatility** — scaling exponent (0.5 random walk); two estimators; log-vol increments with H ~ 0.1.
- **Stochastic volatility, divergences / R-hat / ESS, Monte Carlo SE, non-centered parameterisation, coverage** — latent AR(1) log-vol; MCMC diagnostics; posterior-mean error from finite draws; reparameterised latent path; share of outcomes inside the interval.
- **HMM, forward algorithm, Viterbi, label switching, expected duration, state entropy, Markov switching AR** — latent states with transitions and emissions; filtered recursion; whole-path decode (non-causal); arbitrary ordering fixed by sorting; 1/(1 - A_kk); uncertainty of the filtered distribution; AR with regime-switching parameters.
- **Stream lift, p-Wasserstein, barycenter, Lloyd / k-means++, inertia, MMD, median heuristic, adjusted Rand, silhouette, mixed window** — windows as sorted samples; matched-quantile distance; quantile-wise median/mean; clustering iteration and seeding; within-geometry cost; kernel distribution distance; kernel-width rule; chance-corrected agreement; cohesion score (not used); window straddling a switch.
- **Mixture of experts, impurity importance, paired difference, PSI** — one model per hard state; squared-error reduction over splits; fold-wise design difference; binned stability index.
- **Cointegration, Engle-Granger, Johansen, half-life (spread), process / measurement noise, roll cost** — stationary linear combination; residual unit-root test; system eigenvalue test; -ln 2 / ln(1 + phi); Kalman Q and R; drift from rolling futures.
- **Cross-sectional rank / percentile / z-score, benchmark-relative, breadth, dispersion, `rolling_rank`** — position among assets on a date; minus or divided by the benchmark; fraction above a threshold; cross-asset std; trailing-window rank along an asset's own history.
- **Schema read, family token** — `scan_parquet().collect_schema()` footer read; substring attributing a column to a model-based family.
