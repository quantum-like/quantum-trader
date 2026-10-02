# ml4t-engineer library (features, labels, alternative bars, leakage-safe dataset prep)

> `ml4t-engineer` (v0.1.6, MIT, Stefan Jansen) is the feature-engineering, labeling and dataset-preparation layer of the ml4t stack. It sits between `ml4t-data` (validated OHLCV / tick frames in) and `ml4t-diagnostic` / `ml4t-models` / `ml4t-backtest` (features, labels, sample weights and fold-wise scaled matrices out). It ships 120 registry features in 11 categories, path-dependent labels (triple barrier, ATR barrier, trend scanning, meta-labels), activity-based bars (tick / volume / dollar / imbalance / run) and **train-only preprocessing**. Core interface is Polars (`DataFrame` or `LazyFrame`, same type out); pandas is only a conversion target. Repo `github.com/ml4t/engineer`, docs `ml4trading.io/docs/engineer/`.

Workflow stage: ML4T steps "bars -> features -> labels -> leakage-safe dataset" (chapters 03, 07, 08, 09, 11). Upstream contract: `ml4t-data` (OHLCV with `asset_id`/`timestamp`), `ml4t-specs` (`ArtifactSpec`, market-data spec). Downstream: `ml4t-diagnostic` supplies the purged/embargoed splitters that `MLDatasetBuilder.split` consumes and the `recommendations` dict that `PreprocessingPipeline.from_recommendations` consumes; `ml4t-backtest`/`ml4t-live` consume `sized_signal`, `label_return`, `barrier_hit`.

## Install and import

| Item | Value |
|---|---|
| Python | `>=3.12,<3.15` (3.15 blocked by a Polars compatibility exception) |
| Core deps | `ml4t-specs>=0.1,<0.2`, `numba>=0.57`, `numpy>=1.24,<2.6`, `pandas>=2.0`, `polars>=0.20`, `pydantic>=2,<3`, `pyyaml`, `structlog`, `python-dateutil` |
| Extras | `[ta]` ta-lib (native lib; validation only), `[store]` duckdb+pyarrow (feature store), `[calendars]` pandas-market-calendars, `[viz]`, `[stats]` statsmodels, `[ml]` lightgbm+shap, `[dev]`, `[all]` |
| Dev loop | `uv sync --dev --extra docs --extra ta --extra store --extra viz`; `ruff check`, `ty check`, `pytest tests/ -q` |
| Agent docs | `ml4t.engineer.get_agent_docs() -> dict[str, Path]` returns bundled `AGENTS.md` paths; start with the root one |

```python
from ml4t.engineer import compute_features, features, FeatureCatalog           # features + discovery
from ml4t.engineer import MLDatasetBuilder, create_dataset_builder, FoldResult  # leakage-safe datasets
from ml4t.engineer import StandardScaler, MinMaxScaler, RobustScaler, PreprocessingPipeline
from ml4t.engineer.config import LabelingConfig, PreprocessingConfig, DataContractConfig, load_experiment_config
from ml4t.engineer.labeling import (triple_barrier_labels, atr_triple_barrier_labels, trend_scanning_labels,
    fixed_time_horizon_labels, rolling_percentile_binary_labels, meta_labels, apply_meta_model,
    calculate_label_uniqueness, calculate_sample_weights, sequential_bootstrap, calendar_aware_labels)
from ml4t.engineer.bars import DollarBarSamplerVectorized, ImbalanceBarSampler, TickRunBarSampler
from ml4t.engineer.store.offline import OfflineFeatureStore
```

Note: feature-evaluation configs (`StationarityConfig`, `ACFConfig`, ...) moved to `ml4t-diagnostic`; do not import them from `ml4t.engineer.config`.

## API map by task

### Feature computation and discovery

| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `compute_features` | `(data, features: list[str \| dict] \| Path \| str, *, group_col=None, timestamp_col=None, assume_sorted=False) -> same frame type` | Config-driven entry point; appends feature columns | `features` = names, spec dicts `{"name","params","output"}`, or a YAML path. Sorts by `[*group_cols, timestamp]`, resolves `dependencies` (Kahn), wraps every expr in `.over(group_cols)` |
| `features` (module proxy) / `FeatureCatalog` | `list(category, normalized, ta_lib_compatible, tags, input_type, output_type, has_dependencies, limit)`, `describe(name)`, `search(query, max_results=10)`, `by_input_type`, `by_lookback(max_lookback)`, `lookback(name, **params) -> int`, `categories()`, `tags()`, `stats()` | Discover and size features | `lookback()` = index of the first usable row; use it to trim warm-up or size purge windows |
| `@feature` | `(*, name, category, description, lookback=None, normalized=False, value_range=None, formula="", ta_lib_compatible=False, input_type="close", output_type="indicator", parameters=None, dependencies=None, references=None, tags=None)` | Register a custom feature | Metadata attached at import; no call overhead. Categories: momentum, trend, volatility, volume, statistics, math, price_transform, microstructure, ml, risk, regime |
| `FeatureRegistry` / `get_registry()` | `register(meta)`, `get(name)`, `list_all()`, `list_normalized()`, `list_by_category()`, `list_ta_lib_compatible()`, `get_dependencies(name)` | Global singleton registry | `FeatureMetadata` fields: name, func, category, lookback (callable -> int), normalized, value_range, input_type, dependencies, ... |
| `add_indicators` | `(df, indicators: dict[str, pl.Expr]) -> pl.DataFrame` | Batch `with_columns` keyed by output name | `ml4t.engineer.features.utils.helpers` |
| `returns`, `log_returns`, `percentage_change` | `(column, periods=1) -> pl.Expr` | Return transforms | `features/utils/arithmetic.py` |
| `get_trading_session_filter` | `(start_time="09:30", end_time="16:00") -> pl.Expr` | RTH filter | |

### Alternative bars (`ml4t.engineer.bars`; input tick frame needs `timestamp, price, volume`; imbalance/run need `side` in {+1,-1})

| Class | Signature | Purpose | Notes |
|---|---|---|---|
| `TickBarSampler`, `VolumeBarSampler`, `DollarBarSampler` | `(ticks_per_bar: int)`, `(volume_per_bar: float)`, `(dollars_per_bar: float)`; `.sample(data, include_incomplete=False)` | Standard activity bars | Package exports the `*Vectorized` versions under these names; originals carry an `Original` suffix |
| `TickBarSamplerVectorized`, `VolumeBarSamplerVectorized`, `DollarBarSamplerVectorized` | same | Numba/Polars vectorized, "10,000+ rows/sec" | Docstring kwargs (`volume_threshold`, `dollar_threshold`, `tick_threshold`) do NOT match the real `__init__`; use the signatures here |
| `ImbalanceBarSampler` (volume), `TickImbalanceBarSampler`, `ImbalanceBarSamplerVectorized` | `(expected_ticks_per_bar: int, alpha=0.1, initial_p_buy=0.5, min_bars_warmup=10)` | AFML 2.3 imbalance bars; EWMA-adaptive `E[T]`, `P[b=1]` | TIB: `θ=Σb_t`, `E[θ_T]=E[T]·|2P[b=1]−1|`; VIB: `θ=Σb_t v_t`, `E[θ_T]=E[T]·|2v⁺−E[v]|` |
| `FixedTickImbalanceBarSampler`, `FixedVolumeImbalanceBarSampler` | `(threshold)` | Non-adaptive threshold | |
| `WindowTickImbalanceBarSampler`, `WindowVolumeImbalanceBarSampler` | `(initial_expected_t, bar_window=10, tick_window=1000)` | Window-estimated expectations | |
| `TickRunBarSampler`, `VolumeRunBarSampler`, `DollarRunBarSampler`, `FixedTickRunBarSampler(threshold)` | `(expected_ticks_per_bar, alpha=0.1, initial_p_buy=0.5, min_bars_warmup=10)` | AFML run bars | `θ_T = max{Σ buys, Σ sells}` in the bar, NOT consecutive same-side runs, no reset on direction change; `E[θ_T]=E[T]·max{P[b=1],1−P[b=1]}`. Hint: `expected_ticks_per_bar=50` tick / `100` volume, dollar |
| `BarSampler` (ABC) | `sample`, `_validate_data`, `_create_ohlcv_bar(ticks, additional_cols)` | Base for custom samplers | Output OHLCV bars + per-bar diagnostics (expected T, p_buy, thresholds) |

### Labels (`ml4t.engineer.labeling`)

| Function | Signature | Purpose | Output columns / notes |
|---|---|---|---|
| `triple_barrier_labels` | `(data, config: LabelingConfig, price_col=None, high_col=None, low_col=None, timestamp_col=None, group_col=None, calculate_uniqueness=False, uniqueness_weight_scheme="returns_uniqueness", contract=None, open_col=None)` | First touch of upper / lower / vertical barrier | `label` (Int32: 1 upper, -1 lower, 0 time), `label_time`, `label_price`, `label_return`, `label_bars`, `label_duration`, `barrier_hit` ("upper"/"lower"/"time"); `+label_uniqueness, sample_weight` when `calculate_uniqueness=True`. Touch evaluated on open/high/low; gap-through exits at open |
| `atr_triple_barrier_labels` | `(data: DataFrame \| LazyFrame, atr_tp_multiple=None, atr_sl_multiple=None, atr_period=None, max_holding_bars: int \| str \| None=None, side: 1 \| -1 \| 0 \| str \| None=None, price_col=None, timestamp_col=None, group_col=None, trailing_stop: bool \| float \| str=False, *, config=None, contract=None)` | Volatility-scaled barriers = multiple × ATR | `side` may be a column (per-row direction); `max_holding_bars` may be a duration string; `trailing_stop` bool, ATR multiple or column |
| `trend_scanning_labels` | `(data, min_window=None, max_window=None, step=None, price_col=None, timestamp_col=None, group_col=None, *, t_value_threshold=None, config=None, contract=None)` | AFML trend scanning: forward window with max \|t\| of price-on-time regression | `label` (Int8 sign of slope, null when \|t\| < threshold), `t_value`, `optimal_window` (Int32) |
| `fixed_time_horizon_labels` | `(data, horizon: int \| str=None, method: "returns" \| "log_returns" \| "binary"=None, price_col=None, group_col=None, timestamp_col=None, tolerance=None, threshold=None, *, config=None, contract=None)` | Forward return at bar or time horizon | int horizon -> shift, names `label_return_{h}p` / `label_log_return_{h}p` / `label_direction_{h}p`; duration string -> `join_asof` with `tolerance`, suffix = duration |
| `rolling_percentile_binary_labels` | `(data, horizon: int \| str, percentile: float, direction="long", lookback_window=252*24*12, price_col=None, session_col=None, min_samples=None, group_col=None, timestamp_col=None, tolerance=None, *, config=None, contract=None)` | 1 if forward return beats rolling point-in-time percentile | `forward_return_{h}`, `threshold_p{p}_h{h}`, `label_{direction}_p{p}_h{h}` (Int8); decimal `.` -> `p` (95.5 -> `p95p5`); short: fwd <= threshold |
| `rolling_percentile_multi_labels` | `(data, horizons: list, percentiles: list, direction="long", lookback_window=252*24*12, ...)` | Cartesian product of the above (inference) | |
| `calendar_aware_labels` | `(data, config, calendar: str \| TradingCalendar, price_col=None, timestamp_col=None, group_col=None, contract=None)` | Barriers that do not span session breaks | `calendar` string = pandas_market_calendars name (e.g. "NYSE", "CME_Equity"; inference) via `PandasMarketCalendar`; `SimpleTradingCalendar(gap_threshold_minutes=30).fit(data)` learns breaks from gaps |
| `meta_labels` | `(data, signal_col, return_col, threshold=0.0) -> DataFrame` | 1 if sign(signal) matched return beyond threshold | AFML ch. 3 meta-labeling |
| `compute_bet_size` | `(probability, method="sigmoid" \| "linear" \| "discrete", scale=1.0, threshold=0.5) -> pl.Expr` | Probability -> bet size | |
| `apply_meta_model` | `(data, primary_signal_col, meta_probability_col, bet_size_method="sigmoid", scale=5.0, threshold=0.5, output_col="sized_signal")` | Size primary signal by meta-probability | `scale` default 5.0 here vs 1.0 in `compute_bet_size` |
| `build_concurrency`, `calculate_label_uniqueness` | `(event_indices, label_indices, n_bars=None)` | AFML ch. 4 overlap accounting | uniqueness = mean 1/concurrency over the label's life |
| `calculate_sample_weights` | `(uniqueness, returns, weight_scheme="returns_uniqueness")` | Schemes: `returns_uniqueness`, `uniqueness_only`, `returns_only`, `equal` | |
| `sequential_bootstrap` | `(starts, ends, n_bars=None, n_draws=None, with_replacement=True, random_state=None) -> int64 idx` | Draw less-redundant training sets | |
| `compute_label_statistics` | `(data, label_col) -> dict` | Counts, positive rate | |
| `register_labeling_features` | `(registry=None) -> int` | Registers `triple_barrier_feature`, `trend_scanning_feature`, `fixed_time_horizon_feature` in the registry | `ALL_LABELING_FEATURES` |
| `labeling.utils` | `parse_duration(value) -> timedelta`, `is_duration_string`, `time_horizon_to_bars(timestamps, time_horizon, event_indices=None)`, `get_future_price_at_time(data, time_horizon, price_col="close", ...)`, `resolve_labeling_columns(...)`, `validate_price_no_nans(data, price_col)` | Helpers | |

### Feature families (public dispatchers accept `ndarray | pl.Series | str`; return ndarray for arrays, `pl.Expr` for column names; `*_numba` and `*_polars` variants exist)

| Family | Functions (defaults) | Notes |
|---|---|---|
| Momentum (TA-Lib compatible) | `rsi(close, period=14)`, `macd(close, 12, 26)`, `macd_signal(..., signal_period=9)`, `macd_full -> (macd, signal, hist)`, `macdfix`, `adx/adxr/plus_di/minus_di/dx/plus_dm/minus_dm(h,l,c, 14)`, `aroon/aroonosc(h,l,14)`, `apo/ppo(close, 12, 26, matype=0)`, `bop(o,h,l,c)`, `cci(h,l,c, 20)`, `cmo(close, 14)`, `imi(open, close, 14)`, `mfi(h,l,c,v, 14)`, `mom/roc/rocp/rocr/rocr100(close, 10)`, `sar(h,l, 0.02, 0.2)`, `stochastic(h,l,c, fastk=14, slowk=1, slowd=3, return_pair=False)`, `stochf(h,l,c, 5, 3, matype=0)`, `stochrsi(close, 14, 5, 3)`, `trix(close, 30)`, `ultosc(h,l,c, 7, 14, 28)`, `willr(h,l,c, 14)` | RSI uses Wilder smoothing and is gap-aware per contiguous run; `trix` docstring says `*10000` vs TA-Lib `*100` (unverified) |
| Trend / MAs | `sma(close, period)`, `ema`, `wma`, `dema(30)`, `tema(30)`, `trima(30)`, `kama(30)`, `t3(5, vfactor=0.7)`, `midpoint(14)`, `donchian_channels(high="high", low_col="low", period=20) -> (upper, lower, middle)`, `apply_ma(close, period, matype=0)` | `donchian_*` is column-name-only; `matype` follows TA-Lib codes (0 SMA ... 8 T3; mapping inferred) |
| Volatility | `trange`, `atr(h,l,c, 14)`, `natr(14)`, `bollinger_bands(close, 20, nbdevup=2.0, nbdevdn=2.0) -> (upper, middle, lower)`; estimators `realized_volatility(returns)`, `parkinson(h,l)`, `garman_klass`, `rogers_satchell`, `yang_zhang(o,h,l,c)` all `(period=20, annualize=True, trading_periods=252)`; `ewma_volatility(close="close", span=120, normalize=False, mu=-5.0, sigma=2.0)`; `garch_forecast(returns, horizon=1, omega=1e-5, alpha=0.1, beta=0.85)`; `conditional_volatility_ratio(returns, threshold=0.0, period=20)`; `volatility_of_volatility(close, 20, 20)`; `volatility_percentile_rank(close, 20, lookback=252)`; `volatility_regime_probability(close, 0.01, 0.02, 20, 100) -> dict` | `garch_forecast` uses FIXED parameters (not fitted); fit GARCH in `ml4t-models`/`arch` when you need estimates |
| Volume | `obv(close, volume)`, `ad(h,l,c,v)`, `adosc(h,l,c,v, fast=3, slow=10)` | A/D money-flow multiplier = 0 when High == Low |
| Statistics | `stddev(close, 5, nbdev=1.0, ddof=0)`, `var(close, 5)`, `avgdev(14)`, `linearreg / _angle / _intercept / _slope(14)`, `tsf(14)` = LINEARREG + SLOPE (one-step forecast); structural break: `coefficient_of_variation(feature, 50)`, `rolling_cv_zscore(feature, 50, lookback_multiplier=5)`, `variance_ratio(feature, 100, q=5)`, `rolling_kl_divergence(feature, 100, n_bins=20)`, `rolling_wasserstein(feature, 100)`, `rolling_drift(feature, 100, normalize=True)` | TA-Lib stats use population `ddof=0`; structural-break features compare first vs second half of one window (ADIA Lab 2025) |
| Regime | `choppiness_index(h,l,c, 14)` (0-100), `variance_ratio(close, periods=None, base_period=1, window=None) -> dict`, `fractal_efficiency(close, 10)`, `hurst_exponent(close, period=100, min_lag=2, max_lag=None)`, `trend_intensity_index(close, 30)`, `market_regime_classifier(h,l,c,v, adx_threshold=25.0, chop_threshold_high=61.8, chop_threshold_low=38.2) -> pl.Expr` | Module docstrings list stale signatures; these are authoritative |
| Risk (window 252) | `value_at_risk(returns, 0.95, 252, method="historical" \| "parametric" \| "cornish_fisher")`, `conditional_value_at_risk(...)`, `maximum_drawdown(close, window=None) -> dict` (expanding when None), `downside_deviation(returns, target_return=0.0)`, `tail_ratio`, `higher_moments -> dict` (skew, kurtosis, JB), `risk_adjusted_returns(returns, risk_free_rate=0.0, window=252, close=None, trading_periods=252) -> dict` (Sharpe, Sortino, Calmar needs `close`, Omega, IR), `ulcer_index(close, 14)`, `information_ratio(returns, benchmark_returns, 252)` | |
| Microstructure (period 20) | `amihud_illiquidity(returns, volume, price)`, `kyle_lambda(returns, volume, method="ratio")`, `roll_spread_estimator(close)`, `realized_spread(h,l,c)`, `effective_tick_rule(close)`, `order_flow_imbalance(volume, close, use_tick_rule=True)`, `price_impact_ratio(..., impact_threshold=0.001)`, `quote_stuffing_indicator(volume, num_trades=None, period=5)`, `trade_intensity`, `volume_at_price_ratio(n_bins=10)`, `volume_synchronicity`, `volume_weighted_price_momentum`; order book: `bid_ask_imbalance(period=1)`, `book_depth_ratio`, `weighted_mid_price` (micro-price) | Order-book features need L1/L2 quote columns (`bid_price, ask_price, bid_size, ask_size`), e.g. ALGOSeek TAQ |
| ML helpers | `create_lag_features(feature, lags=None, include_diff=True, include_ratio=False) -> dict`, `cyclical_encode(value, period)`, `directional_targets(returns, thresholds=None, horizon=None)`, `fourier_features(n_components=10, period=None)`, `interaction_features(features, max_degree=2)`, `multi_horizon_returns(close, horizons=None, log_returns=False)`, `percentile_rank_features(feature, windows=None)`, `regime_conditional_features(feature, regime)`, `time_decay_weights(lookback, decay_type="exponential", half_life=None)`, `volatility_adjusted_returns(returns, vol_lookback=20)`; entropy (AFML 18): `rolling_entropy(feature, 50, n_bins=10)`, `rolling_entropy_lz(feature, 100, encoding="quantile")`, `rolling_entropy_plugin(..., word_length=1)`, encoders `encode_binary/quantile/sigma` | `fourier_features` = sin/cos of row position (deterministic, no future data) |
| Cross-asset / composite | `rolling_correlation(s1, s2, 20)`, `beta_to_market(asset, market, 60)`, `correlation_regime_indicator(corr, 0.3, 0.7, 20)`, `lead_lag_correlation(s1, s2, max_lag=10, window=20)`, `multi_asset_dispersion(returns_list, 20, method="std")`, `correlation_matrix_features(returns_list, 20)` (eigenvalues), `transfer_entropy(s1, s2, lag=1, window=100, bins=10)`, `co_integration_score(p1, p2, 60)`, `cross_asset_momentum(returns_list, 20, method="rank")`; composites: `rolling_z_score(column, 252)`, `z_score_composite(data, feature_cols, period=252, weights=None, output_col="composite_score")`, `illiquidity_composite`, `momentum_composite` (period 20) | Composite z-scores follow Cahan & Luo (2013) |
| Fractional differencing (AFML 5) | `ffdiff(close, d, threshold=1e-5) -> Expr \| Series`, `find_optimal_d(close, d_range=(0.0, 1.0), step=0.01, adf_pvalue_threshold=0.05) -> dict`, `get_ffd_weights(d, threshold=1e-5, max_length=10000)`, `fdiff_diagnostics(close, d)` | `find_optimal_d` picks the minimum d passing ADF at p<0.05 |
| Price transforms | `avgprice`, `medprice`, `typprice`, `wclprice`, `midprice(h,l, 14)` | No lookback except `midprice` |
| Math | `maximum/minimum/summation(close, timeperiod=30, implementation="auto")` | |

### Preprocessing and leakage-safe datasets

| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `StandardScaler` | `(columns=None, with_mean=True, with_std=True, ddof=1)` | z-score | `columns=None` = all numeric |
| `MinMaxScaler` | `(columns=None, feature_range=(0.0, 1.0))` | | |
| `RobustScaler` | `(columns=None, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))` | median / IQR | |
| `BaseScaler` | `fit`, `transform`, `fit_transform`, `clone()`, `to_dict()/from_dict()`, `is_fitted`, `fitted_columns`, `statistics` | Train-only fit contract | `transform` before `fit` -> `NotFittedError` |
| `PreprocessingPipeline` | `(recommendations=None, min_confidence=0.0, winsorize_limits=(0.01, 0.99))`; `.from_recommendations(recs, ...)`; `fit/transform/fit_transform/to_dict/from_dict/get_transform_summary()` | Apply per-feature `TransformType` {SCALE, CLIP, WINSORIZE} from `ml4t-diagnostic` recommendations | Test-time clip/winsorize uses training bounds |
| `MLDatasetBuilder` | `(features: pl.DataFrame, labels: pl.Series \| pl.DataFrame, dates: pl.Series \| None=None)`; `.set_scaler(scaler \| PreprocessingConfig \| None)`; `.split(cv: SplitterProtocol, groups=None) -> Iterator[FoldResult]`; `.train_test_split(train_size=0.8, shuffle=False, random_state=None)`; `.to_numpy()`, `.to_pandas()` (raw, unscaled); `.get_feature_percentiles(train_indices, quantiles=[0.1,0.25,0.5,0.75,0.9])`; `.compute_label_percentiles(train_indices, n_quantiles=5)`; `.info -> DatasetInfo` | Fold-wise, train-only scaled matrices | `FoldResult(X_train, X_test, y_train, y_test, train_indices, test_indices, fold_number, scaler)`, `.to_numpy()` |
| `create_dataset_builder` | `(features, labels, dates=None, scaler="standard")` | Convenience constructor | |
| `SplitterProtocol` | `.split(X, y=None, groups=None) -> Generator[(train_idx, test_idx)]` | Any `ml4t.diagnostic.splitters` or sklearn splitter | Builder verifies, does not create, purge/embargo |

### Configuration, calendars, store, relationships, logging, validation

| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `BaseConfig` (Pydantic v2) | `to_dict(exclude_none=False, mode="python")`, `to_json(path, indent=2)`, `from_json`, `to_yaml`, `from_yaml`, `from_dict`, `from_file`, `validate_fully() -> list[str]`, `diff(other) -> dict[path, (old, new)]` | Reproducible config round-trip | Timedeltas encoded portably |
| `LabelingConfig` factories | `triple_barrier(upper_barrier=0.02, lower_barrier=0.01, max_holding_period=20, side=1, trailing_stop=False)`, `atr_barrier(atr_tp_multiple=2.0, atr_sl_multiple=1.0, atr_period=14, max_holding_period=20, side=1, trailing_stop=False)`, `fixed_horizon(horizon=10, return_method="returns", threshold=None)`, `trend_scanning(min_horizon=5, max_horizon=20, t_value_threshold=2.0, step=1)` | One unified model; `.method` set by factory | Barriers `float \| str \| None` (str presumed a column for dynamic barriers; inference); `max_holding_period: int \| str \| timedelta` |
| `PreprocessingConfig` | `standard(with_mean, with_std, columns)`, `minmax(feature_range)`, `robust(quantile_range)`, `none()`; `create_scaler() -> BaseScaler \| None`; `PreprocessingConfig(scaler="standard")` | | |
| `ExperimentConfig` / `load_experiment_config(path, *, validate=True)` / `save_experiment_config(config, path, *, include_defaults=False)` | `.features: list[dict]`, `.labeling`, `.preprocessing` | Versioned YAML experiment document | Root must be a str-keyed mapping; structure validated even when `validate=False` |
| `DataContractConfig.from_mapping(mapping)`; `data_contract_from_market_data_spec(spec)` | | Canonical column mapping shared across ml4t libs; adapts an `ml4t-specs` market-data spec | Passed as `contract=` to every labeler |
| `EquityCalendar(exchange="NYSE", timezone=None)`, `CryptoCalendar(maintenance_window=None)` | `is_session(dt)`, `next_open`, `previous_close`, `sessions_between`, `session_duration`, `filter_sessions(df, timestamp_col="timestamp")`, `add_session_info(df)` | Session logic | Equity uses `pandas_market_calendars` if installed, else weekday/hours fallback with NO holidays |
| `OfflineFeatureStore` | `(path=None, read_only=False)`; `save_features(df, table_name, mode="replace" \| "append" \| "fail")`; `load_features(table, columns=None, filter_expr=None, limit=None)`; `point_in_time_join(labels, features_table, timestamp_col="timestamp", join_keys=None, tolerance=None)`; `list_tables()`, `execute(sql)` | DuckDB + Arrow zero-copy store | `point_in_time_join` = `join_asof(strategy="backward", by=join_keys, tolerance)`; `filter_expr` is raw SQL (caller's responsibility) |
| `compute_correlation_matrix(data, method="pearson" \| "spearman" \| "kendall", min_periods=None, features=None)`, `plot_correlation_heatmap(corr, threshold=None, cmap="RdBu_r", figsize=(10, 8))` | | Feature redundancy screen | Kendall tau-b in-house (no SciPy) |
| `LoggingConfig(level=INFO, performance_warnings=True, data_quality_checks=True, warn_threshold_ms=1000.0)`, `configure_logging`, `logged_feature(...)`, `PerformanceTracker` | | Performance / data-quality logging | |
| `validate_ohlcv_schema(df, require_asset_id=True, allow_flexible_time=True)`, `validate_schema(df, schema)`, `validate_window/period/threshold/lag/probability/percentage/positive/column_exists` | | Fail-fast checks | Exceptions: `ML4TEngineerError(message, context, cause)` -> `ConfigurationError`, `ValidationError`, `InvalidParameterError`, `DataValidationError`, `DataSchemaError`, `InsufficientDataError`, `ComputationError`, `IntegrationError` |
| `artifacts`: `FeatureSpec`, `LabelSpec`, `PredictionSpec` (+ `*Schema`, `*Definition`), all `.from_mapping(mapping)` | | Persisted artifact contracts (ArtifactSpec from `ml4t-specs`) | Interchange with `ml4t-data`, `ml4t-diagnostic`, backtest libs |

## Data contracts

| Contract | Rule |
|---|---|
| Frame type | Polars `DataFrame` or `LazyFrame` in, same type out (`compute_features`; `atr_triple_barrier_labels` also accepts LazyFrame). Feature functions take `pl.Expr \| str` column refs and return `pl.Expr`, `dict[str, Expr]` or `tuple[Expr]`; numba kernels take `NDArray[float64]` |
| Canonical columns | `open, high, low, close, volume` (lowercase); `returns` for return-based features; order-book `bid_price, ask_price, bid_size, ask_size`; tick data for bars `timestamp, price, volume` (+ `side` in {+1, -1} for imbalance/run bars, not derived by the sampler; build it with `effective_tick_rule` or from `ml4t-data`) |
| `COLUMN_ARG_MAP` (dispatch) | `open/high/low/close/volume/returns -> same name`; `price, value, feature, volatility, regime -> close`; `features -> ["close"]`; `KEYWORD_ONLY_PARAMS = {"implementation"}`. `INPUT_TYPE_COLUMNS`: `OHLCV, OHLC, HLC, HL, close, returns, volume` |
| Asset grouping | `compute_features` auto-detects the first of `asset_id > symbol > ticker > product > asset`; labelers (`resolve_group_cols`) use `symbol > product > ticker > asset > asset_id` and append `position` when present alongside `symbol`/`product`. Multi-column grouping via a list. All feature exprs run `.over(group_cols)` |
| Time column | `compute_features` auto-detects `timestamp > event_time > date > datetime`; must be `pl.Date`/`pl.Datetime`. Labelers auto-detect the first Date/Datetime column. Timezone travels with the Datetime dtype; `PandasMarketCalendar` treats naive datetimes as UTC; calendars expose tz strings such as `America/New_York` |
| Column precedence (labelers) | explicit argument > `LabelingConfig` field > `DataContractConfig` (`contract=`) > defaults / auto-detect |
| Horizons and windows | `int` = bars; duration string (`'30m'`, `'1h'`, `'1d'`, `'5d'`, `'1w'`) = time-based (`join_asof` with `tolerance`); `is_duration_string` separates durations from column names; int lookbacks become Polars `f"{n}i"` |
| Feature spec | `{"name": str, "params": dict, "output": str}`; `output` defaults to `name`; YAML config carries the same list. Return shaping: Expr -> `output`; dict -> `output_key`; tuple -> `output_0, output_1, ...`. Outputs are appended, never overwrite |
| Triple-barrier internals | `event_indices (intp)`, `upper/lower_barriers (float64 return thresholds)`, `max_periods (int64)`, `sides (int32 in {1,-1,0})`, `trailing_stops (float64)`; kernel returns `(labels, label_indices, label_prices, label_returns, bar_durations)` |
| Dataset contract | `features` (all columns are features), `labels` (Series; DataFrame -> first column), `dates` (Date/Datetime, no nulls, nondecreasing; equal timestamps always land in one partition) |
| Config objects | Pydantic v2 `BaseConfig` descendants; YAML/JSON round-trip; `diff()` for experiment comparison; `ExperimentConfig` YAML sections `features`, `labeling`, `preprocessing`, versioned |
| Scaler state | `to_dict()/from_dict()` hold `columns`, per-column statistics and constructor config; `PreprocessingPipeline.to_dict()` persists fitted state for production |
| Feature store | DuckDB tables from Arrow; identifiers must pass `_quote_identifier`; `path=None` -> in-memory (inference) |
| Errors | `DataSchemaError` for column/ordering problems, `ValueError` for spec problems; every `ML4TEngineerError` carries `.context` and `.cause` |

## Built-in guardrails

Leakage and ordering:

1. **Train-only scaler fit per fold** (`MLDatasetBuilder.split`): `fold_scaler = scaler.clone()`, `fit_transform(train)`, `transform(test)`; labels untouched. Never fit on the full frame (AFML ch. 7). Check: each `FoldResult.scaler` is a distinct fitted clone; `scaler.statistics` must not change after `transform(test)`.
2. **Strict temporal boundary** (`_validate_fold_boundary`): `intersect1d(train, test)` must be empty and, with `dates`, `max(dates[train]) < min(dates[test])` strictly, else `ValueError("training dates must strictly precede test dates")`. The builder verifies but does NOT purge/embargo: pass `dates` and use `ml4t.diagnostic.splitters.PurgedWalkForwardCV(n_splits=5, embargo_pct=0.01)`.
3. **Splitter index validation**: partitions must be 1-D, nonempty, integer dtype (bool rejected), unique, within `[0, n_samples)`.
4. **Date index validation**: `dates` must be `pl.Date`/`pl.Datetime`, zero nulls, `is_sorted()`; else `ValueError` at construction.
5. **Holdout rules** (`train_test_split`): `train_size` finite, strictly in (0,1), not bool; `shuffle=True` with `dates` raises; >=1 row each side; the cut moves to the last timestamp-change index `<= n_train` so equal timestamps stay together; needs >=2 distinct timestamp groups.
6. **Train-only percentiles**: `get_feature_percentiles` / `compute_label_percentiles` take `train_indices`; "never compute quantiles on test data." `to_numpy()`/`to_pandas()` are raw and unscaled by design.
7. **Point-in-time percentile thresholds** (`rolling_percentile_*`): the rolling quantile uses only forward returns whose horizon has already ended at the decision row (availability index = `t + horizon`); thresholds are null until `min_samples` realized outcomes exist. Check: `threshold_*` null for the first `horizon + min_samples` rows; changing a future price never alters an earlier threshold.
8. **Session-aware forward returns**: `session_col` makes shifts `.over([*group_cols, session_col])`; time horizons use `join_asof` with `tolerance` and null out-of-tolerance matches. Check: the last `horizon` bars per session have null forward returns.
9. **Calendar-aware barriers**: `calendar_aware_labels` derives session ids from a `TradingCalendar` so barrier windows never span maintenance windows, overnight gaps, weekends or holidays.
10. **Mandatory temporal ordering**: `compute_features` needs a detected or declared time column (else `DataSchemaError(... or set assume_sorted=True)`), sorts by `[*group_cols, timestamp]` before any feature; labelers sort by group + timestamp and warn when a specified timestamp column is missing. Set `assume_sorted=True` only when row order is truly temporal.
11. **Per-asset isolation**: every expression is `.over(group_cols)`; `group_col=[]` is refused when a recognized asset column exists (`"Grouping cannot be disabled while a recognized asset column is present"`); nulls in group/time columns raise `DataSchemaError`.
12. **Feature-store PIT join**: `join_asof(strategy="backward", tolerance)` guarantees `feature.timestamp <= label.timestamp`. Caveat: this prevents lookahead only if the stored timestamp is the time the value became KNOWN (publication time); the library does not shift by publication lag (do that in `ml4t-data` / caller).

Labels and sampling:

13. **OHLC-based barrier touch**: intra-bar high/low decide touches; `_resolve_barrier_exit` uses the open for gap-through-barrier, avoiding optimistic close-only fills; trailing stops via `_initialize_trailing_stop` / `_update_trailing_stop`.
14. **No NaN / non-positive prices**: `validate_price_no_nans` and `_finite_float_array(positive=True)` raise before any barrier search; fix upstream in `ml4t-data`, do not fill.
15. **Label overlap accounting**: `calculate_uniqueness=True` yields `label_uniqueness` and `sample_weight`; `sequential_bootstrap` draws less-redundant sets. Overlapping labels inflate effective sample size and break IID assumptions in CV.
16. **LabelingConfig validators**: `validate_side`, `reject_boolean_numeric_setting` (a `True` barrier would silently become 1.0), `validate_finite_numeric_setting` (rejects NaN/inf and nonpositive barriers), `validate_max_horizon` (`max_horizon >= min_horizon`).
17. **min_samples floor** for percentile labels: `max(1, min(1008, lookback_window // 10))` for int lookbacks, `100` for duration lookbacks; must be a positive int (bool rejected).
18. **Bar warm-up and completeness**: adaptive samplers hold initial expectations for `min_bars_warmup=10` bars with EWMA `alpha=0.1`; `include_incomplete=False` drops the trailing partial bar. Drop the first ~10 adaptive bars before modeling (inference).
19. **Run-bar definition**: `θ_T = max{Σ buys, Σ sells}` over the whole bar (AFML 2.3); naive "consecutive same-side run" implementations are explicitly called out as wrong.
20. **Meta-label input validation**: non-finite scalars and non-numeric columns are rejected (nulls permitted).

Specs, registry and execution:

21. **Spec validation before execution**: unknown feature name -> `ValueError` listing the registry; duplicate `output`; `output` equal to an input column; unknown params -> `ValueError` listing accepted params. Fail fast, no partial results.
22. **Dependency ordering**: features with `dependencies` run after their inputs (Kahn); cycles raise `ValueError`.
23. **Never overwrite columns**: outputs are appended; raw inputs and earlier features are untouched. Temporary columns use collision-proof names (`__ml4t_row_index`, `__ml4t_availability_row`, ...) and are dropped before return.
24. **Required column args**: a required parameter outside `COLUMN_ARG_MAP` and not supplied raises `ValueError` with the exact `compute_features(df, [{"name":..., "params": {...}}])` fix; a feature `TypeError` is re-raised as `ValueError` with signature and attempted args.
25. **Authoritative warm-up**: every feature declares a lookback calculator (`_parameter`, `_scaled_parameter`, `_sum`, `_maximum`, bespoke `_stochastic`, `_stochrsi`, `_variance_ratio`, `_ulcer_index`). Use `features.lookback(name, **params)` to trim leading rows or size purge windows; `features.by_lookback(max_lookback)` to filter.
26. **Schema and parameter checks**: `validate_ohlcv_schema(require_asset_id=True)`; windows/periods >= 1, thresholds in [0,1], probabilities 0-1, percentages 0-100, lags bounded, positive values, column existence.
27. **Scaler contracts**: `transform` before `fit` -> `NotFittedError`; missing fitted columns raise; inf/NaN rejected at fit; `PreprocessingPipeline._apply_transform(use_training_boundary=True)` applies training clip/winsorize bounds to test data.
28. **Store write safety**: `mode="fail"` protects against overwrite; empty frames and non-Polars inputs rejected; `_quote_identifier` rejects unsafe identifiers (but `filter_expr` is raw SQL).
29. **Degenerate-bar handling**: A/D multiplier 0 when High == Low; `safe_hurst` guards R/S failures; `var`/`stddev` use `ddof=0` to match TA-Lib.
30. **Deterministic seasonality**: `fourier_features` uses sin/cos of row position only. **Numba fallback**: without Numba decorators are identity (pure Python); `warm_rolling_callback(callback, window)` compiles kernels before `rolling_map`.
31. **Performance/data-quality logging**: `logged_feature` / `PerformanceTracker` warn above `warn_threshold_ms=1000.0`; `log_data_quality=True` reports NaN/inf counts.

## Usage patterns

Config-driven features on a panel (auto-detects `symbol`/`timestamp`; outputs must be distinct):
```python
import polars as pl
from ml4t.engineer import compute_features, features
feats = compute_features(ohlcv, ["rsi", "macd", "atr"])            # names only
feats = compute_features(df, [
    {"name": "sma", "params": {"period": 20}, "output": "sma_20"},
    {"name": "sma", "params": {"period": 50}, "output": "sma_50"},
], group_col="symbol", timestamp_col="date")
feats = compute_features(df, "config/features.yaml")                 # YAML spec list
warm = features.lookback("rsi", period=14)                           # first usable row
features.list(category="momentum"); features.search("volatility"); features.describe("rsi")
```

Direct Polars expressions (composable, per-asset):
```python
from ml4t.engineer.features.utils.helpers import add_indicators
from ml4t.engineer.features.trend.ema import ema
from ml4t.engineer.features.volatility.atr import atr_polars
from ml4t.engineer.features.momentum.rsi import rsi
df = add_indicators(df, {"ema_20": ema("close", 20),
                         "atr_14": atr_polars("high", "low", "close", 14)})
df = df.with_columns(rsi("close", period=14).over("symbol").alias("rsi_14"))
```

Alternative bars and fractional differencing:
```python
from ml4t.engineer.bars import DollarBarSamplerVectorized, ImbalanceBarSampler, TickRunBarSampler
bars  = DollarBarSamplerVectorized(dollars_per_bar=1_000_000).sample(ticks)   # ticks: timestamp, price, volume
ibars = ImbalanceBarSampler(expected_ticks_per_bar=100, alpha=0.1).sample(ticks, include_incomplete=False)  # needs side
rbars = TickRunBarSampler(expected_ticks_per_bar=50).sample(ticks)
from ml4t.engineer.features.fdiff import ffdiff, find_optimal_d
best = find_optimal_d(close, d_range=(0.0, 1.0), step=0.01, adf_pvalue_threshold=0.05)
df = df.with_columns(ffdiff("close", d=best["d"], threshold=1e-5).alias("close_ffd"))
```

Triple barrier / ATR barrier with reproducible config and uniqueness weights:
```python
from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import triple_barrier_labels, atr_triple_barrier_labels, trend_scanning_labels
cfg = LabelingConfig.triple_barrier(upper_barrier=0.02, lower_barrier=0.01, max_holding_period=20)
cfg.to_yaml("labeling.yaml")
labeled = triple_barrier_labels(df, config=cfg, calculate_uniqueness=True)   # + label_uniqueness, sample_weight
atr_cfg = LabelingConfig.atr_barrier(atr_tp_multiple=2.0, atr_sl_multiple=1.0, atr_period=14)
short_labels = atr_triple_barrier_labels(df, config=atr_cfg, side=-1)       # side may also be a column name
ts = trend_scanning_labels(df, min_window=5, max_window=50, t_value_threshold=2.0)  # label, t_value, optimal_window
```

Rolling percentile labels (point-in-time, session-aware) and meta-labeling:
```python
from ml4t.engineer.labeling import rolling_percentile_binary_labels, meta_labels, apply_meta_model
lab = rolling_percentile_binary_labels(df, horizon=30, percentile=95, direction="long",
                                       lookback_window=252 * 24 * 12, session_col="session_date")
lab["label_long_p95_h30"].mean()                                   # ~0.05 by construction
ml = meta_labels(df, signal_col="signal", return_col="fwd_ret", threshold=0.0)
sized = apply_meta_model(df, "signal", "meta_prob", bet_size_method="sigmoid", scale=5.0)  # -> sized_signal
```

Leakage-safe CV with purged splitter from `ml4t-diagnostic`:
```python
from ml4t.engineer import MLDatasetBuilder, StandardScaler
from ml4t.diagnostic.splitters import PurgedWalkForwardCV
builder = MLDatasetBuilder(features_df, labels_series, dates=dates_series).set_scaler(StandardScaler())
for fold in builder.split(PurgedWalkForwardCV(n_splits=5, embargo_pct=0.01)):
    model.fit(fold.X_train, fold.y_train)                          # scaler fit on fold.train only
    preds = model.predict(fold.X_test)
X_tr, X_te, y_tr, y_te = builder.train_test_split(train_size=0.75) # shuffle=False enforced with dates
cuts = builder.get_feature_percentiles(train_indices=fold.train_indices)   # train-only bins
```

Experiment config, persisted scaler and point-in-time feature store:
```python
from ml4t.engineer.config import load_experiment_config
from ml4t.engineer.store.offline import OfflineFeatureStore
cfg = load_experiment_config("experiment.yaml")                    # .features, .labeling, .preprocessing
df = compute_features(df, cfg.features); scaler = cfg.preprocessing.create_scaler()
train_s = scaler.fit_transform(train_df); test_s = scaler.transform(test_df); state = scaler.to_dict()
with OfflineFeatureStore("features.duckdb") as store:
    store.save_features(df, "rsi_features", mode="replace")
    joined = store.point_in_time_join(labels_df, "rsi_features", join_keys=["symbol"], tolerance="1h")
```

## Defaults and configuration keys

| Key | Default |
|---|---|
| `compute_features` | `group_col=None` (auto), `timestamp_col=None` (auto), `assume_sorted=False`; spec keys `name`, `params`, `output` (defaults to `name`) |
| `@feature` | `lookback=None`, `normalized=False`, `value_range=None`, `ta_lib_compatible=False`, `input_type="close"`, `output_type="indicator"` |
| `FeatureCatalog` | `search(max_results=10)`, `list(limit=None)` |
| Bars | `alpha=0.1`, `initial_p_buy=0.5`, `min_bars_warmup=10`, `include_incomplete=False`; window samplers `bar_window=10`, `tick_window=1000`; sizing hints `ticks_per_bar=100`, `volume_per_bar=1000`, `dollars_per_bar=1_000_000`, run `expected_ticks_per_bar=50` (tick) / `100` (volume, dollar) |
| `LabelingConfig.triple_barrier` | `upper_barrier=0.02`, `lower_barrier=0.01`, `max_holding_period=20`, `side=1`, `trailing_stop=False` |
| `LabelingConfig.atr_barrier` | `atr_tp_multiple=2.0`, `atr_sl_multiple=1.0`, `atr_period=14`, `max_holding_period=20` |
| `LabelingConfig.fixed_horizon` | `horizon=10`, `return_method="returns"`, `threshold=None` |
| `LabelingConfig.trend_scanning` | `min_horizon=5`, `max_horizon=20`, `t_value_threshold=2.0`, `step=1` |
| Percentile labels | `lookback_window=252*24*12` (72576 bars, ~1 year of 5-min bars), `min_samples=max(1, min(1008, lookback//10))` (100 for duration lookbacks), `direction="long"` |
| Meta-labels | `meta_labels(threshold=0.0)`; `compute_bet_size(method="sigmoid", scale=1.0, threshold=0.5)`; `apply_meta_model(scale=5.0, output_col="sized_signal")` |
| Uniqueness | `uniqueness_weight_scheme="returns_uniqueness"`; `sequential_bootstrap(with_replacement=True)` |
| Calendars | `EquityCalendar(exchange="NYSE")`, `CryptoCalendar(maintenance_window=None)`, `SimpleTradingCalendar(gap_threshold_minutes=30)`, `filter_sessions(timestamp_col="timestamp")`; RTH filter `"09:30"-"16:00"` |
| Scalers | `StandardScaler(ddof=1)` (TA-Lib `stddev` uses `ddof=0`), `MinMaxScaler(feature_range=(0.0, 1.0))`, `RobustScaler(quantile_range=(25.0, 75.0))`, `PreprocessingConfig(scaler="standard")`, `PreprocessingPipeline(min_confidence=0.0, winsorize_limits=(0.01, 0.99))` |
| Dataset builder | `train_test_split(train_size=0.8, shuffle=False, random_state=None)`; `get_feature_percentiles(quantiles=[0.1,0.25,0.5,0.75,0.9])`; `compute_label_percentiles(n_quantiles=5)`; `create_dataset_builder(scaler="standard")` |
| Config I/O | `to_json(indent=2)`, `to_dict(exclude_none=False, mode="python")`, `load_experiment_config(validate=True)`, `save_experiment_config(include_defaults=False)` |
| TA periods | 14: adx, adxr, aroon, cmo, di/dx, dm, imi, mfi, rsi, stochastic fastk, willr, atr, natr, midprice, midpoint, avgdev, linearreg*, tsf, ulcer_index; 12/26/9: macd, apo, ppo; 20: cci, bollinger, donchian; 10: mom, roc, rocp, rocr; 30: trix, dema, tema, trima, kama, trend_intensity, math; 5: stochf fastk, stochrsi fastk, stddev, var, t3 (vfactor 0.7); stochastic slowk 1 / slowd 3; stochf/stochrsi fastd 3; ultosc 7/14/28 weights 4/2/1; sar 0.02/0.2; adosc 3/10 |
| Volatility | estimators `period=20, annualize=True, trading_periods=252`; `ewma_volatility(span=120, mu=-5.0, sigma=2.0)`; `garch_forecast(omega=1e-5, alpha=0.1, beta=0.85, horizon=1)`; `volatility_percentile_rank(20, 252)`; `volatility_regime_probability(0.01, 0.02, 20, 100)` |
| Regime | `choppiness 14`, `fractal_efficiency 10`, `hurst period=100, min_lag=2, max_lag=None (kernel 100)`, `market_regime_classifier adx=25.0, chop 61.8/38.2` |
| Risk | `window=252`, `confidence_level=0.95`, `trading_periods=252`, `method="historical"`, `ulcer_index 14` |
| Structural break | `coefficient_of_variation 50`, `rolling_cv_zscore 50 x5`, `variance_ratio 100, q=5`, `rolling_kl_divergence 100, n_bins=20`, `rolling_wasserstein 100`, `rolling_drift 100, normalize=True` |
| Microstructure | `period=20` (amihud, kyle, realized_spread, roll, trade_intensity, VAP, synchronicity, VWPM); `price_impact_ratio(impact_threshold=0.001)`; `quote_stuffing_indicator(period=5)`; `bid_ask_imbalance(period=1)`; `volume_at_price_ratio(n_bins=10)` |
| Cross-asset / composite | `rolling_correlation 20`, `beta_to_market 60`, `correlation_regime_indicator 0.3/0.7/20`, `lead_lag_correlation max_lag=10, window=20`, `transfer_entropy lag=1, window=100, bins=10`, `co_integration_score 60`, `cross_asset_momentum 20, "rank"`; `rolling_z_score 252`, `z_score_composite 252`, `illiquidity/momentum_composite 20` |
| FFD / entropy | `threshold=1e-5`, `max_length=10000`, `d_range=(0.0, 1.0)`, `step=0.01`, `adf_pvalue_threshold=0.05`; `rolling_entropy(50, n_bins=10)`, `rolling_entropy_lz(100, "quantile")`, `rolling_entropy_plugin(50, word_length=1)` |
| Store / logging / plots | `OfflineFeatureStore(read_only=False)`, `save_features(mode="replace")`, `point_in_time_join(timestamp_col="timestamp", strategy backward)`; `LoggingConfig(level=INFO, warn_threshold_ms=1000.0)`; `plot_correlation_heatmap(cmap="RdBu_r", figsize=(10, 8), fmt=".2f")` |
| Validation | `validate_ohlcv_schema(require_asset_id=True, allow_flexible_time=True)`, `validate_threshold(min_val=0.0, max_val=1.0)` |

## Where the book uses it

The notes were extracted from the library source and companion-repo READMEs; the chapter mapping below is by topic (inference), not verified against the PDF.

| Book chapter / notebook area | Library pieces | Evidence in notes |
|---|---|---|
| Ch. 03 market microstructure (`03_market_microstructure/`) | `bars/*` (tick/volume/dollar/imbalance/run), `microstructure/*` (Kyle lambda, Amihud, Roll, micro-price, tick rule) | AFML ch. 2.3 formulas implemented verbatim; order-book features expect ALGOSeek-style TAQ quotes |
| Ch. 07 defining the learning task (`07_defining_the_learning_task/03_label_methods`) | `triple_barrier_labels`, `atr_triple_barrier_labels`, `trend_scanning_labels`, `fixed_time_horizon_labels`, `rolling_percentile_*`, `meta_labels`, uniqueness / `sequential_bootstrap` | AFML ch. 3-4; OHLC-based touches, point-in-time percentile thresholds, session awareness |
| Ch. 08 financial features (`08_financial_features/`) | TA-Lib-compatible momentum/trend/volatility/volume, volatility estimators (Parkinson, GK, RS, YZ), `fdiff` (FFD), composites, cross-asset | TA-Lib compatibility flags; `find_optimal_d` minimum d passing ADF p<0.05 |
| Ch. 09 model-based features (`09_model_based_features/`) | `garch_forecast` (fixed params), `ewma_volatility`, regime (`hurst_exponent`, `variance_ratio`, `market_regime_classifier`), structural-break features, entropy | ADIA Lab 2025 structural-break insights; AFML ch. 18 entropy |
| Ch. 11 ML pipeline (`11_ml_pipeline/`) | `MLDatasetBuilder`, scalers, `PreprocessingPipeline.from_recommendations`, `SplitterProtocol` with `PurgedWalkForwardCV` | AFML ch. 7 train-only preprocessing and strict fold boundaries |
| Ch. 19 risk management | `risk.py` (VaR/CVaR methods, drawdown, tail ratio, Sharpe/Sortino/Calmar/Omega) as features | rolling window 252, confidence 0.95 |
| Ch. 25-26 live trading / MLOps | `OfflineFeatureStore.point_in_time_join`, `ExperimentConfig` YAML, `scaler.to_dict()`, `LoggingConfig` | PIT join caveat (publication time vs observation time) |

## Glossary

- **Registry feature**: function decorated with `@feature`, discoverable by name, executed by `compute_features` via signature-aware dispatch.
- **Lookback / warm-up**: leading rows a feature needs before its first valid output; computed by `core/lookbacks.py`, exposed as `features.lookback()`.
- **Normalized feature**: bounded, scale-free output flagged `normalized=True` ("ML-ready").
- **Information-driven bars**: bars sampled on activity (ticks, volume, dollars) or order-flow imbalance/runs instead of clock time.
- **θ (theta)**: cumulative imbalance (TIB Σ signs; VIB Σ sign×volume) or run statistic (max of buy vs sell totals) that closes a bar when it exceeds `E[θ_T]`.
- **E[T]**: EWMA-estimated expected ticks per bar; `alpha` is the EWMA weight; `min_bars_warmup` bars use initial expectations.
- **Triple barrier**: label by first touch of profit-take, stop-loss or vertical (max-holding) barrier; `barrier_hit` records which.
- **ATR-adjusted barrier**: barrier distance = multiple × ATR(`atr_period`), adapting to volatility.
- **Trailing stop**: stop that ratchets with favorable price movement.
- **Trend scanning**: choose the forward window (min..max by step) maximizing |t| of a price-on-time regression; label = sign of slope.
- **Meta-label**: binary label of whether the primary signal was profitable; a secondary model sizes bets (`sized_signal`).
- **Uniqueness / concurrency**: concurrency = labels alive at a bar; uniqueness = mean 1/concurrency over a label's life.
- **Sequential bootstrap**: resampling that favours candidates with high expected uniqueness.
- **Availability index**: row index/timestamp at which a forward return becomes known (`t + horizon`); drives point-in-time thresholds.
- **Session-aware**: computations never cross `session_col` boundaries or calendar breaks.
- **Point-in-time join**: backward as-of join so features are dated at or before the label timestamp.
- **Train-only preprocessing**: fit scalers/quantiles on training rows only, lock statistics, transform test rows.
- **Strict temporal boundary**: `max(train dates) < min(test dates)`, enforced per fold.
- **Purge / embargo**: drop training samples overlapping test label windows (purge) and a gap after the test set (embargo); supplied by `ml4t-diagnostic` splitters, verified by the builder.
- **FFD**: fixed-width-window fractional differencing with weight cutoff `threshold`; `d` the differencing order.
- **Composite score**: weighted sum of rolling z-scores of several features.
- **Micro-price / weighted mid**: size-weighted bid/ask mid-point.
- **Tick rule**: trade-sign classification from price changes (`effective_tick_rule`).
- **Winsorize**: clip to quantile limits (default 1%/99%).
- **Choppiness Index**: 0-100; >61.8 choppy, <38.2 trending. **Hurst exponent**: R/S persistence; 0.5 random walk. **Variance ratio (Lo-MacKinlay)**: Var(q-period)/(q·Var(1-period)); 1 under a random walk. **CV gate**: coefficient of variation as a regime gate (ADIA 2025).
- **TA-Lib compatible**: numerically identical to TA-Lib (init method, `ddof=0`, unstable periods).
- **ArtifactSpec / FeatureSpec / LabelSpec / PredictionSpec**: persisted-artifact contracts (schema + definition) shared across ml4t libraries. **DataContractConfig**: canonical column-name mapping across libraries.

## Related references

- `libraries/ml4t_data.md` — upstream OHLCV / tick frames, `asset_id`/`timestamp` contract, where to apply publication lags and fix NaN prices.
- `libraries/ml4t_diagnostic.md` — `PurgedWalkForwardCV` and other splitters consumed by `MLDatasetBuilder.split`; feature-evaluation configs that moved out of engineer; recommendations for `PreprocessingPipeline`.
- `libraries/ml4t_models.md` — fitted GARCH / HMM / regime models (engineer's `garch_forecast` uses fixed parameters).
- `libraries/ml4t_backtest.md`, `libraries/ml4t_live.md` — consume `sized_signal`, `label_return`, `barrier_hit`, persisted scaler state.
- `chapters/03_market_microstructure.md` — alternative bars, tick rule, Kyle/Amihud/Roll features.
- `chapters/07_defining_the_learning_task.md` — label methods (triple barrier, trend scanning, meta-labels, uniqueness).
- `chapters/08_financial_features.md` — TA indicators, volatility estimators, fractional differencing.
- `chapters/09_model_based_features.md` — GARCH, regime and structural-break features.
- `chapters/11_ml_pipeline.md` — leakage-safe datasets, purged CV, train-only preprocessing.
- `chapters/26_mlops_governance.md` — feature store, experiment configs, persisted preprocessing state.
- `guardrails.md`, `workflow.md`, `glossary.md` — cross-cutting leakage rules and stage sequencing.
- `case_studies/nasdaq100_microstructure.md`, `case_studies/cme_futures.md` — tick data, session gaps and calendar-aware labeling contexts.

Further reading:
- López de Prado, M. (2018). *Advances in Financial Machine Learning* — ch. 2.3 imbalance/run bars, ch. 3 triple barrier / meta-labeling / trend scanning, ch. 4 uniqueness and sequential bootstrap, ch. 5 fractional differencing, ch. 7 purged CV, ch. 18 entropy.
- Kontoyiannis et al. (1998) — LZ entropy estimator (`rolling_entropy_lz`).
- Cahan & Luo (2013) — composite z-score factors.
- Lo & MacKinlay (1988) — variance ratio test. Wilder (1978) — ATR / RSI / ADX. Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang — range-based volatility estimators.
- ADIA Lab (2025) structural-break challenge — CV gate, KL/Wasserstein/drift window features.
