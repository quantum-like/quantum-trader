# ml4t-models library (latent factors, SDF, direct prediction, portfolio learning)

> `ml4t-models` (v0.1.4, import `ml4t.models`) holds the book's finance-specific model families: latent-factor estimators (PCA, RP-PCA, IPCA, conditional autoencoder), a deep stochastic discount factor (SDF), a supervised autoencoder for direct asset prediction, and end-to-end portfolio learners (linear, LSTM, deep). It sits at the "model" stage of the ML4T workflow, after `ml4t-engineer` builds the characteristics panel and before `ml4t-backtest` / `ml4t-diagnostic` evaluate it. Its design principle: finance-native batch contracts instead of generic dataloaders, an explicit split between structural estimation (betas, factors, SDF weights) and prediction (premia forecasts, mappers, heads), checkpoint-aware neural training, and integration boundaries with sibling libraries instead of duplicated evaluation logic. Notes were compiled from the package README, `configs/`, `types.py` and targeted source lookups; training-loop internals are marked (inference).

## Install and import

| Item | Value |
|---|---|
| Package / import | `pip install ml4t-models` -> `import ml4t.models` (lazy `__getattr__` root exports from `_public_api.py`, the "authoritative stable root export manifest"; importing never imports torch) |
| Python | >= 3.12 (3.12/3.13/3.14 stable) |
| Core dependency | `numpy>=1.26,<2.6` only |
| `[deep]` | `torch>=2.13.0` -- needed for `CAEModel`, `SAEModel`, `StochasticDiscountFactorModel`, `LSTMPortfolioModel`, `DeepPortfolioModel`; numpy-only models fail at `fit` with a clear error if torch is absent (inference) |
| `[integration]` | `polars>=1.0.0` + `ml4t-specs>=0.1.0b0` -- long-frame batch builders, `ResultsFrame.to_polars()`, `FeedSpec` bridge to `ml4t.backtest.DataFeed` |
| `[all]` | both extras; `[dev]`, `[docs]` (mkdocs-material) |
| Dev gates | `uv sync --locked --dev --extra docs`; `ruff check`, `ruff format --check`, `ty check`, `pytest --cov-report=json:coverage.json` + `scripts/ci/check_coverage.py`, `mkdocs build --strict`, `uv build` |
| Package layout | `api.py` (Protocols), `types.py` (batches/results), `pipelines.py`, `configs/`, `latent_factors/`, `forecasters/`, `mappers/`, `asset_prediction/`, `stochastic_discount_factor/`, `portfolio/`, `integration/{data,surfaces,backtest}.py`, `_internal/` |

Only names in `_public_api.py` are stable API; `ml4t.models.portfolio` also lazy-loads.

## API map by task

### Protocols (`api.py`)

| Protocol | Methods | Notes |
|---|---|---|
| `LatentFactorModel` | `fit(batch: PanelBatch) -> FitSummary`; `extract(batch, *, checkpoint=None) -> LatentFactorState` | `PanelBatch = PersistentPanelBatch \| CrossSectionBatch` (inference) |
| `FactorForecaster` | `fit(state: LatentFactorState) -> FitSummary`; `predict(state) -> FactorForecastResult` | fits on training-sample factor returns only |
| `AssetMapper` | `predict(state, factor_forecast) -> AssetForecastResult` | beta x lambda |
| `StochasticDiscountFactorEstimator` | `fit(batch: CrossSectionBatch)`; `extract(batch, *, checkpoint: tuple[str,int] \| int \| None) -> StochasticDiscountFactorState` | weight-native |
| `AssetPredictionModel` | `fit(batch, *, validation_batch=None)`; `predict(batch, *, checkpoint=None) -> AssetSignalResult` | SAE |
| `PortfolioModel` | `fit(batch: PortfolioSequenceBatch, *, validation_batch=None) -> FitSummary`; `predict(batch, *, checkpoint=None) -> PortfolioWeightsResult` | |
| `PortfolioPostprocessor` | `transform(batch, weights) -> PortfolioWeightsResult` | constraints |

### Latent-factor estimation (`latent_factors/`)

All subclass `BaseLatentFactorModel[ConfigT](FitObservable, ABC)` with `is_fitted()`, `available_checkpoints() -> tuple[int, ...]`, `save(path) -> Path`, `load(path, *, device=None)`, `last_fit_record()`.

| Class | Signature | Purpose | Notes |
|---|---|---|---|
| `PCAModel` | `PCAModel(PCAConfig)`; `fit(batch)`, `extract(batch, *, checkpoint=None)` | static loadings + factor returns from a fixed universe | requires `PersistentPanelBatch` (`_require_persistent_panel`); `_resolve_asset_order` lets extract batches permute assets but they must contain the fitted universe (inference) |
| `RPPCAModel` | `RPPCAModel(RPPCAConfig)`; same API | risk-premium PCA (Lettau-Pelger): eigendecompose second moment/covariance + `gamma * mean mean'` | `base_moment="covariance"` with `gamma=0` is plain PCA (inference); internals `_cross_section_normalization`, `_risk_premium_matrix`, `_sign_normalize_loadings`, `_orthogonalize`, `_time_series_betas`, `_factor_sharpes` |
| `IPCAModel` | `IPCAModel(IPCAConfig)`; `fit`, `extract`; properties `gamma -> (L_aug, K)`, `train_factor_returns` | instrumented PCA (Kelly-Pruitt-Su): `beta_{i,t} = z_{i,t} Gamma` | requires `CrossSectionBatch`; alternating `_estimate_gamma` / `_estimate_factors` on per-date `Z'Z`, `Z'y` sufficient statistics; `_normalize_theta_y` rotates into the KPS ThetaY identification set; `_augment_chars` may add a constant column (inference) |
| `CAEModel` | `CAEModel(CAEConfig)`; `fit(batch, *, validation_batch=None, patience=50)`, `extract(batch, *, checkpoint=None)`, `extract_per_member(batch, *, checkpoint=None) -> list[LatentFactorState]` | conditional autoencoder (Gu-Kelly-Xiu): beta net on characteristics, factor net on managed portfolios | requires `CrossSectionBatch`; `_compute_managed_portfolios`, `_flatten_training_panel`, `_validation_loss(task_type)`; `extract_per_member` gives one state per ensemble member |

### Factor-premium forecasting and mapping (`forecasters/`, `mappers/`)

| Class / function | Signature | Purpose | Notes |
|---|---|---|---|
| `ExpandingMeanFactorForecaster` | `(ExpandingMeanForecasterConfig \| None)`; `fit(state)`, `predict(state)` | premia = training-sample mean of `state.factor_returns` | default choice in README quickstart |
| `AR1FactorForecaster` | `(AR1ForecasterConfig \| None)` | independent AR(1) per factor | |
| `EWMABaseFactorForecaster` | `(EWMABaseForecasterConfig \| None)` | exponentially weighted mean | `half_life=12.0` |
| `require_estimable_factor_returns` | `(state) -> np.ndarray` | raises if `state.factor_returns is None` | extraction batch had no returns |
| `BetaLambdaMapper` | `(config: MapperConfig \| None = None).predict(state, factor_forecast) -> AssetForecastResult` | `expected_returns[t,i] = sum_k asset_betas[t,i,k] * factor_premia[t,k]` (inference from shapes) | `MapperConfig.model_name="beta_lambda"` |
| `LatentFactorForecastPipeline` | `(model, forecaster, mapper)`; `fit(batch) -> PipelineFitResult`; `predict(batch, *, checkpoint=None) -> LatentFactorPrediction(state, factor_forecast, asset_forecast)` | wires the three stages | fit on the training batch, `predict` on later batches |

### Stochastic discount factor (`stochastic_discount_factor/`)

| Class / function | Signature | Purpose | Notes |
|---|---|---|---|
| `StochasticDiscountFactorModel` | `(StochasticDiscountFactorConfig)`; `fit(batch, *, validation_batch=None, patience=None)`; `extract(batch, *, checkpoint: SDFCheckpoint \| int \| None = None) -> StochasticDiscountFactorState`; `available_checkpoints() -> tuple[SDFCheckpoint, ...]`; `save`/`load(path, device=None)` | Chen-Pelger-Zhu GAN SDF: SDF net (LSTM macro state) vs moment net (learned instruments), phases unconditional -> moment -> conditional | `SDFCheckpoint = (phase, epoch)`; `_validation_metrics` reports per-phase CPZ validation loss and SDF-portfolio Sharpe on the validation panel |
| `LinearStochasticDiscountFactorReturnMapper` | `()`; `fit(state, batch)`, `predict(state) -> AssetForecastResult`; `save`/`load` | linear projection of SDF weights to expected returns | `expected_return_mapper="linear"` ClassVar |
| `StochasticDiscountFactorBetaNetworkHead` | `(config)`; `fit(state, batch, *, validation_state=None, validation_batch=None)`, `predict(batch, *, checkpoint=None) -> AssetSignalResult`, `available_checkpoints()`, `save`/`load` | "paper-faithful beta-network predictive head" regressing returns on SDF exposure | `beta_*` config keys |
| `_internal/stochastic_discount_factor_nn` | `StochasticDiscountFactorNetwork(n_asset_features, n_context_features=0, state_dim=4, hidden_dim=64, dropout=0.05)`, `MomentNetwork(..., n_instruments=8, state_dim=32, dropout=0.05)`, `BetaNetwork(..., state_dim=4, hidden_dim=64, dropout=0.05)`, `construct_stochastic_discount_factor(returns, weights, mask)`, `unconditional_loss(weights, returns, mask, n_obs_per_asset)`, `conditional_loss(weights, instruments, returns, mask, n_obs_per_asset)`, `compute_sharpe(sdf)`, `get_segment_ids(mask)` | building blocks | SDF net forward takes `h0, c0`, returns `(weights, (h, c))` |

### Direct asset prediction (`asset_prediction/`)

| Class | Signature | Purpose | Notes |
|---|---|---|---|
| `SAEModel` | `(SAEConfig)`; `fit(batch, *, validation_batch=None)`, `predict(batch, *, checkpoint=None) -> AssetSignalResult`, `available_checkpoints()`, `save`/`load(path, device=None)` | supervised autoencoder (Jane-Street style): decoder + aux head from bottleneck + main head | requires `CrossSectionBatch`; `_flatten_supervision` pools masked-in rows; `_iterate_batches(batch_size, seed)`; `_sae_loss(task_type, alpha, aux_weight)` |
| `SupervisedAutoencoder` (`_internal/sae_nn`) | `(n_features, n_labels=1, hidden_units=None, dropout_rates=None, noise_std=0.035, output_activation="sigmoid")`; `forward -> (decoded, aux_pred, main_pred)`, `get_betas`, `predict` | network | `Swish`, `GaussianNoise(std=0.1)` nn default vs config `noise_std=0.035` |

### Portfolio learning (`portfolio/`)

All subclass `BasePortfolioModel(FitObservable, ABC)`: `fit(batch, *, validation_batch=None)`, `predict(batch, *, checkpoint=None)`, `available_checkpoints()`, `save`, `load(path, device=None)`.

| Class / function | Signature | Purpose | Notes |
|---|---|---|---|
| `LinearFeaturePortfolioModel` | `(LinearPortfolioConfig)` | pooled ridge of forward returns on features -> scores -> normalized cross-sectional weights | numpy-only; `available_checkpoints()` returns `(1,)` once fitted; `_intercept()`, `_feature_coefficients()` |
| `LSTMPortfolioModel` | `(LSTMPortfolioConfig)`; policy `LSTMPortfolioPolicy(n_assets, n_features, n_groups, config)` | recurrent policy trained on robust Sharpe | `validation_batch = validation_batch or batch` (training batch reused when none given) |
| `DeepPortfolioModel` | `(DeepPortfolioConfig)`; policy `DeepPortfolioPolicy(n_assets, n_features, n_groups, adjacency_mask, config)` | DeePM-style allocator "producing bounded risk weights" | same validation fallback; `_prediction_adjacency_mask` |
| Components | `StaticContextEncoder(n_assets, n_groups, config)`, `FiLM`, `VariableSelection(n_features, d_model, context_dim, hidden_dim, dropout)`, `AdapterBlock(d_model, hidden_mult, dropout)`, `TemporalAttentionBlock` (causal), `CrossSectionalAttention(lag)`, `MacroGraphAttention(adjacency_mask)` | building blocks | causal by construction (see guardrails) |
| Losses | `compute_net_portfolio_returns(weights, forward_returns, vol_scale, mask, costs, gamma_cost, prev_weights=None, turnover_penalty=0.0)`; `sharpe_ratio(returns, annualization_factor, eps, dim=None)`; `softmin_sharpe(window_sharpes, tau)`; `robust_sharpe_loss(weights, forward_returns, vol_scale, mask, costs, burn_in, gamma_cost, annualization_factor, eps, tau, lambda_soft, prev_weights=None, turnover_penalty=0.0) -> PortfolioLossOutput` | "pooled-plus-softmin Sharpe objective", costs charged inside the loss | |
| Postprocessors | `WeightConstraintPostprocessor(gross_exposure=1.0, net_exposure=0.0, max_abs_weight=None, turnover_limit=None).transform(batch, weights)`; `normalize_cross_sectional_weights(weights, *, mask, gross_exposure, net_exposure, max_abs_weight)`; `apply_turnover_limit(weights, *, previous_weights, mask, turnover_limit)` | implementable weights | turnover limit uses `batch.prev_weights` |
| `PortfolioAllocationPipeline` | `(model, *, postprocessors=())`; `fit(batch, *, validation_batch=None) -> PortfolioPipelineFitResult`; `predict(batch, *, checkpoint=None) -> PortfolioPrediction(raw_weights, processed_weights)` | model + constraints | keeps raw and processed weights |
| Runtime | `fit_policy_network(policy, batch, validation_batch, config, device) -> PortfolioTrainingArtifacts`; `evaluate_pooled_sharpe(...) -> float`; `mask_tensor`, `group_ids_tensor`, `costs_tensor`, `previous_weights_tensor`, `adjacency_mask_tensor`, `cpu_state_dict` | training loop | `fit_policy_network` requires a validation batch |

### Integration: frames and backtest hand-off (`integration/`)

| Function | Signature | Purpose |
|---|---|---|
| `resolve_dataset_schema` | `(frame, *, schema=None, timestamp_col=None, entity_col=None, timestamp_candidates=("timestamp","datetime","date","time"), entity_candidates=("asset","symbol","ticker","instrument","security")) -> ResolvedDatasetSchema` | find time/entity columns from names or ml4t-specs metadata |
| `persistent_panel_batch_from_long_frame` | `(frame, *, schema=None, return_col=None, feature_cols=(), timestamp_col=None, entity_col=None, metadata=None)` | long frame -> `PersistentPanelBatch` (PCA / RP-PCA) |
| `cross_section_batch_from_long_frame` | `(frame, *, feature_cols (required), return_col=None, context_cols=(), ...)` | long frame -> `CrossSectionBatch` (IPCA / CAE / SDF / SAE) |
| Frame builders | `predictions_frame_from_asset_forecast(forecast, *, constants=None)`, `predictions_frame_from_asset_signal(signal, ...)`, `signals_frame_from_portfolio_weights(weights, *, constants=None, selected_threshold=1e-9)`, `signals_frame_from_asset_weights(...)`, `weights_frame_from_portfolio_weights(...)`, `weights_frame_from_asset_weights(...)`, `context_frame_from_weights(weights, *, prefix="w_", constants=None)` | results -> `PredictionsFrame` / `SignalsFrame` / `WeightsFrame` (long) / `ContextFrame` (wide) |
| `ResultsFrame` | `.to_dicts()`, `.to_columnar()`, `.to_polars()`, `.write_parquet(path, compression="zstd")` | frame surface without a hard polars dependency |
| `write_backtest_frames` | `(artifact_dir, *, predictions=None, weights=None, compression="zstd") -> dict[str, Path]` | atomic generation of `predictions.parquet` / `weights.parquet` + `manifest.json` |
| `resolve_feed_spec_mapping` | `(frame=None, *, schema, timestamp_col, entity_col, price_col, open/high/low/close/volume_col, bid/ask/mid_col, bid_size/ask_size_col, calendar, timezone, data_frequency, bar_type, timestamp_semantics, session_start_time)` | ml4t-specs `FeedSpec` aliases |
| `backtest_datafeed_inputs` | `(*, prices_frame=None, prices_path=None, signals: PredictionsFrame\|SignalsFrame\|WeightsFrame\|None, context: ContextFrame\|None, ...feed-spec kwargs..., metadata=None) -> BacktestDataFeedInputs` | `.to_datafeed_kwargs()` -> `ml4t.backtest.DataFeed(**kwargs)` |
| Shortcuts | `backtest_inputs_from_asset_forecast(forecast, ...)`, `backtest_inputs_from_asset_signal(signal, ...)`, `backtest_inputs_from_weights(weights, *, as_context=False, context_prefix="w_", ...)` | one call from a result to DataFeed kwargs |

### Persistence, observability, utilities (`_internal/`)

| Function | Purpose |
|---|---|
| `save_artifact(path, *, model_type, config, state, arrays) -> Path`; `load_artifact(path, *, expected_model_type) -> LoadedArtifact`; `require_array(artifact, name, *, ndim)`; `require_array_names(artifact, expected)`; `load_config(artifact, config_type, *, device)` | versioned artifacts used by every `save`/`load` |
| `pack_tensor_tree(value) -> (tree, arrays)` / `unpack_tensor_tree(tree, arrays, *, torch)` | torch state dicts stored as numpy; no pickle on load (inference) |
| `FitObservable.last_fit_record() -> FitRunRecord \| None`; `observed_fit`, `atomic_fit(*attribute_names)`, `state_with_fit_record`, `restore_fit_record` | provenance and atomic fitted state |
| `import_torch()`, `resolve_device(torch, requested)`, `resolve_dtype(torch, requested)`, `seed_torch(torch, seed, device)` | torch runtime |
| `validate_panel_shapes(chars_train, returns_train, chars_val=None, returns_val=None)`; `compute_managed_portfolios(chars, returns)`; `resolve_checkpoint_epochs(max_epoch, *, checkpoint_interval=5, checkpoint_epochs=None, include_final=True)`; `select_checkpoint_epoch(*, checkpoint, configured_default, available)` | latent-factor utilities |
| `summarize_predictions(y_true, y_score, *, task_type) -> dict`; `mean_cross_sectional_spearman(y_true, y_score) -> float \| None` (per-date rank IC average); `average_ranks(values)` | quick metrics mirroring ml4t-diagnostic (classification uses `_binary_auc`, `_binary_log_loss`) |

## Data contracts

All batches and results are `@dataclass(slots=True, frozen=True)`; arrays are coerced to float64 and validated in `__post_init__`. `timestamps` and `asset_ids` are plain tuples with no timezone semantics imposed; they pass through unchanged to results and frames (quickstarts use `tuple(range(T))`). Timezone and session semantics live in the ml4t-specs `FeedSpec` (`timestamp_semantics`, `session_start_time`, `calendar`, `timezone`) at the backtest boundary, not in the batches.

Shape cheat-sheet (T periods, N assets/slots, L characteristics, K factors, C context features, B sequences, F features):

| Object | Field | Shape | Validation |
|---|---|---|---|
| `PersistentPanelBatch` | `returns` / `characteristics` | `(T, N)` / `(T, N, L)` | `characteristics.shape[:2] == (T, N)`; `len(timestamps) == T`; `asset_ids` length N, unique, non-empty strings; props `n_periods`, `n_assets` |
| `CrossSectionBatch` | `characteristics` (required) / `returns` / `factor_returns` / `mask` / `context_features` | `(T, N, L)` / `(T, N)` / `(T, N)` / `(T, N)` bool / `(T, C)` | ragged dated cross-sections with a date-local slot axis; `mask` marks occupied slots (defaults to all-true via `_resolve_mask`, inference); `context_features.shape[0] == T` |
| `PortfolioSequenceBatch` | `features` (required) / `returns` / `vol_scale` / `mask` / `prev_weights` / `group_ids` / `costs` / `adjacency_mask` | `(B, T, N, F)` / `(B, T, N)` / `(B, T, N)` / `(B, T, N)` bool / `(B, N)` / `(N,)` int64 / `(N,)` or `(N, 1)` / `(N, N)` bool | `returns` are forward returns aligned to the decision at t (inference from `forward_returns` loss arg); `vol_scale` = per-asset vol-target multiplier; `costs` = per-asset cost rates; props `batch_size`, `n_periods`, `n_assets` |
| `LatentFactorState` | `asset_betas` / `factor_returns` / `checkpoint_epoch` | `(T, N, K)` / `(T, K)` or None / int | `factor_returns` is None when the extraction batch has no returns; props `n_factors` |
| `FactorForecastResult` | `factor_premia` | `(T, K)` | |
| `AssetForecastResult` / `AssetSignalResult` / `AssetWeightsResult` | `expected_returns` / `signal_values` / `weights` | `(T, N)` | |
| `StochasticDiscountFactorState` | `asset_weights` / `sdf_values` / `checkpoint_epoch` | `(T, N)` / `(T,)` or None / `(phase, epoch)` or int | |
| `PortfolioWeightsResult` | `weights` / `checkpoint_step` | `(B, T, N)` / int | |
| `IPCAModel.gamma` | Gamma | `(L_aug, K)` (inference) | |

Other contracts:

- `FitSummary(converged, train_metrics={}, val_metrics={}, best_epoch=None, history=(), notes=(), run_record=None)`.
- `FitRunRecord(schema_version=1, package_version, model_name, config, seed, resolved_device, resolved_dtype, input_dimensions, input_sha256 (64 hex), stopping_reason, skipped_updates >= 0, elapsed_seconds >= 0, error_type=None)`: redacted provenance; inputs are summarized by dimensions + SHA-256, never stored.
- Configs: frozen dataclasses subclassing `BaseModelConfig(seed=42, device="cpu", dtype="float64")`, each with a `ClassVar model_name` used as the persisted `model_type` (`pca`, `rp_pca`, `ipca`, `cae`, `stochastic_discount_factor`, `sae`, `expanding_mean`, `ar1`, `ewma`, `linear_portfolio`, `lstm_portfolio`, `deep_portfolio`, plus bases `latent_factor`, `asset_prediction`, `portfolio_model`; `MapperConfig.model_name="beta_lambda"`). `LatentFactorConfig.persistent_entities` is True only for PCA/RP-PCA.
- Artifact format: `save_artifact` writes a versioned artifact tagged `model_type`, JSON-encoded config/state, and an array payload with directory fsync; `load_artifact` checks `expected_model_type`; torch weights round-trip through numpy.
- Frame columns (confirmed in `integration/surfaces.py`): `PredictionsFrame` = `("timestamp", "asset", "prediction_value", *constants)`; `SignalsFrame` = `("timestamp", "asset", ["batch_id" if B > 1], "signal_value", "selected", *constants)`; `WeightsFrame` = same with `"weight"`; `ContextFrame` (wide) = `("timestamp", "w_<asset>"..., *constants)`. `selected = |w| > selected_threshold`. Frame `metadata["frame_type"]` is `prediction` / `signal` / `weight`.
- `write_backtest_frames` output: `<artifact_dir>/predictions.parquet`, `<artifact_dir>/weights.parquet`, `manifest.json` (`{"format_version": 1, "files": [...]}`); returns `{name: path}`.
- `PipelineFitResult` / `PortfolioPipelineFitResult` hold per-stage `FitSummary`s; `LatentFactorPrediction(state, factor_forecast, asset_forecast)`; `PortfolioPrediction(raw_weights, processed_weights)`.

## Built-in guardrails

| # | Guardrail | What it does | How to detect / work with it |
|---|---|---|---|
| 1 | Strict config validation | `_require_int` rejects bools, numpy ints, floats (`type(value) is not int`); `_require_real`, `_require_finite`, `_require_probability` ([0,1)); `device` must match `cpu\|mps\|cuda(:[0-9]+)?`; `dtype` in {float32, float64} | `ValueError` at config construction; pass Python ints, not `np.int64` |
| 2 | Checkpoint schedule validation | no duplicate `checkpoint_epochs`; each entry and `default_checkpoint` in `[1, total]` (CAE/SAE `n_epochs`, SDF `n_epochs_cond` and `beta_n_epochs`, portfolio `max_iters`); SDF tuple defaults must name a phase in {unconditional, moment, conditional} | `ValueError` at config time; `extract(checkpoint=...)` accepts only members of `available_checkpoints()` and raises otherwise |
| 3 | Batch shape/identity contracts | shape mismatches, duplicate/empty/mis-sized `asset_ids`, timestamp-length mismatches raise in `__post_init__` | build batches with `*_from_long_frame`, which sort by (timestamp, entity) and resolve columns from schema |
| 4 | Contract-type enforcement | `_require_persistent_panel` (PCA, RP-PCA) and `_require_cross_section` (IPCA, CAE) reject the wrong batch class; `validate_panel_shapes` checks train/val panels | persistent-ID models never receive ragged slots |
| 5 | Structural / predictive separation | forecasters fit only on `state.factor_returns` from the training extraction; `ExpandingMean` uses the training-sample mean; `BetaLambdaMapper` multiplies current betas by forecast premia | fit the pipeline on the training batch and `predict` on later batches; `require_estimable_factor_returns` raises on a return-less state |
| 6 | Atomic fitted state | `atomic_fit(*attribute_names)` swaps fitted attributes in only after success and rolls back on exception (inference) | a failed refit never leaves a half-updated model reporting `is_fitted()` |
| 7 | Redacted fit observability | `observed_fit` attaches a `FitRunRecord` to every fit (failures retained with `error_type`); `state_with_fit_record` makes it mandatory in persisted state; `restore_fit_record` validates on load | `model.last_fit_record()`; compare `input_sha256` to confirm identical training data |
| 8 | Safe, versioned persistence | `load_artifact(expected_model_type=...)` refuses other model families; `require_array_names` / `require_array(ndim=)` check payloads; `load(path, device=None)` moves CUDA-trained models to CPU; `write_backtest_frames` stages then `_publish_generation` swaps the directory | downstream readers never see a partial predictions/weights generation |
| 9 | Portfolio batch validation | `validate_portfolio_training_batch` (returns required), `validate_portfolio_prediction_batch` with `_validate_context` (`group_ids` required when `use_group_embedding`, `costs` when `use_cost_in_context`, inference), `validate_portfolio_identity(batch, fitted_asset_ids)` | embeddings are indexed by asset position: keep the same universe and order as at fit time; training and validation batches must share `n_assets`, `F` and `asset_ids` |
| 10 | Causality inside the deep allocator | `TemporalAttentionBlock` is causal; `CrossSectionalAttention(lag=cross_attention_lag, default 1)` sees other assets only with a lag; `MacroGraphAttention` restricts attention to `adjacency_mask` neighbours | an end-to-end Sharpe objective would otherwise exploit contemporaneous leakage |
| 11 | Loss-level guards | `burn_in` excludes warm-up periods; `sharpe_eps=1e-8`; `gamma_cost=0.5` charges costs inside the objective; `turnover_penalty` adds L1 turnover vs `prev_weights`; `max_grad_norm=1.0`; `softmin_sharpe(tau)` penalizes the worst sub-window | the policy cannot win on a single regime or ignore costs |
| 12 | Early stopping on smoothed validation metric | portfolio: `eval_every=10`, `metric_ema_alpha=0.45`, `metric_min_delta=0.001`, `early_stopping_patience=20`, `early_stopping_burn_in_iters=20`; CAE `patience=50`; SDF `patience=None` (no early stopping unless requested) | `FitSummary.converged`, `best_epoch`, `history`, `run_record.stopping_reason` |
| 13 | Explicit checkpoint extraction | results carry `checkpoint_epoch` / `checkpoint_step`; SDF carries `(phase, epoch)` | every prediction is reproducible to the training step that produced it |
| 14 | Weight implementability | `normalize_cross_sectional_weights` projects each date onto gross, net (must lie in `[-gross, gross]`) and `max_abs_weight` caps via capped-simplex projection; `apply_turnover_limit` rescales to an L1 cap; the pipeline keeps `raw_weights` next to `processed_weights` | inspect both to see how binding the constraints are |
| 15 | Reproducibility | `seed=42`, `seed_torch`, seeded `_iterate_batches`, `resolved_device`/`resolved_dtype` recorded | |
| 16 | Lazy torch import | neural models fail at fit time with a clear error when `[deep]` is missing (inference); PCA/RP-PCA/IPCA/linear portfolio/forecasters/mappers are numpy-only | |
| 17 | Missing-observation handling | `_resolve_mask` derives valid slots; IPCA `_valid_rows` drops non-finite characteristics and (when fitting) missing returns; SAE/CAE pool only masked-in rows; SDF losses take `mask` and `n_obs_per_asset` so pricing errors average per asset over observed dates (inference) | compare `batch.mask.sum()` to `np.isfinite(batch.returns).sum()`; never zero-fill ragged universes |
| 18 | Root export discipline | lazy `__getattr__` against `_public_api.py` | rely only on manifest names |

Quick detection snippet:

```python
rec = model.last_fit_record()            # FitRunRecord | None
rec.stopping_reason, rec.skipped_updates, rec.error_type, rec.input_sha256
model.available_checkpoints()            # tuple[int,...] or tuple[(phase, epoch),...]
summary.converged, summary.best_epoch, summary.val_metrics, summary.history
state.checkpoint_epoch                   # which training step produced this extraction
```

## Usage patterns

Choosing a family:

| Goal | Family | Batch | Output | Prefer when |
|---|---|---|---|---|
| Factor structure of a fixed universe | `PCAModel`, `RPPCAModel` | `PersistentPanelBatch` | `LatentFactorState` (static loadings) | stable panel (ETFs, futures, FX); RP-PCA when you want factors priced in means, not just variance |
| Characteristic-conditional betas | `IPCAModel` | `CrossSectionBatch` | `LatentFactorState`, `gamma` | large ragged equity cross-sections with characteristics; interpretable Gamma |
| Nonlinear conditional betas | `CAEModel` | `CrossSectionBatch` | `LatentFactorState` (per member via `extract_per_member`) | enough data for a neural beta net; ensemble for stability |
| Price the cross-section with one portfolio | `StochasticDiscountFactorModel` | `CrossSectionBatch` (+ `context_features`) | `StochasticDiscountFactorState` (weights, SDF series) | macro-conditional pricing; add `LinearStochasticDiscountFactorReturnMapper` or `StochasticDiscountFactorBetaNetworkHead` for return signals |
| Direct supervised signals | `SAEModel` | `CrossSectionBatch` | `AssetSignalResult` | tabular feature-rich panels; regression or classification `task_type` |
| Learn weights end to end | `LinearFeaturePortfolioModel` -> `LSTMPortfolioModel` -> `DeepPortfolioModel` | `PortfolioSequenceBatch` | `PortfolioWeightsResult` | start linear as baseline; escalate only when validation Sharpe improves net of `gamma_cost` |

Latent-factor pipeline (README quickstart):

```python
batch = CrossSectionBatch(characteristics=np.random.randn(24, 200, 12),
                          returns=np.random.randn(24, 200), timestamps=tuple(range(24)))
pipeline = LatentFactorForecastPipeline(model=IPCAModel(IPCAConfig(n_factors=3)),
    forecaster=ExpandingMeanFactorForecaster(), mapper=BetaLambdaMapper())
pipeline.fit(train_batch)
prediction = pipeline.predict(test_batch)   # .asset_forecast.expected_returns -> (T, N)
```

SDF with phase checkpoints:

```python
batch = CrossSectionBatch(characteristics=np.random.randn(36, 300, 16),
    returns=np.random.randn(36, 300), context_features=np.random.randn(36, 8),
    timestamps=tuple(range(36)))
model = StochasticDiscountFactorModel(StochasticDiscountFactorConfig(checkpoint_epochs=(256, 512, 768, 1024)))
model.fit(batch, validation_batch=val_batch)
state = model.extract(batch, checkpoint=("conditional", 1024))   # or legacy int 1280 = n_epochs_unc(256) + 1024
state.asset_weights.shape                                         # (36, 300)
```

Legacy int addressing (confirmed in `_legacy_checkpoint_lookup`): unconditional epoch `e` -> `e`; moment/conditional epoch `e` -> `n_epochs_unc + e`; sentinels `-1` = `("unconditional", 0)` val-best loss, `-2` = `("unconditional", -1)` val-best Sharpe, `-3` = `("conditional", 0)` val-best loss, `-4` = `("conditional", -1)` val-best Sharpe (stored only when a validation batch was given). With `checkpoint=None` and no `default_checkpoint`, extraction uses the last positive-epoch checkpoint.

Portfolio learner and constrained pipeline:

```python
batch = PortfolioSequenceBatch(features=np.random.randn(8, 63, 20, 10),
    returns=np.random.randn(8, 63, 20), timestamps=tuple(range(63)),
    asset_ids=tuple(f"asset_{i}" for i in range(20)))
model = LSTMPortfolioModel(LSTMPortfolioConfig(max_iters=20, checkpoint_every=5))
model.fit(batch, validation_batch=val_batch)      # omitted -> training batch reused as validation
weights = model.predict(batch, checkpoint=20)     # .weights -> (8, 63, 20)
pipe = PortfolioAllocationPipeline(DeepPortfolioModel(DeepPortfolioConfig()),
    postprocessors=(WeightConstraintPostprocessor(gross_exposure=1.0, net_exposure=0.0,
                                                  max_abs_weight=0.1, turnover_limit=0.5),))
fit = pipe.fit(train_batch, validation_batch=val_batch)   # PortfolioPipelineFitResult
pred = pipe.predict(test_batch)                           # .raw_weights / .processed_weights
```

Hand-off to backtest / diagnostic:

```python
frame = predictions_frame_from_asset_forecast(prediction.asset_forecast)
write_backtest_frames("artifacts/run_001", predictions=frame)    # atomic generation
inputs = backtest_inputs_from_weights(weights, prices_path="prices.parquet", as_context=True)
DataFeed(**inputs.to_datafeed_kwargs())                           # ml4t.backtest
```

From a long frame to batches (schema resolved from column names or ml4t-specs metadata):

```python
cs = cross_section_batch_from_long_frame(df, feature_cols=["mom12", "bm", "size"],
        return_col="fwd_ret", context_cols=["term_spread"], timestamp_col="date", entity_col="ticker")
pp = persistent_panel_batch_from_long_frame(df, return_col="ret", feature_cols=())   # PCA / RP-PCA
```

SDF predictive heads and persistence:

```python
head = StochasticDiscountFactorBetaNetworkHead(config)
head.fit(state, batch, validation_state=val_state, validation_batch=val_batch)
signals = head.predict(test_batch, checkpoint=256)             # AssetSignalResult
lin = LinearStochasticDiscountFactorReturnMapper(); lin.fit(state, batch)
forecast = lin.predict(test_state)                             # AssetForecastResult
path = model.save("artifacts/deep_portfolio"); restored = DeepPortfolioModel.load(path, device="cpu")
```

Workflow rules when using the library:

1. Build batches from long frames with the `*_from_long_frame` helpers; never hand-stack arrays from unsorted frames.
2. Keep `returns` in `CrossSectionBatch` / `PortfolioSequenceBatch` as forward returns aligned to the decision timestamp; the library does not shift them for you (inference from loss signatures). Align labels with the horizon rules in `chapters/07_defining_the_learning_task.md`.
3. Fit on the training batch, `predict`/`extract` on later batches, with purged/embargoed splits built upstream (`chapters/11_ml_pipeline.md`); the library enforces structural/predictive separation but not the split itself.
4. Pass a `validation_batch` to every neural model; without one, LSTM/Deep portfolio models validate on the training batch and SDF stores no val-best sentinels.
5. Extract at an explicit checkpoint (`checkpoint=...`) chosen on validation metrics, and record `state.checkpoint_epoch` with the backtest artifact.
6. Apply `WeightConstraintPostprocessor` before any backtest; compare `raw_weights` to `processed_weights`.
7. Convert results with the frame builders and persist with `write_backtest_frames`; feed `ml4t.backtest.DataFeed` through `backtest_datafeed_inputs` so the `FeedSpec` (calendar, timezone, `timestamp_semantics`) is explicit.
8. Use `CAEModel.extract_per_member` for ensemble dispersion, `IPCAModel.gamma` for characteristic loadings and `train_factor_returns` for the in-sample factor history; `mean_cross_sectional_spearman` gives a quick rank IC before the full diagnostic run.

## Defaults and configuration keys

| Config | Keys and defaults |
|---|---|
| `BaseModelConfig` | `seed=42`, `device="cpu"`, `dtype="float64"` |
| `LatentFactorConfig` / `PCAConfig` | `n_factors=5`; PCA has no extra keys |
| `RPPCAConfig` | `gamma=0.0`, `base_moment="second_moment"` \| `"covariance"`, `scale_by_asset_volatility=False`, `normalize_loadings="unit_length"` \| `"variance"`, `orthogonalize_factors=False` |
| `IPCAConfig` | `max_iter=10_000`, `tol=1e-6` (> 0), `factor_ridge=1e-6`, `gamma_ridge=1e-6` |
| `CAEConfig` | `task_type="regression"` \| `"classification"`, `hidden_units=(32,)`, `n_ensemble=1`, `n_epochs=50`, `checkpoint_interval=5`, `checkpoint_epochs=()`, `default_checkpoint=None`, `lr=1e-3`, `lambda_l1=1e-4`, `batch_size=10_000`; `fit(patience=50)` |
| `StochasticDiscountFactorConfig` architecture | `state_dim_sdf=4`, `state_dim_moment=32`, `hidden_dim=64`, `n_instruments=8`, `dropout=0.05` ([0,1)) |
| ... phases | `n_epochs_unc=256`, `n_epochs_moment=64`, `n_epochs_cond=1024`, `burn_in_epochs=0` (>= 0) |
| ... checkpoints | `checkpoint_interval=None`, `checkpoint_epochs=()` (validated against `n_epochs_cond`), `default_checkpoint=None` (int vs `n_epochs_cond`, or `(phase, epoch)` vs that phase's total) |
| ... optimization | `lr=1e-3`, `weight_decay=0.0` |
| ... beta head | `beta_state_dim=4`, `beta_hidden_dim=64`, `beta_n_epochs=256`, `beta_checkpoint_interval=None`, `beta_checkpoint_epochs=()`, `beta_default_checkpoint=None`, `beta_lr=1e-3` |
| ... ClassVars | `model_name="stochastic_discount_factor"`, `output_mode="weights"`, `expected_return_mapper="linear"` |
| `AssetPredictionConfig` / `SAEConfig` | `task_type="regression"`; `bottleneck_dim=96`, `aux_hidden_dim=96`, `main_hidden_units=(896, 448, 448, 256)` (exactly 4), `dropout_rates=None` (else exactly 8 values in [0,1)), `noise_std=0.035`, `alpha=1.0`, `aux_weight=1.0`, `n_epochs=50`, `batch_size=None` (full batch), `checkpoint_interval=5`, `checkpoint_epochs=()`, `default_checkpoint=None`, `lr=1e-4` |
| Forecasters | `EWMABaseForecasterConfig.half_life=12.0` (> 0); `ExpandingMean` / `AR1` have only base keys |
| `PortfolioConfig` context | `asset_embedding_dim=8`, `group_embedding_dim=4`, `use_group_embedding=False`, `use_cost_in_context=False`, `vvsn_hidden_dim=64`, `dropout=0.1` |
| ... objective | `annualization_factor=252.0`, `sharpe_eps=1e-8`, `gamma_cost=0.5` (>= 0), `turnover_penalty=0.0` (>= 0), `softmin_tau=0.2` (> 0), `softmin_lambda=0.1` (>= 0), `burn_in=0` (>= 0) |
| ... optimization | `batch_size=16`, `learning_rate=1e-4`, `weight_decay=1e-4`, `max_grad_norm=1.0`, `max_iters=200` |
| ... early stopping | `eval_every=10`, `metric_ema_alpha=0.45` ((0,1]), `metric_min_delta=0.001`, `early_stopping_patience=20`, `early_stopping_burn_in_iters=20` |
| ... checkpoints | `checkpoint_every=10`, `checkpoint_steps=()` (no duplicates, <= `max_iters`), `default_checkpoint=None` (<= `max_iters`) |
| `LSTMPortfolioConfig` | `hidden_size=64`, `n_layers=1` |
| `LinearPortfolioConfig` | `ridge_alpha=1e-4`, `fit_intercept=True`, `gross_exposure=1.0`, `net_exposure=0.0`, `max_abs_weight=None` |
| `DeepPortfolioConfig` | `d_model=64`, `n_heads=2`, `lstm_layers=1`, `temporal_mha_layers=1`, `cross_attention_heads=2`, `cross_attention_lag=1` (>= 0), `macro_gnn_heads=2`, `adapter_hidden_mult=2`; `d_model` divisible by `n_heads`, `cross_attention_heads`, `macro_gnn_heads` |
| `WeightConstraintPostprocessor` | `gross_exposure=1.0`, `net_exposure=0.0`, `max_abs_weight=None`, `turnover_limit=None` |
| Integration | `selected_threshold=1e-9`; `prefix` / `context_prefix="w_"`; parquet `compression="zstd"`; schema candidates `("timestamp","datetime","date","time")`, `("asset","symbol","ticker","instrument","security")`; `resolve_checkpoint_epochs(checkpoint_interval=5, include_final=True)`; `FitRunRecord.schema_version == 1` |

Naming asymmetry to remember: portfolio configs use `learning_rate`, `checkpoint_every`, `checkpoint_steps`, `max_iters`; neural latent/SAE/SDF configs use `lr`, `checkpoint_interval`, `checkpoint_epochs`, `n_epochs*`. Checkpoint grid = multiples of `checkpoint_interval` + explicit `checkpoint_epochs` + the final epoch.

## Where the book uses it

The 3rd-edition chapter mapping for these models was not available in the digest; the following is inferred from model provenance and the skill's chapter index (inference):

| Model family | Likely chapter / reference | Source paper (inference) |
|---|---|---|
| PCA, RP-PCA, IPCA, CAE | `chapters/14_latent_factors.md` | Lettau & Pelger (RP-PCA); Kelly, Pruitt & Su (IPCA, ThetaY identification); Gu, Kelly & Xiu (conditional autoencoder, managed portfolios as factor inputs, L1 on the beta net) |
| Deep SDF, beta head | `chapters/14_latent_factors.md` (asset pricing) | Chen, Pelger & Zhu, "Deep learning in asset pricing" (GAN with moment/instrument network, LSTM macro state, unconditional -> moment -> conditional phases; "CPZ" in `_validation_metrics`) |
| SAE | `chapters/13_dl_time_series.md` / `chapters/09_model_based_features.md` | Jane Street-style supervised autoencoder (bottleneck 96, 896/448/448/256, noise 0.035) |
| Linear / LSTM / Deep portfolio | `chapters/17_portfolio_construction.md`, `chapters/18_transaction_costs.md` | DeePM-style deep portfolio management: variable selection, FiLM, temporal/cross-sectional/graph attention, robust softmin Sharpe with costs in the objective |
| Frames and `DataFeed` bridge | `chapters/16_strategy_simulation.md` | hand-off to `ml4t-backtest` |

Case studies that naturally feed these models (inference): `case_studies/us_firm_characteristics.md` and `case_studies/us_equities_panel.md` (characteristic panels -> `CrossSectionBatch` for IPCA/CAE/SDF/SAE); `case_studies/etfs.md`, `case_studies/cme_futures.md`, `case_studies/fx_pairs.md` (fixed universes -> `PersistentPanelBatch` and `PortfolioSequenceBatch`).

Related references:
- `libraries/ml4t_engineer.md` -- builds the characteristic panels and forward-return labels that become `CrossSectionBatch` / `PortfolioSequenceBatch`.
- `libraries/ml4t_backtest.md` -- consumes `PredictionsFrame` / `SignalsFrame` / `WeightsFrame` / `ContextFrame` via `backtest_datafeed_inputs` and `DataFeed`.
- `libraries/ml4t_diagnostic.md` -- the full evaluation of `PredictionsFrame` / `SignalsFrame` ("diagnostic-ready"); `summarize_predictions` and `mean_cross_sectional_spearman` mirror its metrics.
- `libraries/ml4t_data.md` -- point-in-time data feeding the panels.
- `chapters/14_latent_factors.md` -- the methods this library implements.
- `chapters/11_ml_pipeline.md` -- purged/embargoed splits to build before fitting.
- `chapters/07_defining_the_learning_task.md` -- forward-return alignment for `returns` fields.
- `chapters/17_portfolio_construction.md`, `chapters/18_transaction_costs.md` -- weight constraints, turnover and `gamma_cost`.
- `chapters/16_strategy_simulation.md` -- backtest hand-off.
- `chapters/26_mlops_governance.md` -- `FitRunRecord` provenance and versioned artifacts.
- `guardrails.md`, `decision_rules.md`, `workflow.md`, `glossary.md` -- cross-cutting rules this library enforces in code.
- Further reading:
  - Kelly, Pruitt, Su -- Instrumented principal component analysis (IPCA).
  - Lettau, Pelger -- Factors that fit the time series and cross-section of stock returns (RP-PCA).
  - Gu, Kelly, Xiu -- Autoencoder asset pricing models (CAE).
  - Chen, Pelger, Zhu -- Deep learning in asset pricing (GAN SDF).

## Glossary

| Term | Meaning |
|---|---|
| Persistent panel | fixed asset universe across dates, `(T, N)` returns (`PersistentPanelBatch`); vs cross-section: per-date slot axis with `mask`, assets may differ by date (`CrossSectionBatch`) |
| Managed portfolio | per-date cross-sectional OLS of returns on characteristics, `(Z_t'Z_t)^-1 Z_t' r_t`; CAE's factor-network input (`compute_managed_portfolios`) |
| Gamma (IPCA) | `L x K` map from characteristics to betas, `beta_{i,t} = z_{i,t} Gamma`; ThetaY identification: orthonormal Gamma and ordered factor second moments (inference) |
| RP-PCA gamma | weight on the mean-return term added to the second-moment/covariance matrix before eigendecomposition |
| SDF / weight-native | the model outputs portfolio weights `w_t` whose portfolio return defines `M_{t+1} = 1 - w_t' R_{t+1}` (inference); pricing-error loss uses instruments (constant in the unconditional phase, learned by `MomentNetwork` in the conditional phase) |
| SDFCheckpoint | `(phase, epoch)` with phase in {"unconditional", "moment", "conditional"}; legacy int addressing is offset by `n_epochs_unc`; epochs 0 / -1 are val-best loss / Sharpe sentinels |
| Beta network head | predictive head regressing returns on SDF exposure (`StochasticDiscountFactorBetaNetworkHead`) |
| SAE | supervised autoencoder with three heads (decoder, aux predictor from bottleneck, main predictor); `alpha`, `aux_weight` weight the heads |
| vol_scale | per-asset volatility-targeting multiplier applied to weights before computing net returns |
| Robust Sharpe | pooled Sharpe over all periods plus `softmin_lambda` x `softmin_tau`-tempered minimum of window Sharpes |
| FiLM / VVSN | feature-wise linear modulation by static context (asset/group embeddings, costs) / vectorized variable selection network |
| Checkpoint grid | epochs at `checkpoint_interval` multiples + explicit `checkpoint_epochs` + the final epoch |
| FitRunRecord / FitSummary | redacted run provenance (dimensions, SHA-256, stopping reason) vs metrics outcome (converged, best_epoch, history) |
| raw vs processed weights | model output before vs after `WeightConstraintPostprocessor` in `PortfolioAllocationPipeline` |

Reader uncertainty retained from the notes: training-loop bodies were not read (optimizer type, exact `_sae_loss` and `robust_sharpe_loss` formulas, SDF GAN alternation); `CrossSectionBatch.factor_returns` has shape `(T, N_slots)` and its role (possibly precomputed managed-portfolio returns for CAE) is unverified; the 3rd-edition chapter mapping is inferred.
