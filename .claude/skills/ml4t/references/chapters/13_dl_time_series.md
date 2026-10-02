# Chapter 13: Deep Learning for Time Series

> Chapter 13 asks a narrower question than Chapters 11-12: when does the *order* of observations carry signal a cross-sectional model cannot see, and which architecture extracts it? It walks the full architecture ladder (MLP / 1D-CNN / LSTM / GRU, N-BEATS, Linear / DLinear / NLinear vs a vanilla transformer, PatchTST / iTransformer / TFT, TCN, TSMixer, selective SSM / Mamba, GAF+MTF image CNNs, zero-shot Chronos / TinyTimeMixer), then adds uncertainty (MC Dropout, deep ensembles, split conformal), a library landscape (raw PyTorch vs sktime vs Darts) and a cross-case-study synthesis read from walk-forward registries. Its position is deliberately sober: recurrence pays a sequential compute cost without proportional accuracy; transformers were embarrassed by linear baselines; and across the book's own case studies deep learning wins clearly in only a narrow slice of the evidence, while simple sequence models and strong tabular baselines (linear, GBM, TabM) remain hard to dislodge. Every notebook is built so each claim is something you check (no-parameter baseline in the table, shuffle diagnostic, step-cost timing, per-head attention) rather than something you are told. These notes were extracted from the companion repo (`13_dl_time_series/`), not the book PDF; realised IC/MSE numbers print at run time and are recorded here only as directions.

## When to use this reference

- Deciding whether a sequence model (LSTM, TCN, transformer, SSM, mixer) is warranted for a return-prediction panel, and which baseline it must beat first.
- Building windowed sequence datasets from a multi-asset panel: window/target alignment, date-anchored splits, label-horizon purge, canonical row order.
- Framing a financial DL task: direct panel regression with sequential inputs and one forward-return label vs multi-step path forecasting.
- Implementing or adapting N-BEATS, DLinear / NLinear, PatchTST, iTransformer, TCN, TSMixer, Mamba-style selective scans, GASF/MTF image encodings.
- Running the shuffle diagnostic to test whether a model actually reads temporal order.
- Benchmarking training cost across architectures (step timing with CUDA events) and making GPU runs reproducible.
- Reading attention heatmaps or N-BEATS decompositions without over-claiming.
- Evaluating zero-shot time-series foundation models (Chronos, TinyTimeMixer) on financial labels, including pretraining-contamination and horizon-mismatch checks.
- Producing calibrated prediction intervals from a neural forecaster (MC Dropout, deep ensembles, plain and normalized split conformal) and using them for position sizing.
- Choosing between raw PyTorch, sktime and Darts; reading or extending `12_case_study_insights` (cross-case-study DL vs tabular ranking from `registry.db`).

## Core ideas (the why)

- **Recurrence is a compute bottleneck, not only an optimisation one.** Step t cannot start before t-1, so a T-day window is T dependent operations regardless of hardware; GPUs accelerate parallel arithmetic, not dependent chains. Every later architecture keeps a sequence view without paying for the walk: N-BEATS via decomposition, attention via pairwise comparison, TCN via dilated causal convolution, TSMixer via axis-wise MLPs, SSMs via an associative scan (13.1, 13.6).
- **The cost column is the only durable column.** In `01_core_architectures` step time measures a property of the architecture; the error columns measure an accident of one split on a near-unpredictable target (next-day return). When the target is near-unpredictable, architecture separation shows on cost, not accuracy.
- **More capacity and more input are not more information.** A longer window gives an MLP more parameters and more room to memorise training days, not more signal; the training/validation gap is "the part of the drop that was never real".
- **No comparison means anything without a no-parameter forecast in the table.** Zero forecast on returns, persistence on prices. The entire content of the Zeng et al. (2022) critique is that transformer papers compared against each other (13.4).
- **Import a baseline's identity, not its difficulty.** Persistence is demanding on a price level and weak on a return series (repeating a near-zero-mean draw commits to a nonzero path for the whole horizon). A benchmark from electricity-load papers must be re-argued for returns.
- **A structural claim is testable by destroying the structure.** Shuffle days inside each window and re-score with unchanged weights; one forward pass, no accuracy table implies it. Score on the *predictions* (distance), not only on error.
- **What a token *is* decides what attention can do.** One day per token gives attention 60 interchangeable scalars; PatchTST puts local shape inside a token, iTransformer puts a whole feature's history inside one. Neither is "a bigger transformer" (13.5).
- **Interpretability in N-BEATS means named, plottable output parts, not market facts.** The trend polynomial and Fourier cycle are what a block was *allowed* to say under a forecast-only loss; the components are the model's, not the market's (13.2).
- **Attention weight is not importance.** Residual paths route around the attention block; per-head matrices describe one internal computation, not a feature-importance ranking.
- **Causality in a TCN is a property of the trim**, checkable by differentiating an output w.r.t. inputs; it is *not* what keeps the target out of the input. The windowing does that (13.6).
- **TSMixer's thesis:** a dense layer already relates everything to everything; the question is which axis to apply it along. Alternating axes and sharing each map along the axis it does not mix buys the parameter reduction, and that sharing ("the same temporal pattern matters in every feature") is the assumption to doubt first.
- **SSM/Mamba separates two cost claims.** Fixed-size state + bounded per-step work gives O(T) total work (an LSTM has that too; attention gives it up). *Linearity* of the update is a different property: it buys parallelism via an associative scan because each step's coefficients depend only on that step's input. "Selective" means B_t, C_t, Delta_t move with the input; A stays fixed; Delta_t is the forget control.
- **What a transformation discards matters as much as what it builds.** GASF and MTF normalise within the window, so return level and volatility are gone before the CNN sees anything, while the label is a return whose scale is exactly what was removed.
- **Check what a pretrained model is asked to forecast.** Zero-shot models forecast the context series over `PREDICTION_LENGTH`; the label is a forward return over a different horizon; mapping path mean to a ranking score is a modelling decision. Pretraining is its own leakage channel no temporal split can inspect.
- **A spread is not an interval until something calibrates it.** MC Dropout and ensembles produce numbers in return units with no coverage property; split conformal converts spread into a coverage claim under exchangeability, which this setup violates twice (calibration set = early-stopping set; overlapping labels). Only the *normalized* conformal variant uses the model's sigma-hat at all, so plain-vs-normalized is the test of whether the uncertainty method contributed anything (13.8).
- **Ordering and level are distinct questions.** Acting on a forecast uses cross-sectional ordering (IC); fitting minimises squared error (MSE). A model can win one and lose the other; report both.
- **Rank correlation on a price level is not forecasting skill.** Any level-tracking forecast, including last value, scores near one; that is why multi-asset notebooks compute cross-sectional IC.
- **Single chronological splits are architectural demonstrations, not rankings.** The authoritative comparison is walk-forward (Chapter 6 protocol) across case studies in `12_case_study_insights` (13.9); compare only configurations covering the same folds and day counts; a spread of per-fold results describes the folds and is not a confidence interval; the HAC interval on daily IC is the uncertainty estimate.
- **Formulation matters more than sophistication.** Frame finance as direct panel regression with sequential inputs (one label per entity-date, scored by IC), not multi-step path forecasting, when the action is cross-sectional ranking (13.7).
- **A wrapper costs what it does not expose** (custom loss, non-standard schedule, cross-sectional target), which is why the chapter is raw PyTorch; installability is part of the library comparison.

## Method recipes (the how)

### Section map (companion repo `13_dl_time_series/`)

Run with `uv run python 13_dl_time_series/<notebook>.py`; test mode `uv run pytest tests/test_chapter_notebooks.py -v -k "13_dl_time_series"` (reduced data via Papermill). Docker image `ml4t-gpu` for 01-11; `ml4t` for 12. No API keys; `09` and `11` download HuggingFace checkpoints on first run. `SEED=42` everywhere; `DEVICE = cuda if available else cpu`. Wall time / peak host RSS on an RTX 3090 shown.

| Section | Notebook | Does | Data | Runtime |
|---|---|---|---|---|
| 13.1 | `01_core_architectures` | MLP vs 1D-CNN vs LSTM vs GRU on pooled windows; step-cost benchmark; window sweep | daily returns of 8 ETFs (SPY, QQQ, IWM, EFA, EEM, TLT, GLD, USO) via `load_etfs()`, `START_DATE="2015-01-01"`; next-day return | 7 min / 2.0 GB |
| 13.2 | `02_nbeats_interpretable` | N-BEATS generic + interpretable; persistence baseline; trend/season read-out; ratio sweep | SPY close *prices*, standardised on training window; `HORIZON=10` path | 51 s / 1.5 GB |
| 13.4 | `03_great_debate` | Linear / DLinear / NLinear vs vanilla transformer vs zero vs closest-repeat; shuffle diagnostic; window sweep | SPY daily returns (5 funds loaded only to fix the calendar); `HORIZON=24` path | 38 s / 1.6 GB |
| 13.5 | `04_transformers` | PatchTST (reference impl) and iTransformer vs ridge; per-head attention | ETF case-study panel: 8 momentum features, 21-day forward label | 2 min 13 s / 4.9 GB |
| 13.6 | `05_tcn` | TCN vs ridge; receptive-field arithmetic | same panel | 50 s / 4.9 GB |
| 13.6 | `06_tsmixer` | TSMixer vs ridge; parameter-count printout | same panel | 55 s / 4.9 GB |
| 13.6 | `07_mamba_ssm` | Pedagogical selective SSM in pure PyTorch vs ridge(svd) | same panel, date-capped subsample | 2 min 39 s / 4.9 GB |
| 13.6 | `08_cnn_image_encoding` | GASF + MTF images -> 2D CNN vs Ridge+PCA on same pixels | single feature `ret_5d`, `LOOKBACK=20`, `IMAGE_SIZE=32` | 34 s / 3.5 GB |
| 13.6 | `09_foundation_models` | Zero-shot Chronos, TinyTimeMixer vs fitted LSTM and ridge | univariate context (first feature col), `CONTEXT_LENGTH=60` | 3 min 33 s / 3.1 GB |
| 13.8 | `10_uncertainty` | MC Dropout, deep ensembles, Gaussian coverage, split conformal, uncertainty-filtered IC | same panel | 3 min 49 s / 4.8 GB |
| 13.7 | `11_library_landscape` | Same univariate SPY forecast via raw PyTorch, sktime (NeuralForecast, PyTorch Forecasting, PatchTST, Chronos), Darts; last-value baseline | SPY closes, `HORIZON=5` | 23 s / 2.2 GB |
| 13.7, 13.9 | `12_case_study_insights` | Cross-case-study DL ranking from stored walk-forward runs; nothing trained | each `case_studies/<cs>/run_log/registry.db` | 16 s / 1.2 GB |

Section 13.3 (attention for time series: patching, positional encoding, decoder design) and TFT are text-only, no notebook. `12` expects each case study's `dl_lstm.py`, `dl_nlinear.py`, `dl_tsmixer.py`, `dl_tcn.py`, `dl_patchtst.py` to have populated its registry; a case study with no eligible runs shows as a blank row, not a failure. Shared helper module `13_dl_time_series/dl_sequences.py` (imported by 04, 05): dataset loading, multi-asset sequence building, fold construction, prediction saving.

### Shared scaffolding for panel notebooks (04-10)

1. Load the ETF case-study dataset: `mds = load_dl_dataset(...)`; `FEATURE_COLS = mds.feature_names[:8]` = `[ret_5d, ret_10d, ret_21d, ret_42d, ret_63d, ret_126d, ret_189d, ret_252d]`; `TARGET_COL = mds.label_col` (close-to-close `LABEL_HORIZON=21`-session forward return); `mds.dataset.drop_nulls(subset=FEATURE_COLS + [TARGET_COL])`.
2. Print funds-per-date before fitting (ETF breadth grows from ~6 to ~90 names; IC over 6 names is far noisier than over 90).
3. Sort rows canonically by date then symbol before building sequences (otherwise mini-batch composition and every number change, seeds notwithstanding).
4. Build `(batch, lookback, n_features)` tensors with `create_sequences_multi_asset(df, ..., timestamp_col=mds.date_col, symbol_col=mds.entity_cols[0])`; window i covers positions i..i+lookback-1, target at i+lookback+horizon-1 (never inside its own window).
5. Split by unique dates: `train_boundary_idx = int(len(unique_dates)*0.6)`, `val_boundary_idx = int(len(unique_dates)*0.8)`; `val_mask = (ts >= train_end_date) & (ts < val_label_cutoff)`, `test_mask = ts >= val_end_date`. Drop examples dated within `LABEL_HORIZON` sessions of a boundary (purge); print the count. Input windows reaching back across a boundary are legitimate (the past is available at decision time).
6. Baseline: `make_pipeline(StandardScaler(), Ridge(alpha=1.0))` on the window flattened to `LOOKBACK x len(FEATURE_COLS)` columns; alpha fixed, never tuned, identical across the section; scaler fitted on training inputs only. Use `Ridge(alpha=1.0, solver="svd")` when the flattened design is collinear (overlapping trailing returns: Gram condition number ~8e6).
7. Score two ways: left panel = cross-sectional Spearman rank IC per date (ordering quality), right panel = MSE vs predicting zero (level quality). Wrap `ml4t.diagnostic.metrics.cross_sectional_ic_series` so NaN (tied-prediction dates) and null are both filtered and the number of dates averaged over (**coverage**) is printed beside the mean.

### Baseline ladder

| Rung | Baseline | Series | Go/no-go read |
|---|---|---|---|
| 0 | Zero forecast (MSE = return variance) | returns | model MSE / zero MSE must be < 1 |
| 0 | Persistence (every horizon day = last observed price) | prices | model MAE / persistence MAE must be < 1 |
| 0 | "Closest repeat" (last return carried across horizon) | returns | included for identity; worse than zero on returns |
| 1 | `Ridge(alpha=1.0)` after `StandardScaler` on the flattened window | panel | "the comparison that decides anything" |
| 2 | Linear / NLinear / DLinear (13.4) | univariate | serious simple sequence baselines |
| 3 | attention / convolution / mixing / recurrence / SSM | — | only after rungs 0-2 are beaten |

### 01 Core architectures (13.1): MLP / 1D-CNN / LSTM / GRU

- Constants: `SEED=42`, `LOOKBACK=60`, `HORIZON=1`, `HIDDEN_SIZE=64`, `EPOCHS=50`, `BATCH_SIZE=32`, `LOOKBACKS=[30,60,120,240]`, `VALIDATION_START=0.65`, `TEST_START=0.80`, `BENCH_BATCH=256`, `SWEEP_FIRST_TARGET = max(LOOKBACKS) + HORIZON - 1` (=240).
- Data: close -> daily fractional return (comparable across price levels); wide table keeps only dates where all 8 traded; annualised vol = daily SD * sqrt(252).
- `create_sequences(data, lookback, horizon)`; `create_panel_sequences(returns_df, symbols, lookback, horizon, val_start_fraction=0.65, test_start_fraction=0.80, first_target_pos=None)` pools per-symbol windows, places each example by its *target date* at calendar fractions of trading days, stops training `HORIZON` days short of the first validation target (embargo), and `first_target_pos` aligns all window lengths to the same first target date for the sweep.
- Models (not size-matched; parameter counts printed beside cost): `MLPForecaster(lookback, hidden_size)` = flatten + 3 FC; `CNNForecaster(lookback, hidden_size)` = Conv1d kernel 3, `padding=1` (non-causal, harmless because the whole window precedes the target); `LSTMForecaster(hidden_size, num_layers=2)`; `GRUForecaster(hidden_size, num_layers=2)`. Input tensor `(examples, time, 1)`.
- Training: MSE, shuffled batches, full `EPOCHS`, no early stopping; `evaluate_mse` on validation every epoch (read, not acted on). Three loss-curve shapes: both fall (real), both flat (honest no-signal), train falls / val rises (memorising).
- Scoring: `cross_sectional_ic(y_true, y_pred, dates, symbols)` wraps `cross_sectional_ic_series` with `min_obs=5` (library default 10 would drop every date in an 8-fund panel); report IC and IC coverage.
- Step-cost benchmark `benchmark_step(model, x, y, repeats=100, warmup=20)`: one fwd+bwd+update on a fixed batch of 256; discard 20 warmup; report the **minimum** (true cost is a floor; noise only adds); CUDA events on GPU (wall clock would time queueing); run after all fits on a freshly seeded untrained network. Read ordering and scaling, never the milliseconds.
- Window sweep: rebuild MLP and LSTM at each of `LOOKBACKS`; validation dates fixed by the date-anchored split; `SWEEP_FIRST_TARGET` so all lengths train on the same days and same count. Nothing selects a window length.

### 02 N-BEATS (13.2)

- Constants: `LOOKBACK=60`, `HORIZON=10`, `HIDDEN_SIZE=256`, `N_BLOCKS=3` per stack (6 blocks in interpretable mode: first 3 trend, last 3 seasonal), `N_LAYERS=4`, `EPOCHS=50`, `BATCH_SIZE=32`, `lr=1e-3`, early-stopping `patience=7`, `clip_grad_norm_(..., 1.0)`.
- Split by sequence position: `train_target_cutoff = LOOKBACK + int(n_sequences*0.70)`, `val_target_cutoff = LOOKBACK + int(n_sequences*0.85)`; examples assigned by target.
- Data: SPY close prices (trend/seasonality are properties of a level), standardised with mean/SD of the training window only; `create_univariate_sequences(data, lookback, horizon)` targets the whole HORIZON path.
- `NBEATSBlock(lookback, horizon, hidden_size, n_layers, basis_type="generic")`: shared FC stack -> `theta_b`, `theta_f`; yhat = sum_i theta_{f,i} * g_{f,i} as `torch.einsum("bp,tp->bt", theta_f, T_fore)`. Trend basis [1, t, t^2, t^3] (degree 3); seasonal basis Fourier `[sin(2*pi*f*t), cos(2*pi*f*t)]`, `n_harmonics=5` (`n_coeffs=10`, `freqs=arange(1,6)`), bases registered as buffers `S_back`, `S_fore`.
- `NBEATS(lookback, horizon, hidden_size, n_blocks, n_layers, interpretable=True)`: doubly residual stacking, each block receives `residual = input - backcast_prev`, forecasts summed. `train_nbeats(...)`: MSE on the forecast only (no reconstruction term), early stopping on validation.
- Re-seed immediately before constructing each variant so initialisation depends on `SEED`, not on how many epochs the previous model ran before early stopping.
- Metrics: RMSE in training-window SD units; MAE / persistence MAE (persistence = 1.0).
- Decomposition read-out: sum forecasts of the first `N_BLOCKS` (trend) and the rest (seasonal); one window from the middle of the held-back stretch (date stated, not chosen). De-standardise trend with scale AND shift; seasonal with scale only (zero-centred deviation).
- Ratio sweep: 4 lookback/horizon ratios scored on validation windows only; same target dates; all candidates start at the date the longest window first reaches.

### 03 Linear baselines vs vanilla transformer + shuffle diagnostic (13.4)

- Constants: `LOOKBACK=96`, `HORIZON=24`, `D_MODEL=32`, `N_HEADS=4`, `N_LAYERS=2`, `EPOCHS=30`, `BATCH_SIZE=32`, `SYMBOLS=["SPY","QQQ","IWM","TLT","GLD"]` (calendar only). Cutoffs `LOOKBACK + int(n*0.70)` / `LOOKBACK + int(n*0.85)`; examples whose HORIZON straddles a boundary are dropped. ACF plot with +/- 1.96/sqrt(n) band.
- `Linear(lookback, horizon)`: one matrix lookback -> horizon. `DLinear(lookback, horizon, kernel_size=25)`: replicate edge values `(kernel_size-1)//2` each side, then `nn.AvgPool1d(kernel_size, stride=1, padding=0)` (zero padding would pull the average toward zero at the window's start and, worse, its last day); separate linear maps for trend and remainder, summed. `NLinear(lookback, horizon)`: subtract last value, linear map, add it back.
- `SimpleTransformer(lookback, horizon, d_model, n_heads, n_layers, dropout=0.1)`: per-day token projection + learned positional embedding, encoder layers, head flattens all LOOKBACK tokens -> horizon (where most parameters live; the design the critique targeted).
- `train_model(...)`: MSE, early stopping patience 5, `clip_grad_norm_` 1.0, returns best-state model and best val MSE.
- Headline column: **MSE / zero-forecast MSE**; > 1 means worse than saying nothing.
- Shuffle diagnostic: `idx = rng.permutation(lookback); X_shuf = X[:, idx, :]`, re-run with unchanged weights; report (a) `mse_shuf - mse` and (b) `rms(pred_shuf - pred) / rms(pred)` (0 = did not read position; 1 = moved by as much as the forecasts are large). A distance, not a correlation (correlation 1 is satisfied by any affine rescaling).
- Window sweep: 3 lengths, validation only, same target dates, same first example date.

### 04 PatchTST and iTransformer (13.5)

- Constants: `LOOKBACK=60`, `PATCH_SIZE=6`, stride `PATCH_SIZE//2=3`, `D_MODEL=32`, `N_HEADS=2`, `N_LAYERS=2`, `DROPOUT=0.1`, `EPOCHS=30`, `BATCH_SIZE=128`, `LR=0.0005`, `LABEL_HORIZON=21`.
- PatchTST invariants (drop any and it collapses to a generic token transformer): (1) channel-independent patching: each feature is its own univariate sequence through shared weights, no cross-channel mixing in the encoder (the central regulariser); (2) overlapping patches with stride = patch_len/2; (3) RevIN: per-sample per-channel mean/SD removed before the backbone and restored after the head (`revin=True`). Use the authors' reference implementation `case_studies.config.patchtst.patchtst.PatchTST` with a thin scalar-regression head, identical to `dl_patchtst.py` so 13.5 and 13.9 share the model. Attention cost O((L/P)^2) instead of O(L^2): reach for patching before shrinking the window.
- `iTransformer(lookback, n_features, d_model, n_heads, n_layers, dropout)`: instance-normalise, transpose so each of the 8 features is a token carrying its 60-day history, project, encoder with **no positional encoding** (features have no order), scalar regression adapter; attention is 8x8 instead of 60x60. (Internals described in prose only.)
- TFT (text only): covariate-rich multi-horizon forecasting.
- Training: `AdamW(weight_decay=0.01)`, `CosineAnnealingLR(T_max=epochs)`, `clip_grad_norm_` 1.0, patience 5, restore best weights, `torch.cuda.empty_cache()` after. `_chunked_forward(model, X_t, batch_size)` / `_predict_chunked` avoid OOM on full-channel PatchTST validation.
- Attention read-out: `layer.self_attn(q, k, v, need_weights=True, average_attn_weights=False)` (default averages heads); heatmap = distance from uniform 1/N in percentage points; first of two layers; averaged over sampled holdout windows; encoder layers `norm_first=False` so `self_attn` receives the unnormalised projection.

### 05 TCN (13.6)

- Constants: `EPOCHS=30`, `LOOKBACK=60`, `BATCH_SIZE=128`, `N_CHANNELS=32`, `KERNEL_SIZE=3`, `DROPOUT=0.1`, `LR=1e-3`, `LABEL_HORIZON=21`.
- `CausalConv1d(in_channels, out_channels, kernel_size, dilation)`: `nn.Conv1d(..., padding=(k-1)*d, dilation=d)` pads both ends, then drop the right-hand overhang `out[..., :-(k-1)*d]`; output at t depends on inputs <= t; length preserved.
- `TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout)` (Bai, Kolter, Koltun 2018): two weight-normalised causal convs, ReLU, channel-wise dropout, residual, 1x1 conv to align channels. Weight norm, not batch norm, so no mixing across batch members.
- `TCNRegressor(n_features, n_channels=32, kernel_size=3, dropout=0.1, dilations=(1,2,4,8))`: linear head on the **final** causal state (averaging intermediate states would mix shorter effective histories).
- Receptive field R = 1 + sum_{i=0}^{L-1} 2(k-1) d_i, d_i = 2^i, i.e. R = 1 + 2(k-1)(2^L - 1). k=3, L=4 -> R = 61 >= LOOKBACK 60 (notebook prints it). Reports IC and `mse_ratios = [test_mse/zero_mse, ridge_mse/zero_mse]`.

### 06 TSMixer (13.6)

- Time-mixing: transpose to `(batch, features, time)`, one shared `Linear(T, T)` across the `LOOKBACK` days, same for every feature. Feature-mixing: two-layer MLP across features at each timestep, same at every day. Both pre-normalised with residual: X' = X + sigma(W_t · Norm(X)^T)^T.
- Cost: T^2 + 2FH mixing weights vs (TF)^2 for one dense layer over the flattened window; the model prints its parameter count so you can check.
- Classes: `TimeMixingMLP(seq_len, n_features, dropout)`, `FeatureMixingMLP(seq_len, n_features, hidden_dim, dropout)`, `MixerBlock(seq_len, n_features, hidden_dim, dropout)` (authors' basic TSMixer), `TSMixerRegressor(seq_len, n_features, n_blocks=2, hidden_dim=32, dropout=0.1)`. Head: paper's temporal forecast projection with output length 1, then a small feature adapter to the scalar label (the adapter is the notebook's adaptation, not the paper's).
- Config: `LOOKBACK=60`, `D_MODEL=32`, `N_LAYERS=2`, `DROPOUT=0.1`, `EPOCHS=30`, `BATCH_SIZE=128`, `LABEL_HORIZON=21`.
- Adapting to a new tensor layout: check the permutation before each `Linear` (which axis is mixed); keep pre-norm and residual on every mixing layer (nothing else stabilises a 60-day dense stack).

### 07 Selective SSM / Mamba (13.6)

- ZOH recurrence: h_t = exp(Delta_t A) h_{t-1} + (Delta_t B_t) u_t; y_t = C_t^T h_t + D · u_t. A learned diagonal, time-invariant (`log_A` negated before exponentiating so the recurrence is contractive); B_t, C_t, Delta_t functions of u_t; D a direct skip.
- Block: `in_proj` doubles channels (SSM branch + gate branch); `x_proj` emits `d_state*2 + 1` values (B_t, C_t, raw Delta_t); SSM output multiplied by `silu(z)` before output projection.
- Four named departures from reference `mamba_ssm`: (1) Python `for` loop over timesteps, ~100x slower than the CUDA associative scan; (2) Delta_t is one scalar per timestep shared across channels (Mamba gives each channel its own); (3) B-bar_t ~ Delta_t B_t instead of full ZOH; (4) no depthwise causal convolution before the SSM branch.
- API: `selective_scan(log_A, D, x_branch, B, C, dt)`; `SelectiveSSMBlock(d_model, d_state=16, expand=2, dropout=0.1)`; `MambaRegressor(n_features, d_model=32, d_state=16, n_layers=2, expand=2, dropout=0.1)` with linear head on the last timestep (mirrors LSTM final hidden state).
- Config: `LOOKBACK=60`, `D_MODEL=32`, `D_STATE=16`, `N_LAYERS=2`, `DROPOUT=0.1`, `EPOCHS=10`, `BATCH_SIZE=128`, `MAX_TRAIN_SAMPLES=50_000`, `MAX_VAL_SAMPLES=15_000`, `MAX_TEST_SAMPLES=15_000`, `INFER_BATCH_SIZE=1_024`.
- Subsample with `_trim_by_complete_dates(X_arr, y_arr, ts_arr, sym_arr, max_samples)` (most recent whole dates fitting under the cap), never a raw `[-MAX_SAMPLES:]` slice. Baseline `Ridge(alpha=1.0, solver="svd")`.

### 08 GAF / MTF image encoding + CNN (13.6)

- GASF: min-max scale window to [-1, 1], phi_i = arccos(x-tilde_i), GASF_ij = cos(phi_i + phi_j); diagonal = cos(2 phi_i) = 2 x-tilde_i^2 - 1 (shape only). `gramian_angular_field(series, image_size)`.
- MTF: Q quantile bins from that window alone, transition matrix W, MTF_ij = W_{q_i, q_j}. `markov_transition_field(series, image_size, n_bins=8)`.
- Window resampled/interpolated to `IMAGE_SIZE x IMAGE_SIZE` before encoding; `create_image_dataset(X_sequences, image_size)` stacks GASF+MTF into a 2-channel image (production: (2 x F) channels). Every cell is a pair of positions, so a 2D convolution reads neighbourhoods of pairs, not of timesteps.
- CNN: `CNNBlock(in_channels, out_channels, kernel_size=3)` = Conv2d -> BatchNorm -> ReLU -> MaxPool; `ImageCNN(n_channels=2, dropout=0.5)` = 3 blocks (each halves spatial dims) -> adaptive average pooling -> dropout -> linear head.
- Baseline: flatten pixels, standardise, `PCA(n_components=min(100, n_cols, n_rows), random_state=SEED)`, `Ridge(alpha=1.0)`.
- Config: `LOOKBACK=20`, `IMAGE_SIZE=32`, `EPOCHS=10`, `BATCH_SIZE=64`, `DROPOUT=0.5`, `MAX_TRAIN_SAMPLES=40_000`, `MAX_VAL_SAMPLES=10_000`, `MAX_TEST_SAMPLES=10_000`, `INFER_BATCH_SIZE=1_024`, `FEATURE_COLS=["ret_5d"]`.

### 09 Zero-shot foundation models (13.6)

- Chronos: series quantised into tokens, T5 encoder-decoder; `ChronosPipeline.from_pretrained(f"amazon/chronos-t5-{model_size}")` (Chronos-t5-small; inference), `CHRONOS_NUM_SAMPLES=10` sample paths. TinyTimeMixer (IBM Granite, 1-5M params, TSMixer family): `TinyTimeMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-ttm-v1")` from `tsfm_public`.
- Fitted comparators on the same univariate context: `LSTMRegressor(input_size=1, hidden_size=32, n_layers=2)` trained on the label directly, and `Ridge(alpha=1.0)` on the flattened context.
- `prepare_univariate_contexts(df, feature_col, target_col, context_length, date_col="timestamp", symbol_col="symbol") -> (contexts, y, dates, syms)`; `CONTEXT_FEATURE = FEATURE_COLS[0]`.
- Scoring: zero-shot models forecast the context series `PREDICTION_LENGTH=10` steps; mean of the path = ranking score; rank IC only, MSE left blank. Helpers `_scored(candidates)`, `_verdict(zero_shot, fitted)`, `_fmt_params(n)` / `PARAM_COUNTS`; flags `CHRONOS_SUCCESS`, `TTM_SUCCESS`, `REQUIRE_FOUNDATION_MODELS=True` (True raises on runtime failure; False leaves a null row; does not govern missing installs).
- Config: `CONTEXT_LENGTH=60`, `PREDICTION_LENGTH=10`, `LABEL_HORIZON=21`, `EPOCHS=30`, `BATCH_SIZE=64`, `INFER_BATCH_SIZE=1_024`, `MAX_TRAIN/VAL/TEST_SAMPLES=None`.
- Not run: in-context learning, parameter-efficient fine-tuning, test-time ensembling; second-generation Chronos-2 and Moirai-MoE accept multivariate inputs and covariates. sktime wraps Chronos as `ChronosForecaster` for single-series `fit`/`predict` (nb 11).

### 10 Uncertainty: MC Dropout, deep ensembles, split conformal (13.8)

Pipeline sequence (each step depends on the previous):

| Step | Method | Key settings | Output |
|---|---|---|---|
| 0 | Preprocess | NaN/inf features -> 0 (standardised mean; forward-fill would present a stale value as current), count printed; canonical sort (date, symbol) | clean sequences |
| 1 | Point model | `LSTMWithDropout(input_size, hidden_size=32, n_layers=2, dropout=0.2)` (dropout between layers + `nn.Dropout` before head); `train_lstm(...) -> best val loss`, `EPOCHS=30`, `BATCH_SIZE=128`, `LOOKBACK=60` | mu |
| 2a | MC Dropout (Gal & Ghahramani 2016) | `model.train()` at inference, `MC_SAMPLES=50` passes; mean = point, std = uncertainty; no extra training cost | mu, sigma |
| 2b | Deep ensemble (Lakshminarayanan 2017) | `N_ENSEMBLE=5` x `LSTMRegressor(input_size, hidden_size=32, n_layers=2, dropout=0.1)`, `train_member(seed, member_id)`; differ only in init seed and data order; `ens_std` across members | mu, sigma |
| 3 | Calibration diagnostic | `compute_calibration_table(std, abs_error) -> (table, spearman)`: uncertainty quartiles Q1..Q4, mean abs error should rise monotonically; headline = Spearman(std, abs error) | ordering check |
| 4 | Gaussian coverage | `nominal_levels=[0.50, 0.80, 0.95]`, `z = norm.ppf(0.5 + level/2)`, interval mu +/- z*sigma; empirical fraction inside | the gap |
| 5a | Plain split conformal (Vovk 2005) | `q = conformal_quantile(resid_val, alpha)` = finite-sample quantile at (1-alpha)(n+1)/n of abs(y_val - mu_val); test interval mu_test +/- q (constant width; uses no sigma-hat) | coverage |
| 5b | Normalized conformal | `q_n = conformal_quantile(resid_val / (std_val + EPS), alpha)`, `EPS=1e-8`; interval mu_test +/- q_n*std_test; levels 0.50/0.80/0.95 | adaptive coverage |
| 6 | Compare plain vs normalized | normalized is scale-invariant in sigma; constant sigma reproduces plain widths; equal widths/coverage => the uncertainty estimate contributed nothing; read sigma coefficients of variation and minima vs EPS | verdict on sigma-hat |
| 7 | Uncertainty filtering | drop the highest-uncertainty quartile; compare IC on filtered vs full test set | action value |

- Variance decomposition: Var[y-hat] = Var_theta[E[y given x, theta]] (epistemic, ensemble disagreement, reducible) + E_theta[Var[y given x, theta]] (aleatoric, needs per-member heteroscedastic mu/sigma^2 heads). With MSE point heads, total variance = epistemic only.
- Save members with `torch.save(model.state_dict(), path)`; reload into the matching architecture.
- Production: carve a dedicated calibration split before training; `case_studies/utils/conformal.py` stratifies by symbol and fold over stored OOF predictions instead. Sizing: `conformal_weighted` in `case_studies/utils/allocation.py` uses the normalized half-width, calibrated per symbol on residuals known at t - h, h = `max(1, label horizon)` (formula not reproduced in notes).

### 11 Library landscape (13.7)

- Data: SPY closes, `START_DATE="2015-01-01"`; NumPy sequences for raw PyTorch (`create_sequences(data, lookback, horizon)`), pandas Series with business-day index for sktime/Darts. Fix the train/test boundary in sequence space first; compute normalisation mean/std only from prices up to `train_price_end` (last index any training sequence reaches: input window + horizon).
- `evaluate(y_true, y_pred)` -> MSE and Spearman IC across time via `ml4t.diagnostic.metrics.pooled_ic` (single series, so not cross-sectional). Last-value baseline: `HORIZON=5`-day mean forecast = last observed price; one line; the row to read first.
- Raw PyTorch: `LSTMForecaster(hidden_size, num_layers=2)`, ~50 lines; `LOOKBACK=60`, `HORIZON=5`, `HIDDEN_SIZE=64`, `EPOCHS=30`, `BATCH_SIZE=32`.
- sktime: `NeuralForecastLSTM(freq, input_size, max_steps, encoder_hidden_size)` (described, not run: `neuralforecast` needs `ray`, no Python 3.14 wheels, ray-project/ray#56434; `hidden_size` renamed `encoder_hidden_size`); `PytorchForecastingNBeats(max_prediction_length, max_encoder_length, max_epochs, trainer_kwargs)` (described, not run: API mismatch); sktime PatchTST (HF backend; patch size + training args; full training, fine-tuning, zero-shot) runs; sktime `ChronosForecaster` (`fit()` registers history, `predict()` does all compute) runs. Darts LSTM: own `TimeSeries` container, reindexed business-day series, cast to float32 (MPS rejects float64 under Lightning `accelerator="auto"`), `random_state` exposed.
- Table columns: `Lines`, `Fit + Predict (s)` (comparable across rows), `mse`, `ic`, `Target Scale`, `Evaluation Points` (the last three explain why mse is not comparable).

### 12 Cross-case-study synthesis (13.7, 13.9)

- Headline per case-study primary label: `prediction_metrics.ic_mean_daily` with HAC 95% CI (`ic_ci_lo`, `ic_ci_hi`, `ic_t_hac`). Filled markers = abs(t_HAC) > 2 (CI excludes zero); open = overlaps zero.
- Eligibility: full-day, exact-fold candidates only; a missing family stays missing (never filled with a shorter-span or stale candidate). `architecture(config_name)` maps registry `config_name` to display names (four LSTM variants collapse to one `LSTM` row; five architectures map to four classes).
- Sections: 1 scope/coverage; 2 forest of highest-IC DL config per CS; 3a architecture x CS heatmap (blank = no eligible run); 3b count of which architecture is highest most often, annotated with mean IC and min/max; 3c per-checkpoint (epoch) IC trajectory with IQR band across folds (peaked = early stopping has value, flat = benign landscape); 4a per-fold IC box-plus-scatter (stability only); 4b cross-fitted OOF conformal coverage at 90% nominal (below diagonal = too narrow, above = too wide; width in units of outcome std); 5 DL vs strongest tabular (linear Ch11, GBM Ch12, TabM `tabular_dl` Ch12) rescored on the exact intersection of evaluation timestamps (`family_rank1_collect(family)`, `dl_tabular_delta(cs)`), 5a scatter (above diagonal = DL wins point estimate), 5b delta table vs data frequency, universe size, tabular family; 6 multi-label horizon view (`regression_labels(cs)`, only CS with >= 2 complete registered regression labels); 7 architectural classes (`architecture_class_rows(cs)`).
- Computed takeaways at render: number of CS with a complete primary-label DL candidate; number whose HAC interval excludes zero; DL higher point estimate in `n_above` of N comparisons; largest DL-minus-tabular delta by `short_name`; "descriptive without a registered daily paired-difference estimator".

### Training-loop defaults used across the chapter

| Setting | Default | Variants |
|---|---|---|
| Optimizer / lr | Adam or AdamW, lr 1e-3 | 5e-4 (`04`); `weight_decay=0.01` + `CosineAnnealingLR(T_max=epochs)` (`04`) |
| Gradient clipping | `clip_grad_norm_` 1.0 | — |
| Early stopping | patience 5, restore best epoch | 7 (`02`); none in `01` (full `EPOCHS`) |
| Epochs | 30 | 50 (`01`, `02`); 10 (`07`, `08`) |
| Batch size | 128 (panel, `04`-`07`, `10`) | 32 (`01`-`03`, `11`); 64 (`08`, `09`) |
| Hidden / d_model | 32 | 64 (`01`, `11`); 256 (`02`) |
| Layers | 2 | 4 FC per N-BEATS block |
| Dropout | 0.1 | 0.2 (MC-dropout LSTM); 0.5 (CNN head) |
| Lookback | 60 | 96 (`03`); 20 (`08`) |
| Label horizon | 21 sessions (panel) | 1 (`01`); 10 (`02`); 24 (`03`); 5 (`11`) |
| Inference batch | `INFER_BATCH_SIZE=1_024` | chunked forward for PatchTST |
| Loss | MSE | — |

### Reproducibility protocol (GPU)

```python
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # before the first CUDA call (value: inference)
torch.use_deterministic_algorithms(True)
seed_all(SEED)   # seeds Python/NumPy/Torch; call again immediately before EVERY nn.Module constructor
```
Also: `random_state=SEED` for PCA and Darts; canonical (date, symbol) sort before sequence building. Buys bit-identical repeats only on the same machine, software build and GPU; timings are never reproducible.

## Guardrails and pitfalls

- **Partition by target date, not window start** — a training target resolved by days the model is later scored on is leakage; nothing errors, the score is quietly optimistic. Anchor each example on its target date; embargo `HORIZON` (`01`), drop examples within `LABEL_HORIZON=21` days of a boundary (`04`-`10`), drop examples whose horizon straddles a boundary (`03`); print the purge count. In `10` an unpurged boundary also contaminates conformal calibration residuals.
- **Split by date, never by row position** — samples are pooled across assets; a positional slice puts the same day on both sides and cuts cross-sections. Compute boundaries on unique dates; assign each example by its date.
- **Input windows reaching back across a boundary are NOT leakage** — the past is available at decision time; purge only on the label side.
- **Global standardisation of a trending level** — one global mean/SD puts every held-back input outside the training range; an FC network extrapolates with whatever slope its last layer has, and the error reports the setup, not the architecture. Plot the standardised series with the training max marked; normalise within window (NLinear, RevIN) or forecast returns; fit scaler statistics on the training window only. In `11`, normalising with the full series folds test-period price levels into training inputs with no trace in any split: compute statistics from prices up to `train_price_end`.
- **No no-parameter baseline** — two models can both be worse than predicting zero and still produce a "ranking". Always include zero forecast (returns) or persistence (prices); report MSE / zero-MSE; go/no-go < 1.
- **Imported baseline difficulty** — persistence is hard on prices, weak on returns; closest-repeat is worse than zero on returns. Re-argue the baseline for the series at hand; include both.
- **Falling training loss read as learning** — return series are full of day-specific structure a capacity-rich network memorises. Compute validation MSE every epoch; read whether patience fired before the epoch cap rather than assume it.
- **Selecting anything on the test stretch** — every later number becomes a report on a choice already made with the same data. Window/ratio sweeps on validation only; say so; accept that validation curves carry selection optimism (4 candidates fitted -> all 4 scores optimistic).
- **Window sweeps that change sample size or dates** — shorter windows get hundreds more examples and different validation dates; differences read as architecture are really sample size/period. Fix validation/test dates by calendar; set `first_target_pos` / `SWEEP_FIRST_TARGET` so all lengths start on the date the longest window first reaches.
- **NaN vs null in polars IC averaging** — `cross_sectional_ic_series` returns NaN for a date where all predictions tie; polars `drop_nulls` leaves NaN in place and one NaN makes the whole mean NaN (or silently shrinks the set). Filter `pl.col("ic").is_not_null() & pl.col("ic").is_not_nan()`; print the count of dates averaged over (coverage) beside the mean so a model that ties often (near-constant zero-shot forecasts, capped samples) is visible.
- **`min_obs` too high for small panels** — library default 10 discards every date in an 8-fund panel. Set `min_obs=5` for the 8-ETF panel; IC over 6-8 names is coarse (one fund changing place moves it a long way).
- **Panel breadth changing over time** — a date with 6 funds and one with 90 weigh equally in mean IC. Print funds-per-date before fitting; consider a minimum.
- **Subsample by complete dates, not row count** — a raw `[-MAX_SAMPLES:]` slice starts mid-date, leaves a partial cross-section, and makes per-date IC depend on symbol ordering. Use `_trim_by_complete_dates`; leave `09` caps at `None`.
- **Row order affects results** — the sequence builder pools assets in frame order; unfixed order changes mini-batch composition and every number, seeds notwithstanding. Sort canonically by (date, symbol).
- **Timing whole runs on shared hardware** — measures neighbours and Python launch overhead unequally across architectures. Time one step on a fixed 256 batch, 20 warmup, min of 100, CUDA events, after all training, fresh seeded model; read ordering and scaling only.
- **Expecting seed-level reproducibility on GPU** — non-associative float reductions in unordered CUDA kernels diverge over 50 epochs. Deterministic algorithms + fixed cuBLAS workspace before the first CUDA call; exact repeats only on the same machine/build.
- **Model initialisation coupled to the previous run** — a second model draws its init from wherever the global RNG stream was left, which depends on the first model's early-stopping epoch count. `seed_all()` immediately before every constructor.
- **Equalising parameter counts across RNN/MLP/CNN** — shrinks recurrent nets to sizes nobody uses and answers a different question. Use natural `HIDDEN_SIZE` forms; print counts beside cost.
- **Reading N-BEATS backcasts/residuals as "what the basis could not explain"** — the loss has no reconstruction term; backcasts get gradient only through later blocks' forecasts; the LAST block's backcast head receives no gradient and stays at random init. Check where a component connects to the loss; treat the last subtraction as noise.
- **Reading N-BEATS trend/season panels as market facts** — they are what a block was allowed to say under a forecast-only loss; call them named model outputs, not findings about SPY.
- **Shuffle test scored only on error** — average MSE can sit still while predictions change completely. Also report RMS(original - shuffled) / RMS(original); a distance, not a correlation.
- **Confidence intervals on overlapping multi-step windows** — consecutive examples share HORIZON-1 target days; naive CIs are far too narrow. Do not claim significance from window counts; use repeated splits / walk-forward with proper SEs (Chapter 6).
- **Attention heatmaps averaged across heads** — PyTorch `need_weights=True` averages heads by default; two complementary heads average to near-uniform. Use `average_attn_weights=False`; plot per head as distance from 1/N.
- **Attention weight read as feature importance** — the residual path routes around attention; a low-attention feature can dominate; similar patterns do not imply redundant heads (each has its own value projection). To claim a head/feature matters, ablate it and measure prediction change.
- **Trusting "causal" in a class name** — `nn.Conv1d` pads both ends; causality depends on trimming (k-1)*d outputs on the right. Differentiate an output position w.r.t. inputs and confirm zero gradient from the future.
- **Loosening PatchTST invariants** — dropping channel independence, overlap or RevIN reduces it to a generic token transformer. Use the reference implementation; stride = patch_len/2; `revin=True`.
- **OOM on full validation tensors** — full-channel PatchTST on the whole val set exceeds GPU memory. Chunked forward; release GPU memory between models.
- **Single chronological split as a ranking** — one split, one seed, no intervals; IC differences are within sampling noise. Treat as demonstrations; rank only with walk-forward across case studies (`12_case_study_insights`, Chapter 6).
- **Ill-conditioned flattened design** — overlapping trailing returns give Gram condition number ~8e6; sklearn's default ridge solver warns. `Ridge(alpha=1.0, solver="svd")` (test MSE agrees to 4 s.f.); do not blanket-filter warnings.
- **Under-trained network vs untuned baseline (`07`)** — the SSM gets `EPOCHS=10` on a 50k cap because the Python scan is ~100x slower; ridge is closed-form; neither had a search. Read coverage counts before bars; not a general result about selective SSMs.
- **Interpolation adds pixels that are not data** — `IMAGE_SIZE=32 > LOOKBACK=20`; compare the two constants before treating image size as a capacity knob.
- **Per-window normalisation destroys scale** — two windows with the same shape and different volatility become the same picture while the label is a scaled return. Any ranking must come from shape; feed scale separately or use a raw-window model.
- **Attributing a CNN-vs-ridge gap to convolution** — the pipelines differ in inputs (100 PCA components vs all pixels), linearity, penalisation and fitting. Whether the encoding was worth doing needs a raw-window model (nb 01-07).
- **Pretraining-corpus leakage for TSFMs** — the temporal split governs fitting, not pretraining; overlap with the evaluation window compromises a zero-shot claim and no split can detect it. Check model-card corpus dates vs evaluation window; treat zero-shot results on public indices with suspicion.
- **Horizon/target mismatch in zero-shot scoring** — forecasting the context `PREDICTION_LENGTH=10` steps and using the path mean is not a forecast of the 21-session label. Report rank IC only, leave MSE blank, state the forecast-to-score mapping; tying `PREDICTION_LENGTH` to the label horizon and aggregating the path into the target definition is the variation that would allow direct evaluation.
- **Calibration set doubles as early-stopping set (`10`)** — models were selected to make exactly those validation residuals small, biasing conformal intervals optimistic. Carve a third dedicated calibration split before training.
- **Overlapping labels break exchangeability** — a 21-session forward return shares days with its neighbours; effective sample size per quantile is far below the row count; the boundary purge does nothing about within-split dependence. Read coverage as achieved-on-this-test-set, not a finite-sample guarantee; use HAC inference for daily IC.
- **Raw LSTM uncertainty is not usable for risk management** — under the Gaussian assumption MC Dropout (near-zero-width intervals) and ensembles (wider but still far too narrow) severely under-cover at 50/80/95%. Post-hoc split-conformal calibration; compare mean predicted std against the label's own std.
- **Marginal vs per-point coverage** — the conformal quantile forces marginal coverage even when sigma-hat varies without tracking error; what degrades is per-point coverage, the property a position sizer relies on. Read Spearman(std, abs error) and the Q1->Q4 gradient alongside coverage; check sigma coefficients of variation and minima vs `EPS=1e-8`.
- **Member IC spread says nothing about the uncertainty estimate** — IC is a rank statistic; two members can score identically while predicting different magnitudes. Judge `ens_std` by the calibration diagnostics.
- **Comparing MSE across rows of the library table** — raw PyTorch and baseline score a normalised target over hundreds of rolling windows; sktime/Darts rows score raw price levels on one `HORIZON=5` forecast. Compare only `Lines` and `Fit + Predict (s)`; the only like-for-like accuracy pair is baseline vs raw PyTorch.
- **Rank IC on a price level** — anything tracking the level scores near one. Treat single-series `ic` as a sanity diagnostic; use cross-sectional IC for multi-asset work.
- **Wrapper dependency failure** — two of six sktime backends could not run. Count installability in the library decision; keep a raw-PyTorch path.
- **MPS float64 incompatibility** — Darts `TimeSeries` defaults to float64; Lightning `accelerator="auto"` on MPS rejects it. Cast to float32.
- **Unequal coverage when ranking across case studies** — a model evaluated on a shorter or easier window can win on that alone. Restrict to configs covering the same folds and day count; leave blank cells blank.
- **Fold violins read as confidence intervals** — a spread of fold summaries describes the folds. Use HAC 95% CI on the chronological daily IC series; folds for stability only.
- **Mixing weekly and daily IC on one axis** — `us_equities_panel/12_dl_weekly` sets `MAX_FOLDS=4` of 16 on a Friday-resampled panel, so its `fwd_ret_5d` point is a weekly IC over a quarter of the grid. "Complete" is measured against the case study's own fold grid; incomplete labels are excluded from the horizon figure.
- **Missing label surfaces in release bundles** — `case_studies/*/labels` is gitignored; with only `run_log/` no fold grid derives and the filter excludes nothing ("17 of 17 not derivable" means labels are absent, not checked); intraday case studies also fail because `fold_boundary_date` refuses a time of day. Read the printed "not derivable" count and know which reason applies.
- **Exposure rules that read future-computed quantities** — same defect as a leaky feature (Section 19.7). Sizing lag h = `max(1, label horizon)`; the position is chosen before the row recording its outcome.
- **Silencing warnings by blanket filter** — hides real problems. Quiet specific loggers/messages by name (library progress banners in `11`; polars `join_asof` sortedness warning in `12`, where `conformal.py` sorts by entity and step before each join).

## Decision rules and defaults

| Decision | Rule |
|---|---|
| Go/no-go for a trained forecaster | returns: MSE / zero-forecast MSE < 1; prices: MAE / persistence MAE < 1; otherwise the parameters bought nothing |
| Order of attack | zero/persistence -> `Ridge(alpha=1.0)` after `StandardScaler` on the flattened window -> Linear / NLinear / DLinear -> only then attention / convolution / mixing / recurrence / SSM |
| Report | ranking (cross-sectional Spearman IC with coverage) AND calibration (MSE ratio) separately; they can disagree (shrinking toward zero improves MSE without improving IC) |
| Framing | direct panel regression with sequential inputs and a single forward-return label when the action is cross-sectional ranking; multi-step path forecasting only when the path itself is the decision object |
| Splits | `01`: 0.65/0.80 of trading days, target-anchored, `HORIZON=1` embargo; `02`/`03`: 0.70/0.85 of sequence positions; `04`-`10`: 0.60/0.80 of unique dates with 21-day purge. All single cuts for demonstration; deployment estimates need Chapter 6 expanding walk-forward |
| Window / ratio choice | measure on validation with fixed target dates and a common first example date; never from the test stretch; state that it is selection |
| IC `min_obs` | 5 for the 8-ETF panel; default 10 for broader panels |
| Step timing | fixed batch 256, warmup 20, repeats 100, min, CUDA events, after all training, fresh seeded model |
| Shuffle diagnostic | expect linear models to move (distance ~>= 1) and the vanilla transformer to barely move; predictions moving while error does not = reading order and getting nothing (on daily equity returns, a statement about the series) |
| Transformer token choice | long window -> patch (cost O((L/P)^2); patch before shrinking the window); multivariate with meaningful variables -> invert (feature tokens, no positional encoding); covariate-rich multi-horizon -> TFT |
| TCN design | dilations double per layer; choose L so R = 1 + 2(k-1)(2^L - 1) >= LOOKBACK (k=3, L=4 -> 61 for 60); head on the final position only |
| Normalisation | window-level (NLinear last-value subtraction, RevIN per-sample mean/SD) or returns; never global statistics on a trending level |
| SSM vs attention/mixing | choose an SSM when you need a carried state and O(T) with parallel training; use the reference `mamba_ssm` kernel beyond pedagogy (loop is ~100x slower) |
| Image encoding | only when the encoding is the object of study; encode all features ((2 x F) channels) for production; verify `IMAGE_SIZE` vs `LOOKBACK`; budget per-sample preprocessing and memory |
| Sample caps under bounded compute | Mamba 50k/15k/15k, CNN 40k/10k/10k, always by complete dates; foundation models `None` |
| Zero-shot TSFM | for return prediction expect no useful ranking; before concluding consider volatility/VaR targets, fine-tuning, in-context learning, Chronos-2 / Moirai-MoE; state (1) what series was forecast, (2) over what horizon, (3) how the path became a score, (4) whether the pretraining corpus could overlap the test window |
| Uncertainty for sizing | prefer ensembles over MC Dropout (disagreement tracks error more reliably across reruns); use the normalized conformal half-width per symbol lagged by h, not raw sigma-hat (its scale is meaningless; its ordering is what must be checked); scale exposure inversely with uncertainty (Chapter 19) even if quartile filtering moves IC only a few thousandths |
| Library | raw PyTorch (~50 lines) for custom architectures, cross-sectional targets, research; sktime (~3-8 lines) for prototyping, backend swapping, standard benchmarks; Darts (~10 lines) for probabilistic forecasting in a self-contained ecosystem; Chronos/TSFMs (~3 lines, no training) for instant baselines and cold-start; always include a last-value baseline |
| Cross-CS ranking | same folds and same day counts; abs(t_HAC) > 2 for "CI excludes zero"; blank cells stay blank; per-fold boxes for stability only; conformal coverage at 90% nominal as a residual-dispersion diagnostic; DL-vs-tabular delta is descriptive without a paired-difference estimator |
| DL families by case study | US Firm Characteristics (monthly cross-section) carries no DL family; do not expect DL rows there |
| Cost vs error | when the target is near-unpredictable trust the cost column over the error column |
| Before any bar chart in `07`/`08` | read the printed count of dates each IC was averaged over |

## Code patterns and APIs

- Robust IC mean (polars; NaN != null):
  ```python
  from ml4t.diagnostic.metrics import cross_sectional_ic_series
  ic = cross_sectional_ic_series(y_true, y_pred, dates, syms, min_obs=5)   # min_obs default 10
  defined = ic.filter(pl.col("ic").is_not_null() & pl.col("ic").is_not_nan())
  mean_ic, coverage = defined["ic"].mean(), defined.height   # print coverage beside the mean
  ```
  Single-series IC across time: `ml4t.diagnostic.metrics.pooled_ic` (nb 11).
- Chapter-local loaders: `from data import load_etfs` (ETF closes, nb 01-03, 11); `13_dl_time_series/dl_sequences.py`: `load_dl_dataset(...)` -> `mds` (`.dataset`, `.label_col`, `.feature_names`, `.date_col`, `.entity_cols`); `create_sequences_multi_asset`, `create_patched_sequences_multi_asset`, `create_sequence_folds`, `create_expanding_folds`, `create_train_val_split`, `train_model`; `make_predictions_df` / `save_predictions(preds, dataset, model_id) -> Path` / `validate_predictions` / `get_output_path` / `load_predictions`; `resolve_dataset_id` with `CANONICAL_DATASET_IDS`, `DATASET_ALIASES`, `DEFAULT_LABELS` (signatures beyond names not visible in notes).
- Date-anchored panel split (nb 01): `create_panel_sequences(returns_df, symbols, lookback, horizon, val_start_fraction=0.65, test_start_fraction=0.80, first_target_pos=None)`. Date split with purge (nb 04-10): sorted `unique_dates`; boundaries at `int(len*0.6)`, `int(len*0.8)`; drop examples within `LABEL_HORIZON` sessions before a boundary. Capping: `_trim_by_complete_dates(X_arr, y_arr, ts_arr, sym_arr, max_samples)`.
- Training loop (nb 04): `torch.optim.AdamW(lr, weight_decay=0.01)`, `CosineAnnealingLR(T_max=epochs)`, `clip_grad_norm_(params, 1.0)`, patience 5, keep `best_state`, `torch.cuda.empty_cache()`; `_chunked_forward(model, X_t, batch_size)`.
- Step benchmark: `benchmark_step(model, x, y, repeats=100, warmup=20)` with `torch.cuda.Event(enable_timing=True)` pairs, min of repeats.
- Models: `MLPForecaster`, `CNNForecaster`, `LSTMForecaster(hidden_size, num_layers=2)`, `GRUForecaster`; `NBEATSBlock(..., basis_type="generic")`, `NBEATS(..., interpretable=True)`, `train_nbeats`; `Linear`, `DLinear(lookback, horizon, kernel_size=25)`, `NLinear`, `SimpleTransformer(..., dropout=0.1)`; `case_studies.config.patchtst.patchtst.PatchTST(..., revin=True)`; `iTransformer(lookback, n_features, d_model, n_heads, n_layers, dropout)`; `CausalConv1d`, `TCNBlock`, `TCNRegressor(n_features, n_channels=32, kernel_size=3, dropout=0.1, dilations=(1,2,4,8))`; `TimeMixingMLP`, `FeatureMixingMLP`, `MixerBlock`, `TSMixerRegressor(seq_len, n_features, n_blocks=2, hidden_dim=32, dropout=0.1)`; `selective_scan(log_A, D, x_branch, B, C, dt)`, `SelectiveSSMBlock(d_model, d_state=16, expand=2, dropout=0.1)`, `MambaRegressor(n_features, d_model=32, d_state=16, n_layers=2, expand=2, dropout=0.1)`; `gramian_angular_field(series, image_size)`, `markov_transition_field(series, image_size, n_bins=8)`, `create_image_dataset(X_sequences, image_size)`, `CNNBlock(in_channels, out_channels, kernel_size=3)`, `ImageCNN(n_channels=2, dropout=0.5)`; `LSTMWithDropout(input_size, hidden_size=32, n_layers=2, dropout=0.2)`, `LSTMRegressor(input_size, hidden_size=32, n_layers=2, dropout=0.1)`, `train_lstm`, `train_member(seed, member_id)`.
- DLinear moving average: replicate edges by `(kernel_size-1)//2` each side, then `nn.AvgPool1d(kernel_size=25, stride=1, padding=0)`. Causal conv: `nn.Conv1d(..., padding=(k-1)*d, dilation=d)` then `out[..., :-(k-1)*d]`. Per-head attention: `layer.self_attn(q, k, v, need_weights=True, average_attn_weights=False)`.
- Shuffle diagnostic: `idx = rng.permutation(lookback); X_shuf = X[:, idx, :]`; report `mse_shuf - mse` and `rms(pred_shuf - pred) / rms(pred)`.
- Ridge baselines: `make_pipeline(StandardScaler(), Ridge(alpha=1.0))` on `X.reshape(n, -1)`; `Ridge(alpha=1.0, solver="svd")` for collinear designs; CNN baseline `PCA(n_components=min(100, X.shape[1], X.shape[0]), random_state=SEED)` then ridge.
- Foundation models: `ChronosPipeline.from_pretrained("amazon/chronos-t5-small")`, `num_samples=CHRONOS_NUM_SAMPLES` (10); `TinyTimeMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-ttm-v1")` (`tsfm_public`); `prepare_univariate_contexts(df, feature_col, target_col, context_length, date_col="timestamp", symbol_col="symbol")`; flags `CHRONOS_SUCCESS`, `TTM_SUCCESS`, `REQUIRE_FOUNDATION_MODELS`.
- MC Dropout inference:
  ```python
  model.train()                      # keep dropout active
  mc_preds = np.zeros((MC_SAMPLES, len(X_test)))
  with torch.no_grad():
      for i in range(MC_SAMPLES):
          mc_preds[i] = model(X_test_t).cpu().numpy()
  mu, sigma = mc_preds.mean(0), mc_preds.std(0)
  ```
- Conformal: `q = conformal_quantile(resid_val, alpha)`; normalized `q_n = conformal_quantile(resid_val / (std_val + EPS), alpha)`; `lo, hi = mu_test - q_n*std_test, mu_test + q_n*std_test`; Gaussian check `z = norm.ppf(0.5 + level/2)`; `compute_calibration_table(std, abs_error) -> (pl.DataFrame, spearman)`. Production: `case_studies/utils/conformal.py`; sizing `conformal_weighted` in `case_studies/utils/allocation.py`.
- sktime / Darts: `NeuralForecastLSTM(freq, input_size, max_steps, encoder_hidden_size)`, `PytorchForecastingNBeats(max_prediction_length, max_encoder_length, max_epochs, trainer_kwargs)`, sktime PatchTST and `ChronosForecaster` (HF backend), all `fit`/`predict`; Darts LSTM with `random_state`, float32 series. nb 11 helpers `create_sequences(data, lookback, horizon)`, `evaluate(y_true, y_pred)`.
- nb 12 helpers: `architecture(config_name)`, `family_rank1_collect(family)`, `dl_tabular_delta(cs)`, `regression_labels(cs)`, `architecture_class_rows(cs)`, `plot_conformal_coverage(conformal_df)`, `label_placements`, `add_scatter_points`, `format_scatter_axes`, `scatter_legend_elements`, `add_architecture_bars`; registry keys `prediction_metrics.ic_mean_daily`, `ic_ci_lo`, `ic_ci_hi`, `ic_t_hac`; families `deep_learning`, `tabular_dl`.
- Plotting: `from utils.style import COLORS, ml4t_diverging, ml4t_palette, show_with_alt` (importing `COLORS` activates the ml4t Plotly template).
- Repo paths: `13_dl_time_series/01..12_*.py`, `13_dl_time_series/dl_sequences.py`; `case_studies/etfs/`; `case_studies/<cs>/run_log/registry.db`; `case_studies/<cs>/11_model_analysis.py`; `case_studies/<cs>/dl_{lstm,nlinear,tsmixer,tcn,patchtst}.py`; `case_studies/config/patchtst/`; `case_studies/utils/conformal.py`; `case_studies/utils/allocation.py`; `case_studies/us_equities_panel/12_dl_weekly`.

## Evidence from the book

Realised IC/MSE/ms values print at run time and are not in the notes; directions only.

- `01_core_architectures`: large horizontal (step-cost) spread, small vertical (test MSE) spread across MLP/CNN/LSTM/GRU on next-day ETF returns; ICs small, no intervals. Loss curves show all three shapes. Window sweep over {30,60,120,240}: LSTM slower than MLP at every length and cost rises with window; MLP cost near-flat; MLP training error drops with longer windows while validation does not; LSTM curves both flat. Neither turns extra history into a better forecast. On a loaded machine timing is noisier but ordering survives.
- `02_nbeats_interpretable`: every held-back SPY day sits above the training max on the standardised scale; persistence is hard to beat and models can land above the persistence line (MAE ratio > 1). Interpretable vs generic: the constraint costs some accuracy and buys plottable parts. The ratio-sweep curve moves across 4 candidates (ratio is a real setting). Last block's backcast confirmed untrained. Fourier basis at 5 harmonics finds little periodicity in daily equity prices.
- `03_great_debate`: SPY returns have a mean a small fraction of a SD from zero and ACF inside the +/-1.96/sqrt(n) band at almost all lags; zero forecast is near-optimal; closest-repeat is worse than zero. The vanilla transformer's predictions barely move under shuffling while linear models' predictions move by more than their own size; parameter counts differ by a large factor because of the flatten head; error-delta panel small for everybody (a statement about the series).
- `04_transformers` / `05_tcn` / `06_tsmixer`: single purged split vs ridge on the ETF panel, reported as demonstrations; ridge is the reference that decides whether the architecture bought anything. Per-head attention matrices differ across heads; a head-averaged matrix can look uniform when no head is. TCN receptive field 61 covers the 60-day window.
- `07_mamba_ssm`: flattened design Gram condition number ~8e6; `svd` and default solvers agree to 4 s.f.; both models on the same 50k-capped sample; coverage counts a small fraction of full test dates.
- `08_cnn_image_encoding`: GASF diagonal is a deterministic function of the normalised value (2x-tilde^2 - 1); a few training samples are inspected to confirm GASF and MTF channels produce distinct patterns.
- `09_foundation_models`: one feature, Chronos-t5-small and TTM-v1 zero-shot, `PREDICTION_LENGTH=10` vs 21-day label does not rank ETFs usefully, consistent with Rahimikia et al. (2025): Chronos R^2 = -1.37%, TimesFM R^2 = -2.80% on S&P 500 (worse than the mean). DELPHYNE (Ding et al. 2025): adding financial data to pretraining hurts general benchmarks while still failing on finance. Chronos-t5-small carries far more parameters than either fitted baseline; LSTM IC moves slightly between runs on the same GPU, ridge is deterministic. Volatility/VaR targets and fine-tuned models fare better in the cited literature.
- `10_uncertainty`: MC Dropout spread tiny relative to the label's std (two LSTM layers at dropout 0.2 give highly correlated passes); Gaussian intervals at 50/80/95% essentially zero-width; ensemble intervals wider but still far too narrow at every level; pattern consistent across reruns. Ensemble disagreement tracks abs error more reliably than dropout spread (the result that survives reruns). MC mean vs ridge IC ordering flips between runs. Dropping the highest-uncertainty quartile moves IC only a few thousandths, direction varies. Ensemble mean IC is reported against the range of member ICs so the gain from averaging is visible separately.
- `11_library_landscape`: four of six demos run (raw PyTorch, sktime PatchTST, sktime Chronos, Darts); NeuralForecast blocked by ray on Python 3.14, PyTorch Forecasting by API mismatch. Raw PyTorch ~50 lines vs ~3-8 (sktime), ~10 (Darts), ~3 (Chronos). The last-value baseline on a price level is hard to beat and its IC is near one; it wins the like-for-like comparison.
- `12_case_study_insights` / 13.9: takeaways computed at render from the registry (number of CS with complete DL candidates, how many HAC CIs exclude zero, `n_above` of N DL-vs-tabular comparisons, largest delta; inference: values depend on registry state). Four LSTM variants exist in the registry; five architectures map to four classes. README summary: deep learning wins clearly in only a narrow part of the evidence; simple sequence models often beat elaborate forecasting architectures; strong tabular baselines (linear, GBM, TabM) remain hard to dislodge.

## Related references

- `chapters/06_strategy_definition.md` — expanding walk-forward protocol and fold standard errors required before any architecture ranking.
- `chapters/07_defining_the_learning_task.md` — forward-return labels, horizon, and the target-date/purge logic the sequence splits implement.
- `chapters/11_ml_pipeline.md` — linear baselines, purged CV, Section 11.5 predictive uncertainty (Platt scaling, isotonic regression) that `10_uncertainty` extends.
- `chapters/12_gradient_boosting.md` — GBM, TabM and conformal-for-GBM baselines that `12_case_study_insights` compares DL against; the 12.6 HAC/registry reading conventions.
- `chapters/14_latent_factors.md` — latent-factor models as the next model family on the same panels.
- `chapters/15_causal_estimation.md` — causal effects, in contrast to the predictive framing here.
- `chapters/17_portfolio_construction.md` — allocators that consume forecasts and conformal widths.
- `chapters/19_risk_management.md` — position sizing from uncertainty and Section 19.7 adaptive risk controls without leakage (sizing lag h).
- `chapters/21_rl_execution_hedging.md` — sequence models reappear as policy inputs.
- `case_studies/etfs.md` — the panel (8 momentum features, 21-day label, growing fund count) used by notebooks 04-10.
- `case_studies/us_equities_panel.md` — `12_dl_weekly` with `MAX_FOLDS=4`, the weekly-vs-daily IC caveat.
- `case_studies/us_firm_characteristics.md` — monthly cross-section with no DL family.
- `libraries/ml4t_diagnostic.md` — `cross_sectional_ic_series` (`min_obs`), `pooled_ic`, HAC IC inference.
- `libraries/ml4t_models.md` — model/prediction registry conventions behind `registry.db`.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `workflow.md`, `companion_repo.md` — cross-cutting indexes that aggregate this chapter's rules.
- Further reading: Zeng et al. 2022 (Are Transformers Effective for Time Series Forecasting?); Nie et al. 2023 (PatchTST); Liu et al. 2023 (iTransformer); Lim et al. 2021 (TFT); Oreshkin et al. 2019 (N-BEATS); Challu et al. 2022 (N-HiTS); Bai, Kolter, Koltun 2018 (TCN); Chen et al. 2023 (TSMixer); Gu & Dao 2024 (Mamba); Ansari et al. 2024, 2025 (Chronos, Chronos-2); Liu et al. 2024 (Moirai-MoE); Gal & Ghahramani 2016 (MC Dropout); Lakshminarayanan et al. 2017 (deep ensembles); Vovk et al. 2005 (conformal); Jiang et al. 2020 (price images); Zhang et al. 2019 (DeepLOB); Hochreiter & Schmidhuber 1996, Hochreiter et al. 2001 (LSTM, gradient flow); Werbos 1990 (BPTT); Vaswani et al. 2017; Smyl 2020 (ES-RNN); Rangapuram et al. 2018 (deep SSMs); Lim & Zohren 2021 (survey); Aksu et al. 2024 (GIFT-Eval); Qiao et al. 2026 (It's TIME); Rahimikia et al. 2025 (TSFMs in finance); Ding et al. 2025 (DELPHYNE); Rasul et al. 2024 (Lag-Llama); Capponi et al. 2025 (nonstationarity-complexity tradeoff); Saly-Kaufmann et al. 2026 (DL finance benchmark); Hu et al. 2025 (FinMamba); Karadag et al. 2025 (ms-Mamba); Xiao et al. 2025 (TradingAgents).

## Glossary

- **Information coefficient (IC)** — per-date Spearman rank correlation between predicted and realised cross-sectional returns, averaged over dates; +1 perfect order, 0 none. **IC coverage** — count of dates on which IC was defined (not NaN from tied predictions).
- **Embargo / purge** — dropping examples whose label resolves within the forecast horizon of a split boundary (`HORIZON=1` in 01; 21 sessions in 04-10).
- **Zero forecast / persistence / closest repeat** — predict 0 everywhere (MSE = return variance) / repeat the last observed price over the horizon / repeat the last return over the horizon (the LTSF-paper baseline).
- **Last-value baseline** — forecast equal to the last observed price (nb 11).
- **Backcast / forecast (N-BEATS)** — a block's input reconstruction (subtracted before the next block) and its horizon prediction; **doubly residual stacking** = input minus backcasts, forecasts summed; **basis expansion** = learned coefficients times fixed polynomial/Fourier vectors.
- **DLinear / NLinear** — linear maps on moving-average trend and remainder (kernel 25) / on the window minus its last value (added back after).
- **Token** — the unit attention compares: a day (vanilla), a patch of days (PatchTST), a whole feature's history (iTransformer).
- **Channel independence** — each feature passes through shared weights as its own univariate series. **RevIN** — reversible instance normalisation: per-sample per-channel mean/SD removed before the model and restored after the head.
- **Causal convolution / dilation / receptive field** — output at t depends only on inputs <= t (trim (k-1)d right-hand outputs); dilation = gap between inputs a filter reads, doubling per layer; R = 1 + sum 2(k-1)d_i. **Weight normalisation** — reparameterises weights by direction and magnitude; no batch statistics.
- **Time-mixing / feature-mixing** — shared linear map across the time axis, same for every feature / MLP across features at each timestep, same at every day. **Pre-normalisation** — normalising a sublayer's input before the residual branch.
- **Selective SSM** — state space model whose B_t, C_t, Delta_t are computed from the current input. **Zero-order hold (ZOH)** — discretisation giving exp(Delta A) state decay. **Associative scan** — parallel evaluation of an affine recurrence whose per-step coefficients do not depend on prior state.
- **GASF** — Gramian angular summation field, cos(phi_i + phi_j) over arccos of a [-1,1]-scaled window. **MTF** — Markov transition field, cell (i,j) = transition probability between the quantile bins of positions i and j.
- **Zero-shot forecasting** — a pretrained model forecasts a new series with no fitting. **Negative transfer** — pretraining data that worsens downstream performance. **Chronos** — Amazon T5 encoder-decoder over quantised time-series tokens. **TTM** — TinyTimeMixer, IBM Granite 1-5M-parameter MLP-mixer forecaster.
- **Shuffle diagnostic** — permute days inside each input window; measure error delta and prediction distance. **Step time** — fastest warmed-up fwd+bwd+update on a fixed batch (CUDA events).
- **MC Dropout** — dropout kept active at inference; spread over passes approximates posterior uncertainty. **Deep ensemble** — M independently initialised models; disagreement is the uncertainty estimate. **Epistemic uncertainty** — variance of member means; reducible with data. **Aleatoric uncertainty** — within-member predictive variance; irreducible; needs a mu/sigma^2 head.
- **Split conformal** — distribution-free interval from a quantile of held-out residuals; requires exchangeability. **Normalized conformal** — residuals scaled by sigma-hat before the quantile; widths scale with the model's own spread. **Coverage** — fraction of outcomes inside the interval (marginal vs per-point). **Exchangeability** — calibration and test residuals are interchangeable draws.
- **HAC CI** — heteroskedasticity-and-autocorrelation-consistent confidence interval on mean daily IC. **Complete-coverage candidate** — registry run that covered the same folds and day count as the comparison set. **Sizing lag h** — `max(1, label horizon)` steps between the last known residual and the decision it sizes.
- **Walk-forward** — expanding chronological folds (Chapter 6) used for authoritative comparison.
