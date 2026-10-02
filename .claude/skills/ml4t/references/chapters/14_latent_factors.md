# Chapter 14: Latent Factor Models

> Chapter 14 asks the question that precedes the factor zoo: given a panel of returns and characteristics, what common structure can be recovered from the data itself, and what can that structure be trusted to say? It covers six estimators. Four (PCA, RP-PCA, IPCA, the conditional autoencoder) share a three-stage arrangement (Stage 1 extract factors and loadings; Stage 2 forecast the factor premium from history available at the decision; Stage 3 map back to per-asset signals) and differ only inside Stage 1; two leave it (the adversarial SDF learns the pricing kernel directly, the supervised autoencoder predicts end to end). The position the chapter argues is that explaining covariance is not pricing returns: a component can carry most of a panel's variance and none of its expected return, and a low-variance direction can be priced. Several notebooks end with intervals that span zero and say so; every comparison is a validation diagnostic made after selection, and the portfolio decision is deferred to Chapter 20's holdout protocol. This file equips an agent to run PCA / eigenportfolio / yield-curve / IPCA / RP-PCA / CAE / SDF / SAE pipelines with the chapter's leakage, alignment and inference discipline.

## When to use this reference

- Running PCA on a returns panel (sector ETFs, single stocks, yield changes, cross-asset futures) and deciding how many components to keep, whether to standardize, and how to read loadings.
- Building eigenportfolios, a statistical risk model, a two-speed (fast/slow EWMA) factor covariance, or stat-arb residuals from top-K eigenportfolios.
- Decomposing a yield curve into Level/Slope/Curvature or constructing a key-rate factor hedge.
- Estimating IPCA (characteristics-conditioned loadings via ALS) or RP-PCA (pricing-error-penalized PCA) and judging subspace recovery.
- Training a conditional autoencoder (GKX), an adversarial SDF (Chen-Pelger-Zhu), or a supervised autoencoder (Jane Street-style) on a stock panel.
- Turning latent factors into forward asset-level forecasts through the Stage 2 (expanding mean / AR(1) / EWMA) and Stage 3 (loading map) walk-forward adapter.
- Diagnosing loading instability across rolling windows or bootstrap resamples (sign flips, component swaps, rotations, subspace drift).
- Reporting rank IC, AUC, pricing error or factor Sharpe for a latent model with serial-dependence-robust (HAC / block-bootstrap) intervals, or comparing latent vs supervised families on a case-study registry.
- Reviewing someone else's latent-factor notebook for lookahead (full-sample universe, NaN-to-zero returns, stale characteristics, forecast-after-reveal, test-window selection).
- Running or debugging `14_latent_factors/01..09` or the case-study counterparts (`case_studies/etfs/11_latent_factors`, `case_studies/us_firm_characteristics/08_latent_factors`, `case_studies/cme_futures/10_latent_factors`).

## Core ideas (the why)

| Idea | Content | Operational consequence |
|---|---|---|
| Covariance-explaining is not priced (14.1) | A direction carrying most variance can carry zero expected return; a low-variance direction can be priced. | Never read a scree share or reconstruction quality as predictive evidence. Test pricing fit (RP-PCA) and forward IC separately. |
| Factor zoo is a modeling problem, not selection (14.1) | Hundreds of published predictors; the chapter does not pick among them, it asks what structure the panel supports. | Start from the panel, not from a list of anomalies. |
| Three-stage framework (Fig. 14.9 / 14.10) | Stage 1 compresses (T x N) returns to (T x K) factor history; Stage 2 forecasts the premium using only realizations available at the decision; Stage 3 maps through fixed/current loadings to per-asset signals. | PCA + sample-mean Stage 2 reduces to the per-asset historical mean: a sanity baseline, not a forecaster. Non-trivial cross-sectional ranking appears only when Stage 2 conditions on the factor path (AR(1), EWMA, richer ML). |
| How Stage 1 differs across estimators | PCA maximizes explained covariance; RP-PCA adds a pricing-error penalty to that objective; IPCA changes the parameterization (loadings linear in characteristics); CAE makes that map nonlinear. SDF learns the pricing kernel (no factor history to forecast); SAE predicts directly (no factor intermediate). | When notebooks compare estimators, no single difference explains the result; do not attribute an outcome to one design choice. |
| Only the subspace is identified | Rotations and sign flips leave fitted returns unchanged. | Diagnostics must be rotation-invariant (principal angles, projector distance). Ensembles average at the asset-prediction surface, never raw loadings or factors. |
| Reconstruction is not forecasting skill | CAE reconstructs the panel yet its forward rank-IC intervals span zero; SAE reconstruction error says which inputs the bottleneck keeps, not whether they predict. | Keep reconstruction diagnostics and forward-IC inference in separate sections. |
| Scree elbow is a heuristic | A component explaining little variance can still carry systematic structure; variance says nothing about pricing or predictability. | Treat K as a sensitivity sweep on training data; never as a finding. |
| Read the interval, not the point | A loading is an estimate; a portfolio decision reads the interval. | Bootstrap loadings; report IC/AUC intervals; say when they span zero. |
| Validation diagnostics, not holdout tests (14.9) | All chapter comparisons are made after selection on validation data. | Defer the portfolio decision to Chapter 20's holdout protocol. |
| Orthogonality belongs to the fitting sample | Components are orthogonal over the fit window, not inside sub-windows (rolling 63-day PC1-PC2 correlation departs from zero). | Anything relying on independence must refit on the window of use. |
| PCA loading is not CAPM beta | Beta is a regression on a prespecified market portfolio; a loading is an eigenvector of a sample covariance. Neither says whether an exposure is priced. | Run a prespecified market regression when beta is wanted. |
| Fixed income is where latent factors are least contested (14.4) | Yield-curve drivers (inflation expectations, business cycle, term premium) are themselves low-dimensional; equities have thousands of idiosyncratic factors around a smaller systematic core. | Expect 3 components to span yield changes; do not expect the same for equities. |
| HPCA is not HRP | Hierarchical PCA (Avellaneda 2019) is factor discovery; Hierarchical Risk Parity is portfolio construction (Ch. 19). | Do not conflate the two when asked for "hierarchical" methods. |

## Method recipes (the how)

### Estimator overview

| Estimator | Notebook | Stage 1 objective / parameterization | Data | Fits the 3-stage adapter? |
|---|---|---|---|---|
| Correlation-PCA | `14_latent_factors/01_pca_equity_sectors` | Max explained covariance on standardized returns | Sector ETFs 2010-01-01..2024-12-01 | Stage 1 only |
| Eigenportfolios / HPCA / two-speed covariance | `14_latent_factors/02_eigenportfolios` | PCA on top-500 liquid US equities, 2006-01-01..2018-03-27 | US Equities (NASDAQ Data Link) + sector ETFs for labels | Stage 1 + risk adapter |
| Yield-curve PCA | `14_latent_factors/03_yield_curve_decomposition` | PCA on standardized daily yield changes | FRED DGS1..DGS30, 2000-01-01..2024-12-01 | Descriptive; no split |
| IPCA (ALS) | `14_latent_factors/04_ipca` | Loadings linear in lagged characteristics (Gamma) | Self-generated synthetic panel | Full Stage 1+2+3 walk-forward |
| RP-PCA | `14_latent_factors/05_rp_pca` | Eigenvectors of Sigma + kappa * rbar rbar^T | ETF Universe 2006-01-01..2024-12-31, 59 ETFs | Full Stage 1+2+3 |
| Conditional autoencoder (GKX) | `14_latent_factors/06_conditional_autoencoder` | Nonlinear beta network x factor network on managed portfolios | US Equities top 500 by dollar volume | Full, 5-member ensemble |
| Adversarial SDF (CPZ) | `14_latent_factors/07_stochastic_discount_factor` | Learns pricing kernel M = 1 - omega^T R^e | US Equities 2000-01-01..2018-03-27, 200 stocks + FRED | No (beta head gives signals) |
| Supervised autoencoder (SAE) | `14_latent_factors/08_supervised_autoencoder` | Reconstruction + aux + main classification heads | US Equities 1995-01-01..2018-12-31, 300 stocks | No (end to end) |
| Case-study synthesis | `14_latent_factors/09_case_study_insights` | Reads registries; trains nothing | `case_studies/<cs>/run_log/registry.db` | n/a |

Run: `uv run python 14_latent_factors/<nb>.py`; test mode `uv run pytest tests/test_chapter_notebooks.py -v -k "14_latent_factors"`. No API keys. 06 runs in `ml4t-py312` (GPU), 07/08 in `ml4t-gpu`, the rest CPU-only under `ml4t`. Wall / RSS on an RTX 3090: 01 15 s / 1.0 GB; 02 19 s / 2.9 GB; 03 8 s / 1.0 GB; 04 27 s / 1.0 GB; 05 12 s / 1.1 GB; 06 2 min 2 s / 3.7 GB; 07 50 s / 3.7 GB; 08 6 min 45 s / 4.6 GB; 09 13 s / 1.1 GB.

### Correlation-PCA with bootstrap loading intervals (01)

1. Standardize each return series to unit variance (correlation-PCA) so volatile sectors (Energy-type) do not dominate PC1. Decompose Sigma = V Lambda V^T; variance share VE_k = lambda_k / sum_i lambda_i.
2. Constants: `START_DATE="2010-01-01"`, `END_DATE="2024-12-01"`, `N_BOOTSTRAP=100` (teaching budget; raise for final inference), `BLOCK_LENGTH=20`, `SEED=42`, `n_components=2`.
3. Moving-block bootstrap: `moving_block_indices(n_obs, block_length, rng)` concatenates random contiguous 20-day blocks to the original length (01 draws non-wrapping blocks; 03 is circular (inference)). Each resample gets its own scaler + PCA fit. `bootstrap_pca(returns_df, reference_components, n_bootstrap=100, n_components=2, block_length=20, rng=None)` returns loading draws; report 95% CIs.
4. Align components before computing intervals: `align_components(loadings_new, reference)` matches by maximum absolute loading similarity, then aligns sign. Without this a PC2/PC3 swap is read as loading uncertainty.
5. Rolling PCA: fresh correlation-PCA on the trailing 252 observations ending strictly before the plotted date; plot PC1 and PC2 variance shares on the same scale.
6. Score diagnostics: rescale all PC scores by one constant so PC1 daily std matches the equal-weight sector portfolio (preserves sigma_PC2 / sigma_PC1 = sqrt(lambda_2 / lambda_1); component daily vol = sqrt(lambda_k)). Scatter PC1 vs EW market with both at unit variance so slope = correlation; plot the rolling 63-day PC1-PC2 correlation.

### Eigenportfolios, sector heatmap, HPCA, residuals (02)

1. Universe: `TOP_N_STOCKS=500` by average dollar volume over the full declared sample (descriptive only; not point-in-time), `MIN_OBSERVATIONS=252`, 90% coverage filter then NaN -> 0 (attenuates covariance; print the missing-cell share as part of the result). `N_COMPONENTS=10`.
2. Eigenportfolio weights: convert v_k from standardized to raw-return space, then w_k = v_k / ||v_k||_1. L1 fixes gross exposure to 1; signed weights do not sum to one. Apply to daily stock returns to get real in-sample portfolio returns. Do not compound standardized scores (PCA centers the input).
3. Sector labels: assign each stock to the sector ETF with the highest return correlation on overlapping dates; mean loading by sector per component (Figure 14.3). Read the heatmap column before naming a component.
4. HPCA (Avellaneda 2019): Step 1 PCA within each sector -> one sector factor each; Step 2 PCA across sector factors. Buys named rows (sectors) instead of anonymous stocks.
5. Stat-arb residuals (Avellaneda & Lee 2010): regress stock returns on top-K eigenportfolios; AR(1) slope phi on residuals; OU half-life t_1/2 = -ln 2 / ln|phi| only for 0 < phi < 1; report negative phi without a half-life. No trading conclusion from a full-sample fit.
6. Risk decomposition: regress a seeded 20-stock portfolio on 5 orthogonal eigenportfolios; residual = variance not spanned. In-sample; not a universal 80-90% claim.

### Rolling stability, Procrustes alignment, two-speed covariance (02)

| Element | Setting / formula |
|---|---|
| Rolling windows | `ROLL_WINDOW=252`, `ROLL_STEP=21`, `N_ROLL_PCS=5` |
| Sign check | Signed cosine similarity between adjacent-window eigenvectors flags sign reversals |
| Rotation check | Absolute cosine (treats v and -v alike) < 0.8 flags rotations/swaps that sign correction cannot repair |
| Procrustes | B_t = loading_history[t].T, shape (N, K); solve min over R with R^T R = I_K of \|\|B_t R - B_{t-1}\|\|_F; closed form R = U V^T from SVD of B_t^T B_{t-1}; R is K x K (within the retained subspace), not N x N |
| Downstream | Rotate scores and factor covariance consistently; still monitor subspace drift |
| Two-speed PCA (Paleologo 2025 Ch. 7) | `exp_weights(length, half_life)` = exp(-ln2 * ages / half_life); `weighted_covariance(values, weights)`; `FAST_HALFLIFE=20` for residual volatility; `SLOW_HALFLIFE=120` for the normalized correlation / factor structure; `N_PROD_FACTORS=5` |
| Shrinkage | `DIAGONAL_SHRINKAGE=0.10` applied as (1-0.10) * raw + 0.10 * median(raw) to both fast residual variance and omega_eps |
| Noise edge | Compare the eigenvalue spectrum to a BBP (Baik-Ben Arous-Peche 2005) noise edge from effective (weighted) T and N; retain only eigenvalue excess above the edge as factor variance; put the remaining variance on the diagonal (exact edge formula and effective-T computation not shown in the digest) |
| Reported diagnostic | Condition number vs uniform sample covariance (numerical, not realized-risk evidence) |

### Yield-curve decomposition and key-rate hedge (03)

1. Series: `MATURITIES_YEARS=[1,2,3,5,7,10,20,30]` (DGS1..DGS30), `START_DATE="2000-01-01"`, `END_DATE="2024-12-01"`. PCA on standardized daily yield changes (correlation-PCA); covariance-PCA is the alternative only if letting the volatile short end dominate is desired.
2. Model: z(delta y_i) = sum_k beta_{i,k} f_k + eps_i, K=3. Fix sign conventions: Level all-positive (parallel shift up); Slope long-end positive / short-end negative (positive score = steepening); Curvature positive in the middle (positive score lifts the belly).
3. Bootstrap: `N_BOOTSTRAP=500`, `BLOCK_LENGTH=21`, `RANDOM_SEED=42`; circular `moving_block_indices(n_observations, block_length, rng)`; `align_to_reference(candidate, candidate_variance, reference)` uses a one-to-one Hungarian assignment plus sign alignment of the full orthonormal basis.
4. Shock series: score / fitted std, then `ROLLING_WINDOW=63` rolling mean.
5. Reconstruction: delta y_hat = F_K Lambda_K^T; report per-maturity RMSE normalized by that maturity's own variation (raw RMSE just ranks maturities by volatility).
6. Generalized duration hedge: convert standardized loadings back to basis points, multiply by the key-rate DV01 profile for exposure per unit factor score; solve three independent key-rate instruments to zero Level/Slope/Curvature exposure. This is an algebraic match, not a backtest. No train/test split applies: no target, no selection, no performance claim.

### IPCA by alternating least squares (04)

- Model: r^e_{i,t+1} = z_{i,t}^T Gamma f_{t+1} + eps_{i,t+1}. Characteristics at t generate only r_{t+1}.
- Synthetic panel (`generate_ipca_panel(seed)`): `N_PERIODS=500`, `N_ASSETS=100`, `N_CHARACTERISTICS=10`, `N_TRUE_FACTORS=3`, `N_IPCA_FACTORS=3`, `TRAIN_BOUNDARY=350` (349 training pairs), `EMBARGO=1`, 150 evaluation pairs, `SEED=42`. Characteristics follow persistent AR(1), standardized within each cross-section; returns come from the independent factor shock at t+1.
- ALS loop (`fit_ipca(returns, characteristics, n_factors, max_iter=100, tolerance=1e-6)` -> dict with gamma, factors): initialize with PCA on characteristic-managed returns, then alternate until both stabilize:
  - `estimate_factor_history(returns, characteristics, gamma)`: one small ridge-protected cross-sectional LS per date for f_{t+1} given Gamma.
  - `update_gamma(returns, characteristics, factors)`: stack characteristic-factor interactions into one pooled regression; reshape the coefficient vector to Gamma.
  - `normalize_ipca(gamma, factors)`: orthonormalize Gamma and order by factor variance.
- Recovery diagnostics: principal-angle cosines and projection-matrix distance (rotation- and sign-invariant). Never compare Gamma column by column.
- `evaluate_factor_count(n_factors)`: K sweep on the same training panel with the expanding-mean forecaster. Sensitivity, not selection.

### Stage 2 / Stage 3 walk-forward adapter (shared by 04, 05, 06)

| Step | 04 (IPCA) | 05 (RP-PCA) / 06 (CAE) |
|---|---|---|
| Driver | `walk_forward_factor_forecasts(initial_history, realized_factors)`: forecast first, then append | `walk_forward_factor_forecasts(training_history, current_factors)` (05); `walk_forward_forecasts(initial_history, current_factors)` (06): project the current observed return to get the latest realized factor, append, then forecast the next |
| Forecasters | `constant_factor_forecast` (expanding mean), `ar1_factor_forecast` (one AR(1) per factor), `ewma_factor_forecast(history, half_life=EWMA_HALF_LIFE)` with `EWMA_HALF_LIFE=12` | `expanding_mean_forecast`, `ar1_forecast`, `ewma_forecast(history, half_life=60)` with `EWMA_HALF_LIFE=60` |
| Embargo | The embargo-period return updates history at the first evaluation decision but never enters the Stage 1 fit | The current row is the gap (05) |
| Stage 3 | `map_asset_forecasts(characteristics, gamma, factor_forecasts)`: current characteristics, train-only Gamma | `map_asset_forecasts(factor_predictions, loadings)` with train-only loadings / `g_theta(z_{i,t})` |
| Long panel | `as_long_panel(values, value_name)` -> (timestamp, symbol/entity, value) polars frame | `as_long_panel(predictions, realized_returns, timestamps)` |
| Evaluation | `evaluate_asset_forecast(name, predictions, realized_returns, seed)`: forecast R^2 vs the zero-return forecast + Newey-West (HAC) IC interval, `N_BOOTSTRAP=2_000` | `evaluate_forecast(name, predictions, realized_returns, timestamps, seed)` (zero-benchmark MSE, HAC interval on per-date rank IC); `evaluate_forward_prediction(name, prediction, seed)` (06) |

Baseline: the expanding mean. A forecaster matters only if its IC interval clears zero AND the paired difference vs the baseline clears zero.

### RP-PCA (05)

1. Objective: M_kappa = Sigma + kappa * rbar rbar^T; top eigenvectors define static factor portfolios; kappa = 0 is covariance PCA. `fit_rppca(training_returns, n_factors, kappa)` computes covariance and mean on the training window only, with eigenvector signs anchored deterministically (internals `np.cov(centered) + kappa * np.outer(mean, mean)` are an inference from docstrings).
2. Constants: `N_FACTORS=5`, `KAPPAS=[0.0, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]`, `FOCUS_KAPPA=10.0` (pre-specified forecast model), `TRAIN_FRAC=0.7`, `MAX_SYMBOLS=0` (no cap), `EWMA_HALF_LIFE=60`, `N_BOOTSTRAP=2_000`.
3. Panel: compute returns before pivoting (a missing price is not a zero return); `training_complete_symbols(wide_returns, train_end, max_symbols=0)` keeps symbols with an observed return on every training date (59 ETFs); no imputation.
4. Diagnostics: `projection_share(values, loadings)` = fraction of squared magnitude retained by projection, used for training mean fit (pricing fit) and for evaluation reconstruction share relative to the zero-return reconstruction. `loading_space_distance(reference, candidate)` -> (min principal cosine, Frobenius projector distance) vs kappa = 0.
5. Report training mean fit and evaluation reconstruction on separate fixed-range axes and print each range: the two move by very different amounts and an auto-scaled axis hides it.
6. Stage 2/3 as in the shared adapter; one-day-ahead forecasts.

### Conditional autoencoder (06)

- Model (Gu-Kelly-Xiu 2019): r_{i,t} = g_theta(z_{i,t-1})^T f_t + eps_{i,t}, trained as contemporaneous reconstruction; a separate walk-forward adapter forecasts f_{t+1} and maps through g_theta(z_{i,t}).
- Splits: test = last 180 calendar days; validation = preceding 185 days; training = everything earlier. `TOP_N_STOCKS=500` by dollar volume estimated strictly before the validation boundary.
- Features: five price-derived characteristics at the close (confirmed: `Variance` = 21-day rolling return std, `AvgRet21` = 21-day rolling mean; the other three not visible in the digest), cross-sectionally ranked per timestamp. A global trading-date map joins z_{i,t-1} to r_{i,t} per symbol; gaps excluded (no stale forward fill). `RETURN_CLIP_QUANTILES=(0.001, 0.999)` learned from training returns.
- Managed portfolios (`managed_portfolios(frame)`): append a market column to the lagged characteristic matrix Z_t and solve x_t = (Z_t^T Z_t)^{-1} Z_t^T r_t jointly (GKX Eq. 16) -> 6 instruments; `portfolio_tensor(frame)`.
- Architecture: `BetaNetwork` = Linear(n_char, 32) -> BatchNorm1d(32) -> ReLU -> Linear(32, n_factors); `FactorNetwork` = Linear(n_instruments, n_factors, bias=False); `ConditionalAutoencoder` = row-wise dot product.
- Training: `N_FACTORS=5`, `N_EPOCHS=200`, `BATCH_SIZE=10_000`, `ENSEMBLE_SIZE=5` (distinct deterministic seeds), Adam `LEARNING_RATE=0.001`, `LAMBDA_L1=0.0001` applied as `normalized_l1_penalty` = lambda * mean |beta-network weight| (scale independent of architecture size), `EARLY_STOPPING_PATIENCE=30` on unpenalized validation `reconstruction_mse`; `train_single_model(member, train_tensors, valid_tensors)` restores a deep copy of the min-validation-MSE state.
- Evaluation: `evaluate_forward_prediction(name, prediction, seed)` builds a daily cross-sectional IC series first, then a HAC interval (`N_BOOTSTRAP=2_000`); MSE vs zero-return. `mean_rank_ic(prediction)` per member. Average asset predictions across members only; inspect loadings of one representative member.

### Adversarial SDF (07)

- Pricing kernel: M_{t+1} = 1 - omega_t^T R^e_{t+1}; factor F_{t+1} = omega_t^T R^e_{t+1} (factor is 1 - M, not M - 1). `construct_sdf(weights, returns)` returns the factor explicitly. A beta head estimates E_t[R^e_{i,t+1} F_{t+1}].
- Data: `N_STOCKS=200` by dollar volume strictly before the first boundary; 8 trailing characteristics ranked within date to [-0.5, 0.5] (definitions not visible); macro = daily FRED series forward-filled in source-time order then delayed one calendar day, no backfill; lagged fed funds as the daily risk-free proxy; macro mean/scale and `RETURN_CLIP_QUANTILES=(0.001, 0.999)` from training rows; date-local coverage filter; dense tensors carry a Boolean mask (zeros only after the mask). `DataSplit` NamedTuple; `pivot_value(frame, value, dates)`; `make_split(name, frame)`; the boundary return date is dropped at each split (one-period embargo).
- Networks: `SDFNetwork(n_features, n_macro, state_dim=8)` = LSTM(n_macro, 8) macro state + MLP Linear(n_features+8, 64)-ReLU-Linear(64, 32)-ReLU-Linear(32, 1)-Tanh, then per-date gross normalization raw / sum|raw| (clamp min 1e-6). `MomentNetwork(n_features, n_macro, n_instruments)` = LSTM(n_macro, 16) + Linear(n_features+16, 32)-ReLU-Linear(32, n_instruments)-Tanh; `N_INSTRUMENTS=8`. `BetaNetwork(n_features, n_macro)` = LSTM(n_macro, 8) + Linear(n_features+8, 64)-ReLU-Linear(64, 32)-ReLU-Linear(32, 1).
- Loss: `pricing_loss(weights, instruments, split)` averages squared pricing moments through time per asset and instrument with availability weights; `LOSS_SCALE=10_000.0` affects gradient magnitude only.
- Three phases: (1) unconditional warm start vs constant instruments, `UNCONDITIONAL_EPOCHS=512`; (2) `ADVERSARIAL_ROUNDS=3` of {`train_instruments()`: freeze SDF, maximize conditional pricing error for `INSTRUMENT_EPOCHS=64`; `train_conditional(...)`: freeze instruments, update SDF for up to `CONDITIONAL_EPOCHS=2_048`, `EVAL_EVERY=16`, `CONDITIONAL_PATIENCE=20`, checkpoint by validation factor Sharpe}; (3) beta head: target R^e_{i,t+1} F_{t+1} scaled by training factor vol, `beta_loss` masked MSE, `BETA_EPOCHS=512`, `BETA_PATIENCE=30`, checkpoint by validation MSE (`evaluate_beta_train_valid()`). Adam `LEARNING_RATE=0.001` throughout.
- `evaluate_train_valid(conditional)` carries LSTM state train -> validation and has no test branch. `frozen_factor_paths()` runs the frozen SDF sequentially train -> validation -> test (first test access); report per-asset test pricing errors E[M R^e_i] by decile plus the printed concentration share. Rank IC per return date with HAC interval (`N_BOOTSTRAP=2_000`). Keep factor Sharpe and pricing error (assess the kernel) apart from rank IC (assesses the ordering).

### Supervised autoencoder (08)

- Loss: L = MSE(x, x_hat) + 0.5 * BCE(y, y_aux) + BCE(y, y_main); labels are binary return directions at 5 forward horizons up to `MAX_HORIZON=40`; magnitudes never enter inputs or weights.
- Data: `N_STOCKS=300`, `START_DATE="1995-01-01"`, `END_DATE="2018-12-31"`, `UNIVERSE_END="2013-01-01"` (liquidity from training-period dollar volume); 17 trailing price/volume characteristics ranked per date to [-0.5, 0.5] (definitions not visible); `attach_forward_return(frame, horizon)` joins the exact global-calendar target date per symbol (no label if that symbol lacks a price on that date).
- CV: `expanding_splits(dates, n_splits)` with `N_SPLITS=3`, `VALIDATION_DATES=252`, 40-date purge between train and validation in every fold; test = final `TEST_DATES=252` decision dates whose 40-day labels fit, behind a second 40-date embargo; `indices_for_dates(dates)`.
- Architecture (`SupervisedAutoencoder(n_features, n_targets)`): input BatchNorm1d(n_features) -> `GaussianNoise(0.035)` (train only) -> encoder Linear(n_features, 64)-BatchNorm1d(64)-SiLU; decoder Dropout(0.05)-Linear(64, n_features); auxiliary head Linear(n_features, 64)-SiLU-Linear(64, n_targets) (whether it takes the decoded reconstruction or raw inputs is unconfirmed); main head on concat[normalized inputs, 64-d bottleneck] (skip connection): BatchNorm1d(n_features+64)-Dropout(0.10)-Linear(., 256)-SiLU-Dropout(0.25)-Linear(256, 128)-SiLU-Dropout(0.20)-Linear(128, n_targets); logits + `BCEWithLogitsLoss`.
- Training: `N_EPOCHS=25`, `BATCH_SIZE=8_192`, Adam lr=0.001, `EARLY_STOPPING_PATIENCE=5`; `fit_fold(fold, train_dates, valid_dates)` restores an immutable deep copy of the best validation-AUC checkpoint; evaluate the last expanding fold's checkpoint once on test.
- Uncertainty: `block_bootstrap_auc(probability, label, dates, block_length, n_boot, seed)` resamples 40-date blocks, `N_BOOTSTRAP=500`. Compare reconstruction-error bars only across like-shaped features (tied ranks such as price-to-high have less to reconstruct).

### Case-study synthesis (09)

- Constants: `FAMILY="latent_factors"`, `ESTIMATORS=("pca","ipca","cae","sdf","sae")`, `SUPERVISED_FAMILIES=("linear","gbm","tabular_dl","deep_learning")`, `N_BOOT=1000`, `SEED=42`. Selection and reporting column: `ic_mean_daily` (Spearman rank IC within each decision date, averaged); registry HAC 95% intervals with lags from the label horizon.
- Helpers: `label_horizon(label)` (monthly suffix -> 1 period; daily -> stated days); `selected_predictions(case_study, row, score_name)` normalizes CME `product` (supervised) and `symbol` (older latent artifact) to `entity`; `paired_daily_ic(latent, supervised)` inner-joins on timestamp-entity, ranks both scores and the target within date, Pearson of ranks = Spearman, returns the full time-sorted daily series; `mean_daily_score_correlation(left, right)` = per-month Spearman then mean.
- Produce the coverage map first; a blank cell is an unregistered row, not unsuitability. Every ordering is printed at run time, never written into prose.

## Guardrails and pitfalls

- **Full-sample liquidity universe** — Later volume decides historical membership (lookahead, survivorship). 02 labels its universe descriptive-only; 06/07/08 estimate dollar volume strictly before the first validation boundary; fix `UNIVERSE_END` before any fold.
- **Missing price treated as zero return** — Thousands of pre-inception pseudo-zeros attenuate covariance and fabricate flat returns. Compute returns before pivoting, keep nulls, require complete training history (05: 59 ETFs), never impute. 02's NaN -> 0 after a 90% coverage filter is a convention to compare against complete-case or missing-data estimators; 07 carries a Boolean mask and zero-fills only afterwards.
- **Stale characteristics carried into later returns** — A symbol gap joined by shifting rows pairs old information with a new return. Join on a global trading-date map per symbol and drop gaps (06, 07, 08); create a label only if the symbol has a price on the exact target date.
- **Contemporaneous vs lagged timing** — Characteristics at t must pair with r_{t+1}; shifting a contemporaneously generated return tests a different DGP. Generate/align explicitly; keep a one-period embargo between the Stage 1 fit and the first target (04: `EMBARGO=1`; 05: current row is the gap; 07: boundary return date dropped).
- **Forecast then reveal** — Appending the realized factor before forecasting it leaks the target. `walk_forward_*` helpers append each realization only after its prediction; no unattended multi-step paths.
- **Test-window state in any selection** — Choosing ensemble, factor count, Stage 2 forecaster, clipping thresholds or checkpoints on test turns a holdout into training. Learn clipping quantiles, macro scalers, patience/checkpoints on train/validation only; touch test once after freezing (06, 07, 08); 07's `evaluate_train_valid` has no test branch.
- **Checkpoint mutation** — Restoring "best" from a reference restores the final epoch's tensors. Deep-copy the best validation state (06, 07, 08).
- **Overlapping labels inflate confidence** — 40-day labels on adjacent dates are dependent; naive AUC intervals are too narrow. Purge every fold by the longest horizon (40), add a second 40-date test embargo, bootstrap with 40-date blocks (08).
- **Stock-day rows treated as independent in IC inference** — Serial dependence in daily IC understates variance. Compute a per-date cross-sectional IC series, then Newey-West/HAC via `compute_ic_uncertainty` (04-09); never pool stock-days.
- **Ranking read off overlapping individual intervals** — Each interval tests one forecaster vs zero; the difference depends on the covariance of the two series. Compute the paired daily difference and its interval (09 does; 05 does not and says so).
- **Post-selection validation intervals read as holdout tests** — The HAC interval is conditional on validation-based selection; it is a paired stability diagnostic. Defer portfolio decisions to Chapter 20.
- **Sign/order ambiguity misread as instability** — v and -v are the same direction; close eigenvalues swap order; bootstrap CIs then straddle zero spuriously. Align by max-abs similarity + sign (01), Hungarian assignment (03), deterministic sign anchoring (05), signed vs absolute cosine with the 0.8 threshold and K x K Procrustes (02).
- **Averaging loadings across ensemble members** — Each member has its own rotation and scale; the mean is meaningless. Average asset predictions; inspect one representative member's loadings (06).
- **Scree elbow / variance share as pricing or predictability evidence** — Variance explained is silent on expected return. Treat K as a training-data sensitivity sweep (04), test pricing fit separately (05), report IC intervals.
- **Covariance-PCA on heterogeneous volatilities** — The loudest sector/maturity wins PC1 (01, 03). Standardize (correlation-PCA) unless the scale difference is economically intended.
- **In-sample orthogonality assumed inside windows** — Rolling 63-day PC1-PC2 correlation departs from zero; hedges and risk decompositions built on independence break. Refit on the window of use (01).
- **Rising PC1 share read as "stress co-movement"** — A rotation (opposite-signed loadings) and a synchronized sell-off give the same share. Inspect that window's loadings; the rolling loop keeps only shares (01).
- **PCA loading interpreted as CAPM beta or priced exposure** — Different estimand. Run a prespecified market regression; test pricing with Stage 2 + point-in-time evaluation (02).
- **Naming components by number** — Which contrast lands on PC2 is universe- and period-specific (commodity dominated PC2 on the 2006-2018 top-500). Read the sector heatmap column before labeling (02).
- **Stat-arb residual half-lives as trade signals** — Mean-reversion parameters are unstable out of sample and costs dominate. No trading conclusion from a full-sample fit; negative phi gets no OU half-life (02).
- **Better condition number read as better risk** — Numerical diagnostic, not realized portfolio risk. Compare risk models point-in-time walk-forward (Chapter 17).
- **Validation loss below training loss read as generalization** — Different samples/dispersion, and training loss includes the L1 penalty (06) or a scale artifact (07). Compare only movement along the validation curve.
- **SDF sign convention** — F = 1 - M; reversing it flips the beta-signal interpretation. `construct_sdf` returns factor = omega^T R^e explicitly (07).
- **FRED current snapshot is not vintage data** — Revisions leak. Use daily market series only, one-calendar-day delay, no backfill, exclude revised lower-frequency releases; still flag it as a limitation (07).
- **Curated present-day ETF universe** — Survivorship (05). Method exposition only; no strategy claim.
- **Reconstruction-error bars compared across differently shaped features** — Tied ranks have less to reconstruct. Compare bars of like-shaped features only (08).
- **Registry coverage gaps read as results** — A blank cell means no registered row, not unsuitability. Build the coverage map first (09).

## Decision rules and defaults

| Decision | Rule / default |
|---|---|
| Standardize? | Correlation-PCA by default for cross-sectional equity and yield panels; covariance-PCA only when scale differences are economically meaningful. |
| Bootstrap budgets | 100 resamples / 20-day blocks for teaching (01; raise for final inference); 500 / 21-day circular (03); 2,000 HAC draws for IC uncertainty (04-07); 500 x 40-date blocks for AUC (08); 1,000 (09). |
| Rolling PCA | 252-day window, step 21, strictly before the plotted date; 63-day rolling correlation/mean for score diagnostics. |
| Stability threshold | Absolute cosine < 0.8 between adjacent-window eigenvectors => rotation/swap: apply K x K Procrustes and monitor subspace drift; signed cosine alone fixes only sign. |
| Two-speed covariance | Fast half-life 20, slow half-life 120, 5 factors, 10% diagonal shrinkage toward the median, keep only eigenvalue excess above the BBP noise edge. |
| Eigenportfolio weights | L1-normalize in raw-return space; never compound standardized scores. |
| OU half-life | Only for 0 < phi < 1. |
| IPCA | K=3 on the synthetic panel, `max_iter=100`, `tolerance=1e-6`, ridge in the per-date solve; judge recovery by principal-angle cosines / projector distance. |
| Stage 2 go/no-go | A forecaster matters only if its IC interval clears zero AND the paired difference vs the expanding-mean baseline clears zero. EWMA half-life 12 (synthetic daily-ish periods) or 60 (daily ETF/equity). |
| RP-PCA | Pre-specify kappa (= 10) before evaluation; sweep [0, 1, 5, 10, 50, 100, 500] on training fit only; report training mean fit and evaluation reconstruction on separate fixed-range axes. |
| Eligibility / split (05) | Symbol must have an observed return on every training date; 70/30 train/eval split by date. |
| CAE | 5 factors, 5 ensemble members, <= 200 epochs, batch 10,000, lr 1e-3, L1 1e-4 normalized, patience 30, clip returns at training 0.1% / 99.9% quantiles; validation 185 days, test 180 days. |
| SDF | 200 stocks, 8 characteristics, 8 instruments, 3 adversarial rounds, 512 / 64 / 2,048 / 512 epochs, eval every 16, patience 20 / 30, lr 1e-3; select SDF by validation Sharpe, beta by validation MSE; test once after both are frozen. |
| SAE | 300 stocks, 17 features, 5 horizons, purge and embargo = longest horizon (40), 3 expanding folds x 252 validation dates, 252 test dates, 25 epochs, batch 8,192, patience 5, select by validation AUC, evaluate the last fold's checkpoint once. |
| Reporting statistic | Per-date IC then HAC; never pool stock-days. Use `ic_mean_daily` for both selection and reporting so the two statistics match. |

Sequencing: fix calendar boundaries -> fix universe from pre-boundary data -> learn scalers/clips on training -> Stage 1 fit -> validation-selected checkpoint -> walk-forward Stage 2/3 -> single test pass -> HAC / block-bootstrap intervals -> paired differences -> portfolio decision in Chapter 20.

Checklist before interpreting any latent-factor output:
1. Was the universe fixed from data strictly before the first validation/evaluation boundary?
2. Were scalers, clipping quantiles and eligibility rules learned on training rows only?
3. Is every characteristic at t paired with the return at t+1 via a global trading-date map joined on symbol, with gaps dropped?
4. Is there a one-period (or longest-horizon) embargo at every split boundary?
5. Were components aligned (sign + order) before computing intervals, and is the diagnostic rotation-invariant?
6. Is the ensemble averaged at the prediction surface, not the loadings?
7. Does the reported interval test the quantity claimed (per-forecaster vs zero, or a paired difference)?
8. Is the number a validation diagnostic after selection, or a once-only test pass? Say which.

## Code patterns and APIs

- IC inference (04-07): `from ml4t.diagnostic.metrics import cross_sectional_ic_series` and `from ml4t.diagnostic.metrics.uncertainty import compute_ic_uncertainty` (09 imports `compute_ic_uncertainty` from `ml4t.diagnostic.metrics`). Pattern: long polars panel (timestamp, entity, prediction, target) -> per-date Spearman IC series -> HAC interval with lag set by the label horizon.
- Canonical long schema: `as_long_panel(...)` converts period-by-asset matrices to (timestamp, symbol/entity, value) polars frames before any metric.
- Polars feature code (06): `pl.col("adj_close").pct_change().over("symbol")`; `pl.col("return").rolling_std(21).over("symbol").alias("Variance")`; `rolling_mean(21).alias("AvgRet21")`; `(pl.col("close") * pl.col("volume")).alias("dollar_volume")`; `.drop_nulls(subset=[...])`; training-quantile clipping via `series.quantile(q)` then `pl.col("return").clip(lo, hi)`.
- Moving-block bootstrap (01/03; signatures verbatim, body an inference):
  ```python
  def moving_block_indices(n_obs, block_length, rng):
      starts = rng.integers(0, n_obs, size=ceil(n_obs / block_length))
      idx = np.concatenate([(s + np.arange(block_length)) % n_obs for s in starts])
      return idx[:n_obs]   # circular variant (03); 01 draws non-wrapping blocks
  ```
- Procrustes alignment (02), B of shape (N, K): `U, _, Vt = np.linalg.svd(B_t.T @ B_prev); R = U @ Vt; B_aligned = B_t @ R`.
- Exponential weights (02): `np.exp(-np.log(2) * ages / half_life)`; `weighted_covariance(values, weights)` (body inferred from docstring).
- IPCA ALS: `fit_ipca(returns, characteristics, n_factors, max_iter=100, tolerance=1e-6)` -> dict with gamma, factors; inner `estimate_factor_history`, `update_gamma`, `normalize_ipca`; `generate_ipca_panel(seed)`; `evaluate_factor_count(n_factors)`.
- RP-PCA: `fit_rppca(training_returns, n_factors, kappa)`; `projection_share(values, loadings)`; `loading_space_distance(reference, candidate)`; `training_complete_symbols(wide_returns, train_end, max_symbols=0)`.
- Walk-forward adapter interface (04/05/06): `{expanding_mean|constant, ar1, ewma}_forecast(history) -> next factor vector`; `walk_forward_*(initial_history, current_factors) -> dict[name, forecasts]`; `map_asset_forecasts(...)`; `evaluate_asset_forecast` / `evaluate_forecast` / `evaluate_forward_prediction`.
- Torch, CAE (06): `BetaNetwork`, `FactorNetwork`, `ConditionalAutoencoder` (row-wise dot product), `managed_portfolios(frame)`, `portfolio_tensor(frame)`, `normalized_l1_penalty(model)`, `train_epoch`, `reconstruction_mse`, `train_single_model(member, train_tensors, valid_tensors)`, `mean_rank_ic(prediction)`.
- Torch, SDF (07): `SDFNetwork`, `MomentNetwork`, `BetaNetwork` with `nn.LSTM(n_macro, state_dim, batch_first=True)`; `construct_sdf`, `pricing_loss`, `beta_loss`, `train_instruments`, `train_conditional`, `evaluate_train_valid`, `evaluate_beta_train_valid`, `frozen_factor_paths`; `DataSplit`, `pivot_value`, `make_split`.
- Torch, SAE (08): `GaussianNoise`, `SupervisedAutoencoder`, `make_loader`, `fit_fold`, `expanding_splits`, `indices_for_dates`, `attach_forward_return`, `block_bootstrap_auc`.
- Optimizer everywhere: `optim.Adam(lr=0.001)`.
- Registry access (09): `case_studies/<cs>/run_log/registry.db` rows keyed by family/estimator/label with `ic_mean_daily` and HAC bounds; prediction artifacts joined on (timestamp, entity); `label_horizon`, `selected_predictions`, `paired_daily_ic`, `mean_daily_score_correlation`.
- Case-study counterparts: `case_studies/etfs/11_latent_factors.ipynb` (production PCA with walk-forward CV), `case_studies/us_firm_characteristics/08_latent_factors.ipynb` (IPCA, CAE), `case_studies/cme_futures/10_latent_factors.ipynb` (cross-asset PCA).

## Evidence from the book

Numeric results (variance shares, ICs, AUCs, Sharpe, pricing errors) are printed at run time and were not recorded in the digest; only qualitative findings are stated here.

| Notebook | Finding |
|---|---|
| 01 (sector ETFs 2010-2024) | PC1 loadings uniformly positive (market); PC2 separates defensive (Utilities, Staples positive) from cyclical (Energy, Financials, Discretionary negative). PC1 share rises in stress windows; rolling 63-day PC1-PC2 correlation departs from zero. |
| 02 (top-500 US equities 2006-2018) | PC1 captures a substantial share; first 5 components roughly half of cross-sectional variance; commodity exposure (Energy + Materials) dominates PC2, so the textbook defensive-vs-cyclical rotation does not appear where expected. Residual AR(1) slopes near zero (little one-day persistence). PC1 dominates the seeded 20-stock portfolio's risk (in-sample). Two-speed estimate changes the condition number vs uniform covariance; no risk-performance claim. |
| 03 (8 CMTs 2000-2024) | Three components account for essentially all daily-change variance; loadings reproduce Litterman-Scheinkman Level/Slope/Curvature; three key-rate positions zero all three fitted exposures (algebraic). |
| 04 (synthetic, K=3) | ALS recovers the loading subspace (high principal-angle cosines); every Stage 2 forecaster (expanding mean, AR(1), EWMA-12) is indistinguishable from the zero-return baseline because factor premia are independent mean-zero draws; the K sweep documents sensitivity only. |
| 05 (59 ETFs) | Training mean fit rises with kappa while evaluation reconstruction stays nearly flat; MSE differences lie within a couple of percentage points of the zero benchmark; IC intervals answer only "clears zero", not ranking. |
| 06 (CAE, 500 stocks) | Error ratios sit a fraction of a percent from the zero-return forecast; every rank-IC interval spans zero. Good reconstruction, no next-day signal from simple premium adapters. |
| 07 (SDF, 200 stocks) | Validation-Sharpe-selected kernel; test pricing-error concentration share printed; validation MSE below train MSE is a scale artifact. |
| 08 (SAE, 300 stocks) | AUC per horizon with 40-date block-bootstrap intervals (no numbers in digest). |
| 09 (registries) | Coverage uneven (broader panels carry CAE/SDF/SAE); no estimator leads every panel; on the US Firms primary label the five estimators' HAC intervals overlap heavily; latent-vs-supervised paired differences favor each side on different panels; off-diagonal monthly rank correlation among CAE/SDF/SAE is well short of one (diversity for Chapter 20). |

Reader uncertainty carried from the notes: book prose for 14.1-14.9 (including the Figure 14.9/14.10 Stage 2 forecaster catalog) was not available; the remaining three CAE characteristics, the 8 SDF and 17 SAE characteristic definitions, the exact BBP edge / effective-T computation, `fit_rppca` and `weighted_covariance` internals, 01's circular wrapping, the SAE auxiliary head's input, and the registry schema beyond `ic_mean_daily` + HAC bounds are not confirmed.

## Related references

- `chapters/02_financial_data_universe.md` — loading ETF, US-equity and FRED panels used by notebooks 01-08.
- `chapters/07_defining_the_learning_task.md` — purge/embargo, label horizons, forward-return labels used by the SAE and the walk-forward adapters.
- `chapters/09_model_based_features.md` — HMM on factor histories (9.4) and regime features (9.5) built on top of Stage 1 outputs.
- `chapters/11_ml_pipeline.md`, `chapters/12_gradient_boosting.md`, `chapters/13_dl_time_series.md` — the supervised families (linear, gbm, tabular_dl, deep_learning) compared against latent estimators in 09.
- `chapters/15_causal_estimation.md` — causal effects as the next step after statistical structure.
- `chapters/16_strategy_simulation.md` — factor investing and factor-based backtesting of Stage 3 signals.
- `chapters/17_portfolio_construction.md` — walk-forward risk-model comparison (where two-speed covariance and condition-number claims get tested).
- `chapters/18_transaction_costs.md` — costs that dominate stat-arb residual half-life strategies.
- `chapters/19_risk_management.md` — HRP (distinct from HPCA) and risk decomposition in use.
- `chapters/20_strategy_synthesis.md` — holdout protocol and ensemble combination that consume the diversity 09 documents.
- `case_studies/etfs.md` — `case_studies/etfs/11_latent_factors` production PCA with walk-forward CV.
- `case_studies/us_firm_characteristics.md` — `08_latent_factors` IPCA and CAE on the firm-characteristics panel.
- `case_studies/cme_futures.md` — `10_latent_factors` cross-asset PCA; `product` -> `entity` normalization in 09.
- `case_studies/us_equities_panel.md` — the US Equities panel behind 02, 06, 07, 08.
- `libraries/ml4t_diagnostic.md` — `cross_sectional_ic_series`, `compute_ic_uncertainty`, HAC intervals.
- `libraries/ml4t_data.md` — ETF Universe, US Equities and FRED loaders.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `workflow.md`, `companion_repo.md` — cross-cutting indexes.
- Further reading:
  - Litterman & Scheinkman (1991), Common Factors Affecting Bond Returns — yield-curve Level/Slope/Curvature.
  - Avellaneda (2019), Hierarchical PCA and Applications to Portfolio Management; Avellaneda & Lee (2010), Statistical arbitrage in the US equities market.
  - Kelly, Pruitt & Su (2019), Characteristics are covariances — IPCA.
  - Lettau & Pelger (2020), Estimating latent asset-pricing factors — RP-PCA.
  - Gu, Kelly & Xiu (2019), Autoencoder Asset Pricing Models (Eq. 16 managed portfolios); Gu, Kelly & Xiu (2020), Empirical Asset Pricing via Machine Learning.
  - Chen, Pelger & Zhu (2024), Deep Learning in Asset Pricing, Management Science 70(2) 714-750 — adversarial SDF.
  - Yirun Zhang (2021), Jane Street first-place supervised autoencoder.
  - Paleologo (2025), The Elements of Quantitative Investing, Ch. 7 — two-speed covariance.
  - Baik, Ben Arous & Peche (2005) — BBP phase transition / noise edge.
  - Context: Harvey (2017), Harvey & Liu (2019), Harvey et al. (2016), Hou et al. (2015/2020/2021), McLean & Pontiff (2016), Feng et al. (2020), Giglio et al. (2021), Gospodinov et al. (2014/2017), Bryzgalova et al. (2025), Didisheim et al. (2023), Kelly et al. (2025), Engel et al. (2025), Bagnara (2024), Fama & French (1993/2015), Barillas & Shanken (2018), Cochrane (2011), Connor & Korajczyk (2009), Swade et al. (2023), Chen (2024), Jensen et al. (2022), Idzorek et al. (2024), Hwang et al. (2025), Kisiel et al. (2023), Goyal (2012).

## Glossary

- **Latent factor** — unobserved common driver recovered from the return panel rather than prespecified.
- **Covariance-explaining vs priced factor** — direction carrying variance vs direction carrying expected return; the two can diverge.
- **Three-stage framework** — Stage 1 extract factors/loadings; Stage 2 forecast the factor premium from available history; Stage 3 map to asset signals.
- **Correlation-PCA** — PCA on standardized (unit-variance) series.
- **Eigenportfolio** — eigenvector used as portfolio weights, L1-normalized in raw-return space.
- **Scree plot / elbow** — eigenvalues in descending order; a retention heuristic, not evidence.
- **Moving-block bootstrap** — resampling contiguous blocks to preserve short-run dependence.
- **HPCA** — hierarchical PCA: within-sector PCA then across-sector PCA.
- **OU half-life** — -ln 2 / ln|phi| for AR(1) slope 0 < phi < 1.
- **Procrustes rotation** — K x K orthogonal R minimizing ||B_t R - B_{t-1}||_F.
- **BBP edge** — Baik-Ben Arous-Peche phase-transition threshold separating signal eigenvalues from noise.
- **Two-speed covariance** — fast EWMA residual volatility plus slow EWMA factor correlation structure.
- **Level / Slope / Curvature** — first three yield-change PCs: parallel shift, steepening, butterfly.
- **Generalized duration** — sensitivity to each yield-curve factor; hedged with key-rate instruments.
- **IPCA** — instrumented PCA: loadings linear in lagged characteristics, Gamma estimated by ALS.
- **ALS** — alternating least squares between factor realizations and the loading map.
- **Principal angles / projector distance** — rotation-invariant subspace recovery diagnostics.
- **RP-PCA** — PCA on Sigma + kappa * rbar rbar^T to reward priced directions.
- **Managed portfolio** — joint cross-sectional LS coefficient x_t = (Z'Z)^{-1} Z' r_t per date.
- **CAE** — conditional autoencoder: beta network over characteristics, factor network over managed portfolios.
- **SDF / pricing kernel** — M = 1 - omega^T R^e; factor F = omega^T R^e.
- **Moment network** — adversary generating instruments that maximize conditional pricing error.
- **SAE** — supervised autoencoder with reconstruction + aux + main classification heads sharing one bottleneck.
- **Purge / embargo** — drop training dates within the label horizon of validation; separate the test window by a further horizon.
- **HAC / Newey-West interval** — serial-dependence-robust uncertainty for a per-date IC series.
- **Rank IC** — per-date Spearman correlation of prediction and target, averaged; `ic_mean_daily`.
- **Paired daily IC** — both models evaluated on the identical timestamp-entity cross-section.
