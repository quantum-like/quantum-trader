# Chapter 5: Synthetic Financial Data

> Trading research has one realized price path and a process that searches it repeatedly, so backtests are fragile before any generator exists; synthetic paths widen robustness analysis beyond that single history. This chapter equips you to simulate paths with classical baselines (GBM, Merton jump-diffusion, Ornstein-Uhlenbeck, Heston, GARCH, IID/block/stationary bootstrap), to train learned generators (TimeGAN, Tail-GAN, Sig-WGAN, GT-GAN, Diffusion-TS, GReaT, DP-GAN), and above all to validate any of them with one fidelity / utility / privacy protocol. Its position: the validation protocol matters more than the architecture; the structure that matters sits in the tails, in dependence shifts and in conditional dynamics, not at the centre of the distribution; and a learned generator earns its place only by beating the classical baseline on the diagnostics that matter. Fidelity, utility and privacy trade off rather than improving together.

## When to use this reference
- Generating synthetic return or price paths for stress tests, VaR/ES, scenario analysis, or backtest robustness beyond the one realized history.
- Choosing between a classical simulator (GBM, jump-diffusion, OU, Heston, GARCH, bootstrap) and a learned generator for a stated downstream use.
- Calling `ml4t.data.providers.SyntheticProvider` and needing to know which parameters it exposes, fixes internally, or silently ignores.
- Augmenting ML training data with synthetic sequences or tabular rows and deciding whether the augmentation helps (TSTR vs TRTR).
- Evaluating any synthetic dataset: stylized facts, discriminative score, PCA/t-SNE, TSTR, privacy or mode-collapse diagnostics.
- Building or debugging a GAN, diffusion or LLM-tabular generator on financial data (scaling vs output nonlinearity, checkpoints, seeds, loss plateaus).
- Working with naturally irregular data (tick/volume/dollar bars) that a fixed-grid generator cannot represent.
- Producing regime-conditioned stress paths (e.g. high-volatility regime) for the Chapter 20 strategy-synthesis workflow.
- Sharing synthetic versions of sensitive records that need a formal (epsilon, delta) differential-privacy guarantee.
- Reviewing a claim that a generator "matches the data" (moment match, single-draw TSTR, aggregate VaR error, flat losses) for the failure modes this chapter documents.

## Core ideas (the why)

- **Path-limited, adaptive research inflates backtests.** One realized path plus repeated strategy search is selection bias: the best of many trials looks significant (Bailey & Lopez de Prado 2014 Deflated Sharpe; Bailey et al. 2015 PBO; Lopez de Prado 2022 Type I/II errors of Sharpe under multiple testing). Synthetic paths let you test on many histories rather than one.
- **Evaluate first, then build.** Section 5.2 precedes every generator because a generator can look plausible and still be unusable: what matters is in rare events, dependence shifts and conditional dynamics. Hold every generator to the same fidelity / utility / privacy protocol; the three trade off.
- **Why simulate** (`05_synthetic_data/00_classical_simulation`): risk management (VaR, stress, tails), backtesting beyond history, data augmentation for ML, privacy (share synthetic instead of proprietary data).
- **Classical models define the benchmark a learned generator must beat.** They are interpretable, sample-efficient and easy to validate. Parametric models can emit values beyond history but every property is chosen or fitted; bootstraps reproduce the empirical marginal (fat tails included) with no assumptions but cannot exceed observed extremes.
- **Fat tails and volatility clustering are separate properties**; models order differently on each. "Matching one statistic is not matching the series" is the argument for the full protocol.
- **Compare populations, not single draws.** One path per model cannot separate model properties from draw noise (excess-kurtosis SE sqrt(24/T) is about 0.22 at T = 504, larger than the gaps between models). Simulate `N_PATHS = 200` per model with GBM as the null.
- **Pooled first two moments are a weak requirement**: they say nothing about ordering within a sequence, clustering or cross-asset dependence; a sequence-level classifier exploits all three (TimeGAN lesson).
- **Aggregate accuracy across units does not establish accuracy for any unit** (Tail-GAN: mean VaR/ES across 32 strategies agrees while individual strategies do not).
- **Discriminative score and TSTR answer different questions** (indistinguishability vs downstream utility); report both. A TSTR ratio means nothing unless both models clear the naive baseline.
- **Which GAN = which inductive bias matches the failure mode**: temporal structure (TimeGAN), tail risk (Tail-GAN), path fidelity via an analytic distance (Sig-WGAN), irregular sampling (GT-GAN). Never ask "which GAN is best".
- **Returns vs price levels is forced by evaluation, not style**: sigmoid outputs plus min-max scaling fitted on trending training levels leave the reachable support in the holdout; log returns are stationary enough to stay inside.
- **Match the generator to the time grid.** Fixed-grid generators (TimeGAN, Tail-GAN, Sig-CWGAN, Diffusion-TS) assume regular sampling. GT-GAN is for *naturally* irregular data (activity-driven bars, Chapter 3); random masking of daily returns fabricates irregularity and is explicitly "not recommended".
- **Neural ODEs replace discrete recurrence**: h_{t+1} = RNN(h_t, x_t) becomes dh/dt = f_theta(h), h(t) = h(t0) + integral of f_theta(h(s)) ds, so the hidden state can be queried at any time.
- **Diffusion answers GAN instability** (mode collapse, limited interpretability) with a regression objective; Diffusion-TS adds semantics x0_hat = Trend(z) + Season(z) + Residual(z), an inspectable learned STL-like decomposition. Financial returns are unbounded, so image-domain clamps to [-1, 1] must go; the Fourier-domain loss stays because autocorrelation/periodicity is what a returns generator must preserve.
- **Guidance toward a rare regime must be gentler, not harder**: strong classifier gradients toward a minority (high-vol) class drive every sample to extreme volatility, collapsing the very mode guidance was meant to reach.
- **Serialization is the LLM-for-tables insight (GReaT)**: rows become sentences, so numeric and categorical columns are tokens handled uniformly with no distributional assumptions and pre-trained feature-name semantics; GANs need separate categorical encoders.
- **A single sample from a stochastic generator is a draw, not a measurement.** A downstream score can beat the baseline on one draw and fall to chance on the next from the same weights; report the spread.
- **Differential privacy needs per-sample gradients.** Clipping the batch gradient bounds nothing per record; use Opacus, never hand-rolled DP-SGD.
- **Mode collapse is a different object from noise**: +-1 correlations and variance in a couple of directions; a summary distance between correlation matrices cannot see it, so always pair distance metrics with collapse diagnostics.
- **Implementation details change what is simulated**: drift compensator (jump-diffusion), full truncation (Heston), z-score not min-max before signatures, factorial normalization of signature levels, input scaling to the Tail-GAN clamp.
- **Regulatory reality**: when sharing client trading data a privacy-budget ceiling is imposed regardless of where the utility optimum falls.

## Method recipes (the how)

### Classical price models (`05_synthetic_data/00_classical_simulation`)
Conventions: log returns r_t = ln(S_t / S_{t-1}); Fisher excess kurtosis (`scipy.stats.kurtosis(x, fisher=True, bias=False)`); dt = 1/252; S0 = 100; `N_STEPS = 504` (two years); `SEED = 42`.

| Model | Dynamics and discretization | From-scratch function | `SyntheticProvider` exposure | Phenomenon |
|---|---|---|---|---|
| GBM | dS = mu S dt + sigma S dW; exact in log space: S_{t+dt} = S_t exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z). Gaussian log returns, no memory, constant vol | `simulate_gbm(n_steps, mu, sigma, S0=100.0, dt=1/252, rng=None)` | `model="gbm"`, `annual_return`, `annual_volatility`, `seed`: the ONLY model whose full parameter set is exposed, so realized moments are comparable | Null / baseline; option pricing |
| Merton jump-diffusion | dS = mu S dt + sigma S dW + S(e^Y - 1) dN, dN ~ Poisson(lambda), Y ~ N(mu_J, sigma_J^2). Compensator k = exp(mu_J + sigma_J^2/2) - 1. Step: S exp((mu - lambda k - sigma^2/2) dt + sigma sqrt(dt) Z + sum_{i<=N_t} Y_i), N_t ~ Poisson(lambda dt) | `simulate_jump_diffusion(n_steps, mu, sigma, lambda_, mu_jump, sigma_jump, S0, dt, rng)`; notebook uses a negative mean log jump size (downward tilt, up-jumps a printed minority) | `model="gbm_jump"`: only `annual_return`, `annual_volatility`; jumps fixed internally at 5 per year with zero-mean size, so fat tails without skew | Crashes, tail risk (heaviest tails, zero clustering) |
| Ornstein-Uhlenbeck on log price | d(log S) = kappa (theta - log S) dt + sigma dW; half-life ln(2)/kappa. Euler: X + kappa (theta - X) dt + sigma sqrt(dt) Z. Exact: theta + (X - theta) e^{-kappa dt} + sigma sqrt((1 - e^{-2 kappa dt}) / (2 kappa)) Z. Run both from the SAME shock stream so the gap is discretization error only | `simulate_mean_reversion_euler / _exact(n_steps, kappa, theta, sigma, S0, dt, rng)` | `model="mean_revert"`: `base_price` sets the equilibrium (`MR_EQUILIBRIUM = 100.0`), `annual_volatility`; kappa NOT settable (fixed; equals `MR_KAPPA` only by coincidence) | Spreads, commodities, rates; stationary, negative return autocorrelation, no trends |
| Heston | dS = mu S dt + sqrt(v) S dW_S; dv = kappa (theta - v) dt + xi sqrt(v) dW_v; Corr = rho (rho < 0 = leverage). Feller 2 kappa theta > xi^2 keeps the continuous variance strictly positive; below it zero is reached but not absorbed (a valid regime many fitted sets occupy). Full-truncation Euler: max(v, 0) in BOTH drift and diffusion, always | `simulate_heston(n_steps, mu, v0, kappa, theta, xi, rho, S0, dt, rng) -> (prices, variance)`; v0 = theta; `HESTON_MU = 0.05`; Feller assertion reads the live named parameters; assert `heston_var.min() >= 0` | `model="heston"`: `annual_return, heston_kappa, heston_theta, heston_xi, heston_rho`; v0 always = `heston_theta`; leaving `heston_kappa` at its default silently simulates a slower-reverting variance | Stochastic vol, leverage via rho; both stylized facts (weaker than GARCH / JD) |
| GARCH(1,1) | r_t = mu + sigma_t eps_t; sigma_t^2 = omega + alpha (r_{t-1} - mu)^2 + beta sigma_{t-1}^2; alpha + beta < 1; unconditional variance omega / (1 - alpha - beta); leverage needs GJR/EGARCH. Fit by MLE on SPY log returns in PERCENT | `arch_model(spy_log_returns_pct, mean="Constant", vol="GARCH", p=1, q=1).fit(disp="off")`; `simulate_garch(n_steps, mu, omega, alpha, beta, sigma0=None, rng) -> (returns_pct, vol)`; divide by 100; prices = 100 exp(cumsum) | `garch_alpha`, `garch_beta` accepted; `garch_omega` accepted but IGNORED with a warning; omega derived from `annual_volatility` and frequency. Pass annual vol = sqrt(omega / (1 - alpha - beta)) / 100 * sqrt(252) (inference: notes say "convert fitted unconditional variance into annual volatility, divide by 100 before annualizing") | Strongest clustering, second-heaviest tails; calibrated from data |

GARCH vs Heston: discrete vs continuous time; leverage via extensions vs built-in rho; calibration by MLE on returns vs option surface; limited analytics vs semi-closed form.
`SyntheticProvider(...)` returns an OHLCV DataFrame (`df["close"]`) driven by its own RNG; provider paths never match from-scratch paths step for step. Compare realized vol / skew of provider vs from-scratch populations, not the paths.

### Population comparison protocol (`00_classical_simulation`)
1. Seed per use: `STREAMS` names one independent `np.random.Generator` per use, `stream(name)` hands it out; `simulate_path_population(n_paths)` uses independent child streams of the `model_paths` seed; `N_PATHS = 200`.
2. `compute_model_stats(prices, name)` on log returns; `summarize(column)` = median and 5th-95th percentile by model; `dominance(a, b)` = share of (a, b) path pairs where a has the higher excess kurtosis (replaces reading an ordering off medians).
3. Null: GBM's spread over `N_PATHS`. A model "separates" only if its spread sits clear of GBM's 95th percentile on most paths.
4. `clustering_statistic(prices)` = sum of autocorrelations of squared log returns over the first `CLUSTER_LAGS = 20` lags; GBM is the null here too.
5. Compare with SPY on equal-length windows only (nine SPY windows of `N_STEPS`), never full history vs 504-step paths: bias and variance of each ACF estimate depend on sample length.
6. Models are NOT at a common volatility (`GBM_SIGMA` and sqrt(`HESTON_THETA`) at one level; `MR_SIGMA`, `JD_SIGMA` at another; GARCH inherits the SPY fit). Read only scale-free columns (excess kurtosis, skewness, clustering); volatility and drawdown columns are inputs, not findings.

### Bootstraps on SPY log returns (`00_classical_simulation`)
| Method | Mechanics | Function | Preserves |
|---|---|---|---|
| IID | draw T indices uniformly with replacement | `iid_bootstrap(data, n_samples, rng)` | marginal incl. fat tails; destroys autocorrelation |
| Moving block | block length b; start uniform in {1..N-b+1}; append b contiguous returns; trim to T | `block_bootstrap(data, block_size, n_samples, rng)` | within-block dependence |
| Stationary (Politis & Romano 1994) | start uniform in {1..N}; length L ~ Geometric(1/b) with wrap-around; every position has the same marginal | `stationary_bootstrap(data, expected_block_size, n_samples, rng)` | within-block dependence; stationary resample |

- Block length: b ~ T^{1/3}; finance ~22 days (one month); or CV on out-of-sample statistics. Notebook uses 22 for both block methods so they differ only in fixed vs random length.
- Retention ratio = sum of squared-return ACF over the first n_lags for the resample / same sum for the original (1 = as much clustering as SPY, 0 = none); `N_BOOTSTRAP_REPLICATES = 200` per method.

### TimeGAN (`05_synthetic_data/01_timegan`; Yoon, Jarrett & van der Schaar 2019)
- Five modules sharing one latent space: Embedder (raw -> latent), Recovery (latent -> raw), Supervisor (next latent step), Generator (noise -> latent sequence), Discriminator (real vs fake latent). Stacked GRUs built as `nn.ModuleList` to mirror the TF original; every module ends in a sigmoid (outputs in [0, 1]); Supervisor uses `num_layers - 1` layers.
- Data: `load_price_panel(tickers, start_year="2000")` gap-free wide adjusted close; `load_multi_stock_data` -> daily log returns (one row shorter, later-day timestamps). Temporal train/holdout split; MinMax scaler fitted on train only; `create_sequences(data, seq_length)` overlapping windows; noise z ~ U[0, 1) of shape (batch, SEQ_LEN, n_features).
- "Why not price levels" measurement, run BEFORE loading returns: fit the same MinMax scaler on the same training fraction of the PRICE panel, transform the holdout, count cells outside [0, 1] (values a sigmoid generator can never emit); then load returns and assert the holdout's scaled range stays inside [0, 1].

| Parameter | Default |
|---|---|
| `TICKERS` | BA, CAT, DIS, GE, IBM, KO from 2000 (`us_equities`) |
| `SEQ_LEN` / `HIDDEN_DIM` / `NUM_LAYERS` | 24 / 24 / 3 |
| `BATCH_SIZE` / `LEARNING_RATE` (Adam) | 128 / 1e-3 |
| `TRAIN_STEPS` per phase | 10,000 |
| `D_GATING_THRESHOLD` | 0.15 |
| `SEED` / `RETRAIN` | 42 / False |

Three phases, step-based via `infinite_dataloader()`:
1. Embedding (10,000 steps): Embedder + Recovery as an autoencoder, loss 10 * sqrt(MSE(x, x_tilde)).
2. Supervisor (10,000 steps): predict the next latent step from embedded real data, L_S = ||h_{t+1} - s_hat_t||_2.
3. Joint (10,000 steps): per step, 2 generator+embedder updates (adversarial + supervised + moment loss `get_moment_loss` on mean and std of real vs synthetic + reconstruction), then 1 discriminator update taken only if `d_loss.item() > D_GATING_THRESHOLD`, so an already-winning discriminator does not overwhelm the generator.

Generation: loop over `n_synthetic` in `BATCH_SIZE` chunks, `z = torch.rand(batch, SEQ_LEN, n_features)` -> Generator -> Supervisor -> Recovery -> inverse scaler (chain as in the official TimeGAN; inference). Reseed before generation so trained and loaded paths yield identical sequences.
Checkpoint guard: `RUN_CONFIG` (tickers, seq_len, hidden_dim, num_layers, seed, batch_size, learning_rate, d_gating_threshold, digest of training rows) compared field by field; any mismatch or missing field -> retrain; `SKIP_TRAINING = True` only on a full match.
Evaluation: PCA/t-SNE via `plot_fidelity_comparison()` (legacy `np.random.seed` + `np.random.choice` subsampling); `05_synthetic_data/timegan_metrics.py` LSTM discriminator (held-out accuracy, chance 0.5) and LSTM predictor (TSTR/TRTR error ratio, target 1; linear, unbounded output layer).

### Tail-GAN (`05_synthetic_data/02_tailgan_tail_risk`; Cont, Xu & Zhang 2022/2025)
| Parameter | Default |
|---|---|
| `N_EPOCHS` / `BATCH_SIZE` / `LATENT_DIM` | 3000 / 1000 / 1000 |
| `N_COLS` (steps per scenario) / `N_STRATEGIES` | 100 / 32 random long-short strategies |
| `N_SCENARIO_MULTIPLIER` | 10 (scenarios = batch_size * 10) |
| `SEED` | 1 (pinned to preserve the book's VaR 22.5% / ES 21.0% prose numbers) |
| `MAX_SCALED_RETURN` | 0.95 |
| `CONFIG` | alphas, W, NeuralSort temperature (values not visible in the notes) |
| `Cap` in `compute_pnl` | 10.0 |

Pipeline: scale returns / max|return| to 0.95 of the generator's clamp [-1, 1] so the entire empirical tail is reachable without saturating -> `create_scenarios(returns, n_scenarios, seq_len)` rolling windows -> `create_strategies(n_strategies, n_assets)` -> per epoch: `sample_noise(batch_size, latent_dim, noise_name)` ("gaussian" | Student-t) -> Generator (4-layer MLP with BatchNorm, output clamped to [-1, 1]) -> `compute_pnl(R, strategies, Cap=10.0)` (simplified Transform.py) -> `deterministic_neural_sort(s, tau)` so gradients flow through VaR/ES quantiles -> `Discriminator(batch_size, alphas, W, temp, project, strategies, Cap)` emits (VaR, ES) per alpha and strategy, `project_op` projects onto W * VaR <= ES -> `ScoreCriterion(alphas, W)` sums the quantile-specific strictly consistent score `S_quant(v, e, X, alpha, W)` (built from `G1_quant(v, W)`, `G2_quant(e, alpha)`, `G2in_quant(e, alpha)`; optimizes better than the general `S_stats`). Discriminator minimizes the score on real data; generator trained so the discriminator's estimates on fake match (adversarial roles inferred from the paper; inference).
Evaluation: `compute_portfolio_pnl_np`, `empirical_var_es(pnl, alpha)`; relative VaR/ES error in scaled space (valid because both sides are scaled) plus unscaled values; per-strategy median relative error, share overstated, Pearson correlation, 45-degree scatter; PCA/t-SNE with `flatten_method="mean"` (averages across assets; coarse).

### Sig-WGAN and Sig-CWGAN (`05_synthetic_data/03_sigcwgan_signatures`; Ni et al. 2020/2024)
- Signature Sig(X) = (1, int dX, int int dX (x) dX, ...): the expected signature determines the law under mild conditions; linear functionals of it approximate any continuous path function; truncated signatures are computable. Dimension is O(d^depth):

| Assets | Features after augmentation | Dims at depth 4 | Feasibility |
|---|---|---|---|
| 1 | 5 | 780 | fast |
| 2 | 7 | 2,800 | feasible |
| 5 | 13 | 30,940 | slow |
| 10 | 23 | 303,600 | OOM |

| Parameter (paper `configs/STOCKS/SigWGAN.json`) | Default |
|---|---|
| `N_LAGS` / `SIG_DEPTH` | 16 / 4 |
| `LSTM_HIDDEN_DIM` / `LSTM_N_LAYERS` | 50 / 2 |
| `TOTAL_STEPS` / `BATCH_SIZE` / lr | 2500 / 2000 / 1e-3 with scheduler |
| `MC_SAMPLES_TRAIN` / `MC_SAMPLES_EVAL` | 16 / 500 |
| `BROWNIAN_NOISE_SCALE` / `SEED` | 0.1 / 42 |
| `PAPER_REFERENCE` | `{"sig_w1": 2.76, "tstr_ratio": 1.0}` |

- Data: S&P 500 index close log returns 2005-01-01 to 2020-06-01 (`load_sp500_index()` / `load_sp500_log_returns(start_date, end_date)`), single asset; z-score normalization (mean 0, std 1), NEVER min-max to [0, 1]; train/holdout split for TSTR; `create_rolling_windows` of `n_lags`.
- Augmentation, order is critical: Scale(2, dim=0) -> AddTime (prepend time in [0, 1]) -> LeadLag (doubles the path with offset copies; recovers quadratic variation) -> VisiTrans("I") (adds 2 rows + 1 column). Then signature via `signatory` (`pip install --no-build-isolation signatory`, torch first; x86-only, hence the `py312` docker profile) with factorial normalization (level k times k!, because raw level k is O(1/k!)). Helpers: `augment_path_paper`, `compute_path_dim_paper`, `compute_sig_dim_paper`, `compute_signature_with_factorial_norm`, `augment_and_signature_paper`, GPU twins `*_gpu`, `compute_expected_signature_gpu(paths, depth, config, normalise=True)`. Verify the dimension check prints expected == actual before training.
- Generator `LSTMGenerator(input_dim, output_dim, hidden_dim=50, n_layers=2, init_fixed=True)`: Gaussian increments * 0.1 -> cumsum = Brownian path -> LSTM -> linear projection; `ResFNN` initializes the hidden state. NOT an AR-FNN with iid noise.
- Loss Sig-W1(mu, nu) = ||E[Sig(X)]_mu - E[Sig(X)]_nu||_2 (L2 norm, RMSE-style, NOT squared MSE); real expected signature precomputed once; `train_sigwgan_paper(generator, expected_sig_real, config, device) -> losses`. Conditional Sig-CWGAN (explained, not built): for each past window generate m futures, average their signatures, compare with the conditional target.
- Evaluation: `evaluate_sig_w1`, KS statistic on marginals, absolute mean error, `tstr_evaluation_unconditional` (features from paths, label = sign of last return, classifier accuracy real-trained vs synthetic-trained), `compute_stylized_facts` (excess kurtosis, skew, lag-1 ACF of returns and squared returns, ACF to 10 lags), `plot_path_comparison_unconditional` (30 windows each).

### GT-GAN for irregular timestamps (`05_synthetic_data/04_gtgan_irregular`; Jeon et al. 2022)
| Parameter | Default |
|---|---|
| Data | Chapter 3 NVDA dollar bars via `load_chapter3_bars(bar_type="dollar")`; `features = ["close", "volume"]`; fallback `generate_synthetic_irregular_bars(n_bars=2000, seed=42)` with log-normal inter-arrivals (demo-only) |
| `SEQ_LENGTH` / `LATENT_DIM` / `HIDDEN_DIM` / `ODE_HIDDEN` | 32 / 32 / 24 / 24 |
| `MAX_STEPS` / `BATCH_SIZE` / `learning_rate` | 2000 / 128 / 1e-3 |
| `ODE_METHOD` | "rk4" (only "euler" or "rk4"; config cell rejects anything else) |
| `holdout_fraction` / `weights_version` | 0.2 / "2.2.0" (bump when the architecture changes) |
| `GOOD_RECONSTRUCTION_MSE` / `N_SYNTHETIC` | 0.01 / min(200, len(sequences_norm)) |

Pipeline: `load_chapter3_bars(bar_type)` -> `compute_inter_arrival_times(df)` (timestamps normalised to [0, 1]; the variable spacing is the signal the ODE exploits) -> temporal split -> `create_irregular_sequences(df, features, seq_length, times)` (stride 1, per-sequence normalised timestamps; scaler saved in the checkpoint).
Architecture (all ODE-based): `ODEFunc(hidden_dim)` drift dh/dt = f_theta(h), torchdiffeq interface `forward(t, h)` ignoring t; `GRUODECell(input_dim, hidden_dim, ode_method)` GRU update at observation times and ODE evolution between them, each sample stepped by its own `dt_per_sample` (De Brouwer et al. 2019); `ODEEncoder(input_dim, hidden_dim, latent_dim)` -> (mu, logvar) with VAE-style `reparameterize`; `ODEDecoder(latent_dim, hidden_dim, output_dim).forward(z, times)`; `ODEGenerator(noise_dim, latent_dim, hidden_dim)` via `torchdiffeq.odeint` (batch shares one span); `ODEDiscriminator(input_dim, hidden_dim).forward(x, times)`.
Per-sample integration `ode_evolve(func, h, dt, method)`: Euler h + dt f(h) (one evaluation); RK4 h + (dt/6)(k1 + 2k2 + 2k3 + k4) (four). Valid because the drift ignores t, so a column of per-row dt suffices. Adaptive solvers (`torchode`, Lienen & Gunnemann 2022) are blocked by a `torchtyping` incompatibility with current PyTorch.
Training `train_gtgan(model, sequences, times, config, device) -> dict`: step-based via `infinite_dataloader()`, `DataLoader(batch_size=128, shuffle=True)`; reconstruction + adversarial (D real, D fake, G) losses. Checkpoint identity `CHECKPOINT_IDENTITY = ("seq_length", "features", "latent_dim", "hidden_dim", "ode_hidden", "max_steps", "batch_size", "learning_rate", "ode_method", "holdout_fraction", "weights_version", "data_digest", "seed")`; `data_digest` hashes the bars themselves because the config names features and window but not the bars they were cut from.
Generation `generate_synthetic(model, n_samples, times, device)`: the time grid is an INPUT; the model never generates the arrival process. Reseed torch before generation (measured: without it the training and loading paths returned different TSTR ratios from one set of weights).
Evaluation: PCA/t-SNE (motivational only); `evaluate_interpolation(model, sequences, times, device, rng)` decodes at midpoints (t[:-1] + t[1:]) / 2, `smoothness_ratio = mean|diff(interp)| / (mean|diff(real)| + 1e-8)`; `evaluate_statistics(real, synthetic)` KS and correlation error; `gtgan_paper_evaluation(...)` = TimeGAN protocol (discriminative accuracy and AUC, TSTR MAE ratio via `create_prediction_data`, interpolation `bounded_fraction = mean((interp >= lower - 0.1) & (interp <= upper + 0.1))` in normalised units); `plot_irregular_sequences(real_seq, real_times, synthetic_seq, feature_idx=0)`.

### Diffusion-TS (`05_synthetic_data/05_diffusion_ts`; Yuan & Qiao 2024)
| Parameter | Default (paper) |
|---|---|
| `SEQ_LENGTH` / `FEATURE_SIZE` | 60 (24) / 20 ETFs (6): 1,200 values per sample vs paper 144, so train longer for cross-asset structure |
| `d_model` / `n_heads` / `n_layer_enc` / `N_LAYER_DEC` | 64 / 4 / 2 / 4 (paper 2) |
| `TIMESTEPS` / `SAMPLING_TIMESTEPS` (DDIM) / `eta` | 500 / 50 / 0.0 deterministic ("eta = 1 made variance worse, not better") |
| `beta_schedule` / `loss_type` | "cosine" (`cosine_beta_schedule(timesteps, s=0.008)`) / "l1" time-domain + Fourier-domain term (`reg_weight`) |
| `EPOCHS` / `BATCH_SIZE` | 10000 / 64 (`drop_last=True`) |
| `lr` / `warmup_lr` / `WARMUP_STEPS` | 1e-5 / 8e-4 / 500, then `ReduceLROnPlateau(factor=0.5, patience=500, min_lr=1e-5, threshold=0.1)`; gradient clipping |
| `ema_decay` / `GRADIENT_ACCUMULATE_EVERY` | 0.995 (`EMA(model, decay=0.995, update_every=10)`, sample from the EMA model) / 2 |
| `CLASSIFIER_EPOCHS` / `classifier_lr` / `n_regimes` / `MIN_SEQUENCES` | 1500 / 5e-4 / 2 / 20 |
| Data | `load_returns_data(start_date="2005-01-01", n_assets)` (ETF Universe), holdout from `holdout_start="2024-01-01"`, StandardScaler, `create_sequences` overlapping windows |
| Outputs | `N_SYNTHETIC = 500` unconditional; `N_COND = 100` per regime (consumed by Chapter 20 stress tests) |

Architecture: `Conv_MLP(in_dim, out_dim)` 1-D conv embedding; `LearnablePositionalEncoding(d_model, dropout=0.1, max_len=1024)`; `SinusoidalPosEmb(dim)` for the diffusion step; `AdaLayerNorm(n_embd)` scale + shift from the timestep embedding; `Encoder(n_layer=2)` of `EncoderBlock(n_embd=64, n_head=4, mlp_hidden_times=4)` (AdaLN -> `FullAttention` -> FFN); `Decoder(n_layer=4)` of `DecoderBlock` (self-attn + `CrossAttention` to encoder output, then `TrendBlock` with degree-3 polynomial basis [t, t^2, t^3] and `FourierLayer(d_model, low_freq=1, factor=1)` DFT -> `topk_freq` -> inverse DFT, a learned bandpass). Trend and seasonal residuals accumulate across decoder layers; `DiffusionTransformer(n_feat, n_channel, n_layer_enc=2, n_layer_dec=4, n_embd=64, n_heads=4, max_len=2048).forward(x, t)` returns `trend` and `season_error`, sum = x0_hat.
`DiffusionTS(seq_length, feature_size, n_layer_enc=2, n_layer_dec=4, d_model=64, timesteps=500, sampling_timesteps=None, loss_type="l1", n_heads=4, mlp_hidden_times=4, eta=0.0, reg_weight=None)`: x0-prediction; `q_sample(x_start, t, noise)`, `predict_noise_from_start(x_t, t, x0)`, `q_posterior`, `p_mean_variance` WITHOUT clamp; `sample(shape)` full 500 steps; `fast_sample(shape)` DDIM 50 steps; `p_sample(x, t, cond_fn, model_kwargs)`; `fast_sample_cond` / `sample_cond(shape, cond_fn, model_kwargs, eta)`; `generate_mts(batch_size=16, model_kwargs, cond_fn, eta)` entry point; `return_components(x, t)` -> trend, seasonal, residual, x_noised.
Post-hoc variance rescale: `variance_ratio = synthetic_norm.std() / sequences_norm.std()`; if < 0.9, multiply unconditional samples by `sequences_norm.std() / synthetic_norm.std()` and print the factor, then `scaler.inverse_transform`. Regime samples are NOT rescaled.
Regime labels: `GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=42)` on SPY returns, Viterbi `hmm.predict`, relabelled Low-Vol / High-Vol via `label_map`; each sequence labelled by its last day `regime_labels[i + seq_len - 1]` (markdown says "majority regime"; the source uses the final-day label; inference from source); regimes with fewer than `MIN_SEQUENCES = 20` sequences are merged into the nearest. Two regimes because imbalance precludes a 3-way split.
`RegimeClassifier(feature_size, seq_length, num_classes=3, n_layer_enc=2, n_embd=64, n_heads=4, num_head_channels=8).forward(x, t)`: reuses the Encoder + `AttentionPool` (`QKVAttention`, `GroupNorm32`); trained on noised x_t at random timesteps so gradients are meaningful mid-sampling.
Guidance `make_cond_fn(classifier, target_regime, scale=1.0, temperature=1.0)` returns grad_x log_softmax(logits / temperature)[target] * scale; `condition_mean` shifts the mean by sigma^2 grad_x log p(y|x) (Song et al. 2020). `guidance_settings = {0 Low-Vol: scale 0.75, temperature 1.0, eta 0.5; 1 High-Vol: scale 0.3, temperature 2.0, eta 0.7}`; generate with `generate_mts(batch_size=N_COND, cond_fn=..., eta=...)`.
Evaluation: `evaluate_statistics(real_data, synthetic_data)` prints per-asset KS mean/min/max, mean error, std error, lag-1 autocorrelation error mean/max, correlation error (mean |delta| over pairs), `n_assets`; PCA/t-SNE; `tstr_evaluation(train_data, holdout_data, synthetic_data)` with label = next-day |return| (`[:, -1, 0]`) > 90th percentile of absolute TRAINING returns, printing real- and synthetic-trained accuracy, naive baseline `(y_test == majority).mean()`, precision, recall, TSTR ratio; regime comparison (volatility ratio, mean return per regime, shared-y-axis path panels, volatility distributions).

### GReaT LLM tabular generation (`05_synthetic_data/06_llm_tabular_great`; Borisov et al. 2023)
- Features `load_etf_tabular_data(symbols=None, start_date="2015-01-01", n_samples=2000)`, per symbol for i in [20, len - 5): numeric `ret_1d`, `ret_5d`, `ret_20d`, `vol_20d` (20-day std of daily returns), `vol_ratio = volume[i] / mean(volume[i-20:i])`, `fwd_ret_5d` (target), `abs_fwd_ret_5d`; categorical `direction` (up if ret_1d > 0 else down), `momentum` (strong if |ret_20d| > 0.05, weak if > 0.02, else flat), `vol_regime` (high if vol_20d > 0.02, normal if > 0.01, else low); `target = abs_fwd_ret_5d > quantile(0.90)` of the full sample; keep `timestamp` and `symbol` for the temporal split.
- Model: `GReaT(llm="distilgpt2", batch_size=16, epochs=50, experiment_dir=str(get_output_dir(5, "great") / "trainer_great"), save_steps=5000, logging_steps=100)` (82M params; larger backbones trade compute for fidelity without changing the pipeline); fit on `train_df`.
- Sampling: `great.sample(n_samples=500, max_length=500, guided_sampling=True)`. Guided sampling enforces the column schema row by row; needed on short fine-tunes (unguided drops columns on undertrained models); `guided_sampling=False` is faster and equivalent on a well-trained checkpoint.
- Parse accounting: per-column parse counts (rows whose cell converts to a number). Counting nulls understates failures because a cell holding "not the case" is a string the parser left in place.
- TSTR: temporal split `TRAIN_FRACTION = 0.7` (earliest share trains); two identical gradient-boosting classifiers, TRTR on real train vs TSTR on one synthetic draw (`synthetic_training_set(frame)`), compared by AUC on the real test split (`tstr_auc_for_draw(frame)`). Verdict `tstr_utility_level(auc_trtr, auc_tstr)`: `auc_tstr <= 0.5 -> "NONE"`; `auc_trtr <= 0.5 -> "NO BASELINE"`; ratio = auc_tstr / auc_trtr > 0.95 -> "HIGH", > 0.85 -> "MODERATE", else "LIMITED"; `tstr_utility_verdict` wraps it; `tstr_level_tally(levels)` counts verdicts. Fine-tune once, repeat only the draw `TSTR_DRAWS = 5` times (first draw reused), report all AUCs and the tally.
- Reproducibility: fix loader ordering (`Series.unique` order undefined, ties on timestamps shared by ~100 ETFs), then `set_global_seeds(SEED)` pins all five draws to the digit (`be_great.sample()` exposes no seed argument).

### DP-GAN with Opacus (`05_synthetic_data/07_dp_gan`; Abadi et al. 2016)
| Parameter | Default |
|---|---|
| Data | ETF Universe loader from `start_date="2020-01-01"`; `n_features = 6`: `ret_1d, ret_5d, volatility, volume_ratio, momentum_z, range_pct`; `MAX_SYMBOLS = 0` (all default symbols) |
| `HIDDEN_DIM` / `LATENT_DIM` | 128 / 32 |
| `EPOCHS` / `batch_size` / `learning_rate` | 20 / 64 (`drop_last=True`) / 1e-3 |
| `epsilon` / `delta` / `max_grad_norm` | 10.0 / 1e-5 (must be << 1/n) / 1.0 |
| WGAN weight clip | `clip_value = 0.01` (`p.data.clamp_(-0.01, 0.01)`) |
| `N_GENERATE` | min(len(real_data), 5000) |
| Sweep | `epsilon_values = [1.0, 5.0, 10.0, 50.0]`, fresh gen/disc per budget, `epochs=10`, `SWEEP_EVAL_ROWS = min(1000, len(real_data))` random rows |

- (epsilon, delta)-DP: for adjacent D, D' (one record differs) and any output set S, P[M(D) in S] <= e^epsilon P[M(D') in S] + delta. epsilon = privacy budget (lower = more privacy, less utility); delta = breach probability.
- `Generator(latent_dim, hidden_dim, output_dim)`: MLP with `Tanh` output * 3 -> [-3, 3] to match the normalised range; no DP changes (it never sees real data). `Discriminator(input_dim, hidden_dim)`: `nn.GroupNorm(1, hidden_dim)` instead of BatchNorm, no in-place ops; `ModuleValidator.validate(discriminator, strict=False)` then `ModuleValidator.fix(discriminator)`.
- `train_dp_gan(generator, discriminator, train_loader, epochs, target_epsilon, target_delta, max_grad_norm) -> dict`: `PrivacyEngine(accountant="rdp")`; `make_private_with_epsilon(module, optimizer, data_loader, target_epsilon, target_delta, epochs, max_grad_norm)` auto-calibrates `opt_d.noise_multiplier` to spend exactly the target epsilon over the declared epochs (printed); per-sample grads via functorch, per-sample clipping to `max_grad_norm`, calibrated Gaussian noise, RDP accounting; `privacy_engine.get_epsilon(target_delta)` logged per epoch (monotone), `final_epsilon` at the end; Lipschitz via weight clipping, not gradient penalty.
- `generate_samples(generator, n_samples)` denormalises; `evaluate_quality(real, synthetic)` -> mean absolute difference, correlation distance; `collapse_diagnostics(sample, scale)` -> `mean_abs_offdiag_corr` (off-diagonal mean |corr| of `np.corrcoef(sample, rowvar=False)`) and `dims_for_99pct_variance` (SVD of the sample standardised by the REAL per-feature std, `searchsorted(cumulative_share, 0.99) + 1`, so the count is about sample shape, not units).

### Validation protocol for any generator (Section 5.8, shared by every notebook)
| Stage | Compute | Target / reading |
|---|---|---|
| 1 Stylized facts (fidelity) | excess kurtosis, skew, squared-return ACF, cross-asset correlation vs real; per-asset KS mean/min/max; lag-1 autocorrelation error | population comparison, equal-length windows; max near mean = even fit, max far above = concealment |
| 2 Dependence / diversity | PCA, t-SNE (motivational only); discriminative score (held-out accuracy or AUC of a real-vs-synthetic classifier) | 0.5; investigate large departures in either direction; GT-GAN band +-0.15 |
| 3 Task utility | TSTR vs TRTR with naive baseline, base rate, precision, recall, AUC; repeated draws | ratio ~1 (GT-GAN MAE ratio in (0.7, 1.5)); ratio is meaningless unless both models clear the naive baseline |
| 4 Privacy | leakage checks, epsilon spent per epoch, collapse diagnostics | epsilon ceiling from the application / regulator; collapse is not noise |
Only after all four stages compare architectures.

## Guardrails and pitfalls
- **Adaptive search / multiple testing** — repeated strategy search on one path inflates in-sample Sharpe / selection bias makes the best of many trials look significant / apply Deflated Sharpe and PBO; test on many synthetic paths, not the one history.
- **Reading one path per model** — single realizations cannot separate model properties from draw noise / kurtosis SE sqrt(24/T) ~ 0.22 at T = 504 exceeds the gaps between models / simulate `N_PATHS = 200`, compare percentile ranges and dominance shares against the GBM null.
- **Comparing stats across unequal windows** — clustering statistic on SPY full history vs 504-step paths / ACF bias and variance depend on sample length / use equal-length windows on both sides (nine SPY windows).
- **Mixed volatility levels in comparison tables** — models parameterized at different sigma / vol and drawdown columns then report inputs, not findings / read only excess kurtosis, skewness, clustering, or re-run at a common vol.
- **Missing drift compensator in jump-diffusion** — requested mu differs from realized drift / jumps add lambda E[e^Y - 1] to drift / subtract lambda k with k = exp(mu_J + sigma_J^2/2) - 1 computed from the same parameters.
- **Negative variance in Heston Euler** — Euler can propose v < 0 even when Feller holds / the condition is about the continuous process, not the discretization / full truncation max(v, 0) in drift and diffusion; assert `heston_var.min() >= 0`; Feller assertion reads live parameters.
- **`SyntheticProvider` parameter mismatches** — `gbm_jump` fixes 5 symmetric jumps/yr; `mean_revert` fixes kappa; `heston_kappa` default differs; `garch_omega` is ignored with a warning / silently simulates a different process than the from-scratch one / pass full parameter sets where exposed; carry GARCH level via `annual_volatility`; compare realized vol and skew of provider vs from-scratch populations.
- **GARCH percent units** — `arch` fits returns in percent / omega is in percent^2 and simulated returns in percent / divide by 100 before pricing or annualizing.
- **Bootstrap block boundaries** — dependence spanning a boundary is destroyed at any block length; longer blocks leave fewer distinct blocks / only about 2/3 of SPY squared-return dependence survives at b = 22 / measure the retention ratio over 200 replicates; tune block length, not method.
- **Bootstrap cannot exceed history** — no resampled return larger than the max observed / resampling only / use parametric or learned generators for stress beyond history.
- **Price levels with sigmoid / min-max generators** — trending levels leave the [0, 1] range the train-fitted scaler covers / synthetic training data cannot exceed 1 while holdout targets do, confounding TSTR with support shift / model log returns; count holdout cells outside [0, 1] before loading; assert the holdout stays inside the fitted range.
- **Scaler leakage** — fitting MinMax / z-score / StandardScaler on the holdout / holdout information reaches the model / temporal split first; fit on the training period only; the generator never sees the holdout.
- **Stale or fallback checkpoints** — weights trained on another config, data refresh or the synthetic fallback bars load silently with their scaler / every downstream number describes another model; holdout and training land on different scales / store the full config plus a data digest beside the weights (`RUN_CONFIG`, `CHECKPOINT_IDENTITY` incl. `weights_version`); any mismatch or missing field retrains.
- **RNG divergence between train and load paths** — training consumes millions of draws, loading none / synthetic noise and evaluation inits differ; measured: different TSTR ratios from identical GT-GAN weights / reseed (numpy and torch) before generation and evaluation.
- **Moment matching as success** — synthetic mean/std match to 2 decimals yet discriminator accuracy is far from 0.5 / pooled moments ignore ordering, clustering and cross-asset dependence / always add stylized-fact, dependence and discriminative checks.
- **Discriminative accuracy well below 0.5** — not "better than chance" / sampling variation, a classifier that failed to generalize, or a shift between sets / investigate any substantial departure in either direction.
- **TSTR alone, or a direction task** — a ratio near 1 does not mean indistinguishable; return-sign prediction is near coin flip for everyone / utility for one task is not fidelity / report with the discriminative score; use extreme-move classification, where volatility clustering carries signal.
- **Clamped generator output (Tail-GAN)** — outputs in [-1, 1] leave unscaled tails unreachable / fatal for a tail model / divide by max|return| and map to 0.95 of the clamp; verify the full empirical support is reachable before blaming the model.
- **Aggregate hides per-unit error** — mean VaR/ES error across 32 strategies is small while the median per-strategy error is several times larger and about half are overstated / over- and understatements cancel / compute per-strategy median relative error, share overstated, correlation and a 45-degree scatter before quoting a headline.
- **Flat adversarial losses or short runs** — a plateau is not convergence; `MAX_STEPS = 2000` GT-GAN losses flatten onto a common level long before the end / G and D are scored against each other; weights and outputs keep moving / compare checkpoint outputs over time; rely on task metrics; lengthen training before quoting any number as a method property.
- **Discriminator dominance** — D losses near 0 give the generator no useful gradient / watch that D real, D fake and G losses oscillate and roughly balance; reconstruction loss should fall steadily below `GOOD_RECONSTRUCTION_MSE = 0.01`; TimeGAN gates D updates at `d_loss > 0.15`.
- **Pearson ~0 across strategies** — not proof of no relationship / linear association only / use a rank or dependence measure if claiming independence; make claims about agreement (median error) instead.
- **Projection plots as evidence** — PCA/t-SNE can separate what a model cannot and hide what it finds easily; `flatten_method="mean"` averages across assets and shrinks the cloud / coarse coverage view, not the optimized quantity / use the discriminative score as the measured version and read tail metrics on portfolio PnL.
- **Min-max before signatures** — returns in [0, 1] make the cumsum monotone / no down-moves, sign structure destroyed / z-score (centered) normalization.
- **Signature dimension blow-up** — O(d^depth): 10 assets at depth 4 = 303,600 dims / OOM / 1-5 assets at depth 4; check `compute_sig_dim_paper` before training.
- **Unnormalized signature levels** — level k is O(1/k!) / higher levels invisible to the loss / multiply level k by k! in training and evaluation.
- **Sig-W1 absolute value vs paper** — not comparable unless depth, augmentations and input scale match / metric scale depends on all three / read the curve shape (rapid drop in ~150 steps, stable plateau); print `PAPER_REFERENCE` beside run values only when settings match.
- **Thin visual samples and overlapping windows** — 30 overlapping windows per set, or ten real windows with a nine-day shift, are one stretch of tape repeated / little independent information / use the population stylized-facts table and KS/correlation tables.
- **Low-order expected signatures miss higher moments** — Sig-WGAN matches mean/std but loses excess kurtosis, negative skew and squared-return ACF / depth-4 unconditional objective pins location and scale only / conditional Sig-CWGAN, richer augmentations; always check stylized facts.
- **Evaluating on the first N rolling windows** — stride-1 windows on consecutive bars are one period repeated; a quiet or violent week stands in for the whole / draw `real_eval` at random and generate synthetic on that same draw's time grids (also `sweep_eval` and consecutive-day rows in `07_dp_gan`).
- **Numbers typed into prose** — metrics move run to run / text stops tracking the code / print diagnostics from cells; pin seeds (`SEED = 1` in Tail-GAN) when prose numbers must survive.
- **Global RNG vs `default_rng`** — MT19937 and PCG64 sequences differ / persisted projection arrays must match the inline render / use the helper's seeding when re-vendoring figure data.
- **Artificial irregularity** — GT-GAN models an activity-driven arrival process; random masking of regular daily data fabricates it / only feed naturally irregular bars (Chapter 3 tick/volume/dollar); treat the log-normal fallback as demo-only and run Chapter 3 first for production.
- **Batch-shared ODE time grid** — `torchdiffeq.odeint` takes one grid per batch / destroys the per-sample irregularity the model exists to use / step each row by its own dt via `ode_evolve`; use `odeint` only where the batch genuinely shares a span.
- **Interpolation figure read as arrival-process fidelity** — the time grid is an argument to `generate_synthetic`; the figure cannot say whether GT-GAN reproduces bar arrivals / compare decoder values on a real window's timestamps only; use the smoothness ratio for dynamics.
- **Bounded-fraction metric is mostly tolerance** — +-0.1 is wider than the mean gap between adjacent observations and the row never runs the generator / a narrow decoder sanity check / never keep it while discarding the failing discriminative and TSTR rows ("choosing a metric by its answer").
- **Clamping x0_hat to [-1, 1] in diffusion** — returns are StandardScaler-normalised and unbounded / clamping truncates tails / remove the clamp from `p_mean_variance` and `model_predictions`.
- **Variance under-generation by trend + seasonal diffusion** — smooth components cannot carry high-frequency variation / check `variance_ratio`; if < 0.9 apply the post-hoc rescale and print the factor; regime samples stay unrescaled.
- **Over-steering the minority regime** — strong classifier gradients toward a rare class collapse all samples onto extreme volatility / use scale 0.3, temperature 2.0, eta 0.7 for High-Vol vs 0.75 / 1.0 / 0.5 for Low-Vol; tune scale per application.
- **Volatility growing with position in the window** — regime-conditional paths widen left to right / sampler artifact, not a regime property / inspect shared-y-axis side-by-side panels; do not attribute it to the regime.
- **Reading a mean without its range** — absolute errors cannot cancel, so a small mean can hide one badly handled asset / read per-asset max KS and max autocorrelation error beside the means.
- **Mixing conditional and unconditional evidence** — regime histograms plot guided, unrescaled `regime_samples`, not `synthetic_sequences` / keep the two evaluations separate.
- **Imbalanced extreme-move TSTR read by accuracy** — threshold is the 90th pct of TRAINING |returns|; a calmer holdout has ~5% positives, so predict-never scores ~0.95 / print naive baseline and base rate; read precision, recall and AUC; the ratio only means something once both models clear the baseline.
- **Lag-1 autocorrelation of returns as a clustering check** — returns have ~0 AC; clustering lives in squared returns / compute AC on squared returns if volatility clustering is the claim.
- **Decomposition read at a noisy diffusion step** — at t a quarter through the schedule the residual dwarfs trend + seasonal / read vertical scales first.
- **HMM label quality bounds guidance** — a 2-state Gaussian HMM is a simple label source / conditional results are only as good as the labels; see Chapter 9 for richer regime models.
- **Autoregressive parse failures (GReaT)** — undertrained small backbones drop columns or hallucinate tokens, producing strings not nulls / count per-column parses vs rows generated; `guided_sampling=True` on short fine-tunes; more epochs or larger backbones; zero parses leaves nothing to compare.
- **One synthetic draw treated as the generator** — temperature sampling gave AUCs of 0.70, 0.30, 0.78 across executions / repeat the draw `TSTR_DRAWS = 5` times with fit and real sample fixed; claim utility only if the range sits clear of 0.5.
- **Non-deterministic data loading** — `Series.unique` order undefined plus timestamp ties gave a different real sample each run / fix ordering, then `set_global_seeds(SEED)`; verify two consecutive runs agree to the digit.
- **Random (non-temporal) TSTR split** — future rows leak into training / always split by time (`TRAIN_FRACTION = 0.7` earliest) for financial series.
- **Wall-clock from a quiet machine** — three warm byte-identical GReaT runs took 1,603 s, 1,658 s, 3,481 s / per-pass timings are floors; cost by whether the fit (cold) or the draws (warm) dominate.
- **Batch-level clipping passed off as DP** — clipping the summed gradient bounds no individual record / use Opacus per-sample clipping + calibrated noise + RDP accountant; never hand-roll.
- **BatchNorm in a DP model** — batch statistics leak other samples' information / GroupNorm or LayerNorm; `ModuleValidator.validate` and `.fix` before training.
- **Gradient penalty under DP** — differentiates through real/fake interpolations outside Opacus accounting / enforce Lipschitz with weight clipping (0.01).
- **Spending epsilon unevenly or exceeding it** — once the budget is spent further training weakens the guarantee / `make_private_with_epsilon` spreads it across the declared epochs; track `get_epsilon(delta)` every epoch.
- **Collapse misread as privacy noise** — saturated +-1 correlations and thin PCA bands mean a degenerate distribution, not a noisy one / compute `mean_abs_offdiag_corr` and `dims_for_99pct_variance` against the real data's values.
- **Monotone curve read into a one-run sweep** — looser budgets land in a narrow band with no clean ordering; correlation distance is non-monotone / only the tightest budget (epsilon = 1) is a resolved effect; separating moderate budgets needs several independent runs per epsilon, averaged.
- **Choosing epsilon by utility alone** — regulators may cap the budget / treat the ceiling as an external constraint.
- **Generator-specific risks** (chapter objective) — leakage of training records, bias amplification, overfitting to the generator, limited scenario novelty / synthetic data can memorize or amplify / leakage checks, DP training, scenario-novelty review before use.

## Decision rules and defaults
- **Model by phenomenon**: GBM = null / option pricing; jump-diffusion = crashes and tail risk; OU = spreads, commodities, rates; Heston = stochastic vol with leverage via rho; GARCH = strongest clustering, fit by MLE. Never use GBM or OU for VaR/ES (Gaussian tails understate loss by construction). Among JD / GARCH / Heston the median excess kurtosis differs by an order of magnitude, not a rounding choice.
- Need skewed jumps: implement jump-diffusion yourself (provider jumps are symmetric). Need to fit reversion speed: from-scratch OU (provider kappa is fixed).
- Heston: check 2 kappa theta > xi^2; always truncate; start v0 = theta unless you have a reason; pass all four Heston parameters to the provider.
- GARCH: alpha + beta < 1; carry level via annual vol, persistence via alpha and beta; divide `arch` output by 100.
- Bootstrap: block or stationary for returns with clustering; block length ~22 days (or T^{1/3}, or CV); method choice matters less than length; expect ~2/3 dependence retention; IID only when the marginal alone matters.
- Separation test: a model's kurtosis or clustering spread must clear GBM's 95th percentile on most paths; report pairwise dominance shares (e.g. GARCH > Heston on ~3/4 of pairs).
- Classical-vs-SPY comparisons: equal-length windows only.
- **Classical baseline before any learned model**: the learned generator must beat jump-diffusion on tails and GARCH on clustering on the same diagnostics; otherwise prefer the interpretable, sample-efficient baseline (inference from README 5.3).

| Data shape / goal | Generator |
|---|---|
| Regular daily or hourly, continuous features, temporal dynamics | TimeGAN |
| Tail-risk scenarios (VaR/ES of strategies) | Tail-GAN (complex setup) |
| Multi-asset correlations, path fidelity with an analytic loss | Sig-CWGAN (1-5 assets at depth 4) |
| High-dimensional, regime-conditional stress paths | Diffusion-TS |
| Naturally irregular (tick/volume/dollar bars, assets with different trading hours, irregularly sampled backtests) | GT-GAN |
| Mixed-type tabular, no distributional assumptions (slow, expensive) | GReaT |
| Fast, simple tabular with distributional assumptions | Copula |
| Sensitive records needing formal guarantees | DP-GAN |

- Generator inputs: log returns, not price levels, for bounded-output generators; temporal split first; scaler fitted on train only; assert the holdout stays within the fitted range.
- Scaling matched to the output nonlinearity: sigmoid -> MinMax on returns; clamp [-1, 1] -> divide by max|x| to 0.95; signatures -> z-score (never MinMax); diffusion -> StandardScaler with no clamp.
- TimeGAN defaults: seq_len 24, hidden 24, 3 GRU layers, batch 128, lr 1e-3, 10,000 steps per phase, 2 G steps per D step, D step only if d_loss > 0.15.
- Reading scores: discriminative accuracy target 0.5 (investigate large departures either way; GT-GAN band +-0.15); TSTR/TRTR error ratio target 1.0 (> 1 worse; GT-GAN MAE band (0.7, 1.5)); report both; interpolation bounded fraction > 70%; reconstruction MSE < 0.01; smoothness ratio ~1 (<< 1 over-smoothing or rigid ODE, > 1 noisy interpolation).
- Tail-GAN: scale so max|return| -> 0.95; read per-strategy median error, share overstated and scatter before the aggregate (book prose: VaR 22.5%, ES 21.0% with `SEED = 1`).
- Sig-WGAN: z-score; augmentation order Scale(2, 0) -> AddTime -> LeadLag -> VisiTrans("I"); depth 4; factorial normalization; batch 2000; 2500 steps; lr 1e-3; L2-norm loss; verify the dimension check.
- GT-GAN: `ode_method` in {"euler", "rk4"}, default rk4; retrain if any `CHECKPOINT_IDENTITY` field (incl. `data_digest`, `weights_version = "2.2.0"`) differs or `RETRAIN = True`.
- Diffusion-TS: 500 training steps / 50 DDIM sampling steps (~10x speedup, minimal quality loss); `eta = 0.0`; rescale variance when `variance_ratio < 0.9`; guidance Low-Vol {0.75, 1.0, 0.5} vs High-Vol {0.3, 2.0, 0.7} (scale, temperature, eta); higher scale = more separation and more variance; merge regimes with < 20 sequences; 2 regimes, not 3, when imbalanced.
- Extreme-move TSTR label: next-day |r| > 90th percentile of training |r| (`05_diffusion_ts`); |fwd_ret_5d| > 90th percentile of the full sample (`06_llm_tabular_great`). Always print naive baseline, base rate, precision, recall; divide AUCs, not accuracies, at 5% prevalence.
- GReaT verdict: AUC_tstr <= 0.5 NONE; AUC_trtr <= 0.5 NO BASELINE; ratio > 0.95 HIGH, > 0.85 MODERATE, else LIMITED; claim utility only if the 5-draw range sits clear of 0.5. Budget levers: epochs (50), backbone (distilgpt2; GPT-2 medium/large reduce parse errors), `n_generate` (500), `guided_sampling=True` until well trained. Categorical thresholds: momentum strong > 0.05, weak > 0.02; vol_regime high > 0.02, normal > 0.01.
- DP privacy levels: epsilon < 1 very strong (medical / financial PII); 1 <= epsilon <= 10 moderate (most production); epsilon > 10 weak. Defaults epsilon 10, delta 1e-5 (<< 1/n), `max_grad_norm` 1.0, weight clip 0.01, 20 epochs; sweep [1, 5, 10, 50] at 10 epochs.
- DP checklist: no in-place ops -> GroupNorm not BatchNorm -> `ModuleValidator.validate/fix` -> `PrivacyEngine(accountant="rdp")` -> `make_private_with_epsilon` -> track epsilon per epoch -> evaluate with collapse diagnostics beside distance metrics.
- **Pre-training checklist for a learned generator**: (1) stationary target (log returns), output range covers the holdout support; (2) temporal split, scaler on train only, holdout range assertion; (3) scaling matched to the output nonlinearity; (4) seeds pinned per use, reseed before generation; (5) checkpoint stores full config + data digest, mismatch retrains; (6) dimension / GPU / cold-start cost check; (7) diagnostics printed from cells, population comparisons not single draws.
- **Sequencing (all notebooks)**: fix data ordering -> seed -> fit or load identity-checked checkpoint -> reseed -> random eval draw -> paired synthetic generation -> fidelity (KS by asset with min/max, correlation error, autocorrelation) -> utility (TSTR with baseline, precision/recall, AUC, repeated draws) -> privacy where relevant (epsilon, collapse).
- Runtime budgets (cold, GPU): TimeGAN ~12 min; Tail-GAN ~6 min (< 30 s cached); GT-GAN ~6 min; Diffusion-TS ~5 min; GReaT ~13 min (GPU required); DP-GAN ~8 min; Sig-WGAN needs the `py312` docker profile on x86.

## Code patterns and APIs
- Run: `uv run python 05_synthetic_data/<notebook>.py`; tests: `uv run pytest tests/test_chapter_notebooks.py -v -k "05_synthetic_data"` (Papermill reduced data); GPU: `docker compose run --rm ml4t-gpu python 05_synthetic_data/01_timegan.py`; Sig-WGAN: `docker compose --profile py312 run --rm py312 python 05_synthetic_data/03_sigcwgan_signatures.py`.
- Repo helpers: `get_chapter_dir(5)`, `get_output_dir(5, "<gtgan|diffusion_ts|great|dp_gan>")` (checkpoints under `.../checkpoints/`), `set_global_seeds(SEED)`, `load_etfs()`, `load_sp500_index()`, `load_sp500_log_returns(start_date, end_date)`, `load_price_panel(tickers, start_year)`, `load_multi_stock_data(tickers, start_year) -> (returns, timestamps)`, `load_chapter3_bars(bar_type)`, `load_returns_data(start_date, n_assets)`, `load_etf_tabular_data(symbols, start_date, n_samples)`.
- Shared flags: `RETRAIN = False`, `SEED = 42`, `PROGRESS_BARS = False` where present.
- Library provider:
  ```python
  from ml4t.data.providers import SyntheticProvider
  provider = SyntheticProvider(model="heston", annual_return=0.05, heston_kappa=K, heston_theta=TH,
                               heston_xi=XI, heston_rho=RHO, seed=SEED)
  # yields an OHLCV DataFrame (use df["close"]); the exact fetch call is not named in the notes
  # model in {"gbm", "gbm_jump", "mean_revert", "heston", "garch"}; garch_omega is ignored (warned)
  ```
- GARCH fit: `from arch import arch_model; res = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, q=1).fit(disp="off")`.
- Reproducibility (one generator per use):
  ```python
  SEED = 42
  STREAMS = ("gbm", "jump", "mr", "heston", "garch", "bootstrap", "model_paths")
  _ss = np.random.SeedSequence(SEED).spawn(len(STREAMS))                      # (inference)
  def stream(name): return np.random.default_rng(_ss[STREAMS.index(name)])   # (inference)
  ```
- Checkpoint guard (TimeGAN / GT-GAN): build `RUN_CONFIG` or `CHECKPOINT_IDENTITY` with every weight-moving setting plus a digest of the training rows; load only if the saved config equals it field by field and `RETRAIN` is False; set `SKIP_TRAINING = True` on a match; reseed before generation.
- TimeGAN joint step: `z = torch.rand(B, SEQ_LEN, n_features)`; `if d_loss.item() > D_GATING_THRESHOLD: opt_discriminator.step()`; `get_moment_loss(batch, x_hat)`; metrics in `05_synthetic_data/timegan_metrics.py`; figure data under `05_synthetic_data/figures/data/timegan_fidelity/` via `figures/scripts/generate_figure_5_04_timegan_fidelity.py`.
- Tail-GAN: `deterministic_neural_sort(s, tau)`, `S_quant(v, e, X, alpha, W)`, `ScoreCriterion(alphas, W)`, `compute_pnl(R, strategies, Cap=10.0)`, `Discriminator(batch_size, alphas, W, temp, project, strategies, Cap)`, `sample_noise(batch_size, latent_dim, noise_name)`, `empirical_var_es(pnl, alpha)`.
- Sig-WGAN loop (L2 norm, not MSE):
  ```python
  for step in range(n_gradient_steps):
      x_fake = G(batch_size, n_lags, device)
      loss = torch.norm(expected_sig(x_fake) - expected_sig_real, p=2)
      loss.backward(); optimizer.step(); scheduler.step()
  ```
  APIs: `compute_signature_gpu(paths, depth)`, `augment_paths_gpu_paper(paths, config)`, `compute_expected_signature_gpu(paths, depth, config, normalise=True)`, `LSTMGenerator(...).forward(batch_size, n_lags, device)`, `train_sigwgan_paper(generator, expected_sig_real, config, device)`, `evaluate_sig_w1`, `tstr_evaluation_unconditional`, `compute_stylized_facts`; CONFIG keys `n_lags`, `batch_size`, `sig_depth`, `scale_factor`, `scale_dim`, augmentation list `{"name": "Scale", "scale": 2, "dim": 0}`.
- GT-GAN: `GTGAN(input_dim, hidden_dim, latent_dim, noise_dim, ode_method="euler")` with `.encode(x, times)`, `.decode(z, times)`, `.generate(batch_size, times, device)`, `.reparameterize(mu, logvar)`; `GRUODECell(...).forward(x, h, dt_per_sample)`; per-sample RK4:
  ```python
  k1 = f(h); k2 = f(h + 0.5*dt*k1); k3 = f(h + 0.5*dt*k2); k4 = f(h + dt*k3)
  h_new = h + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)   # dt shaped (batch, 1); drift ignores t
  ```
- Diffusion-TS: `DiffusionTS(...)`, `EMA(model, decay=0.995, update_every=10)`, `ema.ema_model.generate_mts(batch_size=N_SYNTHETIC)`, `generate_mts(batch_size, model_kwargs, cond_fn, eta)`, `cycle(dl)`, `extract(a, t, x_shape)`; classifier guidance:
  ```python
  def cond_fn(x, t, **kw):
      with torch.enable_grad():
          x_in = x.detach().requires_grad_(True)
          lp = F.log_softmax(classifier(x_in, t) / temperature, -1)[range(len(x)), target]
          return torch.autograd.grad(lp.sum(), x_in)[0] * scale
  ```
  Variance rescale (normalised space, before `scaler.inverse_transform`): `ratio = syn.std()/train.std(); if ratio < 0.9: syn *= train.std()/syn.std()`.
- GReaT: `from be_great import GReaT` (`uv add be-great`); `GReaT(llm="distilgpt2", batch_size=16, epochs=50, experiment_dir=..., save_steps=5000, logging_steps=100).fit(train_df)`; `great.sample(n_samples=500, max_length=500, guided_sampling=True)`; repeat `TSTR_DRAWS` times and tally `tstr_utility_level`.
- Opacus:
  ```python
  from opacus import PrivacyEngine; from opacus.validators import ModuleValidator
  disc = ModuleValidator.fix(disc)                      # after ModuleValidator.validate(disc, strict=False)
  pe = PrivacyEngine(accountant="rdp")
  disc, opt_d, loader = pe.make_private_with_epsilon(module=disc, optimizer=opt_d, data_loader=loader,
      target_epsilon=10.0, target_delta=1e-5, epochs=20, max_grad_norm=1.0)
  eps_spent = pe.get_epsilon(1e-5)                      # after each epoch; monotone
  for p in disc.parameters(): p.data.clamp_(-0.01, 0.01) # WGAN Lipschitz via weight clipping
  ```
- Collapse diagnostics: `np.corrcoef(sample, rowvar=False)` off-diagonal mean |corr|; `np.linalg.svd((sample - mean) / real_std, compute_uv=False)**2` cumulative share -> dims for 99%.
- Config keys to carry per notebook: 04 `bar_type, seed, seq_length, features, latent_dim, hidden_dim, ode_hidden, max_steps, batch_size, learning_rate, ode_method, holdout_fraction, weights_version`; 05 `seq_length, feature_size, d_model, n_heads, n_layer_enc, n_layer_dec, timesteps, sampling_timesteps, eta, beta_schedule, loss_type, epochs, batch_size, lr, warmup_lr, warmup_steps, ema_decay, gradient_accumulate_every, start_date, holdout_start, classifier_epochs, classifier_lr, guidance_settings, n_regimes`; 06 `symbols, start_date, n_samples, n_generate, epochs, batch_size`; 07 `start_date, n_features, hidden_dim, latent_dim, epochs, batch_size, learning_rate, epsilon, delta, max_grad_norm`.
- Other libraries: `torch`, `torchdiffeq.odeint`, `hmmlearn.hmm.GaussianHMM`, `signatory`, `esig`, `transformers`, `polars` (`pl.DataFrame` tables), `plotly.graph_objects`, matplotlib (`LINE_STYLES` grayscale-safe), sklearn PCA / t-SNE / StandardScaler.

## Evidence from the book
- **Classical population comparison** (200 paths x 504 steps, `00_classical_simulation`): excess kurtosis orders jump-diffusion > GARCH > Heston in every pairwise comparison; jump-diffusion clears GBM's 95th percentile on nearly every path; GARCH > Heston on ~3/4 of path pairs (ranges overlap); the jump model's median excess kurtosis is an order of magnitude above the other two. Mean reversion is indistinguishable from GBM on kurtosis (clears the 95th percentile at chance rate); its narrow price range comes from lower sigma and the restoring drift, not tails. GBM realized excess kurtosis is near 0.
- **Clustering** (sum of 20 squared-return ACFs): GBM, OU and jump-diffusion sit at the null; Heston and GARCH clear it on the large majority of paths, GARCH on every path at ~3x Heston's median. On equal-length windows GARCH's distribution centres slightly above SPY's median and Heston's below; nine SPY windows span most of the models' range (regime differences plus sampling noise; cause not isolated).
- **Bootstrap**: IID keeps the histogram and fat tails with essentially zero clustering; block and stationary at b = 22 each retain roughly two thirds of SPY's squared-return dependence; stationary beats moving block in the majority of paired draws but the distributions overlap heavily.
- **Provider vs from-scratch**: Heston realized annual vols agree up to sampling noise when all four parameters are passed; provider `gbm_jump` skew ~0 vs negative skew from downward-tilted jumps; provider `mean_revert` lines up only because its internal kappa equals `MR_KAPPA`; provider GARCH matches persistence (alpha, beta) but not level unless annual vol is passed; OU Euler vs exact from the same shocks differ only by discretization error.
- **TimeGAN** (6 stocks, log returns, `01_timegan`): synthetic mean and std match real to ~2 decimals, yet PCA shows synthetic sequences in a sliver at the centre of the real cloud (under-dispersed along the main variance directions), t-SNE shows separate regions, and discriminative accuracy is far from chance. The price-level experiment puts many holdout cells outside [0, 1] (count printed).
- **Tail-GAN** (ETF returns, 32 strategies, seed 1, `02_tailgan_tail_risk`): aggregate relative errors VaR 22.5% / ES 21.0%; synthetic VaR and ES means more negative than real (overstates tail severity); PCA/t-SNE show synthetic inside the real cloud; median per-strategy relative error several times the aggregate; share overstated ~1/2; Pearson correlation across strategies near zero; scatter spread on both sides of the 45-degree line; losses plateau well before epoch 3000. Full tail reachable, so the error is distributional mismatch, not clamp truncation. No general-purpose baseline trained, so it shows what targeting achieves, not that alternatives fail.
- **Sig-WGAN** (S&P 500, depth 4, 16-day windows, `03_sigcwgan_signatures`): loss drops rapidly in ~150 steps then plateaus; PCA core overlaps but extreme real days are never reached; both sets symmetric about zero (location right, spread differs); mean error near zero; TSTR ratio near 1 on the return-sign task (expected, near coin flip; paper 1.0; Sig-W1 reference 2.76 not comparable); excess kurtosis falls toward Gaussian, negative skew not reproduced, lag-1 squared-return ACF toward zero.
- **GT-GAN** (short 2000-step run, a few hundred bars, `04_gtgan_irregular`): discriminative accuracy and AUC at their maximum; TSTR MAE ratio outside (0.7, 1.5); bounded fraction clears 70% mostly from the tolerance; PCA/t-SNE show a narrow synthetic band away from real; one plotted window shows real reversing repeatedly while synthetic rises monotonically; losses flatten well before step 2000; KS middling to high is expected at this sample size. Read as architecture demonstration, not achievable quality.
- **Diffusion-TS** (`05_diffusion_ts`): generates less variance than training data in normalised space (rescale applied when ratio < 0.9); eta = 1 made variance worse than eta = 0; unconditional samples already resemble the low-vol regime, so gentle guidance lands near historical volatility; guidance separates regimes but both generated panels range wider than their historical counterparts; paths widen left to right (sampler artifact); at t ~ a quarter of the schedule the residual dominates; neither TSTR classifier reaches the naive baseline, so the ratio establishes nothing about clustering.
- **GReaT** (distilgpt2, 50 epochs, 2,000 rows, 500 per draw, `06_llm_tabular_great`): single-draw TSTR AUC 0.764 vs TRTR 0.736 (ratio 103.8%) was the best of five; five reproducible draws 0.764, 0.720, 0.495, 0.681, 0.750 -> HIGH x3, MODERATE x1, NONE x1, range straddles 0.5 so no utility claim; earlier non-reproducible runs 0.70, 0.30, 0.78; test positive prevalence 5% (predict-never accuracy 0.95); return features worst on KS (spike at zero), synthetic volatility peaks below real, direction nearly one-sided, strongest momentum bucket under-generated; dense synthetic knot in PCA; timings 1,603 / 1,658 / 3,481 s warm with byte-identical outputs (warm runs dominated by draws, fit < 2 s; cold by the fit).
- **DP-GAN** (epsilon 10, delta 1e-5, 20 epochs, `07_dp_gan`): real correlation matrix mostly pale; synthetic saturated +-1 off-diagonal = collapse onto a low-dimensional set, thin PCA bands/arcs; losses stay noisy; epsilon rises monotonically. Sweep [1, 5, 10, 50] at 10 epochs: mean absolute difference several times larger at epsilon = 1 than anywhere else; looser budgets in a narrow band with no ordering; correlation distance non-monotone (rises, falls to a minimum, rises again); not resolvable with one short run per budget.

## Related references
- `chapters/01_process_is_edge.md` — the multiple-testing and adaptive-search argument that motivates synthetic paths.
- `chapters/02_financial_data_universe.md` — `load_etfs()` and the ETF universe feeding notebooks 00, 02, 05, 06, 07.
- `chapters/03_market_microstructure.md` — tick/volume/dollar bars; the NVDA dollar bars GT-GAN consumes and the only legitimate source of irregular timestamps.
- `chapters/09_model_based_features.md` — GARCH, HMM regimes and richer regime models that improve the labels used for diffusion guidance.
- `chapters/11_ml_pipeline.md` — temporal splits, scaler-on-train-only and leakage discipline that the TSTR protocol inherits.
- `chapters/13_dl_time_series.md` — GRU/LSTM/transformer building blocks reused by TimeGAN, GT-GAN and Diffusion-TS.
- `chapters/16_strategy_simulation.md` — backtesting on many synthetic paths instead of one history; Deflated Sharpe and PBO.
- `chapters/19_risk_management.md` — VaR/ES consumers of Tail-GAN and classical tail scenarios.
- `chapters/20_strategy_synthesis.md` — regime-conditioned Diffusion-TS samples used for stress testing.
- `chapters/26_mlops_governance.md` — checkpoint identity, reproducibility and privacy governance of generated data.
- `case_studies/etfs.md` — the ETF data behind the classical, Tail-GAN, Diffusion-TS, GReaT and DP-GAN notebooks.
- `case_studies/nasdaq100_microstructure.md` — Databento bar sampling that produces GT-GAN's input.
- `case_studies/us_equities_panel.md` — the `us_equities` panel behind TimeGAN's six tickers.
- `libraries/ml4t_data.md` — `SyntheticProvider` and the chapter loaders.
- `libraries/ml4t_diagnostic.md` — stylized-fact and signal diagnostics reused when evaluating generators.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `workflow.md`, `companion_repo.md` — cross-cutting indexes that this chapter's items feed.
- Further reading:
  - Yoon, Jarrett & van der Schaar (2019) TimeGAN; Cont, Xu & Zhang (2022/2025) Tail-GAN; Ni et al. (2020/2024) Sig-CWGAN; Jeon et al. (2022) GT-GAN; Yuan & Qiao (2024) Diffusion-TS; Borisov et al. (2023) GReaT; Abadi et al. (2016) DP-SGD; Yousefpour et al. (2021) Opacus.
  - Ho et al. (2020) DDPM; Ho & Salimans (2022) classifier-free guidance; Dhariwal & Nichol (2021); Nichol & Dhariwal (2021) cosine schedule; Song et al. (2020) score-based conditioning; Chen et al. (2018) Neural ODEs; De Brouwer et al. (2019) GRU-ODE; Lienen & Gunnemann (2022) torchode.
  - Politis & Romano (1994) stationary bootstrap; Cont (2001) stylized facts; Bollerslev (1986) GARCH; Heston (1993); Glasserman (2003).
  - Bailey & Lopez de Prado (2014) Deflated Sharpe; Bailey et al. (2015) PBO; Lopez de Prado (2022); Cetingoz & Lehalle (2025); Eckerli & Osterrieder (2021); Gao et al. (2025); Kwon & Lee (2024); Takahashi & Mizuno (2024).
  - Upstream code: `jsyoon0823/TimeGAN`; `SigCGANs/Conditional-Sig-Wasserstein-GANs` (`configs/STOCKS/SigWGAN.json`, `lib/trainers/sig_wgan.py`); `Y-debug-sys/Diffusion-TS`; `kathrinse/be_great`; opacus.ai.

## Glossary
- **Stylized facts** — empirical regularities of returns: fat tails, volatility clustering, leverage effect, negative skew (Cont 2001).
- **Fidelity / utility / privacy** — distributional realism / downstream task performance / resistance to leaking training records; mutually trading off.
- **TSTR / TRTR** — train on synthetic, test on real vs train on real, test on real; ratio of errors (or AUCs), target 1.
- **Discriminative score** — held-out accuracy or AUC of a real-vs-synthetic classifier; 0.5 is the target.
- **Beyond history** — whether a method can emit a value larger than any it was given (parametric: yes; bootstrap: no).
- **Feller condition** — 2 kappa theta > xi^2; the continuous Heston variance stays positive.
- **Full truncation** — max(v, 0) in both drift and diffusion of the Heston Euler step.
- **Drift compensator** — lambda k subtracted so mu is the total expected return in jump-diffusion.
- **Half-life** — ln(2)/kappa for OU reversion.
- **Euler-Maruyama / exact transition** — approximate SDE discretization vs the exact OU step that removes discretization error.
- **Stationary bootstrap** — geometric random block lengths with wrap-around; stationary resample distribution.
- **Clustering statistic** — sum of squared-return autocorrelations over the first 20 lags.
- **Retention ratio** — resample's summed squared-return ACF divided by the original's; 1 = full clustering retained.
- **Dominance share** — fraction of (a, b) path pairs where model a has the higher excess kurtosis.
- **Moment loss** — TimeGAN auxiliary generator loss matching mean and std of real vs synthetic batches.
- **Discriminator gating** — skip the D update while its loss is below 0.15 (TimeGAN).
- **NeuralSort** — differentiable relaxation of sorting with temperature tau.
- **S_quant** — quantile-specific strictly consistent scoring function for the (VaR, ES) pair.
- **W * VaR <= ES projection** — hard constraint keeping discriminator VaR/ES estimates economically consistent.
- **Path signature** — sequence of iterated integrals characterizing a path up to reparameterization.
- **Expected signature** — mean of truncated signatures over a batch of augmented paths; the Sig-WGAN "analytic discriminator".
- **Sig-W1** — L2 distance between expected signatures of two path distributions.
- **Factorial normalization** — multiply signature level k by k!.
- **Lead-lag transform** — path doubling with offset copies; recovers quadratic variation.
- **VisiTrans("I")** — I-visibility transform adding 2 boundary rows and 1 marker column.
- **Brownian noise generator** — cumsum of Gaussian increments (scale 0.1) fed to an LSTM.
- **Information bars** — bars triggered by N trades / V shares / $D traded; inter-arrival time varies with activity.
- **Neural ODE** — hidden state defined by dh/dt = f_theta(h, t), integrated by a solver; queryable at any time.
- **GRU-ODE** — GRU update at observation times, ODE evolution between them.
- **Smoothness ratio** — mean |delta| of ODE interpolation over mean |delta| of real data; ~1 is healthy.
- **Bounded fraction** — share of midpoint interpolations lying within adjacent real values +-0.1.
- **DDPM / DDIM** — diffusion with the full reverse chain / accelerated sampler (eta controls injected noise; 0 = deterministic).
- **x0-prediction** — denoiser outputs the clean sample; noise derived analytically.
- **AdaLayerNorm** — layer norm with scale/shift from the diffusion-timestep embedding.
- **Classifier guidance** — shift the denoising mean by sigma^2 grad_x log p(y|x_t) from a classifier trained on noised data.
- **EMA** — exponential moving average of weights (beta theta_ema + (1 - beta) theta), sampled from for smoother outputs.
- **Variance rescaling** — post-hoc multiplication of normalised samples by train_std / synth_std when the ratio < 0.9.
- **Gaussian HMM** — hidden regimes with state-specific mean/variance; Viterbi decoding of the state path.
- **Serialization (GReaT)** — table rows rendered as sentences for LLM fine-tuning and parsed back.
- **Guided sampling** — be-great sampler that enforces the column schema row by row.
- **Parse count** — number of generated rows whose cell converts to a number for a column.
- **(epsilon, delta)-DP** — output distribution changes by at most e^epsilon (plus delta) when one record changes.
- **DP-SGD** — per-sample gradient clipping to `max_grad_norm` plus calibrated Gaussian noise.
- **RDP accountant** — Renyi-DP moments accountant tracking cumulative epsilon.
- **Noise multiplier** — std of added noise relative to the clipping norm, calibrated by Opacus to the target epsilon.
- **Weight clipping** — clamp discriminator parameters to +-0.01 to enforce the WGAN Lipschitz constraint.
- **Collapse diagnostics** — `mean_abs_offdiag_corr` and `dims_for_99pct_variance`, separating a collapsed generator from a noisy one.
- **KS statistic** — max distance between two empirical CDFs (0 identical, 1 disjoint).
