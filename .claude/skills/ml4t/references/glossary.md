# ML4T Glossary

Cross-cutting vocabulary for *Machine Learning for Trading* (3rd ed., Jansen 2026), its companion repo and the six `ml4t-*` libraries. Compiled from the glossary sections of every chapter, case-study and library reference in this skill; duplicate definitions are merged and every entry carries the source units in a trailing tag so the agent can open the detailed reference. Where a term means different things in different units, the entry lists each meaning with its own tag. Marks such as "(inference)" are carried over from the notes verbatim.

## Source tags

| Tag | File (relative to `references/`) |
|---|---|
| ch01 … ch27 | `chapters/01_process_is_edge.md` … `chapters/27_systematic_edge.md` (number = chapter) |
| cs_cme | `case_studies/cme_futures.md` |
| cs_crypto | `case_studies/crypto_perps_funding.md` |
| cs_etfs | `case_studies/etfs.md` |
| cs_fx | `case_studies/fx_pairs.md` |
| cs_nq100 | `case_studies/nasdaq100_microstructure.md` |
| cs_sp500eo | `case_studies/sp500_equity_option_analytics.md` |
| cs_sp500opt | `case_studies/sp500_options.md` |
| cs_useq | `case_studies/us_equities_panel.md` |
| cs_usfirm | `case_studies/us_firm_characteristics.md` |
| lib_backtest, lib_data, lib_diagnostic, lib_engineer, lib_live, lib_models | `libraries/ml4t_<name>.md` |

## Abbreviations

| Abbrev. | Expansion | Meaning in ML4T | Where |
|---|---|---|---|
| ACI | Adaptive conformal inference | Online update of the miscoverage level α̂_t with step gamma; first 21 dates of a fold run at target alpha (warm-up) | ch11, ch12, ch13 |
| ADV | Average daily volume | Market-native unit (shares, contracts, base asset, FX price updates); dollar ADV for ETF/US-equity eligibility | ch18, cs_etfs, cs_useq |
| ADWIN | Adaptive windowing | ADWIN-style two-window mean-shift detector: alert when adjacent-window means separate by more than pooled SE x sensitivity | ch26 |
| AFML | *Advances in Financial Machine Learning* | Source of information bars, triple barrier, uniqueness, sequential bootstrap, CPCV/PBO, DSR | ch03, ch07, cs_etfs |
| ALS | Alternating least squares | Alternation between factor realizations and the loading map Gamma in IPCA | ch14, lib_models |
| AMM | Automated market maker | On-chain liquidity pool whose pricing can be optimized/arbitraged | ch27 |
| ARI | Adjusted Rand index | Chance-corrected pair agreement of two partitions (random 0, identical 1) | ch01, ch09 |
| ATE / CATE | Average / conditional average treatment effect | DML estimands; subgroup ATE = ATE inside a stratum | ch15 |
| ATR | Average true range | Wilder-smoothed true range (alpha 1/n, n 14); stops and normalization, not a vol estimator | ch07, ch08, ch19, lib_engineer |
| BH | Benjamini-Hochberg | FDR-controlling step-up procedure (`fdr_bh`), alpha 0.05 in the confirmation arm | ch07, ch08, ch15, cs_* |
| BSTS | Bayesian structural time series | Counterfactual from controls' pre-period relation (event studies) | ch15 |
| CAE | Conditional autoencoder | Beta network over characteristics + factor network over managed portfolios | ch14, cs_usfirm, lib_models |
| CDaR | Conditional drawdown at risk | Average of the worst drawdowns on a path | ch17 |
| CPCV / CSCV | Combinatorial purged CV / combinatorially symmetric CV | C(N,k) purged splits assembled into backtest paths / the PBO split scheme | ch06, ch07, ch16, ch17, lib_diagnostic |
| CQR | Conformalized quantile regression | Quantile regressors plus conformal correction; asymmetric, adaptive width | ch11, ch12 |
| CVaR / ES | Conditional VaR / expected shortfall | Mean loss beyond VaR; coherent, subadditive; differentiable via Rockafellar-Uryasev | ch16, ch17, ch19, ch21 |
| DDM | Drift Detection Method | Tracks error rate p and SE s; warns/drifts when p+s exceeds p_min + k s_min | ch26 |
| DML | Double/debiased machine learning | Residualize outcome and treatment on controls, regress residuals; cross-fitted, embargoed | ch15, cs_* |
| DSR | Deflated Sharpe ratio | PSR against the expected maximum Sharpe of N null trials | ch07, ch16, ch20, cs_*, lib_diagnostic |
| EWMA | Exponentially weighted moving average | RiskMetrics variance (lambda 0.94); E[T] bar estimation; weight EMA in diffusion models | ch19, lib_engineer, ch05 |
| FDR / FWER | False discovery rate / family-wise error rate | Expected share of false rejections (BH) / P(at least one false rejection) (Holm) | ch07 |
| FFD | Fixed-width fractional differencing | (1-L)^d weights truncated at `threshold`; d fixed in advance (0.4/0.5) | ch09, cs_etfs, cs_useq, lib_engineer |
| GARCH / GJR-GARCH | (Glosten-Jagannathan-Runkle) generalized ARCH | sigma^2_t = omega + alpha eps^2 + beta sigma^2 (+ gamma for negative shocks, `o: 1`) | ch09, cs_crypto, cs_etfs, cs_sp500eo, cs_sp500opt |
| GBM | (1) Gradient boosting machine; (2) geometric Brownian motion | (1) registry family `gbm` (LightGBM/CatBoost); (2) driftless GBM is the reference for vol-estimator efficiency | ch12, ch26; ch08 |
| GMM | Gaussian mixture model | K multivariate normals fitted by EM with per-point membership probabilities | ch01 |
| GNN / GAT | Graph neural network / graph attention network | GAT used as a single-head dense autoencoder on the holdings graph | ch23 |
| HAC | Heteroskedasticity-and-autocorrelation-consistent | Newey-West standard errors with Bartlett weights; lag spans label overlap | ch04, ch07, ch08, ch10-ch14, cs_*, lib_diagnostic |
| HAR | Heterogeneous autoregression | Realized variance on 1/5/22-period components (bars [5, 15, 60] intraday) | ch09, cs_nq100 |
| HHI | Herfindahl-Hirschman index | Sum of squared weights/shares; reciprocal = effective positions / names / forecasters | ch17, ch22, ch23, ch24 |
| HMM | Hidden Markov model | Latent regimes with transition structure; filtered (causal) vs smoothed (non-causal) posteriors | ch01, ch05, ch09, cs_* |
| HPO | Hyperparameter optimization | Walk-forward HPO with Optuna (TPE, MedianPruner, NSGA-II) | ch12 |
| HRP | Hierarchical risk parity | Correlation-distance clustering, quasi-diagonalization, recursive bisection; never inverts | ch17 |
| HTM | Hold-to-maturity | Option label/strategy with no exit-side trade; makes the spread a one-sided entry cost | ch20, cs_sp500opt |
| IC | Information coefficient | Per-date cross-sectional Spearman between signal and forward return, averaged over dates | ch04, ch07, ch08, ch10-ch14, ch23, ch26, cs_*, lib_diagnostic |
| ICIR / IR | IC information ratio / information ratio | mean IC / std IC (annualized by sqrt(252) in lib_diagnostic); IR also = mean active return / tracking error | ch07, ch08, ch11, lib_diagnostic, ch17 |
| IPCA | Instrumented PCA | Loadings linear in lagged characteristics, Gamma estimated by ALS | ch14, cs_etfs, cs_sp500eo, cs_useq, cs_usfirm, lib_models |
| IRL | Inverse reinforcement learning | MaxEnt IRL recovers reward weights from demonstrations | ch21 |
| ITCH | Nasdaq message-by-order protocol | MBO/L3 feed; message types R, D, X, E, C, U, P, Q; `stock_locate`, `tracking_number`, `price4` | ch03 |
| KG | Knowledge graph | Entities as nodes, typed edges, provenance as properties (Neo4j property graph) | ch23 |
| KS | Kolmogorov-Smirnov | Max gap between two empirical CDFs (0 identical, 1 disjoint); two-sample p-value | ch05, ch09, ch26 |
| LOB | Limit order book | Reconstructed from MBO via an order pool `order_ref -> (side, price, remaining)` | ch03 |
| MAE / MFE | Maximum adverse / favorable excursion | Worst / best move from entry over [t+1, t+H]; edge ratio = avg MFE / avg \|MAE\| | ch07, ch16, ch19, cs_etfs, cs_usfirm, lib_backtest, lib_diagnostic |
| MBO / L2 / L1 / TOB | Market-by-order / level 2 / level 1 / top of book | Order-id feed / price-level depth / best bid-ask only | ch03 |
| MDI / PFI | Mean decrease in impurity / permutation feature importance | Tree importance measures; impurity importance also used for mixture-of-experts features | ch08, ch09, ch12 |
| MDP | Markov decision process | Finite-horizon execution MDP: state (time, inventory, lagged regimes); tabular Q-learning | ch18, ch21 |
| MinTRL | Minimum track record length | Periods needed for an observed Sharpe to clear a threshold; `never` when unattainable | ch07, ch16, cs_crypto, lib_diagnostic |
| MNPI | Material non-public information | Information supplied by someone not entitled to supply it; a hard gate | ch04 |
| MVO | Mean-variance optimization | Markowitz; amplifies estimation error through covariance inversion ("Markowitz curse") | ch17 |
| NBBO | National best bid/offer | What a marketable order meets; TAQ = trades + NBBO, no depth | ch03 |
| NDCG@k | Normalized discounted cumulative gain | Ranking metric over the top-k of a query group (LambdaMART) | ch12 |
| OFI | Order flow imbalance | Message-based (adds minus removes per side) or aggressor-based (buy vol - sell vol)/total in [-1, 1] | ch03, ch08 |
| OU | Ornstein-Uhlenbeck | Mean-reverting process; half-life ln2/kappa (continuous) or -ln2/ln\|phi\| (AR(1)) | ch05, ch09, ch14 |
| PBO | Probability of backtest overfitting | Share of CSCV paths where the IS-best configuration ranks in the bottom half OOS | ch06, ch07, ch16, cs_cme, cs_sp500eo, lib_diagnostic |
| PCMCI | PC1 + momentary conditional independence | Lagged causal discovery; `pc_alpha` is PC1 regularization, not FDR; ParCorr CI test | ch15 |
| PIT | Point-in-time | Only information public on the simulated date; bitemporal rows (valid time, knowledge time) | ch01, ch02, ch04, ch22, ch23, ch26, lib_data, lib_engineer |
| PPO / DQN / SAC | Proximal Policy Optimization / Deep Q-Network / Soft Actor-Critic | Deep RL algorithm names; not defined in the extracted notes (ch21 covers tabular Q-learning, QLBS, behaviour cloning, MaxEnt IRL) | ch21 |
| PSI | Population stability index | Sum over bins of (cur% - ref%) ln(cur%/ref%); bands 0.10 / 0.25; no null distribution | ch02, ch09, ch19, ch26, lib_diagnostic |
| PSR | Probabilistic Sharpe ratio | P(true SR > benchmark) given sample length, skew, kurtosis | ch16, cs_crypto, cs_etfs, cs_usfirm, lib_diagnostic |
| RAG | Retrieval-augmented generation | Index documents, retrieve evidence, generate a grounded answer; Graph RAG selects evidence by query | ch22, ch23 |
| RAS | Rademacher Anti-Serum | Correlation-aware lower bound theta_hat - 2 R_hat - 2 kappa sqrt(log(2/delta)/T) for parameter snooping | ch07, ch08, ch16, cs_crypto, lib_diagnostic |
| RP-PCA | Risk-premium PCA | PCA on Sigma + kappa rbar rbar^T (gamma weight) to reward priced directions | ch14, lib_models |
| RTH | Regular trading hours | US equities 09:30-16:00 New York | ch03, ch25 |
| SAE | Supervised autoencoder | Reconstruction + aux + main heads sharing one bottleneck; the latent-factor family's control | ch14, cs_usfirm, lib_models |
| SDF | Stochastic discount factor / pricing kernel | M = 1 - omega^T R^e; adversarial moment network picks worst-priced test assets | ch14, cs_cme, cs_usfirm, lib_models |
| SHAP | Shapley additive explanations | TreeSHAP O(T L D^2); linear phi_j = beta_j (x_j - x̄_j); Trade-SHAP clusters losing trades | ch11, ch12, ch19, lib_diagnostic |
| TAQ | Trades and quotes | Consolidated trades plus NBBO, no depth | ch03 |
| TCA | Transaction cost analysis | Scoring realized fills against benchmarks after the fact | ch18 |
| TCN | Temporal convolutional network | Causal dilated convolutions with weight normalisation; receptive field R = 1 + sum 2(k-1)d_i | ch13 |
| TFT | Temporal Fusion Transformer | Its GRN / GLU / VSN blocks appear in the ch17 deep allocators (VLSTM, DeePM) (inference: the notes name the blocks, not TFT) | ch17 |
| TIB / VIB | Tick / volume imbalance bars | Close when \|signed ticks\| or \|signed volume\| exceeds E[theta_T] | ch03, lib_engineer |
| TPE | Tree-structured Parzen Estimator | Optuna sampler modelling densities of good vs poor hyperparameters | ch12 |
| TSMixer | Time-series mixer | Alternating time-mixing and feature-mixing MLPs with pre-normalisation | ch13, cs_useq |
| TSTR / TRTR | Train on synthetic, test on real / train on real, test on real | Ratio of errors (or AUCs), target 1 | ch05 |
| TVL | Total value locked | USD value deposited in a chain's DeFi protocols; a stock, price-denominated | ch04 |
| TWAP / VWAP | Time- / volume-weighted average price | Equal shares per interval / shares proportional to expected volume; schedules and benchmarks | ch18, ch21 |
| VaR | Value at risk | Loss not exceeded with probability a over one day; `-q_{1-a}` of returns | ch09, ch16, ch17, ch19 |
| VIF | Variance inflation factor | > 5 concern, > 10 severe | ch11 |
| VRP | Variance / volatility risk premium | Implied minus realized (IV30 - RV20; `vrp_21d = iv_atm - rv_21d`); a state; the DML treatment in options | ch08, cs_sp500eo, cs_sp500opt |
| XBRL | eXtensible Business Reporting Language | Machine-readable us-gaap tagging; Frames API returns one concept for all filers per period | ch04 |

## Glossary

### A

- **Abstention** — RAG: refusing when retrieved context is insufficient; scored by `abstention_correct` / F1, in ch22 nb08 a cost column. Agents: `synthesis_status="abstained"` when a quality gate fails — (ch22, ch24)
- **Accession number** — unique, permanent SEC filing id (leading block = filing agent); a filing's true identity across filers and time — (ch04, ch22)
- **Acceptance datetime** — when EDGAR accepted a filing; after-close acceptance means tradeable next session — (ch04)
- **Accountability mechanism** — external commitment that raises follow-through — (ch27)
- **Accumulation bound / ulp** — tolerance = steps x machine epsilon for a recursive computation; float64 eps = 2.22e-16 — (ch02)
- **ACF / PACF, correlogram, Ljung-Box, ARCH-LM, Jarque-Bera (JB)** — autocorrelation functions and the joint tests on them; ARCH-LM regresses squared returns on lags; JB tests normality from skew and kurtosis (lower = nearer normal) — (ch03, ch09, ch11)
- **Action injection** — retrieved text attempting to trigger a system operation (e.g. `transfer_funds`) — (ch22)
- **Activation frequency / state-transition frequency** — share of symbol-date rows active / share where state differs from the prior row; neither is turnover — (ch16)
- **AdaLayerNorm** — layer norm whose scale/shift come from the diffusion-timestep embedding — (ch05)
- **Added paragraphs** — paragraphs in the newer 10-K not verbatim in the older — (ch04)
- **ADF, KPSS, Phillips-Perron** — unit-root tests; ADF and PP null = unit root, KPSS null = stationary — (ch09)
- **Adjustment (prices)** — backward adjustment scales history so the latest price is unchanged; total-return adjusted = split + dividend (reinvested holding). `ml4t-data` modes `raw | split | dividend | all`; adjusted prices use the latest observation's share basis. Case studies keep adjusted close for returns and traded/raw close for dollars, costs and curve quantities (`adj_close` / `raw_close` / `cum_ratio`, `sec_id` / `adj_factor`) — (ch02, lib_data, cs_etfs, cs_cme, cs_sp500eo)
- **Adverse selection** — the counterparty knows something you do not; price keeps moving after the fill (being filled on the side the market moves away from) — (ch18, ch21)
- **Aggressor** — side that crossed the spread — (ch03)
- **AIC / BIC** — log-likelihood penalised by 2 x #params / #params x log n; lower is better — (ch01)
- **Ablation** — same architecture, data, seed and initial weights with one mechanism switched off — (ch17)
- **Allocator / moment-based allocator** — sizing rule applied to selected positions (inverse_vol, risk_parity, hrp, mvo / mvo_ledoit_wolf read `allocator_lookback` bars: warm-up 240 in crypto, 63 in CME; score / conformal allocators 0; `score_weighted`, `conformal_weighted`). Equal weight is the baseline, not an allocator; covariance-based allocators are skipped intraday as "expensive" — (cs_crypto, cs_cme, cs_etfs, cs_usfirm, cs_nq100)
- **Allocator term sheet** — documented specification of an allocator's inputs, windows, constraints and rebalancing — (ch17)
- **Almgren-Chriss** — static execution schedule minimising expected impact + lambda x variance; sinh decay with kappa^2 = lambda sigma^2 / eta; used as the oracle benchmark — (ch21)
- **Alpha / beta / tracking error / information ratio** — annualized intercept / slope on the benchmark / vol of active returns / mean active return over TE — (ch17)
- **alpha_frac / alpha_max** — Lasso/ENet penalty expressed as a fraction of the fold's `alpha_max`, the smallest penalty that zeros every coefficient (max\|X^T y\|/n) — (ch11, cs_cme, cs_crypto, cs_etfs)
- **Alpha factory** — organization/process that industrializes hypothesis generation, testing and rejection — (ch27)
- **Amihud illiquidity** — mean \|r\| per million dollars traded; intraday: window average of \|return\| / dollar volume — (ch08, cs_nq100)
- **Anchor** — price/time at which a label's return measurement starts (close vs next open) — (ch07)
- **Antithetic paths** — mirror-image shock pairs so the sample mean shock is exactly zero — (ch18)
- **Any-N-in-lookback rule** — retraining trigger counting alerts anywhere in the last N checks — (ch19)
- **Answer key leakage** — a policy reading `case.answerable` — (ch22)
- **AR / AAR / CAR / CAAR** — abnormal return, average AR per relative day, cumulative AR per event, cumulative average AR; tested with BMP (event-induced-variance robust) and Corrado (rank) tests — (ch08)
- **Arrival price** — price on screen at the decision moment; the implementation-shortfall benchmark — (ch18, ch21)
- **Artifact manifest / manifest** — versioned index of result components in a Parquet directory governing validated loading and recovery; MLOps: table mapping a run to artifact paths with existence flags — (lib_backtest, ch26)
- **ArtifactSpec / FeatureSpec / LabelSpec / PredictionSpec / DataContractConfig** — persisted-artifact contracts (schema + definition) shared across the libraries; canonical column-name mapping — (lib_engineer)
- **As-of join (backward)** — for each left row take the most recent right row with key <= left key (`join_asof(strategy="backward")`); aligns slow data (fundamentals, macro, funding) to daily rows using the last value announced at or before each date. **join_asof forward** aligns to the next existing date >= target, the PIT-safe direction for filing dates — (ch02, ch04, ch07, ch08, ch16, ch25, ch26, lib_engineer, ch10)
- **As-of query** — filter knowledge time <= date, rank on valid time, take latest per entity — (ch04)
- **Associative scan** — parallel evaluation of an affine recurrence whose per-step coefficients do not depend on prior state — (ch13)
- **ATR-adjusted barrier** — barrier distance = multiple x ATR(`atr_period`), adapting to volatility — (lib_engineer)
- **Audit journal** — hash-chained JSONL of runtime / order events — (lib_live)
- **Availability clock / index / timestamp** — bar-open timestamp advanced one bar so it names when the bar is known; row index at which a forward return becomes known (`t + horizon`), driving PIT thresholds; 13F: max filing date across included managers (`available_from`) — (cs_crypto, lib_engineer, ch22, ch23)

### B

- **Backcast / forecast (N-BEATS)** — a block's input reconstruction (subtracted before the next block) and its horizon prediction; doubly residual stacking = input minus backcasts, forecasts summed; basis expansion = learned coefficients times fixed polynomial/Fourier vectors — (ch13)
- **Back-adjusted futures / Panama vs ratio adjustment / roll event** — stitched continuous series so rolls do not appear as jumps; additive (Panama) adds the gap to prior history, ratio multiplies; a roll pairs same-date closes; `cum_ratio` change marks a roll; no-rollback constraint; handover window — (ch02, ch18, lib_data, cs_cme)
- **Backdoor criterion / adjustment set** — rule choosing observed variables that block non-causal paths from treatment to outcome (here `{return_24h, volatility_24h}`) — (ch15)
- **Backtest protocol** — written spec of signal timing, execution, rebalancing, sizing, costs, constraints, data availability and benchmark — (ch16)
- **BacktestProfile / availability / provenance** — lazy analytics over six backtest surfaces; per-surface computability flags; native vs reconstructed data warnings — (lib_diagnostic)
- **`backtest_paired_metrics`** — table of paired-bootstrap comparisons (difference, CI95, p) between two existing return series — (cs_sp500opt, cs_useq)
- **Bad control** — a control affected by the treatment — (ch15)
- **Bad day** — session whose directional error rate exceeds max(calibration mean + 0.01, 0.52) — (ch26)
- **Bagging / boosting** — averaging independently trained trees (RF; reduces variance) / sequential fitting to residuals (reduces bias) — (ch12)
- **Balanced accuracy** — mean per-class recall — (ch15)
- **Bandwidth L (HAC)** — lag truncation; horizon-aware rule L = max(h-1, floor(4 (T/100)^(2/9))); `maxlags = label horizon` (5) in sp500eo; bandwidth = horizon in us_equities; 3 settlements in crypto; lags 5 in ch19 — (ch07, cs_cme, cs_sp500eo, cs_useq, cs_crypto, ch19)
- **Bar type** — time / tick / volume / dollar / information-driven bars (N trades, V shares, $D notional; inter-arrival varies with activity); non-time bars carry `bar_params.threshold` — (ch03, ch05, lib_data, lib_engineer)
- **Base rate** — historical event frequency; required evidence type against base-rate neglect — (ch24)
- **Baseline (equal-weight) backtest** — 1/N top-k long-short book registered with `stage='signal'`; the measuring instrument every overlay is compared to — (cs_crypto, cs_fx, cs_etfs, ch26)
- **Baseline checkpoint** — narrow initial setup validated with timing, coverage and trading-intensity checks — (ch06)
- **Basis point** — 0.01%; the unit of nearly every cost — (ch18)
- **BBP edge** — Baik-Ben Arous-Peche phase-transition threshold separating signal eigenvalues from noise — (ch14)
- **Behaviour cloning** — supervised regression state -> action; no reward; suffers distribution shift — (ch21)
- **Benchmark circularity** — a label rule sharing a candidate's mechanism measures similarity to the rule, not quality — (ch22)
- **`benchmark_kind`** — label of a paired-comparison type (`equal_weight_holdout_side_artifact`, `val_rank1_self`, `signal_leader`, `allocation_leader`, `risk_overlay_leader`) — (ch20, cs_cme)
- **Beta network head** — predictive head regressing returns on SDF exposure (`StochasticDiscountFactorBetaNetworkHead`) — (lib_models)
- **Beyond history** — whether a generator can emit a value larger than any it was given (parametric yes, bootstrap no) — (ch05)
- **Bid-ask bounce** — alternation of prints between bid and ask inducing negative AC(1) of trade-price returns (Roll's premise); motivates skip-1 momentum — (ch03, ch18, ch08)
- **BIO tagging / entity-level F1** — Begin/Inside/Outside span labels; a span is correct only if extent and type both match (`seqeval`) — (ch10)
- **Bipartite density** — share of possible institution-stock edges that exist — (ch22)
- **Bitemporal** — a row carrying valid time (period described) and knowledge time (when it became public); event time vs knowledge time — (ch02, ch04)
- **Block / stationary / moving-block bootstrap** — resample contiguous blocks (fixed or geometric length, wrap-around) to keep dependence; 20-row blocks for edge stability; used for IC CIs, Sharpe-difference CIs and Reality Check — (ch05, ch07, ch14, ch15, lib_diagnostic)
- **Block permutation / block placebo / within-time permutation** — permute labels or treatment in contiguous blocks (>= label horizon or treatment window; 12 months, 252 days, 20 rows) within entity to build a serial-dependence-preserving null; studentized version compares t-statistics; plus-one p = (1 + extreme)/(1 + n); <= 19 draws is "Underpowered" — (ch07, ch15, cs_cme, cs_etfs, cs_fx, cs_nq100, cs_usfirm)
- **Blocked (relative-time) CV** — contiguous time-block folds so neighbouring rows never sit in opposite folds — (ch19)
- **BLUE** — best linear unbiased estimator; OLS under linearity, strict exogeneity, no perfect multicollinearity, spherical errors (constant variance, no cross-observation correlation) — (ch11)
- **BM25** — bag-of-words relevance: term frequency saturated by k1, IDF-weighted, length-normalised by b — (ch22)
- **Book pressure** — side x action-weight x size x exp(-lambda x distance), EMA-smoothed — (ch03)
- **Borrow cost** — annual fee on short-side notional; a running cost — (ch18)
- **Boundary leak** — extracted 10-K section containing the next item's heading — (ch04)
- **BoundedEventQueue** — fail-closed buffer raising `FeedOverflowError` on overflow — (lib_live)
- **Breadth** — (a) assets quoted/eligible on a decision date, with floors: CME 20 products, crypto 2 x largest top-k, ETFs `max(top_k_grid)` = 20, US equities >= 100 (2 x top_k 50); (b) number of managers/holders of a stock (13F); (c) number of case studies in which a feature family appears (recurring >= 2); (d) fraction of assets above a threshold; (e) BR in the Fundamental Law = number of independent bets — (cs_cme, cs_crypto, cs_etfs, cs_useq, ch10; ch04, ch22, ch23; ch08; ch09; ch08)
- **Break-even cost / breakeven alpha** — round-trip (or per-leg) cost at which the quantile spread, net Sharpe or cost-sweep Sharpe reaches zero (interpolated zero crossing; censored if still positive at the grid ceiling); breakeven cost rate = gross reference-price P&L / one-way reference notional; breakeven alpha = gross return needed to pay trading cost at a turnover — (ch07, ch16, ch18, ch20, cs_crypto, cs_etfs, cs_fx, cs_nq100, cs_useq)
- **Breusch-Pagan** — heteroscedasticity test on residuals — (ch11)
- **Brier / log score / ECE / sharpness** — mean squared probability error (proper, bounded) / negative log-likelihood (unbounded on confident error) / count-weighted binned calibration gap / mean \|p - 0.5\| (never a quality alone) — (ch24)
- **Brownian noise generator** — cumsum of Gaussian increments (scale 0.1) fed to an LSTM — (ch05)
- **BSTS / spillover / cumulative abnormal log return** — Bayesian structural time-series counterfactual from controls' pre-period relation / a control that itself responds to the event / sum of post-period point effects on daily log returns — (ch15)
- **Burn-in** — sessions a fitted feature spends before its first estimate (500 crypto, 252 options); carries no value; CUSUM/MOSUM reference window — (ch09, cs_crypto, cs_fx, cs_sp500opt, cs_useq)
- **Buying-power reservation / submission precheck** — cash reserved at submission (LEAN parity) / pseudo-execution check at submit (Backtrader parity) — (lib_backtest)

### C

- **C (logistic regression)** — inverse regularization strength — (ch11)
- **Cadence / release clock** — publication frequency of a series, recovered from value changes — (ch04)
- **Calendar-day vs trading grid** — rows for every date vs sessions only — (ch04)
- **CalendarMode** — `"equity"` (exchange sessions) vs `"continuous"` (24/7) for synthetic timestamps — (lib_data)
- **Calibration vs discrimination** — stated probabilities match observed frequencies vs ranking ability (AUC); reliability bins/diagram: equal-width probability bands with count, mean forecast and observed yes share (diagonal = calibrated) — (ch11, ch24)
- **Calmar / Sortino / Omega / tail ratio** — CAGR over \|max drawdown\| / excess return over downside deviation / probability-weighted gains over losses / largest gains vs worst losses — (ch17, ch19)
- **Candidate set / cohort** — named, immutable, hashed set of backtest identities (or `(label, family)` prediction identities) admitted to a comparison, sealed before ranking (`crypto-signal-{label}`, `cme_futures-<stage>-<label>-v1`) — (cs_cme, cs_crypto, cs_fx, cs_sp500opt, cs_useq)
- **Canonical JSON** — deterministic serialization (stable key order/bytes) so equal specs hash equal — (ch26)
- **Canonical key / entity resolution / filer roster** — normalized join key (case, punctuation, suffixes, word order removed) mapping surface names to one node; corpus-derived map of official registrant names — (ch23)
- **Canonical OHLCV schema** — `[timestamp tz-aware, symbol upper, open, high, low, close, volume Float64]`, sorted, unique; with adjustment basis, seam, rescale factor, source attribution, fallback fetch — (ch02)
- **Cantelli inequality** — distribution-free one-sided tail bound 1/(1+k^2) — (ch19)
- **Capacity** — AUM at which own impact consumes the gross return — (ch18)
- **Capped-simplex projection / effective names** — map nonnegative scores to weights summing to one with each <= cap; 1/HHI of the weight vector, maximal (= N) for equal weight — (ch23)
- **Carried values** — aggregate, post-debate blend, final: the same quantity at three pipeline points — (ch24)
- **Carrier / solvent carrier / spine / parent strategy** — the validation rank-1 configuration (spec, `prediction_hash`, drawdown) to which cost, risk and holdout stages are pinned; resolved by `resolve_solvent_carrier`, refused if ruined (`max_drawdown <= -1.0`, `_ruined`; inference: equity never reached zero); re-ranked on common support when conformal candidates abstain. Carrier (feature): the thesis feature (`signed_vol_share_15m`) — (ch20, cs_cme, cs_crypto, cs_etfs, cs_fx, cs_nq100, cs_sp500opt, cs_useq, cs_usfirm)
- **Carry / roll yield / `carry_pct`** — annualized near-minus-far futures spread; positive in backwardation (rolling earns), negative in contango; CME: `(c0 - c1)/c0 * 12` (x12 is a scale factor, not an annual rate); index carry ~ index x (financing - dividend yield) x time — (ch02, ch08, cs_cme)
- **Carry-forward feed** — bars repeating the last quote when nothing traded — (ch04)
- **Cash buffer** — fraction of targets held back as cash to pay commissions — (ch16, lib_backtest)
- **cash_precheck** — hold cash at the hurdle when no expected excess return is positive — (ch17)
- **Causal convolution / dilation / receptive field** — output at t depends only on inputs <= t (trim (k-1)d right-hand outputs); dilation doubles per layer; R = 1 + sum 2(k-1)d_i — (ch13)
- **Causal sufficiency / Markov equivalence class** — no unmeasured common causes (assumed by PC/PCMCI) / DAGs with identical conditional independences — (ch15)
- **Channel independence / RevIN** — each feature passes through shared weights as its own univariate series / reversible instance normalisation: per-sample per-channel mean/SD removed before the model and restored after the head — (ch13)
- **CheckRules** — FinReflectKG-style schema rules embedded in the extraction prompt and re-applied as a validator — (ch23)
- **Checkpoint** — a scoreable model state (GBM tree count, NN/DL epoch, SDF `(phase, epoch)`) registered as its own prediction set and part of the configuration identity (crypto: gbm 10, tabular_dl 8, deep_learning 20, linear 1; solved estimators publish checkpoint 0). Checkpoint grid (lib_models): epochs at `checkpoint_interval` multiples + `checkpoint_epochs` + final. Agents: JSON serialization of `AgentState` for resume/diff/ablation — (ch12, cs_cme, cs_crypto, cs_etfs, cs_fx, cs_nq100, cs_sp500eo, cs_sp500opt, cs_useq, cs_usfirm, lib_models, ch24)
- **Child / parent order** — slices of a large order worked over time — (ch18)
- **Choppiness Index** — 0-100; > 61.8 choppy, < 38.2 trending — (lib_engineer)
- **Chronos / TTM** — Amazon T5 encoder-decoder over quantised time-series tokens / IBM Granite TinyTimeMixer 1-5M-parameter MLP-mixer forecaster; zero-shot forecasting baselines — (ch13)
- **Chunking / sentence-window chunking** — splitting long text into ~512-token pieces for transformer scoring, then aggregating; sentence-aligned chunks with overlap (`SentenceSplitter`) — (ch10, ch22)
- **CI status / Gate 1 / Gate 2** — three-tier classification (wholly above zero, spans zero, wholly below zero); Gate 1 = validation Sharpe CI lower bound >= 0; Gate 2 = holdout-vs-EW paired CI not wholly below zero (inference from helper names); kill gate = CI-bound go/no-go — (cs_fx, cs_sp500opt, cs_cme)
- **CIK / LEI / FIGI / CUSIP / ISIN** — SEC Central Index Key (permanent filer / 13F manager id) / legal-entity / instrument / security identifiers (CUSIP = stock key in 13F) — (ch04, ch22)
- **Circuit breaker** — (a) sizing: drawdown-zoned multiplier on new-entry sizing; (b) MLOps: rule that switches a system off when a measurement crosses a limit, with closed / open / half-open states and a recovery timeout; (c) data: closed -> open after 5 counted `NetworkError`s -> half-open after 300 s — (ch16, ch26, lib_data)
- **Citation accuracy / failure rate** — share of cited chunk ids present in (absent from) the retrieved set — (ch22)
- **Classifier guidance** — shift the denoising mean by sigma^2 grad_x log p(y\|x_t) from a classifier trained on noised data — (ch05)
- **Close-to-close trade accounting** — entry at close t earns t->t+1; exit at close u earns nothing beyond u — (ch19)
- **Cluster bootstrap / percentile interval / test inversion** — resample whole clusters (symbols, portfolios) with replacement; CI from the 2.5th/97.5th percentiles of replicates; p = smallest alpha whose interval excludes zero — (ch10)
- **Clustering statistic / retention ratio** — sum of squared-return autocorrelations over the first 20 lags; resample's value over the original's (1 = full clustering retained) — (ch05)
- **Cointegration / Engle-Granger / Johansen / spread half-life** — stationary linear combination; residual unit-root test; system eigenvalue test; -ln 2 / ln(1 + phi) — (ch09)
- **Collapse diagnostics** — `mean_abs_offdiag_corr` and `dims_for_99pct_variance`, separating a collapsed generator from a noisy one — (ch05)
- **Common support / common-support re-ranking** — the exact timestamp intersection every candidate prices; Sharpe recomputed over it before ranking (required when conformal candidates abstain); paired differences are computed on it — (cs_cme, cs_crypto, cs_fx, cs_usfirm)
- **Complete (registry) / complete result / complete-coverage candidate** — registry flag: written-key digest equals declared-eligible digest with no bad scores; row with no null fold IC and maximum validation-day coverage; run covering the same folds and day count as the comparison set — (cs_crypto, ch11, ch13)
- **Complete quarter / complete period / public period** — reporting period for which every included institution has filed; public period = complete period whose last filing arrived on or before the cutoff — (ch22, ch23)
- **Composite / interaction (features)** — equal-weight mean of member ranks, or weighted sum of rolling z-scores; product of two centred ranks (sign encodes agreement) — (cs_usfirm, lib_engineer)
- **Condition number kappa(Sigma)** — largest/smallest eigenvalue; bounds error amplification under inversion; grows with redundancy; lowered by Ledoit-Wolf — (ch17, ch19)
- **Condition 80000002** — AlgoSeek late-reported trade flag — (ch03)
- **Confounder / mediator / collider** — common cause / intermediate on the causal path / common effect; conditioning on a collider opens a spurious path — (ch07)
- **Conformal prediction** — *Split conformal*: fixed half-width from a quantile of held-out \|residuals\| (nonconformity score), distribution-free under exchangeability (calibration and test residuals interchangeable). *CQR*: quantile regressors + conformal correction (asymmetric). *ACI*: online alpha update. *Normalized conformal*: residuals scaled by sigma-hat. *Mondrian*: per-entity calibration; *horizon embargo* excludes residuals whose labels were unobservable at fold start; *inverse-width weighting* w_i ∝ 1/Delta_i. *Coverage* = share of outcomes inside the interval (marginal vs conditional/per-stratum; coverage gap = actual - target; regime-conditional calibration per ex-ante state). Case studies: conformal width = per-contract quantile of past absolute errors, `1/width` sizing, alpha 0.2, sizing lag h = max(1, horizon), `calibration_version="walk_forward_v3"` — (ch11, ch12, ch13, ch17, cs_cme, cs_crypto, cs_usfirm)
- **Constant-maturity series / same-contract vs held-contract return** — daily reselection near a target maturity/delta; return on one contract vs yesterday's contract repriced today — (ch02)
- **Content-addressed identifier / spec hash** — registry key that is the hash of the canonical specification (training, prediction, backtest); changes on refit; cash, share_type, device and preset values are inputs; invalidated by any `execution.*` change; re-keyed between `v3.0.0` and `v3.1.0` bundles — (ch26, cs_etfs, cs_sp500eo, cs_sp500opt, cs_useq)
- **Context window (word2vec)** — words on each side counted as neighbors — (ch10)
- **Contract qualification / IDEALPRO** — resolving a case-study symbol to an IB `Forex` contract on IB's interbank FX desk (1k base-unit step on majors) — (ch25)
- **ContractSpec** — static futures metadata: multiplier (point value), tick size, margin — (ch16)
- **Contrarian / change signal (COT)** — fade an extreme z-score / react to a large weekly move — (ch04)
- **Controller / tool / memory** — agentic pattern in which RAG is one tool — (ch22)
- **Conviction / extreme / near_certain** — \|p - 0.5\| / p outside [0.2, 0.8] for prediction-market prices — (ch04)
- **Co-ownership similarity** — cosine between two stocks' columns of the normalized institution-by-stock matrix — (ch22)
- **Cophenetic correlation** — correlation between dendrogram join heights and original pairwise distances (conventional bar 0.7) — (ch01)
- **Cornish-Fisher** — skew/kurtosis correction to the Gaussian quantile (truncated expansion) — (ch19)
- **Correctness screens** — coverage, staleness, timing/lag consistency, mask alignment — (ch07)
- **Correlation distance / MST / centralities** — d = sqrt(2(1 - rho)), a metric on [0, 2]; minimum spanning tree of N-1 edges; degree / betweenness / closeness centrality — (ch23)
- **Correlation-PCA** — PCA on standardized (unit-variance) series — (ch14)
- **Corwin-Schultz / Roll spread** — high-low-range spread estimator / serial-covariance estimator 2 sqrt(-Cov(dP_t, dP_{t-1})) — (ch08, ch18)
- **Cost cascade rungs** — O'Donovan-Yu option cost-mitigation levels: naive round trip; HTM (rung 2, full universe); HTM on the liquid bottom-spread quintile (rung 3) — (ch20, cs_sp500opt)
- **Cost class** — dominant (costs first-order) vs material (costs matter but do not preclude trading) — (ch06)
- **Cost cliff** — turnover level where execution-cost drag consumes gross return — (ch18)
- **Cost drag** — `(fees + slippage) / notional` per trade; IR_tc = IR after costs — (lib_backtest, lib_diagnostic)
- **Cost-feasible universe / friction floor** — top-50 cheapest-to-trade names by the round-trip proxy, frozen per split; `friction_floor_bps: 5` is the optimistic end — (cs_nq100)
- **Cost grid / cost sensitivity / cost level** — post-hoc Sharpe recomputation at several cost levels on fixed weights (`cost_grid_bps`); round-trip bps split half commission half slippage; frozen and excluded from selection — (ch17, cs_cme, cs_crypto, cs_fx)
- **Cost margin ratio / uplift breadth** — breakeven / assumed cost (headroom; drives the resilience label) / share of non-EW allocators beating equal weight (broad / narrow / none) — (ch20)
- **Cost recovery / target bps** — return that pays an alternative dataset's carrying cost / that times the required multiple — (ch04)
- **Cost stack** — ordered deductions: spread, impact, commission/fees, financing, fund expenses — (ch18)
- **COT / open interest / net position / contract market** — weekly CFTC positions by trader category (TFF vs Disaggregated); contracts outstanding; long minus short; several CFTC markets map to one product code — (ch04)
- **Counterfactual account path** — equity series that keeps tracking the market after a halt; not realized P&L — (ch26)
- **Covariance-explaining vs priced factor** — direction carrying variance vs direction carrying expected return; can diverge — (ch14)
- **Coverage** — (a) `ic_n_days` / `full_coverage`: number of dates a configuration scored a usable IC; must match for any cross-family comparison; coverage bar restricts to maximum-coverage prediction sets; (b) feature coverage: share of sessions a column holds a value from its first fill (floor 0.70); (c) prediction-set coverage: share of canonical validation dates scored (eligibility for ranking); coverage gate `check_prediction_coverage` with gap kinds `missing_sessions`, `missing_fold`, `undeclared_fold`, `out_of_window`, `unaccounted_window`; (d) conformal/interval coverage (see Conformal prediction) — (ch12, ch08, cs_etfs, cs_nq100, cs_useq; cs_fx, cs_etfs; cs_sp500eo, cs_crypto)
- **`covers_fold_grid` / `covers_zero`** — flag that a selection's folds span the case study's full modelling grid / flag that the 95% HAC interval includes zero — (ch12, cs_usfirm)
- **CPCV** — combinatorial purged cross-validation: C(N,k) splits (S groups, C(S, S/2)) assembled into k C(N,k)/N complete backtest paths; skfolio and `ml4t-diagnostic` built-ins — (ch06, ch07, ch17, lib_diagnostic)
- **Crisis alpha / pre-discovery evidence** — positive return in an equity drawdown / factor returns predating publication (not pre-registered) — (ch17)
- **Cross-encoder / dense bi-encoder** — reads query and passage jointly (cannot be precomputed) / documents and queries embedded independently, ranked by cosine — (ch22)
- **Cross-fitting** — nuisance model never scores a row it was fitted on; the temporal embargo extends it across time — (ch15, cs_fx)
- **Cross-sectional rank / percentile / z-score / within-date clip** — position among assets on a date (ranks mapped to [-0.5, 0.5]; percentile in (0,100), `features.ranked`); benchmark-relative = minus or divided by the benchmark; dispersion = cross-asset std; `rolling_rank` = trailing-window rank along an asset's own history; winsorization and rank computed inside each decision date after the eligibility gate — (ch04, ch09, cs_sp500eo, cs_etfs)
- **Crossed / locked market** — bid > ask (reconstruction error; not executable) / bid = ask — (ch03, ch18)
- **Crossing / worked order / passive** — paying the full half-spread, a fraction, or earning spread (with fill risk) — (ch18)
- **Crowding / crowding_score** — many institutions holding the same name; n_holders / median n_holders (ch23) or breadth / HHI clipped 0.01, ceiling n^2 (ch22) — (ch22, ch23)
- **Curvature (futures curve)** — `(c0 - 2c1 + c2)/c1` — (cs_cme)
- **Cutoff date** — last date agent search may return documents from; results on/after it are dropped — (ch24)
- **CV gate** — coefficient of variation as a regime gate (ADIA 2025) — (lib_engineer)
- **`cv_identity`** — row key that must match across compared latent-factor rows — (cs_useq)
- **Cyclical encoding / time-to-event** — sin/cos mapping of periodic calendar variables; capped countdown to the next scheduled event (`H_MAX = 63`) — (ch08)
- **Cypher / UNWIND / MERGE** — Neo4j query language; UNWIND expands a list parameter into rows for batched writes; MERGE is idempotent create-or-match; `ROW_LIMIT` 25 for validated queries — (ch23)

### D

- **Data plane / execution plane** — the venue providing prices and funding vs the venue accepting orders — (ch25)
- **DDPM / DDIM / x0-prediction** — diffusion with the full reverse chain / accelerated sampler (eta controls injected noise; 0 = deterministic) / denoiser outputs the clean sample, noise derived analytically — (ch05)
- **Decimalization** — 2001 move to decimal ticks; pre-2001 spreads 15-30 bps, post 5-15 bps — (cs_usfirm)
- **Decision clock / decision bar / decision date / decision session** — the panel's numbered settlement index (holes preserved) or 15-minute schedule on which decisions sit, vs the calendar or the 1-minute watch clock; CME: last session of each ISO week (`resolve_rebalance_timestamps(ts, "weekly_friday_close", calendar="CME")`); backtest: complete daily cross section (`session_col`) whose final close is the decision price — (cs_crypto, cs_nq100, cs_cme, lib_backtest)
- **Decision-time admissibility** — only information available at decision time t may enter training for predictions at t — (ch06)
- **Deep ensemble / MC Dropout / epistemic / aleatoric uncertainty** — M independently initialised models (disagreement = uncertainty) / dropout kept active at inference / variance of member means, reducible with data / within-member predictive variance, irreducible, needs a mu/sigma^2 head; MC dropout is declared in model identity — (ch13, cs_cme)
- **Deep hedging / semi-recurrent hedger / self-financing P&L** — learning hedge positions by minimizing a risk measure of terminal P&L under costs; one MLP per timestep taking (information, previous position); hedge gains minus proportional costs minus option payoff — (ch19, ch21)
- **DeePM / SoftMin objective / FiLM / V-VSN / Directed Delay / macro graph prior / static metadata** — regime-robust deep portfolio policy; -SR_pool - lambda SoftMin_tau({SR_b}) over rolling windows (`softmin_lambda`, `softmin_tau`); feature-wise linear modulation by context (asset/group embeddings, costs); vectorized variable selection; cross-sectional attention with one-step lag; binary adjacency as attention mask; per-asset ids, group ids, cost bps — (ch17, lib_models)
- **Define-by-run** — Optuna API building the search space inside the objective via `suggest_*` — (ch12)
- **Degenerate prediction set** — one with a constant-prediction fold; excluded from trials — (cs_cme)
- **Delta hedge / delta threshold** — offset share exposure with the underlying at each close; rehedge when \|net delta\| breaches 0.1 — (cs_sp500opt)
- **Deployment artefact** — model + scaler/imputer + feature order + metadata fit by the live code path; distinct from the research registry artefact — (ch25)
- **Depth imbalance** — `(bid depth - ask depth)/total` snapshot over N levels — (ch03)
- **Derived fields (agent)** — confidence, sentiment, key findings, uncertainties, evidence quality; heuristics from `p_yes` and rationale — (ch24)
- **Deterministic vs probabilistic matching / token_set_ratio** — equal identifiers vs similarity above a threshold; rapidfuzz scorer ignoring word order and extra words; precision / recall / F1 score the matches — (ch04)
- **Development window** — pre-holdout sessions whose label resolves before the holdout (`_label_end < HOLDOUT_START`) — (cs_cme, cs_etfs)
- **Differentiable Sharpe loss / pooled Sharpe / robust Sharpe** — negative annualized Sharpe optimized by backprop; mean/std over all decision endpoints flattened; pooled Sharpe plus `softmin_lambda` x tempered minimum of window Sharpes — (ch17, lib_models)
- **Digest sidecar** — `.digest.json` with content digest, row count, keys, writer, input digests, estimation schedule — (cs_cme)
- **Direction A / B / native AUC** — classification score evaluated as ranker (IC vs continuous return) / regression score evaluated as classifier (mean daily AUC vs direction label) / classifier's AUC on its own label (`auc_mean_daily`) — (ch11, ch12)
- **Dirichlet distribution** — non-negative vectors summing to one; concentration controls spread (random-portfolio nulls) — (ch17)
- **Disappearance effect** — sub-half-share orders round to zero and are never submitted — (ch16)
- **Discriminative score** — held-out accuracy/AUC of a real-vs-synthetic classifier; 0.5 is the target — (ch05)
- **Discriminator gating / weight clipping** — skip the D update while its loss is below 0.15 (TimeGAN) / clamp discriminator parameters to +-0.01 (WGAN Lipschitz) — (ch05)
- **Disposition / four buckets / signal-only** — per-symbol execution outcome (flat, no mapping, unsupported short, dry run, no_credentials, submitted); intended / attempted / accepted / failed per-leg basket counts; a model intent with no venue mapping is logged, not traded — (ch25)
- **Distribution shift (RL)** — a clone acts in states its own errors create, which demonstrations never covered — (ch21)
- **Distributional hypothesis / skip-gram / CBOW / negative sampling** — a word is characterized by its co-occurring words; predict neighbors from the word (`sg=1`) / the word from neighbors (`sg=0`); random words pushed away per update to avoid collapse — (ch10)
- **Diversifier vs "leverage on the market in disguise"** — earns in both regimes vs only in the calm regime — (ch01)
- **DLinear / NLinear** — linear maps on moving-average trend and remainder (kernel 25) / on the window minus its last value (added back after); NLinear is the memoryless sequence-model bar — (ch13, cs_cme, cs_etfs, cs_sp500eo, cs_useq)
- **DML** — residualize outcome and treatment on confounders with nuisance models (`HistGradientBoostingRegressor`) fitted on earlier folds (cross-fitting, embargo), regress residuals; Neyman-orthogonal; Driscoll-Kraay SE; `confounding_bias_pct = (OLS - DML)/DML` (or `(naive - dml)/|dml| * 100`); walk-forward DML refits nuisances per fold; refuted by block permutation with blocks >= `treatment_window` — (ch15, cs_cme, cs_etfs, cs_nq100, cs_sp500eo, cs_sp500opt, cs_useq, cs_usfirm)
- **Dollar factor** — signed equal-weight average return of the 7 USD pairs; a proxy, not an estimated factor — (cs_fx)
- **Domain classifier** — classifier separating reference (0) from current (1); OOF AUC 0.5 means no detectable drift — (ch19, lib_diagnostic)
- **Dominance share** — fraction of (a, b) path pairs where model a has the higher excess kurtosis — (ch05)
- **Drawdown / underwater curve / max drawdown / time-to-recovery** — loss from a running peak (negative %); largest fall from the high-water mark of the compounded stream; that loss on every date; a path statistic — (ch01, ch17, ch19)
- **Drawdown ratio** — candidate max drawdown / incumbent max drawdown (promotion gate) — (ch26)
- **Drift** — covariate (P(X)), concept (P(Y\|X)) and prior (P(Y)) drift; online detection = detecting change as data arrive (inference); SHAP drift = change in per-feature mean \|SHAP\| across folds — (ch01, ch19, ch12)
- **Drift compensator** — lambda k subtracted so mu is the total expected return in jump-diffusion — (ch05)
- **Driscoll-Kraay SE** — Newey-West on time-aggregated panel scores (`hac-groupsum`); robust to dependence over time and across firms in the same period — (ch11, ch15, cs_cme, cs_etfs, cs_sp500eo, cs_usfirm)
- **Driver hypothesis / role separation** — the named economic mechanism a feature proxies; classifying a feature as signal (directional expectation), state (conditions signals; ranks nothing alone) or feasibility/cost input (Kyle lambda, Amihud); flow (window aggregate) vs state (snapshot) — (ch08, cs_etfs, cs_cme)
- **DSR / expected max Sharpe / variance_trials / effective-rank DSR** — probability the observed Sharpe exceeds the expected max of N null trials, `sqrt(V)[(1-g) Phi^-1(1-1/N) + g Phi^-1(1-e^-1/N)]`; dispersion of trial Sharpes drives the correction; trial count raw, Marchenko-Pastur or effective-rank K (correlated variants within family x label); not computed in ch20 — (ch07, ch16, ch20, cs_cme, cs_crypto, cs_etfs, cs_nq100, cs_sp500eo, cs_sp500opt, cs_usfirm, lib_diagnostic)

### E

- **E[T] / P[b=1] / E[theta] / theta** — expected ticks per bar (EWMA, `alpha`, `min_bars_warmup`), buy probability, expected imbalance (AFML); theta = cumulative imbalance or run statistic closing a bar when it exceeds E[theta_T]; threshold spiral = feedback between adaptive threshold and bar length under one-sided flow; `et_drift` = last/first `expected_t` — (ch03, lib_engineer)
- **ECDF** — empirical cumulative distribution; share of runs at or below each value — (ch26)
- **Effective number of trials / K_eff** — independent-trial count implied by correlated candidates; smaller than the raw count — (ch16, lib_diagnostic)
- **Effective positions / HHI** — 1/sum w_i^2 (inverse Herfindahl) vs the 1/N reference; ownership HHI = sum of squared holder value shares (1/n equal, -> 1 dominant); forecaster HHI reciprocal = effective panel size — (ch17, ch22, ch23, ch24)
- **Effective row / observation count** — row count deflated for overlapping forward windows; unchanged for a 1d label — (cs_nq100, cs_sp500opt, cs_useq)
- **Efficient frontier** — upper boundary of the feasible risk/return region (a picture of the sample); execution: schedules not dominated in both expected cost and variance — (ch17, ch18)
- **Eigenportfolio** — eigenvector used as portfolio weights, L1-normalized in raw-return space — (ch14)
- **Eligibility contract / group / `eligible_rows`** — registered rows a prediction set must cover exactly; populations sharing the same eligible rows (sequence vs cross-sectional); rows surviving the sequence lookback (60) — (cs_sp500opt, cs_sp500eo)
- **EMA of weights** — beta theta_ema + (1 - beta) theta, sampled from for smoother diffusion outputs — (ch05)
- **Endpoint / predecessor endpoint / two-pass gradient** — last date of a feature window (the scored decision) / prior decision prepended for continuous turnover / no-grad global gradient then chunked exact backward — (ch17)
- **Entity key / event timestamp / TTL / feature view** — column naming what a row is about (`symbol`) / when its values were true (`timestamp`) / how long a value stays servable (turns a missing row into an error, not a stale answer) / declaration of one feature table — (ch26)
- **Entry confidence drop** — exit when the entry model's probability falls below 0.30 — (ch19)
- **Entry rule / scheme** — `equal_weight_top_k` (ew_top3, ew_top5), `score_weighted_top_k`, `quintile_long_short`; top k long / bottom k short; feasible = `2k <= tradeable`, distinct from filled — (cs_crypto, cs_etfs, cs_usfirm)
- **Episode / switch count / revisited cluster** — maximal run of consecutive months with one label / number of label changes / cluster the model returns to after leaving — (ch01)
- **`epoch_at_high` / `ic_high`** — checkpoint and value of peak validation IC; diagnostic only — (cs_sp500eo)
- **Equal risk contribution / risk parity / risk contribution** — weights such that each asset's share of variance w_i (Sigma w)_i / (w^T Sigma w) is 1/N — (ch17)
- **Era stress test** — rerun the sort on time-split subsamples to separate footprint from artifact — (ch06)
- **Erosion** — no-impact return minus high-impact return — (ch21)
- **Estimand / estimator** — the causal quantity targeted (e.g. E[Y\|do(T=t+1)] - E[Y\|do(T=t)]) / the procedure producing a number for it — (ch15, cs_cme)
- **Estimator efficiency** — asymptotic variance ratio vs close-to-close under driftless GBM — (ch08)
- **EU AI Act** — European regulation mandating explainability (among other duties) for high-risk financial AI — (ch27)
- **Euler-Maruyama / exact transition** — approximate SDE discretization vs the exact OU step that removes discretization error — (ch05)
- **Event / disclosure / extraction time** — when it happened / when it became public / when the pipeline produced the edge — (ch23)
- **Event-time alignment** — IC should concentrate in the window the mechanism predicts — (ch07)
- **Evidence boundary / trial logging / sealed holdout / selection-aware evaluation / scoping invariants** — the separation between exploration and confirmation; mechanisms preserving integrity across many trials; rules fixed before research (README term) — (ch01)
- **EWMA (RiskMetrics)** — `h_t = lambda h_{t-1} + (1-lambda) r_{t-1}^2`, lambda 0.94 — (ch19)
- **Exceedance curve / move-to-cost** — fraction of \|moves\| at least a given multiple of the instrument's own spread or round trip; crossing at 1 = share of moves that clear cost — (cs_cme, cs_crypto, cs_etfs, cs_fx, cs_usfirm)
- **Exclusion taxonomy** — failure types in three buckets: signal invalidity, implementation infeasibility, evidence-quality failure — (ch20)
- **Execution bridge** — three-row comparison: vectorized, zero-cost engine, cost-aware engine — (ch17)
- **Execution convention** — label defined by entry (next-bar VWAP) and exit (H after entry), resolved by timestamp — (cs_nq100)
- **Execution log / reasoning trace / `AgentTrace` / `RunTrace` / `AgentState`** — `ToolExecutor` record (query, status, duration, `n_blocked`), separate from the per-step reasoning trace; outside-view JSON record; inside-view typed state (evidence, open questions, tool trace, gate outcomes, synthesis status) — (ch24)
- **Execution tier / workspace / preview** — canonical (full panel, published schedule, shared registry) vs preview (reduced via `START_DATE`, `MAX_SYMBOLS`, `MAX_FOLDS`, reductions in the identity, root `.preview/<case>`, rewrites `ML4T_OUTPUT_DIR`; needs `WORKSPACE`; cannot publish, reach holdout or join an official population) — (cs_cme, cs_crypto, cs_etfs, cs_fx, cs_nq100, cs_sp500eo, cs_sp500opt, cs_useq, cs_usfirm, ch08)
- **ExecutionMode / ShareType / FillOrdering / RebalanceMode** — `NEXT_BAR` \| `SAME_BAR`; `FRACTIONAL` \| `INTEGER`; `EXIT_FIRST` \| `FIFO` \| `SEQUENTIAL` \| `PRIORITY`; `SNAPSHOT` (targets once, all submitted) / `INCREMENTAL` (recomputed per fill) / `HYBRID` (targets once, cash checked live) — (ch16, lib_backtest)
- **`ExecutionResult` / `Fill` / `min_volume` gate** — limit output (fillable, remaining, placeholder adjusted_price/impact_cost, participation_rate) / what `FillExecutor` emits, price including impact and slippage / floor below which no execution occurs in a bar — (ch18)
- **Expanding past median** — median of all finite observations strictly before row t — (ch16)
- **Experiment budget** — number of configurations tried per family and label — (ch26)
- **Experimental feed** — adapter outside the stable boundary; requires `experimental=True` — (lib_live)
- **Explicit / implicit cost** — commission and venue fees / half-spread and impact — (ch18)
- **Exploration / confirmation arm** — two-pass screening: confirmation PROCEEDs via BH-FDR (`fdr_sig`, alpha 0.05; or alpha 0.10 on 80% then 0.05 held-out); exploration PROCEEDs via sign consistency >= 0.60 and \|IC\| >= 0.01 (ETFs) or 0.005 (FX, sp500eo), note `stable_and_above_threshold`, unconfirmed — (ch07, ch20, cs_cme, cs_etfs, cs_fx, cs_sp500eo)
- **Exposure proxy / terminal shortfall** — lambda x (avg inventory/Q) x sigma_t, linear bps penalty for carrying inventory / eta x unfilled fraction at the close — (ch18)
- **Extremization** — Neyman: push the panel mean from the base rate by d = sqrt(n/(1+(n-1)rho)), d^2 = effective panel size; Platt / log-odds: p' = d p^a/(d p^a + (1-p)^a), logit(p') = a logit(p); exponent k grid-searched over (0.5, 3.0), `at_search_boundary` flags a clamp — (ch24)

### F

- **Fail closed / Warden / fail-closed allowlist** — raise at construction or load time instead of silently substituting; proxy between agent and `ToolExecutor` denying anything not explicitly approved — (ch22, ch24)
- **Faithfulness / unsupported-claim flag** — share of answer tokens present in retrieved context; below `UNSUPPORTED_CLAIM_THRESHOLD` (0.5) / `SUPPORT_THRESHOLD` (0.4) on a non-refusal — (ch22)
- **Falsifiable hypothesis / OOS testing / multiple-testing correction / cognitive bias** — idea stated with a pre-specified disproving result; evaluation on data unused in fitting or selection; adjustment for variants tried; confirmation/narrative/overconfidence errors the workflow counters — (ch27)
- **Fama-MacBeth** — cross-sectional regression (or statistic) per date, inference from the time series of estimates — (ch04, ch11)
- **Family / feature family / family token** — model class grouping (`gbm`, `linear`, `deep_learning`, `tabular_dl`, `latent_factors`); feature name prefix before the first underscore (`mom_21` -> `mom`; none -> `other`); substring attributing a column to a model-based family — (ch26, ch08, ch09)
- **fANOVA** — functional ANOVA decomposing objective variance across hyperparameters — (ch12)
- **Fat-finger check** — `max_price_deviation_pct` limit-price vs market guard — (lib_live)
- **Feature store / training-serving skew / lineage** — component answering "what were this entity's features at this moment" for training and serving; divergence between offline and online feature answers computed by different code; where a served value came from (file, row count, date range, keys); a label's chain of selected signal -> allocation -> risk -> cost rows — (ch26, cs_sp500eo)
- **Feature-expectation matching / partition function Z** — iterate w by mu_E - mu_pi without a partition function; normaliser summing exp reward over a fixed bank of trajectories — (ch21)
- **Feller condition / full truncation** — 2 kappa theta > xi^2 keeps Heston variance positive; max(v, 0) in both drift and diffusion of the Euler step — (ch05)
- **FFD / fractional differencing** — (1-L)^d weights truncated at `threshold` (1e-5); d fixed in advance (0.4 / 0.5); boundary-partial keeps early rows with a shorter filter vs full-window nulls width - 1 rows — (ch09, cs_etfs, cs_useq, lib_engineer)
- **Fidelity / utility / privacy** — distributional realism / downstream task performance / resistance to leaking training records; mutually trading off — (ch05)
- **Filing date / period_end / report_date** — SEC acceptance date, the PIT anchor; fiscal quarter end reported on (not investable timing); 13F quarter-end a position is as of; reporting lag = `filing_date - period_end` — (ch10, ch22, ch23)
- **Filtered vs smoothed (HMM / ARIMA / GARCH / Kalman)** — P(z_t \| x_{1:t}) via forward recursion vs P(z_t \| x_{1:T}) via `predict_proba` / whole-sample inference; only the filtered one is a usable feature — (ch09, cs_cme, cs_crypto, cs_etfs, cs_fx)
- **FinMTEB / Matryoshka embeddings** — finance massive text embedding benchmark / nested representations allowing truncation to lower dimensions — (ch22)
- **FINRA TRF / odd lot / MWCB / LULD / speed bump** — off-exchange trade reporting (dark pools, internalisers) / < 100 shares / market-wide circuit breaker (7% -> 15-min halt) / limit up-limit down / IEX 350 µs delay — (ch03)
- **First notice date** — first day a long holder of a physically settled contract can be assigned delivery; `FirstNoticeDateRoll` rolls before it — (lib_data)
- **Fisher's method, Welch t, Levene, Fligner-Killeen, Jensen-Shannon, Hellinger** — p-value combination and two-sample statistics used as structural-break features — (ch09)
- **FitRunRecord / FitSummary** — redacted run provenance (dimensions, SHA-256, stopping reason) vs metrics outcome (converged, best_epoch, history) — (lib_models)
- **Fixed fractional sizing** — `shares = V r / |entry - stop|` — (ch19)
- **Fixed-horizon label / fwd_ret_h** — forward return (raw/log/binary) over H bars; `close.shift(-h)/close - 1` on adjusted close, endpoint = session t+h; `fwd_ret_dh_{5,10}d` delta-hedged variants are diagnostic only — (ch07, cs_etfs, cs_sp500opt)
- **Fold-derived fields** — `expected_prediction_keys`, `effective_params_by_fold`, `resolved_fold_digest`; recomputed for the holdout fold — (cs_cme)
- **Fold-SE / rank-1 cluster** — `std(ddof=1)` of per-fold Sharpe / sqrt(n_folds); configurations within a fold-SE of the top validation Sharpe (thickness measures selection stability) — (ch20)
- **Forced liquidation** — episode ending with an involuntary unwind on the last bar — (ch21)
- **`forecast_produced`** — artifact flag distinguishing a committed probability from the budget-exhaustion fallback (0.5, "Max steps reached") — (ch24)
- **Form 13F(-HR) / reported 13F value / put/call rows** — quarterly long-equity holdings of managers > $100M, due 45 days after quarter end; disclosed long US-equity/option value, not AUM; option positions (`PutCall`) excluded — (ch04, ch22)
- **Form 3 / 4 / 5 / transaction code** — insider initial holding / trade within 2 business days / year-end exempt transactions; P purchase, S sale, A grant, M/X exercise, G gift, D disposition to issuer, F tax withholding, C conversion, I discretionary — (ch04)
- **Formation cohort / panel size** — universe fixed at the first report period / unique CIKs counted over every filing — (ch23)
- **Forward algorithm / Viterbi / label switching / expected duration / state entropy / Markov switching AR** — filtered recursion; whole-path decode (non-causal); arbitrary ordering fixed by sorting; 1/(1 - A_kk); uncertainty of the filtered distribution; AR with regime-switching parameters — (ch09)
- **Forward fill** — carry the last released value forward (causal) — (ch04)
- **Forward return / overlapping windows** — return over a window after the signal date; consecutive forward returns share days, independent windows ~ n/horizon — (ch04)
- **Fractal efficiency** — path straightness, 1 trend, 0 noise — (ch08)
- **Front month / tenor / position** — contract closest to expiry (`position` 0; c0, c1, c2; Databento `.v.N` / `.c.N` rank); only the front is traded and labelled — (cs_cme, lib_data, ch02)
- **Frozen-evidence replay** — search client replaying saved queries/documents so only model/prompt/aggregation vary — (ch24)
- **Full covariance / `reg_covar`** — each GMM component has its own covariance / ridge added to covariance diagonals for stability — (ch01)
- **Full strategy specification / release configuration** — signal method + allocation method + risk overlay, declared across stages — (ch20)
- **Fundamental Law** — IR ~ IC x sqrt(BR) — (ch08)
- **Funding rate / premium index / funding age / funding event** — 8-hourly (00/08/16 UTC) payment between perp longs and shorts, `F = P + clamp(I - P, -c, c)` with I = 0.01%/8h, c = 0.05%; positive -> longs pay shorts; a transfer, not a fee; premium = (perp - spot)/spot (`premium_index_close`; its outage creates the 57-day hole); funding age = hours between the matched print and the 8H grid; backtest records a zero payment when no position; spec field `funding: position_signed_before_same_timestamp_fills` — (ch02, ch15, ch21, ch25, cs_crypto, lib_backtest, lib_data)

### G

- **Gain importance / rank shift** — LightGBM split-gain importance normalised per fold by its max (`importance_norm`) / `linear_rank - gbm_rank` on the common feature set (positive = GBM promotion) — (ch12)
- **Gamma (IPCA)** — `L x K` map from characteristics to betas, `beta_{i,t} = z_{i,t} Gamma`; orthonormal Gamma and ordered factor second moments for identification (inference) — (lib_models)
- **Gap / backfill / hole** — interval between bars exceeding expected delta (+ tolerance); fetching only gap windows; session spacing > 5 calendar days inside a product's series; gap policy `exclude_windows_crossing_missing_expected_periods` for sequence families — (lib_data, ch02, cs_cme, cs_crypto)
- **Gap (regime) / regime split** — test days above/below the median SPY 21-day annualized vol; Calm Sharpe minus Crisis Sharpe (a holdout diagnostic) — (ch17)
- **Gap-through** — bar opens beyond a stop level; fill at the open, not the stop — (lib_backtest)
- **GARCH(1,1) / persistence / half-life / backcast / `garch_cond_vol`** — sigma^2_t = omega + alpha eps^2_{t-1} + beta sigma^2_{t-1}; persistence alpha + beta (+ gamma/2 for GJR); half-life ln(0.5)/ln(alpha+beta); backcast = variance seed from the estimation window; per-stock conditional vol on the refit schedule, market-level fit below burn-in — (ch09, cs_crypto, cs_etfs, cs_useq)
- **Garman-Klass / Parkinson / Rogers-Satchell / Yang-Zhang** — range-based vol estimators; GK coefficient `2 ln2 - 1`, annualized by sqrt(252); first three within-session (blind to overnight gaps), YZ includes overnight — (ch08, ch09, cs_sp500eo)
- **GASF / MTF** — Gramian angular summation field cos(phi_i + phi_j) over arccos of a [-1,1]-scaled window / Markov transition field of quantile-bin transition probabilities — (ch13)
- **GAT / transductive inference** — single-head dense graph-attention autoencoder; encoder applied to the full graph including held-out nodes' features and edges, never their labels — (ch23)
- **Gating / scaling / conditional** — the three signal x state interaction templates — (ch08)
- **Generalized duration** — sensitivity to each yield-curve factor (level / slope / curvature = parallel shift, steepening, butterfly); hedged with key-rate instruments — (ch14)
- **Generation** — (a) immutable published data dir with a `CURRENT` pointer / JSON manifest; (b) refit under a corrected input artifact superseding prior registry rows; one holdout generation readable at a time; (c) monotonically increasing live state-envelope version for optimistic concurrency — (ch02, lib_data; ch11, cs_crypto, cs_usfirm; lib_live)
- **Granger causality** — lagged X improves prediction of Y (pairwise F-test); predictive, not causal — (ch15)
- **Graph RAG / Text-to-Cypher** — retrieval where a database query, not similarity, selects evidence; LLM-generated Cypher (ch23 nb03 routes instead) — (ch23)
- **GraphSnapshot / snapshot guard / snapshot manifest** — Neo4j node recording a producer's source bytes, cohort policy, periods, counts and run id; consumer-side identity check; JSON with cutoff, window, extractor identity and per-parquet SHA-256 — (ch23)
- **Greedy funnel / stage attrition funnel** — signal (all) -> allocation (top-10) -> risk/cost (top-1) -> holdout (one config); count of case studies passing each gate, independent or cumulative — (cs_sp500eo, ch20)
- **GRN / GLU / VSN / VLSTM** — gated residual network (ELU MLP, GLU gate, residual, LayerNorm) / a * sigmoid(b) / softmax over per-feature GRN embeddings / VSN + shared per-asset LSTM + GRN + tanh head — (ch17)
- **Gross / net exposure / gross leverage** — sum \|positions\| (its reciprocal is the uniform adverse move that ends the account) / signed sum; equal for long-only; as ratios to equity in the engine — (ch16, ch17, ch19, lib_backtest)
- **Gross-normalized weights** — long-short weights scaled so gross exposure is 1 within a date — (ch15)
- **`group_by_dynamic(label="right")`** — Polars window aggregation stamping each window at its right edge — (ch01)
- **Guided sampling / serialization (GReaT) / parse count** — be-great sampler enforcing the column schema row by row; table rows rendered as sentences for LLM fine-tuning; generated rows whose cell converts to a number — (ch05)

### H

- **HAC / Newey-West** — heteroskedasticity-and-autocorrelation-consistent standard errors with Bartlett weights, applied to the per-date IC series or time-sorted panel scores; the lag count must span label overlap (see Bandwidth L); T_eff = T (sigma_naive / sigma_HAC)^2; HC3 admits heteroscedasticity only, clustered within-cluster correlation — (ch04, ch07, ch08, ch10, ch11, ch12, ch13, ch14, ch17, ch19, ch24, cs_*, lib_diagnostic)
- **Half-Kelly / Kelly fraction / min_wealth_multiple** — bet size maximizing expected log terminal wealth: edge/odds (binary; `f* = W - (1-W)/R`), (mu - r_f)/sigma^2 (continuous), Sigma^-1 mu (multi-asset); fractional Kelly = fixed multiple; lowest point of the compounded wealth path relative to start — (ch16, ch17)
- **Half-life** — OU: ln(2)/kappa (continuous) or -ln 2 / ln\|phi\| (AR(1)); GARCH: sessions for half a shock to decay; cointegration spread: -ln 2 / ln(1 + phi); signal: lags until rank autocorrelation halves; liquidation: elapsed time at which half the position is done — (ch05, ch09, ch14, cs_etfs, lib_diagnostic, ch18)
- **Half-spread / quoted / effective / realized spread** — what an immediately executing order gives up (~half the quoted gap); at the touch / where the trade printed vs midpoint / what the liquidity provider kept after the price moved; ETF convention: cost per leg = tier half-spread + $0.0035/share, round trip = 2 x (half_spread + commission); options: `option_spread_fraction` / `cost_fractions` scale how much of the quoted half-spread is paid — (ch18, cs_etfs, cs_sp500opt)
- **HAR** — heterogeneous autoregression of realized variance over 1/5/22 (daily) or [5, 15, 60]-bar components — (ch09, cs_nq100)
- **Hard gate** — an evaluation question that blocks data integration alone (legal; unreconstructible history) — (ch04)
- **Harvey threshold** — t > 3.0 (t >= 3 on the mean-return t-stat under multiple testing) for new factor discovery — (ch07, ch17)
- **Health score / health state** — 0-1 aggregate of feature diagnostic outcomes with flags; `runtime_status()["health"]` in stopped / waiting_for_data / ok / feed_silent / idle_market_closed / broker_disconnected — (lib_diagnostic, ch25)
- **Hidden Markov model (HMM) / Gaussian HMM** — regime model with a transition structure making staying more likely than leaving (the standard choice for actionable regimes); state-specific mean/variance; refit every 63 sessions in the case studies — (ch01, ch05, ch09, cs_crypto, cs_fx)
- **High-water mark / water mark (HWM / LWM)** — highest price since entry through the prior completed bar; running high/low for trailing stops (`WaterMarkSource`, `InitialHwmSource`, `TrailStopTiming`: lagged = previous bar's mark, intrabar = updates before checking, live = current close); session start equity / HWM baselines in `RiskState` — (ch19, lib_backtest, lib_live)
- **Histogram binning / leaf-wise growth / oblivious tree** — bucketing feature values before split search (GPU FP32, LightGBM CUDA FP64); LightGBM grows the leaf with largest loss reduction (`num_leaves`); CatBoost symmetric tree, all nodes at a depth share one split — (ch12)
- **Hit rate** — share of names whose direction the model called correctly — (ch26)
- **Hits@k / MRR / Precision@k / Recall** — fraction of masked items in the top-k; mean reciprocal rank of the true (first relevant) item; share of top-k relevant; share of relevant retrieved — (ch10, ch22)
- **Hive partitioning / storage key / partition pruning** — `col=value/` directories (`year=/month=/`) enabling pruning on date filters; `asset_class/frequency/SYMBOL` logical string, <= 160 bytes — (ch02, lib_data)
- **Holder floor / top-holder share / `top_holder_pct` / `inst_coverage_pct`** — minimum holders for a stock to enter the ranking; largest single manager's share of disclosed value; n_holders / panel size — (ch22, ch23)
- **Holdout** — sealed final-confirmation period, opened once; history never read while models are chosen (offline stand-in for live data). *Holdout seal*: development boundary = `holdout_start` minus horizon in sessions (a label's endpoint, not its observation date); `holdout_endpoint_cutoff`. *Holdout decay*: (validation - holdout Sharpe) / validation, modest if < 0.50. *Holdout vintage*: fitted-feature values frozen at the last pre-window estimate. *Holdout closure*: paired comparison of the selected strategy across the split and against the EW universe — (ch06, ch26, cs_useq, cs_nq100, ch20, ch12, cs_cme, cs_usfirm)
- **Horizon alignment / reference frame / representation / aggregation** — matching a feature's lookback/decay to the label and execution horizon; the three construction knobs, only the first changes the hypothesis — (ch08)
- **`HORIZON_DEPENDENT_PROTOCOL_FIELDS`** — `("cv", "feature_artifacts", "label_artifact")`, the only fields allowed to differ across labels in one pool — (cs_crypto)
- **HPCA** — hierarchical PCA: within-sector PCA then across-sector PCA — (ch14)
- **HRP / quasi-diagonalization / cluster variance** — correlation-distance clustering, reordering the covariance by dendrogram leaf order, recursive bisection with inverse cluster-variance (w'Σw at inverse-variance weights) splits; never inverts — (ch17)
- **HTM daily-MTM cohort engine / cohort / daily MTM** — `_run_htm_daily_mtm` values legs daily, settles at intrinsic, hedges delta; straddles entered on one weekly decision date (up to 5 overlap at 1/N_ROLL capital each); per-cohort premium plus hedge P&L summed to portfolio returns — (cs_sp500opt)
- **Huber / epsilon-insensitive loss / Huber delta / `huber_alpha_scale`** — quadratic then linear / zero inside a band then linear; delta = 0.5 x std of the fold's training labels, quantized — (ch11, cs_crypto, cs_etfs)
- **Hurst exponent / rescaled range / DFA / rough volatility** — scaling exponent (0.5 random walk); R/S and detrended-fluctuation estimators; log-vol increments with H ~ 0.1 — (ch09, lib_engineer)
- **HyDE** — embed an LLM-generated hypothetical answer instead of the query — (ch22)

### I

- **IC / ICIR / pooled IC / paired daily IC / IC coverage / `ic_t`** — IC: per-date (session, settlement, month) cross-sectional Spearman between signal or prediction and forward return, averaged over dates (`ic_mean`, `ic_mean_daily`; +1 perfect order, 0 none); ICIR = mean IC / std IC (annualized by sqrt(252) in lib_diagnostic; not computed in ch10 nb09); pooled IC = one Spearman over all (date, asset) pairs; paired daily IC = both models on the identical timestamp-entity cross-section; IC coverage = dates on which IC was defined (not NaN from ties); `ic_t = ic_mean / (ic_std / sqrt(n_folds_ic))`; `ic_mean` vs `ic_best` = family average vs maximum over a sweep; RAS IC = Rademacher haircut for correlated signals tried — (ch04, ch07, ch08, ch10, ch11, ch12, ch13, ch14, ch15, ch23, ch26, ch20, cs_*, lib_diagnostic)
- **Identity audit / sibling / price-frame identity** — check that all non-varied identity fields match between parent and sibling (a variant differing in exactly one declared field: allocator, risk rule, cost rate); the price frame a backtest reads is the only other thing that changes for the holdout backtest — (cs_fx)
- **Identity-line R^2** — fit against the 45-degree line; negative when the estimate is worse than the benchmark mean — (ch18)
- **Idempotent run log** — re-running lands on the existing row or creates a new hash; never a duplicate row — (ch26)
- **Implementability check** — test that a strategy can be executed (costs, capacity, timing) (README term) — (ch01)
- **Implementation shortfall** — execution cost against the arrival price; sum over fills of (arrival - fill) for a sale, in bps of arrival notional — (ch18, ch21)
- **Implicit daily feed / session close** — daily feed whose timestamps do not state the session boundary (schedule code validates intervals and boundary modes); exchange-calendar boundary for weekly/month-end rebalances; intraday feeds without a calendar must pass `is_session_close` — (lib_backtest)
- **Index integrity / near-duplicate** — correct dtype, per-symbol monotone time, unique (date, symbol) / same key, different values — (ch07)
- **Indirect prompt injection / retrieval poisoning / prompt injection** — adversarial instruction hidden in a retrieved document; fabricated data in an untrusted chunk; scanned with role-override, tool-injection and exfiltration regex families — (ch22, ch24)
- **Inert control / sessions flattened** — overlay leaving Sharpe, drawdown and trades unchanged; sessions where the baseline booked a return and the overlay exactly zero — (cs_crypto)
- **Inference span** — dates after a fitted feature's estimation window (truly point-in-time) — (cs_sp500eo)
- **Information criterion** — penalised likelihood for model order selection — (ch09)
- **Information overload / PKM / T-shaped expertise / decision fatigue** — excess unfiltered input (a career risk); personal knowledge management; deep primary plus broad cross-functional knowledge; degraded judgement raising bias susceptibility — (ch27)
- **`input_identity` / `economic_cashflows`** — digests of data a run read (prices, funding) — (cs_crypto)
- **Institutional ecosystem / quant archetypes / quantamental** — hedge funds, prop shops, banks, asset managers; researcher, trader, developer, PM, risk manager; roles blending systematic and fundamental — (ch27)
- **Integer-share rule / top_k** — whole contracts; products per leg (5, 10) — (cs_cme)
- **Integrated autocorrelation time** — `1 + 2 sum` of initial positive ACF pair sums; periods per independent observation — (cs_crypto)
- **Interaction residual** — bundle effect minus the sum of one-factor effects — (ch16)
- **Interpretability / bias detection / robustness testing / auditability** — the four AI-ethics proficiencies practitioners must demonstrate — (ch27)
- **Inverse volatility / inverse-volatility weights** — w_i ∝ 1/sigma_i, normalized; ignores correlations; not risk parity — (ch17, ch19)
- **IPCA** — instrumented PCA: exposures a shared linear map of (lagged / same-month) characteristics, Gamma fitted jointly with factor returns by ALS — (ch14, cs_etfs, cs_sp500eo, cs_useq, cs_usfirm, lib_models)
- **Item 1 / 1A / 7 / 7A / 8 / longest-span rule** — 10-K Business / Risk Factors / MD&A / market-risk / financial statements; pick the heading match farthest from the next item heading — (ch04)
- **IV surface / `iv_30_atm` / `skew_rr_30_25d` / `ivrv_spread` / `garch_ivrv_spread` / IV term slope / `iv_convergence` / surface policy** — implied vol per name per session across delta (`atm 0.50`, `25d 0.25`, `10d 0.10`) and DTE buckets (`7d [5,10]`, `30d [25,35]`, `90d [80,110]`); 30-day ATM IV (feasibility ranking column); 25-delta risk reversal = put IV - call IV (crash-fear skew); `iv_30_atm - rv_20` in vol points; IV minus GJR-GARCH one-step vol (horizon mismatch acknowledged); short/long-dated ATM IV (> 1 inverted); quote-selection conventions (delta, interpolation, maturity mapping) — (cs_sp500eo, ch08, ch02)

### J-K

- **Jaccard similarity / overlap coefficient** — \|A∩B\| / \|A∪B\| over vocabulary or holder sets; shared holders / min(breadth_a, breadth_b) — (ch04, ch23)
- **Jump share / Lee-Mykland / RV / BV / VR(q)** — `(RV - BV)+ / RV`; return over local bipower sigma with Gumbel critical value; realized variance sum r^2; bipower `mu1^-2 sum |r_i||r_{i-1}|` (jump-robust); variance ratio (1 = random walk) — (ch03, ch09)
- **K-means / Lloyd / k-means++ / inertia / silhouette** — centroid-distance partitioning (no covariance, no probabilities); iteration and seeding; within-geometry cost; mean (b - a)/max(a, b) in -1..1, geometric separation only — (ch01, ch09)
- **Kalman filter / local linear trend / innovation / gain / hedge ratio / `kalman_smoothness`** — state [level, slope]; observation minus prediction; share of innovation believed; units of one asset held against another; process (Q) and measurement (R) noise; inverse of level-estimate uncertainty, depends on noise parameters and session index only; refit every 63 sessions — (ch09, cs_fx)
- **Kalshi / Polymarket / prediction market** — binary event contract paying $1 if the event occurs; price = probability; Kalshi Series / Event / Market, Polymarket condition ID / slug / token ID; `yes_bid` = highest standing YES bid (lower bound); threshold ladder = contracts at different thresholds = implied survival function; USDC on Polygon — (ch04, ch24, lib_data)
- **Kill switch / kill switch latch / kill condition** — portfolio-level rule that halts trading or liquidates (`MaxDrawdownLimit`, `DailyLossLimit`); governance, never swept; persisted `kill_switch_activated` flag with reason, survives restarts, cleared only by the operator (reducing orders optionally allowed); declared no-go thresholds: IC 0.01, edge/cost 1.2, micro-cap 50%, net Sharpe 0.3 — (ch19, ch25, cs_nq100, cs_useq, lib_live)
- **Kish effective sample size** — (sum w)^2 / sum w^2; row-equivalent of a weighted fit — (ch11)
- **Knowledge graph / property graph / triple / quadruple** — entities as nodes, typed edges, provenance as properties; nodes and edges carry key-value properties (Neo4j); (subject, predicate, object); (subject, relation, object, timestamp) event edge in FinDKG format — (ch23)
- **Kupiec POF test / QLIKE** — likelihood-ratio test of VaR exception count vs expected rate / asymmetric volatility-forecast loss `log(h) + sigma^2/h` — (ch19)
- **Kurtosis-matched Student-t** — `nu = 6/kappa + 4`, scale `sigma sqrt((nu-2)/nu)` — (ch19)
- **Kyle's lambda** — slope of price change on signed order flow (60-min regression intraday; price-normalized per-symbol Huber slope); Kyle lambda ratio = mean \|r\| / relative volume, an unsigned impact proxy — (ch08, ch18, cs_nq100)

### L

- **LABEL_AVAILABLE_AS_OF** — last training day whose forward label realizes before the live window — (ch25)
- **Label provenance / test-set contamination** — how labels were produced (annotator judgment vs price-move sign); a checkpoint trained on sentences in your test split — (ch10)
- **LambdaMART / monotone constraint** — gradient-boosted learning-to-rank with `lambdarank` objective and query groups; force a feature's effect non-decreasing (+1) or non-increasing (-1) — (ch12)
- **Largest-remainder allocation** — distribute leftover cash to assets with the largest fractional remainders, subject to a cash buffer — (lib_backtest)
- **Last-value / zero-forecast / persistence / closest-repeat baselines** — forecast = last observed price; predict 0 (MSE = return variance); repeat the last price over the horizon; repeat the last return (LTSF-paper baseline) — (ch13)
- **Latent factor** — unobserved common driver recovered from the return panel; 5 factors declared in CME — (ch14, cs_cme)
- **Leave-one-out** — fit on n-1 rows, score the held-out row; the only honest option at n=10 — (ch24)
- **Ledoit-Wolf shrinkage** — pull the sample covariance toward a well-conditioned target before inversion — (ch17, ch19)
- **Lifecycle V1** — versioned sync callback contract from ml4t-specs shared by backtest and live — (lib_live)
- **Liquid universe / quintile** — bottom 20% of names by half-spread (as a share of premium) per rebalance date — (cs_sp500opt)
- **Lo annualization / Mertens variance** — autocorrelation-corrected scaling of Sharpe across frequencies / Sharpe variance adjusted for skewness and kurtosis — (ch16)
- **Lock-notional** — short policy that locks short notional as collateral instead of crediting proceeds (VectorBT parity) — (lib_backtest)
- **Long-short / dollar-neutral / beta-neutral** — holds both sides / equal-size legs / market exposures cancel — (ch17)
- **Lookback / lag / timing contract / register** — bars read back from the decision timestamp (warm-up floor; `features.lookback()` from `core/lookbacks.py`) / sessions until the input is knowable (1 for option families and yield curve, 0 for price families); declared per family before coding in `setup.yaml` with role, hypothesis, inputs, frame, representation, failure_mode; warmup audit checks no value precedes its window — (cs_cme, cs_crypto, cs_etfs, cs_sp500eo, cs_usfirm, lib_engineer)

### M

- **Macro F1 / majority-class rate** — unweighted mean of per-class F1 / accuracy of always predicting the most common label (zero point for skill) — (ch10)
- **MAD** — median absolute deviation (÷ 0.6745 to sigma scale); default outlier method, threshold 3.0 — (ch02, lib_data)
- **MAE / MFE** — maximum adverse / favorable excursion from the bar after entry over [t+1, t+H] (forward-looking); percentiles seed stop/target priors and calibrate trailing stops; edge ratio = avg MFE / avg \|MAE\|, efficiency = realized return / MFE — (ch07, ch16, ch19, cs_etfs, cs_usfirm, lib_backtest, lib_diagnostic)
- **Managed portfolio** — per-date cross-sectional OLS coefficient x_t = (Z'Z)^{-1} Z' r_t (CAE's factor-network input, `compute_managed_portfolios`); also a long-short quintile portfolio rebalanced with information through t-1 — (ch14, ch15, lib_models)
- **`manager_compatible=False` / Template Method** — provider cannot be driven by `DataManager` (factor, macro, tick, artifact providers); `fetch_ohlcv` / `ProviderUpdater.update_symbol` fix the steps, subclasses fill `_fetch_*` / `_transform_*` — (lib_data)
- **Marked wealth** — cash + inventory valued at next price — (ch21)
- **Market-level column / null-policy carrier** — value constant across pairs on >= 0.90 of sessions (cannot rank); the longest-chain column (`zscore_126d`) whose first valid bar sets row retention — (cs_fx)
- **MARKET_DATA_TYPE / MarketSnapshot** — TWS quote mode 1 real-time, 2 frozen, 3 delayed, 4 delayed-frozen; per-asset latest price with timestamp kept by `SafeBroker` for staleness and sizing checks — (ch25)
- **Markowitz curse / max-Sharpe / minimum-variance portfolio** — MVO amplifies small estimation differences into large weight differences through inversion; highest excess return per unit vol at a hurdle / lowest-variance long-only portfolio (independent of mu) — (ch17)
- **Massart bound** — sqrt(2 ln N / T), max of N standardized means; benchmark for independence — (ch07, ch16)
- **MaxEnt IRL** — reward weights maximising trajectory likelihood exp(w^T Phi)/Z — (ch21)
- **MD&A / 10-K / 10-Q / 8-K** — Management's Discussion and Analysis; annual / quarterly / material-event reports (the 10-K replaces Q4's 10-Q) — (ch04, ch10)
- **Mean pooling** — attention-mask-weighted average of token embeddings; or averaging chunk-level scores into a document value (`sentiment_mean / sentiment_std / pos_pct / neg_pct` from `label_sign x confidence`) — (ch10)
- **Mean-forecast ensemble** — `gbm_mean_leaves31`: average of member predictions at their last checkpoint — (cs_nq100)
- **Mechanics change** — alteration of any setup invariant; a new setup version, not a tuned trial — (ch06)
- **MedianPruner** — stops a trial whose intermediate value is below the median of prior trials at that step — (ch12)
- **Meta-labeling** — secondary model predicting whether a primary directional signal pays, used for sizing (`sized_signal`) — (ch07, lib_engineer)
- **Microprice / weighted mid** — size-weighted bid/ask midpoint leaning toward the thin side of the book — (cs_nq100, lib_engineer)
- **MinTRL / T_plan** — minimum track-record length for an observed Sharpe to clear a threshold (`never` when unattainable; periods to reach 5% significance) / observations needed before collection to detect a target Sharpe with stated power — (ch07, ch16, cs_crypto, lib_diagnostic)
- **Mixture of experts / paired difference** — one model per hard state / fold-wise design difference — (ch09)
- **MMD / median heuristic / stream lift / p-Wasserstein / barycenter / mixed window** — kernel distribution distance; kernel-width rule; windows as sorted samples; matched-quantile distance; quantile-wise median/mean; window straddling a switch — (ch09)
- **MOC** — Market-On-Close order routed to the closing auction (`OrderType.MOC`) — (ch25)
- **Model / signal / strategy diagnostics** — fit quality / ordering of forward returns / P&L after mapping and costs — (ch06)
- **Moment loss / moment network** — TimeGAN auxiliary loss matching mean and std of real vs synthetic batches; SDF adversary generating instruments that maximize conditional pricing error (`MomentNetwork`) — (ch05, ch14, lib_models)
- **Momentum variants** — 12-1 / skip-month: return from t-252 to t-21 (12 months ago to 1 month ago), dropping the reversing recent month; skip-1: excludes the most recent day (bid-ask bounce); residual: return minus beta x market; risk-adjusted: trailing return / annualized trailing vol; momentum crash: underperformance in high-vol regimes; momentum z-score proxy: rolling z-score of close-to-close returns standing in for the perp-spot premium — (ch06, ch07, ch08, ch16, cs_sp500eo, ch25)
- **Monotonicity / quantile spread / decile spread** — Spearman rho of quantile rank vs mean return (or fraction of upward steps); top minus bottom quantile mean return (bps x 10,000 for deciles) — (ch07, cs_usfirm, lib_diagnostic)
- **Month codes** — F-Z; quarterly H=Mar, M=Jun, U=Sep, Z=Dec; `ESH25` = ES, March 2025 — (ch02, lib_data)

### N

- **Narrative change / news surprise / weighted surprise / sentiment momentum** — `1 - cosine_similarity` between consecutive MD&A embeddings; cosine distance between today's news embedding and the mean of the prior 20 news-days'; surprise x sign(mean sentiment); mean sentiment minus its value 20 news-days earlier — (ch10)
- **NAV turnover** — traded notional in units of NAV (round-trip = opened and closed) — (ch18)
- **Negative price policy** — `forbid` (error), `warn`, `allow`; needed for commodities/futures (negative WTI, spreads) — (lib_data)
- **Net of market** — bucket return minus equal-weight universe return; beta removed, costs not deducted — (ch06)
- **Neural ODE / GRU-ODE / smoothness ratio / bounded fraction** — hidden state dh/dt = f_theta(h, t) integrated by a solver, queryable at any time; GRU update at observations, ODE between; mean \|delta\| of interpolation over real (~1 healthy); share of midpoint interpolations within adjacent real values +-0.1 — (ch05)
- **NeuralSort / S_quant / W VaR <= ES projection** — differentiable sorting with temperature tau; strictly consistent scoring function for the (VaR, ES) pair; hard constraint keeping discriminator VaR/ES economically consistent — (ch05)
- **Next-open fill / NEXT_BAR** — an order from a close-derived signal executes at the following session's open (bar after the decision); the default and realistic choice for close-based signals; same-bar fills on the signal bar's close — (ch16, ch17, ch19, ch25, lib_backtest)
- **NISQ / DeFi / yield farming / smart-contract vulnerability** — noisy intermediate-scale quantum era; on-chain protocols offering live alpha with novel risks; deploying capital across protocols for rewards; code-level exploit risk — (ch27)
- **NLV** — net liquidating value = cash + marked positions (`AccountState.total_equity`) — (lib_backtest)
- **No-transaction band / Whalley-Wilmott** — region around the frictionless target where trading is suboptimal under proportional costs; small-cost asymptotic no-trade band around delta — (ch19, ch21)
- **Noise floor / future-perturbation invariance** — per-column determinism noise scaled by `noise_multiplier`; features at t <= T unchanged when inputs after T are destroyed — (lib_diagnostic)
- **NON_OVERLAPPING** — `label_horizon=1` sentinel for independently drawn periods — (ch07)
- **Normalized feature / registry feature** — bounded, scale-free output flagged `normalized=True`; function decorated `@feature`, executed by `compute_features` via signature-aware dispatch — (lib_engineer)
- **NOTEARS / augmented Lagrangian / VAR-LiNGAM / varsortability / amortized inference** — continuous DAG learning with h(W) = tr(e^{W∘W}) - d = 0 and l1 penalty; penalty-plus-multiplier (rho, alpha) scheme; VAR for lags + DirectLiNGAM (ICA, non-Gaussianity) on residuals; degree to which marginal-variance ordering matches causal order in simulations; learning a data-to-structure map from simulated examples — (ch15)
- **NSGA-II / Pareto front** — elitist multi-objective genetic algorithm / non-dominated solutions — (ch12)

### O

- **Observation grid / session index / window** — the panel's actual sessions (holdout interval stepped along it, not the calendar); NYSE exchange-calendar position (windows and labels count sessions, not rows); 60 consecutive sessions (daily) or 12 Fridays (weekly) of one stock's features as one example — (cs_fx, cs_useq)
- **OFI** — message-based `(bid adds - bid removes) - (ask adds - ask removes)` per interval; aggressor-based `(V_buy - V_sell)/(V_buy + V_sell)` in [-1, 1] from tick-rule-classified trades — (ch03, ch08)
- **OHLC invariants / `ohlc_mode`** — high >= low/open/close, low <= open/close, volume >= 0; `strict | drop | warn` handling of violations — (ch02, lib_data)
- **One-way / one-sided turnover** — ½ Σ\|w_t - w_{t-1}\| (sum of absolute weight changes at a rebalance, including entry of an open end-of-sample position); round-trip cost = 2 x one-way cost; net Sharpe charges `COST_BPS` per side on it; vector-L2 turnover is diagnostic only — (ch11, ch16, ch17, ch18)
- **OOF stacking feature** — a model's prediction for a row from a model that never saw it — (ch19)
- **Operator / Skill (`SKILL.md`) / Registry** — thin loop giving an LLM general tools (files, bash, SQL, parquet, skills) with domain logic in skills and `ml4t-*` libraries; concept-first doc with WRONG/CORRECT example and a `## Production Implementation` pointer; SQLite WAL run-log of case-study models, predictions and backtest specs (`registry.db`, `registry_readonly_uri`, `read_only_study` opener that does not `Study.activate()`, `immutable=1` safe only for unwritable bundles) — (ch24, ch25, cs_nq100, ch15)
- **OPRA / `cbbo-1m` / SIP / IEX / CCCAGG** — US options tape / Databento consolidated BBO 1-minute schema / consolidated US tape vs single-exchange free feed (Alpaca `feed`) / CryptoCompare cross-exchange aggregate — (lib_data)
- **Option chain / moneyness / smile, skew, term structure, surface / put-call parity / Greeks / time vs intrinsic value / straddle** — strike/spot; IV geometry; one call and one put on the same underlying, strike and expiry (ATM when strike nearest spot); short straddle earns at most the premium, loses without bound — (ch02, cs_sp500opt)
- **Oracle baseline / oracle benchmark** — retrieval whose predicates define the gold set (recall 1 by construction); a schedule computed with whole-episode information (Almgren-Chriss) — (ch23, ch21)
- **Order imbalance (RL)** — mean-reverting [-1, 1] proxy for arriving-order pressure — (ch21)
- **Order pool / `stock_locate` / `tracking_number` / `price4` / Replace (U) / C / P / Q / UNDEF_PRICE / F_TOB** — `order_ref -> (side, price, remaining)`; ITCH symbol id from R (D/X/E/C/U carry only this); tiebreaker within a timestamp; integer price with four implied decimals; retires one reference and issues another; execute at a price other than the displayed limit; non-displayed trade / cross; DataBento sentinels for clearing a side on top-of-book venues — (ch03)
- **Ownership churn / churn / `supplier_dependency_score` / `supplier_overlap_ratio` / `position_value_cv`** — 1 minus mean Jaccard of consecutive-period holder sets; additions / removals between states (constant under disjoint windows); mean over suppliers of 1 / (covered companies using it); shared / total suppliers; CV of total value across periods — (ch23)

### P

- **Paired comparison / paired difference / paired standard error / paired (stationary) block bootstrap** — day-by-day difference of two strategies (variant minus its own baseline differing in one spec field, on the exact timestamp intersection), resampled in contiguous day blocks to get a CI on the Sharpe difference; SE of per-episode differences on shared seeds — (ch17, ch20, ch21, cs_cme, cs_crypto, cs_nq100)
- **Panel state** — `readable` / `awaiting` / `unreachable` for `features/financial.parquet` — (ch08)
- **Papermill / structlog** — notebook parameterization overriding the parameters cell in the test harness; structured logging used by the AQR provider — (ch01, ch02)
- **Parity test / parity gate / parity profile / preset / target-replay parity / negative control** — field-by-field assertion that two engines produce the same tape on the same bars (fails CI on mismatch); `BacktestConfig` preset reproducing another engine's conventions (`backtrader`, `vectorbt`); two engines fed identical frozen targets must match fills, valuations and terminal value; a one-unit perturbation of the first fill price the check must detect — (ch25, lib_backtest, ch16)
- **Participation rate / percent of volume / participation ratio** — order size / volume available in the interval (the input every impact model and limit consumes); trade a fixed fraction of printed volume, completion time floats; (unrelated) `(sum lambda)^2 / sum lambda^2` of correlation eigenvalues = effective number of independent bets (5.27 in FX) — (ch18, ch21, cs_fx)
- **Path signature / log-signature / signed (Levy) area / expected signature / Sig-W1 / factorial normalization / lead-lag transform / VisiTrans("I") / purge gap** — iterated integrals characterizing a path up to reparameterization; minimal form; loop orientation; mean of truncated signatures (the Sig-WGAN analytic discriminator); L2 distance between expected signatures; multiply level k by k!; path doubling with offset copies (recovers quadratic variation); I-visibility transform adding 2 boundary rows and 1 marker column; rows dropped at split boundaries — (ch05, ch09)
- **Path statistic / restricted stream / pooled vs rolling volatility** — statistic that depends on order (drawdown, run length, time to recovery, stop-loss); a regime's months concatenated in date order; std of all monthly returns vs mean of a trailing-12-month std series — (ch01)
- **PatchTST / token / iTransformer** — transformer over contiguous patches of a 60-step window; the unit attention compares: a day (vanilla), a patch (PatchTST), a whole feature's history (iTransformer) — (ch13, cs_sp500eo)
- **PBO / CSCV** — probability of backtest overfitting: share of combinatorial symmetric splits where the IS-best configuration ranks in the bottom half (below the OOS median); two-fold variant in sp500eo — (ch06, ch07, ch16, cs_cme, cs_sp500eo, lib_diagnostic)
- **PCA / scree plot / elbow / persistent entities** — orthogonal directions of greatest remaining variance (explained-variance ratio, loadings); eigenvalues in descending order (a retention heuristic, not evidence); each panel column must be the same asset throughout for return-panel PCA (`PersistentPanelBatch` `(T, N)` vs `CrossSectionBatch` with `mask`) — (ch01, ch14, cs_cme, cs_etfs, lib_models)
- **PCMCI / pc_alpha / MCI test / ParCorr / `fdr_bh`** — PC1 parent selection + momentary conditional-independence tests for lagged links; PC1 regularization level (None = auto), not FDR; CI test of X_{t-tau} -> Y_t given both parents; linear partial-correlation test; BH adjustment — (ch15)
- **PELT / Binseg / CUSUM / MOSUM / Zivot-Andrews / structural break** — penalised exact / greedy segmentation; cumulative / moving sums against a burn-in reference; one-break unit-root test; regime-change date — (ch01, ch09)
- **PENDING_CANCEL / terminal state / replacement gap** — explicit state for a cancel in flight (exits to CANCELED, FILLED or back to ACCEPTED); FILLED, CANCELED, REJECTED, EXPIRED (no outgoing edges); cancel succeeded but replacement not accepted (`replacement_gaps()`) — (ch25, lib_live)
- **Perpetual future** — futures contract with no expiry, anchored to spot by funding rather than delivery; impact bid-ask, dead zone — (ch02, cs_crypto)
- **Persistence-cost score / signal decay** — phi/(1 - phi + Gamma), teaching proxy for alpha-to-go; exponential loss of captured alpha with rebalance delay — (ch18)
- **Phantom session** — holiday prints by a sliver of the universe — (ch02)
- **Placebo tests** — timing placebo: shift the feature by lags and watch IC decay; shared-driver check: test whether the feature predicts an outcome it should not (Treasury IEF); temporal placebo / placebo-date: regress outcome on the lead of treatment / shift treatment by several days; negative-control outcome: pre-treatment variable the treatment cannot cause; placebo portfolio / benchmark: random-selection or 500 random dollar-neutral portfolios giving a null interval per factor beta; plumbing test: random-signal backtest that must not be profitable — (ch07, ch15, cs_etfs, cs_usfirm, cs_nq100)
- **Placeholder bar / last complete bar / retreat / `lookback_days` / freshness** — ml4t-data update vocabulary; freshness <= 3 days fresh, > 7 stale; update strategies `incremental | append_only | full_refresh | backfill` — (ch02, lib_data)
- **Platt scaling** — sigmoid fitted to raw scores (`CalibratedClassifierCV(method="sigmoid")`) — (ch11, ch24)
- **Point-in-time (PIT) universe / eligibility** — stocks eligible on a date using only prior information (`close > 5`, `adv_21d > 1M`, 21 covered sessions; ETFs: admission to year Y decided from year Y-1, `eligibility.csv`); survivorship-free panels keep delisted assets and rebuild the universe per decision date — (cs_useq, cs_etfs, ch07)
- **Policy collapse** — a learned policy settling on one locally safe action (TWAP pace, widest spread, one position) for lack of exploration — (ch21)
- **Popularity baseline / portfolio sentence** — predictor always returning the most-held items; an institution's holdings ordered by value as a token sequence — (ch10)
- **Population / OfficialPopulation / supersedes / `superseded_members` / `SUPERSEDES_*` / `supersedes_hash`** — named, immutable, declared-before-fit list of identities (training runs, prediction sets, backtests; device-specific); a re-declared population or generation must name the one it replaces (`"live"` = current head); retired vs unpublished members; the lineage filter (vs `identity_status`, a schema-version marker); registry pointer from a refit row to the row it replaces; `official_populations` — (ch11, ch15, cs_cme, cs_crypto, cs_etfs, cs_fx, cs_nq100, cs_sp500eo, cs_sp500opt, cs_useq, cs_usfirm)
- **PositionState / PositionAction / PortfolioState / RuleChain / AllOf / AnyOf / position rule** — one position's state and the rule verdict (HOLD, EXIT_FULL, EXIT_PARTIAL, ADJUST_STOP); equity, high-water mark, positions, daily P&L snapshot for portfolio limits; first-non-HOLD-wins chain / AND / OR (AnyOf aliases RuleChain); client-evaluated protective exit rule tracked in `PositionRuleState`; FX: `stop_loss`, `trailing_stop`, `time_exit` per position — (ch19, lib_live, cs_fx)
- **Post-double-selection (PDS)** — LASSO on outcome and on treatment; union of selected controls enters the final OLS — (ch15)
- **Prediction bridge / UNFOLDING_COLLECTION** — exporting model scores as a file so a managed platform holds only portfolio rules; LEAN file format streaming one date at a time to avoid lookahead — (ch25)
- **Prediction hash / training hash / backtest hash / training identity** — content-addressed identities (`training_hash` -> `prediction_hash` -> `backtest_hash`, each carrying its parent); prediction hash carries its split, used for tie-breaking and exact-row loading; training identity hashes feature lineage, model parameters, CV interval, device and checkpoint and decides registry reuse (a holdout refit produces a new one); retired prediction sets are excluded via `exclude_prediction_hashes` — (ch08, ch26, cs_etfs, cs_sp500eo, cs_sp500opt, cs_useq, cs_usfirm)
- **Price improvement / markout / spread (bps)** — fill vs limit on C messages; forward return from decision time (midpoint, latency-adjusted, executable); `(ask - bid)/mid x 10,000` — (ch03)
- **Primary label / short name** — per-case-study headline target from `PRIMARY_LABELS` / display alias from `SHORT_NAMES` — (ch12)
- **Procrustes rotation / principal angles / projector distance** — K x K orthogonal R minimizing \|\|B_t R - B_{t-1}\|\|_F; rotation-invariant subspace recovery diagnostics — (ch14)
- **Profile / severity** — `DatasetProfile` of `ColumnProfile`s stored as JSON beside the data; `info < warning < error < critical` — (lib_data)
- **Promotion gate / signal correlation / position agreement / realization ratio** — conjunction of fixed criteria to leave shadow mode; per-session Spearman between two models' scores; intersection over union of names held by two books; backtest-to-live performance comparison (§26.2; not implemented) — (ch26)
- **Provenance / source policy / stratified sample** — origin record of evidence (URL, publisher, date); allow/block domain sets applied post-retrieval; equal draw per category (7 per E/S/G) that must be declared — (ch24, ch22)
- **Proxy labels** — relevance labels generated by a rule (term overlap) instead of human judgment — (ch22)
- **PSI** — Σ (p_cur - p_base) ln(p_cur/p_base) over reference-defined bins; bands 0.10 / 0.25; no null distribution; consensus-flagged with Wasserstein and domain-classifier measures — (ch02, ch09, ch19, ch26, lib_diagnostic)
- **PSR** — probabilistic Sharpe ratio: P(true SR > benchmark) given sample length, skew and kurtosis; p-value for one series — (ch16, cs_crypto, cs_etfs, cs_usfirm, lib_diagnostic)
- **Publication lag / release lag / reporting lag / schedule bound / release schedule** — delay between the period a statistic describes and its release (handled with `conservative_lag`); days from reference-period end to publication; `filing_date - period_end`; assumed publication date at the late end of the agency's schedule; frame mapping COT `report_date` -> tz-aware `available_at` — (ch01, ch04, ch08, ch22, lib_data)
- **Purge / embargo / label buffer / feature buffer / buffer** — purge (label buffer): drop training samples whose label window overlaps the validation/test window, i.e. within one label horizon before validation start (21D / 5D in sessions; 35 calendar days for options; H+1 bars intraday; 8H/24H crypto); embargo (feature buffer): extra gap after the test block, one feature lookback or horizon (5 / 21 observations at the holdout boundary); strict temporal boundary `max(train dates) < min(test dates)`; group isolation = no asset in both; supplied by `ml4t-diagnostic` splitters, verified by the builder — (ch06, ch07, ch11, ch12, ch13, ch14, ch15, ch19, cs_cme, cs_crypto, cs_etfs, cs_fx, cs_nq100, cs_sp500opt, cs_useq, lib_diagnostic, lib_engineer)

### Q-R

- **QLBS / quality gate / `qc_*`** — Halperin's Q-learner in the Black-Scholes world (coarse tabular variant); code-level contract on evidence checked before synthesis (coverage, freshness, consistency); solver-convergence flags carried as negative controls — (ch21, ch24, cs_sp500opt)
- **Quantile / quintile analysis** — sorting observations into signal buckets and comparing mean forward returns — (ch10, lib_diagnostic)
- **Quote scale / settlement price / margin pct / SPAN** — `tick * contract_size / tick_value`, 1 or 100; calculated end-of-session value that may sit off the tick grid; initial and maintenance deposit as a fraction of notional (maintenance ~ initial/1.10; leverage = 1/margin_pct) — (cs_cme)
- **Rademacher complexity (R_hat) / RAS / kappa** — expected max correlation of the hypothesis class with random signs (effective search size); lower bound theta_hat - 2 R_hat - 2 kappa sqrt(log(2/delta)/T); a priori bound on per-period observations (1.0 for Spearman IC) — (ch07, ch08, ch16, cs_crypto, lib_diagnostic)
- **Rank autocorrelation / redundancy cut** — period-over-period Spearman of signal ranks (turnover proxy); \|Spearman\| >= 0.7 between features counts as one ordering — (ch06, cs_crypto)
- **Rank-one configuration / selection rule / ranking parity** — max-IC candidate covering all declared folds and the maximum day count; highest validation backtest Sharpe, ties by backtest hash, IC/costs/holdout excluded; agreement between two trackers on each group's top run — (ch12, cs_useq, ch26)
- **Rashomon effect / explanation instability** — many models with similar performance but different attributions; near-identical predictions with different SHAP profiles — (ch12)
- **Rate-limit tuple** — `(calls, period_seconds)` client-side throttle; adaptive limiter halves rate after 429 — (lib_data)
- **ReAct / Tree of Thoughts / Reflexion** — loop alternating reasoning and tool actions (search, forecast) until a forecast or the step budget; branching deliberate search (branch-heavy decisions only); verbal self-reflection stored across attempts (needs memory governance) — (ch24)
- **Reality Check** — White's bootstrap test of best-of-many vs benchmark — (lib_diagnostic)
- **Rebalance id / rebalance step / thresholds** — broker-issued id shared by all orders from one `execute()` call (`priority_notional` gatekeeping); schedule slots between decisions so holdings do not overlap (1 for 8h labels, 3 for 24h); skip when \|dw\| < 0.005 AND notional < $100 — (lib_backtest, cs_crypto, cs_etfs)
- **Reciprocal rank fusion** — `sum_r 1/(k + rank_r(d))` over retrievers, k = 60 — (ch22)
- **Reconciliation / residual delta** — diff of broker positions vs model targets producing `delta_qty`; `SafeBroker.connect()` diff of persisted state vs broker at startup; non-zero `delta_qty` after fills settle — (ch25, lib_live)
- **Reference-price P&L / signed per-share price move** — P&L at the un-slipped execution base price; impact-model output added once to the decision price (positive buys, negative sells) — (ch16, ch18)
- **Reference tape / tape / reference window** — deterministic offline NEXT_BAR replay of the live window used as the reconciliation counterfactual; fixed seeded sequence of (timestamp, bar) tuples replayed through both pipelines; the baseline every drift number is compared to — (ch25, ch19)
- **Refit schedule / estimation schedule / `freeze_after` / fitted feature / refit cadence / `refit_boundaries`** — burn-in, fit on strict prefix, emit until the next refit, refit on the longer prefix (21 GARCH/ARIMA, 63 HMM/Kalman/SV), expanding window; `freeze_after` = pre-holdout count so holdout values come from the last pre-holdout parameters; `refit_boundaries(n_obs, burnin, refit_every)` — (cs_cme, cs_crypto, cs_etfs, cs_fx, cs_sp500opt, cs_useq)
- **Regime** — stretch of time over which the joint behaviour of returns is stable enough to treat as one environment; labels Risk-on / Caution / Crisis / Recovery (vol x direction) or Bear / Bull / High Vol / Calm from lagged 60-day trend and vol vs expanding median; regime heterogeneity = IC by VIX tercile (sign flip vs magnitude) — (ch01, ch16, ch19, ch07)
- **Regularization path / Ridge / LASSO / Elastic Net / `l1_ratio`** — coefficient trajectories as the penalty varies; L2 / L1 / mixed penalties — (ch11)
- **Research lock** — abandoned design that pre-registered the holdout lineage as a one-shot transaction — (cs_fx)
- **Reservation price** — mid shifted against inventory, p (1 - (q/q_max) sigma) — (ch21)
- **Response surface / robust region** — metric as a function of a swept parameter; parameters within `ROBUST_THRESHOLD_PCT` of the peak — (ch08)
- **Restatement / amendment / taxonomy drift** — later filing revising an earlier period (10-K/A, 13F amendments); concept names change (ASC 606 revenue) — (ch04)
- **Retrieve -> Extract -> Compute -> Narrate** — numeric RAG workflow keeping arithmetic in Python on typed, source-tagged values — (ch22)
- **Risk-on / Risk-off** — periods when investors add / shed risky exposure; the clusters with highest / lowest mean equity-index return — (ch01)
- **`risk_triggers`** — count of overlay firings; proof a control acted — (cs_useq)
- **RiskLimitError / SafeBroker** — exception raised when any control rejects an intent (measured value and threshold); wrapper implementing `AsyncBrokerProtocol` enforcing position/order/exposure/daily-loss caps, rate limits, staleness, kill switch, persisted `RiskState`, startup reconciliation — (ch25, lib_live)
- **Robustness value** — partial R^2 an unobserved confounder needs with treatment and outcome to zero the estimate — (ch15)
- **Rockafellar-Uryasev / OCE** — `CVaR_{1-q}(L) = min_w [w + E[(L-w)_+]/q]`; makes CVaR differentiable — (ch19)
- **`rolling_entropy` / `fourier_features` / wavelet decomposition / spectral entropy / dominant period / low-frequency ratio / Parseval / Welch's method** — binned distributional entropy; calendar sin/cos basis of the row index; multi-scale components (non-causal DWT); evenness of power; strongest bin; share in slowest bins; total power = sum of squared deviations; averaged periodograms — (ch09)
- **Rotation ambiguity** — orthogonal re-basis of correlated factors relabels contributions — (ch19)
- **Round-trip cost proxy / round trip** — `2*(per_share/mean_price)*1e4 + 2*median_half_spread_bps` per symbol; twice the per-leg cost (fee paid twice) — (cs_nq100, cs_crypto, cs_usfirm)
- **`rtype` / session date** — 32=1s, 33=1m, 34=1h, 35=1d; date a session ends (CME 16:00 CT; FX 17:00 New York value-date rollover); exchange-calendar trading day assigned to each bar, with session completion inserting missing sessions (forward-filled prices, zero volume); CME_FX session calendar vs FX evaluation calendar — (ch02, lib_data, cs_fx)
- **Run bars / tick test / Lee-Ready / tick rule** — close when the longer one-sided run exceeds expected length; sign of last price change, zeros carry forward; quote test with tick-test fallback; buyer-initiated if a trade prints above the previous trade (`effective_tick_rule`); signed flow = uptick volume - downtick volume — (ch03, ch08, ch18, lib_engineer)

### S

- **SAE** — supervised autoencoder: characteristics -> `n_factors` bottleneck -> return with decoder, aux and main heads (`alpha`, `aux_weight`); no factor interpretation; the family's control — (ch14, cs_etfs, cs_sp500eo, cs_usfirm, lib_models)
- **Scale-out** — tranche exits with stop ratcheted to break-even — (ch19)
- **Schema read** — `scan_parquet().collect_schema()` footer read — (ch09)
- **SDF / pricing kernel / SDFCheckpoint / weight-native** — M = 1 - omega^T R^e with factor F = omega^T R^e; firm-month weights pricing the cross-section under E[m R] = 0 (constant across assets) with macro context; the model outputs weights w_t whose return defines `M_{t+1} = 1 - w_t' R_{t+1}` (inference); `(phase, epoch)` with phase in {unconditional, moment, conditional}, legacy int offset by `n_epochs_unc`, epochs 0 / -1 = val-best loss / Sharpe sentinels — (ch14, cs_cme, cs_etfs, cs_sp500eo, cs_usfirm, lib_models)
- **Seal / seal test** — executable assertion that a construction reads neither the holdout nor the future: rebuild features with later dates withheld and diff shared rows (`assert_values_agree`) — (cs_cme, cs_crypto, cs_sp500eo)
- **Security master / sentence embedding / cosine similarity** — canonical entity records with identifiers and dated aliases; vector text representation; dot product of unit vectors — (ch04)
- **Selection bias / nested CV / 3σ heuristic / validation overfitting** — inflation of the best observed statistic because the selection step is not in the estimate; inner loop selects, outer scores; flag when the top score exceeds the median by > 3 inter-configuration SDs; selecting a configuration that fits the validation window's noise — (ch07, ch11, ch12)
- **Selective SSM / zero-order hold** — state space model whose B_t, C_t, Delta_t come from the current input; discretisation giving exp(Delta A) state decay — (ch13)
- **Sequential bootstrap / uniqueness / concurrency / N_eff / ESS** — resampling favouring candidates with high expected uniqueness; concurrency = labels alive at a bar; uniqueness = mean 1/concurrency over a label's life (share of its forward window no concurrent label spans); N_eff = sum of uniqueness weights ~ N/h for daily-sampled h-session labels (ETFs 20,017 of 418,362; 1.0 for a 1-month label at monthly sampling) — (ch07, cs_cme, cs_crypto, cs_etfs, cs_usfirm, lib_engineer)
- **Sequential detector / signed lead-lag / turbulence proxy** — reads one observation at a time and decides after each whether the stream still looks like its calibration; alert-to-nearest-stress distance with sign (positive = detector led); annualized 21-session rolling std of the cross-sectional median daily return, stress above the prior-year 80th percentile — (ch26)
- **Session-aware / session alignment / calendar-first splitting / symbol-session** — computations never cross `session_col` boundaries or calendar breaks; trading sessions or exchange trading days (`calendar_id`) as atomic fold units; the entity trailing windows are bounded by intraday — (lib_engineer, lib_diagnostic, cs_nq100)
- **Shadow mode / VirtualPortfolio** — real broker connection and prices, orders validated, logged and routed to `VirtualPortfolio` (weighted-average cost basis, flip handling), never sent; candidate runs on live data with no capital — (ch25, ch26, lib_live)
- **SHAP / masker / concentration risk / decision-relevant predictions / interaction values / H-statistic / Trade-SHAP / error pattern / separation score** — Shapley attributions (linear: phi_j = beta_j (x_j - x̄_j); TreeSHAP exact in O(T L D^2), base + sum = prediction); reference distribution SHAP integrates over; one feature's share of a row's \|SHAP\| (flag > 60%); top 20% by \|ŷ\|; main-effect/pairwise decomposition and Friedman-Popescu strength in [0, 1]; clustered SHAP vectors of losing trades characterized with BH-FDR tests and matched to hypothesis templates; centroid distance to nearest other cluster — (ch11, ch12, ch19, lib_diagnostic)
- **Shared-endpoint artifact** — feature ending at t and label starting at t share p_t — (ch07)
- **Shuffle diagnostic / step time** — permute days inside each input window and measure error delta and prediction distance; fastest warmed-up fwd+bwd+update on a fixed batch (CUDA events) — (ch13)
- **Sign consistency / stability selection** — share of non-zero folds (validation windows) agreeing on a coefficient's or IC's sign; bootstrap test of IC sign consistency; exploration floor 0.60 — (ch08, ch11, cs_cme, cs_etfs, cs_fx)
- **Sizing lag h** — `max(1, label horizon)` steps between the last known residual and the decision it sizes — (ch13, cs_cme, cs_crypto)
- **Slot strategy / stride** — fixed weight-per-slot book, per-symbol rolling-percentile entry, max hold, optional signal exit; `train_sequence_stride_horizons` spacing of DL training windows in label horizons — (cs_nq100)
- **Source of edge (SLOW / WRONG / RISK) / strategy family / trading setup / trial taxonomy** — economic reason a strategy pays (durability filter); structural class (feasibility filter); versioned bundle of tradability rules, decision schedule, admissible information, score-to-trade mapping, constraints and costs; strategy > trial family > trial > run — (ch06)
- **`source` (computed/reused)** — whether a backtest ran or was served from the registry — (cs_cme)
- **Split-aware / train-only preprocessing** — parameters learned on train only (scalers, quantiles, winsor bounds), locked, applied to validation/test, refit per fold — (ch07, lib_engineer)
- **Split-scoped identifier** — anonymous firm id persistent only within one released tensor block — (cs_usfirm)
- **Square-root law** — impact proportional to sigma sqrt(Q/ADV) (c sigma sqrt(q/ADV) p) — (ch18, ch21)
- **Stability / stacked vs wide format** — R^2 of a linear fit to the cumulative value curve; one row per `(timestamp, symbol)` (canonical) vs one column per `(value, symbol)` — (ch17, lib_data)
- **Stage / `STAGE_SEQUENCE`** — `signal` (equal-weight top-k baseline) -> `allocation` (alternative sizing on the shortlist) -> `risk_overlay` (one exit rule on the rank-1 parent) -> `cost_sensitivity` (bps grid on the carrier) -> `holdout`, in `backtest_runs` — (ch26, cs_cme, cs_etfs, cs_useq)
- **Staleness / stale cap** — share of consecutive observations identical to the previous one (ceiling 0.50); age vs publication schedule; N consecutive identical closes (default 5) tolerated before nulling; dataset staleness = no update within `stale_days` (7) — (ch02, cs_fx, cs_sp500eo, cs_etfs, cs_nq100, lib_data)
- **Statement count** — AST count of statement nodes in a pipeline function; the only comparable framework metric — (ch24)
- **Stationary / unit root** — moments independent of sample position / today = yesterday + non-decaying shock — (ch09)
- **Statistically unresolved / three decay mechanisms / win-win quadrant** — holdout Sharpe CI spans zero; prediction-quality drift, portfolio-translation drift, structural break; overlay improves both Sharpe and max drawdown (the only quadrant justifying deployment) — (ch20)
- **Stochastic volatility / particle filter / MCMC diagnostics** — latent AR(1) or random-walk log-vol (`sigma_eta` step sd); ~1000 candidate latent-vol values reweighted by each day's return; divergences / R-hat / ESS, Monte Carlo SE, non-centered parameterisation, interval coverage — (ch09, cs_sp500opt)
- **StopFillMode / exit-first ordering** — assumed fill price when a stop/target is breached inside a bar (stop price, close, bar extreme, next open); position-reducing orders fill before entries — (lib_backtest)
- **Stylized facts / volatility clustering** — fat tails, volatility clustering, leverage effect, negative skew (Cont 2001); positive autocorrelation of squared returns — (ch05, ch21)
- **Supervisor / confidence-gated override / adversarial debate** — identifies disagreements, runs bounded cutoff-respecting searches, returns p_yes + confidence (replaces at high, blends at medium 0.4, ignored at low); bull and bear prompts over bounded rounds, gap = \|p_bull - p_bear\|, midpoint blended at `DEBATE_WEIGHT`, stops below 0.05 — (ch24)
- **Survivorship bias / delisting return (`DLRET`, `DLSTCD`) / terminal return** — overstatement from excluding leavers; whether entries (IPOs) are recorded; CRSP delisting fields — (ch02)
- **Swap points** — overnight financing component of FX cost; taxonomy item, never priced — (cs_fx)
- **Synchronous diffusion** — all frontier nodes propagate in one round, so results are loop-order independent — (ch23)
- **Systematic edge / 5-stage ML4T workflow** — durable advantage from a repeatable, bias-resistant research process rather than any single strategy; the book's end-to-end pipeline as an alpha-factory blueprint — (ch27)

### T

- **TA-Lib compatible / Wilder's smoothing** — numerically identical to TA-Lib (init method, `ddof=0`, unstable periods); RSI/ATR smoothing distinct from EWM span — (lib_engineer, ch07)
- **TabM / TabPFN** — MLP ensemble with a shared two-layer backbone and per-member rank-one scaling vectors plus own output heads, trained jointly (registry family `tabular_dl`); prior-data fitted zero-shot tabular foundation model (gated weights) — (ch12, cs_cme, cs_etfs, cs_sp500eo, cs_sp500opt, cs_useq, cs_usfirm)
- **Target intent / child intent** — canonical idempotent portfolio target registered pre-open, lowered to child orders at `process_opening` (backtest) or by `LiveStrategyRuntime` (live), reconciled against fills — (lib_backtest, lib_live)
- **Tearsheet template / theme** — persona layouts `quant_trader | hedge_fund | risk_manager | full`; `default | dark | print | presentation` — (lib_diagnostic)
- **Technical vs statistical failure / three-layer governance** — same inputs produce different outputs (pipeline divergence) vs same outputs no longer predict returns (decay); detection, response, automated safety — (ch26)
- **Temporary / permanent impact / persistence fraction** — liquidity concession that decays vs information revealed that persists; the fraction carrying the permanent part across child orders — (ch18, ch21)
- **Term sheet** — pre-registered pass/fail criteria a strategy must meet — (ch16)
- **TF-IDF** — term frequency x inverse document frequency weighting — (ch10)
- **Three-stage framework (latent factors)** — Stage 1 extract factors/loadings; Stage 2 forecast the factor premium; Stage 3 map to asset signals — (ch14)
- **Tick volume / direct / indirect / cross pair** — FX activity proxy; USD-quoted, USD-base and non-USD pairs — (ch02)
- **Tiered commission / `VolumeShareSlippage`** — rate schedule by notional thresholds; slippage rising with order share of bar volume via `impact_factor` — (ch18)
- **Time-mixing / feature-mixing / pre-normalisation / weight normalisation** — shared linear map across the time axis / MLP across features at each timestep / normalising a sublayer's input before the residual branch / weights reparameterised by direction and magnitude (no batch statistics) — (ch13)
- **Timing risk / risk aversion lambda** — exposure of the unexecuted position to price moves; weight on variance in `E[C] + lambda V[C]` (unit-dependent; labelled by liquidation half-life) — (ch18)
- **TracingLLMClient** — wrapper capturing every prompt/response at the LLM boundary, one per agent — (ch24)
- **Traded folds** — folds with at least one non-zero return day; admission requires all declared folds — (cs_crypto)
- **Trailing percentile rule / cross-sectional rule** — hold when a score beats the p-th percentile of its own last L scores / is in the top (100-p)% of symbols quoted that date — (ch16)
- **Trailing stop** — stop that ratchets with favorable price movement (% from running peak) — (lib_engineer, cs_fx)
- **Trend scanning** — pick the forward window (5-20; min..max by step) with max \|t\| of a price-on-time regression; label = sign of slope — (ch07, lib_engineer)
- **Triage ledger / PROCEED / REVISE / STOP** — per-feature decision record with evidence and a `note` of the route (`fdr_significant` / `stable_and_above_threshold`); a record, not a filter; coverage >= 0.70, staleness <= 0.50; only STOP judges the column (Table 7.2) — (ch07, ch20, cs_cme, cs_crypto, cs_etfs, cs_fx, cs_nq100, cs_sp500opt, cs_usfirm)
- **Triple barrier** — first of take-profit, stop-loss or vertical (time / max-holding) barrier decides the label (`barrier_hit`); outcomes -1 SL, 0 timeout, 1 TP; hit rate, precision/recall/lift, profit factor = sum(TP returns)/\|sum(SL returns)\|, time-to-target (`label_bars`) — (ch07, lib_engineer, lib_diagnostic)
- **Trust flag / evidence gate** — per-chunk provenance marker (`trusted: bool`); refuse when no trusted chunk remains — (ch22)
- **Turnover (proxy)** — mean absolute change in predictions between periods; fraction of names changing quantile per period; units of position change over an RL episode; fraction of book replaced per rebalance — (ch12, lib_diagnostic, ch21, cs_crypto)
- **TWAP / VWAP / Almgren-Chriss** — equal quantity every step (conditions on nothing) / shares proportional to expected volume; also the benchmarks — (ch18, ch21)
- **Two-speed covariance** — fast EWMA residual volatility plus slow EWMA factor correlation structure — (ch14)

### U-V

- **Unified framework** — one `Strategy` class executed by both `ml4t.backtest.Engine` and `ml4t.live.LiveEngine` — (ch25)
- **UNRATE / DFF / T10Y2Y / CPIAUCSL / yield-curve inversion / VIX** — unemployment rate; effective fed funds rate; 10Y-2Y Treasury slope (negative = inverted; preceded most post-war / every US recession since 1970); CPI; 30-day implied S&P 500 vol, annualized — (ch01, ch04)
- **Up / down capture** — strategy's mean return in benchmark-up (down) months over the benchmark's — (ch17)
- **Validation window / validation-to-holdout decay** — sessions a walk-forward fold is scored on just after its cut-off; validation-fold IC minus nested-holdout IC of the selected config — (ch26, ch12)
- **VaR / CVaR (5%)** — loss exceeded by the worst day in twenty / mean loss on those days; empirical loss quantile / mean loss beyond it; expected shortfall = negative mean P&L over the worst q fraction of paths — (ch09, ch16, ch17, ch19, ch21)
- **Variance ratio (Lo-MacKinlay)** — Var(q-period)/(q Var(1-period)); 1 under a random walk, > 1 trending; also a rolling feature — (ch03, ch08, lib_engineer)
- **Variance rescaling** — post-hoc multiplication of normalised samples by train_std / synth_std when the ratio < 0.9 — (ch05)
- **Vectorized vs sequential backtest / vectorized forward-return path** — precomputed aligned arrays x returns vs a bar-by-bar loop carrying positions, cash, fills, equity; one weight vector per rebalance times the realised return, no intra-period prices or trade ledger — (ch16, cs_usfirm)
- **Vintage** — the value of a series as it stood on a past date (ALFRED archives FRED vintages; `vintage_date`); MLOps: the per-fold fitted version of a model-based feature, selected by date not id — (ch01, ch04, ch26, lib_data)
- **vol_scale / volatility targeting** — per-asset multiplier applied to weights before net returns; scale each position by sigma* / (sqrt(252) sigma_hat N) so each contributes equal ex-ante risk — (lib_models, ch17)

### W-Z

- **Walk-forward CV / nested walk-forward / walk-forward fitting / walk-forward HPO** — chronological folds, training precedes validation (expanding or rolling); outer loop over test windows with inner walk-forward for hyperparameter selection; each date's estimate uses only prior data; objective averaged over temporal folds ending before the holdout; the authoritative comparison — (ch01, ch06, ch11, ch12, ch13, ch17)
- **Warm-cache read / materialisation / column projection / range query / anti-join / durable-to-queryable / splayed table / hypertable / witness / geometric-mean speedup / predicate pushdown** — storage-benchmark vocabulary for the ch02 database comparison — (ch02)
- **Wasserstein distance (EMD) / `wass_dist_ratio`** — minimum transport work between distributions; bin-free, order-aware; distance from the recent 21-session return window to the nearest of 2 centroids (preferred over the state label) — (ch19, cs_useq, lib_diagnostic)
- **Watch clock vs decision clock** — 1-minute engine price feed vs the 15-minute decision cadence — (cs_nq100)
- **Watchdog** — engine task monitoring feed silence / broker health, optionally recovering — (lib_live)
- **Winsorization / winsorized label** — clipping values beyond percentile bounds onto those bounds (1st/99th; per-date cross-sectional quantiles measured before applying, on training rows) — (ch07, ch11, cs_etfs, cs_useq, cs_usfirm, lib_engineer)
- **Working / short-term / long-term memory** — context window / durable run state / cross-run retrieval (RAG) — (ch24)
- **Zero-shot forecasting / negative transfer** — a pretrained model forecasts a new series with no fitting; pretraining data that worsens downstream performance — (ch13)

## Reader uncertainty carried from the notes

- us_firm_characteristics: 57 features = 46 released + 11 constructed (inference); the price > $5 / ADV > $1M screen is presumed provider construction; `n_factors`, epoch budgets, Newey-West lag rule, placebo count, cost levels, `holdout_conformal_embargo_steps`, bootstrap parameters, `dsr_mp` / `dsr_er` definitions and `get_top_n_predictions` are not visible — (cs_usfirm)
- ml4t-live: `ContinuityDisposition` members, `FeedQueueSnapshot` fields and the `MarketEvent` payload module are unconfirmed; the 30 s order timeout appears only in a docstring; recommended live-stage limit values are inferences — (lib_live)
- ml4t-models: training-loop bodies unread (optimizer, exact `_sae_loss` / `robust_sharpe_loss`, SDF GAN alternation); `CrossSectionBatch.factor_returns` role unverified; the 3rd-edition chapter mapping is inferred — (lib_models)
- Terms the README names without defining: structural break / data drift / concept drift / online detection (ch01), scoping invariants, implementability check, the 5-stage workflow's stage names (ch27); PPO / DQN / SAC are not defined anywhere in the notes.

## Related references

- `workflow.md` — the stages this vocabulary attaches to (signal -> allocation -> risk overlay -> cost sensitivity -> holdout).
- `guardrails.md` — purge/embargo, PIT, seal, holdout and multiple-testing terms as enforced rules.
- `decision_rules.md` — triage (PROCEED/REVISE/STOP), gates, kill conditions and thresholds quoted here.
- `evidence.md` — where the numeric findings behind terms such as breakeven cost, holdout decay and DSR come from.
- `companion_repo.md` — notebook paths for the chapter glossaries (e.g. `07_defining_the_learning_task/03_label_methods`).
- `chapters/07_defining_the_learning_task.md` — IC/HAC/FDR/RAS/DSR/PBO definitions in full; `chapters/16_strategy_simulation.md` — backtest and Sharpe-inference terms; `chapters/18_transaction_costs.md` — cost and impact vocabulary; `chapters/26_mlops_governance.md` — drift, registry and lineage terms.
- `case_studies/*.md` — per-study meanings of carrier, population, checkpoint, breadth floors and timing contracts; `libraries/*.md` — class, config-key and enum names (`ExecutionMode`, `SafeBroker`, `SDFCheckpoint`, `ohlc_mode`).

### Further reading (as named in the notes)

- López de Prado, *Advances in Financial Machine Learning* (AFML) — bars, triple barrier, uniqueness, CPCV/PBO, DSR.
- Cont (2001) — stylized facts of asset returns.
- Lo & MacKinlay — variance-ratio test; Lo — Sharpe annualization under autocorrelation; Mertens — Sharpe variance with skew/kurtosis.
- Ledoit & Wolf — covariance shrinkage; Markowitz — mean-variance; Kelly — growth-optimal sizing.
- Newey & West; Driscoll & Kraay; Fama & MacBeth — robust and panel inference. Benjamini & Hochberg; Holm; Massart — multiple testing bounds. Harvey et al. — t > 3 factor threshold.
- Almgren & Chriss — optimal execution; Kyle — lambda; Amihud; Roll; Corwin & Schultz; Lee & Ready — microstructure estimators. Whalley & Wilmott — no-trade band; Halperin — QLBS.
- Garman & Klass; Parkinson; Rogers & Satchell; Yang & Zhang — range vol estimators; Corsi — HAR. Rockafellar & Uryasev — CVaR optimization; Cornish & Fisher; Kupiec; Cantelli.
- Kelly & Pruitt (IPCA); Gu, Kelly & Xiu (CAE); Chen, Pelger & Zhu (SDF); Baik, Ben Arous & Péché (BBP edge).
- Runge (PCMCI); Zheng et al. (NOTEARS); Hyvärinen et al. (VAR-LiNGAM); Chernozhukov et al. (DML); Brodersen et al. (BSTS). O'Donovan & Yu — option cost cascade. ADIA (2025) — CV regime gate.
- FinDKG; FinReflectKG; FinMTEB; GReaT; TimeGAN; Sig-WGAN; Opacus (DP-SGD); Chronos; TinyTimeMixer; N-BEATS; PatchTST; iTransformer; TSMixer; TabM; TabPFN; LambdaMART; Optuna (TPE, NSGA-II, fANOVA).
