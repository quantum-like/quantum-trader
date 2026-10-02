# Case study: ETFs (100 multi-asset ETFs, daily data, monthly decisions, fwd_ret_21d)

> 100 ETFs across 9 categories (broad equity, US sectors, countries, government/corporate bonds, currencies, commodities, real estate, factor baskets), daily Yahoo OHLCV 2006-2025, month-end close decisions executed at the next open, long-only top-k equal weight, 5-15 bps per leg. The README calls it "the most cost-favorable configuration in the book", which is why it hosts the broadest model-family comparison (linear, GBM, TabM, LSTM/TSMixer/NLinear, PCA/IPCA/CAE/SDF/SAE, causal DML) on one feature panel. Headline lesson: the gap between IC and Sharpe. The family that ranks best by rank correlation need not rank best by traded Sharpe, selection is on validation backtest Sharpe only, and the cross-stage winner was a risk-overlay run, not the allocation leader ("portfolio construction mediates prediction quality", Ch16-Ch20). Second lesson: on ~100 already-diversified instruments with a daily-sampled monthly label, no single feature clears BH-FDR, raw momentum fails its own HAC test, and the boosted population does not beat ridge; evidence is counted in tens of independent blocks, not hundreds of thousands of rows.

Repo: `case_studies/etfs/` (notebooks `01`..`20` as `.py`/`.ipynb`), contract in `case_studies/etfs/config/setup.yaml`, training menus in `case_studies/etfs/config/training/{fwd_ret_21d,fwd_ret_5d}.yaml`. Run `uv run python case_studies/etfs/NN_name.py` from repo root in numeric order (11a-11e before `11_latent_factors.py`). Results are deliberately NOT restated in the README: read `20_strategy_analysis` and `run_log/registry.db`. Published bundles: `uv run python scripts/download_artifacts.py --cs etfs` fetches `v3.1.0-artifacts` (current); `v3.0.0-artifacts` is the first generation and its hashes do not resolve in 3.1.

## Setup contract

Every key below is read by the notebooks from `config/setup.yaml` (`strategy_id: etfs`, `setup_version: v1`). A threshold retyped in code is a second source of truth; do not do it.

| Area | Key(s) | Value | Note |
|---|---|---|---|
| Universe | `universe.assets`, `n_assets` | 100 tickers: ACWI ACWX AGG BIL BND BNDX DBA DBC DIA DVY EEM EFA EMB EWA EWC EWG EWH EWI EWJ EWL EWN EWP EWQ EWT EWU EWW EWY EWZ EZA FXB FXE FXI FXY GLD GOVT GSG HYG IAU IBB IEF IEFA IEMG IJR INDA ITA ITB IVE IVW IWM IYR JNK KRE LQD MCHI MDY MTUM MUB OIH PPLT QQQ QUAL RSP SCHD SDY SHY SLV SMH SOXX SPY THD TIP TLT UNG USMV USO UUP VCSH VEA VGK VIG VLUE VNQ VTI VTV VUG VWO XBI XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY XME XRT | list is backward-looking (survivorship-biased), declared in `eligibility_note` |
| Eligibility | `eligibility_rule: point_in_time_adv_10m_annual`, `eligibility_file: eligibility.csv` | fund admitted to year Y iff year Y-1 avg daily turnover >= $10M AND >= 200 sessions | the $10M floor is prose only in setup.yaml; `01` hardcodes `ADV_THRESHOLD = 10e6`, `MIN_SESSIONS_PER_YEAR = 200`; not inflation-adjusted |
| Data | Yahoo daily OHLCV, 2006-01-01 to 2025-12-31 | `load_etfs()` adjusted (returns), `load_etfs_unadjusted()` traded (dollars, costs) | Yahoo restates splits but not distributions; splits multiplied back out in the loader |
| Decision | `decision.cadence: monthly_month_end`, `snapshot: close`, `execution_delay: next_bar_open` | decide at month-end close, fill next open | `cadence_by_label: {fwd_ret_5d: weekly_friday_close}` (5d label on the monthly grid was carried ~21 sessions) |
| Labels | `labels.primary: fwd_ret_21d`, `buffer: 21D`, `variants: [fwd_ret_5d]`, `variant_buffers: {fwd_ret_5d: 5D}` | close-to-close forward return over h sessions on adjusted close | `rebalance_step: {fwd_ret_21d: 1, fwd_ret_5d: 1}` (vectorized-backtest thinning) |
| CV | `evaluation.n_splits: 8`, `train_size: 10Y`, `val_size: 1Y`, `holdout_start: 2024-01-01`, `holdout_end: 2025-12-31`, `calendar: NYSE`, `periods_per_year: 252` | purge gap = label horizon (21 sessions); last val window ends 2023-11-29; fold 0 earliest | holdout opened only by 18/19/20 |
| Mapping | `mapping.class: long_only_rank_and_rebalance`, `position_state_space: long_only`, `entry_logic: rank_selection_top_n`, `sizing: equal_weight` | long-only top-k, equal weight baseline | ETF borrow is expensive/impossible |
| Execution | `execution.initial_cash: 100_000`, `share_type: integer`, `allocator_lookback: 63` | IBKR retail cohort; whole shares; 63 daily bars (~3 months) for every moment-based allocator | read via `get_backtest_config()`; changing cash/share_type invalidates every `backtest_hash` |
| Costs | `costs.class: material`, `model: per_share_plus_spread`, `per_share: 0.0035`, `minimum: 0.0`, `spread_convention: half_spread` | IBKR Pro Tiered top tier + tier-assigned half-spread | `asset_spreads`: 0.005 SPY/QQQ/IWM/EFA/EEM/DIA/VTI; 0.01 XLK/XLF/XLV/XLE/XLY/XLI/XLP/XLU/XLB/XLRE/XLC; `default_half_spread_usd: 0.02` else. Yahoo has no bid/ask |
| Rebalance | `backtest.rebalance.default: {min_weight_change: 0.005, min_trade_value: 100.0}`; `benchmark: {0.0, 0.0}` | skip a trade when both thresholds unmet | benchmark thresholds are 0 so 1/N rebalances at all |
| Benchmark | equal weight of every eligible fund (Ch16 baseline; 1/N over the universe in 20) | falsification target for every signal | not repeated among allocators |
| Sweep funnel | `backtest.sweep.top_n_predictions: {signal: 0, allocation: 10, cost_sensitivity: 1, risk_overlay: 1}`, `checkpoints_per_config: 1`, `expensive_allocators_skip: false` | all predictions -> top 10 configs (one checkpoint each) -> top 1 per label | allocation re-sweeps the whole top_k grid per advancing config |
| Concentration | `top_k_grid: {fwd_ret_21d: [5,10,20], fwd_ret_5d: [5,10,20]}` | 5/10/20% of the ~99-name median cross-section (p10 = 82) | `percentile_grid`/`quantile_grid` commented out |
| Allocators | `allocators`: score_weighted, inverse_vol, risk_parity, mvo_ledoit_wolf, hrp, conformal_weighted | no max-weight cap; MVO uses Ledoit-Wolf shrinkage | conformal allocators have their own calibration windows in `conformal.py` |
| Cost grids | `cost_grid_bps: [0,1,2,3,5,7,10,15,20,30,50]`; `cost_grid_half_spread_usd: [0.0,0.005,0.01,0.025,0.05,0.10]` | bps regime (comparator) and per-share regime (declared headline) | |
| Risk overlays | `risk_controls.position`: stop_loss 3/5/10/15%; trailing_stop 1/2/3/5/10/15/20%; time_exit 10/20/40 bars | 14 overlays | |
| GBM | `modeling.gbm: {libraries: [lightgbm], preset: default, device: cpu, max_bin: 255}` | | 63 was the GPU default carried over; quartered the bins |
| Latent factors | `modeling.latent_factors.persistent_entities: true`; `macro_context: {source: alfred_initial_release, policy: alfred_initial_release_close_lagged, version: v1, series: dgs1/2/3/5/7/10/20/30, vixcls, YIELD_CURVE_SLOPE, YIELD_CURVE_5_10, availability_lag_days: 1, alignment: backward_asof}`; `model_kwargs.ipca: {max_iter: 1000, tol: 0.00001, factor_ridge: 0.01, gamma_ridge: 0.01}`; `model_kwargs.sdf: {checkpoint_epochs: [256,512,768,1024], beta_checkpoint_epochs: [256], beta_default_checkpoint: 256}` | | IPCA folds converge at 53-110 alternations; cap 100 fell inside that distribution |
| Causal | `causal: {treatment: skip_recent_6_1, treatment_window: 126, confounders: [vol_21d, vol_126d, regime, yield_curve_slope], method: walk_forward_dml}` | treatment = `close.shift(21)/close.shift(126) - 1` | placebo block must cover the 126-session construction window |
| Features | `features.windows: {momentum: [5,10,21,42,63,126,189,252], volatility: [21,63,126,252], skip_recent: 21, drawdown: [63,126], volume: [21,63]}`; `ranked: [ret_126d, sharpe_126d, vol_63d]`; `regime_threshold: 0.005`; `oscillators: {rsi: [7,14], macd 12/26, adx 14, cci [14,21], stochastic 14, aroon 25, natr 14, choppiness 14, hurst 100, sma [50,200], ema 26, bollinger 20}`; `state: {obv_zscore 63, positive_share 63, extremes 252, correlation 63, curve_zscore 252}` | | `families[]` register: name, pattern, role, hypothesis, inputs, lookback, lag, frame, representation, failure_mode |
| Model-based | `model_based.hmm: {n_states: 2, burnin: 756, refit_every: 63, n_restarts: 10}`; `model_based.garch: {burnin: 504, refit_every: 21}` | counts in trading sessions; schedule, not fold, bounds a fitted feature | artifact carries no fold column |

Feature-family register (`features.families`, all lag 0 except yield curve):

| Family | Pattern | Role | Lookback | Failure mode |
|---|---|---|---|---|
| momentum | `ret_*d\|skip_recent_*\|mom_accel_*` | signal | 252 | reverses at shortest horizons |
| risk-adjusted momentum | `sharpe_*d` | signal | 252 | unbounded when dispersion -> 0 |
| volatility | `vol_[0-9]*d\|vol_ratio_short\|vol_ratio_medium` | state | 252 | lags a shock by ~half its window |
| oscillator and trend | `rsi_*\|macd_*\|adx_*\|cci_*\|stoch_*\|aroon_*\|sma_ratio_*\|ema_ratio_*\|bb_pctb_*` | signal | 200 | saturates in a sustained trend |
| range and drawdown | `natr_*\|chop_*\|hurst_*\|max_dd_*` | state | 126 | Hurst needs a long window |
| volume | `vol_ratio_[0-9]*d\|obv_*` | state | 63 | index events put one date orders of magnitude out |
| extremes and consistency | `pct_positive_*\|dist_52w_*` | signal | 252 | piles up at one in a trend |
| cross-sectional position | `*_rank` | signal (cross-section frame) | 126 | discards level |
| cross-asset regime | `corr_spy_tlt_*` | state | 63 | one pair stands in for the universe |
| yield curve | `regime\|yield_curve_*` | state (macro frame), **lag 1** | 252 | reads revised Treasury history, not initial release |

Training menus (`config/training/*.yaml`; comment out a line to skip, add a preset name to extend; presets live in `case_studies/config/{model_type}/`):

| Family | Configs (fwd_ret_21d) | Checkpoints | Candidates per label |
|---|---|---|---|
| linear | ols; ridge_a{0.001,0.01,0.1,1,10,100,1e3,1e4,1e5,1e6,1e7}; lasso_f{0.015,0.03,0.08,0.2,0.35,0.5,0.7,0.85}; enet_f{same 8} = 28 | 1 | 28 |
| gbm | {default, leaves_7, leaves_15, leaves_31, leaves_63} x {mse, mae, huber} = 15 | 500 iterations, every 50 = 10 | 150 |
| tabular_dl | tabm_s (hidden 64, 4 members), tabm_m (128, 8), tabm_l (256, 16); dropout 0.1, batch 4096 | 200 epochs, every 25 = 8 | 24 |
| deep_learning | nlinear, lstm_h64 (hidden 64, 2 layers), tsmixer (hidden_dim 32, 2 blocks); lookback 60, dropout 0.1, batch 2048 (5d menu: nlinear, lstm_h64 only) | 100 epochs, every 5 = 20 | 60 |
| latent_factors | pca, ipca (`checkpoint_interval: 0`), cae, sae (50 epochs, every 5), sdf; `n_factors: 5` for all | 1 / 1 / 10 / 10 / 4 (+ beta 256) | |
| causal_dml | dml (`n_folds: 5`, `n_placebo: 100`, `seed: 42`; `max_samples: 50000` ignored by `resolve_causal_request`) | | 1 `causal_runs` row |

## Pipeline stages

| Stage | Notebook | Chapter | What it does | What it writes |
|---|---|---|---|---|
| Feasibility | `case_studies/etfs/01_feasibility_analysis` | Ch6 (6.2-6.6) | breadth on decision dates, PIT eligibility, move vs own round trip, panel ACF of monthly returns, fold demo | `eligibility.csv` (symbol, eligible_year) |
| Labels | `02_labels` | Ch7 (7.2) | `fwd_ret_21d`, `fwd_ret_5d` on full history; window asserts; endpoint-sealed dev window; ESS; momentum baseline floor | `labels/fwd_ret_21d.parquet`, `labels/fwd_ret_5d.parquet` + `.digest.json` |
| Features | `03_financial_features` | Ch8 (8.1-8.6) | 57 arithmetic features from the register; gate; within-date clip/rank; warmup audit; holdout-withheld rebuild test | `features/financial.parquet` + sidecar |
| Temporal | `04_model_based_features` | Ch9 | HMM regime (SPY), FFD of 10 reference ETFs, per-ETF GARCH(1,1) on a refit schedule frozen at holdout | `features/model_based.parquet` + sidecar (14 columns) |
| Evaluation | `05_evaluation` | Ch7 (7.3, 7.4), 8.6 | univariate IC screen with HAC SE, BH-FDR, fold sign consistency, quantile shape, redundancy, triage | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` |
| Linear | `06_linear` | Ch11; Ch6 6.7 | 28 configs on both labels; population `etfs-linear-validation-v1` | registry rows; `run_log/training/{hash}/`, `run_log/predictions/{hash}/` |
| GBM | `07_gbm` | Ch12 (12.2, 12.3) | 15 LightGBM configs x 10 checkpoints; `etfs-gbm-validation-v1` | boosters, `learning_curves.parquet`, `fold_metrics.parquet` |
| Tabular DL | `08_tabular_dl` | Ch12 (12.3) | TabM x 3 x 8 checkpoints; `etfs-tabular_dl-validation-v1`; published device cuda | `run_log/training/tabular_dl/` |
| LSTM | `09_dl_lstm` | Ch13 (13.8) | `lstm_h64`; `etfs-lstm-validation-v1` | `run_log/training/deep_learning/` |
| TSMixer | `10_dl_tsmixer` | Ch13 (13.8) | `tsmixer`; `etfs-tsmixer-validation-v1` | same |
| NLinear | `10a_dl_nlinear` | Ch13 (13.8) | `nlinear`, the bar LSTM/TSMixer must beat; republishes full-coverage subset; menu-vs-registry reconciliation | same |
| Latent index | `11_latent_factors` | Ch14 (14.5-14.7), Ch13 13.3 | reads each 11x `MODEL_NAME` via AST; checks menu coverage; prints best registered IC | nothing |
| PCA | `11a_pca` | Ch14 (14.2) | persistent-ID baseline; reads no features | runs + prediction sets |
| IPCA | `11b_ipca` | Ch14 (14.5) | exposures = shared linear map of features; ALS; convergence guard | 1 run + 1 prediction set per label |
| CAE | `11c_conditional_autoencoder` | Ch14 (14.6) | IPCA with `hidden_units=(32,)` network; lr 1e-3 | 10 checkpoints per label |
| SDF | `11d_stochastic_discount_factor` | Ch14 (14.7) | pricing objective; two networks; 11 macro series as context | 4 conditional checkpoints per label |
| SAE | `11e_supervised_autoencoder` | Ch14 (14.7) | encoder (896,448,448,256) -> 96 bottleneck -> head; lr 1e-4; noise 0.035 | 10 checkpoints per label |
| Causal DML | `12_causal_dml` | Ch15 (15.6) | effect of `skip_recent_6_1` on `fwd_ret_21d` given confounders; Driscoll-Kraay SE; block-permutation refutation | 1 row in `causal_runs` |
| Model analysis | `13_model_analysis` | Ch11-15 | coverage map; forest plot (Driscoll-Kraay 95% CI); fold heatmap; family agreement; horizon comparison; supervision ordering | nothing |
| Backtest | `14_backtest` | Ch16 (16.4-16.8) | engine test on a no-information signal; every prediction set x every entry scheme vs equal weight | `backtest_runs` rows at `stage='signal'`; `daily_returns/weights/trades/fills/equity/portfolio_state.parquet`, `spec.json` under `run_log/backtest/{hash}/` |
| Portfolio | `15_portfolio_management` | Ch17 (17.2-17.8) | top-10 signal leaders x `top_k` grid x 6 allocators | rows at `stage='allocation'` |
| Risk | `16_risk_management` | Ch19 (19.3-19.6) | stop-loss / trailing / time-exit grid on allocation leaders; trailing calibrated from adverse excursions | rows at `stage='risk_overlay'` (no un-overlaid row) |
| Costs | `17_costs` | Ch18 (18.2-18.5) | re-prices cross-stage leaders under bps grid and half-spread grid; last stage that selects | rows at `stage='cost_sensitivity'` |
| Holdout predictions | `18_holdout_predictions` | Ch20 | `resolve_solvent_carrier`; refit with `train_end` one label horizon before holdout | 1 `training_runs` + 1 `prediction_sets` row at `split='holdout'` |
| Holdout backtest | `19_holdout_backtest` | Ch20 | trades holdout predictions with registered k / allocator / overlay / cadence / cost unchanged | 1 `backtest_runs` row at `stage='holdout'` |
| Strategy analysis | `20_strategy_analysis` | Ch20 (20.1) | block/paired bootstrap, PSR/DSR, stage transitions, 1/N benchmark, FF5+MOM with placebo, concentration curve | `results/strategy_assessment.json`, `20_strategy_synthesis/output/etfs/etfs_tearsheet.html`, derived tables `cohort_metrics`, `backtest_paired_metrics` |

Registry: `run_log/registry.db` tables `training_runs`, `prediction_sets` (`split` in {validation, holdout}), `backtest_runs` (`stage` in signal -> allocation -> risk_overlay -> cost_sensitivity -> holdout), `causal_runs`, `cohort_metrics`, `backtest_paired_metrics`. Uncertainty from the notes: README says 04 writes "ARIMA, HMM, spectral" and 07 uses Optuna; the notebook text (HMM, fracdiff, GARCH; declared 15-config grid) is more likely current (inference).

## Design decisions and why

| Decision | Why | Where |
|---|---|---|
| Feasibility before any model | only a cost failure (typical move < round trip) can stop the study in 01; the other two stops (ranking unrelated to forward returns; 1/N beats the ranking per unit risk) are measured in 02/05 and 14 where their evidence exists | 01 |
| Breadth over depth | a rank-and-hold book needs many funds quoting at once; floor = `max(top_k_grid)` = 20 on every decision date, counted on decision dates not over the sample | 01 B.2 |
| PIT annual eligibility from prior-year turnover | selecting on whole-sample turnover admits exactly the funds that stayed liquid; "the difference between a backtest and a rehearsal" | 01 |
| Returns from adjusted close, dollars from traded close | early adjusted close "is not the price anyone paid"; turnover floor and per-share costs on the traded series | 01, 03 |
| Per-share cost expressed as a fraction of each fund's own price | 2c is ~10 bps on $20 and ~0.4 bps on $500; divide each move by its own round trip so break-even = 1 for every fund | 01 B.3 |
| Monthly cadence | median 21-session move is 46x the median round trip and within-fund monthly autocorrelation is ~0: nothing fast-fading demands faster trading | 01 B.4/B.5 |
| Long-only, equal weight baseline | short legs would measure borrow cost; risk-weighting folds a covariance estimate into the result so the ranking's contribution is inseparable; Ch17 compares allocators with the ranking fixed | 01 C.3, 15 |
| Label as a formula over tradable prices, dev window cut on label ENDPOINT | a row observed a week before the holdout resolves inside it and is a holdout row | 02 E, 05 seal |
| Baseline floor fixed before features exist | raw 126-session momentum IC on the PIT-eligible panel is the bar every engineered feature must beat; same rows, same `min_obs` | 02 G |
| Timing contract (lookback, lag) declared in the register | warmup audit, timing figure and code read one declaration; yield curve is the one lag-1 family | 03 |
| Ranked features carried as within-date percentiles as well as levels | raw returns drift with period volatility; representation matters as much as quantity | 03 C.7 |
| Per-entity trailing stats before the gate; clip/rank after | a trailing mean must see every bar the fund traded; an untradeable fund must not move a tradeable fund's percentile | 03 C.5-C.8 |
| Refit schedule instead of fold-fitted features | a fold model fitted on the whole training window gives early rows parameters from their own future; schedule = burn-in, fit on prefix, emit until next refit; filtered inference only; freeze at holdout | 04 |
| FFD orders fixed in advance (0.4 equities/gold/REIT, 0.5 bonds/credit) | estimating d reads the data; fix what you can, then test with ADF on development history only | 04 C.2 |
| Report search size beside p-values; exploration arm with stated threshold | BH over a search this wide can leave nothing; IC >= 0.01 is "a stated judgement, not a quantity derived from the data" | 05 |
| Every checkpoint registered as its own prediction set | "reporting the maximum of 150 numbers as though it were one"; a checkpoint is part of the configuration | 07, 08, 09-10a, 11c-11e |
| Selection only on validation backtest Sharpe in 14 | "IC ranks nothing anywhere in this pipeline"; IC ignores cost and turnover; orderings disagree | 06-17 |
| Immutable named populations declared before fitting | narrowed run, device change, preset edit each create a new identity; `supersedes` is the only lineage | 06-10a |
| NLinear as the sequence-model bar | "the smallest thing that still counts as a sequence model"; LSTM/TSMixer must beat it on the same folds and keys | 10a |
| Latent-factor family = one baseline and two pairs | PCA is the bar; IPCA vs CAE differ only in the shape of one map; SDF vs SAE break the two-stage split from opposite ends; SAE vs CAE is NOT controlled | 11 |
| `allocator_lookback: 63` uniform across moment-based allocators | MVO is not granted a longer covariance window than inverse vol; no max-weight cap so each allocator shows its natural concentration | setup.yaml |
| `top_k_grid` [5,10,20] | 5/10/20% of the 99-name median cross-section; the old [10,20] had no concentrated end, where every comparable case study starts | setup.yaml |
| Two cost regimes | per-share + half-spread is the declared headline; bps grid is a comparator; they disagree by construction | 17 |
| Cross-stage selection by measurement | later stages are alternatives, not improvements; if every allocator/overlay is below its parent an allocation-only carry-forward prices a strategy worse than doing nothing | 17 |
| Holdout refit with `train_end` a full label horizon before the holdout | scoring the validation-fitted model on the judging window is circular; refitting removes circularity but not the optimism of a max over >1,000 backtests, which 20 deflates | 18, 20 |
| `cadence_by_label: fwd_ret_5d: weekly_friday_close` | held on the monthly grid the 5d label was carried ~21 sessions; adding a cadence changes the spec identity and creates new rows | setup.yaml |
| `max_bin: 255`, IPCA `max_iter: 1000`, `treatment_window: 126` | GPU default quartered the bins; cap 100 sat inside the 53-110 convergence range; placebo block derived from the construction window, not guessed | setup.yaml |

## Market-specific guardrails

Grouped by stage; WHAT / WHY / HOW compressed. Apply all of them to any ETF-rotation or cross-asset panel.

### Universe, costs, feasibility (01)
- Survivorship of the list: PIT eligibility removes bias within the list, not of the list. Declare it (`eligibility_note`); never call the universe unbiased.
- Never screen liquidity over the full sample. Compute `turnover = traded_close * traded_volume`; keep (symbol, year) with `n_days >= 200` and `avg_turnover >= 10e6`; `eligible_year = year + 1`; write `eligibility.csv`. Membership for holdout years still only reads the prior year.
- Compute the dollar floor on traded prices, never adjusted close (distributions rescale early history).
- Convert cost to a fraction of price: `cost_bps = 2 * (half_spread + per_share) / median_traded_close * 1e4`; divide each |move| by its own fund's `cost_bps.shift(h)`; break-even = 1. Measure only over eligible fund-years (a move in an ineligible fund was never available).
- Count breadth on decision dates (last session of month) against `BREADTH_FLOOR = max(top_k_grid) = 20`.
- Compute panel ACF within each entity, then average (`panel_acf(..., entity_col="symbol", max_lags=12)`); stacking funds "correlates gold with Brazilian equities" at the joins.
- Spreads are tier-assigned, not measured (no bid/ask in daily bars): sanity-check tiers by turnover (median falls ~5x per tier step); re-price under harsher spreads in 17.
- Known limits, state them: annual eligibility vs monthly trading (a fund illiquid in March stays until January); Yahoo restates splits not distributions, so split history is an input; close-to-close label vs next-open fill is unmeasured in 02 and re-measured on fills in 17.
- Hand the WHOLE sample (holdout included) to `generate_cv_splits`; it applies `holdout_start` itself (trimming first shifts first training dates). Draw folds with `fold_timeline` from the returned boundaries; a figure that re-splits disagreed by 11 days.
- Pass the label buffer as an int of trading sessions with the NYSE calendar (`"21D"` -> 21 sessions); a Timedelta gives ~15 trading days (under-buffering). Assert `min(purge_gaps) >= horizon` on the session timeline, `len(splits) == 8`, last `val_end` < holdout.

### Labels (02)
- Build `forward_return` on the FULL price history sorted (symbol, timestamp) before any eligibility filter, so h counts trading sessions.
- Four asserts, because every defect leaves plausible numbers: incomplete tails are null; window calendar span <= `ceil(7h/5) + 7` days; labelled rows == total - h * n_symbols (no cross-symbol window); dtype Float64.
- Cut the development window on `_label_end = timestamp.shift(-h).over(symbol) < HOLDOUT_START`, never on `timestamp`. Label files keep every row; only diagnostics are restricted.
- Row count is not sample size: 418,362 dev rows -> N_eff 20,017 (4.78% ~ 1/21). Compute `effective_sample_size(..., bar_col="session")` on a bar grid built before any row was dropped; a label occupies h return intervals (`ends = events + horizon - 1`), not h+1 bars.
- Sort the IC series by timestamp before `compute_ic_hac_stats`; bind the HAC bandwidth to the label horizon (`label_horizon=21`); report HAC t beside naive t (momentum: naive 5.06 vs HAC 1.44).
- Score the baseline on the same PIT-eligible rows with `min_obs = median eligible-per-date // 2` (44 here); dispersion needs >= `MIN_SYMBOLS_FOR_DISPERSION = 10` names.
- Write with `write_artifact(..., keys=["timestamp","symbol"], written_by="02_labels", inputs={"market_data": value_digest(prices, ["symbol","timestamp","close"])})`; run `quality_report` with `expected_missing={"trailing": (h, ...)}`.

### Financial features (03)
- Read every window/lag from `setup.yaml::features` (`families_from_config(setup)`); never retype.
- Exclude `open, high, low, close, volume, log_return` from the matrix (the contemporaneous log return is the model's own answer).
- Macro: shift the Treasury timestamp by `+1d` (`dt.offset_by("1d")`), `join_asof(strategy="backward")` onto the trading calendar, THEN compute `yield_curve_zscore` over 252 sessions (a calendar-day frame counts weekends). Residual caveat: 03 reads `load_macro()` (revised history) while the register declares `alfred_initial_release_close_lagged`; which source stage 03 actually loads is unclear (notes uncertainty).
- Re-sort by (symbol, timestamp) after every join that feeds a rolling op.
- Order: `per_entity_features` (momentum -> oscillator -> drawdown/extremes -> regime/state -> `trailing_volume_ratio([21,63])`) -> `gate_to_eligible` (semi-join on symbol, year) -> `clip_within_date([vol_ratio_21d, vol_ratio_63d])` at 1st/99th pct of each date -> `cross_sectional_percentile` for ret_126d, sharpe_126d, vol_63d. Column-wide winsorization fails the rebuild test.
- `warmup_audit(per_entity, {ret_252d: 252, skip_recent_12_1: 252, sharpe_252d: 252, vol_252d: 252, dist_52w_high: 252, sma_ratio_200: 200, max_dd_126d: 126, hurst_100: 100, obv_zscore_63d: 63})` raises if a column is populated inside its lookback; NaN is not a value.
- Rebuild test: `assert_values_agree(built.filter(ts < HOLDOUT), build_features(prices.filter(ts < HOLDOUT)), columns=feature_cols, keys=[timestamp, symbol])`; value-vs-null counts as a difference. This catches any whole-sample transform "including the ones nobody thought to flag".
- Coverage is judged against `label_universe(CASE_DIR)` keys, not nulls inside the matrix ("a matrix emitting a thousand rows where a million were owed carries no nulls at all and is wrong"); classify missing keys leading/interior/trailing/absent and test each against the gate's admitted fund-years; residual must be 0.
- Null policy: `drop_nulls(subset=["sharpe_126d"])`; assert no duplicate (timestamp, symbol).
- Use the shared `momentum_volatility_block` helper: an earlier version used rolling sum/std * sqrt(252/w) and inflated Sharpe features by sqrt(w) in four case studies. `max_dd_{w}d` is the current drawdown vs trailing max (zero at a new high), not worst peak-to-trough.
- Use `ml4t.engineer.features` for oscillators so Wilder-vs-SMA smoothing conventions are fixed.
- Persistence check: a feature must hold its ordering >= one decision cycle (21 sessions) to be usable monthly; 6-month features (cross-rebalance rank corr ~0.83) qualify, 1-month return (~0) does not.

### Model-based features (04)
- Two leak channels through a backward-looking formula: parameters estimated from the future of the value they produce, and smoothed rather than filtered inference. Fitting inside a CV fold is not the fix ("this notebook used to make that mistake"). Use `walk_forward_feature(X, timestamps=, burnin=, refit_every=, fit=, apply=, n_features=, freeze_after=, on_fit_error="skip")`; no fold column.
- HMM: SPY observations `[log_ret*100, rolling_std(21)*100*sqrt(252)]`; `fit_hmm_restarts(X, n_states=2, random_state=0, n_restarts=10)` (k-means init, `init_params="st"`, covariance ridge 1e-6, `threadpool_limits(1)` for bit reproducibility; parallel reductions move probabilities ~1e-11 and the digest); `sort_states_by_variance` (0 calm, 1 stressed) so labels cannot swap across refits; `filtered_state_probs` forward recursion only. hmmlearn `predict_proba`/`predict` are smoothed/Viterbi = look-ahead.
- Derive `regime_transition = |diff(p)|` and `regime_log_duration` over the WHOLE emitted series, not per block; raise on interior gaps.
- GARCH: use `garch11_conditional_volatility(r, mu, omega, alpha, beta, backcast)` with the backcast from the estimation window; `arch`'s `.fix()`/`conditional_volatility` derive residuals, backcast and variance bounds from the whole array (128 of 1,500 shared rows moved by up to 0.064%).
- Freeze at the holdout: `freeze_after = pre-holdout session count`; assert `fit_end.max() <= freeze`; holdout values come from holdout returns through the last pre-holdout parameters (they age; state it).
- Causality test: cut one series at `fit_end + refit_every//2` (inside a block, never on a boundary), re-walk, `np.testing.assert_allclose(short, full[:cut], rtol=1e-12, equal_nan=True)`. "Deleting observations after a session must not move that session's value."
- Convert NaN to null before writing (polars keeps NaN as a value: coverage reads full and the all-null guard cannot fire). Cut the skeleton at `HOLDOUT_END` so a later download cannot widen the artifact.
- Report burn-in per column: `sequence_dataset` maps null -> 0.0 -> the feature's mean after normalization, so missing rows are silently fitted as average observations. `explain_column_gaps` with declared prefixes (regime 756+21; garch 504+1; ffd pre-listing sessions); residual must be 0.
- `ProcessPoolExecutor` with an explicit fork context (Python 3.14 defaults to forkserver, which cannot reach notebook-defined functions); `workers = min(len(payloads), cpu_count-1)`.
- Expanding windows keep a structural break forever; `window=int` gives rolling; the right choice is empirical and left open.
- Model-based IC on validation sessions only; GARCH cross-sectional (`min_obs=20`, >= 63 IC sessions, HAC) and regime/ffd time-series (`robust_ic` with `np.random.seed(42)` set immediately before the loop) are different machinery; never put them on shared error bars.

### Feature evaluation (05)
- Constants: `MIN_CROSS_SECTION = min(10, n_symbols)`, `IC_THRESHOLD = 0.01`, `N_QUANTILES = 5`, `MIN_COVERAGE = 0.70`, `MAX_STALENESS = 0.50`, `FDR_ALPHA = 0.05`, `NAIVE_T = 1.96`, `MIN_SIGN_CONSISTENCY = 0.60`, `REDUNDANCY_CUT = 0.7`, `MIN_IC_DATES = 20`, `MIN_FOLD_DATES = 5`, `MIN_SHAPE_DATES = 20`, `HAC_MAXLAGS = 21`.
- Screen on the union of validation ranges (fold-fitted features only exist there; training dates measure fit). Seal at `last_signal_date` = max date whose `timestamp.shift(-21)` < HOLDOUT_START.
- `validate_modeling_inputs(..., max_abs_return=1.0, fail_on_critical=True)` on both matrices over the whole span to the seal.
- Date-level columns (per-date std < 1e-10) get REVISE, never IC 0; keep them in the matrix (they help only via interaction).
- Triage: STOP if coverage < 0.70 or staleness > 0.50; PROCEED if `fdr_significant` OR (`sign_consistency >= 0.60` against the feature's OWN pooled sign AND |mean IC| >= 0.01, note `stable_and_above_threshold`); REVISE for `date_level_feature`, `insufficient_data`, `not_significant_standalone`. Read the `note` column: an exploration promotion "has not been confirmed by anything".
- Print the three counts (naive |t| > 1.96, HAC |t| > 1.96, FDR rejected) and `survivor_share`; a naive t of ~10 is worth ~3 after overlap correction.
- Count evidence, not columns: pairs with |rho| > 0.7 (Spearman over every ~200th date) are one measurement twice; 22 clusters here.
- Quantile shape: cut buckets within each date, average within date then across dates (`quantile_profile`).
- The screen is evidence, not a filter: model notebooks train on the WHOLE matrix.

### Model populations (06-12)
- Flow: `open_study("etfs", execution_tier, workspace)` -> `load_model_configs(study, family, labels, config_names)` -> guard `narrows_declared_catalog(...) and not POPULATION_NAME -> raise` -> `model_requests(...)` -> `r.resolve()` -> `resolved_model_plan` -> `run_model_population(study, resolved, population_name=, supersedes=)`. Re-runs reuse identities.
- Per fold: median-impute then standardize fitted on training rows only (linear, TabM); GBM leaves NaN in place. Lasso/ENet `alpha_frac` resolves per fold against that fold's `alpha_max`. Huber delta = `max(0.5 * nanstd(y_train), eps)` quantized (`quantize_derived`) so reduction order cannot create two identities.
- Read IC tables with `ic_n_days`; compare only `full_coverage` rows (max `ic_n_days` within each label; 21d has fewer scorable dates than 5d). Aggressive L1 and collapsed checkpoints post high IC on self-selected dates: "a metric averaged over a set the model itself selected is not a metric".
- Read `ic_mean_daily` from `prediction_metrics`, not the catalog `ic_mean` (fold mean) beside `ic_n_days` (date count).
- Cross-family IC only on intersected (symbol, timestamp) keys (`common_sample_daily_ic`); sequence models drop warm-up rows.
- Model-notebook IC carries no HAC adjustment: diagnostic, not a test; inference lives in 05 (features) and 13 (families).
- Populations: narrowed `LABELS`/`CONFIG_NAMES` -> own `POPULATION_NAME`; CPU instead of published cuda -> runner refuses unless `DEVICE="cpu"` + own name (device is identity-bearing); resolve `SUPERSEDES_POPULATION` via `population_supersedes(study, name, declared)` (hard-coded hash is wrong on a clean clone); exclude retired/unpublished rows with `split_unpublished_members` BEFORE the per-family maximum; only the runner builds identities.
- Reconcile menu pairs against `training_runs`/`prediction_members_in_force(study)` every run; "fitted" means a prediction set registered AND listed by a population in force. Use `REPO_ROOT` (not `get_case_study_dir`, which `ML4T_OUTPUT_DIR` redirects) to glob notebook sources; walk the AST for `run_model_population` (a substring test matches commented examples).
- IPCA: `_require_ipca_convergence` raises on an unconverged fit; 100 funds vs 5 factors is comfortable, a handful is not. PCA requires `persistent_entities: true`; `run_pca_fold` drops characteristics; `prepare_panel_data` fills `returns` from the request's own label, so 21d and 5d are separate fits.
- SDF macro context sits inside the model identity; read the point-in-time contract from the resolved specification.
- DML: `regime` and `yield_curve_slope` are finalized FRED values, so read the row as a stability analysis on a fixed retrospective panel; report conditional ignorability, overlap and SUTVA beside the estimate (untested); never read the causal sign into a ranking; never hand-cap `max_samples` (removing the 50,000 cap flipped the sign in sp500_equity_option_analytics).

### Backtest, allocation, risk, cost, holdout (13-20)
- Open the study as the first statement in 13/14/20, before any `get_case_study_dir`, registry read or `BacktestExplorer`: under the preview tier opening rewrites `ML4T_OUTPUT_DIR` process-wide and earlier readers address the released registry.
- 13: coverage map first; forest plot with Driscoll-Kraay 95% CI (ordinary SEs on daily IC of an overlapping label are far too small); a family whose CI crosses zero is "not established", not "fifth"; only full-coverage configs; horizons compared only in 13 section 6; causal row apart in section 7.
- 14: run the plumbing backtest on a no-information signal (`score_weighted_top_k` at the smallest feasible `top_k`) before any real signal; raise when every `top_k` is at or above the tradeable count; read a deflated Sharpe "as the price of having searched".
- 15: carry forward signal-stage Sharpe leaders capped at `checkpoints_per_config` (eight near-identical checkpoints of one network would make the allocator comparison a measurement of one prediction); sweep `top_k` x allocator and read the pivot (converge at top-5, separate at top-20); drop MVO methods whose grid is incomplete (`skip_mvo` -> `INCOMPLETE_ALLOCATORS`); compare `source_span` vs `allocator_span`.
- 16: stops need the bar-by-bar engine (a weight x return product cannot express one); report which engine is in use; read each overlay against its own un-overlaid parent; expect tight stops to fire on ordinary intra-month moves and leave capital idle before the rebalance that would have closed the position anyway.
- 17: pool `signal`, `allocation`, `risk_overlay` leaders (16 writes no baseline row); `PER_SHARE_COMMISSION` from setup.yaml with no fallback; bps regime via `set_backtest_costs_bps(spec, commission_bps=level/2, slippage_bps=level/2)`; read slopes and breakevens, never bps vs per-share point against point; a uniform sweep over a per-asset spread map is sensitivity, not pricing.
- 18/19: `resolve_solvent_carrier` in each notebook (never copy a hash); `build_holdout_training_spec` with `train_end` = holdout open minus `label_buffer`; raise if the refit returns the validation training hash; refuse a second configuration on the holdout; `registered_holdout_generations` flags validation-fitted rows "VALIDATION-FITTED - not out of sample"; 19 prints the validation/holdout pair without interpreting it.
- 20: intervals, paired bootstrap, PSR/DSR; gate `gate2_holdout_diff_not_excludes_zero_negatively`; read the concentration curve for its plateau, keep the registered allocation-stage `top_k`; FF5+MOM R^2 is structurally limited on bonds/commodities/currencies, so use `run_placebo_benchmark`; record an absent holdout as absent, not null.
- Never declare a local `INITIAL_CASH`; never mix `v3.0.0` and `v3.1.0` hashes; never copy a registry number into prose.

## Results and lessons

Numbers below are from the notes (bundle `v3.1.0-artifacts`); the registry is authoritative and is rebuilt on every re-derivation.

| Stage | Finding |
|---|---|
| 01 Feasibility | Round trip 0.86-35.23 bps across funds, median 6.29 bps. Median \|21-session move\| 288.3 bps = 46x the median round trip; 97.3% of moves exceed their own fund's round trip (cost does not force a slower cadence). Turnover falls ~5x per spread tier. Eligible funds per decision date 0-96 of 100; below 20 on 12 of 216 dates, all in 2006; never again from 2007-01-31. 8 folds, last validation ends 2023-11-29, purge gap exactly 21 sessions. Within-fund monthly autocorrelation ~0. |
| 02 Labels | Dev std fwd_ret_21d 0.0612 vs fwd_ret_5d 0.0311 (ratio 1.97 vs sqrt(21/5) = 2.05). Cross-sectional dispersion peaks 6.7% in 2008 vs 3.9% median year. 418,362 dev rows -> N_eff 20,017 (4.78%; 1/21 = 4.76%). Panel autocorrelation 0.942 at lag 1 -> -0.019 at lag 21. Price panel 470,662 keys, 100 funds, 5,031 sessions; fwd_ret_21d covers 99.55%, fwd_ret_5d 99.89%; every missing key is the trailing horizon. **Baseline floor: 126-session momentum mean IC 0.0266 over 4,257 dates (44-ETF minimum), naive t 5.06, HAC t 1.44, p 0.150: raw momentum does NOT clear its own test.** |
| 03 Features | 57 features, 404,500 rows, 99 ETFs, 2007-01-03 to 2025-12-31 (digest 9c02a41ef4364257). 22 redundancy clusters at \|rho\| 0.7 ("well over half the columns repeat an ordering another column carries"). Coverage 85.93% of 470,162 label keys; all 66,132 missing keys sit in fund-years the gate did not admit (46,795 leading, 10,319 interior across 11 funds, 3,992 trailing across 5 funds, 1 fund absent); the warmup bites nowhere; 470 keys carry features and no label. Persistence: ret_126d, vol_63d, sharpe_126d > 0.6 autocorrelation at 42 sessions; ret_21d ~0 at 21; rsi_14 < 0.1; cross-rebalance rank correlation ~0.98 vol_63d, ~0.83 six-month features, ~0.2 oscillators, ~0 one-month return. ret_126d p10-p90 band 0.15-0.30 normally, 0.62 in 2008-09, 0.44 in 2020, 0.34 in 2022. |
| 04 Model-based | 14 columns (`regime_prob_stress`, `regime_transition`, `regime_log_duration`, `ffd_spy..ffd_lqd`, `garch_cond_vol`); total feature columns 71. Regime columns: 85 funds lose the same 777 sessions (756 + 21); GARCH: all 100 funds lose exactly 505 (504 + 1); HYG ffd 318 pre-listing nulls; no interior nulls; residual 0; 500 keys past the label universe (100 x 2025-12-24..31). 10 HMM restarts spread 2.31 nats at the first block; both persistences < 1. Burn-in ends 2009-02-04, 1,736 sessions before the earliest scored validation session 2015-12-28. 85 vs 100 funds likely = funds present at burn-in end (inference). |
| 05 Screen | **No feature clears Benjamini-Hochberg at 5%**; every PROCEED is an exploration promotion (`stable_and_above_threshold`). Interpretation: a monthly label sampled daily on ~100 names holds too little independent evidence; not a screen fault. Leading redundant pairs are the same measurement at two window lengths. |
| 06 Linear | Ridge at a very large penalty ranked the 99 funds "well ahead of zero across the full validation period"; IC-vs-alpha curve single-peaked (flat, rise, fall) = collinearity signature; among full-coverage configs dense shrinkage > sparse selection; the most aggressive L1 settings post the highest raw IC but are partial-coverage. 07 calls this "the one place in these case studies where the linear grid worked". |
| 07 GBM | `above_zero` 14 of 15 on both labels; objective order Huber > MAE > MSE (MSE the only objective not 5/5 above zero). Excess kurtosis fwd_ret_5d 10.26 vs fwd_ret_21d 7.00, yet Huber-minus-MSE gap 0.0143 on 21d vs 0.0075 on 5d (tail heaviness does not set the gap). **The boosted population does not beat the linear one** (strongest full-coverage GBM < strongest full-coverage ridge on the same label/features/folds). `ended_lower` 14 of 15, `median_change` negative, interior peaks 3 (21d) / 5 (5d): 500 trees is longer than the data supports. Median within-config checkpoint IC range ~5/8 of the across-config range. |
| 08 TabM | 24 candidates per label; capacity does not move the ranking measure much on < 100 funds ("capacity is not a dial you turn up"). No fixed numbers in the notes. |
| 09/10/10a Sequence | Results rendered from variables (`COMMON_IC`, `PEAK_EPOCH`, `NEGATIVE_FOLDS`), not in the notes. 10a registry audit 2026-08-30: 682 validation prediction sets, 134 superseded, 498 published, 50 live-by-exclusion but listed by no population. Earlier 11c/11d/11e versions fixed a reporting checkpoint and found a different one scored better, motivating full-schedule publication. |
| 11 Latent | 5 members each claimed by exactly one notebook; cross-sectional ICs small by single-stock standards (instruments already diversified). No numeric IC in the notes. |
| 12 DML | One `causal_runs` row (estimate, Driscoll-Kraay SE, naive comparison, refutation p-value); value not in the notes. |
| 13 Analysis | The two label horizons are not equally covered across families. Outcomes of the supervision ordering and horizon comparison are described, not shown. |
| 14 Backtest | The IC ordering and the backtest-Sharpe ordering disagree; selection therefore uses Sharpe. Validation maximum is taken over > 1,000 backtests. |
| 15 Allocation | Allocators converge at top-5 and separate at top-20; read `source_span` vs `allocator_span`. |
| 16-17 Risk, cost | One cent of half-spread ~ 1 bp on a $500 fund and ~ 5 bp on a $20 fund. **Cross-stage validation rank-1 = a risk-overlay run, not the allocation leader.** Breakeven cost levels are not in the notes. |
| 18-20 Holdout | Holdout refit and backtest registered once; no Sharpe, DSR, PSR or holdout-vs-benchmark numbers appear in the notes; read `results/strategy_assessment.json` and `20_strategy_synthesis/output/etfs/etfs_tearsheet.html`. |

Lessons an agent should carry:
1. Feasibility is cheap and decisive: 46x move-to-cost and ~0 own-return persistence justify a monthly cadence before any model exists.
2. Evidence is counted in independent blocks: 418k rows are 20k effective observations; nothing clears FDR; raw momentum fails HAC. Do not read a naive t as evidence.
3. On a collinear matrix of ~100 funds, dense shrinkage (ridge) beat sparse selection and the boosted population; trees pick one near-duplicate per split, remade per fold.
4. Checkpoint variation is ~5/8 of config variation: register every checkpoint and never report a config's best.
5. IC and Sharpe orderings disagree; the risk-overlay stage, not allocation, produced the cross-stage leader; concentration and allocator interact.
6. The holdout is one short-window measurement against a max over > 1,000 backtests: read intervals, PSR/DSR and the paired gate, not point estimates.

## How to adapt this pattern to a new dataset

1. Write `config/setup.yaml` first: `universe`, `decision` (cadence, snapshot, execution_delay, `cadence_by_label` for every horizon that is not the default), `execution` (cash, share_type, `allocator_lookback`), `mapping`, `costs` (model, per-share, tiered half-spreads or bps), `labels` (primary, buffer, variants, `rebalance_step`), `backtest.rebalance` (default and benchmark thresholds), `backtest.sweep` (funnel, `checkpoints_per_config`, `top_k_grid` as 5/10/20% of the median tradeable cross-section, allocators, cost grids, risk controls), `evaluation` (folds, sizes, holdout, calendar, periods_per_year), `modeling`, `causal` (treatment, `treatment_window` from the construction, confounders), `features` (windows, families with lookback/lag/role/failure_mode), `model_based` (burn-in, refit cadence). Put the machine-readable eligibility floor in the config, which the ETF study did not.
2. Load two price series: adjusted for returns, traded for dollars; verify split handling and that every adjusted row has a traded close.
3. Feasibility (01 pattern): PIT eligibility by (asset, year) from the prior year; breadth on decision dates >= `max(top_k_grid)`; `cost_bps` per instrument from its own price; exceedance of |move| / own round trip at the label horizon over eligible fund-years; `panel_acf` within entity; `generate_cv_splits` on the whole sample with the label buffer in sessions; assert purge gap >= horizon. Stop if typical move < round trip.
4. Labels (02): formula over tradable prices on the full history; four window asserts; endpoint-sealed dev window; N_eff and HAC with bandwidth = horizon; scale ratio check across horizons; baseline floor from one unengineered signal on the same eligible rows with `min_obs = median cross-section // 2`; `write_artifact` + digest sidecar + `quality_report`.
5. Features (03): build from the register; per-entity trailing stats -> gate -> within-date clip -> within-date percentile for the `ranked` list; macro as-of with the declared availability lag, reduced to the trading calendar; exclude OHLCV and contemporaneous returns; `warmup_audit`; `assert_values_agree` full-vs-truncated rebuild; coverage against the label universe with every missing key explained; redundancy clusters at 0.7; persistence >= one decision cycle.
6. Model-based (04): `walk_forward_feature` with burn-in, refit cadence, filtered inference, `freeze_after` the holdout, `on_fit_error="skip"`; fix transform orders in advance and test on development history; causality cut-and-rewalk at `rtol=1e-12`; NaN -> null; declared prefixes, residual 0; derive running statistics over the whole emitted series.
7. Screen (05): union of validation ranges, sealed on label endpoint; coverage/staleness gate; per-date Spearman with `MIN_CROSS_SECTION`; HAC; BH at 0.05 with the three counts printed; fold sign consistency against the feature's own sign; quantile shape within date; triage ledger with `note`; keep date-level columns as REVISE. Train on the whole matrix afterwards.
8. Models (06-12): one notebook and one immutable population per family; every checkpoint registered; median-impute/standardize fitted on training rows only; Huber delta from the fold's label spread, quantized; NLinear as the sequence bar; PCA as the latent bar with IPCA-vs-CAE as the controlled pair; convergence guard on ALS; DML with a placebo block >= the treatment window, Driscoll-Kraay SE, confounders declared, no sub-sampling. Reconcile menu vs registry every run.
9. Analysis (13): coverage map, Driscoll-Kraay forest plot, fold heatmap, full-coverage only, common-sample IC across families, causal row outside the ranking.
10. Backtest funnel (14-17): engine test on noise; every prediction set x entry scheme vs equal weight at `stage='signal'`; top-N by baseline Sharpe with one checkpoint per config -> `top_k` x allocator sweep -> overlays on a bar-by-bar engine against their own parent -> two cost regimes with breakevens; select the cross-stage leader by validation Sharpe in the cost stage and nowhere later.
11. Holdout (18-20): `resolve_solvent_carrier`; refit with `train_end` one horizon before the holdout; trade with everything registered unchanged; one configuration ever; assessment with block/paired bootstrap, PSR/DSR, 1/N benchmark, placebo-controlled attribution, concentration plateau, `gate2` negative-exclusion check.
12. Run everything from the config through accessors (`get_backtest_config`, `get_top_n_predictions`, `get_top_k_values_for`, `get_entry_schemes_for`, `get_allocators`, `get_checkpoints_per_config`, `get_cost_grid_bps`, `get_cost_grid_half_spread_usd`); for your own runs use `open_study(..., workspace="~/ml4t-experiments")` and a new `population_name`; restate no result in prose.

## Related references

- `chapters/06_strategy_definition.md`: feasibility (6.2-6.6), search accounting and run log (6.7) behind 01 and the registry.
- `chapters/07_defining_the_learning_task.md`: forward-return labels, endpoint sealing, N_eff, HAC, BH-FDR (02, 05).
- `chapters/08_financial_features.md`: the feature register, timing contract, within-date transforms, search control (03).
- `chapters/09_model_based_features.md`: refit schedules, filtered HMM, GARCH recursion, fractional differencing (04).
- `chapters/11_ml_pipeline.md`: fold-wise imputation/standardization, ridge/lasso/elastic net on a collinear design (06).
- `chapters/12_gradient_boosting.md`: LightGBM grid, objectives, checkpoints, TabM (07, 08).
- `chapters/13_dl_time_series.md`: LSTM, TSMixer, NLinear and the 13.8 case-study results (09, 10, 10a).
- `chapters/14_latent_factors.md`: PCA, IPCA, CAE, SDF, SAE family (11a-11e).
- `chapters/15_causal_estimation.md`: walk-forward DML, placebo blocks, Driscoll-Kraay (12).
- `chapters/16_strategy_simulation.md`: signal backtest, equal-weight falsification, entry schemes (14).
- `chapters/17_portfolio_construction.md`: allocator sweep, concentration x allocator interaction (15).
- `chapters/18_transaction_costs.md`: per-share vs bps regimes, breakevens (17).
- `chapters/19_risk_management.md`: stop-loss, trailing, time-exit overlays and adverse-excursion calibration (16).
- `chapters/20_strategy_synthesis.md`: holdout refit, PSR/DSR, paired bootstrap, cross-case assessment (18-20).
- `chapters/02_financial_data_universe.md`: adjusted vs traded prices, survivorship, PIT universes.
- `chapters/04_fundamental_alternative_data.md`: ALFRED initial-release macro timing used by the yield-curve family and SDF context.
- `case_studies/us_equities_panel.md`, `case_studies/cme_futures.md`, `case_studies/fx_pairs.md`, `case_studies/crypto_perps_funding.md`: sibling studies sharing the refit-schedule, HMM restart and panel-ACF helpers.
- `case_studies/sp500_equity_option_analytics.md`, `case_studies/sp500_options.md`, `case_studies/nasdaq100_microstructure.md`: DML `max_samples` caution and HAR block-scope refits referenced from this study.
- `libraries/ml4t_diagnostic.md`: `WalkForwardCV`, `cross_sectional_ic_series`, `compute_ic_hac_stats`, `benjamini_hochberg_fdr`, `robust_ic`.
- `libraries/ml4t_engineer.md`: oscillator/trend/regime/volume features, `ffdiff`, `calculate_label_uniqueness`.
- `libraries/ml4t_backtest.md`: the bar-by-bar engine the risk overlays require.
- `workflow.md`, `guardrails.md`, `decision_rules.md`, `evidence.md`, `companion_repo.md`, `glossary.md`: cross-cutting procedure, the guardrail catalog, selection rules, evidence summary, repo layout and shared vocabulary.
- Further reading:
  - Benjamini and Hochberg (1995), false discovery rate.
  - Lopez de Prado, Advances in Financial Machine Learning, Ch4 (label uniqueness / sample weights) and Ch5 (fractional differencing).
  - Newey and West, HAC standard errors; Driscoll and Kraay, panel-robust standard errors.
  - Politis and Romano, stationary bootstrap (cited in `robust_ic`).
  - Ledoit and Wolf, covariance shrinkage (MVO allocator).

## Glossary

- ADV / turnover: average daily dollar volume = traded close x traded volume; floor $10M, prior year, >= 200 sessions.
- Adjusted vs traded close: rescaled series for returns vs the price actually paid, for dollars and per-share costs.
- Point-in-time eligibility: admission to year Y decided from year Y-1 only; `eligibility.csv` (symbol, eligible_year).
- Breadth: eligible funds on a decision date; floor = `max(top_k_grid)` = 20.
- Half-spread convention: cost per leg = half the bid-ask gap (tier-assigned) + $0.0035/share; round trip = 2 x (half_spread + commission).
- Exceedance curve: survival function of |move| / own round trip; crossing at 1 = share of moves that clear cost.
- fwd_ret_h: `close.shift(-h)/close - 1` on adjusted close, h trading sessions; label endpoint = session t+h.
- Development window: rows whose label endpoint resolves before `holdout_start`.
- Purge gap / label buffer: sessions between train end and validation start, = label horizon (21D / 5D counted in sessions).
- N_eff: AFML average-uniqueness count; ~N/h for daily-sampled h-session labels (20,017 of 418,362).
- IC: per-date Spearman between feature/prediction and forward return across eligible names; `ic_n_days` = dates contributing; full coverage = max `ic_n_days` within a label.
- HAC / Bartlett: autocorrelation-consistent SE with bandwidth bound to the horizon; naive t treats dates as independent.
- BH-FDR: Benjamini-Hochberg at 0.05 over every feature with an IC series; confirmation arm.
- Exploration arm: sign consistency >= 0.60 (against the feature's own sign) and |IC| >= 0.01; note `stable_and_above_threshold`; unconfirmed.
- Triage ledger: PROCEED / REVISE / STOP per feature with `note`; coverage >= 0.70, staleness <= 0.50.
- Signal vs state (register `role`): ranks assets vs describes the environment; state columns rank nothing alone.
- Timing contract: lookback (bars read) and lag (bars until the input is knowable); yield curve is lag 1.
- Within-date clip / percentile: winsorization and rank computed inside each decision date, after the eligibility gate.
- Refit schedule: burn-in, fit on prefix, emit until next refit, refit on the longer prefix; `refit_boundaries(n_obs, burnin, refit_every)`.
- Filtered vs smoothed: P(z_t | x_1:t) vs P(z_t | x_1:T); only filtered is a feature.
- Freeze at holdout: `freeze_after` = pre-holdout count; holdout values from the last pre-holdout parameters.
- FFD: fixed-width fractional differencing of log price, d fixed in advance (0.4 / 0.5), `threshold=1e-5`.
- GARCH(1,1): sigma^2_t = omega + alpha eps^2_{t-1} + beta sigma^2_{t-1}; persistence alpha+beta; half-life ln(0.5)/ln(alpha+beta); backcast = variance seed from the estimation window.
- Checkpoint: a scoreable model state (GBM iteration, DL epoch, SDF epoch) registered as its own prediction set.
- Population: immutable named list of prediction identities declared before fitting; `supersedes` records lineage; retired vs unpublished members.
- Training hash / prediction hash / backtest hash: content-addressed identities; device, preset values, cash and share_type are inputs.
- alpha_frac / alpha_max: lasso/ENet penalty as a fraction of the smallest per-fold penalty that zeros every coefficient.
- Huber delta: 0.5 x std of the fold's training labels, quantized.
- TabM: MLP ensemble with a shared backbone and per-member rank-1 scaling vector plus output layer.
- NLinear: subtract the last observed level from the lookback window, one linear map; the sequence-model bar.
- Persistent entities: each panel column is the same fund throughout; required for return-panel PCA.
- IPCA / CAE / SDF / SAE: exposures as a shared linear map of features (ALS) / the same with a network / pricing-objective cross-section weighting with macro context / encoder-bottleneck-head without a factor stage.
- DML: effect from residuals of outcome|confounders and treatment|confounders; Driscoll-Kraay SE; block-permutation refutation with blocks >= `treatment_window`.
- Entry scheme: method (`score_weighted_top_k`, `equal_weight_top_k`) + `top_k` + long/short; concentration = `top_k`.
- Stage: `signal` -> `allocation` -> `risk_overlay` -> `cost_sensitivity` -> `holdout` in `backtest_runs`.
- Rebalance step / thresholds: schedule slots advanced per trade; skip when |dw| < 0.005 AND notional < $100 (benchmark: none).
- Adverse excursion: worst intra-holding drawdown; calibrates trailing stops.
- Breakeven cost: level at which net Sharpe reaches zero, per regime.
- Solvent carrier: the validation configuration `resolve_solvent_carrier` hands to the holdout.
- PSR / DSR: probabilistic / deflated Sharpe ratio, deflated for the number of trials searched.
- Placebo portfolio: random-selection benchmark separating universe-driven from selection-driven factor exposure.
- Preview tier / workspace: reduced-scale run that rewrites `ML4T_OUTPUT_DIR`; canonical = published path.
