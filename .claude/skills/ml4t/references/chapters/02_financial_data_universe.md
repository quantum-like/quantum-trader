# Chapter 2: The Financial Data Universe
> Chapter 2 maps the data a systematic strategy consumes (market / fundamental / alternative; US equities, ETFs, CME futures, S&P 500 options, crypto perps, FX, macro) and argues that every dataset embeds definitions (timestamps, adjustments, identifiers, revisions) that decide what the data means, so the convention a file uses must be *measured*, not assumed, and recorded where the loader lives. It equips you to profile coverage before rows, adjust corporate actions and futures rolls correctly, build session-correct bars, join slower series point-in-time, run a four-pillar quality framework (structural validation, anomaly detection, PSI drift, hygiene/quarantine), detect lookahead and quantify survivorship bias, stitch providers at an explicit seam, and run safe incremental updates into partitioned Parquet. The §2.3 due-diligence core holds that many apparent research successes are manufactured by data defects (PIT violations, survivorship, corporate-action errors, identifier mismatches). §2.4 benchmarks file formats, embedded and server engines, and pandas vs Polars, and defaults research to partitioned Parquet + Polars/DuckDB, escalating to server databases only when concurrency or ingest demands it.

## When to use this reference
- Loading any companion dataset (`load_us_equities`, `load_etfs`, `load_cme_futures`, `load_sp500_options_eda`, `load_crypto_perps`, `load_fx_pairs`, `load_macro`) and needing its schema, conventions and known defects.
- Deciding which price representation (raw / split-adjusted / total-return; `adj_*` vs `raw_*` futures) to feed features, labels, backtests or fill simulation.
- Building or validating backward corporate-action adjustment, or comparing two vendors' adjusted closes.
- Converting intraday bars to daily bars for CME futures or FX (session boundaries, not UTC midnight).
- Constructing continuous futures (roll rules, Panama vs ratio adjustment) or validating against a vendor series.
- Working with option chains: IV convergence codes, Greeks validation, straddle selection, same-contract vs chained returns.
- Joining a slower series (premium index, macro, fundamentals) to faster bars point-in-time; deriving Binance funding from premium.
- Auditing a panel for survivorship bias, ticker reuse, unadjusted splits, phantom sessions, or universe completeness.
- Running or designing data-quality checks: `OHLCVValidator`, anomaly detectors, PSI, gap detection, leakage tests, fault injection.
- Setting up acquisition: provider fallback, stitching WikiPrices + Yahoo, `DataManager`/`HiveStorage`, daily deltas, cron, `ml4t-data` CLI.
- Choosing storage (CSV/Parquet/Feather/HDF5), a query engine (DuckDB/ClickHouse/Timescale/...), or pandas vs Polars at a given scale.

## Core ideas (the why)
- **Definitions decide meaning.** Bar open vs close stamps, period stamp vs release date, raw vs total-return, reused tickers vs `instrument_id`, vintages. Measure which convention a file uses, once, and record it at the loader.
- **Profile coverage before rows.** Nulls and OHLC invariants are within-row checks; survivorship is about which rows exist at all. No row-level check surfaces a symbol never written to the file (`01_us_equities_eda`, `15_survivorship_bias_detection`).
- **A discontinuity in a collection record is not a market event.** The WIKI universe grows for 50 years, peaks, declines; markets do not do that, collection processes do. **Absence of exits is absence of evidence**: pre-2014 the panel cannot say what left, so "unknown" beats "small".
- **A universe that fills and holds was chosen; one that rises and falls was collected.** Flat ETF counts after launches = survivorship-filtered pool (`03_etfs_eda`).
- **A corporate action is not a return.** Backward adjustment anchors factor = 1 at the present; adjusted history is not a price anyone paid, and a vendor restatement changes every earlier value. Match representation to decision (`02_corporate_actions`).
- **A tolerance is a claim about accumulated error.** Derive it from the mechanism (steps × machine epsilon); a round 0.1% would pass a wrong dividend convention. **A failed check needs a magnitude before an explanation** (760 strict ETF OHLC "violations" are 1-ulp rounding).
- **Liquidity spread is a constraint, not a footnote.** >10× daily volume across ETF groups; 50× zero-premium span in crypto; option spreads punish wings. Uniform cost/capacity assumptions are wrong at one end.
- **A futures product is not a series.** Product, contract, continuous series are three things; a roll rule is a choice. A late-starting history is usually a capture date (`04_cme_futures_eda`).
- **Aggregate on the session, not the calendar.** UTC midnight splits a CME or FX session across two rows; nothing downstream can recover and nothing looks wrong (`05_futures_session_aggregation`, `12_fx_pairs_eda`).
- **Ship two labelled price series and no unlabelled one.** `adj_*` for returns/momentum/vol/labels, `raw_*` for carry/term structure/notional/costs, reconciled by `cum_ratio`.
- **A check that cannot fail is not a check**; say what a passing check would catch. **A validator that passes has told you nothing until shown a fault**; inject one per check (`13_data_quality_framework`).
- **A symbol is not a date; an argument named for a bar length does not guarantee one; aligning the bars is the validation** (`06_futures_continuous`).
- **The convergence code decides which rows exist.** ~1/3 of option IVs were extrapolated/interpolated/inferred, not solved; an IV from a near-worthless option is not a measurement (`07_sp500_options_eda`, `08_options_greeks_computation`).
- **A constant-maturity options series is not an instrument.** Labels need same-contract returns; backtests need held-contract continuous series; roll-zeroing deletes real P&L one-sidedly (`09_options_continuous`).
- **A point mass in a derived quantity is a property of its formula first.** Binance premium is exactly zero inside the impact spread; funding is pinned at the interest rate inside the clamp (`10_crypto_perps_eda`, `11_crypto_premium_analysis`).
- **An as-of join never fails, so it never reports a gap.** Measure the age of each match against the publication schedule. **A timestamp is a claim about when something was knowable.**
- **Detection is a backstop, not a substitute for construction.** Leakage checks catch leakage correlated with the scored return and miss the rest (`14_point_in_time_validation`).
- **Survivorship bias is an estimate you build**, inheriting every defect underneath; its sign flipped depending on whether corporate actions were repaired first. **Bias-free is not complete**: ~3,200 symbols vs 4,000–5,000 listed.
- **Comparing two providers usually measures their adjustment conventions, not their data**; read the time *shape* of a difference (uniform ratio = basis; step = mis-dated action; noise = rounding). **A stitch is two adjustment bases, not two date ranges** (`16_provider_comparison`, `17_complete_pipeline`).
- **Production pipeline principles:** validate at boundaries; tag every row with `source`; make updates incremental; keep the orchestrator provider-independent. `load()` is cache-first (research), `fetch()` provider-first (refresh).
- **Where an update stops matters more than where it starts.** A bar for the session in progress is not a bar; find the window end by asking the vendor. `lookback_days` is the operating lever for revisions (`19_incremental_updates`).
- **A validator must match the cadence it validates** (equity daily gap > 5 days; 24/7 crypto gap > 1.5 h). **A gap detector locates a discontinuity; deciding what it was takes a calendar.**
- **A benchmark is a comparison only under one policy** with validated row counts; memory-mapping is not a read; no format wins read, write and size; column projection is what makes columnar fast; **speedups are ratios, summarise with the geometric mean** (`20`–`22_*_benchmark`).

## Method recipes (the how)
Notebook citations are repo paths in the companion repository; bare `NN_name` mentions elsewhere in this file mean `02_financial_data_universe/NN_name` (run with `uv run python 02_financial_data_universe/NN_name.py`).

### Coverage profiling and survivorship signatures (`02_financial_data_universe/01_us_equities_eda`, `02_financial_data_universe/03_etfs_eda`, `02_financial_data_universe/04_cme_futures_eda`, `02_financial_data_universe/10_crypto_perps_eda`)
1. Per-symbol lifespans: `group_by("symbol").agg(len, min(timestamp) first_date, max(timestamp) last_date)`; `leaves_early = last_date < dataset_end`.
2. Plot two flows, not the total: universe size (distinct symbols per year) and entries/exits (first and last observations per year). A symbol whose last date is the panel's last date has not left.
3. ETF variant: count symbols with `start.year <= y <= end.year`; diff panel symbol set vs `config.yaml` set both ways.
4. Futures: `get_product_coverage` loads front month (`tenors=[0]`) per product → rows/start/end; contract overlap via `group_by("instrument_id")` first/last trade.
5. Crypto: distinct symbols per month (`dt.truncate("1mo")`); a step chart that only rises is the chosen-universe signature.
6. Run `check_survivorship_bias` and `check_universe_completeness` (see Survivorship recipe) on any new panel.

### Backward corporate-action adjustment and validation (`02_financial_data_universe/02_corporate_actions`)
| Item | Rule |
|---|---|
| Representations | raw; split-adjusted; total-return (split + dividend) |
| Notebook formula | `factor = 1` at last date, walk backwards. Split ratio R on t: `factor_{t-1} = factor_t / R_t`. Dividend D_t with prior close P_{t-1}: `m_t = (P_{t-1} − D_t)/P_{t-1}`, `factor_{t-1} = factor_t × m_t`. `P_adj = P_raw × factor` → adjusted returns = total return with dividends reinvested at prior close (a convention) |
| Library (`ml4t/data/adjustments/core.py`, `_apply_canonical_adjustments`) | per-row price event factor `close / (split_ratio × (close + dividend))` (split-only `1/split_ratio`), volume factor `split_ratio`; `price_adjustment_factor` = cumprod of *later* rows' factors (`shift(-1, fill_value=1).reverse().cum_prod().reverse()`); `adj_<col> = col × factor`, `adj_volume = volume × volume_adjustment_factor`. Needs sorted `date` column; raises on negative dividends/volume. Docstring: uses ex-date close, "differs from data vendors that discount dividends using the prior close" |
| Validation tolerance | `accumulation_bound = 100 × n_rows × np.finfo(np.float64).eps`; pass iff max relative diff ≤ bound |
| Vendor convention test | AAPL 2014-06-09 7:1 split: is pre-split `close` already divided by 7? |

Uncertainty (kept from notes): whether the AAPL validation prints PASSED under the library's ex-date-close convention vs the notebook's prior-close derivation is a runtime result not visible in the digest; the check does not hold panel-wide (`15_survivorship_bias_detection` §3).

### OHLC invariant check with mechanism-derived tolerance (`02_financial_data_universe/03_etfs_eda`, `utils/data_quality.py`)
- `check_ohlc_invariants(df, open_col, high_col, low_col, close_col, volume_col, rtol=OHLC_RELATIVE_TOLERANCE)` → `check, valid_pct, applicable_rows, total_rows`; checks high ≥ low/open/close, low ≤ open/close, volume ≥ 0 (exact); ordering uses `greater >= lesser − rtol × max(|greater|,|lesser|)`; only rows with all required columns non-null count.
- `OHLC_RELATIVE_TOLERANCE = 4 × 2.220446049250313e-16` (4 float64 eps): ~1 ulp of two independently rounded products with headroom, ~11 orders of magnitude below a real defect (one cent on $100 = 1e-4). `rtol=0.0` for exact.
- Measure breach size `(violating side − other side)/price` on strict-violation rows vs eps. On raw exchange bars (crypto, FX) report breach *counts*; any breach is a defect.

### CME session dating and ratio-adjusted daily aggregation (`02_financial_data_universe/05_futures_session_aggregation`)
1. Session: opens Sunday 17:00 CT, ends 16:00 CT (session date = date it ends), maintenance 16:00–17:00, no Saturday session.
2. `add_session_date`: convert UTC → `America/Chicago`; `_after_close = hour >= 16`, `_is_friday = weekday == 5`; `session_date = when(_after_close & ~_is_friday).then(date + 1d).otherwise(date)`.
3. Ratio back-adjust **before** aggregating, per (product, tenor): sort by timestamp; roll where `instrument_id != shift(1)`; `ratio = new contract open / old contract previous close` (adjacent hourly bars); walk backwards with `cumulative *= ratio` so bars before a roll are multiplied; `adj_* = raw_* × cum_ratio`.
4. Aggregate by (session_date, product, tenor): open=first, high=max, low=min, close=last for `adj_*` and `raw_*`; `cum_ratio` last; `volume` sum; `bar_count`; `session_start/end` = min/max timestamp.
5. Validate: bar-count distribution (typical 20–24, modal 23-hour session; shorter = holidays/half days/thin deferred tenors). Daily OHLC invariants cannot fail for valid input (selection, not calculation) but catch broken hourly bars.
6. Output: `continuous_daily.parquet` + `by_product/{PRODUCT}.parquet` to `ML4T_DATA_PATH/futures/market/continuous/daily/` only if `WRITE_TO_DATA=1`, else chapter `output/futures_daily/`.

### Continuous futures construction and vendor validation (`02_financial_data_universe/06_futures_continuous`)
| Parameter / step | Value / rule |
|---|---|
| `MIN_OUTRIGHT_PRICE` | 500.0 for ES; separates outrights from calendar spreads (single-digit negative prices, high volume); scale per product |
| `CALENDAR_ROLL_SESSIONS_BEFORE` | 5 trading sessions (not calendar days) |
| Bar length | from `rtype` (32=1s, 33=1min, 34=1h, 35=1d) and modal timestamp gap (`describe_bars`); individual ES contracts are daily although `frequency="hourly"` was asked |
| Symbol parsing | `^([A-Z]+)([FGHJKMNQUVXZ])(\d+)$`; F..Z = Jan..Dec, quarterly H/M/U/Z; one-digit year code carries no decade → use `expiration` column or price history |
| Expiry from history | `contract_life` per `instrument_id`; ES expires third Friday: `scheduled_contracts` = `last_trade` is Friday (`weekday == 5`) with day 15–21; holds for all above-median-volume contracts |
| Volume roll (`identify_front_month`) | daily volume per `instrument_id` (outrights only); adopt new leader only if not in `retired`; never switch back; `is_roll = front != prev_front`; count days `volume_leader != front` (zero on ES) |
| Calendar roll (`identify_front_month_calendar`) | `session_grid` numbers sessions, folding Sunday-dated bars onto Monday; `roll_out_session = session_no(last_trade) − 5`; front = candidate with earliest `last_trade`; report days it cannot cover (anti-join, not inner join) |
| Disagreement episodes | numbered on the complete ordered frame `(differs & ~differs.shift(1)).cum_sum()`; lengths in sessions |
| Raw (`create_continuous_raw`) | inner join bars to front by timestamp, keep `instrument_id == front_instrument_id` |
| Panama/additive (`_compute_roll_gaps`) | gap = new close − old close on roll date; `cumulative_adjustment` = reversed cumsum shifted by 1 (roll date excluded); `adj = price + cumulative_adjustment` |
| Ratio (`_compute_roll_ratios`) | ratio = new close / old close (old close ≠ 0); `adj = price × cumulative_ratio` |
| Vendor validation | collapse vendor hourly to UTC-day closes (`group_by(date).agg(close.sort_by(timestamp).last())`), join on date, `our_close − utc_day_close`; handover window = [our roll date, old contract `last_trade`]; measure inside vs outside; gap by year in bps = `10_000 × mean|diff| / mean level` |
| Wrapper | `construct_and_validate(product, min_outright_price)` → rows, contracts_used, validation_days, mean/max abs diff, mean_abs_diff_bps; only ES has individual contracts on disk |

Production path: Databento pre-rolled continuous hourly (tenors 0,1,2) → `data/futures/market/continuous/hourly/` → `05_*` → `continuous/daily/continuous_daily.parquet` → `load_cme_futures()` (daily default).

### Options chain analytics (`02_financial_data_universe/07_sp500_options_eda`)
- Parameters: `ATM_BAND=(0.98,1.02)`, `CHAIN_BAND=(0.7,1.3)`, `FORWARD_DAYS=5`, `SURFACE_MAX_DAYS=180`, `SPREAD_BAND=(0.5,1.5)`, `WING_BAND=(0.9,1.1)`, `MIN_MID_PRICE=0.10` (below: quotes in ticks), `SPX_TROUGH_DATE="2020-03-23"`; moneyness = strike/spot.
- Row filter for any analysis: `iv_convergence == "Converged"` and `0 < implied_vol < 2.0`. Read the code set from the data (`group_by("iv_convergence").len()`): input part `Converged|SmallBid|IntrVal|Failed`, fallback part `FlatExtrapol|LinInterp|PutCallPair` (~10 codes present).
- Chain density per (timestamp, symbol): options, calls, puts, distinct strikes, distinct expirations. Smile: nearest expiration IV vs moneyness (calls) plus put-call IV gap at same strike/expiration (pivot on `call_put`; median/p90/max `|C − P|`). Term structure: mean ATM-band IV per expiration vs `days_to_maturity`.
- Spread proxy: `spread_pct = (ask − bid)/mid_price` with `mid_price > 0.10`; bucket moneyness to 0.05 (`round(m×20)/20`); report median, p90, share > 50% of mid, n per bucket; summarise over quotes, not bucket medians. Spread is a liquidity proxy (no volume/OI in file).
- Greeks bounds: delta ∈ [−1,1], gamma ≥ 0, vega ≥ 0, IV > 0 (arithmetic facts); theta ≤ 0 expected shortfall on deep ITM European puts. Report share and count, no pass/fail threshold. PIT checks: `expiration < timestamp`, `days_to_maturity < 0` counts.
- Information preview: ATM IV = converged call with smallest `|moneyness − 1|` in band; `iv_change_5d = iv_atm − iv_atm.shift(5)`, `ret_fwd5 = close.shift(−5)/close − 1` over symbol; robustness: exclude 2020-02-15..2020-04-30, non-overlapping (every 5th row), per-symbol sign, within-day quintiles of `iv_change_5d`. Treat as a hypothesis for Chapter 9.

### Black-Scholes, IV solver, Greeks and validation (`02_financial_data_universe/08_options_greeks_computation`)
- `d1 = (ln(S/K) + (r + σ²/2)T)/(σ√T)`, `d2 = d1 − σ√T`; `C = S N(d1) − K e^{−rT} N(d2)`; `P = K e^{−rT} N(−d2) − S N(−d1)`; `T ≤ 0` → intrinsic. Parity check `C − P = S − K e^{−rT}` at S=K=100, T=0.25, r=0.05, σ=0.20.
- `implied_volatility(market_price, S, K, T, r, option_type, bounds=(0.001, 5.0))`: `brentq(objective, lo, hi, xtol=1e-8)`; None if no sign change, `T ≤ 0`, or solver error.
- Greeks: Δc = N(d1), Δp = N(d1) − 1; Γ = N'(d1)/(S σ √T); Vega = S √T N'(d1); Θc = −S σ N'(d1)/(2√T) − r K e^{−rT} N(d2), Θp = −S σ N'(d1)/(2√T) + r K e^{−rT} N(−d2); ρc = K T e^{−rT} N(d2), ρp = −K T e^{−rT} N(−d2). `compute_all_greeks` returns vega/100, theta/365, rho/100.
- Validation protocol: `N_GREEKS_VALIDATE=2000`, `N_IV_VALIDATE=200`, `SAMPLE_SEED=42`; `.sample(n, seed)` not `head(n)`; filter `days_to_maturity ∈ [20, 90]`, `0.05 < implied_vol < 2.0`, non-null delta; r per date = `load_macro()["dgs1"]/100` joined and forward-filled (1y vs 20–90d term mismatch is a named limitation); `FLAT_RATE_FOR_COMPARISON=0.015` kept to measure its cost; T from `years_to_maturity`; errors = |ours − vendor|.
- Residual attribution: flat vs per-date delta error; bucket by `option_type × moneyness` (<0.9, 0.9–1.1, >1.1); `implied_spot_gap = delta_error / our_gamma`.
- IV recovery: Converged rows only; count failures; bucket by time value (<0.05, 0.05–1.0, >1.0); tick sensitivity `TICK=0.01`: re-solve at mid ± TICK/2, width = |hi − lo|, count rows where either end fails.

### Options series: constant-maturity straddle, same-contract and held-contract returns (`02_financial_data_universe/09_options_continuous`)
- Parameters: `DEMO_SYMBOL="AAPL"`, `DEMO_YEAR=2019`, `DTE_WINDOW=(25,35)`, `TARGET_DELTA=0.50`, `DELTA_TOL=0.15`, `MIN_BID=0.01`, `MAX_REL_SPREAD=0.30`, `HOLDING_PERIOD=10` trading days.
- `select_constant_maturity_straddle`: `rel_spread = (ask − bid)/clip(mid, 0.01)`; filter DTE window, bid ≥ 0.01, rel_spread ≤ 0.30, Converged, `|delta| ∈ [0.35, 0.65]`; split calls/puts; **inner join on (date, strike, expiration)**; `instr_mid = call_mid + put_mid`; rank per date by `|call_abs_delta − 0.5|`, then `|dtm − 30|`, then strike; take first.
- Roll detection: `expiry_moved`, `strike_moved`, `is_roll`; `change_kind` ∈ {expiry changed, strike moved same expiry, same contract}; `daily_return = instr_mid/prev_mid − 1`. Holding spells: id = cumsum of identity change (first row True); run length = group size.
- Same-contract return (labels): `build_exit_lookup` (raw chain, `bid >= MIN_BID`, Converged); exit date = calendar index + h; join call and put exit mids on (exit_date, strike, expiration); `same_contract_ret = exit_mid/entry_mid − 1`. Compare with naive `instr_mid.shift(−h)/instr_mid − 1`.
- `held_contract_returns` (backtests): reprice yesterday's (strike, expiration) legs at today's chain; `held_ret = (call_now + put_now)/prev_mid − 1`. `build_continuous_straddle_series`: `raw_daily_ret`, `zeroed_daily_ret` (0 on rolls), `held_daily_ret` (held_ret on rolls); compound `price_zeroed`, `price_held` from first `instr_mid`.
- Short with daily-reset notional: `∏(1 − r_t) − 1`, not `−(∏(1 + r_t) − 1)`.

### Crypto premium, PIT as-of join and funding (`02_financial_data_universe/10_crypto_perps_eda`, `02_financial_data_universe/11_crypto_premium_analysis`)
- Premium index = `(max(0, ImpactBid − PriceIndex) − max(0, PriceIndex − ImpactAsk)) / PriceIndex`; exactly zero inside the impact spread. Units are decimals (0.001 = 0.1%): ×100 percent, ×10,000 bps.
- Point-mass diagnosis: share of exactly-zero closes; smallest non-zero magnitude vs std; zero rate per symbol vs median hourly dollar volume (`close × volume`); share with all four premium OHLC fields zero; zero share by year.
- Gap check over every symbol: `timestamp.diff().dt.total_hours().over("symbol") > 1`.
- PIT join: `PREMIUM_BAR_HOURS=8`; shift premium `timestamp += 8h` (= `premium_published`, keep `premium_bar_opened`); sort both by (symbol, timestamp); `ohlcv.join_asof(premium, on="timestamp", by="symbol", strategy="backward", check_sortedness=False)`; `premium_age_hours = timestamp − premium_published`, stale if `>= 8`; classify unmatched as pre-first-publication vs interior holes.
- Funding: `INTEREST_RATE=0.0001` per settlement, `FUNDING_CLAMP=0.0005`, `FUNDING_INTERVAL_HOURS=8`, `PERIODS_PER_DAY=3`; `F = P + clamp(I − P, −c, +c)`; inside `|I − P| ≤ c` F = I exactly; outside F = P ∓ c (unbounded). `APY = F × 3 × 365`. Settlement pairing: bar stamped t (open) → `settles_at = t + 8h`; join `load_funding_rates()` on (settles_at, symbol); score MAE and exact-match share for `F` vs misreading `clamp(P) + I`; sweep offsets −8h/0/+8h/+16h and pick by correlation (MAE does not discriminate).
- Regimes: `NOTABLE_PREMIUM=0.001` (10 bps) → High/Low/Neutral; `ROLLING_WINDOW_DAYS=30` → `ROLLING_WINDOW_OBS=90`; `TAIL_CLIP_PCT=0.5` per tail for histograms; `COLOR_CLIP_BPS=20`; `APY_THRESHOLD_PCT=20`.

### FX conventions, gap taxonomy and session grid (`02_financial_data_universe/12_fx_pairs_eda`)
- Parameters: `FREQUENCY="4h"`, `SESSION_TIMEZONE="America/New_York"`, `SESSION_ROLLOVER_HOUR=17`, `LONG_GAP_HOURS=24`, `GAP_HIST_MAX_HOURS=80`, `GAP_HIST_BIN_HOURS=2`. Normalise `EUR_USD` → `EURUSD`.
- `classify_pair`: quote == USD → Direct (invert for USD strength); base == USD → Indirect; else Cross. `volume` is tick volume (quote-update count on one venue), never traded size.
- Gap classes: `gap_hours == 4` one bar; Friday→Sunday weekend close; else "neither" → split by whether the whole universe shares the date (closure, e.g. Dec 24/31) or a subset (candidate fault).
- Session grid detection: distinct hours-of-day in UTC (12) vs local (6) = local grid shifted by DST. Session daily: `local_time = timestamp.replace_time_zone("UTC").convert_time_zone(tz)`; `session = (local_time + 7h).date()`; group by (symbol, session) first/max/min/last/sum/count. Compare with UTC `group_by_dynamic(every="1d")`: complete-day counts, one-bar days, close diffs in bps.

### Data quality framework: validator, anomalies, PSI, hygiene, quarantine (`02_financial_data_universe/13_data_quality_framework`)
| Pillar | API and defaults |
|---|---|
| Structural | `OHLCVValidator(...)` (all checks on, `negative_price_policy="forbid"`, `max_return_threshold=0.5`, `staleness_threshold=5`); `.validate(df)` → `.passed`, `.issues[]` (`.check`, `.severity`, `.row_count`, `.message`), `.critical_count`, `.error_count`; numeric and finite checks always run |
| Fault injection | `high = low − 1` rows 10–12 → `price_consistency`; `volume = −1000` row 20 → `negative_volume`; real null via `Series.scatter(row, None)` → `null_values` (np.nan is not a null) |
| Return outliers | `ReturnOutlierDetector(ReturnOutlierConfig(method, threshold=3.0, min_samples=20)).detect(df, symbol=...)`; cutoffs MAD `median ± τ×MAD/0.6745`, Z `mean ± τσ`, IQR `[Q1 − τIQR, Q3 + τIQR]`; `.value` already a percentage |
| Volume spikes | `VolumeSpikeDetector(VolumeSpikeConfig(window=20, threshold=3.0, min_volume=0, min_samples=20))`; metadata `average_volume` |
| Staleness | `PriceStalenessDetector(PriceStalenessConfig(max_unchanged_days=3, check_close_only=False))` |
| Orchestration | `AnomalyManager(AnomalyConfig(enabled, report_severity_threshold="warning", return_outliers=..., volume_spikes=..., price_staleness=...))`; `.analyze(df, symbol)`, `.analyze_batch({symbol: df})` → `.anomalies`, `.get_critical_anomalies()`; detectors concatenated without reconciliation |
| Drift | `calculate_psi(baseline, current, n_bins=10, epsilon=1e-6, open_outer_bins=True)` → total + per-bin (`baseline_pct`, `current_pct`, `psi_contribution`); edges = baseline quantiles (dedupe by epsilon, outer → ±inf); baseline = first half, current = second half |
| Hygiene | pooled calendar-day step distribution, then `detect_gaps(df, max_gap_days=5)` (`days_since_prev > 5`); `df.unique(subset=["timestamp"], keep="first"\|"last")`; corporate-action candidates `overnight_return = open/prev_close − 1`, flag `|r| > 0.25`, score recall vs `split_ratio != 1.0` and false positives |
| Pipeline | `quality_check_pipeline(df, symbol, quarantine_dir, anomaly_cfg)`: validate → quarantine parquet on any critical → `AnomalyManager.analyze` → dedup keep last → PASS if validation passed and no critical anomalies else REVIEW |
| Logging | `structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))`, never blanket `warnings.filterwarnings("ignore")` |

### Point-in-time validation (`02_financial_data_universe/14_point_in_time_validation`)
- Parameters: `DEMO_SYMBOL="SPY"`, `MA_WINDOW=5`, `MA_WINDOW_LONG=20`, `LEAK_THRESHOLD=0.30`, `MAX_GAP_DAYS=5`, `EXECUTION_LAG_ROWS=1`, `MONTHLY_RELEASE_LAG_DAYS=7`, `QUARTERLY_RELEASE_LAG_DAYS=30`, `MACRO_DEMO_SERIES="unrate"`, `_period_days = {"monthly": 31, "quarterly": 92}`.
- `centered_window_offsets(width)`: run `rolling_mean(window_size=width, center=True)` on a ramp; first output reveals `[T+lo, T+hi]`; future inputs = hi.
- Leakage test: naive `corr(feature level, next return)` fails (≈0 even for tomorrow's close). Scale-invariant: `corr(feature.pct_change(), close.shift(−1)/close − 1)`, finite-filtered; pass return-shaped features directly. Threshold 0.30 sits in the empty gap between clean (~0) and leaking (≈1).
- `validate_signal_trade_lag(signals, execution_lag=1)`: `earliest_execution = timestamp.shift(−lag)` in rows (inherits the trading calendar; 1 row = 1–5 calendar days).
- Macro semantics: per column count value changes per year (cadence) and share landing on day-of-month 1 (period-stamp signature). `stamp_to_availability(frame, column, cadence)`: `available_from = stamp + period_days + release_lag`; join and forward-fill; count disagreements and max gap.
- `PITValidator(df, date_col="timestamp")`: `check_future_leakage(feature, price_col="close")` → HIGH if `|corr| > 0.30`; `check_date_gaps(max_gap_days)`; `check_monotonic_dates()`; `.violations`.
- Vintages: `FREDProvider().fetch_ohlcv("GDP", "2023-01-01", "2023-09-30", frequency="quarterly", vintage_date="2023-11-01")` vs `"2024-06-01"` → `revision_pct`; value mapped to open=high=low=close, volume=1.0; fails loudly without `FRED_API_KEY`.

### Survivorship bias quantification (`02_financial_data_universe/15_survivorship_bias_detection`)
1. Parameters: `SEED=42`, `N_SIMS=1000`, `MC_BAND=(10,90)`, `ANALYSIS_START="2014-01-01"`, `MAX_TRUSTED_RETURN=1.0`, `MIN_SESSION_COVERAGE=0.2`.
2. Read the exit record (`has_left = last_date < dataset_end`, exits by year) and choose the window where exits are recorded (2014-01-01 onward here).
3. Window universe: first date within 30 days of `ANALYSIS_START`; returns `adj_close/adj_close.shift(1) − 1` over symbol with `prev_volume`.
4. Repair (drop, do not winsorise): `suspect = ret > 1.0 OR prev_volume == 0`; keep real crashes (GTAT 2014-10-06, −90%+ on volume).
5. Phantom sessions: keep sessions where `quoted >= 0.2 × len(universe)` (evidence rule, not a holiday calendar).
6. Leavers: `exit_threshold = window_end − 15 days`; leaver if `last_date < exit_threshold`.
7. Scenarios (shares cause/acquisition/other; terminal returns): Empirical 2010–2020 (Eckbo & Lithell 2025, Table A1 Panel B, 2,805 CRSP delistings) 0.257/0.655/0.088 with −0.60/+0.25/0.0; Bull Market 0.18/0.73/0.09 with −0.50/+0.30/0.0; Stress 0.32/0.53/0.15 with −0.80/+0.15/−0.10. `expected_terminal` = share-weighted sum (positive in two of three).
8. `build_panel_matrices`: pivot returns wide; `alive[t,s] = session <= last_session[s]`; `held` = alive shifted; `exits = ~alive & held & ~is_survivor`; contribution = `nan_to_num(ret) × held`; equal weight over `n_held`; `universe_path` adds `terminal/n_held` on exit sessions via `np.add.at`; `survivor_path` has none; $100-based cumprod; `bias = survivors_only − full_universe` (positive = overstatement).
9. `run_monte_carlo`: `rng.choice(3, size=n_leavers, p=shares)` → terminal payoffs → bias; median and 10–90 band.
10. Report four uncertainty sources side by side: MC band (sampling), spread of scenario medians (assumption), repaired vs raw bias (data quality), bias with terminal = 0 (observed-returns share).
11. Heuristics: `check_survivorship_bias(data, symbol_col, date_col)` flags `unique_end_dates < 10`, `delisting_rate < 0.05`, `exit_span_years < 0.8 × panel_span_years` → `likely_biased`. `check_universe_completeness(data, symbol_col, date_col, expected_market_size=4500, expected_ipo_rate=(100, 400))` warns if `recent_new < 50` in final 3 years, mid-panel mean new listings < 100, `coverage_ratio < 0.9`.

### Multi-provider acquisition, fallback and seam-rescaled stitch (`02_financial_data_universe/16_provider_comparison`, `02_financial_data_universe/17_complete_pipeline`)
- Every provider inherits `BaseProvider.fetch_ohlcv(symbol, start, end, frequency="daily")` → canonical Polars schema `[timestamp (tz-aware UTC), symbol (upper), open, high, low, close, volume (Float64)]`, sorted, no duplicates; template: validate inputs → rate limit → `_fetch_and_transform_data` → `_validate_ohlcv` (rejects nulls, requires tz-aware, OHLC invariants, duplicates).
- `fetch_with_fallback(symbol, start, end, providers, provider_names) -> FetchResult` (`provider_used`, `providers_tried`, `error_messages`); on exception record the error, on empty frame record `"Empty result"`, return first non-empty. Demo order `[yahoo, wiki]`.
- `compare_providers_detailed(symbol, start, end, a, b)`: truncate to date, inner join, `close_diff_pct = (close − close_b)/close_b×100`; `DISCREPANCY_PCT=0.1`, `EXACT_MATCH_PCT=0.01`; diagnose by time shape (constant ratio / step / noise).
- Naive `build_complete_history` (WikiPrices to `2018-03-27`, Yahoo from `2018-03-28`, concat/sort/unique) is superseded: it leaves a step.
- `_combine_pipeline_sources(historical, recent, symbol, wiki_end=WIKI_END_DATE)`: (1) `_naive_utc()` both legs (convert to UTC, drop zone; nothing moves); (2) fetch recent leg from `OVERLAP_START="2018-03-01"`; (3) overlap = inner join of closes on calendar day, `raise ValueError` if empty; (4) `scale = yahoo_close / wiki_close` on the **last shared day** (both > 0, non-null), never across the seam; (5) filter `historical ≤ wiki_end`, `recent > wiki_end`; (6) historical `open/high/low/close *= scale`, `volume /= scale`; (7) tag `source="wiki"/"yahoo"`, concat on common columns, sort. Print the row count to confirm no double counting.
- Pipeline validation: `_validate_pipeline_data` (OHLC invariants, `close <= 0`, max gap > `EQUITY_MAX_GAP_DAYS=5`, duplicates) → `{n_rows, date_range, max_gap_days, issues, is_valid}`; `_validate_crypto_coverage` (`gap_hours > CRYPTO_MAX_GAP_HOURS=1.5` adds `int(gap_hours)−1` missing hours; issue if `coverage_pct < 99.0`); `_add_crypto_session_features` (`hour_utc`, `day_of_week`, `funding_window = hour//8*8`, `session_date`, `is_weekend = weekday >= 5`); every row gets `symbol` and `processed_at`.
- Sequencing: fetch → combine (rescale) → validate → tag → store.

### Production storage and updates: `DataManager`, `HiveStorage`, safe daily delta (`02_financial_data_universe/18_data_management`, `02_financial_data_universe/19_incremental_updates`)
- `HiveStorage(StorageConfig(base_path, compression="zstd", partition_granularity="month"))`: key `equities/daily/AAPL` → prefix-encoded directory + `.metadata/equities_daily_AAPL.json`, nested `year=2024/month=1/data.parquet`; every write commits a new generation directory with a `CURRENT` pointer (no half-written partition visible); `read(key, start_date, end_date, columns)` prunes partitions, `end_date` exclusive, timestamp → `Datetime("us","UTC")`; `partitions(key)` sorted by value; `list_keys()` walks disk.
- Granularity by cadence: `year` daily (~252 rows), `month` hourly (~720, default), `day` minute (~1,440), `hour` tick (~3,600).
- Workflow: `dm.load(sym, start, end)` → `dm.update(sym, lookback_days=7)` (prefer the safe variant) → `GapDetector(exclude_weekends=True).detect_gaps(df, frequency="daily")` → `dm.batch_load_universe("sp500", start, end)` → cron `ml4t-data update --all`.
- `update_through_last_complete_bar(manager, storage, symbol, *, provider, lookback_days=7, asset_class="equities", frequency="daily", through=None, max_retreat_days=5) -> int`: `start = last_stored − lookback_days`; `newest = through or last_complete_daily_bar()` = **exchange-local** (`America/New_York`) date − 1; return if `newest <= last_stored`; for `step in 0..5`: `end = newest − step`, stop if `end <= last_stored`; `manager.fetch(...)`, re-raise unless `_rejected_as_incomplete` (`DataValidationError`, possibly as `__cause__`); if a refusal occurred and the frame does not extend past stored max, raise it; merge `pl.concat([stored, fresh]).unique(subset=["timestamp"], keep="last").sort("timestamp")`; `storage.write(merged, key, metadata={start_date, end_date, last_updated=None, data_range, attributes.last_update}, preserve_metadata=True)`.
- `UpdateStrategy`: `INCREMENTAL` (default, after last stored timestamp), `APPEND_ONLY` (audit archives), `FULL_REFRESH` (recovery), `BACKFILL` (holes). `IncrementalUpdater.determine_update_range()` / `update_incremental()` → `UpdateResult(success, update_type, rows_added, rows_updated, rows_before, rows_after, gaps_filled, duration_seconds, errors)`.
- `GapDetector(exclude_weekends=False).detect_gaps(df, frequency="daily", tolerance_days=0)` → `[{"start","end","size_days"}]`; `detect_gaps_in_storage(storage, key, start, end, frequency)` adds leading/trailing gaps. A second `ml4t.data.utils.gaps.GapDetector` (`is_crypto`, `fill_gaps(method="forward")`) is what `StorageManager.update(fill_gaps=True)` uses.
- `data_health_report(storage, symbols)`: `days_stale = (now − last_date).days`; stale if `> STALE_DAYS=7` (`FRESH_DAYS=3` amber); gaps via weekday-aware detector; issues = `OHLCVValidator(max_return_threshold=0.5).validate(df).error_count`.

### Storage and engine benchmarks (`02_financial_data_universe/20_storage_benchmark_file`, `02_financial_data_universe/21_storage_benchmark_database`, `02_financial_data_universe/22_pandas_polars_benchmark`)
- Panel: `generate_ohlcv_data(n_symbols=100, n_rows=10_000, seed=42)` on a real 390-bar session grid from 2024-01-02; scales XS 1k / S 10k (default) / M 100k / L 1M (§2.4) / XL 10M / XXL 100M rows; set `os.environ["BENCHMARK_SCALE"]` **before** importing `utils.storage_benchmarks`.
- Timing policy: `time_write` = `gc.collect()` + single cold shot; `time_read(func, n_runs=TIMING_RUNS=3)` = untimed warm-up + mean (warm-cache); `force_materialize_polars/pandas` touch every column; `validate_result(result, expected_rows, op, tolerance=0.0)` exact row count; `projection_speedup = time_s / columnar_time_s`.
- Formats: CSV, Parquet (`write_parquet`/`read_parquet(columns=...)`), Feather (`write_ipc`/`read_ipc`), HDF5 (`pd.HDFStore` fixed; no projection → columnar = full read).
- Databases: operations write / full read / range (`RANGE_QUERY_SHARE=0.2` of sessions, `[start, end)`) / daily aggregation (first/max/min/last/sum ordered by time, bucketed to **day**) / ASOF join (trade→latest quote by symbol). Durability inside the timed write (`wait_until_rows_visible(count_func, expected_rows, timeout, poll=0.05s)` for QuestDB ILP+WAL and InfluxDB); client row construction measured by `measure_row_build` and reported as a share. `DB_CONFIG` defaults: clickhouse :8123; questdb http 9000 / ilp 9009 / pg 8812; timescaledb :5437 (postgres/benchmark/ml4t); postgres :5436; influxdb :8086 org ml4t bucket market_data; `WAL_FLUSH_TIMEOUT=3`. Docker: `docker compose --profile benchmark up -d`.
- pandas vs Polars: `result_witness(result) -> (rows, abs_sum, signed_sum, char_count)` then `benchmark_operation(name, category, pandas_func, polars_func, n_runs)` requires rows/characters equal, `abs_sum` within `rel_tol=1e-9`; `speedup = pd_time/pl_time`; category and overall `exp(mean(log(speedup)))`. Params `SMA_WINDOW=20`, `VOL_WINDOW=20`, `SHARPE_WINDOW=63`, `EWM_SPAN=20`, `TRADING_DAYS=252`, `HORIZONS=[1,5,21,63,126,252]`, `LAGS=[1,5,21]`, `STRING_MATCH_TARGET_SHARE=0.25`. Memory: `memory_usage(deep=True).sum()` vs `estimated_size("b")`, both /1e6; no psutil RSS deltas.

## Guardrails and pitfalls
**Coverage, identifiers, corporate actions**
- **Survivorship via backfilled snapshot (WIKI)** — symbols alive at collection, backfilled to IPO; exits only from ~2014; pre-2014 bias unknowable. Detect: entries/exits per year, exit span < 80% of panel span. Prevent: survivorship work on 2014–2018 only; CRSP/Compustat with delisting returns for longer spans.
- **Survivorship via chosen universe (ETF panel; Yahoo returns live tickers only)** — count rises then runs flat, no early endings. Treat as a candidate pool; Chapter 6 builds the trading universe.
- **Within-row checks cannot see missing rows** — profile coverage first.
- **Ticker reuse** (HERO, EXXI relisted on the same ticker) — detect gains > 100% with `split_ratio` silent and zero prior volume; prefer permanent identifiers.
- **Unadjusted actions inside "adjusted" columns** (PCO 1:100 reverse split 2017-12-05) — one 200× row moves a 2,000-name EW portfolio by double digits and flips the survivorship sign. Drop `ret > 1.0` or `prev_volume == 0`; run this before the survivorship check.
- **Raw prices as returns** — splits look like −50%, dividends like drops; price-only momentum mis-ranks across yields. Total-return for features/labels/long backtests; raw for fills; split-adjusted only intraday.
- **Vendor `close` ambiguity** — WIKI close is raw; Yahoo may be split-adjusted and varies by interface; ETF panel `close` is fully adjusted. Test on AAPL 2014-06-09 7:1; name the convention per number.
- **Round-number tolerances** / **strict float comparison on adjusted panels** / **percentage hides one bad bar** — derive tolerance from the mechanism; `rtol = 4×eps` on adjusted panels, `0` on raw exchange bars; print counts beside percentages.
- **Equal-weighted group averages ≠ symbol averages**; carry group counts. **Uniform liquidity/cost assumptions** — group-specific costs; convert shares to notional.
- **Corporate-action threshold false positives** — 25% overnight rule recalls all splits but ~1/3 of flags match none (GOOGL Class C in `ex-dividend`, earnings crashes); use a feed. **Quarantining real market moves** — adjust upstream before quarantining.
- **Running survivorship on raw adjusted prices** flips the sign; **delisting rate alone** is not evidence (exit-span check); **terminal returns on the delisting date** double-count premiums (measure terminal = 0); **tight MC band read as certainty** (assumption spread is 10× larger).
- **Phantom sessions** (LFVN, END print on holidays; 2014-01-02 looks empty) — keep sessions with ≥ 20% of universe quoted.
- **`MAX_SYMBOLS` not passed to loaders** — `apply_max_symbols` must reach every consumer. **Blanket `warnings.filterwarnings("ignore")`** hides the dropped-alt-text warning.
**Futures**
- **Capture date read as listing date** — check per-product start dates before a common sample. **Hand-written `ASSET_CLASS_MAP`** — raise on unmapped products.
- **Unadjusted returns across rolls** — ratio-adjust per (product, tenor) *before* aggregating. **Calendar-day aggregation of session markets** — session date rule; detect via hours-of-day and one-bar-day counts.
- **Mixing tenors when counting rolls** (deferred tenors flip often) — report front month separately. **Differencing adjusted levels across tenors** reads roll history, not the curve — carry/term structure from `raw_*`.
- **One-digit year codes** (`ESM1` = 2001/2011/2021/2031) — year from `expiration`. **Bar length assumed from argument name** — `describe_bars` first. **Joining different bar lengths on a bare timestamp** — collapse to a common grid first.
- **Calendar spreads in contract data** (single-digit negatives, high volume) — `MIN_OUTRIGHT_PRICE` floor. **Last trade ≠ expiry for thin listings** — third-Friday filter. **No-rollback is irreversible** — count leader ≠ front days; keep the guard but do not credit it.
- **Calendar days instead of sessions** (5 days before Friday lands on Sunday) — session grid folding Sunday onto Monday. **Episodes numbered on disagreeing rows only** halve lengths. **Inner joins swallow uncovered days** — anti-join and report.
- **Roll convention mattered least in zero-rate years** — handover gap grows ~10× as financing − dividend yield widens; measure by year in bps.
**Options**
- **Stale hand-written solver code tables** (~10 codes, not 5) — read from data; **non-solved IVs as features/targets** (~1/3 of rows) — filter `Converged`, not a time-value proxy.
- **Reading skew off calls alone** (left half is ITM calls) — check put-call gap at same strike/expiration. **Thresholding Greeks bounds** hides the informative theta line — report share and count.
- **Spread model fitted to the median** — p90 > 2× in wings; calibrate to the tail. **Overlapping windows inflate significance** (5-day overlap → effective n ≈ 1/5) — non-overlapping subsample. **Eight names in one year is not a cross-section**; **correlation summarises a non-monotonic quintile shape wrongly**.
- **Constant risk-free rate across 2020** (1y fell > 1 pp) — per-date rate. **Validating on `head(n)`** (all 2 January) — random sample with seed, print days covered. **European formulas on American contracts** — express residuals as `delta_error/gamma` spot gap; do not attribute. **Silently dropping solver failures** — count them.
- **Constant-maturity series as instrument or label** — same-contract returns via raw-chain lookup; never `shift(−h)` on the chain. **Pairing legs independently** yields strangles/diagonals — inner join on (date, strike, expiration). **Zeroing roll-day returns** deletes real P&L (tens of pp/year) — held-contract return; roll cost belongs in Chapter 18. **Negating a compounded long** for a short — state sizing; `∏(1 − r) − 1`. **Run length from an "unchanged" flag** undercounts by one.
- **Plotly figures split across cells** render half-built under Papermill — one figure per cell.
**Crypto and FX**
- **Premium units** (decimals) — scale explicitly. **Exact zeros treated as missing** (1 in 7; up to 1/3 on thin contracts) — do not filter; choose tie handling. **Exact-timestamp join across frequencies** matches 1 in 8 — as-of join.
- **Bar-open stamps hand the future to an as-of join** (8h bar stamped 00:00 closes 08:00) — advance by one bar; sweep offsets by correlation. **As-of null count as coverage** — staleness from the publication clock (one symbol > 2 months stale). **Gap checks on the reference symbol only** — run on every symbol.
- **Clamp misread as applying to the premium** (`clamp(P)+I` removes the dead zone) and **clamp misread as a cap** (realized rates run many times beyond ±(I±c); real cap per contract) — `F = P + clamp(I − P)`; score against realized. **Funding direction off premium sign** (mean premium negative, mean funding positive) — read from realized. **Premium close as funding rate** — Binance uses a time-weighted average; use realized rates where they exist. **"APY above threshold" ≠ "clamp binds"** — label both.
- **FX tick volume as traded volume** — activity indicator only. **Dollar direction depends on symbol position** — classify and invert direct pairs. **"Long gaps are weekends"** — Dec 24/31 close the whole universe yearly; three-way taxonomy.
**Quality framework and PIT**
- **Validator that only passes** — fault injection. **One threshold across MAD/Z/IQR is three cutoffs** — print cutoffs in return units. **PSI closed outer bins** drop the strongest drift evidence — open to ±inf; **PSI quantile bins on a point mass** — read per-bin contributions. **Gap threshold without the step distribution** — set it in the break (5 days); universe-wide gaps are market events. **Dedup keep-first vs keep-last** is a trust question.
- **Centered moving averages** (~half the window in the future; widening worsens) — trailing only. **Level-vs-return leakage heuristic** ≈ 0 even when leaking — both sides in return space. **Leakage checks miss non-price leakage** — construction discipline. **Calendar-day execution lag** — shift by rows.
- **Period-stamped macro panels** (FRED first-of-month stamps, pre-forward-filled; unemployment disagrees with release-lagged on 3 days in 4, > 10 points in spring 2020) — shift to availability. **Revised macro values** flatter backtests — `vintage_date` queries.
**Acquisition, updates, storage**
- **Adjustment-basis mismatch across providers** (Yahoo adjusts through today; WikiPrices through 2018-03-27) — rescale on a shared date, volume by reciprocal. **Rescale factor measured across the seam** bakes a day's move into the factor. **Silent no-overlap** looks like a stitch that needed none — `raise ValueError`. **Mixed timestamp clocks** — `_naive_utc`. **Double-counted rows at the seam** — filter each leg on the seam date, print row count. **Un-audited multi-source panels** — `source` + `processed_at` columns.
- **Half-formed current-session bar from Yahoo** (volume with null OHLC; `DataValidationError: yahoo: Column 'open' contains 1 null values`) — `_drop_priceless_rows`, `update_through_last_complete_bar`; `DataManager.update()` asks for it every day. **Placeholder resolution is not calendar-predictable** (2026-09-03 AAPL still NaN 8.5 h after close; CI red 23:55Z–02:21Z) — retreat loop. **Retreating past a bad row inside the window** — stop when `end <= last_stored`; re-raise if nothing extends. **Exchange date vs UTC date** — bound in `America/New_York`.
- **`DataManager.update(fill_gaps=True)` default** forward-fills weekends/holidays (hundreds of phantom bars) — `fill_gaps=False` or the safe helper. **Gap detector without an exchange calendar** (holidays = 1-day gaps; calendar-dense series report none) — pair with a calendar-aware completeness check. **Wrong cadence threshold** — per market structure. **Validator flags as defects** (extreme-return WARNING on each series' largest move) — score against corporate-action columns.
- **Vendor revisions after first store** (Yahoo ~5 trading days, CRSP ~30) — `lookback_days` overlap, keep newest. **Patching corrupted parquet in place** — `FULL_REFRESH`. **Config drift of `end:` dates** — `--update` extends past the configured end. **Provider instance churn** (~30 ms httpx init, ~2.5× slower) — one instance per run. **Non-atomic parquet writes** — `atomic_write_parquet` (`.tmp` then `replace`); HiveStorage generations. **DataBento costs** — `--estimate-only`; `databento_acknowledge` requires `I UNDERSTAND` unless `--force`.
**Benchmarks**
- **Memory-mapped read as a read**; **warm-cache numbers** (state the policy; cold/networked storage favours compression); **mixed timing policies in one chart**; **truncated results posting the fastest time** (QuestDB `limit=1,{expected+1}`); **acknowledge-early engines** (poll to queryable inside the timing); **timing the wrong operation** (kdb+ `-8!` looked ~40× faster); **aggregation semantics drift** (`MIN(open)` or minute bucketing); **range query that is a full scan** (size as share of sessions); **size column as compression ratio** (engines report different accountings; compare sizes only in nb20); **client cost inside write bars** (`iterrows` 10× `itertuples`).
- **Unlicensed PyKX import** segfaults later numpy work — check q binary and licence first. **Optional engine guarded only for `ImportError`** (ArcticDB `NotImplementedError`; no Linux ARM64 wheel → Docker `benchmark` image) — catch `Exception`. **`expected` vs `available` engines** — status table with coverage. **Environment-dependent numbers** (HDF5 ~71 MB locked vs ~79 MB newer pandas) — `uv run`, print versions and flags. **`BENCHMARK_SCALE` set too late**.
- **Unequal materialisation**, **implementations silently computing different things** (rolling Sharpe crossing symbols; rank / max rank; two sector tables; anti-join branch), **arithmetic mean of speedups**, **megabytes in a seconds table**, **mismatched memory units** (`estimated_size("mb")` is MiB), **RSS deltas as copy cost**, **synthetic anti-join counts** (50 µs offset → every trade unmatched), **`join_asof` sortedness warning with `by`** (sort by `[by, on]`), **`merge_asof` vs `join_asof` sort rules**, **InfluxDB client teardown tracebacks** (drop handles), **hypertable size via `pg_total_relation_size`** (~16 kB; use `hypertable_size()`).

## Decision rules and defaults
| Decision | Rule / default |
|---|---|
| Price representation | features, factor research, long-horizon backtests → total-return adjusted; order-execution simulation → raw; intraday (< 1 day) → split-adjusted |
| Futures adjustment | backtest P&L → Panama (dollar P&L); IC / momentum / vol features → ratio (percentage returns, never negative); live → raw + execution-layer position management; production daily file = ratio with `adj_*`, `raw_*`, `cum_ratio` |
| Roll rule | volume-based follows liquidity (known after the fact; use when validating vs vendor volume-rolled); calendar roll 5 sessions before expiry is reproducible a year ahead; on ES they differ by a session or two a few times a year |
| Validation tolerances | recursive adjustments `100 × n_steps × eps`; OHLC `4 × eps` adjusted / `0.0` raw bars; always counts beside percentages |
| Session boundaries | CME 16:00 CT, Friday-after-close stays Friday, adjust before aggregating; FX 17:00 New York (`decision.session_calendar` in the FX `setup.yaml`) |
| Options row filter | `iv_convergence == "Converged"`, `0 < implied_vol < 2.0`; Greeks validation adds `days_to_maturity ∈ [20, 90]`, `implied_vol > 0.05` |
| Straddle selection | DTE 25–35, `|delta|` within 0.15 of 0.50, bid ≥ 0.01, rel. spread ≤ 0.30, Converged, legs paired on strike+expiration, rank by delta gap → DTE gap → strike |
| Options series by purpose | labels → same-contract holding return; backtest mark-to-market → held-contract continuous; never naive chained or roll-zeroed |
| Crypto joins | advance premium stamps 8h; backward as-of by symbol; stale at age ≥ 8h; settlement at t+8h chosen by correlation sweep |
| Funding constants (Binance USDT-M) | I = 0.0001/8h, c = 0.0005, 3 settlements/day, APY = F × 1095; notable premium 10 bps; regime window 30 days (90 obs) |
| Anomaly defaults | MAD/Z/IQR threshold 3.0, `min_samples` 20; volume window 20, threshold 3.0; staleness 3 days (detector) / 5 (validator); extreme return 0.5 |
| PSI reading | < 0.1 none; 0.1–0.25 moderate, investigate; > 0.25 significant, act; open outer bins; read per-bin |
| Gap thresholds | daily US equities 5 calendar days (`EQUITY_MAX_GAP_DAYS`); crypto hourly 1.5 h, coverage fail < 99%; funding windows [0, 8, 16] UTC |
| Corporate-action candidates | 25% overnight return; expect ~1/3 false positives without a feed |
| Leakage / lag | `|scale-invariant corr| > 0.30` → HIGH; execution lag 1 row after close-of-day signal |
| Macro availability stand-ins | monthly 31 + 7 days; quarterly 92 + 30 days; production uses the release calendar |
| Survivorship sequencing | exit record → window (2014-01-01+) → repair (`ret > 1.0`, `prev_volume == 0`) → phantom sessions (< 20%) → leavers (`last_date < end − 15d`) → survivor vs universe EW paths → scenario MC (1,000 draws, 10–90) → three uncertainty sources |
| Survivorship / completeness red flags | < 10 end dates; delisting rate < 5%; exit span < 80% of panel; < 50 new listings in final 3 years; mid-panel mean < 100/yr; coverage < 90% of ~4,500 |
| Free-panel go/no-go | survivorship work only over 2014–2018; longer spans need CRSP/Compustat/commercial with delisting returns |
| Stitch constants | `WIKI_END_DATE="2018-03-27"`, `OVERLAP_START="2018-03-01"`, `AS_OF_DATE="2025-01-15"` (bump on revision), `EQUITY_START="1990-01-01"`; `DISCREPANCY_PCT=0.1`, `EXACT_MATCH_PCT=0.01`; nb16 Sharpe `RISK_FREE_RATE=2.0`%, 252 days |
| Validator policy | `max_return_threshold=0.5` → WARNING; `staleness_threshold=5` → WARNING; price-consistency → ERROR (fails `passed`); `negative_price_policy` ∈ {forbid (default), warn, allow} |
| Freshness / updates | `FRESH_DAYS=3`, `STALE_DAYS=7`; `UPDATE_LOOKBACK_DAYS=7`; `MAX_RETREAT_DAYS=5`; strategy INCREMENTAL daily / APPEND_ONLY audit / FULL_REFRESH recovery / BACKFILL holes |
| Storage | `StorageConfig(compression="zstd", partition_granularity="month", lock_timeout=30.0, metadata_tracking=True)`; year/daily, month/hourly, day/minute, hour/tick |
| Cron | `0 18 * * 1-5 cd ~/ml4t && ml4t-data update --all --storage-path ./data >> logs/update.log 2>&1`; book: `python data/download_all.py` once (~10 min) then `--update` daily/weekly |
| Production checklist | initial download → schedule `--update` → monitor freshness, gaps, validation before backtests → `OHLCVValidator` on every load |
| Provider hygiene | one `YahooFinanceProvider()` per run; `DEFAULT_RATE_LIMIT=(60, 60.0)`; circuit breaker `failure_threshold=5, reset_timeout=300s`; 3 retries; FRED `(100, 60.0)`; Yahoo `end` exclusive, WikiPrices inclusive |
| File format | long-term/cloud/cross-language → Parquet; local Python interchange → Feather; legacy append pipelines → HDF5; human inspection → CSV |
| Engine reading | full-scan panel for columnar vs row; range panel for pruning/indexes (the backtest question); write bars = client cost to queryable; no engine named fastest, read your run |
| pandas vs Polars | < 100K rows either; 100K–1M prefer Polars (joins/groupby/strings); > 1M Polars; convert to pandas at the plotting boundary; new projects start Polars; pandas CoW + PyArrow strings first |
| Benchmark scale | `BENCHMARK_SCALE="L"` for §2.4; CI injects `S`; `TIMING_RUNS=3`; `RANGE_QUERY_SHARE=0.2`; `LOG_AXIS_RATIO=10.0`; `WITNESS_RELATIVE_TOLERANCE=1e-9` |
| Vendor integration checklist | confirm close convention on a known split; timestamp semantics by measurement; bar length from the frame; units; code/enum sets from data; align grids before comparing; inject faults; counts with percentages |

## Code patterns and APIs
Loaders (`data` package; all accept `max_symbols=0` via `utils.data_quality.apply_max_symbols`):
| Loader | Signature / schema |
|---|---|
| `load_us_equities(symbols=None, start_date=None, end_date=None, max_symbols=0, lazy=False)` | `symbol, timestamp, open, high, low, close, volume, ex-dividend, split_ratio, adj_open/high/low/close, adj_volume`; WIKI 1962–2018-03-27, raw + `adj_*` |
| `load_etfs(...)`, `load_etfs_unadjusted` | `close` already split+dividend adjusted; from 2006-01-03; `ETFDataManager.from_config(path)` → `.config.tickers` |
| `list_cme_products(frequency="hourly")`, `load_cme_futures(products=None, tenors=None, start_date=None, end_date=None, frequency="daily", continuous=True, lazy=False, max_symbols=0)` | hourly: `timestamp, product, tenor, instrument_id, rtype, open..volume`; daily: `session_date, product, tenor, adj_*, raw_*, cum_ratio, volume, bar_count, session_start, session_end`; paths `futures/continuous/product={P}/`, `futures/individual/{P}/data.parquet`, `futures/market/contract_definitions.parquet` |
| `load_sp500_options_eda(symbols=None, option_type="all", start_date=None, end_date=None, include_greeks=True, max_symbols=0)`, `load_sp500_daily_bars(...)` | AAPL, MSFT, GOOGL, AMZN, JPM, BA, XOM, KO, 2019–2020; `timestamp, symbol, expiration, strike, call_put, bid, ask, mid_price, underlying_price, days_to_maturity, delta, gamma, theta, vega, rho, implied_vol, iv_convergence, option_style, years_to_maturity` |
| `load_crypto_perps(frequency="1h"\|"8h", symbols=None)`, `load_crypto_premium(frequency="8h"\|"1h")` | premium columns `timestamp, symbol, premium_index_open/high/low/close`; `case_studies.crypto_perps_funding.funding_data.load_funding_rates()` → `timestamp, symbol, funding_rate` |
| `load_fx_pairs(frequency="4h"\|"daily", pairs=None)` | symbols `EUR_USD`; `volume` = tick volume |
| `load_macro(series=None, start_date=None, end_date=None)`, `load_macro_initial_release`, `load_macro_metadata` | daily grid incl. weekends, pre-forward-filled, period-stamped (`dgs1`, `unrate`) |

Library imports: `from ml4t.data import DataManager`; `from ml4t.data.adjustments import apply_corporate_actions, apply_splits, apply_dividends` (`apply_corporate_actions(prices, split_col="split_ratio", dividend_col="ex-dividend", price_cols=None, volume_col="volume")`, needs `date`); `from ml4t.data.validation import OHLCVValidator`; `from ml4t.data.anomaly import AnomalyManager, PriceStalenessDetector, ReturnOutlierDetector, VolumeSpikeDetector`; `from ml4t.data.anomaly.config import AnomalyConfig, PriceStalenessConfig, ReturnOutlierConfig, VolumeSpikeConfig` (pydantic; `DetectorConfig`: `enabled`, `threshold=3.0`, `window`, `min_samples=20`; `AnomalyConfig`: `asset_overrides`, `symbol_overrides`, `report_severity_threshold`, `save_reports=True`); `from ml4t.data.providers import WikiPricesProvider, YahooFinanceProvider`; `from ml4t.data.providers.fred import FREDProvider`; `from ml4t.data.storage import HiveStorage`; `from ml4t.data.storage.backend import StorageConfig`; `from ml4t.data.universe import Universe`; `from ml4t.data.update_manager import GapDetector, UpdateStrategy, IncrementalUpdater`; `from ml4t.data.core.exceptions import DataValidationError`; `from ml4t.data.etfs import ETFDataManager` (also `CryptoDataManager`, `FuturesDataManager`, `MacroDataManager.from_config`); `from utils.downloading import update_through_last_complete_bar`.

Providers: `YahooFinanceProvider(enable_progress=False, rate_limit=None)` (`yf.download(auto_adjust=True, actions=False)`, `FREQUENCY_MAP` daily→`1d`, hourly→`1h`, minute→`1m`; `name="yahoo"`); `WikiPricesProvider(parquet_path=None, cache_in_memory=False)` (expects `ticker, date, adj_*`; `list_available_symbols()`, `get_date_range(symbol)`, `download(output_path, api_key)`; `name="wiki_prices"`); `FREDProvider(api_key=None)` (`FRED_API_KEY`; `fetch_ohlcv(series, start, end, frequency="daily", vintage_date=None)`, `fetch_series_metadata`, `fetch_multiple`, `close()`).

```python
# CME session date (Polars); ratio-adjust BEFORE aggregating
df.with_columns(pl.col("timestamp").dt.convert_time_zone("America/Chicago").alias("ts_ct")) \
  .with_columns(pl.col("ts_ct").dt.date().alias("_d"), (pl.col("ts_ct").dt.hour() >= 16).alias("_after"),
                (pl.col("ts_ct").dt.weekday() == 5).alias("_fri")) \
  .with_columns(pl.when(pl.col("_after") & ~pl.col("_fri")).then(pl.col("_d") + pl.duration(days=1))
                  .otherwise(pl.col("_d")).alias("session_date"))
# Roll detection keyed by instrument_id, then reverse-cumprod of ratio per (product, tenor)
s = df.sort(["product", "tenor", "timestamp"]).with_columns(
    pl.col("instrument_id").shift(1).over("product", "tenor").alias("_prev_id"),
    pl.col("close").shift(1).over("product", "tenor").alias("_prev_close"))
rolls = s.filter(pl.col("_prev_id").is_not_null() & (pl.col("instrument_id") != pl.col("_prev_id")))
ratio = (pl.col("open") / pl.col("_prev_close")).alias("ratio")
```
```python
# PIT-correct as-of join of a slower series stamped at bar OPEN; staleness on the publication clock
slow = slow.with_columns((pl.col("timestamp") + pl.duration(hours=8)).alias("timestamp"),
                         (pl.col("timestamp") + pl.duration(hours=8)).alias("published")).sort(["symbol", "timestamp"])
joined = fast.sort(["symbol", "timestamp"]).join_asof(slow, on="timestamp", by="symbol",
                                                      strategy="backward", check_sortedness=False)
stale = joined.filter((pl.col("timestamp") - pl.col("published")).dt.total_hours() >= 8)
# Binance funding from premium, paired with its settlement
P = pl.col("premium_index_close")
f = df.with_columns((P + (0.0001 - P).clip(-0.0005, 0.0005)).alias("est_funding_rate"),
                    (pl.col("timestamp") + pl.duration(hours=8)).alias("settles_at"))
```
```python
# Scale-invariant leakage test; untrusted-return rule before any compounding
feat_ret = pl.col(feature).pct_change(); next_ret = pl.col("close").shift(-1) / pl.col("close") - 1
corr = df.with_columns(feat_ret.alias("f"), next_ret.alias("n")).drop_nulls(["f", "n"]) \
         .filter(pl.col("f").is_finite() & pl.col("n").is_finite()).select(pl.corr("f", "n")).item()
returns = returns.with_columns(((pl.col("ret") > 1.0) | (pl.col("prev_volume") == 0)).alias("suspect"))
```
```python
# Seam-rescaled stitch with source attribution (16/17)
overlap = hist.select(ts.dt.date().alias("day"), "close").join(
    recent.select(ts.dt.date().alias("day"), "close"), on="day", suffix="_recent").sort("day")
assert overlap.height, "legs share no date; fetch recent leg from before the seam"
scale = overlap["close_recent"][-1] / overlap["close"][-1]
hist = hist.filter(ts.dt.date() <= seam).with_columns(
    pl.col("open", "high", "low", "close") * scale, pl.col("volume") / scale, pl.lit("wiki").alias("source"))
recent = recent.filter(ts.dt.date() > seam).with_columns(pl.lit("yahoo").alias("source"))
combined = pl.concat([hist.select(cols), recent.select(cols)]).sort("timestamp")
```
```python
# DataManager + HiveStorage + safe daily delta (17 §5 / 18 / 19)
Universe.add_custom("etf_momentum", ["SPY", "QQQ", "IWM", "TLT", "GLD"])
storage = HiveStorage(config=StorageConfig(base_path=path, compression="zstd", partition_granularity="month"))
dm = DataManager(storage=storage, enable_validation=True)
for sym in Universe.get("etf_momentum"):
    key = dm.load(sym, "2024-01-01", AS_OF_DATE, provider="yahoo")      # -> "equities/daily/SPY"
    meta = dm.get_metadata(sym)
df = storage.read(key, start_date=datetime(2024, 1, 1), end_date=datetime(2024, 12, 31)).collect()
result = OHLCVValidator(max_return_threshold=0.5).validate(df)           # result.passed, .error_count
gaps = GapDetector(exclude_weekends=True).detect_gaps(df, frequency="daily")
rows = update_through_last_complete_bar(dm, storage, sym, provider="yahoo", lookback_days=7)
```
Also: `dm.fetch(symbol, start, end, frequency="daily", provider=None)`; `dm.batch_load(symbols, start, end, provider, max_workers=4, fail_on_partial=False)` → stacked frame with `symbol` (the book's multi-asset format); `dm.batch_load_universe("sp500", ...)`; `dm.update(symbol, lookback_days=7, fill_gaps=True, provider=None, initial_load_days=365)` (avoid blind); `dm.update_all()`; `dm.list_symbols()`; `storage.list_keys()/partitions(key)/exists(key)/write(df, key, metadata, preserve_metadata=True)`; `Universe.SP500` (503), `NASDAQ100` (100), `CRYPTO_TOP_100`, `FOREX_MAJORS` (28), `Universe.get(name)`, `list_universes()`, `add_custom`, `remove_custom`.

CLI (`ml4t/data/cli/core.py`: `fetch`, `update`, `validate`, `info`, `list_data`; `cli/batch.py: update_all`; `cli/futures.py: update_futures`): `ml4t-data fetch AAPL --start 2024-01-01 --end 2024-12-31`; `ml4t-data fetch SPY QQQ IWM TLT --provider yahoo --output data/etfs.parquet`; `ml4t-data update --all --storage-path ./data`; `ml4t-data validate ./data/etfs.parquet`; `ml4t-data list --storage-path ./data`; `ml4t-data info --providers`.

Config contract (`data/etfs/market/config.yaml`): `etfs:` with `provider: yahoo`, `start: '2006-01-01'`, `end: '2025-12-31'`, `frequency: daily`, `storage_path: etfs/market`, `tickers:` of 9 groups (`us_equity_broad`, `us_sectors`, ...) each `description` + `symbols` (100 ETFs; Chapter 6 filters by $50M/day liquidity, 10+ years, correlation clustering, effective bets; no leveraged/inverse/crypto). Helpers in `utils.downloading`: `load_section`, `resolve_storage_path`, `flatten_group_values(groups, "symbols")`, `resolve_data_dir` (CLI > `ML4T_DATA_PATH` > `<repo>/data`), `create_base_parser` (`--data-path --dry-run --force --verbose`), `atomic_write_parquet`, `save_dataset_profile`, `last_complete_daily_bar`.

Canonical Polars idioms (`22_pandas_polars_benchmark`): rolling `pl.col("close").rolling_mean(W).over("symbol")`; multi-horizon `pct_change(h).over("symbol")`; rolling Sharpe `rolling_mean(63)/rolling_std(63) * sqrt(252)`; EMA `ewm_mean(span=20, adjust=False).over("symbol")`; daily resample `group_by([ts.dt.date(), "symbol"]).agg(open.first(), high.max(), low.min(), close.last(), volume.sum())`; cross-sectional z `(r − r.mean().over("timestamp"))/r.std().over("timestamp")`; percentile rank `rank().over("timestamp") / count().over("timestamp")` (count, not max rank); anti-join `how="anti"`; lazy `pl.scan_parquet(p).filter(...).group_by(...).agg(...).collect()`.

Benchmark timing pattern: `os.environ["BENCHMARK_SCALE"] = "L"` before import; `write_time, _ = time_write(lambda: df.write_parquet(p))`; `read_time, out = time_read(lambda: force_materialize_polars(pl.read_parquet(p)))`; `validate_result(out, total_rows, "Parquet read")`; `results.append(BenchmarkResult("Parquet", "read", read_time, p.stat().st_size, total_rows))`; `save_benchmark_results(results, "formats")`.

Repo conventions: declare parameters in one cell (Papermill overrides, e.g. `MAX_SYMBOLS`); `show_plotly_with_alt` / `show_with_alt`; `set_global_seeds(SEED)`; `get_output_dir(2, name)`; `from utils import ML4T_DATA_PATH`; run `uv run python 02_financial_data_universe/<nb>.py`; tests `uv run pytest tests/test_chapter_notebooks.py -k "02_financial_data_universe"`.

## Evidence from the book
- **WIKI panel:** 3,199 companies, 1962 to 2018-03-27; 777 leavers, every one in 2014 or later; ~52 years without an exit; nulls and OHLC pass ≥ 99.99%; ~3,200 symbols vs 4,000–5,000 listed since 2010 (peak ~8,090 in 1996, World Bank). Passes end-date and delisting-rate checks, fails exit-span; completeness warns on frozen coverage after 2014 (`01`, `15`).
- **AAPL actions:** 54 cash dividends, splits incl. 2014-06-09 7:1; raw cumulative return understates adjusted by "most of the return"; `apply_corporate_actions` vs Quandl `adj_close` passes within the accumulation bound on AAPL only (`02`).
- **ETF panel:** from 2006-01-03; late starters, no early enders; strict OHLC flags 760 of 470,662 rows, largest breach 2.01e-16 vs eps 2.22e-16; volume spans > 10× across groups; closes 5.60 to 862.50 (`03`). README says 50 ETFs, notebook 100 (inference: README stale).
- **CME futures:** 30 products, 7 buckets (Equity 4, Rates 4, Energy 4, Metals 4, FX 6, Grains 5, Livestock 3); three tenors; a few products enter late; ES front cum_ratio near one; 20–24 bars/session; daily OHLC breaches 0 (`04`, `05`).
- **Continuous ES:** individual contracts daily (`rtype` 35) while vendor continuous is hourly; two-digit-year padding misdates every contract; third-Friday rule holds for all above-median-volume contracts; no-rollback changes zero days; volume and calendar rolls disagree a session or two around rolls; unaligned comparison differs > 10× the aligned; aligned series identical to the cent outside the handover window; largest gaps March/June 2020; handover gap in bps grows ~10× from first four to last three years (`06`).
- **Options:** ~70% Converged, ~10 codes; 2020 ATM IV multiplies within weeks, p90 > 100% on some days; IV peak precedes 2020-03-23 trough; theta ≤ 0 falls short by a small expected amount; median spread flat across moneyness, p90 > 2× in wings; 5-day IV change vs forward return small, negative, stable on non-overlapping windows, sign shared by most of 8 names; quintiles non-monotonic (`07`). 1y Treasury fell > 1 pp in 2020; `head(2000)` = one day; per-date rate cuts delta residual substantially; residual correlates with gamma, spot gap a fraction of a percent; Converged IV recovery: every option solves, median diff ≈ 0.2 vol points, most within 0.01; < $0.05 time value ~100× rarer among Converged rows (`08`). AAPL 2019 straddle: contract changes most days, mean run < 2 days; naive vs same-contract 10-day correlation low; zeroed vs held differ by tens of pp over the year; long and daily-reset short both negative (`09`).
- **Crypto:** 19 USDT-M perps from 2020 (SUI 2023); premium std ≈ 0.1%; 1 in 7 closes exactly zero, < 1% with all four fields zero, zero rate monotone in liquidity over > 50×; OHLC breaches 0; exact join matches 1 in 8; ~0.5% of hourly bars read a premium ≥ 8h old, one symbol > 2 months stale, a multi-day universe-wide outage (`10`). BTCUSDT 2020–2025: published formula beats `clamp(P)+I` several-fold; > 1/3 of settlements exactly at I; correlation peaks at t+8h; realized rates far outside ±(I±c), extremes differ 10× across contracts; close beats bar mean/midpoint; mean premium negative everywhere, mean funding positive nearly everywhere (`11`).
- **FX:** 20 pairs; majors mid-table by tick volume; OHLC clean; nearly every step one bar, rest Friday→Sunday, remainder dominated by Dec 24/31 closing all 20 pairs; 12 UTC vs 6 New York hours; UTC-day aggregation manufactures thousands of one-bar days and disagrees with session closes by several bps on most days (`12`).
- **Quality framework:** five symbols pass; injected faults fire `price_consistency`, `negative_volume`, `null_values`; MAD flags several times more than Z, IQR fewest; largest moves are splits and an earnings crash; PSI driven by one bin beside the zero-return tie group; zero-return share thins in the late 1990s, not at 2001 decimalization; only gap > 5 days is post-9/11; 25% rule recalls all splits, ~1/3 false positives (`13`).
- **PIT (SPY):** centered windows have ~half their inputs in the future and track the close more closely; naive correlations ≈ 0; scale-invariant: trailing low, centered elevated, tomorrow's close ≈ 1; 1 row = 1–5 calendar days; macro series change on the 1st; unemployment disagrees ~3 days in 4, max > 10 points spring 2020; GDP advance revised between 2023-11-01 and 2024-06-01 vintages (`14`).
- **Survivorship 2014–2018:** untrusted returns a small fraction of 1% of rows (HERO 2015-11-06, PCO 2017-12-05, EXXI); phantom sessions = six holidays + Good Friday 2017 + 2014-01-02; raw returns show survivors *underperforming*, repaired returns show overstatement; terminal = 0 moves the bias well under a point; scenario spread exceeds the MC band by 10×; leavers underperform survivors (`15`).
- **Providers / pipeline:** AAPL 2017 Yahoo vs WikiPrices exact matches = 0, uniform ratio = 2020 4:1 split; unrescaled stitch steps at 2018-03-27; equity validator flags the Sept 2001 closure on every symbol; crypto volume-by-hour shows 0/8/16 UTC structure (`16`, `17`). Incremental: 502 rows + exactly the new sessions, no phantom bars; extreme-return WARNING on each series' largest move = check working (`18`, `19`).
- **Benchmarks (L, warm cache, `uv run`):** Parquet a fraction of CSV size; in-memory size identical across formats; projection speeds Parquet/Feather ∝ columns, CSV barely; HDF5 ~71 MB vs ~79 MB with newer pandas; fastest read, fastest write and smallest file are different formats (`20`). Columnar vs row engines separate on scans; partition/index engines win ranges; QuestDB/InfluxDB not free with durability timed; kdb+ ~40× "faster" before timing persistence (`21`). Speedup read off the run; two pairs had silently drifted before the witness; memory ratio dominated by `symbol` representation (`22`).
- Caveats: most numbers print at runtime and are deliberately not quoted; momentum/dividend illustration is stylized; only AAPL validated for adjustments; terminal scenarios literature-calibrated; macro lags round numbers; Greeks residual bounded, not attributed.

## Related references
- `chapters/01_process_is_edge.md` — why data defects manufacture false edges; the process this chapter's checks belong to.
- `chapters/03_market_microstructure.md` — tick/quote data, bars and liquidity measures built on the session-correct bars defined here.
- `chapters/04_fundamental_alternative_data.md` — point-in-time fundamentals and vintages continue the PIT discipline of `14_point_in_time_validation`.
- `chapters/06_strategy_definition.md` — builds the ETF trading universe from the 100-ETF candidate pool; `setup.yaml` `decision.session_calendar`.
- `chapters/07_defining_the_learning_task.md` — same-contract option labels, label horizons, overlap and purging.
- `chapters/08_financial_features.md` — term structure, roll yield and carry from `raw_*`; momentum/vol from `adj_*`; IV surface and premium features.
- `chapters/09_model_based_features.md` — IV-signal information coefficient over a real cross-section (the NB07 hypothesis).
- `chapters/11_ml_pipeline.md` — leakage-safe pipelines; the trailing-only transform rule.
- `chapters/16_strategy_simulation.md` — backtests on session-correct returns, held-contract option series, funding arbitrage.
- `chapters/18_transaction_costs.md` — option roll costs and group-specific liquidity assumptions.
- `chapters/25_live_trading.md`, `chapters/26_mlops_governance.md` — freshness monitoring, incremental updates, data versioning in production.
- `case_studies/etfs.md` — Yahoo-only ETF rotation universe seeded by `16_provider_comparison`.
- `case_studies/crypto_perps_funding.md` — premium/funding conventions, `funding_data.py`, bar-open stamps.
- `case_studies/cme_futures.md` — continuous daily file (`adj_*`, `raw_*`, `cum_ratio`) consumed downstream.
- `case_studies/sp500_options.md`, `case_studies/sp500_equity_option_analytics.md` — convergence filter, straddle selection (`materialize_options.py`), same-contract labels at scale.
- `case_studies/fx_pairs.md` — 5 PM New York session calendar and gap taxonomy.
- `case_studies/us_equities_panel.md`, `case_studies/us_firm_characteristics.md` — WIKI panel limits, survivorship window 2014–2018.
- `libraries/ml4t_data.md` — adjustments, validation, anomaly, providers, DataManager, HiveStorage, update manager, CLI.
- `libraries/ml4t_diagnostic.md` — leakage and drift checks that extend the PIT and PSI tools.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `workflow.md`, `companion_repo.md` — cross-cutting indexes that cite this chapter.
- Further reading: Eckbo & Lithell (2025) JFQA Table A1 Panel B (delisting scenarios); Shumway (1997) delisting bias; Beaver, McNichols & Price (2007) delisting returns; Doidge, Karolyi & Stulz (2017); Elton et al. (1996), Carhart et al. (2002) survivorship; Lopez de Prado (2018) AFML; Lipton & Lopez de Prado (2020); Luo et al. (2014) Seven Sins; Joubert et al. (2024) three types of backtests; Fong et al. (2017) liquidity proxies; Karnaukh et al. (2015) FX liquidity; Makarov & Schoar (2020), He et al. (2024), Cong et al. (2023), Vidal-Tomás (2022) crypto; Easley et al. (2021), O'Hara (2015) microstructure; Ekster & Kolm (2020) alternative data; Berg et al. (2022) ESG divergence; Jones on US equity market data; SEC (2020); Konstantinov (2025); Ng et al. (2025); Lehoczky & Schervish (2018); Loughran & McDonald (2020).

## Glossary
- **Backward adjustment** — scale history so the latest price is unchanged and earlier prices absorb later splits/dividends; **total-return adjusted** = split + dividend, returns equal reinvested holding.
- **Accumulation bound** — tolerance = steps × machine epsilon for a recursive computation; **ulp / float64 eps** = 2.22e-16.
- **OHLC invariants** — high ≥ low/open/close, low ≤ open/close, volume ≥ 0.
- **Survivorship bias / universe completeness** — overstatement from excluding leavers / whether entries (IPOs) are recorded; **delisting return (`DLRET`) / code (`DLSTCD`)**; **terminal return**; **phantom session** (holiday prints by a sliver of the universe).
- **Product / contract / continuous series / tenor** — underlying; specific expiry (`instrument_id`); spliced history; c0 front, c1, c2. **Roll**, **no-rollback constraint**, **Panama (additive)** vs **ratio (multiplicative)** adjustment, **handover window**, **calendar spread**, **carry** ≈ index × (financing − dividend yield) × time, **`rtype`** (32=1s, 33=1m, 34=1h, 35=1d), **month code** F–Z.
- **Session date** — date a session ends (CME 16:00 CT; FX 17:00 New York, the value-date rollover).
- **Option chain / moneyness (strike/spot) / IV / smile, skew, term structure, surface / `iv_convergence` / put-call parity / Greeks / time value vs intrinsic.**
- **Constant-maturity series** — daily reselection near a target maturity/delta; **straddle**; **same-contract holding return**; **held-contract return** (yesterday's contract repriced today).
- **Perp / premium index / impact bid-ask / dead zone / funding rate** `F = P + clamp(I − P, −c, c)` with I = 0.01%/8h, c = 0.05%; positive → longs pay shorts.
- **As-of join** (backward, latest prior row) and **staleness** (age vs publication schedule); **tick volume**; **direct / indirect / cross pair**.
- **MAD** (÷ 0.6745 to σ scale); **PSI** Σ (p_cur − p_base) ln(p_cur/p_base); **quarantine**.
- **PIT correctness / event time vs knowledge time (bitemporal) / vintage / period stamp vs release date / centered moving average / scale-invariant leakage test / execution lag.**
- **Canonical OHLCV schema** — `[timestamp tz-aware, symbol upper, open, high, low, close, volume Float64]`, sorted, unique; **adjustment basis**; **seam**; **rescale factor**; **source attribution**; **fallback fetch**.
- **Placeholder bar / last complete bar / retreat / `lookback_days` / update strategies / gap / calendar-dense vs sparse series / freshness (≤ 3 fresh, > 7 stale).**
- **Hive partitioning / generation (`CURRENT` pointer) / partition pruning / storage key (`asset_class/frequency/SYMBOL`) / Universe.**
- **Warm-cache read / materialisation / column projection / range query / ASOF join / anti-join / durable-to-queryable / client-side row construction / splayed table / hypertable / witness / geometric-mean speedup / predicate pushdown / benchmark scale / funding window (0/8/16 UTC) / Papermill.**
