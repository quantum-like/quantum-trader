# Chapter 16: Strategy Simulation

> A backtest is a structured attempt to falsify a strategy under realistic assumptions; only what survives that attempt is reportable, and the protocol (fill timing, share type, fill ordering, rebalance mode, cost model) must be published with the result or nobody, including the author six months later, can reproduce it. The chapter builds the simulator before the strategy, writes the trading protocol down as a checklist, and measures what each protocol rule is worth so that a disagreement between two engines becomes readable rather than mysterious. It fixes an auditable non-ML ETF momentum/regime baseline scored against a term sheet written before the run, defines the reporting stack (gross and net, protocol-matched benchmark, exposure, closed round trips, reconciliation, turnover, break-even cost), and diagnoses economic value through point-in-time regime slices and a cost-sensitivity sweep. It closes with search-aware inference: Sharpe intervals and MinTRL for one fixed strategy, then DSR, PBO and the Rademacher Anti-Serum for the family that was actually searched. The position: protocol before library, benchmark through the same simulator, and always report how many strategies were looked at before this one.

## When to use this reference

- Writing, reviewing or auditing any backtest (ETFs, equities, crypto, FX, CME futures) before trusting its numbers.
- Choosing between vectorized/array simulation (VectorBT, `weights.shift(1) * returns`) and the event-driven `ml4t-backtest` engine.
- Configuring `BacktestConfig` (execution mode, share type, fill ordering, cash buffer, rebalance mode, commission and slippage units) or explaining why two engines disagree on the same strategy.
- Converting model predictions or scores into positions (fixed threshold vs trailing percentile vs cross-sectional percentile).
- Assembling a performance report: gross vs net, benchmark on the same protocol, exposure, round trips, cost reconciliation, break-even cost rate.
- Slicing returns by regime (volatility/trend states) or attributing a drawdown by state without leaking the return into the label.
- Running a cost-sensitivity sweep or quoting a break-even fee.
- Deciding whether a reported Sharpe is credible: confidence interval, PSR, MinTRL, Deflated Sharpe Ratio, PBO/CSCV, RAS; counting trials after a parameter grid or model search.
- Backtesting futures: contract multipliers, per-contract commission, session boundaries, static margin specs.
- Implementing stateful logic (Kelly sizing from realized P&L, pairs hedged from the lead-leg fill, drawdown circuit breakers) that cannot be an array operation.
- Reading or regenerating the cross-framework parity audit (LEAN, Backtrader, Zipline, VectorBT Pro/OSS vs ML4T).

## Core ideas (the why)

- **Falsification standard (16.1).** A result counts when it survives an attempt to break it. Leakage checks, execution realism, cost sensitivity and regime diagnostics are the evidence, not hygiene applied afterwards. Every notebook in the chapter is an instance of this.
- **Protocol before library (16.2).** Decide signal timing, rebalancing, sizing, fills, costs, constraints, data availability and benchmark first, then pick whichever implementation makes the protocol easiest to write down and check. "Reaching for a library first and inferring the protocol from what it happens to do is how assumptions get made by accident" (`16_strategy_simulation/06_framework_parity`). Many published disagreements are protocol disagreements, not signal disagreements.
- **Event order is part of the model.** The close-to-next-open gap is a modelling choice; encode it as a shift of the whole weight matrix or `ExecutionMode.NEXT_BAR`, never as a condition inside a loop. "Validity comes before size": a close-read signal cannot fill at that close, and no rebalance mode, cost model or sizing rule repairs it (NB07).
- **Vectorized vs event-driven is a question about state, not libraries (16.3).** Array form: `signals[t]`, `weights[t]` from observed data and fixed rules. Sequential form: `action[t] = g(data[0:t], positions[t-1], fills[0:t-1], equity[t-1])`. Neither is realistic by default and neither is the reference implementation; trust the one that encodes the intended protocol.
- **The benchmark must run through the same simulator.** Fills-and-fees strategy returns compared with a frictionless return series compares two protocols, not two strategies (NB01, NB03, NB04, NB09). The benchmark must also face the same warmup: a buy-and-hold that buys on bar one holds through a window the strategy could not trade.
- **A term sheet written before the run** stops criteria drifting to fit the output. A rule that misses its own term sheet remains a useful fixed reference for later model comparisons (16.4).
- **Correlation is not parity.** Two engines' daily returns can correlate about 1 while a small constant daily gap compounds into a large wealth gap. Compare levels over time, through one metric implementation (NB06).
- **A configuration field is a request, not an action.** Change it, confirm something moved, then attribute. One-factor effects do not add when they change what the account holds; report the interaction residual as a quantity (NB07).
- **Contract metadata is accounting (NB02).** Multipliers, tick size and margin are part of the P&L model, not behavioural knobs.
- **Fees paid are not the whole cost of fees (16.6).** Cash held back to pay fees is exposure forgone every subsequent day and compounds.
- **The conversion rule is part of the strategy (NB08).** One prediction set yields streams that differ in activation and flip frequency; a backtest return is a return on the stream, not on the model. A constant cutoff on an uncalibrated regression score is a bet on the score's scale. A state transition is not a trade: turnover needs position sizes.
- **Keep four kinds of evidence apart in a report (NB09):** the portfolio's own path; its relation to a benchmark that faced the same conditions; how much of the time capital was at risk; what closed round trips looked like. Mixing them is how a report flatters a strategy.
- **A state label must be readable before the return it labels (NB10).** Statistics through t-1, compared with the expanding median of their own past, shifted one bar before joining. Market conditions describe the market, not the strategy's exposure.
- **Report an interval, not a number (NB11).** The SE of an annualized Sharpe on one year of daily data is about 1. Single-strategy statistics cannot see a search: every line computed correctly on the best null of 30 still gives the wrong conclusion.
- **Selection bias is measurable (NB12, NB13).** Expected maximum null Sharpe rises with trials; DSR tests against that benchmark, PBO measures how often the in-sample winner ranks below median out of sample, RAS charges for the complexity of the whole candidate class. They answer different questions; use several, with thresholds set before looking.
- **A net Sharpe is one point on a curve (NB14).** The reportable thing is the cost curve and its slope at the baseline fee.
- **Engine parity is a narrow claim (NB15-18):** target replay on frozen inputs with costs and position rules disabled; unsupported pairs are disclosed, never approximated; timings are dated case-and-machine measurements, not rankings.
- **IC alone is insufficient (Ch7 §7.5 link):** prediction quality and trading quality diverge.

## Method recipes (the how)

### Backtest protocol checklist (16.2)

Write every row down before touching a library; publish it with the result.

| Protocol item | Question | Chapter defaults |
|---|---|---|
| Signal timing | When is the signal readable? | Last session close (month-end for the ETF baseline); conditions shifted one row |
| Execution / fill price | Where does the order fill? | Next session open: `ExecutionMode.NEXT_BAR`, `ExecutionPrice.OPEN`; mark at close |
| Rebalancing | How often are targets refreshed? | Monthly (NB01); every 21 sessions (NB02); every session (NB06) |
| Sizing | How are targets turned into orders? | Equal-weight top-N; 95% of capital for single-asset; `floor(alloc / (price x multiplier))` for futures |
| Costs (units!) | Per dollar traded, per share, per contract? | 5 bps per dollar (ETF); 10 bps commission + 5 bps slippage per fill (crypto); $2 per contract per side (futures) |
| Constraints | Short, leverage, cash? | Long-only, no borrowing (NB01); `allow_short_selling`, `allow_leverage` with margin (NB02); `reject_on_insufficient_cash=True` |
| Data availability | Point-in-time? | Backward as-of join; matched-observation age <= `MAX_SLOPE_AGE_DAYS=4` |
| Benchmark | Same simulator? | 60/40 SPY/AGG or matched buy-and-hold through the same engine, dates, fees, slippage, capital fraction, warmup |
| Plumbing | Does a no-information signal earn nothing through the same path? | Seeded random predictions through the identical engine, costs and sizing: PASS iff `\|Sharpe\| < 1.5`, negative after costs at intraday cadence (pattern under Code patterns) |

One baseline, not two: the expectation of a random equal-weight top-k portfolio drawn from the universe is the equal-weight universe return (each name enters with probability k/N at weight 1/k), so a random-top-k plumbing baseline (`run_plumbing_test`, `top_k=20`) and an EW-universe benchmark are the same number in expectation. Use one, say which, and read a long-only random book's Sharpe against the EW-universe Sharpe rather than zero; a random long-short book is the one that should sit near zero before costs.

### First-principles simulator and the ETF baseline (`16_strategy_simulation/01_backtest_first_principles`, §16.2/16.4)

1. Constants: `START_DATE="2010-01-01"`, `END_DATE="2024-01-01"`, `LOOKBACK_PERIOD=126`, `TOP_N=3`, `REGIME_THRESHOLD=0.005` (10Y-2Y spread; just above zero so a flat curve is risk-off), `INITIAL_CASH=100_000.0`, `FEES=0.0005` (5 bps per dollar traded, standing in for commission plus about half the bid-ask on a liquid US ETF), `DEFENSIVE_MIX={"AGG":0.60,"TLT":0.40}`, `BENCHMARK_MIX={"SPY":0.60,"AGG":0.40}`, `MAX_SLOPE_AGE_DAYS=4`. Universe (from NB10): `["SPY","QQQ","IWM","EFA","EEM","AGG","TLT","GLD","VNQ","DBC"]`.
2. Score at month-end t using closes through t only: `m_{i,t} = (P_{i,t}/P_{i,t-L} - 1) / (sd(r_{i,t-L+1:t}) * sqrt(252))`, L=126. Rank cross-sectionally per date; assert no full-sample statistic enters.
3. Regime: spread > threshold -> risk-on -> equal-weight top 3; spread <= threshold (including inverted) -> risk-off -> defensive mix.
4. Macro join: Polars backward as-of join of each ETF date to the most recent yield observation dated on or before it. Row count only answers "did the series start in time"; check the age of the matched observation against the declared 4-day tolerance to detect a stalled feed.
5. Timing: executed-weight matrix = target matrix shifted down one row; first row = defensive mix. Calendar check: rebalance count == months covered + 1.
6. Return identity (weights tradable before t): `R_{p,t} = sum_i w_{i,t} r_{i,t}`. A close-derived signal cannot use it with close-to-close returns, so simulate next-open fill and mark at close.
7. `simulate_portfolio(opens, closes, target_weights, rebalance_mask, initial_cash, fee_rate) -> dict[str, np.ndarray]`: on each rebalance sell first, deduct fees, scale purchases to remaining cash; EOD equity at close; long-only, no borrowing. The alternative "simultaneous full-notional rebalance" silently finances fees from negative cash.
8. `calculate_metrics(returns, periods_per_year=252)`: Sharpe/Sortino from mean periodic excess return with zero risk-free rate (matches `PortfolioAnalysis`); CAGR kept separate. `compute_drawdown(equity, initial_cash)`: percentage drawdown from running peak.
9. Term sheet (3 lines, written first): risk-adjusted return floor; `abs(Max Drawdown %) < 25`; total return over sample > 60/40. Score pass/fail in a table. Reconcile headline Sharpe, Sortino, CAGR, vol and drawdown with `PortfolioAnalysis` to display precision using assertions.
10. Regime slice is descriptive attribution by contemporaneous yield-curve state; no causal claim.
11. Reusable port: `16_strategy_simulation/_etf_baseline.py`, parity-tested by `tests/test_etf_baseline_parity.py`. NB14 imports it (`from _etf_baseline import ...`); NB10 reproduces the baseline with its own `simulate_portfolio`, `expanding_past_median`, `combine_state` and `summarize_returns` and does not import the helper.

### Futures protocol (`16_strategy_simulation/02_futures_backtesting`)

- P&L = (exit - entry) x qty x multiplier; costs in $ per contract; sessions overnight (5 PM CT open); short side needs margin, not stock loan; notional = contracts x price x multiplier.
- `load_contract_specs_from_yaml(yaml_path) -> dict[str, ContractSpec]` (point value + margin %; margin anchored to 2025-12-31 prices and a 2026-05-16 CME rates snapshot, applied statically over 2018-2023). Identity check: point value == tick value / tick size.
- `DataFeed` requires `timestamp` (Datetime), `symbol`, `open`, `high`, `low`, `close`, `volume`; alias CME `product` to `symbol`; timestamp labels the 4 PM CT session close (`load_cme_futures()` session-daily bars).
- Signals and marks on ratio-adjusted continuous OHLC; contract counts from raw front-contract close (avoids splice jumps; no roll orders or roll costs modelled).
- Signal: 63-session trailing return, cross-sectional rank, long top `long_n=2`, short bottom `short_n=2`, `rebalance_every=21`. Sizing: `contracts = floor(allocation / (price x multiplier))`; skip the product if one contract exceeds the allocation.
- Functions: `compute_target_contracts(data, specs, long_n, short_n, capital_base)`, `submit_contract_deltas(target_quantities, broker)` (signed deltas current -> target), `FuturesMomentumStrategy(Strategy)`.
- Config: `CommissionType.PER_CONTRACT` at $2 per contract per fill side (teaching number), `allow_short_selling=True`, `allow_leverage=True` (activates per-product margin % from `ContractSpec`), 5 bps percentage slippage, `ExecutionMode.NEXT_BAR`; pass `contract_specs=` to `Engine`, which threads it to `Broker`.
- Counterfactual: `TargetScheduleStrategy(schedule)` replays the exact signed-contract schedule with all multipliers = 1 so the equity path cannot feed back into sizing; isolates valuation/margin/P&L conversion error.
- Also produced: trade tables with cost decomposition, product and sector P&L attribution, and a six-product vs full-CME-universe comparison under the same rule and costs.

### Signal-to-position conversion (`16_strategy_simulation/08_signal_method_comparison`, §16.2)

| Rule | Active when | Relative to | Use when |
|---|---|---|---|
| Fixed threshold | predicted return > 0 | the score's scale | scores calibrated on a fixed scale |
| Trailing percentile | score >= p-th percentile of the same symbol's previous `lookback` scores | own history (can hold every symbol at once) | drifting scale, stable shape, single asset |
| Cross-sectional percentile | score in top (100-p)% of symbols quoted that date | the universe (always holds a fixed share) | broad universe; needs no fitted time-series threshold |

- Settings: `ROLLING_WINDOWS=[21, 42, 63]`, `PERCENTILES=[75, 80, 85, 90, 95]`, `OPERATING_PERCENTILE=90` (asserted in grid), `OPERATING_WINDOW=max(ROLLING_WINDOWS)=63`. Case studies: `{"crypto_perps_funding": 8-hourly, 3 obs/day; "etfs": daily}`.
- Select the prediction set by a rule indifferent to what you measure: `select_registered_prediction(db_path, family, label)` picks widest date coverage among validation GBM sets, ties by hash (`family="gbm"` default in `load_registered_predictions`). Read the primary label from the case study's `setup.yaml`; open the registry read-only (SQLite `mode=ro`, one `SELECTOR_SQL` query resolving one hash to one path).
- `validate_prediction_set(pred_path, metadata, min_observations)`: required columns; no duplicate (timestamp, symbol); no nulls; every symbol >= longest lookback obs; >= 2 symbols per date; recompute daily rank IC and `artifact_n_days` beside `registered_n_days`.
- `signal_diagnostics`: activation frequency (share of symbol-date rows active) and per-symbol state-transition frequency; `compare_signal_methods` shares one schema. Sweep lookback at fixed percentile, then percentile at the shortest lookback (21). Report per-method grid mean and spread: the spread is sensitivity to the method's own settings; the mean describes the grid, not the method. Scatter: activation rate (x) and transition rate (y) rise together; a rule above the trend is one whose cutoff the scores keep crossing. Output to `get_output_dir(16, "signal_method_comparison")`. No returns, no costs by design.

### Vectorized backtest with VectorBT (`16_strategy_simulation/03_single_asset_vectorbt`)

- BTC/USDT UTC-day aggregation (a calendar convention, not an exchange session); assert unique keys and positive, internally consistent OHLC.
- RSI via VectorBT indicator over multiple parameters at once (`vbt.RSI.run`, inference). Entry: prior-close RSI < lower; exit: prior-close RSI > upper; shift both conditions one row so they are eligible at the next UTC open.
- `vbt.Portfolio.from_signals(close, entries, exits, fees=..., slippage=..., size=..., init_cash=...)` (kwargs are inference); pandas at the boundary. Suppress VectorBT's implicit frictionless close-to-close benchmark; build a matched-cost buy-and-hold (same capital fraction, next-open price, fee, slippage; stays open at sample end so no terminal exit cost).
- Threshold sweep: heatmap with window fixed, diverging scale centred at zero; read for flatness. The highest-Sharpe cell is not a deployment choice. Cross-check total return and max drawdown with `PortfolioAnalysis`.

### Event-driven backtest with ml4t-backtest (`16_strategy_simulation/04_single_asset_ml4t_backtest`; rebuilt for reporting in `09_performance_reporting`)

- Rule (NB04, rebuilt unchanged in NB09): `compute_rsi(close, period=14)` as simple rolling mean of gains/losses (matches VectorBT default; not Wilder); enter when RSI < 30, exit when RSI > 70; `position_size=0.95`; 10 bps commission + 5 bps slippage; `ExecutionMode.NEXT_BAR`. Input: BTC/USDT, three 8-hour bars per UTC day aggregated. `ShareType.FRACTIONAL` and `calendar="crypto"` (365 periods/year) are set in both notebooks (the notes state them for NB09; verified in the NB04 source). The first valid RSI appears only after the full warmup window, so the strategy trades nothing before it (stated in NB09).
- Strategy conventions: parameters as constructor args; `broker.get_position(symbol)` returns `None` or `Position` (`quantity`, `entry_price`, `unrealized_pnl`); indicators via `context.get()` (check None/NaN); size explicitly with `broker.order_target_percent()`; handle early bars lacking history.
- Components: `DataFeed` (price frame + optional context frame aligned on `timestamp`), `Strategy`, `Engine` (orchestrates, applies costs), `Broker` (orders, positions, state), `ExecutionMode` (`SAME_BAR` needs a defensible intrabar timing argument).
- Reconciliation identity per closed round trip: `net P&L = reference-price P&L - slippage cost - commission` (strip entry/exit slippage from recorded fill prices first). Aggregate over closed trades only; assert equality with the change in account value.
- Benchmark, two versions. NB04 `BuyAndHoldStrategy(position_size=0.95)`: a single entry on the first available bar through the same engine, timing and costs (`len(fills) == 1` asserted), so it holds through the RSI warmup, which is the pitfall NB09 fixes; read NB04's comparison as cost- and timing-matched, not warmup-matched. NB09 `BuyAndHoldStrategy(position_size=0.95)`: returns until `context.get("rsi")` is valid, then enters on the first bar the RSI strategy could also have traded, with the same bars, exposure, next-open fills, fractional units, calendar and costs.
- `MAEMFEAnalyzer`, `TradeAnalyzer` (`ml4t.backtest.analytics`): per-trade excursions, descriptive only. `SessionConfig`, `compute_session_pnl` (`ml4t.backtest.sessions`): observations after 17:00 CT carry the next session's label.
- Known sizing difference vs VectorBT: VectorBT resolves % size at the execution row's open; ml4t-backtest converts the target to units at the decision close, queues, fills next open, so the overnight gap moves realized weight off target.

### Stateful strategies (`16_strategy_simulation/05_stateful_strategies`)

Rule of thumb: if the whole protocol can be precomputed as aligned arrays, use arrays; if later actions depend on mutable portfolio state (realized P&L sizing, cross-asset cash/margin, equity-path exposure, reactive orders, pyramiding), use sequential simulation. Signals: `make_signals(prices, signal_fn="momentum"|"random"|"alternating", lookback=20, seed=42)`.

| Pattern | Class and parameters | Mechanism |
|---|---|---|
| Adaptive half-Kelly | `AdaptiveKellySizingStrategy(signal_column="signal", entry_threshold=0.01, exit_threshold=-0.01, base_size=0.10, min_size=0.02, max_size=0.25, kelly_fraction=0.5, min_trades=5)`; baseline `FixedSizeStrategy(size=0.10, entry_threshold=0.02, exit_threshold=-0.01)` | `f* = W - (1-W)/R` (W = win rate, R = avg win / avg loss); `_kelly_size` reads only realized P&L available at decision time, returns `base_size` until both wins and losses exist and `min_trades` reached; next-bar execution. QQQ 2018-01 to 2019-12 (500 bars) |
| Staged pairs | `PairsTradingStrategy(asset_a="XLF", asset_b="KRE", lookback=20, entry_zscore=2.0, exit_zscore=0.5, position_size=0.10)` | Rolling z-score of ratio B/A through current close; z > entry: long A lead leg, then short B hedge sized from the realized lead fill; z < -entry: mirror; exit inside the band or unwind the lead if the hedge cannot complete. Integer shares force the hedge to react to realized fills, which is why the hedge leg is sized from the realized lead-leg fill rather than from the target. Helpers `_reset_pair_state`, `_start_pair`, `_submit_hedge`, `_compute_pair_zscore`, `_handle_pending_pair` (True if it consumed the bar), `_update_pair_position` |
| Drawdown circuit breaker | `DrawdownCircuitBreakerStrategy(signal_column="signal", entry_threshold=0.01, exit_threshold=-0.01, base_size=0.10, caution_threshold=0.05, halt_threshold=0.10, reduction_factor=0.5, recovery_rate=0.01)` | Drawdown as positive loss fraction from peak: Normal (< caution) full sizing, multiplier recovers toward 1; Caution (between) sizing interpolated linearly to zero; Halt (>= halt) no new entries. Applies to new entries only; exits follow signals. SPY 2019-08 to 2021-07 with alternating signal |

Report the realized range of signal-time target fractions to prove the feedback loop fired.

### Array vs engine parity (`16_strategy_simulation/06_framework_parity`)

1. Array path: `port = (weights.shift(1).fillna(0) * returns).sum(axis=1) - cost_drag` (the weight in force over a day is the one the previous close produced; `.fillna(0)` leaves the account empty on the opening session); `cost_drag` from `weights.diff()` with the same one-session lag; cost rate = `commission_rate + slippage_rate` charged in one piece.
2. Engine path: `WeightRebalanceStrategy(assets)` reads target weights from the context frame and calls `rebalance_to_weights` every session; `ExecutionMode.NEXT_BAR`; commission as % of notional plus slippage rate on fill price.
3. Align both return streams on common dates before any statistic; run both through one `PortfolioAnalysis`.
4. Report relative total-return gap and the equity-difference path, then list the remaining unmatched assumptions: acted-on timing, fill price, cost basis, share granularity, cash.
5. Counts: `num_fills` counts fills; `trades` appends only on close/flip/scale-down plus open positions marked at the last session.

### Engine divergence anatomy: one field at a time (`16_strategy_simulation/07_engine_divergence_anatomy`)

1. Input: ETF linear-model predictions from `case_studies/etfs/06_linear` validation window (read, never refit). Fix the universe to assets with price history on the first signal date.
2. Quintile selection with disclosed tie policy: all-identical cross-section -> hold the whole tradable universe equal-weight (count these dates); tie straddling the cutoff -> secondary sort by `symbol` ascending. `select_cross_section`, `form_tie_aware_weights(signals, tradable_assets, min_assets)`, `build_tradable_weights(prices, signals, seed_weights, min_assets)` pushes the ~1e-16 residue onto one held asset so each row sums to exactly 1 (asserted).
3. `make_reference()`: next-bar open fill, fractional shares, exits first, snapshot rebalance targets, percentage commission, slippage OFF; every field set explicitly even when equal to the default. `run_with_config(config, target_weights) -> BacktestResult`; `metrics_row(r)`.
4. Experiments, one lever per run: share type (`FRACTIONAL` vs `INTEGER`: rounding <= 1 share, shrinking with account size, plus "disappearance" of sub-half-share orders); fill ordering (`FillOrdering.EXIT_FIRST` vs `FIFO`, both share types); cash buffer (scale every target row by a factor < 1; check the ratio on every active weight); bundled presets loaded with `BacktestConfig.from_preset("backtrader")` and `from_preset("vectorbt")` against `from_preset("default")`, each row labelled by `config.preset_name` (`backtrader`: next open, whole shares, submission order, buffer; `vectorbt`: same-bar close, fractional, exits first, no buffer; real VectorBT as a fourth row); rebalance mode (`RebalanceMode.SNAPSHOT`/`INCREMENTAL`/`HYBRID` via `run_rebalance_mode(mode)` under a whole-share, submission-order, same-bar-close, buffered profile).
5. Decomposition: sum of the three one-factor effects vs the bundled-profile gap; residual = interaction, reported as a number.

### Performance reporting stack (`16_strategy_simulation/09_performance_reporting`, §16.5)

Settings: `START_DATE="2020-01-01"`, `END_DATE="2024-01-01"`, `INITIAL_CASH=100_000`, `FEES=0.001`, `SLIPPAGE=0.0005`, `RSI_PERIOD=14`, `RSI_LOWER=30`, `RSI_UPPER=70`, `POSITION_SIZE=0.95`, `ROLLING_WINDOW_DAYS=365`, `N_BARS=0` (0 = full range; a positive value keeps the latest N bars for a reduced-scale run). Entry/exit levels are conventional defaults, not fitted.

1. Run gross (commission and slippage off), net, and benchmark with identical signals and timing through `ml4t.backtest.Engine`.
2. `daily_return_frame(result, column_name)` -> unique sorted calendar-day frame; join the three on date with a one-to-one expectation; assert lengths and dates; only then convert to arrays for alpha/beta/IR.
3. Annualize return/risk with 365 obs/year; annualize activity rates over the elapsed interval between first and last keyed observation, not row count / 365.
4. Exposure: gross = sum |positions|, net = signed sum; assert equal for long-only.
5. Round trips from `status="closed"` trades only; assert the count equals the engine's record; `TradeAnalyzer`, `MAEMFEAnalyzer`.
6. Cost reconciliation: recover reference-price P&L by removing entry/exit slippage from slipped fill prices; `net = reference_pnl - slippage_cost - commission`; `assert np.isclose(net, final_value - initial_cash)`.
7. One-way turnover uses every fill including the entry of an open end-of-sample position; recover the execution base price from the slipped fill price and side, not the fill's quote-context `reference_price`. `break_even = reference_gross_pnl / one_way_reference_notional` (NaN if gross <= 0). Aggregate notional and costs before annualizing. The rate bounds total round-trip cost on a fixed trade path: an upper bound on tolerance, not a forecast.
8. Four views of the net series: drawdown underwater, rolling Sharpe (365-day window), monthly-returns heatmap, daily-return distribution. Keep reference-line levels, drop adjective labels (`ml4t.diagnostic.visualization.add_annotation` for the level text).

### Regime slicing with point-in-time states (`16_strategy_simulation/10_regime_backtest_analysis`, §16.6)

- Settings: `MOMENTUM_LOOKBACK=126`, `VOL_LOOKBACK=60`, `TREND_LOOKBACK=126` (equal to the strategy lookback so "trending" spans the history the strategy ranks on), `TOP_N=3`, `YIELD_CURVE_THRESHOLD=0.005`, `FEE_RATE=0.0005`, `PERIODS_PER_YEAR=252`, `MAX_SLOPE_AGE_DAYS=4`; SPY as the regime index.
- States: sigma_t = trailing 60d realized vol, rho_t = trailing 126d total return of SPY; `v_t = 1[sigma_t >= median(sigma_s : s < t)]`, `u_t = 1[rho_t > median(rho_s : s < t)]` via `expanding_past_median` (finite observations strictly before each row); shift one bar before joining to returns. `combine_state`: (Low vol, Up) = Risk-on; (High, Up) = Caution; (High, Down) = Crisis; (Low, Down) = Recovery; "Warmup" before valid.
- `summarize_returns` per state: CAGR of the compounded subsequence, vol, Sharpe = mean/std x sqrt(252), max drawdown of the synthetic path, plus `days` and `share`; last row = pooled active days. Read every state against the pooled row and its `days`.
- Tail: VaR = loss exceeded by the worst 1 day in 20 (5%); CVaR = mean of those days; same estimator on pooled and Crisis, histograms on shared bins.
- Drawdown attribution: for the worst calendar peak-to-trough episode, sum `log(1 + r_t)` by state; assert the parts sum to the episode's log growth ratio.

### Single-strategy Sharpe inference (`16_strategy_simulation/11_sharpe_ratio_inference`, §16.7)

Settings: `N_SIMULATIONS=1000`, SPY `SAMPLE_START="2020"`, `SAMPLE_END="2023"`, `DEMO_TRUE_SR=0.5`, `DEMO_SAMPLE_DAYS=252`, `DEMO_ANNUAL_VOL=0.15` (cancels; scale only), `SEARCH_STRATEGIES=30`, `SEARCH_REAL=5`, `SEARCH_DAYS=504`, `SEARCH_TRUE_SR=0.8`, `SEED=42`.

The notebook imports only `compute_min_trl` and `deflated_sharpe_ratio` (as `lib_dsr`) from `ml4t.diagnostic.evaluation.stats`. Every function marked "NB11 local" below is defined inside `11_sharpe_ratio_inference.py` and cannot be imported from `ml4t`.

| Quantity | Formula | Implementation |
|---|---|---|
| Sampling distribution (iid normal) | `SR_hat ~ N(SR, sqrt((1 + SR^2/2)/T))` | Monte Carlo (NB11 local) |
| Mertens (2002) variance | `Var(SR_hat) = (1/T)[1 - g3*SR_hat + ((g4-1)/4)*SR_hat^2]` (T-1 in PSR derivation) | NB11 local: `sharpe_ratio_variance(sr, n, skewness=0.0, kurtosis=3.0, periods_per_year=252)`, `sharpe_ratio_se(...)` |
| Lo (2002) Eq. 17 annualization | `SR(q) = SR(1) * q / sqrt(q + 2*sum_{k=1}^{q-1}(q-k)*rho_k)`; reduces to `sqrt(q)*SR(1)` when all rho_k = 0; positive autocorrelation lowers the annualized Sharpe, negative raises it | NB11 local: `lo_2002_annualized_sharpe(returns, periods_per_year=252, max_lag=None)` (Newey-West lag cutoff by default; assumes rho_k = 0 beyond it) |
| PSR | `Phi((SR_hat - SR*) / se(SR_hat))` | NB11 local: `probabilistic_sharpe_ratio(observed_sr, benchmark_sr, n, skewness=0.0, kurtosis=3.0)` -> one-sided inference with a CI. The library's `ml4t.diagnostic.evaluation.stats.sharpe_inference.probabilistic_sharpe_ratio(observed_sharpe, benchmark_sharpe=0.0, n_samples=1, skewness=0.0, kurtosis=3.0, return_components=False)` has a different signature |
| MinTRL (canonical, SR_p = SR_ann/sqrt(q)) | `T_min = 1 + z_{1-a}^2 [1 - g3*SR_p + ((g4-1)/4)*SR_p^2] / (SR_p - SR_p*)^2` | NB11 local: `minimum_track_record_length(target_sr, benchmark_sr=0, alpha=0.05, power=0.80, skewness=0.0, kurtosis=3.0, periods_per_year=252)` returns both T_min and T_plan. There is no library function of that name; the library equivalent is `compute_min_trl(returns=None, observed_sharpe=None, target_sharpe=0.0, confidence_level=0.95, frequency="daily", ...)` |
| Power planning (SR_p* = 0, normal) | `T_plan ~ (z_{1-a} + z_{1-b})^2 (1 + SR_p^2/2) / SR_p^2`; use before starting | same NB11 local function, second return |

- NB11 local: `complete_sharpe_inference(returns, benchmark_sr=0, alpha=0.05)` combines Mertens-corrected PSR (treats returns as independent), MinTRL, and Lo annualization reported as a diagnostic that does not feed the significance test; `sharpe_ratio_checklist(returns, target_sr, alpha=0.05)` prints the full checklist.
- Library: `compute_min_trl`; `deflated_sharpe_ratio(returns_1d, frequency="daily")` -> PSR-style `DSRResult` with `.sharpe_ratio_annualized`, `.probability`, `.is_significant` (95%), `.min_trl_years`, `.has_adequate_sample`; on a 2-D array of candidate returns the result additionally carries `.expected_max_sharpe` and `.deflated_sharpe`. Combined non-normal + AR(1) variance in `ml4t.diagnostic.evaluation.stats.sharpe_inference.compute_sharpe_variance` (López de Prado 2025 closed form).

### Deflated Sharpe Ratio and PBO (`16_strategy_simulation/12_dsr_validation`, §16.7)

Settings: `N_SIMULATIONS=10000`, `NULL_STRATEGIES=100`, `NULL_PERIODS=252`, `SELECTED_SHARPE=1.5`, `VARIANT_COUNT=30`, `VARIANT_TRUE_SHARPE=0.5`, `VARIANT_PERIODS=504`, `NONNORMAL_TRIALS=50`, `EULER_MASCHERONI=0.5772156649`, `PBO_OBSERVATIONS=4800`, `N_BLOCKS=10`, `N_STRATEGIES=20`, `PBO_TRUE_SR_SIGNAL=1.5`, `DSR_PASS_THRESHOLD=0.95`, `PBO_REJECT_THRESHOLD=0.25`, `SEED=42`.

1. Expected maximum of N null trials (native frequency), `_expected_max_sharpe(variance_trials, n_trials)`: 0 if `n_trials <= 1` or variance <= 0; else `sqrt(V) * [(1-g)*Phi^-1(1 - 1/N) + g*Phi^-1(1 - e^-1/N)]`, g = Euler-Mascheroni. `sqrt(2 log N) * sigma_SR` is a bound above this expectation.
2. `DSR = Phi((SR_hat - SR0*) * sqrt(T-1) / sqrt(1 - g3*SR_hat + ((g4-1)/4)*SR_hat^2))` with native-frequency Sharpes, SR0* = expected max. Local `deflated_sharpe_ratio(observed_sharpe, skewness=0.0, kurtosis=3.0, n_samples=252, n_trials=1, variance_trials=0.0, confidence_level=0.95, return_format="probability", return_components=False, periods_per_year=252)` accepts annualized inputs; outputs probability / z-score / deflated Sharpe.
3. Library `deflated_sharpe_ratio_from_statistics(observed_sharpe=SR_ann/sqrt(252), n_samples, n_trials, variance_trials=V_ann/252, skewness, excess_kurtosis=k-3).probability`: rescales cross-trial variance with a finite-sample multiple-trial convention, evaluates the non-normal variance at the adjusted benchmark (the local helper evaluates it at the observed Sharpe), supports autocorrelation, returns a MinTRL diagnostic. Agrees with the bare formula in behaviour, not to the last digit. Real-data DSR variant sweeps live in the case studies; everything here is a parametric draw.
4. Required-Sharpe table: solve for the observed Sharpe that clears the deflated threshold at each trial count (sample length, dispersion, skew, kurtosis fixed). Read it before the search.
5. PBO via CSCV: partition into an even number of blocks (10); every choice of half the blocks is IS, its complement OOS (C(10,5) = 252 splits). `cscv_sharpe_matrices(strategy_returns, n_blocks)` builds complementary IS/OOS Sharpe rows (via `_sharpe_from_moments`); `compute_pbo(is_performance=..., oos_performance=...)` -> frozen `PBOResult` with `.pbo`, `.pbo_pct`, `.n_combinations`, `.n_strategies`, `.is_best_rank_oos_median`, `.is_best_rank_oos_mean`, `.degradation_mean`, `.degradation_std` (notebook uses `.to_dict()` when available). Each row must be one complementary split from the same series. Scenarios: all-noise, mixed (strategies 0, 7, 14 have stable edge), block-count sweep on the same panel (changes the number of splits with fixed IS/OOS sizes), `simulate_regime_sensitive_returns(n_observations, n_strategies, n_blocks, seed, block_sharpe_scale=3.0, daily_vol=0.01)`.

### Rademacher Anti-Serum (`16_strategy_simulation/13_ras_protocol`, §16.7)

Settings: `N_SIMULATIONS=5000` (sign draws), `CLASS_PERIODS=252`, `CLASS_CANDIDATES=100`, `SWEEP_PERIODS=504`, `SWEEP_CANDIDATES=500`, `IC_PERIODS=252`, `IC_SIGNALS=50`, `IC_TRUE_SIGNALS=5`, `IC_TRUE_VALUE=0.03`, `IC_KAPPA=0.1`, `IC_HURDLE=0.01`, `GRID_HURDLE=0.5`, `HOURLY_DAYS=90`, `CHECK_SHARPE=1.0`, `CHECK_SAMPLES=504`, `CHECK_TRIALS=100`, `CHECK_VARIANCE=0.3`, `CONFIDENCE=0.05` (delta), `SEED=42`.

- Rademacher complexity `R_hat = E_eps[max_n eps^T x^n / T]`, eps iid +/-1. `normalize_column_norms` scales each candidate path to Euclidean norm sqrt(T); `massart_upper_bound(performance)` is the finite-class bound in native units (depends on the largest column norm, so keep the scale if the matrix is not normalized). Each class's R_hat is reported as a share of its Massart bound.
- RAS bound (Paleologo): `theta_n >= theta_hat_n - 2*R_hat - 3*sqrt(2 log(2/delta)/T) - sqrt(2 log(2N/delta)/T)`; all terms per-period, volatility-standardized.
- Order of operations: standardize by std (ddof=1, no demeaning) -> native mean -> `rademacher_complexity(standardized, n_simulations=5000, random_state=seed)` (library signature `rademacher_complexity(X, n_simulations=10000, random_state=None)`; there is no `seed` kwarg) -> `ras_sharpe_adjustment(observed_sharpe, complexity, n_samples, n_strategies, delta)` -> multiply by `sqrt(periods_per_year)` -> pass iff annualized lower bound > hurdle.
- IC variant: `ras_ic_adjustment(observed_ic, complexity, n_samples, delta, kappa)` with `kappa=IC_KAPPA=0.1` passed explicitly (the library default is `kappa=0.02`; it must exceed max |IC|), hurdle 0.01 on 50 signals of which 5 are real at IC 0.03.
- Grid class: 7 lookbacks x 5 holding periods x 4 position counts = 140 correlated candidates on 504 periods; the trial count is the product of the axes, including axes swept once and discarded. Hourly class: 90 days, annualizer sqrt(8760).

### Cost sensitivity sweep (`16_strategy_simulation/14_cost_sensitivity`, §16.6)

1. Build weights once (fee-independent fixed-schedule rank strategy); rerun `_etf_baseline` at every per-leg fee in `COST_GRID_BP = [0, 1, 2, 5, 10, 15, 25, 40, 60, 100, 150, 200]` (bps of traded notional; dense near realistic fees). The row at the baseline fee must reproduce NB01 to the cent.
2. Break-even two ways: (a) linear = gross growth rate / annual one-way turnover (ignores compounding; too high); (b) interpolate the simulated growth-rate curve between the last positive and first non-positive cost (lower). Both are ceilings on one sample; quote (b).
3. Annual drag ~ annual turnover x fee per leg; check against the simulator's actual charge (estimate slightly low).
4. Waterfall from starting capital (not first close): gross-to-net gap in dollars = fees paid + the return those fees would have earned (path effect); assert nothing unaccounted.
5. `plot_cost_sensitivity` (`ml4t.diagnostic.visualization.backtest.cost_attribution`) deducts a uniform daily drag from gross returns: same shape, different zero crossing; use for report bundles, not for quoting a break-even.

### Break-even cost: one definition (NB09, NB14; ch20 and `decision_rules.md` point here)

The chapter and the companion repo use three estimators; a report must say which one it quotes.

| Estimator | Where | What crosses what |
|---|---|---|
| Fixed-path rate | NB09: `reference_gross_pnl / one_way_reference_notional` | per-leg cost at which the fixed trade path earns zero gross-minus-cost P&L; NaN if gross <= 0 |
| Growth-rate crossing | NB14 sweep: interpolate the simulated growth-rate (CAGR) curve between the last positive and first non-positive fee | per-leg bps at which net CAGR crosses the benchmark; NB14's benchmark is zero growth (rf = 0 convention), so state the hurdle (rf or the declared baseline) whenever another is used |
| Sharpe crossing | ch20 NB06 `_breakeven(curve)` | per-leg bps at which the sweep's net Sharpe (rf = 0) crosses zero; censored `(grid[-1], False)` when still positive at the ceiling |

Reporting rules: (1) report both the CAGR-vs-benchmark crossing and the net-Sharpe-zero crossing, label which one the headline quotes, and never label a "net CAGR = rf" solution as "net Sharpe ~ 0"; (2) state per-leg vs round-trip (the grid is per leg; round-trip = 2x) and whether impact at the reference AUM is inside the sweep or added to the denominator afterwards; (3) the printed sweep must bracket the quoted value with a sign change (`v_lo > 0 >= v_hi`); a crossing above the grid is censored and reported as "> ceiling", never as the ceiling.

Cost margin, defined once: `cost_margin_ratio = breakeven_bps / all_in_assumed_bps`, with `all_in_assumed_bps` = per-leg commission + half-spread + impact at the reference AUM (ch18 taxonomy; break-even bounds the whole allowance, so headroom read against commission alone is overstated). The companion repo's `compute_cost_bps(setup)` (`case_studies/utils/strategy_analysis.py`: mean of `setup.yaml` `costs.per_leg_cost_bps_range`, else the fee-schedule average, else a flagged 10 bps fallback; no impact term) is the narrower repository convention behind ch20's resilience bands; name which denominator a ratio used.

### Cross-framework parity audit (`16_strategy_simulation/15_lean_engine_parity`, `16_case_study_lean_parity`, `17_backtrader_zipline_engine_parity`, `18_vectorbt_engine_parity`, §16.3)

- Protocol: model fitting and target construction happen before any engine; each engine in a required pair gets the same content-addressed data and frozen targets (bundle hash covers data, targets, strategy spec, contract/funding inputs); transaction costs and position rules disabled on both sides. Notebooks read the committed `16_strategy_simulation/resources/framework_parity_audit.json`; regenerating needs LEAN, Backtrader, Zipline Reloaded and both VectorBT editions (outside the `ml4t` image).
- Pass requires all four: complete sorted fill stream matches on timestamp, asset, side, quantity, price, commission (price 8 decimals, quantity 5); identical valuation timestamp set; every account value and terminal value rounds to the same cent; a negative control (first fill price changed by one unit at fill-record precision) is detected. "Exact" is not bit-identical floats; the raw equity and terminal gaps are retained in the audit resource.
- Support matrix: LEAN - ETF, crypto perps (native crypto-future securities), USD-quoted FX, US equity panel; CME unsupported (bundle has continuous root prices, no dated contract chain or roll map). Backtrader - ETF, US equity panel, CME, USD FX. Zipline - ETF, US equity panel only. VectorBT Pro - ETF, FX, equity panel, CME (it supplies multipliers and futures-style leverage). VectorBT OSS - ETF, FX, equity panel (no native multiplier/margin model, hence no CME). Only LEAN is credited with crypto-perp funding. Unsupported rows are excluded from the pass denominator; they are neither failures nor passes.
- Timing: only for correctness-passing pairs; 1 warmup + 10 process-isolated runs; timed region = the engine call only, excluding loading, inference, target construction, adapter prep, extraction, serialization and reporting. Ratio > 1 means ML4T faster.

## Guardrails and pitfalls

Look-ahead, data and universe
- **Same-bar look-ahead** — a close-derived signal filled at that close earns a return that began before the signal existed / shift the whole weight matrix (NB01) or the entry/exit conditions (NB03) one row; use `ExecutionMode.NEXT_BAR`; treat `SAME_BAR` and the `vectorbt` preset curves as "what same-bar filling is worth", not performance.
- **Full-sample statistics in a signal** — leaks future scale and levels / trailing-window score with closes through t only; cross-sectional rank per date; assert no full-sample stat enters.
- **Macro calendar misalignment** — yields publish on the Fed schedule, ETFs on sessions; a forward match uses unpublished data / backward as-of join; check matched-observation age <= `MAX_SLOPE_AGE_DAYS=4`, not row count. The as-of join hides publication gaps by carrying stale values, which is why the age check exists.
- **Vintage/revision bias** — the local FRED file is a current snapshot, not ALFRED vintages / print coverage end and file hash; disclose; prefer vintage data. Treasury revisions are small, so the effect is minor for the ETF baseline (NB01, NB06).
- **Survivorship/selection in the universe** — ten hand-picked ETFs and CME products that exist today / disclose "describes these surviving funds"; check common first bar and identical bar counts before ranking; never claim a survivorship-free estimate.
- **No holdout** — every reported date was used to build or choose the rule / label output "in-sample teaching output"; fix the configuration before the run; never adjust to improve output.
- **Term-sheet drift** — criteria adjusted to the output are meaningless / write floor, drawdown limit and benchmark condition first; score pass/fail in a table.
- **Malformed bars** — bad OHLC corrupts indicators and the broker / assert unique keys and positive, consistent OHLC after aggregation.
- **Calendar vs session boundaries** — CME sessions cross midnight (17:00 CT); UTC days mislabel P&L / `SessionConfig`, `compute_session_pnl`; 4 PM CT session bars.

Accounting, costs and futures mechanics
- **Negative cash financing fees** — simultaneous full-notional rebalance quietly borrows to pay fees / sell first, deduct fees, scale buys to remaining cash; or explicit cash buffer; or `reject_on_insufficient_cash=True`.
- **Double-charging slippage** — fill prices already contain slippage / recover reference-price P&L, subtract slippage and commission once; assert net P&L == change in account value.
- **Quote-context `reference_price` for notional** — may be a different price surface / recover the execution base price from the slipped fill price and side.
- **Open positions in round-trip stats** — unrealized P&L contaminates win rate and payoff / closed trades only (`status="closed"`), count asserted; keep the open position in equity.
- **Fill vs trade counts conflated** — `trades` excludes opens and scale-ups / `num_fills` for activity, `trades` for closed round trips.
- **Missing futures multiplier** — P&L, margin basis and valuation wrong by the point value / `ContractSpec` via `contract_specs=`; verify point value == tick value / tick size; replay with multiplier 1 as a counterfactual.
- **Wrong cost units for futures** — % of notional cannot reproduce a constant $/contract across products / `CommissionType.PER_CONTRACT`; print $2 as % of notional per product.
- **Continuous-contract approximation** — ratio-adjusted series exclude roll orders and costs / disclose; raw front close for contract counts; see the CME case study for rolls.
- **Static contract/margin specs across history** — margins anchored to 2025-12-31 applied to 2018-2023 / disclose; do not treat admissibility as historical.
- **Flat slippage, no impact** — optimistic beyond a small account / treat cost figures as lower bounds; Ch18 size/volume model.
- **Break-even read against commission alone** — it bounds total round-trip cost (spread and impact come out of the same allowance) and holds the trade path fixed / compare against commission + spread + expected impact; quote from the re-simulated sweep.
- **Linear break-even and turnover x fee** — ignore forgone compounding; break-even too high, drag too low / quote the simulated crossing; use the shortcut only to reason.
- **Weights fixed across a cost sweep** — a real implementation trades less as fees rise / disclose: steeper than a well-managed strategy, shallower than a naive one.
- **Average turnover hides fragility** — heavy trading in the years that produced the return / disclose.
- **Cost waterfall measured from first close** — first close contains a session's return and the opening commission / measure from starting capital.
- **Library cost curve for break-even** — uniform daily drag has a different zero crossing / use the sweep for the number, the library for report bundles.

Engine configuration and parity
- **Benchmark on a different protocol** — fills+fees vs frictionless return series / run 60/40 or buy-and-hold through the same simulator, dates, prices, fees, slippage, capital fraction; suppress VectorBT's implicit benchmark.
- **Benchmark that trades during warmup** — on a trending sample the warmup window dominates / benchmark enters on the first bar the strategy could trade.
- **Metric convention mismatch** — hand-rolled Sharpe off by the annualization factor or risk-free convention / assert agreement with `PortfolioAnalysis` to display precision; compare return series through one metric implementation, never library reports.
- **Misaligned return arrays in alpha/beta/IR** — one series a day shorter shifts every benchmark-relative number, nothing raises / one-to-one date join with assertions before converting to arrays.
- **Annualizing activity by row count / 365** — off-by-one year estimates / use the elapsed interval between first and last keyed observation.
- **Adjective labels on reference lines** — an axis label rules on a strategy not yet measured / keep the levels, drop the words (NB09).
- **Correlation as parity test** — correlation ~1 hides a constant daily gap that compounds / compare terminal wealth and the equity difference over time.
- **Unreachable configuration fields** — `FillOrdering` measured exactly zero because `TargetWeightExecutor` sorts sells before buys upstream of the broker; a zero bar reads as "tested and small" / change the field, confirm something moved, then attribute; the field is live for strategies that submit their own orders.
- **Inherited presets** — library defaults change; experiments drift silently / set every execution field explicitly, including those equal to defaults.
- **Whole-share disappearance** — orders < half a share never submit; positions stop tracking targets; rounding residual is one-sided / watch trade count as well as value; test at realistic starting capital.
- **Integer-unit truncation on high-priced assets** — material exposure distortion for BTC / `ShareType.FRACTIONAL`.
- **Target weight of exactly 1.0 with next-open fills** — a gap up at an unknown open cannot be absorbed / position size 0.95.
- **Cash-buffer drag understated** — a permanently uninvested sliver forgoes that fraction of every subsequent return / charge it to the engine requiring the buffer; measure over the full path.
- **Additive decomposition** — levers that change holdings change what others trade / measure the bundle and the parts; report the residual as interaction.
- **Rank ties, permutation dependence, float residue** — sorting an all-equal cross-section returns file order; rows summing to 1 - 1e-16 under-hold at bp level / equal-weight the whole universe on all-equal dates and count them; secondary sort by symbol; push residue onto one held asset and assert exact row sums.
- **Parameter-sweep selection** — the highest in-sample Sharpe cell is not a validated choice / read the surface for flatness; no holdout means no deployment claim; DSR/RAS.
- **Feedback loops that never fire** — adaptive code that never updates looks adaptive / report the realized range of signal-time target fractions.
- **Circuit-breaker permanent halt** — once in cash beyond the halt threshold, equity cannot recover / production needs an explicit reset, external capital or re-entry protocol; the backtest form (`RiskManager.reset_halt()` under a rule written before the run) is in `chapters/19_risk_management.md`, `ml4t.backtest.risk` recipe.
- **Stationarity assumption in pairs** — high return correlation does not make the price ratio stationary / inspect ratio and z-score panels; count state-machine transitions and unwinds.
- **MAE/MFE as stop rules** — read from the trades that produced them / diagnostics only.
- **Parity read as universal equivalence** — the audit tests target replay with costs and position rules disabled under pinned versions / do not apply a pass to production overlays; never approximate an unsupported asset model to add a row; keep synthetic stress rows separate from real-data evidence; do not generalize engine timings.

Signal conversion and regimes
- **Fixed threshold on an uncalibrated score** — becomes always-on or never-on when the score scale drifts / detect activation rate near 0 or 1 or wildly different across models; use percentile rules.
- **Trailing percentile with too little history** — undefined or constant rule; too-short windows flip constantly, too-long ones lag / assert every symbol has >= longest lookback obs; lookbacks 21-63; re-measure per prediction set.
- **Cross-sectional rule on a thin universe** — nothing to rank / assert >= 2 symbols per date; trailing rule for single-asset strategies.
- **Transition rate read as turnover** — overstates concentrated and understates diffuse strategies / compute turnover from fills after sizing.
- **Carrying a signal setting between strategies** — the same lookback/percentile gave very different activation on ETF vs crypto scores / re-run diagnostics on the new score stream.
- **Selecting the artifact by the statistic compared** — measures the favour / rule-based selection (widest coverage, hash tie-break); recompute rank IC and n_days against registry metadata to catch artifact drift; read the label from `setup.yaml`.
- **State label built from the day's own return** — guarantees a strong-looking, meaningless result / closes through t-1, expanding median of own past only, shift one bar; check all three.
- **Fixed volatility thresholds across a decade** — measures the decade, not the state / expanding past median.
- **Thin regime slices** — spread grows as slices thin; no valid naive CI under overlapping windows / read `days`/`share` first; distrust the thinnest state.
- **Slice drawdown read as lived experience** — a path no account followed / compare states only; calendar episodes for investor experience.

Inference and overfitting
- **Sharpe quoted without sample length** — SE ~1 on one year of daily data / report PSR/CI; compute MinTRL.
- **Normal-returns assumption on fat tails** — understates uncertainty in the flattering direction / feed sample skew and kurtosis; note non-normality raises DSR when SR_hat < benchmark and lowers it above.
- **Trusting Lo's correction on short samples** — each rho_k has SE ~1/sqrt(T) multiplied by weights ~q; yearly multipliers range ~5 units, lag-1 sign flips / compute per year; report as a diagnostic, never the significance threshold.
- **Asymptotic variance on short records** — intervals are optimistic / disclose; moment estimates on short records carry wide error that propagates into DSR.
- **Single-strategy stats on a searched strategy** — the best null of 30 shows a respectable Sharpe and high PSR / record every variant; apply DSR/RAS.
- **Under-counting trials** — abandoned variants are forgotten; grid count = product of axes (7x5x4 = 140) / log every trial.
- **Raw trial count with correlated candidates** — over-deflates / estimate effective trials before DSR, or use RAS.
- **Trial variance treated as known** — measured variance across 100 trials swings by a third between draws / disclose.
- **PBO with non-exchangeable blocks** — a decayed edge is flagged unstable / interpret alongside a stationarity check.
- **Fake CSCV matrices** — independent IS/OOS matrices do not implement PBO / each row = one complementary split from the same series.
- **Annualizing before RAS** — mixes units; hourly series under-charged by roughly a factor of six / per-period standardized units, annualize last.
- **Demeaning before RAS** — the bound requires volatility-standardized, non-demeaned returns / divide by std (ddof=1) only.
- **Expecting sample size to erase the complexity charge** — more observations shrink estimation-error terms, not 2*R_hat / disclose.
- **DSR probability read as a performance floor** — DSR is a tail probability, RAS a lower bound / read both.
- **Synthetic iid paths in NB12/13** — real returns have dependence and regimes; costs and portfolio construction omitted / dependence-aware resampling; disclose.

## Decision rules and defaults

| Decision | Rule |
|---|---|
| Array vs engine | Parameter grid or "any signal at all?" -> array; "what would it have earned?", unfilled orders, stops, state-dependent sizing -> sequential engine; "do two implementations agree?" -> either, once assumptions are written down |
| ExecutionMode | `NEXT_BAR` for close-derived daily signals; `SAME_BAR` only with a defensible intrabar timing argument; no universal mode |
| Attention order for engine settings | Fill timing and sizing/share type first; cash buffer second; rebalance mode last (worth far less); fill ordering only when the strategy submits its own orders |
| Reference configuration | Every field explicit; slippage off when isolating commission effects; one lever per run; measure bundle and residual |
| Starting capital | Choose realistically; it decides whether integer sizing is a detail or a constraint (whole-share rounding is fixed dollars) |
| Signal rule | Calibrated fixed-scale scores -> fixed threshold; drifting scale, single asset -> trailing percentile (lookback 21-63, percentiles 75-95, operating point 90 at window 63); broad universe -> cross-sectional. Compare methods only at one shared operating percentile |
| DSR / PBO / RAS | Many variants or parameter optimization -> DSR; complex ML model or feature selection -> RAS; both -> both, report the more conservative. Known trial count and dispersion -> DSR; complementary CSCV performance -> PBO; candidate return/prediction matrix -> RAS; correlated sweep -> RAS or effective trials first; quick diagnostic -> DSR with raw-count caveat; reported research -> publish diagnostic, inputs, assumptions |
| Pre-registered thresholds | DSR pass >= 0.95; PBO reject >= 0.25; RAS pass iff annualized lower bound > hurdle (0.5 grid; IC hurdle 0.01); delta = 0.05. Set before computing |
| Sharpe inference sequencing | Before: define target SR, plan power (alpha 0.05, power 0.80), count variants. After: observed SR with PSR/CI; adjust skew/kurtosis; DSR if more than one strategy; compare with canonical MinTRL. A target needing a decade of data is a reason to seek a stronger effect |
| Overfitting checklist | Count every variant incl. abandoned; measure Sharpe variance across variants; measure selected skew/kurtosis; compute DSR; compute PBO from complementary CSCV (even block count, e.g. 10); compare the three and investigate disagreements; apply thresholds written beforehand |
| Cost sweep | Grid 0-200 bps per leg; quote break-even from the simulated crossing; carry "annual drag ~ turnover x fee"; read the slope at the baseline fee; headroom vs commission + spread + impact; read with benchmark and regime split before concluding economic value; break-even and `cost_margin_ratio` as defined under "Break-even cost: one definition" (both crossings reported, per-leg, impact stated, sign change bracketed) |
| Parity pass | All four criteria (fills to 8/5 decimals incl. commission; identical valuation timestamps; cents on every account and terminal value; negative control detected); timing only for passing pairs |

Numeric defaults by notebook

| Setting | Value |
|---|---|
| ETF baseline (NB01/10/14) | L=126 sessions, top 3 of 10, regime threshold 0.005 on 10Y-2Y, fee 5 bps per dollar traded, $100k, monthly next-open rebalance, defensive 60/40 AGG/TLT, benchmark 60/40 SPY/AGG, as-of tolerance 4 days; rebalance count = months + 1; universe check: common first bar, identical bar counts |
| Term sheet (NB01) | Risk-adjusted return floor; `abs(MaxDD%) < 25`; beat 60/40 total return. NB01 misses two of three (which two not stated in the notes) |
| Futures (NB02) | 63-session momentum; `long_n=2`, `short_n=2`, `rebalance_every=21`; $2/contract/side; 5 bps slippage; `floor(alloc/(price x mult))`, skip if one contract > allocation |
| RSI (NB03/04/09) | window 14, enter < 30, exit > 70, 95% capital, 10 bps commission + 5 bps slippage, 365-day annualization, conditions shifted one row, 365-day rolling Sharpe window |
| Kelly (NB05) | half-Kelly, base 10%, bounds [2%, 25%], min 5 trades |
| Pairs (NB05) | lookback 20, entry abs(z)=2.0, exit abs(z)=0.5, size 10% |
| Breaker (NB05) | caution 5%, halt 10%, reduction 0.5, recovery 0.01/bar |
| Signal conversion (NB08) | windows [21, 42, 63]; percentiles [75..95]; operating percentile 90 at window 63 |
| Regimes (NB10) | vol lookback 60d, trend lookback 126d, expanding past median, one-bar shift; VaR/CVaR at 5% |
| Inference (NB11-13) | alpha 0.05, power 0.80; DSR threshold 0.95; PBO threshold 0.25; 10 CSCV blocks (252 splits); 5000 Rademacher draws; delta 0.05 |
| Cost grid (NB14) | `[0, 1, 2, 5, 10, 15, 25, 40, 60, 100, 150, 200]` bps per leg |
| Parity reporting (NB06) | Align dates; one `PortfolioAnalysis`; relative total-return gap + equity difference path; list unmatched assumptions |

## Code patterns and APIs

Imports
```python
from ml4t.backtest import BacktestConfig, DataFeed, Engine, ExecutionMode, Strategy   # ExecutionMode lives in ml4t.backtest.types, re-exported here
from ml4t.backtest.types import ContractSpec
from ml4t.backtest.config import CommissionType, SlippageType, ShareType, ExecutionPrice
from ml4t.backtest.profiles import BACKTRADER_PROFILE, VECTORBT_PROFILE, get_profile_config, list_profiles
from ml4t.backtest.analytics import MAEMFEAnalyzer, TradeAnalyzer
from ml4t.backtest.sessions import SessionConfig, compute_session_pnl
from ml4t.backtest.execution.rebalancer import RebalanceConfig, TargetWeightExecutor
from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.metrics import sharpe_ratio, sortino_ratio
from ml4t.diagnostic.integration import portfolio_analysis_from_result
from ml4t.diagnostic.visualization import add_annotation
from ml4t.diagnostic.visualization.portfolio import (plot_cumulative_returns,
    plot_drawdown_underwater, plot_monthly_returns_heatmap, plot_rolling_sharpe)
from ml4t.diagnostic.evaluation.stats import (compute_min_trl, deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_statistics, compute_pbo, rademacher_complexity,
    ras_sharpe_adjustment, ras_ic_adjustment)
from ml4t.diagnostic.evaluation.stats.sharpe_inference import compute_sharpe_variance
from ml4t.diagnostic.visualization.backtest.cost_attribution import plot_cost_sensitivity
```
Repo helpers: `data.load_etfs`, `load_cme_futures()`, `utils.reproducibility.set_global_seeds`, `utils.style` (`COLORS`, `FIGSIZE`, `ml4t_diverging`, `show_plotly_with_alt`, `show_with_alt`, `add_message_title`), `get_output_dir(16, ...)`, `validation.adapters.ml4t_adapter` (NB07), `16_strategy_simulation/_etf_baseline.py` (imported by NB14), `16_strategy_simulation/resources/framework_parity_audit.json` (NB15-18). CME case-study backtests: `case_studies/cme_futures/13_backtest.py` and `18_holdout_backtest.py` (NB02's own cross-reference names a `case_studies/cme_futures/strategy/backtest.py` that does not exist in the checkout).

`BacktestConfig` defaults (`libs/src/ml4t_backtest/ml4t/backtest/config.py`)

| Group | Field = default |
|---|---|
| Account | `allow_short_selling=False`, `allow_leverage=False`, `initial_margin=0.5`, `long_maintenance_margin=0.25`, `short_maintenance_margin=0.30`, `fixed_margin_schedule=None`, `margin_pct_schedule=None`, `short_cash_policy=ShortCashPolicy.CREDIT`, `lock_notional_update_mode=LockNotionalUpdateMode.POSITION_LEGS` |
| Execution | `execution_price=ExecutionPrice.OPEN`, `mark_price=ExecutionPrice.PRICE`, `execution_mode=ExecutionMode.NEXT_BAR`, `fill_ordering=FillOrdering.EXIT_FIRST`, `entry_order_priority=EntryOrderPriority.SUBMISSION`, `partial_fills_allowed=False` |
| Stops | `stop_fill_mode=StopFillMode.STOP_PRICE`, `stop_level_basis=StopLevelBasis.FILL_PRICE`, `trail_hwm_source=WaterMarkSource.CLOSE`, `trail_include_entry_bar_extremes=False`, `initial_hwm_source=InitialHwmSource.FILL_PRICE`, `trail_stop_timing=TrailStopTiming.LAGGED`, `stop_slippage_rate=0.0` |
| Shares | `share_type=ShareType.INTEGER`, `share_rounding=ShareRounding.NEAREST` |
| Commission | `commission_type=CommissionType.NONE`, `commission_rate=0.0`, `commission_per_share=0.0`, `commission_per_trade=0.0`, `commission_minimum=0.0` |
| Slippage | `slippage_type=SlippageType.NONE`, `slippage_rate=0.0`, `slippage_fixed=0.0`, `slippage_spread=0.0`, `slippage_spread_by_asset={}`, `slippage_spread_convention=SpreadConvention.FULL_SPREAD` |
| Cash | `initial_cash=100000.0`, `cash_buffer_pct=0.0`, `settlement_delay=0`, `settlement_reduces_buying_power=True`, `reject_on_insufficient_cash=True`, `next_bar_submission_precheck=False`, `next_bar_simple_cash_check=False`, `buying_power_reservation=False`, `next_bar_queue_shadow_validation=False` |
| Methods | `validate(warn=True)`, `get_effective_account_settings()`, `get_effective_account_type()`; `StatsConfig(recent_window_size=50, track_session_stats=True, enabled=True)` |

Enums: `ExecutionPrice {PRICE, CLOSE, OPEN, VWAP, MID, BID, ASK, QUOTE_MID, QUOTE_SIDE}`; `ExecutionMode {NEXT_BAR, SAME_BAR}`; `ShareType {FRACTIONAL, INTEGER}`; `ShareRounding {NEAREST, TRUNCATE}`; `FillOrdering {EXIT_FIRST, FIFO, SEQUENTIAL, PRIORITY}`; `EntryOrderPriority {SUBMISSION, NOTIONAL_DESC, NOTIONAL_ASC, FREE_CASH_ASC, ORDER_VALUE_ASC}`; `ShortCashPolicy {CREDIT, CREDIT_PROCEEDS, LOCK_NOTIONAL}`; `RebalanceMode {SNAPSHOT, INCREMENTAL, HYBRID}`; `MissingPricePolicy {SKIP, USE_LAST}`; `LateAssetPolicy {ALLOW, REQUIRE_HISTORY}`; `CommissionType {NONE, PERCENTAGE, PER_SHARE, PER_CONTRACT (alias of PER_SHARE), PER_TRADE, TIERED}`; `SlippageType {NONE, PERCENTAGE, FIXED, SPREAD, VOLUME_BASED}`; `DataFrequency {DAILY, ..., HOURLY="1h", IRREGULAR}`; `WaterMarkSource {CLOSE, BAR_EXTREME}`; `InitialHwmSource {FILL_PRICE, SIGNAL_PRICE, BAR_CLOSE, BAR_HIGH}`; `SpreadConvention {FULL_SPREAD, HALF_SPREAD}`; `TrailStopTiming {LAGGED, INTRABAR, VBT_PRO}`.

Presets (NB07): `BacktestConfig.from_preset(preset)` loads a bundled profile by name (`"default"`, `"backtrader"`, `"vectorbt"`, `"zipline"`, `"lean"`, `"realistic"`, `"ibkr_us_stocks_fixed"`) via `ml4t.backtest.profiles.get_profile_config(name)` and `BacktestConfig.from_dict(..., strict=True)`, and records the name in `config.preset_name` (`None` for a hand-built config). The profile dicts (`DEFAULT_PROFILE`, `BACKTRADER_PROFILE`, `VECTORBT_PROFILE`, `ZIPLINE_PROFILE`, `LEAN_PROFILE`, ...) and `list_profiles()` live in `ml4t.backtest.profiles`.
```python
bt_config = BacktestConfig.from_preset("backtrader")    # NB07: next open, whole shares, submission order, buffer
vbt_config = BacktestConfig.from_preset("vectorbt")     # same-bar close, fractional, exits first, no buffer
label = bt_config.preset_name or "custom"
```

Strategy skeleton (NB04; method bodies are inference from the notes)
```python
class RSIMeanReversionStrategy(Strategy):
    def __init__(self, rsi_lower=30, rsi_upper=70, position_size=0.95): ...
    def on_data(self, timestamp, data, context, broker):
        rsi = context.get("rsi")            # may be None/NaN early
        pos = broker.get_position(symbol)   # None or Position
        if pos is None and rsi < self.rsi_lower:
            broker.order_target_percent(symbol, self.position_size)
        elif pos is not None and rsi > self.rsi_upper:
            broker.order_target_percent(symbol, 0.0)
```
Futures engine (exact kwargs are inference; values from the notes)
```python
cfg = BacktestConfig(commission_type=CommissionType.PER_CONTRACT, allow_short_selling=True,
    allow_leverage=True, slippage_type=SlippageType.PERCENTAGE, slippage_rate=0.0005,
    execution_mode=ExecutionMode.NEXT_BAR)
Engine(..., config=cfg, contract_specs=specs)   # threads specs to Broker
```
Array parity (NB06)
```python
port = (weights.shift(1).fillna(0) * returns).sum(axis=1) - cost_drag   # cost_drag from weights.diff(), same lag
```
Cost reconciliation and break-even (NB09)
```python
net = reference_pnl - slippage_cost - commission
assert np.isclose(net, final_value - initial_cash)
break_even = reference_gross_pnl / one_way_reference_notional   # NaN if gross <= 0
```
Random-signal plumbing test (guardrails pre-flight 26(d); the case studies call `case_studies/utils/backtest_runner.run_plumbing_test(case_study, prices, strategy_spec, top_k=20, seed=42)` and score it against `PLUMBING_SHARPE_TOLERANCE = 1.5` (defined in `case_studies/us_firm_characteristics/11_backtest.py`, not in the runner); the library-level form needs no registry; a long-only random top-k book is the EW-universe benchmark in expectation, so do not add a second EW row beside it)
```python
rng = np.random.default_rng(42)                                                         # SEED = 42
noise = signals.with_columns(pl.Series("prediction", rng.standard_normal(signals.height)))
plumb = Engine(DataFeed(prices_df=prices, signals_df=noise), strategy, config).run()    # same engine, dates, costs, sizing
sharpe = plumb.metrics["sharpe"]                                                        # NaN = the random book went bankrupt: fix sizing / short leg first
assert abs(sharpe) < 1.5, f"pipeline is the alpha: random-signal Sharpe {sharpe:.2f}"  # and clearly negative after costs at intraday cadence
```
Point-in-time state (NB10): `expanding_past_median(values)` -> median of finite values strictly before each row; state = indicator vs that median; `shift(1)` before joining returns; `combine_state(vol, trend)`.

Library DSR from statistics (NB12)
```python
deflated_sharpe_ratio_from_statistics(observed_sharpe=sr_ann / np.sqrt(252),
    n_samples=n, n_trials=k, variance_trials=var_ann / 252,
    skewness=skew, excess_kurtosis=kurt - 3.0).probability
```
RAS (NB13)
```python
standardized = candidate_returns / candidate_returns.std(axis=0, ddof=1)   # no demeaning
observed_native = standardized.mean(axis=0)
R_hat = rademacher_complexity(standardized, n_simulations=5000)
adjusted_native = ras_sharpe_adjustment(observed_native, complexity=R_hat,
    n_samples=n_periods, n_strategies=n_candidates, delta=confidence)
adjusted_sharpe = adjusted_native * np.sqrt(periods_per_year)   # annualize last
positive_lower_bound = adjusted_sharpe > 0
```
Running: `uv run python 16_strategy_simulation/<notebook>.py`; tests `uv run pytest tests/test_chapter_notebooks.py -v -k "16_strategy_simulation"`; headless `MPLBACKEND=Agg PLOTLY_RENDERER=json`.

## Evidence from the book

- **NB01 (10 ETFs, 2010-2024):** the momentum/regime rule misses two of its three term-sheet criteria (which two is not in the notes) and is kept as the chapter's fixed reference; regime slice is descriptive only.
- **NB02 (CME, 2018-2023):** multiplier-omission error grows with the multiplier; $2/contract expressed as % of notional varies materially across the six demo products, so no single %-of-notional commission reproduces it; six-product vs full-universe numbers not in the notes.
- **NB03 (BTC/USDT):** the RSI threshold surface is flat in one direction and not the other, in-sample.
- **NB04 vs NB03:** all conventions shared except sizing resolution (execution open vs decision close); the residual gap is not attributable without a parity contract. The BTC window, the RSI windows swept and the resulting Sharpe/drawdown are not quantified in the notes.
- **NB05:** Kelly size falls after losing streaks and rebuilds after winning streaks on QQQ 2018-2019; the XLF/KRE ratio blows out in the March 2023 regional-bank stress; the breaker on SPY 2019-2021 stayed halted once beyond 10% drawdown, so breakers do not automatically improve returns.
- **NB06 (14 years, 10 ETFs):** daily-return correlation about 1 yet a compounding relative total-return gap; the array side charged on 78 target changes while the engine attempted 3,522 rebalances, 2,911 reached the market, 6,518 fills each paying commission and slippage; the gap accumulates rather than oscillating; whole-share residual is one-sided. The size of the relative total-return gap and the four summary stats are not quantified in the notes.
- **NB07 (ETF linear-model predictions, ~8 years):** fill ordering = exactly 0 in both share regimes (unreachable through `TargetWeightExecutor`); whole-share effects show first in trade count; cash-buffer drag > buffer size and path-dependent; same-bar-close profiles sit higher than next-open ones and are not comparable; the `vectorbt` preset differs from real VectorBT (honest residual); one-factor sum differs from the Backtrader-profile gap (interaction); rebalance mode worth far less than share type or cash buffer; starting capital changes conclusions. The starting capital, cash-buffer factor, dollar size of each one-factor effect and the exact `make_reference()` kwargs are not in the notes.
- **NB08:** the zero threshold gives different activation rates on ETF vs crypto sets because prediction scales and horizons differ; one trailing rule at one setting gives rates that "are not close" across the two sets; activation and transition rates rise together; lookback 21 yields the widest behaviour range.
- **NB09 (BTC perp 2020-2024, RSI 14/30/70, 10 + 5 bps):** gross P&L is not positive, so break-even and margin print NaN; beta < 1 is an exposure statement (out of market much of the time), not skill; gross exposure equals net by assertion.
- **NB10:** the Crisis-slice 5% VaR/CVaR is effectively the same as, and marginally shallower than, the pooled tail because the rule holds AGG/TLT when the curve is flat; the worst days are equity-holding days before the label caught up; two of four states dominate the day count; point estimates only.
- **NB11:** true SR 0.5 on 252 daily obs gives estimates centred on truth with spread about 1 Sharpe unit and about a quarter negative; on SPY 2020-2023 the Lo multiplier is not close to sqrt(252), negative weighted autocorrelation pushes the annualized Sharpe up, the per-year multiplier ranges over ~5 units and lag-1 autocorrelation flips sign; five years at SR 1 still has a wide interval; the top null of 30 (5 real at SR 0.8, 504 days) shows a respectable Sharpe and high PSR.
- **NB12:** the sqrt(2 log N) bound sits above the extreme-value expectation; cross-trial Sharpe variance about 1 at one year of daily data and swings by a third between draws; DSR for observed Sharpe 1.5 falls as trials rise; in the equal-edge sweep (30 variants, SR 0.5, 504 days) the max-vs-rest gap is luck the correction removes; all-noise vs mixed PBO outcomes are draw-dependent.
- **NB13:** correlated candidates have lower R_hat than independent ones, identical lowest; RAS lower bounds sit left of point estimates; 140 correlated grid candidates on 504 periods give a deeply negative lower bound; hourly 90-day series still carry the complexity charge; annualize-first would have charged roughly a sixth of what is owed.
- **NB14 (ETF baseline, 0-200 bps):** linear break-even exceeds the interpolated simulated crossing and the gap is "not small" over 14 years; turnover x fee is slightly below the simulator's charge; the library drag curve has the same shape and a different zero crossing; the baseline-fee row reproduces NB01 to the cent.
- **NB15-18:** Backtrader, Zipline and LEAN engine-time ratios are above one (ML4T faster) on every passing row; VectorBT ratios change direction across workloads; all supported real-strategy pairs pass at stated precisions; synthetic stress rows pass with different terminal values by design.

## Related references

- `chapters/06_strategy_definition.md` — term-sheet conventions the backtest is scored against.
- `chapters/07_defining_the_learning_task.md` — information coefficient (§7.5) and why it diverges from trading quality.
- `chapters/02_financial_data_universe.md` — CME EDA, 4 PM CT session aggregation, continuous-contract adjustment used by NB02.
- `chapters/17_portfolio_construction.md` — margin-based sizing and sector constraints that take over the baseline.
- `chapters/18_transaction_costs.md` — size/volume-dependent cost model replacing the flat rate in §16.6.
- `chapters/19_risk_management.md` — stateful patterns (breakers, Kelly) in production.
- `case_studies/etfs.md` — ETF panel and the `06_linear` predictions that feed NB07 and NB08.
- `case_studies/cme_futures.md` — contract specs, roll handling and the case-study backtests (`13_backtest`, `18_holdout_backtest`) that NB02's continuous-contract approximation defers to.
- `case_studies/crypto_perps_funding.md` — the crypto-perpetual GBM validation prediction set that NB08 converts to positions (NB03/04/09 load the local BTC perpetual dataset directly; the notes do not tie those bars to this case study).
- `libraries/ml4t_backtest.md` — `Engine`, `Broker`, `DataFeed`, `Strategy`, `BacktestConfig`, `TargetWeightExecutor`, analytics, sessions.
- `libraries/ml4t_diagnostic.md` — `PortfolioAnalysis`, Sharpe inference, DSR, PBO, RAS, reporting figures.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `workflow.md`, `companion_repo.md` — cross-cutting indexes this chapter feeds.
- Further reading: Lo (2002) Statistics of Sharpe Ratios; Mertens (2002); Bailey & López de Prado (2012, 2014) Sharpe frontier, DSR; Bailey et al. (2014, 2015) pseudo-mathematics, PBO; López de Prado (2018) AFML Ch. 14; López de Prado et al. (2025) How to Use the Sharpe Ratio; Paleologo (2024/2025) Elements of Quantitative Investing Ch. 8; White (2000) Reality Check; Hansen (2005) SPA; Harvey et al. (2016); Holm (1979); Benjamini & Hochberg (1995); McLean & Pontiff (2016); Jegadeesh & Titman (1993); Ang & Bekaert (2002); Alankar et al. (2023); Joubert et al. (2024) enhanced backtesting and three types of backtests; Mohri, Rostamizadeh & Talwalkar (2018); github RSv618/rademacher-anti-serum; github zoonek/2025-sharpe-ratio.

## Glossary

- **Backtest protocol** — written spec of signal timing, execution, rebalancing, sizing, costs, constraints, data availability and benchmark.
- **Term sheet** — pre-registered pass/fail criteria a strategy must meet.
- **Next-open fill** — an order from a close-derived signal executes at the following session's open.
- **Backward as-of join** — pair each date with the latest observation dated on or before it.
- **Risk-adjusted momentum** — trailing return divided by annualized trailing volatility.
- **Vectorized vs sequential backtest** — precomputed aligned arrays x returns vs a bar-by-bar loop carrying positions, cash, fills, equity.
- **ExecutionMode / ShareType / FillOrdering** — `NEXT_BAR`|`SAME_BAR`; `FRACTIONAL`|`INTEGER`; `EXIT_FIRST`|`FIFO`|`SEQUENTIAL`|`PRIORITY`.
- **RebalanceMode** — `SNAPSHOT` (targets once, all submitted), `INCREMENTAL` (recomputed per fill), `HYBRID` (targets once, cash checked live).
- **Cash buffer** — fraction of targets held back as cash to pay commissions.
- **Disappearance effect** — sub-half-share orders round to zero and are never submitted.
- **Interaction residual** — bundle effect minus the sum of one-factor effects.
- **Preset** — bundled configuration imitating another engine (`backtrader`, `vectorbt`).
- **ContractSpec** — static futures metadata: multiplier (point value), tick size, margin.
- **Half-Kelly** — half of `f* = W - (1-W)/R`.
- **Circuit breaker** — drawdown-zoned multiplier on new-entry sizing.
- **Activation frequency / state-transition frequency** — share of symbol-date rows active / share where state differs from the prior row; neither is turnover.
- **Trailing percentile rule / cross-sectional rule** — hold when a score beats the p-th percentile of its own last L scores / is in the top (100-p)% of symbols quoted that date.
- **Reference-price P&L** — P&L at the un-slipped execution base price, before slippage and commission.
- **Break-even cost rate** — gross reference-price P&L / total one-way reference notional (NB09 fixed-path estimator); the sweep estimators (net CAGR vs benchmark, net Sharpe = 0) and `cost_margin_ratio` are defined under "Break-even cost: one definition".
- **One-way turnover** — traded notional per side, including the entry of an open end-of-sample position.
- **Gross / net exposure** — sum |positions| / signed sum; equal for long-only.
- **MAE / MFE** — maximum adverse / favourable excursion of a completed trade.
- **Expanding past median** — median of all finite observations strictly before row t.
- **Risk-on / Caution / Crisis / Recovery** — (low vol, up) / (high vol, up) / (high vol, down) / (low vol, down).
- **VaR / CVaR (5%)** — loss exceeded by the worst day in twenty / mean loss on those days.
- **PSR** — probabilistic Sharpe ratio: P(true SR > benchmark) given sample length and shape.
- **MinTRL / T_plan** — minimum track-record length for an observed Sharpe to clear a threshold / observations needed before collection to detect a target Sharpe with stated power.
- **Lo annualization** — autocorrelation-corrected scaling of Sharpe across frequencies.
- **Mertens variance** — Sharpe variance adjusted for skewness and kurtosis.
- **DSR / expected max Sharpe** — PSR against the expected maximum of N null trials, `sqrt(V)[(1-g)Phi^-1(1-1/N) + g Phi^-1(1-e^-1/N)]`.
- **PBO / CSCV** — share of combinatorial symmetric splits where the IS-best strategy ranks in the bottom half OOS.
- **Rademacher complexity / Massart bound / RAS** — expected best alignment of a class with random signs / scale-aware finite-class upper bound on it / lower bound `theta >= theta_hat - 2R_hat - estimation terms`.
- **Effective number of trials** — independent-trial count implied by correlated candidates; smaller than the raw count.
- **Target-replay parity / negative control** — two engines fed identical frozen targets producing matching fills, valuations and terminal value / a one-unit perturbation of the first fill price the check must detect.
