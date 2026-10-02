# Chapter 18: Transaction Costs

> Transaction costs are a workflow constraint, not a backtest adjustment: they enter factor evaluation, simulation, portfolio construction, risk management and production monitoring, and many strategies fail "not because the forecast is wrong, but because the implementation problem was ignored." This chapter equips you to classify every cost by its charging unit, estimate and validate spreads, state (never pretend to measure) a square-root impact coefficient, bound capacity, pick an execution schedule by the benchmark the decision implies, simulate fills with `ml4t.backtest` impact, limit, commission and slippage models, and close the loop with TCA. Its position: separate what is measured (spreads, volumes, volatility, returns) from what is assumed (impact coefficients, fee schedules, persistence, gross Sharpe profiles). Turnover, break-even alpha and capacity then decide whether a strategy is deployable at all.

## When to use this reference

- Adding costs to a backtest or choosing a rung of the cost ladder (spread-only -> linear slippage -> square-root impact) and the matching `ml4t.backtest.models` / `ml4t.backtest.execution` classes.
- Reading a case study's `config/setup.yaml` `costs:` block and deciding whether it is a fee, an all-in cost, $/share, ticks/contract or a spread in bps.
- Estimating bid-ask spreads when only OHLCV exists (Corwin-Schultz, Roll), or validating those estimates against quote data.
- Deciding whether an intended order size is dominated by commission/spread or by impact (crossover size), or sanity-checking an impact number.
- Estimating strategy capacity / deployable AUM, or sizing a parent order as a fraction of ADV.
- Choosing or simulating an execution schedule (TWAP, VWAP, participation cap, Almgren-Chriss, tabular Q-learning) with partial fills carried forward.
- Choosing rebalance frequency: break-even alpha, net Sharpe by cadence, signal decay, persistence-cost ranking.
- Attributing gross-to-net decay to spread, impact, commission, financing or fund expenses, and picking the first lever.
- Judging whether an intraday strategy survives a conservative cost stack (cost cliff, break-even NAV turnover).
- Writing or auditing a TCA report (arrival / VWAP / close benchmarks; measured vs assumed lines).
- Reviewing any backtest that runs with `NoImpact`, fills capped orders in full, or sizes against a bar's total volume.

## Core ideas (the why)

- **Cost is a design constraint across the pipeline.** Treating it as a final haircut is the failure mode the chapter attacks; an impact model must be called on every fill, and backtests default to `NoImpact`, which is why they flatter high-turnover strategies.
- **Three cost classes need three modeling responses.** Explicit (commission, exchange fees: flat per unit), implicit (half-spread paid for immediacy; impact that grows with size), capacity (impact scaling with AUM until it consumes the edge). Collapsing them into "a slippage assumption" hides which one binds.
- **Dominance is size-dependent.** Commission and spread are flat per dollar; impact grows ~sqrt(size). Ask which component dominates at the size you intend to trade, not in general.
- **Units discipline.** A fee has a denominator (per share, per contract, per notional, via spread) and becomes a fraction of notional only after a trade detail (price, multiplier, quoted spread, fee tier) is supplied. Store published rates as text so unlike units cannot be added by accident.
- **Volume x price = dollars only when both describe the same transaction.** Share volume x adjusted close (Yahoo ETFs) is not turnover; a volume that counts price updates (OANDA FX) has no dollar equivalent.
- **Cost parameters are non-stationary.** Condition on time of day, volatility, liquidity and stress, but decompose the conditioning (frequency of estimate vs level) before believing it.
- **The impact coefficient eta cannot be measured from market data.** It needs your own fills vs decision price (Almgren et al. 2005). Without execution records it is an assumption "however precisely you quote it"; substituting a measured volatility into a model with an assumed coefficient is sensitivity analysis, not calibration.
- **Same-interval covariation of signed flow and return is association, not impact.** News causes both; separating temporary (liquidity, decays) from permanent (information, persists) impact needs post-order data.
- **Exponent decides curvature, coefficient decides level, and they are not independent.** Match models at a reference participation before comparing; the exponent changes the optimal schedule, not just the cost level.
- **Size orders as a share of volume, not a share count.** Participation is what impact models consume, what caps act on, and the only quantity comparable across instruments.
- **Execution algorithms manage trade-offs (impact vs timing risk vs forecast risk); they do not eliminate cost.** TWAP assumes all intervals equal; VWAP assumes today's volume shape resembles the recent average. Predictability makes a schedule auditable and exploitable.
- **Benchmark choice defines what counts as cost.** Against VWAP, trading slowly is a virtue; against arrival price (implementation shortfall) it is a risk. A tighter VWAP-tracking distribution means "average, reliably," not cheaper.
- **Almgren-Chriss: only temporary impact responds to the schedule.** Spread and permanent impact are fixed by position size. The frontier is curved: cheap risk reductions sit at the patient end only. Express urgency as a schedule (half-life), not an adjective.
- **Execution is a control problem, not a prediction problem.** A liquidity-only state (time, inventory, lagged spread, lagged vol) keeps an adaptive policy from drifting into directional timing.
- **Impact models are stateless; a backtest must carry the unfilled remainder.** Permanent impact from earlier child orders is the caller's job; capping participation and then quietly filling everything assumes away the constraint.
- **Limits and impact models answer different questions.** The limit decides how much trades now; the impact model prices the concession that slice pays; the broker composes both in sequence. The cap is the speed-vs-impact dial, and `min_volume` is a separate gate that blocks thin bars outright rather than sizing into them.
- **TCA measures benchmark slippage but cannot attribute shortfall to impact vs timing without a counterfactual price.** Any split is a model's opinion and must be labeled as such.
- **Capacity is not a property of a market.** It is where a stated gross return meets a modeled cost under stated turnover, participation ceiling and concentration.
- **Faster trading does not automatically capture more usable signal.** It raises turnover; break-even alpha grows linearly with turnover; the signal's decay rate decides whether acting sooner pays.
- **The cost stack is layered** (spread, impact, commission/fees, financing, fund expenses) and each layer answers to a different lever. Lowering turnover does not fix a borrow-driven problem; composition, not cross-market magnitude, identifies the first lever.
- **The cost cliff is a turnover phenomenon measured in NAV turnover, not trade count.** Cost as a share of gross return is the cleanest diagnostic.
- **Reproduce a library result a second way** (closed-form recomputation that raises on mismatch) before building on it.

## Method recipes (the how)

### Cost taxonomy and units discipline (`18_transaction_costs/01_cost_taxonomy`)

1. Record every published rate as text with `fee_evidence_row(venue, instrument, rate, status, source)`; stamp `SOURCE_CHECK_DATE = "2026-07-21"` (the date schedules were read). Binance stays a case-study assumption because its page is account-tier dependent.
2. Classify each case-study `config/setup.yaml` `costs:` block with `classify_cost_units(costs) -> (unit, is_spread_bps, missing)` / `cost_schema_row(cs_id, setup)` (`CASE_STUDIES` lists the nine ids). Fill `configured_bps` only when the block already states bps; otherwise name the missing trade detail.
3. Convert to bps once, after price / multiplier / quoted spread / fee tier is written down. 1 bp = 0.01%; 10 bps of $1,000,000 = $1,000.
4. Settings: `MAX_SYMBOLS = 0` (0 = all instruments, alphabetical so selection never depends on data).

| `costs:` block type | Meaning | Usable directly? |
|---|---|---|
| fee | charged on top of spread | needs spread + impact added |
| all-in | already contains spread + commission + impact | yes, but do not add another spread |
| $/share or ticks/contract | per-unit | needs price (and multiplier) |
| spread in bps | already a fraction of notional | yes, once a single rate and the half-vs-full convention are stated (nb 02's `classify_cost_units` returns `is_spread_bps=False` for the `spread_bps` block because it is a range per pair class with no half/full convention) |

Finding: almost none of the nine blocks record a round-trip spread in bps; most record a fee, an all-in cost, $/share or ticks/contract.

### Dollar turnover by market (nb 01 / 03)

| Panel | Turnover rule | Dollar axis |
|---|---|---|
| S&P 500, NASDAQ-100 equities | share volume x traded close | yes |
| Crypto (Binance perps) | base-asset volume x price per 8h bar; three funding-aligned bars summed by UTC date | yes |
| CME futures | contracts x price x multiplier, multiplier = tick value / tick size (matches backtest contract spec); notional on raw traded prices, returns on roll-adjusted prices | yes |
| ETFs (Yahoo adjusted OHLC) | none; report shares/day | no |
| FX (OANDA) | volume = price updates; report ticks/day | no |

Use `summarize_dollar_turnover(df, symbol_col, date_col, asset_class)` (keeps `coverage_start` / `coverage_end`); gate capacity analysis with `supports_usd_capacity` / `volume_contract` columns.

### Cost-component dominance and crossover size (nb 01 Section 3)

- Inputs per market (`cost_params` dict keys): `commission_bps`, `spread_bps` (the half-spread, per the notebook prose), `sigma` (daily), `adv_usd`, `impact_eta`. `cost_components(size, params) -> {commission, spread, impact}` in bps; `dominant_component_label(costs)` names the largest or ties.
- Crossover size solves `eta * sigma * sqrt(Q / ADV) = max(commission, half_spread)` (all in bps). Below it a spread-only model may suffice; above it sqrt impact is required.
- Scenario grid (round illustrative figures, labels as in the notebook; commission / spread / eta / sigma / ADV): ETFs 1.0 / 3.0 / 0.05 / 0.015 / 5e8; Crypto Perps 4.0 / 2.0 / 0.03 / 0.04 / 1e9; CME Futures 1.5 / 1.5 / 0.04 / 0.01 / 2e9; FX Pairs 0.0 / 2.0 / 0.02 / 0.005 / 5e9 (spread-only); S&P 500 1.0 / 2.0 / 0.05 / 0.015 / 1e8; US Equities 1.0 / 8.0 / 0.10 / 0.025 / 5e6.

### Breakeven alpha and borrow cost (nb 01 Sections 5-6; nb 09)

- nb 01: `alpha_breakeven = annual_traded_notional_turnover * one_way_cost_bps / 10_000`, turnover = dollars traded per year / average portfolio capital. Report as a sensitivity grid over turnover x cost.
- nb 09 (`calculate_break_even_alpha(annual_turnover, round_trip_cost_bps)`): **break-even alpha = annual one-way turnover x round-trip cost**, with one-way turnover = 0.5 * sum|dw| and round-trip = 2 x one-way cost. Do not multiply round-trip turnover by round-trip cost.
- Borrow: annual rate on short-side borrowed value; a running cost, not per trade, not scalable by turnover; tens of bps/yr (easy large-cap) to several percent (hard-to-locate). Net = gross alpha - trading cost - borrow.

### Spread estimation from OHLCV (`02_spread_estimation`)

Settings: `MAX_SYMBOLS = 50` (per panel, most active by volume), `CS_WINDOW = 20`, `ROLL_WINDOW = 20` sessions (~one trading month); NQ-100 quote benchmark on `NQ100_SYMBOLS = 12`, `2021-10-01`..`2021-12-31`.

| Estimator | Formula | Failure mode | Notes |
|---|---|---|---|
| Corwin-Schultz | `beta = E[sum_{j=1,2} ln(H_j/L_j)^2]`, `gamma = E[ln(H_[1,2]/L_[1,2])^2]`, `alpha = (sqrt(2 beta) - sqrt(beta))/(3 - 2 sqrt 2) - sqrt(gamma/(3 - 2 sqrt 2))`, `S_CS = 2(e^alpha - 1)/(1 + e^alpha)` | negative alpha clamped to 0 (volatility swamps spread) | safe on back-adjusted futures (ratio cancels the factor); use for ranking |
| Roll | `r_t = P_t/P_{t-1} - 1`, `S_Roll = 2 sqrt(-Cov(r_t, r_{t-1}))` | positive covariance -> 0 (price discovery swamps bounce) | degrades at daily frequency on liquid large caps |

Steps:
1. Wrap every shift / rolling expression in polars `.over("symbol")` (`estimate_spreads(df)` adds CS and Roll columns symbol-isolated).
2. Aggregate crypto 8h bars to UTC days first (`aggregate_crypto_daily`: open = first, high = max, low = min, close = last, volume = sum of three) so the 20-row window means the same time everywhere; "both estimators count rows, not time."
3. Report the zero-share next to the median level; a median over a mostly-zero column is a clamp, not a level.
4. Validate on quoted data before trusting any level (next recipe).

### Quote benchmark and validation metrics (nb 02 Sections 2-3)

1. `load_nq_microstructure() -> (raw, regular_session, valid_quote)` LazyFrames; the raw feed bypasses the loader's regular-hours filter, so restore 09:30-16:00 ET before selecting symbols or aggregating (nb 02 filters on hour/minute inline; nb 03 packages the same mask as `regular_session_mask()`).
2. Keep trade and quote panels separate. Quote filter drops: no quote, bid <= 0, no trade, locked (bid == ask), crossed (bid > ask). Daily bar from all regular-session trade minutes (high = max trade, low = min trade, open/close = first/last).
3. Benchmark = per-minute relative quoted spread, volume-weighted, over valid-quote minutes; join on (symbol, date). Symbol ranking by whole-quarter volume is descriptive only (look-ahead in a backtest).
4. Keep an integrity ledger / filter table that sums every exclusion back to the raw row count (nb 02 builds it inline; nb 03's `session_integrity_ledger(raw, regular, selected)` is the reusable form).
5. `compute_validation_metrics(validation)` over per-symbol medians: Pearson r (ranking, scale-invariant), MAE in bps, bias (signed mean error; positive = reads wider than quoted), identity-line R^2 = `1 - sum(y - yhat)^2 / sum(y - ybar)^2` against the 45-degree line (can go negative; not r^2). Let the validation set the weight carried by unvalidated panels.

### VIX-state conditioning (nb 02 Section 6; nb 03 Section 8)

- Four states = VIX quartiles over full history (descriptive; look-ahead if used as a trading rule; use expanding/trailing quantiles for live rules).
- Decompose the mean over sessions into (a) rate of non-zero estimates and (b) level when non-zero before reading a rise as "spreads widen."
- Plot VIX and spread on stacked panels with one time axis, never dual y-axes. `regime_impact_row(regime)` builds the per-state impact row.

### Square-root impact and Kyle-lambda from signed flow (`03_market_impact_calibration`)

- Baseline law: `impact = eta * sigma * sqrt(Q / ADV)` (x 10,000 for bps); sigma = trailing daily volatility (`VOL_WINDOW = 20`), ADV = trailing `ADV_WINDOW = 20` dollar turnover, participation = Q/ADV. Doubling Q raises cost per dollar ~41%; a 100x order pays 10x impact per dollar. Assumes the order is worked over ~a day at constant rate.
- `ETA_SCENARIO` (stated, not estimated): ETFs 0.10, S&P 500 0.15, NASDAQ-100 0.12, CME futures 0.08, crypto perps 0.30, FX 0.05. Keep `eta_assumption` and `median_daily_sigma` in separate columns; trust orderings more than levels.
- `add_impact_features(df, entity_col, time_col="timestamp")` adds returns, native ADV and volatility; `keep_top_symbols(df, entity_col)` with `MAX_SYMBOLS = 50`; `SEED = 42`.
- Kyle-lambda-style slope (Section 3, `estimate_normalized_lambda(symbol, flow) -> dict | None`): signed flow `Q_t = uptick volume - downtick volume` per minute (tick rule); regress `r_t^bps = lambda_part * (1e4 * Q_t / ADV) + eps_t` per symbol with Huber regression; drop the first minute of each session (overnight move is a different quantity); `NQ100_SYMBOLS = 100`, 2021Q4, regular session restored via `regular_session_mask()`, `load_nq_signed_flow_sample() -> (sample, ledger)`. Accept negative R^2 as a property of the robust criterion.
- Cross-section: regress log10(lambda) on log10(dollar turnover) (power law); exclude negative-lambda symbols and report the count; read the slope with its covariance-based CI. Representative symbol = slope closest to median; scatter clipped at `PLOT_CLIP_QUANTILE = 0.005` for display only.
- Intraday volume profile (Section 5): measured from minute bars on the execution grid; U-shaped. Compute participation against the volume of the interval actually traded in.
- Quick sanity (inference from the scenario grids): eta 0.05-0.30 with daily sigma 0.5%-4% at 1% participation gives ~0.25-12 bps; a figure far outside that for a liquid market is a units error.

### Capacity curves (nb 03 Section 6, `capacity_curve(row, aum_grid) -> np.ndarray`)

1. Net alpha per rebalance = `GROSS_ALPHA_BPS (50) - impact(participation)`; participation = AUM x `TURNOVER_PER_REBALANCE (0.30)` / universe total daily dollar volume, spread pro-rata so participation is equal everywhere (formula is inference from the text).
2. Curve is NaN beyond `MAX_FEASIBLE_PARTICIPATION = 0.20` (model-validity limit, not an execution limit).
3. Report two numbers: AUM at the participation ceiling, and AUM where impact = gross return (exists only if inside the ceiling). Deployable AUM = the smaller; if the second does not exist, the ceiling is the binding statement.
4. Run only on markets with dollar volume (crypto, CME with multipliers, S&P 500, NASDAQ-100). State capital, turnover, participation limit and concentration the number rests on.

### Shape normalization: linear vs square-root vs power-law (nb 03 Section 7; nb 05 Part 6; nb 06)

- Match models to charge the same at `MODEL_REFERENCE_PARTICIPATION = 0.01` (nb 05: at the even schedule's own participation). Below the reference the sqrt model charges more; above, less. Compare only the shape afterwards.
- Library defaults coincide at `c_lin * p = c_sqrt * sigma * sqrt(p)` -> p = (0.5 x 0.02 / 0.1)^2 = 1% (inference from defaults).

### TWAP and VWAP schedules with simulated fills (`04_vwap_twap_execution`)

Settings: `EXEC_SYMBOL = "AAPL"` (real AlgoSeek minute bars despite the README's "synthetic" label), `TAQ_START_DATE = "2021-10-01"`, `TAQ_END_DATE = "2021-12-31"`, `PROFILE_END_DATE = "2021-11-15"`, `INTERVAL_MINUTES = 15` (26 intervals 09:30-16:00), `ORDER_SHARES = 100_000`, `IMPACT_BPS = 5.0`, `SEED = 42`, `SESSION_CLOSE_HOUR = 16`.

| Schedule | Slice rule | Needs | Prefer when |
|---|---|---|---|
| TWAP `TWAPAlgorithm(total_shares, start_time, end_time, interval_minutes=5)` | `size_t = total / n_intervals` | nothing | thin names, scheduled-event days, shape-shifting markets |
| VWAP `VWAPAlgorithm(total_shares, start_time, end_time, volume_curve=None, interval_minutes=5)` | `size_t = total * E[vol_t] / sum E[vol]` | volume profile | liquid name with a stable daily volume pattern |
| Percent-of-volume | fixed fraction of printed volume | live volume | participation must be controlled exactly; completion time may float |
| Implementation shortfall / AC | front-loaded by risk aversion | sigma, eta, lambda | signal decays or decision price is the benchmark |

Steps:
1. Estimate the profile on sessions <= `PROFILE_END_DATE` (fixed calendar split, not a sample fraction) via `load_intraday_panel(symbols, start, end, interval_minutes)`; `day_paths(panel, symbol, n_buckets)` keeps only sessions with full `N_BUCKETS` coverage; `normalize_volume_curve(curve, n_intervals)`; `shares_from_profile(profile, total_shares)` allocates integer shares; `build_twap_schedule` / `build_vwap_schedule(..., volume_curve)`; `get_target_at_time(t)` / `latest_cumulative_target(schedule, t)`.
2. `execute_day(shares, price_path, volume_path, impact_bps) -> (realized_vwap, exec_price)`: `participation = shares / max(interval_volume, 1)`, `impact_fraction = impact_bps / 1e4 * sqrt(participation)`, `exec_price = price_path * (1 + impact_fraction)` (buy-side sign; interval volume floored at one share), so `IMPACT_BPS` is paid at 100% interval participation; raises on mismatched lengths, negative shares, or non-finite / non-positive prices.
3. Compute `market_vwap_of(price, volume) = sum(p v)/sum(v)` only after the schedules execute; `run_across_days` over all held-out sessions; `summarize_tracking` reports bias (signed mean), std, MAE and worst session separately.
4. Treat results as lower bounds: impact is charged against volume that includes the order's own shares and nothing goes unfilled.

### Almgren-Chriss optimal execution (`05_almgren_chriss_optimal_execution`)

Sell X shares over horizon T in N periods, `tau = T/N`, synthetic prices.

| Quantity | Formula |
|---|---|
| Price dynamics | `S_k = S_{k-1} - gamma n_k + sigma_P sqrt(tau) xi_k`, `sigma_P = S0 sigma / sqrt(252)`, no drift |
| Permanent impact | `g(n) = gamma n` (information, persists) |
| Temporary impact | `h(n) = epsilon sign(n) + eta n / tau` (liquidity, charged only to the causing slice) |
| Expected cost (IS) | `E[C] = 0.5 gamma X^2 + epsilon X + eta sum_k n_k^2 / tau` (only the eta term depends on the schedule) |
| Variance (timing risk) | `V[C] = sigma_P^2 sum_k tau x_k^2`, x_k = post-trade position |
| Objective | `min E[C] + lambda V[C]` |
| Exact discrete solution | `theta = lambda sigma_P^2 tau^2 / eta`, `alpha = arccosh(1 + theta/2) = 2 asinh(sqrt(theta)/2)`, `x_k = X sinh(alpha (N-k)) / sinh(alpha N)` |
| Continuous approximation | `kappa ~ sqrt(lambda sigma_P^2 / eta)` misstates the optimum when theta is not small; do not use |

Parameters instantiated: `X=100_000, T=5.0 days, N=50, S0=100.0, sigma=0.30 (annual), gamma=25.0 (permanent bps at 100% ADV), eta=160.0 (temporary bps at 100% interval participation), epsilon=1.0 (half-spread bps), ADV=1_000_000`; dataclass defaults differ only in `T=1.0, N=10`. `__post_init__` converts `gamma_price = gamma*S0/(ADV*1e4)`, `eta_price = eta*S0/(ADV*1e4)`, `epsilon_price = epsilon*S0/1e4` and raises if X, T, N, S0, ADV <= 0, eta <= 0, or sigma/gamma/epsilon < 0.

Steps:
1. `optimal_trajectory(params, risk_aversion) -> (times, trajectory)` for `TRAJECTORY_RISK_AVERSIONS = [0.0, 1e-8, 1e-7, 1e-6, 1e-5]`; label each by `liquidation_half_life(times, trajectory)` (even schedule = T/2), never by raw lambda.
2. `compute_efficient_frontier(params, n_points=50)` sweeps lambda over `FRONTIER_LOG10_RANGE = (-9, -5)` (`N_RISK_AVERSIONS = 100` in the notebook): expected cost vs std.
3. `simulate_execution(params, trade_list, n_simulations=1000, seed=42)` for `SIMULATION_RISK_AVERSIONS = [0.0, 1e-7, 1e-6]`: shocks shared across schedules, antithetic mirror pairs (`N_SIMULATIONS` must be even) so the simulated mean equals E[C]; report mean, 5th and 95th percentiles. Simulated fill = mid - half-spread - full temporary impact - half of the slice's own permanent impact; a zero-shock run reconciles exactly with E[C].
4. `ExtendedParams(impact_exponent in (0,1])` + `expected_cost_nonlinear(trade_list, params, reference_participation)` prices the same schedule under a matched power law (coefficient rescaled at the reference participation); the schedule is priced, not re-optimized.
5. `generate_tca_report(executed_trades, benchmark_prices, params)` (see TCA recipe).

### ml4t market impact models (`06_ml4t_execution_demo`, `ml4t.backtest.execution.impact`)

Contract: `MarketImpactModel.calculate(quantity, price, volume, is_buy) -> float` returns a **signed per-share price move in currency units** (positive buy, negative sell) so `fill = decision_price + impact` is always adverse.

| Model | Formula | Defaults | Notes |
|---|---|---|---|
| `NoImpact()` | 0.0 | - | backtest default; flatters turnover |
| `LinearImpact(coefficient=0.1)` | `coefficient * (Q/V) * P` | 0.1 | coefficient x 10,000 = bps at full participation; last share costs same as first; 0.0 when volume invalid |
| `SquareRootImpact(coefficient=0.5, volatility=0.02, adv_factor=1.0)` | `coefficient * sigma * sqrt(Q/ADV) * P`, `ADV = volume * adv_factor` | 0.5 / 0.02 / 1.0 | typical coefficient 0.1-1.0; doubling sigma doubles cost; `adv_factor` = bars per session (26 for 15-min, 1 for daily) |
| `PowerLawImpact(coefficient=0.1, exponent=0.5, min_impact=0.0)` | `coefficient * (Q/V)^exponent * P` | 0.1 / 0.5 / 0.0 | exponent 1 linear, 0.5 sqrt, (0,1) concave, >1 convex; at a fixed coefficient a smaller exponent raises the level for participation < 1, so curvature and coefficient are chosen jointly; `min_impact` is a minimum per-share move (signed when volume invalid), not a fixed fee |

Notebook 06 settings: `UNIVERSE = ["SPY","QQQ","IWM","EEM","DBC"]`, `LOOKBACK_DAYS = 365`, `IMPACT_COEFFICIENT = 0.5`, `SCENARIO_VOLATILITY = 0.02`, `MAX_PARTICIPATION = 0.10`, `PARAMETERIZATION_PARTICIPATION = 0.05`, `PERSISTENCE_FRACTION = 0.50`, `SEED = 42`. Sliced-parent simulation: equal children vs fixed daily volume; reference price for child k+1 = child k reference + `PERSISTENCE_FRACTION` x child k impact, applied identically to all models; first-child oracle asserts `linear_signed_move == coefficient * (child_shares / daily_volume) * price`. Measured: ETF mean price, ADV, daily-return std. Assumed: all coefficients (including library defaults), scenario volatility, persistence, cap.

### Participation limits and the fill pipeline (nb 06 Part 6; `07_ml4t_volume_participation`; `ml4t.backtest.execution.limits`)

Base class `ExecutionLimits(ABC)`; every limit implements `calculate(order_quantity, bar_volume, price) -> ExecutionResult` (`price` is a required third positional, echoed back as the placeholder `adjusted_price`).

| Limit | Behaviour |
|---|---|
| `NoLimits()` | fillable = order, remaining 0; participation = order / bar_volume (0 when volume is missing or zero) |
| `PositiveVolumeLimit()` | full fill only if `bar_volume > 0`, else fillable 0 and remaining = order |
| `VolumeParticipationLimit(max_participation=0.10, min_volume=0.0)` | missing volume -> full fill; `bar_volume < min_volume` -> fillable 0, remaining = order; else `fillable = min(order, bar_volume * max_participation)`, `remaining = order - fillable`, `participation_rate = fillable / bar_volume` |
| `AdaptiveParticipationLimit(base_participation=0.10, volatility_factor=0.5, max_participation=0.25, min_participation=0.02, avg_volatility=0.02)` | `calculate(..., volatility=None)`: `participation = base_participation * (1 - volatility_factor * (volatility / avg_volatility - 1))`, clamped to `[min_participation, max_participation]` (no `volatility` -> base rate); missing volume -> full fill; then the same cap arithmetic as above |

- `ExecutionResult(fillable_quantity, remaining_quantity, adjusted_price, impact_cost=0.0, participation_rate=0.0)`: `adjusted_price` / `impact_cost` are placeholders on the limit's result. The production `FillExecutor` applies limit -> signed impact once -> slippage -> emits a `Fill` whose `price` includes the adjustments; read realized cost from the `Fill`.
- Realistic-fill sequence: (1) size as % of volume; (2) apply participation limit (+ `min_volume`); (3) add signed impact once; (4) apply slippage; (5) apply commission; (6) carry `remaining_quantity` to the next bar; (7) recompute one fill by closed form (`compose_limit_and_impact(quantity, side, decision_price, volume, impact_model, max_participation)` is the nb 06 teaching helper).
- nb 07 walk: `EXEC_SYMBOLS = ["AAPL","MSFT","AMZN","GOOGL","META"]`, `PRIMARY_SYMBOL = "AAPL"`, 2021Q4, `INTERVAL_MINUTES = 15`, `CALIBRATION_SESSIONS = 20` (ADV measured on completed sessions before execution starts), parent = `ORDER_PCT_ADV (0.5) x ADV`, `PARTICIPATION_RATES = [0.05, 0.10, 0.25]`. `participation_walk(intervals, order_shares, max_participation, min_volume=0.0)` fills `cap x realized interval volume` at interval VWAP, queues the remainder to the next observed interval; `execution_frame(rows, order_shares)` adds cumulative shares, cost, completion; `add_execution_traces` draws the four-panel diagnostic (participation share per interval). Cap ladder: 5% stealth, 10% institutional default, 25% aggressive (urgent, very liquid). Participation regimes: low single-digit % absorbed with little trace; ~10% moves price; ~50% cannot fill near arrival; desks cap at "low-to-mid tens of percent".

### RL dynamic execution (`08_ml_dynamic_execution`)

- MDP: AAPL 15-min grid, `N_BUCKETS = 26`, parent `ORDER_PCT_ADV = 0.10` x ADV. State `(time remaining, inventory bin [INV_BINS = 10], spread_{t-1} regime, vol_{t-1} regime)`; regimes binned by quantiles measured on training sessions only; t = 0 uses training medians (`observable_regime`, `encode_state`). No price/return features.
- Action = multiplier on the VWAP base slice from `ACTIONS = [0.5, 0.75, 1.0, 1.3, 1.6]` (1.0 reproduces VWAP, so VWAP lies inside the policy class).
- Per-fill cost `c_t = 0.5 spread_t + kappa sqrt(q_t / V_t)` bps, kappa = `IMPACT_COEF_BPS = 10` (`interval_cost_bps`). Score `sum_t c_t q_t/Q + lambda sum_t (Ibar_t/Q) sigma_t + eta I_H/Q`, lambda = `EXPOSURE_WEIGHT = 0.25`, Ibar_t = mean of pre/post-fill inventory, sigma_t = intra-interval realized vol (bps), eta = `SHORTFALL_PENALTY_BPS = 100` (`schedule_components`, `schedule_score_bps`). Linear exposure proxy, explicitly not the AC quadratic variance.
- Fills `capped_fill(requested, remaining, interval_volume) = min(requested, remaining, MAX_PARTICIPATION (0.25) x interval volume)`; the last interval requests all residual, cap still applies, residual pays the shortfall penalty.
- Learning: `r_t = -[c_t q_t/Q + lambda (Ibar_t/Q) sigma_t]`; `Q(s,a) <- Q(s,a) + ALPHA [r + max_a' Q(s',a') - Q(s,a)]`, `ALPHA = 0.30`, no discounting, epsilon-greedy with `EPSILON_START = 0.5` decayed linearly to 0 over `N_EPOCHS = 800` (`run_episode`, `train_q_policy`); Q-table of a few hundred states.
- Evaluation: chronological split `TRAIN_FRACTION = 0.6`; greedy policy vs TWAP and VWAP (`shares_from_weights`, `execute_static_schedule` carries capped misses forward) with the same cap and penalty; `evaluate(test_days, sessions, q_table)`, `participation_diagnostics` (max, terminal participation, terminal shares), paired session-by-session win counts, `matched_volatility_policy(q_table)` (calm vs volatile actions at fixed time/inventory/spread). Report as descriptive for one symbol, one quarter.

### Frequency trade-off (`09_frequency_tradeoff`)

- Data: `ETF_SYMBOLS = ["SPY","QQQ","IWM","XLF","EEM","XLE","XLU","FXI"]`, 2019-01-01..2023-12-31; `MOMENTUM_LOOKBACK = 63`, `TOP_N = 3`; cadences daily / weekly / biweekly / monthly (`FREQUENCIES`).
- `CostAssumptions(name, spread_bps, impact_bps, commission_bps)` (half-spread, impact allowance, commission; bps one-way) with properties `total_one_way` and `round_trip = 2 x one_way`; `COST_SCENARIOS`: `HIGH_FRICTION_COSTS` 3.0 + 2.0 + 0.0 = 5.0 one-way (10 round-trip), `MEDIUM_FRICTION_COSTS` 1.0 + 3.0 + 0.5 = 4.5 (9), `LOW_FRICTION_COSTS` 0.2 + 0.5 + 0.1 = 0.8 (1.6); high and medium differ by 1 bp round-trip. `FREQUENCIES`: Daily 1 session / 252 per year, Weekly 5 / 52, Biweekly 10 / 26, Monthly 21 / 12.
- `momentum_frequency_backtest(prices, rebalance_days, lookback=63, top_n=3) -> (daily returns, turnover path)`: equal-weight top-N by lagged momentum; weights drift between rebalances; turnover compares the new target with pre-trade drifted weights; one-way turnover = 0.5 sum|dw|.
- `real_net_by_frequency(cost_assumptions)` subtracts scenario cost on each rebalance day, then annualizes (`annualized_sharpe` uses sample vol). `simulate_frequency_comparison(gross_sharpe=2.0, annual_vol=0.15, cost_assumptions)` applies `SCENARIO_GROSS_SHARPE = 2.0`, `SCENARIO_ANNUAL_VOL = 0.15` to every cadence's measured turnover; `evaluate_decay_scenario(..., signal_decay_rate=0.1)` decays captured alpha exponentially with rebalance delay over `DECAY_RATES = [0.01, 0.03, 0.05, 0.10, 0.20]` (`EXAMPLE_DECAY_RATE = 0.05`).
- Persistence-cost proxy (alpha-to-go teaching stand-in): `S(phi, Gamma) = phi / (1 - phi + Gamma)` for AR(1) persistence phi and cost pressure Gamma; dimensionless, may exceed 1, ranking only (log color scale).

### Gross-to-net cost stack (`10_gross_vs_net_performance`)

Waterfall: Gross - spread - impact = Trading P&L; - commission/fees - financing (margin, borrow) = Net Trading P&L; - fund expenses (management, admin) = Investor Net Return.

| `CostStack` field | Default |
|---|---|
| `spread_cost_bps` | 2.0 |
| `impact_cost_bps` | 5.0 |
| `commission_bps` | 1.0 |
| `exchange_fee_bps` | 0.5 |
| `margin_rate_annual` | 0.05 |
| `borrow_rate_annual` | 0.01 |
| `management_fee_annual` | 0.02 |
| `admin_fee_annual` | 0.002 |

Methods: `trading_cost_per_trade(trade_size_pct=0.1)`, `annual_trading_cost(annual_turnover)`, `annual_financing_cost(gross_leverage=1.0, short_pct=0.0)`, `annual_expense_cost()`. `build_strategy(gross_returns, annual_turnover, gross_leverage=1.0, short_pct=0.0, name)` (per-day one-way turnover implied by the annual figure), `apply_cost_stack(strategy, costs)`, `compute_performance(returns)`. Three configurations on real daily ETF returns 2021-01-01..2023-12-31: QQQ high-turnover long-only, dollar-neutral QQQ-IWM leveraged long-short, SPY low-turnover. Section 8: net Sharpe vs annual one-way turnover at fixed cost (linear decline). Section 9: vector-L2 diagnostic (20 assets, fixed base covariance rotated slightly each period, rolling min-variance weights, L1 `sum|w_t - w_{t-1}|` and L2 norms) measures maintenance turnover from covariance drift; not an alpha decomposition. Section 11 bins the three configurations by net Sharpe at two presentation thresholds (> 1.0, (0.5, 1.0], <= 0.5), explicitly not a go/no-go rule.

### Cost cliff for intraday strategies (`11_cost_cliff`)

1. `measure_nasdaq100_spreads(start, end)` with `SPREAD_START_DATE = "2021-12-01"`, `SPREAD_END_DATE = "2021-12-31"`: per-symbol volume-weighted relative quoted spread (bps) over the regular session; anchor = `MEASURED_HALF_SPREAD_BPS = median_rel_spread_bps / 2` (median symbol, not the mega-cap).
2. `IntradayCostStack` dataclass (bps per side) with defaults `spread_half=3.0`, `market_impact=2.0`, `slippage=1.5`, `commission=0.5`, `exchange_fee=0.3`, `clearing_fee=0.1` (properties `one_way_bps` = 7.4, `round_trip_bps` = 14.8 for the bare defaults). The presets the notebook actually applies override them: `CROSSING` = (`MEASURED_HALF_SPREAD_BPS`, 3.0, 2.0, 0.0 commission-free, 0.3, 0.1) = measured half-spread + 5.4 bps one-way; `WORKED_ORDER` = (0.375 x measured half-spread, 2.5, 1.0, 0.3, 0.2, 0.05); `PASSIVE_LOW_COST` = (0.0, 0.5, 0.2, 0.05, -0.2 maker rebate, 0.02). Only the crossing spread is measured; every other field is an illustrative assumption.
3. `IntradayStrategy(NamedTuple)`: name, gross Sharpe, annual vol, daily round-trip NAV turnover tau; hypothetical profiles `HIGH_TURNOVER` (2.5, 0.20, 1.0/day), `MODERATE_TURNOVER` (2.0, 0.15, 0.4), `LOW_TURNOVER` (1.5, 0.12, 0.1); no signal is fit.
4. `calculate_intraday_net_performance(strategy, costs, trading_days=252)`: `C_ann = D tau c_rt / 1e4`, `S_n = (S_g sigma - C_ann) / sigma` (cost changes return, not vol); cost / gross return = fractional Sharpe reduction.
5. `calculate_break_even_turnover(target_net_sharpe, gross_sharpe, annual_vol, costs, trading_days=252)`: max daily round-trip NAV turnover for a target net Sharpe; (inference) `tau_max = (S_g - S_target) sigma 1e4 / (D c_rt)`; the ceilings table uses `target_net_sharpe=0.5`, gross Sharpe in {1.5, 2.0, 2.5, 3.0}, `annual_vol=0.15` under each preset; plot on a log y-axis.
6. Scope: excludes financing, taxes, passive-fill risk, impact uncertainty; vol held fixed. A sensitivity, not a backtest or capacity estimate.

### Commission and slippage models (`12_commission_slippage_comparison`, `ml4t.backtest.models`)

| Family | Classes | Unit |
|---|---|---|
| Commission (equity-style: shares, price) | `NoCommission()`, `PercentageCommission(rate=0.0)`, `PerShareCommission(per_share=0.0, minimum=0.0)`, `TieredCommission(tiers=[(threshold, rate), ...])`, `CombinedCommission(percentage=0.0, fixed=0.0)` | dollars per trade |
| Commission (futures: contracts, price, multiplier) | `FuturesCommission(per_trade=0.0, per_block=0.0, percentage=0.0)` | dollars per trade |
| Slippage (per-unit price adjustment) | `NoSlippage()`, `FixedSlippage(amount=0.0)`, `SpreadSlippage(spread=0.0, convention="full_spread")` (charges half per side by default), `PercentageSlippage(rate=0.0)`, `VolumeShareSlippage(impact_factor=0.0)` (only participation-responsive one) | price per unit |
| Slippage (futures) | `FuturesSlippage(slippage_points=0.0)` | total dollars (multiplier=1.0 default) |

- Notebook variants: `PercentageCommission(0.001)` (10 bp), `PerShareCommission(0.005, minimum=1.0)`, `CombinedCommission(percentage=0.0005, fixed=1.0)`, `TieredCommission([(10_000, 0.001), (50_000, 0.0008), (inf, 0.0005)])`, `FuturesCommission(per_block=2.25)`; `FixedSlippage(0.01)`, `SpreadSlippage(spread=0.04)`, `PercentageSlippage(0.001)`, `VolumeShareSlippage(0.1)`, `FuturesSlippage(slippage_points=0.25)`.
- Asset-class one-way stacks (`asset_class_cost_row`): equities `PerShareCommission(0.005, min 1.0)` + `PercentageSlippage(0.0005)`; ETFs `PercentageCommission(0.0003)` + `VolumeShareSlippage(0.05)`; futures `FuturesCommission(per_block=2.25)` + `FuturesSlippage(0.25)`; crypto `PercentageCommission(0.001)` + `PercentageSlippage(0.002)`.
- Profile grids: `SHARE_PRICE = 100.0`, `SHARE_QUANTITIES = [10, 50, 100, 500, 1000, 5000, 10000]`, `FUTURES_QUANTITY = 10`, `FUTURES_PRICE = 4_000`, `FUTURES_MULTIPLIER = 50`, `SLIPPAGE_PRICE = 100`, `BAR_VOLUME = 1_000_000`, `PARTICIPATION_RATES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]`.
- Harness: `ETF_SYMBOLS = ["SPY","QQQ","IWM","XLF","EEM"]`, 2019-01-02..2023-12-29 (`N_BARS = 1260`), `INITIAL_CASH = 100_000`, `MOMENTUM_LOOKBACK = 63`; `SimpleStrategy(Strategy).on_data(timestamp, data, context, broker)` submits `make_weight_dict(step)` targets (equal-weight top-3 by 63-day momentum) via `TargetWeightExecutor` / `RebalanceConfig`; `NEXT_BAR` fills at open t+1 for a target decided at close t; `run_backtest_variant(weight_dict, commission_model, config: BacktestConfig) -> dict`; cadence grid daily / weekly / 21-session, zero vs 10 bp commission.

### Transaction cost analysis (nb 05 Part 7; book 18.7)

1. Score fills against arrival (implementation shortfall), VWAP and close with `generate_tca_report(executed_trades, benchmark_prices, params)`; `executed_fills(params, trade_list, shocks) -> (fills, benchmarks)`. Report built on the balanced schedule, one simulated path, 15-minute fills from `REPORT_START = 2021-10-01 09:30` (`REPORT_INTERVAL_MINUTES = 15`).
2. The `kind` column separates six measured rows from the one assumed input (half-spread subtracted from every fill); mark every line measurement vs assumption.
3. Pick the benchmark that matches the decision: IS for alpha strategies with decay; VWAP measures "average, reliably". Closing-price benchmark is informative only averaged over many executions.
4. Never report an impact/timing split without naming the model that produced it.
5. Feed realized shortfall back into eta: the coefficient is estimated from execution records (own fills vs decision price, Almgren et al. 2005), never from market data alone; TCA is where those records come from.

## Guardrails and pitfalls

Units, turnover and data contracts
- **Adding fees in different denominators** — per-share + per-contract + bps + spread are not commensurable and the arithmetic runs without error; store rates as text, classify each `costs:` block, fill `configured_bps` only when bps are stated.
- **Volume x price with mismatched series** — adjusted close x share volume (ETF) or tick-count volume (FX) is not dollars traded; check what each column measures; gate with `supports_usd_capacity` / `volume_contract`.
- **Comparing panels with different observation windows** — a busier period looks more liquid; keep `coverage_start` / `coverage_end` in every summary.
- **Mixing share and contract units** — a contract count mislabeled as shares; `FuturesSlippage` returns dollars, not per-unit; keep futures models separate, normalize with the multiplier.
- **Fee schedules read on one date** — schedules and account tiers change; record `SOURCE_CHECK_DATE`, status and source per row.
- **Folding borrow into a per-trade cost** — it accrues with time held, short side only; charge as an annual rate on short notional.
- **Double-counting turnover** — one-way turnover x round-trip cost already charges both legs; one-way = 0.5 sum|dw|.
- **Ignoring weight drift between rebalances** — turnover against stale weights misstates trades; compare target with pre-trade drifted weights.
- **Trade-count turnover ceilings** — cost scales with traded notional; measure daily round-trip NAV turnover.

Spread estimation
- **Trusting an OHLCV estimator where it cannot be checked** — CS and Roll can rank correctly yet be off by an order of magnitude in level; validate on quoted data, report Pearson r AND identity-line R^2 / MAE / bias separately.
- **Averaging an estimator that silently clamps to zero** — a median over a mostly-zero column reads as a narrow spread; report zero-share next to the level (CS is zero on most ETF and S&P 500 sessions).
- **Roll at daily frequency on liquid large caps** — positively autocorrelated price discovery overwhelms bid-ask bounce; check the lag-1 covariance sign; prefer CS or quotes.
- **Rolling windows crossing symbol seams** — an unguarded shift pairs one firm's return with another's; `.over("symbol")` on every shift/rolling expression.
- **Window counted in rows, not time** — 20 rows is a month on daily bars, a week on 8h crypto bars; aggregate to a common daily grid first.
- **Pre-market/overnight minutes in a benchmark or flow regression** — far wider spreads and thin volume dominate averages; restore 09:30-16:00 ET with `regular_session_mask()` before selecting symbols.
- **Discarded rows nobody counts** — the benchmark quietly becomes a different benchmark; keep an integrity ledger that sums back to the raw count; keep trade and quote panels separate.
- **Locked or crossed quotes** — bid == ask or bid > ask cannot be executed against; filter them out.
- **Reading a volatility-conditioned spread rise as widening** — CS reads the high-low range, which widens with volatility on its own; decompose rate vs level; settle with a quote benchmark (one exists here for only one quarter of one market).
- **Mega-cap spread as anchor for a broad universe** — the most liquid name understates what the median holding pays; check which symbol the anchor came from; use the median per-symbol volume-weighted spread across the index (nb 11).

Look-ahead
- **Selecting liquid symbols by full-sample volume** — end-of-window information; descriptive only; in a backtest select from information available at the time.
- **VIX quartile boundaries from full history** — look-ahead if a strategy decides on them; use expanding/trailing quantiles live.
- **Trailing windows ending on the day described** — that day's volume/range are unknown that morning; shift by one for a strategy.
- **Bar-volume look-ahead in `calculate(volume=...)`** — a bar's total volume is known only at close; pass volume accumulated so far, a lagged/forecast figure, or execute on the next bar.
- **Observing the current interval's regime** — condition on interval t-1; the open uses a training-only prior.
- **State bins or VWAP profile from all data** — leaks the test distribution; quantile bins and profiles from training sessions only; chronological split.
- **Decision/fill timing** — filling at close t on a close-t decision is look-ahead; use `NEXT_BAR` (decide close t, fill open t+1).
- **Fitting and evaluating the VWAP profile on the same sessions** — always looks good in-sample; split at a fixed calendar date so added data extends the test window.

Impact modeling
- **Treating eta (or library defaults) as measured** — stated coefficients scale every impact and capacity level; keep `eta_assumption` separate from measured sigma; calibrate from TCA when execution data exists; flag defaults as assumptions.
- **Same-minute flow/return slope read as impact** — news causes both; label it an association; use post-order paths for attribution.
- **OLS on minute returns** — a few jump minutes set the slope for a quarter; use Huber; accept negative R^2; do not switch to OLS to make R^2 look better.
- **Log-regression that drops negative slopes** — selecting on the sign of the dependent variable tilts the sample; report the excluded count; a CI containing zero makes the slope meaningless.
- **Tick-rule misclassification** — trades inside the spread get the wrong sign; prefer quote-based classification when quotes exist.
- **Participation computed against daily ADV** — volume is concentrated at open/close; compute against the interval actually traded; use `adv_factor` or intraday curves.
- **Swapping linear for sqrt without rescaling** — changes coefficient and shape at once; match at a reference participation (0.01 or the even schedule's rate).
- **Comparing model levels without matching coefficients** — sqrt includes sigma in its scale, power-law does not; a higher curve reflects conventions.
- **Misreading near-straight concave curves as convex** — any exponent in (0,1) is concave; check the exponent value.
- **Taking `PowerLawImpact.min_impact` for a fixed fee** — it is a minimum per-share move that scales with quantity; model fixed fees with `CombinedCommission(fixed=...)` or `PerShareCommission(minimum=...)`.
- **Extrapolating sqrt impact to huge participation** — fitted on far smaller orders; stop capacity curves at `MAX_FEASIBLE_PARTICIPATION = 0.20` and label it a model-validity limit.
- **Capacity assuming pro-rata spread across the universe** — a concentrated strategy hits capacity sooner; state capital, turnover, participation limit and concentration.
- **Flat persistence fraction** — real permanent impact varies by order and name; treat `PERSISTENCE_FRACTION = 0.5` as an assumption and keep it identical across models when comparing shapes.
- **Sign error on sells** — a flipped sign makes fills better than the decision price; assert sell fills < decision price before trusting totals.
- **Impact coefficients in raw price units** — a value wrong by three orders of magnitude looks right; quote temporary impact as bps at 100% interval participation, permanent as bps at 100% ADV.

Execution simulation
- **`ExecutionResult.adjusted_price` / `impact_cost` read as realized impact** — they are placeholders (zeros); take price from the `Fill` the `FillExecutor` emits.
- **Silently filling capped orders** — assumes away the constraint; carry `remaining_quantity`; the sum of fills equals the parent only on completion.
- **Executing through thin bars** — even small participation is dangerous in lunch-hour or halted markets; set a `min_volume` floor and report blocked bars.
- **Aggressive caps hide impact risk** — a higher cap speeds completion roughly proportionally but raises per-interval footprint; read the four-panel participation diagnostic, not just the benchmark fill.
- **Small test orders never exercise the cap** — size the parent as a fraction of measured ADV (0.5 in nb 07).
- **Judging a schedule on one session** — either can win by chance; run all held-out sessions; report bias separately from std, MAE, worst.
- **Reading a tight VWAP-tracking distribution as cheaper execution** — it means "average, reliably"; under an arrival benchmark slow trading is a risk; pick the benchmark that matches the decision.
- **Predictable VWAP schedules** — the slice sequence is deterministic given the profile, so other participants can anticipate and trade ahead of a large one; accept as the price of auditability or randomize/adapt (nb 08).
- **Simulation optimism** — impact charged against volume that includes the order's own shares, fills at interval VWAP, nothing unfilled; treat results as lower bounds.
- **Fictitious terminal liquidity** — force-filling at the close hides infeasible schedules; the cap applies in the last interval and the residual pays the 100 bps shortfall penalty.
- **Partial-session data in intraday grids** — missing intervals misalign price/volume/spread/vol arrays; keep only sessions with full `N_BUCKETS` coverage.
- **Price/return features in an execution policy** — the policy drifts into alpha timing; use a liquidity-only state.
- **Scoring RL by its own training objective** — the bps score is a teaching proxy with scenario kappa, lambda, eta; report execution, exposure and shortfall components separately and run sensitivity on lambda.
- **Reading RL results as deployment guarantees** — one symbol, one quarter; report paired session counts and matched-state tables as descriptive.

Almgren-Chriss and TCA
- **Continuous-time kappa approximation** — misstates the optimum when theta is not small; use the exact discrete sinh solution.
- **Lambda has no natural scale** — it converts dollars-squared to dollars, so its value depends on position size and price; a lambda quoted without X, S0 and T is unreadable; label schedules by liquidation half-life, report the sweep, never treat lambda values as universal.
- **Comparing schedules on independent draws** — the difference contains sampling noise; share shocks and use antithetic pairs (N even) so the simulated mean equals E[C].
- **Linear temporary impact in AC** — overstates the cost of trading fast, so the optimizer front-loads too little; price the schedule under the matched power law and expect the sqrt optimum to be more front-loaded.
- **Closing-price benchmark on one execution** — price wanders hundreds of bps over a multi-day horizon; informative only averaged over many executions.
- **TCA reports that split shortfall into impact and timing** — requires a counterfactual price that does not exist; label the split as a model's output and name the model.

Frequency, cost stack and cost cliff
- **In-sample cadence rankings** — fixed universe, full period, no holdout, not point-in-time membership; descriptive only; do not estimate a production-optimal cadence.
- **Cost scenarios too close to discriminate** — a 1 bp round-trip difference cannot separate net lines; make stacks differ materially before concluding.
- **Fixing the wrong cost layer** — a leveraged long-short loses more to financing/borrow than to trading; decompose attribution before choosing the lever.
- **Assuming passive/worked-order fills are available** — lower-cost stacks are sensitivity cases with fill risk; validate against an actual execution process.
- **Optimistic slippage in intraday backtests** — hides the cost cliff; apply conservative stacks; check annual cost / gross return.
- **Cost-cliff approximation scope** — excludes financing, taxes, passive-fill risk, impact uncertainty; a sensitivity, not a capacity estimate.
- **`SpreadSlippage` convention** — default treats the input as a full spread and charges half per side; passing a half-spread doubles the error; check `convention`.
- **Survivorship/selection in fixed ETF universes** (nb 06, 09, 12) — retrospective, not point-in-time; present cost mechanisms, not strategy performance.
- **Commission-sensitivity magnitudes transfer** — the Sharpe-range scalar is panel- and ticket-size-specific; re-run on the target universe.

## Decision rules and defaults

| Decision | Rule / default |
|---|---|
| Unit of record | basis points; convert once price / multiplier / quoted spread / fee tier is written down |
| Estimation windows | 20 sessions for CS, Roll, volatility and ADV (shorter tracks widening sooner, less stable) |
| Spread estimator | validate on quotes first; CS for ranking; distrust either level for backtest subtraction unless identity-line R^2 is acceptable; high zero-share = no level |
| Dominance check | solve the crossover size; below it spread-only may suffice; above it sqrt impact is required |
| Breakeven hurdle | `alpha_be = annual one-way turnover x round-trip cost / 1e4`; gross alpha must clear it plus borrow (short side); below it, do not trade at that cadence |
| Impact coefficients (stated) | nb 03: ETFs 0.10, S&P 500 0.15, NASDAQ-100 0.12, CME 0.08, crypto perps 0.30, FX 0.05; nb 01 grid 0.02-0.10; library `SquareRootImpact` 0.5 (typical 0.1-1.0) |
| Library defaults | `LinearImpact(0.1)`; `SquareRootImpact(0.5, 0.02, 1.0)`; `PowerLawImpact(0.1, 0.5, 0.0)`; `VolumeParticipationLimit(0.10, 0.0)`; `AdaptiveParticipationLimit(0.10, 0.5, 0.25, 0.02, 0.02)` |
| `adv_factor` | bars per session for intraday bars (26 for 15-min), 1 for daily |
| Capacity defaults | gross alpha 50 bps per rebalance, 30% turnover per rebalance, participation ceiling 20%, model reference participation 1% |
| Capacity go/no-go | deployable AUM = min(AUM at participation ceiling, AUM where impact = gross alpha); if the latter is outside the ceiling, the ceiling binds |
| Execution grid | 15-minute child orders (26 per US session); finer tracks the profile with more orders, coarser is easier to supervise |
| Cap ladder | 5% stealth, 10% standard institutional, 25% aggressive (urgent, very liquid) |
| Sliced-order persistence | carry ~50% of each child's impact into the next reference price unless measured otherwise |
| Schedule choice | VWAP for liquid names with stable volume shape; TWAP for thin names, event days, shape-shifting markets; POV when participation must be exact; IS/AC when the signal decays or arrival is the benchmark |
| Schedule evaluation | out-of-sample profile; report bias, std, MAE, worst over all held-out sessions |
| AC usage | pick lambda by implied half-life; sweep 10^-9..10^-5 for a $100 stock, 100k-share, 5-day, 50-period program; >= 1000 antithetic paths; read 5th/95th percentiles |
| AC parameter scale | half-spread 1 bp (NQ-100 median quoted spread ~2 bps), temporary 160 bps at 100% interval participation, permanent 25 bps at 100% ADV (~1/4 of total) |
| AC optimization | only temporary impact is controllable; report spread and permanent impact separately |
| RL execution recipe | 15-min grid, parent 10% ADV, cap 25%, kappa 10 bps, lambda 0.25, eta 100 bps, 60/40 chronological split, 10 inventory bins, actions {0.5, 0.75, 1.0, 1.3, 1.6}, 800 epochs, alpha 0.3, epsilon 0.5 -> 0; evaluate greedy policy with paired win counts |
| Round-trip cost | 2 x (half-spread + impact + commission) |
| Frequency choice | net Sharpe and break-even alpha jointly; higher costs favor lower frequency, faster decay favors higher; daily only when the signal is strong and short-lived |
| Frequency grid | daily, weekly, biweekly, monthly (nb 09); daily, weekly, 21-session (nb 12); do not extrapolate to intraday or quarterly |
| Persistence-cost ranking | prefer high AR(1) phi and low cost pressure Gamma via `S = phi/(1 - phi + Gamma)` |
| Net-of-cost waterfall defaults | 2 bps spread, 5 bps impact, 1 bp commission, 0.5 bp exchange fee, 5%/yr margin, 1%/yr borrow, 2% mgmt, 0.2% admin |
| Intraday conservative stack | the `CROSSING` preset: measured NQ-100 median half-spread + 3.0 impact + 2.0 slippage + 0.0 commission + 0.3 exchange + 0.1 clearing = half-spread + 5.4 bps one-way (x2 round-trip); the bare `IntradayCostStack` defaults sum to 7.4 / 14.8 bps but are not what nb 11 applies; `C_ann = 252 tau c_rt / 1e4`; `S_n = S_g - C_ann / sigma` |
| Intraday go/no-go | annual cost / gross return under the crossing stack approaching 1 -> reject; compare proposed NAV turnover with `calculate_break_even_turnover` at the target net Sharpe |
| Asset-class cost models | equities per-share $0.005 (min $1) + 5 bp slippage; ETFs 3 bp + `VolumeShareSlippage(0.05)`; futures $2.25/block + 0.25 pts; crypto 10 bp + 20 bp; use `VolumeShareSlippage` whenever order size relative to bar volume matters |
| Fee-type rule of thumb | percentage/tiered scale with notional; per-share scales with share count; minimum/fixed fees punish small tickets |
| First lever | slippage-heavy stack -> execution; fee-heavy -> broker/venue schedule; financing-heavy -> leverage/borrow, not turnover |

Sequencing (the chapter's own order of operations, as the notes state it):
1. Classify every cost input by unit; write down the trade detail each still needs (nb 01).
2. Measure dollar turnover only where volume and price describe the same transaction; otherwise keep native units (nb 01/03).
3. Estimate spread from OHLCV, validate on quoted data, record zero-share before trusting a level (nb 02).
4. State or calibrate the impact coefficient; compute the dominance crossover and breakeven hurdle for the intended size and turnover (nb 01/03).
5. Bound capacity at the participation ceiling and at impact = gross alpha (nb 03).
6. Choose a schedule (TWAP/VWAP/POV/AC) by volume forecastability and by the benchmark the decision implies (nb 04/05).
7. After trading, run TCA against arrival/VWAP/close and feed realized shortfall back into the coefficient (nb 05/10).

Added steps from notebooks 06-12 (this reference's synthesis, not the chapter's stated order; they sit between steps 6 and 7):
- Adaptive execution: a participation cap or RL policy conditioned on lagged liquidity regimes when a static schedule is too predictable (nb 07/08).
- Simulate fills as limit -> impact -> slippage -> commission -> carry remainder; recompute one fill by closed form (nb 06/12).
- Test cadence against break-even alpha and decay; attribute gross-to-net by layer; check the intraday cost cliff (nb 09/10/11).

Pre-publication intraday checklist (nb 11): anchor spread to the measured median per-symbol quote; state impact/slippage/fee assumptions explicitly; express activity as daily round-trip NAV turnover; compute `C_ann`, `S_n`, cost/gross ratio and break-even turnover under crossing, worked-order and passive stacks; disclose that passive fills are not established.

Measured-vs-assumed ledger for any cost report: measured = spreads, volumes, volatilities, gross return series; assumed = impact coefficients, persistence fraction, fee schedules, gross Sharpe/turnover profiles, decay rates, exposure weights.

## Code patterns and APIs

Imports (library paths `libs/src/ml4t_backtest/ml4t/backtest/execution/{impact,limits,result}.py`, `libs/src/ml4t_backtest/ml4t/backtest/models.py`):

```python
from ml4t.backtest.execution.impact import NoImpact, LinearImpact, SquareRootImpact, PowerLawImpact, MarketImpactModel
from ml4t.backtest.execution.limits import VolumeParticipationLimit, NoLimits, PositiveVolumeLimit, AdaptiveParticipationLimit
from ml4t.backtest.execution.result import ExecutionResult
from ml4t.backtest.execution.rebalancer import RebalanceConfig, TargetWeightExecutor
from ml4t.backtest import Strategy, BacktestConfig            # plus FillExecutor, Fill in the execution pipeline
from ml4t.backtest.models import (NoCommission, PercentageCommission, PerShareCommission, TieredCommission,
    CombinedCommission, FuturesCommission, NoSlippage, FixedSlippage, SpreadSlippage, PercentageSlippage,
    VolumeShareSlippage, FuturesSlippage)
```

Impact call (signed, always adverse):

```python
model = SquareRootImpact(coefficient=0.5, volatility=sigma_daily, adv_factor=1.0)
move = model.calculate(quantity=q, price=p, volume=adv, is_buy=True)   # signed $/share
fill_price = p + move
```

Limit + impact composition with carried remainder (nb 06 Part 6 pattern; `calculate` takes `price` as a required third argument):

```python
limit = VolumeParticipationLimit(max_participation=0.10, min_volume=0.0)
res = limit.calculate(order_quantity=abs(q), bar_volume=v, price=price)   # nb 06: .calculate(abs(quantity), volume, decision_price)
move = model.calculate(quantity=res.fillable_quantity, price=price, volume=v, is_buy=is_buy)
fill = price + move
carry = res.remaining_quantity        # queue to the next bar; never drop it
```

Almgren-Chriss coefficient conversion and exact trajectory:

```python
gamma_price = gamma_bps * S0 / (ADV * 10_000)      # permanent, bps at 100% ADV
eta_price   = eta_bps   * S0 / (ADV * 10_000)      # temporary, bps at 100% interval participation
eps_price   = eps_bps   * S0 / 10_000              # half-spread
theta = lam * sigma_P**2 * tau**2 / eta_price
a = 2 * np.arcsinh(np.sqrt(theta) / 2)
x_k = X * np.sinh(a * (N - k)) / np.sinh(a * N)
```

Antithetic simulation: draw N/2 shock paths, append their negatives, reuse the same array for every schedule.

Polars panel discipline: `pl.col(...).shift(1).over("symbol")`; rolling windows `.over("symbol")`; sort before `.last()` for an explicit daily close. Settings pattern: module-level UPPERCASE constants with a "what each setting decides" block; `MAX_SYMBOLS = 0` means all.

Shared helper module `18_transaction_costs/_cost_analysis.py` (also imported by case-study `costs.py`; verified from source, not from the notes). Imported by the notebooks: nb 01 `breakeven_alpha(turnover, cost_bps)` (annual one-way turnover x round-trip bps / 1e4 -> decimal); nb 02 `corwin_schultz_spread(high, low, window=1)` and `roll_spread(close, window=20)` (Polars Series/Expr in, spread as a fraction not bps, negatives clamped to 0); nb 03 `compute_adv(volume, window=20)` and `estimate_kyle_lambda(price_changes, signed_volume) -> {lambda_, r_squared, std_err, n_obs}` (Huber with intercept, needs >= 30 non-zero-flow rows). Not called by the chapter notebooks: `compute_adv_usd(volume, close, window=20)`; `calibrate_sqrt_impact(returns, volume, sigma, adv, min_adv=1e3) -> {eta, r_squared, std_err, n_obs}` (no-intercept OLS of `|r|/sigma` on `sqrt(V/ADV)` over market data, i.e. a same-interval association, not the own-fill calibration the chapter requires for eta); `estimate_capacity(adv_usd, impact_coeff, gross_alpha_bps, turnover=1.0, max_participation=0.01) -> {max_aum_usd, breakeven_participation, impact_at_max_bps}` (solves `gross_alpha_bps = impact_coeff * 1e4 * sqrt(participation)`, caps at `max_participation`, `AUM = participation * ADV / turnover`); `get_fee_schedule(asset_class)` over `FEE_SCHEDULES` keys `us_equities`, `etfs`, `crypto_perps`, `cme_futures`, `fx_spot`, `sp500_options`; `standardized_cost_per_100k(asset_class, price=50.0) -> {commission_bps, exchange_bps, total_bps}`.

Chapter helper functions by notebook (all in `18_transaction_costs/`):
- nb 01: `limit_symbols`, `summarize_dollar_turnover`, `fee_evidence_row`, `cost_components`, `dominant_component_label`, `cost_schema_row`, `CASE_STUDIES`.
- nb 02: `load_nq_microstructure`, `estimate_spreads`, `compute_validation_metrics`, `keep_top_symbols`, `aggregate_crypto_daily`, `classify_cost_units`.
- nb 03: `keep_top_symbols`, `add_impact_features`, `regular_session_mask`, `session_integrity_ledger`, `load_nq_signed_flow_sample`, `estimate_normalized_lambda`, `capacity_curve`, `regime_impact_row`, `ETA_SCENARIO`.
- nb 04: `build_time_grid`, `build_twap_schedule`, `latest_cumulative_target`, `TWAPAlgorithm`, `VWAPAlgorithm`, `session_grid`, `load_intraday_panel`, `day_paths`, `normalize_volume_curve`, `build_vwap_schedule`, `shares_from_profile`, `execute_day`, `market_vwap_of`, `run_across_days`, `summarize_tracking`.
- nb 05: `AlmgrenChrissParams` (properties `tau`, `sigma_daily`, `sigma_price_daily`, `gamma_price`, `eta_price`, `epsilon_price`), `compute_trajectory_from_list`, `expected_cost`, `execution_variance`, `execution_std`, `optimal_trajectory`, `trajectory_to_trades`, `liquidation_half_life`, `compute_efficient_frontier`, `simulate_single_execution`, `simulate_execution`, `ExtendedParams`, `expected_cost_nonlinear`, `generate_tca_report`, `executed_fills`.
- nb 06: `compose_limit_and_impact`. nb 07: `load_intraday_panel`, `execution_frame`, `participation_walk`, `add_execution_traces`.
- nb 08: `interval_cost_bps`, `capped_fill`, `schedule_components`, `schedule_score_bps`, `observable_regime`, `encode_state`, `run_episode(sess, q_table, epsilon, learn)`, `train_q_policy(train_days, sessions)`, `shares_from_weights`, `execute_static_schedule`, `evaluate(test_days, sessions, q_table) -> pl.DataFrame`, `participation_diagnostics`, `matched_volatility_policy(q_table)`, `session_arrays`.
- nb 09: `CostAssumptions` (properties `total_one_way`, `round_trip`), `HIGH/MEDIUM/LOW_FRICTION_COSTS`, `FREQUENCIES`, `momentum_frequency_backtest`, `annualized_sharpe`, `calculate_break_even_alpha`, `real_net_by_frequency`, `simulate_frequency_comparison`, `evaluate_decay_scenario`.
- nb 10: `CostStack` methods, `build_strategy`, `apply_cost_stack`, `compute_performance`.
- nb 11: `measure_nasdaq100_spreads`, `MEASURED_HALF_SPREAD_BPS`, `IntradayCostStack` (properties `one_way_bps`, `round_trip_bps`), presets `CROSSING` / `WORKED_ORDER` / `PASSIVE_LOW_COST`, `IntradayStrategy` profiles `HIGH/MODERATE/LOW_TURNOVER`, `calculate_intraday_net_performance`, `calculate_break_even_turnover`.
- nb 12: `asset_class_cost_row`, `SimpleStrategy`, `make_weight_dict`, `run_backtest_variant`.

Config keys: `config/setup.yaml -> costs:` (keys vary by case study; the classifier reads what is present). Panel contract columns: `supports_usd_capacity`, `volume_contract`, `coverage_start`, `coverage_end`, `configured_bps`. Plot helpers from `utils.style`: `COLORS`, `add_message_title`, `ml4t_palette(n, categorical=True)`, `show_with_alt`, `show_plotly_with_alt`.

Running: `uv run python 18_transaction_costs/<notebook>.py`; test mode `uv run pytest tests/test_chapter_notebooks.py -v -k "18_transaction_costs"`; Docker image `ml4t`. Reading order by prerequisite: 03 -> 06 -> 07 -> 08; 01 -> 09 -> 10 -> 11 -> 12 (12 also needs 06). Notebooks write no side-effect files.

## Evidence from the book

- NASDAQ-100 (12 liquid names, 2021Q4): median quoted spread ~2 bps; half (~1 bp) is the half-spread used in the AC example.
- Corwin-Schultz returns exactly zero on most sessions for the ETF and S&P 500 panels (clamp, not level). Roll degrades at daily frequency on liquid large caps because price discovery swamps bid-ask bounce. Estimators rank reasonably; level accuracy is the weak point (exact r / R^2 / MAE / bias not in the notes).
- VIX conditioning: the rise in mean CS spread with VIX is entirely in the level (wider estimate when one exists), not in how often an estimate appears.
- Minute-level signed-flow Huber regressions (~100 NQ-100 symbols, 2021Q4): median R^2 near zero; the coefficient is useless as a bar-level predictor and usable at most as an average-cost input; some per-symbol slopes are negative and excluded from the log-log cross-section (count reported in the notebook; slope and CI not in the notes).
- Square-root curvature: doubling order size raises cost per dollar ~40%; the first slice of participation costs proportionally most.
- Intraday volume (AlgoSeek AAPL): U-shaped; relative spread widest at the open; realized vol highest at the open; day-to-day dispersion around the mean profile is wider than any smooth parametric curve, which is why VWAP tracking is imperfect.
- VWAP vs TWAP on held-out 2021Q4 sessions (100k shares, 15-min grid, 5 bps impact at full participation): slippage-vs-VWAP distributions differ in width, not centre; VWAP front-/back-loads and holds less through midday; narrower = more predictable, not cheaper.
- Almgren-Chriss (100k shares, $100, 30% vol, 5 days, 50 periods, ADV 1M): spread and permanent impact constant across schedules; only temporary impact and risk move; frontier nearly flat at the patient end, steep at the urgent end; simulated mean equals analytical E[C]; the 5th-95th band contains negative shortfalls; linear and sqrt matched at the even schedule agree almost exactly, concentrated schedules are charged less by sqrt, so a linear optimizer front-loads too little; the closing-price benchmark wanders by hundreds of bps on one path.
- nb 06: the power-law curve sits above the sqrt curve despite identical shape (no sigma factor); doubling sigma doubles sqrt impact everywhere; all exponents in (0,1) are concave, and at a fixed coefficient smaller exponents bend more and raise the level; linear and sqrt sliced paths overlay when matched at the common child participation and fills climb only through the 50% carried persistence; two of five ETF orders exceed the 10% cap and fill partially; the closed-form check passes; cross-model table differences reflect conventions.
- nb 07: a half-ADV AAPL parent completes in fewer intervals/sessions as the cap rises 5% -> 10% -> 25%, roughly proportionally; the cost of a higher cap is impact risk, not a worse benchmark fill; below `min_volume` the permitted fill is zero and both policies agree once volume recovers.
- nb 08: Q-table of a few hundred states; paired held-out counts show how often the Q-policy scores below TWAP and VWAP; matched-state table shows actions change with lagged volatility regime (numbers not in the notes; descriptive).
- nb 09 (8 ETFs, 2019-2023): gross Sharpe already slopes down toward daily cadence and the cost gap widens toward daily, so the effects reinforce; high- and medium-friction stacks differ by 1 bp round-trip and give nearly identical panels; under hypothetical inputs short-horizon momentum has the highest raw IC but ranks last on the persistence-cost proxy, value and quality move up.
- nb 10: the leveraged QQQ-IWM long-short loses more Sharpe to margin and borrow than to trading frictions; high-turnover long-only QQQ loses most to per-trade costs; net Sharpe declines linearly with turnover at each cost level; a rotating-covariance min-variance portfolio shows nonzero maintenance turnover.
- nb 11: the measured median NASDAQ-100 half-spread (Dec 2021) anchors the crossing stack, which supplies the largest drag; the high-turnover profile can fall below the scenario net-Sharpe threshold under crossing; lower-cost stacks expand the turnover ceiling enough to need a log axis.
- nb 12: percentage fees are constant in bps; minimum/fixed fees consume a larger share of small tickets; `VolumeShareSlippage` is the only participation-responsive curve (crosses the 10 bps percentage line); monthly momentum Sharpe range across commission models is small and panel-specific; dollar fees and the fee-induced Sharpe gap widen together along the daily cadence path.
- Case-study configs: almost none of the nine `costs:` blocks record a round-trip spread in bps.

## Related references

- `chapters/03_market_microstructure.md` — spreads, quotes, tick rule, order flow; the microstructure regime link (18.3) builds on it.
- `chapters/16_strategy_simulation.md` — the backtest engine these cost, limit and slippage models plug into; `NEXT_BAR` fill timing.
- `chapters/17_portfolio_construction.md` — turnover generated by rebalancing and covariance drift; cost-aware construction.
- `chapters/19_risk_management.md` — `ml4t.backtest.risk` rules, drawdown/daily-loss limits and kill switches; its position that costs must be priced before any exit, sizing or rebalancing claim.
- `chapters/20_strategy_synthesis.md` — cross-case-study cost survival (`20_strategy_synthesis/06_cost_survival`).
- `chapters/21_rl_execution_hedging.md` — deep RL execution agents (DQN / PPO / A2C with stable-baselines3) benchmarked against TWAP and Almgren-Chriss on identical paths with paired standard errors; it lists this chapter (implementation shortfall, impact, spread) as a prerequisite.
- `chapters/25_live_trading.md` — the live runtime where realized fills are produced: one `Strategy` through `ml4t.backtest.Engine` and `ml4t.live.LiveEngine`, parity assertions, order-lifecycle state machine, `SafeBroker` controls, shadow mode.
- `chapters/06_strategy_definition.md` — the strategy as an executable decision process: decision schedule, signal vs holding horizon, cost class and capacity as feasibility constraints, trading-intensity diagnostics (turnover vs cost class).
- `case_studies/nasdaq100_microstructure.md` — AlgoSeek minute bars with quotes and signed flow used for validation, lambda and execution.
- `case_studies/etfs.md` — ETF panels used for impact demos, frequency trade-off, cost stack and commission comparison.
- `case_studies/cme_futures.md` — contract multiplier (tick value / tick size) and futures commission/slippage units.
- `case_studies/crypto_perps_funding.md` — 8h funding-aligned bars that must be aggregated to UTC days.
- `case_studies/fx_pairs.md` — tick-count volume with no dollar axis.
- `case_studies/sp500_options.md`, `case_studies/sp500_equity_option_analytics.md` — option cost literature cited by the chapter.
- `libraries/ml4t_backtest.md` — `ml4t.backtest.execution` and `ml4t.backtest.models` reference.
- `libraries/ml4t_live.md` — the live trading runtime: IB / Alpaca broker adapters, `SafeBroker` pre-trade risk checks, shadow -> paper -> live promotion for the same `Strategy` the backtest runs.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `workflow.md`, `companion_repo.md` — cross-cutting indexes.
- Further reading:
  - Almgren & Chriss (2001), Optimal execution of portfolio transactions; Almgren, Thum, Hauptmann & Li (2005), Direct estimation of equity market impact, Risk 18(7).
  - Corwin & Schultz (JF 67(2)) high-low spread estimator; Roll (JF 39(4)) serial-covariance estimator; Kyle (1985); Hasbrouck (1991); Amihud (2002).
  - Toth et al. (2011) and Sato & Kanazawa (2024) on the square-root law; Obizhaeva & Wang (2013); Cont et al. (2014), Eisler et al. (2010), Hautsch & Huang (2012) order-book impact; Chan (2022) impact decay and capacity.
  - Frazzini, Israel & Moskowitz (2018), Trading costs; Gabaix & Koijen (2021) and Bouchaud (2022) inelastic markets; Chordia et al. (2000) liquidity commonality; Karnaukh et al. (2015) FX liquidity.
  - Avellaneda & Stoikov (2008); Ho & Stoll (1981); Nevmyvaka, Feng & Kearns (2006) RL for optimized trade execution; Donnelly (2022) review.
  - Muravyev & Pearson (2020), O'Donovan & Yu (2024), Heston et al. (2023) option costs; Schwarz et al. (2022) retail execution; Madhavan (2002); Paleologo (2025) alpha-to-go; Said (2022); Taranto et al. (2018).

## Glossary

- Basis point — 0.01%; the unit of nearly every cost here.
- ADV — average daily volume in the market's native unit (shares, contracts, base asset, price updates for FX); `adv_factor` converts bar volume to ADV.
- Participation rate — order size / volume available in the interval traded; the input every impact model and limit consumes.
- Explicit cost — commission and venue fees. Implicit cost — half-spread and impact. Capacity — AUM at which own impact consumes the gross return.
- Half-spread — what an immediately executing order gives up, ~half the quoted gap.
- Quoted / effective / realized spread — at the touch / where the trade printed vs midpoint / what the liquidity provider kept after the price moved.
- Locked market — bid == ask; crossed market — bid > ask; neither executable.
- Bid-ask bounce — alternation of prints between bid and ask inducing negative return autocorrelation (Roll's premise).
- Corwin-Schultz — high-low-range spread estimator; Roll — serial-covariance spread estimator.
- Identity-line R^2 — fit against the 45-degree line; negative when the estimate is worse than the benchmark mean.
- Square-root law — impact proportional to sigma * sqrt(Q/ADV).
- Kyle's lambda — slope of price change on signed order flow; here a price-normalized per-symbol Huber slope.
- Tick rule — buyer-initiated if a trade prints above the previous trade; signed order flow = uptick volume - downtick volume.
- Temporary impact — liquidity concession that decays; permanent impact — information revealed, persists; persistence fraction carries the permanent part across child orders.
- Adverse selection — counterparty knows something you do not; price keeps moving after the fill.
- Signed per-share price move — impact model output; positive buys, negative sells; added once to the decision price.
- Child / parent order — slices of a large order worked over time.
- TWAP / VWAP — equal-shares-per-interval / shares proportional to expected volume; also the benchmarks.
- Percent of volume — trade a fixed fraction of printed volume; completion time floats.
- Implementation shortfall — execution cost against the arrival (decision) price. Arrival price — price at the decision moment.
- Timing risk — exposure of the unexecuted position to price moves.
- Efficient frontier (execution) — schedules not dominated in both expected cost and variance.
- Liquidation half-life — elapsed time at which half the position is done; the unit-free label for lambda.
- Risk aversion lambda — weight on variance in `E[C] + lambda V[C]`; unit-dependent.
- Antithetic paths — mirror-image shock pairs so the sample mean shock is exactly zero.
- `ExecutionResult` — limit output (fillable, remaining, placeholder adjusted_price/impact_cost, participation_rate). `Fill` — what `FillExecutor` emits; its price includes impact and slippage.
- `min_volume` gate — floor below which no execution occurs in a bar.
- Finite-horizon MDP — execution as sequential control with state (time, inventory, lagged regimes); tabular Q-learning — inspectable Q(s,a) table with alpha and epsilon-greedy exploration.
- Exposure proxy — lambda x (avg inventory/Q) x sigma_t, linear bps penalty for carrying inventory. Terminal shortfall — eta x unfilled fraction at the close.
- TCA — transaction cost analysis: scoring realized fills against benchmarks after the fact.
- Breakeven alpha — gross return needed to pay trading cost at a given turnover. One-way turnover — 0.5 sum|dw|; round-trip cost = 2 x one-way cost.
- Signal decay — exponential loss of captured alpha with rebalance delay. Persistence-cost score — phi/(1 - phi + Gamma), teaching proxy for alpha-to-go.
- Cost stack — ordered deductions: spread, impact, commission/fees, financing, fund expenses.
- NAV turnover — traded notional in units of NAV (round-trip = opened and closed). Cost cliff — turnover level where execution-cost drag consumes gross return.
- Crossing / worked order / passive — paying the full half-spread, a fraction, or earning spread (with fill risk).
- Vector-L2 turnover — Euclidean norm of the weight-change vector; diagnostic only.
- Tiered commission — rate schedule by notional thresholds. `VolumeShareSlippage` — slippage rising with order share of bar volume via `impact_factor`.
- Borrow cost — annual fee on short-side notional, a running cost.
- Back-adjusted futures — stitched continuous series scaled so rolls do not appear as jumps.
