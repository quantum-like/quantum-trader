# ml4t-backtest library (event-driven backtesting with realistic execution)

> `ml4t-backtest` (import root `ml4t.backtest`, v0.1.12, MIT, Stefan Jansen) is the strategy-simulation stage of the ML4T workflow (Chapter 16): a point-in-time event loop that turns model predictions (from ml4t-models / ml4t-engineer) into orders, fills, trades and an equity curve under configurable execution, accounting and risk rules, and hands trades/returns to ml4t-diagnostic for evaluation. Its headline promise is **framework parity**: "40+ behavioral knobs" reproduce VectorBT, Backtrader, Zipline and LEAN conventions exactly (17/17 real-strategy pairs match) while a `realistic` preset adds costs and a cash buffer. The same `Strategy` class runs unchanged in `ml4t-live`. Docs: https://www.ml4trading.io/docs/backtest/ ; repo: https://github.com/ml4t/backtest.

## Install and import

- `uv add ml4t-backtest` (Python >=3.12; 3.12/3.13/3.14 tested). Hard deps: `ml4t-specs>=0.1.4,<0.2`, `numpy>=2.3.2`, `pandas>=2.3.3`, `polars>=1.36.1`, `pandas-market-calendars>=5.4.0`, `pyyaml>=6.0.3`.
- Extras: `viz` (dash, matplotlib, plotly); `advanced` (cvxpy, networkx: graph routing + portfolio optimization); `comparison` (backtrader, pyfolio-reloaded, vectorbt, zipline-reloaded; py<3.15; excludes VectorBT Pro and the containerized LEAN engine); `dev`, `docs`, `all`.
- Public imports: `from ml4t.backtest import Engine, run_backtest, Strategy, Broker, DataFeed, BacktestConfig, BacktestResult, CommissionType, OrderType, OrderSide, OrderStatus, ExecutionMode, StopFillMode, StopLevelBasis, ExitReason, Order, Position, Fill, Trade, AssetClass, ContractSpec, RebalanceConfig, TargetWeightExecutor, RebalanceSchedule, RebalanceCadence, is_rebalance_timestamp, resolve_rebalance_timestamps, StopLoss, TakeProfit, TrailingStop, RuleChain, LifecycleDispatcher, callback_trace, default_execution_policy, Artifact*Error`.
- Sub-modules: `ml4t.backtest.config` (`ExecutionPrice`, `SlippageType`, `SpreadConvention`, `ShareType`, `FillOrdering`, ...), `.accounting` (`UnifiedAccountPolicy`, `AccountState`, `Gatekeeper`), `.models` (commission/slippage models), `.execution.{impact,limits,rebalancer,schedule}`, `.risk.position.*`, `.risk.portfolio.{limits,manager}`, `.analytics`, `.calendar`, `.sessions`, `.export`, `.strategies` (templates), `.example_data` (`load_example_prices(category)`, `ExampleRoundTrip(asset, quantity)`).
- `_validation/*` and `_validation_imports.py` are internal (Backtrader/VectorBT/Zipline/LEAN runners, Chapter 16 LEAN harness); do not import them in user code.

## API map by task

### Running a backtest (`engine.py`)
| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `Engine` | `Engine(feed, strategy, config=None, *, contract_specs=None, market_impact_model=None, execution_limits=None, funding_df=None, lifecycle_version=LifecycleVersion.V1, execution_policy=None, target_intent_state=None)` | Orchestrates feed -> strategy -> broker per bar | Single-use: build a new `Engine` per run (`_run_once` guard) |
| `Engine.run()` | `-> BacktestResult` | Execute the loop | `run_dict()` returns the legacy dict; `Engine.from_config(feed, strategy, config, **kw)` |
| `run_backtest` | `run_backtest(prices: pl.DataFrame \| str, strategy, signals=None, context=None, config: BacktestConfig \| str \| None = None, *, feed_spec=None, contract=None, ...same kwargs) -> BacktestResult` | One-call convenience | String paths are read as parquet |

### Data feed (`datafeed.py`)
| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `DataFeed` | `DataFeed(prices_path=None, signals_path=None, context_path=None, prices_df=None, signals_df=None, context_df=None, *, feed_spec=None, contract=None, entity_col=None, timestamp_col=None, price_col=None, open_col=..., high_col, low_col, close_col, volume_col, vwap_col, bid_col, ask_col, mid_col, bid_size_col, ask_size_col, session_col=None)` | Polars long-format multi-asset feed | O(1) timestamp -> slice index; `contract=` is an alias of `feed_spec` |
| iteration | `for timestamp, data, context in feed` | Yields `(datetime, {asset: bar dict}, context dict)` | `len(feed)` / `feed.n_bars` = unique timestamps; `feed.timestamps` |

### Strategy contract (`strategy.py`, `strategies/templates.py`)
| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `Strategy` | ABC: `on_start(broker)`, `on_prepare(broker, config=None)`, `on_data(timestamp, data, context, broker)`, `on_end(broker)` | User logic | `on_start`: broker configured, no bar yet (set position rules here); `on_prepare`: config only, no future feed data; `on_end`: no automatic flattening |
| `SignalFollowingStrategy` | class attrs `signal_column="prediction"`, `position_size=0.05`; override `should_enter_long(signal)`, `should_exit(signal)`, `should_enter_short(signal)` | Threshold on a model prediction | `_use_fractional(allow_fractional, broker)` resolves strategy vs broker share type |
| `MomentumStrategy` / `MeanReversionStrategy` | `calculate_momentum(prices) -> float`; `calculate_zscore(prices, current) -> float \| None` | Classic templates | Return over lookback / z-score |
| `LongShortStrategy` | `rank_assets(data) -> (longs, shorts)`; periodic rebalance in `on_data` | Cross-sectional long/short ML strategy | `on_prepare` retains calendar metadata; `_validate_completed_run` |

### Configuration and presets (`config.py`, `profiles.py`)
| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `BacktestConfig` | dataclass; fields in "Defaults" below | All behavioral knobs | `validate(warn=True) -> list[str]`; `_validate_for_execution()` raises; `describe()` |
| `BacktestConfig.from_preset` | `from_preset(name)` | Parity/realism presets | `vectorbt`, `backtrader`, `zipline`, `lean`, `realistic`, `ibkr_us_stocks_fixed`, `fast`, `default`; strict: `vectorbt_strict`, `vectorbt_oss_strict`, `vectorbt_futures_strict`, `backtrader_strict`, `zipline_strict`; aliases `vectorbt_pro`, `vectorbt_oss`, `quantconnect`->`lean`, `*_compare`->strict |
| `from_yaml` / `to_yaml` / `from_dict(data, preset_name=None, strict=True)` / `to_dict()` | serialization | Reproducible configs | `strict=True` rejects unknown keys (inference) |
| `from_assumptions` | `from_assumptions(*, broker, region, asset_class, plan, overrides=None, **config_overrides)` | Broker assumptions -> preset | Only `broker='ibkr', region='us', asset_class='stocks', plan='fixed'` supported today; else `ValueError` |
| `from_user_config` | `from_user_config(*, config_dir=None, broker=None, region=None, asset_class=None, plan=None, **overrides)` | Layered user defaults | `_resolve_user_config_dir` |
| resolvers | `resolved_feed_spec()`, `resolved_calendar()`, `resolved_timezone()` (UTC fallback), `resolved_data_frequency()`, `resolved_session_start_time()`, `resolved_timestamp_semantics()`, `merge_feed_spec(feed_spec)` | Runtime metadata | `merge_feed_spec` fills missing config from feed metadata without mutating user config |
| `get_effective_account_settings()` / `get_effective_account_type()` | `-> (allow_short_selling, allow_leverage)` / `-> str` | Account type | Cash / Crypto / Margin |
| `profiles` | `list_profiles()`, `get_profile_config(name) -> dict` (deep copy), `get_assumption_preset(*, broker, region, asset_class, plan) -> str` | Inspect preset knob tables | |
| `StatsConfig` | `StatsConfig(recent_window_size=50, track_session_stats=True, enabled=True)` | Per-asset trading stats | `broker.configure_stats(...)` |

### Broker: orders (`broker.py` -> `core/order_book.py`)
| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `submit_order` | `submit_order(asset, quantity, side=None, order_type=OrderType.MARKET, limit_price=None, stop_price=None, trail_amount=None) -> Order \| None` | Core order entry | Negative quantity or explicit `side`; `None` on rejection -> `last_rejection_reason()` |
| `submit_bracket` | `submit_bracket(asset, quantity, take_profit, stop_loss, entry_type=MARKET, entry_limit=None, validate_prices=True) -> (entry, tp, sl) \| None` | Entry + TP + SL | |
| `buy` / `sell` | `buy(asset, shares=None, contracts=None, amount=None, dollars=None, order_type=MARKET, limit_price=None)` | Sized entry | Exactly one quantity spec |
| `order_target_percent` / `order_target_value` / `rebalance_to_weights` | `(asset, target_percent, order_type=MARKET, limit_price=None)`; `rebalance_to_weights(target_weights, order_type=MARKET) -> list[Order]` | Zipline-style targets | |
| `close_position` / `reduce_position` / `flatten_all_positions` / `reduce_all_positions` | `close_position(asset, order_type=MARKET)`; `reduce_position(asset, fraction, ...)`; `flatten_all_positions(reason, order_type=MARKET) -> list[Order]` | Exits | `flatten_all_positions` cancels pending first |
| `update_order` / `cancel_order` / `get_order` / `get_pending_orders(asset=None)` | `update_order(order_id, **OrderUpdate) -> bool` | Order management | `get_pending_orders` returns a copy |
| target intents | `register_target_intent(intent: CanonicalTargetIntent, *, position_rules=None)`, `register_position_rule_policy(policy_id, rules)`, `get_target_intents()`, `get_child_order_intents()`, `get_intent_reconciliations()`, `get_target_rule_reconciliations()`, `export_target_intent_state()` / `restore_target_intent_state(state)` | Causal pre-open target phase (lifecycle V2-style) | Intents are idempotent |

### Broker: state, prices, rules
| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| positions/cash | `get_position(asset) -> Position \| None`, `get_positions()`, `get_cash()`, `cash`, `equity()`, `get_account_value()`, `get_buying_power()` | Read state | `equity()` = cash + positions marked at `mark_price` source |
| journals | `get_trades(asset=None, last_n=None)`, `get_last_trade(asset=None)`, `fills`, `trades`, `orders`, `pending_orders`, `positions` | Append-only lifecycle collections | Replacement raises |
| rejections | `get_rejected_orders(asset=None)`, `last_rejection_reason() -> str \| None`, `Order.rejection_code()` | Diagnose rejected orders | Stable codes, e.g. `insufficient_cash` |
| prices | `get_price_for_source(source: ExecutionPrice, asset, *, side=None, quantity=None, use_open=False)`, `get_mark_price(asset, *, quantity=None, use_open=False)`, `get_last_price(asset)`, `get_quote_mid(asset)`, `get_available_size(asset, side=None)`, `get_quote_context(asset, side=None)`, `mark_account_positions(use_open=False)` | Quote-aware price resolution | "Sensible OHLCV fallbacks"; `get_last_price` returns only positive prices |
| contracts | `get_contract_spec(asset) -> ContractSpec \| None`, `get_multiplier(asset)` | Futures multipliers | 1.0 for equities |
| position rules | `set_position_rules(rules, asset=None)`, `clear_position_rules(asset=None)`, `update_position_context(asset, context)`, `evaluate_position_rules() -> list[Order]` | Stop/TP/trail evaluation | Global or per-asset; context feeds ATR/signal rules |
| stats/sessions | `configure_stats(...)`, `get_asset_stats(asset) -> AssetTradingStats`, `set_session_config(SessionConfig \| None)` | Live-style stats | `AssetTradingStats.win_rate`, `recent_win_rate`, `recent_expectancy`, `session_win_rate` |
| `Broker.from_config` | `from_config(config, execution_limits=None, market_impact_model=None, contract_specs=None)` | Build from config | `Engine` uses this, so `BacktestConfig` defaults govern (not `Broker.__init__` defaults `SAME_BAR`/`CLOSE`) |

### Accounting (`accounting/*`)
| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `AccountState` | `AccountState(initial_cash, policy)`: `total_equity()` (NLV), `buying_power()`, `allows_short_selling()`, `mark_to_market(prices)`, `get_position_quantity(asset)`, `unsettled_cash()`, `add_settlement_hold(bar_index, delay, amount)`, `release_settled(current_bar)` | Cash + positions owner | |
| `UnifiedAccountPolicy` | `UnifiedAccountPolicy(allow_short_selling=False, allow_leverage=False, initial_margin=0.5, long_maintenance_margin=0.25, short_maintenance_margin=0.30, fixed_margin_schedule=None, margin_pct_schedule=None, short_cash_policy="credit")`; `from_config(config)`; `get_margin_requirement(asset, qty, price, for_initial=True, *, multiplier=1.0)`; `is_margin_call(cash, positions)` | Cash / crypto / margin accounts | Cash=(False,False), Crypto=(True,False), Margin=(True,True) |
| `AccountPolicy` (ABC) | `calculate_buying_power`, `get_spendable_cash`, `validate_new_position(...) -> (bool, str)`, `validate_position_change(...)`, `handle_reversal(...)` | Custom policies | |
| `Gatekeeper` | `Gatekeeper(account, commission_model, cash_buffer_pct=0.0, settlement_reduces_buying_power=True, multiplier_resolver=None)`: `validate_order(order, price) -> (ok, reason)`, `validate_order_with_code(...) -> (ok, reason, code)`, `classify_rejection(resulting_quantity) -> str` | Pre-fill validation | Invalid trades never enter the journal |

### Execution cost models (`models.py`, `execution/impact.py`, `execution/limits.py`)
| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| commission models | `NoCommission()`, `PercentageCommission(rate=0.0)`, `PerShareCommission(per_share=0.0, minimum=0.0)`, `TieredCommission(tiers=[(threshold, rate), ...])`, `CombinedCommission(percentage=0.0, fixed=0.0)`, `FuturesCommission(per_trade=0.0, per_block=0.0, percentage=0.0)` | `CommissionModel.calculate(asset, qty, price) -> float` | Tiered: first tier whose `threshold >= trade value` applies; `FuturesCommission.calculate(..., multiplier=1.0)` is "PySystemTrade-compatible" |
| slippage models | `NoSlippage()`, `FixedSlippage(amount=0.0)`, `PercentageSlippage(rate=0.0)`, `SpreadSlippage(spread=0.0, asset_spreads=None, convention="full_spread")`, `VolumeShareSlippage(impact_factor=0.0)`, `FuturesSlippage(slippage_points=0.0)` | `SlippageModel.calculate(asset, qty, price, volume) -> float` (per-unit price adjustment) | `@runtime_checkable`; helpers `calculate_commission`, `estimate_commission` (no state advance), `calculate_slippage` |
| market impact | `NoImpact()`, `LinearImpact(coefficient=0.1)`, `SquareRootImpact(coefficient=0.5, volatility=0.02, adv_factor=1.0)`, `PowerLawImpact(coefficient=0.1, exponent=0.5, min_impact=0.0)` | `calculate(quantity, price, volume, is_buy) -> float` price shift | Linear: `coef*(Q/V)*price`; SquareRoot (Almgren-Chriss): `coef*sigma*sqrt(Q/ADV)*price`; PowerLaw: `coef*(Q/V)^exp*price`; temporary impact only; sells negative |
| execution limits | `NoLimits()`, `PositiveVolumeLimit()`, `VolumeParticipationLimit(max_participation=0.10)`, `AdaptiveParticipationLimit(base_participation=0.10, volatility_factor=0.5, max_participation=0.25, min_participation=0.02)` | `calculate(order_quantity, bar_volume, price) -> ExecutionResult` (`is_partial()`, `is_full()`) | Pass as `Engine(execution_limits=...)`; zero/None volume = no fill under `PositiveVolumeLimit` |

### Rebalancing and schedules (`execution/rebalancer.py`, `execution/schedule.py`)
| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `RebalanceConfig` | dataclass (15 fields, see Defaults) | Weight -> order policy | `__post_init__` coerces `share_rounding` |
| `TargetWeightExecutor` | `TargetWeightExecutor(config=None)`: `execute(target_weights, data, broker, *, timestamp=None, is_session_close=None) -> list[Order]`, `preview(...) -> list[dict]`, `should_rebalance(timestamp, *, is_session_close=None) -> bool`, `validate_completed_run()`, `reset()` | Target weights -> sequential cash-aware orders | Prices new trades at `data[asset]["close"]`; call `should_rebalance` for every event when a schedule is set |
| `WeightProvider` | Protocol `get_weights(data, broker) -> dict[str, float]` | Plug in riskfolio-lib / PyPortfolioOpt / cvxpy | May return negative or levered weights |
| `RebalanceSchedule` | `every_bar()`, `every_session()`, `fixed_n_sessions(n)`, `weekly()`, `month_end()`, `explicit_timestamps(ts)` | Cadence | `RebalanceCadence` = {every_bar, every_session, fixed_n_sessions, weekly, month_end, explicit_timestamps} |
| `is_rebalance_timestamp` | `(timestamp, schedule, *, session_index, calendar=None, timezone=None, session_start_time=None, data_frequency=None, timestamp_semantics=None, is_session_close=None) -> bool` | Causal online check | No future feed timestamps |
| `resolve_rebalance_timestamps` | `(available_timestamps, schedule, *, feed_spec=None, calendar=None, ...) -> pl.Series` | Offline batch resolution | |

### Position-level risk (`risk/position/*`, `risk/types.py`)
| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| static | `StopLoss(pct)`, `TakeProfit(pct)`, `TimeExit(max_bars)` | Exit rules | Reason `time_exit_{max_bars}bars` |
| dynamic | `TrailingStop(pct)`, `TighteningTrailingStop(schedule=[(ret, trail_pct), ...])`, `ScaledExit(targets=[(ret, fraction), ...])` (+`reset()`), `VolatilityStop(multiplier=2.0, atr_key="atr", use_entry_atr=True)` (+`reset()`), `VolatilityTrailingStop(multiplier=3.0, atr_key="atr")` | Path-dependent exits | ATR read from `broker.update_position_context(asset, {"atr": ...})` |
| signal | `SignalExit(signal_name="exit_signal", threshold=0.0)` | Model-driven exit | Long exits on `signal < -threshold`, short on `signal > threshold` |
| composite | `RuleChain(rules)`, `AllOf(rules)`, `AnyOf(rules)` | Combine rules | |
| protocol | `PositionRule.evaluate(state: PositionState) -> PositionAction`; `PositionAction.hold() / exit_full(reason, fill_price, defer_fill) / exit_partial(pct, ...) / adjust_stop(price, reason)` | Custom rules | `ActionType` {HOLD, EXIT_FULL, EXIT_PARTIAL, ADJUST_STOP} (inference from constructors) |

### Portfolio-level risk (`risk/portfolio/*`)
| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `RiskManager` | `RiskManager(limits=[])`: `initialize(initial_equity, timestamp=None)`, `update(equity, positions: dict[str, float], timestamp=None, context=None, broker=None) -> list[LimitResult]`, `can_open_position()`, `can_increase_position(asset, amount) -> (bool, str)`, `is_halted`, `halt_reason`, `warnings`, `current_drawdown`, `reset_halt()`, `get_state(...)` | Apply warn/reduce/halt/liquidate via broker | Reduction/liquidation applied once each |
| limits | `MaxDrawdownLimit`, `MaxPositionsLimit`, `MaxExposureLimit`, `DailyLossLimit`, `GrossExposureLimit`, `NetExposureLimit`, `VaRLimit`, `CVaRLimit`, `BetaLimit`, `SectorExposureLimit`, `FactorExposureLimit` | `PortfolioLimit.check(state) -> LimitResult` | Defaults below; `LimitResult.ok()/warn()/reduce(reason, pct)/halt()/liquidate()` |

### Analytics, results, export (`analytics/*`, `result.py`, `export.py`)
| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `BacktestResult` | `metrics` dict (`total_return_pct`, `sharpe`, ...); `to_trades_dataframe()`, `to_fills_dataframe()`, `to_funding_dataframe()`, `to_rejected_orders_dataframe()`, `to_predictions_dataframe()`, `to_equity_dataframe()`, `to_portfolio_state_dataframe()`, `to_daily_pnl(session_aligned=False)`, `to_daily_returns(calendar=None, session_aligned=None) -> pl.Series`, `to_returns_series()`, `to_trade_records() -> list[dict]`, `to_dict()`, `to_spec_dict()`, `to_parquet(path, include=None, compression="zstd") -> dict[str, Path]`, `from_parquet(path, *, recovery=False)` | Run output and handoff to ml4t-diagnostic | `result["missing"]` raises `KeyError`; use `result.get(key, default)` |
| `enrich_trades_with_signals` | `(trades_df, signals_df, signal_columns=None, timestamp_col="timestamp", asset_col=None, trades_asset_col=None) -> pl.DataFrame` | As-of join of signals at entry/exit | Aligns time zones first |
| `BacktestExporter` | `to_parquet(result, path, compression="zstd", include=None)`, `batch_export(results, base_path, param_values: list[dict], compression="zstd", export_individual=True) -> pl.DataFrame`, `from_parquet(path, *, recovery=False)`, `load_sweep_summary(base_path)`, `generate_json_report(results, output_path, metadata=None)`, `generate_markdown_report(results, output_path, title="Backtest Results", param_names=None, param_values=None)` | Sweeps and CI reports | Docstring shows `param_names=` for `batch_export` but the signature lacks it; pass `param_values` only |
| `EquityCurve` | `from_config(config)`, `append(ts, value)`, `returns()`, `cumulative_returns()`, `total_return`, `years`, `periods_per_year`, `max_drawdown_info() -> (dd, peak_idx, trough_idx)`, `max_dd`, `cagr`, `volatility`, `sharpe`, `sortino`, `drawdown_series()`, `to_dict()` | Equity metrics | Configured cadence preferred over elapsed-time inference |
| `metrics.py` | `returns_from_values`, `volatility(returns, annualize=True)`, `sharpe_ratio(returns, risk_free_rate=0.0, annualize=True)`, `sortino_ratio`, `max_drawdown(values)`, `cagr(initial, final, years)`, `calmar_ratio(cagr, max_dd)` | Functional metrics | |
| `TradeAnalyzer` | `num_trades`, `win_rate`, `profit_factor`, `expectancy`, `payoff_ratio`, `avg_bars_held`, `total_fees`, `total_slippage`, `total_costs`, `avg_cost_drag`, `gross_profit_factor`, `by_side('long'\|'short')`, `by_symbol`, `avg_mfe`, `avg_mae`, `mfe_capture_ratio`, `mae_recovery_ratio`, `to_dict()` | Trade statistics | `expectancy = win_rate*avg_win + (1-win_rate)*avg_loss` |
| `MAEMFEAnalyzer` | `mae_mean/median/std/percentile(q)`, `mfe_*`, `edge_ratio`, `efficiency`, `suggest_stop_loss(percentile=90)`, `suggest_take_profit(percentile=75)`, `optimal_exit_levels(stop_percentile=90, target_percentile=75)`, `distribution_data()` | Excursion-based exit design | `edge_ratio` = avg MFE / avg \|MAE\| |
| `analytics/bridge.py` | `to_trade_record(trade)`, `to_trade_records(trades)`, `to_returns_series(equity_curve)`, `to_equity_dataframe(equity_history, timestamps=None)` | ml4t-diagnostic adapters | |
| `annualization.py` | `get_annualization_factor(calendar) -> int`, `resolve_periods_per_year(data_frequency, *, calendar)`, `should_session_align(*, calendar, feed_spec=None, timestamps=None)` | Periods/year from calendar | |

### Calendar, sessions, funding, lifecycle, pre-open
| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `calendar.py` | `get_schedule(calendar_id, start, end, *, include_breaks=False, include_extended_hours=False) -> pl.DataFrame`, `get_trading_days(...)`, `is_trading_day(id, date)`, `is_market_open(id, dt)`, `next_trading_day(id, from_date, n=1)`, `previous_trading_day`, `get_holidays`, `get_early_closes`, `get_calendar_sessions(id, year)`, `filter_to_trading_days(df, id, timestamp_col="timestamp")`, `filter_to_trading_sessions(df, id, timestamp_col="timestamp", *, naive_tz="UTC", include_breaks=False)`, `generate_trading_minutes(id, start, end, *, freq="1m", include_close=True)`, `list_calendars()`, `get_standard_market_open_time(id)` | Polars wrapper over pandas_market_calendars | Ids: `"XNYS"`, `"NYSE"`, `"CME_Equity"`, `"CME_Bond"`, `"LSE"` (product-specific CME calendars with maintenance breaks) |
| `sessions.py` | `SessionConfig(calendar, timezone, session_start_time)`; `session_date_for_timestamp(ts, *, calendar, timezone, session_start_time, data_frequency, timestamp_semantics) -> date`; `compute_session_pnl(equity_curve, session_config) -> pl.DataFrame`; `assign_session_date(ts, tz, hour, minute=0)`; `align_to_sessions(df, session_config, timestamp_col="timestamp")` | Session-aligned P&L (futures) | CME: `calendar="CME_Equity", timezone="America/Chicago", session_start_time="17:00"` |
| `funding.py` | `FundingEvent`, `FundingPayment`, `index_funding_events(frame, feed_timestamps, feed_assets) -> dict[datetime, list[FundingEvent]]` | Perpetual-swap funding | Pass `funding_df=` to `Engine`; applied before fills at each timestamp |
| `lifecycle.py` | `LifecycleDispatcher(strategy, contract=LIFECYCLE_V1, *, retain_invocations=False)`: `dispatch(phase, broker, *args, event_time=None)`, `callback_counts()`, `validate_completed_run(market_event_count)`; `callback_trace(invocations)` | Versioned, rollback-on-raise callbacks | Trace used for cross-engine parity |
| `preopen.py` | `default_execution_policy(config) -> ExecutionPolicy`; `PreOpenTargetManager(broker, policy, lifecycle_version, *, account, market, calendar, timezone=None, session_start_time=None, data_frequency=None, timestamp_semantics=None)`: `register(intent, *, position_rules=None, active_phase=None)`, `register_position_rule_policy(policy_id, rules)`, `process_opening(timestamp)`, `reconcile(timestamp, *, target_intent_id=None)`, `targets/children/reconciliations/target_rule_reconciliations`, `to_state()/restore_state()`, `capture_transaction_state()/restore_transaction_state()` | Causal pre-open target lowering | Errors: `LateAuctionIntentError`, `UnsupportedPreOpenPolicyError`, `AmbiguousBarPathError`, `PreOpenIntentError` |
| `spec_bridge.py` | `market_data_spec_to_feed_spec(spec) -> FeedSpec`, `market_data_spec_to_runtime_metadata(spec) -> dict` | ml4t-data `MarketDataSpec` -> runtime | |

## Data contracts

- **Price frame** (Polars, long format): one row per `timestamp` x entity. Entity column auto-detected from `("symbol", "asset", "product", "ticker")` in that order or set by `entity_col=`. `close` required; optional `open/high/low/volume/vwap/bid/ask/mid/bid_size/ask_size/session`. Remap names via `*_col` args or a `FeedSpec` from ml4t-specs; `bar["price"]` follows `FeedSpec.price_col`, so read `bar.get("price", bar.get("close"))`.
- **Signals frame**: same keys plus arbitrary signal columns (e.g. `prediction`); exposed as `data[asset]["signals"][col]`. **Context frame**: keyed by timestamp only (portfolio-wide, e.g. regime flags, `historical_returns`).
- **Per-bar payload**: `data[asset]` = bar fields + `"signals": {...}`; only the current bar is materialized (`_slice_for_timestamp` zero-copy slice from a `timestamp -> (offset, length)` index). The feed iterates the sorted union of timestamps across prices/signals/context.
- **Timestamps/timezones**: naive datetimes are interpreted in `BacktestConfig.timezone` (default `"UTC"`). `session_col` triggers decision-session indexing (validates complete daily cross sections, records each session's final close) for daily rebalancing on intraday data. Schedule helpers normalize naive vs aware event times; `BacktestResult._timestamp_dtype` keeps the recorded zone; `enrich_trades_with_signals` aligns zones before joining.
- **Enums (`config.py`)**: `ExecutionPrice` {price, close, open, vwap (needs volume), mid ((high+low)/2), bid, ask, quote_mid, quote_side (buy at ask / sell at bid)}; `ShareType` {fractional, integer}; `ShareRounding` {nearest, truncate}; `FillOrdering` {exit_first, fifo, sequential, priority}; `EntryOrderPriority` {submission, notional_desc, notional_asc, free_cash_asc, order_value_asc}; `ShortCashPolicy` {credit, credit_proceeds, lock_notional}; `LockNotionalUpdateMode` {position_legs, combined_order}; `RebalanceMode` {snapshot, incremental, hybrid}; `MissingPricePolicy` {skip, use_last}; `LateAssetPolicy` {allow, require_history}; `CommissionType` {none, percentage, per_share, per_contract (=per_share), per_trade, tiered}; `SlippageType` {none, percentage, fixed, spread, volume_based}; `SpreadConvention` {full_spread, half_spread}; `DataFrequency` {daily, 1m, 5m, 15m, 30m, 1h, irregular}; `WaterMarkSource` {close, bar_extreme}; `InitialHwmSource` {fill_price, signal_price, bar_close, bar_high}; `TrailStopTiming` {lagged, intrabar, vbt_pro}.
- **Enums (`types.py`)**: `OrderType` {MARKET, MOC (always priced at close), LIMIT, STOP, STOP_LIMIT, TRAILING_STOP}; `OrderSide` {BUY, SELL}; `OrderStatus` {PENDING, FILLED, CANCELLED, REJECTED}; `ExecutionMode` {same_bar (fill at current close), next_bar (fill at next open, Backtrader-like)}; `StopFillMode` {STOP_PRICE (VBT Pro OHLC / Backtrader), CLOSE_PRICE (VBT Pro close-only), BAR_EXTREME, NEXT_BAR_OPEN (Zipline)}; `StopLevelBasis` {FILL_PRICE, SIGNAL_PRICE (Backtrader)}; `AssetClass` {EQUITY (mult 1), FUTURE, FOREX}; `ExitReason` {signal, stop_loss, take_profit, trailing_stop, time_stop, risk_liquidation, end_of_data}.
- **Core records**: `Order(order_id, asset, side, quantity, order_type, limit_price, stop_price, trail_amount, child_intent_id)` + `reject(reason, code)`; `Position.market_value`, `unrealized_pnl(current_price=None)` (multiplier-aware), `pnl_percent`, `notional_value`, `update_water_marks(...)`, `side`; `Trade.direction`, `is_open`, `gross_pnl = (exit-entry)*qty*multiplier`, `net_pnl`, `gross_return`, `net_return`, `total_slippage_cost`, `cost_drag = (fees+slippage)/notional`; `PartialExit.is_win`; `ContractSpec` gives futures multipliers via `contract_specs: dict[str, ContractSpec]`.
- **`PositionState`** (rule input): `asset, side, entry_price, current_price, quantity, initial_quantity, unrealized_pnl, unrealized_return, bars_held, high_water_mark, low_water_mark, bar_open/bar_high/bar_low=None, max_favorable_excursion, max_adverse_excursion, entry_time, current_time, context={}, multiplier=1.0`; properties `is_long`, `is_short`, `is_profitable`, `drawdown_from_peak`. Context keys read by rules: `StopFillMode`, `TrailStopTiming`, `atr_key`, `signal_name`.
- **`PortfolioState`** (limit input): `equity, initial_equity, high_water_mark, current_drawdown (0..1), num_positions, positions: {asset: market value}, daily_pnl, gross_exposure, net_exposure, timestamp, context`. Context keys: `historical_returns` (VaR/CVaR), `asset_betas` (BetaLimit), `asset_sectors` (SectorExposureLimit), `<factor_key>` (FactorExposureLimit).
- **State owners (`core/state.py`)**: `MarketState` (bar caches, `last_prices`, `asset_bars_seen`), `OrderState` (history, pending, `partial_quantities`, `orders_this_bar`, `filled_this_bar`), `RiskState` (rules, `pending_exits`, `stop_exits_this_bar`, `positions_created_this_bar`), `ExecutionJournal` (append-only fills/trades); `AccountState` owns cash/positions.
- **Result artifact** (`to_parquet(dir)`): components `trades`, `fills`, `funding`, `rejected_orders`, `equity`, `portfolio_state`, `metrics` (JSON; non-finite floats tagged and restored), `config.yaml` (replayable resolved config), `spec.yaml` (runtime snapshot with library version and realized run window), plus a versioned manifest. `from_parquet` validates the manifest; `recovery=True` loads what it can and emits `ArtifactDiagnostic`s; `discover_legacy_components()` handles pre-manifest dirs. `config.metadata` dict carries provenance (input paths, strategy ids); `preset_name` records origin.
- **Pre-open intent model**: `CanonicalTargetIntent` -> `CanonicalChildOrderIntent`s -> `IntentReconciliation` (fill, remainder, rejection, rule activation) and `TargetRuleReconciliation`; `RoundingPolicy` governs quantity rounding; residual-cash policy validated by `_validate_residual_policy`. `LIFECYCLE_V1`: `on_start`/`on_prepare`/`on_end` fire exactly once; per-event callbacks counted against `market_event_count`.

## Built-in guardrails

1. **Point-in-time loop.** Strategies see only the current bar in `on_data`; `on_prepare` gets config but no future timestamps; `Engine._validate_strategy_lifecycle` rejects non-conforming strategies. Never pre-compute signals from the full feed inside a strategy.
2. **Single-use `Engine`, append-only journals.** `run()` is guarded by `_run_once`; assigning to `fills/trades/orders/pending_orders/positions` raises (`_reject_lifecycle_collection_replacement`). Create a new `Engine` per run; call `TargetWeightExecutor.reset()` before reusing an executor.
3. **Completion validation.** `_validate_completed_run(market_event_count)` (engine, lifecycle dispatcher, schedule evaluator, strategy) raises after the loop if a collaborator missed events or boundary callbacks did not fire exactly once (purpose partly inference).
4. **Exit-first fill ordering** (default). Orders that reduce without reversing (`is_exit_order`) fill before entries so same-bar exit proceeds fund entries. Alternatives: `fifo`, `sequential`, `priority` (sorted by `EntryOrderPriority`, VectorBT-style call sequencing via `_estimated_free_cash_use`).
5. **Gatekeeper at fill price.** Every order is validated against the `AccountPolicy` for cash, buying power, shorting permission, leverage and reversals; `submit_order` returns `None`, with `last_rejection_reason()` / `get_rejected_orders()` / stable codes (`insufficient_cash`). Always check for `None` and inspect `to_rejected_orders_dataframe()` after a run.
6. **Account-type constraints.** Cash accounts forbid shorts and leverage; margin uses Reg T (`initial_margin=0.5`, `long_maintenance_margin=0.25`, `short_maintenance_margin=0.30`) with `is_margin_call`; `short_cash_policy` decides whether short proceeds are spendable (`credit`), credited (`credit_proceeds`) or locked (`lock_notional`, VectorBT parity; cover rejected with `insufficient_cash` when it exceeds free cash).
7. **Settlement and cash buffer.** `settlement_delay` bars hold sale proceeds (`add_settlement_hold`/`release_settled`); `settlement_reduces_buying_power=True` subtracts unsettled cash; `available = spendable * (1 - cash_buffer_pct)`.
8. **Share rounding and affordability.** `ShareType.INTEGER` truncates; target sizing uses `share_rounding`; `get_max_affordable_quantity` bisects (integer shares, or 64 iterations fractional) then subtracts `CASH_TOLERANCE / fill_price` (`CASH_TOLERANCE = 0.01`; 0 under non-levered `lock_notional`). Partial fills only with `partial_fills_allowed=True`, otherwise rejection (`reject_on_insufficient_cash=True`).
9. **Order-type fill realism** (`FillEngine.check_fill`): LIMIT fills only if BUY `low <= limit` / SELL `high >= limit`; STOP triggers on BUY `high >= stop` / SELL `low <= stop` and fills at the **open** when the bar gaps through, else at the stop; TRAILING_STOP ratchets `high - trail_amount` (sell) / `low + trail_amount` (buy) one way only. No fill at a price the bar never traded.
10. **Risk-exit slippage.** Orders carrying `_risk_fill_price` fill at that price times `(1 - stop_slippage_rate)` for sells / `(1 + rate)` for buys, on top of normal slippage.
11. **Intrabar path ambiguity.** `RiskEngine._evaluate_path` evaluates rules under both high-first and low-first paths and `_more_conservative_path` takes the worse; `AmbiguousBarPathError` refuses post-open rule resolution when order is unknown. `TrailStopTiming.LAGGED` (default) uses the previous bar's water mark so a stop cannot be triggered by the extreme that set it.
12. **Stop rules honor gaps and fill mode.** `StopLoss`/`TakeProfit`/`TrailingStop` check `bar_open` first (long stop: `bar_open <= stop` -> fill at open), then low/high; fill price follows `StopFillMode` (default `STOP_PRICE`). `VolatilityStop`/`VolatilityTrailingStop` hold when `context[atr_key]` is None or <= 0; `VaRLimit`/`CVaRLimit` skip when `len(returns) < lookback_days`.
13. **Price sanity.** `_validate_execution_price` / `_validate_market_impact` reject non-positive prices; `MissingPricePolicy.SKIP` (default) skips assets with no current price during rebalancing; `LateAssetPolicy.REQUIRE_HISTORY` + `late_asset_min_bars` blocks assets that just started trading.
14. **Opt-in submission prechecks** for parity: `next_bar_submission_precheck` (Backtrader pseudo-exec), `buying_power_reservation` (LEAN), `next_bar_queue_shadow_validation`, `next_bar_simple_cash_check`, `_passes_margin_submission_precheck`. Leave off unless reproducing another engine.
15. **Config validation.** `validate(warn=True)` lists edge cases; `_validate_for_execution()` raises on invalid combinations; margin schedules validated; `from_dict(strict=True)` rejects unknown keys (inference). Run `config.validate()` and read warnings before trusting a result.
16. **Funding before fills.** Validated `FundingEvent`s for a timestamp apply before any fill at that time; `index_funding_events` rejects events not mapping to feed timestamps/assets.
17. **Sessions/calendar.** `enforce_sessions=True` with `calendar` skips bars outside sessions; pre-filter with `filter_to_trading_sessions` / `filter_to_trading_days`.
18. **Schedules are causal and strict.** `execute()` raises `ValueError("timestamp is required when RebalanceConfig.schedule is set")`; `should_rebalance` must see every event; schedule/calendar fields are immutable once observed; alignment errors name the nearest observed event (`_raise_explicit_alignment_error`), require a session close for weekly/month-end (`_raise_close_alignment_error`), flag skipped period ends, ambiguous/mixed/short implicit-daily intervals, and calendar-dependent cadences without a calendar (`_validate_schedule_calendar`).
19. **Rebalancer caps and filters.** `sum(abs(w)) > max_gross_leverage + 1e-6` rescales all weights; `target_wt = min(target_wt, max_single_weight)`; negative targets are set to 0 when `allow_short=False`; skip if `abs(weight_delta) < min_weight_change`, price None/<=0, `abs(equity*delta) < min_trade_value`, quantized shares == 0; `equity <= 0` returns `[]`. Reductions run first, equity is re-read after each step, all orders share a `rebalance_id` with `priority_notional`. `cancel_before_rebalance=True` avoids pending-order double counting (else `account_for_pending`).
20. **Non-negative cost models.** `_validate_nonnegative_model_output` rejects negative commission/slippage; `estimate_commission` previews tiered models without advancing state.
21. **Portfolio-limit validation.** Action must be one of `none|warn|reduce|halt|liquidate`; `reduce` only on `MaxDrawdownLimit` with finite `reduction_pct` in (0, 1]; `RiskManager.update` raises `"broker is required to apply action='reduce'"` / `"broker has no positions to reduce"`; each reduction/liquidation applied once.
22. **Lifecycle rollback.** `LifecycleDispatcher` wraps callbacks in `_BrokerTransaction`; a raising callback restores captured broker state.
23. **No automatic end-of-data flattening.** Open positions stay marked; they appear as `Trade.is_open` with `ExitReason.END_OF_DATA`. Flatten in `on_end` (`broker.flatten_all_positions("end_of_data")`) if realized P&L is wanted.
24. **Pre-open causality.** `register()` accepts targets without reading prices; `LateAuctionIntentError` rejects post-cutoff registrations; `UnsupportedPreOpenPolicyError` fires before any order; targets are idempotent across restarts (`to_state`/`restore_state`); `_cost_adjusted_buy_quantity` fits notional + entry costs into the allocation; `_allocate_largest_remainders` respects `cash_buffer`.
25. **Artifact integrity.** `from_parquet` raises `ArtifactNotFoundError`, `ArtifactManifestError`, `ArtifactIncompleteError`, `ArtifactReadError`, `UnsupportedArtifactVersionError`; `to_parquet` raises `ArtifactWriteError`; `recovery=True` degrades with diagnostics.
26. **Floating-point hygiene.** `add_with_zero_cancellation` collapses tolerance-equivalent opposites; `quantity_zero_tolerance` = max(absolute floor, 16 ULP); PnL computed in ledger operation order for cent-exact parity.
27. **Documented non-coverage (LIMITATIONS.md).** Corporate actions, borrow costs, taxes and currency conversion are NOT modeled; bar data cannot order intrabar events; the strategy lifecycle contract is pre-stable and shared with `ml4t-live`. Adjust prices upstream (ml4t-data) and treat results on dividend-paying or hard-to-borrow names as optimistic.
28. **Parity is a regression gate, not a realism claim.** Parity runs disable costs and position rules on both sides; a backtest that matches VectorBT is not therefore tradable. Switch to `realistic` (or broker-specific costs) before drawing conclusions.

## Usage patterns

Minimal signal strategy (README):
```python
from ml4t.backtest import Engine, Strategy, BacktestConfig, DataFeed
class SignalStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        for asset, bar in data.items():
            signal = bar.get("signals", {}).get("prediction", 0)
            price = bar.get("price", bar.get("close", 0))
            position = broker.get_position(asset)
            if position is None and signal > 0.5:
                broker.submit_order(asset, (broker.get_account_value() * 0.10) / price)
            elif position is not None and signal < -0.5:
                broker.close_position(asset)
feed = DataFeed(prices_df=prices, signals_df=signals)   # cols: timestamp, asset, close | prediction
result = Engine(feed, SignalStrategy(), BacktestConfig(initial_cash=100_000)).run()
print(result.metrics["total_return_pct"], result.metrics["sharpe"]); result.to_fills_dataframe().head()
```

Presets, execution modes, costs:
```python
config = BacktestConfig.from_preset("realistic")   # 20 bps commission + 20 bps slippage, 2% cash buffer
config = BacktestConfig.from_preset("vectorbt")    # same-bar close fills, fractional shares, no costs
from ml4t.backtest import ExecutionMode, StopFillMode, CommissionType
from ml4t.backtest.config import SlippageType, SpreadConvention
config = BacktestConfig(execution_mode=ExecutionMode.NEXT_BAR, stop_fill_mode=StopFillMode.STOP_PRICE,
                        commission_rate=0.001, slippage_rate=0.0005, stop_slippage_rate=0.001)  # 10 / 5 / 10 bps
ib = BacktestConfig(commission_type=CommissionType.PER_SHARE, commission_per_share=0.005, commission_minimum=1.0)
spread = BacktestConfig(slippage_type=SlippageType.SPREAD, slippage_spread=0.02,
                        slippage_spread_convention=SpreadConvention.FULL_SPREAD)
for w in config.validate(): print("WARN", w)
```

Position rules and portfolio limits:
```python
from ml4t.backtest import Strategy, StopLoss, TakeProfit, TrailingStop, RuleChain
from ml4t.backtest.risk.position.static import TimeExit
from ml4t.backtest.risk.portfolio.limits import MaxDrawdownLimit, DailyLossLimit
from ml4t.backtest.risk.portfolio.manager import RiskManager
class MyStrategy(Strategy):
    def on_start(self, broker):
        broker.set_position_rules(RuleChain([StopLoss(pct=0.05), TakeProfit(pct=0.15),
                                             TrailingStop(pct=0.03), TimeExit(max_bars=20)]))
        self.rm = RiskManager([MaxDrawdownLimit(max_drawdown=0.2, action="liquidate"), DailyLossLimit()])
        self.rm.initialize(broker.get_account_value())
    def on_data(self, timestamp, data, context, broker):
        self.rm.update(broker.get_account_value(), {a: p.market_value for a, p in broker.get_positions().items()},
                       timestamp, broker=broker)
        if self.rm.is_halted or not self.rm.can_open_position(): return
```

Target-weight rebalancing with a causal schedule:
```python
from ml4t.backtest import TargetWeightExecutor, RebalanceConfig, RebalanceSchedule
cfg = RebalanceConfig(schedule=RebalanceSchedule.month_end(), calendar="NYSE", timezone="America/New_York",
                      min_trade_value=100, min_weight_change=0.01, max_gross_leverage=1.0)
executor = TargetWeightExecutor(cfg)
# in on_data: executor.execute returns [] on non-rebalance events; pass timestamp (and is_session_close on intraday feeds)
weights = {a: b["signals"]["prediction"] for a, b in data.items() if b.get("signals", {}).get("prediction", 0) > 0}
total = sum(weights.values()); orders = executor.execute({a: w / total for a, w in weights.items()}, data, broker, timestamp=timestamp)
# after the run: executor.validate_completed_run(); executor.reset() before reuse
```

Quote-aware execution (microstructure data):
```python
from ml4t.backtest.config import ExecutionPrice
feed = DataFeed(prices_df=quotes, price_col="mid_price", bid_col="bid", ask_col="ask",
                bid_size_col="bid_size", ask_size_col="ask_size")
config = BacktestConfig(execution_price=ExecutionPrice.QUOTE_SIDE, mark_price=ExecutionPrice.QUOTE_SIDE)
```

Reproducible artifact and handoff to ml4t-diagnostic:
```python
config = BacktestConfig.from_yaml("config/my_backtest.yaml")
result = Engine(feed, strategy, config).run()
paths = result.to_parquet("results/run_001")          # trades/fills/equity/... + config.yaml + spec.yaml + manifest
daily = result.to_daily_returns(calendar="NYSE")      # pl.Series for ml4t-diagnostic
records = result.to_trade_records()                   # TradeRecord dicts
rejected = result.to_rejected_orders_dataframe()
```

Parameter sweep export: `summary = BacktestExporter.batch_export(results=results, base_path="./sweep_results", param_values=params)`; `summary.sort("sharpe", descending=True).head(5)`. Session-aligned futures P&L: `compute_session_pnl(equity_curve, SessionConfig(calendar="CME_Equity", timezone="America/Chicago", session_start_time="17:00"))`. Calendar: `get_schedule("XNYS", date(2024,1,1), date(2024,12,31))`, `is_trading_day("XNYS", date(2024,7,4))`. Margin account: `UnifiedAccountPolicy(allow_short_selling=True, allow_leverage=True)` or `BacktestConfig(allow_short_selling=True, allow_leverage=True)`.

Typical pipeline: ml4t-data prices (point-in-time, adjusted) + ml4t-models predictions -> long-format `prices_df` / `signals_df` -> `DataFeed` -> `Strategy` (template or custom) + `BacktestConfig.from_preset("realistic")` + position rules -> `Engine.run()` -> `config.validate()` warnings, `to_rejected_orders_dataframe()` -> `result.to_parquet()` -> `to_daily_returns()` / `to_trade_records()` into ml4t-diagnostic -> same `Strategy` into ml4t-live.

## Defaults and configuration keys

`BacktestConfig` field defaults:
| Group | Keys and defaults |
|---|---|
| Account | `allow_short_selling=False`, `allow_leverage=False`, `initial_margin=0.5`, `long_maintenance_margin=0.25`, `short_maintenance_margin=0.30`, `fixed_margin_schedule=None` ({asset: (initial, maintenance)}), `margin_pct_schedule=None`, `short_cash_policy=CREDIT`, `lock_notional_update_mode=POSITION_LEGS` |
| Execution | `execution_price=OPEN`, `mark_price=PRICE`, `execution_mode=NEXT_BAR`, `stop_fill_mode=STOP_PRICE`, `stop_level_basis=FILL_PRICE`, `trail_hwm_source=CLOSE`, `trail_include_entry_bar_extremes=False`, `initial_hwm_source=FILL_PRICE`, `trail_stop_timing=LAGGED`, `immediate_fill=False` |
| Sizing | `share_type=INTEGER`, `share_rounding=NEAREST` |
| Costs | `commission_type=NONE`, `commission_rate=0.0`, `commission_per_share=0.0`, `commission_per_trade=0.0`, `commission_minimum=0.0`, `slippage_type=NONE`, `slippage_rate=0.0`, `slippage_fixed=0.0`, `slippage_spread=0.0`, `slippage_spread_by_asset={}`, `slippage_spread_convention=FULL_SPREAD`, `stop_slippage_rate=0.0` |
| Cash | `initial_cash=100000.0`, `cash_buffer_pct=0.0`, `settlement_delay=0` (T+0), `settlement_reduces_buying_power=True`, `reject_on_insufficient_cash=True`, `skip_cash_validation=False` (True = Zipline-like unconstrained fills), `partial_fills_allowed=False` |
| Sequencing | `fill_ordering=EXIT_FIRST`, `entry_order_priority=SUBMISSION`, `next_bar_submission_precheck=False`, `next_bar_simple_cash_check=False`, `buying_power_reservation=False`, `next_bar_queue_shadow_validation=False` |
| Rebalancing | `rebalance_mode=INCREMENTAL`, `rebalance_headroom_pct=1.0`, `missing_price_policy=SKIP`, `late_asset_policy=ALLOW`, `late_asset_min_bars=1` |
| Time | `calendar=None`, `timezone="UTC"`, `data_frequency=DAILY`, `enforce_sessions=False` |
| Provenance | `preset_name=None`, `feed_spec=None`, `metadata={}`, `retain_intent_history=False`, `retain_lifecycle_history=False` |

Note: `Broker.__init__` defaults differ (`execution_mode=SAME_BAR`, `execution_price=CLOSE`) and the `ExecutionMode` enum docstring calls `SAME_BAR` the default, but `Engine` builds the broker via `Broker.from_config`, so NEXT_BAR/OPEN govern normal use.

Preset knobs (`profiles.py`):
| Preset | Fill | Shares | Stop fill / level basis | Trail timing / HWM source | Costs | Ordering and account |
|---|---|---|---|---|---|---|
| `default` | next_bar @ open | integer | stop_price / fill_price | lagged / close | none | exit_first; no short, no leverage |
| `backtrader` | next_bar @ open | integer | stop_price / **signal_price** | lagged / close, `initial_hwm_source=signal_price` | none | fifo; short yes, leverage no |
| `vectorbt` (`vectorbt_pro`, `vectorbt_oss`) | **same_bar @ close** | **fractional** | stop_price / fill_price | **intrabar / bar_extreme**, `initial_hwm_source=bar_high` | none | exit_first, `immediate_fill=True`, `partial_fills_allowed=True`; short yes |
| `zipline` | next_bar @ open | integer | **next_bar_open** / fill_price | intrabar / bar_extreme | none | fifo, `skip_cash_validation=True`; short yes |
| `lean` (`quantconnect`) | next_bar @ open (next session) | integer | next_bar_open / fill_price | lagged / close | per_share 0.005, min 1.0, no slippage | sequential, `next_bar_submission_precheck=True`, `buying_power_reservation=False`; short yes, **leverage yes** |
| `realistic` | next_bar @ open | integer | stop_price / fill_price | lagged / close | **pct commission 0.002, pct slippage 0.002, `stop_slippage_rate=0.001`, `cash_buffer_pct=0.02`** | exit_first; no short, no leverage |
| `ibkr_us_stocks_fixed` | next_bar @ open | integer | stop_price / fill_price | lagged / close | per_share 0.005, min 1.0, no slippage | exit_first; cash account |
| `fast` | same_bar @ close | integer | stop_price / fill_price | lagged / close | none | exit_first; short yes |
| strict variants | `vectorbt_strict` = vectorbt + `short_cash_policy=lock_notional`, `fill_ordering=priority`, `entry_order_priority=free_cash_asc`, `immediate_fill=False`; `vectorbt_oss_strict` = + `order_value_asc`; `vectorbt_futures_strict` = + `immediate_fill=True`; `backtrader_strict` = + `next_bar_submission_precheck=True`; `zipline_strict` = zipline copy | | | | | |

`RebalanceConfig`: `min_trade_value=0.0`, `min_weight_change=0.0`, `allow_fractional=None` (defer to `broker.share_type`), `round_lots=False`, `lot_size=100` (`round(shares/lot_size)*lot_size`), `allow_short=False`, `max_single_weight=1.0`, `max_gross_leverage=None` (tolerance 1e-6; set e.g. 5.0 for a CTA margin book), `cancel_before_rebalance=True`, `account_for_pending=True`, `share_rounding=NEAREST` (half away from zero; TRUNCATE = `int(x)`), `schedule=None`, `calendar=None`, `timezone=None`, `session_start_time=None`.

Portfolio limits: `MaxDrawdownLimit(max_drawdown=0.20, action="liquidate", warn_threshold=None, reduction_pct=0.0)` (breach `current_drawdown >= max`); `MaxPositionsLimit(max_positions=10, action="halt")` (`num_positions >= max`); `MaxExposureLimit(max_exposure_pct=0.10, action="warn")` (per-asset abs value / equity, inference); `DailyLossLimit(max_daily_loss_pct=0.02, action="liquidate")` (only when `equity > 0`); `GrossExposureLimit(max_gross_exposure=1.0, action="halt")`; `NetExposureLimit(max_net_exposure=1.0, min_net_exposure=-1.0, action="warn")`; `VaRLimit(threshold=0.05, confidence_level=0.95, lookback_days=20, action="warn", returns_key="historical_returns")`; `CVaRLimit(threshold=0.08, confidence_level=0.95, lookback_days=20, action "warn"|"halt", default not captured, likely "warn" (inference))`; `BetaLimit(max_beta=1.5, min_beta=-0.5, betas_key="asset_betas")`; `SectorExposureLimit(max_sector_exposure=0.30, sectors_key="asset_sectors")`; `FactorExposureLimit(factor_name, max_exposure, min_exposure, factor_key)`. `LimitResult(action="none", reason="", reduction_pct=0.0)`.

Position rules: `VolatilityStop(multiplier=2.0, atr_key="atr", use_entry_atr=True)`; `VolatilityTrailingStop(multiplier=3.0, atr_key="atr")`; `SignalExit(signal_name="exit_signal", threshold=0.0)`; `PositionAction(pct=1.0, defer_fill=False)`; context defaults `StopFillMode.STOP_PRICE`, `TrailStopTiming.LAGGED`. `MAEMFEAnalyzer.suggest_stop_loss(percentile=90)`, `suggest_take_profit(percentile=75)`.

Other: `StatsConfig(recent_window_size=50, track_session_stats=True, enabled=True)`; `Gatekeeper(cash_buffer_pct=0.0, settlement_reduces_buying_power=True)`; `CASH_TOLERANCE=0.01`; fractional affordability bisection 64 iterations; `LifecycleDispatcher(contract=LIFECYCLE_V1, retain_invocations=False)`; export `compression="zstd"`, `export_individual=True`, `from_parquet(recovery=False)`; `to_daily_pnl(session_aligned=False)`, `to_daily_returns(session_aligned=None -> _auto_session_aligned(calendar))`; `assign_session_date(session_start_minute=0)`; LEAN runner `timeout=1800` s; all cost-model rates default 0.0; futures `multiplier=1.0`.

Validation evidence (release gates): fill quantities match at 1e-5, prices at 1e-8, money to the cent; synthetic scenarios exact after 1e-8 quantization; strict scenario counts vectorbt_strict 17/17, vectorbt_oss_strict 16/16, backtrader_strict 17/17, zipline_strict 16/16; real-strategy bar counts ETF 1,995, CME futures 1,595, crypto 2,426, FX 2,108, US equity panel 4,146 (4,027 for Zipline/LEAN); synthetic stress 250 assets x 5,040 sessions = 1,260,000 bars, 427,790 target intents; cross-framework synthetic workload 50 assets x 252 sessions. Pinned: VectorBT Pro 2026.6.27, VectorBT OSS 1.1.0, Backtrader 1.9.78.123, Zipline Reloaded 3.1.1, LEAN 18001. Performance (2026-09-24, 24 CPUs, engine call only, 1 warm-up + 10 runs): medians 0.41-0.75 s (ETF/CME/FX), 1.33 s (crypto), 23-28 s (US equity panel); speed ratios 0.163x (vs VectorBT Pro, US panel) to 24.868x (vs Backtrader, US panel). Commands: `ML4T_COMPARISON_INPROC=1 uv run pytest tests/contracts/test_cross_engine_contracts.py -q`; `uv run python validation/performance_baseline.py --output release-performance-evidence.json`; `python validation/run_all_correctness.py --framework {vectorbt_oss|backtrader|zipline} --scenarios 01,03,05,09`; artifacts `validation/REAL_STRATEGY_RESULTS.json`, `CORRECTNESS_RESULTS.json`, `LARGE_SCALE_RESULTS.json`.

## Where the book uses it

- **Chapter 16 (strategy simulation)**: the `16_case_study_lean_parity` notebook replays each case study's target weights through both LEAN and ml4t-backtest's `lean` profile. Each `chapter16_<case_study>` workspace = `main.py` + `weights.csv` + `rebalance_dates.csv` + `asset_symbols.csv` + LEAN daily zips (prices as x10000 ints); LEAN logs `ml4t_order_events.csv` / `ml4t_daily_equity.csv`; parity is asserted on the sorted fill multiset `(timestamp, asset, side, quantity, 4-decimal price)` plus terminal portfolio value. Dropped names are still sized off last close and liquidated at the next real bar (LEAN fill-forward). Case studies covered: ETF allocation, CME futures, crypto perpetual funding (`funding_df`), FX allocation (USD-quoted pairs), US equity panel.
- **Chapter 17 (portfolio construction)**: `TargetWeightExecutor` + `WeightProvider` consume optimizer weights (riskfolio-lib, PyPortfolioOpt, cvxpy); `LongShortStrategy.rank_assets` is the long/short ML template.
- **Chapter 18 (transaction costs)**: commission/slippage/spread models, `SquareRootImpact` (Almgren-Chriss), `VolumeParticipationLimit`, `TradeAnalyzer.avg_cost_drag`.
- **Chapter 19 (risk management)**: position rules (stops, trails, ATR stops, `MAEMFEAnalyzer.optimal_exit_levels`), `RiskManager` limits (drawdown, daily loss, VaR/CVaR, exposure), kill-switch semantics (`halt`/`liquidate`).
- **Chapter 25 (live trading)**: the same `Strategy`/`Broker` interface and `AssetTradingStats` run in `ml4t-live`; docs cover stateful strategies (Kelly sizing, pairs, circuit breakers).
- **Chapter 3 / microstructure case study**: quote-aware execution (`ExecutionPrice.QUOTE_SIDE`, bid/ask/size columns, `get_quote_context`).
- Docs pages: getting-started/quickstart, user-guide/{data-feed, strategies, stateful-strategies, execution-semantics, configuration, risk-management, rebalancing, results, market-impact, profiles}; `validation/README.md`, `validation/release_checks.txt`, `LIMITATIONS.md`.

## Glossary

- **Same-bar vs next-bar execution**: fill on the signal bar's close vs the following bar's open (`ExecutionMode`); next-bar is the default and the realistic choice for close-based signals.
- **Exit-first ordering**: position-reducing orders fill before entries within a bar.
- **Gap-through**: bar opens beyond a stop level; fill at the open, not the stop.
- **Water mark (HWM/LWM)**: running high/low used by trailing stops (`WaterMarkSource`, `InitialHwmSource`, `TrailStopTiming`); **lagged** uses the previous bar's mark, **intrabar** updates before checking, **live** uses the current close.
- **StopFillMode**: assumed fill price when a stop/target is breached inside a bar (stop price, close, bar extreme, next open).
- **Lock-notional**: short policy that locks short notional as collateral instead of crediting proceeds (VectorBT parity).
- **Buying-power reservation / submission precheck**: cash reserved at submission (LEAN) / pseudo-execution check at submit (Backtrader).
- **Target intent / child intent**: canonical idempotent portfolio target registered pre-open, lowered to child orders at `process_opening`, reconciled against fills.
- **Decision session**: complete daily cross section (`session_col`) whose final close is the decision price.
- **Session close / session date**: exchange-calendar boundary used for weekly/month-end rebalances and session-aligned P&L; intraday feeds without a calendar must pass `is_session_close`.
- **Implicit daily feed**: daily feed whose timestamps do not state the session boundary; schedule code validates intervals and boundary modes.
- **Rebalance id**: broker-issued id shared by all orders from one `execute()` call; enables gatekeeper prioritization via `priority_notional`.
- **Gross leverage / net exposure**: `sum(abs(w))` / signed sum, as ratios to equity.
- **MAE / MFE**: maximum adverse / favorable excursion; edge ratio = avg MFE / avg |MAE|; efficiency = realized return / MFE.
- **Cost drag**: `(fees + slippage) / notional` per trade.
- **NLV**: net liquidating value = cash + marked positions (`AccountState.total_equity`).
- **Parity profile**: a `BacktestConfig` preset reproducing another engine's conventions exactly.
- **Spread convention**: whether a configured spread is the full quoted spread (halved per side) or already per-side.
- **Largest-remainder allocation**: rounding scheme distributing leftover cash to assets with the largest fractional remainders, subject to a cash buffer.
- **Artifact manifest**: versioned index of result components in a Parquet directory governing validated loading and recovery.
- **Funding event / payment**: perpetual-swap periodic cash flow per asset per feed timestamp; zero payment recorded when no position.

## Related references

- `chapters/16_strategy_simulation.md` - the chapter this library implements; LEAN parity case study and backtest pitfalls.
- `chapters/17_portfolio_construction.md` - optimizer weights consumed by `TargetWeightExecutor` / `WeightProvider`.
- `chapters/18_transaction_costs.md` - commission, slippage, spread, impact and participation models.
- `chapters/19_risk_management.md` - position rules, portfolio limits, VaR/CVaR, kill switches.
- `chapters/25_live_trading.md` - the shared `Strategy`/`Broker` contract with ml4t-live.
- `chapters/03_market_microstructure.md` - quote-aware execution and marking.
- `chapters/06_strategy_definition.md` - turning predictions into entry/exit rules before simulation.
- `libraries/ml4t_diagnostic.md` - consumes `to_daily_returns()` / `to_trade_records()`; `enrich_trades_with_signals`.
- `libraries/ml4t_live.md` - same strategy class, live broker adapters.
- `libraries/ml4t_data.md` - point-in-time, adjusted prices and `MarketDataSpec` -> `FeedSpec` bridge.
- `libraries/ml4t_models.md` - source of the `prediction` signal column.
- `case_studies/etfs.md`, `case_studies/cme_futures.md`, `case_studies/crypto_perps_funding.md`, `case_studies/fx_pairs.md`, `case_studies/us_equities_panel.md` - the five LEAN-parity replays (sessions, funding, multipliers).
- `case_studies/nasdaq100_microstructure.md` - bid/ask quote feeds for `QUOTE_SIDE` execution.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `workflow.md`, `companion_repo.md` - cross-cutting rules and where the notebooks live.
- Further reading:
  - Almgren and Chriss, "Optimal execution of portfolio transactions" (square-root impact model behind `SquareRootImpact`).
  - PySystemTrade (Carver) futures commission conventions behind `FuturesCommission`.
  - VectorBT Pro / Backtrader / Zipline Reloaded / QuantConnect LEAN documentation for the conventions each parity preset reproduces.
