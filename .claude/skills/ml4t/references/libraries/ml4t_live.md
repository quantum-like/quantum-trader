# ml4t-live library (production trading, broker integrations)

> `ml4t-live` (0.1.2, MIT, Stefan Jansen) is the **live trading runtime** for ML4T strategies: broker integrations (Interactive Brokers, Alpaca), layered pre-trade risk checks (`SafeBroker`), durable hash-chained persistence, and a **shadow -> paper -> live** promotion path. It is the LAST stage of the workflow: use it only after a strategy has passed research (`ml4t-data`, `ml4t-engineer`, `ml4t-models`, `ml4t-diagnostic`) and simulation (`ml4t-backtest`). It runs the same portable `Strategy` (lifecycle **V1**, contracts from `ml4t-specs`) that `ml4t-backtest` runs, so decision logic is written once and only the broker/feed wiring changes. Repo: https://github.com/ml4t/live ; docs: https://www.ml4trading.io/docs/live/ .

Stable support boundary: `LiveEngine`, `SafeBroker`, IB and Alpaca **execution** adapters, `OKXFundingFeed`, typed bar aggregation. Experimental (require `experimental=True`): Alpaca, IB, generic CCXT and DataBento **data feeds**. Shadow mode never routes orders to the wrapped broker.

## Install and import

| Item | Value |
|---|---|
| Install | `uv add ml4t-live`; DataBento feed only via `uv add 'ml4t-live[experimental]'` |
| Python | >= 3.12 (3.12, 3.13, 3.14); Linux/macOS/Windows; no hardware acceleration |
| Pinned deps | `alpaca-py==0.44.0`, `ccxt>=4.5.31,<=4.5.74`, `httpx==0.28.1`, `ib-async==2.1.0`, `ml4t-backtest>=0.1.0,<0.2`, `ml4t-specs>=0.1.1,<0.2`; extra `experimental`: `databento==0.84.0` |
| Dev gate | `uv sync --all-extras --dev --locked` then `uv run python scripts/qualification/run_stable_gate.py` (lint, format, types, tests, coverage, stress/perf, strict docs, builds, clean installs, security) |
| Import | `from ml4t.live import LiveEngine, SafeBroker, LiveRiskConfig, ExecutionMode, VirtualPortfolio, IBBroker, AlpacaBroker, OKXFundingFeed, BarAggregator, ...` (full `__all__` below); broker objects from `from ml4t.backtest import Order, OrderSide, OrderStatus, OrderType, Position, Strategy, BacktestConfig` |
| Version | `ml4t.live.__version__` (from `_version.py`; fallback `"0.0.0.dev0"`) |

`ml4t.live.__all__` (verified): AlpacaBroker, IBBroker, AlpacaDataFeed, BarAggregator, BarBuffer, FeedContractError, FeedContinuityError, FeedOverflowError, FeedQueueSnapshot, IBDataFeed, OKXFundingFeed, CryptoFeed, DataBentoFeed, ExperimentalFeedError, ExperimentalFeedWarning, LifecycleInvocation, LiveEngine, RuntimeCleanupError, RuntimeErrorContext, RuntimeFailureError, RuntimeState, RuntimeTransition, runtime_error_context, BrokerOrderContractError, CanonicalOrderRequest, OrderValidationError, UnsupportedOrderCapabilityError, LiveLifecycleDispatcher, StrategyCallbackTimeoutError, callback_trace, LiveIntentError, LiveStrategyRuntime, LiveStrategyRuntimeError, ReducingRiskExecutionError, UnsupportedLiveCapabilityError, default_live_execution_policy, AsyncBrokerProtocol, BrokerProtocol, DataFeedProtocol, AcceptedOrderPersistenceError, AuditJournalError, BrokerSnapshotError, ExecutionMode, ExecutionModeError, ConcurrentStateWriterError, CorruptStateError, LiveRiskConfig, OrderReplacementGapError, PersistenceSafetyError, ReconciliationMismatchError, RiskLimitError, RiskState, SafeBroker, UnsafePersistencePathError, VirtualPortfolio, ThreadSafeBrokerWrapper.

Module layout (`ml4t/live/`): `engine.py` (LiveEngine, RuntimeState, watchdog), `safety.py` (ExecutionMode, LiveRiskConfig, RiskState, VirtualPortfolio, SafeBroker), `runtime.py` (LiveStrategyRuntime, default_live_execution_policy), `wrappers.py` (ThreadSafeBrokerWrapper), `lifecycle.py` (LiveLifecycleDispatcher), `orders.py` (CanonicalOrderRequest), `persistence.py` (SecureStateStore, SecureAuditJournal, redact_sensitive), `state_migration.py`, `protocols.py`, `brokers/{alpaca,ib}.py`, `feeds/{aggregator,alpaca_feed,crypto_feed,databento_feed,ib_feed,okx_feed,events,experimental,queue}.py`, `cli/main.py`.

## API map by task

### Engine and lifecycle (engine.py, lifecycle.py)

| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `LiveEngine` | `LiveEngine(strategy, broker: AsyncBrokerProtocol, feed: DataFeedProtocol, *, on_error=None, feed_silence_seconds=None, watchdog_poll_seconds=1.0, halt_on_unhealthy=False, auto_recover=False, recovery_cooldown_seconds=5.0, max_recovery_attempts=3, max_event_age_seconds=None, on_health_change=None, strategy_callback_timeout_seconds=5.0, lifecycle_version=LifecycleVersion.V1, execution_policy=None, strategy_config: BacktestConfig\|None=None)` | Async orchestrator: receives feed events, dispatches to strategy, owns resources | Pass the `SafeBroker`, never a raw adapter |
| `LiveEngine.connect()` | `await connect()` | Acquire broker then feed transactionally | Reconciliation runs inside `SafeBroker.connect()` |
| `LiveEngine.run()` | `await run()` | Main event loop | SIGINT/SIGTERM route into the transactional stop path |
| `LiveEngine.stop()` | `await stop()` | Release feed then broker exactly once (reverse acquisition order) | `RuntimeCleanupError(cleanup_result)` if any resource not released |
| Introspection | `runtime_status(now=None) -> dict`, `runtime_state() -> RuntimeState`, `runtime_transitions() -> tuple[RuntimeTransition, ...]`, `operational_events() -> tuple[dict, ...]`, `stats() -> dict` | Health + equity session context, state machine evidence | All redacted; `RuntimeErrorContext` carries no exception text |
| `RuntimeState` | `StrEnum`: STOPPED, PREFLIGHT, CONNECTING_BROKER, RECONCILING, STARTING_FEED, READY, STARTING_STRATEGY, RUNNING, DEGRADED, RECOVERING, STOPPING, FAILED | Explicit phase of resource/strategy ownership | Members verified from source |
| `runtime_error_context` | `runtime_error_context(error) -> RuntimeErrorContext\|None`; `.to_dict()` -> component, operation, runtime_state, recovery_action | Operator context for a runtime exception | No secrets, no exception text |
| `RuntimeFailureError` | `RuntimeFailureError(reason)` | Async failure reached terminal state (e.g. recovery attempts exhausted) | |
| `LiveLifecycleDispatcher` | `LiveLifecycleDispatcher(strategy, contract=LIFECYCLE_V1, *, callback_timeout_seconds=5.0, event_recorder=None)`; `await dispatch(phase, *args, event_time=None)`; `callback_counts()`, `invocation_count()`, `dropped_invocation_count()`, `validate_completed_run(market_event_count, *, baseline=None)`, `close()` | One worker thread for ALL sync strategy callbacks | `validate_completed_run` enforces exactly-once start/end and consistent event counts |
| `StrategyCallbackTimeoutError` | `(callback, timeout_seconds, elapsed_seconds)` (TimeoutError) | Raised once an over-deadline callback becomes quiescent | No re-entry into the event loop |
| `callback_trace` | `callback_trace(invocations) -> tuple[(phase, status, event_time), ...]` | Cross-engine (backtest vs live) comparison of callback sequences | |

### Safety and risk (safety.py)

| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `ExecutionMode` | `StrEnum`: `SHADOW="shadow"`, `PAPER="paper"`, `LIVE="live"` | Declared routing mode | `ExecutionModeError(ValueError)` when ambiguous |
| `LiveRiskConfig` | dataclass; fields in "Defaults" below; `require_execution_mode() -> ExecutionMode` | All risk limits, mode, persistence paths | `__post_init__` validates; `None` disables a limit |
| `SafeBroker` | `SafeBroker(broker: AsyncBrokerProtocol, config: LiveRiskConfig)` | Risk-controlled wrapper implementing `AsyncBrokerProtocol` | Wrap EVERY adapter, including in shadow |
| `submit_order_async` | `(asset, quantity, side=None, order_type=OrderType.MARKET, limit_price=None, stop_price=None, **kwargs) -> Order` | Full pre-side-effect risk validation then route (or virtual fill in shadow) | Raises `RiskLimitError` |
| `cancel_order_async` / `replace_order_async` | `cancel_order_async(order_id) -> bool`; `replace_order_async(order_id, *, quantity=None, limit_price=None, stop_price=None) -> Order` | Replace = cancel-and-resubmit | `OrderReplacementGapError` if cancel succeeded but resubmit not accepted; see `replacement_gaps()` |
| `close_position_async` / `close_all_positions` | `close_position_async(asset) -> Order\|None`; `close_all_positions() -> list[Order]` | Flatten one / all (emergency) | Allowed under kill switch when `allow_reducing_risk_when_killed` |
| `reduce_position_async` | `(asset, quantity, *, reason: str, idempotency_key: str, fill_price=None) -> Order` | Explicitly reducing order under the kill-switch policy | Requires `reason` and `idempotency_key` |
| Kill switch | `enable_kill_switch(reason="Manual")`, `disable_kill_switch()` ("use with caution") | Persisted halt flag | Auto-activated by drawdown AND daily-loss breach (verified from source) |
| `record_market_snapshot` | `(asset, price, timestamp=None)` | Feed the staleness / fat-finger cache (`MarketSnapshot`) | Engine calls `_record_market_data` per feed update |
| Persistence API | `load_portable_strategy_state() -> dict`, `save_portable_strategy_state(state)`, `record_event(event, **payload)`, `persistence_status() -> dict`, `close_persistence()` | Portable strategy state + audit journal | `persistence_status()` holds no secrets |
| Reconciliation | `connect()`, `reconciliation_report() -> dict\|None`, `preview_reconciliation_async() -> dict`, `preflight_async() -> dict`, `replacement_gaps() -> dict[str, dict]` | Persisted vs live broker comparison | `preflight_async` probes + reconciles without persisting |
| Identity | `assert_paper_trading()`, `assert_live_trading()`, `execution_capabilities()`, `is_connected()` | Prove endpoint AND account match declared mode | Delegates to venue adapter |
| `RiskState` | fields: `date` (YYYY-MM-DD), `daily_loss=0.0`, `orders_placed=0`, `high_water_mark=0.0`, `session_start_equity=None`, `persisted_positions: dict[str,float]`, `persisted_pending_orders: list[dict]`, `portable_strategy_state: dict`, `replacement_gaps: dict`, `shadow_portfolio: dict`, `execution_mode: str\|None`, `kill_switch_activated=False`, `kill_switch_reason=""`; `from_dict`, `to_dict`, `validate`, `save_atomic(state, filepath)`, `load(filepath)`, `create_for_today()` | Persisted risk state that survives restarts | Fields verified from source; kill switch must be reset explicitly |
| `VirtualPortfolio` | `VirtualPortfolio(initial_cash=100_000.0)`; `positions -> dict[str, Position]` (copy), `cash`, `account_value` (cash + market value), `process_fill(order)`, `update_prices(prices: dict[str,float])`, `to_state()`, `restore_state(state)` | Shadow-mode cash/position accounting | Credential-free quick start |
| Errors | `RiskLimitError(Exception)`, `OrderReplacementGapError(RiskLimitError)`, `BrokerSnapshotError(RuntimeError)`, `ReconciliationMismatchError(RuntimeError)` | | |

### Portable intent runtime and sync facade (runtime.py, wrappers.py)

| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `default_live_execution_policy` | `(*, opening_auction: ExecutionBehavior = ExecutionBehavior.DISABLED) -> ExecutionPolicy` | "Explicit conservative live assumptions without native substitution" | Used when `LiveEngine(execution_policy=None)` |
| `LiveStrategyRuntime` | `LiveStrategyRuntime(broker, policy: ExecutionPolicy, lifecycle_version)` | Lower portable `CanonicalTargetIntent` -> child orders; reconcile; evaluate client-side `PositionRule`s | |
| Intent methods | `register_target_intent(intent, *, position_rules=None) -> CanonicalTargetIntent`; `register_position_rule_policy(policy_id, rules)`; `set_position_rules(rules, asset=None)`; `clear_position_rules(asset=None)`; `update_position_context(asset, context)`; `observe_strategy_order(order)` | Declarative targets and protective rules | `observe_strategy_order` activates rules after a sync strategy order fills |
| `process_market_event` | `await process_market_event(timestamp, data, context)` | Processes eligible targets, fills and client rules BEFORE the strategy event | |
| Inspection | `targets()`, `children()`, `reconciliations()`, `position_rule_states()`, `to_state() -> dict` | | |
| Errors | `LiveStrategyRuntimeError`; `UnsupportedLiveCapabilityError` (pre-connect/submit); `LiveIntentError` (late, conflicting, or unsafe to lower); `ReducingRiskExecutionError` (protective action cannot be submitted safely) | | |
| `ThreadSafeBrokerWrapper` | `(async_broker, loop, strategy_runtime=None)`; implements `BrokerProtocol`: `positions()`, `pending_orders()`, `is_connected()`, `get_position(asset)`, `get_positions()`, `get_account_value()`, `get_cash()`, `get_pending_orders(asset=None)`, `submit_order(...)`, `cancel_order(order_id)`, `replace_order(order_id, *, quantity, limit_price, stop_price)`, `close_position(asset)`, plus intent/rule delegates | Sync facade the strategy thread sees | `_run_sync(coro, timeout=5.0)`; docstring: 5 s getters, 30 s orders |

### Orders and persistence (orders.py, persistence.py, state_migration.py)

| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `CanonicalOrderRequest` | `from_input(asset, quantity, side: OrderSide\|None, order_type: OrderType, limit_price=None, stop_price=None, *, capabilities=()) -> CanonicalOrderRequest`; `validate_result(order: Order)` | One unsigned venue request used unchanged for checks and submission | `validate_result` raises `BrokerOrderContractError` if adapter result differs |
| Order errors | `OrderValidationError(ValueError)` (pre-side-effect), `UnsupportedOrderCapabilityError(OrderValidationError)`, `BrokerOrderContractError(RuntimeError)` | | |
| `SecureStateStore` | `(path, *, fault_injector=None)`; `acquire_writer()`, `has_writer()`, `release_writer()`, `load() -> StateSnapshot\|None`, `save(payload, *, expected_generation) -> int` | Checksummed, generation-versioned state envelope with exclusive writer lease | Lock file `<state_file>.lock`; mode 0600 |
| `SecureAuditJournal` | `(path, *, fault_injector=None)`; `validate() -> (sequence, head_hash)`, `append(event) -> dict`, `tail(limit=3) -> list[dict]` | Hash-chained JSONL audit journal | Chain validated on open |
| `redact_sensitive` | `redact_sensitive(value, *, key=None)` | Strip credential/account identifiers from diagnostics | |
| Persistence errors | all subclass `PersistenceSafetyError(RuntimeError)`: `UnsafePersistencePathError`, `CorruptStateError`, `ConcurrentStateWriterError`, `AuditJournalError`, `AcceptedOrderPersistenceError` | | |
| `migrate_portable_strategy_state` | `(value, *, position_quantities) -> (state, migrated: bool)` | Upgrade unversioned 0.1.0b4 portable state | |

### Brokers (brokers/alpaca.py, brokers/ib.py)

| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `AlpacaBroker` | `AlpacaBroker(api_key, secret_key, paper: bool = True)` | Stocks + crypto; REST for account/positions/submit, `TradingStream` WebSocket for order updates | `client_order_id` supported; MARKET/LIMIT/STOP/STOP_LIMIT mapped; `get_account_value_async()` = equity; `get_cash_async()` = signed cash (negative allowed) |
| `IBBroker` | `IBBroker(host="127.0.0.1", port=7497, client_id=1, account=None, market_data_type=None)` | Via `ib_async`; `get_account_value_async()` = NetLiquidation; `get_cash_async()` = available funds | `_create_ib_order(..., outside_rth=False)`; contract cache `_get_contract(asset)`; `_IBAsyncRedactionFilter` strips secrets from logs |
| Common async surface | `execution_capabilities()`, `connect()`, `disconnect()`, `is_connected()`, `is_connected_async()`, `assert_paper_trading()`, `assert_live_trading()`, `positions()`, `pending_orders()`, `get_position(asset)`, `get_positions_async()`, `get_position_async(asset)`, `get_pending_orders_async(asset=None)`, `get_account_value_async()`, `get_cash_async()`, `submit_order_async(...)`, `cancel_order_async(order_id)`, `replace_order_async(order_id, *, quantity, limit_price, stop_price, **kwargs)`, `close_position_async(asset)` | `AsyncBrokerProtocol` | Replace = cancel-and-resubmit on both |
| Adapter internals | Alpaca: `_map_order_status`, `_on_trade_update` (removes completed orders from pending cache), `_run_trading_stream` (background thread), `_sync_positions`/`_sync_orders` on connect. IB: `_on_order_status(trade)`, `_on_position`, `_invalidate_snapshot(message, cause)` | Caches refreshed on connect; poisoned on stream failure | Reads after poisoning raise `BrokerSnapshotError` |

### Feeds (feeds/)

| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `OKXFundingFeed` | `(symbols, *, timeframe="1H", poll_interval_seconds=60.0, queue_capacity=256)` | **Stable**; hourly OHLCV + funding rate (funding changes every 8 h, polled hourly); separate validated bar and funding events | No API key, no geo-restriction; `close()` closes HTTP client |
| `BarAggregator` | `(source_feed, bar_size_minutes=1, assets=None, flush_timeout_seconds=2.0, queue_capacity=256)`; `await start()`, `stop()`, async-iterable of `MarketEvent`; `stats()` | Stable typed bar aggregation from ticks/trades | Emits `EventCompletion.COMPLETE`; `_flush_checker` force-emits when no data arrives |
| `BarBuffer` | `update(price, size=0)`, `update_bar(payload: dict)`, `to_dict()`, `reset()` | OHLCV accumulator | `update_bar` merges without discarding range |
| `AlpacaDataFeed` | `(api_key, secret_key, symbols, *, data_type="bars" ('bars'\|'quotes'\|'trades'), feed="iex" ('iex' free \| 'sip' premium), queue_capacity=1_024, experimental=False)` | Experimental | Crypto detected by `/` in symbol (`BTC/USD`) |
| `CryptoFeed` | `(exchange, symbols, *, timeframe="1m", stream_trades=False, stream_ohlcv=True, api_key=None, api_secret=None, api_passphrase=None, experimental=False)` | Experimental async CCXT WebSocket | Candle batches: prior candles COMPLETE, newest evolving; `close()` |
| `DataBentoFeed` | `(client, symbols, *, mode="historical", replay_speed=1.0, experimental=False)`; `from_file(file_path, symbols, *, replay_speed=1.0, experimental=False)`; `from_live(api_key, dataset, schema, symbols, *, experimental=False)` | Experimental; e.g. dataset `GLBX.MDP3`, symbols `ES.FUT` | Needs `[experimental]` extra |
| `IBDataFeed` | `(ib: IB, symbols, *, exchange="SMART", currency="USD", tick_throttle_ms=100, queue_capacity=1_024, experimental=False)` | Experimental tick-by-tick via `pendingTickers` | Wrap in `BarAggregator` for bars |
| `feeds/events.py` | `utc_datetime(value, field)`, `sequence_unavailable(source, detail) -> GapEvidence`, `validate_event_timing(event, *, processing_time, max_age_seconds, future_tolerance_seconds=5.0)`, `strategy_input(event, *, processing_time) -> (timestamp, data, context)`, `EventContinuityTracker.validate(event) -> ContinuityDisposition`, `.mark_recovery()`, `.snapshot()`, `FeedContinuityError(reason, event).to_dict()` | Timing, continuity and contract validation before strategy dispatch | |
| `feeds/experimental.py` | `require_experimental_opt_in(feed, *, experimental, missing_guarantees)` | Opt-in gate | `ExperimentalFeedError` / `ExperimentalFeedWarning` |
| `BoundedEventQueue` | `(*, capacity, feed)`; `put_nowait`, `await put`, `await get`, `get_nowait`, `finish(*, discard)`, `fail(error, *, discard)`, `qsize`, `empty`, `full`, `snapshot(now=None) -> FeedQueueSnapshot` | Fail-closed buffer | `FeedOverflowError` with `GapEvidence` instead of dropping data |

### Operator CLI (cli/main.py; flags verified from source)

| Command | Flags | Purpose | Notes |
|---|---|---|---|
| `ml4t-live shadow <strategy.py>` | `--feed {okx,alpaca}` (default `okx`), `--duration N` (default 60), `--feed-silence-seconds` (default per feed: okx 90 s, others 30 s), `--state-file` (default `.ml4t_risk_state.json`) | Load strategy module, build feed, wrap `NullBroker` in `IntentPrintingSafeBroker`, print heartbeat (positions, pending, recent intents), stop after duration | Alpaca feed reads `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` env vars |
| `ml4t-live status` | `--state-file` | Read `RiskState`, tail journal (3 entries), probe Alpaca/IB, classify status via `_classify_status(state, probes)` | |
| `ml4t-live preflight {alpaca,ib}` | `--state-file`, `--strict` (fail when reconciliation not clean), `--require-market-open` (fail when US equity session closed) | `_preflight_broker` -> `PreflightResult` | Run before every paper/live start |
| `ml4t-live --version` | | | Entry: `run_cli(argv=None) -> int`, `app()`, `build_parser()` |

Helper objects: `BrokerProbeResult` (from `_probe_alpaca`/`_probe_ib`), `OrderIntentRecord`, `PreflightResult`, `NullBroker` (stub `AsyncBrokerProtocol`, shadow only), `IntentPrintingSafeBroker(SafeBroker)` with `recent_order_intents()`, `_journal_path_for_state_file`, `_tail_journal(path, limit=3)`, `_equity_session_snapshot()`, `_alpaca_paper_mode()`.

## Data contracts

| Contract | Specification |
|---|---|
| Strategy lifecycle V1 | `LIFECYCLE_V1` / `LifecycleVersion.V1` (ml4t-specs). Sync callbacks such as `on_data(timestamp, data, context, broker)` run on ONE worker thread. Engine rejects lifecycle surfaces that need unavailable historical state (`_validate_strategy_lifecycle`) |
| Strategy callback input | `strategy_input(event)` -> `(timestamp: datetime [aware UTC], data: dict[str, dict[str, Any]] [asset -> OHLCV/quote/trade payload], context: dict[str, Any])`. Legacy tuple `(timestamp, data, context)` still accepted from experimental feeds via `_validate_legacy_feed_item` |
| `MarketEvent` (ml4t-specs; inference) | `kind: MarketEventKind` (bar/quote/trade/funding), payload (`BarPayload\|QuotePayload\|TradePayload`), metadata, `EventCompletion` (`COMPLETE` vs evolving), optional sequence / `GapEvidence`. Feeds yield `MarketEvent`; `DataFeedProtocol.__anext__ -> FeedItem` |
| Timestamps | `utc_datetime()` returns an aware UTC datetime and REFUSES to silently assign or convert a zone (`FeedContractError`). `SafeBroker._normalize_timestamp` and engine `_normalize_utc` normalize feed timestamps to UTC for freshness checks |
| Broker objects (ml4t-backtest) | `Order(asset, side: OrderSide, quantity, order_id, status: OrderStatus, filled_price, filled_quantity, ...)`; `OrderType` (MARKET default; LIMIT/STOP/STOP_LIMIT); `Position(quantity, market_value, current_price, ...)`. Quantity is UNSIGNED; `side` chooses direction |
| `ExecutionCapability` (ml4t-specs) | Adapters declare `execution_capabilities() -> frozenset`; orders needing an undeclared capability are refused |
| Portable intents (ml4t-specs) | `CanonicalTargetIntent` -> `CanonicalChildOrderIntent` -> `IntentReconciliation` (child fill + remainder evidence); `PositionRule` -> `PositionRuleState` (high_water / low_water / favorable / adverse, `ExitReason`); `RoundingPolicy`; `ExecutionPolicy` / `ExecutionBehavior` (opening auction DISABLED by default) |
| `RiskState` JSON | Fields listed in API map; `validate()` applies finite / non-negative checks and JSON-tree validation; written after every accepted order and on shutdown |
| State envelope (`SecureStateStore`) | `StateSnapshot` = validated payload + metadata; checksum `_state_checksum(schema_version, generation, payload)`; canonical JSON; optimistic concurrency via `expected_generation`; lock file `<state_file>.lock`; duplicate JSON keys rejected |
| Audit journal (`SecureAuditJournal`) | Append-only JSONL; per-entry `_journal_hash`; head checksum `_journal_head_checksum(sequence, head_hash)`; chain validated on open; `tail()` returns validated trailing records. Path = `journal_file` or derived from state file |
| Portable state versioning | `migrate_portable_strategy_state(value, *, position_quantities)` upgrades the unversioned 0.1.0b4 representation |
| File permissions | State / journal / lock enforce current-user ownership and mode `0600` on POSIX; on Windows restrict the directory with service-account ACLs |
| Diagnostics | `RuntimeTransition`, `operational_events()`, `RuntimeErrorContext` are redacted (no exception text, no secrets) |

## Built-in guardrails

Risk-check ORDER inside `SafeBroker._submit_order_unlocked` (verified from source), all pre-side-effect and all raising `RiskLimitError`: kill switch -> `_check_asset` -> `_check_data_staleness` -> `_check_daily_loss` -> `_check_duplicate` (when applicable) -> `_check_rate_limit` -> `_check_order_limits` -> `_check_position_limits` -> `_check_price_deviation` (limit orders) -> `_check_drawdown` -> kill switch re-check. Counters/histories commit only after acceptance (`_commit_accepted_order`).

| # | Guardrail | Mechanism | Detect / act |
|---|---|---|---|
| 1 | Paper by default | `AlpacaBroker(paper=True)`; `IBBroker(port=7497)` | `assert_paper_trading()` requires IB port in {4002, 7497} AND account matching `DU[0-9]+`; Alpaca requires `paper=True`, client sandbox flag AND `BaseURL.TRADING_PAPER`. `assert_live_trading()` requires IB port in {4001, 7496}, an explicitly selected `account` matching the connected one and NOT `DU`-prefixed; Alpaca requires `paper=False` and `BaseURL.TRADING_LIVE` (verified from source) |
| 2 | Explicit execution mode | `LiveRiskConfig.execution_mode` in `shadow\|paper\|live`; `require_execution_mode()` raises `ExecutionModeError` when `None` ("shadow_mode=False is ambiguous"); `shadow_mode=True` with non-shadow mode raises | Always pass `execution_mode=` |
| 3 | Shadow never routes | In shadow, `SafeBroker` fills into `VirtualPortfolio` and never calls the wrapped broker (prevents "infinite buy loops"); CLI uses `NullBroker` + `IntentPrintingSafeBroker` | `recent_order_intents()`; journal event `shadow_order_filled` |
| 4 | Experimental opt-in | `require_experimental_opt_in` raises `ExperimentalFeedError` unless `experimental=True`, then emits `ExperimentalFeedWarning` listing missing guarantees on first use | Applies to Alpaca, IB, CCXT, DataBento feeds; NOT OKX or `BarAggregator` |
| 5 | Layered order checks | Asset allow/block; duplicate = same asset and `abs(q - quantity) < 0.01` within `dedup_window_seconds` (side NOT compared); rate limit = trailing 60 s window vs `max_orders_per_minute`; `abs(quantity) > max_order_shares` or value > `max_order_value`; position limits (`max_position_shares`, `max_position_value`, `max_total_exposure`, `max_positions`) on PROJECTED position including pending-order reservations (`_pending_risk_reservations`); fat-finger `abs(limit - market)/market > max_price_deviation_pct` (falls back to `Position.current_price`; no price at all -> reject); staleness `now - snapshot.observed_at > max_data_staleness_seconds` (missing snapshot also rejects) | Any limit `None` disables that check; NaN/inf invalid |
| 6 | Daily loss | `daily_loss = max(0, session_start_equity - current_equity)`; `session_start_equity` set on first check; breach ACTIVATES KILL SWITCH and raises | Verified from source; equity unavailable or non-finite -> `BrokerSnapshotError` |
| 7 | Drawdown | `drawdown = (high_water_mark - current)/high_water_mark`, HWM updated each check; breach activates kill switch | Measured from session peak, not daily open (verified) |
| 8 | Kill switch | `kill_switch_enabled` or persisted `RiskState.kill_switch_activated`; new orders rejected; `allow_reducing_risk_when_killed=True` still permits `reduce_position_async` / `close_position_async`; `halt_on_reducing_risk_failure=True` halts on `ReducingRiskExecutionError` | Persists across restarts; reset explicitly via `disable_kill_switch()` |
| 9 | Startup reconciliation | `SafeBroker.connect()` compares persisted positions/pending orders vs live snapshot (`_build_reconciliation_report`); `fail_on_reconciliation_mismatch=True` raises `ReconciliationMismatchError` (fail closed); default `False` logs | `preflight_async()` / `ml4t-live preflight --strict` before trading |
| 10 | Broker snapshot contract | `BrokerSnapshotError` when state unavailable or contract violated; adapters poison/invalidate caches on stream failure (`_poison_snapshot`, `_invalidate_snapshot`); account metrics validated finite (`_validate_account_metric`; Alpaca allows negative cash) | Never trust a stale cache |
| 11 | Order contract fidelity | `CanonicalOrderRequest.validate_result` -> `BrokerOrderContractError`; `UnsupportedOrderCapabilityError`; `OrderReplacementGapError` tracked in `replacement_gaps()` | |
| 12 | Durable persistence or stop | Exclusive writer lease (flock; `ConcurrentStateWriterError`); atomic write + directory fsync; checksum + duplicate-key rejection (`CorruptStateError`); symlink / untrusted path / wrong owner rejection (`UnsafePersistencePathError`); mode forced 0600; `AcceptedOrderPersistenceError` if accepted order cannot be journaled; `fail_on_journal_error=True` -> `AuditJournalError` stops trading | `persistence_status()` |
| 13 | Feed timing | `validate_event_timing` rejects events > `future_tolerance_seconds=5.0` ahead of processing time and (if `max_event_age_seconds` set) stale events BEFORE dispatch; naive timestamps -> `FeedContractError` | |
| 14 | Feed continuity | `EventContinuityTracker` keyed on `(source, symbol, kind)` returns `ContinuityDisposition`; unsafe continuation -> `FeedContinuityError`; `mark_recovery()` after watchdog recovery | |
| 15 | Fail-closed buffering | `BoundedEventQueue` raises `FeedOverflowError` (with `GapEvidence`) instead of dropping data | Capacities: 1,024 (Alpaca, IB feeds), 256 (aggregator, OKX) |
| 16 | Watchdog and recovery | Poll every `watchdog_poll_seconds=1.0`; `feed_silence_seconds` (None = off) flags silence; `halt_on_unhealthy=False`; `auto_recover=False`; `recovery_cooldown_seconds=5.0`; `max_recovery_attempts=3` then `RuntimeFailureError` | Transitions logged and sent to `on_health_change(status_name, status_dict)` |
| 17 | Callback deadline | `strategy_callback_timeout_seconds=5.0`; `StrategyCallbackTimeoutError` once the callback becomes quiescent; `validate_completed_run` enforces exactly-once start/end | |
| 18 | Engine config validation | `_validate_runtime_configuration` rejects non-positive/non-finite `feed_silence_seconds`, `watchdog_poll_seconds`, `recovery_cooldown_seconds`, negative `max_recovery_attempts`; `_validate_event_age` likewise for `max_event_age_seconds` (inference on exact bounds) | |
| 19 | Capability pre-check | `LiveStrategyRuntime._validate_preconnect_capabilities` -> `UnsupportedLiveCapabilityError` BEFORE connection; `LiveIntentError` for late or conflicting targets | |
| 20 | Transactional lifecycle | Acquire broker -> feed, release feed -> broker exactly once; cleanup result retained without masking the primary error (`_retain_failure`, `_add_cleanup_note`); `RuntimeCleanupError` | |
| 21 | Secret redaction | `_IBAsyncRedactionFilter`, `redact_sensitive`, `RuntimeErrorContext` without exception text | |
| 22 | Config sanity | `allowed_assets & blocked_assets` overlap -> `ValueError`; empty `state_file` -> `ValueError`; `journal_file` must differ from `state_file` and `<state_file>.lock` | |

## Usage patterns

Credential-free quick start (README):
```python
from ml4t.backtest import Order, OrderSide, OrderStatus
from ml4t.live import VirtualPortfolio
portfolio = VirtualPortfolio(initial_cash=100_000)
fill = Order(asset="SPY", side=OrderSide.BUY, quantity=10, order_id="shadow-1",
             status=OrderStatus.FILLED, filled_price=500.0, filled_quantity=10)
portfolio.process_fill(fill)
assert portfolio.positions["SPY"].quantity == 10 and portfolio.cash == 95_000
```

Engine wiring, shadow first (engine docstring + config example):
```python
from ml4t.live import LiveEngine, SafeBroker, LiveRiskConfig, IBBroker
config = LiveRiskConfig(max_position_value=25_000.0, max_daily_loss=2_000.0,
                        execution_mode="shadow")
safe = SafeBroker(IBBroker(port=7497), config)
engine = LiveEngine(strategy, safe, feed, feed_silence_seconds=120, auto_recover=True)
await engine.connect()
try:
    await engine.run()
except KeyboardInterrupt:
    await engine.stop()
```

Feeds:
```python
feed = OKXFundingFeed(symbols=["BTC-USDT-SWAP"], timeframe="1H")        # stable, no key
feed = AlpacaDataFeed(api_key, secret_key, ["AAPL", "BTC/USD"], experimental=True)
feed = CryptoFeed(exchange="binance", symbols=["BTC/USDT"], timeframe="1m", experimental=True)
feed = DataBentoFeed.from_live(api_key, dataset="GLBX.MDP3", schema=..., symbols=["ES.FUT"], experimental=True)
bars = BarAggregator(IBDataFeed(ib, ["SPY"], experimental=True), bar_size_minutes=1)
await feed.start()
async for event in feed: ...
```

Strategy side (sync, via `BrokerProtocol` facade): `broker.submit_order("SPY", 10, side=OrderSide.BUY)`; `broker.register_target_intent(intent, position_rules=rule)`; `broker.get_target_intents()`. Write the strategy once against lifecycle V1 and run it unchanged in `ml4t-backtest` and `ml4t-live`; use `callback_trace` to compare callback sequences across engines.

Operator CLI: `ml4t-live shadow strategy.py --feed okx --duration 600`, `ml4t-live status --state-file .ml4t_risk_state.json`, `ml4t-live preflight ib --strict --require-market-open`.

Promotion path (sequence, do not skip stages):

| Stage | `execution_mode` | Broker | Required assertion | Exit criterion |
|---|---|---|---|---|
| 1 Shadow | `"shadow"` | any adapter or `NullBroker`; fills in `VirtualPortfolio` | none (never routes) | intents sane in `recent_order_intents()` / journal; no `RiskLimitError` storms |
| 2 Paper | `"paper"` | `AlpacaBroker(paper=True)` / `IBBroker(port=7497 or 4002)` | `assert_paper_trading()` | `preflight --strict` clean; reconciliation clean across restarts |
| 3 Live | `"live"` | `AlpacaBroker(paper=False)` / `IBBroker(port=7496 or 4001, account="U...")` | `assert_live_trading()` | start with SMALL limits (`max_order_value`, `max_position_value`, `max_daily_loss`) and `fail_on_reconciliation_mismatch=True` (inference on recommended flags) |

Operational checklist: run `ml4t-live preflight` before each session; keep `fail_on_journal_error=True`; feed `record_market_snapshot` (or let the engine do it) so staleness and fat-finger checks have data; after a kill-switch trip, inspect `RiskState.kill_switch_reason` and the journal tail BEFORE `disable_kill_switch()`; treat `BrokerSnapshotError` and `ReconciliationMismatchError` as stop-and-investigate, not retry.

## Defaults and configuration keys

`LiveRiskConfig` (verified from source):

| Key | Default | Validation |
|---|---|---|
| `max_position_value` | 50_000.0 | > 0 or None |
| `max_position_shares` | 1000.0 | > 0 or None |
| `max_total_exposure` | 200_000.0 | > 0 or None |
| `max_positions` | 20 | positive int or None |
| `max_order_value` | 10_000.0 | > 0 or None |
| `max_order_shares` | 500.0 | > 0 or None |
| `max_orders_per_minute` | 10 | positive int or None (trailing 60 s window) |
| `max_daily_loss` | 5_000.0 | > 0 or None; breach trips kill switch |
| `max_drawdown_pct` | 0.05 | 0 < x <= 1 or None; from session HWM; breach trips kill switch |
| `max_price_deviation_pct` | 0.05 | 0 < x <= 1 or None |
| `max_data_staleness_seconds` | 60.0 | > 0 or None |
| `dedup_window_seconds` | 1.0 | >= 0 or None (0 disables) |
| `allowed_assets` / `blocked_assets` | `set()` | no overlap; empty `allowed_assets` = allow all |
| `shadow_mode` | False | must agree with `execution_mode` |
| `execution_mode` | None | `"shadow"` / `"paper"` / `"live"` (case-insensitive); None -> `ExecutionModeError` |
| `kill_switch_enabled` | False | |
| `allow_reducing_risk_when_killed` | True | |
| `halt_on_reducing_risk_failure` | True | |
| `fail_on_reconciliation_mismatch` | False | set True for live (inference) |
| `state_file` | `".ml4t_risk_state.json"` | non-empty; lock at `<state_file>.lock` |
| `journal_file` | None | non-empty if given; not state/lock path; derived from state file when None |
| `fail_on_journal_error` | True | |

Other defaults by component:

| Component | Defaults |
|---|---|
| `VirtualPortfolio` | `initial_cash=100_000.0` |
| `LiveEngine` | `watchdog_poll_seconds=1.0`, `recovery_cooldown_seconds=5.0`, `max_recovery_attempts=3`, `strategy_callback_timeout_seconds=5.0`, `lifecycle_version=LifecycleVersion.V1`, `feed_silence_seconds=None` (silence watchdog off), `max_event_age_seconds=None` (staleness rejection off), `halt_on_unhealthy=False`, `auto_recover=False`, `execution_policy=None` (-> `default_live_execution_policy()`), `strategy_config=None` |
| `LiveLifecycleDispatcher` | `contract=LIFECYCLE_V1`, `callback_timeout_seconds=5.0` |
| `ThreadSafeBrokerWrapper._run_sync` | `timeout=5.0`; docstring: 5 s getters, 30 s orders (30 s appears only in docstring) |
| `validate_event_timing` | `future_tolerance_seconds=5.0` |
| `BarAggregator` | `bar_size_minutes=1`, `flush_timeout_seconds=2.0`, `queue_capacity=256` |
| `AlpacaDataFeed` | `data_type="bars"`, `feed="iex"`, `queue_capacity=1_024`, `experimental=False` |
| `CryptoFeed` | `timeframe="1m"`, `stream_ohlcv=True`, `stream_trades=False`, `experimental=False` |
| `DataBentoFeed` | `mode="historical"`, `replay_speed=1.0`, `experimental=False` |
| `IBDataFeed` | `exchange="SMART"`, `currency="USD"`, `tick_throttle_ms=100`, `queue_capacity=1_024`, `experimental=False` |
| `OKXFundingFeed` | `timeframe="1H"`, `poll_interval_seconds=60.0`, `queue_capacity=256`; funding changes every 8 h |
| `IBBroker` | `host="127.0.0.1"`, `port=7497`, `client_id=1`, `account=None`, `market_data_type=None`; paper ports {7497 TWS, 4002 Gateway}; live ports {7496 TWS, 4001 Gateway} |
| `AlpacaBroker` | `paper=True` |
| `SecureAuditJournal.tail` / CLI `_tail_journal` | `limit=3`; state/journal/lock file mode `0600` |
| `default_live_execution_policy` | `opening_auction=ExecutionBehavior.DISABLED` |
| CLI `shadow` | `--feed okx`, `--duration 60`, feed-silence default 90 s (okx) / 30 s (others), `--state-file .ml4t_risk_state.json` |

## Where the book uses it

No notebooks for this library were in the digest; the usage patterns above are reconstructed from the README, docstrings and signatures. The library is the implementation substrate for the book's backtest-to-live migration and deployment/production material (3rd edition live-trading and MLOps chapters), and its risk controls (drawdown kill switch, daily loss, position and exposure caps) operationalize the risk-management chapter's concepts (inference; chapter numbers not in this digest). The OKX funding feed pairs naturally with the crypto perpetuals / funding case study (inference).

Related references:
- `chapters/25_live_trading.md` — the workflow stage this library implements (shadow -> paper -> live, reconciliation, kill switches).
- `chapters/26_mlops_governance.md` — audit journal, state persistence, drift and operational monitoring context.
- `chapters/19_risk_management.md` — daily loss, drawdown, exposure caps and kill-switch rationale behind `LiveRiskConfig`.
- `chapters/16_strategy_simulation.md` — the backtest engine that shares the portable lifecycle V1 `Strategy`.
- `chapters/18_transaction_costs.md` — execution assumptions that `default_live_execution_policy` makes explicit (opening auction disabled).
- `chapters/21_rl_execution_hedging.md` — execution logic that would sit behind target intents and child orders.
- `libraries/ml4t_backtest.md` — `Strategy`, `Order`, `OrderSide`, `OrderStatus`, `OrderType`, `Position`, `BacktestConfig` consumed here.
- `case_studies/crypto_perps_funding.md` — the stable `OKXFundingFeed` serves funding-rate strategies.
- `case_studies/cme_futures.md` — DataBento `GLBX.MDP3` / `ES.FUT` live feed example.
- `workflow.md` — where the live stage sits in the end-to-end process.
- `guardrails.md` — cross-cutting list; the live-stage guardrails above are the production subset.
- `glossary.md` — shared vocabulary (lifecycle, intents, kill switch).

Further reading:
- ml4t-live docs: installation, backtest-to-live, risk, brokers, feeds, cli, api, qualification (https://www.ml4trading.io/docs/live/).
- `ml4t-specs` for `LifecycleVersion`, `LIFECYCLE_V1`, `LifecyclePhase`, `ExecutionPolicy`, `ExecutionBehavior`, `ExecutionCapability`, `CanonicalTargetIntent`, `CanonicalChildOrderIntent`, `IntentReconciliation`, `PositionRule`, `PositionRuleState`, `ExitReason`, `RoundingPolicy`, `MarketEvent`, `MarketEventKind`, `EventCompletion`, `GapEvidence` (exact module is an inference).

## Glossary

| Term | Meaning |
|---|---|
| Shadow mode | Orders validated and logged, fills simulated in `VirtualPortfolio`; wrapped broker never touched |
| Paper / live | Real venue endpoints; paper = Alpaca paper endpoint / IB ports 7497 or 4002 with `DU` account; identity asserted by `assert_paper_trading` / `assert_live_trading` |
| SafeBroker | Risk-controlled wrapper implementing `AsyncBrokerProtocol` with persistence, reconciliation and kill switch |
| Kill switch | Persisted halt flag that blocks new orders (manual, daily-loss or drawdown trigger); reducing orders optionally allowed |
| Reconciliation | Startup comparison of persisted vs live broker positions / pending orders |
| Target intent / child intent | Portable declarative position target lowered into venue orders by `LiveStrategyRuntime` |
| Position rule | Client-evaluated protective exit rule (stop / trailing / take-profit style) tracked in `PositionRuleState` |
| Lifecycle V1 | Versioned sync callback contract from ml4t-specs shared by backtest and live |
| Experimental feed | Adapter outside the stable boundary; requires `experimental=True` |
| BoundedEventQueue | Fail-closed buffer raising `FeedOverflowError` on overflow |
| Watchdog | Engine task monitoring feed silence / broker health, optionally recovering |
| Audit journal | Hash-chained JSONL of runtime / order events |
| Generation | Monotonically increasing state-envelope version for optimistic concurrency |
| Fat-finger check | `max_price_deviation_pct` limit-price vs market guard |
| Session start equity / HWM | Baselines in `RiskState` for daily-loss and drawdown checks |
| Replacement gap | Cancel succeeded but replacement not accepted; tracked in `replacement_gaps()` |

Reader uncertainty retained from the notes: `ContinuityDisposition` members, `FeedQueueSnapshot` fields and `MarketEvent` payload module are not confirmed; the 30 s order timeout appears only in the wrappers docstring; recommended live-stage limit values are inferences, not library defaults.
