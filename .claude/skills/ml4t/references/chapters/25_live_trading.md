# Chapter 25: Live Trading Systems

> Most algorithmic trading projects fail at the backtest-to-live transition, not for lack of edge but because the production system diverges from the research environment in details nobody notices until money is on them. The chapter's answer is a unified framework: one `Strategy` class runs unchanged through `ml4t.backtest.Engine` (backtest) and `ml4t.live.LiveEngine` (paper/live), and parity is established by field-by-field assertions on the signal tape, stage by stage, as a CI gate. Around that core it builds the operational layer: broker integration (Interactive Brokers, Alpaca, OKX data + Alpaca execution, QuantConnect prediction bridge), an explicit order-lifecycle state machine with 10 states and 19 transitions, and `SafeBroker` defense-in-depth controls (order/position/exposure caps, rate limit, staleness, persisted kill switch, startup reconciliation, shadow mode). The position it argues: deployment artefacts are not research artefacts (hyperparameters cross over, weights do not), venue is an operational not a research decision, fail loudly rather than degrade silently, and begin every live deployment in shadow mode.

## When to use this reference

- Taking a strategy from `ml4t.backtest.Engine` to paper or live execution with `ml4t.live.LiveEngine`.
- Writing or reviewing a backtest-vs-live parity test (single-stage signal tape or staged features/predictions/signals/orders/fills gates).
- Connecting to Interactive Brokers TWS/Gateway (`IBBroker`, `IBDataFeed`, ports, client IDs, `MARKET_DATA_TYPE`, RTH gate) or Alpaca (`AlpacaBroker`, `AlpacaDataFeed`, credential gates, IEX vs SIP).
- Building a scheduled deployment loop: refresh data, refit a deployment model, predict the live cross-section, replay an offline reference tape, stage a basket, write a run record.
- Choosing the training cutoff for a deployment refit so a forward label does not read prices inside the live window (`LABEL_AVAILABLE_AS_OF`).
- Configuring `SafeBroker` / `LiveRiskConfig` limits, the persisted kill switch, stale-data rejection, or interpreting a `reconciliation_report` or `RiskLimitError`.
- Modelling order handling with partial fills, cancel-vs-fill races, rejections and crash recovery (`OrderState`, `VALID_TRANSITIONS`, `PENDING_CANCEL`).
- Reconciling current broker positions against a target basket (`delta_qty`), waiting for fills, flattening with MOC orders.
- Deploying crypto (24/7, funding hours, USD vs USDT, split data/execution venues) or FX (IDEALPRO contract qualification, base-currency sizing).
- Exporting predictions to QuantConnect/LEAN via the prediction-bridge JSON instead of reimplementing features in the platform.
- Planning a staged rollout: shadow mode, paper, live with small positions; interpreting `LiveEngine.runtime_status()` health states.

## Core ideas (the why)

- **Two implementations of one idea is the expensive failure.** A backtester version plus a live rewrite means every difference is a bug the backtest cannot find, because the backtest is not running the code that trades. One `Strategy` class, two engines, by construction (`25_live_trading/01_unified_framework_demo`).
- **Parity is a claim about a codebase; only a comparison that could have failed establishes it.** Compare the tape (every signal, field by field), not the summary: two runs can agree on final value and disagree on when they traded. Use assertions, not similarity scores.
- **Parity is checked stage by stage and each stage is a gate, not a report.** A live pipeline can agree on signals yet differ in the feature that produced them, the prediction, or the order size; each failure lives in a different layer. A test that merely prints disagreements "is a test nobody notices failing", so the suite counts its gates and asserts on the count (`08_pipeline_verification`).
- **Importing backtest and live components into one process is itself the first parity test.** If the infrastructure cannot coexist in one environment, the unified-framework claim has already failed.
- **A mismatch is a failure until a separate contract proves otherwise.** Expected differences (feature warm-up) must be declared explicitly, never tolerated implicitly. Skipped is not passed: when the live path cannot execute, emit explicit FAIL records rather than passing against an empty live log.
- **A parity oracle is not a performance estimate.** Same-bar fills at the close remove execution from the comparison; printed returns are not quotable. Two engines agreeing on a bad strategy agree equally well.
- **Signals are not trades.** The strategy emits an order per crossover; the analyzer counts completed round trips. Odd signal count implies one open position at the end; trade count = entries − 1. Any notebook printing both owes the reader that sentence.
- **Research artefact vs deployment artefact.** Hyperparameters cross over from research to deployment; trained weights do not. The deployment fit is a single model on the full extended panel with the narrower feature subset the live path can compute from a broker feed (no walk-forward CV, no per-fold scaling, no sweep), governed on a chapter-local path (`25_live_trading/output/...`) so it cannot drift into the research registry.
- **Pin configurations, not hashes.** A refit re-keys the training hash while selecting the same configuration, so a hash pin fires when nothing changed (`02_etfs_deployment_loop`). Notebook 06 pins the training hash plus predictions SHA256 but deliberately not the registry file digest, which grows with every run.
- **Borrowed hyperparameters are not tuned for the deployed feature set.** Alpha selected on research families (including HMM/GARCH) is only a starting point for the financial-only deployment model; print the dropped families rather than leaving them implicit.
- **Four buckets, not one status: intended / attempted / accepted / failed.** The intended-vs-accepted gap is where a deployment quietly stops matching its research; no equity curve shows it.
- **Broker is the smallest part of the change.** Moving IB to Alpaca swaps a broker class and a feed class; strategy, risk wrapper, order log and engine are the same objects. Venue is an operational decision, not a research one.
- **Check environment, SDK and account state as separate gates.** Missing key = setup problem, missing package = install problem, rejected account = entitlement problem; one combined "connection failed" hides all three.
- **Shadow mode is where a live deployment starts**: real prices, real account state, real strategy, orders logged to `VirtualPortfolio` and never sent. A mock/simulated broker path is NOT shadow mode; shadow mode requires a real broker connection wrapped in `SafeBroker`. Say which parts of a run were real: an offline run proves the strategy interface and order log, and nothing about fills, latency or rejections.
- **Loud-fail over silent fallback.** When TWS is unreachable or the market is closed, print the operator checklist and stop; never substitute historical bars ("a backtest wearing a live disguise"). "A deployment loop that hides its failure modes teaches the wrong lesson."
- **Data plane and execution plane are separate concerns with separate failure modes.** Crypto needs two venues (OKX data, Alpaca execution) because no single retail venue offers both; FX can use one (IB IDEALPRO). A missing OKX response is a different operational event from a missing Alpaca pair.
- **Research universes do not match deployment universes; declare the gap.** 8 of 19 case-study perps are absent from the Alpaca mapping. Fixed, written-out mappings beat live catalogue lookups: a stale mapping fails visibly against a delisted symbol; a silent lookup changes the universe underneath the reader with nothing recording which one ran.
- **Live-trading data comes from the executing broker.** Warm-up bars and pre-submission quotes come from the same session that routes orders; a research-time loader can carry a different cutoff, survivor universe or vendor, so the model would rank names it was never validated on.
- **Managed platforms: move the boundary rather than reimplement the feature pipeline.** The model stays local; predictions cross over as a file; platform code is portfolio rules only. Cost: the platform can only trade dates the file covers, so live use needs a job that keeps writing it.
- **Order management is a controlled-transition problem.** 10 explicit states plus a transition table replace an open/closed flag. `PENDING_CANCEL` turns the cancel-vs-fill race into a normal transition instead of a reconciliation incident. Store the path, not only the final state. Live order handling fails from misunderstood paths far more than bad syntax.
- **Defense in depth.** `SafeBroker` layers independent order-level, portfolio-level, throughput, mandate and persistent-halt controls so no single misconfiguration produces an unchecked order. All are `LiveRiskConfig` parameters, not hard-coded logic.
- **Kill switches must be independent of the component they control and must outlive the process.** A non-persistent kill switch is a flag in memory; "the recovery from a disorderly day is the worst time to discover the difference."
- **Reconciliation is the loop.** A daily rebalance is "diff current against target, then submit only the delta"; the diff table is the audit artefact. `SafeBroker.connect()` is the single place where stale-session damage gets caught; a non-clean report is a failed authentication probe: refuse to launch.
- **Execution cost is a monitoring signal, not a KPI.** Drift between `fill_price` and `last_close` across rebalances is the Chapter 26 monitoring input.
- **Health states are a narrow vocabulary** because operators act on them under stress: `feed_silent` calls for a different response than `broker_disconnected` even though both look like "data stopped" from inside the strategy loop.
- **24/7 crypto is an infrastructure problem as much as a signal problem**: always-on cloud, heartbeat, reconnection, maintenance windows, liquidation risk.

## Method recipes (the how)

### Unified Strategy interface and single-stage parity test (notebook 01)

Every strategy subclasses `Strategy` with `on_start(broker)`, `on_data(timestamp, data, context, broker)`, `on_end(broker)`; the same signature runs in `ml4t.backtest.Engine` and `ml4t.live.LiveEngine`.

1. Load, filter and clean `raw_data` once; both engines read only it (control variable).
2. Backtest: `Engine` reads the long frame (one row per symbol-date). Live: `HistoricalReplayFeed(data, symbols, broker=None)` yields one dict per bar and raises on a missing bar rather than skipping (a dropped symbol-date shortens history and moves the MAs).
3. `SimulatedBroker(initial_cash=100_000.0)` fills every order immediately at the current close; the feed yields finished bars with no delay. This removes execution and timing from the comparison on purpose.
4. Construct the live strategy fresh (no history) and run `LiveEngine(broker, feed)`.
5. Compare: unequal signal count raises before any field comparison; every field of every matched pair must agree within tolerance (epsilon not stated in notes); the `match` column is True only when all fields agree. Comparison frame columns: engine, timestamp, symbol, side, price, fast_ma, slow_ma, match.
6. Demo rule: `DualMAStrategy` with `FAST_MA=10`, `SLOW_MA=30` sessions, buy when fast rises above slow, sell when it falls back below, alternating, 100 shares fixed, SPY only (`MAX_SYMBOLS=0` means all). Result: 9/9 signals identical.

Treat the replayed live path as a parity test, not a rehearsal: `feed_terminated` / `runtime degraded` / `auto recovery disabled` fire at end of replay exactly as in a production outage.

### Staged pipeline verification harness (notebook 08)

| Stage gate | Compares | Bug class it catches |
|---|---|---|
| Feature parity | per-symbol feature log | data / float ordering |
| Prediction parity | prediction log | model versioning, preprocessing drift |
| Signal parity | BUY/SELL/HOLD log | threshold or position-state bugs |
| Order parity | order log (size, side) | sizing and rounding gaps |
| Fill + broker-state parity | fills, cash, positions | execution adapter drift |
| Unique order IDs | each path independently | duplicate submission |
| Declared difference | feature warm-up | explicit contract, not tolerance |

Steps:
1. Write pure, deterministic policy functions: `compute_features(prices, lookback)`, `compute_prediction(features)` (fixed linear rule), `compute_signal(prediction, threshold, has_position)` with HOLD inside the no-trade band, `compute_size(signal, price, cash, position_quantity)` (fixed-fraction entries, exits bounded by current long).
2. `VerifiableStrategy(Strategy)` (`lookback=10`, `threshold=0.02`) logs every intermediate value via `process_symbol(...)` and `fill_record(...)` before submitting.
3. `generate_test_data()` -> `list[tuple[datetime, dict]]`, `N_BARS=30` (longest window warms up and leaves bars to compare), fixed `SEED`, two regimes that force a BUY and a SELL. Write per-symbol offsets as constants (`SYMBOL_OFFSETS`): `hash(str)` is randomised per process unless `PYTHONHASHSEED` is set, which a notebook cannot set for itself.
4. Reference path: `TestDataFeed` + bare in-memory `BacktestBroker` via `run_backtest()`. Candidate path: `LiveBroker(BacktestBroker)` adds the async surface, wrapped in `SafeBroker(LiveRiskConfig, state_path=_temporary_state_path())` + `ThreadSafeBrokerWrapper`, via `run_live()` on the same tape.
5. Compare with `count_result(name, reference, candidate)` first (zip truncates and hides missing rows), then `matching_records(reference, candidate, fields, float_fields=())` (tolerance only on named float fields; identity fields exact), `sequence_result(...)` (fails on empty or unequal logs), `unique_order_ids_result(...)`; compose with `pipeline_results(bt, live)` and `execution_results(bt, live)` inside `run_verification_tests()`, which fails closed via `skipped_results()` when either replay did not run; label with `result_status(result)`.
6. End with a CI-style assertion on the gate count (`test_parity_gate`, mirrors `tests/live/test_parity.py`). Rerun whenever broker wrappers, sizing logic or preprocessing change.

### ETF deployment loop: seven steps (notebook 02, chapter anchor)

Defaults: `EXPECTED_RIDGE_CONFIG="ridge_a1000000.0"` (None = report, not assert), `LIVE_WINDOW_START="2025-01-01"`, `TOP_K=5`, `CASH_BUFFER=0.02`, `REBALANCE_EVERY_N_DAYS=21` (= label horizon), `FORWARD_HORIZON_DAYS=21`, `INITIAL_CASH=100_000.0`, `COMMISSION_RATE=0.0005`, `NOTIONAL_PER_LEG_USD=5_000.0`, `REFRESH_DATA=False`, execution `NEXT_BAR`. Default execution plane: offline dry run.

| Step | Action | Output |
|---|---|---|
| 1 Refresh | `ml4t.data.etfs.ETFDataManager.update()` reads Hive-partitioned parquets, detects last date per symbol, pulls only the missing tail from Yahoo (skipped when `REFRESH_DATA=False`) | `ML4T_DATA_PATH/etfs/market` extended past the 2025-12-31 book cut |
| 2 Features | `25_live_trading/_etfs_features.py` sequences the case study's `compute_*` functions; drops HMM regimes and GARCH (~10 columns) so refit takes seconds | financial-only feature panel (+ FRED yield-curve regime from `ML4T_DATA_PATH/macro`) |
| 3 Alpha | Query `case_studies/etfs/run_log/registry.db` (via `registry_readonly_uri`) for the Ridge config leading on validation IC for `fwd_ret_21d`; `RIDGE_ALPHA = float(fold_alphas.pop())`; assert against `EXPECTED_RIDGE_CONFIG` | alpha = 1e6; printed `source_config_feature_sets` vs `deployed_feature_sets` and dropped list |
| 4 Cutoff + fit | `days_before_live` = trading days before `LIVE_WINDOW_START`; `LABEL_AVAILABLE_AS_OF = days_before_live[-FORWARD_HORIZON_DAYS - 1]`; `FEATURE_CUTOFF_DATE = days_before_live[-1]`; train mask cuts at `LABEL_AVAILABLE_AS_OF` (raise if fewer than the horizon margin of days exist). Single `sklearn.linear_model.Ridge` on the full panel; persisted imputer; feature column order persisted (predict uses positional alignment) | `25_live_trading/output/etfs_deployment/`: `training_metadata.json`, model, imputer, feature column order |
| 5 Predict | Every cross-section from `LIVE_WINDOW_START`; drop all-null rows; remaining nulls through the persisted imputer | one prediction tape feeding both replay and live submission |
| 6 Replay | `CrossSectionalRidgeStrategy(predictions, top_k, rebalance_every, symbols, cash_buffer)` through `Engine(... BacktestConfig(initial_cash, execution_mode=ExecutionMode.NEXT_BAR, commission_rate))`; assert `strategy.rebalance_log` non-empty | offline reference tape; rebalance log records every target basket incl. unchanged holdings |
| 7 Stage | `submit_basket(rows)`: shares = `NOTIONAL_PER_LEG_USD` / ref price, async market order per leg via `AlpacaBroker`; per-leg status `submitted` / `submit_failed` (error captured) / `no_ref_price` / `dry_run` | run JSON: cross-section, offline tape, per-symbol disposition |

`training_metadata.json` keys: `trained_at` (UTC ISO), `data_range{start,end}`, `feature_cutoff_date`, `label_available_as_of`, `forward_horizon_days`, `live_window_start`, `n_train_rows`, `n_features`, `label` (= `PRIMARY_LABEL`), `model_class="sklearn.linear_model.Ridge"`, `ridge_alpha`, `intercept`, `source_case_study="etfs"`, `source_config_name`, `source_config_feature_sets`, `deployed_feature_sets`.

Run record keys: `run_started_at`, `live_window`, `data_range`, `feature_cutoff_date`, `label_available_as_of`, `forward_horizon_days`, `artefact_dir`, `config{ridge_alpha, top_k, rebalance_every_n_days, primary_label}`, latest cross-section, offline tape, execution disposition (four buckets).

Sizing and reconciliation: size the basket on the broker's current account value × (1 − `CASH_BUFFER`), never on `INITIAL_CASH`. Reconcile by `_normalise(records)` (flatten the signal log to sorted `(date, symbol, side, share delta)` tuples) and compare the intended basket on `latest_ts` against the strategy's recorded target basket on the same scheduled date. The broker session opens only when refreshed data AND explicit order opt-in AND Alpaca paper credentials all agree.

### Interactive Brokers stack (notebook 03) and daily basket rebalance (notebook 12)

Connection: host `127.0.0.1`, port 7497 (TWS paper) or 4002 (Gateway paper); API enabled (Edit > Global Configuration > API > Settings), "Enable ActiveX and Socket Clients" checked, Read-Only API off, loopback trusted IP; CLIENT_ID 10 (nb 03), 11 (nb 11), 12 (nb 12) so notebooks run back-to-back without socket conflicts. `IB_ACCOUNT` env var or None for broker default. `MARKET_DATA_TYPE`: None keeps the session setting (right when the paper account has live Level 1); 1 real-time, 2 frozen, 3 delayed, 4 delayed-frozen; the notebooks request 3.

Notebook 03 order of operations:
1. Import check (`ib_async`, `ml4t.live`) before any connection attempt, so a missing package fails differently from an unreachable TWS.
2. `connect_to_ib()` prints host, port, account, client ID and the account summary (NLV, cash, positions). Unreachable: print checklist and stop.
3. `get_historical_data(symbol, days=5)` via `reqHistoricalData` for deterministic indicator warm-up; print the exact bars used.
4. `MomentumStrategy(lookback=5, threshold=0.02)` (identical class to the backtest; sign handling is an inference).
5. `SafeBroker(IBBroker, LiveRiskConfig(...))` in shadow mode with position/order value caps, rate limit, kill switch, persisted `RiskState`; `safe_broker.connect()` reconciles.
6. `_is_rth_now("America/New_York")` gate; `build_live_stack() -> (BarAggregator, LiveEngine)` wires `IBDataFeed` ticks -> minute bars -> engine.
7. `run_live_demo(duration_seconds=30)`: fixed-duration RTH run proves connectivity without an unattended loop.
8. `fetch_ib_snapshot(symbol) -> mid`: delayed top-of-book snapshot recorded on `SafeBroker` before each order; `demonstrate_order_submission()` sizes SPY/QQQ under `max_order_value` so the demo shows a virtual fill, not a cap rejection.
9. `run_shadow_workflow()`: one outer cleanup boundary; explicit disconnect.

Notebook 12 daily rebalance: `UNIVERSE` of 20 US large caps (AAPL MSFT GOOGL AMZN META NVDA JPM V JNJ PG UNH HD DIS MA BAC KO PEP XOM CVX WMT), `TOP_K_LONG=5`, `TARGET_NOTIONAL_USD=50_000`, `MAX_POSITION_USD=15_000`, `MAX_ORDER_USD=12_000`, `MAX_DAILY_LOSS_USD=5_000`, `WARMUP_DAYS=60`, `OUTSIDE_RTH=False`, `MARKET_DATA_TYPE=3`, `SUBMIT_PAPER_ORDERS=False`, state file `OUTPUT_DIR / "risk_state.json"` with `OUTPUT_DIR = get_output_dir(25, "ib_basket_rebalance_demo")`.

1. `open_ib_session()` fails loudly with setup instructions, never mocks.
2. `SafeBroker(ib, LiveRiskConfig(...), state_path=STATE_FILE)`; `connect()` diffs the persisted snapshot vs broker positions/pending orders; non-clean report => stop.
3. `fetch_warmup_bars(ib_app, universe, days)` via `asyncio.gather` over `fetch_one_symbol_bars` (returns [] when IB has no data); exclude the current UTC date (bar may still be forming); reconcile against raw counts.
4. `compute_target_basket(bars_frame, top_k, notional)` -> `symbol, signal, target_qty` from a 20-day momentum proxy (acknowledged not research-grade).
5. `fetch_current_positions(active_broker, universe)` (every universe name represented) -> `reconcile(current, target_frame)` -> `delta_qty = target_qty - current_qty`; checkpoint the delta table before submission.
6. Seed the `SafeBroker` price cache with the last completed close per symbol (it rejects orders with no recent snapshot).
7. `dry_run_basket(basket)` or `submit_basket(active_broker, basket)` concurrently through `submit_one_basket_order` (close passed as risk-check hint; IB receives a market order).
8. `wait_for_basket_to_settle(active_broker, expected_symbols, timeout_s=15.0)` -> re-run `reconcile` -> `execution_summary(submissions, raw_positions)` (slippage vs `last_close`, aggregated vs notional).
9. `submit_eod_flatten(active_broker, symbols_with_qty)`: one `OrderType.MOC` per position opened by this run, opposite side; `confirm_orders_accepted(active_ib, expected_count, timeout_s=5.0)` waits until status != `PendingSubmit` before teardown.

A continuous-loop deployment replaces the close-seeded price cache with a streaming `IBDataFeed` driven by `LiveEngine` and makes the MOC step conditional on the next-rebalance decision (hold if still ranked long, flatten otherwise).

### Alpaca equities (notebook 04) and crypto spot (notebook 05)

Notebook 04 order of operations:
1. If `ML4T_HEADLESS_PAPERMILL=1`, set `LIVE_FEED=0` and print why (Alpaca WebSocket loop is incompatible with `nest_asyncio`; `asyncio.wait_for` cannot cancel the inner streaming task).
2. `verify_credentials()`: SDK presence and `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` checked separately; execution mode printed.
3. `get_alpaca_account_snapshot()`: cash/equity/positions via REST without starting the streaming session.
4. `ETFMomentumStrategy(lookback=5, threshold=0.02, position_size=10)` on SPY, QQQ, IWM; `signals_to_frame` -> Polars log.
5. `create_safe_broker(underlying)` prints the active `LiveRiskConfig` limits before the feed starts.
6. `AlpacaDataFeed`: channels bars (1-minute default), quotes (bid/ask with sizes), trades; sources IEX (free, 15-min delayed for some symbols) vs SIP (premium, all exchanges).
7. `create_alpaca_engine(strategy)` -> `AlpacaDataFeed > SafeBroker > LiveEngine`; `run_engine_for_duration(engine, duration_s)`; `display_engine_results(strategy, safe_broker, feed, engine)`; `run_live_demo_with_feed()` dispatches live vs `run_simulated_demo()`.
8. `demonstrate_order_types()`: MARKET, LIMIT, STOP, STOP_LIMIT via `OrderType` in shadow mode (skipped offline).
9. Explicit disconnect.

Offline path: `MockBroker(initial_cash=100_000.0)` portfolio `{"cash": float, "positions": {symbol: {qty, entry_price}}}`, fixed-seed bars, per-order status `filled` / `rejected` / `unsupported` recorded from the outcome. This proves the interface and log only.

Notebook 05 crypto: map the 19-perp Binance USDT universe to Alpaca USD spot (11 tradeable; ADA, APT, ATOM, BNB, COMP, INJ, NEAR, SUI not; coverage computed at run time). `CryptoPremiumStrategy(lookback=10, entry_threshold=1.5, exit_threshold=0.25, position_size=0.1)`: `_compute_momentum_zscore(prices, lookback)` = (latest close-to-close return − rolling mean) / rolling stdev; `_route_signal` enters long-only spot when |z| crosses 1.5 (mean-reversion direction) and exits when |z| < 0.25 (exact direction is an inference). `_is_funding_hour(ts)`: naive timestamps treated as UTC, aware converted to UTC, compared against Binance funding hours 00:00/08:00/16:00 UTC. Simulated path `MockCryptoBroker(initial_cash=10_000)` is not shadow mode. Production wiring (not implemented): premium = `(perp − spot)/spot`, rolling z-score.

### Split-venue crypto funding deployment loop (notebook 09)

Six steps: Connect (OKX public REST data plane, no key, + Alpaca paper execution plane) -> Train -> Persist -> Predict -> Trade -> Persist run JSON. Fail at Connect if any venue is unreachable.

| Component | Setting |
|---|---|
| Universe | 19 perps (`CASE_STUDY_UNIVERSE`); `OKX_INSTRUMENT[sym] = f"{sym[:-4]}-USDT-SWAP"`; fixed `ALPACA_USD_PAIR` map of 11 USD spot pairs (USD, not USDT: paper accounts are funded in USD) |
| Bars | `fetch_okx_candles_1h(inst_id, limit=300)` (OKX v5 supports 1H/4H/1D for SWAP, not 8H; limit capped at 300; use 250 -> ~31 complete 8H groups + buffer); `aggregate_to_8h(bars_1h)` aligned to UTC 00/08/16, labelled by window START (the funding settlement at which the decision is made) |
| Funding | `fetch_okx_funding_history(inst_id, limit=100)`; per-symbol BACKWARD as-of join of the latest print at or before each bar; drop symbols whose print is older than `MAX_FUNDING_AGE_HOURS=8.0` against the 8H grid |
| Audit | `audit_okx_candle_payload`, `audit_okx_funding_payload`: raw payload hash + raw -> confirmed -> parsed -> aggregated conservation counts; funding rows conserved with unique publication timestamps; single-symbol connectivity probe before the full cross-section |
| Coverage | gate on `MIN_OKX_LIVE_COVERAGE=0.75` of the 19-perp universe; retired OKX instruments recorded |
| Features | 13 at 8H cadence via `compute_features_8h(panel)`: premium shape, returns, volatility, RSI, VWAP distance from `rolling_zscore_expression(column, alias, window=21)`, `crypto_rsi_expression(window=14)`, `vwap_distance_expression(window=3)` (3 bars = 24h); returns before volatility; needs >= 21 complete 8H bars; serialized `FEATURE_COLS` order is the model input contract |
| Model | LightGBM 3-class on `fwd_dir_8h_3c` ({down, flat, up} -> {0, 1, 2}); `TRAIN_END_DATE="2024-12-31"`, `NUM_BOOST_ROUND=200`, `LEARNING_RATE=0.05`, `NUM_LEAVES=31`, `NUM_THREADS=4`, `SEED=42`, `TRAIN_DEVICE="cpu"`, `deterministic=True`, `force_col_wise=True`; scaler fit on training rows only; metadata records temporal boundary, device, seed, research->live data-source change |
| Decision | `P(up) >= PROB_LONG_THRESHOLD` -> long; `P(down) >= PROB_SHORT_THRESHOLD` -> short; both -> larger tail wins; edge display = P(up) − P(down) (inference); threshold literals not captured |
| Execution | `classify_crypto_intent(row)` -> {flat intent, no mapping (signal-only), unsupported spot short, dry run, no_credentials, submitted}; `submit_crypto_intent(row)` reaches the broker only for an authorized LONG; `submit_basket(intents)` opens Alpaca only when credentials AND `SUBMIT_PAPER_ORDERS=True`; `NOTIONAL_PER_LEG_USD=100.0` |
| Output | `get_output_dir(25, "crypto_funding_deployment")/runs/*.json`: model fingerprint, predict cross-section, per-symbol disposition |

Not a faithful PnL reproduction of the funding strategy: spot positions receive no funding flow, and USD vs USDT quoting is a declared friction.

### FX deployment loop on IB IDEALPRO (notebook 11)

Single venue for both planes. Universe: 20 majors/crosses `AUD_JPY` .. `USD_JPY`; `IB_SYMBOL = sym.replace("_", "")`; `qualify_forex_contracts()` resolves `Forex(symbol)` on IDEALPRO once (6-char localSymbol) and caches on the broker so `submit_order_async` resolves them; missing contracts stay visible and out of the cross-section.

1. `open_ib()`: `IB_HOST="127.0.0.1"`, `IB_PORT=7497` (Gateway 4002), `IB_CLIENT_ID=11`, `IB_CONNECT_TIMEOUT_S=30.0` (a gateway can accept the socket and never finish position/order sync); reject any managed account that is not a paper account.
2. `reqHistoricalDataAsync` with `IB_HISTORICAL_DURATION="60 D"`, fanned out with `asyncio.gather` (`fetch_all_pairs()` / `safe_fetch`); `audit_ib_bars(symbol, bars)` hashes and counts raw bars with duplicate-date diagnostics; keep only sessions strictly before the current UTC date.
3. `compute_features_daily(panel)`: 8 daily features in trading-day windows: returns first, then volatility and short/long volatility ratio, momentum, `daily_rsi_expression(window=14)`.
4. Ridge, `RIDGE_ALPHA=1.0`, `TRAIN_END_DATE="2024-12-31"` (`TRAIN_CUTOFF_EXCLUSIVE` = next day); persist model, scaler, feature schema, metadata so inference validates inputs before scoring. Labels: `case_studies/fx_pairs/labels/fwd_ret_1d.parquet`; data `ML4T_DATA_PATH/fx_pairs` via `data.load_fx_pairs`.
5. Decision: filter predicted return > 0 FIRST, then `TOP_K=5`; long-only in the demo (production would also short bottom-K; IB Forex supports shorts).
6. Sizing: `BASE_QTY_PER_LEG=20_000.0` base-currency units, `round_to_step(qty, step=1000.0)` (IDEALPRO 1k step on majors). Fixed base currency avoids the cross-rate conversion a USD-notional target would need (EUR/USD quoted in USD vs EUR/JPY in JPY).
7. `submit_fx_intent(row)` -> audit record (IB symbol, reference price, status, error); `submit_basket(intents)` processes in rank order deterministically; `close_ib()`. Output `get_output_dir(25, "fx_deployment")/runs/`.

### Prediction bridge to QuantConnect / LEAN (notebook 06)

1. Select holdout (never validation) predictions for the pinned `EXPECTED_TRAINING_HASH` from the ETFs registry: 46,466 rows, 95 symbols × 497 dates (2024-01-02 to 2025-12-23).
2. Serialize as a list of `{date, prediction_by_symbol}`; pin `EXPECTED_PREDICTIONS_SHA256`; do not pin the registry file digest.
3. Upload: `qb.object_store.save('research-to-backtest-factors.json', json_str)` (cloud path: free account -> clone illustrative project -> upload -> run backtest; local path: LEAN CLI via Docker).
4. LEAN side (~30 lines): `PredictionUniverse(PythonData)` with `get_source` -> `SubscriptionDataSource(file, SubscriptionTransportMedium.OBJECT_STORE, FileFormat.UNFOLDING_COLLECTION)` (streams one date at a time, no lookahead) and `reader` building `BaseDataCollection`; algorithm `initialize/_select_assets/_rebalance` subscribes to names with prediction above `PREDICTION_THRESHOLD` (0 = all positive forecasts; raise to narrow breadth), equal-weights, and rebalances at a cadence derived from the promoted configuration's label horizon (5-session forecast -> weekly).

| Aspect | Precomputed predictions | Inline inference |
|---|---|---|
| Pipeline | Train offline, export JSON, read predictions | Train offline, serialize model, call `predict()` |
| Iteration speed | Threshold/weight changes reuse frozen scores | Model changes rerun inference |
| Freshness | Frozen at export time | Always current |
| Reproducibility | Same JSON under versioned engine rules | Depends on model + runtime versions |
| Best for | Portfolio-rule experiments, walk-forward | Live trading with streaming data |
| Book pipeline | Chs 7–15 -> export -> Ch25 QC notebook | Model loaded in `Initialize()` |

### Order lifecycle state machine (notebook 07)

States (10): `PENDING_NEW` (submitted, awaiting ack), `NEW` (acknowledged), `ACCEPTED` (exchange accepted), `PARTIALLY_FILLED`, `PENDING_CANCEL` (cancel in flight), `FILLED`, `CANCELED`, `REJECTED`, `EXPIRED` (time-in-force), `SUSPENDED` (halt). Events (9): `ACKNOWLEDGE, ACCEPT, PARTIAL_FILL, FILL, CANCEL_REQUEST, CANCEL_CONFIRM, REJECT, EXPIRE, SUSPEND`.

| From | Event -> To |
|---|---|
| PENDING_NEW | ACKNOWLEDGE -> NEW; REJECT -> REJECTED |
| NEW | ACCEPT -> ACCEPTED; REJECT -> REJECTED; FILL -> FILLED; CANCEL_REQUEST -> PENDING_CANCEL |
| ACCEPTED | PARTIAL_FILL -> PARTIALLY_FILLED; FILL -> FILLED; CANCEL_REQUEST -> PENDING_CANCEL; EXPIRE -> EXPIRED; SUSPEND -> SUSPENDED |
| PARTIALLY_FILLED | PARTIAL_FILL -> PARTIALLY_FILLED; FILL -> FILLED; CANCEL_REQUEST -> PENDING_CANCEL |
| PENDING_CANCEL | CANCEL_CONFIRM -> CANCELED; FILL -> FILLED (fill wins the race); REJECT -> ACCEPTED (cancel rejected, back to active) |
| SUSPENDED | ACCEPT -> ACCEPTED (resume); CANCEL_REQUEST -> PENDING_CANCEL |
| FILLED, CANCELED, REJECTED, EXPIRED | none (terminal) |

19 edges (2+4+5+3+3+2). Properties: `is_terminal` = {FILLED, CANCELED, REJECTED, EXPIRED}; `is_active` (can fill) = {NEW, ACCEPTED, PARTIALLY_FILLED}; `can_cancel` per state; `SUSPENDED` is non-terminal with an unusual exit profile (whether it is reachable only from ACCEPTED is per the notebook; book Figure 25.2 may differ).

`Order.apply_event(event, metadata)`: look up the edge, raise on an illegal pair, validate fill metadata with `_validate_fill` (requires `qty` and `price`; returns None for non-fill events) BEFORE mutating (atomic guard: state, history, filled_qty unchanged on failure), append a `StateTransition` to `history`, update fills, then call `on_transition`. Weighted-average fill price `avg = (prior_avg * filled_qty + price * qty) / (filled_qty + qty)`; helpers `remaining_qty()`, `fill_pct()`; `__post_init__` rejects non-finite or <= 0 quantity with `ValueError`; `get_valid_events(state)` answers whether an incoming callback is expected, delayed, or structurally impossible (reconciliation aid). Replace/modify flows and durable, idempotent persistence are production scope in `ml4t.live.safety`.

### SafeBroker defense in depth (notebooks 10, 13)

`LiveRiskConfig` fields and defaults (`libs/src/ml4t_live/ml4t/live/safety.py`, class at ~line 76):

| Layer | Field | Default |
|---|---|---|
| Order-level | `max_order_value` / `max_order_shares` | 10_000.0 / 500.0 |
| Portfolio-level | `max_position_value` / `max_position_shares` | 50_000.0 / 1000.0 |
| Portfolio-level | `max_total_exposure` / `max_positions` | 200_000.0 / 20 |
| Throughput | `max_orders_per_minute` / `dedup_window_seconds` | 10 / 1.0 |
| Loss | `max_daily_loss` / `max_drawdown_pct` | 5_000.0 / 0.05 (drawdown enforcement not demonstrated) |
| Price sanity | `max_price_deviation_pct` / `max_data_staleness_seconds` | 0.05 / 60.0 |
| Mandate | `allowed_assets` / `blocked_assets` | `set()` / `set()` |
| Mode | `shadow_mode` / `execution_mode` | False / None |
| Halt | `kill_switch_enabled` / `allow_reducing_risk_when_killed` / `halt_on_reducing_risk_failure` | False / True / True (semantics inferred from names) |
| Recovery | `fail_on_reconciliation_mismatch` / `state_file` / `journal_file` / `fail_on_journal_error` | False / `.ml4t_risk_state.json` / None / True |

Any limit set to `None` disables that check (inference from `float | None` typing). Persisted `RiskState` (atomic JSON writes): `daily_loss=0.0`, `orders_placed=0`, `high_water_mark=0.0`, `session_start_equity=None`, `persisted_positions`, `persisted_pending_orders`, `portable_strategy_state`, `replacement_gaps`, `shadow_portfolio`, `execution_mode`, `kill_switch_activated=False`, `kill_switch_reason=""`.

Runtime behaviours:
- Stale data: `SafeBroker` keeps a `MarketSnapshot` per asset from engine bars; every order intent compares snapshot age to `max_data_staleness_seconds`; older -> `RiskLimitError` carrying measured age and threshold; no retry, no stale-price fallback (nb 13 demo: threshold 1 s, `STALE_DATA_WAIT_SECONDS=1.5`).
- Daily-loss kill switch: every submission reads account value; first read anchors `session_start_equity`; loss = max(0, anchor − current); loss > `max_daily_loss` latches the kill switch (reason persisted) and rejects; reconstruction from the state file keeps the latch and rejects before any downstream check (nb 13: `KILL_SWITCH_LOSS_USD=1_000.0` on `DemoBroker(account_value=100_000.0)`).
- Startup reconciliation: `connect()` emits `reconciliation_report` with `missing_positions` (persisted but broker lacks: manual flatten, after-hours fill, corporate action), `unexpected_positions` (broker has, state lacks: fill after last persist, out-of-band position) and order disagreements. Resolve by investigating in the broker GUI and re-running, or clearing persisted state when known benign.
- Rate limit: nb 10 burst rejected on the 4th order while each passes every other check (demo cap 3/min, inference). Each nb 10 scenario uses its own temp state path `_temporary_state_path(prefix="nb10_state_")`.
- Shadow mode: orders logged, `VirtualPortfolio` updated (weighted-average cost basis, partial exits, position flipping) while the real broker stays flat; prevents infinite buy loops.
- Engine health: `LiveEngine.runtime_status()["health"]` in {`stopped`, `waiting_for_data`, `ok`, `feed_silent`, `idle_market_closed`, `broker_disconnected`}; a feed declaring no equity symbols is treated as a continuous market; a short feed-silence timeout flips to `feed_silent`; `HEALTH_OBSERVATION_SECONDS=6`. Poll in a bounded window and dedupe consecutive identical states.
- Operator surface: `ml4t-live status|shadow` CLI inspects the persisted state file out of process.

### Rollout ladder

1. Shadow mode (`shadow_mode=True`, real broker connection) for 1–2 weeks; verify signals match backtest expectations.
2. Paper (`shadow_mode=False`) with the SAME `SafeBroker` configuration; monitor 2–4 weeks; watch the state file for kill-switch activations.
3. Live gradually with small positions and conservative limits.

## Guardrails and pitfalls

- **Two-pipeline divergence** — a live rewrite differs from the backtest in a detail nobody noticed until money was on it / one `Strategy` class, two engines; run both on the same tape and assert field-by-field equality (nb 01, 08).
- **Comparing summaries instead of tapes** — equal final value can hide different trade timing / build the comparison frame per signal; raise on unequal counts before comparing fields.
- **Signal-only parity** — a live path can match signals yet differ in features, predictions or sizes / gate each stage separately and assert on the gate count in CI.
- **Vacuous parity pass** — empty or skipped logs trivially "match" / `sequence_result` fails on empty or unequal logs; `skipped_results()` emits FAIL for every gate when a replay did not run; the tape must force BUY and SELL.
- **Count-hiding by zip** — zip truncates to the shorter list / `count_result` before value comparison.
- **Float tolerance everywhere** — blanket tolerance masks real drift / tolerance only on named `float_fields`; identity fields exact.
- **Non-reproducible tape via `hash()`** — string hashing is randomised per process unless `PYTHONHASHSEED` is set / write per-symbol offsets as constants; seed every generator.
- **Feed silently dropping bars** — shorter history shifts moving averages and misreports as engine disagreement / replay feed raises on a missing bar.
- **Mistaking feed end for feed fault** — `feed_terminated` / `runtime degraded` fires at end of replay and in production outages alike / do not build monitoring on that log line for backfills; treat replay as a parity test, not a rehearsal.
- **Label lookahead at the live boundary** — `fwd_ret_21d` at t reads prices at t+21, inside the live window if training cuts at `LIVE_WINDOW_START − 1` / cut at `LABEL_AVAILABLE_AS_OF = days_before_live[-H-1]`; persist both cutoffs in `training_metadata.json`.
- **Incomplete current bar** — today's daily bar may still be forming; using it leaks intra-session data into features and sizing / keep only sessions strictly before the current UTC date (nb 11, 12) and reconcile the exclusion against raw counts.
- **Borrowed hyperparameter as an unchecked literal** — the registry leader can move on the next sweep / query the registry for the leader and assert against `EXPECTED_RIDGE_CONFIG`; pin configuration names, not training hashes.
- **Alpha tuned on a different feature set** — deployment drops HMM/GARCH; alpha chosen elsewhere is not tuned for this model / print `source_config_feature_sets` vs `deployed_feature_sets` and the dropped list; re-sweep on the deployment feature set if quality matters.
- **Research-artefact reuse** — registry weights were fit on a wider feature matrix the live path cannot compute / separate deployment fit with the live-computable subset; inherit hyperparameters only.
- **Deployment artefact leaking into the research registry** — mixes CV-evaluated research runs with single-fit deployment runs / write to chapter-local `25_live_trading/output/...`.
- **SQLite `immutable=1` on a live WAL registry** — ignores the -wal file, showing a stale leader or a missing table / `registry_readonly_uri` adds `immutable=1` only when the directory is unwritable; `mode=ro` otherwise.
- **Column-order drift between train and predict** — predict relies on positional alignment / persist feature column order (`FEATURE_COLS`) in the artefact as a contract.
- **Scaler leakage** — fitting the scaler on validation or live rows leaks their mean/scale / fit on training rows alone; persist the scaler with the model.
- **Cross-environment booster hashes** — `deterministic` + `force_col_wise` fix histogram order only within a pinned environment; CUDA builds differ between their own runs / verify by comparing predictions, never artefact hashes; train on CPU with fixed `SEED=42`.
- **Sizing at full account value** — an order sized on today's close fills next bar; an overnight adverse move plus commission makes the last leg unaffordable (8 refusals observed before the buffer) / `CASH_BUFFER=0.02`; size on current account value; check disposition buckets and stop on refusals. A ~2% overnight gap still breaks it.
- **Reading a refused order as part of the reference tape** — the replay would misstate what the deployment did / only filled orders count as accepted; unfilled-on-last-bar under `NEXT_BAR` is counted separately (ran out of tape, not a refusal).
- **Collapsing basket outcome into one status** — intended 5 / accepted 3 is a two-name divergence invisible in the equity curve / record intended/attempted/accepted/failed separately; alert on `failed_basket`.
- **Comparing a scheduled rebalance with an arbitrary daily cross-section** — not a parity test / reconcile on the same scheduled date using the rebalance log (which includes unchanged holdings).
- **Non-deterministic signal ordering** — two runs over the same window would not compare / `_normalise` sorts tuples and reduces timestamps to date strings.
- **Silent fallback / mock substitution when a broker is unreachable** — the notebook reads as if it traded, hides outages / print the operator checklist and stop; RTH gate `_is_rth_now`; separate import check from connection check.
- **Mock path mistaken for shadow mode** — shadow mode implies a real broker connection with virtual routing; a mock proves nothing about the venue / label the mode at every step; only the credential-present path uses `SafeBroker`.
- **Wrong account or session** — invisible until something is submitted to the wrong place / print host, port, account, client ID at connect; `IB_ACCOUNT`; unique CLIENT_ID per notebook; assert paper account.
- **Hanging IB connect** — gateway accepts the socket but never completes position/order sync / bound with `IB_CONNECT_TIMEOUT_S=30.0`.
- **Indicators sized from partial warm-up** — first minutes trade on half-filled indicators after reconnect or open / deterministic warm-up via `reqHistoricalData`; print the exact bars used.
- **Research-time loader for live warm-up** — different cutoff, survivor universe or vendor; model ranks names it was never validated on / fetch warm-up bars and reference prices from the executing broker session.
- **Unparsed broker payloads** — silent row loss or duplicates at the venue boundary / hash and count raw payloads (`audit_okx_*`, `audit_ib_bars`) and require raw/confirmed/parsed/aggregated conservation.
- **Paper account lacks market data** — MARKET orders rejected "No market data available..." / `MARKET_DATA_TYPE=3` (delayed) or 4; `None` when live Level 1 exists.
- **Research-time or mock prices on live orders** — operational discipline is price-before-order / pull a real snapshot quote and record it on `SafeBroker` before every order.
- **No cached price snapshot** — `SafeBroker.submit_order_async` rejects orders with no recent price / seed with the last completed close (one-shot) or stream via `IBDataFeed` + `LiveEngine`.
- **Stale sessions / dangling subscriptions** — the next run inherits broken state / explicit disconnect inside one outer cleanup boundary.
- **Stale session state at launch** — after-hours pending orders, manual GUI flatten, partial fill after last persist / `SafeBroker.connect()` reconciliation report; refuse to launch a cycle until clean or explicitly cleared.
- **Position drift compounding** — without reconciliation every divergence persists / diff current vs target each rebalance; submit only `delta_qty`; checkpoint the delta table before submission so a crash can resume.
- **Fills are asynchronous** — `submit_order_async` returns on queueing; immediate position reads miss fills / `wait_for_basket_to_settle(timeout_s=15.0)` then re-reconcile; read per-order status beside the residual delta.
- **Overnight exposure from a daily basket** — positions opened at the start of day persist / `OrderType.MOC` per opened leg; exchange cutoffs, halts and rejections still apply.
- **Fast teardown races order acceptance** — queued MOC orders never reach the closing auction / `confirm_orders_accepted(timeout_s=5.0)` until status != `PendingSubmit` before disconnect.
- **TWS auto-cancel on disconnect** — `Global Configuration -> API -> Settings -> Auto-cancel API orders on disconnect` defaults enabled on most paper accounts, cancelling queued DAY/MOC orders / reconcile that setting before relying on queued orders.
- **Alpaca WebSocket + `nest_asyncio` hang** — `asyncio.wait_for` cannot cancel the inner streaming task; an unattended run never finishes / under `ML4T_HEADLESS_PAPERMILL=1` force `LIVE_FEED=0` and say so in output.
- **Logging fills blindly** — hides rejections / record `filled` / `rejected` / `unsupported` from the outcome.
- **Credentials in publication runs** — accidental mutation / load env credentials only when `SUBMIT_PAPER_ORDERS=True`; missing credentials produce an explicit `no_credentials` disposition.
- **Funding-hour check in the wrong timezone** — misses every window with no error / naive -> treat as UTC, aware -> convert to UTC before comparing to 00/08/16 UTC.
- **Forward or nearest funding join** — puts a funding print into a bar that preceded it, "a small, invisible piece of the future" / per-symbol backward as-of join only.
- **Null check cannot detect dead feeds** — the as-of join carries the last print forward indefinitely, so `funding_rate` and `premium` are never null once one print arrived / check the age of the matched print; drop symbols older than `MAX_FUNDING_AGE_HOURS=8.0`.
- **Universe mismatch / silent drift at the venue boundary** — 8 of 19 perps are untradeable on Alpaca spot; spot positions receive no funding; live catalogue lookups change the universe per run / explicit mapping table; record unavailable instruments; gate on `MIN_OKX_LIVE_COVERAGE=0.75`; log signal-only names.
- **Quote-currency mismatch** — Alpaca paper auto-funds USD, not USDT; USDT-pair orders fail with insufficient balance / use USD pairs; treat spot/USD vs perp/USDT as a declared friction, not PnL reproduction.
- **Top-K without sign filter** — in a risk-off cross-section every forecast is negative and top-K becomes a basket the model expects to lose / filter predicted return > 0 before the top-K cut.
- **Unqualified FX contracts** — `AUD_JPY` is not an IB contract / qualify `Forex(symbol)` on IDEALPRO once, cache on the broker, keep missing contracts visible and out of the cross-section.
- **USD-notional FX sizing** — quote currencies differ so a uniform USD target is ambiguous / fixed base-currency quantity per leg rounded to the 1k step.
- **Reimplementing features in the platform language** — second implementation = divergence / prediction-bridge export; platform holds only portfolio rules.
- **Pinning the registry file digest** — fires whenever any run is appended / pin training hash + predictions SHA256 only.
- **Exporting validation predictions** — selection bias; the backtest measures the selection / export holdout rows.
- **Rebalance cadence typed as a constant** — selection may move to a different horizon and the schedule silently mismatches (the notebook once said "monthly" in prose while deploying a 5-session signal) / derive cadence from the promoted configuration's label horizon.
- **Platform only trades dates in the file** — frozen predictions go stale / schedule a job that keeps writing the file, or inline inference for streaming.
- **Marking an order canceled on `CANCEL_REQUEST`** — disagrees with broker books when a fill wins the race / explicit `PENDING_CANCEL` with CANCEL_CONFIRM / FILL / REJECT exits.
- **Open/closed boolean order flags** — not every visible order can be modified or canceled / state-aware `is_terminal` / `is_active` / `can_cancel`; reject illegal transitions with an exception.
- **Storing only final order state** — `CANCELED` says nothing about whether it worked for an hour (broker vs strategy fault) / keep the full transition log and transition count.
- **In-memory audit trail in production** — a crash loses history; replayed broker events double-count / durable persistence + idempotent dedup (`ml4t.live.safety`).
- **Fill metadata applied before validation** — partial mutation on a bad event / `_validate_fill` returns before any state change.
- **Single oversized order** — one fat-finger order wipes the session / `max_order_shares`, `max_order_value` reject before the broker sees the request.
- **Sequence of reasonable orders** — individually valid orders aggregate to an unsafe position / `max_position_value`, `max_position_shares`, `max_total_exposure`, `max_positions`.
- **Runaway loop / broker rate ban** — a duplicated signal's repeated small errors are as destructive as one big one / `max_orders_per_minute`, `dedup_window_seconds=1.0`.
- **Symbol-selection bug routes to a banned instrument** — breaks the mandate / `allowed_assets` / `blocked_assets` enforced before the broker.
- **Trading through a feed outage on the last known price** — "how a stop-loss strategy turns into its own opposite" / `max_data_staleness_seconds` per asset on every intent; no retry; stand down.
- **Kill switch that dies with the process** — restart silently reactivates a halted strategy / latch in persisted `RiskState` (atomic JSON); reconstruction keeps the latch; operator clears explicitly.
- **Inconsistent error reporting** — incidents hard to diagnose under time pressure / one helper (`run_demo(coro, expect_error)`) and one error type (`RiskLimitError`) with measured value + threshold.
- **Manual parity checks** — teams skip them under time pressure / automate as a CI gate (`test_parity_gate`); rerun whenever broker wrappers, sizing or preprocessing change.
- **Crypto 24/7 assumptions borrowed from equities** — no session boundary, maintenance windows, liquidation risk / always-on cloud, heartbeat monitoring, reconnection, tighter stops.
- **Relying on a replayed parity test for live behaviour** — no latency, partial fills, rejections, disconnects / nb 07, 08, 12 cover those; nb 10/13 add broker-side controls.

## Decision rules and defaults

- Deployment-loop sequence: Connect -> Train -> Persist artefacts -> Predict live cross-section -> Trade/Plan -> Persist run JSON. Fail at Connect if any venue is unreachable.
- Three separate gates before a strategy touches a broker: environment (keys), SDK (import), account state (snapshot sane). The broker session opens only when refreshed data AND explicit order opt-in (`SUBMIT_PAPER_ORDERS=True`) AND paper credentials all agree; default is offline dry run.
- Launch go/no-go: reconciliation report clean (or persisted state explicitly cleared) AND kill switch not latched AND price snapshot fresh within `max_data_staleness_seconds` -> may submit; otherwise refuse.
- Parity go/no-go: all gates PASS; any skipped replay => FAIL; any undeclared mismatch => FAIL.
- Training cutoff: `LABEL_AVAILABLE_AS_OF = days_before_live[-H-1]`, never `LIVE_WINDOW_START − 1`. Keep only sessions strictly before the current UTC date for live warm-up.
- Set `EXPECTED_RIDGE_CONFIG` / `EXPECTED_TRAINING_HASH` / `EXPECTED_PREDICTIONS_SHA256` to `None` only when the case study is deliberately re-swept or re-promoted; otherwise assert.
- Rebalance cadence = label horizon (5-session forecast -> weekly; 21-day label -> every 21 trading days).
- Size on current account value × (1 − `CASH_BUFFER`); `CASH_BUFFER=0.02`.
- Rank-and-size: filter predicted return > 0, then top-K; long-only in demos.
- Venue split: one venue for both planes when it exists (FX on IB IDEALPRO); otherwise split data (OKX public) from execution (Alpaca paper) and track failures per plane.
- Crypto cadence: hourly fetch (limit 300, use 250) -> 8H windows at UTC 00/08/16 labelled by window start; >= 21 complete 8H bars; funding age <= 8.0 h; live coverage >= 0.75 or stop. Funding hours (Binance) 00:00/08:00/16:00 UTC; hold-across-window decisions belong in execution.
- Rollout: shadow 1–2 weeks -> paper (`shadow_mode=False`) 2–4 weeks with the same `SafeBroker` config -> live gradually with small positions.
- Residual-delta triage: rejected status or large `fill_price` vs `last_close` gap -> investigate before the next rebalance; residual with a risk-cap or kill-switch status -> the control fired, not a bug.
- Health-state response: `feed_silent` -> investigate feed / widen window / halt; `broker_disconnected` -> reconnect; `idle_market_closed` -> expected. Stale-data response options carried in the error: widen the staleness window, investigate the feed, or halt; never trade the stale price.
- Order FSM: only `is_active` states can fill; `can_cancel` per state; reject any (state, event) not in `VALID_TRANSITIONS`.
- Run IB equity notebooks (03, 12) only in RTH 09:30–16:00 New York, Mon–Fri; FX (11) is 24/5. Delete `~/.ml4t/live_state/basket_demo_*.json` between nb 12 rehearsal runs.

| Parameter | Default | Where |
|---|---|---|
| IB ports / client IDs / connect timeout | 7497 TWS paper, 4002 Gateway paper / 10, 11, 12 / 30 s | nb 03, 11, 12 |
| `MARKET_DATA_TYPE` | None if live L1, else 3 (delayed) | nb 03, 12 |
| ETF loop | TOP_K=5, REBALANCE_EVERY_N_DAYS=21, CASH_BUFFER=0.02, COMMISSION_RATE=0.0005, NOTIONAL_PER_LEG_USD=5,000, INITIAL_CASH=100,000, LIVE_WINDOW_START=2025-01-01, alpha=1e6, NEXT_BAR | nb 02 |
| IB basket | TOP_K_LONG=5 of 20, TARGET_NOTIONAL_USD=50,000, MAX_POSITION_USD=15,000, MAX_ORDER_USD=12,000, MAX_DAILY_LOSS_USD=5,000, WARMUP_DAYS=60, settle 15 s, MOC ack 5 s | nb 12 |
| FX loop | TOP_K=5, BASE_QTY_PER_LEG=20,000 rounded to 1,000, RIDGE_ALPHA=1.0, TRAIN_END_DATE=2024-12-31, duration "60 D" | nb 11 |
| Crypto loop | NOTIONAL_PER_LEG_USD=100, MAX_FUNDING_AGE_HOURS=8.0, MIN_OKX_LIVE_COVERAGE=0.75, LightGBM 200 rounds / lr 0.05 / 31 leaves / seed 42 / cpu | nb 09 |
| Crypto spot demo | lookback 10, entry abs(z) >= 1.5, exit abs(z) < 0.25, position_size 0.1, mock cash 10,000 | nb 05 |
| Momentum demos | lookback=5, threshold=0.02, position_size=10 (Alpaca) | nb 03, 04 |
| Parity harness | lookback=10, threshold=0.02, N_BARS=30 | nb 08 |
| `LiveRiskConfig` envelope | 10k USD / 500 sh per order; 50k USD / 1000 sh per position; 200k total; 20 positions; 10 orders/min; 5k daily loss; 5% drawdown; 5% price deviation; 60 s staleness; 1 s dedup | `ml4t.live.safety` |
| QC export | `PREDICTION_THRESHOLD=0` (widest); file `research-to-backtest-factors.json` | nb 06 |

| Feature | Alpaca | Interactive Brokers |
|---|---|---|
| Minimum balance | None | None (higher margin reqs) |
| Commissions | Free | $0–1 per trade |
| Real-time data | Free (IEX) | Paid subscription |
| API complexity | Simple REST | Complex socket protocol |
| Crypto | Yes (24/7) | Limited |
| Paper trading | Yes | Yes |

Equities vs crypto (nb 05): hours 9:30–16:00 ET vs 24/7/365; settlement T+2 vs instant; minimum trade 1 share vs fractional; volatility lower vs several times higher; funding N/A vs 8-hour intervals. Crypto checklist: always-on cloud deployment, heartbeat monitoring, graceful reconnection, tighter stops, liquidation risk on leverage, exchange maintenance windows, funding-time entries/exits.

## Code patterns and APIs

Imports: `from ml4t.backtest import BacktestConfig, DataFeed, Engine, ExecutionMode, Strategy, OrderSide, OrderType`; `from ml4t.backtest.types import Order, OrderSide, OrderStatus, OrderType, Position`; `from ml4t.live import LiveEngine, LiveRiskConfig, RiskLimitError, RiskState, SafeBroker, VirtualPortfolio, AlpacaBroker, AlpacaDataFeed`; `from ml4t.live.safety import SafeBroker, VirtualPortfolio`; `from ml4t.live.wrappers import ThreadSafeBrokerWrapper`; `from ml4t.live.protocols import ExecutionCapability`; `from ml4t.live.engine import LiveEngine`; `from ml4t.live.brokers.ib import IBBroker`; `from ml4t.live.brokers.alpaca import AlpacaBroker`; `from ml4t.live.feeds import BarAggregator, IBDataFeed`; `from ml4t.data.etfs import ETFDataManager`; `async_utils.run_async` (filters the `nest_asyncio` deprecation).

```python
# Backtest reference tape
engine = Engine(feed, strategy, config=BacktestConfig(initial_cash=INITIAL_CASH,
                execution_mode=ExecutionMode.NEXT_BAR, commission_rate=0.0005))
results = engine.run()          # results['final_value'], results['total_return_pct']

# Live wiring: feed > SafeBroker > LiveEngine
cfg = LiveRiskConfig(max_position_value=MAX_POSITION_USD, max_order_value=MAX_ORDER_USD,
                     max_daily_loss=MAX_DAILY_LOSS_USD, allowed_assets=set(UNIVERSE))
safe_broker = SafeBroker(ib_broker, cfg, state_path=STATE_FILE)   # kwarg names inferred from usage
await safe_broker.connect()      # emits reconciliation_report; stop if not clean
engine = LiveEngine(safe_broker, feed)                            # constructor kwargs not shown in notes
run_engine_for_duration(engine, duration_s); engine.runtime_status()["health"]
```

Broker protocol `LiveEngine` / `SafeBroker` require (as implemented by `SimulatedBroker`, `LiveBroker`, `MockBrokerQueries`, `DemoBrokerQueries`): lifecycle `async connect()`, `async disconnect()`, `is_connected()`, `async is_connected_async()`, `execution_capabilities()`, `assert_paper_trading()`; state `positions() -> dict[str, Position]`, `pending_orders() -> list[Order]`, `get_position(asset) -> Position | None`, `get_positions_async`, `get_pending_orders_async`, `get_position_async`, `get_account_value_async() -> float`, `get_cash_async() -> float`; pricing `update_price(asset, price, timestamp)`; orders `async submit_order_async(asset, quantity, side=None, order_type=OrderType.MARKET, limit_price=None, stop_price=None, **kwargs) -> Order`, `async cancel_order_async(order_id) -> bool`, `async close_position_async(asset) -> Order | None`.

Feed protocol (`HistoricalReplayFeed(data, symbols, broker=None)`): `async start()`, `stop()`, `stats() -> dict`, `__aiter__`/`__anext__` yielding `(timestamp, {symbol: {open, high, low, close, volume...}}, context)`; raises on a missing bar. Sync mock shape (`MockBroker`, `MockCryptoBroker`): `update_market(timestamp, prices)`, `get_position(symbol) -> _PosView`, `submit_order(asset, quantity, side=None, **kwargs) -> dict` with status `filled` / `rejected` / `unsupported`.

Reconciliation: `reconcile(current, target_frame)` = outer join on `symbol`, `delta_qty = target_qty - current_qty`; submit only rows with `delta_qty != 0`; re-run after `wait_for_basket_to_settle`. Signal log -> `pl.DataFrame` via `signals_to_frame` / `_signals_to_frame`; `_normalise(records) -> list[tuple]`.

FSM: `VALID_TRANSITIONS: dict[OrderState, dict[OrderEvent, OrderState]]`; `get_valid_events(state)`; `Order.apply_event(event, metadata={"qty": ..., "price": ...})`; `StateTransition` history; `visualize_state_machine()`.

Parity harness: `count_result`, `matching_records(reference, candidate, fields, float_fields=())`, `sequence_result`, `unique_order_ids_result`, `pipeline_results`, `execution_results`, `run_verification_tests`, `skipped_results`, `result_status`; pattern `tests/live/test_parity.py`.

Feature helpers shared by train and live paths (Polars expressions): `rolling_zscore_expression(column, alias, window=21)`, `crypto_rsi_expression(window=14)`, `vwap_distance_expression(window=3)`, `daily_rsi_expression(window=14)`; `compute_features_8h(panel)`, `compute_features_daily(panel)`.

Venue helpers: OKX `fetch_okx_candles_1h(inst_id, limit=300)`, `aggregate_to_8h(bars_1h)`, `fetch_okx_funding_history(inst_id, limit=100)`, instrument `f"{sym[:-4]}-USDT-SWAP"`; IB `reqHistoricalData` / `reqHistoricalDataAsync` (duration `"60 D"`), `asyncio.gather` fan-out, `Forex(symbol)` qualification on IDEALPRO, `OrderType.MOC`, `fetch_ib_snapshot(symbol) -> mid`; QC `qb.object_store.save('research-to-backtest-factors.json', json_str)`, `PredictionUniverse(PythonData)` with `get_source` / `reader`, `FileFormat.UNFOLDING_COLLECTION`.

Registry: `registry_readonly_uri(dir)` -> SQLite URI with `mode=ro` (+ `immutable=1` only if the directory is unwritable).

Async notebook driver: `run_demo(awaitable)` runs the coroutine outside the kernel's active loop; nb 10 variant `run_demo(coro, expect_error=False)` asserts expected allow/block.

Env and deps: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` (read only when submission is enabled), `IB_ACCOUNT`, `ML4T_DATA_PATH`, `ML4T_HEADLESS_PAPERMILL`, `LIVE_FEED`, `CASE_STUDIES_DIR`, `PYTHONHASHSEED` caveat; `ml4t-live>=0.1.0`, `alpaca-py`, `ib_async`, `python-okx` (PyPI name, not `okx`) via `uv sync --extra live`; CLI `ml4t-live status|shadow`.

Paths: `25_live_trading/_etfs_features.py`, `25_live_trading/output/etfs_deployment/`, `get_output_dir(25, "<name>")`, `get_chapter_dir(25)`, `case_studies/etfs/labels/fwd_ret_21d.parquet`, `case_studies/etfs/run_log/registry.db`, `case_studies/crypto_perps_funding/labels/fwd_dir_8h_3c.parquet`, `case_studies/fx_pairs/labels/fwd_ret_1d.parquet`, `ML4T_DATA_PATH/{etfs/market, macro, crypto_perps, crypto_premium, fx_pairs}`, `~/.ml4t/live_state/`, `libs/src/ml4t_live/ml4t/live/safety.py`.

Running: `uv run python 25_live_trading/01_unified_framework_demo.py`; tests `uv run pytest tests/test_chapter_notebooks.py -v -k "25_live_trading"`; headless `MPLBACKEND=Agg PLOTLY_RENDERER=json`; `ML4T_HEADLESS_PAPERMILL=1 papermill 04_alpaca_paper_trading_demo.ipynb out.ipynb`.

## Evidence from the book

- `01_unified_framework_demo`: 9/9 signals identical across `Engine` and `LiveEngine` on one year of SPY daily bars (10/30 MA); trade count = signals − 1 when the last position is open.
- `02_etfs_deployment_loop`: sizing at full account value produced 8 broker refusals over the live window with nothing in the notebook looking; a 2% `CASH_BUFFER` removed them on this window. Registry leader for `fwd_ret_21d` is `ridge_a1000000.0` (alpha = 1e6) by validation IC. Financial-only feature set drops ~10 model-based columns (HMM, GARCH); refit seconds vs minutes; forecast-quality cost not measured.
- `04_alpaca_paper_trading_demo` default offline run establishes interface and log only; nothing about Alpaca fills, latency or rejections.
- `05_alpaca_crypto_live_demo` / `09_crypto_funding_deployment_loop`: 11 of 19 case-study perps tradeable on Alpaca USD spot (ADA, APT, ATOM, BNB, COMP, INJ, NEAR, SUI not; later takeaway notes the list "adds ADA", so coverage is computed at run time); 8 signal-only; coverage floor 0.75; publication run dry-run with no `submitted` records; explicitly "not a faithful PnL reproduction of the funding strategy".
- `06_quantconnect_case_study`: 46,466 holdout predictions, 95 symbols, 497 dates (2024-01-02 to 2025-12-23); threshold sweep shows breadth collapsing to a handful of names at high thresholds and approaching the whole universe near zero.
- `07_order_state_machine`: 10 states, 9 events, 19 transitions; walkthrough acknowledge -> accept -> partial fill -> fill with weighted-average fill price; invalid transition raises; atomic fill guard leaves state/history/filled_qty unchanged on bad metadata.
- `08_pipeline_verification`: on the fixed 30-bar two-regime tape, backtest and live-style replays produce identical feature, prediction, signal, order, fill and broker-state logs with unique order IDs on both paths (5 gated tests + 1 declared warm-up difference); when the live replay cannot run under Papermill every gate reports FAIL rather than PASS.
- `10_safety_risk_demo`: order-size, position, exposure and asset gates each raise `RiskLimitError`; rate-limit burst rejected on the 4th order; kill switch latch survives `SafeBroker` reconstruction; shadow mode moves `VirtualPortfolio` while the mock broker stays flat; `VirtualPortfolio` tracks weighted-average cost basis and position flips. Duplicate-order filtering and price-deviation checks are configurable but not demonstrated anywhere in the chapter.
- `11_fx_deployment_loop`: dry-run basket of top-5 positive-forecast pairs at 20k base units; the contract-qualification table is the coverage result; no fills in publication mode.
- `12_ib_basket_rebalance_demo`: publication mode reports planned notional only; an authorized paper run reports per-leg slippage vs `last_close`; residual deltas after a 15 s settle distinguish unfilled/rejected legs from risk blocks.
- `13_runtime_safety_showcase`: a stale order is rejected after 1.5 s against a 1 s threshold with age and threshold in the error; a 1,000 USD synthetic equity drop trips the kill switch and persists across reconstruction; a state file claiming long 10 AAPL plus a working LIMIT order against an empty broker yields a non-clean report that clears after state reset; observed health sequence `stopped -> waiting_for_data -> ok -> feed_silent -> stopped` within ~6 s.

## Related references

- `chapters/16_strategy_simulation.md` — `ml4t.backtest.Engine`, `BacktestConfig`, `ExecutionMode.NEXT_BAR` that produce the offline reference tape.
- `chapters/26_mlops_governance.md` — monitoring layer that consumes the four-bucket contract, execution-cost drift, `trained_at`, cadence decoupling.
- `chapters/06_strategy_definition.md` — ETF strategy research and the crypto perp-spot premium index the deployments descend from; naive momentum is not research-grade.
- `chapters/07_defining_the_learning_task.md` — forward-return and 3-class direction labels (`fwd_ret_21d`, `fwd_dir_8h_3c`, `fwd_ret_1d`) and why the horizon sets the training cutoff.
- `chapters/08_financial_features.md` — the financial-only feature subset the live path can compute from a broker feed.
- `chapters/09_model_based_features.md` — HMM/GARCH families dropped from the deployment fit.
- `chapters/11_ml_pipeline.md` — Ridge configuration, registry and validation-IC selection that supply the deployment hyperparameters.
- `chapters/12_gradient_boosting.md` — LightGBM determinism settings used by the crypto deployment model.
- `chapters/18_transaction_costs.md` — slippage vs `last_close` as the execution-cost summary.
- `chapters/19_risk_management.md` — kill switches, stops and position caps that `SafeBroker` enforces.
- `chapters/21_rl_execution_hedging.md` — execution-side modelling beyond market/MOC orders.
- `case_studies/etfs.md` — source of the ETF deployment loop and QC export (registry, labels, holdout).
- `case_studies/crypto_perps_funding.md` — 19-perp universe, 8H panel, funding-rate strategy behind notebooks 05 and 09.
- `case_studies/fx_pairs.md` — FX schema and `fwd_ret_1d` behind notebook 11.
- `libraries/ml4t_live.md` — `LiveEngine`, `SafeBroker`, `LiveRiskConfig`, `RiskState`, brokers, feeds, CLI.
- `libraries/ml4t_backtest.md` — `Strategy` base class, `Engine`, order types shared with live.
- `libraries/ml4t_data.md` — `ETFDataManager.update()` incremental refresh.
- `guardrails.md`, `decision_rules.md`, `workflow.md`, `evidence.md`, `glossary.md`, `companion_repo.md` — cross-cutting indexes that reference this chapter.
- Further reading:
  - Almgren & Chriss (2001), Optimal execution of portfolio transactions, Journal of Risk.
  - Bailey & López de Prado (2014), The Deflated Sharpe Ratio.
  - Bailey et al. (2015), The Probability of Backtest Overfitting.
  - Harris (2003), Trading and Exchanges: Market Microstructure for Practitioners.
  - Madhavan (2002), Market Microstructure: A Practitioner's Guide, Financial Analysts Journal.
  - López de Prado (2018), Advances in Financial Machine Learning.
  - Schwarz et al. (2022), The 'Actual Retail Price' of Equity Trades.
  - Illustrative QC project: quantconnect.cloud/backtest/37075c225715df9ef4477dc748b1cbf7.

## Glossary

- **Unified framework** — one `Strategy` class executed by both `ml4t.backtest.Engine` and `ml4t.live.LiveEngine`.
- **Parity test / parity gate** — field-by-field assertion that two engines produce the same tape on the same bars; a stage-level gate that fails CI on mismatch.
- **Tape** — a fixed, seeded sequence of (timestamp, bar) tuples replayed identically through both pipelines.
- **Reference tape** — deterministic offline replay (NEXT_BAR) of the live window used as the counterfactual for reconciliation.
- **Shadow mode** — real broker connection, real prices, orders routed to `VirtualPortfolio` and never sent.
- **VirtualPortfolio** — shadow-mode position tracker with weighted-average cost basis and flip handling.
- **SafeBroker** — wrapper enforcing position/order/exposure/daily-loss caps, rate limits, staleness, kill switch, persisted `RiskState`, startup reconciliation.
- **RiskLimitError** — exception raised by `SafeBroker` when any control rejects an intent, carrying measured value and threshold.
- **Kill switch latch** — persisted `kill_switch_activated` flag with reason; survives restarts; cleared only by the operator.
- **MarketSnapshot** — per-asset latest price with timestamp kept by `SafeBroker` for staleness and sizing checks.
- **Four buckets** — intended / attempted / accepted / failed per-leg basket disposition.
- **Disposition** — per-symbol execution outcome label (flat, no mapping, unsupported short, dry run, no_credentials, submitted).
- **Signal-only** — a model intent for a symbol with no execution-venue mapping; logged, not traded.
- **LABEL_AVAILABLE_AS_OF** — last training day whose forward label realizes before the live window.
- **Deployment artefact** — model + scaler/imputer + feature order + metadata fit by the live code path; distinct from the research registry artefact.
- **Registry** — SQLite WAL database of case-study runs (`registry.db`); read via `registry_readonly_uri`.
- **Prediction bridge** — exporting model scores as a file so a managed platform holds only portfolio rules.
- **UNFOLDING_COLLECTION** — LEAN file format streaming one date at a time to avoid lookahead.
- **Data plane / execution plane** — the venue providing prices and funding versus the venue accepting orders.
- **Backward as-of join** — per-symbol join taking the most recent funding print at or before each bar timestamp.
- **Funding age / funding rate** — hours between the matched funding print and the bar's 8H grid; the periodic perp payment at 00/08/16 UTC (Binance).
- **Momentum z-score proxy** — rolling z-score of close-to-close returns standing in for the perp-spot premium.
- **Reconciliation** — diff of current broker positions vs model targets producing `delta_qty`; also `SafeBroker.connect()` diff of persisted state vs broker.
- **Residual delta** — non-zero `delta_qty` remaining after fills settle.
- **MOC** — Market-On-Close order routed to the closing auction (`OrderType.MOC`).
- **Contract qualification / IDEALPRO** — resolving a case-study symbol to an IB `Forex` contract on IB's interbank FX desk (1k base-unit step on majors).
- **PENDING_CANCEL** — explicit state for a cancel in flight; exits to CANCELED, FILLED or back to ACCEPTED.
- **Terminal state** — FILLED, CANCELED, REJECTED, EXPIRED (no outgoing edges).
- **Health state** — `runtime_status()["health"]` in stopped / waiting_for_data / ok / feed_silent / idle_market_closed / broker_disconnected.
- **RTH** — US-equity regular trading hours 09:30–16:00 New York.
- **MARKET_DATA_TYPE** — TWS quote mode: 1 real-time, 2 frozen, 3 delayed, 4 delayed-frozen.
