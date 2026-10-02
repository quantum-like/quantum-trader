# Chapter 3: Market Microstructure

> Market data is an economic object, not a neutral price series: spreads, depth, order types, venue design and time of day shape every trade and quote you observe, and a model that ignores this learns the frictions first (bid-ask bounce, quote flicker, late prints). This chapter equips you to choose the feed that matches the question (L1/L2/L3/TAQ/enriched bars), parse message data and reconstruct a venue-local limit order book under strict accounting invariants, derive order-flow measures (OFI, depth imbalance, book pressure, markouts), sample ticks into time, activity and information-driven bars with `ml4t.engineer.bars`, classify trade direction (venue label > Lee-Ready > tick test), time intraday jumps (bipower variation + Lee-Mykland), and audit microstructure data as invariants rather than generic cleaning. Its position: data choice is a strategy choice, bar construction is measurement design, a statistical signal is not tradable P&L until it survives spread and latency, and order flow is more defensible as a filter than a forecast. Source material: companion-repo READMEs and notebooks `03_market_microstructure/01`–`18` (book PDF unavailable; sections 3.1, 3.2, 3.6 known only through README summaries and notebook asides).

## When to use this reference

- Choosing between L1, L2 (IEX DEEP), L3/MBO (NASDAQ ITCH, DataBento), TAQ/NBBO or vendor minute bars (AlgoSeek) for a research or trading question.
- Parsing NASDAQ TotalView-ITCH 5.0 binaries or DataBento MBO Parquet, or reconstructing a limit order book (order pool + price levels, replaces, partial fills, crossed-quote checks).
- Building order-flow features: OFI (message-based or aggressor-based), depth imbalance, book pressure, spread in bps, FINRA share, tick-direction ratios.
- Sampling ticks into bars (time/tick/volume/dollar/imbalance/run) with `ml4t.engineer.bars`, calibrating thresholds to a bar-count target, or diagnosing adaptive-threshold drift.
- Classifying trade direction (aggressor) when the feed has no label: Lee-Ready vs tick test, with coverage and accuracy reporting.
- Evaluating a microstructure signal: lagging the predictor, honest standard errors, decile ramps, midpoint vs latency-adjusted vs executable markouts against ~1-2 bps round-trip cost.
- Decomposing intraday realized variance into continuous and jump parts (bipower variation, Lee-Mykland) and emitting jump features for labels or regimes.
- Auditing intraday data: sequencing ties, timezone/RTH filters, zero-price or crossed NBBO, late prints (condition `80000002`), split-unadjusted ticks, OHLC invariants, null semantics, memory limits.
- Reading the AlgoSeek 61-column TAQ minute bars (quote OHLC, trade buckets, tick direction, pressure, FINRA volume) as ML inputs.

## Core ideas (the why)

- **Price data is shaped by frictions.** Every print and quote reflects spread, depth, order type, venue and time of day. Ignore it and the model learns bid-ask bounce before anything economic.
- **Data choice is a strategy choice (section 3.2).** Per-order cancellation and queue position need L3; spread/depth dynamics are fine on L2; TAQ gives the consolidated quote a marketable order actually meets but no depth; enriched vendor bars give pre-computed microstructure fields. Timestamp semantics (exchange vs receipt vs decision time), latency and fragmentation decide both research validity and live feasibility.
- **A venue publishes far more quoting than trading.** Adds and deletes dominate message counts; executions are a sliver; TAQ quotes outnumber trades ~10:1. Book rearrangement is where price discovery happens and a trade-only feed cannot see it.
- **One venue is not the market.** ITCH shows NASDAQ-routed activity only; the listing exchange leads but does not dominate; FINRA TRF is one bar on a chart but many venues (dark pools, internalisers). Liquidity stays fragmented under stress.
- **The visible book is not reachable liquidity.** Orders added and withdrawn within a millisecond appear in snapshots but no slower participant can trade against them; order lifecycle decides tradability.
- **Replace (U) is re-pricing, not withdrawal.** It retires one reference and issues a new one that never appears in an add; treating it as a cancel overstates liquidity leaving. **Track what remains of an order, not what it started as** (deletes remove remaining shares).
- **Crossed quotes measure your reconstruction's error rate**, not the market.
- **Quote spreads in bps**, but remember bps has price in the denominator: a spread-vs-price scatter cannot separate mechanical scaling from liquidity.
- **Compute returns from mid prices, never trade prices.** Trade prices zig-zag across the spread (Roll 1984); tick-level "momentum" on trade prices is the bounce.
- **Bar construction is measurement design (section 3.4).** Markets do not generate information at a constant rate; the sampling rule changes kurtosis, autocorrelation and heteroskedasticity of the training data. Activity-time bars are simple and predictable; imbalance bars buy information-driven sampling at the cost of parameter sensitivity.
- **Direction: venue label > Lee-Ready > tick test.** A label is a record; the others are estimates. The quote test answers from the quote prevailing at the trade; the tick test answers from history and carries errors forward through flat trades.
- **A statistical signal is not tradable P&L.** Judge in bps against round-trip cost (~1-2 bps liquid US equities), after latency and executable markouts, not against a p-value.
- **Standard errors are larger than 1/sqrt(n)** on overlapping forward windows; a sign flip across horizons is not a horizon effect; an estimate inside one SE leaves the question open.
- **Contemporaneous is not predictive.** Same-minute imbalance vs return is near-definitional; lag the predictor.
- **A cross-section answers what one stock cannot.** Fix it (stratified, named) so the figure is reproducible.
- **Order flow is more defensible as a filter than a forecast.** Momentum (informed trader) vs reversal (liquidity concession) readings of one-sided flow differ by horizon and extremity, not sign; thresholds come from the symbol's own percentiles.
- **Count trades and shares separately**; rank tickers by dollars; normalise each ticker's intraday profile before averaging shapes.
- **Data quality is invariants and auditability (section 3.6)**: sequencing, timezone discipline, zero prices, locked/crossed quotes, late prints, session boundaries, splits.
- **RV absorbs jumps; bipower variation does not.** Lee-Mykland times jumps with a multiple-testing-correct threshold adapting to local volatility; jump features change label distributions (Ch7) and regime features (Ch8).

## Method recipes (the how)

### 3.1 ITCH 5.0 binary parsing (`01_itch_parser`, `itch_message_specs.py`)

1. Download via `data/equities/market/microstructure/nasdaq_itch_download.py` (~5 GB compressed, e.g. `01302020.NASDAQ_ITCH50.gz`, one 423M-message day).
2. Frame = 2-byte big-endian length (includes type byte) + 1-byte ASCII type + payload (`length-1` bytes). `read_frame(f, pbar)` -> `(msg_type, payload)` or None on EOF/truncation.
3. `decode_message`: `struct.unpack(FMT_DICT[t], payload)` -> namedtuple -> dict. Timestamp = 6 bytes big-endian int = ns since midnight (`parse_timestamp`); prices are price4 ints (`parse_price4` = int/10000; 3212000 -> $321.20); byte strings ASCII-decoded and stripped.
4. `parse_itch_file(itch_file, trading_day, output_dir, max_buffered_messages=10_000_000, max_messages=None)`: buffer by type, `flush_to_parquet` when buffered total exceeds threshold; per-type dirs `messages/{A,F,E,C,X,D,U,P,Q,R,S,...}/part-{idx:06d}.parquet`; timestamp stored as `base_ts + pl.duration(nanoseconds=...)` Datetime("ns"). Unknown types counted and skipped.
5. `build_parsers()` -> `(NT_DICT, FMT_DICT)` (big-endian `">"`), `build_parquet_schemas()` -> `PARQUET_SCHEMAS` (H->UInt16, I->UInt32, Q->UInt64). Add Order format `>HH6sQsI8sI` = stock_locate, tracking_number, timestamp, order_reference_number, buy_sell_indicator, shares, stock (8s padded), price.
6. Params: `SKIP_PARSING=False`, `MAX_MESSAGES=None`, env `ITCH_KEEP_EXISTING=1` keeps an existing parse. ~22 min, ~8 GB RSS.
7. Rust alternative `github.com/ml4t/itch-parser`: `./target/release/itch_parser <input> <output_dir> <MMDDYYYY>`; identical Parquet schema.

| Message group | Types | Role |
|---|---|---|
| Book-carrying | A/F add, E/C execute, X partial cancel, D delete, U replace | drive the LOB |
| Trade tape | P (non-displayed trade), Q (cross) | never touch the visible book |
| Session | S (system events O/S/Q/M/E/C), R (stock directory), H, Y, L, V/W (MWCB), J (LULD), K (IPO), B (broken trade), I (NOII) | calendar, halts, symbol map |

Timestamps are US/Eastern, timezone-naive; convert to America/New_York before cross-source joins.

### 3.2 LOB reconstruction from ITCH (`02_itch_lob_reconstruction`, `limit_orderbook.py`)

State: `pool = {order_ref: (side, price, shares_remaining)}`, `book = {"B": {price: shares}, "S": {price: shares}}`.

| Msg | Pool | Book |
|---|---|---|
| A/F | record order | add shares at price |
| D | drop order | subtract **remaining** shares |
| X | reduce remaining | subtract `cancelled_shares` |
| E | reduce remaining | subtract `executed_shares` at resting price |
| C | same as E | shares come off the **resting** price (`execution_price` is where it printed, not where it was displayed) |
| U | retire old ref (subtract remaining at old price), record new ref; **side inherited from old order** (U carries no side) | remove at old, add at new |
| P | none | none (trade tape only) |

Steps:
1. `get_stock_locate_mapping(ITCH_DIR)` from R. Filter A/F/P by `stock`; D/X/E/C/U by `stock_locate` (they carry no ticker). `load_itch_messages` divides price/execution_price by 10000.
2. Process all messages from start of day to `END_TIME`; keep snapshots >= `START_TIME` (pool must be warm; pre-market adds get deleted after the open).
3. `reconstruct_lob_with_ofi(add_orders, deletes, cancels, executions, executions_c=None, replaces=None, snapshot_freq="1s", show_progress=True)`: Numba kernel; sort by `np.lexsort((tracking_numbers, timestamps))`; price keys `int64(price*10000+0.5)`; snapshot when `ts - last_snapshot_ts >= interval` and both sides non-empty. Output `timestamp, best_bid, best_ask, mid_price, spread, bid_size_0, ask_size_0, ofi, bid_add, bid_remove, ask_add, ask_remove`. `snapshot_freq` in {100ms, 500ms, 1s, 5s, 10s, 1min} else ValueError. Snapshot buffer bounded by `min(span/interval+2, n_messages+1)` (nopython mode does no bounds checks).
4. `reconstruct_lob(..., n_levels=10, snapshot_freq="1s", snapshot_start=None)`: pure-Python variant emitting `bid_price_{i}/bid_size_{i}/ask_price_{i}/ask_size_{i}`; prints "WARNING: N crossed book states detected".
5. Validate: `crossed = (lob["spread"] < 0).sum()`; print share of valid spreads.
6. Params: `SYMBOL="AAPL"`, `TRADING_DATE="2020-01-30"`, `START_TIME="09:30:00"`, `END_TIME="16:00:00"`, `MESSAGE_LIMIT=None`, `SNAPSHOT_FREQ="1s"`; env `ITCH_SYMBOL` overrides for batch runs. Output `03_market_microstructure/output/nasdaq_itch/order_book/{SYMBOL}/lob_snapshots.parquet` (nb 02 uses the Numba version, so saved snapshots carry level 0 only; `build_depth_scatter_data(n_levels>1)` in nb 03 degrades silently on them).

OFI per interval: `(bid_add - bid_remove) - (ask_add - ask_remove)`; U counts as remove at old price + add at new.

### 3.3 Hand-built OFI, spread metrics and cross-section (`03_itch_lob_analysis`)

- Registry from A/F (`order_reference_number, side, shares, price, timestamp`); removals D/X/E/C joined to registry for side. X -> `cancelled_shares`, E/C -> `executed_shares`, D -> `shares - removed_before` (cumsum of earlier removals per order, deletes sorted last among ties, clipped >= 0). This construction **ignores U** (documented limitation).
- `_pivot_by_side`: truncate timestamp to `freq`, group by (bucket, side), pivot B->bid_*, S->ask_*; full outer join adds/removes on bucket, `fill_null(0)`, **cast to Int64 before subtracting**.
- Alignment: `mid_price.last()` per bucket; `next_return_bps = mid.pct_change().shift(-1) * 1e4`; require >= 5 buckets.
- Report `corr_raw` and `corr_log` with `ofi_log = sign(x) * log1p(|x|)`; the gap is the heavy-tail effect.
- `compute_ofi_correlation` cross-section: per-second `ofi` summed to `BUCKET_FREQ="1m"`; `MIN_BUCKETS=50`; x-axis = daily A+F add count (log); 50 fixed symbols in 5 strata of 10 by daily add count (stratum 1: QQQ, SPY, TQQQ, IWM, AMD, DIA, AAPL, XLK, SH, MSFT; stratum 5: VGI, PMM, WINS, RCON, JOYY, ISIG, BAC-A, CMRE-E, LOAC, BRN); >= 3 symbols to draw; report range, mean, std, count positive. Output `output/lob_analysis/ofi_correlation_cross_section.parquet`.
- `compute_spread_metrics`: `spread_bps = spread / mid_price * 10000`, filter `spread > 0`; 1-min mean spread, 1-min last mid; cast Polars µs timestamps to `datetime64[ns]` for pandas resampling. Depth scatter shaded by `log(depth)`.
- Scan A and F per type (F has an `attribution` column A lacks; one `scan_parquet` over both raises).

### 3.4 Order lifecycle (`04_itch_order_lifecycle_analysis`)

- Adds = A ∪ F via `pl.concat(how="diagonal_relaxed")` -> `order, side, submitted, ticker`; side_num B->1, S->-1.
- Termination = first D/X/U naming the order (`sort("terminated").group_by("order").first()`); `termination_rate = terminated / total_unique_orders`; break down by D/X/U.
- `cancel_time = (cancelled - submitted).dt.total_nanoseconds()/1e9` (never a whole-seconds helper); quantiles 10/25/50/75/90/95/99; histograms on `np.logspace` bins, log x-axis.
- Execution = first E ∪ C per order; `exec_rate`; `exec_time` = time to **first** fill.
- Unified one-outcome-per-order classification: "Executed then Terminated" = partial fill then cancel/delete; "Still live or unknown" = no termination message (includes executed); other category names (inference).
- Venue-wide pass (3 columns, all symbols): time to first D and first E; report share within 500 ms / 1 s / 10 s and median (deletes); share within 1 ms, median, share over 40 min = 2400 s (fills).
- Price improvement on C (enriched from nb 05): `execution_price - order_price` in price4 ints; buy better if negative, sell better if positive; split better/at/worse.
- `_postprocess_message_df` heuristics: integer timestamps -> `pl.duration(nanoseconds=...)`; prices with median > 10000 divided by 10000. Params `SYMBOL="AAPL"`, `MAX_ORDERS=None`; ~33 GB.

### 3.5 Enrichment, replace chains and concentration (`05_itch_trading_activity`)

1. Directory R -> (stock_locate, stock); order attrs A ∪ F -> (order_reference_number, price, buy_sell_indicator, stock). **Assert `order_attrs.order_reference_number.n_unique() == height` before any left join.**
2. `_apply_replacements` (pointer doubling): links = U rows (`new -> ancestor=original, price`), `unique(subset=new, keep="first")`; each pass left-joins links to itself on ancestor to fetch grandparent, `ancestor = coalesce(grandparent, ancestor)`; stop when none found; `MAX_REPLACEMENT_PASSES=40` (2^40 chain length; hitting the cap means a cycle). Join ancestor -> add's side and stock; price from the U itself. Print unresolved count.
3. E enriched: join R on stock_locate, attrs on order ref -> `order_price, side`; `execution_price = order_price` (E executes at limit). C enriched: `price_improvement_raw = execution_price - order_price` (Int64). X enriched: stock only. Outputs `output/nasdaq_itch/enriched/{E,C,X}.parquet`.
4. Canonical trade table from C, E, P, Q -> `timestamp, ticker, shares, price, value, msg_type` (`executed_shares`->shares; `execution_price`/`cross_price`/`price`->price; price/10000 if median > 10000; `value = shares*price`). Outputs `trading_activity/trades.parquet`, `trade_summary.parquet`.
5. Concentration: per-ticker total_value sorted desc, cumulative share; `n_50 = searchsorted(cum, 0.5)+1`, `n_80`; top-50 share. Params `MAX_ROWS=0` (all), `REBUILD_ENRICHED=True`; ~31 GB.

### 3.6 Intraday U-shape and stylized facts (`06_itch_intraday_patterns`, `07_itch_stylized_facts`)

- `intraday_resample(trades, ticker, freq)`: `group_by_dynamic` -> shares, value, last price, trade_count, `vwap = value/shares`. Tiers: sort trade_summary by total_value desc, filter trade_count >= `MIN_TRADES_LOW=500`; high = first, mid = middle index, low = last above floor; low tier 15-min bars, others 5-min.
- `compute_intraday_pattern(trades, top_20_tickers, "30m")`: `time_slot = hour + minute/60`; `vol_pct = shares / ticker daily total`; mean and std across tickers; **filter 9.5 <= slot <= 16**; even-split reference `100/len(times)`; ratio `(open + close) / (2 x midday)`, midday = slot nearest 12.5; label by `slot_label`, not index. Params `PATTERN_TICKERS=20`, `PATTERN_FREQ="30m"`.
- Order flow per ticker (nb 07): adds A/F; cancels from enriched X (raw X has no ticker); `cancel_rate_proxy = sum cancelled_shares(X) / sum shares(A+F)` (explicitly not the total cancel rate).
- Bid-ask bounce: `compute_tick_autocorrelation(trades, ticker, max_lags=10)` on **1-second bars as tick proxy**; `return_bps = price.pct_change()*1e4`; autocorr `cov(r[lag:], r[:-lag])/var`; >= 100 bars; negative lag-1 = bounce.
- Liquidity spectrum: pool = tickers with `trade_count >= MIN_TRADES=500` sorted by value; indices `[0, n//4, n//2, 3n//4, n-1]`; 1-min bars: total_volume, total_value, avg_trade_size, `volatility_bps = std(clip(returns_bps, p1, p99))`, `price_range_pct = (p95 - p5)/median x 100`; volume on log axis.

### 3.7 DataBento MBO engine (`08_databento_lob_reconstruction`)

- Columns `ts_event, ts_recv, action, side, price, size, order_id, flags, publisher_id`; sort by (ts_event, order_id); RTH approximation 13:30-21:00 UTC (DST-naive, flagged); prefer cached `{SYMBOL}_mbo_rth.parquet`. Load via `load_mbo_data(symbols=[SYMBOL], list_files=True)`.
- Flags `F_LAST=128, F_TOB=32, F_SNAPSHOT=8, F_MAYBE_BAD=2`; `UNDEF_PRICE = 2^63-1`. Price scale auto-detect: median > 1e6 -> nanodollars (/1e9).

| Action | Effect |
|---|---|
| A | add to book |
| M | modify; unknown id -> treated as add, `unknown_order_count++` |
| C | cancel (uses **existing order's** side/price, not the message's) |
| F | fill: reduces resting order; decrement clamped `min(size, existing.size)`, `negative_size_count++` when clamped; delete at size <= 0 |
| T | trade: tape only, **no book update** |
| R | reset (clear all); N no-op; side N ignored |
| A/M with `flags & F_TOB` | replace entire side (`_apply_tob_replace`); `price == UNDEF_PRICE` clears side (IEXG, NYSE National) |

- Classes `OrderState -> PriceLevel -> BookSide -> LimitOrderBook -> LOBReconstructor`. `BookSide` keeps `Counter` levels (O(1) add/remove), sorts only on query: `get_top_levels(n)`, `get_best_price()`, `get_total_depth(n)`; per-level order_count not tracked.
- `get_features(n_levels=5)`: `best_bid, best_ask, mid_price, spread, spread_bps, bid_depth, ask_depth, total_depth, depth_imbalance = (bid-ask)/total` in [-1,1], `bid_size_{1..n}, ask_size_{1..n}, bid_price_{1..n}, ask_price_{1..n}` (pad 0/NaN).
- `LOBReconstructor(snapshot_freq_ms=5000, n_levels=5, price_scale=None).process(df)`: numpy arrays (5-10x faster than `to_dicts()`), integer ns timestamps. Params `SYMBOL="NVDA"`, `SNAPSHOT_FREQ_MS=5000`, `N_LEVELS=5`; demo = first 10 minutes. Output `output/databento/{SYMBOL}_lob_features.parquet`.
- Depth-imbalance test: scatter `depth_imbalance` vs next `mid_price.pct_change()*1e4`; read shape (tilt vs blob). Snapshot quantity, distinct from OFI (flow over an interval).

### 3.8 Order-flow predictability and markouts (`09_databento_mbo_analysis`)

- `load_databento_day`: price/1e9 if max > 1e6; `timestamp` Datetime("ns") naive UTC; convert to America/New_York before RTH.
- Book pressure `P = sum s_i w_i q_i exp(-lambda d_i)`: `s` = +1 bid / -1 ask; `w` = +1 add / -1 cancel / +0.5 fill; `q` = size; `d` = |price - rolling mid| (mid = rolling mean of last 100 trade prices via backward `join_asof`, no lookahead); `decay_lambda=0.01`; then `ewm_mean(half_life=100)`.
- `reconstruct_bbo_bars(df, "1m")`: order-keyed top-of-book replay (A/C/F/M/R; T ignored); rescan only when the inside level empties; last quote per bar.
- `process_day_to_bars`: bars from T records in RTH; `buy_volume` (side B), `sell_volume` (side A) **cast Int64**; `ofi = (buy - sell)/(buy + sell)`; `vwap = notional/volume`; join BBO; `mid_quote = (best_bid + best_ask)/2`. Output `output/databento/NVDA_minute_bars.parquet`.
- `compute_markouts(bars, horizons=[1,5,10,30], latency_bars=1)` within `session_date`:

| Markout | Formula | Reads |
|---|---|---|
| midpoint | `mid.shift(-h)/mid - 1` | generous: fills at mid |
| latency-adjusted | `mid.shift(-h)/mid.shift(-latency) - 1` (identically 0 at h = latency; report h > `LATENCY_BARS`) | alpha that decays in flight |
| executable | `best_bid.shift(-h)/best_ask - 1` | pays ask, exits at bid |

- Predictors `ofi_lag1 = ofi.shift(1).over(session)`, `ofi_ma5 = ofi.rolling_mean(5).shift(1)`. Table: horizon, n, Pearson corr, corr/(1/sqrt(n)) "in naive SEs" (upper bound). Deciles `pd.qcut(ofi_lag1, 10)`: mean 5-min markout per decile in bps, top-minus-bottom annotated; histograms clipped at 1/99 pct; "spread cost = mid_mean - exec_mean". Params `SEED=42`, `MAX_FILES=None`, `MAX_ROWS=None`, `LATENCY_BARS=1`.

### 3.9 IEX DEEP L2 book (`10_iex_lob_reconstruction`)

- `iex_parser` (`DEEP_1_0`, `TOPS_1_6`, `Parser`); messages `price_level_update` (new total size at a price; 0 removes level), `quote_update`, `trade_report`. Normalise side: raw may be "B"/"S", bytes, 0/1, "buy"/"sell", 66/83.
- `IEXOrderBook` keeps aggregated size per price; `reconstruct_lob_snapshots(price_levels, symbol, snapshot_interval_sec=1.0)` -> `timestamp, bid_price, bid_size, ask_price, ask_size, spread, midpoint`. Completeness = share of snapshots with both sides; one-sided -> null spread.
- Load `load_iex_hist(feed="deep", get_raw_files=True)`; download `iex_download.py --deep --smallest`. Output `output/iex_deep/{quotes,trades,price_levels}/{YYYYMMDD}.parquet`. Params `MAX_MESSAGES=0`, `SAVE_PARSED=True`. The `--smallest` file is a non-trading day with test symbols only (ZIEXT, ZEXIT, ZXIET).
- IEX vs ITCH: L2 vs L3; 11 vs ~20 message types; 350 µs speed bump; free vs licensed; ~100k-1M vs millions of messages/day.

### 3.10 TAQ crash-day EDA and NBBO cleaning (`11_algoseek_taq_eda`)

1. `load_nasdaq100_taq(symbols=["AAPL"], start_date="2020-03-16", end_date="2020-03-16")`; RTH on `timestamp.dt.time()` in [09:30, 16:00].
2. Trades: `event_type == "TRADE"`, drop `conditions == "80000002"` (late-reported; carries prior close), symbol-day price band `235 <= price <= 265` (catches $277.97 prints).
3. NBBO: `event_type` contains "NB" (`QUOTE BID NB`/`QUOTE ASK NB`), `price > 0` (zero = empty side), pivot + forward_fill + drop_nulls; sample `group_by_dynamic("1s")` last; `spread_bps`; filter `spread > 0` (drop locked/crossed) and `spread_bps < 500`. Stats mean/median/p95/max; plot cap p99.
4. Venue share of executed volume (top 10): group Cboe (BATS, EDGX, EDGA, BATS Y) and NYSE (NYSE, Arca, National).
5. Trade-size buckets: odd lot < 100, small 100-500, medium 501-2000, large > 2000; count% vs volume%.
6. 5-min OHLCV with `trade_count`; drop empty bars. Pre-split prices (AAPL 4:1 on 2020-08-31): divide by 4 before comparing with adjusted series. ~22 GB.

### 3.11 Trade classification: venue label, Lee-Ready, tick test (`12_algoseek_taq_lob_reconstruction`, `15_itch_lee_ready`, `limit_orderbook.classify_trades_lee_ready`)

| Method | Rule | When |
|---|---|---|
| Venue label | DataBento T record `side` B -> +1, A -> -1 (**aggressor** on T; resting-order reading applies to F); N skipped | always when present |
| Lee-Ready | `quote_rule = sign(trade_price - midpoint)`; at midpoint, `tick_rule = diff().sign().fill_null(0)` with last direction carried | no label, quotes/book available |
| Tick test | `sign(price - price.shift(1))`, zero -> last non-zero direction | nothing else exists |

- TAQ as-of NBBO (nb 12): bids `QUOTE BID`, asks `QUOTE ASK`, trades `TRADE`; `concat(how="diagonal")`, sort, `forward_fill` bid/bid_size/ask/ask_size, keep trade rows, drop nulls, `spread > 0`. Minute stats `signed_volume = size x sign`, `imbalance = signed/volume`, `return = close/open - 1`; `pl.corr(imbalance, return)` is **contemporaneous**.
- Validation (nb 15, Table 3.3): `load_databento_mbo` (`ts_event -> timestamp`, Datetime("ns") UTC, price/1e9 if max > 1e6, RTH via `replace_time_zone("UTC").convert_time_zone("America/New_York")`, 09:30 <= t < 16:00); book `{"B": Counter, "A": Counter}` + registry with `_update_book` for R/A/M/C/F; `_apply_lee_ready(price, book, last_price, last_tick_dir)` (no book -> tick test only; no history -> 0). Cohorts: **continuous** (coverage 100%) vs **non-zero** (score only price-moving trades; coverage = share of non-zero ticks). Aggregate over `MAX_VALIDATION_DAYS=5`; output `output/databento/table_3_3_classification_accuracy.parquet` (note `OUTPUT_DIR = get_output_dir(3, "algoseek")` is defined but unused).
- Decision time: validation uses exchange timestamps; in a backtest apply ~1 ms (co-located) to ~10 ms (retail) observation lag before trusting classification.
- ITCH `classify_trades_lee_ready(itch_dir, symbol, start_time, end_time)` -> `timestamp, price, shares, side` in {1,-1,0}; rename shares -> volume for the bar library.

### 3.12 AlgoSeek TAQ minute bars (`13_algoseek_minute_bars_eda`)

- `load_nasdaq100_bars(symbols=..., start_date="2020-01-01", end_date="2020-12-31", include_microstructure=True)` from Hive `equities/nasdaq100/minute_bars/year=YYYY/month=MM.parquet`; 61 columns, continuous 04:00-20:00 ET rows.

| Family | Columns |
|---|---|
| Identifiers | date, symbol, time, timestamp, year, month |
| Quote OHLC | `open/high/low/close_bid/ask_price`, `..._size`, `open_bar_time, high_bid_time, ...` |
| Trade OHLC | `first/high/low/last_trade_price/size/time` |
| Spread | `min_spread, max_spread` |
| Volume | `volume, finra_volume, total_trades, cancel_size` |
| Trade buckets | `trade_at_bid, trade_at_bid_mid, trade_at_mid, trade_at_mid_ask, trade_at_ask, trade_at_cross` |
| Tick direction | `uptick_volume, downtick_volume, repeat_uptick_volume, repeat_downtick_volume, unknown_tick_volume` |
| Pressure and VWAP | `vwap, finra_vwap, trade_to_mid_vol_weight, trade_to_mid_vol_weight_rel, time_weight_bid, time_weight_ask` |
| Quote count | `nbbo_quote_count` |

- Quality: `utils.data_quality.describe_coverage, per_asset_stats, null_rate, check_ohlc_invariants` ([WARN] if valid_pct < 99.99). Trade fields null when no trade (expected ~20% with extended hours); quote fields never null.
- `mid_price = (open_bid + open_ask)/2`; `spread_bps` when mid > 0; ECDF clipped p99.
- OFI: `buy_aggressor = trade_at_ask + trade_at_mid_ask`, `sell_aggressor = trade_at_bid + trade_at_bid_mid`, `OFI = (buy - sell)/(buy + sell)` else 0; exclude `trade_at_cross`.
- Tick ratio counts repeat ticks with parent direction. Pressure `trade_to_mid_vol_weight = sum((price - mid) x vol)/sum(vol)` in dollars; `_rel` divides by spread (mid 0, ask +0.5, bid -0.5).
- `total_volume = volume + finra_volume`; `finra_share`; condition on **changes**, not level. `min_spread == 0` -> locked/crossed rate; RTH hourly averages for hours 10-15. Params `MAX_SYMBOLS=10`, `END_DATE="2020-12-31"`.

### 3.13 Bar sampling: time, tick, volume, dollar (`14_itch_bar_sampling`, `ml4t.engineer.bars`)

1. `load_itch_trades(symbol)`: P messages, sort by timestamp, price/10000. **ITCH P `buy_sell_indicator` is 'B' for every trade** (resting order's side) -> tick test `side = +1 if diff > 0, -1 if diff < 0, else forward-fill; first trade +1`. Library input contract: `timestamp, price, volume` (+ `side` for imbalance/run bars).
2. RTH 09:30-16:00 exchange-local (ITCH naive local; never UTC). Sort before `group_by_dynamic`.
3. Samplers: `TickBarSampler(ticks_per_bar=100)`, `VolumeBarSampler(volume_per_bar=10_000)`, `DollarBarSampler(dollars_per_bar=3_000_000)` (matched to ~$320 AAPL pre-split), `FixedTickImbalanceBarSampler(threshold=20)` (both sides required), `TickRunBarSampler(expected_ticks_per_bar=50, alpha=0.001)`; time bars `group_by_dynamic("1m")`. All `.sample(data, include_incomplete=False)`.
4. `analyze_returns`: `pct_change` of close; `scipy.stats.jarque_bera`, lag-1 autocorr, skew, excess kurtosis (headline chart, 0 = normal); histograms clipped ±2%; durations `diff(timestamp)` in seconds.
5. Imbalance from volume bars: cast `buy_volume/sell_volume` to float64 **before** subtracting.
6. Lee-Ready variant: `FixedTickImbalanceBarSampler(threshold=20)` on `side != 0` if > 50% usable; compare bar counts and JB with tick-test bars. Outputs `output/nasdaq_itch/bars/AAPL_{time_1m,tick_100,volume_10k,dollar_3m,imbalance_tick,imbalance_lee_ready}_bars.parquet`.

| Bar type | Prefer when |
|---|---|
| Time | calendar alignment, familiarity |
| Tick | high-frequency signals |
| Volume | activity clock; sensitive to price level |
| Dollar | general ML features (value-weighted, stable, absorbs price moves) |
| Imbalance / run | information arrival; needs classified sides, parameter care |

### 3.14 Information-driven bars (`16_itch_information_bars`, `ml4t.engineer.bars`)

- Tick imbalance: `theta_T = sum b_t`, `b_t` in {-1,+1}; threshold `E[theta_T] = E[T] x |2P[b=1] - 1|`; close when `|theta| >= E[theta_T]`. Volume imbalance: `theta = sum b_t v_t`, threshold `E[T] x |2v+ - E[v]|`, `v+ = P[b=1] x E[v|b=1]`. Run: `E[theta] = E[T] x max(P[b=1], 1-P[b=1])`, `theta = max(sum buys, sum sells)`.
- Adaptive update (`calculate_tick_imbalance_bars_manual(sides, expected_t=1000, alpha=0.1, min_bars_warmup=10)`): `p_buy` from first `min(1000, n)` sides; on each close if `n_bars > min_bars_warmup`: `E[T] <- a x bar_ticks + (1-a) x E[T]`, `p_buy <- a x bar_p_buy + (1-a) x p_buy`. Verified against `TickImbalanceBarSampler` (`expected_imbalance` column) with `VERIFY_ET=1000, VERIFY_ALPHA=0.001, VERIFY_WARMUP=100`.

| Class | Signature / notes |
|---|---|
| `TickBarSampler(ticks_per_bar)`, `VolumeBarSampler(volume_per_bar)`, `DollarBarSampler(dollars_per_bar)` | vectorized |
| `TickImbalanceBarSampler(expected_ticks_per_bar, alpha=0.1, initial_p_buy=0.5, min_bars_warmup=10)` | adaptive TIB; outputs `expected_t`, `expected_imbalance`, `buy_volume`, `sell_volume` |
| `ImbalanceBarSampler` | VIB, vectorized, same signature |
| `FixedTickImbalanceBarSampler(threshold)`, `FixedVolumeImbalanceBarSampler(threshold)` | production default; `threshold ~ ticks_per_day / N x |2P[b=1] - 1|`, typical 50-500 |
| `WindowTickImbalanceBarSampler(initial_expected_t, bar_window=10, tick_window=1000)`, `WindowVolumeImbalanceBarSampler` | `bar_window` 5-20, `tick_window` 1000-10000 |
| `TickRunBarSampler/VolumeRunBarSampler/DollarRunBarSampler(expected_ticks_per_bar, alpha=0.1, initial_p_buy=0.5, min_bars_warmup=10)`, `FixedTickRunBarSampler(threshold)` | run bars (`ml4t.engineer.bars.run`, also top-level export) |

- Diagnostics `compute_stats`: JB, AC(1), `VR(5) = Var(5-bar sums)/(5 x Var(1-bar))` (1 random walk, > 1 momentum, < 1 mean reversion); >= 30 bars.
- Sweeps: `TIB_ET=[500,700,1000,1500,2000,3000]`, `VIB_ET=[2000,5000,10000,20000,50000]`, `ALPHA=0.001`, `WARMUP=100`; three-approach comparison at `COMPARE_ET=1000`: alpha in {0.001, 0.01, 0.1}; fixed in {50, 100, 200}; window `tick_window` in {2000, 5000, 10000}, `bar_window=10`. `et_drift = expected_t[last]/expected_t[first]`. Data: NVDA MBO trades, `MAX_DAYS=3`.

### 3.15 Multi-day threshold calibration (`17_databento_bar_sampling`, Table 3.7)

1. `_load_day_trades`: T records, RTH via America/New_York, side B->+1, A->-1, N->0; `trades.filter(side != 0)` for **every** bar type (print unclassified share).
2. Daily profile on all prints: trade_count, volume, dollar volume with CV = std/mean; large CV means a single-day threshold does not carry.
3. Grids: time [1,2,5,10] min; tick [500,1000,2000,4000]; volume [25k,50k,100k,200k]; dollar [5M,10M,15M,25M]; `VIB_EXPECTED_T=[5k,10k,20k,50k]`; `TIB_EXPECTED_T=[500,1k,2k,3k]`; `IMBALANCE_ALPHA=[0.001]` ("Not 0.1!"); `IMBALANCE_WARMUP=100`. `_build_imbalance_bars` returns None if < 50% sided; `build_bars_for_day` returns None if < 100 trades or <= 10 bars.
4. Best threshold per type = daily bar count closest to `TARGET_BARS_PER_DAY=500` (~1/min RTH); text says median day, code uses mean (minor contradiction). Fallbacks TIB "E[T]=1000, alpha=0.001", VIB "E[T]=10000, alpha=0.001".
5. `compute_bar_statistics` on log returns: JB, skew, excess kurtosis, AC(1), VR(5) (> 10 bars); >= 30 bars and >= 20 returns.
6. Table 3.7 configs (first day): Time 1-min; Tick 500; Volume 50K; Dollar $5M; Vol Imbalance `build_bars_for_day(day, "imbalance", 500, alpha=0.001)`; report N bars, JB, AC(1), P[buy]. Timing figure dollar threshold `max(100_000, total_dollar/500)`.
7. Outputs `output/databento/NVDA_calibration_results.parquet`, `NVDA_bar_statistics.parquet`, `NVDA_daily_profile.parquet`, `NVDA_{time,tick,volume,dollar,tick_imbalance,volume_imbalance}_bars.parquet` (flagged for Ch5 GT-GAN). Params `N_DAYS=10`, `MAX_ROWS_PER_DAY=0`.

### 3.16 Jump detection: bipower variation + Lee-Mykland (`18_algoseek_jump_detection`)

1. `load_nasdaq100_bars(frequency="5m", symbols=["AMD","AMZN","FB"], 2020, regular_hours=True)`; log returns per (symbol, date), never bridging overnight; `MU1 = sqrt(2/pi)`, `BARS_PER_DAY = 78`.
2. Daily: `RV = sum r_i^2`; `BV = mu1^-2 sum |r_i||r_{i-1}|`; `jump_var = max(RV - BV, 0)`; `jump_share_rv = jump_var/RV` clipped [0,1]; annual sums, `cont_vol_annual = sqrt(BV_annual)`.
3. Lee-Mykland: `L_i = r_i / sigma_i`, `sigma_i^2 = (1/(K-2)) sum_{j=i-K+2}^{i-1} |r_j||r_{j-1}|` within session (first K-1 bars untestable); implement via cumulative pair products `sigma_sq = (ps_cum[i-1] - ps_cum[i-k+1])/(k-2)`. Reject if `|L_i| > S_n beta*(alpha) + C_n`, `C_n = sqrt(2 ln n)/mu1 - (ln pi + ln ln n)/(2 mu1 sqrt(2 ln n))`, `S_n = 1/(mu1 sqrt(2 ln n))`, `beta* = -ln(-ln(1-alpha))`; `n` = median bars per session (one threshold for the study). K recommended ~sqrt(n) (~9); used `LEE_MYKLAND_K=12` (~60 min). `JUMP_ALPHA=0.01`.
4. Features: `jump_count`, `signed_jump_var = sum is_jump x r^2 x sign(r)`, RV/BV/jump_var/jump_share; panel `output/jump_features/daily_jump_features.parquet` (symbol, date, jump_count, signed_jump_var, jump_variance, jump_share, rv_total, continuous_variance).
5. Naive comparison: `naive_z = r / daily_std`, flag `|z| > NAIVE_Z_THRESHOLD=4.0`; bucket both / LM-only / naive-only. Time-of-day: jumps per 1,000 bars by minute; first/last 30 min shares.

### 3.17 Data-quality invariants and sessionization (section 3.6, checks scattered through nbs 01-18)

| Invariant | Check |
|---|---|
| Sequencing | sort `(timestamp, tracking_number)` ITCH; `(ts_event, order_id)` DataBento; deletes last among ties |
| Timezone | ITCH naive exchange-local; DataBento UTC -> America/New_York; production uses a market calendar with DST/holidays |
| Session | RTH 09:30-16:00 ET before any per-bar statistic; never bridge overnight |
| Book validity | crossed count `(spread < 0)`; `unknown_order_count`, `negative_size_count` |
| Quotes | `price > 0`; `spread > 0`; `spread_bps < 500`; `min_spread == 0` rate |
| Prints | condition `80000002`; symbol-day price band; p1/p99 winsorization in thin names |
| Corporate actions | split-adjust ticks before joining adjusted series (AAPL 4:1 2020-08-31); ticker-as-of-date (FB not META in 2020) |
| Bars | `check_ohlc_invariants` >= 99.99%; nulls = no trades; per-symbol coverage >= 50% of longest symbol |
| Joins | assert unique keys; schema-per-type scans; Int64 before subtraction; µs vs ns units |

## Guardrails and pitfalls

- **Replace chains lost** — U issues a new reference absent from adds; resolving D/X/E against adds alone drops every replaced order. Keep a live pool updated by U (nb 02) or pointer-double chains with a cycle cap (nb 05); print unresolved counts.
- **Delete sized at original shares** — double-counts partial fills, drives levels negative. Track remaining shares; size deletes as `shares - removed_before` with deletes sorted last among ties.
- **Cold-start book** — reading from the open hits unknown references on pre-market adds deleted after 09:30. Process from start of day, keep snapshots from START_TIME; expect and print `unknown_order_count` on mid-session windows.
- **Crossed quotes** — bid > ask is impossible; count `(spread < 0)` as the reconstruction error rate.
- **Ignoring C executions** — C removes `executed_shares` at the resting price; include C in removals and first-fill logic; use `execution_price` only for price improvement.
- **P/T trades touching the book** — P (ITCH) and T (DataBento) are tape only; F updates the book.
- **ITCH P side field** — always 'B' (resting side); signed bars degenerate to tick bars. Infer with tick test or Lee-Ready; assert both sides present before imbalance/run bars.
- **Unsigned integer subtraction** — UInt32 `100 - 200` wraps to 4,294,967,196; cast to Int64/float64 before OFI or imbalance (nb 03, 09, 14).
- **Price scaling** — ITCH price4 (/10000), DataBento nanodollars (/1e9); heuristics median > 10000 / max or median > 1e6; print min/median/max.
- **Timezone confusion** — ITCH naive local, never UTC; DataBento `replace_time_zone("UTC").convert_time_zone("America/New_York")` before RTH; a fixed UTC window is an hour wrong half the year (nb 08's 13:30-21:00 UTC is an approximation).
- **Sequencing ties** — multiple messages per nanosecond; sort `(timestamp, tracking_number)` / `(ts_event, order_id)`; Polars does not preserve input order on tied keys, sort deletes last explicitly.
- **Polars timestamp units** — µs out of Polars, ns for pandas; cast `datetime64[ns]`; `dt.cast_time_unit("us")` before joins.
- **Schema mismatch A vs F** — F has `attribution`; scan per type; `concat(how="diagonal_relaxed")`.
- **Duplicate join keys** — duplicated `order_reference_number` multiplies rows; assert `n_unique == height`; `unique(subset=..., keep="first")` on U links.
- **Sentinel prices in adds** — pegged/far orders at ~0 or 1e5; read p1-p99, not min/max.
- **Bad prints in ITCH trades** — winsorize returns p1/p99; range as `(p95 - p5)/median`.
- **Late-reported TAQ trades** — condition `80000002` carries the prior close; drop by code plus symbol-day band.
- **Zero-price NBBO quotes** — empty side published as 0; filter `price > 0`, `spread > 0`, `spread_bps < 500`.
- **Mean spread on a dislocated day** — heavy tail; report median, p95, max; cap axes, read tails from percentiles.
- **Split-unadjusted tick prices** — 4x jump at a join; adjust first.
- **Ticker history** — AlgoSeek stores ticker as of date (FB in 2020); assert per-symbol coverage, raise if missing or < 50% of longest.
- **Null semantics** — minute-bar trade fields null when no trade (~20% with extended hours); never impute; report null rate by session.
- **Blending sessions** — pre/post-market flattens the U-shape and inflates spreads; filter RTH first.
- **Bar index is not the clock** — label by `slot_label(time_slot)`, not index.
- **Un-normalised averaging** — express each ticker's bars as share of its own daily total.
- **Trade-price returns** — bid-ask bounce gives mechanical negative AC(1); use mid prices; trade prices only to demonstrate the bounce.
- **`min_spread == 0`** — locked/crossed is a stress/data condition; exclude `trade_at_cross` from OFI.
- **FINRA share level** — mixes dark-pool and retail internalisation; condition on changes vs recent level.
- **Contemporaneous correlation read as signal** — lag the predictor `ofi.shift(1).over(session)`.
- **Naive standard errors** — overlapping forward returns shrink effective n; `corr/(1/sqrt(n))` is an upper bound; never read sign flips across horizons as horizon effects.
- **Signal without cost** — compute midpoint, latency-adjusted (`LATENCY_BARS=1`) and executable markouts; compare decile spread to ~1-2 bps round trip.
- **In-sample "predictability"** — nb 03 correlations have no holdout; label as descriptive.
- **Heavy-tailed OFI in Pearson** — signed log `sign(x) x log1p(|x|)`; report both.
- **Thin symbols in a cross-section** — `MIN_BUCKETS=50`; print survivors; result is conditioned on activity.
- **Re-sampled cross-section** — fix the 50 symbols (5 strata x 10).
- **Termination is not "never traded"** — partial fill then delete sits in both rates; use one-outcome-per-order classification; count first event only.
- **Replace counted as cancellation** — break termination down by D/X/U.
- **Whole-second durations** — sub-second lifetimes become zero; `total_nanoseconds()/1e9`; log-axis histograms.
- **Right-censoring at the close** — report "still live or unknown", not "never terminated".
- **Venue-wide vs per-symbol definitions** — first D / first E only; report separately, never compare directly.
- **Adaptive imbalance threshold spiral** — EWMA of E[T] and P[b=1] from the sampler's own bars; one-sided flow (NVDA 52-60% buys) raises thresholds -> longer bars -> higher estimates, or collapses to one trade per bar. Use alpha=0.001 (not 0.1), warm-up 100, `FixedTickImbalanceBarSampler` in production, or window samplers; diagnose via bar count, avg bar size, `et_drift`, and `expected_t x expected_imbalance` bar by bar (endpoint ratios hide oscillation).
- **Threshold units across samplers** — TIB in trades, VIB in shares; TIBs give ~800x more bars at the same E[T]; calibrate each on its own grid.
- **Single-day calibration** — daily volume CV is large; calibrate across days off the median day; decide acceptable bar-count variance before looking.
- **Volume threshold after a price move** — prefer dollar bars.
- **Dropping unlabeled trades** — DataBento side N; print the unclassified share; build every bar type from the same classified set.
- **Tick-test coverage illusion** — non-zero cohort scores only moving prices; print coverage beside accuracy.
- **Lee-Ready with exchange timestamps in backtests** — apply ~1 ms (co-located) to ~10 ms (retail) lag.
- **Latency-adjusted markout at h = latency** — zero by construction; report h > LATENCY_BARS only.
- **Jump test bridging overnight** — overnight gap reads as a jump; compute per (symbol, date); accept first K-1 bars untestable.
- **Per-bar significance in jump detection** — dozens of bars/day give false jumps daily; use the Gumbel critical value with `C_n, S_n` from median session length.
- **Naive |z| > 4 with daily sigma** — over-detects calm days, under-detects volatile ones; use local bipower sigma.
- **Memory** — nbs 04/05/11/12/03 peak 33/31/22/22/13 GB and exceed 100 GB cumulatively; predicate pushdown, column selection, kernel restarts (Polars arenas release only on restart), `MAX_*` caps for trials; Numba buffer bounds.
- **Parameter caps distort diagnostics** — `MESSAGE_LIMIT`/`MAX_MESSAGES` truncate each type independently; missing references or absent P under a cap are not parse failures; trust only complete days.
- **Empty-store sentinel** — creating `messages/` before parsing defeats `must_exist`; create only when about to write.
- **Cached results** — `REBUILD_ENRICHED=False` reads cache; committed runs rebuild.
- **Derived data location** — write to chapter `output/`, never into `ML4T_DATA_PATH`.
- **IEX test symbols** — `--smallest` validates the parser only; download a full trading day for real figures.
- **Price-level sort cost** — `Counter` levels, sort on query only; numpy arrays, not `to_dicts()`.

## Decision rules and defaults

| Decision | Rule |
|---|---|
| Parser | learning/debugging -> Python notebook; single day -> either; multi-day backtest/production -> Rust `ml4t/itch-parser` (same schema) |
| Data source | ITCH: free, binary, all NASDAQ, one sample day, no aggressor label -> learning/backtesting. DataBento: ~$10/symbol/month, Parquet, multi-exchange, aggressor on T, multi-day -> production/research and any calibration. IEX DEEP: free L2 -> spread/depth dynamics, not per-order questions. TAQ: consolidated NBBO + cross-venue trades, no depth. AlgoSeek minute bars: ML-ready pre-computed fields |
| Snapshot frequency | 1 s default (tens of thousands rows/day); 100 ms = 10x rows/memory; 5 s for DataBento demo; only {100ms, 500ms, 1s, 5s, 10s, 1min} |
| Depth levels | 1 for touch; 5 when depth beyond the touch matters |
| OFI bucket | 1 min; `MIN_BUCKETS=50` per symbol; >= 3 symbols per cross-section; >= 5 buckets per correlation |
| Liquidity floors | `MIN_TRADES=500` trades/day for tiers; >= 100 one-second bars for autocorrelation; >= 10 one-minute bars for liquidity metrics; top-20 tickers and 30-min bars (13/session) for the U-shape |
| Trade-size buckets | odd lot < 100; small 100-500; medium 501-2000; large > 2000 |
| NBBO cleaning | `price > 0`; `spread > 0`; `spread_bps < 500`; drop `80000002`; symbol-day price band |
| RTH | 09:30-16:00 ET; ITCH naive local; DataBento convert from UTC |
| Trade direction | venue label (DataBento T.side) > Lee-Ready with reconstructed book > tick test; backtest lag 1-10 ms |
| Bar type | general ML features -> dollar; information arrival -> imbalance; HF signals -> tick; calendar alignment -> time |
| Imbalance sampler | production -> `FixedTickImbalanceBarSampler`; research -> `TickImbalanceBarSampler(alpha=0.001, min_bars_warmup=100)` or window-based; require > 50% sided trades and both directions |
| Calibration | target ~500 bars/day (~1/min RTH); sweep a grid; pick threshold whose median-day (text) / mean (code) bar count is closest; check day-to-day std vs mean; then JB lower, abs AC(1) nearer 0, kurtosis lower, VR(5) near 1. Fixed thresholds `ticks_per_day/N x abs(2P[b=1]-1)`, typical 50-500 |
| Example thresholds | AAPL ~$320 (2020-01-30): tick 100, volume 10K, dollar $3M, fixed tick-imbalance 20, run E[T]=50 alpha=0.001. NVDA 2024 (Table 3.7): tick 500, volume 50K, dollar $5M, VIB E[T]=500 alpha=0.001 |
| Adaptive health | bar count and avg bar size vs target; `et_drift` far from 1 means adaptation moved (near 1 is not proof it did not); read `expected_t x expected_imbalance` bar by bar |
| Signal go/no-go | (1) lag the predictor; (2) correlation vs honest SE; (3) decile means show a monotone ramp or standout tails; (4) top-minus-bottom in bps > ~1-2 bps round trip after latency (1 bar) and executable markouts; otherwise use flow as a filter |
| Jump detection | 5-min bars; K=12 (~sqrt(n) ~ 9 minimum); alpha=0.01 family-wise per session; one threshold from median session length; features jump_count, signed_jump_var, jump_share |
| OHLC invariants | valid_pct >= 99.99% else WARN |
| Memory sequencing | run 04, 05, 11, 12, 03 one at a time with kernel restarts on < 32 GB; >= 32 GB recommended (64 ideal) |

## Code patterns and APIs

Run: `uv run python 03_market_microstructure/<notebook>.py`; tests `uv run pytest tests/test_chapter_notebooks.py -v -k "03_market_microstructure"` (Papermill injects reduced parameters via `parameters`-tagged cells). Env: `ML4T_DATA_PATH` (data root), `ITCH_SYMBOL` (nb 02 batch), `ITCH_KEEP_EXISTING=1` (nb 01). Outputs via `get_output_dir(3, "<name>")` -> `03_market_microstructure/output/{nasdaq_itch,lob_analysis,databento,iex_deep,jump_features,algoseek}/`.

Imports in use: `polars as pl`, `numpy`, `pandas`, `pyarrow.dataset`, `numba`, `scipy.stats` (jarque_bera, skew, kurtosis, norm), `matplotlib`, `plotly`, `seaborn`, `tqdm`, `struct`, `iex_parser`; repo helpers `from data import load_nasdaq_itch, load_mbo_data, load_nasdaq100_taq, load_nasdaq100_bars, load_iex_hist`; `from utils.paths import display_path, get_output_dir, require_chapter_inputs`; `from utils.style import COLORS, FIGSIZE, show_with_alt, show_plotly_with_alt, add_message_title`; `from utils.data_quality import check_ohlc_invariants, describe_coverage, null_rate, per_asset_stats`; `from utils.reproducibility import set_global_seeds`; `from utils import ML4T_PATH`; chapter-local `from itch_message_specs import ...`, `from limit_orderbook import get_stock_locate_mapping, load_itch_messages, reconstruct_lob_with_ofi, classify_trades_lee_ready`; library `from ml4t.engineer.bars import TickBarSampler, VolumeBarSampler, DollarBarSampler, ImbalanceBarSampler, TickImbalanceBarSampler, FixedTickImbalanceBarSampler, WindowTickImbalanceBarSampler`, `from ml4t.engineer.bars.run import TickRunBarSampler`.

Loader signatures (`data/equities/loader.py`):
- `load_nasdaq_itch(message_types=None, symbols=None, get_base_path=False, must_exist=True) -> DataFrame | Path`
- `load_mbo_data(symbols=None, start_date=None, end_date=None, list_files=False) -> DataFrame(ts_event, symbol, action, side, price, size, order_id, flags) | list[Path]` (NVDA, Nov 2024, 10 days)
- `load_nasdaq100_taq(symbols=None, event_types=None, start_date=None, end_date=None)` (AAPL 2020-03-13 and 2020-03-16; event types TRADE, TRADE NB, TRADE CANCELLED, QUOTE BID/ASK, QUOTE BID NB/ASK NB)
- `load_nasdaq100_bars(frequency="1m", symbols=None, start_date=None, end_date=None, include_quotes=False, include_microstructure=False, regular_hours=True, lazy=False, max_symbols=0)` (frequency resampling ignored when `include_microstructure=True`)
- `load_iex_hist(feed="deep"|"tops", data_type="all"|"quotes"|"trades"|"price_levels", symbols=None, dates=None, get_raw_files=False)`

Predicate-pushdown load with per-type filter column:
```python
lf = pl.scan_parquet(msg_dir / "*.parquet")
schema = lf.collect_schema()
if "stock" in schema and symbol:
    lf = lf.filter(pl.col("stock") == symbol)
elif "stock_locate" in schema and stock_locate is not None:
    lf = lf.filter(pl.col("stock_locate") == stock_locate)
df = lf.head(limit).collect() if limit else lf.collect()
```

Row counts from Parquet metadata: `n = sum(f.metadata.num_rows for f in ds.dataset(path.as_posix(), format="parquet").get_fragments())`.

Pointer doubling for replace chains (nb 05):
```python
doubled = links.join(links.select(pl.col("order_reference_number").alias("ancestor"),
                                  pl.col("ancestor").alias("grandparent")),
                     on="ancestor", how="left")
links = doubled.select("order_reference_number",
                       pl.coalesce("grandparent", "ancestor").alias("ancestor"), "price")
```

Delete sizing in hand-built OFI (nb 03):
```python
sized = pl.col("shares_removed").fill_null(0).cast(pl.Int64)
removals = removals.sort(["order_reference_number", "timestamp",
                          (pl.col("event_type") == "delete")]) \
    .with_columns((sized.cum_sum() - sized).over("order_reference_number").alias("removed_before"))
```

UTC -> exchange-local RTH filter (DataBento):
```python
_et = pl.col("timestamp").dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York")
df = df.filter(((_et.dt.hour() > 9) | ((_et.dt.hour() == 9) & (_et.dt.minute() >= 30)))
               & (_et.dt.hour() < 16))
```

Lee-Ready in Polars (nb 12):
```python
quote_rule = pl.when(tp > mid).then(1).when(tp < mid).then(-1).otherwise(0)
tick_rule  = pl.col("trade_price").diff().sign().fill_null(0)
trade_sign = pl.when(quote_rule != 0).then(quote_rule).otherwise(tick_rule)
```

Tick-test side for the bar library (nb 14): `side = when(diff > 0).then(1).when(diff < 0).then(-1).otherwise(None)`, then `forward_fill().fill_null(1)`.

Markouts within session (nb 09): `mid.shift(-h).over("session_date")/mid - 1`; executable `best_bid.shift(-h).over(session)/best_ask - 1`. Signed log before Pearson: `np.sign(x) * np.log1p(np.abs(x))`.

Bar samplers: `TickBarSampler(ticks_per_bar=100).sample(trades)` with `trades` columns `timestamp, price, volume, side`; `FixedTickImbalanceBarSampler(threshold=20)`; `TickImbalanceBarSampler(expected_ticks_per_bar=1000, alpha=0.001, min_bars_warmup=100)`; `WindowTickImbalanceBarSampler(initial_expected_t=1000, bar_window=10, tick_window=5000)`; `TickRunBarSampler(expected_ticks_per_bar=50, alpha=0.001)`.

Lee-Mykland threshold (nb 18): `gumbel_threshold(n, alpha)` per section 3.16; session-wise sigma via cumulative pair products.

Drill-down paths (repo root): `03_market_microstructure/limit_orderbook.py` (1287 lines; Numba kernel ~L483-753; Lee-Ready ~L1016), `03_market_microstructure/itch_message_specs.py`, `data/equities/loader.py` (loaders at L314, L905, L984, L1115, L1295); library `libs/src/ml4t_engineer/ml4t/engineer/bars/{base,vectorized,imbalance,run,tick,volume}.py` and `AGENTS.md` (`docs/user-guide/bars.md` absent from the checkout).

## Evidence from the book

Most printed results were condensed out of the digest; figures below come from narrative, docstrings and READMEs. Author caveats apply throughout: one venue, one session (ITCH 2020-01-30), one symbol (NVDA/AAPL), an exceptional day (2020-03-16), no holdout; illustrations of regularities, not evidence for them; cost figures are typical magnitudes, not measured.

- **Table 3.1 parse cost (same 13 GB ITCH day)**: Python ~23 min, ~8 GB; Rust < 5 min, < 500 MB (order of magnitude, not a ratio). README: `01_itch_parser` ~22 min, ~8 GB RSS for 423M messages.
- **Peak RSS**: nb 04 ~33 GB, nb 05 ~31 GB, nbs 11/12 ~22 GB, nb 03 ~13 GB; others < 8 GB.
- **Message composition**: adds and cancels take almost all of an MBO day, trades a sliver; TAQ quotes outnumber trades ~10:1 (AAPL 2020-03-16 RTH). F (attributed add) ~1% of adds overall, up to a third for some thin names.
- **Replace chains** run ~10,000 deep for a market maker rewriting one quote; pointer doubling closes that in 14 passes.
- **Table 3.3 classification (NVDA, 5 days)**: tick test ~78% accurate, Lee-Ready ~94% (~16 pp gap); per-day figures computed, not shown.
- **Spreads**: AAPL normal-day NBBO ~1-2 cents ~ 1-2 bps median; on 2020-03-16 the median is a multiple of that, the mean an order of magnitude above the median, thousands of bps around halts (MWCB halt 09:34-09:49; S&P 500 -12%, worst since 1987; 7% decline triggered the 15-minute halt). Seller-initiated trades "slightly dominate" that day by count and volume.
- **U-shape**: normal open/close ~2-3x midday volume; March 16 opening spike extreme.
- **Order flow vs returns (NVDA minute bars)**: OFI(t-1) correlations "tiny and noisy", decile spread "within noise"; latency erodes short-horizon edges; the executable markout shifts left by ~one spread. Round-trip cost ~1-2 bps in liquid US equities is the yardstick.
- **Liquidity spectrum**: a 1,000-share order has near-zero impact on AAPL vs 50+ bps on an illiquid stock.
- **IEX**: 350 µs speed bump; small single-digit share of US volume; ~100k-1M messages/day vs millions for ITCH.
- **Bars**: NVDA (Nov 2024) flow ~52-60% buys (spirals adaptive thresholds); TIBs produce ~800x more bars than VIBs at the same E[T]; event-driven bars show lower JB and lower AC(1) than time bars; VR(5) ~ 1 for all types (no 5-bar predictability); volume/dollar bars pull excess kurtosis toward 0 vs 1-min bars (AAPL). Daily trade-count/volume/dollar CV is large enough that a one-day threshold does not carry.
- **Jumps (5-min, K=12)**: median day attributes little variance to jumps, upper decile a large fraction; jump rate shows a close-of-day uptick; naive |z| > 4 disagrees substantially in both directions.

## Related references

- `chapters/02_financial_data_universe.md` — feed taxonomy, point-in-time and vendor data upstream of this chapter.
- `chapters/05_synthetic_data.md` — nb 17 bar parquets are flagged for GT-GAN.
- `chapters/06_strategy_definition.md` — order-flow strategy conditioning on extreme flow and spread (nb 03 bridge).
- `chapters/07_defining_the_learning_task.md` — minute-bar microstructure columns as inputs; jump-conditional labels and event-time sampling.
- `chapters/08_financial_features.md` — time-of-day, spread/imbalance/liquidity features, Kyle's lambda, Amihud, VPIN (`08_financial_features/02_microstructure_features`), regime features from the jump panel.
- `chapters/09_model_based_features.md` — evaluating microstructure signal decay and feature predictive power.
- `chapters/12_gradient_boosting.md` — ML models on microstructure features.
- `chapters/18_transaction_costs.md` — price impact and execution-cost modeling from depth, imbalance, liquidity spectrum; execution timing from the U-shape (README cites "Chapter 19", older numbering; inference).
- `chapters/25_live_trading.md` — decision-time vs exchange timestamps, latency lags.
- `case_studies/nasdaq100_microstructure.md` — the ITCH/DataBento/AlgoSeek datasets this chapter runs on.
- `libraries/ml4t_engineer.md` — `ml4t.engineer.bars` samplers (this chapter is their primary consumer).
- `libraries/ml4t_data.md` — loaders `load_nasdaq_itch`, `load_mbo_data`, `load_nasdaq100_taq`, `load_nasdaq100_bars`, `load_iex_hist`.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `companion_repo.md`, `workflow.md` — cross-cutting indexes.
- Further reading: Lee & Ready (1991); Lee & Mykland (2008); Barndorff-Nielsen & Shephard (2004); Roll (1984); Cont, Kukanov & Stoikov (2014); Kyle (1985); Glosten & Milgrom (1985); Easley et al. (2012, 2021); Holden et al. (2014; 2023 exchange-to-SIP latency); Hasbrouck & Saar (2013); O'Hara (2011, 2015); Harris (2003); Bouchaud, Bonart, Donier & Gould (2018); Gould et al. (2013); Lopez de Prado (2018) ch. 2; Zhang et al. DeepLOB (2019); ClusterLOB (2025); Aquilina et al. (2021); SEC (2020) staff report; NASDAQ ITCH 5.0 spec; DataBento MBO docs; IEX DEEP spec; `iex_parser`; martinobdl/ITCH C++ reference.

## Glossary

- **MBO / L3** — message-by-order feed with order IDs (ITCH, DataBento). **L2** — price-level depth, no IDs (IEX DEEP). **L1 / TOB** — best bid/ask only. **TAQ** — consolidated trades plus NBBO, no depth. **NBBO** — national best bid/offer, what a marketable order meets.
- **LOB** — limit order book. **Order pool / registry** — `order_ref -> (side, price, remaining)`. **stock_locate** — ITCH symbol id from R; D/X/E/C/U carry only this. **tracking_number** — ITCH tiebreaker within a timestamp. **price4** — integer price, four implied decimals.
- **Replace (U)** — retires one reference, issues another (re-pricing). **C** — execute at a price other than the displayed limit. **P / Q** — ITCH non-displayed trade / cross. **Crossed** — bid > ask (reconstruction error); **locked** — bid = ask.
- **OFI** — message-based: `(bid adds - bid removes) - (ask adds - ask removes)` per interval (nb 02/03); aggressor-based: `(buy vol - sell vol)/total` in [-1,1] (nb 09/12/13). **Depth imbalance** — `(bid depth - ask depth)/total` snapshot over N levels. **Book pressure** — side x action-weight x size x exp(-lambda x distance), EMA-smoothed.
- **Aggressor** — side that crossed the spread. **Lee-Ready** — quote test with tick-test fallback. **Tick test** — sign of last price change, zeros carry forward.
- **Spread (bps)** — `(ask - bid)/mid x 10,000`. **Bid-ask bounce** — negative AC(1) of trade-price returns. **Price improvement** — fill vs limit on C messages. **Markout** — forward return from decision time: midpoint, latency-adjusted, executable.
- **FINRA TRF** — off-exchange trade reporting (dark pools, internalisers). **Odd lot** — < 100 shares. **MWCB** — market-wide circuit breaker (7% -> 15-min halt). **LULD** — limit up/limit down. **Condition 80000002** — AlgoSeek late-reported trade. **Speed bump** — IEX 350 µs delay. **UNDEF_PRICE / F_TOB** — DataBento sentinels for clearing a side on top-of-book venues.
- **Time / tick / volume / dollar bars** — fixed interval / N trades / N shares / $ notional. **TIB / VIB** — close when |signed ticks| or |signed volume| exceeds threshold. **Run bars** — close when the longer one-sided run exceeds expected length. **E[T], P[b=1], E[theta]** — expected ticks per bar, buy probability, expected imbalance (AFML). **Threshold spiral** — feedback between adaptive threshold and bar length under one-sided flow. **et_drift** — last/first `expected_t`.
- **JB** — Jarque-Bera (lower = nearer normal). **VR(q)** — variance ratio (1 = random walk). **RV** — sum r^2. **BV** — `mu1^-2 sum |r_i||r_{i-1}|`, jump-robust. **Lee-Mykland statistic** — return over local bipower sigma, Gumbel critical value. **Jump share** — `(RV - BV)+ / RV`. **RTH** — 09:30-16:00 ET.
