# ml4t-data library (market-data acquisition, 24 providers)

> `ml4t-data` (dist `ml4t_data` 0.2.0, import `ml4t.data`) is the **data-acquisition stage** of the ML4T workflow: it pulls market, macro, factor, prediction-market, tick and synthetic data from 24 advertised providers (`advertised_provider_specs()` in `providers/registry.py`; `mock` is the 25th entry, `advertised=False`), normalizes every OHLCV response to ONE canonical Polars frame, validates it, and persists it in atomic, versioned Parquet storage with incremental updates. `ml4t-engineer` (features) and `ml4t-backtest` (simulation) consume its stacked frames and `session_date`; `ContractSpec` (multiplier, tick value, settlement) is what backtest P&L needs. Neither downstream package is required. Point-in-time (PIT) correctness is built in where the book warns of leakage: COT release schedules, FRED vintages, lagged roll selection, corporate-action rebasing. Treat it as the place where "garbage in" is stopped before features are built.

## Install and import

| Item | Value |
|---|---|
| Package | `ml4t-data` 0.2.0, MIT, Stefan Jansen; Requires-Python `>=3.12,<3.15`; Production/Stable, `Typing :: Typed` |
| Install | `pip install ml4t-data[yahoo]`; extras `[databento]` (databento>=0.38 + pyarrow>=14), `[oanda]` (oandapyv20), `[cot]` (cot-reports), `[all-providers]`, `[dev]`, `[docs]`, `[all]` |
| Hard deps | polars>=0.20, pandas>=2, numpy>=1.24, pydantic>=2.12,<3, pydantic-settings, httpx, tenacity, pybreaker, filelock>=3.19.1, pandas-market-calendars>=4.3, structlog, click, rich, pyyaml, openpyxl, platformdirs, aiofiles |
| Frame library | **Polars** (`pl.DataFrame`/`pl.LazyFrame`) is primary; `output_format="pandas"|"arrow"` conversion needs pyarrow (`_has_pyarrow()` gate); `utils.conversion.pandas_to_polars/polars_to_pandas` work without it |
| Data root | env `ML4T_DATA_PATH` (default `./data` in CWD). Legacy `QLDM_DATA_ROOT` / `QldmError` still work but emit `DeprecationWarning`; use `ML4T_DATA_PATH` / `ml4t.data.core.exceptions.ML4TDataError` |
| Key imports | `from ml4t.data import DataManager`; `from ml4t.data.providers import SyntheticProvider`; `from ml4t.data.providers.alpaca import AlpacaDataProvider`; `from ml4t.data.storage.hive import HiveStorage`; `from ml4t.data.futures import build_continuous_contract, VolumeBasedRoll, BackAdjustment`; `from ml4t.data.cot import COTFetcher, attach_cot_release_schedule, combine_cot_ohlcv, create_cot_features` |
| Optional-dep refusals | `futures.require_databento()` and module `__getattr__` give an install hint instead of a bare `ImportError` |
| Architecture | `ConfigManager → ProviderManager → FetchManager → BatchManager ← StorageManager → MetadataManager → BulkManager`; `DataManager` is a thin facade. Config precedence `defaults < YAML file < env vars < runtime params` |
| Dev loop | `uv sync --all-extras --all-groups`; `ruff check`; `ruff format --check`; `ty check`; `pytest tests/ -q`; `mkdocs build --strict`. PRs must keep offline deterministic tests; credentialed provider tests run only in integration lanes (`SyntheticProvider(seed=42)` is the offline stand-in) |

## API map by task

### Loading (facade, managers, routing)

| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `DataManager` | `(config_path=None, output_format="polars", providers=None, storage=None, enable_validation=True, progress_callback=None, **kw)` | Facade; context manager closes providers | props `config`, `router: ProviderRouter`, `providers`, `storage`, `output_format` |
| `.fetch` | `(symbol, start, end, frequency="daily", provider=None, **kw)` | One symbol, no storage | pl / pandas / arrow per `output_format` |
| `.fetch_batch` | `(symbols, start, end, frequency="daily", **kw) -> dict[str, df\|None]` | Sequential multi-symbol | |
| `.batch_load` | `(symbols, start, end, frequency="daily", provider=None, max_workers=4, fail_on_partial=False, **kw) -> pl.DataFrame` | Thread-pool parallel, stacked output | `batch_load_universe(universe, ...)`; `batch_load_from_storage(symbols, start, end, frequency="daily", asset_class="equities", provider=None, fetch_missing=True, max_workers=4)` |
| `.load` | `(symbol, start, end, frequency="daily", asset_class="equities", provider=None, bar_type="time", bar_threshold=None, exchange="UNKNOWN", calendar=None) -> str` | Initial fetch + store | returns storage key `asset_class/frequency/symbol` |
| `.import_data` | `(data, symbol, provider, frequency="daily", asset_class="equities", bar_type="time", bar_threshold=None, exchange="UNKNOWN", calendar=None) -> str` | Store an external frame | |
| `.update` | `(symbol, frequency="daily", asset_class="equities", lookback_days=7, fill_gaps=True, provider=None, initial_start=None, initial_end=None, initial_load_days=365) -> str` | Incremental refresh | re-fetches trailing 7 days, dedups keeping newest |
| `.update_all` | `(provider=None, asset_class=None, exchange=None) -> dict[str,str]` | Bulk refresh | `BulkManager.update_all(_max_workers=1)`: "use async_batch_load for parallelism" |
| `.validate_routes` | `(symbols, provider=None) -> None` | Refuse unroutable batch **before** any I/O | raises `ProviderRoutingError` |
| discovery | `list_symbols(provider, asset_class, exchange, bar_type)`, `get_metadata(symbol, asset_class="equities", frequency="daily")`, `get_metadata_for_key(key)`, `list_providers()`, `get_provider_info(name)`, `clear_cache()` | | |
| `.assign_sessions` / `.complete_sessions` | `(df, exchange=None, calendar=None, bar_frequency="auto")` / `(df, exchange=None, calendar=None, fill_gaps=True, fill_method="forward", zero_volume=True)` | Exchange-calendar sessions | delegate to `sessions/` |
| `FetchManager` | `(provider_manager, router, output_format="polars")` | `validate_dates(start, end)`, `fetch_raw(...)` (no conversion), `get_max_history_days(symbol, provider=None)` | |
| `BulkManager` | `(storage_manager, metadata_manager)` | `update_symbols(symbols, frequency="daily", asset_class="equities", ...)`, `get_stale_symbols(max_age_days=7, provider=None, asset_class=None)`, `get_update_summary()` | |
| `async_batch_load` | `(provider: AsyncOHLCVProvider, symbols, start, end, frequency="daily", max_concurrent=10, fail_on_partial=False) -> pl.DataFrame` | asyncio.gather + semaphore | `async_batch_load_dict(..., return_exceptions=False)`; `AsyncBatchManager(provider, max_concurrent=10).load/.load_dict` |
| `ProviderRouter` | `add_pattern(regex, provider)`, `get_provider(symbol, override=None)`, `setup_default_patterns()`, `clear_cache()` | Symbol -> provider | defaults: `^[A-Z]{3}_[A-Z]{3}$` -> `oanda`, `^[A-Z]+\.(v\|V)\.[0-9]+$` -> `databento`; bare tickers / compact or delimited crypto pairs are never inferred |
| `ProviderManager` | `(config)`; `get_provider(name, *, required_capability=None)`, `available_providers()`, `is_available(name)`, `register_provider(name, cls)`, `close_all()` | Lazy, cached provider instances | refuses a provider lacking the capability |
| `ConfigManager` | `(config_path=None, output_format="polars", providers=None, **kw)` | props `default_frequency`, `timezone`; `get_provider_config(name)`, `get_routing_patterns()`, `has_api_key(name)`, `apply_overrides(...)` | |

### Providers (registry `ml4t.data.providers.registry`; 24 advertised + `mock`)

| Name (class) | Capabilities | Credentials / config | Notes |
|---|---|---|---|
| `yahoo` (`YahooFinanceProvider(enable_progress=False, rate_limit=None)`) | ohlcv | none; extra `[yahoo]` | `fetch_batch_ohlcv(symbols, start, end, frequency="daily", chunk_size=50, delay_seconds=1.0)`; `fetch_financials(symbol, statement="income", period="annual", currency=None)`; `fetch_company_metrics(symbol, *, metrics=None)` |
| `alpaca` (`AlpacaDataProvider(api_key=None, api_secret=None, feed="iex", adjustment="raw", rate_limit=None)`) | ohlcv, intraday | `ALPACA_API_KEY`/`ALPACA_API_SECRET` (fallback `APCA_API_KEY_ID`/`APCA_API_SECRET_KEY`); headers not query strings | stocks plain ticker, crypto `BTC/USD`; freq daily/hourly/minute (1/5/15/30); dates `YYYY-MM-DD` or RFC-3339, inclusive; 200 calls/min; feeds `iex` (~2-3% of US volume), `sip` (100%; free plan lacks last 15 min), `otc`, `boats`; adjustment `raw|split|dividend|all`; `fetch_ohlcv(..., asset_class=None)` validated |
| `tiingo` (`TiingoProvider`) | daily ohlcv raw + adjusted | `TIINGO_API_KEY` | |
| `finnhub` (`FinnhubProvider`) | ohlcv, `fetch_quote(symbol)`, financials, `fetch_company_metrics(symbol, *, metric_group="all")` | `FINNHUB_API_KEY` | |
| `eodhd` (`EODHDProvider(api_key=None, exchange="US")`) | ohlcv, `fetch_fundamentals` (raw dict), financials, metrics | `EODHD_API_KEY` | symbols `SYMBOL.EXCHANGE` |
| `massive` (`MassiveProvider` in `polygon.py`) | ohlcv, `fetch_financials(..., limit=100)`, metrics | `MASSIVE_API_KEY` or legacy `POLYGON_API_KEY` | `asset_class: MassiveAssetClass` or inferred from prefix; follows `next_url` pagination |
| `twelve_data` (`TwelveDataProvider`) | ohlcv stocks / ETF / index / FX / crypto | `TWELVE_DATA_API_KEY` | |
| `databento` (`DataBentoProvider(api_key=None, dataset="GLBX.MDP3", adjust_session_dates=False, session_start_hour_utc=0)`) | ohlcv-1m/1h/1d, futures, options | `DATABENTO_API_KEY`; extra `[databento]` | `fetch_continuous_futures(root, start, end, frequency="daily", version=0)` (`ROOT.v.0`); `estimate_opra_cost(symbols, start, end, schema="ohlcv-1d", stype_in="raw_symbol")`; `fetch_option_chain(underlying, session_date, *, expiry=None, right="both", min_strike, max_strike)`; `fetch_option_ohlcv(contract, ..., consolidate_publishers=True)`; `fetch_option_quotes(..., schema="cbbo-1m", consolidated_only=True)`; `fetch_multiple_schemas`; `get_available_datasets/schemas` |
| `oanda` (`OandaProvider(api_key=None, practice=True)`) | FX ohlcv M1/M5/H1/D | `OANDA_API_KEY`; extra `[oanda]` | symbols `EUR_USD` |
| `binance` (`BinanceProvider(market="spot"\|"futures", timeout=30.0)`) | crypto ohlcv | none | `BTCUSDT` |
| `binance_public` (`BinancePublicProvider(market="spot", timeout=60.0)`) | bulk ZIP ohlcv; `fetch_metrics(symbol, start, end)` (OI, long/short ratios); `fetch_premium_index(symbol, start, end, interval="8h")` | none; S3 `https://data.binance.vision`, no geo restriction | `fetch_ohlcv_multi_async(..., max_concurrent=5)` / `fetch_ohlcv_multi_parallel`; `get_available_symbols(search=None)`; daily vs monthly archives, first-date probe + binary search |
| `okx` (`OKXProvider(market="swap"\|"spot", timeout=30.0)`) | ohlcv; `fetch_funding_rates(symbol, start, end)`; `fetch_current_funding_rate(symbol)` | none | `BTC-USDT-SWAP` |
| `coingecko` (`CoinGeckoProvider(api_key=None, use_pro=False)`) | daily ohlcv only | optional `COINGECKO_API_KEY` | `symbol_to_id`, `get_coin_list`, `get_price(ids, vs_currencies)`; bounded daily-history window |
| `cryptocompare` (`CryptoCompareProvider(api_key=None, exchange="CCCAGG", timeout=30.0)`) | ohlcv | `CRYPTOCOMPARE_API_KEY` | CCCAGG = cross-exchange aggregate index |
| `fred` (`FREDProvider`) | ohlcv, series | `FRED_API_KEY` | `fetch_ohlcv(series, start, end, frequency="daily", vintage_date=None)`; `fetch_series_metadata(id)`; `fetch_multiple(ids, start, end, frequency="daily", forward_fill=True)`; client 100 req/min (server 120); e.g. VIXCLS, UNRATE, GDP, DGS10, CPIAUCSL, SP500 |
| `fxmacrodata` (`FXMacroDataProvider`) | macro; `manager_compatible=False` | optional `FXMACRODATA_API_KEY`/`FXMD_API_KEY` | `fetch_announcements(currency, indicator, *, start_date, end_date, limit=500, revisions, include_metadata=False)`, `fetch_predictions`, `fetch_calendar`, `fetch_catalogue`, `fetch_forex(base, quote)`, `fetch_cot`, `fetch_commodities`, `fetch_latest_commodities`, `fetch_market_sessions`, `fetch_risk_sentiment`, `fetch_news`, `fetch_press_releases`, `fetch_endpoint(endpoint, *, currency="usd", indicator="policy_rate", quote="usd", commodity="gold")`; `health()`; returns `PayloadResult` when `include_metadata=True` |
| `aqr` (`AQRFactorProvider(data_path=None)`) | factors; not manager-compatible | local Excel via `download(output_path=None, datasets=None, include_optional=False)` | 16 datasets: QMJ, BAB, HML Devil (US + 23 intl, monthly/daily), 6 size x quality portfolios, 10 quality deciles, VME (8 asset classes), TSMOM (58 futures), momentum indices, 100+yr long-history, ESG frontier, credit premium, commodities; `fetch(dataset, region=None, start, end)`, `fetch_factor(..., region="USA")`, `fetch_factors(dataset, start, end)`, `list_datasets(category)`, `get_dataset_info` |
| `fama_french` (`FamaFrenchProvider(cache_path=None, use_cache=True)`) | factors; not manager-compatible | none | 70+ datasets: FF3 (Mkt, SMB, HML), FF5 (+RMW, CMA), UMD, ST/LT reversal, 6/25/100 sorted, 32 trivariate, industries 5/10/12/17/30/38/48/49, international; `fetch(dataset, frequency="monthly", start, end)`, `fetch_combined(datasets)`, `fetch_factors(dataset="ff3")`, `clear_cache()` |
| `kalshi` (`KalshiProvider`) | ohlcv, events | none for public data; ~10 req/s | `fetch_ohlcv(ticker, start, end, frequency="daily", series_ticker=None)`, `list_markets(status="open", series_ticker=None, limit=100)`, `list_series()`, `get_market_metadata`, `fetch_multiple_markets(tickers, ..., align=True)`; Series (KXINFL, KXFED, KXGDP, KXUNEMPLOY, KXSPX, KXBTC) -> Event `KXINFL-25JAN` -> Market; prices are probabilities 0-1 |
| `polymarket` (`PolymarketProvider`) | ohlcv, events | none; CLOB ~60/min, Gamma ~30/min | `resolve_symbol(slug\|0x condition\|token id, outcome="yes")`, `fetch_ohlcv(..., outcome="yes")`, `fetch_both_outcomes`, `list_markets(active=True, closed=None, category=None, limit=100, offset=0)`, `search_markets(query, limit=20)`, `get_token_prices -> {yes, no}`; OHLC synthesized from price history (`interval="1d"`) |
| `nasdaq_itch` (`ITCHSampleProvider(download_path=None, parsed_path=None)`) | tick; not manager-compatible | none; `https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/` | `list_available_files()`, `download(date_or_filename, output_path=None, verify_size=True, progress_callback)`, `get_local_file`, `load_parsed_messages(message_type, parsed_dir=None, symbol=None)`, `list_parsed_message_types`, `get_dataset_info` |
| `wiki_prices` (`WikiPricesProvider(parquet_path=None, cache_in_memory=False)`) | ohlcv, frozen Quandl WIKI 1962-2018 | config `parquet_path` required | `list_available_symbols`, `get_date_range(symbol)`, `get_dataset_stats`, `download(output_path, api_key, env_file)` |
| `synthetic` (`SyntheticProvider(model="gbm", annual_return=0.08, annual_volatility=0.20, seed=None, base_price=100.0, base_volume=1_000_000, heston_kappa=2.0, heston_theta=0.04, heston_xi=0.3, heston_rho=-0.7, garch_alpha=0.1, garch_beta=0.85, calendar_mode="equity")`) | ohlcv | none | models `gbm\|gbm_jump\|mean_revert\|heston\|garch`; `CalendarMode = "equity"\|"continuous"`; `reset_seed(seed)` |
| `learned_synthetic` (`LearnedSyntheticProvider(samples, metadata=None, seed=None, calendar_mode="equity")`) | synthetic_artifact; not manager-compatible | config `samples` required | `from_samples(path, metadata_path=None, seed=None)`, `from_checkpoint(path, device="cpu")`; `get_samples(n_samples=None, shuffle=True)`, `generator_name`, `n_samples`, `seq_length`, `n_features`; loads TimeGAN / Sig-CWGAN samples from "Chapter 6 notebooks" (docstring) |
| `mock` (`MockProvider(seed=42, pattern=DataPattern.RANDOM, base_price=100.0, volatility=0.01, trend_rate=0.0, gap_probability=0.0, range_bound=0.05)`) | ohlcv; `advertised=False` | none | `validate_data(df) -> dict` |

Provider plumbing: `get_provider_spec(name)` (case-insensitive; `ValueError` listing all names on miss), `advertised_provider_specs()`, `PROVIDER_REGISTRY`; `ProviderSpec(name, module, class_name, description, capabilities: frozenset, credentials, optional_credential_environment, required_configuration, manager_compatible=True, advertised=True, deprecated=False)` with `access_label` (`Yes|Configuration|Optional|No`), `is_configured(config, environ)`, `has_api_key`, `load_class()`; `CredentialRequirement(config_field, environment).is_satisfied`. `BaseProvider(rate_limit: (calls, seconds)|None, session_config=None, circuit_breaker_config=None)` = RateLimit + CircuitBreaker + Validation + Session mixins; template `fetch_ohlcv(symbol, start, end, frequency="daily")` = validate inputs -> rate limit -> circuit breaker -> fetch -> `_validate_ohlcv`; subclasses implement `name` plus `_fetch_and_transform_data` or `_fetch_raw_data` + `_transform_data`; `capabilities() -> ProviderCapabilities(supports_intraday, supports_crypto, supports_forex, supports_futures, requires_api_key, max_history_days, rate_limit=(60, 60.0))`; `fetch_ohlcv_async`, `close()`. `AsyncBaseProvider(rate_limit=None, max_concurrent=None)`: `fetch_ohlcv_async`, `batch_fetch_async(symbols, start, end, frequency="daily", return_exceptions=False) -> dict[str, df|Exception]`. Protocols: `OHLCVProvider`, `FactorProvider` (`fetch(dataset, frequency="monthly")`, `list_datasets`), `EventProvider` (`fetch_events(market, start, end)`, `list_markets`), `AsyncOHLCVProvider`. `providers/fundamentals.py`: `normalize_statement_type`, `normalize_period_type`, `wide_pandas_statement_to_financials(frame, *, symbol, provider, statement_type, period_type, currency, source)`, `rows_to_financials_frame`, `rows_to_company_metrics_frame`, `numeric_mapping_to_metric_rows`, `nested_mapping_to_metric_rows`, `empty_financials_frame`, `empty_company_metrics_frame`.

### Storage

| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `StorageConfig` | `(base_path=resolve_data_root(), strategy="hive", compression="zstd", partition_granularity="month", lock_timeout=30.0, metadata_tracking=True)` | Frozen Pydantic, `extra="forbid"` | `StorageStrategy` {`hive`, `flat`}; `CompressionType` {zstd, lz4, snappy, gzip} or `None` -> `uncompressed`; `PartitionGranularity` year/month/day/hour -> `partition_cols` `["year"]`..`["year","month","day","hour"]` |
| `create_storage` | `(base_path: StorageConfig\|str\|Path, strategy=None, **kw) -> StorageBackend` | Factory | Hive default |
| `StorageBackend` (ABC) | `write(data, key, metadata=None, *, preserve_metadata=False) -> Path`; `read(key, start_date=None, end_date=None, columns=None) -> pl.LazyFrame`; `list_keys()`, `exists`, `delete`, `get_metadata(key)` | Atomic generation/commit writes | `GENERATION_RETENTION = 3`; `normalize_storage_metadata(metadata, key)` |
| `HiveStorage(config)` | ABC + `partitions(key, start_date, end_date) -> list[Partition]` (`label` `"2023-01"`); `IncrementalStorageBackend`: `get_latest_timestamp(symbol, provider)`, `save_chunk(data, symbol, provider, start_time, end_time)`, `update_combined_file(data, symbol, provider) -> int`, `get_combined_file_path`, `read_data(symbol, provider, start_time, end_time)`, `update_metadata(...)` | Default backend; partition pruning (docstring claims "7x", not reproduced) | `year=YYYY/month=MM/` |
| `FlatStorage(config)` | same ABC | One Parquet per key | small datasets |
| `ChunkedStorage` | `(base_path, chunk_size_days=30, compression=CompressionType.SNAPPY)`; `write(DataObject) -> key`; `read(key, start_date, end_date) -> DataObject`; `get_chunk_info(key) -> list[ChunkInfo]`; `list_keys(prefix="")` | Date-chunked for large sets; NOT a `StorageStrategy` | chunk id from (symbol, frequency, start_date) |
| keys / security | `validate_storage_key`, `encode_storage_key`, `decode_storage_key`, `storage_key_path(root, key, suffix="")`, `contained_path(root, *parts)`, `validate_path_component`, `encode_path_component`; `PathValidator.parse_storage_key(key) -> (asset_class, frequency, symbol)`, `.sanitize_path`, `.validate_file_path(path, allowed_extensions=None)`, `.is_safe_filename` | Traversal defence | `MAX_KEY_BYTES = 160`; `PathTraversalError` |
| `MetadataTracker(base_path)` | `get_metadata(key)`, `update_metadata(key, update_record, total_rows, start, end)`, `add_update_record`, `get_update_history(key, limit=10)`, `check_health(key, stale_days=7) -> (status, message)`, `get_summary()` | JSON metadata + history | `UpdateRecord`, `DatasetMetadata` (`to_dict/from_dict`) |
| profiling | `generate_profile(df, source="", timestamp_col="timestamp", symbol_col="symbol") -> DatasetProfile`; `save_profile`, `load_profile`, `get_profile_path(data_path)`; `ProfileMixin` (`generate_profile()`, `load_profile()`) | Column stats stored beside data | `DatasetProfile.to_dataframe()/summary()` |
| `MigrationManager(data_root)` / `BackupManager(data_root)` | `register_migration`, `migrate(target_version=None, dry_run=False)`, `rollback(v)`; `create_backup(backup_name=None, compression=True)`, `restore_backup(path, target_dir=None, overwrite=False)`, `list_backups()` | Schema versions, archives | `create_standard_migrations()` (`add_volume_column`, `rename_to_standard`); `MigrationStatus` pending/in_progress/completed/failed/rolled_back |
| legacy | `find_legacy_storage_entries(base_path, strategy)`, `migrate_legacy_storage(base_path, strategy, key_mapping, *, storage_config=None, chunk_size_days=30)` | Verified pre-0.1 upgrade | `LegacyStorageMigrationError` |
| `AsyncStorageBackend` | `write(DataObject) -> str`, `read`, `exists`, `delete`, `list_keys(prefix="")`, `update(key, data)` | async ABC | |

### Incremental updates and gaps

| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `UpdateStrategy` | `INCREMENTAL \| APPEND_ONLY \| FULL_REFRESH \| BACKFILL` | | |
| `IncrementalUpdater(strategy=INCREMENTAL)` | `determine_update_range(storage, key, req_start, req_end) -> (start, end, "full"\|"none"\|"incremental")`; `update_incremental(storage, tracker, key, new_data, provider, strategy=None) -> UpdateResult`; `apply_strategy(storage, key, new_data, strategy)` | Gap-aware range + merge | UTC-aware bounds via `_ensure_datetime` |
| `update_manager.GapDetector(exclude_weekends=False)` | `detect_gaps(df, frequency="daily", tolerance_days=0) -> list[dict]`; `detect_gaps_in_storage(storage, key, start, end, frequency="daily")` | Expected-delta gaps | |
| `utils.gaps.GapDetector(tolerance=0.1)` | `detect_gaps(df, frequency="daily", timestamp_col="timestamp", is_crypto=False) -> list[DataGap]`; `summarize_gaps(gaps)`; `fill_gaps(df, gaps, method="forward")` | `DataGap.is_significant()` = >1 period | same name, different class |
| `BackfillManager(storage, tracker)` | `identify_candidates(min_gap_days=5, max_age_days=30)`, `prioritize_tasks`, `execute_backfill(symbol, gaps, provider) -> UpdateResult` | Fill material recent gaps | uses `BACKFILL` strategy |
| `GapOptimizer(max_batch_days=30)` | `consolidate_gaps(gaps, merge_threshold_days=7)`, `estimate_api_calls(gaps, merge_threshold_days=7)`, `should_consolidate(gaps, threshold_gaps=3)` | Fewer metered calls | |
| `ProviderUpdater(provider_name, storage: IncrementalStorageBackend, safety_margin_minutes=5)` | `update_symbol(symbol, start_time=None, end_time=None, incremental=True, dry_run=False) -> dict`; `update_symbols(symbols, max_workers=None)`; hooks `_fetch_data`, `_transform_data`, `_get_default_start_time` | Template-method updater | |

### Validation, anomaly detection, sessions, assets

| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `OHLCVValidator` | `(check_nulls=True, check_price_consistency=True, negative_price_policy="forbid"\|"warn"\|"allow", check_negative_volume=True, check_duplicate_timestamps=True, check_chronological_order=True, check_price_staleness=True, check_extreme_returns=True, max_return_threshold=0.5, staleness_threshold=5)` | Single-frame rules | `.validate(df) -> ValidationResult` |
| `CrossValidator` | `(check_price_continuity=True, check_volume_spikes=True, check_weekend_trading=True, check_market_hours=True, volume_spike_threshold=10.0, price_gap_threshold=0.1, is_crypto=False)` | Continuity, spikes, calendar; `validate(df, reference_df=None)` | market hours 9:30-16:00 ET |
| `ValidationRulePresets` | `equity_rules()`, `crypto_rules()`, `forex_rules()`, `commodity_rules()`, `futures_rules()`, `strict_rules()`, `relaxed_rules()`, `for_asset_class(name)` | Per-asset thresholds | `ValidationRuleSet(name, description="", rules={}, default_rule=None).get_rules(symbol)/add_rule(pattern, rule)/save/load`; `create_rule_set_from_config(path)` |
| `ValidationReport` | `add_result(result, validator_name)`, `passed`, `total_issues`, `critical/error/warning_count`, `to_json(indent=2)`, `save/load`, `print_summary()`, `to_dataframe()` | Aggregate | `Severity` info < warning < error < critical |
| `AnomalyManager(config=None, custom_detectors=None)` | `analyze(df, symbol, asset_class=None) -> AnomalyReport`; `analyze_batch(datasets, asset_classes=None)`; `save_report(report, output_dir)`; `filter_by_severity(report, min_severity)`; `get_statistics(report)` | Pre-feature screening | `ReturnOutlierDetector`, `VolumeSpikeDetector`, `PriceStalenessDetector`; `AnomalyReport.has_critical_issues()/get_by_type/to_dataframe`; `AnomalyType` {return_outlier, volume_spike, price_stale, data_gap, price_spike, zero_volume}; `AnomalySeverity` {info, warning, error, critical} |
| `AssetSchema` / `AssetValidator(asset_info)` | `validate_dataframe(df, asset_class) -> (bool, issues)`, `get_required_columns`, `normalize_dataframe`; `.validate(df, frequency="daily") -> dict[rule, msgs]` | Asset-class schemas and rule sets | `AssetClass` {equity, crypto, forex, commodity, fixed_income, economic, index, etf, option, future} + aliases equities/futures/options; `AssetInfo.is_derivative/is_crypto_pair/requires_24_7_calendar` |
| `SessionAssigner(calendar_name)` / `.from_exchange(exchange)` | `assign_sessions(df, start_date=None, end_date=None, outside_session="null"\|"raise"\|"drop", bar_frequency="auto"\|"daily"\|"intraday") -> df + session_date` | Exchange-local trading day | pandas_market_calendars names (e.g. `XNYS`, `CME_Globex_Crypto`) |
| `SessionCompleter(calendar_name)` | `complete_sessions(df, start_date=None, end_date=None, fill_method="forward", zero_volume=True)`; `get_session_info(dt)` | Insert missing sessions | |
| `CryptoCalendar(exchange=None)` | `is_open`, `is_maintenance`, `get_sessions`, `get_expected_sessions_count(start, end, frequency="daily")`, `find_gaps(df, frequency="daily", threshold_multiplier=1.5)` | 24/7 calendar | `CryptoSessionValidator(calendar).validate_continuity(df, frequency)/validate_volume_profile(df)` |
| asset helpers | `MarketHours.get_schedule(asset_class, exchange=None)`; `parse_crypto_symbol(sym) -> (base, quote)`; `normalize_crypto_symbol(sym, separator="/")`; `get_asset_class_from_symbol(sym)`; `assets.ContractSpec` (`tick_value`), `get_contract_spec`, `load_contract_specs(symbols=None, include_all=False)`, `register_contract_spec` | | |

### Corporate actions, COT, futures

| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `apply_corporate_actions` | `(prices, split_col="split_ratio", dividend_col="ex-dividend", price_cols=None, volume_col="volume") -> pl.DataFrame` | Splits + dividends in one pass | adds `adj_close`, `price_adjustment_factor`; `apply_splits(...)`, `apply_dividends(..., close_col="adj_close")` |
| `COTFetcher(config: COTConfig\|None)` | `fetch_product(code, start_year=None, end_year=None, release_schedule=None)`, `fetch_all`, `save_to_hive(df, code)`, `download_all(skip_existing=True)`, `list_available_products`, `get_product_info(code) -> ProductMapping` | CFTC via `cot_reports` | adds net positioning by trader type, WoW changes, z-scores, commercial-vs-speculative divergence; `COTConfig.get_years()`, `load_cot_config(path)` |
| `attach_cot_release_schedule` | `(cot, schedule, cot_date_col="report_date", schedule_date_col="report_date", available_at_col="available_at")` | Add tz-aware `available_at` | REQUIRED before any join |
| `combine_cot_ohlcv` | `(ohlcv, cot, date_col="timestamp", cot_date_col="report_date", available_at_col="available_at", observation_timezone="UTC")` | PIT as-of join: latest report with `available_at <= timestamp` | `combine_cot_ohlcv_pit` = alias; `create_cot_features(df, prefix="cot_", report_date_col=None, open_interest_col=None)`; `load_combined_futures_data(product, ohlcv_path, cot_path, start_date, end_date, release_schedule)` |
| `FuturesDownloader(FuturesDownloadConfig(products, start, end, ...))` | `estimate_cost()`, `download_product(p, skip_existing=True)`, `download_all(skip_existing=True, continue_on_error=True) -> DownloadProgress`, `download_all_parallel(max_workers=4, skip_existing=True)`, `download_test(products=None)`, `list_downloaded`, `get_latest_date(product=None)`, `update(end_date=None)` | Databento parent symbology `{PRODUCT}.FUT`; OHLCV daily + definitions + statistics | `FuturesCategory`; `load_yaml_config(path)`, `load_definitions_config(path)` |
| `ContinuousDownloader(ContinuousDownloadConfig(products, start, end, tenors=[0,1,2], schema="ohlcv-1h"))` | `download_product(p, skip_existing=True) -> (int,int)`, `download_all(skip_existing=True, delay_between_products=1.0)`, `download_all_parallel(max_workers=4)`, `estimate_cost`, `list_status` | `.v.N` (volume) / `.c.N` (calendar) continuous; ONE API call per product, local Hive by product/year | `load_continuous_config(path)` |
| `IndividualDownloader(IndividualDownloadConfig(products={"ES": {"months":[3,6,9,12]}, "CL": {"months": list(range(1,13))}}, years=[2024, 2025]))` | `download_product(p, skip_existing=True, force=False) -> (int, str)`, `download_all`, `estimate_cost`, `list_symbols`, `get_status` | Specific contracts (`ESH24`), `ohlcv-1h`, one file per product | `IndividualProductConfig.is_quarterly()/is_monthly()` |
| `DefinitionsDownloader(products, storage_path, config=None, dataset="GLBX.MDP3", api_key=None)` | `download_snapshot(date, product, max_retries=3)`, `download_snapshots(products=None, dates=None)`, `get_merged_definitions()`, `check_coverage()` | Expiry, tick size, multiplier | yearly snapshots dedup'd |
| parsers (`futures/databento_parser.py`) | `parse_contract_symbol(symbol, *, reference_year=None) -> ContractInfo`; `load_databento_definitions/ohlcv/statistics(stat_types=None)/open_interest(product, storage_path=None)`; `get_expiration_dates(product) -> dict[str, date]`; `parse_databento_raw(product, storage_path=None, include_open_interest=True)`; `parse_databento`; `get_contract_chain(product)`; `get_front_back_contracts(product, as_of_date)`; legacy `parser.parse_quandl_chris[_raw](ticker, data_path, contract_spec)` | | `ESH25` -> ES, `contract_month` `"2025-03"` |
| roll strategies (`futures/roll.py`) | `VolumeBasedRoll(lookback_days=1, min_days_between_rolls=20)`, `OpenInterestBasedRoll(lookback_days=1)`, `TimeBasedRoll(days_before_expiration=5, use_business_days=True)`, `FirstNoticeDateRoll(days_before_first_notice=1)`, `CalendarRoll(rank=0)`, `HighestVolumeRoll(rank=0, min_volume=0)`, `HighestOpenInterestRoll(rank=0, min_oi=0)` | `select_contracts(data, contract_spec=None) -> [date, symbol]`, `identify_rolls -> list[date]`, `identify_roll_events -> list[RollEvent]`; `build_roll_events(data, selections)` | rank-based strategies use the PREVIOUS observation |
| adjustments (`futures/adjustment.py`) | `BackAdjustment()`, `RatioAdjustment()`, `NoAdjustment()` `.adjust(data, roll_events: Sequence[RollEvent])` | Stitch continuous series | |
| `build_continuous_contract` | `(ticker, roll_strategy=None, adjustment_method=None, contract_spec=None, data_source="quandl_chris"\|"databento", storage_path=None)` | End-to-end continuous series | `ContinuousContractBuilder(contract_spec, roll_strategy, adjustment_method).build(ticker, data_source, ...)/build_multiple(tickers, ...)`; prefer `data_source="databento"` (recommended); `quandl_chris` is legacy |
| `futures.schema.ContractSpec` | `is_cash_settled`, `is_physical_settled`, `convert_price(price, from_unit, to_unit="dollars")`, `calculate_contract_value(price, *, price_unit=None)`, `calculate_tick_pnl(ticks)` | Specs for backtest P&L | `FuturesAssetClass`, `SettlementType`, `ExchangeInfo`, `price_conversion_factor(from_unit, to_unit)`, `get_contract_spec(ticker)` |

### Book-reader managers, synthetic, universes, export, config, CLI

| Function / class | Signature | Purpose | Notes |
|---|---|---|---|
| `etfs.ETFDataManager.from_config("configs/ml4t_etfs.yaml")` | `download_all(force=False) -> dict[str,int]`, `update()`, `load_ohlcv(symbol)`, `load_symbols(list)`, `load_all()`, `load_category(cat)`, `get_available_symbols()`, `get_data_summary()` | 50 diversified ETFs via Yahoo; Hive by ticker + combined file + metadata | `ETFConfig.get_all_symbols()/get_categories()`; `ProfileMixin` |
| `crypto.CryptoDataManager.from_config(...)` | `download_premium_index(symbols=None)`, `download_perps(symbols=None)` (parallel), `download_all`, `load_premium_index`, `load_perps`, `load_symbol`, `get_available_symbols`, `get_data_summary` | Binance Public S3, no key; built for the funding-rate arbitrage case study | `_normalize_loaded_timestamps` fixes beta parquet units |
| `macro.MacroDataManager.from_config(...)` | `download_treasury_yields()` (FRED, yfinance fallback), `load_treasury_yields()`, `get_yield_curve_slope()`, `get_regime(threshold=0.5)` | Derived `YIELD_CURVE_SLOPE` regime | `MacroConfig.get_treasury_symbols()/get_derived_series()` |
| `futures.book_downloader.FuturesDataManager.from_config("configs/ml4t_futures.yaml")` | `download_product_ohlcv(product, start_date, end_date)`, `download_product_definitions`, `download_all(include_definitions=True, parallel=1)`, `update(end_date=None)`, `load_ohlcv(product, start, end)`, `load_definitions`, `list_products`, `get_data_summary`, `generate_profile/load_profile/generate_all_profiles` | 70 CME products, yearly Hive partitions | module funcs `download_futures_data(config_path)`, `update_futures_data` |
| `SyntheticRegistry(data_dir=None, checkpoints_subdir="synthetic/checkpoints")` | `list_generators()` (`timegan`, `tailgan`, `sigcwgan`, ...), `list_experiments(gen)`, `list_all`, `get_metadata(gen, experiment=None)` (has `["generator"]["paper"]`), `load_samples(gen, experiment=None, n_samples=None, shuffle=False) -> np.ndarray`, `get_provider(gen, experiment=None, seed=None, calendar_mode="equity") -> LearnedSyntheticProvider`, `summary()` | Trained generator checkpoints | `synthetic.ohlcv_utils`: `create_rng(seed)`, `derive_symbol_seed(seed, symbol)`, `validate_synthetic_frequency`, `returns_to_prices(returns, base_price=100.0, log_returns=True)`, `generate_ohlc_from_close(closes, volatility, rng)`, `generate_volume(returns, base_volume=1_000_000)`, `generate_timestamps(start, end, frequency, calendar_mode)`, `get_bars_per_day`, `get_periods_per_year` |
| `Universe` | `get(name)` (case-insensitive: S&P 500, NASDAQ 100, top-100 crypto, major forex), `list_universes()`, `add_custom(name, symbols)`, `remove_custom(name)` | Packaged JSON lists | |
| `ExportManager(storage: ExportStorage, progress_callback=None)` | `export(key, output_path, format_type="csv", **opts) -> ExportResult`, `export_batch(keys, path, format_type="excel")`, `export_pattern(pattern, path, format_type="excel")`, `list_formats()` | CSV / Excel / JSON | `CSVExporter`, `ExcelExporter` (xlsxwriter or polars, metadata sheet, keeps tz offsets), `JSONExporter` (ISO datetimes); `BaseExporter(config: ExportConfig)` |
| format / conversion | `pivot_to_wide(df, value_cols=None, symbol_col="symbol", timestamp_col="timestamp")`, `pivot_to_stacked(df, timestamp_col="timestamp")`; `pandas_to_polars(df)`, `polars_to_pandas(df)` | Stacked <-> wide; no PyArrow | wide columns `close_AAPL` |
| `ConfigLoader(config_path=None)` / `load_config(path=None)` | `load(override_path=None) -> DataConfig`, `save(config, path=None)`; `_find_config_file()`, env interpolation, `_process_includes`, relative paths resolved vs declaring file | YAML config | `DataConfig` sections providers / universes / datasets / workflows / schedules (+ `storage`, `routing` inferred); `from_yaml`, `to_yaml`, `to_runtime_dict()`, `get_provider/universe/dataset/workflow(name)`, `validate_config() -> list[str]`, `validate_provider_credentials()`; models `ProviderConfig`, `RateLimitConfig`, `ProviderType`, `ScheduleType`, `SymbolUniverse.load_from_file()`, `DatasetConfig`, `ScheduleConfig`, `WorkflowConfig`; `ConfigValidator(config).validate() -> bool`, `.get_summary()`; `write_yaml` atomic, owner-only perms |
| `core.config` | `resolve_data_root(data_root=None)`, `resolve_data_path(*parts, data_root=None)`, `resolve_storage_path(path, *default_parts)`, `expand_path`; `Config` (`base_dir` alias `data_root`; reads env), `RetryConfig`, `CacheConfig` | Paths | |
| resilience utils | `RateLimiter(max_calls, period, burst=None).acquire(blocking=True)/reset()`; `AsyncRateLimiter` (+`remaining_calls`, `time_until_reset`); `AsyncAdaptiveRateLimiter(..., backoff_factor=2.0).handle_rate_limit_response(retry_after, remaining)/restore_rate()`; `MultiProviderRateLimiter.add_provider(name, max_calls, period, burst=None, adaptive=False)`; `GlobalRateLimitManager().get_rate_limiter(name, max_calls, period)`; `FileLock(timeout=30.0, retry_delay=0.1).acquire(path, exclusive=True, blocking=True)`, `file_lock(path, ...)`; `with_retry(max_attempts=3, min_wait=1.0, max_wait=60.0, multiplier=2.0)` | Throttle, lock, retry | mixins `RateLimitMixin`, `RetryMixin`, `CircuitBreakerMixin` (`CircuitBreaker(failure_threshold=5, reset_timeout=300.0, expected_exception=NetworkError, excluded_exceptions=(RateLimitError,))`, `metrics()`, `reset()`), `SessionMixin` (`_get/_post/_get_json`), `AsyncSessionMixin`, `ValidationMixin` |
| CLI `ml4t-data` | `fetch SYMBOL --start --end --frequency --provider --output [--symbols-file --config --dataset --progress]`; `update`; `validate [--all --anomalies --severity --save-report]`; `status [--detailed] [--stale-days 7]`; `export` (CSV/JSON/Parquet); `info`; `list`; `update-all --config --dataset --dry-run`; `download-futures --config --dry-run --force --product --parallel`; `update-futures --config --end-date --dry-run --yes`; `download-cot --config --products --start-year --end-year --output --list-products --dry-run --force`; `version`, `providers`, `config`, `completion SHELL` | Click group `cli(ctx, verbose, quiet)` | option spellings inferred from parameter names except `status --stale-days` (default 7, confirmed); health in {`healthy`, `stale`, `error`}; `cli/utils.load_symbols_from_file`, `save_dataframe`, `save_batch_results` |

### Exceptions (`ml4t.data.core.exceptions`)
`ML4TDataError(message, details=None)` -> `ProviderError(provider, message, details)` -> `NetworkError(..., retry_after=None, retryable=True)` -> `RateLimitError(provider, retry_after=None, remaining=None, limit=None)`; `AuthenticationError(provider)`; `DataValidationError(provider, message, field=None, value=None)`; `SymbolNotFoundError(provider, symbol)`; `DataNotAvailableError(provider, symbol, start, end, frequency)`; `StorageError(message, key=None)` -> `LockError(key, timeout)`; `ConfigurationError(message, parameter=None)` -> `ProviderRoutingError`; `CircuitBreakerOpenError(message, failure_count=None)`; plus `PathTraversalError`, `LegacyStorageMigrationError`, `utils.retry.RetryableError`.

## Data contracts

| Contract | Specification |
|---|---|
| Canonical OHLCV frame (`ValidationMixin.CANONICAL_SCHEMA`, `core/schemas.MultiAssetSchema`) | Stacked/long, fixed column order: `timestamp: Datetime("us","UTC")`, `symbol: String`, `open, high, low, close, volume: Float64` (volume Float64 even for share counts). Extra provider columns kept AFTER the seven canonical ones. Sorted by `[timestamp, symbol]`. Optional per asset class: equities `dividends, splits, adjusted_close`; crypto `trades_count, taker_buy_volume, taker_buy_quote_volume`; futures `open_interest`. Helpers: `MultiAssetSchema.validate(df, strict=True)`, `create_empty(asset_class)`, `add_symbol_column`, `standardize_order`, `cast_to_schema`; `timestamp_bounds(df)`; `align_frames_for_concat(left, right, left_fill_values, right_fill_values)`; `_concat_dtype` never reduces timestamp precision |
| Timestamps | Must be tz-aware Polars Datetime; naive -> `DataValidationError(field="timestamp")` unless frame is empty; non-UTC converted with `dt.convert_time_zone("UTC")`; microsecond precision. Daily exchange bars are UTC-stamped; `session_date` (from `assign_sessions`) carries the exchange-local trading day |
| Symbols | `_expected_ohlcv_symbol` = `symbol.strip().upper()`; every row upper-cased; Alpaca crypto keeps `BTC/USD`; Binance/OKX override to `BTCUSDT` / `BTC-USDT-SWAP`; OANDA `EUR_USD`; EODHD `SYMBOL.EXCHANGE`; Databento continuous `ROOT.v.N` |
| Dates / frequency | `start`/`end` are `YYYY-MM-DD`, inclusive, `start <= end` (Alpaca also RFC-3339; Databento converts inclusive end to its exclusive endpoint). `frequency` canonical: `"daily"` default; `"hourly"`, `"minute"`; `Frequency` enum = tick, second, minute, 5minute, 15minute, 30minute, hour, hourly (alias), daily, weekly, monthly, quarterly, annual (which string a provider accepts is provider-specific); `BarType` = time, volume, trade, dollar, tick |
| `DataObject(data, metadata: Metadata)` (`core/models.py`) | `validate_schema` requires `timestamp` (any `pl.Datetime`) and OHLCV in {Float64, Float32, Int64, Int32}; `to_canonical_schema()` casts Float64, adds `dividends`=0.0 and `splits`=1.0 if missing, returns `[timestamp, open, high, low, close, volume(, dividends)(, splits)]` -- single-symbol, NO `symbol` column (differs from `MultiAssetSchema`; which is the storage-layer contract is unresolved in the notes) |
| `Metadata` | `provider`, `symbol`, `asset_class`, `bar_type="time"`, `bar_params={}` (time `{"frequency": "1min"}`; volume `{"threshold": 1000}`; trade `{"threshold": 500}`; dollar `{"threshold": 100000}`), `exchange="UNKNOWN"`, `calendar=None` (pandas_market_calendars name), `start_date`, `end_date`, `last_updated`, `schema_version="1.0"`, `download_utc_timestamp`, `data_range`, `provider_params`, `attributes`; property `frequency` -> `bar_params["frequency"]` for time bars else `f"{bar_type}_{threshold}"` (e.g. `volume_1000`) |
| Storage key | Logical string `asset_class/frequency/symbol` (e.g. `equities/daily/AAPL`), <= 160 bytes, validated by `validate_storage_key`, encoded to one filename component; `PathValidator.parse_storage_key` returns `(asset_class, frequency, symbol)` |
| Versioned key directory | `<key_path>/generations/<generation_id>/` (immutable data), `.staging-<id>/` (in-flight), `commits/<commit_id>.json` (manifest + metadata). Exactly one current commit; `GENERATION_RETENTION = 3`; `CommitState` = one generation + metadata; `preserve_metadata=True` keeps prior metadata identity |
| Hive layout | Partition columns derived from `timestamp` per granularity (default MONTH -> `year=YYYY/month=MM/`); `read(start_date, end_date)` prunes partitions before scanning. Flat: one Parquet per key. Chunked: fixed-size date chunks (`chunk_size_days=30`) + JSON chunk index. Incremental protocol: per `(symbol, provider)` a combined file + chunk files + metadata (`last_update`, `records_added`, `chunk_file`) |
| `StorageConfig` beta keys | `backend` must be `"filesystem"`; `atomic_writes` must be `True` ("Storage writes are always atomic"); `generate_profile` always rejected ("call generate_profile() explicitly"); `partition_cols` must equal one granularity's list and not conflict with `partition_granularity` |
| Wide format | `pivot_to_wide` -> columns `{value}_{symbol}` (`close_AAPL`); reversible with `pivot_to_stacked`. Docstring: wide "does NOT scale well beyond ~100 symbols" (100 x 5 = 500 columns; most tools struggle > 1000) -- keep stacked |
| Corporate actions | `split_ratio` = new shares per old share on the event date; raw values retained; adjusted values rebased to the share basis of the **latest** observation via future cumulative product -> `price_adjustment_factor`; `apply_dividends` preserves total close-to-close returns into `adj_close` |
| COT | COT frame has `report_date` (snapshot date); schedule frame maps `report_date -> available_at` (tz-aware release time); joined frame carries `available_at`; features prefixed `cot_` incl. percentage-of-open-interest columns |
| Futures rolls | `select_contracts` -> unique chronological `date`/`symbol` rows; `RollEvent` = one switch with paired old/new closes observed on the same switch date; adjustment methods emit adjusted OHLC ("common adjusted-column contract"); `ContractInfo` hashable with `contract_month` `"YYYY-MM"`; price units explicit via `price_conversion_factor(from_unit, to_unit)` (cents vs dollars) |
| Fundamentals | `fetch_financials` -> long form, one row per (symbol, provider, statement_type, period_type, period date, line item, value, currency, source) (inference from helper kwargs); `fetch_company_metrics` -> one row per metric with `period`/`as_of`; `statement` aliases -> `StatementType`, `period` -> `PeriodType` |
| Macro / events | FRED single-value series (OHLC checks skipped); FXMacroData announcement rows carry known-at release timestamps, `PayloadResult` when `include_metadata=True`; Kalshi/Polymarket prices are probabilities in [0, 1], Polymarket YES + NO ~= 1 minus spread |
| Validation artifacts | `ValidationIssue` (severity, message), `ValidationResult`, `ValidationReport` (JSON / DataFrame); `ValidationRuleConfig` (`extra="forbid"`) fields: `check_nulls`, `check_price_consistency`, `negative_price_policy`, `check_negative_volume`, `check_duplicate_timestamps`, `check_chronological_order`, `check_price_staleness`, `check_extreme_returns`, `max_return_threshold`, `staleness_threshold`, `volume_spike_threshold`, `price_gap_threshold`, `check_price_continuity`, `check_volume_spikes`, `check_weekend_trading`, `check_market_hours`, `asset_class=AssetClass.EQUITY`; rule-set YAML `{name, description, default, rules: {pattern: ...}}` |
| Anomaly artifacts | `Anomaly` Pydantic model (severity, type); `AnomalyReport.to_dataframe()`; reports saved under `output_dir` |
| Config YAML | `${ENV_VAR}` interpolation (a literal `"${VAR}"` value counts as NOT configured; `enabled: false` disables a provider); includes; relative paths resolved against the declaring file; `to_yaml` writes env references, never resolved secrets; `routing` = list of `{pattern, provider}`; CLI `storage` section builds the backend |
| Export | `ExportConfig` (Pydantic), `ExportResult`; Excel keeps tz offsets; JSON ISO strings |
| Synthetic checkpoints | `<data_dir>/synthetic/checkpoints/<generator>/<experiment>/` with metadata JSON + sample arrays; `CalendarMode = "equity" | "continuous"` |

## Built-in guardrails

**Point-in-time / lookahead**
1. COT: `report_date` is the snapshot, not public availability, and CFTC holiday releases are not fixed offsets; joining on `report_date` leaks ~3 days of future positioning. Always `attach_cot_release_schedule(cot, official_schedule)` then `combine_cot_ohlcv` (as-of on `available_at <= timestamp`; schedule timestamps forced tz-aware, naive observations localized via `observation_timezone="UTC"`). Detect: any COT feature frame lacking `available_at`.
2. Roll selection: all rank-based strategies pick by the PREVIOUS observation's volume/OI rank (`_lagged_rank_selections(metric, rank, minimum, confirmation, min_days_between_rolls)`) after a PIT warm-up; `HighestVolumeRoll` matches Databento's volume rule; `VolumeBasedRoll(min_days_between_rolls=20)` suppresses whipsaw; `TimeBasedRoll(days_before_expiration=5, use_business_days=True)` uses the observed trading calendar; `FirstNoticeDateRoll(days_before_first_notice=1)` for physically settled contracts. Detect: switch date == first day the new contract leads in volume.
3. Macro revisions: FRED data "often gets revised after initial release"; pass `vintage_date="YYYY-MM-DD"` to get the series as known then ("essential for backtesting macro-based strategies"); FXMacroData carries known-at timestamps and a `revisions` parameter. Detect: a macro feature built from the latest revised series without a vintage.
4. Corporate actions rebased to the latest observation so history never needs re-adjusting when new splits arrive; dividends preserve total returns. Detect: `price_adjustment_factor != 1.0` on the last row.
5. Adjustment is explicit, never implicit: Alpaca `adjustment="raw"` by default (`split|dividend|all` on request); Tiingo exposes raw and adjusted. Never mix adjusted and unadjusted series across providers. Detect: discontinuities at split dates.
6. `APPEND_ONLY` is the PIT-safe update mode (published history immutable, inference on intent); `INCREMENTAL` replaces overlapping timestamps with the newer fetch (`_merge_data`: concat + `unique(keep="last")` + sort); `FULL_REFRESH` replaces the key; `BACKFILL` inserts only rows inside detected gaps.

**Schema and value integrity (every fetch, `ValidationMixin._validate_ohlcv`)**
7. Missing `timestamp/open/high/low/close/volume` -> `DataValidationError("Missing required columns")`; `symbol` auto-added; nulls in any canonical column raise; dtype checks (tz-aware timestamp, String symbol, numeric OHLCV).
8. Symbol identity: a single-symbol response must contain exactly one symbol equal to the requested upper-cased symbol ("Response contains N symbols" / "does not match requested symbol"); prevents silent instrument drift.
9. Finite values: NaN/inf in OHLCV or negative volume raise. OHLC invariants (`high < low`, `high < open/close`, `low > open/close`) under `ohlc_mode="strict"` raise "Found N rows with invalid OHLC relationships"; `"drop"` filters and logs `n_dropped/n_total`; `"warn"` keeps. Set `provider.ohlc_mode = "drop"` for noisy sources. FRED, Kalshi, Polymarket skip OHLC checks (not prices).
10. Duplicate `(timestamp, symbol)` pairs raise "Found N duplicate timestamp and symbol rows"; zero-column no-data responses become an empty canonical frame; `_validate_inputs` rejects empty symbol, non-`%Y-%m-%d` dates, `start > end`.
11. `StorageManager.load/update(enable_validation=True)` run `_validate_data` ("OHLCV and cross-validation") before writes; `AssetValidator` adds crypto/equity/forex/stablecoin/OHLC rule sets; `MultiAssetSchema.validate(strict=True)`, `AssetSchema.validate_dataframe -> (False, issues)` on failure.
12. Post-load screening: `OHLCVValidator` (finite, nulls, OHLC, negative prices per `negative_price_policy`, negative volume, duplicates, order, staleness 5 identical closes, `|return| > 0.5`); `CrossValidator` (price gap > 10%, volume > 10x average, weekend bars for non-crypto, bars outside 9:30-16:00 ET, optional `reference_df`); `AnomalyManager` (MAD/z-score/IQR threshold 3.0, `min_samples=20`, volume z-score window 20, `max_unchanged_days=5`; override precedence base < asset_class < symbol). Detect: `report.has_critical_issues()`, `ValidationReport.passed`, CLI `validate --anomalies --severity`.
13. Asset-class presets (`ValidationRulePresets`): equity ret 0.2 / stale 5 / vol 10.0 / gap 0.05 with weekend + hours checks; crypto 0.5 / 3 / 20.0 / 0.15, no weekend/hours; forex 0.1 / 10 / 5.0 / 0.02, hours off; commodity and futures `negative_price_policy="warn"`, 0.15 / 5 / 8.0 / 0.05 (negative WTI/spreads allowed with warning). Rule resolution: exact symbol -> first prefix glob (`BTC*`) -> `default_rule` -> `ValidationRuleConfig()`.

**Calendar, sessions, gaps**
14. `assign_sessions(outside_session="null"|"raise"|"drop")`: default keeps off-session bars with null `session_date`; use `"raise"` to catch bad timestamps/timezones. `complete_sessions(fill_method="forward", zero_volume=True)` forward-fills prices but writes ZERO volume so synthetic rows stay identifiable and never fabricate activity.
15. Crypto 24/7: `CryptoCalendar.find_gaps(threshold_multiplier=1.5)` flags intervals > 1.5x expected spacing, excluding exchange maintenance windows; `validate_continuity`/`validate_volume_profile` return issue strings; `is_crypto=True` disables weekend checks.
16. Gaps: `utils.gaps.GapDetector(tolerance=0.1)` flags intervals > expected + 10% (inference), suppresses overnight/weekend false positives (`_spans_non_trading_hours`); `update_manager.GapDetector(exclude_weekends=False, tolerance_days=0)` compares to `_get_expected_delta`; filling is opt-in (`fill_gaps(method="forward")`). Backfill proposes only material recent gaps (`min_gap_days=5`, `max_age_days=30`) and consolidates windows closer than 7 days into batches <= 30 days.
17. Timestamp precision/zone never degraded: `_concat_dtype`, `pandas_to_polars` (UTC-normalized, precision kept), `_normalize_loaded_timestamps`, Excel export keeps offsets, storage metadata parsed as aware UTC; `determine_update_range` bounds pass through `_ensure_datetime` (UTC) and return `"none"` when `requested_end <= last_timestamp` (no redundant calls), `"full"` when key absent/empty, else `"incremental"` from `last_timestamp + 1 day`.
18. Staleness health: `MetadataTracker.check_health(stale_days=7)`, `BulkManager.get_stale_symbols(max_age_days=7)`, CLI `status --stale-days 7` -> `healthy|stale|error` from `end_date` or `last_updated` vs now (UTC).

**Provider resilience and cost**
19. Routing refusal before I/O: `validate_routes` rejects a batch whose symbols cannot all be routed (`ProviderRoutingError` = "cannot be assigned to a provider without guessing"); defaults only route `EUR_USD` -> oanda and `ROOT.v.N` -> databento; pass `provider=` for bare tickers. `get_provider(required_capability=...)` refuses e.g. `fama_french` for ohlcv. Partial batches: `fail_on_partial=False` returns successes; `True` raises; `return_exceptions=True` returns per-symbol exceptions.
20. Rate limits: client throttles (Alpaca 200/min, FRED 100 of 120/min, Kalshi ~10/s, Polymarket 60/min CLOB + 30/min Gamma, Yahoo chunks of 50 with 1.0 s delay, Binance Public `max_concurrent=5`); on 429 the delay comes from `Retry-After` headers; `_provider_retry_wait` honors hints "without shortening exponential backoff"; `AsyncAdaptiveRateLimiter` halves rate (`backoff_factor=2.0`) then `restore_rate()`; `GlobalRateLimitManager` shares limiters per provider process-wide.
21. Circuit breaker: 5 consecutive `NetworkError` failures open it; half-open retry after 300 s; `RateLimitError` and auth/validation errors never trip it (`excluded_exceptions`); `metrics()`, `_reset_circuit_breaker()`. Pagination cursors (Alpaca `_next_page_token`, OKX `_next_after`) must progress and are page-capped to stop infinite loops.
22. Databento: `_inclusive_end_string` avoids a missing last day; `consolidate_publishers=True` merges per-publisher OPRA bars so the duplicate guard passes; `estimate_opra_cost`/`estimate_cost()` before any metered spend; `download_test()`, `skip_existing=True`, `continue_on_error=True`, `delay_between_products=1.0`, `max_retries=3`, `DownloadProgress` persisted for resume; continuous downloads make ONE API call per product and partition locally.
23. Feed coverage: Alpaca `feed="iex"` is ~2-3% of US volume (volume features biased low); use `"sip"` for the consolidated tape (free plan lacks the last 15 min); unknown feed or bad `asset_class` -> `DataValidationError` at construction (a typo "would otherwise silently route to the wrong endpoint and surface as a misleading 404").
24. CoinGecko: `_validate_daily_window` rejects windows that return four-day candles; `_round_to_valid_days` snaps to API-legal values; daily OHLC re-aggregated to UTC days with 24h volume at UTC boundaries. Yahoo: `_drop_priceless_rows`; `_resolve_empty_download` distinguishes a truly empty range from a failed download.
25. Contract symbols: `parse_contract_symbol(symbol, reference_year=...)` resolves 1-/2-digit years to the nearest year (`ESH5` -> 2025 not 2005). Price units must be explicit for CHRIS data (`_normalize_price_units`, `ContractSpec.__post_init__`); cents-vs-dollars errors silently scale P&L 100x. Roll adjustments apply each same-date gap/ratio exactly once; frames missing OHLC are rejected.
26. Incremental safety: `update(lookback_days=7)` re-fetches the trailing week to capture late corrections; `ProviderUpdater(safety_margin_minutes=5)` backs the end time off so an in-progress bar is not stored as final (inference); `dry_run=True` computes the range without writing; `initial_load_days=365` bounded by `get_max_history_days`.

**Storage, security, config**
27. Atomic, crash-safe publishes: stage -> rename into `generations/` -> fsync -> one commit manifest; readers never see a torn write; leftover `.staging-*` dirs are deleted under the key's writer lock on the next write; `atomic_writes=False` is rejected. Single-writer `FileLock` per key (`lock_timeout=30.0`, must be > 0; `LockError(key, timeout)`). History bounded to 3 commits; `delete(key)` makes the key inaccessible first.
28. Path containment: `contained_path` rejects symlink escapes; `PathValidator`/`validate_path_component` block `../x` style keys (`PathTraversalError`); `restore_backup` refuses archive members outside the target (`overwrite=False`); `LearnedSyntheticProvider` loads NumPy without pickle and bounded JSON metadata; credential URLs never appear in error messages.
29. Legacy migration refuses to guess: explicit `key_mapping` required, incomplete/ambiguous maps rejected, sources moved to a backup root, each migrated key re-read and `_verify_frame`-compared, rollback on any error (`LegacyStorageMigrationError`).
30. Credential hygiene: `validate_secrets` rejects plain-text secrets; `to_yaml` writes only `${ENV}` references; `write_yaml` atomic with owner-only permissions; `validate_provider_credentials` rejects credentials the provider cannot use; `validate_extra_fields` keeps structural keys out of `settings`; unresolved `${VAR}` or `enabled: false` -> provider not advertised. `StorageConfig` is frozen with `extra="forbid"` so typos fail fast.
31. Synthetic determinism: `create_rng(seed)` uses a stable bit generator; `derive_symbol_seed` avoids Python's randomized `hash()`; `validate_synthetic_frequency` rejects unsupported frequencies first. Offline deterministic tests are mandatory for PRs.

## Usage patterns

Offline smoke test (no credentials):
```python
from ml4t.data.providers import SyntheticProvider
df = SyntheticProvider(seed=42).fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-10", "daily")
assert {"timestamp", "symbol", "open", "high", "low", "close", "volume"} <= set(df.columns)
```

Facade: fetch, persist, update, parallel panel:
```python
from ml4t.data import DataManager
with DataManager() as dm:                       # ML4T_DATA_PATH or ./data
    df = dm.fetch("AAPL", "2024-01-01", "2024-12-31", provider="yahoo")
    key = dm.load("AAPL", "2020-01-01", "2024-12-31", asset_class="equities", provider="yahoo")
    dm.update("AAPL", lookback_days=7, fill_gaps=True)          # key = equities/daily/AAPL
    dm.validate_routes(["AAPL", "MSFT"], provider="yahoo")      # refuse before I/O
    panel = dm.batch_load(["AAPL", "MSFT"], "2024-01-01", "2024-12-31", max_workers=4)
    panel = dm.complete_sessions(dm.assign_sessions(panel, exchange="XNYS"), exchange="XNYS")
```

Point-in-time macro and COT:
```python
from ml4t.data.providers.fred import FREDProvider
fred = FREDProvider()                                        # FRED_API_KEY
gdp_pit = fred.fetch_ohlcv("GDP", "2023-01-01", "2023-12-31", vintage_date="2024-01-15")
from ml4t.data.cot import COTFetcher, attach_cot_release_schedule, combine_cot_ohlcv, create_cot_features
cot = COTFetcher().fetch_product("ES", start_year=2020, end_year=2024)
cot = attach_cot_release_schedule(cot, official_schedule)   # adds tz-aware available_at
feats = create_cot_features(combine_cot_ohlcv(ohlcv, cot), prefix="cot_")
```

Futures: cost-aware download, continuous contract:
```python
from ml4t.data.futures.continuous_downloader import ContinuousDownloader, ContinuousDownloadConfig
cfg = ContinuousDownloadConfig(products=["ES", "CL", "GC"], start="2011-01-01", end="2025-12-31",
                               tenors=[0, 1, 2], schema="ohlcv-1h")
dl = ContinuousDownloader(cfg); print(dl.estimate_cost()); dl.download_all(skip_existing=True)
from ml4t.data.futures import build_continuous_contract, VolumeBasedRoll, BackAdjustment
es = build_continuous_contract("ES", roll_strategy=VolumeBasedRoll(min_days_between_rolls=20),
                               adjustment_method=BackAdjustment(), data_source="databento")
```

Corporate actions, anomaly and validation screens:
```python
from ml4t.data.adjustments import apply_corporate_actions
adj = apply_corporate_actions(raw)                     # adds adj_close, price_adjustment_factor
assert adj["price_adjustment_factor"][-1] == 1.0       # rebased to latest observation
from ml4t.data.anomaly import AnomalyManager
assert not AnomalyManager().analyze(adj, "AAPL", asset_class="equities").has_critical_issues()
from ml4t.data.validation.ohlcv import OHLCVValidator
from ml4t.data.validation.cross_validation import CrossValidator
from ml4t.data.validation.rules import ValidationRulePresets
from ml4t.data.validation.report import ValidationReport
r = ValidationRulePresets.crypto_rules(); rep = ValidationReport()
rep.add_result(OHLCVValidator(max_return_threshold=r.max_return_threshold,
                              staleness_threshold=r.staleness_threshold).validate(df), "ohlcv")
rep.add_result(CrossValidator(is_crypto=True, volume_spike_threshold=r.volume_spike_threshold,
                              price_gap_threshold=r.price_gap_threshold).validate(df), "cross")
rep.print_summary(); assert rep.passed
```

Storage + incremental update outside the facade:
```python
from ml4t.data.storage.config import StorageConfig
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.storage.metadata_tracker import MetadataTracker
from ml4t.data.update_manager import IncrementalUpdater, UpdateStrategy
cfg = StorageConfig(base_path="~/ml4t-data", strategy="hive", compression="zstd", partition_granularity="month")
store = HiveStorage(cfg)
upd = IncrementalUpdater(strategy=UpdateStrategy.APPEND_ONLY)       # PIT-safe: never rewrite history
start, end, kind = upd.determine_update_range(store, key, req_start, req_end)
if kind != "none":
    new = provider.fetch_ohlcv(symbol, start, end, "daily")
    res = upd.update_incremental(store, MetadataTracker(cfg.base_path), key, new, provider="yfinance")
lf = store.read(key, start_date=start, end_date=end, columns=["timestamp", "close"]); df = lf.collect()
```

Book-reader managers, synthetic registry, universes, async:
```python
from ml4t.data.etfs import ETFDataManager; from ml4t.data.macro import MacroDataManager
from ml4t.data.crypto import CryptoDataManager; from ml4t.data.synthetic import SyntheticRegistry
etf = ETFDataManager.from_config("configs/ml4t_etfs.yaml"); etf.download_all(); spy = etf.load_ohlcv("SPY")
macro = MacroDataManager.from_config("configs/ml4t_etfs.yaml"); macro.download_treasury_yields()
regime = macro.get_regime(threshold=0.5)                   # YIELD_CURVE_SLOPE regime
cr = CryptoDataManager.from_config("configs/ml4t_etfs.yaml"); cr.download_premium_index(); prem = cr.load_premium_index()
reg = SyntheticRegistry(); gan = reg.get_provider("timegan", seed=42).fetch_ohlcv("SYNTH", "2024-01-01", "2024-12-31", "daily")
from ml4t.data.universe import Universe; syms = Universe.get("sp500")
from ml4t.data.managers.async_batch import async_batch_load
async with MyAsyncProvider() as p:
    panel = await async_batch_load(p, syms[:50], "2024-01-01", "2024-12-31", max_concurrent=10)
```

CLI: `ml4t-data fetch AAPL --start 2024-01-01 --end 2024-12-31 --provider yahoo`; `ml4t-data update-all --config configs/x.yaml --dry-run`; `ml4t-data download-futures --config configs/ml4t_futures.yaml --dry-run`; `ml4t-data download-cot --list-products`; `ml4t-data validate --all --anomalies --severity warning`; `ml4t-data status --stale-days 7`.

## Defaults and configuration keys

| Key / parameter | Default | Where |
|---|---|---|
| `ML4T_DATA_PATH` | `./data` (CWD); legacy `QLDM_DATA_ROOT` deprecated | env |
| Book configs | `configs/ml4t_etfs.yaml` (50 ETFs), `configs/ml4t_futures.yaml` (70 CME products) | repo |
| `DataManager` | `output_format="polars"`, `enable_validation=True`, `frequency="daily"`, `asset_class="equities"`, `bar_type="time"`, `exchange="UNKNOWN"`, `calendar=None`, `max_workers=4`, `fail_on_partial=False`, `fetch_missing=True` | data_manager.py |
| `update` | `lookback_days=7`, `fill_gaps=True`, `initial_load_days=365`; `update_all(_max_workers=1)`; `get_stale_symbols(max_age_days=7)` | storage_manager / bulk_manager |
| sessions | `assign_sessions(outside_session="null", bar_frequency="auto")`; `complete_sessions(fill_method="forward", zero_volume=True)` | sessions/ |
| async | `max_concurrent=10`, `return_exceptions=False`; `AsyncBaseProvider(max_concurrent=None)` | managers/async_batch, providers/async_base |
| Canonical schema | `timestamp Datetime("us","UTC")`, Float64 numerics, `ohlc_mode="strict"` | providers/mixins/validation.py |
| `ProviderCapabilities.rate_limit` | `(60, 60.0)` = (calls, period seconds) | providers/protocols.py |
| Circuit breaker | `failure_threshold=5`, `reset_timeout=300.0`, `expected_exception=NetworkError`, `excluded_exceptions=(RateLimitError,)` | providers/mixins/circuit_breaker.py |
| Retry | `with_retry(max_attempts=3, min_wait=1.0, max_wait=60.0, multiplier=2.0)`; `AsyncAdaptiveRateLimiter(backoff_factor=2.0)` | utils/ |
| `FileLock` | `timeout=30.0`, `retry_delay=0.1`; `StorageConfig.lock_timeout=30.0` (> 0) | utils/locking, storage/config |
| `StorageConfig` | `strategy=hive`, `compression=zstd`, `partition_granularity=month` (`["year","month"]`), `metadata_tracking=True`, `base_path=resolve_data_root()` | storage/config.py |
| Storage internals | `GENERATION_RETENTION=3`; `ChunkedStorage(chunk_size_days=30, compression=SNAPPY)`; `MAX_KEY_BYTES=160`; `get_update_history(limit=10)`; `check_health(stale_days=7)`; `create_backup(compression=True)`, `restore_backup(overwrite=False)`; `migrate(dry_run=False, target_version=None)` | storage/ |
| Update / gaps | `IncrementalUpdater(strategy=INCREMENTAL)`; `GapDetector(exclude_weekends=False)`, `detect_gaps(frequency="daily", tolerance_days=0)`; `utils.gaps.GapDetector(tolerance=0.1)`, `fill_gaps(method="forward")`; `BackfillManager.identify_candidates(min_gap_days=5, max_age_days=30)`; `GapOptimizer(max_batch_days=30)`, `merge_threshold_days=7`, `threshold_gaps=3`; `ProviderUpdater(safety_margin_minutes=5)` | update_manager, utils/ |
| `OHLCVValidator` | `max_return_threshold=0.5`, `staleness_threshold=5`, `negative_price_policy="forbid"` | validation/ohlcv.py |
| `CrossValidator` | `volume_spike_threshold=10.0`, `price_gap_threshold=0.1`, `is_crypto=False`, market hours 9:30-16:00 ET | validation/cross_validation.py |
| Presets | equity 0.2/5/10.0/0.05; crypto 0.5/3/20.0/0.15 (no weekend/hours); forex 0.1/10/5.0/0.02 (hours off); commodity = futures: neg-price `warn`, 0.15/5/8.0/0.05 | validation/rules.py |
| Anomaly | `DetectorConfig(enabled=True, threshold=3.0, window=None, min_samples=20)`; `ReturnOutlierConfig(method="mad"\|"zscore"\|"iqr", threshold=3.0)`; `VolumeSpikeConfig(window=20, threshold=3.0, min_volume=0)`; `PriceStalenessConfig(max_unchanged_days=5, check_close_only=False)`; `AnomalyConfig(enabled=True, report_severity_threshold="info", save_reports=True, asset_overrides={}, symbol_overrides={})` | anomaly/config.py |
| Crypto calendar | `find_gaps(threshold_multiplier=1.5)`, `frequency="daily"` | calendar/crypto.py |
| Corporate actions | `split_col="split_ratio"`, `dividend_col="ex-dividend"`, `volume_col="volume"`, `close_col="adj_close"` | adjustments/core.py |
| COT | `cot_date_col="report_date"`, `available_at_col="available_at"`, `date_col="timestamp"`, `observation_timezone="UTC"`, `prefix="cot_"`, `download_all(skip_existing=True)` | cot/ |
| Futures | dataset `GLBX.MDP3`; parent `{PRODUCT}.FUT`; continuous `tenors=[0,1,2]`, `schema="ohlcv-1h"`, `.v.N`/`.c.N`; `delay_between_products=1.0`; `max_workers=4`; `max_retries=3`; `skip_existing=True`; `continue_on_error=True`; `include_definitions=True`, `parallel=1`; quarterly months `[3,6,9,12]`; example starts `2016-01-01` (individual) / `2011-01-01` (continuous) | futures/ |
| Rolls | `lookback_days=1`, `min_days_between_rolls=20`, `days_before_expiration=5`, `use_business_days=True`, `days_before_first_notice=1`, `rank=0`, `min_volume=0`, `min_oi=0`; builder `data_source="quandl_chris"` (use `"databento"`) | futures/roll.py, builder |
| Databento provider | `dataset="GLBX.MDP3"`, `adjust_session_dates=False`, `session_start_hour_utc=0`, `version=0`, `stype_in="raw_symbol"`, OPRA schema `ohlcv-1d`, quotes `cbbo-1m`, `right="both"`, `consolidate_publishers=True`, `consolidated_only=True` | providers/databento.py |
| Alpaca | `feed="iex"`, `adjustment="raw"`, 200 calls/min, SIP free-plan delay 15 min | providers/alpaca.py |
| Yahoo | `enable_progress=False`, `chunk_size=50`, `delay_seconds=1.0`, `statement="income"`, `period="annual"` | providers/yahoo.py |
| Crypto providers | Binance `market="spot"`, `timeout=30.0`; BinancePublic `timeout=60.0`, `max_concurrent=5`, premium `interval="8h"`; OKX `market="swap"`; CryptoCompare `exchange="CCCAGG"`; CoinGecko `use_pro=False`, `vs_currency="usd"` | providers/ |
| FRED / FXMacro | FRED 100 req/min, `vintage_date=None`, `forward_fill=True`; FXMacro `limit=500`, `include_metadata=False`, `currency="usd"`, `indicator="policy_rate"`, `quote="usd"`, `commodity="gold"` | providers/ |
| Prediction markets | Kalshi ~10 req/s, `status="open"`, `limit=100`, `align=True`; Polymarket 60/min CLOB, 30/min Gamma, `outcome="yes"`, `interval="1d"`, `active=True`, `search_markets(limit=20)` | providers/ |
| Factors | AQR `region="USA"`, `include_optional=False`, 16 datasets; Fama-French `use_cache=True`, `frequency="monthly"`, `dataset="ff3"`, 70+ datasets | providers/ |
| Other providers | EODHD `exchange="US"`; OANDA `practice=True`; Massive `limit=100`; Finnhub `metric_group="all"`; WikiPrices `cache_in_memory=False` (needs `parquet_path`); ITCH `verify_size=True`; LearnedSynthetic `device="cpu"` (unused), `shuffle=True` (needs `samples`) | providers/ |
| Synthetic | `model="gbm"`, `annual_return=0.08`, `annual_volatility=0.20`, `base_price=100.0`, `base_volume=1_000_000`, Heston `kappa=2.0, theta=0.04, xi=0.3, rho=-0.7`, GARCH `alpha=0.1, beta=0.85`, `calendar_mode="equity"`; Mock `seed=42`, `volatility=0.01`, `trend_rate=0.0`, `gap_probability=0.0`, `range_bound=0.05`; `returns_to_prices(log_returns=True)`; `SyntheticRegistry(checkpoints_subdir="synthetic/checkpoints")`, `load_samples(shuffle=False)` | providers/synthetic, synthetic/ |
| Macro manager | `get_regime(threshold=0.5)` on `YIELD_CURVE_SLOPE` | macro/ |
| Export / format | `format_type="csv"` (single), `"excel"` (batch/pattern); CLI export CSV/JSON/Parquet; `pivot_to_wide(symbol_col="symbol", timestamp_col="timestamp", value_cols=None)`; wide ceiling ~100 symbols | export/, utils/format.py |
| Metadata | `schema_version="1.0"`, `bar_params` thresholds 1000 (volume) / 500 (trade) / 100000 (dollar) | core/models.py |
| Config keys | provider: `api_key`, `api_secret`, `enabled`, `parquet_path`, `samples`, `settings`, `rate_limit` (float -> `RateLimitConfig`); env placeholder `${VAR}`; sections `providers`, `universes`, `datasets`, `workflows`, `schedules` (+ `storage`, `routing` inferred); `StorageConfig` beta keys `backend="filesystem"`, `atomic_writes=True`, `partition_cols`, `generate_profile` (rejected) | config/, storage/config.py |

## Where the book uses it

| Book location (per docstrings/READMEs; chapter mapping partly inferred) | What ml4t-data provides |
|---|---|
| Ch. 2 market data, Ch. 3 microstructure (`ITCHSampleProvider`, tick capability), Ch. 4 fundamental/alternative data (`fetch_financials`, FRED vintages, FXMacroData, Kalshi/Polymarket) | Stacked OHLCV frames, validation reports, PIT macro series, event/probability series |
| Ch. 5 synthetic data ("Chapter 6 notebooks" per `learned_synthetic.py` docstring produce TimeGAN / Sig-CWGAN artifacts; numbering may differ) | `LearnedSyntheticProvider`, `SyntheticRegistry`, `SyntheticProvider` (GBM, jump, OU, Heston, GARCH) |
| Case study `etfs.md` (50 diversified ETFs) and macro regimes | `ETFDataManager`, `MacroDataManager.get_regime(threshold=0.5)` on yield-curve slope |
| Case study `crypto_perps_funding.md` (funding-rate arbitrage) | `CryptoDataManager.download_premium_index/download_perps`, `BinancePublicProvider.fetch_premium_index(interval="8h")`, `OKXProvider.fetch_funding_rates` |
| Case study `cme_futures.md` (70 CME products) | `FuturesDownloader`, `ContinuousDownloader`, roll strategies with lagged ranks, `BackAdjustment`/`RatioAdjustment`, `ContractSpec`, COT PIT workflow |
| Case studies `sp500_options.md`, `sp500_equity_option_analytics.md` | `DataBentoProvider.fetch_option_chain/fetch_option_ohlcv/fetch_option_quotes`, `estimate_opra_cost` |
| Case study `fx_pairs.md` | `OandaProvider` (`EUR_USD`, M1/M5/H1/D), FXMacroData announcements |
| Factor chapters (Ch. 14 latent factors, Ch. 9 model-based features) | `FamaFrenchProvider` (FF3/FF5/UMD, FF48 industries), `AQRFactorProvider` (QMJ, BAB, HML Devil, VME, TSMOM) |
| Ch. 16 simulation, Ch. 18 costs, Ch. 25 live | `ContractSpec` multiplier/tick value; `session_date`; `IncrementalUpdater`/`ProviderUpdater` for daily refresh; `MetadataTracker.check_health` |

Empirical findings recorded in the code: IEX feed ~2-3% of US volume; SIP free plan lacks last 15 min; joining COT on `report_date` leaks ~3 days; Hive partition pruning claimed "7x" faster (docstring, not reproduced); wide format breaks down beyond ~100 symbols.

## Glossary

| Term | Meaning |
|---|---|
| Stacked / long format | One row per `(timestamp, symbol)`; canonical multi-asset layout. **Wide**: one column per `(value, symbol)` |
| Storage key | `asset_class/frequency/symbol` logical string; encoded to one filename component; <= 160 bytes |
| Generation / commit / staging | Immutable published data dir; JSON manifest naming the visible generation; in-flight write dir removed on recovery |
| Hive partitioning | `col=value/` directories (`year=/month=/`) enabling partition pruning on date filters |
| Bar type | time / volume / trade / dollar / tick; non-time bars carry `bar_params.threshold` |
| Point-in-time (PIT) | Only information available at the observation timestamp: COT `available_at`, FRED `vintage_date`, lagged roll ranks, known-at timestamps |
| Release schedule | Frame mapping COT `report_date` -> tz-aware `available_at` |
| Vintage date | Series values as known on a given date (FRED), avoiding revision lookahead |
| `session_date` | Exchange-calendar trading day assigned to each bar; session completion inserts missing sessions with forward-filled prices and zero volume |
| Roll event / back-adjustment / ratio adjustment | Contract switch with paired same-date closes; add the gap to prior history / multiply prior history |
| Tenor | Rank N in Databento `.v.N` (volume) / `.c.N` (calendar) symbology; 0 = front |
| Month codes | Quarterly H=Mar, M=Jun, U=Sep, Z=Dec; monthly products use all 12; `ESH25` = ES, March 2025 |
| First notice date | First day a long holder of a physically settled contract can be assigned delivery; `FirstNoticeDateRoll` rolls before it |
| Parent symbology / definitions / statistics | Databento `ES.FUT` request for all contracts; instrument definition schema (expiry, tick size, multiplier); settlement price and open interest |
| Share basis | Share count a price is quoted in after splits; adjusted prices use the latest observation's basis |
| Premium index / funding rate | Perpetual-swap basis series (Binance Public, OKX) |
| OPRA, `cbbo-1m` | US options tape; Databento consolidated best bid/offer 1-minute schema |
| SIP / IEX | Consolidated US tape vs single-exchange free feed (Alpaca `feed`) |
| Adjustment | `raw | split | dividend | all` corporate-action adjustment of stock bars |
| `ohlc_mode` | `strict | drop | warn` handling of OHLC invariant violations |
| Negative price policy | `forbid` (error), `warn`, `allow`; needed for commodities/futures (negative WTI, spreads) |
| Staleness | N consecutive identical closes (default 5); dataset staleness = no update within `stale_days` (7) |
| Severity | `info < warning < error < critical` (validation and anomaly reports) |
| MAD | Median absolute deviation, default outlier method (threshold 3.0) |
| Update strategy | `incremental | append_only | full_refresh | backfill` |
| Gap / backfill | Interval between bars exceeding expected delta (+ tolerance); fetching only gap windows and inserting them |
| Circuit breaker | closed -> open after 5 counted `NetworkError`s -> half-open after 300 s |
| Rate-limit tuple | `(calls, period_seconds)` client-side throttle; adaptive limiter halves rate after 429 |
| `manager_compatible=False` | Provider cannot be driven by `DataManager` (factor, macro, tick, artifact providers) |
| CalendarMode | `"equity"` (exchange sessions) vs `"continuous"` (24/7) for synthetic timestamps |
| CCCAGG | CryptoCompare cross-exchange aggregate index |
| Kalshi Series / Event / Market; Polymarket condition ID / slug / token ID | Prediction-market taxonomy; prices are probabilities |
| Profile | `DatasetProfile` of `ColumnProfile`s stored as JSON beside the data |
| Template Method | `fetch_ohlcv` / `ProviderUpdater.update_symbol` fix the steps; subclasses fill `_fetch_*` / `_transform_*` hooks |

## Related references

- `workflow.md` -- where data acquisition sits in the end-to-end ML4T pipeline.
- `guardrails.md` -- the PIT / lookahead rules this library enforces mechanically (COT, vintages, rolls, adjustments).
- `decision_rules.md` -- choosing providers, feeds, update strategies and validation presets.
- `companion_repo.md` -- `configs/ml4t_etfs.yaml`, `configs/ml4t_futures.yaml`, data-root layout.
- `chapters/02_financial_data_universe.md` -- market data, corporate actions, stacked OHLCV contract.
- `chapters/03_market_microstructure.md` -- ITCH tick samples, bar types.
- `chapters/04_fundamental_alternative_data.md` -- fundamentals helpers, FRED vintages, FXMacroData, prediction markets.
- `chapters/05_synthetic_data.md` -- `SyntheticProvider`, `LearnedSyntheticProvider`, `SyntheticRegistry`.
- `chapters/08_financial_features.md` -- consumes `session_date`, adjusted prices, COT features.
- `chapters/14_latent_factors.md` -- Fama-French and AQR factor providers.
- `chapters/16_strategy_simulation.md`, `chapters/18_transaction_costs.md` -- `ContractSpec` multipliers and tick values feed P&L and cost models.
- `chapters/25_live_trading.md`, `chapters/26_mlops_governance.md` -- incremental updates, staleness health, atomic storage.
- `case_studies/etfs.md`, `case_studies/crypto_perps_funding.md`, `case_studies/cme_futures.md`, `case_studies/sp500_options.md`, `case_studies/sp500_equity_option_analytics.md`, `case_studies/fx_pairs.md`, `case_studies/nasdaq100_microstructure.md` -- the datasets each case study loads through this library.
- `libraries/ml4t_engineer.md`, `libraries/ml4t_backtest.md`, `libraries/ml4t_live.md` -- downstream consumers of the canonical frame.
- `glossary.md`, `evidence.md` -- shared terms and the evidence index.
- Further reading:
  - Asness, Frazzini & Pedersen (2014) "Quality Minus Junk"; Frazzini & Pedersen (2014) "Betting Against Beta"; Asness & Frazzini (2013) "The Devil in HML's Details".
  - Asness, Moskowitz & Pedersen (2013) "Value and Momentum Everywhere"; Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum".
  - Fama & French (1993, 2015); Carhart (1997).
  - Databento docs (https://databento.com/docs); Binance Public Data (https://data.binance.vision); FRED API; CFTC COT via `cot_reports`; NASDAQ ITCH samples (https://emi.nasdaq.com/ITCH/); library docs https://www.ml4trading.io/docs/data/ and https://www.ml4trading.io/docs/data/providers/; repo https://github.com/ml4t/data.
