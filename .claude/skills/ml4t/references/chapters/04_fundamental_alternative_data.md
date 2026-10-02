# Chapter 4: Fundamental and Alternative Data

> Chapter 4 turns point-in-time (PIT) correctness "from a principle into an implementation discipline": every fundamental carries two dates (valid time = the period described, knowledge time = when the market could know it), and a backtest keyed on the wrong one "simply looks better than it was" with no visible error. It equips you to build bitemporal fundamentals from SEC XBRL with a correct as-of query, parse Form 4 and 13F filings with independent reconciliation, resolve entities on identifiers rather than names, re-date macro/COT/on-chain data to publication with source-specific timestamp authority, run a four-question alternative-data evaluation whose hard gates block alone, read prediction-market feeds for the price they actually carry, and store filing text as an auditable PIT corpus. The position it argues: "a fundamentals pipeline is only as good as its historical eligibility logic"; a wrong entity match is worse than no match; a restated series without a vintage archive cannot support a backtest no matter how strong the signal.

## When to use this reference

- Building or auditing a fundamentals panel (XBRL, 10-K/10-Q) for a backtest and choosing which date to key on (period end vs filing/announcement date).
- Joining any lagged release (FRED macro, CFTC COT, DeFi TVL, earnings) onto a price grid with an as-of join, or deciding a release-lag schedule bound for a feed with no publication timestamp.
- Mapping vendor company names, job postings, or news entities to tickers/CIKs (entity resolution, fuzzy thresholds, alias tables with validity dates).
- Parsing SEC Form 4 insider trades or Form 13F holdings (per-manager or SEC bulk archive) and screening them before aggregation.
- Deciding whether to acquire/integrate an alternative dataset (signal, data, legal, commercial; MNPI; cost-recovery bps at the fund's AUM).
- Building features from prediction-market feeds (Kalshi, Polymarket): bid-only prices, zero-price artifacts, threshold ladders, non-comparable `volume` columns.
- Extracting Item 1A / Item 7 sections from 10-K/10-Q text for NLP and measuring filing-to-filing change.
- Working with the Chen-Pelger-Zhu firm-characteristics panel (`case_studies/us_firm_characteristics.md`): splits, rank normalization, train-only IC with HAC t-stats.
- Any time a daily panel was forward-filled, a forward return overlaps, a regime bucket is read as evidence, or a join silently changed row counts.
- Reviewing a backtest that uses fundamentals or macro data for lookahead, revision leakage, or ticker-keyed joins.

## Core ideas (the why)

- **Two dates on every fundamental; only one is usable.** Period end / reference period = valid time; filing / release / announcement date = knowledge time. "Between the two, nobody in the market knew the number." Use the filing/announcement date as the as-of timestamp for anything that will be backtested.
- **Bitemporal as-of query: filter on knowledge time, rank on valid time.** Knowledge time decides admissibility; valid time decides recency. Filter on valid time → classic lookahead; sort on knowledge time → the latest *announced* row wins and restatements walk the answer backwards in fiscal time. The long lag tail says restatements are normal, not edge cases.
- **Timestamp authority is source-specific.** XBRL Frames give the filing date; FRED stamps the *start* of the reference period (not a knowledge date); COT carries no publication timestamp; DeFi Llama carries no vintages; Kalshi carries only the bid. Join on recorded release timestamps where they exist; otherwise set a schedule bound deliberately *late*: "a bound that is a day or two conservative costs a little signal and an aggressive one costs the validity of the whole backtest."
- **Two separate PIT corrections: publication date and vintage.** Re-dating fixes *when* a value became visible, nothing about *which* value. Agencies revise for years; even daily Treasury yields get revised; a threshold rule can fire on the revised series and not the released one.
- **Restated series without a vintage archive cannot support a backtest** — a hard gate no signal strength repairs. Fix: snapshot the feed daily going forward and wait.
- **Identifiers over names, in a deliberate trust order.** CIK/LEI name an entity permanently; CUSIP/ISIN/FIGI name a security (one company issues several); a ticker names a listing at a point in time and is reused (ZOOM vs ZM, 2020). "A pipeline built on tickers will break on the first rename."
- **A wrong match is worse than no match.** An unmatched row is a visible gap; a wrong join "attaches one company's alternative data to another company's returns and every statistic computed afterwards is contaminated in a way that nothing downstream will flag."
- **String similarity cannot resolve corporate history.** Fuzzy compares spelling; embeddings compare meaning; neither recovers renames or subsidiaries, and embedding scores on those sit among the scores of outright errors, so no threshold separates them. Renames and parent–subsidiary links "are recorded in an alias table with validity dates, or they are wrong." Scoring stages reduce how much must be recorded; they never replace the record.
- **A parser is finished when its output reconciles against its input by an independent route.** Counting raw XML tags uses no part of the parser, so agreement is evidence. "A silent `continue` inside a parse loop is how a panel loses rows without anyone noticing."
- **The transaction code decides whether a Form 4 row means anything.** P/S are decisions with the insider's own money; A/M/F/G/D are compensation machinery on a schedule set months earlier, and they dominate share volume.
- **Establish what one row of a panel means before computing anything.** Calendar-day vs trading-session grid changes every window, per-row statistic and spell length ("252 rows is eight months, not a trading year").
- **Forward fill is the only safe fill, and it is still not enough.** The test of a fill is "which observations the value on a date was computed from." Forward fill, trailing averages, causal forecasts pass; interpolation, backward fill, centred windows, whole-sample seasonal adjustment reach forward "and none of them will announce that they did." Forward fill still leaves the release lag.
- **A statistic whose inputs were carried forward is not the statistic its name claims.** A 4-week claims average is computed on weekly observations, not on the forward-filled daily panel.
- **Thresholds must be measured on prior data only.** A rolling std that includes the week being judged "makes a genuinely large move harder to flag the larger it is." Shift the window.
- **Row count is not sample size.** Daily sampling of a 30-day forward return gives ~30 overlapping views per window; a year of data is "a handful of independent windows." Correct with Newey-West at lag = horizon − 1 and decide from the count of independent windows.
- **Serial correlation in monthly ICs overstates significance.** Report the HAC t-stat and how far it moved from the naive one.
- **Selection on the test years is leakage.** Ranking characteristics by IC over the whole history "has already read the test years, and a model built on that ranking inherits the leak." Compute selection statistics on the training split only.
- **Count the relationships you measured.** Twelve signal-horizon pairs produce a largest correlation whether or not anything is there; "reporting the largest without the count is how a screening exercise turns into a finding."
- **Short, sign-changing samples are "unproven," not "refuted."** What changes the answer is longer price history, not a cleverer signal; often "the binding constraint on an alternative-data study is not the alternative data."
- **Alternative data is an acquisition and engineering decision.** Four questions — Signal, Data, Legal, Commercial — combine "by rule rather than by arithmetic." Signal and Commercial only rank; Data (vintages) and Legal can block alone. A weighted score would let a strong signal outvote a failed hard gate.
- **MNPI is about entitlement, not difficulty.** Satellite images and scraped public pages are hard to get but public; data from a company's own systems supplied by someone with a duty to it is non-public "however cheaply it arrives."
- **Free data is not costless.** Break-even return = annual carrying cost / capital informed; both the cost-recovery bar and the target bar fall with AUM, which is why the same dataset is a reasonable purchase at one firm and not another.
- **A column name is a claim like any other.** Polymarket `volume` counts price observations; Kalshi `volume` counts contracts traded. "Nothing errors when the two are compared, and the answer is about neither."
- **Which price does a feed carry?** Kalshi bars carry `yes_bid` → every price is a lower bound short by the spread; a ladder monotonicity check on bids is a staleness diagnostic, not an arbitrage test.
- **Count how many of a feature's values are real before fitting.** A column defined on every row and non-zero on a handful "is a panel in shape only."
- **Report a graph's density before believing its edges.** Among widely held names held by large managers every pair shares hundreds of holders; the edge set then describes index membership. The fix is the slice, not the threshold.
- **Bulk regulatory files are not clean because they are official.** Wrong-unit filings land at the top of any size ranking; the file supplies its own check (value/shares = a price with a knowable range). Say what a screen removed.
- **Text extraction failure is silent.** An extractor returning the table-of-contents entry returns "a plausible-looking short string." Report extraction rates, not an example. Item numbers are form-specific.
- **Preserve paragraph boundaries;** they are the unit a change detector compares. Vocabulary overlap misses rewrites ("a filer can rewrite a paragraph entirely without introducing a single new word").
- **The anonymized academic panel supports prediction and frictionless portfolio return, not execution.** No prices → no currency weights, spread, depth or cost; identifiers do not cross split boundaries.

## Method recipes (the how)

### Bitemporal as-of query on XBRL fundamentals (nb04)

Source: `04_fundamental_alternative_data/04_sec_xbrl_fundamentals`. Data: `load_sec_xbrl_fundamentals()` built by `data/equities/fundamentals/xbrl_download.py` (default 20 large caps × 2022–2024 × 11 us-gaap concepts, ~2–3 min; CLI `--years 2020,...`, `--ciks 320193`, `--concepts Assets,Revenues,NetIncomeLoss`).

1. Frames API `https://data.sec.gov/api/xbrl/frames/{taxonomy}/{concept}/{unit}/{period}.json` returns one concept for all filers for one **calendar-year (CY) quarter**; non-calendar fiscal years map to the CY quarter. Submissions API `https://data.sec.gov/submissions/CIK{cik}.json` supplies filing dates (cache per CIK).
2. Panel columns: `symbol, cik, entity_name, fiscal_quarter_end` (Date, valid time), `announcement_date` (Date, knowledge time), `accession`, lowercase concept columns.
3. Coverage grid: label `YYYYQn`; count a symbol×quarter cell once if any row has `assets` not null (a "2" = original + amendment; summing counts reads restatements as coverage).
4. Filing lag: `lag_days = announcement_date − fiscal_quarter_end`; histogram in 20-day bins to 800 days; read median (≈ statutory 40-day 10-Q window for large filers) and tail separately; mean far above median = second population (restatements, late-attributed facts dated to the later document — the conservative direction).
5. As-of query (the template):

```python
def query_fundamentals_as_of(df, as_of_date):
    q = pl.lit(as_of_date).str.to_date()
    return (df.filter(pl.col("announcement_date") <= q)
              .sort(["symbol", "fiscal_quarter_end", "announcement_date"])
              .group_by("symbol", maintain_order=True).last())
```

6. Sorting on `announcement_date` within quarter makes the restatement win deterministically. Rows with null `announcement_date` are never admissible — report their share. Lookahead comparison: same query with `filter(fiscal_quarter_end <= q)` on the same null-free universe; count symbols whose chosen quarter differs. Pick `AS_OF_DATE = "2023-06-30"` (just after a quarter end) to maximize the gap.

| Default concept set (`xbrl_download.py`) | Kind |
|---|---|
| Assets, StockholdersEquity, LongTermDebt, CashAndCashEquivalentsAtCarryingValue, Liabilities | instant (suffix "I") |
| Revenues, NetIncomeLoss, OperatingIncomeLoss, GrossProfit, NetCashProvidedByUsedInOperatingActivities, PaymentsToAcquirePropertyPlantAndEquipment | duration |

`revenues` is sparse because ASC 606 (2018) moved the top line to `RevenueFromContractWithCustomerExcludingAssessedTax`, kept as its own column. Downstream: `08_financial_features/04_fundamentals_macro_calendar`.

### EDGAR access with EdgarTools (nb02)

Source: `04_fundamental_alternative_data/02_sec_filing_explorer` (live EDGAR; needs `EDGAR_IDENTITY`).

| Task | API |
|---|---|
| Identity | `set_identity(os.environ["EDGAR_IDENTITY"])` — `"<Name> <email>"`, placeholders blocked, no key |
| Filer | `Company("AAPL")`, `Company("1318605")`/`Company("0001318605")`; `.name .cik .tickers .sic`; `find("Microsoft")` (exploration only) |
| Filings | `company.get_filings()`, `get_filings(form="10-K")`, `form=["3","4","5"]`, `.latest()`; `filing.filing_date`, `.accession_no`, `.is_xbrl` (mandatory large filers 2009, all 2011) |
| Statements | `company.get_financials()` → `.income_statement()`, `.balance_sheet()`, `.cashflow_statement()`; `.to_dataframe()` one column per period (digit-leading column names); 10-K = 3 years income statement, 2 balance sheet |
| Form 4 | `filing.obj()` → `.insider_name`, `.issuer.name`, `.common_stock_purchases`, `.common_stock_sales`; try/except per filing, record `parse_error`; scan `N_RECENT_FORM4 = 10` |
| 13F | `manager.get_filings(form="13F-HR").latest().obj()` → `.holdings`, `.report_period`, `.total_value`, `.total_holdings`; `PutCall` blank = stock, `PUT`/`CALL` = options; concentration: drop option rows, group `Issuer`, top `N_TOP_HOLDINGS = 10` |
| Index search | `get_filings(form="10-K", filing_date=f"{window_start}:")` (trailing colon = from date onward); `RECENT_FILING_DAYS = 7` |
| Documents | `filing.text()`, `.html()`, `.open()`, `.attachments` (exhibits, XBRL instance/schema) |

Labels vary by filer ("Net sales" vs "Revenue"), hence regex `TOP_AND_BOTTOM_LINE` for single-company work and XBRL tags for cross-sectional work.

### Form 4 XML parsing with reconciliation (nb03)

Source: `04_fundamental_alternative_data/03_sec_form4_insider_transactions`. Files: `FORM4_DIR = DATA_DIR/"equities"/"positioning"/"form4"/<ticker>/*.xml` from `form4_download.py --ticker TSLA --count 20`. A file of a few hundred bytes is an error page.

1. Walk the XML tree per block: `<issuerName>`, `<reportingOwner>` (one per insider; joint filings → collect owners as a list, carry `n_owners` on every row), `<rptOwnerName>`, `<officerTitle>`, `<nonDerivativeTransaction>` (common stock), `<derivativeTransaction>` (counted only), `transactionCoding/transactionCode`, `transactionDate/value`, `transactionAmounts/transactionShares/value`, `transactionPricePerShare/value` (absent on some types), `transactionAcquiredDisposedCode/value` (A/D).
2. `parse_form4(path) -> {"header": {issuer, owner ("; "-joined), n_owners, title}, "trades": [...], "n_derivative", "n_incomplete"}`; a block missing code/date/shares is counted incomplete, never dropped; price `None` (never 0) when absent.
3. Declare `TRADE_SCHEMA` explicitly (code Utf8, timestamp Date, shares/price Float64, direction/issuer/owner/title Utf8, n_owners Int64) — Polars infers dtype from first rows, so a run starting with gifts would type `price` as null.
4. Reconcile: `raw_common = Σ text.count("<nonDerivativeTransaction>")`, `raw_derivative` likewise; `assert extracted + excluded_incomplete == raw_common`; `assert excluded_derivative == raw_derivative`.
5. `CODE_MAP`: P Purchase, S Sale, A Grant, D Disposition to issuer, M Derivative exercise, C Conversion, F Tax withholding, I Discretionary, X Option exercise, G Gift. `DECISION_CODES = ["P", "S"]`.
6. Per-insider dollar totals: filter decision codes; `value = Σ when(price not null) shares*price else 0`; separate `unpriced_trades`, `unpriced_shares` columns; compare count vs volume tables; weight nets by priced value per insider and direction.

### Entity resolution: four stages (nb05)

Source: `04_fundamental_alternative_data/05_entity_resolution` (toy data; `job_postings` stands in for the alt-data payload).

| Identifier | Names | Scope / licence |
|---|---|---|
| CIK | filer entity (e.g. 0000789019) | SEC filers, permanent |
| LEI | legal entity | global |
| FIGI | instrument on exchange | global, open licence |
| CUSIP | security | US/Canada, licensed |
| ISIN | security | global |
| Ticker | listing, one exchange at a time | reused after delisting |

1. **Stage 1 deterministic** `deterministic_match(source, reference, identifiers=["cik","ticker"])`: for each identifier in trust order, assert reference key unique (`ValueError` otherwise), left-join, assert row count unchanged, `pl.coalesce` into `matched_ticker`/`match_method` so the earlier, more trusted hit is kept; unmatched rows survive as nulls.
2. **Normalize** `normalize_company_name`: upper-case; strip `LEGAL_SUFFIXES` repeatedly until none remains (INCORPORATED, INC., INC, CORPORATION, CORP., CORP, COMPANY, CO., CO, LLC, LLP, LP, LIMITED, LTD, PLC, SA, AG, NV, SE, GROUP, HOLDINGS); remove `,` `.`; `&`→`AND`; collapse whitespace.
3. **Stage 2 fuzzy** `fuzzy_match(query, candidates, candidates_norm, scorer=rfuzz.token_set_ratio)` via `rapidfuzz.process.extractOne` (no cutoff; best candidate + score 0–100). Scorers: `ratio` (typos), `partial_ratio` (containment), `token_sort_ratio` (word order), `token_set_ratio` (word order + extra words — default, because post-normalization differences are usually extra words: location, share class, division).
4. **Threshold sweep** `score_at_threshold` over `THRESHOLD_GRID = [0,10,...,100]`: precision = TP/(TP+FP) over accepted (1.0 if none), recall = TP/resolvable, F1. Labelled set: 12 rows, 10 resolvable (extra words, casing, subsidiary brand "GOOGLE", punctuation, former names "Facebook, Inc."/"Tesla Motors Inc", suffix, spacing "JP Morgan Chase", abbreviation "J&J") + 2 non-matchable ("Zoom Technologies Inc", "Palantir Technologies") that count against precision only. Chosen `FUZZY_THRESHOLD = 70`.
5. **Stage 3 embeddings** `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")` with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` (skipped if uncached); normalize names, `encode(..., normalize_embeddings=True)`, `similarity = Q @ R.T`, argmax (cosine = dot product of unit vectors). Use only to extend candidates; never auto-accept renames/subsidiaries.
6. **Stage 4 security master** `SecurityMaster.add(canonical_name, ticker, cik, aliases, parent_ticker)`; `resolve(query, cik)`: identifier → alias (normalized exact) → fuzzy ≥ FUZZY_THRESHOLD → unresolved; returns `{"ticker", "stage", "score"}`. The toy class omits validity windows; production aliases need them (Facebook → Meta, October 2021).

### FRED macro panel: grid and cadence (nb06)

Source: `04_fundamental_alternative_data/06_fred_macro_eda`. `load_macro()` = calendar-day grid (row count == calendar days in span; weekends/holidays forward-filled); `load_macro_metadata()` → `series, description, native_frequency, kind, formula` (`kind` separates FRED-published from repo-derived).

- Cadence recovery: `value_changes = Σ (col != col.shift(1))`, `changes_per_year = value_changes / years` — a lower bound on releases (consecutive equal rounded prints look like one; bites on `unrate`, rounded to 0.1pp).
- Derived-column check: max `|t10y2y − (dgs10 − dgs2)|` and `|t10y2y − YIELD_CURVE_SLOPE|`; floating-point-level agreement = same quantity; where they disagree trust the recomputation from inspectable inputs.
- VIX bands `VIX_BANDS = [(20, "Unsettled"), (30, "Frightened")]`; `spells_above` run lengths in rows = calendar days; read median and longest spell together, full history vs window. Inverted-curve share = days with `t10y2y < 0` / total.
- Optional live path: `ml4t.data.macro.MacroDataManager` (`download_treasury_yields()`, `get_yield_curve_slope()` = 10Y−2Y as `YIELD_CURVE_SLOPE`, `get_regime(threshold=0.5)` → `risk_on` if slope > 0.5 else `risk_off`); `ml4t.data.providers.fred.FREDProvider().fetch_ohlcv("UMCSENT", start=...)`.

### Macro PIT re-dating, as-of panel, revisions (nb07)

Source: `04_fundamental_alternative_data/07_macro_data_alignment`.

| Series | `RELEASE_LAGS` (calendar days after period end, late end) |
|---|---|
| icsa (weekly claims) | 5 (exact) |
| walcl (Fed H.4.1) | 2 |
| unrate, payems, civpart (employment report) | 7 |
| cpiaucsl, cpilfesl, indpro | 18 |
| m2sl | 28 |
| pcepi, gdp, gdpc1 | 31 |
| `DAILY_SERIES` dff, dgs1/2/3/5/7/10/20/30, t10y2y, vixcls | 0 |

1. `published_observations(panel, series, frequency, lag)`: monthly/quarterly → group by `truncate("1mo"/"1q")`, first row = `stamped_on`, `period_end = stamped_on + period − 1d`, `published_on = period_end + lag`. Weekly → `stamp_weekday` = weekday on which the value changes most often (mode); keep every row on that weekday (repeats included), `period_end = stamped_on`, `published_on = + lag`.
2. `point_in_time_panel`: start from `timestamp` + daily series; for each lagged series `join_asof(releases.sort("published_on"), left_on="timestamp", right_on="published_on", strategy="backward")`.
3. Staleness: median `published_on − stamped_on` per series (largest GDP, smallest weekly claims). The shipped CPI panel "steps up seven weeks before the release it reports."
4. Features on calendar-day windows `CALENDAR_YEAR, CALENDAR_QUARTER, CALENDAR_MONTH = 365, 90, 30`: `yield_curve_change_90d`, `yield_curve_zscore_365d`, `yield_curve_regime`, `vix_change_30d`, `vix_zscore_90d`, `volatility_regime`, `unemployment_change_365d`, `labor_market_regime`, `inflation_yoy = cpi/cpi.shift(365) − 1`. 4-week claims average: `rolling_mean(4)` on weekly releases, then as-of join.
5. Revisions: `load_macro_initial_release()` (ALFRED first-release values via `data/macro/download_alfred.py`) joined to current on `timestamp`; `revision_episodes` collapses consecutive differing dates with the same (first_published, current) pair into one episode (breaks on date gap > 1 day or value change); summarize count and largest revision per series.
6. Price join: `prices.join(features, on="timestamp", how="left")` — prices on the LEFT; count rows with no macro attached.

| Regime | Thresholds |
|---|---|
| `yield_curve_regime` | <0 inverted, <0.5 flat, <1.5 normal, else steep |
| `volatility_regime` (VIX) | <15 calm, <25 normal, <35 unsettled, else frightened |
| `labor_market_regime` (unrate) | <4.0 tight, <6.0 normal, else slack |
| `MacroDataManager.get_regime(threshold=0.5)` | slope > 0.5 risk_on else risk_off |

### CFTC COT positioning (nb08)

Source: `04_fundamental_alternative_data/08_futures_positioning`; `cot_download.py --products ES,CL,GC`; `load_cot`. Positions as of Tuesday close, published Friday afternoon. Downstream: `08_financial_features/03_structural_cross_instrument_features`.

| Format | Categories | Role |
|---|---|---|
| `traders_in_financial_futures_fut` | dealers, asset managers, leveraged money, other reportables, non-reportables | dealers = intermediaries; asset managers = long-only allocators; leveraged money = hedge funds/CTAs |
| `disaggregated_fut` | producers/merchants, swap dealers, managed money, other reportables | producers/merchants = hedgers (informed about the commodity); managed money = speculative |

Loader columns: `product, report_type, report_date, open_interest`; financial `dealer_*, asset_mgr_*, lev_money_*` (+ `other_rept_long/short`, `nonrept_net`); commodity `commercial_*, managed_money_*, swap_*`. `SPECULATIVE_COLUMN = {"ES": "lev_money_net", "CL": "managed_money_net", "GC": "managed_money_net"}`. `ml4t.data.cot.PRODUCT_MAPPINGS[code]` → `ProductMapping(code, cot_name, report_type, description)`.

1. Dedupe `keep_deepest_market`: sort by `open_interest` desc, `unique(subset=["product","report_date"], keep="first")`; check margin deepest/shallowest per date (min and median) since the loader carries no contract-market id.
2. `add_positioning_zscore(reports, net_column, window=52)`: `(x − rolling_mean) / rolling_std`, `None` where std == 0; window counted in reports, ending on the current report.
3. Availability `available_from = report_date + 7d` (Tuesday → next Tuesday; 2 days slack for Friday after-close release and CFTC holiday policy). Join `prices.join_asof(reports.sort("available_from"), left_on="session_date", right_on="available_from", strategy="backward")`; median attached-report age ≈ one week.
4. Signals: `contrarian_signal = −1 if z > 2.0, +1 if z < −2.0, else 0`; `change_signal` from `weekly_change = diff()` vs `change_threshold = weekly_change.rolling_std(26).shift(1)`.
5. Cross-product: inner-join z-scores on `report_date` for ES/CL/GC, `.corr()`. Identity: dealer + asset_mgr + lev_money + (other_rept_long − other_rept_short) + nonrept_net = 0; three plotted categories leave a residual. Category profile: share of weeks net long, median net, weekly change std.

### On-chain TVL vs ETH with overlap-corrected tests (nb09)

Source: `04_fundamental_alternative_data/09_onchain_fundamentals`; `load_defillama_chain_tvl`, `load_coingecko_ohlcv` (free tier = trailing 365 days). TVL = Σ smart-contract balances at current USD prices: a stock, dollar-denominated while held in crypto, so level correlates with price by construction — test growth or standardized level, never level.

- Settings: `CHAINS = ["Ethereum","Solana","BSC","Arbitrum"]`, `FORWARD_DAYS = 30`, `MOMENTUM_DAYS = 30`, `ZSCORE_DAYS = 90`, `REGIME_Z = 1.0`, `RECENT_DAYS = 30`.
- Composition against the published total with an explicit "All other chains" residual bar.
- ETH price: `load_coingecko_ohlcv("ethereum").unique(subset="timestamp", keep="last", maintain_order=True)` (live intraday snapshot doubles the last day).
- Features: `tvl_growth = pct_change(30)`, `tvl_zscore` over 90 days, `tvl_regime` = null during warm-up (not "neutral"), `expansion` if z > 1, `contraction` if z < −1, else `neutral`.
- Forward return `price.shift(−30)/price − 1` from both ends of the window (not a shifted trailing return).
- Tests: (a) regime means as OLS coefficients on indicators (no intercept) with `cov_type="HAC", cov_kwds={"maxlags": FORWARD_DAYS − 1}`; pairwise contrasts via `fit.t_test(weights)` (ravel `.effect/.sd/.tvalue/.pvalue`); (b) OLS of forward return on `tvl_growth`, naive vs HAC t; independent windows = `len(tested)/FORWARD_DAYS`.

### Bulk 13F screening and co-ownership graph (nb10)

Source: `04_fundamental_alternative_data/10_institutional_holdings_13f` (README lists it under 4.4; the notebook header says 4.1); `13f_download.py --mode bulk --quarters 2024Q3` (~80 MB); `load_13f_bulk_holdings("2024Q3")`. Columns: `cik, accession_no, cusip, issuer` (free text as typed), `value_thousands, shares, filing_date, company_name` (+ `put_call`, `report_date`). Quarter label = SEC filing window (`2024Q3` = 1 Sep–30 Nov; Q1 Mar–May, Q2 Jun–Aug, Q4 Dec–Feb).

1. Spelling check: distinct `issuer` per `cusip`.
2. One filing per manager: aggregate per `(accession_no, cik, company_name, filing_date)`, sort `cik asc, filing_date desc, accession_no desc`, `group_by("cik", maintain_order=True).first()`. Never sort by accession number (leading block = filing agent).
3. Implied-price screen: `implied_price = value_thousands/shares` for shares > 0; median per filing; `believable = median.is_between(1.0, 10_000.0)`; report count and value of too_high, too_low, no_price, and totals before screening.
4. Top `TOP_N = 500` managers by reported value; semi-join positions; group by `cusip` with `issuer.mode().first()`, `managers_holding = cik.n_unique()`, aggregate value/shares.
5. Edges: manager→security (`cik, company_name, cusip` → `position_value`); `co_ownership_edges(positions, min_shared=5, universe_cap=500)` pairwise set intersections; density = edges / (n(n−1)/2) over `THRESHOLD_GRID = [5,50,100,150,200,250,300,350]`; overlap coefficient = shared / min(breadth_a, breadth_b). Notebook edges are illustrative; canonical edges come from `load_13f_edges()`.
6. `--mode bulk` = cross-section, one quarter, assembled weeks after the 45-day window closes; `--mode per-cik` = named managers, many quarters, reacts as filings land.

### Four-question alternative-data evaluation (nb11)

Source: `04_fundamental_alternative_data/11_defi_tvl_evaluation`.

| Question | Can block alone? | Method |
|---|---|---|
| Signal | No (ranks) | `SIGNALS = {growth_7d, growth_30d, zscore_90d}` × `HORIZONS = [7,14,30,60]` = 12 relationships; `relationship()` → observations, independent_windows = n/horizon, Pearson r, HAC t (maxlags = horizon − 1); heatmap colour fixed at ±2; stability = rolling `ROLLING_WINDOW = 180`-day corr of growth_30d vs fwd_30d (mean, range, share positive) |
| Data | Yes (vintages) | history years; absent days = Σ(gap_days − 1 \| gap_days > 1); null count; \|daily_change\| > `EXTREME_DAILY_MOVE = 0.20` whole history vs since `MODERN_ERA_START = "2020-01-01"` (filter by date, not level); vintages reconstructible? |
| Legal | Yes | four recorded questions: how obtained; MNPI?; licence; jurisdiction |
| Commercial | No (ranks) | `annual_cost = DATA_FEES (0) + (INTEGRATION_HOURS 40 + MAINTENANCE_HOURS 20) × HOURLY_RATE 150` = $9,000; `capital = AUM × ALLOCATION_SHARE 0.10`; `cost_recovery_bps = annual_cost/capital × 10,000`; `target_bps = × TARGET_RETURN_ON_COST 3.0`; evaluate at AUM 10M, 50M, 500M, 5B |

Verdict table with `hard gate = [False, True, True, False]`; decide from the failed gate alone (here "Blocked: no vintages" → start daily snapshots, revisit in a year).

### Prediction markets: Kalshi and Polymarket (nb12, nb13)

Sources: `04_fundamental_alternative_data/12_kalshi_prediction_markets` (`load_kalshi`; writes `get_output_dir(4, "kalshi") / "kalshi_features.parquet"`), `13_polymarket_prediction_markets` (`load_polymarket`; writes `get_output_dir(4, "polymarket") / "polymarket_features.parquet"`).

| | Kalshi | Polymarket |
|---|---|---|
| Regulation / settlement | CFTC-designated; USD; $25,000 position limit per contract; tick 1¢ | none; USDC on Polygon; no limit; closed to US persons (selection effect); user-proposed listings |
| Price carried | `yes_bid` → lower bound short by the spread | snapshot; most markets appear once, none more than twice → cross-section only |
| `volume` | contracts traded | number of price observations in the day (provider's proxy) — not comparable |
| Ladder regex | `THRESHOLD_TICKER = r"^KXFED-(?<meeting>[0-9]{2}[A-Z]{3})-T(?<threshold>[0-9]+(?:\.[0-9]+)?)$"` (pays if rate **above** threshold) | `LADDER_TICKER = r"(?i)^BITCOIN-ABOVE-(?<threshold_k>\d+)K-ON-(?<resolves>[A-Z]+-\d+)"` |
| Ladder check | `violations = probability.diff().over("meeting") > 0` on the thresholds' own latest date; adjacent differences ≈ P(landing between levels) minus two spreads | pick the (resolves, timestamp) group with most contracts (ties → latest date, then term); `breaks = close.diff() > 0` |
| Features | `MOMENTUM_DAYS = 5`, `VOLATILITY_DAYS = 10`, `CONFIDENT_PROBABILITY = 0.2`: `probability_change = close − close.shift(5)`, `probability_volatility = close.diff().rolling_std(10)`, `probability_zscore` (None when std == 0), `near_certain = close > 0.8 \| close < 0.2`; intraday range excluded (would measure the repair) | `conviction = \|close − 0.5\|`, `extreme = close > 0.8 \| close < 0.2`, `intraday_range = high − low` |

1. Artifact detection `(high > 0) & (open == 0 | low == 0 | close == 0)`; repair: null such closes, `forward_fill().over("symbol")` (never backward); on non-traded bars set open/high/low = close.
2. Per-contract summary: first/last probability, range, `days_the_price_moved = close.diff().ne(0).sum()`, `days_traded = (volume > 0).sum()`.
3. Parse tickers with `str.extract_groups` + `unnest`. Count `change_defined`, `change_non_zero`, `zscore_defined` before fitting.
4. Same-event comparison: Kalshi threshold via `symbol.str.split("-T").list.last()`; Polymarket policy via regex `fed|powell|interest-rate|shelton`; separate axes; compare only genuinely identical contracts.
5. Live (`LIVE=False` by default): `ml4t.data.providers.polymarket.PolymarketProvider().list_markets(active=True, closed=False, limit=200)` → slug, question, volume, liquidity; `.close()`.

### SEC filing text extraction (nb14)

Source: `04_fundamental_alternative_data/14_text_data_extraction` (live EDGAR). Writes `get_output_dir(4, "sec_text") / "sec_filing_sections.parquet"` keyed on accession number. Downstream: `10_text_feature_engineering/09_filing_text_signals`.

| Form | Sections |
|---|---|
| 10-K | Item 1 Business, 1A Risk Factors, 7 MD&A, 7A Quantitative/Qualitative Disclosures, 8 Financial Statements |
| 10-Q | MD&A = Part I Item 2; Risk Factors = Part II Item 1A (`"after": r"PART\s*II"` marker) |

1. Fetch `Company(TICKER).get_filings(form=FORM, amendments=False)[:N_FILINGS]` (two consecutive AAPL 10-Ks); record `cik, company_name, accession_no, form, filing_date, accepted_at (filing.acceptance_datetime), period_end (filing.period_of_report), text (filing.text())`.
2. `html_to_text(html, drop_tables=False)`: BeautifulSoup `html.parser`, decompose script/style (and tables for sentiment work), `get_text(separator="\n")`. `clean_text`: normalize line endings, remove whole-line furniture ("table of contents", "page N", "N of M"), URLs, rules of `_=-`×3+, bullets; collapse spaces per line; cap blank lines at 2.
3. `SECTIONS[form][section] = {"starts": [regex], "ends": [regex], "after"?}`; e.g. MD&A start `^\s*ITEM\s*7\.?\s*[-–—]?\s*MANAGEMENT(?:'|’)?S?\s*DISCUSSION`, ends `^\s*ITEM\s*7A\b`, `^\s*ITEM\s*8\b`; `FLAGS = re.IGNORECASE | re.MULTILINE | re.DOTALL`.
4. `extract_section(text, section, form)`: optional `after` offset; all start matches; end = first end-pattern match after each (else len(text)); keep the **longest span**; `clean_text` the slice.
5. Quality: `word_count == 0` → "nothing extracted"; `< MIN_SECTION_WORDS = 100` → "too short"; `> MAX_SECTION_WORDS = 50_000` → "ran past the boundary"; `boundary_leak` = MD&A contains `^\s*ITEM\s*8\b` or risk factors contains `^\s*ITEM\s*1B\b`. Report extraction rates.
6. Change: `vocabulary_overlap` (Jaccard of lowercase word sets; words only in new/old) and `added_paragraphs(old, new, min_characters=200)` = paragraphs (split on `\n\n`) of new not verbatim (lowercased) in old. Compare only sections with `quality == "ok"` and no boundary leak on both sides.
7. Store parquet keyed on accession number with timestamps and section; do not store raw filing text (reproducible from accession, 20× the size).

### Academic firm-characteristics panel (nb01)

Source: `04_fundamental_alternative_data/01_academic_characteristics`; `load_firm_characteristics(split="all")` via `data/equities/firm_characteristics/download.py`. Chen, Pelger, Zhu (2021): Jan 1967–Dec 2016 monthly US equity observations, 46 rank-normalized characteristics + next-month excess return, anonymous identifiers.

- Columns: `symbol` (anonymous int; loader offsets each split by 1,000,000 so splits cannot be joined by accident), `timestamp`, `ret` (next-month excess return, original scale), `split` (loader docstring: train 1967–1986, valid 1987–1991, test 1992–2016 — read boundaries from the file, treat as indicative), 46 features. Universe: all CRSP securities with all 46 present (tilts from small caps); ~2,000 stocks per month.
- `CHARACTERISTIC_CATEGORIES`: Valuation (BEME, E2P, S2P, CF2P, D2P, A2ME, Q); Profitability (ROA, ROE, PROF, OP, PM, PCM, NI, RNA); Investment (Investment, NOA, OA, AC, AT, D2A); Momentum (r2_1, r12_2, r12_7, r36_13, ST_REV, LT_Rev, Rel2High); Risk (Beta, MktBeta, IdioVol, Variance, Resid_Var); Liquidity/Size (LME, LTurnover, Spread, SUV); Leverage (Lev, OL, FC2Y, C, CF, DPI2A); Other (ATO, CTO, SGA2S). Yearly variables updated end of June (Fama-French convention); monthly ones end of month for use next month.
- Macro companion: 178 series (124 FRED-MD, 46 cross-sectional medians, 8 Welch-Goyal predictors); `include_macro=True`, join on `timestamp`.
- Rank normalization: per-month cross-sectional rank mapped to [−0.5, +0.5] (Kelly-Pruitt-Su 2019; Kozak-Nagel-Santosh 2020). Checks: min/max over 46 columns, null count = 0; fill evenness = ratio fullest/emptiest of 10 equal bins (1 = flat; larger = ties bunch values).
- Redundancy: pooled Pearson `np.corrcoef` over stock-months (valid only because each month is normalized on its own); list upper-triangle pairs with |corr| ≥ `REDUNDANCY_THRESHOLD = 0.5`.
- IC on **train only**: `cross_sectional_ic_series(preds, returns, date_col="timestamp", entity_col="symbol", method="spearman")`; `compute_ic_hac_stats(ic_series, ic_col="ic", label_horizon=LABEL_HORIZON_MONTHS=1)` → `mean_ic`, `t_stat` (HAC), `naive_t_stat`, `effective_lags`; scatter naive vs HAC t with ±2 rules.
- Return distribution per split: mean/std/min/max/q25/q75; density histogram clipped at ±60%, full range in a table.

## Guardrails and pitfalls

### Fundamentals, XBRL, filings

- **Period-end keying** — a 10-K is filed 1–3 months and a 10-Q ~40 days after period end; keying on period end uses numbers nobody had and the backtest just looks better / carry both dates; filter on `announcement_date`; run the correct-vs-lookahead comparison and count symbols whose quarter differs.
- **Ranking on knowledge time instead of valid time** — returns the most recently announced row; restatements walk the answer backwards / sort `[entity, valid_time, knowledge_time]`, take last per entity.
- **Null knowledge time silently dropped** — coverage loss surfaces later as unexplained / count and print the share of rows without `announcement_date`; compare on the same null-free universe.
- **Filing lag from two populations** — mean describes neither routine filings nor restatements / histogram (20-day bins to 800 days); read median and tail separately; treat late-attributed facts as knowledge-time-late.
- **Null concept read as missing disclosure** — ASC 606 moved revenue to `RevenueFromContractWithCustomerExcludingAssessedTax`; imputing recovers nothing / measure coverage per concept; read the null pattern (industry-wide = tagging) before imputing.
- **Summing coverage counts** — original + amendment reads as extra coverage / count a cell once if > 0.
- **Keying on labels instead of XBRL tags** — "Net sales" vs "Revenue" breaks cross-company joins / us-gaap concept names (Frames API) for cross-sectional work; labels only for single-company exploration.
- **Regex over Form 4 text** — no block boundaries; pairs one trade's code with the next's price; crosses into derivatives / walk the XML tree per `<nonDerivativeTransaction>`.
- **Silent `continue` in a parse loop** — rows vanish unnoticed / count incomplete and derivative blocks; reconcile against raw tag counts by an independent route; assert totals.
- **Reading the first `<reportingOwner>`** — joint filings attach trades to the wrong insider / collect owners as a list; carry `n_owners`.
- **Missing price substituted with zero** — dollar totals understate silently / keep `price=None`; aggregate priced rows only; report `unpriced_trades`/`unpriced_shares`.
- **Polars dtype inference on first rows** — a run starting with gifts types `price` as null (download-order dependent) / declare `TRADE_SCHEMA`.
- **All Form 4 rows as signal** — grants, exercises, withholding, gifts dominate volume and carry no view / gate on P/S; compare count vs volume tables.
- **Netting buys and sells unweighted** — selling is many small compensation trades, buying is concentrated / weight by priced value per insider and direction.
- **Malformed Form 4 XML halting a scan** — one bad filing stops the batch / try/except per filing; record `parse_error` so readable share is visible.
- **Filing date as tradeable timestamp** — after-close acceptance is tradeable next session / carry `acceptance_datetime`.
- **EDGAR without identity** — requests blocked / set `EDGAR_IDENTITY` in `.env`; restart the kernel; Docker `docker compose up ml4t`, not `restart`.

### Entity resolution

- **Keying on tickers** — renamed and reused after delisting (ZOOM/ZM 2020) joins two companies / key on CIK or LEI; identifiers in trust order; alias table with validity dates.
- **Wrong entity match** — contaminates every downstream statistic with no flag / prefer no match; include non-matchable rows in the labelled set; measure precision at the threshold; never threshold embedding scores on renames/subsidiaries.
- **Rule-of-thumb threshold** — precision/recall unknown / sweep `THRESHOLD_GRID`; compute precision/recall/F1 with non-matchable rows; pick from the measurement (70 here).
- **Legal suffixes left in names** — shared by every candidate; inflate every score / normalize (strip suffixes repeatedly, punctuation) before scoring and embedding.
- **Non-unique reference key** — silent row multiplication / assert `n_unique == len`; assert row count unchanged after each join; unmatched rows survive as nulls.
- **Alias without validity window** — resolves a 2023 document to a dead name or a pre-rename document to the new one / carry date ranges per alias.

### 13F

- **13F read as AUM or full book** — long-only US-listed equities/ETOs, ≤45-day lag; shorts, foreign, cash, debt absent / say "reported 13F value"; separate option rows.
- **Summing option rows with stock** — option value = underlying controlled / filter `put_call == ""` before portfolio shares.
- **Grouping by issuer name** — free text per filer; thousands of CUSIPs have multiple spellings / group on CUSIP; `issuer.mode()` for legibility only.
- **Sorting by accession number** — leading block = filing agent, ranks agents not dates / latest `filing_date` per CIK.
- **Unit errors in bulk 13F** — a few filings imply tens of thousands $/share and carry trillions / median implied price per filing in [$1, $10,000]; report count and value removed per bucket.
- **Breadth as crowding unweighted** — $100 and $1bn count the same; index funds hold everything / weight by value or restrict to concentrated managers, deliberately.
- **Co-ownership graph saturation** — density ≈ 100% under any round threshold; overlap coefficient saturates; edges = index membership / report density vs threshold grid; change the slice (universe not selected on breadth; concentrated managers).
- **Bulk 13F for reaction** — positions ≥6 weeks old plus archive assembly / per-CIK filings as they land for reaction; bulk for the cross-section.

### Macro panels and fills

- **Calendar-day grid mistaken for trading grid** — weekend rows carry Friday's value; 252 rows = 8 months / check row count vs calendar days; name windows in calendar days (365/90/30) or move to the trading grid.
- **Row count as release cadence** — forward fill makes every series look daily / count value changes per year (lower bound).
- **Non-causal fills** — interpolation, backward fill, centred MA, whole-sample seasonal adjustment read the future silently / forward fill, trailing stats, causal forecasts only.
- **FRED stamp read as knowledge date** — stamp = first day of reference period; CPI panel steps up ~7 weeks early / re-date: period end + agency lag; backward `join_asof` on `published_on`.
- **Lag measured from the stamp** — monthly series appears ~4 weeks early / count lags from period end; add period length separately.
- **Schedule bound at the early end** — a day early invalidates the backtest / set lags at the late end of the agency's range; prefer recorded vintage dates where an archive exists.
- **Ignoring revisions (vintage)** — database holds revised values that did not exist on trade date / compare against ALFRED first-release; count revision episodes and largest revision; snapshot vintages for feeds without an archive.
- **Rolling stat over forward-filled daily rows** — weights weeks by days occupied / compute on the releases, then as-of join.
- **Macro panel on the left of a price join** — invents weekend rows with null prices that a later ffill fills / prices LEFT, `how="left"`; count rows with no macro attached.
- **Derived column drift** — invisible until something built on it is wrong / recompute from inputs; compare to the metadata formula; trust the recomputation.
- **Regime thresholds that never or always fire** — useless and invisible from the definition / plot regime shares; every label reached, none covers the whole window.

### COT

- **Duplicate contract markets** — several CFTC markets per product code double-count / keep deepest open interest per date; check margin stability.
- **Report-date keying** — Tuesday positions published Friday after close / `available_from = report_date + 7d`; as-of join; accept ~1 week staleness.
- **Z-score window in days** — reports are the grid / 52 reports = 1 year.
- **Zero std in a z-score window** — zero would read as "exactly average" / return null when std == 0 (COT, Kalshi).
- **Self-referential change threshold** — rolling std including the current week raises its own bar / `rolling_std(26).shift(1)`.
- **Only three COT categories** — three do not sum to zero; other/non-reportables are large / measure the five-category identity; call "speculator movement" a partial view.
- **Reading asset-manager sign** — net long every week / use size/z-score, not sign.
- **Contrarian rule across products as diversification** — extremes may coincide / correlate z-scores across products on shared report dates.

### On-chain, overlapping returns, screening

- **TVL level as a signal** — tracks price by construction / growth or z-score.
- **Composition plotted against the selection** — four chains always fill the chart / shares against the published total with a residual bar.
- **Shifted trailing return as forward return** — coincides only when momentum window == horizon / price at both ends of the forward window.
- **Overlapping forward returns treated as independent** — ~30 views per window inflate t / HAC with `maxlags = horizon − 1`; report independent windows = n/horizon.
- **Regime table read as evidence** — three buckets always order; "neutral" carrying the extreme mean is partitioned noise / HAC-covariance bucket means; test contrasts, not means vs zero.
- **Warm-up rows pooled into "neutral"** — ~100 days of no measurement read as a measurement / regime = null until the window fills.
- **Duplicate last day from CoinGecko** — live snapshot doubles the last date / `unique(subset="timestamp", keep="last")`.
- **Price history bounds the study** — free tier = 365 days regardless of TVL history / state what binds; extend with exchange feeds (Ch2), not a cleverer signal.
- **Multiple comparisons in a screen** — largest of 12 exceeds largest of 1 under the null / report the count; fix the heatmap scale at ±2; call it screening.
- **Whole-sample correlation hiding sign changes** — one episode can produce it / rolling 180-day correlation; mean, range, share positive; "unproven".
- **Filtering history by level** — later drawdowns re-cross low levels → false gaps / filter by date (`MODERN_ERA_START`).
- **Restated feed with no vintages** — today's 2021 includes protocols nobody tracked in 2021 / hard gate: block; start daily snapshots; restrict claims to the snapshotted period.
- **Composite scoring of the four questions** — strong signal outvotes a failed gate / keep hard gates separate; decide from the failed gate.
- **MNPI confused with difficulty of access** — / record how obtained and whether the supplier was entitled.
- **Free data treated as costless** — integration + maintenance hours / compute cost-recovery and target bps at the fund's AUM and allocation.

### Prediction markets

- **Bid read as the market's probability** — lower bound short by the spread / note which price the feed carries; ladder checks are staleness diagnostics.
- **Zero-price bars in carry-forward feeds** — a pinned contract looks widest-ranging / null `close == 0 & high > 0`; forward fill within symbol.
- **Backward fill in repair** — puts tomorrow's price on today / forward only.
- **Range/volume features on non-traded bars** — range measures quote age or the repair; Polymarket `volume` is sampling frequency / set OHL = close on untraded bars; drop range; read provider docs.
- **Quote revisions counted as trades** — / report `days_the_price_moved` vs `days_traded`.
- **Features with few real values** — momentum non-zero on a handful of bars / count defined and non-zero values before fitting.
- **Near-settled contracts dominating a universe** — `extreme` flags every row / select open-question markets first.
- **Cross-venue comparison without an identical contract** — different questions about the same institution / compare only identical contracts; separate axes otherwise.

### Text extraction

- **Hardcoded 10-K item numbers on a 10-Q** — MD&A is Part I Item 2; returns nothing silently / pattern set by form; `after` marker for Part II.
- **First heading match** — ToC and cross-references match first / longest-span rule to the next item's heading.
- **Boundary overrun** — section swallows the next item / word count ≤ 50,000 and no next-item heading.
- **Flattening paragraphs** — destroys the change-detection unit irrecoverably / `get_text(separator="\n")`; cap blank lines at two; never join blocks with spaces.
- **Amendments in a consecutive pair** — 10-K/A covers the same period, often omits narrative / `amendments=False`.
- **Comparing a failed extraction** — empty old section scores zero similarity and marks every paragraph new / compare only `quality == "ok"` without `boundary_leak`.
- **Company + date as text key** — not unique with an amendment / key on accession number.

### Academic panel and research hygiene

- **IC significance assuming independent months** — serial correlation overstates t / `compute_ic_hac_stats(label_horizon=...)`; compare `t_stat` vs `naive_t_stat`.
- **Feature ranking over the full history** — reads validation/test years / ICs on the train split only.
- **Identifiers across splits** — no published mapping; offset by a million / never carry a position across a split boundary.
- **Pooled correlations without per-month normalization** — would measure level drift / pool only because each month is rank-normalized.
- **Unregularized linear model on 46 correlated features** — coefficients split arbitrarily within blocks / regularize or use tree ensembles (`chapters/11_ml_pipeline.md`, `chapters/12_gradient_boosting.md`).
- **Raw count histograms across unequal splits** — compares sizes not shapes / density; clip at ±60%; full range in a table.
- **Live queries in reproducible notebooks** — runs differ / `LIVE=False`; shipped snapshots; loaders raise `DataNotFoundError` naming the download command.

## Decision rules and defaults

| Rule | Default / threshold |
|---|---|
| As-of query | `filter(knowledge_time <= as_of)` → `sort([entity, valid_time, knowledge_time])` → last per entity; test with `AS_OF_DATE` just after a quarter end |
| Statutory deadlines | 10-Q ≤ 40 days (large filer); 10-K 1–3 months; Form 4 ≤ 2 business days; 13F ≤ 45 days after quarter end; XBRL mandatory large filers 2009, all 2011; ASC 606 effective 2018 |
| Tool choice | single company / statements / Form 4 & 13F → EdgarTools; bulk thousands of filings → SEC bulk archives; cross-sectional fundamentals (20+ stocks) → XBRL Frames API |
| Identifier trust order | CIK (or LEI) before ticker; CUSIP/ISIN/FIGI for instruments; never ticker as a durable key |
| Entity-resolution sequence | identifiers → dated alias table → fuzzy `token_set_ratio` ≥ 70 (threshold from a precision/recall sweep incl. non-matchable rows) → embeddings only to extend candidates → unresolved stays unresolved |
| Form 4 gate | code ∈ {P, S}; priced rows only; report unpriced separately; reconciliation asserts must pass first |
| 13F screens | one filing per CIK by latest `filing_date`; median implied price ∈ [1, 10,000] USD; drop `PutCall` rows for stock concentration; group on CUSIP; `TOP_N = 500` managers; `MIN_SHARED_MANAGERS = 5`, `CO_OWNERSHIP_UNIVERSE = 500`, density grid 5–350; decide the slice before the threshold |
| Macro release lags (days after period end) | claims 5 (exact); Fed H.4.1 2; employment report 7; CPI 18; industrial production 18; M2 28; PCE 31; GDP advance 31; daily market series 0; weekly stamp weekday = mode of change-weekdays |
| Calendar-day windows | 365 = year, 90 = quarter, 30 = month |
| Regime labels | yield curve <0/<0.5/<1.5/else = inverted/flat/normal/steep; VIX <15/<25/<35/else = calm/normal/unsettled/frightened (nb06 chart bands 20, 30); unrate <4.0/<6.0/else = tight/normal/slack; `get_regime(threshold=0.5)` |
| Fill policy | forward fill, trailing averages, causal forecasts only |
| Join direction | prices LEFT, macro/positioning RIGHT, `join_asof(strategy="backward")` on publication/availability date |
| COT | deepest open-interest market per date; z-score window 52 reports; extreme \|z\| > 2.0; change threshold = `rolling_std(26).shift(1)` of weekly changes; `available_from = report_date + 7d`; speculative column `lev_money_net` (financial) / `managed_money_net` (commodity) |
| Overlap correction | `maxlags = horizon − 1`; independent windows = n/horizon; `compute_ic_hac_stats(label_horizon=h)` sets lag ≥ h−1 (max with Newey-West auto `floor(4(T/100)^(2/9))`, capped at T//2 — from the library docstring, not the chapter text) |
| TVL features | growth 30d; z-score 90d; regime \|z\| > 1; forward 30d; horizons 7/14/30/60; rolling stability 180d; modern era from 2020-01-01; implausible move > 20%/day |
| Alt-data go/no-go | Legal gate (how obtained, MNPI, licence, jurisdiction) → block on failure; Data gate: vintages reconstructible? → block if restated with no archive (snapshot daily; revisit in a year); Signal: count of relationships, HAC t, rolling sign stability → "unproven" if short; Commercial: `cost_recovery_bps = annual_cost/(AUM×allocation)×10,000`, target = ×3; defaults 40 + 20 h/yr × $150, allocation 10% |
| Prediction markets | confirm which price the feed carries; repair zero-price artifacts forward only; OHL = close on untraded bars; `CONFIDENT_PROBABILITY = 0.2` → near-certain/extreme when p > 0.8 or p < 0.2; momentum 5d, volatility/z-score 10d; ladder monotone non-increasing in threshold (diagnostic); select open-question markets first; never compare `volume` across venues |
| Text extraction | pattern set by form; longest span; quality ok iff 100 ≤ words ≤ 50,000 and no next-item heading; compare only when both pass; `added_paragraphs(min_characters=200)`; key on accession; carry acceptance timestamp; drop tables for sentiment, keep for numeric reading |
| Academic panel | feature-selection statistics on train only; report HAC and naive t and the gap; redundancy threshold 0.5; no execution/cost claims |

## Code patterns and APIs

Repo loaders (`from data import ...`; all raise `DataNotFoundError` naming the download script when the parquet is missing):

| Loader | Returns |
|---|---|
| `load_firm_characteristics(split="all"\|"train"\|"valid"\|"test", include_macro=False)` | symbol, timestamp, 46 features, ret, split |
| `load_sec_xbrl_fundamentals(concepts=None, years=None, symbols=None, ciks=None)` | symbol, cik, entity_name, fiscal_quarter_end, announcement_date, accession, concepts |
| `load_13f_bulk_holdings("2024Q3")`; `load_13f_edges()` | bulk positions; canonical multi-quarter edges (`equities/positioning/13f/institution_stock_edges.parquet`) |
| `load_macro()`, `load_macro_metadata()`, `data.macro.loader.load_macro_initial_release()` | calendar-day panel; metadata; ALFRED first-release panel |
| `data.futures.loader.load_cot(products, start_date, end_date)`; `list_cot_products()` | `$ML4T_DATA_PATH/futures/positioning/cot/{PRODUCT}.parquet` (`diagonal_relaxed` concat) |
| `load_etfs(symbols=["SPY"], start_date=...)` | ETF prices |
| `load_defillama_chain_tvl(chain="total"\|"Ethereum"\|...)`; `load_coingecko_ohlcv("ethereum")` | timestamp, tvl_usd; timestamp, price_usd, volume_usd |
| `data.prediction_markets.loader.load_kalshi(symbols, start_date, end_date)`, `load_polymarket(...)` | timestamp, symbol, open, high, low, close, volume |

Utilities: `from utils import DATA_DIR`; `from utils.paths import get_output_dir` (`get_output_dir(4, "kalshi")`); `from utils.reproducibility import set_global_seeds`; `from utils.style import COLORS, ml4t_diverging, show_plotly_with_alt`.

Download scripts: `data/equities/firm_characteristics/download.py`; `data/equities/positioning/form4_download.py --ticker TSLA --count 20`; `data/equities/fundamentals/xbrl_download.py [--years] [--ciks] [--concepts]`; `data/equities/positioning/13f_download.py --mode bulk --quarters 2024Q3` (or `--mode per-cik`); `data/macro/download.py`, `data/macro/download_alfred.py`; `data/futures/positioning/cot_download.py --products ES,CL,GC`; `data/etfs/market/download.py --symbol SPY`; `data/crypto/onchain/download.py [--dataset defillama|coingecko]`; `data/prediction_markets/download.py`. Env: `EDGAR_IDENTITY` (nb02, nb14, `form4_download.py`), `FRED_API_KEY` (live FRED only).

ml4t libraries:
- `ml4t.diagnostic.metrics.cross_sectional_ic_series(predictions, returns, pred_col="prediction", ret_col="forward_return", date_col="date", entity_col=None, method="spearman", min_obs=10)` → [date_col, ic, n_obs] (null ic when undefined).
- `ml4t.diagnostic.metrics.compute_ic_hac_stats(ic_series, ic_col="ic", maxlags=None, label_horizon=None, kernel="bartlett", use_correction=True, allow_naive_fallback=False)` → mean_ic, hac_se, t_stat, p_value, n_periods, effective_lags, naive_se, naive_t_stat, used_naive_fallback, kernel, use_correction (warns if neither maxlags nor label_horizon given).
- `ml4t.data.cot.PRODUCT_MAPPINGS`, `COTConfig`, `COTFetcher` (`ml4t_data/ml4t/data/cot/fetcher.py`).
- `ml4t.data.macro.MacroDataManager(config: MacroConfig)` / `.from_config(path)`: `download_treasury_yields()`, `load_treasury_yields()`, `get_yield_curve_slope()`, `get_regime(threshold=0.5)`; `ml4t.data.providers.fred.FREDProvider`: `fetch_ohlcv(series_id, start=...)`, `fetch_series_metadata`, `fetch_multiple`, `close()`.
- `ml4t.data.providers.polymarket.PolymarketProvider().list_markets(active=True, closed=False, limit=200)`.

Third-party: `edgar` (EdgarTools: `Company, find, get_filings, set_identity`; filter `DeprecationWarning` for module `edgar`); `rapidfuzz.fuzz`, `rapidfuzz.process.extractOne`; `sentence_transformers.SentenceTransformer`; `statsmodels.api.OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": h-1})`, `.t_test(weights)`; `bs4.BeautifulSoup`; `xml.etree.ElementTree`.

```python
# Backward as-of join of lagged releases onto a daily/session grid (nb07, nb08)
result = result.join_asof(
    releases.sort("published_on"),
    left_on="timestamp", right_on="published_on", strategy="backward",
).drop("published_on")
```
```python
# Coalescing deterministic join in trust order (nb05)
result = result.join(lookup, on=identifier, how="left")
assert len(result) == before
result = result.with_columns(
    matched_ticker=pl.coalesce("matched_ticker", "_found"),
    match_method=pl.coalesce("match_method",
        pl.when(pl.col("_found").is_not_null()).then(pl.lit(identifier))),
).drop("_found")
```
```python
# Rolling z-score that is undefined, not zero, when the window never moved (nb08, nb12)
pl.when(std > 0).then((pl.col(c) - mean) / std).otherwise(None)
# Change threshold from strictly prior weeks
prior_change_std = pl.col(c).diff().rolling_std(26).shift(1)
```
```python
# Forward-only artifact repair on a carry-forward feed (nb12)
.with_columns(pl.when((pl.col("close") == 0.0) & (pl.col("high") > 0.0)).then(None)
              .otherwise(pl.col("close")).alias("close"))
.with_columns(pl.col("close").forward_fill().over("symbol"))
```
```python
# HAC regime means and contrasts (nb09)
fit = sm.OLS(y, indicators).fit(cov_type="HAC", cov_kwds={"maxlags": FORWARD_DAYS - 1})
test = fit.t_test(weights)  # weights = [1, -1, 0] etc.; ravel .effect/.sd/.tvalue/.pvalue
```
```python
# Ticker-shaped extraction into columns (nb12/13)
df.with_columns(parsed=pl.col("symbol").str.extract_groups(THRESHOLD_TICKER)).unnest("parsed").drop_nulls("meeting")
```

Run: `uv run python 04_fundamental_alternative_data/<notebook>.py`; tests `uv run pytest tests/test_chapter_notebooks.py -v -k "04_fundamental_alternative_data"`. Every notebook finishes well under a minute, peak memory under 3 GB.

## Evidence from the book

The digest shows code and narrative, not executed outputs; figures below are those stated in prose/alt-text and should be verified by running the notebooks.

- **CPZ panel (nb01):** ~2,000 stocks in a typical month; cross-section rises from ~400 (1967) to ~2,800 (mid-2000s), eases to ~1,900 (2016); later splits have wider, lower-peaked return distributions; every characteristic in [−0.5, 0.5], no missing values; BEME histogram flat; correlation heatmap shows square blocks along the diagonal; every single-characteristic IC is small, momentum and value positive, volatility and short-term reversal negative; HAC vs naive t-stats lie close to the diagonal and move both ways.
- **XBRL (nb04):** total assets reported in nearly every company-quarter; filing-lag histogram has a tall cluster near one month and a thin tail past a year; median is the routine case, mean far above it; `revenues` null for post-ASC-606 filers.
- **Form 4, TSLA (nb03):** most share volume is compensation codes; a single purchase bar exceeds all sale bars together; selling spread across many reporters.
- **Macro (nb06/07):** shipped CPI panel steps up about seven weeks before the release it reports; staleness largest for GDP, smallest for weekly claims; daily Treasury yields are revised "a handful of observations in each, by amounts small against the level and large against a day's move"; every regime label reached, none covers the whole window; unemployment change count in 2024 comes in under 12 releases because of rounding.
- **COT, ES (nb08):** asset managers net long every week; leveraged money and dealers usually short; dealers have the largest weekly change std; five-category sum is zero, three alone leave a residual; median attached-report age ≈ one week; deepest-market margin never narrows toward 1× (evidence, not proof).
- **TVL (nb09):** level correlation with ETH price high by construction; contraction-vs-expansion contrast does not support the hypothesis; "neutral" can carry the extreme mean; ~1 year of daily data = ~12 independent 30-day windows; the free price feed (365 days) binds, not TVL history.
- **Four-question evaluation (nb11):** 12 signal-horizon pairs, none reaches |t| = 2; rolling 180-day correlation changes sign within the year; audit: no missing days or values, implausible (>20%) moves almost all before 2020; legal clear; cost $9,000/yr with cost-recovery bps falling with AUM; verdict "Blocked: no vintages".
- **13F 2024Q3 (nb10):** millions of positions from ~7,000 managers; thousands of CUSIPs with multiple spellings; a handful of filings imply >$10,000/share and carry trillions (enough to rank a small trust above Vanguard); several hundred imply <$1 and carry almost nothing; co-ownership density ≈ 100% up to ~200 shared managers; median overlap coefficient high; top pairs are ordinary index constituents.
- **Kalshi (nb12):** five traded bars out of several hundred; prices move far more often than trades; paths are staircases; momentum non-zero on few bars, z-score undefined on many.
- **Polymarket (nb13):** most markets appear once, none more than twice; every row flagged `extreme`; a Bitcoin ladder exists with the same monotone shape as the Fed ladder; `volume` total not comparable to Kalshi's.
- **Text (nb14):** risk factors several times longer than Business or MD&A; every heading pattern matches in more than one place; high vocabulary overlap with many changed paragraphs is the normal result.
- **Entity resolution (nb05):** chosen fuzzy threshold 70 from the sweep on the 12-row labelled set; embeddings recover abbreviations/paraphrases but not renames or subsidiaries.

Reader uncertainty (keep these marks): README places nb10 under 4.4 while its header says 4.1; `SecurityMaster` has no alias validity dates though the text requires them (production shape not shown); the exact series set in `fred_macro.parquet` is not enumerated beyond the 12 lagged + 11 daily; the book's treatment of corporate actions, taxonomy drift beyond ASC 606, ESG divergence (Berg et al.), Google Trends, satellite/scraped data and the "seven sins" is referenced but not covered by any notebook (inference: book text only); whether Ch8 reuses `query_fundamentals_as_of` / `point_in_time_panel` verbatim is not visible; CPZ split boundaries are indicative (read from file).

## Related references

- `chapters/02_financial_data_universe.md` — exchange price feeds that extend the crypto price history bounding the TVL study; data universe and PIT foundations.
- `chapters/03_market_microstructure.md` — trading-session grid vs calendar-day grid; acceptance-time vs next-session tradeability.
- `chapters/08_financial_features.md` — consumes the as-of fundamentals panel (`04_fundamentals_macro_calendar`) and COT z-scores (`03_structural_cross_instrument_features`).
- `chapters/10_text_feature_engineering.md` — `09_filing_text_signals` builds on the section corpus written by nb14.
- `chapters/07_defining_the_learning_task.md` — overlapping forward-return labels and horizon-aware HAC corrections.
- `chapters/11_ml_pipeline.md`, `chapters/12_gradient_boosting.md` — regularized / tree models for the 46 correlated characteristics; train-only feature selection.
- `chapters/14_latent_factors.md` — SDF GAN that uses the 178-series macro companion of the CPZ panel.
- `chapters/22_rag_financial_research.md`, `chapters/23_knowledge_graphs.md` — `07_institutional_holdings_graph` and graph models consume `load_13f_edges`.
- `chapters/26_mlops_governance.md` — daily vintage snapshotting and data lineage as governance.
- `case_studies/us_firm_characteristics.md` — the CPZ panel case study (`05_linear`, `06_gbm`, `08_latent_factors`, `10_model_analysis`, `11_backtest`).
- `case_studies/crypto_perps_funding.md`, `case_studies/cme_futures.md`, `case_studies/etfs.md` — crypto, futures (COT) and SPY contexts used in this chapter's joins.
- `libraries/ml4t_data.md` — `ml4t.data.cot`, `ml4t.data.macro`, `ml4t.data.providers.fred`, `ml4t.data.providers.polymarket`.
- `libraries/ml4t_diagnostic.md` — `cross_sectional_ic_series`, `compute_ic_hac_stats`.
- `guardrails.md`, `decision_rules.md`, `workflow.md`, `glossary.md`, `companion_repo.md` — cross-cutting PIT/leakage rules and repo conventions.
- Further reading: Croushore 2008 (real-time data / vintages); Ekster & Kolm 2020; Green & Zhang 2024 (alternative data in investment management); Chen, Pelger & Zhu 2021 (panel source); Kelly, Pruitt & Su 2019 and Kozak, Nagel & Santosh 2020 (rank normalization); McCracken & Ng 2016 (FRED-MD); Welch & Goyal 2007; Alexander & Dakos 2019 and Harvey et al. 2022 (crypto data); Baur & Smales 2022 (bitcoin futures smart money); Ng et al. 2025 (prediction-market price discovery); Berg et al. 2022 (ESG divergence); Luo et al. 2014 (Seven Sins); McLean & Pontiff 2016; Hong et al. 2000; Daniel & Titman 2006; Tetlock 2005, 2014; Preis et al. 2013; Joubert et al. 2024; Chi et al. 2024; Lehar & Parlour 2021; Kertkeidkachorn et al. 2023 (FinKG); Freyberger, Neuhierl & Weber 2020.

## Glossary

- **Point-in-time (PIT)** — using only information public on the simulated date. **Bitemporal** — a row carrying valid time (period described) and knowledge time (when it became public).
- **As-of query** — filter knowledge time ≤ date, rank on valid time, take latest per entity. **Backward as-of join** — for each left row, the most recent right row with key ≤ left key (`join_asof(strategy="backward")`).
- **Vintage** — the value of a series as it stood on a past date (ALFRED archives FRED vintages). **Release lag** — days from reference-period end to publication. **Schedule bound** — assumed publication date at the late end of the agency's schedule, used when no timestamp is recorded.
- **Restatement / amendment** — later filing revising an earlier period (10-K/A, 13F amendments). **Taxonomy drift** — concept names change (ASC 606 revenue concept).
- **CIK** — SEC Central Index Key, permanent filer id. **LEI / FIGI / CUSIP / ISIN** — legal-entity / instrument / security identifiers. **Accession number** — unique, permanent SEC filing id; leading block = filing agent. **Acceptance datetime** — when EDGAR accepted a filing; after-close = tradeable next session.
- **XBRL / Frames API** — machine-readable us-gaap tagging; endpoint returning one concept for all filers for one calendar period. **10-K / 10-Q / 8-K** — annual / quarterly / material-event reports. **Form 3 / 4 / 5** — insider initial holding / trade within 2 business days / year-end exempt transactions. **Transaction code** — P purchase, S sale, A grant, M/X exercise, G gift, D disposition to issuer, F tax withholding, C conversion, I discretionary.
- **Form 13F(-HR)** — quarterly long-equity holdings of managers > $100M, due 45 days after quarter end; `PutCall` marks options. **Reported 13F value** — disclosed long US-equity/option value, not AUM. **Breadth / size** — number of managers holding / aggregate value. **Overlap coefficient** — shared holders / min(breadth_a, breadth_b).
- **Security master** — canonical entity records with identifiers and dated aliases. **Deterministic vs probabilistic matching** — equal identifiers vs similarity above a threshold. **Precision / recall / F1** — accepted matches that are right / true matches made / harmonic mean. **token_set_ratio** — rapidfuzz scorer ignoring word order and extra words. **Sentence embedding / cosine similarity** — vector text representation; dot product of unit vectors.
- **Information coefficient (IC)** — per-period rank correlation between signal and following return. **Fama-MacBeth template** — average per-period cross-sectional statistics over time. **Newey-West / HAC** — standard errors robust to heteroskedasticity and autocorrelation. **Cross-sectional rank normalization** — within-period ranks mapped to [−0.5, 0.5]. **Excess return** — net of the risk-free rate.
- **Forward return / overlapping windows** — return over a window after the signal date; consecutive forward returns share days, independent windows ≈ n/horizon. **Forward fill** — carry last released value forward (causal). **Calendar-day vs trading grid** — rows for every date vs sessions only. **Cadence / release clock** — publication frequency, recovered from value changes.
- **Yield curve slope / inversion** — 10Y − 2Y; negative = inverted (preceded every US recession since 1970). **VIX** — 30-day implied S&P 500 volatility, annualized.
- **COT** — weekly CFTC positions by trader category (TFF vs Disaggregated). **Open interest** — contracts outstanding. **Net position** — long minus short. **Contract market** — a CFTC-reported market; several map to one product code. **Contrarian / change signal** — fade an extreme z-score / react to a large weekly move.
- **TVL** — USD value of assets deposited in a chain's DeFi protocols; a stock, price-denominated. **Chain / protocol TVL** — per blockchain / per application.
- **Hard gate** — an evaluation question that blocks integration alone (legal; unreconstructible history). **MNPI** — information supplied by someone not entitled to supply it. **Cost recovery / target bps** — return that pays the carrying cost / that times the required multiple.
- **Prediction market / binary event contract** — pays $1 if the event occurs; price = probability. **yes_bid** — highest standing YES bid (lower bound). **Threshold ladder** — contracts on one event at different thresholds = implied survival function. **Carry-forward feed** — bars repeating the last quote when nothing traded. **Conviction / extreme / near_certain** — |p − 0.5| / p outside [0.2, 0.8]. **USDC / Polygon** — Polymarket's settlement stablecoin / blockchain.
- **Item 1 / 1A / 7 / 7A / 8** — 10-K Business / Risk Factors / MD&A / market-risk disclosures / financial statements. **Longest-span rule** — pick the heading match farthest from the next item heading. **Boundary leak** — extracted section containing the next item's heading. **Jaccard similarity** — |A∩B| / |A∪B| over vocabulary sets. **Added paragraphs** — paragraphs in the newer filing not verbatim in the older.
