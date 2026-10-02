# Chapter 23: Knowledge Graphs

> Chapter 23 ("Knowledge Graphs for Financial AI") equips you to decide when a financial question needs explicit relational structure, build a compact typed knowledge graph from SEC filings with LLM extraction under identity / schema / provenance contracts, answer questions with deterministic Graph RAG (fixed Cypher templates plus a read-only validator), turn graph topology, 13F crowding and correlation networks into leakage-aware tabular features and portfolio weights, and audit all of it under a three-timestamp model (event, disclosure, extraction). Its position: graph building is a governance problem before it is an NLP problem; graphs earn their overhead only for multi-hop, structurally crowded or temporally evolving questions; every measurement must be able to fail (no oracle recall, no validator pass rates over inputs written to pass); and a snapshot manifest records identity, not sufficiency or tradability. Notes were extracted from the companion repo (`23_knowledge_graphs/`), not the book PDF; section-level narrative for 23.1, 23.3, 23.5 and 23.7 is known only from the README and notebook headers.

## When to use this reference

- Building a supplier / customer / competitor graph from 10-K text, or an ownership graph from 13F filings, in Neo4j or networkx
- Running, caching or auditing LLM relationship / event extraction (Qwen2.5-7B-Instruct, CheckRules, run ids, cache sidecars)
- Resolving company names across filings into one node per entity (canonical keys, filer roster, aliases, subsidiaries)
- Answering questions over a graph with Cypher: question routing, read-only validation, cutoff dates, row limits, timeouts
- Comparing graph retrieval with vector retrieval and reporting the benchmark without an oracle arm masquerading as a result
- Computing graph-derived features (PageRank, betweenness, clustering, supplier HHI-style dependency, crowding, Jaccard co-ownership, cross-graph products) for Ch12 / Ch14 models
- Point-in-time handling of 13F holdings: `report_date` vs `available_from` / `filing_date`, complete periods, option rows, panel denominators
- Checking temporal integrity of any extracted edge set: event vs disclosure vs extraction time, snapshot manifests, extraction-time leakage
- Deciding whether GNN / GAT embeddings add anything over tabular features (paired held-out-stock ablation)
- Building correlation networks, MSTs, centrality-based (network-diversified) portfolio weights and a stylized contagion simulation

## Core ideas (the why)

- **Vector retrieval works on isolated chunks.** Ownership chains, supplier networks, contagion paths and the timing of relationship changes need explicit structure: entities as nodes, typed relationships as edges, provenance as properties, something to traverse, audit and reuse.
- **Governance before NLP.** Three contracts make a graph replayable: stable entity identity (CIK for institutions, CUSIP for securities, never names when a key exists), a finite relationship vocabulary, and edge-level provenance (source filing accession, dates, extractor model + revision, run id).
- **Graphs earn their overhead only when the question is multi-hop, crowded or temporally evolving.** Every count in 05's summary table "is reachable from the flat 13F table with a groupby"; what the graph buys is multi-hop traversal (one Cypher pattern instead of a chain of self-joins).
- **Graph RAG is deterministic relational retrieval.** The database does the relational logic; the LLM only verbalizes returned rows. In 03 nothing generates Cypher: the question selects one of three hand-written templates. "The templates, not the validator, are what makes this pipeline safe"; calling it text-to-Cypher "oversold it".
- **A measurement must be able to fail.** Validator pass rates over inputs written to pass (3/3, 3/3, 3/3) measure nothing; an oracle arm has recall 1 by construction; a leakage test that re-applies the filter that built the frame returns zero on every input. Replace each with a control that can come out below one.
- **Three timestamps answer three questions.** Event date (when it happened; cutting on it = lookahead), disclosure date (when a reader could know; cutting on it = point-in-time), extraction date (when the pipeline produced the edge; whether the graph itself existed yet). The third is the leakage the usual disclosure filter cannot see.
- **Two dates on every 13F edge.** `report_date` (quarter-end the position is as of; for quarter-over-quarter comparison) and `available_from` / `filing_date` (when it became public, roughly six weeks later; for point-in-time queries). Using either for the other question is "a look-ahead in one direction and a misaligned quarter in the other".
- **Extracted structure reflects what filings name, not the economy.** Almost every 10-K supplier is named by exactly one filer; shared-supplier degree is an exposure-concentration diagnostic for a reviewed graph, not a disruption probability or causal channel.
- **Schema validity is not semantic truth.** LLM extraction yields candidate edges; precision of the supply-chain triples "is not established here"; confirming an 8-K event is supported by its filing "requires a separate labeled audit".
- **Entity resolution runs on the corpus, not a lookup table.** A canonical key plus the roster of filers the corpus already knows does the work; alias maps rewrite almost nothing when the prompt asks for short forms. String rules cannot fold a subsidiary (KAYAK) into its registrant (Booking Holdings); that needs a corporate hierarchy.
- **Graph structure is information attributes miss, but cross-graph products are candidates, not estimates.** PageRank, betweenness and clustering capture structural importance and bottleneck position; interaction terms of supply topology and ownership are "transparent candidate proxies for downstream testing, not measured causal or predictive risk estimates".
- **Present is not complete (13F).** Filings arrive over a window; a partially filed quarter admitted as-is makes non-filers read as ownership exits. Admit a period only when every institution in the panel has filed.
- **A consumer verifies its input rather than restating it.** 09 checks the Neo4j graph against the identity 02 recorded and that record against the extraction-cache sidecar committed in the repo; only the cache hash ties the graph to something outside the database.
- **Correlation networks complement explicit KGs.** `d = sqrt(2(1 - rho))` is a metric on [0, 2], so the MST is well defined; centrality on the estimated MST describes correlation structure, not causal influence or systemic importance. Equal weight is the ceiling on effective names, so the fair comparator for a network rule is inverse volatility.
- **GNN embeddings are an ablation question, not a general claim.** Train the representation (random projections are not embeddings), fit preprocessing inside the fold, use a pre-target universe, report the spread not the mean, interpret narrowly.
- **A snapshot manifest records identity, not sufficiency.** Hashes and extractor version make a state reproducible; they do not make it tradable.

## Method recipes (the how)

### 0. Decide whether a graph is warranted (23.1)

| Question shape | Use |
|---|---|
| Single-entity lookup, aggregates, counts | Tabular groupby on the flat artifact |
| Semantic "what does the filing say about X" | Vector RAG (Ch22) |
| Multi-hop (2+ joins): "who holds the suppliers of X", co-owners of two issuers | Explicit KG + Cypher |
| Crowding / structural: shared neighbors, co-ownership, bridge nodes | KG projection or correlation network |
| Temporally evolving relationships: when did an edge appear / become public | Temporal KG with three timestamps |

### 1. Filing corpus and coverage (`23_knowledge_graphs/01_sp100_sec_download`)

1. Download: `uv run python data/equities/fundamentals/filings_download.py --form 10-K --universe sp100 --years 2020-2025` (same with `--form 8-K`).
2. Load staged Parquet with `load_sec_filings`; call `validate_filings(filings: pl.DataFrame, expected_form: str)` (required schema, unique symbol-accession keys, consistent form labels, exact text-length metadata). It fails on contract violations.
3. Coverage matrix: one cell per company-year; select pivot columns by name in sorted year order (pivot order is arbitrary after `unique()`); sort rows by coverage so gaps surface at the top. Carry gaps downstream, never impute a filing.
4. Text length is a pipeline diagnostic (10-K spikes come from the download's fixed extraction windows, see Ch22 `01_sec_filing_pipeline`; the 8-K rule keeps the opening), not informativeness, and not comparable across forms.

### 2. LLM triple extraction for the supply-chain KG (`02_supply_chain_kg_construction`)

| Config | Default | Note |
|---|---|---|
| `MODEL_NAME` | `"Qwen/Qwen2.5-7B-Instruct"` | ~14 GB VRAM; README mentions a vLLM-compatible endpoint alternative |
| `ENABLE_THINKING` | `False` | |
| `MAX_NEW_TOKENS` | 512 | |
| `SEED` | 42 | `set_global_seeds(SEED)` |
| `LLM_BATCH_SIZE` | 2 | leaves headroom on a 24 GB GPU |
| `MAX_COMPANIES` | 0 (= all) | |
| `RERUN_EXTRACTION` | `False` | load cache; `True` needs image `ml4t-gpu`; CPU regeneration unsupported |
| `RELATIONSHIP_TYPES` | `["HAS_SUPPLIER", "COMPETES_WITH", "HAS_CUSTOMER"]` | finite vocabulary |

Pipeline: `build_batch_prompt(text, company_name)` (chat-formatted; model told to repeat the filer's name as subject and emit short forms like `TSMC`) -> `extract_relationships_batch(texts, company_names) -> list[list[Triple]]` (one batched forward pass) -> `_parse_json_triples(response, company_name)` (finds the JSON array amid preamble / trailing text; keeps only valid predicates) -> `batched(iterable, n)`. `Triple` dataclass (subject, predicate, object; `to_dict()` for UNWIND). `run_full_extraction(filing_records) -> (triples, seconds)`.

Cache contract: `output/supply_chain_cache/extracted_triples.parquet` + `extracted_triples.meta.json`; `EXPECTED_CACHE_COLUMNS=("subject","predicate","object")`, `CACHE_ROWS_MIN=100`, `CACHE_ROWS_MAX=50_000`; sidecar pins content hash, schema, row count, extractor identity; `_validate_cache(parquet_path, meta_path)` recomputes the hash and fails on drift; `_write_cache_meta` only when regeneration is explicit.

### 3. Entity resolution (02)

Apply in this order:
1. `normalize_entity_text(name)`: collapse whitespace, strip trailing punctuation.
2. `entity_key(name)`: strip case, punctuation, corporate suffixes (`CORPORATE_SUFFIXES` frozenset), then SORT surviving tokens so word order is removed (merges `SCHWAB CHARLES CORP` with `The Charles Schwab Corporation`).
3. `FILER_ROSTER` lookup (dict keyed on name, built from the corpus; Alphabet files under two symbols with one name): a key match takes the filer's official name. This folds a company named as a competitor onto the company that filed its own 10-K.
4. `ENTITY_ALIASES` / `ALIAS_BY_KEY` for non-filers (e.g. TSMC).
5. `GENERIC_ENTITY_KEYS` + `is_actionable_entity(name, predicate)`: reject category phrases ("suppliers", "customers"), matched on the canonical key so case / hyphen variants cannot escape.
6. `build_canonical_names(triples) -> (canonical_map, alias_counts)`; `resolve_entity(name, canonical)`. Canonical form = filer spelling if matched; else most frequent surface form; all-caps forms lose ties (`Apple` beats `APPLE`).
7. Print the alias / rewrite table; assert Neo4j node count == number of distinct resolved names. Unresolvable names (subsidiaries, brands) stay as their own nodes; state the inflated company count.

### 4. Neo4j batch loading, reset and graph identity (02, 05, 08)

- Infrastructure: `docker compose --profile kg up -d neo4j`, then `docker compose run --rm ml4t python 23_knowledge_graphs/<nb>.py`. Env: `NEO4J_URI=bolt://localhost:7687`, `NEO4J_USER=neo4j`, `NEO4J_PASSWORD=password`. Driver: `GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))`.
- UNWIND expands a list parameter to rows so one Cypher statement creates hundreds of nodes / edges per transaction: `run_unwind_batch(session, batch, query) -> int`; `UNWIND_TEMPLATES[predicate]`; `_load_predicate(session, triples, predicate, batch_size)`; `load_to_neo4j_batch(triples, batch_size=1000) -> int`.
- Every template `MERGE`s on `:Entity {name}` and adds the role (Company / Supplier / Customer) as a second label: one name, one node.
- `RESET_STATEMENTS` (02): adopt existing `:Company` nodes into `:Entity`; delete only this notebook's supply-chain relationship types; delete only `:Entity` nodes with no remaining relationships. Never `DETACH DELETE` (08 attaches APPOINTED / ACQUIRED edges to `:Company` and declares `Company.name` unique).
- Identity: `read_graph_identity(session) -> (hash, per_class_counts, n)` hashes supply-chain edges as Neo4j returns them from `RELATIONSHIP_QUERY`; stored under `SNAPSHOT_NAME="ch23_supply_chain"`. 09 repeats the query text and recomputes, so drift in either copy fails the comparison on purpose.
- 05: `ensure_neo4j_constraints` (uniqueness on institution CIK and stock CUSIP), `clear_holdings_subgraph` (only the producer's labels), `build_graph_snapshot(holding_rows)` (source bytes, cohort policy, report periods, graph counts), `read_graph_counts`, `load_13f_to_neo4j() -> dict`.
- 08: `load_event_batches(session, events, batch_size, query, mapper)` fails on any write error; per-relation templates `APPOINTMENT/ACQUISITION/ANNOUNCEMENT/VALUATION_LOAD_QUERY` in `EVENT_LOAD_SPECS`; `_base_event_row(event)`; `load_events_to_neo4j(events, batch_size=500)` is idempotent (stable event ids + `RUN_ID`) and binds a `GraphSnapshot {snapshot_kind: '8k_events'}`; `graph_counts(session)` filters on the run id.
- All notebooks share ONE Neo4j database: 02 and 08 both touch `:Company`; 05 and 08 both write `GraphSnapshot` nodes. Always select a snapshot by `snapshot_kind` (name the producer).

### 5. 13F ownership property graph (`05_institutional_holdings_kg`)

| Element | Key / properties |
|---|---|
| `Institution` | key CIK; `equity_13f_value` = total long US equity value in the formation quarter (not "aum") |
| `Stock` | key CUSIP; `label` = shortened issuer name for display (NOT a ticker; the artifact carries none) |
| `Sector` | key name from `classify_sector` name screen over `SECTOR_TERMS`, else "Other" |
| `HOLDS` | keyed `(institution, stock, report_date)`; `shares`, `value`, `report_date`, `available_from` |
| `IN_SECTOR` | stock -> sector |

Steps: `report_period_calendar(holdings_df)` (one row per report period with filing window and public date = latest filing date in the period) -> `split_derivatives(holdings_df)` (exclude option rows from ownership; print the excluded share) -> `select_real_13f_universe(holdings_df, max_institutions, max_stocks)` (formation cohort ranked once at the FIRST report period) -> `load_real_13f_data(max_institutions=20, max_stocks=100)` (executed cohort: 10 managers x 50 names) -> `build_holding_payloads` (aggregate each institution-stock position within a period) -> `build_real_13f_payloads` (latest period for the in-memory teaching graph; every period for Neo4j) -> `load_13f_to_neo4j()`.

In-memory API: `Node`, `Edge`, `InMemoryGraph` (`add_node`, `add_edge`, `get_node`, `get_outgoing_edges`, `get_incoming_edges` by edge type). Co-ownership `J(s_i, s_j) = |Holders(s_i) ∩ Holders(s_j)| / |Holders(s_i) ∪ Holders(s_j)|` via `jaccard_similarity(set_a, set_b)` projects the bipartite graph to stock-stock. Crowding query: `WHERE holder_count > 2` (print how much it selects).

### 6. Graph RAG with fixed templates and a read-only validator (`03_graph_rag_qa`, 23.3)

Five-stage architecture (README): relational logic delegated to the database, language generation to the LLM. Safety sequence:

1. **Route**: `route_question(question) -> (cypher, params, kind)` matches the WHOLE normalized question (`lower().strip().rstrip("?")`) against `QUESTION_SPECS`; unknown question raises `ValueError`. The entity is never parsed out of the question.
2. **Validate**: `strip_noise(cypher)` removes comments and string literals, then `validate_cypher(cypher) -> (ok, issues)` checks in order: (a) `BLOCKED_KEYWORDS` = {CALL, CREATE, DELETE, DETACH, DROP, LOAD CSV, MERGE, REMOVE, SET} on word boundaries; (b) labels / relationships / properties subset of `GRAPH_SCHEMA` (labels {Institution, Stock, Sector}; relationships {HOLDS, IN_SECTOR}; properties {available_from, cik, cusip, equity_13f_value, issuer, label, name, report_date, sector, shares, value}); (c) every node pattern carries a label or re-binds an already-labelled variable; (d) a `LIMIT` exists and `max(limits) <= ROW_LIMIT` (25); (e) a real comparison `(available_from|report_date)\s*<=?\s*\$cutoff_date`.
3. **Limit**: `enforce_query_limits(cypher)` appends `LIMIT` when absent (cannot lower one).
4. **Execute**: `execute_read_only(cypher, params)` with `TIMEOUT_SECONDS=2.0`; `CUTOFF_DATE` read from 05's `GraphSnapshot`, never hard-coded.
5. **Synthesize**: `synthesize_answer(kind, rows)` restates only returned values.

Template shape: `WHERE h.available_from <= $cutoff_date ... ORDER BY h.report_date DESC` then `collect(h)[0]` keeps the latest report period per institution-stock pair.

Controls to keep in CI: (i) negative cutoff control: `periods_public_at(cutoff)`; re-run a question at an earlier cutoff and assert no future rows and never more rows than at the as-of date; (ii) hostile control set: six queries that must be refused (one per check) plus two that must be accepted; accepted ones are never sent to the database, the decision is the control. Count refusals and acceptances separately.

### 7. Graph vs vector retrieval benchmark (`04_rag_comparison_benchmark`)

- Config: `MAX_STOCKS=250`, `TOP_K_VECTOR=10`, `EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"`, `BENCHMARK_CUTOFF="2026-02-17"`. Needs `data/equities/positioning/13f_download.py` artifacts; no Neo4j.
- Point-in-time snapshot: collapse eligible filings to the latest disclosed row per institution-security pair at the cutoff; rank the universe only AFTER collapsing. `value_thousands` is a legacy field name; post-2022 values are dollars (scale factor 1).
- `QueryKind` enum; `BenchmarkQuestion` carries expected support doc ids; generators `make_holders_question(issuer)`, `make_holdings_question(institution)`, `make_coowners_question(issuer_1, issuer_2)`.
- `graph_retrieve(question)` = deterministic structured lookup using the predicates that defined the gold set (oracle, recall 1 by construction); `vector_retrieve(question, top_k=TOP_K_VECTOR)` by embedding similarity over prose rows.
- `graph_token_budget(doc_ids)` / `vector_token_budget(doc_ids)` share one rough words-per-token multiplier; neither is a tokenizer count.
- Reading rule: draw the oracle as a reference line; report only the vector column as the measurement; assert the oracle identity so breakage stops the notebook.

### 8. 8-K event extraction with CheckRules (`08_8k_event_extraction`)

| Config | Default |
|---|---|
| `MAX_FILINGS` | 50 |
| `MODEL_NAME` / `MODEL_REVISION` | `"Qwen/Qwen2.5-7B-Instruct"` / `"a09a35458c702b33eeacc393d103063234e8bc28"` |
| `ENABLE_THINKING`, `DO_SAMPLE` | `False`, `False` (greedy) |
| `MAX_NEW_TOKENS`, `BATCH_SIZE`, `SEED` | 512, 4 (longer prompts than 02), 42 |
| `EVENT_TYPES` | 1.01 Material Agreement, 2.01 Acquisition/Disposition, 5.02 Executive Change, 7.01 Regulation FD Disclosure, 8.01 Other Events |
| `RELATIONSHIP_TYPES` | ANNOUNCED, APPOINTED, ACQUIRED, VALUED_AT |
| `META_TYPES` | EXECUTIVE_CHANGE, ACQUISITION, STRATEGIC_PARTNERSHIP, DISCLOSURE |
| `MAX_ENTITY_WORDS` | 5 |

Steps:
1. `detect_item_numbers(text) -> list[str]` returns ALL items in schema order; `describe_items(items)` renders "not identified" when none (never default to 8.01).
2. Run identity: `HASHED_FILING_FIELDS=("accession_no","cik","company_name","filing_date","symbol","text")`; `frame_sha256(frame)` order-independent over the frame actually read; `SELECTED_ACCESSIONS_SHA256`; `EXTRACTION_TIME` (UTC ISO) / `EXTRACTION_DATE`; `RUN_ID = sha256(json.dumps(run_identity, sort_keys=True))[:24]`. Test: mutate each hashed field on one row and require the digest to move.
3. Prompt = `EVENT_PROMPT_HEADER` + `EVENT_PROMPT_RULES` ("RULES (CheckRules - follow exactly)"); `build_event_prompt(filing, max_chars=None)` -> `extract_events_llm(filing)` / `extract_events_batch(filings) -> list[ExtractionResult]` -> `extract_json_events(response)` (raises on invalid JSON) -> `parse_event_response(response, filing)`. `ExtractionResult` conserves one outcome per filing (parsed / zero events / parse failure).
4. `EventQuadruple(event_id, subject, relation, object, timestamp, public_date, extraction_date, meta_entity, source_cik, source_accession_no)` = FinDKG quadruple + provenance.
5. `check_rules(event) -> CheckRulesResult(is_valid, violations, suggestions)`: subject / object non-empty, <= 5 words, all-caps, no leading `THE ` / `A ` / `AN `; relation in `VALID_RELATIONS`; `meta_entity` in `META_TYPES`; `timestamp` parses with `date.fromisoformat`.
6. `normalize_event` only uppercases and strips leading articles; `normalize_and_validate(events) -> (accepted, unresolved, stats)` normalizes ONCE; still-failing events go to the unresolved collection for manual review. Conserve counts.
7. `load_events_to_neo4j(events, batch_size=500)` under `RUN_ID`.

### 9. Temporal snapshots with three timestamps (`07_dynamic_kg_temporal`, 23.6)

- Config: `WINDOW_DAYS=90`, `CUTOFF_LAG_DAYS=60` (how the cutoff derives from the latest disclosure is inferred, not read), `SNAPSHOT_KIND="8k_events"`; `TEMPORAL_EVENT_QUERY` matches the snapshot by kind and `r.run_id = snapshot.run_id`.
- `to_iso_date(value)` converts Neo4j temporals before Polars' strict date schema; `load_temporal_events() -> pl.DataFrame` with event date, public (filing) date, extraction date, source and model identity per edge.
- Lag = disclosure date - event date, SIGNED; negative when an 8-K announces an appointment effective later (`post_disclosure_effective_date` counts those); `LAG_BUCKETS` for the histogram.
- `visible_at_cutoff(df, cutoff_date)` filters `public_time <= cutoff`; `build_snapshots(df, window_days)`; `graph_growth(df, window_days)` compares CUMULATIVE states at window ends (additions vary; removals asserted zero because disclosures are not retracted).
- Leakage checks: `disclosure_filter_postcondition(df, cutoff)` (zero by construction, name it a postcondition) and `edges_extracted_after(df, cutoff)` (visible edges with `extraction_time > cutoff`, the orthogonal test).
- `write_snapshot_manifest(out_dir, cutoff_date, window_days, cutoff_lag_days, source_graph, parquet_frames) -> Path` to `get_output_dir(23, "dynamic_kg_temporal")`: cutoff, window, upstream extractor identity, per-parquet row counts and SHA-256.

### 10. Graph-derived ML features (`09_knowledge_graph_features`, 23.4)

Config: `N_COMPANIES=0` (all companies with >= 1 outgoing edge), `CUTOFF_DATE=""` (resolves to `LATEST_AVAILABLE` = max `available_from` over complete periods), `SUPPLY_SNAPSHOT_NAME="ch23_supply_chain"`, `SUPPLY_CACHE_META_PATH=get_chapter_dir(23)/"output"/"supply_chain_cache"/"extracted_triples.meta.json"`, `HEATMAP_ROWS=20`.

| Stage | Functions | Output / rule |
|---|---|---|
| 1 Snapshot guard | `read_supply_snapshot`, `validate_supply_snapshot(companies, relationships, snapshot) -> str`, `fetch_supply_relationships() -> (companies, relationships, snapshot, hash)` | recompute sha256 of sorted relationship lines, per-class edge counts, company count; compare cache hash to committed sidecar; normalize whitespace before hashing; read-only |
| 2 Graph | `build_supply_graph(relationships, focal_companies) -> (nx.MultiDiGraph, set)`; `COMPANY_QUERY` (`MATCH (c:Company) WHERE EXISTS { MATCH (c)-[:HAS_SUPPLIER\|COMPETES_WITH\|HAS_CUSTOMER]->() }`), `RELATIONSHIP_QUERY` (union over three edge classes) | focal = companies with >= 1 outgoing relationship |
| 3 Periods | `report_period_calendar(holdings)`, `complete_report_periods(periods, panel_size)`, `public_report_periods(periods, cutoff_date)` | `PANEL_SIZE = holdings["cik"].n_unique()` over EVERY filing (options-only included); complete iff `institutions == PANEL_SIZE`; public iff `available_from <= cutoff`; assert `COMPLETE_PERIODS` non-empty and `len(PUBLIC_PERIODS) >= 2` |
| 4 Matching | `normalize_name(value)` (case, punctuation, CORPORATION->CORP, INCORPORATED->INC), `resolve_company_match(entity, stock_records, by_norm) -> (match\|None, decision, n_candidates)`, `build_company_mapping(companies, holdings, cutoff_date)` | exact normalized name; else prefix match only when every candidate shares one CUSIP; else null; candidates restricted to issuers first seen on or before the cutoff; audit to `company_mapping.parquet` |
| 5 Topology | `compute_topology_features(graph, companies)` | collapse multigraph to simple digraph; `pagerank = nx.pagerank(simple.reverse(copy=True), alpha=0.85)` (reversed so flow points toward depended-on nodes, inference); `betweenness = nx.betweenness_centrality(undirected)`; `clustering = nx.clustering(undirected)`; `in_degree`, `out_degree`; missing companies get 0 |
| 5 Supply | `compute_supply_chain_features(graph, companies)` via `_outgoing_targets(graph, company, edge_type)` | `n_suppliers`, `n_competitors`, `n_customers`; per supplier `count` = covered companies listing it; `shared_supplier_count = #(count > 1)`; `single_source_count = #(count <= 1)`; `supplier_dependency_score = mean(1/count)` (0.0 if none); `supplier_overlap_ratio = shared_supplier_count / n_suppliers` (0.0 if none) |
| 6 Holdings | `build_holdings_snapshots(holdings, public_periods) -> (latest, prior)`, `compute_stock_ownership_stats(latest, prior, panel_size)`, `compute_crowding_features(holdings, mapping, public_periods, panel_size)` | per CUSIP from long-equity rows only: `n_holders`, `total_value_thousands`, `max_holder_value_thousands`, `ownership_hhi = sum((value/value.sum())^2)`, `prior_value_thousands`, `inst_coverage_pct = n_holders / panel_size`, `inst_value_change = total - prior`, `inst_pct_change`; `crowding_score = n_holders / median(n_holders)`; `top_holder_pct = max_holder / total` |
| 7 Co-ownership | `build_holder_sets(latest_positions, matched_cusips)`, `summarize_coownership(cusip, holders_by_cusip, matched_cusips) -> (avg Jaccard, n peers)`, `compute_coownership_similarity(...)` | `avg_coownership_jaccard` over unique matched CUSIPs; aliases are separate rows, never duplicate peers |
| 8 Temporal | `build_vintage_snapshots(holdings, periods, cusips)`, `summarize_ownership_history(...)`, `compute_ownership_temporal_features(...)`, `compute_temporal_features(companies, holdings, mapping, public_periods)` | `ownership_churn = 1 - mean Jaccard(consecutive holder sets)`; `position_value_cv`; `new_holders_recent`; supply columns `relationship_churn`, `centrality_momentum`, `supplier_change` fixed at 0 (single snapshot) |
| 9 Cross-graph | `compute_cross_graph_features(topology, supply_features, crowding)` | `supply_chain_crowding = supplier_overlap_ratio * crowding_score`; `concentrated_dependency_risk = supplier_dependency_score * ownership_hhi`; `systemic_exposure = betweenness * n_holders`; `customer_concentration_risk = n_customers * top_holder_pct` |
| 10 Matrix | `build_complete_feature_matrix(companies, topology, supply_features, crowding, similarity, temporal, cross_graph)` | start from `pl.DataFrame({"entity": companies})`; left-join each frame on `entity`; ensure `cusip`, `issuer_name` exist |
| 11 Diagnostics | `feature_families` dict + coverage assertion; pairwise correlations (each coefficient on companies observed for that pair); z-score heatmap of 20 most extreme complete profiles | families: topology, supply, holdings, temporal, cross_graph (exact membership partly inferred); heatmap rows via `np.argsort(row_score, kind="stable")[-HEATMAP_ROWS:][::-1]`; aliases sharing a CUSIP averaged for the plot only; `ml4t_diverging` symmetric scale |
| 12 Persist | `OUTPUT_DIR = get_output_dir(23, "knowledge_graph_features")` | `features.parquet` (wide, for gradient boosting), `features_long.parquet` (`unpivot(index=["entity","cusip","issuer_name"], variable_name="feature_name")`, for IC analysis), `feature_metadata.parquet` (`FEATURE_METADATA`: feature_name, category, interpretation), `company_mapping.parquet` |

Run order: `13f_download.py` -> 02 (populate Neo4j, record snapshot) -> 09.

### 11. GAT autoencoder ablation (`06_gnn_feature_engineering`)

| Config | Default |
|---|---|
| `N_ASSETS`, `LOOKBACK_DAYS`, `TARGET_HORIZON` | 200 Wiki-Prices stocks, 504, 21 |
| `CORRELATION_THRESHOLD` | 0.5 (edge if \|corr\| > 0.5; negative and positive co-movement both count) |
| `EMBEDDING_DIM`, `GNN_EPOCHS`, `SEED`, `DEVICE` | 4, 250, 42, `cuda` |
| ridge `penalty` | 0.1 (unpenalized mean); 5 held-out-stock folds |

1. Boundary: the last 21 trading dates form the forward window; liquidity ranking, the 8 node features (momentum, volatility, mean-reversion, trend via `build_node_feature_row(history, symbol)`) and the graph use only data on or before `feature_as_of`; cohort = ranked stocks observed at both target endpoints; re-sort after joins. Target `compute_forward_return(symbol_prices, feature_date, target_end)` = close-to-close.
2. Per fold: `fit_train_scaler(values, train_indices)` (moments from training rows only; scaler exposes its moments for testing) -> `DenseGATAutoencoder(input_dim, embedding_dim)` (single-head attention; feature + edge reconstruction) with losses from `build_graph_losses(train_graph)` (edge-imbalance-weighted `BCEWithLogitsLoss` + `MSELoss`, computed on the training-induced adjacency) -> `fit_graph_embeddings(features, graph, train_indices, fold_seed)` trains on the train-induced subgraph without labels, then transductive inference over the full pre-target graph -> `ridge_predict(train_X, train_y, test_X, penalty=0.1)` -> `compute_ic` (Spearman) -> `evaluate_fold`.
3. Report paired per-fold hybrid-minus-tabular IC deltas and their spread (`spread_label_positions(values, minimum_gap)` for the figure); print the isolated-node count. No confidence statement: folds share training stocks and one return window. Optimizer / learning rate / loss weighting were not visible. README lists `torch_geometric`, but 06 uses a dense plain-torch GAT.

### 12. Correlation networks, MST and network-diversified portfolios (`10_network_portfolio_construction`, 23.5)

| Config | Default |
|---|---|
| `N_ASSETS`, `ESTIMATION_DAYS`, `EVALUATION_DAYS` | 100, 504, 252 |
| `MIN_DATE`, `MIN_SYMBOLS_PER_DATE`, `SEED` | `"2015-01-01"`, 1_000, 42 |
| `CENTRALITY_FLOOR`, `MAX_WEIGHT` | 0.01, 0.10 (feasibility `MAX_WEIGHT * N >= 1`) |
| `CONTAGION_THRESHOLD`; shock magnitude, attenuation, rounds | 0.5; -0.10, 0.5, 5 |

1. Data: US equities dataset (3,199 current and delisted securities); a date is a market date once `MIN_SYMBOLS_PER_DATE` symbols report (lower for subsampled panels); universe, correlations, volatility and all weight vectors fixed at the end of the 504-session estimation window before the 252-session evaluation (universe selection details not shown).
2. Returns: difference of two closes; an exited symbol or missing quote gets a zero return (explicit stale-mark / cash assumption); count such fills over analysis dates only (first row is null for every symbol by construction).
3. Distance `d_ij = sqrt(2 (1 - rho_ij))` in [0, 2]; inverse `rho = 1 - d^2 / 2`.
4. `compute_mst(distance_matrix) -> (mst_matrix, edges)`: `scipy.sparse.csgraph.minimum_spanning_tree(distance).toarray()`; symmetrize `mst + mst.T`; edges `(i, j, d)` for `mst[i, j] > 0`; verify `len(edges) == N - 1`; print total MST distance and the 5 lowest-distance pairs.
5. Centrality: `compute_degree_centrality(adjacency) = (adjacency > 0).sum(axis=1) / (N - 1)`; `build_mst_graph(mst_matrix, n) -> nx.Graph` (`weight = distance`); `nx.betweenness_centrality(mst_graph, weight="weight")`; `nx.closeness_centrality(mst_graph, distance="weight")`. A leaf has degree centrality 1/(N-1) = 0.0101 for N = 100, the same order as the floor.
6. Weights: `network_diversified_weights(centrality, max_weight=MAX_WEIGHT, floor=CENTRALITY_FLOOR)` = `project_capped_weights(1 / (centrality + floor), max_weight)`, called with degree centrality; benchmarks `equal_weight_portfolio(n) = ones/n` and `inverse_volatility_weights(returns, max_weight)` (`cov = np.cov(returns.T)`, `vols = sqrt(diag(cov))`, `inv_vol = 1/(vols + 1e-8)`, project). `project_capped_weights` validates `max_weight > 0`, `max_weight * n >= 1`, finite nonnegative scores with positive sum (else `ValueError("max_weight is infeasible for this portfolio size")` / `ValueError`), then water-fills: scale free scores to the remaining mass, fix any proposal above the cap at the cap, repeat.
7. Concentration table per rule: `hhi = sum(w^2)`, `effective_names = 1/hhi`, `max_weight = w.max()`; assert `isclose(sum, 1)`, `min >= 0`, `max <= MAX_WEIGHT + 1e-12`. Report the network rule's shortfall vs inverse volatility, not vs equal weight (which attains N by construction).
8. Shock: `simulate_shock_diffusion(corr_matrix, shock_asset, shock_magnitude=-0.10, contagion_threshold=0.5, attenuation=0.5, max_rounds=5) -> {shocked_asset, n_impacted, impacts, propagation_rounds}`; synchronous rounds; each frontier source proposes `impacts[source] * corr[source] * attenuation` to not-yet-impacted assets with `corr[source] > threshold`; most negative proposal wins (`np.minimum`). Scenarios rank assets by MST degree (hub vs lower-degree); print `pairs_above_threshold = (corr > CONTAGION_THRESHOLD).sum()`; sensitivity table = `impacts @ weights` for equal and network weights within each scenario.
9. `backtest_portfolio(returns, weights) -> dict`: `returns @ weights` over the later 252 sessions = daily rebalancing to fixed targets, gross of costs and financing; growth of one dollar, drawdown, summary stats per rule. Descriptive only.

## Guardrails and pitfalls

Corpus and extraction (01, 02, 08)
- **Pivot column order** — `pivot(on="year")` returns columns in data order (2022, 2024, 2025, 2020, 2023, 2021) while the axis used a sorted list, so every heatmap column carried the wrong year. Select and name columns explicitly in sorted order.
- **Imputing missing filings** — a missing company-year can be late index entry, filer-identity change or a download miss; carry the gap, never impute; sort rows by coverage.
- **Text length as informativeness** — 10-K spikes are fixed extraction windows; treat length as an extraction diagnostic, never compare across forms.
- **Cache drift** — a regenerated or edited parquet silently changes the graph; `.meta.json` pins content hash, schema, row count, extractor identity; `_validate_cache` fails on mismatch; row-count bounds 100-50,000.
- **Summing stage timings** — on the cached path extraction never ran, so a "total pipeline time" would hang the 27-GPU-minute label on the Neo4j load. Time stages separately; never present one graph size as a scaling benchmark.
- **8-K item defaulting** — a large minority of excerpts name no item and dozens name two; defaulting to 8.01 or first match feeds the model a wrong context. Return all items in schema order; say "not identified".
- **Incomplete run-identity hash** — a hash missing a prompt-visible field (`company_name`, `symbol`) asserts an identity the extraction does not have. Hash every prompt-visible field over the frame actually read; mutate each field and require the digest to move.
- **Normalizer rewriting semantics** — silently fixing relations, dates or categories launders bad extractions. Normalize only casing and leading articles, once; route the rest to an unresolved collection; conserve counts (parsed / zero events / parse failure; accepted / unresolved).
- **Schema validity mistaken for truth** — CheckRules does not judge whether the filing supports the event; precision needs a separate labeled audit; category totals describe the 50-filing sample, not the population.

Entity resolution and loading (02, 05)
- **Fragmented entity nodes** — 82 of 133 subject strings were re-spellings or subsidiaries; each spelling becomes a node and every degree is measured on fragments. Canonical key + filer roster; print the rewrite table; assert node count == distinct resolved names.
- **Merging on role label instead of entity** — Microsoft as filer, supplier and customer becomes three nodes. `MERGE (:Entity {name})` and add the role as a second label.
- **Generic category phrases as nodes** — "suppliers", "customers" are not organizations; `is_actionable_entity` on the canonical key.
- **Subsidiaries named instead of registrants** — no string rule folds KAYAK into Booking Holdings; production resolves against a corporate hierarchy; otherwise leave separate nodes and state the inflated count.
- **Shared-supplier degree read as disruption risk** — counts reflect disclosure requirements; label it an exposure-concentration diagnostic; `supplier_risk_level` cut points (HIGH >= 10 naming companies, MEDIUM >= 5, else LOW) are display buckets, not a risk scale.
- **Top-N supplier ranking hides thin structure** — the twelfth supplier is an arbitrary member of a large tie at one company; cut on the count (`MIN_SHARED_COMPANIES=2`) so the bar count is the finding.
- **Shared Neo4j database clobbering** — 08 declares `Company.name` unique and attaches event edges to `:Company`; a naive reset duplicates or `DETACH DELETE`s them. Adopt `:Company` into `:Entity` before MERGE; delete only relationship-less entity nodes; clear only your own relationship types.
- **Downstream reads a partial or stale graph** — 09 cannot know whether Neo4j holds what 02 wrote; `read_graph_identity` hash recomputed independently; never hard-code expected counts in the consumer; normalize whitespace before hashing so connection details cannot leak into identity.
- **Grouping stocks on `(cusip, issuer)`** — managers spell one issuer differently in one quarter; the security is the CUSIP; MERGE on CUSIP; the load-count assertion catches it; print merged CUSIPs.
- **`aum` / `ticker` misnomers** — 13F covers only long US-listed equity and carries no ticker; name properties for what they hold (`equity_13f_value`, `label`); drop constant placeholders (`strategy="Unknown"`).
- **13F option rows summed as ownership** — a put profits when the issuer falls; summing makes a short view look long and reorders managers by size. `split_derivatives`; print the excluded share (same rule in Ch22 nb 07).
- **Survivorship in cohort selection** — rank the formation cohort once at the first report period so later filings cannot change membership.
- **Crowding "findings" that are selection criteria** — the 50 stocks are the largest positions of ten managers, so `holder_count > 2` selects almost everything; print the selected share; price-impact amplification is an untested hypothesis.
- **Fake rankings of tied Jaccard scores** — with ten holders the scores are ratios of small integers and the top is an alphabetical tiebreak; report how many pairs share the top value; draw holder counts against the full 0..cohort range.
- **Sector panel read as market structure** — `classify_sector` is a name screen; print the "Other" share before drawing.
- **Two graphs counted as one** — in-memory holds the latest period, Neo4j every period; label which graph each summary row counts; write `GraphSnapshot` for consumers.

Graph RAG and benchmark (03, 04)
- **Validator pass rates over queries written to pass** — 3/3 three times measures nothing, and one metric implied another (a date `<= cutoff` is also truthy). Use the hostile control set (six must-refuse, two must-accept).
- **Text policy over Cypher fails both ways** — `MATCH (n) RETURN n LIMIT 1000 // $cutoff_date` scanned every node while a comment satisfied the cutoff check; `LIMIT 100000` passed a presence-only check; unlabeled `MATCH (n)` passed the label check; `'RECALL HOLDINGS'` was refused as `CALL`. Strip comments and string literals first; word-boundary keywords; every node pattern labelled or bound; `max(LIMIT) <= ROW_LIMIT`; require a real `<=` / `<` against `$cutoff_date`.
- **Believing the validator makes generated Cypher safe** — a text policy over a query language has no completeness proof. Fixed templates with fixed parameters; validator as second line; production adds server-side whitelisting, RBAC, query-plan checks, audit logging.
- **Substring question routing** — "who holds apple pie?" matched the Apple holders template; match the whole normalized question; raise on anything unsupported.
- **Swapping `available_from` and `report_date`** — filtering on the report period admits filings not yet made; ordering by availability breaks when a manager files two quarters on one day. Cutoff on `available_from`, latest-per-pair on `report_date`.
- **Hard-coded cutoff** — `latest_quarter == "2026-02-17"` broke the day the artifact rolled forward; read the cutoff from the producer's `GraphSnapshot`; derive the negative-control date from the graph's own availability dates.
- **A cutoff parameter that does nothing** — re-run at an earlier cutoff; assert no future rows and never more rows than the as-of run.
- **Oracle recall presented as a comparative result** — the graph arm's predicates define the gold set, so "delta" is just 1 - vector recall; oracle is a reference line; assert the identity.
- **Token budgets read as retriever efficiency** — three confounds (row count 5-6 vs always 10, fields per row 4 vs 6, prose scaffolding) and no tokenizer count; report per-row figures, name the confounds, claim no inference cost.

Temporal integrity (07)
- **Wrong `GraphSnapshot` selected** — 05's snapshot lacks `extraction_time`; Neo4j sorts null above values, so `ORDER BY extraction_time DESC LIMIT 1` picks the 13F snapshot and the run-id join returns an EMPTY result, not an error. Match `snapshot_kind: '8k_events'`.
- **Clamped negative disclosure lags** — printed mean and figure mean differed under one name; keep lags signed; count `post_disclosure_effective_date` rows.
- **Constant churn metric** — each disclosure falls in exactly one disjoint window, so additions + removals = union and the ratio is 1 for every input; compare cumulative states; assert removals are zero.
- **A leakage test that cannot fail** — `public_time > cutoff` on a frame built by `public_time <= cutoff` only tests Polars; call it a postcondition; test extraction time instead.
- **Extraction-time leakage** — every edge comes from one run more recent than any cutoff, so a backtest reads a graph that did not exist then; no disclosure filter fixes it. Run the extractor repeatedly and keep each run's edges under its extraction date, or refuse edges until `extraction_time <= cutoff` (on a single-run graph: nothing to trade on, say so).

Features (09)
- **Look-ahead through the 13F filing lag** — positions as of `report_date` are unknowable until `filing_date`; admit a period only when its last filing (`available_from`) <= `CUTOFF_DATE`.
- **Partially filed newest quarter treated as complete** — non-filers read as exits (holder counts and values fall, churn rises) purely from an artifact boundary; require `institutions == PANEL_SIZE`; test by withholding one institution's newest filing and asserting the calendar drops that period.
- **Quarters inferred by clustering filing dates within 14 days** — merges chain without limit and label a filing date as a quarter end the data already carries; group by `report_date`, take max `filing_date` as `available_from`.
- **Coverage denominator over long-equity filers** — a manager reporting only options shrinks it; `PANEL_SIZE` over every filing; test by withholding one manager's long rows and asserting `inst_coverage_pct == n_holders / PANEL_SIZE` (RuntimeError otherwise).
- **`n_holders`, `inst_coverage_pct`, `crowding_score` as three features** — one small integer on three scales (bounded by the panel); print with denominators; keep one in modeling (inference).
- **Survivorship / look-ahead in issuer matching** — restrict candidates to issuers present on or before the cutoff.
- **Ambiguous name matches** — a wrong CUSIP pollutes every ownership feature; exact normalized match; prefix only when all candidates share one CUSIP; else null and audit.
- **Alias double-counting in co-ownership** — compute over unique matched CUSIPs.
- **Summing holdings across quarters** — not a tradable snapshot; newest fully public period (plus the prior one for changes) only.
- **Positional column slicing for families** — silent relabeling when columns change; named families plus a coverage assertion.
- **Reading the heatmap as evidence** — rows are selected by the displayed quantity, so extremes exist by construction; use only to see which columns extremes occupy.
- **Cross-graph products as measured risk** — algebraic, untested; metadata labels them candidate proxies; test via IC on the long format.
- **Nonzero supply-chain temporal features** — one snapshot exists; `relationship_churn`, `centrality_momentum`, `supplier_change` fixed at 0 and documented.

GNN ablation (06)
- **Ablation leakage** — the earlier implementation selected the universe with full-sample liquidity, standardized before CV and passed random projections off as embeddings (frozen-chapter numbers are a documented book-code divergence). Pre-target universe; scaler fit on training rows; encoder trained on the train-induced subgraph without labels; losses from the training adjacency only.
- **Reporting the fold mean** — folds disagree in sign; the single most adverse fold moves more than the rest combined; the mean is a small fraction of the spread. Print spread beside mean; draw paired lines; no confidence statement.
- **Isolated nodes cap the ablation** — a stock with no edge above 0.5 gets an embedding of only its own features; print the isolated count; raising the threshold isolates more, lowering it dilutes "strong co-movement".

Networks and portfolios (10)
- **Null first return row counted as missing** — count fills over analysis dates only.
- **Zero-return fill for exits / missing quotes** — a stale-mark assumption that dampens volatility and correlation; report the fill count.
- **Provider rows dated to holidays** — `MIN_SYMBOLS_PER_DATE=1_000` defines a market date; lower it for subsampled panels.
- **Universe / network / weights estimated with evaluation data** — fix everything at the end of the estimation window.
- **Benchmarking effective names against equal weight** — equal weight is the maximum N (1/HHI); only the shortfall is informative; compare to inverse volatility.
- **Infeasible cap** — `max_weight < 1/N` cannot sum to one; `project_capped_weights` raises.
- **Division by zero for isolated nodes** — `CENTRALITY_FLOOR=0.01`; never remove the floor.
- **Reach in shock diffusion as systemic importance** — tree and diffusion graph come from the same correlations; compare impact magnitudes under one weight vector and weights within one scenario; print pairs above threshold.
- **Order-dependent diffusion** — sequential updates depend on loop order; synchronous rounds with `np.minimum`.
- **Summed hypothetical asset losses as portfolio loss** — report `impacts @ weights`.
- **Gross daily-rebalanced backtest taken as achievable** — drift traded every session is where costs and financing arise; label gross; model costs (Ch18) before any claim.
- **One split generalized** — a single 252-session window is descriptive; multiple windows are needed for any selection claim.
- **Centrality-vs-weight plot read as evidence** — it is the rule `w ∝ 1/(c + 0.01)` itself.

## Decision rules and defaults

- Use a graph when the question is multi-hop (2+ joins), crowding / structural, or temporally evolving; otherwise a tabular groupby or vector retrieval suffices.
- KG contract checklist: stable keys (CIK for institutions, CUSIP for securities, canonical name only when nothing better exists); finite relationship vocabulary (02: 3 predicates; 08: 4 relations x 4 meta types); provenance per edge (source accession, CIK, dates, model name + revision, run id); one `GraphSnapshot` node per producer with `snapshot_kind`.
- Graph RAG safety sequence: route to a fixed template -> validate (read-only, schema-bound, labelled, `LIMIT <= 25`, dated against `$cutoff_date`) -> `enforce_query_limits` -> read-only execution with 2.0 s timeout -> synthesize only from returned rows. Keep the hostile control set and the cutoff negative control in CI.
- Two-date rule for holdings: filter `available_from <= cutoff`, order `report_date DESC`, take `collect(h)[0]`.
- Three-timestamp rule: never cut on event date; cut on disclosure date for PIT; additionally require `extraction_time <= cutoff` or maintain versioned extraction runs.
- Entity resolution order: normalize text -> canonical key -> roster lookup -> alias map -> generic filter -> frequency-based canonical form (all-caps loses ties).
- Benchmark reading rule: only an arm whose predicates do not define the gold set is a measurement; a 7-question / 10-institution set shows the shape of failure, not its size.
- 13F feature admission: complete iff `institutions == PANEL_SIZE`; public iff `available_from <= CUTOFF_DATE`; require >= 2 complete public periods; `CUTOFF_DATE=""` resolves to the date the newest complete period became public.
- Match acceptance: exact normalized name; else prefix with a single shared CUSIP; else null and audit.
- Supplier classification: shared if `count > 1` covered customers; single-source if `count <= 1`; dependency score = mean(1/count).
- CheckRules go / no-go for an 8-K event: subject / object non-empty, <= 5 words, uppercase, no leading article; relation in {ANNOUNCED, APPOINTED, ACQUIRED, VALUED_AT}; meta in {EXECUTIVE_CHANGE, ACQUISITION, STRATEGIC_PARTNERSHIP, DISCLOSURE}; ISO timestamp. One normalization pass; anything still failing is held back.
- GNN go decision requires consistent sign of the paired IC delta across folds, which the chapter's run did not show.
- Network portfolio: report the network-diversified shortfall vs inverse volatility; tune the floor by shrinking it until the cap binds, never remove it; post-weighting checks sum = 1, min >= 0, max <= cap + 1e-12, MST edges = N - 1.
- Artifact use: wide parquet for gradient boosting (Ch12), long parquet for IC analysis, mapping parquet for join audit.

| Area | Parameter | Default |
|---|---|---|
| Extraction (02 / 08) | model; sampling; `MAX_NEW_TOKENS`; batch | Qwen2.5-7B-Instruct, pinned `MODEL_REVISION`; greedy `DO_SAMPLE=False`; 512; 2 (10-K, 24 GB GPU) / 4 (8-K) |
| Extraction cache | rows; regeneration | 100-50,000; only on explicit `RERUN_EXTRACTION=True`, with hashed sidecar |
| Neo4j load | `batch_size` | 1000 (02), 500 (08) |
| Supplier display | risk buckets; figure subgraph | HIGH >= 10, MEDIUM >= 5, else LOW; suppliers named by >= 2 companies |
| Graph RAG | `ROW_LIMIT`; `TIMEOUT_SECONDS` | 25; 2.0 |
| Benchmark | `MAX_STOCKS`; `TOP_K_VECTOR`; cutoff | 250; 10; `"2026-02-17"` |
| 13F graph | `load_real_13f_data` | `max_institutions=20, max_stocks=100`; executed 10 x 50; options excluded; cohort fixed at first period |
| Temporal | `WINDOW_DAYS`; `CUTOFF_LAG_DAYS` | 90; 60 |
| Features (09) | `N_COMPANIES`; `HEATMAP_ROWS`; pagerank `alpha` | 0 (all); 20; 0.85 |
| GNN (06) | assets; lookback; horizon; threshold; dim; epochs; ridge; folds | 200; 504; 21; \|corr\| > 0.5; 4; 250; 0.1; 5 |
| Network (10) | assets; estimation; evaluation; cap; floor; market-date filter | 100; 504; 252; 0.10; 0.01; 1,000 symbols |
| Shock (10) | magnitude; threshold; attenuation; rounds | -0.10; 0.5; 0.5; 5 |

## Code patterns and APIs

Paths and helpers: `23_knowledge_graphs/0{1..9}_*.py`, `10_network_portfolio_construction.py` (Jupytext); `get_chapter_dir(23)`, `get_output_dir(23, name)`, `set_global_seeds(SEED)`, `load_sec_filings`, `load_holdings_artifact()`; env `ML4T_DATA_PATH`, `ML4T_OUTPUT_DIR`, `NEO4J_URI/USER/PASSWORD`; style helpers `utils.style` (`add_message_title`, `ml4t_diverging`, `show_with_alt`). Tests: `uv run pytest tests/test_chapter_notebooks.py -v -k "23_knowledge_graphs"`; headless `MPLBACKEND=Agg PLOTLY_RENDERER=json`.

Batched UNWIND load, one template per relationship type (shape inferred from the description of `UNWIND_TEMPLATES`; exact text not read):
```cypher
UNWIND $rows AS row
MERGE (c:Entity {name: row.subject}) SET c:Company
MERGE (s:Entity {name: row.object})  SET s:Supplier
MERGE (c)-[:HAS_SUPPLIER]->(s)
```

Point-in-time holdings template (shape inferred; 03's three templates are hand-written with fixed parameters):
```cypher
MATCH (i:Institution)-[h:HOLDS]->(s:Stock {cusip: $cusip})
WHERE h.available_from <= $cutoff_date
WITH i, s, h ORDER BY h.report_date DESC
WITH i, s, collect(h)[0] AS latest
RETURN i.name, latest.value, latest.report_date LIMIT 25
```

Validator: regexes `LABEL_PATTERN`, `REL_PATTERN`, `PROPERTY_PATTERN`, `NODE_PATTERN`, `BOUND_VARIABLE_PATTERN`, `LIMIT_PATTERN`; keyword check after `strip_noise`:
```python
re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", code.upper())
```

Provenance hashing: `hashlib.sha256` over sorted, field-restricted rows (`frame_sha256`); `RUN_ID = sha256(json.dumps(run_identity, sort_keys=True))[:24]`.

LLM stack: `transformers` `AutoTokenizer` / `AutoModelForCausalLM` with chat template; `torch.cuda.is_available()` gate. Vector baseline: `sentence_transformers.SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")`, top-k cosine.

Graph analytics: `networkx` (`build_networkx_graph(triples, min_companies=2)`; `nx.MultiDiGraph` -> simple graph; `nx.pagerank(G.reverse(copy=True), alpha=0.85)`; `nx.betweenness_centrality`; `nx.clustering`; `nx.closeness_centrality(mst_graph, distance="weight")`); `scipy.sparse.csgraph.minimum_spanning_tree`; visualization `pyvis.network.Network`, matplotlib two-ring layout (`build_static_layout`, `create_static_figure`), D3 export `export_d3_json(G, output_path)` -> `{nodes, links, metadata}` + `D3_HTML_TEMPLATE`.

Polars: `group_by("cusip").agg(...)` with `ownership_hhi = ((value/value.sum())**2).sum()`; `pl.DataFrame({"entity": companies}).join(frame, on="entity", how="left")`; `unpivot(index=[...], variable_name="feature_name")`; `write_parquet`.

Capped-simplex projection (reusable):
```python
weights = np.zeros_like(scores); free = np.ones(n, bool); remaining = 1.0
while free.any():
    proposal = scores[free] / scores[free].sum() * remaining
    capped = proposal > max_weight
    if not capped.any(): weights[free] = proposal; break
    idx = np.flatnonzero(free); weights[idx[capped]] = max_weight; free[idx[capped]] = False
    remaining = 1.0 - weights.sum()
```
Inverse-centrality weights: `project_capped_weights(1 / (centrality + floor), max_weight)`. Correlation distance: `distance = np.sqrt(2 * (1 - corr))`; `rho = 1 - d**2 / 2`.

Synchronous diffusion step:
```python
eligible = (~impacted) & (corr[source] > threshold)
propagated = impacts[source] * corr[source] * attenuation
proposals[eligible] = np.minimum(proposals[eligible], propagated[eligible])
```
Gross daily-rebalanced backtest: `portfolio_returns = returns @ weights`.

GNN: plain `torch.nn.Module` `DenseGATAutoencoder`; `nn.BCEWithLogitsLoss` with positive-class weight from edge imbalance + `nn.MSELoss`; numpy ridge with unpenalized intercept; Spearman `compute_ic`.

## Evidence from the book

- Supply-chain extraction (02): 601 10-K filings, Qwen2.5-7B-Instruct, batch 2, ~27 min on an RTX 3090, ~14 GB VRAM; cache parquet 11 KB. 133 distinct subject strings: 51 unchanged, 61 re-spellings of the registrant, 21 subsidiaries / brands. Word-order sorting merged exactly one group (four Charles Schwab spellings). Almost every supplier is named by exactly one company; shared structure rests on a handful of suppliers. Edge precision not established.
- Graph RAG validator (03): original diagnostics 3/3, 3/3, 3/3 (uninformative). Control set: 6 hostile queries refused, 2 benign accepted; earlier validator versions accepted a whole-graph scan with the cutoff in a comment, `LIMIT 100000` and unlabeled `MATCH (n)`, and refused `'RECALL HOLDINGS'`.
- RAG benchmark (04; cutoff 2026-02-17, 7 questions over 10 institutions): oracle recall 1.0 by construction; the vector arm (MiniLM, top-10) recovered one holder question completely and missed another completely, driven by how the issuer is named in the corpus. Token budget: relational 5-6 rows x 4 fields vs vector 10 rows x 6 fields; the row count alone roughly halves the budget.
- 13F graph (05): 10 managers x 50 stocks; the 50 largest names are held by nearly all ten managers (holder counts cluster near the cohort size); Jaccard values tie heavily; excluded option rows are a non-trivial share of reported value and reorder managers by size; CUSIP merges collapsed duplicate issuer spellings caught by the load-count assertion.
- GNN ablation (06; 200 stocks, one 21-day window, 5 stock folds): paired hybrid-minus-tabular IC deltas disagree in sign; the most adverse fold moves further than the others combined; the mean is a small negative number that is a small fraction of the fold spread; sign undetermined. Frozen-chapter numbers from the flawed earlier implementation are not comparable.
- Temporal event graph (07): disclosure lag is signed with mass left of zero (effective dates after filing); window-to-window churn was identically 1 (artifact of disjoint windows); every visible edge was extracted after any feasible cutoff, so the single-run graph fails the extraction-time check by construction.
- 8-K extraction (08): 50 filings, batch 4; a large minority of staged excerpts name no item and dozens name two; full accounting of parsed / zero-event / parse-failure filings and accepted / unresolved events is printed; no semantic precision or speed-up measured.
- Features (09): on the artifact as shipped every 13F period is complete and every manager reports long equity in every period, so the coverage gate and panel-denominator rule are inert; each is exercised by a synthetic withholding test that must change the answer. No predictive value is measured; crowding proxies and cross-graph terms are explicitly untested candidates. `n_holders`, `inst_coverage_pct`, `crowding_score` are one small integer on three scales.
- Networks (10): MST on 100 assets has 99 edges. The contagion graph at corr > 0.5 is dense enough that a shock from anywhere in its connected part reaches the same assets within 5 rounds, so reach does not separate the hub and lower-degree scenarios; impacts differ by path. Equal weight attains effective names = 100 and max weight 0.01, far below the 0.10 cap. Leaf degree centrality ~0.0101 vs floor 0.01, so the floor roughly halves a leaf's raw inverse score. The 252-session gross evaluation is descriptive; numeric Sharpe / drawdown / concentration values are not in the notes.

## Related references

- `chapters/22_rag_financial_research.md` — vector RAG foundation; `01_sec_filing_pipeline` explains 10-K extraction windows; nb 07 makes the same 13F put-exclusion.
- `chapters/04_fundamental_alternative_data.md` — 13F loader (`data/equities/positioning/13f_download.py`) and SEC EDGAR acquisition (`filings_download.py`).
- `chapters/10_text_feature_engineering.md` — NER and filing-parsing patterns upstream of extraction.
- `chapters/24_autonomous_agents.md` — agents consuming structured retrieval, memory and evidence tracking from the graph.
- `chapters/12_gradient_boosting.md` — consumes the wide cross-graph feature matrix (`features.parquet`).
- `chapters/14_latent_factors.md` — consumes co-ownership and crowding features.
- `chapters/07_defining_the_learning_task.md` — IC evaluation of the long feature format; multiple-testing discipline for candidate proxies.
- `chapters/11_ml_pipeline.md` — fold-internal preprocessing and held-out-group CV mirrored by the GAT ablation.
- `chapters/17_portfolio_construction.md` — equal-weight, inverse-volatility and capped-weight rules the network-diversified portfolio is compared against.
- `chapters/18_transaction_costs.md` — costs and financing the gross daily-rebalanced backtest omits.
- `chapters/19_risk_management.md` — contagion / systemic-risk framing of the shock simulation.
- `chapters/02_financial_data_universe.md` — point-in-time and disclosure-date discipline generalized by the three-timestamp model.
- `case_studies/us_firm_characteristics.md` — 13F / fundamentals panel the ownership features draw on (inference).
- `case_studies/us_equities_panel.md` — US equities dataset (3,199 securities) used by the network notebook (inference).
- `libraries/ml4t_data.md` — staged filings and holdings loaders (`load_sec_filings`, `load_holdings_artifact`).
- `libraries/ml4t_engineer.md` — feature-matrix conventions for the graph-derived features.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `workflow.md`, `companion_repo.md` — cross-cutting indexes.
- Further reading:
  - Arun et al. 2025, FinReflectKG and FinReflectKG-MultiHop (CheckRules; multi-hop QA benchmark)
  - Li & Sanna Passino 2024, FinDKG (quadruple format)
  - Edge et al. 2025, From Local to Global: Graph RAG; Peng et al. 2024, Graph RAG survey; Lewis et al. 2021, RAG; Choi et al. 2025, FinDER
  - Veličković et al. 2018, GAT; Kipf & Welling 2017, GCN; Hamilton et al. 2018, GraphSAGE; Weber et al. 2019, AML GCN
  - Mantegna 1999, MST; Marti et al. 2021, correlation-network review; Konstantinov et al. 2023, Financial Networks and Portfolio Management; Konstantinov & Fabozzi 2025, When Factors Collide
  - Haldane & May 2011, systemic risk; Antón & Polk 2014, Connected Stocks; Greenwood & Thesmar 2011, Stock price fragility
  - Cheng et al. 2020, KG event embeddings; Kertkeidkachorn et al. 2023, FinKG; Elhammadi et al. 2020, high-precision KG pipeline; Miao et al. 2019, dynamic financial KG; Zehra et al. 2021, KG report query system

## Glossary

- **Knowledge graph (KG)** — entities as nodes, typed relationships as edges, provenance as properties.
- **Property graph** — graph model whose nodes and edges carry key-value properties (Neo4j).
- **Cypher** — Neo4j's query language (`MATCH`, `MERGE`, `UNWIND`, `LIMIT`). **UNWIND** expands a list parameter into rows for batched writes; **MERGE** is idempotent create-or-match.
- **Triple** — (subject, predicate, object). **Quadruple** — (subject, relation, object, timestamp) event edge in FinDKG format.
- **CheckRules** — FinReflectKG-style schema rules embedded in the extraction prompt and re-applied as a validator.
- **Entity resolution** — mapping surface names to one canonical node. **Canonical key** — normalized join key (case, punctuation, suffixes, word order removed). **Filer roster** — corpus-derived map of official registrant names.
- **Graph RAG** — retrieval where a database query, not similarity, selects evidence. **Text-to-Cypher** — LLM-generated Cypher (not what 03 does; it routes). **ROW_LIMIT** — maximum rows a validated query may return (25).
- **GraphSnapshot** — Neo4j node recording a producer's source bytes, cohort policy, periods, counts and run id. **Snapshot guard** — consumer-side check that a graph read from Neo4j matches the recorded identity. **Snapshot manifest** — JSON with cutoff, window, extractor identity and per-parquet SHA-256.
- **report_date** — 13F quarter-end a position is as of. **available_from / filing_date** — when the 13F reached EDGAR (~6 weeks later); the last one in a period dates when the period became knowable.
- **Complete period** — report period for which every institution in the panel has filed. **Public period** — complete period whose last filing arrived on or before the cutoff. **Panel size** — unique CIKs in the artifact, counted over every filing. **Formation cohort** — universe fixed at the first report period.
- **Event / disclosure / extraction time** — when it happened / when it became public / when the pipeline produced the edge. **Point-in-time (PIT)** — query restricted to information public at the cutoff.
- **Jaccard similarity** — |A ∩ B| / |A ∪ B| over holder sets. **Crowding** — many institutions holding the same name. **crowding_score** — n_holders / median n_holders. **inst_coverage_pct** — n_holders / panel size. **ownership_hhi** — sum of squared holder value shares. **top_holder_pct** — largest holder's share of total value.
- **supplier_dependency_score** — mean over suppliers of 1 / (covered companies using that supplier). **supplier_overlap_ratio** — shared suppliers / total suppliers. **ownership_churn** — 1 minus mean Jaccard of consecutive-period holder sets. **position_value_cv** — coefficient of variation of total value across periods. **Churn** — additions / removals between states; constant under disjoint windows.
- **GAT** — graph attention network; here a single-head dense autoencoder. **Transductive inference** — encoder applied to the full graph including held-out nodes' features and edges, never their labels. **Information coefficient (IC)** — Spearman rank correlation of predictions and realized returns.
- **Oracle baseline** — retrieval whose predicates define the gold set (recall 1 by construction).
- **Correlation distance** — d = sqrt(2(1 - rho)), a metric on [0, 2]. **MST** — minimum spanning tree: N-1 edges of minimal total distance. **Degree centrality** — degree / (N - 1). **Betweenness centrality** — share of shortest paths through a node. **Closeness centrality** — inverse average shortest-path distance.
- **Capped-simplex projection** — map nonnegative scores to weights summing to one with each <= cap. **Effective names** — 1 / HHI of the weight vector; maximal (= N) for equal weight. **Synchronous diffusion** — all frontier nodes propagate in one round, so results are loop-order independent.
