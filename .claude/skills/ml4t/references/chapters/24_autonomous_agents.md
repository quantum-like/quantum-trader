# Chapter 24: Autonomous Agents

> This chapter moves from fixed prediction functions over prepared feature matrices to agentic workflows that gather, filter and synthesize messy evidence before emitting a probability forecast. Its scope is deliberately **read-only decision support** (probability forecasts, traceability, replay); there is no execution path. It reconstructs the AIA Forecaster (Alur et al., 2025) as research agents → aggregation → debate → supervisor → scoring, and then builds a second capstone, the ML4T Research Operator, that runs Chapter 20 "next experiments" with ten general-purpose tools and an on-demand skills corpus. The position it argues: agents earn their cost only when evidence acquisition and structured judgment are the bottleneck; everything that makes them decision-grade (tool contracts, typed state, quality gates, abstention, point-in-time cutoffs, proper scoring, fail-closed policy proxies) is ordinary engineering, not prompting. Every number that moves a forecast is a declared constant, and the trace, not the probability, is the deliverable.

## When to use this reference

- Designing or reviewing an LLM agent that forecasts a binary event (prediction-market style question) from web/document evidence.
- Writing a tool contract for an LLM agent (search, retrieval, file or SQL access) and deciding what provenance, cutoff and source policy it must enforce.
- Building agent state, checkpoints, quality gates or abstention logic; deciding JSON vs pickle; replay/ablation of an agent run.
- Aggregating several probability forecasts (agents or analysts): mean vs Neyman extremization vs confidence weighting, choosing ρ, calibration (Platt / log-odds exponent).
- Running or evaluating adversarial debate, supervisor review, or a multi-stage forecasting pipeline; deciding whether a stage "earned its cost".
- Scoring probability forecasts (Brier, log score, ECE, sharpness, reliability diagrams) and avoiding in-sample calibration claims or post-resolution contamination.
- Choosing between plain Python, LangGraph and CrewAI for an agent pipeline, or migrating a notebook agent to a service.
- Securing an agent: prompt-injection scanning, Warden policy proxy, allowlists, rate limits, sandboxing a bash-capable operator, OWASP LLM Top 10 mapping.
- Running or reading an ML4T Research Operator run against a case-study registry (ETFs, US firm characteristics) with the `ml4t/skills` corpus.
- Auditing any LLM-agent backtest claim: were search results dated? was the market price in the prompt? were forecasts timestamped before resolution?

## Core ideas (the why)

- **Agents only where the bottleneck is evidence.** If a conventional model already runs on stable, fully structured inputs, keep the statistical/rules pipeline. Agents add value when messy evidence must be acquired and judged (README §24.1).
- **Reasoning patterns are budgeted engineering choices.** ReAct is the default because every turn is either a re-issuable query or a scorable probability. Tree of Thoughts only for branch-heavy decisions; Reflexion only with explicit memory governance (§24.2).
- **A small action space makes a loop auditable.** Two actions (`search`, `forecast`) mean every turn is explainable from the record; each added action adds unexplainable failure modes.
- **Model output is untrusted input.** Parse against the declared schema, return malformed replies for correction, never let them reach a tool.
- **Budget exhaustion must be distinguishable from a forecast.** `run_react_agent` literally returns `0.5, "Max steps reached"` on exhaustion; `ResearchAgent` wraps this with `forecast_produced=False`. Branch on that flag before any averaging, or a timed-out agent becomes a confident coin-flip.
- **The tool contract decides what the agent can know.** Tool quality often matters more than prompt quality; everything downstream is a function of what search returned (§24.4).
- **A publication-date filter separates a backtest from a demonstration.** Undated results are a separate class, not point-in-time safe; the evaluation (not the tool) decides their policy.
- **Enforce policy in the tool layer, not the prompt.** A model told to avoid social media sometimes reads it anyway; a host filtered post-retrieval never reaches the model. A prompt is a request; a proxy is a control (§24.10).
- **Execution log ≠ reasoning trace.** The log is what ran; the trace is what the model said it did. When they disagree, trust the log; debugging starts with their difference.
- **A trace is not state.** `RunTrace` is the run from outside (prompts, replies, results); `AgentState` is what the agent knew from inside (typed evidence, open questions, gate outcomes). Only state can be gated, resumed and diffed field-by-field (§24.3).
- **Memory design is a model-risk problem.** Working memory = context window; short-term = durable `AgentState`; long-term = retrieval across runs (RAG, Ch22).
- **A quality gate is a contract about evidence, in code, before synthesis.** Abstention is an outcome, not a failure: an unsupported forecast is scored as a judgment and costs more than no forecast.
- **Freshness and consistency are different questions.** Retrieval time answers "is this run stale?"; publication time answers "did this run read the future?". Conflating them lets one hide the other.
- **Evidence type is judged while gathering** and cannot be recovered from a log afterwards; write typed state as you go.
- **The artifact is the deliverable, not the number.** Separate what the model said (`p_yes`, rationale) from what was derived (confidence, sentiment, findings, evidence class). Derived fields are cheap uniform heuristics, not estimates. Extremity is not confidence; volume is not quality.
- **The mean is the baseline anything else must beat out of sample.** Extremization is a claim about independence (ρ), not about agreement; panel spread does not enter Neyman's formula. Effective panel size saturates at 1/ρ, so buy diversity by changing what agents read (model, index, framing), not by adding agents.
- **A calibration parameter is fitted, frozen, then evaluated on unseen observations.** Any other order measures how well a curve fits the points it was drawn through; "is the number almost every calibration claim quietly is."
- **Disagreement is information; averaging throws it away.** Debate "spends" disagreement. The gap trajectory is the output, not the midpoint: a closing gap is a case for updating; a constant gap reports how wide the disagreement is. Both look identical as one blended number (§24.7).
- **Adversarial roles are a prompt, not a mechanism.** Bull and bear are one model with two prompts arguing from summaries; convergence is prompt convergence, a cheap stress test, not a second opinion.
- **An override needs a gate and the gate needs a rule.** Without the confidence gate, one model's second opinion silently outranks three agents' evidence. Stated confidence ("high/medium/low", 0.8/0.6/0.5) orders outcomes; nothing estimates it.
- **A pipeline's value is inspectability, not stage-wise improvement.** Four stages are four places to look when a forecast is wrong. Whether any stage improved the estimate is a scoring question that needs resolved questions.
- **An agent handed the market's own probability is not an independent check on it.** Closeness to market is circular.
- **Testing on questions in training data tests recall, not forecasting** (Lopez-Lira, "Memorization Problem"). Use open questions or contamination-aware resolved sets with cutoffs.
- **Freeze the evidence before comparing configurations**; otherwise the comparison includes whatever the search index did that day (§24.9).
- **Start with plain Python.** A framework earns its place against a named pressure (crash recovery → LangGraph checkpointing; named personas → CrewAI roles). A framework that owns the model client owns the observability (§24.5).
- **Operator shape for open experiment spaces.** A research follow-up is a small program, so the operator exposes general tools (files, bash, SQL, parquet, skills) and the LLM decides; knowledge lives in skills and the ml4t libraries, fetched on demand so the prompt does not grow (§24.8).
- **Negative results are first-class.** An agent that confabulates an improvement is worse than no agent. Read the agent's conclusion against its own evidence; overclaiming scope (validation window reported as holdout) gets past every check but a human reader.
- **Record the conversation, not just the answer.** Prompts, replies and tool results in one JSON make a run reproducible after the market closes and the model is retired.

## Method recipes (the how)

### LLM provider abstraction (`24_autonomous_agents/01_react_reasoning`, `agent_providers.py`)

- `LLMClient` protocol has two methods only: `complete(messages) -> str` and `complete_with_usage(messages) -> (str, TokenUsage)` (prompt + completion tokens for cost). Deliberate floor: no streaming, native tool calling or thinking budgets, so agents stay portable across providers.
- `create_llm_client(provider="")` auto-selects the first configured provider in order **Anthropic → OpenAI → Google → OpenRouter → Ollama**; `LLM_PROVIDER` env var overrides (`mock`, `openrouter`, ...). Only effective when `RUN_LIVE=True`; `RUN_LIVE=False` replays pinned captures with zero API calls and is the publication path.
- Keys in `.env` (see `.env.example`): `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY` (+ `OPENROUTER_MODEL`, e.g. `anthropic/claude-sonnet-4`, `deepseek/deepseek-v4-pro`), `TAVILY_API_KEY`. Installs: `uv pip install anthropic httpx` / `openai httpx` / `httpx`; Ollama: `ollama pull qwen2.5:32b`.
- `MockLLMClient` (canned replies: searches once, then commits; `LLM_PROVIDER=mock` is a smoke test, not a reproduction). `TracingLLMClient` wraps any client to capture every prompt/response; wrap once at the client boundary, one tracer per agent.

### ReAct loop (`24_autonomous_agents/01_react_reasoning`)

1. Actions: `{"action":"search","query":"..."}` or `{"action":"forecast","p_yes":0.XX,"rationale":"..."}`; exactly one JSON action per step (same two-action schema as AIA Forecaster).
2. `_init_react_session(question, search_client, market_price) -> (cutoff_date, ToolExecutor, messages)`; messages start with system prompt + `build_step_prompt(q, market_price=None)` and grow two entries per turn (model reply, tool result).
3. Each step: `llm.complete_with_usage(messages, json_mode=True)` → `_parse_action(response) -> (action_type, action|None)`; invalid → `_record_invalid_action(...)` appends the error and asks for correction (costs a turn).
4. `run_react_agent(llm, search_client, question, max_steps=5, market_price=None) -> (p_yes, rationale, traces, token_usage)`; terminates on a valid forecast (p clamped to [0,1]) or step limit, where it returns `0.5, "Max steps reached"`. A production decision layer must retry or abstain, never treat that as a forecast.
5. `MAX_STEPS=5` (mock run uses 3); every step captured as `AgentTrace`; `RunTrace` persists question, steps and full conversation to `forecast_traces/`.
6. Live question via `get_live_question()` (Polymarket API; falls back to the fixed demo question). Use open questions because resolved questions in training data test recall.

### Search tool contract (`24_autonomous_agents/02_tool_contracts`, `agent_tools.py`)

| Element | Contract |
|---|---|
| `SearchClient` protocol | `search(query, max_results=5, cutoff_date: date|None=None) -> list[SearchResult]` |
| `SearchResult` | `title`, `url` (provenance), `snippet`, `published` (date used for filtering), `score` (provider relevance; says nothing about truth) |
| `TavilySearchClient(api_key, allowed_domains=None, blocked_domains=None)` | POST `https://api.tavily.com/search` via `httpx.Client(timeout=30.0)`; parse, `_filter_by_date` if cutoff, then truncate `results[:max_results]` |
| Cutoff semantics | Drop any result published **on or after** `cutoff_date`; the agent reads only strictly earlier documents (demo: cutoff 2025-02-20, NVIDIA reported 2025-02-26, agent sees up to 2025-02-19) |
| Undated results | **Pass by default** (cannot be proven too recent); partition and count separately for any scored historical evaluation |
| `apply_domain_policy(items, *, allowed=None, blocked=None)` | Post-retrieval; matches registered domain + subdomains after stripping `www.` (`reuters.com` covers `uk.reuters.com`); empty set = no constraint; **blocklist first, then allowlist** |
| `DEFAULT_ALLOWED_DOMAINS` | reuters.com, wsj.com, ft.com, bloomberg.com, cnbc.com, nasdaq.com, sec.gov, federalreserve.gov, bls.gov, coindesk.com, cmegroup.com (11 domains) |
| `ToolExecutor(search=None, allowed_domains=None, blocked_domains=None)` | `.execute_search(query, max_results=5, cutoff_date=None)`; `.execution_log: list[ToolExecution]` with tool_name, args, status, `duration_ms`, client, result count, `n_blocked`; absent client → `status="disabled"`, returns `[]`, loop continues |
| `ToolDefinition` | Renders one schema to Anthropic format (`input_schema` top-level) and OpenAI format (`{"type":"function","function":{...}}`) (renderer method names are inferred) |
| `format_search_results` | Numbered list; title / URL / snippet / date each on its own line |

Whether `TavilySearchClient` also applies domain filtering itself in addition to `ToolExecutor` is unclear in the notes (both accept the domain sets).

### Agent state, quality gates, checkpoint and replay (`24_autonomous_agents/03_state_and_memory`, `agent_schemas.py`)

- `AgentState` dataclass, 8 fields in 4 groups: identity (`run_id`, `question`, `cutoff_date`); evidence (`evidence`, `open_questions`); provenance (`tool_trace`); checks (`quality_gates`, `synthesis_status` ∈ pending → in_progress → complete, or `abstained`). `to_json()` = `json.dumps(asdict(self))`; `from_json()` = constructor; auto-generates `run_id`.
- Evidence item: `type` (`search_results` | `base_rate`), `source`, `timestamp` (**retrieval** time, not publication), `query`, `content`. Each search appends one evidence item **and** one tool-trace entry (the trace alone distinguishes "did not look" from "looked, found nothing").

| Gate | Checks | Default | Failure classes |
|---|---|---|---|
| 1 Coverage | Required types {`search_results`, `base_rate`} first, then `min_items` | `min_items=3` | missing type; too few items. Base-rate requirement is a structural defence against base-rate neglect |
| 2 Freshness | Retrieval `timestamp` within window of declared as-of | 24h (widen for quarterly-filing questions) | stale; **future timestamp = failure** (clock/fixture/checkpoint wrong) |
| 3 Consistency | Every search result's publication date parses and is before cutoff | one shared date parser for cutoff and results | missing date (tool could not date); unparseable (provider bug); post-cutoff (hindsight entered the run) |

- Each gate collects **all** failures, not the first. `run_quality_gates(state)` runs all three, stores outcomes on state, returns them; the caller decides (gather more, or set `synthesis_status="abstained"` and record which gate refused).
- Checkpoint as JSON, not pickle (editor-readable, diffable, queryable, version-independent); verify round-trip by comparing serialized forms.
- Replay ablation: restore checkpoint → drop one evidence class (e.g. `base_rate`) → re-run gates → read what moved. Fixture: `AS_OF_ISO` = 2025-02-19 (day before cutoff), fixed `RUN_ID`.
- Limitation: freshness is measured against the declared as-of, not wall clock (enables replay, but a stale run replays as fresh); gates run once before synthesis, so evidence arriving during synthesis is ungated.

### Research agent and artifact (`24_autonomous_agents/04_research_agent`, `agent_research.py`)

- `ResearchAgent(llm, search=None, agent_id="agent_0", max_steps=5, max_search_results=5)`; `.run(question, market_price=None) -> AgentForecastArtifact` (fields: `p_yes`, `rationale`, `confidence`, `sentiment`, `key_findings`, `uncertainties`, `evidence_quality`, `forecast_produced`, `traces`, query/source counts; full list partially inferred).
- System prompt hard constraints: emit valid JSON; **do not search for the market's own price** (verified against saved prompts, not trusted).
- `parse_json(raw) -> dict`: strips ``` fences, regex-extracts `{...}`, tries stripped/unfenced/matched candidates; returns `{"action":"parse_failure"}` otherwise.
- `validate_action(action) -> (type, normalized|None)`: search needs non-empty stripped str query; forecast needs finite numeric `p_yes` (bool rejected) and str rationale; `p_yes` clamped to [0,1]; optional `confidence` clamped.
- Internals: `_handle_search_step` → `(queries_inc, sources_inc)`; `_handle_forecast_step` → `(p_yes, rationale, raw_action)`; `_handle_unknown_step`; `_LoopResult`; `_assemble_artifact`; `_run_research_loop(agent, question, market_price)`.

Derived-field heuristics (label as heuristics wherever printed; `rederive_fields(artifact)` recomputes them on replay):

| Field | Rule |
|---|---|
| `extract_confidence` | model-supplied `confidence` clamped to [0,1] if finite; else `round(2*|p_yes-0.5|, 3)` |
| `extract_sentiment(p_yes)` | >0.8 STRONGLY_BULLISH; >0.6 BULLISH; >0.4 NEUTRAL; >0.2 BEARISH; else STRONGLY_BEARISH |
| `extract_key_findings(rationale)` | line-leading `-`/`•`/`*`/numbered items win; else inline `(1) ... (2)` markers (regex `\(\d+\)\s*`, ≥2 markers); last inline item cut at `SENTENCE_END = (?<![A-Z]\.)(?<=[.!?])\s+(?=[A-Z])` so "U.S. inflation" is not split; empty list if nothing enumerated |
| `extract_uncertainties(rationale)` | sentences containing uncertainty language (keyword list not visible) |
| `assess_evidence_quality(sources, queries)` | HIGH if sources ≥10 and queries ≥3; MEDIUM if sources ≥5 or queries ≥2; else LOW. Volume only |

- `format_agent_summary(artifact) -> str`: id, probability, confidence, opening of rationale, first few key findings. This is what supervisor/debate prompts receive; the full search trail stays out (context-budget decision).
- Gating a finished run from traces: each search step → one `search_results` evidence item, retrieval time = capture time. Coverage fails structurally (no `base_rate` recoverable from a trace); consistency has nothing to check when `cutoff_date=""` (open question).
- Observability: `show_agent_timeline(artifact)` (query → documents with title/date/URL/snippet → forecast with untruncated rationale); `ToolExecutor` audit table (`duration_ms` only on live runs, not persisted). Persistence: `RunTrace.capture(...)`/`.save()` → `forecast_traces/`; `RunTrace.load(...)`; replay never overwrites the pinned trace.

### Aggregation math (`24_autonomous_agents/05_aggregation_math`, functions in `agent_pipeline.py`)

| Method | Formula | Notes |
|---|---|---|
| Simple mean | p̄ | Assumes nothing; the baseline anything else must beat out of sample |
| Neyman extremization | d = sqrt(n / (1 + (n−1)ρ)); p_extreme = p_base + d·(p̄ − p_base) | ρ=0 → d=√n; ρ→1 → d→1. Implementation clamps d ∈ [1, 3] and result to [0.01, 0.99]. `neyman_extremize(probabilities, base=0.5, d=None, correlation=0.5) -> AggregationResult` (module default ρ=0.5; notebooks pass 0.3) |
| Weighted Neyman | HI = Σw_i² (normalized); n_weight = 1/HI; n_adj = n_weight / [1 + (n_weight−1)ρ]; d² = n_adj | `neyman_extremize_weighted(probabilities, weights, base=0.5, ...)`; uniform weights → n; all weight on one → 1 |
| Platt scaling | p' = d·p^a / (d·p^a + (1−p)^a) ≡ σ(a·logit(p) + log d) | `platt_scale(p, a, d=1.0)`; a>1 spreads toward ends, a<1 compresses, a=1 identity; d tilts the fixed point away from 0.5 |
| Log-odds extremization | p' = σ(a·logit(p)) | `logodds_extremize(p, a)`; Platt with d=1; straight line of slope a in logit–logit space |

- `fit_extremization_exponent(forecasts, outcomes, exponent_range=(0.5, 3.0), n_steps=50, ...) -> CalibrationResult`: grid-search the exponent minimizing Brier on a training panel; freeze; score held-out. Returns `at_search_boundary`; **assert the optimum is interior** (a boundary value is a bound, not a minimum) and refuse the fit otherwise.
- `reliability_bins(forecasts, outcomes, n_bands)`: equal-width probability bands with count, mean forecast, observed yes share. `RELIABILITY_BANDS=4` over 80 held-out questions (NB05); `RELIABILITY_BINS=4` over 10 (NB09).
- Also in `agent_pipeline.py`: `clamp_prob`, `brier_score`, `log_score`, `sharpness`, `expected_calibration_error`, `get_model_calibration_d(model)`.
- NB05 simulation: true p ~ symmetric Beta; outcome ~ Bernoulli(p); forecast = p shrunk toward 0.5 + noise (known under-confidence); `SEED` fixed. Effective panel size d² rises steeply with n then saturates at 1/ρ for fixed ρ>0.

### Parallel panel (`24_autonomous_agents/06_multi_agent_research`)

1. `ThreadPoolExecutor`, one task per agent (`run_agent(agent_id)` returns artifact + tracer); shared LLM and search clients must be thread-safe (Anthropic and Tavily are; otherwise construct inside the worker); one `TracingLLMClient` per agent; merge per-agent logs after the pool drains.
2. Settings: `N_AGENTS=3`, `MAX_STEPS=5`, `MAX_SEARCH_RESULTS=5`, `NEYMAN_CORRELATION` (assumed ρ), `LLM_PROVIDER`.
3. Filter to `forecast_produced=True`; compute mean, Neyman (supplied ρ), weighted Neyman (ρ + confidence weights).
4. ρ-sensitivity sweep on the same forecasts; report the **before-the-clamp** column; where the clamp binds, the method has no answer.
5. `show_agents(artifacts)`: all timelines; search audit table rolled across agents (shared queries = shared retrieval); counts of prompts containing the market price and of results carrying publication dates; `replay_llm_calls(trace, agent, content_chars=None)` for raw payloads.
6. `RunTrace.capture(question, parameters, artifacts, aggregation, llm_calls=...)`; `.save()` → `forecast_traces/`.

### Adversarial debate (`24_autonomous_agents/07_adversarial_debate`, `DebateAgent` in `agent_specialists.py`)

- Inputs: `question`, `agent_summaries` (text summaries, not underlying evidence), `aggregate_p_yes` (anchor from `AggregationResult`; if `None`, fall back to the plain mean; test with `is None`, never `or`).
- Rounds: bull then bear. Round 1 bull argues without seeing the bear (`{bear_section}` empty); later bull prompts carry the bear's previous argument and probability; the bear always sees the current bull argument. Each turn returns `(argument, probability, evidence: list[str], TokenUsage)` parsed from JSON.
- Stop rule: after each round stop if `|p_bull − p_bear| <= consensus_threshold`; else continue to `max_rounds`.
- `DebateAgent(llm, max_rounds=3, consensus_threshold=0.05).run(question, agent_summaries, aggregate_p_yes) -> DebateArtifact` (rounds of `DebateRound`).
- Fold-in: `post_debate = (1 − DEBATE_WEIGHT)·aggregate + DEBATE_WEIGHT·midpoint`, `midpoint = (p_bull_final + p_bear_final)/2` (inference: the text says the blend "moves by that weight times the distance between the midpoint and the aggregate", which is equivalent). Weight is low (0.3) because the aggregate rests on three evidence-gathering runs and the midpoint on two prompts arguing from summaries.
- Worth-it test, two independent conditions: (a) panel spread (max − min) `>= MIN_PANEL_DISAGREEMENT=0.05`; (b) gap closure `gap_round1 − gap_final >= MIN_GAP_CLOSURE=0.05`. Judge on the gap, never on blend movement.
- Gap figure: shaded band between bull and bear per round, dotted line = pre-debate aggregate. A band of constant width means nothing reconciled even if both sides drifted together.
- Settings: `DEBATE_ROUNDS=3`, `CONSENSUS_THRESHOLD=0.05`, `DEBATE_WEIGHT=0.3`, `MIN_PANEL_DISAGREEMENT=0.05`, `MIN_GAP_CLOSURE=0.05`, `NEYMAN_CORRELATION=0.3`, `N_AGENTS=3`, `MAX_STEPS=5`. `show_debate_transcript(...)` in `agent_observability`.

### Supervisor and confidence-gated blend (`24_autonomous_agents/08_forecasting_pipeline`, `SupervisorAgent`)

- Three phases, free functions threaded by a thin class: (1) `_supervisor_identify_disagreements(llm, agent_summaries, max_queries) -> (disagreements, queries, TokenUsage)`; (2) `_supervisor_run_searches(search, queries, max_search_results, cutoff_date)` executes only the bounded query list via `ToolExecutor` with the question's cutoff on every request; `_format_supervisor_evidence` renders dates and URLs visibly; (3) `_supervisor_finalize(llm, question, agent_summaries, search_results) -> (p_yes|None, confidence|None, rationale|None, TokenUsage)`, probability parsed and clamped; invalid confidence label → `"medium"` (never an override).
- `SupervisorAgent(llm, search=None, max_queries=3, max_search_results=5).run(question, agent_summaries, cutoff_date=None) -> SupervisorArtifact` (`p_yes`, `confidence`, `rationale`, disagreements, queries, search results). Prompts are the AIA Forecaster production templates.

```python
final_p, conf = post_debate, CONFIDENCE_WHEN_IGNORED        # 0.5
if sup.p_yes is not None and sup.confidence == "high":
    final_p, conf = sup.p_yes, CONFIDENCE_WHEN_OVERRIDDEN   # 0.8
elif sup.confidence == "medium" and sup.p_yes is not None:
    final_p = (1 - medium_weight) * post_debate + medium_weight * sup.p_yes
    conf = CONFIDENCE_WHEN_BLENDED                           # 0.6
return max(0.01, min(0.99, final_p)), conf
```

`SUPERVISOR_MEDIUM_WEIGHT=0.4`; low → ignored. The 0.8/0.6/0.5 scalars order "how much reconciliation the forecast received"; nothing estimates them.

### AIAForecaster pipeline (`24_autonomous_agents/08_forecasting_pipeline`)

| Stage | Component | Output |
|---|---|---|
| 1 Research | `_run_research_agents`: N identical `ResearchAgent`s | `list[AgentForecastArtifact]` (filter `forecast_produced`) |
| 2 Aggregate | `neyman_extremize(..., correlation=0.3)` | aggregate (carried value 1) |
| 3 Debate | `DebateAgent` | midpoint → post-debate blend (carried value 2) |
| 4 Supervise | `SupervisorAgent` + `_blend_final_probability` | final p + confidence (carried value 3) |

- `AIAForecaster(llm, search=None, n_agents=3, max_steps=5, debate_rounds=3, consensus_threshold=0.05, correlation=0.3).forecast(question: ForecastQuestion) -> ForecastResult`. Orchestration lives in `_forecast_one(forecaster, question)` outside the class, so the class is a configuration object. Declare every constant that moves the final number in one settings cell.
- Probability-path chart: line joins the carried values (aggregate → post-debate → final); open markers at the stage that read them (market price shown to agents, per-agent answers, debate midpoint, supervisor p). Gap between carried values = movement; gap between research agents = disagreement.
- Replay rule: post-debate is not stored; rebuild it with the weights the capture used. `carried_probabilities(result, run) -> (aggregate, midpoint, post_debate)` re-derives the final from recorded parts and **raises** if fallback weights do not reproduce it. Live path isolated in `_run_live_questions(questions)`; `_load_pinned_questions()` creates no provider or search client.
- Output per question: one `RunTrace` JSON plus a flat record (forecast, cutoff, `resolved_outcome` (empty at capture), intermediate stages) for a scoring pipeline.

### Scoring and calibration evaluation (`24_autonomous_agents/09_evaluation_and_governance`)

| Measure | Definition | Reading |
|---|---|---|
| Brier | mean (p − y)², y ∈ {0,1} | proper, bounded [0,1], lower better |
| Log score | mean −[y log p + (1−y) log(1−p)] | proper, unbounded on confident error, lower better |
| ECE | count-weighted mean over bins of \|mean forecast − observed frequency\| | lower better; zero by always forecasting the base rate |
| Sharpness | mean \|p − 0.5\| | never a quality alone; read against calibration |
| Reliability diagram | observed frequency vs mean forecast per bin, counts printed on every bin | above diagonal = under-confidence, below = over-confidence |

- Key records by `_question_id(question)` = hash of exact question text; assert alignment before arithmetic; never match by row position. `_build_result(q, agent_probs)` folds three probabilities with the same Neyman call as NB08 (`NEYMAN_CORRELATION=0.3`).
- Calibration-exponent fit twice: in-sample (fit on all ten, score the same ten) vs leave-one-out `_leave_one_out_transform(forecasts, resolved)` (fit k on the other nine, transform the held-out row). Transform `logit(p') = k·logit(p)`, `EXPONENT_RANGE=(0.5, 3.0)`; lower bound = floor on compression toward even odds, upper = ceiling on push toward certainty. Read `at_search_boundary` before the exponent; the spread of the ten fitted exponents is as informative as the score (stable = panel agrees on the correction; swinging = chasing rows).
- Compare configurations on all four measures: Neyman ensemble, plain mean, one agent alone, ensemble after LOO transform; rows differ only in the rule.
- Frozen-evidence replay: a search client that replays saved queries/documents from the execution log hands a second run the same evidence; whatever differs is model/prompt/aggregation and attributable. Holds evidence fixed, not truth: scoring still needs resolved questions and pre-resolution forecasts.

### Warden proxy and injection scan (`24_autonomous_agents/09_evaluation_and_governance`)

- `WardenPolicy(NamedTuple)`: `name: str`, `check: Callable[[tool_name, args], tuple[bool, str]]`. `Warden(policies).check(tool_name, args) -> (allowed, reason)` runs all policies before `ToolExecutor` executes anything.

| Policy | Rule |
|---|---|
| `no_write` | fail-closed allowlist `read_only_tools = {"search"}`; any other tool denied until reviewed and added |
| `domain_allowlist` | `{"sec.gov", "federalreserve.gov", "bls.gov"}`; hostname equals a domain or ends with `.{domain}` (subdomains inherit) |
| `rate_limit` | `_make_rate_limit_policy(limit=2)` stateful counter (production: per agent, reset on a time window) |

- `inspect_untrusted_input(text) -> list[str]` runs `_detect(text, patterns, label)` over three families; **any detection is fail-closed** (payload never reaches an LLM or tool): `_ROLE_OVERRIDE_PATTERNS` (`ignore (all)? previous instructions`, `you are now a`, `system: you`, `forget (everything|all|your)`, case-insensitive); `_TOOL_INJECTION_PATTERNS` (JSON containing `"action"` and `"execute_trade"`, `call function `, `run command `); `_EXFILTRATION_PATTERNS` (`send (this|all|the) ... (to|via)`, `upload (to|this)`, `forward (to|this)`). Scanning is a cheap independent filter; the Warden is the control. Map to OWASP LLM01/02/04/06/07/08 (LLM03/05/09/10 not visible in the notes).

### Framework comparison (`24_autonomous_agents/10_framework_comparison`, ~14 min)

| Variant | Shape | State lives in | Observability | When it earns its place |
|---|---|---|---|---|
| A Native | `native_sdk_pipeline(question, llm, search) -> dict`, one function, four phases | local variables | wrap `llm` in `TracingLLMClient`: full audit at no orchestration cost | default start |
| B CrewAI | `Agent(role, goal, backstory, llm=LLM(model="anthropic/claude-sonnet-4"))`, `Task(description, expected_output, agent, context)`, `Crew(agents, tasks, process=Process.sequential).kickoff()`; read `task.output.raw`, parse JSON by hand | framework | drives LiteLLM directly, so the chapter tracer is blind; result pinned as dict | named personas with distinct charters |
| C LangGraph | `ForecastState(TypedDict, total=False)`; nodes `_node_research/_aggregate/_debate/_supervise` delegate to the same specialist classes; `StateGraph` → `compile()` → `invoke({"question": q})` | typed state; `app.get_state(thread)`; `BaseCheckpointSaver` | nodes read module-level `llm`; rebind to a traced client before invoking | mid-run crash recovery, checkpointing, conditional/parallel edges |

- Settings: `N_AGENTS=3` (only setting all three read), `DEBATE_ROUNDS=2`, `SUPERVISOR_QUERIES=2`, `MAX_STEPS=5`. CrewAI needs one of `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`/`GOOGLE_API_KEY`/`OPENROUTER_API_KEY` (LiteLLM prefix strings like `openrouter/anthropic/claude-sonnet-4`), else the live path skips it; it is a role-prompted approximation that does not share the chapter's search tools.
- Metric: `_orchestration_statements(fn)` = `ast.parse(inspect.getsource(fn))` statement-node count in the function body (immune to formatting; LangGraph includes builder + nodes; CrewAI includes agent/task helpers). Elapsed via `time.perf_counter`. Only the statement count supports a comparison; final probabilities across variants are not a benchmark.

### Research operator (`24_autonomous_agents/11_research_operator`, `research_operator.py`)

- Loop: `import research_operator as ro; ro.run_operator()` selects an OpenAI-compatible endpoint (`_select_openai_endpoint()`: `RESEARCH_OPERATOR_MODEL` explicit, else `OPENAI_API_KEY`, else `OPENROUTER_API_KEY`), dispatches `TOOL_SCHEMAS`, records every call to a trace, ends on `done(summary)`. ~880-line orchestrator.
- Ten tools: generic 7 — `query_registry(sql)` (SQLite run-log registry), `read_file(path, max_bytes=80_000)`, `edit_file(path, old_text, new_text)`, `write_file(path, content)`, `run_bash(command, cwd=None)` (sets `ML4T_OUTPUT_DIR` to the sandbox, `shell=True`, timeout `BASH_TIMEOUT_S`), `read_parquet(...)` (signature partially visible), `write_parquet(path, rows)`; skills 2 — `list_skills(category=None)` (one-line summary per `SKILL.md`), `read_skill(name_or_path)`; terminator 1 — `done(summary)`. Functions `tool_query_registry`, `tool_read_file`, ..., `tool_done`.

| Config | Default |
|---|---|
| `RESEARCH_OPERATOR_CASE_STUDY` | `"etfs"` |
| `RESEARCH_OPERATOR_SKILLS_ROOT` | `<code_repo>/../skills` |
| `RESEARCH_OPERATOR_CODE_REPO`, `RESEARCH_OPERATOR_ENV_FILE`, `RESEARCH_OPERATOR_MODEL` | — |
| `RESEARCH_OPERATOR_MAX_CALLS` | 60 |
| `RESEARCH_OPERATOR_BASH_TIMEOUT` | 600 s |
| `PRICE_IN_PER_MTOK` / `PRICE_OUT_PER_MTOK` | 0.50 / 1.50 (DeepSeek v4 Pro via OpenRouter, May 2026); ~$1 per live run |
| `CASE_STUDY_TASKS: dict[str, str]` | one §20.9 task paragraph per case study |

- Skills: `git clone https://github.com/ml4t/skills` next to the code repo; each `SKILL.md` = problem statement → WRONG/CORRECT example → `## Production Implementation` naming the `ml4t-*` function; new skills under `~/ml4t/skills/{category}/` with standard frontmatter are picked up at runtime; missing repo → `list_skills`/`read_skill` return a hint, not an error.
- Reporting discipline: parse numbers out of the trace (`_matched_comparison(trace)` parses the fixed per-model block the operator's script printed; `_summary_table(summary)` parses the agent's markdown table); a trace missing the block fails loudly. `_run_cost(run)` for cost.
- Reading a run: tool-call histogram should show inspect → read skills → write/edit/run loop; mostly reading = never started; mostly running = stopped checking; read the `done()` summary against the evidence scope (validation vs holdout).

## Guardrails and pitfalls

- **Budget exhaustion treated as a forecast** — `run_react_agent` returns `0.5, "Max steps reached"`; averaged in, a silent agent pulls the panel toward even odds / filter on `forecast_produced` before any mean or Neyman; retry or abstain.
- **Unvalidated model output reaching a tool** — malformed JSON, non-finite or boolean `p_yes`, unknown actions corrupt runs / `parse_json` + `validate_action`; clamp; return errors for correction (costs a turn).
- **Lookahead via search (hindsight)** — searching the open web for a resolved question returns the answer; a perfect score measures the index / `cutoff_date` in the `SearchClient` contract (drop published ≥ cutoff) plus the consistency gate at the other end (fixtures, checkpoints and new tools can bypass the first).
- **Undated results counted as point-in-time safe** — the filter can only exclude what it can date / partition dated vs undated, count both, decide policy in the evaluation. Chapter captures had 0 dated results of 40 (NB04) and none (NB06): demonstrations, not backtests.
- **Market price leakage into prompts** — an agent shown the market probability copies it; closeness is circular / forbid price lookup in the system prompt; assert against saved prompts; withhold `market_price` when evaluating against the market; never report "beat the market" from NB06/08 runs.
- **Testing on questions in training data** — tests recall (Lopez-Lira) / open prediction-market questions or contamination-aware resolved sets with cutoffs.
- **Source policy enforced only in the prompt** — models read disallowed sources anyway / `apply_domain_policy` post-retrieval in `ToolExecutor`; log `n_blocked` so a thin result list is attributable.
- **Trusting the reasoning trace over the execution log** — the trace is the model's account / keep `execution_log` separate; diff the two when debugging.
- **Tool unavailable aborts the run** — nothing left to audit / `status="disabled"`, return `[]`, continue with a visible gap.
- **Message history as the only state / pickle checkpoints** — cannot be queried, diffed, resumed or gated; pickle needs the exact codebase and silently drops fields / typed `AgentState`, JSON via `asdict`, verify round-trip.
- **Base-rate neglect and thin evidence** — a model given only current reporting follows the narrative; one search clears nothing / coverage gate requires `base_rate` type and `min_items=3`; abstain and record which gate refused.
- **Stale run vs stale documents conflated; future retrieval timestamps** — a run that fetched this morning's news three days ago is stale regardless of publication dates; a future timestamp means a wrong clock or fixture / freshness gate on retrieval time (24h, fails on future), consistency gate on publication date; keep both.
- **Inferring evidence type from a log after the fact** — coverage fails structurally and looks like an agent fault / classify each result as it arrives and write to state.
- **Derived fields read as measurements** — confidence = 2|p−0.5| rewards hallucinated certainty; evidence quality counts documents (20 copies of one wire story = HIGH) / label as heuristics; measure calibration on resolved forecasts.
- **Duplicate sources as independent confirmation** — allowlist is a publisher proxy; syndicated copies are one piece of evidence / nothing in the chapter dedups; add dedup of the underlying source before counting.
- **Extremization with an assumed ρ; clamp carrying the answer; spread ignored** — sweeping ρ moves the aggregate further than agents are apart; Neyman can map outside [0,1] and the floor is the clamp, not the method; 20/65/90 and three near-identical forecasts give the same aggregate / report the mean; present Neyman as sensitivity with the before-the-clamp column unless ρ is estimated; use spread as a debate trigger.
- **Extremizing agreement from shared prompts** — on the recession question all three agents returned the same probability and Neyman pushed the aggregate below every one of them / check for shared prompts/evidence before trusting `NEYMAN_CORRELATION=0.3`.
- **Confidence-weighted aggregation** — weights are the extremity heuristic, not skill / unvalidated unless weights come from track records.
- **Calibration fitted and evaluated on the same data; boundary optimum** — reports the improvement it was built to produce; a grid stopping at the range edge (0.5 in NB09) is a clamp / fit → assert interior (`at_search_boundary`) → freeze → score held-out; LOO at small n; report exponent spread.
- **Reliability bins with tiny counts** — outer bands with a handful of questions say nothing / print counts per bin; 4 bins per 10 or 80 questions is already coarse; do not estimate calibration from ten questions.
- **Sharpness as a quality; log vs Brier mismatch** — any push to the ends raises sharpness; log is unbounded on confident error / read sharpness against calibration; pick the rule by the cost of a confident error.
- **Post-resolution probabilities and silent record misalignment** — NB09 inputs were chosen after resolution, so no score measures a forecaster; pairing a forecast with someone else's outcome gives wrong scores no plot reveals / score only pre-resolution forecasts; key by question-text hash and assert alignment.
- **Blend movement mistaken for debate effect; midpoint hides disagreement; debate on an agreeing panel; fixed round count** — the blend moves by `DEBATE_WEIGHT·(midpoint − aggregate)` by construction; far and close pairs share a midpoint / judge on gap closure ≥ 0.05; always show the gap band; skip if spread < 0.05; stop at gap ≤ 0.05, cap 3 rounds.
- **Same-model "independence"** — bull and bear cannot check each other's claims against raw evidence / treat debate as a stress test, never a second opinion.
- **`or` fallback on a probability** — `aggregate_p_yes or mean` falls through on a legitimate 0.0 / test `is None`.
- **Unconditional supervisor override; replay with current weights** — one second opinion outranks three agents; rebuilding post-debate with new weights draws a correction that never happened / gate on stated confidence; read weights from the trace; `carried_probabilities` raises on mismatch.
- **Reading stage movement as refinement** — aggregation assumes unmeasured independence, debate shrinks disagreement not error, supervisor confidence is self-asserted / draw the whole path; score only against resolved questions.
- **Context window growth** — the NB01 loop keeps every message with no eviction / budget `MAX_STEPS` and `MAX_RESULTS=5`; pass summaries (`format_agent_summary`) to later stages.
- **Non-thread-safe clients in parallel panels** — shared client corrupts calls / verify thread safety or construct per worker; one tracer per agent.
- **One capture read as an experiment; agreeing agents = one observation** — one question, three agents: no error bar / repeated trials; vary temperature/retrieval/wording separately; read timelines and the shared-query audit table before trusting any aggregate.
- **Policy in the prompt; blocklist tool policy; injection scan as defence** — agents told not to write files sometimes do; a tool nobody thought of passes; a handful of regexes against an adapting threat / Warden proxy with fail-closed allowlist (`{"search"}`), explicit review to add; scan fail-closed as an extra filter.
- **Comparing configurations on live search; framework owns the client; forecast agreement as framework benchmark** — index moves between runs; CrewAI bypasses the tracer; variants differ in prompts and samples / replay from frozen logs; keep one `LLMClient` boundary; compare AST statement counts only.
- **Live operator executes arbitrary shell** — `shell=True` at host privileges; `ML4T_OUTPUT_DIR` and directory allowlist are conveniences for cooperative models, not a sandbox (escape via `> ~/anything`, `rm -rf`, network egress) / run live only in a container or firejail with restricted filesystem and network; `RUN_LIVE=False` is the only fully safe path.
- **Agent overclaiming scope** — ETFs summary called a validation-window experiment a holdout conclusion / keep the raw artifact unedited; a human restates the result within the evidence actually produced before anyone acts.
- **HAC lag shorter than label overlap** — operator used 5-lag HAC on IC t-stats with 21-day forward-return labels; registry baseline uses 20 / set HAC lags ≥ label horizon − 1; compare point estimates only when lags differ.
- **Signal filter mistaken for retrained model; fundamental-law misuse** — a market-cap-quartile filter changes eligible names, not the model (turnover barely moves); IR = IC·sqrt(breadth) holds IC fixed / label as capacity sensitivity; retrain on the filtered universe for a real answer; treat moved IC and drawdown as separate empirical outcomes.
- **Second copy of numbers; turn histogram over-interpreted; pricing staleness** — retyped results drift; counts reflect the tool surface; rates move / parse the trace programmatically and fail if missing; compare runs on the same surface; change `PRICE_*_PER_MTOK`, not the arithmetic.
- **Cost and latency** — live runs cost per agent; most wall-clock is network / `complete_with_usage` for token accounting; per-call timeout (Tavily 30s); replay by default; stop adding agents past the 1/ρ bend.

## Decision rules and defaults

| Decision | Rule |
|---|---|
| Use an agent at all | Only when evidence acquisition / structured judgment is the bottleneck; otherwise conventional pipeline |
| Reasoning pattern | ReAct by default; ToT only for branch-heavy decisions; Reflexion only with explicit memory governance |
| Step / result budgets | `MAX_STEPS=5` (raise for open models that search more before committing); `MAX_RESULTS`/`MAX_SEARCH_RESULTS=5`; Tavily timeout 30s |
| Panel | `N_AGENTS=3`; effective size d² saturates at 1/ρ, so add diversity (model, index, framing) not agents |
| Provider | Order Anthropic → OpenAI → Google → OpenRouter → Ollama; `LLM_PROVIDER=mock` for CI; `RUN_LIVE=False` replay by default |
| Cutoff | Drop results published on or after `cutoff_date`; undated → separate class, policy decided by the evaluation |
| Domain policy | Blocklist first, then allowlist; empty set = no constraint; start from `DEFAULT_ALLOWED_DOMAINS` |
| Gates | Coverage (types before count, `min_items=3`) → freshness (24h; widen for quarterly filings) → consistency; any failure → gather more or `synthesis_status="abstained"` and record which gate refused |
| Derived fields | Sentiment thresholds 0.8/0.6/0.4/0.2; evidence quality HIGH ≥10 sources & ≥3 queries, MEDIUM ≥5 sources or ≥2 queries; confidence model-supplied else 2\|p−0.5\| (3 dp) |
| Before aggregating | (1) filter `forecast_produced=True`; (2) read timelines for shared retrieval; (3) report the mean; (4) Neyman as ρ-sweep with before-the-clamp values; (5) weighted Neyman only with defensible weights |
| Extremization go/no-go | Only when ρ comes from an estimated dependence model; otherwise leave it out. Pipeline default `NEYMAN_CORRELATION=0.3`; d clamp [1,3]; result clamp [0.01, 0.99] |
| Calibration protocol | Fit exponent by grid search on training resolved forecasts (`EXPONENT_RANGE=(0.5, 3.0)`) → assert interior → freeze → score held-out Brier + reliability bins with counts; LOO at small n; report exponent spread |
| Debate go/no-go | Run only if panel spread ≥ `MIN_PANEL_DISAGREEMENT=0.05`; productive only if closure ≥ `MIN_GAP_CLOSURE=0.05`; stop at gap ≤ `CONSENSUS_THRESHOLD=0.05`; cap `DEBATE_ROUNDS=3` (NB10 uses 2) |
| Fold-in weights | `DEBATE_WEIGHT=0.3` (convention, not fitted); `SUPERVISOR_MEDIUM_WEIGHT=0.4`; high → replace, medium → blend, low → ignore, invalid → medium; final clamp [0.01, 0.99]; confidence scalars 0.8 > 0.6 > 0.5 are ordering only |
| Supervisor | `max_queries=3` (NB10 uses 2), `max_search_results=5`, cutoff applied to every search |
| Replay | Read weights from the trace and assert reproduction; never overwrite pinned traces; always write `RunTrace` JSON on live runs |
| Scoring checklist | (1) forecasts timestamped before resolution; (2) records keyed by question-text hash with alignment asserted; (3) Brier and log (choose by cost of confident error), ECE, sharpness together; (4) reliability diagram with per-bin counts (`RELIABILITY_BINS=4` per 10 questions); (5) LOO calibration at small n; (6) `at_search_boundary` before quoting an exponent |
| Capture validity | Trace file is the expected one; count results with publication dates; count prompts containing the market price. No dates or price present → demonstration, not backtest |
| Security | Warden proxy with fail-closed allowlist (`search` only), domain allowlist (`sec.gov`, `federalreserve.gov`, `bls.gov`), rate limit (2 in demo); injection scan fail-closed; publication cutoffs; no order path; no secrets in prompts; gates and abstention |
| Framework | Start native; LangGraph for crash recovery / checkpointing / conditional or parallel edges; CrewAI for named personas; first ask where run state lives and whether one client wrapper still captures every call |
| Operator live run | Container/firejail with restricted FS and network; `OPENROUTER_API_KEY` or OpenAI-compatible endpoint; `RESEARCH_OPERATOR_MAX_CALLS=60`; `BASH_TIMEOUT=600 s`; skills repo cloned or `RESEARCH_OPERATOR_SKILLS_ROOT` set; expect ~$1 |
| Promotion | Promote a pattern only after a validated improvement; the trace is the design record for upgrading a case-study registry headline |
| Chapter fixtures | `CHAPTER_CONTESTED_QUESTION` ("Will the Federal Reserve hike rates in 2026?", `cutoff_date=""`) for disagreement/debate; `CHAPTER_CLEAR_QUESTION` (US recession by end of 2026; exact wording in `agent_fixtures.py`) for agreement; resolved fixtures carry cutoffs |

## Code patterns and APIs

Repo path: `24_autonomous_agents/` (notebooks `01`–`11` as `.py` Jupytext + `.ipynb`); traces under `forecast_traces/`, operator runs under `operator_artifacts/`. Run: `uv run python 24_autonomous_agents/<notebook>.py`; tests: `uv run pytest tests/test_chapter_notebooks.py -v -k "24_autonomous_agents"`; CLI smoke: `LLM_PROVIDER=mock uv run python 24_autonomous_agents/01_react_reasoning.py`.

| Module | Key names |
|---|---|
| `agent_providers.py` | `LLMClient` (`complete`, `complete_with_usage`), `TokenUsage`, `ChatMessage`, `create_llm_client(provider="")`, `MockLLMClient`, `TracingLLMClient` |
| `agent_tools.py` | `SearchClient`, `SearchResult`, `MockSearchClient`, `TavilySearchClient`, `format_search_results`, `apply_domain_policy`, `ToolExecutor` (`.execute_search`, `.execution_log`), `ToolExecution`, `ToolDefinition`, `DEFAULT_ALLOWED_DOMAINS` |
| `agent_schemas.py` | `AgentState` (`to_json`/`from_json`), `ForecastQuestion` (incl. `cutoff_date`), `ForecastResult`, `AgentForecastArtifact`, `AggregationResult`, `AgentTrace`, `RunTrace` (`.capture`, `.save`, `.load`), gate functions, `run_quality_gates` |
| `agent_research.py` | `parse_json`, `validate_action`, `extract_confidence`, `extract_sentiment`, `extract_key_findings`, `extract_uncertainties`, `assess_evidence_quality`, `rederive_fields`, `ResearchAgent`, `Sentiment`, `EvidenceQuality`, `format_agent_summary`, `build_step_prompt` |
| `agent_pipeline.py` | `clamp_prob`, `platt_scale`, `logodds_extremize`, `get_model_calibration_d`, `neyman_extremize`, `neyman_extremize_weighted`, `CalibrationResult`, `fit_extremization_exponent`, `brier_score`, `log_score`, `sharpness`, `expected_calibration_error`, `reliability_bins` |
| `agent_specialists.py` | `ResearchAgent`, `DebateAgent`, `SupervisorAgent`, `DebateArtifact`, `DebateRound`, `SupervisorArtifact` |
| `agent_observability.py` | `show_agent_timeline`, `show_agents`, `show_debate_transcript`, `replay_llm_calls(trace, agent, content_chars=None)` |
| `agent_fixtures.py` | `CHAPTER_CONTESTED_QUESTION`, `CHAPTER_CLEAR_QUESTION`, resolved questions with cutoffs, `get_live_question()` |
| `research_operator.py` | `run_operator`, `TOOL_SCHEMAS`, `tool_*` functions, `CASE_STUDY_TASKS`, `_select_openai_endpoint` |

```python
# Single research agent (withhold the market price when evaluating against the market)
llm = create_llm_client("")                                   # or LLM_PROVIDER env
executor = ToolExecutor(search=TavilySearchClient(key), allowed_domains=DEFAULT_ALLOWED_DOMAINS)
agent = ResearchAgent(llm, search=search_client, agent_id="agent_0", max_steps=5, max_search_results=5)
art = agent.run(question, market_price=None)
if art.forecast_produced:
    panel.append(art.p_yes)
```

```python
# Parallel panel, one tracer per agent, persisted trace
with ThreadPoolExecutor() as pool:
    futs = [pool.submit(run_agent, f"agent_{i}") for i in range(N_AGENTS)]
answered = [a for a in artifacts if a.forecast_produced]
agg = neyman_extremize([a.p_yes for a in answered], correlation=0.3)   # report mean first
trace = RunTrace.capture(question, artifacts, llm_calls=...); trace.save()  # forecast_traces/
```

```python
# Checkpoint restore and ablation
state = AgentState.from_json(path.read_text())
state.evidence = [e for e in state.evidence if e.type != "base_rate"]
results = run_quality_gates(state)                            # coverage fails -> abstained
```

```python
# Debate, supervisor, full pipeline
debate = DebateAgent(llm, max_rounds=3, consensus_threshold=0.05).run(question, summaries, aggregate_p_yes)
sup = SupervisorAgent(llm, search=search, max_queries=3, max_search_results=5).run(question, summaries, cutoff_date=q.cutoff_date)
result = AIAForecaster(llm, search, n_agents=3, max_steps=5, debate_rounds=3,
                       consensus_threshold=0.05, correlation=0.3).forecast(question)
```

```python
# Calibration (fit, assert interior, freeze, score held-out)
cal = fit_extremization_exponent(train_forecasts, train_outcomes, exponent_range=(0.5, 3.0))
assert not cal.at_search_boundary
held_out = [logodds_extremize(p, cal.exponent) for p in test_forecasts]   # attribute name inferred
```

```python
# Warden before ToolExecutor; injection scan fail-closed
warden = Warden([WardenPolicy("no_write", _no_write_policy),
                 WardenPolicy("domain_allowlist", _domain_allowlist_policy),
                 WardenPolicy("rate_limit", _make_rate_limit_policy(limit=2))])
allowed, reason = warden.check(tool_name, args)
if inspect_untrusted_input(text):
    refuse()
```

```python
# LangGraph variant; operator
graph = StateGraph(ForecastState); graph.add_node("research", _node_research); ...; app = graph.compile()
app.invoke({"question": q}); app.get_state(thread)            # BaseCheckpointSaver for persistence
import research_operator as ro; ro.run_operator()             # RESEARCH_OPERATOR_CASE_STUDY, CASE_STUDY_TASKS
```

Diagnostics the operator reached via skills: `ml4t.diagnostic.api.cross_sectional_ic_series`, `ml4t.diagnostic.api.compute_ic_hac_stats`; allocator named in the registry spec: `score_weighted_top_k`. Pinned CrewAI dict (NB10): `{"agent_probs": [0.25, 0.25, 0.25], "aggregate": 0.1577, "debate_midpoint": 0.335, "supervisor_p": 0.25, "supervisor_confidence": "medium", "final_p": 0.335}`, elapsed 70.3 s.

## Evidence from the book

- **NB01 replay** (2026-06-15, Claude Sonnet + Tavily, Polymarket 2026 rate-path question): the agent used all 5 turns searching, found reporting arguing both ways, never committed, returned the no-answer sentinel. Mock run commits after one search in 3 turns. Saved search results carried no publication dates.
- **NB02 demo**: NVIDIA Q4 FY2025 question, cutoff 2025-02-20 (reported 2025-02-26); mock results span a wire service, paywalled newspaper, message board; publisher allowlist and social blocklist keep different subsets; mock `duration_ms` ≈ 0.
- **NB03**: all three gates pass on the three-item fixture (two `search_results`, one `base_rate`, retrieved at as-of 2025-02-19). Engineered failures: missing `base_rate` → missing type; two records with both types → fails `min_items=3`; a result published 2025-02-27 → consistency; retrieval 48h before as-of → 24h freshness. Dropping base rate from the checkpoint → coverage fails → abstain.
- **NB04 pinned capture** (2026-06-09, "Will the Fed hike rates in 2026?"): 40 search results, 0 dated; 0 prompts contained the market price. Two agents of the same class reached opposite conclusions and disagreed about the current policy rate itself (at least one read something wrong). Reconstructed-state gates: freshness pass; consistency "nothing to check" (`cutoff_date=""`); coverage fails structurally.
- **NB05 (simulated, seeded)**: fitted exponent a > 1 (undoes shrinkage toward 0.5); held-out Brier improves only slightly; the two central bands hold nearly all 80 held-out questions and miss in opposite directions (under-confidence persists after calibration); outer bands hold a handful each. Effective panel size flattens toward 1/ρ. Sweeping ρ with a fixed mean moves the aggregate "a long way."
- **NB06 capture** (2026-06-09, `claude-sonnet-4`, Tavily, 3 agents, US recession by end of 2026, market price in every prompt): three different probabilities from identical agents; all rationales continuous prose (no key findings extracted); both Neyman variants sit further from the base rate than the mean with a small gap between them; under near-independence the formula returns a **negative** probability and the clamp floors it (flat sensitivity curve on the left); no result carried a publication date. Conclusion: report the mean; Neyman as sensitivity. AIA paper finding: identical agents with temperature diversity produce diverse outputs without role specialization.
- **NB07** (rate-hike question, one capture, three rounds): agents spread out on the contested question (vs landing close on the clear one in NB06); the debate narrowed the gap "by a few points", which is smaller disagreement, not more accuracy; no scoring establishes the blend beats the aggregate.
- **NB08**: the largest carried step fell on the post-debate blend for one question and on the supervisor blend for the other; no stage decided the answer and none was idle. On the recession question all three agents returned the same probability and the Neyman aggregate came out below every one of them. Both questions unresolved → `resolved_outcome` empty, unscoreable.
- **NB09** (synthetic, post-resolution inputs): the exponent grid search stopped at the lower bound 0.5 (inputs more extreme than outcomes support); every LOO fold hit the same bound, so in-sample and LOO Brier were identical, which says the fit had no room to overfit, not that the transform generalises. Across four configurations the four measures disagree about which to prefer; a transform toward the ends raises sharpness regardless of score. Warden demo blocked 4 calls (one non-allowlisted domain, one third allowed-domain search over the rate limit, two unapproved mutators); scanner blocked 3 of 5 payloads.
- **NB10** (pinned 2026-06-09 claude-sonnet): orchestration statement count CrewAI > LangGraph > Native. CrewAI run: agent probs all 0.25, aggregate 0.1577, midpoint 0.335, supervisor 0.25 at medium, final 0.335, 70.3 s. Final probabilities across variants are not a benchmark.
- **NB11 ETFs** (§20.9 ensemble of GBM + tabular DL + CAE, validation window only, May 4 2026 capture, DeepSeek v4 Pro ~$1): negative result. Ensemble per-fold IC swings from negative to strongly positive while the LSTM baseline's stays small and positive; `score_weighted_top_k` turns that instability into portfolio losses by sizing on score (Ch20's point that highest rank correlation ≠ highest Sharpe, with the allocator as the divergence point; the agent was not told this). Validation-only because only the LSTM has holdout predictions in the registry. IC t-stats used 5-lag HAC vs the registry's 20-lag baseline (21-day forward-return labels), so IC uncertainties are not comparable.
- **NB11 US firm characteristics** (§20.9 top-three-mcap-quartile filter): validation Sharpe eroded materially when the bottom market-cap quartile is removed (§20.1 small-cap clustering concern confirmed); IC and drawdown both moved; turnover barely moved (signal filter, not reconstruction); no market impact estimated; retraining on the filtered universe flagged as follow-up. Both operator runs reported negative results via `done()` rather than confabulating improvements; one loop, two case studies, only `RESEARCH_OPERATOR_CASE_STUDY` and `CASE_STUDY_TASKS` changed.
- Numeric values not in the notes: exact NB04/NB06/NB07/NB08 probabilities, NB09 Brier/log/ECE values, NB10 native/LangGraph counts and timings, NB11 Sharpe/IC values (parsed from traces at runtime).

## Related references

- `chapters/20_strategy_synthesis.md` — §20.9 next-step tasks the operator executes; §20.1 small-cap clustering; allocator as the IC-vs-Sharpe divergence point.
- `chapters/22_rag_financial_research.md` — long-term memory as retrieval across runs; FinDER-style evaluation.
- `chapters/23_knowledge_graphs.md` — structured evidence and provenance that an agent's tools can expose.
- `chapters/21_rl_execution_hedging.md` — the other agent chapter; contrast read-only forecasting with action-taking policies.
- `chapters/25_live_trading.md` — human approval boundaries, kill switches and the no-order-path rule once an agent output is operational.
- `chapters/26_mlops_governance.md` — observability, release discipline, drift and monitoring that §24.9 presupposes.
- `chapters/07_defining_the_learning_task.md` — point-in-time labels and cutoffs; the same lookahead logic as the publication-date filter.
- `chapters/11_ml_pipeline.md` — fit/freeze/evaluate discipline mirrored by the calibration protocol; multiple-testing caveats for repeated trials.
- `chapters/01_process_is_edge.md` — the trace-as-deliverable and negative-results-first-class stance.
- `case_studies/etfs.md` — NB11 ETF ensemble operator run (validation-only, negative result).
- `case_studies/us_firm_characteristics.md` — NB11 market-cap-quartile filter run.
- `libraries/ml4t_diagnostic.md` — `cross_sectional_ic_series`, `compute_ic_hac_stats` (HAC lags ≥ label horizon − 1).
- `libraries/ml4t_backtest.md` — registry backtest specs and `score_weighted_top_k` allocator the operator re-ran.
- `libraries/ml4t_data.md`, `libraries/ml4t_engineer.md` — libraries named in the skills corpus the operator consults.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `workflow.md`, `companion_repo.md` — cross-cutting indices that collect this chapter's rules.
- Further reading:
  - Alur et al. (2025) AIA Forecaster technical report (arXiv 2511.07678); Yao et al. (2023) ReAct (2210.03629) and Tree of Thoughts (2305.10601); Shinn et al. (2023) Reflexion (2303.11366); Wei et al. (2023) Chain-of-Thought (2201.11903).
  - Lopez-Lira (2025) "Can LLMs Trade?" (2504.10789) and "The Memorization Problem" (SSRN 5217505); Ang et al. (2026) Self-Driving Portfolio (2604.02279); Korinek (2025) AI Agents for Economic Research (NBER w34202).
  - Choi et al. (2025) FinDER (2504.15800); Xie et al. (2024) FinBen (NeurIPS); Zhao et al. (2025) AlphaAgents (2508.11152); Yu et al. (2024) FinCon (2407.06567) and (2025) FinMem (IEEE TBD).
  - Fabozzi & López de Prado (2025) JPM; Kong et al. (2024) JPM; Lee et al. (2025) "Your AI, Not Your View" (2507.20957); Aldridge et al. (2025) Agentic AI in Finance survey; Man Group, AI Agents & Trend; Li et al. (2025) System 1→2 survey (2502.17419); Fang & Moore (2025); OWASP Top 10 for LLM Applications (2025).
  - Book-only topics not in the notebooks: §24.4 Model Context Protocol and sandboxing; §24.3 vector stores; §24.7 Condorcet jury theorem and prediction-market design; §24.6 agent complexity vs calibration trade-off.

## Glossary

- **ReAct** — loop alternating reasoning and tool actions; here two actions (search, forecast) until a forecast or the step budget.
- **Tree of Thoughts / Reflexion** — branching deliberate search (branch-heavy decisions only) / verbal self-reflection stored across attempts (needs memory governance).
- **Prediction-market question** — contract paying out if an event occurs by a date; price = market probability.
- **Cutoff date** — last date search may return documents from; results published on/after it are dropped.
- **Provenance** — origin record of evidence (URL, publisher, publication date).
- **Execution log** — `ToolExecutor` record of what ran (query, status, duration, `n_blocked`); separate from the reasoning trace.
- **Reasoning trace / `AgentTrace`** — per-step record of the model's actions and observations.
- **`RunTrace`** — outside-view JSON record: question, prompts, replies, results, artifacts, aggregation.
- **`AgentState`** — inside-view typed state: evidence, open questions, tool trace, gate outcomes, synthesis status.
- **Quality gate** — code-level contract on evidence checked before synthesis (coverage, freshness, consistency).
- **Abstention** — `synthesis_status="abstained"`; refusing to forecast when gates fail.
- **Base rate** — historical event frequency; required evidence type against base-rate neglect.
- **Working / short-term / long-term memory** — context window / durable run state / cross-run retrieval (RAG).
- **Checkpoint** — JSON serialization of `AgentState` for resume, diff and ablation.
- **`forecast_produced`** — artifact flag distinguishing a committed probability from the budget-exhaustion fallback (0.5, "Max steps reached").
- **Derived fields** — confidence, sentiment, key findings, uncertainties, evidence quality; heuristics from `p_yes` and rationale.
- **Neyman extremization** — push the panel mean away from the base rate by d = sqrt(n/(1+(n−1)ρ)); d² = effective panel size.
- **Herfindahl index** — Σw²; reciprocal = effective number of equally weighted forecasters.
- **Platt scaling / log-odds extremization** — p' = d·p^a/(d·p^a+(1−p)^a); with d=1, logit(p') = a·logit(p).
- **Extremization exponent / `at_search_boundary`** — k in `logit(p') = k·logit(p)`, grid-searched over (0.5, 3.0); flag that the fit sits on the range edge (a clamp, not a minimum).
- **Brier / log score / ECE / sharpness** — mean squared error (proper, bounded) / negative log-likelihood (proper, unbounded on confident error) / count-weighted binned calibration gap / mean |p − 0.5| (never a quality alone).
- **Reliability bins / diagram** — equal-width probability bands with count, mean forecast, observed yes share; diagonal = calibrated.
- **Leave-one-out** — fit on n−1 rows, score the held-out row; the only honest option at n=10.
- **Source policy** — allow/block domain sets applied post-retrieval in the tool layer.
- **`TracingLLMClient`** — wrapper capturing every prompt/response at the LLM boundary, one per agent.
- **Adversarial debate / gap / midpoint / consensus threshold** — bull and bear prompts of one model over bounded rounds; gap = |p_bull − p_bear|, closure = gap_round1 − gap_final; midpoint blended at `DEBATE_WEIGHT`; debate stops below 0.05.
- **Supervisor / confidence-gated override** — identifies disagreements, runs bounded cutoff-respecting searches, returns p_yes + confidence; replaces at high, blends at medium (0.4), ignored at low.
- **Carried values** — aggregate, post-debate blend, final: the same quantity at three pipeline points.
- **Frozen-evidence replay** — search client replaying saved queries/documents so only model/prompt/aggregation vary.
- **Warden / fail-closed allowlist** — proxy between agent and `ToolExecutor` enforcing policies on every call; deny anything not explicitly approved.
- **Prompt injection** — adversarial text in processed inputs that hijacks LLM behaviour; scanned with role-override, tool-injection and exfiltration regex families.
- **Statement count** — AST count of statement nodes in a pipeline function; the only comparable framework metric here.
- **Operator** — thin loop giving an LLM general tools (files, bash, SQL, parquet, skills); domain logic lives in skills and `ml4t-*` libraries.
- **Skill (`SKILL.md`)** — concept-first doc with WRONG/CORRECT example and a `## Production Implementation` pointer to an `ml4t-*` function.
- **Registry** — SQLite run-log of case-study models, predictions and backtest specs queried by the operator.
- **HAC adjustment** — heteroskedasticity-and-autocorrelation-consistent standard errors; lag count must span label overlap.
