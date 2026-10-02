# Chapter 10: Text Feature Engineering

> Chapter 10 climbs the representation ladder — lexicons, bag-of-words and TF-IDF (10.1); Word2Vec/GloVe static embeddings and their extension to asset embeddings learned from 13F holdings (10.2); RNN/LSTM sequence models (10.3); and transformer encoders such as FinBERT, DeBERTa, ModernBERT and Sentence-BERT (10.4) — and then argues that none of it is a tradable signal until timestamps, entity resolution, revisions, aggregation rules, model training cutoffs and evaluation protocols are all point-in-time safe (10.5). The nine notebooks carry the chapter's empirical lessons: co-occurrence measures topic, not polarity; a TF-IDF baseline costs seconds and sets the bar; a published checkpoint may have been trained on your test set; and sentiment / narrative-surprise factors built correctly from FNSPID headlines and 10-Q MD&A are, on the chapter's samples, indistinguishable from zero. The position is "build, then evaluate, then decide": a construction step that raises a statistic has not thereby worked, and a negative result is a result.

## When to use this reference

- Building sentiment, narrative-surprise, topic-exposure or entity-count features from news headlines, wire stories, SEC filings (10-Q/10-K MD&A), transcripts or analyst text.
- Training or choosing word embeddings (Word2Vec, GloVe) or asset embeddings from portfolio holdings (13F "portfolio sentences").
- Choosing, scoring or fine-tuning a transformer checkpoint (FinBERT variants, DeBERTa, ModernBERT, MiniLM sentence encoders) for financial classification or embedding.
- Fine-tuning token classification (NER) with subword tokenizers, or extracting structured entities/amounts/dates from text.
- Applying a sentiment model trained on one corpus to another and diagnosing the accuracy drop (text shift vs labeling standard vs target shift).
- Aligning any text-derived signal to returns: availability timestamps, 1-day lag, forward as-of joins, wire-service dedup, entity resolution, lookback windows, model training cutoffs.
- Evaluating a text signal with daily cross-sectional IC / ICIR / t-stat / quintile long-short, or with pooled IC + cluster bootstrap when the cross-section per date is thin.
- Auditing an NLP pipeline for leakage, test-set contamination, label-order mismatch, negator deletion, or non-deterministic training.
- Deciding which Docker image / runtime budget a chapter notebook needs (`ml4t-py312` for gensim, `ml4t-gpu` for transformers, `ml4t` for CPU diagnostics).

## Core ideas (the why)

**The representation ladder: each rung is defined by what it discards.**

| Representation | Preserves | Loses | Finance verdict |
|---|---|---|---|
| Lexicons, bag-of-words, TF-IDF (10.1; Loughran-McDonald, BM25) | Vocabulary counts; fast, interpretable, domain-adaptable | Context, synonymy, negation, meaning that changes across uses | Still the right first baseline; not enough alone |
| Static embeddings, Word2Vec/GloVe (10.2) | Co-occurrence topic structure; dense similarity | Polarity direction, word sense, OOV tokens, any unit above the word | Intermediate step; one vector per word is not enough for finance |
| RNN/LSTM (10.3) | Order, some negation scope | Long-range dependence, parallel computation, scaling | Historical bridge; the bottlenecks drove the move to attention |
| Transformers / BERT-style encoders (10.4) | Context-dependent token vectors, sentence vectors, long-range dependence | Nothing above 512 tokens without chunking; needs a labeled head for polarity | Practical default for classification, embeddings, extraction |
| PIT-safe feature workflow (10.5) | Timestamps, entity resolution, revisions, aggregation rules, model cutoffs, evaluation protocol | — | What turns a benchmark score into a research/production signal |

- **Distributional hypothesis (Firth 1957) defines "similar" as "used in the same places", not "means the same".** When a corpus writes opposite words in the same frames ("operating profit rose to EUR ... mn" / "operating loss narrowed to EUR ... mn"), the model is told they are similar and says so: `profit`'s nearest neighbor is `loss` and vice versa. This is the hypothesis applied honestly, not a bug in the run. **Co-occurrence measures topic; polarity is not topic.**
- **The tokenizer is part of the model.** Keeping digits put `eur4`, `eur7` and bare figures among `profit`'s neighbors. Decide what a token is before deciding what vectors mean.
- **Corpus size buys precision; domain buys coverage.** GloVe (~6B tokens) separates words a 40k-token domain model confuses; the domain model has read earnings sentences GloVe barely saw. Neither substitutes for the other; ask which failure the task tolerates.
- **Four limits of static embeddings, in order of cost:** (1) polarity is not a direction in the space (distance = interchangeable usage); (2) one vector per word regardless of sense (`Apple`/`apple`, `charge` fee/accusation); (3) nothing for out-of-vocabulary tokens — a new ticker, a typo, an absent term of art gets no vector, not a poor one; (4) no unit above the word ("Net loss narrowed" is positive; none of its words is; averaging ignores order and negation). Contextual models fix 1–3 by making a word's vector a function of its sentence and fix 4 by emitting a sentence vector.
- **A supervised classifier can find what similarity ranking cannot.** Close vectors are still distinct points; labels supply the polarity direction (NB03 does this on averaged vectors).
- **Asset embeddings are the same machinery on a different corpus** (sentence → portfolio, word → stock, co-occurrence → co-holding; Gabaix, Koijen, Richmond, Yogo 2025). The limitation transfers: co-holding is not similar performance; a pair trade is held together precisely because the legs are expected to move apart.
- **Run the cheap baseline before the expensive representation.** TF-IDF on a polarized financial vocabulary (`profit`, `loss`, `narrowed`, `tumbled`) costs seconds, is hard to beat, and sets the bar.
- **Read the preprocessing before attributing a score to the representation.** Averaging word vectors keeps every word and loses all structure ("profit narrowed" == "narrowed profit"); `stop_words="english"` keeps structure and deletes words, including negators.
- **"Pre-trained" FinBERT with a sentiment head is not zero-shot.** It is a supervised model from another corpus applied unchanged; its score measures cross-dataset transfer and is fair only if the test sentences were not in its training data.
- **Label names are not label definitions.** Two datasets can share {negative, neutral, positive} and disagree about what the labels are for (annotator judgment vs sign of a subsequent price move). Read the definition from how labels were produced.
- **Three causes of a cross-dataset drop need three responses:** different text (fix: more in-domain data); different labeling standard (fix: annotation guide / calibration); different target quantity (no adaptation closes it — the question changed).
- **A strong text model is not a tradable signal** until timestamps, entity resolution, revisions, aggregation, model training cutoffs and evaluation protocols are point-in-time safe. Only past information may enter today's signal, and the discipline is per step (embedding, aggregation, lookback, return alignment). One careless join makes every downstream number meaningless, not merely optimistic.
- **Build, then evaluate, then decide.** Never decide on the strength of having built something. Moving an IC from one value indistinguishable from zero to another is not evidence.
- **Evaluate text signals with horizon-aware, coverage-aware, event-time-aligned diagnostics**, not benchmark accuracy alone.
- **Filings and headlines are complements.** Filings are dense, structured management self-assessment; headlines capture market reaction speed. MD&A is the narrative section worth reading (Risk Factors are slow-changing boilerplate); it averages ~9,500 words (median ~7,800) per quarter.
- **Filing date is the point-in-time anchor.** The signal exists when the SEC accepts the filing, not at fiscal `period_end`. One investable event per date: a catch-up filer submitting several quarters' 10-Qs on one date must collapse to the most recent period.
- **Dispersion is a signal, not noise.** Two filings with equal mean sentiment can differ between uniformly mild and a mix of strongly positive and strongly negative passages; carry `sentiment_std` as its own factor.
- **Narrative change is deviation from the boilerplate base rate.** Consecutive MD&As are carried forward and edited, so cosine distances hump near zero with a tail; the tail is where something was rewritten. "The signal is the distance relative to that base rate, not the distance itself."
- **The 10-K gap is a property of the forms, not the sample.** One calendar quarter per year shows roughly a fifth as many 10-Q filings because Q4 disclosure goes into the 10-K; treat it as a different, smaller sample, not a quiet period.
- **Pooled ICs are screening-grade, not headline inference.** The same firm appears at many filings and 5/20-day returns overlap, so the i.i.d. `t = r*sqrt((n-2)/(1-r^2))` overstates significance. Cluster bootstrap on symbol preserves within-firm dependence; the chapter's headline framework is HAC on per-date cross-sectional IC series with adequate breadth (NB07/NB08).
- **Report missing rather than a misleading number.** A draw dominated by one cluster still produces a number; a NaN says that where a computed one would not.
- **Honest chart encoding.** Never annotate a bare Q5−Q1 spread (a difference of pooled means with no interval); use one color for all quintile bars (coloring by sign encodes the outcome twice and makes a one-basis-point difference look categorical); interval length carries as much information as bar height.

## Method recipes (the how)

### Tokenization and Word2Vec on financial sentences (NB01 `10_text_feature_engineering/01_word2vec_training`)

1. Load `data.load_financial_phrasebank()` (polars; HF `takala/financial_phrasebank`, subset `sentences_allagree`, ~2,264 sentences, ~40k tokens).
2. Tokenize: lowercase, strip punctuation, split on whitespace, drop single characters. NB01 keeps numbers deliberately so their effect is visible; in production strip numeric tokens or normalize them to a placeholder.
3. Train `gensim.models.Word2Vec(sentences, vector_size=100, window=5, min_count=3, sg=1, negative=10, workers=1, epochs=20, seed=42)`.
4. Inspect neighbors with `model.wv.most_similar(word, topn=5)` (cosine) / `similar_words_frame(words, top_n=5)`; analogies `a − b + c` with `model.wv.most_similar(positive=[a, c], negative=[b], topn=top_n)` / `analogy_frame(triplets, top_n=3)`.
5. **Polarity block test:** three word groups (positive, negative, topical nouns); pairwise cosine matrix in the native 100-d space ordered by group; compare mean within-polarity similarity vs mean across-polarity similarity (self-similarity excluded). Clustered polarity would show darker diagonal blocks.
6. Compare with `gensim.downloader.api.load("glove-wiki-gigaword-100")` (~128 MB, cached).

| Parameter | NB01 value | Meaning / when to change |
|---|---|---|
| `vector_size` | 100 | Usual range 100–300; a ~40k-token corpus does not justify the top |
| `window` | 5 | Small → interchangeable-in-phrase neighbors; large → shared topic |
| `min_count` | 3 | Too few occurrences cannot locate a vector |
| `sg` | 1 | 1 = Skip-gram (word predicts neighbors); 0 = CBOW (neighbors predict word) |
| `negative` | 10 | Random words pushed away per update; prevents collapse to one point |
| `workers` | 1 | >1 lets the OS decide update order; seed no longer determines the result |
| `epochs` | 20 | 10 for the portfolio corpus in NB02 |
| `seed` | 42 | Plus `set_global_seeds(SEED)` |

Analogy arithmetic scales worst with corpus size (needs all four words well located); treat completions as a check that the arithmetic is defined, not as answers. Writes a trained model + summary to the chapter output dir; nothing downstream reads them. Runs on `ml4t-py312`.

### Asset embeddings from 13F holdings (NB02 `02_asset_embeddings`)

1. Data: 13F bulk `QUARTER="2024Q3"` under `data/equities/positioning/13f/bulk/2024Q3/` via `python data/equities/positioning/13f_download.py --mode bulk --quarters 2024Q3`.
2. **Dedupe first:** collapse to the largest reported value per (cik, stock) — amendments and subsidiaries file multiple 13F-HR per quarter and duplicates would teach a stock to co-occur with itself.
3. Portfolio sentence = for each institution (CIK), stocks ordered by holding value descending. Keep `MAX_INSTITUTIONS=500` by number of holdings.
4. Train skip-gram: `embedding_dim=100`, `window_size=5`, `min_count=5` (stock in ≥5 portfolios), `sg=1`, `epochs=10`, `workers=1`, seed 42. Neighbors via `neighbours_frame(query, top_k=10)` (CUSIP or whole-word name match → `model.wv.most_similar(cusip, topn=top_k)`).
5. **Co-ownership test:** pairwise cosine among recognizable mega-caps vs their similarity to a random draw from the rest of the widely-held universe; the two means are the test. Caveat: popularity alone could produce this — add a popularity-driven baseline.
6. **Masked-asset benchmark (Gabaix et al. 2025):** `predict_masked_asset(portfolio, mask_position, model, window=5)`; `evaluate_benchmark(portfolios, model, position_ranges=[(2,10),(10,50),(50,200)], samples_per_range=100)` (CONFIG uses `samples_per_range=25`); one `HitRecord` per sample; metrics Hits@k (true stock in top-k neighbors of its context) and MRR (≤ 1).
7. Pool the hit rate over masked positions (never average bucket rates); bootstrap the pooled outcomes with the **portfolio** as resampling unit: `bootstrap_by_portfolio(records, n_iterations=1000, alpha=0.025)` percentile CI for Hits@5; `bootstrap_metrics(hit_records, n_iterations=1000, confidence=0.95)`; CONFIG `bootstrap_iterations=1000`.
8. Baselines: analytical random `5 / vocab_size`; **popularity baseline** = always answer the 5 most widely held stocks, scored on the *same stored hit records* (not re-walked). Read the gap, not either rate.

The paper compares three methods — Recommender Systems (PCA on the investor-asset matrix), Word2Vec, BERT (contextual); only Word2Vec is implemented. Runtime ~5–6 min CPU; `ml4t-py312`.

### Three generations on one task (NB03 `03_sentiment_evolution`)

One PhraseBank `allagree` split, `test_size=0.2`, seeds via `set_global_seeds` plus separate seeding of transformers pipelines.

| Model | Recipe | Notes |
|---|---|---|
| TF-IDF + LR | `TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, stop_words="english")` → `LogisticRegression(max_iter=1000, random_state=SEED)` | Fast baseline; audit `stop_words` for negators |
| GloVe-avg + LR | `glove-wiki-gigaword-100`; `document_vector(doc, model, dim=100)` = mean of word vectors → LR | Keeps every word, loses all structure |
| FinBERT-tone (no fine-tune) | `yiyanghkust/finbert-tone` (trained on analyst reports + earnings-call transcripts, NOT PhraseBank); `get_finbert_predictions(texts, batch_size=32)` | Map labels **by name**; assert the dataset's mapping; print the checkpoint id rather than leaving it in config |

Plot the three confusion matrices on one shared color scale and colorbar.

### Fine-tuning transformer classifiers (NB04 `04_bert_finetuning`)

1. Split PhraseBank `allagree` stratified 70/15/15 via two `train_test_split(..., stratify=label)` calls (`test_size=0.15`, `val_size=0.15`); build a `DatasetDict` from pandas (`create_dataset_dict`).
2. Seed: `set_global_seeds(SEED)` **and** `transformers.set_seed(SEED)` (`set_transformers_seed(SEED)`) — shuffling, dropout and pipelines have their own RNGs.
3. Checkpoints: `ProsusAI/finbert` (**contaminated**: fine-tuned on the whole PhraseBank; label order 0:positive, 1:negative, 2:neutral vs the notebook's 0:negative, 1:neutral, 2:positive), `microsoft/deberta-v3-small`, `answerdotai/ModernBERT-base`.
4. Tokenize with `AutoTokenizer`, `max_length=128`, no padding at tokenization; `DataCollatorWithPadding` pads per batch.
5. `Trainer` with `TrainingArguments(num_train_epochs, per_device_train_batch_size, per_device_eval_batch_size, warmup_steps, weight_decay, eval_steps, ...)`: `learning_rate=2e-5`, `weight_decay=0.01`, `warmup_ratio=0.1` in CONFIG but the Trainer receives `warmup_steps=min(50, max(10, train_size//batch_size//4))`, `per_device_eval_batch_size = 2 × train batch`.
6. Papermill controls: `MAX_TRAIN_STEPS=-1` (cap optimizer steps; −1 = run epochs), `MAX_EVAL_SAMPLES=0` (0 = score everything; a prefix of a shuffled stratified split is class-balanced in expectation); reduced mode `eval_steps=max(10, max_steps//5)`, `save_steps=eval_steps*2`. No trailing comments on these parameter lines.
7. Metrics: accuracy + macro F1 — read macro F1. Results table carries a **provenance column**; outputs under `output/bert_finetuning/<model_name_lower>/`. `ml4t-gpu`.

### Financial NER token classification (NB05 `05_financial_ner_finetuning`)

1. Data: `generate_synthetic_ner_data(n_samples=500, seed=42)` — 5 templates (four with 3 entity slots, one with 2), 5 options per slot, independent draws with replacement → only a few hundred distinct sentences. Count exact token sequences shared across train/test before trusting any score.
2. BIO scheme (`B-` opens a span, `I-` continues it, `O` outside); two adjacent orgs vs one two-word org are distinguished only by B-/I-. Fix the tag vocabulary; do not infer it.
3. `tokenize_and_align_labels`: for each `word_id` in `tokenized.word_ids()`, assign the label on the first subword of each word, `-100` on the remaining subwords and on special tokens (`None`) — PyTorch cross-entropy ignores −100, so the loss is counted once per word.
4. Fine-tune `AutoModelForTokenClassification` with `Trainer`.
5. Metrics: entity-level F1 via `seqeval` (a span counts only if extent AND type match); the token-level `sklearn` fallback is more lenient.
6. Features: `extract_entities(text) -> list[(span, type)]` groups consecutive tokens by tag; count entities by `B-` tags only; `extract_entity_features(text) -> dict` gives counts of orgs, amounts, dates, percentages per document. Real corpora named: CoNLL-2003, FiNER-139. `ml4t-gpu`.

### Cross-dataset transfer check (NB06 `06_finbert_cross_dataset`)

1. `load_finmarba_dataset(sample_size=10000)` (HF `baptle/financial_headlines_market_based`, `SAMPLE_SIZE=10000` → ~8k headlines). Label = `Global Sentiment` = sign of `Pct_Change` aggregated across mentioned tickers — a market-move label, not annotator sentiment.
2. Print class balance and the **majority-class rate** (the zero point for skill; not 1/k).
3. Score with `get_finbert_predictions(texts, batch_size=32)` on `ProsusAI/finbert`, no training.
4. Confusion matrix: rows = market move, columns = model reading. Read misclassified examples and label provenance before attributing the drop.
5. Classify the drop as text shift / labeling-standard shift / target shift; only the first two warrant adaptation. `ml4t-gpu`.

### News-to-signal pipeline on headlines (NB07 `07_news_return_signals`)

CONFIG: `embedding_model="sentence-transformers/all-MiniLM-L6-v2"`, `lookback_days=20`, `signal_lag_days=1`, `dedup_threshold=0.9` (unused by hash dedup; used by the optional TF-IDF pass). Data: FNSPID sample (`python data/text/fnspid_download.py`, `--sample 0` for the full set; HF `Zihan1004/FNSPID`) + US equities 1962–2018 prices. Column detection by priority list (headline preferred over summary). Universe filtered to top tickers by coverage (count not visible in notes).

| Stage | Function | Rule |
|---|---|---|
| Dedup | `deduplicate_news(df, text_col, date_col, ticker_col, similarity_threshold=0.9) -> (df, dup_rate)` | Normalize (lowercase, strip punctuation, collapse whitespace); exact hash + 50-char-prefix hash within (ticker, date); O(n) |
| Fuzzy dedup (optional) | `deduplicate_news_tfidf(...)` | Cosine ≥ 0.9 within group; O(n²) per group; run side by side with the hash pass to tune the cutoff |
| Entity resolution | `normalize_ticker` | Map aliases to canonical symbols |
| Embeddings | `encode_texts(texts, batch_size=32)` | Transformer + `mean_pooling(model_output, attention_mask)` (mask-weighted mean) + L2 normalization |
| Sentiment | `score_sentiment_finbert(texts, batch_size=64) -> (scores, probs)` | FinBERT-tone mapped to −1/0/+1 |
| Aggregate | `aggregate_by_ticker_date(news_df, embeddings, sentiment_scores) -> (dict, df)` | Per (ticker, date): mean embedding, mean sentiment, coverage count |
| News surprise | `compute_news_surprise(ticker_embeddings, lookback=20)` | Cosine distance between today's mean embedding and the mean of the prior 20 **news-days'** embeddings for that ticker; needs lookback+1 news dates; lookback is in event days, not calendar/trading days |
| Directional features | `create_directional_features(surprise_df, sentiment_df, lookback=20)` | `weighted_surprise = surprise × sign(sentiment_mean)`; `sentiment_momentum = sentiment_mean − sentiment_mean.shift(20).over("ticker")`; `coverage_count` |
| Lag + align | `target_date = signal_date + duration(days=1)`; `join_asof(..., strategy="forward")` | Forward to the next existing session keeps Friday/pre-holiday news; join to `fwd_ret_1d/5d/20d`; warn if <50% of news dates have prices |
| Screen | per-date `scipy.stats.spearmanr(signal, fwd_ret)` | Flags `abs(mean IC) > 0.03` and `ICIR > 0.5`; quintile means of 1/5/20d returns, spread Q5−Q1 |

Writes `10_text_feature_engineering/output/fnspid/news_features.parquet` (features + forward returns as labels). Other factor families named: sentiment momentum, coverage intensity (attention), topic shift. `ml4t-gpu`.

### Text-signal evaluation framework (NB08 `08_text_feature_evaluation`)

Reads `news_features.parquet` from NB07 (labels NOT recomputed). `EvalConfig` class; `normalize_schema`; `iter_date_groups`.

1. `daily_ic(df, signal_col, ret_col, min_assets)` — cross-sectional Spearman per date; skip dates with fewer than `min_assets` names.
2. `summarize_ic(ic_df)` — mean, std, `ICIR = mean/std`, `t = mean/(std/√n)`.
3. `quintile_long_short(df, signal_col, ret_col, quantiles, min_assets)` — daily top−bottom return series; ranks via `rank(method="average")`.
4. Diagnostic reports per-date tie counts and bucket sizes. Quantiles = 5 and horizons 1/5/20d inferred from NB07's columns (inference; `EvalConfig` values not visible). CPU-only `ml4t`.

### Filing text signals from 10-Q MD&A (NB09 `09_filing_text_signals`)

Parameters cell (`tags=["parameters"]`): `SEED=42`, `MAX_SYMBOLS=50`, `MAX_FILINGS=0`, `BATCH_SIZE=8`, `MAX_TOKENS=512`, `EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"`, `SENTIMENT_MODEL="yiyanghkust/finbert-tone"`, `N_BOOT=1000`, `MIN_VALID_BOOT=200`; zero for `MAX_SYMBOLS`/`MAX_FILINGS` means no cap. Inference only; ~5–7 min GPU at 50 symbols, CPU 5–10× slower, linear in filings. `os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")`.

| Stage | Recipe |
|---|---|
| 1. Load | `data.load_sp500_10q_mda()` (S&P 500 10-Qs 2017–2021 from `data/equities/fundamentals/filings_download.py --form 10-Q --universe sp500`); columns `text, symbol, filing_date, period_end` |
| 2. Cap universe deterministically | `filings.group_by("symbol").len().sort(["len","symbol"], descending=[True, False]).head(MAX_SYMBOLS)` — tie-break on symbol because `group_by` order is arbitrary; `MAX_FILINGS > 0` → `sort(["filing_date","symbol"]).head(MAX_FILINGS)` (test runs only) |
| 3. Dedupe | `filings.sort(["symbol","filing_date","period_end"]).unique(subset=["symbol","filing_date"], keep="last", maintain_order=True)` → latest `period_end` per (symbol, filing_date); print rows collapsed. `word_count = text.str.split(" ").list.len()` |
| 4. Chunk | `chunk_text(text, max_chars)`: split on `"\n\n"`, keep paragraphs `len > 50`; fallback split on `"."` keeping sentences `len > 30` (re-append period); greedily pack paragraphs until `len(current)+len(para) > max_chars`; final fallback `[text[:max_chars]]`. `max_chars=1500` for sentiment (~512 tokens), `1200` for embeddings |
| 5. Sentiment | `score_document_sentiment(text)`: `AutoTokenizer`/`AutoModelForSequenceClassification` `.to(device).eval()` in `pipeline("sentiment-analysis", truncation=True, max_length=512, batch_size=8, device=device)` after `set_seed(SEED)`; chunk score = `label_map[label] × confidence` with `{"Positive": +1, "Neutral": 0, "Negative": -1}` (unknown → 0); aggregates `sentiment_mean`, `sentiment_std` (population; 0 if ≤1 chunk), `sentiment_pos_pct = mean(scores>0)`, `sentiment_neg_pct = mean(scores<0)`, `n_chunks`; empty → 0/0/0 |
| 6. Embed | `SentenceTransformer(EMBEDDING_MODEL, device=str(device))`; `embed_document(text, model)` = mean over `model.encode(chunks, show_progress_bar=False, batch_size=32)`; zeros if no chunks; dim 384 (inference) |
| 7. Narrative change | Order by `(symbol, filing_date)` with `with_row_index("idx")`; walk keeping `prev_emb_by_symbol`; `cos_sim = dot(emb, prev)/(‖emb‖·‖prev‖ + 1e-8)`; `narrative_change = 1 − cos_sim` (0 identical, 2 opposite); first filing per company → `None`. Across the 10-K gap the "consecutive" pair spans two quarters (inference; not addressed) |
| 8. Forward returns | `data.load_sp500_daily_bars()` (AlgoSeek; `symbol, timestamp, close`); `prices.sort(["symbol","timestamp"])`; `fwd_Nd = (close.shift(-N)/close − 1).over("symbol")`, N ∈ {1, 5, 20}, close-to-close from the matched trade date |
| 9. PIT join | Rename `filing_date → timestamp`, sort `(symbol, timestamp)`; `join_asof(prices, on="timestamp", by="symbol", strategy="forward")` = first trading day on or after the filing date; rename back; `drop_nulls(["fwd_5d"])`; assert sortedness before suppressing the "Sortedness of columns cannot be checked" warning |
| 10. Pooled IC + cluster bootstrap | `signal_cols = [sentiment_mean, sentiment_std, sentiment_pos_pct, sentiment_neg_pct, narrative_change]` × `return_cols = [fwd_1d, fwd_5d, fwd_20d]` → up to 15 pairs. Per pair: `drop_nulls` on `[symbol, sig, ret]`, skip if `height < 20`; `_pooled_spearman` → NaN if `n < 5` or zero std else `spearmanr(...)[0]`; draw `len(unique_symbols)` symbols with replacement for `N_BOOT=1000` iterations from one shared `np.random.default_rng(SEED)`, concatenate row indices, recompute; drop NaN replicates; if survivors ≥ `MIN_VALID_BOOT=200`: `ci95_lo/hi = percentile(boot, 2.5/97.5)`, `p_cluster_boot = 2 × min(mean(boot ≤ 0), mean(boot ≥ 0))` (percentile-method test inversion, NOT the recentered reflection test `mean(|boot − boot.mean()| ≥ |ic|)`); else NaN. Output row: `signal, horizon, ic, ci95_lo, ci95_hi, p_cluster_boot, n_obs, n_symbols` (4 dp) |
| 11. Quintiles | `quintile_analysis(df, signal_col, return_col)`: drop nulls, empty if `height < 25`; `qcut(5, labels=["Q1".."Q5"])`; `avg_return`, `std_return`, `n_obs` per bucket; applied to `sentiment_mean` and `narrative_change` vs `fwd_20d`; one color, zero line, lowest bucket left, no Q5−Q1 annotation |
| 12. Save | `OUTPUT_DIR = get_chapter_dir(10)/"output"/"filing_signals"`; `filing_signals.parquet` (eval panel: `symbol, filing_date, sentiment_mean, sentiment_std, sentiment_pos_pct, sentiment_neg_pct, n_chunks, narrative_change, trade_date, fwd_1d, fwd_5d, fwd_20d` — inference) and `ic_summary.parquet` |

ICIR is defined in the narrative but NOT computed in NB09 — per-date cross-sections of a 50-symbol filing panel are too thin. `ml4t-gpu`.

### Sequential models and transformer architecture (10.3–10.4; README only)

No notebook covers RNN/LSTM internals, the attention formula, positional encoding, LoRA settings or BloombergGPT; the README names them (self-attention, multi-head attention, positional encoding, BERT-style encoders, FinBERT, BloombergGPT, domain adaptation, fine-tuning, LoRA, ModernBERT, Sentence-BERT) and the production cascade "pre-train → adapt → fine-tune". Concrete rules are not visible in the notes.

## Guardrails and pitfalls

**Embeddings and tokenization**

- **Co-occurrence places opposites together** — nearest-neighbor similarity used as sentiment conflates `profit` and `loss`. Check the opposite of your target word in the neighbor list before trusting distance as a feature; use a supervised head for polarity.
- **Numeric/currency tokens pollute vocabulary** — `eur4`, bare figures become neighbors of key words. Strip or normalize numbers to a placeholder before training; inspect the top-15 frequent tokens.
- **Multi-threaded Word2Vec is non-deterministic** — the seed no longer fixes the result. Use `workers=1`.
- **t-SNE misrepresents distances** — it distorts edge points (puts `profit`/`loss` at opposite corners, presses mega-caps into one spot). Test geometric claims with cosine in native space (within- vs across-group means); use projections only as illustration.
- **Analogy arithmetic undefined on small corpora** — all four words must be well located. Treat completions as a check that the arithmetic is defined, not as answers.
- **Averaged vectors lose order and negation scope** — "profit narrowed" == "narrowed profit". Use sequence models or contextual sentence embeddings.

**Asset embeddings and benchmarks**

- **Duplicate holdings within a filer** — a stock next to itself teaches self-co-occurrence. Collapse to the max value per (cik, stock) before ordering.
- **Averaging bucket rates or CI endpoints** — buckets of 4k and 12k get equal weight; averaged CI endpoints have no coverage property. Pool over masked positions; bootstrap the pooled outcomes.
- **Bootstrapping positions instead of portfolios** — positions in a portfolio share overlapping windows, so the CI is too narrow. Resample by portfolio.
- **Beating uniform random proves little** — popularity alone explains mega-cap clustering. Add the popularity baseline (top-5 most held) scored on the same hit records; read the gap.
- **Re-deriving benchmark trials for a baseline** — a different sample set and bucket weights make populations incomparable. Score baselines from stored hit records.
- **Co-holding ≠ similar performance** — pair-trade legs are co-held to move apart. Ask what the embedding measures before using it as a risk or similarity input.

**Classification, checkpoints and fine-tuning**

- **Label order mismatch between checkpoint and dataset** — mapping by index yields a plausible low score, not an error. Map by name; assert the dataset mapping; print the checkpoint id.
- **`stop_words="english"` deletes negators** (`not`, `no`, `never`, `nor`) — "profit did not rise" == "profit did rise". Inspect the stop list; keep negators on sentiment tasks.
- **Checkpoint trained on your test corpus (test-set contamination)** — `ProsusAI/finbert` was fine-tuned on all of PhraseBank; its score is not held out, is not attributable to your fine-tuning, and nothing in code or metrics reveals it. Read the model card; carry provenance as a column in the results table (contaminated rows need not be the best rows).
- **Cross-notebook before/after comparisons across different checkpoints** — `finbert-tone` vs `ProsusAI/finbert` are different models. Use the same checkpoint on both sides.
- **Kept pretrained head with permuted labels** — FinBERT's head meanings are permuted relative to new labels, so fine-tuning must first undo the permutation. Align label order or re-initialize the head.
- **Accuracy on imbalanced classes** — the majority class dominates; a model that never predicts a class can hide. Report macro F1 and the confusion matrix; read all three.
- **Confusion matrix says what, not why** — the same counts arise from "unsure" and "answering a different question". Read misclassified examples; check label provenance.
- **Small-split differences are seed noise** — 1–2 points on a few hundred test sentences moves with the seed. Do not act on such differences; compare the training-cost spread to the score spread.
- **Papermill parameter cell with trailing comments** — the name is hidden from the parser, the override is dropped silently, and a reduced run trains at production size. No trailing comments on `MAX_TRAIN_STEPS`/`MAX_EVAL_SAMPLES` lines.
- **Separate RNGs in transformers/Trainer** — shuffling, dropout and pipelines are not covered by `set_global_seeds`. Also call `transformers.set_seed(SEED)`.
- **Padding to max_length** — most compute goes to pad tokens. Use `DataCollatorWithPadding` per batch.

**NER**

- **Synthetic/generated split not held out** — a bounded generator repeats (birthday problem); copies land on both sides and scores sit near the ceiling. Count distinct token sequences shared across train/test; treat the score as a plumbing check.
- **Subword alignment errors** — code runs, trains and reports a number while the loss is counted per subword or on the wrong pieces. First subword gets the label, the rest −100.
- **Token-level vs entity-level NER metric** — token-level gives partial credit for wrong boundaries. Install `seqeval`; quote entity-level F1.
- **Counting non-`O` tokens as entities** — multi-word spans are double-counted and the type distribution skews by span length. Count `B-` tags.

**Labels and cross-dataset transfer**

- **Label names ≠ task** — FinMarBa labels are the sign of subsequent price moves, not sentiment. Inspect the column labels were derived from (`Pct_Change`) before scoring.
- **Accuracy without majority baseline** — the zero point for skill is the majority rate, not 1/k. Print class balance and majority rate first.
- **Neutral-rate mismatch** — market-neutral (small move) ≠ sentiment-neutral (no view). Record the disagreement; do not attribute it without sentiment annotations.

**News-signal construction (10.5)**

- **Wire-service syndication duplicates** — mean sentiment and coverage counts count one story many times. Dedup (exact + 50-char prefix hash within ticker-date) before any per-document aggregation.
- **News timestamp after close → look-ahead** — news on date t used for t→t+1 returns. Lag the signal 1 day; the surprise baseline uses only t−20..t−1; the model must be trained before the evaluation window.
- **Calendar-day lag joined on equality** — drops every Friday and pre-holiday story. `join_asof(strategy="forward")` to the next session; a reversed direction is where look-ahead would enter.
- **Event-day lookback ≠ time lookback** — thin-coverage tickers reach back years or drop out; the universe becomes conditioned on attention. State the selection; consider a trading-day lookback with missing-news handling.
- **Universe conditioned on coverage (selection)** — NB07 drops low-coverage tickers, so all results are conditional. Report the selection explicitly.
- **Model cutoffs as leakage** — a checkpoint trained after the evaluation window has seen future text. Record the training cutoff; evaluate only on post-cutoff data.

**Signal evaluation**

- **Unequal quantile buckets** — ties (`weighted_surprise` is exactly 0 when sentiment is 0; discrete surprise repeats) and single-digit cross-sections make the extremes not the top/bottom fifths. Check per-date bucket sizes and tie counts; prefer daily IC when cross-sections are small.
- **Pooled observations across dates** — a persistent cross-sectional tilt masquerades as accuracy. Compute one IC per date and average those; the date is the unit.
- **Overlapping-horizon serial correlation** — 5d/20d daily ICs are autocorrelated, so `std/√n` is too narrow. Apply Newey-West or a block bootstrap before quoting a t-stat near threshold.
- **Quoting a bucket spread without an interval, or annualized** — a difference of pooled means has no uncertainty; annualizing states units nothing measured. Form the daily long-short series, treat it as the sample, report per period with an interval.
- **Ranking signals whose statistics are all ~0** — ordering among noise is noise. Do not rank; report as indistinguishable from zero.

**Filing signals (NB09)**

- **Backward as-of join = look-ahead** — matching a filing to the last session on or before the filing date attaches the signal to a close that preceded it. Use `join_asof(strategy="forward")` keyed on the SEC acceptance date; assert both frames are sorted on (symbol, timestamp).
- **Period-end vs filing-date anchoring** — text is not public until acceptance, typically weeks after `period_end`. Key on `filing_date`; keep `period_end` only as metadata.
- **Same-date multi-quarter filings** — duplicate (symbol, filing_date) keys fan out the join on both sides and overweight those firms in every IC. `unique(subset=["symbol","filing_date"], keep="last")` after sorting by `period_end`; print the rows collapsed.
- **Pooled-sample dependence** — many filings per firm plus overlapping 5/20-day windows; the i.i.d. t overstates significance. Cluster bootstrap on symbol (`N_BOOT=1000`); report percentile CI and bootstrap p; treat as a screen; headline inference needs HAC on per-date IC series.
- **Bootstrap degeneracy** — replicates dominated by one cluster or constant signal slices. `_pooled_spearman` returns NaN for n<5 or zero variance; require ≥ `MIN_VALID_BOOT=200` survivors or report NaN.
- **Rows without a cluster id** — filings with null symbol cannot be resampled, and a point estimate on rows the interval excludes is inconsistent. Drop them before both the point estimate and the resampling.
- **Non-deterministic universe cut** — `group_by` order is arbitrary, so ties at the top-N cutoff change the sample run to run. Sort by `["len","symbol"]` with the explicit tie-break; `set_global_seeds(SEED)` + `transformers.set_seed(SEED)`.
- **Bootstrap reproducibility** — one RNG shared across all (signal, horizon) pairs means reordering `signal_cols`/`return_cols` changes every later pair's draws. Keep the iteration order fixed or seed per pair (inference).
- **Seasonal artefact from the 10-K** — one calendar quarter per year has ~1/5 the filings because Q4 disclosure is in the 10-K, which this pipeline does not read. Do not read the gap as a quiet period; condition seasonal analyses on form type or add 10-K MD&A.
- **Token-limit truncation** — FinBERT/MiniLM accept 512 tokens (~380 words) vs ~9,500-word MD&As; single-pass scoring silently keeps only the opening. Chunk paragraphs at 1500/1200 chars, `truncation=True, max_length=512`, aggregate by mean.
- **Mean sentiment hides mixed tone** — the same mean arises from uniform-mild and polarized documents. Carry `sentiment_std`, `pos_pct`, `neg_pct` as separate factors; inspect the std histogram first.
- **Raw cosine distance is mostly boilerplate** — distances cluster near zero because MD&A is carried forward and edited. Interpret relative to the base rate (rank/quantile the distance; the IC is rank-based).
- **Unannotated quintile spread** — printing Q5−Q1 reads as a result even when the ordering is non-monotonic. Omit it; rely on the bootstrap CI table.
- **Chart encoding bias** — coloring bars by sign double-encodes and makes ±1bp look categorical. Single color, zero reference line.
- **Intraday acceptance time (inference)** — a filing accepted after the close matched to that day's close leaves a residual one-session look-ahead. Shift to the next session if acceptance time > close, or use the next-day open; the notebook uses date-only matching.
- **Polars sortedness check** — `join_asof` with `by=` cannot verify sortedness and unsorted inputs give wrong matches silently. Explicit sort + assert before suppressing the warning (the notebook's first assert `timestamp.is_sorted() or symbol.n_unique() > 1` is nearly vacuous with >1 symbol — inference).
- **Compute budget** — FinBERT scores one filing at a time; the full S&P 500 takes many times the 50-symbol default. `MAX_SYMBOLS` trades coverage for completion; GPU recommended; `TOKENIZERS_PARALLELISM=false`.
- **Breadth too small for per-date inference** — a 50-symbol filing panel has few filings on any date. Pool with cluster bootstrap here; move to the NB07/NB08 HAC framework only with adequate breadth.
- **Multiple testing (inference)** — 15 signal-horizon pairs screened; a few p<0.05 are expected by chance. Treat as a screen; confirm out of sample / with the HAC framework.

## Decision rules and defaults

**Sequencing for any text classification task:** (1) print class balance and majority rate; (2) verify label mapping by name; (3) run the TF-IDF+LR baseline; (4) only then richer representations; (5) compare on an identical test split.

**Text-to-signal checklist (10.5):** availability timestamp (publication date; lag 1 day) → wire dedup (exact + 50-char prefix hash within (ticker, date)) → entity resolution (canonical tickers) → lookback uses only the past (t−20..t−1) → model trained before the evaluation window (record the cutoff) → `join_asof` forward to the next session → forward returns at 1/5/20 days → per-date IC.

**Filing-signal sequencing (NB09):** load → cap universe deterministically → dedupe per filing date → chunk → score sentiment → embed → narrative change (per symbol, chronological) → join signals → forward returns → forward as-of join → pooled IC + bootstrap → quintiles → save parquet.

**Go/no-go:** build → evaluate → decide. If ICIR ≈ 0, t within noise and the bucket sort is non-monotone, the result is "no measurable edge", not "tune further". Interpretation rules: (a) CI crossing zero → the sign is not information; (b) interval long relative to bar → ignore the point estimate; (c) non-monotonic quintiles → no economic-spread claim; (d) pooled ICs are a screen, never a headline.

| Decision | Rule |
|---|---|
| Pooled IC + cluster bootstrap vs per-date IC + HAC | Pooled/cluster when breadth per date is thin (filing events, small universe); per-date IC series + HAC (NB07/NB08) when the cross-section per date is adequate |
| Quintile spread vs daily IC | Read daily IC, not quintile spreads, when cross-sections are single-digit or ties are common |
| Reporting long-short | Per period with an interval from the daily series; never annualize a pooled-mean difference |
| t-stat near threshold at 5d/20d | Newey-West or block bootstrap first |
| Screening conventions | abs(mean IC) of a few hundredths (code flags > 0.03) = economically interesting; ICIR ≈ 0.5 = robust. Conventions set order of magnitude, not tests |
| Published checkpoint | Read the model card for training data; if it overlaps the test corpus, mark the row contaminated |
| Fine-tuned models within 1–2 points on a few-hundred-sentence split | Treat as noise; decide on training cost |
| Cross-dataset drop | Compare to majority rate; classify as text shift / labeling-standard shift / target shift; only the first two warrant adaptation |
| Geometric claims about embeddings | Within-group vs across-group mean cosine in native dimension; never t-SNE as evidence |
| Numeric tokens | Normalize or strip before training production embeddings |
| Fuzzy dedup cutoff | Cosine 0.9 default; tune by inspecting what it removes relative to the hash pass |

| Component | Parameter | Default |
|---|---|---|
| Word2Vec (text) | `vector_size` / `window` / `min_count` / `sg` / `negative` / `workers` / `epochs` | 100 (range 100–300) / 5 / 3 / 1 / 10 / 1 / 20 |
| Word2Vec (portfolios) | `embedding_dim` / `window_size` / `min_count` / `sg` / `epochs` / `MAX_INSTITUTIONS` | 100 / 5 / 5 / 1 / 10 / 500 |
| Masked-asset benchmark | `position_ranges` / `samples_per_range` / bootstrap | (2,10),(10,50),(50,200) / 25 in CONFIG (100 in function) / 1000 iterations by portfolio; expect 1–2 orders of magnitude above `5/vocab_size`; require lift over the popularity baseline |
| TF-IDF | `max_features` / `ngram_range` / `min_df` / `stop_words` | 5000 / (1,2) / 2 / "english" — audit for negators |
| Fine-tuning | lr / `weight_decay` / warmup / `max_length` / split / metric | 2e-5 / 0.01 / ~10% capped `min(50, max(10, n/bs/4))` / 128 / 70/15/15 stratified / macro F1 |
| NER | tags / alignment / metric / counting | BIO / first subword labeled, others −100 / `seqeval` entity-level F1 / `B-` tags |
| News signals (NB07) | `lookback_days` / `signal_lag_days` / `dedup_threshold` / embedding / sentiment batch | 20 news-days / 1 / 0.9 / `all-MiniLM-L6-v2`, batch 32 / FinBERT-tone, batch 64 |
| Filing signals (NB09) | `MAX_SYMBOLS` / `MAX_FILINGS` / `BATCH_SIZE` / `MAX_TOKENS` / chunk chars / `N_BOOT` / `MIN_VALID_BOOT` / CI | 50 (0 = all) / 0 / 8 / 512 / 1500 sentiment, 1200 embedding (paragraph ≥ 50 chars, sentence fallback ≥ 30) / 1000 / 200 / 2.5–97.5 percentiles |
| Minimum samples (NB09) | IC pair / Spearman / quintiles / CI | ≥ 20 valid rows / ≥ 5 rows and nonzero variance / ≥ 25 rows (5 per bucket) / ≥ 200 valid replicates |
| Dedupe rule (NB09) | Keep | One row per (symbol, filing_date), latest `period_end` |
| Join rule (NB09) | As-of | Forward on filing_date, by symbol; drop rows with null `fwd_5d` |
| Horizons | Forward returns | 1 / 5 / 20 days |
| Runtime | NB02 / NB09 | ~5–6 min CPU / ~5–7 min GPU at 50 symbols, CPU 5–10× slower, linear in filings |

## Code patterns and APIs

**Repo layout and running.** `10_text_feature_engineering/0{1..9}_*.py` (+ `.ipynb`, jupytext percent format), `README.md`; outputs under `10_text_feature_engineering/output/` (`fnspid/news_features.parquet`, `bert_finetuning/<model_name_lower>/`, `filing_signals/{filing_signals,ic_summary}.parquet`). Run `uv run python 10_text_feature_engineering/<notebook>.py`; test mode `uv run pytest tests/test_chapter_notebooks.py -v -k "10_text_feature_engineering"` (Papermill, reduced data). Docker: `docker compose --profile py312 run --rm py312 python 10_text_feature_engineering/<nb>.py` for 01/02/03 (gensim has no Python 3.14 wheel); `docker compose run --rm ml4t-gpu python ...` for 04/05/06/07/09; `ml4t` (CPU) for 08.

**Data loaders.** `data.load_financial_phrasebank()` (polars; `takala/financial_phrasebank`, `sentences_allagree`); `data/text/fnspid_download.py [--sample 0]`; `data/equities/positioning/13f_download.py --mode bulk --quarters 2024Q3`; HF `datasets` for `baptle/financial_headlines_market_based`; `from data import load_sp500_10q_mda, load_sp500_daily_bars` (source: `data/equities/fundamentals/filings_download.py --form 10-Q --universe sp500`; AlgoSeek bars).

**Utilities.** `utils.reproducibility.set_global_seeds(SEED)` (Python, NumPy, Torch) + `transformers.set_seed(SEED)`; `utils.paths.get_chapter_dir(10)`; `utils.style.COLORS, FIGSIZE, show_with_alt(fig, alt_text)` (FIGSIZE keys `dual_h_tall`, `triple_h_tall`, `single`, `single_tall`); `torch.device("cuda" if torch.cuda.is_available() else "cpu")`; `os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")`.

**Libraries.** gensim: `Word2Vec(sentences, vector_size, window, min_count, sg, negative, workers, epochs, seed)`, `model.wv.most_similar(...)`, `gensim.downloader.api.load("glove-wiki-gigaword-100")`. sklearn: `TfidfVectorizer`, `LogisticRegression(max_iter=1000)`, `train_test_split(..., stratify=label)`; `scipy.stats.spearmanr`. HF: `AutoTokenizer`, `AutoModelForSequenceClassification` / `AutoModelForTokenClassification`, `Trainer`, `TrainingArguments(num_train_epochs, per_device_train_batch_size, per_device_eval_batch_size, warmup_steps, weight_decay, eval_steps, ...)`, `DataCollatorWithPadding`, `DatasetDict` from pandas, `pipeline` for `yiyanghkust/finbert-tone`, `set_seed`; `sentence_transformers.SentenceTransformer`; `seqeval`; `np.random.default_rng`.

**Checkpoints.** `yiyanghkust/finbert-tone` (analyst reports + earnings calls; Positive/Neutral/Negative — label strings inferred), `ProsusAI/finbert` (contaminated with PhraseBank; label order 0:positive, 1:negative, 2:neutral), `microsoft/deberta-v3-small`, `answerdotai/ModernBERT-base`, `sentence-transformers/all-MiniLM-L6-v2`, `glove-wiki-gigaword-100`.

**Helper signatures.** `similar_words_frame(words, top_n=5)`; `analogy_frame(triplets, top_n=3)`; `neighbours_frame(query, top_k=10)`; `predict_masked_asset(portfolio, mask_position, model, window=5)`; `evaluate_benchmark(portfolios, model, position_ranges, samples_per_range=100)`; `bootstrap_by_portfolio(records, n_iterations=1000, alpha=0.025)`; `bootstrap_metrics(hit_records, n_iterations=1000, confidence=0.95)`; `document_vector(doc, model, dim=100)`; `get_finbert_predictions(texts, batch_size=32)`; `create_dataset_dict`; `set_transformers_seed(SEED)`; `generate_synthetic_ner_data(n_samples=500, seed=42)`; `tokenize_and_align_labels`; `extract_entities(text) -> list[(span, type)]`; `extract_entity_features(text) -> dict`; `load_finmarba_dataset(sample_size=10000)`; `deduplicate_news(df, text_col, date_col, ticker_col, similarity_threshold=0.9) -> (df, dup_rate)`; `deduplicate_news_tfidf(...)`; `normalize_ticker`; `mean_pooling(model_output, attention_mask)`; `encode_texts(texts, batch_size=32)`; `score_sentiment_finbert(texts, batch_size=64) -> (scores, probs)`; `aggregate_by_ticker_date(news_df, embeddings, sentiment_scores) -> (dict, df)`; `compute_news_surprise(ticker_embeddings, lookback=20)`; `create_directional_features(surprise_df, sentiment_df, lookback=20)`; `EvalConfig`; `normalize_schema`; `iter_date_groups`; `daily_ic(df, signal_col, ret_col, min_assets)`; `summarize_ic(ic_df)`; `quintile_long_short(df, signal_col, ret_col, quantiles, min_assets)`; `_safe_scalar(value, default=0.0)`; `chunk_text(text, max_chars=1500)`; `score_document_sentiment(text)`; `embed_document(text, model)`; `_pooled_spearman(sig, ret)`; `quintile_analysis(df, signal_col, return_col)`.

**Lag a daily signal to the next trading session (NB07):**

```python
features_lagged = (
    features.with_columns((pl.col("signal_date") + pl.duration(days=1)).alias("target_date"))
    .sort("target_date")
    .join_asof(trading_days.sort("date"), left_on="target_date", right_on="date", strategy="forward")
)
```

**Sentiment momentum (polars):** `sentiment_mean - pl.col("sentiment_mean").shift(20).over("ticker")`; quantile ranks via `rank(method="average")`; per-date groups via `iter_date_groups`.

**Subword label alignment (NB05):** for each `word_id` in `tokenized.word_ids()`: label if first occurrence of that `word_id`, else `-100`; `None` (special tokens) → `-100`.

**Deterministic top-N by filing count (NB09):**

```python
top = (filings.group_by("symbol").len()
       .sort(["len", "symbol"], descending=[True, False])
       .head(MAX_SYMBOLS)["symbol"].to_list())
```

**One signal per filing date (NB09):**

```python
filings = (filings.sort(["symbol", "filing_date", "period_end"])
           .unique(subset=["symbol", "filing_date"], keep="last", maintain_order=True))
```

**Forward returns and PIT forward as-of join (NB09):**

```python
fwd_5d = (pl.col("close").shift(-5) / pl.col("close") - 1).over("symbol")

eval_df = (signals.rename({"filing_date": "timestamp"}).sort(["symbol", "timestamp"])
           .join_asof(prices_fwd, on="timestamp", by="symbol", strategy="forward")
           .rename({"timestamp": "filing_date"}).drop_nulls(["fwd_5d"]))
```

**Cluster-bootstrap percentile CI and p-value (NB09):**

```python
ci_lo, ci_hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
p_boot = 2.0 * min(np.mean(boot <= 0.0), np.mean(boot >= 0.0))
```

**Quintile buckets (NB09):** `pl.col(sig).qcut(5, labels=["Q1","Q2","Q3","Q4","Q5"])`.

## Evidence from the book

| Notebook | Finding |
|---|---|
| 01 Word2Vec | Nearest neighbor of `profit` is `loss` and vice versa; `eur4`, `eur7` among neighbors; `eur`, `mn` among the 15 most frequent tokens; within-polarity mean cosine ≈ across-polarity mean (no block separation); darkest off-diagonal cells are `improved`–`negative` and `profit`–`loss`; topical nouns form no block; GloVe (6B tokens) separates what the 40k-token model confuses, the domain model has coverage GloVe lacks |
| 02 Asset embeddings | Mega-caps closer to each other than to a random widely-held draw; matrix splits into a tech block vs a looser bank/oil/carmaker/conglomerate block (not a controlled test of differentiated co-holding); Hits@k falls with position depth; pooled rate dominated by deep buckets; Word2Vec on ~500 portfolios for one quarter sits 1–2 orders of magnitude above `5/vocab_size` (~11k stocks), outside the bootstrap CI. The paper's valuation-variance and text-embedding comparisons are not reproduced |
| 03 Three generations | On allagree (2,264 sentences, 20% test): FinBERT-tone > TF-IDF > GloVe-average; the transformer's gain is concentrated in the negative row (lexical methods scatter negatives across all three columns); all three lose similar positives to neutral. On the mixed-agreement subset (chapter 10.4 table, not measured here) FinBERT-tone slips behind TF-IDF |
| 04 Fine-tuning | Three fine-tuned scores separated by little relative to the axis range; training times differ far more than scores; the FinBERT row is contaminated (an upper bound); large off-diagonal cells fall in similar places across models (observation, not attributable) |
| 05 NER | Scores near ceiling because most distinct test sentences appear verbatim in training; predicted and true entity counts coincide per type; token-level and entity-level metrics agree only because the data is saturated (they would not on real data) |
| 06 Cross-dataset | `ProsusAI/finbert` accuracy on FinMarBa a little above the majority rate (published PhraseBank figures far higher; not reproduced); the model predicts neutral far less often than the labels use it; example: a dollar-slump headline labeled by two tickers moving opposite ways |
| 07 News signals | Neither news surprise nor weighted surprise distinguishable from zero; ICIR far below 0.5; t within noise; multiplying by sign(sentiment) raises IC but from ~0 to ~0; the bucket sort runs opposite to construction; universe = high-attention names only |
| 08 Evaluation | All four signals (`weighted_surprise`, `sentiment_mean`, `sentiment_momentum`, `coverage_count`) have ICIR ≈ 0 at every horizon (1/5/20d); quintile sort non-monotone; a typical date has single-digit names and ≥1 tie. Conclusion: construction and evaluation method demonstrated, no signal produced |
| 09 Filing signals | S&P 500 10-Q MD&A 2017–2021, AlgoSeek prices, default 50 most-filed symbols. MD&A length sharply peaked a few thousand words in with a long right tail (median ~7,800, mean ~9,500 words); filings per calendar quarter repeat a four-quarter pattern with one quarter at ~1/5 (10-K quarter). `sentiment_mean` is a single hump with bulk right of zero (management tone skews positive); `sentiment_std` a symmetric hump well above zero; `sentiment_neg_pct` near zero, `sentiment_pos_pct` to its right. Narrative-change distances hump at small values with a thinning tail. IC table (15 pairs): bars small relative to the axis, on both sides of zero, almost every 95% interval crosses zero, intervals long relative to bars. Quintiles on `fwd_20d`: neither signal monotonic; `sentiment_mean` tallest at Q1, `narrative_change` tallest at Q5 with Q1 second — U-shapes, not spreads. Verdict: screening-grade only; the construction (PIT join, dedupe, cluster bootstrap) is the deliverable more than the alpha |

**Not visible in the notes (do not invent):** exact accuracies, Hits@k, IC values, CI bounds, p-values and counts are printed at runtime; `EvalConfig` field values in NB08; NB04's per-model `num_train_epochs`/`batch_size`; NB07's coverage filter size and `MAX_ARTICLES`; whether NB02's benchmark masks all 500 portfolios or a subset; whether the filing panel carries acceptance timestamps; the book's 10.3/10.4 architecture details and 10.5's full signal-validation protocol and pre-train → adapt → fine-tune cascade. NB09's narrative says "sentence-level" scoring but the code chunks by paragraph (1500 chars) with sentence splitting only as a fallback; a plotting cell rebinds `signals` to a list of names, shadowing the DataFrame (harmless in order, a trap when cells re-run).

## Related references

- `chapters/09_model_based_features.md` — IC analysis and factor-evaluation conventions the text signals are screened with.
- `chapters/08_financial_features.md` — momentum/coverage-style feature construction that text factors join.
- `chapters/07_defining_the_learning_task.md` — forward-return labels (1/5/20d), horizon choice, overlapping-label dependence.
- `chapters/11_ml_pipeline.md` — purged/walk-forward CV and the "model trained before the evaluation window" discipline for text checkpoints.
- `chapters/12_gradient_boosting.md` — consuming saved text features in GBMs; `12_gradient_boosting/10_shap_nlp_sentiment.py` for token/SHAP attribution.
- `chapters/13_dl_time_series.md` — sequence models and attention beyond text.
- `chapters/04_fundamental_alternative_data.md` — SEC filings, 13F positioning and news as alternative data sources; point-in-time availability.
- `chapters/02_financial_data_universe.md` — point-in-time joins, survivorship and universe selection.
- `chapters/16_strategy_simulation.md` — backtesting the saved `news_features.parquet` / `filing_signals.parquet` factors.
- `chapters/22_rag_financial_research.md` — embeddings and chunking reused for retrieval over filings.
- `chapters/23_knowledge_graphs.md` — entity resolution and NER outputs as graph inputs.
- `chapters/26_mlops_governance.md` — model cutoffs, checkpoint provenance, drift.
- `case_studies/us_equities_panel.md` — the US equities price panel used for FNSPID forward returns.
- `case_studies/us_firm_characteristics.md` — firm-level panel the filing signals can be merged with.
- `libraries/ml4t_data.md` — loaders (`load_financial_phrasebank`, `load_sp500_10q_mda`, `load_sp500_daily_bars`) and download scripts.
- `libraries/ml4t_engineer.md` — feature engineering helpers (lags, rolling, cross-sectional ranks).
- `libraries/ml4t_diagnostic.md` — IC/ICIR, HAC and bootstrap diagnostics.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `workflow.md`, `companion_repo.md` — cross-cutting indexes.
- **Further reading:** Mikolov et al. 2013 (Word2Vec); Pennington et al. 2014 (GloVe); Gabaix, Koijen, Richmond, Yogo 2025 (Asset Embeddings, NBER 33651); Vaswani et al. 2017 (Attention Is All You Need); Devlin et al. 2019 (BERT); Warner et al. 2024 (ModernBERT); Araci 2019 and Huang et al. 2020 (FinBERT); Hu et al. 2021 (LoRA); Reimers & Gurevych 2019 (Sentence-BERT); Loughran & McDonald 2011, 2020; Tetlock 2005; Bybee et al. 2023, 2024 (narrative factors); Bhargava et al. 2023 (narrative surprise); Robertson & Zaragoza 2009 (BM25); Hochreiter & Schmidhuber 1996 (LSTM); Wu et al. 2023 (BloombergGPT); Xie et al. 2024 (FinBen); Lundberg et al. 2017 (SHAP); Malo et al. 2014 (Financial PhraseBank); Firth 1957 (distributional hypothesis).

## Glossary

- **Distributional hypothesis** — a word is characterized by the words it co-occurs with; "similar" = used in the same places.
- **Skip-gram / CBOW** — predict neighbors from the word / predict the word from its neighbors (`sg=1` / `sg=0`).
- **Negative sampling** — random words pushed away per update to avoid collapse to one point.
- **Context window** — words on each side counted as neighbors.
- **Portfolio sentence** — an institution's holdings ordered by value, treated as a token sequence.
- **Hits@k** — fraction of masked items found in the top-k predictions; **MRR** — mean reciprocal rank of the true item (≤ 1).
- **Popularity baseline** — predictor that always returns the most-held items.
- **TF-IDF** — term frequency × inverse document frequency weighting.
- **Mean pooling** — attention-mask-weighted average of token embeddings into one vector; or averaging chunk-level scores/embeddings into a document value.
- **News surprise** — cosine distance between today's news embedding and the mean of the prior 20 news-days' embeddings; **weighted surprise** — surprise × sign(mean sentiment); **sentiment momentum** — mean sentiment minus its value 20 news-days earlier.
- **Narrative change** — `1 − cosine_similarity` between a company's consecutive MD&A embeddings.
- **IC** — daily cross-sectional Spearman correlation between signal and forward return; **ICIR** — mean IC / std IC (not computed in NB09); **pooled IC** — one Spearman over all (filing, return) pairs regardless of date.
- **Majority-class rate** — accuracy of always predicting the most common label; the zero point for skill.
- **Macro F1** — unweighted mean of per-class F1.
- **BIO tagging** — Begin/Inside/Outside span labels; **entity-level F1** — span correct only if extent and type both match (`seqeval`).
- **Test-set contamination** — a checkpoint trained on sentences in your test split.
- **Label provenance** — how labels were produced (annotator judgment vs price-move sign).
- **join_asof forward** — align to the next existing date ≥ target; the PIT-safe direction.
- **MD&A** — Management's Discussion and Analysis section of a 10-Q/10-K; **10-Q / 10-K** — quarterly / annual filings (the 10-K replaces Q4's 10-Q).
- **Filing date** — SEC acceptance date, the point-in-time anchor; **period_end** — fiscal quarter end reported on; not investable timing.
- **Chunking** — splitting long text into ~512-token pieces for transformer scoring, then aggregating.
- **sentiment_mean / sentiment_std / pos_pct / neg_pct** — mean, dispersion and sign shares of chunk scores (`label_sign × confidence`) within a filing.
- **Cluster bootstrap** — resampling whole clusters (symbols, portfolios) with replacement to respect within-cluster dependence; **percentile interval** — CI from the 2.5th/97.5th percentiles of replicates; **percentile-method test inversion** — p = smallest alpha whose percentile interval excludes zero.
- **HAC** — heteroskedasticity-and-autocorrelation-consistent standard errors (Newey-West) for per-date IC series.
- **Breadth** — number of assets in a cross-section on a given date.
- **Quintile analysis** — sorting observations into five signal buckets and comparing mean forward returns.
