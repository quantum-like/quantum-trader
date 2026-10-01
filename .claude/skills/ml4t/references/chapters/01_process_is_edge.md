# Chapter 1: The Process Is Your Edge

> Chapter 1 argues that durable trading performance comes from a disciplined research process, not from a more sophisticated model: markets change (structural breaks, regimes, data drift, concept drift), so ML for trading is an adaptation problem and the workflow must carry point-in-time-correct data, scoping invariants, an evidence boundary between exploration and confirmation, deployment discipline, monitoring, and feedback from live trading back to research. It places causal inference and generative AI inside that workflow and names the failure modes they add (leakage, hallucination, workflow bloat). Section 1.4 makes non-stationarity operational with two notebooks (`01_process_is_edge/factor_regimes`, `01_process_is_edge/macro_regimes`) that fit Gaussian-mixture regimes on AQR factor returns and FRED macro indicators and show why whole-sample regime labels are a risk lens and a descriptive tool, never a timing signal. This file equips an agent to run and audit a regime analysis end to end: pick a cluster count by stated objective, name clusters by an external rule, validate against unseen evidence, perturb the fit, report conditional statistics honestly, and decide when a label may be acted on.

## When to use this reference

- Asked to detect, estimate, cluster or label market regimes (GMM, k-means, HMM) on returns, factor series or macro indicators.
- Asked whether a regime label may be used as a model feature, a timing signal or an allocation rule.
- Choosing the number of clusters/regimes, or reconciling BIC vs AIC vs silhouette disagreement.
- Computing per-regime statistics (return, vol, Sharpe, drawdown) or any conditional / "restricted stream" path statistic.
- Building a monthly panel from FRED or other mixed-frequency macro data (resampling stamps, level vs rate, duplicates, staleness, metadata).
- Validating or stress-testing an unsupervised fit (seed/window perturbation, adjusted Rand index, episode and revisit counts).
- Loading AQR Century of Factor Premia data via `AQRFactorProvider` or FRED data via `load_macro` / `load_macro_initial_release`.
- Explaining the ML4T workflow, the evidence boundary, scoping invariants, trial logging, sealed holdouts, selection-aware evaluation, or how causal inference and GenAI fit the workflow.
- Designing governance for an independent researcher (documentation, checkpoints, stop criteria) or diagnosing why a static model degraded.
- Running or debugging `01_process_is_edge/factor_regimes.py`, `01_process_is_edge/macro_regimes.py`, or the `figure_1_5` / `figure_1_6` `inputs.npz` handoff.

## Core ideas (the why)

### Chapter-level position (README sections 1.1-1.5; book prose itself not in the digest)

| Idea | Content | Operational consequence |
|---|---|---|
| Process is the edge (1.1) | The model is rarely the bottleneck; what fails is the research process around it. Durable performance comes from discipline that survives changing markets, noisy signals and frictions. | Spend effort on data correctness, scoping, evaluation and monitoring before model search. |
| ML for trading is an adaptation problem (1.1) | Markets change: structural breaks, regimes, data drift, concept drift; online detection catches change as data arrive. Static models degrade. | Build detection and feedback loops from live trading back to research into the workflow; do not treat research as a model-selection contest. |
| The ML4T workflow is a lifecycle, not a script (1.2) | Point-in-time-correct data infrastructure; scoping invariants fixed before research starts; iterative research modules (features, models); realistic strategy design; deployment discipline; monitoring; auditable artifacts and clear handoffs. | Treat research as a managed lifecycle with artifacts and handoffs. |
| Evidence boundary (1.2) | Explicit separation of exploration from confirmation, preserved by trial logging (every trial recorded so selection can be accounted for), sealed holdouts (data untouched until confirmation) and selection-aware evaluation (statistics that account for how many things were tried). | Log every trial; never open the holdout during exploration; report selection-aware statistics. |
| Causal inference and GenAI sit inside the workflow (1.3) | Causal tools sharpen mechanism, assumptions and diagnosis. GenAI expands research throughput and unstructured-data handling but adds leakage, hallucination and workflow bloat. | New tools raise the value of discipline rather than replacing it. |
| Regimes are a risk lens, not a timing signal (1.4) | Regimes support explanation, robustness checks and live monitoring; they are useful when they identify adverse environments and connect them to predefined risk actions. | Tie an identified environment to a predefined action; never read a label as a forecast. |
| Institutions vs independents (1.5) | Institutions get friction and review for free; independents must manufacture governance through documentation, checkpoints and explicit stop criteria. Reusable infrastructure compounds research quality over time. | Write stop criteria before starting; invest in reusable loaders, harnesses and artifact contracts. |
| Implementability checks and monitoring logic (1.5) | Named as the practical tools for diagnosing strategy vulnerabilities. | Test that a strategy can be executed (costs, capacity, timing) and keep watching it after deployment. |

### factor_regimes: the author's reasoning (section 1.4)

- A **regime** is a stretch of time over which the joint behaviour of returns (means, volatilities, correlations) is stable enough to be treated as one environment; nobody publishes a label, so it must be estimated.
- **The number of clusters is a decision, not an estimate.** BIC, AIC and silhouette reward different objectives and will disagree; pick the criterion whose objective matches what the labels are for, and state which and why. Labels that must appear in a risk report need clusters "distinct enough to survive being described", which BIC's sample-scaled penalty and the silhouette's separation test both ask for; AIC's predictive-likelihood objective buys slivers with their own covariance that "is not a regime anyone can name".
- **Cluster numbers carry no meaning until something outside the model names them.** A different seed can renumber clusters without changing the partition.
- Naming by mean equity return per cluster uses the whole sample "exactly as fitting them did": it describes the partition, "not a rule that could have been applied in 1929".
- **Condition on the environment before trusting a factor's average.** A full-sample premium can be earned entirely in one environment; a strategy that pays in both regimes is a diversifier, one that pays only in calm markets "is leverage on the market in disguise". This is what makes a regime label worth estimating even when nothing can act on it.
- **State how every conditional statistic was built.** Means survive conditioning (order does not matter); path statistics (drawdown, run length, time to recovery, stop-loss) become a different statistic once the path is cut up. Reporting one without saying so is "how a number that means one thing gets read as another".
- **Fitting on the whole sample buys description, not prediction.** Every label is informed by later months, so labels cannot be features. Walk-forward fitting (Chapter 6 onward) is what makes them usable.
- **Short episodes decide what the model is for.** A label that changes several times a year cannot drive reallocation (turnover paid on every switch, including reversals); it is a lens on history and one input to a risk report. A model built to be acted on carries a transition structure (HMM) and is fitted forward in time.
- **A mixture has no notion of time.** Nothing prefers a month to keep its neighbour's label, so adding clusters makes timelines speckle rather than carve eras.

### macro_regimes: the author's reasoning (section 1.4)

- **Why macro inputs:** they are not returns, so a regime from them is a statement about conditions that can be *checked* against the market; a regime from returns "is at risk of restating the volatility it was fitted on". The validation half "separates a regime model that has found something from one that has partitioned the calendar".
- **Level or rate is the first decision, not a preprocessing detail.** A trending column makes elapsed time the dominant direction of variance; the partition comes back as "a partition of the calendar wearing economic names".
- **A separation score is not a validation.** Silhouette answers "are these groups far apart in the space they were fitted in"; a regime model must answer "would this label have told me something the next time conditions like these arrived". Counting episodes and revisits is the cheap way to ask the second question.
- **More series is not more information.** Nine Treasury maturities, a spread stored twice and two GDP measures are one panel with a few directions in it.
- **"The clustering is estimated, the naming is asserted."** Write the naming rule in one place so it can be argued with.
- **Validate against something the model never saw.** Equity volatility and drawdown are an independent check for a clustering fitted on macro data alone. Report which half passed and which did not.
- **Perturb the fit before writing about it** and publish the perturbation table beside the result: "anything downstream that would break under an agreement of this size is not supported by this model, whatever the figure looks like." "The response is not to pick the fit whose names read best."
- **Macro regimes confirm, they do not warn.** Unemployment is measured over a month and published in the next; the policy rate moves in response to conditions the market has already priced; the crisis-aftermath label arrives with the crash, not before it. Connect the environment to a predefined risk action; do not read it as a forecast.
- **Joint clustering over single-indicator thresholds:** indicators are correlated (the same mechanism seen from two sides); the joint model reads all positions at once, which is how it separates a tightening cycle from a recovery although both have an unusual short rate.
- **The tree does not choose the panel.** Hierarchical clustering says which columns duplicate each other; which to keep is decided on what the column means and whether it is published in time to be read when a decision is made.
- "The model is ordinary and takes four lines to fit; what separates a usable result from an anecdote is the check that comes after it, and the discipline to report the check when it is unflattering."

## Method recipes (the how)

### Recipe 0: Sequencing checklist for any regime model (author's order)

1. Read the coverage/scale table per series (start date, count, mean, std).
2. Drop missing rows (`drop_nulls()`); never forward-fill returns.
3. Decide level vs rate for every input; enter trending levels as YoY % change.
4. Standardize (`StandardScaler`) before any distance-based method.
5. Choose K by a stated objective; report BIC, AIC, silhouette, switches and smallest cluster together.
6. Name clusters by an external, stated rule written in one place.
7. Validate against data the model never saw (vol ordering, drawdown).
8. Count episodes and revisited clusters.
9. Perturb (seeds, window) and publish ARI, silhouette and smallest cluster for each run.
10. State the construction of every conditional statistic.
11. Declare the result descriptive or actionable. Actionable requires walk-forward fitting + point-in-time (initial-release) data + publication lag + persistence (HMM).

### Recipe 1: Gaussian mixture regime model (both notebooks)

```python
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(df)           # zero mean, unit variance per column
model = GaussianMixture(n_components=n, covariance_type="full",
                        random_state=SEED, n_init=10, reg_covar=1e-6).fit(x)
labels = model.predict(x); probabilities = model.predict_proba(x)
```

| Parameter | Default | Why |
|---|---|---|
| `covariance_type` | `"full"` | Each cluster gets its own mean vector and full covariance (9 dims on the factor panel; 4 macro core; 25 macro extended; 10 on PCA scores). |
| `n_init` | 10 | EM restarted from ten random starts; keep the highest likelihood (mitigates local optima). |
| `reg_covar` | 1e-6 | Ridge on covariance diagonals for numerical stability. |
| `random_state` | `SEED = 42` | Set globally via `set_global_seeds(SEED)` (seeds `random`, NumPy, Torch; sets `PYTHONHASHSEED` for subprocesses only) AND passed to every estimator. |
| Input scaling | `StandardScaler().fit_transform(df)` | Euclidean distance is dominated by the widest column (commodities 4.59%/mo vs bonds 1.04%/mo; unemployment moves points, yield spreads fractions of one). |
| Missing data | `drop_nulls()` | A repeated return "is an invented one, and a mixture model would treat it as evidence for a cluster". |

Hold results in a frozen dataclass `GmmFitResult(model, labels, probabilities, bic, aic, silhouette)`.

### Recipe 2: Choosing the cluster count (factor_regimes)

Sweep `N_REGIMES_GRID = [2, 3, 4, 5, 6]`; for each n record:

| Criterion | Computation | Reading | What it rewards |
|---|---|---|---|
| BIC | fitted log-likelihood penalised by #free parameters x log(sample size) (sklearn `model.bic(x)`) | lower is better | Penalty grows with sample; reluctant to add clusters on a long panel. |
| AIC | same with constant penalty 2 per parameter (`model.aic(x)`) | lower is better | Predictive likelihood; keeps buying clusters. |
| Silhouette (GMM) | `silhouette_score(x, labels)`: per month (b - a) / max(a, b), a = mean distance to own cluster, b = to nearest other; averaged | -1..1, higher better; negative = typical month closer to another cluster | Geometric separation only; knows nothing about likelihood. |
| Silhouette (k-means) | `KMeans(n_clusters=n, random_state=SEED, n_init=10)` at the same n | same scale | How much separation comes from the covariance shape the mixture may fit vs plain centroid distance. |
| Switches | `int(np.sum(np.diff(labels) != 0))` | lower = steadier | Not a selection criterion: does the partition "hold still long enough to be called a regime"? |
| Smallest cluster | `int(np.bincount(labels).min())` | months | Not a selection criterion: does every cluster have enough months to describe? |

- Guard: `assert N_REGIMES_SELECTED in N_REGIMES_GRID` before indexing `gmm_grid[N_REGIMES_SELECTED]`.
- Plot convention: BIC and AIC on a zoomed y-axis (plus/minus 15% of the span) because differences are small relative to level; silhouette on its natural scale including zero; `plot_selection_criterion(ax, values, label, best)` highlights the favoured bar in amber.
- Rule: choose by purpose. Describable, persistent clusters for a risk report favour BIC + silhouette; AIC alone is never sufficient.

### Recipe 3: Naming clusters

| Notebook | Rule | Nature |
|---|---|---|
| factor_regimes | `mean_equity_by_cluster = equity_returns.groupby(labels_2).mean()`; `risk_on_cluster = idxmax()`, `risk_off_cluster = idxmin()`; names "Risk-on" / "Risk-off" | Whole-sample; descriptive only (same lookahead as the fit). |
| macro_regimes | `name_clusters(means)` applies the first matching rule, in fixed order, to each cluster's mean indicators in original units | Asserted; thresholds are judgements about the post-2002 US economy and do not transfer. |

macro_regimes naming rule (first match wins):

1. `unrate > 10` -> "Crisis"
2. `unrate > 6 and dff < 0.5` -> "Recovery" (high unemployment, policy rate at the floor = aftermath)
3. `dff > 3 and t10y2y < 0.5` -> "Tightening" (high policy rate, flat curve)
4. `cpi_yoy > 4` -> "Inflation"
5. `unrate < 5 and dff < 2` -> "Expansion"
6. else -> "Transition"

Not every branch has to fire. Raise `ValueError` if two clusters receive the same name.

### Recipe 4: Per-regime statistics (factor_regimes)

- Annualization: mean monthly x 12; monthly std x sqrt(12) (`MONTHS_PER_YEAR = 12`).
- Rolling volatility: `equity_returns.rolling(12).std() * sqrt(12)` (`ROLLING_VOL_MONTHS = 12`); report per-regime mean/median/max of that series.
- Factor returns by regime: `factors_df.groupby(regime_name).mean() * 12 * 100`; this is an average within a set of months, "not the return of a strategy that switched between them".
- Regime table on the equity index alone (restricted stream = that regime's months concatenated in date order): Months, Share of sample, Return annualized, Volatility annualized, `sharpe_ratio(returns, periods_per_year=12)` from `ml4t.diagnostic.metrics`, and Max drawdown of the restricted stream:

```python
def drawdown_path(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1
```

- Report two volatility ratios side by side: pooled (std of monthly returns in each regime; what a portfolio holding through the regime experiences) and rolling (mean of the trailing-12m series per regime; what a monitoring dashboard would show).
- Restricted-drawdown diagnostic: trough = `drawdown.idxmin()`; peak = `cumprod().loc[:trough].idxmax()`; then compute the index's own drawdown over the same calendar span for comparison.
- Episodes: `episode_id = (labels != np.roll(labels, 1)).cumsum()`; group by (regime, episode) size; report mean length, longest, count. Switch frequency = switches / months.

### Recipe 5: Point-in-time-honest monthly macro panel (macro_regimes)

```python
def to_monthly(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    return (frame.select([DATE_COL, *columns]).sort(DATE_COL)
            .group_by_dynamic(DATE_COL, every="1mo", label="right")
            .agg([pl.col(c).last() for c in columns])
            .with_columns(pl.col(DATE_COL).dt.offset_by("-1d")))
```

- `label="right"` stamps the January window as 1 February so the row holds only January observations; `offset_by("-1d")` steps the stamp back to 31 January so downstream dates (figure axes, printed dates, npz arrays) are not a month late. Keep right labelling (with `label="left"` the row could hold later observations) and step back a day.
- `CORE_SERIES = ["unrate", "dff", "t10y2y", "cpiaucsl"]`; convert `cpi_yoy = cpiaucsl.pct_change(12) * 100`, drop `cpiaucsl`, drop NaNs (costs 12 months: the panel starts 2003 although `PANEL_START = 2002-01-01`).
- The FRED loader carries non-daily series forward on a daily grid, so read native frequency from `load_macro_metadata()` (columns `series, source_id, native_frequency, group, description, kind, formula`). Build the inventory with `reindex` so an undescribed column shows up as "not described in the metadata file" rather than stopping the notebook.
- Guard: raise `ValueError` if any `CORE_SERIES` column is absent.

### Recipe 6: Validation against the market (macro_regimes)

- `sp500["close"].resample("ME").last()`; `returns = close.pct_change()`; reindex to the macro month-end index; raise if any close is missing.
- Annualized vol per regime = std of the regime's monthly returns x sqrt(12) x 100.
- Drawdown = `aligned["close"] / aligned["close"].cummax() - 1` computed over the **whole index in calendar order**; the per-regime value is the deepest reading observed in one of that regime's months (the worst point of the index while the regime was in force, *not* the drawdown of a portfolio held only in it). Contrast with factor_regimes' restricted stream.
- Order regimes by volatility for the Figure 1.6 panel; that ordering is "monotone by construction and says nothing on its own"; the drawdown column is the informative one.

### Recipe 7: Stability under perturbation (macro_regimes)

- Refit the core GMM with `random_state in (0, 7, 123, 2024)` on the full panel, and with `SEED` after dropping the first 1, 3, 6, 12 months.
- Agreement = `adjusted_rand_score(core_labels, labels)` (for truncations compare against `core_labels[dropped:]`): share of month pairs on which two partitions agree, rescaled so random = 0 and identical = 1; invariant to cluster numbering.
- Record silhouette and smallest cluster for each perturbation; publish the table beside the result.

### Recipe 8: Counter-experiment on the extended panel (macro_regimes)

1. Take every value column; `to_monthly`; filter `>= PANEL_START`; drop columns with missing share > `MAX_MISSING_SHARE = 0.5`; `drop_nulls()`.
2. Intersect with the core panel's months (so the comparison is not partly about the window); standardize with `sklearn.preprocessing.scale`; raise if the indices differ.
3. Duplicate detection: `duplicate_pairs(frame, tolerance=1e-9)`, column pairs whose max absolute difference of standardized series < tolerance.
4. Staleness: `(extended_df.diff() == 0).mean()`, the share of months a column is unchanged from the prior month (a quarterly series carried forward is unchanged 2 months in 3).
5. PCA: `PCA(n_components=min(10, n_cols))`; explained variance ratio and cumulative; loadings `pca.components_.T`; top-4 absolute loadings per component.
6. Fit GMM (4 components) on the 25-column panel, on the 10 PCA scores, and `KMeans` (4 clusters) on the panel; compare silhouettes.
7. Episode diagnostics:

```python
def episode_count(labels):        # maximal runs of consecutive equal labels
    return int(np.sum(np.diff(labels) != 0)) + 1
def revisited_clusters(labels):   # clusters the model returns to after leaving
    starts = [labels[0], *labels[1:][np.diff(labels) != 0]]
    return sum(1 for c in set(starts) if starts.count(c) > 1)
```

A chronological cut produces exactly as many episodes as clusters and revisits none.

8. Assignment-probability heatmap: `imshow(probabilities.T, cmap="Blues", vmin=0, vmax=1)` for core vs extended.

### Recipe 9: Clustering the indicators (macro_regimes)

- Columns: `pdist(extended_df.T)` -> `linkage(..., method="ward")` -> `cophenet(linkage, dist)`; months: the same over `pdist(extended_df)`.
- Ward merges the two groups whose merger adds the least within-group variance. Cophenetic correlation = correlation of tree-join heights with the original pairwise distances; near 1 = faithful; conventional bar `MIN_COPHENETIC = 0.7`.
- Dendrogram with `orientation="right"`, `color_threshold=0`; read the first merge from `column_linkage[0, :2]` and its distance from `column_linkage[0, 2]`.
- Use the tree to find duplicates; decide what to keep by meaning and publication timeliness.

### Recipe 10: Figure conventions (both notebooks)

- `style.add_message_title(ax, claim, subtitle=...)`: titles state the finding. `style.show_with_alt(fig, alt_text)`: every figure ships alt text.
- Swim-lane regime bands via `imshow` of a one-hot matrix (`draw_swimlanes`); decade ticks from the first month of each decade (`decade_ticks`); event rules via `axvline` at the first month of a year (`mark_events`).
- Cumulative equity on a log scale coloured by regime with a `LineCollection` where the segment t-1 -> t takes month t's regime (masking per regime would drop transition segments and draw nothing for one-month episodes).
- Correlation heatmap: lower triangle only (`style.upper_triangle_mask`), `vmin=-1, vmax=1, center=0`, `cmap=style.ml4t_diverging()`.

## Guardrails and pitfalls

- **Whole-sample regime labels used as features or signals** — a GMM fitted on all history assigns labels informed by later months; the label is a description, not a forecast, so using it as a feature is lookahead. Disqualify any label whose fit window includes dates after the label date; build labels walk-forward (Chapter 6+; `09_model_based_features/11_hmm_regimes`, `09_model_based_features/13_regime_as_feature`).
- **Naming clusters from whole-sample outcomes** — risk-on/risk-off assigned by full-sample mean equity return carries the same lookahead ("not a rule that could have been applied in 1929"). State that naming is descriptive; for an actionable model, name from information available at each date.
- **Arbitrary cluster numbering** — a different seed can swap which cluster is 0; code keyed on cluster ids breaks silently and figures do not reproduce. Fix `random_state`, name clusters through an external rule, compare partitions with ARI (label-invariant).
- **EM local optima** — GMM climbs to a local optimum from a random start; two runs can give different partitions (ARI as low as 0.73 here). Use `n_init=10`, a fixed seed, and publish a perturbation table across seeds.
- **Model-selection criteria disagree** — BIC -> 2, silhouette -> 2, AIC -> 6 on the factor panel; AIC buys slivers of 3 months. Choose by purpose; report Switches and Smallest cluster alongside; state which criterion decided and why.
- **Tiny clusters** — a cluster of 3-8 months cannot be described and may send the naming rule down a different branch ("not a regime anyone can name"). Detect with `np.bincount(labels).min()`; treat small-cluster partitions as unsupported.
- **Mixture has no time structure** — nothing prefers a month to keep its neighbour's label; episodes are short (risk-off mean 2.1 months) and adding clusters speckles the timeline. For persistence use an HMM with transition structure; always count switches.
- **Acting on a label that flips several times a year** — 267 switches in 97 years (about one every 4.4 months) means turnover and transaction costs on every switch, including reversals, whether or not the call was right. Use the label as a risk lens / report input; require persistence before any allocation rule.
- **Forward-filling missing returns** — repeating last month's return is an invented observation the mixture treats as evidence for a cluster. Use `drop_nulls()`; report coverage per series before fitting.
- **Unscaled inputs** — columns on different scales let the widest column dominate Euclidean distance. Apply `StandardScaler` before any distance-based method.
- **Conditional path statistics misread** — the restricted-stream max drawdown is -76.9% "over 34 years" for risk-off vs -55.4% for the index over the same span: compounding a cut-up path concatenates losses from different decades with no recovery between; "nobody lived through" it. State the construction of every conditional statistic; means survive conditioning, path statistics (drawdown, run length, time to recovery, stop-loss) do not; prefer the macro_regimes construction (drawdown over the whole index, deepest reading inside each regime) when the question is "where was the index when conditions were X".
- **Reporting one of two defensible volatility ratios as "the" ratio** — pooled 2.23x vs rolling 1.30x answer different questions. Report both with their definitions.
- **Revised macro data (not point-in-time)** — FRED serves the latest revision; unemployment, industrial production and price indices are restated, so the value read for 2008 was not visible in 2008. Descriptive use only; a strategy must read `load_macro_initial_release()` (ALFRED vintages, `data/macro/download_alfred.py`) and respect publication lag (a month's unemployment is released in the first days of the next).
- **Publication lag and policy reaction** — macro labels arrive with or after the event (the deepest drawdown, Feb 2009, sits in "Recovery"); the regime a risk report most wants warning of is the one the model can only confirm. Connect the identified environment to predefined risk actions; never read it as a forecast.
- **Month-stamping error in resampling** — `group_by_dynamic(label="right")` stamps January as 1 February, so every downstream date is a month late. Apply `offset_by("-1d")`; keep right labelling for honesty.
- **Trending levels in a clustering** — CPI level (or GDP, payrolls, M2) makes elapsed time the dominant variance direction; the partition becomes calendar eras wearing economic names (extended model: 4 episodes, 0 revisited). Enter trending series as rates of change (YoY %); check `episode_count` and `revisited_clusters`.
- **Silhouette as validation** — extended panel 0.419 > core 0.327 (k-means 0.448) because consecutive eras of a trending panel are far apart in feature space; the score rewards the failure. Validate against evidence the model never saw and against recurrence (does the model return to a cluster?).
- **Panels assembled by coverage filter alone** — "every series with <= 50% missing" admits 9 Treasury maturities, a duplicated spread (`t10y2y` = `YIELD_CURVE_SLOPE`), quarterly GDP carried forward 67% of months, three price-index levels. Detect with `duplicate_pairs`, `(df.diff() == 0).mean()`, PCA (2 PCs = 80% of 25 columns) and the Ward dendrogram first merge at 4.0e-15. Choose series by meaning and timeliness.
- **Derived columns duplicating raw ones** — metadata `formula` says `YIELD_CURVE_SLOPE = DGS10 - DGS2`, already present as `t10y2y`; double weight in the distance. Read `load_macro_metadata()` `kind`/`formula`; run the duplicate check.
- **Mixed native frequencies on a monthly grid** — quarterly series repeat each value three times: fake persistence and double counting. Check `native_frequency` in metadata and the staleness share.
- **Metadata drifting behind data** — a metadata file maintained beside the data file can fall behind. `reindex` columns against metadata so undescribed columns are flagged, not dropped or fatal.
- **Different windows across compared models** — the core panel waits 12 months for YoY CPI, the extended panel waits for the Fed balance sheet series; the comparison becomes partly about the window. Intersect indices and assert equality.
- **Too few repetitions of a condition** — 2003-2025 holds one severe recession, one pandemic and one inflation episode; the sample cannot say how often a condition recurs, which is the direct cause of perturbation instability. Report the limitation; prefer longer panels where the question allows.
- **Thresholds that do not transfer** — the naming-rule thresholds were set for the post-2002 US economy. Keep the rule in one place, state its assumptions, re-derive for other countries or eras.
- **Century-long panel treated as stationary** — the GMM treats 1927 and 2024 as draws from the same distributions despite changed asset-class composition and microstructure. State as a known limitation; monthly resolution resolves nothing faster than a month.
- **Logging noise hiding problems** — `AQRFactorProvider` logs via structlog. `structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))` keeps load cells readable while warnings stay visible.
- **Grid/constant mismatch** — narrowing `N_REGIMES_GRID` without moving `N_REGIMES_SELECTED`. The explicit `assert` before `gmm_grid[N_REGIMES_SELECTED]` catches it.
- **Chapter-level failure modes (README, less detail)** — GenAI adds leakage, hallucination and workflow bloat; multiple testing is handled by trial logging and selection-aware evaluation; sealed holdouts protect confirmation; implementability checks protect against strategies that cannot be executed; static models degrade under drift, so monitoring and feedback loops are mandatory.

## Decision rules and defaults

| Setting | Default | Note |
|---|---|---|
| `SEED` | 42 | The only Papermill-tagged parameter cell in each notebook; set via `set_global_seeds(SEED)` and per-estimator `random_state`. |
| GMM | `covariance_type="full"`, `n_init=10`, `reg_covar=1e-6` | Identical in both notebooks and in the stability refit helper. |
| k-means | `n_init=10`, `random_state=SEED` | Comparison baseline only. |
| `N_REGIMES_GRID` | `[2, 3, 4, 5, 6]` | Factor-panel sweep. |
| `N_REGIMES_SELECTED` | 2 | Factor panel, on BIC + silhouette (both favour distinct, describable clusters). |
| `N_REGIMES` (macro) | 4 | Inherited from the Two Sigma regime study (four regimes over an 18-factor panel; presumably Botte & Bao 2021, the notebook does not name it): "a choice, not a result". |
| `MONTHS_PER_YEAR` | 12 | Annualization factor. |
| `ROLLING_VOL_MONTHS` | 12 | Short enough to move inside a regime, long enough that one month does not dominate. |
| `PANEL_START` | 2002-01-01 | FRED file begins 2000; the two-year bound is roughly a warm-up; effective clustering start 2003 after YoY CPI. |
| `MAX_MISSING_SHARE` | 0.5 | Coverage filter for the extended panel (demonstrates the trap). |
| `MIN_COPHENETIC` | 0.7 | Conventional bar for a faithful dendrogram. |
| `duplicate_pairs` tolerance | 1e-9 | On standardized columns. |
| Stability protocol | seeds (0, 7, 123, 2024); drop-first (1, 3, 6, 12) months | Report ARI, silhouette, smallest cluster for each. |
| Test harness | no `tests/overrides.yaml` entry for ch01 | Production defaults: 300 s timeout, `MPLBACKEND=Agg`, `ML4T_OUTPUT_DIR` redirected to a seeded temp dir; only `SEED` is injected. |

Rules:

- Choose K by the purpose of the labels; always report Switches and Smallest cluster beside BIC/AIC/silhouette; AIC alone is never sufficient.
- Use regimes for explanation, robustness checks, risk reports and live monitoring tied to predefined actions. Do not use them for timing or reallocation when fitted in-sample or when episodes are short.
- Go/no-go for an actionable label: fitted forward in time; inputs available at the decision date (initial release + publication lag); persistence much longer than the rebalancing-cost horizon. Otherwise mark the result "descriptive".
- Treat any downstream claim that would break under the observed perturbation agreement (ARI 0.73-0.82 here) as unsupported by the model; do not pick the fit whose names read best.
- Prefer joint clustering over single-indicator thresholds; rates over levels for trending series; a few meaningful, timely series over a coverage-filtered panel.
- Within the workflow: log every trial, keep holdouts sealed until confirmation, evaluate with selection-aware statistics (section 1.2).
- Independent researchers: manufacture governance via documentation, checkpoints and explicit stop criteria (section 1.5).

## Code patterns and APIs

Imports actually used by the chapter notebooks:

```python
from ml4t.data.providers import AQRFactorProvider      # ml4t_data/ml4t/data/providers/aqr.py
from ml4t.diagnostic.metrics import sharpe_ratio       # ml4t_diagnostic/.../metrics/risk_adjusted.py
from data import load_macro, load_macro_metadata, load_sp500_index   # repo data/ package
from utils.paths import get_output_dir
from utils.reproducibility import set_global_seeds
import utils.style as style   # COLORS, FIGSIZE, PAGE_WIDTH, add_message_title, show_with_alt,
                              # zero_line, upper_triangle_mask, ml4t_diverging
from sklearn.mixture import GaussianMixture; from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler, scale; from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage; from scipy.spatial.distance import pdist
import polars as pl, pandas as pd, numpy as np, structlog, seaborn as sns
```

| API | Signature / behaviour |
|---|---|
| `AQRFactorProvider().fetch(dataset, region=None, start=None, end=None) -> pl.DataFrame` | `timestamp` + factor columns, returns in decimals; reads `{data_path}/{dataset}.parquet` under `$ML4T_DATA_PATH/factors/aqr`; raises `DataNotAvailableError` naming `uv run python data/factors/aqr_download.py`. Dataset `"century_premia"` (Ilmanen et al. 2021; monthly; start 1920-01; xlsx `Century-of-Factor-Premia-Monthly.xlsx`, `skiprows: 17`; category `long_history`). Other families: `qmj_factors`, `bab_factors`, `tsmom`, VME, `commodities`. |
| AQR columns used | `"All asset classes Value"`, `"All asset classes Momentum"`, `"All asset classes Carry"`, `"All asset classes Defensive"`, `"US Stock Selection Value"`, `"US Stock Selection Momentum"`, `"Equity indices Market"`, `"Fixed income Market"`, `"Commodities Market"`; `EQUITY_COLUMN = "Equity indices Market"`; `SHORT_NAMES` maps to Value, Momentum, Carry, Defensive, US Value, US Momentum, Equity, Bonds, Commodities. |
| `load_macro(series=None, start_date=None, end_date=None)` | Wide polars frame: `timestamp` (Date; on-disk `date` renamed) + one column per series from `$ML4T_DATA_PATH/macro/fred_macro.parquet`; raises `DataNotFoundError(requires_api_key="FRED_API_KEY", download_script="data/macro/download.py")`. |
| `load_macro_initial_release(...)` | ALFRED vintages from `fred_macro_initial_release.parquet` (`data/macro/download_alfred.py`); required for any strategy use of macro data. |
| `load_macro_metadata()` | Columns `series, source_id, native_frequency, group, description, kind, formula`. |
| FRED columns present | dff, dgs1, dgs2, dgs3, dgs5, dgs7, dgs10, dgs20, dgs30, t10y2y, vixcls, icsa, walcl, cpiaucsl, cpilfesl, pcepi, unrate, payems, civpart, indpro, m2sl, gdp, gdpc1, YIELD_CURVE_SLOPE (derived DGS10 - DGS2), YIELD_CURVE_5_10 (derived DGS10 - DGS5). |
| `load_sp500_index()` | `timestamp, open, high, low, close` from the bundled `equities/market/sp500/sp500.csv` (daily from 1980; no download). |
| `sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252, confidence_intervals=False, alpha=0.05, bootstrap_samples=1000, random_state=None)` | Pass `periods_per_year=12` for monthly data. |
| `get_output_dir(chapter, strategy_id, create=True)` | Returns `{chapter_dir}/output/{strategy_id}/`; redirected when `ML4T_OUTPUT_DIR` or `ML4T_CHAPTER_OUTPUT_DIR` is set; `utils.paths.CHAPTERS[1] == "01_process_is_edge"`. |
| `set_global_seeds(SEED)` | Seeds `random`, NumPy, Torch; sets `PYTHONHASHSEED` for subprocesses only. |

Patterns:

- Polars month-end (reusable): `frame.sort(DATE_COL).group_by_dynamic(DATE_COL, every="1mo", label="right").agg([pl.col(c).last() for c in columns]).with_columns(pl.col(DATE_COL).dt.offset_by("-1d"))`.
- Papermill parameter cell: a cell containing only `SEED = 42` under the comment "Production defaults (Papermill overrides these when the test suite runs the notebook)".
- Artifact handoff: `np.savez(ARTIFACT_DIR / "inputs.npz", dates=index.astype("datetime64[ns]").astype("int64"), ...)`. factor_regimes writes `get_output_dir(1, "factor_regimes") / "figure_1_5" / "inputs.npz"` (keys `dates, labels_2, good_regime, rolling_vol, risk_on_mean_vol, risk_off_mean_vol, start_year, end_year`); macro_regimes writes `get_output_dir(1, "macro_regimes") / "figure_1_6" / "inputs.npz"` (keys `dates, macro_labels, regime_order, raw_regime_for_order, annual_vol, max_dd_pct, event_years, event_labels, start_year, end_year`). Consumed by `book/01_process_is_edge/figures/scripts/generate_figure_1_5_factor_regimes_volatility.py` and `generate_figure_1_6_macro_regimes_volatility.py` in the separate book repository (not in this environment), so print figures render from the fit rather than re-estimating.
- Running: `uv run python 01_process_is_edge/<notebook>.py` from the repo root; test mode `uv run pytest tests/test_chapter_notebooks.py -v -k "01_process_is_edge"`. Data prerequisites: `uv run python data/factors/aqr_download.py`; `FRED_API_KEY` + `uv run python data/macro/download.py`. No case study and no `setup.yaml` in this chapter.
- Drill-down paths: `01_process_is_edge/factor_regimes.py`, `01_process_is_edge/macro_regimes.py`, `data/macro/README.md`, `data/macro/loader.py`, `data/equities/loader.py`, `data/factors/aqr_download.py`, `utils/paths.py`, `utils/reproducibility.py`, `utils/style.py`, `tests/test_chapter_notebooks.py`, `tests/overrides.yaml`.

## Evidence from the book

### factor_regimes (`01_process_is_edge/factor_regimes`, executed outputs; AQR data through 2024-12)

- Raw file: 1182 months, 44 series, 1926-07 to 2024-12. Nine series used; 1176 complete months (1927-2024) after `drop_nulls`; US Momentum (starts 1927-01) is the series that shortens the usable history.
- Coverage (mean / std, %/month): Value 0.21/1.40; Momentum 0.30/1.64; Carry 0.21/1.29; Defensive 0.25/1.29; US Value 0.34/4.28; US Momentum 0.67/4.52; Equity 0.65/3.44; Bonds 0.14/1.04; Commodities 0.41/4.59.

| K | BIC | AIC | Silhouette GMM | Silhouette k-means | Switches | Smallest cluster (months) |
|---|---|---|---|---|---|---|
| 2 | 26,170 | 25,617 | 0.270 | 0.234 | 267 | 276 |
| 3 | 26,173 | 25,342 | 0.115 | 0.113 | 448 | 56 |
| 4 | 26,259 | 25,149 | -0.024 | 0.096 | 577 | 3 |
| 5 | 26,388 | 24,998 | 0.006 | 0.087 | 600 | 8 |
| 6 | 26,598 | 24,930 | 0.003 | 0.077 | 654 | 3 |

BIC lowest at 2, silhouette highest at 2, AIC keeps falling to 6; at K >= 4 the silhouette is about 0 or negative and the smallest cluster has 3-8 months.

- Two-cluster model: Risk-on 900 months (76.5%), Risk-off 276 (23.5%).
- Rolling 12-month equity vol by regime (mean/median/max, %): Risk-on 9.8/8.7/29.7; Risk-off 12.7/10.9/31.8.

| Factor | Risk-on (ann. mean %) | Risk-off (ann. mean %) |
|---|---|---|
| Value | +1.6 | +5.3 |
| Momentum | +4.5 | +0.4 |
| Carry | +3.5 | -0.6 |
| Defensive | +4.2 | -0.5 |
| US Value | -1.0 | +20.6 |
| US Momentum | +10.7 | -0.7 |
| Equity | +9.5 | +2.2 |
| Bonds | +0.8 | +4.2 |
| Commodities | +4.0 | +8.5 |

Author's reading: carry and defensive ("safe income" in calm markets) stop paying in risk-off; value and bonds earn more. (Commodities also earn more in risk-off, and US Value is negative in risk-on and strongly positive in risk-off: inference from the output table, not commented by the author.)

- Regime statistics on the equity index (restricted stream): Risk-on return +9.5%, vol 8.6%, Sharpe 1.11, max DD -22.9%; Risk-off return +2.2%, vol 19.1%, Sharpe 0.12, max DD -76.9%. Pooled vol ratio 2.23x vs rolling-average ratio 1.30x.
- Restricted risk-off drawdown: peak August 1986, trough March 2020, -76.9% "over 34 years"; the index itself fell at most 55.4% over the same calendar span. Risk-off average monthly return is positive; the drawdown is an artefact of closing the gaps.
- Episodes: Risk-on mean 6.7 months, longest 110, 134 episodes; Risk-off mean 2.1 months, longest 14, 134 episodes. 267 switches over 97 years, about one every 4.4 months.
- Figure claims: "BIC, AIC and silhouette do not pick the same number of regimes"; "Adding clusters makes the timeline switch more often, not cleaner"; "Risk-off months come in short bursts, not in long bear markets"; "Value and bonds hold up in the risk-off months; carry and defensive do not".

### macro_regimes (`01_process_is_edge/macro_regimes`, executed outputs; FRED data through 2025-12)

- FRED panel 9497 daily rows, 25 series, 2000-01-01 to 2025-12-31; 288 monthly rows (2002-01 to 2025-12); clustering panel 276 months, 2003-2025.
- Core series over the panel (mean, std, min, max): unemployment 5.75, 2.02, 3.40, 14.80; fed funds 1.76, 1.91, 0.04, 5.41; 10Y-2Y 1.06, 0.97, -1.06, 2.84; CPI YoY 2.58, 1.81, -1.96, 8.98.
- Core GMM silhouette 0.327. Cluster means (unrate, dff, t10y2y, cpi_yoy) -> name (months): 0: 4.34, 4.62, 0.02, 3.28 -> Tightening (72); 1: 7.85, 0.11, 1.84, 1.47 -> Recovery (100); 2: 4.19, 1.30, 0.61, 1.91 -> Expansion (51); 3: 5.18, 1.43, 1.46, 4.39 -> Inflation (53). The Crisis and Transition branches did not fire.

| Regime | Months | Annualized vol % | Deepest index drawdown % |
|---|---|---|---|
| Tightening | 72 | 10.4 | 19.4 |
| Expansion | 51 | 12.2 | 14.0 |
| Recovery | 100 | 15.7 | 52.6 |
| Inflation | 53 | 17.7 | 42.2 |

Vol span 7 pp from Tightening to Inflation. The deepest decline (53%, February 2009) falls in "Recovery" (third of four by volatility); the shallowest belongs to Expansion (second). Verdict: the volatility ordering is economically legible (passes); drawdown does not follow it and the crisis-aftermath regime holds the crash (the model confirms rather than warns). Figure 1.6 claim: "Ordering the macro regimes by volatility does not order them by drawdown."

| Perturbation | ARI vs base | Silhouette | Smallest cluster (months) |
|---|---|---|---|
| seed 0 | 1.00 | 0.327 | 51 |
| seed 7 | 0.79 | 0.320 | 16 |
| seed 123 | 0.79 | 0.320 | 16 |
| seed 2024 | 0.81 | 0.308 | 20 |
| drop first 1 | 0.79 | 0.321 | 16 |
| drop first 3 | 0.73 | 0.255 | 4 |
| drop first 6 | 0.82 | 0.323 | 20 |
| drop first 12 | 0.77 | 0.291 | 4 |

Worst agreement 0.73; smallest cluster 4 months vs 51 in the base fit. Some perturbations split off a cluster small enough that the naming rule reaches a different branch.

- Correlations: unemployment and yield spread strongly positive; fed funds strongly negative vs spread; inflation moderately negative vs unemployment; "no pair close to independent".
- Extended panel: the coverage filter keeps 25 of 25 series; 276 months after intersecting. One duplicate pair: `t10y2y` and `YIELD_CURVE_SLOPE`. Stalest column `gdp`, unchanged 67% of months. PCA: PC1 45.6%, PC2 34.8% (cum 80.3%), PC3 7.5%, PC4 5.8% (cum 93.6%), PC5 2.4%, PC10 cum 99.8%. Loadings: PC1 = payems, gdp, t10y2y, YIELD_CURVE_SLOPE (trend); PC2 = dgs10, dgs20, dgs7, dgs30 (long-end level); PC3 = indpro, vixcls, unrate, dgs20; PC4 = icsa, vixcls, YIELD_CURVE_SLOPE, t10y2y. Crisis-timescale series (VIX, claims) appear only in PC3-PC4.
- Silhouettes: core 4 series 0.327; extended 25 series 0.419; extended 10 PCs 0.397; extended k-means 0.448. The score says extended is better.
- Episodes: core 8 episodes, 3 clusters revisited; extended 4 episodes, 0 revisited. The extended model partitioned the calendar into consecutive eras.
- Cophenetic correlation 0.89 for the tree over indicators, 0.71 over months (bar 0.7). First merge: `t10y2y` with `YIELD_CURVE_SLOPE` at distance 4.0e-15. Tree groups: trend-only series; short/medium Treasuries; curve-shape series with unemployment; VIX + claims join last.

### Author's caveats

All labels are descriptive (whole-sample fit, revised FRED data). The century panel is treated as one distribution. Monthly resolution. The post-2002 panel holds one recession, one pandemic, one inflation episode: too few repetitions, hence instability. Naming thresholds are US/post-2002 specific. Re-running after a data refresh will change the numbers and possibly the cluster numbering.

## Related references

- Chapter files: Chapter 6 (walk-forward fitting; what turns a descriptive label into an actionable one); `09_model_based_features.md` (`09_model_based_features/11_hmm_regimes`: HMM with transition structure, fitted forward; `09_model_based_features/13_regime_as_feature`); Chapter 4 (`06_fred_macro_eda`, `07_macro_data_alignment`); `07_defining_the_learning_task.md` (`08_causal_sanity_checks`); Chapter 8 (`04_fundamentals_macro_calendar`); Chapter 16 (`01_backtest_first_principles`, `06_framework_parity`); `17_portfolio_construction.md` (`05_factor_allocation_evidence` also opens AQR factor maps); Chapter 18 (`02_spread_estimation`, `03_market_impact_calibration`); Chapter 26 MLOps/governance (trial logging, sealed holdouts, independent governance details not in this digest). Exact file names for chapters 4, 6, 8, 16, 18 and 26 follow their repo directory names.
- Case-study files: the case-study chapters apply walk-forward regime machinery (specific file names not identified in the digest).
- Library files: `ml4t_data` reference (`ml4t.data.providers.AQRFactorProvider`, `DataNotAvailableError`); `ml4t_diagnostic` reference (`ml4t.diagnostic.metrics.sharpe_ratio`).
- Data docs in the repo: `data/macro/README.md`, `data/macro/download.py`, `data/macro/download_alfred.py`, `data/factors/aqr_download.py`.
- Literature named by the chapter: Botte & Bao 2021 (Two Sigma regime modeling; presumed source of `N_REGIMES = 4`); Ang & Bekaert 2002 (regime shifts in asset allocation); Lo 2004 (Adaptive Markets Hypothesis); Ilmanen et al. 2021 (Century of Factor Premia); Uysal & Mulvey 2021 (regime-switching risk parity); Horvath et al. 2021 (Wasserstein regime clustering); Harvey et al. 2016 (multiple testing); Arnott et al. 2018 (backtesting protocol); Lopez de Prado 2018 (AFML; 10 reasons ML funds fail); Lopez de Prado et al. 2024 and Lopez de Prado & Zoonekynd 2025 (causal factor investing, factor mirage protocol); Pearl 2019 (seven tools of causal inference); Scholkopf et al. 2021 (causal representation learning); Fabozzi et al. 2024 (holism in causal modeling); Fabozzi & Stenholm 2025 (strategic discipline); Luk 2023 and Fang & Moore 2025 (GenAI in asset management); Ryseff et al. 2024 (RAND, root causes of AI project failure); Studer et al. 2021 (CRISP-ML(Q)); Gu, Kelly & Xiu 2020; Giglio et al. 2022; Duffie 2020 (Treasury market after COVID); Easley et al. 2012 (volume clock); Marshall 2023 (stock-bond harmony); Gartner 2018; Lee 2025 (Man Group agentic AI signals).

## Glossary

| Term | Meaning |
|---|---|
| Regime | A stretch of time over which the joint behaviour of returns (means, vols, correlations) is stable enough to be treated as one environment. |
| Structural break / data drift / concept drift / online detection | The README's vocabulary for market change. Definitions not in the digest; drift = input distribution change, concept drift = input-to-target relation change, online detection = detecting change as data arrive (inference). |
| Gaussian mixture model (GMM) | Each observation is a draw from one of K multivariate normals; EM infers the distributions and per-point membership probabilities. |
| Expectation-maximization (EM) | Iterative likelihood climb from a random start; subject to local optima. |
| Full covariance | Each component has its own full covariance matrix. |
| `reg_covar` | Ridge added to covariance diagonals for numerical stability. |
| BIC | Log-likelihood penalised by #parameters x log n; lower is better. |
| AIC | Log-likelihood penalised by 2 x #parameters; lower is better. |
| Silhouette score | Mean over points of (b - a)/max(a, b); -1..1; higher is better; geometric separation only. |
| K-means | Centroid-distance partitioning; no covariance, no probabilities. |
| Switch count | Number of consecutive-month label changes. |
| Episode | Maximal run of consecutive months with the same label. |
| Revisited cluster | A cluster the model returns to after leaving it. |
| Risk-on / Risk-off | Periods when investors add / shed exposure to risky assets; here the clusters with the highest / lowest mean equity-index return. |
| Value, Momentum, Carry, Defensive | AQR long-short cross-asset factors: cheap vs expensive; past-year winners vs losers; high vs low yield; low-risk vs high-risk. |
| Diversifier vs "leverage on the market in disguise" | A strategy that earns in both regimes vs one that earns only in the calm regime. |
| Restricted stream | A regime's months concatenated in date order with the other regime's months removed. |
| Path statistic | A statistic that depends on order (drawdown, run length, time to recovery, stop-loss), unlike a mean. |
| Maximum drawdown | Largest percentage fall from the running high-water mark of the compounded stream. |
| Pooled vs rolling volatility | Std of all monthly returns in a regime vs the mean of a trailing-12-month std series over the regime's months. |
| Hidden Markov model (HMM) | Regime model with a transition structure making staying more likely than leaving; the standard choice for actionable regimes. |
| Walk-forward fitting | Each date's estimate uses only data before it. |
| Point-in-time / initial release | The value of a series as it stood on a given date (ALFRED vintages), as opposed to the latest revision FRED serves. |
| Publication lag | Delay between the period a statistic describes and its release. |
| UNRATE, DFF, T10Y2Y, CPIAUCSL | Unemployment rate; effective fed funds rate; 10-year minus 2-year Treasury yield; CPI (all urban consumers). |
| Yield-curve inversion | T10Y2Y < 0; has preceded most post-war US recessions. |
| `group_by_dynamic(label="right")` | Polars window aggregation stamping each window at its right edge. |
| Adjusted Rand index (ARI) | Pair-agreement between two partitions, chance-corrected (random = 0, identical = 1), invariant to label numbering. |
| Principal component analysis (PCA) | Orthogonal directions of greatest remaining variance; explained-variance ratio; loadings = column weights per component. |
| Ward linkage | Agglomerative clustering merging the two groups whose merger adds the least within-group variance. |
| Cophenetic correlation | Correlation between tree-join heights and original pairwise distances; faithfulness of a dendrogram (conventional bar 0.7). |
| Evidence boundary | The separation between exploration and confirmation in the ML4T workflow. |
| Trial logging / sealed holdout / selection-aware evaluation | Mechanisms that preserve research integrity across many trials. |
| Scoping invariants | Rules fixed before research that do not change during iteration (README term; details not in the digest). |
| Implementability check | A test that a strategy can actually be executed (costs, capacity, timing) (README term). |
| Papermill | Notebook parameterization used by the test harness to override the parameters cell. |
| structlog | Structured logging library used by the AQR provider. |
