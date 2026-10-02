# Chapter 20: Strategy Synthesis

> Chapter 20 passes all nine case studies (CS) through one standardized pipeline (data → features → labels → models → predictions → portfolios → costs → risk overlays → frozen holdout) and reads the cross-section diagnostically. The unit of comparison is each case study's arc from signal to holdout, never a Sharpe league table. Its position: a backtest is only interpretable when the full strategy specification (signal + allocation + overlay) is named, when the headline claim is a paired Sharpe *difference* against the equal-weight baseline with a bootstrap interval, and when every number is a registry read with stated uncertainty. It equips you to run the funnel (signal validity → tradability → cost survival → holdout → risk tolerance), classify why a candidate drops out, and pick the next iteration. Ch20 reads registry results; it generates none.

## When to use this reference

- You must decide whether a candidate strategy "works" after the backtest, cost sweep, overlay sweep and holdout all exist, and need the gate sequence and pass/fail definitions.
- You are aggregating results across several strategies, markets or case studies into one comparison table or funnel figure.
- You need to compare a validation result with a holdout result and want to avoid manufacturing decay (max-of-sweep vs single pick).
- You are asked whether a strategy beats a passive equal-weight baseline and need the paired block-bootstrap recipe (`compute_paired_uncertainty`) and its decision thresholds.
- You are choosing among allocators (EW, inverse-vol, MVO, risk parity, score-weighted, HRP) with the signal held fixed and must judge uplift breadth and selection bias.
- You need breakeven cost, cost-margin ratio and a resilience label from a commission + slippage sweep, or must decide between a bps sweep and an executable-quote cost model (options, intraday).
- You are evaluating whether a risk overlay should be deployed and need the quadrant rule (Sharpe up AND drawdown down, out of sample).
- You are reading `20_strategy_synthesis/output/*.parquet|json` or `case_studies/{cs}/run_log/registry.db` and need the schema, the pins (`RUNG_PINS`, `LABEL_RESTRICTIONS`) and the partial-run refusal rule.
- You are writing a research report and must classify a failed candidate (signal invalidity / implementation infeasibility / evidence-quality failure) and propose a second iteration.
- You are reviewing someone else's synthesis for the documented failure modes (typed constants, absence-as-zero, nondeterministic rung selection, missing EW allocator rows).

## Core ideas (the why)

- **Measurement-first.** Every number is a registry read with known uncertainty (per-fold SE, rank-cluster width). Interpret measurements; do not collapse them into categorical trust labels ("high-confidence / provisional / unreliable"), which hide the trade-off the reader needs.
- **Full strategy specification = signal method + allocation method + risk overlay.** Naming all three is what makes a validation result and a holdout result comparable.
- **Inspect the cluster, not the pick.** A genuine signal shows a thick top: many configurations within one fold-SE of rank 1, pick insensitive to selection-rule perturbation. A thin cluster (large rank1 → rank10 gap) suggests a tail draw, not a stable optimum.
- **Four separate evaluation stages:** signal quality, portfolio translation, cost survival, temporal stability. IC (ranking accuracy) and Sharpe (portfolio construction) are different things; a CS can have one without the other; construction contributes variance IC says nothing about.
- **IC translates into Sharpe only through implementation.** Entry scheme, sizing, cadence and costs sit between. A model comparison reported only in IC has not answered what to trade. The IC leader and the Sharpe leader need not be the same family, and in this registry often are not.
- **The right unit of uncertainty is the paired Sharpe difference** against the passive equal-weight baseline that experienced the same market conditions, with a paired bootstrap CI — not the Sharpe alone.
- **Feature-level survival is a screen, not a predictor of strategy survival.** Univariate IC and joint prediction answer different questions: zero surviving features can still yield a usable model because the model uses combinations no single column carries.
- **Allocation cannot manufacture a signal.** Every allocator reads the same ranking; if the ranking is uninformative, redistributing capital changes the variance of the result, not its expectation, and more free parameters = more ways to fit noise. The spread between allocators is itself the useful number. Equal weight is the baseline worth beating because it has no parameters to fit.
- **Costs are the great equalizer.** The cost gate is set by the instrument traded, not by signal strength; it tests the edge-to-cost ratio, not IC. A cost sweep is informative only at the cost the strategy actually assumes.
- **An overlay is not free.** It cuts the left and right tails together and trades more; a category whose median configuration loses Sharpe is the normal finding. Default = no overlay. Drawdown reduction at a Sharpe cost is insurance, and its worth is a mandate question, not a backtest question.
- **The funnel is the story.** Signal is necessary but not sufficient; failure modes are distinct (signal invalidity / implementation infeasibility / evidence-quality failure), each pointing to a different second-iteration response.
- **Pipeline over model.** Rank in prediction (Ch11–15) and gates passed (Ch16–20) are reported separately. No model family leads everywhere.
- **Evidence quality is not the headline Sharpe.** A managed Sharpe is read alongside the cost environment it was earned in and the holdout that followed it.
- **Three decay mechanisms** between validation and holdout: prediction-quality drift, portfolio-translation drift, structural break / regime change.
- **Absence of a measurement is not a measurement of zero** (recurring across NB03–NB07).
- **The chapter's claim is methodological:** this is the pipeline to run to find out whether a candidate works, not a set of deployable strategies (public low-frequency data, deliberately modest hyperparameter grids, starter configs).

## Method recipes (the how)

### Inputs and upstream requirements

- Test bed: nine CSs — equity ETFs (`etfs`), crypto perpetuals (`crypto_perps_funding`), intraday NASDAQ-100 microstructure (`nasdaq100_microstructure`), equity plus options (`sp500_equity_option_analytics`), monthly firm characteristics (`us_firm_characteristics`), FX (`fx_pairs`), CME futures (`cme_futures`), pure options (`sp500_options`), broad equity panel (`us_equities_panel`). They differ on four dimensions that `overview.parquet` records: asset class, rebalance cadence, universe size and cost assumption (`cost_bps`).
- Upstream requirement: every `case_studies/{cs}/run_log/registry.db` must hold training, prediction and backtest rows on the primary label, plus `causal_runs` rows for §20.8. Holdout rows are written by each CS's own `NN_holdout_predictions` + `NN_holdout_backtest` pair (registered in `prediction_sets` with `split='holdout'`), never by Ch20; when "No holdout evidence" / "Holdout not available" fires, that pair is where to look.
- NB01 input contract: `registry.db` + `config/setup.yaml` per CS only, no JSON inputs ("nothing is hardcoded"). NB08's only hardcoded elements are the HTM cost-cascade figures and the sp500_options cost handling.
- Per-CS drill-down: each CS's `NN_strategy_analysis` notebook (§2 stage-transition waterfall, §6 holdout decay + holdout-vs-benchmark, §7 benchmark-aware diagnostics) is where the detail behind any Ch20 cross-section number lives.

### Selection rule and spine (carrier) resolution — NB01

1. Selected configuration per CS = highest-Sharpe validation backtest across signal, allocation and risk stages **jointly**.
2. Deployed holdout configuration = the same spec retrained on holdout data.
3. Fallback walk: if the holdout retrain yields no usable backtest (degenerate predictions, vol window not matching available history, universe filter rejecting the sample, other generation failure), take the next-highest validation Sharpe with a usable holdout; iterate until one succeeds. The helper feeds the holdout query and the lineage resolver.
4. The selected validation configuration's `prediction_hash` is the **spine / carrier**; cost and risk stages are pinned to it.
5. Apply label and rung restrictions before any ranking:

| Restriction | Rule | Source |
|---|---|---|
| Label restriction (sp500_options) | Pin to hold-to-maturity (HTM) label (`ret_to_expiry` / HTM dispatch) with coherent option costs; exclude the four fixed-horizon straddle labels (vectorized path, generic bps cost) from cluster diagnostics. "Do not retarget." | `LABEL_RESTRICTIONS` |
| Rung restriction (O'Donovan–Yu 2025 cascade) | Rung 1 naive round trip; rung 2 full HTM (demoted variant, §18.8); rung 3 HTM restricted to the liquid bottom-spread quintile = registered strategy. Rungs 1 and 2 share the universe filter, so pin `universe_filter` AND `exit_at_max_days` together. Fetch all rows (`top_n=1_000_000`) before applying `rung["predicate"]`. Repo contents (not inspected in the digest): `sp500_options` → `universe_filter == "liquid"` and `exit_at_max_days` null (rung 3); `nasdaq100_microstructure` → `universe_filter == "cost_feasible"` and `label == "fwd_ret_15m"`, so NASDAQ-100 is pinned too. | `RUNG_PINS` (`predicate`, `universe_filter`, `exit_at_max_days`) |
| Stage-not-applicable | `("sp500_options","allocation")`: HTM short-straddle has fixed 1/n_roll cohort weighting. `("sp500_options","risk")`: HTM expiration structure sets the risk profile. us_firm_characteristics vectorized path has portfolio overlays purged. | `_stage_applicable(cs, stage)` |

CSs without a pin entry skip the filter. `RUNG_PINS` is imported from `case_studies.utils.paired_metrics` (the same module holds `populate_paired_metrics`, which decides the canonical carrier and retired identities); `LABEL_RESTRICTIONS` is defined in `case_studies.utils.strategy_analysis` and NB01 imports it from there (the chapter digest places both in `paired_metrics`; importing `LABEL_RESTRICTIONS` from it fails).

### Rank-1 cluster diagnostics — NB01

- Per CS tuple: (rank1 Sharpe, rank10 Sharpe where ≥10 configs exist, spread, fold-SE, folds-positive). Written to `rank1_cluster_diagnostics.parquet` (legacy filename kept).
- `fold_SE = std(per_fold_sharpe, ddof=1) / sqrt(n_folds)`.
- Thick top = spread (rank1 − rank10) small relative to fold-SE. Temporal stability = folds-positive close to total fold count.
- Thin cluster → treat the pick as a tail draw; expect holdout decay.

### Backtest comparison, Sharpe progression, lineage — NB01

| Artifact | What it holds | Reading rule |
|---|---|---|
| `backtest_comparison.parquet` | Per (CS, stage) Sharpe / CAGR / drawdown; each stage's best taken **independently** | The signal topping one column may be a different model from the allocation topping the next; do not read it as a path. The digest's artifact list also calls it the "canonical record of one spine `prediction_hash` per CS"; the two descriptions conflict, and the independent-best reading is the one the method section gives — treat the spine reading as unresolved |
| `sharpe_progression.parquet` | One `prediction_hash` followed through allocation, costs, risk | `null` = that hash not tested at the stage, not "stage absent" |
| `lineage.parquet` | Selected signal → highest-Sharpe allocation on it → cost-tested version → risk-managed version | Stage-to-stage differences are attributable only where the later stage carries the earlier configuration; paired rows say which transitions qualify |

NASDAQ-100 is excluded from the v3.0 lineage comparison (timing-corrected cost/risk grids deferred to v3.1).

### Paired stationary block bootstrap vs equal-weight benchmark — NB01

Call on daily strategy returns:

```python
compute_paired_uncertainty(c_arr, b_arr, periods_per_year=ppy,
                           case_study=cs, label=label, n_boot=2000, seed=42)
```

| Parameter / output | Value or meaning |
|---|---|
| `n_boot`, `seed` | 2000, 42 |
| Block length | `setup.yaml.labels.{label}.rebalance_step`; fallback = optimal block size; never below the label horizon |
| Minimum series length `_min_paired_n(ppy)` | 6 if ppy ≤ 12 (monthly); 12 if ppy ≤ 52 (weekly); else 21 (daily / 8h / intraday) |
| Benchmark | Side-artifact equal-weight series resolved deterministically per (cs, label): no universe/rung/cadence ambiguity, no fallback-by-recency |
| Outputs | `sharpe_diff` + CI; `ret_diff` (annualized) + CI; `info_ratio` of the daily-return difference; `prob_challenger_wins`; two-sided `p_value` for H0 sharpe_diff = 0 |
| Stored in | `backtest_paired_metrics` with `bootstrap_block_length`, `bootstrap_n`, `benchmark_kind` |
| Confidence level, optimal-block formula | Not visible in the digest (uncertain) |

Six pair types (`benchmark_kind`):

| # | Challenger ↔ benchmark | `benchmark_kind` | Caveat |
|---|---|---|---|
| 1 | Selected signal (overall) ↔ equal weight (overall) | Not named in notes; repo: `equal_weight_side_artifact` (prefix = `SIGNAL_BASELINE_BY_CASE_STUDY.get(cs, "equal_weight")`) | |
| 2 | Selected signal (holdout) ↔ equal-weight holdout window | `equal_weight_holdout_side_artifact` (same prefix rule) | |
| 3 | Selected config holdout ↔ same config validation | `val_rank1_self` | Windows disjoint and unequal. Notes/NB01 comment: both truncated to `min(len(val), len(ho))` and fed to the paired helper. Repo code: `_populate_pair(..., disjoint_windows=True)` passes the untruncated windows to `compute_independent_diff_uncertainty`, which resamples each window independently (own block length per side) and differences the draws; `info_ratio` is NaN (no difference series). `compute_paired_uncertainty` refuses unequal lengths (returns `{}`) rather than truncating. The "truncation caveat" wording on `val_rank1_self` is stale; the live caveat is "disjoint windows, not a paired comparison" |
| 4 | allocation ↔ signal | `signal_leader` | Only if the later stage carries the earlier configuration |
| 5 | risk_overlay ↔ allocation | `allocation_leader` | same |
| 6 | cost_sensitivity ↔ risk_overlay | `risk_overlay_leader` | same; order follows `STAGE_SEQUENCE` |

A stage that does not carry the previous stage's configuration yields no pair. CSs pinned at the signal stage (sp500_options, whose allocation and risk stages are declared not applicable) surface zero transition rows. The notes and an NB01 comment label this "rung 2", but `RUNG_PINS["sp500_options"]` selects `universe_filter == "liquid"` (rung 3, the registered strategy) and NB01's holdout query says the same; the "Rung-2" label in that comment is stale, and only the full-universe rung-2 rows survive for the §18.8 cascade comparison.

Paired reference rule (every row above, and any non-ML reference such as the EW universe, a random top-k book or a fixed rule): the reference runs under exactly the strategy's caps, sizing, rebalance schedule, costs and warmup; any tie-break (e.g. `symbol` ascending on a tie straddling the cutoff, ch16) is stated and orthogonal to the hypothesis's conditioning variables (never a sort on the signal, its inputs or a correlate); if the reference cannot run under those caps (a position cap that only binds for the concentrated book, a short leg the reference cannot hold), the comparison is unpaired, the report says so, and `compute_paired_uncertainty` is not the tool. The same rule is the ch17 "identical inputs and protocol" condition for allocator baselines.

### Stage attrition funnel — independent (NB01) vs cumulative (NB08)

NB01 counts each gate independently against `bt_df` / `holdout_df`; `stage_attrition.json` keys in NB01's own order (the order is irrelevant for independent counts, but it differs from NB08's):

| NB01 key | Test |
|---|---|
| `good_predictor` | `ic_best > 0` (max over the CS's families in `ic_df`) |
| `tradable_gross` | `signal_sharpe > 0` (gross signal-stage Sharpe) |
| `cost_surviving` | `survives_costs` flag in `bt_df` is true (Sharpe > 0 at assumed cost) |
| `risk_tolerable` | `managed_sharpe > 0` |
| `holdout_valid` | `holdout_sharpe > 0` |

NB08 is cumulative, five gates in pipeline order; each gate is applied to the survivors of the one above it:

| NB08 gate | Test |
|---|---|
| 1 Good predictor | Positive validation IC |
| 2 Tradable | Validation ML Sharpe of the **selected configuration** (`backtest.ml_sharpe`) > 0, NOT the risk-stage baseline |
| 3 Cost-surviving | `costs.survives_costs` and `net_sharpe_at_actual > 0`; pass-through if `costs.not_applicable_reason` (sp500_options uses §18.8 option-native bid-ask accounting) |
| 4 Holdout-valid | `holdout_sharpe > 0` |
| 5 Evidence ready | `risk.managed_sharpe > 0` or risk not applicable; NASDAQ-100 (`NASDAQ_ID`) excluded explicitly (point estimate positive, but corrected validation and holdout intervals both cross zero) |

- NB01: a row's failures = 9 − count; adjacent-row differences are NOT gate removals.
- NB08: drop a CS only on a genuine negative at an applicable stage. Use NB08 for strict survivors and named dropouts, NB01 for per-stage attrition.

### Exclusion taxonomy and evidence profile — NB08

`classify_exclusions()` assigns from data:

| Exclusion | Condition | Bucket |
|---|---|---|
| No detectable signal | `best_ic <= 0` | Signal Invalidity |
| Insufficient edge after costs | `net_sr <= 0` and `survives == False` | Signal Invalidity |
| Positive IC but no stable Sharpe | declared in `BUCKETS`; assignment logic not visible (inference: reserved or assigned elsewhere) | Signal Invalidity |
| Cadence-horizon mismatch | declared in `BUCKETS`; logic not visible (inference) | Signal Invalidity |
| Net-negative under realistic costs | sp500_options (hardcoded HTM cascade numbers) | Implementation Infeasibility |
| Unacceptable drawdown | `abs(worst_drawdown_pct) > 50` | Implementation Infeasibility |
| Holdout collapse | `ho_sharpe <= 0` and CS in `positive_val_sharpe` | Evidence-Quality Failure |
| Holdout not available | `ho_sharpe is None` and CS in `positive_val_sharpe` (degenerate/missing holdout predictions) | Evidence-Quality Failure |
| Unreproducible model | declared in `BUCKETS`; logic not visible (inference) | Evidence-Quality Failure |
| No holdout evidence | no holdout row in registry | Evidence-Quality Failure |
| Statistically unresolved | `holdout_sharpe_ci_lo < 0 < holdout_sharpe_ci_hi` | Evidence-Quality Failure |

`build_evidence_profile()`:
- Gate flags as `(applicable, passed)`: positive IC; survives costs (n/a if `cost_na`); positive holdout Sharpe; positive managed Sharpe (n/a if `risk_na`); modest decay; evidence resolved (`cs != NASDAQ_ID`).
- `holdout_decay = (val_sharpe - ho_sharpe) / val_sharpe` when `val_sharpe > 0` and holdout exists; modest if `< 0.50`.
- Tally = passed / applicable; non-applicable stages leave both numerator and denominator. Snapshot (Table 20.11) sorted by gates passed desc, then best IC.
- Structural-feature split: all-gates-pass vs miss-at-least-one; report mean validation IC per side, pass/miss counts by rebalancing frequency and by selected model family.

### Feature triage funnel — NB02

- Input: `case_studies/{cs}/evaluation/triage_ledger.parquet` (regenerated by each CS's `05_evaluation`).
- Ledger columns per feature: IC moment estimates, HAC-adjusted t-stat, BH-adjusted p-value `fdr_p`, fold sign consistency, monotonicity, coverage, decision in {PROCEED, REVISE, STOP}, `note` with the reason.
- Two routes to PROCEED (union):
  1. `fdr_significant`: BH at `FDR_ALPHA = 0.05`, recomputed cross-CS from stored p-values on a common scale.
  2. `stable_and_above_threshold`: |mean IC| above a per-CS effect-size floor of 0.003–0.01 AND sign held across enough folds (which CS uses which floor, and the minimum fold count, are not listed in the notes).
- Show the three counts side by side, never stacked (Table 20.3).
- Forward link: PROCEED rate vs selected config's validation and holdout Sharpe. A handful of points; shown to display the absence of a relationship, not to settle it.

### Signal quality — NB03

- `ic_mean` = mean IC over all non-holdout predictions of a family on the CS primary label (from `setup.yaml`), excluding `causal_dml` (not fit to predict). `ic_best` = family maximum over the same sweep.
- Comparable within a row (fixed label), not across rows (labels range from 15-minute to 21-day forward returns on different instruments).
- Signal-Sharpe heatmap = median Sharpe when a family's predictions become positions; separates ranking quality from average prediction quality.
- Within-family dispersion (best far above median) means configuration choice does the work, and that choice was made on validation.
- Asset-class panel: IC vs universe size, frequency, cost regime; read `best_ic` from `ic_comparison.parquet`. Note: the `models` block of `all_synthesis.json` stores the per-family MAX under the key `ic_mean`.
- §20.3 names an "IC plus stability bundle" (ICIR, positive-fold share, checkpoint sensitivity); its computation is not in the digested notebooks (uncertain).

### IC → Sharpe translation — NB04

- Figure 20.3 two panels: left = holdout IC vs holdout Sharpe with registry-stored intervals; right = validation Sharpe vs holdout Sharpe of the SAME selected configuration (the only place a decay claim can be made).
- IC-leader vs Sharpe-leader agreement counted only where both exist.
- Cadence panel (15-minute vs daily vs monthly) is descriptive only: cadence is confounded with CS.
- Per-CS Sharpe-distribution box plots show the sample each maximum was drawn from (DSR context; DSR not computed).
- Positive-Sharpe rate denominator = variants actually backtested, not registered.
- Fundamental Law mapping (breadth, IC, transfer coefficient) and the top-K sweep (Figure 20.4) parameters are named in §20.4 but not in the digest (uncertain).

### Allocator cross-section — NB05

- Allocators with the signal held fixed: equal weight, inverse-vol, MVO, risk parity, score-weighted, HRP (notes); NB05's actual allocator filter is `equal_weight, inverse_vol, score_weighted, mvo_ledoit_wolf, risk_parity, hrp, conformal_weighted` — MVO is the Ledoit-Wolf variant and conformal-weighted is included. Load `stage: "allocation"` runs restricted to the spine `prediction_hash` only (no Ch19 overlays, which would credit the allocator with overlay work).
- Table 20.6 = max across rebalance and top-K variants within the allocation stage; `extract_top_k(spec_json)` reads `top_k`.
- EW baseline read from the SIGNAL stage on the spine prediction via `is_unallocated` (no `equal_weight` allocator rows exist). Uplift = best allocator − EW on the same spine; taking the best baseline across all predictions would let uplift absorb a change of model or label.
- Uplift breadth (NB05 code, four lowercase labels on the share of non-EW allocators beating EW): `none` if `n_better == 0`; `broad` if `n_better / n_total > 0.5`; `moderate` if `> 0.25`; `narrow` otherwise (share ≤ 0.25). The notes' three-label version ("Narrow if only one or two") predates the split of `narrow` from the zero case.
- Allocator sensitivity (Table 20.7) = best-minus-worst Sharpe spread per CS.
- "When does MVO help" scatter: x = EW baseline Sharpe, y = uplift, bubble = universe size from `overview.parquet`; `uplift_interpretation(mvo_df)` reads the result off the frame.
- Section claim: HRP wins where the signal is broad; the allocator spread widens when the signal is weak.

### Cost survival: breakeven and resilience — NB06

1. Read the release configuration's cost sweep (commission + slippage grid, signal and allocation constant) from Ch18 `cost_sensitivity`-stage rows, only where the sweep sits on the carrier's training lineage AND runs the carrier's strategy. The loader prints which check dropped each absent CS.
2. Net Sharpe at assumed cost: `_sharpe_at(curve, cost_bps)` — linear interpolation on the grid; clamp beyond grid ends (reported by equal bracket); report bracketing points.
3. Breakeven: `_breakeven(curve)` interpolates between the last positive and first non-positive grid point with `w = v_lo / (v_lo - v_hi)`; still positive at ceiling → `(grid[-1], False)` censored; never positive → `(0.0, True)`. This is the net-Sharpe-zero crossing; the single definition lives in ch16 "Break-even cost: one definition" (CAGR-vs-benchmark and Sharpe-zero crossings both reported, per-leg, impact stated, sign change bracketed) and reports here follow it.
4. `cost_margin_bps = breakeven_bps - assumed_cost_bps`; `cost_margin_ratio = breakeven_bps / clip(assumed_cost_bps, lower=1)`. `assumed_cost_bps` is `compute_cost_bps(setup)` (declared per-leg bps, no impact term): the repository convention, narrower than ch16's all-in denominator (commission + half-spread + impact at the reference AUM); name which one a ratio used.
5. Resilience by `cost_margin_ratio`:

| Ratio | Label |
|---|---|
| < 1 | does not survive |
| ≥ 10 | very robust |
| ≥ 3 | robust |
| ≥ 1.5 | marginal |
| otherwise (1 ≤ r < 1.5) | fragile |

6. Assumed cost via `compute_cost_bps(setup)` through `overview.parquet`; charting floor `max(min assumed, 0.5)` bps.
7. Cadence from `setup.yaml` strings (`monthly_month_end`, `daily_ny_close`) mapped by leading word via `CADENCE_PERIODS = ("15min", "hourly", "8_hour", "daily", "weekly", "monthly")`, else `"unspecified"`.
8. `TURNOVER_MULTIPLIER = {"15min": 26.0, "hourly": 6.5, "8_hour": 3.0, "daily": 1.0, "weekly": 1/5, "monthly": 1/21}` — stated assumptions relative to daily, not measured turnover (prose: 15-min ≈ 25x daily, ≈ 500x monthly).

### Options cost realism: HTM cascade — §18.8 / NB06 / NB08

- A bps-of-notional model fails options: the dominant cost is the bid-ask spread on premium, scaling with quote width, not notional. sp500_options has no bps sweep and appears in no cost table.
- Executable-label backtesting at actual bid/ask. One prediction decomposed across three labels: priced at mid unhedged, delta-hedged at mid, priced at executable quotes — separating signal contribution from execution. Ranking on signal AND spread jointly is a different strategy from ranking on signal alone.
- HTM cascade numbers (hardcoded in NB08, reproducing `case_studies/sp500_options/evaluation/htm_cost_sensitivity.parquet`, written by that CS's `15_costs` notebook): max Sharpe = −0.28 at 20% half-spread fraction; −0.47 at 50%; −0.72 at 100% → "Net-negative under realistic costs". The positive-Sharpe variant rate before execution costs is misleading.

### Risk overlays and holdout decay — NB07

- Reads Ch19 `risk_overlay`-stage backtests tagged `chapter: "ch19"`.
- `classify_overlay(name)` maps `risk.name` substrings to categories:

| Substring | Category |
|---|---|
| trailing | `trailing_stop` |
| mae_mfe, calibrated | `mae_mfe_calibrated` |
| stop_loss, sl_ | `stop_loss` |
| take_profit, tp_ | `take_profit` |
| daily, loss_limit, period_loss, bar_loss | `daily_limit` |
| dd_breaker, max_dd | `dd_breaker` |
| vol_target, vol_stop | `vol_target` |
| combined, chain, full | `combined` |
| time_exit | `time_exit` |
| else | `other` |

- Reduce each CS to its single highest-Sharpe overlay ROW (not name). Baseline = allocation-stage strategy, signal-stage fallback; record which baseline was used.
- Deployed overlay = a row of the declared sweep grid, never an off-grid default; print its rank in the sweep (1 = best validation Sharpe) and the population median; if the row sits in the top 5% of the sweep, say so and quote the median as the headline. Freeze date and `precommitted` flag follow the precommitment pattern in `chapters/19_risk_management.md` (`frozen_as_of` in the config; the evaluation window must start after it).
- Outputs: Sharpe delta vs baseline; mean delta per category; category × CS heatmap of max delta (cells with more configs are higher for that reason alone); drawdown-reduction table; quadrant plot (Sharpe delta, drawdown reduction) with four outcomes: both improve / drawdown falls at Sharpe cost (ordinary insurance) / Sharpe rises at drawdown cost / both worsen; panel (b) overlay benefit vs baseline drawdown depth.
- Validation-vs-holdout decay (Figure 20.6) read through the three mechanisms: prediction-quality drift (holdout IC falls), portfolio-translation drift (IC holds, Sharpe falls), structural break / regime change.

### Ensembles — NB08 §7

- Equal-weight blend of the top-3 configurations per CS, computed outside the notebook on per-fold return series (the registry stores only per-fold Sharpe summaries for most CSs). Expected: tighter cross-fold stability at a small cost in peak Sharpe. Not registered this iteration.
- NASDAQ-100 bounded exception: ensemble fixed BEFORE holdout scoring as diversification under overlapping validation uncertainty; the corrected linear holdout is a comparator only and cannot reselect or describe the ensemble as an ex-post rescue.

### Causal credibility — §20.8

- DML estimates (`causal_runs` registry rows) read as a fragility metric with a "publication-standard threshold" and a refutation companion. No notebook in this chapter implements it and method details are not in the digest (uncertain); see `chapters/15_causal_estimation.md`.

### Practitioner workflow — §20.9

Sequence: data + diagnostics → feature triage → signal generation (IC + stability bundle) → strategy construction (allocation, costs, overlays) → frozen holdout → evidence classification → next iteration.

Next-iteration handles (inside the Ch6 iterative workflow):

| Handle | Concrete action |
|---|---|
| Label refinement | horizon, winsorization, classification vs regression — the `model_analysis` label axes |
| Domain features | order flow for NASDAQ-100; carry dynamics for CME; funding structure for crypto |
| Focused tuning | larger budgets only on families surviving the holdout gate |
| Ensembles | simple averages guided by inter-family prediction correlation (variance reduction at constant mean IC) |
| Strategy design | sector constraints, regime conditioning, dynamic position sizing, multi-horizon blending |

## Guardrails and pitfalls

- **Partial full-mode aggregation** — a run seeing some registries is indistinguishable from a complete one; on 2026-08-28 a three-registry worktree stamped itself production and published a holdout Sharpe under nine-CS prose (push gate caught it, notebook did not). Call `refuse_partial_full_mode(expected, subset, unreadable, empty, enforce=True)`, which raises unless every CS has a readable registry with backtests; subset runs (`CASE_STUDIES` non-empty) write no chapter artifacts, only paired metrics into that CS's registry.
- **Comparing validation max-IC with holdout IC** — validation artifact = max over a hyperparameter sweep; holdout = IC of one selected prediction; a max over dozens of draws sits above the typical draw by construction, so every arrow "decays" whether or not anything did ("the figure manufactured the decay it was meant to measure"). Compare like with like: holdout IC vs holdout Sharpe; validation vs holdout Sharpe of the SAME selected configuration.
- **Selection optimism / multiple testing** — selected Sharpe is the maximum of a sample; within-CS variant spread exceeds the spread of medians across CSs; allocator, overlay and cost headroom were all chosen on validation. Show the distribution each max came from; read rank1–rank10 spread vs fold-SE; treat best-of-sweep deltas as inflated by sweep size; DSR would discount for trials but is not computed and no deflation is applied in NB05/NB07.
- **Reporting absence as zero** — four CSs had predictions but no backtests and were plotted at 0% positive; an IC leader compared against an empty string produced four "No" rows. Denominator = variants actually backtested; exclude and NAME unmeasured CSs; heatmap blanks mean "not attempted", never "attempted and failed".
- **Equal-weight allocator rows do not exist** — EW is the baseline and is never listed as an allocator (NB05 cites §4 of a pipeline reference document that the checkout does not ship); on 2026-09-18 all nine registries held zero `equal_weight` rows at `stage='allocation'`, so the old filter returned all-null `ew_sharpe`, `sharpe_diff`, `pct_improvement`. Read the EW baseline from the signal stage on the spine prediction (`is_unallocated`).
- **Typed-in constants drift** — a hardcoded universe-size dict had all nine entries wrong (etfs 64 vs 100, sp500_options 480 vs 627, us_equities_panel 311 vs 3199); an earlier takeaway named leading allocators for six CSs with no allocation backtests. Read every number from `overview.parquet` / registry; compute summaries from the table, never type them.
- **Null spine vs missing key** — a key-existence check let a null spine pass and then drop silently at the filter. Missing key = stale selection file (error); null spine = registry has no backtests (name and exclude).
- **Stale spine pin** — a pin matching no rows would pool cost/risk figures from full-universe rows when the strategy runs on a restricted subset. `build_all_synthesis` raises on a non-matching pin; CSs with empty registries are left unpinned with cost/risk reported not applicable.
- **Overlay name selects many rows** — names like `trailing_3pct` cover many parameterizations, so one name plots several points under one label. Select the single highest-Sharpe ROW per CS.
- **Nondeterministic rung selection** — rungs 1 and 2 of the options cascade share a universe filter; `ORDER BY sharpe DESC LIMIT 1` picks whichever scores higher in current data. Pin `universe_filter` and `exit_at_max_days` together (`RUNG_PINS`); fetch all rows (`top_n=1_000_000`) before applying the predicate.
- **Label mixing in the options CS** — the HTM label with coherent option costs and the four fixed-horizon straddle labels with generic bps costs are not comparable; Table 20.6's options row and the §20.6 cost model depend on the pin. `LABEL_RESTRICTIONS` pins cluster diagnostics and holdout retrain to the HTM label.
- **bps-of-notional cost model on options** — the dominant cost is the bid-ask spread on premium, scaling with quote width; a generic sweep understates it. Use an executable backtest against quotes with the three-label decomposition.
- **Breakeven above the swept grid** — the ceiling is not a breakeven. `_breakeven` returns `observed=False`; mark censored rows rather than printing the ceiling.
- **Cost sweep read at the wrong cost** — gross Sharpe and top-of-grid Sharpe are easy to read and neither decides tradability. Interpolate at the CS's own assumed cost (`setup.yaml`), report bracketing grid points, compare breakeven to assumed cost as a ratio.
- **Cost curve missing for the carrier** — the sweep must sit on the carrier's lineage AND run the carrier's strategy; ETFs carrier has no sweep; sp500_options has 8 rows, all `equal_weight_top_k` + `score_weighted`; NASDAQ-100 has 24 rows, all `equal_weight_top_k` (pass-1 ranking instrument) while its carrier is `slot_persistent_signal_exit`. The loader prints the dropped check per CS; absence is read, not guessed.
- **Uniform proportional per-leg cost** — real costs vary with size, instrument and book state. Use executable quotes where spread dominates (single-name options, intraday equities).
- **Cadence confounded with case study** — each cadence is carried by a different market set, so a cadence difference and a CS difference are the same difference. Treat the cadence table as description; turnover multipliers are stated assumptions.
- **Correlated feature menu under BH-FDR** — overlapping return windows and related vol measures make the effective number of tests smaller than the row count, so BH is conservative. Use the stability route to PROCEED (effect-size floor + fold sign consistency); never read PROCEED as a stricter BH.
- **Comparing PROCEED rates across CSs** — per-CS effect-size floors differ by a factor of three (0.003–0.01), so differences mix market and floor. Compare the BH-FDR column (re-thresholded at one `FDR_ALPHA`) across markets; PROCEED only within a CS.
- **Weak forward-link test** — ~5 CSs reach holdout; five points cannot establish feature-survival → strategy-survival. State the sample size; refuse to read a trend.
- **Polars schema inference on empty registries** — leading all-null `ic` / `sharpe` rows make Polars infer a null column that rejects the first real float. Declare the schema in `build_variant_rows`.
- **Independent vs cumulative funnel confusion** — NB01 per-stage counts can exceed NB08 cumulative counts; two gates can pass the same number while failing different CSs. Use NB08 for strict survivors and named dropouts, NB01 for per-stage attrition.
- **Gate 2 read from the wrong stage** — the risk-stage baseline Sharpe is null for CSs whose risk stage does not apply (sp500_options HTM, us_firm_characteristics vectorized, NASDAQ-100 before its ensemble cost/risk pass) and would drop them. Read the selected config's validation ML Sharpe; non-applicable stages pass through.
- **Holdout CI spanning zero treated as failure** — the window cannot distinguish the strategy from no edge, which differs from finding it wanting. Print the interval beside the point estimate; classify "Statistically unresolved".
- **Fallback baseline for overlays** — a delta against a signal-stage fallback is not comparable with one against an allocation baseline. Record which baseline was used.
- **Max drawdown has no interval** — single realized-path statistic; a few percentage points are indistinguishable. Do not rank overlays on small drawdown differences.
- **Deployed default outside its own sweep** — a default that is not a grid row was chosen somewhere the sweep cannot see, and a rank-1 default is best-of-sweep by construction. Require membership in the declared grid, print the rank, disclose top-5% rows and quote the median as headline (precommitment pattern, ch19).
- **Unpaired reference called paired** — an EW or random-top-k reference run without the strategy's caps, sizing or schedule, or with a tie-break that leans on the conditioning variable, measures protocol differences. Apply the paired reference rule (NB01 section); label the comparison unpaired otherwise.
- **Per-family IC means with unequal sweep counts** — 28 vs 150 configurations give incomparable precision and no interval is attached. Report counts; keep `ic_best` and `ic_mean` on separate axes.
- **Validation IC is in-sample to the selection process** — configurations were chosen by looking at it. The holdout comparison in NB01 is where that selection is priced.
- **Pair #3 window mismatch** — validation and holdout windows are disjoint and of different length; `compute_paired_uncertainty` refuses unequal lengths (returns `{}`, no truncation). NB01 routes the pair through `_populate_pair(..., disjoint_windows=True)` → `compute_independent_diff_uncertainty` (each window resampled on its own block length, draws differenced). Its interval covers the difference between the two windows' own Sharpes, not "is this decay noise?"; `val_rank1_self` always carries that caveat. The notes' "truncate to `min(len)`" recipe is a stale NB01 comment, not what the code does.
- **Ex-post ensemble rescue** — reselecting after seeing the holdout invalidates it. Fix the ensemble before holdout scoring; the corrected holdout is a comparator only.
- **Section cross-reference drift** — eleven references pointed at sections that did not carry the claimed material. Pin comments in the notebooks explain why the §20.5 / §20.6 references are correct; verify a reference carries the material before citing it.
- **Generated artifacts not versioned** — triage ledgers and registries regenerate, so numbers move. Pin nothing to a generation; compute everything at run time.
- **Ranking vs construction disagreement** — holdout Sharpe and holdout IC need not agree; both negative while validation Sharpe is strongly positive = ranking inverted on holdout. Print both columns; the holdout exists to report exactly this.

## Decision rules and defaults

| Decision | Rule / default |
|---|---|
| Skill claim over passive baseline | Requires all three: paired-bootstrap CI for `sharpe_diff` excludes zero; `prob_challenger_wins` close to 1; small `p_value`. CI straddling zero → report "within block-bootstrap sampling error", not failure |
| Paired bootstrap | `n_boot=2000`, `seed=42`; block length from `rebalance_step` (never below label horizon); min series length 6 (monthly), 12 (weekly), 21 (daily or faster) |
| Paired reference | The non-ML reference (EW universe, random top-k, fixed rule) runs under the strategy's exact caps, sizing, schedule, costs and warmup; tie-break stated and orthogonal to the conditioning variables; cannot run under those caps → unpaired, labelled as such |
| Cluster reading | Thick top if rank1 − rank10 spread is small relative to fold-SE; stable if folds-positive ≈ n_folds |
| Feature triage | `FDR_ALPHA = 0.05` (BH); per-CS effect-size floor \|mean IC\| 0.003–0.01 plus fold sign consistency; decisions PROCEED / REVISE / STOP |
| Cost resilience | `cost_margin_ratio = breakeven / max(assumed, 1)`: < 1 does not survive; ≥ 10 very robust; ≥ 3 robust; ≥ 1.5 marginal; else fragile (assumed = `compute_cost_bps(setup)`, no impact; breakeven = net-Sharpe-zero crossing; the all-in denominator and the crossing to quote are defined in ch16 "Break-even cost: one definition") |
| Turnover vs daily | 15min 26x; hourly 6.5x; 8h 3x; weekly 1/5; monthly 1/21 |
| Cumulative gates (NB08), in order | IC > 0 → validation ML Sharpe > 0 → net Sharpe at actual cost > 0 (or cost n/a) → holdout Sharpe > 0 → managed Sharpe > 0 (or risk n/a) and evidence resolved. Drop only on a genuine negative at an applicable stage |
| Evidence profile | holdout decay `(val − ho)/val < 0.50` = modest; `\|worst_drawdown_pct\| > 50` = unacceptable; holdout CI spanning zero = statistically unresolved; gates counted as passed/applicable |
| Allocation uplift breadth | `broad` if > 50% of non-EW allocators beat EW (trustworthy); `moderate` if > 25%; `narrow` if ≤ 25% (possible selection bias); `none` if 0 (NB05 code; notes give a three-level broad / narrow / none scale). Wide best−worst spread → construction is load-bearing and was chosen on validation; tight spread → decide on turnover, capacity, explicability |
| Overlay deployment | Default none; deploy only from the win-win quadrant (Sharpe up AND drawdown down) confirmed out of sample; judge by the population (median configuration), not best-of-sweep; the deployed configuration is a row of the declared grid with its rank printed, top-5% rows disclosed and the median quoted as headline; `frozen_as_of` precedes the evaluation window (ch19 precommitment pattern) |
| Cost model choice | Proportional bps sweep where fees scale with notional; executable quote-based backtest where the bid-ask spread dominates (single-name options, intraday equities) |
| Comparability | IC comparable within a CS row (same label), not across CSs; never put `ic_mean` and `ic_best` on one axis; never put validation max-IC and holdout IC on one axis |
| Full-mode run checklist | All nine registries readable with backtests (else refuse); triage ledgers present (else stop); spine resolved per CS (null → name and exclude); subset run writes paired metrics only |
| Selection | Highest validation Sharpe jointly across signal/allocation/risk; fallback walk down validation ranking until a usable holdout exists |
| Stage attribution | Attribute a stage delta only where the later stage carries the earlier configuration (paired row exists) |
| Practitioner sequencing | data + diagnostics → feature triage → signal generation → strategy construction → frozen holdout → evidence classification → next iteration |

## Code patterns and APIs

Run order and environment:

```bash
uv run python 20_strategy_synthesis/01_aggregate_synthesis.py   # ... through 08_recommendations.py, dependency order
MPLBACKEND=Agg PLOTLY_RENDERER=json uv run python 20_strategy_synthesis/03_signal_quality.py   # headless
uv run pytest tests/test_chapter_notebooks.py -v -k "20_strategy_synthesis"                    # Papermill, reduced data
```

Docker image `ml4t`. Notebooks are Jupytext pairs (`0N_*.py` + `.ipynb`); all read NB01's `output/` unless stated.

Notebook → book tables and figures:

| Notebook | Produces |
|---|---|
| NB01 `01_aggregate_synthesis` | `output/` artifacts (below) + `backtest_paired_metrics` rows in each CS registry |
| NB02 `02_feature_evaluation` | Table 20.3 (cross-CS triage funnel); feature-survival vs strategy-survival figure |
| NB03 `03_signal_quality` | Table 20.4 (family summary); Figure 20.2 (IC vs Sharpe scatter); IC landscape and signal-Sharpe heatmaps; IC vs asset-class panel |
| NB04 `04_signal_to_strategy` | Table 20.5 (family cascade); Figure 20.3 (two-panel IC → Sharpe holdout); Figure 20.4 (top-K sweep); cadence table; per-CS Sharpe box plots; positive-Sharpe rate |
| NB05 `05_portfolio_allocation` | Table 20.6 (allocator winners); Table 20.7 (best−worst spread); uplift breadth; "when does MVO help" scatter |
| NB06 `06_cost_survival` | Table 20.8 (breakeven scorecard); Figure 20.5 (cost waterfall); resilience classification; cadence-regime chart |
| NB07 `07_regime_risk` | Figure 20.6 (validation-vs-holdout decay); Figure 20.7 (overlay impact); quadrant analysis |
| NB08 `08_recommendations` | Figure 20.18 (cumulative funnel waterfall); exclusion taxonomy; Table 20.11 (evidence snapshot); structural features vs gate passage; ensembling note |

Registry access (one `BacktestExplorer` per CS registry; the class lives in the companion repo at `case_studies/utils/backtest_explorer.py`, not in an ml4t library):

```python
explorer.best(stage=stage, top_n=top_n, prediction_hashes=_live_predictions(cs))
explorer.progression(..., universe_filter=rung["universe_filter"],
                     exit_at_max_days=rung["exit_at_max_days"])   # one prediction_hash through stages
from case_studies.utils.paired_metrics import RUNG_PINS            # also populate_paired_metrics (canonical carrier, retired identities)
from case_studies.utils.strategy_analysis import LABEL_RESTRICTIONS  # NB01 imports it here; it is not in paired_metrics
```

NB01 helpers (by role):

| Role | Functions |
|---|---|
| Run-mode safety | `refuse_partial_full_mode(*, expected, subset, unreadable, empty, enforce=True)` |
| Carrier / live rows | `_canonical_carrier(cs)`, `_retired(cs)`, `_live_predictions(cs)`, `_best_live(explorer, cs, stage, top_n)`, `_best_pinned(...)`, `_apply_rung_restriction(df, cs)` → `df.filter(rung["predicate"])`, `_progression_for(explorer, pred_hash, cs)`, `_stage_applicable(cs, stage)` |
| Frame builders | `build_backtest_rows()`, `query_holdout_rows()`, `build_variant_rows()` (declared schema), `write_chapter_artifacts(output_dir, frames, documents, *, subset)`, `build_all_synthesis` (raises on non-matching pin) |
| Paired bootstrap | `_benchmark_returns_from_artifact(cs, label, period="overall")`, `_aligned_returns(cs, h)` → `[timestamp, ret]`, `_min_paired_n(ppy)`, `_populate_pair(cs, challenger_hash, benchmark_hash, benchmark_kind, challenger_returns, benchmark_returns, ppy, label, *, disjoint_windows=False, challenger_overlays_baseline=False, benchmark_label=None)` |
| Lineage | `_full_strategy_spec_from_backtest(db, bt_hash)`, `_val_rank1_carrier(cs)` → `{'spec', 'prediction_hash'}`, `_full_strategy_clauses(spec)` → SQL WHERE + params, `_holdout_lineage_for(cs, leader_label, strategy_spec=None, *, prefer_prediction_hash=None)` → `{backtest_hash, prediction_hash, family, config_name, label}`, `_val_backtest_for_lineage(cs, family, config_name, label)` |
| Schema tolerance | `_optional_metric(db, table, column, alias)` selects the column when present, else a NULL alias |

Other notebooks: NB05 `extract_top_k(spec_json)`, `uplift_interpretation(mvo_df)`; NB06 `_sharpe_at(curve, cost_bps)`, `_breakeven(curve)`, `compute_cost_bps(setup)`, `CADENCE_PERIODS`, `TURNOVER_MULTIPLIER`; NB07 `classify_overlay(name)`, `extract_risk_name(spec_json)` → `spec["strategy"]["risk"]["name"]` (default `"unknown"`); NB08 `classify_exclusions()`, `build_evidence_profile()`, `BUCKETS`, `DISPLAY_NAMES`, `NASDAQ_ID`, `_fmt(v, spec=".2f")`.

Breakeven interpolation (NB06):

```python
for lo, hi, v_lo, v_hi in zip(grid, grid[1:], vals, vals[1:]):
    if v_lo > 0 >= v_hi:
        return lo + (v_lo / (v_lo - v_hi)) * (hi - lo), True
return (grid[-1], False) if vals[-1] > 0 else (0.0, True)
```

Resilience classification (NB06, polars):

```python
pl.when(ratio < 1).then(pl.lit("does not survive"))
  .when(ratio >= 10).then(pl.lit("very robust"))
  .when(ratio >= 3).then(pl.lit("robust"))
  .when(ratio >= 1.5).then(pl.lit("marginal"))
  .otherwise(pl.lit("fragile"))
```

Registry tables and fields:

| Table / key | Fields |
|---|---|
| `prediction_metrics` | `ic_mean` |
| `prediction_sets` | `split='holdout'` — rows written by each CS's `NN_holdout_predictions` + `NN_holdout_backtest` pair, not by Ch20 |
| `backtest_paired_metrics` | `benchmark_kind`, `sharpe_diff`, `ret_diff`, `info_ratio`, `prob_challenger_wins`, `p_value`, `bootstrap_block_length`, `bootstrap_n` |
| `causal_runs` | §20.8 inputs |
| Stages (`STAGE_SEQUENCE`) | `signal`, `allocation`, `risk_overlay`, `cost_sensitivity` |
| Spec JSON | `strategy.risk.name`, `top_k`; Ch19 rows tagged `chapter: "ch19"` |
| Holdout fields | `holdout_sharpe`, `holdout_ic`, `holdout_sharpe_ci_lo`, `holdout_sharpe_ci_hi` |
| `config/setup.yaml` (per CS) | primary label; `labels.{label}.rebalance_step`; cadence strings (`monthly_month_end`, `daily_ny_close`); cost assumption → `cost_bps` |
| `all_synthesis.json` | per CS `pipeline_summary.models[family].ic_mean` (stores the per-family MAX), `.backtest.ml_sharpe`, `.costs.{net_sharpe_at_actual, survives_costs, not_applicable_reason}`, `.risk.{managed_sharpe, worst_drawdown_pct, not_applicable_reason}` |

NB01 `output/` artifacts: `overview.parquet` (asset class, frequency, universe size, cost assumption, primary label, n families), `ic_comparison.parquet` (`ic_mean`, `ic_best`; read by NB03/NB04), `backtest_comparison.parquet`, `sharpe_progression.parquet`, `lineage.parquet`, `holdout_results.parquet` (validation vs holdout Sharpe), `rank1_cluster_diagnostics.parquet`, `measurement_quality.parquet`, `variant_analysis.parquet`, `stage_attrition.json`, `all_synthesis.json` (consumed by NB02–06; an NB01 comment also names a `generate_figures.py` consumer that the checkout does not ship). NB01 also writes `backtest_paired_metrics` rows into each CS registry.

Paths (checked in): `20_strategy_synthesis/0N_*.py`, `case_studies/{cs}/config/setup.yaml`, `case_studies/{cs}/NN_holdout_predictions.py`, `NN_holdout_backtest.py`, `NN_strategy_analysis.py` (NN varies per CS, e.g. etfs 18/19/20, sp500_options 16/17/18), `case_studies/utils/paired_metrics.py`, `case_studies/utils/strategy_analysis.py`, `case_studies/utils/backtest_explorer.py`, `case_studies/sp500_options/15_costs.py`, `tests/test_chapter_notebooks.py`. Run-time outputs a fresh clone lacks until the upstream notebooks run: `20_strategy_synthesis/output/`, `case_studies/{cs}/run_log/registry.db`, `case_studies/{cs}/evaluation/triage_ledger.parquet`, `case_studies/sp500_options/evaluation/htm_cost_sensitivity.parquet`. Cited by notebook comments but not shipped: a pipeline reference document (CASE_STUDY_PIPELINE, §4) and a figure-generation script.

## Evidence from the book

All headline tallies are registry-dependent and computed at run time; the digest deliberately omits typed counts, so the numeric contents of Tables 20.3–20.11 and Figures 20.2–20.7/20.18 are not available here. Structural findings that survive re-runs:

| Finding | Detail | Where |
|---|---|---|
| No model family leads everywhere | Family rankings shift across the pipeline; IC leader and Sharpe leader frequently differ | NB03, NB04 |
| Universe size does not order CSs by IC | Asset-class panel | NB03 |
| Within-CS dispersion dominates | Sharpe dispersion across variants within a CS exceeds the dispersion of medians across CSs | NB04 |
| sp500_options HTM cost cascade | Max Sharpe −0.28 at 20% half-spread fraction; −0.47 at 50%; −0.72 at 100% → net-negative under realistic costs; pre-cost positive-Sharpe rate is misleading | NB08 (hardcoded), `case_studies/sp500_options/evaluation/htm_cost_sensitivity.parquet` (from `15_costs`) |
| Holdout coverage | Five of nine CSs reach the holdout; four CSs have registered predictions but no backtests (mid-rebuild); several registries were empty at writing | NB04 |
| Cost-sweep coverage | Three CSs absent: ETFs (carrier lineage has no sweep), sp500_options (8 rows, not the carrier), NASDAQ-100 (24 `equal_weight_top_k` rows vs carrier `slot_persistent_signal_exit`) | NB06 |
| NASDAQ-100 | Fixed ensemble positive on point estimate, but corrected validation and holdout intervals cross zero; excluded from v3.0 lineage and the evidence gate; cost/risk grids deferred to v3.1 | NB01, NB08 |
| Universe sizes (shipped artifact) | etfs 100; sp500_options 627; us_equities_panel 3199 | `overview.parquet` |
| Sweep sizes | Differ by an order of magnitude across families (e.g. 28 vs 150 configurations) | NB03 |
| Monthly bootstrap | us_firm_characteristics has ~12 holdout observations; the paired bootstrap runs cleanly on n = 12 | NB01 |
| Ensemble of top-3 | Expected tighter cross-fold stability at a small cost in peak Sharpe (prose only; not registered) | NB08 §7 |
| Cost drag by frequency | Higher-frequency strategies show more cost drag (prose claim; magnitudes registry-dependent) | NB06 |
| Allocators | HRP wins where the signal is broad; allocator spread widens when the signal is weak (section claim) | NB05 |
| Overlays | A category whose median configuration loses Sharpe is the normal finding; effectiveness is strategy-specific | NB07 |
| Incidents | 2026-08-28 three-registry run mislabelled as production; 2026-09-18 zero `equal_weight` allocation rows in all nine registries | NB01 guardrails |

## Related references

- `chapters/06_strategy_definition.md` — the iterative research workflow that hosts the next-iteration priorities (§20.9).
- `chapters/07_defining_the_learning_task.md` — §7.3 triage gates and §7.4 FDR control behind the triage ledger (NB02).
- `chapters/11_ml_pipeline.md` — model-family insight notebooks and per-CS `model_analysis` label axes.
- `chapters/12_gradient_boosting.md`, `chapters/13_dl_time_series.md`, `chapters/14_latent_factors.md` — the model families whose IC and Sharpe are compared in NB03/NB04.
- `chapters/15_causal_estimation.md` — DML and `causal_runs` rows read in §20.8.
- `chapters/16_strategy_simulation.md` — IC → Sharpe introduction (repeated here with holdout) and the registry/backtest machinery.
- `chapters/17_portfolio_construction.md` — allocation-stage backtests (EW, inverse-vol, Ledoit-Wolf MVO, risk parity, score-weighted, HRP, conformal-weighted) read in NB05.
- `chapters/18_transaction_costs.md` — cost taxonomy, `cost_sensitivity` sweeps and §18.8 options cost cascade read in NB06.
- `chapters/19_risk_management.md` — risk-overlay machinery (stop-loss, trailing stop, daily loss limit, drawdown breaker, time exit, vol target) read in NB07.
- Per-CS `NN_strategy_analysis` notebooks (via each `case_studies/*.md` reference) — the drill-down for one case study: §2 stage-transition waterfall, §6 holdout decay + holdout-vs-benchmark, §7 benchmark-aware diagnostics.
- `case_studies/sp500_options.md` — HTM label, rung cascade, executable-quote costs, `RUNG_PINS` / `LABEL_RESTRICTIONS`.
- `case_studies/nasdaq100_microstructure.md` — the fixed ensemble, carrier `slot_persistent_signal_exit`, v3.0 exclusions.
- `case_studies/us_firm_characteristics.md` — monthly cadence, vectorized path with overlays purged, n = 12 holdout bootstrap.
- `case_studies/etfs.md`, `case_studies/crypto_perps_funding.md`, `case_studies/sp500_equity_option_analytics.md`, `case_studies/fx_pairs.md`, `case_studies/cme_futures.md`, `case_studies/us_equities_panel.md` — the remaining members of the nine-CS test bed.
- `libraries/ml4t_backtest.md` — the simulation engine behind the registry's backtest rows (it does not document `BacktestExplorer`).
- `libraries/ml4t_diagnostic.md` — IC, fold statistics, FDR and paired-uncertainty tooling.
- `workflow.md` — end-to-end pipeline ordering; `guardrails.md` — cross-cutting leakage / multiple-testing rules; `decision_rules.md` — thresholds collected across chapters; `evidence.md` — empirical findings index; `glossary.md`; `companion_repo.md` — repo layout, `uv run`, Papermill tests, and `BacktestExplorer` (`case_studies/utils/backtest_explorer.py`: `summary / best / progression / champion_lineage / ...`), registry stages and spec JSON.

Further reading:
- Avramov, Cheng & Metzker (2020/2021), Machine Learning vs. Economic Restrictions.
- Bailey & López de Prado (2014), The Deflated Sharpe Ratio.
- Kelly & Xiu (2023), Financial Machine Learning.
- Harvey & Liu (2019), A Census of the Factor Zoo; Harvey, Liu & Zhu (2016), ...and the Cross-Section of Expected Returns.
- Chernozhukov et al. (2018), Double/Debiased Machine Learning.
- Frazzini, Israel & Moskowitz (2018), Trading Costs; Novy-Marx & Velikov (2016), A Taxonomy of Anomalies and Their Trading Costs.
- Grinold & Kahn (2000), Active Portfolio Management (Fundamental Law).
- Gu, Kelly & Xiu (2020), Empirical Asset Pricing via Machine Learning; Freyberger et al. (2020), Dissecting Characteristics Nonparametrically.
- López de Prado (2016), Building Diversified Portfolios that Outperform Out-of-Sample (HRP).
- McLean & Pontiff (2016), Does Academic Research Destroy Stock Return Predictability?
- O'Donovan & Yu (2025), Transaction Costs and Cost Mitigation in Option Investment Strategies.

## Glossary

- **Spine / carrier prediction** — the selected validation configuration's `prediction_hash` to which cost and risk stages are pinned.
- **Full strategy specification** — signal method + allocation method + risk overlay.
- **Rank-1 cluster** — configurations within a fold-SE of the top validation Sharpe; thickness measures selection stability.
- **Fold-SE** — `std(ddof=1)` of per-fold Sharpe divided by `sqrt(n_folds)`.
- **Rung** — a level of the O'Donovan–Yu option cost-mitigation cascade (naive round trip; HTM; HTM on the liquid bottom-spread quintile).
- **HTM** — hold-to-maturity option label/strategy with coherent option costs.
- **Paired stationary block bootstrap** — block resampling of the daily return difference between challenger and benchmark to get a CI on the Sharpe difference.
- **`benchmark_kind`** — label of a pair type (`equal_weight_holdout_side_artifact`, `val_rank1_self`, `signal_leader`, `allocation_leader`, `risk_overlay_leader`).
- **Stage attrition funnel** — count of CSs passing each pipeline gate; independent (NB01) or cumulative (NB08).
- **Holdout decay** — (validation Sharpe − holdout Sharpe) / validation Sharpe; modest if < 0.50.
- **Breakeven cost** — here, per-leg bps at which the cost-sweep net Sharpe crosses zero (`_breakeven`); censored if still positive at the grid ceiling. Single definition, with the CAGR-vs-benchmark crossing, in ch16 "Break-even cost: one definition".
- **Cost margin ratio** — breakeven / assumed cost (headroom); drives the resilience label. Repo denominator `compute_cost_bps(setup)` (no impact); the all-in denominator is defined in ch16.
- **Uplift breadth** — share of non-EW allocators beating equal weight; NB05 labels `broad` (> 0.5) / `moderate` (> 0.25) / `narrow` (≤ 0.25) / `none` (0).
- **Exclusion taxonomy** — data-assigned failure types in three buckets: signal invalidity, implementation infeasibility, evidence-quality failure.
- **PROCEED / REVISE / STOP** — triage ledger decisions; PROCEED via `fdr_significant` or `stable_and_above_threshold`.
- **`ic_mean` vs `ic_best`** — family average IC over the sweep vs family maximum.
- **DSR** — Deflated Sharpe Ratio; discounts a maximum Sharpe for the number of trials behind it (not computed in Ch20).
- **Win-win quadrant** — overlay improves both Sharpe and max drawdown; the only quadrant that justifies deployment.
- **Release / deployed configuration** — the specification declared across signal, allocation and risk-overlay stages that the chapter reports.
- **Three decay mechanisms** — prediction-quality drift, portfolio-translation drift, structural break / regime change.
- **Statistically unresolved** — holdout Sharpe CI spans zero; the window cannot distinguish the strategy from no edge.
