# Reconciling six AI replications of SSRN 7200998

*"Do Market Regimes Improve Machine-Learning Stock Ranking?" (Pascual Miralles & Alfeus, 2026).* Six independent AI-driven replications were run on public data because the paper's Bloomberg inputs are not redistributable. This note lines them up against the paper and against each other, explains why they differ, and states what can be concluded.

Sources: Grok message (Slack, 2026-09-25 16:25), ChatGPT screenshot + `chatGpt.zip` (18:03, 18:12), Kimi table (18:16), Perplexity screenshot + `REPLICATION_REPORT.md` (18:31, 18:42), Claude Code artifact (21:58), "claude science" `replication_report-2.md` (22:01).

## 1. Headline numbers, 2021–2025 out-of-sample, monthly equal-weight Top10

| Run | Data for fundamentals | Stocks / features / train rows | CAGR | Sharpe | Max DD | SPY alpha | Mean IC (p) | Top−SPY p | Top−Random p | Top−Bottom p |
|---|---|---|---|---|---|---|---|---|---|---|
| **Paper (Bloomberg)** | Bloomberg | 206 / 68 / 654,009 | **34.24%** | **1.279** | −12.27% | 13.55% | 0.0424 (0.046) | 0.0050 | 0.0112 | 0.0407 |
| Kimi | not stated | not stated | 19.30% | 0.767 | −16.19% | 1.45% | — | 0.318 | — | — |
| Kimi, "full-history variant" | not stated | not stated | 28.43% | 1.161 | −12.26% | 9.37% | — | 0.011 | — | — |
| ChatGPT | **none** (fundamentals omitted) | 205 / 64 / 619,620 | 21.53% | 0.933 | −13.65% | 6.84% | 0.0163 (0.490) | 0.243 | 0.254 | 0.678 |
| Perplexity | SEC XBRL, +90-day availability lag, backfilled pre-2009 | 206 / 69 / 622,891 | 22.38% | 1.052 | −13.41% | 7.54% | 0.0203 (0.317) | 0.124 | 0.128 | 0.255 |
| Grok | yfinance (sparse) | 205 (no EA) / — / — | 23.66% | 0.952 | −15.4% | 7.08% | 0.012 (n.s.) | ≈0.15 | — | — |
| "claude science" | SEC XBRL companyfacts, filing-date PIT, backfilled pre-2009 | 205 / 69 / 662,574 | 24.90% | 1.080 | −12.43% | 8.37% | 0.0315 (0.132) | 0.061 | 0.073 | 0.278 |
| Claude Code (this session) | SEC XBRL + stockpup (2004–08) + SEC FSDS (2024–25), filing-date PIT | 206 / 69 / 653,098 | 25.00% | 1.071 | −15.53% | 8.89% | 0.0311 (0.093) | 0.080 | 0.067 | 0.151 |

SPY reproduces at 14.71% CAGR / 0.746 Sharpe / −20.47% drawdown in every run that reports it (paper 14.72% / 0.746 / −20.47%), so the calendar, holding-period arithmetic and benchmark are right everywhere. The whole spread between runs is in the stock signal.

## 2. What every run agrees on

1. **The Top10 portfolio beats SPY in the point estimate, by less than the paper says.** Six of six runs: 19–25% CAGR against 14.7% for SPY, versus the paper's 34.2%. Excess return over SPY is 5–10%/yr, not 17%.
2. **The paper's 5%-level significance does not survive.** With the paper's exact specification, Top−SPY p-values are 0.06–0.32 (paper 0.005); IC p-values 0.09–0.49 (paper 0.046). Only Kimi's non-standard "full-history variant" (p = 0.011) and some *alternative* learners (LightGBM, Random Forest, Extra Trees in the Claude, claude-science and ChatGPT runs) clear 5% against SPY, and none of the corrected multiple-testing families in the two reports that ran them (claude science, ChatGPT) keeps a significant test.
3. **Regimes add nothing.** All four runs that tested it (ChatGPT, Perplexity, claude science, Claude Code; Grok did not run the forecasting variants): HMM regime as a feature changes the annualized Top10 return by −0.8 to +4.4 points depending on how pre-2008 rows are handled, with no paired difference significant at 5% (p 0.055–0.97; paper −2.9 points, p 0.17); one model per regime is flat-to-worse (12–27% CAGR vs 22–25% baselines) and always worse on IC. Same conclusion as the paper.
4. **No optimizer beats 1/N inside the Top10; minimum variance is the worst.** All five runs that tested it (Grok, ChatGPT, Perplexity, claude science, Claude Code). Static min-variance paired difference vs equal weight: −7.1%/yr (p 0.036, claude science), −7.5%/yr (p 0.028, Claude Code); paper −8.7% (p 0.018).
5. **Costs erode the edge at ~1.1 CAGR points per 5 bps** (paper 1.2), with 144–169% monthly turnover (paper 148%). At 25 bps the strategy still beats SPY on return but not significantly; at 50 bps it does not beat SPY.
6. **Size dominates.** Log market cap is the top SHAP feature in every run that reports SHAP (paper, Claude Code, claude science), followed by valuation, trend and profitability variables.

## 3. Why the runs differ from each other

The ordering of the runs tracks the quality of the fundamentals block, which is the block the model leans on most:

| Fundamentals input | Runs | CAGR |
|---|---|---|
| None | ChatGPT | 21.5% |
| Sparse or lagged, backfilled | Grok, Perplexity | 22.4–23.7% |
| Point-in-time SEC filings, full coverage | claude science, Claude Code | 24.9–25.0% |

The two most careful point-in-time builds (claude science and Claude Code), done independently with different code paths and data sources, land within 0.1 CAGR point, 0.01 Sharpe, 0.5 points of alpha and 0.0004 of IC of each other. That convergence is the best available estimate of what the published strategy does on public data: **CAGR ≈ 25%, Sharpe ≈ 1.07, SPY alpha ≈ 8.5%/yr, mean IC ≈ 0.031, Top−SPY p ≈ 0.06–0.08.**

Two runs need qualification:

- **ChatGPT omitted the five fundamental features entirely** (64 features). Its 21.5% is a lower bound for the technical/cross-asset signal alone, and its alternative-model ordering (Extra Trees 29.9% > Random Forest 28.1% > LightGBM 23.3% > XGBoost 21.5%) is not comparable with the other runs.
- **Kimi's "full-history variant" (28.4%, p 0.011)** is the only run above 25% and the only standard-spec run that is significant. The message does not say what "full-history" changed (training window, data vintage, or universe), and the 988 MB result archive could not be inspected here. Until that is clarified it should be treated as a specification change, not a replication.

Smaller inconsistencies, all within noise:

- The short side is unstable. Bottom20 CAGR is 13.4% (Claude Code), 14.0% (Perplexity), 16.3% (claude science) and 17.9% (ChatGPT) against the paper's 13.6%; this is why Top−Bottom is never significant.
- HMM regime-month counts of 19/16/13/12 in the ChatGPT and Perplexity runs match the paper exactly, while the Claude runs do not. This is not meaningful: the released `expanding_hmm_regimes.py` re-initialises hmmlearn at every refit, so the numeric state IDs are not comparable across time (see §4).
- The HMM-feature sign flips between runs (−0.8 to +4.4 points). claude science's control shows why: most of the apparent gain in "drop pre-2008 rows" variants comes from discarding back-filled 2005–2007 fundamentals, not from the regime dummies (its 2008–2018 control without dummies reaches 29.2%).

## 4. Cross-claims verified on the Claude Code data

Three claims made by other runs were re-tested on this session's independently built panel:

| Claim (source) | Result here | Verdict |
|---|---|---|
| A size-only rule (buy the 10 smallest names by market cap) matches the XGBoost ranker — 25.8% CAGR, Sharpe 1.07 (Perplexity) | 10 smallest: **25.26% CAGR, Sharpe 1.03**; ML Top10: 25.00%, 1.07; equal-weight universe 15.8%, SPY 14.7%. ML picks sit at the 31st size percentile on average and overlap the smallest-10 by only 2 of 10 names. | **Confirmed.** The ranker is not literally a small-cap screen, but a naive size tilt inside this survivorship-biased large-cap list earns the same return. |
| The released expanding-window HMM label-switches: raw-state agreement with the full-sample HMM is 19.0% over 2021–25, not the paper's 65.6% (claude science; ChatGPT flags the same mechanism) | Raw agreement **18.2%**; best possible fixed relabelling 36.3%. | **Confirmed.** Table 12 (performance by regime) is not comparable across runs or with the paper, and the paper's 65.6% figure is not reproducible from the released code. |
| Ensemble seed choice alone moves Top−SPY significance across the 5% line: five 7-seed sets gave Sharpe 0.95–1.18 and p 0.03–0.17, one of five significant (Perplexity) | Four 7-seed sets on the same panel (paper's seeds plus three alternatives): **CAGR 23.4–28.9%, Sharpe 0.97–1.20, Top−SPY p 0.029–0.149**, one of four below 0.05 (the paper's own seeds give 25.0%, p 0.080). | **Confirmed.** Nothing but the random seeds separates a "significant" 28.9% from a "non-significant" 23.4% on identical data. |

## 5. Why all six differ from the paper

1. **Data vendor, concentrated in the features that matter most.** Bloomberg's daily market cap, P/E, P/B, ROE and D/E are replaced by SEC-derived proxies (or omitted). These are the top SHAP features in both the paper and the replications. claude science's control trained on 2008–2018 only (the years with genuine XBRL data) recovers 29.2% CAGR with p = 0.007; ChatGPT's pooled 2008+ model rises from 21.5% to 26.2%; the validation window 2019–20 shows IC 0.089 (p 0.008) in the claude-science run. The signal is strongest where the fundamentals are cleanest.
2. **Survivorship and a size tilt.** The universe is today's 206 large caps. Within it, "smaller now, larger later" is a mechanical winner, and a size-only rule matches the model.
3. **60 monthly observations and seed noise.** The paper's own robustness tables put the CAGR at 23.0% (walk-forward), 23.4% (per-regime models) and 31.3% (no cross-asset features); its 34.2% headline is the top of a wide range, and Perplexity's seed experiment shows the 5% threshold is within seed noise.
4. **Small code issues** flagged by the replications: the released alpha/beta formula mixes sample covariance with population variance (inflating beta by n/(n−1); ChatGPT); the released feature lists enumerate 69 names while the paper reports 68 (all runs); the 60-day-horizon "CAGR" compounds overlapping windows (claude science, ChatGPT, Perplexity).

## 6. Bottom line

- **Replicates:** every qualitative claim — a tree-based ranker beats SPY and random portfolios, linear models do not, cross-asset features help, regimes and optimizers do not improve equal-weight Top10, costs matter at ~1.1 points per 5 bps.
- **Does not replicate:** the magnitude (34% → ~25% CAGR, Sharpe 1.28 → ~1.07, alpha 13.6% → ~8.5%) and the 5%-level significance of the headline tests.
- **Consensus reading across the six runs:** a real but modest ranking signal (IC ≈ 0.03) whose return is largely explained by a size tilt in a survivor universe, with Sharpe ≈ 1.0 after 5 bps of costs and p ≈ 0.06–0.25 against SPY on 60 months. That is a signal to build on, not a proven edge — which is the position already taken in the Slack thread.
