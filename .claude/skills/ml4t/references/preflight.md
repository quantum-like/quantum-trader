# Pre-flight checklist: 35 questions before trusting a research result

Extracted from the ML4T guardrail catalog so it can be loaded on its own. Answer every question before trusting or reporting a research result; a "no" on any line means the number is not yet evidence. Questions are ordered by workflow stage (`workflow.md`). Tags name the reference that explains the item: `chNN` = `chapters/NN_*.md`; case-study and library tags follow the legend in `guardrails.md`, and the full "what / why / how to detect or prevent" entries sit in the three catalog parts that `guardrails.md` indexes (§1–5 data to features, §6–9 splits to backtest, §10–16 portfolio to LLM systems).

**Data**
1. Have you profiled coverage per symbol and per period (entries/exits per year, all-missing periods, phantom sessions) instead of counting nulls inside rows? (ch02, ch07)
2. Are returns computed from total-return-adjusted prices, with the adjustment convention named per column, raw prices reserved for screens and fills, and `price_adjustment_factor == 1.0` on the last row? (ch02, lib_data, cs_usequities, cs_cme)
3. Have you run corporate-action, ticker-reuse and `ret > 1.0 / prev_volume == 0` checks before any survivorship or portfolio statistic? (ch02, ch07)
4. Is every timestamp tz-aware (UTC-normalised), aggregated on the exchange session rather than the UTC/calendar day, and is the observation grid measured from timestamp spacing rather than read from config? (ch03, ch16, cs_fx, cs_nasdaq)
5. Are bar-open stamps advanced to bar close before any as-of join, and are timestamp units identical on both sides of every join? (ch02, ch03, cs_crypto)

**Point-in-time**
6. Are fundamentals, macro, filings, COT and 13F keyed on availability (announcement, acceptance, release, `available_from`) with vintages where revisions exist, never on period end or report date? (ch01, ch02, ch04, ch22, ch23, lib_data)
7. Is the universe point-in-time (eligibility from trailing data, effective the following period) and is residual list survivorship declared rather than claimed away? (ch08, ch14, cs_etfs, cs_usequities)
8. Are entities keyed on permanent identifiers (CIK, CUSIP, `sec_id`, LEI), not tickers or free-text names? (ch04, ch22, ch23, cs_sp500eo)

**Labels**
9. Is the label anchored where the fill happens (next open / next-bar VWAP), not at the same close that generated the signal? (ch07, ch16, ch19, ch26, cs_nasdaq)
10. Have you reported `effective_sample_size` (N_eff ≈ N/h; a case-study helper in `case_studies/utils/label_diagnostics.py` and `07_defining_the_learning_task/06_ic_inference.py`, not an ml4t-engineer function) and overlap-aware standard errors for every overlapping label, with ml4t-engineer's uniqueness weights (`calculate_uniqueness=True`) / `sequential_bootstrap` where labels overlap? (ch07, ch11, cs_etfs, cs_cme, lib_engineer)
11. Were labels built on the complete sorted series before any eligibility filter, with every unlabelled row reconciled to one cause and the counts summing to frame height? (cs_etfs, cs_usequities, cs_cme, cs_sp500opt)

**Features**
12. Is every cross-sectional statistic computed `.over("timestamp")`, every rolling statistic `.over(symbol)` and trailing-only, with no centred window, full-sample z-score, rank or winsor bound anywhere? (ch02, ch07, ch08, ch11, cs_usequities)
13. Are flow, volume, volatility and premium predictors lagged one bar so no same-bar quantity enters a prediction? (ch03, ch08, ch21, cs_nasdaq)
14. Are model-based features (HMM, GARCH, Kalman, ARIMA, PCA, FFD) produced by a declared burn-in + refit schedule, filtered not smoothed, recursion inputs pinned to the training window, and did a holdout-withheld rebuild / truncation test pass with exact agreement? (ch09, cs_etfs, cs_fx, cs_sp500opt, cs_usequities, lib_diagnostic)
15. Are text and news signals lagged to the next session after acceptance/publication, deduplicated, and scored by a checkpoint whose training cutoff precedes the evaluation window? (ch10, ch13, ch22)

**Splits and protocol**
16. Does every split purge `label_horizon` sessions counted on the market calendar (not calendar days) and embargo at least the feature lookback wherever training can follow test? (ch06, ch11, ch12, lib_diagnostic, cs_etfs)
17. Are splits made by date rather than row position, and is the holdout boundary set by label END date (`holdout_start − horizon` on each instrument's sessions)? (ch12, ch13, cs_fx, cs_sp500eo, cs_usequities)
18. Were scalers, imputers, encoders, winsor bounds and percentile thresholds fit on training rows per fold only? (ch05, ch07, ch11, ch13, lib_engineer)
19. Is the holdout sealed: boundaries fixed at the start, opened once for confirmation, with no selection, ensemble rescue, threshold refit or second configuration after seeing it? (ch06, ch14, ch17, ch20, cs_fx, cs_nasdaq)

**Evaluation**
20. IC inference — (a) is IC computed per date then averaged, never pooled across stock-days; (b) is the t-stat HAC from `compute_ic_hac_stats(..., label_horizon=h)` on the chronologically sorted series (the library sets lag = max(h − 1, Newey-West auto); omitting `label_horizon` uses the T-based bandwidth alone); (c) is a minimum cross-section enforced per date; (d) are NaN and null both dropped; (e) is `ic_n_days` coverage printed beside the mean? (ch07, ch08, ch11, ch13, cs_nasdaq, cs_usequities)
21. Have you included a no-parameter baseline (zero or persistence forecast, majority class, random-signal plumbing backtest) and read coverage before any ranking? (ch10, ch11, ch13, cs_nasdaq, cs_usfirm)

**Multiple testing**
22. Was the searched set (features × horizons × configs × checkpoints × portfolio sizes) declared before results, logged, and corrected with BH-FDR on HAC p-values and DSR/RAS over all variants tried? (ch07, ch08, ch16, cs_sp500opt, lib_diagnostic)
23. Is every checkpoint registered as its own candidate rather than the best epoch picked after seeing validation? (cs_fx, cs_sp500eo, cs_usequities, cs_usfirm)

**Modeling and selection**
24. Is selection made on validation backtest Sharpe after costs (never IC), only among full-coverage candidates on common support, with exactly one field varied per sibling? (ch26, cs_fx, cs_nasdaq, cs_usequities, cs_sp500opt)
25. Are conformal or other uncertainty outputs checked for empirical, regime-stratified coverage before any position is sized on them? (ch11, ch12, ch13, ch17)

**Backtest**
26. Backtest protocol — (a) `NEXT_BAR` execution (decide close t, fill open t+1); (b) every execution field set explicitly and confirmed to move something; (c) a benchmark run through the same engine, dates, fees and capital; (d) a random-signal plumbing test passed; (e) closed trades only for round-trip statistics? (ch16, cs_nasdaq, cs_usfirm, lib_backtest)
27. Is every Sharpe reported with sample length, PSR/MinTRL and DSR deflated by the full declared trial count? (ch07, ch16, cs_sp500opt, lib_diagnostic)

**Portfolio and costs**
28. Are covariance conditioning, effective positions and solver status validated, and allocators compared against equal weight on common support under a cost sweep? (ch17, cs_sp500eo, cs_usequities)
29. Are costs in commensurable units, turnover one-way, borrow/financing charged as time-held rates, impact coefficients labelled as assumptions, and the breakeven reported against the assumed cost rather than the top of the grid? (ch18, ch20, cs_etfs)

**Risk, live, MLOps**
30. Risk — (a) are VaR models exception-backtested (Kupiec at the stated level, clustering inspected); (b) are stops evaluated stop-first on the prior bar's water mark with entry-time ATR; (c) are sizing inputs (volatility, conviction) lagged one session; (d) are drift thresholds calibrated on a period with known drift and joined to realised performance? (ch19, ch26, lib_backtest)
31. Live — (a) one `Strategy` class for research and live; (b) field-by-field parity gates in CI with the gate count asserted; (c) a kill switch persisted across restarts; (d) startup reconciliation that refuses to launch until clean; (e) promotion gates fixed before results; (f) the broker mode asserted explicitly (`assert_paper_trading()` or `assert_live_trading()`) with host, port and account printed? (ch25, ch26, lib_live)
32. Are all numbers read from the registry or artifacts at run time (never typed into prose), populations declared before fitting, and partial or preview runs refused from canonical outputs? (ch20, ch26, cs_etfs, cs_usequities)

**Synthetic, RL, LLM**
33. For synthetic data: temporal split, discriminative and stylized-fact checks beside moments, and `N_PATHS` populations rather than one path? (ch05)
34. For RL: reward paid as `p_{t-1} r_t`, features lagged, episode-level splits, paired SEs on shared seeds, and formation/evaluation separation? (ch21)
35. For RAG/agents: cutoff dates enforced after retrieval, market price withheld from prompts, citations validated against retrieved ids, tool policy enforced outside the prompt, and retrieval rankings judged with human labels? (ch22, ch24)

## Related references

- `guardrails.md` — index and tag legend for the catalog: `guardrails_data_features.md` (§1–5), `guardrails_evaluation_backtest.md` (§6–9), `guardrails_portfolio_live.md` (§10–16).
- `workflow.md` — the stages these questions follow, each with its go/no-go gate.
- `decision_rules.md` — the thresholds behind the numeric questions (purge and embargo, HAC lag, FDR alpha, cost-resilience ladder, promotion gate) and the kill-criteria sheet (§14c).
- `evidence.md` — what the nine case studies measured for each question (naive vs HAC t, FDR survivors, breakeven ratios, holdout decay).
