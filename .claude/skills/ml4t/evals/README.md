# ml4t skill: validation results

Five test prompts (`evals.json`) were run twice each on 2026-10-02 in isolated git worktrees of this
repository: once by an agent told to read `SKILL.md` and follow its routing (`with_skill`), once by
an agent told to work from its own knowledge and not to open the skill (`without_skill`). A separate
grader agent scored every run against the prompt's expectations from the files produced, running
any Python deliverable. A blind judge then compared the two attempts per prompt without knowing
which one had the skill.

## Assertion pass rates

| Eval | Prompt | With skill | Without skill |
|---|---|---|---|
| 1 | Audit `reproduction/code/baseline.py` for leakage and protocol weaknesses | 6/6 | 6/6 |
| 2 | Purged, embargoed walk-forward folds on a monthly calendar plus `PROTOCOL.md` | 5/5 | 5/5 |
| 3 | Cost and capacity analysis (spread + impact, breakeven, AUM at half alpha) | 5/5 | 5/5 |
| 4 | Pre-registered Polymarket research setup (hypothesis, setup contract, labels, PIT data, protocol, baselines) | 6/6 | 6/6 |
| 5 | Volatility-targeting overlay and drawdown kill switch with a lookahead self-test, plus `RISK.md` | 6/6 | 6/6 |

Both configurations pass every assertion. The assertions are presence checks (does the deliverable
flag purge/embargo, name a correction, run its self-test), and a frontier model clears them without
help, so this table says the skill does no harm and the evals are too easy to show its value. The
with-skill runs took about 19 percent longer and used about twice the tokens, which is the cost of
reading the routed reference files.

## What the skill changed (blind head-to-head)

Each prompt's two outputs were given to a blind judge as "A" and "B" (order alternated), who scored
decision-time admissibility, evaluation protocol, costs and feasibility, reporting and falsifiability,
implementability and errors with file-and-line evidence, then named a winner.

| Eval | Winner | Margin | Dimension calls (admissibility, protocol, costs, reporting, implementability, errors) |
|---|---|---|---|
| 1 audit-baseline | with skill | clear | decision-time=without, evaluation=with, costs=with, reporting=with, implementability=with, errors=with |
| 2 walk-forward-protocol | with skill | clear | decision-time=with, evaluation=with, costs=with, reporting=with, implementability=tie |
| 3 cost-capacity | with skill | clear | decision-time=with, evaluation=with, costs=with, reporting=with, implementability=tie |
| 4 polymarket-setup | with skill | clear | decision-time=with, evaluation=with, costs=with, reporting=with, implementability=with |
| 5 risk-overlay | with skill | slight | decision-time=without, evaluation=with, costs=without, reporting=with, implementability=without, errors=without |

The skill won all five. What it changed, according to the judges, was procedural: the with-skill
agents computed what the baselines only named (Mertens Sharpe standard errors, HAC and bootstrap
p-values, a Deflated Sharpe with a trial count derived from the repository's own tables, breakeven
cost and participation-ceiling capacity), sealed the holdout on the label end date, set the embargo to
zero in forward layouts, disclosed prior holdout exposure, defaulted to next-bar fills, shipped the
setup as a machine-readable file with a pre-registration hash, pre-registered thresholds and
disconfirming outcomes, and reported sweeps as populations rather than best-of-grid.

The baselines won individual dimensions where the skill was silent or wrong, and those points fed the
second fix round (see "What was fixed as a result"): the embargo rule was stated as "feature lookback"
only, the holdout boundary was equated with `holdout_start - horizon` (false for variable horizons),
breakeven was defined as a CAGR crossing in one chapter and a Sharpe crossing in another, the impact
coefficient had no stress rule, an undated venue claim about Polymarket was copied verbatim, and the
skill said nothing about the booking lag of a multi-session return, bounds on time out of market, VaR
backtest consequences, or how to cite for readers who do not have the skill.

### Strict re-grade

The judges proposed stricter, file-checkable assertions per eval (stored as `hard_expectations` in
`evals.json`). Re-grading the same outputs against them separates the two configurations:

| Eval | With skill | Without skill |
|---|---|---|
| 1 audit-baseline | 1/6 | 3/6 |
| 2 walk-forward-protocol | 4/5 | 0/5 |
| 3 cost-capacity | 4/6 | 1/6 |
| 4 polymarket-setup | 3/8 | 1/8 |
| 5 risk-overlay | 1/6 | 0/6 |
| **Total** | **13/31 (41%)** | **5/31 (16%)** |

The one strict loss (eval 1) is instructive: its strict assertions were written from the facts the
baseline audit caught and the with-skill audit missed (a weekly-published commodity series stamped
daily, a duplicated share class, the random-top-10 baseline being the equal-weight universe, drawdown
on window endpoints as a lower bound). Those four points are now in the references. The with-skill
audit's strict pass was the one the baseline could not match: a shipped script that regenerates
every quoted statistic.

### What was fixed as a result

The second fix round added or corrected, in the reference files: the two-direction purge and
embargo rule for layouts where training can follow test; settlement-date holdout assignment with a
holdout scoring date rule; a config round-trip check and a rebalance-gap/idle-session recipe for the
baseline checkpoint; one breakeven definition (CAGR crossing and Sharpe crossing both reported, one
quoted) and one cost-margin denominator; an impact-coefficient stress rule (eta >= 0.5 row, net Sharpe
at eta = 1.0, participation-ceiling capacity); booking-lag, re-entry-bound, cadence-parity and
precommitment patterns for overlays and kill switches; Christoffersen and Basel traffic-light rules
for VaR backtests; dated venue facts and a Polymarket mechanics checklist; variable-horizon label
rules; the non-vacuous disconfirmation check; and a citation rule in SKILL.md. The eval outputs
above were produced before these fixes, so the table measures the skill as it was on the first pass.

## Reproducing

The runs were orchestrated with the Workflow scripts kept outside the repository; to repeat by hand,
run each prompt in a fresh worktree with and without the sentence "read `.claude/skills/ml4t/SKILL.md`
and follow it", copy the deliverables, and grade them against `expectations` in `evals.json`.
