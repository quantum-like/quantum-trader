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

See the section below, filled from the judges' reports.

HEAD_TO_HEAD_PLACEHOLDER

## Reproducing

The runs were orchestrated with the Workflow scripts kept outside the repository; to repeat by hand,
run each prompt in a fresh worktree with and without the sentence "read `.claude/skills/ml4t/SKILL.md`
and follow it", copy the deliverables, and grade them against `expectations` in `evals.json`.
