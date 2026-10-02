# Precommitted risk controls

> Cross-cutting protocol for the adaptive controls and kill switches of `chapters/19_risk_management.md` (19.7-19.8) and the overlay comparison of `chapters/20_strategy_synthesis.md` (`07_regime_risk`): how far to lag an overlay input, how a kill rule is allowed to re-enter, at what cadence a rule is backtested, how a sweep pick is disclosed, and which tests and consequences a VaR/ES number carries. Everything here is the mechanical form of positions the book states ("every conditional or adaptive input must be lagged"; "calibrate, freeze, evaluate once"; "a limit class enforces a threshold but never decides one"), not a notebook result; the test statistics in §5 beyond Kupiec are standard practice outside the companion repo and the `ml4t-*` libraries and are marked as such.

## 1. Overlay inputs: the lag rule

"Lag every input one session" is right when the input is a completed daily quantity (a close-to-close return through the prior session, a rolling volatility whose window ends t-1). It is wrong when the per-decision frame holds the strategy's own period return and the decision cadence is coarser than a session: the row for decision t-1 then carries a multi-session forward return (a 21-session holding-period return; an open-anchored label that settles at the next fill), and that return is not booked when the orders for decision t are placed. The rule:

- `lag = 1 + (1 if the prior row's quantity is a multi-session forward return that resolves at or after the next decision snapshot else 0)`, counted in decision periods, not sessions. Daily rolling vol of daily returns read at a monthly cadence: lag 1. A drawdown or realized-PnL kill rule reading the strategy's own monthly returns: lag 2, unless the return is close-anchored and resolves at the very close the decision reads (then 1, and say why).
- Make it mechanical: carry `resolves_at` per row (the snapshot at which that row's return is fully observable: the close of the last session for a close-anchored label, the open of the fill session for an open-anchored one, `chapters/07_defining_the_learning_task.md` Recipe 4) and take the smallest lag whose shifted `resolves_at` never passes the decision snapshot.

```python
def admissible_lag(decision_at: pd.Series, resolves_at: pd.Series, max_lag: int = 3) -> int:
    for k in range(1, max_lag + 1):                                   # decision periods, not sessions
        if (resolves_at.shift(k) <= decision_at).iloc[k:].all():
            return k
    raise ValueError("period return resolves more than max_lag decisions late; redefine the input")

LAG = admissible_lag(frame["decision_at"], frame["resolves_at"])      # 1 for completed daily bars; 2 for an open-anchored period return
realized = frame["period_ret"].rolling(VOL_LOOKBACK).std() * np.sqrt(PERIODS_PER_YEAR)
scale = (VOL_TARGET_ANN / realized).shift(LAG).clip(0.0, MAX_LEVERAGE)
```

State the lag in the term sheet next to the lookback; the self-test asserts `(frame["resolves_at"].shift(LAG) <= frame["decision_at"]).iloc[LAG:].all()` on the exact frame the overlay reads.

## 2. Kill rules: re-entry bound and the longest flat run

A kill rule without a re-entry rule is a rule to stop trading forever (`RiskManager.is_halted` stays True until `reset_halt()` / `initialize()`; `chapters/19_risk_management.md`, "Halt and re-entry in backtests"). A kill rule whose re-entry waits for a market state can keep the book flat for most of the sample (in one reviewed agent run a state-only re-entry rule kept the book flat in 46 percent of months). Every kill rule therefore carries, written before the run:

- a mechanical re-entry path: either a cooldown of `REENTRY_SESSIONS` after which the mandate re-arms on the surviving capital (`rm.initialize(equity, timestamp)`, logged as a new mandate), or a state rule (lagged regime label, drawdown back above the warn threshold) **with a time-based fallback** `MAX_FLAT_SESSIONS` after which the book re-enters anyway, at reduced size if the mandate says so;
- a self-test that prints the longest flat run, the share of sessions flat and the counts of halts and re-entries, and fails when the longest run exceeds `MAX_FLAT_SESSIONS`;
- reporting that keeps halted sessions inside the overlay row as a cost: Sharpe and drawdown over every session with equity flat at the halted level, flattened sessions counted as `(overlay == 0) & (baseline != 0)`, never a Sharpe over active sessions only.

```python
flat = (overlay_w.abs().sum(axis=1) == 0) & (baseline_w.abs().sum(axis=1) != 0)   # flattened sessions (ch19 definition)
runs = flat.ne(flat.shift()).cumsum()
longest = int(flat.groupby(runs).sum().max()) if flat.any() else 0
print(f"flat {flat.mean():.1%} of sessions; longest flat run {longest}; halts {n_halts}; re-entries {n_reentries}")
assert longest <= MAX_FLAT_SESSIONS, "re-entry rule is unbounded: add a time-based fallback"
```

## 3. Cadence parity

A kill rule is backtested at the cadence it will run live. A daily-close `MaxDrawdownLimit` / `DailyLossLimit` is evaluated on daily equity even when the strategy rebalances every 21 sessions: feed daily bars to the `Engine` and let the limits run every bar while `RebalanceSchedule.fixed_n_sessions(21)` governs the decisions, as the ch19 sweep does. A monthly rule is evaluated on monthly equity. When the backtest can only run at the decision cadence, the term sheet states the mismatch and the backtested halt statistics (halt count, flat share, drawdown cut) are reported as not transferable to the live cadence. Self-test: the number of rule evaluations equals the number of sessions for a daily rule and the number of decisions for a decision-cadence rule.

## 4. Precommitment pattern (sweeps, overlays, kill thresholds)

"Frozen" and "precommitted" (`workflow.md` Stage 10, `SKILL.md` guardrail 4, the ch19 sweep protocol) are machine-checkable:

- The overlay or kill-rule configuration file carries `frozen_as_of` (an ISO date) and the declared `grid`; the evaluation output carries `precommitted: bool` = the evaluation window starts after `frozen_as_of` and the configuration file's hash is the one committed at the freeze. The CLI refuses an in-sample evaluation, or labels every number and figure title of one `IN-SAMPLE`; a result without the flag is read as in-sample.
- The deployed configuration must be a row of the declared grid. Its rank in the sweep (calibration window) is printed with the grid size. If the rank is within the top 5 percent of the grid (`rank <= ceil(0.05 * n_cells)`), the report says so and headlines the population median of the sweep (and the share of cells in the win-win quadrant) instead of the pick's own number: the highest-Sharpe cell of a sweep is not the expected outcome of applying the rule, and a best-of-sweep improvement is inflated by the size of the sweep with no deflation applied (`20_strategy_synthesis/07_regime_risk`). A default outside the grid, or a rank-1 default reported as the headline, fails the gate.

```python
cfg = yaml.safe_load(open("overlay.yaml"))   # frozen_as_of: "2021-12-31"; grid: {vol_target: [..], lookback: [..], max_leverage: [..]}; deployed: {..}
grid, dep = cfg["grid"], cfg["deployed"]
cells = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
assert dep in cells, f"deployed configuration is not a row of the declared {len(cells)}-cell grid"
precommitted = pd.Timestamp(eval_start) > pd.Timestamp(cfg["frozen_as_of"])
sweep = pd.DataFrame([{**c, "sharpe": run_overlay(c, calibration_window)} for c in cells]).sort_values("sharpe", ascending=False)
rank = int((sweep[list(grid)] == pd.Series(dep)).all(axis=1).to_numpy().argmax()) + 1
suspect = rank <= math.ceil(0.05 * len(cells))
headline = sweep["sharpe"].median() if suspect else float(sweep.iloc[rank - 1]["sharpe"])
print(f"{'PRECOMMITTED' if precommitted else 'IN-SAMPLE'}: deployed rank {rank}/{len(cells)}; "
      f"headline = {'population median' if suspect else 'deployed cell'} Sharpe {headline:.2f}")
```

## 5. VaR/ES backtest ladder

The chapter's `backtest_var` + `kupiec_test` (`19_risk_management/01_var_cvar`) tests the exception count only. The full ladder, with the consequence written before the test:

| Test | Statistic | Null / reading |
|---|---|---|
| Kupiec POF (unconditional coverage) | `kupiec_test(exceptions, expected_rate) -> (LR_uc, p)`, chi2(1) | count matches `1 - confidence`; exception ratio near 1 |
| Christoffersen independence (standard practice; not in the repo, snippet below) | `LR_ind`, chi2(1), from the 2x2 transition counts of the 0/1 exception series | an exception today does not change tomorrow's exception probability; clustering fails it while Kupiec passes |
| Christoffersen conditional coverage | `LR_cc = LR_uc + LR_ind`, chi2(2) | both at once; the one to report |
| ES (expected shortfall) | Acerbi-Szekely (2014) Z-statistics with critical values simulated under the fitted model, or Du-Escanciano (2017) cumulative-violation tests (standard practice; in neither the repo nor the libraries) | the tail mean beyond VaR is right, not only the frequency |

Basel traffic-light zones (250 trading days at 99 percent): green 0-4 exceptions, yellow 5-9, red >= 10. For another sample length scale the count by `250 / n_obs`; for another confidence re-derive the zone edges from the binomial (at 95 percent the expected count per 250 days is 12.5, so the 99 percent edges do not apply). Consequence ladder, pre-registered in the risk mandate: green -> keep; yellow -> scale exposure down and re-estimate (window, method, conditional model) before the next review; red -> halt the strategy or overlay and re-model; an independence failure with a passing count -> treat as yellow and move to a conditional volatility model (EWMA / GARCH, `chapters/09_model_based_features.md`).

```python
def christoffersen_tests(exceptions, expected_rate):
    """Independence (LR_ind) and conditional coverage (LR_cc = LR_uc + LR_ind) of a 0/1 exception series."""
    e = np.asarray(exceptions, dtype=int); a, b = e[:-1], e[1:]
    n = np.array([[((a == i) & (b == j)).sum() for j in (0, 1)] for i in (0, 1)], dtype=float)   # n[prev, cur]
    with np.errstate(divide="ignore", invalid="ignore"):
        pi_row, pi_pool = n[:, 1] / n.sum(axis=1), n[:, 1].sum() / n.sum()
        ll = lambda p: np.nansum(n[:, 0] * np.log(1 - p) + n[:, 1] * np.log(p))
        lr_ind = 2 * (ll(pi_row) - ll(pi_pool))
    lr_uc, _ = kupiec_test(e, expected_rate)                                                   # 01_var_cvar helper
    return {"lr_ind": lr_ind, "p_ind": float(stats.chi2.sf(lr_ind, 1)),
            "lr_cc": lr_uc + lr_ind, "p_cc": float(stats.chi2.sf(lr_uc + lr_ind, 2)), "n_exceptions": int(e.sum())}

res = christoffersen_tests(bt["exceptions"], bt["expected_rate"])        # bt = backtest_var(returns, window=252, confidence=0.99)
k = res["n_exceptions"] * 250 / bt["n_observations"]                     # exceptions per 250 days at 99 percent
zone = "green" if k <= 4 else "yellow" if k <= 9 else "red"             # consequence looked up in the pre-registered ladder
```

## Related references

- `chapters/19_risk_management.md` — VaR recipe, vol targeting, halt and re-entry, sweep protocol: the sections this file extends.
- `chapters/20_strategy_synthesis.md` — `07_regime_risk`: population-of-sweep reading and the win-win quadrant.
- `chapters/07_defining_the_learning_task.md` — anchors and `resolves_at` for the lag rule; variable-horizon labels.
- `chapters/09_model_based_features.md` — EWMA / GARCH conditional volatility for a VaR that fails independence.
- `decision_rules.md` §14c (kill-criteria sheet), `workflow.md` Stage 10, `preflight.md` 30(a)/(c), `libraries/ml4t_backtest.md` (`RiskManager`, portfolio limits, `RebalanceSchedule`).
- Further reading: Kupiec (1995); Christoffersen (1998); Acerbi & Szekely (2014); Du & Escanciano (2017); Basel Committee (1996), supervisory framework for the use of backtesting with the internal models approach.
