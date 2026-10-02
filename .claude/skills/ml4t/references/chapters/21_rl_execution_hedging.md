# Chapter 21: Reinforcement Learning for Execution and Hedging

> Equips you to cast execution, market making and derivatives hedging as (partially observed) MDPs, calibrate a simulator to real data (GARCH(1,1) on hourly BTCUSDT), train DQN / PPO / A2C with stable-baselines3, and benchmark the learned policy against TWAP, Almgren-Chriss, reservation-price quoting, Black-Scholes delta and Whalley-Wilmott on identical paths with paired standard errors. The chapter's position: RL is a tool for *control*, not prediction, and is credible only where objectives are clear, feedback loops tight and rewards defensible (execution, market making, hedging), not for broad alpha discovery. Any "RL beats the benchmark" claim is only as good as simulator realism, reward design and honest, paired evaluation. The governing caution is the simulation-to-reality gap (non-stationarity, impact reflexivity, latency, fill assumptions, reward hacking), which, not algorithm choice, blocks deployment. Source: companion repo `21_rl_execution_hedging/` (book PDF unavailable; notes extracted from README and notebooks).

## When to use this reference

- Designing state, action and reward for an RL agent that paces an order, quotes two-sided markets, or hedges an option book.
- Choosing between DQN, PPO, A2C and SAC for a trading control task (discrete vs continuous actions, stability vs sample efficiency).
- Building or calibrating a market simulator (GARCH(1,1) volatility, regime Markov chain, spread / depth / impact proxies) and labelling what is fitted, proxied or assumed.
- Benchmarking an RL execution policy against TWAP and Almgren-Chriss, and deciding whether a reported gain is real or borrowed from the end of the horizon.
- Replaying recorded crypto perpetual bars (premium index, funding timing) as an execution environment with correct feature lags and date splits.
- Comparing a learned market-making policy to Avellaneda-Stoikov-style reservation-price rules and isolating what learning adds.
- Deep hedging under discrete rebalancing and proportional costs: choosing an expected-shortfall objective and comparing P&L distributions, not means.
- Inferring a trader's objective from trade logs (inverse RL) or cloning behaviour, and knowing what is identifiable.
- Adding market impact to a backtest and replacing a fixed-bps cost assumption with a participation-rate model (`ml4t.backtest` `SquareRootImpact`).
- Evaluating or reviewing any learned policy: paired SE on shared seeds, position-distribution and turnover diagnostics, resolution thresholds.
- Assessing whether an RL policy is anywhere near deployable (sim-to-real checklist, offline RL, off-policy evaluation, staged deployment).

## Core ideas (the why)

- **Prediction vs control.** Supervised learning forecasts; RL chooses actions when actions change future outcomes. Use RL only where the objective is clear, feedback is tight and the reward is defensible: execution, market making, hedging. Not broad alpha discovery (21.1).
- **The MDP spec is a modeling decision.** State, action, reward, transition and discount embed market assumptions, constraints and economic goals. Partial observability (POMDP) is the norm, which is why state engineering, recurrent models and *private* agent information (inventory, time remaining) matter (21.2).
- **Algorithm map.** Discrete actions -> value-based (DQN, Double DQN, Dueling). Realistic continuous control -> actor-critic (PPO, A2C, SAC); SAC requires continuous actions. Trade-off: stability vs sample efficiency vs action-space flexibility. Risk-sensitive variants exist for finance (21.3).
- **A simulator is mandatory, but must not invent statistics.** An agent's actions must change what it sees next; a recorded series cannot respond. Fit the simulator's volatility to data (GARCH(1,1)) and state which parameters are fitted, which are proxies, which are assumed or clamped.
- **Mean reward says almost nothing.** The distribution of chosen positions and turnover separate a policy that trades from one that collapsed onto a single position.
- **Execution.** RL is a dynamic alternative to TWAP / Almgren-Chriss when liquidity and volatility vary. A low average cost achieved by postponing volume to the end of the horizon is a bet on terminal liquidity, not a saving (21.4).
- **Market making.** Inventory response (reservation price) is the textbook baseline and is programmed into every arm; the only honest question for a learned policy is what skew and adaptive spread width add on top (21.5).
- **Deep hedging.** Under discrete rebalancing and proportional costs exact replication is impossible; the question becomes which terminal P&L distribution to accept. Fix a risk measure (expected shortfall) and minimise it directly; compare on ECDF tails, not means (21.6).
- **IRL vs behaviour cloning.** BC fits a policy with no explanation and fails by compounding error once it drives. IRL fits a reward (an explanation) that can in principle transfer, but a linear reward over correlated features is not identifiable: inferred weights describe covariation with demonstrated behaviour, not preferences (21.7).
- **Sim-to-real gap is the real barrier.** Non-stationarity, impact reflexivity, latency, bad fill assumptions, reward hacking. Remedies: simulator fidelity, offline RL, off-policy evaluation, staged deployment, governance (21.8).
- **Fixed bps per trade is the spread, not the cost of trading.** The larger part of real cost is impact, which depends on participation rate (order vs available volume), not dollar size.
- **Paired evaluation is the correct inference device.** Every strategy on the same seed, difference taken within episode, paired SE reported. Pairing does not guarantee a narrower interval: var(a-b) = var a + var b - 2cov, and opposite positions on one path are negatively correlated.

## Method recipes (the how)

### Formulating a financial control task as a (PO)MDP

| Element | Execution | Market making | Hedging |
|---|---|---|---|
| State (observed) | inventory_ratio, time_ratio, spread, depth, volatility, regime (+ premium, volume ratio, hour, hours-to-funding on recorded bars) | inventory/limit, price change, vol, imbalance, time_ratio, spread bps | log moneyness, time to expiry, volatility, previous position |
| Private state | remaining shares, time left | inventory | current hedge position |
| Action | pace multiplier in [0,1] -> 0.5x..1.5x of even rate | discrete (skew x spread width) = 9 actions | hedge ratio in [0,1] (sigmoid) or unconstrained |
| Reward | -(shortfall + risk penalty + schedule penalty)/total_shares | dW_t - lambda_q (q_t/q_max)^2 p_{t+1}, clipped [-100,100] | terminal: minimise ES_q of self-financing P&L |
| Benchmarks | TWAP, Almgren-Chriss (oracle), funding-aware rule | 3 reservation-price rules | BS delta, Whalley-Wilmott, tabular QLBS |

Steps: (1) write the economic objective first (shortfall, marked wealth, tail loss); (2) decide what the agent can observe at decision time and lag everything else; (3) scale observations to order one (returns x10, vol x100, spread bps /10, premium x100, hours-to-funding /8); (4) express step rewards in percent, not fractions, so SB3 default learning rates make progress; (5) install the action for t+1, never pay the action with a return already visible.

### GARCH(1,1) market calibration (`rl_calibration.CryptoMarketCalibrator`)

- Model: sigma_t^2 = omega + alpha r_{t-1}^2 + beta sigma_{t-1}^2, fit with `arch` on hourly log close returns of `BTCUSDT` (`data.load_crypto_perps`).
- Stationarity ceiling: if alpha + beta >= ceiling (value not read), both are scaled down so the sum sits exactly on the ceiling; omega re-derived as `max(uncond_vol**2 * (1 - alpha - beta), 1e-12)` so long-run variance matches the sample. `GARCHParams.persistence()` = alpha + beta. Moment-based fallback `_fit_garch_moments`: alpha = min(0.1, max(0.02, rho_1 * 0.3)).
- Simulation: start at long-run variance (no burn-in for level), normal draws. Validate with autocorrelation of squared returns up to `MAX_LAG = 48` hours; it should stay positive for tens of hours in both data and sim.
- Regimes: `fit_regimes(n_regimes=2, method="volatility")` thresholds rolling vol -> 2x2 transition matrix, per-regime means / stds. `method="hmm"` uses `GaussianHMM(n_components=n_regimes, n_iter=100)` with fallback to the volatility method.
- `estimate_spread_depth()`: spread proxy = median hourly high-low range clipped to `[0, spread_cap]` (an upper bound on the true spread); depth proxy = median hourly volume; returns normal and stressed values.
- `estimate_impact()`: Amihud ratio (mean |return| per dollar traded) rescaled; clamps `permanent_impact` to [0.01, 0.5] and `temporary_impact` to [0.001, 0.1]. For BTC the estimate falls below the lower clamp, so the env receives the clamp itself.
- `summary()`: unconditional vol annualised with sqrt(8760) hours, P(stay normal), P(stay stressed), vol per regime. Print it and label every parameter fitted / proxy / clamped.

### Algorithm comparison on one environment (`21_rl_execution_hedging/01_algorithms_comparison`)

| Setting | Value |
|---|---|
| `STEPS_PER_EPISODE` | 252 hours (~10.5 days) |
| `TOTAL_TIMESTEPS` | 50_000 per algorithm |
| `TRADING_COST_BPS` | 10 per unit of position change |
| `EVAL_EPISODES` | 10 |
| `VOL_WINDOW` / `MOM_WINDOW` | 20 / 5 |
| `SEED` / `DIAGNOSTIC_SEED` | 314 / 456 |

- `SimpleTradingEnv`: actions short (-1), flat (0), long (+1). Observation (5): current return, previous return, sd of last 20 returns, sum of last 5 returns, current position; vol and momentum multiplied by 10; every window ends at idx.
- Reward: p_{t-1} r_t minus cost of any change, in percent. Action chosen at t is installed for t+1.
- DQN: value-based, off-policy, replay buffer, greedy on Q(s,a). PPO: on-policy actor-critic, clipped objective (`clip_range`). A2C: actor-critic without clipping, very short rollouts, cheaper and less stable; included because SAC needs continuous actions.
- Isolation: each algorithm gets its own env via `make_env(seed)` so one's training cannot advance another's RNG; evaluation builds a fresh `DummyVecEnv` per seed.
- Diagnostics: `run_episode(model, env)` records the position that earned the step, next position, reward, NAV; plot position-count distribution; `Turnover` = units of position change (0 for never moves, 2x episode length for flipping every step). `comparison_row(reference, other)` prints paired and unpaired SE side by side.
- Exact PPO/DQN/A2C hyperparameters (ent_coef, n_steps, batch, gamma) were not visible in the notes.

### Optimal execution: `ExecutionEnv`, TWAP, Almgren-Chriss, PPO (`21_rl_execution_hedging/02_optimal_execution_ppo`)

| Setting | Value |
|---|---|
| `TOTAL_SHARES` / `EXECUTION_HORIZON` | 10_000 / 60 steps |
| `TOTAL_TIMESTEPS` / `EVAL_EPISODES` | 100_000 / 20 |
| `EVAL_SEED_BASE` / `SEED` | 1000 / 314 |
| `RISK_AVERSION` | 1e-4 per unit price variance (env default 0) |
| `SCHEDULE_PENALTY` | 5e-5 (env default 0) |
| `REFERENCE_STRATEGY` | "TWAP" (fixed before running) |
| `max_participation_rate` | 0.35 of depth |

- `ExecutionEnv(gym.Env)`: obs (6) = [inventory_ratio, time_ratio, spread, depth, volatility, regime]; action in [0,1] = pace multiplier (0 -> 0.5x even rate, 1 -> 1.5x); `max_trade_size = min(remaining, max(1, min(schedule_cap, 0.35*depth)))`; the final step sells the entire remainder ignoring the cap.
- Market path: GARCH variance (floor 1e-10), 2-state regime Markov chain from the transition matrix (start normal), spread floor 0.0001, depth floor 100, price floor 0.01; unaffected price is a martingale (zero drift).
- Costs: participation pr = shares/depth; temp_impact = temporary_impact * (pr + pr^2); perm_impact = permanent_impact * (shares/total_shares); plus half-spread. Reward = -(shortfall + risk_penalty + schedule_penalty)/total_shares; risk penalty on inventory_ratio = remaining/total; schedule penalty on deviation_ratio = (shares - reference)/max(reference, 1). Exact penalty functional forms are inferred from grep lines, not full source.
- Implementation shortfall for a sale: IS = sum_i (P_0 - P_i), reported in bps of arrival notional P_0 Q. Components: half-spread, temporary impact (grows with participation), permanent impact, timing risk from waiting.
- TWAP: equal quantity each step; minimises impact when nothing is known about liquidity / vol variation; the desk benchmark.
- Almgren-Chriss (linear impact): q_t = Q sinh(kappa (T - t)) / sinh(kappa T), kappa^2 = lambda sigma^2 / eta. Raising lambda or sigma front-loads; kappa -> 0 gives TWAP. `discrete_almgren_chriss_schedule(total_shares, horizon, sigma_price, eta, gamma, risk_aversion)`. Implemented as an ORACLE: sigma and eta are whole-episode averages no live trader has.
- PPO: entropy bonus `ent_coef` keeps the policy from collapsing onto TWAP early; `ppo_execution` acts greedily (deterministic) at evaluation.
- Diagnostics: `episode_diagnostics` records cost plus fraction traded in the final quarter (even reference 25%) and on the last step (even reference 1/60); `add_execution_rate_band` draws the 10th-90th percentile execution-rate band per step (narrow for static schedules, wide for state-responsive ones).

### Market making: `MarketMakingEnv` + discrete PPO (`21_rl_execution_hedging/03_market_making_ppo`)

| Setting | Value |
|---|---|
| `EPISODE_LENGTH` / `TOTAL_TIMESTEPS` / `EVAL_EPISODES` | 500 / 300_000 / 240 |
| `INVENTORY_LIMIT` / `LAMBDA_INVENTORY` | 100 / 0.001 |
| `LEARNING_RATE` | 3e-4 |
| `SKEW_LEVELS` x `SPREAD_MULTIPLIERS` | [-1, 0, 1] x [0.8, 1.0, 1.25] -> 9 actions via `encode_action(skew_idx, spread_idx)` |
| `RESOLUTION_SIGMA` | 3.0 SE to call an ordering |
| `REFERENCE_BASELINE` | "Reservation + Base Spread" |
| `EVAL_SEED_BASE` / `VIZ_SEED` / `SEED` | 1000 / 999 / 314 |

- Dynamics (`MarketMakingDynamics`, frozen dataclass): GARCH variance (floor 1e-8), shock clipped [-5, 5], return clipped [-0.1, 0.1]; imbalance_t = clip(0.9*imbalance_{t-1} + 0.1*U(-1,1), -1, 1) (mean-reverting AR(1)) nudges price. Arrival rate, sensitivity and base spread values in `MM_DYNAMICS` were not read.
- Quotes (`compute_quotes`): reservation price p^r = p (1 - (q/q_max) sigma); centre = p^r + skew * 0.25 * sigma * p; half_spread = 0.5 * base_spread * p * (1 + 5.0*sigma) * spread_mult; bid = max(centre - hs, 0.01), ask = max(centre + hs, bid + 0.01). The reservation price is applied to every arm, learned and rule-based alike.
- Fills: P(fill) = 1 - exp(-lambda e^{-kappa d/s} m). `fill_probability(distance, imbalance_factor, base_spread)`: scaled = d/base_spread, intensity = arrival_rate * exp(-sensitivity * scaled) * m. Bid m = clip(1 - 0.35*imbalance, 0.2, 2.0); ask m = clip(1 + 0.35*imbalance, 0.2, 2.0) (adverse-selection channel).
- Observation (6), scaled and clipped: inventory/limit [-1,1]; price_change*10 [-1,1]; vol*100 [0,10]; imbalance [-1,1]; time_ratio [0,1]; spread_bps/10 [0,10].
- Reward r_t = dW_t - lambda_q (q_t/q_max)^2 p_{t+1}, with W = cash + inventory marked at next price; clipped [-100, 100]; terminal inventory liquidated at half-spread in the last step; the side that grows inventory stops quoting at the limit.
- Training: SB3 PPO, MLP 2x64 for actor and critic, `VecNormalize` on obs and reward; freeze stats before eval; switch reward normalisation off at eval so wealth is in env units. Entropy bonus prevents early commitment to the widest spread (near-riskless, near-income-free).
- Evaluation: `run_episode(env, seed, model=None, fixed_action=None, vec_normalize=None)` with the seed passed per call; `paired_gap(baseline)` returns mean gap, SE and arm correlation; report gaps to all three baselines with the reference fixed in advance.
- Diagnostic: plot the total quote offset and the skew component (a_skew * 0.25 * sigma * p) against inventory held WHEN the quote was posted.

### Execution on recorded crypto bars: `CryptoExecutionEnv` (`21_rl_execution_hedging/04_crypto_execution_rl`)

| Setting | Value |
|---|---|
| `SYMBOL`, `START_DATE`, `END_DATE` | "BTCUSDT", "2022-01-01", "2024-12-01" |
| `EVAL_START_DATE` | "2024-01-01" (train before, report on/after) |
| `TOTAL_SHARES` / `EXECUTION_HORIZON` | 100.0 contracts / 24 hours |
| `TOTAL_TIMESTEPS` / `EVAL_EPISODES` | 200_000 (~8,000 distinct 24h windows) / 200 |
| `RISK_AVERSION` / `SCHEDULE_PENALTY` | 1e-4 / 5e-4 |
| `FUNDING_INTERVAL_HOURS` | 8 (00:00 / 08:00 / 16:00 UTC) |
| `PREMIUM_BAND_BPS` | 5.0 (rich / neutral / cheap) |
| `max_participation_rate` | 0.10 of previous hour's volume |
| `EVAL_SEED_BASE` / `SEED` | 1000 / 314 |

- Panel: hourly bars (`data.load_crypto_perps`) joined with the premium index (`data.load_crypto_premium`, published every 8h, forward-filled). Lag by one hour: return, rolling volatility, volume, premium index. Allowed unlagged: open price of hour t and the calendar. Split by date with an in-code assertion that train / eval windows do not overlap.
- Observation (7): inventory_ratio, time_ratio, volatility (24h rolling through previous bar), premium_index (x100), volume_ratio (prev hour / 24h avg, capped 5.0), hour_of_day, hours_to_funding (/8). Columns `premium_index_close`, `hours_to_funding`.
- Impact: participation = (shares + concurrent)/(volume + 1e-8); cost ∝ sqrt(pr) + pr (square-root-plus-linear). No cap exemption on the final hour: the remainder is unwound against the same bar and both legs are charged on combined participation (splitting costs the same as one trade); episode flagged forced liquidation when residual > 1e-9 * total_shares.
- Funding-aware rule: even pace, speeds up when premium is rich (saturates at `PREMIUM_SCALE = 0.001`) and when settlement is within `FUNDING_WINDOW_HOURS = 2` (`FUNDING_SPEEDUP = 1.25`). Gap to TWAP = value of conditioning at all; gap to the rule = value of learning vs hand-writing it.
- Conditioning diagnostic: pool all eval steps, group execution rate by premium state and hours-to-funding; read as association only.

### Deep hedging (`21_rl_execution_hedging/05_deep_hedging_pfhedge`)

| Setting | Value |
|---|---|
| `N_PATHS` / `N_STEPS` / `MATURITY` | 10_000 per set / 20 / 30/365 |
| `SPOT` / `STRIKE` | 100 / 100 (ATM); `PFHEDGE_STRIKE = STRIKE/SPOT`; `DT = MATURITY/N_STEPS` |
| `COST_BPS` (`COST_RATE`) | 10 (1e-3) |
| `N_EPOCHS` / `EXPECTED_SHORTFALL_Q` | 200 / 0.05 |
| Heston | `V0_VOL = 0.20` (initial = long-run), `KAPPA = 1.5`, `SIGMA = 0.35`, `RHO = -0.7` |
| `WHALLEY_WILMOTT_RISK_AVERSION` | 1.0 |
| Seeds | `TRAINING_SEEDS = [42, 314, 2718]`, `VALIDATION_SEED = 777`, `EVAL_SEED = 999`, `QLBS_TRAIN_SEED = 10_999`, `SEED = 42` |

- P&L of a self-financing short call: sum_t delta_t (S_{t+1} - S_t) - c sum_t |delta_t - delta_{t-1}| S_t - max(S_T - K, 0), delta_{-1} = 0. `hedging_pnl` (numpy) and `hedging_pnl_torch` (differentiable). Verify pfhedge's P&L against this formula using BS delta positions before comparing anything.
- Objective: ES_q = -E[P&L | P&L <= F^{-1}(q)], q = 0.05. Not variance: only the losing tail counts.
- Heston: dS = sqrt(v) S dW^S, dv = kappa (theta - v) dt + sigma sqrt(v) dW^v, corr rho. BS delta on Heston paths is misspecified, so analytical baselines get each path's realised vol.
- Baselines: BS delta Phi(d_1) at each date; Whalley-Wilmott no-trade band around delta (width grows with cost, shrinks with risk aversion; constants not visible), move to the band edge when outside.
- `pfhedge.Hedger` with 4 inputs (log moneyness, time to expiry, volatility, previous position), MLP (architecture / criterion class name not visible); train 3 seeds on their own paths, select the MEDIAN validation ES on `VALIDATION_SEED` paths; evaluate once on `EVAL_SEED` paths untouched by training or selection.
- `FromScratchHedger(hidden_size=32)`: 3 layers, sigmoid output restricting position to [0,1] (pfhedge does not constrain); `rollout` loops over time (position feeds next input); `expected_shortfall_loss`; full-batch Adam, same paths and epochs as the chosen pfhedge run.
- Tabular QLBS-style Q-learner: `MONEYNESS_BINS = linspace(-0.3, 0.3, 7)`, `TIME_BINS = linspace(0, 1, 5)`, `HEDGE_LEVELS = linspace(0, 1, 11)`; reward R(c) = -c - lambda c^2 with `QLBS_RISK_AVERSION = 0.5`; `QLBS_LEARNING_RATE = 0.1`, `QLBS_DISCOUNT = 0.99`; epsilon decays over paths; trained on its own path set (`QLBS_TRAIN_SEED`). API: `qlbs_update(q_table, path, exploration)`, `state_index`.
- Reporting: `pnl_stats` = mean, sd, ES at q, worst path; ECDF (tails) plus box plot (middle); single-path position overlay.

### Inverse RL vs behaviour cloning (`21_rl_execution_hedging/06_inverse_reinforcement_learning`)

| Setting | Value |
|---|---|
| `TOTAL_SHARES` / `EXECUTION_HORIZON` / `INITIAL_PRICE` | 5_000 / 30 / 100 |
| `N_EXPERT_TRAJECTORIES` / `TRAIN_FRACTION` | 200 / 0.8 (split by whole episode) |
| `BC_EPOCHS` | 500 |
| `IRL_ITERATIONS` / `N_CANDIDATE_TRAJECTORIES` / `MAXENT_L2` | 100 / 400 / 0.05 |
| `FEATURE_MATCH_ITERATIONS` / `N_FEATURE_MATCH_TRAJECTORIES` | 50 / 50 |
| `N_EVALUATION_EPISODES` / `SEED` | 200 / 42 |

- Expert `twap_policy(obs, env)` converts the equal slice through `env.target_shares_to_action`; demonstrations carry only two distinct actions (reference pace, and the conversion's max on the last step) and no state-explained variation.
- Behaviour cloning: ridge regression and a small NN on (state -> action); the gap between them tests linearity. Evaluate by DRIVING the env on common seeds (`evaluate_policy`, `shortfall_bps`), not by held-out fitting error.
- Linear reward R(s,a) = w^T phi(s,a); trajectory reward w^T Phi(tau). MaxEnt IRL: P(tau|w) = exp(w^T Phi(tau))/Z(w); gradient = expert mean feature count - expected count under the current distribution. Approximate Z over a FIXED bank built once (expert paths + 5 proposal kinds: noisy-TWAP, front-loaded, back-loaded, liquidity-sensitive, random) via `proposal_action(obs, env, kind, rng)` and `build_candidate_bank`. Stable log-sum-exp (`maxent_distribution`); standardise features (`prepare_maxent_problem`); `regularized_log_likelihood` with L2; `maxent_irl(..., n_iterations=100, learning_rate=0.05, l2_penalty=0.05)`; map weights back to original units.
- Feature-expectation matching: same direction mu_E - mu_pi_k but the expectation comes from fresh noisy-greedy rollouts (`best_greedy_action` over [0.01, 1.0]); `feature_expectation_irl(..., n_iterations=50, learning_rate=0.2, n_sample_trajectories=50, seed=42)`; no scalar objective, can cycle.
- Reward -> policy diagnostics: MaxEnt uses a one-step Boltzmann policy over an action grid, `inferred_boltzmann_action(obs, weights, temperature=0.2)` (temperature chosen, not inherited); feature matching uses its greedy policy without noise. `gap_to_expert` reports the per-episode paired shortfall difference, never a ratio of means. `extract_features` contents not visible.

### Impact-aware backtest with `ml4t.backtest` (`21_rl_execution_hedging/07_backtest_with_impact`)

| Setting | Value |
|---|---|
| `START_DATE` / `END_DATE` | "2010-01-01" / "2016-12-31" |
| `FORMATION_END_DATE` / `EVALUATION_START_DATE` | "2012-12-31" / "2013-01-01" |
| `LOOKBACK` | 21 days |
| `BOOK_USD` | 5_000_000 fixed across names |
| `IMPACT_COEFFICIENTS` | [0.0, 0.1, 0.3, 0.6] |
| `GROSS_MIN` / `MIN_FORMATION_OBS` | 0.3 formation gross return / 700 sessions |
| `COHORT_SAMPLE` | 60 names per cohort |
| `LIQUID_MAX_PARTICIPATION` / `THIN_MIN_PARTICIPATION` | 0.10 / 1.0 of daily volume |
| `SEED` | 42 |

- Square-root law: impact = c * sigma * sqrt(q/ADV) * p; sigma and ADV from the formation window only (`formation_statistics`). Dollar volume (price x volume) is the liquidity measure (split-invariant); adjusted OHLCV via `data.load_us_equities`.
- Selection: pool = names with formation gross momentum return > 0.3 and >= 700 formation sessions; 5 tiers by ex-ante dollar-volume percentiles; the MEDIAN gross-return name per tier (`select_spectrum`).
- `MomentumStrategy(symbol, lookback=21, min_trade_notional=1_000)`: long-only, fully invested when 21-day momentum > 0 else cash. `run_backtest(stock_data, symbol, impact_coef, book_usd, ex_ante_volatility)`; `run_spectrum` runs every (stock, coefficient) pair; failures are re-raised, never zeroed.
- Erosion = no-impact return - high-impact (c=0.6) return. `flip_rate(prices, candidates, book_usd, n_sample)` counts profit -> loss sign flips per cohort with data attrition reported; both runs pay the same commission and slippage. Scatter no-impact vs high-impact return: vertical distance below the diagonal = impact alone.

### Paired evaluation protocol (all notebooks)

1. Fix the reference arm before running (`REFERENCE_STRATEGY`, `REFERENCE_BASELINE`).
2. Run every arm on the same seed list `range(EVAL_SEED_BASE, EVAL_SEED_BASE + EVAL_EPISODES)`, fresh env per seed.
3. Take the difference per episode; `se = diff.std(ddof=1) / sqrt(n)`; report arm correlation.
4. Print the unpaired SE alongside only to show what pairing changed.
5. Report gaps to ALL baselines; call an ordering only above `RESOLUTION_SIGMA = 3.0` SE, otherwise "cannot call".
6. Add behaviour diagnostics: position-count distribution, turnover, volume location in the horizon, forced-liquidation rate, quote-offset vs inventory.

### Simulation-to-reality gap (21.8)

Barriers: non-stationarity, impact reflexivity (the policy's own trades change the market it was trained on), latency, fill assumptions, reward hacking. Remedies named by the book: simulator fidelity checks, offline RL, off-policy evaluation, staged deployment, governance. No procedure or thresholds were available in the notes; treat these as a checklist, not a recipe.

## Guardrails and pitfalls

- **Reward-timing lookahead** — rewarding the action chosen at t with r_t already visible at t makes every simulated result meaningless. Pay p_{t-1} r_t and install the action for t+1 (`SimpleTradingEnv.step`); every observation window ends at idx.
- **Feature lag on recorded bars** — the current hour's return, volume, vol and premium have not closed when the decision is made, and volume is what impact is charged against. Shift all four by one hour; only the hour's open price and the calendar are unlagged (notebook 04).
- **Train/eval date overlap** — episodes drawn across the split boundary put training windows into reported numbers. Train on bars before `EVAL_START_DATE`, report on/after; assert the split in code, not prose.
- **Formation/evaluation separation in backtests** — selecting names or calibrating sigma / ADV with evaluation-period data scores a name partly on returns it was chosen for. All selection and calibration read bars up to `FORMATION_END_DATE`; report from `EVALUATION_START_DATE`; report attrition without changing selection.
- **Leakage through episode-level splits** — adjacent states in one episode are near-duplicates, so a state-action split makes held-out error a fit to its own answers. Split demonstrations by whole episode (`TRAIN_FRACTION = 0.8` of episodes).
- **Oracle benchmark presented as competitor** — Almgren-Chriss computed from whole-episode sigma and depth uses information no live trader has. State the advantage; read AC as the ceiling of a perfectly informed static plan, not a baseline PPO "improved on".
- **Simulator proxies misread as measurements** — spread = hourly high-low range (upper bound, every strategy overpays), depth = traded volume (rises in stress while real depth falls, so the stressed regime is wider AND deeper and the participation cap looser), impact = clamp values ([0.01, 0.5] perm, [0.001, 0.1] temp; never measured). Print which parameters were fitted, proxied or clamped; condition every claim on the spread process.
- **Zero-expected-reward environment** — GARCH sim draws returns around zero, so no policy has positive expected reward after cost; a positive eval is sampling variation. Judge policies by exposure sizing and cost behaviour, not the sign of reward.
- **Mean reward hides collapse** — two policies with equal mean, one trading and one frozen. Plot position-count distribution and turnover; inspect a shared-episode trajectory.
- **Unpaired standard errors** — per-strategy means are not independent samples; the shared path cancels in the difference. Same seeds for all arms, difference per episode, paired SE; remember pairing can widen the interval when arms are negatively correlated.
- **Baseline selection after the fact** — testing against whichever baseline looks best makes the interval a selection artefact (multiple testing). Fix the reference in advance; report gaps to all baselines.
- **Under-powered ordering claims** — a 1-2 SE gap is evaluation noise at these episode counts. Require `RESOLUTION_SIGMA = 3.0` SE; otherwise "cannot call".
- **Borrowed cost at horizon end** — a low average shortfall from postponing volume to the last step where the cap is lifted is a bet on terminal liquidity. Report fraction traded in the final quarter (ref 25%) and on the last step (ref 1/60); on real bars keep the cap and charge forced liquidation on combined participation.
- **Forced-liquidation rate misread** — the flag counts episodes with an involuntary trade, not amounts; a policy can trade less late than TWAP yet still leave residual. Read the rate beside the volume-location columns.
- **Conditioning diagnostics as causation** — execution rate grouped by premium state differs, but groups differ in inventory, horizon and cap too. Label it association; no claim the policy reads the premium.
- **Normalisation statistics dropped at evaluation** — a `VecNormalize`-trained policy on raw inputs sees a distribution it never saw. Freeze running stats before eval, keep obs normalisation on, switch reward normalisation off so wealth is in env units.
- **Policy collapse during PPO** — early commitment to TWAP (execution) or the widest spread (market making) is locally safe with little incentive to explore. Use the entropy bonus `ent_coef`; inspect the action distribution.
- **Inventory treated as free optionality** — unpenalised inventory carries price risk. Quadratic inventory penalty lambda_q (0.001), terminal liquidation at half-spread, stop quoting the growing side at the limit.
- **Inventory response misattributed to learning** — `compute_quotes` applies the reservation price to every arm. Plot the skew component separately; put inventory at posting time, not post-fill inventory, on the x-axis.
- **Library P&L convention unverified** — five hedges, three computed outside pfhedge; the comparison is void if conventions differ. Recompute P&L from pfhedge's positions with `hedging_pnl` and compare.
- **Seed/model selection on the evaluation set** — picking the best of 3 training runs on eval paths is optimistic. Select the median validation ES on `VALIDATION_SEED = 777` paths; evaluate on `EVAL_SEED = 999` once.
- **Tabular Q memorising paths** — a coarse grid still has capacity to memorise realised prices. Train the QLBS table on its own set (`QLBS_TRAIN_SEED = 10_999`).
- **Device-dependent reproducibility** — seeded PyTorch runs differ across CUDA / PyTorch builds. Print versions; expect different P&L on a different device.
- **Training budget confounded with method** — loss still descending at the last epoch means the budget, not the method, set the result. Inspect the training curve; raise `N_EPOCHS`.
- **IRL non-identifiability** — a linear reward over correlated features is not unique; TWAP conditions on nothing, so any weight on depth / vol is a correlate. Test on a demonstrator with a known objective; compare weights only on standardised features; call them associations.
- **Feature matching has no objective** — iterating mu_E - mu_pi looks like gradient ascent but has no scalar objective, can cycle, and stops where the budget ran out. Use MaxEnt with a fixed candidate bank (log-sum-exp partition) when a likelihood is needed; keep the bank fixed across iterations.
- **Behaviour-cloning distribution shift** — a clone that fits held-out demos closely degrades when driving because small errors take it to unseen states and compound. Evaluate by driving the environment on common seeds.
- **Ratio-of-means comparisons** — policy shortfall / expert shortfall is dominated by price drift, small with large SE, and can sit either side of zero. Use the per-episode paired difference.
- **Fixed-bps cost models** — bps per trade is the spread; impact depends on participation and is larger. Use `SquareRootImpact` with ex-ante sigma and ADV; test across coefficients [0, 0.1, 0.3, 0.6] and liquidity cohorts.
- **Silent failure in batch backtests** — a failed run treated as zero return biases erosion and flip rates. Collect and re-raise failures.
- **Deployment gap (21.8)** — non-stationarity, impact reflexivity, latency, fill assumptions and reward hacking, not algorithm choice, block deployment. Simulator fidelity checks, offline RL, off-policy evaluation, staged deployment, governance.

## Decision rules and defaults

| Decision | Rule |
|---|---|
| Algorithm | Discrete actions -> DQN (or A2C / PPO with discrete heads); continuous -> PPO / A2C / SAC (SAC continuous only); need stability -> PPO (clipped objective); need sample efficiency -> off-policy (DQN / SAC) |
| Simulator vs recorded bars | Simulator when actions must affect the next observation; recorded bars when you need real state variables (basis, funding, volume) and accept that fills are still modelled |
| Reward scale | Step returns in percent so SB3 default learning rates make progress |
| Observation scale | Order one: returns x10, vol x100, spread bps /10, premium x100, hours-to-funding /8 |
| Pacing action | Multiplier in [0,1] mapped to 0.5x-1.5x of even pace |
| Participation cap | 0.35 of depth (sim) / 0.10 of previous-hour volume (recorded bars) |
| Reward weights | Execution risk aversion 1e-4; schedule penalty 5e-5 (sim) / 5e-4 (crypto); market-making inventory penalty 0.001 |
| Training budgets | 50k steps (3-algo comparison), 100k (execution sim), 300k (market making), 200k (crypto execution) |
| Eval episodes | 10 (3-algo) / 20 (execution sim) / 240 (market making) / 200 (crypto) / 200 (IRL) |
| Evaluation protocol | Same seed list for every arm (`EVAL_SEED_BASE = 1000` consecutive), fresh env per seed, per-episode difference, paired SE, reference fixed before running, gaps to all baselines, ordering only above 3 SE |
| Schedule shape checks | Final-quarter share vs 25%; last-step share vs 1/60; 10th-90th percentile execution-rate band |
| Funding-aware rule | Speed up 1.25x within 2 hours of settlement and when premium rich (saturating at 0.001); premium bands at +/-5 bps |
| Deep hedging | 20 rebalances over 30 days, 10 bps cost, ES at 5%, 10k paths per set, 200 epochs, 3 seeds, pick median validation ES; compare on ECDF tails + ES + worst path, not mean |
| IRL | Fixed bank of 400 proposals + expert paths; MaxEnt 100 iterations, lr 0.05, L2 0.05; feature matching 50 iterations, lr 0.2, 50 rollouts; Boltzmann temperature 0.2 for the policy diagnostic |
| Impact backtest | Formation 2010-2012, evaluation 2013-2016; pool = gross return > 0.3 and >= 700 formation sessions; 5 dollar-volume tiers, median name each; liquid cohort = order < 10% of daily volume, thin cohort > 100%; 60 names per cohort; coefficients 0 / 0.1 / 0.3 / 0.6 |
| Model selection | Median validation metric on a validation seed; evaluation set touched once |
| GARCH sanity | Squared-return autocorrelation positive for tens of hours (to lag 48) in data and sim |

Go/no-go for an RL execution claim (all six must hold):
1. Simulator parameters labelled fitted / proxy / assumed (or clamp).
2. Benchmarks run on identical paths.
3. Oracle advantages (Almgren-Chriss whole-episode sigma, eta) disclosed.
4. Cost location in the horizon reported (final quarter, last step).
5. Paired SE above the resolution threshold (3 SE).
6. Forced-liquidation rate read together with the volume-location columns.

## Code patterns and APIs

Run notebooks from the repo root: `uv run python 21_rl_execution_hedging/<notebook>.py`; test mode `uv run pytest tests/test_chapter_notebooks.py -v -k "21_rl_execution_hedging"` (Papermill, reduced data). Helper modules: `rl_calibration.py`, `rl_environments.py`, `market_making_env.py`, `crypto_execution_env.py`. Stack: Polars `pl.DataFrame` panels, plotly `go.Figure`, `arch` (GARCH), `hmmlearn.GaussianHMM` (optional), `gymnasium`, stable-baselines3, `pfhedge`, torch.

| Area | API |
|---|---|
| Data | `data.load_crypto_perps` (hourly BTCUSDT perps), `data.load_crypto_premium` (8-hourly premium index), `data.load_us_equities` (daily adjusted OHLCV) |
| Calibration | `CryptoMarketCalibrator(...)` -> `.load_data()`, `.compute_returns()`, `.fit_garch()` -> `GARCHParams` (alpha, beta, omega, unconditional_vol, `.persistence()`), `.fit_regimes(n_regimes=2, method="volatility"\|"hmm")` -> `RegimeParams(n_regimes, transition_matrix, regime_means, regime_stds)`, `.estimate_spread_depth()` -> (spread_normal, spread_stressed, depth_normal, depth_stressed), `.estimate_impact()` -> (permanent, temporary), `.get_execution_env_params()` -> `ExecutionEnvParams`, `.summary()` |
| Execution env | `rl_environments.ExecutionEnv(..., max_participation_rate=0.35, risk_aversion=0, schedule_penalty=0)`: `reset(seed)`, `step(action)`, `reference_trade_size()`, `max_trade_size(market)`, `action_to_target_shares(action)`, `target_shares_to_action(shares)`; info carries regime, spread, depth, shortfall; `MarketState` |
| Crypto env | `crypto_execution_env.CryptoExecutionEnv(..., max_participation_rate=0.10)`: same action API; info carries premium_index, hours_to_funding, forced flag; `CryptoMarketState` |
| Market making | `MarketMakingDynamics` (frozen dataclass), `fill_probability(distance, imbalance_factor, base_spread)`, `compute_quotes(price, vol, inventory, inventory_limit, skew_level, spread_mult, base_spread)` -> (reservation, centre, bid, ask, half_spread), `MarketMakingEnv(..., lambda_inventory=0.001)`, `encode_action(skew_idx, spread_idx)`, `run_episode(env, seed, model=None, fixed_action=None, vec_normalize=None)`, `paired_gap(baseline)` |
| Execution helpers | `discrete_almgren_chriss_schedule(total_shares, horizon, sigma_price, eta, gamma, risk_aversion)`, `ppo_execution`, `episode_diagnostics`, `add_execution_rate_band`, `comparison_row(reference, other)`, `run_episode(model, env)`, `make_env(seed)` |
| Deep hedging | `pfhedge.instruments.EuropeanOption` on Heston paths (`build_option(seed, n_paths)`), `pfhedge.nn.Hedger` with ES criterion (`build_deep_hedger()`), `hedging_pnl`, `hedging_pnl_torch`, `FromScratchHedger(hidden_size=32)`, `rollout`, `expected_shortfall_loss`, `qlbs_update(q_table, path, exploration)`, `state_index`, `pnl_stats` |
| IRL | `collect_expert_trajectories(env, policy_fn, n_trajectories, seed=42)`, `twap_policy(obs, env)`, `extract_features(state, action)`, `trajectory_feature_counts`, `build_candidate_bank(expert, env, n_candidates, seed)`, `proposal_action(obs, env, kind, rng)`, `prepare_maxent_problem`, `maxent_distribution`, `regularized_log_likelihood`, `maxent_irl`, `feature_expectation_irl`, `best_greedy_action`, `inferred_boltzmann_action(obs, weights, temperature=0.2)`, `evaluate_reward_policy`, `evaluate_policy`, `shortfall_bps`, `gap_to_expert` |
| Backtest | `from ml4t.backtest import BacktestConfig, DataFeed, Engine, ExecutionMode, Strategy`; `from ml4t.backtest.broker import Broker`; `from ml4t.backtest.execution.impact import NoImpact, SquareRootImpact`; `Strategy` subclasses implement `on_start(broker)` and `on_data(timestamp, data, context, broker)`; `MomentumStrategy`, `run_backtest`, `run_spectrum`, `formation_statistics`, `select_spectrum`, `flip_rate` |

SB3 training pattern (reusable):

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

venv = VecNormalize(DummyVecEnv([make_env(SEED)]), norm_obs=True, norm_reward=True)
model = PPO("MlpPolicy", venv, learning_rate=3e-4, ent_coef=..., seed=SEED,
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])))
model.learn(total_timesteps=TOTAL_TIMESTEPS)
venv.training = False; venv.norm_reward = False   # freeze stats before evaluation
```

Paired comparison pattern:

```python
import numpy as np
seeds = range(EVAL_SEED_BASE, EVAL_SEED_BASE + EVAL_EPISODES)
a = np.array([run_episode(make_env(s), s, model=model_a) for s in seeds])   # fresh env per seed
b = np.array([run_episode(make_env(s), s, fixed_action=baseline) for s in seeds])
diff = a - b
se_paired = diff.std(ddof=1) / np.sqrt(len(diff))
se_unpaired = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))     # print alongside
called = abs(diff.mean()) > RESOLUTION_SIGMA * se_paired                    # RESOLUTION_SIGMA = 3.0
```

Reward-timing pattern (`SimpleTradingEnv.step`): compute `reward = position_prev * r_t - cost * |action - position_prev|` in percent, then set `position = action` for the next step; never let the window include the bar being rewarded.

Recorded-bar panel pattern (notebook 04): `shift(1)` return, rolling vol, volume and premium index; keep open price and calendar unshifted; `assert train.max_ts < eval.min_ts` (or equivalent) before building episodes.

Whether notebooks persist models or results to disk is not stated in the notes.

## Evidence from the book

Numeric outputs (mean rewards, shortfall bps, wealth gaps, P&L stats, erosion percentages, flip rates) were stripped from the digest; only protocol-level and qualitative findings are available.

| Notebook | Finding |
|---|---|
| `01_algorithms_comparison` | Squared-return autocorrelation stays positive for tens of hours (to lag 48) in both BTCUSDT data and the GARCH sim. With zero-mean returns and 10 bps cost no policy has positive expected reward; DQN / PPO / A2C differ in position distribution and turnover, not reliably in mean reward over 10 episodes. |
| `02_optimal_execution_ppo` | Amihud-based impact estimate for BTC falls below the lower clamp, so the env uses the clamp (temporary 0.001, permanent 0.01; inference from clamp ranges). The hourly-range spread proxy makes every strategy overpay. Almgren-Chriss is an oracle; the honest reading is PPO vs TWAP on paired paths plus where the volume sat in the horizon. |
| `03_market_making_ppo` | An ordering is called only at 3 SE; with 240 episodes the paired gaps to the three reservation-price rules are reported with arm correlation. The learned contribution is skew and adaptive width only; the inventory response is shared by every arm. |
| `04_crypto_execution_rl` | Evaluation year 2024 is not a quiet corner of the 2022-2024 sample. Cost dispersion is dominated by price moves within the window (removed by pairing). Forced-liquidation rate and volume-location columns are reported per strategy; execution rate grouped by premium / funding is an association only. |
| `05_deep_hedging_pfhedge` | BS delta on Heston paths is misspecified; baselines receive realised vol per path. Deep hedgers minimise ES_0.05 directly; the from-scratch hedger constrains position to [0,1] (sigmoid), pfhedge does not. P&L stats not visible; results are device-dependent. |
| `06_inverse_reinforcement_learning` | TWAP demonstrations carry only two distinct actions, so inferred weights cannot identify a preference; both IRL methods produce illustrative, non-unique weights; BC degrades when driving. |
| `07_backtest_with_impact` | Erosion = no-impact minus c=0.6 return is pure impact; the cross-section need not be monotonic in participation (different trading paths / turnover); flip rate reported for liquid (<10% ADV) vs thin (>100% ADV) cohorts with attrition. Magnitudes not visible. |

## Related references

- `chapters/18_transaction_costs.md` — implementation shortfall, impact, spread as cost of immediacy; prerequisite for the execution and market-making recipes.
- `chapters/19_risk_management.md` — VaR and expected shortfall; prerequisite for the deep-hedging objective.
- `chapters/16_strategy_simulation.md` — backtest engine conventions reused by the impact study.
- `chapters/03_market_microstructure.md` — order books, adverse selection, participation rate, the mechanics behind fill models.
- `chapters/05_synthetic_data.md` — simulated paths and the "do not invent statistics" discipline behind GARCH calibration.
- `chapters/09_model_based_features.md` — GARCH and HMM regime fitting used by `CryptoMarketCalibrator`.
- `chapters/11_ml_pipeline.md` — split-by-episode and formation/evaluation separation as instances of general leakage control.
- `chapters/25_live_trading.md` — staged deployment and latency, the operational side of the sim-to-real gap.
- `chapters/26_mlops_governance.md` — governance, drift and monitoring for a deployed policy.
- `case_studies/crypto_perps_funding.md` — the BTCUSDT perps, premium index and funding data behind notebooks 01-04 and 06.
- `case_studies/us_equities_panel.md` — the daily US equities used by the impact backtest.
- `case_studies/sp500_options.md` — option data context for hedging comparisons.
- `case_studies/nasdaq100_microstructure.md` — microstructure evidence relevant to fill and spread assumptions.
- `libraries/ml4t_backtest.md` — `Engine`, `Broker`, `SquareRootImpact`, `Strategy` hooks used in notebook 07.
- `libraries/ml4t_data.md` — `load_crypto_perps`, `load_crypto_premium`, `load_us_equities`.
- `guardrails.md`, `decision_rules.md`, `evidence.md`, `glossary.md`, `companion_repo.md`, `workflow.md` — cross-cutting indexes that aggregate this chapter's rules.

Further reading:
- Almgren & Chriss (2001), Optimal execution of portfolio transactions.
- Avellaneda & Stoikov (2008), High-frequency trading in a limit order book.
- Buehler et al. (2019), Deep hedging; Halperin (2019), QLBS; Zheng et al. (2023), Option dynamic hedging using RL.
- Halperin et al. (2025), RL and inverse RL: a practitioner's guide for investment management; Dixon & Halperin (2020), G-Learner and GIRL.
- Nevmyvaka et al. (2006) and Hafsi & Vittori (2025), RL for optimal execution; Kearns & Nevmyvaka, ML for market microstructure.
- Kolm & Ritter (2019); Hambly et al. (2023); Sun et al. (2023); Millea (2021), critical survey of deep RL for trading.
- Mnih et al. (2015) DQN; van Hasselt et al. (2015) Double DQN; Wang et al. (2016) Dueling; Schulman et al. (2017) PPO; Haarnoja et al. (2018) SAC; Konda & Tsitsiklis (1999) and Sutton et al. (2000) actor-critic / policy gradient; Sutton & Barto (2018).
- Ziebart et al., MaxEnt IRL; Snoswell et al. (2020); Arora & Doshi (2021) IRL survey; Ho & Ermon (2016) GAIL; Chen et al. (2021) Decision Transformer; Byrd et al. (2020) ABIDES; Yang et al. (2015) GP strategy identification; Li et al. (2025) FlowHFT.

## Glossary

- **Implementation shortfall** — sum over fills of (arrival price - fill price) for a sale, in bps of arrival notional.
- **Arrival price** — price on screen when the execution decision was made.
- **Participation rate** — order quantity divided by available liquidity (depth or volume) in the step.
- **Temporary / permanent impact** — transient price concession from walking the book / lasting price shift from information in the flow.
- **TWAP** — equal quantity every step; conditions on nothing.
- **Almgren-Chriss** — static schedule minimising expected impact + lambda x variance; sinh decay with kappa^2 = lambda sigma^2 / eta.
- **Reservation price** — mid shifted against inventory, p (1 - (q/q_max) sigma).
- **Adverse selection** — being filled on the side the market is moving away from.
- **Order imbalance** — mean-reverting [-1,1] proxy for arriving-order pressure.
- **Marked wealth** — cash + inventory valued at next price.
- **Funding / premium index** — periodic (8h) payment between longs and shorts on perpetuals / the venue's perp-minus-spot fraction it is computed from.
- **Forced liquidation** — episode ending with an involuntary unwind on the last bar.
- **Self-financing P&L** — hedge gains minus proportional costs minus option payoff; no external cash flows after inception.
- **Expected shortfall (CVaR)** — negative mean P&L over the worst q fraction of paths.
- **Whalley-Wilmott** — small-cost asymptotic no-trade band around delta.
- **QLBS** — Halperin's Q-learner in the Black-Scholes world; here a coarse tabular variant.
- **Behaviour cloning** — supervised regression state -> action; no reward.
- **MaxEnt IRL** — reward weights maximising trajectory likelihood exp(w^T Phi)/Z.
- **Feature-expectation matching** — iterate w by mu_E - mu_pi without a partition function.
- **Partition function Z** — normaliser summing exp reward over all (here a fixed bank of) trajectories.
- **Distribution shift** — a clone acts in states its own errors create, which demonstrations never covered.
- **Volatility clustering** — positive autocorrelation of squared returns.
- **Paired standard error** — SE of per-episode differences on shared seeds.
- **Turnover** — units of position change over an episode.
- **Square-root impact law** — impact = c sigma sqrt(q/ADV) p.
- **Erosion** — no-impact return minus high-impact return.
- **Oracle benchmark** — a schedule computed with whole-episode information unavailable at decision time (Almgren-Chriss here).
- **Policy collapse** — a learned policy settling on one locally safe action (TWAP pace, widest spread, one position) for lack of exploration.
