# Reproduction vs paper

## Table 1 — Out-of-sample portfolio performance, 2021–2025

| Portfolio | Metric | Paper | Reproduced |
|---|---|---|---|
| RankModel_Top10 | total_return | 3.359 | 2.052 |
| RankModel_Top10 | CAGR | 34.24% | 25.00% |
| RankModel_Top10 | annual_return | 32.10% | 24.23% |
| RankModel_Top10 | volatility | 21.97% | 18.90% |
| RankModel_Top10 | sharpe | 1.279 | 1.071 |
| RankModel_Top10 | max_drawdown | -12.27% | -15.53% |
| RankModel_Top10 | excess_vs_spy | 17.24% | 9.38% |
| RankModel_Top10 | beta_vs_spy | 1.249 | 1.034 |
| RankModel_Top10 | spy_alpha | 13.55% | 8.89% |
| SPY | total_return | 0.987 | 0.986 |
| SPY | CAGR | 14.72% | 14.71% |
| SPY | annual_return | 14.85% | 14.85% |
| SPY | volatility | 14.54% | 14.54% |
| SPY | sharpe | 0.746 | 0.746 |
| SPY | max_drawdown | -20.47% | -20.47% |
| SPY | beta_vs_spy | 1.017 | 1.017 |
| SPY | spy_alpha | -0.25% | -0.25% |
| Bottom20_Predicted | total_return | 0.888 | 0.871 |
| Bottom20_Predicted | CAGR | 13.56% | 13.35% |
| Bottom20_Predicted | annual_return | 13.97% | 13.93% |
| Bottom20_Predicted | volatility | 15.68% | 16.51% |
| Bottom20_Predicted | sharpe | 0.636 | 0.602 |
| Bottom20_Predicted | max_drawdown | -19.14% | -27.08% |
| Bottom20_Predicted | excess_vs_spy | -0.88% | -0.91% |
| Bottom20_Predicted | beta_vs_spy | 0.779 | 0.960 |
| Bottom20_Predicted | spy_alpha | 2.41% | -0.32% |
| TopMinusBottom | total_return | 1.253 | 0.573 |
| TopMinusBottom | CAGR | 17.64% | 9.48% |
| TopMinusBottom | annual_return | 18.12% | 10.30% |
| TopMinusBottom | volatility | 19.36% | 15.82% |
| TopMinusBottom | sharpe | 0.729 | 0.398 |
| TopMinusBottom | max_drawdown | -15.33% | -16.63% |

## Table 2 — Ranking diagnostics

| Diagnostic | Paper | Reproduced |
|---|---|---|
| avg_top10_monthly_return | 2.67% | 2.02% |
| avg_bottom20_monthly_return | 1.16% | 1.16% |
| avg_top_minus_bottom | 1.51% | 0.86% |
| annualized_top_minus_bottom | 18.12% | 10.30% |
| top_minus_bottom_p | 0.041 | 0.151 |
| annualized_top10_excess_vs_random | 16.30% | 8.63% |
| top_minus_random_p | 0.011 | 0.067 |
| annualized_top10_excess_vs_spy | 17.24% | 9.38% |
| top_minus_spy_p | 0.005 | 0.080 |
| avg_spearman_ic | 0.0424 | 0.0311 |
| spearman_ic_p | 0.046 | 0.092 |
| pct_months_top_beats_spy | 66.67% | 56.67% |
| pct_positive_ic | 0.600 | 0.583 |

## Table 3 — Alternative models (CAGR / Sharpe / Top-SPY p)

| Model | Paper | Reproduced |
|---|---|---|
| XGBoost | 34.24% / 1.279 / 0.0050 | 25.00% / 1.071 / 0.0800 |
| RandomForest | 33.37% / 1.222 / 0.0092 | 27.48% / 1.129 / 0.0480 |
| LightGBM | 31.95% / 1.249 / 0.0062 | 28.70% / 1.224 / 0.0222 |
| ExtraTrees | 28.08% / 1.030 / 0.0684 | 29.46% / 1.050 / 0.0443 |
| Ridge | 15.54% / 0.678 / 0.8368 | 16.51% / 0.749 / 0.7129 |
| NeuralNetwork | 16.23% / 0.679 / 0.7010 | 30.49% / 1.203 / 0.0233 |

## Tables 10–11 — Regime-aware forecasting (CAGR / Sharpe / IC p)

| Specification | Paper | Reproduced |
|---|---|---|
| hmm_as_feature | 30.37% / 1.120 / 0.0433 | 24.27% / 1.091 / 0.1705 |
| one_model_per_regime | 23.44% / 0.905 / 0.3472 | 25.14% / 1.081 / 0.6593 |

## Table 14 — Portfolio optimization (CAGR / Sharpe / SPY alpha)

| Strategy | Paper | Reproduced |
|---|---|---|
| EqualWeight_Top10 | 34.24% / 1.279 / 13.55% | 25.00% / 1.071 / 8.89% |
| Static_MinVar_Top10 | 24.21% / 1.088 / 9.41% | 16.55% / 0.762 / 4.73% |
| Static_ScoreMV_Top10 | 32.16% / 1.074 / 10.81% | 27.17% / 1.091 / 12.75% |
| HMM_MinVar_Top10 | 26.54% / 1.173 / 11.31% | 17.90% / 0.820 / 5.55% |
| HMM_ScoreMV_Top10 | 32.91% / 1.095 / 11.62% | 25.57% / 1.056 / 11.89% |

## Table 20 — Transaction costs (CAGR / Sharpe / Top-SPY p)

Average monthly turnover: paper 148.10%, reproduced 151.29%

| Cost | Paper | Reproduced |
|---|---|---|
| 0 bps | 34.24% / 1.279 / 0.0050 | 25.00% / 1.071 / 0.0800 |
| 5 bps | 33.08% / 1.238 / 0.0076 | 23.89% / 1.022 / 0.1133 |
| 10 bps | 31.92% / 1.197 / 0.0113 | 22.78% / 0.973 / 0.1568 |
| 25 bps | 28.51% / 1.074 / 0.0347 | 19.52% / 0.828 / 0.3634 |
| 50 bps | 23.00% / 0.870 / 0.1638 | 14.26% / 0.586 / 0.9542 |

## Yearly returns (Top10 / SPY)

| Year | Top10 | SPY | Bottom20 |
|---|---|---|---|
| 2021 | 49.55% | 29.05% | 40.00% |
| 2022 | 8.99% | -16.23% | -24.97% |
| 2023 | 15.98% | 18.72% | 17.35% |
| 2024 | 26.89% | 29.56% | 12.07% |
| 2025 | 27.22% | 19.45% | 35.48% |
