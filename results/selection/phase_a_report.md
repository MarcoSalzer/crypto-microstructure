# Phase A — Correlation & Multicollinearity Diagnostics
*Thesis Section 3.4.2 input. Generated: 2026-05-18 14:22:36*

**Catalog**: `results/selection/feature_catalog.csv`  |  **Assets**: BTC, ETH

This report aggregates Phase A outputs from four diagnostics. Correlation and VIF results document the redundancy structure of the corpus but are **not** used to drop features — the reduction is performed exclusively in Section 3.4.3 via gradient-boosted feature importance.

---

## 1. Within-Concept Correlation

Correlation between variants of **the same** base_concept (i.e. between window/scope parametrisations of the same measure).

### 1.1 Quick Overview

| Metric                               | btc                                  | eth                                  |
| ------------------------------------ | ------------------------------------ | ------------------------------------ |
| Features in catalog                  | 1624                                 | 1624                                 |
| Features with valid pairs            | 1624                                 | 1624                                 |
| Base-concept groups                  | 372                                  | 372                                  |
| Total pairs computed                 | 3,673                                | 3,673                                |
| Mean |ρ| across all groups           | 0.3432                               | 0.3499                               |
| Median |ρ| across all groups         | 0.1778                               | 0.1847                               |
| Max |ρ| observed                     | 1.0000                               | 1.0000                               |
| Pairs |ρ|>0.85                       | 346                                  | 379                                  |
| Pairs |ρ|>0.95                       | 182                                  | 191                                  |
| Near-perfect pairs (>0.999)          | 153                                  | 153                                  |
| Groups fully redundant @0.95         | 31                                   | 31                                   |
| Drop candidates @0.95                | 86                                   | 92                                   |
| Drop candidates @0.85                | 196                                  | 209                                  |
| Retain rate @0.95 (%)                | 94.7                                 | 94.3                                 |
| Retain rate @0.85 (%)                | 87.9                                 | 87.1                                 |

### 1.2 Per-Asset Detail

### BTC

**Redundancy distribution across groups:**

- |ρ| > 0.95 in 100% of pairs: **31** groups  [█░░░░░░░░░░░░░░] 8.3%
- |ρ| > 0.95 in >50% of pairs: **0** groups  [░░░░░░░░░░░░░░░] 0.0%
- |ρ| > 0.95 in >0% of pairs: **15** groups  [█░░░░░░░░░░░░░░] 4.0%
- No pairs |ρ| > 0.95: **326** groups  [█████████████░░] 87.6%

**Top 10 most redundant base_concepts (by % pairs > 0.95):**

| base_concept | features | pairs | mean|ρ| | max|ρ| | %>0.95 |
| --- | --- | --- | --- | --- | --- |
| lwp_mid_struct50 | 8 | 28 | 1.000 | 1.000 | 100.0% |
| lwp_mid_struct100 | 8 | 28 | 1.000 | 1.000 | 100.0% |
| lwp_mid_5bps | 8 | 28 | 1.000 | 1.000 | 100.0% |
| lwp_mid_2bps | 2 | 1 | 1.000 | 1.000 | 100.0% |
| lwp_mid_10bps | 8 | 28 | 1.000 | 1.000 | 100.0% |
| lwp_bid_struct50 | 2 | 1 | 1.000 | 1.000 | 100.0% |
| lwp_bid_struct100 | 2 | 1 | 1.000 | 1.000 | 100.0% |
| lwp_bid_5bps | 2 | 1 | 1.000 | 1.000 | 100.0% |
| lwp_bid_2bps | 2 | 1 | 1.000 | 1.000 | 100.0% |
| lwp_bid_1bps | 2 | 1 | 1.000 | 1.000 | 100.0% |

**Top 10 least redundant base_concepts (lowest mean |ρ|, ≥2 features):**

| base_concept | features | pairs | mean|ρ| | max|ρ| | %>0.95 |
| --- | --- | --- | --- | --- | --- |
| liq_imb_sf_struct100 | 2 | 1 | 0.000 | 0.000 | 0.0% |
| z_net_cancel_5bps | 2 | 1 | 0.002 | 0.002 | 0.0% |
| cancel_rate_behind_1bps | 2 | 1 | 0.003 | 0.003 | 0.0% |
| cancel_rate_ahead_1bps | 2 | 1 | 0.003 | 0.003 | 0.0% |
| z_depth_gradient_struct50 | 8 | 28 | 0.010 | 0.062 | 0.0% |
| flow_depth_align_struct50 | 2 | 1 | 0.011 | 0.011 | 0.0% |
| depth_imbalance_struct50 | 2 | 1 | 0.012 | 0.012 | 0.0% |
| absorb_refill_ask_10bps | 2 | 1 | 0.022 | 0.022 | 0.0% |
| absorb_refill_ask_5bps | 2 | 1 | 0.028 | 0.028 | 0.0% |
| median_vacuum_score_10bps | 6 | 15 | 0.031 | 0.108 | 0.0% |

**Redundancy by axis (which dimension drives correlation):**

| differs_on | n pairs | mean|ρ| | median|ρ| | interpretation |
| --- | --- | --- | --- | --- |
| same_variant | 151 | 0.481 | 0.501 | identical parameterisation — pure duplicates |
| window_s | 1,339 | 0.432 | 0.414 | same concept, different window → time-scale redundancy |
| market_scope | 896 | 0.269 | 0.168 | same concept, fut vs spot → cross-market redundancy |
| market_scope,window_s | 1,287 | 0.165 | 0.045 |  |

**Near-perfect pairs (|ρ| > 0.999) — 153 pairs, possible duplicates:**

| feature_a | feature_b | ρ |
| --- | --- | --- |
| refill_vs_pull_spot_2bps_5s | refill_vs_pull_spot_2bps_60s | 1.00000 |
| net_pressure_logratio_5bps_10bps_fut_15s | net_pressure_logratio_5bps_10bps_fut_1s | 1.00000 |
| net_pressure_logratio_5bps_10bps_fut_15s | net_pressure_logratio_5bps_10bps_fut_60s | 1.00000 |
| net_pressure_logratio_5bps_10bps_fut_1s | net_pressure_logratio_5bps_10bps_fut_60s | 1.00000 |
| net_pressure_logratio_5bps_10bps_spot_15s | net_pressure_logratio_5bps_10bps_spot_60s | 1.00000 |
| net_pressure_logratio_5bps_10bps_spot_15s | net_pressure_logratio_5bps_10bps_spot_1s | 1.00000 |
| net_pressure_logratio_5bps_10bps_spot_1s | net_pressure_logratio_5bps_10bps_spot_60s | 1.00000 |
| net_pressure_logratio_2bps_5bps_fut_15s | net_pressure_logratio_2bps_5bps_fut_1s | 1.00000 |
| net_pressure_logratio_2bps_5bps_spot_15s | net_pressure_logratio_2bps_5bps_spot_60s | 1.00000 |
| net_pressure_logratio_2bps_5bps_fut_15s | net_pressure_logratio_2bps_5bps_fut_60s | 1.00000 |
| net_pressure_logratio_2bps_5bps_fut_1s | net_pressure_logratio_2bps_5bps_fut_60s | 1.00000 |
| net_pressure_logratio_2bps_5bps_spot_15s | net_pressure_logratio_2bps_5bps_spot_1s | 1.00000 |
| net_pressure_logratio_2bps_5bps_spot_1s | net_pressure_logratio_2bps_5bps_spot_60s | 1.00000 |
| refill_vs_pull_fut_1bps_15s | refill_vs_pull_fut_1bps_5s | 1.00000 |
| liq_imb_persist_sf_struct50_300s | liq_imb_persist_sf_struct50_900s | 1.00000 |
| liq_imb_sf_struct50_300s | liq_imb_sf_struct50_900s | 1.00000 |
| refill_vs_pull_fut_2bps_5s | refill_vs_pull_fut_2bps_60s | 1.00000 |
| refill_vs_pull_spot_1bps_15s | refill_vs_pull_spot_1bps_5s | 1.00000 |
| lwp_mid_5bps_fut_900s | lwp_mid_5bps_spot_900s | 1.00000 |
| lwp_mid_5bps_fut_300s | lwp_mid_5bps_spot_300s | 1.00000 |
| ... | *(133 more)* | |

**Drop candidate summary:**

- Threshold 0.95: drop **86** → retain **1538** (94.7%)
- Threshold 0.85: drop **196** → retain **1428** (87.9%)

Top 10 groups by drop count @0.95:

| base_concept | drops |
| --- | --- |
| lwp_mid_5bps | 7 |
| lwp_mid_10bps | 7 |
| lwp_mid_struct100 | 7 |
| lwp_mid_struct50 | 7 |
| net_pressure_logratio_2bps_5bps | 4 |
| median_taker_imbalance | 4 |
| net_pressure_logratio_5bps_10bps | 4 |
| l2_update_count_5bps | 3 |
| max_bps_ask | 2 |
| impact_per_signed_persist | 2 |

---

### ETH

**Redundancy distribution across groups:**

- |ρ| > 0.95 in 100% of pairs: **31** groups  [█░░░░░░░░░░░░░░] 8.3%
- |ρ| > 0.95 in >50% of pairs: **1** groups  [░░░░░░░░░░░░░░░] 0.3%
- |ρ| > 0.95 in >0% of pairs: **17** groups  [█░░░░░░░░░░░░░░] 4.6%
- No pairs |ρ| > 0.95: **323** groups  [█████████████░░] 86.8%

**Top 10 most redundant base_concepts (by % pairs > 0.95):**

| base_concept | features | pairs | mean|ρ| | max|ρ| | %>0.95 |
| --- | --- | --- | --- | --- | --- |
| lwp_mid_struct50 | 8 | 28 | 1.000 | 1.000 | 100.0% |
| lwp_mid_struct100 | 8 | 28 | 1.000 | 1.000 | 100.0% |
| lwp_mid_2bps | 2 | 1 | 1.000 | 1.000 | 100.0% |
| lwp_mid_1bps | 2 | 1 | 1.000 | 1.000 | 100.0% |
| lwp_mid_10bps | 8 | 28 | 1.000 | 1.000 | 100.0% |
| lwp_bid_struct50 | 2 | 1 | 1.000 | 1.000 | 100.0% |
| lwp_bid_struct100 | 2 | 1 | 1.000 | 1.000 | 100.0% |
| lwp_bid_5bps | 2 | 1 | 1.000 | 1.000 | 100.0% |
| lwp_bid_2bps | 2 | 1 | 1.000 | 1.000 | 100.0% |
| lwp_bid_1bps | 2 | 1 | 1.000 | 1.000 | 100.0% |

**Top 10 least redundant base_concepts (lowest mean |ρ|, ≥2 features):**

| base_concept | features | pairs | mean|ρ| | max|ρ| | %>0.95 |
| --- | --- | --- | --- | --- | --- |
| liq_imb_sf_struct100 | 2 | 1 | 0.000 | 0.000 | 0.0% |
| queue_pressure_log_5bps | 2 | 1 | 0.001 | 0.001 | 0.0% |
| cancel_rate_ahead_1bps | 2 | 1 | 0.002 | 0.002 | 0.0% |
| queue_imb_5bps | 2 | 1 | 0.002 | 0.002 | 0.0% |
| cancel_rate_behind_1bps | 2 | 1 | 0.004 | 0.004 | 0.0% |
| absorb_refill_ask_2bps | 2 | 1 | 0.018 | 0.018 | 0.0% |
| z_net_cancel_5bps | 2 | 1 | 0.021 | 0.021 | 0.0% |
| median_net_pressure_10bps | 6 | 15 | 0.027 | 0.090 | 0.0% |
| depth_notional_ask_1bps | 2 | 1 | 0.027 | 0.027 | 0.0% |
| median_net_pressure_5bps | 6 | 15 | 0.028 | 0.102 | 0.0% |

**Redundancy by axis (which dimension drives correlation):**

| differs_on | n pairs | mean|ρ| | median|ρ| | interpretation |
| --- | --- | --- | --- | --- |
| same_variant | 151 | 0.504 | 0.504 | identical parameterisation — pure duplicates |
| window_s | 1,339 | 0.445 | 0.432 | same concept, different window → time-scale redundancy |
| market_scope | 896 | 0.277 | 0.188 | same concept, fut vs spot → cross-market redundancy |
| market_scope,window_s | 1,287 | 0.182 | 0.049 |  |

**Near-perfect pairs (|ρ| > 0.999) — 153 pairs, possible duplicates:**

| feature_a | feature_b | ρ |
| --- | --- | --- |
| net_pressure_logratio_5bps_10bps_spot_1s | net_pressure_logratio_5bps_10bps_spot_60s | 1.00000 |
| net_pressure_logratio_2bps_5bps_fut_15s | net_pressure_logratio_2bps_5bps_fut_1s | 1.00000 |
| net_pressure_logratio_5bps_10bps_spot_15s | net_pressure_logratio_5bps_10bps_spot_1s | 1.00000 |
| net_pressure_logratio_5bps_10bps_spot_15s | net_pressure_logratio_5bps_10bps_spot_60s | 1.00000 |
| net_pressure_logratio_5bps_10bps_fut_1s | net_pressure_logratio_5bps_10bps_fut_60s | 1.00000 |
| net_pressure_logratio_2bps_5bps_fut_15s | net_pressure_logratio_2bps_5bps_fut_60s | 1.00000 |
| net_pressure_logratio_2bps_5bps_spot_15s | net_pressure_logratio_2bps_5bps_spot_60s | 1.00000 |
| net_pressure_logratio_5bps_10bps_fut_15s | net_pressure_logratio_5bps_10bps_fut_60s | 1.00000 |
| net_pressure_logratio_5bps_10bps_fut_15s | net_pressure_logratio_5bps_10bps_fut_1s | 1.00000 |
| net_pressure_logratio_2bps_5bps_spot_1s | net_pressure_logratio_2bps_5bps_spot_60s | 1.00000 |
| net_pressure_logratio_2bps_5bps_fut_1s | net_pressure_logratio_2bps_5bps_fut_60s | 1.00000 |
| net_pressure_logratio_2bps_5bps_spot_15s | net_pressure_logratio_2bps_5bps_spot_1s | 1.00000 |
| liq_imb_sf_struct50_300s | liq_imb_sf_struct50_900s | 1.00000 |
| liq_imb_persist_sf_struct50_300s | liq_imb_persist_sf_struct50_900s | 1.00000 |
| refill_vs_pull_spot_2bps_5s | refill_vs_pull_spot_2bps_60s | 1.00000 |
| refill_vs_pull_fut_2bps_5s | refill_vs_pull_fut_2bps_60s | 1.00000 |
| refill_vs_pull_spot_1bps_15s | refill_vs_pull_spot_1bps_5s | 1.00000 |
| refill_vs_pull_fut_1bps_15s | refill_vs_pull_fut_1bps_5s | 1.00000 |
| lwp_mid_5bps_fut_900s | lwp_mid_5bps_spot_900s | 1.00000 |
| lwp_mid_5bps_fut_300s | lwp_mid_5bps_spot_300s | 1.00000 |
| ... | *(133 more)* | |

**Drop candidate summary:**

- Threshold 0.95: drop **92** → retain **1532** (94.3%)
- Threshold 0.85: drop **209** → retain **1415** (87.1%)

Top 10 groups by drop count @0.95:

| base_concept | drops |
| --- | --- |
| lwp_mid_struct100 | 7 |
| lwp_mid_struct50 | 7 |
| lwp_mid_10bps | 7 |
| lwp_mid_5bps | 7 |
| net_pressure_logratio_2bps_5bps | 4 |
| net_pressure_logratio_5bps_10bps | 4 |
| median_taker_imbalance | 4 |
| max_bps_bid | 3 |
| max_bps_ask | 3 |
| l2_update_count_5bps | 2 |

---

### 1.3 Cross-Asset Comparison (within-concept)

**Global correlation level:**

- BTC mean |ρ|: 0.3432
- ETH mean |ρ|: 0.3499
- Delta: 0.0068 — **ETH has higher overall redundancy**

**Groups with largest BTC–ETH difference in mean |ρ| (top 15, BTC > ETH):**

| base_concept | btc_mean | eth_mean | Δ mean | btc_%>0.95 | eth_%>0.95 | Δ %0.95 |
| --- | --- | --- | --- | --- | --- | --- |
| aggressor_absorption_ratio_2bps | 0.725 | 0.514 | +0.210 | 0.0% | 0.0% | +0.0% |
| avg_trade_size | 0.429 | 0.302 | +0.127 | 0.0% | 0.0% | +0.0% |
| max_liq_distance_bid_struct100 | 0.172 | 0.055 | +0.117 | 0.0% | 0.0% | +0.0% |
| max_liq_distance_ask_struct100 | 0.161 | 0.057 | +0.105 | 0.0% | 0.0% | +0.0% |
| fill_rate_ahead_1bps | 0.789 | 0.691 | +0.098 | 0.0% | 0.0% | +0.0% |
| liq_concentration_ask_struct50 | 0.757 | 0.676 | +0.081 | 0.0% | 3.6% | -3.6% |
| mad_impact_per_signed | 0.548 | 0.470 | +0.079 | 0.0% | 0.0% | +0.0% |
| ask_churn_2bps | 0.473 | 0.397 | +0.076 | 0.0% | 0.0% | +0.0% |
| liq_concentration_struct50 | 0.411 | 0.336 | +0.076 | 3.6% | 0.0% | +3.6% |
| mad_queue_pressure_1bps | 0.416 | 0.345 | +0.072 | 0.0% | 0.0% | +0.0% |
| add_rate_ask_1bps | 0.129 | 0.058 | +0.071 | 0.0% | 0.0% | +0.0% |
| depth_notional_ask_5bps | 0.157 | 0.087 | +0.070 | 0.0% | 0.0% | +0.0% |
| depth_notional_ask_10bps | 0.205 | 0.142 | +0.063 | 0.0% | 0.0% | +0.0% |
| z_lwp_minus_mid_struct50 | 0.929 | 0.867 | +0.062 | 0.0% | 0.0% | +0.0% |
| d2_net_pressure_1bps | 0.281 | 0.221 | +0.060 | 0.0% | 0.0% | +0.0% |

**Groups with largest BTC–ETH difference (top 15, ETH > BTC):**

| base_concept | btc_mean | eth_mean | Δ mean | btc_%>0.95 | eth_%>0.95 | Δ %0.95 |
| --- | --- | --- | --- | --- | --- | --- |
| bps_sym | 0.189 | 0.660 | -0.471 | 0.0% | 0.0% | +0.0% |
| depth_gradient_struct50 | 0.100 | 0.423 | -0.323 | 3.6% | 0.0% | +3.6% |
| depth_gradient_bid_struct50 | 0.433 | 0.750 | -0.316 | 0.0% | 0.0% | +0.0% |
| depth_gradient_ask_struct50 | 0.436 | 0.735 | -0.299 | 0.0% | 0.0% | +0.0% |
| depth_imbalance_struct100 | 0.045 | 0.321 | -0.277 | 0.0% | 0.0% | +0.0% |
| liq_concentration_div_minus_struct50 | 0.269 | 0.484 | -0.215 | 0.0% | 0.0% | +0.0% |
| liq_concentration_div_minus_struct100 | 0.520 | 0.731 | -0.211 | 0.0% | 0.0% | +0.0% |
| depth_gradient_div_minus_struct50 | 0.323 | 0.531 | -0.208 | 16.7% | 0.0% | +16.7% |
| book_asymmetry_struct100 | 0.357 | 0.555 | -0.198 | 0.0% | 0.0% | +0.0% |
| liq_imb_struct100 | 0.360 | 0.548 | -0.188 | 0.0% | 0.0% | +0.0% |
| max_bps_bid | 0.473 | 0.628 | -0.155 | 10.7% | 21.4% | -10.7% |
| liq_concentration_struct100 | 0.341 | 0.492 | -0.150 | 0.0% | 0.0% | +0.0% |
| depth_imbalance_struct50 | 0.012 | 0.135 | -0.123 | 0.0% | 0.0% | +0.0% |
| participation_rate | 0.455 | 0.575 | -0.120 | 0.0% | 0.0% | +0.0% |
| depth_gradient_struct100 | 0.147 | 0.263 | -0.117 | 0.0% | 0.0% | +0.0% |

**Groups redundant in BOTH assets (>50% pairs > 0.95): 31 groups**

| base_concept | btc_%>0.95 | eth_%>0.95 |
| --- | --- | --- |
| l2_update_count_5bps | 100.0% | 66.7% |
| lwp_mid_struct100 | 100.0% | 100.0% |
| lwp_mid_struct50 | 100.0% | 100.0% |
| lwp_ask_struct100 | 100.0% | 100.0% |
| dist_to_day_low_bps | 100.0% | 100.0% |
| best_bid | 100.0% | 100.0% |
| mid | 100.0% | 100.0% |
| best_ask | 100.0% | 100.0% |
| dist_to_day_high_bps | 100.0% | 100.0% |
| lwp_mid_1bps | 100.0% | 100.0% |
| liq_imb_sf_struct50 | 100.0% | 100.0% |
| liq_imb_persist_sf_struct50 | 100.0% | 100.0% |
| lwp_mid_5bps | 100.0% | 100.0% |
| lwp_mid_2bps | 100.0% | 100.0% |
| lwp_bid_struct100 | 100.0% | 100.0% |
| lwp_bid_5bps | 100.0% | 100.0% |
| lwp_bid_2bps | 100.0% | 100.0% |
| lwp_bid_1bps | 100.0% | 100.0% |
| lwp_bid_10bps | 100.0% | 100.0% |
| lwp_ask_struct50 | 100.0% | 100.0% |

**Asset-specific redundancy (>50% for one, ≤20% for the other): 1 groups**

| base_concept | btc_%>0.95 | eth_%>0.95 | which asset |
| --- | --- | --- | --- |
| z_lwp_minus_mid_5bps | 0.0% | 100.0% | ETH |

**Drop candidate overlap (threshold 0.95):**

- Dropped in **both** assets: 79
- Dropped only in **BTC**: 7
- Dropped only in **ETH**: 13

Features dropped only in BTC @0.95 (first 20):

  - depth_gradient_div_fut_minus_spot_struct50_300s
  - depth_gradient_fut_struct50_300s
  - l2_update_count_fut_5bps_5s
  - l2_update_count_spot_5bps_1s
  - liq_concentration_fut_struct50_900s
  - liq_sum_spot_300s
  - queue_imb_persist_fut_1bps_900s

Features dropped only in ETH @0.95 (first 20):

  - basis_300s
  - depth_notional_ask_struct100_spot_1s
  - depth_notional_ask_struct50_spot_1s
  - depth_notional_bid_struct100_spot_300s
  - depth_notional_bid_struct100_spot_60s
  - l2_update_count_spot_5bps_5s
  - liq_concentration_ask_spot_struct50_300s
  - liq_concentration_bid_spot_struct50_300s
  - max_bps_ask_fut_1s
  - max_bps_bid_fut_1s
  - trade_count_fut_300s
  - trade_count_fut_900s
  - z_lwp_minus_mid_5bps_300s

---

## 2. Cross-Concept Correlation

Correlation **between** related base_concepts within semantic families. Each pair is annotated with `differs_on` — which axis(es) vary between the features (depth, stem, window, scope).

### 2.1 Per-Family Summary

**BTC** — 19 families:

| family | concept_pairs | pairs>0.70 | pairs>0.85 | pairs>0.95 | mean \|ρ\| | max \|ρ\| |
| --- | --- | --- | --- | --- | --- | --- |
| lwp_variants | 231 | 1644 | 1644 | 1644 | 0.9999 | 1.0000 |
| pull_rate_variants | 325 | 294 | 124 | 93 | 0.8451 | 1.0000 |
| net_pressure_variants | 780 | 247 | 131 | 60 | 0.8547 | 1.0000 |
| book_shape_imbalance | 153 | 203 | 59 | 42 | 0.8327 | 1.0000 |
| queue_pressure_variants | 630 | 351 | 157 | 42 | 0.8309 | 1.0000 |
| pull_vs_refill | 351 | 195 | 78 | 36 | 0.8323 | 1.0000 |
| depth_notional_variants | 231 | 150 | 35 | 13 | 0.8008 | 1.0000 |
| liq_concentration_variants | 45 | 99 | 33 | 10 | 0.8181 | 1.0000 |
| refill_rate_variants | 351 | 291 | 79 | 7 | 0.8002 | 0.9911 |
| taker_imbalance_variants | 10 | 14 | 6 | 6 | 0.8557 | 0.9880 |
| basis_features | 6 | 27 | 14 | 6 | 0.8854 | 1.0000 |
| absorption_variants | 190 | 23 | 6 | 3 | 0.8004 | 1.0000 |
| impact_variants | 10 | 18 | 7 | 3 | 0.8363 | 0.9826 |
| trade_activity | 21 | 22 | 6 | 2 | 0.8085 | 0.9775 |
| vacuum_churn | 15 | 3 | 0 | 0 | 0.7666 | 0.8265 |

**ETH** — 19 families:

| family | concept_pairs | pairs>0.70 | pairs>0.85 | pairs>0.95 | mean \|ρ\| | max \|ρ\| |
| --- | --- | --- | --- | --- | --- | --- |
| lwp_variants | 231 | 1644 | 1644 | 1644 | 0.9999 | 1.0000 |
| pull_rate_variants | 325 | 309 | 134 | 92 | 0.8451 | 1.0000 |
| net_pressure_variants | 780 | 164 | 80 | 64 | 0.8569 | 1.0000 |
| book_shape_imbalance | 153 | 235 | 81 | 42 | 0.8263 | 1.0000 |
| queue_pressure_variants | 630 | 340 | 150 | 42 | 0.8338 | 1.0000 |
| pull_vs_refill | 351 | 195 | 78 | 36 | 0.8330 | 1.0000 |
| refill_rate_variants | 351 | 321 | 86 | 8 | 0.7990 | 0.9955 |
| basis_features | 6 | 27 | 14 | 8 | 0.8781 | 1.0000 |
| taker_imbalance_variants | 10 | 16 | 9 | 6 | 0.8539 | 0.9870 |
| trade_activity | 21 | 13 | 5 | 4 | 0.8560 | 0.9875 |
| absorption_variants | 190 | 23 | 12 | 4 | 0.8522 | 1.0000 |
| depth_notional_variants | 231 | 115 | 23 | 2 | 0.7956 | 1.0000 |
| liq_concentration_variants | 45 | 18 | 2 | 0 | 0.7822 | 0.9329 |
| vacuum_churn | 15 | 2 | 0 | 0 | 0.7909 | 0.8397 |
| impact_variants | 10 | 12 | 3 | 0 | 0.7883 | 0.9132 |

### 2.2 Axis-Disaggregated Summary

Key table for 3.4.2: which redundancy actually arises per `differs_on` axis.

- `depth` → pure depth redundancy (same measure, different depth)
- `stem` → pure statistic-operator redundancy
- `window` → time-scale redundancy
- `scope` → spot↔futures redundancy
- Kombinationen (z.B. `depth+stem`) → gemischte Effekte

**BTC**:

| differs_on | n_pairs | >0.85 | >0.95 | weighted mean \|ρ\| | max \|ρ\| |
| --- | --- | --- | --- | --- | --- |
| `depth` | 597 | 406 | 261 | 0.8992 | 1.0000 |
| `stem+depth+window` | 336 | 240 | 240 | 0.9291 | 1.0000 |
| `stem+depth+window+scope` | 261 | 240 | 240 | 0.9787 | 1.0000 |
| `stem` | 356 | 246 | 198 | 0.9094 | 1.0000 |
| `depth+window` | 368 | 201 | 193 | 0.8845 | 1.0000 |
| `depth+window+scope` | 228 | 192 | 192 | 0.9610 | 1.0000 |
| `stem+depth+scope` | 215 | 186 | 180 | 0.9695 | 1.0000 |
| `stem+depth` | 436 | 251 | 180 | 0.8810 | 1.0000 |
| `depth+scope` | 212 | 170 | 126 | 0.9261 | 1.0000 |
| `stem+window` | 342 | 123 | 57 | 0.8289 | 1.0000 |
| `stem+window+scope` | 105 | 56 | 52 | 0.8784 | 1.0000 |
| `stem+scope` | 147 | 68 | 48 | 0.8766 | 1.0000 |

**ETH**:

| differs_on | n_pairs | >0.85 | >0.95 | weighted mean \|ρ\| | max \|ρ\| |
| --- | --- | --- | --- | --- | --- |
| `depth` | 558 | 379 | 261 | 0.9031 | 1.0000 |
| `stem+depth+window` | 343 | 242 | 240 | 0.9261 | 1.0000 |
| `stem+depth+window+scope` | 266 | 240 | 240 | 0.9748 | 1.0000 |
| `depth+window` | 379 | 203 | 193 | 0.8829 | 1.0000 |
| `depth+window+scope` | 242 | 192 | 192 | 0.9479 | 1.0000 |
| `stem` | 322 | 232 | 185 | 0.9152 | 1.0000 |
| `stem+depth` | 396 | 226 | 182 | 0.8844 | 1.0000 |
| `stem+depth+scope` | 230 | 189 | 180 | 0.9553 | 1.0000 |
| `depth+scope` | 209 | 172 | 126 | 0.9326 | 1.0000 |
| `stem+window` | 312 | 141 | 59 | 0.8469 | 1.0000 |
| `stem+window+scope` | 81 | 54 | 50 | 0.9140 | 1.0000 |
| `stem+scope` | 117 | 66 | 44 | 0.8754 | 1.0000 |

### 2.3 Cross-Asset Family Comparison

| family | btc_mean | eth_mean | Δ | btc_>0.95 | eth_>0.95 |
| --- | --- | --- | --- | --- | --- |
| lwp_variants | 0.9999 | 0.9999 | +0.0000 | 1644 | 1644 |
| basis_features | 0.8854 | 0.8781 | +0.0073 | 6 | 8 |
| taker_imbalance_variants | 0.8557 | 0.8539 | +0.0018 | 6 | 6 |
| net_pressure_variants | 0.8547 | 0.8569 | -0.0022 | 60 | 64 |
| pull_rate_variants | 0.8451 | 0.8451 | +0.0000 | 93 | 92 |
| impact_variants | 0.8363 | 0.7883 | +0.0480 | 3 | 0 |
| book_shape_imbalance | 0.8327 | 0.8263 | +0.0064 | 42 | 42 |
| pull_vs_refill | 0.8323 | 0.8330 | -0.0007 | 36 | 36 |
| queue_pressure_variants | 0.8309 | 0.8338 | -0.0029 | 42 | 42 |
| liq_concentration_variants | 0.8181 | 0.7822 | +0.0359 | 10 | 0 |
| trade_activity | 0.8085 | 0.8560 | -0.0475 | 2 | 4 |
| depth_notional_variants | 0.8008 | 0.7956 | +0.0052 | 13 | 2 |
| absorption_variants | 0.8004 | 0.8522 | -0.0518 | 3 | 4 |
| refill_rate_variants | 0.8002 | 0.7990 | +0.0012 | 7 | 8 |
| vacuum_churn | 0.7666 | 0.7909 | -0.0243 | 0 | 0 |

---

## 3. Cross-Asset (S6) Correlation

Diagnosis of the S6 cross-asset spread features: internal redundancy and overlap with the S5 single-asset sources.

### 3.1 S6 Intra-Correlation

2 S6 base_concept groups analyzed.

**Top 10 redundant S6 groups:**

| base_concept | features | mean \|ρ\| | max \|ρ\| | %>0.95 |
| --- | --- | --- | --- | --- |
| ca_lag_corr_btc_taker_lead_eth_ret | 3 | 0.6381 | 0.7322 | 0.0% |
| ca_lag_corr_eth_taker_lead_btc_ret | 3 | 0.6337 | 0.7321 | 0.0% |

- Fully redundant (100% of pairs >0.95): **0** groups
- Teilweise redundant: **0** groups
- Not redundant (no pairs >0.95): **2** groups

### 3.2 S6 ↔ S5 Cross-Correlation (per asset)

**BTC** — 2 families:

| family | n_pairs | >0.85 | >0.95 | mean \|ρ\| | max \|ρ\| |
| --- | --- | --- | --- | --- | --- |
| lead_lag | 132 | 0 | 0 | 0.0031 | 0.0266 |
| TOP_S5_DISCOVERY | 96 | 0 | 0 | 0.0082 | 0.0492 |

**ETH** — 2 families:

| family | n_pairs | >0.85 | >0.95 | mean \|ρ\| | max \|ρ\| |
| --- | --- | --- | --- | --- | --- |
| lead_lag | 132 | 0 | 0 | 0.0035 | 0.0353 |
| TOP_S5_DISCOVERY | 96 | 0 | 0 | 0.0069 | 0.0528 |

---

## 4. VIF Distribution

Variance Inflation Factor — complementary to the correlation analysis. VIF > 10 indicates problematic multicollinearity. Reporting is documentary (no drops are based on it).

### 4.1 VIF Tier Distribution

| asset | ≤5 | 5–10 | 10–50 | >50 | total |
| --- | --- | --- | --- | --- | --- |
| BTC | 36 (2.0%) | 128 (7.3%) | 413 (23.5%) | 1182 (67.2%) | 1759 |
| ETH | 44 (2.5%) | 124 (7.1%) | 413 (23.5%) | 1174 (66.9%) | 1755 |

### 4.2 High VIF (>10) by base_concept

**BTC** — top 10 base_concepts by max VIF:

| base_concept | n_features | mean VIF | max VIF |
| --- | --- | --- | --- |
| dist_to_fib_618_week_bps | 1 | 13102.58 | 13102.58 |
| dist_to_fib_500_week_bps | 1 | 12732.70 | 12732.70 |
| dist_to_fib_382_week_bps | 1 | 10039.48 | 10039.48 |
| dist_to_fib_786_week_bps | 1 | 9975.28 | 9975.28 |
| lwp_bid_10bps | 2 | 9885.85 | 9922.49 |
| lwp_bid_5bps | 2 | 9886.97 | 9922.28 |
| lwp_mid_10bps | 8 | 8380.16 | 9922.05 |
| best_ask | 2 | 9886.79 | 9921.98 |
| mid | 2 | 9886.79 | 9921.96 |
| best_bid | 2 | 9886.78 | 9921.94 |

**ETH** — top 10 base_concepts by max VIF:

| base_concept | n_features | mean VIF | max VIF |
| --- | --- | --- | --- |
| aggressor_absorption_ratio_5bps | 2 | 6722.72 | 10434.69 |
| lwp_mid_10bps | 8 | 8113.34 | 10031.89 |
| lwp_mid_struct100 | 8 | 8907.20 | 10016.81 |
| lwp_mid_struct50 | 8 | 8906.17 | 10002.56 |
| lwp_bid_10bps | 2 | 9764.72 | 9793.15 |
| lwp_mid_5bps | 8 | 7321.64 | 9793.04 |
| lwp_ask_5bps | 2 | 9764.77 | 9793.04 |
| best_ask | 2 | 9680.86 | 9793.02 |
| mid | 2 | 9680.79 | 9793.01 |
| best_bid | 2 | 9680.72 | 9792.99 |

### 4.3 BTC vs ETH VIF Distribution

- BTC mean VIF: 1965.70, median: 250.74, max: 13102.58
- ETH mean VIF: 1931.94, median: 247.15, max: 10434.69

---

## 5. Main Takeaways

### BTC

- Overall redundancy: **moderate** — selective redundancy, many groups diverse (mean |ρ| = 0.3432)
- Aggressive pruning @0.95: 86 features removable, 94.7% of the set remains
- Moderate pruning @0.85: 196 features removable, 87.9% of the set remains
- Redundanteste Gruppe: `lwp_mid_struct50` (100.0% der Paare > 0.95)
- Main redundancy axis: `same_variant` (mean |ρ| = 0.481)
- 153 near-perfect pairs (|ρ|>0.999) — check potential true duplicates

### ETH

- Overall redundancy: **moderate** — selective redundancy, many groups diverse (mean |ρ| = 0.3499)
- Aggressive pruning @0.95: 92 features removable, 94.3% of the set remains
- Moderate pruning @0.85: 209 features removable, 87.1% of the set remains
- Redundanteste Gruppe: `lwp_mid_struct50` (100.0% der Paare > 0.95)
- Main redundancy axis: `same_variant` (mean |ρ| = 0.504)
- 153 near-perfect pairs (|ρ|>0.999) — check potential true duplicates

### BTC vs ETH

- Both assets show a **very similar** redundancy structure (Δ mean |ρ| < 0.01)
- 31 groups are strongly redundant in **both assets** → safe drop candidates regardless of asset
- 1 group shows **asset-specific** redundancy → asset-separated drop decision recommended
- 79 features identified as drop candidates @0.95 by **both assets** → highest priority for removal

---
*Report generated by `phase_a_summary.py` at 2026-05-18 14:22:36*
