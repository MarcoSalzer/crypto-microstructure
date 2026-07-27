# Results

All figures are from the thesis and its appendices. The central finding:

> **The order book carries measurable short-horizon structure, but no configuration that trades with
> meaningful frequency is profitable net of realistic exchange fees. A statistically valid signal is
> not the same as a tradable edge.**

## 1. Baseline predictability decays within seconds (Ch. 4.1, App. C)

Out-of-sample R² of the LightGBM baseline on forward returns (BTC / ETH):

| Horizon | 1 s | 5 s | 15 s | 30 s | 60 s | 300 s | 900 s |
|---|---|---|---|---|---|---|---|
| **BTC R²** | 0.121 | 0.047 | 0.017 | 0.007 | 0.001 | −0.001 | −0.022 |
| **ETH R²** | 0.098 | 0.031 | 0.010 | 0.002 | −0.001 | −0.004 | 0.000 |

Directional accuracy falls the same way (LightGBM, BTC): 0.78 (1 s) → 0.59 (15 s) → 0.53 (60 s).
Model ordering is LightGBM > MLP > Ridge, as the tabular-data literature predicts. **Excursion** targets
(MFE/MAE) are an order of magnitude more predictable than returns (R² ≈ 0.17–0.19 at 15 s), which is why
they inform the take-profit / stop-loss logic later.

Regenerating the two baseline exhibits in this section — the R²-by-horizon figure and the Appendix C
benchmark table (Table C.3) — requires the external baseline metrics, which are produced on the full data
store and are not committed here; the numbers above are quoted directly from the thesis.

## 2. Fee-aware backtest of the baseline (Ch. 4.2)

Trading every prediction over its horizon nets ≈ **−10 bps per trade** across short horizons and both
assets — essentially the round-trip taker fee. The move you can predict reliably (1 s, R² ≈ 12 %) is far
too small to clear the fee; by the time the move is large enough (~30 s), predictive accuracy has
collapsed to noise. **A large move and a reliable prediction of it never occur at the same horizon.**

## 3. Breakout events and clustering (Ch. 4.3–4.4)

Breakouts are calibrated on window × threshold (retaining cells with ≥1,000 events; Table 5). The
**final clustering is 30 s / 30 bps / k = 8** (PCA 600 BTC, PCA 300 ETH). Internal validity is weak
(silhouette median ≈ 0.017), so configurations are fixed by task-relevant separation, not silhouette.

Two **reversal** states survive selection (one per asset). Conditional on **known** cluster membership,
their directional accuracy peaks at 30–60 s:

| | DA @60 s (membership known) | 95 % BCa CI |
|---|---|---|
| BTC 30 s / 30 bps | **0.73** | [0.66, 0.79] |
| ETH 30 s / 30 bps | **0.78** | [0.69, 0.84] |

But the **partition is unstable** (block ARI ≈ 0.28–0.31; multi-seed ARI ≈ 0.46–0.55), so a live system
cannot reliably recover the grouping. Measured **honestly out of sample** — membership assigned by a
rule fitted on past data alone — accuracy falls to **≈ 0.56 (BTC) / 0.52 (ETH)**, barely above chance
and no better than a direct LightGBM model on the same breakouts. *The signal in the grouping is real;
recovering the grouping is the difficulty.* The **0.73–0.78** figure is an **oracle upper bound** that
assumes membership is known ex post — a property of the evaluation, not an attainable live number.

The two states are genuinely **cross-asset**: ≈42 % of the BTC state is described by ETH features and
≈44 % of the ETH state by BTC features — the S6 cross-asset features entered because they were
informative, operationalising the BTC-leads-ETH structure at the order-book level.

## 4. Tradability — the edge is below the cost line (Ch. 4.5)

A fee-aware backtest of a direct entry model on the 30 s / 30 bps breakouts, across feature sets, exit
rules (fixed TP/SL vs a learned dynamic exit), and confidence gates (264 configs BTC / 276 ETH), returns
**no profitable configuration**; the best case still **loses 4.0 bps/trade (BTC) and 1.8 bps/trade (ETH)**
under the cheaper maker/taker cost. The retained clusters' favourable excursion at 60 s is only
**3.3 bps (BTC) / 4.1 bps (ETH)** — smaller than the round-trip cost (7 bps maker-entry/taker-exit,
10 bps taker/taker) *before any exit rule is applied*. For ETH the adverse excursion overtakes the
favourable one beyond 60 s, so holding longer widens the loss faster than the gain.

## 5. Conclusion

Recurring order-book **feature-combination states exist and are describable** (an overextended price that
reverses — for BTC, aggressive buying absorbed by a bid-heavy book; for ETH, overextension onto a
thinning book). Their directional edge is **real but conditional on a cluster membership that does not
survive an honest out-of-sample assignment**, and the underlying moves are **too small to clear the
fee**. The result sits squarely in the documented **prediction–profitability gap**: the states are real
and predictive, but not tradable at the cost boundary in this market.

These figures come from **cross-validation and backtest simulation**. The out-of-sample directional
accuracy (**≈ 0.52–0.56**) and the ex-post-membership oracle ceiling (**0.73–0.78**) are two distinct
properties of the evaluation and are reported as such.

*Reproducible artifacts in this repo:* `results/clustering/final/` holds the cluster memberships and
per-config detail for the **k = 8 / 30 s / 30 bps PCA sweep** (PCA 50/150/300/600, both assets — the
final configs are BTC PCA 600 and ETH PCA 300), which back Table 6, Figure 8 and Appendices D/E/F. The
full screening grid (336 requested configurations, 312 distinct after the partitions that collapse
when a sparse cell cannot fill k = 8 or k = 10 at 100 events per cluster are deduplicated) is
summarised in `results/clustering/grid_overview.csv` (one config per row; `is_final_retained` flags
the two retained states).

These clustering artifacts are committed as the provenance of the reported results rather than as
inputs for a fresh pipeline run. `grid_overview.csv` in particular is a read-only summary that no
committed script consumes. The `final/` set (the `.npz` memberships and per-config CSVs) is still
read by the reselection and BCa reproduction scripts (`reselect_analyze`, `cluster_confidence_intervals`,
`persistence_bca`) to re-derive the reported confidence intervals; the figure and table generators
that once consumed them were removed in the public-repo cleanup.
