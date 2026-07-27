# Methodology

This document follows the system from raw data to the final tradability test. Ground
truth is the thesis (*Feature-Combination States for Short-Horizon Prediction in
Cryptocurrency Market Microstructure*, NOVA IMS, 2026) and its appendices. Every number
below is taken from the thesis text; the few quantities that exist only in the code, not
in the thesis, are marked "(code)". Section numbers in parentheses refer to the thesis.

## 1. Data collection (3.1)

The collection layer captures every trade execution and the full order-book state for two
assets and persists them in a columnar format (code: `collection`).

**Sources and window.** Two assets, BTC/USDT and ETH/USDT, both on Binance, in spot and
USDT-margined perpetual futures. That is four venue streams, recorded over a continuous
three-month window from 16 February 2026, 03:00 UTC, to 16 May 2026.

**What is captured.** Two data types per venue. Trade data gives, per execution, the
price, the base-asset quantity, a monotonically increasing trade identifier, an event
timestamp at millisecond resolution, and a maker flag; the capture client decodes the
maker flag into an explicit aggressor side (buy or sell), the primitive on which every
later order-flow feature is built, and adds a local receive timestamp and a reconnect
marker. Order-book data is reconstructed locally: Binance sends incremental updates, so
the client fetches a REST snapshot of up to 1,000 levels per side on each connection,
applies the buffered updates, and an emitter samples a sorted, non-crossed state every
100 milliseconds. The fixed cadence yields a deterministic 36,000 order-book rows per
stream per hour, roughly 144,000 across the four streams; the trade rate varies with
activity. A trade row carries 16 fields, an order-book row 21.

**Tick constraint.** Measured on 300 randomly drawn hours per stream, the spread equals a
single tick in 99.9 % of order-book observations for BTC spot (tick 0.01 USDT), 99.8 % for
the BTC perpetual (tick 0.10 USDT), and 99.9 % for each ETH venue (tick 0.01 USDT). All
four streams are Large-Tick. The median spread is 0.001 bps on BTC spot, 0.014 bps on the
BTC perpetual, and 0.048 bps on both ETH venues, so the quoted spread is negligible against
the round-trip exchange fee, and the cost barrier of Section 4.5 is set by the fee.

**Architecture.** A single-process, fully asynchronous application on the asyncio event
loop, so all consumers read one identical stream. Eight adapter tasks run concurrently, one
per combination of asset, market type, and data type; the code is symbol-parametric, so
adding an asset is a one-line configuration change. Each adapter has a supervisor that
restarts it after a five-second delay; reconnection uses exponential backoff doubling to a
30 s ceiling plus jitter. Trades pass a rolling 4,096-entry deduplication buffer and
timestamp and well-formedness checks. Three order-book watchdogs run concurrently: a
no-data watchdog forces reconnection after eight seconds of silence, a validity gate
triggers a resync after three seconds without a valid snapshot, and a staleness guard
clamps exchange timestamps more than two seconds behind wall-clock time. A startup
preflight warns when the spot-futures basis exceeds 50 bps.

**Storage.** Apache Parquet with Zstandard compression, partitioned by UTC hour, two fixed
table schemas enforced at write time, and a two-phase atomic commit (write to a temporary
file, atomic rename at hour rollover), so a reader always sees a complete file or none.

### Three quality-control layers

Quality control is a methodological layer, not an operational afterthought, and it operates
at three distinct scopes.

1. **Raw-stream audit (3.1.4).** Validates the recorded streams at two scales. At the file
   level it computes row count and temporal coverage, reconnection events, gaps beyond three
   seconds, per-venue throughput, and latency at the mean and the 50th, 90th, and 99th
   percentiles; trade files are additionally characterised by aggressor-side distribution,
   price range, and notional volume, and order-book files by spread in bps, crossed-book
   counts, depth statistics, and depth shortfalls. Within each file it partitions the hour
   into 120-second segments (segments under 50 rows are skipped as a data-sufficiency floor)
   and applies eight checks per segment: gaps beyond three seconds, mean spread above 50 bps,
   mean latency above one second, any crossed book, a depth-shortfall proportion above 10 %
   of rows, any reconnection, all required venues present, and non-zero throughput per venue.
   The local receive timestamp is the canonical time axis. This layer is for record-keeping
   and inspection. The hour-level completeness scan (whole raw hours missing, or hours with
   fewer than the four required trades/order-book streams) is `etl/audit/audit_raw_gaps.py`.
2. **Per-second usability gate (3.2.2).** The per-bucket admission decision. A one-second
   bucket is usable if and only if its trailing 60-second window is at least 95 % soft-healthy
   with no run of more than five consecutive unhealthy seconds. This gate, not the raw-stream
   audit, decides which observations enter the training corpus.
3. **Feature-corpus audit (3.3).** After the features are generated, the whole corpus is
   verified column by column, exhaustively rather than by sampling. The central principle is
   that missingness is informative: each feature is judged against the missingness expected
   for its own construction (rolling-window warm-up, the zero-MAD guard on robust scores,
   forward-target edges, inherited missingness) rather than against a single global tolerance,
   and consistency relations between related columns are checked (for example, a dispersion
   measure cannot be defined more often than the median it is built on). This audit is
   implemented by the standalone tools in `etl/audit/`: `audit_s0_to_s5_features.py` runs it
   over the merged corpus, classifying every column into its originating stage (S0–S5) and
   applying that stage's rules, with `--stage` to narrow the run to a single stage;
   `audit_s6_features.py` is the sole auditor of the cross-asset S6 columns, which the merged
   audit does not cover. They read the external feature store and are run on demand
   (`python -m etl.audit.<name>`), not as part of `etl.run_all`.

## 2. Feature engineering, S0 to S6 (3.2)

The pipeline turns the raw inputs into a one-second feature corpus in seven sequential
stages, each writing a Parquet file per asset and hour that feeds the next (code: `etl`,
a declarative spec / operators / engine design). Construction is causal: each value depends
only on data up to its own one-second bucket, with no look-ahead. S0 to S5 are computed
independently per asset; S6 is the only stage that combines the two per-asset streams. A
per-asset concept occupies two columns (BTC and ETH); a cross-asset concept occupies one.

Between S0 and S1 a precompute step materialises the chart-geometry reference levels from the
completed S0 hours — running daily OHLC plus weekly, monthly and volume-profile levels
(`etl/precompute_levels.py`, run as the `levels` stage of `run_all`, ordered
`ohlc` → `levels` → `s1`). These supply the reference levels the S1 distance-to-level,
range-position and volume-profile families read — week_*, prev_week_*, month_*, prev_month_*,
POC/VAH/VAL, price_vs_va and the fibonacci levels — which are otherwise undefined.

The corpus comprises 1,871 feature definitions across 3,644 columns, plus 38 prediction
targets (33 single-asset and 5 cross-asset, 71 columns) and 107 metadata definitions
(214 columns). The per-stage breakdown:

| Stage | Purpose | Features | Notes from the thesis |
|---|---|---|---|
| S0 | Atomic per-bucket state and reference levels | 146 (112 atomic market, 34 reference levels; 30 context fields are metadata) | Price (46), Activity (6), Aggression (12), Bookshape (44), Imbalance (4). Fixed-bps windows at 1, 2, 5, 10 bps and structural windows (struct50, struct100). Absolute price levels and reference levels are retained only as construction inputs. |
| S1 | Derived one-second features, rolling windows, forward targets | 379 (758 columns), 21 targets, 12 metadata | 15 families over rolling windows from 5 s to 1 h. EMA state is carried across hour boundaries. Targets are strictly forward-looking and filtered out of the feature matrix before any model is fit. |
| S2 | Rolling aggregations and dynamics | 615 (largest stage), 16 metadata, 10 targets | 11 families at window sizes 5, 15, 60, 300, 900 s. New at S2: dynamics (277, first and second temporal differences) and impact (20, per-unit-flow price impact). |
| S3 | Composite analytics | 326, 31 metadata, 2 targets | 7 families; dynamics (101), normalisation (76), pressure (72). Robust z-scores clipped to [-20, +20], shock scores to [-50, +50], with a missing value on zero MAD. |
| S4 | Advanced derived features | 236, 18 metadata | 4 families; dynamics (134), cross-market (14), normalisation (34), pressure (54). Depth-geometry diagnostics (slope, curvature, coherence) are metadata. |
| S5 | Signal quality | 58 | 3 families; dynamics (30, three-level shock pipeline), pressure (24, directional persistence bounded in [0, 1]), cross-market (4). Closes the per-asset pipeline. |
| S6 | Cross-asset features | 112 (126 columns), 5 cross-asset targets | The two per-asset streams are inner-joined on the one-second grid; a bucket where either asset fails its usability gate is excluded. 14 per-asset intermediaries (28 columns) plus 98 cross-asset features. Convention BTC minus ETH. Patterns: direct differences, AND and XOR regime-alignment flags, lead-lag cross-correlations at lags of 1, 3, 5 s in both directions, and forward-return spreads at five horizons. |

## 3. Feature reduction: three diagnostics (3.4)

Reduction runs three diagnostics. Two of them only annotate; correlation removes one family
and the importance analysis removes weak features (code: `selection`).

### 3.1 Pairwise correlation (3.4.1)

Pearson correlation is computed separately for BTC and ETH and then combined. Within a base
concept, differences in the lookback window produce the strongest correlation (about 0.43 to
0.45 in absolute terms), differences in market scope are weaker (about 0.27 to 0.28), and
pairs differing on both axes fall to roughly 0.16 to 0.18. A greedy procedure flags a pair
as redundant when its absolute correlation exceeds 0.95 and iteratively removes the feature
in the most such pairs; taking the union of the BTC and ETH flag sets gives 99 intra-concept
redundant features. A second analysis groups related concepts into 19 quantity groups and
flags 274 cross-concept redundant features by the same union rule.

With one exception these flags are annotations, not deletions, because two features that
normally move together can decouple under stress, and that decoupling is itself a signal.
The exception is the level-weighted price family, the only hard removal at the catalogue
level: it collapses onto the mid price at an average absolute correlation of 0.9999 (even
its most parameter-distant pair, 1 s against 900 s, stays above 0.9996), so no decoupling
behaviour remains. Nine features per asset are retained (two level markers plus the seven
normalised level-weighted-price-minus-mid variants); the other 58 per asset, 116 columns,
are removed at the catalogue level. The six S6 lag-correlation features were separately
checked: their largest pairwise correlation is 0.73 (3 s against 5 s), none crosses 0.95, and
against two reference sets of S5 features the maximum absolute correlation is 0.053 with every
mean below 0.01, so they carry cross-asset information the single-asset space does not.

### 3.2 Variance inflation factor (3.4.2)

The VIF measures how well one feature can be reconstructed from all the others at once, the
diagnostic for linear-coefficient stability. It is computed per asset — a separate VIF run for BTC and for ETH, each on
roughly 500,000 rows — over 1,759 linear-candidate features, and the two per-asset results
are combined by union-min (the smaller of the two factors is kept), so a feature is treated
as linear-stable if either asset's VIF is low, reflecting that linear stability is a property
of how a feature is constructed, not of the asset. About 3 % of features fall below a factor of 5
and about 13 % below 10; 87 % exceed 10 and close to two thirds exceed 50, with a median near
169, a mean near 1,760, and a tail beyond 13,000. The information content is therefore far
lower-dimensional than the nominal count. A feature is admitted to the linear-model profile
only when its factor lies below 10, which admits 209 features, materialised as 363 columns.

### 3.3 Feature importance with a null baseline (3.4.4)

The last diagnostic asks which surviving features carry predictive signal, using LightGBM
gain against a null-importance baseline. LightGBM regressors are trained on the full surviving
corpus of 3,528 columns, run in parallel for the two assets (once with BTC targets, once with
ETH targets, the same inputs available to both). Sixteen targets are evaluated per asset:
eight forward log-returns (1, 5, 15, 30, 60, 120, 300, 900 s) and Maximum Favourable and
Maximum Adverse Excursion at 15, 60, 300, 900 s. Excursions are included so that exit-timing
features are not discarded merely for failing to predict end-of-horizon direction.

Two signals are computed per feature, per fold: gain (LightGBM's own reduction in training
loss across splits that use the feature) and a null baseline (a second model trained on a
randomly permuted copy of the target, whose attributed importance is spurious by
construction). A feature's real gain compared against its null gain is its signal-to-noise
ratio. Importance is estimated under a five-fold expanding-window cross-validation with three
seeds, averaged, on a stratified sample of 1.5 million rows per asset spanning the whole
history. **Permutation importance is deliberately not used**: shuffling a feature is harmless
to the model precisely when a correlated neighbour still carries its information, which would
wrongly mark informative-but-redundant features as unimportant; the null baseline avoids that
by raising the noise floor uniformly.

The drop criterion is per asset, not global: a feature is weak in an asset when its averaged
gain-over-null fails to exceed the null floor on every one of the 16 targets in that asset,
and clearing the floor on even one target keeps it. A feature weak in both assets is removed
entirely; one weak in a single asset loses only that asset's column. The procedure repeats on
the reduced set until a round drops less than 5 % of its input. The same pass also marks the
strongest features with three top-decile annotations (top_returns, top_mfe, top_mae), which do
not affect the drop decision.

The procedure converged in two rounds:

| Round | Input columns | Universally weak | Surviving | Drop rate |
|---|---|---|---|---|
| 1 | 3,528 | 143 | 3,385 | 4.1 % |
| 2 | 3,385 | 38 | 3,347 | 1.1 % |

The 181 removed columns break down as 40 features weak in both markets (removed entirely),
92 weak in one market (dropped for that asset only), and 9 cross-asset features. The largest
block is the pull-rate and refill-rate family (67 columns); most of the remainder is
low-frequency chart-geometry context (weekly and monthly levels, and the fibonacci-level
family, which loses 8 of its 12 columns). No microstructure concept is eliminated in full.

In code, these importance scores are produced by `selection/s5_s6_feature_importance.py`
(per-asset, per-target LightGBM gain against a null baseline over the merged S5+S6 corpus).
The drop list is then built from them: `selection/aggregate_fi_results.py` reduces the
per-target scores to the universal-weakness candidates (`fi_drop_candidates.csv`),
`selection/merge_fi_drops.py` folds each round into `consolidated_drop_list.csv`, and
`selection/build_feature_keep.py` combines that with the correlation and VIF drops to emit
`feature_keep.csv`.

### Model-family profiles and the final set (3.4.3, 3.5)

The diagnostics feed three model-family profiles (code: the `use_tree`, `use_linear`, and
`use_cluster` flags in `feature_keep.csv`):

- **Tree / neural profile:** the full surviving set except the 77 absolute price levels,
  3,270 columns. Trees and neural networks are not destabilised by collinearity.
- **Linear profile:** the 363 columns with pooled VIF below 10.
- **Clustering profile:** excludes the intra-concept and cross-concept redundant features.
  The two flag sets overlap, so their union is 258 features; together with 8 features whose
  asset scope narrows through the importance drops, 266 features (524 columns) are withheld,
  leaving the 2,746-column clustering pool.

Across both layers, 297 columns are removed from the 3,644 signal columns (116 for the
level-weighted price family, 181 from the two importance rounds), leaving 3,347 retained
feature columns. Of these, 3,270 are model inputs and 77 are absolute price and level columns
retained only to construct the relative features. With 214 metadata and 71 target columns, the
final dataset has 3,632 columns, out of the 3,929-column corpus (3,644 signal, 214 metadata,
71 targets).

The 77 absolute levels are excluded from every profile as a design choice, independent of the
importance analysis: an absolute level (a moving average, a previous day's high) carries little
that transfers across regimes, since its magnitude drifts with the overall price level rather
than with market structure, so only its relative derivations reach the models.

## 4. Scaling (3.6)

Scaling applies to the clustering and linear profiles only; tree models are invariant to
monotone rescaling. Most of the corpus already arrives in relative or bounded units (bps
distances, ratios, robust z-scores, imbalances, range positions) and needs only centring.
About 435 retained columns are raw non-negative quantities or rates that were never
normalised at source, and these are treated in two steps because one transform cannot fix
both problems. First, a log-of-one-plus transform compresses the strongly right-skewed upper
tail onto a multiplicative scale while remaining defined at zero; it is deliberately not
applied to ratios, scores, or signed quantities. Second, standardisation centres and scales
every feature entering the two profiles to unit variance, with the statistics estimated only
on the training portion of each cross-validation split and applied unchanged to the held-out
portion.

## 5. Baselines, cross-validation, and the feature signature (4.1)

Three baselines map the predictive content across model classes before any strategy work:
Ridge on the 363-column linear profile, LightGBM on the 3,270-column tree profile, and a
PyTorch MLP (hidden layers of 512, 256, 128 units) on the tree profile. Sixteen targets are
predicted per asset (the eight returns and the MFE and MAE at four horizons). After discarding
the final forward-incomputable rows (1.3 % to 2.3 % of the sample depending on horizon), the
effective sample is roughly 6.87 million observations per (asset, target) pair; the residual
4.12 % feature-level missingness is imputed inside each model pipeline.

**Cross-validation design.** Evaluation uses expanding-window cross-validation, not random
k-fold, to keep training strictly in the past (code: `common/cv_engine.py`). The sample is
divided into six contiguous, chronologically ordered blocks; fold k trains on blocks 0 through
k and tests on block k+1, giving five folds whose training window grows from roughly
1.15 million rows to roughly 5.73 million, each test block used exactly once. Every fold is
trained with three seeds (42, 123, 999) and metrics are averaged over folds and seeds. Every
fitted transform (median imputation, standardisation, and for the MLP the target
standardisation) is fitted on the training fold alone and applied unchanged to the test fold;
LightGBM's early-stopping split is taken from within the training fold. The engine is
model-agnostic, so any performance difference is attributable to the model, not the procedure.
Quality is read from R² and the Information Coefficient (Spearman rank correlation). Ridge
selects its regularisation strength per fold on a held-out tail of the training fold and fits
on a 400,000-row subsample; the MLP fits on a 500,000-row subsample; LightGBM trains on up to
5.73 million rows with default hyperparameters.

**Results.** R² declines monotonically with the horizon for every model and both assets.
LightGBM on BTC falls from 0.121 at 1 s to 0.047 at 5 s, 0.017 at 15 s, and 0.007 at 30 s; it
is negligible from 60 s (0.0008 at 60 s, 0.0007 at 120 s, -0.0005 at 300 s) and clearly
negative only at 15 minutes (-0.022). Ridge traces the same curve one level lower (0.030 at
1 s). The Information Coefficient falls from 0.40 at 1 s to 0.05 at 60 s. Directional accuracy
on BTC falls from 0.78 at 1 s to 0.59 at 15 s and 0.53 at 60 s, and on ETH from 0.74 to 0.57
and 0.52. The models order as LightGBM > MLP > Ridge (0.121, 0.100, 0.030 on the BTC 1 s
return), locating much of the predictable structure in feature interactions a linear model
cannot represent. BTC is more predictable than ETH at short horizons (0.121 against 0.098 at
1 s), and excursion targets are an order of magnitude more predictable than returns (LightGBM
reaches about 0.175 on the 15 s favourable excursion and 0.185 on the 15 s adverse excursion).

**Feature signature (4.1.5).** Mapping LightGBM gain importance to concept family and lookback
window shows what the edge is read from and how that changes as it fades. At 1 s the signal is
the current book: the strongest BTC features are the futures-spot basis, the futures deviation
from its one-second VWAP, and the queue imbalance within one basis point of the touch;
one-second features carry 44 % of importance and no slower window more than 14 %. As the horizon
grows, the immediate families (Pressure, Aggression, Liquidity Events, Absorption) fall away
and structural ones (Bookshape, Volume Profile, Impact, Cross-Market) rise. Dynamics is the one
non-monotone case, widening from 19 % at 1 s to 25.5 % at 5 s. The queue imbalance at the touch
is the only feature among the strongest at every horizon: at short horizons as its
instantaneous value, at 30 and 60 s joined by its 900-second persistence.

## 6. Fee-aware prediction backtest (4.2)

The question turns economic: does the signal survive costs? A threshold rule acts on the
continuous LightGBM prediction stream (long above a decision threshold, short below its
negative, flat otherwise), with the threshold swept from 0 to 5 bps. Each trade is held for
the fixed forecast horizon and closed at its end, with no stop, take-profit, or path exit; its
profit is the realised signed forward return minus cost. The cost is the Binance taker fee,
5 bps on entry and 5 on exit, 10 bps round trip.

The result is unambiguous. At a threshold of zero, net profit per trade clusters around minus
10 bps at every short horizon on both assets (between roughly minus 9.7 and minus 10.6 bps):
the model captures almost no edge before costs, and the fee consumes what little there is. The
short horizons fail for two opposite reasons. At 1 s the model is accurate (R² near 12 %) but
the 99th percentile of the absolute one-second BTC return is under 3 bps, so a move large
enough to clear a 10 bps cost almost never arises; by 30 s the moves are larger (99th
percentile about 16 bps) but accuracy has collapsed below 1 % R². A large move and a reliable
prediction of it never occur at the same horizon. Raising the threshold does not lift any
horizon above zero at a meaningful trade count; the single positive cell (BTC, 30 s, highest
threshold) rests on about 10 trades per fold, which is noise.

## 7. Breakout definition and grid calibration (4.3)

If the typical move is too small, the response is to trade fewer moves, selected where a large
move is structurally likely (volatility clustering). A breakout is an interval over which the
absolute move of the futures mid-price exceeds a fixed bps threshold, the same instrument on
which the cost model is applied. Window length and threshold must be matched, since the typical
move grows with the window. Two requirements govern which combinations are retained: no
threshold below the 10 bps round-trip fee, and at least 1,000 events per combination. Four
windows (1, 5, 15, 30 s) are examined at thresholds of 10, 15, 20, 30, 40 bps, for both assets.
The full-window counts:

| Asset / Window | 10 bps | 15 bps | 20 bps | 30 bps | 40 bps |
|---|---|---|---|---|---|
| BTC / 1 s | 1,365 | | | | |
| BTC / 5 s | 16,538 | 3,767 | 1,357 | | |
| BTC / 15 s | 97,452 | 26,366 | 9,576 | 2,147 | |
| BTC / 30 s | 254,933 | 81,311 | 32,425 | 7,751 | 2,459 |
| ETH / 1 s | 4,056 | 1,326 | | | |
| ETH / 5 s | 40,658 | 11,573 | 4,494 | 1,082 | |
| ETH / 15 s | 189,748 | 62,629 | 26,566 | 7,061 | 2,316 |
| ETH / 30 s | 433,493 | 164,833 | 76,046 | 22,451 | 8,105 |

The viable combinations form a triangle in the lower left: the count rises with the window and
falls as the threshold rises. Each asset is carried on every combination it populates, so ETH
contributes three cells BTC does not (1 s at 15 bps, 5 s at 30, 15 s at 40). Screening and
clustering then run on a seed-fixed random subsample of 1,000 hourly files (roughly half the
available hours), on which the counts are roughly halved, so a 500-event screening floor
applies to the subsample counts.

## 8. Clustering (4.4)

### Dimensionality reduction (4.4.1)

Each breakout event is described by the 2,746-column clustering profile. Because distances lose
discriminating power in high dimensions, PCA is applied first: missing entries are median-filled
and the features standardised (neither PCA nor k-means accepts missing values), then the
rotation produces uncorrelated axes ordered by variance with no event lost. The number of
retained components is varied across 50, 150, 300, and 600, the upper end already capturing
close to 90 % of the variance, so that the effect of the representation can be tested directly.

### Method comparison (4.4.2)

Three methodologically distinct methods are compared on equal footing (identical breakout
definition, feature set, and PCA range) on a single representative configuration (ETH, 5 s,
15 bps): k-means, a Gaussian mixture model, and density-based HDBSCAN. The verdict follows
from the high intrinsic dimensionality of the event description. HDBSCAN fails outright,
labelling all events as noise at every dimensionality, because density estimation needs a
low-dimensional space. GMM separates events about as well as k-means but, at the dimensionality
the data demands, can afford only a diagonal covariance matrix, discarding the feature
correlations that were its one potential advantage. K-means is selected: it matches GMM at a
fraction of the complexity and scales cleanly. The remaining analysis builds on k-means.

### Configuration search (4.4.3)

The clustering depends on four axes: the trailing-move horizon (1, 5, 15, 30 s), the threshold
(10, 15, 20, 30, 40 bps), the PCA dimensionality (50, 150, 300, 600), and the cluster count k
(6, 8, 10). A fifth parameter, the lookback (1, 5 s), enters only the real-time classifier of
Section 4.4.5. A region is screened only if the calibration sample holds at least 500 events,
which excludes the ETH 5 s / 30 bps cell and leaves 15 of 16 cells, so the grid requests 336
configurations in total (180 for ETH and 156 for BTC). Clusters under 20 events are ignored during screening,
and the viability criterion requires at least 100 events per cluster; in sparse regions the
100-event minimum can make the requested k=8 or k=10 collapse to the same partition as k=6.
After counting such duplicates once, 140 distinct BTC and 172 distinct ETH configurations
remain, so 312 of the 336 requested configurations are distinct.

Separation is measured per cluster over the 60 s window after the breakout by the MFE/MAE ratio
(mean favourable excursion divided by mean adverse excursion; the primary measure) and the
MFE-lift (the cluster's favourable move divided by that of all breakouts; a secondary check).
Most clusters resolve as reversal rather than continuation (757 of 984 for BTC, 928 of 1,171
for ETH). The best ratios reach about 3.9 (BTC, 30 s, 40 bps) and 3.6 (BTC, 30 s, 30 bps);
ETH's highest anywhere is 3.3 (15 s, 40 bps).

**Selection effect, permutation null, BH-FDR.** Because the screening selects the best cluster
of each configuration by the same statistic that ranks configurations, the headline ratios are
a maximum over 312 cells and inflated even where no real asymmetry exists. Each configuration's
best-cluster ratio is therefore tested against a permutation null in which the cluster labels
are shuffled while their sizes are held fixed, and the per-configuration p-values are corrected
across the grid with the Benjamini-Hochberg procedure to control the false discovery rate. A
real but limited core survives: 72 of the 312 configurations clear a 10 % false discovery rate,
39 of 140 for BTC (median permutation p of 0.09) against 33 of 172 for ETH (median 0.19), so
the BTC asymmetry is genuine across more than a quarter of its grid while for ETH only a
minority is distinguishable from noise.

**PCA sweep and the silhouette argument.** The PCA dimensionality has only a weak effect (the
best ratio moves by a median of about 0.09 across 50 to 600 within a fixed horizon, threshold,
and k), because k-means clusters along the dominant variance directions already captured by
roughly 50 components; the favourable clusters are one regime seen through different lenses
rather than independent opportunities. The cluster count would normally be set by maximising a
silhouette coefficient, but the events carry no strong intrinsic partition: aggregated over the
whole breakout grid (all four horizons, every screened configuration) the silhouette has a
median of 0.02 and a maximum of 0.11, near 0.03 at k=6 and falling toward 0.016 at k=10, and
the two clusters carried forward sit near 0.01, no different from the grid at large. Internal validity therefore cannot select k, so the screening fixes the
structural parameters by task-relevant separation (the MFE/MAE ratio) and defers the final
choice to the directional test.

### Viable cluster selection (4.4.4)

The directional test fixes each cluster's majority direction on the training portion of an
expanding-window split and records how often held-out events follow it (directed accuracy),
so it captures forward-looking predictability rather than a descriptive path property. A
cluster clears the gate at accuracy above 0.55, confirmed by a bias-corrected and accelerated
(BCa) bootstrap confidence interval whose lower bound clears one half. Across the grid, 150
clusters clear the viability, directional, and 100-event conditions together, of which 139
remain significant under the BCa interval; 134 of the 139 are reversal, with a median directed
accuracy of 0.58 and a median 60 s ratio of 1.06, a pervasive but weak reversal tendency.

Restricting to a directed accuracy of at least 0.70 leaves three clusters across two regimes,
both reversal at the 30 s horizon and 30 bps threshold, one per asset. The two BTC candidates
are the same region at different resolution (top-50 Jaccard of 0.19 against a random-pair
median near 0.03), and the k=8 view carries the stronger, more window-consistent asymmetry, so
the k=6 view is set aside, leaving one representative per asset:

| Configuration | n | MFE/MAE (60 s) | MFE (bps) | MAE (bps) | MFE-lift | DA (60 s) | 95 % BCa CI |
|---|---|---|---|---|---|---|---|
| BTC / 30 s / 30 bps, PCA 600, k=8 | 1,075 | 1.36 | 3.3 | 2.5 | 1.12 | 0.73 | [0.66, 0.79] |
| ETH / 30 s / 30 bps, PCA 300, k=8 | 1,346 | 1.03 | 4.1 | 3.9 | 0.96 | 0.78 | [0.69, 0.84] |

Both are reversal and reach a directed accuracy between 0.73 and 0.78 at the 30 to 60 s
horizon, conditional on known membership; the accuracy peaks at 30 to 60 s and then decays
(ETH below chance by 120 s, BTC by 300 s). They differ in exit headroom: BTC pairs its
direction with genuine asymmetry (60 s ratio 1.36, favourable 3.3 bps clear of adverse 2.5 bps),
while ETH has almost none (ratio 1.03, favourable 4.1 bps against adverse 3.9 bps).

### Real-time recognisability and stability (4.4.5, 4.4.6)

A LightGBM probe, refit entirely on each fold, measures whether an emerging breakout can be
recognised as a good cluster from the features available one to five seconds ahead. BTC is
recognisable and robustly so (precision 0.92 at one second ahead, 0.91 at five), ETH only in
the moment before the move (0.83 at one second, 0.64 at five). Precision is measured against
the stable training base rate, so it states how cleanly the structure separates, not a
deployment hit rate.

The partition itself, however, is unstable across every test:

| Test | BTC 30 s / 30 bps | ETH 30 s / 30 bps |
|---|---|---|
| Block ARI vs full (5 blocks) | 0.31 | 0.28 |
| Consecutive-block ARI | 0.24 | 0.21 |
| Multi-seed ARI (5 seeds) | 0.46 | 0.55 |
| Carry-over (train to test, DA > 0.55) | 2 / 4 | 0 / 3 |
| Rolling-window persistence (DA > 0.55) | 6 / 14 | 7 / 15 |

The signal is tied to a region of the feature space, not to the clustering that first isolated
it: k-means groups events by where they sit and never sees the direction the price later takes,
so when the partition drifts the region is lost and the realised accuracy falls back toward one
half. This points the analysis to describe the region directly and to test it as a fixed rule
learned on training data and carried forward without re-clustering.

## 9. The two feature-combination states (4.4.7)

Each state is described by the features whose average value inside the cluster departs most
from the average across all breakouts, at the family level, in standard deviations. The
deviations are only moderate (about 0.5 to 1.2 SD), so each state is a soft region of the
feature space, which is why re-clustering does not recover it cleanly. Both describe an
overextended price that then reverses; every trend feature in each top 50 runs positive (the
price stands about 0.6 SD above its moving averages for BTC and 1.1 for ETH).

- **BTC state:** aggressive buying at a stretched level that fails to sustain. The price sits
  above its high-volume value area (Volume Profile +0.82 SD) and near the top of its hourly
  range (Range +0.77), the taker buy-sell imbalance runs about six times its average-breakout
  level (Aggression +0.60), and the price is stretched above its EMAs (Trend +0.64). Yet the
  resting book is strongly bid-weighted (Absorption -0.65), so the buying is absorbed rather
  than carried further and the price reverses.
- **ETH state:** overextension onto an emptying book. The price is stretched even further above
  its averages (Trend +1.14) and above the value area (Volume Profile +1.06) and the prior
  day's high (Level Events +0.98), trading activity is high (Activity +1.01), and net
  cancellations run about ten times their average-breakout level (Pressure +1.05), so the book
  thins as the price extends.

Both signatures draw on both markets: the BTC state is about 58 % BTC and 42 % ETH features,
the ETH state about 56 % ETH and 44 % BTC (in the ETH state the entire Pressure row is carried
by BTC net-cancellation features). Tested as a forward signal, a LightGBM classifier trained on
the 50 defining features of each state predicts direction out of sample at 0.47 to 0.54 across
5 s to 900 s, about one half (0.50 BTC, 0.51 ETH over the 5 s to 120 s horizons that matter);
the 1 s exception (0.76 BTC, 0.73 ETH) is second-to-second microstructure persistence that has
vanished by 5 s. So the states are describable and coherent, and informative once membership is
known, but the defining features carry no directional signal on their own and the membership
cannot be recovered by re-clustering.

## 10. Directional predictability and tradability: the two reasons (4.5)

Four methods were compared on the 30 s / 30 bps breakouts: two entry feature sets (the full
tree profile against the return-ranked subset) crossed with two exits (a fixed TP/SL grid from
the training excursion distribution against a dynamic exit classifier on the excursion-ranked
features), gated by entry confidence at 0.5, 0.6, 0.7, under both cost models, over five
expanding walk-forward folds and three seeds. The dynamic exit classifier does not beat the
fixed grid (its discrimination is near chance, AUC about 0.5), the return-ranked subset performs
on par with the full profile, and the confidence gate barely moves the result.

The tradability failure has two reasons, and they compound.

1. **The directional edge exists only under known cluster membership, which does not survive an
   honest out-of-sample test.** Measured honestly out of sample on the same breakouts, a
   walk-forward LightGBM reaches a directional accuracy of only 0.49 (BTC) and 0.51 (ETH) at
   60 s; the cluster method, refit on training data alone and applied forward, reaches 0.56 and
   0.52. Both sit well below the 0.73 and 0.78 that assumed membership was known, and the gap is
   the price of that assumption. Across every horizon from 15 s to 900 s neither method exceeds
   0.56, oscillating around 0.5 without a consistent sign, the signature of a near-zero edge. The
   partition is only weakly reproducible (block ARI below 0.32), so a live system would have to
   assign membership from a structure it cannot recover.
2. **The excursions are too small to clear the cost.** The strategy grid comprises 264
   configurations for BTC and 276 for ETH; none is profitable, and the best case still loses
   4.0 bps per trade for BTC and 1.8 for ETH under the cheaper maker/taker cost. The favourable
   move at 60 s is only 3.3 bps for BTC and 4.1 for ETH, against a round-trip cost of 7 bps for a
   maker entry with taker exit (2 bps maker plus 5 bps taker) or 10 bps for taker on both sides,
   so the move is smaller than the fee before any exit rule is applied. The barrier is the fee,
   not the spread: both assets are tick-constrained with a median quoted spread below 0.05 bps.
   For ETH the asymmetry works against it, the adverse move overtaking the favourable one beyond
   60 s (6.7 against 5.9 bps at 120 s, 11.1 against 9.0 at 300 s), so holding longer enlarges the
   downside faster than the upside; the favourable move only approaches the taker cost near 300 s,
   a horizon at which the direction is no better than chance.

Larger excursions were not absent from the configuration search (clusters with a favourable move
above 10 bps existed for both assets), but there the direction was either close to chance (ETH)
or confined to configurations the selection had already ruled out on accuracy (BTC): a large
excursion and a reliable, reproducible direction did not coincide. The two states that survived
selection are therefore describable and internally coherent but not tradable. Their directional
edge exists only under a cluster membership that does not survive an honest out-of-sample test,
and their excursions are too small to clear the cost, so the reported 3.3 and 4.1 bps at 60 s
are an upper bound on an already unprofitable trade.
