# etl/spec/s1/s1_trend.py
# ==============================================================================
# S1 Feature Specs: Trend / EMA
#
# Binance-only pipeline | Source: S0 features (parquet)
# 25 features | Feature IDs: 1334-1358
#
# PURPOSE:
#   Exponential moving average trend features on mid_fut_1s across four
#   timeframes (5m, 15m, 60m, 240m) and two periods (N=50, N=200). The EMA
#   span in seconds is span_s = N * TF_seconds so that a "50-period EMA on
#   5m" directly corresponds to span_s = 50 * 300 = 15000 on the 1s grid.
#
# STRUCTURE:
#   (A) EMA prices (8):
#       ema_{50,200}_{5m,15m,60m,240m}_fut
#       IDs 1334-1341. Operator: derived.ema. Dep: mid_fut_1s.
#
#   (B) Relative distance in bps (8):
#       price_vs_ema_{50,200}_{5m,15m,60m,240m}_bps_fut
#       IDs 1342-1349. Operator: derived.price_vs_ema_bps.
#       Deps: [mid_fut_1s, ema_<N>_<TF>_fut]
#       Positive = price above EMA (bullish bias), negative = below.
#
#   (C) EMA slope in bps (6, 5m intentionally omitted):
#       ema_slope_{50,200}_{15m,60m,240m}_bps_fut
#       IDs 1350-1355. Operator: derived.ema_slope_bps.
#       Dep: ema_<N>_<TF>_fut. shift_s = TF_seconds.
#       Rationale: 5m-slope is noise-dominated and redundant with
#       ret_mid_fut_60s / ret_fwd_15s. Longer timeframes carry signal.
#
#   (D) Trend alignment (3, 5m intentionally omitted):
#       trend_aligned_{15m,60m,240m}_fut
#       IDs 1356-1358. Operator: derived.trend_align.
#       Deps: [mid_fut_1s, ema_50_<TF>_fut, ema_200_<TF>_fut]
#       Ternary: +1 if price > ema_50 > ema_200 (confirmed uptrend),
#                -1 if price < ema_50 < ema_200 (confirmed downtrend),
#                 0 otherwise (range / mixed).
#
# WARMUP:
#   EMAs become stable around span_s/4 rows. For 240m×200 (span 2.88M
#   seconds) the cold-path batch will still show material warmup for
#   several days — fine for training on backfilled data. In the live
#   hot-path the EMA state is bootstrapped from the last cold-path
#   parquet at pipeline start (Phase 7 concern).
#
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep


# Mapping of timeframe label -> seconds.
_TF_SECONDS = {
    "5m":   300,
    "15m":  900,
    "60m":  3600,
    "240m": 14400,
}


S1_TREND_FEATURES: List[FeatureSpec] = [

    # =========================================================================
    # (A) EMAs — ema_{N}_{TF}_fut
    #     span_s = N_periods * TF_seconds on the 1s grid.
    # =========================================================================

    # N=50
    FeatureSpec(
        name="ema_50_5m_fut",
        stage="S1",
        operator="derived.ema",
        params={'market_scope': 'Futures', 'span_s': 50 * _TF_SECONDS["5m"], 'resample': '1s'},
        group="Trend",
        description="Exponential moving average, 50-period on 5m timeframe (span_s = 15000).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1334,
    ),
    FeatureSpec(
        name="ema_50_15m_fut",
        stage="S1",
        operator="derived.ema",
        params={'market_scope': 'Futures', 'span_s': 50 * _TF_SECONDS["15m"], 'resample': '1s'},
        group="Trend",
        description="EMA, 50-period on 15m timeframe (span_s = 45000).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1335,
    ),
    FeatureSpec(
        name="ema_50_60m_fut",
        stage="S1",
        operator="derived.ema",
        params={'market_scope': 'Futures', 'span_s': 50 * _TF_SECONDS["60m"], 'resample': '1s'},
        group="Trend",
        description="EMA, 50-period on 60m timeframe (span_s = 180000).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1336,
    ),
    FeatureSpec(
        name="ema_50_240m_fut",
        stage="S1",
        operator="derived.ema",
        params={'market_scope': 'Futures', 'span_s': 50 * _TF_SECONDS["240m"], 'resample': '1s'},
        group="Trend",
        description="EMA, 50-period on 240m (4h) timeframe (span_s = 720000).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1337,
    ),

    # N=200
    FeatureSpec(
        name="ema_200_5m_fut",
        stage="S1",
        operator="derived.ema",
        params={'market_scope': 'Futures', 'span_s': 200 * _TF_SECONDS["5m"], 'resample': '1s'},
        group="Trend",
        description="EMA, 200-period on 5m timeframe (span_s = 60000).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1338,
    ),
    FeatureSpec(
        name="ema_200_15m_fut",
        stage="S1",
        operator="derived.ema",
        params={'market_scope': 'Futures', 'span_s': 200 * _TF_SECONDS["15m"], 'resample': '1s'},
        group="Trend",
        description="EMA, 200-period on 15m timeframe (span_s = 180000).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1339,
    ),
    FeatureSpec(
        name="ema_200_60m_fut",
        stage="S1",
        operator="derived.ema",
        params={'market_scope': 'Futures', 'span_s': 200 * _TF_SECONDS["60m"], 'resample': '1s'},
        group="Trend",
        description="EMA, 200-period on 60m timeframe (span_s = 720000).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1340,
    ),
    FeatureSpec(
        name="ema_200_240m_fut",
        stage="S1",
        operator="derived.ema",
        params={'market_scope': 'Futures', 'span_s': 200 * _TF_SECONDS["240m"], 'resample': '1s'},
        group="Trend",
        description="EMA, 200-period on 240m (4h) timeframe (span_s = 2880000).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1341,
    ),

    # =========================================================================
    # (B) price_vs_ema_*_bps_fut — relative distance in bps
    # =========================================================================

    # N=50
    FeatureSpec(
        name="price_vs_ema_50_5m_bps_fut",
        stage="S1",
        operator="derived.price_vs_ema_bps",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Trend",
        description="(mid_fut - ema_50_5m_fut) / ema_50_5m_fut * 10000 in bps.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),
                    Dep(name="ema_50_5m_fut", kind="col")),
        feature_id=1342,
    ),
    FeatureSpec(
        name="price_vs_ema_50_15m_bps_fut",
        stage="S1",
        operator="derived.price_vs_ema_bps",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Trend",
        description="(mid_fut - ema_50_15m_fut) / ema_50_15m_fut * 10000 in bps.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),
                    Dep(name="ema_50_15m_fut", kind="col")),
        feature_id=1343,
    ),
    FeatureSpec(
        name="price_vs_ema_50_60m_bps_fut",
        stage="S1",
        operator="derived.price_vs_ema_bps",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Trend",
        description="(mid_fut - ema_50_60m_fut) / ema_50_60m_fut * 10000 in bps.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),
                    Dep(name="ema_50_60m_fut", kind="col")),
        feature_id=1344,
    ),
    FeatureSpec(
        name="price_vs_ema_50_240m_bps_fut",
        stage="S1",
        operator="derived.price_vs_ema_bps",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Trend",
        description="(mid_fut - ema_50_240m_fut) / ema_50_240m_fut * 10000 in bps.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),
                    Dep(name="ema_50_240m_fut", kind="col")),
        feature_id=1345,
    ),

    # N=200
    FeatureSpec(
        name="price_vs_ema_200_5m_bps_fut",
        stage="S1",
        operator="derived.price_vs_ema_bps",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Trend",
        description="(mid_fut - ema_200_5m_fut) / ema_200_5m_fut * 10000 in bps.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),
                    Dep(name="ema_200_5m_fut", kind="col")),
        feature_id=1346,
    ),
    FeatureSpec(
        name="price_vs_ema_200_15m_bps_fut",
        stage="S1",
        operator="derived.price_vs_ema_bps",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Trend",
        description="(mid_fut - ema_200_15m_fut) / ema_200_15m_fut * 10000 in bps.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),
                    Dep(name="ema_200_15m_fut", kind="col")),
        feature_id=1347,
    ),
    FeatureSpec(
        name="price_vs_ema_200_60m_bps_fut",
        stage="S1",
        operator="derived.price_vs_ema_bps",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Trend",
        description="(mid_fut - ema_200_60m_fut) / ema_200_60m_fut * 10000 in bps.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),
                    Dep(name="ema_200_60m_fut", kind="col")),
        feature_id=1348,
    ),
    FeatureSpec(
        name="price_vs_ema_200_240m_bps_fut",
        stage="S1",
        operator="derived.price_vs_ema_bps",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Trend",
        description="(mid_fut - ema_200_240m_fut) / ema_200_240m_fut * 10000 in bps.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),
                    Dep(name="ema_200_240m_fut", kind="col")),
        feature_id=1349,
    ),

    # =========================================================================
    # (C) ema_slope_*_bps_fut — slope over TF_seconds (5m timeframe omitted)
    # =========================================================================

    # N=50
    FeatureSpec(
        name="ema_slope_50_15m_bps_fut",
        stage="S1",
        operator="derived.ema_slope_bps",
        params={'market_scope': 'Futures', 'shift_s': _TF_SECONDS["15m"], 'resample': '1s'},
        group="Trend",
        description=(
            "EMA slope in bps over 15m shift: "
            "(ema_50_15m[t] - ema_50_15m[t-900]) / ema_50_15m[t-900] * 10000."
        ),
        depends_on=(Dep(name="ema_50_15m_fut", kind="col"),),
        feature_id=1350,
    ),
    FeatureSpec(
        name="ema_slope_50_60m_bps_fut",
        stage="S1",
        operator="derived.ema_slope_bps",
        params={'market_scope': 'Futures', 'shift_s': _TF_SECONDS["60m"], 'resample': '1s'},
        group="Trend",
        description="EMA slope in bps over 60m shift on ema_50_60m_fut.",
        depends_on=(Dep(name="ema_50_60m_fut", kind="col"),),
        feature_id=1351,
    ),
    FeatureSpec(
        name="ema_slope_50_240m_bps_fut",
        stage="S1",
        operator="derived.ema_slope_bps",
        params={'market_scope': 'Futures', 'shift_s': _TF_SECONDS["240m"], 'resample': '1s'},
        group="Trend",
        description="EMA slope in bps over 240m shift on ema_50_240m_fut.",
        depends_on=(Dep(name="ema_50_240m_fut", kind="col"),),
        feature_id=1352,
    ),

    # N=200
    FeatureSpec(
        name="ema_slope_200_15m_bps_fut",
        stage="S1",
        operator="derived.ema_slope_bps",
        params={'market_scope': 'Futures', 'shift_s': _TF_SECONDS["15m"], 'resample': '1s'},
        group="Trend",
        description="EMA slope in bps over 15m shift on ema_200_15m_fut.",
        depends_on=(Dep(name="ema_200_15m_fut", kind="col"),),
        feature_id=1353,
    ),
    FeatureSpec(
        name="ema_slope_200_60m_bps_fut",
        stage="S1",
        operator="derived.ema_slope_bps",
        params={'market_scope': 'Futures', 'shift_s': _TF_SECONDS["60m"], 'resample': '1s'},
        group="Trend",
        description="EMA slope in bps over 60m shift on ema_200_60m_fut.",
        depends_on=(Dep(name="ema_200_60m_fut", kind="col"),),
        feature_id=1354,
    ),
    FeatureSpec(
        name="ema_slope_200_240m_bps_fut",
        stage="S1",
        operator="derived.ema_slope_bps",
        params={'market_scope': 'Futures', 'shift_s': _TF_SECONDS["240m"], 'resample': '1s'},
        group="Trend",
        description="EMA slope in bps over 240m shift on ema_200_240m_fut.",
        depends_on=(Dep(name="ema_200_240m_fut", kind="col"),),
        feature_id=1355,
    ),

    # =========================================================================
    # (D) trend_aligned_*_fut — ternary regime flag (5m timeframe omitted)
    # =========================================================================

    FeatureSpec(
        name="trend_aligned_15m_fut",
        stage="S1",
        operator="derived.trend_align",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Trend",
        description=(
            "Trend alignment on 15m timeframe: +1 if price>ema_50>ema_200, "
            "-1 if price<ema_50<ema_200, else 0."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",      kind="col"),
            Dep(name="ema_50_15m_fut",  kind="col"),
            Dep(name="ema_200_15m_fut", kind="col"),
        ),
        feature_id=1356,
    ),
    FeatureSpec(
        name="trend_aligned_60m_fut",
        stage="S1",
        operator="derived.trend_align",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Trend",
        description="Trend alignment on 60m timeframe.",
        depends_on=(
            Dep(name="mid_fut_1s",      kind="col"),
            Dep(name="ema_50_60m_fut",  kind="col"),
            Dep(name="ema_200_60m_fut", kind="col"),
        ),
        feature_id=1357,
    ),
    FeatureSpec(
        name="trend_aligned_240m_fut",
        stage="S1",
        operator="derived.trend_align",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Trend",
        description="Trend alignment on 240m (4h) timeframe.",
        depends_on=(
            Dep(name="mid_fut_1s",       kind="col"),
            Dep(name="ema_50_240m_fut",  kind="col"),
            Dep(name="ema_200_240m_fut", kind="col"),
        ),
        feature_id=1358,
    ),
]