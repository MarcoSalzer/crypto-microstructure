# etl/spec/s5/s5_dynamics.py
# ==============================================================================
# S5 Dynamics Features
# ==============================================================================
# Overview:
#   Temporal shock detection and per-market directional-persistence analytics
#   for absorption-break composite scores, liquidity-vacuum scores, and per-market
#   net-add/cancel order-flow pressure series.
#
#   Four sub-families:
#
#   1) Absorption-break shock pipeline  (6 features):
#      Computes rolling median → rolling MAD → robust shock for
#      absorption_break_{fut,spot}_15s (the S4 composite scores).
#      Dependency chain (depth 2):
#        median_absorption_break_*_15s   ← derived.rolling_median  (Level 0)
#        mad_absorption_break_*_15s      ← derived.rolling_mad     (Level 1)
#        absorption_break_shock_*_15s    ← derived.robust_shock    (Level 2)
#      Window: 15s (matches the timeframe of the absorption_break signal).
#
#   2) Vacuum-score shock pipeline  (18 features):
#      Same 3-level pipeline for vacuum_score_{fut,spot}_{1,2}bps_{15,60}s
#      (the S4 liquidity-vacuum scores).
#      Variants per market: 1bps@15s, 2bps@15s, 2bps@60s (no 1bps@60s variant).
#      Windows: 15s or 60s depending on the source signal's timeframe.
#      Dependency chain depth: 2 (same as absorption-break).
#
#   3) Per-market net-add persistence  (12 features):
#      Directional persistence of net-add rolling sums (from S4) for
#      {fut,spot} × {2,5,10}bps × {15,60}s using the S5 signal_persist formula:
#        persist = abs(roll_mean(x)) / (roll_mean(abs(x)) + eps)
#      No intra-S5 dependencies — reads S4 columns directly.
#
#   4) Per-market net-cancel persistence  (12 features):
#      Same as sub-family 3 but for net_cancel rolling sums.
#
# Features (48 total):
#   Absorption-break shock pipeline (6):
#     - median_absorption_break_{fut,spot}_15s               (2)
#     - mad_absorption_break_{fut,spot}_15s                  (2)
#     - absorption_break_shock_{fut,spot}_15s                (2)
#   Vacuum-score shock pipeline (18):
#     - median_vacuum_score_{fut,spot}_{1bps_15s, 2bps_15s, 2bps_60s}   (6)
#     - mad_vacuum_score_{fut,spot}_{1bps_15s, 2bps_15s, 2bps_60s}      (6)
#     - vacuum_score_shock_{fut,spot}_{1bps_15s, 2bps_15s, 2bps_60s}    (6)
#     Note: 1bps variant exists at 15s only; 2bps has both 15s and 60s.
#           No 1bps@60s variant exists in S4 input.
#   Net-add persistence (12):
#     - net_add_persist_{fut,spot}_{2,5,10}bps_{15,60}s     (12)
#   Net-cancel persistence (12):
#     - net_cancel_persist_{fut,spot}_{2,5,10}bps_{15,60}s  (12)
#
# Operators used:
#   derived.rolling_median  — rolling median (Level 0 of shock pipelines)
#   derived.rolling_mad     — rolling MAD    (Level 1, depends on median)
#   derived.robust_shock    — shock score    (Level 2, depends on median + MAD)
#   derived.signal_persist  — S5 persistence formula for net-add/cancel
#
# NaN policy:
#   Rolling operators: first (window_s - 1) rows → NaN.
#   robust_shock: NaN where any of base / median / MAD is NaN, or where MAD = 0.
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep

S5_DYNAMICS_FEATURES: List[FeatureSpec] = [

    # =========================================================================
    # 1. ABSORPTION-BREAK SHOCK PIPELINE
    # =========================================================================
    # Level 0: rolling median

    FeatureSpec(
        name="median_absorption_break_fut_15s",
        stage="S5",
        operator="derived.rolling_median",
        params={
            "market_scope": "Futures",
            "input_col": "absorption_break_fut_15s",
            "window_s": "15",
        },
        label="Median Absorption Break Fut 15S (Binance)",
        group="Dynamics",
        description="Rolling median of absorption_break_fut_15s (Futures, 15s window).",
        depends_on=(Dep(name="absorption_break_fut_15s", kind="col"),),
        feature_id=5004,
    ),

    FeatureSpec(
        name="median_absorption_break_spot_15s",
        stage="S5",
        operator="derived.rolling_median",
        params={
            "market_scope": "Spot",
            "input_col": "absorption_break_spot_15s",
            "window_s": "15",
        },
        label="Median Absorption Break Spot 15S (Binance)",
        group="Dynamics",
        description="Rolling median of absorption_break_spot_15s (Spot, 15s window).",
        depends_on=(Dep(name="absorption_break_spot_15s", kind="col"),),
        feature_id=5005,
    ),

    # Level 1: rolling MAD (depends on Level 0 median)

    FeatureSpec(
        name="mad_absorption_break_fut_15s",
        stage="S5",
        operator="derived.rolling_mad",
        params={
            "market_scope": "Futures",
            "base_col": "absorption_break_fut_15s",
            "median_col": "median_absorption_break_fut_15s",
            "window_s": "15",
        },
        label="MAD Absorption Break Fut 15S (Binance)",
        group="Dynamics",
        description=(
            "Rolling MAD of absorption_break_fut_15s (Futures, 15s). "
            "Depends on median_absorption_break_fut_15s (intra-S5)."
        ),
        depends_on=(
            Dep(name="absorption_break_fut_15s", kind="col"),
            Dep(name="median_absorption_break_fut_15s", kind="col"),
        ),
        feature_id=5006,
    ),

    FeatureSpec(
        name="mad_absorption_break_spot_15s",
        stage="S5",
        operator="derived.rolling_mad",
        params={
            "market_scope": "Spot",
            "base_col": "absorption_break_spot_15s",
            "median_col": "median_absorption_break_spot_15s",
            "window_s": "15",
        },
        label="MAD Absorption Break Spot 15S (Binance)",
        group="Dynamics",
        description=(
            "Rolling MAD of absorption_break_spot_15s (Spot, 15s). "
            "Depends on median_absorption_break_spot_15s (intra-S5)."
        ),
        depends_on=(
            Dep(name="absorption_break_spot_15s", kind="col"),
            Dep(name="median_absorption_break_spot_15s", kind="col"),
        ),
        feature_id=5007,
    ),

    # Level 2: robust shock (depends on Level 0 + Level 1)

    FeatureSpec(
        name="absorption_break_shock_fut_15s",
        stage="S5",
        operator="derived.robust_shock",
        params={
            "market_scope": "Futures",
            "base_col": "absorption_break_fut_15s",
            "median_col": "median_absorption_break_fut_15s",
            "mad_col": "mad_absorption_break_fut_15s",
        },
        label="Absorption Break Shock Fut 15S (Binance)",
        group="Dynamics",
        description=(
            "Robust shock for absorption_break_fut_15s: "
            "|x - median| / (MAD + eps). Detects outlier absorption events (Futures, 15s)."
        ),
        depends_on=(
            Dep(name="absorption_break_fut_15s", kind="col"),
            Dep(name="mad_absorption_break_fut_15s", kind="col"),
            Dep(name="median_absorption_break_fut_15s", kind="col"),
        ),
        feature_id=5008,
    ),

    FeatureSpec(
        name="absorption_break_shock_spot_15s",
        stage="S5",
        operator="derived.robust_shock",
        params={
            "market_scope": "Spot",
            "base_col": "absorption_break_spot_15s",
            "median_col": "median_absorption_break_spot_15s",
            "mad_col": "mad_absorption_break_spot_15s",
        },
        label="Absorption Break Shock Spot 15S (Binance)",
        group="Dynamics",
        description=(
            "Robust shock for absorption_break_spot_15s: "
            "|x - median| / (MAD + eps). Detects outlier absorption events (Spot, 15s)."
        ),
        depends_on=(
            Dep(name="absorption_break_spot_15s", kind="col"),
            Dep(name="mad_absorption_break_spot_15s", kind="col"),
            Dep(name="median_absorption_break_spot_15s", kind="col"),
        ),
        feature_id=5009,
    ),

    # =========================================================================
    # 2. VACUUM-SCORE SHOCK PIPELINE
    # =========================================================================
    # Level 0: rolling medians for vacuum_score variants

    # Level 1: rolling MADs (depend on Level 0 medians)

    # Level 2: robust shocks (depend on Level 0 + Level 1)

    # =========================================================================
    # 3. PER-MARKET NET-ADD PERSISTENCE
    # =========================================================================

    # --- Futures, 2bps ---

    FeatureSpec(
        name="net_add_persist_fut_2bps_15s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Futures",
            "input_col": "net_add_fut_2bps_15s",
            "window_s": "15",
        },
        label="Net Add Persist Fut 2Bps 15S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_add_fut_2bps_15s (Futures, 2bps, 15s).",
        depends_on=(Dep(name="net_add_fut_2bps_15s", kind="col"),),
        feature_id=5010,
    ),

    FeatureSpec(
        name="net_add_persist_fut_2bps_60s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Futures",
            "input_col": "net_add_fut_2bps_60s",
            "window_s": "60",
        },
        label="Net Add Persist Fut 2Bps 60S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_add_fut_2bps_60s (Futures, 2bps, 60s).",
        depends_on=(Dep(name="net_add_fut_2bps_60s", kind="col"),),
        feature_id=5011,
    ),

    # --- Futures, 5bps ---

    FeatureSpec(
        name="net_add_persist_fut_5bps_15s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Futures",
            "input_col": "net_add_fut_5bps_15s",
            "window_s": "15",
        },
        label="Net Add Persist Fut 5Bps 15S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_add_fut_5bps_15s (Futures, 5bps, 15s).",
        depends_on=(Dep(name="net_add_fut_5bps_15s", kind="col"),),
        feature_id=5012,
    ),

    FeatureSpec(
        name="net_add_persist_fut_5bps_60s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Futures",
            "input_col": "net_add_fut_5bps_60s",
            "window_s": "60",
        },
        label="Net Add Persist Fut 5Bps 60S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_add_fut_5bps_60s (Futures, 5bps, 60s).",
        depends_on=(Dep(name="net_add_fut_5bps_60s", kind="col"),),
        feature_id=5013,
    ),

    # --- Futures, 10bps ---

    FeatureSpec(
        name="net_add_persist_fut_10bps_15s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Futures",
            "input_col": "net_add_fut_10bps_15s",
            "window_s": "15",
        },
        label="Net Add Persist Fut 10Bps 15S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_add_fut_10bps_15s (Futures, 10bps, 15s).",
        depends_on=(Dep(name="net_add_fut_10bps_15s", kind="col"),),
        feature_id=5014,
    ),

    FeatureSpec(
        name="net_add_persist_fut_10bps_60s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Futures",
            "input_col": "net_add_fut_10bps_60s",
            "window_s": "60",
        },
        label="Net Add Persist Fut 10Bps 60S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_add_fut_10bps_60s (Futures, 10bps, 60s).",
        depends_on=(Dep(name="net_add_fut_10bps_60s", kind="col"),),
        feature_id=5015,
    ),

    # --- Spot, 2bps ---

    FeatureSpec(
        name="net_add_persist_spot_2bps_15s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Spot",
            "input_col": "net_add_spot_2bps_15s",
            "window_s": "15",
        },
        label="Net Add Persist Spot 2Bps 15S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_add_spot_2bps_15s (Spot, 2bps, 15s).",
        depends_on=(Dep(name="net_add_spot_2bps_15s", kind="col"),),
        feature_id=5016,
    ),

    FeatureSpec(
        name="net_add_persist_spot_2bps_60s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Spot",
            "input_col": "net_add_spot_2bps_60s",
            "window_s": "60",
        },
        label="Net Add Persist Spot 2Bps 60S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_add_spot_2bps_60s (Spot, 2bps, 60s).",
        depends_on=(Dep(name="net_add_spot_2bps_60s", kind="col"),),
        feature_id=5017,
    ),

    # --- Spot, 5bps ---

    FeatureSpec(
        name="net_add_persist_spot_5bps_15s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Spot",
            "input_col": "net_add_spot_5bps_15s",
            "window_s": "15",
        },
        label="Net Add Persist Spot 5Bps 15S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_add_spot_5bps_15s (Spot, 5bps, 15s).",
        depends_on=(Dep(name="net_add_spot_5bps_15s", kind="col"),),
        feature_id=5018,
    ),

    FeatureSpec(
        name="net_add_persist_spot_5bps_60s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Spot",
            "input_col": "net_add_spot_5bps_60s",
            "window_s": "60",
        },
        label="Net Add Persist Spot 5Bps 60S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_add_spot_5bps_60s (Spot, 5bps, 60s).",
        depends_on=(Dep(name="net_add_spot_5bps_60s", kind="col"),),
        feature_id=5019,
    ),

    # --- Spot, 10bps ---

    FeatureSpec(
        name="net_add_persist_spot_10bps_15s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Spot",
            "input_col": "net_add_spot_10bps_15s",
            "window_s": "15",
        },
        label="Net Add Persist Spot 10Bps 15S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_add_spot_10bps_15s (Spot, 10bps, 15s).",
        depends_on=(Dep(name="net_add_spot_10bps_15s", kind="col"),),
        feature_id=5020,
    ),

    FeatureSpec(
        name="net_add_persist_spot_10bps_60s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Spot",
            "input_col": "net_add_spot_10bps_60s",
            "window_s": "60",
        },
        label="Net Add Persist Spot 10Bps 60S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_add_spot_10bps_60s (Spot, 10bps, 60s).",
        depends_on=(Dep(name="net_add_spot_10bps_60s", kind="col"),),
        feature_id=5021,
    ),

    # =========================================================================
    # 4. PER-MARKET NET-CANCEL PERSISTENCE
    # =========================================================================

    # --- Futures, 2bps ---

    FeatureSpec(
        name="net_cancel_persist_fut_2bps_15s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Futures",
            "input_col": "net_cancel_fut_2bps_15s",
            "window_s": "15",
        },
        label="Net Cancel Persist Fut 2Bps 15S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_cancel_fut_2bps_15s (Futures, 2bps, 15s).",
        depends_on=(Dep(name="net_cancel_fut_2bps_15s", kind="col"),),
        feature_id=5022,
    ),

    FeatureSpec(
        name="net_cancel_persist_fut_2bps_60s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Futures",
            "input_col": "net_cancel_fut_2bps_60s",
            "window_s": "60",
        },
        label="Net Cancel Persist Fut 2Bps 60S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_cancel_fut_2bps_60s (Futures, 2bps, 60s).",
        depends_on=(Dep(name="net_cancel_fut_2bps_60s", kind="col"),),
        feature_id=5023,
    ),

    # --- Futures, 5bps ---

    FeatureSpec(
        name="net_cancel_persist_fut_5bps_15s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Futures",
            "input_col": "net_cancel_fut_5bps_15s",
            "window_s": "15",
        },
        label="Net Cancel Persist Fut 5Bps 15S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_cancel_fut_5bps_15s (Futures, 5bps, 15s).",
        depends_on=(Dep(name="net_cancel_fut_5bps_15s", kind="col"),),
        feature_id=5024,
    ),

    FeatureSpec(
        name="net_cancel_persist_fut_5bps_60s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Futures",
            "input_col": "net_cancel_fut_5bps_60s",
            "window_s": "60",
        },
        label="Net Cancel Persist Fut 5Bps 60S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_cancel_fut_5bps_60s (Futures, 5bps, 60s).",
        depends_on=(Dep(name="net_cancel_fut_5bps_60s", kind="col"),),
        feature_id=5025,
    ),

    # --- Futures, 10bps ---

    FeatureSpec(
        name="net_cancel_persist_fut_10bps_15s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Futures",
            "input_col": "net_cancel_fut_10bps_15s",
            "window_s": "15",
        },
        label="Net Cancel Persist Fut 10Bps 15S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_cancel_fut_10bps_15s (Futures, 10bps, 15s).",
        depends_on=(Dep(name="net_cancel_fut_10bps_15s", kind="col"),),
        feature_id=5026,
    ),

    FeatureSpec(
        name="net_cancel_persist_fut_10bps_60s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Futures",
            "input_col": "net_cancel_fut_10bps_60s",
            "window_s": "60",
        },
        label="Net Cancel Persist Fut 10Bps 60S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_cancel_fut_10bps_60s (Futures, 10bps, 60s).",
        depends_on=(Dep(name="net_cancel_fut_10bps_60s", kind="col"),),
        feature_id=5027,
    ),

    # --- Spot, 2bps ---

    FeatureSpec(
        name="net_cancel_persist_spot_2bps_15s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Spot",
            "input_col": "net_cancel_spot_2bps_15s",
            "window_s": "15",
        },
        label="Net Cancel Persist Spot 2Bps 15S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_cancel_spot_2bps_15s (Spot, 2bps, 15s).",
        depends_on=(Dep(name="net_cancel_spot_2bps_15s", kind="col"),),
        feature_id=5028,
    ),

    FeatureSpec(
        name="net_cancel_persist_spot_2bps_60s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Spot",
            "input_col": "net_cancel_spot_2bps_60s",
            "window_s": "60",
        },
        label="Net Cancel Persist Spot 2Bps 60S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_cancel_spot_2bps_60s (Spot, 2bps, 60s).",
        depends_on=(Dep(name="net_cancel_spot_2bps_60s", kind="col"),),
        feature_id=5029,
    ),

    # --- Spot, 5bps ---

    FeatureSpec(
        name="net_cancel_persist_spot_5bps_15s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Spot",
            "input_col": "net_cancel_spot_5bps_15s",
            "window_s": "15",
        },
        label="Net Cancel Persist Spot 5Bps 15S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_cancel_spot_5bps_15s (Spot, 5bps, 15s).",
        depends_on=(Dep(name="net_cancel_spot_5bps_15s", kind="col"),),
        feature_id=5030,
    ),

    FeatureSpec(
        name="net_cancel_persist_spot_5bps_60s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Spot",
            "input_col": "net_cancel_spot_5bps_60s",
            "window_s": "60",
        },
        label="Net Cancel Persist Spot 5Bps 60S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_cancel_spot_5bps_60s (Spot, 5bps, 60s).",
        depends_on=(Dep(name="net_cancel_spot_5bps_60s", kind="col"),),
        feature_id=5031,
    ),

    # --- Spot, 10bps ---

    FeatureSpec(
        name="net_cancel_persist_spot_10bps_15s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Spot",
            "input_col": "net_cancel_spot_10bps_15s",
            "window_s": "15",
        },
        label="Net Cancel Persist Spot 10Bps 15S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_cancel_spot_10bps_15s (Spot, 10bps, 15s).",
        depends_on=(Dep(name="net_cancel_spot_10bps_15s", kind="col"),),
        feature_id=5032,
    ),

    FeatureSpec(
        name="net_cancel_persist_spot_10bps_60s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Spot",
            "input_col": "net_cancel_spot_10bps_60s",
            "window_s": "60",
        },
        label="Net Cancel Persist Spot 10Bps 60S (Binance)",
        group="Dynamics",
        description="Directional persistence of net_cancel_spot_10bps_60s (Spot, 10bps, 60s).",
        depends_on=(Dep(name="net_cancel_spot_10bps_60s", kind="col"),),
        feature_id=5033,
    ),

]