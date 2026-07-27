# etl/spec/s0/s0_health_spec.py
# ==============================================================================
# S0 Health Feature Specifications (GLOBAL)
#
# PURPOSE:
#   Declarative spec for all health/data-quality features. One central,
#   global health gate per bucket_dt_utc (no per-venue/market split in output).
#
# ARCHITECTURE CONTEXT:
#   Pipeline: Binance-only, multi-asset (BTC/ETH/BNB).
#   L2 source: lobdeep only (lob20 removed from pipeline).
#   L2 combinations per asset: Binance x {Spot, Futures} = 2 combos.
#
# IMPORTANT:
#   - data_usability_flag and its diagnostics (usability_bad_count_win,
#     usability_bad_ratio_win, usability_max_bad_streak_win,
#     usability_warmup_flag, unusable_reason_code) are computed in
#     s0_context_batch.py via rolling-window logic — NOT via operator dispatch.
#     They do NOT need FeatureSpec entries here.
#   - Health is GLOBALLY aggregated: join key = bucket_dt_utc.
#   - Trades are event-based/sparse, NOT mandatory per bucket for health.
#
# ==============================================================================

from __future__ import annotations

from typing import List

from etl.spec import FeatureSpec, Dep


S0_HEALTH_FEATURES: List[FeatureSpec] = [

    # =========================================================================
    # HEALTH: GLOBAL FLAGS (one row per bucket_dt_utc)
    # =========================================================================

    FeatureSpec(
        name="data_health_flag",
        stage="S0",
        operator="health.data_health",
        params={"resample": "1s"},
        label="Data Health Flag (Global, L2-only)",
        group="Data Health",
        description=(
            "GLOBAL (L2-only): 1 if bucket is healthy across ALL L2 combos "
            "(Binance lobdeep x Spot + Futures = 2 combos). "
            "Criteria: has data, valid exch_ts_ms, no reconnect, no gap >2 buckets, "
            "no crossed book. Trades are excluded from global health."
        ),
        depends_on=(
            Dep("source:lob_deep", match_params=()),
        ),
        feature_id=30,
    ),

    FeatureSpec(
        name="data_health_flag_soft",
        stage="S0",
        operator="health.data_health_soft",
        params={"resample": "1s", "missing_budget": "1"},
        label="Data Health Flag Soft (B1a, Global)",
        group="Data Health",
        description=(
            "GLOBAL (B1a soft health): 1 if bucket passes relaxed health check. "
            "Allows up to missing_budget L2 combos to be missing per bucket. "
            "With 2 combos (Binance lobdeep Spot + Futures), budget=1 means "
            "at most one side can be missing. Computed in s0_context_batch.py."
        ),
        depends_on=(
            Dep("source:lob_deep", match_params=()),
        ),
        feature_id=31,
    ),

    FeatureSpec(
        name="l2_coverage_flag",
        stage="S0",
        operator="health.l2_coverage",
        params={"resample": "1s"},
        label="L2 Coverage Flag (Global)",
        group="Data Health",
        description=(
            "GLOBAL: 1 if ANY L2 snapshot exists for this bucket across Binance "
            "lobdeep (Spot or Futures) with depth_actual >= 1."
        ),
        depends_on=(
            Dep("source:lob_deep", match_params=()),
        ),
        feature_id=32,
    ),

    # -------------------------------------------------------------------------
    # Depth gate (lobdeep is the only L2 source in the Binance pipeline)
    # -------------------------------------------------------------------------

    FeatureSpec(
        name="depth_lobdeep_global",
        stage="S0",
        operator="health.depth_availability",
        params={"resample": "1s"},
        label="Depth Availability (lobdeep, Global)",
        group="Data Health",
        description=(
            "GLOBAL: MIN(depth_actual) across lobdeep (Binance Spot + Futures). "
            "Use this gate for all depth-dependent features. "
            "Expected values: ~1000 (Binance full depth)."
        ),
        depends_on=(
            Dep("source:lob_deep", match_params=()),
        ),
        feature_id=33,
    ),

    FeatureSpec(
        name="lob50_health_flag",
        stage="S0",
        operator="health.depth_gate",
        params={"resample": "1s", "min_depth": "50"},
        label="LOB50 Health Flag (Global)",
        group="Data Health",
        description=(
            "GLOBAL: 1 if depth_lobdeep_global >= 50. Gate for features "
            "requiring deep orderbook (book shape, struct depths, etc.)."
        ),
        depends_on=(
            Dep("source:lob_deep", match_params=()),
        ),
        feature_id=34,
    ),

    FeatureSpec(
        name="trades_coverage_flag",
        stage="S0",
        operator="health.trades_coverage",
        params={"resample": "1s"},
        label="Trades Coverage Flag (Global)",
        group="Data Health",
        description=(
            "GLOBAL: 1 if ANY Binance trades stream produced data in this bucket "
            "(Spot or Futures). Trades are sparse/event-based; missing buckets are "
            "normal and do NOT affect data_health_flag."
        ),
        depends_on=(
            Dep("source:trades", match_params=()),
        ),
        feature_id=35,
    ),
]


# ==============================================================================
# Convenience helpers (re-exported for spec consumers)
# ==============================================================================

def get_health_feature_names() -> List[str]:
    """
    Ordered list of ALL health-related column names in the S0 output.

    Includes:
      - Core flags from S0_HEALTH_FEATURES specs above
      - depth_availability (min across all L2 sources)
      - Diagnostic counters (l2_total_combos, l2_bad_combos, etc.)
      - data_usability_flag + diagnostics (computed in s0_context_batch.py)

    NOTE: depth_lob20_global and lob20_health_flag have been removed
    (lob20 source no longer exists in the Binance-only pipeline).
    """
    return [
        # --- core flags (FeatureSpec-defined) ---
        "data_health_flag",
        "data_health_flag_soft",
        "l2_coverage_flag",
        "depth_availability",
        "depth_lobdeep_global",
        "lob50_health_flag",
        "trades_coverage_flag",
        # --- diagnostics (computed in s0_health.py, no FeatureSpec) ---
        "l2_total_combos",
        "l2_bad_combos",
        "l2_missing_combos",
        "l2_invalid_ts_combos",
        "l2_reconnect_combos",
        "l2_gap_combos",
        "l2_crossed_combos",
        "l2_bad_bitmask",
        "health_reason_code",
        # --- usability (computed in s0_context_batch.py, no FeatureSpec) ---
        "data_usability_flag",
        "usability_bad_ratio_win",
        "usability_bad_count_win",
        "usability_max_bad_streak_win",
        "usability_warmup_flag",
        "unusable_reason_code",
    ]


def get_health_feature_dtypes() -> dict:
    """Return {column_name: dtype} for all health features."""
    return {
        # core flags
        "data_health_flag": "int8",
        "data_health_flag_soft": "int8",
        "l2_coverage_flag": "int8",
        "depth_availability": "int16",
        "depth_lobdeep_global": "int16",
        "lob50_health_flag": "int8",
        "trades_coverage_flag": "int8",
        # diagnostics
        "l2_total_combos": "int16",
        "l2_bad_combos": "int16",
        "l2_missing_combos": "int16",
        "l2_invalid_ts_combos": "int16",
        "l2_reconnect_combos": "int16",
        "l2_gap_combos": "int16",
        "l2_crossed_combos": "int16",
        "l2_bad_bitmask": "int16",
        "health_reason_code": "int8",
        # usability (context_batch)
        "data_usability_flag": "int8",
        "usability_bad_ratio_win": "float32",
        "usability_bad_count_win": "int16",
        "usability_max_bad_streak_win": "int16",
        "usability_warmup_flag": "int8",
        "unusable_reason_code": "int8",
    }