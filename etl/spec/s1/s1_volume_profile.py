# etl/spec/s1/s1_volume_profile.py
# ==============================================================================
# S1 Feature Specs: Volume Profile
#
# Binance-only pipeline | Source: S0 features + VP artefact from
# generate_volume_profile.py (joined via _join_volume_profile()).
# 12 features | Feature IDs: 1414-1425
#
# PURPOSE:
#   Derived signals from pre-computed Volume-Profile levels (POC / VAH / VAL)
#   across three rolling windows (60m, 240m, 1d). The level columns and the
#   POC-migration columns are NOT defined as FeatureSpec here — they are
#   PASSTHROUGH columns injected by the VP context join:
#
#     poc_{60m,240m,1d}_fut
#     vah_{60m,240m,1d}_fut
#     val_{60m,240m,1d}_fut
#     poc_migration_{60m,240m,1d}_bps_fut  (precomputed in the VP script;
#                                           kept passthrough because the hot-
#                                           path buffer cannot hold the 12h
#                                           of poc_1d_fut history required
#                                           for the 1d migration shift.)
#
# STRUCTURE:
#   (A) 9 BPS-distance features (range.dist_to_level_bps):
#       dist_to_{poc,vah,val}_{60m,240m,1d}_bps_fut
#       Sign = mid - level (positive if above).
#
#   (B) 3 categorical features (derived.price_vs_va):
#       price_vs_va_{60m,240m,1d}_fut
#       Values: 2 if price > VAH, 0 if price < VAL, 1 inside VA.
#
# NaN BEHAVIOUR:
#   If the VP parquet is missing, the passthrough columns are NaN and
#   all 12 features here degrade gracefully to NaN.
#
# NAMING CONVENTION:
#   dist_to_{poc|vah|val}_{TF}_bps_fut  (distances)
#   price_vs_va_{TF}_fut                (categorical)
#
#               POC-migration moved to passthrough per plan §6.2 — rationale:
#               the 1d-migration needs shift_s=43200s = 12h of historical POC
#               values, which exceeds the hot-path FeatureBuffer capacity
#               (MAX_WINDOW_S = 3600). Precomputing in the VP script and
#               consuming as PASSTHROUGH is much simpler than adding a
#               secondary slow-tick buffer to the hot path.
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep


S1_VOLUME_PROFILE_FEATURES: List[FeatureSpec] = [

    # =========================================================================
    # (A) BPS-distance features (9) — range.dist_to_level_bps
    #     (mid_fut - level) / mid_fut * 10000
    # =========================================================================

    # --- Distance to POC per TF ---
    FeatureSpec(
        name="dist_to_poc_60m_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Volume Profile",
        description=(
            "BPS distance from mid_fut to poc_60m_fut "
            "(Point-of-Control of the last 60m volume profile)."
        ),
        depends_on=(
            Dep(name="mid_fut_1s", kind="col"),
            Dep(name="poc_60m_fut", kind="col"),
        ),
        feature_id=1414,
    ),
    FeatureSpec(
        name="dist_to_poc_240m_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Volume Profile",
        description="BPS distance from mid_fut to poc_240m_fut.",
        depends_on=(
            Dep(name="mid_fut_1s",  kind="col"),
            Dep(name="poc_240m_fut", kind="col"),
        ),
        feature_id=1415,
    ),
    FeatureSpec(
        name="dist_to_poc_1d_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Volume Profile",
        description="BPS distance from mid_fut to poc_1d_fut (daily POC).",
        depends_on=(
            Dep(name="mid_fut_1s", kind="col"),
            Dep(name="poc_1d_fut", kind="col"),
        ),
        feature_id=1416,
    ),

    # --- Distance to VAH per TF ---
    FeatureSpec(
        name="dist_to_vah_60m_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Volume Profile",
        description="BPS distance from mid_fut to vah_60m_fut (Value-Area High).",
        depends_on=(
            Dep(name="mid_fut_1s", kind="col"),
            Dep(name="vah_60m_fut", kind="col"),
        ),
        feature_id=1417,
    ),
    FeatureSpec(
        name="dist_to_vah_240m_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Volume Profile",
        description="BPS distance from mid_fut to vah_240m_fut.",
        depends_on=(
            Dep(name="mid_fut_1s",  kind="col"),
            Dep(name="vah_240m_fut", kind="col"),
        ),
        feature_id=1418,
    ),
    FeatureSpec(
        name="dist_to_vah_1d_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Volume Profile",
        description="BPS distance from mid_fut to vah_1d_fut.",
        depends_on=(
            Dep(name="mid_fut_1s", kind="col"),
            Dep(name="vah_1d_fut", kind="col"),
        ),
        feature_id=1419,
    ),

    # --- Distance to VAL per TF ---
    FeatureSpec(
        name="dist_to_val_60m_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Volume Profile",
        description="BPS distance from mid_fut to val_60m_fut (Value-Area Low).",
        depends_on=(
            Dep(name="mid_fut_1s", kind="col"),
            Dep(name="val_60m_fut", kind="col"),
        ),
        feature_id=1420,
    ),
    FeatureSpec(
        name="dist_to_val_240m_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Volume Profile",
        description="BPS distance from mid_fut to val_240m_fut.",
        depends_on=(
            Dep(name="mid_fut_1s",  kind="col"),
            Dep(name="val_240m_fut", kind="col"),
        ),
        feature_id=1421,
    ),
    FeatureSpec(
        name="dist_to_val_1d_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Volume Profile",
        description="BPS distance from mid_fut to val_1d_fut.",
        depends_on=(
            Dep(name="mid_fut_1s", kind="col"),
            Dep(name="val_1d_fut", kind="col"),
        ),
        feature_id=1422,
    ),

    # =========================================================================
    # (B) Categorical price-vs-VA features (3) — derived.price_vs_va
    #     Values: 2 if mid > VAH, 0 if mid < VAL, 1 inside VA.
    # =========================================================================

    FeatureSpec(
        name="price_vs_va_60m_fut",
        stage="S1",
        operator="derived.price_vs_va",
        params={"market_scope": "Futures", "resample": "1s"},
        group="Volume Profile",
        description=(
            "Categorical: 2 if mid_fut > vah_60m_fut, 0 if mid_fut < val_60m_fut, "
            "else 1 (inside the 60m Value Area)."
        ),
        depends_on=(
            Dep(name="mid_fut_1s", kind="col"),
            Dep(name="vah_60m_fut", kind="col"),
            Dep(name="val_60m_fut", kind="col"),
        ),
        feature_id=1423,
    ),
    FeatureSpec(
        name="price_vs_va_240m_fut",
        stage="S1",
        operator="derived.price_vs_va",
        params={"market_scope": "Futures", "resample": "1s"},
        group="Volume Profile",
        description=(
            "Categorical: 2 if mid_fut > vah_240m_fut, 0 if mid_fut < val_240m_fut, "
            "else 1 (inside the 240m Value Area)."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",   kind="col"),
            Dep(name="vah_240m_fut", kind="col"),
            Dep(name="val_240m_fut", kind="col"),
        ),
        feature_id=1424,
    ),
    FeatureSpec(
        name="price_vs_va_1d_fut",
        stage="S1",
        operator="derived.price_vs_va",
        params={"market_scope": "Futures", "resample": "1s"},
        group="Volume Profile",
        description=(
            "Categorical: 2 if mid_fut > vah_1d_fut, 0 if mid_fut < val_1d_fut, "
            "else 1 (inside the 1d Value Area)."
        ),
        depends_on=(
            Dep(name="mid_fut_1s", kind="col"),
            Dep(name="vah_1d_fut", kind="col"),
            Dep(name="val_1d_fut", kind="col"),
        ),
        feature_id=1425,
    ),
]