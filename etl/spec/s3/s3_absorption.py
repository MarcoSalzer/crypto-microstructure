# etl/spec/s3/s3_absorption.py
# ========================================================================
# S3 Absorption Features
# Composite absorption analytics that combine trade aggression with orderbook
# replenishment dynamics. These features detect when aggressive taker flow
# overwhelms the market's ability to refill liquidity — a key precursor to
# directional moves.
# 
# Key concepts:
#   - absorb_refill_mid: Midpoint of ask/bid absorption refill rates.
#     Measures average refill-weighted pressure across both sides.
#   - absorption_asymmetry: Normalized difference between ask-side and bid-side
#     absorption volume. Detects directional bias in absorption patterns.
#   - absorption_break / absorption_break_flag: Composite signals that fire when
#     trade absorption ratio is elevated, taker imbalance is directional, and
#     orderbook refill is insufficient — indicating a potential breakout.
#     Computed at two horizons: 15s (short-term burst detection) and
#     60s (sustained regime detection).
#   - trade_absorption_ratio (bps-scoped / time-windowed): TAR extended to
#     specific depth bands and longer rolling windows.
# 
# Dependencies: S2 absorb_refill_ask/bid, trade_absorption_ratio, taker_imbalance,
#               impact_per_signed.
#
# Feature count: 16  (was 14; added 2 × 15s absorption_break variants moved from S4)
# [FIX-3600] Removed all window_s=3600 features (trade_absorption_ratio _3600s rolling means).
#            3600s rolling windows require 1h prior data even with context;
#            produce 99-100% NaN on 1h Parquet files. The 900s variants
#            provide equivalent microstructure regime coverage.
# [MOVED-FROM-S4] absorption_break_fut_15s and absorption_break_spot_15s were
#            previously computed in S4 using derived.absorption_break. They have
#            been moved here because all their deps (absorb_refill, impact_per_signed,
#            taker_imbalance, trade_absorption_ratio) are S2 outputs available at
#            S3 stage. Using s3.absorption_break operator (same logic as _60s variants).
# Feature ID range: see individual entries
# ========================================================================

from typing import List
from etl.spec import FeatureSpec, Dep

S3_ABSORPTION_FEATURES: List[FeatureSpec] = [
    FeatureSpec(
        name="absorb_refill_mid_fut_2bps_1s",
        stage="S3",
        operator="s3.absorb_refill_mid",
        params={"market_scope": "Futures"},
        label="Absorption: absorb_refill_mid_fut_2bps_1s (Futures) [1s] (Binance)",
        group="Absorption",
        description="Midpoint of ask/bid absorption refill: (absorb_refill_ask_fut_2bps_1s + absorb_refill_bid_fut_2bps_1s) / 2.",
        depends_on=(
            Dep(name="absorb_refill_ask_fut_2bps_1s", kind="col"),
            Dep(name="absorb_refill_bid_fut_2bps_1s", kind="col"),
        ),
        feature_id=3000,
    ),
    FeatureSpec(
        name="absorb_refill_mid_fut_5bps_1s",
        stage="S3",
        operator="s3.absorb_refill_mid",
        params={"market_scope": "Futures"},
        label="Absorption: absorb_refill_mid_fut_5bps_1s (Futures) [1s] (Binance)",
        group="Absorption",
        description="Midpoint of ask/bid absorption refill: (absorb_refill_ask_fut_5bps_1s + absorb_refill_bid_fut_5bps_1s) / 2.",
        depends_on=(
            Dep(name="absorb_refill_ask_fut_5bps_1s", kind="col"),
            Dep(name="absorb_refill_bid_fut_5bps_1s", kind="col"),
        ),
        feature_id=3001,
    ),
    FeatureSpec(
        name="absorption_asymmetry_fut_1bps_60s",
        stage="S3",
        operator="s3.absorption_asymmetry",
        params={"market_scope": "Futures"},
        label="Absorption: absorption_asymmetry_fut_1bps_60s (Futures) [60s] (Binance)",
        group="Absorption",
        description="Normalized absorption asymmetry: (ask_vol - bid_vol) / (ask_vol + bid_vol) over rolling window.",
        depends_on=(
            Dep(name="absorption_volume_ask_fut_1bps_60s", kind="col"),
            Dep(name="absorption_volume_bid_fut_1bps_60s", kind="col"),
        ),
        feature_id=3002,
    ),
    FeatureSpec(
        name="absorption_asymmetry_spot_1bps_60s",
        stage="S3",
        operator="s3.absorption_asymmetry",
        params={"market_scope": "Spot"},
        label="Absorption: absorption_asymmetry_spot_1bps_60s (Spot) [60s] (Binance)",
        group="Absorption",
        description="Normalized absorption asymmetry: (ask_vol - bid_vol) / (ask_vol + bid_vol) over rolling window.",
        depends_on=(
            Dep(name="absorption_volume_ask_spot_1bps_60s", kind="col"),
            Dep(name="absorption_volume_bid_spot_1bps_60s", kind="col"),
        ),
        feature_id=3003,
    ),
    FeatureSpec(
        name="absorption_break_flag_fut_1bps_60s",
        stage="S3",
        operator="s3.absorption_break_flag",
        params={"market_scope": "Futures"},
        label="Absorption: absorption_break_flag_fut_1bps_60s (Futures) [60s] (Binance)",
        group="Absorption",
        description="Binary flag for absorption break: high TAR + aligned imbalance + low refill.",
        depends_on=(
            Dep(name="absorb_refill_ask_fut_2bps_1s", kind="col"),
            Dep(name="absorb_refill_bid_fut_2bps_1s", kind="col"),
            Dep(name="impact_per_signed_fut_60s", kind="col"),
            Dep(name="taker_imbalance_fut_60s", kind="col"),
            Dep(name="trade_absorption_ratio_fut_60s", kind="col"),
        ),
        feature_id=3004,
    ),
    FeatureSpec(
        name="absorption_break_flag_spot_1bps_60s",
        stage="S3",
        operator="s3.absorption_break_flag",
        params={"market_scope": "Spot"},
        label="Absorption: absorption_break_flag_spot_1bps_60s (Spot) [60s] (Binance)",
        group="Absorption",
        description="Binary flag for absorption break: high TAR + aligned imbalance + low refill.",
        depends_on=(
            Dep(name="absorb_refill_ask_spot_2bps_1s", kind="col"),
            Dep(name="absorb_refill_bid_spot_2bps_1s", kind="col"),
            Dep(name="impact_per_signed_spot_60s", kind="col"),
            Dep(name="taker_imbalance_spot_60s", kind="col"),
            Dep(name="trade_absorption_ratio_spot_60s", kind="col"),
        ),
        feature_id=3005,
    ),
    FeatureSpec(
        name="absorption_break_fut_60s",
        stage="S3",
        operator="s3.absorption_break",
        params={"market_scope": "Futures"},
        label="Absorption: absorption_break_fut_60s (Futures) [60s] (Binance)",
        group="Absorption",
        description="Continuous absorption break score: composite of TAR, imbalance, and refill metrics.",
        depends_on=(
            Dep(name="absorb_refill_ask_fut_2bps_1s", kind="col"),
            Dep(name="absorb_refill_bid_fut_2bps_1s", kind="col"),
            Dep(name="impact_per_signed_fut_60s", kind="col"),
            Dep(name="taker_imbalance_fut_60s", kind="col"),
            Dep(name="trade_absorption_ratio_fut_60s", kind="col"),
        ),
        feature_id=3006,
    ),
    FeatureSpec(
        name="absorption_break_spot_60s",
        stage="S3",
        operator="s3.absorption_break",
        params={"market_scope": "Spot"},
        label="Absorption: absorption_break_spot_60s (Spot) [60s] (Binance)",
        group="Absorption",
        description="Continuous absorption break score: composite of TAR, imbalance, and refill metrics.",
        depends_on=(
            Dep(name="absorb_refill_ask_spot_2bps_1s", kind="col"),
            Dep(name="absorb_refill_bid_spot_2bps_1s", kind="col"),
            Dep(name="impact_per_signed_spot_60s", kind="col"),
            Dep(name="taker_imbalance_spot_60s", kind="col"),
            Dep(name="trade_absorption_ratio_spot_60s", kind="col"),
        ),
        feature_id=3007,
    ),
    FeatureSpec(
        name="trade_absorption_ratio_fut_2bps_1s",
        stage="S3",
        operator="s3.trade_absorption_ratio_bps",
        params={"market_scope": "Futures", "bps": "2"},
        label="Absorption: trade_absorption_ratio_fut_2bps_1s (Futures) [1s] (Binance)",
        group="Absorption",
        description="Trade absorption ratio scoped to 2bps depth band.",
        depends_on=(Dep(name="trade_absorption_ratio_fut_1s", kind="col"),),
        feature_id=3008,
    ),
    FeatureSpec(
        name="trade_absorption_ratio_fut_5bps_1s",
        stage="S3",
        operator="s3.trade_absorption_ratio_bps",
        params={"market_scope": "Futures", "bps": "5"},
        label="Absorption: trade_absorption_ratio_fut_5bps_1s (Futures) [1s] (Binance)",
        group="Absorption",
        description="Trade absorption ratio scoped to 5bps depth band.",
        depends_on=(Dep(name="trade_absorption_ratio_fut_1s", kind="col"),),
        feature_id=3009,
    ),
    FeatureSpec(
        name="trade_absorption_ratio_fut_60s",
        stage="S3",
        operator="derived.roll_mean",
        params={"market_scope": "Futures", "window_s": "60", "min_periods": 10},
        label="Absorption: trade_absorption_ratio_fut_60s (Futures) [60s] (Binance)",
        group="Absorption",
        description="Rolling mean of trade_absorption_ratio_fut_1s over 60s window.",
        depends_on=(Dep(name="trade_absorption_ratio_fut_1s", kind="col"),),
        feature_id=3010,
    ),
    FeatureSpec(
        name="trade_absorption_ratio_fut_900s",
        stage="S3",
        operator="derived.roll_mean",
        params={"market_scope": "Futures", "window_s": "900", "min_periods": 60},
        label="Absorption: trade_absorption_ratio_fut_900s (Futures) [900s] (Binance)",
        group="Absorption",
        description="Rolling mean of trade_absorption_ratio_fut_1s over 900s window.",
        depends_on=(Dep(name="trade_absorption_ratio_fut_1s", kind="col"),),
        feature_id=3011,
    ),
    FeatureSpec(
        name="trade_absorption_ratio_spot_60s",
        stage="S3",
        operator="derived.roll_mean",
        params={"market_scope": "Spot", "window_s": "60", "min_periods": 10},
        label="Absorption: trade_absorption_ratio_spot_60s (Spot) [60s] (Binance)",
        group="Absorption",
        description="Rolling mean of trade_absorption_ratio_spot_1s over 60s window.",
        depends_on=(Dep(name="trade_absorption_ratio_spot_1s", kind="col"),),
        feature_id=3012,
    ),
    FeatureSpec(
        name="trade_absorption_ratio_spot_900s",
        stage="S3",
        operator="derived.roll_mean",
        params={"market_scope": "Spot", "window_s": "900", "min_periods": 60},
        label="Absorption: trade_absorption_ratio_spot_900s (Spot) [900s] (Binance)",
        group="Absorption",
        description="Rolling mean of trade_absorption_ratio_spot_1s over 900s window.",
        depends_on=(Dep(name="trade_absorption_ratio_spot_1s", kind="col"),),
        feature_id=3013,
    ),

    # === absorption_break 15s — moved from S4 ===
    # Short-horizon (15s) burst detection using impact_per_signed_15s and
    # taker_imbalance_15s as inputs. All deps are S2 outputs, so this
    # can be computed at S3 stage. Uses the same s3.absorption_break operator
    # as the 60s variants above.

    FeatureSpec(
        name="absorption_break_fut_15s",
        stage="S3",
        operator="s3.absorption_break",
        params={"market_scope": "Futures"},
        label="Absorption: absorption_break_fut_15s (Futures) [15s] (Binance)",
        group="Absorption",
        description="Continuous absorption break score with 15s-horizon inputs: composite of TAR, imbalance, and refill metrics.",
        depends_on=(
            Dep(name="absorb_refill_ask_fut_2bps_1s", kind="col"),
            Dep(name="absorb_refill_bid_fut_2bps_1s", kind="col"),
            Dep(name="impact_per_signed_fut_15s", kind="col"),
            Dep(name="taker_imbalance_fut_15s", kind="col"),
            Dep(name="trade_absorption_ratio_fut_60s", kind="col"),
        ),
        feature_id=3014,
    ),

    FeatureSpec(
        name="absorption_break_spot_15s",
        stage="S3",
        operator="s3.absorption_break",
        params={"market_scope": "Spot"},
        label="Absorption: absorption_break_spot_15s (Spot) [15s] (Binance)",
        group="Absorption",
        description="Continuous absorption break score with 15s-horizon inputs: composite of TAR, imbalance, and refill metrics.",
        depends_on=(
            Dep(name="absorb_refill_ask_spot_2bps_1s", kind="col"),
            Dep(name="absorb_refill_bid_spot_2bps_1s", kind="col"),
            Dep(name="impact_per_signed_spot_15s", kind="col"),
            Dep(name="taker_imbalance_spot_15s", kind="col"),
            Dep(name="trade_absorption_ratio_spot_60s", kind="col"),
        ),
        feature_id=3015,
    ),

]