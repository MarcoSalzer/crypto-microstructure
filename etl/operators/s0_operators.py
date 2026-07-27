# ==============================================================================
# S0 Operator Registry — Simplified, Binance-Only
#
# ARCHITECTURAL PRINCIPLES (S0 STAGE):
#
#   • S0 features are strictly bucket-level (stateless).
#   • No rolling windows.
#   • No multi-venue support (Binance-only pipeline).
#   • All depth is BPS-based (not level-count based).
#   • Depth is notional-only (px * qty).
#
# DESIGN CONTRACT:
#
#   OperatorSpec defines:
#       - name
#       - input source kind
#       - required params
#       - optional params (with defaults)
#
#   Temporal semantics:
#       - All operators operate on bucketed data.
#       - resample defines bucket size (default "1s").
#       - agg applies only where meaningful (e.g., trades).
#
# ==============================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


# =============================================================================
# Operator Contract Definition
# =============================================================================

@dataclass(frozen=True)
class OperatorSpec:
    """
    Declarative contract for S0 feature operators.

    S0 is bucket-level only:
        • No rolling logic
        • No temporal windows
        • No multi-venue logic

    All features operate on:
        - trades streams
        - lobdeep snapshots
    """
    name: str

    # Expected raw source type
    # ("source:lobdeep",) or ("source:trades",)
    input_kinds: Tuple[str, ...]

    # Mandatory parameters in FeatureSpec.params
    required_params: Tuple[str, ...]

    # Optional parameters with defaults
    optional_params_defaults: Mapping[str, str]

    # Short description for documentation
    description_hint: str


# =============================================================================
# S0 Operator Registry
# =============================================================================

S0_OPERATORS: Dict[str, OperatorSpec] = {

    # =====================================================================
    # L2 TOP-OF-BOOK (snapshot-level)
    # =====================================================================

    "l2.best_bid": OperatorSpec(
        name="l2.best_bid",
        input_kinds=("source:lobdeep",),
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Best bid price (last snapshot per bucket).",
    ),

    "l2.best_ask": OperatorSpec(
        name="l2.best_ask",
        input_kinds=("source:lobdeep",),
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Best ask price (last snapshot per bucket).",
    ),

    "l2.mid": OperatorSpec(
        name="l2.mid",
        input_kinds=("source:lobdeep",),
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Mid price = (best_bid + best_ask) / 2.",
    ),

    "l2.spread": OperatorSpec(
        name="l2.spread",
        input_kinds=("source:lobdeep",),
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Bid-ask spread = best_ask - best_bid.",
    ),

    # =====================================================================
    # TRADES (bucket aggregation)
    # =====================================================================

    "trades.trade_count": OperatorSpec(
        name="trades.trade_count",
        input_kinds=("source:trades",),
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s", "agg": "count"},
        description_hint="Number of trades per bucket.",
    ),

    "trades.volume": OperatorSpec(
        name="trades.volume",
        input_kinds=("source:trades",),
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s", "agg": "sum"},
        description_hint="Executed base volume (sum of qty) per bucket.",
    ),

    "trades.notional": OperatorSpec(
        name="trades.notional",
        input_kinds=("source:trades",),
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s", "agg": "sum"},
        description_hint="Total executed notional (sum of qty*price) per bucket.",
    ),

    # ---------------------------------------------------------------------
    # Volume-based taker flow (base units)
    # ---------------------------------------------------------------------

    "trades.taker_buy_volume": OperatorSpec(
        name="trades.taker_buy_volume",
        input_kinds=("source:trades",),
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s", "agg": "sum"},
        description_hint="Buyer-initiated executed base volume (sum of qty) per bucket.",
    ),

    "trades.taker_sell_volume": OperatorSpec(
        name="trades.taker_sell_volume",
        input_kinds=("source:trades",),
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s", "agg": "sum"},
        description_hint="Seller-initiated executed base volume (sum of qty) per bucket.",
    ),

    "trades.signed_volume": OperatorSpec(
        name="trades.signed_volume",
        input_kinds=("source:trades",),
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s", "agg": "sum"},
        description_hint="Signed base volume: +qty (buy), -qty (sell) per bucket.",
    ),

    # ---------------------------------------------------------------------
    # Notional-based taker flow (quote units)
    # ---------------------------------------------------------------------

    "trades.taker_buy_notional": OperatorSpec(
        name="trades.taker_buy_notional",
        input_kinds=("source:trades",),
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s", "agg": "sum"},
        description_hint="Buyer-initiated notional (sum of qty*price) per bucket.",
    ),

    "trades.taker_sell_notional": OperatorSpec(
        name="trades.taker_sell_notional",
        input_kinds=("source:trades",),
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s", "agg": "sum"},
        description_hint="Seller-initiated notional (sum of qty*price) per bucket.",
    ),

    "trades.signed_notional": OperatorSpec(
        name="trades.signed_notional",
        input_kinds=("source:trades",),
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s", "agg": "sum"},
        description_hint="Signed notional: +buy, -sell per bucket.",
    ),

    # =====================================================================
    # BPS DEPTH — METADATA FEATURES
    # =====================================================================

    "depth_bps.max_bps_side": OperatorSpec(
        name="depth_bps.max_bps_side",
        input_kinds=("source:lobdeep",),
        required_params=("market_scope", "side"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Max available BPS depth on one side (bid or ask).",
    ),

    "depth_bps.bps_sym": OperatorSpec(
        name="depth_bps.bps_sym",
        input_kinds=("source:lobdeep",),
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Symmetric available depth = min(max_bps_bid, max_bps_ask).",
    ),

    # =====================================================================
    # BPS DEPTH — FIXED WINDOWS (cross-asset comparable)
    # =====================================================================

    "depth_bps.notional_fixed_bps": OperatorSpec(
        name="depth_bps.notional_fixed_bps",
        input_kinds=("source:lobdeep",),
        required_params=("market_scope", "side", "bps_lo", "bps_hi"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Notional within fixed BPS window [bps_lo, bps_hi].",
    ),

    "depth_bps.imbalance_fixed_bps": OperatorSpec(
        name="depth_bps.imbalance_fixed_bps",
        input_kinds=("source:lobdeep",),
        required_params=("market_scope", "bps_lo", "bps_hi"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Imbalance within fixed BPS window.",
    ),

    # =====================================================================
    # BPS DEPTH — STRUCTURAL (adaptive regime-aware)
    # =====================================================================

    "depth_bps.notional_struct_alpha": OperatorSpec(
        name="depth_bps.notional_struct_alpha",
        input_kinds=("source:lobdeep",),
        required_params=("market_scope", "side", "alpha"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Notional within 0 → alpha * bps_sym.",
    ),

    "depth_bps.imbalance_struct_alpha": OperatorSpec(
        name="depth_bps.imbalance_struct_alpha",
        input_kinds=("source:lobdeep",),
        required_params=("market_scope", "alpha"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Imbalance within 0 → alpha * bps_sym.",
    ),

    # =====================================================================
    # MAX LIQUIDITY DISTANCE — FIXED + STRUCTURAL
    # =====================================================================
    # Distance (in bps) from mid to the price level with maximum notional (px*qty)
    # within a given window. This replaces old per-level / K-based "max_liq_distance"
    # concepts with BPS windows.

    "depth_bps.max_liq_distance_fixed_bps": OperatorSpec(
        name="depth_bps.max_liq_distance_fixed_bps",
        input_kinds=("source:lobdeep",),
        required_params=("market_scope", "side", "bps_lo", "bps_hi"),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "BPS distance from mid to the level with maximum notional (px*qty) "
            "within fixed BPS window [bps_lo, bps_hi] on the given side."
        ),
    ),

    "depth_bps.max_liq_distance_struct_alpha": OperatorSpec(
        name="depth_bps.max_liq_distance_struct_alpha",
        input_kinds=("source:lobdeep",),
        required_params=("market_scope", "side", "alpha"),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "BPS distance from mid to the level with maximum notional (px*qty) "
            "within structural window [0, alpha*bps_sym] on the given side."
        ),
    ),

    # =====================================================================
    # LWP — FIXED BPS (Liquidity-Weighted Price)
    # =====================================================================
    # LWP = sum(px * qty) / sum(qty) within a BPS depth window.
    # Replaces old K-level LWP (K10, K20) with cross-asset comparable
    # BPS-based windows.
    #
    # NaN contract:
    #   - Arrays empty/None/mismatched → NaN
    #   - mid <= 0 or NaN → NaN
    #   - No levels in window → NaN (can't compute weighted average of nothing)
    #   - sum(qty) == 0 in window → NaN (0/0 guard)

    "depth_bps.lwp_fixed_bps": OperatorSpec(
        name="depth_bps.lwp_fixed_bps",
        input_kinds=("source:lobdeep",),
        required_params=("market_scope", "side", "bps_lo", "bps_hi"),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "Liquidity-weighted price on one side within fixed BPS window. "
            "LWP = sum(px*qty) / sum(qty) for levels in [bps_lo, bps_hi]. "
            "NaN if: no levels in window, sum(qty)==0, arrays empty, mid invalid."
        ),
    ),

    "depth_bps.lwp_mid_fixed_bps": OperatorSpec(
        name="depth_bps.lwp_mid_fixed_bps",
        input_kinds=("source:lobdeep",),
        required_params=("market_scope", "bps_lo", "bps_hi"),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "Mid of bid/ask liquidity-weighted prices within fixed BPS window. "
            "LWP_mid = (LWP_bid + LWP_ask) / 2. NaN if either side is NaN."
        ),
    ),

    # =====================================================================
    # LWP — STRUCTURAL (adaptive regime-aware)
    # =====================================================================

    "depth_bps.lwp_struct_alpha": OperatorSpec(
        name="depth_bps.lwp_struct_alpha",
        input_kinds=("source:lobdeep",),
        required_params=("market_scope", "side", "alpha"),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "Liquidity-weighted price on one side within 0 → alpha * bps_sym. "
            "NaN if bps_sym is NaN, no levels in window, or sum(qty)==0."
        ),
    ),

    "depth_bps.lwp_mid_struct_alpha": OperatorSpec(
        name="depth_bps.lwp_mid_struct_alpha",
        input_kinds=("source:lobdeep",),
        required_params=("market_scope", "alpha"),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "Mid of bid/ask LWP within 0 → alpha * bps_sym. "
            "NaN if bps_sym is NaN or either side is NaN."
        ),
    ),
}