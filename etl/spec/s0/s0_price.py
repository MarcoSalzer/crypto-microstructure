# etl/spec/s0/s0_price.py
# ==============================================================================
# S0 Feature Specs: Price
#
# Binance-only pipeline, BTC/ETH/BNB.
# Source: lobdeep (L2 orderbook snapshots)
#
# CONTENTS (all 1s bucket):
#   - Top-of-book: best_ask, best_bid, mid, spread      (8 features)
#   - BPS metadata: bps_sym                              (2 features)
#   - LWP fixed BPS: lwp_{ask|bid|mid}_{1|2|5|10}bps    (24 features)
#   - LWP struct: lwp_{ask|bid|mid}_struct{50|100}       (12 features)
#                                                  Total: 46 features
#
# DESIGN NOTES:
#   - bps_sym lives in Price because it's a price-regime primitive
#     (symmetric depth boundary), not a bookshape metric.
#   - LWP (Liquidity-Weighted Price) is a price primitive: sum(px*qty)/sum(qty).
#     Even though it uses depth data, the output is a price level.
#     Later stages drop raw depth, so LWP must be persisted at S0.
#   - struct75 has been removed from the pipeline (only struct50/100 remain).
#
# Clean S0 Contract:
#   params: market_scope, resample, [side], [bps_lo, bps_hi], [alpha]
#   No venue_scope, window_s, agg, depth_mode.
#
# Feature ID blocks:
#   Top-of-book: 30–37
#   BPS metadata (bps_sym): 54–55
#   LWP fixed: 500–523
#   LWP struct: 600–611
#
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep


def _dep_lobdeep(market: str):
    return (Dep("source:lobdeep", match_params=("market_scope",)),)


# ==============================================================================
# TOP-OF-BOOK (4 types × 2 markets = 8 features)
# ==============================================================================

S0_PRICE_TOB: List[FeatureSpec] = [

    # === BEST ASK ===
    FeatureSpec(
        name="best_ask_fut_1s",
        stage="S0",
        operator="l2.best_ask",
        params={"market_scope": "Futures", "resample": "1s"},
        label="Best Ask (Futures) [1s]",
        group="Price",
        description="Top-of-book best ask price, last lobdeep snapshot per 1s bucket.",
        depends_on=_dep_lobdeep("Futures"),
        feature_id=40,
    ),
    FeatureSpec(
        name="best_ask_spot_1s",
        stage="S0",
        operator="l2.best_ask",
        params={"market_scope": "Spot", "resample": "1s"},
        label="Best Ask (Spot) [1s]",
        group="Price",
        description="Top-of-book best ask price, last lobdeep snapshot per 1s bucket.",
        depends_on=_dep_lobdeep("Spot"),
        feature_id=41,
    ),

    # === BEST BID ===
    FeatureSpec(
        name="best_bid_fut_1s",
        stage="S0",
        operator="l2.best_bid",
        params={"market_scope": "Futures", "resample": "1s"},
        label="Best Bid (Futures) [1s]",
        group="Price",
        description="Top-of-book best bid price, last lobdeep snapshot per 1s bucket.",
        depends_on=_dep_lobdeep("Futures"),
        feature_id=42,
    ),
    FeatureSpec(
        name="best_bid_spot_1s",
        stage="S0",
        operator="l2.best_bid",
        params={"market_scope": "Spot", "resample": "1s"},
        label="Best Bid (Spot) [1s]",
        group="Price",
        description="Top-of-book best bid price, last lobdeep snapshot per 1s bucket.",
        depends_on=_dep_lobdeep("Spot"),
        feature_id=43,
    ),

    # === MID ===
    FeatureSpec(
        name="mid_fut_1s",
        stage="S0",
        operator="l2.mid",
        params={"market_scope": "Futures", "resample": "1s"},
        label="Mid Price (Futures) [1s]",
        group="Price",
        description="Mid price = (best_bid + best_ask) / 2, last lobdeep snapshot per 1s bucket.",
        depends_on=_dep_lobdeep("Futures"),
        feature_id=44,
    ),
    FeatureSpec(
        name="mid_spot_1s",
        stage="S0",
        operator="l2.mid",
        params={"market_scope": "Spot", "resample": "1s"},
        label="Mid Price (Spot) [1s]",
        group="Price",
        description="Mid price = (best_bid + best_ask) / 2, last lobdeep snapshot per 1s bucket.",
        depends_on=_dep_lobdeep("Spot"),
        feature_id=45,
    ),

    # === SPREAD ===
    FeatureSpec(
        name="spread_fut_1s",
        stage="S0",
        operator="l2.spread",
        params={"market_scope": "Futures", "resample": "1s"},
        label="Spread (Futures) [1s]",
        group="Price",
        description="Bid-ask spread = best_ask - best_bid, last lobdeep snapshot per 1s bucket.",
        depends_on=_dep_lobdeep("Futures"),
        feature_id=46,
    ),
    FeatureSpec(
        name="spread_spot_1s",
        stage="S0",
        operator="l2.spread",
        params={"market_scope": "Spot", "resample": "1s"},
        label="Spread (Spot) [1s]",
        group="Price",
        description="Bid-ask spread = best_ask - best_bid, last lobdeep snapshot per 1s bucket.",
        depends_on=_dep_lobdeep("Spot"),
        feature_id=47,
    ),
]


# ==============================================================================
# BPS SYMMETRIC DEPTH (moved from Bookshape — price-regime primitive)
# ==============================================================================

S0_PRICE_BPS_SYM: List[FeatureSpec] = [
    FeatureSpec(
        name="bps_sym_fut_1s",
        stage="S0",
        operator="depth_bps.bps_sym",
        params={"market_scope": "Futures", "resample": "1s"},
        label="BPS Symmetric Depth (Futures) [1s]",
        group="Price",
        description="min(max_bps_bid, max_bps_ask). Symmetric depth boundary for struct windows.",
        depends_on=_dep_lobdeep("Futures"),
        feature_id=48,
    ),
    FeatureSpec(
        name="bps_sym_spot_1s",
        stage="S0",
        operator="depth_bps.bps_sym",
        params={"market_scope": "Spot", "resample": "1s"},
        label="BPS Symmetric Depth (Spot) [1s]",
        group="Price",
        description="min(max_bps_bid, max_bps_ask). Symmetric depth boundary for struct windows.",
        depends_on=_dep_lobdeep("Spot"),
        feature_id=49,
    ),
]


# ==============================================================================
# LWP — FIXED BPS (Liquidity-Weighted Price)
# ==============================================================================
# LWP = sum(px * qty) / sum(qty) within a BPS depth window.
# Persisted at S0 because raw depth is dropped later.
# NaN if: no levels in window, sum(qty)==0, arrays empty, mid invalid.

def _fixed_bps_lwp_specs() -> List[FeatureSpec]:
    specs = []
    fid = 500

    for bps in (1, 2, 5, 10):
        # --- bid / ask side LWP ---
        for side in ("bid", "ask"):
            for market, market_label, market_key in (
                ("Futures", "Futures", "fut"),
                ("Spot", "Spot", "spot"),
            ):
                specs.append(FeatureSpec(
                    name=f"lwp_{side}_{bps}bps_{market_key}_1s",
                    stage="S0",
                    operator="depth_bps.lwp_fixed_bps",
                    params={
                        "market_scope": market,
                        "resample": "1s",
                        "side": side,
                        "bps_lo": "0",
                        "bps_hi": str(bps),
                    },
                    label=f"LWP {side.title()} 0-{bps}bps ({market_label}) [1s]",
                    group="Price",
                    description=(
                        f"Liquidity-weighted {side} price within 0-{bps} bps from mid. "
                        f"LWP = sum(px*qty) / sum(qty)."
                    ),
                    depends_on=_dep_lobdeep(market),
                    feature_id=fid,
                ))
                fid += 1

        # --- mid LWP = (lwp_bid + lwp_ask) / 2 ---
        for market, market_label, market_key in (
            ("Futures", "Futures", "fut"),
            ("Spot", "Spot", "spot"),
        ):
            specs.append(FeatureSpec(
                name=f"lwp_mid_{bps}bps_{market_key}_1s",
                stage="S0",
                operator="depth_bps.lwp_mid_fixed_bps",
                params={
                    "market_scope": market,
                    "resample": "1s",
                    "bps_lo": "0",
                    "bps_hi": str(bps),
                },
                label=f"LWP Mid 0-{bps}bps ({market_label}) [1s]",
                group="Price",
                description=(
                    f"Mid of bid/ask liquidity-weighted prices within 0-{bps} bps. "
                    f"LWP_mid = (LWP_bid + LWP_ask) / 2."
                ),
                depends_on=_dep_lobdeep(market),
                feature_id=fid,
            ))
            fid += 1

    return specs


# ==============================================================================
# LWP — STRUCTURAL (adaptive, regime-aware)
# ==============================================================================
# Only struct50 and struct100 (struct75 removed from pipeline).

def _struct_lwp_specs() -> List[FeatureSpec]:
    specs = []
    fid = 600

    for alpha, alpha_label in ((0.5, "50"), (1.0, "100")):
        # --- bid / ask side LWP ---
        for side in ("bid", "ask"):
            for market, market_label, market_key in (
                ("Futures", "Futures", "fut"),
                ("Spot", "Spot", "spot"),
            ):
                specs.append(FeatureSpec(
                    name=f"lwp_{side}_struct{alpha_label}_{market_key}_1s",
                    stage="S0",
                    operator="depth_bps.lwp_struct_alpha",
                    params={
                        "market_scope": market,
                        "resample": "1s",
                        "side": side,
                        "alpha": str(alpha),
                    },
                    label=f"LWP {side.title()} struct{alpha_label}% ({market_label}) [1s]",
                    group="Price",
                    description=(
                        f"Liquidity-weighted {side} price within 0 to {alpha}*bps_sym."
                    ),
                    depends_on=_dep_lobdeep(market),
                    feature_id=fid,
                ))
                fid += 1

        # --- mid LWP ---
        for market, market_label, market_key in (
            ("Futures", "Futures", "fut"),
            ("Spot", "Spot", "spot"),
        ):
            specs.append(FeatureSpec(
                name=f"lwp_mid_struct{alpha_label}_{market_key}_1s",
                stage="S0",
                operator="depth_bps.lwp_mid_struct_alpha",
                params={
                    "market_scope": market,
                    "resample": "1s",
                    "alpha": str(alpha),
                },
                label=f"LWP Mid struct{alpha_label}% ({market_label}) [1s]",
                group="Price",
                description=(
                    f"Mid of bid/ask liquidity-weighted prices within 0 to {alpha}*bps_sym."
                ),
                depends_on=_dep_lobdeep(market),
                feature_id=fid,
            ))
            fid += 1

    return specs


# ==============================================================================
# FINAL ASSEMBLY
# ==============================================================================

S0_PRICE_FEATURES: List[FeatureSpec] = (
    S0_PRICE_TOB
    + S0_PRICE_BPS_SYM
    + _fixed_bps_lwp_specs()
    + _struct_lwp_specs()
)