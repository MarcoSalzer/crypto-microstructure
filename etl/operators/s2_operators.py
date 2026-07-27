# ==============================================================================
# S2 Operator Registry — Complete registry of all operators used by S2 specs.
#
# 51 operators | Synced with all S2 spec files.
#
# CONVENTION:
#   n_input_cols = 0 means variable arity (operator handles any dep count).
#   n_input_cols > 0 is a hard constraint checked by the engine.
# ==============================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


@dataclass(frozen=True)
class S2OperatorSpec:
    name: str
    n_input_cols: int
    required_params: Tuple[str, ...]
    optional_params_defaults: Mapping[str, str]
    description_hint: str


S2_OPERATORS: Dict[str, S2OperatorSpec] = {

    # =====================================================================
    # ABSORPTION
    # =====================================================================

    "l2.absorb_refill_ask": S2OperatorSpec(
        name="l2.absorb_refill_ask",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Refill-weighted sell pressure: taker_sell_vol * add_rate_ask.",
    ),

    "l2.absorb_refill_bid": S2OperatorSpec(
        name="l2.absorb_refill_bid",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Refill-weighted buy pressure: taker_buy_vol * add_rate_bid.",
    ),

    "l2.aggr_absorp_ratio_ask": S2OperatorSpec(
        name="l2.aggr_absorp_ratio_ask",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"eps": "1e-12"},
        description_hint="Buy aggression vs ask liquidity: taker_buy_vol / (depth_ask + eps).",
    ),

    "l2.aggr_absorp_ratio_bid": S2OperatorSpec(
        name="l2.aggr_absorp_ratio_bid",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"eps": "1e-12"},
        description_hint="Sell aggression vs bid liquidity: taker_sell_vol / (depth_bid + eps).",
    ),

    "trades.trade_absorption_ratio_1s": S2OperatorSpec(
        name="trades.trade_absorption_ratio_1s",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"eps": "1e-12"},
        description_hint="|ret| / (volume + eps). Price impact denominator.",
    ),

    # =====================================================================
    # ACTIVITY / QUEUE
    # =====================================================================

    "l2.queue_imbalance_1s": S2OperatorSpec(
        name="l2.queue_imbalance_1s",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"eps": "1e-12"},
        description_hint="(depth_bid - depth_ask) / (depth_bid + depth_ask + eps).",
    ),

    "l2.queue_pressure_log_1s": S2OperatorSpec(
        name="l2.queue_pressure_log_1s",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"eps": "1e-12"},
        description_hint="log((depth_bid + eps) / (depth_ask + eps)).",
    ),

    "deriv.spot_fut_taker_activity_share_1s": S2OperatorSpec(
        name="deriv.spot_fut_taker_activity_share_1s",
        n_input_cols=4,
        required_params=(),
        optional_params_defaults={},
        description_hint="spot_act / (spot_act + fut_act) per 1s.",
    ),

    "derived.participation_rate_1s": S2OperatorSpec(
        name="derived.participation_rate_1s",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="(buy+sell vol) / EWMA(buy+sell vol, hl=3600s).",
    ),

    # =====================================================================
    # DEVIATION / BASIS / CROSS-MARKET
    # =====================================================================

    "deriv.basis_mid": S2OperatorSpec(
        name="deriv.basis_mid",
        n_input_cols=2,
        required_params=(),
        optional_params_defaults={"resample": "1s"},
        description_hint="Rolling mean of (mid_fut - mid_spot) over 60s.",
    ),

    "derived.basis_vwap": S2OperatorSpec(
        name="derived.basis_vwap",
        n_input_cols=2,
        required_params=(),
        optional_params_defaults={"resample": "1s"},
        description_hint="vwap_fut - vwap_spot per 1s bucket.",
    ),

    "derived.price_deviation_bps": S2OperatorSpec(
        name="derived.price_deviation_bps",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="(mid - vwap) / mid * 10000 in bps.",
    ),

    "deriv.queue_pressure_log_div": S2OperatorSpec(
        name="deriv.queue_pressure_log_div",
        n_input_cols=2,
        required_params=(),
        optional_params_defaults={"resample": "1s"},
        description_hint="Futures minus Spot queue pressure log-ratio.",
    ),

    "derived.z_volume_asym": S2OperatorSpec(
        name="derived.z_volume_asym",
        n_input_cols=2,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Z-scored volume asymmetry: (fut - spot) / rolling_std.",
    ),

    # =====================================================================
    # IMBALANCE
    # =====================================================================

    "trades.taker_imbalance_bucket": S2OperatorSpec(
        name="trades.taker_imbalance_bucket",
        n_input_cols=1,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "60s"},
        description_hint="Mean taker imbalance over rolling window from 1s buckets.",
    ),

    # =====================================================================
    # LIQUIDITY EVENTS
    # =====================================================================

    "l2.churn_ask": S2OperatorSpec(
        name="l2.churn_ask",
        n_input_cols=0,  # 2 or 3 deps depending on spec
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Ask-side churn: (add_rate + cancel_rate) / (depth + eps).",
    ),

    "l2.churn_bid": S2OperatorSpec(
        name="l2.churn_bid",
        n_input_cols=0,  # 2 or 3 deps depending on spec
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Bid-side churn: (add_rate + cancel_rate) / (depth + eps).",
    ),

    "l2.net_add_pressure": S2OperatorSpec(
        name="l2.net_add_pressure",
        n_input_cols=0,  # 2 or 4 deps depending on spec
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="(add_bid - cancel_bid) - (add_ask - cancel_ask).",
    ),

    "l2.net_cancel_pressure": S2OperatorSpec(
        name="l2.net_cancel_pressure",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="cancel_rate_bid - cancel_rate_ask.",
    ),

    "l2.pull_rate": S2OperatorSpec(
        name="l2.pull_rate",
        n_input_cols=0,  # 2 or 4 deps: [cancel_col, depth_col] or [ask,bid,depth_ask,depth_bid]
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Normalized cancel rate: cancel_col / (depth_col + eps) "
                         "or (cancel_ask+cancel_bid) / (depth_ask+depth_bid + eps).",
    ),

    "l2.refill_rate": S2OperatorSpec(
        name="l2.refill_rate",
        n_input_cols=0,  # 2 or 4 deps: [add_col, depth_col] or [ask,bid,depth_ask,depth_bid]
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Normalized add rate: add_col / (depth_col + eps) "
                         "or (add_ask+add_bid) / (depth_ask+depth_bid + eps).",
    ),

    "l2.refill_rate_behind": S2OperatorSpec(
        name="l2.refill_rate_behind",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Refill rate on the side behind the trade direction.",
    ),

    "l2.cancel_rate_ahead": S2OperatorSpec(
        name="l2.cancel_rate_ahead",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Cancel rate on the side ahead of the trade direction.",
    ),

    "l2.cancel_rate_behind": S2OperatorSpec(
        name="l2.cancel_rate_behind",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Cancel rate on the side behind the trade direction.",
    ),

    # =====================================================================
    # ROLLING AGGREGATION
    # =====================================================================

    "derived.roll_mean": S2OperatorSpec(
        name="derived.roll_mean",
        n_input_cols=0,  # 1 or 2 deps depending on spec
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Rolling mean of input column(s) over window_s seconds.",
    ),

    "derived.roll_sum": S2OperatorSpec(
        name="derived.roll_sum",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Rolling sum of input column over window_s seconds.",
    ),

    "derived.roll_median": S2OperatorSpec(
        name="derived.roll_median",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Rolling median of input column over window_s seconds.",
    ),

    # =====================================================================
    # ROLLING STATISTICS
    # =====================================================================

    "derived.mad": S2OperatorSpec(
        name="derived.mad",
        n_input_cols=0,  # 1 or 2 deps depending on spec
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Median Absolute Deviation: median(|x - median(x)|).",
    ),

    "derived.robust_zscore": S2OperatorSpec(
        name="derived.robust_zscore",
        n_input_cols=0,  # 1+ deps
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="(x - median) / (1.4826 * MAD + eps). Robust z-score.",
    ),

    "derived.shock_detect": S2OperatorSpec(
        name="derived.shock_detect",
        n_input_cols=0,  # 3 deps, but resolved by prefix (raw/median_/mad_)
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Shock detector: (x - median) / (mad + eps).",
    ),

    "derived.shock": S2OperatorSpec(
        name="derived.shock",
        n_input_cols=2,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Shock signal: uses pre-computed median/MAD.",
    ),

    # =====================================================================
    # TEMPORAL DERIVATIVES
    # =====================================================================

    "derived.d1": S2OperatorSpec(
        name="derived.d1",
        n_input_cols=1,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="First temporal difference: x[t] - x[t-1].",
    ),

    "derived.d2": S2OperatorSpec(
        name="derived.d2",
        n_input_cols=0,  # 1 or 2 deps depending on spec
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Second temporal difference: d1[t] - d1[t-1].",
    ),

    # =====================================================================
    # ARITHMETIC
    # =====================================================================

    "derived.sub": S2OperatorSpec(
        name="derived.sub",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Subtraction: col_a - col_b.",
    ),

    "derived.product": S2OperatorSpec(
        name="derived.product",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Element-wise product: col_a * col_b. Used for alignment/interaction features.",
    ),

    "derived.ratio": S2OperatorSpec(
        name="derived.ratio",
        n_input_cols=0,  # 1, 2, or 4 deps
        required_params=("market_scope",),
        optional_params_defaults={"eps": "1e-12"},
        description_hint="Ratio: col_a / col_b (flexible dep count).",
    ),

    "derived.asymmetry": S2OperatorSpec(
        name="derived.asymmetry",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"eps": "1e-12"},
        description_hint="(a - b) / (a + b + eps). Normalized asymmetry.",
    ),

    # =====================================================================
    # PRICE
    # =====================================================================

    "derived.mid_touch_dev": S2OperatorSpec(
        name="derived.mid_touch_dev",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="mid_touch deviation: mid_touch - mid.",
    ),

    "derived.price_acceleration": S2OperatorSpec(
        name="derived.price_acceleration",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="d2(price) / rolling_std. Price momentum acceleration.",
    ),

    "derived.ret_vwap": S2OperatorSpec(
        name="derived.ret_vwap",
        n_input_cols=2,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Rolling return of VWAP.",
    ),

    "derived.z_rv": S2OperatorSpec(
        name="derived.z_rv",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Z-scored realized volatility.",
    ),

    # =====================================================================
    # PERSISTENCE / AUTOCORRELATION
    # =====================================================================

    "derived.autocorr": S2OperatorSpec(
        name="derived.autocorr",
        n_input_cols=0,  # 1 or 4 deps
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Rolling lag-1 autocorrelation. High = trending.",
    ),

    # =====================================================================
    # IMPACT
    # =====================================================================

    "derived.impact_per_liquidity": S2OperatorSpec(
        name="derived.impact_per_liquidity",
        n_input_cols=0,  # 2–4 deps depending on spec
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="|ret| * vol / depth. Market impact per liquidity.",
    ),

    "derived.impact_per_signed": S2OperatorSpec(
        name="derived.impact_per_signed",
        n_input_cols=0,  # 2 or 3 deps
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="ret / (signed_vol + eps). Signed market impact.",
    ),

    # =====================================================================
    # META / REGIME
    # =====================================================================

    "derived.breakout_regime_flag": S2OperatorSpec(
        name="derived.breakout_regime_flag",
        n_input_cols=2,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Binary breakout flag: 1 if price outside N*MAD band.",
    ),

    "derived.dir_consistency": S2OperatorSpec(
        name="derived.dir_consistency",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Directional consistency over rolling window.",
    ),

    "derived.unidir_ratio": S2OperatorSpec(
        name="derived.unidir_ratio",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Fraction of unidirectional ticks in window.",
    ),

    "derived.depth_coherence": S2OperatorSpec(
        name="derived.depth_coherence",
        n_input_cols=0,  # 2–4 deps depending on spec
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Rolling correlation of depth changes bid vs ask.",
    ),

    "derived.depth_slope": S2OperatorSpec(
        name="derived.depth_slope",
        n_input_cols=0,  # 2–4 deps depending on spec
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Depth profile slope across BPS bands.",
    ),

    "derived.depth_curvature": S2OperatorSpec(
        name="derived.depth_curvature",
        n_input_cols=0,  # 3–4 deps depending on spec
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Depth profile curvature (second derivative of depth vs distance).",
    ),
}