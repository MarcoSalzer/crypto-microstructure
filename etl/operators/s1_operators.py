# ==============================================================================
# S1 Operator Registry — Complete
#
# 65 operators covering all S1 spec modules.
# Every operator string used in etl/spec/s1/*.py MUST appear here.
#
#       EMA/Trend       (4): derived.ema, derived.price_vs_ema_bps,
#                            derived.ema_slope_bps, derived.trend_align
#       Level-Distance  (1): range.dist_to_level_bps
#       Reclaim/Break   (5): derived.above_level, derived.below_level,
#                            derived.reclaim_flag, derived.break_flag_high,
#                            derived.break_flag_low
#       Fibonacci       (1): derived.fib_dist_bps
#       Volume Profile  (2): derived.price_vs_va, derived.poc_migration_bps
#     Note: range.ext_position is REUSED for weekly/monthly range-pos features;
#           no alias operator was introduced. Net: 48 → 65 operators.
#               Removed l2.lwp (broken), l2.max_liq_distance (uncomputable at S1).
#               Net: 45 → 43 operators.
#               (renamed to l2.mid_touch). Added derived.zscore_diff,
#               derived.basis_bps. Corrected n_input_cols from spec audit.
# ==============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple


@dataclass(frozen=True)
class S1OperatorSpec:
    name: str
    n_input_cols: int
    required_params: Tuple[str, ...]
    optional_params_defaults: Mapping[str, str] = field(default_factory=dict)
    description_hint: str = ""


S1_OPERATORS: Dict[str, S1OperatorSpec] = {

    # =====================================================================
    # DERIVED — Generic arithmetic / rolling / statistical
    # =====================================================================

    "derived.add": S1OperatorSpec(
        name="derived.add",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Sum = col_a + col_b.",
    ),

    "derived.sub": S1OperatorSpec(
        name="derived.sub",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Difference = col_a - col_b.",
    ),

    "derived.ratio": S1OperatorSpec(
        name="derived.ratio",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Ratio = col_a / (col_b + eps).",
    ),

    "derived.count_ratio": S1OperatorSpec(
        # [CROSS-DIV-FIX 2026-04-27]
        # Variant of derived.ratio for count/volume inputs that can be
        # legitimately zero in seconds without market activity.
        # Semantics:
        #   0 / 0   -> 0    (no activity in either market)
        #   0 / x   -> 0    (only the other market traded)
        #   x / 0   -> NaN  (genuinely undefined)
        #   x / y   -> clip(x/y, 0, 1e6)
        # Why a separate operator: keeping derived.ratio unchanged preserves
        # behaviour for the 24 other consumers (s2_bookshape, etc.) where
        # 0/0 is genuinely undefined (depths, imbalances are never legitimately 0).
        name="derived.count_ratio",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "Count/volume-aware ratio. 0/0 = 0 (no activity = no divergence), "
            "0/x = 0, x/0 = NaN, normal = clip(x/y, 0, 1e6)."
        ),
    ),

    "derived.share": S1OperatorSpec(
        name="derived.share",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Share = col_a / (col_a + col_b + eps). Range [0, 1].",
    ),

    "derived.roll_mean": S1OperatorSpec(
        name="derived.roll_mean",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Rolling mean of col over window_s seconds.",
    ),

    "derived.roll_sum": S1OperatorSpec(
        name="derived.roll_sum",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Rolling sum of col over window_s seconds. min_periods=window_s.",
    ),

    "derived.robust_zscore": S1OperatorSpec(
        name="derived.robust_zscore",
        n_input_cols=1,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Robust z-score: (x - median) / (1.4826 * MAD + eps). Single column.",
    ),

    "derived.zscore_diff": S1OperatorSpec(
        name="derived.zscore_diff",
        n_input_cols=2,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "z-score of (col_a - col_b): compute diff, then robust z-score over "
            "rolling window_s. Used for 2-input z-score features."
        ),
    ),

    "derived.log_return": S1OperatorSpec(
        name="derived.log_return",
        n_input_cols=1,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Log return = log(col / col.shift(1)). First row -> NaN.",
    ),

    # =====================================================================
    # DERIVED — Price / Cross-Market
    # =====================================================================

    "derived.basis": S1OperatorSpec(
        name="derived.basis",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Basis = mid_fut - mid_spot (USD).",
    ),

    "deriv.basis_mid": S1OperatorSpec(
        name="deriv.basis_mid",
        n_input_cols=2,
        required_params=(),
        optional_params_defaults={"resample": "1s"},
        description_hint="Basis: rolling mean of (mid_fut - mid_spot).",
    ),

    "derived.basis_bps": S1OperatorSpec(
        name="derived.basis_bps",
        n_input_cols=2,
        required_params=(),
        optional_params_defaults={"resample": "1s"},
        description_hint="Basis in BPS: (mid_fut - mid_spot) / (mid_spot + eps) * 10000.",
    ),

    "derived.ret_fwd": S1OperatorSpec(
        name="derived.ret_fwd",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Forward return = col.shift(-window_s) / col - 1.",
    ),

    "deriv.spot_fut_taker_activity_share_1s": S1OperatorSpec(
        name="deriv.spot_fut_taker_activity_share_1s",
        n_input_cols=4,
        required_params=(),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "Spot share = (buy_spot + sell_spot) / "
            "(buy_fut + sell_fut + buy_spot + sell_spot + eps)."
        ),
    ),

    # =====================================================================
    # DERIVED — Activity / Meta
    # =====================================================================

    "derived.participation_rate_1s": S1OperatorSpec(
        name="derived.participation_rate_1s",
        n_input_cols=1,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Participation rate: fraction of seconds with non-zero value.",
    ),

    "derived.range_pct": S1OperatorSpec(
        name="derived.range_pct",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Range percent: (max - min) / (mid + eps) over window.",
    ),

    "derived.range_pos": S1OperatorSpec(
        name="derived.range_pos",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Position within range: (x - min) / (max - min + eps) over window.",
    ),

    # =====================================================================
    # RANGE — External OHLC operators (daily range, injected via OHLC join)
    # These operators receive pre-computed high/low columns from the OHLC
    # parquet (not rolling windows), enabling full-day range features.
    # =====================================================================

    "range.dist_to_high_bps": S1OperatorSpec(
        name="range.dist_to_high_bps",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint=(
            "BPS distance from mid to external high: (high - mid) / mid * 10000. "
            "Deps: [mid_col, high_col]. 0 = at high, positive = below high."
        ),
    ),

    "range.dist_to_low_bps": S1OperatorSpec(
        name="range.dist_to_low_bps",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint=(
            "BPS distance from mid to external low: (mid - low) / mid * 10000. "
            "Deps: [mid_col, low_col]. 0 = at low, positive = above low."
        ),
    ),

    "range.ext_position": S1OperatorSpec(
        name="range.ext_position",
        n_input_cols=3,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint=(
            "Position within external range: (mid - low) / (high - low + eps). "
            "Range [0, 1]. Deps: [mid_col, low_col, high_col]."
        ),
    ),

    "range.ext_range_bps": S1OperatorSpec(
        name="range.ext_range_bps",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint=(
            "External range in BPS: (high - low) / mid * 10000. "
            "Deps: [high_col, low_col]. Volatility proxy for the range period."
        ),
    ),

    # =====================================================================
    # TRADES — Trade-based operators
    # =====================================================================

    "trades.avg_trade_size": S1OperatorSpec(
        name="trades.avg_trade_size",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Average trade size = volume / trade_count. NaN if count == 0.",
    ),

    "trades.taker_imbalance": S1OperatorSpec(
        name="trades.taker_imbalance",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Taker imbalance = (buy - sell) / (buy + sell + eps). Range [-1, 1].",
    ),

    "trades.vwap": S1OperatorSpec(
        name="trades.vwap",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="VWAP = notional / volume. NaN if volume == 0.",
    ),

    # =====================================================================
    # L2 — Bookshape
    # =====================================================================

    "l2.book_asymmetry": S1OperatorSpec(
        name="l2.book_asymmetry",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "(depth_bid - depth_ask) / (depth_bid + depth_ask + eps). "
            "Deps: [bid, ask]. Range [-1, 1]."
        ),
    ),

    "l2.depth_gradient_ask": S1OperatorSpec(
        name="l2.depth_gradient_ask",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Ask depth gradient: (outer - inner) / outer. Deps: [inner_depth, outer_depth].",
    ),

    "l2.depth_gradient_bid": S1OperatorSpec(
        name="l2.depth_gradient_bid",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Bid depth gradient: (outer - inner) / outer. Deps: [inner_depth, outer_depth].",
    ),

    "l2.liq_cluster_asymmetry": S1OperatorSpec(
        name="l2.liq_cluster_asymmetry",
        n_input_cols=2,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Rolling CV difference between bid/ask depth. Cluster asymmetry.",
    ),

    "l2.liq_concentration_ask": S1OperatorSpec(
        name="l2.liq_concentration_ask",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Ask liquidity concentration: inner/outer depth ratio. Deps: [inner_depth, outer_depth].",
    ),

    "l2.liq_concentration_bid": S1OperatorSpec(
        name="l2.liq_concentration_bid",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Bid liquidity concentration: inner/outer depth ratio. Deps: [inner_depth, outer_depth].",
    ),

    "l2.liq_sum": S1OperatorSpec(
        name="l2.liq_sum",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Total resting liquidity: depth_bid + depth_ask.",
    ),

    # NOTE: l2.max_liq_distance removed — requires per-price-level order book data
    # not available at S1. The S0 max_liq_distance features are computed correctly from raw data.

    # =====================================================================
    # L2 — Price
    # =====================================================================

    # NOTE: l2.lwp removed — lwp_fut/spot_1s features were broken (ask/bid ratio ≈ 1.0).
    # Use lwp_mid_*_1s from S0 instead. True LWP requires best-level quantities.

    "l2.mid_touch": S1OperatorSpec(
        name="l2.mid_touch",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Mid-touch price: (bid + ask) / 2.",
    ),

    # =====================================================================
    # L2 — Pressure
    # =====================================================================

    "l2.queue_imbalance_1s": S1OperatorSpec(
        name="l2.queue_imbalance_1s",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "(depth_bid - depth_ask) / (depth_bid + depth_ask + eps). "
            "Deps: [bid, ask]."
        ),
    ),

    "l2.queue_pressure": S1OperatorSpec(
        name="l2.queue_pressure",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s", "window_s": "1"},
        description_hint=(
            "(depth_bid - depth_ask) / (depth_bid + depth_ask). Linear pressure. "
            "Deps: [bid, ask]. Optional rolling via window_s."
        ),
    ),

    "l2.queue_pressure_log_1s": S1OperatorSpec(
        name="l2.queue_pressure_log_1s",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "log((depth_bid + eps) / (depth_ask + eps)). Log pressure. "
            "Deps: [bid, ask]."
        ),
    ),

    "l2.net_pressure": S1OperatorSpec(
        name="l2.net_pressure",
        n_input_cols=6,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "(add_bid - cancel_bid) - (add_ask - cancel_ask). Net order flow pressure. "
            "Deps: [add_bid, cancel_bid, add_ask, cancel_ask, depth_ask, depth_bid]."
        ),
    ),

    # =====================================================================
    # L2 — Absorption / Liquidity Events
    # =====================================================================

    "l2.absorb_refill_ask": S1OperatorSpec(
        name="l2.absorb_refill_ask",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Ask absorption-refill ratio.",
    ),

    "l2.absorb_refill_bid": S1OperatorSpec(
        name="l2.absorb_refill_bid",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Bid absorption-refill ratio.",
    ),

    "l2.aggr_absorp_ratio_ask": S1OperatorSpec(
        name="l2.aggr_absorp_ratio_ask",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Ask aggressor absorption ratio.",
    ),

    "l2.aggr_absorp_ratio_bid": S1OperatorSpec(
        name="l2.aggr_absorp_ratio_bid",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Bid aggressor absorption ratio.",
    ),

    "l2.add_rate_ask": S1OperatorSpec(
        name="l2.add_rate_ask",
        n_input_cols=1,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Ask liquidity added = max(0, depth(t) - depth(t-1)).",
    ),

    "l2.add_rate_bid": S1OperatorSpec(
        name="l2.add_rate_bid",
        n_input_cols=1,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Bid liquidity added = max(0, depth(t) - depth(t-1)).",
    ),

    "l2.cancel_rate_ask": S1OperatorSpec(
        name="l2.cancel_rate_ask",
        n_input_cols=1,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Ask liquidity removed = max(0, depth(t-1) - depth(t)).",
    ),

    "l2.cancel_rate_bid": S1OperatorSpec(
        name="l2.cancel_rate_bid",
        n_input_cols=1,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Bid liquidity removed = max(0, depth(t-1) - depth(t)).",
    ),

    "l2.fill_rate_ahead": S1OperatorSpec(
        name="l2.fill_rate_ahead",
        n_input_cols=4,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Fill rate ahead of best: volume absorbed within BPS band.",
    ),

    "l2.pull_rate": S1OperatorSpec(
        name="l2.pull_rate",
        n_input_cols=4,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Pull rate: cancel_rate / (depth + eps). Liquidity withdrawal intensity.",
    ),

    "l2.refill_rate": S1OperatorSpec(
        name="l2.refill_rate",
        n_input_cols=4,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Refill rate: add_rate / (depth + eps). Liquidity replenishment intensity.",
    ),

    # =====================================================================
    # L2 — Activity / Meta
    # =====================================================================

    "l2.l2_update_count": S1OperatorSpec(
        name="l2.l2_update_count",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="L2 order book update count per bucket.",
    ),

    # =====================================================================
    # FORWARD-LOOKING — Cold-path only (Hot-path filters these out)
    # These operators use look-ahead on the mid price; they are meaningful
    # in batch training but cannot be computed in real-time. Phase 1 of the
    # 119-feature expansion (2026-04-17).
    # =====================================================================

    "derived.mae_fwd": S1OperatorSpec(
        name="derived.mae_fwd",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "Max Adverse Excursion (long-perspective): "
            "(mid[t] - min(mid[t..t+w])) / mid[t] * 10000. In BPS."
        ),
    ),

    "derived.mfe_fwd": S1OperatorSpec(
        name="derived.mfe_fwd",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "Max Favorable Excursion (long-perspective): "
            "(max(mid[t..t+w]) - mid[t]) / mid[t] * 10000. In BPS."
        ),
    ),

    "derived.rv_fwd": S1OperatorSpec(
        name="derived.rv_fwd",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "Forward realized volatility: sqrt(sum(r_1s^2, forward over w)). "
            "Dep is mid_fut_1s; 1s log-returns are computed inline."
        ),
    ),

    # =====================================================================
    # EMA / TREND
    # EMAs are computed directly on the 1s grid with adjusted span to avoid
    # resampling artifacts at bucket boundaries.
    # =====================================================================

    "derived.ema": S1OperatorSpec(
        name="derived.ema",
        n_input_cols=1,
        required_params=("market_scope", "span_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "Exponential Moving Average: ewm(span=span_s, adjust=False). "
            "span_s = N_periods * TF_seconds (e.g. 50 EMA on 5m = span_s 15000)."
        ),
    ),

    "derived.price_vs_ema_bps": S1OperatorSpec(
        name="derived.price_vs_ema_bps",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "Relative distance: (price - ema) / ema * 10000. "
            "Deps: [price_col, ema_col]."
        ),
    ),

    "derived.ema_slope_bps": S1OperatorSpec(
        name="derived.ema_slope_bps",
        n_input_cols=1,
        required_params=("market_scope", "shift_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "EMA slope in BPS: (ema[t] - ema[t-shift]) / ema[t-shift] * 10000."
        ),
    ),

    "derived.trend_align": S1OperatorSpec(
        name="derived.trend_align",
        n_input_cols=3,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "Trend alignment: 1 if price>ema_short>ema_long, "
            "-1 if price<ema_short<ema_long, else 0. "
            "Deps: [price, ema_short, ema_long]."
        ),
    ),

    # =====================================================================
    # LEVEL-DISTANCE (generic, reused for prev_day / week / month / POC / VAH / VAL)
    # For semantic consistency with range.dist_to_high_bps / low_bps:
    #   range.dist_to_high_bps   — high-specific: (high - mid) / mid * 10000
    #   range.dist_to_low_bps    — low-specific:  (mid - low) / mid * 10000
    #   range.dist_to_level_bps  — generic:       (price - level) / price * 10000
    #                              sign follows (price - level), no asymmetric semantics
    # =====================================================================

    "range.dist_to_level_bps": S1OperatorSpec(
        name="range.dist_to_level_bps",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint=(
            "Generic BPS distance: (price - level) / price * 10000. "
            "Deps: [price_col, level_col]. Sign follows (price - level); "
            "use range.dist_to_high_bps/low_bps for asymmetric semantics."
        ),
    ),

    # =====================================================================
    # RECLAIM / BREAK (Debounced binary flags)
    # Simple above/below are 1-tick checks; reclaim/break are debounced
    # event flags that stay active for window_s seconds after the cross.
    # =====================================================================

    "derived.above_level": S1OperatorSpec(
        name="derived.above_level",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "Binary: 1 if price > level else 0. Deps: [price, level]."
        ),
    ),

    "derived.below_level": S1OperatorSpec(
        name="derived.below_level",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "Binary: 1 if price < level else 0. Deps: [price, level]."
        ),
    ),

    "derived.reclaim_flag": S1OperatorSpec(
        name="derived.reclaim_flag",
        n_input_cols=2,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "Debounced reclaim signal: 1 for window_s seconds after price "
            "crosses UP through level, as long as price stays above. "
            "Deps: [price, level]."
        ),
    ),

    "derived.break_flag_high": S1OperatorSpec(
        name="derived.break_flag_high",
        n_input_cols=2,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "Debounced upward break: 1 for window_s seconds after price "
            "crosses UP through resistance level, while price remains above. "
            "Deps: [price, level]."
        ),
    ),

    "derived.break_flag_low": S1OperatorSpec(
        name="derived.break_flag_low",
        n_input_cols=2,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "Debounced downward break: 1 for window_s seconds after price "
            "crosses DOWN through support level, while price remains below. "
            "Deps: [price, level]."
        ),
    ),

    # =====================================================================
    # FIBONACCI
    # =====================================================================

    "derived.fib_dist_bps": S1OperatorSpec(
        name="derived.fib_dist_bps",
        n_input_cols=3,
        required_params=("market_scope", "fib_level"),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "Distance from price to fibonacci level of the range. "
            "fib_price = low + fib_level * (high - low). "
            "Output: (price - fib_price) / price * 10000. "
            "Deps: [price, low, high]. fib_level ∈ [0, 1]."
        ),
    ),

    # =====================================================================
    # VOLUME PROFILE
    # POC / VAH / VAL level columns are injected via PASSTHROUGH from
    # vp_{asset}_{date}.parquet; these operators compute derived signals
    # from those passthrough levels.
    # =====================================================================

    "derived.price_vs_va": S1OperatorSpec(
        name="derived.price_vs_va",
        n_input_cols=3,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "Categorical: 2 if price>vah, 0 if price<val, else 1 (inside VA). "
            "Deps: [price, vah, val]."
        ),
    ),

    "derived.poc_migration_bps": S1OperatorSpec(
        name="derived.poc_migration_bps",
        n_input_cols=1,
        required_params=("market_scope", "shift_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint=(
            "POC migration in BPS: (poc[t] - poc[t-shift]) / poc[t-shift] * 10000. "
            "shift_s is typically half the VP window length "
            "(60m → 1800, 240m → 7200, 1d → 43200)."
        ),
    ),
}