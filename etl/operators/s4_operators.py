# ==============================================================================
# S4 Operator Registry — Complete registry of all operators used by S4 specs.
#
# 17 operators | Synced with all S4 spec files.
#
# Removed (were only used by S4 specs that were deleted as S3 duplicates):
#   - derived.absorption_break : absorption_break_{15s,60s} moved to / already in S3
#   - derived.vacuum_score     : vacuum_score_* were S3 duplicates, removed from S4
# ==============================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


@dataclass(frozen=True)
class S4OperatorSpec:
    name: str
    n_input_cols: int
    required_params: Tuple[str, ...]
    optional_params_defaults: Mapping[str, str]
    description_hint: str


S4_OPERATORS: Dict[str, S4OperatorSpec] = {

    # =====================================================================
    # TEMPORAL
    # =====================================================================

    "derived.d1": S4OperatorSpec(
        name="derived.d1",
        n_input_cols=1,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="First temporal difference: x[t] - x[t-1].",
    ),

    "derived.d2": S4OperatorSpec(
        name="derived.d2",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"resample": "1s"},
        description_hint="Second temporal difference: d1[t] - d1[t-1].",
    ),

    "derived.rolling_median": S4OperatorSpec(
        name="derived.rolling_median",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Rolling median over window_s seconds.",
    ),

    "derived.rolling_mad": S4OperatorSpec(
        name="derived.rolling_mad",
        n_input_cols=2,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Rolling MAD: median(|x - median(x)|) over window_s.",
    ),

    "derived.robust_shock": S4OperatorSpec(
        name="derived.robust_shock",
        n_input_cols=0,  # 1 or 3 deps
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="|x - median| / (MAD + eps). Shock magnitude.",
    ),

    # =====================================================================
    # CROSS-MARKET
    # =====================================================================

    "derived.cross_market_div": S4OperatorSpec(
        name="derived.cross_market_div",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Cross-market divergence: fut - spot.",
    ),

    "derived.ratio": S4OperatorSpec(
        name="derived.ratio",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"eps": "1e-12", "abs_den": "true"},
        description_hint="Generic ratio: num / (|den| + eps).",
    ),

    # =====================================================================
    # DEPTH STRUCTURE
    # =====================================================================

    "derived.depth_coherence": S4OperatorSpec(
        name="derived.depth_coherence",
        n_input_cols=4,
        required_params=("market_scope",),
        optional_params_defaults={"window_s": "0"},  # kept for compat specs, ignored
        description_hint="Row-wise sign-agreement fraction across depth bands. "
                         "For 4 bands → 6 pairs; returns fraction of pairs with "
                         "matching non-zero sign. Range [0,1].",
    ),

    "derived.depth_slope": S4OperatorSpec(
        name="derived.depth_slope",
        n_input_cols=4,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Depth profile slope across BPS bands.",
    ),

    "derived.depth_curvature": S4OperatorSpec(
        name="derived.depth_curvature",
        n_input_cols=4,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Depth profile curvature (second derivative).",
    ),

    # =====================================================================
    # REGIME / SIGNAL QUALITY
    # =====================================================================

    "derived.signal_persist": S4OperatorSpec(
        name="derived.signal_persist",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"zero_eps": "0.0"},
        description_hint="Persistence: fraction of consistent sign sub-buckets.",
    ),

    "derived.signal_flip_rate": S4OperatorSpec(
        name="derived.signal_flip_rate",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="count(sign_changes) / window_s. Higher = indecisive.",
    ),

    "derived.pct_rank": S4OperatorSpec(
        name="derived.pct_rank",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Rolling percentile rank within window.",
    ),

    # =====================================================================
    # NORMALIZATION
    # =====================================================================

    "derived.robust_zscore": S4OperatorSpec(
        name="derived.robust_zscore",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"min_periods": "5"},
        description_hint="(x - median) / (1.4826 * MAD + eps). Robust z-score.",
    ),

    # =====================================================================
    # AGGREGATION
    # =====================================================================

    "derived.roll_sum": S4OperatorSpec(
        name="derived.roll_sum",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Rolling sum over window_s seconds.",
    ),

    "derived.logratio": S4OperatorSpec(
        name="derived.logratio",
        n_input_cols=1,
        required_params=("market_scope",),
        optional_params_defaults={"eps": "1e-12"},
        description_hint="log(col_a / (col_b + eps)) or log-transform of single column.",
    ),

    # =====================================================================
    # UTILITY
    # =====================================================================

    "derived.passthrough": S4OperatorSpec(
        name="derived.passthrough",
        n_input_cols=1,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Identity passthrough: output = input. Used for aliasing.",
    ),
}