# ==============================================================================
# Central operator specifications for Stage 5 (S5) feature computation.
#
# S5 operators build higher-order signal quality metrics on top of the S4
# feature table. Five operators are used in this stage:
#
#   derived.rolling_median  — rolling median of a single input column.
#                             Foundation for robust shock pipelines.
#   derived.rolling_mad     — rolling MAD: median(|x - median_x|).
#                             Requires a pre-computed median column as input.
#   derived.robust_shock    — event-strength score: |x - median| / (MAD + eps).
#                             Uses pre-computed median and MAD columns (intra-stage
#                             deps); NOT the same as inline robust_zscore.
#   derived.robust_zscore   — inline robust z-score: (x - rolling_median(x)) /
#                             (rolling_MAD(x) * scale + eps). Self-contained;
#                             does NOT require pre-computed median/MAD deps.
#   derived.signal_persist  — directional persistence (S5 definition):
#                             abs(roll_mean(x)) / (roll_mean(abs(x)) + eps).
#                             Range [0, 1]. 1 = fully consistent direction.
#                             NOTE: This definition differs from the S4 variant
#                             (which used sign-consistency fraction). The S5
#                             formula is threshold-free and purely magnitude-based.
#
# Operator count: 5
#
# Dependency contract for intra-S5 chains:
#   rolling_median  → no intra-S5 deps (reads S4 columns directly)
#   rolling_mad     → depends on rolling_median output (intra-S5)
#   robust_shock    → depends on rolling_mad AND rolling_median (intra-S5)
#   robust_zscore   → no intra-S5 deps (self-contained)
#   signal_persist  → no intra-S5 deps (reads S4 columns directly)
#
# Maximum dependency depth: 2 levels (robust_shock → rolling_mad → rolling_median).
# ==============================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


@dataclass(frozen=True)
class S5OperatorSpec:
    name: str
    n_input_cols: int
    required_params: Tuple[str, ...]
    optional_params_defaults: Mapping[str, str]
    description_hint: str


S5_OPERATORS: Dict[str, S5OperatorSpec] = {

    # =====================================================================
    # ROLLING STATISTICS
    # =====================================================================

    "derived.rolling_median": S5OperatorSpec(
        name="derived.rolling_median",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Rolling median over window_s seconds.",
    ),

    "derived.rolling_mad": S5OperatorSpec(
        name="derived.rolling_mad",
        n_input_cols=2,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Rolling MAD: median(|x - median(x)|). Requires pre-computed median.",
    ),

    "derived.robust_shock": S5OperatorSpec(
        name="derived.robust_shock",
        n_input_cols=3,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"eps": "1e-9"},
        description_hint="|x - median| / (MAD + eps). Shock magnitude. Requires median + MAD.",
    ),

    # =====================================================================
    # NORMALIZATION
    # =====================================================================

    "derived.robust_zscore": S5OperatorSpec(
        name="derived.robust_zscore",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"eps": "1e-9"},
        description_hint=(
            "(x - median) / (1.4826 * MAD + eps). Self-contained robust z-score."
        ),
    ),

    # =====================================================================
    # SIGNAL QUALITY
    # =====================================================================

    "derived.signal_persist": S5OperatorSpec(
        name="derived.signal_persist",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"eps": "1e-9"},
        description_hint=(
            "abs(roll_mean(x)) / (roll_mean(abs(x)) + eps). "
            "Threshold-free directional persistence."
        ),
    ),
}