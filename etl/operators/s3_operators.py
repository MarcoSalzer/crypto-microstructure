# ==============================================================================
# S3 Operator Registry — Complete registry of all operators used by S3 specs.
#
# 28 operators | Synced with all S3 spec files.
# ==============================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


@dataclass(frozen=True)
class S3OperatorSpec:
    name: str
    n_input_cols: int
    required_params: Tuple[str, ...]
    optional_params_defaults: Mapping[str, str]
    description_hint: str


S3_OPERATORS: Dict[str, S3OperatorSpec] = {

    # =====================================================================
    # GENERIC DERIVED
    # =====================================================================

    "derived.logratio": S3OperatorSpec(
        name="derived.logratio",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"eps": "1e-12"},
        description_hint="log(col_a / (col_b + eps)). Log-ratio of two columns.",
    ),

    "derived.ratio": S3OperatorSpec(
        name="derived.ratio",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"eps": "1e-12"},
        description_hint="col_a / (col_b + eps). Generic ratio.",
    ),

    "derived.robust_zscore": S3OperatorSpec(
        name="derived.robust_zscore",
        n_input_cols=0,  # 1 or 2 deps
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="(x - median) / (1.4826 * MAD + eps). Robust z-score.",
    ),

    "derived.roll_mean": S3OperatorSpec(
        name="derived.roll_mean",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Rolling mean of input column over window_s seconds.",
    ),

    "derived.roll_sum": S3OperatorSpec(
        name="derived.roll_sum",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={"resample": "1s"},
        description_hint="Rolling sum of input column over window_s seconds.",
    ),

    "derived.signal_flip_rate": S3OperatorSpec(
        name="derived.signal_flip_rate",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="count(sign_changes) / window_s. Higher = more indecisive.",
    ),

    "derived.signal_persist": S3OperatorSpec(
        name="derived.signal_persist",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Directional persistence: fraction of consistent sign sub-buckets.",
    ),

    # =====================================================================
    # S3 STAGE-SPECIFIC — Absorption
    # =====================================================================

    "s3.absorb_refill_mid": S3OperatorSpec(
        name="s3.absorb_refill_mid",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="(absorb_ask + absorb_bid) / 2. Mid absorption-refill.",
    ),

    "s3.absorption_asymmetry": S3OperatorSpec(
        name="s3.absorption_asymmetry",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"eps": "1e-12"},
        description_hint="(ask - bid) / (ask + bid + eps). Absorption side asymmetry.",
    ),

    "s3.absorption_break": S3OperatorSpec(
        name="s3.absorption_break",
        n_input_cols=5,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Composite absorption break signal (5 inputs).",
    ),

    "s3.absorption_break_flag": S3OperatorSpec(
        name="s3.absorption_break_flag",
        n_input_cols=5,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Binary flag: 1 if absorption break conditions met.",
    ),

    "s3.trade_absorption_ratio_bps": S3OperatorSpec(
        name="s3.trade_absorption_ratio_bps",
        n_input_cols=1,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Trade absorption ratio in basis points.",
    ),

    # =====================================================================
    # S3 STAGE-SPECIFIC — Cross-Market
    # =====================================================================

    "s3.cross_div": S3OperatorSpec(
        name="s3.cross_div",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Cross-market divergence: fut_signal - spot_signal.",
    ),

    "s3.cross_div_delta": S3OperatorSpec(
        name="s3.cross_div_delta",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Change in cross-market divergence: d1(cross_div).",
    ),

    "s3.cross_persist": S3OperatorSpec(
        name="s3.cross_persist",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Cross-market persistence: abs(roll_mean(x)) / (roll_mean(abs(x)) + eps).",
    ),

    "s3.cross_share": S3OperatorSpec(
        name="s3.cross_share",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"eps": "1e-12"},
        description_hint="Market share: col_a / (col_a + col_b + eps).",
    ),

    # =====================================================================
    # S3 STAGE-SPECIFIC — Temporal (d1, d2, median, MAD, shock)
    # =====================================================================

    "s3.temporal_d1": S3OperatorSpec(
        name="s3.temporal_d1",
        n_input_cols=1,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="First temporal difference: x[t] - x[t-1].",
    ),

    "s3.temporal_d2": S3OperatorSpec(
        name="s3.temporal_d2",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Second temporal difference: d1[t] - d1[t-1].",
    ),

    "s3.roll_median": S3OperatorSpec(
        name="s3.roll_median",
        n_input_cols=1,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Rolling median over window_s seconds.",
    ),

    "s3.roll_mad": S3OperatorSpec(
        name="s3.roll_mad",
        n_input_cols=2,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Rolling MAD: median(|x - median(x)|) over window_s.",
    ),

    "s3.shock": S3OperatorSpec(
        name="s3.shock",
        n_input_cols=3,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Shock score: |x - median| / (MAD + eps). Uses pre-computed median/MAD.",
    ),

    # =====================================================================
    # S3 STAGE-SPECIFIC — Meta / Depth
    # =====================================================================

    "s3.dir_consistency_persist": S3OperatorSpec(
        name="s3.dir_consistency_persist",
        n_input_cols=2,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Directional consistency persistence across markets.",
    ),

    "s3.dir_consistency_asym": S3OperatorSpec(
        name="s3.dir_consistency_asym",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"eps": "1e-12"},
        description_hint="Directional consistency asymmetry: (fut - spot) / (fut + spot + eps).",
    ),

    "s3.qp_depth_coherence": S3OperatorSpec(
        name="s3.qp_depth_coherence",
        n_input_cols=4,
        required_params=("market_scope", "window_s"),
        optional_params_defaults={},
        description_hint="Queue-pressure depth coherence across BPS bands.",
    ),

    "s3.qp_depth_slope": S3OperatorSpec(
        name="s3.qp_depth_slope",
        n_input_cols=4,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Queue-pressure depth slope across BPS bands.",
    ),

    "s3.qp_depth_curvature": S3OperatorSpec(
        name="s3.qp_depth_curvature",
        n_input_cols=4,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Queue-pressure depth curvature (second derivative).",
    ),

    # =====================================================================
    # S3 STAGE-SPECIFIC — Liquidity Events
    # =====================================================================

    "s3.refill_vs_pull_ratio": S3OperatorSpec(
        name="s3.refill_vs_pull_ratio",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={"eps": "1e-12"},
        description_hint="refill_rate / (pull_rate + eps). Liquidity replenishment ratio.",
    ),

    "s3.vacuum_score": S3OperatorSpec(
        name="s3.vacuum_score",
        n_input_cols=2,
        required_params=("market_scope",),
        optional_params_defaults={},
        description_hint="Composite vacuum score: pull_rate - refill_rate (liquidity drain).",
    ),
}