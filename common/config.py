"""
config.py — Global configuration for the prediction pipeline.
=============================================================
Single source of truth for paths, horizons, seeds, feature bundles,
and spread constants. Imported by all other modules.

Trading cost is the Binance exchange fee (the dominant cost): the inside spread
is ~0.01 bps (1 tick / mid price), so SPREAD_BPS is the taker round-trip
(10 bps = 0.05% each side); MAKER_COST_BPS covers limit-order strategies.
KEEP_LIST selects columns from the log1p-transformed feature set ('ml_features_log1p').
"""

from pathlib import Path

from common.paths import DATA_ROOT, REDUCTION_DIR, RESULTS_DIR as _RESULTS_DIR, REPO_ROOT

# ─── Paths ────────────────────────────────────────────────────────────────────
# All data-store paths resolve through common/paths.py (env THESIS_DATA_ROOT
# or configs/paths.yaml). The ~94 GB feature/data store is external to this repo.

BASE_DIR       = DATA_ROOT

# ML_FEATURES points to the log1p-transformed feature set.
# log1p was applied to 435 non-negative quantity/rate features (volume,
# notional, depth, order-flow rates); see selection/apply_log1p.py +
# log1p_final_columns.csv. log1p is monotonic, so tree models are unaffected;
# linear/cluster models benefit from the reduced skew. The raw (untransformed)
# features remain available at BASE_DIR/"ml_features" if a revert is needed.
ML_FEATURES    = BASE_DIR / "ml_features_log1p"
# Feature-reduction artifacts are committed under results/selection (REDUCTION_DIR).
KEEP_LIST      = REDUCTION_DIR / "feature_keep.csv"
CATALOG_PATH   = REDUCTION_DIR / "feature_catalog.csv"

RESULTS_DIR    = _RESULTS_DIR

# ─── Assets & Targets ────────────────────────────────────────────────────────

ASSETS = ["btc", "eth"]

HORIZONS = {
    "1s":   {"priority": 1, "label": "1s"},
    "5s":   {"priority": 2, "label": "5s"},
    "15s":  {"priority": 3, "label": "15s (primary)"},
    "30s":  {"priority": 4, "label": "30s"},
    "60s":  {"priority": 5, "label": "60s"},
    "120s": {"priority": 6, "label": "120s"},
    "300s": {"priority": 7, "label": "300s"},
    "900s": {"priority": 8, "label": "900s"},
}

# Horizons available per target family. Returns cover all eight horizons;
# the excursion targets (MFE/MAE) were engineered at four horizons only.
RETURN_HORIZONS = ["1s", "5s", "15s", "30s", "60s", "120s", "300s", "900s"]
MFE_MAE_HORIZONS = ["15s", "60s", "300s", "900s"]

# Target families. Each maps a (family, horizon) pair to a column-name
# template. The {asset} placeholder is filled by target_col().
TARGET_FAMILIES = {
    "ret": "ret_fwd_{horizon}_{asset}",
    "mfe": "mfe_fwd_{horizon}_bps_{asset}",
    "mae": "mae_fwd_{horizon}_bps_{asset}",
}


def parse_target(target: str) -> tuple[str, str]:
    """
    Split a flat target token like 'ret_15s' or 'mfe_60s' into
    (family, horizon). Raises ValueError on an unknown family or on a
    horizon that the family does not provide.
    """
    parts = target.split("_", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Target '{target}' is malformed. Expected '<family>_<horizon>', "
            f"e.g. 'ret_15s', 'mfe_60s', 'mae_300s'.")
    family, horizon = parts
    if family not in TARGET_FAMILIES:
        raise ValueError(
            f"Unknown target family '{family}' in '{target}'. "
            f"Valid families: {sorted(TARGET_FAMILIES)}.")
    valid_horizons = RETURN_HORIZONS if family == "ret" else MFE_MAE_HORIZONS
    if horizon not in valid_horizons:
        raise ValueError(
            f"Family '{family}' has no horizon '{horizon}'. "
            f"Valid horizons for '{family}': {valid_horizons}.")
    return family, horizon


def target_col(target: str, asset: str) -> str:
    """
    Resolve a flat target token plus an asset into the dataset column name.

    target : flat token, either '<family>_<horizon>' (e.g. 'ret_15s',
             'mfe_60s') OR a bare horizon (e.g. '15s'), in which case the
             'ret' family is assumed for backward compatibility.
    asset  : 'btc' or 'eth'.
    """
    # Backward compatibility: a bare horizon means a return target.
    if target in HORIZONS:
        family, horizon = "ret", target
    else:
        family, horizon = parse_target(target)
    template = TARGET_FAMILIES[family]
    return template.format(horizon=horizon, asset=asset)


def all_targets(families: tuple[str, ...] = ("ret", "mfe", "mae")) -> list[str]:
    """Return the flat list of every target token for the given families."""
    out = []
    for fam in families:
        horizons = RETURN_HORIZONS if fam == "ret" else MFE_MAE_HORIZONS
        out += [f"{fam}_{h}" for h in horizons]
    return out

# ─── Cross-Validation ────────────────────────────────────────────────────────

N_FOLDS_MAIN      = 5
N_FOLDS_ROBUST    = 7
DEFAULT_SEEDS     = [42, 123, 999]
ROBUST_SEEDS      = [42, 123, 999, 7, 31, 73, 256]

# Unified rerun defaults — used by all scripts via:
#   from common.config import RERUN_SEEDS, RERUN_N_JOBS, RERUN_FOLDS
#   seeds = args.seeds if args.seeds else RERUN_SEEDS
RERUN_SEEDS  = [42, 123, 999]
RERUN_N_JOBS = 8
RERUN_FOLDS  = 5

# ─── Trading Costs ────────────────────────────────────────────────────────────
#
# Inside spread on Binance BTC/ETH: ~0.01 bps (negligible).
# Confirmed from s5_features: spread_fut_1s median = $0.10 on $85k = 0.012 bps.
#
# SPREAD_BPS is the best-bid-to-best-ask inside spread cost, i.e. the Binance
# exchange fee (bps_sym — the full-visible-book spread — is a different, larger
# quantity and must not be used for trading-cost calculations).
#
# Actual trading cost = Binance exchange fees:
#   Standard tier: Maker 0.02%, Taker 0.05%
#   → Round-trip taker (market orders): 10 bps
#   → Round-trip maker (limit orders):   4 bps
#   → One-way taker: 5.0 bps
#   → One-way maker: 2.0 bps
#
# SPREAD_BPS is used by all profitability scripts as the round-trip cost.

SPREAD_BPS = {
    "btc": {"spot": 10.0, "fut": 10.0},
    "eth": {"spot": 10.0, "fut": 10.0},
}

# For maker-only / limit-order strategies
MAKER_COST_BPS = {
    "btc": {"spot": 4.0, "fut": 4.0},
    "eth": {"spot": 4.0, "fut": 4.0},
}

# Compat reference (DO NOT USE for trading cost calculations):
# BPS_SYM_FULL_BOOK = {"btc": {"spot": 24.49, "fut": 18.61},
#                       "eth": {"spot": 75.78, "fut": 52.74}}

# ─── Feature Bundles (for the ablation study) ───────────────────────────────

FAMILY_TO_BUNDLE = {
    "imbalance":        "B1_imbalance",
    "pressure":         "B1_imbalance",
    "aggression":       "B1_imbalance",
    "bookshape":        "B2_bookshape",
    "impact":           "B2_bookshape",
    "absorption":       "B3_flow",
    "liquidity_events": "B3_flow",
    "activity":         "B3_flow",
    "cross_market":     "B4_cross_market",
    "dynamics":         "B5_dynamics",
    "normalization":    "B5_dynamics",
    "price":            "B6_context",
    "range":            "B6_context",
    "returns":          "B6_context",
    "meta":             "B6_context",
    "calendar":         "B6_context",
    "health":           "B6_context",
    "cross_asset":      "B7_cross_asset",
}

BUNDLE_NAMES = [
    "B1_imbalance",
    "B2_bookshape",
    "B3_flow",
    "B4_cross_market",
    "B5_dynamics",
    "B6_context",
    "B7_cross_asset",
]

BUNDLE_DESCRIPTIONS = {
    "B1_imbalance":    "Bid/ask asymmetry: depth imbalance, queue pressure, taker imbalance, aggression",
    "B2_bookshape":    "LOB structure: depth levels, gradients, liquidity concentration, impact",
    "B3_flow":         "Order flow events: absorption, refill, trade activity, add/cancel rates",
    "B4_cross_market": "Spot↔Futures divergence: basis, spread ratios, volume/trade count shares",
    "B5_dynamics":     "Temporal patterns: derivatives (d1/d2), z-scores, median/MAD normalisation",
    "B6_context":      "Price context: levels, range position, backward returns, directional consistency",
    "B7_cross_asset":  "Cross-asset BTC↔ETH differentials (S6 features)",
}

# ─── LightGBM Defaults ───────────────────────────────────────────────────────

LGBM_PARAMS = {
    "objective":        "regression",
    "metric":           "mse",
    "num_leaves":       31,
    "max_depth":        6,
    "learning_rate":    0.05,
    "colsample_bytree": 0.5,
    "subsample":        0.8,
    "subsample_freq":   1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "min_child_samples":100,
    "n_estimators":     1000,
    "n_jobs":           -1,
    "verbose":          -1,
}