# ==============================================================================
# S6 Operator Registry — Multi-Pair (BTC ↔ ETH ↔ BNB)
# ==============================================================================
# Canonical reference for every operator used in s6_cross_asset.py.
#
# Inherited operators (originally defined in earlier stages) are re-listed here
# verbatim so that the S6 engine has a self-contained, importable registry
# without depending on operator dicts from S1–S5.
#
# New operators (first defined here) include their full implementation
# as a callable accepting (df, **params) and returning a pd.Series.
#
# MULTI-PAIR GENERALISATION (2026-03):
#   All pairwise operators now use generic param keys col_a / col_b
#   instead of the original btc_col / eth_col.  Backward compatibility
#   is maintained: if col_a is absent, btc_col is used as fallback
#   (likewise col_b → eth_col).
#
# ── Operator index ────────────────────────────────────────────────────────────
#   INHERITED
#     derived.sub               originally defined in S1 (element-wise A − B)
#     derived.robust_zscore     originally defined in S3/S4 (median/MAD z-score)
#
#   NEW (S6)
#     derived.cross_asset_diff  col_a − col_b on merged DataFrame
#     derived.bps_mid_dev       (lwp_mid − mid) / mid * 10_000
#     derived.rolling_ols_beta  rolling OLS slope  y ~ x  (single regressor)
#     derived.beta_residual     y − beta * x  (element-wise residual)
#     derived.cross_lag_corr    rolling corr( lead_col.shift(lag), lag_col )
#     derived.bps_spread        spread / mid * 10_000  (bid-ask cost in bps)
#     derived.regime_xor        |flag_a − flag_b|  (divergence indicator)
#     derived.regime_align      flag_a AND flag_b  (co-movement indicator)
#
# ── Usage contract ────────────────────────────────────────────────────────────
#   Every operator callable has the signature:
#       fn(df: pd.DataFrame, **params) -> pd.Series
#
#   The S6 engine resolves params from FeatureSpec.params, passes the merged
#   DataFrame, and assigns the returned Series to FeatureSpec.name.
#
#   All operators are registered in S6_OPERATOR_REGISTRY (dict at bottom).
#
# ==============================================================================

import numpy as np
import pandas as pd
from typing import Dict, Callable


# ==============================================================================
# Param resolution helpers (backward-compatible)
# ==============================================================================

def _col_a(params: dict) -> str:
    """Resolve first column: col_a (preferred) → btc_col (compat fallback)."""
    return params.get("col_a") or params["btc_col"]


def _col_b(params: dict) -> str:
    """Resolve second column: col_b (preferred) → eth_col (compat fallback)."""
    return params.get("col_b") or params["eth_col"]


# ==============================================================================
# INHERITED OPERATORS
# ==============================================================================

def _op_sub(df: pd.DataFrame, **params) -> pd.Series:
    """Element-wise subtraction: df[input_col_a] − df[input_col_b]."""
    col_a = params["input_col_a"]
    col_b = params["input_col_b"]
    return df[col_a] - df[col_b]


def _op_robust_zscore(df: pd.DataFrame, **params) -> pd.Series:
    """
    Robust z-score via rolling median / MAD.
    z = (x - median) / (1.4826 * MAD + eps).

    Optional param zscore_clip (float): if provided, clips output to
    [-zscore_clip, +zscore_clip].  Use for features whose upstream series
    occasionally has near-zero MAD (e.g. depth slope/curvature), which would
    otherwise produce extreme z-scores and propagate to cross-asset diffs.
    """
    col    = params["input_col"]
    w      = int(params["window_s"])
    clip   = float(params["zscore_clip"]) if "zscore_clip" in params else None
    eps    = 1e-8
    x      = df[col]
    med    = x.rolling(w, min_periods=w).median()
    mad    = (x - med).abs().rolling(w, min_periods=w).median()
    z      = (x - med) / (1.4826 * mad + eps)
    return z.clip(-clip, clip) if clip is not None else z


# ==============================================================================
# NEW OPERATORS  (S6-only)
# ==============================================================================

# ------------------------------------------------------------------------------
# derived.cross_asset_diff
# ------------------------------------------------------------------------------
# Formula:  result = df[col_a] − df[col_b]
# Params:
#   col_a  (str) — column carrying the first asset value  (post-merge suffix)
#   col_b  (str) — column carrying the second asset value (post-merge suffix)
#   Compat fallback: btc_col → col_a, eth_col → col_b
# NaN policy: NaN propagates if either input is NaN.

def _op_cross_asset_diff(df: pd.DataFrame, **params) -> pd.Series:
    """
    Cross-asset difference: df[col_a] − df[col_b].
    Both columns must be on the same normalisation scale.
    """
    return df[_col_a(params)] - df[_col_b(params)]


# ------------------------------------------------------------------------------
# derived.bps_mid_dev
# ------------------------------------------------------------------------------
# Formula:  dev_bps = (lwp_mid − mid) / mid * 10_000
# Params:
#   lwp_col  (str) — liquidity-weighted-price column name
#   mid_col  (str) — best-bid/ask midpoint column name
# NaN policy: NaN where mid = 0 or either input is NaN.

def _op_bps_mid_dev(df: pd.DataFrame, **params) -> pd.Series:
    """
    BPS deviation of LWP from mid: (lwp_mid - mid) / mid * 10_000.
    """
    lwp = df[params["lwp_col"]]
    mid = df[params["mid_col"]]
    return (lwp - mid) / mid * 10_000.0


# ------------------------------------------------------------------------------
# derived.rolling_ols_beta
# ------------------------------------------------------------------------------
# Formula:  beta_t = cov_roll(y, x) / var_roll(x)
# Params:
#   y_col, x_col, window_s, beta_clip (default 5.0)
# NaN policy: first (window_s − 1) rows → NaN; clipped to ±beta_clip.

def _op_rolling_ols_beta(df: pd.DataFrame, **params) -> pd.Series:
    """
    Rolling OLS slope: cov(y, x) / var(x) over window_s rows.
    Clipped to [−beta_clip, +beta_clip].
    """
    y         = df[params["y_col"]]
    x         = df[params["x_col"]]
    w         = int(params["window_s"])
    clip      = float(params.get("beta_clip", 5.0))
    eps       = 1e-12

    roll_cov  = y.rolling(w, min_periods=w).cov(x)
    roll_var  = x.rolling(w, min_periods=w).var()

    beta      = roll_cov / (roll_var + eps)
    return beta.clip(-clip, clip)


# ------------------------------------------------------------------------------
# derived.beta_residual
# ------------------------------------------------------------------------------
# Formula:  residual_t = y_t − beta_t * x_t
# Params: y_col, x_col, beta_col

def _op_beta_residual(df: pd.DataFrame, **params) -> pd.Series:
    """Beta-adjusted residual: y - beta * x."""
    y    = df[params["y_col"]]
    x    = df[params["x_col"]]
    beta = df[params["beta_col"]]
    return y - beta * x


# ------------------------------------------------------------------------------
# derived.cross_lag_corr
# ------------------------------------------------------------------------------
# Formula:  corr_t = rolling_corr( lead_col.shift(lag_s), lag_col, window_s )
# Params: lead_col, lag_col, lag_s, window_s

def _op_cross_lag_corr(df: pd.DataFrame, **params) -> pd.Series:
    """
    Rolling cross-correlation with integer lag.
    corr( lead_col.shift(lag_s), lag_col, window=window_s ).
    """
    lead_col = params["lead_col"]
    lag_col  = params["lag_col"]
    lag_s    = int(params["lag_s"])
    w        = int(params["window_s"])

    lead_shifted = df[lead_col].shift(lag_s)
    target       = df[lag_col]

    return lead_shifted.rolling(w, min_periods=w).corr(target)


# ------------------------------------------------------------------------------
# derived.bps_spread
# ------------------------------------------------------------------------------
# Formula:  spread_bps = spread / mid * 10_000
# Params: spread_col, mid_col

def _op_bps_spread(df: pd.DataFrame, **params) -> pd.Series:
    """Bid-ask spread in basis points: spread / mid * 10_000."""
    spread = df[params["spread_col"]]
    mid    = df[params["mid_col"]]
    return spread / mid * 10_000.0


# ------------------------------------------------------------------------------
# derived.regime_xor
# ------------------------------------------------------------------------------
# Formula:  result = |flag_a − flag_b|  (XOR on binary flags)
# Params: col_a / col_b  (compat: btc_col / eth_col)

def _op_regime_xor(df: pd.DataFrame, **params) -> pd.Series:
    """
    Regime divergence flag: |flag_a - flag_b|.
    1 = divergent regimes, 0 = aligned regimes.
    """
    a = df[_col_a(params)]
    b = df[_col_b(params)]
    return (a.astype(float) - b.astype(float)).abs()


# ------------------------------------------------------------------------------
# derived.regime_align
# ------------------------------------------------------------------------------
# Formula:  result = flag_a AND flag_b  (logical AND on binary flags)
# Params: col_a / col_b  (compat: btc_col / eth_col)

def _op_regime_align(df: pd.DataFrame, **params) -> pd.Series:
    """
    Regime co-movement flag: flag_a AND flag_b.
    1 = both assets in breakout, 0 = at least one not.
    """
    a = df[_col_a(params)].fillna(0).astype(int)
    b = df[_col_b(params)].fillna(0).astype(int)
    return (a & b).astype(float)


# ==============================================================================
# OPERATOR REGISTRY
# ==============================================================================

S6_OPERATOR_REGISTRY: Dict[str, Callable] = {

    # ── Inherited ─────────────────────────────────────────────────────────────
    "derived.sub":                _op_sub,
    "derived.robust_zscore":      _op_robust_zscore,

    # ── New (S6 original) ─────────────────────────────────────────────────────
    "derived.cross_asset_diff":   _op_cross_asset_diff,
    "derived.bps_mid_dev":        _op_bps_mid_dev,
    "derived.rolling_ols_beta":   _op_rolling_ols_beta,
    "derived.beta_residual":      _op_beta_residual,
    "derived.cross_lag_corr":     _op_cross_lag_corr,

    # ── New (S6 Round 2 extension) ────────────────────────────────────────────
    "derived.bps_spread":         _op_bps_spread,
    "derived.regime_xor":         _op_regime_xor,
    "derived.regime_align":       _op_regime_align,
}


# ==============================================================================
# OPERATOR METADATA
# ==============================================================================

S6_OPERATOR_METADATA: Dict[str, dict] = {
    "derived.sub": {
        "origin":      "S1 (inherited)",
        "formula":     "A - B",
        "nan_cause":   "NaN in either input",
        "new_in_s6":   False,
        "complexity":  "trivial",
    },
    "derived.robust_zscore": {
        "origin":      "S3/S4 (inherited)",
        "formula":     "(x - roll_median(x,w)) / (1.4826 * roll_MAD(x,w) + eps)",
        "nan_cause":   "rolling warmup (first w-1 rows)",
        "new_in_s6":   False,
        "complexity":  "low",
    },
    "derived.cross_asset_diff": {
        "origin":      "S6 (new)",
        "formula":     "df[col_a] - df[col_b]",
        "nan_cause":   "NaN in either asset column",
        "new_in_s6":   True,
        "complexity":  "trivial",
    },
    "derived.bps_mid_dev": {
        "origin":      "S6 (new)",
        "formula":     "(lwp_mid - mid) / mid * 10_000",
        "nan_cause":   "mid = 0  or  NaN in either input",
        "new_in_s6":   True,
        "complexity":  "low",
    },
    "derived.rolling_ols_beta": {
        "origin":      "S6 (new)",
        "formula":     "cov_roll(y, x, w) / (var_roll(x, w) + eps) clipped to ±beta_clip",
        "nan_cause":   "rolling warmup (first w-1 rows); var(x) ≈ 0 intervals",
        "new_in_s6":   True,
        "complexity":  "medium",
    },
    "derived.beta_residual": {
        "origin":      "S6 (new)",
        "formula":     "y - beta * x",
        "nan_cause":   "NaN in beta_col (propagated from rolling_ols_beta warmup)",
        "new_in_s6":   True,
        "complexity":  "low",
    },
    "derived.cross_lag_corr": {
        "origin":      "S6 (new)",
        "formula":     "rolling_corr(lead_col.shift(lag_s), lag_col, w)",
        "nan_cause":   "rolling warmup (first w + lag_s - 1 rows)",
        "new_in_s6":   True,
        "complexity":  "medium",
    },
    "derived.bps_spread": {
        "origin":      "S6 Round 2 (new)",
        "formula":     "spread / mid * 10_000",
        "nan_cause":   "mid = 0  or  NaN in either input",
        "new_in_s6":   True,
        "complexity":  "trivial",
    },
    "derived.regime_xor": {
        "origin":      "S6 Round 2 (new)",
        "formula":     "|flag_a - flag_b|  (binary XOR equivalent)",
        "nan_cause":   "NaN in either flag input",
        "new_in_s6":   True,
        "complexity":  "trivial",
    },
    "derived.regime_align": {
        "origin":      "S6 Round 2 (new)",
        "formula":     "flag_a AND flag_b  (binary co-movement)",
        "nan_cause":   "NaN in either flag input (filled 0 before AND)",
        "new_in_s6":   True,
        "complexity":  "trivial",
    },
}