"""
metrics.py — Evaluation metrics for prediction models.
=====================================================
ML metrics (primary) and trading metrics (bonus) used across
all model types (Ridge, LightGBM, MLP).

Metrics:
  ML:      R², MSE, RMSE, MAE, Directional Accuracy, IC, IC_IR
  Trading: Spread-bridging rate, simple PnL backtest, Sharpe
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ─── Core fold-level metrics ─────────────────────────────────────────────────

def compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute all primary metrics for a single fold.
    Used by cv_engine.run_cv for each seed.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]

    if len(yt) < 10:
        return {"r2": -999, "mse": 999, "mae": 999, "rmse": 999,
                "dir_acc": 0.5, "ic": 0.0}

    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    mse  = float(np.mean((yt - yp) ** 2))
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(mse))

    # Directional accuracy: fraction where sign(pred) == sign(actual)
    # Exclude rows where either is exactly zero
    nonzero = (yt != 0) & (yp != 0)
    if nonzero.sum() > 0:
        dir_acc = float((np.sign(yt[nonzero]) == np.sign(yp[nonzero])).mean())
    else:
        dir_acc = 0.5

    # Information Coefficient (Spearman rank correlation)
    if len(yt) > 2:
        ic, _ = stats.spearmanr(yt, yp)
        ic = float(ic) if np.isfinite(ic) else 0.0
    else:
        ic = 0.0

    return {
        "r2":      round(r2, 8),
        "mse":     round(mse, 12),
        "mae":     round(mae, 10),
        "rmse":    round(rmse, 10),
        "dir_acc": round(dir_acc, 6),
        "ic":      round(ic, 8),
    }


# ─── Aggregate metrics across folds ──────────────────────────────────────────

def ic_ir(ic_per_fold: list[float]) -> float:
    """
    Information Coefficient Information Ratio.
    IC_IR = mean(IC) / std(IC). Higher = more stable prediction quality.
    """
    ics = np.array(ic_per_fold)
    std = ics.std()
    if std < 1e-10:
        return 0.0
    return float(ics.mean() / std)


# ─── Trading metrics (bonus) ─────────────────────────────────────────────────

def simple_backtest(
    predictions: np.ndarray,
    actuals: np.ndarray,
    spread_bps: float,
    threshold_mult: float = 0.5,
) -> dict:
    """
    Simple signal-based backtest.

    Strategy:
      - Long  when pred > threshold (= spread_bps * threshold_mult)
      - Short when pred < -threshold
      - Flat otherwise
    Cost: full spread per round-trip (worst case = market orders both sides).

    Args:
        predictions: predicted returns in same units as actuals (log returns)
        actuals:     realized returns
        spread_bps:  full spread in basis points
        threshold_mult: fraction of spread as signal threshold

    Returns:
        dict with pnl_bps, sharpe, hit_rate, trade_count, etc.
    """
    mask = np.isfinite(predictions) & np.isfinite(actuals)
    pred = predictions[mask]
    act  = actuals[mask]

    # Convert spread to same scale as returns (bps → decimal)
    spread_dec = spread_bps / 10_000
    threshold  = spread_dec * threshold_mult

    signal = np.where(pred > threshold, 1,
              np.where(pred < -threshold, -1, 0))

    n_trades = (signal != 0).sum()
    if n_trades == 0:
        return {
            "pnl_gross_bps": 0, "pnl_net_bps": 0, "sharpe": 0,
            "hit_rate": 0, "trade_count": 0, "trade_pct": 0,
            "spread_bridge_rate": 0,
        }

    # Gross PnL (before costs)
    gross_pnl = signal * act
    # Cost = spread per round-trip when trading
    costs = np.abs(signal) * spread_dec
    # Net PnL
    net_pnl = gross_pnl - costs

    # Convert to bps for readability
    gross_bps = float(gross_pnl.sum() * 10_000)
    net_bps   = float(net_pnl.sum() * 10_000)

    # Sharpe on net PnL (annualised assuming 1-second bars)
    trading_mask = signal != 0
    net_when_trading = net_pnl[trading_mask]
    if net_when_trading.std() > 0:
        sharpe = float(
            net_when_trading.mean() / net_when_trading.std()
            * np.sqrt(365 * 24 * 3600)
        )
    else:
        sharpe = 0.0

    # Hit rate (directional accuracy when trading)
    correct = (np.sign(pred[trading_mask]) == np.sign(act[trading_mask]))
    hit_rate = float(correct.mean())

    # Spread-bridging rate: fraction of predictions that exceed the full spread
    bridge_rate = float((np.abs(pred) > spread_dec).mean())

    return {
        "pnl_gross_bps":     round(gross_bps, 2),
        "pnl_net_bps":       round(net_bps, 2),
        "sharpe":            round(sharpe, 2),
        "hit_rate":          round(hit_rate, 4),
        "trade_count":       int(n_trades),
        "trade_pct":         round(float(n_trades / len(pred) * 100), 2),
        "spread_bridge_rate":round(bridge_rate, 4),
    }


# ─── Classification metrics (for thesis evaluation) ──────────────────────────

def directional_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Matthews Correlation Coefficient on direction (up/down).
    Ignores zero-return rows. MCC ∈ [-1, 1], 0 = random.
    """
    mask = (y_true != 0) & (y_pred != 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 10:
        return 0.0

    true_dir = (np.sign(y_true[mask]) + 1) / 2  # 0 or 1
    pred_dir = (np.sign(y_pred[mask]) + 1) / 2

    tp = ((true_dir == 1) & (pred_dir == 1)).sum()
    tn = ((true_dir == 0) & (pred_dir == 0)).sum()
    fp = ((true_dir == 0) & (pred_dir == 1)).sum()
    fn = ((true_dir == 1) & (pred_dir == 0)).sum()

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)