# cluster_eval.py
# ==============================================================================
# Honest cluster evaluation — replaces the in-sample da = max(up, 1-up) gate.
#
# THE BUG it fixes:
#   da = max(up_ratio, 1 - up_ratio)  picks the favourable side AFTER seeing the
#   data, so it is >= 0.5 by construction and inflated on small clusters. It
#   then feeds good_cluster selection (da > 0.55) and est_pnl = mean_abs * da.
#
# THE FIX:
#   majority direction is fixed on TRAIN; DA and PnL are measured on a held-out
#   TEST split in THAT fixed direction. directed_oos_da CAN be < 0.5 (a cluster
#   with no real edge scores ~0.5; a two-sided/overfit one can score below).
#   est_pnl_oos uses mean SIGNED return - cost (no magnitude x accuracy inflation).
#
# Part of the prediction/ package. Used by cluster_engine (replace the
# in-sample block) and cluster_validation (reporting).
# ==============================================================================
from __future__ import annotations
import numpy as np


def directed_oos_da(majority_dir: int, test_dirs: np.ndarray) -> float:
    """Fraction of test events moving in the (train-fixed) majority direction.
    Honest: NOT max()'d. ~0.5 for a directionless cluster, <0.5 if the
    training direction reverses out of sample."""
    test_dirs = np.asarray(test_dirs)
    m = test_dirs != 0
    if m.sum() == 0:
        return float("nan")
    return float((np.sign(test_dirs[m]) == np.sign(majority_dir)).mean())


def select_good_clusters_oos(
    labels_tr: np.ndarray, dirs_tr: np.ndarray,
    labels_te: np.ndarray, dirs_te: np.ndarray, ret_te_bps: np.ndarray,
    taker_cost: float, da_min: float = 0.55, min_n_test: int = 30,
    min_n_train: int = 10, require_pnl: bool = True,
):
    """
    Majority direction + side chosen on TRAIN; DA / PnL measured OOS on TEST.

    Returns (good_set, majority_map, stats_per_cluster). A cluster is 'good'
    only if BOTH the directed OOS DA clears da_min AND the directed OOS PnL is
    positive — both on held-out data.
    """
    good, majority, stats = [], {}, {}
    for c in np.unique(labels_tr):
        m_tr = labels_tr == c
        if m_tr.sum() < min_n_train:
            continue
        up_tr = float((dirs_tr[m_tr] > 0).mean())
        maj = 1 if up_tr > 0.5 else -1
        majority[int(c)] = maj

        m_te = labels_te == c
        n_te = int(m_te.sum())
        if n_te < min_n_test:
            stats[int(c)] = dict(n_test=n_te, majority=maj, up_train=round(up_tr, 3),
                                 da_oos=float("nan"), est_pnl_oos=float("nan"),
                                 too_small=True)
            continue

        da_oos = directed_oos_da(maj, dirs_te[m_te])
        signed_ret = ret_te_bps[m_te] * maj          # honest signed return
        est_pnl_oos = float(signed_ret.mean() - taker_cost)

        stats[int(c)] = dict(
            n_test=n_te, majority=maj, up_train=round(up_tr, 3),
            da_oos=round(da_oos, 4), est_pnl_oos=round(est_pnl_oos, 2),
            too_small=False,
        )
        # DA-only gate when require_pnl=False (4.4: signal only; PnL -> 4.5).
        # PnL is still computed and reported, just not used as a gate.
        pnl_ok = (est_pnl_oos > 0) if require_pnl else True
        if (not np.isnan(da_oos)) and da_oos > da_min and pnl_ok:
            good.append(int(c))
    return set(good), majority, stats


def characterisation_verdict(da_oos: float, feature_consistency: float,
                             n_test: int, est_pnl_oos: float,
                             da_min: float = 0.55, fc_min: float = 0.80,
                             n_min: int = 30) -> dict:
    """
    Extended lens: separate PREDICTIVE/STRUCTURAL viability from TRADEABILITY.
    feature_consistency comes from the validation script (share of defining
    features with stable signature across blocks).
    """
    predictive = (not np.isnan(da_oos)) and da_oos > da_min and n_test >= n_min
    stable     = (feature_consistency is not None) and feature_consistency >= fc_min
    tradeable  = (not np.isnan(est_pnl_oos)) and est_pnl_oos > 0
    return dict(
        characterisation_viable=bool(predictive and stable),  # the title's lens
        tradeable=bool(tradeable),                            # the economic test
        predictive=bool(predictive), stable=bool(stable),
    )