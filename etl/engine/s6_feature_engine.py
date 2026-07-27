# ==============================================================================
# S6 Feature Engine — Cross-Asset (BTC ↔ ETH ↔ BNB)
# ==============================================================================
# Overview:
#   Computes cross-asset features by merging BTC, ETH and BNB S5 Parquet files
#   on timestamp, executing S6 operators in topological order, and writing the
#   resulting cross-asset features to a dedicated S6 Parquet file.
#
# ── Input paths ───────────────────────────────────────────────────────────────
#   BTC S5:  {base_dir}/s5_features/s5_features_btc_{date}_{hour:02d}.parquet
#   ETH S5:  {base_dir}/s5_features/s5_features_eth_{date}_{hour:02d}.parquet
#   BNB S5:  {base_dir}/s5_features/s5_features_bnb_{date}_{hour:02d}.parquet
#
# ── Output path ───────────────────────────────────────────────────────────────
#   S6 out:  {base_dir}/s6_features_{asset_tag}/s6_features_{asset_tag}_{date}_{hour:02d}.parquet
#            where {asset_tag} reflects actually available assets, e.g. btceth or btcethbnb
#
#   Contains only the ca_* feature columns; upstream S5 columns are NOT written.
#
# ── RAM budget ────────────────────────────────────────────────────────────────
#   3600 rows × ~4000 merged cols × 8 bytes ≈ 115 MB peak.
#   After _extract_s6_columns, only ~300 ca_* cols remain (~8 MB).
#   The merged DataFrame is explicitly deleted to free memory.
#
# ==============================================================================

from __future__ import annotations

import argparse
import gc
import logging
from collections import defaultdict, deque
from functools import reduce
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from etl.spec import FeatureSpec, Dep
from etl.spec.s6.s6_cross_asset import (
    S6_CROSS_ASSET_FEATURES,
    ALL_ASSETS,
    ASSET_PAIRS,
)
from etl.operators.s6_operators import S6_OPERATOR_REGISTRY

logger = logging.getLogger(__name__)


# ==============================================================================
# Constants
# ==============================================================================

_S5_SUBDIR   = "s5_features"
# Output is written to a per-asset-set subdirectory:
#   s6_features_btceth/     — always produced (BTC+ETH only)
#   s6_features_btcethbnb/  — produced additionally when BNB is available

_INTERMEDIARY_GROUP = "Cross-Asset-Intermediary"

# ── Per-group NaN thresholds (fraction of rows) ────────────────────────────
_NAN_THRESH: Dict[str, float] = {
    "Cross-Asset":              0.15,
    "Cross-Asset-Intermediary": 0.15,
}

# ── Per-feature overrides for features with longer rolling windows ─────────
_NAN_THRESH_BY_FEATURE: Dict[str, float] = {}
for _a in ALL_ASSETS:
    for _b in ALL_ASSETS:
        if _a == _b:
            continue
        for _l in [1, 3, 5]:
            _NAN_THRESH_BY_FEATURE[
                f"ca_lag_corr_{_a}_taker_lead_{_b}_ret_{_l}s"
            ] = 0.12
for _pair in ASSET_PAIRS:
    _tag = f"{_pair[0]}{_pair[1]}"
    _NAN_THRESH_BY_FEATURE[f"ca_net_pressure_persist_spot_5bps_spread_300s_{_tag}"] = 0.12
    _NAN_THRESH_BY_FEATURE[f"ca_net_pressure_persist_fut_5bps_spread_300s_{_tag}"] = 0.12

# F26: Activity z-score intermediaries have 900s rolling window → first 899 rows
# NaN in a 3600-row hourly chunk ≈ 25%.  Their diffs inherit the same warmup.
for _asset in ALL_ASSETS:
    _NAN_THRESH_BY_FEATURE[f"ca_z_trade_count_spot_300s_{_asset}"] = 0.27
    _NAN_THRESH_BY_FEATURE[f"ca_z_avg_trade_size_spot_900s_{_asset}"] = 0.27
for _pair in ASSET_PAIRS:
    _tag = f"{_pair[0]}{_pair[1]}"
    _NAN_THRESH_BY_FEATURE[f"ca_activity_trade_count_spot_300s_spread_{_tag}"] = 0.27
    _NAN_THRESH_BY_FEATURE[f"ca_activity_avg_trade_size_spot_900s_spread_{_tag}"] = 0.27


# ==============================================================================
# Path helpers
# ==============================================================================

def _s5_path(base_dir: str, asset: str, date: str, hour: int) -> Path:
    return (
        Path(base_dir)
        / _S5_SUBDIR
        / f"s5_features_{asset}_{date}_{hour:02d}.parquet"
    )


def _s6_path(base_dir: str, date: str, hour: int, assets: List[str]) -> Path:
    """
    Output path for a given asset set, inside its own subdirectory:
        s6_features_btceth/s6_features_btceth_{date}_{hh}.parquet
        s6_features_btcethbnb/s6_features_btcethbnb_{date}_{hh}.parquet
    """
    tag     = "".join(sorted(assets))
    out_dir = Path(base_dir) / f"s6_features_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"s6_features_{tag}_{date}_{hour:02d}.parquet"


# ==============================================================================
# Load & merge helpers
# ==============================================================================

def _load_s5(path: Path, asset: str) -> pd.DataFrame:
    """Load a S5 Parquet file and suffix all columns with _{asset}."""
    if not path.exists():
        raise FileNotFoundError(
            f"S5 Parquet not found for asset={asset}: {path}\n"
            f"  Ensure the S5 pipeline has run for this date/hour."
        )
    df = pd.read_parquet(path)
    logger.info(
        "Loaded S5 %-3s | %d rows × %d cols | %s",
        asset.upper(), len(df), len(df.columns), path.name,
    )
    df.columns = [f"{c}_{asset}" for c in df.columns]
    return df


def _merge_all_assets(
    asset_dfs: Dict[str, pd.DataFrame],
    how: str = "inner",
) -> pd.DataFrame:
    """Join all asset S5 DataFrames on their shared timestamp index."""
    dfs = list(asset_dfs.values())
    merged = reduce(lambda left, right: left.join(right, how=how), dfs)

    for asset, df in asset_dfs.items():
        n_dropped = len(df) - len(merged)
        if n_dropped > 0:
            logger.warning(
                "Merge: %d %s-only rows dropped (inner join).",
                n_dropped, asset.upper(),
            )

    total_cols = sum(len(df.columns) for df in asset_dfs.values())
    logger.info(
        "Merged | %d rows × %d cols  (from %d assets, %d source cols)",
        len(merged), len(merged.columns), len(asset_dfs), total_cols,
    )
    return merged


# ==============================================================================
# Topological sort  (Kahn's algorithm)
# ==============================================================================

def _topological_sort(specs: List[FeatureSpec]) -> List[FeatureSpec]:
    name_to_spec: Dict[str, FeatureSpec] = {s.name: s for s in specs}
    in_degree:  Dict[str, int]        = defaultdict(int)
    successors: Dict[str, List[str]]  = defaultdict(list)

    for spec in specs:
        in_degree[spec.name]
        for dep in spec.depends_on:
            if dep.name in name_to_spec:
                successors[dep.name].append(spec.name)
                in_degree[spec.name] += 1

    queue:        deque[str] = deque(n for n, d in in_degree.items() if d == 0)
    sorted_names: List[str]  = []

    while queue:
        node = queue.popleft()
        sorted_names.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(sorted_names) != len(specs):
        cycle_nodes = [n for n, d in in_degree.items() if d > 0]
        raise ValueError(
            f"Dependency cycle in S6 specs — cannot topologically sort. "
            f"Unresolved nodes ({len(cycle_nodes)}): {cycle_nodes}"
        )

    return [name_to_spec[n] for n in sorted_names]


# ==============================================================================
# NaN audit
# ==============================================================================

def _nan_audit(
    df_s6: pd.DataFrame,
    specs: List[FeatureSpec],
    available_assets: List[str],
) -> None:
    """
    Audit NaN fractions for computed features.
    Specs that reference assets not in `available_assets` are silently skipped
    (they were intentionally not computed for this asset set).
    """
    missing_assets = set(ALL_ASSETS) - set(available_assets)
    group_stats: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "over": []})

    for spec in specs:
        # Skip features that belong to a pair/asset not in this run
        if any(f"_{a}" in spec.name for a in missing_assets):
            continue

        if spec.name not in df_s6.columns:
            logger.error("NaN audit: column '%s' absent from output!", spec.name)
            continue

        nan_frac = df_s6[spec.name].isna().mean()
        grp      = spec.group
        thresh   = _NAN_THRESH_BY_FEATURE.get(
            spec.name,
            _NAN_THRESH.get(grp, 0.15),
        )

        group_stats[grp]["total"] += 1
        if nan_frac > thresh:
            group_stats[grp]["over"].append(
                (spec.name, f"{nan_frac:.1%}", f"thresh={thresh:.0%}")
            )

    for grp, stats in sorted(group_stats.items()):
        if stats["over"]:
            logger.warning(
                "NaN audit [%-28s]: %d/%d features over threshold:\n    %s",
                grp, len(stats["over"]), stats["total"],
                "\n    ".join(f"{n}  {f}  ({t})" for n, f, t in stats["over"]),
            )
        else:
            logger.info(
                "NaN audit [%-28s]: all %d features within threshold",
                grp, stats["total"],
            )


# ==============================================================================
# Feature computation
# ==============================================================================

def _compute_features(
    df_merged: pd.DataFrame,
    specs: List[FeatureSpec],
) -> pd.DataFrame:
    computed = 0
    skipped  = 0
    errors   = 0

    for spec in specs:
        op_fn = S6_OPERATOR_REGISTRY.get(spec.operator)
        if op_fn is None:
            logger.error(
                "Unknown operator '%s' for '%s' — skipping.",
                spec.operator, spec.name,
            )
            errors += 1
            continue

        missing = [
            d.name for d in spec.depends_on
            if d.kind == "col" and d.name not in df_merged.columns
        ]
        if missing:
            logger.debug(
                "Feature '%s': missing upstream columns %s — skipping.",
                spec.name, missing,
            )
            skipped += 1
            continue

        try:
            df_merged[spec.name] = op_fn(df_merged, **spec.params)
            computed += 1
        except Exception as exc:
            logger.exception("Feature '%s' raised: %s", spec.name, exc)
            errors += 1

    logger.info(
        "Computation: %d/%d features computed  (%d skipped, %d errors)",
        computed, len(specs), skipped, errors,
    )
    return df_merged


def _extract_s6_columns(
    df_merged: pd.DataFrame,
    specs: List[FeatureSpec],
) -> pd.DataFrame:
    s6_cols = [s.name for s in specs if s.name in df_merged.columns]
    missing  = [s.name for s in specs if s.name not in df_merged.columns]

    if missing:
        logger.info(
            "%d feature column(s) skipped (missing asset data): showing first 10: %s",
            len(missing), missing[:10],
        )

    return df_merged[s6_cols].copy()


# ==============================================================================
# Engine class
# ==============================================================================

class S6FeatureEngine:
    """
    Computes all S6 cross-asset features (BTC ↔ ETH ↔ BNB) for a given
    date and hour.

    Storage layout (default base_dir = "data_storage"):
        Input:   data_storage/s5_features/s5_features_{btc,eth,bnb}_{date}_{hh}.parquet
        Output:  data_storage/s6_features_{btceth|btcethbnb}/s6_features_{btceth|btcethbnb}_{date}_{hh}.parquet
    """

    def __init__(
        self,
        base_dir:  str  = "data_storage",
        merge_how: str  = "inner",
        overwrite: bool = False,
    ) -> None:
        self.base_dir  = base_dir
        self.merge_how = merge_how
        self.overwrite = overwrite

        self._specs: List[FeatureSpec] = _topological_sort(S6_CROSS_ASSET_FEATURES)
        logger.debug(
            "S6FeatureEngine ready — %d specs sorted  |  base_dir=%s",
            len(self._specs), base_dir,
        )

    def _run_asset_set(
        self,
        date:        str,
        hour:        int,
        asset_paths: Dict[str, Path],
    ) -> Optional[Path]:
        """
        Core computation for one specific asset set (e.g. btceth or btcethbnb).
        Merges the given asset S5 files, computes all applicable S6 features,
        and writes the result to the asset-set-specific output directory.

        Returns the output Path on success, None otherwise.
        """
        assets    = sorted(asset_paths.keys())
        out_path  = _s6_path(self.base_dir, date, hour, assets=assets)
        asset_tag = "".join(assets)

        if out_path.exists() and not self.overwrite:
            logger.info("S6 [%s] exists, skipping: %s", asset_tag, out_path.name)
            return out_path

        # Load S5 inputs for this asset set
        asset_dfs: Dict[str, pd.DataFrame] = {}
        for asset, path in asset_paths.items():
            asset_dfs[asset] = _load_s5(path, asset)

        # Merge on timestamp
        df_merged = _merge_all_assets(asset_dfs, how=self.merge_how)
        del asset_dfs
        gc.collect()

        if len(df_merged) == 0:
            logger.error(
                "S6 [%s] merged DataFrame is empty — no overlapping timestamps. Aborting.",
                asset_tag,
            )
            return None

        # Compute features (topological order; specs for missing assets are
        # silently skipped inside _compute_features via missing-column check)
        df_merged = _compute_features(df_merged, self._specs)

        # Strip upstream columns → keep only ca_* S6 features
        df_s6 = _extract_s6_columns(df_merged, self._specs)
        del df_merged
        gc.collect()

        # NaN audit (only audits features relevant to this asset set)
        _nan_audit(df_s6, self._specs, available_assets=assets)

        # Write
        df_s6.to_parquet(out_path, index=True, compression="snappy")
        size_kb = out_path.stat().st_size // 1024
        logger.info(
            "Written [%s]: %s  [%d rows × %d cols, %d KB]",
            asset_tag, out_path.name, len(df_s6), len(df_s6.columns), size_kb,
        )
        return out_path

    def run(self, date: str, hour: int) -> List[Path]:
        """
        Compute S6 cross-asset features for `date` / `hour`.

        Always produces a btceth output.  When BNB is also available for the
        same date/hour, an additional btcethbnb output is produced.

        Returns a list of successfully written Paths (1 or 2 elements).
        """
        logger.info("━━ S6 run  date=%s  hour=%02d ━━", date, hour)

        # Discover which S5 files exist for this date/hour
        available_paths: Dict[str, Path] = {}
        for asset in ALL_ASSETS:
            path = _s5_path(self.base_dir, asset, date, hour)
            if path.exists():
                available_paths[asset] = path
            else:
                logger.info(
                    "S5 not found for %s — will be excluded from this hour's runs: %s",
                    asset.upper(), path.name,
                )

        if not {"btc", "eth"}.issubset(available_paths):
            logger.error(
                "S6 aborted for %s H%02d — BTC and ETH S5 files are required.",
                date, hour,
            )
            return []

        written: List[Path] = []

        # ── Run 1: BTC + ETH only (always) ────────────────────────────────────
        btceth_paths = {a: available_paths[a] for a in ["btc", "eth"]}
        path = self._run_asset_set(date, hour, btceth_paths)
        if path is not None:
            written.append(path)

        # ── Run 2: BTC + ETH + BNB (only when BNB is available) ───────────────
        if "bnb" in available_paths:
            btcethbnb_paths = {a: available_paths[a] for a in ["btc", "eth", "bnb"]}
            path = self._run_asset_set(date, hour, btcethbnb_paths)
            if path is not None:
                written.append(path)
        else:
            logger.info(
                "BNB not available for %s H%02d — skipping btcethbnb output.",
                date, hour,
            )

        return written

    def run_range(
        self,
        date:  str,
        hours: Optional[List[int]] = None,
    ) -> List[Path]:
        if hours is None:
            hours = list(range(24))

        written: List[Path] = []
        for h in hours:
            written.extend(self.run(date=date, hour=h))

        logger.info(
            "run_range complete: %d output(s) written across %d hours  (date=%s)",
            len(written), len(hours), date,
        )
        return written


# ==============================================================================
# Feature summary utility
# ==============================================================================

def feature_summary(specs: Optional[List[FeatureSpec]] = None) -> pd.DataFrame:
    if specs is None:
        specs = S6_CROSS_ASSET_FEATURES

    rows = []
    for s in specs:
        rows.append({
            "feature_id": s.feature_id,
            "name":       s.name,
            "operator":   s.operator,
            "group":      s.group,
            "col_a":      s.params.get("col_a",
                          s.params.get("btc_col",
                          s.params.get("lead_col",
                          s.params.get("y_col", "—")))),
            "col_b":      s.params.get("col_b",
                          s.params.get("eth_col",
                          s.params.get("lag_col",
                          s.params.get("x_col", "—")))),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("feature_id")
        .reset_index(drop=True)
    )


# ==============================================================================
# CLI entry point
# ==============================================================================

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "S6 Feature Engine — compute cross-asset BTC/ETH/BNB features.\n"
            "\n"
            "Input:  data_storage/s5_features/s5_features_{btc,eth,bnb}_{date}_{hour}.parquet\n"
            "Output: data_storage/s6_features_{btceth|btcethbnb}/s6_features_{btceth|btcethbnb}_{date}_{hour}.parquet"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date", required=True, help="ISO date, e.g. 2024-01-15")
    parser.add_argument("--hour", type=int, default=None, help="Single hour 0–23.")
    parser.add_argument("--hours", nargs="+", type=int, default=None)
    parser.add_argument("--base-dir", default="data_storage")
    parser.add_argument("--merge-how", default="inner", choices=["inner", "outer"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary", action="store_true",
                        help="Print the feature catalogue table and exit.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.summary:
        df = feature_summary()
        print(df.to_string(index=False))
        n_primary = df["group"].eq("Cross-Asset").sum()
        n_interm = df["group"].eq("Cross-Asset-Intermediary").sum()
        print(f"\nTotal: {len(df)} features  "
              f"({n_primary} primary, {n_interm} intermediary)")
        return

    engine = S6FeatureEngine(
        base_dir  = args.base_dir,
        merge_how = args.merge_how,
        overwrite = args.overwrite,
    )

    if args.hour is not None:
        paths = engine.run(date=args.date, hour=args.hour)
        for p in paths:
            print(f"  Written: {p}")
    else:
        paths = engine.run_range(date=args.date, hours=args.hours)
        print(f"  {len(paths)} output(s) written.")


if __name__ == "__main__":
    _cli()