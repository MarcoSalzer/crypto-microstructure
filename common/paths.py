"""Centralised, parameterised paths.

Replaces the hard-coded remote paths of the original code. Resolution order for
the external data store:

1. environment variable ``THESIS_DATA_ROOT``
2. ``data_root`` in ``configs/paths.yaml``
3. fallback ``<repo>/sample_data`` (only the ingestion demo works with this)

The ~94 GB feature/data store is external to this repository; point
``THESIS_DATA_ROOT`` at it to run the full pipeline. The committed small
artifacts (feature-reduction CSVs, final cluster memberships) live under
``results/`` and are addressed via ``REDUCTION_DIR`` and ``RESULTS_DIR``.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml_config() -> dict:
    cfg_path = REPO_ROOT / "configs" / "paths.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml  # optional dependency
        with open(cfg_path) as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


_CFG = _load_yaml_config()


def _resolve(env_var: str, cfg_key: str, default: Path) -> Path:
    val = os.environ.get(env_var) or _CFG.get(cfg_key)
    if val:
        p = Path(val).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p)
    return default


# External feature/data store (raw_data, s0..s6_features, ml_features_log1p, ...).
DATA_ROOT: Path = _resolve("THESIS_DATA_ROOT", "data_root", REPO_ROOT / "sample_data")

# Committed small artifacts.
RESULTS_DIR: Path = _resolve("THESIS_RESULTS_DIR", "results_dir", REPO_ROOT / "results")
REDUCTION_DIR: Path = _resolve("THESIS_REDUCTION_DIR", "reduction_dir", RESULTS_DIR / "selection")
CLUSTER_FINAL_DIR: Path = RESULTS_DIR / "clustering" / "final"
SAMPLE_DATA_DIR: Path = REPO_ROOT / "sample_data"

__all__ = [
    "REPO_ROOT",
    "DATA_ROOT",
    "RESULTS_DIR",
    "REDUCTION_DIR",
    "CLUSTER_FINAL_DIR",
    "SAMPLE_DATA_DIR",
]
