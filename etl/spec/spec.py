# etl/spec/spec.py
# ==============================================================================
# Base dataclasses for feature specifications.
# Shared by all spec modules across all stages (S0, S1, ...).
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Dep:
    """
    Dependency descriptor.

    Compatible with S0FeatureEngine source selection:
      - If name starts with "source:", engine can disambiguate streams
        (e.g. "source:lob_deep", "source:trades").
      - match_params can be used for parameter-based matching.

    The `kind` field is unused in S0 (all deps point to raw sources via name).
    Later stages (S1+) may populate kind to reference computed feature names.
    """
    name: str
    match_params: Sequence[str] = field(default_factory=tuple)

    # Reserved for S1+ stages where features depend on other features.
    kind: str = ""


@dataclass(frozen=True)
class FeatureSpec:
    """
    Declarative feature spec: defines WHAT to compute, not HOW.
    """
    name: str
    stage: str
    operator: str
    params: Dict[str, Any] = field(default_factory=dict)

    # metadata (optional)
    label: str = ""
    group: str = ""
    description: str = ""

    # deps: keep your type, but we store immutably
    depends_on: Tuple[Dep, ...] = field(default_factory=tuple)

    # traceability (optional)
    feature_id: Optional[int] = None
    excel_row: Optional[int] = None

    def __post_init__(self) -> None:
        # accept list/tuple/Sequence but store as tuple
        deps = self.depends_on
        if not isinstance(deps, tuple):
            object.__setattr__(self, "depends_on", tuple(deps))