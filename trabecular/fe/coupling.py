"""
trabecular.fe.coupling
======================
Thin re-export layer that makes the fe_coupling scripts importable through
the package without path manipulation.

This module bridges the legacy ``fe_coupling/`` directory and the new
``trabecular`` package until the full refactor is complete. All public
functions are re-exported here so downstream code only ever needs::

    from trabecular.fe.coupling import run_fe_analysis, generate_bone_volume

Nothing in this module contains logic — it is purely an import shim.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure fe_coupling/ is importable
_FE_COUPLING = Path(__file__).resolve().parent.parent.parent / "fe_coupling"
_SCRIPTS     = Path(__file__).resolve().parent.parent.parent / "scripts"
for _p in (_FE_COUPLING, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from step3_generator_fe_coupling import (   # noqa: E402
    generate_bone_volume,
    generate_bone_volume_calibrated,
    generate_grayscale,
    run_fe_analysis,
)

__all__ = [
    "generate_bone_volume",
    "generate_bone_volume_calibrated",
    "generate_grayscale",
    "run_fe_analysis",
]
