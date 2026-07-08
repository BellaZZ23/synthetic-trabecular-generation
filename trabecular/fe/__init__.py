"""
trabecular.fe
=============
Finite-element sub-package.

Public API
----------
run_fe_analysis      — voxel hex-mesh FE solver (from fe_coupling.coupling)
run_techmesh_analysis — faster tet-mesh FE solver (from fe_coupling.solver)
measure_all_bonej    — BoneJ-equivalent morphometrics (from fe_coupling.measurements)
"""
from trabecular.fe.solver import run_techmesh_analysis
from trabecular.fe.coupling import run_fe_analysis, generate_bone_volume, generate_bone_volume_calibrated, generate_grayscale
from trabecular.fe.measurements import measure_all_bonej

__all__ = [
    "run_fe_analysis",
    "run_techmesh_analysis",
    "generate_bone_volume",
    "generate_bone_volume_calibrated",
    "generate_grayscale",
    "measure_all_bonej",
]
