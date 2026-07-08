"""
trabecular
==========
Synthetic trabecular bone generation, micro-FE analysis, and DVC validation
pipeline — with a pluggable quantum similarity slot.

This package is the JOSS-ready library layer sitting beneath the Streamlit
dashboard. All core functions are importable without Streamlit.

Quick start
-----------
    from trabecular.generator import generate
    from trabecular.dvc import warp_volume, recover_displacement, displacement_rmse
    from trabecular.alignment import align_volumes
    from trabecular.fe import run_techmesh_analysis, measure_all_bonej
    from trabecular.validation import run_validation_loop
    from trabecular.quantum import ncc_similarity, qrc_similarity

Example
-------
    # Generate a synthetic bone volume
    vol = generate(nx=64, ny=64, nz=32, target_bvtv=0.33, seed=42)
    mask = vol["bone_mask"]

    # Run FE analysis
    fe = run_techmesh_analysis(mask, voxel_mm=0.039)

    # Known-deformation validation round-trip
    result = run_validation_loop(mask, voxel_um=39.0, verbose=True)
    print(f"DVC RMSE: {result['rmse_voxels']:.3f} voxels")

Package layout
--------------
trabecular/
    generator.py          Bone volume generation (v15.3 GRF zero-crossing)
    alignment.py          Rigid + resample volume alignment
    dvc.py                Phase-correlation DVC: warp · recover · score
    validation.py         FE-driven known-deformation validation loop
    fe/
        solver.py         TechMesh tet-FE solver (scikit-fem)
        coupling.py       Import shim for fe_coupling/ scripts
        measurements.py   BoneJ-equivalent morphometrics
    quantum/
        __init__.py       NCC and QRC similarity backends + registry
        classical_baseline.py  UMAP / PCA / RandProj baselines
"""
from trabecular._version import __version__

# ── Always-available (numpy + scipy only) ────────────────────────────────────
from trabecular.alignment import align_volumes, AlignmentReport
from trabecular.dvc import (
    warp_volume,
    recover_displacement,
    displacement_rmse,
    uniform_strain_field,
    smooth_random_field,
)
from trabecular.quantum import ncc_similarity, qrc_similarity, SIMILARITY_BACKENDS

# ── Heavy imports: generator, FE solver, validation ──────────────────────────
# These pull in fe_coupling/ which needs scipy + optionally scikit-fem.
# Imported lazily so `import trabecular` works in environments where only
# the core stack (numpy, scipy, scikit-image) is installed.

def _lazy_import(name: str):
    """Raise a helpful ImportError when a heavy sub-module is first used."""
    def _stub(*args, **kwargs):
        raise ImportError(
            f"trabecular.{name} requires the full fe_coupling dependencies. "
            "Run: pip install trabecular[fe]"
        )
    return _stub


try:
    from trabecular.generator import generate, generate_calibrated, make_grayscale
except Exception:  # pragma: no cover
    generate = generate_calibrated = make_grayscale = _lazy_import("generator")

try:
    from trabecular.fe import (
        run_fe_analysis,
        run_techmesh_analysis,
        measure_all_bonej,
    )
except Exception:  # pragma: no cover
    run_fe_analysis = run_techmesh_analysis = _lazy_import("fe.solver")
    measure_all_bonej = _lazy_import("fe.measurements")

try:
    from trabecular.validation import (
        fe_displacement_to_voxel_field,
        make_fe_known_deformation,
        run_validation_loop,
    )
except Exception:  # pragma: no cover
    fe_displacement_to_voxel_field = _lazy_import("validation")
    make_fe_known_deformation = run_validation_loop = _lazy_import("validation")

__all__ = [
    "__version__",
    # Generator
    "generate",
    "generate_calibrated",
    "make_grayscale",
    # Alignment
    "align_volumes",
    "AlignmentReport",
    # DVC
    "warp_volume",
    "recover_displacement",
    "displacement_rmse",
    "uniform_strain_field",
    "smooth_random_field",
    # Validation
    "fe_displacement_to_voxel_field",
    "make_fe_known_deformation",
    "run_validation_loop",
    # FE
    "run_fe_analysis",
    "run_techmesh_analysis",
    "measure_all_bonej",
    # Quantum slot
    "ncc_similarity",
    "qrc_similarity",
    "SIMILARITY_BACKENDS",
]
