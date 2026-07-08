"""
trabecular.quantum
==================
Pluggable quantum / classical similarity slot.

This sub-package defines the interface that both classical baselines and
quantum kernels must implement, so the DVC comparison step can swap backends
without touching anything upstream.

The published design rule
--------------------------
UMAP → quantum kernel preserves parity with the classical NCC baseline.
Linear reducers (PCA / random projection) lose ~9–12 accuracy points.
The reducer feeding the quantum slot is therefore the deciding factor.

Slot interface
--------------
Any backend must be callable as::

    score = backend(ref_patch, def_patch)   # → float in [-1, 1]

where ref_patch and def_patch are flattened sub-volumes (1D float arrays).

Available backends
------------------
from trabecular.quantum import ncc_similarity       # classical NCC (default)
from trabecular.quantum import qrc_similarity       # QRC-based similarity
from trabecular.quantum.classical_baseline import   # UMAP + NCC baseline
    UMAPSimilarity

SIMILARITY_BACKENDS dict maps name → callable for the Pipeline page selector.
"""
from __future__ import annotations
import numpy as np


# ── Classical NCC ────────────────────────────────────────────────────────────

def ncc_similarity(ref_patch: np.ndarray, def_patch: np.ndarray) -> float:
    """
    Normalised cross-correlation: scalar in [-1, 1].
    This is the default classical DVC similarity metric.
    """
    r = ref_patch - ref_patch.mean()
    d = def_patch - def_patch.mean()
    denom = (np.linalg.norm(r) * np.linalg.norm(d))
    return float(np.dot(r, d) / denom) if denom > 1e-10 else 0.0


# ── QRC similarity (uses scripts/quantum_reservoir.py) ───────────────────────

def qrc_similarity(ref_patch: np.ndarray, def_patch: np.ndarray,
                   n_qubits: int = 4, n_layers: int = 3,
                   seed: int = 42) -> float:
    """
    Quantum Reservoir Computing similarity: cosine similarity between
    the reservoir feature vectors of the two patches.

    Input patches are compressed to n_qubits features via PCA before
    angle-encoding (patches are typically much higher-dimensional than n_qubits).
    """
    try:
        from scripts.quantum_reservoir import QuantumReservoir  # type: ignore
    except ImportError:
        import sys
        from pathlib import Path
        _scripts = Path(__file__).resolve().parent.parent.parent / "scripts"
        sys.path.insert(0, str(_scripts))
        from quantum_reservoir import QuantumReservoir  # type: ignore

    qrc = QuantumReservoir(n_qubits=n_qubits, n_layers=n_layers, seed=seed)

    # Compress each patch to n_qubits features (variance-preserving projection)
    rng     = np.random.default_rng(seed)
    d       = len(ref_patch)
    proj    = rng.standard_normal((d, n_qubits))
    proj   /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8

    r_feat  = ref_patch @ proj
    d_feat  = def_patch @ proj

    # Normalise to [-1, 1] for angle encoding
    scale  = max(np.abs(r_feat).max(), np.abs(d_feat).max(), 1e-8)
    r_feat /= scale; d_feat /= scale

    fv_r = qrc.feature_vector(r_feat)
    fv_d = qrc.feature_vector(d_feat)

    denom = np.linalg.norm(fv_r) * np.linalg.norm(fv_d)
    return float(np.dot(fv_r, fv_d) / denom) if denom > 1e-10 else 0.0


# ── Backend registry ─────────────────────────────────────────────────────────

SIMILARITY_BACKENDS: dict = {
    "NCC (classical)": ncc_similarity,
    "QRC similarity":  qrc_similarity,
}

__all__ = [
    "ncc_similarity",
    "qrc_similarity",
    "SIMILARITY_BACKENDS",
]
