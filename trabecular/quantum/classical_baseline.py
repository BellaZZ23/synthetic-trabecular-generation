"""
trabecular.quantum.classical_baseline
======================================
Classical similarity baselines that run through the same reducer → kernel
pipeline as the quantum slot, for fair comparison.

Published result
----------------
UMAP reducer → NCC: matches quantum-kernel parity.
PCA / random-projection → NCC: loses ~9–12 accuracy points.
The reducer is therefore the deciding factor — not just the kernel.

Classes
-------
UMAPSimilarity       UMAP-reduced NCC (best classical baseline)
PCASimilarity        PCA-reduced NCC
RandomProjSimilarity Random-projection NCC
"""
from __future__ import annotations
import numpy as np
from typing import Optional


class _ReducedSimilarity:
    """Base class: reduce patches then measure NCC."""

    def __init__(self, n_components: int = 8, seed: int = 42):
        self.n_components = n_components
        self.seed         = seed
        self._fitted      = False

    def fit(self, patches: np.ndarray) -> "_ReducedSimilarity":
        """Fit reducer on a collection of patches (N, d)."""
        raise NotImplementedError

    def _reduce(self, patch: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, ref_patch: np.ndarray, def_patch: np.ndarray) -> float:
        """Similarity ∈ [-1, 1] between two patches."""
        if not self._fitted:
            # Lazy fit on the two patches themselves (no training data available)
            self.fit(np.stack([ref_patch, def_patch]))
        r = self._reduce(ref_patch)
        d = self._reduce(def_patch)
        r = r - r.mean(); d = d - d.mean()
        denom = np.linalg.norm(r) * np.linalg.norm(d)
        return float(np.dot(r, d) / denom) if denom > 1e-10 else 0.0


class PCASimilarity(_ReducedSimilarity):
    """PCA-reduced NCC similarity.

    Approximates the published classical baseline that loses ~9–12 points
    relative to UMAP when fed to a quantum kernel.
    """

    def fit(self, patches: np.ndarray) -> "PCASimilarity":
        X    = patches - patches.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        self._components = Vt[:self.n_components]  # (n_components, d)
        self._fitted     = True
        return self

    def _reduce(self, patch: np.ndarray) -> np.ndarray:
        return self._components @ (patch - patch.mean())


class RandomProjSimilarity(_ReducedSimilarity):
    """Random-projection NCC similarity (Johnson–Lindenstrauss).

    The weakest baseline — sets a floor for comparison.
    """

    def fit(self, patches: np.ndarray) -> "RandomProjSimilarity":
        rng = np.random.default_rng(self.seed)
        d   = patches.shape[1]
        self._proj   = rng.standard_normal((d, self.n_components))
        self._proj  /= np.linalg.norm(self._proj, axis=0, keepdims=True) + 1e-8
        self._fitted = True
        return self

    def _reduce(self, patch: np.ndarray) -> np.ndarray:
        return patch @ self._proj


class UMAPSimilarity(_ReducedSimilarity):
    """
    UMAP-reduced NCC similarity — the best classical baseline.

    Published finding: UMAP preserves quantum–classical parity while linear
    reducers lose ~9–12 accuracy points. Requires ``umap-learn``.

    Parameters
    ----------
    n_components  : embedding dimension (default 8; must be ≤ n_patches − 2)
    n_neighbors   : UMAP neighbourhood size (default 5)
    min_dist      : UMAP minimum distance (default 0.1)
    seed          : random state

    Usage
    -----
    # Offline: fit on a training set of patches
    sim = UMAPSimilarity(n_components=8).fit(training_patches)

    # Online: score pairs
    score = sim(ref_patch, def_patch)
    """

    def __init__(self, n_components: int = 8, n_neighbors: int = 5,
                 min_dist: float = 0.1, seed: int = 42):
        super().__init__(n_components=n_components, seed=seed)
        self.n_neighbors = n_neighbors
        self.min_dist    = min_dist
        self._reducer: Optional[object] = None

    def fit(self, patches: np.ndarray) -> "UMAPSimilarity":
        try:
            import umap  # type: ignore
        except ImportError:
            raise ImportError(
                "umap-learn is required for UMAPSimilarity. "
                "Install with: pip install umap-learn"
            )
        n = min(self.n_components, patches.shape[0] - 2)
        self._reducer = umap.UMAP(
            n_components=n,
            n_neighbors=min(self.n_neighbors, patches.shape[0] - 1),
            min_dist=self.min_dist,
            random_state=self.seed,
        )
        self._embedding = self._reducer.fit_transform(patches)
        self._fitted    = True
        return self

    def _reduce(self, patch: np.ndarray) -> np.ndarray:
        if self._reducer is None:
            raise RuntimeError("Call fit() with a set of patches first.")
        return self._reducer.transform(patch[None])[0]


# ── Convenience registry ──────────────────────────────────────────────────────

CLASSICAL_BASELINES: dict = {
    "UMAP + NCC":   UMAPSimilarity,
    "PCA + NCC":    PCASimilarity,
    "RandProj + NCC": RandomProjSimilarity,
}

__all__ = [
    "UMAPSimilarity",
    "PCASimilarity",
    "RandomProjSimilarity",
    "CLASSICAL_BASELINES",
]
