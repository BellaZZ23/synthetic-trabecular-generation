"""
tests/test_quantum.py
======================
Tests for trabecular.quantum — the pluggable similarity slot.

Covers: NCC classical baseline, QRC similarity, backend registry,
and classical reducer baselines (PCA, random projection).

Run with plots:
    pytest tests/test_quantum.py --plots -v
"""
import numpy as np
import pytest
from trabecular.quantum import (
    ncc_similarity,
    qrc_similarity,
    SIMILARITY_BACKENDS,
)
from trabecular.quantum.classical_baseline import (
    PCASimilarity,
    RandomProjSimilarity,
    UMAPSimilarity,
    CLASSICAL_BASELINES,
)


@pytest.fixture
def patch_pair(rng):
    """Two similar patches (small perturbation) and two dissimilar patches."""
    ref   = rng.standard_normal(64).astype(np.float32)
    close = ref + rng.standard_normal(64) * 0.05
    far   = -ref + rng.standard_normal(64) * 0.05
    return ref, close, far


# ── NCC ───────────────────────────────────────────────────────────────────────

class TestNccSimilarity:
    def test_identical_patches(self, rng):
        p = rng.standard_normal(128).astype(float)
        assert abs(ncc_similarity(p, p) - 1.0) < 1e-6

    def test_opposite_patches(self, rng):
        p = rng.standard_normal(128).astype(float)
        assert abs(ncc_similarity(p, -p) + 1.0) < 1e-6

    def test_range(self, patch_pair):
        ref, close, far = patch_pair
        assert -1.0 <= ncc_similarity(ref, close) <= 1.0
        assert -1.0 <= ncc_similarity(ref, far)   <= 1.0

    def test_similar_greater_than_dissimilar(self, patch_pair):
        ref, close, far = patch_pair
        assert ncc_similarity(ref, close) > ncc_similarity(ref, far)

    def test_zero_patch_returns_zero(self):
        p = np.zeros(64)
        q = np.ones(64)
        result = ncc_similarity(p, q)
        assert result == 0.0

    def test_scalar_output(self, patch_pair):
        ref, close, _ = patch_pair
        result = ncc_similarity(ref, close)
        assert isinstance(result, float)


# ── QRC similarity ────────────────────────────────────────────────────────────

class TestQrcSimilarity:
    def test_range(self, patch_pair):
        ref, close, far = patch_pair
        s1 = qrc_similarity(ref, close, n_qubits=3, n_layers=2, seed=42)
        s2 = qrc_similarity(ref, far,   n_qubits=3, n_layers=2, seed=42)
        assert -1.0 <= s1 <= 1.0
        assert -1.0 <= s2 <= 1.0

    def test_identical_patches_high_similarity(self, rng):
        p  = rng.standard_normal(32).astype(float)
        s  = qrc_similarity(p, p, n_qubits=3, n_layers=2, seed=0)
        assert s > 0.9, f"Identical patches should have high QRC similarity, got {s:.4f}"

    def test_reproducible(self, patch_pair):
        ref, close, _ = patch_pair
        s1 = qrc_similarity(ref, close, n_qubits=3, n_layers=2, seed=42)
        s2 = qrc_similarity(ref, close, n_qubits=3, n_layers=2, seed=42)
        assert abs(s1 - s2) < 1e-10

    def test_scalar_output(self, patch_pair):
        ref, close, _ = patch_pair
        result = qrc_similarity(ref, close, n_qubits=3, n_layers=2, seed=0)
        assert isinstance(result, float)


# ── Backend registry ──────────────────────────────────────────────────────────

class TestSimilarityBackends:
    def test_registry_has_ncc(self):
        assert "NCC (classical)" in SIMILARITY_BACKENDS

    def test_registry_has_qrc(self):
        assert "QRC similarity" in SIMILARITY_BACKENDS

    def test_all_backends_callable(self, patch_pair):
        ref, close, _ = patch_pair
        for name, fn in SIMILARITY_BACKENDS.items():
            result = fn(ref, close)
            assert isinstance(result, float), f"{name} did not return float"
            assert -1.0 <= result <= 1.0, f"{name} out of range: {result}"

    def test_classical_baselines_registry(self):
        for name, cls in CLASSICAL_BASELINES.items():
            assert callable(cls), f"{name} not callable"


# ── Classical baselines ───────────────────────────────────────────────────────

class TestPCASimilarity:
    def test_identical_patches(self, rng):
        patches = rng.standard_normal((20, 64)).astype(float)
        sim = PCASimilarity(n_components=4).fit(patches)
        p   = patches[0]
        assert abs(sim(p, p) - 1.0) < 1e-5

    def test_range(self, rng):
        patches = rng.standard_normal((20, 64)).astype(float)
        sim = PCASimilarity(n_components=4).fit(patches)
        for i in range(5):
            s = sim(patches[i], patches[i+1])
            assert -1.0 <= s <= 1.0


class TestRandomProjSimilarity:
    def test_identical_patches(self, rng):
        patches = rng.standard_normal((10, 64)).astype(float)
        sim = RandomProjSimilarity(n_components=8, seed=0).fit(patches)
        p   = patches[0]
        assert abs(sim(p, p) - 1.0) < 1e-5

    def test_range(self, rng):
        patches = rng.standard_normal((10, 64)).astype(float)
        sim = RandomProjSimilarity(n_components=8, seed=0).fit(patches)
        s   = sim(patches[0], patches[1])
        assert -1.0 <= s <= 1.0


# ── Comparison plot: NCC vs QRC vs PCA on a range of patch pairs ─────────────

def test_plot_similarity_comparison(rng, save_plot):
    import matplotlib.pyplot as plt

    n_pairs   = 30
    noise_levels = np.linspace(0.0, 2.0, n_pairs)
    base      = rng.standard_normal(64).astype(float)

    ncc_scores  = []
    qrc_scores  = []
    pca_scores  = []

    patches_all = [base + rng.standard_normal(64) * s for s in noise_levels]
    patches_arr = np.stack(patches_all)
    pca = PCASimilarity(n_components=8).fit(patches_arr)

    for noisy in patches_all:
        ncc_scores.append(ncc_similarity(base, noisy))
        qrc_scores.append(qrc_similarity(base, noisy, n_qubits=3, n_layers=2, seed=0))
        pca_scores.append(pca(base, noisy))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(noise_levels, ncc_scores,  "o-",  color="#378ADD", lw=2, label="NCC (classical)")
    ax.plot(noise_levels, qrc_scores,  "s--", color="#7F77DD", lw=2, label="QRC similarity")
    ax.plot(noise_levels, pca_scores,  "^:",  color="#E85D3A", lw=2, label="PCA + NCC")
    ax.axhline(0, color="gray", lw=1, ls=":")
    ax.set_xlabel("Noise level added to patch")
    ax.set_ylabel("Similarity score")
    ax.set_title("Similarity backends vs noise level\n"
                 "(base patch vs progressively noisier version)",
                 fontsize=11, fontweight="bold")
    ax.legend(); ax.set_ylim(-1.1, 1.1)
    plt.tight_layout()
    save_plot(fig, "04_similarity_comparison")
