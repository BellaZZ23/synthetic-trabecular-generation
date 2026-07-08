"""
tests/test_qrc.py
==================
Tests for scripts/quantum_reservoir.py — Quantum Reservoir Computing.

Covers: state-vector circuit, entanglement entropy, sweet spot detection,
classification accuracy, and temporal (load-sequence) mode.

Run with plots:
    pytest tests/test_qrc.py --plots -v
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from quantum_reservoir import (
    QuantumReservoir,
    TemporalQuantumReservoir,
    entanglement_entropy,
)


# ── Entanglement entropy ──────────────────────────────────────────────────────

class TestEntanglementEntropy:
    def test_product_state_zero_entropy(self):
        """|0...0⟩ is a product state → entropy = 0."""
        n     = 4
        state = np.zeros(2**n, dtype=complex)
        state[0] = 1.0
        S = entanglement_entropy(state, n)
        assert abs(S) < 1e-10

    def test_bell_state_entropy(self):
        """(|00⟩ + |11⟩)/√2 has entropy = 1 bit."""
        state = np.zeros(4, dtype=complex)
        state[0] = state[3] = 1.0 / np.sqrt(2)
        S = entanglement_entropy(state, n=2, partition=1)
        assert abs(S - 1.0) < 1e-6

    def test_entropy_non_negative(self):
        rng   = np.random.default_rng(0)
        n     = 4
        state = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        state /= np.linalg.norm(state)
        S = entanglement_entropy(state, n)
        assert S >= -1e-10

    def test_entropy_bounded_by_partition(self):
        """S ≤ partition (in bits)."""
        n       = 4
        rng     = np.random.default_rng(1)
        state   = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        state  /= np.linalg.norm(state)
        partition = 2
        S = entanglement_entropy(state, n, partition=partition)
        assert S <= partition + 1e-6


# ── Feature vector ────────────────────────────────────────────────────────────

class TestQuantumReservoirFeatures:
    @pytest.fixture
    def qrc(self):
        return QuantumReservoir(n_qubits=4, n_layers=3, seed=42)

    def test_feature_vector_shape_pauli_z(self):
        qrc = QuantumReservoir(n_qubits=4, n_layers=2, seed=0, readout="pauli_z")
        x   = np.array([0.1, 0.3, -0.2, 0.5])
        fv  = qrc.feature_vector(x)
        assert fv.shape == (4,), f"Expected (4,), got {fv.shape}"

    def test_feature_vector_shape_correlators(self):
        qrc = QuantumReservoir(n_qubits=4, n_layers=2, seed=0, readout="correlators")
        x   = np.array([0.1, 0.3, -0.2, 0.5])
        fv  = qrc.feature_vector(x)
        assert fv.shape == (8,), f"Expected (8,), got {fv.shape}"

    def test_feature_matrix_shape(self, qrc):
        X  = np.random.default_rng(5).uniform(-1, 1, (10, 4))
        Phi = qrc.feature_matrix(X)
        assert Phi.shape == (10, 8)

    def test_reproducible(self, qrc):
        x  = np.array([0.2, -0.4, 0.6, -0.1])
        f1 = qrc.feature_vector(x)
        f2 = qrc.feature_vector(x)
        assert np.allclose(f1, f2)

    def test_different_inputs_different_features(self, qrc):
        rng = np.random.default_rng(99)
        x1  = rng.uniform(-1, 1, 4)
        x2  = rng.uniform(-1, 1, 4)
        f1  = qrc.feature_vector(x1)
        f2  = qrc.feature_vector(x2)
        assert not np.allclose(f1, f2)

    def test_pauli_z_range(self, qrc):
        """⟨Zᵢ⟩ values must lie in [-1, 1]."""
        qrc_z = QuantumReservoir(n_qubits=4, n_layers=3, seed=42, readout="pauli_z")
        x  = np.array([0.1, -0.3, 0.7, -0.5])
        fv = qrc_z.feature_vector(x)
        assert np.all(fv >= -1.0 - 1e-6) and np.all(fv <= 1.0 + 1e-6)


# ── Linear readout ────────────────────────────────────────────────────────────

class TestLinearReadout:
    @pytest.fixture(scope="class")
    def trained_qrc(self):
        rng = np.random.default_rng(0)
        X   = rng.standard_normal((80, 4))
        y   = (X[:, 0] > 0).astype(float)
        qrc = QuantumReservoir(n_qubits=4, n_layers=8, seed=42)
        split = 56
        qrc.fit(X[:split], y[:split])
        return qrc, X, y, split

    def test_predict_shape(self, trained_qrc):
        qrc, X, _, split = trained_qrc
        preds = qrc.predict(X[split:])
        assert preds.shape == (len(X) - split,)

    def test_predict_class_binary(self, trained_qrc):
        qrc, X, _, split = trained_qrc
        preds = qrc.predict_class(X[split:])
        assert set(np.unique(preds)).issubset({0, 1})


def test_qrc_accuracy_above_random():
    """QRC should beat random (50%) on a linearly separable problem.

    Kept outside the class to avoid class-scoped fixture sharing.
    Verified: seed=0, n_layers=8 → 75% test accuracy.
    """
    rng   = np.random.default_rng(0)
    X     = rng.standard_normal((80, 4))
    y     = (X[:, 0] > 0).astype(float)
    qrc   = QuantumReservoir(n_qubits=4, n_layers=8, seed=42)
    split = 56
    qrc.fit(X[:split], y[:split])
    acc   = float((qrc.predict_class(X[split:]) == y[split:]).mean())
    print(f"\n  QRC test accuracy: {acc:.1%}")
    assert acc > 0.55, f"Expected > 55% accuracy, got {acc:.1%}"


# ── Entanglement sweet spot ───────────────────────────────────────────────────

class TestEntropyVsDepth:
    def test_zero_entropy_at_depth_zero(self):
        qrc = QuantumReservoir(n_qubits=4, n_layers=8, seed=0)
        x   = np.array([0.1, 0.2, 0.3, 0.4])
        curve = qrc.entropy_vs_depth(x, max_layers=8)
        assert curve[0] < 1e-6, "Initial |0...0⟩ state has zero entropy"

    def test_entropy_increases(self):
        qrc   = QuantumReservoir(n_qubits=4, n_layers=8, seed=42)
        x     = np.array([0.2, -0.3, 0.5, -0.1])
        curve = qrc.entropy_vs_depth(x, max_layers=8)
        # Entropy should generally increase (not necessarily monotone, but
        # the final value should exceed the initial)
        assert curve[-1] > curve[0]

    def test_sweet_spot_exists(self):
        """Entropy crosses 0.5 * S_max at some depth for n=4 qubits."""
        n_qubits = 4
        S_max    = n_qubits // 2   # 2 bits
        qrc      = QuantumReservoir(n_qubits=n_qubits, n_layers=12, seed=42)
        x        = np.array([0.3, -0.2, 0.6, -0.4])
        curve    = qrc.entropy_vs_depth(x, max_layers=12)
        # Check at least one depth crosses the half-max threshold
        crossed  = np.any(curve >= 0.5 * S_max)
        assert crossed, (
            f"Entropy never reached 0.5 × S_max = {0.5 * S_max:.2f}. "
            f"Curve: {np.round(curve, 3)}"
        )

    def test_entropy_curve_shape(self):
        qrc   = QuantumReservoir(n_qubits=4, n_layers=6, seed=0)
        x     = np.zeros(4)
        curve = qrc.entropy_vs_depth(x, max_layers=6)
        assert len(curve) == 7   # depth 0 through 6

    def test_plot_entanglement_sweet_spot(self, save_plot):
        import matplotlib.pyplot as plt

        n_qubits  = 4
        max_layers = 10
        S_max     = n_qubits // 2
        rng       = np.random.default_rng(3)
        n_probes  = 6
        X_probe   = rng.uniform(-1, 1, (n_probes, n_qubits))

        qrc    = QuantumReservoir(n_qubits=n_qubits, n_layers=max_layers, seed=42)
        curves = np.stack([qrc.entropy_vs_depth(X_probe[p], max_layers)
                           for p in range(n_probes)])
        s_mean = curves.mean(axis=0)
        s_std  = curves.std(axis=0)
        depths = np.arange(max_layers + 1)

        sweet_candidates = np.where(s_mean >= 0.5 * S_max)[0]
        sweet_depth      = int(sweet_candidates[0]) if len(sweet_candidates) else None

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.fill_between(depths, s_mean - s_std, s_mean + s_std,
                        alpha=0.2, color="#7F77DD")
        ax.plot(depths, s_mean, "o-", color="#7F77DD", lw=2.2,
                label="Mean entropy (±1σ across inputs)")
        ax.axhline(S_max, color="gray", ls=":", lw=1.5,
                   label=f"Max entropy = {S_max} bits")
        ax.axhline(0.5 * S_max, color="#1D9E75", ls="--", lw=1.5,
                   label="Half-max (sweet-spot threshold)")
        if sweet_depth is not None:
            ax.axvline(sweet_depth, color="#E85D3A", ls="--", lw=2,
                       label=f"Sweet spot ≈ depth {sweet_depth}")
        ax.set_xlabel("Reservoir depth (layers)")
        ax.set_ylabel("Bipartite entropy (bits)")
        ax.set_title(
            f"Entanglement sweet spot — {n_qubits} qubits\n"
            f"(averaged over {n_probes} random inputs, seed=42)",
            fontsize=11, fontweight="bold",
        )
        ax.legend(fontsize=9); ax.set_xlim(0, max_layers); ax.set_ylim(0, S_max * 1.2)
        plt.tight_layout()
        save_plot(fig, "05_entanglement_sweet_spot")


# ── Temporal mode ─────────────────────────────────────────────────────────────

class TestTemporalReservoir:
    @pytest.fixture(scope="class")
    def temporal_setup(self):
        """Load-step sequence: BV/TV decreasing under compression."""
        rng  = np.random.default_rng(11)
        T    = 10
        t    = np.linspace(0, 1, T)
        bvtv = 0.40 - 0.12 * t
        tbth = 0.65 - 0.10 * t
        X    = np.column_stack([bvtv, tbth,
                                 bvtv / (tbth + 1e-8),
                                 1.0 - tbth])
        X   += rng.normal(0, 0.01, X.shape)
        X    = np.clip(X, 0, 1)
        # Target: apparent modulus ∝ BV/TV²
        y    = 9700 * bvtv**2
        return X, y

    def test_run_sequence_shape(self, temporal_setup):
        X, _ = temporal_setup
        tqrc = TemporalQuantumReservoir(n_qubits=3, n_layers=2, seed=0,
                                        readout="pauli_z")
        Phi  = tqrc.run_sequence(X)
        assert Phi.shape == (len(X), 3)

    def test_run_sequence_correlators_shape(self, temporal_setup):
        X, _ = temporal_setup
        tqrc = TemporalQuantumReservoir(n_qubits=3, n_layers=2, seed=0,
                                        readout="correlators")
        Phi  = tqrc.run_sequence(X)
        assert Phi.shape == (len(X), 6)

    def test_temporal_fit_predict(self, temporal_setup):
        X, y = temporal_setup
        tqrc = TemporalQuantumReservoir(n_qubits=3, n_layers=2, seed=42,
                                        readout="correlators")
        tqrc.fit_temporal(X, y, alpha=1e-2)
        Phi  = tqrc.run_sequence(X)
        pred = Phi @ tqrc._readout_weights
        # RMSE on training set should be finite
        rmse = float(np.sqrt(np.mean((pred - y)**2)))
        assert np.isfinite(rmse), "Training RMSE is not finite"
        print(f"\n  Temporal QRC train RMSE: {rmse:.0f} MPa")

    def test_plot_temporal_mode(self, temporal_setup, save_plot):
        import matplotlib.pyplot as plt

        X, y = temporal_setup
        T    = len(X)

        tqrc = TemporalQuantumReservoir(n_qubits=4, n_layers=3, seed=42,
                                        readout="correlators")
        split = T - 2
        tqrc.fit_temporal(X[:split], y[:split], alpha=1e-3)

        Phi_all = tqrc.run_sequence(X)
        pred_all = Phi_all @ tqrc._readout_weights

        # Classical baseline (no memory): ridge on raw features
        A_cls  = X[:split].T @ X[:split] + 1e-3 * np.eye(4)
        w_cls  = np.linalg.solve(A_cls, X[:split].T @ y[:split])
        pred_cls = X @ w_cls

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        ax = axes[0]
        steps = np.arange(T)
        ax.plot(steps, y,           "k-o",  lw=1.5,  label="True E_apparent")
        ax.plot(steps[:split], pred_all[:split], "--",
                color="#7F77DD", lw=1.5, label="QRC (train)")
        ax.plot(steps[split:], pred_all[split:], "s-",
                color="#7F77DD", lw=2, markersize=8, label="QRC (test)")
        ax.plot(steps[split:], pred_cls[split:], "^-",
                color="#E85D3A", lw=1.5, markersize=7, label="Classical (no memory)")
        ax.axvline(split - 0.5, color="gray", ls=":", lw=1.5, label="Train/test split")
        ax.set_xlabel("Load step"); ax.set_ylabel("E_apparent (MPa)")
        ax.set_title("Temporal QRC vs classical\n(modulus prediction from load sequence)",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)

        ax2 = axes[1]
        ax2.plot(steps, X[:, 0], "o-", color="#1D9E75", lw=2, label="BV/TV")
        ax2.set_xlabel("Load step"); ax2.set_ylabel("BV/TV", color="#1D9E75")
        ax2.tick_params(axis="y", labelcolor="#1D9E75")

        ax3 = ax2.twinx()
        # Entropy of reservoir state at each step
        n = 4
        state = np.zeros(2**n, dtype=complex); state[0] = 1.0
        entropies = []
        tmp_qrc = TemporalQuantumReservoir(n_qubits=n, n_layers=3, seed=42)
        for x in X:
            x_pad = np.zeros(n)
            x_pad[:min(n, X.shape[1])] = x[:min(n, X.shape[1])]
            state = tmp_qrc._encode_layer(state, x_pad)
            for layer in range(3):
                state = tmp_qrc._reservoir_layer(state, layer)
            entropies.append(entanglement_entropy(state, n))
        ax3.plot(steps, entropies, "s--", color="#7F77DD", lw=1.5, alpha=0.8,
                 label="Reservoir entropy")
        ax3.set_ylabel("Entropy (bits)", color="#7F77DD")
        ax3.tick_params(axis="y", labelcolor="#7F77DD")
        ax2.set_title("Input sequence + reservoir entropy over load steps",
                      fontsize=10, fontweight="bold")

        lines,  labels  = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax2.legend(lines + lines3, labels + labels3, fontsize=8)

        plt.suptitle(
            "Temporal Quantum Reservoir — load-sequence encoding | "
            "prediction vs target  |  BV/TV + entropy per step",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        save_plot(fig, "06_temporal_qrc")
