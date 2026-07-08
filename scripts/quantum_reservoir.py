"""
quantum_reservoir.py — Quantum Reservoir Computing for trabecular bone.

Architecture
------------
  Input x (d-dim) ──► Angle encoding ──► Fixed random reservoir ──► Linear readout

The reservoir is FIXED at init (random angles + CNOT ring). Only the
readout layer is trained (ridge regression). Computational power comes
from the reservoir's entanglement and interference, not from training it.

Simulation: pure numpy state-vector (exact, 2^n amplitudes).
Practical for n_qubits ≤ 8 in a browser; recommended ≤ 6 for real-time.

Novel target result
-------------------
Entanglement sweet spot: the reservoir depth where QRC performance peaks.
  Too shallow  → insufficient mixing, low expressivity
  Sweet spot   → half-saturation of bipartite entropy (~log2(2^(n/2)))
  Too deep     → approaches Haar-random, information washed out

Temporal mode: encode load-step sequences x₁..xₜ into the reservoir
sequentially; its state carries memory across steps → novel for bone DVC.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# ─── Gate library ───────────────────────────────────────────────────────────

def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)


def _apply_single(state: np.ndarray, gate: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Apply a 2x2 gate to `qubit` of an n-qubit state vector (shape 2^n)."""
    psi = state.reshape([2] * n)
    # einsum: contract gate with the qubit index
    psi = np.tensordot(gate, psi, axes=([1], [qubit]))
    # tensordot moves the new axis to front — move it back to `qubit`
    axes = list(range(1, qubit + 1)) + [0] + list(range(qubit + 1, n))
    return psi.transpose(axes).reshape(-1)


def _apply_cnot(state: np.ndarray, ctrl: int, tgt: int, n: int) -> np.ndarray:
    """CNOT with `ctrl` controlling `tgt`."""
    psi = state.reshape([2] * n)
    # Flip target bit when control = 1
    idx_ctrl1 = [slice(None)] * n
    idx_ctrl1[ctrl] = 1
    idx_ctrl1 = tuple(idx_ctrl1)
    sub = psi[idx_ctrl1]
    # Swap target axis within the control=1 sub-tensor
    sub = np.flip(sub, axis=tgt if tgt < ctrl else tgt - 1)
    psi[idx_ctrl1] = sub
    return psi.reshape(-1)


# ─── Entanglement entropy ────────────────────────────────────────────────────

def entanglement_entropy(state: np.ndarray, n: int, partition: Optional[int] = None) -> float:
    """
    Bipartite von Neumann entropy S = -Tr(ρ_A log ρ_A).

    Splits the n-qubit system at `partition` (default: n//2).
    Returns entropy in bits (log base 2). Maximum = partition qubits.
    """
    if partition is None:
        partition = n // 2
    nA = partition
    nB = n - nA
    psi = state.reshape(2 ** nA, 2 ** nB)
    # SVD of the reshaped state gives Schmidt coefficients
    _, sv, _ = np.linalg.svd(psi, full_matrices=False)
    probs = sv ** 2
    probs = probs[probs > 1e-15]
    return float(-np.sum(probs * np.log2(probs)))


# ─── Quantum Reservoir ───────────────────────────────────────────────────────

class QuantumReservoir:
    """
    Fixed random quantum reservoir with trained linear readout.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (2–8 practical; ≤6 recommended for speed).
    n_layers : int
        Reservoir depth: each layer = Ry(data) + fixed Rz + CNOT ring.
    seed : int
        Fixes the random reservoir angles (reproducible).
    readout : {"pauli_z", "correlators"}
        Feature vector from the reservoir state.
        "pauli_z"     — n_qubits expectation values ⟨Zᵢ⟩
        "correlators" — adds n_qubits ⟨ZᵢZᵢ₊₁⟩ two-qubit correlators
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 3, seed: int = 42,
                 readout: str = "correlators"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.readout  = readout
        rng = np.random.default_rng(seed)
        # Fixed reservoir angles: (layer, qubit, [theta_ry, theta_rz])
        self._res_angles = rng.uniform(0, 2 * np.pi, (n_layers, n_qubits, 2))
        self._readout_weights: Optional[np.ndarray] = None
        self._readout_bias: Optional[float] = None

    # ── Core circuit ─────────────────────────────────────────────────────────

    def _encode_layer(self, state: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Angle-encode input x: Ry(π·xᵢ) on each qubit (x normalised to [-1,1])."""
        n = self.n_qubits
        for q in range(min(n, len(x))):
            state = _apply_single(state, _ry(np.pi * float(x[q])), q, n)
        return state

    def _reservoir_layer(self, state: np.ndarray, layer: int) -> np.ndarray:
        """One reservoir layer: fixed Ry + Rz per qubit, then CNOT ring."""
        n = self.n_qubits
        angles = self._res_angles[layer]
        for q in range(n):
            state = _apply_single(state, _ry(angles[q, 0]), q, n)
            state = _apply_single(state, _rz(angles[q, 1]), q, n)
        # CNOT ring
        for q in range(n):
            state = _apply_cnot(state, q, (q + 1) % n, n)
        return state

    def _run_circuit(self, x: np.ndarray) -> np.ndarray:
        """Run the full reservoir circuit for input x. Returns final state."""
        n = self.n_qubits
        d = len(x)
        state = np.zeros(2 ** n, dtype=complex)
        state[0] = 1.0  # |0...0⟩

        for layer in range(self.n_layers):
            # Interleave: encode → reservoir layer
            x_pad = np.zeros(n)
            x_pad[:min(n, d)] = x[:min(n, d)]
            state = self._encode_layer(state, x_pad)
            state = self._reservoir_layer(state, layer)
        return state

    # ── Feature extraction ────────────────────────────────────────────────────

    def _pauli_z_expectation(self, state: np.ndarray) -> np.ndarray:
        """⟨Zᵢ⟩ for each qubit. Shape: (n_qubits,)."""
        n = self.n_qubits
        psi = state.reshape([2] * n)
        evs = np.zeros(n)
        for q in range(n):
            # Sum |amplitude|² where qubit q = 0 (eigenvalue +1) vs 1 (-1)
            axes_sum = tuple(i for i in range(n) if i != q)
            p0 = np.abs(psi.take([0], axis=q)).reshape(-1) ** 2
            p1 = np.abs(psi.take([1], axis=q)).reshape(-1) ** 2
            evs[q] = float(p0.sum() - p1.sum())
        return evs

    def _zz_correlators(self, state: np.ndarray) -> np.ndarray:
        """⟨ZᵢZᵢ₊₁⟩ nearest-neighbour correlators. Shape: (n_qubits,)."""
        n = self.n_qubits
        psi = state.reshape([2] * n)
        corr = np.zeros(n)
        for q in range(n):
            q2 = (q + 1) % n
            ev = 0.0
            for b0 in [0, 1]:
                for b1 in [0, 1]:
                    idx = [slice(None)] * n
                    idx[q] = b0
                    idx[q2] = b1
                    amp2 = np.abs(psi[tuple(idx)]) ** 2
                    sign = (1 - 2 * b0) * (1 - 2 * b1)  # ±1
                    ev += sign * float(amp2.sum())
            corr[q] = ev
        return corr

    def feature_vector(self, x: np.ndarray) -> np.ndarray:
        """Map input x to reservoir feature vector."""
        state = self._run_circuit(np.asarray(x, dtype=float))
        fv = self._pauli_z_expectation(state)
        if self.readout == "correlators":
            fv = np.concatenate([fv, self._zz_correlators(state)])
        return fv

    def feature_matrix(self, X: np.ndarray) -> np.ndarray:
        """Map (N, d) input matrix to (N, n_features) reservoir features."""
        return np.stack([self.feature_vector(x) for x in X])

    # ── Linear readout ────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray, alpha: float = 1e-3) -> "QuantumReservoir":
        """
        Train the linear readout via ridge regression.

        Parameters
        ----------
        X : (N, d) input features (will be normalised to [-1, 1] internally)
        y : (N,) targets (regression or classification labels)
        alpha : ridge regularisation
        """
        self._x_mean = X.mean(axis=0)
        self._x_std  = X.std(axis=0) + 1e-8
        Xn = (X - self._x_mean) / self._x_std

        Phi = self.feature_matrix(Xn)   # (N, n_features)
        # Ridge: w = (ΦᵀΦ + αI)⁻¹ Φᵀy
        A = Phi.T @ Phi + alpha * np.eye(Phi.shape[1])
        b = Phi.T @ y
        self._readout_weights = np.linalg.solve(A, b)
        self._readout_bias    = float(y.mean() - Phi.mean(axis=0) @ self._readout_weights)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for (N, d) inputs."""
        if self._readout_weights is None:
            raise RuntimeError("Call fit() first.")
        Xn = (X - self._x_mean) / self._x_std
        Phi = self.feature_matrix(Xn)
        return Phi @ self._readout_weights + self._readout_bias

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """Binary classification: threshold at 0.5."""
        return (self.predict(X) >= 0.5).astype(int)

    # ── Entanglement diagnostics ──────────────────────────────────────────────

    def reservoir_entropy(self, x: np.ndarray) -> float:
        """Von Neumann entropy of the reservoir state for input x."""
        state = self._run_circuit(np.asarray(x, dtype=float))
        return entanglement_entropy(state, self.n_qubits)

    def entropy_vs_depth(self, x: np.ndarray, max_layers: Optional[int] = None) -> np.ndarray:
        """
        Entanglement entropy after 0, 1, ..., max_layers reservoir layers.
        Returns array of length max_layers + 1.
        Used to locate the entanglement sweet spot.
        """
        if max_layers is None:
            max_layers = self.n_layers
        n = self.n_qubits
        x_pad = np.zeros(n)
        d = len(x)
        x_pad[:min(n, d)] = x[:min(n, d)]

        entropies = []
        state = np.zeros(2 ** n, dtype=complex)
        state[0] = 1.0
        entropies.append(entanglement_entropy(state, n))

        for layer in range(min(max_layers, len(self._res_angles))):
            state = self._encode_layer(state, x_pad)
            state = self._reservoir_layer(state, layer)
            entropies.append(entanglement_entropy(state, n))

        return np.array(entropies)


# ─── Temporal reservoir (load-sequence mode) ─────────────────────────────────

class TemporalQuantumReservoir(QuantumReservoir):
    """
    QRC in temporal mode: processes a sequence of inputs x₁..xₜ by running
    the reservoir circuit continuously (no reset between steps).

    This gives the reservoir *memory* — its state at step t depends on all
    prior inputs. In the bone DVC context: x₁..xₜ are morphometric snapshots
    at successive load steps; the readout predicts the apparent modulus at t.

    The memory fades with depth (more entanglement layers → shorter memory).
    The sweet spot balances memory retention against feature richness.
    """

    def run_sequence(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X_seq : (T, d) sequence of T input vectors

        Returns
        -------
        features : (T, n_features) feature vectors at each time step
        """
        n = self.n_qubits
        state = np.zeros(2 ** n, dtype=complex)
        state[0] = 1.0  # |0...0⟩

        features = []
        for x in X_seq:
            x_pad = np.zeros(n)
            x_pad[:min(n, len(x))] = x[:min(n, len(x))]
            state = self._encode_layer(state, x_pad)
            for layer in range(self.n_layers):
                state = self._reservoir_layer(state, layer)
            fv = self._pauli_z_expectation(state)
            if self.readout == "correlators":
                fv = np.concatenate([fv, self._zz_correlators(state)])
            features.append(fv)
        return np.stack(features)

    def fit_temporal(self, X_seq: np.ndarray, y_seq: np.ndarray,
                     alpha: float = 1e-3) -> "TemporalQuantumReservoir":
        """Train readout on a sequence."""
        Phi = self.run_sequence(X_seq)
        self._x_mean = np.zeros(X_seq.shape[1])
        self._x_std  = np.ones(X_seq.shape[1])
        A = Phi.T @ Phi + alpha * np.eye(Phi.shape[1])
        self._readout_weights = np.linalg.solve(A, Phi.T @ y_seq)
        self._readout_bias    = 0.0
        return self


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("QRC self-test: 4 qubits, 3 layers")
    rng = np.random.default_rng(7)

    # Synthetic bone morphometrics: [BV/TV, Tb.Th, Tb.N, Tb.Sp]
    X = rng.uniform(0, 1, (40, 4))
    y = (X[:, 0] > 0.5).astype(float)  # high vs low BV/TV

    qrc = QuantumReservoir(n_qubits=4, n_layers=3, seed=42)
    split = 30
    qrc.fit(X[:split], y[:split])
    preds = qrc.predict_class(X[split:])
    acc = float((preds == y[split:]).mean())
    print(f"Classification accuracy: {acc:.2f}")

    x0 = X[0]
    print(f"\nEntropy vs depth for sample x₀:")
    entropies = qrc.entropy_vs_depth(x0, max_layers=6)
    for d, e in enumerate(entropies):
        print(f"  depth {d}: S = {e:.4f} bits")
    print(f"Max bipartite entropy (n/2 qubits): {qrc.n_qubits // 2:.1f} bits")
    print("Done.")