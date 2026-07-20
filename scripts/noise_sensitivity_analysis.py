"""
noise_sensitivity_analysis.py
─────────────────────────────
Evaluates how depolarising gate noise degrades the UMAP quantum kernel.

For each noise level the script:
  1. Builds a Qiskit Aer noise model (depolarising on 1q and 2q gates)
  2. Computes a 150×150 kernel matrix via noisy circuit simulation
  3. Measures effective rank and SVM classification accuracy (BV/TV)
  4. Saves a PDF/PNG figure suitable for the paper

Prerequisites
─────────────
pip install qiskit qiskit-aer qiskit-machine-learning scikit-learn umap-learn

Before running, save your processed data from the original pipeline:

    np.save('umap_features_8d.npy', X_umap_scaled_to_pi)  # shape (500, 8)
    np.save('labels_bvtv.npy', y_bvtv_binary)              # shape (500,)

Both arrays should match the original paper's preprocessing exactly.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit import transpile
from qiskit.circuit import QuantumCircuit

# ── Configuration ─────────────────────────────────────────────────────────────
N_SUBSAMPLE  = 150      # kernel matrix size; reduce to 100 if too slow
N_QUBITS     = 8
N_REPS       = 2
SHOTS        = 1024     # shots per circuit for noisy simulation
CV_FOLDS     = 5
SEED         = 42

# Gate error rates for 1-qubit gates.
# 2-qubit error is set to 10× (typical NISQ hardware ratio).
NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]

# ── Helpers ───────────────────────────────────────────────────────────────────

def effective_rank(K):
    """Effective rank = (Σλ)² / Σλ² (Brüningk et al. formulation)."""
    eigvals = np.maximum(np.linalg.eigvalsh(K), 0)
    s = eigvals.sum()
    return 1.0 if s == 0 else s ** 2 / (eigvals ** 2).sum()


def kernel_entry_statevector(fm, xi, xj, params):
    """Exact kernel entry K(xi,xj) via statevector inner product."""
    sv_i = Statevector.from_instruction(fm.assign_parameters(dict(zip(params, xi))))
    sv_j = Statevector.from_instruction(fm.assign_parameters(dict(zip(params, xj))))
    return float(abs(sv_i.inner(sv_j)) ** 2)


def build_fidelity_circuit(fm, xi, xj, params):
    """Build circuit for K(xi,xj) = |⟨0|U†(xj)U(xi)|0⟩|²."""
    qc = QuantumCircuit(fm.num_qubits)
    qc.compose(fm.assign_parameters(dict(zip(params, xi))), inplace=True)
    qc.compose(fm.assign_parameters(dict(zip(params, xj))).inverse(), inplace=True)
    return qc


def compute_kernel_matrix(X, feature_map, noise_model=None):
    """
    Compute symmetric kernel matrix.
    noise_model=None  → exact statevector (fast).
    noise_model=<obj> → Aer noisy simulation with SHOTS shots.
    """
    n = len(X)
    K = np.eye(n)
    params = list(feature_map.parameters)
    all_zeros = '0' * feature_map.num_qubits

    if noise_model is None:
        for i in range(n):
            for j in range(i + 1, n):
                v = kernel_entry_statevector(feature_map, X[i], X[j], params)
                K[i, j] = K[j, i] = v
    else:
        sim = AerSimulator(noise_model=noise_model)
        for i in range(n):
            for j in range(i + 1, n):
                qc = build_fidelity_circuit(feature_map, X[i], X[j], params)
                qc.measure_all()
                counts = sim.run(transpile(qc, sim), shots=SHOTS).result().get_counts()
                K[i, j] = K[j, i] = counts.get(all_zeros, 0) / SHOTS

    return K


def svm_cv_accuracy(K, y):
    """SVM accuracy with precomputed kernel, stratified k-fold."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    accs = []
    for tr, te in skf.split(K, y):
        clf = SVC(kernel='precomputed', C=1.0)
        clf.fit(K[np.ix_(tr, tr)], y[tr])
        accs.append(clf.score(K[np.ix_(te, tr)], y[te]))
    return float(np.mean(accs)), float(np.std(accs))


def make_noise_model(p1q):
    """Depolarising noise model. 2-qubit error = 10× 1-qubit error (NISQ ratio)."""
    p2q = min(p1q * 10, 1.0)
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(
        depolarizing_error(p1q, 1), ['u1', 'u2', 'u3', 'rz', 'ry', 'rx', 'h', 'x'])
    nm.add_all_qubit_quantum_error(
        depolarizing_error(p2q, 2), ['cx', 'cz', 'ecr'])
    return nm


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
try:
    X_umap = np.load('umap_features_8d.npy')
    y      = np.load('labels_bvtv.npy')
except FileNotFoundError as e:
    raise FileNotFoundError(
        "Missing data files. Save your UMAP-reduced features (scaled to [0,π]) "
        "and binary BV/TV labels from the original pipeline:\n"
        "  np.save('umap_features_8d.npy', X_umap_scaled_to_pi)\n"
        "  np.save('labels_bvtv.npy', y_bvtv_binary)"
    ) from e

print(f"  Loaded: X={X_umap.shape}, y={y.shape}")
rng  = np.random.default_rng(SEED)
idx  = rng.choice(len(X_umap), N_SUBSAMPLE, replace=False)
X_s, y_s = X_umap[idx], y[idx]

# ── Feature map ───────────────────────────────────────────────────────────────
fm    = ZZFeatureMap(feature_dimension=N_QUBITS, reps=N_REPS)
depth = fm.decompose().depth()
print(f"ZZFeatureMap: {N_QUBITS} qubits, {N_REPS} reps, decomposed depth = {depth}")

# ── Noise sweep ───────────────────────────────────────────────────────────────
print(f"\nRunning noise sweep over {len(NOISE_LEVELS)} levels "
      f"({N_SUBSAMPLE}×{N_SUBSAMPLE} kernel each) …\n")

results = []
for p in NOISE_LEVELS:
    tag = f"p={p:.3f}"
    print(f"  {tag}", end='  ', flush=True)
    t0 = time.time()

    nm = None if p == 0.0 else make_noise_model(p)
    K  = compute_kernel_matrix(X_s, fm, noise_model=nm)

    er          = effective_rank(K)
    off_diag    = (K.sum() - np.trace(K)) / (N_SUBSAMPLE * (N_SUBSAMPLE - 1))
    acc, std    = svm_cv_accuracy(K, y_s)
    elapsed     = time.time() - t0

    results.append(dict(noise=p, eff_rank=er, off_diag=off_diag,
                         acc=acc, std=std, time_min=elapsed / 60))
    print(f"eff_rank={er:.1f}/{N_SUBSAMPLE}  acc={acc:.3f}±{std:.3f}  "
          f"({elapsed/60:.1f} min)")

np.save('noise_sensitivity_results.npy',
        np.array([(r['noise'], r['eff_rank'], r['off_diag'], r['acc'], r['std'])
                  for r in results],
                 dtype=[('noise','f8'),('eff_rank','f8'),('off_diag','f8'),
                        ('acc','f8'),('std','f8')]))
print("\nResults saved to noise_sensitivity_results.npy")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'Noise (1q)':>12}  {'Noise (2q)':>12}  {'Eff. Rank':>10}  "
      f"{'Off-diag':>10}  {'Accuracy':>10}")
for r in results:
    p2 = min(r['noise'] * 10, 1.0)
    print(f"{r['noise']:>12.3f}  {p2:>12.3f}  {r['eff_rank']:>10.1f}  "
          f"{r['off_diag']:>10.4f}  {r['acc']:>10.3f}")

# ── Figure ────────────────────────────────────────────────────────────────────
noise_vals = [r['noise']    for r in results]
accs       = [r['acc']      for r in results]
stds       = [r['std']      for r in results]
er_vals    = [r['eff_rank'] for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

ax1.errorbar(noise_vals, accs, yerr=stds, marker='o',
             color='steelblue', capsize=4, linewidth=1.5)
ax1.axhline(accs[0], linestyle='--', color='gray', linewidth=1, label='Noiseless')
ax1.set_xlabel('Depolarising noise rate (1-qubit gate)', fontsize=11)
ax1.set_ylabel('BV/TV classification accuracy', fontsize=11)
ax1.set_title('Noise Sensitivity — UMAP Quantum Kernel', fontsize=11)
ax1.set_xscale('symlog', linthresh=0.0005)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(noise_vals, er_vals, marker='s', color='coral', linewidth=1.5)
ax2.axhline(er_vals[0], linestyle='--', color='gray', linewidth=1, label='Noiseless')
ax2.axhline(N_SUBSAMPLE, linestyle=':', color='black', linewidth=1,
            label=f'Identity limit ({N_SUBSAMPLE})')
ax2.set_xlabel('Depolarising noise rate (1-qubit gate)', fontsize=11)
ax2.set_ylabel(f'Effective rank  (max = {N_SUBSAMPLE})', fontsize=11)
ax2.set_title('Kernel Concentration vs Noise', fontsize=11)
ax2.set_xscale('symlog', linthresh=0.0005)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
for ext in ('pdf', 'png'):
    plt.savefig(f'noise_sensitivity.{ext}', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: noise_sensitivity.pdf / noise_sensitivity.png")
