"""
replot_scalability_params.py
─────────────────────────────────────────────────────────────
Replot the proportional-N scalability figure with the x-axis
changed from "Number of qubits" to "Number of circuit parameters".

For ZZFeatureMap(n_qubits, reps=2), the number of unique gate
parameters is:  n_params = reps × n × (n + 1) / 2

  4q, reps=2 →  20 parameters
  6q, reps=2 →  42 parameters
  8q, reps=2 →  72 parameters
 10q, reps=2 → 110 parameters

Run from:  synthetic_trabeculae/
  python scripts/replot_scalability_params.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Load saved results ─────────────────────────────────────────────────────────
res = np.load('scalability_proportional_results.npy', allow_pickle=True)

# Build a dict keyed by n_qubits
results = {r['n_qubits']: r for r in res}
QUBIT_COUNTS = [4, 6, 8, 10]
N_REPS = 2

# ── Compute ZZFeatureMap parameter counts analytically ─────────────────────────
# Each rep contributes:
#   n single-qubit Rz gates  +  n*(n-1)/2 ZZ-interaction Rz gates
#   = n + n*(n-1)/2 = n*(n+1)/2 parameters per rep
def n_params(n_qubits, reps=N_REPS):
    return reps * (n_qubits * (n_qubits + 1) // 2)

param_counts = [n_params(q) for q in QUBIT_COUNTS]
print("ZZFeatureMap parameter counts (reps=2):")
for q, p in zip(QUBIT_COUNTS, param_counts):
    print(f"  {q}q → {p} parameters")

# ── CI helper ─────────────────────────────────────────────────────────────────
def ci95(values):
    a = np.array(values)
    return 1.96 * np.std(a, ddof=1) / np.sqrt(len(a))

# ── Gather means and CIs ───────────────────────────────────────────────────────
means  = [np.mean(results[q]['all_accs']) for q in QUBIT_COUNTS]
cis    = [ci95(results[q]['all_accs'])    for q in QUBIT_COUNTS]
Ns     = [results[q]['N']                 for q in QUBIT_COUNTS]
depths = [results[q]['depth']             for q in QUBIT_COUNTS]

print("\nAccuracy summary:")
for q, p, m, c, N in zip(QUBIT_COUNTS, param_counts, means, cis, Ns):
    print(f"  {q}q / {p} params / N={N}: {m:.3f} ± {c:.3f} (95% CI)")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

color = 'steelblue'

# Error bars (95% CI)
ax.errorbar(param_counts, means, yerr=cis,
            marker='o', color=color, linewidth=2, markersize=9,
            capsize=5, capthick=1.5, zorder=3, label='Q-ZZ Kernel (UMAP)')

# Scatter individual dataset points
for q, p in zip(QUBIT_COUNTS, param_counts):
    pts = results[q]['all_accs']
    ax.scatter([p] * len(pts), pts,
               alpha=0.2, s=18, color=color, zorder=2)

# Chance line
ax.axhline(0.5, linestyle=':', color='black', linewidth=1.2, label='Chance (0.50)')

# Shade undersampled region (8q and 10q — N < Hilbert dim for 10q)
ax.axvspan(param_counts[2] - 3, param_counts[3] + 3,
           alpha=0.06, color='gray', label='N ≤ Hilbert dim (undersampled)')
ax.text(param_counts[3], 0.92,
        'N < Hilbert\ndim', ha='center', fontsize=8, color='gray')

# x-axis ticks: one per qubit count, labelled with param count and qubit info
ax.set_xticks(param_counts)
ax.set_xticklabels(
    [f'{p}\n({q}q, N={N})' for p, q, N in zip(param_counts, QUBIT_COUNTS, Ns)],
    fontsize=9)

ax.set_xlabel('Number of circuit parameters  (ZZFeatureMap, reps=2)', fontsize=11)
ax.set_ylabel('BV/TV classification accuracy  (10-dataset mean, 95% CI)', fontsize=11)
ax.set_title(
    'QKSVM Scalability: Accuracy vs Circuit Complexity\n'
    r'(N $\propto$ Hilbert dimension; accuracy remains stable across circuit sizes)',
    fontsize=10)

ax.set_ylim(0.38, 1.00)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()

for ext in ('png', 'pdf'):
    fname = f'scalability_proportional_params.{ext}'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"Saved: {fname}")

plt.close()
print("\nDone.")
