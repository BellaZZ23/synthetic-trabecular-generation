"""
replot_hyperparameter.py
────────────────────────
Loads saved hyperparameter_results.npy and regenerates the figure.
Run after hyperparameter_optimization.py has completed.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Config (must match hyperparameter_optimization.py) ───────────────────────
C_VALUES         = [0.01, 0.1, 1.0, 10.0, 100.0]
REPS_VALUES      = [1, 2, 3]
NEIGHBORS_VALUES = [5, 15, 30, 50]
N_DATASETS       = 10
N_QUBITS         = 8
N_MAX            = 500

def ci95(vals):
    return 1.96 * float(np.std(vals, ddof=1)) / np.sqrt(len(vals))

# ── Load ──────────────────────────────────────────────────────────────────────
data     = np.load('hyperparameter_results.npy', allow_pickle=True).item()
exp1     = data['exp1']
exp2     = data['exp2']
exp3     = data['exp3']

# ── Best configs ──────────────────────────────────────────────────────────────
best_q_key  = max(exp1.keys(), key=lambda k: np.mean(exp1[k]['q']))
best_C, use_center = best_q_key
best_reps   = max(exp2.keys(), key=lambda r: np.mean(exp2[r]['accs']))
best_nn     = max(exp3.keys(), key=lambda n: np.mean(exp3[n]['q']))

best_q_acc  = np.mean(exp3[best_nn]['q'])
best_cl_acc = np.mean(exp3[best_nn]['cl'])
baseline    = np.mean(exp1[(1.0, False)]['q'])

print(f"Best quantum:   C={best_C}, center={use_center}, "
      f"reps={best_reps}, n_neighbors={best_nn}")
print(f"  Accuracy: {best_q_acc:.3f} ± {ci95(exp3[best_nn]['q']):.3f}")
print(f"  Classical: {best_cl_acc:.3f} ± {ci95(exp3[best_nn]['cl']):.3f}")
print(f"  Baseline:  {baseline:.3f}")

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1 — C tuning
ax = axes[0]
for center, fmt, lbl, col in [
        (False, '-o', 'Quantum (raw kernel)',      'steelblue'),
        (True,  '-s', 'Quantum (centered kernel)', 'cornflowerblue')]:
    means = [np.mean(exp1[(C, center)]['q']) for C in C_VALUES]
    cis   = [ci95(exp1[(C, center)]['q'])    for C in C_VALUES]
    ax.errorbar(range(len(C_VALUES)), means, yerr=cis, fmt=fmt,
                color=col, capsize=4, linewidth=1.6, markersize=7, label=lbl)
cl_means = [np.mean(exp1[(C, False)]['cl']) for C in C_VALUES]
cl_cis   = [ci95(exp1[(C, False)]['cl'])    for C in C_VALUES]
ax.errorbar(range(len(C_VALUES)), cl_means, yerr=cl_cis, fmt='--^',
            color='seagreen', capsize=4, linewidth=1.4, markersize=7,
            label='Classical RBF')
ax.axhline(0.5, linestyle=':', color='black', linewidth=1, label='Chance')
ax.set_xticks(range(len(C_VALUES)))
ax.set_xticklabels([str(c) for c in C_VALUES])
ax.set_xlabel('C (SVM regularisation)', fontsize=11)
ax.set_ylabel('Accuracy (5-fold CV, 95% CI)', fontsize=11)
ax.set_title('Effect of C and Kernel Centering\n'
             f'reps=2, n_neighbors=15, N={N_MAX}', fontsize=10)
ax.legend(fontsize=8)
ax.set_ylim(0.45, 0.82)
ax.grid(True, alpha=0.3)

# Panel 2 — reps comparison
ax = axes[1]
means  = [np.mean(exp2[r]['accs']) for r in REPS_VALUES]
cis    = [ci95(exp2[r]['accs'])    for r in REPS_VALUES]
depths = [exp2[r]['depth']          for r in REPS_VALUES]
ax.errorbar(REPS_VALUES, means, yerr=cis, fmt='-o', color='steelblue',
            capsize=4, linewidth=1.6, markersize=8)
for r, d, m, ci in zip(REPS_VALUES, depths, means, cis):
    ax.annotate(f'depth={d}', (r, m + ci + 0.004),
                ha='center', fontsize=8, color='gray')
ax.axhline(0.5, linestyle=':', color='black', linewidth=1, label='Chance')

# Add best classical reference
best_cl_exp2 = np.mean(exp1[(best_C, use_center)]['cl'])
ax.axhline(best_cl_exp2, linestyle='--', color='seagreen',
           linewidth=1.4, label=f'Classical RBF ({best_cl_exp2:.3f})')
ax.set_xlabel('ZZFeatureMap reps', fontsize=11)
ax.set_ylabel('Accuracy (5-fold CV, 95% CI)', fontsize=11)
ax.set_title(f'Effect of Circuit Reps\n'
             f'C={best_C}, center={use_center}, n_neighbors=15', fontsize=10)
ax.set_xticks(REPS_VALUES)
ax.set_ylim(0.45, 0.82)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3 — UMAP n_neighbors
ax = axes[2]
q_means  = [np.mean(exp3[nn]['q'])  for nn in NEIGHBORS_VALUES]
q_cis    = [ci95(exp3[nn]['q'])     for nn in NEIGHBORS_VALUES]
cl_means = [np.mean(exp3[nn]['cl']) for nn in NEIGHBORS_VALUES]
cl_cis   = [ci95(exp3[nn]['cl'])    for nn in NEIGHBORS_VALUES]
ax.errorbar(NEIGHBORS_VALUES, q_means,  yerr=q_cis,  fmt='-o',
            color='steelblue', capsize=4, linewidth=1.6, markersize=7,
            label='Quantum ZZ kernel')
ax.errorbar(NEIGHBORS_VALUES, cl_means, yerr=cl_cis, fmt='--^',
            color='seagreen', capsize=4, linewidth=1.4, markersize=7,
            label='Classical RBF')
ax.axhline(0.5, linestyle=':', color='black', linewidth=1, label='Chance')

# Highlight best quantum point
ax.scatter([best_nn], [np.mean(exp3[best_nn]['q'])],
           s=120, zorder=5, color='gold', edgecolors='steelblue',
           linewidth=2, label=f'Best: nn={best_nn} ({np.mean(exp3[best_nn]["q"]):.3f})')
ax.set_xlabel('UMAP n_neighbors', fontsize=11)
ax.set_ylabel('Accuracy (5-fold CV, 95% CI)', fontsize=11)
ax.set_title(f'Effect of UMAP Neighbourhood Scale\n'
             f'C={best_C}, center={use_center}, reps={best_reps}', fontsize=10)
ax.legend(fontsize=8)
ax.set_ylim(0.45, 0.82)
ax.grid(True, alpha=0.3)

plt.suptitle(
    f'Hyperparameter Optimisation — {N_QUBITS}q ZZFeatureMap, N={N_MAX},  '
    f'{N_DATASETS} independent datasets  |  '
    f'Best: Q={best_q_acc:.3f}  Classical={best_cl_acc:.3f}',
    fontsize=10, y=1.01)
plt.tight_layout()
for ext in ('pdf', 'png'):
    plt.savefig(f'hyperparameter_optimization.{ext}', dpi=150, bbox_inches='tight')
print("Saved: hyperparameter_optimization.pdf / .png")
plt.show()
