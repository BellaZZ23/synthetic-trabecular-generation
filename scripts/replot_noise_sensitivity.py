"""
replot_noise_sensitivity.py
────────────────────────────
Loads saved noise_sensitivity_results.npy and regenerates a clean
publication-quality figure with:

  - Fixed x-axis: p=0 shown as a separate noiseless reference point,
    p>0 on a true log scale (no symlog ambiguity)
  - Classical RBF-SVM baseline computed from saved umap_features_8d.npy
  - 95% confidence intervals from the CV std (std/sqrt(n_folds))
  - Cleaner two-panel layout with consistent styling

Run from:  synthetic_trabeculae/
  python scripts/replot_noise_sensitivity.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

# ── Load saved results ────────────────────────────────────────────────────────
try:
    raw = np.load('noise_sensitivity_results.npy', allow_pickle=True)
except FileNotFoundError:
    raise FileNotFoundError(
        "noise_sensitivity_results.npy not found.\n"
        "Run noise_sensitivity_analysis.py first."
    )

# Handle both structured array and list-of-dicts formats
if raw.dtype.names:
    # structured array
    noise_vals = raw['noise'].tolist()
    er_vals    = raw['eff_rank'].tolist()
    acc_vals   = raw['acc'].tolist()
    std_vals   = raw['std'].tolist()
else:
    records    = raw.tolist()
    noise_vals = [r['noise']    for r in records]
    er_vals    = [r['eff_rank'] for r in records]
    acc_vals   = [r['acc']      for r in records]
    std_vals   = [r['std']      for r in records]

N_SUBSAMPLE = 150   # from noise_sensitivity_analysis.py config
CV_FOLDS    = 5

# 95% CI = 1.96 × std / sqrt(n_folds)
ci_vals = [1.96 * s / np.sqrt(CV_FOLDS) for s in std_vals]

print(f"Loaded {len(noise_vals)} noise levels: {noise_vals}")
print(f"Accuracies: {[f'{a:.3f}' for a in acc_vals]}")

# ── Classical RBF baseline ────────────────────────────────────────────────────
classical_acc = None
try:
    X_umap = np.load('umap_features_8d.npy')
    y      = np.load('labels_bvtv.npy')
    import numpy as np_
    rng = np_.random.default_rng(42)
    idx = rng.choice(len(X_umap), N_SUBSAMPLE, replace=False)
    X_s, y_s = X_umap[idx], y[idx]

    skf  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(X_s, y_s):
        clf = SVC(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X_s[tr], y_s[tr])
        accs.append(clf.score(X_s[te], y_s[te]))
    classical_acc = float(np.mean(accs))
    classical_ci  = 1.96 * float(np.std(accs, ddof=1)) / np.sqrt(CV_FOLDS)
    print(f"\nClassical RBF-SVM (UMAP features, {CV_FOLDS}-fold CV): "
          f"{classical_acc:.3f} ± {classical_ci:.3f} (95% CI)")
except FileNotFoundError:
    print("\numap_features_8d.npy not found — skipping classical baseline.")
    classical_ci = 0.0

# ── Separate noiseless from noisy ─────────────────────────────────────────────
noiseless_idx = [i for i, p in enumerate(noise_vals) if p == 0.0]
noisy_idx     = [i for i, p in enumerate(noise_vals) if p  > 0.0]

noiseless_acc = acc_vals[noiseless_idx[0]] if noiseless_idx else None
noiseless_ci  = ci_vals[noiseless_idx[0]]  if noiseless_idx else 0.0
noiseless_er  = er_vals[noiseless_idx[0]]  if noiseless_idx else None

noisy_p   = [noise_vals[i] for i in noisy_idx]
noisy_acc = [acc_vals[i]   for i in noisy_idx]
noisy_ci  = [ci_vals[i]    for i in noisy_idx]
noisy_er  = [er_vals[i]    for i in noisy_idx]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# ── Panel 1: Accuracy vs noise ─────────────────────────────────────────────
ax = ax1

# Noisy points on log x-axis
ax.errorbar(noisy_p, noisy_acc, yerr=noisy_ci,
            marker='o', color='steelblue', capsize=4, linewidth=1.8,
            markersize=7, label='Quantum kernel (noisy)', zorder=3)

# Noiseless reference — horizontal dashed line
if noiseless_acc is not None:
    ax.axhline(noiseless_acc, linestyle='--', color='steelblue',
               linewidth=1.4, alpha=0.8, label=f'Noiseless ({noiseless_acc:.3f})')
    ax.fill_between([min(noisy_p) * 0.5, max(noisy_p) * 2],
                    noiseless_acc - noiseless_ci,
                    noiseless_acc + noiseless_ci,
                    alpha=0.10, color='steelblue')

# Classical RBF baseline
if classical_acc is not None:
    ax.axhline(classical_acc, linestyle='-.', color='seagreen',
               linewidth=1.4, alpha=0.9,
               label=f'Classical RBF ({classical_acc:.3f})')
    ax.fill_between([min(noisy_p) * 0.5, max(noisy_p) * 2],
                    classical_acc - classical_ci,
                    classical_acc + classical_ci,
                    alpha=0.10, color='seagreen')

ax.axhline(0.5, linestyle=':', color='black', linewidth=1, label='Chance (0.50)')

ax.set_xscale('log')
ax.set_xlabel('Depolarising noise rate  (1-qubit gates)', fontsize=11)
ax.set_ylabel('BV/TV classification accuracy  (95% CI)', fontsize=11)
ax.set_title('Noise Sensitivity — UMAP Quantum Kernel\n'
             '(2-qubit error = 10× 1-qubit error)', fontsize=10)
ax.legend(fontsize=9, loc='lower left')
ax.set_ylim(0.40, 0.80)
ax.grid(True, alpha=0.3, which='both')
ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())

# Annotate current NISQ range
ax.axvspan(0.001, 0.01, alpha=0.07, color='crimson')
ax.text(0.003, 0.775, 'Typical\nNISQ\nrange', ha='center',
        fontsize=7.5, color='crimson', va='top')

# ── Panel 2: Effective rank vs noise ──────────────────────────────────────────
ax = ax2
ax.semilogx(noisy_p, noisy_er, marker='s', color='coral',
            linewidth=1.8, markersize=7, label='Quantum kernel (noisy)')
if noiseless_er is not None:
    ax.axhline(noiseless_er, linestyle='--', color='coral',
               linewidth=1.4, alpha=0.8,
               label=f'Noiseless ({noiseless_er:.1f}/{N_SUBSAMPLE})')
ax.axhline(N_SUBSAMPLE, linestyle=':', color='black', linewidth=1,
           label=f'Identity limit ({N_SUBSAMPLE})')

ax.set_xscale('log')
ax.set_xlabel('Depolarising noise rate  (1-qubit gates)', fontsize=11)
ax.set_ylabel(f'Effective rank  (max = {N_SUBSAMPLE})', fontsize=11)
ax.set_title('Kernel Concentration vs Noise\n'
             '(higher rank = closer to identity = less useful)', fontsize=10)
ax.legend(fontsize=9)
ax.set_ylim(120, N_SUBSAMPLE + 3)
ax.grid(True, alpha=0.3, which='both')
ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())

# Annotate NISQ range
ax.axvspan(0.001, 0.01, alpha=0.07, color='crimson')
ax.text(0.003, 151, 'NISQ\nrange', ha='center',
        fontsize=7.5, color='crimson', va='bottom')

plt.tight_layout()
for ext in ('pdf', 'png'):
    plt.savefig(f'noise_sensitivity_final.{ext}', dpi=150, bbox_inches='tight')
print("\nSaved: noise_sensitivity_final.pdf / .png")
plt.show()
