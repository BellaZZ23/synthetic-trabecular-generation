"""
hyperparameter_optimization.py
───────────────────────────────
Systematic search for the best quantum kernel SVM configuration.
Fixed at 8 qubits (paper setting) with N=500 (proportional to Hilbert dim 256).

Tests four improvement axes:
  1. SVM C parameter       [0.01, 0.1, 1, 10, 100]
  2. Kernel centering      [True, False]
  3. ZZFeatureMap reps     [1, 2, 3]
  4. UMAP n_neighbors      [5, 15, 30, 50]

For each configuration: 10 independent datasets × 5-fold CV → mean ± 95% CI.
Classical RBF-SVM with optimal C included as baseline at each UMAP setting.

Key optimisation: statevector caching makes each N=500 kernel ~30s instead of hours.

Run from:  synthetic_trabeculae/
  python scripts/hyperparameter_optimization.py
"""

import json
import time
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import skew, kurtosis

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from umap import UMAP

from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

warnings.filterwarnings("ignore", category=UserWarning)

# ── Configuration ─────────────────────────────────────────────────────────────
DATASETS_BASE  = Path("output/independent_test_v2")
N_DATASETS     = 10
N_QUBITS       = 8
N_MAX          = 500
SEED           = 42
IMAGE_SIZE     = 64
CV_FOLDS       = 5

# Search axes
C_VALUES       = [0.01, 0.1, 1.0, 10.0, 100.0]
REPS_VALUES    = [1, 2, 3]
NEIGHBORS_VALUES = [5, 15, 30, 50]
UMAP_MIN_DIST  = 0.1

# ── Texture feature extraction ────────────────────────────────────────────────
try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_GLCM = True
except ImportError:
    SKIMAGE_GLCM = False

def _extract_glcm_features(img_u8):
    if not SKIMAGE_GLCM:
        return np.array([])
    distances = [1, 3]
    angles    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    props     = ["contrast", "dissimilarity", "homogeneity",
                 "energy", "correlation", "ASM"]
    img_q = (img_u8 // 8).astype(np.uint8)
    glcm  = graycomatrix(img_q, distances=distances, angles=angles,
                         levels=32, symmetric=True, normed=True)
    feats = []
    for prop in props:
        vals = graycoprops(glcm, prop)
        feats.extend([float(vals.mean()), float(vals.std())])
    return np.array(feats, dtype=np.float32)

def _extract_statistical_features(img):
    feats = [float(img.mean()), float(img.std()), float(np.median(img)),
             float(skew(img.ravel())), float(kurtosis(img.ravel())),
             float(img.min()), float(img.max()),
             float(np.percentile(img, 25)), float(np.percentile(img, 75))]
    hist, _ = np.histogram(img.ravel(), bins=16, range=(0, 1))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
    feats.extend(hist.tolist())
    gx = ndi.sobel(img, axis=1); gy = ndi.sobel(img, axis=0)
    gmag = np.sqrt(gx**2 + gy**2)
    feats += [float(gmag.mean()), float(gmag.std()),
              float(np.percentile(gmag, 75)), float(np.percentile(gmag, 95))]
    lv = ndi.uniform_filter(img**2, size=5) - ndi.uniform_filter(img, size=5)**2
    feats += [float(lv.mean()), float(lv.std())]
    return np.array(feats, dtype=np.float32)

def _extract_texture_features(sl):
    img     = Image.fromarray((sl * 255).astype(np.uint8), mode="L")
    img     = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    resized = np.array(img, dtype=np.float32) / 255.0
    img_u8  = (resized * 255).astype(np.uint8)
    stat    = _extract_statistical_features(resized)
    glcm    = _extract_glcm_features(img_u8)
    return np.concatenate([stat, glcm]) if glcm.size > 0 else stat

def load_dataset_features(dataset_path):
    dataset_dir = dataset_path / "dataset"
    if not dataset_dir.exists():
        dataset_dir = dataset_path
    sample_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    X_rows, bvtv_vals = [], []
    for d in sample_dirs:
        gray_path = d / "gray.tif"; mask_path = d / "mask.tif"
        if not gray_path.exists(): continue
        vol = tiff.imread(str(gray_path)).astype(np.float32)
        vmax = vol.max()
        if vmax > 0: vol /= vmax
        mid = vol[vol.shape[0] // 2]
        feat = _extract_texture_features(mid)
        bvtv = float((tiff.imread(str(mask_path)) > 0).mean()) \
               if mask_path.exists() else float((vol > 0.5).mean())
        met_path = d / "metrics.json"
        if met_path.exists():
            with open(met_path) as f: met = json.load(f)
            bvtv = float(met.get("morphometrics", {}).get("BVTV", bvtv))
        X_rows.append(feat); bvtv_vals.append(bvtv)
    X = np.array(X_rows, dtype=np.float32)
    bvtv_arr = np.array(bvtv_vals, dtype=np.float32)
    return X, (bvtv_arr > float(np.median(bvtv_arr))).astype(np.int32)

# ── Kernel helpers ────────────────────────────────────────────────────────────
def center_kernel(K):
    """Kernel centering: removes constant component, emphasises off-diagonal structure."""
    n   = K.shape[0]
    one = np.ones((n, n)) / n
    return K - one @ K - K @ one + one @ K @ one

def compute_kernel_cached(X, feature_map):
    """O(N) statevector evals + O(N²) numpy inner products."""
    params = list(feature_map.parameters)
    svs = [Statevector.from_instruction(
               feature_map.assign_parameters(dict(zip(params, xi))))
           for xi in X]
    n = len(X); K = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            K[i, j] = K[j, i] = float(abs(svs[i].inner(svs[j])) ** 2)
    return K

def effective_rank(K):
    eigvals = np.maximum(np.linalg.eigvalsh(K), 0)
    s = eigvals.sum()
    return 1.0 if s == 0 else float(s ** 2 / (eigvals ** 2).sum())

def cv_quantum(K, y, C=1.0, center=False):
    K_use = center_kernel(K) if center else K
    skf   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    accs  = []
    for tr, te in skf.split(K_use, y):
        clf = SVC(kernel='precomputed', C=C)
        clf.fit(K_use[np.ix_(tr, tr)], y[tr])
        accs.append(clf.score(K_use[np.ix_(te, tr)], y[te]))
    return float(np.mean(accs))

def cv_classical(X, y, C=1.0):
    skf  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    accs = []
    for tr, te in skf.split(X, y):
        clf = SVC(kernel='rbf', C=C, gamma='scale')
        clf.fit(X[tr], y[tr]); accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))

def ci95(vals):
    return 1.96 * float(np.std(vals, ddof=1)) / np.sqrt(len(vals))

# ── Load datasets ─────────────────────────────────────────────────────────────
print("=" * 65)
print(f"Hyperparameter optimisation — {N_QUBITS} qubits, N={N_MAX}")
print("=" * 65)

dataset_paths = [DATASETS_BASE / f"dataset_{i:02d}" for i in range(N_DATASETS)]
for p in dataset_paths:
    if not p.exists(): raise FileNotFoundError(f"Missing: {p}")

print(f"\nPre-extracting features from {N_DATASETS} datasets …")
all_X, all_y = [], []
rng = np.random.default_rng(SEED)
for i, dp in enumerate(dataset_paths):
    print(f"  dataset_{i:02d} … ", end="", flush=True)
    t0 = time.time(); X, y = load_dataset_features(dp)
    all_X.append(X); all_y.append(y)
    print(f"{X.shape[0]} samples  ({time.time()-t0:.1f}s)")

# ── Experiment 1: C parameter + kernel centering at fixed reps=2, neighbors=15
print("\n" + "=" * 65)
print("EXPERIMENT 1: C tuning + kernel centering")
print(f"  reps=2, UMAP n_neighbors=15, N={N_MAX}")
print("=" * 65)

fm_base = ZZFeatureMap(feature_dimension=N_QUBITS, reps=2)
exp1_results = {}  # (C, center) -> list of 10 accs

# Pre-compute kernels for all datasets (reuse across C values)
print("\nPre-computing kernels for all datasets …")
all_K, all_X_umap = [], []
for ds_idx in range(N_DATASETS):
    idx = rng.choice(len(all_X[ds_idx]), N_MAX, replace=False)
    X_s = all_X[ds_idx][idx]; y_s = all_y[ds_idx][idx]
    X_sc = RobustScaler().fit_transform(X_s)
    X_um = UMAP(n_components=N_QUBITS, n_neighbors=15, min_dist=UMAP_MIN_DIST,
                random_state=SEED).fit_transform(X_sc)
    X_q  = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_um)
    print(f"  ds_{ds_idx:02d} kernel … ", end="", flush=True)
    t0 = time.time()
    K = compute_kernel_cached(X_q, fm_base)
    all_K.append((K, y_s, X_um))
    print(f"er={effective_rank(K):.1f}/{N_MAX}  ({time.time()-t0:.0f}s)")

print("\nC sweep (quantum + classical):")
print(f"  {'C':>8}  {'Centered':>9}  {'Q-Acc':>8}  {'Cl-Acc':>8}")
print("  " + "─" * 40)
for C, center in product(C_VALUES, [False, True]):
    accs_q, accs_cl = [], []
    for K, y_s, X_um in all_K:
        accs_q.append(cv_quantum(K, y_s, C=C, center=center))
        accs_cl.append(cv_classical(X_um, y_s, C=C))
    exp1_results[(C, center)] = dict(q=accs_q, cl=accs_cl)
    print(f"  C={C:>6}  center={str(center):>5}  "
          f"Q={np.mean(accs_q):.3f}±{ci95(accs_q):.3f}  "
          f"Cl={np.mean(accs_cl):.3f}±{ci95(accs_cl):.3f}")

# Best configs
best_q  = max(exp1_results.items(), key=lambda x: np.mean(x[1]['q']))
best_cl = max(exp1_results.items(), key=lambda x: np.mean(x[1]['cl']))
print(f"\n  Best Q:   C={best_q[0][0]}, center={best_q[0][1]}  "
      f"→ {np.mean(best_q[1]['q']):.3f}")
print(f"  Best Cl:  C={best_cl[0][0]}, center={best_cl[0][1]}  "
      f"→ {np.mean(best_cl[1]['cl']):.3f}")

# ── Experiment 2: ZZFeatureMap reps (1, 2, 3) ─────────────────────────────
print("\n" + "=" * 65)
print("EXPERIMENT 2: ZZFeatureMap reps [1, 2, 3]")
print(f"  Best C from Exp 1, UMAP n_neighbors=15, N={N_MAX}")
print("=" * 65)

best_C = best_q[0][0]; use_center = best_q[0][1]
exp2_results = {}

for n_reps in REPS_VALUES:
    fm_r = ZZFeatureMap(feature_dimension=N_QUBITS, reps=n_reps)
    depth = fm_r.decompose().depth()
    print(f"\n  reps={n_reps}  depth={depth}")
    accs_q = []
    for ds_idx in range(N_DATASETS):
        idx = rng.choice(len(all_X[ds_idx]), N_MAX, replace=False)
        X_s = all_X[ds_idx][idx]; y_s = all_y[ds_idx][idx]
        X_sc = RobustScaler().fit_transform(X_s)
        X_um = UMAP(n_components=N_QUBITS, n_neighbors=15, min_dist=UMAP_MIN_DIST,
                    random_state=SEED).fit_transform(X_sc)
        X_q  = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_um)
        t0 = time.time()
        K = compute_kernel_cached(X_q, fm_r)
        acc = cv_quantum(K, y_s, C=best_C, center=use_center)
        accs_q.append(acc)
        print(f"    ds_{ds_idx:02d}  acc={acc:.3f}  "
              f"er={effective_rank(K):.1f}/{N_MAX}  ({time.time()-t0:.0f}s)")
    exp2_results[n_reps] = dict(accs=accs_q, depth=depth)
    print(f"  → reps={n_reps}: {np.mean(accs_q):.3f} ± {ci95(accs_q):.3f}")

# ── Experiment 3: UMAP n_neighbors ─────────────────────────────────────────
print("\n" + "=" * 65)
print("EXPERIMENT 3: UMAP n_neighbors [5, 15, 30, 50]")
print(f"  Best C and reps from above, N={N_MAX}")
print("=" * 65)

best_reps = max(exp2_results.items(), key=lambda x: np.mean(x[1]['accs']))[0]
fm_best = ZZFeatureMap(feature_dimension=N_QUBITS, reps=best_reps)
exp3_results = {}

for nn in NEIGHBORS_VALUES:
    print(f"\n  n_neighbors={nn}")
    accs_q, accs_cl = [], []
    for ds_idx in range(N_DATASETS):
        idx = rng.choice(len(all_X[ds_idx]), N_MAX, replace=False)
        X_s = all_X[ds_idx][idx]; y_s = all_y[ds_idx][idx]
        X_sc = RobustScaler().fit_transform(X_s)
        X_um = UMAP(n_components=N_QUBITS, n_neighbors=nn, min_dist=UMAP_MIN_DIST,
                    random_state=SEED).fit_transform(X_sc)
        X_q  = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_um)
        t0 = time.time()
        K = compute_kernel_cached(X_q, fm_best)
        acc_q  = cv_quantum(K, y_s, C=best_C, center=use_center)
        acc_cl = cv_classical(X_um, y_s, C=best_C)
        accs_q.append(acc_q); accs_cl.append(acc_cl)
        print(f"    ds_{ds_idx:02d}  Q={acc_q:.3f}  Cl={acc_cl:.3f}  ({time.time()-t0:.0f}s)")
    exp3_results[nn] = dict(q=accs_q, cl=accs_cl)
    print(f"  → Q={np.mean(accs_q):.3f}±{ci95(accs_q):.3f}  "
          f"Cl={np.mean(accs_cl):.3f}±{ci95(accs_cl):.3f}")

# ── Save all results ──────────────────────────────────────────────────────────
np.save('hyperparameter_results.npy',
        {'exp1': exp1_results, 'exp2': exp2_results, 'exp3': exp3_results},
        allow_pickle=True)
print("\nResults saved → hyperparameter_results.npy")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL SUMMARY — best configuration found")
print("=" * 65)

best_nn  = max(exp3_results.items(), key=lambda x: np.mean(x[1]['q']))
print(f"  Quantum  best: C={best_C}, center={use_center}, "
      f"reps={best_reps}, n_neighbors={best_nn[0]}")
print(f"  Accuracy: {np.mean(best_nn[1]['q']):.3f} ± {ci95(best_nn[1]['q']):.3f}")
print(f"  Classical best at same setting: "
      f"{np.mean(best_nn[1]['cl']):.3f} ± {ci95(best_nn[1]['cl']):.3f}")
print(f"  Baseline (C=1, no center, reps=2, nn=15): "
      f"{np.mean(exp1_results[(1.0, False)]['q']):.3f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1 — C tuning (no center vs centered, quantum)
ax = axes[0]
C_arr = C_VALUES
for center, ls, lbl in [(False, '-o', 'Q (raw kernel)'),
                         (True,  '-s', 'Q (centered kernel)')]:
    means = [np.mean(exp1_results[(C, center)]['q']) for C in C_arr]
    cis   = [ci95(exp1_results[(C, center)]['q'])    for C in C_arr]
    ax.errorbar(range(len(C_arr)), means, yerr=cis, fmt=ls,
                capsize=4, linewidth=1.6, markersize=7, label=lbl)
cl_means = [np.mean(exp1_results[(C, False)]['cl']) for C in C_arr]
ax.plot(range(len(C_arr)), cl_means, '--^', color='seagreen',
        linewidth=1.4, label='Classical RBF')
ax.axhline(0.5, ':', color='black', linewidth=1)
ax.set_xticks(range(len(C_arr))); ax.set_xticklabels([str(c) for c in C_arr])
ax.set_xlabel('C (SVM regularisation)', fontsize=11)
ax.set_ylabel('Accuracy (5-fold CV, 95% CI)', fontsize=11)
ax.set_title('Effect of C and Kernel Centering', fontsize=10)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel 2 — reps comparison
ax = axes[1]
reps_list = REPS_VALUES
means = [np.mean(exp2_results[r]['accs']) for r in reps_list]
cis   = [ci95(exp2_results[r]['accs'])    for r in reps_list]
depths = [exp2_results[r]['depth']         for r in reps_list]
ax.errorbar(reps_list, means, yerr=cis, fmt='-o', color='steelblue',
            capsize=4, linewidth=1.6, markersize=8)
ax.axhline(0.5, linestyle=':', color='black', linewidth=1)
for r, d, m in zip(reps_list, depths, means):
    ax.annotate(f'depth={d}', (r, m), textcoords='offset points',
                xytext=(5, 5), fontsize=8, color='gray')
ax.set_xlabel('ZZFeatureMap reps', fontsize=11)
ax.set_ylabel('Accuracy (5-fold CV, 95% CI)', fontsize=11)
ax.set_title('Effect of Circuit Depth (reps)', fontsize=10)
ax.set_xticks(reps_list); ax.grid(True, alpha=0.3)

# Panel 3 — UMAP n_neighbors
ax = axes[2]
nn_list = NEIGHBORS_VALUES
q_means  = [np.mean(exp3_results[nn]['q'])  for nn in nn_list]
q_cis    = [ci95(exp3_results[nn]['q'])      for nn in nn_list]
cl_means = [np.mean(exp3_results[nn]['cl']) for nn in nn_list]
cl_cis   = [ci95(exp3_results[nn]['cl'])     for nn in nn_list]
ax.errorbar(nn_list, q_means,  yerr=q_cis,  fmt='-o', color='steelblue',
            capsize=4, linewidth=1.6, markersize=7, label='Quantum ZZ')
ax.errorbar(nn_list, cl_means, yerr=cl_cis, fmt='--^', color='seagreen',
            capsize=4, linewidth=1.4, markersize=7, label='Classical RBF')
ax.axhline(0.5, linestyle=':', color='black', linewidth=1)
ax.set_xlabel('UMAP n_neighbors', fontsize=11)
ax.set_ylabel('Accuracy (5-fold CV, 95% CI)', fontsize=11)
ax.set_title('Effect of UMAP Neighbourhood Scale', fontsize=10)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.suptitle(
    f'Hyperparameter Optimisation — {N_QUBITS}q ZZFeatureMap, '
    f'N={N_MAX}, {N_DATASETS} independent datasets',
    fontsize=10, y=1.01)
plt.tight_layout()
for ext in ('pdf', 'png'):
    plt.savefig(f'hyperparameter_optimization.{ext}', dpi=150, bbox_inches='tight')
print("Saved: hyperparameter_optimization.pdf / .png")
plt.show()
