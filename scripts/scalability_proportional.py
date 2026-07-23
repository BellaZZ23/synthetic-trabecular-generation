"""
scalability_proportional.py  (v2 — full comparative analysis)
──────────────────────────────────────────────────────────────
Scalability sweep with N proportional to Hilbert dimension.

Compares three approaches at each qubit count:
  1. Quantum ZZ kernel  +  UMAP reduction   (paper's method)
  2. Quantum ZZ kernel  +  PCA reduction    (linear baseline for reduction)
  3. Classical RBF-SVM  +  UMAP reduction   (classical kernel baseline)

Improvements over v1:
  - PCA alongside UMAP to test whether reduction method matters
  - Classical RBF-SVM as a reference line so quantum adds-value claims are grounded
  - 5-fold stratified CV per dataset (not single 80/20 split) for reliable estimates
  - 95% confidence intervals  (mean ± 1.96 × std / sqrt(N_DATASETS))
  - Wilcoxon signed-rank tests between adjacent qubit counts
  - N = min(SCALE_FACTOR × 2^n, N_MAX) so each qubit count is fairly sampled
  - Statevector caching: O(N) circuit evals instead of O(N²)

N plan  (SCALE_FACTOR=4, N_MAX=500):
  4q  →  N=64    (4.0× Hilbert dim  16)
  6q  →  N=256   (4.0× Hilbert dim  64)
  8q  →  N=500   (2.0× Hilbert dim 256)  [capped]
 10q  →  N=500   (0.5× Hilbert dim 1024) [capped — undersampled]

Run from:  synthetic_trabeculae/
  python scripts/scalability_proportional.py
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import skew, kurtosis, wilcoxon
from scipy.stats import t as t_dist

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from umap import UMAP

from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

warnings.filterwarnings("ignore", category=UserWarning)

# ── Configuration ─────────────────────────────────────────────────────────────
DATASETS_BASE = Path("output/independent_test_v2")
N_DATASETS    = 10
QUBIT_COUNTS  = [4, 6, 8, 10]
N_REPS        = 2
SEED          = 42
IMAGE_SIZE    = 64
CV_FOLDS      = 5

SCALE_FACTOR  = 4
N_MAX         = 500

UMAP_NEIGHBORS = 15
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
    hist     = hist.astype(np.float32) / (hist.sum() + 1e-8)
    feats.extend(hist.tolist())
    gx   = ndi.sobel(img, axis=1)
    gy   = ndi.sobel(img, axis=0)
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


def load_dataset_features(dataset_path: Path):
    dataset_dir = dataset_path / "dataset"
    if not dataset_dir.exists():
        dataset_dir = dataset_path
    sample_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    X_rows, bvtv_vals = [], []
    for d in sample_dirs:
        gray_path = d / "gray.tif"
        mask_path = d / "mask.tif"
        if not gray_path.exists():
            continue
        vol  = tiff.imread(str(gray_path)).astype(np.float32)
        vmax = vol.max()
        if vmax > 0:
            vol /= vmax
        mid  = vol[vol.shape[0] // 2]
        feat = _extract_texture_features(mid)
        bvtv = float((tiff.imread(str(mask_path)) > 0).mean()) \
               if mask_path.exists() else float((vol > 0.5).mean())
        met_path = d / "metrics.json"
        if met_path.exists():
            with open(met_path) as f:
                met = json.load(f)
            bvtv = float(met.get("morphometrics", {}).get("BVTV", bvtv))
        X_rows.append(feat)
        bvtv_vals.append(bvtv)
    if not X_rows:
        raise RuntimeError(f"No valid samples in {dataset_path}")
    X        = np.array(X_rows, dtype=np.float32)
    bvtv_arr = np.array(bvtv_vals, dtype=np.float32)
    return X, (bvtv_arr > float(np.median(bvtv_arr))).astype(np.int32)


# ── Quantum kernel helpers ────────────────────────────────────────────────────

def effective_rank(K):
    eigvals = np.maximum(np.linalg.eigvalsh(K), 0)
    s = eigvals.sum()
    return 1.0 if s == 0 else float(s ** 2 / (eigvals ** 2).sum())


def compute_kernel_cached(X, feature_map):
    """O(N) statevector evals + O(N²) numpy inner products."""
    params = list(feature_map.parameters)
    svs = [Statevector.from_instruction(
               feature_map.assign_parameters(dict(zip(params, xi))))
           for xi in X]
    n = len(X)
    K = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            K[i, j] = K[j, i] = float(abs(svs[i].inner(svs[j])) ** 2)
    return K


def cv_precomputed(K, y, n_splits=CV_FOLDS):
    """Stratified k-fold CV on a precomputed kernel matrix."""
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    accs = []
    for tr, te in skf.split(K, y):
        clf = SVC(kernel='precomputed', C=1.0)
        clf.fit(K[np.ix_(tr, tr)], y[tr])
        accs.append(clf.score(K[np.ix_(te, tr)], y[te]))
    return float(np.mean(accs))


def cv_classical(X, y, n_splits=CV_FOLDS):
    """Stratified k-fold CV for classical RBF-SVM."""
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    accs = []
    for tr, te in skf.split(X, y):
        clf = SVC(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def ci95(values):
    """95% confidence interval half-width: 1.96 × std / sqrt(n)."""
    n   = len(values)
    sem = np.std(values, ddof=1) / np.sqrt(n)
    return float(1.96 * sem)


def wilcoxon_p(a, b):
    """Wilcoxon signed-rank test p-value (two-sided). Returns 1.0 if identical."""
    try:
        _, p = wilcoxon(a, b, alternative='two-sided')
        return float(p)
    except Exception:
        return 1.0


# ── Pre-load datasets ─────────────────────────────────────────────────────────
print("=" * 65)
print("Scalability — proportional N, PCA vs UMAP vs Classical (v2)")
print(f"N = min({SCALE_FACTOR}×2^n, {N_MAX})   CV={CV_FOLDS}-fold   CI=95%")
print("=" * 65)

dataset_paths = [DATASETS_BASE / f"dataset_{i:02d}" for i in range(N_DATASETS)]
for p in dataset_paths:
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")

print("\nSample sizes:")
for n_q in QUBIT_COUNTS:
    N   = min(SCALE_FACTOR * (2 ** n_q), N_MAX)
    tag = " (capped — undersampled)" if N == N_MAX and N < 2**n_q else ""
    print(f"  {n_q}q  N={N:>3}  ratio={N/(2**n_q):.2f}×{tag}")

print(f"\nPre-extracting features …\n")
all_X, all_y = [], []
for i, dp in enumerate(dataset_paths):
    print(f"  dataset_{i:02d} … ", end="", flush=True)
    t0   = time.time()
    X, y = load_dataset_features(dp)
    all_X.append(X)
    all_y.append(y)
    print(f"{X.shape[0]} samples  ({time.time()-t0:.1f}s)")

print("\nStarting sweep …\n")

# ── Main sweep ────────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

# results[n_q] = {umap_q, pca_q, rbf_classical} each a list of N_DATASETS accuracies
all_results = {}

for n_q in QUBIT_COUNTS:
    hilbert = 2 ** n_q
    N       = min(SCALE_FACTOR * hilbert, N_MAX)
    capped  = N == N_MAX and N < hilbert

    print(f"{'─'*65}")
    print(f"  {n_q}q  Hilbert={hilbert}  N={N}  ({N/hilbert:.2f}×)"
          + ("  ← undersampled" if capped else "  ← oversampled ✓"))
    print(f"{'─'*65}")

    fm    = ZZFeatureMap(feature_dimension=n_q, reps=N_REPS)
    depth = fm.decompose().depth()
    print(f"  Circuit depth: {depth}\n")

    ds_umap_q, ds_pca_q, ds_rbf, ds_er_umap, ds_er_pca = [], [], [], [], []
    t0_qubit = time.time()

    for ds_idx in range(N_DATASETS):
        X_raw = all_X[ds_idx]
        y_raw = all_y[ds_idx]

        idx = rng.choice(len(X_raw), N, replace=False)
        X_s = X_raw[idx]
        y_s = y_raw[idx]

        X_scaled = RobustScaler().fit_transform(X_s)

        # ── UMAP reduction ──────────────────────────────────────────────
        X_umap = UMAP(n_components=n_q,
                      n_neighbors=min(UMAP_NEIGHBORS, N - 1),
                      min_dist=UMAP_MIN_DIST,
                      random_state=SEED).fit_transform(X_scaled)
        X_umap_q = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_umap)

        # ── PCA reduction ───────────────────────────────────────────────
        X_pca  = PCA(n_components=n_q, random_state=SEED).fit_transform(X_scaled)
        X_pca_q = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_pca)

        # ── Quantum kernels (cached) ────────────────────────────────────
        tk = time.time()
        K_umap = compute_kernel_cached(X_umap_q, fm)
        K_pca  = compute_kernel_cached(X_pca_q,  fm)
        kernel_time = time.time() - tk

        er_umap = effective_rank(K_umap)
        er_pca  = effective_rank(K_pca)

        # ── Accuracies ──────────────────────────────────────────────────
        acc_umap_q = cv_precomputed(K_umap, y_s)
        acc_pca_q  = cv_precomputed(K_pca,  y_s)
        acc_rbf    = cv_classical(X_umap, y_s)   # classical RBF on UMAP features

        ds_umap_q.append(acc_umap_q)
        ds_pca_q.append(acc_pca_q)
        ds_rbf.append(acc_rbf)
        ds_er_umap.append(er_umap)
        ds_er_pca.append(er_pca)

        elapsed = (time.time() - t0_qubit) / 60
        print(f"    ds_{ds_idx:02d}  "
              f"Q-UMAP={acc_umap_q:.3f}  Q-PCA={acc_pca_q:.3f}  "
              f"Classical={acc_rbf:.3f}  "
              f"er_umap={er_umap:.1f}/{N}  kernel={kernel_time:.0f}s  "
              f"total={elapsed:.1f}min")

    all_results[n_q] = dict(
        N           = N,
        hilbert     = hilbert,
        depth       = depth,
        capped      = capped,
        umap_q      = ds_umap_q,
        pca_q       = ds_pca_q,
        rbf         = ds_rbf,
        er_umap     = ds_er_umap,
        er_pca      = ds_er_pca,
    )

    mean_uq = np.mean(ds_umap_q);  ci_uq = ci95(ds_umap_q)
    mean_pq = np.mean(ds_pca_q);   ci_pq = ci95(ds_pca_q)
    mean_rb = np.mean(ds_rbf);     ci_rb = ci95(ds_rbf)
    total_t = (time.time() - t0_qubit) / 60

    print(f"\n  {n_q}q SUMMARY:")
    print(f"    Q-UMAP   = {mean_uq:.3f} ± {ci_uq:.3f} (95% CI)")
    print(f"    Q-PCA    = {mean_pq:.3f} ± {ci_pq:.3f} (95% CI)")
    print(f"    Classical= {mean_rb:.3f} ± {ci_rb:.3f} (95% CI)")
    print(f"    Time: {total_t:.1f} min\n")

np.save('scalability_proportional_v2_results.npy', all_results)
print("Results saved → scalability_proportional_v2_results.npy\n")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'Qubits':>7}  {'N':>5}  {'Q-UMAP (95%CI)':>18}  "
      f"{'Q-PCA (95%CI)':>17}  {'Classical (95%CI)':>20}")
print("─" * 75)
for n_q in QUBIT_COUNTS:
    r  = all_results[n_q]
    uq = f"{np.mean(r['umap_q']):.3f}±{ci95(r['umap_q']):.3f}"
    pq = f"{np.mean(r['pca_q']):.3f}±{ci95(r['pca_q']):.3f}"
    rb = f"{np.mean(r['rbf']):.3f}±{ci95(r['rbf']):.3f}"
    flag = "*" if r['capped'] else " "
    print(f"{n_q:>7}{flag} {r['N']:>5}  {uq:>18}  {pq:>17}  {rb:>20}")
print("  * capped — undersampled")

# ── Wilcoxon tests between adjacent qubit counts ──────────────────────────────
print("\nWilcoxon signed-rank tests (Q-UMAP, adjacent qubit counts):")
print(f"  {'Comparison':>15}  {'p-value':>10}  {'Significant?':>14}")
print("  " + "─" * 44)
for i in range(len(QUBIT_COUNTS) - 1):
    qa = QUBIT_COUNTS[i];  qb = QUBIT_COUNTS[i + 1]
    p  = wilcoxon_p(all_results[qa]['umap_q'], all_results[qb]['umap_q'])
    sig = "Yes (p<0.05)" if p < 0.05 else "No"
    print(f"  {qa}q vs {qb}q:         p = {p:.4f}    {sig}")

print("\nWilcoxon signed-rank tests (Q-UMAP vs Classical, each qubit count):")
print(f"  {'Qubit count':>12}  {'p-value':>10}  {'Quantum > Classical?':>22}")
print("  " + "─" * 50)
for n_q in QUBIT_COUNTS:
    r  = all_results[n_q]
    p  = wilcoxon_p(r['umap_q'], r['rbf'])
    diff = np.mean(r['umap_q']) - np.mean(r['rbf'])
    sig  = f"{'Yes' if p < 0.05 else 'No'} (Δ={diff:+.3f})"
    print(f"  {n_q}q:          p = {p:.4f}    {sig}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

qubits  = QUBIT_COUNTS
colors  = {'Q-UMAP': 'steelblue', 'Q-PCA': 'darkorange', 'Classical RBF': 'seagreen'}
markers = {'Q-UMAP': 'o',         'Q-PCA': 's',          'Classical RBF': '^'}

methods = [
    ('Q-UMAP',       'umap_q'),
    ('Q-PCA',        'pca_q'),
    ('Classical RBF','rbf'),
]

# Panel 1 — accuracy comparison
ax = axes[0]
for label, key in methods:
    means = [np.mean(all_results[q][key]) for q in qubits]
    cis   = [ci95(all_results[q][key])    for q in qubits]
    ax.errorbar(qubits, means, yerr=cis,
                marker=markers[label], color=colors[label],
                capsize=5, linewidth=1.8, markersize=8, label=label)
    # Scatter individual dataset points
    for q in qubits:
        ax.scatter([q] * N_DATASETS, all_results[q][key],
                   alpha=0.15, s=15, color=colors[label])

# Mark undersampled qubits
capped_qs = [q for q in qubits if all_results[q]['capped']]
if capped_qs:
    ax.axvspan(min(capped_qs) - 0.4, max(capped_qs) + 0.4,
               alpha=0.05, color='gray')
    ax.text(min(capped_qs), 0.915, 'Undersampled\n(N < Hilbert dim)',
            ha='center', fontsize=7.5, color='gray')

ax.axhline(0.5, linestyle=':', color='black', linewidth=1, label='Chance')
ax.set_xlabel('Number of qubits', fontsize=11)
ax.set_ylabel('BV/TV accuracy  (5-fold CV mean)', fontsize=11)
ax.set_title('Accuracy vs Qubits\n(N proportional to Hilbert dim, 95% CI)', fontsize=10)
ax.set_xticks(qubits)
ax.set_xticklabels([f'{q}q\nN={all_results[q]["N"]}' for q in qubits])
ax.legend(fontsize=9)
ax.set_ylim(0.38, 0.95)
ax.grid(True, alpha=0.3)

# Panel 2 — kernel concentration (UMAP vs PCA quantum kernels)
ax = axes[1]
for label, key_er, col in [('Q-UMAP', 'er_umap', 'steelblue'),
                             ('Q-PCA',  'er_pca',  'darkorange')]:
    means = [np.mean(all_results[q][key_er]) / all_results[q]['N'] for q in qubits]
    ax.plot(qubits, means, marker='o', color=col, linewidth=1.8,
            markersize=8, label=label)

ax.axhline(1.0, linestyle='--', color='black', linewidth=1, label='Identity limit')
if capped_qs:
    ax.axvspan(min(capped_qs) - 0.4, max(capped_qs) + 0.4,
               alpha=0.05, color='gray', label='Undersampled')
ax.set_xlabel('Number of qubits', fontsize=11)
ax.set_ylabel('Effective rank / N  (lower = more structure)', fontsize=11)
ax.set_title('Kernel Concentration vs Qubits\nUMAP vs PCA encoding', fontsize=10)
ax.set_xticks(qubits)
ax.set_xticklabels([f'{q}q\nN={all_results[q]["N"]}' for q in qubits])
ax.set_ylim(0, 1.1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3 — Wilcoxon p-values (Q-UMAP vs Classical per qubit)
ax = axes[2]
p_vals  = [wilcoxon_p(all_results[q]['umap_q'], all_results[q]['rbf'])
           for q in qubits]
deltas  = [np.mean(all_results[q]['umap_q']) - np.mean(all_results[q]['rbf'])
           for q in qubits]
bar_colors = ['#d9534f' if p >= 0.05 else '#5cb85c' for p in p_vals]
bars = ax.bar([f'{q}q' for q in qubits], deltas, color=bar_colors,
              edgecolor='black', linewidth=0.8, alpha=0.8)
for i, (bar, p) in enumerate(zip(bars, p_vals)):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.003 if bar.get_height() >= 0 else -0.012),
            f'p={p:.3f}', ha='center', va='bottom', fontsize=9)
ax.axhline(0, linestyle='-', color='black', linewidth=1)
ax.axhline(0, linestyle='--', color='gray', linewidth=0.5, alpha=0.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#5cb85c', label='p < 0.05 (significant)'),
                   Patch(color='#d9534f', label='p ≥ 0.05 (not significant)')],
          fontsize=8)
ax.set_xlabel('Qubit count', fontsize=11)
ax.set_ylabel('Accuracy delta  (Q-UMAP − Classical RBF)', fontsize=11)
ax.set_title('Q-UMAP vs Classical RBF\nWilcoxon signed-rank test', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(
    f'QKSVM Scalability — Proportional Sampling  '
    f'(N={SCALE_FACTOR}×2^n, capped {N_MAX})  |  '
    f'{N_DATASETS} independent datasets  |  {CV_FOLDS}-fold CV',
    fontsize=10, y=1.01)
plt.tight_layout()
for ext in ('pdf', 'png'):
    plt.savefig(f'scalability_proportional_v2.{ext}', dpi=150, bbox_inches='tight')
print("Saved: scalability_proportional_v2.pdf / .png")
plt.show()
