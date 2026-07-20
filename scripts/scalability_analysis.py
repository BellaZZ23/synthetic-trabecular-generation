"""
scalability_analysis.py  (v3 — independent-dataset validation)
───────────────────────────────────────────────────────────────
Tests how quantum kernel SVM performance and kernel structure change
as qubit count increases from 4 → 10, using UMAP-reduced texture features.

Validation strategy:
  For each qubit count n:
    For each of 10 independently generated datasets:
      1. Extract 43 texture features from gray.tif mid-slices
      2. Reduce to n dimensions with UMAP (same hyperparameters as paper)
      3. Scale to [0, pi] and build ZZFeatureMap(n_qubits=n, reps=2)
      4. Subsample N_SUBSAMPLE points; compute kernel matrix (statevector)
      5. 80/20 stratified train/test split -> one accuracy value
    -> 10 independent accuracy values per qubit count -> mean +/- std

Error bars reflect genuine dataset-to-dataset variability, not CV fold noise.

Prerequisites
-------------
pip install qiskit qiskit-aer scikit-learn umap-learn tifffile pillow scipy scikit-image

Dataset layout expected:
  output/independent_test_v2/dataset_00/dataset/sample_000/gray.tif
  output/independent_test_v2/dataset_00/dataset/sample_000/mask.tif
  ...
  output/independent_test_v2/dataset_09/dataset/sample_499/{gray,mask}.tif

Compute time estimate (N_SUBSAMPLE=100, statevector):
   4q -> ~1  min/dataset x 10 = ~10  min
   6q -> ~3  min/dataset x 10 = ~30  min
   8q -> ~8  min/dataset x 10 = ~80  min
  10q -> ~25 min/dataset x 10 = ~250 min
  Total: ~6 hours. Run overnight.
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
from scipy.stats import skew, kurtosis

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from umap import UMAP

from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

warnings.filterwarnings("ignore", category=UserWarning)

# ── Configuration ─────────────────────────────────────────────────────────────
DATASETS_BASE  = Path("output/independent_test_v2")
N_DATASETS     = 10
N_SUBSAMPLE    = 100           # samples per dataset used for kernel computation
TRAIN_FRAC     = 0.8           # 80 train / 20 test
QUBIT_COUNTS   = [4, 6, 8, 10]
N_REPS         = 2             # ZZFeatureMap repetitions (matches paper)
SEED           = 42
IMAGE_SIZE     = 64            # resize mid-slice before feature extraction

# UMAP hyperparameters — identical to original paper
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST  = 0.1

# ── Texture feature extraction (mirrors full_pipeline.py / prepare_analysis_data.py)

try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_GLCM = True
except ImportError:
    SKIMAGE_GLCM = False
    print("Warning: scikit-image not found; GLCM features will be skipped.")


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
    """Extract 43 texture features from a 2-D grayscale mid-slice."""
    img     = Image.fromarray((sl * 255).astype(np.uint8), mode="L")
    img     = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    resized = np.array(img, dtype=np.float32) / 255.0
    img_u8  = (resized * 255).astype(np.uint8)
    stat    = _extract_statistical_features(resized)
    glcm    = _extract_glcm_features(img_u8)
    return np.concatenate([stat, glcm]) if glcm.size > 0 else stat


def load_dataset_features(dataset_path: Path):
    """
    Extract texture features and BV/TV binary labels from one independent dataset.
    Returns:
        X        — (N, n_features) float32 texture matrix (raw, unscaled)
        y_binary — (N,) int32 binary BV/TV label (median split)
    """
    dataset_dir = dataset_path / "dataset"
    if not dataset_dir.exists():
        dataset_dir = dataset_path   # fallback: samples directly in path

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

        if mask_path.exists():
            mask = tiff.imread(str(mask_path))
            bvtv = float((mask > 0).mean())
        else:
            bvtv = float((vol > 0.5).mean())   # fallback threshold

        met_path = d / "metrics.json"
        if met_path.exists():
            with open(met_path) as f:
                met = json.load(f)
            bvtv = float(met.get("morphometrics", {}).get("BVTV", bvtv))

        X_rows.append(feat)
        bvtv_vals.append(bvtv)

    if len(X_rows) == 0:
        raise RuntimeError(f"No valid samples found in {dataset_path}")

    X        = np.array(X_rows, dtype=np.float32)
    bvtv_arr = np.array(bvtv_vals, dtype=np.float32)
    y_binary = (bvtv_arr > float(np.median(bvtv_arr))).astype(np.int32)
    return X, y_binary


# ── Quantum kernel helpers ────────────────────────────────────────────────────

def effective_rank(K):
    """Effective rank = (sum(lambda))^2 / sum(lambda^2)."""
    eigvals = np.maximum(np.linalg.eigvalsh(K), 0)
    s = eigvals.sum()
    return 1.0 if s == 0 else float(s ** 2 / (eigvals ** 2).sum())


def compute_kernel_matrix(X, feature_map):
    """Exact kernel matrix via statevector inner products (noiseless)."""
    n      = len(X)
    K      = np.eye(n)
    params = list(feature_map.parameters)
    for i in range(n):
        sv_i = Statevector.from_instruction(
            feature_map.assign_parameters(dict(zip(params, X[i]))))
        for j in range(i + 1, n):
            sv_j = Statevector.from_instruction(
                feature_map.assign_parameters(dict(zip(params, X[j]))))
            K[i, j] = K[j, i] = float(abs(sv_i.inner(sv_j)) ** 2)
    return K


def svm_accuracy_split(K, y, train_frac=TRAIN_FRAC):
    """Single stratified 80/20 split accuracy with precomputed kernel."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - train_frac,
                                 random_state=SEED)
    tr, te = next(sss.split(K, y))
    clf = SVC(kernel='precomputed', C=1.0)
    clf.fit(K[np.ix_(tr, tr)], y[tr])
    return float(clf.score(K[np.ix_(te, tr)], y[te]))


# ── Phase 1: Pre-extract features from all 10 datasets ───────────────────────
print("=" * 65)
print("Scalability analysis — independent-dataset validation (v3)")
print("=" * 65)

dataset_paths = [DATASETS_BASE / f"dataset_{i:02d}" for i in range(N_DATASETS)]
for p in dataset_paths:
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found: {p}\n"
            f"Expected 10 datasets in: {DATASETS_BASE}"
        )

print(f"\nPre-extracting texture features from {N_DATASETS} independent datasets …\n")
all_X, all_y = [], []
for i, dp in enumerate(dataset_paths):
    print(f"  dataset_{i:02d} … ", end="", flush=True)
    t0   = time.time()
    X, y = load_dataset_features(dp)
    all_X.append(X)
    all_y.append(y)
    print(f"{X.shape[0]} samples, {X.shape[1]} features  "
          f"({time.time()-t0:.1f}s)  "
          f"BV/TV split: {y.sum()} high / {(y==0).sum()} low")

print(f"\nFeature extraction complete.  Starting qubit sweep …\n")

# ── Phase 2: Scalability sweep ───────────────────────────────────────────────
rng     = np.random.default_rng(SEED)
results = []

for n_q in QUBIT_COUNTS:
    print(f"{'─'*65}")
    print(f"  {n_q} qubits  (Hilbert dim = {2**n_q},  kernel size = {N_SUBSAMPLE}x{N_SUBSAMPLE})")
    print(f"{'─'*65}")

    fm    = ZZFeatureMap(feature_dimension=n_q, reps=N_REPS)
    depth = fm.decompose().depth()
    print(f"  ZZFeatureMap circuit depth (decomposed): {depth}\n")

    ds_accs, ds_er, ds_offdiag = [], [], []
    t_qubit_start = time.time()

    for ds_idx in range(N_DATASETS):
        X_raw = all_X[ds_idx]
        y_raw = all_y[ds_idx]

        # Random subsample of N_SUBSAMPLE from this dataset
        idx  = rng.choice(len(X_raw), N_SUBSAMPLE, replace=False)
        X_s  = X_raw[idx]
        y_s  = y_raw[idx]

        # Scale -> UMAP -> [0, pi]
        X_scaled  = RobustScaler().fit_transform(X_s)
        reducer   = UMAP(n_components=n_q, n_neighbors=UMAP_NEIGHBORS,
                         min_dist=UMAP_MIN_DIST, random_state=SEED)
        X_reduced = reducer.fit_transform(X_scaled)
        X_q       = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_reduced)

        # Kernel matrix (timing kernel step separately for diagnostics)
        tk = time.time()
        K  = compute_kernel_matrix(X_q, fm)
        kernel_time = time.time() - tk

        # Metrics
        er       = effective_rank(K)
        off_diag = float((K.sum() - np.trace(K)) / (N_SUBSAMPLE * (N_SUBSAMPLE - 1)))
        acc      = svm_accuracy_split(K, y_s)

        ds_accs.append(acc)
        ds_er.append(er)
        ds_offdiag.append(off_diag)

        elapsed = (time.time() - t_qubit_start) / 60
        print(f"    ds_{ds_idx:02d}  acc={acc:.3f}  "
              f"eff_rank={er:.1f}/{N_SUBSAMPLE} ({er/N_SUBSAMPLE:.3f})  "
              f"off_diag={off_diag:.4f}  "
              f"kernel={kernel_time:.0f}s  total={elapsed:.1f}min")

    mean_acc = float(np.mean(ds_accs))
    std_acc  = float(np.std(ds_accs))
    mean_er  = float(np.mean(ds_er))
    mean_od  = float(np.mean(ds_offdiag))
    total_t  = (time.time() - t_qubit_start) / 60

    results.append(dict(
        n_qubits    = n_q,
        hilbert_dim = 2 ** n_q,
        depth       = depth,
        eff_rank    = mean_er,
        ratio       = mean_er / N_SUBSAMPLE,
        off_diag    = mean_od,
        acc         = mean_acc,
        std         = std_acc,
        all_accs    = ds_accs,
        time_min    = total_t,
    ))
    print(f"\n  SUMMARY {n_q}q:  acc = {mean_acc:.3f} +/- {std_acc:.3f}  "
          f"mean_eff_rank = {mean_er:.1f}  "
          f"total = {total_t:.1f} min\n")

np.save('scalability_results_independent.npy', results)
print("Results saved -> scalability_results_independent.npy\n")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'Qubits':>7}  {'Hilbert':>9}  {'Depth':>6}  "
      f"{'Eff.Rank':>9}  {'Ratio':>7}  {'Off-diag':>9}  {'Accuracy':>14}")
print("─" * 75)
for r in results:
    print(f"{r['n_qubits']:>7}  {r['hilbert_dim']:>9}  {r['depth']:>6}  "
          f"{r['eff_rank']:>9.1f}  {r['ratio']:>7.3f}  "
          f"{r['off_diag']:>9.4f}  "
          f"{r['acc']:>6.3f} +/- {r['std']:.3f}")

# ── Figure ─────────────────────────────────────────────────────────────────────
qubits  = [r['n_qubits'] for r in results]
accs    = [r['acc']      for r in results]
stds    = [r['std']      for r in results]
ratios  = [r['ratio']    for r in results]
depths  = [r['depth']    for r in results]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

# Panel 1 — accuracy
ax = axes[0]
ax.errorbar(qubits, accs, yerr=stds, marker='o', color='steelblue',
            capsize=5, linewidth=1.8, markersize=7, zorder=3, label='Mean ± std')
ax.axhline(0.5, linestyle=':', color='black', linewidth=1, label='Chance')
for r in results:
    ax.scatter([r['n_qubits']] * N_DATASETS, r['all_accs'],
               alpha=0.25, s=20, color='steelblue', zorder=2)
ax.set_xlabel('Number of qubits', fontsize=11)
ax.set_ylabel('BV/TV classification accuracy', fontsize=11)
ax.set_title(f'Accuracy vs Qubit Count\n({N_DATASETS} independent datasets)', fontsize=10)
ax.set_xticks(qubits)
ax.legend(fontsize=9)
ax.set_ylim(0.3, 0.9)
ax.grid(True, alpha=0.3)

# Panel 2 — kernel concentration
ax = axes[1]
ax.plot(qubits, ratios, marker='s', color='coral', linewidth=1.8, markersize=7)
ax.axhline(1.0, linestyle='--', color='black', linewidth=1,
           label='Identity matrix (ratio=1)')
ax.set_xlabel('Number of qubits', fontsize=11)
ax.set_ylabel('Effective rank / N samples', fontsize=11)
ax.set_title('Kernel Concentration vs Qubit Count', fontsize=11)
ax.set_xticks(qubits)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3)

# Panel 3 — circuit depth
ax = axes[2]
ax.plot(qubits, depths, marker='^', color='seagreen', linewidth=1.8, markersize=7)
ax.set_xlabel('Number of qubits', fontsize=11)
ax.set_ylabel('Circuit depth (decomposed gates)', fontsize=11)
ax.set_title('Circuit Depth vs Qubit Count', fontsize=11)
ax.set_xticks(qubits)
ax.grid(True, alpha=0.3)

plt.tight_layout()
for ext in ('pdf', 'png'):
    plt.savefig(f'scalability_analysis.{ext}', dpi=150, bbox_inches='tight')
print("Saved: scalability_analysis.pdf / scalability_analysis.png")
plt.show()
