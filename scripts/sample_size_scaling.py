"""
sample_size_scaling.py
──────────────────────
Tests how QKSVM performance and kernel concentration change as training
set size grows, at fixed 8 qubits (the paper's setting).

Key insight: the ZZFeatureMap encodes into a 2^8 = 256-dimensional Hilbert
space. When N < 256, samples can be (nearly) orthogonal — the kernel
approaches identity and loses discriminative power. Once N > 256 samples
cannot all be orthogonal, so the kernel is forced to have off-diagonal
structure and accuracy should recover.

This experiment quantifies that transition.

For each sample size N:
  For each of 10 independent datasets:
    1. Extract 43 texture features from gray.tif mid-slices
    2. UMAP -> 8 dimensions (matches paper)
    3. Scale to [0, pi], compute 100x100 kernel (statevector)
    4. 80/20 stratified train/test split -> one accuracy value
  -> 10 independent values -> mean +/- std

Compute time estimate (N_QUBITS=8, statevector):
  Each dataset: ~5-8 min kernel + ~30s feature extraction
  7 sizes x 10 datasets = 70 runs x ~6 min = ~7 hours
  Run overnight.
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
DATASETS_BASE = Path("output/independent_test_v2")
N_DATASETS    = 10
N_QUBITS      = 8       # fixed — paper's setting
N_REPS        = 2       # ZZFeatureMap repetitions
HILBERT_DIM   = 2 ** N_QUBITS   # = 256; vertical reference line in figure
TRAIN_FRAC    = 0.8
SEED          = 42
IMAGE_SIZE    = 64

# Sample sizes to test — straddle the Hilbert dimension (256)
SAMPLE_SIZES  = [50, 100, 150, 200, 300, 400, 500]

UMAP_NEIGHBORS = 15
UMAP_MIN_DIST  = 0.1

# ── Texture feature extraction ────────────────────────────────────────────────
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
        if mask_path.exists():
            mask = tiff.imread(str(mask_path))
            bvtv = float((mask > 0).mean())
        else:
            bvtv = float((vol > 0.5).mean())
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
    eigvals = np.maximum(np.linalg.eigvalsh(K), 0)
    s = eigvals.sum()
    return 1.0 if s == 0 else float(s ** 2 / (eigvals ** 2).sum())


def compute_kernel_matrix(X, feature_map):
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


def svm_accuracy_split(K, y):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - TRAIN_FRAC,
                                 random_state=SEED)
    tr, te = next(sss.split(K, y))
    clf = SVC(kernel='precomputed', C=1.0)
    clf.fit(K[np.ix_(tr, tr)], y[tr])
    return float(clf.score(K[np.ix_(te, tr)], y[te]))


# ── Pre-load all datasets ─────────────────────────────────────────────────────
print("=" * 65)
print(f"Sample size scaling — {N_QUBITS} qubits fixed  (Hilbert dim = {HILBERT_DIM})")
print("=" * 65)

dataset_paths = [DATASETS_BASE / f"dataset_{i:02d}" for i in range(N_DATASETS)]
for p in dataset_paths:
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

print(f"\nPre-extracting features from {N_DATASETS} datasets …\n")
all_X, all_y = [], []
for i, dp in enumerate(dataset_paths):
    print(f"  dataset_{i:02d} … ", end="", flush=True)
    t0   = time.time()
    X, y = load_dataset_features(dp)
    all_X.append(X)
    all_y.append(y)
    print(f"{X.shape[0]} samples  ({time.time()-t0:.1f}s)")

print(f"\nFeature extraction complete. Starting sample size sweep …\n")

# ── Build feature map (fixed for all runs) ────────────────────────────────────
fm    = ZZFeatureMap(feature_dimension=N_QUBITS, reps=N_REPS)
depth = fm.decompose().depth()
print(f"ZZFeatureMap: {N_QUBITS} qubits, {N_REPS} reps, depth={depth}\n")

# ── Sample size sweep ─────────────────────────────────────────────────────────
rng     = np.random.default_rng(SEED)
results = []

for N in SAMPLE_SIZES:
    label = f"N={N:>3}  ({'< ' if N < HILBERT_DIM else '>=' if N >= HILBERT_DIM else '= '}Hilbert dim {HILBERT_DIM})"
    print(f"{'─'*65}")
    print(f"  {label}")
    print(f"{'─'*65}")

    ds_accs, ds_er, ds_offdiag = [], [], []
    t_size_start = time.time()

    for ds_idx in range(N_DATASETS):
        X_raw = all_X[ds_idx]
        y_raw = all_y[ds_idx]

        # Subsample N points
        idx  = rng.choice(len(X_raw), N, replace=False)
        X_s  = X_raw[idx]
        y_s  = y_raw[idx]

        # Scale -> UMAP(8) -> [0, pi]
        X_scaled  = RobustScaler().fit_transform(X_s)
        reducer   = UMAP(n_components=N_QUBITS, n_neighbors=min(UMAP_NEIGHBORS, N-1),
                         min_dist=UMAP_MIN_DIST, random_state=SEED)
        X_reduced = reducer.fit_transform(X_scaled)
        X_q       = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_reduced)

        # Kernel
        tk = time.time()
        K  = compute_kernel_matrix(X_q, fm)
        kernel_time = time.time() - tk

        er       = effective_rank(K)
        off_diag = float((K.sum() - np.trace(K)) / (N * (N - 1)))
        acc      = svm_accuracy_split(K, y_s)

        ds_accs.append(acc)
        ds_er.append(er)
        ds_offdiag.append(off_diag)

        elapsed = (time.time() - t_size_start) / 60
        print(f"    ds_{ds_idx:02d}  acc={acc:.3f}  "
              f"eff_rank={er:.1f}/{N} ({er/N:.3f})  "
              f"off_diag={off_diag:.4f}  "
              f"kernel={kernel_time:.0f}s  total={elapsed:.1f}min")

    mean_acc = float(np.mean(ds_accs))
    std_acc  = float(np.std(ds_accs))
    mean_er  = float(np.mean(ds_er))
    mean_od  = float(np.mean(ds_offdiag))
    total_t  = (time.time() - t_size_start) / 60

    results.append(dict(
        N        = N,
        eff_rank = mean_er,
        ratio    = mean_er / N,
        off_diag = mean_od,
        acc      = mean_acc,
        std      = std_acc,
        all_accs = ds_accs,
        time_min = total_t,
    ))
    print(f"\n  SUMMARY N={N}:  acc = {mean_acc:.3f} +/- {std_acc:.3f}  "
          f"mean_eff_rank = {mean_er:.1f}  total = {total_t:.1f} min\n")

np.save('sample_size_results.npy', results)
print("Results saved -> sample_size_results.npy\n")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'N':>6}  {'vs dim':>8}  {'Eff.Rank':>9}  "
      f"{'Ratio':>7}  {'Off-diag':>9}  {'Accuracy':>14}")
print("─" * 65)
for r in results:
    cmp = "< dim" if r['N'] < HILBERT_DIM else ">= dim"
    print(f"{r['N']:>6}  {cmp:>8}  {r['eff_rank']:>9.1f}  "
          f"{r['ratio']:>7.3f}  {r['off_diag']:>9.4f}  "
          f"{r['acc']:>6.3f} +/- {r['std']:.3f}")

# ── Figure ─────────────────────────────────────────────────────────────────────
Ns      = [r['N']       for r in results]
accs    = [r['acc']     for r in results]
stds    = [r['std']     for r in results]
ratios  = [r['ratio']   for r in results]
offdiag = [r['off_diag'] for r in results]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel 1 — accuracy vs N
ax = axes[0]
ax.errorbar(Ns, accs, yerr=stds, marker='o', color='steelblue',
            capsize=5, linewidth=1.8, markersize=7, zorder=3, label='Mean ± std')
ax.axhline(0.5, linestyle=':', color='black', linewidth=1, label='Chance')
ax.axvline(HILBERT_DIM, linestyle='--', color='crimson', linewidth=1.5,
           label=f'Hilbert dim = {HILBERT_DIM}')
for r in results:
    ax.scatter([r['N']] * N_DATASETS, r['all_accs'],
               alpha=0.25, s=20, color='steelblue', zorder=2)
ax.set_xlabel('Training set size N', fontsize=11)
ax.set_ylabel('BV/TV classification accuracy', fontsize=11)
ax.set_title(f'Accuracy vs Sample Size\n({N_QUBITS} qubits, {N_DATASETS} independent datasets)',
             fontsize=10)
ax.legend(fontsize=9)
ax.set_ylim(0.3, 0.9)
ax.grid(True, alpha=0.3)

# Panel 2 — kernel concentration ratio vs N
ax = axes[1]
ax.plot(Ns, ratios, marker='s', color='coral', linewidth=1.8, markersize=7)
ax.axhline(1.0, linestyle='--', color='black', linewidth=1, label='Identity limit')
ax.axvline(HILBERT_DIM, linestyle='--', color='crimson', linewidth=1.5,
           label=f'Hilbert dim = {HILBERT_DIM}')
ax.set_xlabel('Training set size N', fontsize=11)
ax.set_ylabel('Effective rank / N', fontsize=11)
ax.set_title('Kernel Concentration vs Sample Size', fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3)

# Panel 3 — off-diagonal mean vs N (log scale)
ax = axes[2]
ax.semilogy(Ns, offdiag, marker='^', color='seagreen', linewidth=1.8, markersize=7)
ax.axvline(HILBERT_DIM, linestyle='--', color='crimson', linewidth=1.5,
           label=f'Hilbert dim = {HILBERT_DIM}')
ax.set_xlabel('Training set size N', fontsize=11)
ax.set_ylabel('Mean off-diagonal kernel value', fontsize=11)
ax.set_title('Kernel Off-Diagonal Density vs Sample Size', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

plt.suptitle(f'UMAP Quantum Kernel ({N_QUBITS}q, ZZFeatureMap reps={N_REPS}) — '
             f'Sample Size Scaling', fontsize=11, y=1.01)
plt.tight_layout()
for ext in ('pdf', 'png'):
    plt.savefig(f'sample_size_scaling.{ext}', dpi=150, bbox_inches='tight')
print("Saved: sample_size_scaling.pdf / sample_size_scaling.png")
plt.show()
