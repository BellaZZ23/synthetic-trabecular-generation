"""
prepare_analysis_data.py
─────────────────────────
Extracts the data files needed by noise_sensitivity_analysis.py and
scalability_analysis.py from an existing full_pipeline.py output directory.

Saves into the current working directory:
  umap_features_8d.npy       — UMAP features scaled to [0, π], shape (N, 8)
  texture_features_raw.npy   — raw 43-feature texture matrix, shape (N, 43)
  labels_bvtv.npy            — binary BV/TV label (median split), shape (N,)

Usage:
  python prepare_analysis_data.py --outdir output/full_pipeline

If your pipeline used --n-components other than 8, this script re-runs UMAP
with n_components=8 on the raw texture features automatically.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile as tiff
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import RobustScaler, MinMaxScaler

try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_GLCM = True
except ImportError:
    SKIMAGE_GLCM = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

LABEL_KEYS = ["BVTV", "TbTh_um_p50", "TbN_per_mm", "TbSp_um_p50"]


# ── Texture extraction (copied from full_pipeline.py) ─────────────────────────

def _extract_glcm_features(img_u8):
    if not SKIMAGE_GLCM:
        return np.array([])
    distances = [1, 3]
    angles    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    props     = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    img_q     = (img_u8 // 8).astype(np.uint8)
    glcm      = graycomatrix(img_q, distances=distances, angles=angles,
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


def _extract_texture_features(sl, image_size=64):
    img     = Image.fromarray((sl * 255).astype(np.uint8), mode="L")
    img     = img.resize((image_size, image_size), Image.BILINEAR)
    resized = np.array(img, dtype=np.float32) / 255.0
    img_u8  = (resized * 255).astype(np.uint8)
    stat    = _extract_statistical_features(resized)
    glcm    = _extract_glcm_features(img_u8)
    return np.concatenate([stat, glcm]) if glcm.size > 0 else stat


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="output/full_pipeline",
                        help="Path to full_pipeline.py output directory")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir      = Path(args.outdir)
    dataset_dir = outdir / "dataset"
    features_dir = outdir / "features"

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            f"Make sure --outdir points to your full_pipeline.py output folder."
        )

    # ── Step 1: Extract raw texture features from gray.tif files ─────────────
    print("Extracting raw texture features from gray.tif files …")
    X_rows, Y_rows, bvtv_vals = [], [], []

    sample_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    for d in sample_dirs:
        gray_path = d / "gray.tif"
        mask_path = d / "mask.tif"
        if not gray_path.exists():
            continue

        # Extract texture features from grayscale mid-slice
        vol  = tiff.imread(str(gray_path)).astype(np.float32)
        vmax = vol.max()
        if vmax > 0:
            vol /= vmax
        mid  = vol[vol.shape[0] // 2]
        feat = _extract_texture_features(mid, args.image_size)

        # Compute BV/TV from mask.tif if available; else estimate from gray
        if mask_path.exists():
            mask = tiff.imread(str(mask_path))
            bvtv = float((mask > 0).mean())
        else:
            # Fallback: threshold gray volume at midpoint
            bvtv = float((vol > 0.5).mean())

        # Load morphometrics from metrics.json if it exists
        met_path = d / "metrics.json"
        if met_path.exists():
            with open(met_path) as f:
                met = json.load(f)
            morph = met.get("morphometrics", {})
            bvtv  = float(morph.get("BVTV", bvtv))  # prefer measured value
            Y_rows.append([morph.get(k, 0.0) for k in LABEL_KEYS])
        else:
            Y_rows.append([bvtv, 0.0, 0.0, 0.0])  # only BV/TV available

        X_rows.append(feat)
        bvtv_vals.append(bvtv)

    if len(X_rows) == 0:
        raise RuntimeError(
            "No valid samples found. Each sample folder needs at least gray.tif."
        )

    X = np.array(X_rows, dtype=np.float32)
    Y = np.array(Y_rows, dtype=np.float32)
    bvtv_arr = np.array(bvtv_vals, dtype=np.float32)

    print(f"  Extracted {X.shape[0]} samples, {X.shape[1]} features each")

    # ── Step 2: Binary BV/TV label via median split ───────────────────────────
    median_bvtv = float(np.median(bvtv_arr))
    y_binary    = (bvtv_arr > median_bvtv).astype(np.int32)
    print(f"  BV/TV median split at {median_bvtv:.3f} → "
          f"{y_binary.sum()} high / {(1-y_binary).sum()} low")

    # ── Step 3: Scale raw features and run UMAP with 8 components ────────────
    print("\nRunning UMAP (8 components) for noise sensitivity script …")
    X_scaled = RobustScaler().fit_transform(X)

    # Check if a saved npz already has 8 components — use it if so
    umap_npz = features_dir / "umap_quantum_ready.npz"
    umap_8d  = None

    if umap_npz.exists():
        data = np.load(umap_npz, allow_pickle=True)
        Z_tr = data["Z_train"]
        Z_te = data["Z_test"]
        if Z_tr.shape[1] == 8:
            # Already 8 components — concatenate train+test in original order
            print(f"  Found existing 8-component UMAP in {umap_npz.name} — using it")
            umap_8d = np.vstack([Z_tr, Z_te])
            # Note: train/test split order may differ from X order;
            # safer to re-run UMAP on full X
            umap_8d = None  # force fresh run for consistent sample order
        else:
            print(f"  Existing UMAP has {Z_tr.shape[1]} components — re-running with 8")

    if umap_8d is None:
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn is required. Run: pip install umap-learn")
        reducer = umap.UMAP(n_components=8, n_neighbors=15,
                            min_dist=0.1, random_state=args.seed)
        Z_umap  = reducer.fit_transform(X_scaled)
        # Scale to [0, π] for quantum angle encoding
        umap_8d = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(Z_umap)
        print(f"  UMAP complete: {umap_8d.shape}")

    # ── Step 4: Save files ────────────────────────────────────────────────────
    np.save("umap_features_8d.npy",     umap_8d)
    np.save("texture_features_raw.npy", X)
    np.save("labels_bvtv.npy",          y_binary)

    print("\nSaved:")
    print(f"  umap_features_8d.npy       {umap_8d.shape}  (scaled to [0, π])")
    print(f"  texture_features_raw.npy   {X.shape}")
    print(f"  labels_bvtv.npy            {y_binary.shape}  "
          f"(1=high BV/TV, median={median_bvtv:.3f})")
    print("\nYou can now run:")
    print("  python scalability_analysis.py")
    print("  python noise_sensitivity_analysis.py")


if __name__ == "__main__":
    main()
