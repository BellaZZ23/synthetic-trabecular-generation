#!/usr/bin/env python3
"""
dim_reduction_pipeline_v2.py

v2: Fair comparison between PCA, RP, PLS, and UMAP using shared downstream evaluation.

Changes from v1:
  - Added PLSRegression (supervised baseline)
  - Added UMAP (best nonlinear method per survey literature)
  - Added downstream Ridge regression scoring for all methods (mean R², mean MAE)
  - Added RobustScaler option (--scaler robust)
  - Fixed embedding plot titles to show method name
  - All methods now compared on same downstream task
  - Added --no-pls and --no-umap flags

Usage:
  python dim_reduction_pipeline_v2.py `
      --dataset-dir output\final_dataset_v6\dataset `
      --outdir output\final_dataset_v6\features_v2 `
      --n-components 16 `
      --feature-type texture `
      --scaler robust `
      --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile as tiff
from scipy import ndimage as ndi
from scipy.stats import skew, kurtosis

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("WARNING: umap-learn not installed. Run: pip install umap-learn")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_GLCM = True
except ImportError:
    SKIMAGE_GLCM = False
    print("WARNING: skimage GLCM unavailable")

LABEL_KEYS = ["BVTV", "TbTh_um_p50", "TbN_per_mm", "TbSp_um_p50"]


# ──────────────────────────────────────────────────────────
#  DATA LOADING
# ──────────────────────────────────────────────────────────

def discover_samples(dataset_dir: Path) -> list[dict]:
    samples = []
    for d in sorted(dataset_dir.iterdir()):
        metrics_path = d / "metrics.json"
        if not d.is_dir() or not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        gray_path = d / "gray.tif"
        mask_path = d / "mask.tif"
        if not gray_path.exists() and not mask_path.exists():
            continue
        samples.append({
            "name": d.name, "dir": d, "metrics": metrics,
            "gray_path": gray_path if gray_path.exists() else None,
            "mask_path": mask_path if mask_path.exists() else None,
        })
    print(f"Discovered {len(samples)} samples in {dataset_dir}")
    return samples


def load_volume(path: Path) -> np.ndarray:
    vol = tiff.imread(str(path)).astype(np.float32)
    vmax = vol.max()
    if vmax > 0: vol /= vmax
    return vol


def extract_slices(vol: np.ndarray, mode: str = "mid", n_slices: int = 5) -> list[np.ndarray]:
    Z = vol.shape[0]
    if mode == "mid":    return [vol[Z // 2]]
    elif mode == "multi": return [vol[i] for i in np.linspace(0, Z-1, n_slices, dtype=int)]
    elif mode == "all":   return [vol[i] for i in range(Z)]
    elif mode == "mip":   return [vol.max(axis=0)]
    else: raise ValueError(f"Unknown slice mode: {mode}")


def resize_slice(s: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray((s * 255).astype(np.uint8), mode="L")
    return np.array(img.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


# ──────────────────────────────────────────────────────────
#  TEXTURE FEATURES
# ──────────────────────────────────────────────────────────

def extract_glcm_features(img_u8: np.ndarray) -> np.ndarray:
    if not SKIMAGE_GLCM: return np.array([])
    img_q = (img_u8 // 8).astype(np.uint8)
    glcm  = graycomatrix(img_q, distances=[1, 3],
                         angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         levels=32, symmetric=True, normed=True)
    feats = []
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        vals = graycoprops(glcm, prop)
        feats.extend([float(vals.mean()), float(vals.std())])
    return np.array(feats, dtype=np.float32)


def extract_statistical_features(img: np.ndarray) -> np.ndarray:
    feats = [float(img.mean()), float(img.std()), float(np.median(img)),
             float(skew(img.ravel())), float(kurtosis(img.ravel())),
             float(img.min()), float(img.max()),
             float(np.percentile(img, 25)), float(np.percentile(img, 75))]
    hist, _ = np.histogram(img.ravel(), bins=16, range=(0, 1))
    feats.extend((hist.astype(np.float32) / (hist.sum() + 1e-8)).tolist())
    gx = ndi.sobel(img, axis=1); gy = ndi.sobel(img, axis=0)
    gmag = np.sqrt(gx**2 + gy**2)
    feats += [float(gmag.mean()), float(gmag.std()),
              float(np.percentile(gmag, 75)), float(np.percentile(gmag, 95))]
    lv = ndi.uniform_filter(img**2, size=5) - ndi.uniform_filter(img, size=5)**2
    feats += [float(lv.mean()), float(lv.std())]
    return np.array(feats, dtype=np.float32)


def extract_texture_features(sl: np.ndarray, image_size: int = 64) -> np.ndarray:
    resized = resize_slice(sl, image_size)
    stat    = extract_statistical_features(resized)
    glcm    = extract_glcm_features((resized * 255).astype(np.uint8))
    return np.concatenate([stat, glcm]) if glcm.size > 0 else stat


# ──────────────────────────────────────────────────────────
#  FEATURE MATRIX
# ──────────────────────────────────────────────────────────

def extract_labels(metrics: dict) -> dict:
    morph = metrics.get("morphometrics", {})
    return {k: float(morph[k]) for k in LABEL_KEYS if morph.get(k) is not None}


def extract_generator_params(metrics: dict) -> dict:
    params = metrics.get("params", {}); ridge = params.get("ridge", {})
    targets = metrics.get("targets", {})
    return {"bvtv_target": targets.get("bvtv_target"),
            "tbth_um_target": targets.get("tbth_um_target"),
            "base_sigma": ridge.get("base_sigma"),
            "seed": metrics.get("seed")}


def build_feature_matrix(samples, use_gray=True, slice_mode="mid",
                          n_slices=5, image_size=64, feature_type="texture"):
    X_rows, Y_rows, info, gen_params = [], [], [], []
    for s in samples:
        path = s["gray_path"] if (use_gray and s["gray_path"]) else s["mask_path"]
        if path is None: continue
        vol    = load_volume(path)
        slices = extract_slices(vol, mode=slice_mode, n_slices=n_slices)
        labels = extract_labels(s["metrics"])
        params = extract_generator_params(s["metrics"])
        if len(labels) < len(LABEL_KEYS): continue
        for si, sl in enumerate(slices):
            feat = extract_texture_features(sl, image_size) if feature_type == "texture" \
                   else resize_slice(sl, image_size).ravel()
            X_rows.append(feat)
            Y_rows.append([labels[k] for k in LABEL_KEYS])
            info.append({"sample": s["name"], "slice": si, "dir": str(s["dir"])})
            gen_params.append(params)

    X = np.array(X_rows, dtype=np.float32)
    Y = np.array(Y_rows, dtype=np.float32)
    print(f"Feature matrix: X={X.shape}, Y={Y.shape}")
    print(f"  Feature type: {feature_type}")
    print(f"  Texture features per image: {X.shape[1]}")
    print(f"  Labels: {LABEL_KEYS}")
    return X, Y, info, gen_params


# ──────────────────────────────────────────────────────────
#  SCALER + DOWNSTREAM EVALUATION
# ──────────────────────────────────────────────────────────

def make_scaler(name: str):
    """Return StandardScaler or RobustScaler."""
    if name == "robust":
        return RobustScaler()
    return StandardScaler()


def downstream_regression_score(
    Z_train: np.ndarray, Z_test: np.ndarray,
    Y_train: np.ndarray, Y_test: np.ndarray,
    label_names: list[str], alpha: float = 1.0,
) -> dict:
    """
    Train Ridge regressors on latent features and evaluate on test set.
    Provides a fair comparison across PCA, RP, and PLS on the same task.
    """
    per_label = {}; r2_vals = []; mae_vals = []
    for j, label in enumerate(label_names):
        reg  = Ridge(alpha=alpha)
        reg.fit(Z_train, Y_train[:, j])
        pred = reg.predict(Z_test)
        r2   = float(r2_score(Y_test[:, j], pred))
        mae  = float(mean_absolute_error(Y_test[:, j], pred))
        per_label[label] = {"r2": r2, "mae": mae}
        r2_vals.append(r2); mae_vals.append(mae)
    return {"per_label": per_label,
            "mean_r2":  float(np.mean(r2_vals)),
            "mean_mae": float(np.mean(mae_vals))}


# ──────────────────────────────────────────────────────────
#  DIMENSIONALITY REDUCTION METHODS
# ──────────────────────────────────────────────────────────

def run_pca(X_train, X_test, Y_train, Y_test,
            n_components, label_names, scaler_name="standard", seed=42):
    scaler    = make_scaler(scaler_name)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    n_comp    = min(n_components, X_train_s.shape[0], X_train_s.shape[1])

    pca      = PCA(n_components=n_comp, random_state=seed)
    Z_train  = pca.fit_transform(X_train_s)
    Z_test   = pca.transform(X_test_s)
    explained  = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    recon_mse  = float(np.mean((X_train_s - pca.inverse_transform(Z_train))**2))

    downstream = downstream_regression_score(Z_train, Z_test, Y_train, Y_test, label_names)

    print(f"\nPCA: {n_comp} components")
    print(f"  Scaler: {scaler_name}")
    print(f"  Variance explained: {cumulative[-1]:.4f} ({cumulative[-1]*100:.1f}%)")
    print(f"  Top 5 components: {explained[:5]}")
    print(f"  Reconstruction MSE: {recon_mse:.6f}")
    print(f"  Downstream mean R²:  {downstream['mean_r2']:.4f}")
    print(f"  Downstream mean MAE: {downstream['mean_mae']:.4f}")
    for k, v in downstream["per_label"].items():
        print(f"    {k}: R²={v['r2']:.3f}  MAE={v['mae']:.4f}")

    return {"method": "PCA", "model": pca, "scaler": scaler,
            "Z_train": Z_train, "Z_test": Z_test,
            "n_components": n_comp,
            "explained_variance_ratio": explained.tolist(),
            "cumulative_variance": cumulative.tolist(),
            "reconstruction_mse": recon_mse,
            "downstream_metrics": downstream}


def run_random_projection(X_train, X_test, Y_train, Y_test,
                           n_components, label_names,
                           method="gaussian", scaler_name="standard", seed=42):
    scaler    = make_scaler(scaler_name)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    n_comp    = min(n_components, X_train_s.shape[1])

    rp = (GaussianRandomProjection(n_components=n_comp, random_state=seed)
          if method == "gaussian"
          else SparseRandomProjection(n_components=n_comp, random_state=seed))

    Z_train = rp.fit_transform(X_train_s)
    Z_test  = rp.transform(X_test_s)

    # Distance preservation check
    n_chk  = min(200, X_train_s.shape[0])
    idx    = np.random.default_rng(seed).choice(X_train_s.shape[0], n_chk, replace=False)
    D_orig = np.linalg.norm(X_train_s[idx, None, :] - X_train_s[None, idx, :], axis=-1)
    D_proj = np.linalg.norm(Z_train[idx, None, :] - Z_train[None, idx, :], axis=-1)
    mask   = D_orig > 0
    ratios = D_proj[mask] / D_orig[mask] if mask.any() else np.array([1.0])
    dist_preservation = {"mean_ratio": float(np.mean(ratios)),
                         "std_ratio":  float(np.std(ratios))}

    downstream = downstream_regression_score(Z_train, Z_test, Y_train, Y_test, label_names)

    print(f"\nRandom Projection ({method}): {n_comp} components")
    print(f"  Scaler: {scaler_name}")
    print(f"  Distance preservation: mean={dist_preservation['mean_ratio']:.4f} "
          f"std={dist_preservation['std_ratio']:.4f}")
    print(f"  Downstream mean R²:  {downstream['mean_r2']:.4f}")
    print(f"  Downstream mean MAE: {downstream['mean_mae']:.4f}")
    for k, v in downstream["per_label"].items():
        print(f"    {k}: R²={v['r2']:.3f}  MAE={v['mae']:.4f}")

    return {"method": f"RP_{method}", "model": rp, "scaler": scaler,
            "Z_train": Z_train, "Z_test": Z_test,
            "n_components": n_comp,
            "distance_preservation": dist_preservation,
            "downstream_metrics": downstream}


def run_pls(X_train, X_test, Y_train, Y_test,
            n_components, label_names, scaler_name="standard"):
    """
    Supervised dimensionality reduction.
    PLS learns latent directions maximally covarying with morphometric labels.
    This is the supervised baseline — expected to outperform unsupervised methods.
    """
    scaler    = make_scaler(scaler_name)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    n_comp    = max(1, min(n_components, X_train_s.shape[0]-1,
                           X_train_s.shape[1], Y_train.shape[1]))

    pls     = PLSRegression(n_components=n_comp)
    Z_train = pls.fit_transform(X_train_s, Y_train)[0]
    Z_test  = pls.transform(X_test_s)

    downstream = downstream_regression_score(Z_train, Z_test, Y_train, Y_test, label_names)

    print(f"\nPLS: {n_comp} components")
    print(f"  Scaler: {scaler_name}")
    print(f"  Downstream mean R²:  {downstream['mean_r2']:.4f}")
    print(f"  Downstream mean MAE: {downstream['mean_mae']:.4f}")
    for k, v in downstream["per_label"].items():
        print(f"    {k}: R²={v['r2']:.3f}  MAE={v['mae']:.4f}")

    return {"method": "PLS", "model": pls, "scaler": scaler,
            "Z_train": Z_train, "Z_test": Z_test,
            "n_components": n_comp,
            "downstream_metrics": downstream}


# ──────────────────────────────────────────────────────────
#  LABEL CORRELATION ANALYSIS
# ──────────────────────────────────────────────────────────

def label_correlation_analysis(Z, Y, method_name, label_names):
    corr_info = {}
    print(f"\n{method_name} — Label correlations (top components):")
    for li, ln in enumerate(label_names):
        rs = [float(np.corrcoef(Z[:, ci], Y[:, li])[0, 1])
              for ci in range(Z.shape[1])]
        best_ci = int(np.argmax(np.abs(rs)))
        best_r  = rs[best_ci]
        corr_info[ln] = {"correlations": rs, "best_component": best_ci, "best_r": best_r}
        print(f"  {ln}: best PC{best_ci} r={best_r:.3f}")
    return corr_info


# ──────────────────────────────────────────────────────────
#  PLOTS
# ──────────────────────────────────────────────────────────

def plot_pca_variance(pca_result, outdir):
    ev = pca_result["explained_variance_ratio"]
    cv = pca_result["cumulative_variance"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.bar(range(len(ev)), ev, color="steelblue")
    a1.set_xlabel("PC"); a1.set_ylabel("Variance Ratio"); a1.set_title("PCA Variance per Component")
    a2.plot(cv, "o-", color="darkorange")
    a2.axhline(0.95, color="r", ls="--", alpha=0.5, label="95%")
    a2.set_xlabel("Components"); a2.set_ylabel("Cumulative Variance"); a2.legend()
    plt.tight_layout()
    plt.savefig(outdir / "pca_variance.png", dpi=150); plt.close()


def plot_correlation_heatmap(corr_info, outdir):
    labels = list(corr_info.keys())
    if not labels: return
    n_comp = len(list(corr_info.values())[0]["correlations"])
    mat    = np.array([[corr_info[ln]["correlations"][ci]
                        for ci in range(n_comp)] for ln in labels])
    method = list(corr_info.values())[0].get("method", "")
    fig, ax = plt.subplots(figsize=(max(8, n_comp * 0.6), 4))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Component Index"); ax.set_title(f"Label-Component Correlations")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fname = outdir / f"{'pca' if not method else method.lower()}_correlation_heatmap.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved: {fname}")


def plot_2d_embedding(Z, Y, label_idx, label_name, method_name, outdir):
    if Z.shape[1] < 2: return
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=Y[:, label_idx], cmap="viridis", s=30, alpha=0.7)
    plt.colorbar(sc, ax=ax).set_label(label_name)
    # FIX: use method_name in title instead of hardcoding "PCA"
    ax.set_title(f"{method_name}: colored by {label_name}")
    ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")
    plt.tight_layout()
    fname = outdir / f"{method_name.lower().replace(' ', '_')}_2d_{label_name}.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved: {fname}")


def plot_downstream_comparison(comparison, outdir):
    """Bar chart comparing downstream R² across all methods."""
    methods = list(comparison.keys())
    r2_vals = [comparison[m]["downstream_mean_r2"] for m in methods]
    colors  = ["steelblue" if "PCA" in m else
               "darkorange" if "RP" in m else
               "forestgreen" for m in methods]
    fig, ax = plt.subplots(figsize=(max(6, len(methods)*1.5), 5))
    bars = ax.bar(methods, r2_vals, color=colors, alpha=0.85)
    ax.set_ylabel("Mean R² (Ridge regression on test set)")
    ax.set_title("Downstream Morphometric Prediction — Method Comparison")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(outdir / "downstream_comparison.png", dpi=150); plt.close()
    print(f"  Saved: {outdir / 'downstream_comparison.png'}")


def plot_per_label_r2(comparison, outdir):
    """Per-label R² heatmap across methods."""
    methods = list(comparison.keys())
    mat = np.array([[comparison[m]["downstream_per_label"].get(k, {}).get("r2", 0)
                     for k in LABEL_KEYS] for m in methods])
    fig, ax = plt.subplots(figsize=(max(8, len(LABEL_KEYS)*2), max(4, len(methods)*0.8)))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(LABEL_KEYS))); ax.set_xticklabels(LABEL_KEYS, rotation=30, ha="right")
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
    ax.set_title("Per-Label R² by Method")
    plt.colorbar(im, ax=ax, label="R²")
    for i in range(len(methods)):
        for j in range(len(LABEL_KEYS)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / "per_label_r2_heatmap.png", dpi=150); plt.close()
    print(f"  Saved: {outdir / 'per_label_r2_heatmap.png'}")


# ──────────────────────────────────────────────────────────
#  QUANTUM EXPORT
# ──────────────────────────────────────────────────────────

def export_for_quantum(Z_train, Z_test, Y_train, Y_test,
                       method_name, info_train, info_test, gp_train, outdir):
    mm        = MinMaxScaler(feature_range=(0, np.pi))
    Z_train_q = mm.fit_transform(Z_train); Z_test_q = mm.transform(Z_test)
    mm01      = MinMaxScaler(feature_range=(0, 1))
    Z_01_tr   = mm01.fit_transform(Z_train); Z_01_te = mm01.transform(Z_test)

    fname = outdir / f"{method_name.lower().replace(' ', '_')}_quantum_ready.npz"
    np.savez(fname, Z_train=Z_train_q, Z_test=Z_test_q,
             Z_train_01=Z_01_tr, Z_test_01=Z_01_te,
             Y_train=Y_train, Y_test=Y_test,
             label_names=LABEL_KEYS, n_features=Z_train_q.shape[1])
    print(f"  Quantum-ready features saved: {fname}")
    print(f"    Train: {Z_train_q.shape}, Test: {Z_test_q.shape}")
    print(f"    Feature range: [0, pi] for angle encoding")
    return {"file": str(fname), "n_qubits_needed": int(Z_train_q.shape[1]),
            "encoding": "angle", "feature_range": "[0, pi]"}


# ──────────────────────────────────────────────────────────
#  COMPARISON SUMMARY
# ──────────────────────────────────────────────────────────


def run_umap(X_train, X_test, Y_train, Y_test,
             n_components, label_names,
             scaler_name="standard", seed=42,
             n_neighbors=15, min_dist=0.1):
    """
    UMAP: best nonlinear method for preserving both local and global structure.
    Per survey literature, consistently outperforms PCA and t-SNE for downstream tasks.
    Unlike t-SNE, UMAP can transform new test points making it suitable for QSVM.
    """
    if not UMAP_AVAILABLE:
        print("  UMAP skipped — install with: pip install umap-learn")
        return None

    scaler    = make_scaler(scaler_name)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    n_comp    = min(n_components, X_train_s.shape[1])

    reducer = umap.UMAP(n_components=n_comp, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=seed, transform_seed=seed)
    Z_train = reducer.fit_transform(X_train_s)
    Z_test  = reducer.transform(X_test_s)

    downstream = downstream_regression_score(Z_train, Z_test, Y_train, Y_test, label_names)

    print(f"\nUMAP: {n_comp} components (n_neighbors={n_neighbors}, min_dist={min_dist})")
    print(f"  Scaler: {scaler_name}")
    print(f"  Downstream mean R²:  {downstream['mean_r2']:.4f}")
    print(f"  Downstream mean MAE: {downstream['mean_mae']:.4f}")
    for k, v in downstream["per_label"].items():
        print(f"    {k}: R²={v['r2']:.3f}  MAE={v['mae']:.4f}")

    return {"method": "UMAP", "model": reducer, "scaler": scaler,
            "Z_train": Z_train, "Z_test": Z_test,
            "n_components": n_comp,
            "n_neighbors": n_neighbors, "min_dist": min_dist,
            "downstream_metrics": downstream}

def compare_methods(pca_result, rp_results, pls_result=None, umap_result=None) -> dict:
    c = {
        "PCA": {
            "n_components": pca_result["n_components"],
            "variance_explained": pca_result["cumulative_variance"][-1],
            "reconstruction_mse": pca_result["reconstruction_mse"],
            "downstream_mean_r2":  pca_result["downstream_metrics"]["mean_r2"],
            "downstream_mean_mae": pca_result["downstream_metrics"]["mean_mae"],
            "downstream_per_label": pca_result["downstream_metrics"]["per_label"],
        }
    }
    for rp in rp_results:
        c[rp["method"]] = {
            "n_components": rp["n_components"],
            "distance_preservation_mean": rp["distance_preservation"]["mean_ratio"],
            "distance_preservation_std":  rp["distance_preservation"]["std_ratio"],
            "downstream_mean_r2":  rp["downstream_metrics"]["mean_r2"],
            "downstream_mean_mae": rp["downstream_metrics"]["mean_mae"],
            "downstream_per_label": rp["downstream_metrics"]["per_label"],
        }
    if pls_result is not None:
        c["PLS"] = {
            "n_components": pls_result["n_components"],
            "downstream_mean_r2":  pls_result["downstream_metrics"]["mean_r2"],
            "downstream_mean_mae": pls_result["downstream_metrics"]["mean_mae"],
            "downstream_per_label": pls_result["downstream_metrics"]["per_label"],
        }
    if umap_result is not None:
        c["UMAP"] = {
            "n_components": umap_result["n_components"],
            "n_neighbors": umap_result["n_neighbors"],
            "min_dist": umap_result["min_dist"],
            "downstream_mean_r2":  umap_result["downstream_metrics"]["mean_r2"],
            "downstream_mean_mae": umap_result["downstream_metrics"]["mean_mae"],
            "downstream_per_label": umap_result["downstream_metrics"]["per_label"],
        }
    return c


# ──────────────────────────────────────────────────────────
#  CLI + MAIN
# ──────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="Dim reduction pipeline v2 — fair downstream comparison")
    p.add_argument("--dataset-dir",  type=str, required=True)
    p.add_argument("--outdir",       type=str, default="output/features_v2")
    p.add_argument("--n-components", type=int, default=16)
    p.add_argument("--slice-mode",   type=str, default="mid",
                   choices=["mid", "multi", "all", "mip"])
    p.add_argument("--n-slices",     type=int, default=5)
    p.add_argument("--image-size",   type=int, default=64)
    p.add_argument("--use-mask",     action="store_true")
    p.add_argument("--test-split",   type=float, default=0.2)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--rp-method",    type=str, default="both",
                   choices=["gaussian", "sparse", "both"])
    p.add_argument("--feature-type", type=str, default="texture",
                   choices=["pixels", "texture"])
    p.add_argument("--scaler",       type=str, default="standard",
                   choices=["standard", "robust"],
                   help="Feature scaling: standard (Z-score) or robust (median/IQR)")
    p.add_argument("--no-pls",        action="store_true",
                   help="Disable supervised PLS baseline")
    p.add_argument("--no-umap",       action="store_true",
                   help="Disable UMAP (use if umap-learn not installed)")
    p.add_argument("--umap-neighbors",type=int, default=15,
                   help="UMAP n_neighbors (controls local vs global balance)")
    p.add_argument("--umap-min-dist", type=float, default=0.1,
                   help="UMAP min_dist (controls point clustering tightness)")
    return p


def main():
    args   = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DIMENSIONALITY REDUCTION PIPELINE v2")
    print(f"  Feature type: {args.feature_type}")
    print(f"  Scaler: {args.scaler}")
    print(f"  PLS enabled: {not args.no_pls}")
    print("=" * 60)

    samples = discover_samples(Path(args.dataset_dir))
    if not samples:
        raise FileNotFoundError(f"No samples in {args.dataset_dir}")

    X, Y, info, gen_params = build_feature_matrix(
        samples, use_gray=not args.use_mask,
        slice_mode=args.slice_mode, n_slices=args.n_slices,
        image_size=args.image_size, feature_type=args.feature_type)

    if X.shape[0] < 4:
        print(f"WARNING: Only {X.shape[0]} samples — too few"); return

    # Grouped split by sample name (prevents slice leakage)
    sample_names   = np.array([i["sample"] for i in info])
    unique_samples = np.unique(sample_names)
    rng_split      = np.random.default_rng(args.seed)
    shuffled       = rng_split.permutation(unique_samples)
    n_test         = max(1, int(len(shuffled) * args.test_split))
    test_set       = set(shuffled[:n_test])

    train_mask = np.array([sample_names[i] not in test_set for i in range(len(info))])
    test_mask  = ~train_mask
    X_train, X_test = X[train_mask], X[test_mask]
    Y_train, Y_test = Y[train_mask], Y[test_mask]
    idx_train = np.where(train_mask)[0]; idx_test = np.where(test_mask)[0]
    info_train = [info[i] for i in idx_train]; info_test = [info[i] for i in idx_test]
    gp_train   = [gen_params[i] for i in idx_train]

    print(f"  Grouped split: {len(unique_samples)-n_test} train, {n_test} test volumes")
    print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # ── PCA ──
    print("\n" + "─"*40 + "\n  PCA\n" + "─"*40)
    pca_result = run_pca(X_train, X_test, Y_train, Y_test,
                         args.n_components, LABEL_KEYS,
                         scaler_name=args.scaler, seed=args.seed)
    pca_corr = label_correlation_analysis(pca_result["Z_train"], Y_train, "PCA", LABEL_KEYS)
    plot_pca_variance(pca_result, outdir)
    plot_correlation_heatmap(pca_corr, outdir)
    for li, ln in enumerate(LABEL_KEYS):
        plot_2d_embedding(pca_result["Z_train"], Y_train, li, ln, "PCA", outdir)
    export_for_quantum(pca_result["Z_train"], pca_result["Z_test"],
                       Y_train, Y_test, "PCA", info_train, info_test, gp_train, outdir)

    # ── Random Projection ──
    rp_results = []
    rp_methods = ["gaussian", "sparse"] if args.rp_method == "both" else [args.rp_method]
    for rp_method in rp_methods:
        print("\n" + "─"*40 + f"\n  RANDOM PROJECTION ({rp_method})\n" + "─"*40)
        rp_result = run_random_projection(X_train, X_test, Y_train, Y_test,
                                          args.n_components, LABEL_KEYS,
                                          method=rp_method,
                                          scaler_name=args.scaler, seed=args.seed)
        rp_results.append(rp_result)
        rp_corr = label_correlation_analysis(rp_result["Z_train"], Y_train,
                                              f"RP_{rp_method}", LABEL_KEYS)
        plot_correlation_heatmap(rp_corr, outdir)
        for li, ln in enumerate(LABEL_KEYS):
            plot_2d_embedding(rp_result["Z_train"], Y_train, li, ln,
                              f"RP_{rp_method}", outdir)
        export_for_quantum(rp_result["Z_train"], rp_result["Z_test"],
                           Y_train, Y_test, f"RP_{rp_method}",
                           info_train, info_test, gp_train, outdir)

    # ── PLS ──
    pls_result = None
    if not args.no_pls:
        print("\n" + "─"*40 + "\n  PLS (supervised baseline)\n" + "─"*40)
        pls_result = run_pls(X_train, X_test, Y_train, Y_test,
                             args.n_components, LABEL_KEYS,
                             scaler_name=args.scaler)
        pls_corr = label_correlation_analysis(pls_result["Z_train"], Y_train, "PLS", LABEL_KEYS)
        plot_correlation_heatmap(pls_corr, outdir)
        for li, ln in enumerate(LABEL_KEYS):
            plot_2d_embedding(pls_result["Z_train"], Y_train, li, ln, "PLS", outdir)
        export_for_quantum(pls_result["Z_train"], pls_result["Z_test"],
                           Y_train, Y_test, "PLS",
                           info_train, info_test, gp_train, outdir)

    # ── UMAP ──
    umap_result = None
    if not args.no_umap:
        print("\n" + "─"*40 + "\n  UMAP (nonlinear, local+global)\n" + "─"*40)
        umap_result = run_umap(X_train, X_test, Y_train, Y_test,
                               args.n_components, LABEL_KEYS,
                               scaler_name=args.scaler, seed=args.seed,
                               n_neighbors=args.umap_neighbors,
                               min_dist=args.umap_min_dist)
        if umap_result is not None:
            umap_corr = label_correlation_analysis(umap_result["Z_train"], Y_train,
                                                   "UMAP", LABEL_KEYS)
            plot_correlation_heatmap(umap_corr, outdir)
            for li, ln in enumerate(LABEL_KEYS):
                plot_2d_embedding(umap_result["Z_train"], Y_train, li, ln, "UMAP", outdir)
            export_for_quantum(umap_result["Z_train"], umap_result["Z_test"],
                               Y_train, Y_test, "UMAP",
                               info_train, info_test, gp_train, outdir)

    # ── Comparison ──
    comparison = compare_methods(pca_result, rp_results, pls_result=pls_result,
                                 umap_result=umap_result)
    plot_downstream_comparison(comparison, outdir)
    plot_per_label_r2(comparison, outdir)

    summary = {
        "pipeline_version": "2.0",
        "feature_type": args.feature_type,
        "scaler": args.scaler,
        "dataset_dir": str(args.dataset_dir),
        "n_samples": len(samples),
        "n_images": X.shape[0],
        "image_size": args.image_size,
        "raw_features": X.shape[1],
        "n_components": args.n_components,
        "slice_mode": args.slice_mode,
        "comparison": comparison,
        "label_keys": LABEL_KEYS,
    }
    with open(outdir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "="*60)
    print("  SUMMARY — DOWNSTREAM PREDICTION (mean R²)")
    print("="*60)
    winner = max(comparison, key=lambda m: comparison[m]["downstream_mean_r2"])
    for method, info_m in comparison.items():
        r2  = info_m["downstream_mean_r2"]
        mae = info_m["downstream_mean_mae"]
        tag = " ← BEST" if method == winner else ""
        print(f"  {method:<15} R²={r2:.4f}  MAE={mae:.4f}{tag}")
    print(f"\n  Feature type:  {args.feature_type}")
    print(f"  Scaler:        {args.scaler}")
    print(f"  Samples:       {len(samples)}")
    print(f"  Raw features:  {X.shape[1]}")
    print(f"  Components:    {args.n_components}")
    print(f"  PCA variance:  {pca_result['cumulative_variance'][-1]*100:.1f}%")
    print(f"\n  Outputs in: {outdir}/")
    print("="*60)


if __name__ == "__main__":
    main()