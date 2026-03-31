#!/usr/bin/env python3
"""
dim_reduction_pipeline.py

Dimensionality reduction pipeline for synthetic trabecular bone micro-CT images.
Supports two feature modes:
  --feature-type pixels   : flatten raw grayscale pixels (original, ~4096 features)
  --feature-type texture  : extract GLCM + statistical texture descriptors (~50 features)

Texture mode dramatically improves PCA variance explained by replacing noisy
raw pixels with morphometrically meaningful descriptors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import tifffile as tiff
from scipy import ndimage as ndi
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_GLCM = True
except ImportError:
    SKIMAGE_GLCM = False
    print("WARNING: skimage.feature not available, GLCM features disabled")


# ──────────────────────────────────────────────────────────
#  1. DATA LOADING
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
            print(f"  Skipping {d.name}: no gray.tif or mask.tif")
            continue
        samples.append({
            "name": d.name,
            "dir": d,
            "metrics": metrics,
            "gray_path": gray_path if gray_path.exists() else None,
            "mask_path": mask_path if mask_path.exists() else None,
        })
    print(f"Discovered {len(samples)} samples in {dataset_dir}")
    return samples


def load_volume(path: Path) -> np.ndarray:
    vol = tiff.imread(str(path)).astype(np.float32)
    vmax = vol.max()
    if vmax > 0:
        vol /= vmax
    return vol


def extract_slices(vol: np.ndarray, mode: str = "mid", n_slices: int = 5) -> list[np.ndarray]:
    Z = vol.shape[0]
    if mode == "mid":
        return [vol[Z // 2]]
    elif mode == "multi":
        indices = np.linspace(0, Z - 1, n_slices, dtype=int)
        return [vol[i] for i in indices]
    elif mode == "all":
        return [vol[i] for i in range(Z)]
    elif mode == "mip":
        return [vol.max(axis=0)]
    else:
        raise ValueError(f"Unknown slice mode: {mode}")


def resize_slice(s: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray((s * 255).astype(np.uint8), mode="L")
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


# ──────────────────────────────────────────────────────────
#  2. TEXTURE FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────

def extract_glcm_features(img_u8: np.ndarray) -> np.ndarray:
    """
    Extract GLCM texture features from a uint8 image.
    Returns ~12 features: contrast, dissimilarity, homogeneity,
    energy, correlation, ASM at distances [1,3] averaged over angles.
    """
    if not SKIMAGE_GLCM:
        return np.array([])

    distances = [1, 3]
    angles    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    props     = ["contrast", "dissimilarity", "homogeneity",
                 "energy", "correlation", "ASM"]

    # Reduce grey levels to 32 for speed
    img_q = (img_u8 // 8).astype(np.uint8)
    glcm  = graycomatrix(img_q, distances=distances, angles=angles,
                         levels=32, symmetric=True, normed=True)

    feats = []
    for prop in props:
        vals = graycoprops(glcm, prop)   # shape (n_distances, n_angles)
        feats.extend([float(vals.mean()), float(vals.std())])

    return np.array(feats, dtype=np.float32)


def extract_statistical_features(img: np.ndarray) -> np.ndarray:
    """
    Statistical and gradient features from a [0,1] float image.
    Returns ~29 features.
    """
    feats = []

    # Global intensity statistics
    feats += [
        float(img.mean()),
        float(img.std()),
        float(np.median(img)),
        float(skew(img.ravel())),
        float(kurtosis(img.ravel())),
        float(img.min()),
        float(img.max()),
        float(np.percentile(img, 25)),
        float(np.percentile(img, 75)),
    ]

    # Intensity histogram (16 bins)
    hist, _ = np.histogram(img.ravel(), bins=16, range=(0, 1))
    hist    = hist.astype(np.float32) / (hist.sum() + 1e-8)
    feats.extend(hist.tolist())

    # Gradient magnitude statistics (edge content — captures strut boundaries)
    gx   = ndi.sobel(img, axis=1)
    gy   = ndi.sobel(img, axis=0)
    gmag = np.sqrt(gx**2 + gy**2)
    feats += [
        float(gmag.mean()),
        float(gmag.std()),
        float(np.percentile(gmag, 75)),
        float(np.percentile(gmag, 95)),
    ]

    # Local variance (texture roughness)
    local_var = ndi.uniform_filter(img**2, size=5) - ndi.uniform_filter(img, size=5)**2
    feats += [
        float(local_var.mean()),
        float(local_var.std()),
    ]

    return np.array(feats, dtype=np.float32)


def extract_texture_features(sl: np.ndarray, image_size: int = 64) -> np.ndarray:
    """Combined texture feature vector from a single 2D slice."""
    resized    = resize_slice(sl, image_size)
    img_u8     = (resized * 255).astype(np.uint8)
    stat_feats = extract_statistical_features(resized)
    glcm_feats = extract_glcm_features(img_u8)

    if glcm_feats.size > 0:
        return np.concatenate([stat_feats, glcm_feats])
    else:
        return stat_feats


# ──────────────────────────────────────────────────────────
#  3. MORPHOMETRIC LABEL EXTRACTION
# ──────────────────────────────────────────────────────────

LABEL_KEYS = [
    "BVTV",
    "TbTh_um_p50",
    "TbN_per_mm",
    "TbSp_um_p50",
]


def extract_labels(metrics: dict, keys: list[str] = None) -> dict:
    keys  = keys or LABEL_KEYS
    morph = metrics.get("morphometrics", {})
    return {k: float(morph[k]) for k in keys if morph.get(k) is not None}


def extract_generator_params(metrics: dict) -> dict:
    params  = metrics.get("params", {})
    ridge   = params.get("ridge", {})
    targets = metrics.get("targets", {})
    return {
        "bvtv_target":           targets.get("bvtv_target"),
        "tbth_um_target":        targets.get("tbth_um_target"),
        "tbn_target":            targets.get("tbn_target"),
        "tbsp_um_target":        targets.get("tbsp_um_target"),
        "base_sigma":            ridge.get("base_sigma"),
        "aniso_ratio":           ridge.get("aniso_ratio"),
        "proto_q_hi":            ridge.get("proto_q_hi"),
        "proto_q_lo":            ridge.get("proto_q_lo"),
        "skeleton_prune_lmin":   ridge.get("skeleton_prune_lmin"),
        "reconnect_close_iters": ridge.get("reconnect_close_iters"),
        "radius_jitter":         ridge.get("radius_jitter"),
        "seed":                  metrics.get("seed"),
    }


# ──────────────────────────────────────────────────────────
#  4. FEATURE MATRIX CONSTRUCTION
# ──────────────────────────────────────────────────────────

def build_feature_matrix(
    samples: list[dict],
    use_gray: bool = True,
    slice_mode: str = "mid",
    n_slices: int = 5,
    image_size: int = 64,
    feature_type: str = "texture",
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    """
    feature_type='pixels'  : flatten raw grayscale (original, ~4096 features)
    feature_type='texture' : GLCM + statistical descriptors (~50 features)
    """
    X_rows     = []
    Y_rows     = []
    info       = []
    gen_params = []

    for s in samples:
        path = s["gray_path"] if (use_gray and s["gray_path"]) else s["mask_path"]
        if path is None:
            continue

        vol    = load_volume(path)
        slices = extract_slices(vol, mode=slice_mode, n_slices=n_slices)
        labels = extract_labels(s["metrics"])
        params = extract_generator_params(s["metrics"])

        for si, sl in enumerate(slices):
            if feature_type == "texture":
                feat = extract_texture_features(sl, image_size)
            else:
                resized = resize_slice(sl, image_size)
                feat    = resized.ravel()

            X_rows.append(feat)
            Y_rows.append([labels.get(k, 0.0) for k in LABEL_KEYS])
            info.append({"sample": s["name"], "slice_idx": si, "source": str(path)})
            gen_params.append(params)

    X = np.array(X_rows, dtype=np.float32)
    Y = np.array(Y_rows, dtype=np.float32)

    print(f"Feature matrix: X={X.shape}, Y={Y.shape}")
    print(f"  Feature type: {feature_type}")
    if feature_type == "texture":
        print(f"  Texture features per image: {X.shape[1]}")
    else:
        print(f"  Image size: {image_size}x{image_size} = {image_size**2} raw features")
    print(f"  Labels: {LABEL_KEYS}")
    return X, Y, info, gen_params


# ──────────────────────────────────────────────────────────
#  5. PCA
# ──────────────────────────────────────────────────────────

def run_pca(X_train: np.ndarray, X_test: np.ndarray, n_components: int) -> dict:
    scaler     = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)
    X_test_s   = scaler.transform(X_test)
    n_comp     = min(n_components, X_train_s.shape[0], X_train_s.shape[1])

    pca        = PCA(n_components=n_comp, random_state=42)
    Z_train    = pca.fit_transform(X_train_s)
    Z_test     = pca.transform(X_test_s)
    explained  = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    X_recon    = pca.inverse_transform(Z_train)
    recon_mse  = float(np.mean((X_train_s - X_recon) ** 2))

    print(f"\nPCA: {n_comp} components")
    print(f"  Variance explained: {cumulative[-1]:.4f} ({cumulative[-1]*100:.1f}%)")
    print(f"  Top 5 components: {explained[:5]}")
    print(f"  Reconstruction MSE: {recon_mse:.6f}")

    return {
        "method": "PCA", "model": pca, "scaler": scaler,
        "Z_train": Z_train, "Z_test": Z_test,
        "n_components": n_comp,
        "explained_variance_ratio": explained.tolist(),
        "cumulative_variance": cumulative.tolist(),
        "reconstruction_mse": recon_mse,
    }


# ──────────────────────────────────────────────────────────
#  6. RANDOM PROJECTION
# ──────────────────────────────────────────────────────────

def run_random_projection(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    method: str = "gaussian",
    seed: int = 42,
) -> dict:
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    n_comp    = min(n_components, X_train_s.shape[1])

    rp = (GaussianRandomProjection(n_components=n_comp, random_state=seed)
          if method == "gaussian"
          else SparseRandomProjection(n_components=n_comp, random_state=seed))

    Z_train = rp.fit_transform(X_train_s)
    Z_test  = rp.transform(X_test_s)

    n_check = min(200, X_train_s.shape[0])
    idx     = np.random.default_rng(seed).choice(X_train_s.shape[0], n_check, replace=False)
    D_orig  = np.linalg.norm(X_train_s[idx, None, :] - X_train_s[None, idx, :], axis=-1)
    D_proj  = np.linalg.norm(Z_train[idx, None, :] - Z_train[None, idx, :], axis=-1)
    mask    = D_orig > 0
    ratios  = D_proj[mask] / D_orig[mask] if mask.any() else np.array([1.0])
    dist_preservation = {
        "mean_ratio": float(np.mean(ratios)),
        "std_ratio":  float(np.std(ratios)),
    }

    print(f"\nRandom Projection ({method}): {n_comp} components")
    print(f"  Distance preservation: mean={dist_preservation['mean_ratio']:.4f} "
          f"std={dist_preservation['std_ratio']:.4f}")

    return {
        "method": f"RP_{method}", "model": rp, "scaler": scaler,
        "Z_train": Z_train, "Z_test": Z_test,
        "n_components": n_comp,
        "distance_preservation": dist_preservation,
    }


# ──────────────────────────────────────────────────────────
#  7. ANALYSIS
# ──────────────────────────────────────────────────────────

def label_correlation_analysis(Z, Y, method_name, label_names) -> dict:
    n_comp   = Z.shape[1]
    n_labels = Y.shape[1]
    corr     = np.zeros((n_comp, n_labels))

    for i in range(n_comp):
        for j in range(n_labels):
            if np.std(Z[:, i]) > 1e-8 and np.std(Y[:, j]) > 1e-8:
                corr[i, j] = np.corrcoef(Z[:, i], Y[:, j])[0, 1]

    print(f"\n{method_name} — Label correlations (top components):")
    for j, ln in enumerate(label_names):
        best_comp = int(np.argmax(np.abs(corr[:, j])))
        print(f"  {ln}: best PC{best_comp} r={corr[best_comp, j]:.3f}")

    return {"method": method_name, "correlation_matrix": corr.tolist(), "label_names": label_names}


def compare_methods(pca_result, rp_results) -> dict:
    c = {"PCA": {
        "n_components": pca_result["n_components"],
        "variance_explained": pca_result["cumulative_variance"][-1],
        "reconstruction_mse": pca_result["reconstruction_mse"],
    }}
    for rp in rp_results:
        c[rp["method"]] = {
            "n_components": rp["n_components"],
            "distance_preservation_mean": rp["distance_preservation"]["mean_ratio"],
            "distance_preservation_std":  rp["distance_preservation"]["std_ratio"],
        }
    return c


# ──────────────────────────────────────────────────────────
#  8. VISUALIZATION
# ──────────────────────────────────────────────────────────

def plot_pca_variance(pca_result, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ev = pca_result["explained_variance_ratio"]
    cv = pca_result["cumulative_variance"]

    ax1.bar(range(len(ev)), ev, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("PCA: Individual Variance")

    ax2.plot(range(len(cv)), cv, "o-", color="darkorange")
    ax2.axhline(0.95, color="red",   ls="--", alpha=0.5, label="95%")
    ax2.axhline(0.90, color="green", ls="--", alpha=0.5, label="90%")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance")
    ax2.set_title("PCA: Cumulative Variance")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(outdir / "pca_variance.png", dpi=150)
    plt.close()
    print(f"  Saved: {outdir / 'pca_variance.png'}")


def plot_2d_embedding(Z, Y, label_idx, label_name, method_name, outdir):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc      = ax.scatter(Z[:, 0], Z[:, 1], c=Y[:, label_idx], cmap="viridis",
                         s=30, alpha=0.7, edgecolors="k", linewidths=0.3)
    cbar    = plt.colorbar(sc, ax=ax)
    cbar.set_label(label_name)
    ax.set_xlabel(f"{method_name} Component 1")
    ax.set_ylabel(f"{method_name} Component 2")
    ax.set_title(f"PCA: colored by {label_name}")
    plt.tight_layout()
    fname = f"pca_2d_{label_name}.png" if method_name == "PCA" else f"{method_name.lower()}_2d_{label_name}.png"
    plt.savefig(outdir / fname, dpi=150)
    plt.close()
    print(f"  Saved: {outdir / fname}")


def plot_correlation_heatmap(corr_result, outdir):
    corr   = np.array(corr_result["correlation_matrix"])
    labels = corr_result["label_names"]
    method = corr_result["method"]
    n_show = min(corr.shape[0], 20)

    fig, ax = plt.subplots(figsize=(8, max(4, n_show * 0.4)))
    im = ax.imshow(corr[:n_show, :], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(range(n_show))
    ax.set_yticklabels([f"C{i}" for i in range(n_show)])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(f"{method}: Component-Label Correlations")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fname = f"{method.lower()}_correlation_heatmap.png"
    plt.savefig(outdir / fname, dpi=150)
    plt.close()
    print(f"  Saved: {outdir / fname}")


def plot_method_comparison(Z_pca, Z_rp, Y, label_idx, label_name, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sc1 = ax1.scatter(Z_pca[:, 0], Z_pca[:, 1], c=Y[:, label_idx], cmap="viridis", s=30, alpha=0.7)
    ax1.set_title(f"PCA (colored by {label_name})")
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
    plt.colorbar(sc1, ax=ax1).set_label(label_name)

    sc2 = ax2.scatter(Z_rp[:, 0], Z_rp[:, 1], c=Y[:, label_idx], cmap="viridis", s=30, alpha=0.7)
    ax2.set_title(f"Random Projection (colored by {label_name})")
    ax2.set_xlabel("RP1"); ax2.set_ylabel("RP2")
    plt.colorbar(sc2, ax=ax2).set_label(label_name)

    plt.tight_layout()
    plt.savefig(outdir / f"comparison_pca_rp_{label_name}.png", dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────
#  9. QUANTUM EXPORT
# ──────────────────────────────────────────────────────────

def export_for_quantum(Z_train, Z_test, Y_train, Y_test,
                       method_name, info_train, info_test, gen_params_train, outdir):
    mm         = MinMaxScaler(feature_range=(0, np.pi))
    Z_train_q  = mm.fit_transform(Z_train)
    Z_test_q   = mm.transform(Z_test)
    mm01       = MinMaxScaler(feature_range=(0, 1))
    Z_train_01 = mm01.fit_transform(Z_train)
    Z_test_01  = mm01.transform(Z_test)

    fname = outdir / f"{method_name.lower()}_quantum_ready.npz"
    np.savez(fname,
             Z_train=Z_train_q, Z_test=Z_test_q,
             Z_train_01=Z_train_01, Z_test_01=Z_test_01,
             Y_train=Y_train, Y_test=Y_test,
             label_names=LABEL_KEYS,
             n_features=Z_train_q.shape[1],
             n_train=Z_train_q.shape[0],
             n_test=Z_test_q.shape[0])
    print(f"  Quantum-ready features saved: {fname}")
    print(f"    Train: {Z_train_q.shape}, Test: {Z_test_q.shape}")
    print(f"    Feature range: [0, pi] for angle encoding")

    with open(outdir / f"{method_name.lower()}_generator_params.json", "w") as f:
        json.dump({
            "method": method_name,
            "n_features": int(Z_train_q.shape[1]),
            "n_train": int(Z_train_q.shape[0]),
            "n_test": int(Z_test_q.shape[0]),
            "label_names": LABEL_KEYS,
            "generator_params_train": gen_params_train,
            "sample_info_train": info_train,
            "sample_info_test": info_test,
        }, f, indent=2, default=str)

    return {
        "file": str(fname),
        "n_qubits_needed": int(Z_train_q.shape[1]),
        "encoding": "angle",
        "feature_range": "[0, pi]",
    }


# ──────────────────────────────────────────────────────────
#  10. MAIN
# ──────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="PCA & RP pipeline for trabecular images")
    p.add_argument("--dataset-dir",  type=str, required=True)
    p.add_argument("--outdir",       type=str, default="output/features")
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
                   choices=["pixels", "texture"],
                   help="'texture' uses GLCM+stats (recommended), 'pixels' uses raw pixels")
    return p


def main():
    args   = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DIMENSIONALITY REDUCTION PIPELINE")
    print(f"  Feature type: {args.feature_type}")
    print("=" * 60)

    samples = discover_samples(Path(args.dataset_dir))
    if len(samples) == 0:
        raise FileNotFoundError(f"No samples found in {args.dataset_dir}")

    X, Y, info, gen_params = build_feature_matrix(
        samples,
        use_gray=not args.use_mask,
        slice_mode=args.slice_mode,
        n_slices=args.n_slices,
        image_size=args.image_size,
        feature_type=args.feature_type,
    )

    if X.shape[0] < 4:
        print(f"\nWARNING: Only {X.shape[0]} images.")
        return

    X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
        X, Y, np.arange(X.shape[0]),
        test_size=args.test_split, random_state=args.seed,
    )
    info_train = [info[i] for i in idx_train]
    info_test  = [info[i] for i in idx_test]
    gp_train   = [gen_params[i] for i in idx_train]

    print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # ── PCA ──
    print("\n" + "─" * 40)
    print("  PCA")
    print("─" * 40)
    pca_result = run_pca(X_train, X_test, args.n_components)
    pca_corr   = label_correlation_analysis(pca_result["Z_train"], Y_train, "PCA", LABEL_KEYS)

    plot_pca_variance(pca_result, outdir)
    plot_correlation_heatmap(pca_corr, outdir)
    for li, ln in enumerate(LABEL_KEYS):
        plot_2d_embedding(pca_result["Z_train"], Y_train, li, ln, "PCA", outdir)

    pca_quantum = export_for_quantum(
        pca_result["Z_train"], pca_result["Z_test"],
        Y_train, Y_test, "PCA", info_train, info_test, gp_train, outdir,
    )

    # ── Random Projection ──
    rp_results = []
    rp_methods = ["gaussian", "sparse"] if args.rp_method == "both" else [args.rp_method]

    for rp_method in rp_methods:
        print("\n" + "─" * 40)
        print(f"  RANDOM PROJECTION ({rp_method})")
        print("─" * 40)
        rp_result = run_random_projection(X_train, X_test, args.n_components,
                                          method=rp_method, seed=args.seed)
        rp_results.append(rp_result)
        rp_corr = label_correlation_analysis(rp_result["Z_train"], Y_train,
                                              f"RP_{rp_method}", LABEL_KEYS)
        plot_correlation_heatmap(rp_corr, outdir)
        for li, ln in enumerate(LABEL_KEYS):
            plot_2d_embedding(rp_result["Z_train"], Y_train, li, ln,
                              f"RP_{rp_method}", outdir)
        export_for_quantum(
            rp_result["Z_train"], rp_result["Z_test"],
            Y_train, Y_test, f"RP_{rp_method}",
            info_train, info_test, gp_train, outdir,
        )

    if rp_results:
        for li, ln in enumerate(LABEL_KEYS):
            plot_method_comparison(pca_result["Z_train"], rp_results[0]["Z_train"],
                                   Y_train, li, ln, outdir)

    # ── Summary ──
    comparison = compare_methods(pca_result, rp_results)
    summary = {
        "pipeline_version": "2.0",
        "feature_type": args.feature_type,
        "dataset_dir": str(args.dataset_dir),
        "n_samples": len(samples),
        "n_images": X.shape[0],
        "image_size": args.image_size,
        "raw_features": X.shape[1],
        "n_components": args.n_components,
        "slice_mode": args.slice_mode,
        "comparison": comparison,
        "quantum_export": {"pca": pca_quantum, "n_qubits": args.n_components,
                           "encoding": "angle [0, pi]"},
        "label_keys": LABEL_KEYS,
    }
    with open(outdir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Feature type: {args.feature_type}")
    print(f"  Samples: {len(samples)}")
    print(f"  Raw features: {X.shape[1]}")
    print(f"  Reduced to: {args.n_components} components")
    print(f"  PCA variance explained: {pca_result['cumulative_variance'][-1]*100:.1f}%")
    for rp in rp_results:
        dp = rp["distance_preservation"]
        print(f"  {rp['method']} distance preservation: "
              f"{dp['mean_ratio']:.3f} +/- {dp['std_ratio']:.3f}")
    print(f"\n  Outputs in: {outdir}/")
    print(f"  Quantum-ready .npz files scaled to [0, pi] for angle encoding")
    print("=" * 60)


if __name__ == "__main__":
    main()