#!/usr/bin/env python3
"""
dim_reduction_pipeline.py

Dimensionality reduction pipeline for synthetic trabecular bone micro-CT images.
Performs PCA and Random Projection, extracts morphometric labels, and prepares
feature vectors for downstream classical/quantum ML.

Usage:
  python dim_reduction_pipeline.py \
      --dataset-dir output/v16_dense_v8 \
      --outdir output/features \
      --n-components 16 \
      --slice-mode mid \
      --image-size 64 \
      --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from PIL import Image
import tifffile as tiff
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────
#  1. DATA LOADING
# ──────────────────────────────────────────────────────────

def discover_samples(dataset_dir: Path) -> list[dict]:
    """Find all sample directories and load their metrics + images."""
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
    """Load a 3D TIFF volume as float32 [0, 1]."""
    vol = tiff.imread(str(path)).astype(np.float32)
    vmax = vol.max()
    if vmax > 0:
        vol /= vmax
    return vol


def extract_slices(vol: np.ndarray, mode: str = "mid", n_slices: int = 5) -> list[np.ndarray]:
    """Extract 2D slices from a 3D volume.

    Modes:
        mid      - single middle slice
        multi    - n_slices evenly spaced
        all      - every slice
        mip      - maximum intensity projection (single image)
    """
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
    """Resize a 2D slice to (size, size) using PIL."""
    img = Image.fromarray((s * 255).astype(np.uint8), mode="L")
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


# ──────────────────────────────────────────────────────────
#  2. MORPHOMETRIC LABEL EXTRACTION
# ──────────────────────────────────────────────────────────

# These are the four key parameters your supervisor identified
LABEL_KEYS = [
    "BVTV",
    "TbTh_um_p50",
    "TbN_per_mm",
    "TbSp_um_p50",
]

# Additional labels available from the generator
EXTENDED_LABEL_KEYS = [
    "BVTV",
    "TbTh_um_p50",
    "TbTh_um_p90",
    "TbN_per_mm",
    "TbSp_um_p50",
    "TbSp_um_p90",
    "Euler",
    "ConnProxy",
    "n_components",
    "lcc_frac",
]


def extract_labels(metrics: dict, keys: list[str] = None) -> dict:
    """Pull morphometric labels from a sample's metrics.json."""
    keys = keys or LABEL_KEYS
    morph = metrics.get("morphometrics", {})
    labels = {}
    for k in keys:
        v = morph.get(k)
        if v is not None:
            labels[k] = float(v)
    return labels


def extract_generator_params(metrics: dict) -> dict:
    """Pull the generator control parameters used for this sample."""
    params = metrics.get("params", {})
    ridge = params.get("ridge", {})
    targets = metrics.get("targets", {})

    return {
        # Morphometric targets
        "bvtv_target": targets.get("bvtv_target"),
        "tbth_um_target": targets.get("tbth_um_target"),
        "tbn_target": targets.get("tbn_target"),
        "tbsp_um_target": targets.get("tbsp_um_target"),
        # Network topology controls
        "base_sigma": ridge.get("base_sigma"),
        "aniso_ratio": ridge.get("aniso_ratio"),
        "rod_weight": ridge.get("rod_weight"),
        "plate_weight": ridge.get("plate_weight"),
        "sheet_q": ridge.get("sheet_q"),
        # Proto-network thresholds
        "proto_q_hi": ridge.get("proto_q_hi"),
        "proto_q_lo": ridge.get("proto_q_lo"),
        "proto_close_iters": ridge.get("proto_close_iters"),
        # Skeleton refinement
        "skeleton_prune_lmin": ridge.get("skeleton_prune_lmin"),
        "reconnect_close_iters": ridge.get("reconnect_close_iters"),
        # Thickening
        "radius_mode": ridge.get("radius_mode"),
        "radius_jitter": ridge.get("radius_jitter"),
        "radius_smooth_sigma": ridge.get("radius_smooth_sigma"),
        "radius_scale_hint": ridge.get("radius_scale_hint"),
        # Seed
        "seed": metrics.get("seed"),
    }


# ──────────────────────────────────────────────────────────
#  3. FEATURE MATRIX CONSTRUCTION
# ──────────────────────────────────────────────────────────

def build_feature_matrix(
    samples: list[dict],
    use_gray: bool = True,
    slice_mode: str = "mid",
    n_slices: int = 5,
    image_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    """Build flattened feature matrix X and label matrix Y.

    Returns:
        X: (n_images, image_size*image_size) flattened pixel features
        Y: (n_images, n_labels) morphometric labels
        sample_info: per-image metadata
        gen_params: per-image generator parameters
    """
    X_rows = []
    Y_rows = []
    info = []
    gen_params = []

    for s in samples:
        path = s["gray_path"] if (use_gray and s["gray_path"]) else s["mask_path"]
        if path is None:
            continue

        vol = load_volume(path)
        slices = extract_slices(vol, mode=slice_mode, n_slices=n_slices)

        labels = extract_labels(s["metrics"])
        params = extract_generator_params(s["metrics"])

        for si, sl in enumerate(slices):
            resized = resize_slice(sl, image_size)
            X_rows.append(resized.ravel())
            Y_rows.append([labels.get(k, 0.0) for k in LABEL_KEYS])
            info.append({
                "sample": s["name"],
                "slice_idx": si,
                "source": str(path),
            })
            gen_params.append(params)

    X = np.array(X_rows, dtype=np.float32)
    Y = np.array(Y_rows, dtype=np.float32)

    print(f"Feature matrix: X={X.shape}, Y={Y.shape}")
    print(f"  Image size: {image_size}x{image_size} = {image_size**2} raw features")
    print(f"  Labels: {LABEL_KEYS}")
    return X, Y, info, gen_params


# ──────────────────────────────────────────────────────────
#  4. PCA
# ──────────────────────────────────────────────────────────

def run_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    scaler: Optional[StandardScaler] = None,
) -> dict:
    """Run PCA dimensionality reduction.

    Returns dict with transformed data, model, and diagnostics.
    """
    if scaler is None:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
    else:
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

    n_comp = min(n_components, X_train_s.shape[0], X_train_s.shape[1])

    pca = PCA(n_components=n_comp, random_state=42)
    Z_train = pca.fit_transform(X_train_s)
    Z_test = pca.transform(X_test_s)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # Reconstruction error
    X_recon = pca.inverse_transform(Z_train)
    recon_mse = float(np.mean((X_train_s - X_recon) ** 2))

    print(f"\nPCA: {n_comp} components")
    print(f"  Variance explained: {cumulative[-1]:.4f} ({cumulative[-1]*100:.1f}%)")
    print(f"  Top 5 components: {explained[:5]}")
    print(f"  Reconstruction MSE: {recon_mse:.6f}")

    return {
        "method": "PCA",
        "model": pca,
        "scaler": scaler,
        "Z_train": Z_train,
        "Z_test": Z_test,
        "n_components": n_comp,
        "explained_variance_ratio": explained.tolist(),
        "cumulative_variance": cumulative.tolist(),
        "reconstruction_mse": recon_mse,
    }


# ──────────────────────────────────────────────────────────
#  5. RANDOM PROJECTION
# ──────────────────────────────────────────────────────────

def run_random_projection(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    method: str = "gaussian",
    seed: int = 42,
    scaler: Optional[StandardScaler] = None,
) -> dict:
    """Run Random Projection dimensionality reduction.

    Methods: 'gaussian' or 'sparse'
    """
    if scaler is None:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
    else:
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

    n_comp = min(n_components, X_train_s.shape[1])

    if method == "gaussian":
        rp = GaussianRandomProjection(n_components=n_comp, random_state=seed)
    elif method == "sparse":
        rp = SparseRandomProjection(n_components=n_comp, random_state=seed)
    else:
        raise ValueError(f"Unknown RP method: {method}")

    Z_train = rp.fit_transform(X_train_s)
    Z_test = rp.transform(X_test_s)

    # Pairwise distance preservation (Johnson-Lindenstrauss quality)
    n_check = min(200, X_train_s.shape[0])
    idx = np.random.default_rng(seed).choice(X_train_s.shape[0], n_check, replace=False)
    D_orig = np.linalg.norm(
        X_train_s[idx, None, :] - X_train_s[None, idx, :], axis=-1
    )
    D_proj = np.linalg.norm(
        Z_train[idx, None, :] - Z_train[None, idx, :], axis=-1
    )
    mask = D_orig > 0
    if mask.any():
        ratios = D_proj[mask] / D_orig[mask]
        dist_preservation = {
            "mean_ratio": float(np.mean(ratios)),
            "std_ratio": float(np.std(ratios)),
            "min_ratio": float(np.min(ratios)),
            "max_ratio": float(np.max(ratios)),
        }
    else:
        dist_preservation = {"mean_ratio": None}

    print(f"\nRandom Projection ({method}): {n_comp} components")
    print(f"  Distance preservation: mean={dist_preservation['mean_ratio']:.4f} "
          f"std={dist_preservation['std_ratio']:.4f}")

    return {
        "method": f"RP_{method}",
        "model": rp,
        "scaler": scaler,
        "Z_train": Z_train,
        "Z_test": Z_test,
        "n_components": n_comp,
        "distance_preservation": dist_preservation,
    }


# ──────────────────────────────────────────────────────────
#  6. ANALYSIS & COMPARISON
# ──────────────────────────────────────────────────────────

def compare_methods(pca_result: dict, rp_results: list[dict]) -> dict:
    """Compare PCA vs RP in terms of feature quality."""
    comparison = {
        "PCA": {
            "n_components": pca_result["n_components"],
            "variance_explained": pca_result["cumulative_variance"][-1],
            "reconstruction_mse": pca_result["reconstruction_mse"],
        }
    }
    for rp in rp_results:
        comparison[rp["method"]] = {
            "n_components": rp["n_components"],
            "distance_preservation_mean": rp["distance_preservation"]["mean_ratio"],
            "distance_preservation_std": rp["distance_preservation"]["std_ratio"],
        }
    return comparison


def label_correlation_analysis(
    Z: np.ndarray,
    Y: np.ndarray,
    method_name: str,
    label_names: list[str],
) -> dict:
    """Compute correlation between reduced features and morphometric labels."""
    n_comp = Z.shape[1]
    n_labels = Y.shape[1]
    corr = np.zeros((n_comp, n_labels))

    for i in range(n_comp):
        for j in range(n_labels):
            if np.std(Z[:, i]) > 1e-8 and np.std(Y[:, j]) > 1e-8:
                corr[i, j] = np.corrcoef(Z[:, i], Y[:, j])[0, 1]

    print(f"\n{method_name} — Label correlations (top components):")
    for j, ln in enumerate(label_names):
        best_comp = int(np.argmax(np.abs(corr[:, j])))
        best_corr = corr[best_comp, j]
        print(f"  {ln}: best PC{best_comp} r={best_corr:.3f}")

    return {
        "method": method_name,
        "correlation_matrix": corr.tolist(),
        "label_names": label_names,
    }


# ──────────────────────────────────────────────────────────
#  7. VISUALIZATION
# ──────────────────────────────────────────────────────────

def plot_pca_variance(pca_result: dict, outdir: Path):
    """Plot explained variance and cumulative variance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ev = pca_result["explained_variance_ratio"]
    cv = pca_result["cumulative_variance"]
    n = len(ev)

    ax1.bar(range(n), ev, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("PCA: Individual Variance")

    ax2.plot(range(n), cv, "o-", color="darkorange")
    ax2.axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="95%")
    ax2.axhline(y=0.90, color="green", linestyle="--", alpha=0.5, label="90%")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance")
    ax2.set_title("PCA: Cumulative Variance")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(outdir / "pca_variance.png", dpi=150)
    plt.close()
    print(f"  Saved: {outdir / 'pca_variance.png'}")


def plot_2d_embedding(
    Z: np.ndarray,
    Y: np.ndarray,
    label_idx: int,
    label_name: str,
    method_name: str,
    outdir: Path,
):
    """Scatter plot of first two components colored by a label."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=Y[:, label_idx], cmap="viridis",
                    s=30, alpha=0.7, edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label=label_name)
    ax.set_xlabel(f"{method_name} Component 1")
    ax.set_ylabel(f"{method_name} Component 2")
    ax.set_title(f"{method_name}: colored by {label_name}")
    plt.tight_layout()
    fname = f"{method_name.lower()}_2d_{label_name}.png"
    plt.savefig(outdir / fname, dpi=150)
    plt.close()
    print(f"  Saved: {outdir / fname}")


def plot_correlation_heatmap(corr_result: dict, outdir: Path):
    """Plot heatmap of component-label correlations."""
    corr = np.array(corr_result["correlation_matrix"])
    labels = corr_result["label_names"]
    method = corr_result["method"]

    n_show = min(corr.shape[0], 20)
    fig, ax = plt.subplots(figsize=(8, max(4, n_show * 0.4)))
    im = ax.imshow(corr[:n_show, :], aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1)
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
    """Side-by-side comparison of PCA vs RP embeddings."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sc1 = ax1.scatter(Z_pca[:, 0], Z_pca[:, 1], c=Y[:, label_idx],
                      cmap="viridis", s=30, alpha=0.7)
    ax1.set_title(f"PCA (colored by {label_name})")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    plt.colorbar(sc1, ax=ax1)

    sc2 = ax2.scatter(Z_rp[:, 0], Z_rp[:, 1], c=Y[:, label_idx],
                      cmap="viridis", s=30, alpha=0.7)
    ax2.set_title(f"Random Projection (colored by {label_name})")
    ax2.set_xlabel("RP1")
    ax2.set_ylabel("RP2")
    plt.colorbar(sc2, ax=ax2)

    plt.tight_layout()
    plt.savefig(outdir / f"comparison_pca_rp_{label_name}.png", dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────
#  8. EXPORT FOR QUANTUM KERNELS
# ──────────────────────────────────────────────────────────

def export_for_quantum(
    Z_train: np.ndarray,
    Z_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    method_name: str,
    info_train: list[dict],
    info_test: list[dict],
    gen_params_train: list[dict],
    outdir: Path,
):
    """Export reduced features + labels for quantum kernel experiments.

    Saves .npz files ready for qiskit-machine-learning or pennylane.
    Normalizes features to [0, pi] range suitable for angle encoding.
    """
    # Normalize to [0, pi] for angle encoding
    mm = MinMaxScaler(feature_range=(0, np.pi))
    Z_train_q = mm.fit_transform(Z_train)
    Z_test_q = mm.transform(Z_test)

    # Also save [0, 1] normalized version
    mm01 = MinMaxScaler(feature_range=(0, 1))
    Z_train_01 = mm01.fit_transform(Z_train)
    Z_test_01 = mm01.transform(Z_test)

    fname = outdir / f"{method_name.lower()}_quantum_ready.npz"
    np.savez(
        fname,
        Z_train=Z_train_q,
        Z_test=Z_test_q,
        Z_train_01=Z_train_01,
        Z_test_01=Z_test_01,
        Y_train=Y_train,
        Y_test=Y_test,
        label_names=LABEL_KEYS,
        n_features=Z_train_q.shape[1],
        n_train=Z_train_q.shape[0],
        n_test=Z_test_q.shape[0],
    )
    print(f"  Quantum-ready features saved: {fname}")
    print(f"    Train: {Z_train_q.shape}, Test: {Z_test_q.shape}")
    print(f"    Feature range: [0, pi] for angle encoding")

    # Save generator params as JSON for traceability
    params_fname = outdir / f"{method_name.lower()}_generator_params.json"
    with open(params_fname, "w") as f:
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
#  9. MAIN PIPELINE
# ──────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="PCA & RP pipeline for trabecular images")
    p.add_argument("--dataset-dir", type=str, required=True,
                   help="Directory containing sample_000/, sample_001/, etc.")
    p.add_argument("--outdir", type=str, default="output/features")
    p.add_argument("--n-components", type=int, default=16,
                   help="Number of reduced dimensions (= qubits for quantum)")
    p.add_argument("--slice-mode", type=str, default="mid",
                   choices=["mid", "multi", "all", "mip"])
    p.add_argument("--n-slices", type=int, default=5,
                   help="Number of slices for 'multi' mode")
    p.add_argument("--image-size", type=int, default=64,
                   help="Resize images to NxN before flattening")
    p.add_argument("--use-mask", action="store_true",
                   help="Use binary mask instead of grayscale")
    p.add_argument("--test-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rp-method", type=str, default="both",
                   choices=["gaussian", "sparse", "both"])
    return p


def main():
    args = build_parser().parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DIMENSIONALITY REDUCTION PIPELINE")
    print("=" * 60)

    # ── Load data ──
    samples = discover_samples(Path(args.dataset_dir))
    if len(samples) == 0:
        raise FileNotFoundError(f"No samples found in {args.dataset_dir}")

    X, Y, info, gen_params = build_feature_matrix(
        samples,
        use_gray=not args.use_mask,
        slice_mode=args.slice_mode,
        n_slices=args.n_slices,
        image_size=args.image_size,
    )

    if X.shape[0] < 4:
        print(f"\nWARNING: Only {X.shape[0]} images. Generate more samples for "
              f"meaningful dimensionality reduction. Recommend >= 30.")

    # ── Train/test split ──
    if X.shape[0] >= 4:
        X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
            X, Y, np.arange(X.shape[0]),
            test_size=args.test_split, random_state=args.seed,
        )
        info_train = [info[i] for i in idx_train]
        info_test = [info[i] for i in idx_test]
        gp_train = [gen_params[i] for i in idx_train]
    else:
        X_train, X_test = X, X
        Y_train, Y_test = Y, Y
        info_train, info_test = info, info
        gp_train = gen_params

    print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # ── PCA ──
    print("\n" + "─" * 40)
    print("  PCA")
    print("─" * 40)
    pca_result = run_pca(X_train, X_test, args.n_components)

    pca_corr = label_correlation_analysis(
        pca_result["Z_train"], Y_train, "PCA", LABEL_KEYS
    )

    plot_pca_variance(pca_result, outdir)
    plot_correlation_heatmap(pca_corr, outdir)
    for li, ln in enumerate(LABEL_KEYS):
        plot_2d_embedding(
            pca_result["Z_train"], Y_train, li, ln, "PCA", outdir
        )

    pca_quantum = export_for_quantum(
        pca_result["Z_train"], pca_result["Z_test"],
        Y_train, Y_test, "PCA", info_train, info_test, gp_train, outdir,
    )

    # ── Random Projection ──
    rp_results = []
    rp_methods = (
        ["gaussian", "sparse"] if args.rp_method == "both"
        else [args.rp_method]
    )

    for rp_method in rp_methods:
        print("\n" + "─" * 40)
        print(f"  RANDOM PROJECTION ({rp_method})")
        print("─" * 40)
        rp_result = run_random_projection(
            X_train, X_test, args.n_components,
            method=rp_method, seed=args.seed,
        )
        rp_results.append(rp_result)

        rp_corr = label_correlation_analysis(
            rp_result["Z_train"], Y_train, f"RP_{rp_method}", LABEL_KEYS
        )

        plot_correlation_heatmap(rp_corr, outdir)
        for li, ln in enumerate(LABEL_KEYS):
            plot_2d_embedding(
                rp_result["Z_train"], Y_train, li, ln,
                f"RP_{rp_method}", outdir,
            )

        rp_quantum = export_for_quantum(
            rp_result["Z_train"], rp_result["Z_test"],
            Y_train, Y_test, f"RP_{rp_method}",
            info_train, info_test, gp_train, outdir,
        )

    # ── Side-by-side comparison ──
    if rp_results:
        for li, ln in enumerate(LABEL_KEYS):
            plot_method_comparison(
                pca_result["Z_train"], rp_results[0]["Z_train"],
                Y_train, li, ln, outdir,
            )

    # ── Summary ──
    comparison = compare_methods(pca_result, rp_results)

    summary = {
        "pipeline_version": "1.0",
        "dataset_dir": str(args.dataset_dir),
        "n_samples": len(samples),
        "n_images": X.shape[0],
        "image_size": args.image_size,
        "raw_features": X.shape[1],
        "n_components": args.n_components,
        "slice_mode": args.slice_mode,
        "comparison": comparison,
        "quantum_export": {
            "pca": pca_quantum,
            "n_qubits": args.n_components,
            "encoding": "angle [0, pi]",
        },
        "label_keys": LABEL_KEYS,
        "controllable_generator_params": [
            "bvtv", "tbth_um", "tbn_per_mm", "tbsp_um",
            "base_sigma", "aniso_ratio", "rod_weight", "plate_weight",
            "sheet_q", "proto_q_hi", "proto_q_lo", "proto_close_iters",
            "skeleton_prune_lmin", "reconnect_close_iters",
            "radius_mode", "radius_jitter", "radius_smooth_sigma",
            "radius_scale_hint", "marrow_mean", "bone_mean",
            "solid_fill_sigma", "noise_sd", "round_sigma",
            "warp_amp", "warp_sigma", "hessian_sigma",
            "enforce_lcc", "min_component_size",
        ],
    }

    with open(outdir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Samples: {len(samples)}")
    print(f"  Images:  {X.shape[0]}")
    print(f"  Raw features: {X.shape[1]}")
    print(f"  Reduced to: {args.n_components} components")
    print(f"  PCA variance explained: "
          f"{pca_result['cumulative_variance'][-1]*100:.1f}%")
    for rp in rp_results:
        dp = rp["distance_preservation"]
        print(f"  {rp['method']} distance preservation: "
              f"{dp['mean_ratio']:.3f} +/- {dp['std_ratio']:.3f}")
    print(f"\n  Outputs in: {outdir}/")
    print(f"  Quantum-ready .npz files scaled to [0, pi] for angle encoding")
    print("=" * 60)


if __name__ == "__main__":
    main()