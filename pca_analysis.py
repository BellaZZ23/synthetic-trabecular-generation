<<<<<<< HEAD
=======
#pca_analysis
>>>>>>> 3085935c66235e46ea21c79646b12d97ccb4bce1
#!/usr/bin/env python3
"""
PCA analysis for synthetic trabecular images.

- Loads PCA-ready dataset from X.npy and metadata.csv (exported by synthetic-trabecular-generation.py).
- Runs PCA (via SVD) on image- or patch-level data.
- Saves:
    * principal components as .npy
    * eigen-images / eigen-patches as PNGs
    * sample scores (PC coordinates) + metadata as a CSV
"""

from pathlib import Path
import argparse
import csv
import math

import numpy as np
from PIL import Image


# --------------------- I/O helpers --------------------- #

def load_pca_dataset(pca_dir):
    """
    Load X.npy and metadata.csv from a PCA dataset directory.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    meta_rows : list[dict]
    """
    pca_dir = Path(pca_dir)
    X_path = pca_dir / "X.npy"
    meta_path = pca_dir / "metadata.csv"

    if not X_path.exists():
        raise FileNotFoundError(f"Could not find X.npy in {pca_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find metadata.csv in {pca_dir}")

    X = np.load(X_path)

    with open(meta_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        meta_rows = list(reader)

    if X.shape[0] != len(meta_rows):
        raise ValueError(
            f"Mismatch between X samples ({X.shape[0]}) and metadata rows ({len(meta_rows)})"
        )

    return X, meta_rows


def ensure_square_side(n_features):
    """
    Infer side length for square images/patches from feature dimension.
    Raises if not perfect square.
    """
    side = int(round(math.sqrt(n_features)))
    if side * side != n_features:
        raise ValueError(
            f"Feature length {n_features} is not a perfect square; "
            f"cannot reshape into (H, W) image."
        )
    return side


# --------------------- PCA core ------------------------ #

def run_pca(X, n_components):
    """
    Run PCA using SVD.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    n_components : int

    Returns
    -------
    mean_vec : (n_features,) mean vector
    components : (n_components, n_features) principal directions
    explained_variance : (n_components,) eigenvalues
    explained_variance_ratio : (n_components,) fraction of total variance
    scores : (n_samples, n_components) PC coordinates of each sample
    """
    # Center
    mean_vec = X.mean(axis=0)
    Xc = X - mean_vec

    # SVD: Xc = U S V^T
    # rows of V^T are principal directions
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    # principal components
    components = Vt[:n_components]

    # explained variance
    n_samples = X.shape[0]
    eigvals = (S ** 2) / (n_samples - 1)
    explained_variance = eigvals[:n_components]
    explained_variance_ratio = explained_variance / eigvals.sum()

    # scores (PC coordinates)
    scores = Xc @ components.T

    return mean_vec, components, explained_variance, explained_variance_ratio, scores


# --------------------- Saving outputs ------------------ #

def save_components_as_images(components, out_dir, prefix="pc"):
    """
    Save each component as a normalized PNG, assuming square images/patches.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_components, n_features = components.shape
    side = ensure_square_side(n_features)

    for i in range(n_components):
        comp = components[i].reshape(side, side)

        cmin, cmax = comp.min(), comp.max()
        if cmax > cmin:
            img = (comp - cmin) / (cmax - cmin) * 255.0
        else:
            img = np.zeros_like(comp)

        img = img.astype(np.uint8)
        Image.fromarray(img, mode="L").save(out_dir / f"{prefix}{i+1}.png")


def save_pca_results(out_dir,
                     mean_vec,
                     components,
                     explained_variance,
                     explained_variance_ratio,
                     scores,
                     meta_rows):
    """
    Save PCA results:
      - mean, components, scores as .npy
      - eigen-images as PNGs
      - scores + metadata as a CSV
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save numeric arrays
    np.save(out_dir / "mean.npy", mean_vec)
    np.save(out_dir / "components.npy", components)
    np.save(out_dir / "scores.npy", scores)
    np.save(out_dir / "explained_variance.npy", explained_variance)
    np.save(out_dir / "explained_variance_ratio.npy", explained_variance_ratio)

    # Save components as images
    save_components_as_images(components, out_dir / "eigen_images")

    # Save scores + metadata as CSV
    scores_csv = out_dir / "scores_with_metadata.csv"
    n_samples, n_components = scores.shape

    # gather fieldnames from metadata plus PC columns
    meta_fieldnames = list(meta_rows[0].keys()) if meta_rows else []
    pc_columns = [f"PC{i+1}" for i in range(n_components)]
    fieldnames = meta_fieldnames + pc_columns

    with open(scores_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx in range(n_samples):
            row = {}
            # metadata (all as strings)
            row.update(meta_rows[idx])

            # PC scores
            for j, col in enumerate(pc_columns):
                row[col] = float(scores[idx, j])

            writer.writerow(row)


# --------------------- CLI wrapper --------------------- #

def build_parser():
    p = argparse.ArgumentParser(
        description="PCA analysis for synthetic trabecular images (uses X.npy + metadata.csv)."
    )
    p.add_argument(
        "--pca-dir",
        type=str,
        default="data/pca",
        help="Directory containing X.npy and metadata.csv (exported by generator).",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="data/pca_results",
        help="Directory where PCA outputs will be saved.",
    )
    p.add_argument(
        "--n-components",
        type=int,
        default=10,
        help="Number of principal components to retain.",
    )
    p.add_argument(
        "--standardize",
        action="store_true",
        help="If set, z-score each feature before PCA (not usually needed for image intensities).",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 1) Load dataset
    print(f"Loading PCA dataset from: {args.pca_dir}")
    X, meta_rows = load_pca_dataset(args.pca_dir)
    n_samples, n_features = X.shape
    print(f"  Loaded X with shape: {X.shape} (samples × features)")

    # 2) Optional standardization (z-score features)
    if args.standardize:
        print("Standardizing features (z-score)...")
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        X_proc = (X - mean) / std
    else:
        X_proc = X

    # 3) Run PCA
    print(f"Running PCA with n_components={args.n_components}...")
    mean_vec, components, ev, evr, scores = run_pca(X_proc, args.n_components)

    # 4) Report variance explained
    print("Explained variance ratio (first components):")
    for i, r in enumerate(evr):
        print(f"  PC{i+1}: {r:.4f}")
    print(f"  Cumulative (PC1..PC{len(evr)}): {evr.cumsum()[-1]:.4f}")

    # 5) Save results
    print(f"Saving PCA outputs to: {args.outdir}")
    save_pca_results(args.outdir, mean_vec, components, ev, evr, scores, meta_rows)
    print("Done.")


if __name__ == "__main__":
    main()
