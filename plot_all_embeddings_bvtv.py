#!/usr/bin/env python3
"""
Plot Component 1 vs Component 2 coloured by BV/TV for all 5 reduction methods.
Loads the *_quantum_ready.npz files saved by dim_reduction_pipeline_v2.py.

Usage:
  python plot_all_embeddings_bvtv.py \
      --features-dir output/v8_dataset/features \
      --outfile output/v8_dataset/features/all_methods_bvtv.png
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = [
    ("pca",          "PCA"),
    ("rp_gaussian",  "RP Gaussian"),
    ("rp_sparse",    "RP Sparse"),
    ("pls",          "PLS"),
    ("umap",         "UMAP"),
]

BVTV_LABEL_IDX = 0  # first column in Y is BV/TV


def load_method(features_dir: Path, key: str):
    """Load Z_train and Y_train from the quantum-ready npz file."""
    fname = features_dir / f"{key}_quantum_ready.npz"
    if not fname.exists():
        print(f"  WARNING: {fname} not found, skipping")
        return None, None
    data = np.load(fname)
    # Use 0-1 scaled if available, otherwise fall back to Z_train
    Z = data["Z_train_01"] if "Z_train_01" in data else data["Z_train"]
    Y = data["Y_train"]
    return Z, Y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str,
                        default="output/v8_dataset/features",
                        help="Directory containing *_quantum_ready.npz files")
    parser.add_argument("--outfile", type=str,
                        default=None,
                        help="Output PNG path (default: <features-dir>/all_methods_bvtv.png)")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    outfile = Path(args.outfile) if args.outfile else features_dir / "all_methods_bvtv.png"

    # Filter to methods that exist
    available = []
    for key, label in METHODS:
        Z, Y = load_method(features_dir, key)
        if Z is not None:
            available.append((key, label, Z, Y))

    if not available:
        print("ERROR: No method files found. Check --features-dir path.")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    # Shared colour limits from BV/TV range
    all_bvtv = np.concatenate([Y[:, BVTV_LABEL_IDX] for _, _, _, Y in available])
    vmin, vmax = all_bvtv.min(), all_bvtv.max()

    for ax, (key, label, Z, Y) in zip(axes, available):
        bvtv = Y[:, BVTV_LABEL_IDX]
        sc = ax.scatter(Z[:, 0], Z[:, 1],
                        c=bvtv, cmap="viridis", s=18, alpha=0.7,
                        vmin=vmin, vmax=vmax, edgecolors="none")
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Component 1", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Component 2", fontsize=10)
        ax.tick_params(labelsize=8)

    # Single shared colourbar on the right
    cbar = fig.colorbar(sc, ax=axes, location="right", shrink=0.85, pad=0.02)
    cbar.set_label("BV/TV", fontsize=11)

    fig.suptitle("Dimensionality Reduction — 2D Embeddings Coloured by BV/TV",
                 fontsize=14, fontweight="bold", y=1.02)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


if __name__ == "__main__":
    main()