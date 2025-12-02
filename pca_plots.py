#!/usr/bin/env python3
"""
pca_plots.py

Visual diagnostics for PCA of synthetic trabecular images.

- Loads scores_with_metadata.csv from pca_analysis.py output
- Plots:
    * Explained variance ratio (bar plot)
    * PC1 vs PC2 coloured by pattern (categorical, with colours/markers)
    * PC1 vs PC2 coloured by a continuous field (e.g. thickness)
    * Optional 3D PC1–PC3 scatter coloured by pattern
"""

from pathlib import Path
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_scores_with_metadata(path_csv):
    """
    Load scores + metadata from CSV.

    Returns
    -------
    meta_rows : list[dict]
    scores : np.ndarray (n_samples, n_components)
    pc_names : list[str] e.g. ["PC1", "PC2", ...]
    """
    path_csv = Path(path_csv)
    if not path_csv.exists():
        raise FileNotFoundError(f"Could not find scores CSV: {path_csv}")

    with open(path_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("scores_with_metadata.csv is empty")

    # Identify PC columns
    all_fields = reader.fieldnames
    pc_names = [fn for fn in all_fields if fn.startswith("PC")]
    if not pc_names:
        raise ValueError("No PC columns found (expected columns starting with 'PC').")

    # Build scores array
    scores = np.zeros((len(rows), len(pc_names)), dtype=float)
    for i, row in enumerate(rows):
        for j, pc in enumerate(pc_names):
            scores[i, j] = float(row[pc])

    return rows, scores, pc_names


def load_explained_variance_ratio(path_npy):
    """
    Load explained variance ratio from .npy file produced by pca_analysis.py.
    """
    path_npy = Path(path_npy)
    if not path_npy.exists():
        raise FileNotFoundError(f"Could not find explained_variance_ratio.npy at {path_npy}")
    return np.load(path_npy)


# ---------------------- plotting helpers ---------------------- #

def ensure_outdir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_explained_variance_ratio(evr, outdir):
    outdir = ensure_outdir(outdir)
    n = len(evr)
    xs = np.arange(1, n + 1)

    plt.figure()
    plt.bar(xs, evr)
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA explained variance ratio")
    plt.xticks(xs)
    plt.tight_layout()
    plt.savefig(outdir / "explained_variance_ratio.png", dpi=300)
    plt.close()


def plot_pc_scatter_categorical(scores, meta_rows, pc_names, category_field, outdir, evr=None):
    """
    PC1 vs PC2 scatter plot coloured by a categorical field (e.g., pattern),
    using different colours/markers and a clear legend.
    """
    outdir = ensure_outdir(outdir)
    if scores.shape[1] < 2:
        print("Not enough PCs (need at least 2) for scatter plot.")
        return

    # Extract category labels, e.g. 'grid', 'vertical', 'horizontal'
    cats = np.array([row.get(category_field, "NA") for row in meta_rows])
    unique_cats = np.unique(cats)

    # Define consistent colours/markers
    base_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    base_markers = ["o", "s", "^", "D", "P"]

    color_map = {}
    marker_map = {}
    for i, c in enumerate(unique_cats):
        color_map[c] = base_colors[i % len(base_colors)]
        marker_map[c] = base_markers[i % len(base_markers)]

    plt.figure()

    for c in unique_cats:
        mask = (cats == c)
        plt.scatter(
            scores[mask, 0],
            scores[mask, 1],
            c=color_map[c],
            marker=marker_map[c],
            s=15,
            alpha=0.8,
            label=str(c),
        )

    # Axis labels including explained variance if available
    if evr is not None and len(evr) >= 2:
        xlab = f"{pc_names[0]} ({evr[0] * 100:.1f}% var)"
        ylab = f"{pc_names[1]} ({evr[1] * 100:.1f}% var)"
    else:
        xlab = pc_names[0]
        ylab = pc_names[1]

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(f"{pc_names[0]} vs {pc_names[1]} coloured by {category_field}")
    plt.legend(title=category_field, fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / f"scatter_{pc_names[0]}_{pc_names[1]}_by_{category_field}.png", dpi=300)
    plt.close()


def plot_pc_scatter_continuous(scores, meta_rows, pc_names, field, outdir, evr=None):
    """
    PC1 vs PC2 scatter coloured by a continuous field (e.g., thickness).
    """
    outdir = ensure_outdir(outdir)
    if scores.shape[1] < 2:
        print("Not enough PCs (need at least 2) for scatter plot.")
        return

    values = []
    for row in meta_rows:
        v = row.get(field, "")
        try:
            values.append(float(v))
        except ValueError:
            values.append(float("nan"))

    values = np.array(values, dtype=float)
    if np.all(np.isnan(values)):
        print(f"Field '{field}' not usable as continuous (all NaN). Skipping.")
        return

    plt.figure()
    sc = plt.scatter(scores[:, 0], scores[:, 1], c=values, s=10, alpha=0.8)
    # Axis labels including explained variance if available
    if evr is not None and len(evr) >= 2:
        xlab = f"{pc_names[0]} ({evr[0] * 100:.1f}% var)"
        ylab = f"{pc_names[1]} ({evr[1] * 100:.1f}% var)"
    else:
        xlab = pc_names[0]
        ylab = pc_names[1]

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(f"{pc_names[0]} vs {pc_names[1]} coloured by {field}")
    cb = plt.colorbar(sc)
    cb.set_label(field)
    plt.tight_layout()
    plt.savefig(outdir / f"scatter_{pc_names[0]}_{pc_names[1]}_by_{field}.png", dpi=300)
    plt.close()


def plot_pc_scatter_3d_categorical(scores, meta_rows, pc_names, category_field, outdir):
    """
    3D PC1–PC3 scatter plot coloured by a categorical field.
    """
    outdir = ensure_outdir(outdir)
    if scores.shape[1] < 3:
        print("Not enough PCs (need at least 3) for 3D scatter plot.")
        return

    cats = np.array([row.get(category_field, "NA") for row in meta_rows])
    unique_cats = np.unique(cats)

    base_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    base_markers = ["o", "s", "^", "D", "P"]

    color_map = {}
    marker_map = {}
    for i, c in enumerate(unique_cats):
        color_map[c] = base_colors[i % len(base_colors)]
        marker_map[c] = base_markers[i % len(base_markers)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for c in unique_cats:
        mask = (cats == c)
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            scores[mask, 2],
            c=color_map[c],
            marker=marker_map[c],
            s=15,
            alpha=0.8,
            label=str(c),
        )

    ax.set_xlabel(pc_names[0])
    ax.set_ylabel(pc_names[1])
    ax.set_zlabel(pc_names[2])
    ax.set_title(f"3D PCA coloured by {category_field}")
    ax.legend(title=category_field, fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / f"scatter3d_{pc_names[0]}_{pc_names[1]}_{pc_names[2]}_by_{category_field}.png", dpi=300)
    plt.close()


# ---------------------- CLI ---------------------- #

def build_parser():
    p = argparse.ArgumentParser(
        description="Plot PCA diagnostics from scores_with_metadata.csv and explained_variance_ratio.npy."
    )
    p.add_argument(
        "--pca-results-dir",
        type=str,
        default="data/pca_results",
        help="Directory where pca_analysis.py saved its outputs.",
    )
    p.add_argument(
        "--scores-csv",
        type=str,
        default=None,
        help="Optional explicit path to scores_with_metadata.csv "
             "(otherwise taken from pca-results-dir).",
    )
    p.add_argument(
        "--evr-npy",
        type=str,
        default=None,
        help="Optional explicit path to explained_variance_ratio.npy "
             "(otherwise taken from pca-results-dir).",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory to save plots (defaults to <pca-results-dir>/plots).",
    )
    p.add_argument(
        "--continuous-field",
        type=str,
        default="thickness_um",
        help="Metadata field to treat as continuous for colouring PC scatter (e.g., thickness_um).",
    )
    p.add_argument(
        "--category-field",
        type=str,
        default="pattern",
        help="Metadata field to treat as categorical for colouring PC scatter (e.g., pattern).",
    )
    p.add_argument(
        "--plot-3d",
        action="store_true",
        help="If set, also generate a 3D PC1–PC3 scatter plot coloured by the category field.",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    pca_dir = Path(args.pca_results_dir)
    if args.scores_csv is None:
        scores_csv = pca_dir / "scores_with_metadata.csv"
    else:
        scores_csv = Path(args.scores_csv)

    if args.evr_npy is None:
        evr_npy = pca_dir / "explained_variance_ratio.npy"
    else:
        evr_npy = Path(args.evr_npy)

    if args.outdir is None:
        outdir = pca_dir / "plots"
    else:
        outdir = Path(args.outdir)

    print(f"Loading scores + metadata from: {scores_csv}")
    meta_rows, scores, pc_names = load_scores_with_metadata(scores_csv)

    print(f"Loading explained variance ratio from: {evr_npy}")
    evr = load_explained_variance_ratio(evr_npy)

    print("Plotting explained variance ratio...")
    plot_explained_variance_ratio(evr, outdir)

    print(f"Plotting {pc_names[0]} vs {pc_names[1]} by categorical field '{args.category_field}'...")
    plot_pc_scatter_categorical(scores, meta_rows, pc_names, args.category_field, outdir, evr=evr)

    print(f"Plotting {pc_names[0]} vs {pc_names[1]} by continuous field '{args.continuous_field}'...")
    plot_pc_scatter_continuous(scores, meta_rows, pc_names, args.continuous_field, outdir, evr=evr)

    if args.plot_3d:
        print(f"Plotting 3D PCA by categorical field '{args.category_field}'...")
        plot_pc_scatter_3d_categorical(scores, meta_rows, pc_names, args.category_field, outdir)

    print(f"Plots saved under: {outdir}")


if __name__ == "__main__":
    main()
