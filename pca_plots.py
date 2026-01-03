#!/usr/bin/env python3
"""
pca_plots.py

Visual diagnostics for PCA of synthetic trabecular images (voting3d + Option-1 2D exports).

Loads:
- scores_with_metadata.csv (from pca_analysis.py output)
- explained_variance_ratio.npy (from pca_analysis.py output)

Saves plots:
- explained_variance_ratio.png (bar)
- cumulative_explained_variance.png (line)
- PC scatter plots for (PC1,PC2), (PC1,PC3), (PC2,PC3) if available
  * colored by a continuous metadata field (e.g., bv_tv_3d, tau)
  * optionally colored by a categorical metadata field (e.g., closing_iters, k, alternate_close)

Notes:
- Defaults are tuned for the new voting3d metadata (NOT the old pattern/thickness fields).
"""

from pathlib import Path
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt


# ---------------------- loading ---------------------- #

def load_scores_with_metadata(path_csv: Path):
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
        fieldnames = reader.fieldnames or []

    if not rows:
        raise ValueError("scores_with_metadata.csv is empty")

    pc_names = [fn for fn in fieldnames if fn.startswith("PC")]
    if not pc_names:
        raise ValueError("No PC columns found (expected columns starting with 'PC').")

    scores = np.zeros((len(rows), len(pc_names)), dtype=float)
    for i, row in enumerate(rows):
        for j, pc in enumerate(pc_names):
            scores[i, j] = float(row[pc])

    return rows, scores, pc_names


def load_explained_variance_ratio(path_npy: Path) -> np.ndarray:
    path_npy = Path(path_npy)
    if not path_npy.exists():
        raise FileNotFoundError(f"Could not find explained_variance_ratio.npy at {path_npy}")
    return np.load(path_npy)


# ---------------------- plotting helpers ---------------------- #

def ensure_outdir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_explained_variance_ratio(evr: np.ndarray, outdir: Path):
    outdir = ensure_outdir(outdir)
    xs = np.arange(1, len(evr) + 1)

    plt.figure()
    plt.bar(xs, evr)
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA explained variance ratio")
    plt.xticks(xs)
    plt.tight_layout()
    plt.savefig(outdir / "explained_variance_ratio.png", dpi=300)
    plt.close()


def plot_cumulative_variance_ratio(evr: np.ndarray, outdir: Path):
    outdir = ensure_outdir(outdir)
    xs = np.arange(1, len(evr) + 1)
    cum = np.cumsum(evr)

    plt.figure()
    plt.plot(xs, cum, marker="o")
    plt.ylim(0, 1.01)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Cumulative explained variance")
    plt.xticks(xs)
    plt.tight_layout()
    plt.savefig(outdir / "cumulative_explained_variance.png", dpi=300)
    plt.close()


def _as_float_or_nan(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def plot_scatter_continuous(scores: np.ndarray,
                            pc_names: list[str],
                            meta_rows: list[dict],
                            field: str,
                            outdir: Path,
                            i: int,
                            j: int,
                            fname: str):
    outdir = ensure_outdir(outdir)
    if scores.shape[1] <= max(i, j):
        return

    vals = np.array([_as_float_or_nan(r.get(field, "")) for r in meta_rows], dtype=float)
    if np.all(np.isnan(vals)):
        print(f"Field '{field}' not usable as continuous (all NaN). Skipping {fname}.")
        return

    plt.figure()
    sc = plt.scatter(scores[:, i], scores[:, j], c=vals, s=10, alpha=0.8)
    plt.xlabel(pc_names[i])
    plt.ylabel(pc_names[j])
    plt.title(f"{pc_names[i]} vs {pc_names[j]} coloured by {field}")
    cb = plt.colorbar(sc)
    cb.set_label(field)
    plt.tight_layout()
    plt.savefig(outdir / fname, dpi=300)
    plt.close()


def categorical_colors(categories: list[str]):
    unique = sorted(set(categories))
    mapping = {cat: idx for idx, cat in enumerate(unique)}
    values = np.array([mapping[c] for c in categories], dtype=float)
    return values, mapping


def plot_scatter_categorical(scores: np.ndarray,
                             pc_names: list[str],
                             meta_rows: list[dict],
                             field: str,
                             outdir: Path,
                             i: int,
                             j: int,
                             fname: str,
                             max_legend_items: int = 25):
    outdir = ensure_outdir(outdir)
    if scores.shape[1] <= max(i, j):
        return

    cats = [str(r.get(field, "NA")) for r in meta_rows]
    cat_values, mapping = categorical_colors(cats)

    plt.figure()
    plt.scatter(scores[:, i], scores[:, j], c=cat_values, s=10, alpha=0.8)
    plt.xlabel(pc_names[i])
    plt.ylabel(pc_names[j])
    plt.title(f"{pc_names[i]} vs {pc_names[j]} coloured by {field}")

    # Legend (cap size to avoid huge legends)
    items = list(mapping.items())
    if len(items) <= max_legend_items:
        handles = []
        labels = []
        for cat, idx in items:
            handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=6))
            labels.append(cat)
        plt.legend(handles, labels, title=field, fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(outdir / fname, dpi=300)
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
        help="Optional explicit path to scores_with_metadata.csv (otherwise taken from pca-results-dir).",
    )
    p.add_argument(
        "--evr-npy",
        type=str,
        default=None,
        help="Optional explicit path to explained_variance_ratio.npy (otherwise taken from pca-results-dir).",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory to save plots (defaults to <pca-results-dir>/plots).",
    )

    # Updated defaults for voting3d pipeline
    p.add_argument(
        "--continuous-field",
        type=str,
        default="bv_tv_3d",
        help="Metadata field to treat as continuous for colouring (e.g., bv_tv_3d, tau).",
    )
    p.add_argument(
        "--category-field",
        type=str,
        default="closing_iters",
        help="Metadata field to treat as categorical for colouring (e.g., closing_iters, k, alternate_close).",
    )
    p.add_argument(
        "--no-categorical",
        action="store_true",
        help="Disable categorical scatter plots.",
    )
    p.add_argument(
        "--no-continuous",
        action="store_true",
        help="Disable continuous scatter plots.",
    )
    return p


def main():
    args = build_parser().parse_args()

    pca_dir = Path(args.pca_results_dir)
    scores_csv = Path(args.scores_csv) if args.scores_csv else (pca_dir / "scores_with_metadata.csv")
    evr_npy = Path(args.evr_npy) if args.evr_npy else (pca_dir / "explained_variance_ratio.npy")
    outdir = Path(args.outdir) if args.outdir else (pca_dir / "plots")

    print(f"Loading scores + metadata from: {scores_csv}")
    meta_rows, scores, pc_names = load_scores_with_metadata(scores_csv)

    print(f"Loading explained variance ratio from: {evr_npy}")
    evr = load_explained_variance_ratio(evr_npy)

    print("Plotting explained variance ratio...")
    plot_explained_variance_ratio(evr, outdir)

    print("Plotting cumulative explained variance...")
    plot_cumulative_variance_ratio(evr, outdir)

    # PC pairs to plot
    pairs = [(0, 1), (0, 2), (1, 2)]
    pairs = [(i, j) for (i, j) in pairs if len(pc_names) > max(i, j)]

    if not pairs:
        print("Not enough PCs for scatter plots (need at least 2).")
        print(f"Plots saved under: {outdir}")
        return

    if not args.no_continuous:
        for i, j in pairs:
            fname = f"scatter_{pc_names[i]}_{pc_names[j]}_by_{args.continuous_field}.png"
            print(f"Plotting {pc_names[i]} vs {pc_names[j]} by continuous field '{args.continuous_field}'...")
            plot_scatter_continuous(scores, pc_names, meta_rows, args.continuous_field, outdir, i, j, fname)

    if not args.no_categorical:
        for i, j in pairs:
            fname = f"scatter_{pc_names[i]}_{pc_names[j]}_by_{args.category_field}.png"
            print(f"Plotting {pc_names[i]} vs {pc_names[j]} by categorical field '{args.category_field}'...")
            plot_scatter_categorical(scores, pc_names, meta_rows, args.category_field, outdir, i, j, fname)

    print(f"Plots saved under: {outdir}")


if __name__ == "__main__":
    main()
