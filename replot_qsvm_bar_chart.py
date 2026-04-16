#!/usr/bin/env python3
"""
Re-plot QSVM accuracy comparison from saved qsvm_tight_results.json
and compute Wilcoxon significance directly from saved fold-score .npy files.

Usage:
  python replot_qsvm_bar_chart.py ^
      --results-json output/v8_qsvm_tight/qsvm_tight_results.json ^
      --results-dir output/v8_qsvm_tight ^
      --outfile output/v8_qsvm_tight/accuracy_comparison_tight_replot.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


DISPLAY_NAMES = {
    "pca": "PCA",
    "rp_gaussian": "RP Gaussian",
    "rp_sparse": "RP Sparse",
    "pls": "PLS",
    "umap": "UMAP",
}


def p_to_sig(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def load_fold_scores(results_dir: Path, method: str):
    c_path = results_dir / f"{method}_classical_fold_scores.npy"
    q_path = results_dir / f"{method}_quantum_fold_scores.npy"

    if not c_path.exists() or not q_path.exists():
        return None, None

    classical = np.load(c_path)
    quantum = np.load(q_path)

    if len(classical) != len(quantum):
        raise ValueError(
            f"Fold count mismatch for {method}: "
            f"{len(classical)} classical vs {len(quantum)} quantum"
        )
    return classical, quantum


def compute_wilcoxon_p(results_dir: Path, method: str, alternative: str = "two-sided"):
    classical, quantum = load_fold_scores(results_dir, method)
    if classical is None or quantum is None:
        return None

    diff = quantum - classical
    try:
        _, p = wilcoxon(diff, alternative=alternative)
        return float(p)
    except ValueError:
        return None


def draw_sig_bracket(ax, x1, x2, y, h, text, fontsize=9):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=1.2, c="black", clip_on=False)
    ax.text(
        (x1 + x2) / 2,
        y + h + 0.003,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        clip_on=False,
    )


def plot_from_json(results_json: Path, results_dir: Path, outfile: Path, dpi: int = 300) -> None:
    with open(results_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload["results"]
    methods = list(results.keys())
    method_labels = [DISPLAY_NAMES.get(m, m.upper()) for m in methods]

    c_acc = np.array([results[m]["classical"]["accuracy"] for m in methods], dtype=float)
    c_std = np.array([results[m]["classical"]["accuracy_std"] for m in methods], dtype=float)

    q_acc = np.array([
        results[m]["quantum"]["accuracy"] if results[m]["quantum"]["accuracy"] is not None else np.nan
        for m in methods
    ], dtype=float)
    q_std = np.array([
        results[m]["quantum"]["accuracy_std"] if results[m]["quantum"]["accuracy_std"] is not None else 0.0
        for m in methods
    ], dtype=float)

    x = np.arange(len(methods))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(methods) * 2.2), 6.8))

    b1 = ax.bar(
        x - w / 2, c_acc, w, yerr=c_std, capsize=4,
        label="Classical SVM (RBF)", color="#4472C4", alpha=0.85
    )
    b2 = ax.bar(
        x + w / 2, q_acc, w, yerr=q_std, capsize=4,
        label="Quantum SVM (ZZ)", color="#ED7D31", alpha=0.85
    )

    # Numeric labels above error bars
    label_gap = 0.012
    for bars, errs in ((b1, c_std), (b2, q_std)):
        for bar, err in zip(bars, errs):
            h = bar.get_height()
            if np.isfinite(h) and h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + err + label_gap,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    clip_on=False,
                )

    # Significance brackets
    bracket_height = 0.010
    bracket_gap = 0.045
    bracket_tops = []
    shown_sig = False

    for i, method in enumerate(methods):
        if not np.isfinite(q_acc[i]):
            continue

        p_val = compute_wilcoxon_p(results_dir, method, alternative="two-sided")
        sig_text = p_to_sig(p_val)

        # Skip non-significant ones if you want a cleaner figure:
        # if sig_text == "ns":
        #     continue

        pair_top = max(c_acc[i] + c_std[i], q_acc[i] + q_std[i])
        y = pair_top + bracket_gap

        x1 = x[i] - w / 2
        x2 = x[i] + w / 2

        draw_sig_bracket(ax, x1, x2, y, bracket_height, sig_text, fontsize=9)
        bracket_tops.append(y + bracket_height + 0.015)
        shown_sig = True

    # Dynamic ylim with enough room for bars, labels, and brackets
    top_candidates = []
    for mean, std in zip(c_acc, c_std):
        if np.isfinite(mean):
            top_candidates.append(mean + std + label_gap)
    for mean, std in zip(q_acc, q_std):
        if np.isfinite(mean):
            top_candidates.append(mean + std + label_gap)

    if shown_sig:
        top_candidates.extend(bracket_tops)

    ymax_needed = max(top_candidates) if top_candidates else 1.0
    ax.set_ylim(0, min(1.15, ymax_needed + 0.03))

    task = payload.get("task", "classification")
    title_task = {
        "classify-bvtv": "BV/TV Classification",
        "classify-tbn": "Tb.N Classification",
        "classify-tbsp": "Tb.Sp Classification",
    }.get(task, task)

    ax.set_ylabel("Accuracy")
    ax.set_title(f"Classical vs Quantum SVM — {title_task}", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=15, ha="right")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="Random baseline")
    ax.legend()

    # Optional significance note
    fig.text(
        0.99, 0.01,
        "ns: p≥0.05, *: p<0.05, **: p<0.01, ***: p<0.001",
        ha="right", va="bottom", fontsize=8
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1], pad=1.2)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {outfile}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-json", type=str, required=True,
                   help="Path to qsvm_tight_results.json")
    p.add_argument("--results-dir", type=str, required=True,
                   help="Directory containing *_classical_fold_scores.npy and *_quantum_fold_scores.npy")
    p.add_argument("--outfile", type=str, required=True,
                   help="Path to output PNG")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    plot_from_json(
        Path(args.results_json),
        Path(args.results_dir),
        Path(args.outfile),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()