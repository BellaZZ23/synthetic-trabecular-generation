#!/usr/bin/env python3
"""
run_5x2cv_test.py

Runs the Dietterich 5×2 CV paired t-test using existing precomputed
kernel matrices. Also runs Nadeau-Bengio for comparison. Produces an
updated bar chart with corrected significance annotations.

Usage:
  python run_5x2cv_test.py \
      --features-dir output/v8_dataset/features \
      --kernel-dir output/v8_qsvm_tight \
      --task classify-bvtv
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon as wilcoxon_test
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL_KEYS = ["BVTV", "TbTh_um_p50", "TbN_per_mm", "TbSp_um_p50"]
METHODS = ["pca", "rp_gaussian", "pls", "umap"]
LABELS  = {"pca": "PCA", "rp_gaussian": "RP Gaussian",
           "pls": "PLS", "umap": "UMAP"}


def create_labels(Y, task):
    if task == "classify-bvtv":
        vals = Y[:, LABEL_KEYS.index("BVTV")]
        return (vals >= np.median(vals)).astype(int)
    elif task == "classify-tbn":
        vals = Y[:, LABEL_KEYS.index("TbN_per_mm")]
        return (vals >= np.median(vals)).astype(int)
    raise ValueError(f"Unknown task: {task}")


# ═══════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════

def dietterich_5x2cv_test(K, X_01, y, n_repeats=5):
    """
    5×2 CV paired t-test (Dietterich, 1998).

    For each of 5 repeats, split data 50/50 twice (2-fold CV).
    Purpose-built for classifier comparison with proper df.
    """
    diffs = []       # d_i^(j) for repeat i, fold j
    variances = []   # s_i^2 per repeat

    for rep in range(n_repeats):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5,
                                          random_state=rep)
        idx_a, idx_b = next(splitter.split(X_01, y))

        rep_diffs = []
        for train_idx, test_idx in [(idx_a, idx_b), (idx_b, idx_a)]:
            # Classical
            svm_c = SVC(kernel="rbf", gamma="scale", C=1.0, random_state=42)
            svm_c.fit(X_01[train_idx], y[train_idx])
            acc_c = accuracy_score(y[test_idx], svm_c.predict(X_01[test_idx]))

            # Quantum (precomputed)
            svm_q = SVC(kernel="precomputed", C=1.0, random_state=42)
            svm_q.fit(K[np.ix_(train_idx, train_idx)], y[train_idx])
            acc_q = accuracy_score(y[test_idx],
                                   svm_q.predict(K[np.ix_(test_idx, train_idx)]))

            rep_diffs.append(acc_q - acc_c)

        diffs.append(rep_diffs)
        mean_d = np.mean(rep_diffs)
        var_d = sum((d - mean_d) ** 2 for d in rep_diffs)
        variances.append(var_d)

    # t = d_1^(1) / sqrt( (1/5) * sum(s_i^2) )
    pooled_var = np.mean(variances)
    if pooled_var == 0:
        return {"p_value": 1.0, "t_stat": 0.0, "df": 5,
                "mean_diff": 0.0, "all_diffs": [], "error": "zero variance"}

    t_stat = diffs[0][0] / np.sqrt(pooled_var)
    df = 5
    p_value = 2 * stats.t.sf(abs(t_stat), df)

    all_diffs = [d for rep in diffs for d in rep]
    return {
        "t_stat": float(t_stat),
        "df": df,
        "p_value": float(p_value),
        "mean_diff": float(np.mean(all_diffs)),
        "all_diffs": [float(d) for d in all_diffs],
        "q_wins": sum(1 for d in all_diffs if d > 0),
        "n_folds": len(all_diffs),
    }


def nadeau_bengio_test(classical, quantum, n_splits=5, n_repeats=5, n_total=500):
    """Corrected paired t-test (Nadeau & Bengio, 2003)."""
    diff = quantum - classical
    n_folds = len(diff)
    n_test = n_total // n_splits
    n_train = n_total - n_test
    mean_diff = diff.mean()
    var_diff = diff.var(ddof=1)
    corrected_var = (1 / n_folds + n_test / n_train) * var_diff
    t_stat = mean_diff / np.sqrt(corrected_var)
    p_value = 2 * stats.t.sf(abs(t_stat), n_folds - 1)
    return {"t_stat": float(t_stat), "p_value": float(p_value),
            "mean_diff": float(mean_diff), "df": n_folds - 1}


# ═══════════════════════════════════════════════════════════════════════
# BAR CHART
# ═══════════════════════════════════════════════════════════════════════

def significance_label(p):
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "n.s."


def plot_results(results, outdir, task_label):
    """
    Bar chart with error bars and THREE significance annotations:
      - Wilcoxon (uncorrected)
      - Nadeau-Bengio (corrected)
      - 5×2 CV (gold standard)
    """
    methods = [m for m in METHODS if m in results]
    n = len(methods)

    fig, ax = plt.subplots(figsize=(max(10, n * 2.5), 7))
    x = np.arange(n)
    w = 0.35

    c_acc = [results[m]["classical_mean"] for m in methods]
    c_std = [results[m]["classical_std"] for m in methods]
    q_acc = [results[m]["quantum_mean"] for m in methods]
    q_std = [results[m]["quantum_std"] for m in methods]

    b1 = ax.bar(x - w/2, c_acc, w, yerr=c_std, capsize=4,
                label="Classical SVM (RBF)", color="#4472C4", alpha=0.85)
    b2 = ax.bar(x + w/2, q_acc, w, yerr=q_std, capsize=4,
                label="Quantum SVM (ZZ)", color="#ED7D31", alpha=0.85)

    # Value labels on bars
    for bars in [b1, b2]:
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width()/2, h + 0.025,
                        f"{h:.3f}", ha="center", fontsize=8)

    # Significance brackets
    for i, m in enumerate(methods):
        r = results[m]
        y_top = max(c_acc[i] + c_std[i], q_acc[i] + q_std[i]) + 0.06
        bracket_h = 0.015

        # Bracket line
        ax.plot([i - w/2, i - w/2, i + w/2, i + w/2],
                [y_top, y_top + bracket_h, y_top + bracket_h, y_top],
                color="black", linewidth=1.0)

        # 5×2 CV label (primary, bold)
        p_5x2 = r["5x2cv"]["p_value"]
        sig_5x2 = significance_label(p_5x2)
        ax.text(i, y_top + bracket_h + 0.008,
                f"{sig_5x2}",
                ha="center", fontsize=11, fontweight="bold")

        # Smaller annotation with all three p-values
        p_wilcox = r["wilcoxon"]["p_value"]
        p_nb = r["nadeau_bengio"]["p_value"]
        detail = f"W:{significance_label(p_wilcox)} NB:{significance_label(p_nb)} 5×2:{sig_5x2}"
        ax.text(i, y_top + bracket_h + 0.038,
                detail, ha="center", fontsize=6.5, color="gray")

    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Classical vs Quantum SVM — {task_label}", fontsize=13,
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods], fontsize=10)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="Random baseline")
    ax.legend(loc="lower left", fontsize=9)

    # Significance key
    fig.text(0.98, 0.02,
             "n.s.: p≥0.05  *: p<0.05  **: p<0.01  ***: p<0.001\n"
             "W = Wilcoxon  NB = Nadeau-Bengio  5×2 = Dietterich 5×2 CV",
             ha="right", fontsize=7.5, color="gray", style="italic")

    plt.tight_layout()
    out_path = outdir / "accuracy_comparison_corrected.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  Saved figure: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=str, default="output/v8_dataset/features")
    p.add_argument("--kernel-dir", type=str, default="output/v8_qsvm_tight")
    p.add_argument("--task", type=str, default="classify-bvtv")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output dir for figure (default: kernel-dir)")
    args = p.parse_args()

    features_dir = Path(args.features_dir)
    kernel_dir = Path(args.kernel_dir)
    outdir = Path(args.outdir) if args.outdir else kernel_dir

    task_labels = {
        "classify-bvtv": "BV/TV Classification",
        "classify-tbn": "Tb.N Classification",
    }

    print("=" * 70)
    print("  5×2 CV Paired t-test + Nadeau-Bengio + Wilcoxon")
    print(f"  Task: {args.task}")
    print("=" * 70)

    all_results = {}

    for method in METHODS:
        npz_path = features_dir / f"{method}_quantum_ready.npz"
        kernel_path = kernel_dir / f"kernel_{method}.npy"

        if not npz_path.exists():
            print(f"\n  {method}: features not found, skipping")
            continue
        if not kernel_path.exists():
            print(f"\n  {method}: kernel not found, skipping")
            continue

        # Load data
        data = np.load(npz_path, allow_pickle=True)
        X_01 = np.vstack([
            data["Z_train_01"] if "Z_train_01" in data else data["Z_train"],
            data["Z_test_01"] if "Z_test_01" in data else data["Z_test"],
        ])
        Y_all = np.vstack([data["Y_train"], data["Y_test"]])
        y = create_labels(Y_all, args.task)
        K = np.load(kernel_path)

        print(f"\n{'─' * 60}")
        print(f"  {method.upper()} ({len(y)} samples)")
        print(f"{'─' * 60}")

        # ── 5×5 repeated CV (for means, stds, Wilcoxon) ──────────
        rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
        c_scores, q_scores = [], []

        for tr, te in rkf.split(K, y):
            svm_c = SVC(kernel="rbf", gamma="scale", C=1.0, random_state=42)
            svm_c.fit(X_01[tr], y[tr])
            c_scores.append(accuracy_score(y[te], svm_c.predict(X_01[te])))

            svm_q = SVC(kernel="precomputed", C=1.0, random_state=42)
            svm_q.fit(K[np.ix_(tr, tr)], y[tr])
            q_scores.append(accuracy_score(y[te], svm_q.predict(K[np.ix_(te, tr)])))

        c_scores = np.array(c_scores)
        q_scores = np.array(q_scores)
        diffs = q_scores - c_scores

        print(f"  Classical: {c_scores.mean():.4f} ± {c_scores.std():.4f}")
        print(f"  Quantum:   {q_scores.mean():.4f} ± {q_scores.std():.4f}")
        print(f"  Q wins: {(diffs > 0).sum()}/25")

        # ── Test 1: Wilcoxon (uncorrected) ────────────────────────
        if np.all(diffs == 0):
            w_stat, w_p = 0.0, 1.0
        else:
            w_stat, w_p = wilcoxon_test(diffs, alternative="two-sided")
        print(f"\n  Wilcoxon:       p = {w_p:.6f}  {significance_label(w_p)}")

        # ── Test 2: Nadeau-Bengio ─────────────────────────────────
        nb = nadeau_bengio_test(c_scores, q_scores, n_total=len(y))
        print(f"  Nadeau-Bengio:  p = {nb['p_value']:.6f}  {significance_label(nb['p_value'])}")

        # ── Test 3: 5×2 CV (Dietterich) ──────────────────────────
        fivex2 = dietterich_5x2cv_test(K, X_01, y)
        print(f"  5×2 CV:         p = {fivex2['p_value']:.6f}  {significance_label(fivex2['p_value'])}")
        print(f"    t = {fivex2['t_stat']:.3f}, df = {fivex2['df']}, "
              f"mean diff = {fivex2['mean_diff']:+.4f}, "
              f"Q wins = {fivex2['q_wins']}/{fivex2['n_folds']}")

        all_results[method] = {
            "classical_mean": float(c_scores.mean()),
            "classical_std": float(c_scores.std()),
            "quantum_mean": float(q_scores.mean()),
            "quantum_std": float(q_scores.std()),
            "gap": float(diffs.mean()),
            "wilcoxon": {"stat": float(w_stat), "p_value": float(w_p)},
            "nadeau_bengio": {"t_stat": nb["t_stat"], "p_value": nb["p_value"]},
            "5x2cv": {"t_stat": fivex2["t_stat"], "p_value": fivex2["p_value"],
                       "mean_diff": fivex2["mean_diff"],
                       "q_wins": fivex2["q_wins"]},
        }

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY — {args.task}")
    print(f"{'=' * 80}")
    print(f"  {'Method':<14} {'Gap':>7} {'Wilcoxon':>12} {'N-B corrected':>15} {'5×2 CV':>12}")
    print(f"  {'─' * 64}")
    for m in METHODS:
        if m not in all_results:
            continue
        r = all_results[m]
        g = r["gap"]
        wp = r["wilcoxon"]["p_value"]
        np_ = r["nadeau_bengio"]["p_value"]
        fp = r["5x2cv"]["p_value"]
        print(f"  {LABELS[m]:<14} {g:+.3f}   "
              f"p={wp:.4f} {significance_label(wp):>4}  "
              f"p={np_:.4f} {significance_label(np_):>4}  "
              f"p={fp:.4f} {significance_label(fp):>4}")
    print(f"{'=' * 80}")

    # ── Plot ──────────────────────────────────────────────────────
    if all_results:
        plot_results(all_results, outdir, task_labels.get(args.task, args.task))

    # ── Save JSON ─────────────────────────────────────────────────
    json_path = outdir / "statistical_tests_corrected.json"
    with open(json_path, "w") as f:
        json.dump({"task": args.task, "results": all_results}, f, indent=2)
    print(f"  Saved results: {json_path}")


if __name__ == "__main__":
    main()