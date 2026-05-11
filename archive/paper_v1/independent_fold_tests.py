#!/usr/bin/env python3
"""
run_nadeau_bengio.py

1. Loads your precomputed kernel matrices (kernel_*.npy)
2. Re-runs the same 5×5 CV to extract per-fold scores
3. Saves fold scores as .npy files
4. Runs the Nadeau-Bengio corrected paired t-test

Usage:
  python run_nadeau_bengio.py \
      --features-dir output/v8_dataset/features \
      --kernel-dir output/v8_qsvm_tight \
      --task classify-bvtv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

LABEL_KEYS = ["BVTV", "TbTh_um_p50", "TbN_per_mm", "TbSp_um_p50"]
METHODS = ["pca", "rp_gaussian", "pls", "umap"]


def create_labels(Y, task):
    if task == "classify-bvtv":
        vals = Y[:, LABEL_KEYS.index("BVTV")]
        return (vals >= np.median(vals)).astype(int)
    elif task == "classify-tbn":
        vals = Y[:, LABEL_KEYS.index("TbN_per_mm")]
        return (vals >= np.median(vals)).astype(int)
    raise ValueError(f"Unknown task: {task}")


def nadeau_bengio_corrected_ttest(classical, quantum, n_splits=5, n_repeats=5, n_total=500):
    """Corrected paired t-test (Nadeau & Bengio, 2003)."""
    diff = quantum - classical
    n_folds = len(diff)
    n_test = n_total // n_splits
    n_train = n_total - n_test

    mean_diff = diff.mean()
    var_diff = diff.var(ddof=1)

    # Correction: inflates variance to account for fold overlap
    corrected_var = (1 / n_folds + n_test / n_train) * var_diff
    t_stat = mean_diff / np.sqrt(corrected_var)
    df = n_folds - 1
    p_value = 2 * stats.t.sf(abs(t_stat), df)

    return {
        "mean_diff": mean_diff,
        "t_stat": t_stat,
        "df": df,
        "p_value": p_value,
        "variance_inflation": (1 / n_folds + n_test / n_train) / (1 / n_folds),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=str, default="output/v8_dataset/features")
    p.add_argument("--kernel-dir", type=str, default="output/v8_qsvm_tight")
    p.add_argument("--task", type=str, default="classify-bvtv")
    p.add_argument("--cv-seed", type=int, default=42)
    args = p.parse_args()

    features_dir = Path(args.features_dir)
    kernel_dir = Path(args.kernel_dir)

    print("=" * 70)
    print("  Nadeau-Bengio Corrected Paired t-test")
    print(f"  Task: {args.task}")
    print("=" * 70)

    for method in METHODS:
        # ── Load features ─────────────────────────────────────────
        npz_path = features_dir / f"{method}_quantum_ready.npz"
        kernel_path = kernel_dir / f"kernel_{method}.npy"

        if not npz_path.exists():
            print(f"\n  {method}: features not found ({npz_path}), skipping")
            continue
        if not kernel_path.exists():
            print(f"\n  {method}: kernel not found ({kernel_path}), skipping")
            continue

        data = np.load(npz_path, allow_pickle=True)
        X_01 = np.vstack([
            data["Z_train_01"] if "Z_train_01" in data else data["Z_train"],
            data["Z_test_01"] if "Z_test_01" in data else data["Z_test"],
        ])
        Y_all = np.vstack([data["Y_train"], data["Y_test"]])
        y = create_labels(Y_all, args.task)
        K = np.load(kernel_path)

        n_total = len(y)
        print(f"\n{'─' * 60}")
        print(f"  {method.upper()} ({n_total} samples, kernel {K.shape})")
        print(f"{'─' * 60}")

        # ── Run identical 5×5 CV for both ─────────────────────────
        rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
                                       random_state=args.cv_seed)

        c_scores = []
        q_scores = []

        for fold_idx, (tr, te) in enumerate(rkf.split(K, y)):
            # Classical
            svm_c = SVC(kernel="rbf", gamma="scale", C=1.0, random_state=args.cv_seed)
            svm_c.fit(X_01[tr], y[tr])
            c_scores.append(accuracy_score(y[te], svm_c.predict(X_01[te])))

            # Quantum
            svm_q = SVC(kernel="precomputed", C=1.0, random_state=args.cv_seed)
            svm_q.fit(K[np.ix_(tr, tr)], y[tr])
            q_scores.append(accuracy_score(y[te], svm_q.predict(K[np.ix_(te, tr)])))

        c_scores = np.array(c_scores)
        q_scores = np.array(q_scores)

        # ── Save fold scores ──────────────────────────────────────
        np.save(kernel_dir / f"{method}_classical_fold_scores.npy", c_scores)
        np.save(kernel_dir / f"{method}_quantum_fold_scores.npy", q_scores)

        # ── Print per-fold comparison ─────────────────────────────
        diffs = q_scores - c_scores
        q_wins = (diffs > 0).sum()
        ties = (diffs == 0).sum()
        c_wins = (diffs < 0).sum()

        print(f"  Classical: {c_scores.mean():.4f} ± {c_scores.std():.4f}")
        print(f"  Quantum:   {q_scores.mean():.4f} ± {q_scores.std():.4f}")
        print(f"  Per-fold:  Q wins {q_wins}, C wins {c_wins}, ties {ties}")

        # ── Uncorrected Wilcoxon ──────────────────────────────────
        from scipy.stats import wilcoxon as wilcoxon_test
        if np.all(diffs == 0):
            w_p = 1.0
        else:
            _, w_p = wilcoxon_test(diffs, alternative="two-sided")
        print(f"\n  Wilcoxon (uncorrected):     p = {w_p:.6f}")

        # ── Nadeau-Bengio corrected ───────────────────────────────
        nb = nadeau_bengio_corrected_ttest(
            c_scores, q_scores,
            n_splits=5, n_repeats=5, n_total=n_total,
        )

        sig = "***" if nb["p_value"] < 0.001 else \
              "**"  if nb["p_value"] < 0.01 else \
              "*"   if nb["p_value"] < 0.05 else "n.s."

        print(f"  Nadeau-Bengio (corrected):  p = {nb['p_value']:.6f}  {sig}")
        print(f"    t-stat = {nb['t_stat']:.3f}, df = {nb['df']}")
        print(f"    variance inflation = {nb['variance_inflation']:.2f}×")
        print(f"    mean diff = {nb['mean_diff']:+.4f}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print("  If UMAP's corrected p-value is still < 0.05, write:")
    print()
    print('  "Statistical significance was confirmed using the Nadeau-Bengio')
    print('   corrected paired t-test, which accounts for non-independence')
    print('   of fold estimates in repeated cross-validation."')
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()