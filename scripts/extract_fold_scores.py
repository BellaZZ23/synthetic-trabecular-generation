#!/usr/bin/env python3
"""
Re-run repeated stratified CV from precomputed kernel matrices
to extract per-fold accuracy arrays, then run Wilcoxon tests.

Uses the saved kernel_*.npy files — no Qiskit needed, runs in seconds.

Usage:
  python extract_fold_scores.py \
      --features-dir output/v8_dataset/features \
      --kernel-dir output/v8_qsvm_tight \
      --task classify-bvtv
"""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.stats import wilcoxon

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


def run_cv(K, X_01, y, n_splits=5, n_repeats=5, seed=42):
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                  random_state=seed)
    q_accs, c_accs = [], []

    for tr, te in rkf.split(K, y):
        # Quantum: precomputed kernel
        K_train = K[np.ix_(tr, tr)]
        K_test = K[np.ix_(te, tr)]
        svm_q = SVC(kernel="precomputed", C=1.0, random_state=seed)
        svm_q.fit(K_train, y[tr])
        q_accs.append(accuracy_score(y[te], svm_q.predict(K_test)))

        # Classical: RBF on [0,1] features
        svm_c = SVC(kernel="rbf", gamma="scale", C=1.0, random_state=seed)
        svm_c.fit(X_01[tr], y[tr])
        c_accs.append(accuracy_score(y[te], svm_c.predict(X_01[te])))

    return np.array(c_accs), np.array(q_accs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=str, default="output/v8_dataset/features")
    p.add_argument("--kernel-dir", type=str, default="output/v8_qsvm_tight")
    p.add_argument("--task", type=str, default="classify-bvtv",
                   choices=["classify-bvtv", "classify-tbn"])
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-repeats", type=int, default=5)
    args = p.parse_args()

    features_dir = Path(args.features_dir)
    kernel_dir = Path(args.kernel_dir)

    print(f"Task: {args.task}")
    print(f"CV: {args.cv_folds}x{args.cv_repeats} = {args.cv_folds * args.cv_repeats} folds\n")

    all_fold_scores = {}

    for method in METHODS:
        npz_path = features_dir / f"{method}_quantum_ready.npz"
        kernel_path = kernel_dir / f"kernel_{method}.npy"

        if not npz_path.exists():
            print(f"  {method}: features not found, skipping")
            continue
        if not kernel_path.exists():
            print(f"  {method}: kernel not found, skipping")
            continue

        # Load features and kernel
        data = np.load(npz_path)
        X_01 = np.vstack([data["Z_train_01"], data["Z_test_01"]]) \
            if "Z_train_01" in data else np.vstack([data["Z_train"], data["Z_test"]])
        Y = np.vstack([data["Y_train"], data["Y_test"]])
        y = create_labels(Y, args.task)
        K = np.load(kernel_path)

        print(f"  {method}: {K.shape[0]} samples, {X_01.shape[1]} features")

        c_accs, q_accs = run_cv(K, X_01, y, args.cv_folds, args.cv_repeats)
        all_fold_scores[method] = {"classical": c_accs, "quantum": q_accs}

        # Save per-fold scores
        np.save(kernel_dir / f"{method}_classical_fold_scores.npy", c_accs)
        np.save(kernel_dir / f"{method}_quantum_fold_scores.npy", q_accs)

        print(f"    Classical: {c_accs.mean():.4f} ± {c_accs.std():.4f}")
        print(f"    Quantum:   {q_accs.mean():.4f} ± {q_accs.std():.4f}")
        print(f"    Gap:       {(q_accs.mean() - c_accs.mean()):+.4f}")

    # Wilcoxon tests
    print(f"\n{'='*60}")
    print(f"  WILCOXON SIGNED-RANK TESTS")
    print(f"{'='*60}")

    for method, scores in all_fold_scores.items():
        c = scores["classical"]
        q = scores["quantum"]
        diff = q - c

        # Two-sided test
        stat, p_two = wilcoxon(diff, alternative="two-sided")
        # One-sided: quantum > classical
        _, p_greater = wilcoxon(diff, alternative="greater")

        if p_two < 0.001:
            sig = "***"
        elif p_two < 0.01:
            sig = "**"
        elif p_two < 0.05:
            sig = "*"
        else:
            sig = "n.s."

        print(f"\n  {method.upper()}")
        print(f"    Mean diff: {diff.mean():+.4f}")
        print(f"    Wilcoxon stat: {stat:.1f}")
        print(f"    p (two-sided):  {p_two:.6f}  {sig}")
        print(f"    p (Q > C):      {p_greater:.6f}")
        print(f"    Folds where Q > C: {(diff > 0).sum()}/{len(diff)}")

    # Save summary
    summary = {}
    for method, scores in all_fold_scores.items():
        diff = scores["quantum"] - scores["classical"]
        stat, p_two = wilcoxon(diff, alternative="two-sided")
        _, p_greater = wilcoxon(diff, alternative="greater")
        summary[method] = {
            "classical_mean": float(scores["classical"].mean()),
            "quantum_mean": float(scores["quantum"].mean()),
            "mean_diff": float(diff.mean()),
            "wilcoxon_stat": float(stat),
            "p_two_sided": float(p_two),
            "p_greater": float(p_greater),
            "folds_q_wins": int((diff > 0).sum()),
            "n_folds": len(diff),
        }

    out_json = kernel_dir / f"wilcoxon_results_{args.task}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_json}")


if __name__ == "__main__":
    main()