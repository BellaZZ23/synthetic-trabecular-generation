#!/usr/bin/env python3
"""
independent_datasets_test.py — v2

Generates N independent synthetic datasets using full_pipeline.py,
computes quantum kernels for each, runs single train/test splits,
and performs a standard paired t-test on truly independent pairs.

Each dataset uses a different --seed, producing entirely separate
synthetic volumes, feature extractions, and dim reductions.

Usage:
  python independent_datasets_test.py ^
      --reference-image data\real\reference.png ^
      --voi-dirs data\derived\VOI1 data\derived\VOI4 ^
      --params-json output\v8_dataset\optimization_result.json ^
      --n-datasets 10 ^
      --methods umap ^
      --task classify-bvtv ^
      --outdir output\independent_test

Runtime: ~2 hrs/dataset/method for kernel. UMAP only = ~20 hrs total.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon as wilcoxon_test
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("ERROR: Qiskit not installed")

LABEL_KEYS = ["BVTV", "TbTh_um_p50", "TbN_per_mm", "TbSp_um_p50"]
METHOD_LABELS = {"pca": "PCA", "rp_gaussian": "RP Gaussian",
                 "pls": "PLS", "umap": "UMAP"}


def create_labels(Y, task):
    if task == "classify-bvtv":
        vals = Y[:, LABEL_KEYS.index("BVTV")]
        return (vals >= np.median(vals)).astype(int)
    elif task == "classify-tbn":
        vals = Y[:, LABEL_KEYS.index("TbN_per_mm")]
        return (vals >= np.median(vals)).astype(int)
    raise ValueError(f"Unknown task: {task}")


def run_full_pipeline(reference_image, voi_dirs, params_json,
                      outdir, num_samples, seed, xy=128, z=40):
    """Call full_pipeline.py to generate one independent dataset."""
    cmd = [
        sys.executable, "full_pipeline.py",
        "--reference-image", str(reference_image),
        "--params-json", str(params_json),
        "--outdir", str(outdir),
        "--num-samples", str(num_samples),
        "--seed", str(seed),
        "--xy", str(xy),
        "--z", str(z),
        "--skip-optimize",
    ]
    if voi_dirs:
        cmd += ["--voi-dirs"] + [str(v) for v in voi_dirs]

    print(f"    Running full_pipeline.py (seed={seed})...")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"    Pipeline FAILED (seed={seed})")
        return False

    print(f"    Pipeline complete in {elapsed/60:.1f} min")
    return True


def compute_quantum_kernel(X_pi, n_qubits=8, reps=2):
    """Compute N×N quantum kernel matrix."""
    X = X_pi[:, :n_qubits]
    nf = X.shape[1]
    fm = ZZFeatureMap(feature_dimension=nf, reps=reps, entanglement="linear")
    kernel = FidelityQuantumKernel(feature_map=fm)
    print(f"    Computing {X.shape[0]}x{X.shape[0]} kernel ({nf} qubits)...")
    t0 = time.time()
    K = kernel.evaluate(x_vec=X)
    print(f"    Kernel done in {(time.time()-t0)/60:.1f} min")
    return K


def run_single_split(K, X_01, y, test_size=0.2, seed=42):
    """Single train/test split, returns (classical_acc, quantum_acc)."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                      random_state=seed)
    tr, te = next(splitter.split(X_01, y))

    svm_c = SVC(kernel="rbf", gamma="scale", C=1.0, random_state=42)
    svm_c.fit(X_01[tr], y[tr])
    acc_c = accuracy_score(y[te], svm_c.predict(X_01[te]))

    svm_q = SVC(kernel="precomputed", C=1.0, random_state=42)
    svm_q.fit(K[np.ix_(tr, tr)], y[tr])
    acc_q = accuracy_score(y[te], svm_q.predict(K[np.ix_(te, tr)]))

    return float(acc_c), float(acc_q)


def plot_paired(all_results, outdir):
    """Paired dot plot showing per-dataset results."""
    methods = list(all_results.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 5), sharey=True)
    if n == 1: axes = [axes]

    for ax, method in zip(axes, methods):
        r = all_results[method]
        c = np.array(r["classical_accs"])
        q = np.array(r["quantum_accs"])
        nd = len(c)

        for i in range(nd):
            color = "#2ca02c" if q[i] > c[i] else "#d62728"
            ax.plot([0, 1], [c[i], q[i]], color=color, alpha=0.4, lw=1)

        ax.scatter([0]*nd, c, color="#4472C4", s=50, zorder=5,
                   edgecolors="white", lw=0.5, label="Classical")
        ax.scatter([1]*nd, q, color="#ED7D31", s=50, zorder=5,
                   edgecolors="white", lw=0.5, label="Quantum")
        ax.plot([0, 1], [c.mean(), q.mean()], "k-", lw=2.5, zorder=6,
                marker="D", markersize=8)

        p = r.get("p_value", 1.0)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else \
              "*" if p < 0.05 else "n.s."
        ax.set_title(f"{METHOD_LABELS.get(method, method)}\n"
                     f"p={p:.4f} ({sig})", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Classical\n(RBF)", "Quantum\n(ZZ)"], fontsize=9)
        ax.axhline(0.5, color="gray", ls="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Independent Datasets — Paired Comparison",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = outdir / "independent_paired_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  Saved figure: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-image", type=str, required=True)
    p.add_argument("--voi-dirs", nargs="+", type=str, default=None)
    p.add_argument("--params-json", type=str, required=True,
                   help="Optimized params JSON from your original run")
    p.add_argument("--n-datasets", type=int, default=10)
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--methods", nargs="+", default=["umap"])
    p.add_argument("--task", type=str, default="classify-bvtv")
    p.add_argument("--n-qubits", type=int, default=8)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--xy", type=int, default=128)
    p.add_argument("--z", type=int, default=40)
    p.add_argument("--outdir", type=str, default="output/independent_test")
    p.add_argument("--skip-generation", action="store_true",
                   help="Skip pipeline runs, use existing datasets")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  INDEPENDENT DATASETS TEST v2")
    print(f"  {args.n_datasets} datasets x {args.num_samples} samples each")
    print(f"  Methods: {args.methods}")
    print(f"  Task: {args.task}")
    print(f"  Calls full_pipeline.py with --skip-optimize")
    print("=" * 70)

    if not HAS_QISKIT:
        print("Qiskit required. Exiting."); return

    all_results = {m: {"classical_accs": [], "quantum_accs": []}
                   for m in args.methods}

    total_t0 = time.time()

    for ds_idx in range(args.n_datasets):
        seed = 1000 + ds_idx * 100
        ds_dir = outdir / f"dataset_{ds_idx:02d}"
        features_dir = ds_dir / "features"

        print(f"\n{'━' * 60}")
        print(f"  DATASET {ds_idx+1}/{args.n_datasets} (seed={seed})")
        print(f"{'━' * 60}")

        # ── Step 1: Generate via full_pipeline.py ─────────────────
        if not args.skip_generation:
            ok = run_full_pipeline(
                reference_image=args.reference_image,
                voi_dirs=args.voi_dirs,
                params_json=args.params_json,
                outdir=ds_dir,
                num_samples=args.num_samples,
                seed=seed,
                xy=args.xy, z=args.z,
            )
            if not ok:
                print(f"    Skipping dataset {ds_idx}")
                continue

        # ── Step 2: Load features, compute kernel, evaluate ───────
        for method in args.methods:
            # Try common filename patterns
            candidates = [
                features_dir / f"{method.lower()}_quantum_ready.npz",
                features_dir / f"{method.upper()}_quantum_ready.npz",
            ]
            if method == "rp_gaussian":
                candidates.append(features_dir / "RP_gaussian_quantum_ready.npz")

            npz_path = None
            for c in candidates:
                if c.exists():
                    npz_path = c; break

            if npz_path is None:
                print(f"    {method}: no feature file found in {features_dir}")
                # List what IS there for debugging
                if features_dir.exists():
                    files = [f.name for f in features_dir.glob("*.npz")]
                    print(f"    Available: {files}")
                continue

            data = np.load(npz_path, allow_pickle=True)
            X_pi = np.vstack([data["Z_train"], data["Z_test"]])
            X_01 = np.vstack([
                data["Z_train_01"] if "Z_train_01" in data else data["Z_train"],
                data["Z_test_01"] if "Z_test_01" in data else data["Z_test"],
            ])
            Y_all = np.vstack([data["Y_train"], data["Y_test"]])
            y = create_labels(Y_all, args.task)

            n_qubits = 4 if method == "pls" else args.n_qubits

            # Check for cached kernel
            kernel_path = ds_dir / f"kernel_{method}.npy"
            if kernel_path.exists():
                print(f"    {method}: loading cached kernel")
                K = np.load(kernel_path)
            else:
                K = compute_quantum_kernel(X_pi, n_qubits=n_qubits)
                np.save(kernel_path, K)

            acc_c, acc_q = run_single_split(K, X_01, y,
                                             test_size=args.test_size)
            all_results[method]["classical_accs"].append(acc_c)
            all_results[method]["quantum_accs"].append(acc_q)

            diff = acc_q - acc_c
            print(f"    {method}: C={acc_c:.3f}  Q={acc_q:.3f}  diff={diff:+.3f}")

    # ── Statistical tests ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  RESULTS — Paired t-test (truly independent datasets)")
    print(f"{'=' * 70}")

    for method in args.methods:
        r = all_results[method]
        c = np.array(r["classical_accs"])
        q = np.array(r["quantum_accs"])

        if len(c) < 2:
            print(f"\n  {method}: only {len(c)} dataset(s), need >= 2")
            continue

        diff = q - c
        n = len(diff)
        mean_diff = diff.mean()
        se = diff.std(ddof=1) / np.sqrt(n)
        t_stat = mean_diff / se if se > 0 else 0
        df = n - 1
        p_value = 2 * stats.t.sf(abs(t_stat), df)

        if not np.all(diff == 0):
            _, w_p = wilcoxon_test(diff, alternative="two-sided")
        else:
            w_p = 1.0

        q_wins = int((diff > 0).sum())
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else \
              "*" if p_value < 0.05 else "n.s."

        r.update({
            "mean_diff": float(mean_diff),
            "t_stat": float(t_stat), "df": df,
            "p_value": float(p_value),
            "p_wilcoxon": float(w_p),
            "q_wins": q_wins, "n_datasets": n,
        })

        print(f"\n  {METHOD_LABELS.get(method, method).upper()}")
        print(f"  {'─' * 50}")
        print(f"  Classical:  {c.mean():.4f} ± {c.std():.4f}")
        print(f"  Quantum:    {q.mean():.4f} ± {q.std():.4f}")
        print(f"  Mean diff:  {mean_diff:+.4f}")
        print(f"  Q wins:     {q_wins}/{n}")
        print(f"  Paired t:   t={t_stat:.3f}, df={df}, p={p_value:.6f} {sig}")
        print(f"  Wilcoxon:   p={w_p:.6f}")
        print(f"  Diffs:      {[f'{d:+.3f}' for d in diff]}")

    total_elapsed = time.time() - total_t0
    print(f"\n  Total: {total_elapsed/3600:.1f} hours")

    # ── Plot + Save ───────────────────────────────────────────────
    plot_paired(all_results, outdir)

    json_safe = {}
    for method, r in all_results.items():
        json_safe[method] = {
            k: ([float(x) for x in v] if isinstance(v, list) else v)
            for k, v in r.items()
        }
    json_path = outdir / "independent_test_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "task": args.task, "n_datasets": args.n_datasets,
            "num_samples": args.num_samples,
            "total_hours": total_elapsed / 3600,
            "results": json_safe,
        }, f, indent=2)
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()