#!/usr/bin/env python3
"""
qsvm_tighter_v1.py — Tighter quantum SVM via full kernel + repeated CV

Strategy:
  1. Compute the FULL 500x500 quantum kernel matrix ONCE (~1 hour per method)
  2. Do 5x5 repeated stratified CV by indexing into the precomputed matrix
  3. This gives quantum results with proper std estimates, matching classical eval

Runtime: ~1 hour per feature method (kernel), then seconds for all CV splits.
Total: ~5 hours for 5 methods (same as before, but now with 25-fold estimates).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold

try:
    from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    HAS_QISKIT = True
    print("Qiskit available")
except ImportError:
    HAS_QISKIT = False
    print("WARNING: Qiskit not installed")

LABEL_KEYS = ["BVTV", "TbTh_um_p50", "TbN_per_mm", "TbSp_um_p50"]
ALL_FEATURES = ["pca", "rp_gaussian", "rp_sparse", "pls", "umap"]


def discover_feature_files(features_dir, requested=None):
    npz_files = {}
    for name in ALL_FEATURES:
        path = features_dir / f"{name}_quantum_ready.npz"
        if path.exists():
            if requested is None or name in requested:
                npz_files[name] = path
                print(f"  Found: {name}")
    return npz_files


def load_features(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    out = {"Y_train": data["Y_train"], "Y_test": data["Y_test"],
           "n_features": int(data["n_features"]),
           "Z_train": data["Z_train"], "Z_test": data["Z_test"]}
    out["Z_train_01"] = data["Z_train_01"] if "Z_train_01" in data.files else data["Z_train"]
    out["Z_test_01"]  = data["Z_test_01"]  if "Z_test_01"  in data.files else data["Z_test"]
    return out


def create_labels(Y, task):
    if task == "classify-bvtv":
        vals = Y[:, LABEL_KEYS.index("BVTV")]
        return (vals >= np.median(vals)).astype(int), ["sparse", "dense"]
    elif task == "classify-tbn":
        vals = Y[:, LABEL_KEYS.index("TbN_per_mm")]
        return (vals >= np.median(vals)).astype(int), ["few_struts", "many_struts"]
    elif task == "classify-tbsp":
        vals = Y[:, LABEL_KEYS.index("TbSp_um_p50")]
        return (vals >= np.median(vals)).astype(int), ["dense_spacing", "wide_spacing"]
    raise ValueError(f"Unknown task: {task}")


def compute_full_kernel(X_all_pi, n_qubits, fmap_type="ZZ", reps=2):
    """Compute the full N×N quantum kernel matrix once."""
    X = X_all_pi[:, :n_qubits]
    nf = X.shape[1]
    print(f"    Building {fmap_type} feature map: {nf} qubits, {reps} reps")

    if fmap_type == "ZZ":
        fm = ZZFeatureMap(feature_dimension=nf, reps=reps, entanglement="linear")
    elif fmap_type == "Z":
        fm = ZFeatureMap(feature_dimension=nf, reps=reps)
    else:
        fm = PauliFeatureMap(feature_dimension=nf, reps=reps,
                             paulis=["Z", "ZZ"], entanglement="linear")

    depth = fm.depth()
    kernel = FidelityQuantumKernel(feature_map=fm)

    print(f"    Computing full kernel ({X.shape[0]}x{X.shape[0]})...")
    t0 = time.time()
    K = kernel.evaluate(x_vec=X)
    kt = time.time() - t0
    print(f"    Kernel computed in {kt:.1f}s ({kt/60:.1f} min)")

    return K, {"kernel_time_s": kt, "circuit_depth": depth,
               "n_qubits": nf, "feature_map": fmap_type, "reps": reps}


def run_cv_on_precomputed_kernel(K, y, n_splits=5, n_repeats=5, seed=42):
    """Run repeated stratified CV using precomputed kernel matrix."""
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                  random_state=seed)
    accs, f1s = [], []

    for fold_idx, (tr, te) in enumerate(rkf.split(K, y)):
        K_train = K[np.ix_(tr, tr)]
        K_test  = K[np.ix_(te, tr)]

        svm = SVC(kernel="precomputed", C=1.0, random_state=seed)
        svm.fit(K_train, y[tr])
        yp = svm.predict(K_test)

        accs.append(float(accuracy_score(y[te], yp)))
        f1s.append(float(f1_score(y[te], yp, average="weighted", zero_division=0)))

    n_evals = len(accs)
    print(f"    Quantum Repeated CV ({n_splits}x{n_repeats}={n_evals} folds):")
    print(f"    Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"    F1 Score: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    return {
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_score": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "all_accuracies": accs,
        "all_f1s": f1s,
        "n_evaluations": n_evals,
        "cv_type": f"RepeatedStratifiedKFold({n_splits}x{n_repeats})",
    }


def run_classical_cv(X_all, y, n_splits=5, n_repeats=5, seed=42):
    """Classical RBF SVM with same CV scheme for fair comparison."""
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                  random_state=seed)
    accs, f1s = [], []

    for tr, te in rkf.split(X_all, y):
        svm = SVC(kernel="rbf", gamma="scale", C=1.0, random_state=seed)
        svm.fit(X_all[tr], y[tr])
        yp = svm.predict(X_all[te])
        accs.append(float(accuracy_score(y[te], yp)))
        f1s.append(float(f1_score(y[te], yp, average="weighted", zero_division=0)))

    print(f"    Classical Repeated CV ({n_splits}x{n_repeats}={len(accs)} folds):")
    print(f"    Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"    F1 Score: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    return {
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_score": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "all_accuracies": accs,
        "n_evaluations": len(accs),
        "cv_type": f"RepeatedStratifiedKFold({n_splits}x{n_repeats})",
    }


def plot_results(all_results, outdir):
    """Bar chart with error bars for both classical and quantum."""
    methods = list(all_results.keys())
    fig, ax = plt.subplots(figsize=(max(10, len(methods)*2), 6))

    x = np.arange(len(methods)); w = 0.35

    c_acc = [all_results[m]["classical"]["accuracy"] for m in methods]
    c_std = [all_results[m]["classical"]["accuracy_std"] for m in methods]
    q_acc = [all_results[m]["quantum"]["accuracy"] for m in methods]
    q_std = [all_results[m]["quantum"]["accuracy_std"] for m in methods]

    b1 = ax.bar(x - w/2, c_acc, w, yerr=c_std, capsize=4,
                label="Classical SVM (RBF)", color="#4472C4", alpha=0.85)
    b2 = ax.bar(x + w/2, q_acc, w, yerr=q_std, capsize=4,
                label="Quantum SVM (ZZ)", color="#ED7D31", alpha=0.85)

    for bars in [b1, b2]:
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width()/2, h + 0.02,
                        f"{h:.3f}", ha="center", fontsize=8)

    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.1)
    ax.set_title("Classical vs Quantum SVM — Repeated CV (Both)")
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="Random baseline")
    ax.legend(); plt.tight_layout()
    plt.savefig(outdir / "accuracy_comparison_tight.png", dpi=150); plt.close()
    print(f"  Saved: {outdir / 'accuracy_comparison_tight.png'}")


def main():
    p = argparse.ArgumentParser(description="Tighter QSVM via full kernel + repeated CV")
    p.add_argument("--features-dir", type=str, required=True)
    p.add_argument("--outdir",       type=str, default="output/qsvm_tight")
    p.add_argument("--n-qubits",     type=int, default=8)
    p.add_argument("--task",         type=str, default="classify-bvtv",
                   choices=["classify-bvtv", "classify-tbn", "classify-tbsp"])
    p.add_argument("--feature-map",  type=str, default="ZZ", choices=["ZZ", "Z", "Pauli"])
    p.add_argument("--reps",         type=int, default=2)
    p.add_argument("--cv-folds",     type=int, default=5)
    p.add_argument("--cv-repeats",   type=int, default=5)
    p.add_argument("--features",     nargs="+", default=None,
                   help="Specific feature sets e.g. --features pca rp_gaussian")
    p.add_argument("--skip-quantum", action="store_true")
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    features_dir = Path(args.features_dir)

    print(f"\n{'='*60}")
    print(f"  TIGHTER QSVM — Full Kernel + Repeated CV")
    print(f"  Both classical and quantum get {args.cv_folds}x{args.cv_repeats} CV")
    print(f"{'='*60}")
    print(f"  Task: {args.task}")
    print(f"  Qubits: {args.n_qubits}, Feature map: {args.feature_map}, Reps: {args.reps}")

    npz_files = discover_feature_files(features_dir, requested=args.features)
    if not npz_files:
        print("ERROR: No feature files found"); return

    all_results = {}

    for source, npz_path in npz_files.items():
        print(f"\n{'─'*50}")
        print(f"  FEATURES: {source.upper()}")
        print(f"{'─'*50}")

        data = load_features(str(npz_path))

        # Combine train+test for full CV
        X_all_01 = np.vstack([data["Z_train_01"], data["Z_test_01"]])
        X_all_pi = np.vstack([data["Z_train"], data["Z_test"]])
        Y_all    = np.vstack([data["Y_train"], data["Y_test"]])
        y_all, class_names = create_labels(Y_all, args.task)

        print(f"  Total samples: {len(y_all)}")
        print(f"  Class distribution: {dict(zip(*np.unique(y_all, return_counts=True)))}")

        # Classical CV
        print(f"\n  Classical SVM:")
        cl = run_classical_cv(X_all_01, y_all,
                              n_splits=args.cv_folds,
                              n_repeats=args.cv_repeats)

        # Quantum: compute full kernel, then CV
        q = {"accuracy": None, "accuracy_std": None, "f1_score": None,
             "f1_std": None, "error": "skipped"}

        if HAS_QISKIT and not args.skip_quantum:
            print(f"\n  Quantum Kernel (computing full {len(y_all)}x{len(y_all)} matrix):")
            K, k_info = compute_full_kernel(X_all_pi, args.n_qubits,
                                             args.feature_map, args.reps)

            # Save kernel for reuse
            np.save(outdir / f"kernel_{source}.npy", K)
            print(f"  Saved kernel: {outdir / f'kernel_{source}.npy'}")

            q = run_cv_on_precomputed_kernel(K, y_all,
                                              n_splits=args.cv_folds,
                                              n_repeats=args.cv_repeats)
            q.update(k_info)

        all_results[source] = {"classical": cl, "quantum": q,
                                "class_names": class_names}

        # Print gap
        if q.get("accuracy") is not None:
            gap = q["accuracy"] - cl["accuracy"]
            print(f"\n  Gap (quantum - classical): {gap:+.3f}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY — {args.task}")
    print(f"{'='*70}")
    print(f"  {'Feature':<15} {'Classical':>18} {'Quantum':>18} {'Gap':>8}")
    print(f"  {'-'*62}")
    for source, r in all_results.items():
        cl = r["classical"]; q = r["quantum"]
        cl_s = f"{cl['accuracy']:.3f}±{cl['accuracy_std']:.3f}"
        if q.get("accuracy") is not None:
            q_s  = f"{q['accuracy']:.3f}±{q['accuracy_std']:.3f}"
            gap  = q["accuracy"] - cl["accuracy"]
            gap_s = f"{gap:+.3f}"
        else:
            q_s = "N/A"; gap_s = "—"
        print(f"  {source:<15} {cl_s:>18} {q_s:>18} {gap_s:>8}")
    print(f"{'='*70}")

    # Plot
    plot_results(all_results, outdir)

    # Save JSON
    json_out = {}
    for source, r in all_results.items():
        json_out[source] = {
            "classical": {k: v for k, v in r["classical"].items()
                          if k != "all_accuracies"},
            "quantum": {k: v for k, v in r["quantum"].items()
                        if k not in ("all_accuracies", "all_f1s")},
        }
    with open(outdir / "qsvm_tight_results.json", "w") as f:
        json.dump({"task": args.task, "n_qubits": args.n_qubits,
                   "feature_map": args.feature_map, "reps": args.reps,
                   "cv": f"{args.cv_folds}x{args.cv_repeats}",
                   "results": json_out}, f, indent=2, default=str)
    print(f"\n  Results saved: {outdir / 'qsvm_tight_results.json'}")


if __name__ == "__main__":
    main()