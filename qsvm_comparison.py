#!/usr/bin/env python3
"""
qsvm_comparison.py  — v1.1
Fix 12: Repeated stratified k-fold CV for classical SVM (5 folds x 3 repeats).
        Quantum SVM uses single stratified split (kernel computation too slow for CV).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import MinMaxScaler, StratifiedShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold

# ── Qiskit imports ──
try:
    from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    HAS_QISKIT = True
    print("Qiskit available: quantum kernel enabled")
except ImportError:
    HAS_QISKIT = False
    print("WARNING: Qiskit not installed. Only classical SVM will run.")

LABEL_KEYS = ["BVTV", "TbTh_um_p50", "TbN_per_mm", "TbSp_um_p50"]


# ═══════════════════════════════════════════════════════════
#  1. DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_features(npz_path: str) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    return {
        "Z_train":    data["Z_train"],
        "Z_test":     data["Z_test"],
        "Z_train_01": data["Z_train_01"],
        "Z_test_01":  data["Z_test_01"],
        "Y_train":    data["Y_train"],
        "Y_test":     data["Y_test"],
        "n_features": int(data["n_features"]),
    }


def create_labels(Y: np.ndarray, task: str) -> tuple[np.ndarray, list[str]]:
    if task == "classify-bvtv":
        vals = Y[:, LABEL_KEYS.index("BVTV")]
        return (vals >= np.median(vals)).astype(int), ["sparse", "dense"]
    elif task == "classify-tbn":
        vals = Y[:, LABEL_KEYS.index("TbN_per_mm")]
        return (vals >= np.median(vals)).astype(int), ["few_struts", "many_struts"]
    elif task == "classify-tbth":
        vals = Y[:, LABEL_KEYS.index("TbTh_um_p50")]
        return (vals >= np.median(vals)).astype(int), ["thin", "thick"]
    elif task == "classify-multi":
        vals = Y[:, LABEL_KEYS.index("BVTV")]
        p33 = np.percentile(vals, 33); p66 = np.percentile(vals, 66)
        labels = np.zeros(len(vals), dtype=int)
        labels[vals >= p66] = 2
        labels[(vals >= p33) & (vals < p66)] = 1
        return labels, ["sparse", "medium", "dense"]
    else:
        raise ValueError(f"Unknown task: {task}")


def build_morphometric_features(Y_train, Y_test, n_components):
    mm_pi  = MinMaxScaler(feature_range=(0, np.pi))
    mm_01  = MinMaxScaler(feature_range=(0, 1))
    Zq_tr  = mm_pi.fit_transform(Y_train);  Zq_te  = mm_pi.transform(Y_test)
    Z01_tr = mm_01.fit_transform(Y_train);  Z01_te = mm_01.transform(Y_test)
    n = Y_train.shape[1]
    if n < n_components:
        pad_tr = np.zeros((Y_train.shape[0], n_components - n))
        pad_te = np.zeros((Y_test.shape[0],  n_components - n))
        Zq_tr  = np.hstack([Zq_tr,  pad_tr]); Zq_te  = np.hstack([Zq_te,  pad_te])
        Z01_tr = np.hstack([Z01_tr, pad_tr]); Z01_te = np.hstack([Z01_te, pad_te])
    return {
        "Z_train":    Zq_tr[:, :n_components],
        "Z_test":     Zq_te[:, :n_components],
        "Z_train_01": Z01_tr[:, :n_components],
        "Z_test_01":  Z01_te[:, :n_components],
        "n_features": min(n, n_components),
    }


# ═══════════════════════════════════════════════════════════
#  2. CLASSICAL SVM — FIX 12: Repeated Stratified K-Fold CV
# ═══════════════════════════════════════════════════════════

def run_classical_svm_cv(
    X_all: np.ndarray,
    y_all: np.ndarray,
    class_names: list[str],
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
) -> dict:
    """
    FIX 12: Repeated stratified k-fold cross-validation for classical SVM.
    Combines train+test data and runs 5-fold x 3 repeats = 15 evaluations.
    Reports mean ± std for accuracy and F1.
    """
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    accs, f1s, precs, recs, times = [], [], [], [], []

    for fold_idx, (tr_idx, te_idx) in enumerate(rkf.split(X_all, y_all)):
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]
        t0  = time.time()
        svm = SVC(kernel="rbf", gamma="scale", C=1.0, random_state=seed)
        svm.fit(X_tr, y_tr)
        y_pred = svm.predict(X_te)
        times.append(time.time() - t0)
        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="weighted", zero_division=0))
        precs.append(precision_score(y_te, y_pred, average="weighted", zero_division=0))
        recs.append(recall_score(y_te, y_pred, average="weighted", zero_division=0))

    print(f"    Repeated CV ({n_splits}x{n_repeats}={len(accs)} folds):")
    print(f"    Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"    F1 Score: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    return {
        "method":        "Classical SVM (RBF) — Repeated CV",
        "cv_type":       f"RepeatedStratifiedKFold({n_splits}x{n_repeats})",
        "accuracy":      float(np.mean(accs)),
        "accuracy_std":  float(np.std(accs)),
        "f1_score":      float(np.mean(f1s)),
        "f1_std":        float(np.std(f1s)),
        "precision":     float(np.mean(precs)),
        "recall":        float(np.mean(recs)),
        "all_accuracies":  [float(a) for a in accs],
        "all_f1s":         [float(f) for f in f1s],
        "train_time_s":  float(np.mean(times)),
        "class_names":   class_names,
        "n_samples":     len(y_all),
        "n_evaluations": len(accs),
    }


def run_classical_svm_single(X_train, X_test, y_train, y_test, class_names) -> dict:
    """Single-split classical SVM (used for direct comparison with quantum on same split)."""
    t0  = time.time()
    svm = SVC(kernel="rbf", gamma="scale", C=1.0, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    train_time = time.time() - t0
    y_pred     = svm.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    return {
        "method": "Classical SVM (RBF) — single split",
        "accuracy": float(acc), "f1_score": float(f1),
        "precision": float(prec), "recall": float(rec),
        "confusion_matrix": cm.tolist(),
        "train_time_s": float(train_time),
        "y_pred": y_pred.tolist(),
        "class_names": class_names,
        "n_train": len(y_train), "n_test": len(y_test),
    }


# ═══════════════════════════════════════════════════════════
#  3. QUANTUM KERNEL SVM
# ═══════════════════════════════════════════════════════════

def run_quantum_svm(X_train_pi, X_test_pi, y_train, y_test,
                     n_qubits, class_names, feature_map_type="ZZ", reps=2) -> dict:
    if not HAS_QISKIT:
        return {"method": "Quantum Kernel SVM", "error": "Qiskit not installed",
                "accuracy": None, "f1_score": None}

    X_tr   = X_train_pi[:, :n_qubits]
    X_te   = X_test_pi[:, :n_qubits]
    n_feat = X_tr.shape[1]

    print(f"    Quantum kernel: {n_feat} qubits, {feature_map_type}, {reps} reps")

    if feature_map_type == "ZZ":
        feature_map = ZZFeatureMap(feature_dimension=n_feat, reps=reps, entanglement="linear")
    elif feature_map_type == "Z":
        feature_map = ZFeatureMap(feature_dimension=n_feat, reps=reps)
    else:
        feature_map = PauliFeatureMap(feature_dimension=n_feat, reps=reps,
                                       paulis=["Z","ZZ"], entanglement="linear")

    t0 = time.time()
    try:
        kernel  = FidelityQuantumKernel(feature_map=feature_map)
        print(f"    Computing training kernel ({X_tr.shape[0]}x{X_tr.shape[0]})...")
        K_train = kernel.evaluate(x_vec=X_tr)
        print(f"    Computing test kernel ({X_te.shape[0]}x{X_tr.shape[0]})...")
        K_test  = kernel.evaluate(x_vec=X_te, y_vec=X_tr)
    except Exception as e:
        return {"method": f"Quantum Kernel SVM ({feature_map_type})",
                "error": str(e), "accuracy": None, "f1_score": None}

    kernel_time = time.time() - t0
    print(f"    Kernel computed in {kernel_time:.1f}s")

    t0     = time.time()
    svm    = SVC(kernel="precomputed", C=1.0, random_state=42)
    svm.fit(K_train, y_train)
    y_pred = svm.predict(K_test)
    train_time = time.time() - t0

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"    Accuracy: {acc:.3f}, F1: {f1:.3f}")

    return {
        "method":          f"Quantum Kernel SVM ({feature_map_type})",
        "cv_type":         "single stratified split (CV too slow for quantum)",
        "accuracy":        float(acc),
        "accuracy_std":    None,
        "f1_score":        float(f1),
        "f1_std":          None,
        "precision":       float(prec),
        "recall":          float(rec),
        "confusion_matrix":cm.tolist(),
        "kernel_time_s":   float(kernel_time),
        "train_time_s":    float(train_time),
        "y_pred":          y_pred.tolist(),
        "class_names":     class_names,
        "n_qubits":        n_feat,
        "feature_map":     feature_map_type,
        "reps":            reps,
        "circuit_depth":   feature_map.depth(),
        "n_train":         len(y_train),
        "n_test":          len(y_test),
    }


# ═══════════════════════════════════════════════════════════
#  4. VISUALIZATION
# ═══════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, class_names, title, outpath):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(class_names)
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i,j] > cm.max()/2 else "black"
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", color=color, fontsize=14)
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()


def plot_accuracy_comparison(results, outdir):
    methods   = sorted(set(r["feature_source"] for r in results))
    clf_accs  = {}
    clf_stds  = {}
    for r in results:
        key = (r["feature_source"], r["classifier_type"])
        clf_accs[key] = r["result"].get("accuracy") or 0
        clf_stds[key] = r["result"].get("accuracy_std") or 0

    x     = np.arange(len(methods)); width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))

    c_accs = [clf_accs.get((m,"classical"), 0) for m in methods]
    c_stds = [clf_stds.get((m,"classical"), 0) for m in methods]
    q_accs = [clf_accs.get((m,"quantum"),   0) for m in methods]
    q_stds = [clf_stds.get((m,"quantum"),   0) for m in methods]

    bars1 = ax.bar(x - width/2, c_accs, width, yerr=c_stds, capsize=4,
                   label="Classical SVM (RBF)", color="#4472C4")
    if any(a > 0 for a in q_accs):
        bars2 = ax.bar(x + width/2, q_accs, width, yerr=q_stds, capsize=4,
                       label="Quantum Kernel SVM", color="#ED7D31")
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x()+bar.get_width()/2, h+0.015,
                        f"{h:.3f}", ha="center", fontsize=8)
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.015,
                f"{h:.3f}", ha="center", fontsize=8)

    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.1)
    ax.set_title("Classification Accuracy: Classical vs Quantum SVM\n"
                 "(Classical: mean ± std over 15 CV folds; Quantum: single split)")
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="Random baseline")
    ax.legend()
    plt.tight_layout(); plt.savefig(outdir/"accuracy_comparison.png", dpi=150); plt.close()
    print(f"  Saved: {outdir/'accuracy_comparison.png'}")


def plot_comparison_table(results, outdir):
    methods = sorted(set(r["feature_source"] for r in results))
    rows    = []
    for m in methods:
        row = [m]
        for ct in ["classical", "quantum"]:
            match = [r for r in results if r["feature_source"]==m and r["classifier_type"]==ct]
            if match:
                res = match[0]["result"]
                acc = res.get("accuracy"); std = res.get("accuracy_std")
                f1  = res.get("f1_score"); fstd = res.get("f1_std")
                acc_str = f"{acc:.3f}±{std:.3f}" if (acc is not None and std is not None) else (f"{acc:.3f}" if acc else "N/A")
                f1_str  = f"{f1:.3f}±{fstd:.3f}"  if (f1  is not None and fstd is not None) else (f"{f1:.3f}"  if f1  else "N/A")
                row += [acc_str, f1_str]
            else:
                row += ["—", "—"]
        rows.append(row)

    col_labels = ["Feature Method",
                  "Classical Acc", "Classical F1",
                  "Quantum Acc",   "Quantum F1"]
    fig, ax = plt.subplots(figsize=(14, max(3, len(rows)*0.9+2)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.8)
    for j in range(len(col_labels)):
        tbl[0,j].set_facecolor("#4472C4"); tbl[0,j].set_text_props(color="white", fontweight="bold")
    for i in range(len(rows)):
        color = "#D9E2F3" if i%2==0 else "white"
        for j in range(len(col_labels)): tbl[i+1,j].set_facecolor(color)
    ax.set_title("Classical vs Quantum SVM — Full Comparison Table", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout(); plt.savefig(outdir/"comparison_table.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {outdir/'comparison_table.png'}")


# ═══════════════════════════════════════════════════════════
#  5. MAIN
# ═══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=str, required=True)
    p.add_argument("--outdir",       type=str, default="output/qsvm_results")
    p.add_argument("--n-qubits",     type=int, default=8)
    p.add_argument("--task",         type=str, default="classify-bvtv",
                   choices=["classify-bvtv","classify-tbn","classify-tbth","classify-multi"])
    p.add_argument("--feature-map",  type=str, default="ZZ", choices=["ZZ","Z","Pauli"])
    p.add_argument("--reps",         type=int, default=2)
    p.add_argument("--cv-folds",     type=int, default=5,
                   help="Number of CV folds for classical SVM (Fix 12)")
    p.add_argument("--cv-repeats",   type=int, default=3,
                   help="Number of CV repeats for classical SVM (Fix 12)")
    p.add_argument("--skip-quantum", action="store_true")
    args = p.parse_args()

    outdir       = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    features_dir = Path(args.features_dir)

    print(f"\n{'='*60}")
    print(f"  CLASSICAL vs QUANTUM SVM COMPARISON  (v1.1)")
    print(f"  Fix 12: Classical SVM uses {args.cv_folds}x{args.cv_repeats} Repeated Stratified CV")
    print(f"{'='*60}")
    print(f"  Features: {features_dir}")
    print(f"  Task: {args.task}")
    print(f"  Qubits: {args.n_qubits}")
    print(f"  Quantum: {'enabled' if (HAS_QISKIT and not args.skip_quantum) else 'disabled'}")

    # Discover feature files
    npz_files = {}
    for name in ["pca", "rp_gaussian", "rp_sparse"]:
        path = features_dir / f"{name}_quantum_ready.npz"
        if path.exists():
            npz_files[name] = path
            print(f"  Found: {name}")
    if not npz_files:
        print("ERROR: No *_quantum_ready.npz files found"); return

    first_data = load_features(str(list(npz_files.values())[0]))
    morph_data = build_morphometric_features(
        first_data["Y_train"], first_data["Y_test"], args.n_qubits)
    morph_data["Y_train"] = first_data["Y_train"]
    morph_data["Y_test"]  = first_data["Y_test"]

    all_results = []
    feature_sources = list(npz_files.keys()) + ["morphometric"]

    for source in feature_sources:
        print(f"\n{'─'*50}\n  FEATURES: {source.upper()}\n{'─'*50}")

        data = morph_data if source == "morphometric" else load_features(str(npz_files[source]))

        y_train_lbl, class_names = create_labels(data["Y_train"], args.task)
        y_test_lbl,  _           = create_labels(data["Y_test"],  args.task)

        print(f"  Classes: {class_names}")
        print(f"  Train dist: {dict(zip(*np.unique(y_train_lbl, return_counts=True)))}")
        print(f"  Test  dist: {dict(zip(*np.unique(y_test_lbl,  return_counts=True)))}")

        # ── FIX 12: Repeated CV for classical SVM ──
        print(f"\n  Running Classical SVM (Repeated CV)...")
        X_all = np.vstack([data["Z_train_01"], data["Z_test_01"]])
        y_all = np.concatenate([y_train_lbl, y_test_lbl])
        cv_result = run_classical_svm_cv(
            X_all, y_all, class_names,
            n_splits=args.cv_folds, n_repeats=args.cv_repeats)
        all_results.append({
            "feature_source": source, "classifier_type": "classical", "result": cv_result})

        # Also run single split for confusion matrix and quantum comparison
        single_result = run_classical_svm_single(
            data["Z_train_01"], data["Z_test_01"],
            y_train_lbl, y_test_lbl, class_names)
        cm = np.array(single_result["confusion_matrix"])
        plot_confusion_matrix(cm, class_names,
            f"Classical SVM — {source}\nAcc={single_result['accuracy']:.3f} (single split)",
            outdir / f"cm_classical_{source}.png")

        # ── Quantum SVM — single stratified split ──
        if HAS_QISKIT and not args.skip_quantum:
            print(f"\n  Running Quantum Kernel SVM (single stratified split)...")
            # Use stratified split for quantum to ensure class balance
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            q_tr, q_te = next(sss.split(X_all, y_all))
            q_result = run_quantum_svm(
                X_all[q_tr], X_all[q_te],
                y_all[q_tr], y_all[q_te],
                n_qubits=args.n_qubits, class_names=class_names,
                feature_map_type=args.feature_map, reps=args.reps)
            if q_result.get("accuracy") is not None:
                cm_q = np.array(q_result["confusion_matrix"])
                plot_confusion_matrix(cm_q, class_names,
                    f"Quantum SVM ({args.feature_map}) — {source}\nAcc={q_result['accuracy']:.3f}",
                    outdir / f"cm_quantum_{source}.png")
            all_results.append({
                "feature_source": source, "classifier_type": "quantum", "result": q_result})

    # ── Summary ──
    print(f"\n{'='*70}\n  RESULTS SUMMARY\n{'='*70}")
    print(f"  {'Feature':<18} {'Classifier':<35} {'Accuracy':>10} {'F1':>8}")
    print(f"  {'-'*72}")
    for r in all_results:
        res = r["result"]
        acc = res.get("accuracy"); std = res.get("accuracy_std")
        f1  = res.get("f1_score")
        acc_str = (f"{acc:.3f}±{std:.3f}" if (acc is not None and std) else
                   f"{acc:.3f}" if acc is not None else "N/A")
        f1_str  = f"{f1:.3f}" if f1 is not None else "N/A"
        print(f"  {r['feature_source']:<18} {res['method']:<35} {acc_str:>10} {f1_str:>8}")
    print(f"{'='*70}")

    plot_comparison_table(all_results, outdir)
    plot_accuracy_comparison(all_results, outdir)

    summary = {
        "pipeline_version": "1.1",
        "fix12_applied": "RepeatedStratifiedKFold for classical SVM",
        "task": args.task, "n_qubits": args.n_qubits,
        "feature_map": args.feature_map, "reps": args.reps,
        "cv_folds": args.cv_folds, "cv_repeats": args.cv_repeats,
        "results": [{
            "feature_source":  r["feature_source"],
            "classifier_type": r["classifier_type"],
            "accuracy":        r["result"].get("accuracy"),
            "accuracy_std":    r["result"].get("accuracy_std"),
            "f1_score":        r["result"].get("f1_score"),
            "f1_std":          r["result"].get("f1_std"),
            "precision":       r["result"].get("precision"),
            "recall":          r["result"].get("recall"),
            "train_time_s":    r["result"].get("train_time_s"),
            "kernel_time_s":   r["result"].get("kernel_time_s"),
            "cv_type":         r["result"].get("cv_type"),
            "n_evaluations":   r["result"].get("n_evaluations"),
            "class_names":     r["result"].get("class_names"),
        } for r in all_results],
    }
    with open(outdir/"qsvm_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved: {outdir/'qsvm_results.json'}")


if __name__ == "__main__":
    main()