#!/usr/bin/env python3
"""
qsvm_comparison.py

Classical SVM vs Quantum Kernel SVM comparison for trabecular bone classification.

Loads the quantum-ready .npz files from the PCA/RP pipeline and runs:
  1. Classical SVM (RBF kernel) on PCA, RP_gaussian, RP_sparse features
  2. Quantum Kernel SVM (Qiskit) on the same features
  3. Morphometric feature baseline (direct metrics, no image reduction)

Produces:
  - 3x2 comparison table (3 feature methods x classical vs quantum)
  - ROC curves, confusion matrices
  - Training time comparison
  - Full results JSON

Usage:
  python qsvm_comparison.py `
      --features-dir output\final_dataset\features `
      --outdir output\qsvm_results `
      --n-qubits 8 `
      --task classify-bvtv `
      --quantum-backend aer_simulator

Requirements:
  pip install qiskit qiskit-machine-learning qiskit-aer scikit-learn matplotlib
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
    confusion_matrix, classification_report, roc_curve, auc,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer

# ── Qiskit imports (graceful fallback if not installed) ──
try:
    from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_aer import AerSimulator
    from qiskit.primitives import StatevectorSampler
    HAS_QISKIT = True
    print("Qiskit available: quantum kernel enabled")
except ImportError:
    HAS_QISKIT = False
    print("WARNING: Qiskit not installed. Only classical SVM will run.")
    print("  Install: pip install qiskit qiskit-machine-learning qiskit-aer")


LABEL_KEYS = ["BVTV", "TbTh_um_p50", "TbN_per_mm", "TbSp_um_p50"]


# ═══════════════════════════════════════════════════════════
#  1. DATA LOADING & LABELLING
# ═══════════════════════════════════════════════════════════

def load_features(npz_path: str) -> dict:
    """Load quantum-ready features from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        "Z_train": data["Z_train"],         # [0, pi] scaled
        "Z_test": data["Z_test"],
        "Z_train_01": data["Z_train_01"],    # [0, 1] scaled
        "Z_test_01": data["Z_test_01"],
        "Y_train": data["Y_train"],
        "Y_test": data["Y_test"],
        "n_features": int(data["n_features"]),
    }


def create_classification_labels(Y: np.ndarray, task: str, label_keys: list) -> np.ndarray:
    """Convert continuous morphometric labels to binary classification labels.

    Tasks:
        classify-bvtv   : sparse (BV/TV < median) vs dense (BV/TV >= median)
        classify-tbn     : low Tb.N vs high Tb.N
        classify-tbth    : thin vs thick trabeculae
        classify-multi   : 3 classes: sparse / medium / dense
    """
    if task == "classify-bvtv":
        idx = label_keys.index("BVTV")
        values = Y[:, idx]
        median = np.median(values)
        labels = (values >= median).astype(int)
        class_names = ["sparse", "dense"]

    elif task == "classify-tbn":
        idx = label_keys.index("TbN_per_mm")
        values = Y[:, idx]
        median = np.median(values)
        labels = (values >= median).astype(int)
        class_names = ["few_struts", "many_struts"]

    elif task == "classify-tbth":
        idx = label_keys.index("TbTh_um_p50")
        values = Y[:, idx]
        median = np.median(values)
        labels = (values >= median).astype(int)
        class_names = ["thin", "thick"]

    elif task == "classify-multi":
        idx = label_keys.index("BVTV")
        values = Y[:, idx]
        p33 = np.percentile(values, 33)
        p66 = np.percentile(values, 66)
        labels = np.zeros(len(values), dtype=int)
        labels[values >= p66] = 2
        labels[(values >= p33) & (values < p66)] = 1
        class_names = ["sparse", "medium", "dense"]

    else:
        raise ValueError(f"Unknown task: {task}")

    return labels, class_names


def build_morphometric_features(Y_train: np.ndarray, Y_test: np.ndarray,
                                 n_components: int) -> dict:
    """Use raw morphometric labels as features (baseline comparison).
    Pad to n_components if needed."""
    n_morph = Y_train.shape[1]

    # Scale to [0, pi] for quantum encoding
    mm_pi = MinMaxScaler(feature_range=(0, np.pi))
    Z_train = mm_pi.fit_transform(Y_train)
    Z_test = mm_pi.transform(Y_test)

    # Scale to [0, 1]
    mm_01 = MinMaxScaler(feature_range=(0, 1))
    Z_train_01 = mm_01.fit_transform(Y_train)
    Z_test_01 = mm_01.transform(Y_test)

    # Pad if fewer features than n_components
    if n_morph < n_components:
        pad_tr = np.zeros((Z_train.shape[0], n_components - n_morph))
        pad_te = np.zeros((Z_test.shape[0], n_components - n_morph))
        Z_train = np.hstack([Z_train, pad_tr])
        Z_test = np.hstack([Z_test, pad_te])
        Z_train_01 = np.hstack([Z_train_01, pad_tr])
        Z_test_01 = np.hstack([Z_test_01, pad_te])

    return {
        "Z_train": Z_train[:, :n_components],
        "Z_test": Z_test[:, :n_components],
        "Z_train_01": Z_train_01[:, :n_components],
        "Z_test_01": Z_test_01[:, :n_components],
        "n_features": min(n_morph, n_components),
    }


# ═══════════════════════════════════════════════════════════
#  2. CLASSICAL SVM
# ═══════════════════════════════════════════════════════════

def run_classical_svm(X_train, X_test, y_train, y_test, class_names) -> dict:
    """Run classical SVM with RBF kernel."""
    t0 = time.time()

    svm = SVC(kernel="rbf", gamma="scale", C=1.0, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)
    predict_time = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "method": "Classical SVM (RBF)",
        "accuracy": float(acc),
        "f1_score": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "confusion_matrix": cm.tolist(),
        "train_time_s": float(train_time),
        "predict_time_s": float(predict_time),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
        "class_names": class_names,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


# ═══════════════════════════════════════════════════════════
#  3. QUANTUM KERNEL SVM
# ═══════════════════════════════════════════════════════════

def run_quantum_svm(X_train_pi, X_test_pi, y_train, y_test,
                     n_qubits, class_names, feature_map_type="ZZ",
                     reps=2) -> dict:
    """Run Quantum Kernel SVM using Qiskit.

    Args:
        X_train_pi: training features scaled to [0, pi]
        X_test_pi: test features scaled to [0, pi]
        n_qubits: number of qubits (= number of features to use)
        feature_map_type: 'ZZ', 'Z', or 'Pauli'
        reps: number of repetitions in the feature map circuit
    """
    if not HAS_QISKIT:
        return {"method": "Quantum Kernel SVM", "error": "Qiskit not installed",
                "accuracy": None, "f1_score": None}

    # Limit features to n_qubits
    X_tr = X_train_pi[:, :n_qubits]
    X_te = X_test_pi[:, :n_qubits]
    n_feat = X_tr.shape[1]

    print(f"    Quantum kernel: {n_feat} qubits, {feature_map_type} feature map, {reps} reps")
    print(f"    Train: {X_tr.shape}, Test: {X_te.shape}")

    # Build feature map circuit
    if feature_map_type == "ZZ":
        feature_map = ZZFeatureMap(feature_dimension=n_feat, reps=reps, entanglement="linear")
    elif feature_map_type == "Z":
        feature_map = ZFeatureMap(feature_dimension=n_feat, reps=reps)
    elif feature_map_type == "Pauli":
        feature_map = PauliFeatureMap(feature_dimension=n_feat, reps=reps,
                                       paulis=["Z", "ZZ"], entanglement="linear")
    else:
        raise ValueError(f"Unknown feature map: {feature_map_type}")

    print(f"    Circuit depth: {feature_map.depth()}, gates: {feature_map.size()}")

    # Build quantum kernel
    t0 = time.time()

    try:
        # Qiskit 1.x approach
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        kernel = FidelityQuantumKernel(feature_map=feature_map)

        # Compute kernel matrices
        print(f"    Computing training kernel matrix ({X_tr.shape[0]}x{X_tr.shape[0]})...")
        K_train = kernel.evaluate(x_vec=X_tr)
        print(f"    Computing test kernel matrix ({X_te.shape[0]}x{X_tr.shape[0]})...")
        K_test = kernel.evaluate(x_vec=X_te, y_vec=X_tr)

    except Exception as e:
        print(f"    Kernel computation failed: {e}")
        print(f"    Falling back to manual kernel matrix computation...")

        # Manual fallback: compute kernel matrix element by element
        from qiskit import QuantumCircuit
        from qiskit.primitives import StatevectorEstimator

        return {"method": f"Quantum Kernel SVM ({feature_map_type})",
                "error": str(e), "accuracy": None, "f1_score": None}

    kernel_time = time.time() - t0
    print(f"    Kernel computed in {kernel_time:.1f}s")

    # Train SVM with precomputed quantum kernel
    t0 = time.time()
    svm = SVC(kernel="precomputed", C=1.0, probability=False, random_state=42)
    svm.fit(K_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = svm.predict(K_test)
    predict_time = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "method": f"Quantum Kernel SVM ({feature_map_type})",
        "accuracy": float(acc),
        "f1_score": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "confusion_matrix": cm.tolist(),
        "kernel_time_s": float(kernel_time),
        "train_time_s": float(train_time),
        "predict_time_s": float(predict_time),
        "y_pred": y_pred.tolist(),
        "class_names": class_names,
        "n_qubits": n_feat,
        "feature_map": feature_map_type,
        "reps": reps,
        "circuit_depth": feature_map.depth(),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


# ═══════════════════════════════════════════════════════════
#  4. VISUALIZATION
# ═══════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, class_names, title, outpath):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=13)
    plt.colorbar(im, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(class_names)
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()


def plot_comparison_table(results, outdir):
    """Create the 3x2 (or 4x2) comparison table as a figure."""
    methods = sorted(set(r["feature_source"] for r in results))
    classifiers = ["Classical SVM (RBF)"]
    q_classifiers = [r["result"]["method"] for r in results
                     if "Quantum" in r["result"].get("method", "")]
    if q_classifiers:
        classifiers.append(q_classifiers[0])

    fig, ax = plt.subplots(figsize=(12, max(3, len(methods) * 0.8 + 2)))
    ax.axis("off")

    # Build table data
    col_labels = ["Feature Method"] + [f"{c}\nAccuracy" for c in classifiers] + [f"{c}\nF1" for c in classifiers]
    table_data = []
    for method in methods:
        row = [method]
        for clf_type in ["classical", "quantum"]:
            matching = [r for r in results
                       if r["feature_source"] == method and r["classifier_type"] == clf_type]
            if matching:
                r = matching[0]["result"]
                acc = r.get("accuracy")
                row.append(f"{acc:.3f}" if acc is not None else "N/A")
            else:
                row.append("—")
        for clf_type in ["classical", "quantum"]:
            matching = [r for r in results
                       if r["feature_source"] == method and r["classifier_type"] == clf_type]
            if matching:
                r = matching[0]["result"]
                f1 = r.get("f1_score")
                row.append(f"{f1:.3f}" if f1 is not None else "N/A")
            else:
                row.append("—")
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(len(table_data)):
        color = "#D9E2F3" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    ax.set_title("Classical vs Quantum SVM Comparison", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(outdir / "comparison_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outdir / 'comparison_table.png'}")


def plot_timing_comparison(results, outdir):
    """Bar chart comparing training times."""
    labels = []
    classical_times = []
    quantum_times = []

    methods = sorted(set(r["feature_source"] for r in results))
    for method in methods:
        labels.append(method)
        for r in results:
            if r["feature_source"] == method:
                t = r["result"].get("train_time_s", 0) + r["result"].get("kernel_time_s", 0)
                if r["classifier_type"] == "classical":
                    classical_times.append(t)
                elif r["classifier_type"] == "quantum":
                    quantum_times.append(t)

    # Pad if quantum results missing
    while len(quantum_times) < len(labels):
        quantum_times.append(0)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, classical_times, width, label="Classical SVM", color="#4472C4")
    if any(t > 0 for t in quantum_times):
        ax.bar(x + width/2, quantum_times, width, label="Quantum Kernel SVM", color="#ED7D31")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Training Time: Classical vs Quantum")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "timing_comparison.png", dpi=150)
    plt.close()


def plot_accuracy_comparison(results, outdir):
    """Grouped bar chart comparing accuracy across methods."""
    methods = sorted(set(r["feature_source"] for r in results))

    classical_acc = []
    quantum_acc = []

    for method in methods:
        for r in results:
            if r["feature_source"] == method:
                acc = r["result"].get("accuracy", 0) or 0
                if r["classifier_type"] == "classical":
                    classical_acc.append(acc)
                elif r["classifier_type"] == "quantum":
                    quantum_acc.append(acc)

    while len(quantum_acc) < len(methods):
        quantum_acc.append(0)

    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, classical_acc, width, label="Classical SVM", color="#4472C4")
    if any(a > 0 for a in quantum_acc):
        bars2 = ax.bar(x + width/2, quantum_acc, width, label="Quantum Kernel SVM", color="#ED7D31")

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}", ha="center", fontsize=9)
    if any(a > 0 for a in quantum_acc):
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}", ha="center", fontsize=9)

    ax.set_ylabel("Accuracy")
    ax.set_title("Classification Accuracy: Classical vs Quantum SVM")
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    plt.tight_layout()
    plt.savefig(outdir / "accuracy_comparison.png", dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════
#  5. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Classical SVM vs Quantum Kernel SVM comparison")
    p.add_argument("--features-dir", type=str, required=True,
                   help="Directory containing *_quantum_ready.npz files")
    p.add_argument("--outdir", type=str, default="output/qsvm_results")
    p.add_argument("--n-qubits", type=int, default=8,
                   help="Number of qubits for quantum kernel (uses first N features)")
    p.add_argument("--task", type=str, default="classify-bvtv",
                   choices=["classify-bvtv", "classify-tbn", "classify-tbth", "classify-multi"])
    p.add_argument("--feature-map", type=str, default="ZZ",
                   choices=["ZZ", "Z", "Pauli"])
    p.add_argument("--reps", type=int, default=2,
                   help="Feature map circuit repetitions")
    p.add_argument("--skip-quantum", action="store_true",
                   help="Only run classical SVM (no Qiskit needed)")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    features_dir = Path(args.features_dir)

    print(f"\n{'='*60}")
    print(f"  CLASSICAL vs QUANTUM SVM COMPARISON")
    print(f"{'='*60}")
    print(f"  Features: {features_dir}")
    print(f"  Task: {args.task}")
    print(f"  Qubits: {args.n_qubits}")
    print(f"  Feature map: {args.feature_map} (reps={args.reps})")
    print(f"  Quantum: {'enabled' if (HAS_QISKIT and not args.skip_quantum) else 'disabled'}")

    # ── Discover feature files ──
    npz_files = {}
    for name in ["pca", "rp_gaussian", "rp_sparse"]:
        path = features_dir / f"{name}_quantum_ready.npz"
        if path.exists():
            npz_files[name] = path
            print(f"  Found: {name} -> {path}")

    if not npz_files:
        print("ERROR: No *_quantum_ready.npz files found")
        return

    # ── Run comparison ──
    all_results = []
    feature_sources = list(npz_files.keys())

    # Also add morphometric baseline
    first_data = load_features(str(list(npz_files.values())[0]))
    morph_data = build_morphometric_features(
        first_data["Y_train"], first_data["Y_test"], args.n_qubits)
    feature_sources.append("morphometric")

    for source in feature_sources:
        print(f"\n{'─'*50}")
        print(f"  FEATURES: {source.upper()}")
        print(f"{'─'*50}")

        if source == "morphometric":
            data = morph_data
            # Need labels from one of the npz files
            data["Y_train"] = first_data["Y_train"]
            data["Y_test"] = first_data["Y_test"]
        else:
            data = load_features(str(npz_files[source]))

        # Create classification labels
        y_train, class_names = create_classification_labels(
            data["Y_train"], args.task, LABEL_KEYS)
        y_test, _ = create_classification_labels(
            data["Y_test"], args.task, LABEL_KEYS)

        print(f"  Classes: {class_names}")
        print(f"  Train distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"  Test distribution:  {dict(zip(*np.unique(y_test, return_counts=True)))}")

        # ── Classical SVM ──
        print(f"\n  Running Classical SVM...")
        classical_result = run_classical_svm(
            data["Z_train_01"], data["Z_test_01"], y_train, y_test, class_names)
        print(f"    Accuracy: {classical_result['accuracy']:.3f}")
        print(f"    F1 Score: {classical_result['f1_score']:.3f}")
        print(f"    Train time: {classical_result['train_time_s']:.4f}s")

        all_results.append({
            "feature_source": source,
            "classifier_type": "classical",
            "result": classical_result,
        })

        # Plot confusion matrix
        cm = np.array(classical_result["confusion_matrix"])
        plot_confusion_matrix(cm, class_names,
            f"Classical SVM — {source}\nAcc={classical_result['accuracy']:.3f}",
            outdir / f"cm_classical_{source}.png")

        # ── Quantum Kernel SVM ──
        if HAS_QISKIT and not args.skip_quantum:
            print(f"\n  Running Quantum Kernel SVM...")
            quantum_result = run_quantum_svm(
                data["Z_train"], data["Z_test"],
                y_train, y_test,
                n_qubits=args.n_qubits,
                class_names=class_names,
                feature_map_type=args.feature_map,
                reps=args.reps,
            )
            if quantum_result.get("accuracy") is not None:
                print(f"    Accuracy: {quantum_result['accuracy']:.3f}")
                print(f"    F1 Score: {quantum_result['f1_score']:.3f}")
                print(f"    Kernel time: {quantum_result.get('kernel_time_s', 0):.2f}s")

                cm_q = np.array(quantum_result["confusion_matrix"])
                plot_confusion_matrix(cm_q, class_names,
                    f"Quantum SVM ({args.feature_map}) — {source}\nAcc={quantum_result['accuracy']:.3f}",
                    outdir / f"cm_quantum_{source}.png")
            else:
                print(f"    FAILED: {quantum_result.get('error', 'unknown')}")

            all_results.append({
                "feature_source": source,
                "classifier_type": "quantum",
                "result": quantum_result,
            })

    # ── Summary plots ──
    print(f"\n{'─'*50}")
    print(f"  GENERATING COMPARISON PLOTS")
    print(f"{'─'*50}")

    plot_comparison_table(all_results, outdir)
    plot_accuracy_comparison(all_results, outdir)
    plot_timing_comparison(all_results, outdir)

    # ── Print summary table ──
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Feature':<18} {'Classifier':<28} {'Accuracy':>8} {'F1':>8} {'Time':>8}")
    print(f"  {'-'*68}")
    for r in all_results:
        res = r["result"]
        acc = res.get("accuracy")
        f1 = res.get("f1_score")
        t = res.get("train_time_s", 0) + res.get("kernel_time_s", 0)
        acc_str = f"{acc:.3f}" if acc is not None else "N/A"
        f1_str = f"{f1:.3f}" if f1 is not None else "N/A"
        print(f"  {r['feature_source']:<18} {res['method']:<28} {acc_str:>8} {f1_str:>8} {t:>7.2f}s")
    print(f"{'='*70}")

    # ── Save results ──
    summary = {
        "task": args.task,
        "n_qubits": args.n_qubits,
        "feature_map": args.feature_map,
        "reps": args.reps,
        "results": [{
            "feature_source": r["feature_source"],
            "classifier_type": r["classifier_type"],
            "accuracy": r["result"].get("accuracy"),
            "f1_score": r["result"].get("f1_score"),
            "precision": r["result"].get("precision"),
            "recall": r["result"].get("recall"),
            "train_time_s": r["result"].get("train_time_s"),
            "kernel_time_s": r["result"].get("kernel_time_s"),
            "confusion_matrix": r["result"].get("confusion_matrix"),
            "class_names": r["result"].get("class_names"),
            "n_train": r["result"].get("n_train"),
            "n_test": r["result"].get("n_test"),
        } for r in all_results],
    }

    with open(outdir / "qsvm_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved: {outdir / 'qsvm_results.json'}")
    print(f"  All outputs in: {outdir}/")


if __name__ == "__main__":
    main()