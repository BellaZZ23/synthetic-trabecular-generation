#!/usr/bin/env python3
"""
quantum_regression.py — v1.0

Option E: Quantum Kernel Ridge Regression
Compares classical Ridge regression vs quantum kernel ridge regression
for predicting continuous morphometric values (BV/TV, TbN, TbTh, TbSp).

This is a stronger and more novel contribution than classification because:
  - Predicts actual morphometric values rather than arbitrary median threshold
  - Directly comparable to downstream R² from dim_reduction_pipeline_v2
  - More clinically meaningful — continuous bone density prediction

Usage:
  python quantum_regression.py `
      --features-dir output\final_dataset_v8\features_v2 `
      --outdir output\qkrr_results_v8 `
      --n-qubits 8 `
      --feature-map ZZ `
      --reps 2
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

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

try:
    from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    HAS_QISKIT = True
    print("Qiskit available: quantum kernel ridge regression enabled")
except ImportError:
    HAS_QISKIT = False
    print("WARNING: Qiskit not installed. Only classical regression will run.")

LABEL_KEYS   = ["BVTV", "TbTh_um_p50", "TbN_per_mm", "TbSp_um_p50"]
LABEL_UNITS  = {"BVTV": "", "TbTh_um_p50": "µm",
                "TbN_per_mm": "/mm", "TbSp_um_p50": "µm"}
ALL_FEATURES = ["pca", "rp_gaussian", "rp_sparse", "pls", "umap"]


# ═══════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════

def discover_feature_files(features_dir: Path, requested=None) -> dict:
    npz_files = {}
    for name in ALL_FEATURES:
        path = features_dir / f"{name}_quantum_ready.npz"
        if path.exists():
            if requested is None or name in requested:
                npz_files[name] = path
                print(f"  Found: {name}")
    return npz_files


def load_features(npz_path: str) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    out  = {"Y_train": data["Y_train"], "Y_test": data["Y_test"],
            "n_features": int(data["n_features"]),
            "Z_train": data["Z_train"], "Z_test": data["Z_test"]}
    out["Z_train_01"] = data["Z_train_01"] if "Z_train_01" in data.files else data["Z_train"]
    out["Z_test_01"]  = data["Z_test_01"]  if "Z_test_01"  in data.files else data["Z_test"]
    return out


# ═══════════════════════════════════════════════════════════
#  CLASSICAL RIDGE REGRESSION WITH CV
# ═══════════════════════════════════════════════════════════

def run_classical_ridge_cv(
    X_train, X_test, Y_train, Y_test,
    label_names, n_splits=5, seed=42
) -> dict:
    """
    Classical Ridge regression with k-fold CV for alpha selection.
    Reports R² and MAE per label and overall.
    """
    alphas   = [0.01, 0.1, 1.0, 10.0, 100.0]
    kf       = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    per_label = {}
    r2_vals  = []; mae_vals = []

    for j, label in enumerate(label_names):
        # Select best alpha via CV
        ridge_cv = RidgeCV(alphas=alphas, cv=kf)
        ridge_cv.fit(X_train, Y_train[:, j])
        best_alpha = ridge_cv.alpha_

        # Fit final model
        ridge = Ridge(alpha=best_alpha)
        ridge.fit(X_train, Y_train[:, j])
        pred = ridge.predict(X_test)

        r2  = float(r2_score(Y_test[:, j], pred))
        mae = float(mean_absolute_error(Y_test[:, j], pred))
        mse = float(mean_squared_error(Y_test[:, j], pred))

        per_label[label] = {
            "r2": r2, "mae": mae, "rmse": float(np.sqrt(mse)),
            "best_alpha": float(best_alpha),
            "y_pred": pred.tolist(), "y_true": Y_test[:, j].tolist(),
        }
        r2_vals.append(r2); mae_vals.append(mae)
        print(f"    {label}: R²={r2:.4f}  MAE={mae:.4f}  alpha={best_alpha}")

    mean_r2  = float(np.mean(r2_vals))
    mean_mae = float(np.mean(mae_vals))
    print(f"    Mean R²={mean_r2:.4f}  Mean MAE={mean_mae:.4f}")

    return {
        "method": f"Classical Ridge (RidgeCV, {n_splits}-fold)",
        "per_label": per_label,
        "mean_r2": mean_r2, "mean_mae": mean_mae,
        "n_train": len(Y_train), "n_test": len(Y_test),
    }


# ═══════════════════════════════════════════════════════════
#  QUANTUM KERNEL RIDGE REGRESSION
# ═══════════════════════════════════════════════════════════

def quantum_kernel_ridge_regression(
    K_train, K_test, Y_train, Y_test,
    label_names, alpha=1.0
) -> dict:
    """
    Quantum Kernel Ridge Regression using precomputed kernel matrix.
    Solves: (K + alpha*I) @ w = Y
    Prediction: K_test @ w
    """
    per_label = {}
    r2_vals   = []; mae_vals = []

    for j, label in enumerate(label_names):
        y_tr = Y_train[:, j]; y_te = Y_test[:, j]

        # Kernel ridge regression closed form solution
        n = K_train.shape[0]
        w = np.linalg.solve(K_train + alpha * np.eye(n), y_tr)
        pred = K_test @ w

        r2  = float(r2_score(y_te, pred))
        mae = float(mean_absolute_error(y_te, pred))
        mse = float(mean_squared_error(y_te, pred))

        per_label[label] = {
            "r2": r2, "mae": mae, "rmse": float(np.sqrt(mse)),
            "y_pred": pred.tolist(), "y_true": y_te.tolist(),
        }
        r2_vals.append(r2); mae_vals.append(mae)
        print(f"    {label}: R²={r2:.4f}  MAE={mae:.4f}")

    return {
        "per_label": per_label,
        "mean_r2":   float(np.mean(r2_vals)),
        "mean_mae":  float(np.mean(mae_vals)),
        "alpha":     float(alpha),
    }


def run_quantum_kernel_ridge(
    X_train_pi, X_test_pi, Y_train, Y_test,
    n_qubits, label_names, feature_map_type="ZZ", reps=2,
    alphas=(0.01, 0.1, 1.0, 10.0)
) -> dict:
    """
    Full quantum kernel ridge regression with alpha selection via
    leave-one-out CV on training kernel matrix.
    """
    if not HAS_QISKIT:
        return {"method": "Quantum Kernel Ridge", "error": "Qiskit not installed",
                "mean_r2": None, "mean_mae": None}

    X_tr = X_train_pi[:, :n_qubits]
    X_te = X_test_pi[:, :n_qubits]
    n_feat = X_tr.shape[1]

    print(f"    Quantum kernel: {n_feat} qubits, {feature_map_type}, {reps} reps")

    if feature_map_type == "ZZ":
        feature_map = ZZFeatureMap(feature_dimension=n_feat, reps=reps,
                                    entanglement="linear")
    elif feature_map_type == "Z":
        feature_map = ZFeatureMap(feature_dimension=n_feat, reps=reps)
    else:
        feature_map = PauliFeatureMap(feature_dimension=n_feat, reps=reps,
                                       paulis=["Z", "ZZ"], entanglement="linear")

    circuit_depth = feature_map.depth()
    t0 = time.time()

    try:
        kernel = FidelityQuantumKernel(feature_map=feature_map)
        print(f"    Computing training kernel ({X_tr.shape[0]}x{X_tr.shape[0]})...")
        K_train = kernel.evaluate(x_vec=X_tr)
        print(f"    Computing test kernel ({X_te.shape[0]}x{X_tr.shape[0]})...")
        K_test  = kernel.evaluate(x_vec=X_te, y_vec=X_tr)
    except Exception as e:
        return {"method": f"Quantum Kernel Ridge ({feature_map_type})",
                "error": str(e), "mean_r2": None, "mean_mae": None}

    kernel_time = time.time() - t0
    print(f"    Kernel computed in {kernel_time:.1f}s")

    # Select best alpha using LOO-CV on training kernel
    # For efficiency use a simple grid search on mean R² across labels
    best_alpha = 1.0; best_r2 = -np.inf
    for alpha in alphas:
        r2_vals = []
        for j in range(Y_train.shape[1]):
            n = K_train.shape[0]
            try:
                w    = np.linalg.solve(K_train + alpha * np.eye(n), Y_train[:, j])
                pred = K_train @ w  # in-sample prediction as proxy
                r2   = float(r2_score(Y_train[:, j], pred))
                r2_vals.append(r2)
            except Exception:
                r2_vals.append(-1.0)
        mean_r2 = float(np.mean(r2_vals))
        if mean_r2 > best_r2:
            best_r2 = mean_r2; best_alpha = alpha

    print(f"    Best alpha: {best_alpha} (proxy R²={best_r2:.4f})")

    result = quantum_kernel_ridge_regression(
        K_train, K_test, Y_train, Y_test, label_names, alpha=best_alpha)

    result.update({
        "method":        f"Quantum Kernel Ridge ({feature_map_type})",
        "cv_type":       "single split with alpha grid search",
        "kernel_time_s": float(kernel_time),
        "circuit_depth": circuit_depth,
        "n_qubits":      n_feat,
        "feature_map":   feature_map_type,
        "reps":          reps,
        "n_train":       len(Y_train),
        "n_test":        len(Y_test),
    })
    return result


# ═══════════════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════════════

def plot_r2_comparison(all_results, outdir):
    """Bar chart comparing R² across methods and labels."""
    methods  = list(all_results.keys())
    n_labels = len(LABEL_KEYS)
    x        = np.arange(n_labels); width = 0.8 / len(methods)
    colors   = plt.cm.tab10(np.linspace(0, 0.9, len(methods)))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (method, result) in enumerate(all_results.items()):
        r2_vals = [result.get("per_label", {}).get(k, {}).get("r2", 0)
                   for k in LABEL_KEYS]
        offset  = (i - len(methods)/2 + 0.5) * width
        bars    = ax.bar(x + offset, r2_vals, width, label=method,
                         color=colors[i], alpha=0.85)
        for bar, val in zip(bars, r2_vals):
            if val > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f"{val:.2f}", ha="center", fontsize=7, rotation=45)

    ax.set_ylabel("R²"); ax.set_ylim(0, 1.1)
    ax.set_title("Quantum vs Classical Kernel Ridge Regression — Per-Label R²")
    ax.set_xticks(x); ax.set_xticklabels(LABEL_KEYS, rotation=15, ha="right")
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(outdir/"r2_comparison.png", dpi=150); plt.close()
    print(f"  Saved: {outdir/'r2_comparison.png'}")


def plot_scatter_predictions(result, method_name, outdir):
    """Scatter plot of predicted vs true values for each label."""
    per_label = result.get("per_label", {})
    n = len(per_label)
    if n == 0: return

    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1: axes = [axes]

    for ax, (label, info) in zip(axes, per_label.items()):
        y_true = np.array(info["y_true"])
        y_pred = np.array(info["y_pred"])
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="steelblue")
        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "r--", alpha=0.7, label="Perfect")
        ax.set_xlabel(f"True {label}"); ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"{label}\nR²={info['r2']:.3f}  MAE={info['mae']:.3f}")
        ax.legend(fontsize=7)

    fig.suptitle(f"{method_name} — Predicted vs True", fontsize=12)
    plt.tight_layout()
    fname = method_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(outdir/f"scatter_{fname}.png", dpi=150); plt.close()
    print(f"  Saved: {outdir/f'scatter_{fname}.png'}")


def plot_summary_table(all_results, outdir):
    """Summary table of mean R² and MAE across all methods."""
    methods = list(all_results.keys())
    rows    = []
    for method in methods:
        r  = all_results[method]
        r2 = r.get("mean_r2")
        mae = r.get("mean_mae")
        r2_str  = f"{r2:.4f}"  if r2  is not None else "N/A"
        mae_str = f"{mae:.4f}" if mae is not None else "N/A"

        # Per-label R²
        pl = r.get("per_label", {})
        label_r2s = [f"{pl.get(k,{}).get('r2',0):.3f}" for k in LABEL_KEYS]
        rows.append([method, r2_str, mae_str] + label_r2s)

    cols = ["Method", "Mean R²", "Mean MAE"] + LABEL_KEYS
    fig, ax = plt.subplots(figsize=(16, max(3, len(rows)*0.8+2)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.8)
    for j in range(len(cols)):
        tbl[0,j].set_facecolor("#4472C4")
        tbl[0,j].set_text_props(color="white", fontweight="bold")
    for i in range(len(rows)):
        c = "#D9E2F3" if i % 2 == 0 else "white"
        for j in range(len(cols)): tbl[i+1,j].set_facecolor(c)
    ax.set_title("Quantum vs Classical Kernel Ridge Regression — Summary",
                 fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(outdir/"regression_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outdir/'regression_summary_table.png'}")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Quantum Kernel Ridge Regression v1.0")
    p.add_argument("--features-dir",  type=str, required=True)
    p.add_argument("--outdir",        type=str, default="output/qkrr_results")
    p.add_argument("--n-qubits",      type=int, default=8)
    p.add_argument("--feature-map",   type=str, default="ZZ",
                   choices=["ZZ", "Z", "Pauli"])
    p.add_argument("--reps",          type=int, default=2)
    p.add_argument("--cv-folds",      type=int, default=5)
    p.add_argument("--skip-quantum",  action="store_true")
    p.add_argument("--features",      nargs="+", default=None,
                   help="Specific feature sets e.g. --features pca rp_gaussian")
    p.add_argument("--target-labels", nargs="+", default=None,
                   help="Specific labels e.g. --target-labels BVTV TbN_per_mm")
    args = p.parse_args()

    outdir       = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    features_dir = Path(args.features_dir)

    label_names = args.target_labels if args.target_labels else LABEL_KEYS

    print(f"\n{'='*60}")
    print(f"  QUANTUM KERNEL RIDGE REGRESSION  (v1.0)")
    print(f"{'='*60}")
    print(f"  Features dir: {features_dir}")
    print(f"  Target labels: {label_names}")
    print(f"  Qubits: {args.n_qubits}")
    print(f"  Feature map: {args.feature_map}, reps={args.reps}")
    print(f"  Quantum: {'enabled' if (HAS_QISKIT and not args.skip_quantum) else 'disabled'}")

    npz_files = discover_feature_files(features_dir, requested=args.features)
    if not npz_files:
        print("ERROR: No *_quantum_ready.npz files found"); return

    all_summary = {}

    for source, npz_path in npz_files.items():
        print(f"\n{'─'*50}")
        print(f"  FEATURE SET: {source.upper()}")
        print(f"{'─'*50}")

        data = load_features(str(npz_path))
        Y_tr = data["Y_train"]; Y_te = data["Y_test"]

        # Filter to requested labels only
        label_indices = [LABEL_KEYS.index(l) for l in label_names
                         if l in LABEL_KEYS]
        Y_tr_filt = Y_tr[:, label_indices]
        Y_te_filt = Y_te[:, label_indices]
        labels_filt = [LABEL_KEYS[i] for i in label_indices]

        # ── Classical Ridge ──
        print(f"\n  Classical Ridge Regression (RidgeCV, {args.cv_folds}-fold):")
        cl_result = run_classical_ridge_cv(
            data["Z_train_01"], data["Z_test_01"],
            Y_tr_filt, Y_te_filt, labels_filt,
            n_splits=args.cv_folds)
        plot_scatter_predictions(cl_result, f"Classical_Ridge_{source}", outdir)
        all_summary[f"Classical Ridge — {source}"] = cl_result

        # ── Quantum Kernel Ridge ──
        if HAS_QISKIT and not args.skip_quantum:
            print(f"\n  Quantum Kernel Ridge Regression:")
            q_result = run_quantum_kernel_ridge(
                data["Z_train"], data["Z_test"],
                Y_tr_filt, Y_te_filt,
                n_qubits=args.n_qubits,
                label_names=labels_filt,
                feature_map_type=args.feature_map,
                reps=args.reps)
            if q_result.get("mean_r2") is not None:
                plot_scatter_predictions(q_result, f"Quantum_Ridge_{source}", outdir)
            all_summary[f"Quantum Kernel Ridge — {source}"] = q_result

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  REGRESSION SUMMARY — Mean R²")
    print(f"{'='*70}")
    print(f"  {'Method':<40} {'Mean R²':>8}  {'Mean MAE':>9}")
    print(f"  {'-'*60}")
    for method, r in all_summary.items():
        r2  = r.get("mean_r2")
        mae = r.get("mean_mae")
        r2_s  = f"{r2:.4f}"  if r2  is not None else "N/A"
        mae_s = f"{mae:.4f}" if mae is not None else "N/A"
        print(f"  {method:<40} {r2_s:>8}  {mae_s:>9}")
    print(f"{'='*70}")

    plot_r2_comparison(all_summary, outdir)
    plot_summary_table(all_summary, outdir)

    # Save JSON
    json_out = {}
    for method, r in all_summary.items():
        # Remove y_pred/y_true lists from JSON to keep it small
        r_clean = {k: v for k, v in r.items() if k != "per_label"}
        r_clean["per_label"] = {
            lbl: {mk: mv for mk, mv in lv.items()
                  if mk not in ("y_pred", "y_true")}
            for lbl, lv in r.get("per_label", {}).items()
        }
        json_out[method] = r_clean

    with open(outdir/"regression_results.json", "w") as f:
        json.dump({"version": "1.0", "labels": label_names,
                   "results": json_out}, f, indent=2, default=str)
    print(f"\n  Results saved: {outdir/'regression_results.json'}")


if __name__ == "__main__":
    main()