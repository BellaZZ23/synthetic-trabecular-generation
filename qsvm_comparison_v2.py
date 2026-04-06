#!/usr/bin/env python3
"""
qsvm_comparison_v2.py  v1.2
Updates from v1.1:
  - Auto-discovers ALL quantum-ready .npz files (PCA, RP Gaussian, RP Sparse, PLS, UMAP)
  - Removed duplicate RepeatedStratifiedKFold import
  - Added --skip-morphometric flag (circular for BV/TV task)
  - Added --features flag to run specific feature sets only
  - Reports circuit depth in summary JSON
  - Default cv_repeats raised to 5 (25 evaluations)
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit

try:
    from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    HAS_QISKIT = True
    print("Qiskit available: quantum kernel enabled")
except ImportError:
    HAS_QISKIT = False
    print("WARNING: Qiskit not installed. Only classical SVM will run.")

LABEL_KEYS        = ["BVTV", "TbTh_um_p50", "TbN_per_mm", "TbSp_um_p50"]
ALL_FEATURE_NAMES = ["pca", "rp_gaussian", "rp_sparse", "pls", "umap"]


def discover_feature_files(features_dir, requested=None):
    npz_files = {}
    for name in ALL_FEATURE_NAMES:
        path = features_dir / f"{name}_quantum_ready.npz"
        if path.exists():
            if requested is None or name in requested:
                npz_files[name] = path
                print(f"  Found: {name}")
    return npz_files


def load_features(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    out  = {"Y_train": data["Y_train"], "Y_test": data["Y_test"],
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
    elif task == "classify-tbth":
        vals = Y[:, LABEL_KEYS.index("TbTh_um_p50")]
        return (vals >= np.median(vals)).astype(int), ["thin", "thick"]
    elif task == "classify-multi":
        vals = Y[:, LABEL_KEYS.index("BVTV")]
        p33  = np.percentile(vals, 33); p66 = np.percentile(vals, 66)
        labels = np.zeros(len(vals), dtype=int)
        labels[vals >= p66] = 2; labels[(vals >= p33) & (vals < p66)] = 1
        return labels, ["sparse", "medium", "dense"]
    raise ValueError(f"Unknown task: {task}")


def build_morphometric_features(Y_train, Y_test, n_components):
    mm_pi = MinMaxScaler(feature_range=(0, np.pi))
    mm_01 = MinMaxScaler(feature_range=(0, 1))
    Zq_tr = mm_pi.fit_transform(Y_train); Zq_te = mm_pi.transform(Y_test)
    Z1_tr = mm_01.fit_transform(Y_train); Z1_te = mm_01.transform(Y_test)
    n = Y_train.shape[1]
    if n < n_components:
        p_tr = np.zeros((Y_train.shape[0], n_components-n))
        p_te = np.zeros((Y_test.shape[0],  n_components-n))
        Zq_tr = np.hstack([Zq_tr,p_tr]); Zq_te = np.hstack([Zq_te,p_te])
        Z1_tr = np.hstack([Z1_tr,p_tr]); Z1_te = np.hstack([Z1_te,p_te])
    nc = min(n, n_components)
    return {"Z_train": Zq_tr[:,:nc], "Z_test": Zq_te[:,:nc],
            "Z_train_01": Z1_tr[:,:nc], "Z_test_01": Z1_te[:,:nc],
            "Y_train": Y_train, "Y_test": Y_test, "n_features": nc}


def run_classical_svm_cv(X_all, y_all, class_names, n_splits=5, n_repeats=5, seed=42):
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    accs, f1s, precs, recs, times = [], [], [], [], []
    for tr, te in rkf.split(X_all, y_all):
        t0  = time.time()
        svm = SVC(kernel="rbf", gamma="scale", C=1.0, random_state=seed)
        svm.fit(X_all[tr], y_all[tr]); yp = svm.predict(X_all[te])
        times.append(time.time()-t0)
        accs.append(accuracy_score(y_all[te], yp))
        f1s.append(f1_score(y_all[te], yp, average="weighted", zero_division=0))
        precs.append(precision_score(y_all[te], yp, average="weighted", zero_division=0))
        recs.append(recall_score(y_all[te], yp, average="weighted", zero_division=0))
    n_evals = len(accs)
    print(f"    Repeated CV ({n_splits}x{n_repeats}={n_evals} folds):")
    print(f"    Accuracy: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"    F1 Score: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
    return {"method": "Classical SVM (RBF) — Repeated CV",
            "cv_type": f"RepeatedStratifiedKFold({n_splits}x{n_repeats})",
            "accuracy": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "f1_score": float(np.mean(f1s)),  "f1_std":      float(np.std(f1s)),
            "precision": float(np.mean(precs)), "recall": float(np.mean(recs)),
            "all_accuracies": [float(a) for a in accs],
            "train_time_s": float(np.mean(times)),
            "class_names": class_names, "n_samples": len(y_all), "n_evaluations": n_evals}


def run_classical_svm_single(X_tr, X_te, y_tr, y_te, class_names):
    t0 = time.time()
    svm = SVC(kernel="rbf", gamma="scale", C=1.0, random_state=42)
    svm.fit(X_tr, y_tr); tt = time.time()-t0; yp = svm.predict(X_te)
    return {"method": "Classical SVM (RBF) — single split",
            "accuracy": float(accuracy_score(y_te, yp)),
            "f1_score": float(f1_score(y_te, yp, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_te, yp).tolist(),
            "train_time_s": float(tt), "class_names": class_names,
            "n_train": len(y_tr), "n_test": len(y_te)}


def run_quantum_svm(X_tr_pi, X_te_pi, y_tr, y_te, n_qubits, class_names, fmap="ZZ", reps=2):
    if not HAS_QISKIT:
        return {"method": "Quantum Kernel SVM", "error": "Qiskit not installed",
                "accuracy": None, "f1_score": None}
    X_tr = X_tr_pi[:,:n_qubits]; X_te = X_te_pi[:,:n_qubits]; nf = X_tr.shape[1]
    print(f"    Quantum kernel: {nf} qubits, {fmap}, {reps} reps")
    if fmap == "ZZ":   fm = ZZFeatureMap(feature_dimension=nf, reps=reps, entanglement="linear")
    elif fmap == "Z":  fm = ZFeatureMap(feature_dimension=nf, reps=reps)
    else:              fm = PauliFeatureMap(feature_dimension=nf, reps=reps, paulis=["Z","ZZ"], entanglement="linear")
    depth = fm.depth(); t0 = time.time()
    try:
        k = FidelityQuantumKernel(feature_map=fm)
        print(f"    Computing training kernel ({X_tr.shape[0]}x{X_tr.shape[0]})...")
        Ktr = k.evaluate(x_vec=X_tr)
        print(f"    Computing test kernel ({X_te.shape[0]}x{X_tr.shape[0]})...")
        Kte = k.evaluate(x_vec=X_te, y_vec=X_tr)
    except Exception as e:
        return {"method": f"Quantum Kernel SVM ({fmap})", "error": str(e), "accuracy": None, "f1_score": None}
    kt = time.time()-t0; print(f"    Kernel computed in {kt:.1f}s")
    svm = SVC(kernel="precomputed", C=1.0, random_state=42)
    svm.fit(Ktr, y_tr); yp = svm.predict(Kte)
    acc = accuracy_score(y_te, yp); f1 = f1_score(y_te, yp, average="weighted", zero_division=0)
    print(f"    Accuracy: {acc:.3f}, F1: {f1:.3f}")
    return {"method": f"Quantum Kernel SVM ({fmap})",
            "cv_type": "single stratified split (CV too slow for quantum)",
            "accuracy": float(acc), "accuracy_std": None,
            "f1_score": float(f1), "f1_std": None,
            "precision": float(precision_score(y_te, yp, average="weighted", zero_division=0)),
            "recall":    float(recall_score(y_te, yp, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_te, yp).tolist(),
            "kernel_time_s": float(kt), "class_names": class_names,
            "n_qubits": nf, "feature_map": fmap, "reps": reps, "circuit_depth": depth,
            "n_train": len(y_tr), "n_test": len(y_te)}


def plot_confusion_matrix(cm, class_names, title, outpath):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap="Blues"); ax.set_title(title, fontsize=12); plt.colorbar(im,ax=ax)
    t = np.arange(len(class_names))
    ax.set_xticks(t); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(t); ax.set_yticklabels(class_names)
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                    color="white" if cm[i,j]>cm.max()/2 else "black",fontsize=14)
    plt.tight_layout(); plt.savefig(outpath,dpi=150); plt.close()


def plot_accuracy_comparison(results, outdir):
    methods = sorted(set(r["feature_source"] for r in results))
    ca = {}; cs = {}
    for r in results:
        k = (r["feature_source"], r["classifier_type"])
        ca[k] = r["result"].get("accuracy") or 0
        cs[k] = r["result"].get("accuracy_std") or 0
    x = np.arange(len(methods)); w = 0.35
    fig,ax = plt.subplots(figsize=(max(10,len(methods)*2),5))
    c_a = [ca.get((m,"classical"),0) for m in methods]
    c_s = [cs.get((m,"classical"),0) for m in methods]
    q_a = [ca.get((m,"quantum"),  0) for m in methods]
    b1 = ax.bar(x-w/2, c_a, w, yerr=c_s, capsize=4, label="Classical SVM (RBF)", color="#4472C4")
    if any(a>0 for a in q_a):
        b2 = ax.bar(x+w/2, q_a, w, label="Quantum Kernel SVM", color="#ED7D31")
        for b in b2:
            h = b.get_height()
            if h>0: ax.text(b.get_x()+b.get_width()/2, h+0.01, f"{h:.3f}", ha="center", fontsize=8)
    for b in b1:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h+0.01, f"{h:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0,1.15)
    ax.set_title("Classification Accuracy: Classical vs Quantum SVM")
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="Random baseline")
    ax.legend(); plt.tight_layout()
    plt.savefig(outdir/"accuracy_comparison.png",dpi=150); plt.close()
    print(f"  Saved: {outdir/'accuracy_comparison.png'}")


def plot_comparison_table(results, outdir):
    methods = sorted(set(r["feature_source"] for r in results))
    rows = []
    for m in methods:
        row = [m]
        for ct in ["classical","quantum"]:
            match = [r for r in results if r["feature_source"]==m and r["classifier_type"]==ct]
            if match:
                res = match[0]["result"]
                acc=res.get("accuracy"); std=res.get("accuracy_std")
                f1=res.get("f1_score");  fstd=res.get("f1_std")
                acc_s = f"{acc:.3f}+/-{std:.3f}" if (acc and std) else (f"{acc:.3f}" if acc else "N/A")
                f1_s  = f"{f1:.3f}+/-{fstd:.3f}" if (f1 and fstd)  else (f"{f1:.3f}"  if f1  else "N/A")
                row += [acc_s, f1_s]
            else: row += ["—","—"]
        rows.append(row)
    cols = ["Feature","Classical Acc","Classical F1","Quantum Acc","Quantum F1"]
    fig,ax = plt.subplots(figsize=(14,max(3,len(rows)*0.9+2))); ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1,1.8)
    for j in range(len(cols)):
        tbl[0,j].set_facecolor("#4472C4"); tbl[0,j].set_text_props(color="white",fontweight="bold")
    for i in range(len(rows)):
        c = "#D9E2F3" if i%2==0 else "white"
        for j in range(len(cols)): tbl[i+1,j].set_facecolor(c)
    ax.set_title("Classical vs Quantum SVM Comparison",fontsize=13,fontweight="bold",pad=20)
    plt.tight_layout(); plt.savefig(outdir/"comparison_table.png",dpi=150,bbox_inches="tight"); plt.close()
    print(f"  Saved: {outdir/'comparison_table.png'}")


def main():
    p = argparse.ArgumentParser(description="Classical vs Quantum SVM v1.2")
    p.add_argument("--features-dir",      type=str, required=True)
    p.add_argument("--outdir",            type=str, default="output/qsvm_results")
    p.add_argument("--n-qubits",          type=int, default=8)
    p.add_argument("--task",              type=str, default="classify-bvtv",
                   choices=["classify-bvtv","classify-tbn","classify-tbth","classify-multi"])
    p.add_argument("--feature-map",       type=str, default="ZZ", choices=["ZZ","Z","Pauli"])
    p.add_argument("--reps",              type=int, default=2)
    p.add_argument("--cv-folds",          type=int, default=5)
    p.add_argument("--cv-repeats",        type=int, default=5)
    p.add_argument("--skip-quantum",      action="store_true")
    p.add_argument("--skip-morphometric", action="store_true",
                   help="Exclude morphometric baseline (circular for BV/TV task)")
    p.add_argument("--features",          nargs="+", default=None,
                   help="Specific feature sets e.g. --features pca rp_sparse umap")
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    features_dir = Path(args.features_dir)

    print(f"\n{'='*60}")
    print(f"  CLASSICAL vs QUANTUM SVM COMPARISON  (v1.2)")
    print(f"  Classical SVM: {args.cv_folds}x{args.cv_repeats} Repeated Stratified CV")
    print(f"{'='*60}")
    print(f"  Features dir: {features_dir}")
    print(f"  Task:         {args.task}")
    print(f"  Qubits:       {args.n_qubits}")
    print(f"  Quantum:      {'enabled' if (HAS_QISKIT and not args.skip_quantum) else 'disabled'}")
    print(f"  Morphometric: {'excluded' if args.skip_morphometric else 'included (may be circular)'}")

    npz_files = discover_feature_files(features_dir, requested=args.features)
    if not npz_files: print("ERROR: No *_quantum_ready.npz files found"); return

    first_data = load_features(str(list(npz_files.values())[0]))
    morph_data = build_morphometric_features(first_data["Y_train"], first_data["Y_test"], args.n_qubits)

    all_results    = []
    feature_sources = list(npz_files.keys())
    if not args.skip_morphometric: feature_sources.append("morphometric")

    for source in feature_sources:
        print(f"\n{'─'*50}\n  FEATURES: {source.upper()}\n{'─'*50}")
        data = morph_data if source=="morphometric" else load_features(str(npz_files[source]))
        if "Y_train" not in data: data["Y_train"]=first_data["Y_train"]; data["Y_test"]=first_data["Y_test"]

        y_tr, cnames = create_labels(data["Y_train"], args.task)
        y_te, _      = create_labels(data["Y_test"],  args.task)
        print(f"  Classes: {cnames}")
        print(f"  Train dist: {dict(zip(*np.unique(y_tr, return_counts=True)))}")
        print(f"  Test  dist: {dict(zip(*np.unique(y_te, return_counts=True)))}")

        print(f"\n  Running Classical SVM (Repeated CV)...")
        X_all = np.vstack([data["Z_train_01"], data["Z_test_01"]])
        y_all = np.concatenate([y_tr, y_te])
        cv_r = run_classical_svm_cv(X_all, y_all, cnames, args.cv_folds, args.cv_repeats)
        all_results.append({"feature_source": source, "classifier_type": "classical", "result": cv_r})

        sr = run_classical_svm_single(data["Z_train_01"], data["Z_test_01"], y_tr, y_te, cnames)
        cm = np.array(sr["confusion_matrix"])
        plot_confusion_matrix(cm, cnames, f"Classical SVM — {source}\nAcc={sr['accuracy']:.3f}",
                              outdir/f"cm_classical_{source}.png")

        if HAS_QISKIT and not args.skip_quantum:
            print(f"\n  Running Quantum Kernel SVM (single stratified split)...")
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            qt, qe = next(sss.split(X_all, y_all))
            qr = run_quantum_svm(X_all[qt], X_all[qe], y_all[qt], y_all[qe],
                                  args.n_qubits, cnames, args.feature_map, args.reps)
            if qr.get("accuracy") is not None:
                cmq = np.array(qr["confusion_matrix"])
                plot_confusion_matrix(cmq, cnames,
                    f"Quantum SVM ({args.feature_map}) — {source}\nAcc={qr['accuracy']:.3f}",
                    outdir/f"cm_quantum_{source}.png")
            all_results.append({"feature_source": source, "classifier_type": "quantum", "result": qr})

    print(f"\n{'='*70}\n  RESULTS SUMMARY\n{'='*70}")
    print(f"  {'Feature':<18} {'Classifier':<38} {'Accuracy':>10}   {'F1':>6}")
    print(f"  {'-'*74}")
    for r in all_results:
        res=r["result"]; acc=res.get("accuracy"); std=res.get("accuracy_std"); f1=res.get("f1_score")
        acc_s = f"{acc:.3f}+/-{std:.3f}" if (acc is not None and std) else (f"{acc:.3f}" if acc is not None else "N/A")
        f1_s  = f"{f1:.3f}" if f1 is not None else "N/A"
        print(f"  {r['feature_source']:<18} {res['method']:<38} {acc_s:>10}   {f1_s:>6}")
    print(f"{'='*70}")

    plot_comparison_table(all_results, outdir)
    plot_accuracy_comparison(all_results, outdir)

    summary = {"pipeline_version":"1.2","task":args.task,"n_qubits":args.n_qubits,
               "feature_map":args.feature_map,"reps":args.reps,
               "cv_folds":args.cv_folds,"cv_repeats":args.cv_repeats,
               "skip_morphometric":args.skip_morphometric,"feature_sets_run":feature_sources,
               "results":[{"feature_source":r["feature_source"],"classifier_type":r["classifier_type"],
                            "accuracy":r["result"].get("accuracy"),"accuracy_std":r["result"].get("accuracy_std"),
                            "f1_score":r["result"].get("f1_score"),"f1_std":r["result"].get("f1_std"),
                            "precision":r["result"].get("precision"),"recall":r["result"].get("recall"),
                            "train_time_s":r["result"].get("train_time_s"),"kernel_time_s":r["result"].get("kernel_time_s"),
                            "circuit_depth":r["result"].get("circuit_depth"),"cv_type":r["result"].get("cv_type"),
                            "n_evaluations":r["result"].get("n_evaluations"),"class_names":r["result"].get("class_names")
                           } for r in all_results]}
    with open(outdir/"qsvm_results.json","w") as f: json.dump(summary,f,indent=2,default=str)
    print(f"\n  Results saved: {outdir/'qsvm_results.json'}")


if __name__ == "__main__":
    main()