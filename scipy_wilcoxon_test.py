#!/usr/bin/env python3
"""
Wilcoxon signed-rank test for UMAP classical vs quantum SVM accuracy.
Requires per-fold accuracy arrays saved by qsvm_tighter_v1.py.

Usage:
  python scipy_wilcoxon_test.py --results-dir output/v8_qsvm_tight

Expects files like:
  <results-dir>/umap_classical_fold_scores.npy   (25 values)
  <results-dir>/umap_quantum_fold_scores.npy     (25 values)

If your fold scores are stored differently (e.g. inside a .npz or .json),
update the loading section below.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

METHODS = ["pca", "rp_gaussian", "pls", "umap"]


def load_fold_scores(results_dir: Path, method: str):
    """Try multiple file formats to find per-fold scores."""
    # Option 1: separate .npy files
    c_npy = results_dir / f"{method}_classical_fold_scores.npy"
    q_npy = results_dir / f"{method}_quantum_fold_scores.npy"
    if c_npy.exists() and q_npy.exists():
        return np.load(c_npy), np.load(q_npy)

    # Option 2: combined .npz file
    npz = results_dir / f"{method}_cv_results.npz"
    if npz.exists():
        data = np.load(npz)
        keys = list(data.keys())
        print(f"  Found {npz.name} with keys: {keys}")
        # Common key patterns
        c_key = next((k for k in keys if "classical" in k.lower() and "fold" in k.lower()), None)
        q_key = next((k for k in keys if "quantum" in k.lower() and "fold" in k.lower()), None)
        if c_key and q_key:
            return data[c_key], data[q_key]

    # Option 3: JSON with fold arrays
    for pattern in [f"{method}_results.json", "all_results.json", "qsvm_results.json"]:
        jpath = results_dir / pattern
        if jpath.exists():
            with open(jpath) as f:
                data = json.load(f)
            print(f"  Found {jpath.name}, searching for fold scores...")
            # Navigate nested structures
            if method in data:
                d = data[method]
            else:
                d = data
            for prefix in ["classical", "quantum"]:
                for key in d.get(prefix, {}):
                    if "fold" in key.lower() and "score" in key.lower():
                        print(f"    {prefix}: key='{key}' ({len(d[prefix][key])} values)")

    return None, None


def run_wilcoxon(classical, quantum, method, alternative="two-sided"):
    """Run Wilcoxon signed-rank test and print results."""
    diff = quantum - classical
    stat, p = wilcoxon(diff, alternative=alternative)
    mean_diff = diff.mean()
    print(f"\n{'='*50}")
    print(f"  {method.upper()}")
    print(f"{'='*50}")
    print(f"  Classical: {classical.mean():.4f} +/- {classical.std():.4f}")
    print(f"  Quantum:   {quantum.mean():.4f} +/- {quantum.std():.4f}")
    print(f"  Mean diff:  {mean_diff:+.4f}")
    print(f"  Wilcoxon stat: {stat:.1f}")
    print(f"  p-value ({alternative}): {p:.6f}")
    if p < 0.001:
        sig = "*** (p < 0.001)"
    elif p < 0.01:
        sig = "** (p < 0.01)"
    elif p < 0.05:
        sig = "* (p < 0.05)"
    else:
        sig = "n.s."
    print(f"  Significance: {sig}")
    return stat, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="output/v8_qsvm_tight")
    parser.add_argument("--alternative", type=str, default="two-sided",
                        choices=["two-sided", "greater", "less"],
                        help="Alternative hypothesis. 'greater' = quantum > classical")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: {results_dir} not found")
        return

    print(f"Looking for fold scores in: {results_dir}")
    print(f"Files found: {sorted(p.name for p in results_dir.iterdir())[:20]}")

    found_any = False
    for method in METHODS:
        print(f"\nLoading {method}...")
        classical, quantum = load_fold_scores(results_dir, method)
        if classical is not None and quantum is not None:
            found_any = True
            assert len(classical) == len(quantum), \
                f"Fold count mismatch: {len(classical)} vs {len(quantum)}"
            run_wilcoxon(classical, quantum, method, args.alternative)
        else:
            print(f"  Could not find per-fold scores for {method}")
            print(f"  Check qsvm_tighter_v1.py — you may need to re-run")
            print(f"  with fold scores saved to .npy files.")

    if not found_any:
        print("\n" + "="*50)
        print("No fold scores found. To generate them, add this to your")
        print("CV loop in qsvm_tighter_v1.py after computing scores:")
        print()
        print("  np.save(f'{results_dir}/{method}_classical_fold_scores.npy',")
        print("          classical_scores)")
        print("  np.save(f'{results_dir}/{method}_quantum_fold_scores.npy',")
        print("          quantum_scores)")


if __name__ == "__main__":
    main()