#!/usr/bin/env python3
"""
Kernel matrix diagnostics: effective rank, spectral entropy, diagonal dominance.
Loads precomputed quantum kernel matrices (.npy) from qsvm_tighter_v1.py.

Usage:
  python kernel_diagnostics.py --kernel-dir output/v8_qsvm_tight

Expects files like:
  <kernel-dir>/pca_quantum_kernel.npy       (500x500 matrix)
  <kernel-dir>/umap_quantum_kernel.npy      etc.

Adjust filename patterns below if your naming differs.
"""

import argparse
from pathlib import Path
import numpy as np

METHODS = ["pca", "rp_gaussian", "pls", "umap"]

# Common filename patterns for kernel matrices
KERNEL_PATTERNS = [
    "{method}_quantum_kernel.npy",
    "{method}_kernel_matrix.npy",
    "{method}_zz_kernel.npy",
    "kernel_{method}.npy",
]


def find_kernel(kernel_dir: Path, method: str):
    """Try multiple naming patterns to find the kernel .npy file."""
    for pattern in KERNEL_PATTERNS:
        fname = kernel_dir / pattern.format(method=method)
        if fname.exists():
            return np.load(fname)

    # Fallback: search for any .npy containing the method name
    for f in kernel_dir.glob(f"*{method}*.npy"):
        K = np.load(f)
        if K.ndim == 2 and K.shape[0] == K.shape[1]:
            print(f"  Found kernel: {f.name} ({K.shape})")
            return K

    return None


def kernel_diagnostics(K, label):
    """Compute and print kernel matrix diagnostics."""
    n = K.shape[0]

    # Eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending
    pos_eigs = eigenvalues[eigenvalues > 1e-12]

    # Effective rank (Vershynin / Roy & Bhattacharyya)
    p = pos_eigs / pos_eigs.sum()
    entropy = -np.sum(p * np.log(p))
    effective_rank = np.exp(entropy)

    # Diagonal vs off-diagonal statistics
    diag = np.diag(K)
    mask = ~np.eye(n, dtype=bool)
    offdiag = K[mask]

    # Top eigenvalue concentration
    top1_pct = eigenvalues[0] / pos_eigs.sum() * 100
    top5_pct = eigenvalues[:5].sum() / pos_eigs.sum() * 100
    top10_pct = eigenvalues[:10].sum() / pos_eigs.sum() * 100

    # Frobenius distance from identity
    frob_from_I = np.linalg.norm(K - np.eye(n), "fro")
    frob_K = np.linalg.norm(K, "fro")

    print(f"\n{'='*60}")
    print(f"  {label.upper()} — Kernel Matrix Diagnostics ({n}x{n})")
    print(f"{'='*60}")
    print(f"  Spectral entropy:       {entropy:.3f}")
    print(f"  Effective rank:         {effective_rank:.1f} / {n}")
    print(f"  Effective rank ratio:   {effective_rank/n:.3f}")
    print(f"  ---")
    print(f"  Diagonal mean:          {diag.mean():.6f}")
    print(f"  Diagonal std:           {diag.std():.6f}")
    print(f"  Off-diagonal mean:      {offdiag.mean():.6f}")
    print(f"  Off-diagonal std:       {offdiag.std():.6f}")
    print(f"  Off-diagonal max:       {offdiag.max():.6f}")
    print(f"  Diag/off-diag ratio:    {diag.mean()/max(offdiag.mean(), 1e-12):.2f}")
    print(f"  ---")
    print(f"  Top-1 eigenvalue:       {top1_pct:.1f}% of trace")
    print(f"  Top-5 eigenvalues:      {top5_pct:.1f}% of trace")
    print(f"  Top-10 eigenvalues:     {top10_pct:.1f}% of trace")
    print(f"  ||K - I||_F / ||K||_F:  {frob_from_I/frob_K:.4f}")
    print(f"  Num positive eigs:      {len(pos_eigs)} / {n}")

    return {
        "method": label,
        "n": n,
        "entropy": entropy,
        "effective_rank": effective_rank,
        "eff_rank_ratio": effective_rank / n,
        "diag_mean": diag.mean(),
        "offdiag_mean": offdiag.mean(),
        "diag_offdiag_ratio": diag.mean() / max(offdiag.mean(), 1e-12),
        "top1_pct": top1_pct,
        "top10_pct": top10_pct,
        "frob_from_I_ratio": frob_from_I / frob_K,
    }


def print_summary_table(results):
    """Print a compact comparison table."""
    print(f"\n{'='*60}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*60}")
    header = f"{'Method':<14} {'Eff.Rank':>8} {'Ratio':>6} {'Entropy':>8} {'Diag/Off':>9} {'||K-I||':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['method']:<14} {r['effective_rank']:>8.1f} {r['eff_rank_ratio']:>6.3f} "
              f"{r['entropy']:>8.3f} {r['diag_offdiag_ratio']:>9.2f} "
              f"{r['frob_from_I_ratio']:>8.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-dir", type=str, default="output/v8_qsvm_tight")
    args = parser.parse_args()

    kernel_dir = Path(args.kernel_dir)
    if not kernel_dir.exists():
        print(f"ERROR: {kernel_dir} not found")
        return

    print(f"Searching for kernel matrices in: {kernel_dir}")
    npy_files = sorted(kernel_dir.glob("*.npy"))
    print(f"Found .npy files: {[f.name for f in npy_files]}")

    results = []
    for method in METHODS:
        print(f"\nLoading {method}...")
        K = find_kernel(kernel_dir, method)
        if K is not None:
            if K.shape[0] != K.shape[1]:
                print(f"  WARNING: not square ({K.shape}), skipping")
                continue
            r = kernel_diagnostics(K, method)
            results.append(r)
        else:
            print(f"  No kernel matrix found for {method}")

    if results:
        print_summary_table(results)


if __name__ == "__main__":
    main()