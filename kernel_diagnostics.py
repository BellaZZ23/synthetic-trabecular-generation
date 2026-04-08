import numpy as np

def kernel_diagnostics(K, label):
    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]  # numerical zeros
    
    # Effective rank (Roy & Bhattacharyya 2007)
    p = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(p * np.log(p))
    effective_rank = np.exp(entropy)
    
    # Diagonal dominance
    diag_mean = np.diag(K).mean()
    offdiag_mean = (K.sum() - np.trace(K)) / (K.shape[0]**2 - K.shape[0])
    
    print(f"{label}:")
    print(f"  Effective rank: {effective_rank:.1f} / {K.shape[0]}")
    print(f"  Spectral entropy: {entropy:.3f}")
    print(f"  Diag mean: {diag_mean:.4f}, Off-diag mean: {offdiag_mean:.4f}")
    print(f"  Diag/off-diag ratio: {diag_mean/offdiag_mean:.2f}")

# Load each kernel matrix
for method in ['pca', 'rp_gaussian', 'pls', 'umap']:
    K = np.load(f"output/v8_qsvm_tight/{method}_quantum_kernel.npy")
    kernel_diagnostics(K, method)