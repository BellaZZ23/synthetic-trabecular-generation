from scipy.stats import wilcoxon

# classical_folds = np.array of 25 UMAP classical accuracies
# quantum_folds = np.array of 25 UMAP quantum accuracies

stat, p = wilcoxon(quantum_folds, classical_folds, alternative='greater')
print(f"Wilcoxon signed-rank: stat={stat}, p={p:.4f}")
# p < 0.05 → significant; p > 0.05 → "statistically indistinguishable"