"""
Kernel matrix heatmaps: UMAP vs PCA (and optionally RP Gaussian).
Uses your existing saved kernel matrices if available, otherwise recomputes
on N=200 samples. Outputs: kernel_heatmaps.png (transparent-background version
also saved for PowerPoint overlay).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import StatevectorSampler

SEED     = 42
N_Q      = 8
N_PLOT   = 100   # subset to display (full matrix is still computed on N_COMP)
N_COMP   = 200   # samples for kernel computation

NAVY  = "#080C3C"
CYAN  = "#3ECFB0"
MUTED = "#7A9EC8"

# ── LOAD YOUR DATA ────────────────────────────────────────────────────────────
import pandas as pd

df = pd.read_csv(
    r"C:\Users\Isabella\OneDrive - zeki\Documents\Research\synthetic_trabeculae"
    r"\output\v8_dataset\features\features_v8.csv"
)

print("Shape:", df.shape)

# Binarise BV/TV at 0.5 (high / low)
threshold = 0.5
X = df.drop(columns=["sample", "bvtv"]).values
y = (df["bvtv"].values >= threshold).astype(int)

print(f"Class balance: {y.sum()} high / {(1-y).sum()} low  (threshold={threshold})")

# ── SUBSAMPLE & REDUCE ────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
idx = rng.choice(len(X), size=N_COMP, replace=False)
# Sort by class so block structure is visible
sort_order = np.argsort(y[idx])
idx = idx[sort_order]
X_sub, y_sub = X[idx], y[idx]

scaler = StandardScaler()
X_sc = scaler.fit_transform(X_sub)

# UMAP → 8 dims
reducer_umap = umap.UMAP(n_components=N_Q, random_state=SEED)
X_umap = reducer_umap.fit_transform(X_sc)
X_umap_norm = (X_umap - X_umap.min(0)) / (X_umap.max(0) - X_umap.min(0) + 1e-8) * 2 * np.pi

# PCA → 8 dims
pca = PCA(n_components=N_Q, random_state=SEED)
X_pca = pca.fit_transform(X_sc)
X_pca_norm = (X_pca - X_pca.min(0)) / (X_pca.max(0) - X_pca.min(0) + 1e-8) * 2 * np.pi

# ── COMPUTE KERNELS ───────────────────────────────────────────────────────────
feature_map = ZZFeatureMap(feature_dimension=N_Q, reps=2)
kernel = FidelityQuantumKernel(feature_map=feature_map)

print("Computing UMAP kernel matrix...")
K_umap = kernel.evaluate(x_vec=X_umap_norm[:N_PLOT])

print("Computing PCA kernel matrix...")
K_pca  = kernel.evaluate(x_vec=X_pca_norm[:N_PLOT])

# ── STATS ─────────────────────────────────────────────────────────────────────
def eff_rank(K):
    ev = np.linalg.eigvalsh(K); ev = np.clip(ev, 0, None)
    p = ev / ev.sum(); p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))

def off_diag_max(K):
    M = K.copy(); np.fill_diagonal(M, 0); return M.max()

stats = {
    "UMAP": {"K": K_umap, "eff_rank": eff_rank(K_umap), "off_diag_max": off_diag_max(K_umap)},
    "PCA":  {"K": K_pca,  "eff_rank": eff_rank(K_pca),  "off_diag_max": off_diag_max(K_pca)},
}

for name, s in stats.items():
    print(f"{name}: eff_rank={s['eff_rank']:.1f}/{N_PLOT}  max_off_diag={s['off_diag_max']:.3f}")

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="none")

cmap_umap = plt.cm.Blues
cmap_pca  = plt.cm.Greys

for ax, (name, s), cmap in zip(axes, stats.items(), [cmap_umap, cmap_pca]):
    K = s["K"]
    im = ax.imshow(K, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_title(
        f"{name}  (8q)",
        fontsize=13, fontweight="bold", color=NAVY, pad=8,
        fontfamily="sans-serif"
    )
    ax.set_xlabel(
        f"eff. rank = {s['eff_rank']:.0f} / {N_PLOT}   "
        f"max off-diag = {s['off_diag_max']:.3f}",
        fontsize=9, color=MUTED
    )
    ax.set_xticks([]); ax.set_yticks([])

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Kernel value", fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7, colors=MUTED)

    for spine in ax.spines.values():
        spine.set_edgecolor("#C2D5EE")

fig.suptitle(
    "ZZ Quantum Kernel Matrices — UMAP vs PCA",
    fontsize=14, fontweight="bold", color=NAVY, y=1.02
)
plt.tight_layout()

# Save with white background
fig.patch.set_facecolor("white")
plt.savefig("kernel_heatmaps.png", dpi=300, bbox_inches="tight",
            facecolor="white")

# Save transparent for PowerPoint overlay
fig.patch.set_facecolor("none")
for ax in axes:
    ax.set_facecolor("none")
plt.savefig("kernel_heatmaps_transparent.png", dpi=300, bbox_inches="tight",
            transparent=True)

print("Saved kernel_heatmaps.png and kernel_heatmaps_transparent.png")
plt.show()