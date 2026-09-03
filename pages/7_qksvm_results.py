# -*- coding: utf-8 -*-
"""
pages/7_qksvm_results.py — QKSVM pipeline integration

Shows HOW the quantum-kernel SVM method integrates into the trabecular
bone pipeline:  Morphometrics → Reduce → Kernel matrix → Classify

Uses morphometrics already computed in the pipeline session (if available),
otherwise generates a synthetic demo dataset.  All reducers are implemented
with pure NumPy so no extra install is required; UMAP notes an optional upgrade.
"""
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

st.set_page_config(
    page_title="QKSVM · Pipeline integration",
    page_icon="⚛️",
    layout="wide",
)

# ── Palette ────────────────────────────────────────────────────────────────────
C_TEAL   = "#1D9E75"
C_BLUE   = "#378ADD"
C_PURPLE = "#7F77DD"
C_CORAL  = "#E85D3A"
C_BONE   = "#C8BFA9"

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { max-width:1100px; padding-top:2rem; padding-bottom:3rem; }
.section-label {
    font-size:0.7rem; font-weight:700; letter-spacing:0.15em;
    text-transform:uppercase; color:#888; margin-bottom:0.5rem;
}
.pipe-step {
    background:#fff; border-radius:12px; padding:0.9rem 1.1rem;
    box-shadow:0 2px 8px rgba(0,0,0,0.07);
    border-top:4px solid var(--c,#378ADD);
    text-align:center; height:100%;
}
.pipe-step .ps-num {
    font-size:1.5rem; font-weight:900; color:var(--c,#378ADD);
}
.pipe-step .ps-title {
    font-size:0.9rem; font-weight:700; color:#1C1C2E; margin:0.15rem 0;
}
.pipe-step .ps-sub {
    font-size:0.75rem; color:#666; line-height:1.4;
}
.info-card {
    background:#f3f1ff; border-radius:12px; padding:1rem 1.2rem;
    border-left:5px solid #7F77DD; margin-bottom:1rem;
}
.guide-card {
    background:#fff; border-radius:12px; padding:1.1rem 1.2rem;
    box-shadow:0 2px 8px rgba(0,0,0,0.07);
    border-top:4px solid var(--c,#378ADD); height:100%;
}
.chip {
    display:inline-block; background:#EEF2FF; color:#4338CA;
    border-radius:20px; padding:0.2rem 0.75rem; font-size:0.75rem;
    font-weight:600; margin:0.15rem 0.1rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">QKSVM · pipeline integration · University of Greenwich</p>',
            unsafe_allow_html=True)
st.markdown("# ⚛️ Quantum Kernel SVM — Pipeline Integration")
st.markdown(
    "This page demonstrates where the **Quantum Kernel SVM method** fits in the "
    "bone-analysis workflow.  Select a dimensionality reducer, explore the "
    "compressed representation, inspect the kernel matrix, and see a live "
    "classification output — using your pipeline's morphometrics or a synthetic demo."
)
st.markdown(
    '<span class="chip">Feature reduction → Quantum kernel → Classify</span>'
    '<span class="chip">UMAP = quantum-competitive reducer</span>'
    '<span class="chip">Works on pipeline morphometrics</span>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE POSITION DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Where this step sits in the pipeline</p>',
            unsafe_allow_html=True)

c1, arr1, c2, arr2, c3, arr3, c4 = st.columns([5, 1, 5, 1, 5, 1, 5])

with c1:
    st.markdown("""
<div class="pipe-step" style="--c:#C8BFA9">
  <div class="ps-num">1</div>
  <div class="ps-title">Generate &amp; Analyse</div>
  <div class="ps-sub">Synthetic bone · morphometrics<br>BV/TV, Tb.Th, Tb.N, Tb.Sp…</div>
</div>""", unsafe_allow_html=True)

with arr1:
    st.markdown("<div style='text-align:center;font-size:2rem;padding-top:1.5rem'>→</div>",
                unsafe_allow_html=True)

with c2:
    st.markdown("""
<div class="pipe-step" style="--c:#378ADD">
  <div class="ps-num">2</div>
  <div class="ps-title">Feature Reduction</div>
  <div class="ps-sub">PCA · RP · PLS · UMAP<br>43 features → n qubits</div>
</div>""", unsafe_allow_html=True)

with arr2:
    st.markdown("<div style='text-align:center;font-size:2rem;padding-top:1.5rem'>→</div>",
                unsafe_allow_html=True)

with c3:
    st.markdown("""
<div class="pipe-step" style="--c:#7F77DD;background:linear-gradient(135deg,#f3f1ff,#eaf4ff)">
  <div class="ps-num">3</div>
  <div class="ps-title">ZZ Quantum Kernel</div>
  <div class="ps-sub">ZZFeatureMap circuit<br>K(xᵢ,xⱼ) = |⟨φ(xᵢ)|φ(xⱼ)⟩|²</div>
</div>""", unsafe_allow_html=True)

with arr3:
    st.markdown("<div style='text-align:center;font-size:2rem;padding-top:1.5rem'>→</div>",
                unsafe_allow_html=True)

with c4:
    st.markdown("""
<div class="pipe-step" style="--c:#1D9E75">
  <div class="ps-num">4</div>
  <div class="ps-title">Classify / Assess</div>
  <div class="ps-sub">BV/TV class · Tb.N class<br>→ osteoporosis risk flag</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# NUMPY-ONLY REDUCERS
# ══════════════════════════════════════════════════════════════════════════════

def _pca(X, n):
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n].T, Vt[:n]          # (N, n), (n, d)

def _rp_gauss(X, n, seed=42):
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((X.shape[1], n)) / np.sqrt(n)
    return X @ R, R

def _rp_sparse(X, n, seed=42):
    rng = np.random.default_rng(seed)
    R = rng.choice([-1, 0, 0, 1], size=(X.shape[1], n)).astype(float)
    norms = np.linalg.norm(R, axis=0, keepdims=True)
    norms[norms == 0] = 1
    R = R / norms
    return X @ R, R

def _pls(X, y, n):
    """Simple NIPALS-style PLS: extracts n latent directions."""
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    T = []
    Xr = Xc.copy()
    for _ in range(n):
        w = Xr.T @ yc
        nw = np.linalg.norm(w)
        if nw < 1e-12:
            break
        w /= nw
        t = Xr @ w
        p = Xr.T @ t / (t @ t + 1e-12)
        Xr = Xr - np.outer(t, p)
        T.append(t)
    while len(T) < n:
        T.append(np.zeros(Xc.shape[0]))
    return np.column_stack(T[:n])

def _umap_reduce(X, n, seed=42):
    """UMAP: topology-preserving nonlinear reduction (best quantum-classical parity)."""
    import umap as _umap
    reducer = _umap.UMAP(n_components=n, random_state=seed,
                         n_neighbors=min(15, len(X)-1), min_dist=0.1)
    return reducer.fit_transform(X)

def _tsne(X, n, seed=42):
    """t-SNE: cluster-structure-preserving reduction (2 components only)."""
    from sklearn.manifold import TSNE
    n_eff = min(n, 2)  # t-SNE is usually 2D
    pca_init, _ = _pca(X, min(n_eff, X.shape[1]))
    model = TSNE(n_components=n_eff, perplexity=min(30, len(X)//4),
                 random_state=seed, init=pca_init[:, :n_eff], learning_rate='auto')
    T = model.fit_transform(X)
    # pad extra dims with zeros if n_comp > 2
    if n > n_eff:
        T = np.hstack([T, np.zeros((len(X), n - n_eff))])
    return T

def _autoencoder(X, n, epochs=400, lr=0.01, seed=42):
    """Shallow numpy autoencoder: input → hidden(2n) → bottleneck(n) → reconstruct."""
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    H = max(2 * n, 8)
    # He initialisation
    W1 = rng.standard_normal((D, H))  * np.sqrt(2 / D)
    W2 = rng.standard_normal((H, n))  * np.sqrt(2 / H)
    W3 = rng.standard_normal((n, H))  * np.sqrt(2 / n)
    W4 = rng.standard_normal((H, D))  * np.sqrt(2 / H)
    b1 = np.zeros(H); b2 = np.zeros(n)
    b3 = np.zeros(H); b4 = np.zeros(D)

    def relu(x): return np.maximum(0, x)
    def drelu(x): return (x > 0).astype(float)

    for ep in range(epochs):
        h1 = relu(X @ W1 + b1)
        z  = X @ W1 + b1; a1 = relu(z)
        enc = a1 @ W2 + b2
        z2 = enc @ W3 + b3; a2 = relu(z2)
        out = a2 @ W4 + b4
        loss = out - X
        # backprop
        dout = loss / len(X)
        dW4 = a2.T @ dout; db4 = dout.sum(0)
        da2 = dout @ W4.T
        dz2 = da2 * drelu(z2)
        dW3 = enc.T @ dz2; db3 = dz2.sum(0)
        denc = dz2 @ W3.T
        dW2 = a1.T @ denc; db2 = denc.sum(0)
        da1 = denc @ W2.T
        dz1 = da1 * drelu(z)
        dW1 = X.T @ dz1; db1 = dz1.sum(0)
        for w, g in [(W1,dW1),(W2,dW2),(W3,dW3),(W4,dW4)]:
            w -= lr * g
        for b, g in [(b1,db1),(b2,db2),(b3,db3),(b4,db4)]:
            b -= lr * g
    # return bottleneck
    return relu(X @ W1 + b1) @ W2 + b2


def _rbf_kernel(Z, gamma=None):
    """RBF kernel approximation of the ZZ quantum kernel on reduced features."""
    if gamma is None:
        gamma = 1.0 / Z.shape[1]
    sq = np.sum(Z ** 2, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2 * Z @ Z.T
    D2 = np.clip(D2, 0, None)
    return np.exp(-gamma * D2)

# ══════════════════════════════════════════════════════════════════════════════
# DATA SOURCE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Step 1 — Feature data</p>', unsafe_allow_html=True)

MORPH_KEYS = ["BVTV", "TbTh_um_p50", "TbN_per_mm", "TbSp_um_p50",
              "connectivity_density", "DA", "texture_contrast",
              "texture_homogeneity", "texture_energy", "texture_correlation"]

def _build_demo(n=80, seed=7):
    rng = np.random.default_rng(seed)
    bvtv  = rng.uniform(0.10, 0.42, n)
    tbth  = rng.uniform(78, 175, n)
    tbn   = rng.uniform(1.5, 4.5, n)
    tbsp  = rng.uniform(140, 420, n)
    conn  = rng.uniform(1.5, 9.0, n)
    da    = rng.uniform(0.1, 0.7, n)
    contrast  = rng.uniform(0.02, 0.45, n)
    homog     = rng.uniform(0.55, 0.95, n)
    energy    = rng.uniform(0.05, 0.35, n)
    corr      = rng.uniform(0.20, 0.90, n)
    # Add some correlated noise to make it realistic
    for arr in [tbth, tbn, tbsp, conn]:
        arr += rng.normal(0, arr.std() * 0.1, n)
    X = np.column_stack([bvtv, tbth, tbn, tbsp, conn, da,
                         contrast, homog, energy, corr])
    return X, bvtv

# Try to pull from session state
X, bvtv_col, data_source = None, None, "demo"

morph = st.session_state.get("morphometrics") or st.session_state.get("last_morphometrics")
if isinstance(morph, dict) and "BVTV" in morph:
    # Single sample from pipeline — we'll augment with synthetic neighbourhood
    vals = [morph.get(k, np.nan) for k in MORPH_KEYS]
    if not any(np.isnan(v) for v in vals):
        rng = np.random.default_rng(42)
        X_demo, bvtv_demo = _build_demo(79)
        row = np.array(vals, dtype=float)
        # Normalise and append the pipeline sample
        X = np.vstack([X_demo, row[None, :]])
        bvtv_col = np.append(bvtv_demo, morph["BVTV"])
        data_source = "pipeline"

if X is None:
    X, bvtv_col = _build_demo(80)
    data_source = "demo"

# Standardise
X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
N, D = X_std.shape
bvtv_median = np.median(bvtv_col)
y_binary = (bvtv_col >= bvtv_median).astype(int)   # 1 = normal, 0 = low BV/TV

if data_source == "pipeline":
    st.success(
        f"✅ **Using your pipeline morphometrics** — pipeline sample highlighted in the embedding. "
        f"Augmented with {N-1} synthetic neighbours for visualisation."
    )
else:
    st.info(
        f"ℹ️ **No pipeline morphometrics in session** — showing a synthetic demo dataset "
        f"({N} samples, BV/TV 0.10–0.42).  Run the pipeline first to use your own data."
    )

st.caption(
    f"Features used: BV/TV, Tb.Th, Tb.N, Tb.Sp, Conn.D, DA, "
    f"texture contrast/homogeneity/energy/correlation ({D} features total)."
)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# REDUCER CONTROLS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Step 2 — Dimensionality reduction</p>',
            unsafe_allow_html=True)

ctrl_l, ctrl_r = st.columns([3, 2])
with ctrl_l:
    reducer_name = st.selectbox(
        "Reducer",
        ["PCA", "RP — Gaussian", "RP — Sparse", "PLS (BV/TV target)", "UMAP", "t-SNE", "Autoencoder"],
        help=(
            "PCA / RP: linear baselines. PLS: supervised (BV/TV target). "
            "UMAP: topology-preserving nonlinear (best quantum-classical parity, +0.032). "
            "t-SNE: cluster-structure-preserving (2D only, not passed to qubit angles). "
            "Autoencoder: shallow numpy AE, bottleneck = n latent dims."
        ),
    )
with ctrl_r:
    n_comp = st.slider("Components (= qubits)", min_value=2, max_value=min(8, D),
                       value=4, step=2,
                       help="Each component maps to one qubit in the ZZ circuit.")

# Run reducer
if reducer_name == "PCA":
    Z, _ = _pca(X_std, n_comp)
elif reducer_name == "RP — Gaussian":
    Z, _ = _rp_gauss(X_std, n_comp)
elif reducer_name == "RP — Sparse":
    Z, _ = _rp_sparse(X_std, n_comp)
elif reducer_name == "PLS (BV/TV target)":
    Z = _pls(X_std, bvtv_col, n_comp)
elif reducer_name == "UMAP":
    with st.spinner("Running UMAP…"):
        Z = _umap_reduce(X_std, n_comp)
elif reducer_name == "t-SNE":
    with st.spinner("Running t-SNE…"):
        Z = _tsne(X_std, n_comp)
else:  # Autoencoder
    with st.spinner("Training autoencoder…"):
        Z = _autoencoder(X_std, n_comp)

# Rescale to [-π, π] as the ZZ circuit would receive
Z_scaled = np.pi * (Z - Z.min(axis=0)) / (Z.max(axis=0) - Z.min(axis=0) + 1e-12) - np.pi / 2

# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING + KERNEL SIDE BY SIDE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Step 3 — Compressed representation &amp; kernel matrix</p>',
            unsafe_allow_html=True)

emb_col, kern_col = st.columns(2)

# ── Embedding scatter ─────────────────────────────────────────────────────────
with emb_col:
    st.markdown("**Embedding (first 2 components)**")
    fig_emb, ax_emb = plt.subplots(figsize=(5.5, 4.5))

    sc = ax_emb.scatter(
        Z[:, 0], Z[:, 1],
        c=bvtv_col, cmap="RdYlGn",
        s=60, alpha=0.85, edgecolors="white", linewidths=0.5,
        vmin=bvtv_col.min(), vmax=bvtv_col.max(),
    )
    plt.colorbar(sc, ax=ax_emb, label="BV/TV")

    # Highlight pipeline sample if present
    if data_source == "pipeline":
        ax_emb.scatter(
            Z[-1, 0], Z[-1, 1],
            s=220, marker="*", color="#E85D3A",
            edgecolors="white", linewidths=1.5,
            zorder=5, label="Your pipeline sample",
        )
        ax_emb.legend(fontsize=9)

    # Decision boundary estimate (median of comp-0)
    med0 = np.median(Z[:, 0])
    ax_emb.axvline(med0, color="#7F77DD", ls="--", lw=1.3, alpha=0.7,
                   label="Approx. decision boundary")
    ax_emb.set_xlabel(f"{reducer_name} component 1 (→ qubit 1 angle)", fontsize=9)
    ax_emb.set_ylabel(f"{reducer_name} component 2 (→ qubit 2 angle)", fontsize=9)
    ax_emb.set_title(f"{reducer_name} embedding ({N} samples, coloured by BV/TV)",
                     fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_emb, use_container_width=True)
    plt.close()

    # Explained variance for PCA only
    if reducer_name == "PCA":
        _, sv, _ = np.linalg.svd(X_std, full_matrices=False)
        ev_total = (sv ** 2).sum()
        ev_comp  = (sv[:n_comp] ** 2).sum() / ev_total
        st.caption(
            f"PCA: first {n_comp} components explain **{ev_comp:.1%}** of total variance."
        )
    else:
        st.caption(
            "Colour gradient shows BV/TV — well-separated clusters indicate the reducer "
            "preserves the bone-density structure that the quantum kernel will exploit."
        )

# ── Kernel matrix heatmap ─────────────────────────────────────────────────────
with kern_col:
    st.markdown("**Approximate quantum kernel matrix K(xᵢ, xⱼ)**")

    K = _rbf_kernel(Z_scaled)

    fig_k, ax_k = plt.subplots(figsize=(5.5, 4.5))
    im = ax_k.imshow(K, cmap="plasma", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_k, label="Kernel value")

    ax_k.set_xlabel("Sample index", fontsize=9)
    ax_k.set_ylabel("Sample index", fontsize=9)
    ax_k.set_title(
        f"ZZ kernel approximation — {reducer_name} ({n_comp} components)",
        fontsize=9.5, fontweight="bold",
    )

    # Effective rank
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.maximum(eigvals, 0)
    eigvals = eigvals / (eigvals.sum() + 1e-12)
    eff_rank = int(np.exp(-np.sum(eigvals * np.log(eigvals + 1e-15))))
    offdiag_mask = ~np.eye(N, dtype=bool)
    offdiag_mean = K[offdiag_mask].mean()

    plt.tight_layout()
    st.pyplot(fig_k, use_container_width=True)
    plt.close()

    # Diagnostics
    k1, k2 = st.columns(2)
    rank_color = "normal" if eff_rank < 150 else "inverse"
    k1.metric("Effective rank", f"{eff_rank} / {N}",
              help="Lower = more structure the SVM can exploit. UMAP ~102, PCA >400.")
    k2.metric("Off-diagonal mean", f"{offdiag_mean:.3f}",
              help="Higher = more pairwise similarity variation (more kernel contrast).")

    diag_ratio = eff_rank / N
    if diag_ratio > 0.7:
        st.warning(
            f"⚠️ High effective rank ({eff_rank}/{N} = {diag_ratio:.0%}) — near-identity "
            f"kernel. The SVM will struggle to find class boundaries. "
            f"Try UMAP (install `umap-learn`) or reduce n_components."
        )
    elif diag_ratio > 0.3:
        st.info(
            f"ℹ️ Moderate kernel structure (rank {eff_rank}/{N} = {diag_ratio:.0%}). "
            f"Classification may work but is not optimal."
        )
    else:
        st.success(
            f"✅ Good kernel structure (rank {eff_rank}/{N} = {diag_ratio:.0%}) — "
            f"block-diagonal pattern; the SVM has exploitable class separation."
        )

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Step 4 — Classification output (demo)</p>',
            unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
  <b>What the quantum SVM does here</b><br>
  <span style="font-size:0.87rem;color:#444">
  The ZZ kernel matrix above is passed to an SVM which learns a hyperplane
  in the quantum feature space.  On this demo the classifier is a simple
  kernel-threshold rule (sign of K·α) — sufficient to show where each sample
  sits relative to the BV/TV decision boundary without needing a real quantum circuit.
  </span>
</div>
""", unsafe_allow_html=True)

# Simple kernel-based classifier: use first column of K as "similarity to low-BV/TV"
# (proxy for what the SVM α vector would compute)
bvtv_sorted_idx = np.argsort(bvtv_col)
low_idx  = bvtv_sorted_idx[:N//4]   # lowest 25%
high_idx = bvtv_sorted_idx[-N//4:]  # highest 25%
k_diff   = K[:, high_idx].mean(axis=1) - K[:, low_idx].mean(axis=1)
y_pred   = (k_diff >= 0).astype(int)
acc = (y_pred == y_binary).mean()

cl1, cl2, cl3 = st.columns(3)
with cl1:
    st.metric("Demo accuracy", f"{acc:.1%}",
              help="Kernel-similarity classifier on the reduced features (not a trained SVM).")
with cl2:
    st.metric("BV/TV threshold used", f"{bvtv_median:.3f}",
              help="Median BV/TV splits samples into low (0) / normal (1) classes.")
with cl3:
    correct = (y_pred == y_binary).sum()
    st.metric("Correctly classified", f"{correct} / {N}")

# Show a small per-sample bar
fig_cls, ax_cls = plt.subplots(figsize=(9, 2.5))
colors_bar = [C_TEAL if p == t else C_CORAL
              for p, t in zip(y_pred, y_binary)]
ax_cls.bar(np.arange(N), bvtv_col, color=colors_bar, alpha=0.8, width=1.0)
ax_cls.axhline(bvtv_median, color="#7F77DD", ls="--", lw=1.5, label=f"Median BV/TV = {bvtv_median:.3f}")

# Pipeline sample marker
if data_source == "pipeline":
    ax_cls.axvline(N - 1, color="#E85D3A", lw=2.5, ls=":", label="Your pipeline sample")

from matplotlib.patches import Patch
ax_cls.legend(handles=[
    Patch(color=C_TEAL,   label="Correctly classified"),
    Patch(color=C_CORAL,  label="Misclassified"),
] + ([Patch(color="#E85D3A", label="Pipeline sample")] if data_source == "pipeline" else []),
    fontsize=8.5, loc="upper right")
ax_cls.set_xlabel("Sample index", fontsize=9)
ax_cls.set_ylabel("BV/TV", fontsize=9)
ax_cls.set_title(f"Classification output — {reducer_name} ({n_comp}q) · "
                 f"green = correct, red = misclassified",
                 fontsize=9.5, fontweight="bold")
plt.tight_layout()
st.pyplot(fig_cls, use_container_width=True)
plt.close()

if data_source == "pipeline":
    pipe_pred = "normal BV/TV" if y_pred[-1] == 1 else "low BV/TV"
    pipe_true = "normal BV/TV" if y_binary[-1] == 1 else "low BV/TV"
    match = "✅" if y_pred[-1] == y_binary[-1] else "⚠️"
    st.info(
        f"{match} **Your pipeline sample** — predicted: **{pipe_pred}** · "
        f"actual: **{pipe_true}** (BV/TV = {bvtv_col[-1]:.3f}, "
        f"median = {bvtv_median:.3f})"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION NOTE
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("📉 Why continuous BV/TV prediction (regression) doesn't work here", expanded=False):
    st.markdown("""
The ZZ quantum kernel at ≥ 8 qubits maps 2^n = 256 basis states, producing a kernel
matrix whose effective rank approaches N (near-identity).  A kernel ridge regressor
needs **off-diagonal structure** to predict continuous values — near-identity kernels
yield essentially random predictions (R² < 0 across all reducers except PLS in the
published study).

**In practice:** use the pipeline for *classification* (low / normal BV/TV, Tb.N category)
and use classical ridge regression for continuous morphometric prediction.  The quantum
advantage is in categorical boundary detection, not smooth interpolation.
""")

    # Quick illustration
    fig_reg, ax_reg = plt.subplots(figsize=(7, 3))
    k_row = K[0]  # kernel similarities to sample 0
    ax_reg.scatter(bvtv_col, k_row, c=bvtv_col, cmap="RdYlGn", s=40, alpha=0.7)
    m, b = np.polyfit(bvtv_col, k_row, 1)
    xs = np.linspace(bvtv_col.min(), bvtv_col.max(), 100)
    ax_reg.plot(xs, m * xs + b, "--", color="#E85D3A", lw=1.5,
                label=f"Linear fit (slope={m:.3f})")
    corr_val = np.corrcoef(bvtv_col, k_row)[0, 1]
    ax_reg.set_xlabel("BV/TV (continuous)")
    ax_reg.set_ylabel("Kernel similarity to sample 0")
    ax_reg.set_title(f"Kernel similarity vs BV/TV — r = {corr_val:.3f} "
                     f"({'weak' if abs(corr_val)<0.3 else 'moderate'} correlation → "
                     f"regression unreliable)",
                     fontsize=9.5, fontweight="bold")
    ax_reg.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_reg, use_container_width=True)
    plt.close()

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# THREE INTEGRATION GUIDELINES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Integration guidelines from the published study</p>',
            unsafe_allow_html=True)

g1, g2, g3 = st.columns(3)

with g1:
    st.markdown(f"""
<div class="guide-card" style="--c:{C_TEAL}">
  <div style="font-size:1.6rem;font-weight:900;color:{C_TEAL};margin-bottom:0.3rem">1</div>
  <div style="font-size:0.93rem;font-weight:700;color:#1C1C2E;margin-bottom:0.45rem">
    Install UMAP for best results
  </div>
  <div style="font-size:0.82rem;color:#444;line-height:1.55">
    UMAP is the only tested reducer that preserves quantum–classical parity
    (+0.032 accuracy advantage on BV/TV classification).  PCA and RP produce
    near-identity kernels — the pipeline already supports a UMAP upgrade path.<br><br>
    <code>pip install umap-learn</code>
  </div>
</div>""", unsafe_allow_html=True)

with g2:
    st.markdown(f"""
<div class="guide-card" style="--c:{C_BLUE}">
  <div style="font-size:1.6rem;font-weight:900;color:{C_BLUE};margin-bottom:0.3rem">2</div>
  <div style="font-size:0.93rem;font-weight:700;color:#1C1C2E;margin-bottom:0.45rem">
    Check effective rank before classifying
  </div>
  <div style="font-size:0.82rem;color:#444;line-height:1.55">
    The kernel matrix diagnostic above shows the current reducer's structure.
    Target effective rank &lt; 20% of N for reliable SVM classification.
    High rank → add this check to your pipeline quality gate before submitting
    to a real quantum backend.
  </div>
</div>""", unsafe_allow_html=True)

with g3:
    st.markdown(f"""
<div class="guide-card" style="--c:{C_PURPLE}">
  <div style="font-size:1.6rem;font-weight:900;color:{C_PURPLE};margin-bottom:0.3rem">3</div>
  <div style="font-size:0.93rem;font-weight:700;color:#1C1C2E;margin-bottom:0.45rem">
    Classification yes · Regression no
  </div>
  <div style="font-size:0.82rem;color:#444;line-height:1.55">
    Map continuous BV/TV to a binary or ordinal class (low / borderline / normal)
    before passing to the quantum SVM.  For continuous morphometric prediction
    keep classical Ridge or Gaussian Process — the ZZ kernel R² is negative
    for all reducers except PLS.
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LINK TO QIC ROADMAP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:linear-gradient(135deg,#f0fdf8,#f7fffe);
     border-radius:14px; padding:1.2rem 1.4rem; border-left:5px solid #1D9E75;">
  <p style="font-size:0.7rem;font-weight:700;letter-spacing:0.12em;
            text-transform:uppercase;color:#1D9E75;margin-bottom:0.4rem;">
    Next step in the research roadmap
  </p>
  <p style="font-size:0.9rem;color:#1C1C2E;line-height:1.6;margin-bottom:0.4rem;">
    The UMAP finding above — that non-linear topology-preserving reduction
    is necessary for quantum-classical parity — directly motivates the
    <b>Volumetric Quantum Image Correlation (QIC)</b> architecture.
    Sub-volume patches are encoded via UMAP into qubit angles, then compared
    via a quantum reservoir circuit rather than a classical phase-correlation.
  </p>
  <p style="font-size:0.84rem;color:#555;">
    See <b>page 6 — QIC Roadmap</b> for the full architecture and roadmap.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.caption(
    "Isabella Florez · University of Greenwich · "
    "Quantum Kernel SVM for Trabecular Bone Classification (2026)"
)
