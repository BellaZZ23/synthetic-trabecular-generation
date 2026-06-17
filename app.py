"""
app.py — Landing page for the Quantum-AI micro-CT bone imaging dashboard.

This is the HOME page of a multi-page Streamlit app. The pipeline itself
lives in the sidebar pages:
    Data Loader → ROI Detection → Generator → FE Solver → Pipeline → 3D Viewer

This page frames the project, shows the pipeline at a glance, and explains
where the quantum step plugs in. It is intentionally self-contained
(streamlit + numpy + matplotlib only) so it can never crash on a missing
dependency during a live talk.
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Quantum-AI bone imaging",
                   page_icon="🦴", layout="wide")

# Palette
C_BONE = "#C8BFA9"
C_BLUE = "#378ADD"
C_TEAL = "#1D9E75"
C_PURPLE = "#7F77DD"
C_CORAL = "#E85D3A"


# ══════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════
st.title("Quantum-AI synergy for micro-CT bone imaging")
st.markdown(
    "##### A classical imaging pipeline for trabecular bone — built so the "
    "quantum step drops into a single, well-defined slot."
)

intro, start = st.columns([3, 2])
with intro:
    st.markdown(
        "This dashboard takes trabecular bone from **real or synthetic "
        "micro-CT** through segmentation, mechanical analysis, and "
        "deformation comparison. The comparison step is the heart of "
        "**digital volume correlation (DVC)** — and the one place a "
        "**quantum kernel** replaces the classical similarity metric, the "
        "basis of **Quantum Image Correlation (QIC)**."
    )
with start:
    st.info(
        "**Start here**\n\n"
        "1. Open **Data Loader** in the sidebar\n"
        "2. Click **🎬 Load demo data**\n"
        "3. Walk through **Generator → Pipeline → 3D Viewer**"
    )


# ══════════════════════════════════════════════════════════════
# PIPELINE AT A GLANCE
# ══════════════════════════════════════════════════════════════
st.divider()
st.subheader("The pipeline at a glance")

fig, ax = plt.subplots(figsize=(11, 2.4))
ax.axis("off")
stages = [
    ("Load",       "real µCT / DVC",   C_BLUE,  False),
    ("Generate",   "synthetic bone",   C_TEAL,  False),
    ("Analyse",    "segment · FE",     C_TEAL,  False),
    ("Compare",    "DVC similarity",   C_TEAL,  False),
    ("Quantum",    "kernel · QIC",     C_PURPLE, True),
]
for i, (lab, sub, col, q) in enumerate(stages):
    x = i * 2.18
    ax.add_patch(plt.Rectangle((x, 0.35), 1.85, 1.05, facecolor=col,
                 edgecolor="black", alpha=0.92,
                 linestyle="--" if q else "-", linewidth=2.2))
    ax.text(x + 0.925, 1.0, lab, ha="center", va="center", fontsize=12.5,
            fontweight="bold", color="white")
    ax.text(x + 0.925, 0.62, sub, ha="center", va="center", fontsize=8.5,
            color="white", style="italic")
    if i < len(stages) - 1:
        ax.annotate("", xy=(x + 2.18, 0.875), xytext=(x + 1.85, 0.875),
                    arrowprops=dict(arrowstyle="->", lw=2.2))
ax.text(4 * 2.18 + 0.925, 1.62, "pluggable slot", ha="center", fontsize=8.5,
        style="italic", color=C_PURPLE)
ax.set_xlim(-0.2, 11.1); ax.set_ylim(0, 1.85)
plt.tight_layout()
st.pyplot(fig); plt.close()

p1, p2, p3 = st.columns(3)
with p1:
    st.markdown(
        "**Load** — *Data Loader*\n\n"
        "Upload a scan, or one-click a pre-processed D²IM specimen "
        "(reference scan, bone mask, DVC displacement field)."
    )
    st.markdown(
        "**Generate** — *Generator*\n\n"
        "Synthetic trabecular bone via a zero-crossing Gaussian random "
        "field, calibrated to a target BV/TV — with known ground truth."
    )
with p2:
    st.markdown(
        "**Analyse** — *ROI Detection · FE Solver*\n\n"
        "Isolate the trabecular ROI, measure morphometrics, and solve "
        "micro-FE for strain and apparent stiffness."
    )
    st.markdown(
        "**Compare** — *Pipeline*\n\n"
        "Align reference and deformed sub-volumes and compare the fields — "
        "the DVC similarity step."
    )
with p3:
    st.markdown(
        "**Quantum** — *the slot*\n\n"
        "The similarity backend is swappable: classical today, a quantum "
        "kernel next. Everything else stays put."
    )
    st.markdown(
        "**Visualise** — *3D Viewer*\n\n"
        "Map strain / displacement onto the 3D bone surface throughout."
    )


# ══════════════════════════════════════════════════════════════
# THE QUANTUM STEP
# ══════════════════════════════════════════════════════════════
st.divider()
st.subheader("Where the quantum step plugs in")

qcol1, qcol2 = st.columns([3, 2])
with qcol1:
    st.markdown(
        "A sub-volume is far too high-dimensional for a quantum feature map "
        "directly, so the workflow is **hybrid**: a classical reducer "
        "compresses each sub-volume, then a quantum feature map and kernel "
        "measure similarity. The comparison backend is the *only* place the "
        "quantum step enters — nothing upstream changes."
    )

    # Hybrid workflow schematic
    fig, ax = plt.subplots(figsize=(8.6, 1.7))
    ax.axis("off")
    hstages = [("Sub-\nvolume", C_BONE, False), ("UMAP\nreduce", C_BLUE, False),
               ("Quantum\nfeature map", C_PURPLE, True),
               ("Quantum\nkernel", C_PURPLE, True), ("Similarity", C_TEAL, False)]
    for i, (lab, col, q) in enumerate(hstages):
        x = i * 1.78
        ax.add_patch(plt.Rectangle((x, 0), 1.45, 1.0, facecolor=col,
                     edgecolor="black", alpha=0.92,
                     linestyle="--" if q else "-", linewidth=2))
        ax.text(x + 0.725, 0.5, lab, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white" if col != C_BONE else "black")
        if i < len(hstages) - 1:
            ax.annotate("", xy=(x + 1.78, 0.5), xytext=(x + 1.45, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=1.8))
    ax.set_xlim(-0.2, 9.1); ax.set_ylim(-0.1, 1.15)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with qcol2:
    st.markdown("**The published design rule**")
    st.info(
        "For trabecular-bone classification, the dimensionality-reduction "
        "method decides whether the quantum kernel stays competitive: "
        "**UMAP preserves quantum–classical parity**, while linear methods "
        "(PCA / random projection / PLS) lose ~9–12 points.\n\n"
        "So the reducer feeding the quantum kernel is the deciding factor — "
        "and the long-term aim is **QIC**, a successor to DVC."
    )

st.markdown("**How the kernel registers — the platform is quantum-ready**")
st.code(
    '# Similarity is a pluggable backend (classical or quantum)\n'
    'SIMILARITY_BACKENDS = {\n'
    '    "NCC (classical)": ncc,\n'
    '    "Quantum kernel":  quantum_similarity,   # <- drops in here\n'
    '}\n\n'
    'def quantum_similarity(ref_patch, def_patch):\n'
    '    """Scalar in [-1, 1]: fidelity of a quantum feature map,\n'
    '    e.g. |<phi(ref)|phi(def)>|^2 from a quantum kernel."""\n'
    '    ...',
    language="python",
)

st.success(
    "Classical backbone complete · segmentation validated against ground "
    "truth · real DVC fields loaded · quantum slot defined and registered — "
    "the kernel drops in next."
)

st.caption(
    "Isabella Florez · University of Greenwich · "
    "github.com/BellaZZ23/synthetic-trabecular-generation"
)