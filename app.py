# -*- coding: utf-8 -*-
"""
app.py — Landing page for the Quantum-AI micro-CT bone imaging dashboard.

Multi-page Streamlit app. Sidebar pages:
    Data Loader → ROI Detection → Generator → FE Solver → Pipeline → 3D Viewer → QRC

This page is intentionally self-contained (streamlit + numpy + matplotlib only).
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Quantum-AI bone imaging",
    page_icon="🦴",
    layout="wide",
)

# ── Palette ────────────────────────────────────────────────────────────────────
C_BONE   = "#C8BFA9"
C_BLUE   = "#378ADD"
C_TEAL   = "#1D9E75"
C_PURPLE = "#7F77DD"
C_CORAL  = "#E85D3A"

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Wider content column */
.block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem; }

/* Pipeline stage cards */
.stage-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.25rem 1.1rem 1rem;
    border-top: 4px solid var(--card-color, #378ADD);
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    height: 100%;
    transition: box-shadow 0.2s;
}
.stage-card:hover { box-shadow: 0 6px 22px rgba(0,0,0,0.12); }
.stage-icon  { font-size: 1.75rem; margin-bottom: 0.4rem; }
.stage-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
               text-transform: uppercase; color: var(--card-color); margin-bottom: 0.2rem; }
.stage-title { font-size: 1.05rem; font-weight: 700; color: #1C1C2E; margin-bottom: 0.4rem; }
.stage-body  { font-size: 0.85rem; color: #555; line-height: 1.5; }

/* Quantum highlight card */
.q-card {
    background: linear-gradient(135deg, #f3f1ff 0%, #eaf4ff 100%);
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    border-left: 5px solid #7F77DD;
    margin-bottom: 1rem;
}

/* Feature chip */
.chip {
    display: inline-block;
    background: #EEF2FF;
    color: #4338CA;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 0.15rem 0.1rem;
}

/* Section header */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.5rem;
}

/* Connector arrow between cards */
.arrow { text-align: center; color: #BCC0CC; font-size: 1.3rem; padding-top: 1.3rem; }

/* Code block override */
.stCode { border-radius: 10px !important; }

/* Metric card */
.metric-card {
    background: #fff;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}
.metric-val  { font-size: 1.8rem; font-weight: 800; color: #1C1C2E; }
.metric-lab  { font-size: 0.75rem; color: #888; font-weight: 600;
               text-transform: uppercase; letter-spacing: 0.08em; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
hero_l, hero_r = st.columns([5, 2])

with hero_l:
    st.markdown('<p class="section-label">PhD research · University of Greenwich</p>',
                unsafe_allow_html=True)
    st.markdown("# Quantum-AI synergy\nfor micro-CT bone imaging")
    st.markdown(
        "A validated trabecular bone pipeline — from synthetic generation "
        "through FE analysis and DVC tracking — with a quantum reservoir "
        "computing slot that replaces the classical similarity metric."
    )
    st.markdown(
        '<span class="chip">GRF generator</span>'
        '<span class="chip">Morphometric calibration</span>'
        '<span class="chip">Micro-FE</span>'
        '<span class="chip">Phase-correlation DVC</span>'
        '<span class="chip">Quantum Reservoir Computing</span>'
        '<span class="chip">JOSS package</span>',
        unsafe_allow_html=True,
    )

with hero_r:
    st.markdown("""
<div style="background:#fff; border-radius:14px; padding:1.4rem 1.3rem;
            box-shadow:0 2px 14px rgba(0,0,0,0.08); margin-top:0.5rem;">
    <p style="font-weight:700; font-size:0.95rem; margin-bottom:0.8rem;">
        ▶ Start here
    </p>
    <ol style="font-size:0.85rem; color:#444; line-height:2; margin:0; padding-left:1.2rem;">
        <li>Open <b>Data Loader</b> in the sidebar</li>
        <li>Click <b>Load demo data</b></li>
        <li>Walk through <b>Generator → Pipeline</b></li>
        <li>Explore <b>Quantum Reservoir</b> on page 5</li>
    </ol>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CARDS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-label">The pipeline</p>', unsafe_allow_html=True)

stages = [
    (C_BLUE,   "📂", "Stage 1", "Load",
     "Upload a real µCT scan or one-click a pre-processed D²IM specimen — reference scan, bone mask, DVC displacement field."),
    (C_TEAL,   "🧬", "Stage 2", "Generate",
     "Synthetic trabecular bone via a zero-crossing Gaussian random field, morphometrically calibrated to a target BV/TV and Tb.Th."),
    (C_TEAL,   "🔬", "Stage 3", "Analyse",
     "Isolate the trabecular ROI, compute BoneJ-equivalent morphometrics, and solve micro-FE for strain and apparent stiffness."),
    (C_CORAL,  "↔️", "Stage 4", "Compare",
     "Align reference and deformed sub-volumes on a common grid, then measure similarity — the DVC step. The quantum slot lives here."),
    (C_PURPLE, "⚛️", "Stage 5", "Quantum",
     "The similarity backend is a registered callable — swap classical NCC for a quantum reservoir kernel without touching upstream code."),
    (C_BONE,   "🧊", "Stage 6", "Visualise",
     "Map strain and displacement onto the 3D bone surface. Checkerboard overlay for alignment QA."),
]

cols = st.columns([1, 0.18, 1, 0.18, 1, 0.18, 1, 0.18, 1, 0.18, 1])
card_indices = [0, 2, 4, 6, 8, 10]

for idx, (col_idx, (color, icon, label, title, body)) in enumerate(
        zip(card_indices, stages)):
    with cols[col_idx]:
        st.markdown(f"""
<div class="stage-card" style="--card-color:{color}">
    <div class="stage-icon">{icon}</div>
    <div class="stage-label">{label}</div>
    <div class="stage-title">{title}</div>
    <div class="stage-body">{body}</div>
</div>
""", unsafe_allow_html=True)
    if idx < len(stages) - 1:
        with cols[card_indices[idx] + 1]:
            st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# WHAT MAKES THIS DIFFERENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-label">How this differs from existing generators</p>',
            unsafe_allow_html=True)

diff_cols = st.columns(3)

with diff_cols[0]:
    st.markdown("""
<div class="stage-card" style="--card-color:#1D9E75">
    <div class="stage-icon">🎯</div>
    <div class="stage-title">Morphometric calibration loop</div>
    <div class="stage-body">
        TPMS (gyroid) and spinodoid generators produce <em>structurally plausible</em>
        bone. This generator closes a calibration loop: specify BV/TV, Tb.Th and Tb.N →
        generate → measure → verify. The output is quantifiably matched to a real specimen.
    </div>
</div>""", unsafe_allow_html=True)

with diff_cols[1]:
    st.markdown("""
<div class="stage-card" style="--card-color:#378ADD">
    <div class="stage-icon">⚙️</div>
    <div class="stage-title">Closed mechanical loop</div>
    <div class="stage-body">
        The pipeline doesn't stop at geometry. Micro-FE gives apparent modulus and
        strain fields; DVC validation quantifies how well the synthetic bone
        <em>deforms</em> like the real specimen. Sub-voxel RMSE is the target.
        No other open generator closes this loop.
    </div>
</div>""", unsafe_allow_html=True)

with diff_cols[2]:
    st.markdown("""
<div class="stage-card" style="--card-color:#7F77DD">
    <div class="stage-icon">⚛️</div>
    <div class="stage-title">Quantum slot architecture</div>
    <div class="stage-body">
        The DVC similarity step is a registered callable — classical NCC or quantum
        reservoir kernel, swapped without changing any upstream code. UMAP preserves
        quantum–classical parity; linear reducers lose ~9–12 points. The entanglement
        sweet spot is a testable, reproducible property of the reservoir.
    </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# QUANTUM STEP — HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-label">Where the quantum step plugs in</p>',
            unsafe_allow_html=True)

qleft, qright = st.columns([3, 2])

with qleft:
    # Hybrid workflow — clean HTML instead of matplotlib
    st.markdown("""
<div style="background:#fff; border-radius:14px; padding:1.4rem 1.5rem;
            box-shadow:0 2px 12px rgba(0,0,0,0.07); margin-bottom:1rem;">
    <p style="font-size:0.7rem; font-weight:700; letter-spacing:0.12em;
              text-transform:uppercase; color:#888; margin-bottom:1rem;">
        Hybrid quantum-classical workflow
    </p>
    <div style="display:flex; align-items:center; gap:0.4rem; flex-wrap:wrap;">
        <div style="background:#EEF6FF; border:2px solid #378ADD; border-radius:10px;
                    padding:0.6rem 0.9rem; text-align:center; min-width:90px;">
            <div style="font-size:0.75rem; font-weight:700; color:#378ADD;">Sub-volume</div>
            <div style="font-size:0.65rem; color:#888;">3D patch</div>
        </div>
        <span style="color:#BCC0CC; font-size:1.2rem;">→</span>
        <div style="background:#EEF6FF; border:2px solid #378ADD; border-radius:10px;
                    padding:0.6rem 0.9rem; text-align:center; min-width:90px;">
            <div style="font-size:0.75rem; font-weight:700; color:#378ADD;">UMAP</div>
            <div style="font-size:0.65rem; color:#888;">reduce</div>
        </div>
        <span style="color:#BCC0CC; font-size:1.2rem;">→</span>
        <div style="background:#F3F1FF; border:2px dashed #7F77DD; border-radius:10px;
                    padding:0.6rem 0.9rem; text-align:center; min-width:90px;">
            <div style="font-size:0.75rem; font-weight:700; color:#7F77DD;">Quantum</div>
            <div style="font-size:0.65rem; color:#888;">feature map</div>
        </div>
        <span style="color:#BCC0CC; font-size:1.2rem;">→</span>
        <div style="background:#F3F1FF; border:2px dashed #7F77DD; border-radius:10px;
                    padding:0.6rem 0.9rem; text-align:center; min-width:90px;">
            <div style="font-size:0.75rem; font-weight:700; color:#7F77DD;">Kernel</div>
            <div style="font-size:0.65rem; color:#888;">similarity</div>
        </div>
        <span style="color:#BCC0CC; font-size:1.2rem;">→</span>
        <div style="background:#EDFAF4; border:2px solid #1D9E75; border-radius:10px;
                    padding:0.6rem 0.9rem; text-align:center; min-width:90px;">
            <div style="font-size:0.75rem; font-weight:700; color:#1D9E75;">Score</div>
            <div style="font-size:0.65rem; color:#888;">∈ [−1, 1]</div>
        </div>
    </div>
    <p style="font-size:0.75rem; color:#888; margin-top:0.8rem; margin-bottom:0;">
        Dashed border = quantum steps. Everything else is classical and unchanged.
    </p>
</div>
""", unsafe_allow_html=True)

    st.code(
        '# Similarity is a pluggable backend — one-line swap\n'
        'SIMILARITY_BACKENDS = {\n'
        '    "NCC (classical)": ncc_similarity,\n'
        '    "QRC similarity":  qrc_similarity,   # ← drops in here\n'
        '}\n\n'
        'score = SIMILARITY_BACKENDS["QRC similarity"](ref_patch, def_patch)',
        language="python",
    )

with qright:
    st.markdown("""
<div class="q-card">
    <p style="font-size:0.7rem; font-weight:700; letter-spacing:0.12em;
              text-transform:uppercase; color:#7F77DD; margin-bottom:0.6rem;">
        Published finding
    </p>
    <p style="font-size:0.9rem; color:#1C1C2E; line-height:1.6; margin-bottom:0.8rem;">
        The dimensionality-reduction method determines whether the quantum kernel
        stays competitive. <strong>UMAP preserves quantum–classical parity.</strong>
        Linear methods (PCA, random projection) lose ~9–12 accuracy points.
    </p>
    <p style="font-size:0.75rem; color:#666;">
        → The reducer feeding the quantum slot is the deciding factor.
    </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="q-card" style="border-left-color:#1D9E75;
     background:linear-gradient(135deg,#f0fdf8 0%,#f7fffe 100%);">
    <p style="font-size:0.7rem; font-weight:700; letter-spacing:0.12em;
              text-transform:uppercase; color:#1D9E75; margin-bottom:0.6rem;">
        Entanglement sweet spot
    </p>
    <p style="font-size:0.9rem; color:#1C1C2E; line-height:1.6; margin-bottom:0;">
        Reservoir performance peaks when bipartite von Neumann entropy is
        half-saturated. Too shallow → separable states.
        Too deep → Haar-random, information washed out.
    </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# QRC PREVIEW — 3 feature cards
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-label">Quantum Reservoir Computing — what\'s new on page 5</p>',
            unsafe_allow_html=True)

qrc1, qrc2, qrc3 = st.columns(3)

with qrc1:
    st.markdown("""
<div class="stage-card" style="--card-color:#7F77DD">
    <div class="stage-icon">🔁</div>
    <div class="stage-title">Fixed reservoir</div>
    <div class="stage-body">
        A random quantum circuit acts as a fixed, high-dimensional feature map.
        Only the linear readout is trained — fast, stable, no vanishing gradients.
        95% of the quantum benefit with none of the training overhead.
    </div>
</div>""", unsafe_allow_html=True)

with qrc2:
    st.markdown("""
<div class="stage-card" style="--card-color:#7F77DD">
    <div class="stage-icon">📈</div>
    <div class="stage-title">Entanglement sweet spot</div>
    <div class="stage-body">
        Performance peaks where bipartite entropy is half-saturated — detectable
        from the entropy-vs-depth curve. Verified automatically by the test suite
        for any input. This is the novel measurable result.
    </div>
</div>""", unsafe_allow_html=True)

with qrc3:
    st.markdown("""
<div class="stage-card" style="--card-color:#7F77DD">
    <div class="stage-icon">⏱️</div>
    <div class="stage-title">Temporal / load-sequence mode</div>
    <div class="stage-body">
        Morphometric snapshots x₁→x₂→…→xₜ fed sequentially without resetting.
        The reservoir's quantum memory bridges gaps between DVC load steps —
        encoding mechanical history across the loading sequence.
    </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STATUS / METRICS ROW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-label">Pipeline status</p>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
metrics = [
    ("95", "Tests passing", C_TEAL),
    ("< 1 vox", "DVC round-trip RMSE", C_BLUE),
    ("0.1.0", "Package version", C_PURPLE),
    ("6", "Pipeline stages", C_CORAL),
]
for col, (val, lab, color) in zip([m1, m2, m3, m4], metrics):
    with col:
        st.markdown(f"""
<div class="metric-card" style="border-top:3px solid {color}">
    <div class="metric-val" style="color:{color}">{val}</div>
    <div class="metric-lab">{lab}</div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.caption(
    "Isabella Florez · University of Greenwich · "
    "github.com/BellaZZ23/synthetic-trabecular-generation · "
    "branch: joss-packaging"
)
