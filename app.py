# -*- coding: utf-8 -*-
"""app.py — Welcome page for the trabecular bone analysis pipeline."""
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Trabecular bone pipeline",
    page_icon="🦴",
    layout="wide",
)

C_BONE   = "#C8BFA9"
C_BLUE   = "#378ADD"
C_TEAL   = "#1D9E75"
C_PURPLE = "#7F77DD"
C_CORAL  = "#E85D3A"

st.markdown("""
<style>
.block-container { max-width:1080px; padding-top:2rem; padding-bottom:3rem; }
.step-card {
    background:#fff; border-radius:14px; padding:1.3rem 1.2rem;
    border-left:5px solid var(--c,#378ADD);
    box-shadow:0 2px 10px rgba(0,0,0,0.07); height:100%;
}
.step-num  { font-size:2rem; font-weight:900; color:var(--c,#378ADD); line-height:1; }
.step-name { font-size:1rem; font-weight:700; color:#1C1C2E; margin:0.2rem 0 0.5rem; }
.step-body { font-size:0.83rem; color:#555; line-height:1.55; }
.step-note { font-size:0.76rem; color:#999; margin-top:0.5rem; font-style:italic; }
.qs-box {
    background:linear-gradient(135deg,#f0fdf8,#eaf4ff);
    border-radius:14px; padding:1.4rem 1.6rem;
    border:2px solid #1D9E75;
}
.section-label {
    font-size:0.68rem; font-weight:700; letter-spacing:0.15em;
    text-transform:uppercase; color:#aaa; margin-bottom:0.6rem;
}
.chip {
    display:inline-block; background:#EEF2FF; color:#4338CA;
    border-radius:20px; padding:0.18rem 0.7rem;
    font-size:0.73rem; font-weight:600; margin:0.12rem 0.1rem;
}
.stat-card {
    background:#fff; border-radius:12px; padding:0.9rem 1rem;
    text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.06);
    border-top:3px solid var(--c,#378ADD);
}
.stat-val { font-size:1.7rem; font-weight:800; color:var(--c,#378ADD); }
.stat-lab { font-size:0.72rem; color:#888; font-weight:600;
            text-transform:uppercase; letter-spacing:0.07em; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
hl, hr = st.columns([3, 2])
with hl:
    st.markdown("# 🦴 Trabecular Bone Analysis Pipeline")
    st.markdown(
        "Generate calibrated synthetic trabecular bone, run micro-FE analysis, "
        "validate against DVC measurements, and explore quantum similarity methods — "
        "all in one tool. Each step works independently with demo data."
    )
    st.markdown(
        '<span class="chip">Synthetic generation</span>'
        '<span class="chip">Micro-FE solver</span>'
        '<span class="chip">DVC validation</span>'
        '<span class="chip">3D strain viewer</span>'
        '<span class="chip">Quantum reservoir</span>',
        unsafe_allow_html=True,
    )

with hr:
    st.markdown("""
<div class="qs-box">
<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.13em;
          text-transform:uppercase;color:#1D9E75;margin-bottom:0.7rem;">
  Quick start — no data needed
</p>
<p style="font-size:0.87rem;color:#1C1C2E;line-height:1.7;margin:0;">
  <b>1.</b> Go to <b>Generator</b> in the sidebar<br>
  <b>2.</b> Click a bone preset (Osteoporotic / Healthy / Dense)<br>
  <b>3.</b> Hit <b>Generate</b> — takes ~5 s<br>
  <b>4.</b> Open <b>FE Solver</b> → <b>Run compression</b><br>
  <b>5.</b> View strain in <b>3D Viewer</b><br>
  <b>6.</b> Try <b>Quantum Reservoir</b> on the same volume
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Pipeline steps</p>', unsafe_allow_html=True)

steps = [
    (C_BLUE,   "1", "Data Loader",
     "Upload a real µCT scan (TIFF stack or .npy) and register a deformed state. "
     "The loader extracts morphometric targets that feed the generator automatically.",
     "Optional — skip if using synthetic data only"),
    (C_TEAL,   "2", "Generator",
     "Set BV/TV, Tb.Th and Tb.N targets — or pick a preset. The GRF generator "
     "runs a calibration loop until the measured morphometrics match your targets.",
     "Demo presets: Osteoporotic · Healthy · Dense"),
    (C_CORAL,  "3", "FE Solver",
     "Runs uniaxial compression, tension or torque on the generated volume. "
     "Outputs apparent modulus, von Mises strain field, and strain energy density.",
     "TechMesh (tet mesh) or voxel FE — both available"),
    (C_PURPLE, "4", "3D Viewer",
     "Interactive 3D bone surface with strain colour-mapped onto the mesh. "
     "Load a DVC displacement field and overlay it for co-registered comparison.",
     "Drag to rotate · scroll to zoom · load strain TIFFs"),
    (C_TEAL,   "5", "Quantum Reservoir",
     "The DVC similarity step implemented as a quantum reservoir circuit. "
     "Swap it in for classical NCC without changing any upstream code. "
     "Includes entanglement sweet-spot explorer and temporal load-sequence mode.",
     "Works on any generated volume — no real scan required"),
    (C_BLUE,   "6", "QKSVM Integration",
     "Explore how quantum kernel SVMs classify bone morphometrics. "
     "Select a dimensionality reducer, view the compressed embedding, "
     "inspect the kernel matrix, and see live BV/TV classification.",
     "Uses session morphometrics or synthetic demo data"),
]

cols = st.columns(3)
for i, (color, num, name, body, note) in enumerate(steps):
    with cols[i % 3]:
        st.markdown(
            f'<div class="step-card" style="--c:{color}; margin-bottom:1rem;">'
            f'<div class="step-num">{num}</div>'
            f'<div class="step-name">{name}</div>'
            f'<div class="step-body">{body}</div>'
            f'<div class="step-note">{note}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# WHAT THE GENERATOR PRODUCES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">What the generator produces</p>',
            unsafe_allow_html=True)

gl, gm, gr = st.columns(3)

with gl:
    st.markdown(f"""
<div class="step-card" style="--c:{C_TEAL}">
  <div style="font-size:1.4rem;margin-bottom:0.3rem">🎯</div>
  <div class="step-name">Morphometrically calibrated</div>
  <div class="step-body">
    Specify target BV/TV and Tb.Th — the generator iterates until
    measured values match within tolerance. You get <em>quantifiably
    matched</em> bone, not just structurally plausible geometry.
  </div>
</div>""", unsafe_allow_html=True)

with gm:
    st.markdown(f"""
<div class="step-card" style="--c:{C_BLUE}">
  <div style="font-size:1.4rem;margin-bottom:0.3rem">⚙️</div>
  <div class="step-name">Mechanically validated</div>
  <div class="step-body">
    FE analysis gives apparent modulus and strain fields. DVC round-trip
    validation checks that the synthetic bone <em>deforms</em> like real
    bone — not just looks like it. Round-trip RMSE: <b>0.53 vox</b>.
  </div>
</div>""", unsafe_allow_html=True)

with gr:
    st.markdown(f"""
<div class="step-card" style="--c:{C_PURPLE}">
  <div style="font-size:1.4rem;margin-bottom:0.3rem">⚛️</div>
  <div class="step-name">Quantum-ready</div>
  <div class="step-body">
    The similarity backend is a registered callable. Classical NCC or
    quantum reservoir kernel — swapped at runtime without touching the
    generator or FE code. UMAP keeps quantum and classical accuracy matched.
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STATUS METRICS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Pipeline benchmarks</p>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
metrics = [
    (C_TEAL,   "0.27 vox", "DVC RMSE · FE-driven"),
    (C_BLUE,   "0.53 vox", "DVC RMSE · round-trip"),
    (C_CORAL,  "~5 s",     "Typical generation time"),
    (C_PURPLE, "8 qubits", "Quantum reservoir (default)"),
    (C_BONE,   "6 stages", "End-to-end pipeline"),
]
for col, (color, val, lab) in zip([m1,m2,m3,m4,m5], metrics):
    with col:
        st.markdown(
            f'<div class="stat-card" style="--c:{color}">'
            f'<div class="stat-val">{val}</div>'
            f'<div class="stat-lab">{lab}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# QUANTUM PLUG-IN DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">How the quantum step plugs in</p>',
            unsafe_allow_html=True)

diag_l, diag_r = st.columns([3, 2])
with diag_l:
    st.markdown("""
<div style="background:#fff;border-radius:14px;padding:1.4rem 1.5rem;
            box-shadow:0 2px 10px rgba(0,0,0,0.07);">
  <div style="display:flex;align-items:center;gap:0.4rem;flex-wrap:wrap;
              margin-bottom:0.8rem;">
    <div style="background:#EEF6FF;border:2px solid #378ADD;border-radius:9px;
                padding:0.5rem 0.8rem;text-align:center;min-width:80px;">
      <div style="font-size:0.73rem;font-weight:700;color:#378ADD;">Sub-volume</div>
      <div style="font-size:0.62rem;color:#888;">3D patch</div>
    </div>
    <span style="color:#ccc;font-size:1.2rem;">→</span>
    <div style="background:#EEF6FF;border:2px solid #378ADD;border-radius:9px;
                padding:0.5rem 0.8rem;text-align:center;min-width:80px;">
      <div style="font-size:0.73rem;font-weight:700;color:#378ADD;">UMAP</div>
      <div style="font-size:0.62rem;color:#888;">reduce</div>
    </div>
    <span style="color:#ccc;font-size:1.2rem;">→</span>
    <div style="background:#F3F1FF;border:2px dashed #7F77DD;border-radius:9px;
                padding:0.5rem 0.8rem;text-align:center;min-width:80px;">
      <div style="font-size:0.73rem;font-weight:700;color:#7F77DD;">Quantum</div>
      <div style="font-size:0.62rem;color:#888;">feature map</div>
    </div>
    <span style="color:#ccc;font-size:1.2rem;">→</span>
    <div style="background:#EDFAF4;border:2px solid #1D9E75;border-radius:9px;
                padding:0.5rem 0.8rem;text-align:center;min-width:80px;">
      <div style="font-size:0.73rem;font-weight:700;color:#1D9E75;">Score</div>
      <div style="font-size:0.62rem;color:#888;">∈ [−1,1]</div>
    </div>
  </div>
  <p style="font-size:0.73rem;color:#aaa;margin:0;">
    Dashed border = quantum steps. Everything upstream is unchanged.
  </p>
</div>
""", unsafe_allow_html=True)
    st.code(
        'from trabecular.quantum import SIMILARITY_BACKENDS\n\n'
        '# One-line swap — upstream code unchanged\n'
        'score = SIMILARITY_BACKENDS["QRC (quantum reservoir)"](ref, def_)',
        language="python",
    )

with diag_r:
    st.markdown(f"""
<div class="step-card" style="--c:{C_PURPLE};margin-bottom:0.8rem">
  <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;
              text-transform:uppercase;color:{C_PURPLE};margin-bottom:0.3rem">
    Quantum reservoir
  </div>
  <div class="step-body">
    Fixed random entangling circuit — only the linear readout is trained.
    Performance peaks where bipartite von Neumann entropy is half-saturated
    (the entanglement sweet spot).
  </div>
</div>
<div class="step-card" style="--c:{C_TEAL}">
  <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;
              text-transform:uppercase;color:{C_TEAL};margin-bottom:0.3rem">
    UMAP is the key choice
  </div>
  <div class="step-body">
    The reducer feeding the quantum kernel determines whether it stays
    competitive. UMAP preserves topology; PCA and random projection
    produce near-identity kernels (−9–12 accuracy points).
  </div>
</div>
""", unsafe_allow_html=True)

# FOOTER
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.caption(
    "University of Greenwich · Isabella Florez · "
    "github.com/BellaZZ23/synthetic-trabecular-generation"
)

# ── Sidebar workflow guide ─────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
<style>
.wf-section { font-size:0.62rem; font-weight:800; letter-spacing:0.14em;
  text-transform:uppercase; color:#aaa; margin:0.7rem 0 0.15rem 0; }
.wf-pages { font-size:0.78rem; color:#555; line-height:1.9; padding-left:0.3rem; }
</style>
<div class="wf-section">① Prepare</div>
<div class="wf-pages">
  📂 Data Loader<br>
  🔍 ROI Detection
</div>
<div class="wf-section">② Run</div>
<div class="wf-pages">
  🧬 Generator<br>
  ⚙️ FE Solver<br>
  🔗 Pipeline<br>
  🧊 3D Viewer
</div>
<div class="wf-section">③ Analyse</div>
<div class="wf-pages">
  ⚛️ Quantum Reservoir<br>
  🗺️ QIC Roadmap<br>
  🔬 QKSVM Results<br>
  🧽 Foam Materials
</div>
""",
    unsafe_allow_html=True,
)
