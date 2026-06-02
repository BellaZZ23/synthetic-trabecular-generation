"""
Quantum Bone Pipeline — Streamlit Dashboard
============================================
Entry point. Run with:
    streamlit run app.py
"""
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Quantum Bone Pipeline",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Quantum bone pipeline")
st.sidebar.caption("v15.3 zero-crossing generator + micro-FE")


# ══════════════════════════════════════════════════════════════
# SESSION STATE SUMMARY
# ══════════════════════════════════════════════════════════════

def session_badge(condition: bool, label_on: str, label_off: str):
    if condition:
        return f"✅ {label_on}"
    return f"⬜ {label_off}"

# Gather status
has_real_scan   = "real_volume"      in st.session_state
has_real_mask   = "real_bone_mask"   in st.session_state
has_d2im        = "d2im_scan"        in st.session_state
has_registered  = st.session_state.get("strain_registered", False)
has_strain      = "strain_volume_3d" in st.session_state
has_generated   = "bone_volume"      in st.session_state
has_gray        = "pipeline_gray"    in st.session_state
has_fe          = "pipeline_fe"      in st.session_state or "fe_results" in st.session_state
has_augmented   = "aug_results"      in st.session_state
has_d2im_fe     = "d2im_fe"          in st.session_state
ma_score        = st.session_state.get("ma_score")


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════

st.markdown("# 🦴 Quantum bone pipeline")
st.caption(
    "Synthetic trabecular bone generation · Micro-FE coupling · "
    "DVC strain integration · Mechanically aware classification"
)
st.divider()


# ══════════════════════════════════════════════════════════════
# SESSION STATUS PANEL
# ══════════════════════════════════════════════════════════════

st.subheader("Session status")
st.caption("What's loaded and ready in this session.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Data**")
    st.write(session_badge(has_real_scan,  "Real µCT scan loaded",      "No real scan"))
    st.write(session_badge(has_real_mask,  "Bone mask ready",           "No bone mask"))
    st.write(session_badge(has_d2im,       "D²IM data loaded",          "No D²IM data"))
    st.write(session_badge(has_registered, "Strain field registered",   "No registration"))

    if has_real_scan:
        vol = st.session_state["real_volume"]
        vox = st.session_state.get("real_voxel_um", "?")
        st.caption(f"Scan: {vol.shape[2]}×{vol.shape[1]}×{vol.shape[0]}, {vox} µm")

    if has_d2im:
        scan_d = st.session_state["d2im_scan"]
        spec   = st.session_state.get("d2im_specimen", "")
        st.caption(f"D²IM: {spec}, {scan_d.shape[2]}×{scan_d.shape[1]}×{scan_d.shape[0]}")

with col2:
    st.markdown("**Generation**")
    st.write(session_badge(has_generated, "Synthetic volume ready",   "No volume generated"))
    st.write(session_badge(has_gray,      "Grayscale µCT ready",      "No grayscale"))
    st.write(session_badge(has_augmented, "Augmented sequence ready", "No augmented data"))

    if has_generated:
        vol_g = st.session_state["bone_volume"]
        m     = vol_g["morphometrics"]
        mask  = vol_g["bone_mask"]
        st.caption(
            f"Volume: {mask.shape[2]}×{mask.shape[1]}×{mask.shape[0]} | "
            f"BV/TV={m['BVTV']:.3f} | "
            f"Tb.Th={m['TbTh_um_p50']:.0f} µm"
        )

    if has_augmented:
        aug = st.session_state["aug_results"]
        st.caption(f"Augmented: {len(aug)} steps")

with col3:
    st.markdown("**Analysis**")
    st.write(session_badge(has_fe,      "FE results ready",         "No FE run"))
    st.write(session_badge(has_strain,  "Strain field ready (3D)",  "No strain field"))
    st.write(session_badge(has_d2im_fe, "D²IM FE complete",         "No D²IM FE"))

    if has_fe:
        fe_data = (st.session_state.get("pipeline_fe") or
                   st.session_state.get("fe_results"))
        if fe_data:
            solver = fe_data.get("solver", "voxel")
            n_elem = fe_data.get("n_elements", 0)
            e_app  = fe_data.get("apparent_modulus")
            st.caption(
                f"Solver: {solver} | {n_elem:,} elements"
                + (f" | E={e_app:.0f} MPa" if e_app else "")
            )

    if ma_score is not None:
        st.metric(
            "Mechanical awareness score", f"{ma_score:.3f}",
            help="E_apparent/Voigt × LCC fraction. Range 0–1."
        )

st.divider()


# ══════════════════════════════════════════════════════════════
# QUICK ACTIONS
# ══════════════════════════════════════════════════════════════

st.subheader("Quick actions")
st.caption("Jump to the right page for your next step.")

qa1, qa2, qa3, qa4, qa5 = st.columns(5)

with qa1:
    st.markdown("#### 📂 Data loader")
    if has_d2im:
        st.success("D²IM loaded")
    elif has_real_scan:
        st.info("Scan loaded — run registration")
    else:
        st.warning("No data loaded")
    st.markdown(
        "Load real µCT or D²IM processed files. "
        "Register image pairs. Extract morphometrics."
    )

with qa2:
    st.markdown("#### 🦴 Generator")
    if has_generated and has_gray:
        st.success("Volume + grayscale ready")
    elif has_generated:
        st.info("Volume ready — enable grayscale")
    else:
        st.warning("No volume generated")
    st.markdown(
        "Generate synthetic trabecular bone with "
        "controllable BV/TV, Tb.Th, plate/rod ratio."
    )

with qa3:
    st.markdown("#### 🔧 FE solver")
    if has_fe:
        fe_d = (st.session_state.get("pipeline_fe") or
                st.session_state.get("fe_results"))
        solver = fe_d.get("solver", "voxel") if fe_d else "?"
        st.success(f"FE done ({solver})")
    elif has_generated:
        st.info("Volume ready — run FE")
    else:
        st.warning("Generate a volume first")
    st.markdown(
        "TechMesh tet FE or voxel solver. "
        "Compression, tension, torque. "
        "Heterogeneous E from grayscale."
    )

with qa4:
    st.markdown("#### 🔬 Pipeline")
    if has_d2im_fe:
        st.success("D²IM pipeline complete")
    elif has_augmented:
        st.success("Augmented sequence ready")
    elif has_fe:
        st.info("FE done — check compare tab")
    else:
        st.warning("Run synthetic or D²IM pipeline")
    st.markdown(
        "D²IM end-to-end, augmented generation "
        "between load steps, compare synthetic "
        "vs real with Pearson r + RMSE."
    )

with qa5:
    st.markdown("#### 🧊 3D viewer")
    if has_strain and has_registered:
        st.success("Strain overlay ready")
    elif has_strain:
        st.warning("Strain loaded — not registered")
    elif has_generated:
        st.info("Volume ready — run FE first")
    else:
        st.warning("Nothing to view yet")
    st.markdown(
        "Interactive 3D bone surface with "
        "strain overlay. Clip planes, "
        "colourmap, subsample controls."
    )

st.divider()


# ══════════════════════════════════════════════════════════════
# DEMO FLOW
# ══════════════════════════════════════════════════════════════

st.subheader("Demo flow")

with st.expander("June 18 demo sequence — click to expand", expanded=False):
    st.markdown("""
**Recommended sequence for the June 18 all-day meeting:**

1. **Data Loader** → sidebar: `NumPy (.npy)` → upload `reference_scan_S9_INT_UL_AP_50.npy`
   - Set voxel size **50 µm**
   - Strain input → **Image + strain field** → upload `displacement_magnitude_S9_INT_UL_AP_50.npy`
   - Run **rigid-body registration** → check before/after panel and RMSE
   - Run **Measure morphometrics** → note BV/TV, Tb.Th

2. **Pipeline → D²IM tab** → select specimen `S9_INT_UL_AP_50` → Load D²IM data
   - Click **Measure morphometrics**
   - Set XY=64, Z=24, seed=100
   - Click **▶ Generate + FE** (TechMesh solver)
   - Note mechanical awareness score

3. **Pipeline → Augmented generation tab**
   - Fill State 0 from D²IM morphometrics
   - Set State N: BV/TV −0.05, Tb.Th −15 µm
   - 5 steps, linear interpolation
   - Click **▶ Generate augmented sequence**
   - Show BV/TV + E_apparent plots across steps

4. **Pipeline → Compare tab**
   - Show synthetic von Mises vs D²IM displacement overlay
   - Point to Pearson r and RMSE

5. **3D viewer**
   - Strain source: FE results (session)
   - Downsample: 2, Clip axis: Z at 50%, Colour scale: Plasma
   - Show **3D bone + Displacement magnitude (registered)**

**Key messages:**
- Real DVC data → registration → morphometric extraction → synthetic generation
- Mechanical awareness: synthetic bone that responds to load like the real specimen
- Augmented data generation bridges the gaps between DVC load steps
- Full pipeline from undeformed µCT to strain-aware synthetic volumes
""")


# ══════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════

st.subheader("About")

ab1, ab2 = st.columns(2)

with ab1:
    st.markdown("""
**Project**
Quantum-AI Synergy for Next-Generation Imaging of Biological Tissues
University of Greenwich, School of Engineering (2025–2028)

**Supervisors**
Gianluca Tozzi · James · Eduardo · Ahmed

**Publications**
- *Quantum kernel SVMs for trabecular bone classification*
  submitted to Quantum Machine Intelligence (IF 4.4)
- QML in biomedical imaging literature review (PRISMA 2020, n=61)
- D²IM data: Soar et al. (2024), Figshare doi:10.6084/m9.figshare.25404220
""")

with ab2:
    st.markdown("""
**Pipeline components**
- v15.3 zero-crossing Gaussian random field generator
- BV/TV + Tb.Th calibration (iterative σ search)
- Heterogeneous micro-FE: voxel solver + TechMesh (scikit-fem)
- SimpleITK rigid-body registration (phase correlation)
- BoneJ-equivalent measurements (DA, SMI, thickness maps)
- D²IM DVC displacement field integration

**Stack**
Python · Streamlit · NumPy · scikit-image · SimpleITK
scikit-fem · Plotly · matplotlib · scipy
""")

with st.expander("Session state inspector"):
    keys = list(st.session_state.keys())
    if keys:
        for k in sorted(keys):
            v = st.session_state[k]
            if isinstance(v, np.ndarray):
                st.text(f"  {k}: ndarray {v.shape} {v.dtype}")
            elif isinstance(v, dict):
                st.text(f"  {k}: dict  keys={list(v.keys())[:6]}")
            elif isinstance(v, list):
                st.text(f"  {k}: list  len={len(v)}")
            elif isinstance(v, bool):
                st.text(f"  {k}: bool  {v}")
            elif isinstance(v, (int, float)):
                st.text(f"  {k}: {type(v).__name__}  {v:.4g}")
            else:
                st.text(f"  {k}: {type(v).__name__}")
    else:
        st.info("Session is empty — nothing loaded yet.")

st.divider()
st.caption(
    "Isabella Florez · University of Greenwich, School of Engineering · "
    "Quantum-AI Synergy for Next-Generation Imaging of Biological Tissues"
)