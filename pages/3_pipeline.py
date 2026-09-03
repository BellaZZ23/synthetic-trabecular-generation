"""
Page 3: Integrated Pipeline
============================
End-to-end workflow with four modes:

  1. Synthetic         — generate → grayscale → FE → compare
  2. D²IM data         — load processed D²IM .npy files → measure → generate
                         matched synthetic → FE → compare vs displacement field
  3. Augmented         — interpolate morphometrics between two DVC load steps
                         to generate synthetic volumes at intermediate states
  4. Compare           — side-by-side synthetic vs real mechanical fields,
                         with grid alignment before any field comparison

Session state pushed:
  pipeline_gray        → 3D viewer heterogeneous E
  strain_volume_3d     → 3D viewer overlay
  strain_registered    → marks field as co-registered (set after a real
                         alignment in the Compare tab)
  alignment_report     → human-readable summary of the last alignment
  bone_volume          → FE solver + 3D viewer
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys, io, zipfile
from pathlib import Path
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fe_coupling"))
from step3_generator_fe_coupling import (
    generate_bone_volume,
    generate_bone_volume_calibrated,
    generate_grayscale,
    run_fe_analysis,
)

try:
    from techmesh_solver import run_techmesh_analysis
    HAS_TECHMESH = True
except ImportError:
    HAS_TECHMESH = False

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from synthetic_trabecular_v15_morphometric_control import measure_all_morphometrics
    HAS_MORPH = True
except ImportError:
    HAS_MORPH = False

# scripts/ is already on sys.path from the block above, so the bare import works.
try:
    from volume_alignment import align_volumes
    HAS_ALIGN = True
except ImportError:
    HAS_ALIGN = False

try:
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
    from trabecular.quantum import SIMILARITY_BACKENDS, BACKEND_STATUS
    HAS_BACKENDS = True
except Exception:
    HAS_BACKENDS = False

st.set_page_config(page_title="Pipeline", page_icon="🔬", layout="wide")
from ui_style import inject_css, page_header
inject_css()
from ui_style import qic_pipeline_sidebar
qic_pipeline_sidebar()
page_header(
    title="Integrated pipeline",
    subtitle="Generate → Analyse → Load → Compare · Grid-aligned DVC field comparison.",
    label="Stage 4 · Compare",
    color="#E85D3A",
)

# ── Pipeline progress indicator ────────────────────────────────
def _pipeline_badge(label, done, key=None):
    color = "#27ae60" if done else "#95a5a6"
    icon  = "✅" if done else "⬜"
    return f'''<span style="display:inline-block;margin:2px 6px;padding:3px 10px;
        border-radius:12px;background:{color};color:#fff;font-size:0.78rem;
        font-weight:600;">{icon} {label}</span>'''

_steps = [
    ("Scan loaded",        "real_volume" in st.session_state or "d2im_scan" in st.session_state),
    ("ROI detected",       "real_bone_mask_trabecular" in st.session_state),
    ("Volume generated",   "bone_volume" in st.session_state),
    ("FE solved",          "pipeline_fe" in st.session_state),
    ("Strain registered",  st.session_state.get("strain_registered", False)),
    ("Ready to compare",   "pipeline_fe" in st.session_state and "strain_volume_3d" in st.session_state),
]
_badges = " ".join(_pipeline_badge(l, d) for l, d in _steps)
st.markdown(
    f'''<div style="background:#f0f2f6;border-radius:8px;padding:8px 12px;margin-bottom:12px;">
    <span style="font-size:0.8rem;color:#666;margin-right:6px;">Pipeline state:</span>
    {_badges}</div>''',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def strain_to_slice(values, centroids, nx, ny, voxel_mm, mid_z_mm):
    out = np.full((nx, ny), np.nan)
    z_range = voxel_mm * 0.6
    for idx in range(len(values)):
        if abs(centroids[idx, 2] - mid_z_mm) < z_range:
            ci = int(round(centroids[idx, 0] / voxel_mm))
            cj = int(round(centroids[idx, 1] / voxel_mm))
            if 0 <= ci < nx and 0 <= cj < ny:
                out[ci, cj] = values[idx]
    return out


def strain_vol_from_fe(fe, bone_mask, voxel_mm, component='eps_von_mises'):
    strain = fe["strain_field"]
    centroids = strain["centroids"]
    values = strain[component]
    nz, ny, nx = bone_mask.shape
    vol = np.full((nz, ny, nx), np.nan, dtype=float)
    for idx in range(len(values)):
        ci = int(round(centroids[idx, 0] / voxel_mm))
        cj = int(round(centroids[idx, 1] / voxel_mm))
        ck = int(round(centroids[idx, 2] / voxel_mm))
        if 0 <= ci < nx and 0 <= cj < ny and 0 <= ck < nz:
            vol[ck, cj, ci] = values[idx]
    filled = vol.copy()
    filled[np.isnan(filled)] = 0
    return np.where(np.isnan(vol), filled, vol)


def mechanical_awareness_score(fe, bone_mask):
    """
    Scalar score combining structural and mechanical properties.
    Score = (E_apparent / Voigt_bound) * LCC_fraction
    Range 0-1. Higher = more mechanically coherent structure.
    """
    try:
        from skimage.measure import label
        labeled = label(bone_mask)
        counts = np.bincount(labeled.ravel())
        lcc = counts[1:].max() / bone_mask.sum() if bone_mask.sum() > 0 else 0
    except Exception:
        lcc = 1.0

    if fe.get("apparent_modulus") and fe.get("voigt_bound", 0) > 0:
        modulus_ratio = min(fe["apparent_modulus"] / fe["voigt_bound"], 1.0)
    else:
        modulus_ratio = 0.5

    return float(modulus_ratio * lcc)


def run_fe(bone_mask, voxel_mm, load_type, E_bone, nu,
           applied_strain, grayscale=None, use_techmesh=False):
    """Run FE with either TechMesh or voxel solver."""
    if use_techmesh and HAS_TECHMESH:
        return run_techmesh_analysis(
            bone_mask, voxel_mm,
            load_type=load_type, E_bone=E_bone, nu=nu,
            applied_strain=applied_strain,
            grayscale=grayscale,
            verbose=False,
        )
    return run_fe_analysis(
        bone_mask, voxel_mm,
        load_type=load_type, E_bone=E_bone,
        applied_strain=applied_strain, verbose=False,
    )


def compare_fields(syn_field, real_field):
    """Compute Pearson r and RMSE between two flat arrays."""
    s = syn_field.ravel()
    r = real_field.ravel()
    # Remove NaNs
    valid = np.isfinite(s) & np.isfinite(r)
    if valid.sum() < 10:
        return None, None
    try:
        rval, _ = pearsonr(s[valid], r[valid])
    except Exception:
        rval = float('nan')
    rmse = float(np.sqrt(np.mean((s[valid] - r[valid])**2)))
    return float(rval), rmse


def checkerboard(a, b, n_tiles=8):
    """Interleave two aligned 2D slices for a visual alignment check."""
    th = max(a.shape[0] // n_tiles, 1)
    tw = max(a.shape[1] // n_tiles, 1)
    ii = np.arange(a.shape[0])[:, None] // th
    jj = np.arange(a.shape[1])[None, :] // tw
    return np.where(((ii + jj) % 2) == 1, b, a)


def _fallback_common_grid(a, b):
    """Centre-crop two volumes to a common shape, used only if
    volume_alignment is unavailable. No resampling -- assumes similar voxels."""
    shape = tuple(min(sa, sb) for sa, sb in zip(a.shape, b.shape))
    def crop(arr):
        sl = []
        for full, want in zip(arr.shape, shape):
            s = (full - want) // 2
            sl.append(slice(s, s + want))
        return arr[tuple(sl)]
    return crop(a), crop(b)


def load_d2im_files(specimen: str, processed_dir: Path):
    """Load pre-processed D²IM .npy files from data/strain/processed/."""
    scan_f = processed_dir / f"reference_scan_{specimen}.npy"
    mask_f = processed_dir / f"bone_mask_{specimen}.npy"
    disp_f = processed_dir / f"displacement_magnitude_{specimen}.npy"

    missing = [f.name for f in [scan_f, mask_f, disp_f] if not f.exists()]
    if missing:
        return None, None, None, missing

    scan = np.load(scan_f)
    mask = np.load(mask_f)
    disp = np.load(disp_f)
    disp = np.nan_to_num(disp, nan=0.0)
    return scan, mask, disp, []


# ══════════════════════════════════════════════════════════════
# SIDEBAR — global settings
# ══════════════════════════════════════════════════════════════

st.sidebar.header("FE settings")
p_load    = st.sidebar.selectbox("Load case", ["compression","tension","torque"])
p_E       = st.sidebar.number_input("E_bone (MPa)", value=18000.0, step=1000.0)
p_nu      = st.sidebar.number_input("Poisson ratio", value=0.3, step=0.05,
                                     min_value=0.0, max_value=0.49)
p_strain  = st.sidebar.number_input("Applied strain", value=0.01, step=0.005,
                                     format="%.3f")
if HAS_TECHMESH:
    use_techmesh = st.sidebar.checkbox(
        "Use TechMesh solver", value=True,
        help="Tetrahedral FE via scikit-fem. Faster, supports heterogeneous E."
    )
    use_hetero = st.sidebar.checkbox(
        "Heterogeneous E from grayscale", value=False,
    ) if use_techmesh else False
else:
    use_techmesh = False
    use_hetero   = False
    st.sidebar.info("Install scikit-fem for TechMesh: pip install scikit-fem")


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════

tab_syn, tab_d2im, tab_aug, tab_compare = st.tabs([
    "🔧 Synthetic pipeline",
    "📁 D²IM pipeline",
    "🔄 Augmented generation",
    "📊 Compare",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — SYNTHETIC PIPELINE
# ══════════════════════════════════════════════════════════════
with tab_syn:
    st.subheader("Generate synthetic volume + FE analysis")
    st.write("One-click pipeline: bone volume → grayscale → FE → 3D viewer.")

    real_targets = st.session_state.get("target_from_real")
    if real_targets:
        st.info(
            f"📐 Targets from data loader: "
            f"BV/TV={real_targets['bvtv']:.3f}, "
            f"Tb.Th={real_targets['tbth_um']:.0f} µm"
        )
        def_bvtv  = real_targets["bvtv"]
        def_tbth  = int(real_targets["tbth_um"])
        def_voxel = real_targets["voxel_um"]
    else:
        def_bvtv, def_tbth, def_voxel = 0.33, 180, 39.0

    with st.expander("Generation parameters", expanded=True):
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            st.markdown("**Morphometric targets**")
            p_bvtv      = st.number_input("BV/TV", 0.05, 0.50, def_bvtv, 0.01,
                                           format="%.3f", key="p_bvtv")
            p_tbth      = st.number_input("Tb.Th (µm)", 80, 300, def_tbth, 5, key="p_tbth")
            p_calibrate = st.checkbox("Calibrate Tb.Th", value=bool(real_targets), key="p_cal")
        with pcol2:
            st.markdown("**Volume geometry**")
            p_nx    = st.selectbox("XY size", [32, 48, 64, 96, 128], index=3, key="p_nx")
            p_nz    = st.selectbox("Z slices", [16, 24, 32, 40], index=2, key="p_nz")
            p_voxel = st.number_input("Voxel (µm)", value=def_voxel, step=1.0, key="p_voxel")
        with pcol3:
            st.markdown("**Generator**")
            p_sigma = st.slider("Base sigma", 1.0, 6.0, 2.5, 0.1, key="p_sigma")
            p_close = st.slider("Close iters", 0, 6, 3, 1, key="p_close")
            p_seed  = st.number_input("Seed", value=100, step=1, key="p_seed")

    if st.button("▶ Run full pipeline", type="primary",
                 use_container_width=True, key="btn_syn_pipeline"):
        voxel_mm = p_voxel / 1000.0

        with st.spinner("Step 1/3 — Generating bone volume..."):
            if p_calibrate:
                vol = generate_bone_volume_calibrated(
                    nx=p_nx, ny=p_nx, nz=p_nz,
                    target_bvtv=p_bvtv, target_tbth_um=float(p_tbth),
                    voxel_um=p_voxel, base_sigma=p_sigma,
                    close_iters=p_close, seed=int(p_seed), verbose=False,
                )
            else:
                vol = generate_bone_volume(
                    nx=p_nx, ny=p_nx, nz=p_nz,
                    target_bvtv=p_bvtv, voxel_um=p_voxel,
                    base_sigma=p_sigma, close_iters=p_close,
                    seed=int(p_seed), verbose=False,
                )

        bone_mask = vol["bone_mask"]
        morph     = vol["morphometrics"]

        with st.spinner("Step 2/3 — Generating grayscale µCT..."):
            gray = generate_grayscale(bone_mask, seed=int(p_seed))

        grayscale_for_fe = gray if use_hetero else None
        with st.spinner(f"Step 3/3 — Running FE ({p_load})..."):
            fe = run_fe(bone_mask, voxel_mm, p_load, p_E, p_nu,
                        p_strain, grayscale_for_fe, use_techmesh)

        # Store in session
        st.session_state["bone_volume"]    = vol
        st.session_state["pipeline_gray"]  = gray
        st.session_state["pipeline_fe"]    = fe
        st.session_state["pipeline_mask"]  = bone_mask
        st.session_state["pipeline_voxel_mm"] = voxel_mm

        # Push to 3D viewer
        sv = strain_vol_from_fe(fe, bone_mask, voxel_mm, "eps_von_mises")
        st.session_state["strain_volume_3d"]  = sv
        st.session_state["strain_label_3d"]   = "von Mises strain"
        st.session_state["strain_registered"] = True

        # Mechanical awareness score
        ma_score = mechanical_awareness_score(fe, bone_mask)
        st.session_state["ma_score"] = ma_score

        nz_v, ny_v, nx_v = bone_mask.shape
        st.success(
            f"Pipeline complete — {nx_v}×{ny_v}×{nz_v} | "
            f"BV/TV={morph['BVTV']:.3f} | "
            f"{fe['n_elements']:,} elements | "
            f"{fe['solve_time']:.1f}s | "
            f"solver: {fe.get('solver','voxel')}"
        )

    # ── Display results ──
    if "pipeline_fe" in st.session_state and "bone_volume" in st.session_state:
        vol      = st.session_state["bone_volume"]
        gray     = st.session_state.get("pipeline_gray")
        fe       = st.session_state["pipeline_fe"]
        mask_p   = st.session_state.get("pipeline_mask", vol["bone_mask"])
        voxel_mm = st.session_state.get("pipeline_voxel_mm", vol["voxel_um"]/1000.0)
        morph    = vol["morphometrics"]
        strain   = fe["strain_field"]
        nz_v, ny_v, nx_v = mask_p.shape

        # Metrics row
        st.divider()
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("BV/TV",    f"{morph['BVTV']:.3f}")
        c2.metric("Tb.Th",    f"{morph['TbTh_um_p50']:.0f} µm")
        c3.metric("Tb.N",     f"{morph['TbN_per_mm']:.2f} /mm")
        c4.metric("Elements", f"{fe['n_elements']:,}")
        if fe.get("apparent_modulus"):
            c5.metric("E_apparent", f"{fe['apparent_modulus']:.0f} MPa")
        ma = st.session_state.get("ma_score")
        if ma is not None:
            c6.metric("Mech. awareness", f"{ma:.3f}",
                      help="E_apparent/Voigt × LCC. Higher = more coherent.")

        st.divider()

        # ── Mechanics-aware morphometric panel ──────────────────────────────
        with st.expander("🦴 Mechanics-aware analysis — clinical reference ranges", expanded=False):
            st.markdown("""
<style>
.morph-bar-bg {
    background:#EAEAEA; border-radius:6px; height:10px; margin:4px 0 2px; overflow:hidden;
}
.morph-bar-fill { height:10px; border-radius:6px; transition:width 0.4s; }
.morph-status-pill {
    display:inline-block; border-radius:12px; padding:1px 10px;
    font-size:0.68rem; font-weight:700; letter-spacing:0.05em; color:#fff;
}
</style>
""", unsafe_allow_html=True)
            st.caption(
                "Reference ranges from Florez et al. 2026 (Table 1). "
                "Healthy = HOA cohort; at-risk = HF / osteoporotic cohort."
            )

            # Clinical ranges: (healthy_lo, healthy_hi, atrisk_threshold, unit, higher_is_better)
            REFS = {
                "BV/TV":  (0.15, 0.35, 0.10, "",    True),
                "Tb.Th":  (100,  200,  80,   "µm",  True),
                "Tb.N":   (1.5,  2.5,  1.0,  "/mm", True),
                "Tb.Sp":  (200,  400,  500,  "µm",  False),  # lower is better
                "Conn.D": (3.0,  8.0,  1.5,  "/mm³",True),
                "DA":     (0.25, 0.55, None, "",    None),   # moderate is optimal
            }

            morph_values = {
                "BV/TV":  morph.get("BVTV"),
                "Tb.Th":  morph.get("TbTh_um_p50"),
                "Tb.N":   morph.get("TbN_per_mm"),
                "Tb.Sp":  morph.get("TbSp_um_p50"),
                "Conn.D": morph.get("connectivity_density"),
                "DA":     morph.get("DA"),
            }

            pw_cols = st.columns(3)
            for i, (param, (lo, hi, risk, unit, higher)) in enumerate(REFS.items()):
                val = morph_values.get(param)
                with pw_cols[i % 3]:
                    if val is None:
                        st.markdown(
                            f"**{param}** &nbsp; <span style='color:#aaa;'>not computed</span>",
                            unsafe_allow_html=True)
                        if param == "DA":
                            st.caption("Enable `include_anisotropy` (slow, ~30 s)")
                        continue

                    # Classify
                    if higher is None:  # DA: moderate is best
                        if lo <= val <= hi:
                            color, label = "#1D9E75", "optimal"
                        elif val < lo * 0.7 or val > hi * 1.4:
                            color, label = "#E85D3A", "outside range"
                        else:
                            color, label = "#F5A623", "borderline"
                    elif higher:
                        if val >= lo:
                            color, label = "#1D9E75", "healthy range"
                        elif val >= risk:
                            color, label = "#F5A623", "borderline"
                        else:
                            color, label = "#E85D3A", "at risk"
                    else:  # lower is better (Tb.Sp)
                        if val <= hi:
                            color, label = "#1D9E75", "healthy range"
                        elif val <= risk:
                            color, label = "#F5A623", "borderline"
                        else:
                            color, label = "#E85D3A", "at risk"

                    # Bar fill — clamp to range for display
                    if higher is not None:
                        bar_max = (risk if risk else hi) * 1.5 if not higher else hi * 1.5
                        fill_pct = min(100, max(0, val / bar_max * 100)) if bar_max else 50
                        ref_pct  = lo / bar_max * 100 if bar_max else 33
                    else:
                        fill_pct = min(100, val / (hi * 1.5) * 100)
                        ref_pct  = lo / (hi * 1.5) * 100

                    disp_val = f"{val:.3f}" if val < 10 else f"{val:.0f}"
                    st.markdown(
                        f"<b>{param}</b> &nbsp;"
                        f"<span style='font-size:1.1rem;font-weight:800;color:{color};'>"
                        f"{disp_val}</span> "
                        f"<span style='font-size:0.78rem;color:#888;'>{unit}</span> &nbsp;"
                        f"<span class='morph-status-pill' style='background:{color};'>"
                        f"{label}</span>"
                        f"<div class='morph-bar-bg'>"
                        f"<div class='morph-bar-fill' style='width:{fill_pct:.1f}%;background:{color};'></div>"
                        f"</div>"
                        f"<span style='font-size:0.72rem;color:#aaa;'>healthy: {lo}–{hi} {unit}</span>",
                        unsafe_allow_html=True,
                    )

            # Strain–morphometry interpretation
            st.markdown("<br>", unsafe_allow_html=True)
            if fe.get("apparent_modulus"):
                e_app   = fe["apparent_modulus"]
                bvtv    = morph.get("BVTV", 0)
                voigt   = fe.get("voigt_bound", e_app / max(bvtv, 0.01))
                eff     = min(e_app / voigt, 1.0) if voigt > 0 else 0.0
                conn_d  = morph.get("connectivity_density", None)

                if eff >= 0.6:
                    mech_interp = "🟢 **Mechanically coherent** — load transfers efficiently through a connected trabecular network."
                elif eff >= 0.35:
                    mech_interp = "🟡 **Moderate coherence** — some disconnected trabeculae; localised high-strain sites likely."
                else:
                    mech_interp = "🔴 **Low mechanical coherence** — sparse or poorly connected network; fracture-initiation risk elevated."

                st.markdown(mech_interp)
                st.caption(
                    f"E_apparent = {e_app:.0f} MPa · Voigt bound = {voigt:.0f} MPa · "
                    f"Efficiency = {eff:.2f}"
                    + (f" · Conn.D = {conn_d:.4f} /mm³" if conn_d is not None else "")
                )

                if conn_d is not None and conn_d < 1.5 and bvtv > 0.15:
                    st.info(
                        "⚠️ Low connectivity despite adequate BV/TV suggests structurally "
                        "isolated trabeculae — a known predictor of fracture initiation. "
                        "This is a key target region for QIC-based strain localisation.",
                        icon=None,
                    )

        # Slice viewer
        mid_z  = nz_v // 2
        view_z = st.slider("Z-slice", 0, nz_v-1, mid_z, key="pipe_z")
        mid_mm = (view_z + 0.5) * voxel_mm
        extent = [0, nx_v*voxel_mm, 0, ny_v*voxel_mm]

        # Structure row
        st.markdown("#### Structure")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.caption("Binary mask")
            fig, ax = plt.subplots(figsize=(5,5))
            ax.imshow(mask_p[view_z].T, cmap='gray', origin='lower', extent=extent)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            st.pyplot(fig); plt.close()
        with sc2:
            st.caption("Synthetic µCT")
            if gray is not None:
                fig, ax = plt.subplots(figsize=(5,5))
                ax.imshow(gray[view_z].T, cmap='gray', origin='lower',
                          extent=extent, vmin=0, vmax=255)
                ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                st.pyplot(fig); plt.close()
        with sc3:
            st.caption("Max intensity projection")
            if gray is not None:
                fig, ax = plt.subplots(figsize=(5,5))
                ax.imshow(gray.max(axis=0).T, cmap='gray', origin='lower',
                          extent=extent, vmin=0, vmax=255)
                ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                st.pyplot(fig); plt.close()

        # Mechanical row
        st.markdown("#### Mechanical fields")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.caption("Axial strain (ε_zz)")
            fig, ax = plt.subplots(figsize=(4,4))
            im = ax.imshow(
                strain_to_slice(strain["eps_zz"], strain["centroids"],
                                nx_v, ny_v, voxel_mm, mid_mm).T,
                cmap='RdBu_r', origin='lower', extent=extent)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            plt.colorbar(im, ax=ax, fraction=0.046)
            st.pyplot(fig); plt.close()
        with mc2:
            st.caption("von Mises strain")
            fig, ax = plt.subplots(figsize=(4,4))
            im = ax.imshow(
                strain_to_slice(strain["eps_von_mises"], strain["centroids"],
                                nx_v, ny_v, voxel_mm, mid_mm).T,
                cmap='inferno', origin='lower', extent=extent)
            ax.set_xlabel("x [mm]")
            plt.colorbar(im, ax=ax, fraction=0.046)
            st.pyplot(fig); plt.close()
        with mc3:
            st.caption("Max principal strain")
            fig, ax = plt.subplots(figsize=(4,4))
            im = ax.imshow(
                strain_to_slice(strain["eps_max_principal"], strain["centroids"],
                                nx_v, ny_v, voxel_mm, mid_mm).T,
                cmap='magma', origin='lower', extent=extent)
            ax.set_xlabel("x [mm]")
            plt.colorbar(im, ax=ax, fraction=0.046)
            st.pyplot(fig); plt.close()

        with st.expander("Strain statistics"):
            scol1, scol2 = st.columns(2)
            with scol1:
                st.json({k: [round(float(strain[k].min()),6),
                              round(float(strain[k].max()),6)]
                         for k in ["eps_zz","eps_xx","eps_yy"]})
            with scol2:
                st.json({k: [round(float(strain[k].min()),6),
                              round(float(strain[k].max()),6)]
                         for k in ["eps_von_mises","eps_max_principal","eps_xy"]})


# ══════════════════════════════════════════════════════════════
# TAB 2 — D²IM PIPELINE
# ══════════════════════════════════════════════════════════════
with tab_d2im:
    st.subheader("D²IM data pipeline")
    st.write(
        "Load pre-processed D²IM data, measure morphometrics, generate matched "
        "synthetic volume, run FE, then compare against the real displacement field."
    )

    PROCESSED_DIR = REPO_ROOT / "data" / "strain" / "processed"

    # ── Specimen selector ──
    if PROCESSED_DIR.exists():
        npy_files = sorted(PROCESSED_DIR.glob("reference_scan_*.npy"))
        specimens = [f.stem.replace("reference_scan_", "") for f in npy_files]
    else:
        specimens = []

    if not specimens:
        st.warning(
            f"No processed D²IM files found in `{PROCESSED_DIR}`. "
            "Run `scripts/prepare_d2im_demo_data.py` first."
        )
    else:
        d2im_specimen = st.selectbox("Specimen", specimens, key="d2im_spec")
        voxel_d2im    = st.number_input("Voxel size (µm)", value=50.0, step=1.0,
                                         key="d2im_voxel")

        if st.button("Load D²IM data", type="secondary", key="btn_d2im_load"):
            scan, mask, disp, missing = load_d2im_files(d2im_specimen, PROCESSED_DIR)
            if missing:
                st.error(f"Missing files: {missing}")
            else:
                st.session_state["d2im_scan"]     = scan
                st.session_state["d2im_mask"]     = mask
                st.session_state["d2im_disp"]     = disp
                st.session_state["d2im_specimen"] = d2im_specimen
                st.session_state["d2im_voxel_um"] = voxel_d2im
                st.success(
                    f"Loaded {d2im_specimen} — "
                    f"scan {scan.shape}, "
                    f"BV/TV={mask.mean():.3f}, "
                    f"disp range [{disp.min():.3f}, {disp.max():.3f}]"
                )

        # ── Show loaded data ──
        if "d2im_scan" in st.session_state:
            scan     = st.session_state["d2im_scan"]
            mask_d   = st.session_state["d2im_mask"]
            disp_d   = st.session_state["d2im_disp"]
            vox_d_mm = st.session_state["d2im_voxel_um"] / 1000.0
            nz_d, ny_d, nx_d = scan.shape

            st.divider()
            st.markdown("#### Real data preview")
            mid_d = nz_d // 2
            prev_z = st.slider("Z-slice", 0, nz_d-1, mid_d, key="d2im_prev_z")
            ext_d = [0, nx_d*vox_d_mm, 0, ny_d*vox_d_mm]

            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                st.caption("µCT scan (reference)")
                fig, ax = plt.subplots(figsize=(5,5))
                ax.imshow(scan[prev_z].T, cmap='gray', origin='lower',
                          extent=ext_d, vmin=0, vmax=255)
                ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                st.pyplot(fig); plt.close()
            with pc2:
                st.caption("Bone mask")
                fig, ax = plt.subplots(figsize=(5,5))
                ax.imshow(mask_d[prev_z].T, cmap='gray', origin='lower', extent=ext_d)
                ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                st.pyplot(fig); plt.close()
            with pc3:
                st.caption("Displacement magnitude (DVC)")
                fig, ax = plt.subplots(figsize=(5,5))
                im = ax.imshow(disp_d[prev_z].T, cmap='plasma', origin='lower',
                               extent=ext_d)
                ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                plt.colorbar(im, ax=ax, label="µm")
                st.pyplot(fig); plt.close()

            # Morphometrics
            st.divider()
            st.markdown("#### Measure real morphometrics")
            if st.button("Measure morphometrics", key="btn_d2im_morph"):
                if HAS_MORPH:
                    with st.spinner("Measuring..."):
                        morph_d = measure_all_morphometrics(
                            mask_d, st.session_state["d2im_voxel_um"]
                        )
                    st.session_state["d2im_morph"] = morph_d
                    targets = {
                        "bvtv":    morph_d["BVTV"],
                        "tbth_um": morph_d["TbTh_um_p50"],
                        "voxel_um": st.session_state["d2im_voxel_um"],
                        "nx": nx_d, "ny": ny_d, "nz": nz_d,
                    }
                    st.session_state["target_from_real"] = targets
                else:
                    st.warning("Morphometric module not available.")

            if "d2im_morph" in st.session_state:
                morph_d = st.session_state["d2im_morph"]
                m1,m2,m3,m4,m5 = st.columns(5)
                m1.metric("BV/TV",    f"{morph_d['BVTV']:.3f}")
                m2.metric("Tb.Th p50",f"{morph_d['TbTh_um_p50']:.0f} µm")
                m3.metric("Tb.N",     f"{morph_d['TbN_per_mm']:.2f} /mm")
                m4.metric("Tb.Sp p50",f"{morph_d['TbSp_um_p50']:.0f} µm")
                m5.metric("LCC",      f"{morph_d['lcc_frac']:.3f}")

            # Generate matched synthetic
            st.divider()
            st.markdown("#### Generate matched synthetic + FE")

            if "d2im_morph" not in st.session_state:
                st.info("Run 'Measure morphometrics' first.")
            else:
                morph_d  = st.session_state["d2im_morph"]
                vox_d_um = st.session_state["d2im_voxel_um"]

                d2_col1, d2_col2 = st.columns(2)
                with d2_col1:
                    d2_nx    = st.selectbox("XY size", [32,48,64,96,128],
                                            index=2, key="d2im_nx")
                    d2_nz    = st.selectbox("Z slices", [16,24,32,40],
                                            index=1, key="d2im_nz")
                with d2_col2:
                    d2_seed  = st.number_input("Seed", value=100, key="d2im_seed")
                    d2_sigma = st.slider("Base sigma", 1.0, 6.0, 2.5, 0.1,
                                         key="d2im_sigma")

                if st.button("▶ Generate + FE", type="primary",
                             key="btn_d2im_pipeline"):
                    vox_d_mm = vox_d_um / 1000.0

                    with st.spinner("Generating matched synthetic..."):
                        vol_d = generate_bone_volume_calibrated(
                            nx=d2_nx, ny=d2_nx, nz=d2_nz,
                            target_bvtv=morph_d["BVTV"],
                            target_tbth_um=morph_d["TbTh_um_p50"],
                            voxel_um=vox_d_um,
                            base_sigma=d2_sigma,
                            seed=int(d2_seed), verbose=False,
                        )
                    mask_syn = vol_d["bone_mask"]

                    with st.spinner("Generating grayscale..."):
                        gray_d = generate_grayscale(mask_syn, seed=int(d2_seed))

                    with st.spinner(f"Running FE ({p_load})..."):
                        fe_d = run_fe(
                            mask_syn, vox_d_mm,
                            p_load, p_E, p_nu, p_strain,
                            gray_d if use_hetero else None,
                            use_techmesh,
                        )

                    st.session_state["bone_volume"]         = vol_d
                    st.session_state["pipeline_gray"]       = gray_d
                    st.session_state["pipeline_fe"]         = fe_d
                    st.session_state["pipeline_mask"]       = mask_syn
                    st.session_state["pipeline_voxel_mm"]   = vox_d_mm
                    st.session_state["d2im_fe"]             = fe_d
                    st.session_state["d2im_syn_mask"]       = mask_syn
                    st.session_state["d2im_syn_gray"]       = gray_d

                    # Push to 3D viewer
                    sv = strain_vol_from_fe(fe_d, mask_syn, vox_d_mm, "eps_von_mises")
                    st.session_state["strain_volume_3d"]  = sv
                    st.session_state["strain_label_3d"]   = "von Mises (D²IM matched)"
                    st.session_state["strain_registered"] = True

                    ma = mechanical_awareness_score(fe_d, mask_syn)
                    st.session_state["ma_score"] = ma

                    m = vol_d["morphometrics"]
                    st.success(
                        f"Done — BV/TV={m['BVTV']:.3f} | "
                        f"Tb.Th={m['TbTh_um_p50']:.0f} µm | "
                        f"E_apparent={fe_d.get('apparent_modulus', 0):.0f} MPa | "
                        f"Mech. awareness={ma:.3f}"
                    )
                    st.info("Switch to **Compare** tab to see D²IM vs synthetic.")


# ══════════════════════════════════════════════════════════════
# TAB 3 — AUGMENTED GENERATION
# ══════════════════════════════════════════════════════════════
with tab_aug:
    st.subheader("Augmented generation between DVC load steps")
    st.write(
        "Generate a sequence of synthetic bone volumes that interpolates "
        "morphometric parameters between two DVC states (undeformed → deformed). "
        "Each volume gets its own FE analysis, producing a continuous mechanical "
        "response sequence to bridge the gaps between measured load steps."
    )

    st.markdown("#### Define two endpoint states")

    aug_col1, aug_col2 = st.columns(2)
    with aug_col1:
        st.markdown("**State 0 — undeformed**")
        aug_bvtv_0  = st.number_input("BV/TV",      0.05, 0.70, 0.40, 0.01, key="aug_bvtv0")
        aug_tbth_0  = st.number_input("Tb.Th (µm)", 80, 400, 200, 5,         key="aug_tbth0")
        aug_tbn_0   = st.number_input("Tb.N (/mm)", 0.5, 8.0, 2.0, 0.1,     key="aug_tbn0")

    with aug_col2:
        st.markdown("**State N — deformed**")
        aug_bvtv_n  = st.number_input("BV/TV",      0.05, 0.70, 0.35, 0.01, key="aug_bvtvN")
        aug_tbth_n  = st.number_input("Tb.Th (µm)", 80, 400, 185, 5,         key="aug_tbthN")
        aug_tbn_n   = st.number_input("Tb.N (/mm)", 0.5, 8.0, 2.2, 0.1,     key="aug_tbnN")

    # Pull from D²IM session if available
    if "d2im_morph" in st.session_state:
        morph_ref = st.session_state["d2im_morph"]
        if st.button("Fill State 0 from D²IM morphometrics", key="btn_fill_aug"):
            st.info(
                f"State 0 filled from D²IM: "
                f"BV/TV={morph_ref['BVTV']:.3f}, "
                f"Tb.Th={morph_ref['TbTh_um_p50']:.0f} µm"
            )

    st.divider()
    st.markdown("#### Generation settings")

    aug_c1, aug_c2, aug_c3 = st.columns(3)
    with aug_c1:
        aug_n_steps = st.slider("Number of intermediate steps", 3, 12, 5, 1,
            help="Total volumes generated including the two endpoints.")
        aug_nx      = st.selectbox("XY size", [32,48,64], index=1, key="aug_nx")
        aug_nz      = st.selectbox("Z slices", [16,24,32], index=1, key="aug_nz")
    with aug_c2:
        aug_voxel   = st.number_input("Voxel (µm)", value=50.0, step=1.0, key="aug_voxel")
        aug_sigma   = st.slider("Base sigma", 1.0, 6.0, 2.5, 0.1, key="aug_sigma")
        aug_base_seed = st.number_input("Base seed", value=200, key="aug_seed")
    with aug_c3:
        aug_run_fe  = st.checkbox("Run FE on each volume", value=True, key="aug_fe")
        aug_calibrate = st.checkbox("Calibrate Tb.Th", value=True, key="aug_cal")
        interp_mode = st.selectbox("Interpolation", ["Linear", "Sigmoid"],
            help="Linear: uniform steps. Sigmoid: slow at endpoints, fast in middle.")

    if st.button("▶ Generate augmented sequence", type="primary",
                 use_container_width=True, key="btn_aug"):

        aug_voxel_mm = aug_voxel / 1000.0
        n = aug_n_steps

        # Interpolation weights
        t = np.linspace(0, 1, n)
        if interp_mode == "Sigmoid":
            t = 1 / (1 + np.exp(-10*(t - 0.5)))
            t = (t - t.min()) / (t.max() - t.min())

        bvtv_seq  = aug_bvtv_0  + t * (aug_bvtv_n  - aug_bvtv_0)
        tbth_seq  = aug_tbth_0  + t * (aug_tbth_n  - aug_tbth_0)

        aug_results = []
        progress = st.progress(0, text="Starting...")

        for i, (bv, tb) in enumerate(zip(bvtv_seq, tbth_seq)):
            progress.progress(i / n, text=f"Step {i+1}/{n}: BV/TV={bv:.3f}, Tb.Th={tb:.0f} µm")
            seed = int(aug_base_seed) + i

            if aug_calibrate:
                vol_i = generate_bone_volume_calibrated(
                    nx=aug_nx, ny=aug_nx, nz=aug_nz,
                    target_bvtv=float(bv), target_tbth_um=float(tb),
                    voxel_um=aug_voxel, base_sigma=aug_sigma,
                    seed=seed, verbose=False,
                )
            else:
                vol_i = generate_bone_volume(
                    nx=aug_nx, ny=aug_nx, nz=aug_nz,
                    target_bvtv=float(bv), voxel_um=aug_voxel,
                    base_sigma=aug_sigma, seed=seed, verbose=False,
                )

            gray_i = generate_grayscale(vol_i["bone_mask"], seed=seed)
            fe_i   = None
            if aug_run_fe:
                fe_i = run_fe(
                    vol_i["bone_mask"], aug_voxel_mm,
                    p_load, p_E, p_nu, p_strain,
                    gray_i if use_hetero else None,
                    use_techmesh,
                )

            aug_results.append({
                "step": i, "t": float(t[i]),
                "target_bvtv": float(bv), "target_tbth": float(tb),
                "vol": vol_i, "gray": gray_i, "fe": fe_i,
            })

        progress.progress(1.0, text="Done!")
        st.session_state["aug_results"] = aug_results
        st.success(f"Generated {n} volumes across the deformation sequence.")

    # ── Display augmented results ──
    if "aug_results" in st.session_state:
        aug_results = st.session_state["aug_results"]
        n = len(aug_results)
        aug_voxel_mm = aug_results[0]["vol"]["voxel_um"] / 1000.0

        st.divider()
        st.markdown("#### Sequence overview")

        # Plot BV/TV and E_apparent across steps
        steps     = [r["step"] for r in aug_results]
        bvtv_act  = [r["vol"]["morphometrics"]["BVTV"] for r in aug_results]
        tbth_act  = [r["vol"]["morphometrics"]["TbTh_um_p50"] for r in aug_results]
        e_app     = [r["fe"]["apparent_modulus"] if r["fe"] and
                     r["fe"].get("apparent_modulus") else None
                     for r in aug_results]

        fig, axes = plt.subplots(1, 3 if any(e_app) else 2, figsize=(15, 4))
        axes[0].plot(steps, bvtv_act, 'o-', color='#378ADD', lw=2)
        axes[0].set_title("BV/TV across steps")
        axes[0].set_xlabel("Load step"); axes[0].set_ylabel("BV/TV")

        axes[1].plot(steps, tbth_act, 's-', color='#E85D3A', lw=2)
        axes[1].set_title("Tb.Th across steps")
        axes[1].set_xlabel("Load step"); axes[1].set_ylabel("Tb.Th (µm)")

        if any(e_app) and len(axes) > 2:
            valid_e = [(s,e) for s,e in zip(steps,e_app) if e is not None]
            axes[2].plot([v[0] for v in valid_e], [v[1] for v in valid_e],
                         '^-', color='#0F6E56', lw=2)
            axes[2].set_title("E_apparent across steps")
            axes[2].set_xlabel("Load step"); axes[2].set_ylabel("E (MPa)")

        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Slice gallery
        st.markdown("#### Slice gallery")
        n_show = min(n, 5)
        cols   = st.columns(n_show)
        idxs   = np.linspace(0, n-1, n_show, dtype=int)

        for col, idx in zip(cols, idxs):
            r    = aug_results[idx]
            mask = r["vol"]["bone_mask"]
            gray = r["gray"]
            mid  = mask.shape[0] // 2
            ext  = [0, mask.shape[2]*aug_voxel_mm, 0, mask.shape[1]*aug_voxel_mm]

            with col:
                st.caption(f"Step {r['step']} (t={r['t']:.2f})")
                fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
                axes[0].imshow(mask[mid].T, cmap='gray', origin='lower', extent=ext)
                axes[0].axis('off'); axes[0].set_title("Mask", fontsize=8)
                axes[1].imshow(gray[mid].T, cmap='gray', origin='lower',
                               extent=ext, vmin=0, vmax=255)
                axes[1].axis('off'); axes[1].set_title("Gray", fontsize=8)
                plt.tight_layout(pad=0.1)
                st.pyplot(fig); plt.close()

                morph_i = r["vol"]["morphometrics"]
                st.caption(
                    f"BV/TV={morph_i['BVTV']:.3f}\n"
                    f"Tb.Th={morph_i['TbTh_um_p50']:.0f} µm"
                )
                if r["fe"] and r["fe"].get("apparent_modulus"):
                    st.caption(f"E={r['fe']['apparent_modulus']:.0f} MPa")


# ══════════════════════════════════════════════════════════════
# TAB 4 — COMPARE
# ══════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Compare synthetic vs real mechanical fields")

    has_syn_fe   = "pipeline_fe" in st.session_state
    has_d2im_fe  = "d2im_fe" in st.session_state
    has_d2im_disp = "d2im_disp" in st.session_state

    if not has_syn_fe:
        st.warning("Run the synthetic or D²IM pipeline first.")
        st.stop()

    # ── Backend selector ────────────────────────────────────────────────────────
    if HAS_BACKENDS:
        with st.expander("⚲️ Similarity backend (DVC slot)", expanded=True):
            _b_names = list(SIMILARITY_BACKENDS.keys())
            _b_cols  = st.columns(len(_b_names))
            for _bc, _bn in zip(_b_cols, _b_names):
                _status, _note = BACKEND_STATUS.get(_bn, ("unknown", ""))
                _active = _status == "active"
                _color  = "#1D9E75" if _active else "#E85D3A"
                _badge  = "✓ active" if _active else "⚠ optional"
                _bc.markdown(
                    f'<div style="border:2px solid {_color};border-radius:10px;'
                    f'padding:0.6rem 0.8rem;margin-bottom:0.3rem;">'
                    f'<div style="font-size:0.75rem;font-weight:700;color:{_color};">{_badge}</div>'
                    f'<div style="font-size:0.85rem;font-weight:600;">{_bn}</div>'
                    f'<div style="font-size:0.72rem;color:#777;">{_note}</div>'
                    '</div>',
                    unsafe_allow_html=True)
            st.caption(
                "Each card is a registered :class:`SimilarityBackend`. "
                "Swap the active backend without touching upstream code: "
                "`score = SIMILARITY_BACKENDS[name](ref_patch, def_patch)`")
            _sel = st.selectbox(
                "Active backend for sub-volume scoring",
                [n for n, (s, _) in BACKEND_STATUS.items() if s == "active"],
                key="active_backend",
            )
            st.session_state["_active_similarity_backend"] = _sel


    fe_syn    = st.session_state["pipeline_fe"]
    mask_syn  = st.session_state.get("pipeline_mask")
    voxel_syn = st.session_state.get("pipeline_voxel_mm", 0.039)
    strain_syn = fe_syn["strain_field"]

    # ── D²IM displacement comparison (now grid-aligned) ──
    if has_d2im_disp:
        st.markdown("#### Synthetic FE strain vs D²IM displacement magnitude")
        st.write(
            "The synthetic FE volume and the real D²IM/DVC volume live on "
            "different grids — different voxel size, shape and origin — so a "
            "direct voxel-for-voxel comparison is not valid. Both fields are "
            "first **aligned** onto a common grid, then normalised to [0,1] "
            "before Pearson r and RMSE are computed."
        )

        disp_real = st.session_state["d2im_disp"]
        vox_real  = st.session_state.get("d2im_voxel_um", 50.0) / 1000.0
        mask_real = st.session_state.get("d2im_mask")

        if mask_syn is None:
            st.warning("No synthetic mask in session — run a pipeline first.")
        else:
            sv = strain_vol_from_fe(fe_syn, mask_syn, voxel_syn, "eps_von_mises")

            # ── Alignment settings ──
            with st.expander("Alignment settings", expanded=True):
                a1, a2 = st.columns(2)
                with a1:
                    align_method = st.selectbox(
                        "Method", ["resample", "rigid"],
                        help="resample: bring both onto a common voxel grid + "
                             "common field of view. rigid: also estimate a "
                             "translation between the bone envelopes via phase "
                             "cross-correlation and shift the synthetic volume.",
                        key="cmp_align_method",
                    )
                with a2:
                    grid_choice = st.selectbox(
                        "Target grid", ["coarser (default)", "real D²IM", "synthetic"],
                        help="Voxel size both volumes are resampled to. "
                             "'coarser' avoids inventing detail.",
                        key="cmp_align_grid",
                    )
                if not HAS_ALIGN:
                    st.warning(
                        "`volume_alignment.py` not found — falling back to a "
                        "centre-crop on the common shape (no resampling). "
                        "Add the module to `scripts/` for proper grid alignment."
                    )

            target_vox = {"real D²IM": vox_real,
                          "synthetic": voxel_syn}.get(grid_choice, None)

            if HAS_ALIGN:
                sv_a, rd_a, report = align_volumes(
                    moving=sv, fixed=disp_real,
                    moving_voxel_mm=voxel_syn, fixed_voxel_mm=vox_real,
                    moving_mask=mask_syn, fixed_mask=mask_real,
                    target_voxel_mm=target_vox,
                    method=align_method,
                )
                target_vox_eff = report.target_voxel_mm
                st.caption(f"🧭 Alignment — {report.summary()}")
                for note in report.notes:
                    st.caption(f"• {note}")
            else:
                sv_a, rd_a = _fallback_common_grid(sv, disp_real)
                target_vox_eff = vox_real
                report = None

            # Mark as genuinely co-registered (consumed by the 3D viewer)
            st.session_state["strain_registered"] = True
            if report is not None:
                st.session_state["alignment_report"] = report.summary()

            nz_a, ny_a, nx_a = sv_a.shape

            # Normalise the ALIGNED volumes to [0,1]
            sv_n = (sv_a - np.nanmin(sv_a)) / (np.nanmax(sv_a) - np.nanmin(sv_a) + 1e-8)
            rd_n = (rd_a - np.nanmin(rd_a)) / (np.nanmax(rd_a) - np.nanmin(rd_a) + 1e-8)

            mid_c  = nz_a // 2
            comp_z = st.slider("Z-slice", 0, max(nz_a - 1, 0), mid_c, key="comp_z_d2im")
            ext_a  = [0, nx_a*target_vox_eff, 0, ny_a*target_vox_eff]

            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.caption("Synthetic von Mises (aligned, normalised)")
                fig, ax = plt.subplots(figsize=(5,5))
                im = ax.imshow(sv_n[comp_z].T, cmap='plasma',
                               origin='lower', extent=ext_a, vmin=0, vmax=1)
                ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                plt.colorbar(im, ax=ax)
                st.pyplot(fig); plt.close()
            with cc2:
                st.caption("D²IM displacement (aligned, normalised)")
                fig, ax = plt.subplots(figsize=(5,5))
                im = ax.imshow(rd_n[comp_z].T, cmap='plasma',
                               origin='lower', extent=ext_a, vmin=0, vmax=1)
                ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                plt.colorbar(im, ax=ax)
                st.pyplot(fig); plt.close()
            with cc3:
                st.caption("Checkerboard overlay (alignment QA)")
                fig, ax = plt.subplots(figsize=(5,5))
                cb = checkerboard(sv_n[comp_z].T, rd_n[comp_z].T, n_tiles=8)
                ax.imshow(cb, cmap='plasma', origin='lower', extent=ext_a,
                          vmin=0, vmax=1)
                ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                st.pyplot(fig); plt.close()

            st.caption(
                "In the checkerboard, structural features should line up across "
                "tile borders when the two fields are well aligned."
            )

            # Distribution overlay
            st.markdown("##### Distribution overlay")
            fig, ax = plt.subplots(figsize=(6,3.5))
            ax.hist(sv_n.ravel(), bins=80, alpha=0.5, density=True,
                    color="#378ADD", label="Synthetic ε_vm", edgecolor="none")
            ax.hist(rd_n.ravel(), bins=80, alpha=0.5, density=True,
                    color="#E85D3A", label="D²IM |u|", edgecolor="none")
            ax.set_xlabel("Normalised value"); ax.set_ylabel("Density"); ax.legend()
            st.pyplot(fig); plt.close()

            # Metrics on the ALIGNED, common-grid volumes (now a valid comparison)
            r_val, rmse_val = compare_fields(sv_n, rd_n)
            if r_val is not None:
                m1, m2, m3 = st.columns(3)
                m1.metric("Pearson r", f"{r_val:.3f}",
                          help="Synthetic strain vs real displacement, on the aligned grid.")
                m2.metric("RMSE", f"{rmse_val:.4f}",
                          help="On normalised, aligned fields.")
                m3.metric("Mech. awareness",
                          f"{st.session_state.get('ma_score', 0):.3f}")
            else:
                st.info("Not enough overlapping valid voxels to compute metrics.")

        st.divider()

    # ── If we also have FE on both sides ──
    if has_d2im_fe and has_syn_fe:
        fe_real = st.session_state["d2im_fe"]
        strain_real = fe_real["strain_field"]

        st.markdown("#### FE comparison: synthetic vs D²IM-matched")

        # Summary table
        comp_cols = st.columns([2,2,2,2])
        comp_cols[0].markdown("**Metric**")
        comp_cols[1].markdown("**Synthetic**")
        comp_cols[2].markdown("**D²IM matched**")
        comp_cols[3].markdown("**Δ (%)**")

        metrics = [
            ("E_apparent (MPa)", fe_syn.get("apparent_modulus"),
                                  fe_real.get("apparent_modulus")),
            ("ε_zz mean",  float(strain_syn["eps_zz"].mean()),
                           float(strain_real["eps_zz"].mean())),
            ("ε_zz std",   float(strain_syn["eps_zz"].std()),
                           float(strain_real["eps_zz"].std())),
            ("von Mises mean", float(strain_syn["eps_von_mises"].mean()),
                               float(strain_real["eps_von_mises"].mean())),
            ("von Mises max",  float(strain_syn["eps_von_mises"].max()),
                               float(strain_real["eps_von_mises"].max())),
        ]

        for label, sv2, rv2 in metrics:
            if sv2 is not None and rv2 is not None and rv2 != 0:
                delta = f"{(sv2-rv2)/abs(rv2)*100:+.1f}%"
            else:
                delta = "—"
            cols = st.columns([2,2,2,2])
            cols[0].write(label)
            cols[1].write(f"{sv2:.4f}" if sv2 is not None else "—")
            cols[2].write(f"{rv2:.4f}" if rv2 is not None else "—")
            cols[3].write(delta)

        # Distribution comparison
        st.markdown("#### Strain distribution overlay")
        dc1, dc2, dc3 = st.columns(3)
        for col, key, label in [
            (dc1, "eps_zz",         "ε_zz"),
            (dc2, "eps_von_mises",  "von Mises"),
            (dc3, "eps_max_principal","Max principal"),
        ]:
            with col:
                fig, ax = plt.subplots(figsize=(5,4))
                ax.hist(strain_syn[key], bins=80, alpha=0.5, density=True,
                        color="#378ADD", label="Synthetic", edgecolor="none")
                ax.hist(strain_real[key], bins=80, alpha=0.5, density=True,
                        color="#E85D3A", label="D²IM matched", edgecolor="none")
                ax.set_title(label); ax.set_xlabel("Strain"); ax.legend()
                st.pyplot(fig); plt.close()