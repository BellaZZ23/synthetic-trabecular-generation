"""
Page 3: Integrated Pipeline
============================
End-to-end workflow with four modes:

  1. Synthetic         — generate → grayscale → FE → compare
  2. D²IM data         — load processed D²IM .npy files → measure → generate
                         matched synthetic → FE → compare vs displacement field
  3. Augmented         — interpolate morphometrics between two DVC load steps
                         to generate synthetic volumes at intermediate states
  4. Compare           — side-by-side synthetic vs real mechanical fields

Session state pushed:
  pipeline_gray        → 3D viewer heterogeneous E
  strain_volume_3d     → 3D viewer overlay
  strain_registered    → marks field as co-registered
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

st.set_page_config(page_title="Pipeline", page_icon="🔬", layout="wide")
st.title("Integrated pipeline")
st.caption("Generate → Analyse → Load → Compare")


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

    fe_syn    = st.session_state["pipeline_fe"]
    mask_syn  = st.session_state.get("pipeline_mask")
    voxel_syn = st.session_state.get("pipeline_voxel_mm", 0.039)
    strain_syn = fe_syn["strain_field"]

    # ── D²IM displacement comparison ──
    if has_d2im_disp:
        st.markdown("#### Synthetic FE strain vs D²IM displacement magnitude")
        st.write(
            "Comparing synthetic von Mises strain against the real DVC "
            "displacement magnitude from D²IM. Both are normalised to [0,1] "
            "before computing Pearson r and RMSE."
        )

        disp_real = st.session_state["d2im_disp"]
        vox_real  = st.session_state.get("d2im_voxel_um", 50.0) / 1000.0
        nz_r2, ny_r2, nx_r2 = disp_real.shape

        # Build synthetic von Mises volume at real scan resolution
        if mask_syn is not None:
            nz_s, ny_s, nx_s = mask_syn.shape
            sv = strain_vol_from_fe(fe_syn, mask_syn, voxel_syn, "eps_von_mises")

            # Normalise both to [0,1]
            sv_n = (sv - sv.min()) / (sv.max() - sv.min() + 1e-8)
            rd_n = (disp_real - disp_real.min()) / \
                   (disp_real.max() - disp_real.min() + 1e-8)

            # Compare on common z-slice
            mid_c = min(nz_s, nz_r2) // 2
            comp_z = st.slider("Z-slice", 0, min(nz_s, nz_r2)-1, mid_c,
                                key="comp_z_d2im")

            ext_s = [0, nx_s*voxel_syn, 0, ny_s*voxel_syn]
            ext_r = [0, nx_r2*vox_real, 0, ny_r2*vox_real]

            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.caption("Synthetic von Mises (normalised)")
                fig, ax = plt.subplots(figsize=(5,5))
                im = ax.imshow(sv_n[comp_z].T, cmap='plasma',
                               origin='lower', extent=ext_s, vmin=0, vmax=1)
                ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                plt.colorbar(im, ax=ax)
                st.pyplot(fig); plt.close()

            with cc2:
                st.caption("D²IM displacement magnitude (normalised)")
                fig, ax = plt.subplots(figsize=(5,5))
                im = ax.imshow(rd_n[comp_z].T, cmap='plasma',
                               origin='lower', extent=ext_r, vmin=0, vmax=1)
                ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                plt.colorbar(im, ax=ax)
                st.pyplot(fig); plt.close()

            with cc3:
                st.caption("Distribution overlay")
                fig, ax = plt.subplots(figsize=(5,5))
                ax.hist(sv_n.ravel(), bins=80, alpha=0.5, density=True,
                        color="#378ADD", label="Synthetic ε_vm", edgecolor="none")
                ax.hist(rd_n.ravel(), bins=80, alpha=0.5, density=True,
                        color="#E85D3A", label="D²IM |u|", edgecolor="none")
                ax.set_xlabel("Normalised value")
                ax.set_ylabel("Density"); ax.legend()
                st.pyplot(fig); plt.close()

            # Pearson r and RMSE on flattened volumes
            r_val, rmse_val = compare_fields(sv_n, rd_n)
            if r_val is not None:
                m1, m2, m3 = st.columns(3)
                m1.metric("Pearson r",
                          f"{r_val:.3f}",
                          help="Correlation between synthetic strain and real displacement.")
                m2.metric("RMSE",
                          f"{rmse_val:.4f}",
                          help="Root-mean-square error on normalised fields.")
                m3.metric("Mech. awareness",
                          f"{st.session_state.get('ma_score', 0):.3f}")

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