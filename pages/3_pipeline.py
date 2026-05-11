"""
Page 3: Integrated Pipeline
============================
End-to-end workflow:
  1. Generate synthetic bone volume + grayscale micro-CT
  2. Run micro-FE to get mechanical fields (strain, displacement)
  3. Load real mechanical data (DIC strain maps, FE exports)
  4. Compare synthetic vs real — structural AND mechanical
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys, io, tempfile, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fe_coupling"))
from step3_generator_fe_coupling import (
    generate_bone_volume,
    generate_bone_volume_calibrated,
    generate_grayscale,
    run_fe_analysis,
)

# ── Try to import morphometric functions ──
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
    """Map element-centroid strain values to a 2-D image at a given z."""
    out = np.full((nx, ny), np.nan)
    z_range = voxel_mm * 0.6
    for idx in range(len(values)):
        cz = centroids[idx, 2]
        if abs(cz - mid_z_mm) < z_range:
            ci = int(round(centroids[idx, 0] / voxel_mm))
            cj = int(round(centroids[idx, 1] / voxel_mm))
            if 0 <= ci < nx and 0 <= cj < ny:
                out[ci, cj] = values[idx]
    return out


def disp_to_slice(mesh, values, nx, ny, voxel_mm, mid_z_idx):
    """Map nodal displacement to a 2-D image at a given z-slice."""
    out = np.full((nx, ny), np.nan)
    tol = voxel_mm * 0.1
    for n_idx in range(mesh.nvertices):
        x, y, z = mesh.p[:, n_idx]
        if abs(z - mid_z_idx * voxel_mm) < tol or abs(z - (mid_z_idx + 1) * voxel_mm) < tol:
            i = int(round(x / voxel_mm))
            j = int(round(y / voxel_mm))
            if 0 <= i < nx and 0 <= j < ny:
                if np.isnan(out[i, j]):
                    out[i, j] = values[n_idx]
    return out


def load_mechanical_tiffs(uploaded_files):
    """Load strain/displacement TIFF stack as a volume."""
    from PIL import Image
    slices = []
    for f in sorted(uploaded_files, key=lambda x: x.name):
        img = Image.open(f)
        n_frames = getattr(img, 'n_frames', 1)
        if n_frames > 1:
            for i in range(n_frames):
                img.seek(i)
                slices.append(np.array(img, dtype=np.float64))
        else:
            slices.append(np.array(img, dtype=np.float64))
    return np.stack(slices, axis=0)


def load_mechanical_npy(uploaded_file):
    """Load a .npy mechanical field."""
    return np.load(io.BytesIO(uploaded_file.read()))


def load_mechanical_zip(uploaded_file):
    """Load a ZIP of TIFF strain maps."""
    from PIL import Image
    slices = []
    with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as zf:
        tiff_names = sorted([
            n for n in zf.namelist()
            if n.lower().endswith(('.tif', '.tiff')) and not n.startswith('__')
        ])
        for name in tiff_names:
            with zf.open(name) as f:
                img = Image.open(f)
                slices.append(np.array(img, dtype=np.float64))
    return np.stack(slices, axis=0)


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════

tab_gen, tab_load, tab_compare = st.tabs([
    "🔧 Generate & analyse",
    "📁 Load mechanical data",
    "📊 Compare",
])

# ──────────────────────────────────────────────────────────────
# TAB 1 — GENERATE & ANALYSE
# ──────────────────────────────────────────────────────────────
with tab_gen:
    st.subheader("Generate synthetic volume + FE analysis")
    st.write("One-click pipeline: bone volume → grayscale micro-CT → mechanical fields.")

    # Check for targets from data loader
    real_targets = st.session_state.get("target_from_real", None)
    if real_targets:
        st.info(
            f"📐 Using targets from data loader: BV/TV={real_targets['bvtv']:.3f}, "
            f"Tb.Th={real_targets['tbth_um']:.0f} µm"
        )
        def_bvtv = real_targets["bvtv"]
        def_tbth = int(real_targets["tbth_um"])
        def_voxel = real_targets["voxel_um"]
    else:
        def_bvtv = 0.33
        def_tbth = 180
        def_voxel = 39.0

    # ── Parameters ──
    with st.expander("Generation parameters", expanded=True):
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            st.markdown("**Morphometric targets**")
            p_bvtv = st.number_input("BV/TV", 0.05, 0.50, def_bvtv, 0.01, format="%.3f", key="p_bvtv")
            p_tbth = st.number_input("Tb.Th target (µm)", 80, 300, def_tbth, 5, key="p_tbth")
            p_calibrate = st.checkbox("Calibrate Tb.Th", value=bool(real_targets), key="p_cal")
        with pcol2:
            st.markdown("**Volume geometry**")
            p_nx = st.selectbox("XY size", [32, 48, 64, 96, 128], index=3, key="p_nx")
            p_nz = st.selectbox("Z slices", [16, 24, 32, 40], index=2, key="p_nz")
            p_voxel = st.number_input("Voxel (µm)", value=def_voxel, step=1.0, key="p_voxel")
        with pcol3:
            st.markdown("**Field & FE**")
            p_sigma = st.slider("Base sigma", 1.0, 6.0, 2.5, 0.1, key="p_sigma")
            p_close = st.slider("Close iters", 0, 6, 3, 1, key="p_close")
            p_load = st.selectbox("Load case", ["compression", "tension", "torque"], key="p_load")
            p_E = st.number_input("E_bone (MPa)", value=18000.0, step=1000.0, key="p_E")
            p_strain = st.number_input("Applied strain", value=0.01, step=0.005, format="%.3f", key="p_strain")

    pcol_s1, pcol_s2 = st.columns(2)
    with pcol_s1:
        p_seed = st.number_input("Seed", value=100, step=1, key="p_seed")
    with pcol_s2:
        total = p_nx * p_nx * p_nz
        st.metric("Total voxels", f"{total:,}")

    # ── Run pipeline ──
    if st.button("Run full pipeline", type="primary", use_container_width=True, key="btn_pipeline"):
        voxel_mm = p_voxel / 1000.0

        # Step 1: Generate bone volume
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
        morph = vol["morphometrics"]
        nz_v, ny_v, nx_v = bone_mask.shape

        # Step 2: Generate grayscale
        with st.spinner("Step 2/3 — Generating grayscale micro-CT..."):
            gray = generate_grayscale(bone_mask, seed=int(p_seed))

        # Step 3: Run FE
        with st.spinner(f"Step 3/3 — Running FE ({p_load})..."):
            fe = run_fe_analysis(
                bone_mask, voxel_mm,
                load_type=p_load, E_bone=p_E,
                applied_strain=p_strain, verbose=False,
            )

        # Store everything
        st.session_state["bone_volume"] = vol
        st.session_state["pipeline_gray"] = gray
        st.session_state["pipeline_fe"] = fe

        st.success(
            f"Pipeline complete — {nx_v}×{ny_v}×{nz_v} | "
            f"BV/TV={morph['BVTV']:.3f} | "
            f"{fe['n_elements']} elements | "
            f"{fe['solve_time']:.1f}s"
        )

    # ── Display results ──
    if "pipeline_fe" in st.session_state and "bone_volume" in st.session_state:
        vol = st.session_state["bone_volume"]
        gray = st.session_state.get("pipeline_gray")
        fe = st.session_state["pipeline_fe"]
        bone_mask = vol["bone_mask"]
        morph = vol["morphometrics"]
        nz_v, ny_v, nx_v = bone_mask.shape
        voxel_mm = vol["voxel_um"] / 1000.0
        strain = fe["strain_field"]
        mesh = fe["mesh"]
        ux, uy, uz = fe["displacement"]

        # Morphometrics + FE summary
        st.divider()
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("BV/TV", f"{morph['BVTV']:.3f}")
        c2.metric("Tb.Th", f"{morph['TbTh_um_p50']:.0f} µm")
        c3.metric("Tb.N", f"{morph['TbN_per_mm']:.2f} /mm")
        c4.metric("Elements", f"{fe['n_elements']:,}")
        if fe["apparent_modulus"] is not None:
            c5.metric("E_app", f"{fe['apparent_modulus']:.0f} MPa")
            c6.metric("E/E_voigt", f"{fe['apparent_modulus']/fe['voigt_bound']:.3f}")
        elif fe["apparent_shear_modulus"] is not None:
            c5.metric("G_app", f"{fe['apparent_shear_modulus']:.0f} MPa")
            c6.metric("Solve time", f"{fe['solve_time']:.1f}s")

        st.divider()

        # Slice viewer
        mid_z = nz_v // 2
        if nz_v > 1:
            view_z = st.slider("Z-slice", 0, nz_v - 1, mid_z, key="pipe_z")
        else:
            view_z = 0
        mid_z_mm = (view_z + 0.5) * voxel_mm
        extent = [0, nx_v * voxel_mm, 0, ny_v * voxel_mm]

        # Row 1: Structure + grayscale
        st.markdown("#### Structural views")
        sc1, sc2, sc3 = st.columns(3)

        with sc1:
            st.caption("Binary mask")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(bone_mask[view_z].T, cmap='gray', origin='lower', extent=extent)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            st.pyplot(fig); plt.close()

        with sc2:
            st.caption("Synthetic micro-CT")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(gray[view_z].T, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=255)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            st.pyplot(fig); plt.close()

        with sc3:
            st.caption("Max intensity projection")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(gray.max(axis=0).T, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=255)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            st.pyplot(fig); plt.close()

        # Row 2: Mechanical fields
        st.markdown("#### Mechanical fields")
        mc1, mc2, mc3, mc4 = st.columns(4)

        with mc1:
            st.caption("|u| displacement")
            disp_mag = disp_to_slice(mesh, np.sqrt(ux**2 + uy**2 + uz**2), nx_v, ny_v, voxel_mm, view_z)
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(disp_mag.T, cmap='hot', origin='lower', extent=extent)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            plt.colorbar(im, ax=ax, fraction=0.046)
            st.pyplot(fig); plt.close()

        with mc2:
            st.caption("Axial strain (ε_zz)")
            ezz = strain_to_slice(strain["eps_zz"], strain["centroids"], nx_v, ny_v, voxel_mm, mid_z_mm)
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(ezz.T, cmap='RdBu_r', origin='lower', extent=extent)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            plt.colorbar(im, ax=ax, fraction=0.046)
            st.pyplot(fig); plt.close()

        with mc3:
            st.caption("von Mises strain")
            evm = strain_to_slice(strain["eps_von_mises"], strain["centroids"], nx_v, ny_v, voxel_mm, mid_z_mm)
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(evm.T, cmap='inferno', origin='lower', extent=extent)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            plt.colorbar(im, ax=ax, fraction=0.046)
            st.pyplot(fig); plt.close()

        with mc4:
            st.caption("Max principal strain")
            emp = strain_to_slice(strain["eps_max_principal"], strain["centroids"], nx_v, ny_v, voxel_mm, mid_z_mm)
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(emp.T, cmap='magma', origin='lower', extent=extent)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            plt.colorbar(im, ax=ax, fraction=0.046)
            st.pyplot(fig); plt.close()

        # Strain statistics
        with st.expander("Strain statistics"):
            scol1, scol2 = st.columns(2)
            with scol1:
                st.json({
                    "eps_zz range": [round(strain["eps_zz"].min(), 6), round(strain["eps_zz"].max(), 6)],
                    "eps_xx range": [round(strain["eps_xx"].min(), 6), round(strain["eps_xx"].max(), 6)],
                    "eps_yy range": [round(strain["eps_yy"].min(), 6), round(strain["eps_yy"].max(), 6)],
                })
            with scol2:
                st.json({
                    "von Mises range": [round(strain["eps_von_mises"].min(), 6), round(strain["eps_von_mises"].max(), 6)],
                    "max principal range": [round(strain["eps_max_principal"].min(), 6), round(strain["eps_max_principal"].max(), 6)],
                    "shear eps_xy range": [round(strain["eps_xy"].min(), 6), round(strain["eps_xy"].max(), 6)],
                })

        # Strain distribution
        with st.expander("Strain distributions"):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].hist(strain["eps_zz"], bins=80, color='#378ADD', alpha=0.8, edgecolor='none', density=True)
            axes[0].set_title("ε_zz"); axes[0].set_xlabel("Strain"); axes[0].set_ylabel("Density")
            axes[1].hist(strain["eps_von_mises"], bins=80, color='#E85D3A', alpha=0.8, edgecolor='none', density=True)
            axes[1].set_title("von Mises"); axes[1].set_xlabel("Strain")
            axes[2].hist(strain["eps_max_principal"], bins=80, color='#7B2D8E', alpha=0.8, edgecolor='none', density=True)
            axes[2].set_title("Max principal"); axes[2].set_xlabel("Strain")
            plt.tight_layout()
            st.pyplot(fig); plt.close()


# ──────────────────────────────────────────────────────────────
# TAB 2 — LOAD MECHANICAL DATA
# ──────────────────────────────────────────────────────────────
with tab_load:
    st.subheader("Load real mechanical data")
    st.write(
        "Import strain or displacement fields from DIC, external FE software, "
        "or experimental measurements for comparison against synthetic results."
    )

    load_type = st.radio(
        "Data type",
        ["Strain field (ε)", "Displacement field (u)", "Full mechanical dataset (.npz)"],
        key="mech_load_type",
    )

    mech_format = st.selectbox(
        "File format",
        ["TIFF stack", "TIFF stack (ZIP)", "NumPy (.npy)", "NumPy archive (.npz)"],
        key="mech_format",
    )

    lcol1, lcol2 = st.columns(2)
    with lcol1:
        mech_voxel = st.number_input("Voxel size (µm)", value=39.0, step=1.0, key="mech_voxel")
    with lcol2:
        if load_type == "Strain field (ε)":
            mech_component = st.selectbox(
                "Strain component",
                ["eps_zz (axial)", "eps_xx", "eps_yy", "eps_xy (shear)",
                 "eps_von_mises", "eps_max_principal"],
                key="mech_comp",
            )
        else:
            mech_component = st.selectbox(
                "Displacement component",
                ["uz (axial)", "ux", "uy", "|u| (magnitude)"],
                key="mech_comp_d",
            )

    mech_uploaded = None
    if mech_format == "TIFF stack":
        mech_uploaded = st.file_uploader(
            "Upload mechanical field TIFFs", type=["tif", "tiff"],
            accept_multiple_files=True, key="mech_tiff",
        )
    elif mech_format == "TIFF stack (ZIP)":
        mech_uploaded = st.file_uploader(
            "Upload ZIP of TIFFs", type=["zip"], key="mech_zip",
        )
    elif mech_format == "NumPy (.npy)":
        mech_uploaded = st.file_uploader(
            "Upload .npy array", type=["npy"], key="mech_npy",
            help="Shape: (Z, Y, X) — one component per file.",
        )
    elif mech_format == "NumPy archive (.npz)":
        mech_uploaded = st.file_uploader(
            "Upload .npz archive", type=["npz"], key="mech_npz",
            help="Expected keys: eps_zz, eps_von_mises, etc. or ux, uy, uz.",
        )

    if mech_uploaded:
        try:
            with st.spinner("Loading mechanical data..."):
                if mech_format == "TIFF stack" and len(mech_uploaded) > 0:
                    mech_vol = load_mechanical_tiffs(mech_uploaded)
                elif mech_format == "TIFF stack (ZIP)":
                    mech_vol = load_mechanical_zip(mech_uploaded)
                elif mech_format == "NumPy (.npy)":
                    mech_vol = load_mechanical_npy(mech_uploaded)
                elif mech_format == "NumPy archive (.npz)":
                    npz_data = np.load(io.BytesIO(mech_uploaded.read()))
                    mech_vol = None  # handled separately below

            # ── Display loaded data ──
            if mech_format == "NumPy archive (.npz)":
                st.success(f"Loaded .npz with keys: {list(npz_data.keys())}")
                st.session_state["real_mechanical_npz"] = dict(npz_data)

                # Show each field
                for key in npz_data.keys():
                    arr = npz_data[key]
                    if arr.ndim == 3:
                        nz_m, ny_m, nx_m = arr.shape
                        mid = nz_m // 2
                        voxel_mm_m = mech_voxel / 1000.0

                        st.markdown(f"**{key}** — shape {arr.shape}")
                        fig, ax = plt.subplots(figsize=(5, 5))
                        ext = [0, nx_m * voxel_mm_m, 0, ny_m * voxel_mm_m]
                        im = ax.imshow(arr[mid].T, cmap='RdBu_r', origin='lower', extent=ext)
                        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                        ax.set_title(f"{key} z={mid}")
                        plt.colorbar(im, ax=ax)
                        st.pyplot(fig); plt.close()

                        st.caption(f"Range: [{arr.min():.6f}, {arr.max():.6f}]")
            else:
                nz_m, ny_m, nx_m = mech_vol.shape
                voxel_mm_m = mech_voxel / 1000.0
                st.success(f"Loaded field: {nx_m}×{ny_m}×{nz_m}")

                # Store
                comp_key = mech_component.split(" ")[0]
                st.session_state["real_mechanical"] = {
                    "data": mech_vol,
                    "component": comp_key,
                    "voxel_um": mech_voxel,
                    "type": load_type,
                }

                # Preview
                mid_m = nz_m // 2
                if nz_m > 1:
                    mech_slice = st.slider("Z-slice", 0, nz_m - 1, mid_m, key="mech_z")
                else:
                    mech_slice = 0

                ext_m = [0, nx_m * voxel_mm_m, 0, ny_m * voxel_mm_m]

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                im0 = axes[0].imshow(mech_vol[mech_slice].T, cmap='RdBu_r', origin='lower', extent=ext_m)
                axes[0].set_title(f"{comp_key} — z={mech_slice}")
                axes[0].set_xlabel("x [mm]"); axes[0].set_ylabel("y [mm]")
                plt.colorbar(im0, ax=axes[0])

                axes[1].hist(mech_vol.ravel(), bins=100, color='#378ADD', alpha=0.8,
                             edgecolor='none', density=True)
                axes[1].set_title(f"{comp_key} distribution")
                axes[1].set_xlabel("Value"); axes[1].set_ylabel("Density")
                plt.tight_layout()
                st.pyplot(fig); plt.close()

                st.json({
                    "Component": comp_key,
                    "Shape": list(mech_vol.shape),
                    "Min": round(float(mech_vol.min()), 6),
                    "Max": round(float(mech_vol.max()), 6),
                    "Mean": round(float(mech_vol.mean()), 6),
                    "Std": round(float(mech_vol.std()), 6),
                })

        except Exception as e:
            st.error(f"Failed to load mechanical data: {e}")

    # ── Option: generate mechanical data from uploaded bone mask ──
    st.divider()
    st.markdown("#### Or: run FE on uploaded structural data")
    st.write("If you loaded a real micro-CT scan in the Data Loader, you can run FE on it here.")

    if "real_bone_mask" in st.session_state:
        real_mask = st.session_state["real_bone_mask"]
        real_voxel = st.session_state.get("real_voxel_um", 39.0)
        nz_r, ny_r, nx_r = real_mask.shape
        st.info(f"Real bone mask available: {nx_r}×{ny_r}×{nz_r}, voxel={real_voxel:.0f} µm")

        fe_col1, fe_col2, fe_col3 = st.columns(3)
        with fe_col1:
            real_load = st.selectbox("Load case", ["compression", "tension", "torque"], key="real_fe_load")
        with fe_col2:
            real_E = st.number_input("E_bone (MPa)", value=18000.0, step=1000.0, key="real_fe_E")
        with fe_col3:
            real_strain = st.number_input("Applied strain", value=0.01, step=0.005, format="%.3f", key="real_fe_s")

        if st.button("Run FE on real data", type="primary", key="btn_real_fe"):
            with st.spinner(f"Running FE on real data ({real_load})..."):
                real_fe = run_fe_analysis(
                    real_mask, real_voxel / 1000.0,
                    load_type=real_load, E_bone=real_E,
                    applied_strain=real_strain, verbose=False,
                )
            st.session_state["real_fe_results"] = real_fe
            st.success(
                f"FE complete — {real_fe['n_elements']} elements, "
                f"{real_fe['solve_time']:.1f}s"
            )
            if real_fe["apparent_modulus"] is not None:
                st.metric("E_apparent", f"{real_fe['apparent_modulus']:.0f} MPa")
    else:
        st.info("No real bone mask in session. Load a scan in the **Data Loader** first.")


# ──────────────────────────────────────────────────────────────
# TAB 3 — COMPARE
# ──────────────────────────────────────────────────────────────
with tab_compare:
    st.subheader("Compare synthetic vs real mechanical fields")

    has_syn_fe = "pipeline_fe" in st.session_state
    has_real_mech = "real_mechanical" in st.session_state
    has_real_npz = "real_mechanical_npz" in st.session_state
    has_real_fe = "real_fe_results" in st.session_state

    if not has_syn_fe:
        st.warning("No synthetic FE results yet. Run the pipeline in the **Generate & Analyse** tab.")
    if not (has_real_mech or has_real_npz or has_real_fe):
        st.warning("No real mechanical data yet. Load data or run FE on real data in the **Load Mechanical Data** tab.")

    if has_syn_fe and (has_real_mech or has_real_npz or has_real_fe):

        syn_fe = st.session_state["pipeline_fe"]
        syn_vol = st.session_state["bone_volume"]
        syn_strain = syn_fe["strain_field"]
        syn_mask = syn_vol["bone_mask"]
        nz_s, ny_s, nx_s = syn_mask.shape
        voxel_mm_s = syn_vol["voxel_um"] / 1000.0

        # ── If we have real FE results, compare directly ──
        if has_real_fe:
            real_fe = st.session_state["real_fe_results"]
            real_strain = real_fe["strain_field"]

            st.markdown("#### FE comparison: synthetic vs real")

            # Summary metrics
            comp_cols = st.columns([2, 2, 2, 2])
            comp_cols[0].markdown("**Metric**")
            comp_cols[1].markdown("**Synthetic**")
            comp_cols[2].markdown("**Real**")
            comp_cols[3].markdown("**Δ (%)**")

            fe_metrics = [
                ("E_apparent (MPa)", syn_fe.get("apparent_modulus"), real_fe.get("apparent_modulus")),
                ("Voigt bound (MPa)", syn_fe.get("voigt_bound"), real_fe.get("voigt_bound")),
                ("ε_zz mean", float(syn_strain["eps_zz"].mean()), float(real_strain["eps_zz"].mean())),
                ("ε_zz std", float(syn_strain["eps_zz"].std()), float(real_strain["eps_zz"].std())),
                ("von Mises mean", float(syn_strain["eps_von_mises"].mean()), float(real_strain["eps_von_mises"].mean())),
                ("von Mises max", float(syn_strain["eps_von_mises"].max()), float(real_strain["eps_von_mises"].max())),
            ]

            for label, sv, rv in fe_metrics:
                if sv is not None and rv is not None and rv != 0:
                    delta = f"{(sv - rv) / abs(rv) * 100:+.1f}%"
                else:
                    delta = "—"
                cols = st.columns([2, 2, 2, 2])
                cols[0].write(label)
                cols[1].write(f"{sv:.4f}" if sv is not None else "—")
                cols[2].write(f"{rv:.4f}" if rv is not None else "—")
                cols[3].write(delta)

            st.divider()

            # Strain distribution comparison
            st.markdown("#### Strain distribution overlay")
            dcol1, dcol2, dcol3 = st.columns(3)

            with dcol1:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.hist(syn_strain["eps_zz"], bins=80, alpha=0.5, density=True,
                        color="#378ADD", label="Synthetic", edgecolor="none")
                ax.hist(real_strain["eps_zz"], bins=80, alpha=0.5, density=True,
                        color="#E85D3A", label="Real", edgecolor="none")
                ax.set_title("ε_zz"); ax.set_xlabel("Strain"); ax.legend()
                st.pyplot(fig); plt.close()

            with dcol2:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.hist(syn_strain["eps_von_mises"], bins=80, alpha=0.5, density=True,
                        color="#378ADD", label="Synthetic", edgecolor="none")
                ax.hist(real_strain["eps_von_mises"], bins=80, alpha=0.5, density=True,
                        color="#E85D3A", label="Real", edgecolor="none")
                ax.set_title("von Mises"); ax.set_xlabel("Strain"); ax.legend()
                st.pyplot(fig); plt.close()

            with dcol3:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.hist(syn_strain["eps_max_principal"], bins=80, alpha=0.5, density=True,
                        color="#378ADD", label="Synthetic", edgecolor="none")
                ax.hist(real_strain["eps_max_principal"], bins=80, alpha=0.5, density=True,
                        color="#E85D3A", label="Real", edgecolor="none")
                ax.set_title("Max principal"); ax.set_xlabel("Strain"); ax.legend()
                st.pyplot(fig); plt.close()

        # ── If we have loaded mechanical fields, compare those ──
        elif has_real_mech:
            real_mech = st.session_state["real_mechanical"]
            real_data = real_mech["data"]
            comp_key = real_mech["component"]
            nz_r, ny_r, nx_r = real_data.shape
            voxel_mm_r = real_mech["voxel_um"] / 1000.0

            st.markdown(f"#### Comparing: synthetic vs real **{comp_key}**")

            # Map synthetic strain component to match
            syn_comp_map = {
                "eps_zz": syn_strain["eps_zz"],
                "eps_xx": syn_strain["eps_xx"],
                "eps_yy": syn_strain["eps_yy"],
                "eps_xy": syn_strain["eps_xy"],
                "eps_von_mises": syn_strain["eps_von_mises"],
                "eps_max_principal": syn_strain["eps_max_principal"],
            }

            max_z = min(nz_s, nz_r) - 1
            if max_z > 0:
                comp_z = st.slider("Z-slice", 0, max_z, max_z // 2, key="comp_z")
            else:
                comp_z = 0

            mid_z_mm_s = (comp_z + 0.5) * voxel_mm_s
            ext_s = [0, nx_s * voxel_mm_s, 0, ny_s * voxel_mm_s]
            ext_r = [0, nx_r * voxel_mm_r, 0, ny_r * voxel_mm_r]

            ccol1, ccol2, ccol3 = st.columns(3)

            with ccol1:
                st.caption(f"Synthetic {comp_key}")
                if comp_key in syn_comp_map:
                    syn_img = strain_to_slice(
                        syn_comp_map[comp_key], syn_strain["centroids"],
                        nx_s, ny_s, voxel_mm_s, mid_z_mm_s)
                    fig, ax = plt.subplots(figsize=(5, 5))
                    im = ax.imshow(syn_img.T, cmap='RdBu_r', origin='lower', extent=ext_s)
                    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                    plt.colorbar(im, ax=ax)
                    st.pyplot(fig); plt.close()
                else:
                    st.info(f"Component '{comp_key}' not in synthetic results")

            with ccol2:
                st.caption(f"Real {comp_key}")
                fig, ax = plt.subplots(figsize=(5, 5))
                im = ax.imshow(real_data[comp_z].T, cmap='RdBu_r', origin='lower', extent=ext_r)
                ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                plt.colorbar(im, ax=ax)
                st.pyplot(fig); plt.close()

            with ccol3:
                st.caption("Distribution overlay")
                fig, ax = plt.subplots(figsize=(5, 5))
                if comp_key in syn_comp_map:
                    ax.hist(syn_comp_map[comp_key], bins=80, alpha=0.5, density=True,
                            color="#378ADD", label="Synthetic", edgecolor="none")
                ax.hist(real_data.ravel(), bins=80, alpha=0.5, density=True,
                        color="#E85D3A", label="Real", edgecolor="none")
                ax.set_xlabel("Value"); ax.set_ylabel("Density"); ax.legend()
                st.pyplot(fig); plt.close()

            # Statistics comparison
            if comp_key in syn_comp_map:
                syn_vals = syn_comp_map[comp_key]
                st.markdown("#### Statistics")
                stat_cols = st.columns([2, 2, 2])
                stat_cols[0].markdown("**Statistic**")
                stat_cols[1].markdown("**Synthetic**")
                stat_cols[2].markdown("**Real**")

                for stat_name, syn_fn, real_fn in [
                    ("Mean", np.mean, np.mean),
                    ("Std", np.std, np.std),
                    ("Min", np.min, np.min),
                    ("Max", np.max, np.max),
                    ("Median", np.median, np.median),
                    ("p5", lambda x: np.percentile(x, 5), lambda x: np.percentile(x, 5)),
                    ("p95", lambda x: np.percentile(x, 95), lambda x: np.percentile(x, 95)),
                ]:
                    cols = st.columns([2, 2, 2])
                    cols[0].write(stat_name)
                    cols[1].write(f"{syn_fn(syn_vals):.6f}")
                    cols[2].write(f"{real_fn(real_data):.6f}")
