"""
Page 1: Bone Volume Generator
==============================
Interactive controls for the v15.3 zero-crossing generator.
Generates a volume and displays the bone structure, morphometrics,
and mid-slice visualisation.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Import generator from fe_coupling
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fe_coupling"))
from step3_generator_fe_coupling import generate_bone_volume, generate_grayscale

st.set_page_config(page_title="Bone generator", page_icon="🦴", layout="wide")
st.title("Bone volume generator")
st.caption("v15.3 zero-crossing Gaussian random field")

# ── Sidebar controls ──
st.sidebar.header("Generator parameters")

col_morph, col_geom = st.sidebar.columns(2)
with col_morph:
    target_bvtv = st.slider("Target BV/TV", 0.10, 0.45, 0.30, 0.01)
with col_geom:
    voxel_um = st.number_input("Voxel size (um)", value=39.0, step=1.0)

col_xy, col_z = st.sidebar.columns(2)
with col_xy:
    nx = st.selectbox("XY size", [16, 32, 48, 64, 96, 128], index=1)
with col_z:
    nz = st.selectbox("Z slices", [8, 16, 24, 32, 40], index=1)

st.sidebar.subheader("Field parameters")
base_sigma = st.sidebar.slider("Base sigma", 1.5, 5.0, 2.5, 0.1,
    help="Controls the spatial frequency of the random field. Higher = coarser structure.")
warp_amp = st.sidebar.slider("Warp amplitude", 0.0, 3.0, 1.2, 0.1,
    help="Elastic deformation amplitude. Higher = more irregular shapes.")
warp_sigma = st.sidebar.slider("Warp sigma", 5.0, 20.0, 12.0, 1.0,
    help="Correlation length of the elastic deformation.")

seed = st.sidebar.number_input("Random seed", value=42, step=1)
show_grayscale = st.sidebar.checkbox("Generate grayscale", value=True)

# ── Generate button ──
if st.sidebar.button("Generate volume", type="primary", use_container_width=True):

    with st.spinner("Generating bone volume..."):
        vol = generate_bone_volume(
            nx=nx, ny=nx, nz=nz,
            target_bvtv=target_bvtv,
            voxel_um=voxel_um,
            base_sigma=base_sigma,
            warp_amp=warp_amp,
            warp_sigma=warp_sigma,
            seed=int(seed),
            verbose=False,
        )

    # Store in session state for use by FE page
    st.session_state["bone_volume"] = vol

    bone_mask = vol["bone_mask"]
    morph = vol["morphometrics"]
    nz_actual, ny_actual, nx_actual = bone_mask.shape
    mid_z = nz_actual // 2
    voxel_mm = voxel_um / 1000.0

    # ── Morphometrics cards ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BV/TV", f"{morph['BVTV']:.3f}", f"{morph['BVTV'] - target_bvtv:+.3f}")
    c2.metric("Tb.Th", f"{morph['TbTh_um_p50']:.0f} um")
    c3.metric("Tb.N", f"{morph['TbN_per_mm']:.2f} /mm")
    c4.metric("Tb.Sp", f"{morph['TbSp_um_p50']:.0f} um")
    c5.metric("LCC", f"{morph['lcc_frac']:.3f}")

    st.divider()

    # ── Slice viewer ──
    slice_idx = st.slider("Z-slice", 0, nz_actual - 1, mid_z)

    if show_grayscale:
        gray = generate_grayscale(bone_mask, seed=int(seed))

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("Binary mask")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(bone_mask[slice_idx].T, cmap='gray', origin='lower',
                  extent=[0, nx_actual*voxel_mm, 0, ny_actual*voxel_mm])
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title(f"z-slice {slice_idx}")
        st.pyplot(fig)
        plt.close()

    with col_b:
        if show_grayscale:
            st.subheader("Synthetic micro-CT")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(gray[slice_idx].T, cmap='gray', origin='lower',
                      extent=[0, nx_actual*voxel_mm, 0, ny_actual*voxel_mm],
                      vmin=0, vmax=255)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            ax.set_title(f"Grayscale z-slice {slice_idx}")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Enable 'Generate grayscale' in sidebar to see synthetic micro-CT")

    with col_c:
        st.subheader("3D projection")
        # Max-intensity projection along z
        fig, ax = plt.subplots(figsize=(5, 5))
        mip = bone_mask.max(axis=0)
        ax.imshow(mip.T, cmap='gray', origin='lower',
                  extent=[0, nx_actual*voxel_mm, 0, ny_actual*voxel_mm])
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title("Max intensity projection (z)")
        st.pyplot(fig)
        plt.close()

    # ── Full morphometrics table ──
    with st.expander("Full morphometric measurements"):
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.json({
                "BV/TV": round(morph["BVTV"], 4),
                "Tb.Th p50 (um)": round(morph["TbTh_um_p50"], 1),
                "Tb.Th p90 (um)": round(morph["TbTh_um_p90"], 1),
                "Tb.N (/mm)": round(morph["TbN_per_mm"], 3),
            })
        with mcol2:
            st.json({
                "Tb.Sp p50 (um)": round(morph["TbSp_um_p50"], 1),
                "Tb.Sp p90 (um)": round(morph["TbSp_um_p90"], 1),
                "Euler number": morph["Euler"],
                "LCC fraction": round(morph["lcc_frac"], 4),
                "Components": morph["n_components"],
            })

    # ── Calibration info ──
    with st.expander("Generator calibration details"):
        st.json(vol["calibration"])

    st.success(f"Volume generated: {nx_actual}x{ny_actual}x{nz_actual} voxels, "
               f"seed={vol['seed']}, BV/TV={morph['BVTV']:.3f}")

else:
    if "bone_volume" in st.session_state:
        st.info("Previous volume loaded from session. Click 'Generate volume' to create a new one, "
                "or go to the FE solver page to analyse it.")
    else:
        st.info("Adjust parameters in the sidebar and click 'Generate volume' to start.")
