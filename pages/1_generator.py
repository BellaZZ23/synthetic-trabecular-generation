"""
Page 1: Bone Volume Generator
==============================
Interactive controls for the v15.3 zero-crossing generator.
All parameters exposed with published honeycomb defaults.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fe_coupling"))
from step3_generator_fe_coupling import generate_bone_volume, generate_bone_volume_calibrated, generate_grayscale

st.set_page_config(page_title="Bone generator", page_icon="🦴", layout="wide")
st.title("Bone volume generator")
st.caption("v15.3 zero-crossing Gaussian random field — published honeycomb defaults")

# ── Check for targets pushed from Data Loader ──
real_targets = st.session_state.get("target_from_real", None)
if real_targets:
    st.info(
        f"📐 Targets loaded from real data: BV/TV={real_targets['bvtv']:.3f}, "
        f"Tb.Th={real_targets['tbth_um']:.0f} µm, "
        f"volume={real_targets['nx']}×{real_targets['ny']}×{real_targets['nz']}"
    )
    default_bvtv = real_targets["bvtv"]
    default_tbth = int(real_targets["tbth_um"])
    default_voxel = real_targets["voxel_um"]
else:
    default_bvtv = 0.33
    default_tbth = 180
    default_voxel = 39.0

# ── Sidebar: Morphometric targets ──
st.sidebar.header("Morphometric targets")
target_bvtv = st.sidebar.slider("Target BV/TV", 0.05, 0.50, default_bvtv, 0.01)
tbth_um = st.sidebar.slider("Target Tb.Th (um)", 80, 300, default_tbth, 5)
calibrate_tbth = st.sidebar.checkbox("Calibrate Tb.Th (iterative)", value=bool(real_targets),
    help="Iteratively adjusts base_sigma to match target Tb.Th. Slower but more accurate.")

# ── Sidebar: Volume geometry ──
st.sidebar.header("Volume geometry")
gcol1, gcol2 = st.sidebar.columns(2)
with gcol1:
    nx = st.selectbox("XY size (voxels)", [32, 48, 64, 96, 128], index=4)
with gcol2:
    nz = st.selectbox("Z slices", [16, 24, 32, 40, 60], index=3)
voxel_um = st.sidebar.number_input("Voxel size (um)", value=default_voxel, step=1.0)

# ── Sidebar: Field parameters ──
st.sidebar.header("Field parameters")
base_sigma = st.sidebar.slider("Base sigma", 1.0, 6.0, 2.5, 0.1,
    help="Spatial frequency of the Gaussian field. Higher = coarser plates.")
aniso_ratio = st.sidebar.slider("Anisotropy ratio", 0.5, 3.0, 1.0, 0.1,
    help="1.0 = isotropic. >1 elongates in z (vertebral-like).")

# ── Sidebar: Elastic warp ──
st.sidebar.header("Elastic warp")
warp_amp = st.sidebar.slider("Warp amplitude", 0.0, 3.0, 1.2, 0.1,
    help="Deformation strength. Higher = more natural irregularity.")
warp_sigma = st.sidebar.slider("Warp sigma", 3.0, 25.0, 12.0, 0.5,
    help="Correlation length of the deformation field.")

# ── Sidebar: Architecture ──
st.sidebar.header("Architecture")
plate_weight = st.sidebar.slider("Plate weight", 0.0, 1.0, 0.7, 0.05,
    help=">=0.5 uses plate path (zero-crossing walls). <0.5 uses rod+skeleton path.")
rod_weight = st.sidebar.slider("Rod weight", 0.0, 1.0, 0.3, 0.05)
if plate_weight + rod_weight > 0:
    st.sidebar.caption(f"Plate/rod ratio: {plate_weight/(plate_weight+rod_weight):.0%} / {rod_weight/(plate_weight+rod_weight):.0%}")

# ── Sidebar: Morphological cleanup ──
st.sidebar.header("Morphological cleanup")
proto_close_iters = st.sidebar.slider("Close iterations", 0, 6, 3, 1,
    help="Binary closing to connect gaps. More = thicker, more connected.")
min_component = st.sidebar.slider("Min component size", 0, 1000, 400, 50,
    help="Remove disconnected fragments smaller than this (voxels).")

# ── Sidebar: Grayscale synthesis ──
st.sidebar.header("Grayscale synthesis")
show_grayscale = st.sidebar.checkbox("Generate grayscale", value=True)
bone_mean = st.sidebar.slider("Bone mean intensity", 50, 200, 90, 5)
marrow_mean = st.sidebar.slider("Marrow mean intensity", 5, 50, 15, 1)
solid_fill_sigma = st.sidebar.slider("Solid fill sigma", 0.2, 2.0, 0.8, 0.1,
    help="Distance-based fill smoothness inside bone.")
noise_sd = st.sidebar.slider("Noise SD", 0.0, 10.0, 2.0, 0.5)
bg_tex_sd = st.sidebar.slider("Background texture SD", 0.0, 5.0, 0.5, 0.1)

# ── Sidebar: Seed ──
st.sidebar.header("Reproducibility")
seed = st.sidebar.number_input("Random seed", value=100, step=1)

# ── Size warning ──
total_voxels = nx * nx * nz
if total_voxels > 200_000:
    est_time = "~30-60s for generation, ~2-5min for FE"
elif total_voxels > 50_000:
    est_time = "~5-15s for generation, ~30-90s for FE"
else:
    est_time = "~1-3s for generation, ~10-30s for FE"

# ── Generate button ──
if st.sidebar.button(
    f"Generate {nx}x{nx}x{nz} volume",
    type="primary",
    use_container_width=True,
    help=f"{total_voxels:,} voxels — {est_time}",
):
    spinner_msg = f"Generating {nx}x{nx}x{nz} volume (BV/TV={target_bvtv:.2f})"
    if calibrate_tbth:
        spinner_msg += f", calibrating Tb.Th→{tbth_um} µm..."
    else:
        spinner_msg += "..."

    with st.spinner(spinner_msg):
        if calibrate_tbth:
            vol = generate_bone_volume_calibrated(
                nx=nx, ny=nx, nz=nz,
                target_bvtv=target_bvtv,
                target_tbth_um=float(tbth_um),
                voxel_um=voxel_um,
                base_sigma=base_sigma,
                warp_amp=warp_amp,
                warp_sigma=warp_sigma,
                plate_weight=plate_weight,
                close_iters=proto_close_iters,
                min_component=min_component,
                seed=int(seed),
                verbose=False,
            )
        else:
            vol = generate_bone_volume(
                nx=nx, ny=nx, nz=nz,
                target_bvtv=target_bvtv,
                voxel_um=voxel_um,
                base_sigma=base_sigma,
                warp_amp=warp_amp,
                warp_sigma=warp_sigma,
                plate_weight=plate_weight,
                close_iters=proto_close_iters,
                min_component=min_component,
                seed=int(seed),
                verbose=False,
            )

    st.session_state["bone_volume"] = vol
    st.session_state["gray_params"] = {
        "bone_mean": bone_mean,
        "marrow_mean": marrow_mean,
        "solid_fill_sigma": solid_fill_sigma,
        "noise_sd": noise_sd,
        "bg_tex_sd": bg_tex_sd,
    }

    bone_mask = vol["bone_mask"]
    morph = vol["morphometrics"]
    nz_a, ny_a, nx_a = bone_mask.shape
    mid_z = nz_a // 2
    voxel_mm = voxel_um / 1000.0

    # ── Morphometrics cards ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BV/TV", f"{morph['BVTV']:.3f}",
              f"{morph['BVTV'] - target_bvtv:+.3f} vs target")
    c2.metric("Tb.Th (p50)", f"{morph['TbTh_um_p50']:.0f} um")
    c3.metric("Tb.N", f"{morph['TbN_per_mm']:.2f} /mm")
    c4.metric("Tb.Sp (p50)", f"{morph['TbSp_um_p50']:.0f} um")
    c5.metric("LCC", f"{morph['lcc_frac']:.3f}",
              "connected" if morph['lcc_frac'] >= 0.99 else "fragmented")

    st.divider()

    # ── Slice viewer ──
    slice_idx = st.slider("Z-slice", 0, nz_a - 1, mid_z)

    # Generate grayscale if requested
    gray = None
    if show_grayscale:
        gray = generate_grayscale(
            bone_mask, seed=int(seed),
            bone_mean=bone_mean, marrow_mean=marrow_mean,
            solid_fill_sigma=solid_fill_sigma,
            noise_sd=noise_sd, bg_tex_sd=bg_tex_sd,
        )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("Binary mask")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(bone_mask[slice_idx].T, cmap='gray', origin='lower',
                  extent=[0, nx_a*voxel_mm, 0, ny_a*voxel_mm])
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title(f"z-slice {slice_idx}")
        st.pyplot(fig); plt.close()

    with col_b:
        if gray is not None:
            st.subheader("Synthetic micro-CT")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(gray[slice_idx].T, cmap='gray', origin='lower',
                      extent=[0, nx_a*voxel_mm, 0, ny_a*voxel_mm],
                      vmin=0, vmax=255)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            ax.set_title(f"Grayscale z-slice {slice_idx}")
            st.pyplot(fig); plt.close()
        else:
            st.info("Enable grayscale in sidebar")

    with col_c:
        st.subheader("Max intensity projection")
        fig, ax = plt.subplots(figsize=(5, 5))
        if gray is not None:
            mip = gray.max(axis=0)
            ax.imshow(mip.T, cmap='gray', origin='lower',
                      extent=[0, nx_a*voxel_mm, 0, ny_a*voxel_mm],
                      vmin=0, vmax=255)
        else:
            mip = bone_mask.max(axis=0)
            ax.imshow(mip.T, cmap='gray', origin='lower',
                      extent=[0, nx_a*voxel_mm, 0, ny_a*voxel_mm])
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title("MIP (z-axis)")
        st.pyplot(fig); plt.close()

    # ── Histogram ──
    if gray is not None:
        with st.expander("Intensity histogram"):
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(gray.ravel(), bins=128, color='#378ADD', alpha=0.8,
                    edgecolor='none', density=True)
            ax.axvline(marrow_mean, color='#993C1D', ls='--', label=f'Marrow mean={marrow_mean}')
            ax.axvline(bone_mean, color='#0F6E56', ls='--', label=f'Bone mean={bone_mean}')
            ax.set_xlabel("Intensity"); ax.set_ylabel("Density")
            ax.legend(); ax.set_xlim(0, 255)
            st.pyplot(fig); plt.close()

    # ── Full morphometrics ──
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

    with st.expander("Calibration details"):
        st.json(vol["calibration"])
        if "calibration_log" in vol:
            st.markdown("**Tb.Th calibration iterations:**")
            for step in vol["calibration_log"]:
                st.text(f"  σ={step['sigma']:.3f} → Tb.Th={step['tbth_um']:.1f} µm")

    st.success(
        f"Volume: {nx_a}x{ny_a}x{nz_a} | seed={vol['seed']} | "
        f"BV/TV={morph['BVTV']:.3f} (target {target_bvtv:.3f}) | "
        f"mode={'plate' if plate_weight >= 0.5 else 'rod'}"
    )

else:
    if "bone_volume" in st.session_state:
        st.info("Previous volume in session. Click 'Generate' for a new one, or go to FE solver.")
    else:
        st.info("Set parameters in the sidebar and click Generate. "
                "Published defaults: 128x128x40, BV/TV=0.33, plate_weight=0.7.")