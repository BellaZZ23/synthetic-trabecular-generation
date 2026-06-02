"""
Page 1: Bone Volume Generator
==============================
Interactive controls for the v15.3 zero-crossing generator.
All parameters exposed with published honeycomb defaults.

Fixes vs previous version:
  - aniso_ratio and rod_weight now passed to generate_bone_volume()
  - pipeline_gray, pipeline_mask, pipeline_voxel_mm pushed to session
    after generation so FE solver (heterogeneous E) and pipeline page
    can use them without re-generating
  - real scan texture shown alongside synthetic for visual calibration
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fe_coupling"))
from step3_generator_fe_coupling import (
    generate_bone_volume,
    generate_bone_volume_calibrated,
    generate_grayscale,
)

st.set_page_config(page_title="Bone generator", page_icon="🦴", layout="wide")
st.title("Bone volume generator")
st.caption("v15.3 zero-crossing Gaussian random field — optimised trabecular defaults")

# ── Targets from Data Loader ──────────────────────────────────
real_targets = st.session_state.get("target_from_real")
if real_targets:
    st.info(
        f"📐 Targets from data loader: "
        f"BV/TV={real_targets['bvtv']:.3f}, "
        f"Tb.Th={real_targets['tbth_um']:.0f} µm, "
        f"volume={real_targets['nx']}×{real_targets['ny']}×{real_targets['nz']}"
    )
    default_bvtv  = real_targets["bvtv"]
    default_tbth  = int(real_targets["tbth_um"])
    default_voxel = real_targets["voxel_um"]
else:
    default_bvtv, default_tbth, default_voxel = 0.33, 180, 39.0


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

# ── Morphometric targets ──
st.sidebar.header("Morphometric targets")
target_bvtv    = st.sidebar.slider("Target BV/TV", 0.05, 0.50, default_bvtv, 0.01)
tbth_um        = st.sidebar.slider("Target Tb.Th (µm)", 80, 300, default_tbth, 5)
calibrate_tbth = st.sidebar.checkbox(
    "Calibrate Tb.Th (iterative)", value=bool(real_targets),
    help="Iteratively adjusts base_sigma to match target Tb.Th. Slower but more accurate.",
)

# ── Volume geometry ──
st.sidebar.header("Volume geometry")
gcol1, gcol2 = st.sidebar.columns(2)
with gcol1:
    nx = st.selectbox("XY size (voxels)", [32, 48, 64, 96, 128], index=4)
with gcol2:
    nz = st.selectbox("Z slices", [16, 24, 32, 40, 60], index=3)
voxel_um = st.sidebar.number_input("Voxel size (µm)", value=default_voxel, step=1.0)

# ── Field parameters ──
st.sidebar.header("Field parameters")
base_sigma  = st.sidebar.slider(
    "Base sigma", 1.0, 6.0, 2.2, 0.1,
    help="Spatial frequency of the Gaussian field. Higher = coarser plates.",
)
aniso_ratio = st.sidebar.slider(
    "Anisotropy ratio", 0.5, 3.0, 1.0, 0.1,
    help="1.0 = isotropic. >1 elongates in Z (vertebral-like).",
)

# ── Elastic warp ──
st.sidebar.header("Elastic warp")
warp_amp   = st.sidebar.slider(
    "Warp amplitude", 0.0, 3.0, 2.0, 0.1,
    help="Deformation strength. Higher = more natural irregularity.",
)
warp_sigma = st.sidebar.slider(
    "Warp sigma", 3.0, 25.0, 10.0, 0.5,
    help="Correlation length of the deformation field.",
)

# ── Architecture ──
st.sidebar.header("Architecture")
plate_weight = st.sidebar.slider(
    "Plate weight", 0.0, 1.0, 0.6, 0.05,
    help="≥0.5 uses plate path (zero-crossing walls). <0.5 uses rod+skeleton path.",
)
rod_weight = st.sidebar.slider(
    "Rod weight", 0.0, 1.0, 0.3, 0.05,
    help="Weight of rod-like structures in the mixed architecture.",
)
if plate_weight + rod_weight > 0:
    st.sidebar.caption(
        f"Plate/rod ratio: "
        f"{plate_weight/(plate_weight+rod_weight):.0%} / "
        f"{rod_weight/(plate_weight+rod_weight):.0%}"
    )

# ── Morphological cleanup ──
st.sidebar.header("Morphological cleanup")
proto_close_iters = st.sidebar.slider(
    "Close iterations", 0, 6, 1, 1,
    help="Binary closing to connect gaps. More = thicker, more connected.",
)
min_component = st.sidebar.slider(
    "Min component size", 0, 1000, 200, 50,
    help="Remove disconnected fragments smaller than this (voxels).",
)

# ── Grayscale synthesis ──
st.sidebar.header("Grayscale synthesis")
show_grayscale   = st.sidebar.checkbox("Generate grayscale", value=True)
bone_mean        = st.sidebar.slider("Bone mean intensity", 50, 200, 90, 5)
marrow_mean      = st.sidebar.slider("Marrow mean intensity", 5, 50, 15, 1)
solid_fill_sigma = st.sidebar.slider(
    "Solid fill sigma", 0.2, 2.0, 0.8, 0.1,
    help="Distance-based fill smoothness inside bone.",
)
noise_sd   = st.sidebar.slider("Noise SD", 0.0, 10.0, 2.0, 0.5)
bg_tex_sd  = st.sidebar.slider("Background texture SD", 0.0, 5.0, 0.5, 0.1)

# ── Reproducibility ──
st.sidebar.header("Reproducibility")
seed = st.sidebar.number_input("Random seed", value=100, step=1)

# ── Size estimate ──
total_voxels = nx * nx * nz
est_time = (
    "~30–60s generation, ~2–5min FE" if total_voxels > 200_000
    else "~5–15s generation, ~30–90s FE" if total_voxels > 50_000
    else "~1–3s generation, ~10–30s FE"
)


# ══════════════════════════════════════════════════════════════
# GENERATE
# ══════════════════════════════════════════════════════════════

if st.sidebar.button(
    f"Generate {nx}×{nx}×{nz} volume",
    type="primary",
    use_container_width=True,
    help=f"{total_voxels:,} voxels — {est_time}",
):
    spinner_msg = f"Generating {nx}×{nx}×{nz} (BV/TV={target_bvtv:.2f}"
    spinner_msg += f", Tb.Th→{tbth_um} µm..." if calibrate_tbth else ")..."

    with st.spinner(spinner_msg):
        shared_kwargs = dict(
            nx=nx, ny=nx, nz=nz,
            voxel_um=voxel_um,
            base_sigma=base_sigma,
            aniso_ratio=aniso_ratio,       # ← fixed: was missing
            warp_amp=warp_amp,
            warp_sigma=warp_sigma,
            plate_weight=plate_weight,
            rod_weight=rod_weight,         # ← fixed: was missing
            close_iters=proto_close_iters,
            min_component=min_component,
            seed=int(seed),
            verbose=False,
        )

        if calibrate_tbth:
            vol = generate_bone_volume_calibrated(
                target_bvtv=target_bvtv,
                target_tbth_um=float(tbth_um),
                **shared_kwargs,
            )
        else:
            vol = generate_bone_volume(
                target_bvtv=target_bvtv,
                **shared_kwargs,
            )

    bone_mask = vol["bone_mask"]
    morph     = vol["morphometrics"]
    nz_a, ny_a, nx_a = bone_mask.shape
    voxel_mm  = voxel_um / 1000.0

    # Generate grayscale
    gray = None
    if show_grayscale:
        with st.spinner("Generating grayscale..."):
            gray = generate_grayscale(
                bone_mask, seed=int(seed),
                bone_mean=bone_mean, marrow_mean=marrow_mean,
                solid_fill_sigma=solid_fill_sigma,
                noise_sd=noise_sd, bg_tex_sd=bg_tex_sd,
            )

    # ── Push to session ──────────────────────────────────────
    st.session_state["bone_volume"]       = vol
    st.session_state["pipeline_mask"]     = bone_mask          # ← new
    st.session_state["pipeline_voxel_mm"] = voxel_mm           # ← new
    st.session_state["gray_params"] = {
        "bone_mean": bone_mean, "marrow_mean": marrow_mean,
        "solid_fill_sigma": solid_fill_sigma,
        "noise_sd": noise_sd, "bg_tex_sd": bg_tex_sd,
    }
    if gray is not None:
        st.session_state["pipeline_gray"] = gray                # ← new


    # ══════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════

    # Morphometrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BV/TV",     f"{morph['BVTV']:.3f}",
              f"{morph['BVTV']-target_bvtv:+.3f} vs target")
    c2.metric("Tb.Th p50", f"{morph['TbTh_um_p50']:.0f} µm")
    c3.metric("Tb.N",      f"{morph['TbN_per_mm']:.2f} /mm")
    c4.metric("Tb.Sp p50", f"{morph['TbSp_um_p50']:.0f} µm")
    c5.metric("LCC",       f"{morph['lcc_frac']:.3f}",
              "connected" if morph['lcc_frac'] >= 0.99 else "fragmented")

    st.divider()

    # Slice viewer
    mid_z     = nz_a // 2
    slice_idx = st.slider("Z-slice", 0, nz_a - 1, mid_z)
    extent    = [0, nx_a*voxel_mm, 0, ny_a*voxel_mm]

    # Determine columns: 3 synthetic + optional real reference
    has_real_scan = "real_volume" in st.session_state
    n_cols        = 4 if has_real_scan else 3
    cols          = st.columns(n_cols)

    # Col 0: binary mask
    with cols[0]:
        st.subheader("Binary mask")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(bone_mask[slice_idx].T, cmap='gray',
                  origin='lower', extent=extent)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title(f"z-slice {slice_idx}")
        st.pyplot(fig); plt.close()

    # Col 1: synthetic grayscale
    with cols[1]:
        if gray is not None:
            st.subheader("Synthetic µCT")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(gray[slice_idx].T, cmap='gray', origin='lower',
                      extent=extent, vmin=0, vmax=255)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            ax.set_title(f"Grayscale z-slice {slice_idx}")
            st.pyplot(fig); plt.close()
        else:
            st.info("Enable grayscale synthesis in sidebar.")

    # Col 2: max intensity projection
    with cols[2]:
        st.subheader("Max intensity projection")
        fig, ax = plt.subplots(figsize=(5, 5))
        mip_src = gray if gray is not None else bone_mask
        mip     = mip_src.max(axis=0)
        cmap_mip = 'gray'
        vmin_mip = (0, 255) if gray is not None else (None, None)
        kwargs   = dict(cmap=cmap_mip, origin='lower', extent=extent)
        if gray is not None:
            kwargs.update(vmin=0, vmax=255)
        ax.imshow(mip.T, **kwargs)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title("MIP (z-axis)")
        st.pyplot(fig); plt.close()

    # Col 3 (optional): real scan reference for visual comparison
    if has_real_scan:
        real_vol    = st.session_state["real_volume"]
        real_vox_um = st.session_state.get("real_voxel_um", voxel_um)
        real_vox_mm = real_vox_um / 1000.0
        nz_r, ny_r, nx_r = real_vol.shape
        real_z   = min(slice_idx, nz_r - 1)
        ext_real = [0, nx_r*real_vox_mm, 0, ny_r*real_vox_mm]

        with cols[3]:
            st.subheader("Real µCT (reference)")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(real_vol[real_z].T, cmap='gray', origin='lower',
                      extent=ext_real, vmin=0, vmax=255)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            ax.set_title(f"Real z-slice {real_z}")
            st.pyplot(fig); plt.close()

        # Intensity distribution comparison
        if gray is not None:
            with st.expander("Intensity comparison: synthetic vs real"):
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.hist(gray.ravel(), bins=128, alpha=0.5, density=True,
                        color='#378ADD', label='Synthetic', edgecolor='none')
                ax.hist(real_vol.ravel(), bins=128, alpha=0.5, density=True,
                        color='#E85D3A', label='Real', edgecolor='none')
                ax.axvline(marrow_mean, color='#993C1D', ls='--',
                           label=f'Marrow target={marrow_mean}')
                ax.axvline(bone_mean, color='#0F6E56', ls='--',
                           label=f'Bone target={bone_mean}')
                ax.set_xlabel("Intensity"); ax.set_ylabel("Density")
                ax.legend(); ax.set_xlim(0, 255)
                st.pyplot(fig); plt.close()

    # Intensity histogram (synthetic only)
    if gray is not None and not has_real_scan:
        with st.expander("Intensity histogram"):
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(gray.ravel(), bins=128, color='#378ADD', alpha=0.8,
                    edgecolor='none', density=True)
            ax.axvline(marrow_mean, color='#993C1D', ls='--',
                       label=f'Marrow mean={marrow_mean}')
            ax.axvline(bone_mean, color='#0F6E56', ls='--',
                       label=f'Bone mean={bone_mean}')
            ax.set_xlabel("Intensity"); ax.set_ylabel("Density")
            ax.legend(); ax.set_xlim(0, 255)
            st.pyplot(fig); plt.close()

    # Full morphometrics
    with st.expander("Full morphometric measurements"):
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.json({
                "BV/TV":          round(morph["BVTV"], 4),
                "Tb.Th p50 (µm)": round(morph["TbTh_um_p50"], 1),
                "Tb.Th p90 (µm)": round(morph["TbTh_um_p90"], 1),
                "Tb.N (/mm)":     round(morph["TbN_per_mm"], 3),
            })
        with mcol2:
            st.json({
                "Tb.Sp p50 (µm)": round(morph["TbSp_um_p50"], 1),
                "Tb.Sp p90 (µm)": round(morph["TbSp_um_p90"], 1),
                "Euler number":   morph["Euler"],
                "LCC fraction":   round(morph["lcc_frac"], 4),
                "Components":     morph["n_components"],
            })

    # Calibration log
    with st.expander("Calibration details"):
        st.json(vol["calibration"])
        if "calibration_log" in vol:
            st.markdown("**Tb.Th calibration iterations:**")
            for step in vol["calibration_log"]:
                st.text(f"  σ={step['sigma']:.3f} → Tb.Th={step['tbth_um']:.1f} µm")

    # Parameters used (useful for reproducibility)
    with st.expander("Parameters used"):
        st.json({
            "nx": nx_a, "ny": ny_a, "nz": nz_a,
            "voxel_um": voxel_um,
            "target_bvtv": target_bvtv,
            "target_tbth_um": tbth_um if calibrate_tbth else None,
            "base_sigma": base_sigma,
            "aniso_ratio": aniso_ratio,
            "warp_amp": warp_amp,
            "warp_sigma": warp_sigma,
            "plate_weight": plate_weight,
            "rod_weight": rod_weight,
            "close_iters": proto_close_iters,
            "min_component": min_component,
            "seed": int(seed),
            "calibrated": calibrate_tbth,
        })

    st.success(
        f"Volume: {nx_a}×{ny_a}×{nz_a} | seed={vol['seed']} | "
        f"BV/TV={morph['BVTV']:.3f} (target {target_bvtv:.3f}) | "
        f"mode={'plate' if plate_weight >= 0.5 else 'rod'} | "
        f"aniso={aniso_ratio:.1f} | "
        f"{'pipeline_gray ✓' if gray is not None else 'no grayscale'}"
    )

else:
    if "bone_volume" in st.session_state:
        vol   = st.session_state["bone_volume"]
        morph = vol["morphometrics"]
        st.info(
            f"Previous volume in session: "
            f"{vol['bone_mask'].shape[2]}×{vol['bone_mask'].shape[1]}×{vol['bone_mask'].shape[0]} | "
            f"BV/TV={morph['BVTV']:.3f} | "
            f"Tb.Th={morph['TbTh_um_p50']:.0f} µm | "
            f"{'grayscale ready ✓' if 'pipeline_gray' in st.session_state else 'no grayscale'}"
        )
        st.caption("Click **Generate** for a new volume, or go to FE solver / Pipeline.")
    else:
        st.info(
            "Set parameters in the sidebar and click **Generate**. "
            "Defaults: 128×128×40, BV/TV=0.33, sigma=2.2, warp=2.0, close=1."
        )