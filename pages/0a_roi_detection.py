"""
Page 0a: ROI Detection & Preprocessing
========================================
Runs BEFORE registration and morphometric measurement.

Purpose: the D²IM masks cover the entire vertebral body including the
cortical shell. Measuring trabecular morphometrics on the full mask
gives meaningless results (Tb.Th ~1000 µm, Tb.N ~0.2/mm) because the
cortical shell dominates.

This page:
  1. Loads the scan + full mask from session (set in Data Loader)
  2. Detects the trabecular ROI using distance transform on the mask
     — erodes by the cortical shell thickness to expose the interior
  3. Optionally uses scan intensity to refine the cortical boundary
  4. Saves the trabecular ROI mask back to session as
     real_bone_mask_trabecular for use in registration + morphometrics
  5. Shows before/after morphometrics so you can tune the erosion depth

Run this page after uploading data in the Data Loader, before
running registration or measuring morphometrics.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from scipy.ndimage import (
    binary_erosion, binary_fill_holes,
    distance_transform_edt, gaussian_filter, label
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
try:
    from synthetic_trabecular_v15_morphometric_control import measure_all_morphometrics
    HAS_MORPH = True
except ImportError:
    HAS_MORPH = False

st.set_page_config(
    page_title="ROI detection", page_icon="🔍", layout="wide"
)
st.title("ROI detection & preprocessing")
st.caption(
    "Extract the trabecular compartment from the full bone mask "
    "before registration and morphometric measurement."
)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def detect_trabecular_roi(mask: np.ndarray,
                          voxel_um: float,
                          erosion_um: float = 2000.0,
                          smooth_um: float = 0.0) -> np.ndarray:
    """
    Extract trabecular ROI by eroding the full bone mask by the
    estimated cortical shell thickness.

    Parameters
    ----------
    mask        : (nz, ny, nx) uint8 full bone mask
    voxel_um    : voxel size in µm
    erosion_um  : cortical shell thickness to remove in µm
                  (default 2000 µm = 2 mm — typical vertebral cortex)
    smooth_um   : optional Gaussian smoothing before erosion (µm)

    Returns
    -------
    roi_mask : (nz, ny, nx) uint8 trabecular interior mask
    """
    erosion_vox = max(1, int(round(erosion_um / voxel_um)))

    work = mask.astype(bool)

    if smooth_um > 0:
        sigma = smooth_um / voxel_um
        work = gaussian_filter(work.astype(float), sigma=sigma) > 0.5

    # Erode slice-by-slice for speed on thin volumes
    if work.ndim == 3 and work.shape[0] <= 20:
        roi = np.zeros_like(work)
        for z in range(work.shape[0]):
            if work[z].any():
                roi[z] = binary_erosion(
                    work[z], iterations=erosion_vox
                )
    else:
        roi = binary_erosion(work, iterations=erosion_vox)

    return roi.astype(np.uint8)


def detect_trabecular_roi_scan(scan: np.ndarray,
                                mask: np.ndarray,
                                voxel_um: float,
                                cortical_percentile: float = 85.0,
                                erosion_um: float = 500.0) -> np.ndarray:
    """
    Use scan intensity to detect cortical shell, then extract interior.

    High-intensity voxels in the scan = cortical bone.
    Fill holes in the cortical shell to get the full bone envelope,
    then erode slightly to get the trabecular interior.

    Parameters
    ----------
    scan               : (nz, ny, nx) uint8 grayscale scan
    mask               : (nz, ny, nx) uint8 full bone mask
    voxel_um           : voxel size in µm
    cortical_percentile: intensity percentile to define cortical shell
    erosion_um         : final erosion after fill (µm)
    """
    erosion_vox = max(1, int(round(erosion_um / voxel_um)))

    roi = np.zeros_like(mask)
    bone_voxels = scan[mask > 0]
    thresh = np.percentile(bone_voxels, cortical_percentile)

    for z in range(scan.shape[0]):
        sl = scan[z]
        mk = mask[z]
        if not mk.any():
            continue

        # High intensity within mask = cortical
        cortical = (sl > thresh) & (mk > 0)

        # Fill holes to get full bone envelope
        filled = binary_fill_holes(cortical | mk.astype(bool))

        # Remove the cortical shell itself
        interior = filled & ~cortical

        # Small erosion to clean edges
        if interior.any() and erosion_vox > 0:
            interior = binary_erosion(interior, iterations=erosion_vox)

        roi[z] = interior.astype(np.uint8)

    return roi


def measure_roi(mask, voxel_um):
    if not HAS_MORPH or mask.sum() < 100:
        return None
    try:
        return measure_all_morphometrics(mask, voxel_um)
    except Exception as e:
        return None


# ══════════════════════════════════════════════════════════════
# CHECK SESSION
# ══════════════════════════════════════════════════════════════

has_scan = "real_volume"    in st.session_state
has_mask = "real_bone_mask" in st.session_state

if not has_scan or not has_mask:
    st.warning(
        "No scan or mask in session. "
        "Go to **Data Loader** and upload your µCT data first."
    )
    st.stop()

scan     = st.session_state["real_volume"]
mask     = st.session_state["real_bone_mask"]
voxel_um = st.session_state.get("real_voxel_um", 50.0)
voxel_mm = voxel_um / 1000.0
nz, ny, nx = scan.shape

st.success(
    f"Scan: {nx}×{ny}×{nz}, voxel={voxel_um:.0f} µm, "
    f"Full mask BV/TV={mask.mean():.3f}"
)


# ══════════════════════════════════════════════════════════════
# SIDEBAR — ROI settings
# ══════════════════════════════════════════════════════════════

st.sidebar.header("ROI detection method")
method = st.sidebar.radio(
    "Method",
    ["Distance transform (mask erosion)", "Intensity-based (scan + mask)"],
    help=(
        "Distance transform: erodes the full mask by cortical thickness. "
        "Simple and robust for most specimens.\n\n"
        "Intensity-based: uses scan grey levels to detect the cortical "
        "shell boundary. More accurate but needs tuning."
    ),
)

st.sidebar.header("Parameters")

if method == "Distance transform (mask erosion)":
    erosion_um = st.sidebar.slider(
        "Cortical erosion depth (µm)", 500, 5000,
        int(round(voxel_um * 40)),   # default: 40 voxels
        step=int(voxel_um),
        help=(
            f"Thickness of cortical shell to remove. "
            f"At {voxel_um:.0f} µm voxel size, 1 voxel = {voxel_um:.0f} µm. "
            f"Typical vertebral cortex: 1–3 mm (1000–3000 µm)."
        ),
    )
    erosion_vox = int(erosion_um / voxel_um)
    st.sidebar.caption(f"= {erosion_vox} voxels at {voxel_um:.0f} µm")
    cortical_pct = 85.0
else:
    erosion_um = st.sidebar.slider(
        "Final erosion after fill (µm)", 0, 1000,
        int(voxel_um * 5), step=int(voxel_um),
    )
    cortical_pct = st.sidebar.slider(
        "Cortical intensity percentile", 50, 99, 85, 1,
        help="Voxels above this percentile = cortical shell.",
    )

st.sidebar.header("Preview")
mid_z    = nz // 2
prev_z   = st.sidebar.slider("Preview Z-slice", 0, nz-1, mid_z)
show_3d  = st.sidebar.checkbox("Measure morphometrics after detection", value=True)


# ══════════════════════════════════════════════════════════════
# DISTANCE TRANSFORM VISUALISATION
# ══════════════════════════════════════════════════════════════

with st.expander("Distance transform analysis", expanded=True):
    st.write(
        "The distance transform shows how far each bone voxel is from "
        "the surface. The cortical shell = low distance values (near surface). "
        "The trabecular core = high distance values (deep interior)."
    )

    dist_2d = distance_transform_edt(mask[prev_z])
    dist_max = dist_2d.max()

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(scan[prev_z].T, cmap='gray', origin='lower',
                       extent=[0, nx*voxel_mm, 0, ny*voxel_mm])
        axes[0].set_title("Scan"); axes[0].set_xlabel("x [mm]")

        axes[1].imshow(mask[prev_z].T, cmap='gray', origin='lower',
                       extent=[0, nx*voxel_mm, 0, ny*voxel_mm])
        axes[1].set_title(f"Full mask (BV/TV={mask[prev_z].mean():.3f})")
        axes[1].set_xlabel("x [mm]")

        im = axes[2].imshow(dist_2d.T, cmap='hot', origin='lower',
                            extent=[0, nx*voxel_mm, 0, ny*voxel_mm])
        axes[2].set_title(f"Distance transform (max={dist_max:.0f} vox = {dist_max*voxel_mm:.1f} mm)")
        axes[2].set_xlabel("x [mm]")
        plt.colorbar(im, ax=axes[2], label='Distance (voxels)')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with dcol2:
        # Histogram of distance values — shows cortical/trabecular split
        dist_vals = dist_2d[mask[prev_z] > 0]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(dist_vals, bins=60, color='#E85D3A', alpha=0.8,
                edgecolor='none', density=True)
        if method == "Distance transform (mask erosion)":
            ax.axvline(erosion_um/voxel_um, color='#378ADD', lw=2, ls='--',
                       label=f'Erosion threshold ({erosion_um:.0f} µm)')
            ax.legend()
        ax.set_xlabel("Distance from surface (voxels)")
        ax.set_ylabel("Density")
        ax.set_title("Distance histogram — peak left = cortex, right = trabeculae")
        st.pyplot(fig); plt.close()
        st.caption(
            f"Max distance: {dist_max:.0f} voxels = {dist_max*voxel_mm:.1f} mm. "
            f"Set erosion depth to where the histogram has a trough "
            f"between the cortical peak and the trabecular distribution."
        )


# ══════════════════════════════════════════════════════════════
# DETECT ROI
# ══════════════════════════════════════════════════════════════

if st.button("▶ Detect trabecular ROI", type="primary",
             use_container_width=True):
    with st.spinner("Detecting trabecular ROI..."):
        if method == "Distance transform (mask erosion)":
            roi_mask = detect_trabecular_roi(
                mask, voxel_um, erosion_um=erosion_um
            )
        else:
            roi_mask = detect_trabecular_roi_scan(
                scan, mask, voxel_um,
                cortical_percentile=cortical_pct,
                erosion_um=erosion_um,
            )

    if roi_mask.sum() < 100:
        st.error(
            "ROI is empty after erosion — cortical erosion depth is too large. "
            "Reduce the erosion depth and try again."
        )
        st.stop()

    st.session_state["real_bone_mask_trabecular"] = roi_mask
    st.session_state["roi_erosion_um"] = erosion_um
    st.session_state["roi_method"] = method

    bvtv_full = mask.mean()
    bvtv_roi  = roi_mask.mean()
    st.success(
        f"ROI detected — "
        f"shape: {roi_mask.shape} | "
        f"full BV/TV: {bvtv_full:.3f} → ROI BV/TV: {bvtv_roi:.3f} | "
        f"erosion: {erosion_um:.0f} µm ({int(erosion_um/voxel_um)} voxels)"
    )

# ── Display ROI if available ──
if "real_bone_mask_trabecular" in st.session_state:
    roi_mask  = st.session_state["real_bone_mask_trabecular"]
    erosion_v = int(st.session_state.get("roi_erosion_um", 0) / voxel_um)

    st.divider()
    st.subheader("ROI result")

    # Side-by-side comparison
    rcol1, rcol2, rcol3 = st.columns(3)
    ext = [0, nx*voxel_mm, 0, ny*voxel_mm]

    with rcol1:
        st.caption("Full bone mask")
        fig, ax = plt.subplots(figsize=(5, 6))
        ax.imshow(mask[prev_z].T, cmap='gray', origin='lower', extent=ext)
        ax.set_title(f"BV/TV={mask[prev_z].mean():.3f}")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        st.pyplot(fig); plt.close()

    with rcol2:
        st.caption("Trabecular ROI")
        fig, ax = plt.subplots(figsize=(5, 6))
        ax.imshow(roi_mask[prev_z].T, cmap='gray', origin='lower', extent=ext)
        ax.set_title(f"BV/TV={roi_mask[prev_z].mean():.3f}")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        st.pyplot(fig); plt.close()

    with rcol3:
        st.caption("Scan + ROI overlay")
        fig, ax = plt.subplots(figsize=(5, 6))
        ax.imshow(scan[prev_z].T, cmap='gray', origin='lower',
                  extent=ext, vmin=0, vmax=255, alpha=0.7)
        ax.imshow(
            np.ma.masked_where(roi_mask[prev_z].T == 0, roi_mask[prev_z].T),
            cmap='autumn', origin='lower', extent=ext, alpha=0.5
        )
        ax.set_title("Scan + trabecular ROI (orange)")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        st.pyplot(fig); plt.close()

    # Morphometrics comparison
    if show_3d and HAS_MORPH:
        st.divider()
        st.subheader("Morphometrics: before vs after ROI")

        mcol1, mcol2 = st.columns(2)

        with mcol1:
            st.markdown("**Full mask**")
            with st.spinner("Measuring full mask..."):
                morph_full = measure_roi(mask, voxel_um)
            if morph_full:
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("BV/TV",    f"{morph_full['BVTV']:.3f}")
                m2.metric("Tb.Th p50",f"{morph_full['TbTh_um_p50']:.0f} µm")
                m3.metric("Tb.N",     f"{morph_full['TbN_per_mm']:.2f} /mm")
                m4.metric("Tb.Sp p50",f"{morph_full['TbSp_um_p50']:.0f} µm")
                if morph_full['TbTh_um_p50'] > 500:
                    st.warning("Tb.Th > 500 µm — cortical shell still dominant.")

        with mcol2:
            st.markdown("**Trabecular ROI**")
            with st.spinner("Measuring ROI..."):
                morph_roi = measure_roi(roi_mask, voxel_um)
            if morph_roi:
                r1,r2,r3,r4 = st.columns(4)
                r1.metric("BV/TV",    f"{morph_roi['BVTV']:.3f}")
                r2.metric("Tb.Th p50",f"{morph_roi['TbTh_um_p50']:.0f} µm")
                r3.metric("Tb.N",     f"{morph_roi['TbN_per_mm']:.2f} /mm")
                r4.metric("Tb.Sp p50",f"{morph_roi['TbSp_um_p50']:.0f} µm")

                # Quality check
                tbth = morph_roi['TbTh_um_p50']
                tbn  = morph_roi['TbN_per_mm']
                if 100 <= tbth <= 400 and 0.5 <= tbn <= 5.0:
                    st.success(
                        "Morphometrics look like trabecular bone. "
                        "Ready to use as generator targets."
                    )
                elif tbth > 400:
                    st.warning(
                        f"Tb.Th still high ({tbth:.0f} µm). "
                        "Increase erosion depth by 500–1000 µm and re-run."
                    )
                else:
                    st.warning(
                        f"Tb.Th very low ({tbth:.0f} µm). "
                        "Erosion may be too deep — reduce by 500 µm."
                    )

                # Push to session as generator targets
                if 100 <= tbth <= 400:
                    targets = {
                        "bvtv":    round(morph_roi["BVTV"], 3),
                        "tbth_um": round(morph_roi["TbTh_um_p50"], 0),
                        "voxel_um": voxel_um,
                        "nx": nx, "ny": ny, "nz": nz,
                    }
                    st.session_state["target_from_real"]   = targets
                    st.session_state["real_morphometrics"] = morph_roi

    # ── Use ROI as main mask ──
    st.divider()
    st.info(
        "The trabecular ROI is now stored as `real_bone_mask_trabecular` in session. "
        "It will be used automatically for registration and morphometrics "
        "on the Data Loader and Pipeline pages."
    )

    ucol1, ucol2 = st.columns(2)
    with ucol1:
        if st.button(
            "✅ Use ROI as main bone mask",
            type="primary", use_container_width=True,
        ):
            st.session_state["real_bone_mask"] = roi_mask
            st.success(
                "ROI set as main bone mask. "
                "Return to Data Loader or Pipeline to continue."
            )

    with ucol2:
        if st.button(
            "🔄 Try different erosion depth",
            use_container_width=True,
        ):
            if "real_bone_mask_trabecular" in st.session_state:
                del st.session_state["real_bone_mask_trabecular"]
            st.rerun()