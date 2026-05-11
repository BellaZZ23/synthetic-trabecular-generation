"""
Page 0: Data Loader
====================
Load real micro-CT data for:
  1. Parameter extraction — measure morphometrics and push them
     to the Generator page as targets.
  2. Validation — compare a synthetic volume against real data
     side-by-side.

Supports: TIFF stacks (.tif/.tiff), NIfTI (.nii/.nii.gz),
          NumPy arrays (.npy), and raw binary volumes.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys, io, tempfile, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fe_coupling"))
from step3_generator_fe_coupling import generate_grayscale

# ── Try to import the morphometric function from the v15 generator ──
try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from synthetic_trabecular_v15_morphometric_control import (
        measure_all_morphometrics,
        keep_largest_component,
    )
    HAS_MORPH = True
except ImportError:
    HAS_MORPH = False

st.set_page_config(page_title="Data loader", page_icon="📂", layout="wide")
st.title("Data loader")
st.caption("Load real micro-CT volumes for parameter extraction or validation")


# ══════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════

def load_tiff_stack(uploaded_files):
    """Load a list of uploaded TIFF files as a 3-D volume."""
    from PIL import Image
    slices = []
    for f in sorted(uploaded_files, key=lambda x: x.name):
        img = Image.open(f)
        slices.append(np.array(img))
    return np.stack(slices, axis=0)  # (Z, Y, X)


def load_nifti(uploaded_file):
    """Load a NIfTI (.nii / .nii.gz) file."""
    import nibabel as nib
    with tempfile.NamedTemporaryFile(suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()
        nii = nib.load(tmp.name)
        data = np.asarray(nii.dataobj)
    # Ensure (Z, Y, X) ordering — NIfTI is typically (X, Y, Z)
    if data.ndim == 3:
        data = np.transpose(data, (2, 1, 0))
    return data


def load_numpy(uploaded_file):
    """Load a .npy file."""
    return np.load(io.BytesIO(uploaded_file.read()))


def load_zip_tiffs(uploaded_file):
    """Load a ZIP archive containing TIFF slices."""
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
                slices.append(np.array(img))
    return np.stack(slices, axis=0)


def binarise(volume, threshold):
    """Threshold a grayscale volume to binary."""
    return (volume >= threshold).astype(np.uint8)


# ══════════════════════════════════════════════════════════════
# SIDEBAR — Upload
# ══════════════════════════════════════════════════════════════

st.sidebar.header("Upload real data")
file_format = st.sidebar.selectbox(
    "File format",
    ["TIFF stack (individual files)", "TIFF stack (ZIP)", "NIfTI (.nii/.nii.gz)", "NumPy (.npy)"],
)

uploaded = None
if file_format == "TIFF stack (individual files)":
    uploaded = st.sidebar.file_uploader(
        "Upload TIFF slices", type=["tif", "tiff"],
        accept_multiple_files=True,
        help="Select all slices. They will be sorted by filename.",
    )
elif file_format == "TIFF stack (ZIP)":
    uploaded = st.sidebar.file_uploader(
        "Upload ZIP of TIFF slices", type=["zip"],
    )
elif file_format == "NIfTI (.nii/.nii.gz)":
    uploaded = st.sidebar.file_uploader(
        "Upload NIfTI file", type=["nii", "gz"],
    )
elif file_format == "NumPy (.npy)":
    uploaded = st.sidebar.file_uploader(
        "Upload .npy array", type=["npy"],
        help="Expected shape: (Z, Y, X), uint8 or uint16 grayscale.",
    )

st.sidebar.header("Volume info")
voxel_um = st.sidebar.number_input("Voxel size (µm)", value=39.0, step=1.0,
    help="Must match the scan resolution for correct Tb.Th / Tb.Sp values.")

st.sidebar.header("Binarisation")
auto_threshold = st.sidebar.checkbox("Auto threshold (Otsu)", value=True)
manual_threshold = st.sidebar.slider("Manual threshold", 0, 255, 80, 1,
    help="Only used when auto-threshold is off.")

# ══════════════════════════════════════════════════════════════
# LOAD VOLUME
# ══════════════════════════════════════════════════════════════

volume = None

if uploaded:
    try:
        with st.spinner("Loading volume..."):
            if file_format == "TIFF stack (individual files)" and len(uploaded) > 0:
                volume = load_tiff_stack(uploaded)
            elif file_format == "TIFF stack (ZIP)":
                volume = load_zip_tiffs(uploaded)
            elif file_format == "NIfTI (.nii/.nii.gz)":
                volume = load_nifti(uploaded)
            elif file_format == "NumPy (.npy)":
                volume = load_numpy(uploaded)
    except Exception as e:
        st.error(f"Failed to load: {e}")
        volume = None

if volume is not None:
    # Normalise to 0-255 uint8 if needed
    if volume.dtype != np.uint8:
        vmin, vmax = volume.min(), volume.max()
        if vmax > vmin:
            volume = ((volume - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            volume = np.zeros_like(volume, dtype=np.uint8)

    nz, ny, nx = volume.shape
    st.success(f"Loaded volume: {nx}×{ny}×{nz} voxels, voxel size = {voxel_um:.1f} µm")

    # ── Threshold ──
    if auto_threshold:
        from skimage.filters import threshold_otsu
        thresh = int(threshold_otsu(volume))
        st.sidebar.info(f"Otsu threshold: {thresh}")
    else:
        thresh = manual_threshold

    bone_mask = binarise(volume, thresh)

    # Store in session
    st.session_state["real_volume"] = volume
    st.session_state["real_bone_mask"] = bone_mask
    st.session_state["real_voxel_um"] = voxel_um

    # ══════════════════════════════════════════════════════════
    # TABS — Extract / Validate
    # ══════════════════════════════════════════════════════════

    tab_extract, tab_validate = st.tabs([
        "📐 Parameter extraction",
        "✅ Validation",
    ])

    # ─────────────────────────────────────────────────────────
    # TAB 1 — Parameter extraction
    # ─────────────────────────────────────────────────────────
    with tab_extract:
        st.subheader("Extract morphometric parameters")
        st.write("Measure the real volume and use its morphometrics as generator targets.")

        mid_z = nz // 2
        slice_idx = st.slider("Preview Z-slice", 0, nz - 1, mid_z, key="extract_slice")
        voxel_mm = voxel_um / 1000.0

        col_raw, col_bin = st.columns(2)
        with col_raw:
            st.caption("Grayscale")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(volume[slice_idx].T, cmap='gray', origin='lower',
                      extent=[0, nx * voxel_mm, 0, ny * voxel_mm], vmin=0, vmax=255)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            ax.set_title(f"z-slice {slice_idx}")
            st.pyplot(fig); plt.close()

        with col_bin:
            st.caption(f"Binary mask (threshold={thresh})")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(bone_mask[slice_idx].T, cmap='gray', origin='lower',
                      extent=[0, nx * voxel_mm, 0, ny * voxel_mm])
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            ax.set_title(f"z-slice {slice_idx}")
            st.pyplot(fig); plt.close()

        # Histogram
        with st.expander("Intensity histogram"):
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(volume.ravel(), bins=128, color='#378ADD', alpha=0.8,
                    edgecolor='none', density=True)
            ax.axvline(thresh, color='red', ls='--', lw=2, label=f'Threshold={thresh}')
            ax.set_xlabel("Intensity"); ax.set_ylabel("Density")
            ax.legend(); ax.set_xlim(0, 255)
            st.pyplot(fig); plt.close()

        # Morphometric analysis
        if not HAS_MORPH:
            st.warning("Could not import `measure_all_morphometrics`. "
                       "Check that `synthetic_trabecular_v15_morphometric_control.py` "
                       "is on the Python path.")
        else:
            if st.button("Measure morphometrics", type="primary", key="btn_measure"):
                with st.spinner("Measuring..."):
                    morph = measure_all_morphometrics(bone_mask, voxel_um)

                st.session_state["real_morphometrics"] = morph

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("BV/TV", f"{morph['BVTV']:.3f}")
                c2.metric("Tb.Th (p50)", f"{morph['TbTh_um_p50']:.0f} µm")
                c3.metric("Tb.N", f"{morph['TbN_per_mm']:.2f} /mm")
                c4.metric("Tb.Sp (p50)", f"{morph['TbSp_um_p50']:.0f} µm")
                c5.metric("LCC", f"{morph['lcc_frac']:.3f}")

                with st.expander("Full measurements"):
                    mcol1, mcol2 = st.columns(2)
                    with mcol1:
                        st.json({
                            "BV/TV": round(morph["BVTV"], 4),
                            "Tb.Th p50 (µm)": round(morph["TbTh_um_p50"], 1),
                            "Tb.Th p90 (µm)": round(morph["TbTh_um_p90"], 1),
                            "Tb.N (/mm)": round(morph["TbN_per_mm"], 3),
                        })
                    with mcol2:
                        st.json({
                            "Tb.Sp p50 (µm)": round(morph["TbSp_um_p50"], 1),
                            "Tb.Sp p90 (µm)": round(morph["TbSp_um_p90"], 1),
                            "Euler number": morph["Euler"],
                            "LCC fraction": round(morph["lcc_frac"], 4),
                            "Components": morph["n_components"],
                        })

                # Push to generator
                st.divider()
                st.subheader("Use as generator targets")
                st.write(
                    "Click below to store these morphometrics in session state. "
                    "Then navigate to the **Bone Generator** page — the sidebar "
                    "sliders will show the matched values."
                )
                if st.button("Push to generator targets", key="btn_push"):
                    st.session_state["target_from_real"] = {
                        "bvtv": round(morph["BVTV"], 3),
                        "tbth_um": round(morph["TbTh_um_p50"], 0),
                        "voxel_um": voxel_um,
                        "nx": nx,
                        "ny": ny,
                        "nz": nz,
                    }
                    st.success(
                        f"Stored targets: BV/TV={morph['BVTV']:.3f}, "
                        f"Tb.Th={morph['TbTh_um_p50']:.0f} µm, "
                        f"volume={nx}×{ny}×{nz}. "
                        f"Go to **Bone Generator** to generate a matched volume."
                    )

    # ─────────────────────────────────────────────────────────
    # TAB 2 — Validation
    # ─────────────────────────────────────────────────────────
    with tab_validate:
        st.subheader("Validate synthetic vs real")
        st.write("Compare a generated synthetic volume against the loaded real data.")

        if "bone_volume" not in st.session_state:
            st.info("No synthetic volume in session yet. Go to **Bone Generator**, "
                    "generate a volume, then return here.")
        else:
            syn_vol = st.session_state["bone_volume"]
            syn_mask = syn_vol["bone_mask"]
            syn_morph = syn_vol["morphometrics"]
            nz_s, ny_s, nx_s = syn_mask.shape

            # Measure real if not done yet
            if "real_morphometrics" not in st.session_state:
                with st.spinner("Measuring real morphometrics..."):
                    real_morph = measure_all_morphometrics(bone_mask, voxel_um)
                    st.session_state["real_morphometrics"] = real_morph
            else:
                real_morph = st.session_state["real_morphometrics"]

            # ── Morphometric comparison table ──
            st.markdown("#### Morphometric comparison")
            metrics = [
                ("BV/TV", "BVTV", ".3f", ""),
                ("Tb.Th p50", "TbTh_um_p50", ".0f", " µm"),
                ("Tb.Th p90", "TbTh_um_p90", ".0f", " µm"),
                ("Tb.N", "TbN_per_mm", ".2f", " /mm"),
                ("Tb.Sp p50", "TbSp_um_p50", ".0f", " µm"),
                ("Tb.Sp p90", "TbSp_um_p90", ".0f", " µm"),
                ("Euler", "Euler", "d", ""),
                ("LCC frac", "lcc_frac", ".3f", ""),
                ("Components", "n_components", "d", ""),
            ]

            cols = st.columns([2, 2, 2, 2])
            cols[0].markdown("**Metric**")
            cols[1].markdown("**Real**")
            cols[2].markdown("**Synthetic**")
            cols[3].markdown("**Δ (%)**")

            for label, key, fmt, unit in metrics:
                rv = real_morph[key]
                sv = syn_morph[key]
                if isinstance(rv, (int, np.integer)):
                    delta_str = f"{sv - rv:+d}"
                elif rv != 0:
                    delta_str = f"{(sv - rv) / rv * 100:+.1f}%"
                else:
                    delta_str = "—"
                cols = st.columns([2, 2, 2, 2])
                cols[0].write(label)
                cols[1].write(f"{rv:{fmt}}{unit}")
                cols[2].write(f"{sv:{fmt}}{unit}")
                cols[3].write(delta_str)

            st.divider()

            # ── Side-by-side slice comparison ──
            st.markdown("#### Slice comparison")

            # Use the smaller volume's z-range for the slider
            max_z = min(nz, nz_s) - 1
            comp_slice = st.slider("Z-slice", 0, max_z, max_z // 2, key="val_slice")

            voxel_mm_r = voxel_um / 1000.0
            voxel_mm_s = syn_vol["voxel_um"] / 1000.0

            col_r, col_s, col_diff = st.columns(3)

            with col_r:
                st.caption("Real (binary)")
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(bone_mask[comp_slice].T, cmap='gray', origin='lower',
                          extent=[0, nx * voxel_mm_r, 0, ny * voxel_mm_r])
                ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                ax.set_title(f"Real z={comp_slice}")
                st.pyplot(fig); plt.close()

            with col_s:
                st.caption("Synthetic (binary)")
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(syn_mask[comp_slice].T, cmap='gray', origin='lower',
                          extent=[0, nx_s * voxel_mm_s, 0, ny_s * voxel_mm_s])
                ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                ax.set_title(f"Synthetic z={comp_slice}")
                st.pyplot(fig); plt.close()

            with col_diff:
                st.caption("BV/TV by slice")
                real_bvtv_z = [bone_mask[z].mean() for z in range(nz)]
                syn_bvtv_z = [syn_mask[z].mean() for z in range(nz_s)]
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.plot(real_bvtv_z, label="Real", color="#378ADD", lw=2)
                ax.plot(syn_bvtv_z, label="Synthetic", color="#E85D3A", lw=2, ls="--")
                ax.axhline(real_morph["BVTV"], color="#378ADD", lw=0.8, alpha=0.4)
                ax.axhline(syn_morph["BVTV"], color="#E85D3A", lw=0.8, alpha=0.4)
                ax.set_xlabel("Z-slice"); ax.set_ylabel("BV/TV")
                ax.set_title("BV/TV per slice"); ax.legend()
                st.pyplot(fig); plt.close()

            # ── Thickness distribution comparison ──
            with st.expander("Thickness distributions (if available)"):
                st.info(
                    "For a full Tb.Th distribution comparison, run the "
                    "distance-transform thickness measurement on both volumes. "
                    "This requires the full morphometric pipeline. The summary "
                    "percentiles (p50, p90) are compared in the table above."
                )

            st.divider()

            # ── Grayscale comparison ──
            if st.checkbox("Compare grayscale micro-CT", key="val_gray"):
                gray_syn = generate_grayscale(syn_mask, seed=syn_vol["seed"])
                col_rg, col_sg = st.columns(2)
                with col_rg:
                    st.caption("Real grayscale")
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(volume[comp_slice].T, cmap='gray', origin='lower',
                              extent=[0, nx * voxel_mm_r, 0, ny * voxel_mm_r],
                              vmin=0, vmax=255)
                    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                    st.pyplot(fig); plt.close()

                with col_sg:
                    st.caption("Synthetic grayscale")
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(gray_syn[comp_slice].T, cmap='gray', origin='lower',
                              extent=[0, nx_s * voxel_mm_s, 0, ny_s * voxel_mm_s],
                              vmin=0, vmax=255)
                    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                    st.pyplot(fig); plt.close()

                # Histogram overlay
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.hist(volume.ravel(), bins=128, alpha=0.5, density=True,
                        color="#378ADD", label="Real", edgecolor="none")
                ax.hist(gray_syn.ravel(), bins=128, alpha=0.5, density=True,
                        color="#E85D3A", label="Synthetic", edgecolor="none")
                ax.set_xlabel("Intensity"); ax.set_ylabel("Density")
                ax.legend(); ax.set_xlim(0, 255)
                ax.set_title("Intensity distribution overlay")
                st.pyplot(fig); plt.close()

else:
    st.info(
        "Upload a real micro-CT volume using the sidebar. "
        "Supported formats: TIFF stack, ZIP of TIFFs, NIfTI, or NumPy (.npy)."
    )