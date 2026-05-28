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
from step3_generator_fe_coupling import (
    generate_grayscale,
    generate_bone_volume,
    generate_bone_volume_calibrated,
)

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

# ── Try to import BoneJ-equivalent measurements ──
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fe_coupling"))
    from bonej_measurements import measure_all_bonej
    HAS_BONEJ = True
except ImportError:
    HAS_BONEJ = False

st.set_page_config(page_title="Data loader", page_icon="📂", layout="wide")
st.title("Data loader")
st.caption("Load real micro-CT volumes for parameter extraction or validation")


# ══════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════

def load_tiff_stack(uploaded_files):
    """Load uploaded TIFF files as a 3-D volume.
    Handles both individual-slice TIFFs and multi-frame TIFFs."""
    from PIL import Image
    slices = []
    for f in sorted(uploaded_files, key=lambda x: x.name):
        img = Image.open(f)
        # Check for multi-frame TIFF (e.g. ImageJ stacks)
        n_frames = getattr(img, 'n_frames', 1)
        if n_frames > 1:
            for i in range(n_frames):
                img.seek(i)
                slices.append(np.array(img))
        else:
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


def run_batch_generation(targets, n_samples, base_sigma, close_iters, base_seed,
                         calibrate_tbth, bone_mean=90.0, marrow_mean=15.0,
                         solid_fill_sigma=0.8, noise_sd=2.0, bg_tex_sd=0.5,
                         progress_bar=None):
    """Generate multiple synthetic volumes from a set of targets.

    Returns list of dicts, each with 'volume', 'grayscale', 'seed'.
    """
    samples = []
    for i in range(n_samples):
        seed = base_seed + i
        if progress_bar:
            progress_bar.progress(
                (i) / n_samples,
                text=f"Generating sample {i+1}/{n_samples} (seed={seed})..."
            )

        if calibrate_tbth:
            vol = generate_bone_volume_calibrated(
                nx=targets["nx"], ny=targets["ny"], nz=targets["nz"],
                target_bvtv=targets["bvtv"],
                target_tbth_um=float(targets["tbth_um"]),
                voxel_um=targets["voxel_um"],
                base_sigma=base_sigma,
                close_iters=close_iters,
                seed=seed, verbose=False,
            )
        else:
            vol = generate_bone_volume(
                nx=targets["nx"], ny=targets["ny"], nz=targets["nz"],
                target_bvtv=targets["bvtv"],
                voxel_um=targets["voxel_um"],
                base_sigma=base_sigma,
                close_iters=close_iters,
                seed=seed, verbose=False,
            )

        gray = generate_grayscale(
            vol["bone_mask"], seed=seed,
            bone_mean=bone_mean, marrow_mean=marrow_mean,
            solid_fill_sigma=solid_fill_sigma,
            noise_sd=noise_sd, bg_tex_sd=bg_tex_sd,
        )
        samples.append({"volume": vol, "grayscale": gray, "seed": seed})

    if progress_bar:
        progress_bar.progress(1.0, text="Done!")
    return samples


def display_sample_gallery(samples, voxel_um):
    """Display a gallery of generated samples with key metrics."""
    voxel_mm = voxel_um / 1000.0
    n = len(samples)

    # Summary table
    st.markdown("#### Sample summary")
    hcols = st.columns([1, 1, 1, 1, 1, 1])
    hcols[0].markdown("**#**")
    hcols[1].markdown("**Seed**")
    hcols[2].markdown("**BV/TV**")
    hcols[3].markdown("**Tb.Th (µm)**")
    hcols[4].markdown("**Tb.N (/mm)**")
    hcols[5].markdown("**LCC**")

    for i, s in enumerate(samples):
        m = s["volume"]["morphometrics"]
        cols = st.columns([1, 1, 1, 1, 1, 1])
        cols[0].write(f"{i+1}")
        cols[1].write(f"{s['seed']}")
        cols[2].write(f"{m['BVTV']:.3f}")
        cols[3].write(f"{m['TbTh_um_p50']:.0f}")
        cols[4].write(f"{m['TbN_per_mm']:.2f}")
        cols[5].write(f"{m['lcc_frac']:.3f}")

    # Visual gallery — show up to 4 per row
    st.markdown("#### Slice gallery")
    per_row = min(n, 4)
    for row_start in range(0, n, per_row):
        row_samples = samples[row_start:row_start + per_row]
        img_cols = st.columns(len(row_samples))
        for j, s in enumerate(row_samples):
            mask = s["volume"]["bone_mask"]
            gray = s["grayscale"]
            nz_s, ny_s, nx_s = mask.shape
            mid = nz_s // 2
            ext = [0, nx_s * voxel_mm, 0, ny_s * voxel_mm]

            with img_cols[j]:
                st.caption(f"Sample {row_start + j + 1} (seed={s['seed']})")
                fig, axes = plt.subplots(1, 2, figsize=(6, 3))
                axes[0].imshow(mask[mid].T, cmap='gray', origin='lower', extent=ext)
                axes[0].set_title("Binary", fontsize=9)
                axes[0].set_xlabel("x [mm]", fontsize=7)
                axes[1].imshow(gray[mid].T, cmap='gray', origin='lower', extent=ext,
                               vmin=0, vmax=255)
                axes[1].set_title("Grayscale", fontsize=9)
                axes[1].set_xlabel("x [mm]", fontsize=7)
                plt.tight_layout()
                st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════
# SIDEBAR — Input mode
# ══════════════════════════════════════════════════════════════

input_mode = st.sidebar.radio(
    "Input mode",
    ["Upload micro-CT scan", "Enter metrics manually"],
    help="Upload a scan to extract parameters, or type in known values directly.",
)

uploaded = None
voxel_um = 39.0

if input_mode == "Upload micro-CT scan":
    st.sidebar.header("Upload real data")
    file_format = st.sidebar.selectbox(
        "File format",
        ["TIFF stack (individual files)", "TIFF stack (ZIP)", "NIfTI (.nii/.nii.gz)", "NumPy (.npy)"],
    )

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
# MODE: MANUAL METRICS
# ══════════════════════════════════════════════════════════════

if input_mode == "Enter metrics manually":
    st.subheader("Enter morphometric parameters")
    st.write(
        "Type in known values from published data, a previous scan report, "
        "or literature. These will be pushed directly to the generator as targets."
    )

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.markdown("**Structural parameters**")
        man_bvtv = st.number_input("BV/TV", 0.01, 0.80, 0.33, 0.01, format="%.3f")
        man_tbth = st.number_input("Tb.Th p50 (µm)", 10.0, 500.0, 180.0, 5.0, format="%.0f")
        man_tbn = st.number_input("Tb.N (/mm)", 0.1, 10.0, 2.0, 0.1, format="%.2f",
            help="Optional — for reference only, not used as a generator target.")
    with mcol2:
        st.markdown("**Spacing & geometry**")
        man_tbsp = st.number_input("Tb.Sp p50 (µm)", 10.0, 1000.0, 300.0, 10.0, format="%.0f",
            help="Optional — for reference only.")
        man_voxel = st.number_input("Voxel size (µm)", 1.0, 200.0, 39.0, 1.0, format="%.1f")

    st.divider()

    st.markdown("**Volume size for generation**")
    vcol1, vcol2, vcol3 = st.columns(3)
    with vcol1:
        man_nx = st.selectbox("XY size (voxels)", [32, 48, 64, 96, 128], index=4, key="man_nx")
    with vcol2:
        man_nz = st.selectbox("Z slices", [16, 24, 32, 40, 60], index=3, key="man_nz")
    with vcol3:
        st.metric("Total voxels", f"{man_nx * man_nx * man_nz:,}")

    st.divider()

    # Summary
    st.markdown("#### Summary")
    st.json({
        "BV/TV": man_bvtv,
        "Tb.Th p50 (µm)": man_tbth,
        "Tb.N (/mm)": man_tbn,
        "Tb.Sp p50 (µm)": man_tbsp,
        "Voxel size (µm)": man_voxel,
        "Volume": f"{man_nx}×{man_nx}×{man_nz}",
    })

    st.divider()
    st.subheader("Generate synthetic samples")

    gcol1, gcol2, gcol3 = st.columns(3)
    with gcol1:
        man_n_samples = st.number_input("Number of samples", 1, 50, 5, 1, key="man_n")
        man_base_seed = st.number_input("Starting seed", value=100, step=1, key="man_seed")
    with gcol2:
        man_sigma = st.slider("Base sigma", 1.0, 6.0, 2.5, 0.1, key="man_sigma")
        man_close = st.slider("Close iters", 0, 6, 3, 1, key="man_close")
    with gcol3:
        man_calibrate = st.checkbox("Calibrate Tb.Th", value=True, key="man_cal",
            help="Iteratively adjust base_sigma to match Tb.Th target.")

    with st.expander("Grayscale synthesis"):
        grcol1, grcol2 = st.columns(2)
        with grcol1:
            man_bone_mean = st.slider("Bone mean intensity", 50, 200, 90, 5, key="man_bmean")
            man_marrow_mean = st.slider("Marrow mean intensity", 5, 50, 15, 1, key="man_mmean")
            man_fill_sigma = st.slider("Solid fill sigma", 0.2, 2.0, 0.8, 0.1, key="man_fsig")
        with grcol2:
            man_noise_sd = st.slider("Noise SD", 0.0, 10.0, 2.0, 0.5, key="man_nsd")
            man_bg_tex = st.slider("Background texture SD", 0.0, 5.0, 0.5, 0.1, key="man_btex")

    if st.button(f"Generate {man_n_samples} sample(s)", type="primary",
                 width='stretch', key="btn_gen_manual"):
        targets = {
            "bvtv": round(man_bvtv, 3),
            "tbth_um": round(man_tbth, 0),
            "voxel_um": man_voxel,
            "nx": man_nx, "ny": man_nx, "nz": man_nz,
        }
        st.session_state["target_from_real"] = targets

        progress = st.progress(0, text="Starting...")
        samples = run_batch_generation(
            targets, man_n_samples,
            base_sigma=man_sigma, close_iters=man_close,
            base_seed=int(man_base_seed),
            calibrate_tbth=man_calibrate,
            bone_mean=man_bone_mean, marrow_mean=man_marrow_mean,
            solid_fill_sigma=man_fill_sigma,
            noise_sd=man_noise_sd, bg_tex_sd=man_bg_tex,
            progress_bar=progress,
        )
        st.session_state["generated_samples"] = samples
        st.session_state["bone_volume"] = samples[0]["volume"]

    # Always show gallery from session_state (persists across reruns)
    if "generated_samples" in st.session_state:
        display_sample_gallery(st.session_state["generated_samples"], man_voxel)


# ══════════════════════════════════════════════════════════════
# MODE: UPLOAD SCAN
# ══════════════════════════════════════════════════════════════

volume = None

if input_mode == "Upload micro-CT scan" and uploaded:
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
        # Use percentile clipping to avoid outlier hot/dead pixels
        vmin = np.percentile(volume, 0.5)
        vmax = np.percentile(volume, 99.5)
        if vmax > vmin:
            clipped = np.clip(volume.astype(np.float32), vmin, vmax)
            volume = ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            volume = np.zeros_like(volume, dtype=np.uint8)
        st.sidebar.caption(f"Raw range: {volume.min()}–{volume.max()} | "
                           f"Clipped at p0.5={vmin:.0f}, p99.5={vmax:.0f}")

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
        if nz > 1:
            slice_idx = st.slider("Preview Z-slice", 0, nz - 1, mid_z, key="extract_slice")
        else:
            slice_idx = 0
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
        if not HAS_MORPH and not HAS_BONEJ:
            st.warning("No morphometric measurement module available.")
        else:
            # Method selector
            methods = []
            if HAS_BONEJ:
                methods.append("BoneJ (validated)")
            if HAS_MORPH:
                methods.append("Built-in (v15)")
            morph_method = st.radio("Measurement method", methods, horizontal=True,
                key="morph_method")

            # BoneJ options
            include_da = False
            include_smi = False
            if morph_method == "BoneJ (validated)":
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    include_da = st.checkbox("Degree of Anisotropy (DA)", value=False,
                        key="bonej_da",
                        help="Mean Intercept Length — adds ~10-30s.")
                with bcol2:
                    include_smi = st.checkbox("Structure Model Index (SMI)", value=False,
                        key="bonej_smi",
                        help="Plates=0, Rods=3, Spheres=4.")

            # Measure button
            if st.button("Measure morphometrics", type="primary", key="btn_measure"):
                with st.spinner("Measuring..."):
                    if morph_method == "BoneJ (validated)":
                        morph = measure_all_bonej(bone_mask, voxel_um,
                                                   include_anisotropy=include_da,
                                                   include_smi=include_smi)
                    else:
                        morph = measure_all_morphometrics(bone_mask, voxel_um)
                st.session_state["real_morphometrics"] = morph
                st.session_state["morph_method_used"] = morph_method

            # Display from session_state
            if "real_morphometrics" in st.session_state:
                morph = st.session_state["real_morphometrics"]
                method_used = st.session_state.get("morph_method_used", "")

                # Main metrics row
                metric_cols = st.columns(5 + (1 if 'DA' in morph else 0) + (1 if 'SMI' in morph else 0))
                metric_cols[0].metric("BV/TV", f"{morph['BVTV']:.3f}")
                metric_cols[1].metric("Tb.Th (p50)", f"{morph['TbTh_um_p50']:.0f} µm")
                metric_cols[2].metric("Tb.N", f"{morph['TbN_per_mm']:.2f} /mm")
                metric_cols[3].metric("Tb.Sp (p50)", f"{morph['TbSp_um_p50']:.0f} µm")
                metric_cols[4].metric("LCC", f"{morph['lcc_frac']:.3f}")
                col_idx = 5
                if 'DA' in morph:
                    da_label = "isotropic" if morph['DA'] < 0.3 else "anisotropic"
                    metric_cols[col_idx].metric("DA", f"{morph['DA']:.3f}", da_label)
                    col_idx += 1
                if 'SMI' in morph:
                    smi_label = "plates" if morph['SMI'] < 1 else ("rods" if morph['SMI'] > 2 else "mixed")
                    metric_cols[col_idx].metric("SMI", f"{morph['SMI']:.2f}", smi_label)

                if method_used:
                    st.caption(f"Measured with: {method_used}")

                with st.expander("Full measurements"):
                    mcol1, mcol2 = st.columns(2)
                    with mcol1:
                        d1 = {
                            "BV/TV": round(morph["BVTV"], 4),
                            "Tb.Th p50 (µm)": round(morph["TbTh_um_p50"], 1),
                            "Tb.Th p90 (µm)": round(morph["TbTh_um_p90"], 1),
                            "Tb.N (/mm)": round(morph["TbN_per_mm"], 3),
                        }
                        if 'DA' in morph:
                            d1["DA"] = round(morph["DA"], 4)
                        st.json(d1)
                    with mcol2:
                        d2 = {
                            "Tb.Sp p50 (µm)": round(morph["TbSp_um_p50"], 1),
                            "Tb.Sp p90 (µm)": round(morph["TbSp_um_p90"], 1),
                            "Euler number": morph["Euler"],
                            "LCC fraction": round(morph["lcc_frac"], 4),
                            "Components": morph["n_components"],
                        }
                        if 'SMI' in morph:
                            d2["SMI"] = round(morph["SMI"], 2)
                        if 'connectivity_density' in morph:
                            d2["Conn. density"] = round(morph["connectivity_density"], 6)
                        st.json(d2)

                # Thickness map visualization (BoneJ only)
                if 'thickness_map' in morph:
                    with st.expander("Thickness & spacing maps"):
                        mid_t = bone_mask.shape[0] // 2
                        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                        im0 = axes[0].imshow(morph['thickness_map'][mid_t].T,
                                              cmap='hot', origin='lower',
                                              extent=[0, nx*voxel_mm, 0, ny*voxel_mm])
                        axes[0].set_title("Tb.Th map (µm)")
                        axes[0].set_xlabel("x [mm]"); axes[0].set_ylabel("y [mm]")
                        plt.colorbar(im0, ax=axes[0])
                        im1 = axes[1].imshow(morph['spacing_map'][mid_t].T,
                                              cmap='cool', origin='lower',
                                              extent=[0, nx*voxel_mm, 0, ny*voxel_mm])
                        axes[1].set_title("Tb.Sp map (µm)")
                        axes[1].set_xlabel("x [mm]")
                        plt.colorbar(im1, ax=axes[1])
                        plt.tight_layout()
                        st.pyplot(fig); plt.close()

                # ── Generate synthetic samples ──
                st.divider()
                st.subheader("Generate matched synthetic samples")

                gcol1, gcol2, gcol3 = st.columns(3)
                with gcol1:
                    up_n_samples = st.number_input("Number of samples", 1, 50, 5, 1, key="up_n")
                    up_base_seed = st.number_input("Starting seed", value=100, step=1, key="up_seed")
                with gcol2:
                    up_sigma = st.slider("Base sigma", 1.0, 6.0, 2.5, 0.1, key="up_sigma")
                    up_close = st.slider("Close iters", 0, 6, 3, 1, key="up_close")
                with gcol3:
                    up_calibrate = st.checkbox("Calibrate Tb.Th", value=True, key="up_cal",
                        help="Iteratively adjust base_sigma to match measured Tb.Th.")

                with st.expander("Grayscale synthesis"):
                    grcol1, grcol2 = st.columns(2)
                    with grcol1:
                        up_bone_mean = st.slider("Bone mean intensity", 50, 200, 90, 5, key="up_bmean")
                        up_marrow_mean = st.slider("Marrow mean intensity", 5, 50, 15, 1, key="up_mmean")
                        up_fill_sigma = st.slider("Solid fill sigma", 0.2, 2.0, 0.8, 0.1, key="up_fsig")
                    with grcol2:
                        up_noise_sd = st.slider("Noise SD", 0.0, 10.0, 2.0, 0.5, key="up_nsd")
                        up_bg_tex = st.slider("Background texture SD", 0.0, 5.0, 0.5, 0.1, key="up_btex")

                if st.button(f"Generate {up_n_samples} sample(s)", type="primary",
                             width='stretch', key="btn_gen_upload"):
                    targets = {
                        "bvtv": round(morph["BVTV"], 3),
                        "tbth_um": round(morph["TbTh_um_p50"], 0),
                        "voxel_um": voxel_um,
                        "nx": nx, "ny": ny, "nz": nz,
                    }
                    st.session_state["target_from_real"] = targets

                    progress = st.progress(0, text="Starting...")
                    samples = run_batch_generation(
                        targets, up_n_samples,
                        base_sigma=up_sigma, close_iters=up_close,
                        base_seed=int(up_base_seed),
                        calibrate_tbth=up_calibrate,
                        bone_mean=up_bone_mean, marrow_mean=up_marrow_mean,
                        solid_fill_sigma=up_fill_sigma,
                        noise_sd=up_noise_sd, bg_tex_sd=up_bg_tex,
                        progress_bar=progress,
                    )
                    st.session_state["generated_samples"] = samples
                    st.session_state["bone_volume"] = samples[0]["volume"]

                # Always show gallery from session_state
                if "generated_samples" in st.session_state:
                    display_sample_gallery(
                        st.session_state["generated_samples"], voxel_um)

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
                ("Euler", "Euler", ".0f", ""),
                ("LCC frac", "lcc_frac", ".3f", ""),
                ("Components", "n_components", ".0f", ""),
            ]

            cols = st.columns([2, 2, 2, 2])
            cols[0].markdown("**Metric**")
            cols[1].markdown("**Real**")
            cols[2].markdown("**Synthetic**")
            cols[3].markdown("**Δ (%)**")

            for label, key, fmt, unit in metrics:
                rv = real_morph[key]
                sv = syn_morph[key]
                try:
                    if isinstance(rv, (int, np.integer)):
                        delta_str = f"{int(sv) - int(rv):+d}"
                    elif rv != 0:
                        delta_str = f"{(sv - rv) / rv * 100:+.1f}%"
                    else:
                        delta_str = "—"
                except (TypeError, ValueError):
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
            if max_z > 0:
                comp_slice = st.slider("Z-slice", 0, max_z, max_z // 2, key="val_slice")
            else:
                comp_slice = 0

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

elif input_mode == "Upload micro-CT scan":
    st.info(
        "Upload a real micro-CT volume using the sidebar. "
        "Supported formats: TIFF stack, ZIP of TIFFs, NIfTI, or NumPy (.npy)."
    )