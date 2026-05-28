"""
Page 4: 3D Viewer & Strain Mapping
===================================
  1. Build a 3D surface model from bone volumes (synthetic or real)
  2. Load strain / displacement fields from TIFFs or FE results
  3. Map mechanical fields onto the 3D surface
  4. Interactive rotation, zoom, and slice controls via Plotly
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import sys, io, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fe_coupling"))
from step3_generator_fe_coupling import run_fe_analysis

st.set_page_config(page_title="3D viewer", page_icon="🧊", layout="wide")
st.title("3D viewer & strain mapping")
st.caption("Visualise bone structure and mechanical fields in 3D")


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def extract_surface(bone_mask, voxel_mm, level=0.5):
    """Extract triangulated surface from binary volume using marching cubes."""
    from skimage.measure import marching_cubes
    verts, faces, normals, _ = marching_cubes(
        bone_mask.astype(float), level=level,
        spacing=(voxel_mm, voxel_mm, voxel_mm),
    )
    return verts, faces, normals


def build_mesh_figure(verts, faces, color_values=None, colorscale='Viridis',
                      color_label='', opacity=1.0, title=''):
    """Build a Plotly Mesh3d figure from vertices and faces."""
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    mesh_kwargs = dict(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=opacity,
        flatshading=True,
    )

    if color_values is not None:
        mesh_kwargs["intensity"] = color_values
        mesh_kwargs["colorscale"] = colorscale
        mesh_kwargs["colorbar"] = dict(title=color_label, thickness=15)
        mesh_kwargs["cmin"] = np.nanpercentile(color_values, 2)
        mesh_kwargs["cmax"] = np.nanpercentile(color_values, 98)
    else:
        mesh_kwargs["color"] = '#C8BFA9'

    fig = go.Figure(data=[go.Mesh3d(**mesh_kwargs)])
    fig.update_layout(
        scene=dict(
            xaxis_title='x [mm]',
            yaxis_title='y [mm]',
            zaxis_title='z [mm]',
            aspectmode='data',
        ),
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
    )
    return fig


def sample_field_at_vertices(verts, field_volume, voxel_mm):
    """Sample a 3D field at surface vertex positions using trilinear interpolation."""
    from scipy.ndimage import map_coordinates
    # Convert vertex positions (mm) back to voxel coordinates
    coords = verts / voxel_mm
    # field_volume is (Z, Y, X), verts are (z, y, x) after marching_cubes
    values = map_coordinates(
        field_volume.astype(float),
        [coords[:, 0], coords[:, 1], coords[:, 2]],
        order=1, mode='nearest',
    )
    return values


def strain_volume_from_fe(fe_results, bone_mask, voxel_mm, component='eps_von_mises'):
    """Convert element-centroid strain data to a voxel volume for surface mapping."""
    strain = fe_results["strain_field"]
    centroids = strain["centroids"]
    values = strain[component]
    nz, ny, nx = bone_mask.shape

    vol = np.full((nz, ny, nx), np.nan, dtype=float)
    for idx in range(len(values)):
        ci = int(round(centroids[idx, 0] / voxel_mm))
        cj = int(round(centroids[idx, 1] / voxel_mm))
        ck = int(round(centroids[idx, 2] / voxel_mm))
        if 0 <= ck < nz and 0 <= cj < ny and 0 <= ci < nx:
            vol[ck, cj, ci] = values[idx]

    # Fill NaN with nearest neighbor
    from scipy.ndimage import generic_filter
    mask_nan = np.isnan(vol) & (bone_mask > 0)
    if mask_nan.any():
        from scipy.ndimage import distance_transform_edt
        filled = vol.copy()
        filled[np.isnan(filled)] = 0
        vol = np.where(np.isnan(vol), filled, vol)

    return vol


def load_tiff_volume(uploaded_files):
    """Load TIFF stack as float volume."""
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


def load_tiff_zip(uploaded_file):
    """Load ZIP of TIFFs as float volume."""
    from PIL import Image
    slices = []
    with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as zf:
        names = sorted([n for n in zf.namelist()
                        if n.lower().endswith(('.tif', '.tiff')) and not n.startswith('__')])
        for name in names:
            with zf.open(name) as f:
                img = Image.open(f)
                slices.append(np.array(img, dtype=np.float64))
    return np.stack(slices, axis=0)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

st.sidebar.header("3D settings")
mesh_opacity = st.sidebar.slider("Mesh opacity", 0.3, 1.0, 0.9, 0.05)
colorscale = st.sidebar.selectbox("Colour scale", [
    "Viridis", "Plasma", "Inferno", "RdBu", "RdBu_r",
    "Turbo", "Hot", "Jet", "Bone",
])

st.sidebar.header("Crop / clip")
st.sidebar.caption("Crop the volume to see internal trabecular structure")
clip_axis = st.sidebar.selectbox("Clip axis", ["None", "X", "Y", "Z"], index=3)
clip_pct = st.sidebar.slider("Clip position (%)", 10, 90, 50, 5,
    help="Remove this % of the volume along the clip axis to reveal internal structure.")

st.sidebar.header("Subsample")
subsample = st.sidebar.selectbox("Downsample factor", [1, 2, 4], index=0,
    help="Reduce volume size for faster rendering and FE. 2 = half resolution.")

st.sidebar.header("Strain source")
strain_source = st.sidebar.radio("Load strain from", [
    "FE results (session)",
    "Upload TIFF stack",
    "Upload NumPy (.npy)",
    "None (structure only)",
])


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════

tab_3d, tab_load_strain, tab_fe = st.tabs([
    "🧊 3D model",
    "📂 Load strain data",
    "⚙️ Run FE & map",
])


# ──────────────────────────────────────────────────────────────
# TAB 1 — 3D MODEL
# ──────────────────────────────────────────────────────────────
with tab_3d:
    st.subheader("3D bone model")

    # Find a bone mask
    bone_mask = None
    voxel_um = 39.0
    source_label = ""

    if "bone_volume" in st.session_state:
        vol = st.session_state["bone_volume"]
        bone_mask = vol["bone_mask"]
        voxel_um = vol["voxel_um"]
        source_label = "synthetic"
    elif "real_bone_mask" in st.session_state:
        bone_mask = st.session_state["real_bone_mask"]
        voxel_um = st.session_state.get("real_voxel_um", 39.0)
        source_label = "real"

    if bone_mask is None:
        st.info("No bone volume in session. Generate one in the **Bone Generator** "
                "or load a scan in the **Data Loader**.")
    else:
        nz, ny, nx = bone_mask.shape
        voxel_mm = voxel_um / 1000.0
        st.success(f"Bone mask: {nx}×{ny}×{nz} ({source_label}), "
                   f"voxel={voxel_um:.0f} µm, BV/TV={bone_mask.mean():.3f}")

        # Apply subsample
        view_mask = bone_mask.copy()
        view_voxel = voxel_mm
        if subsample > 1:
            view_mask = view_mask[::subsample, ::subsample, ::subsample]
            view_voxel = voxel_mm * subsample
            nz_v, ny_v, nx_v = view_mask.shape
            st.caption(f"Subsampled: {nx_v}×{ny_v}×{nz_v}")

        # Apply clip to reveal internal structure
        if clip_axis != "None":
            nz_v, ny_v, nx_v = view_mask.shape
            if clip_axis == "Z":
                cut = int(nz_v * clip_pct / 100)
                view_mask = view_mask[:cut, :, :]
            elif clip_axis == "Y":
                cut = int(ny_v * clip_pct / 100)
                view_mask = view_mask[:, :cut, :]
            elif clip_axis == "X":
                cut = int(nx_v * clip_pct / 100)
                view_mask = view_mask[:, :, :cut]
            st.caption(f"Clipped {clip_axis} at {clip_pct}% → shape {view_mask.shape[::-1]}")

        # Extract surface from cropped/subsampled volume
        with st.spinner("Extracting 3D surface (marching cubes)..."):
            try:
                verts, faces, normals = extract_surface(view_mask, view_voxel)
                st.session_state["mesh_verts"] = verts
                st.session_state["mesh_faces"] = faces
                st.caption(f"{len(verts):,} vertices, {len(faces):,} triangles")
            except Exception as e:
                st.error(f"Surface extraction failed: {e}")
                verts, faces = None, None

        if verts is not None:
            # Check for strain data to overlay
            strain_vol = st.session_state.get("strain_volume_3d")
            strain_label = st.session_state.get("strain_label_3d", "")

            if strain_vol is not None and strain_source != "None (structure only)":
                with st.spinner("Mapping strain to surface..."):
                    vertex_strain = sample_field_at_vertices(verts, strain_vol, view_voxel)

                fig = build_mesh_figure(
                    verts, faces,
                    color_values=vertex_strain,
                    colorscale=colorscale,
                    color_label=strain_label,
                    opacity=mesh_opacity,
                    title=f"3D bone + {strain_label}",
                )
            else:
                fig = build_mesh_figure(
                    verts, faces,
                    opacity=mesh_opacity,
                    title="3D bone structure",
                )

            st.plotly_chart(fig, width='stretch')

            # Also show a 2D slice alongside for reference
            with st.expander("2D slice reference"):
                nz_view = view_mask.shape[0]
                ny_view = view_mask.shape[1]
                nx_view = view_mask.shape[2]
                mid_z = nz_view // 2
                if nz_view > 1:
                    ref_z = st.slider("Z-slice", 0, nz_view - 1, mid_z, key="ref3d_z")
                else:
                    ref_z = 0
                extent = [0, nx_view * view_voxel, 0, ny_view * view_voxel]

                if strain_vol is not None and strain_source != "None (structure only)":
                    fig2, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(view_mask[ref_z].T, cmap='gray', origin='lower', extent=extent)
                    axes[0].set_title("Binary mask"); axes[0].set_xlabel("x [mm]"); axes[0].set_ylabel("y [mm]")
                    im = axes[1].imshow(strain_vol[ref_z].T, cmap=colorscale.lower(),
                                        origin='lower', extent=extent)
                    axes[1].set_title(strain_label); axes[1].set_xlabel("x [mm]")
                    plt.colorbar(im, ax=axes[1])
                else:
                    fig2, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(view_mask[ref_z].T, cmap='gray', origin='lower', extent=extent)
                    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                    ax.set_title(f"z-slice {ref_z}")
                plt.tight_layout()
                st.pyplot(fig2); plt.close()


# ──────────────────────────────────────────────────────────────
# TAB 2 — LOAD STRAIN DATA
# ──────────────────────────────────────────────────────────────
with tab_load_strain:
    st.subheader("Load strain / displacement field")
    st.write("Import strain or displacement maps from DIC, external FE, "
             "or experimental measurements to map onto the 3D model.")

    load_format = st.selectbox("File format", [
        "TIFF stack (individual files)",
        "TIFF stack (ZIP)",
        "NumPy (.npy)",
    ], key="strain3d_fmt")

    lcol1, lcol2 = st.columns(2)
    with lcol1:
        strain_component = st.selectbox("Field type", [
            "von Mises strain",
            "Axial strain (ε_zz)",
            "Transverse strain (ε_xx)",
            "Shear strain (ε_xy)",
            "Max principal strain",
            "Displacement magnitude",
            "Custom field",
        ], key="strain3d_comp")
    with lcol2:
        strain_voxel = st.number_input("Voxel size (µm)", value=39.0, step=1.0, key="strain3d_vox")

    strain_uploaded = None
    if load_format == "TIFF stack (individual files)":
        strain_uploaded = st.file_uploader("Upload strain TIFFs", type=["tif", "tiff"],
                                           accept_multiple_files=True, key="strain3d_tiff")
    elif load_format == "TIFF stack (ZIP)":
        strain_uploaded = st.file_uploader("Upload ZIP", type=["zip"], key="strain3d_zip")
    elif load_format == "NumPy (.npy)":
        strain_uploaded = st.file_uploader("Upload .npy", type=["npy"], key="strain3d_npy")

    if strain_uploaded:
        try:
            with st.spinner("Loading strain field..."):
                if load_format == "TIFF stack (individual files)" and len(strain_uploaded) > 0:
                    strain_data = load_tiff_volume(strain_uploaded)
                elif load_format == "TIFF stack (ZIP)":
                    strain_data = load_tiff_zip(strain_uploaded)
                elif load_format == "NumPy (.npy)":
                    strain_data = np.load(io.BytesIO(strain_uploaded.read()))

            nz_s, ny_s, nx_s = strain_data.shape
            st.success(f"Loaded strain field: {nx_s}×{ny_s}×{nz_s}")
            st.json({
                "Min": round(float(strain_data.min()), 6),
                "Max": round(float(strain_data.max()), 6),
                "Mean": round(float(strain_data.mean()), 6),
            })

            # Store for 3D mapping
            st.session_state["strain_volume_3d"] = strain_data
            st.session_state["strain_label_3d"] = strain_component

            # Preview mid-slice
            mid = nz_s // 2
            voxel_mm_s = strain_voxel / 1000.0
            fig, ax = plt.subplots(figsize=(6, 6))
            ext = [0, nx_s * voxel_mm_s, 0, ny_s * voxel_mm_s]
            im = ax.imshow(strain_data[mid].T, cmap='RdBu_r', origin='lower', extent=ext)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            ax.set_title(f"{strain_component} — z={mid}")
            plt.colorbar(im, ax=ax)
            st.pyplot(fig); plt.close()

            st.info("Go to the **3D model** tab to see the strain mapped onto the bone surface.")

        except Exception as e:
            st.error(f"Failed to load: {e}")


# ──────────────────────────────────────────────────────────────
# TAB 3 — RUN FE & MAP
# ──────────────────────────────────────────────────────────────
with tab_fe:
    st.subheader("Run FE analysis and map to 3D")
    st.write("Run micro-FE directly on the bone volume, then view "
             "strain fields on the 3D model.")

    bone_mask_fe = None
    voxel_um_fe = 39.0

    if "bone_volume" in st.session_state:
        vol = st.session_state["bone_volume"]
        bone_mask_fe = vol["bone_mask"]
        voxel_um_fe = vol["voxel_um"]
        nz_f, ny_f, nx_f = bone_mask_fe.shape
        st.info(f"Synthetic bone: {nx_f}×{ny_f}×{nz_f}, voxel={voxel_um_fe:.0f} µm")
    elif "real_bone_mask" in st.session_state:
        bone_mask_fe = st.session_state["real_bone_mask"]
        voxel_um_fe = st.session_state.get("real_voxel_um", 39.0)
        nz_f, ny_f, nx_f = bone_mask_fe.shape
        st.info(f"Real bone: {nx_f}×{ny_f}×{nz_f}, voxel={voxel_um_fe:.0f} µm")

    if bone_mask_fe is None:
        st.warning("No bone volume available. Load or generate one first.")
    else:
        nz_f, ny_f, nx_f = bone_mask_fe.shape
        total_voxels = nx_f * ny_f * nz_f
        bone_voxels = int(bone_mask_fe.sum())

        # Size warning and subsample for FE
        fe_sub = st.selectbox("FE subsample", [1, 2, 4], index=0, key="fe3d_sub",
            help="Downsample before FE. 2× halves each dimension → 8× fewer elements.")

        fe_mask = bone_mask_fe
        fe_voxel_um = voxel_um_fe
        if fe_sub > 1:
            fe_mask = bone_mask_fe[::fe_sub, ::fe_sub, ::fe_sub]
            fe_voxel_um = voxel_um_fe * fe_sub

        fe_bone = int(fe_mask.sum())
        nz_fe, ny_fe, nx_fe = fe_mask.shape

        if fe_bone > 100_000:
            st.warning(f"⚠️ {fe_bone:,} bone elements — FE will be slow (minutes). "
                       f"Consider subsample=2 ({fe_bone // 8:,} elements).")
        elif fe_bone > 30_000:
            st.info(f"{fe_bone:,} bone elements — expect ~30-90s.")
        else:
            st.caption(f"{fe_bone:,} bone elements — should be quick (~10-30s).")

        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            fe_load = st.selectbox("Load case", ["compression", "tension", "torque"], key="fe3d_load")
        with fcol2:
            fe_E = st.number_input("E_bone (MPa)", value=18000.0, step=1000.0, key="fe3d_E")
        with fcol3:
            fe_strain = st.number_input("Applied strain", value=0.01, step=0.005,
                                        format="%.3f", key="fe3d_strain")

        use_hetero = st.checkbox("Heterogeneous E from grayscale",
            key="fe3d_hetero",
            help="Map greyscale intensity → per-element E via power law. "
                 "Requires a grayscale volume in session (from generator or pipeline).")

        strain_comp = st.selectbox("Strain component to map", [
            "eps_von_mises",
            "eps_zz",
            "eps_xx",
            "eps_yy",
            "eps_xy",
            "eps_max_principal",
        ], key="fe3d_comp")

        # Find grayscale if heterogeneous requested
        gray_for_fe = None
        if use_hetero:
            if "pipeline_gray" in st.session_state:
                gray_for_fe = st.session_state["pipeline_gray"]
                if fe_sub > 1:
                    gray_for_fe = gray_for_fe[::fe_sub, ::fe_sub, ::fe_sub]
                st.caption(f"Using grayscale from session (mean={gray_for_fe[fe_mask > 0].mean():.0f})")
            else:
                st.warning("No grayscale in session. Generate one first or uncheck.")
                use_hetero = False

        if st.button("Run FE & build 3D strain map", type="primary",
                     width='stretch', key="btn_fe3d"):

            voxel_mm_fe = fe_voxel_um / 1000.0

            # Step 1: Run FE on (possibly subsampled) volume
            spinner_msg = f"Running FE ({fe_load}) on {nx_fe}×{ny_fe}×{nz_fe}"
            if use_hetero:
                spinner_msg += " [heterogeneous E]"
            with st.spinner(spinner_msg + "..."):
                fe = run_fe_analysis(
                    fe_mask, voxel_mm_fe,
                    load_type=fe_load, E_bone=fe_E,
                    applied_strain=fe_strain,
                    grayscale=gray_for_fe if use_hetero else None,
                    verbose=False,
                )
            st.session_state["pipeline_fe"] = fe

            if fe["apparent_modulus"] is not None:
                st.metric("E_apparent", f"{fe['apparent_modulus']:.0f} MPa")

            # Step 2: Convert strain to volume
            with st.spinner("Converting strain to voxel volume..."):
                strain_vol = strain_volume_from_fe(
                    fe, fe_mask, voxel_mm_fe, component=strain_comp,
                )

            st.session_state["strain_volume_3d"] = strain_vol
            st.session_state["strain_label_3d"] = strain_comp
            st.session_state["fe_voxel_mm"] = voxel_mm_fe

            # Step 3: Extract surface and map
            with st.spinner("Building 3D surface..."):
                verts, faces, normals = extract_surface(fe_mask, voxel_mm_fe)
                vertex_strain = sample_field_at_vertices(verts, strain_vol, voxel_mm_fe)

            st.session_state["mesh_verts"] = verts
            st.session_state["mesh_faces"] = faces

            # Build figure
            fig = build_mesh_figure(
                verts, faces,
                color_values=vertex_strain,
                colorscale=colorscale,
                color_label=strain_comp,
                opacity=mesh_opacity,
                title=f"3D bone — {strain_comp} ({fe_load})",
            )
            st.plotly_chart(fig, width='stretch')

            # Strain statistics
            strain_data = fe["strain_field"]
            with st.expander("Strain statistics"):
                st.json({
                    f"{strain_comp} min": round(float(strain_data[strain_comp].min()), 6),
                    f"{strain_comp} max": round(float(strain_data[strain_comp].max()), 6),
                    f"{strain_comp} mean": round(float(strain_data[strain_comp].mean()), 6),
                    "Elements": fe["n_elements"],
                    "Solve time": f"{fe['solve_time']:.1f}s",
                })

            st.success("Done — switch to the **3D model** tab to interact with the result.")

        # Show previous results
        elif "pipeline_fe" in st.session_state and "strain_volume_3d" in st.session_state:
            verts = st.session_state.get("mesh_verts")
            faces = st.session_state.get("mesh_faces")
            strain_vol = st.session_state["strain_volume_3d"]
            strain_label = st.session_state.get("strain_label_3d", "")

            if verts is not None:
                voxel_mm_fe = st.session_state.get("fe_voxel_mm", voxel_um_fe / 1000.0)
                vertex_strain = sample_field_at_vertices(verts, strain_vol, voxel_mm_fe)
                fig = build_mesh_figure(
                    verts, faces,
                    color_values=vertex_strain,
                    colorscale=colorscale,
                    color_label=strain_label,
                    opacity=mesh_opacity,
                    title=f"3D bone — {strain_label} (previous run)",
                )
                st.plotly_chart(fig, width='stretch')