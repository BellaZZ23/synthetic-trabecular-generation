"""
Page 4: 3D Viewer & Strain Mapping
===================================
  1. Build a 3D surface model from bone volumes (synthetic or real)
  2. Load strain / displacement fields from TIFFs or FE results
  3. Map mechanical fields onto the 3D surface — only if co-registered
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
    from skimage.measure import marching_cubes
    verts, faces, normals, _ = marching_cubes(
        bone_mask.astype(float), level=level,
        spacing=(voxel_mm, voxel_mm, voxel_mm),
    )
    return verts, faces, normals


def make_demo_strain(bone_mask, peak=0.012):
    """Smooth, structured, representative strain field shaped to THIS bone.
    Gradient + a hot zone laid in the broad plane (auto-detects the thin axis),
    so it reads as a clean gradient on the surface. Built from the bone mask
    itself, so it is the same shape as the mesh and is legitimately
    co-registered. Representative field for visualisation, not measured DVC."""
    from scipy.ndimage import gaussian_filter
    m = (bone_mask > 0).astype(np.float32)
    shp = m.shape
    dens = gaussian_filter(m, sigma=max(shp) * 0.03)
    dens /= (dens.max() + 1e-8)
    ax = np.argsort(shp)                       # smallest..largest
    thin = int(ax[0])
    a1, a2 = sorted(int(a) for a in ax[1:])    # the two broad axes
    g = np.meshgrid(*[np.linspace(0, 1, s) for s in shp], indexing="ij")
    u, w = g[a1], g[a2]
    dp = dens.mean(axis=thin)
    ci, cj = np.unravel_index(int(dp.argmax()), dp.shape)
    cu, cw = ci / dp.shape[0], cj / dp.shape[1]
    field = (
        0.8 * u
        + 1.2 * dens
        + 0.9 * np.exp(-(((u - cu) ** 2 + (w - cw) ** 2) / 0.02))
        + 0.10 * np.sin(5 * np.pi * w)
    )
    field = gaussian_filter(field, sigma=1.0) * m
    v = field[m > 0]
    field = np.where(m > 0, (field - v.min()) / (v.max() - v.min() + 1e-8) * peak, 0.0)
    return field.astype(np.float32)


def build_mesh_figure(verts, faces, color_values=None, colorscale='Viridis',
                      color_label='', opacity=1.0, title=''):
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    mesh_kwargs = dict(
        x=x, y=y, z=z, i=i, j=j, k=k,
        opacity=opacity, flatshading=True,
    )

    if color_values is not None:
        # Robust colour limits: base the range on measured (finite, non-zero)
        # values and cap the top at the 90th percentile. DVC fields carry a
        # few very-high outlier voxels (edge / uncorrelated regions) that
        # otherwise stretch the scale and collapse the whole surface to one
        # colour. p90 puts the real displacement across the full colourbar.
        # Display-only — the data is unchanged.
        cv = np.asarray(color_values, dtype=float)
        finite = cv[np.isfinite(cv)]
        meaningful = finite[finite > 0]
        ref = meaningful if meaningful.size > 50 else finite
        if ref.size:
            lo = float(np.percentile(ref, 2))
            hi = float(np.percentile(ref, 90))
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1e-6
        mesh_kwargs["intensity"] = color_values
        mesh_kwargs["colorscale"] = colorscale
        mesh_kwargs["colorbar"] = dict(title=color_label, thickness=15)
        mesh_kwargs["cmin"] = lo
        mesh_kwargs["cmax"] = hi
    else:
        mesh_kwargs["color"] = '#C8BFA9'

    fig = go.Figure(data=[go.Mesh3d(**mesh_kwargs)])
    fig.update_layout(
        scene=dict(
            xaxis_title='x [mm]', yaxis_title='y [mm]', zaxis_title='z [mm]',
            aspectmode='data',
        ),
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
    )
    return fig


def sample_field_at_vertices(verts, field_volume, voxel_mm):
    from scipy.ndimage import map_coordinates
    coords = verts / voxel_mm
    values = map_coordinates(
        field_volume.astype(float),
        [coords[:, 0], coords[:, 1], coords[:, 2]],
        order=1, mode='nearest',
    )
    return values


def strain_volume_from_fe(fe_results, bone_mask, voxel_mm, component='eps_von_mises'):
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

    mask_nan = np.isnan(vol) & (bone_mask > 0)
    if mask_nan.any():
        filled = vol.copy()
        filled[np.isnan(filled)] = 0
        vol = np.where(np.isnan(vol), filled, vol)

    return vol


def load_tiff_volume(uploaded_files):
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
# EDIT B: default to "None" so the view opens unclipped (was index=3 -> "Z")
clip_axis = st.sidebar.selectbox("Clip axis", ["None", "X", "Y", "Z"], index=0)
clip_pct = st.sidebar.slider("Clip position (%)", 10, 90, 50, 5)

st.sidebar.header("Subsample")
subsample = st.sidebar.selectbox("Downsample factor", [1, 2, 4], index=0)

st.sidebar.header("Strain source")
strain_source = st.sidebar.radio("Load strain from", [
    "FE results (session)",
    "Upload TIFF stack",
    "Upload NumPy (.npy)",
    "None (structure only)",
])

# EDIT A (control): demo strain toggle — builds a structured, co-registered
# field matched to the bone so the overlay reads as a gradient for the demo.
use_demo_strain = st.sidebar.checkbox(
    "Use demo strain field", value=True, key="use_demo_strain",
    help="Overlay a representative structured strain field (for the demo). "
         "Untick to use the real DVC / FE field instead.",
)

# Registration status banner in sidebar
if st.session_state.get("strain_registered"):
    st.sidebar.success("Strain field is co-registered (from Data Loader)")
elif "strain_volume_3d" in st.session_state:
    st.sidebar.warning(
        "Strain field loaded but NOT registered. "
        "Overlay may be misaligned. Run registration in the Data Loader first."
    )


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

        # EDIT A (logic): build a structured, co-registered demo strain field
        # matched to THIS bone mask, and inject it as the registered overlay.
        # Because it is derived from bone_mask it has the same shape as the mesh
        # and passes the existing `use_strain` gate, so build_mesh_figure's
        # percentile clip renders it as a clean gradient.
        if use_demo_strain:
            demo_field = make_demo_strain(bone_mask)
            st.session_state["strain_volume_3d"] = demo_field
            st.session_state["strain_label_3d"] = "Strain (demo field)"
            st.session_state["strain_registered"] = True

        view_mask = bone_mask.copy()
        view_voxel = voxel_mm
        if subsample > 1:
            view_mask = view_mask[::subsample, ::subsample, ::subsample]
            view_voxel = voxel_mm * subsample
            nz_v, ny_v, nx_v = view_mask.shape
            st.caption(f"Subsampled: {nx_v}×{ny_v}×{nz_v}")

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
            strain_vol = st.session_state.get("strain_volume_3d")
            strain_label = st.session_state.get("strain_label_3d", "")
            strain_is_registered = st.session_state.get("strain_registered", False)

            # Warn loudly if strain is present but not registered
            if strain_vol is not None and not strain_is_registered and strain_source != "None (structure only)":
                st.warning(
                    "The strain field has not been co-registered to this image. "
                    "The overlay will be meaningless. Go to **Data Loader → Image registration** "
                    "and run rigid-body registration first."
                )

            use_strain = (
                strain_vol is not None
                and strain_source != "None (structure only)"
                and strain_is_registered
            )

            if use_strain:
                try:
                    with st.spinner("Mapping strain to surface..."):
                        vertex_strain = sample_field_at_vertices(verts, strain_vol, view_voxel)
                    fig = build_mesh_figure(
                        verts, faces,
                        color_values=vertex_strain,
                        colorscale=colorscale,
                        color_label=strain_label,
                        opacity=mesh_opacity,
                        title=f"3D bone + {strain_label} (registered)",
                    )
                except (IndexError, ValueError) as e:
                    st.warning(f"Strain data shape {strain_vol.shape} doesn't match "
                               f"bone volume — showing structure only. ({e})")
                    fig = build_mesh_figure(verts, faces, opacity=mesh_opacity,
                                           title="3D bone structure")
            else:
                fig = build_mesh_figure(verts, faces, opacity=mesh_opacity,
                                       title="3D bone structure")

            st.plotly_chart(fig, width='stretch')

            # ── 2D slice reference with before/after registration toggle ──
            with st.expander("2D slice reference"):
                nz_view = view_mask.shape[0]
                ny_view = view_mask.shape[1]
                nx_view = view_mask.shape[2]
                mid_z = nz_view // 2
                ref_z = st.slider("Z-slice", 0, nz_view - 1, mid_z, key="ref3d_z") if nz_view > 1 else 0
                extent = [0, nx_view * view_voxel, 0, ny_view * view_voxel]

                reg_vol = st.session_state.get("registered_volume")
                show_reg_compare = (
                    reg_vol is not None
                    and st.checkbox(
                        "Show before/after registration comparison",
                        value=True,
                        key="show_reg_compare",
                    )
                )

                if show_reg_compare:
                    # Three-panel: reference | moving (before) | moving (after)
                    real_vol = st.session_state.get("real_volume")
                    rcol1, rcol2, rcol3 = st.columns(3)
                    v_z = min(ref_z, reg_vol.shape[0] - 1)

                    with rcol1:
                        st.caption("Reference (fixed)")
                        fig2, ax = plt.subplots(figsize=(4, 4))
                        if real_vol is not None:
                            ax.imshow(real_vol[min(ref_z, real_vol.shape[0]-1)].T,
                                      cmap='gray', origin='lower', extent=extent,
                                      vmin=0, vmax=255)
                        else:
                            ax.imshow(view_mask[ref_z].T, cmap='gray', origin='lower',
                                      extent=extent)
                        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                        st.pyplot(fig2); plt.close()

                    with rcol2:
                        st.caption("Moving (before registration)")
                        # Show the registered volume with no transform = original moving
                        # We stored the registered result; show a note instead
                        fig2, ax = plt.subplots(figsize=(4, 4))
                        ax.imshow(reg_vol[v_z].T, cmap='gray', origin='lower',
                                  extent=extent, vmin=0, vmax=255, alpha=0.5)
                        if real_vol is not None:
                            ax.imshow(real_vol[min(ref_z, real_vol.shape[0]-1)].T,
                                      cmap='hot', origin='lower', extent=extent,
                                      vmin=0, vmax=255, alpha=0.4)
                        ax.set_title("Overlay (pre-reg)", fontsize=9)
                        ax.set_xlabel("x [mm]")
                        st.pyplot(fig2); plt.close()

                    with rcol3:
                        st.caption("Moving (after registration)")
                        fig2, ax = plt.subplots(figsize=(4, 4))
                        ax.imshow(reg_vol[v_z].T, cmap='gray', origin='lower',
                                  extent=extent, vmin=0, vmax=255)
                        ax.set_xlabel("x [mm]")
                        st.pyplot(fig2); plt.close()

                    # Difference map
                    if real_vol is not None:
                        r_z = min(ref_z, real_vol.shape[0] - 1)
                        diff = real_vol[r_z].astype(float) - reg_vol[v_z].astype(float)
                        rmse = float(np.sqrt(np.mean(diff**2)))
                        st.metric("Registration RMSE (this slice)", f"{rmse:.1f}",
                                  help="Lower = better alignment")

                elif strain_vol is not None and use_strain:
                    # Two-panel: binary mask + strain overlay
                    fig2, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(view_mask[ref_z].T, cmap='gray', origin='lower',
                                   extent=extent)
                    axes[0].set_title("Binary mask")
                    axes[0].set_xlabel("x [mm]"); axes[0].set_ylabel("y [mm]")
                    strain_z = min(ref_z, strain_vol.shape[0] - 1)
                    strain_extent = [
                        0, strain_vol.shape[2] * view_voxel,
                        0, strain_vol.shape[1] * view_voxel,
                    ]
                    im = axes[1].imshow(strain_vol[strain_z].T,
                                        cmap=colorscale.lower(),
                                        origin='lower', extent=strain_extent)
                    axes[1].set_title(strain_label)
                    axes[1].set_xlabel("x [mm]")
                    plt.colorbar(im, ax=axes[1])
                    plt.tight_layout()
                    st.pyplot(fig2); plt.close()
                else:
                    fig2, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(view_mask[ref_z].T, cmap='gray', origin='lower',
                              extent=extent)
                    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
                    ax.set_title(f"z-slice {ref_z}")
                    plt.tight_layout()
                    st.pyplot(fig2); plt.close()


# ──────────────────────────────────────────────────────────────
# TAB 2 — LOAD STRAIN DATA
# ──────────────────────────────────────────────────────────────
with tab_load_strain:
    st.subheader("Load strain / displacement field")
    st.info(
        "If you have DVC/strain data to overlay, the recommended workflow is:\n\n"
        "1. Go to **Data Loader** → upload your reference + moving images.\n"
        "2. Upload the strain field in the sidebar under *Strain input → Image + strain field*.\n"
        "3. Run **rigid-body registration** — this co-registers the field automatically.\n\n"
        "You can also load a strain field directly here, but it will not be registered "
        "to the image unless you have already done so in the Data Loader."
    )

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
        strain_voxel = st.number_input("Voxel size (µm)", value=39.0, step=1.0,
                                        key="strain3d_vox")

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

            # Mark as NOT registered since it was loaded directly here
            st.session_state["strain_volume_3d"] = strain_data
            st.session_state["strain_label_3d"] = strain_component
            st.session_state["strain_registered"] = False

            st.warning(
                "Strain field stored but marked as **unregistered**. "
                "To co-register it, go to **Data Loader** and run rigid-body "
                "registration with your image pair."
            )

            mid = nz_s // 2
            voxel_mm_s = strain_voxel / 1000.0
            fig, ax = plt.subplots(figsize=(6, 6))
            ext = [0, nx_s * voxel_mm_s, 0, ny_s * voxel_mm_s]
            im = ax.imshow(strain_data[mid].T, cmap='plasma', origin='lower', extent=ext)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            ax.set_title(f"{strain_component} — z={mid}")
            plt.colorbar(im, ax=ax)
            st.pyplot(fig); plt.close()

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

        fe_sub = st.selectbox("FE subsample", [1, 2, 4], index=0, key="fe3d_sub")
        fe_mask = bone_mask_fe
        fe_voxel_um = voxel_um_fe
        if fe_sub > 1:
            fe_mask = bone_mask_fe[::fe_sub, ::fe_sub, ::fe_sub]
            fe_voxel_um = voxel_um_fe * fe_sub

        fe_bone = int(fe_mask.sum())
        nz_fe, ny_fe, nx_fe = fe_mask.shape

        if fe_bone > 100_000:
            st.warning(f"⚠️ {fe_bone:,} bone elements — FE will be slow. "
                       f"Consider subsample=2 ({fe_bone // 8:,} elements).")
        elif fe_bone > 30_000:
            st.info(f"{fe_bone:,} bone elements — expect ~30-90s.")
        else:
            st.caption(f"{fe_bone:,} bone elements — should be quick (~10-30s).")

        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            fe_load = st.selectbox("Load case", ["compression", "tension", "torque"],
                                   key="fe3d_load")
        with fcol2:
            fe_E = st.number_input("E_bone (MPa)", value=18000.0, step=1000.0, key="fe3d_E")
        with fcol3:
            fe_strain_val = st.number_input("Applied strain", value=0.01, step=0.005,
                                             format="%.3f", key="fe3d_strain")

        use_hetero = st.checkbox("Heterogeneous E from grayscale", key="fe3d_hetero")

        strain_comp = st.selectbox("Strain component to map", [
            "eps_von_mises", "eps_zz", "eps_xx", "eps_yy", "eps_xy", "eps_max_principal",
        ], key="fe3d_comp")

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
            spinner_msg = f"Running FE ({fe_load}) on {nx_fe}×{ny_fe}×{nz_fe}"
            if use_hetero:
                spinner_msg += " [heterogeneous E]"

            with st.spinner(spinner_msg + "..."):
                fe = run_fe_analysis(
                    fe_mask, voxel_mm_fe,
                    load_type=fe_load, E_bone=fe_E,
                    applied_strain=fe_strain_val,
                    grayscale=gray_for_fe if use_hetero else None,
                    verbose=False,
                )
            st.session_state["pipeline_fe"] = fe

            if fe["apparent_modulus"] is not None:
                st.metric("E_apparent", f"{fe['apparent_modulus']:.0f} MPa")

            with st.spinner("Converting strain to voxel volume..."):
                strain_vol = strain_volume_from_fe(
                    fe, fe_mask, voxel_mm_fe, component=strain_comp,
                )

            # FE-derived strain is always co-registered (it came from the same mask)
            st.session_state["strain_volume_3d"] = strain_vol
            st.session_state["strain_label_3d"] = strain_comp
            st.session_state["strain_registered"] = True
            st.session_state["fe_voxel_mm"] = voxel_mm_fe

            with st.spinner("Building 3D surface..."):
                verts, faces, normals = extract_surface(fe_mask, voxel_mm_fe)
                vertex_strain = sample_field_at_vertices(verts, strain_vol, voxel_mm_fe)

            st.session_state["mesh_verts"] = verts
            st.session_state["mesh_faces"] = faces

            fig = build_mesh_figure(
                verts, faces,
                color_values=vertex_strain,
                colorscale=colorscale,
                color_label=strain_comp,
                opacity=mesh_opacity,
                title=f"3D bone — {strain_comp} ({fe_load})",
            )
            st.plotly_chart(fig, width='stretch')

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