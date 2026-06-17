"""
pages/4_3d_viewer.py  --  3D bone + strain overlay

Drop-in replacement. Defensive by design:
  * finds the bone volume from session_state, else from data/demo/ files
  * for the overlay, uses data/demo/demo_strain.npy IF its shape matches the
    rendered volume; otherwise builds a matching structured strain field on the
    fly (cheap: numpy + scipy only, no FE, no quantum) so it can never mismatch
  * percentile-clips the colour scale so it always reads as a gradient

The strain shown is a REPRESENTATIVE field illustrating how the pipeline
visualises strain -- not the measured DVC result (that is the D2IM displacement).
"""

import os
import glob
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from skimage import measure

DEMO_DIR = os.path.join("data", "demo")

st.header("3D bone model")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_demo(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(DEMO_DIR, pat)))
        if hits:
            return np.load(hits[0]), os.path.basename(hits[0])
    return None, None


def get_bone_volume():
    """Bone volume from session first, then committed demo files."""
    for k in ["strain_volume_3d", "real_bone_mask", "real_volume",
              "bone_mask", "generated_volume", "volume"]:
        v = st.session_state.get(k)
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.size:
            return np.asarray(v), f"session:{k}"
    v, src = _load_demo(["*bone_mask*.npy", "*mask*.npy", "*reference_scan*.npy"])
    return (v, src) if v is not None else (None, None)


def to_mask(vol):
    """Binary bone mask from a binary or grayscale volume."""
    vol = np.asarray(vol)
    vals = np.unique(vol.ravel()[: min(vol.size, 100000)])
    if vol.dtype == bool or (vals.size <= 3 and set(vals).issubset({0, 1})):
        return (vol > 0).astype(np.uint8)
    # grayscale -> Otsu threshold
    try:
        from skimage.filters import threshold_otsu
        t = threshold_otsu(vol)
    except Exception:
        t = np.percentile(vol, 60)
    return (vol > t).astype(np.uint8)


def build_strain(mask, peak=0.012):
    """Smooth structured strain field, gradient + hot zone in the broad plane."""
    m = (mask > 0).astype(np.float32)
    shp = m.shape
    density = gaussian_filter(m, sigma=max(shp) * 0.03)
    density /= (density.max() + 1e-8)
    axes = np.argsort(shp)                      # smallest..largest
    thin = int(axes[0])
    a1, a2 = sorted(int(a) for a in axes[1:])   # the two broad axes
    grids = np.meshgrid(*[np.linspace(0, 1, s) for s in shp], indexing="ij")
    u, w = grids[a1], grids[a2]
    dens_plane = density.mean(axis=thin)
    ci, cj = np.unravel_index(int(np.argmax(dens_plane)), dens_plane.shape)
    cu, cw = ci / dens_plane.shape[0], cj / dens_plane.shape[1]
    field = (
        0.8 * u
        + 1.2 * density
        + 0.9 * np.exp(-(((u - cu) ** 2 + (w - cw) ** 2) / 0.02))
        + 0.10 * np.sin(5 * np.pi * w)
    )
    field = gaussian_filter(field, sigma=1.0) * m
    v = field[m > 0]
    field = np.where(m > 0, (field - v.min()) / (v.max() - v.min() + 1e-8) * peak, 0.0)
    return field.astype(np.float32)


def get_overlay(mask):
    """Prefer a precomputed demo strain that matches the volume; else build one."""
    p = os.path.join(DEMO_DIR, "demo_strain.npy")
    if os.path.exists(p):
        s = np.load(p)
        if s.shape == mask.shape:
            return s, "Strain (demo field)"
    return build_strain(mask), "Strain (demo field)"


def sample_at_vertices(vol, verts):
    """Nearest-voxel sample of a volume at marching-cubes vertices."""
    idx = np.round(verts).astype(int)
    for ax in range(3):
        idx[:, ax] = np.clip(idx[:, ax], 0, vol.shape[ax] - 1)
    return vol[idx[:, 0], idx[:, 1], idx[:, 2]]


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
vol, src = get_bone_volume()
if vol is None:
    st.warning("No bone volume found. Click **Load demo data** on the Data Loader "
               "page first, or add a mask to data/demo/.")
    st.stop()

mask = to_mask(vol)
voxel_um = float(st.session_state.get("real_voxel_um", 50))   # 50 um for the S9 specimen
vox_mm = voxel_um / 1000.0
bvtv = float(mask.mean())

st.success(f"Bone mask: {mask.shape[0]}×{mask.shape[1]}×{mask.shape[2]}  "
           f"(voxel = {voxel_um:.0f} µm, BV/TV = {bvtv:.3f})   ·  source: {src}")

# ---------------------------------------------------------------------------
# optional clip to reveal interior (only on a broad axis, never the thin slab)
# ---------------------------------------------------------------------------
overlay, color_label = get_overlay(mask)

clip = st.checkbox("Clip at 50% to reveal interior", value=False)
if clip:
    clip_ax = int(np.argmax(mask.shape))        # clip the LARGEST axis
    if mask.shape[clip_ax] > 20:
        keep = mask.shape[clip_ax] // 2
        sl = [slice(None)] * 3
        sl[clip_ax] = slice(0, keep)
        mask = mask[tuple(sl)]
        overlay = overlay[tuple(sl)]
        st.caption(f"Clipped axis {clip_ax} at 50% → shape {mask.shape}")

# ---------------------------------------------------------------------------
# mesh + render
# ---------------------------------------------------------------------------
if mask.sum() < 10:
    st.error("Mask is empty after thresholding — check the input volume.")
    st.stop()

verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5)
st.caption(f"{len(verts):,} vertices, {len(faces):,} triangles")

vertex_vals = sample_at_vertices(overlay, verts)

finite = vertex_vals[np.isfinite(vertex_vals) & (vertex_vals != 0)]
if finite.size:
    cmin = float(np.percentile(finite, 2))
    cmax = float(np.percentile(finite, 98))
else:
    cmin, cmax = 0.0, 1.0
if cmax <= cmin:
    cmax = cmin + 1e-6

x = verts[:, 0] * vox_mm
y = verts[:, 1] * vox_mm
z = verts[:, 2] * vox_mm

st.subheader("3D bone + strain (registered)")
fig = go.Figure(
    data=[go.Mesh3d(
        x=x, y=y, z=z,
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=vertex_vals,
        cmin=cmin, cmax=cmax,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title=color_label),
        flatshading=True,
        lighting=dict(ambient=0.55, diffuse=0.8, specular=0.2),
    )]
)
fig.update_layout(
    template="plotly_dark",
    height=640,
    margin=dict(l=0, r=0, t=0, b=0),
    scene=dict(
        xaxis_title="x [mm]", yaxis_title="y [mm]", zaxis_title="z [mm]",
        aspectmode="data",
    ),
)
st.plotly_chart(fig, use_container_width=True)