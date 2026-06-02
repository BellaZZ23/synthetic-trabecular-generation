"""
Page 2: FE Solver
==================
Run compression, tension, or torque on a bone volume.
Two solvers available:
  - TechMesh (scikit-fem) — tetrahedral FE, faster, heterogeneous E
  - Voxel FE (built-in)   — original voxel-based solver

Results are pushed to session state so the 3D viewer picks them up
immediately without re-running.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fe_coupling"))
from step3_generator_fe_coupling import generate_bone_volume, run_fe_analysis

try:
    from techmesh_solver import run_techmesh_analysis
    HAS_TECHMESH = True
except ImportError:
    HAS_TECHMESH = False

st.set_page_config(page_title="FE solver", page_icon="🔧", layout="wide")
st.title("Micro-FE solver")
st.caption("Uniaxial compression, tension, and torque with strain field extraction")


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def nodal_to_slice(mesh, values, voxel_mm, z_idx, nx, ny):
    out = np.full((nx, ny), np.nan)
    tol = voxel_mm * 0.1
    for n_idx in range(mesh.nvertices):
        x, y, z = mesh.p[:, n_idx]
        if abs(z - z_idx*voxel_mm) < tol or abs(z - (z_idx+1)*voxel_mm) < tol:
            i = int(round(x / voxel_mm))
            j = int(round(y / voxel_mm))
            if 0 <= i < nx and 0 <= j < ny:
                if np.isnan(out[i, j]):
                    out[i, j] = values[n_idx]
    return out


def element_to_slice(values, centroids, nx, ny, voxel_mm, mid_z_mm):
    out = np.full((nx, ny), np.nan)
    z_range = voxel_mm * 0.6
    for e in range(len(values)):
        if abs(centroids[e, 2] - mid_z_mm) < z_range:
            ci = int(round(centroids[e, 0] / voxel_mm))
            cj = int(round(centroids[e, 1] / voxel_mm))
            if 0 <= ci < nx and 0 <= cj < ny:
                out[ci, cj] = values[e]
    return out


def strain_vol_from_fe(fe, bone_mask, voxel_mm, component='eps_von_mises'):
    """Convert element strain to a voxel volume for 3D viewer."""
    strain = fe["strain_field"]
    centroids = strain["centroids"]
    values = strain[component]
    nz, ny, nx = bone_mask.shape
    vol = np.full((nz, ny, nx), np.nan, dtype=float)
    for idx in range(len(values)):
        ci = int(round(centroids[idx, 0] / voxel_mm))
        cj = int(round(centroids[idx, 1] / voxel_mm))
        ck = int(round(centroids[idx, 2] / voxel_mm))
        if 0 <= ci < nx and 0 <= cj < ny and 0 <= ck < nz:
            vol[ck, cj, ci] = values[idx]
    filled = vol.copy()
    filled[np.isnan(filled)] = 0
    vol = np.where(np.isnan(vol), filled, vol)
    return vol


# ══════════════════════════════════════════════════════════════
# VOLUME SOURCE
# ══════════════════════════════════════════════════════════════

st.sidebar.header("Volume source")
has_generated = "bone_volume" in st.session_state
has_real      = "real_bone_mask" in st.session_state

source_options = []
if has_generated:
    source_options.append("Generated (from generator page)")
if has_real:
    source_options.append("Real µCT (from data loader)")
if not source_options:
    source_options.append("Quick generate here")

vol_source = st.sidebar.radio("Use volume from", source_options)

# ── Volume selection ──
bone_mask = None
voxel_um  = 39.0

if "Generated" in vol_source and has_generated:
    vol_data  = st.session_state["bone_volume"]
    bone_mask = vol_data["bone_mask"]
    voxel_um  = vol_data["voxel_um"]
    morph     = vol_data["morphometrics"]

elif "Real" in vol_source and has_real:
    bone_mask = st.session_state["real_bone_mask"]
    voxel_um  = st.session_state.get("real_voxel_um", 39.0)
    morph     = None

elif "Quick generate" in vol_source:
    st.subheader("Quick generate")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    q_bvtv = qcol1.slider("BV/TV", 0.10, 0.45, 0.30, 0.01, key="q_bvtv")
    q_xy   = qcol2.selectbox("XY", [16, 32, 48, 64], index=1, key="q_xy")
    q_z    = qcol3.selectbox("Z",  [8, 16, 24, 32],  index=1, key="q_z")
    q_seed = qcol4.number_input("Seed", value=42, key="q_seed")
    if st.button("Generate + continue", type="primary"):
        with st.spinner("Generating..."):
            vol_data = generate_bone_volume(
                nx=q_xy, ny=q_xy, nz=q_z,
                target_bvtv=q_bvtv, seed=int(q_seed), verbose=False,
            )
        st.session_state["bone_volume"] = vol_data
        st.rerun()
    st.stop()

if bone_mask is None:
    st.info("No bone volume available. Generate one in the Generator page or "
            "load a scan in the Data Loader.")
    st.stop()

voxel_mm = voxel_um / 1000.0
nz, ny, nx = bone_mask.shape

# ── Volume info ──
ic1, ic2, ic3, ic4 = st.columns(4)
ic1.metric("Volume", f"{nx}×{ny}×{nz}")
ic2.metric("BV/TV", f"{bone_mask.mean():.3f}")
if morph:
    ic3.metric("Tb.Th", f"{morph['TbTh_um_p50']:.0f} µm")
    ic4.metric("LCC",   f"{morph['lcc_frac']:.3f}")
else:
    ic3.metric("Voxel", f"{voxel_um:.0f} µm")
    ic4.metric("Source", "real µCT")

st.divider()


# ══════════════════════════════════════════════════════════════
# SOLVER SETTINGS SIDEBAR
# ══════════════════════════════════════════════════════════════

st.sidebar.header("Solver")

if HAS_TECHMESH:
    solver_choice = st.sidebar.radio(
        "Engine",
        ["TechMesh (scikit-fem tet)", "Voxel FE (built-in)"],
        help=(
            "TechMesh: tetrahedral mesh via Kuhn subdivision. "
            "Faster and supports heterogeneous E from grayscale.\n\n"
            "Voxel FE: original voxel-based solver."
        ),
    )
    use_techmesh = "TechMesh" in solver_choice
else:
    st.sidebar.info("TechMesh not installed. Run: pip install scikit-fem meshio")
    use_techmesh = False

st.sidebar.header("FE parameters")
load_type = st.sidebar.radio("Load case", ["compression", "tension", "torque"])

E_bone = st.sidebar.number_input("E_bone (MPa)", value=18000.0, step=1000.0)
nu     = st.sidebar.number_input("Poisson ratio", value=0.3, step=0.05,
                                  min_value=0.0, max_value=0.49)

if load_type == "torque":
    strain_deg = st.sidebar.slider("Rotation (deg)", 0.1, 5.0, 0.57, 0.01)
    applied_strain = float(np.radians(strain_deg))
    st.sidebar.caption(f"= {applied_strain:.4f} rad")
else:
    applied_strain = st.sidebar.slider("Applied strain", 0.001, 0.05, 0.01, 0.001,
                                        format="%.3f")

# ── Heterogeneous E ──
use_hetero = False
if use_techmesh:
    st.sidebar.header("Material")
    use_hetero = st.sidebar.checkbox(
        "Heterogeneous E from grayscale",
        value=False,
        help=(
            "Maps grayscale intensity → per-element Young's modulus via "
            "power law: E = E_bone × (I/255)^n. "
            "Requires a grayscale volume in session."
        ),
    )
    if use_hetero:
        e_power = st.sidebar.slider("Power law exponent (n)", 1.0, 3.0, 2.0, 0.1,
            help="n=2 is standard for trabecular bone (Carter & Hayes 1977).")
        e_min   = st.sidebar.number_input("E_min (MPa)", value=100.0, step=50.0)

        gray_available = (
            "pipeline_gray" in st.session_state or
            "real_volume"   in st.session_state
        )
        if gray_available:
            st.sidebar.success("Grayscale found in session")
        else:
            st.sidebar.warning(
                "No grayscale in session. Generate one in the Generator "
                "or Pipeline page, or load a scan in the Data Loader."
            )
            use_hetero = False

# ── Subsample for large volumes ──
st.sidebar.header("Performance")
n_bone = int(bone_mask.sum())
if n_bone > 100_000:
    st.sidebar.warning(f"{n_bone:,} bone voxels — consider subsampling.")
fe_sub = st.sidebar.selectbox("Subsample factor", [1, 2, 4], index=0,
    help="2 = half resolution, 8× fewer elements.")

# ── Strain component to push to 3D viewer ──
st.sidebar.header("3D viewer output")
push_component = st.sidebar.selectbox(
    "Strain to push to 3D viewer",
    ["eps_von_mises", "eps_zz", "eps_xx", "eps_yy",
     "eps_xy", "eps_max_principal"],
)

# ── Run button ──
total_vox = nx * ny * nz
est = ("~10-30s" if total_vox < 50_000
       else "~30-90s" if total_vox < 200_000
       else "~2-5min")
run_btn = st.sidebar.button(
    f"Run {load_type}",
    type="primary",
    use_container_width=True,
    help=f"{n_bone:,} bone voxels — {est}",
)


# ══════════════════════════════════════════════════════════════
# RUN FE
# ══════════════════════════════════════════════════════════════

if run_btn:

    # Apply subsample
    run_mask   = bone_mask[::fe_sub, ::fe_sub, ::fe_sub]
    run_voxel  = voxel_mm * fe_sub
    run_gray   = None

    if use_hetero:
        gray_src = (st.session_state.get("pipeline_gray") or
                    st.session_state.get("real_volume"))
        if gray_src is not None:
            run_gray = gray_src[::fe_sub, ::fe_sub, ::fe_sub]

    if use_techmesh:
        with st.spinner(
            f"TechMesh: building tet mesh + solving "
            f"({load_type}, {run_mask.shape})..."
        ):
            fe = run_techmesh_analysis(
                run_mask, run_voxel,
                load_type=load_type,
                E_bone=E_bone,
                nu=nu,
                applied_strain=applied_strain,
                grayscale=run_gray if use_hetero else None,
                E_min=e_min   if use_hetero else 100.0,
                E_power=e_power if use_hetero else 2.0,
                verbose=False,
            )
    else:
        with st.spinner(f"Voxel FE: running {load_type} ({run_mask.shape})..."):
            fe = run_fe_analysis(
                run_mask, run_voxel,
                load_type=load_type,
                E_bone=E_bone,
                nu=nu,
                applied_strain=applied_strain,
                verbose=False,
            )

    st.session_state["fe_results"]    = fe
    st.session_state["fe_bone_mask"]  = run_mask
    st.session_state["fe_voxel_mm"]   = run_voxel

    # Push strain volume to 3D viewer session
    strain_vol = strain_vol_from_fe(fe, run_mask, run_voxel, push_component)
    st.session_state["strain_volume_3d"] = strain_vol
    st.session_state["strain_label_3d"]  = push_component
    st.session_state["strain_registered"] = True
    st.session_state["fe_voxel_mm_3d"]   = run_voxel

    st.success(
        f"Done — {fe['n_elements']:,} elements | "
        f"solve {fe['solve_time']:.1f}s | "
        f"solver: {fe.get('solver','voxel')}"
    )


# ══════════════════════════════════════════════════════════════
# RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════

if "fe_results" not in st.session_state:
    st.info("Configure the solver in the sidebar and click Run.")
    st.stop()

fe        = st.session_state["fe_results"]
run_mask  = st.session_state.get("fe_bone_mask", bone_mask)
run_voxel = st.session_state.get("fe_voxel_mm",  voxel_mm)
ux, uy, uz = fe["displacement"]
strain     = fe["strain_field"]
nz_r, ny_r, nx_r = run_mask.shape
lt = fe["load_type"]

# ── Summary metrics ──
st.subheader(f"Results — {lt}  ·  solver: {fe.get('solver','voxel')}")

rc1, rc2, rc3, rc4, rc5 = st.columns(5)
rc1.metric("Elements",   f"{fe['n_elements']:,}")
rc2.metric("Nodes",      f"{fe.get('n_nodes', '—'):,}")
rc3.metric("Solve time", f"{fe['solve_time']:.1f}s")

if lt in ("compression", "tension") and fe["apparent_modulus"] is not None:
    E_app = fe["apparent_modulus"]
    voigt = fe["voigt_bound"]
    rc4.metric("E_apparent", f"{E_app:.0f} MPa")
    ratio = E_app / voigt if voigt > 0 else 0
    rc5.metric("E / Voigt", f"{ratio:.3f}",
               "PASS" if ratio <= 1.01 else "FAIL")
elif lt == "torque" and fe.get("apparent_shear_modulus") is not None:
    rc4.metric("G_apparent", f"{fe['apparent_shear_modulus']:.0f} MPa")
    rc5.metric("Applied", f"{np.degrees(applied_strain):.2f}°")

# Heterogeneous E summary
if fe.get("E_elem") is not None:
    E_e = fe["E_elem"]
    st.info(
        f"Heterogeneous E — mean: {E_e.mean():.0f} MPa  |  "
        f"min: {E_e.min():.0f}  max: {E_e.max():.0f} MPa  |  "
        f"std: {E_e.std():.0f} MPa"
    )

st.divider()

# ── 3D viewer shortcut ──
st.success(
    f"Strain field ({push_component}) pushed to 3D viewer. "
    "Switch to the **3D viewer** page to see it on the surface."
)

st.divider()

# ── Slice selector ──
mid_z = nz_r // 2
slice_idx = st.slider("Z-slice", 0, nz_r - 1, mid_z, key="fe_slice")
mid_z_mm  = (slice_idx + 0.5) * run_voxel
extent    = [0, nx_r * run_voxel, 0, ny_r * run_voxel]

has_mesh = hasattr(fe.get("mesh"), "nvertices")


# ── Row 1: Structure + displacement ──
st.subheader("Displacement fields")
d1, d2, d3, d4 = st.columns(4)

with d1:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(run_mask[slice_idx].T, cmap='gray', origin='lower', extent=extent)
    ax.set_title(f"Bone (z={slice_idx})")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    st.pyplot(fig); plt.close()

def _disp_slice(component_vals):
    if has_mesh:
        return nodal_to_slice(fe["mesh"], component_vals,
                               run_voxel, slice_idx, nx_r, ny_r)
    else:
        # Voxel solver: reshape displacement to volume
        try:
            vol = component_vals.reshape(nz_r+1, ny_r, nx_r)
            return vol[slice_idx]
        except Exception:
            return np.full((nx_r, ny_r), np.nan)

with d2:
    mag = np.sqrt(ux**2 + uy**2 + uz**2)
    sl  = _disp_slice(mag)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(sl.T, cmap='plasma', origin='lower', extent=extent)
    ax.set_title("|u| magnitude [mm]"); ax.set_xlabel("x [mm]")
    plt.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig); plt.close()

with d3:
    sl = _disp_slice(uz)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(sl.T, cmap='RdBu', origin='lower', extent=extent)
    ax.set_title("uz [mm]"); ax.set_xlabel("x [mm]")
    plt.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig); plt.close()

with d4:
    sl = _disp_slice(ux)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(sl.T, cmap='RdBu', origin='lower', extent=extent)
    ax.set_title("ux [mm]"); ax.set_xlabel("x [mm]")
    plt.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig); plt.close()


# ── Row 2: Strain fields ──
st.subheader("Strain fields")
s1, s2, s3 = st.columns(3)

def _strain_slice(key):
    return element_to_slice(
        strain[key], strain["centroids"],
        nx_r, ny_r, run_voxel, mid_z_mm
    )

with s1:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(_strain_slice("eps_zz").T,
                   cmap='RdBu_r', origin='lower', extent=extent)
    ax.set_title("Axial strain (ε_zz)")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    plt.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig); plt.close()

with s2:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(_strain_slice("eps_von_mises").T,
                   cmap='inferno', origin='lower', extent=extent)
    ax.set_title("von Mises strain")
    ax.set_xlabel("x [mm]")
    plt.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig); plt.close()

with s3:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(_strain_slice("eps_max_principal").T,
                   cmap='magma', origin='lower', extent=extent)
    ax.set_title("Max principal strain")
    ax.set_xlabel("x [mm]")
    plt.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig); plt.close()


# ── Heterogeneous E map ──
if fe.get("E_elem") is not None:
    st.subheader("Heterogeneous E map")
    E_slice = element_to_slice(
        fe["E_elem"], strain["centroids"],
        nx_r, ny_r, run_voxel, mid_z_mm
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(E_slice.T, cmap='hot', origin='lower', extent=extent)
    axes[0].set_title(f"E per element (z={slice_idx})")
    axes[0].set_xlabel("x [mm]"); axes[0].set_ylabel("y [mm]")
    plt.colorbar(im0, ax=axes[0], label="E (MPa)")

    axes[1].hist(fe["E_elem"], bins=80, color='#378ADD',
                 alpha=0.8, edgecolor='none', density=True)
    axes[1].set_xlabel("E (MPa)"); axes[1].set_ylabel("Density")
    axes[1].set_title("E distribution across elements")
    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ── Strain distribution ──
with st.expander("Strain distributions"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, label, color in [
        (axes[0], "eps_zz",          "ε_zz",        "#378ADD"),
        (axes[1], "eps_von_mises",   "von Mises",   "#E85D3A"),
        (axes[2], "eps_max_principal","Max principal","#7B2D8E"),
    ]:
        ax.hist(strain[key], bins=80, color=color, alpha=0.8,
                edgecolor='none', density=True)
        ax.set_title(label); ax.set_xlabel("Strain"); ax.set_ylabel("Density")
    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ── Statistics ──
with st.expander("Strain statistics"):
    scol1, scol2 = st.columns(2)
    with scol1:
        st.json({k: {"min": round(float(strain[k].min()), 6),
                     "max": round(float(strain[k].max()), 6),
                     "mean": round(float(strain[k].mean()), 6)}
                 for k in ["eps_xx","eps_yy","eps_zz"]})
    with scol2:
        st.json({k: {"min": round(float(strain[k].min()), 6),
                     "max": round(float(strain[k].max()), 6),
                     "mean": round(float(strain[k].mean()), 6)}
                 for k in ["eps_xy","eps_von_mises","eps_max_principal"]})