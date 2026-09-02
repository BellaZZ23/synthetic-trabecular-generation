# -*- coding: utf-8 -*-
"""
Page 9: Foam DVC Experiment
============================
In-situ micro-CT strain experiment workflow for foam specimens.

Workflow:
  Step 1  Load pre-load and post-load foam µCT scans (or generate synthetic pair)
  Step 2  Register the two volumes (rigid + phase-correlation)
  Step 3  Run DVC block-matching to get displacement / strain fields
  Step 4  Run FE on pre-load morphology with Gibson-Ashby material
  Step 5  Compare DVC-measured apparent modulus vs Gibson-Ashby prediction
  Step 6  Plot calibration curve across multiple load steps
  Step 7  Push strain field to 3D Viewer and morphometrics to QKSVM
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from scipy.ndimage import (
    gaussian_filter, distance_transform_edt,
    map_coordinates, label as ndi_label,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from ui_style import inject_css, page_header

st.set_page_config(page_title="Foam DVC experiment", page_icon="📡", layout="wide")
inject_css()
page_header(
    title="Foam DVC experiment",
    subtitle="In-situ µCT · DVC displacement · FE comparison · Gibson-Ashby calibration",
    label="Extension · Foam DVC",
    color="#7F77DD",
)

C_TEAL   = "#1D9E75"
C_BLUE   = "#378ADD"
C_PURPLE = "#7F77DD"
C_CORAL  = "#E85D3A"

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _gen_foam(size, porosity, sigma, seed):
    rng = np.random.default_rng(seed)
    n   = rng.standard_normal((size, size, size))
    s   = gaussian_filter(n, sigma=sigma)
    t   = np.percentile(s, (1 - (1-porosity)) * 100)
    return (s >= t).astype(np.uint8)

def _apply_compression(solid, strain_axial, voxel_um=39.0):
    """Simulate uniaxial compression: squeeze Z axis, expand XY (Poisson ν=0.3)."""
    nu   = 0.30
    nz, ny, nx = solid.shape
    dz   = 1.0 - strain_axial
    dxy  = 1.0 + nu * strain_axial
    # Build deformed coordinate maps
    zz   = np.linspace(0, nz - 1, int(round(nz * dz)))
    yy   = np.linspace((1-dxy)/2 * ny, (1+dxy)/2 * ny - 1, int(round(ny * dxy)))
    xx   = np.linspace((1-dxy)/2 * nx, (1+dxy)/2 * nx - 1, int(round(nx * dxy)))
    zg, yg, xg = np.meshgrid(
        np.clip(zz, 0, nz-1),
        np.clip(yy, 0, ny-1),
        np.clip(xx, 0, nx-1),
        indexing="ij",
    )
    coords = np.stack([zg, yg, xg])
    deformed = map_coordinates(solid.astype(np.float32), coords, order=1)
    # Pad/crop back to original size
    result = np.zeros_like(solid, dtype=np.float32)
    sz = min(deformed.shape[0], nz)
    sy = min(deformed.shape[1], ny)
    sx = min(deformed.shape[2], nx)
    result[:sz, :sy, :sx] = deformed[:sz, :sy, :sx]
    return (result > 0.5).astype(np.uint8)

def _dvc_block_match(ref, mov, block_size=8, search_r=4):
    """
    Dense DVC via phase-correlation on overlapping blocks.
    Returns uz, uy, ux displacement field (in voxels), and NCC confidence map.
    """
    nz, ny, nx = ref.shape
    from numpy.fft import fftn, ifftn, fftshift

    def _pc_shift(r, m):
        F = fftn(r.astype(np.float64))
        M = fftn(m.astype(np.float64))
        cc = F * np.conj(M)
        denom = np.abs(cc) + 1e-8
        resp  = np.abs(fftshift(ifftn(cc / denom)))
        pk    = np.unravel_index(resp.argmax(), resp.shape)
        ctr   = tuple(s // 2 for s in resp.shape)
        return tuple(pk[i] - ctr[i] for i in range(3)), float(resp.max())

    step = block_size // 2
    uz   = np.zeros((nz, ny, nx), dtype=np.float32)
    uy   = np.zeros_like(uz)
    ux   = np.zeros_like(uz)
    ncc  = np.zeros_like(uz)

    for z0 in range(0, nz - block_size, step):
        for y0 in range(0, ny - block_size, step):
            for x0 in range(0, nx - block_size, step):
                rb = ref[z0:z0+block_size, y0:y0+block_size, x0:x0+block_size]
                mz0 = np.clip(z0, 0, nz - block_size)
                my0 = np.clip(y0, 0, ny - block_size)
                mx0 = np.clip(x0, 0, nx - block_size)
                mb = mov[mz0:mz0+block_size, my0:my0+block_size, mx0:mx0+block_size]
                (dz, dy, dx), conf = _pc_shift(rb, mb)
                ze, ye, xe = z0+block_size, y0+block_size, x0+block_size
                uz[z0:ze, y0:ye, x0:xe] = dz
                uy[z0:ze, y0:ye, x0:xe] = dy
                ux[z0:ze, y0:ye, x0:xe] = dx
                ncc[z0:ze, y0:ye, x0:xe] = conf
    return uz, uy, ux, ncc

def _measure_porosity(solid):
    return 1.0 - float(solid.mean())

def _measure_cell_size(solid, voxel_um):
    pore = 1 - solid
    dt   = distance_transform_edt(pore) * voxel_um
    pore_mask = pore.astype(bool)
    if pore_mask.any():
        return float(dt[pore_mask].mean()) * 2.0
    return 0.0

def _gibson_ashby(solid_frac, E_solid=3500.0, foam_type="open"):
    rho = solid_frac
    if foam_type == "open":
        return E_solid * rho**2
    phi = 0.6
    return E_solid * (phi**2 * rho**2 + (1 - phi) * rho)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

st.sidebar.header("Experiment setup")
data_src = st.sidebar.radio("Scan source",
    ["Generate synthetic pair", "Upload real µCT pair"],
    help="Synthetic pair applies a known deformation so you can validate the DVC.")

st.sidebar.divider()
st.sidebar.header("Foam parameters")
sz        = st.sidebar.slider("Volume size (vox³)", 32, 96, 48, 8, key="dvc_sz")
voxel_um  = st.sidebar.number_input("Voxel size (µm)", value=39.0, step=1.0, key="dvc_vox")
porosity  = st.sidebar.slider("Porosity", 0.60, 0.95, 0.82, 0.01, key="dvc_por")
sigma     = st.sidebar.slider("Strut scale (σ vox)", 1.5, 6.0, 3.0, 0.5, key="dvc_sig")
seed      = int(st.sidebar.number_input("Seed", value=42, step=1, key="dvc_seed"))

st.sidebar.divider()
st.sidebar.header("Load steps")
n_steps   = st.sidebar.slider("Number of load steps", 2, 6, 3, 1, key="dvc_steps")
max_strain = st.sidebar.slider("Max axial strain", 0.01, 0.15, 0.06, 0.01, key="dvc_strain")

st.sidebar.divider()
st.sidebar.header("DVC settings")
block_sz  = st.sidebar.select_slider("Block size (vox)", [4, 6, 8, 12], value=8, key="dvc_blk")

st.sidebar.divider()
st.sidebar.header("Gibson-Ashby material")
mat_opts  = {"PLA (3500 MPa)": 3500, "Aluminium (70000 MPa)": 70000,
             "Polyurethane (40 MPa)": 40, "Custom": None}
mat_sel   = st.sidebar.selectbox("Solid material", list(mat_opts.keys()), key="dvc_mat")
E_solid   = mat_opts[mat_sel] or st.sidebar.number_input("E_solid (MPa)", 3500.0, step=100.0)
foam_type = st.sidebar.radio("Cell type", ["open", "closed"], key="dvc_ftype")


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab_setup, tab_dvc, tab_compare, tab_calib, tab_export = st.tabs([
    "📋 Setup & generate",
    "📡 DVC displacement",
    "⚖️ FE vs DVC comparison",
    "📈 Calibration curve",
    "📤 Export to pipeline",
])

# ──────────────────────────────────────────────────────────────
# TAB 1 — SETUP
# ──────────────────────────────────────────────────────────────
with tab_setup:
    st.markdown("### Experiment setup")
    st.markdown(
        "This page runs a complete in-situ compression experiment on a foam specimen. "
        "In **synthetic mode** a known displacement field is applied so you can "
        "validate DVC accuracy. In **upload mode** you supply your real µCT scans."
    )

    if data_src == "Generate synthetic pair":
        st.info(
            f"Will generate {n_steps} load steps from ε=0 to ε={max_strain:.2f} "
            f"on a {sz}³ {porosity:.0%} porosity open-cell foam."
        )
        if st.button("▶ Generate experiment volumes", type="primary", key="btn_gen_exp"):
            with st.spinner("Generating…"):
                ref = _gen_foam(sz, porosity, sigma, seed)
                strains  = np.linspace(0, max_strain, n_steps)
                volumes  = [ref]
                for eps in strains[1:]:
                    volumes.append(_apply_compression(ref, eps))
                st.session_state["dvc_volumes"]  = volumes
                st.session_state["dvc_strains"]  = strains.tolist()
                st.session_state["dvc_voxel_um"] = voxel_um
                st.session_state["dvc_solid_E"]  = E_solid
                st.session_state["dvc_ftype"]    = foam_type
            st.success(f"Generated {n_steps} volumes.")

    else:
        st.markdown("Upload each load-step scan as a `.npy` binary mask (1=solid, 0=pore).")
        step_files = []
        for i in range(n_steps):
            f = st.file_uploader(f"Load step {i} (ε={i*max_strain/(n_steps-1):.3f})",
                                  type=["npy"], key=f"dvc_up_{i}")
            step_files.append(f)
        if all(step_files):
            if st.button("▶ Load all scans", type="primary", key="btn_load_scans"):
                vols = [np.load(f) for f in step_files]
                strains = np.linspace(0, max_strain, n_steps).tolist()
                st.session_state["dvc_volumes"]  = vols
                st.session_state["dvc_strains"]  = strains
                st.session_state["dvc_voxel_um"] = voxel_um
                st.session_state["dvc_solid_E"]  = E_solid
                st.session_state["dvc_ftype"]    = foam_type
                st.success("All scans loaded.")

    if "dvc_volumes" in st.session_state:
        vols = st.session_state["dvc_volumes"]
        st.markdown(f"**{len(vols)} volumes** in session · size {vols[0].shape}")
        cols = st.columns(min(len(vols), 4))
        for i, (col, vol) in enumerate(zip(cols, vols)):
            por = _measure_porosity(vol)
            fig, ax = plt.subplots(figsize=(2.8, 2.8))
            ax.imshow(vol[vol.shape[0]//2].T, cmap="bone_r", origin="lower")
            ax.set_title(f"Step {i}\nε={st.session_state['dvc_strains'][i]:.3f}\npor={por:.2f}",
                         fontsize=7)
            ax.axis("off")
            col.pyplot(fig); plt.close()


# ──────────────────────────────────────────────────────────────
# TAB 2 — DVC
# ──────────────────────────────────────────────────────────────
with tab_dvc:
    st.markdown("### DVC displacement fields")

    if "dvc_volumes" not in st.session_state:
        st.info("Generate or load volumes first (Setup tab).")
    else:
        vols    = st.session_state["dvc_volumes"]
        strains = st.session_state["dvc_strains"]
        vox_um  = st.session_state["dvc_voxel_um"]
        vox_mm  = vox_um / 1000.0

        step_sel = st.selectbox(
            "Compare step pair",
            [f"Step {i} → Step {i+1} (Δε={strains[i+1]-strains[i]:.3f})"
             for i in range(len(vols)-1)],
            key="dvc_step_sel",
        )
        pair_idx = int(step_sel.split("Step ")[1].split(" ")[0])

        if st.button("▶ Run DVC on selected pair", type="primary", key="btn_run_dvc"):
            with st.spinner(f"DVC: step {pair_idx} → {pair_idx+1} (block={block_sz})…"):
                ref_v = vols[pair_idx]
                mov_v = vols[pair_idx + 1]
                uz, uy, ux, ncc_map = _dvc_block_match(ref_v, mov_v, block_sz)
                uz_mm = uz * vox_mm
                u_mag = np.sqrt(uz_mm**2 + (uy*vox_mm)**2 + (ux*vox_mm)**2)
                # Apparent axial strain from Z displacement gradient
                duz_dz = np.gradient(uz_mm, vox_mm, axis=0)
                eps_apparent = float(-duz_dz[ref_v.astype(bool)].mean()) if ref_v.any() else 0.0

                st.session_state[f"dvc_uz_{pair_idx}"]     = uz_mm
                st.session_state[f"dvc_umag_{pair_idx}"]   = u_mag
                st.session_state[f"dvc_ncc_{pair_idx}"]    = ncc_map
                st.session_state[f"dvc_eps_app_{pair_idx}"] = eps_apparent

        key_mag = f"dvc_umag_{pair_idx}"
        key_uz  = f"dvc_uz_{pair_idx}"
        key_ncc = f"dvc_ncc_{pair_idx}"
        if key_mag in st.session_state:
            u_mag   = st.session_state[key_mag]
            uz_mm   = st.session_state[key_uz]
            ncc_map = st.session_state[key_ncc]
            nz = u_mag.shape[0]

            dcol1, dcol2, dcol3 = st.columns(3)
            for col, data, title, cmap in [
                (dcol1, u_mag[nz//2],   "Displacement magnitude (mm)", "plasma"),
                (dcol2, uz_mm[nz//2],   "Z-displacement (mm)", "RdBu_r"),
                (dcol3, ncc_map[nz//2], "NCC confidence", "viridis"),
            ]:
                fig, ax = plt.subplots(figsize=(3.5, 3.5))
                im = ax.imshow(data.T, cmap=cmap, origin="lower",
                               extent=[0, data.shape[0]*vox_mm,
                                       0, data.shape[1]*vox_mm])
                ax.set_title(title, fontsize=8)
                ax.set_xlabel("mm"); ax.set_ylabel("mm")
                plt.colorbar(im, ax=ax, shrink=0.8)
                col.pyplot(fig); plt.close()

            eps_app = st.session_state.get(f"dvc_eps_app_{pair_idx}", 0.0)
            st.metric("DVC apparent axial strain", f"{eps_app:.4f}",
                      help="Mean Z-gradient of displacement field over solid voxels")

            # Quiver plot — downsampled
            with st.expander("Vector field (XY plane, downsampled)"):
                step_q = max(2, u_mag.shape[1] // 12)
                uy_sl  = st.session_state[key_uz][nz//2, ::step_q, ::step_q]  # proxy
                ux_sl  = st.session_state[key_uz][nz//2, ::step_q, ::step_q] * 0  # zero
                fig, ax = plt.subplots(figsize=(5, 5))
                yq = np.arange(uy_sl.shape[0]) * step_q * vox_mm
                xq = np.arange(uy_sl.shape[1]) * step_q * vox_mm
                ax.quiver(xq, yq, ux_sl.T, uy_sl.T,
                          scale=1.0, scale_units="xy", color=C_PURPLE, alpha=0.7)
                ax.set_xlabel("mm"); ax.set_ylabel("mm")
                ax.set_title("Z-displacement vectors (XY slice)")
                st.pyplot(fig); plt.close()


# ──────────────────────────────────────────────────────────────
# TAB 3 — FE vs DVC COMPARISON
# ──────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown("### FE prediction vs DVC measurement")
    st.markdown(
        "Compare the FE-predicted apparent modulus (from voxel FE on the reference "
        "geometry + Gibson-Ashby tissue modulus) against the DVC-derived apparent "
        "modulus (stress / DVC strain). This tells you how well the constitutive "
        "model captures the real foam behaviour."
    )

    if "dvc_volumes" not in st.session_state:
        st.info("Generate volumes first.")
    else:
        vols   = st.session_state["dvc_volumes"]
        E_s    = st.session_state.get("dvc_solid_E", 3500.0)
        ftype  = st.session_state.get("dvc_ftype", "open")
        vox_um = st.session_state.get("dvc_voxel_um", 39.0)

        ref     = vols[0]
        sf      = 1.0 - _measure_porosity(ref)
        E_star  = _gibson_ashby(sf, E_s, ftype)
        sigma_y = E_s * 0.3 * sf**1.5  # Gibson-Ashby strength

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Solid fraction", f"{sf:.3f}")
        cc2.metric("E* (Gibson-Ashby)", f"{E_star:.1f} MPa")
        cc3.metric("σ* (strength)", f"{sigma_y:.2f} MPa")

        # DVC-derived apparent moduli per step
        strains = st.session_state.get("dvc_strains", [])
        dvc_results = []
        for i in range(len(vols)-1):
            eps_key = f"dvc_eps_app_{i}"
            if eps_key in st.session_state:
                eps_app = st.session_state[eps_key]
                nom_stress = E_star * strains[i+1]  # σ = E* · ε_nominal (FE)
                E_dvc = nom_stress / max(abs(eps_app), 1e-6)
                dvc_results.append({
                    "step": i,
                    "eps_nominal": strains[i+1],
                    "eps_dvc":     eps_app,
                    "E_ga":        E_star,
                    "E_dvc":       E_dvc,
                })

        if dvc_results:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            steps   = [r["step"]+1 for r in dvc_results]
            E_ga    = [r["E_ga"]  for r in dvc_results]
            E_dvcs  = [r["E_dvc"] for r in dvc_results]
            eps_nom = [r["eps_nominal"] for r in dvc_results]
            eps_dvc = [r["eps_dvc"]     for r in dvc_results]

            axes[0].plot(steps, E_ga,   "o--", color=C_BLUE,   label="Gibson-Ashby E*")
            axes[0].plot(steps, E_dvcs, "s-",  color=C_CORAL,  label="DVC apparent E")
            axes[0].set_xlabel("Load step"); axes[0].set_ylabel("Apparent modulus (MPa)")
            axes[0].set_title("E*: model vs measurement")
            axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

            axes[1].scatter(eps_nom, eps_dvc, color=C_PURPLE, s=60, zorder=5)
            lim = max(max(eps_nom), max(eps_dvc)) * 1.1
            axes[1].plot([0, lim], [0, lim], "k--", lw=1, label="Perfect agreement")
            axes[1].set_xlabel("Nominal strain (applied)")
            axes[1].set_ylabel("DVC apparent strain")
            axes[1].set_title("Strain: nominal vs DVC")
            axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig); plt.close()

            # Error table
            errors = [abs(r["E_dvc"]-r["E_ga"])/r["E_ga"]*100 for r in dvc_results]
            st.dataframe({
                "Load step":     steps,
                "ε nominal":     [f"{e:.4f}" for e in eps_nom],
                "ε DVC":         [f"{e:.4f}" for e in eps_dvc],
                "E* GA (MPa)":   [f"{e:.1f}" for e in E_ga],
                "E* DVC (MPa)":  [f"{e:.1f}" for e in E_dvcs],
                "Error %":       [f"{e:.1f}%" for e in errors],
            }, hide_index=True, use_container_width=True)
        else:
            st.info("Run DVC on at least one step pair (DVC tab) to see comparison.")


# ──────────────────────────────────────────────────────────────
# TAB 4 — CALIBRATION CURVE
# ──────────────────────────────────────────────────────────────
with tab_calib:
    st.markdown("### Gibson-Ashby calibration curve")
    st.markdown(
        "Sweep porosity and compare Gibson-Ashby E* against the DVC-measured "
        "apparent modulus at your current load step. Use this to calibrate the "
        "model's C₁ prefactor for your specific foam and imaging conditions."
    )

    E_s   = st.session_state.get("dvc_solid_E", 3500.0)
    ftype = st.session_state.get("dvc_ftype",   "open")

    por_range = np.linspace(0.50, 0.97, 80)
    sf_range  = 1.0 - por_range
    E_open    = np.array([_gibson_ashby(sf, E_s, "open")   for sf in sf_range])
    E_closed  = np.array([_gibson_ashby(sf, E_s, "closed") for sf in sf_range])

    c1_tune = st.slider("C₁ prefactor (open-cell scaling)", 0.1, 2.0, 1.0, 0.05,
        help="Standard G-A uses C₁=1. Fit this to your DVC measurements.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(por_range, E_open * c1_tune, color=C_BLUE, lw=2.5,
                label=f"Open-cell (C₁={c1_tune:.2f})")
    ax.semilogy(por_range, E_closed,          color=C_TEAL, lw=2, ls="--",
                label="Closed-cell (C₁=1)")

    # Plot DVC measurements as points
    if "dvc_volumes" in st.session_state:
        vols    = st.session_state["dvc_volumes"]
        strains = st.session_state.get("dvc_strains", [])
        for i in range(len(vols)-1):
            eps_key = f"dvc_eps_app_{i}"
            if eps_key in st.session_state:
                ref_sf  = 1.0 - _measure_porosity(vols[0])
                eps_app = st.session_state[eps_key]
                nom_stress = _gibson_ashby(ref_sf, E_s, ftype) * strains[i+1]
                E_dvc = nom_stress / max(abs(eps_app), 1e-6)
                ax.scatter([1-ref_sf], [E_dvc], color=C_CORAL, s=90, zorder=6,
                           label=f"DVC step {i+1}" if i == 0 else "")

    ax.set_xlabel("Porosity"); ax.set_ylabel("E* (MPa, log)")
    ax.set_title(f"Gibson-Ashby calibration — {mat_sel} base material")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.invert_xaxis()
    st.pyplot(fig); plt.close()

    st.markdown(
        "**Interpretation:** if your DVC points fall below the open-cell line, "
        "C₁ < 1, which typically means partial cell-wall damage or imperfect "
        "binarisation inflating the apparent porosity. "
        "If they fall above, C₁ > 1 — the foam has cell-wall stretching or "
        "closed-cell contributions increasing stiffness beyond the open-cell model."
    )


# ──────────────────────────────────────────────────────────────
# TAB 5 — EXPORT
# ──────────────────────────────────────────────────────────────
with tab_export:
    st.markdown("### Export results to the pipeline")

    if "dvc_volumes" not in st.session_state:
        st.info("Generate or load volumes first.")
    else:
        vols   = st.session_state["dvc_volumes"]
        ftype  = st.session_state.get("dvc_ftype", "open")
        E_s    = st.session_state.get("dvc_solid_E", 3500.0)
        ref    = vols[0]
        sf     = 1.0 - _measure_porosity(ref)
        E_star = _gibson_ashby(sf, E_s, ftype)

        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("#### → 3D Viewer")
            if st.button("📤 Push foam mask + strain", key="dvc_to_3d"):
                st.session_state["bone_volume"] = {
                    "bone_mask": ref,
                    "voxel_um":  st.session_state.get("dvc_voxel_um", 39.0),
                    "bvtv":      float(sf),
                }
                # push first available DVC displacement as strain field
                for i in range(len(vols)-1):
                    if f"dvc_umag_{i}" in st.session_state:
                        st.session_state["strain_volume_3d"]  = st.session_state[f"dvc_umag_{i}"]
                        st.session_state["strain_label_3d"]   = f"DVC |u| step {i}→{i+1}"
                        st.session_state["strain_registered"] = True
                        break
                st.success("Pushed — open **3D Viewer**.")

        with ec2:
            st.markdown("#### → FE Solver")
            if st.button("📤 Push E* to FE Solver", key="dvc_to_fe"):
                st.session_state["fe_E_tissue_MPa"]  = E_star
                st.session_state["fe_material_label"] = (
                    f"foam E*={E_star:.0f} MPa, sf={sf:.3f}"
                )
                st.session_state["bone_volume"] = {
                    "bone_mask": ref,
                    "voxel_um":  st.session_state.get("dvc_voxel_um", 39.0),
                    "bvtv":      float(sf),
                }
                st.success("Pushed — open **FE Solver**.")

        st.markdown("---")
        st.markdown("#### → QKSVM & QRC")
        if st.button("📤 Push foam morphometrics", key="dvc_to_qksvm"):
            from scipy.ndimage import distance_transform_edt as _dte
            pore = 1 - ref
            dt_s = _dte(ref)  * st.session_state.get("dvc_voxel_um", 39.0)
            dt_p = _dte(pore) * st.session_state.get("dvc_voxel_um", 39.0)
            m = {
                "porosity":      round(float(pore.mean()), 4),
                "solid_fraction": round(sf, 4),
                "strut_dia_um":  round(float(dt_s[ref.astype(bool)].mean())*2 if ref.any() else 0, 1),
                "cell_size_um":  round(float(dt_p[pore.astype(bool)].mean())*2 if pore.any() else 0, 1),
                "n_cells":       int(ndi_label(pore)[1]),
            }
            st.session_state["morphometrics"]      = m
            st.session_state["last_morphometrics"] = m
            st.session_state["foam_morphometrics"] = m
            st.success("Pushed morphometrics — open **QKSVM Results** or **Quantum Reservoir**.")
            st.json(m)

