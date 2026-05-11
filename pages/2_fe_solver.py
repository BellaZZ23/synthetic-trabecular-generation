"""
Page 2: FE Solver
==================
Run compression, tension, or torque on a generated bone volume.
Visualise displacement fields, strain fields, and apparent properties.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fe_coupling"))
from step3_generator_fe_coupling import (
    generate_bone_volume, run_fe_analysis,
)

st.set_page_config(page_title="FE solver", page_icon="🔧", layout="wide")
st.title("Micro-FE solver")
st.caption("Uniaxial compression, tension, and torque with strain field extraction")

# ── Check if a volume exists in session state ──
has_volume = "bone_volume" in st.session_state

if not has_volume:
    st.warning("No bone volume in session. Generate one here or go to the Generator page first.")
    st.subheader("Quick generate")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    q_bvtv = qcol1.slider("BV/TV", 0.10, 0.45, 0.30, 0.01, key="q_bvtv")
    q_xy = qcol2.selectbox("XY", [16, 32, 48, 64], index=1, key="q_xy")
    q_z = qcol3.selectbox("Z", [8, 16, 24, 32], index=1, key="q_z")
    q_seed = qcol4.number_input("Seed", value=42, key="q_seed")

    if st.button("Generate + continue", type="primary"):
        with st.spinner("Generating..."):
            vol = generate_bone_volume(
                nx=q_xy, ny=q_xy, nz=q_z,
                target_bvtv=q_bvtv, seed=int(q_seed), verbose=False,
            )
        st.session_state["bone_volume"] = vol
        st.rerun()
    st.stop()

vol = st.session_state["bone_volume"]
bone_mask = vol["bone_mask"]
morph = vol["morphometrics"]
nz, ny, nx = bone_mask.shape
voxel_mm = vol["voxel_um"] / 1000.0

# ── Volume info bar ──
ic1, ic2, ic3, ic4 = st.columns(4)
ic1.metric("Volume", f"{nx}x{ny}x{nz}")
ic2.metric("BV/TV", f"{morph['BVTV']:.3f}")
ic3.metric("Tb.Th", f"{morph['TbTh_um_p50']:.0f} um")
ic4.metric("LCC", f"{morph['lcc_frac']:.3f}")

st.divider()

# ── Solver controls ──
st.sidebar.header("FE parameters")
load_type = st.sidebar.radio("Load case", ["compression", "tension", "torque"])

E_bone = st.sidebar.number_input("E_bone (MPa)", value=18000.0, step=1000.0)
nu = st.sidebar.number_input("Poisson ratio", value=0.3, step=0.05, min_value=0.0, max_value=0.49)

if load_type == "torque":
    strain_val = st.sidebar.slider("Rotation (deg)", 0.1, 5.0, 0.57, 0.01)
    applied_strain = np.radians(strain_val)
    st.sidebar.caption(f"= {applied_strain:.4f} rad")
else:
    applied_strain = st.sidebar.slider("Applied strain", 0.001, 0.05, 0.01, 0.001)

# ── Run solver ──
if st.sidebar.button(f"Run {load_type}", type="primary", use_container_width=True):

    with st.spinner(f"Running {load_type} FE analysis..."):
        fe = run_fe_analysis(
            bone_mask,
            voxel_size_mm=voxel_mm,
            load_type=load_type,
            E_bone=E_bone,
            nu=nu,
            applied_strain=applied_strain,
            verbose=True,
        )

    st.session_state["fe_results"] = fe

# ── Display results ──
if "fe_results" in st.session_state:
    fe = st.session_state["fe_results"]
    ux, uy, uz = fe["displacement"]
    strain = fe["strain_field"]
    mesh = fe["mesh"]
    lt = fe["load_type"]

    # ── Results metrics ──
    st.subheader(f"Results: {lt}")
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Elements", f"{fe['n_elements']:,}")
    rc2.metric("Solve time", f"{fe['solve_time']:.1f}s")

    if lt in ("compression", "tension"):
        rc3.metric("E_apparent", f"{fe['apparent_modulus']:.0f} MPa")
        rc4.metric("E / E_voigt", f"{fe['apparent_modulus']/fe['voigt_bound']:.3f}",
                   "PASS" if fe['apparent_modulus'] <= fe['voigt_bound'] * 1.01 else "FAIL")
    elif lt == "torque" and fe["apparent_shear_modulus"] is not None:
        rc3.metric("G_apparent", f"{fe['apparent_shear_modulus']:.0f} MPa")
        rc4.metric("Rotation", f"{np.degrees(fe['applied_strain']):.2f} deg")

    st.divider()

    # ── Slice selector ──
    mid_z = nz // 2
    slice_idx = st.slider("Z-slice for visualisation", 0, nz - 1, mid_z, key="fe_slice")
    mid_z_mm = (slice_idx + 0.5) * voxel_mm
    tol = voxel_mm * 0.1
    extent = [0, nx * voxel_mm, 0, ny * voxel_mm]

    # ── Helper functions ──
    def nodal_to_slice(values, z_idx):
        out = np.full((nx, ny), np.nan)
        for n_idx in range(mesh.nvertices):
            x, y, z = mesh.p[:, n_idx]
            if abs(z - z_idx * voxel_mm) < tol or abs(z - (z_idx + 1) * voxel_mm) < tol:
                i = int(round(x / voxel_mm))
                j = int(round(y / voxel_mm))
                if 0 <= i < nx and 0 <= j < ny:
                    if np.isnan(out[i, j]):
                        out[i, j] = values[n_idx]
        return out

    def element_to_slice(values, centroids, z_mm):
        out = np.full((nx, ny), np.nan)
        z_range = voxel_mm * 0.6
        for e_idx in range(len(values)):
            cz = centroids[e_idx, 2]
            if abs(cz - z_mm) < z_range:
                ci = int(round(centroids[e_idx, 0] / voxel_mm))
                cj = int(round(centroids[e_idx, 1] / voxel_mm))
                if 0 <= ci < nx and 0 <= cj < ny:
                    out[ci, cj] = values[e_idx]
        return out

    # ── Row 1: Displacement fields ──
    st.subheader("Displacement fields")
    d1, d2, d3, d4 = st.columns(4)

    with d1:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(bone_mask[slice_idx].T, cmap='gray', origin='lower', extent=extent)
        ax.set_title(f"Bone (z={slice_idx})"); ax.set_xlabel("x"); ax.set_ylabel("y")
        st.pyplot(fig); plt.close()

    with d2:
        fig, ax = plt.subplots(figsize=(4, 4))
        mag = nodal_to_slice(np.sqrt(ux**2 + uy**2 + uz**2), slice_idx)
        im = ax.imshow(mag.T, cmap='hot', origin='lower', extent=extent)
        ax.set_title("|u| magnitude"); ax.set_xlabel("x")
        plt.colorbar(im, ax=ax, shrink=0.8); st.pyplot(fig); plt.close()

    with d3:
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(nodal_to_slice(uz, slice_idx).T, cmap='RdBu', origin='lower', extent=extent)
        ax.set_title("uz"); ax.set_xlabel("x")
        plt.colorbar(im, ax=ax, shrink=0.8); st.pyplot(fig); plt.close()

    with d4:
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(nodal_to_slice(ux, slice_idx).T, cmap='RdBu', origin='lower', extent=extent)
        ax.set_title("ux"); ax.set_xlabel("x")
        plt.colorbar(im, ax=ax, shrink=0.8); st.pyplot(fig); plt.close()

    # ── Row 2: Strain fields ──
    st.subheader("Strain fields")
    s1, s2, s3 = st.columns(3)

    with s1:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ezz = element_to_slice(strain["eps_zz"], strain["centroids"], mid_z_mm)
        im = ax.imshow(ezz.T, cmap='RdBu_r', origin='lower', extent=extent)
        ax.set_title("Axial strain (eps_zz)"); ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, shrink=0.8); st.pyplot(fig); plt.close()

    with s2:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        evm = element_to_slice(strain["eps_von_mises"], strain["centroids"], mid_z_mm)
        im = ax.imshow(evm.T, cmap='inferno', origin='lower', extent=extent)
        ax.set_title("von Mises strain"); ax.set_xlabel("x")
        plt.colorbar(im, ax=ax, shrink=0.8); st.pyplot(fig); plt.close()

    with s3:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        emp = element_to_slice(strain["eps_max_principal"], strain["centroids"], mid_z_mm)
        im = ax.imshow(emp.T, cmap='magma', origin='lower', extent=extent)
        ax.set_title("Max principal strain"); ax.set_xlabel("x")
        plt.colorbar(im, ax=ax, shrink=0.8); st.pyplot(fig); plt.close()

    # ── Strain statistics ──
    with st.expander("Strain field statistics"):
        scol1, scol2 = st.columns(2)
        with scol1:
            st.json({
                "eps_xx": {"min": round(float(strain["eps_xx"].min()), 6),
                           "max": round(float(strain["eps_xx"].max()), 6)},
                "eps_yy": {"min": round(float(strain["eps_yy"].min()), 6),
                           "max": round(float(strain["eps_yy"].max()), 6)},
                "eps_zz": {"min": round(float(strain["eps_zz"].min()), 6),
                           "max": round(float(strain["eps_zz"].max()), 6)},
            })
        with scol2:
            st.json({
                "eps_xy": {"min": round(float(strain["eps_xy"].min()), 6),
                           "max": round(float(strain["eps_xy"].max()), 6)},
                "von_mises": {"min": round(float(strain["eps_von_mises"].min()), 6),
                              "max": round(float(strain["eps_von_mises"].max()), 6)},
                "max_principal": {"min": round(float(strain["eps_max_principal"].min()), 6),
                                  "max": round(float(strain["eps_max_principal"].max()), 6)},
            })

else:
    st.info("Select a load case in the sidebar and click Run to start the FE analysis.")
