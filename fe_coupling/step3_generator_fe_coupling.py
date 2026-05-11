"""
Step 3: Generator-to-FE Coupling
=================================
Connects the v15 zero-crossing generator to the micro-FE solver.
Generates a bone volume using the published generator, converts it
to a hex mesh, applies uniaxial compression, and validates.

This replaces the simplified inline GRF from step2 with your actual
v15.3 generator.

Usage:
    python step3_generator_fe_coupling.py

    # With custom parameters:
    python step3_generator_fe_coupling.py --bvtv 0.25 --xy 32 --z 16

    # Full-size volume (slower, needs more RAM):
    python step3_generator_fe_coupling.py --bvtv 0.33 --xy 64 --z 32
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path
from scipy.ndimage import label
from scipy.ndimage import generate_binary_structure

# ── Import the v15 generator ──
# Adjust this path if your repo structure differs
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from synthetic_trabecular_v15_morphometric_control import (
    make_isotropic_field,
    normalize,
    smooth_warp,
    calibrate_bvtv_by_zero_crossings,
    keep_largest_component,
    measure_all_morphometrics,
    microct_gray_solid,
    RidgeParams,
    GrayParams,
)


# ══════════════════════════════════════════════════════════════
# GENERATOR WRAPPER
# ══════════════════════════════════════════════════════════════

def generate_bone_volume(
    nx: int = 128,
    ny: int = 128,
    nz: int = 40,
    target_bvtv: float = 0.33,
    voxel_um: float = 39.0,
    base_sigma: float = 2.5,
    warp_amp: float = 1.2,
    warp_sigma: float = 12.0,
    plate_weight: float = 0.7,
    close_iters: int = 3,
    min_component: int = 400,
    seed: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Generate a synthetic trabecular bone volume using the v15.3
    zero-crossing generator.

    Returns a dict with:
        bone_mask:    np.ndarray (nx, ny, nz) binary bone mask
        morphometrics: dict with BV/TV, Tb.Th, Tb.N, Tb.Sp measurements
        calibration:  dict with wall_thickness, bvtv_got, etc.
        field:        np.ndarray the raw Gaussian field (for debugging)
        seed:         int the random seed used
    """
    shape = (nz, ny, nx)  # v15 uses (Z, Y, X) ordering
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"Generating bone volume: {nx}x{ny}x{nz} voxels")
        print(f"  Target BV/TV: {target_bvtv:.3f}")
        print(f"  Voxel size: {voxel_um:.0f} um")
        print(f"  base_sigma={base_sigma}, warp_amp={warp_amp}")

    t0 = time.time()

    # Step 1: Create isotropic Gaussian random field
    field = make_isotropic_field(shape, rng, base_sigma)
    field = smooth_warp(field, rng, warp_sigma, warp_amp)
    field = normalize(field)

    # Step 2: Zero-crossing wall extraction (plate-dominant path)
    bone_mask, cal_info = calibrate_bvtv_by_zero_crossings(
        field,
        target_bvtv,
        close_iters=close_iters,
        min_component=min_component,
        round_sigma=0.35,
    )

    # Step 3: Keep largest connected component
    bone_mask = keep_largest_component(bone_mask)

    t1 = time.time()

    # Step 4: Measure morphometrics
    morphometrics = measure_all_morphometrics(bone_mask, voxel_um)

    if verbose:
        print(f"  Generated in {t1-t0:.2f}s")
        print(f"  BV/TV: target={target_bvtv:.3f}, measured={morphometrics['BVTV']:.3f}")
        print(f"  Tb.Th (p50): {morphometrics['TbTh_um_p50']:.1f} um")
        print(f"  Tb.N: {morphometrics['TbN_per_mm']:.2f} /mm")
        print(f"  Tb.Sp (p50): {morphometrics['TbSp_um_p50']:.1f} um")
        print(f"  LCC fraction: {morphometrics['lcc_frac']:.3f}")
        print(f"  Components: {morphometrics['n_components']}")

    return {
        "bone_mask": bone_mask,  # shape (Z, Y, X), uint8 0/1
        "morphometrics": morphometrics,
        "calibration": cal_info,
        "field": field,
        "seed": seed,
        "shape_zyx": shape,
        "voxel_um": voxel_um,
    }


# ══════════════════════════════════════════════════════════════
# GRAYSCALE WRAPPER
# ══════════════════════════════════════════════════════════════

def generate_grayscale(bone_mask, seed=100, bone_mean=90.0, marrow_mean=15.0,
                       solid_fill_sigma=0.8, noise_sd=2.0, bg_tex_sd=0.5):
    """Generate synthetic micro-CT grayscale from a bone mask."""
    rng = np.random.default_rng(seed)
    gp = GrayParams(
        bone_mean=bone_mean,
        marrow_mean=marrow_mean,
        solid_fill_sigma=solid_fill_sigma,
        noise_sd=noise_sd,
        bg_tex_sd=bg_tex_sd,
    )
    return microct_gray_solid(bone_mask, gp, rng, br=2.0)


# ══════════════════════════════════════════════════════════════
# FE SOLVER (from step2, cleaned up)
# ══════════════════════════════════════════════════════════════

def bone_to_hex_mesh(bone_mask, voxel_size_mm):
    """
    Convert a binary bone mask to a hexahedral FE mesh.

    Args:
        bone_mask: np.ndarray (Z, Y, X) binary, 1=bone 0=marrow
        voxel_size_mm: float, voxel size in mm

    Returns:
        mesh: skfem MeshHex object
        n_elements: int
        n_nodes: int
    """
    from skfem import MeshHex

    nz, ny, nx = bone_mask.shape

    def node_index(i, j, k):
        return i + j * (nx + 1) + k * (nx + 1) * (ny + 1)

    # Build element connectivity for bone voxels
    # Note: bone_mask is (Z, Y, X) but we iterate as (x, y, z) for the mesh
    elements = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if bone_mask[k, j, i] == 1:  # (Z, Y, X) indexing
                    n = [
                        node_index(i,   j,   k),
                        node_index(i+1, j,   k),
                        node_index(i+1, j+1, k),
                        node_index(i,   j+1, k),
                        node_index(i,   j,   k+1),
                        node_index(i+1, j,   k+1),
                        node_index(i+1, j+1, k+1),
                        node_index(i,   j+1, k+1),
                    ]
                    elements.append(n)

    elements = np.array(elements, dtype=np.int64)

    # Build node coordinates
    n_nodes_total = (nx + 1) * (ny + 1) * (nz + 1)
    coords = np.zeros((3, n_nodes_total))
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                idx = node_index(i, j, k)
                coords[0, idx] = i * voxel_size_mm
                coords[1, idx] = j * voxel_size_mm
                coords[2, idx] = k * voxel_size_mm

    # Renumber to active nodes only
    used_nodes = np.unique(elements.flatten())
    old_to_new = np.full(n_nodes_total, -1, dtype=np.int64)
    old_to_new[used_nodes] = np.arange(len(used_nodes))

    new_coords = coords[:, used_nodes]
    new_elements = old_to_new[elements]

    mesh = MeshHex(new_coords, new_elements.T)
    return mesh, len(elements), len(used_nodes)


def run_fe_analysis(
    bone_mask,
    voxel_size_mm,
    load_type="compression",
    E_bone=18_000.0,
    nu=0.3,
    applied_strain=0.01,
    verbose=True,
):
    """
    Run FE analysis on a bone volume with different load cases.

    Args:
        bone_mask:       np.ndarray (Z, Y, X) binary bone mask
        voxel_size_mm:   float, voxel size in mm
        load_type:       str, one of "compression", "tension", "torque"
        E_bone:          float, Young's modulus of bone tissue [MPa]
        nu:              float, Poisson's ratio
        applied_strain:  float, applied strain magnitude (or rotation angle
                         in radians for torque, default 0.01 rad ~ 0.57 deg)
        verbose:         bool

    Returns dict with:
        displacement:      (ux, uy, uz) arrays at nodes
        strain_field:      dict with per-element strain components
        apparent_modulus:   float [MPa] (compression/tension) or None (torque)
        apparent_shear_mod: float [MPa] (torque only) or None
        voigt_bound:       float [MPa]
        reaction_force:    float [N]
        load_type:         str
        mesh, basis, solution, solve_time, n_elements, n_nodes
    """
    from skfem import (
        Basis, ElementVector, ElementHex1, asm, solve, condense,
    )
    from skfem.models.elasticity import linear_elasticity, lame_parameters

    valid_loads = ("compression", "tension", "torque")
    if load_type not in valid_loads:
        raise ValueError(f"load_type must be one of {valid_loads}, got '{load_type}'")

    nz, ny, nx = bone_mask.shape
    bvtv = bone_mask.astype(bool).mean()

    if verbose:
        print(f"\nFE analysis [{load_type}]: {nx}x{ny}x{nz}, BV/TV={bvtv:.3f}")

    # Connectivity check
    st26 = generate_binary_structure(3, 2)
    labeled, n_components = label(bone_mask, structure=st26)
    if n_components > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        largest = sizes.argmax()
        bone_mask = (labeled == largest).astype(np.uint8)
        if verbose:
            print(f"  {n_components} components, keeping largest ({bone_mask.sum()} voxels)")

    # ── Build mesh (shared across all load cases) ──
    t0 = time.time()
    mesh, n_elem, n_nodes = bone_to_hex_mesh(bone_mask, voxel_size_mm)
    if verbose:
        print(f"  Mesh: {n_elem} elements, {n_nodes} nodes")

    lam, mu = lame_parameters(E_bone, nu)
    elem = ElementVector(ElementHex1())
    ib = Basis(mesh, elem)
    if verbose:
        print(f"  DOFs: {ib.N}")

    # ── Assemble stiffness (shared) ──
    t1 = time.time()
    K = asm(linear_elasticity(lam, mu), ib)
    t2 = time.time()
    if verbose:
        print(f"  Assembly: {t2-t1:.1f}s")

    # ── Geometric bounds ──
    x_min, x_max = mesh.p[0].min(), mesh.p[0].max()
    y_min, y_max = mesh.p[1].min(), mesh.p[1].max()
    z_min, z_max = mesh.p[2].min(), mesh.p[2].max()
    height = z_max - z_min
    width_x = x_max - x_min
    width_y = y_max - y_min
    cx = 0.5 * (x_min + x_max)  # centre x
    cy = 0.5 * (y_min + y_max)  # centre y
    tol = voxel_size_mm * 0.1

    # ── DOF sets for faces ──
    dofs_x0 = ib.get_dofs(mesh.nodes_satisfying(lambda x: x[0] < x_min + tol))
    dofs_y0 = ib.get_dofs(mesh.nodes_satisfying(lambda x: x[1] < y_min + tol))
    dofs_z0 = ib.get_dofs(mesh.nodes_satisfying(lambda x: x[2] < z_min + tol))
    dofs_z1 = ib.get_dofs(mesh.nodes_satisfying(lambda x: x[2] > z_max - tol))

    # ── Apply BCs per load type ──
    x_prescribed = np.zeros(ib.N)
    dofs_D = []

    if load_type in ("compression", "tension"):
        # Symmetry BCs: ux=0 on x=0, uy=0 on y=0, uz=0 on z=0
        for dof in dofs_x0.nodal['u^1']:
            dofs_D.append(dof)
        for dof in dofs_y0.nodal['u^2']:
            dofs_D.append(dof)
        for dof in dofs_z0.nodal['u^3']:
            dofs_D.append(dof)

        # Prescribed displacement on top face
        sign = -1.0 if load_type == "compression" else +1.0
        applied_disp = sign * applied_strain * height

        for dof in dofs_z1.nodal['u^3']:
            x_prescribed[dof] = applied_disp
            dofs_D.append(dof)

        if verbose:
            print(f"  BCs: symmetry + {load_type} uz={applied_disp:.6f} mm on top")

    elif load_type == "torque":
        # Torque about the z-axis:
        #   Bottom face (z=0): fully fixed (ux=uy=uz=0)
        #   Top face (z=max): prescribed rotation about z-axis
        #     ux = -theta * (y - cy)
        #     uy = +theta * (x - cx)
        #     uz = 0  (no axial extension)
        theta = applied_strain  # reuse param as rotation angle [rad]

        # Bottom: fix all DOFs
        for dof in dofs_z0.nodal['u^1']:
            dofs_D.append(dof)
        for dof in dofs_z0.nodal['u^2']:
            dofs_D.append(dof)
        for dof in dofs_z0.nodal['u^3']:
            dofs_D.append(dof)

        # Top: prescribed rotation + fix uz
        top_node_indices = mesh.nodes_satisfying(lambda x: x[2] > z_max - tol)
        top_dofs = ib.get_dofs(top_node_indices)

        for ni in top_node_indices:
            xi, yi = mesh.p[0, ni], mesh.p[1, ni]
            # Find the DOF indices for this node
            ux_dof = ib.nodal_dofs[0][ni]
            uy_dof = ib.nodal_dofs[1][ni]
            uz_dof = ib.nodal_dofs[2][ni]

            x_prescribed[ux_dof] = -theta * (yi - cy)
            x_prescribed[uy_dof] = +theta * (xi - cx)
            x_prescribed[uz_dof] = 0.0

            dofs_D.extend([ux_dof, uy_dof, uz_dof])

        if verbose:
            print(f"  BCs: fixed bottom + torque theta={theta:.4f} rad ({np.degrees(theta):.2f} deg) on top")

    dofs_D = np.unique(np.array(dofs_D))
    if verbose:
        print(f"  Constrained DOFs: {len(dofs_D)}")

    # ── Solve ──
    t3 = time.time()
    f_ext = np.zeros(ib.N)
    u = solve(*condense(K, f_ext, x=x_prescribed, D=dofs_D))
    t4 = time.time()
    if verbose:
        print(f"  Solve: {t4-t3:.1f}s")

    # ── Extract nodal displacements ──
    ux = u[ib.nodal_dofs[0]]
    uy = u[ib.nodal_dofs[1]]
    uz = u[ib.nodal_dofs[2]]

    # ── Extract element strain fields ──
    strain_field = compute_element_strains(mesh, ib, u, verbose=verbose)

    # ── Compute apparent properties ──
    f_reaction = K @ u
    area_full = (nx * voxel_size_mm) * (ny * voxel_size_mm)
    E_voigt = bvtv * E_bone
    E_apparent = None
    G_apparent = None

    if load_type in ("compression", "tension"):
        top_z_dofs = dofs_z1.nodal['u^3']
        reaction_z = np.sum(f_reaction[top_z_dofs])
        sigma_apparent = abs(reaction_z) / area_full
        E_apparent = sigma_apparent / applied_strain

        if verbose:
            print(f"\n  Results [{load_type}]:")
            print(f"    Apparent modulus: {E_apparent:.1f} MPa")
            print(f"    Voigt bound: {E_voigt:.1f} MPa")
            print(f"    E/E_voigt: {E_apparent/E_voigt:.3f}")
            print(f"    Voigt check: {'PASS' if E_apparent <= E_voigt * 1.01 else 'FAIL'}")

    elif load_type == "torque":
        # Compute torque from reaction forces on top face
        top_nodes = mesh.nodes_satisfying(lambda x: x[2] > z_max - tol)
        torque_z = 0.0
        for ni in top_nodes:
            fx = f_reaction[ib.nodal_dofs[0][ni]]
            fy = f_reaction[ib.nodal_dofs[1][ni]]
            xi, yi = mesh.p[0, ni] - cx, mesh.p[1, ni] - cy
            torque_z += xi * fy - yi * fx

        # Approximate shear modulus: G ≈ T * L / (J * theta)
        # J ≈ pi/32 * (D^4) for solid cylinder, but for porous bone
        # we use the polar moment of inertia of the cross-section
        # This is approximate — just for comparison
        r_max = 0.5 * max(width_x, width_y)
        J_approx = 0.5 * area_full * r_max**2  # rough polar moment
        G_apparent = abs(torque_z) * height / (J_approx * theta) if theta > 0 else 0.0
        G_voigt = E_voigt / (2 * (1 + nu))

        if verbose:
            print(f"\n  Results [torque]:")
            print(f"    Reaction torque: {torque_z:.6f} N.mm")
            print(f"    Approx shear modulus: {G_apparent:.1f} MPa")
            print(f"    Voigt shear bound: {G_voigt:.1f} MPa")

    total_time = t4 - t0
    if verbose:
        print(f"    Total time: {total_time:.1f}s")

    return {
        "displacement": (ux, uy, uz),
        "strain_field": strain_field,
        "apparent_modulus": E_apparent,
        "apparent_shear_modulus": G_apparent,
        "voigt_bound": E_voigt,
        "reaction_force": float(np.sum(f_reaction[dofs_z1.nodal['u^3']])),
        "applied_strain": applied_strain,
        "load_type": load_type,
        "bvtv": bvtv,
        "E_bone": E_bone,
        "mesh": mesh,
        "basis": ib,
        "solution": u,
        "solve_time": total_time,
        "n_elements": n_elem,
        "n_nodes": n_nodes,
    }


# Keep backward compatibility
def run_fe_compression(bone_mask, voxel_size_mm, **kwargs):
    """Backward-compatible wrapper for compression only."""
    return run_fe_analysis(bone_mask, voxel_size_mm, load_type="compression", **kwargs)


# ══════════════════════════════════════════════════════════════
# STRAIN FIELD EXTRACTION
# ══════════════════════════════════════════════════════════════

def compute_element_strains(mesh, basis, u, verbose=True):
    """
    Compute engineering strain tensor components at each element centroid.

    Returns dict with arrays of shape (n_elements,):
        eps_xx, eps_yy, eps_zz:  normal strains
        eps_xy, eps_xz, eps_yz:  shear strains (engineering, = 2 * tensor shear)
        eps_von_mises:           von Mises equivalent strain
        eps_max_principal:       maximum principal strain
        centroids:               (n_elements, 3) element centroid coordinates
    """
    from skfem import Basis, ElementVector, ElementHex1

    if verbose:
        print(f"  Computing element strains...")
    t0 = time.time()

    # Displacement gradient: du_i/dx_j at element centroids
    # Use the basis to interpolate derivatives
    ux = u[basis.nodal_dofs[0]]
    uy = u[basis.nodal_dofs[1]]
    uz = u[basis.nodal_dofs[2]]

    n_elem = mesh.nelements
    eps_xx = np.zeros(n_elem)
    eps_yy = np.zeros(n_elem)
    eps_zz = np.zeros(n_elem)
    eps_xy = np.zeros(n_elem)
    eps_xz = np.zeros(n_elem)
    eps_yz = np.zeros(n_elem)
    centroids = np.zeros((n_elem, 3))

    # For each element, compute strain from nodal displacements
    # using shape function derivatives at the element centroid
    for e_idx in range(n_elem):
        # Element node indices
        elem_nodes = mesh.t[:, e_idx]

        # Node coordinates for this element
        x_nodes = mesh.p[0, elem_nodes]
        y_nodes = mesh.p[1, elem_nodes]
        z_nodes = mesh.p[2, elem_nodes]

        # Element centroid
        centroids[e_idx] = [x_nodes.mean(), y_nodes.mean(), z_nodes.mean()]

        # Nodal displacements for this element
        ux_e = ux[elem_nodes]
        uy_e = uy[elem_nodes]
        uz_e = uz[elem_nodes]

        # Shape function derivatives at centroid (xi=eta=zeta=0)
        # For 8-node hex, dN/dxi at centroid:
        # Isoparametric mapping: dx/dxi = sum(dNi/dxi * xi)
        dN_dxi = np.array([
            [-1, -1, -1],  # node 0
            [+1, -1, -1],  # node 1
            [+1, +1, -1],  # node 2
            [-1, +1, -1],  # node 3
            [-1, -1, +1],  # node 4
            [+1, -1, +1],  # node 5
            [+1, +1, +1],  # node 6
            [-1, +1, +1],  # node 7
        ]) / 8.0  # 1/8 factor for trilinear hex

        # Jacobian: J[i,j] = sum_n(dN_n/dxi_j * x_n_i)
        coords_e = np.column_stack([x_nodes, y_nodes, z_nodes])  # (8, 3)
        J = dN_dxi.T @ coords_e  # (3, 3)

        # dN/dx = J^{-1} @ dN/dxi
        try:
            J_inv = np.linalg.inv(J)
        except np.linalg.LinAlgError:
            continue  # skip degenerate elements

        dN_dx = (J_inv @ dN_dxi.T).T  # (8, 3)

        # Displacement gradient: du_i/dx_j = sum_n(dN_n/dx_j * u_n_i)
        du_dx = np.zeros((3, 3))
        du_dx[0, :] = dN_dx.T @ ux_e  # dux/dx, dux/dy, dux/dz
        du_dx[1, :] = dN_dx.T @ uy_e  # duy/dx, duy/dy, duy/dz
        du_dx[2, :] = dN_dx.T @ uz_e  # duz/dx, duz/dy, duz/dz

        # Engineering strain: eps_ij = 0.5*(du_i/dx_j + du_j/dx_i)
        eps_xx[e_idx] = du_dx[0, 0]
        eps_yy[e_idx] = du_dx[1, 1]
        eps_zz[e_idx] = du_dx[2, 2]
        eps_xy[e_idx] = du_dx[0, 1] + du_dx[1, 0]  # engineering shear
        eps_xz[e_idx] = du_dx[0, 2] + du_dx[2, 0]
        eps_yz[e_idx] = du_dx[1, 2] + du_dx[2, 1]

    # Von Mises equivalent strain
    eps_von_mises = np.sqrt(
        2.0/3.0 * (
            (eps_xx - eps_yy)**2 +
            (eps_yy - eps_zz)**2 +
            (eps_zz - eps_xx)**2 +
            1.5 * (eps_xy**2 + eps_xz**2 + eps_yz**2)
        )
    )

    # Maximum principal strain (eigenvalue of symmetric strain tensor)
    eps_max_principal = np.zeros(n_elem)
    for e_idx in range(n_elem):
        eps_tensor = np.array([
            [eps_xx[e_idx],      0.5*eps_xy[e_idx], 0.5*eps_xz[e_idx]],
            [0.5*eps_xy[e_idx],  eps_yy[e_idx],     0.5*eps_yz[e_idx]],
            [0.5*eps_xz[e_idx],  0.5*eps_yz[e_idx], eps_zz[e_idx]],
        ])
        eigs = np.linalg.eigvalsh(eps_tensor)
        eps_max_principal[e_idx] = eigs[-1]  # largest eigenvalue

    t1 = time.time()
    if verbose:
        print(f"    Strain computation: {t1-t0:.2f}s")
        print(f"    eps_zz range: [{eps_zz.min():.6f}, {eps_zz.max():.6f}]")
        print(f"    von Mises range: [{eps_von_mises.min():.6f}, {eps_von_mises.max():.6f}]")

    return {
        "eps_xx": eps_xx, "eps_yy": eps_yy, "eps_zz": eps_zz,
        "eps_xy": eps_xy, "eps_xz": eps_xz, "eps_yz": eps_yz,
        "eps_von_mises": eps_von_mises,
        "eps_max_principal": eps_max_principal,
        "centroids": centroids,
    }


# ══════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════

def plot_results(bone_volume, fe_results, output_path=None):
    """Plot bone structure, displacement field, strain field, and summary."""
    import matplotlib.pyplot as plt

    bone_mask = bone_volume["bone_mask"]
    morph = bone_volume["morphometrics"]
    ux, uy, uz = fe_results["displacement"]
    strain = fe_results["strain_field"]
    mesh = fe_results["mesh"]
    load_type = fe_results["load_type"]
    nz, ny, nx = bone_mask.shape
    voxel_mm = bone_volume["voxel_um"] / 1000.0
    mid_z = nz // 2
    tol = voxel_mm * 0.1

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"FE analysis: {load_type} | {nx}x{ny}x{nz} @ {bone_volume['voxel_um']:.0f}um",
                 fontsize=14, fontweight='bold')

    # Helper: map nodal values to a 2D slice
    def nodal_to_slice(values, mid_z_idx):
        out = np.full((nx, ny), np.nan)
        for n_idx in range(mesh.nvertices):
            x, y, z = mesh.p[:, n_idx]
            if abs(z - mid_z_idx * voxel_mm) < tol or abs(z - (mid_z_idx+1) * voxel_mm) < tol:
                i = int(round(x / voxel_mm))
                j = int(round(y / voxel_mm))
                if 0 <= i < nx and 0 <= j < ny:
                    if np.isnan(out[i, j]):
                        out[i, j] = values[n_idx]
        return out

    # Helper: map element centroid values to a 2D slice
    def element_to_slice(values, centroids, mid_z_mm):
        out = np.full((nx, ny), np.nan)
        z_range = voxel_mm * 0.6
        for e_idx in range(len(values)):
            cz = centroids[e_idx, 2]
            if abs(cz - mid_z_mm) < z_range:
                ci = int(round(centroids[e_idx, 0] / voxel_mm))
                cj = int(round(centroids[e_idx, 1] / voxel_mm))
                if 0 <= ci < nx and 0 <= cj < ny:
                    out[ci, cj] = values[e_idx]
        return out

    extent = [0, nx*voxel_mm, 0, ny*voxel_mm]
    mid_z_mm = (mid_z + 0.5) * voxel_mm

    # Row 1, Col 1: Bone structure
    ax = axes[0, 0]
    ax.imshow(bone_mask[mid_z].T, cmap='gray', origin='lower', extent=extent)
    ax.set_title(f'Bone structure (z={mid_z})')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')

    # Row 1, Col 2: Displacement magnitude
    ax = axes[0, 1]
    disp_mag = nodal_to_slice(np.sqrt(ux**2 + uy**2 + uz**2), mid_z)
    im = ax.imshow(disp_mag.T, cmap='hot', origin='lower', extent=extent)
    ax.set_title('|u| displacement [mm]')
    ax.set_xlabel('x [mm]')
    plt.colorbar(im, ax=ax)

    # Row 1, Col 3: uz displacement
    ax = axes[0, 2]
    im = ax.imshow(nodal_to_slice(uz, mid_z).T, cmap='RdBu', origin='lower', extent=extent)
    ax.set_title('uz displacement [mm]')
    ax.set_xlabel('x [mm]')
    plt.colorbar(im, ax=ax)

    # Row 1, Col 4: ux displacement (shows lateral expansion / torsion)
    ax = axes[0, 3]
    im = ax.imshow(nodal_to_slice(ux, mid_z).T, cmap='RdBu', origin='lower', extent=extent)
    ax.set_title('ux displacement [mm]')
    ax.set_xlabel('x [mm]')
    plt.colorbar(im, ax=ax)

    # Row 2, Col 1: eps_zz (axial strain)
    ax = axes[1, 0]
    ezz = element_to_slice(strain["eps_zz"], strain["centroids"], mid_z_mm)
    im = ax.imshow(ezz.T, cmap='RdBu_r', origin='lower', extent=extent)
    ax.set_title('Axial strain (eps_zz)')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
    plt.colorbar(im, ax=ax)

    # Row 2, Col 2: von Mises strain
    ax = axes[1, 1]
    evm = element_to_slice(strain["eps_von_mises"], strain["centroids"], mid_z_mm)
    im = ax.imshow(evm.T, cmap='inferno', origin='lower', extent=extent)
    ax.set_title('von Mises strain')
    ax.set_xlabel('x [mm]')
    plt.colorbar(im, ax=ax)

    # Row 2, Col 3: max principal strain
    ax = axes[1, 2]
    emp = element_to_slice(strain["eps_max_principal"], strain["centroids"], mid_z_mm)
    im = ax.imshow(emp.T, cmap='magma', origin='lower', extent=extent)
    ax.set_title('Max principal strain')
    ax.set_xlabel('x [mm]')
    plt.colorbar(im, ax=ax)

    # Row 2, Col 4: Summary text
    ax = axes[1, 3]
    ax.axis('off')
    lines = [
        f"Load: {load_type}",
        f"{'─'*28}",
        f"BV/TV:    {morph['BVTV']:.3f}",
        f"Tb.Th:    {morph['TbTh_um_p50']:.1f} um",
        f"Tb.N:     {morph['TbN_per_mm']:.2f} /mm",
        f"LCC:      {morph['lcc_frac']:.3f}",
        f"{'─'*28}",
        f"Elements: {fe_results['n_elements']}",
        f"Nodes:    {fe_results['n_nodes']}",
        f"Time:     {fe_results['solve_time']:.1f}s",
        f"{'─'*28}",
    ]
    if load_type in ("compression", "tension"):
        lines += [
            f"E_app:    {fe_results['apparent_modulus']:.1f} MPa",
            f"E_voigt:  {fe_results['voigt_bound']:.1f} MPa",
            f"E/E_v:    {fe_results['apparent_modulus']/fe_results['voigt_bound']:.3f}",
            f"Voigt:    {'PASS' if fe_results['apparent_modulus'] <= fe_results['voigt_bound'] * 1.01 else 'FAIL'}",
        ]
    elif load_type == "torque":
        if fe_results['apparent_shear_modulus'] is not None:
            lines += [
                f"G_app:    {fe_results['apparent_shear_modulus']:.1f} MPa",
                f"Theta:    {np.degrees(fe_results['applied_strain']):.2f} deg",
            ]
    lines += [
        f"{'─'*28}",
        f"eps_zz:   [{strain['eps_zz'].min():.5f}, {strain['eps_zz'].max():.5f}]",
        f"von Mises:[{strain['eps_von_mises'].min():.5f}, {strain['eps_von_mises'].max():.5f}]",
    ]
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved: {output_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generator-FE coupling")
    parser.add_argument("--bvtv", type=float, default=0.30)
    parser.add_argument("--xy", type=int, default=32)
    parser.add_argument("--z", type=int, default=16)
    parser.add_argument("--voxel-um", type=float, default=39.0)
    parser.add_argument("--base-sigma", type=float, default=2.5)
    parser.add_argument("--warp-amp", type=float, default=1.2)
    parser.add_argument("--warp-sigma", type=float, default=12.0)
    parser.add_argument("--E-bone", type=float, default=18000.0)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--strain", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load", type=str, default="all",
                        choices=["compression", "tension", "torque", "all"])
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("GENERATOR-FE COUPLING (v15.3 -> scikit-fem)")
    print("=" * 60)

    # Step 1: Generate bone volume (once)
    vol = generate_bone_volume(
        nx=args.xy, ny=args.xy, nz=args.z,
        target_bvtv=args.bvtv,
        voxel_um=args.voxel_um,
        base_sigma=args.base_sigma,
        warp_amp=args.warp_amp,
        warp_sigma=args.warp_sigma,
        seed=args.seed,
    )

    # Step 2: Run FE for each load case
    voxel_mm = args.voxel_um / 1000.0
    load_cases = ["compression", "tension", "torque"] if args.load == "all" else [args.load]
    out_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR

    all_results = {}
    for lc in load_cases:
        print(f"\n{'─'*60}")
        strain_val = args.strain if lc != "torque" else args.strain  # radians for torque
        fe = run_fe_analysis(
            vol["bone_mask"],
            voxel_size_mm=voxel_mm,
            load_type=lc,
            E_bone=args.E_bone,
            nu=args.nu,
            applied_strain=strain_val,
        )
        all_results[lc] = fe

        out_path = str(out_dir / f"results_{lc}.png")
        plot_results(vol, fe, output_path=out_path)

    # Step 3: Print comparison summary
    print(f"\n{'='*60}")
    print("SUMMARY — all load cases")
    print(f"{'='*60}")
    print(f"  Generator: v15.3 zero-crossing")
    print(f"  Volume: {args.xy}x{args.xy}x{args.z} at {args.voxel_um} um")
    print(f"  BV/TV: {vol['morphometrics']['BVTV']:.3f} (target {args.bvtv:.3f})")
    print()
    for lc, fe in all_results.items():
        s = fe["strain_field"]
        if lc in ("compression", "tension"):
            print(f"  {lc:>12}: E_app={fe['apparent_modulus']:.1f} MPa | "
                  f"E/E_voigt={fe['apparent_modulus']/fe['voigt_bound']:.3f} | "
                  f"eps_zz=[{s['eps_zz'].min():.5f}, {s['eps_zz'].max():.5f}] | "
                  f"von Mises max={s['eps_von_mises'].max():.5f}")
        elif lc == "torque":
            print(f"  {lc:>12}: G_app={fe['apparent_shear_modulus']:.1f} MPa | "
                  f"theta={np.degrees(fe['applied_strain']):.2f} deg | "
                  f"eps_xy=[{s['eps_xy'].min():.5f}, {s['eps_xy'].max():.5f}] | "
                  f"von Mises max={s['eps_von_mises'].max():.5f}")


if __name__ == "__main__":
    main()