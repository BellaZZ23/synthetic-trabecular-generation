"""
fe_deformation.py  --  use the micro-FE solver as a physically-real
known-deformation source for the validation loop.

Your FE solver (step3_generator_fe_coupling.run_fe_analysis) returns nodal
displacements in mm at mesh corners. This converts them to a dense voxel
displacement field (ndim, *bone_mask.shape) in VOXEL units, so deformation.py
can warp a volume by a REAL FE deformation, then recover + score it.

Axis mapping (their convention):
    bone_mask shape = (nz, ny, nx) = (axis0=Z, axis1=Y, axis2=X)
    FE gives ux (along X), uy (along Y), uz (along Z) at nodes, in mm.
    field u[0]=Z-disp=uz, u[1]=Y-disp=uy, u[2]=X-disp=ux, all in voxels.
"""

import numpy as np


def fe_displacement_to_voxel_field(fe_result, bone_mask, voxel_um):
    """FE nodal displacement (mm, corners) -> dense voxel field (3,*shape) in voxels."""
    mesh = fe_result["mesh"]
    ux, uy, uz = fe_result["displacement"]        # per used node, mm
    voxel_mm = voxel_um / 1000.0
    nz, ny, nx = bone_mask.shape

    P = np.asarray(mesh.p, float)                 # (3, n_nodes): x,y,z in mm
    ci = np.round(P[0] / voxel_mm).astype(int)    # x -> i  (0..nx)
    cj = np.round(P[1] / voxel_mm).astype(int)    # y -> j  (0..ny)
    ck = np.round(P[2] / voxel_mm).astype(int)    # z -> k  (0..nz)
    ok = (ci >= 0) & (ci <= nx) & (cj >= 0) & (cj <= ny) & (ck >= 0) & (ck <= nz)

    # corner grids in VOXEL units (component, k, j, i)
    Uc = np.full((3, nz + 1, ny + 1, nx + 1), np.nan, dtype=float)
    Uc[0, ck[ok], cj[ok], ci[ok]] = uz[ok] / voxel_mm     # axis0 = Z
    Uc[1, ck[ok], cj[ok], ci[ok]] = uy[ok] / voxel_mm     # axis1 = Y
    Uc[2, ck[ok], cj[ok], ci[ok]] = ux[ok] / voxel_mm     # axis2 = X
    Uc = np.nan_to_num(Uc, nan=0.0)               # marrow corners -> no motion

    # average the 8 corners of each cell -> voxel-centre field (3, nz, ny, nx)
    U = (
        Uc[:, :-1, :-1, :-1] + Uc[:, 1:, :-1, :-1] +
        Uc[:, :-1, 1:, :-1]  + Uc[:, :-1, :-1, 1:] +
        Uc[:, 1:, 1:, :-1]   + Uc[:, 1:, :-1, 1:]  +
        Uc[:, :-1, 1:, 1:]   + Uc[:, 1:, 1:, 1:]
    ) / 8.0
    return U


def make_fe_known_deformation(bone_mask, voxel_um, load_type="compression",
                              applied_strain=0.02, E_bone=18000.0, nu=0.3,
                              amplify=1.0, verbose=False):
    """Run FE and return (u_true_voxels, deformed_volume-ready field).
    `amplify` scales the field so DVC recovery is above the sub-voxel noise floor
    for testing (physically it's still the same deformation shape)."""
    from step3_generator_fe_coupling import run_fe_analysis
    fe = run_fe_analysis(bone_mask, voxel_um / 1000.0, load_type=load_type,
                         E_bone=E_bone, nu=nu, applied_strain=applied_strain,
                         verbose=verbose)
    u = fe_displacement_to_voxel_field(fe, bone_mask, voxel_um) * amplify
    return u, fe