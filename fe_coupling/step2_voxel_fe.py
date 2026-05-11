"""
Step 2: Voxel-to-FE Conversion on a Small Test Volume
======================================================
Generates a small synthetic trabecular bone volume (32x32x16),
converts bone voxels to hexahedral elements, applies uniaxial
compression, solves, and validates apparent stiffness against
the Voigt upper bound.

This is the bridge between your generator and the FE solver.

Usage:
    python step2_voxel_fe.py
"""

import numpy as np
from skfem import *
from skfem.models.elasticity import linear_elasticity, lame_parameters
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time

# ══════════════════════════════════════════════════════════════
# 1. GENERATE A SMALL SYNTHETIC BONE VOLUME
# ══════════════════════════════════════════════════════════════
# Simplified version of your GRF zero-crossing generator
# for a 32x32x16 test volume

NX, NY, NZ = 32, 32, 16
TARGET_BVTV = 0.30  # target bone volume fraction
VOXEL_SIZE = 39e-3  # 39 µm in mm

print("=" * 60)
print("STEP 2: Voxel-to-FE on synthetic bone volume")
print("=" * 60)
print(f"\nVolume size: {NX}×{NY}×{NZ} voxels")
print(f"Voxel size: {VOXEL_SIZE*1000:.0f} µm")
print(f"Physical size: {NX*VOXEL_SIZE:.2f} × {NY*VOXEL_SIZE:.2f} × {NZ*VOXEL_SIZE:.2f} mm")

# Generate Gaussian random field
rng = np.random.default_rng(42)
noise = rng.standard_normal((NX, NY, NZ))
grf = gaussian_filter(noise, sigma=2.5)

# Zero-crossing wall detection: |field| < threshold → bone
# Binary search for threshold to hit target BV/TV
lo, hi = 0.0, 2.0
for _ in range(50):
    tau = (lo + hi) / 2
    bone = (np.abs(grf) < tau).astype(np.float64)
    bvtv = bone.mean()
    if bvtv < TARGET_BVTV:
        lo = tau
    else:
        hi = tau

bone_mask = (np.abs(grf) < tau).astype(int)
measured_bvtv = bone_mask.mean()

# Extract largest connected component to avoid singular matrix
# (disconnected bone fragments have no load path → unconstrained DOFs)
from scipy.ndimage import label
labeled, n_components = label(bone_mask)
if n_components > 1:
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    largest = sizes.argmax()
    bone_mask = (labeled == largest).astype(int)
    print(f"\nConnectivity: {n_components} components found, keeping largest ({bone_mask.sum()} voxels)")

measured_bvtv = bone_mask.mean()
n_bone_voxels = bone_mask.sum()
n_total_voxels = NX * NY * NZ

print(f"\nGenerated volume:")
print(f"  Target BV/TV: {TARGET_BVTV:.3f}")
print(f"  Measured BV/TV: {measured_bvtv:.3f}")
print(f"  Bone voxels: {n_bone_voxels} / {n_total_voxels}")

# ══════════════════════════════════════════════════════════════
# 2. BUILD HEX MESH FROM BONE VOXELS
# ══════════════════════════════════════════════════════════════
# Each bone voxel becomes one 8-node hex element.
# Marrow voxels are skipped (empty space / zero stiffness).

print(f"\nBuilding hex mesh from bone voxels...")
t0 = time.time()

# Node numbering: (NX+1) x (NY+1) x (NZ+1) potential nodes
# Node (i,j,k) → index = i + j*(NX+1) + k*(NX+1)*(NY+1)
def node_index(i, j, k):
    return i + j * (NX + 1) + k * (NX + 1) * (NY + 1)

# Build element connectivity for bone voxels only
elements = []  # each row: 8 node indices for one hex
bone_ijk = []  # store (i,j,k) of each bone voxel for reference

for k in range(NZ):
    for j in range(NY):
        for i in range(NX):
            if bone_mask[i, j, k] == 1:
                # 8 corners of the hex element at voxel (i,j,k)
                # scikit-fem hex ordering: 
                # bottom face (z=k): (i,j), (i+1,j), (i+1,j+1), (i,j+1)
                # top face (z=k+1):  (i,j), (i+1,j), (i+1,j+1), (i,j+1)
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
                bone_ijk.append((i, j, k))

elements = np.array(elements, dtype=np.int64)
print(f"  Raw elements: {len(elements)}")

# Build node coordinate array (all potential nodes)
all_nodes_x = np.arange(NX + 1, dtype=np.float64) * VOXEL_SIZE
all_nodes_y = np.arange(NY + 1, dtype=np.float64) * VOXEL_SIZE
all_nodes_z = np.arange(NZ + 1, dtype=np.float64) * VOXEL_SIZE

n_nodes_total = (NX + 1) * (NY + 1) * (NZ + 1)
coords = np.zeros((3, n_nodes_total))
for k in range(NZ + 1):
    for j in range(NY + 1):
        for i in range(NX + 1):
            idx = node_index(i, j, k)
            coords[0, idx] = all_nodes_x[i]
            coords[1, idx] = all_nodes_y[j]
            coords[2, idx] = all_nodes_z[k]

# Renumber: keep only nodes that appear in elements
used_nodes = np.unique(elements.flatten())
old_to_new = np.full(n_nodes_total, -1, dtype=np.int64)
old_to_new[used_nodes] = np.arange(len(used_nodes))

new_coords = coords[:, used_nodes]
new_elements = old_to_new[elements]

print(f"  Active nodes: {len(used_nodes)} (of {n_nodes_total} potential)")
print(f"  Active elements: {len(new_elements)}")

# Create scikit-fem mesh
mesh = MeshHex(new_coords, new_elements.T)
t1 = time.time()
print(f"  Mesh built in {t1-t0:.2f}s")
print(f"  Physical dimensions: x=[{new_coords[0].min():.3f}, {new_coords[0].max():.3f}] mm")
print(f"                       y=[{new_coords[1].min():.3f}, {new_coords[1].max():.3f}] mm")
print(f"                       z=[{new_coords[2].min():.3f}, {new_coords[2].max():.3f}] mm")

# ══════════════════════════════════════════════════════════════
# 3. ASSEMBLE AND SOLVE
# ══════════════════════════════════════════════════════════════
E_bone = 18_000.0  # MPa
nu = 0.3
lam, mu = lame_parameters(E_bone, nu)

elem = ElementVector(ElementHex1())
ib = Basis(mesh, elem)
print(f"\nFE model:")
print(f"  Elements: {mesh.nelements}")
print(f"  Nodes: {mesh.nvertices}")
print(f"  DOFs: {ib.N}")

print(f"\nAssembling stiffness matrix...")
t0 = time.time()
K = asm(linear_elasticity(lam, mu), ib)
t1 = time.time()
print(f"  Assembled in {t1-t0:.2f}s, shape={K.shape}, nnz={K.nnz}")

# Boundary conditions: symmetry on x=0, y=0, z=0 faces
# Prescribed displacement on z=max face
z_min = new_coords[2].min()
z_max = new_coords[2].max()
x_min = new_coords[0].min()
y_min = new_coords[1].min()
height = z_max - z_min

applied_strain = 0.01  # 1% compression
applied_disp = -applied_strain * height

print(f"\nBoundary conditions:")
print(f"  Specimen height: {height:.3f} mm")
print(f"  Applied strain: {applied_strain}")
print(f"  Applied displacement: {applied_disp:.6f} mm")

tol = VOXEL_SIZE * 0.1  # tolerance for face detection

dofs_x0 = ib.get_dofs(mesh.nodes_satisfying(lambda x: x[0] < x_min + tol))
dofs_y0 = ib.get_dofs(mesh.nodes_satisfying(lambda x: x[1] < y_min + tol))
dofs_z0 = ib.get_dofs(mesh.nodes_satisfying(lambda x: x[2] < z_min + tol))
dofs_z1 = ib.get_dofs(mesh.nodes_satisfying(lambda x: x[2] > z_max - tol))

x_prescribed = np.zeros(ib.N)
dofs_D = []

# Symmetry BCs
for dof in dofs_x0.nodal['u^1']:
    dofs_D.append(dof)
for dof in dofs_y0.nodal['u^2']:
    dofs_D.append(dof)
for dof in dofs_z0.nodal['u^3']:
    dofs_D.append(dof)

# Prescribed compression on top
for dof in dofs_z1.nodal['u^3']:
    x_prescribed[dof] = applied_disp
    dofs_D.append(dof)

dofs_D = np.unique(np.array(dofs_D))
print(f"  Constrained DOFs: {len(dofs_D)} (of {ib.N})")

print(f"\nSolving...")
t0 = time.time()
f = np.zeros(ib.N)
u = solve(*condense(K, f, x=x_prescribed, D=dofs_D))
t1 = time.time()
print(f"  Solved in {t1-t0:.2f}s")

# ══════════════════════════════════════════════════════════════
# 4. EXTRACT RESULTS AND VALIDATE
# ══════════════════════════════════════════════════════════════

# Displacement components
ux = u[ib.nodal_dofs[0]]
uy = u[ib.nodal_dofs[1]]
uz = u[ib.nodal_dofs[2]]

print(f"\nDisplacement field:")
print(f"  ux range: [{ux.min():.6f}, {ux.max():.6f}] mm")
print(f"  uy range: [{uy.min():.6f}, {uy.max():.6f}] mm")
print(f"  uz range: [{uz.min():.6f}, {uz.max():.6f}] mm")

# Compute reaction force on top face → apparent stiffness
f_reaction = K @ u
top_z_dofs = dofs_z1.nodal['u^3']
reaction_z = np.sum(f_reaction[top_z_dofs])

# Cross-section area = full volume cross-section (not just bone)
area_full = (NX * VOXEL_SIZE) * (NY * VOXEL_SIZE)
sigma_apparent = -reaction_z / area_full
E_apparent = sigma_apparent / applied_strain

# Voigt upper bound
E_voigt = measured_bvtv * E_bone

print(f"\n{'='*60}")
print("APPARENT STIFFNESS VALIDATION")
print(f"{'='*60}")
print(f"  Reaction force (z, top): {reaction_z:.4f} N")
print(f"  Cross-section area: {area_full:.4f} mm²")
print(f"  Apparent stress: {sigma_apparent:.2f} MPa")
print(f"  Apparent modulus: {E_apparent:.1f} MPa")
print(f"  Voigt upper bound: {E_voigt:.1f} MPa (BV/TV × E_bone)")
print(f"  E_apparent / E_voigt: {E_apparent/E_voigt:.3f}")

if E_apparent <= E_voigt * 1.01:  # 1% tolerance for numerics
    print(f"\n  ✓ VOIGT BOUND CHECK PASSED")
    print(f"    Apparent stiffness is below the upper bound as expected.")
else:
    print(f"\n  ✗ VOIGT BOUND CHECK FAILED")
    print(f"    E_apparent > E_voigt — check BCs and material assignment.")

print(f"\n  E_apparent / E_bone: {E_apparent/E_bone:.4f}")
print(f"  (For comparison, BV/TV = {measured_bvtv:.3f})")
print(f"  The ratio E_app/E_bone is typically less than BV/TV")
print(f"  for trabecular structures due to bending effects.")

# ══════════════════════════════════════════════════════════════
# 5. VISUALISE DISPLACEMENT FIELD
# ══════════════════════════════════════════════════════════════

# Map displacements back to the voxel grid for visualisation
# Pick a mid-slice in z
mid_z = NZ // 2

# Build a displacement magnitude map for the mid-slice
disp_mag_slice = np.full((NX, NY), np.nan)
uz_slice = np.full((NX, NY), np.nan)

# For each node, find its grid position and store displacement
node_coords = new_coords
for n_idx in range(mesh.nvertices):
    x, y, z = node_coords[:, n_idx]
    # Check if this node is at or near the mid-slice z level
    z_target = (mid_z + 0.5) * VOXEL_SIZE  # center of mid-slice voxels
    if abs(z - mid_z * VOXEL_SIZE) < tol or abs(z - (mid_z + 1) * VOXEL_SIZE) < tol:
        i = int(round(x / VOXEL_SIZE))
        j = int(round(y / VOXEL_SIZE))
        if 0 <= i < NX and 0 <= j < NY:
            mag = np.sqrt(ux[n_idx]**2 + uy[n_idx]**2 + uz[n_idx]**2)
            if np.isnan(disp_mag_slice[i, j]) or mag > disp_mag_slice[i, j]:
                disp_mag_slice[i, j] = mag
                uz_slice[i, j] = uz[n_idx]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Bone structure at mid-slice
ax = axes[0]
ax.imshow(bone_mask[:, :, mid_z].T, cmap='gray', origin='lower',
          extent=[0, NX*VOXEL_SIZE, 0, NY*VOXEL_SIZE])
ax.set_title(f'Bone structure (z-slice {mid_z})')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')

# Displacement magnitude
ax = axes[1]
im = ax.imshow(disp_mag_slice.T, cmap='hot', origin='lower',
               extent=[0, NX*VOXEL_SIZE, 0, NY*VOXEL_SIZE])
ax.set_title('Displacement magnitude [mm]')
ax.set_xlabel('x [mm]')
plt.colorbar(im, ax=ax, label='|u| [mm]')

# Z-displacement
ax = axes[2]
im = ax.imshow(uz_slice.T, cmap='RdBu', origin='lower',
               extent=[0, NX*VOXEL_SIZE, 0, NY*VOXEL_SIZE])
ax.set_title('uz displacement [mm]')
ax.set_xlabel('x [mm]')
plt.colorbar(im, ax=ax, label='uz [mm]')

plt.tight_layout()
plt.savefig('fe_coupling/step2_results.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: step2_results.png")
print("Done!")
