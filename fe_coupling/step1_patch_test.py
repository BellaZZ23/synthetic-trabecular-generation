"""
Step 1: Single-Element Patch Test
=================================
Creates a single hexahedral element (unit cube), applies 1% uniaxial
compression in z, and verifies the displacement field matches the
analytical solution for linear elasticity.

If this passes, your element formulation, material model, and BC
application are all correct.

Usage:
    python step1_patch_test.py
"""

import numpy as np
from skfem import *
from skfem.models.elasticity import linear_elasticity, lame_parameters
import matplotlib.pyplot as plt

# ── Material properties (cortical bone, isotropic approximation) ──
E_bone = 18_000.0   # Young's modulus [MPa] - typical cortical bone
nu = 0.3             # Poisson's ratio

lam, mu = lame_parameters(E_bone, nu)
print(f"Material: E = {E_bone} MPa, ν = {nu}")
print(f"Lamé parameters: λ = {lam:.1f}, μ = {mu:.1f}")

# ── Create a unit cube mesh (single hex element) ──
mesh = MeshHex.init_tensor(
    np.array([0.0, 1.0]),  # x: 0 to 1
    np.array([0.0, 1.0]),  # y: 0 to 1
    np.array([0.0, 1.0]),  # z: 0 to 1
)
print(f"\nMesh: {mesh.nelements} element, {mesh.nvertices} nodes")
print(f"Node coordinates:\n{mesh.p.T}")

# ── Define the element and function space ──
# Trilinear hexahedral element for 3D vector field (displacement)
elem = ElementVector(ElementHex1())
ib = Basis(mesh, elem)
print(f"DOFs: {ib.N}")

# ── Assemble stiffness matrix ──
K = asm(linear_elasticity(lam, mu), ib)
print(f"Stiffness matrix: {K.shape}")

# ── Boundary conditions ──
# Applied strain: 1% compression in z
applied_strain = 0.01
applied_disp = -applied_strain * 1.0  # cube height = 1.0

# ── Boundary conditions for FREE uniaxial compression ──
# Goal: compress in z, allow free Poisson expansion in x and y.
#
# To prevent rigid body motion (3 translations + 3 rotations) we need
# exactly 6 constraints beyond the prescribed displacement:
#   - Bottom face (z=0): uz = 0 on ALL bottom nodes (prevents z-translation)
#   - Origin node (0,0,0): ux = 0, uy = 0 (prevents x,y translation)
#   - Node (1,0,0): uy = 0 (prevents rotation about z-axis)
#   - Node (0,1,0): ux = 0 (prevents rotation about z-axis, redundant but symmetric)
# This leaves all lateral DOFs free to expand under Poisson effect.
#
# Top face (z=1): uz = applied_disp on ALL top nodes

dofs_bottom = ib.get_dofs(lambda x: np.isclose(x[2], 0.0))
dofs_top = ib.get_dofs(lambda x: np.isclose(x[2], 1.0))

# ── Boundary conditions: SYMMETRY approach (textbook patch test) ──
# Fix ux=0 on x=0 face, uy=0 on y=0 face, uz=0 on z=0 face.
# Top face (z=1): uz = applied_disp.
# This models one octant of a symmetric compression with free Poisson expansion.
# The analytical solution is exact: ux = eps_xx*x, uy = eps_yy*y, uz = eps_zz*z

dofs_x0 = ib.get_dofs(lambda x: np.isclose(x[0], 0.0))  # x=0 face
dofs_y0 = ib.get_dofs(lambda x: np.isclose(x[1], 0.0))  # y=0 face
dofs_z0 = ib.get_dofs(lambda x: np.isclose(x[2], 0.0))  # z=0 face (bottom)
dofs_z1 = ib.get_dofs(lambda x: np.isclose(x[2], 1.0))  # z=1 face (top)

x_prescribed = np.zeros(ib.N)
dofs_D = []

# Symmetry: ux=0 on x=0 face
for dof in dofs_x0.nodal['u^1']:
    dofs_D.append(dof)

# Symmetry: uy=0 on y=0 face
for dof in dofs_y0.nodal['u^2']:
    dofs_D.append(dof)

# Symmetry: uz=0 on z=0 face (bottom)
for dof in dofs_z0.nodal['u^3']:
    dofs_D.append(dof)

# Prescribed: uz = applied_disp on z=1 face (top)
for dof in dofs_z1.nodal['u^3']:
    x_prescribed[dof] = applied_disp
    dofs_D.append(dof)

dofs_D = np.unique(np.array(dofs_D))

print(f"\nBCs applied (symmetry approach):")
print(f"  ux=0 on x=0 face: {len(dofs_x0.nodal['u^1'])} DOFs")
print(f"  uy=0 on y=0 face: {len(dofs_y0.nodal['u^2'])} DOFs")
print(f"  uz=0 on z=0 face: {len(dofs_z0.nodal['u^3'])} DOFs")
print(f"  uz={applied_disp} on z=1 face: {len(dofs_z1.nodal['u^3'])} DOFs")
print(f"  Total constrained: {len(dofs_D)} DOFs (of {ib.N})")

dofs_D = np.unique(np.array(dofs_D))

n_bottom_uz = len(dofs_bottom.nodal['u^3'])
n_top_uz = len(dofs_top.nodal['u^3'])
print(f"\nBCs applied:")
print(f"  Bottom uz=0: {n_bottom_uz} DOFs")
print(f"  Top uz={applied_disp}: {n_top_uz} DOFs")
print(f"  Rigid body constraints: symmetry BCs on 3 faces")
print(f"  Total constrained: {len(dofs_D)} DOFs")

# ── Solve ──
f = np.zeros(ib.N)
u = solve(*condense(K, f, x=x_prescribed, D=dofs_D))
print(f"\nSolution vector (all {len(u)} DOFs):")

# ── Extract displacements at each node ──
ux = u[ib.nodal_dofs[0]]  # x-displacements
uy = u[ib.nodal_dofs[1]]  # y-displacements
uz = u[ib.nodal_dofs[2]]  # z-displacements

print("\nNode displacements:")
print(f"  {'Node':>4}  {'x':>8}  {'y':>8}  {'z':>8}  |  {'ux':>10}  {'uy':>10}  {'uz':>10}")
print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  |  {'─'*10}  {'─'*10}  {'─'*10}")
for i in range(mesh.nvertices):
    x, y, z = mesh.p[:, i]
    print(f"  {i:>4}  {x:>8.3f}  {y:>8.3f}  {z:>8.3f}  |  {ux[i]:>10.6f}  {uy[i]:>10.6f}  {uz[i]:>10.6f}")

# ── Analytical solution ──
# Uniaxial compression in z with lateral faces free:
#   epsilon_zz = -0.01 (applied)
#   sigma_zz = E * epsilon_zz = -180 MPa
#   epsilon_xx = epsilon_yy = -nu * epsilon_zz = +0.003 (Poisson expansion)
#   uz(z) = epsilon_zz * z
#   ux(x) = -nu * epsilon_zz * x
#   uy(y) = -nu * epsilon_zz * y
eps_zz = -applied_strain
eps_xx = -nu * eps_zz  # lateral expansion
eps_yy = eps_xx

print(f"\n{'='*60}")
print("ANALYTICAL vs NUMERICAL COMPARISON")
print(f"{'='*60}")
print(f"Applied strain (zz): {eps_zz}")
print(f"Expected lateral strain (xx=yy): {eps_xx}")

max_err = 0.0
for i in range(mesh.nvertices):
    x, y, z = mesh.p[:, i]
    ux_exact = eps_xx * x
    uy_exact = eps_yy * y
    uz_exact = eps_zz * z
    
    err_x = abs(ux[i] - ux_exact)
    err_y = abs(uy[i] - uy_exact)
    err_z = abs(uz[i] - uz_exact)
    max_err = max(max_err, err_x, err_y, err_z)

print(f"\nMaximum absolute error: {max_err:.2e}")

if max_err < 1e-10:
    print("\n✓ PATCH TEST PASSED — numerical solution matches analytical exactly")
    print("  Your element formulation, material model, and BCs are correct.")
else:
    print(f"\n✗ PATCH TEST FAILED — error = {max_err:.2e}")
    print("  Check: BC application, material tensor, element type")

# ── Compute apparent stiffness as a sanity check ──
f_reaction = K @ u
# Reaction forces on top face, z-component
top_z_dofs = dofs_z1.nodal['u^3']
reaction_z = np.sum(f_reaction[top_z_dofs])
area = 1.0 * 1.0  # cross-section area of unit cube
sigma_apparent = -reaction_z / area  # compression is negative force
E_apparent = sigma_apparent / applied_strain

print(f"\nApparent stiffness check:")
print(f"  Reaction force (z, top face): {reaction_z:.4f}")
print(f"  Apparent modulus: {E_apparent:.1f} MPa")
print(f"  Expected (E_bone): {E_bone:.1f} MPa")
print(f"  Match: {'✓' if abs(E_apparent - E_bone) < 0.1 else '✗'}")
