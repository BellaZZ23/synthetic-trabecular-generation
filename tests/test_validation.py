"""
tests/test_validation.py
=========================
Tests for trabecular.validation — FE-driven known-deformation validation.

These tests use a mock FE result dict (not the real solver) so they run
without scikit-fem installed. The published FE-driven RMSE (0.27 voxels)
is validated separately when scikit-fem is available.

Run with plots:
    pytest tests/test_validation.py --plots -v
"""
import numpy as np
import pytest
from trabecular.validation import fe_displacement_to_voxel_field


def _make_mock_fe_result(mask: np.ndarray, voxel_mm: float,
                         disp_mm: float = 0.01) -> dict:
    """
    Build a minimal mock FE result dict with a known uniform Z-displacement.

    Places nodes at every voxel corner. Each node's uz = disp_mm (uniform).
    This gives a uniform voxel-centre field of disp_mm / voxel_mm voxels in Z.
    """
    nz, ny, nx = mask.shape

    # Mesh: nodes at all (nx+1)*(ny+1)*(nz+1) corners
    iz_n, iy_n, ix_n = np.mgrid[0:nz+1, 0:ny+1, 0:nx+1]
    x_nodes = ix_n.ravel() * voxel_mm
    y_nodes = iy_n.ravel() * voxel_mm
    z_nodes = iz_n.ravel() * voxel_mm
    p       = np.stack([x_nodes, y_nodes, z_nodes])    # (3, n_nodes)

    n_nodes = p.shape[1]
    ux  = np.zeros(n_nodes)
    uy  = np.zeros(n_nodes)
    uz  = np.full(n_nodes, disp_mm)   # uniform Z-displacement

    class MockMesh:
        pass

    mesh   = MockMesh()
    mesh.p = p

    return {
        "mesh":         mesh,
        "displacement": (ux, uy, uz),
    }


class TestFeDisplacementToVoxelField:
    def test_output_shape(self, small_mask):
        voxel_mm  = 0.039
        fe_result = _make_mock_fe_result(small_mask, voxel_mm, disp_mm=0.01)
        u = fe_displacement_to_voxel_field(fe_result, small_mask, voxel_um=39.0)
        assert u.shape == (3, *small_mask.shape), \
            f"Expected (3, {small_mask.shape}), got {u.shape}"

    def test_uniform_z_displacement(self, small_mask):
        """Uniform uz = voxel_mm → dense field u[0] ≈ 1 voxel everywhere."""
        voxel_um  = 39.0
        voxel_mm  = voxel_um / 1000.0
        fe_result = _make_mock_fe_result(small_mask, voxel_mm, disp_mm=voxel_mm)
        u = fe_displacement_to_voxel_field(fe_result, small_mask, voxel_um=voxel_um)
        # u[0] is Z-displacement; should be ~1.0 everywhere (1 voxel)
        bone_z = u[0][small_mask > 0]
        assert bone_z.mean() == pytest.approx(1.0, abs=0.05), \
            f"Expected u_z ≈ 1.0 voxel, got {bone_z.mean():.4f}"

    def test_zero_displacement_gives_zero_field(self, small_mask):
        voxel_mm  = 0.039
        fe_result = _make_mock_fe_result(small_mask, voxel_mm, disp_mm=0.0)
        u = fe_displacement_to_voxel_field(fe_result, small_mask, voxel_um=39.0)
        assert np.allclose(u, 0.0, atol=1e-10)

    def test_xy_displacement_is_zero_for_pure_z(self, small_mask):
        """If only uz is set, u[1] (Y) and u[2] (X) should be zero."""
        voxel_mm  = 0.039
        fe_result = _make_mock_fe_result(small_mask, voxel_mm, disp_mm=0.05)
        u = fe_displacement_to_voxel_field(fe_result, small_mask, voxel_um=39.0)
        assert np.allclose(u[1], 0.0, atol=1e-10)   # Y-displacement
        assert np.allclose(u[2], 0.0, atol=1e-10)   # X-displacement

    def test_displacement_scales_with_amplitude(self, small_mask):
        """Doubling disp_mm should double the voxel-unit field."""
        voxel_um  = 39.0
        voxel_mm  = voxel_um / 1000.0
        fe1 = _make_mock_fe_result(small_mask, voxel_mm, disp_mm=0.01)
        fe2 = _make_mock_fe_result(small_mask, voxel_mm, disp_mm=0.02)
        u1  = fe_displacement_to_voxel_field(fe1, small_mask, voxel_um=voxel_um)
        u2  = fe_displacement_to_voxel_field(fe2, small_mask, voxel_um=voxel_um)
        ratio = u2[0][small_mask > 0].mean() / (u1[0][small_mask > 0].mean() + 1e-10)
        assert abs(ratio - 2.0) < 0.05, f"Expected 2× scaling, got ratio={ratio:.3f}"

    def test_plot_voxel_field(self, small_mask, save_plot):
        import matplotlib.pyplot as plt

        voxel_um  = 39.0
        voxel_mm  = voxel_um / 1000.0
        # Linearly increasing Z-displacement (simulates compression)
        fe_result = _make_mock_fe_result(small_mask, voxel_mm, disp_mm=voxel_mm * 0.5)
        u = fe_displacement_to_voxel_field(fe_result, small_mask, voxel_um=voxel_um)

        mid_z   = small_mask.shape[0] // 2
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))

        axes[0].imshow(small_mask[mid_z].T, cmap="gray", origin="lower")
        axes[0].set_title(f"Bone mask (z={mid_z})")

        im1 = axes[1].imshow(u[0, mid_z].T, cmap="RdBu_r", origin="lower")
        axes[1].set_title("u_z displacement field (voxels)")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, label="voxels")

        mag = np.sqrt(np.sum(u**2, axis=0))
        im2 = axes[2].imshow(mag[mid_z].T, cmap="plasma", origin="lower", vmin=0)
        axes[2].set_title("Displacement magnitude (voxels)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, label="voxels")

        plt.suptitle(
            "FE nodal displacement → dense voxel field conversion\n"
            "(mock FE: uniform 0.5-voxel Z-displacement)",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        save_plot(fig, "07_fe_voxel_field")
