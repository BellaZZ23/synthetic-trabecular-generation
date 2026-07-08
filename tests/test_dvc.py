"""
tests/test_dvc.py
=================
Tests for trabecular.dvc — the phase-correlation DVC round-trip.

Key test: apply a KNOWN displacement field to a textured volume, recover it
with phase-correlation, and assert RMSE < 1 voxel. This is the core
validation result from the paper.

Run with plots:
    pytest tests/test_dvc.py --plots -v
"""
import numpy as np
import pytest
from trabecular.dvc import (
    warp_volume,
    recover_displacement,
    displacement_rmse,
    uniform_strain_field,
    shear_field,
    bending_field,
    smooth_random_field,
    sample_field_at_points,
)


# ── Field factories ───────────────────────────────────────────────────────────

class TestFieldFactories:
    def test_uniform_strain_shape(self, small_mask):
        S = small_mask.shape
        u = uniform_strain_field(S, axis=0, strain=0.01)
        assert u.shape == (3, *S)

    def test_uniform_strain_axis(self):
        S = (10, 20, 20)
        u = uniform_strain_field(S, axis=2, strain=0.05)
        # Displacement only along axis 2; axes 0 and 1 should be zero
        assert np.allclose(u[0], 0)
        assert np.allclose(u[1], 0)
        assert u[2].max() > 0

    def test_uniform_strain_gradient(self):
        S = (8, 16, 16)
        strain = 0.02
        u = uniform_strain_field(S, axis=1, strain=strain)
        # u[1][k, j, i] = strain * j  → gradient = strain
        grad = np.gradient(u[1], axis=1).mean()
        assert abs(grad - strain) < 0.01

    def test_shear_field_shape(self, small_mask):
        S = small_mask.shape
        u = shear_field(S, u_axis=0, grad_axis=2, gamma=0.01)
        assert u.shape == (3, *S)

    def test_bending_field_shape(self, small_mask):
        S = small_mask.shape
        u = bending_field(S, u_axis=0, grad_axis=1, amplitude=2.0)
        assert u.shape == (3, *S)

    def test_smooth_random_field_amplitude(self, small_mask):
        S = small_mask.shape
        amplitude = 1.5
        u = smooth_random_field(S, sigma=4.0, amplitude=amplitude, seed=0)
        assert u.shape == (3, *S)
        assert np.abs(u).max() <= amplitude + 1e-6

    def test_smooth_random_field_reproducible(self, small_mask):
        S = small_mask.shape
        u1 = smooth_random_field(S, seed=7)
        u2 = smooth_random_field(S, seed=7)
        assert np.allclose(u1, u2)


# ── Warping ───────────────────────────────────────────────────────────────────

class TestWarpVolume:
    def test_zero_field_identity(self, grayscale_volume):
        u = np.zeros((3, *grayscale_volume.shape))
        warped = warp_volume(grayscale_volume, u)
        assert np.allclose(warped, grayscale_volume, atol=1e-4)

    def test_warp_output_shape(self, grayscale_volume):
        u = np.zeros((3, *grayscale_volume.shape))
        warped = warp_volume(grayscale_volume, u)
        assert warped.shape == grayscale_volume.shape

    def test_warp_shifts_content(self, small_mask):
        # Rigid shift of 2 voxels along axis 2
        S   = (10, 20, 20)
        vol = np.zeros(S); vol[5, 10, 5:15] = 1.0
        u   = np.zeros((3, *S)); u[2] = -2.0   # shift content +2 along axis 2
        warped = warp_volume(vol, u)
        # Content should appear at columns 7–17 now
        assert warped[5, 10, 7:12].mean() > 0.5


# ── Scoring ───────────────────────────────────────────────────────────────────

class TestDisplacementRmse:
    def test_identical_fields_zero_rmse(self, small_mask):
        S = small_mask.shape
        u = smooth_random_field(S, seed=1)
        scores = displacement_rmse(u, u)
        assert scores["rmse_voxels"] < 1e-10

    def test_rmse_in_voxels(self, small_mask):
        S = small_mask.shape
        u_true = uniform_strain_field(S, axis=0, strain=0.01)
        u_est  = np.zeros_like(u_true)   # worst case: zero estimate
        scores = displacement_rmse(u_true, u_est)
        assert scores["rmse_voxels"] > 0

    def test_rmse_with_voxel_um(self, small_mask):
        S = small_mask.shape
        u = smooth_random_field(S, seed=2)
        scores = displacement_rmse(u, np.zeros_like(u), voxel_um=39.0)
        assert "rmse_um" in scores
        assert abs(scores["rmse_um"] - scores["rmse_voxels"] * 39.0) < 1e-6

    def test_rmse_with_mask(self, small_mask):
        S = small_mask.shape
        u_true = smooth_random_field(S, seed=3)
        u_est  = np.zeros_like(u_true)
        s_full = displacement_rmse(u_true, u_est)
        s_mask = displacement_rmse(u_true, u_est, mask=small_mask)
        # Masked and full RMSE differ (unless mask = all ones)
        assert isinstance(s_mask["rmse_voxels"], float)

    def test_sample_field_at_points(self, small_mask):
        S = small_mask.shape
        u = smooth_random_field(S, seed=4)
        pts = np.array([[2, 5, 5], [3, 8, 10], [0, 0, 0]])
        samples = sample_field_at_points(u, pts)
        assert samples.shape == (3, 3)


# ── Round-trip: impose → recover → score (THE key validation test) ────────────

class TestRoundTrip:
    """
    Apply a known uniform-strain field to a textured synthetic volume,
    recover with phase-correlation DVC, score recovered vs known.
    Target: RMSE < 1.0 voxel (published: ~0.53 voxels on synthetic data).
    """

    @pytest.fixture(scope="class")
    def round_trip_result(self, small_mask, grayscale_volume):
        """Run the round-trip once and cache the result for all assertions."""
        S      = small_mask.shape
        strain = -0.015
        u_true = uniform_strain_field(S, axis=0, strain=strain)
        defd   = warp_volume(grayscale_volume, u_true)
        centers, disps, dense_u = recover_displacement(
            grayscale_volume, defd,
            block=12, step=6, mask=small_mask, upsample=10,
        )
        scores = displacement_rmse(u_true, dense_u, mask=small_mask, voxel_um=39.0)
        return {
            "u_true": u_true, "dense_u": dense_u,
            "centers": centers, "disps": disps,
            "scores": scores,
        }

    def test_blocks_matched(self, round_trip_result):
        assert len(round_trip_result["centers"]) > 0, \
            "Phase-correlation found no valid blocks — increase volume size or reduce block size."

    def test_rmse_below_one_voxel(self, round_trip_result):
        rmse = round_trip_result["scores"]["rmse_voxels"]
        print(f"\n  DVC round-trip RMSE: {rmse:.3f} voxels")
        assert rmse < 1.0, f"RMSE {rmse:.3f} >= 1.0 voxel"

    def test_rmse_reported_in_microns(self, round_trip_result):
        assert "rmse_um" in round_trip_result["scores"]

    def test_plot_round_trip(self, round_trip_result, save_plot):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        res    = round_trip_result
        u_true = res["u_true"]
        dense  = res["dense_u"]
        S      = u_true.shape[1:]
        mid_z  = S[0] // 2

        diff_mag = np.sqrt(np.sum((u_true - dense) ** 2, axis=0))

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))

        im0 = axes[0].imshow(u_true[0, mid_z].T, cmap="RdBu_r", origin="lower")
        axes[0].set_title("Known field — u_z (axis 0, voxels)")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(dense[0, mid_z].T, cmap="RdBu_r", origin="lower")
        axes[1].set_title("Recovered field — u_z")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        im2 = axes[2].imshow(diff_mag[mid_z].T, cmap="inferno", origin="lower",
                              vmin=0)
        axes[2].set_title(
            f"Error magnitude (voxels)\nRMSE = "
            f"{res['scores']['rmse_voxels']:.3f} vox "
            f"({res['scores']['rmse_um']:.1f} µm)"
        )
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        plt.suptitle("DVC round-trip validation — known deformation vs recovered",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        save_plot(fig, "01_dvc_round_trip")
