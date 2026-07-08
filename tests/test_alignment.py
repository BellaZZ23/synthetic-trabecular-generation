"""
tests/test_alignment.py
========================
Tests for trabecular.alignment — volume alignment and resampling.

Run with plots:
    pytest tests/test_alignment.py --plots -v
"""
import numpy as np
import pytest
from trabecular.alignment import align_volumes, AlignmentReport, _common_crop, _normalise


class TestHelpers:
    def test_common_crop_same_shape(self):
        a = np.ones((10, 20, 30))
        b = np.ones((10, 20, 30))
        ca, cb, shape = _common_crop(a, b)
        assert shape == (10, 20, 30)
        assert ca.shape == cb.shape == (10, 20, 30)

    def test_common_crop_different_shapes(self):
        a = np.ones((12, 24, 32))
        b = np.ones((10, 20, 28))
        ca, cb, shape = _common_crop(a, b)
        assert shape == (10, 20, 28)
        assert ca.shape == cb.shape == (10, 20, 28)

    def test_normalise_range(self, rng):
        v = rng.standard_normal((10, 10, 10))
        n = _normalise(v)
        assert float(n.min()) >= 0.0 - 1e-6
        assert float(n.max()) <= 1.0 + 1e-6

    def test_normalise_all_zeros(self):
        v = np.zeros((5, 5, 5))
        n = _normalise(v)
        assert np.allclose(n, 0)


class TestAlignmentReport:
    def test_summary_string(self):
        r = AlignmentReport(
            target_voxel_mm=0.05,
            common_shape=(20, 30, 30),
            shift_voxels=(0.0, 0.0, 0.0),
            method="resample",
            overlap_fraction=0.85,
        )
        s = r.summary()
        assert "resample" in s
        assert "50.0 um" in s
        assert "0.85" in s


class TestAlignVolumes:
    def test_same_voxel_same_shape(self, small_mask, grayscale_volume):
        """Same voxel size → no resampling, just common crop (no-op here)."""
        vm, vf = 0.039, 0.039
        m_a, f_a, report = align_volumes(
            grayscale_volume, grayscale_volume, vm, vf,
        )
        assert m_a.shape == f_a.shape
        assert report.method == "resample"
        assert abs(report.target_voxel_mm - 0.039) < 1e-6

    def test_different_voxel_sizes(self, small_mask, grayscale_volume):
        """39 µm synthetic vs 50 µm real — result should have consistent shapes."""
        # Create a "real" volume at 50 µm (coarser = smaller array)
        from scipy.ndimage import zoom
        real = zoom(grayscale_volume.astype(float), zoom=0.039/0.050, order=1)
        m_a, f_a, report = align_volumes(
            grayscale_volume, real,
            moving_voxel_mm=0.039, fixed_voxel_mm=0.050,
        )
        assert m_a.shape == f_a.shape
        # Target defaults to coarser (0.050)
        assert abs(report.target_voxel_mm - 0.050) < 1e-6
        assert "coarser" in " ".join(report.notes).lower()

    def test_custom_target_voxel(self, grayscale_volume):
        """Explicit target_voxel_mm is respected."""
        m_a, f_a, report = align_volumes(
            grayscale_volume, grayscale_volume,
            moving_voxel_mm=0.039, fixed_voxel_mm=0.039,
            target_voxel_mm=0.050,
        )
        assert abs(report.target_voxel_mm - 0.050) < 1e-6

    def test_rigid_mode_runs(self, grayscale_volume):
        """rigid mode should run without error (scikit-image available)."""
        m_a, f_a, report = align_volumes(
            grayscale_volume, grayscale_volume,
            moving_voxel_mm=0.039, fixed_voxel_mm=0.039,
            method="rigid",
        )
        assert m_a.shape == f_a.shape
        # Method is either "rigid" (if skimage available) or fell back to "resample"
        assert report.method in ("rigid", "resample")

    def test_mask_overlap_fraction(self, small_mask):
        """Identical masks → overlap fraction = 1.0."""
        mask_f = small_mask.astype(float)
        _, _, report = align_volumes(
            mask_f, mask_f,
            moving_voxel_mm=0.039, fixed_voxel_mm=0.039,
            moving_mask=small_mask, fixed_mask=small_mask,
        )
        assert abs(report.overlap_fraction - 1.0) < 0.01

    def test_plot_alignment(self, grayscale_volume, save_plot):
        import matplotlib.pyplot as plt
        from scipy.ndimage import zoom

        real = zoom(grayscale_volume.astype(float), zoom=0.039/0.050, order=1)
        m_a, f_a, report = align_volumes(
            grayscale_volume, real,
            moving_voxel_mm=0.039, fixed_voxel_mm=0.050,
        )

        mid = m_a.shape[0] // 2
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))

        axes[0].imshow(grayscale_volume[grayscale_volume.shape[0]//2].T,
                       cmap="gray", origin="lower", vmin=0, vmax=255)
        axes[0].set_title("Synthetic (39 µm, original)")

        axes[1].imshow(real[real.shape[0]//2].T,
                       cmap="gray", origin="lower", vmin=0, vmax=255)
        axes[1].set_title("Real-proxy (50 µm, original)")

        axes[2].imshow(m_a[mid].T, cmap="gray", origin="lower", vmin=0, vmax=255)
        axes[2].set_title(f"Synthetic aligned to 50 µm grid\n{report.summary()}")

        plt.suptitle("Volume alignment — 39 µm synthetic → 50 µm common grid",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        save_plot(fig, "02_alignment")
