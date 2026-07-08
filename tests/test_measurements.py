"""
tests/test_measurements.py
===========================
Tests for trabecular.fe.measurements — BoneJ-equivalent morphometrics.

Run with plots:
    pytest tests/test_measurements.py --plots -v
"""
import numpy as np
import pytest
from trabecular.fe.measurements import (
    measure_thickness,
    measure_connectivity,
    measure_all_bonej,
)

VOXEL_UM = 39.0
REQUIRED_KEYS = {
    "BVTV", "TbTh_um_p50", "TbTh_um_p90", "TbTh_um_mean",
    "TbSp_um_p50", "TbSp_um_p90", "TbSp_um_mean",
    "TbN_per_mm", "Euler", "connectivity", "n_components",
    "lcc_frac", "thickness_map", "spacing_map",
}


class TestMeasureThickness:
    def test_bone_thickness_nonzero(self, small_mask):
        result = measure_thickness(small_mask, VOXEL_UM, is_bone=True)
        assert result["p50"] > 0, "Median Tb.Th should be positive"

    def test_marrow_thickness_nonzero(self, small_mask):
        result = measure_thickness(small_mask, VOXEL_UM, is_bone=False)
        assert result["p50"] > 0, "Median Tb.Sp should be positive"

    def test_empty_mask_returns_zeros(self):
        empty = np.zeros((10, 10, 10), dtype=np.uint8)
        result = measure_thickness(empty, VOXEL_UM, is_bone=True)
        assert result["p50"] == 0.0

    def test_thickness_in_microns(self, small_mask):
        result = measure_thickness(small_mask, VOXEL_UM, is_bone=True)
        # For a 32×32 volume at 39 µm, Tb.Th should be between 1 and 10 voxels
        assert VOXEL_UM < result["p50"] < VOXEL_UM * 15

    def test_map_shape_matches_mask(self, small_mask):
        result = measure_thickness(small_mask, VOXEL_UM, is_bone=True)
        assert result["map"].shape == small_mask.shape

    def test_percentile_ordering(self, small_mask):
        result = measure_thickness(small_mask, VOXEL_UM, is_bone=True)
        assert result["p10"] <= result["p50"] <= result["p90"]


class TestMeasureConnectivity:
    def test_returns_required_keys(self, small_mask):
        result = measure_connectivity(small_mask)
        for key in ("euler_number", "connectivity", "n_components", "lcc_fraction"):
            assert key in result

    def test_lcc_fraction_range(self, small_mask):
        result = measure_connectivity(small_mask)
        assert 0.0 <= result["lcc_fraction"] <= 1.0

    def test_n_components_positive(self, small_mask):
        result = measure_connectivity(small_mask)
        assert result["n_components"] >= 1

    def test_euler_is_integer(self, small_mask):
        result = measure_connectivity(small_mask)
        assert isinstance(result["euler_number"], int)


class TestMeasureAllBonej:
    def test_required_keys_present(self, small_mask):
        result = measure_all_bonej(small_mask, VOXEL_UM, verbose=False)
        missing = REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_bvtv_matches_mask(self, small_mask):
        result = measure_all_bonej(small_mask, VOXEL_UM, verbose=False)
        expected_bvtv = float(small_mask.mean())
        assert abs(result["BVTV"] - expected_bvtv) < 1e-6

    def test_bvtv_in_range(self, small_mask):
        result = measure_all_bonej(small_mask, VOXEL_UM, verbose=False)
        assert 0.0 < result["BVTV"] < 1.0

    def test_tbth_positive(self, small_mask):
        result = measure_all_bonej(small_mask, VOXEL_UM, verbose=False)
        assert result["TbTh_um_p50"] > 0

    def test_tbsp_positive(self, small_mask):
        result = measure_all_bonej(small_mask, VOXEL_UM, verbose=False)
        assert result["TbSp_um_p50"] > 0

    def test_tbn_positive(self, small_mask):
        result = measure_all_bonej(small_mask, VOXEL_UM, verbose=False)
        assert result["TbN_per_mm"] > 0

    def test_lcc_frac_range(self, small_mask):
        result = measure_all_bonej(small_mask, VOXEL_UM, verbose=False)
        assert 0.0 <= result["lcc_frac"] <= 1.0

    def test_plot_morphometrics(self, small_mask, save_plot):
        import matplotlib.pyplot as plt

        result = measure_all_bonej(small_mask, VOXEL_UM, verbose=False)
        tmap   = result["thickness_map"]
        smap   = result["spacing_map"]
        mid_z  = small_mask.shape[0] // 2

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))

        axes[0].imshow(small_mask[mid_z].T, cmap="gray", origin="lower")
        axes[0].set_title(f"Bone mask (z={mid_z})\nBV/TV = {result['BVTV']:.3f}")

        im1 = axes[1].imshow(tmap[mid_z].T, cmap="plasma", origin="lower",
                              vmin=0, vmax=tmap[small_mask > 0].max() * 1.1)
        axes[1].set_title(
            f"Tb.Th map (µm)\np50 = {result['TbTh_um_p50']:.0f} µm"
        )
        plt.colorbar(im1, ax=axes[1], fraction=0.046, label="µm")

        im2 = axes[2].imshow(smap[mid_z].T, cmap="viridis", origin="lower",
                              vmin=0)
        axes[2].set_title(
            f"Tb.Sp map (µm)\np50 = {result['TbSp_um_p50']:.0f} µm"
        )
        plt.colorbar(im2, ax=axes[2], fraction=0.046, label="µm")

        plt.suptitle(
            f"BoneJ morphometrics | BV/TV={result['BVTV']:.3f} | "
            f"Tb.Th={result['TbTh_um_p50']:.0f} µm | "
            f"Tb.N={result['TbN_per_mm']:.2f} /mm | "
            f"LCC={result['lcc_frac']:.3f}",
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout()
        save_plot(fig, "03_morphometrics")
