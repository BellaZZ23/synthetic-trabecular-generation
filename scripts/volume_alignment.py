"""
volume_alignment.py
====================
Spatial alignment of synthetic and real (D2IM / DVC) volumetric fields onto a
common coordinate grid, so that simulated FE strain and measured displacement
fields can be compared on a like-for-like basis.

Why this exists
---------------
The synthetic generator and the real microCT/DVC scans live on different grids:
  * different voxel sizes        (e.g. 39 um synthetic vs 50 um D2IM)
  * different array shapes / FOV
  * different coordinate origins

Comparing them directly (ravel + Pearson r) is invalid: the arrays have
different lengths and the voxels do not refer to the same physical points.
This module brings both volumes onto a shared grid *before* any metric is taken.

One solid, widely used approach (extend later if needed):
  1. resample both volumes to a common (target) voxel size  -> scipy.ndimage.zoom
  2. centre-crop to a common field of view                  -> shared shape
  3. (optional) estimate a rigid translation between the
     bone envelopes and shift the moving volume into register
                                                            -> phase cross-correlation
  4. return aligned (moving, fixed) plus a report of what was done

Dependencies: numpy, scipy. scikit-image is optional and only needed for the
"rigid" translation step (it is already used elsewhere in the pipeline).
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
from scipy.ndimage import zoom, shift as nd_shift

try:
    from skimage.registration import phase_cross_correlation
    _HAS_SKIMAGE_REG = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_SKIMAGE_REG = False


@dataclass
class AlignmentReport:
    """Lightweight record of what the alignment did (useful for logs / demo)."""
    target_voxel_mm: float
    common_shape: Tuple[int, int, int]
    shift_voxels: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    method: str = "resample"
    overlap_fraction: float = 1.0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        sx = ", ".join(f"{s:+.2f}" for s in self.shift_voxels)
        return (
            f"{self.method} | target voxel {self.target_voxel_mm * 1000:.1f} um | "
            f"grid {self.common_shape} | shift (z,y,x)=({sx}) vox | "
            f"mask overlap {self.overlap_fraction:.2f}"
        )


def _resample_to_voxel(vol, src_voxel_mm, target_voxel_mm, order):
    """Resample a 3D array from its source voxel size to the target voxel size."""
    if abs(src_voxel_mm - target_voxel_mm) < 1e-9:
        return vol.astype(float, copy=True)
    factor = src_voxel_mm / target_voxel_mm  # >1 upsample (finer), <1 downsample (coarser)
    return zoom(vol.astype(float), zoom=factor, order=order, mode="nearest")


def _common_crop(a, b):
    """Centre-crop two arrays to their common (minimum) shape on every axis."""
    out_shape = tuple(min(sa, sb) for sa, sb in zip(a.shape, b.shape))

    def crop(arr):
        slices = []
        for full, want in zip(arr.shape, out_shape):
            start = (full - want) // 2
            slices.append(slice(start, start + want))
        return arr[tuple(slices)]

    return crop(a), crop(b), out_shape


def _normalise(v):
    """Min-max normalise a volume to [0, 1], treating non-finite voxels as 0."""
    finite = np.isfinite(v)
    if not finite.any():
        return np.zeros_like(v, dtype=float)
    lo = float(np.nanmin(v[finite]))
    hi = float(np.nanmax(v[finite]))
    out = (v - lo) / (hi - lo + 1e-8)
    out[~finite] = 0.0
    return out


def align_volumes(
    moving,
    fixed,
    moving_voxel_mm,
    fixed_voxel_mm,
    moving_mask=None,
    fixed_mask=None,
    target_voxel_mm: Optional[float] = None,
    method: str = "resample",
    moving_is_mask: bool = False,
    fixed_is_mask: bool = False,
):
    """
    Bring ``moving`` onto the same physical grid as ``fixed``.

    Parameters
    ----------
    moving, fixed : 3D arrays (z, y, x)
        The field to be aligned and the reference field.
    moving_voxel_mm, fixed_voxel_mm : float
        Voxel sizes (mm) of the two volumes.
    moving_mask, fixed_mask : 3D bool arrays, optional
        Bone masks, used only to report a structural overlap fraction.
    target_voxel_mm : float, optional
        Voxel size both volumes are resampled to. Defaults to the *coarser* of
        the two grids so no detail is invented.
    method : {"resample", "rigid"}
        "resample" -> grid match only (common voxel + common FOV).
        "rigid"    -> resample, then estimate a translation (phase
                      cross-correlation) and shift ``moving`` into register.
    moving_is_mask, fixed_is_mask : bool
        If True the corresponding volume is resampled with nearest-neighbour
        (order 0) instead of linear interpolation.

    Returns
    -------
    moving_aligned, fixed_resampled, report : (ndarray, ndarray, AlignmentReport)
        The two volumes share the same shape and grid after this call.
    """
    notes: List[str] = []

    if target_voxel_mm is None:
        target_voxel_mm = max(moving_voxel_mm, fixed_voxel_mm)
        notes.append(
            f"target voxel defaulted to coarser grid ({target_voxel_mm * 1000:.1f} um)"
        )

    m_order = 0 if moving_is_mask else 1
    f_order = 0 if fixed_is_mask else 1
    m_rs = _resample_to_voxel(moving, moving_voxel_mm, target_voxel_mm, m_order)
    f_rs = _resample_to_voxel(fixed, fixed_voxel_mm, target_voxel_mm, f_order)

    m_rs, f_rs, common_shape = _common_crop(m_rs, f_rs)

    shift_voxels: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    if method == "rigid":
        if not _HAS_SKIMAGE_REG:
            notes.append("scikit-image registration unavailable; using resample-only")
            method = "resample"
        else:
            ref_m = _normalise(m_rs)
            ref_f = _normalise(f_rs)
            try:
                shift_est, _, _ = phase_cross_correlation(
                    ref_f, ref_m, upsample_factor=4
                )
                m_rs = nd_shift(m_rs, shift=shift_est, order=1, mode="nearest")
                shift_voxels = tuple(float(s) for s in shift_est)
                notes.append("rigid translation via phase cross-correlation")
            except Exception as exc:  # pragma: no cover - defensive
                notes.append(f"rigid alignment failed ({exc}); using resample-only")
                method = "resample"

    # Structural overlap fraction (rough QA number) if masks are available.
    overlap = 1.0
    if moving_mask is not None and fixed_mask is not None:
        mm = _resample_to_voxel(
            moving_mask.astype(float), moving_voxel_mm, target_voxel_mm, 0
        )
        fm = _resample_to_voxel(
            fixed_mask.astype(float), fixed_voxel_mm, target_voxel_mm, 0
        )
        mm, fm, _ = _common_crop(mm, fm)
        if any(s != 0.0 for s in shift_voxels):
            mm = nd_shift(mm, shift=shift_voxels, order=0, mode="nearest")
        union = int(np.logical_or(mm > 0.5, fm > 0.5).sum())
        inter = int(np.logical_and(mm > 0.5, fm > 0.5).sum())
        overlap = float(inter / union) if union > 0 else 1.0

    report = AlignmentReport(
        target_voxel_mm=float(target_voxel_mm),
        common_shape=tuple(int(s) for s in common_shape),
        shift_voxels=shift_voxels,
        method=method,
        overlap_fraction=overlap,
        notes=notes,
    )
    return m_rs, f_rs, report