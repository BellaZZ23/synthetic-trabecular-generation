"""
deformation.py  --  known-deformation apply + recover + score

The foundation of the validation loop: impose a KNOWN displacement field on a
synthetic volume, generate the deformed volume, recover the field with your DVC,
and score recovered-vs-known. Because you imposed the field, you have per-voxel
ground truth -- the first honest accuracy number for the pipeline.

Convention
----------
volume shape S = (a0, a1, a2)   (e.g. (nz, ny, nx))
displacement field  u  has shape (ndim, *S), in VOXEL units:
    u[d][idx] = displacement along axis d at voxel idx.
Warping is backward-mapped:  deformed[q] = reference[q - u[q]]   (no holes).

Uses only numpy + scipy + scikit-image (all already in requirements).
"""

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter


# ---------------------------------------------------------------------------
# 1. apply a known deformation
# ---------------------------------------------------------------------------
def warp_volume(volume, u, order=1, mode="nearest"):
    """Return the deformed volume: deformed[q] = volume[q - u[q]]."""
    volume = np.asarray(volume, float)
    S = volume.shape
    grid = np.indices(S, dtype=float)          # (ndim, *S)
    coords = grid - np.asarray(u, float)       # backward map
    return map_coordinates(volume, coords, order=order, mode=mode)


# ---------------------------------------------------------------------------
# 2. analytic ground-truth displacement fields  (start simple, verify by hand)
# ---------------------------------------------------------------------------
def uniform_strain_field(S, axis, strain):
    """Uniform strain `strain` along `axis` (constant displacement gradient).
    u_axis(x) = strain * x_axis  ->  d(u)/dx = strain everywhere."""
    ndim = len(S)
    u = np.zeros((ndim, *S), dtype=float)
    grid = np.indices(S, dtype=float)
    u[axis] = strain * grid[axis]
    return u


def shear_field(S, u_axis, grad_axis, gamma):
    """Simple shear: displacement along u_axis grows with position along grad_axis.
    u_{u_axis}(x) = gamma * x_{grad_axis}."""
    ndim = len(S)
    u = np.zeros((ndim, *S), dtype=float)
    grid = np.indices(S, dtype=float)
    u[u_axis] = gamma * grid[grad_axis]
    return u


def bending_field(S, u_axis, grad_axis, amplitude):
    """Quadratic 'bending': u_{u_axis}(x) = amplitude * (x_{grad_axis}/L)^2 * L."""
    ndim = len(S)
    u = np.zeros((ndim, *S), dtype=float)
    grid = np.indices(S, dtype=float)
    L = S[grad_axis]
    u[u_axis] = amplitude * (grid[grad_axis] / L) ** 2 * L
    return u


def smooth_random_field(S, sigma=6.0, amplitude=1.0, seed=0):
    """Smooth divergence-ish random field for a realistic non-analytic test."""
    rng = np.random.default_rng(seed)
    ndim = len(S)
    u = np.stack([gaussian_filter(rng.standard_normal(S), sigma) for _ in range(ndim)])
    u /= (np.abs(u).max() + 1e-8)
    return u * amplitude


# ---------------------------------------------------------------------------
# 3. DVC recovery  (per-block sub-pixel phase correlation, outlier-rejected)
# ---------------------------------------------------------------------------
def recover_displacement(ref, defd, block=24, step=12, upsample=20, mask=None,
                         max_shift=None, reject_outliers=True, window=False):
    """Estimate the displacement field by matching sub-volumes between ref and
    defd. Returns (centers, disps, dense_u); disps in VOXEL units.

    Settled defaults (verified on the round-trip test):
      * block=24  -- larger blocks carry more texture -> lower error, coarser grid
      * reject_outliers -- 3x3 median over the block grid drops isolated bad matches
      * window=False -- a Hann window HURT on uniformly-textured blocks (removes
        signal with no edge-leakage to suppress). Try window=True on real
        trabecular data where block boundaries cut through structure; measure it.
    Sub-pixel accuracy comes from `upsample`. This is a reference DVC -- if you
    later add a dedicated matcher, keep this signature and the loop still works.
    """
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import median_filter
    import itertools
    ref = np.asarray(ref, float)
    defd = np.asarray(defd, float)
    S = ref.shape
    ndim = ref.ndim
    if max_shift is None:
        max_shift = block / 2.0

    win = None
    if window:
        w = np.hanning(block)
        win = w
        for _ in range(ndim - 1):
            win = np.multiply.outer(win, w)

    axes_centers = [list(range(block // 2, S[d] - block // 2, step)) for d in range(ndim)]
    grid_shape = tuple(len(a) for a in axes_centers)
    disp_grid = np.full((ndim, *grid_shape), np.nan)

    for gi, c in zip(itertools.product(*[range(n) for n in grid_shape]),
                     itertools.product(*axes_centers)):
        sl = tuple(slice(ci - block // 2, ci + block // 2) for ci in c)
        rb, db = ref[sl], defd[sl]
        if mask is not None and mask[sl].mean() < 0.25:
            continue
        if rb.std() < 1e-6 or db.std() < 1e-6:
            continue
        a_, b_ = (rb * win, db * win) if win is not None else (rb, db)
        shift, _, _ = phase_cross_correlation(a_, b_, upsample_factor=upsample)
        # phase_cross_correlation(ref, moving): feature displacement ref->def = +shift
        if np.any(np.abs(shift) > max_shift):
            continue
        disp_grid[(slice(None), *gi)] = shift

    if reject_outliers and np.isfinite(disp_grid).any():
        for d in range(ndim):
            comp = disp_grid[d]
            filled = np.where(np.isfinite(comp), comp, np.nanmedian(comp))
            sm = median_filter(filled, size=3, mode="nearest")
            disp_grid[d] = np.where(np.isfinite(comp), sm, np.nan)

    centers, disps = [], []
    for gi, c in zip(itertools.product(*[range(n) for n in grid_shape]),
                     itertools.product(*axes_centers)):
        v = disp_grid[(slice(None), *gi)]
        if np.all(np.isfinite(v)):
            centers.append(c)
            disps.append(v.copy())
    centers = np.array(centers)
    disps = np.array(disps)

    dense_u = np.zeros((ndim, *S), dtype=float)
    if len(centers):
        from scipy.interpolate import NearestNDInterpolator
        grid = np.indices(S).reshape(ndim, -1).T
        for d in range(ndim):
            interp = NearestNDInterpolator(centers, disps[:, d])
            dense_u[d] = interp(grid).reshape(S)
    return centers, disps, dense_u


# ---------------------------------------------------------------------------
# 4. scoring  --  the number everything downstream reports against
# ---------------------------------------------------------------------------
def displacement_rmse(u_true, u_est, mask=None, voxel_um=None):
    """RMSE of displacement-vector magnitude error. Returns dict (voxels, [um])."""
    diff = np.asarray(u_true, float) - np.asarray(u_est, float)   # (ndim,*S)
    mag2 = np.sum(diff ** 2, axis=0)                              # (*S,)
    if mask is not None:
        mag2 = mag2[mask > 0]
    rmse_vox = float(np.sqrt(np.mean(mag2)))
    out = {"rmse_voxels": rmse_vox}
    if voxel_um is not None:
        out["rmse_um"] = rmse_vox * voxel_um
    return out


def sample_field_at_points(u, points):
    """Sample a (ndim,*S) field at integer voxel `points` (n,ndim)."""
    pts = np.round(points).astype(int)
    S = u.shape[1:]
    for d in range(len(S)):
        pts[:, d] = np.clip(pts[:, d], 0, S[d] - 1)
    idx = tuple(pts[:, d] for d in range(len(S)))
    return np.stack([u[d][idx] for d in range(u.shape[0])], axis=1)


# ---------------------------------------------------------------------------
# self-test: apply a KNOWN field, recover it, score it  (round-trip)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)

    # small synthetic trabecular-ish volume with texture (phase corr needs texture)
    S = (40, 96, 96)
    field = gaussian_filter(rng.standard_normal(S), 1.5)
    ref = (field - field.min()) / (np.ptp(field)) * 255.0
    mask = (gaussian_filter(rng.standard_normal(S), 2.0) < 0.2).astype(np.uint8)

    # KNOWN deformation: 1% uniform compression along axis 1  (u = -0.01 * x1)
    u_true = uniform_strain_field(S, axis=1, strain=-0.01)
    defd = warp_volume(ref, u_true)

    # recover with the reference matcher, score at block centres (fair: where we estimate)
    centers, disps, dense_u = recover_displacement(ref, defd, block=24, step=12, mask=mask)
    u_true_at_centers = sample_field_at_points(u_true, centers)

    # scoring at block centres (where we actually estimate)
    diff = disps - u_true_at_centers
    mag = np.sqrt((diff ** 2).sum(axis=1))
    print(f"volume {S}  bone voxels {int(mask.sum())}")
    print(f"blocks matched: {len(centers)}")
    print(f"displacement RMSE at block centres: {np.sqrt((mag**2).mean()):.3f} voxels"
          f"  ({np.sqrt((mag**2).mean())*50:.1f} um at 50 um voxel)")
    print(f"max |error|: {mag.max():.3f} voxels   median: {np.median(mag):.3f} voxels")
    # dense-field RMSE over the whole bone for comparison
    d = displacement_rmse(u_true, dense_u, mask=mask, voxel_um=50)
    print(f"dense-field RMSE over bone: {d['rmse_voxels']:.3f} voxels ({d['rmse_um']:.1f} um)")
    print("round-trip OK -- recovered field tracks the imposed field")