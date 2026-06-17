"""
make_demo_strain.py  --  run ONCE locally, then commit the output .npy

    python scripts/make_demo_strain.py

Builds a smooth, structured, representative strain field shaped to the actual
bone geometry (strain concentrates where the trabeculae are densest, along a
loading axis) and saves it as  data/demo/demo_strain.npy  -- same shape as the
bone mask. Nothing is computed in the app at demo time; the viewer just loads
this file. Uses only numpy + scipy (already in requirements).

Files are matched by pattern, so the  _S9_INT_UL_AP_50  suffix (or any rename)
is found automatically.

This is a REPRESENTATIVE field for visualising how the pipeline shows strain --
not the measured DVC result. The measured field is the D2IM displacement.
"""

import os
import glob
import numpy as np
from scipy.ndimage import gaussian_filter

DEMO_DIR = os.path.join("data", "demo")

# ----------------------------------------------------------------------------
# 1. Load the bone mask (and grayscale reference if present) -- by pattern
# ----------------------------------------------------------------------------
def load_match(patterns, what="file"):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(DEMO_DIR, pat)))
        if hits:
            print(f"loaded {what}:", hits[0])
            return np.load(hits[0])
    raise FileNotFoundError(f"no {what} matching {patterns} in {DEMO_DIR}")

mask = load_match(["*bone_mask*.npy", "*mask*.npy"], what="mask")
mask = (mask > 0).astype(np.float32)
X, Y, Z = mask.shape
print(f"mask shape {mask.shape}  bone voxels {int(mask.sum())}")

ref = None
for pat in ["*reference_scan*.npy", "*real_volume*.npy", "*grayscale*.npy"]:
    hits = sorted(glob.glob(os.path.join(DEMO_DIR, pat)))
    if hits:
        ref = np.load(hits[0]).astype(np.float32)
        print("using grayscale for density weighting:", hits[0])
        break

# ----------------------------------------------------------------------------
# 2. Local bone density  ->  so strain follows the real microstructure
# ----------------------------------------------------------------------------
density = gaussian_filter(mask, sigma=max(X, Y, Z) * 0.03)
if ref is not None and ref.shape == mask.shape and ref.max() > ref.min():
    g = (ref - ref.min()) / (ref.max() - ref.min())
    density = 0.5 * density + 0.5 * gaussian_filter(g, sigma=2.0)
density /= (density.max() + 1e-8)

# ----------------------------------------------------------------------------
# 3. Build the field  -- gradient + hot zone in the BROAD plane
#    (auto-detects the two largest axes; the thin axis is just slab depth)
# ----------------------------------------------------------------------------
axes_by_size = np.argsort(mask.shape)        # smallest -> largest
thin = axes_by_size[0]                        # the 5-deep slab axis
a1, a2 = sorted(axes_by_size[1:])             # the two broad in-plane axes

# normalised coords along the two broad axes
n1 = np.linspace(0, 1, mask.shape[a1])
n2 = np.linspace(0, 1, mask.shape[a2])
grids = np.meshgrid(np.linspace(0,1,mask.shape[0]),
                    np.linspace(0,1,mask.shape[1]),
                    np.linspace(0,1,mask.shape[2]), indexing="ij")
u = grids[a1]        # gradient axis  (broad)
w = grids[a2]        # second broad axis

# hot-zone centre on the densest spot in the broad plane
dens_plane = density.mean(axis=thin)                       # 2D over broad plane
ci, cj = np.unravel_index(int(np.argmax(dens_plane)), dens_plane.shape)
cu, cw = ci / dens_plane.shape[0], cj / dens_plane.shape[1]
print(f"hot-zone centre (broad plane) = ({cu:.2f}, {cw:.2f})")

field = (
    0.8 * u                                                # gradient across the face
    + 1.2 * density                                        # concentrate in dense bone
    + 0.9 * np.exp(-(((u - cu) ** 2 + (w - cw) ** 2) / 0.02))  # hot zone on the face
    + 0.10 * np.sin(5 * np.pi * w)                         # banding for texture
)

field = gaussian_filter(field, sigma=1.0)
field = field * mask
# ----------------------------------------------------------------------------
# 4. Rescale in-bone values to a tidy strain-like range (cosmetic), save
# ----------------------------------------------------------------------------
v = field[mask > 0]
PEAK_STRAIN = 0.012                                    # ~1.2 % peak, tweak to taste
field = np.where(
    mask > 0,
    (field - v.min()) / (v.max() - v.min() + 1e-8) * PEAK_STRAIN,
    0.0,
).astype(np.float32)

out = os.path.join(DEMO_DIR, "demo_strain.npy")
np.save(out, field)
vv = field[mask > 0]
print(f"saved {out}")
print(f"  shape={field.shape}  range=[{vv.min():.4f}, {vv.max():.4f}]  "
      f"mean={vv.mean():.4f}  p2={np.percentile(vv,2):.4f}  p98={np.percentile(vv,98):.4f}")