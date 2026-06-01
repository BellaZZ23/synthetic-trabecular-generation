"""
prepare_d2im_demo_data.py
=========================
Run this script ONCE locally before the demo.

Fixes applied vs v1:
  - Scan and Mask are now matched by slice index suffix (_NNNN) so only
    the overlapping slices are kept — no more shape mismatch.
  - Target/U/V/W TIFFs are tiny DVC grid files (not full-res volumes).
    They are read correctly as float32 2-D grids per load step, then the
    LAST load step (maximum deformation) is upsampled to match the scan
    spatial dimensions to produce a usable displacement magnitude map.

Outputs written to  data/strain/processed/ :
  reference_scan_<spec>.npy          uint8  (Z, Y, X)
  bone_mask_<spec>.npy               uint8  (Z, Y, X)
  displacement_magnitude_<spec>.npy  float32 (Z, Y, X)  — upsampled DVC grid
  disp_U/V/W_<spec>.npy              float32 raw DVC grids, last load step

Usage
-----
  cd "C:/Users/Isabella/OneDrive - zeki/Documents/Research/synthetic_trabeculae"
  python scripts/prepare_d2im_demo_data.py --specimen S9_INT_UL_AP_50
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import zoom


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "strain" / "D2IM_Data"
SCAN_DIR  = DATA_ROOT / "Input"  / "Scan"
MASK_DIR  = DATA_ROOT / "Input"  / "Mask"
U_DIR     = DATA_ROOT / "Target" / "U"
V_DIR     = DATA_ROOT / "Target" / "V"
W_DIR     = DATA_ROOT / "Target" / "W"
OUT_DIR   = REPO_ROOT / "data"   / "strain" / "processed"


# ── helpers ───────────────────────────────────────────────────

def get_specimen_files(folder: Path, specimen: str):
    """Return sorted files whose name starts with specimen."""
    files = sorted(
        [f for f in folder.glob("*.tif*") if f.name.startswith(specimen)],
        key=lambda f: int(f.stem.split("_")[-1]),
    )
    if not files:
        raise FileNotFoundError(
            f"No files for '{specimen}' in {folder}\n"
            f"Available: {[f.name for f in folder.glob('*.tif*')][:8]}"
        )
    return files


def get_slice_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def load_matched_stack(scan_files, mask_files):
    """
    Keep only slices whose index exists in BOTH scan and mask.
    Returns (scan_vol uint8, mask_vol uint8).
    """
    scan_idx  = {get_slice_index(f): f for f in scan_files}
    mask_idx  = {get_slice_index(f): f for f in mask_files}
    common    = sorted(set(scan_idx) & set(mask_idx))

    if not common:
        raise ValueError(
            f"No overlapping slice indices between scan {sorted(scan_idx)} "
            f"and mask {sorted(mask_idx)}"
        )

    print(f"  Scan slices : {sorted(scan_idx.keys())}")
    print(f"  Mask slices : {sorted(mask_idx.keys())}")
    print(f"  Common      : {common}  ({len(common)} slices)")

    scan_vol = np.stack([np.array(Image.open(scan_idx[i])) for i in common], axis=0)
    mask_vol = np.stack([np.array(Image.open(mask_idx[i])) for i in common], axis=0)
    return scan_vol, mask_vol


def normalise_uint8(vol: np.ndarray) -> np.ndarray:
    if vol.dtype == np.uint8:
        return vol
    vmin, vmax = np.percentile(vol, 0.5), np.percentile(vol, 99.5)
    if vmax > vmin:
        return ((np.clip(vol.astype(np.float32), vmin, vmax) - vmin)
                / (vmax - vmin) * 255).astype(np.uint8)
    return np.zeros_like(vol, dtype=np.uint8)


def load_dvc_grid(files):
    """
    Load DVC displacement grid TIFFs.
    Each file = one load step = one 2-D float32 grid (rows × cols).
    Returns array of shape (n_steps, rows, cols).
    """
    grids = []
    for f in files:
        arr = np.array(Image.open(f), dtype=np.float32)
        grids.append(arr)
    return np.stack(grids, axis=0)          # (steps, rows, cols)


def upsample_to_volume(grid_2d: np.ndarray,
                        target_shape: tuple) -> np.ndarray:
    """
    Upsample a 2-D DVC grid (rows, cols) to a 3-D volume (Z, Y, X)
    by tiling along Z then zooming spatially.
    The DVC grid represents a single mid-plane summary of the field.
    """
    nz, ny, nx = target_shape
    gr, gc = grid_2d.shape

    # Zoom the 2-D grid to match (ny, nx)
    zy = ny / gr
    zx = nx / gc
    grid_full = zoom(grid_2d, (zy, zx), order=1)      # (ny, nx)

    # Tile to 3-D: same field replicated across all z-slices
    vol = np.broadcast_to(grid_full[np.newaxis], (nz, ny, nx)).copy()
    return vol.astype(np.float32)


# ── main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specimen", default="S9_INT_UL_AP_50",
                        help="Specimen prefix, e.g. S9_INT_UL_AP_50")
    args = parser.parse_args()
    spec = args.specimen
    tag  = f"_{spec}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nD2IM root  : {DATA_ROOT}")
    print(f"Output dir : {OUT_DIR}")
    print(f"Specimen   : {spec}\n")

    # ── 1. Scan + mask (matched by slice index) ───────────────
    print("── Matching scan and mask slices ──")
    scan_files = get_specimen_files(SCAN_DIR, spec)
    mask_files = get_specimen_files(MASK_DIR, spec)
    scan_vol, mask_raw = load_matched_stack(scan_files, mask_files)

    scan_vol = normalise_uint8(scan_vol)
    mask_vol = (mask_raw > 127).astype(np.uint8) if mask_raw.max() > 1 else mask_raw.astype(np.uint8)

    nz, ny, nx = scan_vol.shape
    print(f"  Final volume shape : {scan_vol.shape}  BV/TV={mask_vol.mean():.3f}")

    np.save(OUT_DIR / f"reference_scan{tag}.npy",  scan_vol)
    np.save(OUT_DIR / f"bone_mask{tag}.npy",        mask_vol)
    print(f"  Saved scan  → reference_scan{tag}.npy")
    print(f"  Saved mask  → bone_mask{tag}.npy\n")

    # ── 2. DVC displacement grids ─────────────────────────────
    print("── Loading DVC displacement grids (Target/U, V, W) ──")
    u_files = get_specimen_files(U_DIR, spec)
    v_files = get_specimen_files(V_DIR, spec)
    w_files = get_specimen_files(W_DIR, spec)

    U_steps = load_dvc_grid(u_files)   # (n_steps, rows, cols)
    V_steps = load_dvc_grid(v_files)
    W_steps = load_dvc_grid(w_files)

    n_steps = U_steps.shape[0]
    print(f"  Load steps : {n_steps}")
    print(f"  Grid shape : {U_steps.shape[1:]}  (DVC spatial grid per step)")

    # Use the LAST load step = maximum deformation
    U_last = U_steps[-1]
    V_last = V_steps[-1]
    W_last = W_steps[-1]

    mag_last = np.sqrt(U_last**2 + V_last**2 + W_last**2)
    print(f"  Displacement magnitude (last step): "
          f"min={np.nanmin(mag_last):.4f}  max={np.nanmax(mag_last):.4f}")

    # Upsample to scan spatial dimensions
    print(f"  Upsampling DVC grid {U_last.shape} → volume {(nz, ny, nx)} ...")
    mag_vol = upsample_to_volume(mag_last, (nz, ny, nx))
    U_vol   = upsample_to_volume(U_last,   (nz, ny, nx))
    V_vol   = upsample_to_volume(V_last,   (nz, ny, nx))
    W_vol   = upsample_to_volume(W_last,   (nz, ny, nx))

    np.save(OUT_DIR / f"displacement_magnitude{tag}.npy", mag_vol)
    np.save(OUT_DIR / f"disp_U{tag}.npy", U_vol)
    np.save(OUT_DIR / f"disp_V{tag}.npy", V_vol)
    np.save(OUT_DIR / f"disp_W{tag}.npy", W_vol)
    print(f"  Saved → displacement_magnitude{tag}.npy  shape={mag_vol.shape}")
    print(f"  Saved individual U/V/W components\n")

    # ── Summary ───────────────────────────────────────────────
    print("── Ready for demo ──────────────────────────────────")
    print(f"  reference_scan          : data/strain/processed/reference_scan{tag}.npy")
    print(f"  bone_mask               : data/strain/processed/bone_mask{tag}.npy")
    print(f"  displacement_magnitude  : data/strain/processed/displacement_magnitude{tag}.npy")
    print()
    print("In Data Loader:")
    print("  1. File format   → NumPy (.npy)")
    print("  2. Upload        → reference_scan.npy")
    print("  3. Strain input  → Image + strain field")
    print("  4. Strain field  → displacement_magnitude.npy")
    print("  5. Run registration → rigid-body")
    print()
    print("Note: displacement_magnitude is upsampled from the DVC coarse grid.")
    print("It shows the spatial pattern of deformation, not voxel-level precision.")
    print("That is expected and sufficient for the demo.")


if __name__ == "__main__":
    main()