"""
prepare_d2im_demo_data.py
=========================
Run this script ONCE locally before the demo.
It reads the D2IM Figshare data from your repo, and writes
three ready-to-use .npy files into data/strain/processed/:

  reference_scan.npy      — uint8 grayscale volume  (Z, Y, X)
  bone_mask.npy           — uint8 binary mask        (Z, Y, X)
  displacement_magnitude.npy — float32 |U,V,W| field (Z, Y, X)

Usage
-----
  cd C:/Users/Isabella/OneDrive - zeki/Documents/Research/synthetic_trabeculae
  python scripts/prepare_d2im_demo_data.py

  # or specify a specimen explicitly:
  python scripts/prepare_d2im_demo_data.py --specimen S9_INT_UL_AP_50
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image


# ── Config ────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "strain" / "D2IM_Data"

SCAN_DIR   = DATA_ROOT / "Input" / "Scan"
MASK_DIR   = DATA_ROOT / "Input" / "Mask"
U_DIR      = DATA_ROOT / "Target" / "U"
V_DIR      = DATA_ROOT / "Target" / "V"
W_DIR      = DATA_ROOT / "Target" / "W"

OUT_DIR    = REPO_ROOT / "data" / "strain" / "processed"


# ── Helpers ───────────────────────────────────────────────────

def load_tiff_stack(folder: Path, specimen: str = None) -> np.ndarray:
    """
    Load all TIFFs in a folder as a (Z, Y, X) volume.
    If specimen is given, only files whose name contains that string are loaded.
    Handles both single-frame and multi-frame TIFFs.
    """
    pattern = f"*{specimen}*" if specimen else "*.tif*"
    files = sorted(folder.glob(pattern))
    if not files:
        # Try one level deeper — some D2IM folders have per-specimen subfolders
        subdirs = [d for d in folder.iterdir() if d.is_dir()]
        if subdirs:
            # Pick the first matching subfolder
            for sd in subdirs:
                if specimen and specimen.lower() not in sd.name.lower():
                    continue
                files = sorted(sd.glob("*.tif*"))
                if files:
                    print(f"  Found {len(files)} files in subfolder: {sd.name}")
                    break

    if not files:
        raise FileNotFoundError(
            f"No TIFF files found in {folder} "
            f"(specimen filter: {specimen or 'none'})\n"
            f"Contents: {list(folder.iterdir())[:10]}"
        )

    slices = []
    for f in files:
        img = Image.open(f)
        n_frames = getattr(img, "n_frames", 1)
        if n_frames > 1:
            for i in range(n_frames):
                img.seek(i)
                slices.append(np.array(img))
        else:
            slices.append(np.array(img))

    vol = np.stack(slices, axis=0)
    print(f"  Loaded {len(files)} file(s) → shape {vol.shape}, dtype {vol.dtype}")
    return vol


def normalise_uint8(vol: np.ndarray) -> np.ndarray:
    """Percentile-clip and rescale any dtype to uint8."""
    if vol.dtype == np.uint8:
        return vol
    vmin, vmax = np.percentile(vol, 0.5), np.percentile(vol, 99.5)
    if vmax > vmin:
        clipped = np.clip(vol.astype(np.float32), vmin, vmax)
        return ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    return np.zeros_like(vol, dtype=np.uint8)


def load_displacement_components(u_dir, v_dir, w_dir,
                                  specimen=None) -> tuple:
    """Load U, V, W displacement component stacks."""
    print("Loading U...")
    U = load_tiff_stack(u_dir, specimen).astype(np.float32)
    print("Loading V...")
    V = load_tiff_stack(v_dir, specimen).astype(np.float32)
    print("Loading W...")
    W = load_tiff_stack(w_dir, specimen).astype(np.float32)
    return U, V, W


def compute_displacement_magnitude(U, V, W) -> np.ndarray:
    """Compute voxel-wise displacement magnitude |d| = sqrt(U²+V²+W²)."""
    return np.sqrt(U**2 + V**2 + W**2).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare D2IM demo data")
    parser.add_argument(
        "--specimen", type=str, default=None,
        help="Specimen name filter, e.g. S9_INT_UL_AP_50. "
             "Leave blank to use all files in each folder.",
    )
    parser.add_argument(
        "--skip-scan", action="store_true",
        help="Skip loading the scan (use if Input/Scan is missing).",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specimen = args.specimen
    tag = f"_{specimen}" if specimen else ""

    print(f"\nD2IM data root : {DATA_ROOT}")
    print(f"Output dir     : {OUT_DIR}")
    print(f"Specimen filter: {specimen or 'all'}\n")

    # ── 1. Reference scan ──────────────────────────────────────
    if not args.skip_scan:
        print("── Loading reference scan (Input/Scan) ──")
        scan = normalise_uint8(load_tiff_stack(SCAN_DIR, specimen))
        out_scan = OUT_DIR / f"reference_scan{tag}.npy"
        np.save(out_scan, scan)
        print(f"  Saved → {out_scan}\n")
    else:
        print("  Skipping scan.\n")

    # ── 2. Bone mask ───────────────────────────────────────────
    print("── Loading bone mask (Input/Mask) ──")
    mask_raw = load_tiff_stack(MASK_DIR, specimen)
    # Binarise: D2IM masks may be 0/255 or 0/1
    if mask_raw.max() > 1:
        mask = (mask_raw > 127).astype(np.uint8)
    else:
        mask = mask_raw.astype(np.uint8)
    out_mask = OUT_DIR / f"bone_mask{tag}.npy"
    np.save(out_mask, mask)
    bvtv = mask.mean()
    print(f"  BV/TV = {bvtv:.3f}")
    print(f"  Saved → {out_mask}\n")

    # ── 3. Displacement magnitude ─────────────────────────────
    print("── Loading displacement components (Target/U, V, W) ──")
    U, V, W = load_displacement_components(U_DIR, V_DIR, W_DIR, specimen)

    # Sanity check shapes match
    shapes = {U.shape, V.shape, W.shape}
    if len(shapes) > 1:
        print(f"  WARNING: U/V/W shapes differ: {U.shape}, {V.shape}, {W.shape}")
        print("  Cropping to smallest common shape...")
        min_shape = tuple(min(s[i] for s in [U.shape, V.shape, W.shape]) for i in range(3))
        U = U[:min_shape[0], :min_shape[1], :min_shape[2]]
        V = V[:min_shape[0], :min_shape[1], :min_shape[2]]
        W = W[:min_shape[0], :min_shape[1], :min_shape[2]]

    mag = compute_displacement_magnitude(U, V, W)
    out_mag = OUT_DIR / f"displacement_magnitude{tag}.npy"
    np.save(out_mag, mag)
    print(f"  Magnitude range: [{mag.min():.4f}, {mag.max():.4f}]")
    print(f"  Saved → {out_mag}\n")

    # ── 4. Also save individual components ───────────────────
    for arr, name in [(U, "disp_U"), (V, "disp_V"), (W, "disp_W")]:
        out = OUT_DIR / f"{name}{tag}.npy"
        np.save(out, arr)
        print(f"  Saved component → {out}")

    # ── Summary ───────────────────────────────────────────────
    print("\n── Ready for demo ──────────────────────────────────")
    print("Load these files into the dashboard:")
    print(f"  Reference scan   : data/strain/processed/reference_scan{tag}.npy")
    print(f"  Bone mask        : data/strain/processed/bone_mask{tag}.npy")
    print(f"  Displacement mag : data/strain/processed/displacement_magnitude{tag}.npy")
    print("\nIn Data Loader:")
    print("  1. File format      → NumPy (.npy)")
    print("  2. Upload           → reference_scan.npy")
    print("  3. Strain input     → Image + strain field")
    print("  4. Strain field     → displacement_magnitude.npy")
    print("  5. Run registration → rigid-body")
    print("\nIn 3D Viewer:")
    print("  Strain overlay will appear once registration is complete.")


if __name__ == "__main__":
    main()