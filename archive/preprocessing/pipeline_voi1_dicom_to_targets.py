#!/usr/bin/env python3
"""
pipeline_voi1_dicom_to_targets.py

Batch pipeline for VOI1 micro-CT DICOM series:
1) DICOM folder -> 3D grayscale TIFF (uint8)
2) Segment bone mask (Otsu or Sauvola) -> mask TIFF (0/255)
3) Extract priors (BVTV, Tb.Th/Tb.Sp percentiles, Euler) -> JSON

Expected input structure:
data/real/VOI1/
  specimen01/
    *.dcm (or nested folders containing .dcm)
  specimen02/
  ...

Outputs:
data/derived/VOI1/
  specimen01_gray.tif
  specimen01_mask.tif
  specimen01_targets.json
  ...

Dependencies:
pip install numpy scipy tifffile pillow scikit-image pydicom
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import tifffile as tiff
import pydicom
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.measure import euler_number


# -------------------------
# DICOM loading
# -------------------------
def find_dicom_files(folder: Path) -> List[Path]:
    # Search recursively for .dcm (some datasets nest slices in subfolders)
    files = list(folder.rglob("*.dcm"))
    if files:
        return files
    # Some DICOMs have no .dcm extension, try reading all files (best-effort)
    # (Only if no .dcm found)
    candidates = [p for p in folder.rglob("*") if p.is_file()]
    return candidates


def dicom_sort_key(ds: pydicom.dataset.FileDataset) -> float:
    # Best: ImagePositionPatient[2], fallback to InstanceNumber
    try:
        ipp = ds.ImagePositionPatient
        return float(ipp[2])
    except Exception:
        pass
    try:
        return float(ds.InstanceNumber)
    except Exception:
        return 0.0


def load_dicom_series(folder: Path) -> Tuple[np.ndarray, Tuple[float, float, float], dict]:
    """
    Returns:
      volume: float32 (Z,Y,X)
      spacing_mm: (z, y, x) in mm
      meta: dict
    """
    files = find_dicom_files(folder)
    if len(files) == 0:
        raise RuntimeError(f"No files found in {folder}")

    slices = []
    sort_vals = []
    ds0 = None

    # Read all slices
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=False, force=True)
            if not hasattr(ds, "pixel_array"):
                continue
            arr = ds.pixel_array
            if arr is None:
                continue
            if ds0 is None:
                ds0 = ds
            slices.append(arr.astype(np.float32))
            sort_vals.append(dicom_sort_key(ds))
        except Exception:
            continue

    if len(slices) < 2:
        raise RuntimeError(f"Could not read enough DICOM slices from {folder} (read {len(slices)})")

    order = np.argsort(np.array(sort_vals))
    vol = np.stack([slices[i] for i in order], axis=0).astype(np.float32)

    # Spacing
    # PixelSpacing is [row_spacing, col_spacing] == (y, x) in mm
    if ds0 is not None and hasattr(ds0, "PixelSpacing"):
        y_mm = float(ds0.PixelSpacing[0])
        x_mm = float(ds0.PixelSpacing[1])
    else:
        y_mm = x_mm = 0.039  # fallback guess

    # Z spacing: try SliceThickness or infer from ImagePositionPatient
    z_mm = None
    if ds0 is not None and hasattr(ds0, "SliceThickness"):
        try:
            z_mm = float(ds0.SliceThickness)
        except Exception:
            z_mm = None

    # Infer z spacing from positions if available
    if z_mm is None:
        # Re-read a few datasets to get z positions
        zpos = []
        for f in files[: min(50, len(files))]:
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                ipp = ds.ImagePositionPatient
                zpos.append(float(ipp[2]))
            except Exception:
                pass
        zpos = sorted(set(zpos))
        if len(zpos) >= 2:
            diffs = np.diff(zpos)
            z_mm = float(np.median(np.abs(diffs)))
        else:
            z_mm = 0.039  # fallback guess

    meta = {
        "source_folder": str(folder),
        "shape_zyx": [int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])],
        "spacing_mm_zyx": [float(z_mm), float(y_mm), float(x_mm)],
    }
    return vol, (float(z_mm), float(y_mm), float(x_mm)), meta


# -------------------------
# Preprocess + write grayscale TIFF
# -------------------------
def to_uint8_percentile(vol_f32: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(vol_f32, [p_lo, p_hi])
    if hi <= lo + 1e-6:
        return np.zeros_like(vol_f32, dtype=np.uint8)
    x = (vol_f32 - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


# -------------------------
# Segmentation
# -------------------------
def segment_mask(gray_u8: np.ndarray,
                 method: str = "otsu",
                 blur_sigma: float = 0.8,
                 close_iters: int = 2,
                 open_iters: int = 1,
                 invert: bool = False,
                 sauvola_window: int = 51,
                 sauvola_k: float = 0.15) -> np.ndarray:
    g = gray_u8.astype(np.float32)
    if blur_sigma > 0:
        g = ndi.gaussian_filter(g, sigma=float(blur_sigma))

    method = method.lower().strip()
    if method == "otsu":
        thr = float(threshold_otsu(g))
        m = g >= thr
    elif method == "sauvola":
        win = int(sauvola_window)
        win = max(15, win | 1)
        thr_map = threshold_sauvola(g, window_size=win, k=float(sauvola_k))
        m = g >= thr_map
    else:
        raise ValueError("method must be otsu or sauvola")

    if invert:
        m = ~m

    st = ndi.generate_binary_structure(3, 1)
    if int(close_iters) > 0:
        m = ndi.binary_closing(m, structure=st, iterations=int(close_iters))
    if int(open_iters) > 0:
        m = ndi.binary_opening(m, structure=st, iterations=int(open_iters))

    return m.astype(np.uint8)


# -------------------------
# Priors / metrics
# -------------------------
def compute_targets(mask01: np.ndarray, voxel_um: float) -> dict:
    m = mask01.astype(bool)

    bvtv = float(m.mean())
    eul = float(euler_number(m, connectivity=3))
    conn_proxy = float(1.0 - eul)

    # EDT thickness proxies
    dt_b = ndi.distance_transform_edt(m) * float(voxel_um)
    dt_m = ndi.distance_transform_edt(~m) * float(voxel_um)

    tbth = dt_b[m]
    tbsp = dt_m[~m]

    def pct(x: np.ndarray, p: float) -> float:
        if x.size == 0:
            return 0.0
        return float(np.percentile(x, p))

    return {
        "BVTV": bvtv,
        "Euler": eul,
        "ConnProxy": conn_proxy,
        "TbTh_um_p10": pct(tbth, 10),
        "TbTh_um_p50": pct(tbth, 50),
        "TbTh_um_p90": pct(tbth, 90),
        "TbTh_um_p95": pct(tbth, 95),
        "TbSp_um_p10": pct(tbsp, 10),
        "TbSp_um_p50": pct(tbsp, 50),
        "TbSp_um_p90": pct(tbsp, 90),
        "TbSp_um_p95": pct(tbsp, 95),
        "voxel_um": float(voxel_um),
        "mask_shape_zyx": [int(mask01.shape[0]), int(mask01.shape[1]), int(mask01.shape[2])],
    }


# -------------------------
# Main batch driver
# -------------------------
def process_specimen(specimen_dir: Path, out_dir: Path, args: argparse.Namespace) -> None:
    name = specimen_dir.name
    out_gray = out_dir / f"{name}_gray.tif"
    out_mask = out_dir / f"{name}_mask.tif"
    out_targets = out_dir / f"{name}_targets.json"
    out_meta = out_dir / f"{name}_meta.json"

    print(f"\n=== {name} ===")
    print(f"Input: {specimen_dir}")

    vol_f32, spacing_mm, meta = load_dicom_series(specimen_dir)

    # Dataset says 39 um. We'll store spacing but use voxel_um for metrics.
    meta["spacing_mm_zyx"] = list(spacing_mm)

    gray_u8 = to_uint8_percentile(vol_f32, p_lo=args.p_lo, p_hi=args.p_hi)
    tiff.imwrite(out_gray, gray_u8.astype(np.uint8))
    print(f"Saved gray: {out_gray}")

    mask01 = segment_mask(
        gray_u8,
        method=args.seg_method,
        blur_sigma=args.seg_blur,
        close_iters=args.seg_close,
        open_iters=args.seg_open,
        invert=bool(args.seg_invert),
        sauvola_window=args.sauvola_window,
        sauvola_k=args.sauvola_k,
    )
    tiff.imwrite(out_mask, (mask01 * 255).astype(np.uint8))
    print(f"Saved mask: {out_mask}")

    targets = compute_targets(mask01, voxel_um=args.voxel_um)
    with open(out_targets, "w") as f:
        json.dump(targets, f, indent=2)
    print(f"Saved targets: {out_targets}")

    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta: {out_meta}")

    print(f"BVTV={targets['BVTV']:.3f} | Euler={targets['Euler']:.1f} | TbTh(p90)={targets['TbTh_um_p90']:.1f}um")


def main():
    p = argparse.ArgumentParser(description="VOI1 batch pipeline: DICOM -> gray.tif -> mask.tif -> targets.json")

    p.add_argument("--in-root", type=str, default=r"data\real\VOI1",
                   help="Folder containing specimen01/, specimen02/, ... DICOM series.")
    p.add_argument("--out-root", type=str, default=r"data\derived\VOI1",
                   help="Output folder for converted TIFFs + targets.")

    p.add_argument("--voxel-um", type=float, default=39.0, help="Voxel size in micrometers (dataset: 39).")

    # grayscale normalize
    p.add_argument("--p-lo", type=float, default=1.0, help="Percentile low clip for uint8 scaling.")
    p.add_argument("--p-hi", type=float, default=99.0, help="Percentile high clip for uint8 scaling.")

    # segmentation
    p.add_argument("--seg-method", type=str, default="otsu", choices=["otsu", "sauvola"])
    p.add_argument("--seg-blur", type=float, default=0.8)
    p.add_argument("--seg-close", type=int, default=2)
    p.add_argument("--seg-open", type=int, default=1)
    p.add_argument("--seg-invert", type=int, default=0)

    p.add_argument("--sauvola-window", type=int, default=51)
    p.add_argument("--sauvola-k", type=float, default=0.15)

    # filter which specimens to process
    p.add_argument("--pattern", type=str, default="specimen*",
                   help="Glob pattern under in-root to identify specimen folders.")

    args = p.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    specimens = sorted([d for d in in_root.glob(args.pattern) if d.is_dir()])
    if not specimens:
        raise SystemExit(f"No specimen folders found under {in_root} matching {args.pattern}")

    print(f"Found {len(specimens)} specimen folders under {in_root}")
    print(f"Outputs -> {out_root}")

    for d in specimens:
        process_specimen(d, out_root, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
