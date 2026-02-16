#!/usr/bin/env python3
r"""
pipeline_voi1_multiframe_dcm_to_targets.py

Handles VOI1 dataset where each scan is stored as a SINGLE DICOM file (often multi-frame).
For each specimen*/ *Scan*.dcm:
  1) read DICOM -> 3D volume
  2) normalize -> uint8 grayscale TIFF
  3) segment (Otsu or Sauvola) -> mask TIFF (0/255)
  4) extract priors -> targets.json (BV/TV, Tb.Th/Tb.Sp percentiles, Euler)

Run (PowerShell):
pip install pydicom numpy scipy tifffile scikit-image pillow
python .\pipeline_voi1_multiframe_dcm_to_targets.py `
  --in-root data\real\VOI1 `
  --out-root data\derived\VOI1 `
  --voxel-um 39 `
  --seg-method otsu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile as tiff
import pydicom
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.measure import euler_number


def dicom_to_volume(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Returns volume as float32 with shape (Z,Y,X).
    Supports multi-frame (NumberOfFrames) and classic 2D.
    """
    arr = ds.pixel_array  # pydicom handles multi-frame decoding
    arr = np.asarray(arr)

    # Common cases:
    # - multi-frame: (frames, rows, cols) => (Z,Y,X)
    # - single frame: (rows, cols) => treat as Z=1
    if arr.ndim == 3:
        vol = arr
    elif arr.ndim == 2:
        vol = arr[None, ...]
    else:
        raise RuntimeError(f"Unexpected pixel_array shape: {arr.shape}")

    return vol.astype(np.float32)


def get_spacing_um(ds: pydicom.dataset.FileDataset, fallback_um: float) -> Tuple[float, float, float]:
    """
    Returns voxel spacing (z_um, y_um, x_um).
    Uses PixelSpacing if present; tries SpacingBetweenSlices/SliceThickness for z.
    Falls back to fallback_um for any missing.
    """
    y_um = x_um = fallback_um
    z_um = fallback_um

    # PixelSpacing is (row_spacing, col_spacing) in mm
    if hasattr(ds, "PixelSpacing"):
        try:
            y_um = float(ds.PixelSpacing[0]) * 1000.0
            x_um = float(ds.PixelSpacing[1]) * 1000.0
        except Exception:
            pass

    # Z spacing often in SpacingBetweenSlices or SliceThickness (mm)
    for tag in ("SpacingBetweenSlices", "SliceThickness"):
        if hasattr(ds, tag):
            try:
                z_um = float(getattr(ds, tag)) * 1000.0
                break
            except Exception:
                pass

    return float(z_um), float(y_um), float(x_um)


def to_uint8_percentile(vol_f32: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(vol_f32, [p_lo, p_hi])
    if hi <= lo + 1e-6:
        return np.zeros_like(vol_f32, dtype=np.uint8)
    x = (vol_f32 - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def segment_mask(gray_u8: np.ndarray,
                 method: str = "otsu",
                 blur_sigma: float = 0.8,
                 close_iters: int = 2,
                 open_iters: int = 1,
                 invert: bool = False,
                 sauvola_window: int = 51,
                 sauvola_k: float = 0.15) -> Tuple[np.ndarray, dict]:
    g = gray_u8.astype(np.float32)
    if blur_sigma > 0:
        g = ndi.gaussian_filter(g, sigma=float(blur_sigma))

    method = method.lower().strip()
    info = {"seg_method": method, "blur_sigma": float(blur_sigma)}

    if method == "otsu":
        thr = float(threshold_otsu(g))
        m = g >= thr
        info["thr"] = thr
    elif method == "sauvola":
        win = int(sauvola_window)
        win = max(15, win | 1)
        thr_map = threshold_sauvola(g, window_size=win, k=float(sauvola_k))
        m = g >= thr_map
        info.update({"window": win, "k": float(sauvola_k)})
    else:
        raise ValueError("method must be otsu or sauvola")

    if invert:
        m = ~m
        info["invert"] = True
    else:
        info["invert"] = False

    st = ndi.generate_binary_structure(3, 1)
    if int(close_iters) > 0:
        m = ndi.binary_closing(m, structure=st, iterations=int(close_iters))
    if int(open_iters) > 0:
        m = ndi.binary_opening(m, structure=st, iterations=int(open_iters))

    info.update({"close_iters": int(close_iters), "open_iters": int(open_iters)})
    return m.astype(np.uint8), info


def compute_targets(mask01: np.ndarray, voxel_um: Tuple[float, float, float]) -> dict:
    m = mask01.astype(bool)
    bvtv = float(m.mean())
    eul = float(euler_number(m, connectivity=3))
    conn_proxy = float(1.0 - eul)

    z_um, y_um, x_um = voxel_um
    sampling = (float(z_um), float(y_um), float(x_um))

    dt_b = ndi.distance_transform_edt(m, sampling=sampling)
    dt_m = ndi.distance_transform_edt(~m, sampling=sampling)

    tbth = dt_b[m]
    tbsp = dt_m[~m]

    def pct(x: np.ndarray, p: float) -> float:
        return float(np.percentile(x, p)) if x.size else 0.0

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
        "voxel_um_zyx": [float(z_um), float(y_um), float(x_um)],
        "shape_zyx": [int(mask01.shape[0]), int(mask01.shape[1]), int(mask01.shape[2])],
    }


def main():
    ap = argparse.ArgumentParser(description="VOI1 multiframe DICOM pipeline -> gray/mask/targets.")
    ap.add_argument("--in-root", type=str, default=r"data\real\VOI1")
    ap.add_argument("--out-root", type=str, default=r"data\derived\VOI1")

    ap.add_argument("--voxel-um", type=float, default=39.0, help="Fallback voxel size in um (dataset: 39).")

    ap.add_argument("--p-lo", type=float, default=1.0)
    ap.add_argument("--p-hi", type=float, default=99.0)

    ap.add_argument("--seg-method", type=str, default="otsu", choices=["otsu", "sauvola"])
    ap.add_argument("--seg-blur", type=float, default=0.8)
    ap.add_argument("--seg-close", type=int, default=2)
    ap.add_argument("--seg-open", type=int, default=1)
    ap.add_argument("--seg-invert", type=int, default=0)
    ap.add_argument("--sauvola-window", type=int, default=51)
    ap.add_argument("--sauvola-k", type=float, default=0.15)

    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    dcm_files = sorted(in_root.rglob("*.dcm"))
    if not dcm_files:
        raise SystemExit(f"No .dcm files found under {in_root}")

    print(f"Found {len(dcm_files)} DICOM files under {in_root}")
    print(f"Writing outputs to {out_root}")

    for dcm_path in dcm_files:
        # Expect: .../specimenXX/SpecimenX_VOI1_ScanY.dcm
        specimen = dcm_path.parent.name
        stem = dcm_path.stem  # filename without extension
        print(f"\n=== {specimen} | {dcm_path.name} ===")

        ds = pydicom.dcmread(str(dcm_path), force=True)
        vol = dicom_to_volume(ds)

        voxel_um = get_spacing_um(ds, fallback_um=float(args.voxel_um))

        gray_u8 = to_uint8_percentile(vol, p_lo=args.p_lo, p_hi=args.p_hi)
        gray_out = out_root / f"{specimen}_{stem}_gray.tif"
        tiff.imwrite(gray_out, gray_u8.astype(np.uint8))
        print("Saved gray:", gray_out)

        mask01, seg_info = segment_mask(
            gray_u8,
            method=args.seg_method,
            blur_sigma=args.seg_blur,
            close_iters=args.seg_close,
            open_iters=args.seg_open,
            invert=bool(int(args.seg_invert)),
            sauvola_window=args.sauvola_window,
            sauvola_k=args.sauvola_k,
        )
        mask_out = out_root / f"{specimen}_{stem}_mask.tif"
        tiff.imwrite(mask_out, (mask01 * 255).astype(np.uint8))
        print("Saved mask:", mask_out)

        targets = compute_targets(mask01, voxel_um=voxel_um)
        targets.update({"source_dcm": str(dcm_path), "segmentation": seg_info})

        targets_out = out_root / f"{specimen}_{stem}_targets.json"
        with open(targets_out, "w") as f:
            json.dump(targets, f, indent=2)
        print("Saved targets:", targets_out)
        print(f"BVTV={targets['BVTV']:.3f} | Euler={targets['Euler']:.1f} | TbTh(p90)={targets['TbTh_um_p90']:.1f}um")

    print("\nDone.")


if __name__ == "__main__":
    main()
