#!/usr/bin/env python3
r"""
pipeline_voi1_multiframe_dcm_to_targets.py (UPDATED for v13.1 skeleton-based evidence)

Handles VOI1 dataset where each scan is stored as a SINGLE DICOM file (often multi-frame).
For each specimen*/ *Scan*.dcm:
  1) read DICOM -> 3D volume
  2) normalize -> uint8 grayscale TIFF
  3) segment (Otsu or Sauvola) -> mask TIFF (0/255)
  4) extract targets -> targets.json:
       - BV/TV
       - Tb.Th / Tb.Sp percentiles (distance transform)
       - Euler characteristic
       - OPTIONAL skeleton graph stats (endpoints/junctions ratio) for "bone-like mesh" evidence

New in this update:
  - Optional skeletonization of the segmented mask (or its thinned variant) using:
      --skeleton-mode skimage   (default, in-Python)
      --skeleton-mode fiji      (optional headless Fiji/BoneJ skeletonize)
  - Skeleton pruning (spur removal) and basic graph stats:
      endpoints, junctions, endpoint/junction ratio, skeleton voxels
  - Optional debug outputs:
      skeleton_raw.tif, skeleton_pruned.tif

Run (PowerShell):
pip install pydicom numpy scipy tifffile scikit-image pillow
python .\pipeline_voi1_multiframe_dcm_to_targets.py `
  --in-root data\real\VOI1 `
  --out-root data\derived\VOI1 `
  --voxel-um 39 `
  --seg-method otsu `
  --compute-skeleton-metrics 1 `
  --skeleton-mode skimage `
  --skeleton-prune-lmin 10 `
  --debug-skeleton 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import tifffile as tiff
import pydicom
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.measure import euler_number

# Skeleton import (Python fallback)
try:
    from skimage.morphology import skeletonize_3d  # type: ignore
except Exception:
    skeletonize_3d = None


# -----------------------------
# DICOM
# -----------------------------
def dicom_to_volume(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Returns volume as float32 with shape (Z,Y,X).
    Supports multi-frame (NumberOfFrames) and classic 2D.
    """
    arr = ds.pixel_array
    arr = np.asarray(arr)

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

    if hasattr(ds, "PixelSpacing"):
        try:
            y_um = float(ds.PixelSpacing[0]) * 1000.0
            x_um = float(ds.PixelSpacing[1]) * 1000.0
        except Exception:
            pass

    for tag in ("SpacingBetweenSlices", "SliceThickness"):
        if hasattr(ds, tag):
            try:
                z_um = float(getattr(ds, tag)) * 1000.0
                break
            except Exception:
                pass

    return float(z_um), float(y_um), float(x_um)


# -----------------------------
# Preprocess / Segmentation
# -----------------------------
def to_uint8_percentile(vol_f32: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(vol_f32, [p_lo, p_hi])
    if hi <= lo + 1e-6:
        return np.zeros_like(vol_f32, dtype=np.uint8)
    x = (vol_f32 - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def segment_mask(
    gray_u8: np.ndarray,
    method: str = "otsu",
    blur_sigma: float = 0.8,
    close_iters: int = 2,
    open_iters: int = 1,
    invert: bool = False,
    sauvola_window: int = 51,
    sauvola_k: float = 0.15,
) -> Tuple[np.ndarray, dict]:
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

    st = ndi.generate_binary_structure(3, 1)  # 6-connected cleanup for segmentation stage
    if int(close_iters) > 0:
        m = ndi.binary_closing(m, structure=st, iterations=int(close_iters))
    if int(open_iters) > 0:
        m = ndi.binary_opening(m, structure=st, iterations=int(open_iters))

    info.update({"close_iters": int(close_iters), "open_iters": int(open_iters)})
    return m.astype(np.uint8), info


# -----------------------------
# Base targets (BVTV, Tb.Th, Tb.Sp, Euler)
# -----------------------------
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


# -----------------------------
# Skeleton metrics (evidence for "mesh-like" structure)
# -----------------------------
def _neighbor_degree_26(skel01: np.ndarray) -> np.ndarray:
    st26 = ndi.generate_binary_structure(3, 2)
    n = ndi.convolve(skel01.astype(np.uint8), st26.astype(np.uint8), mode="constant", cval=0)
    return (n - skel01.astype(np.uint8)).astype(np.int16)

def _prune_short_end_branches(skel01: np.ndarray, lmin: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Prunes endpoint branches shorter than lmin voxels (geodesic distance to nearest junction),
    iterating until stable.
    """
    lmin = int(max(1, lmin))
    st26 = ndi.generate_binary_structure(3, 2)
    sk = skel01.astype(bool)

    removed_total = 0
    it = 0
    while True:
        it += 1
        deg = _neighbor_degree_26(sk.astype(np.uint8))
        endpoints = sk & (deg == 1)
        junctions = sk & (deg >= 3)

        if not endpoints.any() or not junctions.any():
            break

        dist = np.full(sk.shape, np.inf, dtype=np.float32)
        dist[junctions] = 0.0

        frontier = junctions.copy()
        d = 0
        while d < lmin and frontier.any():
            d += 1
            nbr = ndi.binary_dilation(frontier, structure=st26) & sk & (dist == np.inf)
            dist[nbr] = float(d)
            frontier = nbr

        to_remove = endpoints & (dist < float(lmin))
        n_remove = int(to_remove.sum())
        if n_remove == 0:
            break

        sk[to_remove] = False
        removed_total += n_remove
        if it > 50:
            break

    info = {"prune_lmin": lmin, "prune_iters": it, "vox_removed": removed_total}
    return sk.astype(np.uint8), info

def skeletonize_with_skimage(vol01: np.ndarray) -> np.ndarray:
    if skeletonize_3d is None:
        raise RuntimeError("skimage.skeletonize_3d unavailable; install scikit-image or use --skeleton-mode fiji.")
    return skeletonize_3d(vol01.astype(bool)).astype(np.uint8)

def skeletonize_with_fiji(vol01: np.ndarray, fiji_exe: str, workdir: Path, command_name: str) -> np.ndarray:
    """
    Export vol as tif, call Fiji headless to skeletonize, read output.
    Works with many Fiji installs via 'Skeletonize (2D/3D)'.
    If you have a specific BoneJ command, pass it via --fiji-command.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    in_tif = workdir / "mask_for_fiji.tif"
    out_tif = workdir / "skeleton_from_fiji.tif"
    tiff.imwrite(in_tif, (vol01.astype(np.uint8) * 255).astype(np.uint8))

    macro = f"""
    open("{in_tif.as_posix()}");
    run("{command_name}");
    saveAs("Tiff", "{out_tif.as_posix()}");
    close();
    """

    with tempfile.NamedTemporaryFile("w", suffix=".ijm", delete=False) as mf:
        mf.write(macro)
        macro_path = mf.name

    try:
        subprocess.run(
            [fiji_exe, "--headless", "-macro", macro_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Fiji skeletonization failed.\n"
            f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}\n"
            f"Macro used: {macro_path}\n"
            f"Input: {in_tif}\n"
        ) from e

    sk = tiff.imread(out_tif)
    return (sk > 0).astype(np.uint8)

def compute_skeleton_metrics(
    mask01: np.ndarray,
    skeleton_mode: str = "skimage",
    fiji_exe: Optional[str] = None,
    fiji_command: str = "Skeletonize (2D/3D)",
    prune_lmin: int = 10,
    thin_first: bool = True,
    thin_iters: int = 1,
    debug_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Computes skeleton graph evidence metrics from the segmented mask.

    thin_first:
      - If True, we lightly "peel" the mask once to reduce plates to a cleaner core
        before skeletonization. This makes skeleton more representative of trabecular midlines.

    Returns dict with:
      skel_voxels, endpoints, junctions, endpoint_junction_ratio, prune_info
    """
    m = mask01.astype(bool)
    if not m.any():
        return {"skel_voxels": 0, "endpoints": 0, "junctions": 0, "endpoint_junction_ratio": None, "note": "empty_mask"}

    # Optional thinning pre-step (helps avoid skeletonizing thick plates as messy surfaces)
    if thin_first and int(thin_iters) > 0:
        st26 = ndi.generate_binary_structure(3, 2)
        core = m.copy()
        for _ in range(int(thin_iters)):
            er = ndi.binary_erosion(core, structure=st26, iterations=1, border_value=0)
            if er.sum() < 1000:  # avoid collapsing tiny structures
                break
            core = er
    else:
        core = m

    # Skeletonize
    if skeleton_mode == "skimage":
        skel_raw = skeletonize_with_skimage(core.astype(np.uint8))
    elif skeleton_mode == "fiji":
        if not fiji_exe:
            raise RuntimeError("--skeleton-mode fiji requires --fiji-exe.")
        wd = debug_dir if debug_dir is not None else Path(tempfile.mkdtemp())
        skel_raw = skeletonize_with_fiji(core.astype(np.uint8), fiji_exe=fiji_exe, workdir=wd, command_name=fiji_command)
    else:
        raise ValueError("skeleton_mode must be skimage or fiji")

    skel_pruned, prune_info = _prune_short_end_branches(skel_raw, lmin=int(prune_lmin))

    deg = _neighbor_degree_26(skel_pruned)
    sk = skel_pruned.astype(bool)
    endpoints = int((sk & (deg == 1)).sum())
    junctions = int((sk & (deg >= 3)).sum())
    ratio = (float(endpoints) / float(max(1, junctions))) if junctions > 0 else None

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        tiff.imwrite(debug_dir / "skeleton_raw.tif", (skel_raw * 255).astype(np.uint8))
        tiff.imwrite(debug_dir / "skeleton_pruned.tif", (skel_pruned * 255).astype(np.uint8))
        # also save the "core" used for skeletonization
        tiff.imwrite(debug_dir / "skeleton_input_core.tif", (core.astype(np.uint8) * 255).astype(np.uint8))

    return {
        "skel_voxels": int(sk.sum()),
        "endpoints": endpoints,
        "junctions": junctions,
        "endpoint_junction_ratio": ratio,
        "prune_info": prune_info,
        "skeleton_mode": skeleton_mode,
        "thin_first": bool(thin_first),
        "thin_iters": int(thin_iters),
        "fiji_command": fiji_command if skeleton_mode == "fiji" else None,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="VOI1 multiframe DICOM pipeline -> gray/mask/targets (+ optional skeleton metrics).")
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

    # NEW: skeleton evidence metrics
    ap.add_argument("--compute-skeleton-metrics", type=int, default=0, help="Compute skeleton graph evidence metrics.")
    ap.add_argument("--skeleton-mode", type=str, default="skimage", choices=["skimage", "fiji"])
    ap.add_argument("--fiji-exe", type=str, default=None, help="Path to Fiji executable for --skeleton-mode fiji")
    ap.add_argument("--fiji-command", type=str, default="Skeletonize (2D/3D)", help="ImageJ command name to run")
    ap.add_argument("--skeleton-prune-lmin", type=int, default=10)
    ap.add_argument("--skeleton-thin-first", type=int, default=1, help="Erode mask lightly before skeletonize.")
    ap.add_argument("--skeleton-thin-iters", type=int, default=1)

    ap.add_argument("--debug-skeleton", type=int, default=0, help="Save skeleton debug tifs per scan.")

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
        specimen = dcm_path.parent.name
        stem = dcm_path.stem
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

        # NEW: skeleton evidence metrics
        if bool(int(args.compute_skeleton_metrics)):
            dbg_dir = (out_root / "debug_skeleton" / f"{specimen}_{stem}") if bool(int(args.debug_skeleton)) else None
            skm = compute_skeleton_metrics(
                mask01=mask01,
                skeleton_mode=str(args.skeleton_mode),
                fiji_exe=args.fiji_exe,
                fiji_command=str(args.fiji_command),
                prune_lmin=int(args.skeleton_prune_lmin),
                thin_first=bool(int(args.skeleton_thin_first)),
                thin_iters=int(args.skeleton_thin_iters),
                debug_dir=dbg_dir,
            )
            targets["skeleton_metrics"] = skm

        targets_out = out_root / f"{specimen}_{stem}_targets.json"
        with open(targets_out, "w") as f:
            json.dump(targets, f, indent=2)
        print("Saved targets:", targets_out)

        msg = f"BVTV={targets['BVTV']:.3f} | Euler={targets['Euler']:.1f} | TbTh(p90)={targets['TbTh_um_p90']:.1f}um"
        if "skeleton_metrics" in targets:
            sm = targets["skeleton_metrics"]
            msg += f" | Skel(end/junc)={sm.get('endpoint_junction_ratio')}"
        print(msg)

    print("\nDone.")


if __name__ == "__main__":
    main()