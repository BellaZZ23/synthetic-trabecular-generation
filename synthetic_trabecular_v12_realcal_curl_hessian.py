#!/usr/bin/env python3
"""
synthetic_trabecular_v12_realcal_curl_hessian.py

v12 upgrades:
- Better curvature via divergence-free "curl noise" coordinate warp (less blobby, more bending)
- More bone-like rod/plate topology via Hessian eigenvalue responses (tubular + sheet enhancement)
- Optional real micro-CT reading (multi-page TIFF or folder-of-slices) + optional segmentation
- Optional parameter fitting to real data by matching descriptor targets (BV/TV, thickness quantiles, Euler)

Outputs (in --outdir):
- mid.png        (binary mid-slice)
- mask.tif       (binary 3D stack, 0/255)
- metrics.json   (BV/TV, thickness quantiles EDT proxies, Euler)
- (optional) gray.tif + gray_mid.png if --write-gray 1
- (optional) real_mid.png + real_mask.tif + real_metrics.json if --real provided

Dependencies:
- numpy
- scipy
- pillow
- tifffile
- scikit-image (euler_number, threshold_otsu)

Examples:

# Pure synthetic
python synthetic_trabecular_v12_realcal_curl_hessian.py ^
  --outdir data/v12_syn --xy 256 --z 160 --seed 23 --bvtv 0.26 --invert-phase 1 --write-gray 1

# Read real micro-CT (tif stack), segment, export metrics + a copy of the binary
python synthetic_trabecular_v12_realcal_curl_hessian.py ^
  --outdir data/v12_with_real --real path/to/real_stack.tif --real-is-binary 0 --save-real 1

# Fit parameters to match real descriptors, then generate synthetic with matched BV/TV etc.
python synthetic_trabecular_v12_realcal_curl_hessian.py ^
  --outdir data/v12_fit --real path/to/real_stack.tif --real-is-binary 0 --fit-to-real 1 --fit-trials 60
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List

import numpy as np
from PIL import Image
import tifffile as tiff
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from skimage.measure import euler_number
from skimage.filters import threshold_otsu


# -----------------------------
# Params
# -----------------------------
@dataclass
class FieldParams:
    # Base smoothing of random field (higher -> larger features)
    sigma: float = 4.2

    # Rod/plate enhancement strengths (0..~1.5)
    rod_strength: float = 0.85
    plate_strength: float = 0.65

    # Hessian scale for rod/plate detection (voxels)
    hessian_sigma: float = 1.6

    # Curvature warp settings
    warp_sigma: float = 12.0   # smoothness of displacement (voxels)
    warp_amp: float = 3.5      # amplitude of displacement (voxels)

    # Final small blur to remove harsh edges
    final_sigma: float = 0.7


@dataclass
class MorphParams:
    # Mild bridging after threshold (keep small; don't overfill)
    dilate_iters: int = 1
    close_iters: int = 2
    open_iters: int = 0


@dataclass
class GrayParams:
    write_gray: bool = False
    pve_sigma: float = 1.2
    bone_mean: float = 215.0
    marrow_mean: float = 50.0
    ct_noise_sd: float = 6.0
    bg_texture_sd: float = 2.0
    unsharp: float = 0.4


# -----------------------------
# IO helpers
# -----------------------------
def save_png_u8(img_u8: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_u8.astype(np.uint8), mode="L").save(path)

def save_tif_u8(stack_u8: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(path, stack_u8.astype(np.uint8), imagej=True, dtype=np.uint8)

def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# -----------------------------
# Utility
# -----------------------------
def normalize(field: np.ndarray) -> np.ndarray:
    f = field.astype(np.float32)
    f -= float(f.mean())
    f /= float(f.std() + 1e-6)
    return f

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def safe_float(x: Union[int, float, np.number]) -> float:
    return float(x)


# -----------------------------
# Field shaping
# -----------------------------
def curl_warp_field(field: np.ndarray, rng: np.random.Generator,
                    warp_sigma: float, warp_amp: float) -> np.ndarray:
    """
    Divergence-free warp using curl of a smooth vector potential.
    Produces more bending and less "puffy" distortion than independent dx/dy/dz fields.
    """
    if warp_amp <= 0:
        return field

    Ax = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma)
    Ay = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma)
    Az = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma)

    # curl(A) = (dAz/dy - dAy/dz, dAx/dz - dAz/dx, dAy/dx - dAx/dy)
    dAz_dy = np.gradient(Az, axis=1)
    dAy_dz = np.gradient(Ay, axis=0)
    dAx_dz = np.gradient(Ax, axis=0)
    dAz_dx = np.gradient(Az, axis=2)
    dAy_dx = np.gradient(Ay, axis=2)
    dAx_dy = np.gradient(Ax, axis=1)

    dz = (dAz_dy - dAy_dz)
    dy = (dAx_dz - dAz_dx)
    dx = (dAy_dx - dAx_dy)

    mag = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6
    dx = dx / mag * warp_amp
    dy = dy / mag * warp_amp
    dz = dz / mag * warp_amp

    Z, Y, X = field.shape
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    coords = np.array([zz + dz, yy + dy, xx + dx])
    warped = map_coordinates(field, coords, order=1, mode="reflect")
    return warped.astype(np.float32)

def hessian_eigs_3d(f: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hessian eigenvalues at scale sigma (Gaussian second derivatives).
    Returns eigenvalues sorted by absolute value: |l1| <= |l2| <= |l3|.
    NOTE: np.linalg.eigvalsh on full volume can be heavy for very large volumes.
    """
    # Second derivatives (Gaussian)
    fxx = ndi.gaussian_filter(f, sigma=sigma, order=(0, 0, 2))
    fyy = ndi.gaussian_filter(f, sigma=sigma, order=(0, 2, 0))
    fzz = ndi.gaussian_filter(f, sigma=sigma, order=(2, 0, 0))
    fxy = ndi.gaussian_filter(f, sigma=sigma, order=(0, 1, 1))
    fxz = ndi.gaussian_filter(f, sigma=sigma, order=(1, 0, 1))
    fyz = ndi.gaussian_filter(f, sigma=sigma, order=(1, 1, 0))

    # Assemble Hessian per voxel (symmetric 3x3)
    H = np.stack([
        np.stack([fzz, fyz, fxz], axis=-1),
        np.stack([fyz, fyy, fxy], axis=-1),
        np.stack([fxz, fxy, fxx], axis=-1),
    ], axis=-2)  # (..., 3, 3)

    w = np.linalg.eigvalsh(H.reshape(-1, 3, 3)).reshape(f.shape + (3,))
    idx = np.argsort(np.abs(w), axis=-1)
    w_sorted = np.take_along_axis(w, idx, axis=-1)
    l1, l2, l3 = w_sorted[..., 0], w_sorted[..., 1], w_sorted[..., 2]
    return l1, l2, l3

def rod_plate_enhance(field: np.ndarray,
                      strength_rod: float,
                      strength_plate: float,
                      sigma: float = 1.6) -> np.ndarray:
    """
    Heuristic rod/plate enhancement from Hessian eigenvalue patterns.

    Intuition (sorted by |l|):
      - Plate-like: l3 dominates, l1 ~ 0 (and often l2 ~ 0)
      - Rod-like: l2 and l3 dominate, l1 ~ 0

    We form smooth responses and add them to the field.
    """
    if strength_rod <= 0 and strength_plate <= 0:
        return field

    l1, l2, l3 = hessian_eigs_3d(field, sigma=sigma)
    eps = 1e-6

    # Ratio-based "how rod-like / plate-like"
    r1 = np.abs(l1) / (np.abs(l3) + eps)
    r2 = np.abs(l2) / (np.abs(l3) + eps)

    # Responses: higher when ratios are small (i.e., l3 dominates)
    plate = np.exp(-(r1 ** 2))
    rod = np.exp(-(r1 ** 2)) * np.exp(-(r2 ** 2))

    plate = ndi.gaussian_filter(plate.astype(np.float32), sigma=1.0)
    rod = ndi.gaussian_filter(rod.astype(np.float32), sigma=1.0)

    return field + safe_float(strength_plate) * plate + safe_float(strength_rod) * rod

def generate_field(shape: Tuple[int, int, int], fp: FieldParams, rng: np.random.Generator) -> np.ndarray:
    # Base GRF
    f = rng.normal(0, 1, size=shape).astype(np.float32)
    f = ndi.gaussian_filter(f, sigma=safe_float(fp.sigma))

    # Better curvature (divergence-free warp)
    f = curl_warp_field(f, rng, warp_sigma=safe_float(fp.warp_sigma), warp_amp=safe_float(fp.warp_amp))

    # Bone-like rod/plate enhancement (Hessian-based)
    f = rod_plate_enhance(
        f,
        strength_rod=safe_float(fp.rod_strength),
        strength_plate=safe_float(fp.plate_strength),
        sigma=safe_float(fp.hessian_sigma),
    )

    # Final blur
    if safe_float(fp.final_sigma) > 0:
        f = ndi.gaussian_filter(f, sigma=safe_float(fp.final_sigma))

    return normalize(f)


# -----------------------------
# Thresholding + morphology
# -----------------------------
def threshold_to_bvtv(field: np.ndarray, bvtv: float, invert_phase: bool) -> Tuple[np.ndarray, float]:
    bvtv = float(np.clip(bvtv, 0.001, 0.999))
    thr = float(np.quantile(field, 1.0 - bvtv))
    vol01 = (field < thr).astype(np.uint8) if invert_phase else (field >= thr).astype(np.uint8)
    return vol01, thr

def apply_morphology(vol01: np.ndarray, mp: MorphParams) -> np.ndarray:
    v = vol01.astype(bool)
    st = ndi.generate_binary_structure(3, 1)
    if int(mp.open_iters) > 0:
        v = ndi.binary_opening(v, structure=st, iterations=int(mp.open_iters))
    if int(mp.dilate_iters) > 0:
        v = ndi.binary_dilation(v, structure=st, iterations=int(mp.dilate_iters))
    if int(mp.close_iters) > 0:
        v = ndi.binary_closing(v, structure=st, iterations=int(mp.close_iters))
    return v.astype(np.uint8)


# -----------------------------
# Gray micro-CT look (optional)
# -----------------------------
def microct_gray(vol01: np.ndarray, gp: GrayParams, rng: np.random.Generator) -> np.ndarray:
    x = vol01.astype(np.float32)
    if safe_float(gp.pve_sigma) > 0:
        x = ndi.gaussian_filter(x, sigma=safe_float(gp.pve_sigma))
    x = clamp01(x)

    gray = safe_float(gp.marrow_mean) + x * (safe_float(gp.bone_mean) - safe_float(gp.marrow_mean))

    if safe_float(gp.bg_texture_sd) > 0:
        gray += rng.normal(0.0, safe_float(gp.bg_texture_sd), size=gray.shape).astype(np.float32)
    if safe_float(gp.ct_noise_sd) > 0:
        gray += rng.normal(0.0, safe_float(gp.ct_noise_sd), size=gray.shape).astype(np.float32)

    if safe_float(gp.unsharp) > 0:
        blurred = ndi.gaussian_filter(gray, sigma=max(0.5, safe_float(gp.pve_sigma)))
        gray = gray + safe_float(gp.unsharp) * (gray - blurred)

    return np.clip(gray, 0.0, 255.0).astype(np.uint8)


# -----------------------------
# Metrics (BoneJ-like proxies)
# -----------------------------
def bvtv(vol01: np.ndarray) -> float:
    return float(np.mean(vol01 > 0))

def thickness_quantiles_um(vol01: np.ndarray, voxel_um_xy: float, voxel_um_z: float) -> Dict[str, float]:
    bone = vol01.astype(bool)
    if bone.sum() == 0:
        return {
            "tbth_um_p50": 0.0, "tbth_um_p75": 0.0, "tbth_um_p90": 0.0, "tbth_um_p95": 0.0,
            "tbsp_um_p50": 0.0, "tbsp_um_p75": 0.0, "tbsp_um_p90": 0.0, "tbsp_um_p95": 0.0,
        }

    sampling = (float(voxel_um_z), float(voxel_um_xy), float(voxel_um_xy))
    dt_b = ndi.distance_transform_edt(bone, sampling=sampling)
    dt_m = ndi.distance_transform_edt(~bone, sampling=sampling)

    rb = dt_b[bone]
    rm = dt_m[~bone]

    # These EDT values are "radius-like" (distance to boundary). BoneJ thickness differs,
    # but these are good stable proxies for matching distributions.
    def q(x, p): return float(np.percentile(x, p)) if x.size else 0.0

    return {
        "tbth_um_p50": q(rb, 50), "tbth_um_p75": q(rb, 75), "tbth_um_p90": q(rb, 90), "tbth_um_p95": q(rb, 95),
        "tbsp_um_p50": q(rm, 50), "tbsp_um_p75": q(rm, 75), "tbsp_um_p90": q(rm, 90), "tbsp_um_p95": q(rm, 95),
    }

def euler_conn(vol01: np.ndarray) -> Dict[str, float]:
    eul = float(euler_number(vol01.astype(bool), connectivity=3))
    conn = float(1.0 - eul)
    return {"euler": eul, "conn_proxy": conn}


# -----------------------------
# Real micro-CT reading / segmentation
# -----------------------------
def _list_image_files(folder: Path) -> List[Path]:
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts and p.is_file()]
    return files

def read_volume(path: str) -> np.ndarray:
    """
    Reads:
      - multi-page TIFF stack
      - or a folder containing slice images (tif/png/jpg)
    Returns float32 volume in (Z, Y, X).
    """
    p = Path(path)
    if p.is_dir():
        files = _list_image_files(p)
        if not files:
            raise FileNotFoundError(f"No image files found in folder: {p}")
        slices = []
        for fp in files:
            if fp.suffix.lower() in (".tif", ".tiff"):
                arr = tiff.imread(fp)
            else:
                arr = np.array(Image.open(fp))
            if arr.ndim == 3:
                arr = arr[..., 0]
            slices.append(arr)
        vol = np.stack(slices, axis=0)
    else:
        vol = tiff.imread(str(p))
        vol = np.asarray(vol)

    # Handle common shapes
    if vol.ndim == 2:
        vol = vol[None, ...]
    if vol.ndim == 4 and vol.shape[-1] in (1, 3):
        vol = vol[..., 0]

    # Heuristic axis fix: if looks like (Y,X,Z), move last to front
    # (This is only a guess; adjust if your datasets differ.)
    if vol.shape[0] < 8 and vol.shape[-1] > 32:
        vol = np.moveaxis(vol, -1, 0)

    return vol.astype(np.float32)

def segment_bone_otsu(vol_gray: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Otsu threshold + mild cleanup. Returns (vol01, thr_used).
    """
    g = vol_gray.astype(np.float32)
    thr = float(threshold_otsu(g))
    bone = (g >= thr)

    st = ndi.generate_binary_structure(3, 1)
    bone = ndi.binary_opening(bone, structure=st, iterations=1)
    bone = ndi.binary_closing(bone, structure=st, iterations=2)
    return bone.astype(np.uint8), thr


# -----------------------------
# Fit params to real (simple random search)
# -----------------------------
def fit_params_to_real(real_vol01: np.ndarray,
                       rng: np.random.Generator,
                       shape: Tuple[int, int, int],
                       voxel_um_xy: float, voxel_um_z: float,
                       invert_phase: bool,
                       trials: int = 60) -> FieldParams:
    """
    Random search to match descriptors (BV/TV, EDT thickness quantiles, Euler).
    This is a practical “no-ML” way to tune the generator to a dataset/ROI.
    """
    real_targets: Dict[str, float] = {
        "BVTV": bvtv(real_vol01),
        **thickness_quantiles_um(real_vol01, voxel_um_xy, voxel_um_z),
        **euler_conn(real_vol01),
    }

    # Weights: BV/TV strongly enforced; others moderate
    weights = {
        "BVTV": 10.0,
        "tbth_um_p90": 1.0,
        "tbth_um_p50": 0.6,
        "tbsp_um_p90": 0.6,
        "euler": 0.25,
    }

    def loss(m: Dict[str, float]) -> float:
        L = 0.0
        for k, w in weights.items():
            if k in real_targets and k in m:
                L += w * abs(float(m[k]) - float(real_targets[k]))
        return float(L)

    best_fp: Optional[FieldParams] = None
    best_L = 1e18

    # Fixed morphology during fitting (keep stable)
    mp = MorphParams(dilate_iters=1, close_iters=2, open_iters=0)

    for _ in range(int(trials)):
        fp = FieldParams(
            sigma=float(rng.uniform(3.0, 6.0)),
            rod_strength=float(rng.uniform(0.2, 1.4)),
            plate_strength=float(rng.uniform(0.2, 1.2)),
            hessian_sigma=float(rng.uniform(1.0, 2.4)),
            warp_sigma=float(rng.uniform(8.0, 18.0)),
            warp_amp=float(rng.uniform(1.0, 6.0)),
            final_sigma=float(rng.uniform(0.3, 1.2)),
        )

        field = generate_field(shape, fp, rng)
        vol01, _ = threshold_to_bvtv(field, bvtv=float(real_targets["BVTV"]), invert_phase=bool(invert_phase))
        vol01 = apply_morphology(vol01, mp)

        m = {
            "BVTV": bvtv(vol01),
            **thickness_quantiles_um(vol01, voxel_um_xy, voxel_um_z),
            **euler_conn(vol01),
        }

        L = loss(m)
        if L < best_L:
            best_L, best_fp = L, fp

    assert best_fp is not None
    return best_fp


# -----------------------------
# CLI + main
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v12 trabecular generator (curl-warp + Hessian rod/plate + optional real calibration).")

    p.add_argument("--outdir", type=str, default="data/v12")
    p.add_argument("--xy", type=int, default=256)
    p.add_argument("--z", type=int, default=160)
    p.add_argument("--seed", type=int, default=23)

    p.add_argument("--bvtv", type=float, default=0.26)
    p.add_argument("--invert-phase", type=int, default=1, help="1 often yields better trabecular-like networks.")

    p.add_argument("--voxel-um", type=float, default=10.0, help="XY voxel size (um).")
    p.add_argument("--z-um", type=float, default=10.0, help="Z voxel size (um).")

    # Optional grayscale export
    p.add_argument("--write-gray", type=int, default=0)

    # Real micro-CT input
    p.add_argument("--real", type=str, default="", help="Path to real micro-CT (multi-page .tif/.tiff) or folder of slices.")
    p.add_argument("--real-is-binary", type=int, default=0, help="1 if real volume is already binary (0/1 or 0/255).")
    p.add_argument("--save-real", type=int, default=0, help="1 to export real binary + metrics into outdir.")
    p.add_argument("--real-crop", type=str, default="", help='Optional crop "z0:z1,y0:y1,x0:x1" (end-exclusive).')

    # Fit to real
    p.add_argument("--fit-to-real", type=int, default=0, help="1 to fit generator params to real descriptors before generating.")
    p.add_argument("--fit-trials", type=int, default=60, help="Number of random trials for fitting.")
    return p

def parse_crop(s: str) -> Optional[Tuple[slice, slice, slice]]:
    """
    Parse crop string like "10:150,20:220,30:230" into slices for (z,y,x).
    Empty string -> None.
    """
    if not s:
        return None
    parts = s.split(",")
    if len(parts) != 3:
        raise ValueError('Crop must be "z0:z1,y0:y1,x0:x1"')
    sl = []
    for p in parts:
        a, b = p.split(":")
        sl.append(slice(int(a), int(b)))
    return sl[0], sl[1], sl[2]

def ensure_binary01(vol: np.ndarray) -> np.ndarray:
    """
    Converts volumes with values {0,255} or arbitrary grayscale to binary 0/1.
    If already 0/1, returns as uint8.
    """
    v = vol
    if v.dtype != np.uint8:
        v = v.astype(np.float32)
    # If it looks like {0,255}
    uniq = np.unique(v[:: max(1, v.shape[0] // 8), :: max(1, v.shape[1] // 8), :: max(1, v.shape[2] // 8)])
    if uniq.size <= 4 and (np.isin(uniq, [0, 1, 255]).all()):
        return (v > 0).astype(np.uint8)
    # Otherwise interpret as grayscale and threshold at mid
    return (v > np.mean(v)).astype(np.uint8)

def main():
    args = build_parser().parse_args()
    rng = np.random.default_rng(int(args.seed))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    Z = int(args.z)
    H = W = int(args.xy)
    shape = (Z, H, W)

    voxel_um_xy = float(args.voxel_um)
    voxel_um_z = float(args.z_um)
    invert_phase = bool(int(args.invert_phase))

    # Defaults tuned for curvature + rod/plate mix
    fp = FieldParams(
        sigma=4.2,
        rod_strength=0.85,
        plate_strength=0.65,
        hessian_sigma=1.6,
        warp_sigma=12.0,
        warp_amp=3.5,
        final_sigma=0.7,
    )

    mp = MorphParams(
        dilate_iters=1,
        close_iters=2,
        open_iters=0,
    )

    gp = GrayParams(
        write_gray=bool(int(args.write_gray)),
        pve_sigma=1.2,
        ct_noise_sd=6.0,
        bg_texture_sd=2.0,
        unsharp=0.4,
    )

    # --- Optional: load real micro-CT and compute metrics / fit params
    real_metrics: Optional[Dict[str, Any]] = None
    if str(args.real).strip():
        real_vol = read_volume(args.real)
        crop = parse_crop(str(args.real_crop).strip())
        if crop is not None:
            real_vol = real_vol[crop[0], crop[1], crop[2]]

        # Match generator shape by center-cropping or padding
        rz, ry, rx = real_vol.shape
        def center_crop_or_pad(vol: np.ndarray, target: Tuple[int,int,int]) -> np.ndarray:
            tz, ty, tx = target
            z0 = max(0, (vol.shape[0] - tz)//2)
            y0 = max(0, (vol.shape[1] - ty)//2)
            x0 = max(0, (vol.shape[2] - tx)//2)
            cropped = vol[z0:z0+tz, y0:y0+ty, x0:x0+tx]
            # Pad if smaller
            padz = max(0, tz - cropped.shape[0])
            pady = max(0, ty - cropped.shape[1])
            padx = max(0, tx - cropped.shape[2])
            if padz or pady or padx:
                cropped = np.pad(
                    cropped,
                    pad_width=((padz//2, padz - padz//2),
                               (pady//2, pady - pady//2),
                               (padx//2, padx - padx//2)),
                    mode="edge",
                )
            return cropped

        real_vol = center_crop_or_pad(real_vol, shape)

        if int(args.real_is_binary) == 1:
            real_vol01 = ensure_binary01(real_vol)
            real_thr = None
        else:
            real_vol01, real_thr = segment_bone_otsu(real_vol)

        real_metrics = {
            "BVTV": bvtv(real_vol01),
            "real_threshold_otsu": float(real_thr) if real_thr is not None else None,
            **thickness_quantiles_um(real_vol01, voxel_um_xy, voxel_um_z),
            **euler_conn(real_vol01),
            "shape": list(real_vol01.shape),
            "voxel_um_xy": voxel_um_xy,
            "voxel_um_z": voxel_um_z,
        }

        if int(args.save_real) == 1:
            save_tif_u8((real_vol01 * 255).astype(np.uint8), outdir / "real_mask.tif")
            save_png_u8((real_vol01[Z // 2] * 255).astype(np.uint8), outdir / "real_mid.png")
            save_json(real_metrics, outdir / "real_metrics.json")

        if int(args.fit_to_real) == 1:
            fp = fit_params_to_real(
                real_vol01=real_vol01,
                rng=rng,
                shape=shape,
                voxel_um_xy=voxel_um_xy,
                voxel_um_z=voxel_um_z,
                invert_phase=invert_phase,
                trials=int(args.fit_trials),
            )

    # --- Generate synthetic
    field = generate_field(shape, fp, rng)
    vol01, thr = threshold_to_bvtv(field, bvtv=float(args.bvtv), invert_phase=invert_phase)
    vol01 = apply_morphology(vol01, mp)

    # Save binary outputs
    mid = (vol01[Z // 2] * 255).astype(np.uint8)
    save_png_u8(mid, outdir / "mid.png")
    save_tif_u8((vol01 * 255).astype(np.uint8), outdir / "mask.tif")

    # Optional grayscale
    if gp.write_gray:
        gray = microct_gray(vol01, gp, rng)
        save_tif_u8(gray.astype(np.uint8), outdir / "gray.tif")
        save_png_u8(gray[Z // 2].astype(np.uint8), outdir / "gray_mid.png")

    # Metrics
    metrics: Dict[str, Any] = {
        "BVTV": bvtv(vol01),
        "threshold_quantile": float(thr),
        "invert_phase": invert_phase,
        **thickness_quantiles_um(vol01, voxel_um_xy=voxel_um_xy, voxel_um_z=voxel_um_z),
        **euler_conn(vol01),
        "params": {"field": asdict(fp), "morph": asdict(mp), "gray": asdict(gp)},
        "shape": [Z, H, W],
        "voxel_um_xy": voxel_um_xy,
        "voxel_um_z": voxel_um_z,
    }
    if real_metrics is not None:
        metrics["real_metrics"] = real_metrics

    save_json(metrics, outdir / "metrics.json")

    print(
        f"Saved to: {outdir}\n"
        f"BV/TV={metrics['BVTV']:.3f} | "
        f"Tb.Th(p90)={metrics['tbth_um_p90']:.1f}um | Tb.Sp(p90)={metrics['tbsp_um_p90']:.1f}um | "
        f"Euler={metrics['euler']:.1f}\n"
        f"FieldParams: {asdict(fp)}"
    )


if __name__ == "__main__":
    main()
