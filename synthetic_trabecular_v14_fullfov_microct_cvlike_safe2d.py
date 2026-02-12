#!/usr/bin/env python3
r"""
synthetic_trabecular_v14_fullfov_microct_cvlike_safe2d.py

v14 (FULL FOV) — no cylindrical ROI mask.

Upgrades vs v13:
- SAFE curl-warp: automatically falls back to a 2D divergence-free warp per-slice when Z < 3
  (so --z 1 works and won't crash np.gradient).
- Docstring is raw (r"""...""") to avoid Windows backslash escape warnings.

Still includes:
- Curvature via divergence-free curl warp (3D or 2D fallback)
- Rod/plate topology via Hessian eigenvalue enhancement
- Correct BV/TV handling for invert_phase
- Micro-CT-like grayscale: partial-volume blur + low-frequency shading + marrow texture + noise (+ optional unsharp)
- Optional CV-like segmentation from grayscale (Sauvola local threshold or Otsu)

Outputs (in --outdir):
- mid.png           (binary from field, mid-slice)
- mask.tif          (binary stack, 0/255)
- gray_mid.png      (if --write-gray 1)
- gray.tif          (if --write-gray 1)
- metrics.json
- seg_mid.png       (if --segment-from-gray 1)
- seg_mask.tif      (if --segment-from-gray 1)

Dependencies:
- numpy
- scipy
- pillow
- tifffile
- scikit-image (euler_number, threshold_otsu, threshold_sauvola)

PowerShell examples:

# Single-slice threshold demo (works because Z=1 uses 2D warp fallback)
python .\synthetic_trabecular_v14_fullfov_microct_cvlike_safe2d.py `
  --outdir data\v14_threshold_demo `
  --xy 512 --z 1 `
  --seed 23 `
  --bvtv 0.22 `
  --invert-phase 1 `
  --write-gray 1 `
  --segment-from-gray 0

# 3D volume
python .\synthetic_trabecular_v14_fullfov_microct_cvlike_safe2d.py `
  --outdir data\v14_3d_demo `
  --xy 256 --z 160 `
  --seed 23 `
  --bvtv 0.18 `
  --invert-phase 1 `
  --write-gray 1 `
  --segment-from-gray 1 `
  --seg-method sauvola
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image
import tifffile as tiff
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from skimage.measure import euler_number
from skimage.filters import threshold_otsu, threshold_sauvola


# -----------------------------
# Params
# -----------------------------
@dataclass
class FieldParams:
    sigma: float = 4.2

    # Hessian rod/plate enhancement
    rod_strength: float = 0.85
    plate_strength: float = 0.65
    hessian_sigma: float = 1.6

    # Curl warp
    warp_sigma: float = 12.0
    warp_amp: float = 3.5

    # Final smoothing
    final_sigma: float = 0.7

    # Optional anisotropy multipliers on base sigma: (Z, Y, X)
    aniso_z: float = 1.15
    aniso_y: float = 1.00
    aniso_x: float = 1.00


@dataclass
class MorphParams:
    dilate_iters: int = 1
    close_iters: int = 2
    open_iters: int = 0


@dataclass
class GrayParams:
    write_gray: bool = False

    # Partial volume blur (voxels)
    pve_sigma: float = 1.2

    # Intensities
    bone_mean: float = 215.0
    marrow_mean: float = 55.0

    # Noise
    ct_noise_sd: float = 6.0
    bg_texture_sd: float = 2.0

    # Unsharp
    unsharp: float = 0.35

    # Full-FOV shading/texture
    shading_sigma: float = 45.0
    shading_amp: float = 14.0
    marrow_tex_sigma: float = 7.0
    marrow_tex_amp: float = 5.0


@dataclass
class SegParams:
    segment_from_gray: bool = False
    method: str = "sauvola"  # "sauvola" or "otsu"
    sauvola_window: int = 51
    sauvola_k: float = 0.15
    preblur_sigma: float = 1.0
    close_iters: int = 2
    open_iters: int = 1


# -----------------------------
# IO
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
# Utilities
# -----------------------------
def normalize(field: np.ndarray) -> np.ndarray:
    f = field.astype(np.float32)
    f -= float(f.mean())
    f /= float(f.std() + 1e-6)
    return f

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


# -----------------------------
# SAFE curl warp (3D + 2D fallback)
# -----------------------------
def curl_warp_field(field: np.ndarray, rng: np.random.Generator,
                    warp_sigma: float, warp_amp: float) -> np.ndarray:
    """
    Divergence-free warp:
      - If Z >= 3: full 3D curl noise warp
      - If Z < 3:  2D divergence-free warp per-slice in (Y,X), avoids np.gradient crash
    """
    if warp_amp <= 0:
        return field

    Z, Y, X = field.shape

    # 2D fallback for thin volumes
    if Z < 3:
        out = field.copy().astype(np.float32)
        for z in range(Z):
            A = ndi.gaussian_filter(rng.normal(0, 1, (Y, X)).astype(np.float32), sigma=warp_sigma)
            dA_dy = np.gradient(A, axis=0)
            dA_dx = np.gradient(A, axis=1)

            # 2D divergence-free vector field from scalar potential A:
            # v = ( dA/dy, -dA/dx )
            dy = dA_dy
            dx = -dA_dx

            mag = np.sqrt(dx * dx + dy * dy) + 1e-6
            dx = dx / mag * warp_amp
            dy = dy / mag * warp_amp

            yy, xx = np.meshgrid(np.arange(Y), np.arange(X), indexing="ij")
            coords = np.array([yy + dy, xx + dx])
            out[z] = map_coordinates(out[z], coords, order=1, mode="reflect").astype(np.float32)
        return out

    # 3D curl warp
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

    mag = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
    dx = dx / mag * warp_amp
    dy = dy / mag * warp_amp
    dz = dz / mag * warp_amp

    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    coords = np.array([zz + dz, yy + dy, xx + dx])
    warped = map_coordinates(field, coords, order=1, mode="reflect")
    return warped.astype(np.float32)


# -----------------------------
# Hessian rod/plate enhancement
# -----------------------------
def hessian_eigs_3d(f: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fxx = ndi.gaussian_filter(f, sigma=sigma, order=(0, 0, 2))
    fyy = ndi.gaussian_filter(f, sigma=sigma, order=(0, 2, 0))
    fzz = ndi.gaussian_filter(f, sigma=sigma, order=(2, 0, 0))
    fxy = ndi.gaussian_filter(f, sigma=sigma, order=(0, 1, 1))
    fxz = ndi.gaussian_filter(f, sigma=sigma, order=(1, 0, 1))
    fyz = ndi.gaussian_filter(f, sigma=sigma, order=(1, 1, 0))

    H = np.stack(
        [
            np.stack([fzz, fyz, fxz], axis=-1),
            np.stack([fyz, fyy, fxy], axis=-1),
            np.stack([fxz, fxy, fxx], axis=-1),
        ],
        axis=-2,
    )  # (..., 3, 3)

    w = np.linalg.eigvalsh(H.reshape(-1, 3, 3)).reshape(f.shape + (3,))
    idx = np.argsort(np.abs(w), axis=-1)
    w_sorted = np.take_along_axis(w, idx, axis=-1)
    l1, l2, l3 = w_sorted[..., 0], w_sorted[..., 1], w_sorted[..., 2]
    return l1, l2, l3

def rod_plate_enhance(field: np.ndarray, strength_rod: float, strength_plate: float, sigma: float) -> np.ndarray:
    if strength_rod <= 0 and strength_plate <= 0:
        return field

    l1, l2, l3 = hessian_eigs_3d(field, sigma=sigma)
    eps = 1e-6
    r1 = np.abs(l1) / (np.abs(l3) + eps)
    r2 = np.abs(l2) / (np.abs(l3) + eps)

    plate = np.exp(-(r1 ** 2))
    rod = np.exp(-(r1 ** 2)) * np.exp(-(r2 ** 2))

    plate = ndi.gaussian_filter(plate.astype(np.float32), sigma=1.0)
    rod = ndi.gaussian_filter(rod.astype(np.float32), sigma=1.0)
    return field + float(strength_plate) * plate + float(strength_rod) * rod


# -----------------------------
# Field generation
# -----------------------------
def generate_field(shape: Tuple[int, int, int], fp: FieldParams, rng: np.random.Generator) -> np.ndarray:
    f = rng.normal(0, 1, size=shape).astype(np.float32)

    sig = float(fp.sigma)
    f = ndi.gaussian_filter(
        f,
        sigma=(sig * float(fp.aniso_z), sig * float(fp.aniso_y), sig * float(fp.aniso_x)),
    )

    f = curl_warp_field(f, rng, warp_sigma=float(fp.warp_sigma), warp_amp=float(fp.warp_amp))

    f = rod_plate_enhance(
        f,
        strength_rod=float(fp.rod_strength),
        strength_plate=float(fp.plate_strength),
        sigma=float(fp.hessian_sigma),
    )

    if float(fp.final_sigma) > 0:
        f = ndi.gaussian_filter(f, sigma=float(fp.final_sigma))

    return normalize(f)


# -----------------------------
# Thresholding + morphology (BV/TV correct with invert_phase)
# -----------------------------
def threshold_to_bvtv(field: np.ndarray, bvtv: float, invert_phase: bool) -> Tuple[np.ndarray, float]:
    bvtv = float(np.clip(bvtv, 0.001, 0.999))
    if invert_phase:
        thr = float(np.quantile(field, bvtv))
        vol01 = (field <= thr).astype(np.uint8)
    else:
        thr = float(np.quantile(field, 1.0 - bvtv))
        vol01 = (field >= thr).astype(np.uint8)
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
# Micro-CT grayscale (FULL FOV)
# -----------------------------
def lowfreq_shading(shape: Tuple[int, int, int], rng: np.random.Generator, sigma: float, amp: float) -> np.ndarray:
    f = rng.normal(0, 1, size=shape).astype(np.float32)
    f = ndi.gaussian_filter(f, sigma=float(sigma))
    f = f / (float(f.std()) + 1e-6)
    return float(amp) * f

def marrow_texture(shape: Tuple[int, int, int], rng: np.random.Generator, sigma: float, amp: float) -> np.ndarray:
    t = rng.normal(0, 1, size=shape).astype(np.float32)
    t = ndi.gaussian_filter(t, sigma=float(sigma))
    t = t / (float(t.std()) + 1e-6)
    return float(amp) * t

def microct_gray_fullfov(vol01: np.ndarray, gp: GrayParams, rng: np.random.Generator) -> np.ndarray:
    x = vol01.astype(np.float32)
    if float(gp.pve_sigma) > 0:
        x = ndi.gaussian_filter(x, sigma=float(gp.pve_sigma))
    x = clamp01(x)

    gray = float(gp.marrow_mean) + x * (float(gp.bone_mean) - float(gp.marrow_mean))
    gray = gray + lowfreq_shading(gray.shape, rng, sigma=float(gp.shading_sigma), amp=float(gp.shading_amp))
    gray = gray + marrow_texture(gray.shape, rng, sigma=float(gp.marrow_tex_sigma), amp=float(gp.marrow_tex_amp))

    if float(gp.bg_texture_sd) > 0:
        gray += rng.normal(0.0, float(gp.bg_texture_sd), size=gray.shape).astype(np.float32)
    if float(gp.ct_noise_sd) > 0:
        gray += rng.normal(0.0, float(gp.ct_noise_sd), size=gray.shape).astype(np.float32)

    if float(gp.unsharp) > 0:
        blurred = ndi.gaussian_filter(gray, sigma=max(0.6, float(gp.pve_sigma)))
        gray = gray + float(gp.unsharp) * (gray - blurred)

    return np.clip(gray, 0.0, 255.0).astype(np.uint8)


# -----------------------------
# CV-like segmentation on grayscale (fast stand-in)
# -----------------------------
def segment_from_gray(gray_u8: np.ndarray, sp: SegParams) -> Tuple[np.ndarray, Dict[str, Any]]:
    g = gray_u8.astype(np.float32)
    if float(sp.preblur_sigma) > 0:
        g = ndi.gaussian_filter(g, sigma=float(sp.preblur_sigma))

    method = str(sp.method).lower().strip()
    if method == "otsu":
        thr = float(threshold_otsu(g))
        bone = (g >= thr)
        info = {"seg_method": "otsu", "thr": thr}
    elif method == "sauvola":
        win = int(sp.sauvola_window)
        win = max(15, win | 1)  # odd, >= 15
        thr_map = threshold_sauvola(g, window_size=win, k=float(sp.sauvola_k))
        bone = (g >= thr_map)
        info = {"seg_method": "sauvola", "window": win, "k": float(sp.sauvola_k)}
    else:
        raise ValueError(f"Unknown seg method: {sp.method}")

    st = ndi.generate_binary_structure(3, 1)
    if int(sp.close_iters) > 0:
        bone = ndi.binary_closing(bone, structure=st, iterations=int(sp.close_iters))
    if int(sp.open_iters) > 0:
        bone = ndi.binary_opening(bone, structure=st, iterations=int(sp.open_iters))

    return bone.astype(np.uint8), info


# -----------------------------
# Metrics
# -----------------------------
def bvtv(vol01: np.ndarray) -> float:
    return float(np.mean(vol01 > 0))

def thickness_quantiles_um(vol01: np.ndarray, voxel_um_xy: float, voxel_um_z: float) -> Dict[str, float]:
    bone = vol01.astype(bool)
    if bone.sum() == 0 or (~bone).sum() == 0:
        return {
            "tbth_um_p50": 0.0, "tbth_um_p75": 0.0, "tbth_um_p90": 0.0, "tbth_um_p95": 0.0,
            "tbsp_um_p50": 0.0, "tbsp_um_p75": 0.0, "tbsp_um_p90": 0.0, "tbsp_um_p95": 0.0,
        }

    sampling = (float(voxel_um_z), float(voxel_um_xy), float(voxel_um_xy))
    dt_b = ndi.distance_transform_edt(bone, sampling=sampling)
    dt_m = ndi.distance_transform_edt(~bone, sampling=sampling)

    rb = dt_b[bone]
    rm = dt_m[~bone]

    def q(x, p): return float(np.percentile(x, p)) if x.size else 0.0

    return {
        "tbth_um_p50": q(rb, 50), "tbth_um_p75": q(rb, 75), "tbth_um_p90": q(rb, 90), "tbth_um_p95": q(rb, 95),
        "tbsp_um_p50": q(rm, 50), "tbsp_um_p75": q(rm, 75), "tbsp_um_p90": q(rm, 90), "tbsp_um_p95": q(rm, 95),
    }

def euler_conn(vol01: np.ndarray) -> Dict[str, float]:
    eul = float(euler_number(vol01.astype(bool), connectivity=3))
    return {"euler": eul, "conn_proxy": float(1.0 - eul)}


# -----------------------------
# CLI + main
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v14 full-FOV trabecular generator (safe 2D curl warp for Z<3).")

    p.add_argument("--outdir", type=str, default="data/v14")
    p.add_argument("--xy", type=int, default=256)
    p.add_argument("--z", type=int, default=160)
    p.add_argument("--seed", type=int, default=23)

    p.add_argument("--bvtv", type=float, default=0.20)
    p.add_argument("--invert-phase", type=int, default=1)

    p.add_argument("--voxel-um", type=float, default=10.0)
    p.add_argument("--z-um", type=float, default=10.0)

    p.add_argument("--write-gray", type=int, default=1)

    p.add_argument("--segment-from-gray", type=int, default=0)
    p.add_argument("--seg-method", type=str, default="sauvola", choices=["sauvola", "otsu"])
    p.add_argument("--sauvola-window", type=int, default=51)
    p.add_argument("--sauvola-k", type=float, default=0.15)

    p.add_argument("--seg-close-iters", type=int, default=2)
    p.add_argument("--seg-open-iters", type=int, default=1)
    p.add_argument("--seg-preblur", type=float, default=1.0)

    return p

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

    fp = FieldParams()
    mp = MorphParams()
    gp = GrayParams(write_gray=bool(int(args.write_gray)))

    sp = SegParams(
        segment_from_gray=bool(int(args.segment_from_gray)),
        method=str(args.seg_method),
        sauvola_window=int(args.sauvola_window),
        sauvola_k=float(args.sauvola_k),
        preblur_sigma=float(args.seg_preblur),
        close_iters=int(args.seg_close_iters),
        open_iters=int(args.seg_open_iters),
    )

    # Generate binary from field
    field = generate_field(shape, fp, rng)
    vol01, thr = threshold_to_bvtv(field, bvtv=float(args.bvtv), invert_phase=invert_phase)
    vol01 = apply_morphology(vol01, mp)

    # Save binary outputs
    save_png_u8((vol01[Z // 2] * 255).astype(np.uint8), outdir / "mid.png")
    save_tif_u8((vol01 * 255).astype(np.uint8), outdir / "mask.tif")

    # Grayscale
    gray_u8: Optional[np.ndarray] = None
    if gp.write_gray:
        gray_u8 = microct_gray_fullfov(vol01, gp, rng)
        save_tif_u8(gray_u8, outdir / "gray.tif")
        save_png_u8(gray_u8[Z // 2], outdir / "gray_mid.png")

    # Optional segmentation from grayscale
    seg01: Optional[np.ndarray] = None
    seg_info: Dict[str, Any] = {}
    if sp.segment_from_gray:
        if gray_u8 is None:
            gray_u8 = microct_gray_fullfov(vol01, gp, rng)
        seg01, seg_info = segment_from_gray(gray_u8, sp)
        save_tif_u8((seg01 * 255).astype(np.uint8), outdir / "seg_mask.tif")
        save_png_u8((seg01[Z // 2] * 255).astype(np.uint8), outdir / "seg_mid.png")

    metrics: Dict[str, Any] = {
        "threshold_quantile_or_bvtv": float(thr),
        "invert_phase": invert_phase,
        "voxel_um_xy": voxel_um_xy,
        "voxel_um_z": voxel_um_z,
        "shape": [Z, H, W],
        "params": {"field": asdict(fp), "morph": asdict(mp), "gray": asdict(gp), "seg": asdict(sp)},
        "binary_field": {
            "BVTV": bvtv(vol01),
            **thickness_quantiles_um(vol01, voxel_um_xy, voxel_um_z),
            **euler_conn(vol01),
        },
    }

    if seg01 is not None:
        metrics["binary_grayseg"] = {
            **seg_info,
            "BVTV": bvtv(seg01),
            **thickness_quantiles_um(seg01, voxel_um_xy, voxel_um_z),
            **euler_conn(seg01),
        }

    save_json(metrics, outdir / "metrics.json")

    print(
        f"Saved to: {outdir}\n"
        f"Field-binary BV/TV={metrics['binary_field']['BVTV']:.3f} | "
        f"Tb.Th(p90)={metrics['binary_field']['tbth_um_p90']:.1f}um | "
        f"Euler={metrics['binary_field']['euler']:.1f}"
    )
    if seg01 is not None:
        print(
            f"Gray-seg BV/TV={metrics['binary_grayseg']['BVTV']:.3f} | "
            f"Tb.Th(p90)={metrics['binary_grayseg']['tbth_um_p90']:.1f}um | "
            f"Euler={metrics['binary_grayseg']['euler']:.1f} | "
            f"Seg={metrics['binary_grayseg'].get('seg_method','?')}"
        )


if __name__ == "__main__":
    main()
