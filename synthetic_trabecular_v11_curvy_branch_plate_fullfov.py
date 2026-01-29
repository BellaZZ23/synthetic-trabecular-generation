#!/usr/bin/env python3
"""
synthetic_trabecular_v11_curvy_branch_plate_fullfov.py

v11 (curvy + branch-like connectivity) — complete standalone generator.

What this version is tuned to do:
- Produce trabecular-like interconnected structures with MORE curvature
- Encourage plate-like sheets + branch junctions (plate–rod mix)
- Avoid "salt & pepper" islands by using smooth fields + gentle morphology

Outputs (in --outdir):
- mid.png        (binary mid-slice)
- mask.tif       (binary 3D stack, 0/255)
- metrics.json   (BV/TV, Tb.Th/Tb.Sp EDT proxies, Euler)
- (optional) gray.tif + gray_mid.png if --write-gray 1

Dependencies:
- numpy
- scipy
- pillow
- tifffile
- scikit-image (euler_number)

PowerShell run:
python .\synthetic_trabecular_v11_curvy_branch_plate_fullfov.py `
  --outdir data\v11_curvy_check `
  --xy 256 --z 160 `
  --seed 23 `
  --bvtv 0.26 `
  --voxel-um 10 `
  --write-gray 1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from PIL import Image
import tifffile as tiff
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from skimage.measure import euler_number


# -----------------------------
# Params
# -----------------------------
@dataclass
class FieldParams:
    # Base smoothing of random field (higher -> larger features)
    sigma: float = 4.2

    # Plate-like sheet emphasis (0..1)
    plate_strength: float = 0.65

    # Branch/junction emphasis (0..1+)
    branch_strength: float = 0.85

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


# -----------------------------
# Field shaping (curvature + plates + branches)
# -----------------------------
def normalize(field: np.ndarray) -> np.ndarray:
    f = field.astype(np.float32)
    f -= float(f.mean())
    f /= float(f.std() + 1e-6)
    return f

def warp_field(field: np.ndarray, rng: np.random.Generator, warp_sigma: float, warp_amp: float) -> np.ndarray:
    """
    Smooth coordinate warp to introduce curvature.
    Produces smoothly curving trabeculae rather than straight-ish blobs.
    """
    if warp_amp <= 0:
        return field

    Z, Y, X = field.shape

    dz = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma) * warp_amp
    dy = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma) * warp_amp
    dx = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma) * warp_amp

    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    coords = np.array([zz + dz, yy + dy, xx + dx])
    warped = map_coordinates(field, coords, order=1, mode="reflect")
    return warped.astype(np.float32)

def plate_bias(field: np.ndarray, strength: float) -> np.ndarray:
    """
    Encourages plate-like structures by local integration along planes.
    """
    if strength <= 0:
        return field
    fx = ndi.uniform_filter(field, size=(1, 1, 9))
    fy = ndi.uniform_filter(field, size=(1, 9, 1))
    fz = ndi.uniform_filter(field, size=(9, 1, 1))
    plates = (fx + fy + fz) / 3.0
    return (1.0 - strength) * field + strength * plates

def branch_link_bias(field: np.ndarray, strength: float) -> np.ndarray:
    """
    Encourages branch-like connectivity by boosting ridge/medial structures.
    We use gradient magnitude as a simple ridge proxy and then smooth it.
    """
    if strength <= 0:
        return field

    gz, gy, gx = np.gradient(field)
    grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    grad_mag = grad_mag / (float(grad_mag.max()) + 1e-6)

    ridge = ndi.gaussian_filter(grad_mag, sigma=2.0)
    return field + float(strength) * ridge

def generate_field(shape: Tuple[int, int, int], fp: FieldParams, rng: np.random.Generator) -> np.ndarray:
    # Base GRF
    f = rng.normal(0, 1, size=shape).astype(np.float32)
    f = ndi.gaussian_filter(f, sigma=float(fp.sigma))

    # Curvature (warp)
    f = warp_field(f, rng, warp_sigma=float(fp.warp_sigma), warp_amp=float(fp.warp_amp))

    # Plates + branch junctions
    f = plate_bias(f, float(fp.plate_strength))
    f = branch_link_bias(f, float(fp.branch_strength))

    # Small blur to smooth final field
    if float(fp.final_sigma) > 0:
        f = ndi.gaussian_filter(f, sigma=float(fp.final_sigma))

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
def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def microct_gray(vol01: np.ndarray, gp: GrayParams, rng: np.random.Generator) -> np.ndarray:
    x = vol01.astype(np.float32)
    if float(gp.pve_sigma) > 0:
        x = ndi.gaussian_filter(x, sigma=float(gp.pve_sigma))
    x = clamp01(x)

    gray = float(gp.marrow_mean) + x * (float(gp.bone_mean) - float(gp.marrow_mean))

    if float(gp.bg_texture_sd) > 0:
        gray += rng.normal(0.0, float(gp.bg_texture_sd), size=gray.shape).astype(np.float32)
    if float(gp.ct_noise_sd) > 0:
        gray += rng.normal(0.0, float(gp.ct_noise_sd), size=gray.shape).astype(np.float32)

    if float(gp.unsharp) > 0:
        blurred = ndi.gaussian_filter(gray, sigma=max(0.5, float(gp.pve_sigma)))
        gray = gray + float(gp.unsharp) * (gray - blurred)

    return np.clip(gray, 0.0, 255.0).astype(np.uint8)


# -----------------------------
# Metrics (BoneJ-like proxies)
# -----------------------------
def bvtv(vol01: np.ndarray) -> float:
    return float(np.mean(vol01 > 0))

def tbth_tbsp_p90_um(vol01: np.ndarray, voxel_um_xy: float, voxel_um_z: float) -> Dict[str, float]:
    bone = vol01.astype(bool)
    if bone.sum() == 0 or (~bone).sum() == 0:
        return {"tbth_um_p90": 0.0, "tbsp_um_p90": 0.0}

    sampling = (float(voxel_um_z), float(voxel_um_xy), float(voxel_um_xy))
    dt_b = ndi.distance_transform_edt(bone, sampling=sampling)
    dt_m = ndi.distance_transform_edt(~bone, sampling=sampling)

    tbth = float(np.percentile(dt_b[bone], 90))
    tbsp = float(np.percentile(dt_m[~bone], 90))
    return {"tbth_um_p90": tbth, "tbsp_um_p90": tbsp}

def euler_conn(vol01: np.ndarray) -> Dict[str, float]:
    eul = float(euler_number(vol01.astype(bool), connectivity=3))
    conn = float(1.0 - eul)
    return {"euler": eul, "conn_proxy": conn}


# -----------------------------
# CLI + main
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v11 curvy + branch-like + plate-like trabecular generator.")

    p.add_argument("--outdir", type=str, default="data/v11_curvy")
    p.add_argument("--xy", type=int, default=256)
    p.add_argument("--z", type=int, default=160)
    p.add_argument("--seed", type=int, default=23)

    p.add_argument("--bvtv", type=float, default=0.26)
    p.add_argument("--invert-phase", type=int, default=1, help="1 often yields better trabecular-like networks.")

    p.add_argument("--voxel-um", type=float, default=10.0, help="XY voxel size (um).")
    p.add_argument("--z-um", type=float, default=10.0, help="Z voxel size (um).")

    # Optional grayscale export
    p.add_argument("--write-gray", type=int, default=0)
    return p

def main():
    args = build_parser().parse_args()
    rng = np.random.default_rng(int(args.seed))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Defaults tuned for curvature + connectivity
    fp = FieldParams(
        sigma=4.2,
        plate_strength=0.65,
        branch_strength=0.85,
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

    Z = int(args.z)
    H = W = int(args.xy)

    field = generate_field((Z, H, W), fp, rng)

    vol01, thr = threshold_to_bvtv(field, bvtv=float(args.bvtv), invert_phase=bool(int(args.invert_phase)))
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
        "threshold": float(thr),
        "invert_phase": bool(int(args.invert_phase)),
        **tbth_tbsp_p90_um(vol01, voxel_um_xy=float(args.voxel_um), voxel_um_z=float(args.z_um)),
        **euler_conn(vol01),
        "params": {"field": asdict(fp), "morph": asdict(mp), "gray": asdict(gp)},
    }

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"Saved to: {outdir}\n"
        f"BV/TV={metrics['BVTV']:.3f} | Tb.Th(p90)={metrics['tbth_um_p90']:.1f}um | "
        f"Tb.Sp(p90)={metrics['tbsp_um_p90']:.1f}um | Euler={metrics['euler']:.1f}"
    )


if __name__ == "__main__":
    main()
