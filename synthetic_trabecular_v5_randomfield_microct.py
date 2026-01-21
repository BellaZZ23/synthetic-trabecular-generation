#!/usr/bin/env python3
r"""
synthetic_trabecular_v5_randomfield_microct.py

Goal: generate synthetic trabecular bone patterns that look closer to *thresholded micro-CT slices*
(labyrinth / sheet-like trabecular network inside a circular FOV), while also producing a full 3D volume.

Core idea (2D-first, 3D-consistent):
- Generate a 3D Gaussian random field (GRF) with controllable correlation length (sigma_xy, sigma_z)
- Optionally combine two GRFs (multiscale) to get more realistic heterogeneity
- Threshold the field to hit a target BV/TV (bone volume fraction)
- Optional cleanup (closing) and optional "keep largest component"
- Create micro-CT-like grayscale via partial-volume effect (PVE) + intensity mapping + noise + optional unsharp
- Export:
    - 3D mask TIFF (0/255)
    - 3D gray TIFF (0–255)
    - 2D mid-slice PNGs (binary + gray), with optional circular FOV mask
    - JSON + CSV logs

This generator is better suited than “rods + voting” when you want *foam/plate-like labyrinth* geometry
and 2D slices similar to typical micro-CT thresholding figures.

Dependencies:
- numpy, pillow, tifffile
- scipy recommended (Gaussian filtering + morphology + labeling)

PowerShell quick test:
python .\synthetic_trabecular_v5_randomfield_microct.py `
  --outdir data\v5_rf_test --n-volumes 3 --xy 256 --z 160 --seed 42 `
  --bvtv 0.18 --sigma-xy 3.2 --sigma-z 2.0 --multiscale 1 `
  --fov-circle 1 --fov-radius-frac 0.48 `
  --close-iters 1 --keep-largest 1 `
  --write-gray 1 --pve-sigma 1.2 --bone-mean 215 --marrow-mean 50 --ct-noise-sd 9 --bg-texture-sd 4 --unsharp 0.9 `
  --export-2d 1
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import tifffile as tiff

try:
    from scipy import ndimage as ndi
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# -----------------------------
# Params
# -----------------------------
@dataclass
class FieldParams:
    sigma_xy: float = 3.2
    sigma_z: float = 2.0
    multiscale: bool = True
    sigma2_xy: float = 8.0
    sigma2_z: float = 5.0
    mix2: float = 0.35  # weight of second (coarser) field


@dataclass
class MorphParams:
    close_iters: int = 1
    keep_largest: bool = True


@dataclass
class CTParams:
    write_gray: bool = True
    pve_sigma: float = 1.2
    bone_mean: float = 215.0
    marrow_mean: float = 50.0
    ct_noise_sd: float = 9.0
    bg_texture_sd: float = 4.0
    unsharp: float = 0.9


@dataclass
class FOVParams:
    fov_circle: bool = True
    fov_radius_frac: float = 0.48  # fraction of min(H,W)/2


# -----------------------------
# Helpers
# -----------------------------
def save_png(arr_u8: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_u8.astype(np.uint8), mode="L").save(out_path)

def save_stack_u8(stack_u8: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(out_path, stack_u8.astype(np.uint8), imagej=True, dtype=np.uint8)

def init_csv(path: Path, fieldnames) -> tuple:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if not exists:
        w.writeheader()
    return f, w

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def circle_mask(H: int, W: int, radius_frac: float) -> np.ndarray:
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    r = float(radius_frac) * (min(H, W) / 2.0)
    yy, xx = np.ogrid[0:H, 0:W]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r * r)

def apply_fov_2d(img: np.ndarray, fov: np.ndarray, outside_value: float) -> np.ndarray:
    out = img.copy()
    out[~fov] = outside_value
    return out


# -----------------------------
# Random field generation (3D)
# -----------------------------
def generate_random_field_3d(
    Z: int,
    H: int,
    W: int,
    fp: FieldParams,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Create 3D Gaussian random field with anisotropic correlation lengths.
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for the random-field generator (Gaussian filtering).")

    base = rng.normal(0.0, 1.0, size=(Z, H, W)).astype(np.float32)
    f1 = ndi.gaussian_filter(base, sigma=(float(fp.sigma_z), float(fp.sigma_xy), float(fp.sigma_xy)))

    if not fp.multiscale:
        f = f1
    else:
        base2 = rng.normal(0.0, 1.0, size=(Z, H, W)).astype(np.float32)
        f2 = ndi.gaussian_filter(base2, sigma=(float(fp.sigma2_z), float(fp.sigma2_xy), float(fp.sigma2_xy)))
        f = (1.0 - float(fp.mix2)) * f1 + float(fp.mix2) * f2

    # normalize to mean 0, std 1
    f = f - float(f.mean())
    s = float(f.std()) + 1e-8
    f = f / s
    return f.astype(np.float32)

def threshold_to_bvtv(field: np.ndarray, bvtv: float) -> np.ndarray:
    """
    Threshold a scalar field to achieve target BV/TV (bone fraction).
    Bone = field >= thr.
    """
    bvtv = float(np.clip(bvtv, 0.001, 0.999))
    # Want fraction of ones = bvtv => threshold at (1 - bvtv) quantile
    thr = float(np.quantile(field, 1.0 - bvtv))
    return (field >= thr).astype(np.uint8)


# -----------------------------
# Morphology / connectivity control
# -----------------------------
def postprocess_binary(vol01: np.ndarray, mp: MorphParams) -> np.ndarray:
    if not _HAS_SCIPY:
        return vol01

    v = vol01.astype(bool)

    # mild closing to smooth and connect thin gaps
    if int(mp.close_iters) > 0:
        st = ndi.generate_binary_structure(3, 1)  # 6-neighborhood
        v = ndi.binary_closing(v, structure=st, iterations=int(mp.close_iters))

    if mp.keep_largest:
        st = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity
        lab, n = ndi.label(v, structure=st)
        if n > 0:
            counts = np.bincount(lab.ravel())
            if len(counts) > 1:
                lcc = int(np.argmax(counts[1:]) + 1)
                v = (lab == lcc)

    return v.astype(np.uint8)


# -----------------------------
# Micro-CT slice appearance
# -----------------------------
def microct_gray_from_binary(
    vol01: np.ndarray,
    ctp: CTParams,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Reconstructed-slice look:
    - PVE blur on the binary volume
    - intensity mapping: marrow -> low, bone -> high
    - add background texture + CT noise
    - optional unsharp mask
    """
    if not _HAS_SCIPY:
        # simple fallback
        x = vol01.astype(np.float32)
        return (ctp.marrow_mean + x * (ctp.bone_mean - ctp.marrow_mean)).clip(0, 255).astype(np.uint8)

    x = vol01.astype(np.float32)
    if float(ctp.pve_sigma) > 0:
        x = ndi.gaussian_filter(x, sigma=float(ctp.pve_sigma))
    x = clamp01(x)

    gray = float(ctp.marrow_mean) + x * (float(ctp.bone_mean) - float(ctp.marrow_mean))

    if float(ctp.bg_texture_sd) > 0:
        gray += rng.normal(0.0, float(ctp.bg_texture_sd), size=gray.shape).astype(np.float32)

    if float(ctp.ct_noise_sd) > 0:
        gray += rng.normal(0.0, float(ctp.ct_noise_sd), size=gray.shape).astype(np.float32)

    if float(ctp.unsharp) > 0:
        blurred = ndi.gaussian_filter(gray, sigma=max(0.5, float(ctp.pve_sigma)))
        gray = gray + float(ctp.unsharp) * (gray - blurred)

    return np.clip(gray, 0.0, 255.0).astype(np.uint8)


# -----------------------------
# CLI
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="v5 random-field synthetic trabecular generator (2D-like microCT slices + 3D volume).")
    p.add_argument("--outdir", type=str, default="data/v5_randomfield")
    p.add_argument("--n-volumes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    # volume size
    p.add_argument("--xy", type=int, default=256, help="XY size (H=W=xy)")
    p.add_argument("--z", type=int, default=160, help="Z depth (# slices)")

    # target bone volume fraction
    p.add_argument("--bvtv", type=float, default=0.18, help="Target BV/TV (bone fraction).")

    # random field correlation lengths
    p.add_argument("--sigma-xy", type=float, default=3.2)
    p.add_argument("--sigma-z", type=float, default=2.0)
    p.add_argument("--multiscale", type=int, default=1, help="1=use multiscale GRF mix, 0=single scale.")
    p.add_argument("--sigma2-xy", type=float, default=8.0)
    p.add_argument("--sigma2-z", type=float, default=5.0)
    p.add_argument("--mix2", type=float, default=0.35)

    # morphology
    p.add_argument("--close-iters", type=int, default=1)
    p.add_argument("--keep-largest", type=int, default=1)

    # circular FOV (2D look like microCT)
    p.add_argument("--fov-circle", type=int, default=1)
    p.add_argument("--fov-radius-frac", type=float, default=0.48)

    # grayscale microCT
    p.add_argument("--write-gray", type=int, default=1)
    p.add_argument("--pve-sigma", type=float, default=1.2)
    p.add_argument("--bone-mean", type=float, default=215.0)
    p.add_argument("--marrow-mean", type=float, default=50.0)
    p.add_argument("--ct-noise-sd", type=float, default=9.0)
    p.add_argument("--bg-texture-sd", type=float, default=4.0)
    p.add_argument("--unsharp", type=float, default=0.9)

    # exports
    p.add_argument("--export-2d", type=int, default=1, help="1=export mid slice PNGs, 0=skip")
    return p


# -----------------------------
# Main
# -----------------------------
def main():
    args = build_parser().parse_args()
    rng = np.random.default_rng(int(args.seed))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    H = W = int(args.xy)
    Z = int(args.z)

    fp = FieldParams(
        sigma_xy=float(args.sigma_xy),
        sigma_z=float(args.sigma_z),
        multiscale=bool(int(args.multiscale)),
        sigma2_xy=float(args.sigma2_xy),
        sigma2_z=float(args.sigma2_z),
        mix2=float(args.mix2),
    )

    mp = MorphParams(
        close_iters=int(args.close_iters),
        keep_largest=bool(int(args.keep_largest)),
    )

    fovp = FOVParams(
        fov_circle=bool(int(args.fov_circle)),
        fov_radius_frac=float(args.fov_radius_frac),
    )

    ctp = CTParams(
        write_gray=bool(int(args.write_gray)),
        pve_sigma=float(args.pve_sigma),
        bone_mean=float(args.bone_mean),
        marrow_mean=float(args.marrow_mean),
        ct_noise_sd=float(args.ct_noise_sd),
        bg_texture_sd=float(args.bg_texture_sd),
        unsharp=float(args.unsharp),
    )

    csv_path = outdir / "volumes.csv"
    fields = [
        "volume_id",
        "mask_tif",
        "gray_tif",
        "mid_png",
        "gray_mid_png",
        "xy",
        "z",
        "bvtv_target",
        "bvtv_actual",
        "sigma_xy",
        "sigma_z",
        "multiscale",
        "sigma2_xy",
        "sigma2_z",
        "mix2",
        "close_iters",
        "keep_largest",
        "fov_circle",
        "fov_radius_frac",
        "pve_sigma",
        "bone_mean",
        "marrow_mean",
        "ct_noise_sd",
        "bg_texture_sd",
        "unsharp",
        "seed",
    ]
    f_csv, w_csv = init_csv(csv_path, fields)

    try:
        for i in range(int(args.n_volumes)):
            vid = f"vol_{i:05d}"

            # --- Generate 3D random field & threshold ---
            field = generate_random_field_3d(Z=Z, H=H, W=W, fp=fp, rng=rng)
            vol01 = threshold_to_bvtv(field, bvtv=float(args.bvtv))

            # --- Morph cleanup / optional largest component ---
            vol01 = postprocess_binary(vol01, mp=mp)

            bvtv_actual = float(np.mean(vol01 > 0))

            # --- Grayscale microCT ---
            gray = None
            if ctp.write_gray:
                gray = microct_gray_from_binary(vol01, ctp=ctp, rng=rng)

            # --- Save 3D stacks ---
            mask_tif = outdir / f"{vid}_mask.tif"
            save_stack_u8((vol01 * 255).astype(np.uint8), mask_tif)

            gray_tif_name = ""
            if gray is not None:
                gray_tif = outdir / f"{vid}_gray.tif"
                save_stack_u8(gray.astype(np.uint8), gray_tif)
                gray_tif_name = gray_tif.name

            # --- Export mid-slice 2D PNGs (the “paper-style” view) ---
            mid_png_name = ""
            gray_mid_png_name = ""

            if bool(int(args.export_2d)):
                zmid = Z // 2
                bin_mid = (vol01[zmid] * 255).astype(np.uint8)
                gray_mid = gray[zmid].astype(np.uint8) if gray is not None else None

                if fovp.fov_circle:
                    fov = circle_mask(H, W, radius_frac=float(fovp.fov_radius_frac))
                    bin_mid = apply_fov_2d(bin_mid, fov, outside_value=0)
                    if gray_mid is not None:
                        gray_mid = apply_fov_2d(gray_mid, fov, outside_value=0)

                mid_png_name = f"{vid}_mid.png"
                save_png(bin_mid, outdir / mid_png_name)

                if gray_mid is not None:
                    gray_mid_png_name = f"{vid}_gray_mid.png"
                    save_png(gray_mid, outdir / gray_mid_png_name)

            # --- JSON metadata ---
            meta = {
                "volume_id": vid,
                "files": {
                    "mask_tif": mask_tif.name,
                    "gray_tif": gray_tif_name or None,
                    "mid_png": mid_png_name or None,
                    "gray_mid_png": gray_mid_png_name or None,
                },
                "size": {"xy": H, "z": Z},
                "params": {
                    "field": asdict(fp),
                    "morph": asdict(mp),
                    "fov": asdict(fovp),
                    "ct": asdict(ctp),
                    "bvtv_target": float(args.bvtv),
                },
                "metrics": {"bvtv_actual": bvtv_actual},
                "seed": int(args.seed),
                "scipy_available": bool(_HAS_SCIPY),
            }
            with open(outdir / f"{vid}.json", "w") as f:
                json.dump(meta, f, indent=2)

            # --- CSV row ---
            w_csv.writerow({
                "volume_id": vid,
                "mask_tif": mask_tif.name,
                "gray_tif": gray_tif_name,
                "mid_png": mid_png_name,
                "gray_mid_png": gray_mid_png_name,
                "xy": H,
                "z": Z,
                "bvtv_target": float(args.bvtv),
                "bvtv_actual": bvtv_actual,
                "sigma_xy": fp.sigma_xy,
                "sigma_z": fp.sigma_z,
                "multiscale": int(fp.multiscale),
                "sigma2_xy": fp.sigma2_xy,
                "sigma2_z": fp.sigma2_z,
                "mix2": fp.mix2,
                "close_iters": mp.close_iters,
                "keep_largest": int(mp.keep_largest),
                "fov_circle": int(fovp.fov_circle),
                "fov_radius_frac": fovp.fov_radius_frac,
                "pve_sigma": ctp.pve_sigma,
                "bone_mean": ctp.bone_mean,
                "marrow_mean": ctp.marrow_mean,
                "ct_noise_sd": ctp.ct_noise_sd,
                "bg_texture_sd": ctp.bg_texture_sd,
                "unsharp": ctp.unsharp,
                "seed": int(args.seed),
            })

            print(f"[{i+1}/{args.n_volumes}] {vid} | BV/TV={bvtv_actual:.3f} | sig_xy={fp.sigma_xy} sig_z={fp.sigma_z} | multiscale={fp.multiscale}")

    finally:
        f_csv.close()

    if not _HAS_SCIPY:
        print("WARNING: scipy not available; this generator requires scipy for GRF smoothing.")


if __name__ == "__main__":
    main()
