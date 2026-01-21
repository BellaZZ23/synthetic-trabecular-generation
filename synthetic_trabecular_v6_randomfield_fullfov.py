#!/usr/bin/env python3
r"""
synthetic_trabecular_v6_randomfield_fullfov.py

v6 Synthetic trabecular generator (FULL FOV, no circle mask) focused on micro-CT-like
2D slices and a consistent 3D volume.

Key design (2D realism + 3D volume):
- Generate a 3D Gaussian Random Field (GRF) with anisotropic correlation lengths
- Optional multiscale mixing for more realistic heterogeneity
- Threshold to hit target BV/TV (bone volume fraction)
- Optional morphology (closing/opening) + keep largest component for purified network
- Micro-CT-like grayscale via:
    * partial volume effect (PVE) blur
    * marrow/bone intensity mapping
    * background texture + CT noise
    * optional unsharp mask (edge crispness)

Morphometrics (logged):
- BV/TV (bone volume fraction)  [target often ~0.15–0.30 depending on cohort/site] :contentReference[oaicite:1]{index=1}
- Tb.Th proxy (EDT p90 inside bone, in microns)  [often ~200–400 µm in healthy, varies by site] :contentReference[oaicite:2]{index=2}
- Tb.Sp proxy (EDT p90 in marrow, in microns)
- Conn and Conn.D proxy via Euler characteristic (Conn ≈ 1 - χ, Conn.D = Conn/TV)
- DA proxy (anisotropy proxy from covariance of bone voxel coordinates; not BoneJ MIL but useful)

Dependencies:
- numpy, pillow, tifffile
- scipy REQUIRED for GRF smoothing + morphology + EDT + Euler computations

PowerShell quick run:
python .\synthetic_trabecular_v6_randomfield_fullfov.py `
  --outdir data\v6_rf_fullfov --n-volumes 5 --xy 256 --z 160 --seed 42 `
  --bvtv 0.18 --sigma-x 3.0 --sigma-y 3.0 --sigma-z 2.0 --multiscale 1 `
  --close-iters 1 --open-iters 0 --keep-largest 1 `
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
from skimage.measure import euler_number


import numpy as np
from PIL import Image
import tifffile as tiff
from scipy import ndimage as ndi  # v6 requires scipy


# -----------------------------
# Dataclasses
# -----------------------------
@dataclass
class FieldParams:
    sigma_x: float = 3.0
    sigma_y: float = 3.0
    sigma_z: float = 2.0
    multiscale: bool = True
    sigma2_x: float = 9.0
    sigma2_y: float = 9.0
    sigma2_z: float = 6.0
    mix2: float = 0.35  # weight of coarse field
    nonlinearity: float = 0.0  # 0=off; >0 uses tanh(nonlinearity*field) to sharpen phases


@dataclass
class MorphParams:
    close_iters: int = 1
    open_iters: int = 0
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


# -----------------------------
# Helpers
# -----------------------------
def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

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

def oddify(n: int) -> int:
    n = int(max(1, n))
    return n | 1


# -----------------------------
# Random Field (3D GRF)
# -----------------------------
def generate_grf_3d(Z: int, H: int, W: int, fp: FieldParams, rng: np.random.Generator) -> np.ndarray:
    """
    3D Gaussian Random Field with anisotropic correlation lengths:
    - scipy gaussian_filter uses sigma=(z, y, x) for volume shaped (Z,H,W)
    """
    base = rng.normal(0.0, 1.0, size=(Z, H, W)).astype(np.float32)
    f1 = ndi.gaussian_filter(base, sigma=(float(fp.sigma_z), float(fp.sigma_y), float(fp.sigma_x)))

    if fp.multiscale:
        base2 = rng.normal(0.0, 1.0, size=(Z, H, W)).astype(np.float32)
        f2 = ndi.gaussian_filter(base2, sigma=(float(fp.sigma2_z), float(fp.sigma2_y), float(fp.sigma2_x)))
        f = (1.0 - float(fp.mix2)) * f1 + float(fp.mix2) * f2
    else:
        f = f1

    # normalize
    f = f - float(f.mean())
    f = f / (float(f.std()) + 1e-8)

    # optional nonlinearity to sharpen / increase plate-like transitions
    if float(fp.nonlinearity) > 0:
        k = float(fp.nonlinearity)
        f = np.tanh(k * f)

    return f.astype(np.float32)

def threshold_to_bvtv(field: np.ndarray, bvtv: float) -> Tuple[np.ndarray, float]:
    """
    Choose threshold so fraction of bone voxels ~ bvtv.
    bone = field >= thr where thr is (1-bvtv) quantile.
    """
    bvtv = float(np.clip(bvtv, 0.001, 0.999))
    thr = float(np.quantile(field, 1.0 - bvtv))
    vol01 = (field >= thr).astype(np.uint8)
    return vol01, thr


# -----------------------------
# Morphology + connectivity purification
# -----------------------------
def keep_largest_component_3d(vol01: np.ndarray) -> np.ndarray:
    bone = vol01.astype(bool)
    st = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity
    lab, n = ndi.label(bone, structure=st)
    if n <= 1:
        return vol01
    counts = np.bincount(lab.ravel())
    if len(counts) <= 1:
        return vol01
    lcc = int(np.argmax(counts[1:]) + 1)
    return (lab == lcc).astype(np.uint8)

def postprocess(vol01: np.ndarray, mp: MorphParams) -> np.ndarray:
    v = vol01.astype(bool)
    st = ndi.generate_binary_structure(3, 1)  # 6-neighborhood

    if int(mp.open_iters) > 0:
        v = ndi.binary_opening(v, structure=st, iterations=int(mp.open_iters))

    if int(mp.close_iters) > 0:
        v = ndi.binary_closing(v, structure=st, iterations=int(mp.close_iters))

    out = v.astype(np.uint8)
    if mp.keep_largest:
        out = keep_largest_component_3d(out)
    return out


# -----------------------------
# Micro-CT grayscale synthesis (full FOV)
# -----------------------------
def microct_gray(vol01: np.ndarray, ctp: CTParams, rng: np.random.Generator) -> np.ndarray:
    """
    Reconstructed-slice look:
    - PVE blur on binary volume
    - intensity mapping bone/marrow
    - background texture + CT noise
    - optional unsharp
    """
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
# Morphometric proxies (logged)
# -----------------------------
def bvtv(vol01: np.ndarray) -> float:
    return float(np.mean(vol01 > 0))

def tbth_tbsp_p90_um(vol01: np.ndarray, pixel_um: float, z_um: float) -> Dict[str, Optional[float]]:
    """
    EDT proxies converted to microns using sampling=(z, y, x).
    Tb.Th proxy: p90 of EDT within bone
    Tb.Sp proxy: p90 of EDT within marrow
    """
    bone = vol01.astype(bool)
    if bone.sum() == 0 or (~bone).sum() == 0:
        return {"tbth_um_p90": 0.0, "tbsp_um_p90": 0.0}

    sampling = (float(z_um), float(pixel_um), float(pixel_um))
    dt_bone = ndi.distance_transform_edt(bone, sampling=sampling)
    dt_marrow = ndi.distance_transform_edt(~bone, sampling=sampling)

    tbth = float(np.percentile(dt_bone[bone], 90))
    tbsp = float(np.percentile(dt_marrow[~bone], 90))
    return {"tbth_um_p90": tbth, "tbsp_um_p90": tbsp}

def conn_density_euler(vol01: np.ndarray, pixel_um: float, z_um: float) -> Dict[str, Optional[float]]:
    """
    Conn ≈ 1 - Euler characteristic.
    Conn.D = Conn / TV (mm^3).
    Conventions vary; treat as a proxy.
    """
    bone = vol01.astype(bool)
    eul = float(euler_number(bone, connectivity=3))
    conn = float(1.0 - eul)

    voxel_vol_um3 = float(pixel_um) * float(pixel_um) * float(z_um)
    tv_mm3 = (vol01.size * voxel_vol_um3) / 1e9
    conn_d = float(conn / tv_mm3) if tv_mm3 > 0 else None
    return {"euler": eul, "conn": conn, "conn_d_per_mm3": conn_d}

def da_proxy(vol01: np.ndarray, pixel_um: float, z_um: float) -> Dict[str, Optional[float]]:
    """
    DA proxy from covariance of bone voxel coordinates (physical units).
    Not BoneJ MIL, but useful for tuning anisotropy.

    Returns:
      da_proxy = 1 - (a/c) where a<=c are sqrt(eigenvalues)
      axis_ratio_c_over_a = c/a
    """
    coords = np.argwhere(vol01 > 0)  # (N,3) z,y,x
    if coords.shape[0] < 10:
        return {"da_proxy": 0.0, "axis_ratio_c_over_a": 1.0}

    # convert to physical
    z = coords[:, 0].astype(np.float64) * float(z_um)
    y = coords[:, 1].astype(np.float64) * float(pixel_um)
    x = coords[:, 2].astype(np.float64) * float(pixel_um)
    P = np.stack([x, y, z], axis=1)

    P = P - P.mean(axis=0, keepdims=True)
    C = (P.T @ P) / max(1, (P.shape[0] - 1))
    w = np.linalg.eigvalsh(C)
    w = np.clip(w, 1e-12, None)
    a, b, c = np.sqrt(np.sort(w))  # a<=b<=c

    ratio = float(c / a) if a > 0 else 1.0
    da = float(1.0 - (a / c)) if c > 0 else 0.0
    da = float(np.clip(da, 0.0, 1.0))
    return {"da_proxy": da, "axis_ratio_c_over_a": ratio}


# -----------------------------
# CLI
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="v6 synthetic trabecular generator (random field, full FOV, microCT-like slices + 3D).")
    p.add_argument("--outdir", type=str, default="data/v6_randomfield_fullfov")
    p.add_argument("--n-volumes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--xy", type=int, default=256)
    p.add_argument("--z", type=int, default=160)

    p.add_argument("--pixel-um", type=float, default=10.0, help="Voxel size in XY (microns).")
    p.add_argument("--z-step-um", type=float, default=10.0, help="Voxel size in Z (microns).")

    p.add_argument("--bvtv", type=float, default=0.18)

    # random field knobs (anisotropy control)
    p.add_argument("--sigma-x", type=float, default=3.0)
    p.add_argument("--sigma-y", type=float, default=3.0)
    p.add_argument("--sigma-z", type=float, default=2.0)
    p.add_argument("--multiscale", type=int, default=1)
    p.add_argument("--sigma2-x", type=float, default=9.0)
    p.add_argument("--sigma2-y", type=float, default=9.0)
    p.add_argument("--sigma2-z", type=float, default=6.0)
    p.add_argument("--mix2", type=float, default=0.35)
    p.add_argument("--nonlinearity", type=float, default=0.0, help="0=off; try 0.8–2.0 to sharpen phases.")

    # morphology / purification
    p.add_argument("--close-iters", type=int, default=1)
    p.add_argument("--open-iters", type=int, default=0)
    p.add_argument("--keep-largest", type=int, default=1)

    # microCT gray
    p.add_argument("--write-gray", type=int, default=1)
    p.add_argument("--pve-sigma", type=float, default=1.2)
    p.add_argument("--bone-mean", type=float, default=215.0)
    p.add_argument("--marrow-mean", type=float, default=50.0)
    p.add_argument("--ct-noise-sd", type=float, default=9.0)
    p.add_argument("--bg-texture-sd", type=float, default=4.0)
    p.add_argument("--unsharp", type=float, default=0.9)

    # exports
    p.add_argument("--export-2d", type=int, default=1)
    p.add_argument("--export-mip", type=int, default=0, help="1=export MIP PNGs too (binary + gray).")
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

    pixel_um = float(args.pixel_um)
    z_um = float(args.z_step_um)

    fp = FieldParams(
        sigma_x=float(args.sigma_x),
        sigma_y=float(args.sigma_y),
        sigma_z=float(args.sigma_z),
        multiscale=bool(int(args.multiscale)),
        sigma2_x=float(args.sigma2_x),
        sigma2_y=float(args.sigma2_y),
        sigma2_z=float(args.sigma2_z),
        mix2=float(args.mix2),
        nonlinearity=float(args.nonlinearity),
    )

    mp = MorphParams(
        close_iters=int(args.close_iters),
        open_iters=int(args.open_iters),
        keep_largest=bool(int(args.keep_largest)),
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
        "mip_png",
        "gray_mip_png",
        "xy",
        "z",
        "pixel_um",
        "z_step_um",
        "bvtv_target",
        "bvtv_actual",
        "thr_quantile",
        "sigma_x",
        "sigma_y",
        "sigma_z",
        "multiscale",
        "sigma2_x",
        "sigma2_y",
        "sigma2_z",
        "mix2",
        "nonlinearity",
        "close_iters",
        "open_iters",
        "keep_largest",
        "tbth_um_p90",
        "tbsp_um_p90",
        "euler",
        "conn",
        "conn_d_per_mm3",
        "da_proxy",
        "axis_ratio_c_over_a",
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

            # --- 1) Generate GRF ---
            field = generate_grf_3d(Z=Z, H=H, W=W, fp=fp, rng=rng)

            # --- 2) Threshold to BV/TV ---
            vol01, thr = threshold_to_bvtv(field, bvtv=float(args.bvtv))

            # --- 3) Morphology / purification ---
            vol01 = postprocess(vol01, mp=mp)

            bvtv_actual = bvtv(vol01)

            # --- 4) Grayscale microCT ---
            gray = None
            if ctp.write_gray:
                gray = microct_gray(vol01, ctp=ctp, rng=rng)

            # --- 5) Metrics ---
            ts = tbth_tbsp_p90_um(vol01, pixel_um=pixel_um, z_um=z_um)
            cd = conn_density_euler(vol01, pixel_um=pixel_um, z_um=z_um)
            da = da_proxy(vol01, pixel_um=pixel_um, z_um=z_um)

            # --- 6) Save stacks ---
            mask_tif = outdir / f"{vid}_mask.tif"
            save_stack_u8((vol01 * 255).astype(np.uint8), mask_tif)

            gray_tif_name = ""
            if gray is not None:
                gray_tif = outdir / f"{vid}_gray.tif"
                save_stack_u8(gray.astype(np.uint8), gray_tif)
                gray_tif_name = gray_tif.name

            # --- 7) Export 2D mid slice (full FOV) ---
            mid_png = ""
            gray_mid_png = ""
            mip_png = ""
            gray_mip_png = ""

            if bool(int(args.export_2d)):
                zmid = Z // 2
                bin_mid = (vol01[zmid] * 255).astype(np.uint8)
                mid_png = f"{vid}_mid.png"
                save_png(bin_mid, outdir / mid_png)

                if gray is not None:
                    gray_mid_png = f"{vid}_gray_mid.png"
                    save_png(gray[zmid].astype(np.uint8), outdir / gray_mid_png)

            # --- Optional MIP exports ---
            if bool(int(args.export_mip)):
                bin_mip = (vol01.max(axis=0) * 255).astype(np.uint8)
                mip_png = f"{vid}_mip.png"
                save_png(bin_mip, outdir / mip_png)

                if gray is not None:
                    gray_mip_png = f"{vid}_gray_mip.png"
                    save_png(gray.max(axis=0).astype(np.uint8), outdir / gray_mip_png)

            # --- JSON metadata ---
            meta = {
                "volume_id": vid,
                "files": {
                    "mask_tif": mask_tif.name,
                    "gray_tif": gray_tif_name or None,
                    "mid_png": mid_png or None,
                    "gray_mid_png": gray_mid_png or None,
                    "mip_png": mip_png or None,
                    "gray_mip_png": gray_mip_png or None,
                },
                "size": {"xy": H, "z": Z},
                "voxel_um": {"pixel_um": pixel_um, "z_step_um": z_um},
                "params": {
                    "field": asdict(fp),
                    "morph": asdict(mp),
                    "ct": asdict(ctp),
                    "bvtv_target": float(args.bvtv),
                    "threshold_value": thr,
                },
                "metrics": {
                    "bvtv_actual": bvtv_actual,
                    **ts,
                    **cd,
                    **da,
                },
                "seed": int(args.seed),
            }
            with open(outdir / f"{vid}.json", "w") as f:
                json.dump(meta, f, indent=2)

            # --- CSV row ---
            w_csv.writerow({
                "volume_id": vid,
                "mask_tif": mask_tif.name,
                "gray_tif": gray_tif_name,
                "mid_png": mid_png,
                "gray_mid_png": gray_mid_png,
                "mip_png": mip_png,
                "gray_mip_png": gray_mip_png,
                "xy": H,
                "z": Z,
                "pixel_um": pixel_um,
                "z_step_um": z_um,
                "bvtv_target": float(args.bvtv),
                "bvtv_actual": bvtv_actual,
                "thr_quantile": thr,
                "sigma_x": fp.sigma_x,
                "sigma_y": fp.sigma_y,
                "sigma_z": fp.sigma_z,
                "multiscale": int(fp.multiscale),
                "sigma2_x": fp.sigma2_x,
                "sigma2_y": fp.sigma2_y,
                "sigma2_z": fp.sigma2_z,
                "mix2": fp.mix2,
                "nonlinearity": fp.nonlinearity,
                "close_iters": mp.close_iters,
                "open_iters": mp.open_iters,
                "keep_largest": int(mp.keep_largest),
                "tbth_um_p90": ts["tbth_um_p90"],
                "tbsp_um_p90": ts["tbsp_um_p90"],
                "euler": cd["euler"],
                "conn": cd["conn"],
                "conn_d_per_mm3": cd["conn_d_per_mm3"],
                "da_proxy": da["da_proxy"],
                "axis_ratio_c_over_a": da["axis_ratio_c_over_a"],
                "pve_sigma": ctp.pve_sigma,
                "bone_mean": ctp.bone_mean,
                "marrow_mean": ctp.marrow_mean,
                "ct_noise_sd": ctp.ct_noise_sd,
                "bg_texture_sd": ctp.bg_texture_sd,
                "unsharp": ctp.unsharp,
                "seed": int(args.seed),
            })

            print(
                f"[{i+1}/{args.n_volumes}] {vid} | BV/TV={bvtv_actual:.3f} | "
                f"Tb.Th~{ts['tbth_um_p90']:.1f}um | Tb.Sp~{ts['tbsp_um_p90']:.1f}um | "
                f"Conn.D~{cd['conn_d_per_mm3'] if cd['conn_d_per_mm3'] is not None else 'NA'} | "
                f"DA~{da['da_proxy']:.3f} | sig=({fp.sigma_x},{fp.sigma_y},{fp.sigma_z})"
            )

    finally:
        f_csv.close()


if __name__ == "__main__":
    main()
