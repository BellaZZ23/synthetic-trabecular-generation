#!/usr/bin/env python3
"""
synthetic_trabecular_v2_voting3d.py

v2 synthetic trabecular generator upgraded with a 3D Binary Template Model (voting strategy)
+ Option 1 exports for PCA (2D representations).

Pipeline per 3D sample:
1) Generate k unique 2D binary lattices (plates+rods "growth-like" constructive network)
2) Stack into a 3D volume (Z = k by default)
3) Apply 3D sliding-window voting to decide trabecular bone vs space:
      S(i,j,z) = 1 if sum(window) >= tau * window_size else 0
4) Apply 3D morphological closing with alternating structuring elements to reduce artifacts
5) Output:
   - mask volume stack (TIFF)
   - optional grayscale volume stack (TIFF)
   - Option 1: export 2D PNGs (mid-slice, MIP, mean) for PCA and inspection
   - per-sample metrics JSON + CSV logs

Notes:
- This is a 3D proxy (not true biological remodeling), but it enforces:
  - continuity through voting + closing
  - avoids naive stacking discontinuity
  - allows network-like "growth" because we construct elements sequentially per lattice
"""

from __future__ import annotations
from pathlib import Path
import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from typing import Tuple, List

import numpy as np
from PIL import Image
import tifffile as tiff

try:
    from scipy import ndimage as ndi
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# -----------------------------
# Calibration defaults
# -----------------------------
PIXEL_SIZE_UM = 10.0
Z_STEP_UM     = 1.0


# -----------------------------
# Parameters
# -----------------------------
@dataclass
class LatticeParams2D:
    bv_tv: float = 0.18         # target area fraction proxy (per 2D lattice, approximate)
    pn: int = 6                 # plates per lattice
    rn: int = 12                # rods per lattice
    bv_tv_tol: float = 0.02
    max_attempts: int = 6000

    # size distributions (um)
    plate_thickness_um: Tuple[float, float] = (80.0, 160.0)
    plate_area_um2: Tuple[float, float] = (2.0e5, 6.0e5)
    rod_length_um: Tuple[float, float] = (300.0, 900.0)
    rod_diameter_um: Tuple[float, float] = (60.0, 140.0)

    # nearest-neighbor min constraints (um) - keep things realistic-ish
    nnd_pp_min_um: float = 250.0
    nnd_rr_min_um: float = 120.0

    # orientation priors (2D axial)
    plate_preferred_angle_deg: float = 0.0
    rod_preferred_angle_deg: float = 90.0
    plate_align_prob: float = 0.6
    rod_align_prob: float = 0.6
    plate_angle_spread_deg: float = 20.0
    rod_angle_spread_deg: float = 25.0


@dataclass
class Voting3DParams:
    k_lattices: int = 9                 # number of unique 2D lattices stacked (depth)
    # sliding window size (wy, wx, wz)
    win_y: int = 5
    win_x: int = 5
    win_z: int = 5
    tau: float = 0.52                   # threshold fraction: controls density/connectivity
    # 3D morphological closing enhancement
    closing_iters: int = 2
    # alternate structure: cube vs cross (avoids predictable morphology)
    alternate_structure: bool = True


@dataclass
class RenderParams:
    # soft grayscale volume (optional)
    write_gray: bool = True
    soft_sigma_px: float = 1.0
    partial_gamma: float = 0.9


# -----------------------------
# Helpers
# -----------------------------
def um_to_px(val_um: float, pixel_size_um: float = PIXEL_SIZE_UM) -> int:
    return int(round(val_um / pixel_size_um))

def clamp01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)

def rand_range(rng: np.random.Generator, lohi: Tuple[float, float]) -> float:
    lo, hi = lohi
    return float(rng.uniform(lo, hi))

def sample_angle_deg(rng: np.random.Generator, preferred: float, p_align: float, spread: float) -> float:
    if rng.random() < p_align:
        return float(rng.normal(preferred, spread))
    return float(rng.uniform(0.0, 180.0))

def min_dist(pt: Tuple[float, float], pts: List[Tuple[float, float]]) -> float:
    if not pts:
        return float("inf")
    x, y = pt
    arr = np.asarray(pts, dtype=np.float32)
    d2 = (arr[:, 0] - x) ** 2 + (arr[:, 1] - y) ** 2
    return float(np.sqrt(d2.min()))

def rasterize_thick_segment(H: int, W: int, p0, p1, radius_px: float) -> np.ndarray:
    x0, y0 = p0
    x1, y1 = p1
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    vx = x1 - x0
    vy = y1 - y0
    denom = vx * vx + vy * vy + 1e-8
    t = ((xx - x0) * vx + (yy - y0) * vy) / denom
    t = np.clip(t, 0.0, 1.0)
    projx = x0 + t * vx
    projy = y0 + t * vy
    dist = np.sqrt((xx - projx) ** 2 + (yy - projy) ** 2)
    return (dist <= radius_px).astype(np.uint8)

def rasterize_rotated_rect(H: int, W: int, center, length_px: float, width_px: float, angle_deg: float) -> np.ndarray:
    cx, cy = center
    th = math.radians(angle_deg)
    c = math.cos(th)
    s = math.sin(th)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xt = xx - cx
    yt = yy - cy
    xr =  c * xt + s * yt
    yr = -s * xt + c * yt

    halfL = 0.5 * length_px
    halfW = 0.5 * width_px
    inside = (np.abs(xr) <= halfL) & (np.abs(yr) <= halfW)
    return inside.astype(np.uint8)

def area_fraction(mask01: np.ndarray) -> float:
    return float(np.mean(mask01 > 0))

def gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr
    if _HAS_SCIPY:
        return ndi.gaussian_filter(arr, sigma=float(sigma))
    # fallback: no blur
    return arr


# -----------------------------
# 2D lattice generator (unique lattices, constructive "growth")
# -----------------------------
def generate_lattice_2d(H: int, W: int, rng: np.random.Generator, p: LatticeParams2D) -> np.ndarray:
    plate_centers: List[Tuple[float, float]] = []
    rod_midpoints: List[Tuple[float, float]] = []

    nnd_pp_min_px = max(1, um_to_px(p.nnd_pp_min_um))
    nnd_rr_min_px = max(1, um_to_px(p.nnd_rr_min_um))

    mask = np.zeros((H, W), dtype=np.uint8)

    # plates
    attempts = 0
    placed_plates = 0
    while placed_plates < p.pn and attempts < p.max_attempts:
        attempts += 1
        PA = rand_range(rng, p.plate_area_um2)
        PT = rand_range(rng, p.plate_thickness_um)
        t_px = max(1, um_to_px(PT))

        # length_px derived from area proxy
        px_um = PIXEL_SIZE_UM
        length_px = max(8.0, PA / (max(1.0, t_px) * (px_um ** 2)))

        ang = sample_angle_deg(rng, p.plate_preferred_angle_deg, p.plate_align_prob, p.plate_angle_spread_deg)
        cx = float(rng.uniform(0.15 * W, 0.85 * W))
        cy = float(rng.uniform(0.15 * H, 0.85 * H))

        if min_dist((cx, cy), plate_centers) < nnd_pp_min_px:
            continue

        pm = rasterize_rotated_rect(H, W, (cx, cy), length_px=length_px, width_px=float(t_px), angle_deg=ang)
        mask = np.maximum(mask, pm)
        plate_centers.append((cx, cy))
        placed_plates += 1

    # rods (prefer connecting to existing bone to encourage network connectivity)
    attempts = 0
    placed_rods = 0
    ys_b, xs_b = np.where(mask > 0)
    while placed_rods < p.rn and attempts < p.max_attempts:
        attempts += 1
        RL = rand_range(rng, p.rod_length_um)
        RD = rand_range(rng, p.rod_diameter_um)
        length_px = max(8.0, float(um_to_px(RL)))
        radius_px = max(1.0, float(um_to_px(0.5 * RD)))

        ang = sample_angle_deg(rng, p.rod_preferred_angle_deg, p.rod_align_prob, p.rod_angle_spread_deg)
        th = math.radians(ang)
        dx = math.cos(th)
        dy = math.sin(th)

        # anchor on existing bone sometimes (growth-like)
        if len(xs_b) > 0 and rng.random() < 0.65:
            idx = int(rng.integers(0, len(xs_b)))
            x0 = float(xs_b[idx])
            y0 = float(ys_b[idx])
        else:
            x0 = float(rng.uniform(0.1 * W, 0.9 * W))
            y0 = float(rng.uniform(0.1 * H, 0.9 * H))

        x1 = x0 + length_px * dx
        y1 = y0 + length_px * dy
        if x1 < 2 or x1 > (W - 3) or y1 < 2 or y1 > (H - 3):
            x1 = x0 - length_px * dx
            y1 = y0 - length_px * dy
        if x1 < 2 or x1 > (W - 3) or y1 < 2 or y1 > (H - 3):
            continue

        mx = 0.5 * (x0 + x1)
        my = 0.5 * (y0 + y1)
        if min_dist((mx, my), rod_midpoints) < nnd_rr_min_px:
            continue

        rm = rasterize_thick_segment(H, W, (x0, y0), (x1, y1), radius_px=radius_px)
        mask = np.maximum(mask, rm)
        rod_midpoints.append((mx, my))
        placed_rods += 1

        ys_b, xs_b = np.where(mask > 0)  # update anchors

    # tune BV/TV (area fraction) roughly with dilation/erosion if available
    bv = area_fraction(mask)
    if _HAS_SCIPY:
        st = ndi.generate_binary_structure(2, 1)
        steps = 0
        while abs(bv - p.bv_tv) > p.bv_tv_tol and steps < 10:
            steps += 1
            if bv < p.bv_tv:
                mask = ndi.binary_dilation(mask > 0, structure=st, iterations=1).astype(np.uint8)
            else:
                mask = ndi.binary_erosion(mask > 0, structure=st, iterations=1).astype(np.uint8)
            bv = area_fraction(mask)

    return mask.astype(np.uint8)


# -----------------------------
# 3D voting model (Binary Template Model)
# -----------------------------
def vote_3d(stack01: np.ndarray, v: Voting3DParams) -> np.ndarray:
    """
    stack01: uint8 (Z,H,W)
    Applies 3D sliding-window voting using convolution.
    """
    Z, H, W = stack01.shape
    wy, wx, wz = int(v.win_y), int(v.win_x), int(v.win_z)
    wy = max(1, wy | 1)  # force odd
    wx = max(1, wx | 1)
    wz = max(1, wz | 1)

    # kernel in (Z,Y,X) order
    kernel = np.ones((wz, wy, wx), dtype=np.uint8)
    window_size = int(kernel.size)

    if _HAS_SCIPY:
        votes = ndi.convolve(stack01.astype(np.uint16), kernel.astype(np.uint16), mode="constant", cval=0)
    else:
        # fallback: naive (slower) using padding and summation
        padz, pady, padx = wz // 2, wy // 2, wx // 2
        padded = np.pad(stack01.astype(np.uint8), ((padz, padz), (pady, pady), (padx, padx)), mode="constant")
        votes = np.zeros_like(stack01, dtype=np.uint16)
        for dz in range(wz):
            for dy in range(wy):
                for dx in range(wx):
                    votes += padded[dz:dz+Z, dy:dy+H, dx:dx+W]

    thr = int(math.ceil(float(v.tau) * window_size))
    out = (votes >= thr).astype(np.uint8)
    return out


def closing_3d(vol01: np.ndarray, iters: int, alternate: bool) -> np.ndarray:
    if not _HAS_SCIPY or iters <= 0:
        return vol01.astype(np.uint8)

    vol = (vol01 > 0)
    # two structuring elements: cube and cross
    cube = np.ones((3, 3, 3), dtype=bool)
    cross = ndi.generate_binary_structure(3, 1)  # 6-neighborhood

    for i in range(int(iters)):
        st = cross if (alternate and (i % 2 == 0)) else cube
        vol = ndi.binary_closing(vol, structure=st, iterations=1)

    return vol.astype(np.uint8)


# -----------------------------
# Outputs
# -----------------------------
def save_stack_u8(stack_u8: np.ndarray, out_path: Path, z_step_um: float):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(
        out_path,
        stack_u8,
        imagej=True,
        metadata={"unit": "micron", "spacing": float(z_step_um)},
        dtype=np.uint8,
    )

def save_png(arr_u8: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_u8, mode="L").save(out_path)

def init_csv(path: Path, fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if not exists:
        w.writeheader()
    return f, w


# -----------------------------
# Main
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="v2 trabecular generator with 3D voting template model.")
    p.add_argument("--outdir", type=str, default="data/v2_voting3d")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--n-volumes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    # 2D lattice params
    p.add_argument("--bv-tv", type=float, default=0.18)
    p.add_argument("--pn", type=int, default=6)
    p.add_argument("--rn", type=int, default=12)

    # voting params
    p.add_argument("--k", type=int, default=9, help="Number of unique 2D lattices stacked (depth).")
    p.add_argument("--win-x", type=int, default=5)
    p.add_argument("--win-y", type=int, default=5)
    p.add_argument("--win-z", type=int, default=5)
    p.add_argument("--tau", type=float, default=0.52, help="Voting threshold fraction (density control).")
    p.add_argument("--closing-iters", type=int, default=2)
    p.add_argument("--no-alt-close", action="store_true")

    # render
    p.add_argument("--z-step-um", type=float, default=Z_STEP_UM)
    p.add_argument("--write-gray", action="store_true")
    p.add_argument("--soft-sigma-px", type=float, default=1.0)

    # Option 1: 3D -> 2D exports for PCA
    p.add_argument("--export-2d", action="store_true",
                   help="Export 2D PNGs (mid-slice, MIP, mean) for PCA/inspection.")
    p.add_argument("--export-2d-mode", type=str, default="all",
                   choices=["all", "mid", "mip", "mean"],
                   help="Which 2D representation(s) to export.")

    return p


def main():
    args = build_parser().parse_args()
    rng = np.random.default_rng(int(args.seed))

    H = W = int(args.size)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lat_p = LatticeParams2D(
        bv_tv=float(args.bv_tv),
        pn=int(args.pn),
        rn=int(args.rn),
    )
    vot_p = Voting3DParams(
        k_lattices=int(args.k),
        win_x=int(args.win_x),
        win_y=int(args.win_y),
        win_z=int(args.win_z),
        tau=float(args.tau),
        closing_iters=int(args.closing_iters),
        alternate_structure=(not bool(args.no_alt_close)),
    )
    ren_p = RenderParams(
        write_gray=bool(args.write_gray),
        soft_sigma_px=float(args.soft_sigma_px),
    )

    images_csv = outdir / "volumes.csv"
    f_img, w_img = init_csv(images_csv, fieldnames=[
        "volume_id",
        "mask_tif",
        "gray_tif",
        "mid_png",
        "mip_png",
        "mean_png",
        "gray_mid_png",
        "gray_mip_png",
        "H", "W", "Z",
        "pixel_size_um",
        "z_step_um",
        "bv_tv_3d",
        "seed",
        "k",
        "win_x", "win_y", "win_z",
        "tau",
        "closing_iters",
        "alternate_close",
        "pn", "rn", "bv_tv_2d_target",
        "export_2d",
        "export_2d_mode"
    ])

    try:
        for vi in range(int(args.n_volumes)):
            volume_id = f"vol_{vi:05d}"

            # 1) generate k unique 2D lattices
            k = int(vot_p.k_lattices)
            lattices = []
            lattice_seeds = []
            for zi in range(k):
                sub_seed = int(rng.integers(0, 2**31 - 1))
                lattice_seeds.append(sub_seed)
                sub_rng = np.random.default_rng(sub_seed)
                latt = generate_lattice_2d(H, W, sub_rng, lat_p)
                lattices.append(latt)

            stack01 = np.stack(lattices, axis=0).astype(np.uint8)  # (Z,H,W)

            # 2) voting strategy (3D sliding window)
            voted01 = vote_3d(stack01, vot_p)

            # 3) alternating 3D morphological closing
            voted01 = closing_3d(voted01, iters=vot_p.closing_iters, alternate=vot_p.alternate_structure)

            # ---- Option 1: export 2D representations (binary) ----
            mid_png = mip_png = mean_png = ""
            if args.export_2d:
                zmid = voted01.shape[0] // 2
                mid_slice = (voted01[zmid] * 255).astype(np.uint8)
                mip_xy = (voted01.max(axis=0) * 255).astype(np.uint8)
                mean_xy = np.clip(voted01.mean(axis=0) * 255.0, 0, 255).astype(np.uint8)

                mode = args.export_2d_mode
                if mode in ("all", "mid"):
                    mid_png = f"{volume_id}_mid.png"
                    save_png(mid_slice, outdir / mid_png)
                if mode in ("all", "mip"):
                    mip_png = f"{volume_id}_mip.png"
                    save_png(mip_xy, outdir / mip_png)
                if mode in ("all", "mean"):
                    mean_png = f"{volume_id}_mean.png"
                    save_png(mean_xy, outdir / mean_png)

            # 4) grayscale volume (optional)
            gray_stack = None
            gray_mid_png = gray_mip_png = ""
            if ren_p.write_gray:
                soft = gaussian_blur(voted01.astype(np.float32), sigma=float(ren_p.soft_sigma_px))
                soft = clamp01(soft) ** float(ren_p.partial_gamma)
                gray_stack = (255.0 * soft).astype(np.uint8)

                # ---- Option 1: export 2D representations (grayscale) ----
                if args.export_2d:
                    zmid = gray_stack.shape[0] // 2
                    gray_mid_png = f"{volume_id}_gray_mid.png"
                    gray_mip_png = f"{volume_id}_gray_mip.png"
                    save_png(gray_stack[zmid].astype(np.uint8), outdir / gray_mid_png)
                    save_png(gray_stack.max(axis=0).astype(np.uint8), outdir / gray_mip_png)

            # metrics
            bv_3d = float(np.mean(voted01 > 0))

            # save outputs (3D stacks)
            mask_path = outdir / f"{volume_id}_mask.tif"
            gray_path = outdir / f"{volume_id}_gray.tif" if ren_p.write_gray else None
            meta_path = outdir / f"{volume_id}.json"

            save_stack_u8((voted01 * 255).astype(np.uint8), mask_path, z_step_um=float(args.z_step_um))
            if ren_p.write_gray and gray_stack is not None:
                save_stack_u8(gray_stack, gray_path, z_step_um=float(args.z_step_um))

            # metadata JSON with explicit params + export listing
            meta = {
                "volume_id": volume_id,
                "files": {
                    "mask_tif": mask_path.name,
                    "gray_tif": gray_path.name if gray_path else None,
                },
                "exports_2d": {
                    "enabled": bool(args.export_2d),
                    "mode": str(args.export_2d_mode),
                    "mid_png": mid_png or None,
                    "mip_png": mip_png or None,
                    "mean_png": mean_png or None,
                    "gray_mid_png": gray_mid_png or None,
                    "gray_mip_png": gray_mip_png or None,
                },
                "globals": {
                    "pixel_size_um": float(PIXEL_SIZE_UM),
                    "z_step_um": float(args.z_step_um),
                    "H": H, "W": W, "Z": int(voted01.shape[0]),
                    "seed": int(args.seed),
                },
                "lattice_2d_params": asdict(lat_p),
                "voting_3d_params": asdict(vot_p),
                "render_params": asdict(ren_p),
                "lattice_seeds": lattice_seeds,
                "metrics": {
                    "bv_tv_3d": bv_3d,
                    "note": "BV/TV here is a 3D occupancy fraction of the voted binary template."
                }
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            # CSV row
            w_img.writerow({
                "volume_id": volume_id,
                "mask_tif": mask_path.name,
                "gray_tif": gray_path.name if gray_path else "",
                "mid_png": mid_png,
                "mip_png": mip_png,
                "mean_png": mean_png,
                "gray_mid_png": gray_mid_png,
                "gray_mip_png": gray_mip_png,
                "H": H, "W": W, "Z": int(voted01.shape[0]),
                "pixel_size_um": float(PIXEL_SIZE_UM),
                "z_step_um": float(args.z_step_um),
                "bv_tv_3d": bv_3d,
                "seed": int(args.seed),
                "k": int(vot_p.k_lattices),
                "win_x": int(vot_p.win_x),
                "win_y": int(vot_p.win_y),
                "win_z": int(vot_p.win_z),
                "tau": float(vot_p.tau),
                "closing_iters": int(vot_p.closing_iters),
                "alternate_close": bool(vot_p.alternate_structure),
                "pn": int(lat_p.pn),
                "rn": int(lat_p.rn),
                "bv_tv_2d_target": float(lat_p.bv_tv),
                "export_2d": bool(args.export_2d),
                "export_2d_mode": str(args.export_2d_mode),
            })

            print(f"[{vi+1}/{args.n_volumes}] Saved {volume_id} | 3D BV/TV={bv_3d:.3f}")

    finally:
        f_img.close()

    print(f"\nDone. Written to: {outdir}")
    if not _HAS_SCIPY:
        print("Note: scipy not available → voting uses slow fallback and 3D closing is skipped. "
              "Install scipy for best results: pip install scipy")


if __name__ == "__main__":
    main()
