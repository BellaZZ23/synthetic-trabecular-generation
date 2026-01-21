#!/usr/bin/env python3
"""
synthetic_trabecular_v4_microct_curvature.py

v4.1: Connectivity-first trabecular generator with curvature + micro-CT-like grayscale.

This script addresses:
1) Curvature control:
   - Curved rods in 2D via polyline growth
   - Curved 3D bridging paths (optional) to enforce connectivity with less "straight-line" look

2) Micro-CT-like grayscale:
   - blur mode (legacy)
   - beerlambert mode (slab Beer–Lambert + Gaussian noise)
   - microct mode (reconstructed-slice look): partial volume effect (PVE) + marrow/bone intensity mapping
     + CT noise + background texture + optional unsharp mask

3) Morphometric acceptance targets (optional):
   - LCC fraction (connectivity)
   - BV/TV (3D)
   - Thickness and spacing EDT proxies (converted to microns using voxel sizes)
   - Retry loop keeps the best candidate, accepts early if targets satisfied

4) Connectivity density proxy:
   - Euler characteristic -> Conn = 1 - Euler, Conn.D = Conn / TV (mm^3). Conventions vary.

Outputs per volume:
- *_mask.tif (binary volume)
- *_gray.tif (grayscale volume, if enabled)
- optional mid/mip/mean PNGs for quick inspection
- per-volume JSON metadata
- volumes.csv summary

Dependencies:
- numpy, pillow, tifffile
- scipy recommended (labeling, morphology, EDT, euler number, convolution)

PowerShell quick test:
python .\synthetic_trabecular_v4_microct_curvature.py `
  --outdir data\v4_microct_like_c --n-volumes 3 --size 256 --seed 42 `
  --pn 0 --rn 35 --thin-iters 1 `
  --curve-prob 0.9 --curve-amp-px 8 --curve-drift-deg 8 --curve-segments 8 `
  --min-lcc-frac 0.95 --max-tries 8 --auto-tune --bridge-3d --max-bridges 2 --bridge-radius 1 --bridge-smooth 1 `
  --write-gray --gray-mode microct --pve-sigma 1.1 --bone-mean 215 --marrow-mean 50 --ct-noise-sd 10 --bg-texture-sd 5 --unsharp 0.8 `
  --export-2d
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any

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
PIXEL_SIZE_UM_DEFAULT = 10.0
Z_STEP_UM_DEFAULT = 1.0


# -----------------------------
# Parameters
# -----------------------------
@dataclass
class LatticeParams2D:
    bv_tv: float = 0.18
    pn: int = 6
    rn: int = 12
    bv_tv_tol: float = 0.02
    max_attempts: int = 6000

    # size distributions (um)
    plate_thickness_um: Tuple[float, float] = (80.0, 160.0)
    plate_area_um2: Tuple[float, float] = (2.0e5, 6.0e5)
    rod_length_um: Tuple[float, float] = (300.0, 900.0)
    rod_diameter_um: Tuple[float, float] = (60.0, 140.0)

    # nearest-neighbor min constraints (um)
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
    k_lattices: int = 9
    win_y: int = 5
    win_x: int = 5
    win_z: int = 5
    tau: float = 0.52
    closing_iters: int = 2
    alternate_structure: bool = True
    z_continuity_bias: bool = True


@dataclass
class CurvatureParams:
    curve_prob: float = 0.7
    curve_amp_px: int = 12
    curve_drift_deg: float = 12.0
    curve_segments: int = 6

    bridge_curve_amp_px: int = 8
    bridge_curve_segments: int = 10


# -----------------------------
# Helpers
# -----------------------------
def um_to_px(val_um: float, pixel_um: float) -> int:
    return int(round(val_um / pixel_um))

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

def area_fraction(mask01: np.ndarray) -> float:
    return float(np.mean(mask01 > 0))

def oddify(n: int) -> int:
    n = int(max(1, n))
    return n | 1

def gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr
    if _HAS_SCIPY:
        return ndi.gaussian_filter(arr, sigma=float(sigma))
    return arr

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
    xr = c * xt + s * yt
    yr = -s * xt + c * yt

    halfL = 0.5 * length_px
    halfW = 0.5 * width_px
    inside = (np.abs(xr) <= halfL) & (np.abs(yr) <= halfW)
    return inside.astype(np.uint8)

def rasterize_polyline_thick(H: int, W: int, pts: List[Tuple[float, float]], radius_px: float) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.uint8)
    for a, b in zip(pts[:-1], pts[1:]):
        out = np.maximum(out, rasterize_thick_segment(H, W, a, b, radius_px=radius_px))
    return out

def save_stack_u8(stack_u8: np.ndarray, out_path: Path, z_step_um: float):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(
        out_path,
        stack_u8.astype(np.uint8),
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
# Metrics
# -----------------------------
def connectivity_metrics_3d(vol01: np.ndarray) -> Dict[str, Optional[float]]:
    if not _HAS_SCIPY:
        return {"cc_count": None, "lcc_frac": None}

    bone = vol01.astype(bool)
    st = np.ones((3, 3, 3), dtype=bool)  # 26-neighborhood
    lab, n = ndi.label(bone, structure=st)
    bone_vox = int(bone.sum())

    if bone_vox == 0:
        return {"cc_count": float(n), "lcc_frac": 0.0}

    counts = np.bincount(lab.ravel())
    if len(counts) <= 1:
        return {"cc_count": float(n), "lcc_frac": 1.0}

    lcc = int(counts[1:].max())
    return {"cc_count": float(n), "lcc_frac": float(lcc / bone_vox)}

def thickness_spacing_um(vol01: np.ndarray, pixel_um: float, z_um: float) -> Dict[str, Optional[float]]:
    if not _HAS_SCIPY:
        return {"thickness_um_p90": None, "spacing_um_p90": None}

    bone = vol01.astype(bool)
    if bone.sum() == 0 or (~bone).sum() == 0:
        return {"thickness_um_p90": 0.0, "spacing_um_p90": 0.0}

    sampling = (float(z_um), float(pixel_um), float(pixel_um))
    dt_bone = ndi.distance_transform_edt(bone, sampling=sampling)
    dt_space = ndi.distance_transform_edt(~bone, sampling=sampling)

    thick = float(np.percentile(dt_bone[bone], 90))
    space = float(np.percentile(dt_space[~bone], 90))
    return {"thickness_um_p90": thick, "spacing_um_p90": space}

def connectivity_density(vol01: np.ndarray, pixel_um: float, z_um: float) -> Dict[str, Optional[float]]:
    if not _HAS_SCIPY:
        return {"euler": None, "conn": None, "conn_d_per_mm3": None}

    bone = vol01.astype(bool)
    try:
        eul = float(ndi.euler_number(bone, connectivity=3))
    except Exception:
        return {"euler": None, "conn": None, "conn_d_per_mm3": None}

    conn = float(1.0 - eul)

    voxel_vol_um3 = float(pixel_um) * float(pixel_um) * float(z_um)
    tv_mm3 = (vol01.size * voxel_vol_um3) / 1e9
    conn_d = float(conn / tv_mm3) if tv_mm3 > 0 else None
    return {"euler": eul, "conn": conn, "conn_d_per_mm3": conn_d}


# -----------------------------
# Curvature: 2D curved rods + 3D curved bridges
# -----------------------------
def make_curved_polyline_2d(
    rng: np.random.Generator,
    x0: float,
    y0: float,
    length_px: float,
    base_angle_deg: float,
    segments: int,
    drift_deg: float,
    amp_px: float,
) -> List[Tuple[float, float]]:
    segments = max(2, int(segments))
    pts: List[Tuple[float, float]] = [(x0, y0)]

    angle = math.radians(base_angle_deg)
    step = float(length_px) / float(segments - 1)
    phase = float(rng.uniform(0.0, 2 * math.pi))

    x, y = x0, y0
    for i in range(1, segments):
        angle += math.radians(float(rng.normal(0.0, drift_deg)))
        dx = step * math.cos(angle)
        dy = step * math.sin(angle)

        t = i / (segments - 1)
        w = float(amp_px) * math.sin(2 * math.pi * t + phase)

        px = -math.sin(angle)
        py =  math.cos(angle)

        x = x + dx + w * px * 0.15
        y = y + dy + w * py * 0.15
        pts.append((x, y))
    return pts

def make_curved_path_3d(
    rng: np.random.Generator,
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    npts: int,
    amp: float,
) -> List[Tuple[float, float, float]]:
    npts = max(4, int(npts))
    z0, y0, x0 = p0
    z1, y1, x1 = p1

    ts = np.linspace(0.0, 1.0, npts)

    vx = x1 - x0
    vy = y1 - y0
    norm = math.hypot(vx, vy) + 1e-8
    px = -vy / norm
    py =  vx / norm

    mag = float(rng.uniform(-amp, amp))
    pts: List[Tuple[float, float, float]] = []
    for t in ts:
        z = z0 + t * (z1 - z0)
        y = y0 + t * (y1 - y0)
        x = x0 + t * (x1 - x0)
        bend = (math.sin(math.pi * t)) ** 2
        y = y + bend * mag * py
        x = x + bend * mag * px
        pts.append((z, y, x))
    return pts

def draw_tube_3d_from_points(vol01: np.ndarray, pts_zyx: List[Tuple[float, float, float]], radius: int) -> np.ndarray:
    Z, H, W = vol01.shape
    rr = int(max(1, radius))
    for (z, y, x) in pts_zyx:
        zi = int(round(z))
        yi = int(round(y))
        xi = int(round(x))

        zmin, zmax = max(0, zi - rr), min(Z, zi + rr + 1)
        ymin, ymax = max(0, yi - rr), min(H, yi + rr + 1)
        xmin, xmax = max(0, xi - rr), min(W, xi + rr + 1)

        zgrid, ygrid, xgrid = np.ogrid[zmin:zmax, ymin:ymax, xmin:xmax]
        mask = (zgrid - zi) ** 2 + (ygrid - yi) ** 2 + (xgrid - xi) ** 2 <= rr * rr
        vol01[zmin:zmax, ymin:ymax, xmin:xmax][mask] = 1
    return vol01


# -----------------------------
# Morphology for blob reduction (tube emphasis)
# -----------------------------
def tubular_thin_3d(vol01: np.ndarray, iters: int) -> np.ndarray:
    """
    Light thinning to reduce blob/plate masses and emphasize tubes.
    Keep iterations small to avoid disconnecting.
    """
    if not _HAS_SCIPY or iters <= 0:
        return vol01
    v = vol01.astype(bool)
    st = ndi.generate_binary_structure(3, 1)  # 6-neighborhood
    for _ in range(int(iters)):
        v = ndi.binary_opening(v, structure=st, iterations=1)
        v = ndi.binary_erosion(v, structure=st, iterations=1)
    return v.astype(np.uint8)


# -----------------------------
# 2D lattice generator (plates + curved rods)
# -----------------------------
def generate_lattice_2d(
    H: int,
    W: int,
    rng: np.random.Generator,
    p: LatticeParams2D,
    curv: CurvatureParams,
    pixel_um: float,
) -> np.ndarray:
    plate_centers: List[Tuple[float, float]] = []
    rod_midpoints: List[Tuple[float, float]] = []

    nnd_pp_min_px = max(1, um_to_px(p.nnd_pp_min_um, pixel_um))
    nnd_rr_min_px = max(1, um_to_px(p.nnd_rr_min_um, pixel_um))

    mask = np.zeros((H, W), dtype=np.uint8)

    # plates
    attempts = 0
    placed_plates = 0
    while placed_plates < p.pn and attempts < p.max_attempts:
        attempts += 1
        PA = rand_range(rng, p.plate_area_um2)
        PT = rand_range(rng, p.plate_thickness_um)
        t_px = max(1, um_to_px(PT, pixel_um))
        length_px = max(8.0, PA / (max(1.0, t_px) * (pixel_um ** 2)))

        ang = sample_angle_deg(rng, p.plate_preferred_angle_deg, p.plate_align_prob, p.plate_angle_spread_deg)
        cx = float(rng.uniform(0.15 * W, 0.85 * W))
        cy = float(rng.uniform(0.15 * H, 0.85 * H))

        if min_dist((cx, cy), plate_centers) < nnd_pp_min_px:
            continue

        pm = rasterize_rotated_rect(H, W, (cx, cy), length_px=length_px, width_px=float(t_px), angle_deg=ang)
        mask = np.maximum(mask, pm)
        plate_centers.append((cx, cy))
        placed_plates += 1

    # rods
    attempts = 0
    placed_rods = 0
    ys_b, xs_b = np.where(mask > 0)

    while placed_rods < p.rn and attempts < p.max_attempts:
        attempts += 1

        RL = rand_range(rng, p.rod_length_um)
        RD = rand_range(rng, p.rod_diameter_um)
        length_px = max(8.0, float(um_to_px(RL, pixel_um)))
        radius_px = max(1.0, float(um_to_px(0.5 * RD, pixel_um)))

        ang = sample_angle_deg(rng, p.rod_preferred_angle_deg, p.rod_align_prob, p.rod_angle_spread_deg)

        # anchor selection prefers existing structure
        if plate_centers and rng.random() < 0.45:
            cx, cy = plate_centers[int(rng.integers(0, len(plate_centers)))]
            x0, y0 = float(cx), float(cy)
        elif len(xs_b) > 0 and rng.random() < 0.55:
            idx = int(rng.integers(0, len(xs_b)))
            x0 = float(xs_b[idx])
            y0 = float(ys_b[idx])
        else:
            x0 = float(rng.uniform(0.1 * W, 0.9 * W))
            y0 = float(rng.uniform(0.1 * H, 0.9 * H))

        # curved polyline with probability
        if rng.random() < float(curv.curve_prob):
            pts = make_curved_polyline_2d(
                rng=rng,
                x0=x0,
                y0=y0,
                length_px=length_px,
                base_angle_deg=ang,
                segments=int(curv.curve_segments),
                drift_deg=float(curv.curve_drift_deg),
                amp_px=float(curv.curve_amp_px),
            )
            if any((x < 2 or x > (W - 3) or y < 2 or y > (H - 3)) for (x, y) in pts):
                continue

            mx = float(np.mean([q[0] for q in pts]))
            my = float(np.mean([q[1] for q in pts]))
            if min_dist((mx, my), rod_midpoints) < nnd_rr_min_px:
                continue

            rm = rasterize_polyline_thick(H, W, pts, radius_px=radius_px)
        else:
            th = math.radians(ang)
            dx = math.cos(th)
            dy = math.sin(th)
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
        ys_b, xs_b = np.where(mask > 0)

    # rough BV/TV tuning (2D)
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
# 3D voting + closing
# -----------------------------
def vote_3d(stack01: np.ndarray, v: Voting3DParams) -> np.ndarray:
    wy, wx, wz = oddify(v.win_y), oddify(v.win_x), oddify(v.win_z)
    kernel = np.ones((wz, wy, wx), dtype=np.uint8)
    window_size = int(kernel.size)

    if _HAS_SCIPY:
        votes = ndi.convolve(stack01.astype(np.uint16), kernel.astype(np.uint16), mode="constant", cval=0)
        if v.z_continuity_bias:
            st = ndi.generate_binary_structure(3, 1)
            prior = ndi.binary_dilation(stack01.astype(bool), structure=st, iterations=1)
            votes = votes + prior.astype(votes.dtype)
            window_size += 1
    else:
        Z, H, W = stack01.shape
        padz, pady, padx = wz // 2, wy // 2, wx // 2
        padded = np.pad(stack01.astype(np.uint8), ((padz, padz), (pady, pady), (padx, padx)), mode="constant")
        votes = np.zeros_like(stack01, dtype=np.uint16)
        for dz in range(wz):
            for dy in range(wy):
                for dx in range(wx):
                    votes += padded[dz:dz + Z, dy:dy + H, dx:dx + W]

    thr = int(math.ceil(float(v.tau) * float(window_size)))
    return (votes >= thr).astype(np.uint8)

def closing_3d(vol01: np.ndarray, iters: int, alternate: bool) -> np.ndarray:
    if not _HAS_SCIPY or iters <= 0:
        return vol01.astype(np.uint8)

    vol = (vol01 > 0)
    cube = np.ones((3, 3, 3), dtype=bool)
    cross = ndi.generate_binary_structure(3, 1)

    for i in range(int(iters)):
        st = cross if (alternate and (i % 2 == 0)) else cube
        vol = ndi.binary_closing(vol, structure=st, iterations=1)

    return vol.astype(np.uint8)


# -----------------------------
# Bridging: curved connectivity enforcement
# -----------------------------
def bridge_components_3d(
    vol01: np.ndarray,
    rng: np.random.Generator,
    curv: CurvatureParams,
    max_bridges: int,
    radius: int,
    smooth: int,
) -> np.ndarray:
    if not _HAS_SCIPY or max_bridges <= 0:
        return vol01

    bone = vol01.astype(bool)
    st = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity
    lab, n = ndi.label(bone, structure=st)
    if n <= 1:
        return vol01

    counts = np.bincount(lab.ravel())
    comp_ids = np.argsort(counts[1:])[::-1] + 1
    comp_ids = comp_ids[: max(2, min(len(comp_ids), max_bridges + 1))]

    centroids = []
    for cid in comp_ids:
        coords = np.argwhere(lab == cid)
        if coords.size == 0:
            continue
        cz, cy, cx = coords.mean(axis=0)
        centroids.append((float(cz), float(cy), float(cx)))

    if len(centroids) < 2:
        return vol01

    out = (vol01 > 0).astype(np.uint8)
    base = centroids[0]
    bridges_done = 0

    for target in centroids[1:]:
        if rng.random() < float(curv.curve_prob):
            pts = make_curved_path_3d(
                rng=rng,
                p0=base,
                p1=target,
                npts=int(curv.bridge_curve_segments),
                amp=float(curv.bridge_curve_amp_px),
            )
        else:
            npts = max(4, int(curv.bridge_curve_segments))
            ts = np.linspace(0.0, 1.0, npts)
            z0, y0, x0 = base
            z1, y1, x1 = target
            pts = [(z0 + t*(z1-z0), y0 + t*(y1-y0), x0 + t*(x1-x0)) for t in ts]

        out = draw_tube_3d_from_points(out, pts, radius=int(radius))
        bridges_done += 1
        if bridges_done >= int(max_bridges):
            break

    if smooth and smooth > 0:
        out = ndi.binary_dilation(
            out.astype(bool),
            structure=ndi.generate_binary_structure(3, 1),
            iterations=int(smooth),
        ).astype(np.uint8)

    return out


# -----------------------------
# Grayscale simulation modes
# -----------------------------
def gray_from_blur(vol01: np.ndarray, sigma: float, gamma: float) -> np.ndarray:
    soft = gaussian_blur(vol01.astype(np.float32), sigma=float(sigma))
    soft = clamp01(soft)
    if gamma != 1.0:
        soft = soft ** float(gamma)
    return (255.0 * soft).astype(np.uint8)

def gray_from_beerlambert(
    vol01: np.ndarray,
    mu: float,
    nz: int,
    gamma: float,
    noise_sd_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    nz = oddify(nz)
    r = nz // 2
    Z, H, W = vol01.shape

    padded = np.pad(vol01.astype(np.float32), ((r, r), (0, 0), (0, 0)), mode="constant")
    csum = np.cumsum(padded, axis=0)
    slab_sum = csum[nz:] - csum[:-nz]  # (Z,H,W)

    gray = 1.0 - np.exp(-float(mu) * slab_sum)
    gray = clamp01(gray)
    if gamma != 1.0:
        gray = gray ** float(gamma)

    gray_u8 = (255.0 * gray).astype(np.float32)
    sd = float(noise_sd_frac) * 255.0
    if sd > 0:
        gray_u8 += rng.normal(0.0, sd, size=gray_u8.shape).astype(np.float32)

    return np.clip(gray_u8, 0.0, 255.0).astype(np.uint8)

def gray_from_microct_slice(
    vol01: np.ndarray,
    pve_sigma: float,
    bone_mean: float,
    marrow_mean: float,
    ct_noise_sd: float,
    bg_texture_sd: float,
    unsharp: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Micro-CT-like reconstructed slice appearance:
    - Partial volume blur to soften boundaries (PVE)
    - Map to intensities (bone bright, marrow dark)
    - Add background texture + CT noise
    - Optional unsharp mask to regain crisp trabecular edges
    """
    x = vol01.astype(np.float32)

    if _HAS_SCIPY and pve_sigma > 0:
        x_blur = ndi.gaussian_filter(x, sigma=float(pve_sigma))
    else:
        x_blur = x
    x_blur = clamp01(x_blur)

    gray = float(marrow_mean) + x_blur * (float(bone_mean) - float(marrow_mean))

    if bg_texture_sd > 0:
        gray += rng.normal(0.0, float(bg_texture_sd), size=gray.shape).astype(np.float32)

    if ct_noise_sd > 0:
        gray += rng.normal(0.0, float(ct_noise_sd), size=gray.shape).astype(np.float32)

    if _HAS_SCIPY and unsharp > 0:
        blurred = ndi.gaussian_filter(gray, sigma=float(max(0.5, pve_sigma)))
        gray = gray + float(unsharp) * (gray - blurred)

    return np.clip(gray, 0.0, 255.0).astype(np.uint8)


# -----------------------------
# CLI
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Trabecular generator v4.1 (curvature + micro-CT grayscale).")
    p.add_argument("--outdir", type=str, default="data/v4_microct")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--n-volumes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    # calibration
    p.add_argument("--pixel-um", type=float, default=PIXEL_SIZE_UM_DEFAULT)
    p.add_argument("--z-step-um", type=float, default=Z_STEP_UM_DEFAULT)

    # 2D params
    p.add_argument("--bv-tv", type=float, default=0.18)
    p.add_argument("--pn", type=int, default=6)
    p.add_argument("--rn", type=int, default=12)

    # 3D voting params
    p.add_argument("--k", type=int, default=9)
    p.add_argument("--win-x", type=int, default=5)
    p.add_argument("--win-y", type=int, default=5)
    p.add_argument("--win-z", type=int, default=5)
    p.add_argument("--tau", type=float, default=0.52)
    p.add_argument("--closing-iters", type=int, default=2)
    p.add_argument("--no-alt-close", action="store_true")
    p.add_argument("--no-z-bias", action="store_true")

    # blob reduction / tube emphasis
    p.add_argument("--thin-iters", type=int, default=0,
                   help="Tubular thinning iterations (reduces blob/plate appearance). Start with 1.")

    # curvature controls
    p.add_argument("--curve-prob", type=float, default=0.7)
    p.add_argument("--curve-amp-px", type=int, default=12)
    p.add_argument("--curve-drift-deg", type=float, default=12.0)
    p.add_argument("--curve-segments", type=int, default=6)
    p.add_argument("--bridge-curve-amp-px", type=int, default=8)
    p.add_argument("--bridge-curve-segments", type=int, default=10)

    # connectivity controls
    p.add_argument("--min-lcc-frac", type=float, default=0.90)
    p.add_argument("--max-tries", type=int, default=8)
    p.add_argument("--auto-tune", action="store_true")
    p.add_argument("--bridge-3d", action="store_true")
    p.add_argument("--max-bridges", type=int, default=3)
    p.add_argument("--bridge-radius", type=int, default=2)
    p.add_argument("--bridge-smooth", type=int, default=1)

    # morphometric targets (optional)
    p.add_argument("--target-bv-tv", type=float, default=-1.0, help="If >0, enforce BV/TV target with tolerance.")
    p.add_argument("--bv-tv-tol", type=float, default=0.04)
    p.add_argument("--target-thickness-um", type=float, default=-1.0)
    p.add_argument("--thickness-tol-um", type=float, default=150.0)
    p.add_argument("--target-spacing-um", type=float, default=-1.0)
    p.add_argument("--spacing-tol-um", type=float, default=300.0)

    # grayscale output
    p.add_argument("--write-gray", action="store_true")
    p.add_argument("--gray-mode", type=str, default="microct", choices=["blur", "beerlambert", "microct"])
    # blur mode
    p.add_argument("--soft-sigma-px", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.9)
    # beerlambert mode
    p.add_argument("--mu", type=float, default=0.08)
    p.add_argument("--nz", type=int, default=3)
    p.add_argument("--noise-sd", type=float, default=0.2, help="Beer–Lambert noise SD fraction of 255.")
    # microct mode
    p.add_argument("--pve-sigma", type=float, default=1.1)
    p.add_argument("--bone-mean", type=float, default=215.0)
    p.add_argument("--marrow-mean", type=float, default=50.0)
    p.add_argument("--ct-noise-sd", type=float, default=10.0)
    p.add_argument("--bg-texture-sd", type=float, default=5.0)
    p.add_argument("--unsharp", type=float, default=0.8)

    # exports
    p.add_argument("--export-2d", action="store_true")
    p.add_argument("--export-2d-mode", type=str, default="all", choices=["all", "mid", "mip", "mean"])

    return p


# -----------------------------
# Acceptance logic
# -----------------------------
def meets_targets(metrics: Dict[str, Optional[float]], args: Any) -> bool:
    # LCC is mandatory
    lcc = metrics.get("lcc_frac")
    if lcc is None or float(lcc) < float(args.min_lcc_frac):
        return False

    if float(args.target_bv_tv) > 0:
        if abs(float(metrics["bv_tv_3d"]) - float(args.target_bv_tv)) > float(args.bv_tv_tol):
            return False

    if float(args.target_thickness_um) > 0 and metrics.get("thickness_um_p90") is not None:
        if abs(float(metrics["thickness_um_p90"]) - float(args.target_thickness_um)) > float(args.thickness_tol_um):
            return False

    if float(args.target_spacing_um) > 0 and metrics.get("spacing_um_p90") is not None:
        if abs(float(metrics["spacing_um_p90"]) - float(args.target_spacing_um)) > float(args.spacing_tol_um):
            return False

    return True

def score_candidate(metrics: Dict[str, Optional[float]], args: Any) -> Tuple[float, float, float, float]:
    """
    Higher is better:
      1) LCC
      2) BV/TV closeness (if target enabled)
      3) thickness closeness (if target enabled)
      4) spacing closeness (if target enabled)
    """
    lcc = float(metrics.get("lcc_frac") or 0.0)

    if float(args.target_bv_tv) > 0:
        bv_term = -abs(float(metrics["bv_tv_3d"]) - float(args.target_bv_tv))
    else:
        bv_term = -0.0

    if float(args.target_thickness_um) > 0 and metrics.get("thickness_um_p90") is not None:
        th_term = -abs(float(metrics["thickness_um_p90"]) - float(args.target_thickness_um))
    else:
        th_term = -0.0

    if float(args.target_spacing_um) > 0 and metrics.get("spacing_um_p90") is not None:
        sp_term = -abs(float(metrics["spacing_um_p90"]) - float(args.target_spacing_um))
    else:
        sp_term = -0.0

    return (lcc, bv_term, th_term, sp_term)


# -----------------------------
# Main
# -----------------------------
def main():
    args = build_parser().parse_args()
    rng = np.random.default_rng(int(args.seed))

    H = W = int(args.size)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pixel_um = float(args.pixel_um)
    z_um = float(args.z_step_um)

    lat_p = LatticeParams2D(bv_tv=float(args.bv_tv), pn=int(args.pn), rn=int(args.rn))
    vot_p = Voting3DParams(
        k_lattices=int(args.k),
        win_x=int(args.win_x), win_y=int(args.win_y), win_z=int(args.win_z),
        tau=float(args.tau),
        closing_iters=int(args.closing_iters),
        alternate_structure=(not bool(args.no_alt_close)),
        z_continuity_bias=(not bool(args.no_z_bias)),
    )
    curv = CurvatureParams(
        curve_prob=float(args.curve_prob),
        curve_amp_px=int(args.curve_amp_px),
        curve_drift_deg=float(args.curve_drift_deg),
        curve_segments=int(args.curve_segments),
        bridge_curve_amp_px=int(args.bridge_curve_amp_px),
        bridge_curve_segments=int(args.bridge_curve_segments),
    )

    images_csv = outdir / "volumes.csv"
    fieldnames = [
        "volume_id",
        "mask_tif",
        "gray_tif",
        "mid_png", "mip_png", "mean_png", "gray_mid_png", "gray_mip_png",
        "H", "W", "Z",
        "pixel_um", "z_step_um",
        "pn", "rn", "bv_tv_2d_target",
        "k", "win_x", "win_y", "win_z",
        "tau", "tau_used",
        "closing_iters", "closing_iters_used",
        "thin_iters",
        "auto_tune",
        "bridge_3d", "max_bridges", "bridge_radius", "bridge_smooth",
        "curve_prob", "curve_amp_px", "curve_drift_deg", "curve_segments",
        "bridge_curve_amp_px", "bridge_curve_segments",
        "gray_mode",
        "soft_sigma_px", "gamma",
        "mu", "nz", "noise_sd",
        "pve_sigma", "bone_mean", "marrow_mean", "ct_noise_sd", "bg_texture_sd", "unsharp",
        "bv_tv_3d",
        "cc_count", "lcc_frac",
        "thickness_um_p90", "spacing_um_p90",
        "euler", "conn", "conn_d_per_mm3",
        "accepted",
    ]
    f_csv, w_csv = init_csv(images_csv, fieldnames=fieldnames)

    try:
        for vi in range(int(args.n_volumes)):
            volume_id = f"vol_{vi:05d}"

            best = None  # (score, voted01, gray_stack, metrics, used_params, lattice_seeds)

            tau_base = float(vot_p.tau)
            close_base = int(vot_p.closing_iters)

            for attempt in range(int(args.max_tries)):
                # auto tuning: relax tau and increase closing if needed
                if args.auto_tune and attempt > 0:
                    tau_used = max(0.35, tau_base - 0.02 * attempt)
                    close_used = min(close_base + attempt // 2, close_base + 3)
                else:
                    tau_used = tau_base
                    close_used = close_base

                # 1) generate k 2D lattices
                k = int(vot_p.k_lattices)
                lattices = []
                lattice_seeds = []
                for zi in range(k):
                    sub_seed = int(rng.integers(0, 2**31 - 1))
                    lattice_seeds.append(sub_seed)
                    sub_rng = np.random.default_rng(sub_seed)
                    lattices.append(generate_lattice_2d(H, W, sub_rng, lat_p, curv, pixel_um))

                stack01 = np.stack(lattices, axis=0).astype(np.uint8)  # (Z,H,W)

                # 2) vote + close
                vot_local = Voting3DParams(**asdict(vot_p))
                vot_local.tau = float(tau_used)
                vot_local.closing_iters = int(close_used)

                voted01 = vote_3d(stack01, vot_local)
                voted01 = closing_3d(voted01, iters=vot_local.closing_iters, alternate=vot_local.alternate_structure)

                # 3) bridge (optional)
                if args.bridge_3d:
                    voted01 = bridge_components_3d(
                        voted01,
                        rng=rng,
                        curv=curv,
                        max_bridges=int(args.max_bridges),
                        radius=int(args.bridge_radius),
                        smooth=int(args.bridge_smooth),
                    )

                # 4) thin blobs (optional) - do AFTER bridging so you don't thin away connectors too early
                if int(args.thin_iters) > 0:
                    voted01 = tubular_thin_3d(voted01, iters=int(args.thin_iters))

                # 5) metrics
                bv_3d = float(np.mean(voted01 > 0))
                conn = connectivity_metrics_3d(voted01)
                ts = thickness_spacing_um(voted01, pixel_um=pixel_um, z_um=z_um)
                cd = connectivity_density(voted01, pixel_um=pixel_um, z_um=z_um)

                metrics: Dict[str, Optional[float]] = {
                    "bv_tv_3d": bv_3d,
                    "cc_count": conn["cc_count"],
                    "lcc_frac": conn["lcc_frac"],
                    "thickness_um_p90": ts["thickness_um_p90"],
                    "spacing_um_p90": ts["spacing_um_p90"],
                    "euler": cd["euler"],
                    "conn": cd["conn"],
                    "conn_d_per_mm3": cd["conn_d_per_mm3"],
                }

                # 6) grayscale (optional)
                gray_stack = None
                if bool(args.write_gray):
                    if str(args.gray_mode) == "blur":
                        gray_stack = gray_from_blur(
                            voted01,
                            sigma=float(args.soft_sigma_px),
                            gamma=float(args.gamma),
                        )
                    elif str(args.gray_mode) == "beerlambert":
                        gray_stack = gray_from_beerlambert(
                            voted01,
                            mu=float(args.mu),
                            nz=int(args.nz),
                            gamma=float(args.gamma),
                            noise_sd_frac=float(args.noise_sd),
                            rng=rng,
                        )
                    else:  # microct
                        gray_stack = gray_from_microct_slice(
                            voted01,
                            pve_sigma=float(args.pve_sigma),
                            bone_mean=float(args.bone_mean),
                            marrow_mean=float(args.marrow_mean),
                            ct_noise_sd=float(args.ct_noise_sd),
                            bg_texture_sd=float(args.bg_texture_sd),
                            unsharp=float(args.unsharp),
                            rng=rng,
                        )

                used_params = {"tau_used": float(tau_used), "closing_iters_used": int(close_used)}

                score = score_candidate(metrics, args)
                if best is None or score > best[0]:
                    best = (score, voted01.copy(), None if gray_stack is None else gray_stack.copy(), metrics, used_params, lattice_seeds)

                if meets_targets(metrics, args):
                    break

            assert best is not None
            _, voted01, gray_stack, metrics, used_params, lattice_seeds = best
            accepted = meets_targets(metrics, args)

            # save 3D outputs
            mask_path = outdir / f"{volume_id}_mask.tif"
            save_stack_u8((voted01 * 255).astype(np.uint8), mask_path, z_step_um=z_um)

            gray_path_name = ""
            if bool(args.write_gray) and gray_stack is not None:
                gray_path = outdir / f"{volume_id}_gray.tif"
                gray_path_name = gray_path.name
                save_stack_u8(gray_stack.astype(np.uint8), gray_path, z_step_um=z_um)

            # 2D exports
            mid_png = mip_png = mean_png = ""
            gray_mid_png = gray_mip_png = ""
            if bool(args.export_2d):
                zmid = voted01.shape[0] // 2
                mid_slice = (voted01[zmid] * 255).astype(np.uint8)
                mip_xy = (voted01.max(axis=0) * 255).astype(np.uint8)
                mean_xy = np.clip(voted01.mean(axis=0) * 255.0, 0, 255).astype(np.uint8)

                mode = str(args.export_2d_mode)
                if mode in ("all", "mid"):
                    mid_png = f"{volume_id}_mid.png"
                    save_png(mid_slice, outdir / mid_png)
                if mode in ("all", "mip"):
                    mip_png = f"{volume_id}_mip.png"
                    save_png(mip_xy, outdir / mip_png)
                if mode in ("all", "mean"):
                    mean_png = f"{volume_id}_mean.png"
                    save_png(mean_xy, outdir / mean_png)

                if bool(args.write_gray) and gray_stack is not None:
                    gray_mid_png = f"{volume_id}_gray_mid.png"
                    gray_mip_png = f"{volume_id}_gray_mip.png"
                    save_png(gray_stack[zmid].astype(np.uint8), outdir / gray_mid_png)
                    save_png(gray_stack.max(axis=0).astype(np.uint8), outdir / gray_mip_png)

            # JSON metadata
            meta: Dict[str, Any] = {
                "volume_id": volume_id,
                "files": {
                    "mask_tif": mask_path.name,
                    "gray_tif": gray_path_name or None,
                    "mid_png": mid_png or None,
                    "mip_png": mip_png or None,
                    "mean_png": mean_png or None,
                    "gray_mid_png": gray_mid_png or None,
                    "gray_mip_png": gray_mip_png or None,
                },
                "globals": {
                    "H": int(H), "W": int(W), "Z": int(voted01.shape[0]),
                    "pixel_um": float(pixel_um),
                    "z_step_um": float(z_um),
                    "seed": int(args.seed),
                },
                "params": {
                    "lattice_2d": asdict(lat_p),
                    "voting_3d": asdict(vot_p),
                    "curvature": asdict(curv),
                    "morphology": {
                        "thin_iters": int(args.thin_iters),
                    },
                    "connectivity": {
                        "min_lcc_frac": float(args.min_lcc_frac),
                        "max_tries": int(args.max_tries),
                        "auto_tune": bool(args.auto_tune),
                        "bridge_3d": bool(args.bridge_3d),
                        "max_bridges": int(args.max_bridges),
                        "bridge_radius": int(args.bridge_radius),
                        "bridge_smooth": int(args.bridge_smooth),
                    },
                    "targets": {
                        "target_bv_tv": float(args.target_bv_tv),
                        "bv_tv_tol": float(args.bv_tv_tol),
                        "target_thickness_um": float(args.target_thickness_um),
                        "thickness_tol_um": float(args.thickness_tol_um),
                        "target_spacing_um": float(args.target_spacing_um),
                        "spacing_tol_um": float(args.spacing_tol_um),
                    },
                    "gray": {
                        "write_gray": bool(args.write_gray),
                        "gray_mode": str(args.gray_mode),
                        "blur": {"soft_sigma_px": float(args.soft_sigma_px), "gamma": float(args.gamma)},
                        "beerlambert": {"mu": float(args.mu), "nz": int(args.nz), "noise_sd_frac": float(args.noise_sd)},
                        "microct": {
                            "pve_sigma": float(args.pve_sigma),
                            "bone_mean": float(args.bone_mean),
                            "marrow_mean": float(args.marrow_mean),
                            "ct_noise_sd": float(args.ct_noise_sd),
                            "bg_texture_sd": float(args.bg_texture_sd),
                            "unsharp": float(args.unsharp),
                        },
                    },
                },
                "used_params": used_params,
                "lattice_seeds": lattice_seeds,
                "metrics": metrics,
                "accepted": bool(accepted),
                "notes": [
                    "To approach micro-CT fig (c): set pn=0, increase rn, enable thin-iters, and use gray-mode microct.",
                    "Conn/Conn.D derived from Euler characteristic are approximate; tools may use different conventions.",
                ],
            }
            with open(outdir / f"{volume_id}.json", "w") as f:
                json.dump(meta, f, indent=2)

            # CSV row
            w_csv.writerow({
                "volume_id": volume_id,
                "mask_tif": mask_path.name,
                "gray_tif": gray_path_name,
                "mid_png": mid_png,
                "mip_png": mip_png,
                "mean_png": mean_png,
                "gray_mid_png": gray_mid_png,
                "gray_mip_png": gray_mip_png,
                "H": H, "W": W, "Z": int(voted01.shape[0]),
                "pixel_um": float(pixel_um),
                "z_step_um": float(z_um),
                "pn": int(lat_p.pn),
                "rn": int(lat_p.rn),
                "bv_tv_2d_target": float(lat_p.bv_tv),
                "k": int(vot_p.k_lattices),
                "win_x": int(vot_p.win_x),
                "win_y": int(vot_p.win_y),
                "win_z": int(vot_p.win_z),
                "tau": float(vot_p.tau),
                "tau_used": float(used_params["tau_used"]),
                "closing_iters": int(vot_p.closing_iters),
                "closing_iters_used": int(used_params["closing_iters_used"]),
                "thin_iters": int(args.thin_iters),
                "auto_tune": bool(args.auto_tune),
                "bridge_3d": bool(args.bridge_3d),
                "max_bridges": int(args.max_bridges),
                "bridge_radius": int(args.bridge_radius),
                "bridge_smooth": int(args.bridge_smooth),
                "curve_prob": float(curv.curve_prob),
                "curve_amp_px": int(curv.curve_amp_px),
                "curve_drift_deg": float(curv.curve_drift_deg),
                "curve_segments": int(curv.curve_segments),
                "bridge_curve_amp_px": int(curv.bridge_curve_amp_px),
                "bridge_curve_segments": int(curv.bridge_curve_segments),
                "gray_mode": str(args.gray_mode),
                "soft_sigma_px": float(args.soft_sigma_px),
                "gamma": float(args.gamma),
                "mu": float(args.mu),
                "nz": int(args.nz),
                "noise_sd": float(args.noise_sd),
                "pve_sigma": float(args.pve_sigma),
                "bone_mean": float(args.bone_mean),
                "marrow_mean": float(args.marrow_mean),
                "ct_noise_sd": float(args.ct_noise_sd),
                "bg_texture_sd": float(args.bg_texture_sd),
                "unsharp": float(args.unsharp),
                "bv_tv_3d": float(metrics["bv_tv_3d"]),
                "cc_count": metrics["cc_count"],
                "lcc_frac": metrics["lcc_frac"],
                "thickness_um_p90": metrics["thickness_um_p90"],
                "spacing_um_p90": metrics["spacing_um_p90"],
                "euler": metrics["euler"],
                "conn": metrics["conn"],
                "conn_d_per_mm3": metrics["conn_d_per_mm3"],
                "accepted": bool(accepted),
            })

            lcc = metrics.get("lcc_frac")
            lcc_str = f"{float(lcc):.3f}" if lcc is not None else "None"
            th_str = f"{metrics['thickness_um_p90']:.1f}" if metrics.get("thickness_um_p90") is not None else "NA"
            sp_str = f"{metrics['spacing_um_p90']:.1f}" if metrics.get("spacing_um_p90") is not None else "NA"
            print(
                f"[{vi+1}/{args.n_volumes}] {volume_id} | "
                f"BV/TV={metrics['bv_tv_3d']:.3f} | LCC={lcc_str} | Tb.Th~{th_str}um | Tb.Sp~{sp_str}um | "
                f"gray={args.gray_mode} | tau_used={used_params['tau_used']:.3f} | accepted={accepted}"
            )

    finally:
        f_csv.close()

    if not _HAS_SCIPY:
        print(
            "Note: scipy not available → 3D closing/connectivity/EDT/Euler are limited.\n"
            "Install scipy for best results: pip install scipy"
        )


if __name__ == "__main__":
    main()
