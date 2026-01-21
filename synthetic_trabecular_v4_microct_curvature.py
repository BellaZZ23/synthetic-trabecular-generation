#!/usr/bin/env python3
"""
synthetic_trabecular_v4_microct_curvature.py

Connectivity-first trabecular generator upgraded to better match micro-CT expectations:

Adds:
1) Curvature control (curved rods in 2D + curved 3D bridging tubes)
2) Micro-CT grayscale simulation mode (Beer–Lambert slab model + Gaussian noise)
3) Morphometric acceptance targets (BV/TV, thickness, spacing, LCC) in the retry loop
4) Connectivity density proxy via Euler characteristic (Conn and Conn.D)
5) Cleaner logging: per-volume JSON + CSV

Dependencies:
- numpy, pillow, tifffile
- scipy strongly recommended (labeling, morphology, EDT, euler number, convolution)

PowerShell example:
python .\synthetic_trabecular_v4_microct_curvature.py --outdir data\v4 --n-volumes 20 --size 512 --seed 42 `
  --write-gray --gray-mode beerlambert --mu 0.08 --nz 3 --noise-sd 0.2 `
  --min-lcc-frac 0.95 --max-tries 8 --auto-tune --bridge-3d `
  --target-bv-tv 0.18 --bv-tv-tol 0.04 `
  --target-thickness-um 200 --thickness-tol-um 120 `
  --target-spacing-um 700 --spacing-tol-um 300 `
  --curve-prob 0.7 --curve-amp-px 12 --curve-drift-deg 12
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List, Dict, Optional

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
    curve_prob: float = 0.7          # probability a rod/bridge is curved
    curve_amp_px: int = 12           # curvature amplitude (pixels) for 2D polyline
    curve_drift_deg: float = 12.0    # random angular drift per segment (degrees)
    curve_segments: int = 6          # segments per rod polyline

    bridge_curve_amp_px: int = 8     # curvature amplitude (voxels) for 3D bridges
    bridge_curve_segments: int = 10  # points along 3D curved path


@dataclass
class GrayParams:
    write_gray: bool = True
    gray_mode: str = "blur"          # blur | beerlambert
    # blur mode
    soft_sigma_px: float = 1.0
    gamma: float = 0.9
    # beer-lambert mode
    mu: float = 0.08                 # attenuation coefficient
    nz: int = 3                      # slab thickness (odd recommended)
    noise_sd_frac: float = 0.2       # Gaussian noise SD as fraction of 255


# -----------------------------
# Helpers
# -----------------------------
def um_to_px(val_um: float, pixel_size_um: float) -> int:
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
    """
    EDT proxies converted to microns.
    We approximate voxel spacing isotropic in XY and use z_um for Z.
    """
    if not _HAS_SCIPY:
        return {"thickness_um_p90": None, "spacing_um_p90": None}

    bone = vol01.astype(bool)
    if bone.sum() == 0 or (~bone).sum() == 0:
        return {"thickness_um_p90": 0.0, "spacing_um_p90": 0.0}

    # distance_transform_edt supports sampling (voxel size) via 'sampling'
    sampling = (float(z_um), float(pixel_um), float(pixel_um))
    dt_bone = ndi.distance_transform_edt(bone, sampling=sampling)
    dt_space = ndi.distance_transform_edt(~bone, sampling=sampling)

    thick = float(np.percentile(dt_bone[bone], 90))
    space = float(np.percentile(dt_space[~bone], 90))
    return {"thickness_um_p90": thick, "spacing_um_p90": space}

def connectivity_density(vol01: np.ndarray, pixel_um: float, z_um: float) -> Dict[str, Optional[float]]:
    """
    Conn = 1 - Euler characteristic (common convention in bone morphometry).
    Conn.D = Conn / TV (total volume).
    This is an approximation; exact conventions vary by connectivity definition.
    """
    if not _HAS_SCIPY:
        return {"euler": None, "conn": None, "conn_d_per_mm3": None}

    bone = vol01.astype(bool)

    # scipy's euler_number connectivity parameter ranges 1..ndim
    # use max connectivity (3) for 3D which aligns most with 26-neighborhood intuition
    try:
        eul = float(ndi.euler_number(bone, connectivity=3))
    except Exception:
        return {"euler": None, "conn": None, "conn_d_per_mm3": None}

    conn = float(1.0 - eul)

    # total volume in mm^3
    voxel_vol_um3 = float(pixel_um) * float(pixel_um) * float(z_um)
    tv_mm3 = (vol01.size * voxel_vol_um3) / 1e9
    conn_d = float(conn / tv_mm3) if tv_mm3 > 0 else None

    return {"euler": eul, "conn": conn, "conn_d_per_mm3": conn_d}


# -----------------------------
# Curvature: 2D curved rods + 3D curved bridge tubes
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
    """
    Build a polyline starting at (x0,y0) of given length, with angular drift and lateral waviness.
    """
    segments = max(2, int(segments))
    pts: List[Tuple[float, float]] = [(x0, y0)]

    angle = math.radians(base_angle_deg)
    step = float(length_px) / float(segments - 1)

    # random phase for a gentle wave
    phase = float(rng.uniform(0.0, 2 * math.pi))

    x, y = x0, y0
    for i in range(1, segments):
        # drift the direction
        angle += math.radians(float(rng.normal(0.0, drift_deg)))
        dx = step * math.cos(angle)
        dy = step * math.sin(angle)

        # add perpendicular waviness (small, smooth)
        t = i / (segments - 1)
        w = float(amp_px) * math.sin(2 * math.pi * t + phase)
        # perpendicular vector
        px = -math.sin(angle)
        py =  math.cos(angle)

        x = x + dx + w * px * 0.15
        y = y + dy + w * py * 0.15
        pts.append((x, y))

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

def make_curved_path_3d(
    rng: np.random.Generator,
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    npts: int,
    amp: float,
) -> List[Tuple[float, float, float]]:
    """
    Create a curved 3D path between p0 and p1 by adding a mid-curve offset.
    """
    npts = max(4, int(npts))
    z0, y0, x0 = p0
    z1, y1, x1 = p1

    # base linear interpolation
    ts = np.linspace(0.0, 1.0, npts)

    # choose a random offset direction roughly perpendicular to (p1-p0) in XY plane
    vx = x1 - x0
    vy = y1 - y0
    norm = math.hypot(vx, vy) + 1e-8
    px = -vy / norm
    py =  vx / norm

    # random sign and magnitude
    mag = float(rng.uniform(-amp, amp))
    # apply offset strongest in middle, zero at ends
    pts: List[Tuple[float, float, float]] = []
    for t in ts:
        z = z0 + t * (z1 - z0)
        y = y0 + t * (y1 - y0)
        x = x0 + t * (x1 - x0)

        bend = (math.sin(math.pi * t)) ** 2  # 0 at ends, 1 near middle
        y = y + bend * mag * py
        x = x + bend * mag * px

        pts.append((z, y, x))
    return pts


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

        # length derived from area proxy
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

    # rods: curved polylines encouraged
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

        # anchor selection: prefer attaching to existing structure
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

        # build rod path (curved with probability)
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
            # must remain in bounds
            if any((x < 2 or x > (W - 3) or y < 2 or y > (H - 3)) for (x, y) in pts):
                continue
            mx = float(np.mean([p[0] for p in pts]))
            my = float(np.mean([p[1] for p in pts]))
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

    # rough BV/TV tuning
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
# Curved 3D bridging (connectivity enforcement, but less "straight")
# -----------------------------
def bridge_components_3d(
    vol01: np.ndarray,
    rng: np.random.Generator,
    curv: CurvatureParams,
    max_bridges: int = 3,
    radius: int = 2,
    smooth: int = 1,
) -> np.ndarray:
    if not _HAS_SCIPY or max_bridges <= 0:
        return vol01

    bone = vol01.astype(bool)
    st = np.ones((3, 3, 3), dtype=bool)
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
        # curved path with probability curve_prob
        if rng.random() < float(curv.curve_prob):
            pts = make_curved_path_3d(
                rng=rng,
                p0=base,
                p1=target,
                npts=int(curv.bridge_curve_segments),
                amp=float(curv.bridge_curve_amp_px),
            )
        else:
            # straight fallback
            npts = int(curv.bridge_curve_segments)
            ts = np.linspace(0.0, 1.0, max(4, npts))
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
# Grayscale simulation: blur or Beer–Lambert slab + noise
# -----------------------------
def gray_from_blur(vol01: np.ndarray, sigma: float, gamma: float) -> np.ndarray:
    soft = gaussian_blur(vol01.astype(np.float32), sigma=float(sigma))
    soft = clamp01(soft)
    if gamma != 1.0:
        soft = soft ** float(gamma)
    return (255.0 * soft).astype(np.uint8)

def gray_from_beerlambert(vol01: np.ndarray, mu: float, nz: int, gamma: float, noise_sd_frac: float,
                          rng: np.random.Generator) -> np.ndarray:
    """
    For each z: slab_sum = sum_{z' in slab} S(z')
    gray = 255*(1-exp(-mu*slab_sum)) + Gaussian noise
    """
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


# -----------------------------
# CLI
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Trabecular generator v4 (curvature + micro-CT gray + morphometric targets).")
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
    p.add_argument("--gray-mode", type=str, default="blur", choices=["blur", "beerlambert"])
    p.add_argument("--soft-sigma-px", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--mu", type=float, default=0.08)
    p.add_argument("--nz", type=int, default=3)
    p.add_argument("--noise-sd", type=float, default=0.2)

    # exports
    p.add_argument("--export-2d", action="store_true")
    p.add_argument("--export-2d-mode", type=str, default="all", choices=["all", "mid", "mip", "mean"])

    return p


# -----------------------------
# Generation: one volume with targets + retries
# -----------------------------
def meets_targets(metrics: Dict[str, Optional[float]], args) -> bool:
    # LCC
    if metrics.get("lcc_frac") is not None:
        if float(metrics["lcc_frac"]) < float(args.min_lcc_frac):
            return False
    else:
        return False

    # BV/TV target
    if float(args.target_bv_tv) > 0:
        if abs(float(metrics["bv_tv_3d"]) - float(args.target_bv_tv)) > float(args.bv_tv_tol):
            return False

    # thickness target
    if float(args.target_thickness_um) > 0 and metrics.get("thickness_um_p90") is not None:
        if abs(float(metrics["thickness_um_p90"]) - float(args.target_thickness_um)) > float(args.thickness_tol_um):
            return False

    # spacing target
    if float(args.target_spacing_um) > 0 and metrics.get("spacing_um_p90") is not None:
        if abs(float(metrics["spacing_um_p90"]) - float(args.target_spacing_um)) > float(args.spacing_tol_um):
            return False

    return True

def score_candidate(metrics: Dict[str, Optional[float]], args) -> Tuple[float, float, float, float]:
    """
    Higher is better:
      1) LCC
      2) BV/TV closeness (if target enabled)
      3) thickness closeness (if target enabled)
      4) spacing closeness (if target enabled)
    """
    lcc = float(metrics.get("lcc_frac") or 0.0)

    # closeness: negative absolute error (so closer -> higher)
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
    gray_p = GrayParams(
        write_gray=bool(args.write_gray),
        gray_mode=str(args.gray_mode),
        soft_sigma_px=float(args.soft_sigma_px),
        gamma=float(args.gamma),
        mu=float(args.mu),
        nz=int(args.nz),
        noise_sd_frac=float(args.noise_sd),
    )

    images_csv = outdir / "volumes.csv"
    fieldnames = [
        "volume_id",
        "mask_tif",
        "gray_tif",
        "mid_png","mip_png","mean_png","gray_mid_png","gray_mip_png",
        "H","W","Z",
        "pixel_um","z_step_um",
        "pn","rn","bv_tv_2d_target",
        "k","win_x","win_y","win_z",
        "tau","tau_used","closing_iters","closing_iters_used",
        "auto_tune",
        "bridge_3d","max_bridges","bridge_radius","bridge_smooth",
        "curve_prob","curve_amp_px","curve_drift_deg","curve_segments",
        "bridge_curve_amp_px","bridge_curve_segments",
        "gray_mode","mu","nz","noise_sd","soft_sigma_px","gamma",
        "bv_tv_3d",
        "cc_count","lcc_frac",
        "thickness_um_p90","spacing_um_p90",
        "euler","conn","conn_d_per_mm3",
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
                # auto tune knobs (simple)
                if args.auto_tune and attempt > 0:
                    tau_used = max(0.35, tau_base - 0.02 * attempt)
                    close_used = min(close_base + attempt // 2, close_base + 3)
                else:
                    tau_used = tau_base
                    close_used = close_base

                # 1) generate k lattices
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

                # 3) optional bridging (curved)
                if args.bridge_3d:
                    voted01 = bridge_components_3d(
                        voted01,
                        rng=rng,
                        curv=curv,
                        max_bridges=int(args.max_bridges),
                        radius=int(args.bridge_radius),
                        smooth=int(args.bridge_smooth),
                    )

                # 4) metrics
                bv_3d = float(np.mean(voted01 > 0))
                conn = connectivity_metrics_3d(voted01)
                ts = thickness_spacing_um(voted01, pixel_um=pixel_um, z_um=z_um)
                cd = connectivity_density(voted01, pixel_um=pixel_um, z_um=z_um)

                metrics = {
                    "bv_tv_3d": bv_3d,
                    "cc_count": conn["cc_count"],
                    "lcc_frac": conn["lcc_frac"],
                    "thickness_um_p90": ts["thickness_um_p90"],
                    "spacing_um_p90": ts["spacing_um_p90"],
                    "euler": cd["euler"],
                    "conn": cd["conn"],
                    "conn_d_per_mm3": cd["conn_d_per_mm3"],
                }

                # 5) grayscale
                gray_stack = None
                if gray_p.write_gray:
                    if gray_p.gray_mode == "blur":
                        gray_stack = gray_from_blur(voted01.astype(np.float32), sigma=float(gray_p.soft_sigma_px), gamma=float(gray_p.gamma))
                    else:
                        gray_stack = gray_from_beerlambert(
                            voted01,
                            mu=float(gray_p.mu),
                            nz=int(gray_p.nz),
                            gamma=float(gray_p.gamma),
                            noise_sd_frac=float(gray_p.noise_sd_frac),
                            rng=rng,
                        )

                used_params = {
                    "tau_used": float(tau_used),
                    "closing_iters_used": int(close_used),
                }

                score = score_candidate(metrics, args)
                if best is None or score > best[0]:
                    best = (score, voted01.copy(), None if gray_stack is None else gray_stack.copy(), metrics, used_params, lattice_seeds)

                # accept early if meets all enabled targets
                if meets_targets(metrics, args):
                    break

            assert best is not None
            _, voted01, gray_stack, metrics, used_params, lattice_seeds = best
            accepted = meets_targets(metrics, args)

            # outputs
            mask_path = outdir / f"{volume_id}_mask.tif"
            save_stack_u8((voted01 * 255).astype(np.uint8), mask_path, z_step_um=z_um)

            gray_path = ""
            if gray_p.write_gray and gray_stack is not None:
                gray_path = (outdir / f"{volume_id}_gray.tif").name
                save_stack_u8(gray_stack.astype(np.uint8), outdir / gray_path, z_step_um=z_um)

            # 2D exports
            mid_png = mip_png = mean_png = ""
            gray_mid_png = gray_mip_png = ""
            if args.export_2d:
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

                if gray_p.write_gray and gray_stack is not None:
                    gray_mid_png = f"{volume_id}_gray_mid.png"
                    gray_mip_png = f"{volume_id}_gray_mip.png"
                    save_png(gray_stack[zmid].astype(np.uint8), outdir / gray_mid_png)
                    save_png(gray_stack.max(axis=0).astype(np.uint8), outdir / gray_mip_png)

            # metadata
            meta = {
                "volume_id": volume_id,
                "files": {
                    "mask_tif": mask_path.name,
                    "gray_tif": gray_path or None,
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
                    "gray": asdict(gray_p),
                    "targets": {
                        "min_lcc_frac": float(args.min_lcc_frac),
                        "target_bv_tv": float(args.target_bv_tv),
                        "bv_tv_tol": float(args.bv_tv_tol),
                        "target_thickness_um": float(args.target_thickness_um),
                        "thickness_tol_um": float(args.thickness_tol_um),
                        "target_spacing_um": float(args.target_spacing_um),
                        "spacing_tol_um": float(args.spacing_tol_um),
                    },
                    "runtime": {
                        "max_tries": int(args.max_tries),
                        "auto_tune": bool(args.auto_tune),
                        "bridge_3d": bool(args.bridge_3d),
                        "max_bridges": int(args.max_bridges),
                        "bridge_radius": int(args.bridge_radius),
                        "bridge_smooth": int(args.bridge_smooth),
                    },
                },
                "used_params": used_params,
                "lattice_seeds": lattice_seeds,
                "metrics": metrics,
                "accepted": bool(accepted),
                "notes": [
                    "Curvature is introduced via polyline rods and curved 3D bridges.",
                    "Beer–Lambert grayscale uses slab thickness nz; blur mode remains available.",
                    "Thickness/spacing are EDT-based proxies converted to microns using voxel sampling.",
                    "Conn.D is approximated via Euler characteristic; conventions may vary across tools.",
                ],
            }
            with open(outdir / f"{volume_id}.json", "w") as f:
                json.dump(meta, f, indent=2)

            # CSV row
            w_csv.writerow({
                "volume_id": volume_id,
                "mask_tif": mask_path.name,
                "gray_tif": gray_path,
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
                "gray_mode": str(gray_p.gray_mode),
                "mu": float(gray_p.mu),
                "nz": int(gray_p.nz),
                "noise_sd": float(gray_p.noise_sd_frac),
                "soft_sigma_px": float(gray_p.soft_sigma_px),
                "gamma": float(gray_p.gamma),
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
            print(
                f"[{vi+1}/{args.n_volumes}] {volume_id} | "
                f"BV/TV={metrics['bv_tv_3d']:.3f} | LCC={lcc_str} | "
                f"Tb.Th~{metrics['thickness_um_p90'] if metrics['thickness_um_p90'] is not None else 'NA'}um | "
                f"mode={gray_p.gray_mode} | tau_used={used_params['tau_used']:.3f} | accepted={accepted}"
            )

    finally:
        f_csv.close()

    if not _HAS_SCIPY:
        print("Note: scipy not available → 3D closing/connectivity/EDT/Euler will be limited.")


if __name__ == "__main__":
    main()
