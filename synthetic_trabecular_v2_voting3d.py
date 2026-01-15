#!/usr/bin/env python3
"""
synthetic_trabecular_v3_connectivity_first.py

Synthetic trabecular 3D generator with "connectivity-first" controls + 3D component bridging.

What’s new vs your v2:
1) Connectivity metrics are computed per 3D volume:
   - cc_count (26-connectivity)
   - lcc_frac (largest connected component fraction)
2) Acceptance loop per requested volume:
   - regenerate up to --max-tries
   - keeps best candidate if constraints aren’t met
3) Optional automatic parameter nudging to improve connectivity:
   - relax tau slightly when connectivity is poor
   - increase closing iterations (bounded)
4) Adds EDT-based proxies (simple but useful) for:
   - thickness_proxy (bone EDT p90)
   - spacing_proxy (space EDT p90)
5) Adds optional parameter sweep mode for interpretability/PCA:
   - sweep tau and/or window size lists
6) Logs metrics to both JSON and CSV.
7) NEW: Optional 3D component bridging to actively enforce global connectivity:
   - bridges largest components after voting+closing (tubular connectors)
   - optional smoothing dilation after bridging

Outputs per accepted volume:
- mask 3D TIFF
- optional grayscale 3D TIFF
- optional 2D PNGs: mid-slice, MIP, mean (binary)
- optional 2D PNGs for grayscale: mid-slice, MIP
- JSON metadata + metrics
- CSV summary rows

Requirements:
- numpy, pillow, tifffile
- scipy strongly recommended (for fast voting, morphology, EDT, labeling)

Author: (you)
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
PIXEL_SIZE_UM = 10.0
Z_STEP_UM = 1.0


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

    # continuity bias: encourages bone persistence across neighborhood
    z_continuity_bias: bool = True


@dataclass
class RenderParams:
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
    xr = c * xt + s * yt
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
    return arr


# -----------------------------
# Metrics (connectivity + proxies)
# -----------------------------
def connectivity_metrics_3d(vol01: np.ndarray) -> Dict[str, Optional[float]]:
    """
    26-connectivity labeling for bone phase.
    Returns cc_count and lcc_frac (largest component fraction of bone voxels).
    """
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


def thickness_spacing_proxies(vol01: np.ndarray) -> Dict[str, Optional[float]]:
    """
    EDT-based proxies:
    - thickness_proxy: p90 of EDT within bone
    - spacing_proxy: p90 of EDT within space
    """
    if not _HAS_SCIPY:
        return {"thickness_proxy": None, "spacing_proxy": None}

    bone = vol01.astype(bool)
    if bone.sum() == 0 or (~bone).sum() == 0:
        return {"thickness_proxy": 0.0, "spacing_proxy": 0.0}

    dt_bone = ndi.distance_transform_edt(bone)
    dt_space = ndi.distance_transform_edt(~bone)

    thick = float(np.percentile(dt_bone[bone], 90))
    space = float(np.percentile(dt_space[~bone], 90))
    return {"thickness_proxy": thick, "spacing_proxy": space}


# -----------------------------
# NEW: 3D bridging utilities (generator enforcement)
# -----------------------------
def draw_tube_3d(vol01: np.ndarray, p0, p1, radius: int) -> np.ndarray:
    """
    Draw a thick 3D tube between points p0 and p1 into vol01.
    vol01: uint8 (Z,H,W) with 0/1.
    p0/p1: (z,y,x)
    radius: voxel radius
    """
    Z, H, W = vol01.shape
    z0, y0, x0 = [float(v) for v in p0]
    z1, y1, x1 = [float(v) for v in p1]

    n = int(max(abs(z1 - z0), abs(y1 - y0), abs(x1 - x0)) * 2) + 1
    zz = np.linspace(z0, z1, n)
    yy = np.linspace(y0, y1, n)
    xx = np.linspace(x0, x1, n)

    rr = int(max(1, radius))
    for z, y, x in zip(zz, yy, xx):
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


def bridge_components_3d(vol01: np.ndarray, max_bridges: int = 3, radius: int = 2, smooth: int = 0) -> np.ndarray:
    """
    Actively connect disconnected components in 3D by drawing tubes between component centroids.
    - Labels bone with 26-connectivity
    - Chooses largest components and connects largest -> next largest sequentially
    - Optional dilation smoothing

    Requires scipy.
    """
    if not _HAS_SCIPY or max_bridges <= 0:
        return vol01

    bone = vol01.astype(bool)
    st = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity
    lab, n = ndi.label(bone, structure=st)
    if n <= 1:
        return vol01

    counts = np.bincount(lab.ravel())
    comp_ids = np.argsort(counts[1:])[::-1] + 1  # descending by size, skip background
    # Need at least 2 comps; cap how many comps we consider
    comp_ids = comp_ids[: max(2, min(len(comp_ids), max_bridges + 1))]

    centroids = []
    for cid in comp_ids:
        coords = np.argwhere(lab == cid)
        if coords.size == 0:
            continue
        cz, cy, cx = coords.mean(axis=0)
        centroids.append((cz, cy, cx))

    if len(centroids) < 2:
        return vol01

    out = (vol01 > 0).astype(np.uint8)

    base = centroids[0]
    bridges_done = 0
    for target in centroids[1:]:
        out = draw_tube_3d(out, base, target, radius=int(radius))
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
# 2D lattice generator
# -----------------------------
def generate_lattice_2d(H: int, W: int, rng: np.random.Generator, p: LatticeParams2D) -> np.ndarray:
    plate_centers: List[Tuple[float, float]] = []
    rod_midpoints: List[Tuple[float, float]] = []

    nnd_pp_min_px = max(1, int(round(p.nnd_pp_min_um / PIXEL_SIZE_UM)))
    nnd_rr_min_px = max(1, int(round(p.nnd_rr_min_um / PIXEL_SIZE_UM)))

    mask = np.zeros((H, W), dtype=np.uint8)

    # plates
    attempts = 0
    placed_plates = 0
    while placed_plates < p.pn and attempts < p.max_attempts:
        attempts += 1
        PA = rand_range(rng, p.plate_area_um2)
        PT = rand_range(rng, p.plate_thickness_um)
        t_px = max(1, um_to_px(PT))

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

    # rods (prefer anchoring on plate centers sometimes to create long connections)
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
# 3D voting + morphology
# -----------------------------
def vote_3d(stack01: np.ndarray, v: Voting3DParams) -> np.ndarray:
    """
    stack01: uint8 (Z,H,W)
    Applies 3D sliding-window voting using convolution.
    Optional continuity bias to encourage stable networks.
    """
    wy, wx, wz = int(v.win_y), int(v.win_x), int(v.win_z)
    wy = max(1, wy | 1)
    wx = max(1, wx | 1)
    wz = max(1, wz | 1)

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

    thr = int(math.ceil(float(v.tau) * window_size))
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
# CLI
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Synthetic trabecular generator (connectivity-first v3 + bridging).")
    p.add_argument("--outdir", type=str, default="data/v3_connectivity_first")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--n-volumes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    # 2D lattice params
    p.add_argument("--bv-tv", type=float, default=0.18)
    p.add_argument("--pn", type=int, default=6)
    p.add_argument("--rn", type=int, default=12)

    # voting params
    p.add_argument("--k", type=int, default=9)
    p.add_argument("--win-x", type=int, default=5)
    p.add_argument("--win-y", type=int, default=5)
    p.add_argument("--win-z", type=int, default=5)
    p.add_argument("--tau", type=float, default=0.52)
    p.add_argument("--closing-iters", type=int, default=2)
    p.add_argument("--no-alt-close", action="store_true")
    p.add_argument("--no-z-bias", action="store_true", help="Disable continuity bias in voting.")

    # acceptance / connectivity controls
    p.add_argument("--min-lcc-frac", type=float, default=0.85,
                   help="Minimum largest connected component fraction (bone phase) to accept.")
    p.add_argument("--max-tries", type=int, default=8,
                   help="Attempts per volume to meet connectivity. Best candidate kept.")
    p.add_argument("--auto-tune", action="store_true",
                   help="Auto-adjust tau/closing slightly to improve connectivity during retries.")

    # NEW: bridging controls
    p.add_argument("--bridge-3d", action="store_true",
                   help="Actively bridge disconnected components in 3D after voting+closing.")
    p.add_argument("--max-bridges", type=int, default=3,
                   help="Maximum number of bridges to add per attempt.")
    p.add_argument("--bridge-radius", type=int, default=2,
                   help="Radius (voxels) of the bridging tube.")
    p.add_argument("--bridge-smooth", type=int, default=1,
                   help="Optional dilation iterations after bridging to smooth connections.")

    # render
    p.add_argument("--z-step-um", type=float, default=Z_STEP_UM)
    p.add_argument("--write-gray", action="store_true")
    p.add_argument("--soft-sigma-px", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.9)

    # Option 1: exports for PCA
    p.add_argument("--export-2d", action="store_true")
    p.add_argument("--export-2d-mode", type=str, default="all",
                   choices=["all", "mid", "mip", "mean"])

    # sweep mode (optional)
    p.add_argument("--sweep-tau", type=str, default="",
                   help="Comma-separated tau values (e.g. 0.45,0.5,0.55). Overrides --tau and runs a sweep.")
    p.add_argument("--sweep-win", type=str, default="",
                   help="Comma-separated odd window sizes (applies to x=y=z), e.g. 3,5,7. Optional.")

    return p


def parse_csv_floats(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_ints(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# -----------------------------
# Core generation (one volume with acceptance loop)
# -----------------------------
def generate_one_volume(
    volume_id: str,
    H: int,
    W: int,
    rng: np.random.Generator,
    lat_p: LatticeParams2D,
    vot_p: Voting3DParams,
    ren_p: RenderParams,
    args,
    outdir: Path,
) -> Dict[str, object]:
    best_candidate = None  # (score_tuple, voted01, gray_stack, metrics, lattice_seeds, used_params)

    tau_base = float(vot_p.tau)
    close_base = int(vot_p.closing_iters)

    for attempt in range(int(args.max_tries)):
        if args.auto_tune and attempt > 0:
            tau = max(0.40, tau_base - 0.02 * attempt)
            closing_iters = min(close_base + attempt // 2, close_base + 3)
        else:
            tau = tau_base
            closing_iters = close_base

        # 1) generate k 2D lattices
        k = int(vot_p.k_lattices)
        lattices = []
        lattice_seeds = []
        for zi in range(k):
            sub_seed = int(rng.integers(0, 2**31 - 1))
            lattice_seeds.append(sub_seed)
            sub_rng = np.random.default_rng(sub_seed)
            lattices.append(generate_lattice_2d(H, W, sub_rng, lat_p))

        stack01 = np.stack(lattices, axis=0).astype(np.uint8)

        # 2) vote
        vot_local = Voting3DParams(**asdict(vot_p))
        vot_local.tau = float(tau)
        vot_local.closing_iters = int(closing_iters)

        voted01 = vote_3d(stack01, vot_local)

        # 3) close
        voted01 = closing_3d(voted01, iters=vot_local.closing_iters, alternate=vot_local.alternate_structure)

        # NEW: bridge components (generator enforcement)
        if args.bridge_3d:
            voted01 = bridge_components_3d(
                voted01,
                max_bridges=int(args.max_bridges),
                radius=int(args.bridge_radius),
                smooth=int(args.bridge_smooth),
            )

        # 4) metrics
        bv_3d = float(np.mean(voted01 > 0))
        conn = connectivity_metrics_3d(voted01)
        prox = thickness_spacing_proxies(voted01)

        metrics = {
            "bv_tv_3d": bv_3d,
            "cc_count": conn["cc_count"],
            "lcc_frac": conn["lcc_frac"],
            "thickness_proxy": prox["thickness_proxy"],
            "spacing_proxy": prox["spacing_proxy"],
        }

        # 5) grayscale optional
        gray_stack = None
        if ren_p.write_gray:
            soft = gaussian_blur(voted01.astype(np.float32), sigma=float(ren_p.soft_sigma_px))
            soft = clamp01(soft) ** float(ren_p.partial_gamma)
            gray_stack = (255.0 * soft).astype(np.uint8)

        lcc = metrics["lcc_frac"] if metrics["lcc_frac"] is not None else 0.0
        score = (
            float(lcc),
            -abs(float(metrics["bv_tv_3d"]) - float(lat_p.bv_tv)),
            float(metrics["thickness_proxy"] or 0.0),
        )

        used_params = {
            "tau_used": float(tau),
            "closing_iters_used": int(closing_iters),
            "bridge_3d": bool(args.bridge_3d),
            "max_bridges": int(args.max_bridges),
            "bridge_radius": int(args.bridge_radius),
            "bridge_smooth": int(args.bridge_smooth),
        }

        if best_candidate is None or score > best_candidate[0]:
            best_candidate = (score, voted01, gray_stack, metrics, lattice_seeds, used_params)

        # acceptance
        if metrics["lcc_frac"] is not None and float(metrics["lcc_frac"]) >= float(args.min_lcc_frac):
            break

    assert best_candidate is not None
    _, voted01, gray_stack, metrics, lattice_seeds, used_params = best_candidate

    # exports (Option 1)
    exports = {
        "enabled": bool(args.export_2d),
        "mode": str(args.export_2d_mode),
        "mid_png": None,
        "mip_png": None,
        "mean_png": None,
        "gray_mid_png": None,
        "gray_mip_png": None,
    }

    if args.export_2d:
        zmid = voted01.shape[0] // 2
        mid_slice = (voted01[zmid] * 255).astype(np.uint8)
        mip_xy = (voted01.max(axis=0) * 255).astype(np.uint8)
        mean_xy = np.clip(voted01.mean(axis=0) * 255.0, 0, 255).astype(np.uint8)

        mode = str(args.export_2d_mode)
        if mode in ("all", "mid"):
            name = f"{volume_id}_mid.png"
            save_png(mid_slice, outdir / name)
            exports["mid_png"] = name
        if mode in ("all", "mip"):
            name = f"{volume_id}_mip.png"
            save_png(mip_xy, outdir / name)
            exports["mip_png"] = name
        if mode in ("all", "mean"):
            name = f"{volume_id}_mean.png"
            save_png(mean_xy, outdir / name)
            exports["mean_png"] = name

        if ren_p.write_gray and gray_stack is not None:
            name = f"{volume_id}_gray_mid.png"
            save_png(gray_stack[zmid].astype(np.uint8), outdir / name)
            exports["gray_mid_png"] = name
            name = f"{volume_id}_gray_mip.png"
            save_png(gray_stack.max(axis=0).astype(np.uint8), outdir / name)
            exports["gray_mip_png"] = name

    # save 3D outputs
    mask_path = outdir / f"{volume_id}_mask.tif"
    save_stack_u8((voted01 * 255).astype(np.uint8), mask_path, z_step_um=float(args.z_step_um))

    gray_path = None
    if ren_p.write_gray and gray_stack is not None:
        gray_path = outdir / f"{volume_id}_gray.tif"
        save_stack_u8(gray_stack, gray_path, z_step_um=float(args.z_step_um))

    # metadata
    meta = {
        "volume_id": volume_id,
        "files": {
            "mask_tif": mask_path.name,
            "gray_tif": gray_path.name if gray_path else None,
        },
        "exports_2d": exports,
        "globals": {
            "pixel_size_um": float(PIXEL_SIZE_UM),
            "z_step_um": float(args.z_step_um),
            "H": int(H),
            "W": int(W),
            "Z": int(voted01.shape[0]),
            "seed": int(args.seed),
        },
        "lattice_2d_params": asdict(lat_p),
        "voting_3d_params": asdict(vot_p),
        "render_params": asdict(ren_p),
        "lattice_seeds": lattice_seeds,
        "used_params": used_params,
        "metrics": metrics,
        "notes": [
            "Connectivity-first generator: LCC fraction is primary acceptance metric.",
            "Optional 3D bridging enforces global connectivity by drawing tubular connectors between largest components."
        ],
    }

    meta_path = outdir / f"{volume_id}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "mask_tif": mask_path.name,
        "gray_tif": gray_path.name if gray_path else "",
        "exports": exports,
        "metrics": metrics,
        "used_params": used_params,
        "meta_json": meta_path.name,
        "Z": int(voted01.shape[0]),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    args = build_parser().parse_args()
    rng = np.random.default_rng(int(args.seed))

    H = W = int(args.size)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # sweep parsing
    tau_list = parse_csv_floats(args.sweep_tau)
    win_list = parse_csv_ints(args.sweep_win)

    if win_list and any((w % 2) == 0 or w < 1 for w in win_list):
        raise ValueError("--sweep-win values must be odd positive integers (e.g. 3,5,7).")

    lat_p = LatticeParams2D(
        bv_tv=float(args.bv_tv),
        pn=int(args.pn),
        rn=int(args.rn),
    )

    ren_p = RenderParams(
        write_gray=bool(args.write_gray),
        soft_sigma_px=float(args.soft_sigma_px),
        partial_gamma=float(args.gamma),
    )

    images_csv = outdir / "volumes.csv"
    fieldnames = [
        "volume_id",
        "mask_tif",
        "gray_tif",
        "mid_png",
        "mip_png",
        "mean_png",
        "gray_mid_png",
        "gray_mip_png",
        "meta_json",
        "H", "W", "Z",
        "pixel_size_um",
        "z_step_um",
        "seed",
        "pn", "rn",
        "bv_tv_2d_target",
        "k",
        "win_x", "win_y", "win_z",
        "tau",
        "closing_iters",
        "alternate_close",
        "z_continuity_bias",
        "min_lcc_frac",
        "max_tries",
        "auto_tune",
        "bridge_3d",
        "max_bridges",
        "bridge_radius",
        "bridge_smooth",
        "tau_used",
        "closing_iters_used",
        "bv_tv_3d",
        "cc_count",
        "lcc_frac",
        "thickness_proxy",
        "spacing_proxy",
        "export_2d",
        "export_2d_mode",
        "sweep_tau",
        "sweep_win",
    ]
    f_img, w_img = init_csv(images_csv, fieldnames=fieldnames)

    # Build sweep grid
    if tau_list or win_list:
        if not tau_list:
            tau_list = [float(args.tau)]
        if not win_list:
            win_list = [int(args.win_x)]

        sweep_grid = [(float(t), int(w)) for t in tau_list for w in win_list]
        run_plan = [(t, w, i) for (t, w) in sweep_grid for i in range(int(args.n_volumes))]
        print(f"Sweep enabled: {len(sweep_grid)} conditions, total volumes planned: {len(run_plan)}")
    else:
        run_plan = [(float(args.tau), int(args.win_x), i) for i in range(int(args.n_volumes))]

    try:
        for idx, (tau_val, win_val, vi) in enumerate(run_plan):
            vot_p = Voting3DParams(
                k_lattices=int(args.k),
                win_x=int(win_val),
                win_y=int(win_val) if (tau_list or win_list) else int(args.win_y),
                win_z=int(win_val) if (tau_list or win_list) else int(args.win_z),
                tau=float(tau_val),
                closing_iters=int(args.closing_iters),
                alternate_structure=(not bool(args.no_alt_close)),
                z_continuity_bias=(not bool(args.no_z_bias)),
            )

            if tau_list or win_list:
                cond_dir = outdir / f"sweep_tau_{tau_val:.3f}_win_{win_val}"
                cond_dir.mkdir(parents=True, exist_ok=True)
                v_outdir = cond_dir
                volume_id = f"vol_{vi:05d}"
            else:
                v_outdir = outdir
                volume_id = f"vol_{vi:05d}"

            res = generate_one_volume(
                volume_id=volume_id,
                H=H,
                W=W,
                rng=rng,
                lat_p=lat_p,
                vot_p=vot_p,
                ren_p=ren_p,
                args=args,
                outdir=v_outdir,
            )

            exports = res["exports"]
            metrics = res["metrics"]
            used = res["used_params"]

            w_img.writerow({
                "volume_id": volume_id,
                "mask_tif": res["mask_tif"],
                "gray_tif": res["gray_tif"],
                "mid_png": exports["mid_png"] or "",
                "mip_png": exports["mip_png"] or "",
                "mean_png": exports["mean_png"] or "",
                "gray_mid_png": exports["gray_mid_png"] or "",
                "gray_mip_png": exports["gray_mip_png"] or "",
                "meta_json": res["meta_json"],
                "H": H,
                "W": W,
                "Z": res["Z"],
                "pixel_size_um": float(PIXEL_SIZE_UM),
                "z_step_um": float(args.z_step_um),
                "seed": int(args.seed),
                "pn": int(lat_p.pn),
                "rn": int(lat_p.rn),
                "bv_tv_2d_target": float(lat_p.bv_tv),
                "k": int(vot_p.k_lattices),
                "win_x": int(vot_p.win_x),
                "win_y": int(vot_p.win_y),
                "win_z": int(vot_p.win_z),
                "tau": float(vot_p.tau),
                "closing_iters": int(vot_p.closing_iters),
                "alternate_close": bool(vot_p.alternate_structure),
                "z_continuity_bias": bool(vot_p.z_continuity_bias),
                "min_lcc_frac": float(args.min_lcc_frac),
                "max_tries": int(args.max_tries),
                "auto_tune": bool(args.auto_tune),
                "bridge_3d": bool(args.bridge_3d),
                "max_bridges": int(args.max_bridges),
                "bridge_radius": int(args.bridge_radius),
                "bridge_smooth": int(args.bridge_smooth),
                "tau_used": float(used["tau_used"]),
                "closing_iters_used": int(used["closing_iters_used"]),
                "bv_tv_3d": float(metrics["bv_tv_3d"]),
                "cc_count": metrics["cc_count"],
                "lcc_frac": metrics["lcc_frac"],
                "thickness_proxy": metrics["thickness_proxy"],
                "spacing_proxy": metrics["spacing_proxy"],
                "export_2d": bool(args.export_2d),
                "export_2d_mode": str(args.export_2d_mode),
                "sweep_tau": bool(tau_list),
                "sweep_win": bool(win_list),
            })

            lcc_display = metrics["lcc_frac"]
            lcc_str = f"{lcc_display:.3f}" if isinstance(lcc_display, (float, int)) else "None"
            print(
                f"[{idx+1}/{len(run_plan)}] {volume_id} | BV/TV={metrics['bv_tv_3d']:.3f} | "
                f"LCC={lcc_str} | tau_used={used['tau_used']:.3f} | bridge={bool(args.bridge_3d)}"
            )

    finally:
        f_img.close()

    print(f"\nDone. Written to: {outdir}")
    if not _HAS_SCIPY:
        print(
            "Note: scipy not available → voting uses slow fallback and bridging/connectivity/EDT/morphology are limited.\n"
            "Install scipy for best results: pip install scipy"
        )


if __name__ == "__main__":
    main()
