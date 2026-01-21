#!/usr/bin/env python3

"""
synthetic_microct_trabecular_ws.py

A NEW generator script based on the methodology in:
Grande-Barreto et al., "Generation of Synthetic Images of Trabecular Bone Based on Micro-CT Scans"
(Information 2022, 14, 375)

Pipeline (matches the paper's modular design):
1) Small-world (Watts–Strogatz-like) 2D graph -> rasterize to binary "SW" images
2) Stack SW images into 3D and apply 3D sliding-window voting (binary template model):
      S(x,y,z)=1 if sum(window)/window_size >= rho else 0
3) Apply alternating 3D morphological closing to reduce artifacts
4) Projection model (Beer–Lambert inspired):
      gray = 255 * (1 - exp(-mu * sum_{slab thickness Nz} S))
   (produces brighter intensities where more bone exists)
5) Add Gaussian white noise (micro-CT typical; SD fraction suggested 0.1–0.9 in the paper)
6) Save:
   - 3D binary mask (TIFF)
   - 3D micro-CT-like grayscale stack (TIFF)
   - optional 2D PNGs (mid, MIP, mean)
   - per-volume JSON + CSV log

Notes:
- This implements the paper’s idea of small-world + morphology + voting + Beer–Lambert + noise.
- It does not require a real template/dataset (generic synthetic micro-CT).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List, Set

import numpy as np
from PIL import Image
import tifffile as tiff

try:
    from scipy import ndimage as ndi
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# -----------------------------
# Parameters
# -----------------------------
@dataclass
class SmallWorldParams:
    n_nodes: int = 90               # number of nodes in the WS graph
    k_neighbors: int = 4            # each node connects to k neighbors (must be even)
    beta_rewire: float = 0.35       # rewiring probability (controls randomness)
    node_radius_px: int = 2
    edge_radius_px: int = 1
    node_margin_frac: float = 0.08  # keep nodes away from borders


@dataclass
class BinaryTemplateParams:
    # Volume size
    X: int = 256
    Y: int = 256
    Z: int = 150

    # Voting window (Omega)
    win_x: int = 5
    win_y: int = 5
    win_z: int = 10

    # Density threshold (rho)
    rho: float = 0.35

    # Morphology
    closing_iters: int = 2
    alternate_structure: bool = True


@dataclass
class ProjectionParams:
    mu: float = 0.08                 # attenuation coefficient (cm^-1) - paper suggests 0.08 @ ~100keV
    nz: int = 3                      # slab thickness for projection model (Nz); odd recommended
    gamma: float = 1.0               # optional contrast shaping
    noise_sd_frac: float = 0.2       # Gaussian noise SD as fraction of 255 (paper suggests 0.1 to 0.9)


# -----------------------------
# Helpers
# -----------------------------
def clamp01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)

def save_stack_u8(stack_u8: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(out_path, stack_u8.astype(np.uint8), imagej=True, dtype=np.uint8)

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

def rasterize_disk(H: int, W: int, cx: float, cy: float, r: int) -> np.ndarray:
    yy, xx = np.ogrid[0:H, 0:W]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    out = np.zeros((H, W), dtype=np.uint8)
    out[mask] = 1
    return out

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

def oddify(n: int) -> int:
    n = int(max(1, n))
    return n | 1


# -----------------------------
# Small-world WS-like graph generation
# -----------------------------
def ws_graph_edges(n: int, k: int, beta: float, rng: np.random.Generator) -> Set[Tuple[int, int]]:
    """
    Simple Watts–Strogatz-style graph:
    - start with ring lattice: each i connects to i+1..i+k/2
    - rewire each edge (i, j) with probability beta to (i, new_j)
    Returns undirected edges as (min, max).
    """
    if k % 2 != 0:
        raise ValueError("k_neighbors must be even for WS ring lattice.")

    edges: Set[Tuple[int, int]] = set()

    # initial ring lattice
    half = k // 2
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))

    # rewiring
    # iterate over original directed-ish edges (i->i+d) and rewire only those
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            a0, b0 = (i, j) if i < j else (j, i)
            if (a0, b0) not in edges:
                continue

            if rng.random() < beta:
                # remove old edge
                edges.remove((a0, b0))

                # pick new target
                forbidden = {i}
                # avoid existing neighbors of i
                for (a, b) in edges:
                    if a == i:
                        forbidden.add(b)
                    elif b == i:
                        forbidden.add(a)

                # sample new_j not in forbidden
                candidates = [x for x in range(n) if x not in forbidden]
                if not candidates:
                    # revert if stuck
                    edges.add((a0, b0))
                    continue

                new_j = int(rng.choice(candidates))
                a1, b1 = (i, new_j) if i < new_j else (new_j, i)
                edges.add((a1, b1))

    return edges


def rasterize_small_world_image(H: int, W: int, sw: SmallWorldParams, rng: np.random.Generator) -> np.ndarray:
    """
    Create a 2D binary image from a WS-like small-world graph:
    - sample node positions
    - draw nodes as disks
    - draw edges as thick segments
    """
    # node positions
    margin_x = sw.node_margin_frac * W
    margin_y = sw.node_margin_frac * H
    xs = rng.uniform(margin_x, W - margin_x, size=sw.n_nodes)
    ys = rng.uniform(margin_y, H - margin_y, size=sw.n_nodes)
    pts = list(zip(xs, ys))

    edges = ws_graph_edges(sw.n_nodes, sw.k_neighbors, sw.beta_rewire, rng)

    img = np.zeros((H, W), dtype=np.uint8)

    # edges
    for (u, v) in edges:
        x0, y0 = pts[u]
        x1, y1 = pts[v]
        seg = rasterize_thick_segment(H, W, (x0, y0), (x1, y1), radius_px=float(sw.edge_radius_px))
        img = np.maximum(img, seg)

    # nodes
    for (x, y) in pts:
        d = rasterize_disk(H, W, x, y, int(sw.node_radius_px))
        img = np.maximum(img, d)

    return img


# -----------------------------
# Binary template model: 3D voting + closing
# -----------------------------
def vote_3d(stack01: np.ndarray, rho: float, win_x: int, win_y: int, win_z: int) -> np.ndarray:
    """
    stack01: uint8 (Z,Y,X)
    S(x,y,z)=1 if average over window >= rho else 0
    """
    win_x = oddify(win_x)
    win_y = oddify(win_y)
    win_z = oddify(win_z)

    kernel = np.ones((win_z, win_y, win_x), dtype=np.uint8)
    window_size = int(kernel.size)

    if _HAS_SCIPY:
        votes = ndi.convolve(stack01.astype(np.uint16), kernel.astype(np.uint16), mode="constant", cval=0)
    else:
        # fallback (slow)
        Z, Y, X = stack01.shape
        padz, pady, padx = win_z // 2, win_y // 2, win_x // 2
        padded = np.pad(stack01, ((padz, padz), (pady, pady), (padx, padx)), mode="constant")
        votes = np.zeros_like(stack01, dtype=np.uint16)
        for dz in range(win_z):
            for dy in range(win_y):
                for dx in range(win_x):
                    votes += padded[dz:dz+Z, dy:dy+Y, dx:dx+X]

    thr = int(math.ceil(float(rho) * window_size))
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
# Projection model (Beer–Lambert-inspired) + noise
# -----------------------------
def microct_projection_stack(vol01: np.ndarray, mu: float, nz: int, gamma: float, noise_sd_frac: float,
                            rng: np.random.Generator) -> np.ndarray:
    """
    Create a micro-CT-like grayscale stack from a binary volume using a slab Beer–Lambert model.

    For each z:
      sum_slab(x,y) = sum_{z' in slab} S(x,y,z')
      gray = 255 * (1 - exp(-mu * sum_slab))

    Then apply gamma and add Gaussian noise (SD = noise_sd_frac * 255).
    """
    Z, Y, X = vol01.shape
    nz = oddify(nz)
    r = nz // 2

    # pad in z
    padded = np.pad(vol01.astype(np.float32), ((r, r), (0, 0), (0, 0)), mode="constant")
    # cumulative sum for fast slab sums
    csum = np.cumsum(padded, axis=0)  # (Z+2r, Y, X)
    # slab sum for each z: sum(z..z+nz-1) = csum[z+nz] - csum[z]
    slab_sum = csum[nz:] - csum[:-nz]  # (Z, Y, X)

    # Beer–Lambert inspired mapping (brighter with more bone)
    gray = 1.0 - np.exp(-float(mu) * slab_sum)
    gray = clamp01(gray)

    if gamma != 1.0:
        gray = gray ** float(gamma)

    gray_u8 = (255.0 * gray).astype(np.float32)

    # Gaussian white noise
    sd = float(noise_sd_frac) * 255.0
    if sd > 0:
        noise = rng.normal(0.0, sd, size=gray_u8.shape).astype(np.float32)
        gray_u8 = gray_u8 + noise

    gray_u8 = np.clip(gray_u8, 0.0, 255.0).astype(np.uint8)
    return gray_u8


# -----------------------------
# CLI / main
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Synthetic micro-CT trabecular generator (WS + voting + Beer–Lambert + noise).")
    p.add_argument("--outdir", type=str, default="data/ws_microct")
    p.add_argument("--n-volumes", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    # Volume size
    p.add_argument("--x", type=int, default=256)
    p.add_argument("--y", type=int, default=256)
    p.add_argument("--z", type=int, default=150)

    # Small-world params
    p.add_argument("--n-nodes", type=int, default=90)
    p.add_argument("--k-neighbors", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.35)
    p.add_argument("--node-radius", type=int, default=2)
    p.add_argument("--edge-radius", type=int, default=1)

    # Binary template model params
    p.add_argument("--rho", type=float, default=0.35, help="Density threshold for voting (paper: 0.35, 0.38).")
    p.add_argument("--win-x", type=int, default=5)
    p.add_argument("--win-y", type=int, default=5)
    p.add_argument("--win-z", type=int, default=10)
    p.add_argument("--closing-iters", type=int, default=2)
    p.add_argument("--no-alt-close", action="store_true")

    # Projection model params
    p.add_argument("--mu", type=float, default=0.08, help="Attenuation coefficient (cm^-1).")
    p.add_argument("--nz", type=int, default=3, help="Projection slab thickness Nz (odd recommended).")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--noise-sd", type=float, default=0.2, help="Noise SD fraction of 255 (paper suggests 0.1–0.9).")

    # Exports
    p.add_argument("--export-2d", action="store_true", help="Export mid/mip/mean PNGs for binary+grayscale.")
    p.add_argument("--export-2d-mode", type=str, default="all", choices=["all", "mid", "mip", "mean"])

    return p


def main():
    args = build_parser().parse_args()
    rng = np.random.default_rng(int(args.seed))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sw = SmallWorldParams(
        n_nodes=int(args.n_nodes),
        k_neighbors=int(args.k_neighbors),
        beta_rewire=float(args.beta),
        node_radius_px=int(args.node_radius),
        edge_radius_px=int(args.edge_radius),
    )

    bt = BinaryTemplateParams(
        X=int(args.x),
        Y=int(args.y),
        Z=int(args.z),
        win_x=int(args.win_x),
        win_y=int(args.win_y),
        win_z=int(args.win_z),
        rho=float(args.rho),
        closing_iters=int(args.closing_iters),
        alternate_structure=(not bool(args.no_alt_close)),
    )

    pr = ProjectionParams(
        mu=float(args.mu),
        nz=int(args.nz),
        gamma=float(args.gamma),
        noise_sd_frac=float(args.noise_sd),
    )

    csv_path = outdir / "volumes.csv"
    f_csv, w_csv = init_csv(csv_path, fieldnames=[
        "volume_id",
        "mask_tif",
        "microct_tif",
        "mid_png",
        "mip_png",
        "mean_png",
        "gray_mid_png",
        "gray_mip_png",
        "X","Y","Z",
        "rho","win_x","win_y","win_z",
        "mu","nz","gamma","noise_sd",
        "n_nodes","k_neighbors","beta","node_radius","edge_radius",
        "seed",
        "bv_tv_3d",
    ])

    try:
        for i in range(int(args.n_volumes)):
            volume_id = f"vol_{i:05d}"
            # 1) build Z small-world images
            sw_imgs = []
            sw_seeds = []
            for zi in range(bt.Z):
                sub_seed = int(rng.integers(0, 2**31 - 1))
                sw_seeds.append(sub_seed)
                sub_rng = np.random.default_rng(sub_seed)
                img2d = rasterize_small_world_image(bt.Y, bt.X, sw, sub_rng)  # (Y,X)
                sw_imgs.append(img2d)

            stack01 = np.stack(sw_imgs, axis=0).astype(np.uint8)  # (Z,Y,X)

            # 2) voting to create binary template
            vol01 = vote_3d(stack01, rho=bt.rho, win_x=bt.win_x, win_y=bt.win_y, win_z=bt.win_z)

            # 3) closing
            vol01 = closing_3d(vol01, iters=bt.closing_iters, alternate=bt.alternate_structure)

            # 4) micro-CT grayscale via projection model + 5) noise
            gray = microct_projection_stack(vol01, mu=pr.mu, nz=pr.nz, gamma=pr.gamma,
                                            noise_sd_frac=pr.noise_sd_frac, rng=rng)

            # metrics
            bv_tv_3d = float(np.mean(vol01 > 0))

            # save TIFFs
            mask_path = outdir / f"{volume_id}_mask.tif"
            microct_path = outdir / f"{volume_id}_microct.tif"
            save_stack_u8((vol01 * 255).astype(np.uint8), mask_path)
            save_stack_u8(gray, microct_path)

            # optional PNG exports
            mid_png = mip_png = mean_png = ""
            gray_mid_png = gray_mip_png = ""
            if args.export_2d:
                zmid = vol01.shape[0] // 2
                bin_mid = (vol01[zmid] * 255).astype(np.uint8)
                bin_mip = (vol01.max(axis=0) * 255).astype(np.uint8)
                bin_mean = np.clip(vol01.mean(axis=0) * 255.0, 0, 255).astype(np.uint8)

                mode = str(args.export_2d_mode)
                if mode in ("all", "mid"):
                    mid_png = f"{volume_id}_mid.png"
                    save_png(bin_mid, outdir / mid_png)
                if mode in ("all", "mip"):
                    mip_png = f"{volume_id}_mip.png"
                    save_png(bin_mip, outdir / mip_png)
                if mode in ("all", "mean"):
                    mean_png = f"{volume_id}_mean.png"
                    save_png(bin_mean, outdir / mean_png)

                # grayscale exports (mid + mip)
                gray_mid_png = f"{volume_id}_gray_mid.png"
                gray_mip_png = f"{volume_id}_gray_mip.png"
                save_png(gray[zmid], outdir / gray_mid_png)
                save_png(gray.max(axis=0), outdir / gray_mip_png)

            # JSON metadata
            meta = {
                "volume_id": volume_id,
                "files": {
                    "mask_tif": mask_path.name,
                    "microct_tif": microct_path.name,
                    "mid_png": mid_png or None,
                    "mip_png": mip_png or None,
                    "mean_png": mean_png or None,
                    "gray_mid_png": gray_mid_png or None,
                    "gray_mip_png": gray_mip_png or None,
                },
                "params": {
                    "small_world": asdict(sw),
                    "binary_template": asdict(bt),
                    "projection": asdict(pr),
                },
                "seeds": {
                    "seed": int(args.seed),
                    "sw_seeds": sw_seeds,
                },
                "metrics": {
                    "bv_tv_3d": bv_tv_3d,
                },
            }
            with open(outdir / f"{volume_id}.json", "w") as f:
                json.dump(meta, f, indent=2)

            # CSV row
            w_csv.writerow({
                "volume_id": volume_id,
                "mask_tif": mask_path.name,
                "microct_tif": microct_path.name,
                "mid_png": mid_png,
                "mip_png": mip_png,
                "mean_png": mean_png,
                "gray_mid_png": gray_mid_png,
                "gray_mip_png": gray_mip_png,
                "X": bt.X, "Y": bt.Y, "Z": bt.Z,
                "rho": bt.rho, "win_x": bt.win_x, "win_y": bt.win_y, "win_z": bt.win_z,
                "mu": pr.mu, "nz": pr.nz, "gamma": pr.gamma, "noise_sd": pr.noise_sd_frac,
                "n_nodes": sw.n_nodes, "k_neighbors": sw.k_neighbors, "beta": sw.beta_rewire,
                "node_radius": sw.node_radius_px, "edge_radius": sw.edge_radius_px,
                "seed": int(args.seed),
                "bv_tv_3d": bv_tv_3d,
            })

            print(f"[{i+1}/{args.n_volumes}] {volume_id} | BV/TV={bv_tv_3d:.3f} | rho={bt.rho:.3f} | noiseSD={pr.noise_sd_frac:.2f}")

    finally:
        f_csv.close()

    if not _HAS_SCIPY:
        print("Note: scipy not available → voting uses slow fallback and 3D closing is skipped.")


if __name__ == "__main__":
    main()
