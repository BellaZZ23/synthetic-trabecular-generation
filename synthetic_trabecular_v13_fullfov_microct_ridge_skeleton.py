#!/usr/bin/env python3
r"""
synthetic_trabecular_v13_fullfov_microct_ridge_skeleton.py

Ridge/skeleton-based trabecular generator (better topology than GRF threshold blobs).

Core idea:
  - Make smooth field
  - Compute Hessian eigenvalues -> vesselness/ridge response
  - Threshold ridge response -> thin network
  - Optional skeletonize -> cleaner medial network
  - Thicken network using distance transform threshold (controls thickness)
  - Tune BV/TV by adjusting thickness threshold

Outputs:
  - mid.png, mask.tif
  - gray_mid.png, gray.tif (optional)
  - metrics.json

Deps:
  numpy, scipy, pillow, tifffile, scikit-image
"""

from __future__ import annotations

import argparse, json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image
import tifffile as tiff
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from skimage.measure import euler_number
from skimage.morphology import skeletonize_3d


# -----------------------------
# Params
# -----------------------------
@dataclass
class RidgeParams:
    base_sigma: float = 3.8          # smooth field scale (controls spacing)
    warp_sigma: float = 14.0         # curvature smoothness
    warp_amp: float = 4.8            # curvature strength

    hessian_sigma: float = 1.4       # ridge detection scale (controls thinness)
    ridge_strength: float = 1.0      # ridge response gain
    ridge_quantile: float = 0.92     # higher -> thinner initial network

    use_skeleton: bool = True

    # Thickness control (in voxels): final bone = distance_to_network <= thick_thr
    thick_thr_vox: float = 1.2       # main knob for thickness
    thick_jitter: float = 0.25       # thickness variability (0..0.6)
    jitter_sigma: float = 6.0        # smooth jitter field scale

    # Connectivity preservation
    reconnect_close_iters: int = 3
    prune_small_components: int = 0  # set >0 to remove islands by size threshold (voxels)

@dataclass
class GrayParams:
    write_gray: bool = True
    # surface-weighted sharp µCT look
    marrow_mean: float = 15.0
    bone_mean: float = 240.0
    shell_sigma_vox: float = 0.9     # how quickly brightness falls off from boundary
    pve_sigma: float = 0.5           # PSF blur (keep small for sharp µCT)
    noise_sd: float = 3.0
    bg_tex_sd: float = 1.0
    unsharp: float = 0.6
    unsharp_sigma: float = 0.8


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

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


# -----------------------------
# Field + warp (curvature)
# -----------------------------
def normalize(f: np.ndarray) -> np.ndarray:
    x = f.astype(np.float32)
    x -= float(x.mean())
    x /= float(x.std() + 1e-6)
    return x

def smooth_warp(field: np.ndarray, rng: np.random.Generator, warp_sigma: float, warp_amp: float) -> np.ndarray:
    if warp_amp <= 0:
        return field
    dz = ndi.gaussian_filter(rng.normal(0,1,field.shape), sigma=warp_sigma) * warp_amp
    dy = ndi.gaussian_filter(rng.normal(0,1,field.shape), sigma=warp_sigma) * warp_amp
    dx = ndi.gaussian_filter(rng.normal(0,1,field.shape), sigma=warp_sigma) * warp_amp
    Z,Y,X = field.shape
    zz,yy,xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    coords = np.array([zz+dz, yy+dy, xx+dx])
    return map_coordinates(field, coords, order=1, mode="reflect").astype(np.float32)


# -----------------------------
# Hessian ridge response (Frangi-ish)
# -----------------------------
def hessian_eigs_3d(f: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fxx = ndi.gaussian_filter(f, sigma=sigma, order=(0,0,2))
    fyy = ndi.gaussian_filter(f, sigma=sigma, order=(0,2,0))
    fzz = ndi.gaussian_filter(f, sigma=sigma, order=(2,0,0))
    fxy = ndi.gaussian_filter(f, sigma=sigma, order=(0,1,1))
    fxz = ndi.gaussian_filter(f, sigma=sigma, order=(1,0,1))
    fyz = ndi.gaussian_filter(f, sigma=sigma, order=(1,1,0))

    H = np.stack(
        [
            np.stack([fzz, fyz, fxz], axis=-1),
            np.stack([fyz, fyy, fxy], axis=-1),
            np.stack([fxz, fxy, fxx], axis=-1),
        ],
        axis=-2
    )  # (...,3,3)

    w = np.linalg.eigvalsh(H.reshape(-1,3,3)).reshape(f.shape + (3,))
    idx = np.argsort(np.abs(w), axis=-1)
    w = np.take_along_axis(w, idx, axis=-1)
    l1, l2, l3 = w[...,0], w[...,1], w[...,2]  # |l1|<=|l2|<=|l3|
    return l1, l2, l3

def vesselness_ridge(f: np.ndarray, sigma: float) -> np.ndarray:
    """
    High response on ridge/line-like structures.
    Uses eigenvalue ratios (Frangi-like heuristic).
    """
    l1,l2,l3 = hessian_eigs_3d(f, sigma=sigma)
    eps = 1e-6

    # ratios: ridge when |l3| large, l1,l2 small-ish relative to l3
    r1 = (np.abs(l1) / (np.abs(l3) + eps))
    r2 = (np.abs(l2) / (np.abs(l3) + eps))

    # line-like (rod): both r1 and r2 small
    V = np.exp(-(r1*r1)/(0.5*0.5)) * np.exp(-(r2*r2)/(0.5*0.5))

    V = V.astype(np.float32)
    V = V / (float(V.max()) + 1e-6)
    return V


# -----------------------------
# Build bone network from ridges
# -----------------------------
def make_network(shape: Tuple[int,int,int], rp: RidgeParams, rng: np.random.Generator) -> np.ndarray:
    # base smooth field
    f = rng.normal(0,1,size=shape).astype(np.float32)
    f = ndi.gaussian_filter(f, sigma=float(rp.base_sigma))
    f = smooth_warp(f, rng, warp_sigma=float(rp.warp_sigma), warp_amp=float(rp.warp_amp))
    f = normalize(f)

    # ridge response
    R = vesselness_ridge(f, sigma=float(rp.hessian_sigma))
    R = np.clip(R * float(rp.ridge_strength), 0.0, 1.0)

    # thin network by quantile threshold
    q = float(np.clip(rp.ridge_quantile, 0.5, 0.995))
    thr = float(np.quantile(R, q))
    net = (R >= thr)

    # skeletonize (optional)
    if bool(rp.use_skeleton) and shape[0] >= 2:
        net = skeletonize_3d(net).astype(bool)

    # reconnect a bit (closing)
    if int(rp.reconnect_close_iters) > 0:
        st = ndi.generate_binary_structure(3,1)
        net = ndi.binary_closing(net, structure=st, iterations=int(rp.reconnect_close_iters))

    return net.astype(np.uint8), {"ridge_q": q, "ridge_thr": thr}


def thicken_network_to_bone(net01: np.ndarray, rp: RidgeParams, target_bvtv: float, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert thin network -> bone mask using distance-to-network threshold.
    We tune the threshold to hit BV/TV automatically (binary search).
    """
    net = net01.astype(bool)

    # distance from network
    dist = ndi.distance_transform_edt(~net)  # distance to nearest net voxel

    # smooth jitter field to vary thickness
    if float(rp.thick_jitter) > 0:
        j = rng.normal(0,1,size=dist.shape).astype(np.float32)
        j = ndi.gaussian_filter(j, sigma=float(rp.jitter_sigma))
        j = j / (float(j.std()) + 1e-6)
        jitter = float(rp.thick_jitter) * j
    else:
        jitter = 0.0

    # binary search thickness threshold to match BV/TV
    target = float(np.clip(target_bvtv, 0.01, 0.95))
    lo, hi = 0.2, max(0.8, float(rp.thick_thr_vox) * 3.5)

    best_thr = float(rp.thick_thr_vox)
    for _ in range(18):
        mid = 0.5*(lo+hi)
        bone = (dist <= (mid + jitter))
        bvtv = float(bone.mean())
        if bvtv < target:
            lo = mid
        else:
            hi = mid
        best_thr = mid

    bone = (dist <= (best_thr + jitter))

    # optional pruning of small islands
    if int(rp.prune_small_components) > 0:
        st = ndi.generate_binary_structure(3,1)
        lab, n = ndi.label(bone, structure=st)
        if n > 0:
            counts = np.bincount(lab.ravel())
            keep = np.zeros_like(counts, dtype=bool)
            keep[counts >= int(rp.prune_small_components)] = True
            keep[0] = False
            bone = keep[lab]

    return bone.astype(np.uint8), {"thick_thr_fit": float(best_thr)}


# -----------------------------
# µCT sharp grayscale (surface-weighted)
# -----------------------------
def microct_gray_surface(bone01: np.ndarray, gp: GrayParams, rng: np.random.Generator) -> np.ndarray:
    bone = bone01.astype(bool)

    # boundary emphasis: brightness highest near boundary
    # distance inside bone
    d_in = ndi.distance_transform_edt(bone).astype(np.float32)
    shell = np.exp(- (d_in / max(0.2, float(gp.shell_sigma_vox)))**2 )
    shell = shell * bone.astype(np.float32)

    gray = float(gp.marrow_mean) + shell * (float(gp.bone_mean) - float(gp.marrow_mean))

    # partial volume blur (small)
    if float(gp.pve_sigma) > 0:
        gray = ndi.gaussian_filter(gray, sigma=float(gp.pve_sigma))

    # texture + noise
    if float(gp.bg_tex_sd) > 0:
        gray += rng.normal(0.0, float(gp.bg_tex_sd), size=gray.shape).astype(np.float32)
    if float(gp.noise_sd) > 0:
        gray += rng.normal(0.0, float(gp.noise_sd), size=gray.shape).astype(np.float32)

    # unsharp for crisp edges
    if float(gp.unsharp) > 0:
        blurred = ndi.gaussian_filter(gray, sigma=max(0.4, float(gp.unsharp_sigma)))
        gray = gray + float(gp.unsharp) * (gray - blurred)

    return np.clip(gray, 0, 255).astype(np.uint8)


# -----------------------------
# Metrics
# -----------------------------
def bvtv(vol01: np.ndarray) -> float:
    return float(np.mean(vol01 > 0))

def thickness_pcts_um(vol01: np.ndarray, voxel_um: float) -> Dict[str, float]:
    bone = vol01.astype(bool)
    if bone.sum() == 0 or (~bone).sum() == 0:
        return {"TbTh_p50": 0.0, "TbTh_p90": 0.0, "TbSp_p50": 0.0, "TbSp_p90": 0.0}
    dt_b = ndi.distance_transform_edt(bone) * float(voxel_um)
    dt_m = ndi.distance_transform_edt(~bone) * float(voxel_um)
    tbth = dt_b[bone]
    tbsp = dt_m[~bone]
    return {
        "TbTh_p50": float(np.percentile(tbth, 50)),
        "TbTh_p90": float(np.percentile(tbth, 90)),
        "TbSp_p50": float(np.percentile(tbsp, 50)),
        "TbSp_p90": float(np.percentile(tbsp, 90)),
    }

def euler_conn(vol01: np.ndarray) -> Dict[str, float]:
    e = float(euler_number(vol01.astype(bool), connectivity=3))
    return {"Euler": e, "ConnProxy": float(1.0 - e)}


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v13 ridge/skeleton trabecular generator (bone-like topology).")
    p.add_argument("--outdir", type=str, default="data/v13_ridge")
    p.add_argument("--xy", type=int, default=512)
    p.add_argument("--z", type=int, default=160)
    p.add_argument("--seed", type=int, default=23)

    p.add_argument("--bvtv", type=float, default=0.18)
    p.add_argument("--invert-phase", type=int, default=0)  # kept for compatibility; ridge pipeline assumes bone=1

    p.add_argument("--voxel-um", type=float, default=39.0)
    p.add_argument("--priors-json", type=str, default=None,
               help="Path to aggregated priors JSON (VOI1+VOI4)")


    # Ridge knobs
    p.add_argument("--base-sigma", type=float, default=3.8)
    p.add_argument("--warp-sigma", type=float, default=14.0)
    p.add_argument("--warp-amp", type=float, default=4.8)
    p.add_argument("--hessian-sigma", type=float, default=1.4)
    p.add_argument("--ridge-q", type=float, default=0.92)
    p.add_argument("--ridge-strength", type=float, default=1.0)
    p.add_argument("--thick-jitter", type=float, default=0.25)
    p.add_argument("--thick-thr-vox", type=float, default=1.2)
    p.add_argument("--reconnect-close-iters", type=int, default=3)
    p.add_argument("--use-skeleton", type=int, default=1)

    # Gray
    p.add_argument("--write-gray", type=int, default=1)
    return p

def main():
    args = build_parser().parse_args()
    rng = np.random.default_rng(int(args.seed))

    # -----------------------------
# Load priors if provided
# -----------------------------
if args.priors_json is not None:
    pri_path = Path(args.priors_json)
    if pri_path.exists():
        print(f"Loading priors from: {pri_path}")
        with open(pri_path, "r") as f:
            pri = json.load(f)

        # --- BVTV ---
        if "BVTV" in pri:
            args.bvtv = float(pri["BVTV"])
            print(f"  BVTV -> {args.bvtv:.3f}")

        # --- Thickness (convert microns to voxels) ---
        if "tbth_um_p90" in pri:
            tbth_um = float(pri["tbth_um_p90"])
            args.thick_thr_vox = (tbth_um / float(args.voxel_um)) * 0.45
            print(f"  thick_thr_vox -> {args.thick_thr_vox:.2f}")

        # --- Connectivity tuning ---
        if "euler" in pri:
            eul = float(pri["euler"])

            # more negative Euler = more connected structure
            if eul < -1000:
                args.ridge_q = max(0.75, args.ridge_q - 0.03)
                args.reconnect_close_iters += 2
                print("  Connectivity increased (VOI4-like)")
    else:
        print(f"Priors file not found: {pri_path}")


    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    shape = (int(args.z), int(args.xy), int(args.xy))

    rp = RidgeParams(
        base_sigma=float(args.base_sigma),
        warp_sigma=float(args.warp_sigma),
        warp_amp=float(args.warp_amp),
        hessian_sigma=float(args.hessian_sigma),
        ridge_strength=float(args.ridge_strength),
        ridge_quantile=float(args.ridge_q),
        use_skeleton=bool(int(args.use_skeleton)),
        thick_thr_vox=float(args.thick_thr_vox),
        thick_jitter=float(args.thick_jitter),
        reconnect_close_iters=int(args.reconnect_close_iters),
    )

    gp = GrayParams(write_gray=bool(int(args.write_gray)))

    # 1) thin network
    net01, net_info = make_network(shape, rp, rng)

    # 2) thicken to match BVTV
    bone01, thick_info = thicken_network_to_bone(net01, rp, target_bvtv=float(args.bvtv), rng=rng)

    # Save outputs
    Z = shape[0]
    save_tif_u8((bone01 * 255).astype(np.uint8), outdir / "mask.tif")
    save_png_u8((bone01[Z // 2] * 255).astype(np.uint8), outdir / "mid.png")

    if gp.write_gray:
        gray = microct_gray_surface(bone01, gp, rng)
        save_tif_u8(gray, outdir / "gray.tif")
        save_png_u8(gray[Z // 2], outdir / "gray_mid.png")

    met = {
        "BVTV": bvtv(bone01),
        **thickness_pcts_um(bone01, voxel_um=float(args.voxel_um)),
        **euler_conn(bone01),
        "net_info": net_info,
        "thick_info": thick_info,
        "params": {"ridge": asdict(rp), "gray": asdict(gp)},
        "shape_zyx": list(shape),
    }
    save_json(met, outdir / "metrics.json")

    print(
        f"Saved to {outdir}\n"
        f"BVTV={met['BVTV']:.3f} | TbTh(p90)={met['TbTh_p90']:.1f}um | Euler={met['Euler']:.1f}"
    )

if __name__ == "__main__":
    main()
