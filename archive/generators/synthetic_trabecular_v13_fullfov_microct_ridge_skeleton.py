#!/usr/bin/env python3
r"""
synthetic_trabecular_v13_1_fullfov_microct_ridge_skeleton_robust.py

v13.1 ridge → proto-network → skeleton → radius-field thickening trabecular generator.

Fix in this update:
  - Fiji skeletonization now uses Fiji launcher-friendly `--run` with a tiny Jython script
    (works with fiji-windows-x64.exe). This avoids `-macro`, which is unreliable / unsupported
    in some Fiji launchers.

What’s new vs your v13:
  A) Proto-network stage (hysteresis threshold + cleanup) BEFORE skeletonization
  B) Robust skeletonization modes:
       --skeleton-mode skimage (default, in-Python)
       --skeleton-mode fiji   (optional; calls Fiji headless to run Skeletonize 3D / BoneJ)
  C) Skeleton pruning (spur removal) via endpoint-to-junction geodesic distance
  D) Radius-field thickening around skeleton (Tb.Th-driven), with BV/TV fit by scaling radius field
  E) Debug outputs (proto_network.tif, skeleton.tif, etc.) via --debug-skeleton

Outputs:
  mask.tif, mid.png, (optional) gray.tif, gray_mid.png, metrics.json
  plus debug intermediates when --debug-skeleton 1:
    ridge_response.tif, proto_network.tif, skeleton_raw.tif, skeleton_pruned.tif, radius_field_u8.tif
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image
import tifffile as tiff
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from skimage.measure import euler_number

# Skeleton import (Python fallback)
try:
    from skimage.morphology import skeletonize_3d  # type: ignore
except Exception:
    skeletonize_3d = None


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

    # Proto-network hysteresis thresholding on ridge response
    proto_q_hi: float = 0.94         # strong ridges
    proto_q_lo: float = 0.86         # weak ridges allowed if connected to strong
    proto_close_iters: int = 1       # close small gaps BEFORE skeletonize
    proto_open_iters: int = 0        # optional speckle removal
    proto_min_component: int = 600   # remove tiny components BEFORE skeletonize (voxels)

    # Skeleton
    use_skeleton: bool = True
    skeleton_prune_lmin: int = 10    # prune endpoint branches shorter than this (vox)
    reconnect_close_iters: int = 2   # light close AFTER skeletonize (optional)

    # Radius-field thickness model
    radius_mode: str = "branch"      # "branch" or "voxel"
    radius_jitter: float = 0.25      # relative jitter of radius at skeleton points (0..0.6)
    radius_smooth_sigma: float = 2.0 # smooth radius values along skeleton
    radius_scale_hint: float = 1.0   # initial scaling of radius field (BV/TV fitting will adjust)

    # Post-processing
    prune_small_components: int = 0  # after final mask; 0 disables


@dataclass
class GrayParams:
    write_gray: bool = True
    marrow_mean: float = 15.0
    bone_mean: float = 240.0
    shell_sigma_vox: float = 0.9
    pve_sigma: float = 0.5
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
    dz = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma) * warp_amp
    dy = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma) * warp_amp
    dx = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma) * warp_amp

    Z, Y, X = field.shape
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    coords = np.array([zz + dz, yy + dy, xx + dx])
    return map_coordinates(field, coords, order=1, mode="reflect").astype(np.float32)


# -----------------------------
# Hessian ridge response (Frangi-ish)
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
    w = np.take_along_axis(w, idx, axis=-1)
    l1, l2, l3 = w[..., 0], w[..., 1], w[..., 2]  # |l1|<=|l2|<=|l3|
    return l1, l2, l3

def vesselness_ridge(f: np.ndarray, sigma: float) -> np.ndarray:
    l1, l2, l3 = hessian_eigs_3d(f, sigma=sigma)
    eps = 1e-6
    r1 = (np.abs(l1) / (np.abs(l3) + eps))
    r2 = (np.abs(l2) / (np.abs(l3) + eps))
    V = np.exp(-(r1 * r1) / (0.5 * 0.5)) * np.exp(-(r2 * r2) / (0.5 * 0.5))
    V = V.astype(np.float32)
    V = V / (float(V.max()) + 1e-6)
    return V


# -----------------------------
# Anti-block rounding
# -----------------------------
def anti_block_round(bone01: np.ndarray, sigma: float) -> np.ndarray:
    if float(sigma) <= 0:
        return bone01.astype(np.uint8)
    x = bone01.astype(np.float32)
    x = ndi.gaussian_filter(x, sigma=float(sigma))
    return (x >= 0.5).astype(np.uint8)


# -----------------------------
# Proto-network hysteresis + cleanup
# -----------------------------
def hysteresis_on_response(R: np.ndarray, q_lo: float, q_hi: float) -> Tuple[np.ndarray, Dict[str, float]]:
    q_lo = float(np.clip(q_lo, 0.5, 0.995))
    q_hi = float(np.clip(q_hi, q_lo + 1e-3, 0.999))
    thr_hi = float(np.quantile(R, q_hi))
    thr_lo = float(np.quantile(R, q_lo))

    strong = (R >= thr_hi)
    weak = (R >= thr_lo)

    st26 = ndi.generate_binary_structure(3, 2)
    lab, n = ndi.label(weak, structure=st26)
    if n == 0:
        return strong.astype(np.uint8), {"thr_lo": thr_lo, "thr_hi": thr_hi}

    strong_labels = np.unique(lab[strong])
    keep = np.zeros(n + 1, dtype=bool)
    keep[strong_labels] = True
    keep[0] = False
    out = keep[lab]

    return out.astype(np.uint8), {"thr_lo": thr_lo, "thr_hi": thr_hi}

def remove_small_components(vol: np.ndarray, min_size: int) -> np.ndarray:
    if int(min_size) <= 0:
        return vol.astype(np.uint8)
    st26 = ndi.generate_binary_structure(3, 2)
    lab, n = ndi.label(vol.astype(bool), structure=st26)
    if n == 0:
        return vol.astype(np.uint8)
    counts = np.bincount(lab.ravel())
    keep = counts >= int(min_size)
    keep[0] = False
    return keep[lab].astype(np.uint8)

def morph_iters(vol: np.ndarray, op: str, iters: int) -> np.ndarray:
    if int(iters) <= 0:
        return vol.astype(np.uint8)
    st26 = ndi.generate_binary_structure(3, 2)
    x = vol.astype(bool)
    if op == "close":
        x = ndi.binary_closing(x, structure=st26, iterations=int(iters))
    elif op == "open":
        x = ndi.binary_opening(x, structure=st26, iterations=int(iters))
    else:
        raise ValueError(f"Unknown op: {op}")
    return x.astype(np.uint8)


# -----------------------------
# Skeletonization (skimage or Fiji)
# -----------------------------
def skeletonize_with_skimage(proto01: np.ndarray) -> np.ndarray:
    if skeletonize_3d is None:
        raise RuntimeError("skimage.skeletonize_3d unavailable; install scikit-image or use --skeleton-mode fiji.")
    return skeletonize_3d(proto01.astype(bool)).astype(np.uint8)

def skeletonize_with_fiji(
    proto01: np.ndarray,
    fiji_exe: str,
    outdir: Path,
    command_name: str = "Skeletonize (2D/3D)",
) -> np.ndarray:
    """
    Fiji launcher-friendly skeletonization:
      - Write input as TIFF
      - Run Fiji with: --headless --run <jython_script.py>
      - Jython script opens image, runs ImageJ command, saves output.

    This works with `fiji-windows-x64.exe` (Jaunch launcher).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    in_tif = outdir / "proto_network_for_fiji.tif"
    out_tif = outdir / "skeleton_from_fiji.tif"
    save_tif_u8((proto01 * 255).astype(np.uint8), in_tif)

    jython = f"""
from ij import IJ
from ij.io import FileSaver

in_path = r"{in_tif.as_posix()}"
out_path = r"{out_tif.as_posix()}"
cmd = r"{command_name}"

imp = IJ.openImage(in_path)
if imp is None:
    raise Exception("Failed to open image: " + in_path)

IJ.run(imp, cmd, "")

fs = FileSaver(imp)
ok = fs.saveAsTiff(out_path)
if not ok:
    raise Exception("Failed to save: " + out_path)

imp.close()
print("OK")
"""

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as sf:
        sf.write(jython)
        script_path = sf.name

    try:
        subprocess.run(
            [fiji_exe, "--headless", "--run", script_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Fiji skeletonization failed.\n"
            f"Command name: {command_name}\n"
            f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}\n"
            f"Script used: {script_path}\n"
            f"Input: {in_tif}\n"
            f"Output expected: {out_tif}\n"
        ) from e

    if not out_tif.exists():
        raise RuntimeError(f"Fiji did not produce output skeleton tif: {out_tif}")

    sk = tiff.imread(out_tif)
    return (sk > 0).astype(np.uint8)


# -----------------------------
# Skeleton pruning (spur removal)
# -----------------------------
def neighbor_degree_26(skel: np.ndarray) -> np.ndarray:
    st = ndi.generate_binary_structure(3, 2)
    n = ndi.convolve(skel.astype(np.uint8), st.astype(np.uint8), mode="constant", cval=0)
    return (n - skel.astype(np.uint8)).astype(np.int16)

def prune_short_end_branches(skel01: np.ndarray, lmin: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    lmin = int(max(1, lmin))
    st = ndi.generate_binary_structure(3, 2)
    sk = skel01.astype(bool)

    removed_total = 0
    it = 0
    while True:
        it += 1
        deg = neighbor_degree_26(sk.astype(np.uint8))
        endpoints = sk & (deg == 1)
        junctions = sk & (deg >= 3)

        if not endpoints.any():
            break
        if not junctions.any():
            break

        dist = np.full(sk.shape, np.inf, dtype=np.float32)
        dist[junctions] = 0.0

        frontier = junctions.copy()
        d = 0
        while d < lmin and frontier.any():
            d += 1
            nbr = ndi.binary_dilation(frontier, structure=st) & sk & (dist == np.inf)
            dist[nbr] = float(d)
            frontier = nbr

        to_remove = endpoints & (dist < float(lmin))
        n_remove = int(to_remove.sum())
        if n_remove == 0:
            break

        sk[to_remove] = False
        removed_total += n_remove
        if it > 50:
            break

    info = {"prune_lmin": lmin, "prune_iters": it, "vox_removed": removed_total}
    return sk.astype(np.uint8), info


# -----------------------------
# Build proto-network + skeleton
# -----------------------------
def make_proto_and_skeleton(
    shape: Tuple[int, int, int],
    rp: RidgeParams,
    rng: np.random.Generator,
    skeleton_mode: str,
    fiji_exe: Optional[str],
    fiji_command: str,
    debug_dir: Optional[Path],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    f = rng.normal(0, 1, size=shape).astype(np.float32)
    f = ndi.gaussian_filter(f, sigma=float(rp.base_sigma))
    f = smooth_warp(f, rng, warp_sigma=float(rp.warp_sigma), warp_amp=float(rp.warp_amp))
    f = normalize(f)

    R = vesselness_ridge(f, sigma=float(rp.hessian_sigma))
    R = np.clip(R * float(rp.ridge_strength), 0.0, 1.0)

    proto01, hyst_info = hysteresis_on_response(R, q_lo=float(rp.proto_q_lo), q_hi=float(rp.proto_q_hi))
    proto01 = morph_iters(proto01, "close", int(rp.proto_close_iters))
    proto01 = morph_iters(proto01, "open", int(rp.proto_open_iters))
    proto01 = remove_small_components(proto01, int(rp.proto_min_component))

    used_skel = False
    skel_raw = proto01.copy().astype(np.uint8)
    if bool(rp.use_skeleton):
        if skeleton_mode == "skimage":
            skel_raw = skeletonize_with_skimage(proto01)
            used_skel = True
        elif skeleton_mode == "fiji":
            if not fiji_exe:
                raise RuntimeError("--skeleton-mode fiji requires --fiji-exe (e.g. C:\\...\\Fiji\\fiji-windows-x64.exe)")
            outdir = debug_dir if debug_dir is not None else Path(tempfile.mkdtemp())
            skel_raw = skeletonize_with_fiji(proto01, fiji_exe=fiji_exe, outdir=outdir, command_name=fiji_command)
            used_skel = True
        else:
            raise ValueError(f"Unknown skeleton mode: {skeleton_mode}")

    skel_pruned, prune_info = prune_short_end_branches(skel_raw, lmin=int(rp.skeleton_prune_lmin))

    if int(rp.reconnect_close_iters) > 0:
        skel_pruned = morph_iters(skel_pruned, "close", int(rp.reconnect_close_iters))

    if debug_dir is not None:
        save_tif_u8((R * 255).astype(np.uint8), debug_dir / "ridge_response.tif")
        save_tif_u8((proto01 * 255).astype(np.uint8), debug_dir / "proto_network.tif")
        save_tif_u8((skel_raw * 255).astype(np.uint8), debug_dir / "skeleton_raw.tif")
        save_tif_u8((skel_pruned * 255).astype(np.uint8), debug_dir / "skeleton_pruned.tif")

    info: Dict[str, Any] = {
        "hysteresis": hyst_info,
        "used_skeleton": used_skel,
        "skeleton_mode": skeleton_mode,
        "fiji_command": fiji_command if skeleton_mode == "fiji" else None,
        "prune_info": prune_info,
    }
    return skel_pruned.astype(np.uint8), info


# -----------------------------
# Radius-field thickening + BV/TV fit (topology-preserving)
# -----------------------------
def radius_samples_for_skeleton(
    skel01: np.ndarray,
    rng: np.random.Generator,
    base_radius_vox: float,
    mode: str,
    jitter: float,
    smooth_sigma: float,
) -> np.ndarray:
    sk = skel01.astype(bool)
    rad = np.zeros(skel01.shape, dtype=np.float32)
    if not sk.any():
        return rad

    base = float(max(0.3, base_radius_vox))
    jitter = float(np.clip(jitter, 0.0, 0.9))
    st26 = ndi.generate_binary_structure(3, 2)

    if mode == "branch":
        lab, n = ndi.label(sk, structure=st26)
        if n == 0:
            return rad
        for i in range(1, n + 1):
            r = base * float(np.exp(rng.normal(0.0, 0.35 * jitter)))
            rad[lab == i] = r
    else:
        noise = rng.normal(0.0, 1.0, size=skel01.shape).astype(np.float32)
        rad[sk] = base * np.clip(1.0 + jitter * noise[sk], 0.25, 3.0)

    if float(smooth_sigma) > 0:
        w = sk.astype(np.float32)
        num = ndi.gaussian_filter(rad, sigma=float(smooth_sigma))
        den = ndi.gaussian_filter(w, sigma=float(smooth_sigma)) + 1e-6
        rad = num / den
        rad[~sk] = 0.0

    return rad

def thicken_from_skeleton_radius_field(
    skel01: np.ndarray,
    rng: np.random.Generator,
    target_bvtv: float,
    base_radius_vox: float,
    radius_mode: str,
    radius_jitter: float,
    radius_smooth_sigma: float,
    radius_scale_hint: float,
    debug_dir: Optional[Path],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    sk = skel01.astype(bool)
    if not sk.any():
        bone = np.zeros_like(skel01, dtype=np.uint8)
        return bone, {"error": "Empty skeleton"}

    rad_skel = radius_samples_for_skeleton(
        skel01, rng=rng, base_radius_vox=base_radius_vox,
        mode=radius_mode, jitter=radius_jitter, smooth_sigma=radius_smooth_sigma
    )

    dist, inds = ndi.distance_transform_edt(~sk, return_indices=True)
    iz, iy, ix = inds
    rad_field = rad_skel[iz, iy, ix].astype(np.float32)

    floor_r = float(max(0.25, 0.25 * base_radius_vox))
    rad_field = np.maximum(rad_field, floor_r)

    target = float(np.clip(target_bvtv, 0.01, 0.95))
    lo, hi = 0.25, 3.0

    best_scale = float(np.clip(radius_scale_hint, lo, hi))
    best_err = float("inf")
    best_bvtv = None

    for _ in range(24):
        mid = 0.5 * (lo + hi)
        bone = (dist <= (mid * rad_field))
        b = float(bone.mean())
        err = abs(b - target)
        if err < best_err:
            best_err = err
            best_scale = mid
            best_bvtv = b
        if b < target:
            lo = mid
        else:
            hi = mid

    bone = (dist <= (best_scale * rad_field))
    final_bvtv = float(bone.mean())

    if debug_dir is not None:
        rf = rad_field / (rad_field.max() + 1e-6)
        save_tif_u8((rf * 255).astype(np.uint8), debug_dir / "radius_field_u8.tif")

    info = {
        "base_radius_vox": float(base_radius_vox),
        "radius_mode": str(radius_mode),
        "radius_jitter": float(radius_jitter),
        "radius_smooth_sigma": float(radius_smooth_sigma),
        "scale_fit": float(best_scale),
        "bvtv_target": float(target),
        "bvtv_after_thicken": float(final_bvtv),
        "bvtv_best_seen": float(best_bvtv) if best_bvtv is not None else None,
        "scale_bounds_end": [float(lo), float(hi)],
        "warn_target_miss": bool(abs(final_bvtv - target) > 0.10),
    }
    return bone.astype(np.uint8), info


# -----------------------------
# µCT sharp grayscale (surface-weighted)
# -----------------------------
def microct_gray_surface(bone01: np.ndarray, gp: GrayParams, rng: np.random.Generator) -> np.ndarray:
    bone = bone01.astype(bool)
    d_in = ndi.distance_transform_edt(bone).astype(np.float32)
    shell = np.exp(-((d_in / max(0.2, float(gp.shell_sigma_vox))) ** 2))
    shell = shell * bone.astype(np.float32)

    gray = float(gp.marrow_mean) + shell * (float(gp.bone_mean) - float(gp.marrow_mean))

    if float(gp.pve_sigma) > 0:
        gray = ndi.gaussian_filter(gray, sigma=float(gp.pve_sigma))

    if float(gp.bg_tex_sd) > 0:
        gray += rng.normal(0.0, float(gp.bg_tex_sd), size=gray.shape).astype(np.float32)
    if float(gp.noise_sd) > 0:
        gray += rng.normal(0.0, float(gp.noise_sd), size=gray.shape).astype(np.float32)

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

def skeleton_graph_stats(skel01: np.ndarray) -> Dict[str, Any]:
    sk = skel01.astype(bool)
    if not sk.any():
        return {"skel_voxels": 0, "junctions": 0, "endpoints": 0, "endpoint_junction_ratio": None}
    deg = neighbor_degree_26(sk.astype(np.uint8))
    endpoints = int((sk & (deg == 1)).sum())
    junctions = int((sk & (deg >= 3)).sum())
    ratio = (float(endpoints) / float(max(1, junctions))) if junctions > 0 else None
    return {
        "skel_voxels": int(sk.sum()),
        "junctions": junctions,
        "endpoints": endpoints,
        "endpoint_junction_ratio": ratio,
    }


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v13.1 ridge→skeleton→radius-field trabecular generator.")
    p.add_argument("--outdir", type=str, default="data/synth/v13_1")
    p.add_argument("--xy", type=int, default=512)
    p.add_argument("--z", type=int, default=160)
    p.add_argument("--seed", type=int, default=23)

    p.add_argument("--bvtv", type=float, default=None)
    p.add_argument("--voxel-um", type=float, default=39.0)
    p.add_argument("--priors-json", type=str, default=None, help="Path to aggregated priors JSON")

    # Ridge/warp
    p.add_argument("--base-sigma", type=float, default=3.8)
    p.add_argument("--warp-sigma", type=float, default=14.0)
    p.add_argument("--warp-amp", type=float, default=4.8)
    p.add_argument("--hessian-sigma", type=float, default=1.4)
    p.add_argument("--ridge-strength", type=float, default=1.0)

    # Proto hysteresis
    p.add_argument("--proto-q-hi", type=float, default=0.94)
    p.add_argument("--proto-q-lo", type=float, default=0.86)
    p.add_argument("--proto-close-iters", type=int, default=1)
    p.add_argument("--proto-open-iters", type=int, default=0)
    p.add_argument("--proto-min-component", type=int, default=600)

    # Skeleton
    p.add_argument("--use-skeleton", type=int, default=1)
    p.add_argument("--skeleton-mode", type=str, default="skimage", choices=["skimage", "fiji"])
    p.add_argument("--fiji-exe", type=str, default=None, help="Path to Fiji executable for --skeleton-mode fiji")
    p.add_argument("--fiji-command", type=str, default="Skeletonize (2D/3D)", help="ImageJ command name to run")
    p.add_argument("--skeleton-prune-lmin", type=int, default=10)
    p.add_argument("--reconnect-close-iters", type=int, default=2)

    # Radius-field thickening
    p.add_argument("--radius-mode", type=str, default="branch", choices=["branch", "voxel"])
    p.add_argument("--radius-jitter", type=float, default=0.25)
    p.add_argument("--radius-smooth-sigma", type=float, default=2.0)
    p.add_argument("--radius-scale-hint", type=float, default=1.0)

    # Anti-block rounding
    p.add_argument("--round-sigma", type=float, default=0.7, help="Anti-block rounding sigma (vox). 0 disables.")

    # Gray
    p.add_argument("--write-gray", type=int, default=1)

    # Debug
    p.add_argument("--debug-skeleton", type=int, default=0, help="Save debug intermediates (tifs).")

    return p


def apply_priors(args: argparse.Namespace) -> Dict[str, Any]:
    if args.priors_json is None:
        return {}

    pri_path = Path(args.priors_json)
    if not pri_path.exists():
        print(f"Priors file not found: {pri_path}")
        return {}

    with open(pri_path, "r") as f:
        pri = json.load(f)

    print(f"Loading priors from: {pri_path}")

    if "BVTV" in pri and args.bvtv is None:
        args.bvtv = float(pri["BVTV"])
        print(f"  BVTV -> {args.bvtv:.3f}")

    if "tbth_um_p90" in pri:
        tbth_um = float(pri["tbth_um_p90"])
        base_radius_vox = (tbth_um / float(args.voxel_um)) * 0.5
        setattr(args, "_base_radius_vox_from_priors", base_radius_vox)
        print(f"  base_radius_vox (from tbth_um_p90) -> {base_radius_vox:.2f} vox (tbth_um_p90={tbth_um:.1f})")

    if "euler" in pri:
        eul = float(pri["euler"])
        if eul < -1000:
            args.proto_q_hi = max(0.88, float(args.proto_q_hi) - 0.02)
            args.proto_close_iters = int(args.proto_close_iters) + 1
            args.reconnect_close_iters = int(args.reconnect_close_iters) + 1
            print(
                f"  Connectivity increased: proto_q_hi -> {args.proto_q_hi:.3f}, "
                f"proto_close_iters -> {args.proto_close_iters}, reconnect_close_iters -> {args.reconnect_close_iters}"
            )

    return pri


def prune_small_components_final(bone: np.ndarray, min_size: int) -> np.ndarray:
    if int(min_size) <= 0:
        return bone.astype(np.uint8)
    st26 = ndi.generate_binary_structure(3, 2)
    lab, n = ndi.label(bone.astype(bool), structure=st26)
    if n == 0:
        return bone.astype(np.uint8)
    counts = np.bincount(lab.ravel())
    keep = counts >= int(min_size)
    keep[0] = False
    return keep[lab].astype(np.uint8)


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(int(args.seed))

    priors = apply_priors(args)

    if args.bvtv is None:
        args.bvtv = 0.18

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    debug_dir = outdir / "debug" if bool(int(args.debug_skeleton)) else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    shape = (int(args.z), int(args.xy), int(args.xy))

    rp = RidgeParams(
        base_sigma=float(args.base_sigma),
        warp_sigma=float(args.warp_sigma),
        warp_amp=float(args.warp_amp),
        hessian_sigma=float(args.hessian_sigma),
        ridge_strength=float(args.ridge_strength),

        proto_q_hi=float(args.proto_q_hi),
        proto_q_lo=float(args.proto_q_lo),
        proto_close_iters=int(args.proto_close_iters),
        proto_open_iters=int(args.proto_open_iters),
        proto_min_component=int(args.proto_min_component),

        use_skeleton=bool(int(args.use_skeleton)),
        skeleton_prune_lmin=int(args.skeleton_prune_lmin),
        reconnect_close_iters=int(args.reconnect_close_iters),

        radius_mode=str(args.radius_mode),
        radius_jitter=float(args.radius_jitter),
        radius_smooth_sigma=float(args.radius_smooth_sigma),
        radius_scale_hint=float(args.radius_scale_hint),
    )

    gp = GrayParams(write_gray=bool(int(args.write_gray)))

    base_radius_vox = float(getattr(args, "_base_radius_vox_from_priors", 1.2))

    skel01, skel_info = make_proto_and_skeleton(
        shape=shape,
        rp=rp,
        rng=rng,
        skeleton_mode=str(args.skeleton_mode),
        fiji_exe=args.fiji_exe,
        fiji_command=str(args.fiji_command),
        debug_dir=debug_dir,
    )

    bone01, thick_info = thicken_from_skeleton_radius_field(
        skel01=skel01,
        rng=rng,
        target_bvtv=float(args.bvtv),
        base_radius_vox=base_radius_vox,
        radius_mode=str(args.radius_mode),
        radius_jitter=float(args.radius_jitter),
        radius_smooth_sigma=float(args.radius_smooth_sigma),
        radius_scale_hint=float(args.radius_scale_hint),
        debug_dir=debug_dir,
    )

    bone01 = anti_block_round(bone01, sigma=float(args.round_sigma))
    bone01 = prune_small_components_final(bone01, min_size=int(rp.prune_small_components))

    Z = shape[0]
    save_tif_u8((bone01 * 255).astype(np.uint8), outdir / "mask.tif")
    save_png_u8((bone01[Z // 2] * 255).astype(np.uint8), outdir / "mid.png")

    if gp.write_gray:
        gray = microct_gray_surface(bone01, gp, rng)
        save_tif_u8(gray, outdir / "gray.tif")
        save_png_u8(gray[Z // 2], outdir / "gray_mid.png")

    met: Dict[str, Any] = {
        "BVTV": bvtv(bone01),
        **thickness_pcts_um(bone01, voxel_um=float(args.voxel_um)),
        **euler_conn(bone01),
        "skeleton_stats": skeleton_graph_stats(skel01),
        "skeleton_info": skel_info,
        "thick_info": thick_info,
        "priors_used": priors,
        "params": {"ridge": asdict(rp), "gray": asdict(gp), "round_sigma": float(args.round_sigma)},
        "shape_zyx": list(shape),
        "skeletonize_3d_available": bool(skeletonize_3d is not None),
    }
    save_json(met, outdir / "metrics.json")

    if thick_info.get("warn_target_miss", False):
        print(
            f"Warning: BVTV target {float(args.bvtv):.3f} not reached "
            f"(after_thicken={float(thick_info.get('bvtv_after_thicken', -1)):.3f}). "
            "Try: lower proto-q-hi / increase proto-close-iters / increase reconnect-close-iters / increase radius-scale-hint."
        )

    print(
        f"Saved to {outdir}\n"
        f"BVTV={met['BVTV']:.3f} | TbTh(p90)={met['TbTh_p90']:.1f}um | Euler={met['Euler']:.1f} | "
        f"Skel(end/junc)={met['skeleton_stats'].get('endpoint_junction_ratio')} | "
        f"SkeletonMode={skel_info.get('skeleton_mode')} | round_sigma={float(args.round_sigma):.2f}"
    )


if __name__ == "__main__":
    main()