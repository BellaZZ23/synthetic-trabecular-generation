#!/usr/bin/env python3
r"""
synthetic_trabecular_v15_morphometric_control.py

v15: VOI-driven trabecular generator.

Generation core: IDENTICAL to v14 (skeleton, thickening, connectivity).
    No changes to: base_sigma derivation, skeleton_prune_lmin (8),
    reconnect_close_iters (3), BV/TV binary search range (0.25-3.0),
    solid_fill_sigma default (3.0 for v14 compat when no VOI).

Safe fixes (do NOT alter geometry):
    FIX A  Measurement x2 correction (reporting only)
    FIX B  Adaptive fill sigma for grayscale (mask unchanged)
    FIX C  VOI targets.json loading from pipeline_voi1
    FIX D  ASCII validation labels

Primary input: *_targets.json from pipeline_voi1_multiframe_dcm_to_targets.py
Fallback: --profile tamimi-hf / tamimi-hoa, or explicit --bvtv / --tbth-um etc.

Usage (PowerShell):
    python synthetic_trabecular_v15_morphometric_control.py `
        --targets-json data\derived\VOI1\specimen01_Specimen1_VOI1_Scan1_targets.json `
        --outdir output\specimen01 `
        --seed 42

    python synthetic_trabecular_v15_morphometric_control.py `
        --targets-dir data\derived\VOI1 `
        --outdir output\batch `
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
from PIL import Image
import tifffile as tiff
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from skimage.measure import euler_number

try:
    from skimage.morphology import skeletonize_3d  # type: ignore
except ImportError:
    try:
        from skimage.morphology import skeletonize as skeletonize_3d  # type: ignore
    except ImportError:
        skeletonize_3d = None


# ---------------------------------------------------------------
# Literature validation bounds (Tamimi et al., Bone 140, 2020)
# Used ONLY for warnings, never as generation targets
# ---------------------------------------------------------------
TAMIMI_BOUNDS = {
    "BVTV":       {"lo": 0.05,  "hi": 0.50,  "unit": "fraction"},
    "TbTh_um":    {"lo": 80.0,  "hi": 300.0, "unit": "um"},
    "TbN_per_mm": {"lo": 0.5,   "hi": 5.0,   "unit": "/mm"},
    "TbSp_um":    {"lo": 150.0, "hi": 1200.0, "unit": "um"},
}
TAMIMI_HF = {"BVTV": 0.2037, "TbTh_um": 180.0, "TbN_per_mm": 1.5, "TbSp_um": 580.0}
TAMIMI_HOA = {"BVTV": 0.2862, "TbTh_um": 130.0, "TbN_per_mm": 2.58, "TbSp_um": 420.0}


# ---------------------------------------------------------------
# VOI targets loader (FIX C)
# ---------------------------------------------------------------
def load_voi_targets(targets_json: str) -> Dict[str, Any]:
    p = Path(targets_json)
    if not p.exists():
        raise FileNotFoundError(f"VOI targets file not found: {p}")
    with open(p) as f:
        data = json.load(f)
    print(f"Loaded VOI targets from: {p}")
    print(f"  BV/TV  = {data.get('BVTV')}")
    print(f"  Tb.Th  = {data.get('TbTh_um_p90', data.get('TbTh_um_p50'))} um")
    print(f"  Tb.Sp  = {data.get('TbSp_um_p50')} um")
    print(f"  Voxel  = {data.get('voxel_um_zyx')} um")
    print(f"  Shape  = {data.get('shape_zyx')}")
    return data


def find_targets_files(targets_dir: str) -> List[Path]:
    d = Path(targets_dir)
    files = sorted(d.glob("*_targets.json"))
    if not files:
        raise FileNotFoundError(f"No *_targets.json files found in {d}")
    return files


def extract_generation_params(voi: Dict[str, Any],
                              args: argparse.Namespace) -> Dict[str, Any]:
    """Extract generation targets from VOI data.
    VOI pipeline reports DT inscribed radius -> x2 for diameter."""
    rc = 2.0 if bool(int(args.prior_uses_radius)) else 1.0
    if rc > 1.0:
        print(f"  [FIX C] x{rc:.0f} radius->diameter correction")

    bvtv = args.bvtv if args.bvtv is not None else voi.get("BVTV")
    if bvtv is None:
        raise ValueError("No BV/TV in VOI or CLI")

    tbth_um = args.tbth_um
    if tbth_um is None:
        for k in ("TbTh_um_p90", "TbTh_um_p50", "tbth_um_p90", "tbth_um_p50"):
            if k in voi and voi[k] is not None and float(voi[k]) > 0:
                tbth_um = float(voi[k]) * rc
                print(f"  Tb.Th <- {tbth_um:.1f} um (from '{k}' x{rc:.0f})")
                break
    if tbth_um is None:
        raise ValueError("No Tb.Th in VOI or CLI")

    tbsp_um = args.tbsp_um
    if tbsp_um is None:
        for k in ("TbSp_um_p50", "TbSp_um_p90", "tbsp_um_p50"):
            if k in voi and voi[k] is not None and float(voi[k]) > 0:
                tbsp_um = float(voi[k]) * rc
                print(f"  Tb.Sp <- {tbsp_um:.1f} um (from '{k}' x{rc:.0f})")
                break
    if tbsp_um is None:
        raise ValueError("No Tb.Sp in VOI or CLI")

    tbn_per_mm = args.tbn_per_mm
    if tbn_per_mm is None:
        tbn_per_mm = float(bvtv) / (float(tbth_um) / 1000.0)
        print(f"  Tb.N <- {tbn_per_mm:.2f} /mm (derived: BV/TV / Tb.Th)")

    voxel_um = args.voxel_um
    if voxel_um is None:
        vox_zyx = voi.get("voxel_um_zyx")
        if vox_zyx is not None and len(vox_zyx) >= 1:
            voxel_um = float(vox_zyx[0])
            print(f"  Voxel <- {voxel_um:.1f} um (from VOI)")
    if voxel_um is None:
        voxel_um = 39.0

    shape_z = args.z
    shape_xy = args.xy
    shp = voi.get("shape_zyx")
    if shp is not None and len(shp) >= 3:
        if shape_z is None: shape_z = int(shp[0])
        if shape_xy is None: shape_xy = int(shp[1])
        print(f"  Shape <- Z={shape_z}, XY={shape_xy} (from VOI)")
    if shape_z is None: shape_z = 160
    if shape_xy is None: shape_xy = 512

    return {
        "bvtv": float(bvtv), "tbth_um": float(tbth_um),
        "tbn_per_mm": float(tbn_per_mm), "tbsp_um": float(tbsp_um),
        "voxel_um": float(voxel_um),
        "shape_z": int(shape_z), "shape_xy": int(shape_xy),
    }


# ---------------------------------------------------------------
# Params (IDENTICAL to v14)
# ---------------------------------------------------------------
@dataclass
class RidgeParams:
    base_sigma: float = 3.8
    warp_sigma: float = 14.0
    warp_amp: float = 4.8
    hessian_sigma: float = 1.4
    ridge_strength: float = 1.0
    proto_q_hi: float = 0.92
    proto_q_lo: float = 0.84
    proto_close_iters: int = 2
    proto_open_iters: int = 0
    proto_min_component: int = 400
    use_skeleton: bool = True
    skeleton_prune_lmin: int = 8       # v14 value
    reconnect_close_iters: int = 3     # v14 value
    radius_mode: str = "branch"
    radius_jitter: float = 0.15
    radius_smooth_sigma: float = 3.0
    radius_scale_hint: float = 1.0
    prune_small_components: int = 0


@dataclass
class GrayParams:
    write_gray: bool = True
    marrow_mean: float = 15.0
    bone_mean: float = 240.0
    solid_fill_sigma: Optional[float] = None   # None = auto (FIX B)
    pve_sigma: float = 0.5
    noise_sd: float = 3.0
    bg_tex_sd: float = 1.0
    unsharp: float = 0.6
    unsharp_sigma: float = 0.8


# ---------------------------------------------------------------
# IO
# ---------------------------------------------------------------
def save_png_u8(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img.astype(np.uint8), mode="L").save(path)

def save_tif_u8(stack: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(path, stack.astype(np.uint8), imagej=True, dtype=np.uint8)

def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------
# Parameter derivation (IDENTICAL to v14)
# ---------------------------------------------------------------
def tbth_um_to_radius_vox(tbth_um: float, voxel_um: float) -> float:
    return max(0.5, (tbth_um / voxel_um) / 2.0)

def tbn_per_mm_to_base_sigma(tbn_per_mm: float, voxel_um: float) -> float:
    period_um = 1000.0 / max(0.1, float(tbn_per_mm))
    period_vox = period_um / float(voxel_um)
    return float(max(1.5, period_vox / 4.0))

def compute_adaptive_fill_sigma(base_radius_vox: float) -> float:
    """FIX B: adaptive fill sigma for grayscale only."""
    return float(np.clip(0.35 * base_radius_vox, 0.3, 1.5))


# ---------------------------------------------------------------
# Field + warp (IDENTICAL to v14)
# ---------------------------------------------------------------
def normalize(f: np.ndarray) -> np.ndarray:
    x = f.astype(np.float32)
    x -= float(x.mean())
    x /= float(x.std() + 1e-6)
    return x

def smooth_warp(field: np.ndarray, rng: np.random.Generator,
                warp_sigma: float, warp_amp: float) -> np.ndarray:
    if warp_amp <= 0:
        return field
    dz = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma) * warp_amp
    dy = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma) * warp_amp
    dx = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma) * warp_amp
    Z, Y, X = field.shape
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    coords = np.array([zz + dz, yy + dy, xx + dx])
    return map_coordinates(field, coords, order=1, mode="reflect").astype(np.float32)


# ---------------------------------------------------------------
# Hessian ridge response (IDENTICAL to v14)
# ---------------------------------------------------------------
def hessian_eigs_3d(f: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fxx = ndi.gaussian_filter(f, sigma=sigma, order=(0, 0, 2))
    fyy = ndi.gaussian_filter(f, sigma=sigma, order=(0, 2, 0))
    fzz = ndi.gaussian_filter(f, sigma=sigma, order=(2, 0, 0))
    fxy = ndi.gaussian_filter(f, sigma=sigma, order=(0, 1, 1))
    fxz = ndi.gaussian_filter(f, sigma=sigma, order=(1, 0, 1))
    fyz = ndi.gaussian_filter(f, sigma=sigma, order=(1, 1, 0))
    H = np.stack([
        np.stack([fzz, fyz, fxz], axis=-1),
        np.stack([fyz, fyy, fxy], axis=-1),
        np.stack([fxz, fxy, fxx], axis=-1),
    ], axis=-2)
    w = np.linalg.eigvalsh(H.reshape(-1, 3, 3)).reshape(f.shape + (3,))
    idx = np.argsort(np.abs(w), axis=-1)
    w = np.take_along_axis(w, idx, axis=-1)
    return w[..., 0], w[..., 1], w[..., 2]

def vesselness_ridge(f: np.ndarray, sigma: float) -> np.ndarray:
    l1, l2, l3 = hessian_eigs_3d(f, sigma=sigma)
    eps = 1e-6
    r1 = np.abs(l1) / (np.abs(l3) + eps)
    r2 = np.abs(l2) / (np.abs(l3) + eps)
    V = np.exp(-(r1 * r1) / 0.25) * np.exp(-(r2 * r2) / 0.25)
    V = V.astype(np.float32)
    V = V / (float(V.max()) + 1e-6)
    return V


# ---------------------------------------------------------------
# Morphological utilities (IDENTICAL to v14)
# ---------------------------------------------------------------
def anti_block_round(bone01: np.ndarray, sigma: float) -> np.ndarray:
    if float(sigma) <= 0:
        return bone01.astype(np.uint8)
    x = ndi.gaussian_filter(bone01.astype(np.float32), sigma=float(sigma))
    return (x >= 0.5).astype(np.uint8)

def keep_largest_component(vol: np.ndarray) -> np.ndarray:
    st26 = ndi.generate_binary_structure(3, 2)
    lab, n = ndi.label(vol.astype(bool), structure=st26)
    if n == 0:
        return vol.astype(np.uint8)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return (lab == int(counts.argmax())).astype(np.uint8)

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


# ---------------------------------------------------------------
# Proto-network hysteresis (IDENTICAL to v14)
# ---------------------------------------------------------------
def hysteresis_on_response(R: np.ndarray, q_lo: float,
                           q_hi: float) -> Tuple[np.ndarray, Dict[str, float]]:
    q_lo = float(np.clip(q_lo, 0.5, 0.995))
    q_hi = float(np.clip(q_hi, q_lo + 1e-3, 0.999))
    thr_hi = float(np.quantile(R, q_hi))
    thr_lo = float(np.quantile(R, q_lo))
    strong = R >= thr_hi
    weak = R >= thr_lo
    st26 = ndi.generate_binary_structure(3, 2)
    lab, n = ndi.label(weak, structure=st26)
    if n == 0:
        return strong.astype(np.uint8), {"thr_lo": thr_lo, "thr_hi": thr_hi}
    strong_labels = np.unique(lab[strong])
    keep = np.zeros(n + 1, dtype=bool)
    keep[strong_labels] = True
    keep[0] = False
    return keep[lab].astype(np.uint8), {"thr_lo": thr_lo, "thr_hi": thr_hi}


# ---------------------------------------------------------------
# Skeletonization (IDENTICAL to v14)
# ---------------------------------------------------------------
def skeletonize_with_skimage(proto01: np.ndarray) -> np.ndarray:
    if skeletonize_3d is None:
        raise RuntimeError("skimage.skeletonize_3d unavailable")
    return skeletonize_3d(proto01.astype(bool)).astype(np.uint8)

def skeletonize_with_fiji(proto01: np.ndarray, fiji_exe: str, outdir: Path,
                          command_name: str = "Skeletonize (2D/3D)") -> np.ndarray:
    outdir.mkdir(parents=True, exist_ok=True)
    in_tif = outdir / "proto_network_for_fiji.tif"
    out_tif = outdir / "skeleton_from_fiji.tif"
    save_tif_u8((proto01 * 255).astype(np.uint8), in_tif)
    jython = f"""
from ij import IJ
from ij.io import FileSaver
imp = IJ.openImage(r"{in_tif.as_posix()}")
if imp is None:
    raise Exception("Failed to open: " + r"{in_tif.as_posix()}")
IJ.run(imp, r"{command_name}", "")
FileSaver(imp).saveAsTiff(r"{out_tif.as_posix()}")
imp.close()
print("OK")
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as sf:
        sf.write(jython)
        script_path = sf.name
    try:
        subprocess.run([fiji_exe, "--headless", "--run", script_path],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Fiji skeletonization failed.\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        ) from e
    if not out_tif.exists():
        raise RuntimeError(f"Fiji did not produce output: {out_tif}")
    return (tiff.imread(out_tif) > 0).astype(np.uint8)


# ---------------------------------------------------------------
# Skeleton pruning (IDENTICAL to v14)
# ---------------------------------------------------------------
def neighbor_degree_26(skel: np.ndarray) -> np.ndarray:
    st = ndi.generate_binary_structure(3, 2)
    n = ndi.convolve(skel.astype(np.uint8), st.astype(np.uint8), mode="constant", cval=0)
    return (n - skel.astype(np.uint8)).astype(np.int16)

def prune_short_end_branches(skel01: np.ndarray,
                             lmin: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    lmin = int(max(1, lmin))
    st = ndi.generate_binary_structure(3, 2)
    sk = skel01.astype(bool)
    removed_total = 0
    for it in range(1, 51):
        deg = neighbor_degree_26(sk.astype(np.uint8))
        endpoints = sk & (deg == 1)
        junctions = sk & (deg >= 3)
        if not endpoints.any() or not junctions.any():
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
    return sk.astype(np.uint8), {"prune_lmin": lmin, "vox_removed": removed_total}


# ---------------------------------------------------------------
# Proto-network + skeleton pipeline (IDENTICAL to v14)
# ---------------------------------------------------------------
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
    f = smooth_warp(f, rng, float(rp.warp_sigma), float(rp.warp_amp))
    f = normalize(f)

    R = vesselness_ridge(f, sigma=float(rp.hessian_sigma))
    R = np.clip(R * float(rp.ridge_strength), 0.0, 1.0)

    proto01, hyst_info = hysteresis_on_response(R, q_lo=float(rp.proto_q_lo),
                                                q_hi=float(rp.proto_q_hi))
    proto01 = morph_iters(proto01, "close", int(rp.proto_close_iters))
    proto01 = morph_iters(proto01, "open", int(rp.proto_open_iters))
    proto01 = remove_small_components(proto01, int(rp.proto_min_component))

    st26 = ndi.generate_binary_structure(3, 2)
    if proto01.astype(bool).sum() > 0:
        proto01 = ndi.binary_closing(proto01.astype(bool), structure=st26,
                                     iterations=1).astype(np.uint8)

    skel_raw = proto01.copy().astype(np.uint8)
    used_skel = False
    if bool(rp.use_skeleton):
        if skeleton_mode == "skimage":
            skel_raw = skeletonize_with_skimage(proto01)
            used_skel = True
        elif skeleton_mode == "fiji":
            if not fiji_exe:
                raise RuntimeError("--skeleton-mode fiji requires --fiji-exe")
            wd = debug_dir if debug_dir is not None else Path(tempfile.mkdtemp())
            skel_raw = skeletonize_with_fiji(proto01, fiji_exe=fiji_exe, outdir=wd,
                                            command_name=fiji_command)
            used_skel = True

    skel_pruned, prune_info = prune_short_end_branches(skel_raw,
                                                       lmin=int(rp.skeleton_prune_lmin))

    # v14: reconnection closing (NO re-skeletonize — preserves structure)
    if int(rp.reconnect_close_iters) > 0:
        skel_pruned = morph_iters(skel_pruned, "close", int(rp.reconnect_close_iters))

    if debug_dir is not None:
        save_tif_u8((R * 255).astype(np.uint8), debug_dir / "ridge_response.tif")
        save_tif_u8((proto01 * 255).astype(np.uint8), debug_dir / "proto_network.tif")
        save_tif_u8((skel_raw * 255).astype(np.uint8), debug_dir / "skeleton_raw.tif")
        save_tif_u8((skel_pruned * 255).astype(np.uint8), debug_dir / "skeleton_pruned.tif")

    return skel_pruned.astype(np.uint8), {
        "hysteresis": hyst_info,
        "used_skeleton": used_skel,
        "skeleton_mode": skeleton_mode,
        "fiji_command": fiji_command if skeleton_mode == "fiji" else None,
        "prune_info": prune_info,
    }


# ---------------------------------------------------------------
# Radius field + thickening + BV/TV fit (IDENTICAL to v14)
# ---------------------------------------------------------------
def radius_samples_for_skeleton(
    skel01: np.ndarray, rng: np.random.Generator,
    base_radius_vox: float, mode: str,
    jitter: float, smooth_sigma: float,
) -> np.ndarray:
    sk = skel01.astype(bool)
    rad = np.zeros(skel01.shape, dtype=np.float32)
    if not sk.any():
        return rad
    base = float(max(0.5, base_radius_vox))
    jitter = float(np.clip(jitter, 0.0, 0.9))
    st26 = ndi.generate_binary_structure(3, 2)
    if mode == "branch":
        lab, n = ndi.label(sk, structure=st26)
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
    skel01: np.ndarray, rng: np.random.Generator,
    target_bvtv: float, base_radius_vox: float,
    radius_mode: str, radius_jitter: float,
    radius_smooth_sigma: float, radius_scale_hint: float,
    debug_dir: Optional[Path],
) -> Tuple[np.ndarray, Dict[str, Any]]:

    sk = skel01.astype(bool)
    if not sk.any():
        return np.zeros_like(skel01, dtype=np.uint8), {"error": "Empty skeleton"}

    rad_skel = radius_samples_for_skeleton(
        skel01, rng=rng, base_radius_vox=base_radius_vox,
        mode=radius_mode, jitter=radius_jitter, smooth_sigma=radius_smooth_sigma)

    dist, inds = ndi.distance_transform_edt(~sk, return_indices=True)
    iz, iy, ix = inds
    rad_field = rad_skel[iz, iy, ix].astype(np.float32)

    min_r = float(max(0.5, 0.3 * base_radius_vox))
    rad_field = np.maximum(rad_field, min_r)

    # v14 binary search range: 0.25 to 3.0
    target = float(np.clip(target_bvtv, 0.01, 0.95))
    lo, hi = 0.25, 3.0
    best_scale = float(np.clip(radius_scale_hint, lo, hi))
    best_err = float("inf")

    for _ in range(24):
        mid = 0.5 * (lo + hi)
        bone = dist <= (mid * rad_field)
        b = float(bone.mean())
        err = abs(b - target)
        if err < best_err:
            best_err = err
            best_scale = mid
        if b < target:
            lo = mid
        else:
            hi = mid

    bone = (dist <= (best_scale * rad_field)).astype(np.uint8)
    final_bvtv = float(bone.mean())

    if debug_dir is not None:
        rf = rad_field / (rad_field.max() + 1e-6)
        save_tif_u8((rf * 255).astype(np.uint8), debug_dir / "radius_field_u8.tif")

    return bone, {
        "base_radius_vox": float(base_radius_vox),
        "min_radius_floor": float(min_r),
        "radius_mode": radius_mode,
        "scale_fit": float(best_scale),
        "bvtv_target": float(target),
        "bvtv_after_thicken": float(final_bvtv),
        "warn_target_miss": bool(abs(final_bvtv - target) > 0.10),
    }


# ---------------------------------------------------------------
# Grayscale rendering (v14 core + FIX B adaptive sigma option)
# ---------------------------------------------------------------
def microct_gray_solid(bone01: np.ndarray, gp: GrayParams,
                       rng: np.random.Generator,
                       base_radius_vox: float = 2.0) -> np.ndarray:
    bone = bone01.astype(bool)
    d_in = ndi.distance_transform_edt(bone).astype(np.float32)

    # FIX B: if solid_fill_sigma is None, use adaptive; otherwise use the value
    if gp.solid_fill_sigma is not None:
        sigma = float(gp.solid_fill_sigma)
    else:
        sigma = compute_adaptive_fill_sigma(base_radius_vox)

    fill = 1.0 - np.exp(-(d_in / max(0.2, sigma)) ** 2)
    fill = fill * bone.astype(np.float32)

    gray = float(gp.marrow_mean) + fill * (float(gp.bone_mean) - float(gp.marrow_mean))

    if float(gp.pve_sigma) > 0:
        gray = ndi.gaussian_filter(gray, sigma=float(gp.pve_sigma))
    if float(gp.bg_tex_sd) > 0:
        gray = gray + rng.normal(0.0, float(gp.bg_tex_sd), size=gray.shape).astype(np.float32)
    if float(gp.noise_sd) > 0:
        gray = gray + rng.normal(0.0, float(gp.noise_sd), size=gray.shape).astype(np.float32)
    if float(gp.unsharp) > 0:
        blurred = ndi.gaussian_filter(gray, sigma=max(0.4, float(gp.unsharp_sigma)))
        gray = gray + float(gp.unsharp) * (gray - blurred)

    return np.clip(gray, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------
# Morphometric measurement (FIX A: x2 correction, reporting only)
# ---------------------------------------------------------------
def measure_all_morphometrics(vol01: np.ndarray,
                              voxel_um: float) -> Dict[str, float]:
    bone = vol01.astype(bool)
    bvtv_val = float(bone.mean())

    dt_bone = ndi.distance_transform_edt(bone) * float(voxel_um)
    dt_marrow = ndi.distance_transform_edt(~bone) * float(voxel_um)

    tbth_vals = dt_bone[bone]
    tbsp_vals = dt_marrow[~bone]

    def pct(x, p):
        return float(np.percentile(x, p)) if x.size else 0.0

    # FIX A: x2 for full diameter (reporting only, does not affect generation)
    tbth_p50 = 2.0 * pct(tbth_vals, 50)
    tbth_p90 = 2.0 * pct(tbth_vals, 90)
    tbsp_p50 = 2.0 * pct(tbsp_vals, 50)
    tbsp_p90 = 2.0 * pct(tbsp_vals, 90)

    tbth_mean_mm = (2.0 * float(np.mean(tbth_vals)) / 1000.0) if tbth_vals.size else 1e-6
    tbn = bvtv_val / tbth_mean_mm if tbth_mean_mm > 0 else 0.0

    euler = float(euler_number(bone, connectivity=3))

    st26 = ndi.generate_binary_structure(3, 2)
    lab, n = ndi.label(bone, structure=st26)
    lcc_frac = 0.0
    n_components = int(n)
    if n > 0:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        lcc_frac = float(counts.max()) / float(max(1, bone.sum()))

    return {
        "BVTV": bvtv_val,
        "TbTh_um_p50": tbth_p50,
        "TbTh_um_p90": tbth_p90,
        "TbSp_um_p50": tbsp_p50,
        "TbSp_um_p90": tbsp_p90,
        "TbN_per_mm": float(tbn),
        "Euler": euler,
        "ConnProxy": float(1.0 - euler),
        "n_components": n_components,
        "lcc_frac": lcc_frac,
    }

def skeleton_graph_stats(skel01: np.ndarray) -> Dict[str, Any]:
    sk = skel01.astype(bool)
    if not sk.any():
        return {"skel_voxels": 0, "junctions": 0, "endpoints": 0,
                "endpoint_junction_ratio": None}
    deg = neighbor_degree_26(sk.astype(np.uint8))
    endpoints = int((sk & (deg == 1)).sum())
    junctions = int((sk & (deg >= 3)).sum())
    ratio = float(endpoints) / float(max(1, junctions)) if junctions > 0 else None
    return {"skel_voxels": int(sk.sum()), "junctions": junctions,
            "endpoints": endpoints, "endpoint_junction_ratio": ratio}


# ---------------------------------------------------------------
# Validation (FIX D: ASCII labels)
# ---------------------------------------------------------------
def validate_morphometrics(measured: Dict[str, float],
                           targets: Dict[str, float]) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    metric_checks = [
        ("BVTV",        "bvtv_target",    0.05, "BV/TV"),
        ("TbTh_um_p50", "tbth_um_target", 0.15, "Tb.Th (p50, um)"),
        ("TbN_per_mm",  "tbn_target",     0.20, "Tb.N (/mm)"),
        ("TbSp_um_p50", "tbsp_um_target", 0.15, "Tb.Sp (p50, um)"),
    ]
    for meas_key, tgt_key, tol, label in metric_checks:
        tgt_val = targets.get(tgt_key)
        meas_val = measured.get(meas_key)
        if tgt_val is not None and meas_val is not None and float(tgt_val) > 0:
            rel_err = abs(float(meas_val) - float(tgt_val)) / float(tgt_val)
            checks[label] = {
                "measured": float(meas_val), "target": float(tgt_val),
                "rel_error": float(rel_err), "tolerance": float(tol),
                "pass": bool(rel_err <= tol),
            }
    lcc = measured.get("lcc_frac", 0.0)
    checks["Connectivity (LCC)"] = {
        "lcc_frac": float(lcc),
        "n_components": int(measured.get("n_components", -1)),
        "pass": bool(lcc >= 0.80),
        "note": "LCC >= 0.80 required",
    }
    return checks

def check_tamimi_bounds(measured: Dict[str, float]) -> List[str]:
    warnings = []
    for meas_key, bound_key in [("BVTV","BVTV"),("TbTh_um_p50","TbTh_um"),
                                 ("TbN_per_mm","TbN_per_mm"),("TbSp_um_p50","TbSp_um")]:
        val = measured.get(meas_key)
        b = TAMIMI_BOUNDS.get(bound_key)
        if val is not None and b is not None:
            if float(val) < b["lo"] or float(val) > b["hi"]:
                warnings.append(f"{meas_key}={float(val):.2f} outside [{b['lo']},{b['hi']}] {b['unit']}")
    return warnings


# ---------------------------------------------------------------
# Single-specimen generation
# ---------------------------------------------------------------
def generate_one(params: Dict[str, Any], args: argparse.Namespace,
                 outdir: Path, voi_source: Optional[Dict] = None) -> Dict[str, Any]:

    rng = np.random.default_rng(int(args.seed))
    bvtv = params["bvtv"]
    tbth_um = params["tbth_um"]
    tbn_per_mm = params["tbn_per_mm"]
    tbsp_um = params["tbsp_um"]
    voxel_um = params["voxel_um"]
    shape = (params["shape_z"], params["shape_xy"], params["shape_xy"])

    # v14 derivation (unchanged)
    base_radius_vox = tbth_um_to_radius_vox(tbth_um, voxel_um)
    base_sigma = (float(args.base_sigma) if args.base_sigma is not None
                  else tbn_per_mm_to_base_sigma(tbn_per_mm, voxel_um))

    print(f"\n  Targets: BV/TV={bvtv:.3f}, Tb.Th={tbth_um:.0f}um, "
          f"Tb.N={tbn_per_mm:.2f}/mm, Tb.Sp={tbsp_um:.0f}um")
    print(f"  Derived: base_sigma={base_sigma:.2f}vox, "
          f"base_radius={base_radius_vox:.2f}vox")
    print(f"  Volume:  {shape}, voxel={voxel_um:.1f}um")

    outdir.mkdir(parents=True, exist_ok=True)
    debug_dir = outdir / "debug" if bool(int(args.debug_skeleton)) else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    rp = RidgeParams(
        base_sigma=base_sigma,
        warp_sigma=float(args.warp_sigma), warp_amp=float(args.warp_amp),
        hessian_sigma=float(args.hessian_sigma),
        ridge_strength=float(args.ridge_strength),
        proto_q_hi=float(args.proto_q_hi), proto_q_lo=float(args.proto_q_lo),
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
    gp = GrayParams(
        write_gray=bool(int(args.write_gray)),
        solid_fill_sigma=args.solid_fill_sigma,
    )

    # --- Generate (v14 pipeline) ---
    skel01, skel_info = make_proto_and_skeleton(
        shape=shape, rp=rp, rng=rng,
        skeleton_mode=str(args.skeleton_mode),
        fiji_exe=args.fiji_exe,
        fiji_command=str(args.fiji_command),
        debug_dir=debug_dir)
    print(f"  Skeleton: {int(skel01.sum())} voxels")

    bone01, thick_info = thicken_from_skeleton_radius_field(
        skel01=skel01, rng=rng, target_bvtv=bvtv,
        base_radius_vox=base_radius_vox,
        radius_mode=str(args.radius_mode),
        radius_jitter=float(args.radius_jitter),
        radius_smooth_sigma=float(args.radius_smooth_sigma),
        radius_scale_hint=float(args.radius_scale_hint),
        debug_dir=debug_dir)

    bone01 = anti_block_round(bone01, sigma=float(args.round_sigma))
    if int(args.min_component_size) > 0:
        bone01 = remove_small_components(bone01, min_size=int(args.min_component_size))
    if bool(int(args.enforce_lcc)):
        bone01 = keep_largest_component(bone01)

    # Save outputs
    void01 = (1 - bone01).astype(np.uint8)
    Z = shape[0]
    save_tif_u8((bone01 * 255).astype(np.uint8), outdir / "mask.tif")
    save_tif_u8((void01 * 255).astype(np.uint8), outdir / "void.tif")
    save_png_u8((bone01[Z // 2] * 255).astype(np.uint8), outdir / "mid.png")

    if gp.write_gray:
        gray = microct_gray_solid(bone01, gp, rng, base_radius_vox=base_radius_vox)
        save_tif_u8(gray, outdir / "gray.tif")
        save_png_u8(gray[Z // 2], outdir / "gray_mid.png")

    # --- Measure (FIX A: x2 corrected) + validate ---
    morphometrics = measure_all_morphometrics(bone01, voxel_um=voxel_um)
    targets_dict = {
        "bvtv_target": bvtv, "tbth_um_target": tbth_um,
        "tbn_target": tbn_per_mm, "tbsp_um_target": tbsp_um,
    }
    validation = validate_morphometrics(morphometrics, targets_dict)
    tamimi_warnings = check_tamimi_bounds(morphometrics)

    met: Dict[str, Any] = {
        "version": "v15 (v14-core)",
        "source": "VOI" if voi_source is not None else "manual/literature",
        "voi_targets_file": str(voi_source.get("source_dcm", "")) if voi_source else None,
        "morphometrics": morphometrics,
        "targets": targets_dict,
        "validation": validation,
        "tamimi_bound_warnings": tamimi_warnings,
        "skeleton_stats": skeleton_graph_stats(skel01),
        "skeleton_info": skel_info,
        "thick_info": thick_info,
        "params": {"ridge": asdict(rp), "gray": asdict(gp),
                    "round_sigma": float(args.round_sigma)},
        "shape_zyx": list(shape),
        "voxel_um": voxel_um,
    }
    save_json(met, outdir / "metrics.json")

    # Print validation
    print(f"\n  {'=' * 55}")
    print(f"  {'Metric':<22} {'Target':>9} {'Measured':>10} {'Error':>7}")
    print(f"  {'-' * 55}")
    for label, chk in validation.items():
        if label == "Connectivity (LCC)":
            s = "PASS" if chk["pass"] else "FAIL"
            print(f"  {'Connectivity (LCC)':<22} {'>=0.80':>9} "
                  f"{chk['lcc_frac']:>10.3f} {'':>7} {s}")
        else:
            s = "PASS" if chk["pass"] else "FAIL"
            print(f"  {label:<22} {chk['target']:>9.2f} "
                  f"{chk['measured']:>10.2f} {chk['rel_error']:>6.1%} {s}")
    if tamimi_warnings:
        print(f"\n  Tamimi bound warnings:")
        for w in tamimi_warnings:
            print(f"    ! {w}")
    print(f"  {'=' * 55}")
    return met


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="v15 trabecular generator (v14 core + VOI loading + safe fixes)")

    # PRIMARY: VOI targets
    p.add_argument("--targets-json", type=str, default=None)
    p.add_argument("--targets-dir", type=str, default=None)

    # FALLBACK: literature
    p.add_argument("--profile", type=str, default=None,
                   choices=["tamimi-hf", "tamimi-hoa"])

    # Output
    p.add_argument("--outdir", type=str, default="data/synth/v15")
    p.add_argument("--seed", type=int, default=23)

    # Overrides
    p.add_argument("--bvtv", type=float, default=None)
    p.add_argument("--tbth-um", type=float, default=None)
    p.add_argument("--tbn-per-mm", type=float, default=None)
    p.add_argument("--tbsp-um", type=float, default=None)
    p.add_argument("--voxel-um", type=float, default=None)
    p.add_argument("--xy", type=int, default=None)
    p.add_argument("--z", type=int, default=None)

    # Prior correction
    p.add_argument("--prior-uses-radius", type=int, default=1,
                   help="1 = VOI pipeline reports DT radius (default, x2 applied)")

    # v14 generation params (all defaults match v14 exactly)
    p.add_argument("--base-sigma", type=float, default=None)
    p.add_argument("--warp-sigma", type=float, default=14.0)
    p.add_argument("--warp-amp", type=float, default=4.8)
    p.add_argument("--hessian-sigma", type=float, default=1.4)
    p.add_argument("--ridge-strength", type=float, default=1.0)
    p.add_argument("--proto-q-hi", type=float, default=0.92)
    p.add_argument("--proto-q-lo", type=float, default=0.84)
    p.add_argument("--proto-close-iters", type=int, default=2)
    p.add_argument("--proto-open-iters", type=int, default=0)
    p.add_argument("--proto-min-component", type=int, default=400)
    p.add_argument("--use-skeleton", type=int, default=1)
    p.add_argument("--skeleton-mode", type=str, default="skimage",
                   choices=["skimage", "fiji"])
    p.add_argument("--fiji-exe", type=str, default=None)
    p.add_argument("--fiji-command", type=str, default="Skeletonize (2D/3D)")
    p.add_argument("--skeleton-prune-lmin", type=int, default=8)     # v14
    p.add_argument("--reconnect-close-iters", type=int, default=3)   # v14
    p.add_argument("--radius-mode", type=str, default="branch",
                   choices=["branch", "voxel"])
    p.add_argument("--radius-jitter", type=float, default=0.15)
    p.add_argument("--radius-smooth-sigma", type=float, default=3.0)
    p.add_argument("--radius-scale-hint", type=float, default=1.0)
    p.add_argument("--enforce-lcc", type=int, default=1)
    p.add_argument("--min-component-size", type=int, default=500)
    p.add_argument("--round-sigma", type=float, default=0.7)
    p.add_argument("--solid-fill-sigma", type=float, default=None,
                   help="None=adaptive (FIX B). Set 3.0 for v14-exact grayscale.")
    p.add_argument("--write-gray", type=int, default=1)
    p.add_argument("--debug-skeleton", type=int, default=0)

    return p


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    if args.targets_json is not None:
        voi = load_voi_targets(args.targets_json)
        params = extract_generation_params(voi, args)
        outdir = Path(args.outdir)
        print(f"\n{'=' * 60}")
        print(f"  Generating from VOI: {args.targets_json}")
        print(f"{'=' * 60}")
        generate_one(params, args, outdir, voi_source=voi)
        print(f"\nOutputs: {outdir}/")

    elif args.targets_dir is not None:
        target_files = find_targets_files(args.targets_dir)
        print(f"\nFound {len(target_files)} VOI targets files")
        for tf in target_files:
            specimen_name = tf.stem.replace("_targets", "")
            voi = load_voi_targets(str(tf))
            params = extract_generation_params(voi, args)
            outdir = Path(args.outdir) / specimen_name
            print(f"\n{'=' * 60}")
            print(f"  Generating: {specimen_name}")
            print(f"{'=' * 60}")
            generate_one(params, args, outdir, voi_source=voi)
        print(f"\nAll outputs: {args.outdir}/")

    elif args.profile is not None:
        ref = TAMIMI_HF if args.profile == "tamimi-hf" else TAMIMI_HOA
        print(f"Using Tamimi {args.profile} profile (FALLBACK)")
        if args.bvtv is None:       args.bvtv = ref["BVTV"]
        if args.tbth_um is None:    args.tbth_um = ref["TbTh_um"]
        if args.tbn_per_mm is None: args.tbn_per_mm = ref["TbN_per_mm"]
        if args.tbsp_um is None:    args.tbsp_um = ref["TbSp_um"]
        if args.voxel_um is None:   args.voxel_um = 39.0
        params = {
            "bvtv": float(args.bvtv), "tbth_um": float(args.tbth_um),
            "tbn_per_mm": float(args.tbn_per_mm), "tbsp_um": float(args.tbsp_um),
            "voxel_um": float(args.voxel_um),
            "shape_z": args.z or 160, "shape_xy": args.xy or 512,
        }
        generate_one(params, args, Path(args.outdir))
        print(f"\nOutputs: {args.outdir}/")

    else:
        print("ERROR: Provide one of:")
        print("  --targets-json <path>")
        print("  --targets-dir  <path>")
        print("  --profile tamimi-hf|tamimi-hoa")
        raise SystemExit(1)


if __name__ == "__main__":
    main()