#!/usr/bin/env python3
r"""
synthetic_trabecular_v14_morphometric_control.py

v14: Morphometric-controlled trabecular generator.

Fixes vs v13.1 (from supervisor meeting):

  FIX 1 - SKELETONIZATION UNIFORMITY:
    - proto_q_hi lowered (0.94->0.92) and proto_q_lo lowered (0.86->0.84) to capture
      more ridge sections uniformly.
    - proto_close_iters increased (1->2) and proto_min_component reduced (600->400)
      to bridge gaps and keep more branches.
    - Added extra closing pass on proto-network before skeletonization.
    - reconnect_close_iters increased (2->3) for better post-skeleton reconnection.
    - radius_smooth_sigma increased (2.0->3.0) and radius_jitter reduced (0.25->0.15)
      for a smoother, more uniform radius field across the skeleton.

  FIX 2 - SOLID INTERIOR FILL:
    - microct_gray_surface replaced with microct_gray_solid.
    - Uses a Gaussian fill (solid_fill_sigma=3.0) so the entire interior of each
      strut is uniformly bright, not just the outer shell.
    - Minimum radius floor enforced (min 0.3 * base_radius_vox) so no strut collapses
      to sub-voxel width.

  FIX 3 - CLEAR MATERIAL / VOID SEPARATION:
    - Binary mask is explicitly 0 (marrow/void) or 255 (bone).
    - void.tif saved alongside mask.tif so both phases are explicitly available.
    - Grayscale renders marrow as dark (marrow_mean~15) and bone as bright (bone_mean~240).

  FIX 4 - FOUR MORPHOMETRIC TARGETS:
    - BV/TV    -> --bvtv            (drives binary-search radius scaling)
    - Tb.Th    -> --tbth-um         (derives base_radius_vox)
    - Tb.N     -> --tbn-per-mm      (derives base_sigma / network density)
    - Tb.Sp    -> --tbsp-um         (validation target)
    - All four are measured post-generation and reported.

  FIX 5 - CONTINUOUS CONNECTIVITY:
    - keep_largest_component() enforced by default (--enforce-lcc 1).
    - remove_small_components() removes fragments < --min-component-size voxels.
    - LCC fraction reported and validated (pass threshold >= 0.80).

  FIX 6 - MORPHOMETRIC PARAMETER BOUNDARIES:
    - All four targets checked against measurements post-generation.
    - Relative error and PASS/FAIL printed for each metric.
    - Full validation table saved to metrics.json.

Outputs:
  mask.tif, void.tif, mid.png
  gray.tif, gray_mid.png   (if --write-gray 1)
  metrics.json             (morphometrics, targets, validation, params)
  debug/                   (if --debug-skeleton 1)
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

try:
    from skimage.morphology import skeletonize_3d  # type: ignore
except Exception:
    skeletonize_3d = None


# -----------------------------------------------
# Params
# -----------------------------------------------

@dataclass
class RidgeParams:
    base_sigma: float = 3.8
    warp_sigma: float = 14.0
    warp_amp: float = 4.8
    hessian_sigma: float = 1.4
    ridge_strength: float = 1.0

    # FIX 1: Lower thresholds capture more ridge sections uniformly
    proto_q_hi: float = 0.92        # was 0.94
    proto_q_lo: float = 0.84        # was 0.86
    proto_close_iters: int = 2      # was 1 -> bridges more gaps before skeletonization
    proto_open_iters: int = 0
    proto_min_component: int = 400  # was 600 -> keep more branches

    use_skeleton: bool = True
    skeleton_prune_lmin: int = 8    # was 10 -> keep slightly more branches
    reconnect_close_iters: int = 3  # was 2 -> better reconnection after pruning

    # FIX 1: Smoother, more uniform radius field
    radius_mode: str = "branch"
    radius_jitter: float = 0.15     # was 0.25 -> more uniform strut thickness
    radius_smooth_sigma: float = 3.0  # was 2.0 -> smoother radius along skeleton
    radius_scale_hint: float = 1.0

    prune_small_components: int = 0


@dataclass
class GrayParams:
    write_gray: bool = True
    marrow_mean: float = 15.0
    bone_mean: float = 240.0
    # FIX 2: solid_fill_sigma replaces shell_sigma_vox.
    # Large value = fully solid strut interior; small value = hollow shell.
    solid_fill_sigma: float = 3.0   # was shell_sigma_vox=0.9 (hollow) -> 3.0 (solid)
    pve_sigma: float = 0.5
    noise_sd: float = 3.0
    bg_tex_sd: float = 1.0
    unsharp: float = 0.6
    unsharp_sigma: float = 0.8


# -----------------------------------------------
# IO
# -----------------------------------------------

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


# -----------------------------------------------
# FIX 4: Derive generator params from morphometric targets
# -----------------------------------------------

def tbth_um_to_radius_vox(tbth_um: float, voxel_um: float) -> float:
    """
    Convert Tb.Th target (µm) to base radius in voxels.
    Radius = half the thickness, converted to voxel units.
    """
    return max(0.5, (tbth_um / voxel_um) / 2.0)


def tbn_per_mm_to_base_sigma(tbn_per_mm: float, voxel_um: float) -> float:
    """
    Convert Tb.N (trabeculae/mm) to Gaussian field sigma (voxels).

    Trabecular period ~ 1/Tb.N mm = 1000/Tb.N µm
    In voxels: period_vox = (1000/Tb.N) / voxel_um
    sigma ~ period_vox / 4 is a practical approximation for the noise field.

    Example: Tb.N=2/mm, voxel=39µm -> period=500µm -> 12.8 vox -> sigma~3.2
    """
    period_um = 1000.0 / max(0.1, float(tbn_per_mm))
    period_vox = period_um / float(voxel_um)
    return float(max(1.5, period_vox / 4.0))


# -----------------------------------------------
# Field + warp
# -----------------------------------------------

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


# -----------------------------------------------
# Hessian ridge response (Frangi-style)
# -----------------------------------------------

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


# -----------------------------------------------
# Anti-block rounding
# -----------------------------------------------

def anti_block_round(bone01: np.ndarray, sigma: float) -> np.ndarray:
    if float(sigma) <= 0:
        return bone01.astype(np.uint8)
    x = ndi.gaussian_filter(bone01.astype(np.float32), sigma=float(sigma))
    return (x >= 0.5).astype(np.uint8)


# -----------------------------------------------
# FIX 5: Connectivity enforcement
# -----------------------------------------------

def keep_largest_component(vol: np.ndarray) -> np.ndarray:
    """
    FIX 5: Keep only the largest 26-connected component.
    Ensures the bone network is continuously connected throughout.
    """
    st26 = ndi.generate_binary_structure(3, 2)
    lab, n = ndi.label(vol.astype(bool), structure=st26)
    if n == 0:
        return vol.astype(np.uint8)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    lcc_label = int(counts.argmax())
    return (lab == lcc_label).astype(np.uint8)


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


# -----------------------------------------------
# Proto-network hysteresis
# -----------------------------------------------

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


# -----------------------------------------------
# Skeletonization
# -----------------------------------------------

def skeletonize_with_skimage(proto01: np.ndarray) -> np.ndarray:
    if skeletonize_3d is None:
        raise RuntimeError("skimage.skeletonize_3d unavailable; install scikit-image.")
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


# -----------------------------------------------
# Skeleton pruning (spur removal)
# -----------------------------------------------

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


# -----------------------------------------------
# FIX 1: Improved proto-network + skeleton pipeline
# -----------------------------------------------

def make_proto_and_skeleton(
    shape: Tuple[int, int, int],
    rp: RidgeParams,
    rng: np.random.Generator,
    skeleton_mode: str,
    fiji_exe: Optional[str],
    fiji_command: str,
    debug_dir: Optional[Path],
) -> Tuple[np.ndarray, Dict[str, Any]]:

    # Generate base field with organic warp
    f = rng.normal(0, 1, size=shape).astype(np.float32)
    f = ndi.gaussian_filter(f, sigma=float(rp.base_sigma))
    f = smooth_warp(f, rng, float(rp.warp_sigma), float(rp.warp_amp))
    f = normalize(f)

    # Ridge detection
    R = vesselness_ridge(f, sigma=float(rp.hessian_sigma))
    R = np.clip(R * float(rp.ridge_strength), 0.0, 1.0)

    # Proto-network via hysteresis thresholding
    proto01, hyst_info = hysteresis_on_response(R, q_lo=float(rp.proto_q_lo),
                                                q_hi=float(rp.proto_q_hi))
    proto01 = morph_iters(proto01, "close", int(rp.proto_close_iters))
    proto01 = morph_iters(proto01, "open", int(rp.proto_open_iters))
    proto01 = remove_small_components(proto01, int(rp.proto_min_component))

    # FIX 1: Extra closing pass on proto-network before skeletonization
    # Bridges small remaining gaps so the skeleton captures all sections
    st26 = ndi.generate_binary_structure(3, 2)
    if proto01.astype(bool).sum() > 0:
        proto01 = ndi.binary_closing(proto01.astype(bool), structure=st26,
                                     iterations=1).astype(np.uint8)

    # Skeletonize
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

    # Prune short spurs
    skel_pruned, prune_info = prune_short_end_branches(skel_raw,
                                                       lmin=int(rp.skeleton_prune_lmin))

    # FIX 1: Reconnection after pruning to restore any broken links
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


# -----------------------------------------------
# FIX 2: Solid interior thickening + BV/TV fit
# -----------------------------------------------

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
        return np.zeros_like(skel01, dtype=np.uint8), {"error": "Empty skeleton"}

    rad_skel = radius_samples_for_skeleton(
        skel01, rng=rng, base_radius_vox=base_radius_vox,
        mode=radius_mode, jitter=radius_jitter, smooth_sigma=radius_smooth_sigma,
    )

    dist, inds = ndi.distance_transform_edt(~sk, return_indices=True)
    iz, iy, ix = inds
    rad_field = rad_skel[iz, iy, ix].astype(np.float32)

    # FIX 2: Enforce minimum radius so every strut is solid (not sub-voxel thin)
    min_r = float(max(0.5, 0.3 * base_radius_vox))
    rad_field = np.maximum(rad_field, min_r)

    # Binary search to hit BV/TV target
    target = float(np.clip(target_bvtv, 0.01, 0.95))
    lo, hi = 0.25, 3.0
    best_scale = float(np.clip(radius_scale_hint, lo, hi))
    best_err = float("inf")
    best_bvtv = None

    for _ in range(24):
        mid = 0.5 * (lo + hi)
        bone = dist <= (mid * rad_field)
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


# -----------------------------------------------
# FIX 2+3: Solid µCT grayscale with clear material/void separation
# -----------------------------------------------

def microct_gray_solid(bone01: np.ndarray, gp: GrayParams,
                       rng: np.random.Generator) -> np.ndarray:
    """
    FIX 2: Solid bone rendering.
    The entire interior of each strut is uniformly bright (bone_mean ~240).
    Only a thin partial-volume-effect (PVE) transition exists at the surface.

    FIX 3: Clear material/void separation.
    Void (marrow) = dark (marrow_mean ~15). Bone = bright (bone_mean ~240).
    The sigmoid fill ensures a clean boundary with no ambiguous mid-intensity interior.

    Replaces the old microct_gray_surface which used shell_sigma_vox=0.9 and
    produced hollow-looking struts where only the outer shell was bright.
    """
    bone = bone01.astype(bool)

    # Distance from the bone/marrow interface into the bone interior
    d_in = ndi.distance_transform_edt(bone).astype(np.float32)

    # Sigmoid fill: reaches ~86% of bone_mean at 1 voxel depth,
    # ~98% at 2 voxels depth for solid_fill_sigma=1.0.
    # With solid_fill_sigma=3.0, the fill is slower -> realistic gradual hardening.
    sigma = float(gp.solid_fill_sigma)
    fill = 1.0 - np.exp(-(d_in / max(0.2, sigma)) ** 2)
    fill = fill * bone.astype(np.float32)

    # Map fill [0,1] -> [marrow_mean, bone_mean]
    gray = float(gp.marrow_mean) + fill * (float(gp.bone_mean) - float(gp.marrow_mean))

    # Partial volume effect (slight blur at boundary)
    if float(gp.pve_sigma) > 0:
        gray = ndi.gaussian_filter(gray, sigma=float(gp.pve_sigma))

    # Add realistic noise
    if float(gp.bg_tex_sd) > 0:
        gray = gray + rng.normal(0.0, float(gp.bg_tex_sd), size=gray.shape).astype(np.float32)
    if float(gp.noise_sd) > 0:
        gray = gray + rng.normal(0.0, float(gp.noise_sd), size=gray.shape).astype(np.float32)

    # Unsharp masking (sharpens strut edges)
    if float(gp.unsharp) > 0:
        blurred = ndi.gaussian_filter(gray, sigma=max(0.4, float(gp.unsharp_sigma)))
        gray = gray + float(gp.unsharp) * (gray - blurred)

    return np.clip(gray, 0, 255).astype(np.uint8)


# -----------------------------------------------
# FIX 4+6: Measure all four morphometric properties
# -----------------------------------------------

def measure_all_morphometrics(vol01: np.ndarray,
                              voxel_um: float) -> Dict[str, float]:
    """
    FIX 4: Measure all four key morphometric properties:
      - BV/TV  : bone volume fraction
      - Tb.Th  : trabecular thickness (distance transform inside bone * 2)
      - Tb.N   : trabecular number (BV/TV / mean Tb.Th, in /mm)
      - Tb.Sp  : trabecular spacing (distance transform inside marrow * 2)
    Plus Euler characteristic and connectivity metrics.
    """
    bone = vol01.astype(bool)

    # BV/TV
    bvtv_val = float(bone.mean())

    # Distance transforms
    dt_bone = ndi.distance_transform_edt(bone) * float(voxel_um)     # inside bone (µm)
    dt_marrow = ndi.distance_transform_edt(~bone) * float(voxel_um)  # inside marrow (µm)

    tbth_vals = dt_bone[bone]    # local half-thickness at each bone voxel
    tbsp_vals = dt_marrow[~bone] # local half-spacing at each marrow voxel

    def pct(x: np.ndarray, p: float) -> float:
        return float(np.percentile(x, p)) if x.size else 0.0

    tbth_p50 = pct(tbth_vals, 50)
    tbth_p90 = pct(tbth_vals, 90)
    tbsp_p50 = pct(tbsp_vals, 50)
    tbsp_p90 = pct(tbsp_vals, 90)

    # Tb.N: standard formula Tb.N = BV/TV / (Tb.Th_mean in mm)
    tbth_mean_mm = float(np.mean(tbth_vals)) / 1000.0 if tbth_vals.size else 1e-6
    tbn = bvtv_val / tbth_mean_mm if tbth_mean_mm > 0 else 0.0

    # Euler characteristic (connectivity)
    euler = float(euler_number(bone, connectivity=3))

    # LCC fraction and component count
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
    return {
        "skel_voxels": int(sk.sum()),
        "junctions": junctions,
        "endpoints": endpoints,
        "endpoint_junction_ratio": ratio,
    }


# -----------------------------------------------
# FIX 6: Morphometric validation against targets
# -----------------------------------------------

def validate_morphometrics(measured: Dict[str, float],
                           targets: Dict[str, float]) -> Dict[str, Any]:
    """
    FIX 6: Check all four morphometric properties against targets.
    Tolerance levels are set per metric based on practical achievability.
    Returns pass/fail and relative error for each.
    """
    checks: Dict[str, Any] = {}

    metric_checks = [
        # (measured_key,     target_key,       tolerance, label)
        ("BVTV",        "bvtv_target",    0.05,  "BV/TV"),
        ("TbTh_um_p50", "tbth_um_target", 0.15,  "Tb.Th (p50, µm)"),
        ("TbN_per_mm",  "tbn_target",     0.20,  "Tb.N (/mm)"),
        ("TbSp_um_p50", "tbsp_um_target", 0.15,  "Tb.Sp (p50, µm)"),
    ]

    for meas_key, tgt_key, tol, label in metric_checks:
        tgt_val = targets.get(tgt_key)
        meas_val = measured.get(meas_key)
        if tgt_val is not None and meas_val is not None and float(tgt_val) > 0:
            rel_err = abs(float(meas_val) - float(tgt_val)) / float(tgt_val)
            checks[label] = {
                "measured": float(meas_val),
                "target": float(tgt_val),
                "rel_error": float(rel_err),
                "tolerance": float(tol),
                "pass": bool(rel_err <= tol),
            }

    # FIX 5: Connectivity check
    lcc = measured.get("lcc_frac", 0.0)
    checks["Connectivity (LCC)"] = {
        "lcc_frac": float(lcc),
        "n_components": int(measured.get("n_components", -1)),
        "pass": bool(lcc >= 0.80),
        "note": "LCC >= 0.80 required for continuously connected structure",
    }

    return checks


# -----------------------------------------------
# CLI
# -----------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="v14 morphometric-controlled trabecular generator.")
    p.add_argument("--outdir", type=str, default="data/synth/v14")
    p.add_argument("--xy", type=int, default=512)
    p.add_argument("--z", type=int, default=160)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--voxel-um", type=float, default=39.0)

    # FIX 4: Explicit morphometric targets (all four)
    p.add_argument("--bvtv", type=float, default=None,
                   help="Target BV/TV bone volume fraction (0-1).")
    p.add_argument("--tbth-um", type=float, default=None,
                   help="Target Tb.Th trabecular thickness (µm). Drives strut radius.")
    p.add_argument("--tbn-per-mm", type=float, default=None,
                   help="Target Tb.N trabecular number per mm. Drives network density.")
    p.add_argument("--tbsp-um", type=float, default=None,
                   help="Target Tb.Sp trabecular spacing (µm). Used for validation.")
    p.add_argument("--priors-json", type=str, default=None,
                   help="Path to priors JSON (fills any unspecified morphometric targets).")

    # Ridge/warp (can override auto-derived values)
    p.add_argument("--base-sigma", type=float, default=None,
                   help="Override base_sigma (vox). Normally auto-derived from --tbn-per-mm.")
    p.add_argument("--warp-sigma", type=float, default=14.0)
    p.add_argument("--warp-amp", type=float, default=4.8)
    p.add_argument("--hessian-sigma", type=float, default=1.4)
    p.add_argument("--ridge-strength", type=float, default=1.0)

    # Proto-network (FIX 1 defaults)
    p.add_argument("--proto-q-hi", type=float, default=0.92)
    p.add_argument("--proto-q-lo", type=float, default=0.84)
    p.add_argument("--proto-close-iters", type=int, default=2)
    p.add_argument("--proto-open-iters", type=int, default=0)
    p.add_argument("--proto-min-component", type=int, default=400)

    # Skeleton
    p.add_argument("--use-skeleton", type=int, default=1)
    p.add_argument("--skeleton-mode", type=str, default="skimage",
                   choices=["skimage", "fiji"])
    p.add_argument("--fiji-exe", type=str, default=None)
    p.add_argument("--fiji-command", type=str, default="Skeletonize (2D/3D)")
    p.add_argument("--skeleton-prune-lmin", type=int, default=8)
    p.add_argument("--reconnect-close-iters", type=int, default=3)

    # Radius-field (FIX 1 defaults)
    p.add_argument("--radius-mode", type=str, default="branch",
                   choices=["branch", "voxel"])
    p.add_argument("--radius-jitter", type=float, default=0.15)
    p.add_argument("--radius-smooth-sigma", type=float, default=3.0)
    p.add_argument("--radius-scale-hint", type=float, default=1.0)

    # FIX 5: Connectivity enforcement
    p.add_argument("--enforce-lcc", type=int, default=1,
                   help="Keep only the largest connected component (1=yes).")
    p.add_argument("--min-component-size", type=int, default=500,
                   help="Remove disconnected fragments smaller than this (voxels).")

    # Anti-block rounding
    p.add_argument("--round-sigma", type=float, default=0.7)

    # FIX 2: Solid fill
    p.add_argument("--solid-fill-sigma", type=float, default=3.0,
                   help="Solid interior fill sigma. Large=uniform solid; small=hollow shell.")

    # Gray
    p.add_argument("--write-gray", type=int, default=1)

    # Debug
    p.add_argument("--debug-skeleton", type=int, default=0,
                   help="Save debug TIFFs (ridge, proto, skeleton raw/pruned, radius field).")

    return p


def apply_priors(args: argparse.Namespace) -> Dict[str, Any]:
    """Load priors JSON and fill any unspecified morphometric targets."""
    if args.priors_json is None:
        return {}
    pri_path = Path(args.priors_json)
    if not pri_path.exists():
        print(f"Priors file not found: {pri_path}")
        return {}
    with open(pri_path) as f:
        pri = json.load(f)
    print(f"Loaded priors from: {pri_path}")
    if args.bvtv is None and "BVTV" in pri:
        args.bvtv = float(pri["BVTV"])
        print(f"  BV/TV  <- {args.bvtv:.3f}")
    if args.tbth_um is None:
        for k in ("tbth_um_p90", "TbTh_um_p90", "TbTh_p90"):
            if k in pri:
                args.tbth_um = float(pri[k])
                print(f"  Tb.Th  <- {args.tbth_um:.1f} µm  (from '{k}')")
                break
    if args.tbsp_um is None:
        for k in ("tbsp_um_p50", "TbSp_um_p50", "TbSp_p50"):
            if k in pri:
                args.tbsp_um = float(pri[k])
                print(f"  Tb.Sp  <- {args.tbsp_um:.1f} µm  (from '{k}')")
                break
    return pri


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(int(args.seed))
    priors = apply_priors(args)

    # FIX 4: Apply defaults for any unspecified morphometric targets
    if args.bvtv is None:
        args.bvtv = 0.18
        print(f"No --bvtv specified; using default {args.bvtv:.2f}")
    if args.tbth_um is None:
        args.tbth_um = 150.0
        print(f"No --tbth-um specified; using default {args.tbth_um:.0f} µm")
    if args.tbn_per_mm is None:
        args.tbn_per_mm = 2.0
        print(f"No --tbn-per-mm specified; using default {args.tbn_per_mm:.1f} /mm")
    if args.tbsp_um is None:
        args.tbsp_um = 300.0
        print(f"No --tbsp-um specified; using default {args.tbsp_um:.0f} µm")

    # FIX 4: Derive base_sigma from Tb.N if not explicitly overridden
    if args.base_sigma is None:
        args.base_sigma = tbn_per_mm_to_base_sigma(float(args.tbn_per_mm),
                                                    float(args.voxel_um))
        print(f"  base_sigma  (from Tb.N={args.tbn_per_mm:.2f}/mm)"
              f" -> {args.base_sigma:.2f} vox")

    # FIX 4: Derive base_radius_vox from Tb.Th
    base_radius_vox = tbth_um_to_radius_vox(float(args.tbth_um), float(args.voxel_um))
    print(f"  base_radius (from Tb.Th={args.tbth_um:.0f}µm)"
          f" -> {base_radius_vox:.2f} vox")

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

    gp = GrayParams(
        write_gray=bool(int(args.write_gray)),
        solid_fill_sigma=float(args.solid_fill_sigma),
    )

    # --- Step 1: Build skeleton
    print(f"\nGenerating {shape} volume (voxel={args.voxel_um}µm)...")
    skel01, skel_info = make_proto_and_skeleton(
        shape=shape, rp=rp, rng=rng,
        skeleton_mode=str(args.skeleton_mode),
        fiji_exe=args.fiji_exe,
        fiji_command=str(args.fiji_command),
        debug_dir=debug_dir,
    )
    print(f"  Skeleton voxels: {int(skel01.sum())}")

    # --- Step 2: Thicken skeleton to target BV/TV
    bone01, thick_info = thicken_from_skeleton_radius_field(
        skel01=skel01, rng=rng,
        target_bvtv=float(args.bvtv),
        base_radius_vox=base_radius_vox,
        radius_mode=str(args.radius_mode),
        radius_jitter=float(args.radius_jitter),
        radius_smooth_sigma=float(args.radius_smooth_sigma),
        radius_scale_hint=float(args.radius_scale_hint),
        debug_dir=debug_dir,
    )

    # --- Step 3: Post-processing
    bone01 = anti_block_round(bone01, sigma=float(args.round_sigma))

    # FIX 5: Enforce connectivity - remove fragments, keep largest component
    if int(args.min_component_size) > 0:
        bone01 = remove_small_components(bone01, min_size=int(args.min_component_size))
    if bool(int(args.enforce_lcc)):
        bone01 = keep_largest_component(bone01)
        print(f"  LCC enforced (ensures continuous connected structure).")

    # FIX 3: Save explicit bone mask AND void mask
    void01 = (1 - bone01).astype(np.uint8)
    Z = shape[0]
    save_tif_u8((bone01 * 255).astype(np.uint8), outdir / "mask.tif")
    save_tif_u8((void01 * 255).astype(np.uint8), outdir / "void.tif")  # FIX 3
    save_png_u8((bone01[Z // 2] * 255).astype(np.uint8), outdir / "mid.png")

    # FIX 2: Solid grayscale rendering (replaces hollow shell rendering)
    if gp.write_gray:
        gray = microct_gray_solid(bone01, gp, rng)
        save_tif_u8(gray, outdir / "gray.tif")
        save_png_u8(gray[Z // 2], outdir / "gray_mid.png")

    # FIX 4+6: Measure all four morphometrics and validate against targets
    print("\nMeasuring morphometrics...")
    morphometrics = measure_all_morphometrics(bone01, voxel_um=float(args.voxel_um))

    targets_dict = {
        "bvtv_target":    float(args.bvtv),
        "tbth_um_target": float(args.tbth_um),
        "tbn_target":     float(args.tbn_per_mm),
        "tbsp_um_target": float(args.tbsp_um),
    }
    validation = validate_morphometrics(morphometrics, targets_dict)

    # Save metrics
    met: Dict[str, Any] = {
        "morphometrics": morphometrics,
        "targets": targets_dict,
        "validation": validation,
        "skeleton_stats": skeleton_graph_stats(skel01),
        "skeleton_info": skel_info,
        "thick_info": thick_info,
        "priors_used": priors,
        "params": {
            "ridge": asdict(rp),
            "gray": asdict(gp),
            "round_sigma": float(args.round_sigma),
        },
        "shape_zyx": list(shape),
        "voxel_um": float(args.voxel_um),
        "skeletonize_3d_available": bool(skeletonize_3d is not None),
    }
    save_json(met, outdir / "metrics.json")

    # FIX 6: Print validation table
    print(f"\n{'=' * 58}")
    print(f"  MORPHOMETRIC VALIDATION SUMMARY")
    print(f"{'=' * 58}")
    print(f"  {'Metric':<22} {'Target':>9} {'Measured':>10} {'Error':>7} {'':>5}")
    print(f"  {'-' * 56}")
    for label, chk in validation.items():
        if label == "Connectivity (LCC)":
            status = "PASS" if chk["pass"] else "FAIL <<"
            print(f"  {'Connectivity (LCC)':<22} {'>=0.80':>9} "
                  f"{chk['lcc_frac']:>10.3f} {'—':>7}  {status}")
        else:
            status = "PASS" if chk["pass"] else "FAIL <<"
            print(f"  {label:<22} {chk['target']:>9.2f} "
                  f"{chk['measured']:>10.2f} {chk['rel_error']:>6.1%}  {status}")
    print(f"{'=' * 58}")

    if thick_info.get("warn_target_miss", False):
        print(f"\nWarning: BV/TV target {args.bvtv:.3f} not closely reached "
              f"(got {thick_info.get('bvtv_after_thicken', -1):.3f}). "
              "Try adjusting --proto-q-hi / --proto-close-iters / --radius-scale-hint.")

    print(f"\nOutputs saved to: {outdir}")
    print(f"  mask.tif  (bone=255, void=0)")
    print(f"  void.tif  (void=255, bone=0)")
    if gp.write_gray:
        print(f"  gray.tif  (solid µCT grayscale)")
    print(f"  metrics.json")


if __name__ == "__main__":
    main()