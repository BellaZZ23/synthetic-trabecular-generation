#!/usr/bin/env python3
r"""
synthetic_trabecular_v8_randomfield_connectivity.py

v8: Connectivity-first trabecular generator (FULL FOV) tuned to produce an
interconnected trabecular network (like your reference image).

What’s new vs v7 (the key changes for “that connectivity”):
1) Connectivity-first enforcement:
   - enforce percolation (x/y/z) as a *hard* target (optional)
   - target higher LCC fraction (default 0.97–0.995)
   - bounded “bridge” loop (dilate + close) until connectivity targets met

2) BV/TV correction that preserves connectivity:
   - after bridging, BV/TV usually increases
   - we "thin" the network using an EDT-core trimming strategy:
       keep voxels with larger distance-to-boundary first (removes skinny surface voxels)
     while repeatedly verifying percolation/LCC.
   - this keeps the network connected while meeting BV/TV.

3) Phase logic (bone vs void):
   - many GRF thresholds produce isolated “islands” depending on phase.
   - v8 supports generating either:
       bone = field >= thr   (default)
     or
       bone = field < thr    ("invert phase")
     which often yields a more connected trabecular meshwork.

Dependencies:
- numpy, pillow, tifffile
- scipy (required)
- scikit-image (euler_number)

PowerShell smoke test:
python .\synthetic_trabecular_v8_randomfield_connectivity.py `
  --outdir data\v8_smoketest --n-volumes 1 --xy 96 --z 64 --seed 1 `
  --bvtv 0.18 --sigma-x 2.2 --sigma-y 2.2 --sigma-z 1.8 --multiscale 0 `
  --enforce-perc 1 --perc-axes xyz --min-lcc-fraction 0.985 `
  --bridge-max-attempts 5 --bridge-dilate-step 1 --bridge-close-step 1 `
  --thin-to-bvtv 1 --thin-max-iter 10 --thin-close-iters 1 `
  --write-gray 1 --pve-sigma 0.9 --ct-noise-sd 4 --bg-texture-sd 2 --unsharp 0.6 `
  --export-2d 1

Tip:
- If you still see fragmented islands, try:
    --invert-phase 1
  and/or increase:
    --sigma-x/y/z slightly, and/or --bridge-max-attempts
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
from PIL import Image
import tifffile as tiff
from scipy import ndimage as ndi
from skimage.measure import euler_number


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
    mix2: float = 0.35
    nonlinearity: float = 0.0


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


@dataclass
class ConnectivityParams:
    min_lcc_fraction: float = 0.985
    max_components: int = 5000
    use_26_connectivity: bool = True

    enforce_perc: bool = True
    perc_axes: str = "xyz"  # any subset of x,y,z


@dataclass
class BridgeParams:
    bridge_max_attempts: int = 6
    bridge_dilate_step: int = 1   # per attempt
    bridge_close_step: int = 1    # per attempt
    bridge_open_iters: int = 0    # usually keep 0 (opening breaks links)


@dataclass
class ThinParams:
    thin_to_bvtv: bool = True
    thin_max_iter: int = 12
    thin_close_iters: int = 1   # small closing after thinning to heal pinholes
    thin_eps_bvtv: float = 5e-4


@dataclass
class SliceSamplingParams:
    enable: bool = True
    n_slices_per_axis: int = 8
    random: bool = False
    warn_mean_components_per_slice: float = 20.0


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

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def bvtv(vol01: np.ndarray) -> float:
    return float(np.mean(vol01 > 0))


# -----------------------------
# Random Field (3D GRF)
# -----------------------------
def generate_grf_3d(Z: int, H: int, W: int, fp: FieldParams, rng: np.random.Generator) -> np.ndarray:
    base = rng.normal(0.0, 1.0, size=(Z, H, W)).astype(np.float32)
    f1 = ndi.gaussian_filter(base, sigma=(float(fp.sigma_z), float(fp.sigma_y), float(fp.sigma_x)))

    if fp.multiscale:
        base2 = rng.normal(0.0, 1.0, size=(Z, H, W)).astype(np.float32)
        f2 = ndi.gaussian_filter(base2, sigma=(float(fp.sigma2_z), float(fp.sigma2_y), float(fp.sigma2_x)))
        f = (1.0 - float(fp.mix2)) * f1 + float(fp.mix2) * f2
    else:
        f = f1

    f = f - float(f.mean())
    f = f / (float(f.std()) + 1e-8)

    if float(fp.nonlinearity) > 0:
        k = float(fp.nonlinearity)
        f = np.tanh(k * f)

    return f.astype(np.float32)

def adjust_bvtv_by_threshold(field: np.ndarray, target_bvtv: float, invert_phase: bool, max_iter: int = 12) -> Tuple[np.ndarray, float]:
    """
    Binary search threshold to match BV/TV on the field.
    If invert_phase=True: bone = field < thr
    Else: bone = field >= thr
    """
    target = float(np.clip(target_bvtv, 0.001, 0.999))
    lo, hi = float(field.min()), float(field.max())
    thr = float(np.quantile(field, 1.0 - target))  # good starting point for non-invert

    for _ in range(max_iter):
        if invert_phase:
            vol01 = (field < thr).astype(np.uint8)
        else:
            vol01 = (field >= thr).astype(np.uint8)

        frac = float(vol01.mean())
        if abs(frac - target) < 5e-4:
            return vol01, thr

        # if too much bone, move threshold appropriately depending on phase
        if frac > target:
            if invert_phase:
                # bone=field<thr, decrease thr => fewer bone
                hi = thr
                thr = 0.5 * (lo + thr)
            else:
                # bone=field>=thr, increase thr => fewer bone
                lo = thr
                thr = 0.5 * (thr + hi)
        else:
            if invert_phase:
                # increase thr => more bone
                lo = thr
                thr = 0.5 * (thr + hi)
            else:
                # decrease thr => more bone
                hi = thr
                thr = 0.5 * (lo + thr)

    if invert_phase:
        vol01 = (field < thr).astype(np.uint8)
    else:
        vol01 = (field >= thr).astype(np.uint8)
    return vol01, thr


# -----------------------------
# Base morphology (light touch)
# -----------------------------
def keep_largest_component_3d(vol01: np.ndarray, use_26: bool = True) -> np.ndarray:
    st = np.ones((3, 3, 3), dtype=bool) if use_26 else ndi.generate_binary_structure(3, 1)
    lab, n = ndi.label(vol01.astype(bool), structure=st)
    if n <= 1:
        return vol01
    counts = np.bincount(lab.ravel())
    if len(counts) <= 1:
        return vol01
    lcc = int(np.argmax(counts[1:]) + 1)
    return (lab == lcc).astype(np.uint8)

def postprocess(vol01: np.ndarray, mp: MorphParams) -> np.ndarray:
    v = vol01.astype(bool)
    st = ndi.generate_binary_structure(3, 1)  # 6-neigh

    if mp.open_iters > 0:
        v = ndi.binary_opening(v, structure=st, iterations=int(mp.open_iters))
    if mp.close_iters > 0:
        v = ndi.binary_closing(v, structure=st, iterations=int(mp.close_iters))

    out = v.astype(np.uint8)
    if mp.keep_largest:
        out = keep_largest_component_3d(out, use_26=True)
    return out


# -----------------------------
# Connectivity metrics + percolation
# -----------------------------
def connected_components_3d(vol01: np.ndarray, use_26: bool) -> Tuple[np.ndarray, int, np.ndarray]:
    st = np.ones((3, 3, 3), dtype=bool) if use_26 else ndi.generate_binary_structure(3, 1)
    lab, n = ndi.label(vol01.astype(bool), structure=st)
    counts = np.bincount(lab.ravel())
    return lab, int(n), counts

def percolates(labels: np.ndarray, axes: str) -> Dict[str, bool]:
    Z, H, W = labels.shape
    out = {"percolate_x": False, "percolate_y": False, "percolate_z": False}
    axes = set(list(axes.lower()))

    if "x" in axes:
        left = set(np.unique(labels[:, :, 0])) - {0}
        right = set(np.unique(labels[:, :, W - 1])) - {0}
        out["percolate_x"] = bool(len(left & right) > 0)

    if "y" in axes:
        top = set(np.unique(labels[:, 0, :])) - {0}
        bottom = set(np.unique(labels[:, H - 1, :])) - {0}
        out["percolate_y"] = bool(len(top & bottom) > 0)

    if "z" in axes:
        front = set(np.unique(labels[0, :, :])) - {0}
        back = set(np.unique(labels[Z - 1, :, :])) - {0}
        out["percolate_z"] = bool(len(front & back) > 0)

    return out

def connectivity_metrics(vol01: np.ndarray, cp: ConnectivityParams) -> Dict[str, Any]:
    bone = vol01.astype(bool)
    lab, n, counts = connected_components_3d(bone.astype(np.uint8), use_26=cp.use_26_connectivity)

    bone_vox = int(bone.sum())
    if bone_vox == 0:
        m = {"n_components": 0, "lcc_fraction": 0.0, "lcc_size": 0, "bone_voxels": 0}
        m.update({"percolate_x": False, "percolate_y": False, "percolate_z": False})
        m["connectivity_ok"] = False
        return m

    lcc_size = int(np.max(counts[1:])) if n > 0 and len(counts) > 1 else bone_vox
    lcc_fraction = float(lcc_size / max(1, bone_vox))

    perc = percolates(lab, cp.perc_axes) if cp.enforce_perc else {"percolate_x": True, "percolate_y": True, "percolate_z": True}
    perc_ok = True
    if cp.enforce_perc:
        for ax in set(list(cp.perc_axes.lower())):
            if ax == "x":
                perc_ok = perc_ok and bool(perc["percolate_x"])
            if ax == "y":
                perc_ok = perc_ok and bool(perc["percolate_y"])
            if ax == "z":
                perc_ok = perc_ok and bool(perc["percolate_z"])

    ok = (lcc_fraction >= float(cp.min_lcc_fraction)) and (n <= int(cp.max_components)) and perc_ok

    m = {
        "n_components": int(n),
        "lcc_fraction": float(lcc_fraction),
        "lcc_size": int(lcc_size),
        "bone_voxels": int(bone_vox),
        **perc,
        "connectivity_ok": bool(ok),
    }
    return m


# -----------------------------
# Bridging loop (connectivity-first)
# -----------------------------
def enforce_connectivity_by_bridging(vol01: np.ndarray, cp: ConnectivityParams, bp: BridgeParams) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Iteratively strengthen links:
      dilate (small) + close (small) + keep largest
    until connectivity targets are met (LCC + optional percolation).
    """
    st = ndi.generate_binary_structure(3, 1)  # 6-neigh for morphology
    v = vol01.astype(bool)

    info = {"bridge_applied": False, "bridge_attempts": 0, "connectivity_before": None, "connectivity_after": None}

    m0 = connectivity_metrics(v.astype(np.uint8), cp)
    info["connectivity_before"] = m0
    if m0["connectivity_ok"]:
        info["connectivity_after"] = m0
        return v.astype(np.uint8), info

    for attempt in range(1, int(bp.bridge_max_attempts) + 1):
        info["bridge_applied"] = True
        info["bridge_attempts"] = attempt

        di = int(bp.bridge_dilate_step) * attempt
        ci = int(bp.bridge_close_step) * attempt

        if di > 0:
            v = ndi.binary_dilation(v, structure=st, iterations=di)
        if ci > 0:
            v = ndi.binary_closing(v, structure=st, iterations=ci)

        if int(bp.bridge_open_iters) > 0:
            v = ndi.binary_opening(v, structure=st, iterations=int(bp.bridge_open_iters))

        v01 = v.astype(np.uint8)
        # keep-largest helps remove islands and focuses on meshwork
        v01 = keep_largest_component_3d(v01, use_26=cp.use_26_connectivity)
        v = v01.astype(bool)

        m = connectivity_metrics(v01, cp)
        if m["connectivity_ok"]:
            info["connectivity_after"] = m
            info["dilate_iters_used"] = di
            info["close_iters_used"] = ci
            return v01, info

    m_end = connectivity_metrics(v.astype(np.uint8), cp)
    info["connectivity_after"] = m_end
    return v.astype(np.uint8), info


# -----------------------------
# Thin-to-BVTV while preserving connectivity (EDT-core trimming)
# -----------------------------
def thin_preserve_connectivity(vol01: np.ndarray, target_bvtv: float, cp: ConnectivityParams, tp: ThinParams) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Goal: reduce BV/TV back to target after bridging WITHOUT breaking connectivity.

    Approach:
    - compute EDT inside bone (distance to boundary)
    - trim surface voxels first (keep core voxels with high EDT)
    - after each trim, optionally small closing, keep-largest, then validate connectivity
    - if connectivity fails, keep more voxels (lower EDT threshold)
    """
    target = float(np.clip(target_bvtv, 0.001, 0.999))
    v = vol01.astype(np.uint8)
    info: Dict[str, Any] = {"thin_applied": False, "thin_iters": 0, "bvtv_before": bvtv(v), "bvtv_after": None}

    if not tp.thin_to_bvtv:
        info["bvtv_after"] = bvtv(v)
        return v, info

    if bvtv(v) <= target + float(tp.thin_eps_bvtv):
        info["bvtv_after"] = bvtv(v)
        return v, info

    info["thin_applied"] = True

    st = ndi.generate_binary_structure(3, 1)

    # Precompute EDT once; we re-use it and threshold it.
    dt = ndi.distance_transform_edt(v.astype(bool)).astype(np.float32)
    if float(dt.max()) <= 0:
        info["bvtv_after"] = bvtv(v)
        return v, info

    # We'll search for an EDT threshold that yields target BV/TV.
    # But to preserve connectivity, we relax the threshold if connectivity breaks.
    lo, hi = 0.0, float(dt.max())

    best = v.copy()
    best_bv = bvtv(best)

    for it in range(int(tp.thin_max_iter)):
        info["thin_iters"] = it + 1

        # propose a threshold aiming to hit target BV/TV
        mid = 0.5 * (lo + hi)
        cand = (dt >= mid).astype(np.uint8)

        # heal tiny cracks introduced by trimming
        if int(tp.thin_close_iters) > 0:
            cand = ndi.binary_closing(cand.astype(bool), structure=st, iterations=int(tp.thin_close_iters)).astype(np.uint8)

        cand = keep_largest_component_3d(cand, use_26=cp.use_26_connectivity)

        bv = bvtv(cand)
        connm = connectivity_metrics(cand, cp)

        # track closest valid solution
        if connm["connectivity_ok"]:
            if abs(bv - target) < abs(best_bv - target) or (best_bv > target and bv <= best_bv):
                best = cand
                best_bv = bv

        # Decide search direction:
        # If BV too high OR connectivity ok but still too high -> increase threshold (keep less)
        # If BV too low OR connectivity fails -> decrease threshold (keep more)
        if (bv > target + float(tp.thin_eps_bvtv)) and connm["connectivity_ok"]:
            lo = mid
        else:
            hi = mid

    info["bvtv_after"] = float(best_bv)
    info["connectivity_after"] = connectivity_metrics(best, cp)
    return best.astype(np.uint8), info


# -----------------------------
# Slice sampling + fragmentation scoring
# -----------------------------
def slice_fragmentation_metrics(vol01: np.ndarray, ssp: SliceSamplingParams) -> Dict[str, Any]:
    if not ssp.enable:
        return {"slice_sampling_enabled": False}

    rng = np.random.default_rng(12345)
    Z, H, W = vol01.shape
    st2 = ndi.generate_binary_structure(2, 2)  # 8-connectivity

    def sample_indices(n: int, length: int) -> List[int]:
        n = max(1, int(n))
        if length <= n:
            return list(range(length))
        if ssp.random:
            return rng.choice(length, size=n, replace=False).tolist()
        return np.linspace(0, length - 1, n).round().astype(int).tolist()

    idx_z = sample_indices(ssp.n_slices_per_axis, Z)
    idx_y = sample_indices(ssp.n_slices_per_axis, H)
    idx_x = sample_indices(ssp.n_slices_per_axis, W)

    comps = []
    def n_components_2d(s2d: np.ndarray) -> int:
        _, n = ndi.label(s2d.astype(bool), structure=st2)
        return int(n)

    for k in idx_z:
        comps.append(n_components_2d(vol01[k, :, :]))
    for k in idx_y:
        comps.append(n_components_2d(vol01[:, k, :]))
    for k in idx_x:
        comps.append(n_components_2d(vol01[:, :, k]))

    comps = np.array(comps, dtype=np.float32)
    return {
        "slice_sampling_enabled": True,
        "n_slices_total": int(comps.size),
        "mean_components_per_slice": float(comps.mean()),
        "p90_components_per_slice": float(np.percentile(comps, 90)),
        "warn_fragmentation": bool(float(comps.mean()) > float(ssp.warn_mean_components_per_slice)),
    }


# -----------------------------
# Micro-CT grayscale synthesis
# -----------------------------
def microct_gray(vol01: np.ndarray, ctp: CTParams, rng: np.random.Generator) -> np.ndarray:
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
# Morphometric proxies (kept)
# -----------------------------
def tbth_tbsp_p90_um(vol01: np.ndarray, pixel_um: float, z_um: float) -> Dict[str, Optional[float]]:
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
    bone = vol01.astype(bool)
    eul = float(euler_number(bone, connectivity=3))
    conn = float(1.0 - eul)

    voxel_vol_um3 = float(pixel_um) * float(pixel_um) * float(z_um)
    tv_mm3 = (vol01.size * voxel_vol_um3) / 1e9
    conn_d = float(conn / tv_mm3) if tv_mm3 > 0 else None
    return {"euler": eul, "conn": conn, "conn_d_per_mm3": conn_d}

def da_proxy(vol01: np.ndarray, pixel_um: float, z_um: float) -> Dict[str, Optional[float]]:
    coords = np.argwhere(vol01 > 0)
    if coords.shape[0] < 10:
        return {"da_proxy": 0.0, "axis_ratio_c_over_a": 1.0}

    z = coords[:, 0].astype(np.float64) * float(z_um)
    y = coords[:, 1].astype(np.float64) * float(pixel_um)
    x = coords[:, 2].astype(np.float64) * float(pixel_um)
    P = np.stack([x, y, z], axis=1)
    P = P - P.mean(axis=0, keepdims=True)
    C = (P.T @ P) / max(1, (P.shape[0] - 1))
    w = np.linalg.eigvalsh(C)
    w = np.clip(w, 1e-12, None)
    a, b, c = np.sqrt(np.sort(w))
    ratio = float(c / a) if a > 0 else 1.0
    da = float(1.0 - (a / c)) if c > 0 else 0.0
    da = float(np.clip(da, 0.0, 1.0))
    return {"da_proxy": da, "axis_ratio_c_over_a": ratio}


# -----------------------------
# CLI
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="v8 synthetic trabecular generator (connectivity-first).")
    p.add_argument("--outdir", type=str, default="data/v8_randomfield_connectivity")
    p.add_argument("--n-volumes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--xy", type=int, default=256)
    p.add_argument("--z", type=int, default=160)

    p.add_argument("--pixel-um", type=float, default=10.0)
    p.add_argument("--z-step-um", type=float, default=10.0)

    p.add_argument("--bvtv", type=float, default=0.18)
    p.add_argument("--invert-phase", type=int, default=0, help="1=bone is the low phase (field<thr). Often improves mesh connectivity.")

    # field knobs
    p.add_argument("--sigma-x", type=float, default=2.6)
    p.add_argument("--sigma-y", type=float, default=2.6)
    p.add_argument("--sigma-z", type=float, default=2.2)
    p.add_argument("--multiscale", type=int, default=0)
    p.add_argument("--sigma2-x", type=float, default=9.0)
    p.add_argument("--sigma2-y", type=float, default=9.0)
    p.add_argument("--sigma2-z", type=float, default=6.0)
    p.add_argument("--mix2", type=float, default=0.35)
    p.add_argument("--nonlinearity", type=float, default=0.0)

    # light morphology
    p.add_argument("--close-iters", type=int, default=1)
    p.add_argument("--open-iters", type=int, default=0)
    p.add_argument("--keep-largest", type=int, default=1)

    # connectivity targets
    p.add_argument("--min-lcc-fraction", type=float, default=0.985)
    p.add_argument("--max-components", type=int, default=5000)
    p.add_argument("--use-26-connectivity", type=int, default=1)
    p.add_argument("--enforce-perc", type=int, default=1)
    p.add_argument("--perc-axes", type=str, default="xyz")

    # bridging
    p.add_argument("--bridge-max-attempts", type=int, default=6)
    p.add_argument("--bridge-dilate-step", type=int, default=1)
    p.add_argument("--bridge-close-step", type=int, default=1)
    p.add_argument("--bridge-open-iters", type=int, default=0)

    # thinning back to BV/TV
    p.add_argument("--thin-to-bvtv", type=int, default=1)
    p.add_argument("--thin-max-iter", type=int, default=12)
    p.add_argument("--thin-close-iters", type=int, default=1)

    # slice sampling
    p.add_argument("--slice-sampling", type=int, default=1)
    p.add_argument("--n-slices-per-axis", type=int, default=6)
    p.add_argument("--slice-random", type=int, default=0)
    p.add_argument("--warn-mean-components", type=float, default=20.0)

    # microCT gray
    p.add_argument("--write-gray", type=int, default=1)
    p.add_argument("--pve-sigma", type=float, default=1.0)
    p.add_argument("--bone-mean", type=float, default=215.0)
    p.add_argument("--marrow-mean", type=float, default=50.0)
    p.add_argument("--ct-noise-sd", type=float, default=6.0)
    p.add_argument("--bg-texture-sd", type=float, default=3.0)
    p.add_argument("--unsharp", type=float, default=0.7)

    # exports
    p.add_argument("--export-2d", type=int, default=1)
    p.add_argument("--export-mip", type=int, default=0)
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

    cp = ConnectivityParams(
        min_lcc_fraction=float(args.min_lcc_fraction),
        max_components=int(args.max_components),
        use_26_connectivity=bool(int(args.use_26_connectivity)),
        enforce_perc=bool(int(args.enforce_perc)),
        perc_axes=str(args.perc_axes),
    )

    bp = BridgeParams(
        bridge_max_attempts=int(args.bridge_max_attempts),
        bridge_dilate_step=int(args.bridge_dilate_step),
        bridge_close_step=int(args.bridge_close_step),
        bridge_open_iters=int(args.bridge_open_iters),
    )

    thp = ThinParams(
        thin_to_bvtv=bool(int(args.thin_to_bvtv)),
        thin_max_iter=int(args.thin_max_iter),
        thin_close_iters=int(args.thin_close_iters),
    )

    ssp = SliceSamplingParams(
        enable=bool(int(args.slice_sampling)),
        n_slices_per_axis=int(args.n_slices_per_axis),
        random=bool(int(args.slice_random)),
        warn_mean_components_per_slice=float(args.warn_mean_components),
    )

    csv_path = outdir / "volumes.csv"
    fields = [
        "volume_id",
        "mask_tif","gray_tif","mid_png","gray_mid_png",
        "xy","z","bvtv_target","bvtv_initial","bvtv_after_bridge","bvtv_final",
        "invert_phase",
        "sigma_x","sigma_y","sigma_z","multiscale","nonlinearity",
        "min_lcc_fraction","enforce_perc","perc_axes",
        "bridge_attempts",
        "n_components","lcc_fraction","percolate_x","percolate_y","percolate_z","connectivity_ok",
        "thin_iters",
        "mean_components_per_slice","p90_components_per_slice","warn_fragmentation",
        "tbth_um_p90","tbsp_um_p90","euler","conn","conn_d_per_mm3","da_proxy","axis_ratio_c_over_a",
        "seed",
    ]
    f_csv, w_csv = init_csv(csv_path, fields)

    try:
        for i in range(int(args.n_volumes)):
            vid = f"vol_{i:05d}"

            # 1) GRF
            field = generate_grf_3d(Z=Z, H=H, W=W, fp=fp, rng=rng)

            # 2) threshold to BV/TV (phase selectable)
            invert_phase = bool(int(args.invert_phase))
            vol01, thr = adjust_bvtv_by_threshold(field, target_bvtv=float(args.bvtv), invert_phase=invert_phase, max_iter=12)
            bvtv_initial = bvtv(vol01)

            # 3) light morphology
            vol01 = postprocess(vol01, mp=mp)

            # 4) connectivity-first bridging
            vol01, bridge_info = enforce_connectivity_by_bridging(vol01, cp=cp, bp=bp)
            bvtv_after_bridge = bvtv(vol01)

            # 5) thin back to BV/TV while preserving connectivity
            vol01, thin_info = thin_preserve_connectivity(vol01, target_bvtv=float(args.bvtv), cp=cp, tp=thp)
            bvtv_final = bvtv(vol01)

            # 6) connectivity metrics
            connm = connectivity_metrics(vol01, cp)

            # 7) slice sampling metrics
            slicem = slice_fragmentation_metrics(vol01, ssp)

            # 8) microCT gray
            gray = None
            if ctp.write_gray:
                gray = microct_gray(vol01, ctp=ctp, rng=rng)

            # 9) metrics
            ts = tbth_tbsp_p90_um(vol01, pixel_um=pixel_um, z_um=z_um)
            cd = conn_density_euler(vol01, pixel_um=pixel_um, z_um=z_um)
            da = da_proxy(vol01, pixel_um=pixel_um, z_um=z_um)

            # 10) save
            mask_tif = outdir / f"{vid}_mask.tif"
            save_stack_u8((vol01 * 255).astype(np.uint8), mask_tif)

            gray_tif_name = ""
            if gray is not None:
                gray_tif = outdir / f"{vid}_gray.tif"
                save_stack_u8(gray.astype(np.uint8), gray_tif)
                gray_tif_name = gray_tif.name

            mid_png = ""
            gray_mid_png = ""
            if bool(int(args.export_2d)):
                zmid = Z // 2
                mid_png = f"{vid}_mid.png"
                save_png((vol01[zmid] * 255).astype(np.uint8), outdir / mid_png)
                if gray is not None:
                    gray_mid_png = f"{vid}_gray_mid.png"
                    save_png(gray[zmid].astype(np.uint8), outdir / gray_mid_png)

            # JSON
            meta = {
                "volume_id": vid,
                "files": {"mask_tif": mask_tif.name, "gray_tif": (gray_tif_name or None), "mid_png": (mid_png or None), "gray_mid_png": (gray_mid_png or None)},
                "size": {"xy": H, "z": Z},
                "voxel_um": {"pixel_um": pixel_um, "z_step_um": z_um},
                "params": {"field": asdict(fp), "morph": asdict(mp), "ct": asdict(ctp), "connectivity": asdict(cp), "bridge": asdict(bp), "thin": asdict(thp)},
                "threshold_value": float(thr),
                "bvtv": {"target": float(args.bvtv), "initial": float(bvtv_initial), "after_bridge": float(bvtv_after_bridge), "final": float(bvtv_final)},
                "bridge_info": bridge_info,
                "thin_info": thin_info,
                "metrics": {"connectivity": connm, "slice_fragmentation": slicem, **ts, **cd, **da},
                "seed": int(args.seed),
            }
            with open(outdir / f"{vid}.json", "w") as f:
                json.dump(meta, f, indent=2)

            # CSV
            w_csv.writerow({
                "volume_id": vid,
                "mask_tif": mask_tif.name,
                "gray_tif": gray_tif_name,
                "mid_png": mid_png,
                "gray_mid_png": gray_mid_png,

                "xy": H,
                "z": Z,
                "bvtv_target": float(args.bvtv),
                "bvtv_initial": float(bvtv_initial),
                "bvtv_after_bridge": float(bvtv_after_bridge),
                "bvtv_final": float(bvtv_final),

                "invert_phase": int(invert_phase),

                "sigma_x": fp.sigma_x,
                "sigma_y": fp.sigma_y,
                "sigma_z": fp.sigma_z,
                "multiscale": int(fp.multiscale),
                "nonlinearity": fp.nonlinearity,

                "min_lcc_fraction": cp.min_lcc_fraction,
                "enforce_perc": int(cp.enforce_perc),
                "perc_axes": cp.perc_axes,

                "bridge_attempts": int(bridge_info.get("bridge_attempts", 0)),

                "n_components": int(connm.get("n_components", 0)),
                "lcc_fraction": safe_float(connm.get("lcc_fraction", 0.0)),
                "percolate_x": int(bool(connm.get("percolate_x", False))),
                "percolate_y": int(bool(connm.get("percolate_y", False))),
                "percolate_z": int(bool(connm.get("percolate_z", False))),
                "connectivity_ok": int(bool(connm.get("connectivity_ok", False))),

                "thin_iters": int(thin_info.get("thin_iters", 0)),

                "mean_components_per_slice": safe_float(slicem.get("mean_components_per_slice", 0.0)),
                "p90_components_per_slice": safe_float(slicem.get("p90_components_per_slice", 0.0)),
                "warn_fragmentation": int(bool(slicem.get("warn_fragmentation", False))),

                "tbth_um_p90": ts["tbth_um_p90"],
                "tbsp_um_p90": ts["tbsp_um_p90"],
                "euler": cd["euler"],
                "conn": cd["conn"],
                "conn_d_per_mm3": cd["conn_d_per_mm3"],
                "da_proxy": da["da_proxy"],
                "axis_ratio_c_over_a": da["axis_ratio_c_over_a"],

                "seed": int(args.seed),
            })

            # Console summary
            print(
                f"[{i+1}/{args.n_volumes}] {vid} | BV/TV {bvtv_initial:.3f}->{bvtv_after_bridge:.3f}->{bvtv_final:.3f} | "
                f"LCC={connm.get('lcc_fraction',0.0):.3f} comps={connm.get('n_components',0)} ok={connm.get('connectivity_ok',False)} | "
                f"Perc(x,y,z)=({int(connm.get('percolate_x',0))},{int(connm.get('percolate_y',0))},{int(connm.get('percolate_z',0))}) | "
                f"bridge={bridge_info.get('bridge_attempts',0)} thin={thin_info.get('thin_iters',0)} | "
                f"slices mean comps={safe_float(slicem.get('mean_components_per_slice',0.0)):.1f}"
            )

    finally:
        f_csv.close()


if __name__ == "__main__":
    main()
