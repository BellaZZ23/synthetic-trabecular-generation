#!/usr/bin/env python3
r"""
synthetic_trabecular_v9_randomfield_fullfov.py

v9 Synthetic trabecular generator (FULL FOV) based on your v7, with fixes to help
achieve trabecular-like interconnected white "bone" networks:

Key v9 changes (per your request):
1) Add --invert-phase (bone = field < thr) option.
   - GRFs often produce better mesh connectivity in the opposite phase.
2) Stabilize connectivity fix-up:
   - DO NOT multiply dilation/closing by attempt (no runaway blobs).
   - Apply fixed dilation + fixed closing each attempt.
3) Avoid deleting networks during fix-up:
   - keep-largest is applied ONLY ONCE at the end (optional),
     not inside each fix attempt.
4) Add optional "bridge" morphology that is connectivity-friendly:
   - fixed dilation then fixed closing during attempts.
5) Keep v7 logging (JSON + CSV) and slice sampling.

Dependencies:
- numpy, pillow, tifffile
- scipy (required)
- scikit-image (euler_number)

PowerShell quick run (try both phases):
python .\synthetic_trabecular_v9_randomfield_fullfov.py `
  --outdir data\v9_test0 --n-volumes 1 --xy 512 --z 256 --seed 21 `
  --bvtv 0.22 --invert-phase 0 `
  --sigma-x 4.5 --sigma-y 4.5 --sigma-z 3.5 --multiscale 1 --sigma2-x 11 --sigma2-y 11 --sigma2-z 8 --mix2 0.60 `
  --close-iters 2 --open-iters 0 --keep-largest 0 `
  --auto-fix-connectivity 1 --fix-max-attempts 12 --fix-dilate-iters 1 --fix-close-iters 2 `
  --min-lcc-fraction 0.85 --check-percolation 1 --export-2d 1 --export-mip 1

Then switch:
  --invert-phase 1
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
    keep_largest: bool = True  # applied ONLY once at the end in v9


@dataclass
class FixupParams:
    auto_fix_connectivity: bool = True
    fix_max_attempts: int = 8

    # v9: fixed, stable steps per attempt (no multiplying by attempt)
    fix_dilate_iters: int = 0
    fix_close_iters: int = 1

    # run a light opening before attempts? Usually False for connectivity
    fix_open_iters: int = 0

    # after all attempts, optionally do one last close to smooth
    final_close_iters: int = 0


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
    min_lcc_fraction: float = 0.90
    max_components: int = 20000
    use_26_connectivity: bool = True
    check_percolation: bool = True


@dataclass
class SliceSamplingParams:
    enable: bool = True
    n_slices_per_axis: int = 8
    random: bool = False
    warn_mean_components_per_slice: float = 25.0


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
    Robust threshold solve using binary search on threshold values.

    invert_phase=False: bone = field >= thr
    invert_phase=True : bone = field <  thr
    """
    target = float(np.clip(target_bvtv, 0.001, 0.999))
    lo, hi = float(field.min()), float(field.max())

    # initial guess
    thr = float(np.quantile(field, 1.0 - target))

    for _ in range(max_iter):
        if invert_phase:
            vol01 = (field < thr).astype(np.uint8)
        else:
            vol01 = (field >= thr).astype(np.uint8)

        frac = float(vol01.mean())
        if abs(frac - target) < 5e-4:
            return vol01, thr

        # move threshold depending on phase
        if frac > target:
            # too much bone -> reduce bone
            if invert_phase:
                # decrease thr => fewer values < thr
                hi = thr
                thr = 0.5 * (lo + thr)
            else:
                # increase thr => fewer values >= thr
                lo = thr
                thr = 0.5 * (thr + hi)
        else:
            # too little bone -> increase bone
            if invert_phase:
                lo = thr
                thr = 0.5 * (thr + hi)
            else:
                hi = thr
                thr = 0.5 * (lo + thr)

    if invert_phase:
        vol01 = (field < thr).astype(np.uint8)
    else:
        vol01 = (field >= thr).astype(np.uint8)
    return vol01, thr


# -----------------------------
# Connectivity validation + percolation
# -----------------------------
def connected_components_3d(bone: np.ndarray, use_26: bool = True) -> Tuple[np.ndarray, int, np.ndarray]:
    st = np.ones((3, 3, 3), dtype=bool) if use_26 else ndi.generate_binary_structure(3, 1)
    lab, n = ndi.label(bone.astype(bool), structure=st)
    counts = np.bincount(lab.ravel())
    return lab, int(n), counts

def percolation_metrics(labels: np.ndarray, n: int) -> Dict[str, Any]:
    if n <= 0:
        return {"percolate_x": False, "percolate_y": False, "percolate_z": False}

    Z, H, W = labels.shape
    left = set(np.unique(labels[:, :, 0])) - {0}
    right = set(np.unique(labels[:, :, W - 1])) - {0}
    top = set(np.unique(labels[:, 0, :])) - {0}
    bottom = set(np.unique(labels[:, H - 1, :])) - {0}
    front = set(np.unique(labels[0, :, :])) - {0}
    back = set(np.unique(labels[Z - 1, :, :])) - {0}

    return {
        "percolate_x": bool(len(left & right) > 0),
        "percolate_y": bool(len(top & bottom) > 0),
        "percolate_z": bool(len(front & back) > 0),
    }

def connectivity_metrics(vol01: np.ndarray, cp: ConnectivityParams) -> Dict[str, Any]:
    bone = vol01.astype(bool)
    lab, n, counts = connected_components_3d(bone, use_26=cp.use_26_connectivity)

    if bone.sum() == 0:
        out = {"n_components": 0, "lcc_fraction": 0.0, "lcc_size": 0, "bone_voxels": 0, "connectivity_ok": False}
        if cp.check_percolation:
            out.update({"percolate_x": False, "percolate_y": False, "percolate_z": False})
        return out

    lcc_size = int(np.max(counts[1:])) if (n > 0 and len(counts) > 1) else int(bone.sum())
    bone_vox = int(bone.sum())
    lcc_fraction = float(lcc_size / max(1, bone_vox))

    ok = (lcc_fraction >= float(cp.min_lcc_fraction)) and (n <= int(cp.max_components))
    out = {
        "n_components": int(n),
        "lcc_fraction": float(lcc_fraction),
        "lcc_size": int(lcc_size),
        "bone_voxels": int(bone_vox),
        "connectivity_ok": bool(ok),
    }
    if cp.check_percolation:
        out.update(percolation_metrics(lab, n))
    return out

def keep_largest_component_3d(vol01: np.ndarray, use_26: bool = True) -> np.ndarray:
    bone = vol01.astype(bool)
    lab, n, counts = connected_components_3d(bone, use_26=use_26)
    if n <= 1 or len(counts) <= 1:
        return vol01
    lcc = int(np.argmax(counts[1:]) + 1)
    return (lab == lcc).astype(np.uint8)


# -----------------------------
# Morphology (base) + Fix-up loop (v9 stable + no deletion)
# -----------------------------
def postprocess_base(vol01: np.ndarray, mp: MorphParams) -> np.ndarray:
    v = vol01.astype(bool)
    st = ndi.generate_binary_structure(3, 1)

    if int(mp.open_iters) > 0:
        v = ndi.binary_opening(v, structure=st, iterations=int(mp.open_iters))
    if int(mp.close_iters) > 0:
        v = ndi.binary_closing(v, structure=st, iterations=int(mp.close_iters))

    return v.astype(np.uint8)

def fix_connectivity_stable(vol01: np.ndarray, fx: FixupParams, cp: ConnectivityParams) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    v9 fix-up:
    - Uses fixed dilation/closing per attempt (stable)
    - Does NOT apply keep-largest during attempts (prevents wiping out networks)
    - Returns best attempt (highest LCC fraction) even if threshold not met
    """
    m0 = connectivity_metrics(vol01, cp)
    if (not fx.auto_fix_connectivity) or m0["connectivity_ok"]:
        return vol01, {"fix_applied": False, "fix_attempts": 0, "connectivity": m0}

    st = ndi.generate_binary_structure(3, 1)
    best = vol01.copy().astype(np.uint8)
    best_m = m0
    best_score = safe_float(m0.get("lcc_fraction", 0.0))

    v = vol01.astype(bool)

    for attempt in range(1, int(fx.fix_max_attempts) + 1):
        # optional dilation
        if int(fx.fix_dilate_iters) > 0:
            v = ndi.binary_dilation(v, structure=st, iterations=int(fx.fix_dilate_iters))
        # optional opening (rarely helpful for connectivity)
        if int(fx.fix_open_iters) > 0:
            v = ndi.binary_opening(v, structure=st, iterations=int(fx.fix_open_iters))
        # closing bridges gaps
        if int(fx.fix_close_iters) > 0:
            v = ndi.binary_closing(v, structure=st, iterations=int(fx.fix_close_iters))

        cand = v.astype(np.uint8)
        m = connectivity_metrics(cand, cp)
        score = safe_float(m.get("lcc_fraction", 0.0))

        if score > best_score:
            best = cand.copy()
            best_m = m
            best_score = score

        if m["connectivity_ok"]:
            # optional final smoothing close
            if int(fx.final_close_iters) > 0:
                cand = ndi.binary_closing(cand.astype(bool), structure=st, iterations=int(fx.final_close_iters)).astype(np.uint8)
                m = connectivity_metrics(cand, cp)
            return cand, {"fix_applied": True, "fix_attempts": attempt, "connectivity": m}

    # return best found
    return best, {"fix_applied": True, "fix_attempts": int(fx.fix_max_attempts), "connectivity": best_m}


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
    bone_fracs = []

    def n_components_2d(slice2d: np.ndarray) -> int:
        _, n = ndi.label(slice2d.astype(bool), structure=st2)
        return int(n)

    for k in idx_z:
        s = vol01[k, :, :]
        comps.append(n_components_2d(s))
        bone_fracs.append(float(s.mean()))
    for k in idx_y:
        s = vol01[:, k, :]
        comps.append(n_components_2d(s))
        bone_fracs.append(float(s.mean()))
    for k in idx_x:
        s = vol01[:, :, k]
        comps.append(n_components_2d(s))
        bone_fracs.append(float(s.mean()))

    comps = np.array(comps, dtype=np.float32)
    bone_fracs = np.array(bone_fracs, dtype=np.float32)

    return {
        "slice_sampling_enabled": True,
        "n_slices_total": int(comps.size),
        "mean_components_per_slice": float(comps.mean()),
        "p90_components_per_slice": float(np.percentile(comps, 90)),
        "mean_slice_bone_fraction": float(bone_fracs.mean()),
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
# Morphometric proxies
# -----------------------------
def tbth_tbsp_p90_um(vol01: np.ndarray, pixel_um: float, z_um: float) -> Dict[str, Optional[float]]:
    bone = vol01.astype(bool)
    if bone.sum() == 0 or (~bone).sum() == 0:
        return {"tbth_um_p90": 0.0, "tbsp_um_p90": 0.0}
    sampling = (float(z_um), float(pixel_um), float(pixel_um))
    dt_bone = ndi.distance_transform_edt(bone, sampling=sampling)
    dt_marrow = ndi.distance_transform_edt(~bone, sampling=sampling)
    return {
        "tbth_um_p90": float(np.percentile(dt_bone[bone], 90)),
        "tbsp_um_p90": float(np.percentile(dt_marrow[~bone], 90)),
    }

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
    return {"da_proxy": float(np.clip(da, 0.0, 1.0)), "axis_ratio_c_over_a": ratio}


# -----------------------------
# CLI
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="v9 synthetic trabecular generator (GRF, full FOV, invert phase + stable fix-up).")
    p.add_argument("--outdir", type=str, default="data/v9_randomfield_fullfov")
    p.add_argument("--n-volumes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--xy", type=int, default=256)
    p.add_argument("--z", type=int, default=160)

    p.add_argument("--pixel-um", type=float, default=10.0)
    p.add_argument("--z-step-um", type=float, default=10.0)

    p.add_argument("--bvtv", type=float, default=0.20)
    p.add_argument("--invert-phase", type=int, default=0, help="1: bone=field<thr, 0: bone=field>=thr")

    # field knobs
    p.add_argument("--sigma-x", type=float, default=4.5)
    p.add_argument("--sigma-y", type=float, default=4.5)
    p.add_argument("--sigma-z", type=float, default=3.5)
    p.add_argument("--multiscale", type=int, default=1)
    p.add_argument("--sigma2-x", type=float, default=11.0)
    p.add_argument("--sigma2-y", type=float, default=11.0)
    p.add_argument("--sigma2-z", type=float, default=8.0)
    p.add_argument("--mix2", type=float, default=0.60)
    p.add_argument("--nonlinearity", type=float, default=0.0)

    # base morphology
    p.add_argument("--close-iters", type=int, default=2)
    p.add_argument("--open-iters", type=int, default=0)
    p.add_argument("--keep-largest", type=int, default=0, help="Applied only at the end (after fix-up).")

    # fix-up (stable)
    p.add_argument("--auto-fix-connectivity", type=int, default=1)
    p.add_argument("--fix-max-attempts", type=int, default=12)
    p.add_argument("--fix-dilate-iters", type=int, default=1, help="Fixed dilation per attempt.")
    p.add_argument("--fix-close-iters", type=int, default=2, help="Fixed closing per attempt.")
    p.add_argument("--fix-open-iters", type=int, default=0)
    p.add_argument("--final-close-iters", type=int, default=0)

    # connectivity thresholds
    p.add_argument("--min-lcc-fraction", type=float, default=0.85)
    p.add_argument("--max-components", type=int, default=20000)
    p.add_argument("--use-26-connectivity", type=int, default=1)
    p.add_argument("--check-percolation", type=int, default=1)

    # slice sampling
    p.add_argument("--slice-sampling", type=int, default=1)
    p.add_argument("--n-slices-per-axis", type=int, default=10)
    p.add_argument("--slice-random", type=int, default=0)
    p.add_argument("--warn-mean-components", type=float, default=25.0)

    # microCT gray
    p.add_argument("--write-gray", type=int, default=1)
    p.add_argument("--pve-sigma", type=float, default=1.3)
    p.add_argument("--bone-mean", type=float, default=215.0)
    p.add_argument("--marrow-mean", type=float, default=50.0)
    p.add_argument("--ct-noise-sd", type=float, default=5.0)
    p.add_argument("--bg-texture-sd", type=float, default=2.0)
    p.add_argument("--unsharp", type=float, default=0.5)

    # exports
    p.add_argument("--export-2d", type=int, default=1)
    p.add_argument("--export-mip", type=int, default=1)
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

    fx = FixupParams(
        auto_fix_connectivity=bool(int(args.auto_fix_connectivity)),
        fix_max_attempts=int(args.fix_max_attempts),
        fix_dilate_iters=int(args.fix_dilate_iters),
        fix_close_iters=int(args.fix_close_iters),
        fix_open_iters=int(args.fix_open_iters),
        final_close_iters=int(args.final_close_iters),
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
        check_percolation=bool(int(args.check_percolation)),
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
        "mask_tif","gray_tif","mid_png","gray_mid_png","mip_png","gray_mip_png",
        "xy","z","pixel_um","z_step_um",
        "bvtv_target","bvtv_actual","thr_value","invert_phase",
        "sigma_x","sigma_y","sigma_z","multiscale","sigma2_x","sigma2_y","sigma2_z","mix2","nonlinearity",
        "close_iters","open_iters","keep_largest",
        "fix_attempts","fix_dilate_iters","fix_close_iters",
        "n_components","lcc_fraction","percolate_x","percolate_y","percolate_z","connectivity_ok",
        "mean_components_per_slice","p90_components_per_slice","warn_fragmentation",
        "tbth_um_p90","tbsp_um_p90","euler","conn","conn_d_per_mm3","da_proxy","axis_ratio_c_over_a",
        "pve_sigma","ct_noise_sd","bg_texture_sd","unsharp",
        "seed",
    ]
    f_csv, w_csv = init_csv(csv_path, fields)

    try:
        for i in range(int(args.n_volumes)):
            vid = f"vol_{i:05d}"

            # 1) GRF
            field = generate_grf_3d(Z=Z, H=H, W=W, fp=fp, rng=rng)

            # 2) threshold (with phase option)
            invert_phase = bool(int(args.invert_phase))
            vol01, thr = adjust_bvtv_by_threshold(field, target_bvtv=float(args.bvtv), invert_phase=invert_phase, max_iter=12)

            # 3) base morphology (no keep-largest here)
            vol01 = postprocess_base(vol01, mp=mp)

            # 4) fix-up (stable, no keep-largest during attempts)
            vol01, fix_info = fix_connectivity_stable(vol01, fx=fx, cp=cp)
            connm = fix_info["connectivity"]
            fix_attempts = int(fix_info.get("fix_attempts", 0))

            # 5) apply keep-largest ONCE at the end if requested
            if mp.keep_largest:
                vol01 = keep_largest_component_3d(vol01, use_26=cp.use_26_connectivity)
                connm = connectivity_metrics(vol01, cp)  # refresh

            # 6) slice metrics
            slicem = slice_fragmentation_metrics(vol01, ssp)

            # 7) standard metrics
            bvtv_actual = bvtv(vol01)
            ts = tbth_tbsp_p90_um(vol01, pixel_um=pixel_um, z_um=z_um)
            cd = conn_density_euler(vol01, pixel_um=pixel_um, z_um=z_um)
            da = da_proxy(vol01, pixel_um=pixel_um, z_um=z_um)

            # 8) grayscale
            gray = None
            if ctp.write_gray:
                gray = microct_gray(vol01, ctp=ctp, rng=rng)

            # 9) save
            mask_tif = outdir / f"{vid}_mask.tif"
            save_stack_u8((vol01 * 255).astype(np.uint8), mask_tif)

            gray_tif_name = ""
            if gray is not None:
                gray_tif = outdir / f"{vid}_gray.tif"
                save_stack_u8(gray.astype(np.uint8), gray_tif)
                gray_tif_name = gray_tif.name

            mid_png = ""
            gray_mid_png = ""
            mip_png = ""
            gray_mip_png = ""

            if bool(int(args.export_2d)):
                zmid = Z // 2
                mid_png = f"{vid}_mid.png"
                save_png((vol01[zmid] * 255).astype(np.uint8), outdir / mid_png)
                if gray is not None:
                    gray_mid_png = f"{vid}_gray_mid.png"
                    save_png(gray[zmid].astype(np.uint8), outdir / gray_mid_png)

            if bool(int(args.export_mip)):
                mip_png = f"{vid}_mip.png"
                save_png((vol01.max(axis=0) * 255).astype(np.uint8), outdir / mip_png)
                if gray is not None:
                    gray_mip_png = f"{vid}_gray_mip.png"
                    save_png(gray.max(axis=0).astype(np.uint8), outdir / gray_mip_png)

            # JSON
            meta = {
                "volume_id": vid,
                "files": {
                    "mask_tif": mask_tif.name,
                    "gray_tif": gray_tif_name or None,
                    "mid_png": mid_png or None,
                    "gray_mid_png": gray_mid_png or None,
                    "mip_png": mip_png or None,
                    "gray_mip_png": gray_mip_png or None,
                },
                "size": {"xy": H, "z": Z},
                "voxel_um": {"pixel_um": pixel_um, "z_step_um": z_um},
                "params": {
                    "field": asdict(fp),
                    "morph": asdict(mp),
                    "fixup": asdict(fx),
                    "ct": asdict(ctp),
                    "connectivity": asdict(cp),
                    "slice_sampling": asdict(ssp),
                    "bvtv_target": float(args.bvtv),
                    "invert_phase": bool(invert_phase),
                    "threshold_value": float(thr),
                },
                "metrics": {
                    "bvtv_actual": float(bvtv_actual),
                    "connectivity": connm,
                    "connectivity_fix": fix_info,
                    "slice_fragmentation": slicem,
                    **ts, **cd, **da,
                },
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
                "mip_png": mip_png,
                "gray_mip_png": gray_mip_png,

                "xy": H,
                "z": Z,
                "pixel_um": pixel_um,
                "z_step_um": z_um,

                "bvtv_target": float(args.bvtv),
                "bvtv_actual": float(bvtv_actual),
                "thr_value": float(thr),
                "invert_phase": int(invert_phase),

                "sigma_x": fp.sigma_x,
                "sigma_y": fp.sigma_y,
                "sigma_z": fp.sigma_z,
                "multiscale": int(fp.multiscale),
                "sigma2_x": fp.sigma2_x,
                "sigma2_y": fp.sigma2_y,
                "sigma2_z": fp.sigma2_z,
                "mix2": fp.mix2,
                "nonlinearity": fp.nonlinearity,

                "close_iters": mp.close_iters,
                "open_iters": mp.open_iters,
                "keep_largest": int(mp.keep_largest),

                "fix_attempts": fix_attempts,
                "fix_dilate_iters": fx.fix_dilate_iters,
                "fix_close_iters": fx.fix_close_iters,

                "n_components": int(connm.get("n_components", 0)),
                "lcc_fraction": safe_float(connm.get("lcc_fraction", 0.0)),
                "percolate_x": int(bool(connm.get("percolate_x", False))),
                "percolate_y": int(bool(connm.get("percolate_y", False))),
                "percolate_z": int(bool(connm.get("percolate_z", False))),
                "connectivity_ok": int(bool(connm.get("connectivity_ok", False))),

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

                "pve_sigma": ctp.pve_sigma,
                "ct_noise_sd": ctp.ct_noise_sd,
                "bg_texture_sd": ctp.bg_texture_sd,
                "unsharp": ctp.unsharp,

                "seed": int(args.seed),
            })

            print(
                f"[{i+1}/{args.n_volumes}] {vid} | BV/TV={bvtv_actual:.3f} | invert={int(invert_phase)} | "
                f"LCC={connm.get('lcc_fraction',0.0):.3f} (ok={bool(connm.get('connectivity_ok',False))}, comps={int(connm.get('n_components',0))}, fix={fix_attempts}) | "
                f"Percolate(x,y,z)=({int(connm.get('percolate_x',0))},{int(connm.get('percolate_y',0))},{int(connm.get('percolate_z',0))}) | "
                f"Slices mean comps={safe_float(slicem.get('mean_components_per_slice',0.0)):.1f}"
            )

    finally:
        f_csv.close()


if __name__ == "__main__":
    main()
