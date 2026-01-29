#!/usr/bin/env python3
r"""
synthetic_trabecular_v7_randomfield_fullfov.py

v7 Synthetic trabecular generator (FULL FOV) upgrading v6 with:
1) Explicit 3D connectivity validation (largest component fraction + component count)
2) Optional percolation checks (material connects opposing faces)
3) Slice sampling & fragmentation scoring across axes (not just a single mid-slice)
4) Two-stage thresholding:
   - primary threshold hits target BV/TV on the raw GRF
   - optional postprocess re-thresholding or BV/TV correction (morph can change BV/TV)
5) Automatic "fix-up" loop: if connectivity fails -> small closing / dilation + re-check (bounded retries)
6) Template/dataset conditioning hook (optional):
   - load a real/template binary or grayscale volume
   - extract target stats (BV/TV + correlation length proxy) to guide generator knobs
7) Modular grayscale micro-CT simulation (kept but separated cleanly)
8) Rich metrics logging into JSON + CSV:
   - connectivity metrics, percolation metrics, slice fragmentation metrics

Dependencies:
- numpy, pillow, tifffile
- scipy (required)
- scikit-image (euler_number already used in v6)
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

    # v7: fix-up controls for connectivity
    auto_fix_connectivity: bool = True
    fix_max_attempts: int = 3
  # total attempts including initial postprocess
    fix_close_iters_step: int = 1  # add this many close iterations per failed attempt
    fix_dilate_iters_step: int = 0 # optionally add dilation before closing for bridging


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
    # largest connected component fraction threshold
    min_lcc_fraction: float = 0.90
    # maximum components allowed (optional guard)
    max_components: int = 2000
    # 3D connectivity: 26-connectivity for material
    use_26_connectivity: bool = True
    # percolation: check connectivity between opposite faces
    check_percolation: bool = True


@dataclass
class SliceSamplingParams:
    enable: bool = True
    n_slices_per_axis: int = 8
    random: bool = True
    # fragmentation metric thresholds (informational; not used to auto-fix)
    warn_mean_components_per_slice: float = 25.0


@dataclass
class TemplateParams:
    template_path: Optional[str] = None
    template_is_gray: bool = False
    template_threshold: int = 128  # for grayscale->binary
    # if provided, we adjust BV/TV target to template BV/TV unless user locks bvtv
    match_template_bvtv: bool = False
    # mild knob adaptation: adjust sigmas based on template correlation proxy
    adapt_sigmas: bool = False


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

def oddify(n: int) -> int:
    n = int(max(1, n))
    return n | 1

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


# -----------------------------
# Template loading + simple feature extraction
# -----------------------------
def load_template_volume(tp: TemplateParams) -> Optional[np.ndarray]:
    """
    Loads a template volume (tif stack). Returns binary (0/1) uint8.
    If template_is_gray=True, binarizes using template_threshold.
    """
    if not tp.template_path:
        return None

    p = Path(tp.template_path)
    if not p.exists():
        raise FileNotFoundError(f"Template not found: {p}")

    vol = tiff.imread(str(p))
    # Expect Z,H,W. If H,W,Z, user can pre-reorder; we won't guess silently.
    if vol.ndim != 3:
        raise ValueError(f"Template must be a 3D stack. Got shape {vol.shape}")

    if tp.template_is_gray:
        vol01 = (vol.astype(np.float32) >= float(tp.template_threshold)).astype(np.uint8)
    else:
        # accept binary 0/1 or 0/255
        v = vol.astype(np.float32)
        if v.max() > 1.5:
            v = (v >= 128).astype(np.uint8)
        else:
            v = (v > 0.5).astype(np.uint8)
        vol01 = v

    return vol01

def correlation_length_proxy(vol01: np.ndarray) -> float:
    """
    Very lightweight proxy: compute the first moment of the 3D autocorrelation
    of the binary volume after mean-centering, via FFT.
    Returns a single scalar "corr_len" in voxels (approx).
    Note: this is a coarse proxy intended to adapt sigma knobs.
    """
    x = vol01.astype(np.float32)
    x = x - x.mean()
    # avoid all zeros
    if np.allclose(x, 0.0):
        return 0.0

    F = np.fft.fftn(x)
    ac = np.fft.ifftn(np.abs(F) ** 2).real
    ac = np.fft.fftshift(ac)
    ac = ac / (ac.max() + 1e-8)

    # radial-ish summary by sampling along principal axes (fast)
    zc, yc, xc = [s // 2 for s in ac.shape]
    line_x = ac[zc, yc, :]
    line_y = ac[zc, :, xc]
    line_z = ac[:, yc, xc]

    def halfmax_width(line: np.ndarray) -> float:
        c = line.size // 2
        # find first index from center where drops below 0.5
        right = np.where(line[c:] < 0.5)[0]
        if right.size == 0:
            return float(line.size)
        return float(right[0])

    w = (halfmax_width(line_x) + halfmax_width(line_y) + halfmax_width(line_z)) / 3.0
    return float(w)

def extract_template_stats(vol01: np.ndarray) -> Dict[str, float]:
    return {
        "template_bvtv": float(np.mean(vol01 > 0)),
        "template_corr_len_vox": correlation_length_proxy(vol01),
    }

def maybe_adapt_field_params_from_template(fp: FieldParams, stats: Dict[str, float]) -> FieldParams:
    """
    Optional: adjust sigma knobs based on template correlation proxy.
    This is deliberately mild: we scale all sigmas toward the proxy value.
    """
    corr = stats.get("template_corr_len_vox", 0.0)
    if corr <= 0:
        return fp

    # current average sigma
    s_avg = (fp.sigma_x + fp.sigma_y + fp.sigma_z) / 3.0
    if s_avg <= 0:
        return fp

    # scale toward corr/2 (since GRF sigma acts like blur radius)
    target = max(0.5, corr / 2.0)
    scale = np.clip(target / s_avg, 0.5, 2.0)  # mild scaling
    fp2 = FieldParams(**asdict(fp))
    fp2.sigma_x *= float(scale)
    fp2.sigma_y *= float(scale)
    fp2.sigma_z *= float(scale)
    # keep secondary scales proportional
    fp2.sigma2_x *= float(scale)
    fp2.sigma2_y *= float(scale)
    fp2.sigma2_z *= float(scale)
    return fp2


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

def threshold_to_bvtv(field: np.ndarray, bvtv: float) -> Tuple[np.ndarray, float]:
    bvtv = float(np.clip(bvtv, 0.001, 0.999))
    thr = float(np.quantile(field, 1.0 - bvtv))
    vol01 = (field >= thr).astype(np.uint8)
    return vol01, thr

def adjust_bvtv_by_threshold(field: np.ndarray, target_bvtv: float, max_iter: int = 12) -> Tuple[np.ndarray, float]:
    """
    Robust threshold solve using binary search on threshold values.
    Useful if you want to guarantee BV/TV precisely on the *field*.
    """
    target = float(np.clip(target_bvtv, 0.001, 0.999))
    lo, hi = float(field.min()), float(field.max())
    thr = float(np.quantile(field, 1.0 - target))
    for _ in range(max_iter):
        vol01 = (field >= thr).astype(np.uint8)
        frac = float(vol01.mean())
        if abs(frac - target) < 5e-4:
            return vol01, thr
        if frac > target:
            # too much bone -> raise threshold
            lo = thr
            thr = 0.5 * (thr + hi)
        else:
            # too little bone -> lower threshold
            hi = thr
            thr = 0.5 * (lo + thr)
    vol01 = (field >= thr).astype(np.uint8)
    return vol01, thr


# -----------------------------
# Connectivity validation + percolation
# -----------------------------
def connected_components_3d(bone: np.ndarray, use_26: bool = True) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Returns labels, n_components, counts (bincount of labels).
    """
    st = np.ones((3, 3, 3), dtype=bool) if use_26 else ndi.generate_binary_structure(3, 1)  # 6-neigh
    lab, n = ndi.label(bone.astype(bool), structure=st)
    counts = np.bincount(lab.ravel())
    return lab, int(n), counts

def connectivity_metrics(vol01: np.ndarray, cp: ConnectivityParams) -> Dict[str, Any]:
    bone = vol01.astype(bool)
    lab, n, counts = connected_components_3d(bone, use_26=cp.use_26_connectivity)

    if bone.sum() == 0:
        return {
            "n_components": 0,
            "lcc_fraction": 0.0,
            "lcc_size": 0,
            "bone_voxels": 0,
            "connectivity_ok": False,
        }

    # counts[0] is background
    if len(counts) <= 1:
        lcc_size = int(bone.sum())
    else:
        lcc_size = int(np.max(counts[1:])) if n > 0 else int(bone.sum())

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

def percolation_metrics(labels: np.ndarray, n: int) -> Dict[str, Any]:
    """
    Percolation: does any *single* connected component touch opposing faces?
    Checked for x, y, z directions.
    """
    if n <= 0:
        return {"percolate_x": False, "percolate_y": False, "percolate_z": False}

    Z, H, W = labels.shape
    # labels on faces
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

def keep_largest_component_3d(vol01: np.ndarray, use_26: bool = True) -> np.ndarray:
    bone = vol01.astype(bool)
    lab, n, counts = connected_components_3d(bone, use_26=use_26)
    if n <= 1:
        return vol01
    if len(counts) <= 1:
        return vol01
    lcc = int(np.argmax(counts[1:]) + 1)
    return (lab == lcc).astype(np.uint8)


# -----------------------------
# Morphology + connectivity fix-up
# -----------------------------
def postprocess(vol01: np.ndarray, mp: MorphParams, use_26_for_largest: bool = True) -> np.ndarray:
    v = vol01.astype(bool)
    st = ndi.generate_binary_structure(3, 1)  # 6-neighborhood for morphology

    if int(mp.open_iters) > 0:
        v = ndi.binary_opening(v, structure=st, iterations=int(mp.open_iters))
    if int(mp.close_iters) > 0:
        v = ndi.binary_closing(v, structure=st, iterations=int(mp.close_iters))

    out = v.astype(np.uint8)
    if mp.keep_largest:
        out = keep_largest_component_3d(out, use_26=use_26_for_largest)
    return out

def fix_connectivity(vol01: np.ndarray, mp: MorphParams, cp: ConnectivityParams) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply bounded fix-up steps if connectivity fails.
    Strategy:
      - start with current vol01
      - if fail: optionally dilate a bit, then increase closing iterations gradually
      - always re-apply keep-largest if enabled
    """
    metrics0 = connectivity_metrics(vol01, cp)
    if metrics0["connectivity_ok"] or not mp.auto_fix_connectivity:
        return vol01, {"fix_attempts": 0, "fix_applied": False, "connectivity": metrics0}

    v = vol01.copy().astype(np.uint8)
    st = ndi.generate_binary_structure(3, 1)

    applied = False
    last_metrics = metrics0
    close_iters_base = int(mp.close_iters)

    for attempt in range(1, int(mp.fix_max_attempts) + 1):
        applied = True

        # optional tiny dilation to bridge gaps
        dil = int(mp.fix_dilate_iters_step) * attempt
        if dil > 0:
            v = ndi.binary_dilation(v.astype(bool), structure=st, iterations=dil).astype(np.uint8)

        # increase closing
        ci = close_iters_base + int(mp.fix_close_iters_step) * attempt
        if ci > 0:
            v = ndi.binary_closing(v.astype(bool), structure=st, iterations=ci).astype(np.uint8)

        # optional opening is NOT increased (opening can break connectivity), keep as-is
        if mp.keep_largest:
            v = keep_largest_component_3d(v, use_26=cp.use_26_connectivity)

        last_metrics = connectivity_metrics(v, cp)
        if last_metrics["connectivity_ok"]:
            return v.astype(np.uint8), {
                "fix_attempts": attempt,
                "fix_applied": applied,
                "connectivity": last_metrics,
                "close_iters_used": ci,
                "dilate_iters_used": dil,
            }

    # return best effort result
    return v.astype(np.uint8), {
        "fix_attempts": int(mp.fix_max_attempts),
        "fix_applied": applied,
        "connectivity": last_metrics,
        "close_iters_used": close_iters_base + int(mp.fix_close_iters_step) * int(mp.fix_max_attempts),
        "dilate_iters_used": int(mp.fix_dilate_iters_step) * int(mp.fix_max_attempts),
    }


# -----------------------------
# Slice sampling + fragmentation scoring
# -----------------------------
def slice_fragmentation_metrics(vol01: np.ndarray, ssp: SliceSamplingParams) -> Dict[str, Any]:
    if not ssp.enable:
        return {"slice_sampling_enabled": False}

    rng = np.random.default_rng(12345)  # deterministic sampling to compare runs
    Z, H, W = vol01.shape

    st2 = ndi.generate_binary_structure(2, 2)  # 8-connectivity in 2D

    def sample_indices(n: int, length: int) -> List[int]:
        n = max(1, int(n))
        if length <= n:
            return list(range(length))
        if ssp.random:
            return rng.choice(length, size=n, replace=False).tolist()
        # evenly spaced
        return np.linspace(0, length - 1, n).round().astype(int).tolist()

    idx_z = sample_indices(ssp.n_slices_per_axis, Z)
    idx_y = sample_indices(ssp.n_slices_per_axis, H)
    idx_x = sample_indices(ssp.n_slices_per_axis, W)

    comps = []
    bone_fracs = []

    def n_components_2d(slice2d: np.ndarray) -> int:
        lab, n = ndi.label(slice2d.astype(bool), structure=st2)
        return int(n)

    # z-slices (xy planes)
    for k in idx_z:
        s = vol01[k, :, :]
        comps.append(n_components_2d(s))
        bone_fracs.append(float(s.mean()))
    # y-slices (xz planes)
    for k in idx_y:
        s = vol01[:, k, :]
        comps.append(n_components_2d(s))
        bone_fracs.append(float(s.mean()))
    # x-slices (yz planes)
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
# Morphometric proxies (kept from v6)
# -----------------------------
def bvtv(vol01: np.ndarray) -> float:
    return float(np.mean(vol01 > 0))

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
    coords = np.argwhere(vol01 > 0)  # z,y,x
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
    p = argparse.ArgumentParser(description="v7 synthetic trabecular generator (GRF, full FOV, connectivity + slice validation + template hook).")
    p.add_argument("--outdir", type=str, default="data/v7_randomfield_fullfov")
    p.add_argument("--n-volumes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--xy", type=int, default=256)
    p.add_argument("--z", type=int, default=160)

    p.add_argument("--pixel-um", type=float, default=10.0)
    p.add_argument("--z-step-um", type=float, default=10.0)

    p.add_argument("--bvtv", type=float, default=0.18, help="Target BV/TV for thresholding the random field.")
    p.add_argument("--lock-bvtv", type=int, default=1, help="1=use provided --bvtv; 0=may be overridden by template if match_template_bvtv enabled.")

    # field knobs
    p.add_argument("--sigma-x", type=float, default=3.0)
    p.add_argument("--sigma-y", type=float, default=3.0)
    p.add_argument("--sigma-z", type=float, default=2.0)
    p.add_argument("--multiscale", type=int, default=1)
    p.add_argument("--sigma2-x", type=float, default=9.0)
    p.add_argument("--sigma2-y", type=float, default=9.0)
    p.add_argument("--sigma2-z", type=float, default=6.0)
    p.add_argument("--mix2", type=float, default=0.35)
    p.add_argument("--nonlinearity", type=float, default=0.0)

    # morphology/purification
    p.add_argument("--close-iters", type=int, default=1)
    p.add_argument("--open-iters", type=int, default=0)
    p.add_argument("--keep-largest", type=int, default=1)

    # v7 connectivity fix-up
    p.add_argument("--auto-fix-connectivity", type=int, default=1)
    p.add_argument("--fix-max-attempts", type=int, default=3)
    p.add_argument("--fix-close-iters-step", type=int, default=1)
    p.add_argument("--fix-dilate-iters-step", type=int, default=0)

    # connectivity thresholds
    p.add_argument("--min-lcc-fraction", type=float, default=0.90)
    p.add_argument("--max-components", type=int, default=2000)
    p.add_argument("--use-26-connectivity", type=int, default=1)
    p.add_argument("--check-percolation", type=int, default=1)

    # slice sampling
    p.add_argument("--slice-sampling", type=int, default=1)
    p.add_argument("--n-slices-per-axis", type=int, default=8)
    p.add_argument("--slice-random", type=int, default=1)
    p.add_argument("--warn-mean-components", type=float, default=25.0)

    # template hook
    p.add_argument("--template-path", type=str, default=None, help="Optional .tif 3D stack to guide stats.")
    p.add_argument("--template-is-gray", type=int, default=0)
    p.add_argument("--template-threshold", type=int, default=128)
    p.add_argument("--match-template-bvtv", type=int, default=0)
    p.add_argument("--adapt-sigmas", type=int, default=0)

    # microCT gray
    p.add_argument("--write-gray", type=int, default=1)
    p.add_argument("--pve-sigma", type=float, default=1.2)
    p.add_argument("--bone-mean", type=float, default=215.0)
    p.add_argument("--marrow-mean", type=float, default=50.0)
    p.add_argument("--ct-noise-sd", type=float, default=9.0)
    p.add_argument("--bg-texture-sd", type=float, default=4.0)
    p.add_argument("--unsharp", type=float, default=0.9)

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
        auto_fix_connectivity=bool(int(args.auto_fix_connectivity)),
        fix_max_attempts=int(args.fix_max_attempts),
        fix_close_iters_step=int(args.fix_close_iters_step),
        fix_dilate_iters_step=int(args.fix_dilate_iters_step),
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

    tp = TemplateParams(
        template_path=args.template_path,
        template_is_gray=bool(int(args.template_is_gray)),
        template_threshold=int(args.template_threshold),
        match_template_bvtv=bool(int(args.match_template_bvtv)),
        adapt_sigmas=bool(int(args.adapt_sigmas)),
    )

    # --- Optional template guidance ---
    template_stats: Optional[Dict[str, float]] = None
    template_vol01 = load_template_volume(tp)
    if template_vol01 is not None:
        template_stats = extract_template_stats(template_vol01)

        # match BV/TV
        lock_bvtv = bool(int(args.lock_bvtv))
        if tp.match_template_bvtv and (not lock_bvtv):
            args.bvtv = float(template_stats["template_bvtv"])

        # adapt sigmas
        if tp.adapt_sigmas:
            fp = maybe_adapt_field_params_from_template(fp, template_stats)

    csv_path = outdir / "volumes.csv"
    fields = [
        # identity + files
        "volume_id",
        "mask_tif",
        "gray_tif",
        "mid_png",
        "gray_mid_png",
        "mip_png",
        "gray_mip_png",
        # size
        "xy",
        "z",
        "pixel_um",
        "z_step_um",
        # targets/threshold
        "bvtv_target",
        "bvtv_actual",
        "thr_value",
        # field params
        "sigma_x","sigma_y","sigma_z",
        "multiscale","sigma2_x","sigma2_y","sigma2_z","mix2","nonlinearity",
        # morph/fix
        "close_iters","open_iters","keep_largest",
        "auto_fix_connectivity","fix_attempts",
        # connectivity
        "n_components","lcc_fraction","percolate_x","percolate_y","percolate_z","connectivity_ok",
        # slice sampling
        "mean_components_per_slice","p90_components_per_slice","warn_fragmentation",
        # morphometrics proxies
        "tbth_um_p90","tbsp_um_p90","euler","conn","conn_d_per_mm3","da_proxy","axis_ratio_c_over_a",
        # CT
        "pve_sigma","bone_mean","marrow_mean","ct_noise_sd","bg_texture_sd","unsharp",
        # template
        "template_path","template_bvtv","template_corr_len_vox",
        # seed
        "seed",
    ]
    f_csv, w_csv = init_csv(csv_path, fields)

    try:
        for i in range(int(args.n_volumes)):
            vid = f"vol_{i:05d}"

            # --- 1) Generate GRF ---
            field = generate_grf_3d(Z=Z, H=H, W=W, fp=fp, rng=rng)

            # --- 2) Threshold to BV/TV (robust threshold solve on the field) ---
            vol01, thr = adjust_bvtv_by_threshold(field, target_bvtv=float(args.bvtv), max_iter=12)

            # --- 3) Postprocess morphology ---
            vol01 = postprocess(vol01, mp=mp, use_26_for_largest=cp.use_26_connectivity)

            # --- 4) Validate + auto-fix connectivity if needed ---
            vol01, fix_info = fix_connectivity(vol01, mp=mp, cp=cp)
            connm = fix_info["connectivity"]
            fix_attempts = int(fix_info.get("fix_attempts", 0))

            # --- 5) Slice fragmentation metrics (multi-axis sampling) ---
            slicem = slice_fragmentation_metrics(vol01, ssp=ssp)

            # --- 6) Compute standard metrics ---
            bvtv_actual = bvtv(vol01)
            ts = tbth_tbsp_p90_um(vol01, pixel_um=pixel_um, z_um=z_um)
            cd = conn_density_euler(vol01, pixel_um=pixel_um, z_um=z_um)
            da = da_proxy(vol01, pixel_um=pixel_um, z_um=z_um)

            # --- 7) Grayscale microCT ---
            gray = None
            if ctp.write_gray:
                gray = microct_gray(vol01, ctp=ctp, rng=rng)

            # --- 8) Save stacks ---
            mask_tif = outdir / f"{vid}_mask.tif"
            save_stack_u8((vol01 * 255).astype(np.uint8), mask_tif)

            gray_tif_name = ""
            if gray is not None:
                gray_tif = outdir / f"{vid}_gray.tif"
                save_stack_u8(gray.astype(np.uint8), gray_tif)
                gray_tif_name = gray_tif.name

            # --- 9) Export mid slice + optional MIP ---
            mid_png = ""
            gray_mid_png = ""
            mip_png = ""
            gray_mip_png = ""

            if bool(int(args.export_2d)):
                zmid = Z // 2
                bin_mid = (vol01[zmid] * 255).astype(np.uint8)
                mid_png = f"{vid}_mid.png"
                save_png(bin_mid, outdir / mid_png)

                if gray is not None:
                    gray_mid_png = f"{vid}_gray_mid.png"
                    save_png(gray[zmid].astype(np.uint8), outdir / gray_mid_png)

            if bool(int(args.export_mip)):
                bin_mip = (vol01.max(axis=0) * 255).astype(np.uint8)
                mip_png = f"{vid}_mip.png"
                save_png(bin_mip, outdir / mip_png)

                if gray is not None:
                    gray_mip_png = f"{vid}_gray_mip.png"
                    save_png(gray.max(axis=0).astype(np.uint8), outdir / gray_mip_png)

            # --- 10) JSON metadata ---
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
                    "ct": asdict(ctp),
                    "connectivity": asdict(cp),
                    "slice_sampling": asdict(ssp),
                    "template": asdict(tp),
                    "bvtv_target": float(args.bvtv),
                    "threshold_value": float(thr),
                },
                "template_stats": template_stats,
                "metrics": {
                    "bvtv_actual": float(bvtv_actual),
                    "connectivity": connm,
                    "connectivity_fix": fix_info,
                    "slice_fragmentation": slicem,
                    **ts,
                    **cd,
                    **da,
                },
                "seed": int(args.seed),
            }
            with open(outdir / f"{vid}.json", "w") as f:
                json.dump(meta, f, indent=2)

            # --- 11) CSV row ---
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
                "auto_fix_connectivity": int(mp.auto_fix_connectivity),
                "fix_attempts": fix_attempts,

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
                "bone_mean": ctp.bone_mean,
                "marrow_mean": ctp.marrow_mean,
                "ct_noise_sd": ctp.ct_noise_sd,
                "bg_texture_sd": ctp.bg_texture_sd,
                "unsharp": ctp.unsharp,

                "template_path": tp.template_path or "",
                "template_bvtv": (template_stats["template_bvtv"] if template_stats else ""),
                "template_corr_len_vox": (template_stats["template_corr_len_vox"] if template_stats else ""),

                "seed": int(args.seed),
            })

            # --- Console summary ---
            conn_ok = bool(connm.get("connectivity_ok", False))
            mean_comp = safe_float(slicem.get("mean_components_per_slice", 0.0))
            warn_frag = bool(slicem.get("warn_fragmentation", False))

            print(
                f"[{i+1}/{args.n_volumes}] {vid} | BV/TV={bvtv_actual:.3f} | "
                f"LCC={connm.get('lcc_fraction', 0.0):.3f} (ok={conn_ok}, comps={connm.get('n_components', 0)}, fix={fix_attempts}) | "
                f"Percolate(x,y,z)=({int(connm.get('percolate_x',0))},{int(connm.get('percolate_y',0))},{int(connm.get('percolate_z',0))}) | "
                f"Slices mean comps={mean_comp:.1f} (warn={warn_frag}) | "
                f"Tb.Th~{ts['tbth_um_p90']:.1f}um | Tb.Sp~{ts['tbsp_um_p90']:.1f}um | "
                f"Conn.D~{cd['conn_d_per_mm3'] if cd['conn_d_per_mm3'] is not None else 'NA'} | "
                f"DA~{da['da_proxy']:.3f} | sig=({fp.sigma_x:.2f},{fp.sigma_y:.2f},{fp.sigma_z:.2f})"
            )

    finally:
        f_csv.close()


if __name__ == "__main__":
    main()
