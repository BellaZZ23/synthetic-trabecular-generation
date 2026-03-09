#!/usr/bin/env python3
r"""
synthetic_trabecular_v15_morphometric_control.py

v15: Pooled VOI-driven synthetic trabecular bone dataset generator.

Designed for ML dataset creation. Reads all *_targets.json from VOI1 and VOI4,
pools the morphometric statistics, then generates N synthetic volumes with
controlled variation across BV/TV, Tb.Th, Tb.N, and random seed.

Pipeline:
  1. DICOM -> pipeline_voi1 -> *_targets.json  (you already did this)
  2. This script reads ALL targets files from --voi-dirs
  3. Pools: computes mean, std, min, max for each morphometric
  4. Generates --num-samples synthetic volumes, varying parameters
  5. Each sample gets a unique seed + sampled morphometric targets
  6. Outputs: mask.tif, void.tif, gray.tif, mid.png, gray_mid.png, metrics.json

Usage (PowerShell):
  # Generate 20 synthetic samples from pooled VOI1+VOI4 data:
  python synthetic_trabecular_v15_morphometric_control.py `
      --voi-dirs data\derived\VOI1 data\derived\VOI4 `
      --outdir output\ml_dataset `
      --num-samples 20 `
      --voxel-um 39 `
      --xy 512 --z 160 `
      --solid-fill-sigma 3.0 `
      --base-seed 42

  # Quick test (small volumes):
  python synthetic_trabecular_v15_morphometric_control.py `
      --voi-dirs data\derived\VOI1 data\derived\VOI4 `
      --outdir output\ml_test `
      --num-samples 5 `
      --voxel-um 39 `
      --xy 128 --z 40 `
      --solid-fill-sigma 3.0 `
      --base-seed 42

  # Single VOI file (backward compat):
  python synthetic_trabecular_v15_morphometric_control.py `
      --targets-json data\derived\VOI1\specimen01_Specimen1_VOI1_Scan1_targets.json `
      --outdir output\single `
      --voxel-um 39 `
      --solid-fill-sigma 3.0

  # Literature fallback:
  python synthetic_trabecular_v15_morphometric_control.py `
      --profile tamimi-hf `
      --outdir output\lit `
      --solid-fill-sigma 3.0

Generation core: IDENTICAL to v14.
Safe-only fixes: A (x2 measurement), B (adaptive fill), C (VOI loading), D (ASCII labels).
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
# Literature bounds (validation only)
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
# VOI pooling: read all targets, fix voxel, compute ranges
# ---------------------------------------------------------------
def load_all_voi_targets(voi_dirs: List[str], voxel_um_override: float) -> Dict[str, Any]:
    """Load all *_targets.json from multiple VOI directories.
    Forces voxel_um to the override value (fixes the 1000um DICOM bug).
    Returns pooled statistics."""
    all_targets = []
    for d in voi_dirs:
        dp = Path(d)
        files = sorted(dp.glob("*_targets.json"))
        if not files:
            print(f"  Warning: no *_targets.json in {dp}")
            continue
        for f in files:
            with open(f) as fh:
                data = json.load(fh)
            # Force correct voxel size
            data["_voxel_um_forced"] = voxel_um_override
            data["_source_file"] = str(f)
            all_targets.append(data)
            print(f"  Loaded: {f.name}")

    if not all_targets:
        raise FileNotFoundError("No *_targets.json files found in any --voi-dirs")

    print(f"\nPooling {len(all_targets)} VOI targets files (voxel forced to {voxel_um_override}um)")

    # Extract morphometrics with voxel correction
    # The VOI pipeline reports DT values in raw units (voxel_um from DICOM).
    # If DICOM said 1000um but real voxel is 39um, we need to rescale.
    bvtv_list = []
    tbth_list = []  # will store corrected p90 values
    tbsp_list = []  # will store corrected p50 values

    for t in all_targets:
        bvtv_list.append(float(t.get("BVTV", 0)))

        # Get the raw voxel from the file
        raw_vox = 1.0
        vz = t.get("voxel_um_zyx")
        if vz and len(vz) >= 1:
            raw_vox = float(vz[0])

        # Scale factor: raw DT was computed at raw_vox, we want it at voxel_um_override
        # DT values are in raw_vox units already, so:
        #   real_um = raw_dt_value * (voxel_um_override / raw_vox)  ... NO
        # Actually the pipeline does: dt * voxel_um -> um. But voxel_um was wrong (1000).
        # So raw_dt_um = dt_voxels * 1000. Real should be dt_voxels * 39.
        # Correction: multiply by (39 / 1000)
        correction = voxel_um_override / max(1.0, raw_vox)

        tbth_raw = t.get("TbTh_um_p90", t.get("TbTh_um_p50", 0))
        tbsp_raw = t.get("TbSp_um_p50", 0)

        # Apply voxel correction, then x2 for diameter (the pipeline reports radius)
        tbth_corrected = float(tbth_raw) * correction * 2.0
        tbsp_corrected = float(tbsp_raw) * correction * 2.0

        tbth_list.append(tbth_corrected)
        tbsp_list.append(tbsp_corrected)

    bvtv_arr = np.array(bvtv_list)
    tbth_arr = np.array(tbth_list)
    tbsp_arr = np.array(tbsp_list)

    # Derive Tb.N from each specimen
    tbn_arr = bvtv_arr / (tbth_arr / 1000.0 + 1e-9)

    pooled = {
        "n_specimens": len(all_targets),
        "voxel_um": voxel_um_override,
        "BVTV":  {"mean": float(bvtv_arr.mean()), "std": float(bvtv_arr.std()),
                   "min": float(bvtv_arr.min()), "max": float(bvtv_arr.max())},
        "TbTh_um": {"mean": float(tbth_arr.mean()), "std": float(tbth_arr.std()),
                     "min": float(tbth_arr.min()), "max": float(tbth_arr.max())},
        "TbSp_um": {"mean": float(tbsp_arr.mean()), "std": float(tbsp_arr.std()),
                     "min": float(tbsp_arr.min()), "max": float(tbsp_arr.max())},
        "TbN_per_mm": {"mean": float(tbn_arr.mean()), "std": float(tbn_arr.std()),
                        "min": float(tbn_arr.min()), "max": float(tbn_arr.max())},
        "per_specimen": [
            {"file": t["_source_file"], "BVTV": b, "TbTh_um": th, "TbSp_um": sp, "TbN_per_mm": n}
            for t, b, th, sp, n in zip(all_targets, bvtv_list, tbth_list.tolist() if hasattr(tbth_list, 'tolist') else tbth_list,
                                        tbsp_list.tolist() if hasattr(tbsp_list, 'tolist') else tbsp_list,
                                        tbn_arr.tolist())
        ],
    }

    print(f"\n  Pooled ranges (corrected to {voxel_um_override}um, x2 diameter):")
    print(f"    BV/TV:  {pooled['BVTV']['mean']:.3f} +/- {pooled['BVTV']['std']:.3f}  "
          f"[{pooled['BVTV']['min']:.3f} - {pooled['BVTV']['max']:.3f}]")
    print(f"    Tb.Th:  {pooled['TbTh_um']['mean']:.1f} +/- {pooled['TbTh_um']['std']:.1f} um  "
          f"[{pooled['TbTh_um']['min']:.1f} - {pooled['TbTh_um']['max']:.1f}]")
    print(f"    Tb.Sp:  {pooled['TbSp_um']['mean']:.1f} +/- {pooled['TbSp_um']['std']:.1f} um  "
          f"[{pooled['TbSp_um']['min']:.1f} - {pooled['TbSp_um']['max']:.1f}]")
    print(f"    Tb.N:   {pooled['TbN_per_mm']['mean']:.2f} +/- {pooled['TbN_per_mm']['std']:.2f} /mm  "
          f"[{pooled['TbN_per_mm']['min']:.2f} - {pooled['TbN_per_mm']['max']:.2f}]")

    return pooled


def sample_targets_from_pool(pooled: Dict[str, Any], rng: np.random.Generator,
                             n: int) -> List[Dict[str, float]]:
    """Sample N sets of morphometric targets from the pooled distribution.
    Uses truncated normal: mean +/- 1.5*std, clamped to observed [min, max]."""
    samples = []
    for i in range(n):
        def samp(key):
            s = pooled[key]
            val = rng.normal(s["mean"], max(s["std"], s["mean"] * 0.05))
            return float(np.clip(val, s["min"] * 0.8, s["max"] * 1.2))

        bvtv = samp("BVTV")
        bvtv = float(np.clip(bvtv, 0.05, 0.60))

        tbth = samp("TbTh_um")
        tbth = float(np.clip(tbth, 60.0, 400.0))

        tbsp = samp("TbSp_um")
        tbsp = float(np.clip(tbsp, 100.0, 1500.0))

        # Derive Tb.N from BV/TV and Tb.Th (consistent relationship)
        tbn = bvtv / (tbth / 1000.0)

        samples.append({
            "bvtv": bvtv, "tbth_um": tbth,
            "tbn_per_mm": tbn, "tbsp_um": tbsp,
            "sample_index": i,
        })

    return samples


# ---------------------------------------------------------------
# Single VOI loading (backward compat)
# ---------------------------------------------------------------
def load_voi_targets(targets_json: str) -> Dict[str, Any]:
    p = Path(targets_json)
    if not p.exists():
        raise FileNotFoundError(f"VOI targets file not found: {p}")
    with open(p) as f:
        data = json.load(f)
    print(f"Loaded VOI targets from: {p}")
    return data


def extract_single_params(voi: Dict[str, Any],
                          args: argparse.Namespace) -> Dict[str, Any]:
    voxel_um = args.voxel_um or 39.0
    raw_vox = 1.0
    vz = voi.get("voxel_um_zyx")
    if vz and len(vz) >= 1:
        raw_vox = float(vz[0])
    correction = voxel_um / max(1.0, raw_vox)

    bvtv = args.bvtv if args.bvtv is not None else voi.get("BVTV")
    tbth_raw = voi.get("TbTh_um_p90", voi.get("TbTh_um_p50", 0))
    tbth_um = args.tbth_um if args.tbth_um is not None else float(tbth_raw) * correction * 2.0
    tbsp_raw = voi.get("TbSp_um_p50", 0)
    tbsp_um = args.tbsp_um if args.tbsp_um is not None else float(tbsp_raw) * correction * 2.0
    tbn = args.tbn_per_mm if args.tbn_per_mm is not None else float(bvtv) / (float(tbth_um) / 1000.0)

    shp = voi.get("shape_zyx")
    sz = args.z or (int(shp[0]) if shp else 160)
    sxy = args.xy or (int(shp[1]) if shp else 512)

    print(f"  Corrected: BV/TV={bvtv:.3f}, Tb.Th={tbth_um:.1f}um, "
          f"Tb.N={tbn:.2f}/mm, Tb.Sp={tbsp_um:.1f}um")

    return {
        "bvtv": float(bvtv), "tbth_um": float(tbth_um),
        "tbn_per_mm": float(tbn), "tbsp_um": float(tbsp_um),
        "voxel_um": float(voxel_um),
        "shape_z": sz, "shape_xy": sxy,
    }


# ---------------------------------------------------------------
# Params (v14 exact)
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
    skeleton_prune_lmin: int = 8
    reconnect_close_iters: int = 3
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
    solid_fill_sigma: Optional[float] = None
    pve_sigma: float = 0.5
    noise_sd: float = 3.0
    bg_tex_sd: float = 1.0
    unsharp: float = 0.6
    unsharp_sigma: float = 0.8


# ---------------------------------------------------------------
# IO
# ---------------------------------------------------------------
def save_png_u8(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img.astype(np.uint8), mode="L").save(path)

def save_tif_u8(stack, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(path, stack.astype(np.uint8), imagej=True, dtype=np.uint8)

def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------
# v14 generation functions (ALL IDENTICAL TO v14)
# ---------------------------------------------------------------
def tbth_um_to_radius_vox(tbth_um, voxel_um):
    return max(0.5, (tbth_um / voxel_um) / 2.0)

def tbn_per_mm_to_base_sigma(tbn_per_mm, voxel_um):
    period_um = 1000.0 / max(0.1, float(tbn_per_mm))
    return float(max(1.5, period_um / float(voxel_um) / 4.0))

def compute_adaptive_fill_sigma(base_radius_vox):
    return float(np.clip(0.35 * base_radius_vox, 0.3, 1.5))

def normalize(f):
    x = f.astype(np.float32); x -= float(x.mean()); x /= float(x.std() + 1e-6); return x

def smooth_warp(field, rng, warp_sigma, warp_amp):
    if warp_amp <= 0: return field
    dz = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma) * warp_amp
    dy = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma) * warp_amp
    dx = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma) * warp_amp
    Z, Y, X = field.shape
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    coords = np.array([zz + dz, yy + dy, xx + dx])
    return map_coordinates(field, coords, order=1, mode="reflect").astype(np.float32)

def hessian_eigs_3d(f, sigma):
    fxx = ndi.gaussian_filter(f, sigma=sigma, order=(0,0,2))
    fyy = ndi.gaussian_filter(f, sigma=sigma, order=(0,2,0))
    fzz = ndi.gaussian_filter(f, sigma=sigma, order=(2,0,0))
    fxy = ndi.gaussian_filter(f, sigma=sigma, order=(0,1,1))
    fxz = ndi.gaussian_filter(f, sigma=sigma, order=(1,0,1))
    fyz = ndi.gaussian_filter(f, sigma=sigma, order=(1,1,0))
    H = np.stack([np.stack([fzz,fyz,fxz],axis=-1),np.stack([fyz,fyy,fxy],axis=-1),
                  np.stack([fxz,fxy,fxx],axis=-1)], axis=-2)
    w = np.linalg.eigvalsh(H.reshape(-1,3,3)).reshape(f.shape+(3,))
    idx = np.argsort(np.abs(w), axis=-1)
    w = np.take_along_axis(w, idx, axis=-1)
    return w[...,0], w[...,1], w[...,2]

def vesselness_ridge(f, sigma):
    l1, l2, l3 = hessian_eigs_3d(f, sigma=sigma)
    eps = 1e-6
    r1 = np.abs(l1) / (np.abs(l3) + eps)
    r2 = np.abs(l2) / (np.abs(l3) + eps)
    V = np.exp(-(r1*r1)/0.25) * np.exp(-(r2*r2)/0.25)
    V = V.astype(np.float32)
    return V / (float(V.max()) + 1e-6)

def anti_block_round(bone01, sigma):
    if float(sigma) <= 0: return bone01.astype(np.uint8)
    return (ndi.gaussian_filter(bone01.astype(np.float32), sigma=float(sigma)) >= 0.5).astype(np.uint8)

def keep_largest_component(vol):
    st26 = ndi.generate_binary_structure(3, 2)
    lab, n = ndi.label(vol.astype(bool), structure=st26)
    if n == 0: return vol.astype(np.uint8)
    counts = np.bincount(lab.ravel()); counts[0] = 0
    return (lab == int(counts.argmax())).astype(np.uint8)

def remove_small_components(vol, min_size):
    if int(min_size) <= 0: return vol.astype(np.uint8)
    st26 = ndi.generate_binary_structure(3, 2)
    lab, n = ndi.label(vol.astype(bool), structure=st26)
    if n == 0: return vol.astype(np.uint8)
    counts = np.bincount(lab.ravel()); keep = counts >= int(min_size); keep[0] = False
    return keep[lab].astype(np.uint8)

def morph_iters(vol, op, iters):
    if int(iters) <= 0: return vol.astype(np.uint8)
    st26 = ndi.generate_binary_structure(3, 2); x = vol.astype(bool)
    if op == "close": x = ndi.binary_closing(x, structure=st26, iterations=int(iters))
    elif op == "open": x = ndi.binary_opening(x, structure=st26, iterations=int(iters))
    return x.astype(np.uint8)

def hysteresis_on_response(R, q_lo, q_hi):
    q_lo = float(np.clip(q_lo, 0.5, 0.995)); q_hi = float(np.clip(q_hi, q_lo+1e-3, 0.999))
    thr_hi = float(np.quantile(R, q_hi)); thr_lo = float(np.quantile(R, q_lo))
    strong = R >= thr_hi; weak = R >= thr_lo
    st26 = ndi.generate_binary_structure(3, 2)
    lab, n = ndi.label(weak, structure=st26)
    if n == 0: return strong.astype(np.uint8), {"thr_lo": thr_lo, "thr_hi": thr_hi}
    sl = np.unique(lab[strong]); keep = np.zeros(n+1, dtype=bool); keep[sl] = True; keep[0] = False
    return keep[lab].astype(np.uint8), {"thr_lo": thr_lo, "thr_hi": thr_hi}

def skeletonize_with_skimage(proto01):
    if skeletonize_3d is None: raise RuntimeError("skimage.skeletonize_3d unavailable")
    return skeletonize_3d(proto01.astype(bool)).astype(np.uint8)

def skeletonize_with_fiji(proto01, fiji_exe, outdir, command_name="Skeletonize (2D/3D)"):
    outdir.mkdir(parents=True, exist_ok=True)
    in_tif = outdir / "proto_for_fiji.tif"; out_tif = outdir / "skel_from_fiji.tif"
    save_tif_u8((proto01*255).astype(np.uint8), in_tif)
    jython = f'from ij import IJ\nfrom ij.io import FileSaver\nimp = IJ.openImage(r"{in_tif.as_posix()}")\nIJ.run(imp, r"{command_name}", "")\nFileSaver(imp).saveAsTiff(r"{out_tif.as_posix()}")\nimp.close()\n'
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as sf:
        sf.write(jython); script_path = sf.name
    subprocess.run([fiji_exe, "--headless", "--run", script_path], check=True, capture_output=True, text=True)
    return (tiff.imread(out_tif) > 0).astype(np.uint8)

def neighbor_degree_26(skel):
    st = ndi.generate_binary_structure(3, 2)
    n = ndi.convolve(skel.astype(np.uint8), st.astype(np.uint8), mode="constant", cval=0)
    return (n - skel.astype(np.uint8)).astype(np.int16)

def prune_short_end_branches(skel01, lmin):
    lmin = int(max(1, lmin)); st = ndi.generate_binary_structure(3, 2)
    sk = skel01.astype(bool); removed_total = 0
    for _ in range(50):
        deg = neighbor_degree_26(sk.astype(np.uint8))
        endpoints = sk & (deg == 1); junctions = sk & (deg >= 3)
        if not endpoints.any() or not junctions.any(): break
        dist = np.full(sk.shape, np.inf, dtype=np.float32); dist[junctions] = 0.0
        frontier = junctions.copy(); d = 0
        while d < lmin and frontier.any():
            d += 1; nbr = ndi.binary_dilation(frontier, structure=st) & sk & (dist == np.inf)
            dist[nbr] = float(d); frontier = nbr
        to_remove = endpoints & (dist < float(lmin)); n_remove = int(to_remove.sum())
        if n_remove == 0: break
        sk[to_remove] = False; removed_total += n_remove
    return sk.astype(np.uint8), {"prune_lmin": lmin, "vox_removed": removed_total}

def make_proto_and_skeleton(shape, rp, rng, skeleton_mode, fiji_exe, fiji_command, debug_dir):
    f = rng.normal(0, 1, size=shape).astype(np.float32)
    f = ndi.gaussian_filter(f, sigma=float(rp.base_sigma))
    f = smooth_warp(f, rng, float(rp.warp_sigma), float(rp.warp_amp))
    f = normalize(f)
    R = vesselness_ridge(f, sigma=float(rp.hessian_sigma))
    R = np.clip(R * float(rp.ridge_strength), 0.0, 1.0)
    proto01, hyst_info = hysteresis_on_response(R, q_lo=float(rp.proto_q_lo), q_hi=float(rp.proto_q_hi))
    proto01 = morph_iters(proto01, "close", int(rp.proto_close_iters))
    proto01 = morph_iters(proto01, "open", int(rp.proto_open_iters))
    proto01 = remove_small_components(proto01, int(rp.proto_min_component))
    st26 = ndi.generate_binary_structure(3, 2)
    if proto01.astype(bool).sum() > 0:
        proto01 = ndi.binary_closing(proto01.astype(bool), structure=st26, iterations=1).astype(np.uint8)
    skel_raw = proto01.copy().astype(np.uint8); used_skel = False
    if bool(rp.use_skeleton):
        if skeleton_mode == "skimage":
            skel_raw = skeletonize_with_skimage(proto01); used_skel = True
        elif skeleton_mode == "fiji":
            if not fiji_exe: raise RuntimeError("--skeleton-mode fiji requires --fiji-exe")
            wd = debug_dir or Path(tempfile.mkdtemp())
            skel_raw = skeletonize_with_fiji(proto01, fiji_exe=fiji_exe, outdir=wd, command_name=fiji_command); used_skel = True
    skel_pruned, prune_info = prune_short_end_branches(skel_raw, lmin=int(rp.skeleton_prune_lmin))
    if int(rp.reconnect_close_iters) > 0:
        skel_pruned = morph_iters(skel_pruned, "close", int(rp.reconnect_close_iters))
    if debug_dir is not None:
        save_tif_u8((R*255).astype(np.uint8), debug_dir/"ridge_response.tif")
        save_tif_u8((proto01*255).astype(np.uint8), debug_dir/"proto_network.tif")
        save_tif_u8((skel_raw*255).astype(np.uint8), debug_dir/"skeleton_raw.tif")
        save_tif_u8((skel_pruned*255).astype(np.uint8), debug_dir/"skeleton_pruned.tif")
    return skel_pruned.astype(np.uint8), {"hysteresis": hyst_info, "used_skeleton": used_skel,
        "skeleton_mode": skeleton_mode, "prune_info": prune_info}

def radius_samples_for_skeleton(skel01, rng, base_radius_vox, mode, jitter, smooth_sigma):
    sk = skel01.astype(bool); rad = np.zeros(skel01.shape, dtype=np.float32)
    if not sk.any(): return rad
    base = float(max(0.5, base_radius_vox)); jitter = float(np.clip(jitter, 0.0, 0.9))
    st26 = ndi.generate_binary_structure(3, 2)
    if mode == "branch":
        lab, n = ndi.label(sk, structure=st26)
        for i in range(1, n+1):
            rad[lab == i] = base * float(np.exp(rng.normal(0.0, 0.35 * jitter)))
    else:
        noise = rng.normal(0.0, 1.0, size=skel01.shape).astype(np.float32)
        rad[sk] = base * np.clip(1.0 + jitter * noise[sk], 0.25, 3.0)
    if float(smooth_sigma) > 0:
        w = sk.astype(np.float32)
        num = ndi.gaussian_filter(rad, sigma=float(smooth_sigma))
        den = ndi.gaussian_filter(w, sigma=float(smooth_sigma)) + 1e-6
        rad = num / den; rad[~sk] = 0.0
    return rad

def thicken_from_skeleton_radius_field(skel01, rng, target_bvtv, base_radius_vox,
                                       radius_mode, radius_jitter, radius_smooth_sigma,
                                       radius_scale_hint, debug_dir):
    sk = skel01.astype(bool)
    if not sk.any(): return np.zeros_like(skel01, dtype=np.uint8), {"error": "Empty skeleton"}
    rad_skel = radius_samples_for_skeleton(skel01, rng=rng, base_radius_vox=base_radius_vox,
        mode=radius_mode, jitter=radius_jitter, smooth_sigma=radius_smooth_sigma)
    dist, inds = ndi.distance_transform_edt(~sk, return_indices=True)
    iz, iy, ix = inds; rad_field = rad_skel[iz, iy, ix].astype(np.float32)
    min_r = float(max(0.5, 0.3 * base_radius_vox))
    rad_field = np.maximum(rad_field, min_r)
    target = float(np.clip(target_bvtv, 0.01, 0.95))
    lo, hi = 0.25, 3.0
    best_scale = float(np.clip(radius_scale_hint, lo, hi)); best_err = float("inf")
    for _ in range(24):
        mid = 0.5 * (lo + hi); bone = dist <= (mid * rad_field); b = float(bone.mean())
        err = abs(b - target)
        if err < best_err: best_err = err; best_scale = mid
        if b < target: lo = mid
        else: hi = mid
    bone = (dist <= (best_scale * rad_field)).astype(np.uint8)
    return bone, {"base_radius_vox": float(base_radius_vox), "min_radius_floor": float(min_r),
        "radius_mode": radius_mode, "scale_fit": float(best_scale),
        "bvtv_target": float(target), "bvtv_after_thicken": float(bone.mean()),
        "warn_target_miss": bool(abs(float(bone.mean()) - target) > 0.10)}

def microct_gray_solid(bone01, gp, rng, base_radius_vox=2.0):
    bone = bone01.astype(bool)
    d_in = ndi.distance_transform_edt(bone).astype(np.float32)
    sigma = float(gp.solid_fill_sigma) if gp.solid_fill_sigma is not None else compute_adaptive_fill_sigma(base_radius_vox)
    fill = (1.0 - np.exp(-(d_in / max(0.2, sigma)) ** 2)) * bone.astype(np.float32)
    gray = float(gp.marrow_mean) + fill * (float(gp.bone_mean) - float(gp.marrow_mean))
    if float(gp.pve_sigma) > 0: gray = ndi.gaussian_filter(gray, sigma=float(gp.pve_sigma))
    if float(gp.bg_tex_sd) > 0: gray += rng.normal(0, float(gp.bg_tex_sd), size=gray.shape).astype(np.float32)
    if float(gp.noise_sd) > 0: gray += rng.normal(0, float(gp.noise_sd), size=gray.shape).astype(np.float32)
    if float(gp.unsharp) > 0:
        blurred = ndi.gaussian_filter(gray, sigma=max(0.4, float(gp.unsharp_sigma)))
        gray += float(gp.unsharp) * (gray - blurred)
    return np.clip(gray, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------
# Measurement (FIX A: x2) + Validation (FIX D: ASCII)
# ---------------------------------------------------------------
def measure_all_morphometrics(vol01, voxel_um):
    bone = vol01.astype(bool); bvtv = float(bone.mean())
    dt_bone = ndi.distance_transform_edt(bone) * float(voxel_um)
    dt_marrow = ndi.distance_transform_edt(~bone) * float(voxel_um)
    tbth_vals = dt_bone[bone]; tbsp_vals = dt_marrow[~bone]
    def pct(x, p): return float(np.percentile(x, p)) if x.size else 0.0
    tbth_p50 = 2.0 * pct(tbth_vals, 50); tbth_p90 = 2.0 * pct(tbth_vals, 90)
    tbsp_p50 = 2.0 * pct(tbsp_vals, 50); tbsp_p90 = 2.0 * pct(tbsp_vals, 90)
    tbth_mean_mm = (2.0 * float(np.mean(tbth_vals)) / 1000.0) if tbth_vals.size else 1e-6
    tbn = bvtv / tbth_mean_mm if tbth_mean_mm > 0 else 0.0
    euler = float(euler_number(bone, connectivity=3))
    st26 = ndi.generate_binary_structure(3, 2); lab, n = ndi.label(bone, structure=st26)
    lcc_frac = 0.0; n_comp = int(n)
    if n > 0: counts = np.bincount(lab.ravel()); counts[0] = 0; lcc_frac = float(counts.max()) / float(max(1, bone.sum()))
    return {"BVTV": bvtv, "TbTh_um_p50": tbth_p50, "TbTh_um_p90": tbth_p90,
            "TbSp_um_p50": tbsp_p50, "TbSp_um_p90": tbsp_p90, "TbN_per_mm": float(tbn),
            "Euler": euler, "ConnProxy": float(1.0-euler), "n_components": n_comp, "lcc_frac": lcc_frac}

def skeleton_graph_stats(skel01):
    sk = skel01.astype(bool)
    if not sk.any(): return {"skel_voxels": 0, "junctions": 0, "endpoints": 0, "endpoint_junction_ratio": None}
    deg = neighbor_degree_26(sk.astype(np.uint8))
    ep = int((sk & (deg == 1)).sum()); jn = int((sk & (deg >= 3)).sum())
    return {"skel_voxels": int(sk.sum()), "junctions": jn, "endpoints": ep,
            "endpoint_junction_ratio": float(ep)/max(1,jn) if jn > 0 else None}

def validate_morphometrics(measured, targets):
    checks = {}
    for mk, tk, tol, lab in [("BVTV","bvtv_target",0.05,"BV/TV"),("TbTh_um_p50","tbth_um_target",0.15,"Tb.Th (p50, um)"),
                              ("TbN_per_mm","tbn_target",0.20,"Tb.N (/mm)"),("TbSp_um_p50","tbsp_um_target",0.15,"Tb.Sp (p50, um)")]:
        tv = targets.get(tk); mv = measured.get(mk)
        if tv and mv and float(tv) > 0:
            re = abs(float(mv)-float(tv))/float(tv)
            checks[lab] = {"measured": float(mv), "target": float(tv), "rel_error": re, "tolerance": tol, "pass": bool(re <= tol)}
    lcc = measured.get("lcc_frac", 0.0)
    checks["Connectivity (LCC)"] = {"lcc_frac": float(lcc), "n_components": int(measured.get("n_components",-1)),
        "pass": bool(lcc >= 0.80), "note": "LCC >= 0.80 required"}
    return checks

def check_tamimi_bounds(measured):
    warnings = []
    for mk, bk in [("BVTV","BVTV"),("TbTh_um_p50","TbTh_um"),("TbN_per_mm","TbN_per_mm"),("TbSp_um_p50","TbSp_um")]:
        v = measured.get(mk); b = TAMIMI_BOUNDS.get(bk)
        if v is not None and b is not None:
            if float(v) < b["lo"] or float(v) > b["hi"]:
                warnings.append(f"{mk}={float(v):.2f} outside [{b['lo']},{b['hi']}] {b['unit']}")
    return warnings


# ---------------------------------------------------------------
# Generate one sample
# ---------------------------------------------------------------
def generate_one(params, args, outdir, sample_label="", seed_override=None):
    seed = seed_override if seed_override is not None else int(args.base_seed)
    rng = np.random.default_rng(seed)
    bvtv = params["bvtv"]; tbth_um = params["tbth_um"]
    tbn_per_mm = params["tbn_per_mm"]; tbsp_um = params["tbsp_um"]
    voxel_um = params.get("voxel_um", float(args.voxel_um or 39.0))
    shape = (params.get("shape_z", args.z or 160), params.get("shape_xy", args.xy or 512),
             params.get("shape_xy", args.xy or 512))

    base_radius_vox = tbth_um_to_radius_vox(tbth_um, voxel_um)
    base_sigma = float(args.base_sigma) if args.base_sigma is not None else tbn_per_mm_to_base_sigma(tbn_per_mm, voxel_um)

    print(f"\n  [{sample_label}] seed={seed}")
    print(f"    BV/TV={bvtv:.3f}, Tb.Th={tbth_um:.0f}um, Tb.N={tbn_per_mm:.2f}/mm, Tb.Sp={tbsp_um:.0f}um")
    print(f"    sigma={base_sigma:.2f}vox, radius={base_radius_vox:.2f}vox, shape={shape}")

    outdir.mkdir(parents=True, exist_ok=True)
    debug_dir = outdir / "debug" if bool(int(args.debug_skeleton)) else None
    if debug_dir: debug_dir.mkdir(parents=True, exist_ok=True)

    rp = RidgeParams(base_sigma=base_sigma, warp_sigma=float(args.warp_sigma), warp_amp=float(args.warp_amp),
        hessian_sigma=float(args.hessian_sigma), ridge_strength=float(args.ridge_strength),
        proto_q_hi=float(args.proto_q_hi), proto_q_lo=float(args.proto_q_lo),
        proto_close_iters=int(args.proto_close_iters), proto_open_iters=int(args.proto_open_iters),
        proto_min_component=int(args.proto_min_component), use_skeleton=bool(int(args.use_skeleton)),
        skeleton_prune_lmin=int(args.skeleton_prune_lmin), reconnect_close_iters=int(args.reconnect_close_iters),
        radius_mode=str(args.radius_mode), radius_jitter=float(args.radius_jitter),
        radius_smooth_sigma=float(args.radius_smooth_sigma), radius_scale_hint=float(args.radius_scale_hint))
    gp = GrayParams(write_gray=bool(int(args.write_gray)), solid_fill_sigma=args.solid_fill_sigma)

    skel01, skel_info = make_proto_and_skeleton(shape=shape, rp=rp, rng=rng,
        skeleton_mode=str(args.skeleton_mode), fiji_exe=args.fiji_exe,
        fiji_command=str(args.fiji_command), debug_dir=debug_dir)

    bone01, thick_info = thicken_from_skeleton_radius_field(skel01=skel01, rng=rng,
        target_bvtv=bvtv, base_radius_vox=base_radius_vox, radius_mode=str(args.radius_mode),
        radius_jitter=float(args.radius_jitter), radius_smooth_sigma=float(args.radius_smooth_sigma),
        radius_scale_hint=float(args.radius_scale_hint), debug_dir=debug_dir)

    bone01 = anti_block_round(bone01, sigma=float(args.round_sigma))
    if int(args.min_component_size) > 0:
        bone01 = remove_small_components(bone01, min_size=int(args.min_component_size))
    if bool(int(args.enforce_lcc)):
        bone01 = keep_largest_component(bone01)

    void01 = (1 - bone01).astype(np.uint8); Z = shape[0]
    save_tif_u8((bone01*255).astype(np.uint8), outdir/"mask.tif")
    save_tif_u8((void01*255).astype(np.uint8), outdir/"void.tif")
    save_png_u8((bone01[Z//2]*255).astype(np.uint8), outdir/"mid.png")

    if gp.write_gray:
        gray = microct_gray_solid(bone01, gp, rng, base_radius_vox=base_radius_vox)
        save_tif_u8(gray, outdir/"gray.tif")
        save_png_u8(gray[Z//2], outdir/"gray_mid.png")

    morphometrics = measure_all_morphometrics(bone01, voxel_um=voxel_um)
    targets_dict = {"bvtv_target": bvtv, "tbth_um_target": tbth_um,
                    "tbn_target": tbn_per_mm, "tbsp_um_target": tbsp_um}
    validation = validate_morphometrics(morphometrics, targets_dict)
    tw = check_tamimi_bounds(morphometrics)

    met = {"version": "v15 (v14-core, pooled)", "sample_label": sample_label, "seed": seed,
           "morphometrics": morphometrics, "targets": targets_dict, "validation": validation,
           "tamimi_warnings": tw, "skeleton_stats": skeleton_graph_stats(skel01),
           "thick_info": thick_info, "params": {"ridge": asdict(rp), "gray": asdict(gp)},
           "shape_zyx": list(shape), "voxel_um": voxel_um}
    save_json(met, outdir/"metrics.json")

    print(f"    BV/TV: target={bvtv:.3f} measured={morphometrics['BVTV']:.3f}")
    if tw:
        for w in tw: print(f"    ! {w}")
    return met


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="v15 pooled VOI trabecular generator for ML datasets")

    # === POOLED MODE (primary) ===
    p.add_argument("--voi-dirs", nargs="+", type=str, default=None,
                   help="Directories with *_targets.json (e.g. data\\derived\\VOI1 data\\derived\\VOI4)")
    p.add_argument("--num-samples", type=int, default=10,
                   help="Number of synthetic samples to generate")
    p.add_argument("--base-seed", type=int, default=42)

    # === SINGLE MODE (backward compat) ===
    p.add_argument("--targets-json", type=str, default=None)

    # === LITERATURE FALLBACK ===
    p.add_argument("--profile", type=str, default=None, choices=["tamimi-hf", "tamimi-hoa"])

    # === Output ===
    p.add_argument("--outdir", type=str, default="output/ml_dataset")

    # === Overrides ===
    p.add_argument("--bvtv", type=float, default=None)
    p.add_argument("--tbth-um", type=float, default=None)
    p.add_argument("--tbn-per-mm", type=float, default=None)
    p.add_argument("--tbsp-um", type=float, default=None)
    p.add_argument("--voxel-um", type=float, default=39.0)
    p.add_argument("--xy", type=int, default=None)
    p.add_argument("--z", type=int, default=None)

    # === v14 generation (all defaults match v14) ===
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
    p.add_argument("--skeleton-mode", type=str, default="skimage", choices=["skimage", "fiji"])
    p.add_argument("--fiji-exe", type=str, default=None)
    p.add_argument("--fiji-command", type=str, default="Skeletonize (2D/3D)")
    p.add_argument("--skeleton-prune-lmin", type=int, default=8)
    p.add_argument("--reconnect-close-iters", type=int, default=3)
    p.add_argument("--radius-mode", type=str, default="branch", choices=["branch", "voxel"])
    p.add_argument("--radius-jitter", type=float, default=0.15)
    p.add_argument("--radius-smooth-sigma", type=float, default=3.0)
    p.add_argument("--radius-scale-hint", type=float, default=1.0)
    p.add_argument("--enforce-lcc", type=int, default=1)
    p.add_argument("--min-component-size", type=int, default=500)
    p.add_argument("--round-sigma", type=float, default=0.7)
    p.add_argument("--solid-fill-sigma", type=float, default=None,
                   help="Set 3.0 for v14-exact grayscale. None=adaptive.")
    p.add_argument("--write-gray", type=int, default=1)
    p.add_argument("--debug-skeleton", type=int, default=0)

    return p


def main():
    args = build_parser().parse_args()

    if args.voi_dirs is not None:
        # === POOLED MODE: learn from all VOI data, generate N samples ===
        print(f"{'=' * 60}")
        print(f"  POOLED MODE: reading VOI data from {len(args.voi_dirs)} directories")
        print(f"  Generating {args.num_samples} synthetic samples for ML training")
        print(f"{'=' * 60}")

        pooled = load_all_voi_targets(args.voi_dirs, voxel_um_override=float(args.voxel_um))
        save_json(pooled, Path(args.outdir) / "pooled_statistics.json")

        pool_rng = np.random.default_rng(int(args.base_seed))
        samples = sample_targets_from_pool(pooled, pool_rng, int(args.num_samples))

        print(f"\n  Sampled {len(samples)} target sets:")
        for i, s in enumerate(samples):
            print(f"    [{i:03d}] BV/TV={s['bvtv']:.3f} Tb.Th={s['tbth_um']:.0f}um "
                  f"Tb.N={s['tbn_per_mm']:.2f}/mm Tb.Sp={s['tbsp_um']:.0f}um")

        all_metrics = []
        for s in samples:
            idx = s["sample_index"]
            seed = int(args.base_seed) + idx + 1
            label = f"sample_{idx:03d}"
            outdir = Path(args.outdir) / label

            s["voxel_um"] = float(args.voxel_um)
            s["shape_z"] = args.z or 160
            s["shape_xy"] = args.xy or 512

            met = generate_one(s, args, outdir, sample_label=label, seed_override=seed)
            all_metrics.append(met)

        # Save dataset manifest
        manifest = {
            "version": "v15 (v14-core, pooled)",
            "num_samples": len(all_metrics),
            "pooled_from": args.voi_dirs,
            "pooled_statistics": pooled,
            "voxel_um": float(args.voxel_um),
            "samples": [
                {"label": m["sample_label"], "seed": m["seed"],
                 "bvtv_target": m["targets"]["bvtv_target"],
                 "bvtv_measured": m["morphometrics"]["BVTV"],
                 "outdir": str(Path(args.outdir) / m["sample_label"])}
                for m in all_metrics
            ],
        }
        save_json(manifest, Path(args.outdir) / "dataset_manifest.json")
        print(f"\n{'=' * 60}")
        print(f"  Dataset complete: {len(all_metrics)} samples in {args.outdir}/")
        print(f"  Manifest: {args.outdir}/dataset_manifest.json")
        print(f"{'=' * 60}")

    elif args.targets_json is not None:
        # === SINGLE MODE ===
        voi = load_voi_targets(args.targets_json)
        params = extract_single_params(voi, args)
        generate_one(params, args, Path(args.outdir), sample_label="single")
        print(f"\nOutputs: {args.outdir}/")

    elif args.profile is not None:
        # === LITERATURE FALLBACK ===
        ref = TAMIMI_HF if args.profile == "tamimi-hf" else TAMIMI_HOA
        print(f"Using Tamimi {args.profile} profile (FALLBACK)")
        params = {"bvtv": ref["BVTV"], "tbth_um": ref["TbTh_um"],
                  "tbn_per_mm": ref["TbN_per_mm"], "tbsp_um": ref["TbSp_um"],
                  "voxel_um": float(args.voxel_um), "shape_z": args.z or 160, "shape_xy": args.xy or 512}
        generate_one(params, args, Path(args.outdir), sample_label="literature")
        print(f"\nOutputs: {args.outdir}/")

    else:
        print("ERROR: Provide one of:")
        print("  --voi-dirs data\\derived\\VOI1 data\\derived\\VOI4   (pooled ML dataset)")
        print("  --targets-json <path>                              (single specimen)")
        print("  --profile tamimi-hf|tamimi-hoa                     (literature fallback)")
        raise SystemExit(1)


if __name__ == "__main__":
    main()