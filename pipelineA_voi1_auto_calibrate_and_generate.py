#!/usr/bin/env python3
r"""
pipelineA_voi1_auto_calibrate_and_generate.py

ONE-SCRIPT Pipeline A automation for VOI1 (multiframe DICOM files):
  A) Read all .dcm under --in-root (specimen*/Scan*.dcm)
  B) Convert each scan -> uint8 gray TIFF, segment -> mask TIFF, compute priors -> targets.json
  C) Aggregate targets across scans -> VOI1_targets_mean.json
  D) Auto-calibrate synthetic generator params via random search on SMALL volumes
  E) Generate final FULL-res synthetic volume using best params + µCT sharp grayscale

Dependencies:
  python -m pip install numpy scipy tifffile pillow scikit-image pydicom

Example:
  .\.venv\Scripts\python.exe .\pipelineA_voi1_auto_calibrate_and_generate.py `
    --in-root data\real\VOI1 `
    --derived-root data\derived\VOI1 `
    --synth-outdir data\synth\VOI1\v13_auto_calibrated `
    --voxel-um 39 `
    --seg-method otsu `
    --trials 80 `
    --calib-xy 128 --calib-z 96 `
    --final-xy 512 --final-z 160 `
    --seed 23
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import tifffile as tiff
import pydicom
from PIL import Image
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.measure import euler_number


# -----------------------------
# IO helpers
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
# A) DICOM multiframe loading
# -----------------------------
def dicom_to_volume(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    arr = np.asarray(ds.pixel_array)
    if arr.ndim == 3:
        vol = arr  # (frames, rows, cols) -> (Z,Y,X)
    elif arr.ndim == 2:
        vol = arr[None, ...]
    else:
        raise RuntimeError(f"Unexpected pixel_array shape: {arr.shape}")
    return vol.astype(np.float32)

def get_spacing_um(ds: pydicom.dataset.FileDataset, fallback_um: float) -> Tuple[float, float, float]:
    y_um = x_um = fallback_um
    z_um = fallback_um

    if hasattr(ds, "PixelSpacing"):
        try:
            y_um = float(ds.PixelSpacing[0]) * 1000.0
            x_um = float(ds.PixelSpacing[1]) * 1000.0
        except Exception:
            pass

    for tag in ("SpacingBetweenSlices", "SliceThickness"):
        if hasattr(ds, tag):
            try:
                z_um = float(getattr(ds, tag)) * 1000.0
                break
            except Exception:
                pass

    return float(z_um), float(y_um), float(x_um)

def to_uint8_percentile(vol_f32: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(vol_f32, [p_lo, p_hi])
    if hi <= lo + 1e-6:
        return np.zeros_like(vol_f32, dtype=np.uint8)
    x = (vol_f32 - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


# -----------------------------
# B) Segmentation + priors
# -----------------------------
def segment_mask(gray_u8: np.ndarray,
                 method: str = "otsu",
                 blur_sigma: float = 0.8,
                 close_iters: int = 2,
                 open_iters: int = 1,
                 invert: bool = False,
                 sauvola_window: int = 51,
                 sauvola_k: float = 0.15) -> Tuple[np.ndarray, Dict[str, Any]]:
    g = gray_u8.astype(np.float32)
    if blur_sigma > 0:
        g = ndi.gaussian_filter(g, sigma=float(blur_sigma))

    method = method.lower().strip()
    info: Dict[str, Any] = {"method": method, "blur_sigma": float(blur_sigma)}

    if method == "otsu":
        thr = float(threshold_otsu(g))
        m = g >= thr
        info["thr"] = thr
    elif method == "sauvola":
        win = max(15, int(sauvola_window) | 1)
        thr_map = threshold_sauvola(g, window_size=win, k=float(sauvola_k))
        m = g >= thr_map
        info.update({"window": win, "k": float(sauvola_k)})
    else:
        raise ValueError("seg-method must be 'otsu' or 'sauvola'")

    if invert:
        m = ~m
        info["invert"] = True
    else:
        info["invert"] = False

    st = ndi.generate_binary_structure(3, 1)
    if int(close_iters) > 0:
        m = ndi.binary_closing(m, structure=st, iterations=int(close_iters))
    if int(open_iters) > 0:
        m = ndi.binary_opening(m, structure=st, iterations=int(open_iters))

    info.update({"close_iters": int(close_iters), "open_iters": int(open_iters)})
    return m.astype(np.uint8), info

def connected_stats(mask01: np.ndarray) -> Dict[str, float]:
    m = mask01.astype(bool)
    if m.sum() == 0:
        return {"n_components": 0.0, "lcc_frac": 0.0}

    st = ndi.generate_binary_structure(3, 1)
    lab, n = ndi.label(m, structure=st)
    if n == 0:
        return {"n_components": 0.0, "lcc_frac": 0.0}

    counts = np.bincount(lab.ravel())
    counts[0] = 0
    lcc = counts.max()
    return {"n_components": float(n), "lcc_frac": float(lcc / (m.sum() + 1e-6))}

def compute_targets(mask01: np.ndarray, voxel_um_zyx: Tuple[float, float, float]) -> Dict[str, Any]:
    m = mask01.astype(bool)
    bvtv = float(m.mean())
    eul = float(euler_number(m, connectivity=3))
    conn_proxy = float(1.0 - eul)

    z_um, y_um, x_um = voxel_um_zyx
    sampling = (float(z_um), float(y_um), float(x_um))

    dt_b = ndi.distance_transform_edt(m, sampling=sampling)
    dt_m = ndi.distance_transform_edt(~m, sampling=sampling)

    tbth = dt_b[m]
    tbsp = dt_m[~m]

    def pct(x: np.ndarray, p: float) -> float:
        return float(np.percentile(x, p)) if x.size else 0.0

    cc = connected_stats(mask01)

    return {
        "BVTV": bvtv,
        "Euler": eul,
        "ConnProxy": conn_proxy,
        "TbTh_um_p50": pct(tbth, 50),
        "TbTh_um_p90": pct(tbth, 90),
        "TbSp_um_p50": pct(tbsp, 50),
        "TbSp_um_p90": pct(tbsp, 90),
        **cc,
        "voxel_um_zyx": [float(z_um), float(y_um), float(x_um)],
        "shape_zyx": [int(mask01.shape[0]), int(mask01.shape[1]), int(mask01.shape[2])],
    }


# -----------------------------
# C) Aggregate targets
# -----------------------------
AGG_KEYS = [
    "BVTV", "Euler",
    "TbTh_um_p50", "TbTh_um_p90",
    "TbSp_um_p50", "TbSp_um_p90",
    "n_components", "lcc_frac",
]

def aggregate_targets(target_files: List[Path]) -> Dict[str, Any]:
    vals: Dict[str, List[float]] = {k: [] for k in AGG_KEYS}
    voxel_um_zyx: Optional[List[float]] = None

    for f in target_files:
        d = json.load(open(f, "r"))
        for k in AGG_KEYS:
            vals[k].append(float(d.get(k, 0.0)))
        if voxel_um_zyx is None and "voxel_um_zyx" in d:
            voxel_um_zyx = list(d["voxel_um_zyx"])

    mean = {k: float(np.mean(vals[k])) for k in AGG_KEYS}
    std = {f"{k}_std": float(np.std(vals[k])) for k in AGG_KEYS}

    return {
        "n_scans": len(target_files),
        "voxel_um_zyx": voxel_um_zyx,
        **mean,
        **std,
    }


# -----------------------------
# D) Synthetic generator core (v11-like connectivity + curvature)
# -----------------------------
@dataclass
class FieldParams:
    sigma: float = 4.2
    plate_strength: float = 0.80
    branch_strength: float = 1.25
    warp_sigma: float = 12.0
    warp_amp: float = 4.5
    final_sigma: float = 0.8

@dataclass
class MorphParams:
    dilate_iters: int = 1
    close_iters: int = 5
    open_iters: int = 0
    thin_erode_iters: int = 0
    reconnect_close_iters: int = 3

@dataclass
class MicroCTParams:
    pve_sigma: float = 0.8
    bone_mean: float = 235.0
    marrow_mean: float = 20.0
    noise_sd: float = 3.0
    bg_tex_sd: float = 1.0
    unsharp: float = 0.55
    unsharp_sigma: float = 0.9

def normalize_z(field: np.ndarray) -> np.ndarray:
    f = field.astype(np.float32)
    f -= float(f.mean())
    f /= float(f.std() + 1e-6)
    return f

def warp_field(field: np.ndarray, rng: np.random.Generator, warp_sigma: float, warp_amp: float) -> np.ndarray:
    if warp_amp <= 0:
        return field
    dz = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma) * warp_amp
    dy = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma) * warp_amp
    dx = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma) * warp_amp

    Z, Y, X = field.shape
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    coords = np.array([zz + dz, yy + dy, xx + dx])
    return map_coordinates(field, coords, order=1, mode="reflect").astype(np.float32)

def plate_bias(field: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return field
    fx = ndi.uniform_filter(field, size=(1, 1, 9))
    fy = ndi.uniform_filter(field, size=(1, 9, 1))
    fz = ndi.uniform_filter(field, size=(9, 1, 1))
    plates = (fx + fy + fz) / 3.0
    return (1.0 - strength) * field + strength * plates

def branch_link_bias(field: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return field
    Z = field.shape[0]
    if Z < 2:
        g2 = field[0] if Z == 1 else field.mean(axis=0)
        gy, gx = np.gradient(g2)
        grad = np.sqrt(gx * gx + gy * gy)
        grad = grad / (float(grad.max()) + 1e-6)
        ridge2 = ndi.gaussian_filter(grad, sigma=2.0).astype(np.float32)
        ridge = np.repeat(ridge2[None, ...], Z, axis=0)
        return field + float(strength) * ridge

    gz, gy, gx = np.gradient(field)
    grad = np.sqrt(gx * gx + gy * gy + gz * gz)
    grad = grad / (float(grad.max()) + 1e-6)
    ridge = ndi.gaussian_filter(grad, sigma=2.0)
    return field + float(strength) * ridge

def generate_field(shape: Tuple[int, int, int], fp: FieldParams, rng: np.random.Generator) -> np.ndarray:
    f = rng.normal(0, 1, size=shape).astype(np.float32)
    f = ndi.gaussian_filter(f, sigma=float(fp.sigma))
    f = warp_field(f, rng, warp_sigma=float(fp.warp_sigma), warp_amp=float(fp.warp_amp))
    f = plate_bias(f, float(fp.plate_strength))
    f = branch_link_bias(f, float(fp.branch_strength))
    if float(fp.final_sigma) > 0:
        f = ndi.gaussian_filter(f, sigma=float(fp.final_sigma))
    return normalize_z(f)

def threshold_to_bvtv(field: np.ndarray, bvtv: float, invert_phase: bool) -> Tuple[np.ndarray, float]:
    bvtv = float(np.clip(bvtv, 0.001, 0.999))
    if invert_phase:
        thr = float(np.quantile(field, bvtv))
        vol01 = (field <= thr).astype(np.uint8)
    else:
        thr = float(np.quantile(field, 1.0 - bvtv))
        vol01 = (field >= thr).astype(np.uint8)
    return vol01, thr

def apply_morphology(vol01: np.ndarray, mp: MorphParams) -> np.ndarray:
    v = vol01.astype(bool)
    st = ndi.generate_binary_structure(3, 1)

    if int(mp.open_iters) > 0:
        v = ndi.binary_opening(v, structure=st, iterations=int(mp.open_iters))
    if int(mp.dilate_iters) > 0:
        v = ndi.binary_dilation(v, structure=st, iterations=int(mp.dilate_iters))
    if int(mp.close_iters) > 0:
        v = ndi.binary_closing(v, structure=st, iterations=int(mp.close_iters))

    if int(mp.thin_erode_iters) > 0:
        v = ndi.binary_erosion(v, structure=st, iterations=int(mp.thin_erode_iters))

    if int(mp.reconnect_close_iters) > 0:
        v = ndi.binary_closing(v, structure=st, iterations=int(mp.reconnect_close_iters))

    return v.astype(np.uint8)

def microct_gray(vol01: np.ndarray, gp: MicroCTParams, rng: np.random.Generator) -> np.ndarray:
    x = vol01.astype(np.float32)
    if float(gp.pve_sigma) > 0:
        x = ndi.gaussian_filter(x, sigma=float(gp.pve_sigma))
    x = clamp01(x)

    gray = float(gp.marrow_mean) + x * (float(gp.bone_mean) - float(gp.marrow_mean))

    if float(gp.bg_tex_sd) > 0:
        gray += rng.normal(0.0, float(gp.bg_tex_sd), size=gray.shape).astype(np.float32)
    if float(gp.noise_sd) > 0:
        gray += rng.normal(0.0, float(gp.noise_sd), size=gray.shape).astype(np.float32)

    if float(gp.unsharp) > 0:
        blurred = ndi.gaussian_filter(gray, sigma=max(0.4, float(gp.unsharp_sigma)))
        gray = gray + float(gp.unsharp) * (gray - blurred)

    return np.clip(gray, 0.0, 255.0).astype(np.uint8)


# -----------------------------
# E) Calibration (random search)
# -----------------------------
def score_synth(metrics: Dict[str, float], target: Dict[str, float]) -> float:
    """
    Lower is better.
    We heavily penalize fragmentation and reward large connected component.
    """
    def rel(a, b, eps=1e-6):
        return abs(a - b) / (abs(b) + eps)

    s = 0.0
    s += 3.0 * rel(metrics["BVTV"], target["BVTV"])
    s += 2.0 * rel(metrics["TbTh_um_p50"], target["TbTh_um_p50"])
    s += 2.0 * rel(metrics["TbTh_um_p90"], target["TbTh_um_p90"])
    s += 1.5 * rel(metrics["TbSp_um_p50"], target["TbSp_um_p50"])
    s += 1.5 * rel(metrics["TbSp_um_p90"], target["TbSp_um_p90"])

    # Euler: use absolute diff scaled
    s += 1.0 * (abs(metrics["Euler"] - target["Euler"]) / (abs(target["Euler"]) + 10.0))

    # Fragmentation penalties
    # Want FEW components and high LCC fraction
    s += 2.5 * max(0.0, (metrics["n_components"] - target["n_components"]) / (target["n_components"] + 1.0))
    s += 4.0 * max(0.0, (target["lcc_frac"] - metrics["lcc_frac"]))

    return float(s)

def synth_metrics(vol01: np.ndarray, voxel_um_zyx: Tuple[float, float, float]) -> Dict[str, float]:
    t = compute_targets(vol01, voxel_um_zyx)
    # keep only needed keys
    return {k: float(t[k]) for k in AGG_KEYS}

def random_params(rng: np.random.Generator) -> Tuple[FieldParams, MorphParams]:
    # Ranges tuned for trabecular-like structure and connectivity
    fp = FieldParams(
        sigma=float(rng.uniform(3.6, 5.4)),
        plate_strength=float(rng.uniform(0.60, 0.90)),
        branch_strength=float(rng.uniform(0.90, 1.60)),
        warp_sigma=float(rng.uniform(10.0, 16.0)),
        warp_amp=float(rng.uniform(3.2, 6.0)),
        final_sigma=float(rng.uniform(0.6, 1.0)),
    )
    mp = MorphParams(
        dilate_iters=int(rng.integers(0, 2)),
        close_iters=int(rng.integers(3, 7)),
        open_iters=int(rng.integers(0, 2)),
        thin_erode_iters=int(rng.integers(0, 2)),
        reconnect_close_iters=int(rng.integers(2, 5)),
    )
    return fp, mp


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", type=str, default=r"data\real\VOI1")
    ap.add_argument("--derived-root", type=str, default=r"data\derived\VOI1")
    ap.add_argument("--synth-outdir", type=str, default=r"data\synth\VOI1\v13_auto_calibrated")

    ap.add_argument("--voxel-um", type=float, default=39.0, help="Fallback voxel size (um) if DICOM tags missing.")

    # DICOM->gray normalize
    ap.add_argument("--p-lo", type=float, default=1.0)
    ap.add_argument("--p-hi", type=float, default=99.0)

    # segmentation
    ap.add_argument("--seg-method", type=str, default="otsu", choices=["otsu", "sauvola"])
    ap.add_argument("--seg-blur", type=float, default=0.8)
    ap.add_argument("--seg-close", type=int, default=2)
    ap.add_argument("--seg-open", type=int, default=1)
    ap.add_argument("--seg-invert", type=int, default=0)
    ap.add_argument("--sauvola-window", type=int, default=51)
    ap.add_argument("--sauvola-k", type=float, default=0.15)

    # calibration
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--calib-xy", type=int, default=128)
    ap.add_argument("--calib-z", type=int, default=96)
    ap.add_argument("--seed", type=int, default=23)

    # final synth size
    ap.add_argument("--final-xy", type=int, default=512)
    ap.add_argument("--final-z", type=int, default=160)

    # phase: for real masks bone should be 1, for synth we pick upper-tail as bone by default
    ap.add_argument("--invert-phase", type=int, default=0)

    args = ap.parse_args()
    rng = np.random.default_rng(int(args.seed))

    in_root = Path(args.in_root)
    derived_root = Path(args.derived_root)
    derived_root.mkdir(parents=True, exist_ok=True)

    # --- Step A/B: process DICOMs -> gray/mask/targets
    dcm_files = sorted(in_root.rglob("*.dcm"))
    if not dcm_files:
        raise SystemExit(f"No .dcm files found under {in_root}")

    print(f"Found {len(dcm_files)} DICOM files under {in_root}")
    target_files: List[Path] = []

    for dcm_path in dcm_files:
        specimen = dcm_path.parent.name
        stem = dcm_path.stem

        ds = pydicom.dcmread(str(dcm_path), force=True)
        vol = dicom_to_volume(ds)
        voxel_um_zyx = get_spacing_um(ds, fallback_um=float(args.voxel_um))

        gray_u8 = to_uint8_percentile(vol, p_lo=args.p_lo, p_hi=args.p_hi)
        gray_out = derived_root / f"{specimen}_{stem}_gray.tif"
        save_tif_u8(gray_u8, gray_out)

        mask01, seg_info = segment_mask(
            gray_u8,
            method=args.seg_method,
            blur_sigma=args.seg_blur,
            close_iters=args.seg_close,
            open_iters=args.seg_open,
            invert=bool(int(args.seg_invert)),
            sauvola_window=args.sauvola_window,
            sauvola_k=args.sauvola_k,
        )
        mask_out = derived_root / f"{specimen}_{stem}_mask.tif"
        save_tif_u8((mask01 * 255).astype(np.uint8), mask_out)

        targets = compute_targets(mask01, voxel_um_zyx=voxel_um_zyx)
        targets.update({"source_dcm": str(dcm_path), "segmentation": seg_info})

        targets_out = derived_root / f"{specimen}_{stem}_targets.json"
        save_json(targets, targets_out)
        target_files.append(targets_out)

        print(f"{specimen} | {stem} -> BVTV={targets['BVTV']:.3f} Euler={targets['Euler']:.1f} "
              f"LCC={targets['lcc_frac']:.3f} comps={targets['n_components']:.0f}")

    # --- Step C: aggregate targets
    agg = aggregate_targets(target_files)
    agg_out = derived_root / "VOI1_targets_mean.json"
    save_json(agg, agg_out)
    print("\nAggregated targets saved:", agg_out)
    print(json.dumps({k: agg[k] for k in ['n_scans','BVTV','TbTh_um_p90','TbSp_um_p90','Euler','lcc_frac','n_components']}, indent=2))

    # calibration uses mean voxel spacing if available
    if agg.get("voxel_um_zyx") is None:
        voxel_um_zyx = (float(args.voxel_um), float(args.voxel_um), float(args.voxel_um))
    else:
        v = agg["voxel_um_zyx"]
        voxel_um_zyx = (float(v[0]), float(v[1]), float(v[2]))

    # --- Step D: auto-calibrate generator
    target = {k: float(agg[k]) for k in AGG_KEYS}

    calib_shape = (int(args.calib_z), int(args.calib_xy), int(args.calib_xy))
    best = {"score": 1e18, "fp": None, "mp": None, "metrics": None}

    print(f"\nCalibrating on small volume {calib_shape} for {int(args.trials)} trials...")

    for i in range(int(args.trials)):
        fp, mp = random_params(rng)

        field = generate_field(calib_shape, fp, rng)
        vol01, _ = threshold_to_bvtv(field, bvtv=float(target["BVTV"]), invert_phase=bool(int(args.invert_phase)))
        vol01 = apply_morphology(vol01, mp)

        met = synth_metrics(vol01, voxel_um_zyx=voxel_um_zyx)
        sc = score_synth(met, target)

        if sc < best["score"]:
            best.update({"score": sc, "fp": fp, "mp": mp, "metrics": met})
            print(f"  best@{i:03d} score={sc:.4f} BVTV={met['BVTV']:.3f} "
                  f"TbTh_p90={met['TbTh_um_p90']:.1f} Euler={met['Euler']:.1f} "
                  f"LCC={met['lcc_frac']:.3f} comps={met['n_components']:.0f}")

    assert best["fp"] is not None and best["mp"] is not None
    best_params = {
        "best_score": float(best["score"]),
        "field": asdict(best["fp"]),
        "morph": asdict(best["mp"]),
        "calib_shape_zyx": list(calib_shape),
        "target_means": target,
        "best_metrics": best["metrics"],
        "invert_phase": bool(int(args.invert_phase)),
    }

    best_out = derived_root / "VOI1_best_params.json"
    save_json(best_params, best_out)
    print("\nBest params saved:", best_out)

    # --- Step E: generate final full-res synthetic
    synth_outdir = Path(args.synth_outdir)
    synth_outdir.mkdir(parents=True, exist_ok=True)

    final_shape = (int(args.final_z), int(args.final_xy), int(args.final_xy))
    print(f"\nGenerating final synthetic volume {final_shape} -> {synth_outdir}")

    fp_final = FieldParams(**best_params["field"])
    mp_final = MorphParams(**best_params["morph"])
    gp_final = MicroCTParams()  # µCT sharp look defaults

    field = generate_field(final_shape, fp_final, rng)
    vol01, thr = threshold_to_bvtv(field, bvtv=float(target["BVTV"]), invert_phase=bool(int(args.invert_phase)))
    vol01 = apply_morphology(vol01, mp_final)

    gray = microct_gray(vol01, gp_final, rng)

    Z = final_shape[0]
    save_tif_u8((vol01 * 255).astype(np.uint8), synth_outdir / "mask.tif")
    save_png_u8((vol01[Z // 2] * 255).astype(np.uint8), synth_outdir / "mid.png")
    save_tif_u8(gray, synth_outdir / "gray.tif")
    save_png_u8(gray[Z // 2], synth_outdir / "gray_mid.png")

    final_metrics = compute_targets(vol01, voxel_um_zyx=voxel_um_zyx)
    out_metrics = {
        "final_metrics": final_metrics,
        "target_means": target,
        "threshold": float(thr),
        "best_params_file": str(best_out),
        "final_shape_zyx": list(final_shape),
        "params": {"field": asdict(fp_final), "morph": asdict(mp_final), "microct": asdict(gp_final)},
    }
    save_json(out_metrics, synth_outdir / "metrics.json")

    print(
        f"\nDONE\n"
        f"Outputs: {synth_outdir}\n"
        f"Final BVTV={final_metrics['BVTV']:.3f} (target {target['BVTV']:.3f}) | "
        f"TbTh_p90={final_metrics['TbTh_um_p90']:.1f} (target {target['TbTh_um_p90']:.1f}) | "
        f"Euler={final_metrics['Euler']:.1f} (target {target['Euler']:.1f}) | "
        f"LCC={final_metrics['lcc_frac']:.3f} (target {target['lcc_frac']:.3f})"
    )


if __name__ == "__main__":
    main()
