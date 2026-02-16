#!/usr/bin/env python3
"""
synthetic_trabecular_v13_microct_sharp_highcontrast_fullfov.py

v13 redo: focus on µCT-like appearance:
- Bright, sharp trabeculae on dark marrow (high contrast)
- Connected, curvy architecture (v11-style warp + ridge/junction boost)
- FULL FOV (no cylindrical ROI mask)
- Optional Laplace-Hamming-like preprocessing + normalized (permille) thresholding

Paper alignment:
- PCD-CT scans in the paper are filtered using Laplace-Hamming (ε=0.5, cut-off=0.4),
  then normalized and segmented using a threshold of 60 permille. :contentReference[oaicite:2]{index=2}
  This script implements a practical *approximation* of that idea:
  (1) Laplace-based edge enhancement with strength epsilon
  (2) optional smoothing to emulate cut-off
  (3) normalization to [0,1]
  (4) threshold at permille/1000.

Outputs (in --outdir):
- mid.png            (binary from field, mid-slice)
- mask.tif           (binary 3D stack, 0/255)
- gray_mid.png       (if --write-gray 1)
- gray.tif           (if --write-gray 1)
- metrics.json
- lh_enh_mid.png     (if --write-lh 1)  edge-enhanced normalized mid-slice
- lh_seg_mid.png     (if --lh-seg 1)    LH-like segmentation mid-slice
- lh_seg_mask.tif    (if --lh-seg 1)

Dependencies:
- numpy
- scipy
- pillow
- tifffile
- scikit-image (euler_number)

Recommended µCT demo:
python .\synthetic_trabecular_v13_microct_sharp_highcontrast_fullfov.py `
  --outdir data\v13_microct_demo `
  --xy 512 --z 160 `
  --seed 23 `
  --bvtv 0.18 `
  --invert-phase 0 `
  --write-gray 1 `
  --microct-pve-sigma 0.8 `
  --microct-unsharp 0.5 `
  --microct-noise 3 `
  --write-lh 1 `
  --lh-seg 1 `
  --lh-epsilon 0.5 `
  --lh-cutoff 0.4 `
  --lh-permille 600
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image
import tifffile as tiff
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from skimage.measure import euler_number


# -----------------------------
# Params
# -----------------------------
@dataclass
class FieldParams:
    # Base smoothing of random field (higher -> larger features)
    sigma: float = 4.2

    # Plate-like sheet emphasis (0..1)
    plate_strength: float = 0.70

    # Branch/junction emphasis (0..1+)
    branch_strength: float = 1.05

    # Curvature warp settings (v11 warp)
    warp_sigma: float = 12.0
    warp_amp: float = 4.0

    # Final small blur to remove harsh edges
    final_sigma: float = 0.8


@dataclass
class MorphParams:
    # Bridge/connect first
    open_iters: int = 0
    dilate_iters: int = 1
    close_iters: int = 3

    # Thin after connectivity (0..2 typical)
    thin_erode_iters: int = 1

    # Reconnect after thinning
    reconnect_close_iters: int = 2


@dataclass
class MicroCTParams:
    write_gray: bool = True

    # Partial-volume blur (µCT in paper uses Gaussian filter σ=0.8, support=1) :contentReference[oaicite:3]{index=3}
    pve_sigma: float = 0.8

    # Intensities: HIGH contrast (bright trabeculae, dark marrow)
    bone_mean: float = 235.0
    marrow_mean: float = 20.0

    # Noise (keep low for sharp µCT look)
    noise_sd: float = 3.0
    bg_tex_sd: float = 1.0

    # Unsharp for crisp trabecular edges
    unsharp: float = 0.55
    unsharp_sigma: float = 0.9

    # Optional very mild low-freq shading (set to 0 for “clean µCT”)
    shading_sigma: float = 0.0
    shading_amp: float = 0.0


@dataclass
class LHParams:
    write_lh: bool = False
    lh_seg: bool = False

    # Laplace-Hamming parameters (paper uses ε=0.5, cut-off=0.4) :contentReference[oaicite:4]{index=4}
    epsilon: float = 0.5
    cutoff: float = 0.4

    # Normalized threshold in permille (paper uses 60 permille) :contentReference[oaicite:5]{index=5}
    permille: int = 600  # 600 = 0.60

    # Pre-smoothing prior to Laplace (stabilizes noise)
    preblur_sigma: float = 0.9


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
# Utilities
# -----------------------------
def normalize_zscore(field: np.ndarray) -> np.ndarray:
    f = field.astype(np.float32)
    f -= float(f.mean())
    f /= float(f.std() + 1e-6)
    return f

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


# -----------------------------
# Field shaping (curvature + plates + branches)
# -----------------------------
def warp_field(field: np.ndarray, rng: np.random.Generator, warp_sigma: float, warp_amp: float) -> np.ndarray:
    """Smooth coordinate warp to introduce curvature (v11-style)."""
    if warp_amp <= 0:
        return field

    dz = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma) * warp_amp
    dy = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma) * warp_amp
    dx = ndi.gaussian_filter(rng.normal(0, 1, field.shape), sigma=warp_sigma) * warp_amp

    Z, Y, X = field.shape
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    coords = np.array([zz + dz, yy + dy, xx + dx])
    warped = map_coordinates(field, coords, order=1, mode="reflect")
    return warped.astype(np.float32)

def plate_bias(field: np.ndarray, strength: float) -> np.ndarray:
    """Encourage plate-like sheets by local integration along axes."""
    if strength <= 0:
        return field
    fx = ndi.uniform_filter(field, size=(1, 1, 9))
    fy = ndi.uniform_filter(field, size=(1, 9, 1))
    fz = ndi.uniform_filter(field, size=(9, 1, 1))
    plates = (fx + fy + fz) / 3.0
    return (1.0 - strength) * field + strength * plates

def branch_link_bias(field: np.ndarray, strength: float) -> np.ndarray:
    """
    Boost ridge/junction structures.
    Safe for thin Z: if Z < 2, compute 2D gradient per slice.
    """
    if strength <= 0:
        return field

    Z, Y, X = field.shape

    if Z < 2:
        # 2D gradient on the single slice (or thin slab), then broadcast
        g2 = field[0] if Z == 1 else field.mean(axis=0)
        gy, gx = np.gradient(g2)
        grad_mag2 = np.sqrt(gx * gx + gy * gy)
        grad_mag2 = grad_mag2 / (float(grad_mag2.max()) + 1e-6)
        ridge2 = ndi.gaussian_filter(grad_mag2, sigma=2.0).astype(np.float32)
        ridge = np.repeat(ridge2[None, ...], Z, axis=0)
        return field + float(strength) * ridge

    gz, gy, gx = np.gradient(field)
    grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    grad_mag = grad_mag / (float(grad_mag.max()) + 1e-6)
    ridge = ndi.gaussian_filter(grad_mag, sigma=2.0)
    return field + float(strength) * ridge

def generate_field(shape: Tuple[int, int, int], fp: FieldParams, rng: np.random.Generator) -> np.ndarray:
    f = rng.normal(0, 1, size=shape).astype(np.float32)
    f = ndi.gaussian_filter(f, sigma=float(fp.sigma))

    f = warp_field(f, rng, warp_sigma=float(fp.warp_sigma), warp_amp=float(fp.warp_amp))
    f = plate_bias(f, float(fp.plate_strength))
    f = branch_link_bias(f, float(fp.branch_strength))

    if float(fp.final_sigma) > 0:
        f = ndi.gaussian_filter(f, sigma=float(fp.final_sigma))

    return normalize_zscore(f)


# -----------------------------
# Thresholding + morphology
# -----------------------------
def threshold_to_bvtv(field: np.ndarray, bvtv: float, invert_phase: bool) -> Tuple[np.ndarray, float]:
    """
    Foreground fraction ~= bvtv for either phase choice:
      - invert_phase False: foreground = upper tail >= q(1-bvtv)
      - invert_phase True : foreground = lower tail <= q(bvtv)
    """
    bvtv = float(np.clip(bvtv, 0.001, 0.999))
    if invert_phase:
        thr = float(np.quantile(field, bvtv))
        vol01 = (field <= thr).astype(np.uint8)
    else:
        thr = float(np.quantile(field, 1.0 - bvtv))
        vol01 = (field >= thr).astype(np.uint8)
    return vol01, thr

def apply_morphology(vol01: np.ndarray, mp: MorphParams) -> np.ndarray:
    """
    Bridge/connect first -> thin -> reconnect.
    This helps keep connectivity while achieving thinner trabeculae.
    """
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


# -----------------------------
# µCT grayscale forward model (sharp, high-contrast)
# -----------------------------
def lowfreq_noise(shape: Tuple[int, int, int], rng: np.random.Generator, sigma: float) -> np.ndarray:
    n = rng.normal(0, 1, size=shape).astype(np.float32)
    n = ndi.gaussian_filter(n, sigma=float(sigma))
    n = n / (float(n.std()) + 1e-6)
    return n

def microct_gray_sharp(vol01: np.ndarray, gp: MicroCTParams, rng: np.random.Generator) -> np.ndarray:
    """
    Make bright trabeculae and dark marrow with crisp edges.
    """
    x = vol01.astype(np.float32)

    # Partial volume blur (keep low to remain sharp)
    if float(gp.pve_sigma) > 0:
        x = ndi.gaussian_filter(x, sigma=float(gp.pve_sigma))
    x = clamp01(x)

    gray = float(gp.marrow_mean) + x * (float(gp.bone_mean) - float(gp.marrow_mean))

    # Optional very mild shading
    if float(gp.shading_amp) > 0 and float(gp.shading_sigma) > 0:
        gray = gray + lowfreq_noise(gray.shape, rng, sigma=float(gp.shading_sigma)) * float(gp.shading_amp)

    # Fine background texture + low noise
    if float(gp.bg_tex_sd) > 0:
        gray += rng.normal(0.0, float(gp.bg_tex_sd), size=gray.shape).astype(np.float32)
    if float(gp.noise_sd) > 0:
        gray += rng.normal(0.0, float(gp.noise_sd), size=gray.shape).astype(np.float32)

    # Unsharp for edge crispness
    if float(gp.unsharp) > 0:
        blurred = ndi.gaussian_filter(gray, sigma=max(0.4, float(gp.unsharp_sigma)))
        gray = gray + float(gp.unsharp) * (gray - blurred)

    return np.clip(gray, 0.0, 255.0).astype(np.uint8)


# -----------------------------
# Laplace-Hamming-like preprocessing + normalized threshold (permille)
# -----------------------------
def lh_like_enhance(gray_u8: np.ndarray, lp: LHParams) -> np.ndarray:
    """
    Practical approximation of Laplace-Hamming filtering:
    - preblur to stabilize
    - laplace edge response
    - epsilon controls how much edge enhancement is applied
    - cutoff emulated by smoothing the edge term (stronger cutoff -> smoother)
    - normalize to [0,1]
    """
    g = gray_u8.astype(np.float32)

    if float(lp.preblur_sigma) > 0:
        g = ndi.gaussian_filter(g, sigma=float(lp.preblur_sigma))

    lap = ndi.laplace(g)

    # emulate "cut-off" by smoothing laplacian term:
    # higher cutoff -> less smoothing; lower cutoff -> more smoothing.
    cutoff = float(np.clip(lp.cutoff, 0.05, 0.95))
    lap_sigma = (1.0 - cutoff) * 3.0  # heuristic mapping
    if lap_sigma > 0:
        lap = ndi.gaussian_filter(lap, sigma=lap_sigma)

    enh = g - float(lp.epsilon) * lap

    # normalize to [0,1]
    enh = enh - float(enh.min())
    enh = enh / (float(enh.max()) + 1e-6)
    return enh.astype(np.float32)

def lh_like_segment(gray_u8: np.ndarray, lp: LHParams) -> Tuple[np.ndarray, Dict[str, Any]]:
    enh = lh_like_enhance(gray_u8, lp)
    thr = float(np.clip(lp.permille, 1, 999)) / 1000.0
    seg = (enh >= thr).astype(np.uint8)
    info = {
        "lh_like": True,
        "epsilon": float(lp.epsilon),
        "cutoff": float(lp.cutoff),
        "permille": int(lp.permille),
        "thr_norm": thr,
        "preblur_sigma": float(lp.preblur_sigma),
    }
    return seg, info


# -----------------------------
# Metrics
# -----------------------------
def bvtv(vol01: np.ndarray) -> float:
    return float(np.mean(vol01 > 0))

def euler_conn(vol01: np.ndarray) -> Dict[str, float]:
    eul = float(euler_number(vol01.astype(bool), connectivity=3))
    return {"euler": eul, "conn_proxy": float(1.0 - eul)}


# -----------------------------
# CLI + main
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v13 µCT-sharp, high-contrast trabecular generator (full FOV).")

    p.add_argument("--outdir", type=str, default="data/v13_microct")
    p.add_argument("--xy", type=int, default=512)
    p.add_argument("--z", type=int, default=160)
    p.add_argument("--seed", type=int, default=23)

    p.add_argument("--bvtv", type=float, default=0.18)

    # IMPORTANT: default invert-phase=0 tends to make the connected network be WHITE in your case
    p.add_argument("--invert-phase", type=int, default=0)

    # Field knobs
    p.add_argument("--field-sigma", type=float, default=4.2)
    p.add_argument("--plate-strength", type=float, default=0.70)
    p.add_argument("--branch-strength", type=float, default=1.05)
    p.add_argument("--warp-sigma", type=float, default=12.0)
    p.add_argument("--warp-amp", type=float, default=4.0)
    p.add_argument("--final-sigma", type=float, default=0.8)

    # Morph knobs
    p.add_argument("--open-iters", type=int, default=0)
    p.add_argument("--dilate-iters", type=int, default=1)
    p.add_argument("--close-iters", type=int, default=3)
    p.add_argument("--thin-erode-iters", type=int, default=1)
    p.add_argument("--reconnect-close-iters", type=int, default=2)

    # µCT gray
    p.add_argument("--write-gray", type=int, default=1)
    p.add_argument("--microct-pve-sigma", type=float, default=0.8)
    p.add_argument("--microct-bone-mean", type=float, default=235.0)
    p.add_argument("--microct-marrow-mean", type=float, default=20.0)
    p.add_argument("--microct-noise", type=float, default=3.0)
    p.add_argument("--microct-bgtex", type=float, default=1.0)
    p.add_argument("--microct-unsharp", type=float, default=0.55)
    p.add_argument("--microct-unsharp-sigma", type=float, default=0.9)
    p.add_argument("--microct-shading-sigma", type=float, default=0.0)
    p.add_argument("--microct-shading-amp", type=float, default=0.0)

    # LH-like outputs
    p.add_argument("--write-lh", type=int, default=0)
    p.add_argument("--lh-seg", type=int, default=0)
    p.add_argument("--lh-epsilon", type=float, default=0.5)
    p.add_argument("--lh-cutoff", type=float, default=0.4)
    p.add_argument("--lh-permille", type=int, default=600)
    p.add_argument("--lh-preblur", type=float, default=0.9)

    return p

def main():
    args = build_parser().parse_args()
    rng = np.random.default_rng(int(args.seed))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    Z = int(args.z)
    H = W = int(args.xy)
    shape = (Z, H, W)

    fp = FieldParams(
        sigma=float(args.field_sigma),
        plate_strength=float(args.plate_strength),
        branch_strength=float(args.branch_strength),
        warp_sigma=float(args.warp_sigma),
        warp_amp=float(args.warp_amp),
        final_sigma=float(args.final_sigma),
    )

    mp = MorphParams(
        open_iters=int(args.open_iters),
        dilate_iters=int(args.dilate_iters),
        close_iters=int(args.close_iters),
        thin_erode_iters=int(args.thin_erode_iters),
        reconnect_close_iters=int(args.reconnect_close_iters),
    )

    gp = MicroCTParams(
        write_gray=bool(int(args.write_gray)),
        pve_sigma=float(args.microct_pve_sigma),
        bone_mean=float(args.microct_bone_mean),
        marrow_mean=float(args.microct_marrow_mean),
        noise_sd=float(args.microct_noise),
        bg_tex_sd=float(args.microct_bgtex),
        unsharp=float(args.microct_unsharp),
        unsharp_sigma=float(args.microct_unsharp_sigma),
        shading_sigma=float(args.microct_shading_sigma),
        shading_amp=float(args.microct_shading_amp),
    )

    lp = LHParams(
        write_lh=bool(int(args.write_lh)),
        lh_seg=bool(int(args.lh_seg)),
        epsilon=float(args.lh_epsilon),
        cutoff=float(args.lh_cutoff),
        permille=int(args.lh_permille),
        preblur_sigma=float(args.lh_preblur),
    )

    # --- Generate connected curvy field
    field = generate_field(shape, fp, rng)

    # --- Threshold to target BV/TV and select phase
    vol01, thr = threshold_to_bvtv(field, bvtv=float(args.bvtv), invert_phase=bool(int(args.invert_phase)))

    # --- Morphology: connect -> thin -> reconnect
    vol01 = apply_morphology(vol01, mp)

    # Save binary
    save_png_u8((vol01[Z // 2] * 255).astype(np.uint8), outdir / "mid.png")
    save_tif_u8((vol01 * 255).astype(np.uint8), outdir / "mask.tif")

    # --- µCT-like grayscale
    gray_u8: Optional[np.ndarray] = None
    if gp.write_gray:
        gray_u8 = microct_gray_sharp(vol01, gp, rng)
        save_tif_u8(gray_u8, outdir / "gray.tif")
        save_png_u8(gray_u8[Z // 2], outdir / "gray_mid.png")

    # --- LH-like enhanced + segmentation
    lh_info: Dict[str, Any] = {}
    if lp.write_lh or lp.lh_seg:
        if gray_u8 is None:
            gray_u8 = microct_gray_sharp(vol01, gp, rng)

        enh01 = lh_like_enhance(gray_u8, lp)  # float32 [0,1]
        enh_u8 = np.clip(enh01 * 255.0, 0, 255).astype(np.uint8)
        save_png_u8(enh_u8[Z // 2], outdir / "lh_enh_mid.png")

        if lp.lh_seg:
            seg01, lh_info = lh_like_segment(gray_u8, lp)
            save_png_u8((seg01[Z // 2] * 255).astype(np.uint8), outdir / "lh_seg_mid.png")
            save_tif_u8((seg01 * 255).astype(np.uint8), outdir / "lh_seg_mask.tif")

    # --- Metrics
    metrics: Dict[str, Any] = {
        "binary_field": {
            "BVTV": bvtv(vol01),
            **euler_conn(vol01),
        },
        "threshold": float(thr),
        "invert_phase": bool(int(args.invert_phase)),
        "shape": [Z, H, W],
        "params": {
            "field": asdict(fp),
            "morph": asdict(mp),
            "microct": asdict(gp),
            "lh": asdict(lp),
        },
    }
    if lh_info:
        metrics["lh_like"] = lh_info

    save_json(metrics, outdir / "metrics.json")

    print(
        f"Saved to: {outdir}\n"
        f"BV/TV={metrics['binary_field']['BVTV']:.3f} | "
        f"Euler={metrics['binary_field']['euler']:.1f} | "
        f"ConnProxy={metrics['binary_field']['conn_proxy']:.1f} | "
        f"invert_phase={metrics['invert_phase']}"
    )


if __name__ == "__main__":
    main()
