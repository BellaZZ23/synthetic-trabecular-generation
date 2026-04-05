#!/usr/bin/env python3
r"""
synthetic_trabecular_v16_morphometric_control.py

v16: refined trabecular generator.
Fixes applied:
  - Seed diversity bug fixed: seed = base_seed + k * 997 (prime spacing)
  - Plate-dominant defaults updated to match real VOI1 DICOM honeycomb structure
  - rod_weight=0.2, plate_weight=0.8, base_sigma=2.8 produces best visual match

Suggested run (plate-dominant, matches real VOI1):
  python synthetic_trabecular_v16_morphometric_control.py `
      --voi-dirs data\derived\VOI1 data\derived\VOI4 `
      --outdir output\v16_plate `
      --num-samples 5 `
      --xy 128 --z 40 `
      --voxel-um 39 `
      --bvtv 0.35 `
      --tbth-um 180 `
      --base-sigma 2.8 `
      --aniso-ratio 1.1 `
      --warp-amp 3.0 `
      --hessian-sigma 1.4 `
      --proto-q-hi 0.80 `
      --proto-q-lo 0.68 `
      --proto-close-iters 2 `
      --rod-weight 0.2 `
      --plate-weight 0.8 `
      --sheet-q 0.88 `
      --radius-jitter 0.08 `
      --round-sigma 0.4 `
      --marrow-mean 15 `
      --bone-mean 70 `
      --solid-fill-sigma 1.2 `
      --base-seed 42
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import tifffile as tiff
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from skimage.measure import euler_number

try:
    from skimage.morphology import skeletonize_3d
except ImportError:
    try:
        from skimage.morphology import skeletonize as skeletonize_3d
    except ImportError:
        skeletonize_3d = None


VOI_PRESETS = {
    "voi1": {"xy": 300, "z": 432},
    "voi2": {"xy": 152, "z": 432},
    "voi3": {"xy": 152, "z": 432},
    "voi4": {"xy": 152, "z": 432},
    "voi5": {"xy": 152, "z": 432},
}

TAMIMI_BOUNDS = {
    "BVTV":       (0.05, 0.50),
    "TbTh_um":    (80.0, 300.0),
    "TbN_per_mm": (0.5,  5.0),
    "TbSp_um":    (150.0, 1200.0),
}

TAMIMI_HF  = {"BVTV": 0.2037, "TbTh_um": 180.0, "TbN_per_mm": 1.5,  "TbSp_um": 580.0}
TAMIMI_HOA = {"BVTV": 0.2862, "TbTh_um": 130.0, "TbN_per_mm": 2.58, "TbSp_um": 420.0}


# ──────────────────────────────────────────────────────────
#  VOI TARGET LOADING
# ──────────────────────────────────────────────────────────

def load_all_voi_targets(voi_dirs, voxel_um):
    all_t = []
    for d in voi_dirs:
        for f in sorted(Path(d).glob("*_targets.json")):
            with open(f) as fh:
                data = json.load(fh)
            data["_src"] = str(f)
            all_t.append(data)
            print(f"  Loaded: {f.name}")

    if not all_t:
        raise FileNotFoundError("No *_targets.json found")

    print(f"\nPooling {len(all_t)} files (voxel={voxel_um}um)")
    bv, th, sp = [], [], []
    for t in all_t:
        bv.append(float(t.get("BVTV", 0)))
        rv = 1.0
        vz = t.get("voxel_um_zyx")
        if vz and len(vz) >= 1:
            rv = float(vz[0])
        c = voxel_um / max(1.0, rv)
        th.append(float(t.get("TbTh_um_p90", t.get("TbTh_um_p50", 0))) * c * 2.0)
        sp.append(float(t.get("TbSp_um_p50", 0)) * c * 2.0)

    bv = np.array(bv); th = np.array(th); sp = np.array(sp)
    tn = bv / (th / 1000.0 + 1e-9)

    def st(a):
        return {"mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max())}

    p = {"n": len(all_t), "voxel_um": voxel_um,
         "BVTV": st(bv), "TbTh_um": st(th), "TbSp_um": st(sp), "TbN_per_mm": st(tn)}

    print(f"  BV/TV: {p['BVTV']['mean']:.3f}+/-{p['BVTV']['std']:.3f} "
          f"[{p['BVTV']['min']:.3f}-{p['BVTV']['max']:.3f}]")
    print(f"  Tb.Th: {p['TbTh_um']['mean']:.1f}+/-{p['TbTh_um']['std']:.1f}um")
    print(f"  Tb.N:  {p['TbN_per_mm']['mean']:.2f}+/-{p['TbN_per_mm']['std']:.2f}/mm")
    return p


def sample_targets_from_pool(pooled, rng, n):
    samples = []
    for i in range(n):
        def s(k, lo, hi):
            v = pooled[k]
            return float(np.clip(rng.normal(v["mean"], max(v["std"], v["mean"]*0.05)), lo, hi))
        bvtv = s("BVTV", 0.10, 0.45)
        tbth = s("TbTh_um", 100, 280)
        tbsp = s("TbSp_um", 200, 900)
        tbn  = float(np.clip(bvtv / (tbth / 1000.0), 0.5, 4.0))
        samples.append({"bvtv": bvtv, "tbth_um": tbth, "tbn_per_mm": tbn,
                         "tbsp_um": tbsp, "sample_index": i})
    return samples


def load_voi_targets(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    with open(p) as f:
        return json.load(f)


def extract_single_params(voi, args):
    vu = args.voxel_um or 39.0
    rv = 1.0
    vz = voi.get("voxel_um_zyx")
    if vz and len(vz) >= 1:
        rv = float(vz[0])
    c  = vu / max(1.0, rv)
    bv   = args.bvtv    if args.bvtv    is not None else voi.get("BVTV")
    tbth = args.tbth_um if args.tbth_um is not None else float(voi.get("TbTh_um_p90", voi.get("TbTh_um_p50", 0))) * c * 2.0
    tbsp = args.tbsp_um if args.tbsp_um is not None else float(voi.get("TbSp_um_p50", 0)) * c * 2.0
    tbn  = args.tbn_per_mm if args.tbn_per_mm is not None else float(bv) / (float(tbth) / 1000.0)
    sh   = voi.get("shape_zyx")
    sz   = args.z  or (int(sh[0]) if sh else 160)
    sxy  = args.xy or (int(sh[1]) if sh else 300)
    return {"bvtv": float(bv), "tbth_um": float(tbth), "tbn_per_mm": float(tbn),
            "tbsp_um": float(tbsp), "voxel_um": float(vu), "shape_z": sz, "shape_xy": sxy}


# ──────────────────────────────────────────────────────────
#  DATACLASSES
# ──────────────────────────────────────────────────────────

@dataclass
class RidgeParams:
    base_sigma:           float = 2.8
    warp_sigma:           float = 14.0
    warp_amp:             float = 3.0
    hessian_sigma:        float = 1.4
    ridge_strength:       float = 1.0
    proto_q_hi:           float = 0.80
    proto_q_lo:           float = 0.68
    proto_close_iters:    int   = 2
    proto_open_iters:     int   = 0
    proto_min_component:  int   = 250
    use_skeleton:         bool  = True
    skeleton_prune_lmin:  int   = 6
    reconnect_close_iters:int   = 0
    radius_mode:          str   = "branch"
    radius_jitter:        float = 0.08
    radius_smooth_sigma:  float = 3.0
    radius_scale_hint:    float = 1.0
    prune_small_components:int  = 0
    aniso_ratio:          float = 1.1
    rod_weight:           float = 0.2
    plate_weight:         float = 0.8
    coarse_weight:        float = 0.50
    medium_weight:        float = 0.35
    fine_weight:          float = 0.15
    sheet_q:              float = 0.88
    bridge_dilate_iters:  int   = 0
    bridge_close_iters:   int   = 0


@dataclass
class GrayParams:
    write_gray:       bool            = True
    marrow_mean:      float           = 15.0
    bone_mean:        float           = 70.0
    solid_fill_sigma: Optional[float] = 1.2
    pve_sigma:        float           = 0.5
    noise_sd:         float           = 4.0
    bg_tex_sd:        float           = 2.0
    unsharp:          float           = 0.6
    unsharp_sigma:    float           = 0.8


# ──────────────────────────────────────────────────────────
#  I/O HELPERS
# ──────────────────────────────────────────────────────────

def save_png_u8(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img.astype(np.uint8), mode="L").save(path)

def save_tif_u8(s, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(path, s.astype(np.uint8), imagej=True, dtype=np.uint8)

def save_json(o, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(o, fh, indent=2)

def tbth_um_to_radius_vox(t, v):    return max(0.5, (t / v) / 2.0)
def tbn_per_mm_to_base_sigma(t, v): return float(max(1.5, 1000.0 / max(0.1, float(t)) / float(v) / 4.0))
def compute_adaptive_fill_sigma(r): return float(np.clip(0.35 * r, 0.3, 1.5))


# ──────────────────────────────────────────────────────────
#  FIELD GENERATION
# ──────────────────────────────────────────────────────────

def normalize(f):
    x = f.astype(np.float32)
    x -= float(x.mean())
    x /= float(x.std() + 1e-6)
    return x

def smooth_warp(field, rng, ws, wa):
    if wa <= 0: return field
    sh = field.shape
    dz = ndi.gaussian_filter(rng.normal(0, 1, sh), sigma=ws) * wa
    dy = ndi.gaussian_filter(rng.normal(0, 1, sh), sigma=ws) * wa
    dx = ndi.gaussian_filter(rng.normal(0, 1, sh), sigma=ws) * wa
    Z, Y, X = sh
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    return map_coordinates(field, np.array([zz+dz, yy+dy, xx+dx]),
                           order=1, mode="reflect").astype(np.float32)

def hessian_eigs_3d(f, sigma):
    fxx = ndi.gaussian_filter(f, sigma=sigma, order=(0,0,2))
    fyy = ndi.gaussian_filter(f, sigma=sigma, order=(0,2,0))
    fzz = ndi.gaussian_filter(f, sigma=sigma, order=(2,0,0))
    fxy = ndi.gaussian_filter(f, sigma=sigma, order=(0,1,1))
    fxz = ndi.gaussian_filter(f, sigma=sigma, order=(1,0,1))
    fyz = ndi.gaussian_filter(f, sigma=sigma, order=(1,1,0))
    H = np.stack([np.stack([fzz,fyz,fxz],axis=-1),
                  np.stack([fyz,fyy,fxy],axis=-1),
                  np.stack([fxz,fxy,fxx],axis=-1)], axis=-2)
    w = np.linalg.eigvalsh(H.reshape(-1,3,3)).reshape(f.shape+(3,))
    idx = np.argsort(np.abs(w), axis=-1)
    w = np.take_along_axis(w, idx, axis=-1)
    return w[...,0], w[...,1], w[...,2]

def vesselness_ridge(f, sigma):
    l1, l2, l3 = hessian_eigs_3d(f, sigma=sigma)
    eps = 1e-6
    r1  = np.abs(l1) / (np.abs(l3) + eps)
    r2  = np.abs(l2) / (np.abs(l3) + eps)
    V   = np.exp(-(r1*r1)/0.25) * np.exp(-(r2*r2)/0.25)
    return (V / (float(V.max()) + 1e-6)).astype(np.float32)

def make_directional_field(shape, rng, sig_long, sig_short):
    f_z = ndi.gaussian_filter(rng.normal(0,1,size=shape).astype(np.float32),
                               sigma=(sig_long, sig_short, sig_short))
    f_y = ndi.gaussian_filter(rng.normal(0,1,size=shape).astype(np.float32),
                               sigma=(sig_short, sig_long, sig_short))
    f_x = ndi.gaussian_filter(rng.normal(0,1,size=shape).astype(np.float32),
                               sigma=(sig_short, sig_short, sig_long))
    return np.maximum(np.maximum(f_z, f_y), f_x)

def make_multiscale_anisotropic_field(shape, rng, base_sigma, aniso_ratio,
                                       coarse_weight=0.50, medium_weight=0.35, fine_weight=0.15):
    bs = float(base_sigma); ar = float(aniso_ratio)
    def one(scale, weight):
        return weight * make_directional_field(shape, rng,
                                                max(0.8, bs*scale*ar),
                                                max(0.8, bs*scale))
    return normalize(one(1.8, coarse_weight) + one(1.0, medium_weight) + one(0.6, fine_weight))

def plate_likeness_field(f, base_sigma, sheet_q=0.88):
    g1 = ndi.gaussian_filter(f, sigma=max(0.8, base_sigma * 0.8))
    g2 = ndi.gaussian_filter(f, sigma=max(0.8, base_sigma * 1.4))
    p  = normalize(0.65 * g1 + 0.35 * g2)
    th = float(np.quantile(p, float(np.clip(sheet_q, 0.55, 0.98))))
    return (p >= th).astype(np.float32)


# ──────────────────────────────────────────────────────────
#  MORPHOLOGY HELPERS
# ──────────────────────────────────────────────────────────

def anti_block_round(b, s):
    if float(s) <= 0: return b.astype(np.uint8)
    return (ndi.gaussian_filter(b.astype(np.float32), sigma=float(s)) >= 0.5).astype(np.uint8)

def keep_largest_component(v):
    st = ndi.generate_binary_structure(3,2)
    l, n = ndi.label(v.astype(bool), structure=st)
    if n == 0: return v.astype(np.uint8)
    c = np.bincount(l.ravel()); c[0] = 0
    return (l == int(c.argmax())).astype(np.uint8)

def remove_small_components(v, ms):
    if int(ms) <= 0: return v.astype(np.uint8)
    st = ndi.generate_binary_structure(3,2)
    l, n = ndi.label(v.astype(bool), structure=st)
    if n == 0: return v.astype(np.uint8)
    c = np.bincount(l.ravel()); k = c >= int(ms); k[0] = False
    return k[l].astype(np.uint8)

def morph_iters(v, op, it):
    if int(it) <= 0: return v.astype(np.uint8)
    st = ndi.generate_binary_structure(3,2); x = v.astype(bool)
    if op == "close": x = ndi.binary_closing(x, structure=st, iterations=int(it))
    elif op == "open": x = ndi.binary_opening(x, structure=st, iterations=int(it))
    return x.astype(np.uint8)

def hysteresis_on_response(R, ql, qh):
    ql = float(np.clip(ql, 0.5, 0.995)); qh = float(np.clip(qh, ql+1e-3, 0.999))
    th = float(np.quantile(R, qh)); tl = float(np.quantile(R, ql))
    s  = R >= th; w = R >= tl
    st = ndi.generate_binary_structure(3,2); l, n = ndi.label(w, structure=st)
    if n == 0: return s.astype(np.uint8), {"thr_lo": tl, "thr_hi": th}
    sl = np.unique(l[s]); k = np.zeros(n+1, dtype=bool); k[sl] = True; k[0] = False
    return k[l].astype(np.uint8), {"thr_lo": tl, "thr_hi": th}

def skeletonize_with_skimage(p):
    if skeletonize_3d is None: raise RuntimeError("skeletonize_3d unavailable")
    return skeletonize_3d(p.astype(bool)).astype(np.uint8)

def neighbor_degree_26(sk):
    st = ndi.generate_binary_structure(3,2)
    n  = ndi.convolve(sk.astype(np.uint8), st.astype(np.uint8), mode="constant", cval=0)
    return (n - sk.astype(np.uint8)).astype(np.int16)

def prune_short_end_branches(sk01, lmin):
    lmin = int(max(1, lmin)); st = ndi.generate_binary_structure(3,2)
    sk = sk01.astype(bool); rm = 0
    for _ in range(50):
        deg = neighbor_degree_26(sk.astype(np.uint8))
        ep  = sk & (deg == 1); jn = sk & (deg >= 3)
        if not ep.any() or not jn.any(): break
        d = np.full(sk.shape, np.inf, dtype=np.float32); d[jn] = 0.0
        fr = jn.copy(); dd = 0
        while dd < lmin and fr.any():
            dd += 1
            nb = ndi.binary_dilation(fr, structure=st) & sk & (d == np.inf)
            d[nb] = float(dd); fr = nb
        tr = ep & (d < float(lmin)); nr = int(tr.sum())
        if nr == 0: break
        sk[tr] = False; rm += nr
    return sk.astype(np.uint8), {"prune_lmin": lmin, "vox_removed": rm}

def dilate_erode_bridge(vol01, dilate_iters=0, close_iters=0):
    x = vol01.astype(bool); st = ndi.generate_binary_structure(3,2)
    if int(dilate_iters) > 0: x = ndi.binary_dilation(x, structure=st, iterations=int(dilate_iters))
    if int(close_iters)  > 0: x = ndi.binary_closing(x, structure=st, iterations=int(close_iters))
    if int(dilate_iters) > 0: x = ndi.binary_erosion(x, structure=st, iterations=int(dilate_iters))
    return x.astype(np.uint8)

def component_stats_3d(vol01):
    st = ndi.generate_binary_structure(3,2); lab, n = ndi.label(vol01.astype(bool), structure=st)
    if n == 0: return {"n_components": 0, "lcc_frac": 0.0}
    counts = np.bincount(lab.ravel()); counts[0] = 0
    return {"n_components": int(n), "lcc_frac": float(counts.max()) / float(max(1, counts.sum()))}


# ──────────────────────────────────────────────────────────
#  SKELETON & PROTO NETWORK
# ──────────────────────────────────────────────────────────

def make_proto_and_skeleton(shape, rp, rng, skel_mode, fiji_exe, fiji_cmd, dbg):
    f = make_multiscale_anisotropic_field(shape=shape, rng=rng,
                                           base_sigma=float(rp.base_sigma),
                                           aniso_ratio=float(rp.aniso_ratio),
                                           coarse_weight=float(rp.coarse_weight),
                                           medium_weight=float(rp.medium_weight),
                                           fine_weight=float(rp.fine_weight))
    f = smooth_warp(f, rng, float(rp.warp_sigma), float(rp.warp_amp))
    f = normalize(f)

    rod_R = vesselness_ridge(f, sigma=float(rp.hessian_sigma))
    rod_R = np.clip(rod_R * float(rp.ridge_strength), 0.0, 1.0)

    if float(rp.plate_weight) <= 0:
        plate_R = np.zeros_like(rod_R, dtype=np.float32)
        R = rod_R
    else:
        plate_R = plate_likeness_field(f, base_sigma=float(rp.base_sigma), sheet_q=float(rp.sheet_q))
        R = np.clip(float(rp.rod_weight)*rod_R + float(rp.plate_weight)*plate_R, 0.0, 1.0)

    p01, hy = hysteresis_on_response(R, float(rp.proto_q_lo), float(rp.proto_q_hi))
    p01 = morph_iters(p01, "close", int(rp.proto_close_iters))
    p01 = morph_iters(p01, "open",  int(rp.proto_open_iters))
    p01 = dilate_erode_bridge(p01, dilate_iters=int(rp.bridge_dilate_iters),
                               close_iters=int(rp.bridge_close_iters))
    p01 = remove_small_components(p01, int(rp.proto_min_component))

    sr = p01.copy().astype(np.uint8); used_skeleton = False
    if bool(rp.use_skeleton) and skel_mode == "skimage":
        sr = skeletonize_with_skimage(p01); used_skeleton = True

    sp, pi = prune_short_end_branches(sr, lmin=int(rp.skeleton_prune_lmin))
    if int(rp.reconnect_close_iters) > 0:
        sp = morph_iters(sp, "close", int(rp.reconnect_close_iters))
        if used_skeleton: sp = skeletonize_with_skimage(sp)

    if dbg:
        dbg.mkdir(parents=True, exist_ok=True)
        fn = (f - f.min()) / (f.max() - f.min() + 1e-6)
        save_tif_u8((fn*255).astype(np.uint8),      dbg/"field_norm.tif")
        save_tif_u8((rod_R*255).astype(np.uint8),   dbg/"rod_response.tif")
        save_tif_u8((plate_R*255).astype(np.uint8), dbg/"plate_response.tif")
        save_tif_u8((R*255).astype(np.uint8),       dbg/"ridge_plate_blend.tif")
        save_tif_u8((p01*255).astype(np.uint8),     dbg/"proto_network.tif")
        save_tif_u8((sr*255).astype(np.uint8),      dbg/"skeleton_raw.tif")
        save_tif_u8((sp*255).astype(np.uint8),      dbg/"skeleton_pruned.tif")

    return sp.astype(np.uint8), {"hysteresis": hy, "used_skeleton": used_skeleton,
                                  "skel_mode": skel_mode, "prune": pi,
                                  "rod_weight": float(rp.rod_weight),
                                  "plate_weight": float(rp.plate_weight)}


# ──────────────────────────────────────────────────────────
#  THICKENING
# ──────────────────────────────────────────────────────────

def radius_samples_for_skeleton(sk01, rng, br, mode, jit, ss):
    sk  = sk01.astype(bool); rad = np.zeros(sk01.shape, dtype=np.float32)
    if not sk.any(): return rad
    b = float(max(0.5, br)); j = float(np.clip(jit, 0.0, 0.9))
    st = ndi.generate_binary_structure(3,2)
    if mode == "branch":
        l, n = ndi.label(sk, structure=st)
        for i in range(1, n+1):
            rad[l==i] = b * float(np.exp(rng.normal(0.0, 0.35*j)))
    else:
        ns = rng.normal(0.0, 1.0, size=sk01.shape).astype(np.float32)
        rad[sk] = b * np.clip(1.0 + j*ns[sk], 0.25, 3.0)
    if float(ss) > 0:
        w = sk.astype(np.float32)
        num = ndi.gaussian_filter(rad, sigma=float(ss))
        den = ndi.gaussian_filter(w,   sigma=float(ss)) + 1e-6
        rad = num/den; rad[~sk] = 0.0
    return rad

def thicken_from_skeleton_radius_field(sk01, rng, tbvtv, br, rm, rj, rss, rsh, dbg):
    sk = sk01.astype(bool)
    if not sk.any(): return np.zeros_like(sk01, dtype=np.uint8), {"error": "Empty"}
    rs  = radius_samples_for_skeleton(sk01, rng=rng, br=br, mode=rm, jit=rj, ss=rss)
    dist, inds = ndi.distance_transform_edt(~sk, return_indices=True)
    iz, iy, ix = inds; rf = rs[iz, iy, ix].astype(np.float32)
    mr = float(max(0.5, 0.3*br)); rf = np.maximum(rf, mr)
    tgt = float(np.clip(tbvtv, 0.01, 0.95))
    lo, hi = 0.25, 3.0; bs = float(np.clip(rsh, lo, hi)); be = float("inf")
    for _ in range(24):
        mid  = 0.5*(lo+hi); bone = dist <= (mid*rf); b = float(bone.mean()); e = abs(b-tgt)
        if e < be: be = e; bs = mid
        if b < tgt: lo = mid
        else:       hi = mid
    bone = (dist <= (bs*rf)).astype(np.uint8)
    if dbg is not None:
        rfv = rf / (rf.max() + 1e-6)
        save_tif_u8((rfv*255).astype(np.uint8), dbg/"radius_field_u8.tif")
    return bone, {"base_r": float(br), "min_r": float(mr), "scale": float(bs),
                  "bvtv_tgt": float(tgt), "bvtv_got": float(bone.mean()),
                  "warn_target_miss": bool(abs(float(bone.mean())-tgt) > 0.10)}


# ──────────────────────────────────────────────────────────
#  GRAYSCALE
# ──────────────────────────────────────────────────────────

def microct_gray_solid(bone01, gp, rng, br=2.0):
    bone = bone01.astype(bool)
    d    = ndi.distance_transform_edt(bone).astype(np.float32)
    sig  = float(gp.solid_fill_sigma) if gp.solid_fill_sigma is not None else compute_adaptive_fill_sigma(br)
    fill = (1.0 - np.exp(-(d / max(0.2, sig))**2)) * bone.astype(np.float32)
    g    = float(gp.marrow_mean) + fill * (float(gp.bone_mean) - float(gp.marrow_mean))
    if float(gp.pve_sigma)  > 0: g = ndi.gaussian_filter(g, sigma=float(gp.pve_sigma))
    if float(gp.bg_tex_sd)  > 0: g += rng.normal(0, float(gp.bg_tex_sd),  size=g.shape).astype(np.float32)
    if float(gp.noise_sd)   > 0: g += rng.normal(0, float(gp.noise_sd),   size=g.shape).astype(np.float32)
    if float(gp.unsharp)    > 0:
        bl = ndi.gaussian_filter(g, sigma=max(0.4, float(gp.unsharp_sigma)))
        g += float(gp.unsharp) * (g - bl)
    return np.clip(g, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────
#  MORPHOMETRICS
# ──────────────────────────────────────────────────────────

def measure_all_morphometrics(v, vu):
    bone = v.astype(bool); bv = float(bone.mean())
    db = ndi.distance_transform_edt(bone)  * float(vu)
    dm = ndi.distance_transform_edt(~bone) * float(vu)
    tv = db[bone]; sv = dm[~bone]
    def p(x, q): return float(np.percentile(x, q)) if x.size else 0.0
    t50 = 2*p(tv,50); t90 = 2*p(tv,90)
    s50 = 2*p(sv,50); s90 = 2*p(sv,90)
    tm  = (2*float(np.mean(tv))/1000.0) if tv.size else 1e-6
    tn  = bv/tm if tm > 0 else 0.0
    eu  = float(euler_number(bone, connectivity=3))
    st  = ndi.generate_binary_structure(3,2); l, n = ndi.label(bone, structure=st)
    lf  = 0.0; nc = int(n)
    if n > 0:
        c = np.bincount(l.ravel()); c[0] = 0
        lf = float(c.max()) / float(max(1, bone.sum()))
    return {"BVTV": bv, "TbTh_um_p50": t50, "TbTh_um_p90": t90,
            "TbSp_um_p50": s50, "TbSp_um_p90": s90, "TbN_per_mm": float(tn),
            "Euler": eu, "ConnProxy": float(1-eu), "n_components": nc, "lcc_frac": lf}

def skeleton_graph_stats(sk01):
    sk = sk01.astype(bool)
    if not sk.any(): return {"skel_voxels": 0, "junctions": 0, "endpoints": 0, "ej_ratio": None}
    deg = neighbor_degree_26(sk.astype(np.uint8))
    ep  = int((sk & (deg==1)).sum()); jn = int((sk & (deg>=3)).sum())
    return {"skel_voxels": int(sk.sum()), "junctions": jn, "endpoints": ep,
            "ej_ratio": float(ep)/max(1,jn) if jn > 0 else None}

def connectivity_score(sk01, bone01=None):
    sg   = skeleton_graph_stats(sk01)
    jn   = float(sg["junctions"]); ep = float(sg["endpoints"])
    score = 2000.0*np.tanh(jn/4000.0) - 200.0*np.tanh(ep/2000.0)
    ej   = sg["ej_ratio"] if sg["ej_ratio"] is not None else 999.0
    score -= 100.0 * min(float(ej), 10.0)
    if bone01 is not None:
        cs    = component_stats_3d(bone01); lcc = float(cs["lcc_frac"]); ncomp = int(cs["n_components"])
        score += 300.0*lcc - 20.0*max(0, ncomp-1)
        morph  = measure_all_morphometrics(bone01, 1.0); eu = float(morph["Euler"])
        if eu < -5000: score -= 300.0*np.tanh(abs(eu)/10000.0)
    return {"score": float(score), "junctions": int(sg["junctions"]),
            "endpoints": int(sg["endpoints"]),
            "ej_ratio": None if sg["ej_ratio"] is None else float(sg["ej_ratio"])}

def validate_morphometrics(m, t):
    ch = {}
    for mk, tk, tol, lb in [("BVTV","bvtv_target",0.05,"BV/TV"),
                              ("TbTh_um_p50","tbth_um_target",0.15,"Tb.Th"),
                              ("TbN_per_mm","tbn_target",0.20,"Tb.N"),
                              ("TbSp_um_p50","tbsp_um_target",0.15,"Tb.Sp")]:
        tv = t.get(tk); mv = m.get(mk)
        if tv and mv and float(tv) > 0:
            re = abs(float(mv)-float(tv))/float(tv)
            ch[lb] = {"measured": float(mv), "target": float(tv),
                      "rel_error": re, "tolerance": float(tol), "pass": bool(re<=tol)}
    lcc = m.get("lcc_frac", 0.0)
    ch["Connectivity (LCC)"] = {"lcc_frac": float(lcc),
                                  "n_components": int(m.get("n_components",-1)),
                                  "pass": bool(lcc >= 0.80)}
    return ch

def check_tamimi_bounds(m):
    w = []
    for mk, bk in [("BVTV","BVTV"),("TbTh_um_p50","TbTh_um"),
                    ("TbN_per_mm","TbN_per_mm"),("TbSp_um_p50","TbSp_um")]:
        v = m.get(mk); b = TAMIMI_BOUNDS.get(bk)
        if v is not None and b is not None:
            if float(v) < b[0] or float(v) > b[1]:
                w.append(f"{mk}={float(v):.2f} outside [{b[0]},{b[1]}]")
    return w


# ──────────────────────────────────────────────────────────
#  MAIN GENERATION FUNCTION
# ──────────────────────────────────────────────────────────

def generate_one(params, args, outdir, label="", seed_override=None):
    base_seed = seed_override if seed_override is not None else int(args.base_seed)
    bvtv = params["bvtv"]; tbth = params["tbth_um"]
    tbn  = params["tbn_per_mm"]; tbsp = params["tbsp_um"]
    vu   = params.get("voxel_um", float(args.voxel_um or 39.0))
    shape = (params.get("shape_z", args.z or 160),
             params.get("shape_xy", args.xy or 300),
             params.get("shape_xy", args.xy or 300))
    br = tbth_um_to_radius_vox(tbth, vu)
    bs = float(args.base_sigma) if args.base_sigma is not None else tbn_per_mm_to_base_sigma(tbn, vu)

    print(f"\n  [{label}] base_seed={base_seed}")
    print(f"    BV/TV={bvtv:.3f} Tb.Th={tbth:.0f}um Tb.N={tbn:.2f}/mm "
          f"sigma={bs:.2f} radius={br:.2f} shape={shape}")

    outdir.mkdir(parents=True, exist_ok=True)
    dbg = outdir/"debug" if bool(int(args.debug_skeleton)) else None
    if dbg: dbg.mkdir(parents=True, exist_ok=True)

    rp = RidgeParams(
        base_sigma=bs, warp_sigma=float(args.warp_sigma), warp_amp=float(args.warp_amp),
        hessian_sigma=float(args.hessian_sigma), ridge_strength=float(args.ridge_strength),
        proto_q_hi=float(args.proto_q_hi), proto_q_lo=float(args.proto_q_lo),
        proto_close_iters=int(args.proto_close_iters), proto_open_iters=int(args.proto_open_iters),
        proto_min_component=int(args.proto_min_component),
        use_skeleton=bool(int(args.use_skeleton)), skeleton_prune_lmin=int(args.skeleton_prune_lmin),
        reconnect_close_iters=int(args.reconnect_close_iters),
        radius_mode=str(args.radius_mode), radius_jitter=float(args.radius_jitter),
        radius_smooth_sigma=float(args.radius_smooth_sigma),
        radius_scale_hint=float(args.radius_scale_hint),
        aniso_ratio=float(args.aniso_ratio),
        rod_weight=float(args.rod_weight), plate_weight=float(args.plate_weight),
        coarse_weight=float(args.coarse_weight), medium_weight=float(args.medium_weight),
        fine_weight=float(args.fine_weight), sheet_q=float(args.sheet_q),
        bridge_dilate_iters=int(args.bridge_dilate_iters),
        bridge_close_iters=int(args.bridge_close_iters),
    )
    gp = GrayParams(
        write_gray=bool(int(args.write_gray)), solid_fill_sigma=args.solid_fill_sigma,
        marrow_mean=float(args.marrow_mean), bone_mean=float(args.bone_mean),
        noise_sd=float(args.noise_sd), bg_tex_sd=float(args.bg_tex_sd),
    )

    n_tries = int(getattr(args, "retry_attempts", 4))
    best = None; best_score = -np.inf

    for k in range(n_tries):
        # FIX: Use prime multiplier so samples explore different seed regions.
        # Before: seed = base_seed + k  →  all samples converge on same best seed
        # After:  seed = base_seed + k*997  →  each sample gets unique seed space
        seed    = int(base_seed) + k * 997
        rng     = np.random.default_rng(seed)
        try_dbg = (dbg / f"try_{k:02d}") if dbg else None

        sk01, si = make_proto_and_skeleton(shape=shape, rp=rp, rng=rng,
                                            skel_mode=str(args.skeleton_mode),
                                            fiji_exe=args.fiji_exe,
                                            fiji_cmd=str(args.fiji_command),
                                            dbg=try_dbg)
        bone01, ti = thicken_from_skeleton_radius_field(sk01, rng, bvtv, br,
                                                         str(args.radius_mode),
                                                         float(args.radius_jitter),
                                                         float(args.radius_smooth_sigma),
                                                         float(args.radius_scale_hint),
                                                         dbg=try_dbg)
        bone01 = anti_block_round(bone01, float(args.round_sigma))
        if int(args.min_component_size) > 0:
            bone01 = remove_small_components(bone01, int(args.min_component_size))

        conn  = connectivity_score(sk01, bone01=bone01)
        score = float(conn["score"])
        if score > best_score:
            best_score = score
            best = {"seed": seed, "sk01": sk01.copy(), "bone01": bone01.copy(),
                    "si": si, "ti": ti, "conn": conn}

    if best is None:
        raise RuntimeError("Failed to generate any candidate sample")

    seed   = best["seed"]; sk01 = best["sk01"]; bone01 = best["bone01"]
    si     = best["si"];   ti   = best["ti"];   conn   = best["conn"]

    if bool(int(args.enforce_lcc)):
        bone01 = keep_largest_component(bone01)

    void01 = (1-bone01).astype(np.uint8); Z = shape[0]
    save_tif_u8((bone01*255).astype(np.uint8), outdir/"mask.tif")
    save_tif_u8((void01*255).astype(np.uint8), outdir/"void.tif")
    save_png_u8((bone01[Z//2]*255).astype(np.uint8), outdir/"mid.png")

    if gp.write_gray:
        rng_gray = np.random.default_rng(seed + 10000)
        gray = microct_gray_solid(bone01, gp, rng_gray, br=br)
        save_tif_u8(gray, outdir/"gray.tif")
        save_png_u8(gray[Z//2], outdir/"gray_mid.png")

    morph = measure_all_morphometrics(bone01, vu)
    tgt   = {"bvtv_target": bvtv, "tbth_um_target": tbth,
              "tbn_target": tbn,  "tbsp_um_target": tbsp}
    val   = validate_morphometrics(morph, tgt)
    tw    = check_tamimi_bounds(morph)

    met = {
        "version": "v16", "label": label, "seed": seed,
        "best_connectivity_score": best_score, "connectivity": conn,
        "morphometrics": morph, "targets": tgt, "validation": val,
        "tamimi_warnings": tw, "skeleton_stats": skeleton_graph_stats(sk01),
        "component_stats": component_stats_3d(bone01), "thick_info": ti, "proto_info": si,
        "params": {"ridge": asdict(rp), "gray": asdict(gp),
                   "retry_attempts": n_tries, "round_sigma": float(args.round_sigma)},
        "shape_zyx": list(shape), "voxel_um": vu,
    }
    save_json(met, outdir/"metrics.json")

    print(f"    chosen seed={seed} after {n_tries} tries (prime-spaced)")
    print(f"    connectivity score={best_score:.2f}  "
          f"junctions={conn['junctions']} endpoints={conn['endpoints']}")
    print(f"    BV/TV: target={bvtv:.3f} measured={morph['BVTV']:.3f}")
    return met


# ──────────────────────────────────────────────────────────
#  PARSER & MAIN
# ──────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="v16 trabecular generator (seed-fixed, plate-dominant)")
    p.add_argument("--voi-dirs",   nargs="+", type=str, default=None)
    p.add_argument("--targets-json", type=str, default=None)
    p.add_argument("--profile",    type=str, default=None, choices=["tamimi-hf","tamimi-hoa"])
    p.add_argument("--outdir",     type=str, default="output/ml_dataset")
    p.add_argument("--num-samples",type=int, default=10)
    p.add_argument("--base-seed",  type=int, default=42)
    p.add_argument("--voi-preset", type=str, default=None, choices=list(VOI_PRESETS.keys()))
    p.add_argument("--bvtv",       type=float, default=None)
    p.add_argument("--tbth-um",    dest="tbth_um",    type=float, default=None)
    p.add_argument("--tbn-per-mm", dest="tbn_per_mm", type=float, default=None)
    p.add_argument("--tbsp-um",    dest="tbsp_um",    type=float, default=None)
    p.add_argument("--voxel-um",   type=float, default=39.0)
    p.add_argument("--xy",         type=int,   default=None)
    p.add_argument("--z",          type=int,   default=None)
    p.add_argument("--marrow-mean",type=float, default=15.0)
    p.add_argument("--bone-mean",  type=float, default=70.0)
    p.add_argument("--noise-sd",   type=float, default=4.0)
    p.add_argument("--bg-tex-sd",  type=float, default=2.0)
    p.add_argument("--base-sigma", type=float, default=None)
    p.add_argument("--warp-sigma", type=float, default=14.0)
    p.add_argument("--warp-amp",   type=float, default=3.0)
    p.add_argument("--hessian-sigma",     type=float, default=1.4)
    p.add_argument("--ridge-strength",    type=float, default=1.0)
    p.add_argument("--proto-q-hi",        type=float, default=0.80)
    p.add_argument("--proto-q-lo",        type=float, default=0.68)
    p.add_argument("--proto-close-iters", type=int,   default=2)
    p.add_argument("--proto-open-iters",  type=int,   default=0)
    p.add_argument("--proto-min-component", type=int, default=250)
    p.add_argument("--use-skeleton",      type=int,   default=1)
    p.add_argument("--skeleton-mode",     type=str,   default="skimage", choices=["skimage","fiji"])
    p.add_argument("--fiji-exe",          type=str,   default=None)
    p.add_argument("--fiji-command",      type=str,   default="Skeletonize (2D/3D)")
    p.add_argument("--skeleton-prune-lmin",   type=int,   default=6)
    p.add_argument("--reconnect-close-iters", type=int,   default=0)
    p.add_argument("--radius-mode",       type=str,   default="branch", choices=["branch","voxel"])
    p.add_argument("--radius-jitter",     type=float, default=0.08)
    p.add_argument("--radius-smooth-sigma", type=float, default=3.0)
    p.add_argument("--radius-scale-hint",   type=float, default=1.0)
    p.add_argument("--aniso-ratio",    type=float, default=1.1)
    p.add_argument("--rod-weight",     type=float, default=0.2)
    p.add_argument("--plate-weight",   type=float, default=0.8)
    p.add_argument("--coarse-weight",  type=float, default=0.50)
    p.add_argument("--medium-weight",  type=float, default=0.35)
    p.add_argument("--fine-weight",    type=float, default=0.15)
    p.add_argument("--sheet-q",        type=float, default=0.88)
    p.add_argument("--bridge-dilate-iters", type=int, default=0)
    p.add_argument("--bridge-close-iters",  type=int, default=0)
    p.add_argument("--enforce-lcc",         type=int, default=1)
    p.add_argument("--min-component-size",  type=int, default=500)
    p.add_argument("--round-sigma",         type=float, default=0.4)
    p.add_argument("--solid-fill-sigma",    type=float, default=1.2)
    p.add_argument("--write-gray",          type=int,  default=1)
    p.add_argument("--debug-skeleton",      type=int,  default=0)
    p.add_argument("--retry-attempts",      type=int,  default=4)
    return p


def print_validation_summary(validation):
    print(f"\n{'='*58}\n  MORPHOMETRIC VALIDATION SUMMARY\n{'='*58}")
    print(f"  {'Metric':<22} {'Target':>9} {'Measured':>10} {'Error':>7}")
    print(f"  {'-'*56}")
    for label, chk in validation.items():
        if label == "Connectivity (LCC)":
            status = "PASS" if chk["pass"] else "FAIL <<"
            print(f"  {'Connectivity (LCC)':<22} {'>=0.80':>9} {chk['lcc_frac']:>10.3f} {'—':>7}  {status}")
        else:
            status = "PASS" if chk["pass"] else "FAIL <<"
            print(f"  {label:<22} {chk['target']:>9.2f} {chk['measured']:>10.2f} "
                  f"{chk['rel_error']:>6.1%}  {status}")
    print(f"{'='*58}")


def main():
    args = build_parser().parse_args()

    if args.voi_preset:
        pr = VOI_PRESETS[args.voi_preset]
        if args.xy is None: args.xy = pr["xy"]
        if args.z  is None: args.z  = pr["z"]

    if args.voi_dirs is not None:
        print(f"{'='*60}\n  POOLED: {len(args.voi_dirs)} dirs, {args.num_samples} samples\n"
              f"  Gray: marrow={args.marrow_mean}, bone={args.bone_mean}\n{'='*60}")
        pooled = load_all_voi_targets(args.voi_dirs, voxel_um=float(args.voxel_um))
        save_json(pooled, Path(args.outdir)/"pooled_statistics.json")

        prng    = np.random.default_rng(int(args.base_seed))
        samples = sample_targets_from_pool(pooled, prng, int(args.num_samples))

        for s in samples:
            if args.bvtv       is not None: s["bvtv"]       = float(args.bvtv)
            if args.tbth_um    is not None: s["tbth_um"]    = float(args.tbth_um)
            if args.tbsp_um    is not None: s["tbsp_um"]    = float(args.tbsp_um)
            if args.tbn_per_mm is not None:
                s["tbn_per_mm"] = float(args.tbn_per_mm)
            elif args.bvtv is not None or args.tbth_um is not None:
                s["tbn_per_mm"] = float(np.clip(
                    float(s["bvtv"]) / (float(s["tbth_um"]) / 1000.0), 0.5, 4.0))

        print(f"\n  Sampled {len(samples)} targets:")
        for s in samples:
            print(f"    [{s['sample_index']:03d}] BV/TV={s['bvtv']:.3f} "
                  f"Tb.Th={s['tbth_um']:.0f}um Tb.N={s['tbn_per_mm']:.2f}/mm")

        all_metrics = []
        for s in samples:
            s["voxel_um"] = float(args.voxel_um)
            s["shape_z"]  = args.z  or 160
            s["shape_xy"] = args.xy or 300
            m = generate_one(s, args,
                              Path(args.outdir) / f"sample_{s['sample_index']:03d}",
                              label=f"sample_{s['sample_index']:03d}",
                              seed_override=int(args.base_seed) + s["sample_index"] + 1)
            all_metrics.append(m)
            print_validation_summary(m["validation"])

        save_json({
            "version": "v16", "n": len(all_metrics), "pooled": pooled,
            "gray": {"marrow": args.marrow_mean, "bone": args.bone_mean},
            "samples": [{"label": m["label"], "seed": m["seed"],
                         "bvtv": m["targets"]["bvtv_target"],
                         "bvtv_m": m["morphometrics"]["BVTV"],
                         "connectivity_score": m["best_connectivity_score"]}
                        for m in all_metrics],
        }, Path(args.outdir)/"dataset_manifest.json")
        print(f"\n{'='*60}\n  Dataset: {len(all_metrics)} samples in {args.outdir}/\n{'='*60}")

    elif args.targets_json is not None:
        voi    = load_voi_targets(args.targets_json)
        params = extract_single_params(voi, args)
        m      = generate_one(params, args, Path(args.outdir), label="single")
        print_validation_summary(m["validation"])

    elif args.profile is not None:
        ref    = TAMIMI_HF if args.profile == "tamimi-hf" else TAMIMI_HOA
        params = {"bvtv": ref["BVTV"], "tbth_um": ref["TbTh_um"],
                  "tbn_per_mm": ref["TbN_per_mm"], "tbsp_um": ref["TbSp_um"],
                  "voxel_um": float(args.voxel_um),
                  "shape_z": args.z or 160, "shape_xy": args.xy or 300}
        m = generate_one(params, args, Path(args.outdir), label="literature")
        print_validation_summary(m["validation"])
    else:
        print("Provide: --voi-dirs, --targets-json, or --profile")
        raise SystemExit(1)


if __name__ == "__main__":
    main()