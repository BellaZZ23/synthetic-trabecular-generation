#!/usr/bin/env python3
r"""
synthetic_trabecular_v15_morphometric_control.py
v15 — HONEYCOMB FIX (v3 — zero-crossing walls)

Fixes A-C from previous version PLUS:

  Fix D: plate_likeness_field returns continuous values (not binary).

  Fix E: Skeletonization skipped for plate-dominant structures.

  Fix F: ZERO-CROSSING WALL EXTRACTION — bone is defined as the
         thin surface where the isotropic Gaussian field crosses zero:
         bone = abs(field) < wall_thickness.
         This naturally produces connected plate/wall topology because
         zero-level-sets of smooth Gaussian fields are connected surfaces.
         Wall thickness is binary-searched to hit target BV/TV.
         Replaces superlevel-set thresholding (field >= threshold) which
         produced disconnected blobs.

Best honeycomb params:
  python synthetic_trabecular_v15_morphometric_control.py \
      --voi-dirs data/derived/VOI1 data/derived/VOI4 \
      --outdir output/honeycomb_test \
      --num-samples 5 \
      --xy 128 --z 40 \
      --voxel-um 39 \
      --bvtv 0.33 \
      --tbth-um 180 \
      --base-sigma 2.5 \
      --aniso-ratio 1.0 \
      --warp-amp 1.2 \
      --warp-sigma 12.0 \
      --plate-weight 0.7 \
      --rod-weight 0.3 \
      --proto-close-iters 3 \
      --marrow-mean 15 \
      --bone-mean 90 \
      --solid-fill-sigma 0.8 \
      --noise-sd 2.0 \
      --bg-tex-sd 0.5 \
      --base-seed 100
"""
from __future__ import annotations
import argparse, json
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
    try: from skimage.morphology import skeletonize as skeletonize_3d
    except ImportError: skeletonize_3d = None

VOI_PRESETS = {"voi1":{"xy":300,"z":432},"voi2":{"xy":152,"z":432},
               "voi3":{"xy":152,"z":432},"voi4":{"xy":152,"z":432},"voi5":{"xy":152,"z":432}}
TAMIMI_BOUNDS = {"BVTV":(0.05,0.50),"TbTh_um":(80.0,300.0),
                 "TbN_per_mm":(0.5,5.0),"TbSp_um":(150.0,1200.0)}


# ──────────────────────────────────────────────────────────
#  VOI LOADING
# ──────────────────────────────────────────────────────────

def load_all_voi_targets(voi_dirs, voxel_um):
    all_t = []
    for d in voi_dirs:
        for f in sorted(Path(d).glob("*_targets.json")):
            with open(f) as fh: data = json.load(fh)
            data["_src"] = str(f); all_t.append(data)
            print(f"  Loaded: {f.name}")
    if not all_t: raise FileNotFoundError("No *_targets.json found")
    print(f"\nPooling {len(all_t)} files (voxel={voxel_um}um)")
    bv, th, sp = [], [], []
    for t in all_t:
        bv.append(float(t.get("BVTV", 0)))
        rv = 1.0; vz = t.get("voxel_um_zyx")
        if vz and len(vz) >= 1: rv = float(vz[0])
        c = voxel_um / max(1.0, rv)
        th.append(float(t.get("TbTh_um_p90", t.get("TbTh_um_p50", 0))) * c * 2.0)
        sp.append(float(t.get("TbSp_um_p50", 0)) * c * 2.0)
    bv = np.array(bv); th = np.array(th); sp = np.array(sp)
    tn = bv / (th / 1000.0 + 1e-9)
    def st(a): return {"mean":float(a.mean()),"std":float(a.std()),
                       "min":float(a.min()),"max":float(a.max())}
    p = {"n":len(all_t),"voxel_um":voxel_um,
         "BVTV":st(bv),"TbTh_um":st(th),"TbSp_um":st(sp),"TbN_per_mm":st(tn)}
    print(f"  BV/TV: {p['BVTV']['mean']:.3f}+/-{p['BVTV']['std']:.3f}")
    print(f"  Tb.Th: {p['TbTh_um']['mean']:.1f}+/-{p['TbTh_um']['std']:.1f}um")
    return p


def sample_targets_from_pool(pooled, rng, n):
    samples = []
    for i in range(n):
        def s(k, lo, hi):
            v = pooled[k]
            return float(np.clip(rng.normal(v["mean"], max(v["std"], v["mean"]*0.05)), lo, hi))
        bvtv = s("BVTV", 0.10, 0.40)
        tbth = s("TbTh_um", 130, 240)
        tbsp = s("TbSp_um", 200, 700)
        tbn  = float(np.clip(bvtv / (tbth / 1000.0), 0.5, 4.0))
        samples.append({"bvtv":bvtv,"tbth_um":tbth,"tbn_per_mm":tbn,
                        "tbsp_um":tbsp,"sample_index":i})
    return samples


# ──────────────────────────────────────────────────────────
#  DATACLASSES
# ──────────────────────────────────────────────────────────

@dataclass
class RidgeParams:
    base_sigma:          float = 2.5
    warp_sigma:          float = 12.0
    warp_amp:            float = 1.2
    hessian_sigma:       float = 1.4
    ridge_strength:      float = 1.0
    proto_q_hi:          float = 0.78
    proto_q_lo:          float = 0.65
    proto_close_iters:   int   = 3
    proto_open_iters:    int   = 0
    proto_min_component: int   = 400
    use_skeleton:        bool  = True
    skeleton_prune_lmin: int   = 8
    reconnect_close_iters:int  = 3
    radius_mode:         str   = "branch"
    radius_jitter:       float = 0.04
    radius_smooth_sigma: float = 3.0
    radius_scale_hint:   float = 1.0
    prune_small_components:int = 0
    aniso_ratio:         float = 1.0
    plate_weight:        float = 0.7
    rod_weight:          float = 0.3


@dataclass
class GrayParams:
    write_gray:       bool            = True
    marrow_mean:      float           = 15.0
    bone_mean:        float           = 90.0
    solid_fill_sigma: Optional[float] = 0.8
    pve_sigma:        float           = 0.5
    noise_sd:         float           = 5.0
    bg_tex_sd:        float           = 2.0
    unsharp:          float           = 0.6
    unsharp_sigma:    float           = 0.8


# ──────────────────────────────────────────────────────────
#  IO
# ──────────────────────────────────────────────────────────

def save_png_u8(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img.astype(np.uint8), mode="L").save(path)

def save_tif_u8(s, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(path, s.astype(np.uint8), imagej=True, dtype=np.uint8)

def save_json(o, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(o, f, indent=2)

def tbth_um_to_radius_vox(t, v): return max(0.5, (t / v) / 2.0)
def tbn_per_mm_to_base_sigma(t, v): return float(max(1.5, 1000.0 / max(0.1, float(t)) / float(v) / 4.0))
def compute_adaptive_fill_sigma(r): return float(np.clip(0.35 * r, 0.3, 1.5))


# ──────────────────────────────────────────────────────────
#  FIELD GENERATION
# ──────────────────────────────────────────────────────────

def normalize(f):
    x = f.astype(np.float32)
    x -= float(x.mean()); x /= float(x.std() + 1e-6)
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

def make_isotropic_field(shape, rng, base_sigma):
    """
    FIX B: Pure isotropic Gaussian random field.
    Produces foam/plate-like network when thresholded.
    """
    f = rng.normal(0, 1, size=shape).astype(np.float32)
    f = ndi.gaussian_filter(f, sigma=float(base_sigma))
    return normalize(f)

def plate_likeness_field(f, base_sigma):
    """
    FIX C + FIX D: Continuous plate/sheet detector.
    Returns a continuous field (NOT binary) so that blending with
    vesselness produces a smooth response for hysteresis thresholding.
    """
    g1 = ndi.gaussian_filter(f, sigma=max(0.8, base_sigma * 0.6))
    g2 = ndi.gaussian_filter(f, sigma=max(0.8, base_sigma * 1.2))

    gx = ndi.sobel(g1, axis=2); gy = ndi.sobel(g1, axis=1); gz = ndi.sobel(g1, axis=0)
    gmag = np.sqrt(gx**2 + gy**2 + gz**2).astype(np.float32)
    gmag = normalize(gmag)

    # FIX D: Return CONTINUOUS blend — no binary threshold here.
    # The old code thresholded to 0/1 which made hysteresis meaningless.
    return normalize(0.5 * g2 + 0.5 * gmag)

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
    r1 = np.abs(l1) / (np.abs(l3) + eps)
    r2 = np.abs(l2) / (np.abs(l3) + eps)
    V  = np.exp(-(r1*r1)/0.25) * np.exp(-(r2*r2)/0.25)
    return (V / (float(V.max()) + 1e-6)).astype(np.float32)


# ──────────────────────────────────────────────────────────
#  MORPHOLOGY HELPERS
# ──────────────────────────────────────────────────────────

def anti_block_round(b, s):
    if float(s) <= 0: return b.astype(np.uint8)
    return (ndi.gaussian_filter(b.astype(np.float32), sigma=float(s)) >= 0.5).astype(np.uint8)

def keep_largest_component(v):
    st = ndi.generate_binary_structure(3, 2)
    l, n = ndi.label(v.astype(bool), structure=st)
    if n == 0: return v.astype(np.uint8)
    c = np.bincount(l.ravel()); c[0] = 0
    return (l == int(c.argmax())).astype(np.uint8)

def remove_small_components(v, ms):
    if int(ms) <= 0: return v.astype(np.uint8)
    st = ndi.generate_binary_structure(3, 2)
    l, n = ndi.label(v.astype(bool), structure=st)
    if n == 0: return v.astype(np.uint8)
    c = np.bincount(l.ravel()); k = c >= int(ms); k[0] = False
    return k[l].astype(np.uint8)

def morph_iters(v, op, it):
    if int(it) <= 0: return v.astype(np.uint8)
    st = ndi.generate_binary_structure(3, 2); x = v.astype(bool)
    if op == "close": x = ndi.binary_closing(x, structure=st, iterations=int(it))
    elif op == "open": x = ndi.binary_opening(x, structure=st, iterations=int(it))
    return x.astype(np.uint8)

def hysteresis_on_response(R, ql, qh):
    ql = float(np.clip(ql, 0.5, 0.995)); qh = float(np.clip(qh, ql+1e-3, 0.999))
    th = float(np.quantile(R, qh)); tl = float(np.quantile(R, ql))
    s = R >= th; w = R >= tl
    st = ndi.generate_binary_structure(3, 2); l, n = ndi.label(w, structure=st)
    if n == 0: return s.astype(np.uint8), {"thr_lo": tl, "thr_hi": th}
    sl = np.unique(l[s]); k = np.zeros(n+1, dtype=bool); k[sl] = True; k[0] = False
    return k[l].astype(np.uint8), {"thr_lo": tl, "thr_hi": th}

def skeletonize_with_skimage(p):
    if skeletonize_3d is None: raise RuntimeError("skeletonize_3d unavailable")
    return skeletonize_3d(p.astype(bool)).astype(np.uint8)

def neighbor_degree_26(sk):
    st = ndi.generate_binary_structure(3, 2)
    n  = ndi.convolve(sk.astype(np.uint8), st.astype(np.uint8), mode="constant", cval=0)
    return (n - sk.astype(np.uint8)).astype(np.int16)

def prune_short_end_branches(sk01, lmin):
    lmin = int(max(1, lmin)); st = ndi.generate_binary_structure(3, 2)
    sk = sk01.astype(bool); rm = 0
    for _ in range(50):
        deg = neighbor_degree_26(sk.astype(np.uint8))
        ep = sk & (deg == 1); jn = sk & (deg >= 3)
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


# ──────────────────────────────────────────────────────────
#  PROTO NETWORK — FIX D, E
# ──────────────────────────────────────────────────────────

def calibrate_bvtv_by_zero_crossings(field, target_bvtv,
                                      close_iters, min_component, round_sigma,
                                      tol=0.005, max_iter=40):
    """
    FIX F: Zero-crossing wall extraction.
    Bone = thin walls where the Gaussian field crosses zero.
    bone = abs(field) < wall_thickness
    Wider wall_thickness = more bone (higher BV/TV).
    Binary-search wall_thickness to hit target BV/TV.

    This produces PLATE topology naturally — the zero-crossings of a
    smooth Gaussian field are connected surfaces (walls/sheets), not
    disconnected blobs.
    """
    st26 = ndi.generate_binary_structure(3, 2)
    f_abs = np.abs(field).astype(np.float32)

    lo, hi = 0.01, float(np.percentile(f_abs, 95))
    best_mask = None; best_err = float("inf")

    for _ in range(max_iter):
        wt = 0.5 * (lo + hi)
        mask = (f_abs < wt).astype(np.uint8)

        # Morphological cleanup
        if close_iters > 0:
            mask = ndi.binary_closing(mask.astype(bool), structure=st26,
                                       iterations=close_iters).astype(np.uint8)
        if min_component > 0:
            mask = remove_small_components(mask, min_component)
        if round_sigma > 0:
            mask = anti_block_round(mask, round_sigma)

        bvtv = float(mask.astype(bool).mean())
        err = abs(bvtv - target_bvtv)

        if err < best_err:
            best_err = err; best_mask = mask.copy()

        if err < tol:
            break

        # Wider wall = more bone
        if bvtv < target_bvtv:
            lo = wt
        else:
            hi = wt

    return best_mask, {"bvtv_target": target_bvtv,
                       "bvtv_got": float(best_mask.astype(bool).mean()),
                       "wall_thickness": float(0.5 * (lo + hi)),
                       "best_err": best_err}


def make_proto_and_skeleton(shape, rp, rng, skel_mode, fiji_exe, fiji_cmd, dbg,
                            target_bvtv=None):
    """
    FIX A: base_sigma floor removed.
    FIX B: Isotropic field for foam/honeycomb.
    FIX F: When plate_weight >= 0.5, extract bone as ZERO-CROSSINGS
           of the isotropic Gaussian field: abs(field) < wall_thickness.
           This gives connected plate/wall topology naturally.
           No skeleton, no vesselness needed for plate path.
    """
    # Isotropic Gaussian field
    f = make_isotropic_field(shape, rng, float(rp.base_sigma))
    f = smooth_warp(f, rng, float(rp.warp_sigma), float(rp.warp_amp))
    f = normalize(f)

    # Vesselness (rod/edge detector)
    rod_R = vesselness_ridge(f, sigma=float(rp.hessian_sigma))
    rod_R = normalize(np.clip(rod_R * float(rp.ridge_strength), 0.0, 1.0))

    # FIX D: Continuous plate field
    plate_R = plate_likeness_field(f, base_sigma=float(rp.base_sigma))

    # Blend
    pw = float(rp.plate_weight); rw = float(rp.rod_weight)
    R = normalize(rw * rod_R + pw * plate_R)

    plate_dominant = pw >= 0.5

    if plate_dominant and target_bvtv is not None:
        # ── FIX F: zero-crossing plate path ─────────────────
        # Bone = thin walls where the isotropic field crosses zero.
        # abs(field) < wall_thickness gives connected plate surfaces.
        # No skeleton, no vesselness, no blending needed — just the
        # raw Gaussian field's zero-set, which is naturally plate-like.
        print(f"    [plate mode] Zero-crossing walls for BV/TV={target_bvtv:.3f}")
        bone01, cal_info = calibrate_bvtv_by_zero_crossings(
            f, target_bvtv,
            close_iters=int(rp.proto_close_iters),
            min_component=int(rp.proto_min_component),
            round_sigma=0.35,
        )
        bone01 = keep_largest_component(bone01)

        if dbg:
            save_tif_u8((f - f.min()) / (f.max() - f.min() + 1e-6) * 255, dbg/"field.tif")
            save_tif_u8(normalize(rod_R) * 127 + 128, dbg/"rod_response.tif")
            save_tif_u8(normalize(plate_R) * 127 + 128, dbg/"plate_response.tif")
            save_tif_u8(normalize(R) * 127 + 128, dbg/"blended_response.tif")
            save_tif_u8((bone01*255).astype(np.uint8), dbg/"proto_network.tif")

        return bone01.astype(np.uint8), {
            "mode": "plate_direct",
            "plate_weight": pw, "rod_weight": rw,
            "calibration": cal_info,
            "used_skeleton": False,
        }

    else:
        # ── Rod-dominant path: original skeleton+thicken ────
        p01, hy = hysteresis_on_response(R, float(rp.proto_q_lo), float(rp.proto_q_hi))
        p01 = morph_iters(p01, "close", int(rp.proto_close_iters))
        p01 = morph_iters(p01, "open",  int(rp.proto_open_iters))
        p01 = remove_small_components(p01, int(rp.proto_min_component))

        st26 = ndi.generate_binary_structure(3, 2)
        if p01.astype(bool).sum() > 0:
            p01 = ndi.binary_closing(p01.astype(bool), structure=st26,
                                      iterations=1).astype(np.uint8)

        sr = p01.copy().astype(np.uint8); us = False
        if bool(rp.use_skeleton) and skel_mode == "skimage":
            sr = skeletonize_with_skimage(p01); us = True

        sp, pi = prune_short_end_branches(sr, lmin=int(rp.skeleton_prune_lmin))
        if int(rp.reconnect_close_iters) > 0:
            sp = morph_iters(sp, "close", int(rp.reconnect_close_iters))

        if dbg:
            save_tif_u8((f - f.min()) / (f.max() - f.min() + 1e-6) * 255, dbg/"field.tif")
            save_tif_u8((rod_R*255).astype(np.uint8),   dbg/"rod_response.tif")
            save_tif_u8(normalize(plate_R) * 127 + 128, dbg/"plate_response.tif")
            save_tif_u8(normalize(R) * 127 + 128,       dbg/"blended_response.tif")
            save_tif_u8((p01*255).astype(np.uint8),     dbg/"proto_network.tif")
            save_tif_u8((sp*255).astype(np.uint8),      dbg/"skeleton_pruned.tif")

        return sp.astype(np.uint8), {"hysteresis": hy, "used_skeleton": us,
                                      "skel_mode": skel_mode, "prune": pi,
                                      "plate_weight": pw, "rod_weight": rw,
                                      "mode": "rod_skeleton"}


# ──────────────────────────────────────────────────────────
#  THICKENING (rod path only)
# ──────────────────────────────────────────────────────────

def radius_samples_for_skeleton(sk01, rng, br, mode, jit, ss):
    sk  = sk01.astype(bool); rad = np.zeros(sk01.shape, dtype=np.float32)
    if not sk.any(): return rad
    b = float(max(0.5, br)); j = float(np.clip(jit, 0.0, 0.9))
    st = ndi.generate_binary_structure(3, 2)
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
        rad = num / den; rad[~sk] = 0.0
    return rad

def thicken_from_skeleton_radius_field(sk01, rng, tbvtv, br, rm, rj, rss, rsh, dbg):
    sk = sk01.astype(bool)
    if not sk.any(): return np.zeros_like(sk01, dtype=np.uint8), {"error": "Empty"}
    rs   = radius_samples_for_skeleton(sk01, rng=rng, br=br, mode=rm, jit=rj, ss=rss)
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
    return bone, {"base_r":float(br),"min_r":float(mr),"scale":float(bs),
                  "bvtv_tgt":float(tgt),"bvtv_got":float(bone.mean()),
                  "warn_target_miss":bool(abs(float(bone.mean())-tgt) > 0.10)}


# ──────────────────────────────────────────────────────────
#  GRAYSCALE
# ──────────────────────────────────────────────────────────

def microct_gray_solid(bone01, gp, rng, br=2.0):
    bone = bone01.astype(bool)
    d    = ndi.distance_transform_edt(bone).astype(np.float32)
    sig  = float(gp.solid_fill_sigma) if gp.solid_fill_sigma is not None \
           else compute_adaptive_fill_sigma(br)
    fill = (1.0 - np.exp(-(d / max(0.2, sig))**2)) * bone.astype(np.float32)
    g    = float(gp.marrow_mean) + fill * (float(gp.bone_mean) - float(gp.marrow_mean))
    if float(gp.pve_sigma)  > 0: g = ndi.gaussian_filter(g, sigma=float(gp.pve_sigma))
    if float(gp.bg_tex_sd)  > 0: g += rng.normal(0, float(gp.bg_tex_sd), size=g.shape).astype(np.float32)
    if float(gp.noise_sd)   > 0: g += rng.normal(0, float(gp.noise_sd),  size=g.shape).astype(np.float32)
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
    t50=2*p(tv,50); t90=2*p(tv,90); s50=2*p(sv,50); s90=2*p(sv,90)
    tm = (2*float(np.mean(tv))/1000.0) if tv.size else 1e-6
    tn = bv/tm if tm > 0 else 0.0
    eu = float(euler_number(bone, connectivity=3))
    st = ndi.generate_binary_structure(3, 2); l, n = ndi.label(bone, structure=st)
    lf = 0.0; nc = int(n)
    if n > 0:
        c = np.bincount(l.ravel()); c[0] = 0
        lf = float(c.max()) / float(max(1, bone.sum()))
    return {"BVTV":bv,"TbTh_um_p50":t50,"TbTh_um_p90":t90,
            "TbSp_um_p50":s50,"TbSp_um_p90":s90,"TbN_per_mm":float(tn),
            "Euler":eu,"ConnProxy":float(1-eu),"n_components":nc,"lcc_frac":lf}

def skeleton_graph_stats(sk01):
    sk = sk01.astype(bool)
    if not sk.any(): return {"skel_voxels":0,"junctions":0,"endpoints":0,"ej_ratio":None}
    deg = neighbor_degree_26(sk.astype(np.uint8))
    ep  = int((sk & (deg==1)).sum()); jn = int((sk & (deg>=3)).sum())
    return {"skel_voxels":int(sk.sum()),"junctions":jn,"endpoints":ep,
            "ej_ratio":float(ep)/max(1,jn) if jn > 0 else None}

def validate_morphometrics(m, t):
    ch = {}
    for mk, tk, tol, lb in [("BVTV","bvtv_target",0.05,"BV/TV"),
                              ("TbTh_um_p50","tbth_um_target",0.15,"Tb.Th"),
                              ("TbN_per_mm","tbn_target",0.20,"Tb.N"),
                              ("TbSp_um_p50","tbsp_um_target",0.15,"Tb.Sp")]:
        tv = t.get(tk); mv = m.get(mk)
        if tv and mv and float(tv) > 0:
            re = abs(float(mv)-float(tv))/float(tv)
            ch[lb] = {"measured":float(mv),"target":float(tv),
                      "rel_error":re,"pass":bool(re<=tol)}
    lcc = m.get("lcc_frac", 0.0)
    ch["LCC"] = {"lcc_frac":float(lcc),"pass":bool(lcc>=0.80)}
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
#  MAIN GENERATION — FIX E: plate vs rod path
# ──────────────────────────────────────────────────────────

def generate_one(params, args, outdir, label="", seed_override=None):
    seed = seed_override if seed_override is not None else int(args.base_seed)
    rng  = np.random.default_rng(seed)
    bvtv = params["bvtv"]; tbth = params["tbth_um"]
    tbn  = params["tbn_per_mm"]; tbsp = params["tbsp_um"]
    vu   = params.get("voxel_um", float(args.voxel_um or 39.0))
    shape = (params.get("shape_z", args.z or 160),
             params.get("shape_xy", args.xy or 300),
             params.get("shape_xy", args.xy or 300))
    br = tbth_um_to_radius_vox(tbth, vu)
    bs = float(args.base_sigma) if args.base_sigma is not None \
         else tbn_per_mm_to_base_sigma(tbn, vu)
    # FIX A: NO base_sigma floor

    pw = float(args.plate_weight)
    plate_dominant = pw >= 0.5

    print(f"\n  [{label}] seed={seed}  mode={'plate' if plate_dominant else 'rod'}")
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
        use_skeleton=bool(int(args.use_skeleton)),
        skeleton_prune_lmin=int(args.skeleton_prune_lmin),
        reconnect_close_iters=int(args.reconnect_close_iters),
        radius_mode=str(args.radius_mode), radius_jitter=float(args.radius_jitter),
        radius_smooth_sigma=float(args.radius_smooth_sigma),
        radius_scale_hint=float(args.radius_scale_hint),
        aniso_ratio=float(args.aniso_ratio),
        plate_weight=float(args.plate_weight),
        rod_weight=float(args.rod_weight),
    )
    gp = GrayParams(
        write_gray=bool(int(args.write_gray)),
        solid_fill_sigma=args.solid_fill_sigma,
        marrow_mean=float(args.marrow_mean), bone_mean=float(args.bone_mean),
        noise_sd=float(args.noise_sd), bg_tex_sd=float(args.bg_tex_sd),
    )

    # ── FIX E: pass target_bvtv so plate path can calibrate directly ──
    result01, si = make_proto_and_skeleton(
        shape=shape, rp=rp, rng=rng,
        skel_mode=str(args.skeleton_mode),
        fiji_exe=args.fiji_exe,
        fiji_cmd=str(args.fiji_command), dbg=dbg,
        target_bvtv=bvtv,
    )

    if plate_dominant:
        # Plate path: result01 is already the final bone mask with calibrated BV/TV
        bone01 = result01
        ti = si  # calibration info is in si
    else:
        # Rod path: result01 is a skeleton, need to thicken
        bone01, ti = thicken_from_skeleton_radius_field(
            result01, rng, bvtv, br,
            str(args.radius_mode), float(args.radius_jitter),
            float(args.radius_smooth_sigma), float(args.radius_scale_hint), dbg)
        bone01 = anti_block_round(bone01, float(args.round_sigma))

    if int(args.min_component_size) > 0:
        bone01 = remove_small_components(bone01, int(args.min_component_size))
    if bool(int(args.enforce_lcc)):
        bone01 = keep_largest_component(bone01)

    void01 = (1 - bone01).astype(np.uint8); Z = shape[0]
    save_tif_u8((bone01*255).astype(np.uint8), outdir/"mask.tif")
    save_tif_u8((void01*255).astype(np.uint8), outdir/"void.tif")
    save_png_u8((bone01[Z//2]*255).astype(np.uint8), outdir/"mid.png")

    if gp.write_gray:
        gray = microct_gray_solid(bone01, gp, rng, br=br)
        save_tif_u8(gray, outdir/"gray.tif")
        save_png_u8(gray[Z//2], outdir/"gray_mid.png")

    morph = measure_all_morphometrics(bone01, vu)
    tgt   = {"bvtv_target":bvtv,"tbth_um_target":tbth,
              "tbn_target":tbn,"tbsp_um_target":tbsp}
    val   = validate_morphometrics(morph, tgt)
    tw    = check_tamimi_bounds(morph)

    # skeleton stats only meaningful for rod path
    sk_stats = {}
    if not plate_dominant:
        sk_stats = skeleton_graph_stats(result01)

    met = {"version":"v15.3_zerocross","label":label,"seed":seed,
           "morphometrics":morph,"targets":tgt,"validation":val,
           "tamimi_warnings":tw,"skeleton_stats":sk_stats,
           "thick_info":ti,"params":{"ridge":asdict(rp),"gray":asdict(gp)},
           "shape_zyx":list(shape),"voxel_um":vu}
    save_json(met, outdir/"metrics.json")
    print(f"    BV/TV: target={bvtv:.3f} measured={morph['BVTV']:.3f}")
    if tw:
        for w in tw: print(f"    ! {w}")
    return met


# ──────────────────────────────────────────────────────────
#  PARSER + MAIN
# ──────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="v15.3 zero-crossing trabecular generator")
    p.add_argument("--voi-dirs",  nargs="+", type=str, default=None)
    p.add_argument("--targets-json", type=str, default=None)
    p.add_argument("--profile",   type=str, default=None, choices=["tamimi-hf","tamimi-hoa"])
    p.add_argument("--outdir",    type=str, default="output/ml_dataset")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--base-seed",   type=int, default=42)
    p.add_argument("--voi-preset",  type=str, default=None, choices=list(VOI_PRESETS.keys()))
    p.add_argument("--bvtv",        type=float, default=None)
    p.add_argument("--tbth-um",     dest="tbth_um",    type=float, default=None)
    p.add_argument("--tbn-per-mm",  dest="tbn_per_mm", type=float, default=None)
    p.add_argument("--tbsp-um",     dest="tbsp_um",    type=float, default=None)
    p.add_argument("--voxel-um",    type=float, default=39.0)
    p.add_argument("--xy",          type=int,   default=None)
    p.add_argument("--z",           type=int,   default=None)
    p.add_argument("--marrow-mean", type=float, default=15.0)
    p.add_argument("--bone-mean",   type=float, default=90.0)
    p.add_argument("--noise-sd",    type=float, default=5.0)
    p.add_argument("--bg-tex-sd",   type=float, default=2.0)
    p.add_argument("--base-sigma",  type=float, default=None)
    p.add_argument("--warp-sigma",  type=float, default=12.0)
    p.add_argument("--warp-amp",    type=float, default=1.2)
    p.add_argument("--hessian-sigma",     type=float, default=1.4)
    p.add_argument("--ridge-strength",    type=float, default=1.0)
    p.add_argument("--proto-q-hi",        type=float, default=0.78)
    p.add_argument("--proto-q-lo",        type=float, default=0.65)
    p.add_argument("--proto-close-iters", type=int,   default=3)
    p.add_argument("--proto-open-iters",  type=int,   default=0)
    p.add_argument("--proto-min-component", type=int, default=400)
    p.add_argument("--use-skeleton",      type=int,   default=1)
    p.add_argument("--skeleton-mode",     type=str,   default="skimage",
                   choices=["skimage","fiji"])
    p.add_argument("--fiji-exe",          type=str,   default=None)
    p.add_argument("--fiji-command",      type=str,   default="Skeletonize (2D/3D)")
    p.add_argument("--skeleton-prune-lmin",   type=int,   default=8)
    p.add_argument("--reconnect-close-iters", type=int,   default=3)
    p.add_argument("--radius-mode",       type=str,   default="branch",
                   choices=["branch","voxel"])
    p.add_argument("--radius-jitter",     type=float, default=0.04)
    p.add_argument("--radius-smooth-sigma", type=float, default=3.0)
    p.add_argument("--radius-scale-hint",   type=float, default=1.0)
    p.add_argument("--aniso-ratio",    type=float, default=1.0)
    p.add_argument("--plate-weight",   type=float, default=0.7,
                   help="Weight for plate field (>=0.5 uses plate path, <0.5 uses rod+skeleton path)")
    p.add_argument("--rod-weight",     type=float, default=0.3)
    p.add_argument("--enforce-lcc",    type=int,   default=1)
    p.add_argument("--min-component-size", type=int, default=500)
    p.add_argument("--round-sigma",    type=float, default=0.35)
    p.add_argument("--solid-fill-sigma", type=float, default=0.8)
    p.add_argument("--write-gray",     type=int,   default=1)
    p.add_argument("--debug-skeleton", type=int,   default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.voi_preset:
        pr = VOI_PRESETS[args.voi_preset]
        if args.xy is None: args.xy = pr["xy"]
        if args.z  is None: args.z  = pr["z"]

    if args.voi_dirs is not None:
        print(f"{'='*60}\n  POOLED: {len(args.voi_dirs)} dirs, {args.num_samples} samples")
        print(f"  Gray: marrow={args.marrow_mean}, bone={args.bone_mean}")
        print(f"  Plate weight: {args.plate_weight}  Rod weight: {args.rod_weight}")
        pw = float(args.plate_weight)
        print(f"  Mode: {'PLATE (zero-crossing walls)' if pw >= 0.5 else 'ROD (skeleton+thicken)'}")
        print(f"{'='*60}")
        pooled = load_all_voi_targets(args.voi_dirs, voxel_um=float(args.voxel_um))
        save_json(pooled, Path(args.outdir)/"pooled_statistics.json")
        prng    = np.random.default_rng(int(args.base_seed))
        samples = sample_targets_from_pool(pooled, prng, int(args.num_samples))

        for s in samples:
            if args.bvtv       is not None: s["bvtv"]       = float(args.bvtv)
            if args.tbth_um    is not None: s["tbth_um"]    = float(args.tbth_um)
            if args.tbsp_um    is not None: s["tbsp_um"]    = float(args.tbsp_um)
            if args.tbn_per_mm is not None: s["tbn_per_mm"] = float(args.tbn_per_mm)
            elif args.bvtv is not None or args.tbth_um is not None:
                s["tbn_per_mm"] = float(np.clip(
                    float(s["bvtv"]) / (float(s["tbth_um"]) / 1000.0), 0.5, 4.0))

        print(f"\n  Sampled {len(samples)} targets:")
        for s in samples:
            print(f"    [{s['sample_index']:03d}] BV/TV={s['bvtv']:.3f} "
                  f"Tb.Th={s['tbth_um']:.0f}um Tb.N={s['tbn_per_mm']:.2f}/mm")

        am = []
        for s in samples:
            s["voxel_um"] = float(args.voxel_um)
            s["shape_z"]  = args.z  or 160
            s["shape_xy"] = args.xy or 300
            m = generate_one(s, args,
                             Path(args.outdir)/f"sample_{s['sample_index']:03d}",
                             label=f"sample_{s['sample_index']:03d}",
                             seed_override=int(args.base_seed)+s["sample_index"]+1)
            am.append(m)

        save_json({"version":"v15.3_zerocross","n":len(am),"pooled":pooled,
                   "gray":{"marrow":args.marrow_mean,"bone":args.bone_mean},
                   "samples":[{"label":m["label"],"seed":m["seed"],
                                "bvtv":m["targets"]["bvtv_target"],
                                "bvtv_m":m["morphometrics"]["BVTV"]} for m in am]},
                  Path(args.outdir)/"dataset_manifest.json")
        print(f"\n{'='*60}\n  Dataset: {len(am)} samples in {args.outdir}/\n{'='*60}")

    else:
        print("Provide: --voi-dirs, --targets-json, or --profile")
        raise SystemExit(1)


if __name__ == "__main__":
    main()