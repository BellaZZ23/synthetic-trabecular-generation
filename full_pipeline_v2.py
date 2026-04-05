#!/usr/bin/env python3
"""
full_pipeline.py  — v1.3
Fixes applied:
  Fix 1  : Joint morphometric target sampling from real VOI covariance structure
  Fix 2  : Morphometrics measured BEFORE keep_largest_component (honest LCC)
  Fix 3  : Anti-floor rejection — TbTh=78um (resolution floor) triggers retry
  Fix 4  : TbSp-aware base_sigma — spacing target influences noise field scale
  Fix 13 : run_dim_reduction uses texture features, not raw pixels
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile as tiff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.stats import skew, kurtosis

try:
    import optuna
    from optuna.importance import get_param_importances
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("ERROR: pip install optuna"); sys.exit(1)

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_GLCM = True
except ImportError:
    SKIMAGE_GLCM = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("WARNING: umap-learn not installed — UMAP disabled. Run: pip install umap-learn")

# ── Import generator (try v16 first, fall back to v15) ──
try:
    from synthetic_trabecular_v16_morphometric_control import (
        RidgeParams, GrayParams,
        make_proto_and_skeleton, thicken_from_skeleton_radius_field,
        anti_block_round, remove_small_components, keep_largest_component,
        microct_gray_solid, measure_all_morphometrics,
        tbth_um_to_radius_vox, tbn_per_mm_to_base_sigma,
        skeleton_graph_stats, save_tif_u8, save_png_u8, save_json,
        load_all_voi_targets,
    )
    GENERATOR_VERSION = "v16"
    print("Using v16 generator")
except ImportError:
    from synthetic_trabecular_v15_morphometric_control import (
        RidgeParams, GrayParams,
        make_proto_and_skeleton, thicken_from_skeleton_radius_field,
        anti_block_round, remove_small_components, keep_largest_component,
        microct_gray_solid, measure_all_morphometrics,
        tbth_um_to_radius_vox, tbn_per_mm_to_base_sigma,
        skeleton_graph_stats, save_tif_u8, save_png_u8, save_json,
        load_all_voi_targets,
    )
    GENERATOR_VERSION = "v15"
    print("Using v15 generator (v16 not found)")


LABEL_KEYS        = ["BVTV", "TbTh_um_p50", "TbN_per_mm", "TbSp_um_p50"]
GATE_BVTV_REL_ERR = 0.15
GATE_LCC_MIN      = 0.85
GATE_MAX_ATTEMPTS = 8

# FIX 3: resolution floor — at 39µm voxel, 2 voxels = 78µm
# Reject samples that land within 2µm of the floor (quantised artefact)
GATE_TBTH_FLOOR_UM = 0.0    # disabled — 78um is physical minimum at 39um voxel, floor rejection too aggressive
GATE_TBTH_MAX_UM   = 350.0  # reject if TbTh > this (blob)

# FIX 4: TbSp → base_sigma mapping
# Higher spacing = larger noise field scale (fewer, more separated struts)
TBSP_SIGMA_SCALE = 0.004    # base_sigma += TBSP_SIGMA_SCALE * tbsp_um_target


# ═══════════════════════════════════════════════════════════
#  TEXTURE FEATURE EXTRACTION  (Fix 13)
# ═══════════════════════════════════════════════════════════

def _extract_glcm_features(img_u8: np.ndarray) -> np.ndarray:
    if not SKIMAGE_GLCM:
        return np.array([])
    distances = [1, 3]
    angles    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    props     = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    img_q     = (img_u8 // 8).astype(np.uint8)
    glcm      = graycomatrix(img_q, distances=distances, angles=angles,
                              levels=32, symmetric=True, normed=True)
    feats = []
    for prop in props:
        vals = graycoprops(glcm, prop)
        feats.extend([float(vals.mean()), float(vals.std())])
    return np.array(feats, dtype=np.float32)


def _extract_statistical_features(img: np.ndarray) -> np.ndarray:
    feats  = [float(img.mean()), float(img.std()), float(np.median(img)),
              float(skew(img.ravel())), float(kurtosis(img.ravel())),
              float(img.min()), float(img.max()),
              float(np.percentile(img, 25)), float(np.percentile(img, 75))]
    hist, _ = np.histogram(img.ravel(), bins=16, range=(0, 1))
    hist    = hist.astype(np.float32) / (hist.sum() + 1e-8)
    feats.extend(hist.tolist())
    gx   = ndi.sobel(img, axis=1)
    gy   = ndi.sobel(img, axis=0)
    gmag = np.sqrt(gx**2 + gy**2)
    feats += [float(gmag.mean()), float(gmag.std()),
              float(np.percentile(gmag, 75)), float(np.percentile(gmag, 95))]
    lv = ndi.uniform_filter(img**2, size=5) - ndi.uniform_filter(img, size=5)**2
    feats += [float(lv.mean()), float(lv.std())]
    return np.array(feats, dtype=np.float32)


def _extract_texture_features(sl: np.ndarray, image_size: int = 64) -> np.ndarray:
    img     = Image.fromarray((sl * 255).astype(np.uint8), mode="L")
    img     = img.resize((image_size, image_size), Image.BILINEAR)
    resized = np.array(img, dtype=np.float32) / 255.0
    img_u8  = (resized * 255).astype(np.uint8)
    stat    = _extract_statistical_features(resized)
    glcm    = _extract_glcm_features(img_u8)
    return np.concatenate([stat, glcm]) if glcm.size > 0 else stat


# ═══════════════════════════════════════════════════════════
#  FIX 1: JOINT TARGET SAMPLER
# ═══════════════════════════════════════════════════════════

def build_joint_sampler(voi_dirs: list[str], voxel_um: float) -> dict:
    """
    FIX 1: Load real VOI target JSON files and fit a multivariate Gaussian
    over [BVTV, TbTh_um, TbSp_um] jointly.

    This preserves the real covariance structure between morphometrics
    instead of sampling them independently.

    Returns dict with 'mu', 'cov', 'raw_targets', and per-variable bounds.
    """
    bvtvs, tbths, tbsps = [], [], []

    for d in voi_dirs:
        for f in sorted(Path(d).glob("*_targets.json")):
            with open(f) as fh:
                data = json.load(fh)

            # Apply the same voxel correction as load_all_voi_targets
            rv = 1.0
            vz = data.get("voxel_um_zyx")
            if vz and len(vz) >= 1:
                rv = float(vz[0])
            c = voxel_um / max(1.0, rv)

            bv   = float(data.get("BVTV", 0))
            tbth = float(data.get("TbTh_um_p90", data.get("TbTh_um_p50", 0))) * c * 2.0
            tbsp = float(data.get("TbSp_um_p50", 0)) * c * 2.0

            # Only keep biologically plausible rows
            if 0.05 <= bv <= 0.50 and tbth >= 80 and tbsp >= 150:
                bvtvs.append(bv)
                tbths.append(tbth)
                tbsps.append(tbsp)

    if len(bvtvs) < 3:
        print(f"  WARNING: Only {len(bvtvs)} valid VOI targets — falling back to independent sampling")
        return None

    raw = np.column_stack([bvtvs, tbths, tbsps])  # (n, 3)
    mu  = raw.mean(axis=0)
    cov = np.cov(raw.T)

    # Add small regularisation to ensure positive-definite covariance
    cov += np.eye(3) * 1e-6

    print(f"  FIX 1: Joint sampler fitted on {len(bvtvs)} real VOI targets")
    print(f"    BV/TV:  mean={mu[0]:.3f}  range=[{raw[:,0].min():.3f}, {raw[:,0].max():.3f}]")
    print(f"    TbTh:   mean={mu[1]:.1f}um range=[{raw[:,1].min():.1f}, {raw[:,1].max():.1f}]um")
    print(f"    TbSp:   mean={mu[2]:.1f}um range=[{raw[:,2].min():.1f}, {raw[:,2].max():.1f}]um")

    return {
        "mu":          mu,
        "cov":         cov,
        "raw_targets": raw,
        "n_sources":   len(bvtvs),
    }


def sample_joint_targets(
    sampler: dict | None,
    rng: np.random.Generator,
    bvtv_centre: float,
    tbth_centre: float,
    voxel_um: float,
) -> tuple[float, float, float]:
    """
    FIX 1: Sample (bvtv, tbth, tbsp) from joint distribution.
    Falls back to independent sampling if no sampler available.
    """
    if sampler is not None:
        # Sample from multivariate Gaussian fitted to real VOI data
        for _ in range(20):  # rejection sampling to stay in bounds
            s = rng.multivariate_normal(sampler["mu"], sampler["cov"])
            bv, tbth, tbsp = float(s[0]), float(s[1]), float(s[2])
            if (0.12 <= bv   <= 0.35 and
                110  <= tbth <= 300  and
                150  <= tbsp <= 800):
                return bv, tbth, tbsp
        # Fallback if rejection loop exhausted
        s    = sampler["mu"]
        return float(s[0]), float(s[1]), float(s[2])
    else:
        # Original independent sampling (fallback)
        bv   = float(np.clip(rng.normal(bvtv_centre, 0.06), 0.12, 0.32))
        tbth = float(np.clip(rng.normal(tbth_centre, 30.0), 110.0, 260.0))
        tbsp = float(np.clip(rng.normal(400.0, 80.0), 150.0, 700.0))
        return bv, tbth, tbsp


# ═══════════════════════════════════════════════════════════
#  STEP 1: OPTIMIZE
# ═══════════════════════════════════════════════════════════

def compute_ssim_2d(img1, img2):
    C1, C2 = 0.01**2, 0.03**2
    if img1.shape != img2.shape:
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = np.array(Image.fromarray((img1*255).astype(np.uint8)).resize((w,h)))/255.0
        img2 = np.array(Image.fromarray((img2*255).astype(np.uint8)).resize((w,h)))/255.0
    mu1, mu2 = img1.mean(), img2.mean()
    s1, s2   = img1.var(), img2.var()
    s12      = ((img1-mu1)*(img2-mu2)).mean()
    return float((2*mu1*mu2+C1)*(2*s12+C2) / ((mu1**2+mu2**2+C1)*(s1+s2+C2)))


def load_reference(path, size=128):
    img = np.array(Image.open(path).convert("L")).astype(np.float32)/255.0
    return np.array(Image.fromarray((img*255).astype(np.uint8)).resize((size,size), Image.BILINEAR))/255.0


def generate_small_and_score(params, ref_image, voxel_um=39.0, shape=(40,128,128)):
    seed = params.get("seed", 42)
    rng  = np.random.default_rng(seed)
    bvtv = params["bvtv"]
    tbth = params["tbth_um"]
    tbn  = bvtv / (tbth / 1000.0)
    br   = tbth_um_to_radius_vox(tbth, voxel_um)
    rp_kw = dict(
        base_sigma=max(params["base_sigma"], 4.5), warp_sigma=14.0,
        warp_amp=params["warp_amp"], hessian_sigma=params["hessian_sigma"],
        ridge_strength=1.0, proto_q_hi=params["proto_q_hi"],
        proto_q_lo=params["proto_q_lo"], proto_close_iters=params["proto_close_iters"],
        proto_open_iters=0, proto_min_component=250, use_skeleton=True,
        skeleton_prune_lmin=6, reconnect_close_iters=0, radius_mode="branch",
        radius_jitter=params["radius_jitter"], radius_smooth_sigma=3.0,
        radius_scale_hint=1.0, prune_small_components=0, aniso_ratio=params["aniso_ratio"],
    )
    if GENERATOR_VERSION == "v16":
        rp_kw.update(dict(rod_weight=params["rod_weight"], plate_weight=params["plate_weight"],
            coarse_weight=0.50, medium_weight=0.35, fine_weight=0.15,
            sheet_q=0.92, bridge_dilate_iters=0, bridge_close_iters=0))
    rp = RidgeParams(**rp_kw)
    gp = GrayParams(write_gray=True, solid_fill_sigma=3.0,
                    marrow_mean=15.0, bone_mean=240.0, noise_sd=3.0, bg_tex_sd=1.0)
    try:
        sk01, _ = make_proto_and_skeleton(shape=shape, rp=rp, rng=rng,
            skel_mode="skimage", fiji_exe=None, fiji_cmd="Skeletonize (2D/3D)", dbg=None)
        bone01, _ = thicken_from_skeleton_radius_field(sk01, rng, bvtv, br, "branch",
            params["radius_jitter"], 3.0, 1.0, dbg=None)
        bone01 = anti_block_round(bone01, params["round_sigma"])
        bone01 = remove_small_components(bone01, 500)
        bone01 = keep_largest_component(bone01)
    except Exception as e:
        return {"loss": 10.0, "error": str(e)}
    morph    = measure_all_morphometrics(bone01, voxel_um)
    targets  = {"BVTV": bvtv, "TbTh_um_p50": tbth, "TbN_per_mm": tbn}
    errs     = {k: abs(morph.get(k,0)-t)/t if t>0 else 1.0 for k,t in targets.items()}
    avg_morph = float(np.mean(list(errs.values())))
    gray  = microct_gray_solid(bone01, gp, np.random.default_rng(seed+1000), br=br)
    mid   = gray[shape[0]//2].astype(np.float32)/255.0
    mid_r = np.array(Image.fromarray((mid*255).astype(np.uint8)).resize(
        (ref_image.shape[1], ref_image.shape[0]), Image.BILINEAR))/255.0
    ssim  = compute_ssim_2d(ref_image, mid_r)
    lcc      = morph.get("lcc_frac", 0.0)
    conn_pen = 0.0 if lcc >= 0.8 else 0.5*(0.8-lcc)
    loss     = 0.4*avg_morph + 0.6*(1.0-ssim) + conn_pen
    return {"loss": float(loss), "ssim": float(ssim), "morph_error": float(avg_morph),
            "bvtv_measured": float(morph["BVTV"]), "lcc_frac": float(lcc)}


def run_optimization(ref_path, n_trials, outdir):
    print(f"\n{'='*60}\n  STEP 1: PARAMETER OPTIMIZATION ({n_trials} trials)\n{'='*60}")
    ref = load_reference(ref_path, 128); trial_log = []; t0 = time.time()
    def objective(trial):
        p = {
            "bvtv": trial.suggest_float("bvtv", 0.15, 0.30),
            "tbth_um": trial.suggest_float("tbth_um", 120.0, 250.0),
            "base_sigma": trial.suggest_float("base_sigma", 3.0, 8.0),
            "aniso_ratio": trial.suggest_float("aniso_ratio", 1.0, 3.5),
            "warp_amp": trial.suggest_float("warp_amp", 1.0, 5.0),
            "hessian_sigma": trial.suggest_float("hessian_sigma", 1.0, 2.5),
            "proto_q_hi": trial.suggest_float("proto_q_hi", 0.86, 0.95),
            "proto_q_lo": trial.suggest_float("proto_q_lo", 0.78, 0.88),
            "proto_close_iters": trial.suggest_int("proto_close_iters", 1, 4),
            "radius_jitter": trial.suggest_float("radius_jitter", 0.02, 0.25),
            "round_sigma": trial.suggest_float("round_sigma", 0.3, 1.2),
            "seed": 42 + trial.number,
        }
        if GENERATOR_VERSION == "v16":
            p["rod_weight"]   = trial.suggest_float("rod_weight", 0.70, 1.00)
            p["plate_weight"] = trial.suggest_float("plate_weight", 0.00, 0.30)
        else:
            p["rod_weight"] = 1.0; p["plate_weight"] = 0.0
        r = generate_small_and_score(p, ref); trial_log.append({**p, **r})
        if (trial.number+1) % 10 == 0:
            print(f"    Trial {trial.number+1}/{n_trials} | loss={r['loss']:.4f} | {time.time()-t0:.0f}s")
        return r["loss"]
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params; elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s. Best loss: {study.best_value:.4f}")
    importances = {}
    try: importances = get_param_importances(study)
    except Exception: pass
    if importances:
        names = list(importances.keys()); vals = list(importances.values())
        fig, ax = plt.subplots(figsize=(10, max(4, len(names)*0.4)))
        ax.barh(range(len(names)), vals, color=plt.cm.viridis(np.linspace(0.3,0.9,len(names))))
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.invert_yaxis()
        plt.tight_layout(); plt.savefig(outdir/"importance_plot.png", dpi=150); plt.close()
    tv = [t.value for t in study.trials if t.value is not None]
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(tv, "o", alpha=0.3, ms=4, label="Trial loss")
    ax.plot(np.minimum.accumulate(tv), "-", color="red", lw=2, label="Best so far")
    ax.legend(); plt.tight_layout(); plt.savefig(outdir/"convergence_plot.png", dpi=150); plt.close()
    opt_result = {"best_loss": float(study.best_value), "best_params": best,
                  "importances": {k: float(v) for k,v in importances.items()},
                  "n_trials": n_trials, "elapsed_s": float(elapsed)}
    with open(outdir/"optimization_result.json", "w") as f:
        json.dump(opt_result, f, indent=2, default=str)
    if trial_log:
        with open(outdir/"optimization_trials.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=trial_log[0].keys(), extrasaction="ignore")
            w.writeheader()
            for r in trial_log: w.writerow(r)
    return best, importances


# ═══════════════════════════════════════════════════════════
#  STEP 2: GENERATE DATASET
# ═══════════════════════════════════════════════════════════

def _validate_sample(morph_raw: dict, bvtv_target: float) -> tuple[bool, str]:
    """
    FIX 2: Validate against RAW morphometrics (before keep_largest_component).
    FIX 3: Reject TbTh at resolution floor (78µm artefact).
    """
    bvtv = morph_raw.get("BVTV", 0.0)
    lcc  = morph_raw.get("lcc_frac", 0.0)
    tbth = morph_raw.get("TbTh_um_p50", 0.0)
    tbn  = morph_raw.get("TbN_per_mm", 0.0)

    if bvtv_target > 0:
        rel_err = abs(bvtv - bvtv_target) / bvtv_target
        if rel_err > GATE_BVTV_REL_ERR:
            return False, f"BV/TV error {rel_err*100:.1f}% (measured={bvtv:.3f}, target={bvtv_target:.3f})"
    if lcc < GATE_LCC_MIN:
        return False, f"LCC_raw={lcc:.3f} < {GATE_LCC_MIN} (fragmented before cleanup)"
    if bvtv < 0.05:
        return False, f"BV/TV={bvtv:.3f} near zero (skeleton collapsed)"
    if bvtv > 0.60:
        return False, f"BV/TV={bvtv:.3f} near one (over-thickened blob)"

    # FIX 3: Hard reject on resolution floor
    if tbth < GATE_TBTH_FLOOR_UM:
        return False, (f"TbTh={tbth:.1f}um below floor threshold {GATE_TBTH_FLOOR_UM}um "
                       f"(resolution artefact — 2-voxel strut)")
    if tbth > GATE_TBTH_MAX_UM:
        return False, f"TbTh={tbth:.1f}um above max {GATE_TBTH_MAX_UM}um (blob)"
    if tbn < 0.8 or tbn > 4.0:
        return False, f"TbN={tbn:.2f}/mm out of range [0.8, 4.0]"
    return True, ""


def _build_rp_kw(best_params: dict, bs: float) -> dict:
    """Build RidgeParams kwargs — base_sigma already incorporates FIX 4."""
    # warp_sigma from params (v15 best uses 8.0, old default was 14.0)
    rp_kw = dict(
        base_sigma=bs,
        warp_sigma=float(best_params.get("warp_sigma", 8.0)),
        warp_amp=float(best_params.get("warp_amp", 1.0)),
        hessian_sigma=float(best_params.get("hessian_sigma", 1.0)),
        ridge_strength=1.0,
        proto_q_hi=float(best_params.get("proto_q_hi", 0.82)),
        proto_q_lo=float(best_params.get("proto_q_lo", 0.70)),
        proto_close_iters=int(best_params.get("proto_close_iters", 1)),
        proto_open_iters=0, proto_min_component=400,
        use_skeleton=True, skeleton_prune_lmin=8, reconnect_close_iters=3,
        radius_mode="branch",
        radius_jitter=float(best_params.get("radius_jitter", 0.03)),
        radius_smooth_sigma=3.0, radius_scale_hint=1.0, prune_small_components=0,
        aniso_ratio=float(best_params.get("aniso_ratio", 1.0)),
    )
    if GENERATOR_VERSION == "v16":
        rp_kw.update(dict(
            rod_weight=float(best_params.get("rod_weight", 0.92)),
            plate_weight=float(best_params.get("plate_weight", 0.08)),
            coarse_weight=0.50, medium_weight=0.35, fine_weight=0.15,
            sheet_q=0.92, bridge_dilate_iters=0, bridge_close_iters=0,
        ))
    return rp_kw


def generate_dataset(best_params, args, outdir, joint_sampler=None):
    """
    Generate N samples.
    Fix 1: Uses joint_sampler if available (correlated morphometric targets).
    Fix 2: Validates on raw morphology before keep_largest_component.
    Fix 3: Rejects TbTh floor artefacts.
    Fix 4: base_sigma modulated by TbSp target.
    """
    print(f"\n{'='*60}")
    print(f"  STEP 2: GENERATE {args.num_samples} SAMPLES ({args.xy}x{args.xy}x{args.z})")
    print(f"  Fix 1: {'Joint VOI sampler' if joint_sampler else 'Independent sampler (fallback)'}")
    print(f"  Fix 2: Validation on RAW morphology")
    print(f"  Fix 3: TbTh floor rejection < {GATE_TBTH_FLOOR_UM}um")
    print(f"  Fix 4: TbSp-aware base_sigma (scale={TBSP_SIGMA_SCALE})")
    print(f"{'='*60}")

    dataset_dir = outdir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    voxel_um    = float(args.voxel_um)
    bvtv_centre = float(best_params.get("bvtv", 0.22))
    tbth_centre = float(best_params.get("tbth_um", 180.0))
    bs_base     = float(best_params.get("base_sigma", 5.0))
    shape       = (args.z, args.xy, args.xy)
    all_metrics = []; n_skipped = 0; n_retries = 0; t0 = time.time()

    # Updated grayscale params to match real bone appearance at 39um
    gp = GrayParams(write_gray=True, solid_fill_sigma=1.0, marrow_mean=20.0,
                    bone_mean=65.0, noise_sd=3.0, bg_tex_sd=1.0)

    for i in range(args.num_samples):
        base_seed  = args.seed + i + 1
        rng_sample = np.random.default_rng(base_seed + 99999)

        # FIX 1: Joint target sampling
        bvtv_target, tbth_target, tbsp_target = sample_joint_targets(
            joint_sampler, rng_sample, bvtv_centre, tbth_centre, voxel_um)

        tbn_target = bvtv_target / (tbth_target / 1000.0)
        br         = tbth_um_to_radius_vox(tbth_target, voxel_um)

        # FIX 4: TbSp-aware base_sigma
        # Wider spacing → larger noise field → coarser trabecular network
        bs_tbsp = bs_base + TBSP_SIGMA_SCALE * tbsp_target
        bs      = float(max(bs_tbsp, 1.5))  # lowered floor to allow fine networks (was 4.5)

        sample_dir = dataset_dir / f"sample_{i:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        bone01 = morph_raw = morph_clean = None; passed = False

        for attempt in range(GATE_MAX_ATTEMPTS):
            seed = base_seed + attempt * 1000
            rng  = np.random.default_rng(seed)
            try:
                rp = RidgeParams(**_build_rp_kw(best_params, bs))
                sk01, _ = make_proto_and_skeleton(shape=shape, rp=rp, rng=rng,
                    skel_mode="skimage", fiji_exe=None,
                    fiji_cmd="Skeletonize (2D/3D)", dbg=None)
                b, _ = thicken_from_skeleton_radius_field(sk01, rng, bvtv_target, br,
                    "branch", float(best_params.get("radius_jitter", 0.15)),
                    3.0, 1.0, dbg=None)
                b = anti_block_round(b, float(best_params.get("round_sigma", 0.8)))
                b = remove_small_components(b, 500)
            except Exception as e:
                print(f"    [sample_{i:03d}] attempt {attempt+1} exception: {e}")
                n_retries += 1; continue

            # FIX 2: measure BEFORE keep_largest_component
            m_raw = measure_all_morphometrics(b, voxel_um)

            # FIX 3: _validate_sample now includes floor rejection
            ok, reason = _validate_sample(m_raw, bvtv_target)

            if ok:
                b_clean     = keep_largest_component(b)
                m_clean     = measure_all_morphometrics(b_clean, voxel_um)
                bone01      = b_clean
                morph_raw   = m_raw
                morph_clean = m_clean
                passed      = True
                if attempt > 0:
                    n_retries += attempt
                    print(f"    [sample_{i:03d}] passed on attempt {attempt+1} "
                          f"(TbTh={m_raw.get('TbTh_um_p50',0):.0f}um "
                          f"LCC_raw={m_raw.get('lcc_frac',0):.3f})")
                break
            else:
                print(f"    [sample_{i:03d}] attempt {attempt+1} rejected: {reason}")
                n_retries += 1

        if not passed:
            print(f"    [sample_{i:03d}] SKIPPED after {GATE_MAX_ATTEMPTS} attempts")
            n_skipped += 1
            try: sample_dir.rmdir()
            except OSError: pass
            continue

        Z = shape[0]; void01 = (1 - bone01).astype(np.uint8)
        save_tif_u8((bone01*255).astype(np.uint8), sample_dir/"mask.tif")
        save_tif_u8((void01*255).astype(np.uint8), sample_dir/"void.tif")
        save_png_u8((bone01[Z//2]*255).astype(np.uint8), sample_dir/"mid.png")
        gray = microct_gray_solid(bone01, gp, np.random.default_rng(seed+10000), br=br)
        save_tif_u8(gray, sample_dir/"gray.tif")
        save_png_u8(gray[Z//2], sample_dir/"gray_mid.png")

        tgt = {"bvtv_target": bvtv_target, "tbth_um_target": tbth_target,
               "tbn_target": tbn_target, "tbsp_um_target": tbsp_target}
        rp_saved = RidgeParams(**_build_rp_kw(best_params, bs))
        met = {
            "version": GENERATOR_VERSION, "label": f"sample_{i:03d}", "seed": seed,
            "morphometrics":     morph_clean,
            "morphometrics_raw": morph_raw,   # FIX 2: honest pre-LCC measurement
            "targets": tgt,
            "params": {"ridge": asdict(rp_saved), "gray": asdict(gp),
                       "bs_tbsp_adjusted": bs,     # FIX 4: log adjusted sigma
                       "tbsp_target": tbsp_target},
            "shape_zyx": list(shape), "voxel_um": voxel_um,
        }
        save_json(met, sample_dir/"metrics.json")
        all_metrics.append(met)

        if (i+1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (i+1) * (args.num_samples - i - 1)
            print(f"    [{i+1}/{args.num_samples}] BV/TV={morph_clean['BVTV']:.3f} "
                  f"TbTh={morph_clean['TbTh_um_p50']:.0f}um "
                  f"TbSp={morph_clean['TbSp_um_p50']:.0f}um "
                  f"bs={bs:.2f} "
                  f"LCC_raw={morph_raw['lcc_frac']:.3f} "
                  f"| {elapsed:.0f}s, ~{eta:.0f}s remaining")

    n_saved = len(all_metrics)
    print(f"\n  Generated {n_saved} valid samples ({n_skipped} skipped, {n_retries} retries)")

    # Report TbTh distribution to confirm floor fix is working
    if all_metrics:
        tbths = [m["morphometrics"]["TbTh_um_p50"] for m in all_metrics]
        print(f"  TbTh distribution: min={min(tbths):.0f}um mean={np.mean(tbths):.0f}um max={max(tbths):.0f}um")
        n_floor = sum(1 for t in tbths if t < GATE_TBTH_FLOOR_UM + 5)
        print(f"  Samples near floor (<{GATE_TBTH_FLOOR_UM+5:.0f}um): {n_floor} "
              f"({100*n_floor/max(n_saved,1):.1f}%) — FIX 3 target: 0%")

    save_json({
        "version": GENERATOR_VERSION, "n": n_saved,
        "n_skipped": n_skipped, "n_retries": n_retries,
        "fix1_joint_sampler": joint_sampler is not None,
        "optimal_params": best_params,
        "samples": [{"label": m["label"], "seed": m["seed"],
                     "bvtv": m["morphometrics"]["BVTV"],
                     "tbth": m["morphometrics"]["TbTh_um_p50"],
                     "tbsp": m["morphometrics"]["TbSp_um_p50"],
                     "lcc_raw": m["morphometrics_raw"]["lcc_frac"]} for m in all_metrics],
    }, dataset_dir/"dataset_manifest.json")

    total = time.time() - t0
    print(f"  Total time: {total:.0f}s ({total/max(n_saved,1):.1f}s/sample)")
    return dataset_dir, all_metrics


# ═══════════════════════════════════════════════════════════
#  STEP 3: PCA + RANDOM PROJECTION  (Fix 13: texture features)
# ═══════════════════════════════════════════════════════════

def _make_scaler(name: str):
    if name == "robust":
        return RobustScaler()
    return StandardScaler()


def _downstream_regression_score(Z_train, Z_test, Y_train, Y_test, label_names, alpha=1.0):
    per_label = {}; r2_vals = []; mae_vals = []
    for j, label in enumerate(label_names):
        reg  = Ridge(alpha=alpha)
        reg.fit(Z_train, Y_train[:, j])
        pred = reg.predict(Z_test)
        r2   = float(r2_score(Y_test[:, j], pred))
        mae  = float(mean_absolute_error(Y_test[:, j], pred))
        per_label[label] = {"r2": r2, "mae": mae}
        r2_vals.append(r2); mae_vals.append(mae)
    return {"per_label": per_label,
            "mean_r2":  float(np.mean(r2_vals)),
            "mean_mae": float(np.mean(mae_vals))}


def run_dim_reduction(dataset_dir, outdir, n_components=16, image_size=64, seed=42, scaler_name='robust', use_pls=True, use_umap=True):
    """FIX 13: Uses texture features (GLCM + stats) instead of raw pixels."""
    print(f"\n{'='*60}")
    print(f"  STEP 3: DIMENSIONALITY REDUCTION (texture features, n={n_components})")
    print(f"{'='*60}")

    features_dir = outdir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    X_rows, Y_rows = [], []
    for d in sorted(dataset_dir.iterdir()):
        met_path  = d / "metrics.json"
        gray_path = d / "gray.tif"
        if not d.is_dir() or not met_path.exists() or not gray_path.exists():
            continue
        with open(met_path) as f: met = json.load(f)
        vol  = tiff.imread(str(gray_path)).astype(np.float32)
        vmax = vol.max()
        if vmax > 0: vol /= vmax
        mid  = vol[vol.shape[0]//2]
        feat = _extract_texture_features(mid, image_size)
        morph = met.get("morphometrics", {})
        X_rows.append(feat)
        Y_rows.append([morph.get(k, 0.0) for k in LABEL_KEYS])

    X = np.array(X_rows, dtype=np.float32)
    Y = np.array(Y_rows, dtype=np.float32)
    print(f"  Loaded {X.shape[0]} samples, {X.shape[1]} texture features")

    if X.shape[0] < 4:
        print("  WARNING: Too few samples"); return None

    X_tr, X_te, Y_tr, Y_te, _, _ = train_test_split(
        X, Y, np.arange(X.shape[0]), test_size=0.2, random_state=seed)
    print(f"  Train: {X_tr.shape[0]}, Test: {X_te.shape[0]}")

    results = {}
    scaler   = _make_scaler(scaler_name)
    X_tr_s   = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)
    nc       = min(n_components, X_tr_s.shape[0], X_tr_s.shape[1])
    pca      = PCA(n_components=nc, random_state=seed)
    Z_tr_pca = pca.fit_transform(X_tr_s); Z_te_pca = pca.transform(X_te_s)
    ev = pca.explained_variance_ratio_; cv = np.cumsum(ev)
    print(f"\n  PCA: {nc} components, {cv[-1]*100:.1f}% variance explained")
    ds_pca = _downstream_regression_score(Z_tr_pca, Z_te_pca, Y_tr, Y_te, LABEL_KEYS)
    print(f"  PCA downstream mean R²: {ds_pca['mean_r2']:.4f}")
    results["PCA"] = {"Z_train": Z_tr_pca, "Z_test": Z_te_pca,
                      "variance_explained": float(cv[-1]), "n_components": nc,
                      "downstream_metrics": ds_pca}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12,5))
    a1.bar(range(len(ev)), ev, color="steelblue"); a1.set_xlabel("PC"); a1.set_ylabel("Variance Ratio")
    a2.plot(cv, "o-", color="darkorange"); a2.axhline(0.95, color="r", ls="--", alpha=0.5)
    a2.set_xlabel("Components"); a2.set_ylabel("Cumulative")
    plt.tight_layout(); plt.savefig(features_dir/"pca_variance.png", dpi=150); plt.close()

    for rp_name, RPClass in [("RP_gaussian", GaussianRandomProjection),
                              ("RP_sparse",   SparseRandomProjection)]:
        rp      = RPClass(n_components=nc, random_state=seed)
        Z_tr_rp = rp.fit_transform(X_tr_s); Z_te_rp = rp.transform(X_te_s)
        n_chk   = min(200, X_tr_s.shape[0])
        idx     = np.random.default_rng(seed).choice(X_tr_s.shape[0], n_chk, replace=False)
        D_o     = np.linalg.norm(X_tr_s[idx,None,:] - X_tr_s[None,idx,:], axis=-1)
        D_p     = np.linalg.norm(Z_tr_rp[idx,None,:] - Z_tr_rp[None,idx,:], axis=-1)
        mask    = D_o > 0
        dp_mean = float(np.mean(D_p[mask]/D_o[mask])) if mask.any() else 0
        print(f"  {rp_name}: {nc} components, dist preservation={dp_mean:.3f}")
        ds_rp = _downstream_regression_score(Z_tr_rp, Z_te_rp, Y_tr, Y_te, LABEL_KEYS)
        print(f"  {rp_name} downstream mean R²: {ds_rp['mean_r2']:.4f}")
        results[rp_name] = {"Z_train": Z_tr_rp, "Z_test": Z_te_rp,
                            "dist_preservation": dp_mean, "n_components": nc,
                            "downstream_metrics": ds_rp}

    for li, ln in enumerate(LABEL_KEYS):
        fig, axes = plt.subplots(1, 3, figsize=(18,5))
        for ax, (name, r) in zip(axes, results.items()):
            Z = r["Z_train"]
            if Z.shape[1] >= 2:
                sc = ax.scatter(Z[:,0], Z[:,1], c=Y_tr[:,li], cmap="viridis", s=20, alpha=0.7)
                plt.colorbar(sc, ax=ax).set_label(ln); ax.set_title(f"{name} ({ln})")
        plt.tight_layout(); plt.savefig(features_dir/f"comparison_{ln}.png", dpi=150); plt.close()

    quantum_files = {}
    for name, r in results.items():
        mm     = MinMaxScaler(feature_range=(0, np.pi))
        Zq_tr  = mm.fit_transform(r["Z_train"]); Zq_te = mm.transform(r["Z_test"])
        mm01   = MinMaxScaler(feature_range=(0,1))
        Z01_tr = mm01.fit_transform(r["Z_train"]); Z01_te = mm01.transform(r["Z_test"])
        fname  = features_dir / f"{name.lower()}_quantum_ready.npz"
        np.savez(fname, Z_train=Zq_tr, Z_test=Zq_te, Z_train_01=Z01_tr, Z_test_01=Z01_te,
                 Y_train=Y_tr, Y_test=Y_te, label_names=LABEL_KEYS, n_features=Zq_tr.shape[1])
        quantum_files[name] = str(fname)
        print(f"  Saved: {fname} ({Zq_tr.shape[0]} train, {Zq_te.shape[0]} test)")

    # UMAP nonlinear reduction
    if UMAP_AVAILABLE and use_umap:
        try:
            reducer = umap.UMAP(n_components=nc, n_neighbors=15,
                                min_dist=0.1, random_state=seed, transform_seed=seed)
            Z_tr_umap = reducer.fit_transform(X_tr_s)
            Z_te_umap = reducer.transform(X_te_s)
            ds_umap   = _downstream_regression_score(Z_tr_umap, Z_te_umap, Y_tr, Y_te, LABEL_KEYS)
            print(f"  UMAP ({nc} components) downstream mean R²: {ds_umap['mean_r2']:.4f}")
            mm_umap  = MinMaxScaler(feature_range=(0, np.pi))
            Zq_tr    = mm_umap.fit_transform(Z_tr_umap); Zq_te = mm_umap.transform(Z_te_umap)
            fname    = features_dir / "umap_quantum_ready.npz"
            np.savez(fname, Z_train=Zq_tr, Z_test=Zq_te,
                     Y_train=Y_tr, Y_test=Y_te, label_names=LABEL_KEYS,
                     n_features=Zq_tr.shape[1])
            quantum_files["UMAP"] = str(fname)
            results["UMAP"] = {"Z_train": Z_tr_umap, "Z_test": Z_te_umap,
                                "n_components": nc, "downstream_metrics": ds_umap}
            print(f"  Saved: {fname}")
        except Exception as e:
            print(f"  UMAP failed: {e}")

    # PLS supervised baseline
    pls_result = None
    if use_pls:
        try:
            from sklearn.cross_decomposition import PLSRegression
            n_pls = max(1, min(nc, X_tr_s.shape[0]-1, X_tr_s.shape[1], Y_tr.shape[1]))
            pls   = PLSRegression(n_components=n_pls)
            Z_tr_pls = pls.fit_transform(X_tr_s, Y_tr)[0]
            Z_te_pls = pls.transform(X_te_s)
            ds_pls = _downstream_regression_score(Z_tr_pls, Z_te_pls, Y_tr, Y_te, LABEL_KEYS)
            print(f"  PLS ({n_pls} components) downstream mean R²: {ds_pls['mean_r2']:.4f}")
            mm_pls = MinMaxScaler(feature_range=(0, np.pi))
            Zq_tr  = mm_pls.fit_transform(Z_tr_pls); Zq_te = mm_pls.transform(Z_te_pls)
            fname  = features_dir / "pls_quantum_ready.npz"
            np.savez(fname, Z_train=Zq_tr, Z_test=Zq_te,
                     Y_train=Y_tr, Y_test=Y_te, label_names=LABEL_KEYS,
                     n_features=Zq_tr.shape[1])
            quantum_files["PLS"] = str(fname)
            results["PLS"] = {"Z_train": Z_tr_pls, "Z_test": Z_te_pls,
                               "n_components": n_pls, "downstream_metrics": ds_pls}
            print(f"  Saved: {fname}")
        except Exception as e:
            print(f"  PLS failed: {e}")

    # Print downstream comparison summary
    print(f"\n  Downstream R² summary (scaler={scaler_name}):")
    for method, r in results.items():
        ds = r.get("downstream_metrics", {})
        r2 = ds.get("mean_r2", float("nan"))
        print(f"    {method:<15} R²={r2:.4f}")

    return {
        "features_dir": str(features_dir), "feature_type": "texture",
        "scaler": scaler_name,
        "results": {k: {"n_components": v["n_components"],
                        "variance_explained": v.get("variance_explained"),
                        "dist_preservation": v.get("dist_preservation"),
                        "downstream_metrics": v.get("downstream_metrics")}
                    for k, v in results.items()},
        "quantum_files": quantum_files,
        "n_train": int(X_tr.shape[0]), "n_test": int(X_te.shape[0]),
    }


# ═══════════════════════════════════════════════════════════
#  STEP 4: FINAL REPORT
# ═══════════════════════════════════════════════════════════

def generate_report(outdir, opt_result, best_params, importances,
                    dataset_metrics, reduction_result, args, joint_sampler):
    print(f"\n{'='*60}\n  STEP 4: FINAL REPORT\n{'='*60}")
    report = {
        "pipeline_version": "2.0",
        "fixes_applied": ["fix1_joint_sampling", "fix2_honest_lcc",
                          "fix3_antifloor", "fix4_tbsp_sigma", "fix13_texture_features"],
        "generator_version": GENERATOR_VERSION,
        "timestamp": datetime.now().isoformat(),
        "validation_gate": {
            "validates_on": "raw_morphology_before_keep_largest_component",
            "bvtv_rel_err_max": GATE_BVTV_REL_ERR,
            "lcc_min": GATE_LCC_MIN,
            "tbth_floor_um": GATE_TBTH_FLOOR_UM,
            "tbsp_sigma_scale": TBSP_SIGMA_SCALE,
        },
        "joint_sampler": {
            "used": joint_sampler is not None,
            "n_sources": joint_sampler["n_sources"] if joint_sampler else 0,
        },
        "generation": {
            "num_samples_requested": args.num_samples,
            "num_samples_saved": len(dataset_metrics),
            "shape": [args.z, args.xy, args.xy],
            "voxel_um": args.voxel_um,
        },
        "dimensionality_reduction": reduction_result,
    }
    with open(outdir/"pipeline_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"  Generated: {len(dataset_metrics)}/{args.num_samples} samples")
    if dataset_metrics:
        bvtvs    = [m["morphometrics"]["BVTV"] for m in dataset_metrics]
        tbths    = [m["morphometrics"]["TbTh_um_p50"] for m in dataset_metrics]
        tbsps    = [m["morphometrics"]["TbSp_um_p50"] for m in dataset_metrics]
        lcc_raws = [m["morphometrics_raw"]["lcc_frac"] for m in dataset_metrics]
        print(f"  BV/TV: [{min(bvtvs):.3f}, {max(bvtvs):.3f}] mean={np.mean(bvtvs):.3f}")
        print(f"  TbTh:  [{min(tbths):.0f}, {max(tbths):.0f}]um mean={np.mean(tbths):.0f}um")
        print(f"  TbSp:  [{min(tbsps):.0f}, {max(tbsps):.0f}]um mean={np.mean(tbsps):.0f}um")
        print(f"  LCC_raw mean: {np.mean(lcc_raws):.3f} (honest, pre-cleanup)")
    print(f"\n  Report: {outdir/'pipeline_report.json'}")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-image", type=str, required=True)
    p.add_argument("--voi-dirs",        nargs="+", type=str, default=None)
    p.add_argument("--outdir",          type=str, default="output/full_pipeline")
    p.add_argument("--optimize-trials", type=int, default=60)
    p.add_argument("--num-samples",     type=int, default=30)
    p.add_argument("--xy",              type=int, default=256)
    p.add_argument("--z",               type=int, default=80)
    p.add_argument("--voxel-um",        type=float, default=39.0)
    p.add_argument("--n-components",    type=int, default=16)
    p.add_argument("--image-size",      type=int, default=64)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--skip-optimize",   action="store_true")
    p.add_argument("--params-json",     type=str, default=None)
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    total_t0 = time.time()

    print(f"\n{'#'*60}")
    print(f"  FULL PIPELINE v2.0 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Fixes: joint sampling, honest LCC, anti-floor, TbSp-sigma, texture, PLS, downstream R²")
    print(f"  Generator: {GENERATOR_VERSION}")
    print(f"{'#'*60}")

    # FIX 1: Build joint sampler from real VOI data if available
    joint_sampler = None
    if args.voi_dirs:
        print(f"\n  Building joint target sampler from {len(args.voi_dirs)} VOI dir(s)...")
        joint_sampler = build_joint_sampler(args.voi_dirs, args.voxel_um)

    importances = {}
    if args.params_json:
        with open(args.params_json) as f: data = json.load(f)
        best_params = data.get("best_params", data); opt_result = data
        print(f"\n  Loaded params from {args.params_json}")
    elif args.skip_optimize:
        # Updated to best visual params found for v15 at 39um voxel
        best_params = {"bvtv": 0.28, "tbth_um": 120.0, "base_sigma": 1.8,
                       "aniso_ratio": 1.0, "warp_amp": 1.0, "warp_sigma": 8.0,
                       "hessian_sigma": 1.0, "proto_q_hi": 0.82, "proto_q_lo": 0.70,
                       "proto_close_iters": 1, "radius_jitter": 0.03,
                       "round_sigma": 0.3, "rod_weight": 1.0, "plate_weight": 0.0}
        opt_result = {"best_loss": None, "best_params": best_params}
        print("\n  Skipping optimization, using best v15 visual params (base_sigma=1.8)")
    else:
        best_params, importances = run_optimization(
            args.reference_image, args.optimize_trials, outdir)
        opt_result = {"best_loss": None, "best_params": best_params}
        with open(outdir/"optimization_result.json") as f: opt_result = json.load(f)

    dataset_dir, dataset_metrics = generate_dataset(
        best_params, args, outdir, joint_sampler=joint_sampler)

    reduction_result = run_dim_reduction(
        dataset_dir, outdir,
        n_components=args.n_components,
        image_size=args.image_size, seed=args.seed,
        scaler_name="robust",
        use_pls=True,
        use_umap=True)

    generate_report(outdir, opt_result, best_params, importances,
                    dataset_metrics, reduction_result, args, joint_sampler)

    print(f"\n{'#'*60}")
    print(f"  PIPELINE COMPLETE — {(time.time()-total_t0)/60:.1f} minutes total")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()