#!/usr/bin/env python3
"""
full_pipeline.py

End-to-end automated pipeline for synthetic trabecular bone ML/QML dataset.

Steps:
  1. OPTIMIZE  — Bayesian optimization finds best generator parameters (Optuna)
  2. GENERATE  — Produce N synthetic volumes with optimal params (v16 generator)
  3. REDUCE    — PCA + Random Projection dimensionality reduction
  4. EXPORT    — Quantum-ready .npz files scaled to [0, pi]
  5. REPORT    — Importance plot, convergence, comparison figures, summary JSON

Usage:
  python full_pipeline.py `
      --reference-image path\\to\\real_voi1_midslice.png `
      --voi-dirs data\\derived\\VOI1 data\\derived\\VOI4 `
      --outdir output\\full_pipeline `
      --optimize-trials 60 `
      --num-samples 30 `
      --xy 256 --z 80 `
      --n-components 16 `
      --voxel-um 39

Overnight run (~6-8 hours):
  python full_pipeline.py `
      --reference-image path\\to\\real_voi1_midslice.png `
      --voi-dirs data\\derived\\VOI1 data\\derived\\VOI4 `
      --outdir output\\full_pipeline `
      --optimize-trials 80 `
      --num-samples 50 `
      --xy 512 --z 160 `
      --n-components 16 `
      --voxel-um 39
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import tifffile as tiff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import optuna
    from optuna.importance import get_param_importances
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("ERROR: pip install optuna"); sys.exit(1)

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

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


LABEL_KEYS = ["BVTV", "TbTh_um_p50", "TbN_per_mm", "TbSp_um_p50"]

# ── Validation gate thresholds ──
GATE_BVTV_REL_ERR  = 0.15   # reject if |measured - target| / target > 15%
GATE_LCC_MIN       = 0.85   # reject if largest connected component < 85% of bone
GATE_MAX_ATTEMPTS  = 8      # resample up to 8 times before skipping


# ═══════════════════════════════════════════════════════════
#  STEP 1: OPTIMIZE
# ═══════════════════════════════════════════════════════════

def compute_ssim_2d(img1, img2):
    C1, C2 = 0.01**2, 0.03**2
    if img1.shape != img2.shape:
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = np.array(Image.fromarray((img1*255).astype(np.uint8)).resize((w,h)))/255.0
        img2 = np.array(Image.fromarray((img2*255).astype(np.uint8)).resize((w,h)))/255.0
    mu1, mu2 = img1.mean(), img2.mean()
    s1, s2 = img1.var(), img2.var()
    s12 = ((img1-mu1)*(img2-mu2)).mean()
    return float((2*mu1*mu2+C1)*(2*s12+C2) / ((mu1**2+mu2**2+C1)*(s1+s2+C2)))


def load_reference(path, size=128):
    img = np.array(Image.open(path).convert("L")).astype(np.float32)/255.0
    return np.array(Image.fromarray((img*255).astype(np.uint8)).resize((size,size), Image.BILINEAR))/255.0


def generate_small_and_score(params, ref_image, voxel_um=39.0, shape=(40,128,128)):
    """Generate one small volume, return combined loss."""
    seed = params.get("seed", 42)
    rng = np.random.default_rng(seed)
    bvtv = params["bvtv"]
    tbth = params["tbth_um"]
    tbn = bvtv / (tbth / 1000.0)
    br = tbth_um_to_radius_vox(tbth, voxel_um)

    rp_kw = dict(
        base_sigma=max(params["base_sigma"], 4.5), warp_sigma=14.0, warp_amp=params["warp_amp"],
        hessian_sigma=params["hessian_sigma"], ridge_strength=1.0,
        proto_q_hi=params["proto_q_hi"], proto_q_lo=params["proto_q_lo"],
        proto_close_iters=params["proto_close_iters"], proto_open_iters=0,
        proto_min_component=250, use_skeleton=True, skeleton_prune_lmin=6,
        reconnect_close_iters=0, radius_mode="branch", radius_jitter=params["radius_jitter"],
        radius_smooth_sigma=3.0, radius_scale_hint=1.0, prune_small_components=0,
        aniso_ratio=params["aniso_ratio"],
    )
    if GENERATOR_VERSION == "v16":
        rp_kw.update(dict(rod_weight=params["rod_weight"], plate_weight=params["plate_weight"],
            coarse_weight=0.50, medium_weight=0.35, fine_weight=0.15,
            sheet_q=0.92, bridge_dilate_iters=0, bridge_close_iters=0))

    rp = RidgeParams(**rp_kw)
    gp = GrayParams(write_gray=True, solid_fill_sigma=3.0, marrow_mean=15.0, bone_mean=240.0, noise_sd=3.0, bg_tex_sd=1.0)

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

    morph = measure_all_morphometrics(bone01, voxel_um)
    targets = {"BVTV": bvtv, "TbTh_um_p50": tbth, "TbN_per_mm": tbn}
    morph_errs = {k: abs(morph.get(k,0)-t)/t if t>0 else 1.0 for k,t in targets.items()}
    avg_morph = float(np.mean(list(morph_errs.values())))

    gray = microct_gray_solid(bone01, gp, np.random.default_rng(seed+1000), br=br)
    mid = gray[shape[0]//2].astype(np.float32)/255.0
    mid_r = np.array(Image.fromarray((mid*255).astype(np.uint8)).resize(
        (ref_image.shape[1], ref_image.shape[0]), Image.BILINEAR))/255.0
    ssim = compute_ssim_2d(ref_image, mid_r)

    lcc = morph.get("lcc_frac", 0.0)
    conn_pen = 0.0 if lcc >= 0.8 else 0.5*(0.8-lcc)
    loss = 0.4*avg_morph + 0.6*(1.0-ssim) + conn_pen

    return {"loss": float(loss), "ssim": float(ssim), "morph_error": float(avg_morph),
            "bvtv_measured": float(morph["BVTV"]), "lcc_frac": float(lcc)}


def run_optimization(ref_path, n_trials, outdir):
    """Step 1: Find optimal generator parameters."""
    print(f"\n{'='*60}")
    print(f"  STEP 1: PARAMETER OPTIMIZATION ({n_trials} trials)")
    print(f"{'='*60}")

    ref = load_reference(ref_path, 128)
    trial_log = []
    t0 = time.time()

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
            p["rod_weight"] = trial.suggest_float("rod_weight", 0.70, 1.00)
            p["plate_weight"] = trial.suggest_float("plate_weight", 0.00, 0.30)
        else:
            p["rod_weight"] = 1.0; p["plate_weight"] = 0.0

        r = generate_small_and_score(p, ref)
        trial_log.append({**p, **r})
        n = trial.number + 1
        if n % 10 == 0:
            print(f"    Trial {n}/{n_trials} | loss={r['loss']:.4f} ssim={r.get('ssim',0):.3f} | {time.time()-t0:.0f}s")
        return r["loss"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    elapsed = time.time() - t0
    print(f"\n  Optimization done in {elapsed:.0f}s ({elapsed/n_trials:.1f}s/trial)")
    print(f"  Best loss: {study.best_value:.4f}")

    importances = {}
    try:
        importances = get_param_importances(study)
        print(f"\n  Parameter importance:")
        for k, v in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"    {k:<22} {v:.4f} {'#'*int(v*40)}")
    except Exception:
        pass

    if importances:
        names = list(importances.keys())
        vals = list(importances.values())
        fig, ax = plt.subplots(figsize=(10, max(4, len(names)*0.4)))
        ax.barh(range(len(names)), vals, color=plt.cm.viridis(np.linspace(0.3,0.9,len(names))))
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        ax.set_xlabel("Importance"); ax.set_title(f"Parameter Importance ({n_trials} trials)")
        ax.invert_yaxis()
        for i,v in enumerate(vals): ax.text(v+0.005, i, f"{v:.3f}", va="center", fontsize=9)
        plt.tight_layout(); plt.savefig(outdir/"importance_plot.png", dpi=150); plt.close()

    tv = [t.value for t in study.trials if t.value is not None]
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(tv, "o", alpha=0.3, ms=4, label="Trial loss")
    ax.plot(np.minimum.accumulate(tv), "-", color="red", lw=2, label="Best so far")
    ax.set_xlabel("Trial"); ax.set_ylabel("Loss"); ax.set_title("Optimization Convergence")
    ax.legend(); plt.tight_layout(); plt.savefig(outdir/"convergence_plot.png", dpi=150); plt.close()

    opt_result = {
        "best_loss": float(study.best_value), "best_params": best,
        "importances": {k: float(v) for k,v in importances.items()} if importances else {},
        "n_trials": n_trials, "elapsed_s": float(elapsed),
    }
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

def _validate_sample(bone01: np.ndarray, morph: dict, bvtv_target: float) -> tuple[bool, str]:
    """
    Validation gate: returns (passed, reason_if_failed).
    Checks BV/TV error, LCC fraction, Tb.Th and Tb.N plausibility.
    Called after every generation attempt.
    """
    bvtv_measured = morph.get("BVTV", 0.0)
    lcc_frac      = morph.get("lcc_frac", 0.0)
    tbth          = morph.get("TbTh_um_p50", 0.0)
    tbn           = morph.get("TbN_per_mm", 0.0)

    # Check 1: BV/TV must be within 15% of target
    if bvtv_target > 0:
        rel_err = abs(bvtv_measured - bvtv_target) / bvtv_target
        if rel_err > GATE_BVTV_REL_ERR:
            return False, (f"BV/TV error {rel_err*100:.1f}% > {GATE_BVTV_REL_ERR*100:.0f}% "
                           f"(measured={bvtv_measured:.3f}, target={bvtv_target:.3f})")

    # Check 2: largest connected component must dominate
    if lcc_frac < GATE_LCC_MIN:
        return False, f"LCC={lcc_frac:.3f} < {GATE_LCC_MIN} (fragmented network)"

    # Check 3: volume must not be nearly empty or nearly full
    if bvtv_measured < 0.05:
        return False, f"BV/TV={bvtv_measured:.3f} near zero (skeleton collapsed)"
    if bvtv_measured > 0.60:
        return False, f"BV/TV={bvtv_measured:.3f} near one (over-thickened blob)"

    # Check 4: Tb.Th must be in a physically plausible range
    # floor ~78um at 39um voxel size (2 voxels minimum), ceiling 350um
    if tbth < 78.0 or tbth > 350.0:
        return False, f"TbTh={tbth:.1f}um out of range [78, 350] (resolution floor or blob)"

    # Check 5: Tb.N must be in a plausible range
    # catches sparse skeletons (<0.8) and over-thickened blobs (>4.0)
    if tbn < 0.8 or tbn > 4.0:
        return False, f"TbN={tbn:.2f}/mm out of range [0.8, 4.0] (sparse or merged network)"

    return True, ""


def _build_rp_kw(best_params: dict, bs: float) -> dict:
    """Shared RidgeParams kwargs builder."""
    rp_kw = dict(
        base_sigma=bs, warp_sigma=14.0,
        warp_amp=float(best_params.get("warp_amp", 3.0)),
        hessian_sigma=float(best_params.get("hessian_sigma", 1.4)),
        ridge_strength=1.0,
        proto_q_hi=float(best_params.get("proto_q_hi", 0.92)),
        proto_q_lo=float(best_params.get("proto_q_lo", 0.84)),
        proto_close_iters=int(best_params.get("proto_close_iters", 2)),
        proto_open_iters=0, proto_min_component=400,
        use_skeleton=True, skeleton_prune_lmin=8, reconnect_close_iters=3,
        radius_mode="branch",
        radius_jitter=float(best_params.get("radius_jitter", 0.15)),
        radius_smooth_sigma=3.0, radius_scale_hint=1.0, prune_small_components=0,
        aniso_ratio=float(best_params.get("aniso_ratio", 3.0)),
    )
    if GENERATOR_VERSION == "v16":
        rp_kw.update(dict(
            rod_weight=float(best_params.get("rod_weight", 0.92)),
            plate_weight=float(best_params.get("plate_weight", 0.08)),
            coarse_weight=0.50, medium_weight=0.35, fine_weight=0.15,
            sheet_q=0.92, bridge_dilate_iters=0, bridge_close_iters=0,
        ))
    return rp_kw


def generate_dataset(best_params, args, outdir):
    """Step 2: Generate N samples with optimal parameters + validation gate."""
    print(f"\n{'='*60}")
    print(f"  STEP 2: GENERATE {args.num_samples} SAMPLES ({args.xy}x{args.xy}x{args.z})")
    print(f"  Validation gate: BV/TV error <{GATE_BVTV_REL_ERR*100:.0f}%, "
          f"LCC >{GATE_LCC_MIN}, max {GATE_MAX_ATTEMPTS} attempts/sample")
    print(f"{'='*60}")

    dataset_dir = outdir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    voxel_um     = float(args.voxel_um)
    bvtv_centre  = float(best_params.get("bvtv", 0.22))
    tbth_centre  = float(best_params.get("tbth_um", 180.0))
    bs_base      = float(best_params.get("base_sigma", 5.0))
    shape        = (args.z, args.xy, args.xy)

    all_metrics  = []
    n_skipped    = 0
    n_retries    = 0
    t0           = time.time()

    gp = GrayParams(write_gray=True, solid_fill_sigma=3.0, marrow_mean=15.0,
                    bone_mean=240.0, noise_sd=3.0, bg_tex_sd=1.0)

    for i in range(args.num_samples):
        base_seed   = args.seed + i + 1
        rng_sample  = np.random.default_rng(base_seed + 99999)
        bvtv_target = float(np.clip(rng_sample.normal(bvtv_centre, 0.06), 0.12, 0.32))
        tbth_target = float(np.clip(rng_sample.normal(tbth_centre, 30.0), 110.0, 260.0))
        tbn_target  = bvtv_target / (tbth_target / 1000.0)
        br          = tbth_um_to_radius_vox(tbth_target, voxel_um)
        bs          = max(bs_base, 4.5)

        sample_dir  = dataset_dir / f"sample_{i:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        bone01 = morph = None
        passed = False

        # ── Validation gate: retry loop ──
        for attempt in range(GATE_MAX_ATTEMPTS):
            seed = base_seed + attempt * 1000   # different seed each attempt
            rng  = np.random.default_rng(seed)

            try:
                rp    = RidgeParams(**_build_rp_kw(best_params, bs))
                sk01, _ = make_proto_and_skeleton(
                    shape=shape, rp=rp, rng=rng,
                    skel_mode="skimage", fiji_exe=None,
                    fiji_cmd="Skeletonize (2D/3D)", dbg=None)
                b, _ = thicken_from_skeleton_radius_field(
                    sk01, rng, bvtv_target, br, "branch",
                    float(best_params.get("radius_jitter", 0.15)), 3.0, 1.0, dbg=None)
                b = anti_block_round(b, float(best_params.get("round_sigma", 0.8)))
                b = remove_small_components(b, 500)
                b = keep_largest_component(b)
            except Exception as e:
                print(f"    [sample_{i:03d}] attempt {attempt+1} exception: {e}")
                n_retries += 1
                continue

            m = measure_all_morphometrics(b, voxel_um)
            ok, reason = _validate_sample(b, m, bvtv_target)

            if ok:
                bone01 = b
                morph  = m
                passed = True
                if attempt > 0:
                    n_retries += attempt
                    print(f"    [sample_{i:03d}] passed on attempt {attempt+1}")
                break
            else:
                print(f"    [sample_{i:03d}] attempt {attempt+1} rejected: {reason}")
                n_retries += 1

        if not passed:
            print(f"    [sample_{i:03d}] SKIPPED after {GATE_MAX_ATTEMPTS} attempts")
            n_skipped += 1
            # Remove empty dir so it won't confuse the PCA loader
            try:
                sample_dir.rmdir()
            except OSError:
                pass
            continue

        # ── Save accepted sample ──
        Z = shape[0]
        void01 = (1 - bone01).astype(np.uint8)
        save_tif_u8((bone01 * 255).astype(np.uint8), sample_dir / "mask.tif")
        save_tif_u8((void01 * 255).astype(np.uint8), sample_dir / "void.tif")
        save_png_u8((bone01[Z // 2] * 255).astype(np.uint8), sample_dir / "mid.png")

        gray = microct_gray_solid(bone01, gp, np.random.default_rng(seed + 10000), br=br)
        save_tif_u8(gray, sample_dir / "gray.tif")
        save_png_u8(gray[Z // 2], sample_dir / "gray_mid.png")

        tgt = {"bvtv_target": bvtv_target, "tbth_um_target": tbth_target,
               "tbn_target": tbn_target, "tbsp_um_target": 0}
        rp_saved = RidgeParams(**_build_rp_kw(best_params, bs))
        met = {
            "version": GENERATOR_VERSION, "label": f"sample_{i:03d}", "seed": seed,
            "morphometrics": morph, "targets": tgt,
            "params": {"ridge": asdict(rp_saved), "gray": asdict(gp)},
            "shape_zyx": list(shape), "voxel_um": voxel_um,
        }
        save_json(met, sample_dir / "metrics.json")
        all_metrics.append(met)

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (args.num_samples - i - 1)
            print(f"    [{i+1}/{args.num_samples}] BV/TV={morph['BVTV']:.3f} "
                  f"LCC={morph['lcc_frac']:.3f} | {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    # ── Summary ──
    n_saved = len(all_metrics)
    print(f"\n  Generated {n_saved} valid samples "
          f"({n_skipped} skipped, {n_retries} total retries)")

    save_json({
        "version": GENERATOR_VERSION, "n": n_saved,
        "n_skipped": n_skipped, "n_retries": n_retries,
        "optimal_params": best_params,
        "samples": [{"label": m["label"], "seed": m["seed"],
                     "bvtv": m["morphometrics"]["BVTV"]} for m in all_metrics],
    }, dataset_dir / "dataset_manifest.json")

    total = time.time() - t0
    print(f"  Total time: {total:.0f}s ({total/max(n_saved,1):.1f}s/sample)")
    return dataset_dir, all_metrics


# ═══════════════════════════════════════════════════════════
#  STEP 3: PCA + RANDOM PROJECTION
# ═══════════════════════════════════════════════════════════

def run_dim_reduction(dataset_dir, outdir, n_components=16, image_size=64, seed=42):
    """Step 3: PCA and Random Projection on generated images."""
    print(f"\n{'='*60}")
    print(f"  STEP 3: DIMENSIONALITY REDUCTION (n_components={n_components})")
    print(f"{'='*60}")

    features_dir = outdir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    X_rows, Y_rows, infos = [], [], []
    for d in sorted(dataset_dir.iterdir()):
        met_path  = d / "metrics.json"
        gray_path = d / "gray.tif"
        if not d.is_dir() or not met_path.exists() or not gray_path.exists():
            continue
        with open(met_path) as f:
            met = json.load(f)

        vol  = tiff.imread(str(gray_path)).astype(np.float32)
        vmax = vol.max()
        if vmax > 0: vol /= vmax
        mid  = vol[vol.shape[0] // 2]

        img  = Image.fromarray((mid * 255).astype(np.uint8), mode="L")
        img  = img.resize((image_size, image_size), Image.BILINEAR)
        flat = np.array(img, dtype=np.float32) / 255.0

        X_rows.append(flat.ravel())
        morph = met.get("morphometrics", {})
        Y_rows.append([morph.get(k, 0.0) for k in LABEL_KEYS])
        infos.append({"sample": d.name})

    X = np.array(X_rows, dtype=np.float32)
    Y = np.array(Y_rows, dtype=np.float32)
    print(f"  Loaded {X.shape[0]} images, {X.shape[1]} raw features")

    if X.shape[0] < 4:
        print("  WARNING: Too few samples for meaningful reduction")
        return None

    X_tr, X_te, Y_tr, Y_te, idx_tr, idx_te = train_test_split(
        X, Y, np.arange(X.shape[0]), test_size=0.2, random_state=seed)
    print(f"  Train: {X_tr.shape[0]}, Test: {X_te.shape[0]}")

    results = {}

    # PCA
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)
    nc       = min(n_components, X_tr_s.shape[0], X_tr_s.shape[1])

    pca      = PCA(n_components=nc, random_state=seed)
    Z_tr_pca = pca.fit_transform(X_tr_s)
    Z_te_pca = pca.transform(X_te_s)
    ev       = pca.explained_variance_ratio_
    cv       = np.cumsum(ev)
    print(f"\n  PCA: {nc} components, {cv[-1]*100:.1f}% variance explained")

    results["PCA"] = {"Z_train": Z_tr_pca, "Z_test": Z_te_pca,
                      "variance_explained": float(cv[-1]), "n_components": nc}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.bar(range(len(ev)), ev, color="steelblue")
    a1.set_xlabel("PC"); a1.set_ylabel("Variance Ratio")
    a2.plot(cv, "o-", color="darkorange")
    a2.axhline(0.95, color="r", ls="--", alpha=0.5)
    a2.set_xlabel("Components"); a2.set_ylabel("Cumulative")
    a2.set_title("PCA Cumulative Variance")
    plt.tight_layout(); plt.savefig(features_dir / "pca_variance.png", dpi=150); plt.close()

    # Random Projection
    for rp_name, RPClass in [("RP_gaussian", GaussianRandomProjection),
                              ("RP_sparse",   SparseRandomProjection)]:
        rp       = RPClass(n_components=nc, random_state=seed)
        Z_tr_rp  = rp.fit_transform(X_tr_s)
        Z_te_rp  = rp.transform(X_te_s)

        n_chk    = min(200, X_tr_s.shape[0])
        idx      = np.random.default_rng(seed).choice(X_tr_s.shape[0], n_chk, replace=False)
        D_o      = np.linalg.norm(X_tr_s[idx, None, :] - X_tr_s[None, idx, :], axis=-1)
        D_p      = np.linalg.norm(Z_tr_rp[idx, None, :] - Z_tr_rp[None, idx, :], axis=-1)
        mask     = D_o > 0
        dp_mean  = float(np.mean(D_p[mask] / D_o[mask])) if mask.any() else 0

        print(f"  {rp_name}: {nc} components, dist preservation={dp_mean:.3f}")
        results[rp_name] = {"Z_train": Z_tr_rp, "Z_test": Z_te_rp,
                            "dist_preservation": dp_mean, "n_components": nc}

    # Scatter comparison
    for li, ln in enumerate(LABEL_KEYS):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (name, r) in zip(axes, results.items()):
            Z  = r["Z_train"]
            if Z.shape[1] >= 2:
                sc = ax.scatter(Z[:, 0], Z[:, 1], c=Y_tr[:, li],
                                cmap="viridis", s=20, alpha=0.7)
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label(ln)
                ax.set_title(f"{name} ({ln})")
        plt.tight_layout()
        plt.savefig(features_dir / f"comparison_{ln}.png", dpi=150)
        plt.close()

    # Quantum export
    quantum_files = {}
    for name, r in results.items():
        mm      = MinMaxScaler(feature_range=(0, np.pi))
        Zq_tr   = mm.fit_transform(r["Z_train"])
        Zq_te   = mm.transform(r["Z_test"])

        mm01    = MinMaxScaler(feature_range=(0, 1))
        Z01_tr  = mm01.fit_transform(r["Z_train"])
        Z01_te  = mm01.transform(r["Z_test"])

        fname   = features_dir / f"{name.lower()}_quantum_ready.npz"
        np.savez(fname, Z_train=Zq_tr, Z_test=Zq_te,
                 Z_train_01=Z01_tr, Z_test_01=Z01_te,
                 Y_train=Y_tr, Y_test=Y_te,
                 label_names=LABEL_KEYS, n_features=Zq_tr.shape[1])
        quantum_files[name] = str(fname)
        print(f"  Saved: {fname} ({Zq_tr.shape[0]} train, {Zq_te.shape[0]} test, {Zq_tr.shape[1]}D)")

    # Correlation heatmaps
    for name, r in results.items():
        Z   = r["Z_train"]; nc2 = Z.shape[1]; nl = Y_tr.shape[1]
        corr = np.zeros((nc2, nl))
        for ii in range(nc2):
            for jj in range(nl):
                if np.std(Z[:, ii]) > 1e-8 and np.std(Y_tr[:, jj]) > 1e-8:
                    corr[ii, jj] = np.corrcoef(Z[:, ii], Y_tr[:, jj])[0, 1]

        fig, ax = plt.subplots(figsize=(8, max(4, nc2 * 0.3)))
        im = ax.imshow(corr[:min(nc2, 20), :], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_yticks(range(min(nc2, 20)))
        ax.set_yticklabels([f"C{ii}" for ii in range(min(nc2, 20))])
        ax.set_xticks(range(len(LABEL_KEYS)))
        ax.set_xticklabels(LABEL_KEYS, rotation=45, ha="right")
        ax.set_title(f"{name}: Component-Label Correlations")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(features_dir / f"{name.lower()}_correlation.png", dpi=150)
        plt.close()

    return {
        "features_dir": str(features_dir),
        "results": {k: {"n_components": v["n_components"],
                        "variance_explained": v.get("variance_explained"),
                        "dist_preservation": v.get("dist_preservation")}
                    for k, v in results.items()},
        "quantum_files": quantum_files,
        "n_train": int(X_tr.shape[0]),
        "n_test":  int(X_te.shape[0]),
    }


# ═══════════════════════════════════════════════════════════
#  STEP 4: FINAL REPORT
# ═══════════════════════════════════════════════════════════

def generate_report(outdir, opt_result, best_params, importances, dataset_metrics, reduction_result, args):
    print(f"\n{'='*60}")
    print(f"  STEP 4: FINAL REPORT")
    print(f"{'='*60}")

    report = {
        "pipeline_version": "1.1",
        "generator_version": GENERATOR_VERSION,
        "timestamp": datetime.now().isoformat(),
        "reference_image": args.reference_image,
        "voi_dirs": args.voi_dirs,
        "validation_gate": {
            "bvtv_rel_err_max": GATE_BVTV_REL_ERR,
            "lcc_min": GATE_LCC_MIN,
            "max_attempts": GATE_MAX_ATTEMPTS,
        },
        "optimization": {
            "n_trials": args.optimize_trials,
            "best_loss": opt_result.get("best_loss"),
            "best_params": best_params,
            "importances": {k: float(v) for k,v in importances.items()} if importances else {},
            "top_3_params": [k for k,v in sorted(importances.items(), key=lambda x:-x[1])[:3]] if importances else [],
        },
        "generation": {
            "num_samples_requested": args.num_samples,
            "num_samples_saved": len(dataset_metrics),
            "shape": [args.z, args.xy, args.xy],
            "voxel_um": args.voxel_um,
            "avg_bvtv": float(np.mean([m["morphometrics"]["BVTV"] for m in dataset_metrics])) if dataset_metrics else None,
        },
        "dimensionality_reduction": reduction_result,
        "quantum_ready": {
            "n_components": args.n_components,
            "encoding": "angle [0, pi]",
            "n_qubits_needed": args.n_components,
        },
        "paper_figures": [
            "importance_plot.png",
            "convergence_plot.png",
            "pca_variance.png",
        ] + [f"comparison_{ln}.png" for ln in LABEL_KEYS],
    }

    with open(outdir / "pipeline_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Optimization: {args.optimize_trials} trials, best loss={opt_result.get('best_loss') or '?'}")
    if importances:
        top3 = [k for k,v in sorted(importances.items(), key=lambda x:-x[1])[:3]]
        print(f"  Top 3 important params: {', '.join(top3)}")
    print(f"  Generated: {len(dataset_metrics)}/{args.num_samples} samples at {args.xy}x{args.xy}x{args.z}")
    if dataset_metrics:
        bvtvs = [m["morphometrics"]["BVTV"] for m in dataset_metrics]
        print(f"  BV/TV range: [{min(bvtvs):.3f}, {max(bvtvs):.3f}]")
    if reduction_result:
        for k, v in reduction_result.get("results", {}).items():
            ve = v.get("variance_explained")
            dp = v.get("dist_preservation")
            if ve: print(f"  {k}: {ve*100:.1f}% variance explained")
            if dp: print(f"  {k}: {dp:.3f} distance preservation")

    print(f"\n  Full report: {outdir / 'pipeline_report.json'}")
    print(f"  All outputs in: {outdir}/")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Full pipeline: optimize -> generate -> reduce -> quantum export")
    p.add_argument("--reference-image", type=str, required=True)
    p.add_argument("--voi-dirs", nargs="+", type=str, default=None)
    p.add_argument("--outdir", type=str, default="output/full_pipeline")
    p.add_argument("--optimize-trials", type=int, default=60)
    p.add_argument("--num-samples", type=int, default=30)
    p.add_argument("--xy", type=int, default=256)
    p.add_argument("--z", type=int, default=80)
    p.add_argument("--voxel-um", type=float, default=39.0)
    p.add_argument("--n-components", type=int, default=16)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-optimize", action="store_true")
    p.add_argument("--params-json", type=str, default=None)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    total_t0 = time.time()
    print(f"\n{'#'*60}")
    print(f"  FULL PIPELINE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Generator: {GENERATOR_VERSION}")
    print(f"  Reference: {args.reference_image}")
    print(f"  Output: {outdir}")
    print(f"{'#'*60}")

    importances = {}
    if args.params_json:
        with open(args.params_json) as f:
            data = json.load(f)
        best_params = data.get("best_params", data)
        opt_result  = data
        print(f"\n  Loaded pre-computed params from {args.params_json}")
    elif args.skip_optimize:
        best_params = {"bvtv": 0.22, "tbth_um": 180.0, "base_sigma": 5.0,
                       "aniso_ratio": 3.0, "warp_amp": 3.0, "hessian_sigma": 1.4,
                       "proto_q_hi": 0.92, "proto_q_lo": 0.84, "proto_close_iters": 2,
                       "radius_jitter": 0.15, "round_sigma": 0.7,
                       "rod_weight": 1.0, "plate_weight": 0.0}
        opt_result  = {"best_loss": None, "best_params": best_params}
        print(f"\n  Skipping optimization, using v15 proven defaults")
    else:
        best_params, importances = run_optimization(args.reference_image, args.optimize_trials, outdir)
        opt_result = {"best_loss": None, "best_params": best_params}
        with open(outdir / "optimization_result.json") as f:
            opt_result = json.load(f)

    dataset_dir, dataset_metrics = generate_dataset(best_params, args, outdir)

    reduction_result = run_dim_reduction(
        dataset_dir, outdir,
        n_components=args.n_components,
        image_size=args.image_size,
        seed=args.seed,
    )

    generate_report(outdir, opt_result, best_params, importances,
                    dataset_metrics, reduction_result, args)

    total_elapsed = time.time() - total_t0
    print(f"\n{'#'*60}")
    print(f"  PIPELINE COMPLETE — {total_elapsed/60:.1f} minutes total")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()