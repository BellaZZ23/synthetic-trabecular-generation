#!/usr/bin/env python3
"""
parameter_importance_analysis.py

Runs a quick Optuna study to determine which generator parameters
have the most impact on structural similarity + morphometric accuracy.

Requires: pip install optuna
Uses: synthetic_trabecular_v16_morphometric_control.py (must be in same dir)

Usage:
  python parameter_importance_analysis.py `
      --reference-image path\to\real_voi1_midslice.png `
      --voi-dirs data\derived\VOI1 data\derived\VOI4 `
      --n-trials 60 `
      --outdir output\importance_analysis

Output:
  - importance_plot.png (which parameters matter most)
  - importance_scores.json (numerical importance values)
  - trial_results.csv (all trials with params + scores)
  - best_params.json (optimal parameter set found)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import optuna
    from optuna.importance import get_param_importances
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("ERROR: pip install optuna")
    sys.exit(1)

# Import the v16 generator functions
try:
    from synthetic_trabecular_v16_morphometric_control import (
        RidgeParams, GrayParams,
        make_proto_and_skeleton, thicken_from_skeleton_radius_field,
        anti_block_round, remove_small_components, keep_largest_component,
        microct_gray_solid, measure_all_morphometrics,
        tbth_um_to_radius_vox, tbn_per_mm_to_base_sigma,
        normalize, skeleton_graph_stats, component_stats_3d,
    )
    HAS_V16 = True
except ImportError:
    print("WARNING: v16 generator not found, trying v15...")
    try:
        from synthetic_trabecular_v15_morphometric_control import (
            RidgeParams, GrayParams,
            make_proto_and_skeleton, thicken_from_skeleton_radius_field,
            anti_block_round, remove_small_components, keep_largest_component,
            microct_gray_solid, measure_all_morphometrics,
            tbth_um_to_radius_vox, tbn_per_mm_to_base_sigma,
            normalize, skeleton_graph_stats,
        )
        HAS_V16 = False
    except ImportError:
        print("ERROR: Neither v16 nor v15 generator found in current directory")
        sys.exit(1)


def compute_ssim_2d(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simple SSIM between two 2D grayscale images [0,1]."""
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    # Resize to match if needed
    if img1.shape != img2.shape:
        from PIL import Image as PILImage
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = np.array(PILImage.fromarray((img1 * 255).astype(np.uint8)).resize((w, h))) / 255.0
        img2 = np.array(PILImage.fromarray((img2 * 255).astype(np.uint8)).resize((w, h))) / 255.0

    mu1 = img1.mean()
    mu2 = img2.mean()
    sig1_sq = img1.var()
    sig2_sq = img2.var()
    sig12 = ((img1 - mu1) * (img2 - mu2)).mean()

    num = (2 * mu1 * mu2 + C1) * (2 * sig12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sig1_sq + sig2_sq + C2)
    return float(num / den)


def load_reference(path: str, target_size: int = 128) -> np.ndarray:
    """Load and resize reference image to [0,1]."""
    img = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
    img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize(
        (target_size, target_size), Image.BILINEAR)) / 255.0
    return img


def generate_and_score(
    params: dict,
    ref_image: np.ndarray,
    voxel_um: float = 39.0,
    shape: tuple = (40, 128, 128),
    morph_weight: float = 0.4,
    ssim_weight: float = 0.6,
) -> dict:
    """Generate one small volume with given params, compute combined loss."""

    seed = params.get("seed", 42)
    rng = np.random.default_rng(seed)

    bvtv = params["bvtv"]
    tbth = params["tbth_um"]
    tbn = bvtv / (tbth / 1000.0)
    br = tbth_um_to_radius_vox(tbth, voxel_um)
    bs = params["base_sigma"]

    # Build RidgeParams
    rp_kwargs = dict(
        base_sigma=bs,
        warp_sigma=14.0,
        warp_amp=params["warp_amp"],
        hessian_sigma=params["hessian_sigma"],
        ridge_strength=1.0,
        proto_q_hi=params["proto_q_hi"],
        proto_q_lo=params["proto_q_lo"],
        proto_close_iters=params["proto_close_iters"],
        proto_open_iters=0,
        proto_min_component=250,
        use_skeleton=True,
        skeleton_prune_lmin=6,
        reconnect_close_iters=0,
        radius_mode="branch",
        radius_jitter=params["radius_jitter"],
        radius_smooth_sigma=3.0,
        radius_scale_hint=1.0,
        prune_small_components=0,
        aniso_ratio=params["aniso_ratio"],
    )

    # v16 has extra params
    if HAS_V16:
        rp_kwargs.update(dict(
            rod_weight=params["rod_weight"],
            plate_weight=params["plate_weight"],
            coarse_weight=0.50,
            medium_weight=0.35,
            fine_weight=0.15,
            sheet_q=0.92,
            bridge_dilate_iters=0,
            bridge_close_iters=0,
        ))

    rp = RidgeParams(**rp_kwargs)
    gp = GrayParams(
        write_gray=True,
        solid_fill_sigma=3.0,
        marrow_mean=15.0,
        bone_mean=240.0,
        noise_sd=3.0,
        bg_tex_sd=1.0,
    )

    # Generate
    try:
        sk01, si = make_proto_and_skeleton(
            shape=shape, rp=rp, rng=rng,
            skel_mode="skimage", fiji_exe=None,
            fiji_cmd="Skeletonize (2D/3D)", dbg=None,
        )

        bone01, ti = thicken_from_skeleton_radius_field(
            sk01, rng, bvtv, br, "branch",
            params["radius_jitter"], 3.0, 1.0, dbg=None,
        )

        bone01 = anti_block_round(bone01, params["round_sigma"])
        bone01 = remove_small_components(bone01, 500)
        bone01 = keep_largest_component(bone01)

    except Exception as e:
        return {"loss": 10.0, "error": str(e)}

    # Morphometric accuracy
    morph = measure_all_morphometrics(bone01, voxel_um)

    morph_errors = {}
    targets = {"BVTV": bvtv, "TbTh_um_p50": tbth, "TbN_per_mm": tbn}
    for k, tgt in targets.items():
        meas = morph.get(k, 0)
        if tgt > 0:
            morph_errors[k] = abs(meas - tgt) / tgt
        else:
            morph_errors[k] = 1.0

    avg_morph_error = float(np.mean(list(morph_errors.values())))

    # SSIM
    gray = microct_gray_solid(bone01, gp, np.random.default_rng(seed + 1000), br=br)
    Z = shape[0]
    mid_slice = gray[Z // 2].astype(np.float32) / 255.0

    # Resize to match reference
    mid_resized = np.array(Image.fromarray((mid_slice * 255).astype(np.uint8)).resize(
        (ref_image.shape[1], ref_image.shape[0]), Image.BILINEAR)) / 255.0

    ssim = compute_ssim_2d(ref_image, mid_resized)

    # Connectivity bonus
    lcc = morph.get("lcc_frac", 0.0)
    conn_penalty = 0.0 if lcc >= 0.8 else 0.5 * (0.8 - lcc)

    # Combined loss (lower is better)
    loss = morph_weight * avg_morph_error + ssim_weight * (1.0 - ssim) + conn_penalty

    return {
        "loss": float(loss),
        "ssim": float(ssim),
        "morph_error": float(avg_morph_error),
        "morph_errors": morph_errors,
        "bvtv_measured": float(morph["BVTV"]),
        "lcc_frac": float(lcc),
        "n_components": int(morph["n_components"]),
    }


def run_study(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load reference
    ref = load_reference(args.reference_image, target_size=128)
    print(f"Reference image: {args.reference_image} ({ref.shape})")

    shape = (40, 128, 128)  # Small for speed
    trial_results = []
    t0 = time.time()

    def objective(trial):
        params = {
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

        if HAS_V16:
            params["rod_weight"] = trial.suggest_float("rod_weight", 0.70, 1.00)
            params["plate_weight"] = trial.suggest_float("plate_weight", 0.00, 0.30)
        else:
            params["rod_weight"] = 1.0
            params["plate_weight"] = 0.0

        result = generate_and_score(params, ref, shape=shape)

        # Store for analysis
        trial_result = {**params, **result}
        trial_results.append(trial_result)

        n = trial.number + 1
        if n % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Trial {n}/{args.n_trials} | loss={result['loss']:.4f} "
                  f"ssim={result.get('ssim', 0):.3f} morph_err={result.get('morph_error', 0):.3f} "
                  f"| {elapsed:.0f}s elapsed")

        return result["loss"]

    # Run study
    print(f"\nRunning {args.n_trials} trials at {shape} for parameter importance...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    elapsed = time.time() - t0
    print(f"\nCompleted {args.n_trials} trials in {elapsed:.0f}s ({elapsed/args.n_trials:.1f}s/trial)")

    # Best parameters
    best = study.best_params
    best_val = study.best_value
    print(f"\nBest loss: {best_val:.4f}")
    print(f"Best params:")
    for k, v in sorted(best.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Parameter importance
    try:
        importances = get_param_importances(study)
        print(f"\nParameter importance:")
        for k, v in sorted(importances.items(), key=lambda x: -x[1]):
            bar = "#" * int(v * 50)
            print(f"  {k:<22} {v:.4f} {bar}")
    except Exception as e:
        print(f"  Could not compute importance: {e}")
        importances = {}

    # ── Save results ──

    # 1. Importance plot
    if importances:
        names = list(importances.keys())
        values = list(importances.values())

        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4)))
        y_pos = range(len(names))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
        ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel("Importance (fraction of variance explained)")
        ax.set_title("Generator Parameter Importance\n(Bayesian Optimization, "
                     f"{args.n_trials} trials)")
        ax.invert_yaxis()
        for i, v in enumerate(values):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(outdir / "importance_plot.png", dpi=150)
        plt.close()
        print(f"\nSaved: {outdir / 'importance_plot.png'}")

    # 2. Loss convergence plot
    fig, ax = plt.subplots(figsize=(10, 5))
    trials_vals = [t.value for t in study.trials if t.value is not None]
    best_so_far = np.minimum.accumulate(trials_vals)
    ax.plot(trials_vals, "o", alpha=0.3, markersize=4, label="Trial loss")
    ax.plot(best_so_far, "-", color="red", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Combined Loss")
    ax.set_title("Optimization Convergence")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "convergence_plot.png", dpi=150)
    plt.close()
    print(f"Saved: {outdir / 'convergence_plot.png'}")

    # 3. Best params JSON
    with open(outdir / "best_params.json", "w") as f:
        json.dump({
            "best_loss": float(best_val),
            "best_params": {k: float(v) if isinstance(v, (int, float)) else v
                           for k, v in best.items()},
            "n_trials": args.n_trials,
            "shape_used": list(shape),
            "reference_image": args.reference_image,
            "importances": {k: float(v) for k, v in importances.items()} if importances else {},
        }, f, indent=2)
    print(f"Saved: {outdir / 'best_params.json'}")

    # 4. All trial results CSV
    if trial_results:
        import csv
        csv_path = outdir / "trial_results.csv"
        keys = trial_results[0].keys()
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in trial_results:
                # Flatten nested dicts
                flat = {}
                for k, v in r.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            flat[f"{k}_{kk}"] = vv
                    else:
                        flat[k] = v
                w.writerow(flat)
        print(f"Saved: {csv_path}")

    # 5. Importance scores JSON
    if importances:
        with open(outdir / "importance_scores.json", "w") as f:
            json.dump({
                "importances": {k: float(v) for k, v in importances.items()},
                "ranking": [k for k, v in sorted(importances.items(), key=lambda x: -x[1])],
                "top_3": [k for k, v in sorted(importances.items(), key=lambda x: -x[1])[:3]],
            }, f, indent=2)
        print(f"Saved: {outdir / 'importance_scores.json'}")

    # Print the v16 command with best params
    print(f"\n{'='*60}")
    print("  RECOMMENDED v16 COMMAND WITH OPTIMAL PARAMS:")
    print(f"{'='*60}")
    print(f"python synthetic_trabecular_v16_morphometric_control.py `")
    print(f"    --voi-dirs data\\derived\\VOI1 data\\derived\\VOI4 `")
    print(f"    --outdir output\\optimized_dataset `")
    print(f"    --num-samples 30 `")
    print(f"    --xy 256 --z 80 `")
    print(f"    --voxel-um 39 `")
    if "bvtv" in best:
        print(f"    --bvtv {best['bvtv']:.3f} `")
    if "tbth_um" in best:
        print(f"    --tbth-um {best['tbth_um']:.0f} `")
    if "base_sigma" in best:
        print(f"    --base-sigma {best['base_sigma']:.2f} `")
    if "aniso_ratio" in best:
        print(f"    --aniso-ratio {best['aniso_ratio']:.2f} `")
    if "warp_amp" in best:
        print(f"    --warp-amp {best['warp_amp']:.2f} `")
    if "hessian_sigma" in best:
        print(f"    --hessian-sigma {best['hessian_sigma']:.2f} `")
    if "proto_q_hi" in best:
        print(f"    --proto-q-hi {best['proto_q_hi']:.3f} `")
    if "proto_q_lo" in best:
        print(f"    --proto-q-lo {best['proto_q_lo']:.3f} `")
    if "proto_close_iters" in best:
        print(f"    --proto-close-iters {best['proto_close_iters']} `")
    if "rod_weight" in best:
        print(f"    --rod-weight {best['rod_weight']:.3f} `")
    if "plate_weight" in best:
        print(f"    --plate-weight {best['plate_weight']:.3f} `")
    if "radius_jitter" in best:
        print(f"    --radius-jitter {best['radius_jitter']:.3f} `")
    if "round_sigma" in best:
        print(f"    --round-sigma {best['round_sigma']:.2f} `")
    print(f"    --solid-fill-sigma 3.0 `")
    print(f"    --marrow-mean 15 --bone-mean 240 `")
    print(f"    --base-seed 42")


def main():
    p = argparse.ArgumentParser(description="Generator parameter importance analysis")
    p.add_argument("--reference-image", type=str, required=True,
                   help="Path to a real VOI1 mid-slice PNG for SSIM comparison")
    p.add_argument("--voi-dirs", nargs="+", type=str, default=None,
                   help="VOI directories (not used for generation, just for context)")
    p.add_argument("--n-trials", type=int, default=60,
                   help="Number of optimization trials (60 ~= 30min)")
    p.add_argument("--outdir", type=str, default="output/importance_analysis")
    args = p.parse_args()
    run_study(args)


if __name__ == "__main__":
    main()