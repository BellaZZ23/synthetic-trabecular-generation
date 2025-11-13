#!/usr/bin/env python3
# synthetic-trabecular-generation.py
from pathlib import Path
import argparse, json, csv
import numpy as np
from PIL import Image
import tifffile as tiff

# -------- calibration defaults (edit as needed) --------
PIXEL_SIZE_UM = 10.0   # X/Y micron per pixel
Z_STEP_UM     = 1.0    # micron per slice

# ================== core generators ====================
def vertical_rods_mask(h, w, rod_width_px=8, gap_px=32, left_margin_px=0):
    x = np.arange(w)[None, :]
    period = rod_width_px + gap_px
    in_rod = (x - left_margin_px) % period < rod_width_px
    return np.repeat(in_rod, h, axis=0).astype(np.uint8)

def horizontal_rods_mask(h, w, rod_width_px=8, gap_px=32, top_margin_px=0):
    y = np.arange(h)[:, None]
    period = rod_width_px + gap_px
    in_rod = (y - top_margin_px) % period < rod_width_px
    return np.repeat(in_rod, w, axis=1).astype(np.uint8)

def orthotropic_grid_mask(h, w,
                          rod_width_x_px=8, gap_x_px=32,
                          rod_width_y_px=8, gap_y_px=32,
                          offset_x_px=0, offset_y_px=0):
    v = vertical_rods_mask(h, w, rod_width_x_px, gap_x_px, offset_x_px)
    hmask = horizontal_rods_mask(h, w, rod_width_y_px, gap_y_px, offset_y_px)
    return np.maximum(v, hmask)

def trabecula_mask_for_grayscale(h, w,
                                 rod_width_x_px=8, gap_x_px=32,
                                 rod_width_y_px=8, gap_y_px=32,
                                 offset_x_px=0, offset_y_px=0):
    """
    Mask for grayscale rendering ONLY.
    This is the union of vertical + horizontal rods.
    Pixels in the rods = 1 (will get grayscale).
    Everything else = 0 (will stay black).
    """
    v = vertical_rods_mask(h, w, rod_width_x_px, gap_x_px, offset_x_px)
    hmask = horizontal_rods_mask(h, w, rod_width_y_px, gap_y_px, offset_y_px)
    trab = np.maximum(v, hmask).astype(np.uint8)
    return trab

# =================== utilities =========================
def um_to_px(val_um, pixel_size_um=None):
    if pixel_size_um is None:
        pixel_size_um = PIXEL_SIZE_UM
    return int(round(val_um / pixel_size_um))

def save_binary_png(mask01, out_path):
    """mask01: 0/1 bone mask; save as 8-bit (bone=255, bg=0)"""
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    img = (mask01 * 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(out_path)

def save_tiff_stack(mask01, out_path, n_slices=2, z_step_um=None):
    """ImageJ-friendly TIFF with micron units + z spacing; sidecar JSON for XY."""
    if z_step_um is None:
        z_step_um = Z_STEP_UM
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    img = (mask01 * 255).astype(np.uint8)
    stack = np.stack([img] * n_slices, axis=0)  # (Z,H,W)
    tiff.imwrite(out_path, stack, imagej=True,
                 metadata={'unit': 'micron', 'spacing': float(z_step_um)},
                 dtype=np.uint8)
    with open(out_path.with_suffix(".tif.json"), "w") as f:
        json.dump({"pixel_size_um": float(PIXEL_SIZE_UM),
                   "z_step_um": float(z_step_um)}, f, indent=2)

def bs_ts(mask01):
    """2D bone area fraction BS/TS on a single slice"""
    bs = int(mask01.sum()); ts = int(mask01.size)
    return bs, ts, (bs/ts if ts else 0.0)

def apply_grayscale(mask01, mode="none", noise_sigma=0.0, poisson=False):
    """
    Apply grayscale gradient and/or noise ONLY within the white (bone) regions.
    Black background remains pure 0.

    mask01 : np.ndarray
        Binary mask (0 = background, 1 = bone).
    Returns
    -------
    np.ndarray : 8-bit grayscale image.
    """
    # Make sure input is binary (0/1)
    mask01 = (mask01 > 0).astype(np.float32)
    H, W = mask01.shape

    # Base grayscale field (only bone regions)
    if mode == "none":
        arr = mask01 * 255.0  # pure white bone, black background
    elif mode == "linear_x":
        gradient = np.linspace(0, 1, W, dtype=np.float32)[None, :].repeat(H, 0)
        arr = gradient * 255.0 * mask01  # apply gradient only inside bone
    elif mode == "linear_y":
        gradient = np.linspace(0, 1, H, dtype=np.float32)[:, None].repeat(W, 1)
        arr = gradient * 255.0 * mask01
    else:
        raise ValueError(f"Unknown grayscale mode: {mode}")

    # Add Gaussian noise within bone only
    if noise_sigma and noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, arr.shape).astype(np.float32)
        arr = np.where(mask01 > 0, arr + noise, 0)

    # Optional Poisson noise within bone only
    if poisson:
        lam = np.clip(arr / 255.0, 0, 1) * 20.0
        poisson_arr = np.random.poisson(lam).astype(np.float32) / 20.0 * 255.0
        arr = np.where(mask01 > 0, poisson_arr, 0)

    # Clamp to 8-bit range and return
    return np.clip(arr, 0, 255).astype(np.uint8)

def extra_morphometrics(mask01):
    """
    Placeholder for future morphometric metrics
    (e.g. connectivity, anisotropy). Currently returns empty dict.
    """
    return {}

def append_csv(csv_path, row, header=None):
    csv_path = Path(csv_path); csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        if header is None:
            header = list(row.keys())
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)

# ======================= helpers =======================
def compute_intensity_features(mask01, img_gray):
    """
    Compute simple intensity features inside bone:
    mean, std, and histogram (16 bins).
    """
    mask_bool = mask01 > 0
    if not np.any(mask_bool):
        return {
            "mean_intensity": 0.0,
            "std_intensity": 0.0,
            **{f"hist_bin_{i}": 0.0 for i in range(16)},
        }

    bone_pixels = img_gray[mask_bool]
    mean_intensity = float(bone_pixels.mean())
    std_intensity = float(bone_pixels.std())
    hist, _ = np.histogram(bone_pixels, bins=16, range=(0, 255))
    hist = hist.astype(float) / float(bone_pixels.size)

    feats = {
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
    }
    for i, hval in enumerate(hist):
        feats[f"hist_bin_{i}"] = float(hval)
    return feats

def _write_row_with_common_fields(csv_path, base_row, mask01, img_gray_or_none,
                                  args, pix, pattern):
    """
    Helper to enrich a CSV row with common image-formation and morphometric info.
    """
    bs, ts, frac = bs_ts(mask01)
    row = dict(base_row)
    row.update({
        "pattern": pattern,
        "pixel_size_um": float(pix),
        "BS": bs,
        "TS": ts,
        "BS_TS": frac,
        "grayscale": base_row.get("grayscale", None),
        "noise_sigma": float(args.noise_sigma),
        "poisson": int(bool(args.poisson)),
        "z_step_um": float(args.z_step_um),
        "slices": int(args.slices),
    })

    # add extra morphometrics
    row.update(extra_morphometrics(mask01))

    # add intensity features if grayscale image is available
    if img_gray_or_none is not None:
        row.update(compute_intensity_features(mask01, img_gray_or_none))

    append_csv(csv_path, row)

# ======================= CLI tasks =====================
def cmd_single(args):
    H = W = args.size
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # convert µm→px
    if args.pattern == "grid":
        tx_um = args.thickness_x_um if args.thickness_x_um is not None else args.thickness_um
        ty_um = args.thickness_y_um if args.thickness_y_um is not None else args.thickness_um
        gx_um = args.spacing_x_um   if args.spacing_x_um   is not None else args.spacing_um
        gy_um = args.spacing_y_um   if args.spacing_y_um   is not None else args.spacing_um

        tx = um_to_px(tx_um)
        ty = um_to_px(ty_um)
        gx = um_to_px(gx_um)
        gy = um_to_px(gy_um)

        mask = orthotropic_grid_mask(H, W, tx, gx, ty, gy, 0, 0)
        name = f"grid_thx{tx_um}_thy{ty_um}_spx{gx_um}_spy{gy_um}"
    elif args.pattern == "vertical":
        t = um_to_px(args.thickness_um)
        g = um_to_px(args.spacing_um)
        mask = vertical_rods_mask(H, W, t, g, 0)
        name = f"vertical_th{args.thickness_um}_sp{args.spacing_um}"
    elif args.pattern == "horizontal":
        t = um_to_px(args.thickness_um)
        g = um_to_px(args.spacing_um)
        mask = horizontal_rods_mask(H, W, t, g, 0)
        name = f"horizontal_th{args.thickness_um}_sp{args.spacing_um}"
    else:
        raise ValueError("pattern must be grid|vertical|horizontal")

    # metrics (2D)
    bs, ts, frac = bs_ts(mask)
    print(f"[{name.upper()}] BS={bs}, TS={ts}, BS/TS={frac:.3f}")

    # save binary & stack
    save_binary_png(mask, out / f"{name}.png")
    save_tiff_stack(mask, out / f"{name}_stack.tif",
                    n_slices=args.slices, z_step_um=args.z_step_um)

    img_gray = None
    if args.grayscale != "none" or args.noise_sigma > 0 or args.poisson:
        img_gray = apply_grayscale(mask, mode=args.grayscale,
                                   noise_sigma=args.noise_sigma, poisson=args.poisson)
        Image.fromarray(img_gray).save(out / f"{name}_grayscale.png")

    # csv
    csv_path = out / "results_2d_bs_ts.csv"
    base_row = {
        "name": name,
        "grayscale": args.grayscale,
    }
    if args.pattern == "grid":
        base_row.update({
            "thickness_um_x": tx_um,
            "spacing_um_x": gx_um,
            "thickness_um_y": ty_um,
            "spacing_um_y": gy_um,
        })
    else:
        base_row.update({
            "thickness_um": args.thickness_um,
            "spacing_um": args.spacing_um,
        })

    _write_row_with_common_fields(csv_path, base_row, mask, img_gray,
                                  args, args.pixel_size_um, args.pattern)

def _sweep_grid_pattern(args, pix, H, W,
                        thx_um, thy_um, spx_um, spy_um,
                        modes, csv_path):
    out = Path(args.outdir)

    tx = um_to_px(thx_um, pixel_size_um=pix)
    ty = um_to_px(thy_um, pixel_size_um=pix)
    gx = um_to_px(spx_um, pixel_size_um=pix)
    gy = um_to_px(spy_um, pixel_size_um=pix)

    grid_mask = orthotropic_grid_mask(H, W, tx, gx, ty, gy)
    grid_mask_for_gray = trabecula_mask_for_grayscale(H, W, tx, gx, ty, gy)

    base = out / f"pix{int(pix)}um" / f"grid_thx{thx_um}_thy{thy_um}_spx{spx_um}_spy{spy_um}"
    base.mkdir(parents=True, exist_ok=True)

    # save the pure binary bone mask for reference
    save_binary_png(grid_mask, base / "preview.png")

    for mode in modes:
        sub_gray = base / f"grayscale-{mode}"
        sub_gray.mkdir(parents=True, exist_ok=True)

        img_gray = None
        if mode != "none" or args.noise_sigma > 0 or args.poisson:
            img_gray = apply_grayscale(
                grid_mask_for_gray,
                mode=mode,
                noise_sigma=args.noise_sigma,
                poisson=args.poisson
            )
            Image.fromarray(img_gray).save(sub_gray / "preview_grayscale.png")

        save_tiff_stack(grid_mask_for_gray, sub_gray / "stack.tif",
                        n_slices=args.slices,
                        z_step_um=args.z_step_um)

        base_row = {
            "name": sub_gray.name,
            "grayscale": mode,
            "thickness_um_x": thx_um,
            "spacing_um_x": spx_um,
            "thickness_um_y": thy_um,
            "spacing_um_y": spy_um,
        }
        _write_row_with_common_fields(csv_path, base_row, grid_mask_for_gray,
                                      img_gray, args, pix, "grid")

def _sweep_vertical_pattern(args, pix, H, W,
                            th_um, sp_um,
                            modes, csv_path):
    out = Path(args.outdir)

    t = um_to_px(th_um, pixel_size_um=pix)
    g = um_to_px(sp_um, pixel_size_um=pix)

    vert_mask = vertical_rods_mask(H, W, t, g)
    base = out / f"pix{int(pix)}um" / f"vertical_th{th_um}_sp{sp_um}"
    base.mkdir(parents=True, exist_ok=True)

    save_binary_png(vert_mask, base / "preview.png")

    for mode in modes:
        sub_gray = base / f"grayscale-{mode}"
        sub_gray.mkdir(parents=True, exist_ok=True)

        img_gray = None
        if mode != "none" or args.noise_sigma > 0 or args.poisson:
            img_gray = apply_grayscale(
                vert_mask,
                mode=mode,
                noise_sigma=args.noise_sigma,
                poisson=args.poisson
            )
            Image.fromarray(img_gray).save(sub_gray / "preview_grayscale.png")

        save_tiff_stack(vert_mask, sub_gray / "stack.tif",
                        n_slices=args.slices,
                        z_step_um=args.z_step_um)

        base_row = {
            "name": sub_gray.name,
            "grayscale": mode,
            "thickness_um": th_um,
            "spacing_um": sp_um,
        }
        _write_row_with_common_fields(csv_path, base_row, vert_mask,
                                      img_gray, args, pix, "vertical")

def _sweep_horizontal_pattern(args, pix, H, W,
                              th_um, sp_um,
                              modes, csv_path):
    out = Path(args.outdir)

    t = um_to_px(th_um, pixel_size_um=pix)
    g = um_to_px(sp_um, pixel_size_um=pix)

    horiz_mask = horizontal_rods_mask(H, W, t, g)
    base = out / f"pix{int(pix)}um" / f"horizontal_th{th_um}_sp{sp_um}"
    base.mkdir(parents=True, exist_ok=True)

    save_binary_png(horiz_mask, base / "preview.png")

    for mode in modes:
        sub_gray = base / f"grayscale-{mode}"
        sub_gray.mkdir(parents=True, exist_ok=True)

        img_gray = None
        if mode != "none" or args.noise_sigma > 0 or args.poisson:
            img_gray = apply_grayscale(
                horiz_mask,
                mode=mode,
                noise_sigma=args.noise_sigma,
                poisson=args.poisson
            )
            Image.fromarray(img_gray).save(sub_gray / "preview_grayscale.png")

        save_tiff_stack(horiz_mask, sub_gray / "stack.tif",
                        n_slices=args.slices,
                        z_step_um=args.z_step_um)

        base_row = {
            "name": sub_gray.name,
            "grayscale": mode,
            "thickness_um": th_um,
            "spacing_um": sp_um,
        }
        _write_row_with_common_fields(csv_path, base_row, horiz_mask,
                                      img_gray, args, pix, "horizontal")

def cmd_sweep(args):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "results_2d_bs_ts.csv"

    # accept multiple grayscale modes (or fall back to one)
    modes = getattr(args, "grayscale_modes", None) or [args.grayscale]

    # Geometry parameter sets (anisotropic if provided, otherwise isotropic)
    th_x_list = args.thickness_x_um or args.thickness_um
    th_y_list = args.thickness_y_um or args.thickness_um
    sp_x_list = args.spacing_x_um   or args.spacing_um
    sp_y_list = args.spacing_y_um   or args.spacing_um

    for pix in args.pixel_sizes_um:
        # update global calibration for TIFF sidecar
        global PIXEL_SIZE_UM
        PIXEL_SIZE_UM = float(pix)

        H = W = args.size

        for thx in th_x_list:
            for thy in th_y_list:
                for spx in sp_x_list:
                    for spy in sp_y_list:

                        if "grid" in args.patterns:
                            _sweep_grid_pattern(args, pix, H, W,
                                                thx, thy, spx, spy,
                                                modes, csv_path)

                        if "vertical" in args.patterns:
                            _sweep_vertical_pattern(args, pix, H, W,
                                                    thx, spx,
                                                    modes, csv_path)

                        if "horizontal" in args.patterns:
                            _sweep_horizontal_pattern(args, pix, H, W,
                                                      thy, spy,
                                                      modes, csv_path)

    print(f"✓ Sweep done → {csv_path}")

# =================== argument parser ===================
def build_parser():
    p = argparse.ArgumentParser(
        description="Synthetic trabecular generator (2D→3D stack, BS/TS, rich metadata)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--outdir", type=str, default="data/outputs")
    common.add_argument("--size", type=int, default=512, help="image size H=W")
    common.add_argument("--slices", type=int, default=2, help="Z slices in stack")
    common.add_argument("--pixel-size-um", type=float, default=PIXEL_SIZE_UM)
    common.add_argument("--z-step-um", type=float, default=Z_STEP_UM)
    common.add_argument("--grayscale", choices=["none", "linear_x", "linear_y"], default="none")
    common.add_argument("--noise-sigma", type=float, default=0.0)
    common.add_argument("--poisson", action="store_true")

    s1 = sub.add_parser("single", parents=[common], help="Generate one pattern")
    s1.add_argument("--pattern", choices=["grid", "vertical", "horizontal"], required=True)
    s1.add_argument("--thickness-um", type=float, default=120.0)
    s1.add_argument("--spacing-um", type=float, default=400.0)
    s1.add_argument("--thickness-x-um", type=float, default=None,
                    help="grid only: X (vertical bars) thickness")
    s1.add_argument("--spacing-x-um", type=float, default=None,
                    help="grid only: X (vertical bars) spacing")
    s1.add_argument("--thickness-y-um", type=float, default=None,
                    help="grid only: Y (horizontal bars) thickness")
    s1.add_argument("--spacing-y-um", type=float, default=None,
                    help="grid only: Y (horizontal bars) spacing")

    s2 = sub.add_parser("sweep", parents=[common],
                        help="Parameter sweep over geometry and image formation")
    s2.add_argument("--pixel-sizes-um", type=float, nargs="+", default=[5, 10, 20, 30])

    # isotropic defaults
    s2.add_argument("--thickness-um", type=float, nargs="+", default=[100, 125, 150],
                    help="isotropic thickness set (used if X/Y not specified)")
    s2.add_argument("--spacing-um", type=float, nargs="+", default=[300, 400, 500],
                    help="isotropic spacing set (used if X/Y not specified)")

    # optional anisotropic sets
    s2.add_argument("--thickness-x-um", type=float, nargs="+", default=None,
                    help="optional X-direction thickness values")
    s2.add_argument("--thickness-y-um", type=float, nargs="+", default=None,
                    help="optional Y-direction thickness values")
    s2.add_argument("--spacing-x-um", type=float, nargs="+", default=None,
                    help="optional X-direction spacing values")
    s2.add_argument("--spacing-y-um", type=float, nargs="+", default=None,
                    help="optional Y-direction spacing values")

    # which patterns to generate
    s2.add_argument("--patterns",
                    choices=["grid", "vertical", "horizontal"],
                    nargs="+",
                    default=["grid", "vertical"],
                    help="Patterns to include in sweep")

    s2.add_argument("--grayscale-modes",
                    choices=["none", "linear_x", "linear_y"],
                    nargs="+",
                    default=["none", "linear_x", "linear_y"],
                    help="Grayscale profiles to sweep over (default: all).")

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    # push CLI calib into globals so TIFF sidecar is correct
    global PIXEL_SIZE_UM, Z_STEP_UM
    PIXEL_SIZE_UM = float(getattr(args, "pixel_size_um", PIXEL_SIZE_UM))
    Z_STEP_UM     = float(getattr(args, "z_step_um", Z_STEP_UM))
    if args.cmd == "single":
        cmd_single(args)
    elif args.cmd == "sweep":
        cmd_sweep(args)

if __name__ == "__main__":
    main()
