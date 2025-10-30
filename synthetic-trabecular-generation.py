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

# =================== utilities =========================
def um_to_px(val_um, pixel_size_um=PIXEL_SIZE_UM):
    return int(round(val_um / pixel_size_um))

def save_binary_png(mask01, out_path):
    """mask01: 0/1 bone mask; save as 8-bit (bone=255, bg=0)"""
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    img = (mask01 * 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(out_path)

def save_tiff_stack(mask01, out_path, n_slices=2, z_step_um=Z_STEP_UM):
    """ImageJ-friendly TIFF with micron units + z spacing; sidecar JSON for XY."""
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

def append_csv(csv_path, row, header=None):
    csv_path = Path(csv_path); csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header or list(row.keys()))
        if not exists: w.writeheader()
        w.writerow(row)

# ======================= CLI tasks =====================
def cmd_single(args):
    H = W = args.size
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    # convert µm→px
    if args.pattern == "grid":
        tx = um_to_px(args.thickness_x_um); gx = um_to_px(args.spacing_x_um)
        ty = um_to_px(args.thickness_y_um or args.thickness_x_um)
        gy = um_to_px(args.spacing_y_um or args.spacing_x_um)
        mask = orthotropic_grid_mask(H,W, tx,gx, ty,gy, 0,0)
        name = "grid"
    elif args.pattern == "vertical":
        t = um_to_px(args.thickness_um); g = um_to_px(args.spacing_um)
        mask = vertical_rods_mask(H,W, t,g,0); name = "rods_vertical"
    elif args.pattern == "horizontal":
        t = um_to_px(args.thickness_um); g = um_to_px(args.spacing_um)
        mask = horizontal_rods_mask(H,W, t,g,0); name = "rods_horizontal"
    else:
        raise ValueError("pattern must be grid|vertical|horizontal")

    # metrics (2D)
    bs, ts, frac = bs_ts(mask)
    print(f"[{name.upper()}] BS={bs}, TS={ts}, BS/TS={frac:.3f}")

    # save binary & stack
    save_binary_png(mask, out/f"{name}.png")
    save_tiff_stack(mask, out/f"{name}_stack.tif", n_slices=args.slices, z_step_um=args.z_step_um)

    # optional grayscale preview
    if args.grayscale != "none" or args.noise_sigma>0 or args.poisson:
        img_gray = apply_grayscale(mask, mode=args.grayscale,
                                   noise_sigma=args.noise_sigma, poisson=args.poisson)
        Image.fromarray(img_gray).save(out/f"{name}_grayscale.png")

    # csv
    row = {"name": name, "pattern": args.pattern, "pixel_size_um": args.pixel_size_um,
           "BS": bs, "TS": ts, "BS_TS": frac}
    if args.pattern=="grid":
        row.update({"thickness_um_x": args.thickness_x_um, "spacing_um_x": args.spacing_x_um,
                    "thickness_um_y": (args.thickness_y_um or args.thickness_x_um),
                    "spacing_um_y": (args.spacing_y_um or args.spacing_x_um)})
    else:
        row.update({"thickness_um": args.thickness_um, "spacing_um": args.spacing_um})
    append_csv(out/"results_2d_bs_ts.csv", row)

def cmd_sweep(args):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "results_2d_bs_ts.csv"

    # accept multiple grayscale modes (or fall back to one)
    modes = getattr(args, "grayscale_modes", None) or [args.grayscale]

    for pix in args.pixel_sizes_um:
        # update global calibration for TIFF sidecar
        global PIXEL_SIZE_UM
        PIXEL_SIZE_UM = float(pix)

        H = W = args.size

        for th in args.thickness_um:
            for sp in args.spacing_um:
                # =====================================================
                # GRID PATTERN
                # =====================================================
                tx = um_to_px(th); gx = um_to_px(sp)
                grid_mask = orthotropic_grid_mask(H, W, tx, gx, tx, gx)
                bs, ts, frac = bs_ts(grid_mask)

                base = out / f"pix{int(pix)}um" / f"grid_th{th}_sp{sp}"
                base.mkdir(parents=True, exist_ok=True)
                save_binary_png(grid_mask, base / "preview.png")

                # loop over grayscale modes
                for mode in modes:
                    sub_gray = base / f"grayscale-{mode}"
                    sub_gray.mkdir(parents=True, exist_ok=True)

                    if mode != "none" or args.noise_sigma > 0 or args.poisson:
                        img_gray = apply_grayscale(
                            grid_mask,
                            mode=mode,
                            noise_sigma=args.noise_sigma,
                            poisson=args.poisson
                        )
                        Image.fromarray(img_gray).save(sub_gray / "preview_grayscale.png")

                    save_tiff_stack(grid_mask, sub_gray / "stack.tif",
                                    n_slices=args.slices,
                                    z_step_um=args.z_step_um)

                    append_csv(csv_path, {
                        "name": sub_gray.name,
                        "pattern": "grid",
                        "pixel_size_um": pix,
                        "thickness_um_x": th, "spacing_um_x": sp,
                        "thickness_um_y": th, "spacing_um_y": sp,
                        "grayscale": mode,
                        "BS": bs, "TS": ts, "BS_TS": frac
                    })

                # =====================================================
                # VERTICAL PATTERN
                # =====================================================
                t = um_to_px(th); g = um_to_px(sp)
                vert_mask = vertical_rods_mask(H, W, t, g)
                bs, ts, frac = bs_ts(vert_mask)

                base = out / f"pix{int(pix)}um" / f"vertical_th{th}_sp{sp}"
                base.mkdir(parents=True, exist_ok=True)
                save_binary_png(vert_mask, base / "preview.png")

                for mode in modes:
                    sub_gray = base / f"grayscale-{mode}"
                    sub_gray.mkdir(parents=True, exist_ok=True)

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

                    append_csv(csv_path, {
                        "name": sub_gray.name,
                        "pattern": "vertical",
                        "pixel_size_um": pix,
                        "thickness_um": th, "spacing_um": sp,
                        "grayscale": mode,
                        "BS": bs, "TS": ts, "BS_TS": frac
                    })

    print(f"✓ Sweep done → {csv_path}")

# =================== argument parser ===================
def build_parser():
    p = argparse.ArgumentParser(description="Synthetic trabecular generator (2D→3D stack, BS/TS).")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--outdir", type=str, default="data/outputs")
    common.add_argument("--size", type=int, default=512, help="image size H=W")
    common.add_argument("--slices", type=int, default=2, help="Z slices in stack")
    common.add_argument("--pixel-size-um", type=float, default=PIXEL_SIZE_UM)
    common.add_argument("--z-step-um", type=float, default=Z_STEP_UM)
    common.add_argument("--grayscale", choices=["none","linear_x","linear_y"], default="none")
    common.add_argument("--noise-sigma", type=float, default=0.0)
    common.add_argument("--poisson", action="store_true")

    s1 = sub.add_parser("single", parents=[common], help="Generate one pattern")
    s1.add_argument("--pattern", choices=["grid","vertical","horizontal"], required=True)
    s1.add_argument("--thickness-um", type=float, default=120.0)
    s1.add_argument("--spacing-um", type=float, default=400.0)
    s1.add_argument("--thickness-x-um", type=float, default=None,
                    help="grid only: X (vertical bars)")
    s1.add_argument("--spacing-x-um", type=float, default=None)
    s1.add_argument("--thickness-y-um", type=float, default=None,
                    help="grid only: Y (horizontal bars)")
    s1.add_argument("--spacing-y-um", type=float, default=None)

    s2 = sub.add_parser("sweep", parents=[common], help="Parameter sweep (grid + vertical)")
    s2.add_argument("--pixel-sizes-um", type=float, nargs="+", default=[5,10,20,30])
    s2.add_argument("--thickness-um", type=float, nargs="+", default=[100,125,150])
    s2.add_argument("--spacing-um", type=float, nargs="+", default=[300,400,500])

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
        # massage grid x/y defaults
        if args.pattern == "grid":
            if args.thickness_x_um is None: args.thickness_x_um = args.thickness_um
            if args.spacing_x_um   is None: args.spacing_x_um   = args.spacing_um
        cmd_single(args)
    elif args.cmd == "sweep":
        cmd_sweep(args)

if __name__ == "__main__":
    main()
