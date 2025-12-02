#!/usr/bin/env python3
"""
synthetic_trabecular_v2_heterogeneous_curved.py

v2 synthetic trabecular generator:
- Multi-zone heterogeneous images (3–5 zones per image)
- Mixture of patterns:
    * grid
    * vertical rods
    * horizontal rods
    * sinusoidal curved bands
    * radial ring / arc-like trabeculae
- Simple grayscale models (none, linear_x, linear_y)
- Outputs:
    * per-image mask + grayscale PNGs
    * per-image TIFF stacks
    * images.csv  (one row per image)
    * zones.csv   (one row per zone per image)
"""

from pathlib import Path
import argparse
import json
import csv
import math

import numpy as np
from PIL import Image
import tifffile as tiff

# -----------------------------------------------------------
# Global calibration defaults
# -----------------------------------------------------------
PIXEL_SIZE_UM = 10.0   # X/Y micron per pixel
Z_STEP_UM     = 1.0    # micron per slice


# -----------------------------------------------------------
# Basic straight pattern generators
# -----------------------------------------------------------
def vertical_rods_mask(h, w, rod_width_px=8, gap_px=32, left_margin_px=0):
    """
    Vertical rods spaced periodically along x.
    """
    x = np.arange(w)[None, :]
    period = rod_width_px + gap_px
    in_rod = (x - left_margin_px) % period < rod_width_px
    return np.repeat(in_rod, h, axis=0).astype(np.uint8)


def horizontal_rods_mask(h, w, rod_width_px=8, gap_px=32, top_margin_px=0):
    """
    Horizontal rods spaced periodically along y.
    """
    y = np.arange(h)[:, None]
    period = rod_width_px + gap_px
    in_rod = (y - top_margin_px) % period < rod_width_px
    return np.repeat(in_rod, w, axis=1).astype(np.uint8)


def orthotropic_grid_mask(h, w,
                          rod_width_x_px=8, gap_x_px=32,
                          rod_width_y_px=8, gap_y_px=32,
                          offset_x_px=0, offset_y_px=0):
    """
    Grid of vertical + horizontal rods.
    """
    v = vertical_rods_mask(h, w, rod_width_x_px, gap_x_px, offset_x_px)
    hmask = horizontal_rods_mask(h, w, rod_width_y_px, gap_y_px, offset_y_px)
    return np.maximum(v, hmask)


# -----------------------------------------------------------
# Curved / radial pattern generators
# -----------------------------------------------------------
def sinusoidal_bands_mask(h, w, band_thickness_px=6,
                          wavelength_px=80, n_bands=3, amplitude_px=20):
    """
    Multiple sinusoidal horizontal trabeculae:
      y_center_k(x) = base_k + amplitude * sin(2π x / wavelength)
    """
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w), dtype=np.uint8)

    base_positions = np.linspace(h * 0.2, h * 0.8, n_bands)
    for base_y in base_positions:
        y_center = base_y + amplitude_px * np.sin(2.0 * np.pi * xx / wavelength_px)
        band = np.abs(yy - y_center) <= (band_thickness_px / 2.0)
        mask[band] = 1

    return mask


def radial_rings_mask(h, w, ring_thickness_px=6,
                      ring_gap_px=24, center=None,
                      angle_start_deg=0.0, angle_end_deg=360.0):
    """
    Concentric circular rings (full or partial arcs).
    angle_start_deg / angle_end_deg allow arcs instead of full rings.
    """
    if center is None:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = center

    yy, xx = np.mgrid[0:h, 0:w]
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy)

    period = ring_thickness_px + ring_gap_px
    in_ring = (r % period) < ring_thickness_px

    # restrict by angle if not full 360
    angle = np.degrees(np.arctan2(dy, dx))  # range [-180, 180]
    # wrap angles into [0, 360)
    angle[angle < 0] += 360.0
    a0 = angle_start_deg % 360.0
    a1 = angle_end_deg % 360.0
    if a0 < a1:
        in_sector = (angle >= a0) & (angle <= a1)
    else:
        # wrap-around case
        in_sector = (angle >= a0) | (angle <= a1)

    mask = (in_ring & in_sector).astype(np.uint8)
    return mask


# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------
def um_to_px(val_um, pixel_size_um=PIXEL_SIZE_UM):
    return int(round(val_um / pixel_size_um))


def apply_grayscale(mask01, mode="none"):
    """
    Simple grayscale model:
      - mode="none"      → uniform bright bone
      - mode="linear_x"  → left→right gradient within bone
      - mode="linear_y"  → top→bottom gradient within bone
    Background stays black.
    """
    mask01 = (mask01 > 0).astype(np.float32)
    h, w = mask01.shape

    if mode == "none":
        arr = mask01 * 255.0
    elif mode == "linear_x":
        grad = np.linspace(0.2, 1.0, w, dtype=np.float32)[None, :].repeat(h, 0)
        arr = grad * 255.0 * mask01
    elif mode == "linear_y":
        grad = np.linspace(0.2, 1.0, h, dtype=np.float32)[:, None].repeat(w, 1)
        arr = grad * 255.0 * mask01
    else:
        raise ValueError(f"Unknown grayscale mode: {mode}")

    return np.clip(arr, 0, 255).astype(np.uint8)


def save_png(arr, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(out_path)


def save_stack(mask01, out_path, n_slices=2, z_step_um=Z_STEP_UM):
    """
    Save a simple Z-stack (repeated 2D slice) with micron metadata.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = (mask01 * 255).astype(np.uint8)
    stack = np.stack([img] * n_slices, axis=0)  # (Z, H, W)
    tiff.imwrite(
        out_path,
        stack,
        imagej=True,
        metadata={"unit": "micron", "spacing": float(z_step_um)},
        dtype=np.uint8,
    )
    with open(out_path.with_suffix(".tif.json"), "w") as f:
        json.dump(
            {"pixel_size_um": float(PIXEL_SIZE_UM), "z_step_um": float(z_step_um)},
            f,
            indent=2,
        )


# -----------------------------------------------------------
# Zone definition & heterogeneous composition
# -----------------------------------------------------------
def generate_zone_mask(pattern, H, W, params_px, rng):
    """
    pattern : 'grid' | 'vertical' | 'horizontal' | 'sinusoidal' | 'radial'
    params_px : dict with pattern-specific parameters in pixels
    """
    if pattern == "grid":
        t_px = params_px["thickness_px"]
        g_px = params_px["spacing_px"]
        mask = orthotropic_grid_mask(H, W, t_px, g_px, t_px, g_px)

    elif pattern == "vertical":
        t_px = params_px["thickness_px"]
        g_px = params_px["spacing_px"]
        mask = vertical_rods_mask(H, W, t_px, g_px)

    elif pattern == "horizontal":
        t_px = params_px["thickness_px"]
        g_px = params_px["spacing_px"]
        mask = horizontal_rods_mask(H, W, t_px, g_px)

    elif pattern == "sinusoidal":
        t_px = params_px["thickness_px"]
        g_px = params_px["spacing_px"]
        wavelength = max(g_px, 16)
        amplitude = max(int(0.1 * H), 10)
        n_bands = int(rng.integers(2, 5))
        mask = sinusoidal_bands_mask(
            H,
            W,
            band_thickness_px=t_px,
            wavelength_px=wavelength,
            n_bands=n_bands,
            amplitude_px=amplitude,
        )

    elif pattern == "radial":
        t_px = params_px["thickness_px"]
        g_px = params_px["spacing_px"]
        cx = W / 2.0 + rng.uniform(-0.1 * W, 0.1 * W)
        cy = H / 2.0 + rng.uniform(-0.1 * H, 0.1 * H)
        angle_span = rng.uniform(120.0, 300.0)
        start_angle = rng.uniform(0.0, 360.0)
        end_angle = (start_angle + angle_span) % 360.0
        mask = radial_rings_mask(
            H,
            W,
            ring_thickness_px=t_px,
            ring_gap_px=g_px,
            center=(cx, cy),
            angle_start_deg=start_angle,
            angle_end_deg=end_angle,
        )

    else:
        raise ValueError(f"Unknown pattern '{pattern}'")

    return mask.astype(np.uint8)


def apply_zone_extent(full_mask, H, W, zone):
    """
    Restrict full_mask to the rectangular extent of the zone.
    zone has keys:
      - extent_axis: 'x' or 'y'
      - start, end  : indices along that axis
    """
    restricted = np.zeros_like(full_mask, dtype=np.uint8)
    if zone["extent_axis"] == "x":
        x0, x1 = zone["start"], zone["end"]
        restricted[:, x0:x1] = full_mask[:, x0:x1]
    else:
        y0, y1 = zone["start"], zone["end"]
        restricted[y0:y1, :] = full_mask[y0:y1, :]
    return restricted


def generate_heterogeneous_image(H, W, rng, patterns):
    """
    Generate a single heterogeneous image with 3–5 zones.

    Returns
    -------
    mask_hetero  : 0/1 uint8 mask
    gray_hetero  : 0–255 uint8 grayscale
    zones_meta   : list[dict] describing each zone
    """
    mask_hetero = np.zeros((H, W), dtype=np.uint8)
    gray_hetero = np.zeros((H, W), dtype=np.uint8)
    zones_meta = []

    # Choose 3–5 zones and whether we split along x or y
    n_zones = int(rng.integers(3, 6))
    axis = rng.choice(["x", "y"])
    length = W if axis == "x" else H
    boundaries = np.linspace(0, length, n_zones + 1, dtype=int)

    for zi in range(n_zones):
        start = int(boundaries[zi])
        end = int(boundaries[zi + 1])

        pattern = rng.choice(patterns)

        # sample geometric parameters in microns
        thickness_um = float(rng.uniform(80.0, 160.0))
        spacing_um = float(rng.uniform(250.0, 600.0))

        thickness_px = max(1, um_to_px(thickness_um))
        spacing_px = max(2, um_to_px(spacing_um))

        grayscale_mode = rng.choice(["none", "linear_x", "linear_y"])

        params_px = {
            "thickness_px": thickness_px,
            "spacing_px": spacing_px,
        }

        full_mask = generate_zone_mask(pattern, H, W, params_px, rng)

        zone = {
            "extent_axis": axis,
            "start": start,
            "end": end,
            "pattern": pattern,
            "thickness_um": thickness_um,
            "spacing_um": spacing_um,
            "thickness_px": thickness_px,
            "spacing_px": spacing_px,
            "grayscale_mode": grayscale_mode,
        }

        zone_mask = apply_zone_extent(full_mask, H, W, zone)
        zone_gray = apply_grayscale(zone_mask, mode=grayscale_mode)

        bone = zone_mask > 0
        mask_hetero[bone] = 1
        gray_hetero[bone] = zone_gray[bone]

        zones_meta.append(zone)

    return mask_hetero, gray_hetero, zones_meta


# -----------------------------------------------------------
# CSV helpers
# -----------------------------------------------------------
def init_csv(path, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
    return f, writer


# -----------------------------------------------------------
# CLI / main
# -----------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="v2 heterogeneous synthetic trabecular generator with curved patterns."
    )
    p.add_argument("--outdir", type=str, default="data/hetero_v2",
                   help="Output directory root.")
    p.add_argument("--size", type=int, default=512,
                   help="Image size H=W.")
    p.add_argument("--slices", type=int, default=2,
                   help="Number of slices in TIFF stack.")
    p.add_argument("--z-step-um", type=float, default=Z_STEP_UM,
                   help="Z step (micron) between slices.")
    p.add_argument("--n-images", type=int, default=50,
                   help="How many heterogeneous images to generate.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility.")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    H = W = args.size
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # CSVs
    images_csv_path = outdir / "images.csv"
    zones_csv_path = outdir / "zones.csv"

    img_f, img_writer = init_csv(
        images_csv_path,
        fieldnames=[
            "image_id",
            "filename_mask",
            "filename_gray",
            "filename_stack",
            "pixel_size_um",
            "z_step_um",
            "n_zones",
        ],
    )
    zone_f, zone_writer = init_csv(
        zones_csv_path,
        fieldnames=[
            "image_id",
            "zone_id",
            "extent_axis",
            "start",
            "end",
            "pattern",
            "thickness_um",
            "spacing_um",
            "thickness_px",
            "spacing_px",
            "grayscale_mode",
        ],
    )

    patterns = ["grid", "vertical", "horizontal", "sinusoidal", "radial"]

    try:
        for i in range(args.n_images):
            image_id = f"img_{i:04d}"
            mask, gray, zones_meta = generate_heterogeneous_image(H, W, rng, patterns)

            # Paths
            mask_path = outdir / f"{image_id}_mask.png"
            gray_path = outdir / f"{image_id}_gray.png"
            stack_path = outdir / f"{image_id}_stack.tif"

            # Save images
            save_png((mask * 255).astype(np.uint8), mask_path)
            save_png(gray, gray_path)
            save_stack(mask, stack_path, n_slices=args.slices, z_step_um=args.z_step_um)

            # Image-level metadata
            img_writer.writerow(
                {
                    "image_id": image_id,
                    "filename_mask": mask_path.name,
                    "filename_gray": gray_path.name,
                    "filename_stack": stack_path.name,
                    "pixel_size_um": PIXEL_SIZE_UM,
                    "z_step_um": args.z_step_um,
                    "n_zones": len(zones_meta),
                }
            )

            # Zone-level metadata
            for zi, z in enumerate(zones_meta):
                zone_writer.writerow(
                    {
                        "image_id": image_id,
                        "zone_id": zi,
                        "extent_axis": z["extent_axis"],
                        "start": z["start"],
                        "end": z["end"],
                        "pattern": z["pattern"],
                        "thickness_um": z["thickness_um"],
                        "spacing_um": z["spacing_um"],
                        "thickness_px": z["thickness_px"],
                        "spacing_px": z["spacing_px"],
                        "grayscale_mode": z["grayscale_mode"],
                    }
                )

            print(f"[{i+1}/{args.n_images}] Saved {image_id}")

    finally:
        img_f.close()
        zone_f.close()

    print(f"\nDone. Images + metadata written under: {outdir}")


if __name__ == "__main__":
    main()
