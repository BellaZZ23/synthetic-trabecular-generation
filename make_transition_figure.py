#!/usr/bin/env python3
"""
generate_transition_figure_v2.py

Generates Figure 1 for the Quantum Kernel SVM paper.
  - Row 1: 2D grayscale mid-slices (as before)
  - Row 2: 3D volumetric isosurface renderings of the trabecular bone mask

Requires: numpy, matplotlib, Pillow, scikit-image (for marching_cubes)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── optional: 3-D rendering ──────────────────
try:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    HAS_3D = True
except ImportError:
    HAS_3D = False
    print(
        "WARNING: scikit-image not found. "
        "Install it (`pip install scikit-image`) for 3-D volumetric views."
    )


# ─────────────────────────────────────────────
# USER SETTINGS
# ─────────────────────────────────────────────

GENERATOR = Path("synthetic_trabecular_v15_morphometric_control.py")

VOI_DIRS = [
    Path(r"data\derived\VOI1"),
    Path(r"data\derived\VOI4"),
]

OUTROOT = Path(r"output\fig_transition_final")

BVTV_VALUES = [0.30, 0.36, 0.42, 0.48]

COMMON_ARGS = [
    "--num-samples", "1",
    "--xy", "128",
    "--z", "40",
    "--voxel-um", "39",
    "--tbth-um", "180",
    "--base-sigma", "3.0",
    "--aniso-ratio", "1.0",
    "--warp-amp", "1.5",
    "--warp-sigma", "10.0",
    "--plate-weight", "0.7",
    "--rod-weight", "0.3",
    "--proto-close-iters", "1",
    "--proto-open-iters", "0",
    "--proto-min-component", "400",
    "--min-component-size", "0",
    "--round-sigma", "0.0",
    "--marrow-mean", "20",
    "--bone-mean", "95",
    "--solid-fill-sigma", "1.8",
    "--noise-sd", "3.0",
    "--bg-tex-sd", "1.0",
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def run_generator(bvtv: float, seed: int, outdir: Path) -> None:
    cmd = [
        sys.executable,
        str(GENERATOR),
        "--voi-dirs",
        *[str(p) for p in VOI_DIRS],
        "--outdir",
        str(outdir),
        "--bvtv",
        str(bvtv),
        "--base-seed",
        str(seed),
        *COMMON_ARGS,
    ]
    print("\nRunning command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def read_measured_bvtv(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("morphometrics", {}).get("BVTV")


def find_volume(sample_dir: Path) -> np.ndarray | None:
    """
    Locate and load the 3-D binary mask volume from the generator output.

    Adapt the filename / loading logic below to match whatever your
    generator actually writes.  Common patterns:
        mask.npy, bone_mask.npy, volume.npy, gray_volume.npy
    """
    # ---- try .npy masks first ------------------------------------------
    for name in ("mask.npy", "bone_mask.npy", "binary_mask.npy", "volume.npy"):
        p = sample_dir / name
        if p.exists():
            vol = np.load(p)
            # If it's a grayscale volume, threshold it to get a binary mask
            if vol.dtype != bool:
                threshold = 0.5 * (vol.max() + vol.min())
                vol = vol > threshold
            return vol

    # ---- try grayscale volume and threshold -----------------------------
    for name in ("gray_volume.npy", "grayscale.npy"):
        p = sample_dir / name
        if p.exists():
            gray = np.load(p)
            threshold = 0.5 * (gray.max() + gray.min())
            return gray > threshold

    # ---- try multi-page TIFF mask ---------------------------------------
    for name in ("mask.tif", "mask.tiff"):
        p = sample_dir / name
        if p.exists():
            img = Image.open(p)
            slices = []
            for i in range(img.n_frames):
                img.seek(i)
                slices.append(np.array(img))
            vol = np.stack(slices, axis=0)
            if vol.dtype != bool:
                vol = vol > 0
            return vol

    # ---- try a stack of slice images ------------------------------------
    slice_files = sorted(sample_dir.glob("slice_*.png"))
    if not slice_files:
        slice_files = sorted(sample_dir.glob("gray_z*.png"))
    if slice_files:
        slices = [np.array(Image.open(f).convert("L")) for f in slice_files]
        gray = np.stack(slices, axis=0)  # shape (Z, H, W)
        threshold = 0.5 * (gray.max() + gray.min())
        return gray > threshold

    return None


def render_volume_to_image(
    mask: np.ndarray,
    voxel_size_um: float = 39.0,
    step_size: int = 2,
    elev: float = 30,
    azim: float = -55,
    z_scale: float = 2.5,
    img_size: tuple[int, int] = (400, 400),
) -> np.ndarray:
    """
    Render a 3-D isosurface of the binary bone mask and return as an RGBA image.

    Renders to an off-screen figure, reads back as a numpy array, and closes
    the figure — avoids matplotlib 3-D bbox bugs in the final composite.
    """
    import io

    verts, faces, normals, _ = marching_cubes(
        mask.astype(float),
        level=0.5,
        step_size=step_size,
        allow_degenerate=False,
    )

    # Scale vertices to physical units (µm → mm)
    verts_mm = verts * voxel_size_um / 1000.0
    # Stretch z so the thin slab doesn't look flat
    verts_mm[:, 0] *= z_scale

    # ── Directional shading ───────────────────────────────────────
    light_dir = np.array([0.3, 0.5, 1.0])
    light_dir /= np.linalg.norm(light_dir)

    face_normals = normals[faces].mean(axis=1)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    face_normals /= norms

    shade = np.abs(face_normals @ light_dir)
    shade = 0.35 + 0.65 * shade

    base = np.array([0.92, 0.89, 0.83])
    face_colors = np.outer(shade, base)
    face_colors = np.clip(face_colors, 0, 1)
    face_colors = np.column_stack([face_colors, np.full(len(shade), 0.85)])

    mesh = Poly3DCollection(
        verts_mm[faces],
        linewidths=0,
        antialiased=False,
    )
    mesh.set_facecolor(face_colors)

    # ── Render to off-screen figure ───────────────────────────────
    dpi = 100
    w_in = img_size[0] / dpi
    h_in = img_size[1] / dpi
    tmp_fig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
    ax = tmp_fig.add_subplot(111, projection="3d")

    ax.add_collection3d(mesh)
    for setter, idx in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
        setter(verts_mm[:, idx].min(), verts_mm[:, idx].max())
    ax.view_init(elev=elev, azim=azim)
    ax.axis("off")

    tmp_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    tmp_fig.savefig(buf, format="png", dpi=dpi, transparent=True)
    plt.close(tmp_fig)

    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGBA"))
    buf.close()
    return img


# ─────────────────────────────────────────────
# FIGURE BUILDER (2-ROW VERSION)
# ─────────────────────────────────────────────

def build_figure(
    image_info: list[tuple[Path, str, float, float | None, Path]],
    outpath: Path,
) -> None:
    """
    Build a two-row figure:
      Row 1 – 2-D grayscale mid-slices
      Row 2 – 3-D volumetric isosurface renderings

    Parameters
    ----------
    image_info : list of (img_path, panel_label, target_bvtv,
                           measured_bvtv, sample_dir)
    outpath : where to save the figure
    """
    n = len(image_info)
    has_volumes = HAS_3D  # only render 3-D row if scikit-image is available

    if has_volumes:
        # Check that at least one volume file actually exists
        volumes = []
        for _, _, _, _, sample_dir in image_info:
            vol = find_volume(sample_dir)
            volumes.append(vol)
        has_volumes = any(v is not None for v in volumes)
    else:
        volumes = [None] * n

    nrows = 2 if has_volumes else 1
    fig = plt.figure(figsize=(12, 3.2 * nrows + 0.6))

    # ── Row 1: 2-D mid-slices ─────────────────────────────────────────
    for i, (img_path, panel_label, target_bvtv, measured_bvtv, _) in enumerate(
        image_info
    ):
        ax = fig.add_subplot(nrows, n, i + 1)
        img = Image.open(img_path).convert("L")
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)

        if measured_bvtv is None:
            title = f"{panel_label}\nBV/TV={target_bvtv:.2f}"
        else:
            title = f"{panel_label}\nBV/TV={target_bvtv:.2f}\n(meas. {measured_bvtv:.2f})"

        ax.set_title(title, fontsize=11)
        ax.axis("off")

    # ── Row 2: 3-D volumetric views (rendered to images) ────────────
    if has_volumes:
        for i, (_, panel_label, target_bvtv, _, sample_dir) in enumerate(
            image_info
        ):
            ax = fig.add_subplot(nrows, n, n + i + 1)

            vol = volumes[i]
            if vol is not None:
                vol_img = render_volume_to_image(vol)
                ax.imshow(vol_img)
            else:
                ax.text(
                    0.5, 0.5, "volume\nnot found",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=9, color="gray",
                )
            # Sequential label: A→E, B→F, C→G, D→H
            vol_label = chr(ord(panel_label) + n)
            ax.set_title(vol_label, fontsize=11)
            ax.axis("off")

    plt.subplots_adjust(wspace=0.02, hspace=0.12, bottom=0.10)

    # ── Caption text at the bottom ────────────────────────────────────
    fig.text(
        0.5,
        0.01,
        "Synthetic trabecular bone generated with increasing target BV/TV "
        "(0.30\u20130.48) at 39 \u03bcm isotropic voxel resolution.\n"
        "(A\u2013D) Grayscale mid-slice cross-sections. "
        "(E\u2013H) Three-dimensional isosurface renderings of the corresponding volumes.",
        ha="center",
        fontsize=9,
        style="italic",
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure to: {outpath.resolve()}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    if not GENERATOR.exists():
        raise FileNotFoundError(
            f"Could not find generator script: {GENERATOR.resolve()}\n"
            "Set GENERATOR to the correct path."
        )

    for voi_dir in VOI_DIRS:
        if not voi_dir.exists():
            raise FileNotFoundError(
                f"Could not find VOI directory: {voi_dir.resolve()}\n"
                "Update VOI_DIRS in the script."
            )

    OUTROOT.mkdir(parents=True, exist_ok=True)

    # image_info now carries the sample directory as well
    image_info: list[tuple[Path, str, float, float | None, Path]] = []

    for i, bvtv in enumerate(BVTV_VALUES):
        panel_label = chr(65 + i)  # A, B, C, D  (adjust if panel A is real bone)
        sample_outdir = OUTROOT / panel_label
        seed = 101 + i

        run_generator(bvtv=bvtv, seed=seed, outdir=sample_outdir)

        sample_dir = sample_outdir / "sample_000"
        img_path = sample_dir / "gray_mid.png"
        metrics_path = sample_dir / "metrics.json"

        if not img_path.exists():
            raise FileNotFoundError(f"Expected image not found: {img_path}")

        measured_bvtv = read_measured_bvtv(metrics_path)
        image_info.append((img_path, panel_label, bvtv, measured_bvtv, sample_dir))

    figure_path = OUTROOT / "transition_figure_v2.png"
    build_figure(image_info, figure_path)


if __name__ == "__main__":
    main()