#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


# ─────────────────────────────────────────────
# USER SETTINGS
# ─────────────────────────────────────────────

# Path to your generator script
GENERATOR = Path("synthetic_trabecular_v15_morphometric_control.py")

# VOI folders used by your generator
VOI_DIRS = [
    Path(r"data\derived\VOI1"),
    Path(r"data\derived\VOI4"),
]

# Output folder for the figure and generated samples
OUTROOT = Path(r"output\fig_transition_final")

# Morphology transition values
BVTV_VALUES = [0.30, 0.36, 0.42, 0.48]

# Best parameters from your tuned run
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


def build_figure(image_info: list[tuple[Path, str, float, float | None]], outpath: Path) -> None:
    fig, axes = plt.subplots(1, len(image_info), figsize=(12, 3.6))
    if len(image_info) == 1:
        axes = [axes]

    for ax, (img_path, panel_label, target_bvtv, measured_bvtv) in zip(axes, image_info):
        img = Image.open(img_path).convert("L")
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)

        if measured_bvtv is None:
            title = f"{panel_label}\nBV/TV={target_bvtv:.2f}"
        else:
            title = f"{panel_label}\nBV/TV={target_bvtv:.2f}\n(meas. {measured_bvtv:.2f})"

        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.subplots_adjust(wspace=0.06, bottom=0.22)

    fig.text(
        0.5,
        0.08,
        "Synthetic trabecular bone slices generated with increasing target BV/TV. "
        "All images are shown at 39 μm voxel resolution. Higher BV/TV produces "
        "progressively denser plate-dominant architecture.",
        ha="center",
        fontsize=10,
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


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

    image_info: list[tuple[Path, str, float, float | None]] = []

    for i, bvtv in enumerate(BVTV_VALUES):
        panel_label = chr(65 + i)  # A, B, C, D
        sample_outdir = OUTROOT / panel_label
        seed = 101 + i

        run_generator(bvtv=bvtv, seed=seed, outdir=sample_outdir)

        img_path = sample_outdir / "sample_000" / "gray_mid.png"
        metrics_path = sample_outdir / "sample_000" / "metrics.json"

        if not img_path.exists():
            raise FileNotFoundError(f"Expected image not found: {img_path}")

        measured_bvtv = read_measured_bvtv(metrics_path)
        image_info.append((img_path, panel_label, bvtv, measured_bvtv))

    figure_path = OUTROOT / "transition_figure.png"
    build_figure(image_info, figure_path)

    print(f"\nSaved figure to:\n{figure_path.resolve()}")


if __name__ == "__main__":
    main()