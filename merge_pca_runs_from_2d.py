#!/usr/bin/env python3
"""
merge_pca_runs_from_2d.py

Merges many generator run folders (each containing volumes.csv + 2D PNGs)
into one PCA dataset: X.npy + metadata.csv
"""

from pathlib import Path
import argparse
import csv
import numpy as np
from PIL import Image


def read_csv_rows(csv_path: Path):
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def load_png_flat(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    return arr.reshape(-1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=str, default="data/v2_sweep_big",
                    help="Directory containing run_XXX subfolders.")
    ap.add_argument("--outdir", type=str, default="data/pca_sweep_big",
                    help="Output directory for X.npy + metadata.csv")
    ap.add_argument("--use", type=str, default="mip",
                    choices=["mid", "mip", "mean", "gray_mid", "gray_mip"],
                    help="Which 2D representation to use.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional cap on samples (0 = no cap).")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    key_map = {
        "mid": "mid_png",
        "mip": "mip_png",
        "mean": "mean_png",
        "gray_mid": "gray_mid_png",
        "gray_mip": "gray_mip_png",
    }
    col = key_map[args.use]

    X_list = []
    meta_out = []

    run_dirs = sorted([p for p in sweep_dir.glob("run_*") if p.is_dir()])
    if not run_dirs:
        raise SystemExit(f"No run_* folders found under {sweep_dir}")

    for run_dir in run_dirs:
        csv_path = run_dir / "volumes.csv"
        if not csv_path.exists():
            continue

        rows = read_csv_rows(csv_path)
        for r in rows:
            rel = (r.get(col) or "").strip()
            if not rel:
                continue

            img_path = run_dir / rel
            if not img_path.exists():
                continue

            X_list.append(load_png_flat(img_path))

            r2 = dict(r)
            r2["run_dir"] = run_dir.name
            r2["pca_input_mode"] = args.use
            r2["pca_input_file"] = rel
            meta_out.append(r2)

            if args.limit and len(X_list) >= args.limit:
                break

        if args.limit and len(X_list) >= args.limit:
            break

    if not X_list:
        raise SystemExit("No samples collected. Did you run the sweep with --export-2d --export-2d-mode mip ?")

    X = np.stack(X_list, axis=0)
    np.save(outdir / "X.npy", X)

    meta_path = outdir / "metadata.csv"
    fieldnames = list(meta_out[0].keys())
    with open(meta_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in meta_out:
            w.writerow(r)

    print(f"Saved merged PCA dataset to: {outdir}")
    print(f"  X.npy shape = {X.shape}")
    print(f"  metadata.csv rows = {len(meta_out)}")


if __name__ == "__main__":
    main()
