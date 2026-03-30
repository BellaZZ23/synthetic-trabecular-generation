#!/usr/bin/env python3
import argparse, json, shutil
import numpy as np
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=str, default="output/final_dataset/dataset")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir)
    samples = []
    for path in sorted(dataset_dir.glob("sample_*/metrics.json")):
        m = json.load(open(path))
        morph = m["morphometrics"]
        samples.append({
            "dir": path.parent,
            "BVTV": morph["BVTV"],
            "TbTh": morph["TbTh_um_p50"],
            "TbN":  morph["TbN_per_mm"],
            "TbSp": morph["TbSp_um_p50"],
        })

    # Compute bounds (2 std) per metric
    for key in ["BVTV", "TbTh", "TbN", "TbSp"]:
        vals = np.array([s[key] for s in samples])
        mean, std = vals.mean(), vals.std()
        for s in samples:
            if abs(s[key] - mean) > 2 * std:
                s["outlier"] = True

    keep   = [s for s in samples if not s.get("outlier", False)]
    remove = [s for s in samples if s.get("outlier", False)]

    print(f"Total: {len(samples)}, Keep: {len(keep)}, Remove: {len(remove)}")
    for s in remove:
        print(f"  Removing {s['dir'].name}: BV/TV={s['BVTV']:.3f} "
              f"TbTh={s['TbTh']:.0f} TbN={s['TbN']:.2f} TbSp={s['TbSp']:.0f}")
        dest = dataset_dir.parent / "outliers" / s["dir"].name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s["dir"]), str(dest))

    print(f"\nCleaned dataset: {len(keep)} samples in {dataset_dir}")

if __name__ == "__main__":
    main()