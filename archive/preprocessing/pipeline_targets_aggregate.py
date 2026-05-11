#!/usr/bin/env python3
"""
pipeline_targets_aggregate.py

Combine multiple *_targets.json files into one priors file.

Used for Pipeline A:
VOI1 + VOI4 -> structural priors for generator calibration

Example:
python pipeline_targets_aggregate.py `
  --inputs data\derived\VOI1 data\derived\VOI4 `
  --output data\derived\priors\VOI1_VOI4_priors.json
"""

import argparse
import json
from pathlib import Path
import numpy as np


# -----------------------------
# Fields we want to aggregate
# -----------------------------
FIELDS = [
    "BVTV",
    "tbth_um_p50",
    "tbth_um_p90",
    "tbsp_um_p50",
    "tbsp_um_p90",
    "euler",
    "conn_proxy",
]


def load_targets(folder: Path):
    """Load all *_targets.json from a folder."""
    values = {f: [] for f in FIELDS}

    json_files = list(folder.glob("*_targets.json"))
    if len(json_files) == 0:
        print(f"Warning: no targets found in {folder}")
        return values

    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        for field in FIELDS:
            if field in data:
                values[field].append(data[field])

    return values


def merge_values(list_of_dicts):
    """Merge field lists across multiple folders."""
    merged = {f: [] for f in FIELDS}

    for d in list_of_dicts:
        for f in FIELDS:
            merged[f].extend(d[f])

    return merged


def compute_priors(values):
    """
    Compute robust priors.
    Median is better than mean for structural metrics.
    """
    priors = {}

    for field, arr in values.items():
        if len(arr) == 0:
            continue

        arr = np.array(arr, dtype=float)
        priors[field] = float(np.median(arr))
        priors[field + "_mean"] = float(np.mean(arr))
        priors[field + "_std"] = float(np.std(arr))

    return priors


def main():
    parser = argparse.ArgumentParser(description="Aggregate VOI targets into priors")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input derived folders (e.g. data/derived/VOI1 data/derived/VOI4)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output priors JSON file",
    )

    args = parser.parse_args()

    input_folders = [Path(p) for p in args.inputs]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading targets from:")
    for p in input_folders:
        print("  ", p)

    all_values = []
    for folder in input_folders:
        all_values.append(load_targets(folder))

    merged = merge_values(all_values)
    priors = compute_priors(merged)

    with open(output_path, "w") as f:
        json.dump(priors, f, indent=2)

    print("\nSaved priors to:", output_path)
    print("\nPriors summary:")
    for k, v in priors.items():
        if not k.endswith("_std") and not k.endswith("_mean"):
            print(f"{k:15s}: {v:.3f}")


if __name__ == "__main__":
    main()
