#!/usr/bin/env python3
"""
make_demo_data.py
=================
Create a small, committable copy of one D2IM specimen so the cloud demo
has data without shipping the full-size volumes.

Run ONCE from the repo root (the folder containing app.py):

    python make_demo_data.py

It reads the full-size processed files from data/strain/processed/ and
writes downsampled copies (same filenames) into data/demo/. Your originals
are NOT touched. Then commit data/demo/ (see the git steps in chat).

Adjust FACTOR if the files are still too big/small. FACTOR=2 keeps every
2nd voxel on each axis (1/8 the size); FACTOR=1 copies as-is.
"""
import numpy as np
from pathlib import Path

SPEC = "S9_INT_UL_AP_50"          # the AP specimen from your processed folder
FACTOR = 2                         # downsample factor per axis (2 -> 1/8 size)
KINDS = ["reference_scan", "bone_mask", "displacement_magnitude"]

SRC = Path("data/strain/processed")
OUT = Path("data/demo")
OUT.mkdir(parents=True, exist_ok=True)

for kind in KINDS:
    src = SRC / f"{kind}_{SPEC}.npy"
    if not src.exists():
        print(f"  MISSING: {src} — check the path/specimen name")
        continue
    a = np.load(src)
    small = a[::FACTOR, ::FACTOR, ::FACTOR] if a.ndim == 3 else a[::FACTOR]
    dst = OUT / f"{kind}_{SPEC}.npy"
    np.save(dst, small)
    print(f"  {kind:24s} {a.shape} -> {small.shape}  "
          f"({small.nbytes / 1e6:.1f} MB)  -> {dst}")

print("\nDone. Commit with:")
print(f"  git add -f data/demo/*_{SPEC}.npy")