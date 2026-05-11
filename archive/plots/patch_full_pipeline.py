#!/usr/bin/env python3
"""
patch_full_pipeline.py
Applies Fix 2 (wider BV/TV and Tb.Th spread) to full_pipeline.py in-place.
Run once from your project root:
    python patch_full_pipeline.py
"""
import re, sys
from pathlib import Path

target = Path("full_pipeline.py")
if not target.exists():
    sys.exit(f"ERROR: {target} not found. Run from your project root.")

src = target.read_text(encoding="utf-8")
original = src

# Fix 2a: wider BV/TV spread  0.04 → 0.06, clamp [0.15,0.30] → [0.12,0.32]
src = src.replace(
    "bvtv = float(np.clip(rng_sample.normal(bvtv_centre, 0.04), 0.15, 0.30))",
    "bvtv = float(np.clip(rng_sample.normal(bvtv_centre, 0.06), 0.12, 0.32))  # FIX2: wider spread"
)

# Fix 2b: wider Tb.Th spread  25.0 → 30.0, clamp [130,240] → [110,260]
src = src.replace(
    "tbth = float(np.clip(rng_sample.normal(tbth_centre, 25.0), 130.0, 240.0))",
    "tbth = float(np.clip(rng_sample.normal(tbth_centre, 30.0), 110.0, 260.0))  # FIX2: wider spread"
)

if src == original:
    print("WARNING: no replacements made — check that full_pipeline.py matches expected lines.")
    print("  Expected (BV/TV line):  bvtv = float(np.clip(rng_sample.normal(bvtv_centre, 0.04), 0.15, 0.30))")
    print("  Expected (Tb.Th line):  tbth = float(np.clip(rng_sample.normal(tbth_centre, 25.0), 130.0, 240.0))")
    sys.exit(1)

target.write_text(src, encoding="utf-8")

# Verify
changed_lines = [l for l in src.splitlines() if "FIX2" in l]
print(f"Patched {target}. Changed lines:")
for l in changed_lines:
    print(f"  {l.strip()}")
print("\nDone. You can now regenerate samples or re-run full_pipeline.py.")