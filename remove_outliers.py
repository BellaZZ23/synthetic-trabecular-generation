import json, glob, numpy as np, shutil
from pathlib import Path

dataset_dir = Path('output/final_dataset/dataset')
samples = []
for p in sorted(dataset_dir.glob('sample_*/metrics.json')):
    m = json.load(open(p))
    morph = m['morphometrics']
    samples.append({
        'dir': p.parent,
        'BVTV': morph['BVTV'],
        'TbTh': morph['TbTh_um_p50'],
        'TbN': morph['TbN_per_mm'],
        'TbSp': morph['TbSp_um_p50'],
    })

# Compute bounds (2 std)
for key in ['BVTV', 'TbTh', 'TbN', 'TbSp']:
    vals = np.array([s[key] for s in samples])
    mean, std = vals.mean(), vals.std()
    for s in samples:
        if abs(s[key] - mean) > 2 * std:
            s['outlier'] = True

keep = [s for s in samples if not s.get('outlier', False)]
remove = [s for s in samples if s.get('outlier', False)]

print(f'Total: {len(samples)}, Keep: {len(keep)}, Remove: {len(remove)}')
for s in remove:
    print(f"  Removing {s['dir'].name}: BV/TV={s['BVTV']:.3f} TbTh={s['TbTh']:.0f} TbN={s['TbN']:.2f} TbSp={s['TbSp']:.0f}")
    # Move to outliers folder instead of deleting
    dest = dataset_dir.parent / 'outliers' / s['dir'].name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(s['dir']), str(dest))

print(f'\nCleaned dataset: {len(keep)} samples in {dataset_dir}')
