import numpy as np, json, shutil
from pathlib import Path
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

dataset_dir = Path('output/final_dataset/dataset')
samples = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and (d/'gray.tif').exists()])

# Load mid-slices
import tifffile
X, paths = [], []
for d in samples:
    vol = tifffile.imread(str(d/'gray.tif')).astype(np.float32)
    vol /= max(vol.max(), 1)
    mid = vol[vol.shape[0]//2]
    img = np.array(Image.fromarray((mid*255).astype(np.uint8)).resize((64,64)))/255.0
    X.append(img.ravel())
    paths.append(d)

X = np.array(X)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
pca = PCA(n_components=2)
Z = pca.fit_transform(Xs)

# Find outliers: distance from centroid > 2 std
centroid = Z.mean(axis=0)
dists = np.sqrt(((Z - centroid)**2).sum(axis=1))
threshold = dists.mean() + 2.0 * dists.std()

outlier_dir = dataset_dir.parent / 'outliers_pca'
outlier_dir.mkdir(exist_ok=True)
removed = 0
for i, (d, dist) in enumerate(zip(paths, dists)):
    if dist > threshold:
        print(f'  Removing {d.name}: PCA distance={dist:.1f} (threshold={threshold:.1f})')
        shutil.move(str(d), str(outlier_dir / d.name))
        removed += 1

print(f'\nRemoved {removed} PCA outliers. Remaining: {len(paths)-removed} samples')
