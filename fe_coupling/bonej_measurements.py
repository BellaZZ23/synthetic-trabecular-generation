"""
BoneJ-equivalent Morphometric Measurements
============================================
Python implementations of core BoneJ algorithms.
Reference: Doube et al. (2010) BoneJ. Bone 47(6):1076-9
"""
import numpy as np
import time
from scipy.ndimage import distance_transform_edt, label, generate_binary_structure


def measure_thickness(mask, voxel_um, is_bone=True):
    """Local thickness via distance transform (Hildebrand & Rüegsegger 1997)."""
    target = mask.astype(np.uint8) if is_bone else (mask == 0).astype(np.uint8)
    if target.sum() == 0:
        return {k: 0.0 for k in ['p10','p25','p50','p75','p90','mean','std','max','map']}

    edt = distance_transform_edt(target) * voxel_um
    thickness_map = 2.0 * edt
    values = thickness_map[target > 0]
    values = values[values > 0]

    if len(values) == 0:
        return {k: 0.0 for k in ['p10','p25','p50','p75','p90','mean','std','max','map']}

    return {
        'p10': float(np.percentile(values, 10)),
        'p25': float(np.percentile(values, 25)),
        'p50': float(np.percentile(values, 50)),
        'p75': float(np.percentile(values, 75)),
        'p90': float(np.percentile(values, 90)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'max': float(np.max(values)),
        'map': thickness_map,
    }


def measure_connectivity(mask):
    """Connectivity via Euler number (BoneJ Connectivity plugin)."""
    struct26 = generate_binary_structure(3, 3)
    labeled, n_components = label(mask, structure=struct26)

    try:
        from skimage.measure import euler_number as sk_euler
        euler = sk_euler(mask.astype(bool), connectivity=3)
    except ImportError:
        euler = n_components

    connectivity = 1 - euler
    bone_volume = mask.sum()

    if n_components > 0 and bone_volume > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        lcc_frac = float(sizes.max()) / float(bone_volume)
    else:
        lcc_frac = 0.0

    return {
        'euler_number': int(euler),
        'connectivity': int(connectivity),
        'connectivity_density': float(connectivity / mask.size) if mask.size > 0 else 0,
        'n_components': int(n_components),
        'lcc_fraction': float(lcc_frac),
    }


def measure_anisotropy(mask, voxel_um, n_directions=256, n_lines=80):
    """Degree of Anisotropy via Mean Intercept Length."""
    nz, ny, nx = mask.shape
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n_directions)
    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_directions)
    directions = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])

    rng = np.random.default_rng(42)
    mil_values = np.zeros(n_directions)

    for d_idx, direction in enumerate(directions):
        total_length = 0
        total_intercepts = 0
        for _ in range(n_lines):
            start = np.array([rng.integers(0, nz), rng.integers(0, ny), rng.integers(0, nx)], dtype=float)
            prev_val = 0
            intercepts = 0
            length = 0
            pos = start.copy()
            for _ in range(max(nz, ny, nx) * 2):
                pos += direction
                iz, iy, ix = int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))
                if not (0 <= iz < nz and 0 <= iy < ny and 0 <= ix < nx):
                    break
                curr_val = mask[iz, iy, ix]
                if curr_val != prev_val:
                    intercepts += 1
                prev_val = curr_val
                length += 1
            if intercepts > 0:
                total_length += length * voxel_um
                total_intercepts += intercepts
        mil_values[d_idx] = total_length / max(total_intercepts, 1)

    M = np.zeros((3, 3))
    for d_idx, direction in enumerate(directions):
        M += mil_values[d_idx] * np.outer(direction, direction)
    M /= n_directions

    eigenvalues = np.sort(np.linalg.eigvalsh(M))[::-1]
    DA = 1.0 - eigenvalues[-1] / eigenvalues[0] if eigenvalues[0] > 0 else 0

    return {'DA': float(DA), 'eigenvalues': eigenvalues.tolist()}


def measure_smi(mask, voxel_um):
    """Structure Model Index: 0=plates, 3=rods, 4=spheres."""
    from skimage.measure import marching_cubes
    from scipy.ndimage import binary_dilation

    spacing = (voxel_um, voxel_um, voxel_um)
    V = float(mask.sum()) * voxel_um ** 3

    try:
        verts, faces, _, _ = marching_cubes(mask.astype(float), 0.5, spacing=spacing)
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        S = float(np.sum(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)))
    except Exception:
        return {'SMI': float('nan'), 'surface_area_um2': 0, 'bone_volume_um3': V}

    try:
        dilated = binary_dilation(mask).astype(float)
        verts_d, faces_d, _, _ = marching_cubes(dilated, 0.5, spacing=spacing)
        v0d, v1d, v2d = verts_d[faces_d[:, 0]], verts_d[faces_d[:, 1]], verts_d[faces_d[:, 2]]
        S_d = float(np.sum(0.5 * np.linalg.norm(np.cross(v1d - v0d, v2d - v0d), axis=1)))
    except Exception:
        return {'SMI': float('nan'), 'surface_area_um2': S, 'bone_volume_um3': V}

    dr = voxel_um
    dS = S_d - S
    SMI = 6 * V * dS / (dr * S ** 2) if S > 0 else float('nan')

    return {'SMI': float(SMI), 'surface_area_um2': S, 'bone_volume_um3': V}


def measure_all_bonej(mask, voxel_um, include_anisotropy=False,
                       include_smi=False, verbose=True):
    """Run all BoneJ-equivalent measurements."""
    t0 = time.time()

    bvtv = float(mask.sum()) / float(mask.size)
    tbth = measure_thickness(mask, voxel_um, is_bone=True)
    tbsp = measure_thickness(mask, voxel_um, is_bone=False)
    tbn = bvtv / (tbth['p50'] / 1000.0) if tbth['p50'] > 0 else 0
    conn = measure_connectivity(mask)

    results = {
        'BVTV': bvtv,
        'TbTh_um_p50': tbth['p50'],
        'TbTh_um_p90': tbth['p90'],
        'TbTh_um_mean': tbth['mean'],
        'TbSp_um_p50': tbsp['p50'],
        'TbSp_um_p90': tbsp['p90'],
        'TbSp_um_mean': tbsp['mean'],
        'TbN_per_mm': tbn,
        'Euler': conn['euler_number'],
        'connectivity': conn['connectivity'],
        'connectivity_density': conn['connectivity_density'],
        'n_components': conn['n_components'],
        'lcc_frac': conn['lcc_fraction'],
        'thickness_map': tbth['map'],
        'spacing_map': tbsp['map'],
    }

    if include_anisotropy:
        aniso = measure_anisotropy(mask, voxel_um)
        results['DA'] = aniso['DA']
        results['MIL_eigenvalues'] = aniso['eigenvalues']

    if include_smi:
        smi = measure_smi(mask, voxel_um)
        results['SMI'] = smi['SMI']
        results['surface_area_um2'] = smi['surface_area_um2']

    if verbose:
        print(f"BoneJ measurements: {time.time()-t0:.1f}s | "
              f"BV/TV={bvtv:.3f}, Tb.Th={tbth['p50']:.0f}um, Tb.N={tbn:.2f}/mm")

    return results