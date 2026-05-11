"""
Pipeline figure visuals for boxes 2, 4, 5, 6, 7.
Generates publication-ready PNGs with transparent backgrounds.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, distance_transform_edt, label
import matplotlib.patches as patches

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

OUT = 'paper/figures/pipeline'

# ── Shared: generate a synthetic bone slice ──────────────────────────
def make_bone_slice(size=128, seed=42, bvtv_target=0.30):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((size, size))
    field = gaussian_filter(noise, sigma=3.0)
    
    # Elastic warp for irregularity
    warp_x = gaussian_filter(rng.standard_normal((size, size)), sigma=10) * 1.5
    warp_y = gaussian_filter(rng.standard_normal((size, size)), sigma=10) * 1.5
    y_grid, x_grid = np.mgrid[0:size, 0:size]
    x_warped = np.clip((x_grid + warp_x).astype(int), 0, size - 1)
    y_warped = np.clip((y_grid + warp_y).astype(int), 0, size - 1)
    field = field[y_warped, x_warped]
    
    # Zero-crossing wall detection — calibrate threshold for target BV/TV
    lo, hi = 0.05, 1.5
    for _ in range(30):
        tau = (lo + hi) / 2
        mask = np.abs(field) < tau
        current_bvtv = mask.mean()
        if current_bvtv < bvtv_target:
            lo = tau
        else:
            hi = tau
    bone_mask = np.abs(field) < tau
    
    # Convert to greyscale
    dist = distance_transform_edt(bone_mask)
    greyscale = np.where(bone_mask, 160 + 60 * np.clip(dist / 3, 0, 1), 30)
    greyscale += rng.normal(0, 5, greyscale.shape)
    greyscale = gaussian_filter(greyscale, sigma=0.8)
    return np.clip(greyscale, 0, 255), bone_mask


# ═══════════════════════════════════════════════════════════════════════
# BOX 2: Sub-volume extraction
# ═══════════════════════════════════════════════════════════════════════
def make_subvolume_figure():
    fig, ax = plt.subplots(figsize=(5, 5))
    bone_slice, _ = make_bone_slice(128, seed=42, bvtv_target=0.30)
    ax.imshow(bone_slice, cmap='bone', origin='lower', vmin=0, vmax=255)
    
    outer = patches.Rectangle((3, 3), 122, 122, linewidth=1.5,
                               edgecolor='#4FC3F7', facecolor='none', linestyle='--')
    ax.add_patch(outer)
    
    inner = patches.Rectangle((28, 28), 72, 72, linewidth=2.5,
                               edgecolor='#FF7043', facecolor='#FF704318', linestyle='-')
    ax.add_patch(inner)
    
    for corner in [(28, 28), (100, 28), (28, 100), (100, 100)]:
        ax.plot(*corner, 's', color='#FF7043', markersize=5, markeredgewidth=0)
    
    ax.annotate('', xy=(100, 22), xytext=(28, 22),
                arrowprops=dict(arrowstyle='<->', color='#FF7043', lw=1.5))
    ax.text(64, 14, 'VOI', ha='center', va='center', color='#FF7043',
            fontsize=11, fontweight='bold')
    
    ax.set_xlim(-5, 133)
    ax.set_ylim(-5, 133)
    ax.axis('off')
    fig.savefig(f'{OUT}/box2_subvolume.png', dpi=300,
                bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.close()
    print("  Box 2 saved")


# ═══════════════════════════════════════════════════════════════════════
# BOX 4: Quantum circuit (ZZ Feature Map)
# ═══════════════════════════════════════════════════════════════════════
def make_circuit_figure():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    n_qubits = 4
    qubit_y = [3 - i * 0.85 for i in range(n_qubits)]
    
    # Wires
    for i, y in enumerate(qubit_y):
        ax.plot([0.3, 6.8], [y, y], '-', color='#AAAAAA', lw=0.8)
        ax.text(0.08, y, f'q{i}', ha='right', va='center',
                fontsize=9, color='#888888', fontfamily='monospace')
    
    # Hadamard gates
    for y in qubit_y:
        box = patches.FancyBboxPatch((0.78, y - 0.2), 0.44, 0.4,
              boxstyle="round,pad=0.04", facecolor='#5C6BC0', edgecolor='#3949AB', lw=1.2)
        ax.add_patch(box)
        ax.text(1.0, y, 'H', ha='center', va='center', color='white',
                fontsize=11, fontweight='bold')
    
    # Rz encoding (layer 1)
    for i, y in enumerate(qubit_y):
        box = patches.FancyBboxPatch((1.72, y - 0.2), 0.56, 0.4,
              boxstyle="round,pad=0.04", facecolor='#26A69A', edgecolor='#00897B', lw=1.2)
        ax.add_patch(box)
        ax.text(2.0, y, f'Rz(x{i})', ha='center', va='center', color='white',
                fontsize=8, fontweight='bold')
    
    # ZZ entangling (CNOT pairs)
    zz_pairs = [(0, 1), (1, 2), (2, 3)]
    for j, (q1, q2) in enumerate(zz_pairs):
        x_pos = 3.1 + j * 0.65
        y1, y2 = qubit_y[q1], qubit_y[q2]
        ax.plot([x_pos, x_pos], [y1, y2], '-', color='#EF5350', lw=2.0)
        ax.plot(x_pos, y1, 'o', color='#EF5350', markersize=7, markeredgewidth=0)
        circle = plt.Circle((x_pos, y2), 0.13, facecolor='white',
                           edgecolor='#EF5350', lw=1.8)
        ax.add_patch(circle)
        ax.plot([x_pos - 0.08, x_pos + 0.08], [y2, y2], '-', color='#EF5350', lw=1.3)
        ax.plot([x_pos, x_pos], [y2 - 0.08, y2 + 0.08], '-', color='#EF5350', lw=1.3)
    
    # Rz encoding (layer 2)
    for i, y in enumerate(qubit_y):
        box = patches.FancyBboxPatch((5.22, y - 0.2), 0.56, 0.4,
              boxstyle="round,pad=0.04", facecolor='#26A69A', edgecolor='#00897B', lw=1.2)
        ax.add_patch(box)
        ax.text(5.5, y, f'Rz(x{i})', ha='center', va='center', color='white',
                fontsize=8, fontweight='bold')
    
    # Measurement
    for y in qubit_y:
        box = patches.FancyBboxPatch((6.28, y - 0.2), 0.4, 0.4,
              boxstyle="round,pad=0.04", facecolor='#78909C', edgecolor='#546E7A', lw=1.2)
        ax.add_patch(box)
        arc = patches.Arc((6.48, y - 0.02), 0.22, 0.22, angle=0,
                          theta1=180, theta2=360, color='white', lw=1.5)
        ax.add_patch(arc)
        ax.plot([6.48, 6.55], [y - 0.02, y + 0.13], '-', color='white', lw=1.3)
    
    # Stage labels
    ax.text(1.0, 3.45, 'Hadamard', ha='center', fontsize=7.5, color='#5C6BC0', fontstyle='italic')
    ax.text(2.0, 3.45, 'Encode', ha='center', fontsize=7.5, color='#26A69A', fontstyle='italic')
    ax.text(3.7, 3.45, 'ZZ entangle', ha='center', fontsize=7.5, color='#EF5350', fontstyle='italic')
    ax.text(5.5, 3.45, 'Encode', ha='center', fontsize=7.5, color='#26A69A', fontstyle='italic')
    
    ax.set_xlim(-0.2, 7.0)
    ax.set_ylim(-0.2, 3.7)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig(f'{OUT}/box4_quantum_circuit.png', dpi=300,
                bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.close()
    print("  Box 4 saved")


# ═══════════════════════════════════════════════════════════════════════
# BOX 5: Kernel matrix heatmap
# ═══════════════════════════════════════════════════════════════════════
def make_kernel_figure():
    rng = np.random.default_rng(42)
    n = 200
    half = n // 2
    
    K = np.zeros((n, n))
    
    # Within-class blocks
    block1 = rng.uniform(0.2, 0.9, (half, half))
    block1 = gaussian_filter(block1, sigma=4)
    block2 = rng.uniform(0.2, 0.9, (half, half))
    block2 = gaussian_filter(block2, sigma=4)
    
    K[:half, :half] = block1
    K[half:, half:] = block2
    
    # Between-class (near zero)
    K[:half, half:] = rng.uniform(0, 0.05, (half, half))
    K[half:, :half] = rng.uniform(0, 0.05, (half, half))
    
    K = (K + K.T) / 2
    np.fill_diagonal(K, 1.0)
    
    cmap = LinearSegmentedColormap.from_list('qk',
        ['#0D1B2A', '#1B3A5C', '#26A69A', '#FFD54F', '#FFFFFF'], N=256)
    
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(K, cmap=cmap, vmin=0, vmax=1, aspect='equal')
    
    ax.axhline(y=half - 0.5, color='#FF7043', lw=1.2, ls='--', alpha=0.8)
    ax.axvline(x=half - 0.5, color='#FF7043', lw=1.2, ls='--', alpha=0.8)
    
    ax.text(half // 2, -8, 'Sparse', ha='center', fontsize=9, color='#4FC3F7', fontweight='bold')
    ax.text(half + half // 2, -8, 'Dense', ha='center', fontsize=9, color='#FF7043', fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label('κ(xᵢ, xⱼ)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    ax.set_xlabel('Sample index', fontsize=10)
    ax.set_ylabel('Sample index', fontsize=10)
    ax.tick_params(labelsize=8)
    
    fig.savefig(f'{OUT}/box5_kernel_matrix.png', dpi=300,
                bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.close()
    print("  Box 5 saved")


# ═══════════════════════════════════════════════════════════════════════
# BOX 6: Displacement field reconstruction
# ═══════════════════════════════════════════════════════════════════════
def make_displacement_figure():
    bone_slice, bone_mask = make_bone_slice(128, seed=42, bvtv_target=0.30)
    rng = np.random.default_rng(123)
    y_coords, x_coords = np.mgrid[0:128, 0:128]
    
    uy = -0.08 * y_coords + 0.015 * np.sin(x_coords * 0.12) * y_coords / 128
    ux = 0.025 * np.sin(y_coords * 0.08) * (x_coords - 64) / 64
    
    noise_ux = gaussian_filter(rng.standard_normal((128, 128)), sigma=6) * 0.8
    noise_uy = gaussian_filter(rng.standard_normal((128, 128)), sigma=6) * 0.8
    ux += noise_ux * bone_mask
    uy += noise_uy * bone_mask
    
    disp_mag = np.sqrt(ux ** 2 + uy ** 2)
    disp_masked = np.where(bone_mask, disp_mag, np.nan)
    
    cmap_d = LinearSegmentedColormap.from_list('disp',
        ['#1A237E', '#1565C0', '#26A69A', '#FFD54F', '#FF7043', '#B71C1C'], N=256)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(bone_slice, cmap='bone', origin='lower', vmin=0, vmax=255, alpha=0.35)
    im = ax.imshow(disp_masked, cmap=cmap_d, origin='lower', alpha=0.85,
                   vmin=0, vmax=np.nanpercentile(disp_masked, 98))
    
    step = 8
    mask_sparse = bone_mask[::step, ::step]
    ax.quiver(x_coords[::step, ::step][mask_sparse],
              y_coords[::step, ::step][mask_sparse],
              ux[::step, ::step][mask_sparse],
              uy[::step, ::step][mask_sparse],
              color='white', alpha=0.6, scale=12, width=0.004,
              headwidth=3, headlength=3)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label('|u| (μm)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    ax.axis('off')
    fig.savefig(f'{OUT}/box6_displacement.png', dpi=300,
                bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.close()
    print("  Box 6 saved")


# ═══════════════════════════════════════════════════════════════════════
# BOX 7: Principal strain map
# ═══════════════════════════════════════════════════════════════════════
def make_strain_figure():
    bone_slice, bone_mask = make_bone_slice(128, seed=42, bvtv_target=0.30)
    rng = np.random.default_rng(456)
    y_coords, x_coords = np.mgrid[0:128, 0:128]
    
    # Base compressive gradient
    base_strain = -0.008 * (y_coords / 128)
    
    # Strain concentrations at thin trabeculae
    dist = distance_transform_edt(bone_mask)
    dist_norm = dist / (dist.max() + 1e-6)
    local = -0.025 * (1 - dist_norm) ** 2 * bone_mask
    
    # Heterogeneous component
    hetero = gaussian_filter(rng.standard_normal((128, 128)), sigma=5) * 0.008
    
    strain = (base_strain + local + hetero) * bone_mask
    strain_masked = np.where(bone_mask, strain, np.nan)
    
    cmap_s = LinearSegmentedColormap.from_list('strain',
        ['#0D47A1', '#1565C0', '#42A5F5', '#BBDEFB',
         '#FFFFFF',
         '#FFCDD2', '#EF5350', '#C62828', '#8B0000'], N=256)
    
    vmax = np.nanpercentile(np.abs(strain_masked), 98)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(bone_slice, cmap='bone', origin='lower', vmin=0, vmax=255, alpha=0.3)
    im = ax.imshow(strain_masked, cmap=cmap_s, origin='lower', alpha=0.85,
                   vmin=-vmax, vmax=vmax)
    
    # Mark hotspots
    strain_abs = np.abs(np.nan_to_num(strain_masked))
    threshold = np.percentile(strain_abs[strain_abs > 0], 96)
    hotspot_mask = strain_abs > threshold
    labelled, n_clusters = label(hotspot_mask)
    count = 0
    for cid in range(1, n_clusters + 1):
        cy, cx = np.where(labelled == cid)
        if len(cy) > 8:
            ax.add_patch(plt.Circle((cx.mean(), cy.mean()), 7,
                         facecolor='none', edgecolor='#FFD54F', lw=1.5, ls='--'))
            count += 1
            if count >= 5:
                break
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label('ε₃ (min. principal strain)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    ax.axis('off')
    fig.savefig(f'{OUT}/box7_strain.png', dpi=300,
                bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.close()
    print("  Box 7 saved")


if __name__ == '__main__':
    print("Generating pipeline visuals...\n")
    make_subvolume_figure()
    make_circuit_figure()
    make_kernel_figure()
    make_displacement_figure()
    make_strain_figure()
    print("\nDone — all 5 PNGs in paper/figures/pipeline/")
